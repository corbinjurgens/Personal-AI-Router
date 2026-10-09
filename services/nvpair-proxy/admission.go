// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Node-wide admission (FORK_DESIGN.md §3.3).
//
// One controller per process, shared by every facade, because the limits it
// enforces are the machine's: one GPU, one memory pool, one set of resident
// models across every engine. A per-facade controller would let Ollama and
// LM Studio each believe they had the machine to themselves, which is the
// same reasoning that puts the reservation map on the process (spec.md §3.1).
//
// It runs on the destination side: in the cluster ingress for work a peer
// sends here, and in handleHTTP before dispatching to this node's own engine.
// The broker feeds it policy, availability, drains, residency and saved
// engine intent over the node/* requests in nodepolicy/wire.go, and it reports
// back with admission/state, admission/unload and admission/wake.

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"nvpair-shared/nodepolicy"
)

// admKey identifies a model as one engine names it, after normalization.
type admKey struct {
	engine string
	model  string
}

// notification is a proxy → broker notification queued while the controller's
// lock is held and sent once it is released, so a backpressured broker pipe
// never stalls admission for every facade.
type notification struct {
	method string
	params any
}

// admissionController is the node-wide admission state and decision.
type admissionController struct {
	mu sync.Mutex
	// changed is closed and replaced on every state change, waking every
	// waiter to re-evaluate. A condition variable cannot be selected on
	// together with a deadline and a request context; a channel can.
	changed chan struct{}

	policy       nodepolicy.Policy
	availability nodepolicy.Availability
	drain        map[string]bool
	intent       map[string]bool

	// loaded is the broker-reported residency (node/set-residency), keyed by
	// engine then normalized model. residencyKnown stays false until the first
	// report: before it, only in-flight models count as resident, so a broker
	// that has not started reporting cannot strand every request in an
	// eviction wait for a model nobody can see.
	loaded         map[string]map[string]bool
	residencyKnown bool
	residencyAt    time.Time

	inflight       map[admKey]int
	active         int
	activeByEngine map[string]int
	queued         int
	lastActivity   map[string]int64
	// finishedAt is when a model last finished a request here. A model that
	// finished after the latest residency report is presumed still loaded:
	// the engine keeps it, and the broker has not had the chance to say so.
	// Without this, two back-to-back requests for different models on a
	// one-model node would both be admitted between residency polls.
	finishedAt map[admKey]time.Time

	unloadSentAt map[admKey]time.Time
	wakeSentAt   map[string]time.Time

	tickets    map[uint64]*admissionTicket
	nextTicket uint64

	// State reporting: at most one admission/state a second while it changes,
	// and immediately when the active count reaches zero. stateSeq orders
	// snapshots so a stale one computed before a newer one is never sent after
	// it; sendMu serializes the writes.
	stateSentAt time.Time
	stateTimer  *time.Timer
	stateSeq    uint64
	sendMu      sync.Mutex
	sentSeq     uint64

	notify    func(method string, params any)
	healthy   func(engine string) bool
	normalize func(engine, model string) string
}

func newAdmissionController(notify func(string, any), healthy func(string) bool, normalize func(string, string) string) *admissionController {
	return &admissionController{
		changed:        make(chan struct{}),
		policy:         nodepolicy.Default(),
		availability:   nodepolicy.Available,
		drain:          map[string]bool{},
		intent:         map[string]bool{},
		loaded:         map[string]map[string]bool{},
		inflight:       map[admKey]int{},
		activeByEngine: map[string]int{},
		lastActivity:   map[string]int64{},
		finishedAt:     map[admKey]time.Time{},
		unloadSentAt:   map[admKey]time.Time{},
		wakeSentAt:     map[string]time.Time{},
		tickets:        map[uint64]*admissionTicket{},
		notify:         notify,
		healthy:        healthy,
		normalize:      normalize,
	}
}

// signalLocked wakes every waiter. Caller holds mu.
func (a *admissionController) signalLocked() {
	close(a.changed)
	a.changed = make(chan struct{})
}

// --- broker inputs -------------------------------------------------------

func (a *admissionController) setPolicy(p nodepolicy.Policy) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.policy = p
	a.signalLocked()
}

// currentPolicy snapshots the policy. Profiles and tiers are slices the
// controller never mutates in place (setPolicy replaces the whole value), so a
// shallow copy is safe to read without the lock.
func (a *admissionController) currentPolicy() nodepolicy.Policy {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.policy
}

func (a *admissionController) currentAvailability() nodepolicy.Availability {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.availability
}

// currentActive is the number of admitted executions running now.
func (a *admissionController) currentActive() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.active
}

func (a *admissionController) engineIntent(engine string) (enabled, known bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	enabled, known = a.intent[engine]
	return enabled, known
}

// setAvailability applies node/set-availability. With cancelActive every
// admitted execution on this node is cancelled after the state is in force,
// so nothing admitted in between escapes the cancel.
func (a *admissionController) setAvailability(state nodepolicy.Availability, cancelActive bool) {
	a.mu.Lock()
	a.availability = state
	a.signalLocked()
	var cancels []func()
	if cancelActive {
		for _, t := range a.tickets {
			if t.cancel != nil {
				cancels = append(cancels, t.cancel)
			}
		}
	}
	a.mu.Unlock()
	for _, cancel := range cancels {
		cancel()
	}
}

func (a *admissionController) setDrain(engine string, drain bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if drain {
		a.drain[engine] = true
	} else {
		delete(a.drain, engine)
	}
	a.signalLocked()
}

func (a *admissionController) setIntent(enabled map[string]bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.intent = make(map[string]bool, len(enabled))
	for engine, on := range enabled {
		a.intent[engine] = on
	}
	a.signalLocked()
}

func (a *admissionController) setResidency(loadedByEngine map[string][]string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	now := time.Now()
	loaded := make(map[string]map[string]bool, len(loadedByEngine))
	for engine, models := range loadedByEngine {
		set := make(map[string]bool, len(models))
		for _, m := range models {
			if n := a.normalize(engine, m); n != "" {
				set[n] = true
			}
		}
		loaded[engine] = set
	}
	a.loaded = loaded
	a.residencyKnown = true
	a.residencyAt = now
	// A report supersedes every presumption older than it.
	for k, t := range a.finishedAt {
		if !t.After(now) {
			delete(a.finishedAt, k)
		}
	}
	for k := range a.unloadSentAt {
		if !loaded[k.engine][k.model] {
			delete(a.unloadSentAt, k)
		}
	}
	a.signalLocked()
}

// isLoaded reports whether the broker last reported (engine, model) resident.
func (a *admissionController) isLoaded(engine, model string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.loaded[engine][a.normalize(engine, model)]
}

// backendChanged is called when a facade's local backend changes, so a
// request waiting for a woken engine re-checks its health.
func (a *admissionController) backendChanged() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.signalLocked()
}

// --- decision ------------------------------------------------------------

// admissionRequest is one request asking to execute on this node.
type admissionRequest struct {
	engine string
	model  string
	// waitCap is the X-PAIR-Admission-Wait bound. Negative means the caller
	// set none, so the policy's own timeouts apply.
	waitCap time.Duration
	// cancel aborts the execution once admitted; node/set-availability with
	// cancelActive calls it.
	cancel func()
}

// admissionDecision is the outcome. Exactly one of ticket, reject and err is
// set: err means the request's context ended while it waited.
type admissionDecision struct {
	ticket     *admissionTicket
	profile    nodepolicy.Profile
	hasProfile bool
	reject     nodepolicy.RejectReason
	err        error
}

// admissionTicket is an admitted execution. release must be called exactly
// when the execution ends; it is idempotent so a deferred release can back up
// an explicit one.
type admissionTicket struct {
	a      *admissionController
	id     uint64
	key    admKey
	cancel func()
	once   sync.Once
}

// bounded returns the wait a policy timeout allows under the caller's cap.
func bounded(seconds int, waitCap time.Duration) time.Duration {
	d := time.Duration(seconds) * time.Second
	if waitCap >= 0 && waitCap < d {
		d = waitCap
	}
	return d
}

// admit decides whether a request may execute here, waiting where the policy
// allows. The decision order is FORK_DESIGN.md §3.3's.
func (a *admissionController) admit(ctx context.Context, req admissionRequest) admissionDecision {
	model := a.normalize(req.engine, req.model)
	key := admKey{engine: req.engine, model: model}
	start := time.Now()
	var (
		queueDeadline time.Time
		wakeDeadline  time.Time
		evictDeadline time.Time
		queuedHere    bool
	)

	a.mu.Lock()
	// leave undoes this request's queued count. Caller holds mu.
	leave := func() *stateSnapshot {
		if !queuedHere {
			return nil
		}
		queuedHere = false
		a.queued--
		return a.noteChangeLocked()
	}
	enqueue := func() *stateSnapshot {
		if queuedHere {
			return nil
		}
		queuedHere = true
		a.queued++
		return a.noteChangeLocked()
	}
	reject := func(reason nodepolicy.RejectReason) admissionDecision {
		st := leave()
		a.mu.Unlock()
		a.sendState(st)
		slog.Debug("admission rejected", "engine", req.engine, "model", model, "reason", reason)
		return admissionDecision{reject: reason}
	}

	for {
		pol := a.policy
		var sends []notification

		// 1. Availability, then a settings-restart drain of this engine.
		switch a.availability {
		case nodepolicy.Available:
		case nodepolicy.Paused:
			return reject(nodepolicy.RejectPaused)
		default:
			return reject(nodepolicy.RejectDraining)
		}
		if a.drain[req.engine] {
			return reject(nodepolicy.RejectDraining)
		}

		// 2. Is the engine running? Health is what node/set-local-backend
		// reports, which is also what a wake waits for.
		if !a.healthy(req.engine) {
			// An engine with no saved intent is unmanaged, or the broker has not
			// said yet; either way nothing would start it, so waiting for a wake
			// would only add the wake timeout to a rejection.
			if on, known := a.intent[req.engine]; !known || !on || !pol.Idle.StartOnDemand {
				return reject(nodepolicy.RejectOff)
			}
			if wakeDeadline.IsZero() {
				wakeDeadline = start.Add(bounded(pol.Idle.WakeTimeoutSeconds, req.waitCap))
			}
			if last, ok := a.wakeSentAt[req.engine]; !ok || time.Since(last) > wakeResendInterval {
				a.wakeSentAt[req.engine] = time.Now()
				sends = append(sends, notification{nodepolicy.NotifyAdmissionWake, nodepolicy.WakeRequest{Engine: req.engine}})
			}
			if !time.Now().Before(wakeDeadline) {
				a.mu.Unlock()
				a.send(sends)
				a.mu.Lock()
				return reject(nodepolicy.RejectWake)
			}
			st := enqueue()
			if err := a.waitLocked(ctx, wakeDeadline, sends, st); err != nil {
				return a.abandon(leave, err)
			}
			continue
		}

		// 3. Profile.
		profile, hasProfile := a.profileLocked(key)

		if model != "" {
			// 4. Per-model concurrency.
			limit := pol.Admission.MaxConcurrentPerModel
			if hasProfile && profile.MaxConcurrent > 0 {
				limit = profile.MaxConcurrent
			}
			if limit > 0 && a.inflight[key] >= limit {
				if queueDeadline.IsZero() {
					queueDeadline = start.Add(bounded(pol.Admission.QueueTimeoutSeconds, req.waitCap))
				}
				if !time.Now().Before(queueDeadline) {
					return reject(nodepolicy.RejectBusy)
				}
				st := enqueue()
				if err := a.waitLocked(ctx, queueDeadline, nil, st); err != nil {
					return a.abandon(leave, err)
				}
				continue
			}

			// 5 and 6. Resident models and the memory budget.
			plan, evict := a.fitLocked(key, pol)
			switch plan {
			case fitNever:
				return reject(nodepolicy.RejectNoFit)
			case fitEvict:
				if !pol.Admission.SwitchModels {
					plan = fitWait
					break
				}
				if evictDeadline.IsZero() {
					evictDeadline = time.Now().Add(time.Duration(pol.Admission.SwitchTimeoutSeconds) * time.Second)
				}
				sends = append(sends, a.unloadLocked(evict, pol)...)
				if !time.Now().Before(evictDeadline) {
					// The switch timeout passed: admit anyway and let the
					// engine's own memory handling take over.
					plan = fitOK
					break
				}
				st := enqueue()
				if err := a.waitLocked(ctx, evictDeadline, sends, st); err != nil {
					return a.abandon(leave, err)
				}
				continue
			}
			if plan == fitWait {
				if queueDeadline.IsZero() {
					queueDeadline = start.Add(bounded(pol.Admission.QueueTimeoutSeconds, req.waitCap))
				}
				if !time.Now().Before(queueDeadline) {
					return reject(nodepolicy.RejectNoFit)
				}
				st := enqueue()
				if err := a.waitLocked(ctx, queueDeadline, sends, st); err != nil {
					return a.abandon(leave, err)
				}
				continue
			}
		}

		// 8. Admit.
		if queuedHere {
			queuedHere = false
			a.queued--
		}
		a.nextTicket++
		t := &admissionTicket{a: a, id: a.nextTicket, key: key, cancel: req.cancel}
		a.tickets[t.id] = t
		if model != "" {
			a.inflight[key]++
		}
		a.active++
		a.activeByEngine[req.engine]++
		a.lastActivity[req.engine] = time.Now().UnixMilli()
		st := a.noteChangeLocked()
		a.mu.Unlock()
		a.send(sends)
		a.sendState(st)
		return admissionDecision{ticket: t, profile: profile, hasProfile: hasProfile}
	}
}

// wakeResendInterval spaces repeated admission/wake notifications for one
// engine, so a burst against a stopped engine asks the broker once.
const wakeResendInterval = 10 * time.Second

// abandon ends a wait whose request context ended. Caller holds mu.
func (a *admissionController) abandon(leave func() *stateSnapshot, err error) admissionDecision {
	st := leave()
	a.mu.Unlock()
	a.sendState(st)
	return admissionDecision{err: err}
}

// waitLocked releases mu, sends what the caller queued, and blocks until the
// state changes, the deadline passes, or ctx ends; it returns holding mu again.
func (a *admissionController) waitLocked(ctx context.Context, deadline time.Time, sends []notification, st *stateSnapshot) error {
	ch := a.changed
	a.mu.Unlock()
	a.send(sends)
	a.sendState(st)
	timer := time.NewTimer(time.Until(deadline))
	select {
	case <-ch:
	case <-timer.C:
	case <-ctx.Done():
	}
	timer.Stop()
	a.mu.Lock()
	return ctx.Err()
}

type fitPlan int

const (
	fitOK fitPlan = iota
	// fitEvict: unloading the returned idle models makes room.
	fitEvict
	// fitWait: even unloading every idle model would not make room, so only
	// in-flight work finishing can.
	fitWait
	// fitNever: the model alone exceeds the memory budget.
	fitNever
)

// residentLocked is the resident set admission counts: models reported loaded,
// models with work in flight, and models presumed still loaded because they
// finished after the latest residency report. Caller holds mu.
func (a *admissionController) residentLocked() map[admKey]bool {
	set := make(map[admKey]bool)
	for k, n := range a.inflight {
		if n > 0 {
			set[k] = true
		}
	}
	if a.residencyKnown {
		for engine, models := range a.loaded {
			for m := range models {
				set[admKey{engine: engine, model: m}] = true
			}
		}
		for k, t := range a.finishedAt {
			if t.After(a.residencyAt) {
				set[k] = true
			}
		}
	}
	return set
}

// fitLocked decides whether key fits beside the resident set under the
// resident-model cap and the memory budget, and which idle models to unload
// if it does not. Idle models are unloaded least recently used first.
func (a *admissionController) fitLocked(key admKey, pol nodepolicy.Policy) (fitPlan, []admKey) {
	resident := a.residentLocked()
	if resident[key] {
		return fitOK, nil
	}
	maxResident := pol.Admission.MaxResidentModels
	budget := pol.Admission.MemoryBudgetBytes
	need := a.memoryLocked(key)
	if budget > 0 && need > budget {
		return fitNever, nil
	}
	count := len(resident) + 1
	mem := need
	for k := range resident {
		mem += a.memoryLocked(k)
	}
	fits := func() bool {
		return (maxResident <= 0 || count <= maxResident) && (budget <= 0 || mem <= budget)
	}
	if fits() {
		return fitOK, nil
	}
	idle := make([]admKey, 0, len(resident))
	for k := range resident {
		if a.inflight[k] == 0 {
			idle = append(idle, k)
		}
	}
	sort.Slice(idle, func(i, j int) bool {
		ti, tj := a.finishedAt[idle[i]], a.finishedAt[idle[j]]
		if !ti.Equal(tj) {
			return ti.Before(tj)
		}
		if idle[i].engine != idle[j].engine {
			return idle[i].engine < idle[j].engine
		}
		return idle[i].model < idle[j].model
	})
	var evict []admKey
	for _, k := range idle {
		evict = append(evict, k)
		count--
		mem -= a.memoryLocked(k)
		if fits() {
			return fitEvict, evict
		}
	}
	return fitWait, nil
}

// memoryLocked is a model's profile memoryBytes; unknown counts as 0.
func (a *admissionController) memoryLocked(k admKey) int64 {
	if pr, ok := a.profileLocked(k); ok {
		return pr.MemoryBytes
	}
	return 0
}

// profileLocked looks a model's profile up by engine and normalized model, so
// a profile written "llama3" covers a request for "llama3:latest".
func (a *admissionController) profileLocked(k admKey) (nodepolicy.Profile, bool) {
	if k.model == "" {
		return nodepolicy.Profile{}, false
	}
	for _, pr := range a.policy.Profiles {
		if pr.Engine == k.engine && a.normalize(pr.Engine, pr.Model) == k.model {
			return pr, true
		}
	}
	return nodepolicy.Profile{}, false
}

// unloadLocked builds the admission/unload notifications for evict, skipping
// models already asked for within the switch timeout so concurrent waiters
// for one switch ask once.
func (a *admissionController) unloadLocked(evict []admKey, pol nodepolicy.Policy) []notification {
	resend := time.Duration(pol.Admission.SwitchTimeoutSeconds) * time.Second
	if resend < 5*time.Second {
		resend = 5 * time.Second
	}
	byEngine := map[string][]string{}
	now := time.Now()
	for _, k := range evict {
		if t, ok := a.unloadSentAt[k]; ok && now.Sub(t) < resend {
			continue
		}
		a.unloadSentAt[k] = now
		byEngine[k.engine] = append(byEngine[k.engine], k.model)
	}
	engines := make([]string, 0, len(byEngine))
	for e := range byEngine {
		engines = append(engines, e)
	}
	sort.Strings(engines)
	out := make([]notification, 0, len(engines))
	for _, e := range engines {
		out = append(out, notification{nodepolicy.NotifyAdmissionUnload, nodepolicy.UnloadRequest{Engine: e, Models: byEngine[e]}})
	}
	return out
}

// release ends an admitted execution.
func (t *admissionTicket) release() {
	if t == nil {
		return
	}
	t.once.Do(func() {
		a := t.a
		a.mu.Lock()
		delete(a.tickets, t.id)
		now := time.Now()
		if t.key.model != "" {
			if a.inflight[t.key] <= 1 {
				delete(a.inflight, t.key)
			} else {
				a.inflight[t.key]--
			}
			a.finishedAt[t.key] = now
		}
		a.active--
		if a.activeByEngine[t.key.engine] <= 1 {
			delete(a.activeByEngine, t.key.engine)
		} else {
			a.activeByEngine[t.key.engine]--
		}
		a.lastActivity[t.key.engine] = now.UnixMilli()
		a.signalLocked()
		st := a.noteChangeLocked()
		a.mu.Unlock()
		a.sendState(st)
	})
}

// --- reporting -----------------------------------------------------------

func (a *admissionController) send(sends []notification) {
	for _, n := range sends {
		a.notify(n.method, n.params)
	}
}

// stateSnapshot is an admission/state payload with its order.
type stateSnapshot struct {
	seq   uint64
	state nodepolicy.AdmissionState
}

// noteChangeLocked records a state change and returns a snapshot to send now,
// or nil when the one-a-second limit defers it to a timer. Caller holds mu.
func (a *admissionController) noteChangeLocked() *stateSnapshot {
	now := time.Now()
	if a.active == 0 || now.Sub(a.stateSentAt) >= time.Second {
		if a.stateTimer != nil {
			a.stateTimer.Stop()
			a.stateTimer = nil
		}
		a.stateSentAt = now
		st := a.snapshotLocked()
		return &st
	}
	if a.stateTimer == nil {
		a.stateTimer = time.AfterFunc(time.Second-now.Sub(a.stateSentAt), a.flushState)
	}
	return nil
}

func (a *admissionController) flushState() {
	a.mu.Lock()
	a.stateTimer = nil
	a.stateSentAt = time.Now()
	st := a.snapshotLocked()
	a.mu.Unlock()
	a.sendState(&st)
}

// snapshotLocked copies the reportable state and stamps its order. Caller
// holds mu.
func (a *admissionController) snapshotLocked() stateSnapshot {
	a.stateSeq++
	byEngine := make(map[string]int, len(a.activeByEngine))
	for e, n := range a.activeByEngine {
		byEngine[e] = n
	}
	last := make(map[string]int64, len(a.lastActivity))
	for e, ms := range a.lastActivity {
		last[e] = ms
	}
	return stateSnapshot{seq: a.stateSeq, state: nodepolicy.AdmissionState{
		Active:         a.active,
		ActiveByEngine: byEngine,
		Queued:         a.queued,
		LastActivityMs: last,
	}}
}

// sendState writes a snapshot unless a newer one has already gone out, so two
// releases racing to report cannot leave the broker holding the older count.
func (a *admissionController) sendState(st *stateSnapshot) {
	if st == nil {
		return
	}
	a.sendMu.Lock()
	defer a.sendMu.Unlock()
	if st.seq <= a.sentSeq {
		return
	}
	a.sentSeq = st.seq
	a.notify(nodepolicy.NotifyAdmissionState, st.state)
}

// --- HTTP surface --------------------------------------------------------

// parseAdmissionWait reads X-PAIR-Admission-Wait: the longest, in whole
// seconds, the router lets this node queue the request. Absent or malformed
// means no cap beyond the policy's own, which is what a router that predates
// the header sends.
func parseAdmissionWait(h http.Header) time.Duration {
	raw := strings.TrimSpace(h.Get(nodepolicy.AdmissionWaitHeader))
	if raw == "" {
		return -1
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 {
		return -1
	}
	return time.Duration(n) * time.Second
}

// retryAfterSeconds is the Retry-After hint for a rejection: roughly how long
// the condition behind it usually lasts. Advisory, like the proxy's own 503.
func retryAfterSeconds(reason nodepolicy.RejectReason) int {
	switch reason {
	case nodepolicy.RejectPaused, nodepolicy.RejectOff:
		return 30
	case nodepolicy.RejectDraining, nodepolicy.RejectNoFit, nodepolicy.RejectWake:
		return 5
	default:
		return 1
	}
}

// writeAdmissionRejection answers a request admission refused: 503 with the
// reason in X-PAIR-Admission, so a routing node fails over to its next
// candidate exactly as it would for any other 503.
func writeAdmissionRejection(w http.ResponseWriter, reason nodepolicy.RejectReason) {
	w.Header().Set(nodepolicy.AdmissionHeader, string(reason))
	w.Header().Set("Retry-After", strconv.Itoa(retryAfterSeconds(reason)))
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(http.StatusServiceUnavailable)
	body, err := json.Marshal(map[string]string{
		"error":  fmt.Sprintf("this node is not accepting the request (%s)", reason),
		"code":   "admission",
		"reason": string(reason),
	})
	if err != nil {
		body = []byte(`{"error":"admission rejected","code":"admission"}`)
	}
	_, _ = w.Write(body)
}
