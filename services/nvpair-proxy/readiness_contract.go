// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"reflect"
	"slices"
	"strings"
	"sync"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

type strictAdmissionError struct{ Reason string }

func (e *strictAdmissionError) Error() string { return "strict admission: " + e.Reason }
func (e *strictAdmissionError) Is(target error) bool {
	t, ok := target.(*strictAdmissionError)
	return ok && t.Reason == e.Reason
}

var (
	errStrictUnavailable = &strictAdmissionError{Reason: "unavailable"}
	errStrictOccupied    = &strictAdmissionError{Reason: "occupied"}
	errStrictCold        = &strictAdmissionError{Reason: "cold-residency"}
)

type strictTimer interface {
	Chan() <-chan time.Time
	Stop()
}
type realStrictTimer struct{ t *time.Timer }

func (t realStrictTimer) Chan() <-chan time.Time { return t.t.C }
func (t realStrictTimer) Stop()                  { t.t.Stop() }

var timeNow = time.Now

func newStrictTimer(d time.Duration) strictTimer { return realStrictTimer{time.NewTimer(d)} }

type nodeStrictModel = nodepolicy.StrictAdmissionModel
type strictAdmissionState struct {
	// models is the installed capability and qualified what each model was
	// qualified under. Both are derived together by refreshStrictCapabilityLocked
	// from the policy, the engine identity and the qualification registry, and
	// never set otherwise. A key's generation changes whenever its entry or
	// current serving envelope does.
	models     map[admKey]nodeStrictModel
	qualified  map[admKey]installedQualification
	generation map[admKey]uint64
	// identity is the broker's node/set-model-identity, by engine; registry
	// is the node's qualification evidence, read at start.
	identity   map[string]nodepolicy.EngineIdentity
	registry   qualificationRegistry
	markers    map[uint64]capacityMarker
	path       string
	storageErr error
	now        func() time.Time
	timer      func(time.Duration) strictTimer
	write      func(string, []byte, os.FileMode) error
	read       func(string) ([]byte, error)
}

// refreshStrictCapabilityLocked derives the strict capability from what it
// rests on: the active policy's qualified profiles, the qualification registry
// and the broker's engine identity. A profile's model is installed only when
// its qualification exists for the same engine and normalized model, the
// engine's identity is present, its launch and the model's content digest are
// ones the qualification covers, the profile's defaults are the qualified ones,
// its context and output bounds are within the qualified ones, and its
// effective concurrency is finite. Its routing capabilities must be covered
// by the qualification's shapes. Two profiles normalizing to one model
// install neither. Keys whose installed entry changed get a new generation;
// setModelIdentity also fences envelope-only changes. It reports
// whether the capability changed. Caller holds a.mu.
func (a *admissionController) refreshStrictCapabilityLocked() bool {
	models := make(map[admKey]nodeStrictModel)
	qualified := make(map[admKey]installedQualification)
	profiles := make(map[admKey]int)
	for _, pr := range a.policy.Profiles {
		profiles[admKey{engine: pr.Engine, model: a.normalize(pr.Engine, pr.Model)}]++
	}
	for _, pr := range a.policy.Profiles {
		k := admKey{engine: pr.Engine, model: a.normalize(pr.Engine, pr.Model)}
		// Any second profile for the model, qualified or not, makes its
		// bounds ambiguous.
		if pr.Qualification == nil || profiles[k] != 1 {
			continue
		}
		if m, q, ok := a.qualifiedModelLocked(k, pr); ok {
			models[k], qualified[k] = m, installedQualification{q: q, profile: pr}
		}
	}
	changed := false
	for k := range a.strict.models {
		if _, kept := models[k]; !kept {
			a.strict.generation[k]++
			changed = true
		}
	}
	for k, m := range models {
		old, had := a.strict.models[k]
		if !had || old != m || !sameInstalledQualification(a.strict.qualified[k], qualified[k]) {
			a.strict.generation[k]++
			changed = true
		}
	}
	a.strict.models, a.strict.qualified = models, qualified
	if changed {
		a.signalLocked()
	}
	return changed
}

// qualifiedModelLocked is the strict entry for profile pr at key k, when every
// condition of refreshStrictCapabilityLocked holds. Caller holds a.mu.
func (a *admissionController) qualifiedModelLocked(k admKey, pr nodepolicy.Profile) (nodeStrictModel, *localQualification, bool) {
	q, ok := a.strict.registry.lookup(*pr.Qualification)
	if !ok || q.Engine != k.engine || a.normalize(q.Engine, q.Model) != k.model || !a.validStrictModelKey(k) {
		return nodeStrictModel{}, nil, false
	}
	id, ok := a.strict.identity[k.engine]
	if !ok || !slices.Contains(q.LaunchDigests, id.LaunchDigest) {
		return nodeStrictModel{}, nil, false
	}
	digest, ok := a.modelDigestLocked(id, k)
	if !ok || !slices.Contains(q.ModelDigests, digest) {
		return nodeStrictModel{}, nil, false
	}
	if pr.RequestOptionsMode.Effective() != q.RequestOptionsMode.Effective() || !nodepolicy.EqualRequestOptions(pr.RequestOptions, q.RequestOptions) || pr.ContextTokens > q.ContextTokens || pr.MaxOutputTokens > q.MaxOutputTokens {
		return nodeStrictModel{}, nil, false
	}
	m := nodeStrictModel{
		Engine:        k.engine,
		Model:         k.model,
		MaxConcurrent: a.concurrencyLimitLocked(k),
		Identity:      nodepolicy.StrictModelIdentity{EngineRunID: id.EngineRunID, SettingsFingerprint: id.SettingsFingerprint, ModelDigest: digest},
	}
	if m.Validate() != nil {
		return nodeStrictModel{}, nil, false
	}
	if id.Serving != nil {
		envelope, ok := a.servingEnvelopeLocked(id, k, digest)
		if !ok || int64(q.ContextTokens) > envelope.PerSlotContextTokens || int64(q.MaxOutputTokens) > envelope.MaxOutputTokens ||
			int64(m.MaxConcurrent) > envelope.Slots || pr.MemoryBytes < *envelope.Memory.VRAMBytes {
			return nodeStrictModel{}, nil, false
		}
	}
	for _, capability := range pr.Capabilities {
		switch capability {
		case nodepolicy.CapTools:
			if !*q.Shapes.Tools {
				return nodeStrictModel{}, nil, false
			}
		case nodepolicy.CapStructured:
			if !slices.Contains(q.Shapes.StructuredOutput, "json_object") || !slices.Contains(q.Shapes.StructuredOutput, "json_schema") {
				return nodeStrictModel{}, nil, false
			}
		default:
			return nodeStrictModel{}, nil, false
		}
	}
	return m, q, true
}

// servingEnvelopeLocked resolves the same normalized model as content identity.
// A present serving map must supply a valid, unambiguous content-bound envelope.
// Canonicalize only the model name so equivalent aliases can agree.
func (a *admissionController) servingEnvelopeLocked(id nodepolicy.EngineIdentity, k admKey, digest string) (engines.ServingEnvelope, bool) {
	var found engines.ServingEnvelope
	present := false
	for name, envelope := range id.Serving {
		if a.normalize(k.engine, name) != k.model {
			continue
		}
		if envelope.Validate(name, digest) != nil {
			return engines.ServingEnvelope{}, false
		}
		envelope.ModelID = k.model
		if present && !reflect.DeepEqual(found, envelope) {
			return engines.ServingEnvelope{}, false
		}
		found, present = envelope, true
	}
	return found, present
}

// modelDigestLocked finds the content digest the identity lists for k's model
// under any name the engine normalizes to it. Names that disagree give none.
func (a *admissionController) modelDigestLocked(id nodepolicy.EngineIdentity, k admKey) (string, bool) {
	found := ""
	for name, digest := range id.ModelDigests {
		if a.normalize(k.engine, name) != k.model {
			continue
		}
		if found != "" && found != digest {
			return "", false
		}
		found = digest
	}
	return found, found != ""
}

func sameInstalledQualification(x, y installedQualification) bool {
	return x.q == y.q && x.profile.ContextTokens == y.profile.ContextTokens &&
		x.profile.RequestOptionsMode.Effective() == y.profile.RequestOptionsMode.Effective() &&
		x.profile.MaxOutputTokens == y.profile.MaxOutputTokens && nodepolicy.EqualRequestOptions(x.profile.RequestOptions, y.profile.RequestOptions)
}

// setModelIdentity replaces the engine identity (nodepolicy.MethodSetModelIdentity)
// and reinstalls the capability. It reports whether the capability changed.
func (a *admissionController) setModelIdentity(engines map[string]nodepolicy.EngineIdentity) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	changed := false
	for k := range a.strict.models {
		old, next := a.strict.identity[k.engine], engines[k.engine]
		oldDigest, _ := a.modelDigestLocked(old, k)
		nextDigest, _ := a.modelDigestLocked(next, k)
		oldEnvelope, oldOK := a.servingEnvelopeLocked(old, k, oldDigest)
		nextEnvelope, nextOK := a.servingEnvelopeLocked(next, k, nextDigest)
		// The outward entry omits the envelope. Fence validations and held
		// tickets even when a replacement still satisfies every strict bound.
		if (old.Serving == nil) != (next.Serving == nil) || oldOK && nextOK && !reflect.DeepEqual(oldEnvelope, nextEnvelope) {
			a.strict.generation[k]++
			changed = true
		}
	}
	a.strict.identity = engines
	changed = a.refreshStrictCapabilityLocked() || changed
	// Launch-only changes can leave strict model entries unchanged, but a
	// resident final grant binds the complete engine identity.
	a.signalLocked()
	return changed
}

// strictValidation is a request body validated against the qualification its
// model is installed under, and the generation of that installation.
type strictValidation struct {
	body       []byte
	bound      strictBound
	generation uint64
}

// qualifyStrict validates a strict request before anything else happens for
// it: the model must be installed under the identity the request names, and
// the body, with the profile's defaults merged in, must fit the qualification.
// AdmitStrict then admits only while that installation is unchanged.
func (a *admissionController) qualifyStrict(engine, model, route string, body []byte, identity nodepolicy.StrictModelIdentity, contexts ...context.Context) (strictValidation, error) {
	a.mu.Lock()
	k := admKey{engine: engine, model: a.normalize(engine, model)}
	m, ok := a.strict.models[k]
	installed := a.strict.qualified[k]
	generation := a.strict.generation[k]
	a.mu.Unlock()
	if !ok || m.Identity != identity {
		return strictValidation{}, errStrictUnavailable
	}
	effective, bound, err := installed.q.validateRequest(installed.profile, route, body, contexts...)
	if err != nil {
		return strictValidation{}, err
	}
	return strictValidation{body: effective, bound: bound, generation: generation}, nil
}

// StrictReadiness is what the controller can admit under the strict contract
// for the active policy revision: the installed capability, its models under
// the normalized keys they were installed with and sorted by engine and model,
// filtered by the active served policy so a model this node may not serve is
// not advertised, or none when nothing is installed, nothing is servable, or
// strict storage is untrusted (AdmitStrict fails closed then). The filter is
// outward only: installed, qualified and generation state stay in place so
// already-admitted tickets keep their evidence. Policy activation and identity
// changes have already rederived the capability. ProxyRunID is the process's;
// reportStrictReadiness stamps it.
func (a *admissionController) StrictReadiness() nodepolicy.StrictAdmissionReadiness {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := nodepolicy.StrictAdmissionReadiness{PolicyRevision: a.policy.Revision}
	if a.strict.storageErr != nil || len(a.strict.models) == 0 {
		return out
	}
	models := make([]nodepolicy.StrictAdmissionModel, 0, len(a.strict.models))
	for _, m := range a.strict.models {
		// Publication follows the active served policy: a model this node may
		// not serve is not part of the capability it advertises, while the
		// internal installation, qualification and generation state stays
		// intact so already-admitted tickets keep their evidence.
		if !modelServed(a.policy.ServedModels, m.Engine, m.Model, a.normalize) {
			continue
		}
		models = append(models, m)
	}
	if len(models) == 0 {
		return out
	}
	slices.SortFunc(models, func(x, y nodepolicy.StrictAdmissionModel) int {
		if c := strings.Compare(x.Engine, y.Engine); c != 0 {
			return c
		}
		return strings.Compare(x.Model, y.Model)
	})
	out.Capability = &nodepolicy.StrictAdmissionCapability{Version: nodepolicy.StrictAdmissionVersion, Models: models}
	return out
}

// reportStrictReadiness sends the broker this process's strict readiness for
// the active revision (nodepolicy.NotifyStrictReadiness), stamped with the run
// id its PolicyAck carries. Before any policy is active there is no revision to
// name and nothing is sent. Reports go out from the control read loop only,
// which keeps a report computed from older state from following a newer one.
func (p *Proxy) reportStrictReadiness() {
	report := p.admission.StrictReadiness()
	if report.PolicyRevision == "" {
		return
	}
	report.ProxyRunID = p.runID
	if err := p.codec.Notify(nodepolicy.NotifyStrictReadiness, report); err != nil {
		slog.Debug("failed to send strict admission readiness", "err", err)
	}
}

// AdmitStrict shares the concrete path's lock, in-flight counts, residency,
// notifications and tickets. Engine IDs are canonical built-in IDs; models use
// the existing engine adapter normalization (Ollama adds :latest, LM Studio exact).
// generation is the installation qualifyStrict validated the request against;
// a model reinstalled since, by a policy, identity or qualification change,
// is unavailable to it.
func (a *admissionController) AdmitStrict(ctx context.Context, engine, model string, req nodepolicy.StrictAdmissionRequest, generation uint64) (strictAdmissionTicket, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	a.mu.Lock()
	defer func() { st := a.noteChangeLocked(); a.mu.Unlock(); a.sendState(st) }()
	// Tickets, quarantine markers and completion history outlive the request's
	// decoded storage. Share one owned bounded identifier across those borrowers.
	k := admKey{engine: engine, model: strings.Clone(a.normalize(engine, model))}
	start := a.strict.now()
	pre := start.Add(time.Duration(req.PreOutputBudgetMs) * time.Millisecond)
	queue := start.Add(time.Duration(req.QueueBudgetMs) * time.Millisecond)
	cold := start.Add(time.Duration(req.ColdStartBudgetMs) * time.Millisecond)
	queued := false
	var waitedUntil time.Time
	var waitedReason error
	coldNeeded := false
	defer func() {
		if queued {
			a.queued--
			a.signalLocked()
		}
	}()
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if !modelServed(a.policy.ServedModels, engine, model, a.normalize) {
			return nil, errStrictUnavailable
		}
		m, ok := a.strict.models[k]
		if !ok || a.strict.generation[k] != generation || m.MaxConcurrent < 1 || m.Identity != req.Identity || a.strict.storageErr != nil || a.availability != nodepolicy.Available || a.drain[engine] {
			return nil, errStrictUnavailable
		}
		if on, known := a.intent[engine]; known && !on {
			return nil, errStrictUnavailable
		}
		now := a.strict.now()
		if coldNeeded && !now.Before(cold) {
			return nil, errStrictCold
		}
		if !waitedUntil.IsZero() && !now.Before(waitedUntil) {
			return nil, waitedReason
		}
		if !now.Before(pre) {
			return nil, errStrictOccupied
		}
		var sends []notification
		deadline := pre
		reason := errStrictOccupied
		if a.inflight[k] >= m.MaxConcurrent {
			if queue.Before(deadline) {
				deadline = queue
			}
		} else {
			plan, evict := a.fitLocked(k, a.policy)
			// In-flight and quarantined capacity are not proof that the model
			// finished loading. Only the destination residency report is warm.
			_, unloading := a.unloadSentAt[k]
			resident := a.loaded[engine][k.model] && !unloading
			healthy := a.healthy(engine)
			if plan == fitNever {
				return nil, errStrictUnavailable
			}
			if healthy && resident && plan == fitOK {
				dispatchDeadline := pre
				if coldNeeded && cold.Before(dispatchDeadline) {
					dispatchDeadline = cold
				}
				now = a.strict.now()
				if !now.Before(dispatchDeadline) {
					if coldNeeded {
						return nil, errStrictCold
					}
					return nil, errStrictOccupied
				}
				if a.nextTicket == ^uint64(0) {
					return nil, errStrictUnavailable
				}
				a.nextTicket++
				base := &admissionTicket{a: a, id: a.nextTicket, key: k}
				a.tickets[base.id] = base
				a.inflight[k]++
				a.active++
				a.activeByEngine[engine]++
				a.lastActivity[engine] = now.UnixMilli()
				return &strictTicket{base: base, generation: a.strict.generation[k], identity: req.Identity, deadline: dispatchDeadline, ctx: ctx, warmOnly: true, servedModels: a.policy.ServedModels}, nil
			}
			if plan == fitWait {
				if queue.Before(deadline) {
					deadline = queue
				}
			} else {
				coldNeeded = true
				reason = errStrictCold
				if cold.Before(deadline) {
					deadline = cold
				}
				if !now.Before(deadline) {
					return nil, reason
				}
				if !healthy {
					if on, known := a.intent[engine]; !known || !on || !a.policy.Idle.StartOnDemand {
						return nil, errStrictUnavailable
					}
					if last, ok := a.wakeSentAt[engine]; !ok || now.Sub(last) >= wakeResendInterval {
						a.wakeSentAt[engine] = now
						sends = append(sends, notification{nodepolicy.NotifyAdmissionWake, nodepolicy.WakeRequest{Engine: engine}})
					}
				} else if plan == fitEvict && a.policy.Admission.SwitchModels {
					sends = a.unloadLocked(evict, a.policy)
				} else if plan == fitEvict {
					return nil, errStrictCold
				}
				// Cold unloaded models need no extra wake RPC: the inference
				// call loads them, but only with a nonzero cold allowance.
				if healthy && plan == fitOK && !unloading {
					now = a.strict.now()
					if !now.Before(deadline) {
						return nil, errStrictCold
					}
					if a.nextTicket == ^uint64(0) {
						return nil, errStrictUnavailable
					}
					a.nextTicket++
					base := &admissionTicket{a: a, id: a.nextTicket, key: k}
					a.tickets[base.id] = base
					a.inflight[k]++
					a.active++
					a.activeByEngine[engine]++
					a.lastActivity[engine] = now.UnixMilli()
					return &strictTicket{base: base, generation: a.strict.generation[k], identity: req.Identity, deadline: deadline, ctx: ctx, servedModels: a.policy.ServedModels}, nil
				}
			}
		}
		if coldNeeded && !cold.After(deadline) {
			deadline, reason = cold, errStrictCold
		}
		if !now.Before(deadline) {
			return nil, reason
		}
		if !queued {
			a.queued++
			queued = true
		}
		ch := a.changed
		waitedUntil, waitedReason = deadline, reason
		timer := a.strict.timer(deadline.Sub(now))
		st := a.noteChangeLocked()
		a.mu.Unlock()
		a.send(sends)
		a.sendState(st)
		select {
		case <-ch:
		case <-timer.Chan():
		case <-ctx.Done():
		}
		timer.Stop()
		a.mu.Lock()
	}
}

type strictTicket struct {
	// Permission belongs to the admission snapshot: restrictive activation
	// does not revoke existing work. Policy slices are replaced, never mutated.
	servedModels nodepolicy.ServedModels
	base         *admissionTicket
	identity     nodepolicy.StrictModelIdentity
	generation   uint64
	deadline     time.Time
	ctx          context.Context
	dispatched   bool
	finished     bool
	warmOnly     bool
}

func (t *strictTicket) BeforeStrictDispatch() error {
	a := t.base.a
	a.mu.Lock()
	defer func() { st := a.noteChangeLocked(); a.mu.Unlock(); a.sendState(st) }()
	if t.finished {
		return errors.New("strict ticket already finished")
	}
	if t.dispatched {
		return nil
	}
	if !modelServed(t.servedModels, t.base.key.engine, t.base.key.model, a.normalize) {
		return errStrictUnavailable
	}
	mc, ok := a.strict.models[t.base.key]
	if !ok || a.strict.generation[t.base.key] != t.generation {
		t.finishLocked(strictNotDispatched)
		return errStrictUnavailable
	}
	if !ok || mc.Identity != t.identity || a.availability != nodepolicy.Available || a.drain[t.base.key.engine] || !a.healthy(t.base.key.engine) {
		return errStrictUnavailable
	}
	if on, known := a.intent[t.base.key.engine]; known && !on {
		return errStrictUnavailable
	}
	if err := t.ctx.Err(); err != nil {
		return err
	}
	if !a.strict.now().Before(t.deadline) {
		return errStrictCold
	}
	if _, unloading := a.unloadSentAt[t.base.key]; t.warmOnly && (!a.loaded[t.base.key.engine][t.base.key.model] || unloading) {
		// Release as not-dispatched under the same lock, durably clearing
		// any marker from a previously failed dispatch attempt.
		t.finishLocked(strictNotDispatched)
		return errStrictCold
	}
	m := capacityMarker{ID: t.base.id, Engine: t.base.key.engine, Model: t.base.key.model}
	if _, exists := a.strict.markers[m.ID]; !exists && len(a.strict.markers) >= maxStrictCapacitySlots {
		t.finishLocked(strictNotDispatched)
		return errStrictUnavailable
	}
	a.strict.markers[m.ID] = m
	if err := a.persistStrictCapacityLocked(); err != nil {
		return fmt.Errorf("persist strict dispatch: %w", err)
	}
	if err := t.ctx.Err(); err != nil {
		return err
	}
	if !a.strict.now().Before(t.deadline) {
		return errStrictCold
	}
	t.dispatched = true
	return nil
}

// Bound connection, headers and first body with the same remaining monotonic
// deadline used by admission. A successful first byte ends this timer; it does
// not impose a duration limit on an active healthy generation.
func (t *strictTicket) watchPreOutput(cancel context.CancelFunc) func() {
	a := t.base.a
	a.mu.Lock()
	timer := a.strict.timer(t.deadline.Sub(a.strict.now()))
	a.mu.Unlock()
	done := make(chan struct{})
	var once sync.Once
	stop := func() { once.Do(func() { close(done); timer.Stop() }) }
	go func() {
		select {
		case <-timer.Chan():
			once.Do(func() { cancel(); close(done) })
		case <-t.ctx.Done():
			stop()
		case <-done:
		}
	}()
	return stop
}

func (a *admissionController) releaseStrictSlotLocked(k admKey) {
	a.inflight[k]--
	if a.inflight[k] == 0 {
		delete(a.inflight, k)
	}
	a.active--
	a.activeByEngine[k.engine]--
	if a.activeByEngine[k.engine] == 0 {
		delete(a.activeByEngine, k.engine)
	}
	if a.residencyKnown {
		a.finishedAt[k] = a.strict.now()
	}
	a.lastActivity[k.engine] = a.strict.now().UnixMilli()
}

func (t *strictTicket) Finish(e strictTerminalEvidence) {
	a := t.base.a
	a.mu.Lock()
	defer func() { st := a.noteChangeLocked(); a.mu.Unlock(); a.sendState(st) }()
	t.finishLocked(e)
}

func (t *strictTicket) finishLocked(e strictTerminalEvidence) {
	a := t.base.a
	if t.finished {
		return
	}
	t.finished = true
	delete(a.tickets, t.base.id)
	_, marked := a.strict.markers[t.base.id]
	if e == strictUncertain || (t.dispatched && e == strictNotDispatched) {
		a.strict.markers[t.base.id] = capacityMarker{ID: t.base.id, Engine: t.base.key.engine, Model: t.base.key.model}
		if err := a.persistStrictCapacityLocked(); err != nil {
			slog.Error("strict capacity remains quarantined after persistence failure", "err", err)
		}
		a.signalLocked()
		return
	}
	if e != strictNotDispatched && e != strictQualifiedComplete && e != strictAuthoritativeIdle && e != strictVerifiedReplacement {
		// Unknown evidence cannot release a possibly dispatched slot.
		a.strict.markers[t.base.id] = capacityMarker{ID: t.base.id, Engine: t.base.key.engine, Model: t.base.key.model}
		if err := a.persistStrictCapacityLocked(); err != nil {
			slog.Error("strict capacity quarantine persistence failed", "err", err)
		}
		return
	}
	if marked {
		m := a.strict.markers[t.base.id]
		delete(a.strict.markers, t.base.id)
		if err := a.persistStrictCapacityLocked(); err != nil {
			a.strict.markers[t.base.id] = m
			slog.Error("strict capacity clear failed; retaining quarantine", "err", err)
			return
		}
	}
	a.releaseStrictSlotLocked(t.base.key)
	a.signalLocked()
}

func parseStrictAdmission(h http.Header) (*nodepolicy.StrictAdmissionRequest, error) {
	var values []string
	present := false
	for k, v := range h {
		if strings.EqualFold(k, nodepolicy.StrictAdmissionHeader) {
			present = true
			values = append(values, v...)
		}
	}
	if !present {
		return nil, nil
	}
	if len(values) != 1 {
		return nil, errors.New("strict admission requires exactly one header value")
	}
	var req nodepolicy.StrictAdmissionRequest
	if err := json.Unmarshal([]byte(values[0]), &req); err != nil {
		return nil, fmt.Errorf("invalid strict admission header: %w", err)
	}
	return &req, nil
}

func stripInternalRoutingHeaders(h http.Header) {
	for k := range h {
		if strings.HasPrefix(strings.ToLower(k), "x-pair-") || strings.EqualFold(k, engineIdentityProbeHeader) {
			delete(h, k)
		}
	}
}

func writeStrictAdmissionError(w http.ResponseWriter, err error) {
	status := http.StatusServiceUnavailable
	if errors.Is(err, errStrictOccupied) {
		status = http.StatusTooManyRequests
	}
	if errors.Is(err, context.DeadlineExceeded) {
		status = http.StatusGatewayTimeout
	}
	writeIngressError(w, status, "strict-admission", err.Error())
}

// strictDestinationAdmission is A1's process-wide admission seam, shared by all
// facades and paired ingress. ingress.go:serveStrictLocal, reached from
// serveIngress and handleHTTP's self dispatch, calls qualifyStrict and then
// AdmitStrict after trusted-contract parsing, before wake/unload or inference.
// The capability is derived from local qualification and the broker's engine
// identity (B). Identity, availability, finite capacity and the qualified
// installation are all revalidated atomically. Deadlines are monotonic, derived
// from remaining durations, and may only shrink. Concrete requests keep their
// existing path.
type strictDestinationAdmission interface {
	qualifyStrict(engine, model, route string, body []byte, identity nodepolicy.StrictModelIdentity, contexts ...context.Context) (strictValidation, error)
	AdmitStrict(ctx context.Context, engine, model string, req nodepolicy.StrictAdmissionRequest, generation uint64) (strictAdmissionTicket, error)
	StrictReadiness() nodepolicy.StrictAdmissionReadiness
}

// strictAdmissionTicket distinguishes cancellation from proved termination.
// BeforeStrictDispatch must durably mark the engine dirty before any inference
// bytes are sent; failure prevents dispatch. Finish is idempotent. Uncertain
// work retains capacity across cancellation/restart until authoritative engine
// reconciliation; elapsed time alone never releases it.
type strictAdmissionTicket interface {
	BeforeStrictDispatch() error
	Finish(strictTerminalEvidence)
}

type strictTerminalEvidence string

const (
	strictNotDispatched       strictTerminalEvidence = "not-dispatched"
	strictQualifiedComplete   strictTerminalEvidence = "qualified-complete"
	strictAuthoritativeIdle   strictTerminalEvidence = "authoritative-idle"
	strictVerifiedReplacement strictTerminalEvidence = "verified-replacement"
	strictUncertain           strictTerminalEvidence = "uncertain"
)
