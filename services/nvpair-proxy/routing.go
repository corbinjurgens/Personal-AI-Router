// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"strconv"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/nodepolicy"
)

// Compact the existing I states without changing their transitions or text.
// Zero still denotes the original empty/uninitialized state.
type routingState uint8

func (s routingState) String() string {
	switch s {
	case routingPrepared:
		return "prepared"
	case routingReserving:
		return "reserving-admitting"
	case routingPreSend:
		return "pre-send"
	case routingAwaitBody:
		return "accepted-await-body"
	case routingCommitted:
		return "committed"
	case routingEnding:
		return "ending"
	case routingTerminal:
		return "terminal"
	}
	return ""
}

// Entry handlers attach the immutable request deadline for shared destination
// preparation. It is metadata from the owner, not an active-stream deadline.
type routingDeadlineContextKey struct{}

func routingPreparationContext(ctx context.Context) (context.Context, context.CancelFunc) {
	deadline, _ := ctx.Value(routingDeadlineContextKey{}).(time.Time)
	return context.WithDeadline(ctx, deadline)
}

func inferencePreparationRefusal(err error) error {
	if stderrors.Is(err, context.DeadlineExceeded) {
		return &bufferBusyError{cause: err}
	}
	return err
}

const (
	routingPrepared routingState = iota + 1
	routingReserving
	routingPreSend
	routingAwaitBody
	routingCommitted
	routingEnding
	routingTerminal
)

// routingSpend is the sole request owner, including free local/peer dispatch.
// Its existing I2 mutex arbitrates every attempt and irreversible commitment.
// Helpers ending in Locked require that mutex. No callback, close, join or IO
// runs while it is held; permission lock order remains G -> admission -> request.
func newRoutingRequest(cancel func(string)) *routingSpend {
	return &routingSpend{onCancel: cancel, excluded: map[string]bool{}, state: routingPrepared}
}

func (r *routingSpend) transitionLocked(state routingState, reason string) {
	r.state, r.reason = state, reason
}

func (r *routingSpend) newDispatchAttempt(ctx context.Context, cancel context.CancelFunc, nodeID string, descriptor ...*routingDestinationDescriptor) (*publicAttemptLifecycle, error) {
	if r == nil || ctx == nil || ctx.Err() != nil {
		return nil, errRoutingSpend
	}
	started := time.Now()
	r.mu.Lock()
	defer r.mu.Unlock()
	a, err := r.newAttemptLocked(started, descriptor...)
	if err != nil {
		return nil, err
	}
	a.ctx, a.cancel, a.nodeID = ctx, cancel, nodeID
	a.state = routingReserving
	r.transitionLocked(a.state, "destination admission")
	return a, nil
}

func (a *publicAttemptLifecycle) acceptResponse() bool {
	if a == nil || a.request == nil {
		return false
	}
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != a || r.cancelled || r.done || a.aborted || a.terminal != nil || (a.ctx != nil && a.ctx.Err() != nil) {
		return false
	}
	a.accepted = true
	a.state = routingAwaitBody
	r.transitionLocked(a.state, "accepted destination response")
	return true
}

func (a *publicAttemptLifecycle) commit() bool {
	if a == nil || a.request == nil {
		return false
	}
	r := a.request
	r.mu.Lock()
	if r.active != a || r.cancelled || r.done || a.aborted || a.terminal != nil || a.state == routingEnding || !a.accepted || (a.ctx != nil && a.ctx.Err() != nil) {
		r.mu.Unlock()
		return false
	}
	if !a.committed && !r.deadline.IsZero() && !time.Now().Before(r.deadline) {
		cancel, _, _ := r.cancelLockedCause(routingDeadlineReason, factDeadline)
		r.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return false
	}
	a.committed, r.committed = true, true
	a.nativeFirstByteLocked()
	a.state = routingCommitted
	r.transitionLocked(a.state, "first accepted raw response byte")
	stop := r.stopPreOutput
	r.mu.Unlock()
	if stop != nil {
		stop()
	}
	return true
}

// expireFirstBody arbitrates this timer against raw commitment at the same
// owner. An ended precommit attempt cannot accept a late byte after Close.
func (a *publicAttemptLifecycle) expireFirstBody() bool {
	r := a.request
	r.mu.Lock()
	committed := r.committed
	var cancel context.CancelFunc
	if r.active == a && !r.done && !r.cancelled && !a.aborted && a.terminal == nil {
		if !committed && !r.deadline.IsZero() && !time.Now().Before(r.deadline) {
			cancel, _, _ = r.cancelLockedCause(routingDeadlineReason, factDeadline)
		} else if !committed {
			a.state = routingEnding
			r.transitionLocked(a.state, errFirstBodyTimeout.Error())
			a.observeAttemptLocked(routingAttemptResult{factAttemptFailed, factNoContent, 0, true})
			// Body Close interrupts this failed upstream read. Cancelling its
			// context would misclassify the failure as requester cancellation.
		}
	}
	r.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	return committed
}

// Called with sink metadata locked, never with the request mutex held. This
// transition admits planning before cancellation or denies it after cancellation.
func (a *publicAttemptLifecycle) admitInterruption(reason string) bool {
	if a == nil || a.request == nil {
		return false
	}
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != a || r.done || r.cancelled || !r.committed || a.aborted || a.terminal != nil || (a.ctx != nil && a.ctx.Err() != nil) {
		return false
	}
	a.state = routingEnding
	r.transitionLocked(a.state, reason)
	a.observeAttemptLocked(routingAttemptResult{factInterrupted, factStreamInterrupted, 0, true})
	return true
}

const routingDeadlineReason = "request deadline exceeded before first output"

// The same request watch spans concrete or owned public attempts. It is armed
// once from the live request context, never from a temporary preparation child.
func (r *routingSpend) watchPreOutput(ctx context.Context) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.stopPreOutput != nil || r.deadline.IsZero() || r.done || r.cancelled || r.committed {
		return
	}
	waitCtx, cancel := context.WithDeadline(ctx, r.deadline)
	stop := context.AfterFunc(waitCtx, func() {
		if stderrors.Is(waitCtx.Err(), context.DeadlineExceeded) {
			r.expireBeforeOutput()
		}
	})
	r.stopPreOutput = func() bool { stopped := stop(); cancel(); return stopped }
}

// Expiry and first successful raw bytes contend under the same request mutex.
// Keep the client parent live for the typed failure response: deadline-bound
// preparation already wakes, and only the still-pending transport is cancelled.
func (r *routingSpend) expireBeforeOutput() bool {
	r.mu.Lock()
	if r.committed {
		r.mu.Unlock()
		return false
	}
	cancel, _, changed := r.cancelLockedCause(routingDeadlineReason, factDeadline)
	r.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	return changed
}

func (r *routingSpend) deadlineExpired() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.cancelled && r.reason == routingDeadlineReason
}

func (a *publicAttemptLifecycle) abandon() bool {
	if a == nil || a.request == nil {
		return false
	}
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != a || r.done || r.cancelled || r.committed || a.committed || a.aborted || a.terminal != nil {
		return false
	}
	a.aborted = true
	a.observeAttemptLocked(routingAttemptResult{factAttemptFailed, factTargetLeft, 0, true})
	a.state = routingEnding
	r.transitionLocked(a.state, "destination left discovery")
	return true
}

func (a *publicAttemptLifecycle) settled() bool {
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.active != a || a.committed || a.aborted || a.terminal != nil || r.done || r.cancelled
}

func (a *publicAttemptLifecycle) canDeliverResponse() bool {
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.active == a && !r.done && !r.cancelled && !a.aborted && a.terminal == nil
}

func (a *publicAttemptLifecycle) requestCommitted() bool {
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.committed
}

// Raw successful bytes arbitrate commitment before SC or E can parse, allocate
// or reject a fragment. Synthetic control bytes never pass through this body.
type routingResponseBody struct {
	io.ReadCloser
	attempt *publicAttemptLifecycle
}

func (b *routingResponseBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if n > 0 && !b.attempt.commit() {
		return 0, context.Canceled
	}
	return n, err
}

func (a *publicAttemptLifecycle) abandoned() bool {
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	return a.aborted
}

func (a *publicAttemptLifecycle) redispatched() bool {
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	return a.redispatch && a.aborted
}

func (r *routingSpend) exclude(nodeID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.excluded == nil {
		r.excluded = map[string]bool{}
	}
	r.excluded[nodeID] = true
}

func (r *routingSpend) isExcluded(nodeID string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.excluded[nodeID]
}

func (r *routingSpend) hasExclusions() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.excluded) != 0
}

func (r *routingSpend) requestCancelled() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.cancelled
}

func (r *routingSpend) finish() {
	r.mu.Lock()
	r.foldCancellationLocked()
	r.done = true
	r.body = nil
	if r.active != nil {
		r.active.state = routingTerminal
		r.active = nil
	}
	r.transitionLocked(routingTerminal, "request finished")
	stop := r.stopPreOutput
	if r.committed {
		// The committing reader already owns the stop call. Its scheduling
		// cannot block terminal sink delivery or request cleanup.
		stop = nil
	}
	r.mu.Unlock()
	if stop != nil {
		stop()
	}
}

func (r *routingSpend) cancelWithReason(reason string) bool {
	return r.cancelWithCause(reason, factUnknown)
}

func (r *routingSpend) cancelWithCause(reason string, cause routingFactReason) bool {
	r.mu.Lock()
	cancel, callback, changed := r.cancelLockedCause(reason, cause)
	r.mu.Unlock()
	if !changed {
		return false
	}
	if cancel != nil {
		cancel()
	}
	if callback != nil {
		callback(reason)
	}
	return true
}

// cancelLocked requires mu and records the cancellation before any callback.
func (r *routingSpend) cancelLocked(reason string) (context.CancelFunc, func(string), bool) {
	return r.cancelLockedCause(reason, factUnknown)
}

func (r *routingSpend) cancelLockedCause(reason string, cause routingFactReason) (context.CancelFunc, func(string), bool) {
	if r.done || r.cancelled {
		return nil, nil, false
	}
	r.cancelled = true
	r.nativeCancelLocked(cause)
	r.transitionLocked(routingEnding, reason)
	var cancel context.CancelFunc
	if r.active != nil {
		cancel = r.active.cancel
	}
	return cancel, r.onCancel, true
}

func (r *routingSpend) abortLocal() { r.abortLocalIf(nil) }

func (r *routingSpend) abortLocalIf(expected *publicAttemptLifecycle) {
	r.mu.Lock()
	if r.done || r.cancelled || (expected != nil && (r.active != expected || expected.terminal != nil)) {
		r.mu.Unlock()
		return
	}
	a := r.active
	if a != nil && !r.committed && !a.aborted && a.terminal == nil {
		a.aborted, a.redispatch = true, true
		a.observeAttemptLocked(routingAttemptResult{factRedispatched, factNodePaused, 0, true})
		a.state = routingEnding
		r.transitionLocked(a.state, "local availability withdrawn")
		if r.excluded == nil {
			r.excluded = map[string]bool{}
		}
		r.excluded[a.nodeID] = true
		cancel := a.cancel
		r.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return
	}
	r.mu.Unlock()
	r.cancelWithCause("cancelled: this node stopped accepting work", factNodePaused)
}

// dispatch is the existing concrete inference retry loop, owned by this request.
// dispatchPolicy bounds one request on the common driver. A concrete route
// keeps the fixed failover budget and job deadline; a hybrid alias spends its
// tier's Limits once across local, peer and public candidates.
type dispatchPolicy struct {
	maxAttempts         int
	deadline            time.Time
	queueWait, coldWait time.Duration
	hybrid              bool
	apiOnly             bool
	publicAllowed       bool
	routingTiers        map[string]nodepolicy.TierRouting
}

func concreteDispatchPolicy(start time.Time) dispatchPolicy {
	return dispatchPolicy{maxAttempts: maxDispatchAttempts, deadline: start.Add(jobDeadline)}
}

// serveHybridInference enters the common HTTP request owner after listener
// gates. An enabled compatible facade lends its dialect and ID sequence, never
// its socket. OpenAI routes resolve across the enabled engine overlays.
// Without a compatible facade the request-local adapter has no overlay or
// listener: it can only refuse, as public dispatch is unavailable. It never
// emits a workload or competes with a real facade's workload ID sequence.
func (p *Proxy) serveHybridInference(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var f *facade
	for _, enabled := range p.enabledFacades() {
		if isInferenceRequest(enabled.profile, r.Method, r.URL.Path) && (f == nil || enabled.profile.Name < f.profile.Name) {
			f = enabled
		}
	}
	noFacade := f == nil
	if noFacade {
		for _, profile := range profiles {
			if isInferenceRequest(profile, r.Method, r.URL.Path) {
				f = newFacade(p, profile, NewDiscovery(), 0)
				break
			}
		}
	}
	if f == nil {
		writeIngressError(w, http.StatusNotFound, "not_found", "no such inference route")
		return
	}
	pol := p.admission.currentPolicy()
	policy := hybridDispatchPolicy(start, pol.Routing.Limits, nodepolicy.TierRouting{})
	// Published policy maps are immutable: retain this snapshot's alias rules
	// through body preparation without copying them before buffer admission.
	// Live admission and per-attempt authority checks remain independent.
	policy.routingTiers = pol.Routing.Tiers
	// Do not acquire a newly enabled overlay during body preparation on this
	// unregistered adapter. Its numeric notification ID has no facade sequence.
	policy.apiOnly = noFacade
	reqID := strconv.FormatUint(f.nextRequestID.Add(1), 10)
	f.serveInference(w, r, start, reqID, defaultStreamFrameIdle, policy)
}

// hybridDispatchPolicy reads one request's Limits. Public selection also
// requires build availability; each candidate still passes the request-bound
// publicAttemptPermitted guard before dispatch.
func hybridDispatchPolicy(start time.Time, limits nodepolicy.Limits, rules nodepolicy.TierRouting) dispatchPolicy {
	seconds := func(n int) time.Duration { return time.Duration(n) * time.Second }
	return dispatchPolicy{
		maxAttempts:   limits.MaxAttempts,
		deadline:      start.Add(seconds(limits.PreOutputBudgetSeconds)),
		queueWait:     seconds(limits.QueueSeconds),
		coldWait:      seconds(limits.ColdStartSeconds),
		hybrid:        true,
		publicAllowed: rules.RemoteAllowed && nodepolicy.PublicDispatchAvailable(),
	}
}

// peerSlotContext uses the remaining hybrid queue allowance for the origin's
// peer-service slot, leaving at least half the pre-output time for the
// destination's strict admission and first response. Concrete and local
// attempts retain their context.
func peerSlotContext(ctx context.Context, policy dispatchPolicy, cand candidate, queueLeft, preOutputLeft time.Duration) (context.Context, time.Duration) {
	if !policy.hybrid || cand.self || cand.peerUUID == "" {
		return ctx, 0
	}
	wait := min(queueLeft, max(preOutputLeft/2, 0))
	return clustertrust.WithPeerSlotWait(ctx, wait), wait
}

// publicChatSource supplies what a permitted public attempt needs: the
// deployment's qualification, transport, credential resolver and parser
// budget. No production source exists until deployment qualification lands,
// so a permitted candidate without one is skipped like a refused one.
type publicChatSource func(chatAccountSnapshot) (q *chatQualification, transport *publicClient,
	resolve func(nodepolicy.CredentialRef) (string, bool), budget chatByteBudget, ok bool)

var publicChatInputs publicChatSource

// hybridCandidateKey names one hybrid local candidate and the strict entry its
// node advertised for it.
func hybridCandidateKey(c candidate) string { return c.id + "\x00" + c.engine + "\x00" + c.model }

// writeStrictSelfRefusal answers a refused hybrid self candidate the way
// serveStrictLocal answers a refused strict request, and returns the status.
func writeStrictSelfRefusal(w http.ResponseWriter, err error) int {
	err = inferencePreparationRefusal(err)
	var invalid *strictQualificationError
	var capacity interface{ HTTPStatus() int }
	switch {
	case stderrors.Is(err, errAmbiguousModel):
		writeIngressError(w, http.StatusBadRequest, "ambiguous-model", ambiguousModelMessage)
		return http.StatusBadRequest
	case stderrors.As(err, &invalid):
		writeIngressError(w, http.StatusBadRequest, "strict-qualification", err.Error())
		return http.StatusBadRequest
	case stderrors.As(err, &capacity):
		writeIngressError(w, capacity.HTTPStatus(), "inference-buffer", "inference request could not be prepared")
		return capacity.HTTPStatus()
	}
	status := http.StatusServiceUnavailable
	switch {
	case stderrors.Is(err, errStrictOccupied):
		status = http.StatusTooManyRequests
	case stderrors.Is(err, context.DeadlineExceeded):
		status = http.StatusGatewayTimeout
	}
	writeStrictAdmissionError(w, err)
	return status
}

func (ctl *routingSpend) dispatch(f *facade, w http.ResponseWriter, r *http.Request, start time.Time, reqID string, bodyBytes []byte, model string, streamIdle time.Duration, cancelReasonOr func(string) string, requestOwner *routingRequest, policy dispatchPolicy) {
	p := f.host
	var err error
	ctl.mu.Lock()
	ctl.body = bodyBytes
	// The policy's pre-output deadline is the request's one deadline.
	ctl.deadline = policy.deadline
	ctl.mu.Unlock()
	deadline := policy.deadline
	preparationCtx, stopPreparation := context.WithDeadline(r.Context(), deadline)
	defer stopPreparation()
	ctl.watchPreOutput(r.Context())
	// The attempt owns its typed local admission. Panic cleanup follows the
	// transport's body-close defer, before releasing request buffer borrowers.
	var currentAttempt *publicAttemptLifecycle
	// A hybrid self attempt holds a strict ticket instead. Its evidence turns
	// uncertain once inference bytes may be sent, and Finish then keeps the
	// capacity quarantined until reconciliation, as serveStrictLocal does.
	var strictHeld strictAdmissionTicket
	strictEvidence := strictNotDispatched
	stopStrictWatch := func() {}
	finishStrict := func() {
		stopStrictWatch()
		stopStrictWatch = func() {}
		if strictHeld != nil {
			strictHeld.Finish(strictEvidence)
			strictHeld, strictEvidence = nil, strictNotDispatched
		}
	}
	defer func() {
		if currentAttempt != nil {
			currentAttempt.ticket.release()
		}
		finishStrict()
	}()

	routingModel := model
	// A tier request resolves to concrete (node, engine, model) candidates
	// from the policy instead of to owners of the literal model name.
	tier := ""
	var needs requestNeeds
	crossEngine := false
	if policy.hybrid {
		tier, _, _ = nodepolicy.ParseAlias(model)
	} else if nodepolicy.IsTier(model) {
		tier = model
	}
	if tier != "" {
		needs, err = parseRequestNeedsOwned(preparationCtx, bodyBytes, r.URL.Path)
		if err != nil {
			ctl.observeIngress(routingIngressResult{factFailed, failureFactReason(err), bufferRequestStatus(err)})
			writeIngressError(w, bufferRequestStatus(err), "inference-buffer", "inference request could not be prepared")
			return
		}
		crossEngine = openAICompatibleInference[r.URL.Path]
	}
	// strictFor holds the strict entry each hybrid local candidate advertised
	// when it was last resolved. The destination revalidates it.
	strictFor := map[string]nodepolicy.StrictAdmissionModel{}
	resolve := func() []candidate {
		var cands []candidate
		if policy.hybrid {
			// The tier order filtered to qualified declared roles; never a
			// second selector. A hybrid request without a tier has none.
			if tier != "" && !policy.apiOnly {
				for _, h := range f.hybridLocalCandidates(tier, needs, crossEngine) {
					strictFor[hybridCandidateKey(h.candidate)] = h.strict
					cands = append(cands, h.candidate)
				}
			}
		} else if tier != "" {
			cands = f.resolveTierCandidates(tier, needs, crossEngine)
		} else {
			cands = f.resolveCandidates(routingModel)
		}
		if !ctl.hasExclusions() {
			return cands
		}
		kept := make([]candidate, 0, len(cands))
		for _, c := range cands {
			if !ctl.isExcluded(c.id) {
				kept = append(kept, c)
			}
		}
		return kept
	}
	candidates := resolve()
	// held is this request's claim on a node's capacity while it is in flight.
	// resMu guards it because failover re-points it from the ReverseProxy's
	// callbacks while the disconnect watcher may be releasing it.
	var (
		resMu sync.Mutex
		held  reservation
	)
	sendBody := bodyBytes
	preparedCtx := r.Context()
	// Only admitted metadata crosses from a temporary preparation context to
	// the live attempt. Cancelling/expiring preparation must not end its stream.
	rebasePrepared := func(ctx context.Context) {
		preparedCtx = rebasePerformancePreparation(ctx, preparedCtx)
	}
	var preparedCandidate candidate
	var preparedDefaults []byte
	var preparedGeneration uint64
	var streamRequest originStreamRequest
	preparedValid := false
	releaseSendBody, releasePreparation := func() {}, func() {}
	dropPrepared := func() {
		releasePreparation()
		releaseSendBody()
		releaseSendBody, releasePreparation = func() {}, func() {}
		preparedValid = false
	}
	defer dropPrepared()
	var beforeRebuild func()
	prepareCandidate := func(ctx context.Context, c candidate, profile nodepolicy.Profile, hasProfile bool, generation uint64) error {
		var defaults []byte
		if hasProfile {
			defaults = profile.RequestOptions
		}
		if preparedValid && preparedCandidate == c && generation == preparedGeneration && bytes.Equal(defaults, preparedDefaults) {
			rebasePrepared(ctx)
			return nil
		}
		if beforeRebuild != nil {
			beforeRebuild()
		}
		// Retained grants let only the same accepted stream drain. They do not
		// grant retry permission or budget: a new candidate or generation must
		// drop/reprepare and pass admission for its new dispatch.
		dropPrepared()
		var err error
		sendBody, releaseSendBody, err = transformBody(ctx, bodyBytes, c.model, c.model != "", defaults)
		if err != nil {
			releaseSendBody = func() {}
			return err
		}
		concrete := model
		if c.model != "" {
			concrete = c.model
		}
		if hasProfile && !bodyModelMatches(sendBody, concrete) {
			return errAmbiguousModel
		}
		preparedCtx = ctx
		streamRequest, err = parseOriginStreamRequest(r.URL.Path, sendBody)
		if err != nil {
			return err
		}
		mode := performancePreparePeer
		if c.self {
			local := c.fac
			if local == nil {
				local = f
			}
			mode = local.performancePreparationMode(r.URL.Path)
		}
		preparedCtx, releasePreparation, err = preparePerformanceBuffers(ctx, sendBody, streamRequest, mode)
		if err != nil {
			releasePreparation = func() {}
			return err
		}
		preparedCandidate, preparedDefaults, preparedGeneration, preparedValid = c, defaults, generation, true
		return nil
	}
	// prepareStrictCandidate qualifies a hybrid self candidate's body as
	// serveStrictLocal does and prepares exactly the validated body it sends.
	// A strict body serves one attempt and is never reused.
	prepareStrictCandidate := func(ctx context.Context, c candidate, local *facade, engine, concrete string, identity nodepolicy.StrictModelIdentity, generation uint64) (uint64, error) {
		if beforeRebuild != nil {
			beforeRebuild()
		}
		dropPrepared()
		raw, releaseRaw, err := transformBody(ctx, bodyBytes, c.model, c.model != "", nil)
		if err != nil {
			return 0, err
		}
		validated, err := p.admission.qualifyStrict(engine, concrete, r.URL.Path, raw, identity, ctx)
		if err != nil {
			releaseRaw()
			return 0, err
		}
		sendBody, releaseSendBody = validated.body, releaseRaw
		if !bodyModelMatches(sendBody, concrete) {
			return 0, errAmbiguousModel
		}
		streamRequest, err = parseOriginStreamRequest(r.URL.Path, sendBody)
		if err != nil {
			return 0, err
		}
		preparedCtx, releasePreparation, err = preparePerformanceBuffers(ctx, sendBody, streamRequest, local.performancePreparationMode(r.URL.Path))
		if err != nil {
			releasePreparation = func() {}
			return 0, err
		}
		preparedCandidate, preparedDefaults, preparedGeneration, preparedValid = c, nil, generation, false
		return validated.generation, nil
	}
	if model != "" {
		candidates, held, err = f.reserveRoundPrepared(preparationCtx, candidates, tier != "", model, prepareCandidate, resolve)
		if err != nil {
			err = inferencePreparationRefusal(err)
			ctl.observeIngress(routingIngressResult{factFailed, failureFactReason(err), bufferRequestStatus(err)})
			if r.Context().Err() == nil {
				writeInferencePreparationError(w, err)
			}
			return
		}
	}
	roundReservations := held.held
	roundGeneration := preparedGeneration
	// Released exactly once, whichever way the request ends: normal unwind,
	// client disconnect, or an early return below. Without this a node stays
	// counted as loaded until the next scheduler snapshot, so a steady trickle
	// of short requests makes it look busier than it is and drives dispatch
	// away from it.
	//
	// Both accessors release resMu by defer. The release itself runs from a
	// defer, so a resMu stranded by a panic in the move would deadlock this
	// request against its own cleanup.
	moveHeld := func(nodeID string) {
		resMu.Lock()
		defer resMu.Unlock()
		held = p.moveReservation(held, nodeID)
	}
	defer func() {
		resMu.Lock()
		defer resMu.Unlock()
		p.releaseReservation(held)
		held = reservation{}
	}()
	// Public candidates follow every local one, so their tier is not empty
	// while public selection remains. They are offered inside the loop.
	publicPending := policy.hybrid && policy.publicAllowed && tier != "" && p.accounts != nil
	if len(candidates) == 0 && tier != "" && !publicPending {
		ctl.observeIngress(routingIngressResult{factFailed, factNoOwner, http.StatusNotFound})
		// Nothing anywhere can answer this tier for this request, which waiting
		// will not change.
		slog.Warn("proxy request rejected",
			"id", reqID, "method", r.Method, "path", r.URL.Path,
			"remote", r.RemoteAddr, "reason", "no eligible model for tier", "tier", tier)
		writeProxyError(w, http.StatusNotFound, []string{"no eligible model for tier ", tier}, needs.names(), true)
		_ = f.notify("proxy/request", RequestEvent{
			ID:       reqID,
			Method:   r.Method,
			Path:     r.URL.Path,
			Status:   http.StatusNotFound,
			Duration: time.Since(start).Milliseconds(),
			Error:    "no eligible model for tier",
		})
		return
	}
	if len(candidates) == 0 && !publicPending {
		ctl.observeIngress(routingIngressResult{factFailed, factNoOwner, http.StatusBadGateway})
		rejectionBody := `{"error":"no active node selected or available"}`
		rejectionError := "no active node"
		if model != "" {
			rejectionBody = `{"error":"no available node advertises the requested model"}`
			rejectionError = "no node advertises requested model"
		}
		slog.Warn("proxy request rejected",
			"id", reqID, "method", r.Method, "path", r.URL.Path,
			"remote", r.RemoteAddr, "reason", rejectionError)
		http.Error(w, rejectionBody, http.StatusBadGateway)
		_ = f.notify("proxy/request", RequestEvent{
			ID:       reqID,
			Method:   r.Method,
			Path:     r.URL.Path,
			Status:   http.StatusBadGateway,
			Duration: time.Since(start).Milliseconds(),
			Error:    rejectionError,
		})
		return
	}
	// shouldRetry reports whether an upstream status warrants failing over to
	// the next candidate: busy/unavailable/gateway statuses, plus a 404 on an
	// inference call (an advertised owner's inventory may have become stale).
	// Genuine client errors (400/401/422…) are not retried — they'd fail
	// identically on every node.
	shouldRetry := func(code int) bool {
		return code == http.StatusNotFound || retryableProxyStatus(code)
	}

	var (
		servedNodeID string
		servedTarget string
		ttfbMs       int64
		proxyErr     string
		finalStatus  int
		started      bool
		wl           *Workload
	)

	// wlSeq numbers this workload's events. Every mutation of wl below happens
	// either before the watcher goroutine exists or under wlMu, so a plain
	// counter is enough; see Workload.Seq for why the sequence is needed.
	var wlSeq int64
	nextWlSeq := func() int64 { wlSeq++; return wlSeq }

	// Emit workload:submitted the moment the request is admitted, before any
	// dispatch. A burst of concurrent inference requests must surface as job
	// cards immediately — the upstream engine serializes work on a single GPU
	// slot, so a card that waited for the upstream response would leave every
	// queued job invisible until the node dequeued it, one at a time (a prior
	// regression). "queued" is what makes that visible without claiming the
	// engine is generating, which only becomes true at the commit point below.
	//
	// scheduledOn is left empty here: nothing has been dispatched yet, and the
	// scheduler counts pending work by scheduledOn, so naming a node we have
	// not tried would inflate its load. Each dispatch and each gap between
	// attempts re-points it.
	//
	// Model-less inference still has a routing request key. Lifecycle payloads
	// require a nonempty concrete model in both the broker projection and the
	// workload-manager validator, so only model-bearing requests emit workloads.
	if model != "" {
		createdMs := start.UnixMilli()
		// A tier request's model is the concrete model it is headed for: the
		// leading candidate's until a dispatch says otherwise.
		wlModel := model
		if tier != "" && len(candidates) > 0 && candidates[0].model != "" {
			wlModel = candidates[0].model
		}
		wl = &Workload{
			ID:             reqID,
			Model:          wlModel,
			RequestedModel: tier,
			Engine:         f.profile.Name,
			RunID:          p.runID,
			State:          "queued",
			CreatedAt:      createdMs,
			Seq:            nextWlSeq(),
		}
		key := workloadKey{engine: f.profile.Name, id: reqID}
		if err := p.registerWorkload(requestOwner, key); err != nil {
			ctl.observeIngress(routingIngressResult{factFailed, factUnknown, http.StatusServiceUnavailable})
			writeIngressError(w, http.StatusServiceUnavailable, "request-unavailable", "routing request workload is unavailable")
			return
		}
		p.emitWorkload(workloadSubmittedMethod, *wl)
	}

	// The terminal workload transition (completed/errored) can be reached from
	// two places: the normal path after the stream copy unwinds below, and the
	// disconnect watcher that fires while the copy is still blocked. terminalOnce
	// guarantees exactly one is emitted; wlMu guards the shared wl fields the
	// watcher (a separate goroutine) and ModifyResponse's failover re-point both
	// touch; terminated suppresses a late started re-point once we've finalized.
	var (
		terminalOnce sync.Once
		wlMu         sync.Mutex
		terminated   bool
	)
	emitTerminal := func(state, errMsg string) {
		if wl == nil {
			return
		}
		terminalOnce.Do(func() {
			p.unregisterWorkload(requestOwner)
			now := time.Now().UnixMilli()
			wlMu.Lock()
			terminated = true
			wl.CompletedAt = &now
			wl.State = state
			if errMsg != "" {
				wl.Error = &errMsg
			}
			wl.Seq = nextWlSeq()
			snapshot := *wl
			wlMu.Unlock()
			// Only "completed" gets its own method; every other terminal state,
			// including "cancelled", rides workload:errored. The workload
			// manager does not validate method against state and consumers read
			// the state out of the payload, so a new terminal state needs no new
			// method on the wire.
			method := workloadCompletedMethod
			if state != "completed" {
				method = workloadErroredMethod
			}
			p.emitWorkload(method, snapshot)
		})
	}

	// Watch for the client going away while the request is in flight. The
	// terminal event is otherwise emitted only after the stream copy returns;
	// a client that disconnects mid-stream can leave the copy blocked, so we
	// emit the terminal here the moment r.Context() is cancelled instead of
	// waiting for the unwind. Cancelling r.Context() (client close, or our own
	// shutdown) also propagates to the ReverseProxy's upstream request, so the
	// engine stops generating. terminalOnce keeps this from double-emitting
	// with the normal path. The half-open case (no FIN, r.Context() never
	// fires) is caught instead by statusCapture's write deadline below.
	if wl != nil {
		reqCtx := r.Context()
		finished := make(chan struct{})
		defer close(finished)
		go func() {
			select {
			case <-reqCtx.Done():
				emitTerminal("cancelled", cancelReasonOr("client disconnected before completion"))
			case <-finished:
			}
		}()
	}

	// committedSC is the statusCapture of the candidate we committed to
	// streaming; its wroteErr tells us after the fact whether the client write
	// failed (dead/half-open client) so we can mark the workload failed. It is
	// assigned at the commit point as well as after the loop, because the
	// post-loop assignment does not always run — see finalize below.
	var committedSC *statusCapture
	var committedStream *streamFrameReader

	// finalize reports this request's outcome, and it is deferred because a
	// mid-copy error is never returned to us: ReverseProxy converts one into
	// panic(http.ErrAbortHandler) whenever it detects a real server, and that
	// panic unwinds straight past the end of this function. Reported inline,
	// both the terminal workload event and the proxy/request completion were
	// lost whenever an upstream died mid-stream while the client was healthy,
	// leaving the job "running" for the life of the broker. That is worse than
	// a stuck card: the scheduler counts queued and running by scheduledOn, so
	// the ghost permanently inflated that node's pending count and biased
	// routing away from a node whose only fault was a crashed engine.
	//
	// Client-side failures never had this problem. net/http cancels the
	// request context synchronously before a write error propagates
	// (checkConnErrorWriter), so the watcher above has always emitted by the
	// time the panic arrives. The upstream-died case is the one that had no
	// reporter at all.
	//
	// Registered after the reservation release above, so on unwind this runs
	// first and the node is still counted as loaded while we report.
	defer func() {
		aborted := recover()
		if ctl.sink != nil && currentAttempt != nil {
			ctl.observeSink(currentAttempt, ctl.sink.nativeFact())
			if aborted != nil {
				currentAttempt.observeAttempt(routingAttemptResult{factInterrupted, factStreamInterrupted, 0, true})
				ctl.observeIngress(routingIngressResult{factFailed, factStreamInterrupted, finalStatus})
			}
		}
		// A managed copy ends with normal HTTP EOF even on protocol failure;
		// its separate result, rather than the committed status, is decisive.
		if committedStream != nil {
			result := committedStream.Result()
			if result.Outcome == streamOutcomeInterrupted || result.Outcome == streamOutcomeUpstreamError {
				proxyErr = fmt.Sprintf("stream interrupted after output began on %s: %s", servedNodeID, result.Reason)
			}
		}
		if aborted != nil && proxyErr == "" {
			proxyErr = fmt.Sprintf("stream interrupted after output began on %s: upstream response copy aborted", servedNodeID)
		}
		reportErr := proxyErr
		if committedSC != nil && finalStatus == 0 {
			// The panic skipped the post-loop assignment, but the committed
			// capture still carries the status the upstream sent.
			finalStatus = committedSC.status
		}

		slog.Debug("proxy request complete",
			"id", reqID,
			"node_id", servedNodeID,
			"method", r.Method,
			"path", r.URL.Path,
			"target", servedTarget,
			"status", finalStatus,
			"duration_ms", time.Since(start).Milliseconds(),
			"ttfb_ms", ttfbMs,
			"err", reportErr,
		)

		_ = f.notify("proxy/request", RequestEvent{
			ID:       reqID,
			NodeID:   servedNodeID,
			Method:   r.Method,
			Path:     r.URL.Path,
			Target:   servedTarget,
			Status:   finalStatus,
			Duration: time.Since(start).Milliseconds(),
			TTFB:     ttfbMs,
			Error:    reportErr,
		})

		// Terminal workload transition pairs with the workload:started emitted
		// at forward time above. Cancellation, an upstream/transport error, a
		// failed client write (dead/half-open client), or any non-2xx status is
		// a failure; a clean 2xx is a completion. The Workload carries the same
		// id so the broker (and peers) can collapse the start/finish pair.
		// Routed through emitTerminal so the disconnect watcher and this path
		// emit exactly once.
		if wl != nil {
			switch {
			case r.Context().Err() != nil:
				// The request was cancelled before it finished — either the
				// client disconnected or, on shutdown, we cancelled it to stop
				// the in-flight inference. A mid-stream cancel never reaches
				// ErrorHandler (the 200 headers are already sent), so without
				// this branch it would be misreported as completed. (The
				// watcher above usually beats us to it; emitTerminal makes that
				// a no-op.) Cancelled rather than failed: nothing went wrong
				// here, the requester stopped waiting.
				emitTerminal("cancelled", cancelReasonOr("request cancelled before completion"))
			case committedSC != nil && committedSC.wroteErr != nil:
				// The response committed but a write to (or flush toward) the
				// client failed — typically the idle deadline tripping on a
				// dead/half-open client. The same event as the branch above,
				// differing only in how we noticed: a client that vanished
				// without a FIN never cancels the context, so the write
				// deadline is what surfaces it. Classified identically.
				emitTerminal("cancelled", "client connection lost: "+committedSC.wroteErr.Error())
			case proxyErr != "" || finalStatus < http.StatusOK || finalStatus >= http.StatusMultipleChoices:
				msg := proxyErr
				if msg == "" {
					msg = fmt.Sprintf("upstream returned HTTP %d", finalStatus)
				}
				emitTerminal("failed", msg)
			default:
				emitTerminal("completed", "")
			}
		}

		if aborted != nil {
			// Preserve ReverseProxy's contract with http.Server, which recovers
			// ErrAbortHandler silently and closes the connection. Swallowing it
			// here would leave the client waiting on a response that will never
			// be finished or closed.
			panic(aborted)
		}
	}()

	// repointWorkload records where the job is currently placed: a node id while
	// an attempt is in flight, empty between attempts. The scheduler counts
	// pending work by scheduledOn, so clearing it is what stops a node we have
	// given up on from still looking busy. The state stays "queued" throughout,
	// because "running" means the engine is generating and the broker's store
	// would reject a return to "queued" as a backwards transition.
	repointWorkload := func(nodeID, concrete string) {
		if wl == nil {
			return
		}
		wlMu.Lock()
		if terminated || (wl.ScheduledOn == nodeID && (concrete == "" || wl.Model == concrete)) {
			wlMu.Unlock()
			return
		}
		wl.ScheduledOn = nodeID
		if concrete != "" {
			wl.Model = concrete
		}
		wl.Seq = nextWlSeq()
		snapshot := *wl
		wlMu.Unlock()
		p.emitWorkload(workloadSubmittedMethod, snapshot)
	}

	// releaseHeld drops the capacity claim. Called before every backoff: the
	// reservation map is process-wide and shared with the other engine's
	// facade, so holding a claim on a node we have stopped using would push
	// that engine's traffic away from a node that is in fact free. It is the
	// same reasoning as clearing scheduledOn, applied to the proxy's own
	// estimate rather than the scheduler's.
	releaseHeld := func() {
		resMu.Lock()
		defer resMu.Unlock()
		p.releaseReservation(held)
		held = reservation{}
	}
	// takeHeld re-reserves for a fresh round, replacing any claim still held.
	// reserveCandidate both orders the round and takes the claim, so a retry
	// that re-resolves has to go back through it.
	takeHeld := func(cands []candidate) ([]candidate, error) {
		releaseHeld()
		out, proposal, err := f.reserveRoundPrepared(preparationCtx, cands, tier != "", model, prepareCandidate, resolve)
		resMu.Lock()
		held = proposal
		resMu.Unlock()
		roundReservations = proposal.held
		roundGeneration = preparedGeneration
		return out, inferencePreparationRefusal(err)
	}
	beforeRebuild = func() { repointWorkload("", ""); releaseHeld() }

	// Retry loop. Each iteration dispatches to one candidate; when a round's
	// candidates are exhausted it backs off and re-resolves, so a node that
	// recovered or a model that finished pulling becomes eligible mid-retry.
	//
	// Bounded by maxDispatchAttempts for dispatch failures and by jobDeadline
	// for the whole pre-commit life. Both bounds are checked before every
	// dispatch and against pending transport. Silent connection/headers/body
	// waits consume the same deadline; only a successful first raw body byte
	// stops that bound so an accepted healthy stream can continue to completion.
	//
	// Retrying is only possible before the first byte reaches the client; past
	// the commit point we are bound to that node, and a stream truncated from
	// there fails explicitly for supported streams. proxy/request-started fires
	// at the commit so it names the node that actually served. The self-forward guard lives in
	// resolveCandidates.
	maxDispatches := policy.maxAttempts
	dispatches := 0
	// A hybrid request's queue and cold-start allowances are totals across its
	// attempts. Each wait spends both, conservatively; none outlasts the one
	// pre-output deadline.
	queueLeft, coldLeft := policy.queueWait, policy.coldWait
	spendWaits := func(d time.Duration) {
		queueLeft, coldLeft = max(queueLeft-d, 0), max(coldLeft-d, 0)
	}
	// reserved is time the attempt may still spend before the destination
	// receives the request (a peer slot wait). The destination's budgets are
	// reduced by it, so waits on both sides never exceed one allowance.
	strictRequestFor := func(m nodepolicy.StrictAdmissionModel, reserved time.Duration) (nodepolicy.StrictAdmissionRequest, bool) {
		pre := (time.Until(deadline) - reserved).Milliseconds()
		if pre < 1 {
			return nodepolicy.StrictAdmissionRequest{}, false
		}
		budget := func(d time.Duration) int { return int(min(max(d-reserved, 0).Milliseconds(), pre)) }
		return nodepolicy.StrictAdmissionRequest{Version: nodepolicy.StrictAdmissionVersion,
			QueueBudgetMs: budget(queueLeft), ColdStartBudgetMs: budget(coldLeft), PreOutputBudgetMs: int(pre),
			Request: ctl.key, Identity: m.Identity}, true
	}
	// dispatchPublic offers the tier's API entries once, in configured order,
	// after the first round of local and peer candidates. A candidate that
	// publicAttemptPermitted refuses is skipped before any preparation and
	// spends no attempt; in this build that is every candidate. A permitted one
	// enters G's wrapper exactly once as a new attempt of this request.
	triedAccounts := map[string]bool{}
	dispatchPublic := func() bool {
		for _, entry := range p.admission.currentPolicy().Tiers[tier] {
			if entry.Kind != nodepolicy.EntryAPI {
				continue
			}
			if dispatches >= maxDispatches || !time.Now().Before(deadline) || r.Context().Err() != nil {
				return false
			}
			snapshot, err := p.accounts.prepareChatAccount(entry.Deployment)
			if err != nil || triedAccounts[snapshot.account.ID] || p.publicAttemptPermitted(tier, snapshot) != nil || publicChatInputs == nil {
				continue
			}
			q, transport, resolveCredential, budget, ok := publicChatInputs(snapshot)
			if !ok {
				continue
			}
			lifecycle, err := ctl.newAttempt(destinationDescriptor(snapshot.deployment.Model, snapshot.deployment.ID, snapshot.account.ID, true))
			if err != nil {
				return false
			}
			triedAccounts[snapshot.account.ID] = true
			dispatches++
			attemptCtx, cancelAttempt := context.WithCancel(r.Context())
			_, err = p.runPublicChatAttempt(attemptCtx, tier, snapshot, bodyBytes, q, transport, resolveCredential, budget, lifecycle,
				func(b []byte) error { _, err := w.Write(b); return err })
			cancelAttempt()
			lifecycle.finished(publicAttemptTerminal{Accounting: publicAccountingRetained, Err: err})
			ctl.mu.Lock()
			committed := ctl.committed
			ctl.mu.Unlock()
			if err == nil || committed {
				if err != nil {
					ctl.observeIngress(routingIngressResult{factFailed, failureFactReason(err), http.StatusOK})
				}
				// Output reached the client: this attempt is the answer even if
				// it then failed, and nothing else may write a response.
				servedNodeID, finalStatus = entry.Deployment, http.StatusOK
				if err != nil {
					proxyErr = "public attempt failed after output began: " + err.Error()
				}
				return true
			}
			// Refusal at any stage ends this attempt only. Each account is
			// offered once per request, so a cooldown is never retried here.
			proxyErr = "public attempt failed: " + err.Error()
		}
		return false
	}

	responseFinished := false
	respondedWithError := false
	regenExhausted := false
	// selfRejection is why this node refused its own candidate when that
	// refusal excluded it, and is the answer if nothing else remains.
	var selfRejection nodepolicy.RejectReason
	round := candidates
	next := 0
	firstAttempt := true

	for !responseFinished {
		if !firstAttempt {
			beforeRebuild()
			dropPrepared()
		}
		firstAttempt = false
		if dispatches >= maxDispatches {
			break
		}
		if !time.Now().Before(deadline) {
			ctl.expireBeforeOutput()
			break
		}
		// Abandon-now: the requester has gone, so there is nobody left to
		// receive an answer and the remaining attempts are better spent on work
		// someone is waiting for. The watcher has already emitted the terminal
		// as cancelled; the guard on the exhaustion response below keeps this
		// from also inventing a reason for a client that stopped listening.
		if r.Context().Err() != nil {
			break
		}
		if model != "" {
			p.admission.mu.Lock()
			stale := roundGeneration != p.admission.policyGeneration
			p.admission.mu.Unlock()
			if stale {
				beforeRebuild()
				var err error
				round, err = takeHeld(resolve())
				if err != nil {
					respondedWithError = true
					ctl.observeIngress(routingIngressResult{factFailed, failureFactReason(err), bufferRequestStatus(err)})
					finalStatus = bufferRequestStatus(err)
					proxyErr = "inference request preparation failed"
					if r.Context().Err() == nil {
						writeInferencePreparationError(w, err)
					}
					break
				}
				next = 0
			}
		}
		if next >= len(round) {
			// Round exhausted, or no owner was available. Either way the job is
			// on no node right now, so drop both the placement and the capacity
			// claim before waiting.
			repointWorkload("", "")
			releaseHeld()
			if publicPending {
				publicPending = false
				if dispatchPublic() {
					responseFinished = true
					break
				}
				if policy.hybrid && len(round) == 0 && len(resolve()) == 0 {
					ctl.observeIngress(routingIngressResult{factFailed, factNoOwner, http.StatusNotFound})
					// No local candidate remains and every public one is spent:
					// the tier has no eligible model, as the first check says.
					respondedWithError = true
					finalStatus = http.StatusNotFound
					proxyErr = "no eligible model for tier"
					writeProxyError(w, http.StatusNotFound, []string{"no eligible model for tier ", tier}, needs.names(), true)
					break
				}
				continue
			}
			// This node refused its own candidate for a reason no backoff will
			// change, and nothing else can take the request: answer now.
			if selfRejection != "" && len(resolve()) == 0 {
				ctl.observeIngress(routingIngressResult{factFailed, admissionFactReason(selfRejection), http.StatusServiceUnavailable})
				respondedWithError = true
				finalStatus = http.StatusServiceUnavailable
				proxyErr = "admission rejected: " + string(selfRejection)
				writeAdmissionRejection(w, selfRejection)
				break
			}
			if !waitBeforeRetry(preparationCtx, backoffFor(dispatches, time.Until(deadline))) {
				break
			}
			round = resolve()
			if model != "" {
				var err error
				round, err = takeHeld(round)
				if err != nil {
					respondedWithError = true
					ctl.observeIngress(routingIngressResult{factFailed, failureFactReason(err), bufferRequestStatus(err)})
					finalStatus = bufferRequestStatus(err)
					proxyErr = "inference request preparation failed"
					if r.Context().Err() == nil {
						writeInferencePreparationError(w, err)
					}
					break
				}
			}
			next = 0
			if len(round) == 0 && ctl.hasExclusions() {
				// A cancel, or this node's own pause, excluded the only node
				// left. Waiting for an owner would only run out the deadline on
				// a node this request may not use.
				if selfRejection != "" {
					ctl.observeIngress(routingIngressResult{factFailed, admissionFactReason(selfRejection), http.StatusServiceUnavailable})
					respondedWithError = true
					finalStatus = http.StatusServiceUnavailable
					proxyErr = "admission rejected: " + string(selfRejection)
					writeAdmissionRejection(w, selfRejection)
					break
				}
				regenExhausted = true
				break
			}
			if len(round) == 0 {
				// Every owner went away mid-flight. Not a rejection: keep
				// waiting for one to come back. This consumes no attempt, which
				// is why jobDeadline is the only bound on it.
				continue
			}
		}
		cand := round[next]
		next++
		// A node a cancel excluded stays out of this request, including later
		// entries of a round resolved before the exclusion.
		if ctl.isExcluded(cand.id) {
			continue
		}
		dispatches++
		// lastPermitted means no further dispatch can happen, so this attempt's
		// failure is the client's answer. It is not "last candidate in the
		// round" any more: another round may follow.
		lastPermitted := dispatches >= maxDispatches ||
			(!time.Now().Before(deadline))
		last := lastPermitted
		// What this attempt asks for: a tier candidate names its concrete model
		// and may be another engine of this process.
		candModel := model
		if cand.model != "" {
			candModel = cand.model
		}
		candEngine := cand.engineOr(f.profile.Name)
		candFacade := cand.fac
		if candFacade == nil {
			candFacade = f
		}

		// Each attempt runs on its own context, a child of the request's, so we
		// can abandon this dispatch without touching the request itself. The
		// distinction is load-bearing: the disconnect watcher and the terminal
		// classification both read r.Context(), the parent, so cancelling a
		// child cannot be mistaken for the client going away.
		attemptCtx, cancelAttempt := context.WithCancel(r.Context())
		att, err := ctl.newDispatchAttempt(attemptCtx, cancelAttempt, cand.id, destinationDescriptor(candModel, cand.id, candEngine, false))
		if err != nil {
			cancelAttempt()
			break
		}
		currentAttempt = att

		// The body this attempt sends: the request's own, with a tier's concrete
		// model in place of the tier name and every other byte untouched.
		prepare := func(ctx context.Context, profile nodepolicy.Profile, hasProfile bool, generation uint64) error {
			return prepareCandidate(ctx, cand, profile, hasProfile, generation)
		}
		preparationFailed := func(err error) {
			err = inferencePreparationRefusal(err)
			att.observeAttempt(routingAttemptResult{factRejected, failureFactReason(err), 0, true})
			ctl.observeIngress(routingIngressResult{factFailed, failureFactReason(err), bufferRequestStatus(err)})
			cancelAttempt()
			att.finished(publicAttemptTerminal{})
			respondedWithError = true
			finalStatus = bufferRequestStatus(err)
			proxyErr = "inference request preparation failed"
			writeInferencePreparationError(w, err)
		}
		if !cand.self {
			p.admission.mu.Lock()
			generation := p.admission.policyGeneration
			p.admission.mu.Unlock()
			prepareCtx, stop := context.WithDeadline(attemptCtx, deadline)
			err := prepare(prepareCtx, nodepolicy.Profile{}, false, generation)
			stop()
			if err != nil {
				preparationFailed(err)
				break
			}
		}
		// How long the destination may queue this attempt: not at all while
		// this round still has somewhere else to go, so a busy node hands the
		// request straight back; the policy's queue timeout for the last one.
		admissionWait := 0
		if next >= len(round) {
			admissionWait = p.admission.currentPolicy().Admission.QueueTimeoutSeconds
		}
		admissionTimeout := time.Duration(admissionWait) * time.Second
		admissionTimeout = min(admissionTimeout, max(time.Until(deadline), 0))

		// This node's own engine is admitted here, before dispatch, the way a
		// peer's ingress admits what we send it. A rejection is a failed
		// candidate like a peer's 503, and writes nothing to the client unless
		// it is the answer.
		target := cand.url
		admissionStarted := time.Now()
		var performanceWait time.Duration
		if cand.self && policy.hybrid {
			// The router admits its own hybrid candidate strictly, as
			// serveStrictLocal admits a peer's: the body is qualified first,
			// then AdmitStrict spends the remaining queue and cold allowances.
			var refusal error
			refusalCause := factReasonNone
			if m, ok := strictFor[hybridCandidateKey(cand)]; !ok {
				refusal = errStrictUnavailable
			} else if req, ok := strictRequestFor(m, 0); !ok {
				refusal = context.DeadlineExceeded
			} else {
				p.admission.mu.Lock()
				generation := p.admission.policyGeneration
				p.admission.mu.Unlock()
				prepareCtx, stop := context.WithDeadline(attemptCtx, deadline)
				strictGeneration, err := prepareStrictCandidate(prepareCtx, cand, candFacade, candEngine, candModel, m.Identity, generation)
				stop()
				if err == nil {
					strictHeld, err = p.admission.AdmitStrict(attemptCtx, candEngine, candModel, req, strictGeneration)
				}
				spendWaits(time.Since(admissionStarted))
				refusal = err
			}
			if refusal == nil {
				t := strictHeld.(*strictTicket)
				p.admission.mu.Lock()
				t.base.cancel = func() { ctl.abortLocalIf(att) }
				p.admission.mu.Unlock()
				stopStrictWatch = t.watchPreOutput(cancelAttempt)
				performanceWait = time.Since(admissionStarted)
				u, ok := candFacade.localBackendTarget()
				if !ok {
					refusalCause = factEngineNotRunning
					refusal = errStrictUnavailable
				}
				target = u
			}
			if refusal != nil {
				if refusalCause == factReasonNone {
					refusalCause = failureFactReason(refusal)
				}
				att.observeAdmission(time.Since(admissionStarted))
				att.observeAttempt(routingAttemptResult{factRejected, refusalCause, 0, true})
				finishStrict()
				cancelAttempt()
				att.finished(publicAttemptTerminal{})
				proxyErr = "strict admission refused: " + refusal.Error()
				if !last {
					continue
				}
				respondedWithError = true
				servedNodeID, servedTarget = cand.id, cand.url.Host
				finalStatus = writeStrictSelfRefusal(w, refusal)
				ctl.observeIngress(routingIngressResult{factFailed, refusalCause, finalStatus})
				break
			}
		} else if cand.self {
			decision := func() admissionDecision {
				ctx, stop := context.WithDeadline(attemptCtx, deadline)
				defer stop()
				return p.admission.admit(ctx, admissionRequest{
					resolvedInventory: true,
					engine:            candEngine,
					model:             candModel,
					waitCap:           admissionTimeout,
					cancel:            func() { ctl.abortLocalIf(att) },
					prepare:           prepare,
				})
			}()
			att.observeAdmission(time.Since(admissionStarted))
			rejectSelf := func(errText string, answer func()) bool {
				cancelAttempt()
				att.finished(publicAttemptTerminal{})
				proxyErr = errText
				if !last {
					slog.Debug("proxy self candidate not admitted, failing over",
						"id", reqID, "node_id", cand.id, "engine", candEngine, "err", errText)
					return false
				}
				respondedWithError = true
				servedNodeID = cand.id
				servedTarget = target.Host
				finalStatus = http.StatusServiceUnavailable
				answer()
				return true
			}
			if decision.err != nil {
				if attemptCtx.Err() == nil {
					preparationFailed(decision.err)
					break
				}
				cancelAttempt()
				att.finished(publicAttemptTerminal{})
				if att.redispatched() {
					dispatches--
				}
				// Otherwise the request itself ended while it waited; the loop's
				// own check ends it.
				continue
			}
			if decision.reject != "" {
				att.observeAttempt(routingAttemptResult{factRejected, admissionFactReason(decision.reject), 0, true})
				if decision.reject == nodepolicy.RejectBusy && len(round) == 1 {
					// With no alternative, queue slices must not spend the
					// failover budget. The job deadline still ends the wait.
					dispatches--
					last = false
				}
				if rejectSelf("admission rejected: "+string(decision.reject), func() {
					ctl.observeIngress(routingIngressResult{factFailed, admissionFactReason(decision.reject), http.StatusServiceUnavailable})
					writeAdmissionRejection(w, decision.reject)
				}) {
					break
				}
				// A node that is pausing or paused will refuse every later
				// round too, so it leaves this request rather than spending
				// its retry budget on the same refusal.
				if p.admission.currentAvailability() != nodepolicy.Available {
					ctl.exclude(cand.id)
					selfRejection = decision.reject
				}
				continue
			}
			att.ticket = decision.ticket
			performanceWait = time.Since(admissionStarted)
			if cand.dormant {
				// Admission waited for the wake; the engine's address is the one
				// it came up on.
				u, ok := candFacade.localBackendTarget()
				if !ok {
					att.observeAttempt(routingAttemptResult{factRejected, factEngineNotRunning, 0, true})
					att.ticket.release()
					if rejectSelf("local engine is not running", func() {
						ctl.observeIngress(routingIngressResult{factFailed, factEngineNotRunning, http.StatusServiceUnavailable})
						writeIngressError(w, http.StatusServiceUnavailable, "no-local-backend",
							"no local inference backend is available on this node")
					}) {
						break
					}
					continue
				}
				target = u
			}
		}
		if cand.self && policy.hybrid {
			att.observeAdmission(time.Since(admissionStarted))
		}
		rebasePrepared(attemptCtx)
		attemptCtx = preparedCtx
		if model != "" {
			resMu.Lock()
			hasClaim := held.held
			resMu.Unlock()
			if !hasClaim {
				var proposal reservation
				stable := preparedGeneration == roundGeneration
				if stable {
					proposal, stable = p.claimPreparedTarget(attemptCtx, cand.id, preparedGeneration, roundReservations)
				}
				if !stable {
					att.ticket.release()
					finishStrict()
					cancelAttempt()
					att.finished(publicAttemptTerminal{})
					next--
					dispatches--
					continue // Rebuild in this same driver without spending a dispatch.
				}
				resMu.Lock()
				held = proposal
				resMu.Unlock()
			}
		}
		repointWorkload(cand.id, candModel)

		// Watch the target while the attempt is pre-commit. A node that has
		// dropped out of discovery is not coming back to answer a request
		// already sent to it, and without this the attempt sat out the whole
		// first-content budget before failing over — while the broker had
		// already stamped the record failed from its own node-loss sweep, so
		// the UI said failed while the client still waited.
		//
		// Polling rather than subscribing keeps this out of Discovery's
		// contract, and the interval only bounds detection latency. Watching
		// stops at the commit point, because cancelling after that would kill a
		// stream that is working.
		//
		// A dormant self candidate is not in discovery until its engine has
		// been advertised again, so it is not watched.
		//
		// Read the interval once, on this goroutine: the watcher must not
		// depend on a value that could change under it mid-attempt.
		watchEvery := targetWatchInterval
		if !cand.dormant {
			go func() {
				t := time.NewTicker(watchEvery)
				defer t.Stop()
				for {
					select {
					case <-attemptCtx.Done():
						return
					case <-t.C:
						// The commit path may have claimed the outcome since the
						// last tick, in which case this attempt is no longer ours
						// to cancel.
						if att.settled() {
							return
						}
						if candFacade.discovery.Has(cand.id) {
							continue
						}
						if att.abandon() {
							cancelAttempt()
						}
						return
					}
				}
			}()
		}

		// ReverseProxy derives the outbound request from the one it is given,
		// so the attempt context has to travel on a copy. The replayed body
		// and the per-attempt admission header go on the copy for the same
		// reason, with headers cloned so one attempt's never leak into the
		// next.
		// The request travels on its own context so the slot bound below never
		// reassigns attemptCtx, which the target watcher already reads.
		// The destination's budgets are reduced by the origin's slot bound
		// because acquisition happens inside the transport after the strict
		// header is written. ErrPeerSlotOccupied costs one hybrid attempt.
		requestCtx, peerSlotWait := peerSlotContext(attemptCtx, policy, cand, queueLeft, time.Until(deadline))
		attemptReq := r.WithContext(requestCtx)
		attemptReq.Header = r.Header.Clone()
		if streamRequest.protocol != streamProtocolUnknown {
			// Explicit identity also prevents Go's transport from negotiating gzip.
			attemptReq.Header.Set("Accept-Encoding", "identity")
		}
		if bodyBytes != nil {
			setReplayBody(attemptReq, sendBody)
		}
		if cand.self {
			attemptReq.Header.Del(nodepolicy.AdmissionWaitHeader)
		} else {
			attemptReq.Header.Set(nodepolicy.AdmissionWaitHeader, strconv.Itoa(admissionWait))
		}
		if policy.hybrid && !cand.self {
			// The first production producer of the strict header. The
			// destination revalidates identity, origin and body itself.
			header := ""
			if m, ok := strictFor[hybridCandidateKey(cand)]; ok {
				if req, ok := strictRequestFor(m, peerSlotWait); ok {
					if raw, err := json.Marshal(req); err == nil {
						header = string(raw)
					}
				}
			}
			if header == "" {
				att.observeAttempt(routingAttemptResult{factRejected, factUnknown, 0, true})
				cancelAttempt()
				att.finished(publicAttemptTerminal{})
				proxyErr = "strict admission request unavailable"
				if !last {
					continue
				}
				respondedWithError = true
				servedNodeID, servedTarget = cand.id, target.Host
				finalStatus = writeStrictSelfRefusal(w, errStrictUnavailable)
				ctl.observeIngress(routingIngressResult{factFailed, factUnknown, finalStatus})
				break
			}
			attemptReq.Header.Set(nodepolicy.StrictAdmissionHeader, header)
		}
		retry := false
		retryBusy := false
		sc := &statusCapture{ResponseWriter: w, status: http.StatusOK, idle: idleClientWriteTimeout}
		var performance *performanceAttempt
		if cand.self {
			var err error
			performance, err = candFacade.newPerformanceAttemptOwned(preparedCtx, candModel, r.URL.Path, sendBody, performanceWait)
			if err != nil {
				att.ticket.release()
				finishStrict()
				preparationFailed(err)
				break
			}
		}
		if strictHeld != nil {
			// The engine is marked dirty before any inference byte is sent;
			// from here a release is uncertain until reconciliation.
			if err := strictHeld.BeforeStrictDispatch(); err != nil {
				att.observeAttempt(routingAttemptResult{factRejected, failureFactReason(err), 0, true})
				finishStrict()
				cancelAttempt()
				att.finished(publicAttemptTerminal{})
				proxyErr = "strict admission refused: " + err.Error()
				if !last {
					continue
				}
				respondedWithError = true
				servedNodeID, servedTarget = cand.id, target.Host
				finalStatus = writeStrictSelfRefusal(w, err)
				ctl.observeIngress(routingIngressResult{factFailed, failureFactReason(err), finalStatus})
				break
			}
			strictEvidence = strictUncertain
		}

		proxy := &httputil.ReverseProxy{
			Director: func(req *http.Request) {
				if cand.self {
					stripInternalRoutingHeaders(req.Header)
					req.Header.Del(peerStreamFramingHeader)
					withLocalCredential(req.Header, candFacade.localInferenceCredential(r, target))
				}
				req.URL.Scheme = target.Scheme
				req.URL.Host = target.Host
				req.Host = target.Host
			},
			// A remote cluster peer is dialed over mTLS (per-peer pinned config);
			// self/manual candidates use the plain transport. See candidateTransport.
			Transport: performanceTransport{RoundTripper: proxyResponseTransport{p.candidateTransport(cand)}, attempt: performance},
			// ModifyResponse fires when the upstream's status line + headers
			// have arrived but before the body streams. That's both the retry
			// decision point and, on commit, the time-to-first-byte boundary.
			ModifyResponse: func(resp *http.Response) error {
				att.observeStatus(resp.StatusCode)
				if resp.StatusCode < 200 || resp.StatusCode >= 300 {
					att.observeAttempt(routingAttemptResult{factRejected, factUpstreamStatus, resp.StatusCode, true})
				}
				metadata := resp.Header.Values(peerStreamFramingHeader)
				resp.Header.Del(peerStreamFramingHeader)
				retryBusy = len(round) == 1 && resp.StatusCode == http.StatusServiceUnavailable &&
					resp.Header.Get(nodepolicy.AdmissionHeader) == string(nodepolicy.RejectBusy)
				if (!last || retryBusy) && shouldRetry(resp.StatusCode) {
					// Abort before streaming: ReverseProxy closes resp.Body and
					// calls ErrorHandler with our sentinel, then we try next.
					retry = true
					return retrySignal{}
				}
				if resp.StatusCode < 200 || resp.StatusCode >= 300 {
					ctl.observeIngress(routingIngressResult{factFailed, factUpstreamStatus, resp.StatusCode})
				}
				if cand.peerUUID != "" && streamRequest.protocol != streamProtocolUnknown && resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
					var err error
					streamRequest, err = streamRequest.withPeerMetadata(metadata)
					if err != nil {
						att.observeAttempt(routingAttemptResult{factAttemptFailed, factPeerFraming, resp.StatusCode, true})
						return err // existing attempt failure path; never commit body bytes
					}
				}
				if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
					if !att.acceptResponse() {
						return context.Canceled
					}
					resp.Body = &routingResponseBody{ReadCloser: resp.Body, attempt: att}
				}
				// Hold the commit until the engine actually produces content.
				// Only for inference: a control endpoint's headers mean what
				// they say, and the model-list fan-out has its own client.
				// Waiting here costs a non-streaming response nothing, since
				// its headers and body arrive together. Framing is selected
				// separately, after the winning claim.
				var peeked io.ReadCloser
				bodyWait := max(time.Until(deadline), time.Nanosecond)
				if len(round) != 1 {
					bodyWait = min(firstBodyTimeout, bodyWait)
				}
				body, err := awaitFirstBody(resp.Body, bodyWait)
				if err != nil {
					if ctl.sink != nil && att.requestCommitted() {
						if !att.canDeliverResponse() {
							return context.Canceled
						}
						// Raw commitment precedes peek publication. A timer or
						// read error after that point ends this accepted status
						// through the sink, never ErrorHandler or another send.
						if !stderrors.Is(err, errFirstBodyTimeout) {
							_ = resp.Body.Close()
						}
						reason := streamInterruptReasonReadFailed
						if stderrors.Is(err, errFirstBodyTimeout) {
							reason = streamInterruptReasonIdle
						}
						body = ctl.sink.interruptedBodyForAttempt(streamRequest.protocol, reason, att)
						if body == nil {
							return context.Canceled
						}
						resp.ContentLength = -1
						clearOriginStreamIntegrity(resp.Header)
					} else {
						att.observeAttempt(routingAttemptResult{factAttemptFailed, factNoContent, resp.StatusCode, true})
						proxyErr = err.Error()
						if !last {
							retry = true
							slog.Warn("proxy upstream produced no content, failing over",
								"id", reqID, "node_id", cand.id, "target", target.Host,
								"path", r.URL.Path, "err", err)
							return retrySignal{}
						}
						slog.Warn("proxy upstream produced no content, retries exhausted",
							"id", reqID, "node_id", cand.id, "target", target.Host,
							"method", r.Method, "path", r.URL.Path,
							"duration_ms", time.Since(start).Milliseconds(), "err", err)
						return err
					}
				}
				peeked = body
				// Claim the attempt before any commit side effect. Losing the
				// claim means the watcher has already cancelled this attempt,
				// so committing would stream through a context that is about to
				// die and truncate the response we just started.
				if !att.canDeliverResponse() {
					if att.requestCommitted() {
						return context.Canceled
					}
					if att.redispatched() {
						// A cancel moved this request on before it committed.
						retry = true
						return retrySignal{}
					}
					proxyErr = errTargetGone.Error()
					att.observeAttempt(routingAttemptResult{factAttemptFailed, factTargetLeft, 0, true})
					if !last {
						retry = true
						slog.Warn("proxy target left discovery as the response committed, failing over",
							"id", reqID, "node_id", cand.id, "target", target.Host,
							"path", r.URL.Path)
						return retrySignal{}
					}
					return errTargetGone
				}
				// A hybrid self attempt's strict timer ends at the same commit.
				stopStrictWatch()
				// The peeked byte goes back only once the commit is ours, so a
				// lost claim leaves the body untouched for ReverseProxy to close.
				if peeked != nil {
					resp.Body = peeked
				}
				var reader *streamFrameReader
				if err == nil {
					reader, err = installResponseStream(resp, streamRequest, rebasePerformancePreparation(r.Context(), preparedCtx), streamIdle, performance)
				} else {
					err = nil
				}
				if err != nil {
					// Raw bytes already committed this request. Buffer refusal
					// must end through its sink without a new status or retry.
					if ctl.sink == nil || !att.requestCommitted() {
						return err
					}
					_ = resp.Body.Close()
					ending := ctl.sink.interruptedBodyForAttempt(streamRequest.protocol, streamInterruptReasonBudget, att)
					if ending == nil {
						return context.Canceled
					}
					resp.Body = ending
					resp.ContentLength = -1
					clearOriginStreamIntegrity(resp.Header)
				}
				committedStream = reader
				if resp.StatusCode != http.StatusSwitchingProtocols {
					sc.upstreamBody = &originResponseBody{ReadCloser: resp.Body, response: resp, managed: committedStream != nil}
					resp.Body = sc.upstreamBody
				}
				// Name what actually answered (FORK_DESIGN.md §3.8). Set on the
				// committed response only, so a failed-over attempt's headers
				// can never reach the client.
				resp.Header.Set(nodepolicy.ModelHeader, candModel)
				resp.Header.Set(nodepolicy.EngineHeader, candEngine)
				resp.Header.Set(nodepolicy.NodeHeader, cand.id)
				if tier != "" {
					resp.Header.Set(nodepolicy.TierHeader, tier)
				} else {
					resp.Header.Del(nodepolicy.TierHeader)
				}
				// Committing to this candidate — body stream is about to begin.
				ttfbMs = time.Since(start).Milliseconds()
				servedNodeID = cand.id
				servedTarget = target.Host
				proxyErr = "" // clear any error recorded from a failed-over candidate
				if ctl.sink != nil {
					proxyErr = ctl.sink.result().Reason
				}
				// Arm the liveness report only now. statusCapture also carries
				// the proxy's OWN error bodies — ReverseProxy's ErrorHandler
				// writes a failed dial's message through it — and those bytes
				// prove nothing about the node. Reaching here means the upstream
				// returned a status line, so everything written from this point
				// came from the node. Same goroutine as the body copy, so no
				// synchronization is needed.
				sc.upstreamAlive = func() { p.reportActivity(cand.id) }
				// Preserve the upstream response, including absent CORS permissions.
				if !started {
					started = true
					_ = f.notify("proxy/request-started", RequestStartedEvent{
						ID:     reqID,
						NodeID: cand.id,
						Method: r.Method,
						Path:   r.URL.Path,
						Target: target.Host,
					})
					// The claim already follows each dispatch, so this is a
					// no-op unless something reordered underneath us; it stays
					// as the belt-and-braces guarantee that the node that
					// actually served is the one counted as loaded.
					moveHeld(cand.id)
					// Record the commit here, not just after the loop: a copy
					// error aborts this handler by panic, and finalize needs to
					// know the response had committed (and on which capture) to
					// classify it.
					committedSC = sc
					// queued -> running: the engine is producing content, so
					// this is the first moment "running" is true. It also fixes
					// the placement on the node that actually served, which may
					// differ from the last node the queued updates named.
					// Guarded by wlMu against the disconnect watcher, and
					// skipped once terminated so a late transition cannot
					// resurrect a workload we have already finalized.
					if wl != nil {
						wlMu.Lock()
						if !terminated {
							startedMs := time.Now().UnixMilli()
							wl.State = "running"
							wl.StartedAt = &startedMs
							wl.ScheduledOn = cand.id
							wl.Model = candModel
							wl.Seq = nextWlSeq()
							snapshot := *wl
							wlMu.Unlock()
							p.emitWorkload(workloadStartedMethod, snapshot)
						} else {
							wlMu.Unlock()
						}
					}
				}
				return nil
			},
			ErrorHandler: func(ew http.ResponseWriter, _ *http.Request, err error) {
				if _, ok := err.(retrySignal); ok {
					return // retryable status — the loop advances to the next candidate
				}
				if ctl.deadlineExpired() {
					ctl.observeIngress(routingIngressResult{factFailed, factDeadline, http.StatusServiceUnavailable})
					respondedWithError = true
					servedNodeID, servedTarget = cand.id, target.Host
					proxyErr = routingDeadlineReason
					writeProxyError(ew, http.StatusServiceUnavailable, []string{proxyErr}, nil, false)
					return
				}
				if ctl.requestCancelled() || r.Context().Err() != nil {
					proxyErr = cancelReasonOr("request cancelled before completion")
					return
				}
				// A deliberate abort says nothing about the node or its
				// address, and is never the client's answer.
				if att.redispatched() {
					retry = true
					return
				}
				cause := factTransport
				var framing *peerFramingMetadataError
				if stderrors.As(err, &framing) {
					cause = factPeerFraming
				}
				if stderrors.Is(err, errFirstBodyTimeout) {
					cause = factNoContent
				}
				if stderrors.Is(err, errTargetGone) {
					cause = factTargetLeft
				}
				if stderrors.Is(err, clustertrust.ErrPeerSlotOccupied) {
					cause = factUnknown
				}
				att.observeAttempt(routingAttemptResult{factAttemptFailed, cause, 0, true})
				// Transport/dial error (not a status-based retry): forget this
				// node's confirmed address so the next request re-confirms and
				// can fail over to another of its published addresses
				// (multi-homed peer). The in-request failover below moves on to
				// the next node.
				// A hybrid attempt that found every peer slot occupied says
				// nothing about the address, which stays confirmed.
				if !stderrors.Is(err, clustertrust.ErrPeerSlotOccupied) {
					candFacade.targets.Forget(cand.id)
				}
				if !last {
					// Transport/dial error with candidates left: fail over.
					retry = true
					proxyErr = err.Error()
					slog.Warn("proxy upstream error, failing over",
						"id", reqID, "node_id", cand.id, "target", target.Host,
						"path", r.URL.Path, "err", err)
					return
				}
				// No further dispatch is permitted: terminal, surface it.
				respondedWithError = true
				servedNodeID = cand.id
				servedTarget = target.Host
				proxyErr = err.Error()
				slog.Warn("proxy upstream error, retries exhausted",
					"id", reqID, "node_id", cand.id, "target", target.Host,
					"method", r.Method, "path", r.URL.Path,
					"duration_ms", time.Since(start).Milliseconds(), "err", err)
				// A node that answered but never produced content timed out
				// rather than failed to be reached, and 504 says so.
				status := http.StatusBadGateway
				if stderrors.Is(err, errFirstBodyTimeout) {
					status = http.StatusGatewayTimeout
				}
				ctl.observeIngress(routingIngressResult{factFailed, cause, status})
				writeProxyError(ew, status, []string{"upstream error: ", proxyErr}, nil, false)
			},
		}

		func() {
			returned := false
			defer func() {
				// ReverseProxy closes on returned copy errors, but a client
				// writer panic can bypass that path. Join SC before root cleanup.
				if !returned && sc.upstreamBody != nil {
					_ = sc.upstreamBody.Close()
				}
				performance.finish(sc, returned)
				if ctl.sink != nil && !retry {
					ctl.observeSink(att, ctl.sink.nativeFact())
				}
			}()
			proxy.ServeHTTP(sc, attemptReq)
			if !retry && sc.wroteErr == nil {
				_ = sc.FlushError()
			}
			returned = true
		}()
		cancelAttempt()
		if policy.hybrid && !cand.self {
			// A peer attempt's slot wait and destination queue are spent by
			// the time it ends; charge them to the request's allowances.
			spendWaits(time.Since(admissionStarted))
		}
		// The admitted execution ends with the attempt: a retried attempt no
		// longer occupies this node, and a committed one has finished copying.
		att.ticket.release()
		finishStrict()
		att.finished(publicAttemptTerminal{})
		switch {
		case att.redispatched():
			// Moved on by a cancel, not failed: it costs no dispatch.
			retry = true
			dispatches--
			proxyErr = "attempt aborted to re-dispatch the request"
		case att.abandoned():
			// Name the real reason rather than the bare "context canceled" the
			// transport reports, which reads identically to a client hangup.
			proxyErr = errTargetGone.Error()
		}
		if retryBusy && !att.redispatched() {
			// Capture admission's reason before retrySignal discards the
			// response; only a sole candidate's busy refusal is free.
			dispatches--
		}
		if !retry {
			finalStatus = sc.status
			committedSC = sc
			responseFinished = true
		}
	}

	// Nothing committed and no attempt wrote the client's answer, so the loop
	// ended between attempts: either the budget ran out or, more usually, every
	// owner went away and none came back before the deadline. The dispatch
	// paths answer for themselves — a final transport failure writes its own
	// 502 and a non-retryable status is passed through — so this is the only
	// outcome left without a response.
	//
	// An abandoned request is excluded: writing to a client that has gone
	// achieves nothing, and attributing its end to a missing node would be a
	// fabricated reason for something that was the requester's own choice.
	if !responseFinished && !respondedWithError && r.Context().Err() == nil {
		// Name what the loop was actually doing when it ran out, which is what
		// the caller needs to know. The three cases are distinguishable: the
		// budget is gone; or the deadline passed while an owner was still there
		// to try, because slow attempts can reach 10 minutes before they reach
		// five dispatches; or the deadline passed with the candidate set empty,
		// which is the wait for an owner to come back.
		reason := "no node advertising the requested model became available before the retry deadline"
		cause := factNoOwner
		switch {
		case regenExhausted:
			cause = factRegenerateExhausted
			reason = "no other eligible node remained after the previous one was excluded"
		case dispatches >= maxDispatches:
			cause = factAttemptsExhausted
			reason = "every dispatch attempt failed"
		case len(round) > 0:
			cause = factDeadline
			reason = "the retry deadline passed while dispatch attempts were still failing"
		}
		ctl.observeIngress(routingIngressResult{factFailed, cause, http.StatusServiceUnavailable})
		proxyErr = reason
		finalStatus = http.StatusServiceUnavailable
		slog.Warn("proxy request exhausted its retry budget",
			"id", reqID, "method", r.Method, "path", r.URL.Path,
			"dispatches", dispatches, "duration_ms", time.Since(start).Milliseconds(),
			"reason", reason)
		// Advisory only: the caller knows its own patience better than we do.
		w.Header().Set("Retry-After", "1")
		writeProxyError(w, http.StatusServiceUnavailable, []string{reason}, nil, false)
	}

	// Reporting happens in the deferred finalize above, so it survives the
	// ErrAbortHandler panic that a mid-copy error raises.
}
