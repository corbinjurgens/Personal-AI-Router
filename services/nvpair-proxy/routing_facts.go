// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"io"
	"strconv"
	"sync/atomic"
	"time"

	"nvpair-shared/nodepolicy"
)

// Native facts are observations, never routing authority. All writers enter
// through the request mutex. Readers use only atomics and immutable inline
// descriptors; neither a reader nor a retired source retains an inference owner.
type routingFactReason uint8

const (
	factReasonNone routingFactReason = iota
	factUnknown
	factClientGone
	factCancel
	factRegenerate
	factNodePaused
	factProxyShutdown
	factTargetLeft
	factTransport
	factUpstreamStatus
	factNoContent
	factStreamInterrupted
	factUpstreamError
	factAdmissionPaused
	factAdmissionDraining
	factAdmissionBusy
	factAdmissionNoFit
	factAdmissionEngineOff
	factAdmissionWakeTimeout
	factEngineNotRunning
	factRequestInvalid
	factNoOwner
	factRegenerateExhausted
	factAttemptsExhausted
	factDeadline
	factRequestCeiling
	factAccountCeiling
	factAccountCooldown
	factAccountFrozen
	factAccountCredential
	factAccountExhausted
	factAccountConflict
	factLedgerUnavailable
	factDeploymentQuarantined
	factPeerFraming
)

var routingFactReasons = [...]string{"", "unknown", "client-gone", "cancel", "regenerate", "node-paused", "proxy-shutdown", "target-left", "transport", "upstream-status", "no-content", "stream-interrupted", "upstream-error", "admission-paused", "admission-draining", "admission-busy", "admission-no-fit", "admission-engine-off", "admission-wake-timeout", "engine-not-running", "request-invalid", "no-owner", "regenerate-exhausted", "attempts-exhausted", "deadline", "request-ceiling", "account-ceiling", "account-cooldown", "account-frozen", "account-credential", "account-exhausted", "account-conflict", "ledger-unavailable", "deployment-quarantined", "peer-framing"}

type routingFactState uint8

const (
	factRouting routingFactState = iota
	factStreaming
	factCompleted
	factFailed
	factCancelled
)

type routingFactOutcome uint8

const (
	factPending routingFactOutcome = iota
	factCommitted
	factRejected
	factAttemptFailed
	factRedispatched
	factInterrupted
	factAttemptCancelled
)

const (
	factMoneyApplicable uint64 = 1 << iota
	factCeilingKnown
	factHeldKnown
	factChargeKnown
	factFirstByteKnown
	factEndKnown
	factFrozen
	factLimitExceeded
	factHeldUnavailable
	factChargeUnavailable
)

const (
	factAccountingNone uint8 = iota
	factAccountingPending
	factAccountingRecorded
	factAccountingRetained
	factAccountingFailed
	factAccountingUnknown
)

const (
	factAccountUnknown uint8 = iota
	factAccountOK
	factAccountCooling
	factAccountIsFrozen
	factAccountBadCredential
	factAccountIsExhausted
	factAccountProbe
	factAccountHasConflict
)

const factMaxRevision = uint64(9007199254740991)
const factAttached = uint64(1) << 63

func factDisposition(state uint8, reason routingFactReason, status int) uint64 {
	return uint64(state) | uint64(reason)<<8 | uint64(status)<<16
}

// These descriptors own their bytes. Preparing them is restricted to the
// originating goroutine, outside identity, permission, admission and request locks.
type routingRequestedDescriptor struct {
	model      [512]byte
	length     uint16
	tier, mode uint8
}

type routingDestinationDescriptor struct {
	model                         [512]byte
	first, second                 [128]byte
	modelLen, firstLen, secondLen uint16
	kind                          uint8 // 1 local/peer, 2 API
	modelKnown                    bool  // ROOT amendment: supported model-less local inference.
}

type routingWorkloadDescriptor struct {
	engine, id       [128]byte
	generation       uint64
	engineLen, idLen uint16
}

type routingProcessDescriptor struct {
	origin, run       [128]byte
	originLen, runLen uint16
}

func requestedDescriptor(model, tier, path string) *routingRequestedDescriptor {
	d := &routingRequestedDescriptor{}
	if nodepolicy.ValidateModelIdentifier(model) == nil {
		d.length = uint16(copy(d.model[:], model))
	}
	switch tier {
	case "weak":
		d.tier = 1
	case "medium":
		d.tier = 2
	case "strong":
		d.tier = 3
	}
	switch path {
	case "/api/chat", "/v1/chat/completions", "/v1/messages":
		d.mode = 1
	case "/api/generate", "/v1/completions":
		d.mode = 2
	case "/api/embed", "/api/embeddings", "/v1/embeddings":
		d.mode = 3
	}
	return d
}

func destinationDescriptor(model, first, second string, api bool) *routingDestinationDescriptor {
	if (model != "" || api) && nodepolicy.ValidateModelIdentifier(model) != nil || !factIdentifier(first) || !factIdentifier(second) {
		return nil
	}
	d := &routingDestinationDescriptor{kind: 1, modelKnown: model != ""}
	if api {
		d.kind = 2
	}
	d.modelLen = uint16(copy(d.model[:], model))
	d.firstLen = uint16(copy(d.first[:], first))
	d.secondLen = uint16(copy(d.second[:], second))
	return d
}

func factIdentifier(s string) bool {
	if len(s) == 0 || len(s) > 128 {
		return false
	}
	for i, c := range []byte(s) {
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' {
			continue
		}
		if i > 0 && (c == '.' || c == '_' || c == ':' || c == '-') {
			continue
		}
		return false
	}
	return true
}

type routingNativeAttempt struct {
	ordinal, started, ended                                                             atomic.Uint64
	destination                                                                         atomic.Pointer[routingDestinationDescriptor]
	disposition, admission, firstByte, rawStarted, flags, accountVersion, held, charged atomic.Uint64
}

// Exactly sixteen request words and twelve words per ring cell on 64-bit Go.
// flags also packs accounting (bits 16..23); attempt flags packs account state
// (bits 24..31). Zero money and unavailable money remain distinguishable.
type routingNativeFacts struct {
	epoch, id, created, ended, disposition, ceiling, granted, held, charged, flags, total, binding, served atomic.Uint64
	requested                                                                                              atomic.Pointer[routingRequestedDescriptor]
	workload                                                                                               atomic.Pointer[routingWorkloadDescriptor]
	sinkVersion                                                                                            atomic.Uint64
	attempts                                                                                               [32]routingNativeAttempt
}

// begin/end require request.mu. Overflow freezes observation only. There is no
// callback here: the source-copy reader reports the fixed limit flag to J313.
func (f *routingNativeFacts) begin() bool {
	if f == nil || f.flags.Load()&factFrozen != 0 {
		return false
	}
	e := f.epoch.Load()
	if e/2 >= factMaxRevision-1 {
		f.epoch.Store(e + 1)
		f.flags.Store(f.flags.Load() | factFrozen | factLimitExceeded)
		f.epoch.Store(e + 2)
		return false
	}
	f.epoch.Store(e + 1)
	return true
}

func (f *routingNativeFacts) end() { f.epoch.Add(1) }

func (r *routingSpend) observeRequested(d *routingRequestedDescriptor) {
	r.mu.Lock()
	var old *routingRequestedDescriptor
	if r.facts != nil {
		old = r.facts.requested.Load()
	}
	r.mu.Unlock()
	if old != nil && *old == *d {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if f := r.facts; f.begin() {
		f.requested.Store(d)
		f.end()
	}
}

func factNow() uint64 { return uint64(max(time.Now().UnixMilli(), 0)) }

// One fixed 24B monotonic reference, charged to proxy control C. Attempt clocks
// occupy the native continuation word while SC has no continuation executor.
// No per-attempt clock or payload pointer is added to an I callback tail.
var routingFactsClock = time.Now()

func factTick() uint64 { return uint64(max(time.Since(routingFactsClock).Nanoseconds(), 0)) }

// attachRoutingFacts runs after successful canonical registration, before any
// body preparation. The independent source owner survives registry map discard.
func (p *Proxy) attachRoutingFacts(request *routingRequest, start time.Time) {
	if request == nil {
		return
	}
	id, err := strconv.ParseUint(request.key.RequestID, 10, 63)
	if err != nil {
		return
	}
	f := &routingNativeFacts{}
	f.id.Store(id)
	f.created.Store(uint64(start.UnixMilli()))
	f.epoch.Store(2)
	r := request.owner
	r.mu.Lock()
	if r.facts != nil || r.done {
		r.mu.Unlock()
		return
	}
	if r.cancelled {
		state := factCancelled
		if r.cancelCause == factDeadline {
			state = factFailed
		}
		f.disposition.Store(factDisposition(uint8(state), r.cancelCause, 0))
		f.ended.Store(max(f.created.Load(), factNow()))
		f.flags.Store(factEndKnown)
	}
	r.facts = f
	r.mu.Unlock()
	var process *routingProcessDescriptor
	if p.routingFactSource() == nil {
		process = &routingProcessDescriptor{}
		process.originLen = uint16(copy(process.origin[:], request.key.OriginNodeID))
		process.runLen = uint16(copy(process.run[:], request.key.ProxyRunID))
	}
	p.identityMu.Lock()
	if p.requestStorage.source == nil {
		p.requestStorage.source = &routingSourceDirectory{process: process}
	}
	p.requestStorage.source.insert(f)
	p.identityMu.Unlock()
}

func (r *routingSpend) freezeFacts() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if f := r.facts; f != nil {
		if f.begin() {
			f.flags.Store(f.flags.Load() | factFrozen)
			f.end()
		}
		r.facts = nil
	}
	r.observationContext, r.processContext = nil, nil
}

// Strict execution already has its own admission/cancellation authority. This
// observation handle does not enter active, acceptResponse or commit, and so
// cannot introduce ordinary dispatch arbitration into that path.
type routingFactAttempt struct {
	request *routingSpend
	ordinal uint64
}

func (r *routingSpend) strictFactAttempt(d *routingDestinationDescriptor) *routingFactAttempt {
	started := time.Now()
	a := &routingFactAttempt{request: r}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.facts == nil || r.facts.flags.Load()&(factEndKnown|factFrozen) != 0 {
		return nil
	}
	r.ordinal++
	a.ordinal = r.ordinal
	r.nativeAttemptLocked(a, d, started)
	return a
}

type routingStrictFactBody struct {
	io.ReadCloser
	attempt *routingFactAttempt
}

func (b *routingStrictFactBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if n > 0 {
		r := b.attempt.request
		r.mu.Lock()
		b.attempt.nativeFirstByteLocked()
		r.mu.Unlock()
	}
	return n, err
}

func (r *routingSpend) nativeAttemptLocked(a *routingFactAttempt, d *routingDestinationDescriptor, started time.Time) {
	at := uint64(max(started.UnixMilli(), 0))
	tick := uint64(max(started.Sub(routingFactsClock).Nanoseconds(), 0))
	f := r.facts
	if d == nil || f == nil || f.flags.Load()&factEndKnown != 0 || !f.begin() {
		return
	}
	// Preserve the original request creation time across wall-clock rollback.
	// Chaining through the previous end also bounds every retained predecessor
	// by the eventual single request-end event, without an attempt scan.
	at = max(at, f.created.Load())
	if previous := f.latestAttempt(); previous != nil {
		at = max(at, previous.started.Load(), previous.ended.Load())
	}
	c := &f.attempts[(a.ordinal-1)%32]
	c.ordinal.Store(a.ordinal)
	c.started.Store(at)
	c.ended.Store(0)
	c.destination.Store(d)
	c.disposition.Store(0)
	c.admission.Store(0)
	c.firstByte.Store(0)
	c.rawStarted.Store(tick)
	c.flags.Store(0)
	if d.kind == 2 {
		c.flags.Store(factMoneyApplicable)
	}
	c.accountVersion.Store(0)
	c.held.Store(0)
	c.charged.Store(0)
	f.total.Store(a.ordinal)
	if w := f.workload.Load(); w != nil && f.binding.Load() == factAttached && d.kind == 1 &&
		w.engineLen == d.secondLen && w.engine == d.second {
		f.binding.Store(a.ordinal | factAttached)
	}
	if binding := f.binding.Load(); binding&^factAttached != 0 && a.ordinal-(binding&^factAttached) >= 32 {
		f.binding.Store(binding &^ factAttached)
	}
	f.end()
}

func (a *routingFactAttempt) nativeCellLocked() (*routingNativeFacts, *routingNativeAttempt) {
	f := a.request.facts
	if f == nil {
		return nil, nil
	}
	c := &f.attempts[(a.ordinal-1)%32]
	if c.ordinal.Load() != a.ordinal {
		return f, nil
	}
	return f, c
}

func (f *routingNativeFacts) latestAttempt() *routingNativeAttempt {
	ordinal := f.total.Load()
	if ordinal == 0 {
		return nil
	}
	c := &f.attempts[(ordinal-1)%32]
	if c.ordinal.Load() != ordinal {
		return nil
	}
	return c
}

// One owner event supplies one time for both ends. Attempt creation chains its
// start through the preceding end, so the latest cell suffices for all bounds.
func (f *routingNativeFacts) eventEnd(c *routingNativeAttempt, at uint64) uint64 {
	at = max(at, f.created.Load())
	if c != nil {
		at = max(at, c.started.Load(), c.ended.Load())
	}
	return at
}

// Requires an already odd epoch. Status belongs to the actual upstream, never
// to a generated request-level error. Older ended attempts remain historical.
func (f *routingNativeFacts) endLatest(c *routingNativeAttempt, outcome routingFactOutcome, reason routingFactReason, at uint64) {
	if c == nil || c.flags.Load()&factEndKnown != 0 {
		return
	}
	c.disposition.Store(c.disposition.Load()&^uint64(0xffff) | factDisposition(uint8(outcome), reason, 0))
	c.ended.Store(at)
	c.flags.Store(c.flags.Load() | factEndKnown)
}

type routingAttemptResult struct {
	outcome routingFactOutcome
	reason  routingFactReason
	status  int
	ended   bool
}

func (a *routingFactAttempt) observeAttempt(result routingAttemptResult) {
	if a == nil || a.request == nil {
		return
	}
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	r.foldCancellationLocked()
	a.observeAttemptLocked(result)
}

func (a *routingFactAttempt) observeAttemptLocked(result routingAttemptResult) {
	f, c := a.nativeCellLocked()
	if c == nil {
		return
	}
	d := factDisposition(uint8(result.outcome), result.reason, result.status)
	old := c.disposition.Load()
	if c.flags.Load()&factEndKnown != 0 {
		return
	}
	if result.status == 0 {
		d |= old & (uint64(0xffff) << 16)
	}
	// A generic finish cannot erase a decision already written by its owner.
	if result.reason == factReasonNone && routingFactReason(old>>8&255) != factReasonNone {
		d = old
	}
	if old == d && (!result.ended || c.flags.Load()&factEndKnown != 0) {
		return
	}
	if !f.begin() {
		return
	}
	c.disposition.Store(d)
	if result.ended && c.flags.Load()&factEndKnown == 0 {
		c.ended.Store(f.eventEnd(c, factNow()))
		c.flags.Store(c.flags.Load() | factEndKnown)
	}
	f.end()
}

func (a *routingFactAttempt) observeStatus(status int) {
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	f, c := a.nativeCellLocked()
	if c == nil || status < 100 || status > 599 {
		return
	}
	d := c.disposition.Load()&0xffff | uint64(status)<<16
	if c.disposition.Load() != d && f.begin() {
		c.disposition.Store(d)
		f.end()
	}
}

func (a *routingFactAttempt) observeAdmission(elapsed time.Duration) {
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	f, c := a.nativeCellLocked()
	n := uint64(max(elapsed.Milliseconds(), 0))
	if c != nil && c.admission.Load() != n && f.begin() {
		c.admission.Store(n)
		f.end()
	}
}

func (a *routingFactAttempt) nativeFirstByteLocked() {
	f, c := a.nativeCellLocked()
	if c == nil || c.flags.Load()&factFirstByteKnown != 0 || !f.begin() {
		return
	}
	c.firstByte.Store((max(factTick(), c.rawStarted.Load()) - c.rawStarted.Load()) / uint64(time.Millisecond))
	c.flags.Store(c.flags.Load() | factFirstByteKnown)
	if f.flags.Load()&factEndKnown == 0 {
		f.disposition.Store(uint64(factStreaming))
	}
	f.end()
}

type routingIngressResult struct {
	state  routingFactState
	reason routingFactReason
	status int
}

func (r *routingSpend) observeIngress(result routingIngressResult) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.foldCancellationLocked()
	r.observeIngressLocked(result)
}

func (r *routingSpend) observeIngressLocked(result routingIngressResult) {
	r.observeIngressLockedAt(result, factNow())
}

func (r *routingSpend) observeIngressLockedAt(result routingIngressResult, at uint64) {
	outcome := factAttemptFailed
	if result.state == factCancelled {
		outcome = factAttemptCancelled
	}
	if result.state == factCompleted {
		outcome = factCommitted
	}
	r.terminalFactLocked(result, outcome, at)
}

func (a *routingFactAttempt) observeTerminal(result routingIngressResult, outcome routingFactOutcome) {
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	r.foldCancellationLocked()
	r.terminalFactLocked(result, outcome, factNow())
}

func (r *routingSpend) terminalFactLocked(result routingIngressResult, outcome routingFactOutcome, at uint64) {
	f := r.facts
	if f == nil {
		return
	}
	if f.flags.Load()&factEndKnown != 0 {
		old := f.disposition.Load()
		if result.status != 0 && old>>16 == 0 && old&0xffff == factDisposition(uint8(result.state), result.reason, 0) && f.begin() {
			f.disposition.Store(old | uint64(result.status)<<16)
			f.end()
		}
		return
	}
	if !f.begin() {
		return
	}
	c := f.latestAttempt()
	at = f.eventEnd(c, at)
	f.endLatest(c, outcome, result.reason, at)
	f.disposition.Store(factDisposition(uint8(result.state), result.reason, result.status))
	f.ended.Store(at)
	f.flags.Store(f.flags.Load() | factEndKnown)
	f.end()
}

func (r *routingSpend) nativeCancelLocked(cause routingFactReason) {
	r.cancelCause = cause
	state := factCancelled
	if cause == factDeadline {
		state = factFailed
	}
	r.observeIngressLocked(routingIngressResult{state, cause, 0})
}

// The entry and concrete result owners call this before generic finish can set
// done. It records their typed context observation even if AfterFunc has not
// entered cancellation yet. It neither runs cancellation effects nor replaces
// an already completed inference result.
func routingContextCause(ctx, process context.Context) routingFactReason {
	if process != nil && process.Err() != nil {
		return factProxyShutdown
	}
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return factDeadline
	}
	return factClientGone
}

func (r *routingSpend) foldCancellationLocked() {
	ctx := r.observationContext
	if ctx == nil || ctx.Err() == nil {
		return
	}
	cause := r.cancelCause
	if cause == factReasonNone {
		cause = routingContextCause(ctx, r.processContext)
	}
	r.nativeCancelLocked(cause)
}

func admissionFactReason(reason nodepolicy.RejectReason) routingFactReason {
	switch reason {
	case nodepolicy.RejectPaused:
		return factAdmissionPaused
	case nodepolicy.RejectDraining:
		return factAdmissionDraining
	case nodepolicy.RejectBusy:
		return factAdmissionBusy
	case nodepolicy.RejectNoFit:
		return factAdmissionNoFit
	case nodepolicy.RejectOff:
		return factAdmissionEngineOff
	case nodepolicy.RejectWake:
		return factAdmissionWakeTimeout
	}
	return factUnknown
}

func failureFactReason(err error) routingFactReason {
	var identifier *requestIdentifierError
	var qualification *strictQualificationError
	var flag *invalidStreamFlagError
	var framing *peerFramingMetadataError
	var ambiguous *ambiguousOptionsError
	if errors.As(err, &identifier) || errors.As(err, &qualification) || errors.As(err, &flag) || errors.As(err, &ambiguous) || errors.Is(err, errAmbiguousModel) {
		return factRequestInvalid
	}
	if errors.As(err, &framing) {
		return factPeerFraming
	}
	var reject *strictAdmissionError
	if errors.As(err, &reject) && nodepolicy.RejectReason(reject.Reason) != "unavailable" {
		return admissionFactReason(nodepolicy.RejectReason(reject.Reason))
	}
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return factDeadline
	case errors.Is(err, errFirstBodyTimeout):
		return factNoContent
	case errors.Is(err, errTargetGone):
		return factTargetLeft
	case errors.Is(err, errAccountCeiling):
		return factAccountCeiling
	case errors.Is(err, errAccountCooldown):
		return factAccountCooldown
	case errors.Is(err, errAccountFrozen), errors.Is(err, errUsageAboveBound):
		return factAccountFrozen
	case errors.Is(err, errAccountCredential), errors.Is(err, errChatCredential):
		return factAccountCredential
	case errors.Is(err, errAccountConflict):
		return factAccountConflict
	case errors.Is(err, errDeploymentQuarantined):
		return factDeploymentQuarantined
	case errors.Is(err, errLedgerUnavailable), errors.Is(err, errLedgerUnconfigured), errors.Is(err, errChatAccounting):
		return factLedgerUnavailable
	}
	return factUnknown
}

// D authors this value under its mutex and transfers it synchronously after
// unlock. It contains no ledger, hold ID, policy, credential or account pointer.
type accountAttemptObservation struct {
	version           uint64
	held, charged     int64
	flags             uint64
	state, accounting uint8
	reason            routingFactReason
	refused           bool
}

func (a *publicAttemptLifecycle) observeAccount(o accountAttemptObservation) {
	if a == nil || a.request == nil {
		return
	}
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	previous := a.accountObservation
	if o.version <= previous.version {
		return
	}
	a.accountObservation = o
	oldValue, newValue := previous, o
	oldValue.version, newValue.version = 0, 0
	if oldValue == newValue {
		return
	}
	f := r.facts
	if !f.begin() {
		return
	}
	// Each actual attempt owns its previous contribution even after ring trim.
	// These are D observations, not sums reconstructed from display history.
	held, heldErr := addNanoUSD(int64(f.held.Load())-previous.held, o.held)
	charged, chargeErr := addNanoUSD(int64(f.charged.Load())-previous.charged, o.charged)
	flags := f.flags.Load() | factMoneyApplicable | factHeldKnown | factChargeKnown
	if heldErr != nil || o.flags&factHeldKnown == 0 {
		flags |= factHeldUnavailable
	}
	if chargeErr != nil || o.flags&factChargeKnown == 0 {
		flags |= factChargeUnavailable
	}
	if flags&factHeldUnavailable != 0 {
		flags &^= factHeldKnown
	} else {
		f.held.Store(uint64(held))
	}
	if flags&factChargeUnavailable != 0 {
		flags &^= factChargeKnown
	} else {
		f.charged.Store(uint64(charged))
	}
	accounting := o.accounting
	priorAccounting := uint8(flags >> 16)
	// A request retains an accounting failure/uncertainty from trimmed attempts.
	if priorAccounting == factAccountingFailed || priorAccounting == factAccountingUnknown {
		accounting = priorAccounting
	}
	if accounting == factAccountingRecorded && held > 0 {
		accounting = factAccountingRetained
	}
	f.flags.Store(flags&^uint64(255<<16) | uint64(accounting)<<16)
	if c := &f.attempts[(a.ordinal-1)%32]; c.ordinal.Load() == a.ordinal {
		c.held.Store(uint64(o.held))
		c.charged.Store(uint64(o.charged))
		c.accountVersion.Store(o.version)
		c.flags.Store(c.flags.Load()&^(uint64(255<<16)|uint64(255<<24)|factHeldKnown|factChargeKnown) | o.flags | uint64(o.accounting)<<16 | uint64(o.state)<<24 | factMoneyApplicable)
		if o.reason != factReasonNone && c.flags.Load()&factEndKnown == 0 {
			d := c.disposition.Load()
			c.disposition.Store(d&^uint64(255<<8) | uint64(o.reason)<<8)
		}
		if o.refused && c.flags.Load()&factEndKnown == 0 {
			c.disposition.Store(c.disposition.Load()&^uint64(255) | uint64(factRejected))
			c.ended.Store(f.eventEnd(c, factNow()))
			c.flags.Store(c.flags.Load() | factEndKnown)
		}
	}
	f.end()
}

// The registry calls this with the actual association generation while
// identity -> request is held. Delayed observations of an older association
// cannot detach a newly bound tuple; duplicate detaches are no-ops.
func (r *routingSpend) nativeWorkloadDetachedLocked(generation uint64) {
	f := r.facts
	if f == nil {
		return
	}
	w := f.workload.Load()
	if w == nil || w.generation != generation || f.binding.Load()&factAttached == 0 || !f.begin() {
		return
	}
	f.binding.Store(f.binding.Load() &^ factAttached)
	f.end()
}

type routingSinkFact struct {
	version             uint64
	status              int
	protocol, delivered bool
	outcome             streamOutcome
	reason              routingFactReason
}

// Strict origins retain their existing sink-less transport and result owners.
// EOF/protocol and statusCapture's actual write/flush result qualify delivery;
// neither HTTP success nor handler return alone supplies a terminal fact.
func strictRoutingResult(sc *statusCapture, reader *streamFrameReader, returned bool, ctx context.Context) routingSinkFact {
	o := routingSinkFact{version: 1, status: sc.status}
	if reader != nil {
		result := reader.Result()
		o.outcome = result.Outcome
		o.protocol = result.Outcome == streamOutcomeCompleted
		switch result.Outcome {
		case streamOutcomeInterrupted:
			o.reason = factStreamInterrupted
		case streamOutcomeUpstreamError:
			o.reason = factUpstreamError
		case streamOutcomeCancelled:
			o.reason = factUnknown
		}
		if result.Reason == streamInterruptReasonPeerLink {
			o.reason = factPeerFraming
		}
	} else if sc.upstreamBody != nil && sc.upstreamBody.complete {
		o.protocol, o.outcome = true, streamOutcomeCompleted
	}
	o.delivered = returned && sc.upstreamBody != nil && sc.upstreamBody.complete && sc.wroteErr == nil && ctx.Err() == nil
	if sc.wroteErr != nil {
		o.reason = factClientGone
	}
	if ctx.Err() != nil {
		o.outcome = streamOutcomeCancelled
		if o.reason == factReasonNone {
			o.reason = factUnknown
		}
	}
	if !returned && o.reason == factReasonNone && ctx.Err() == nil {
		o.reason = factStreamInterrupted
	}
	return o
}

func (s *responseSink) nativeFact() routingSinkFact {
	s.mu.Lock()
	defer s.mu.Unlock()
	o := routingSinkFact{version: s.factVersion, status: s.status, protocol: s.protocolComplete, outcome: s.factOutcome}
	o.delivered = s.streamFinished && s.status != 0 && s.flushSucceeded && s.writeErr == nil && s.written == s.produced && s.flushed >= s.produced
	if s.writeErr != nil {
		o.reason = factClientGone
	} else {
		switch s.factOutcome {
		case streamOutcomeInterrupted:
			o.reason = factStreamInterrupted
		case streamOutcomeUpstreamError:
			o.reason = factUpstreamError
		case streamOutcomeCancelled:
			o.reason = factUnknown // The request owner folds its typed cause first.
		}
		if s.reason == streamInterruptReasonPeerLink {
			o.reason = factPeerFraming
		}
	}
	return o
}

func (r *routingSpend) observeSink(a *publicAttemptLifecycle, o routingSinkFact) {
	if a == nil {
		return
	}
	a.routingFactAttempt.observeSink(o)
}

func (a *routingFactAttempt) observeSink(o routingSinkFact) {
	r := a.request
	at := factNow()
	r.mu.Lock()
	defer r.mu.Unlock()
	r.foldCancellationLocked()
	f, c := a.nativeCellLocked()
	if f == nil || a.ordinal != f.total.Load() || o.version <= f.sinkVersion.Load() || !f.begin() {
		return
	}
	f.sinkVersion.Store(o.version)
	at = f.eventEnd(c, at)
	if f.flags.Load()&factEndKnown != 0 {
		at = f.ended.Load()
	}
	if f.flags.Load()&factEndKnown != 0 && f.disposition.Load()>>16 == 0 && o.status != 0 {
		f.disposition.Store(f.disposition.Load() | uint64(o.status)<<16)
	}
	if f.flags.Load()&factEndKnown == 0 && (o.delivered || o.reason != factReasonNone) {
		state := factFailed
		if o.protocol && o.delivered && o.status >= 200 && o.status < 300 {
			state = factCompleted
			f.served.Store(a.ordinal)
		}
		if o.reason == factClientGone || o.outcome == streamOutcomeCancelled {
			state = factCancelled
		}
		f.disposition.Store(factDisposition(uint8(state), o.reason, o.status))
		f.ended.Store(at)
		f.flags.Store(f.flags.Load() | factEndKnown)
	}
	if c != nil && (o.delivered || o.reason != factReasonNone) {
		outcome := factAttemptFailed
		if o.protocol && o.delivered {
			outcome = factCommitted
		} else if c.flags.Load()&factFirstByteKnown != 0 {
			outcome = factInterrupted
		}
		if o.reason == factClientGone || o.outcome == streamOutcomeCancelled {
			outcome = factAttemptCancelled
		}
		reason := o.reason
		if state := routingFactState(f.disposition.Load() & 255); state == factCancelled || state == factFailed && routingFactReason(f.disposition.Load()>>8&255) == factDeadline {
			reason = routingFactReason(f.disposition.Load() >> 8 & 255)
			outcome = factAttemptCancelled
			if reason == factDeadline {
				outcome = factAttemptFailed
			}
		}
		// The sink also sees generated error responses. Only ModifyResponse/
		// the public transport owns an attempt's actual upstream status.
		c.disposition.Store(factDisposition(uint8(outcome), reason, int(c.disposition.Load()>>16)))
		if c.flags.Load()&factEndKnown == 0 {
			c.ended.Store(at)
			c.flags.Store(c.flags.Load() | factEndKnown)
		}
	}
	f.end()
}

// Heap mutation is serialized by identityMu. Only registration changes its
// epoch; an evicted live request has no insertion hook on later updates.
type routingSourceDirectory struct {
	epoch, count atomic.Uint64
	heap         [256]atomic.Pointer[routingNativeFacts]
	process      *routingProcessDescriptor
	reader       routingSourceScan // The sole cleared pointer bank, used by J313's sole copier.
}

func (p *Proxy) routingFactSource() *routingSourceDirectory {
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	return p.requestStorage.source
}

func sourceLess(a, b *routingNativeFacts) bool {
	ac, bc := a.created.Load(), b.created.Load()
	return ac < bc || ac == bc && a.id.Load() < b.id.Load()
}

func (d *routingSourceDirectory) insert(f *routingNativeFacts) {
	d.epoch.Add(1)
	defer d.epoch.Add(1)
	n := int(d.count.Load())
	if n == len(d.heap) {
		if !sourceLess(d.heap[0].Load(), f) {
			return
		}
		d.heap[0].Store(f)
		for i := 0; ; {
			left := 2*i + 1
			if left >= n {
				break
			}
			child := left
			if left+1 < n && sourceLess(d.heap[left+1].Load(), d.heap[left].Load()) {
				child++
			}
			if !sourceLess(d.heap[child].Load(), d.heap[i].Load()) {
				break
			}
			a := d.heap[i].Load()
			d.heap[i].Store(d.heap[child].Load())
			d.heap[child].Store(a)
			i = child
		}
		return
	}
	d.heap[n].Store(f)
	d.count.Store(uint64(n + 1))
	for i := n; i > 0; {
		parent := (i - 1) / 2
		if !sourceLess(d.heap[i].Load(), d.heap[parent].Load()) {
			break
		}
		a := d.heap[parent].Load()
		d.heap[parent].Store(d.heap[i].Load())
		d.heap[i].Store(a)
		i = parent
	}
}

// J313 owns exactly one scan bank and one bounded snapshot scratch. A failed
// scan/copy must be retried on a later copier turn, never in a producer loop.
type routingSourceScan struct {
	sources [256]*routingNativeFacts
	epoch   uint64
	count   int
}

func (s *routingSourceScan) clear() { clear(s.sources[:]); s.count = 0; s.epoch = 0 }

func (d *routingSourceDirectory) scan() bool {
	s := &d.reader
	s.clear()
	e := d.epoch.Load()
	if e&1 != 0 {
		return false
	}
	n := int(d.count.Load())
	for i := 0; i < n; i++ {
		s.sources[i] = d.heap[i].Load()
	}
	if d.epoch.Load() != e {
		s.clear()
		return false
	}
	s.epoch, s.count = e, n
	return true
}

type routingNativeAttemptCopy struct {
	ordinal, started, ended, disposition, admission, firstByte, continuation, flags, accountVersion, held, charged uint64
	destination                                                                                                    routingDestinationDescriptor
}

type routingNativeCopy struct {
	revision, id, created, ended, disposition, ceiling, granted, held, charged, flags, total, binding, served, sinkVersion uint64
	requested                                                                                                              routingRequestedDescriptor
	workload                                                                                                               routingWorkloadDescriptor
	attempts                                                                                                               [32]routingNativeAttemptCopy
}

func (f *routingNativeFacts) copyInto(out *routingNativeCopy) bool {
	*out = routingNativeCopy{}
	e := f.epoch.Load()
	if e&1 != 0 {
		return false
	}
	out.revision, out.id, out.created, out.ended = e/2, f.id.Load(), f.created.Load(), f.ended.Load()
	out.disposition, out.ceiling, out.granted, out.held, out.charged = f.disposition.Load(), f.ceiling.Load(), f.granted.Load(), f.held.Load(), f.charged.Load()
	out.flags, out.total, out.binding, out.served, out.sinkVersion = f.flags.Load(), f.total.Load(), f.binding.Load(), f.served.Load(), f.sinkVersion.Load()
	if p := f.requested.Load(); p != nil {
		out.requested = *p
	}
	if p := f.workload.Load(); p != nil {
		out.workload = *p
	}
	for i := range f.attempts {
		c, v := &f.attempts[i], &out.attempts[i]
		v.ordinal, v.started, v.ended, v.disposition = c.ordinal.Load(), c.started.Load(), c.ended.Load(), c.disposition.Load()
		v.admission, v.firstByte, v.flags = c.admission.Load(), c.firstByte.Load(), c.flags.Load()
		v.continuation = 0 // No continuation executor; rawStarted is producer-only.
		v.accountVersion, v.held, v.charged = c.accountVersion.Load(), c.held.Load(), c.charged.Load()
		if p := c.destination.Load(); p != nil {
			v.destination = *p
		}
	}
	if f.epoch.Load() != e {
		*out = routingNativeCopy{}
		return false
	}
	return true
}

// accept verifies membership and revision before J assigns any projection seq.
// A directory change invalidates the whole scan, even if this key survived.
func (d *routingSourceDirectory) accept(index int, revision uint64, out *routingNativeCopy) bool {
	*out = routingNativeCopy{}
	s := &d.reader
	if index < 0 || index >= s.count || s.epoch&1 != 0 || d.epoch.Load() != s.epoch {
		return false
	}
	f := s.sources[index]
	if f == nil || d.heap[index].Load() != f || !f.copyInto(out) || out.revision <= revision || d.epoch.Load() != s.epoch {
		*out = routingNativeCopy{}
		return false
	}
	return true
}
