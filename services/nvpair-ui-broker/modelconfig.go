// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"reflect"
	"sort"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

const residentDispatchRequestMethod = "engine:dispatch-resident-request"
const residentDispatchReplyMethod = "engine:dispatch-resident-reply"

// Reuse the Serve-owned idle tick for failed attempts. Successful replies wait
// at most two ticks for watcher publication before evaluating inventory again.
const residentObservationTimeout = 2 * idleCheckInterval

type residentPendingObservation struct {
	evidence modelConfigEvidence
	expires  time.Time
}

type residentUnsettled struct {
	op       *modelConfigAttempt
	finish   nodepolicy.FinishResidentParams
	terminal bool
	// A terminal reply ends native execution even if finish cannot be delivered.
	// Retain its uncertainty as bounded observation, not a permanent proxy hold.
	expires time.Time
}

// modelConfigEvidence is an immutable capture. RPCs never hold policy locks.
// Source handles are compared separately from the serialized evidence.
type modelConfigEvidence struct {
	Policy        nodepolicy.Policy
	Live          nodepolicy.Availability
	Paused        bool
	Recovery      bool
	Activation    policyActivation
	Readiness     nodepolicy.StrictAdmissionReadiness
	Identity      nodepolicy.SetModelIdentityParams
	Intent        map[string]bool
	Running       map[string]bool
	Loaded        map[string][]string
	LoadedCurrent map[string]map[string]bool
	Drains        map[string]bool
	Active        map[string]int
	Proxy         *proxyProcess        `json:"-"`
	Worker        *rpcWorker           `json:"-"`
	Source        *engineManagerSource `json:"-"`
	Run           string
	Key           [32]byte `json:"-"`
}

type modelConfigAttempt struct {
	ctx          context.Context
	evidence     modelConfigEvidence
	prepare      nodepolicy.PrepareResidentParams
	reservation  string
	deadline     time.Time
	grantStarted bool
	granted      bool
	recovery     *nodepolicy.ObserveResidentParams
}

type modelConfigRuntime struct {
	wake           chan struct{}
	busy           bool
	last           *modelConfigEvidence
	retryAt        time.Time
	status         nodepolicy.ModelConfiguration
	sequence       uint64
	sequenceProxy  *proxyProcess
	sequenceRun    string
	active         *modelConfigAttempt
	loadedSource   *engineManagerSource
	loadedProxy    *proxyProcess
	loadedRequests map[nodepolicy.Role]nodepolicy.ResidentLoadOptions
	loadedIdentity nodepolicy.SetModelIdentityParams
	confirmed      map[nodepolicy.Role]modelConfigEvidence
	unsettled      map[nodepolicy.Role]residentUnsettled
	pending        map[nodepolicy.Role]residentPendingObservation
}

func (b *Broker) signalModelConfig() {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if r.models.wake == nil {
		r.models.wake = make(chan struct{}, 1)
	}
	select {
	case r.models.wake <- struct{}{}:
	default:
	}
	r.mu.Unlock()
}

func (b *Broker) captureModelConfig() modelConfigEvidence {
	b.workersMu.Lock()
	w, source := b.engineMgr, b.engineMgrSource
	b.workersMu.Unlock()
	pp := b.policyProxy()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	e := modelConfigEvidence{Policy: r.current, Live: r.live, Paused: r.localPauseRequested, Recovery: r.recoveryRequired,
		Activation: r.activationLocked(pp, r.current.Revision), Readiness: r.strictReadinessLocked(pp, b.engineModelNaming),
		Identity: r.modelIdentityLocked(), Intent: r.intent, Running: r.running, Loaded: r.loaded, Drains: r.drains,
		Active: r.admission.ActiveByEngine, Proxy: pp, Worker: w, Source: source, Run: r.runID}
	if source != r.engineManagerSource {
		e.Source = nil
	}
	if r.models.loadedSource != source || r.models.loadedProxy != pp {
		e.Loaded = nil
	}
	e.LoadedCurrent = make(map[string]map[string]bool)
	for _, ref := range b.residentRefs(e.Policy) {
		previous, current := r.models.loadedIdentity.Engines[ref.Engine], e.Identity.Engines[ref.Engine]
		pr, qualified := b.residentProfile(e.Policy, ref)
		loaded, known := r.models.loadedRequests[ref]
		if qualified && known && loaded == residentLoadRequest(pr) && previous.EngineRunID == current.EngineRunID && previous.SettingsFingerprint == current.SettingsFingerprint && previous.LaunchDigest == current.LaunchDigest && previous.ModelDigests[ref.Model] == current.ModelDigests[ref.Model] && reflect.DeepEqual(previous.Serving[ref.Model], current.Serving[ref.Model]) {
			if e.LoadedCurrent[ref.Engine] == nil {
				e.LoadedCurrent[ref.Engine] = make(map[string]bool)
			}
			e.LoadedCurrent[ref.Engine][ref.Model] = true
		}
	}
	// Marshal while locked, then decode to detach every map and policy slice.
	data, err := json.Marshal(e)
	if err != nil {
		e.Loaded = nil
		e.Readiness = nodepolicy.StrictAdmissionReadiness{}
		e.Source = nil
		return e
	}
	e.Key = sha256.Sum256(data)
	var detached modelConfigEvidence
	if json.Unmarshal(data, &detached) != nil {
		e.Loaded = nil
		e.Readiness = nodepolicy.StrictAdmissionReadiness{}
		e.Source = nil
		return e
	}
	detached.Key, detached.Proxy, detached.Worker, detached.Source = e.Key, pp, w, e.Source
	return detached
}

func sameModelEvidence(a, b modelConfigEvidence) bool {
	return a.Key == b.Key && a.Proxy == b.Proxy && a.Worker == b.Worker && a.Source == b.Source
}

// Admission accounting and loaded readback can change because of this very
// operation. They trigger later evaluation but do not revoke its authority.
func sameModelAuthority(a, b modelConfigEvidence) bool {
	if a.Proxy != b.Proxy || a.Worker != b.Worker || a.Source != b.Source {
		return false
	}
	a.Key, b.Key = [32]byte{}, [32]byte{}
	a.Active, b.Active = nil, nil
	a.Loaded, b.Loaded = nil, nil
	a.LoadedCurrent, b.LoadedCurrent = nil, nil
	a.Proxy, b.Proxy = nil, nil
	a.Worker, b.Worker = nil, nil
	a.Source, b.Source = nil, nil
	return reflect.DeepEqual(a, b)
}

func (b *Broker) modelRef(engine, model string) nodepolicy.Role {
	if naming, ok := b.engineModelNaming(engine); ok {
		model = engines.NormalizeModelName(model, naming)
	}
	return nodepolicy.Role{Engine: engine, Model: model}
}

func (b *Broker) residentRefs(p nodepolicy.Policy) []nodepolicy.Role {
	seen := map[nodepolicy.Role]bool{}
	refs := []nodepolicy.Role{}
	for _, ref := range p.ResidentModels {
		ref = b.modelRef(ref.Engine, ref.Model)
		if !seen[ref] {
			refs = append(refs, ref)
			seen[ref] = true
		}
	}
	sort.Slice(refs, func(i, j int) bool {
		if refs[i].Engine != refs[j].Engine {
			return refs[i].Engine < refs[j].Engine
		}
		return refs[i].Model < refs[j].Model
	})
	return refs
}

func (b *Broker) residentProfile(p nodepolicy.Policy, ref nodepolicy.Role) (nodepolicy.Profile, bool) {
	var result nodepolicy.Profile
	found := false
	for _, pr := range p.Profiles {
		if b.modelRef(pr.Engine, pr.Model) == ref {
			if found {
				return nodepolicy.Profile{}, false
			}
			result, found = pr, true
		}
	}
	return result, found
}

// Use the exact dispatch payload for inventory freshness, so every effective
// load option participates and unrelated policy edits do not invalidate it.
func residentLoadRequest(pr nodepolicy.Profile) nodepolicy.ResidentLoadOptions {
	return nodepolicy.ResidentLoadOptions{ContextLength: pr.ContextTokens, KeepAliveSeconds: 60}
}

func (b *Broker) modelConfigEntry(e modelConfigEvidence, ref nodepolicy.Role) (nodepolicy.ModelConfigurationEntry, nodepolicy.Profile, nodepolicy.StrictModelIdentity, int) {
	entry := nodepolicy.ModelConfigurationEntry{Engine: ref.Engine, Model: ref.Model, DesiredResident: true, State: nodepolicy.ModelStateUnknown, Reason: nodepolicy.ModelReasonObservationUnknown}
	var identity nodepolicy.StrictModelIdentity
	pr, qualified := b.residentProfile(e.Policy, ref)
	count := 0
	for _, r := range b.residentRefs(e.Policy) {
		if r.Engine == ref.Engine {
			count++
		}
	}
	if models, known := e.Loaded[ref.Engine]; known && e.Source != nil {
		loaded := false
		for _, model := range models {
			if b.modelRef(ref.Engine, model) == ref {
				loaded = true
			}
		}
		entry.ObservedLoaded = &loaded
	}
	block := func(reason nodepolicy.ModelConfigurationReason) {
		entry.State, entry.Reason = nodepolicy.ModelStateBlocked, reason
	}
	switch {
	case e.Live != nodepolicy.Available || e.Paused:
		block(nodepolicy.ModelReasonPausedLocal)
	case !e.Intent[ref.Engine]:
		block(nodepolicy.ModelReasonEngineOffLocal)
	case e.Drains[ref.Engine]:
		block(nodepolicy.ModelReasonSwitchBusy)
	case e.Activation.Activation != activationActive || e.Source == nil || e.Worker == nil:
		entry.Reason = nodepolicy.ModelReasonProjectionUnknown
	case !e.Running[ref.Engine]:
		block(nodepolicy.ModelReasonEngineOff)
	case !qualified:
		block(nodepolicy.ModelReasonUnqualified)
	case entry.ObservedLoaded == nil:
	default:
		if e.Readiness.Capability != nil {
			for _, m := range e.Readiness.Capability.Models {
				if b.modelRef(m.Engine, m.Model) == ref {
					identity = m.Identity
				}
			}
		}
		id := e.Identity.Engines[ref.Engine]
		envelope, ok := id.Serving[ref.Model]
		if identity.Validate() != nil {
			block(nodepolicy.ModelReasonUnqualified)
		} else if !ok || envelope.Validate(ref.Model, identity.ModelDigest) != nil || envelope.PerSlotContextTokens != int64(pr.ContextTokens) || envelope.Retention != "-1" || envelope.RouterCapacity < int64(count) {
			block(nodepolicy.ModelReasonUnsupportedLoad)
		} else {
			entry.State, entry.Reason = nodepolicy.ModelStatePending, ""
		}
	}
	return entry, pr, identity, count
}

// A completed load is a fact about a declaration, independent of eligibility.
// Worker replacement does not establish that the native allocation unloaded.
func (b *Broker) residentDeclarationCurrent(previous, current modelConfigEvidence, ref nodepolicy.Role) bool {
	if previous.Policy.Revision != current.Policy.Revision {
		return false
	}
	for _, desired := range b.residentRefs(current.Policy) {
		if desired == ref {
			return true
		}
	}
	return false
}

func sameResidentAllocation(a, b modelConfigEvidence, ref nodepolicy.Role) bool {
	return a.Policy.Revision == b.Policy.Revision && a.Proxy == b.Proxy && a.Worker == b.Worker && a.Source == b.Source && a.Run == b.Run && reflect.DeepEqual(a.Identity.Engines[ref.Engine], b.Identity.Engines[ref.Engine])
}

// Pending facts prevent redispatch even when their original source can no
// longer prove satisfaction. Local vetoes still determine the published state.
func residentProofLocked(r *policyRuntime, e modelConfigEvidence, ref nodepolicy.Role, entry nodepolicy.ModelConfigurationEntry) nodepolicy.ModelConfigurationEntry {
	if pending, ok := r.models.pending[ref]; ok && pending.evidence.Policy.Revision == e.Policy.Revision && r.now().Before(pending.expires) && entry.State == nodepolicy.ModelStatePending {
		if e.LoadedCurrent[ref.Engine][ref.Model] && entry.ObservedLoaded != nil && *entry.ObservedLoaded && !sameResidentAllocation(pending.evidence, e, ref) {
			return entry // Fresh replacement-source inventory still needs envelope readback.
		}
		entry.State, entry.Reason = nodepolicy.ModelStateUnknown, nodepolicy.ModelReasonObservationUnknown
		if sameResidentAllocation(pending.evidence, e, ref) {
			loaded := true
			entry.State, entry.Reason, entry.ObservedLoaded = nodepolicy.ModelStateSatisfied, "", &loaded
		}
		return entry
	}
	if proof, ok := r.models.confirmed[ref]; ok && sameModelAuthority(proof, e) && entry.State == nodepolicy.ModelStatePending && entry.ObservedLoaded != nil && *entry.ObservedLoaded {
		entry.State = nodepolicy.ModelStateSatisfied
	}
	return entry
}

// modelConfiguration publishes only evidence belonging to this revision.
func (b *Broker) modelConfiguration(e modelConfigEvidence) nodepolicy.ModelConfiguration {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.models.last != nil && sameModelEvidence(*r.models.last, e) && r.models.status.PolicyRevision == e.Policy.Revision {
		out := r.models.status
		out.Models = append([]nodepolicy.ModelConfigurationEntry{}, out.Models...)
		return out
	}
	out := nodepolicy.ModelConfiguration{PolicyRevision: e.Policy.Revision, ObservationGeneration: r.models.status.ObservationGeneration, Models: []nodepolicy.ModelConfigurationEntry{}}
	for _, ref := range b.residentRefs(e.Policy) {
		entry, _, _, _ := b.modelConfigEntry(e, ref)
		entry = residentProofLocked(r, e, ref, entry)
		out.Models = append(out.Models, entry)
	}
	return out
}

func (b *Broker) modelConfigurationFor(policy nodepolicy.Policy) nodepolicy.ModelConfiguration {
	e := b.captureModelConfig()
	if e.Policy.Revision == policy.Revision {
		return b.modelConfiguration(e)
	}
	// A concurrent save must not attach newer status to the earlier set result.
	e.Policy = policy
	e.Activation = policyActivation{Activation: activationPending}
	e.Loaded = nil
	e.Source = nil
	return b.modelConfiguration(e)
}

// reconcileModelConfig is called by the Serve-owned idle loop. Concurrent calls
// coalesce; failures retry on the existing idle tick without spinning on wakes.
func (b *Broker) reconcileModelConfig(ctx context.Context) {
	e := b.captureModelConfig()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if r.models.busy || (r.models.last != nil && sameModelEvidence(*r.models.last, e) && r.now().Before(r.models.retryAt)) {
		r.mu.Unlock()
		return
	}
	r.models.busy = true
	for ref, proof := range r.models.confirmed {
		if !b.residentDeclarationCurrent(proof, e, ref) {
			delete(r.models.confirmed, ref)
		}
	}
	for ref, hold := range r.models.unsettled {
		// Keep original ownership across worker replacement; it is not unload
		// evidence. A terminal reply can instead leave bounded observation;
		// the Serve-owned idle tick expires it even if the proxy is retired.
		if !b.residentDeclarationCurrent(hold.op.evidence, e, ref) || (hold.terminal && !r.now().Before(hold.expires)) {
			delete(r.models.unsettled, ref)
		}
	}
	for ref, pending := range r.models.pending {
		if !b.residentDeclarationCurrent(pending.evidence, e, ref) || !r.now().Before(pending.expires) {
			delete(r.models.pending, ref)
			delete(r.models.confirmed, ref)
		} else if entry, _, _, _ := b.modelConfigEntry(e, ref); sameResidentAllocation(pending.evidence, e, ref) && e.LoadedCurrent[ref.Engine][ref.Model] && entry.ObservedLoaded != nil && *entry.ObservedLoaded {
			// The correlated load already proved the envelope. A same-authority
			// watcher publication completes its observation phase.
			if r.models.confirmed == nil {
				r.models.confirmed = make(map[nodepolicy.Role]modelConfigEvidence)
			}
			r.models.confirmed[ref] = pending.evidence
			delete(r.models.pending, ref)
		}
	}
	r.models.last = &e
	r.models.retryAt = r.now().Add(idleCheckInterval)
	r.models.status = nodepolicy.ModelConfiguration{PolicyRevision: e.Policy.Revision, ObservationGeneration: r.models.status.ObservationGeneration + 1, Models: []nodepolicy.ModelConfigurationEntry{}}
	for _, ref := range b.residentRefs(e.Policy) {
		entry, _, _, _ := b.modelConfigEntry(e, ref)
		r.models.status.Models = append(r.models.status.Models, entry)
	}
	r.mu.Unlock()
	defer func() {
		r.mu.Lock()
		r.models.busy = false
		r.models.active = nil
		r.mu.Unlock()
		b.notifyPolicyChanged()
	}()
	for i, ref := range b.residentRefs(e.Policy) {
		entry, pr, id, count := b.modelConfigEntry(e, ref)
		r.mu.Lock()
		entry = residentProofLocked(r, e, ref, entry)
		hold, unsettled := r.models.unsettled[ref]
		r.mu.Unlock()
		if entry.State == nodepolicy.ModelStatePending && ctx.Err() == nil {
			switch {
			case unsettled:
				entry.State, entry.Reason = nodepolicy.ModelStateUnknown, nodepolicy.ModelReasonOperationTimeout
				// Native execution has ended. A replacement authority may confirm
				// that allocation, but only with fresh inventory and envelope proof.
				if hold.terminal && hold.finish.Outcome == nodepolicy.ResidentLoaded && !sameResidentAllocation(hold.op.evidence, e, ref) && e.LoadedCurrent[ref.Engine][ref.Model] && entry.ObservedLoaded != nil && *entry.ObservedLoaded {
					entry = b.confirmResident(ctx, e, entry)
					if entry.State == nodepolicy.ModelStateSatisfied {
						r.mu.Lock()
						delete(r.models.unsettled, ref)
						r.mu.Unlock()
						break
					}
				}
				cleanup, cancel := context.WithTimeout(ctx, proxyCallTimeout)
				var finished nodepolicy.FinishResidentResult
				settled := residentRPC(cleanup, hold.op.evidence.Proxy, nodepolicy.MethodFinishResident, hold.finish, &finished) && finished.ReservationID == hold.finish.ReservationID
				if settled && finished.RecoveryID != "" {
					settled = hold.terminal && b.residentSourceCurrent(hold.op) && b.recoverResident(cleanup, hold.op, finished)
				}
				cancel()
				if settled {
					r.mu.Lock()
					delete(r.models.unsettled, ref)
					if hold.finish.Outcome == nodepolicy.ResidentLoaded && finished.Outcome == nodepolicy.ResidentLoaded {
						if r.models.pending == nil {
							r.models.pending = make(map[nodepolicy.Role]residentPendingObservation)
						}
						r.models.pending[ref] = residentPendingObservation{evidence: hold.op.evidence, expires: r.now().Add(residentObservationTimeout)}
					}
					r.models.last = nil // Evaluate the settled inventory on the next wake.
					r.mu.Unlock()
					b.signalModelConfig()
				}
			case e.LoadedCurrent[ref.Engine][ref.Model] && entry.ObservedLoaded != nil && *entry.ObservedLoaded:
				entry = b.confirmResident(ctx, e, entry)
				if entry.State == nodepolicy.ModelStateSatisfied {
					r.mu.Lock()
					delete(r.models.pending, ref)
					r.mu.Unlock()
				}
			default:
				entry = b.reconcileResident(ctx, e, entry, pr, id, count)
			}
		}
		r.mu.Lock()
		_, awaitingObservation := r.models.pending[ref]
		if entry.State == nodepolicy.ModelStateSatisfied && !awaitingObservation {
			if r.models.confirmed == nil {
				r.models.confirmed = make(map[nodepolicy.Role]modelConfigEvidence)
			}
			r.models.confirmed[ref] = e
		}
		r.models.status.Models[i] = entry
		r.mu.Unlock()
		if !sameModelAuthority(e, b.captureModelConfig()) {
			return
		}
	}
}

// A fresh loaded inventory can confirm an existing native allocation after a
// worker restart. Read the manager-owned observed envelope; cold declarations
// and name presence alone never authorize satisfaction or another load.
func (b *Broker) confirmResident(parent context.Context, e modelConfigEvidence, entry nodepolicy.ModelConfigurationEntry) nodepolicy.ModelConfigurationEntry {
	entry.State, entry.Reason = nodepolicy.ModelStateUnknown, nodepolicy.ModelReasonObservationUnknown
	ctx, cancel := context.WithTimeout(parent, proxyCallTimeout)
	defer cancel()
	var launch settings.LaunchState
	if switchManagerCallOn(ctx, e.Worker, "engine:get-launch", map[string]string{"engine": entry.Engine}, &launch) != nil || !sameModelEvidence(e, b.captureModelConfig()) || !launch.Running || launch.Adopted || launch.Engine != entry.Engine || launch.ServingReadback == nil || launch.ServingReadback.Status != "observed" {
		return entry
	}
	envelope := e.Identity.Engines[entry.Engine].Serving[entry.Model]
	matched := 0
	for _, current := range launch.ServingReadback.LaunchValidated {
		if current.ModelID == entry.Model {
			if !reflect.DeepEqual(current, envelope) {
				return entry
			}
			matched++
		}
	}
	if matched != 1 {
		return entry
	}
	matched = 0
	for _, observed := range launch.ServingReadback.Observed {
		if observed.ModelID == entry.Model {
			if observed.PerSlotContextTokens != envelope.PerSlotContextTokens || observed.Slots != envelope.Slots {
				return entry
			}
			matched++
		}
	}
	if matched == 1 {
		entry.State, entry.Reason = nodepolicy.ModelStateSatisfied, ""
	}
	return entry
}

func residentRPC(ctx context.Context, pp *proxyProcess, method string, params any, result any) bool {
	data, err := json.Marshal(params)
	if err != nil || pp == nil {
		return false
	}
	raw, rpcErr, err := pp.Call(ctx, method, data)
	return err == nil && rpcErr == nil && json.Unmarshal(raw, result) == nil
}

func (b *Broker) residentCurrent(op *modelConfigAttempt) bool {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	active := r.models.active == op && r.now().Before(op.deadline)
	r.mu.Unlock()
	return active && op.ctx.Err() == nil && b.residentSourceCurrent(op)
}

func (b *Broker) residentSourceCurrent(op *modelConfigAttempt) bool {
	e := b.captureModelConfig()
	return e.Loaded[op.prepare.Engine] != nil && sameModelAuthority(op.evidence, e) && b.engineMgrPairCurrent(op.evidence.Worker, op.evidence.Source)
}

func (b *Broker) reconcileResident(parent context.Context, e modelConfigEvidence, entry nodepolicy.ModelConfigurationEntry, pr nodepolicy.Profile, id nodepolicy.StrictModelIdentity, count int) nodepolicy.ModelConfigurationEntry {
	bound := min(time.Minute, settings.SwitchDrainBudget, time.Duration(e.Policy.Admission.SwitchTimeoutSeconds)*time.Second)
	if bound <= 0 {
		entry.State, entry.Reason = nodepolicy.ModelStateBlocked, nodepolicy.ModelReasonUnqualified
		return entry
	}
	ctx, cancel := context.WithTimeout(parent, bound)
	defer cancel()
	deadline, _ := ctx.Deadline()
	var manager switchManagerSnapshot
	if switchManagerCallOn(ctx, e.Worker, settings.SwitchPrepareMethod, settings.SwitchPrepareRequest{}, &manager) != nil || manager.Error != nil || manager.Configuration.ManagerIncarnation == "" || !sameModelEvidence(e, b.captureModelConfig()) {
		entry.State, entry.Reason = nodepolicy.ModelStateUnknown, nodepolicy.ModelReasonProjectionUnknown
		return entry
	}
	incarnation := manager.Configuration.ManagerIncarnation
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if r.models.sequenceProxy != e.Proxy || r.models.sequenceRun != e.Run {
		r.models.sequence = 0
		r.models.sequenceProxy = e.Proxy
		r.models.sequenceRun = e.Run
	}
	r.models.sequence++
	opID, err := nodepolicy.ResidentOperationID(r.models.sequence)
	if err != nil {
		r.mu.Unlock()
		entry.State, entry.Reason = nodepolicy.ModelStateBlocked, nodepolicy.ModelReasonProjectionUnknown
		return entry
	}
	op := &modelConfigAttempt{ctx: ctx, evidence: e, deadline: deadline, prepare: nodepolicy.PrepareResidentParams{PolicyRevision: e.Policy.Revision, ProxyRunID: e.Run, ManagerIncarnation: incarnation, Engine: entry.Engine, Model: entry.Model, Identity: id, OperationID: opID, TimeoutMs: int(time.Until(deadline).Milliseconds())}}
	r.models.active = op
	r.mu.Unlock()
	holdUnsettled := true
	var finish nodepolicy.FinishResidentParams
	terminal := false
	defer func() {
		r.mu.Lock()
		// An uncertain dispatched hold belongs to the original proxy. Ordinary
		// inventory cannot discharge it or authorize another load.
		if op.granted && holdUnsettled {
			if r.models.unsettled == nil {
				r.models.unsettled = make(map[nodepolicy.Role]residentUnsettled)
			}
			r.models.unsettled[nodepolicy.Role{Engine: entry.Engine, Model: entry.Model}] = residentUnsettled{op: op, finish: finish, terminal: terminal, expires: r.now().Add(residentObservationTimeout)}
		}
		if r.models.active == op {
			r.models.active = nil
		}
		r.mu.Unlock()
	}()
	entry.State, entry.Reason = nodepolicy.ModelStateUnknown, nodepolicy.ModelReasonOperationTimeout
	if op.prepare.Validate() != nil {
		return entry
	}
	if !b.residentCurrent(op) {
		return entry
	}
	var prepared nodepolicy.PrepareResidentResult
	if !residentRPC(ctx, e.Proxy, nodepolicy.MethodPrepareResident, op.prepare, &prepared) {
		return entry
	}
	if len(prepared.Refusals) > 0 {
		entry.State, entry.Reason = nodepolicy.ModelStateBlocked, prepared.Refusals[0].Reason
		return entry
	}
	r.mu.Lock()
	op.reservation = prepared.ReservationID
	r.mu.Unlock()
	result := nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}
	if b.residentCurrent(op) {
		params := nodepolicy.ReconcileResidentParams{OperationID: opID, PolicyRevision: e.Policy.Revision, DeclaredResidentCount: count, Engine: entry.Engine, Model: entry.Model, Identity: id, TimeoutMs: int(time.Until(deadline).Milliseconds()), Reservation: nodepolicy.ResidentReservationEvidence{ReservationID: prepared.ReservationID, ProxyRunID: e.Run, ManagerIncarnation: incarnation}, Load: residentLoadRequest(pr)}
		if params.Validate() == nil {
			raw, rpcErr, callErr := e.Worker.CallNoTimeout(ctx, nodepolicy.MethodReconcileResident, settingsJSON(params))
			result.Outcome = nodepolicy.ResidentUnknown
			terminal = callErr == nil && rpcErr == nil && json.Unmarshal(raw, &result) == nil
			if !terminal {
				result = nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentUnknown}
			}
		}
	}
	// A response must attest this exact operation's source and identity.
	if o := result.Observation; o != nil && (o.ManagerIncarnation != incarnation || o.Engine != entry.Engine || o.Model != entry.Model || o.Identity != id) {
		result = nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentUnknown}
		terminal = false
	}
	r.mu.Lock()
	granted := op.granted
	r.mu.Unlock()
	if result.Outcome == nodepolicy.ResidentLoaded && !granted {
		result = nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentUnknown}
		terminal = false
	}
	if terminal && result.Outcome == nodepolicy.ResidentLoaded {
		r.mu.Lock()
		if r.models.loadedRequests == nil {
			r.models.loadedRequests = make(map[nodepolicy.Role]nodepolicy.ResidentLoadOptions)
		}
		r.models.loadedRequests[nodepolicy.Role{Engine: entry.Engine, Model: entry.Model}] = residentLoadRequest(pr)
		r.mu.Unlock()
	}
	finish = nodepolicy.FinishResidentParams{ReservationID: prepared.ReservationID, ProxyRunID: e.Run, OperationID: opID, Outcome: result.Outcome, Observation: result.Observation}
	// Finish is cleanup on the original proxy even after revocation. A separate
	// bounded cleanup call does not extend the load/grant deadline.
	cleanup, stop := context.WithTimeout(parent, proxyCallTimeout)
	defer stop()
	var finished nodepolicy.FinishResidentResult
	if !residentRPC(cleanup, e.Proxy, nodepolicy.MethodFinishResident, finish, &finished) {
		return entry
	}
	if finished.ReservationID != prepared.ReservationID {
		return entry
	}
	if finished.RecoveryID != "" {
		// Only a received terminal manager response is a completion barrier.
		// Watcher caches and elapsed time can never discharge this hold.
		if terminal && b.residentSourceCurrent(op) {
			holdUnsettled = !b.recoverResident(cleanup, op, finished)
		}
		return entry
	}
	holdUnsettled = false
	if result.Outcome == nodepolicy.ResidentLoaded && finished.Outcome == nodepolicy.ResidentLoaded {
		// Completion survives changes in desire, including a veto arriving
		// after dispatch. Eligibility controls status, not retention of fact.
		if b.residentCurrent(op) {
			loaded := true
			entry.State, entry.Reason, entry.ObservedLoaded = nodepolicy.ModelStateSatisfied, "", &loaded
		}
		r.mu.Lock()
		if r.models.pending == nil {
			r.models.pending = make(map[nodepolicy.Role]residentPendingObservation)
		}
		r.models.pending[nodepolicy.Role{Engine: entry.Engine, Model: entry.Model}] = residentPendingObservation{evidence: e, expires: r.now().Add(residentObservationTimeout)}
		r.mu.Unlock()
	} else if result.Outcome == nodepolicy.ResidentNotDispatched {
		entry.State, entry.Reason = nodepolicy.ModelStateBlocked, nodepolicy.ModelReasonUnsupportedLoad
	}
	return entry
}

func (b *Broker) recoverResident(ctx context.Context, op *modelConfigAttempt, finish nodepolicy.FinishResidentResult) bool {
	if !b.residentSourceCurrent(op) {
		return false
	}
	if op.recovery != nil {
		// The proxy retains an exact idempotent receipt. Never substitute a
		// later inventory read after an acknowledgement was lost.
		var result nodepolicy.ObserveResidentResult
		return residentRPC(ctx, op.evidence.Proxy, nodepolicy.MethodObserveResident, *op.recovery, &result) && result.ReservationID == finish.ReservationID && result.Settled
	}
	raw, rpcErr, err := op.evidence.Worker.CallNoTimeout(ctx, "engine:models", nil)
	if err != nil || rpcErr != nil || !b.residentSourceCurrent(op) {
		return false
	}
	var inventory struct {
		Loaded map[string][]string `json:"loadedByEngine"`
	}
	if json.Unmarshal(raw, &inventory) != nil {
		return false
	}
	models, known := inventory.Loaded[op.prepare.Engine]
	if !known || models == nil {
		return false
	}
	loaded := false
	for _, model := range models {
		if b.modelRef(op.prepare.Engine, model) == b.modelRef(op.prepare.Engine, op.prepare.Model) {
			loaded = true
		}
	}
	p := nodepolicy.ObserveResidentParams{ReservationID: finish.ReservationID, ProxyRunID: op.prepare.ProxyRunID, OperationID: op.prepare.OperationID, RecoveryID: finish.RecoveryID, Sequence: 1, OperationComplete: true, Observation: nodepolicy.ResidentCapacityObservation{ManagerIncarnation: op.prepare.ManagerIncarnation, Engine: op.prepare.Engine, Model: op.prepare.Model, Identity: op.prepare.Identity, Loaded: loaded}}
	op.recovery = &p
	var result nodepolicy.ObserveResidentResult
	return residentRPC(ctx, op.evidence.Proxy, nodepolicy.MethodObserveResident, p, &result) && result.ReservationID == finish.ReservationID && result.Settled
}

// The request is strictly decoded before leaving the manager reader pump.
func (b *Broker) dispatchResidentFrom(source *engineManagerSource, raw json.RawMessage) {
	var request struct {
		ID     string                            `json:"id"`
		Params nodepolicy.DispatchResidentParams `json:"params"`
	}
	if !residentEnvelope(raw) {
		return
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&request) != nil || request.ID == "" || len(request.ID) > 128 {
		return
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	op := r.models.active
	if op == nil || op.evidence.Source != source || op.grantStarted || op.reservation == "" {
		r.mu.Unlock()
		return
	}
	p := request.Params
	want := nodepolicy.DispatchResidentParams{ReservationID: op.reservation, ProxyRunID: op.prepare.ProxyRunID, ManagerIncarnation: op.prepare.ManagerIncarnation, OperationID: op.prepare.OperationID, PolicyRevision: op.prepare.PolicyRevision, Engine: op.prepare.Engine, Model: op.prepare.Model, Identity: op.prepare.Identity}
	if !reflect.DeepEqual(p, want) {
		r.mu.Unlock()
		return
	}
	op.grantStarted = true
	r.mu.Unlock()
	go func() {
		result := nodepolicy.DispatchResidentResult{Reason: nodepolicy.DispatchReasonReservationRevoked}
		if b.residentCurrent(op) {
			ctx, cancel := context.WithDeadline(op.ctx, op.deadline)
			defer cancel()
			var grant nodepolicy.DispatchResidentResult
			if residentRPC(ctx, op.evidence.Proxy, nodepolicy.MethodDispatchResident, p, &grant) && b.residentCurrent(op) {
				result = grant
			}
		}
		r.mu.Lock()
		op.granted = result.Dispatched
		r.mu.Unlock()
		b.replyPeerRelay(source, op.evidence.Worker, residentDispatchReplyMethod, struct {
			ID     string                            `json:"id"`
			Result nodepolicy.DispatchResidentResult `json:"result"`
		}{request.ID, result})
	}()
}

func residentEnvelope(raw json.RawMessage) bool {
	raw = bytes.TrimSpace(raw)
	d := json.NewDecoder(bytes.NewReader(raw))
	start, err := d.Token()
	if err != nil || start != json.Delim('{') {
		return false
	}
	seen := map[string]bool{}
	for d.More() {
		key, err := d.Token()
		if err != nil || (key != "id" && key != "params") {
			return false
		}
		name, ok := key.(string)
		if !ok || seen[name] {
			return false
		}
		seen[name] = true
		var value json.RawMessage
		if d.Decode(&value) != nil || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return false
		}
	}
	if _, err := d.Token(); err != nil {
		return false
	}
	return len(seen) == 2 && !d.More() && d.InputOffset() == int64(len(bytes.TrimSpace(raw)))
}
