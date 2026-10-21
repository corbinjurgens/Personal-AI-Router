// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"fmt"
	"reflect"
	"time"

	"nvpair-shared/nodepolicy"
)

// Released records may retire at this bound. The ordered sequence watermark
// permanently rejects retired IDs without retaining lifetime history.
const maxResidentOperations = 4096

type residentAdmissionState struct {
	ops map[string]*residentHold
	// Manager generations are source-global. Each dispatched hold snapshots
	// its own floor, so unrelated completions may arrive out of order.
	generations     map[string]uint64
	latestSource    string
	highestSequence uint64
	generation      uint64
}

type residentHold struct {
	request      nodepolicy.PrepareResidentParams
	sequence     uint64
	result       nodepolicy.PrepareResidentResult
	key          admKey
	identity     nodepolicy.EngineIdentity
	generation   uint64
	memory       int64
	context      int
	deadline     time.Time
	counted      bool
	dispatched   bool
	revoked      bool
	observation  uint64
	finish       *nodepolicy.FinishResidentParams
	finishResult nodepolicy.FinishResidentResult
	recovery     *nodepolicy.ObserveResidentParams
}

func (a *admissionController) residentPermissionLocked(k admKey) nodepolicy.ModelConfigurationReason {
	if a.availability != nodepolicy.Available {
		return nodepolicy.ModelReasonPausedLocal
	}
	if a.drain[k.engine] {
		return nodepolicy.ModelReasonSwitchBusy
	}
	if !a.intent[k.engine] || !a.healthy(k.engine) {
		return nodepolicy.ModelReasonEngineOff
	}
	if !modelServed(a.policy.ServedModels, k.engine, k.model, a.normalize) {
		return nodepolicy.ModelReasonNotServed
	}
	desired := false
	for _, ref := range a.policy.ResidentModels {
		if ref.Engine == k.engine && a.normalize(ref.Engine, ref.Model) == k.model {
			desired = true
		}
	}
	if !desired {
		return nodepolicy.ModelReasonNotServed
	}
	id, known := a.strict.identity[k.engine]
	if !known {
		return nodepolicy.ModelReasonUnqualified
	}
	installed := 0
	for model := range id.ModelDigests {
		if a.normalize(k.engine, model) == k.model {
			installed++
		}
	}
	if installed == 0 {
		return nodepolicy.ModelReasonNotInstalled
	}
	if installed != 1 {
		return nodepolicy.ModelReasonUnqualified
	}
	if _, ok := a.strict.models[k]; !ok || a.strict.storageErr != nil {
		return nodepolicy.ModelReasonUnqualified
	}
	if _, ok := a.residentContextLocked(k); !ok {
		return nodepolicy.ModelReasonUnqualified
	}
	if _, unloading := a.unloadSentAt[k]; unloading {
		return nodepolicy.ModelReasonSwitchBusy
	}
	return ""
}

// Resolve a unique normalized profile against the current local qualification.
// Caller holds the admission mutex; a global protocol ceiling is not a bound.
func (a *admissionController) residentContextLocked(k admKey) (int, bool) {
	policy := a.policy
	policy.Profiles = append([]nodepolicy.Profile(nil), policy.Profiles...)
	count := 0
	for i := range policy.Profiles {
		pr := &policy.Profiles[i]
		pr.Model = a.normalize(pr.Engine, pr.Model)
		if pr.Engine == k.engine && pr.Model == k.model {
			count++
		}
	}
	pr, ok := policy.ProfileFor(k.engine, k.model)
	if !ok || count != 1 || pr.Qualification == nil {
		return 0, false
	}
	m, q, ok := a.qualifiedModelLocked(k, pr)
	if !ok || nodepolicy.ValidateResidentContext(pr.ContextTokens, q.ContextTokens) != nil {
		return 0, false
	}
	id := a.strict.identity[k.engine]
	if id.Serving != nil {
		envelope, ok := a.servingEnvelopeLocked(id, k, m.Identity.ModelDigest)
		// A model-only load cannot shrink native allocation. Qualification
		// must cover the actual allocation, not just smaller request bounds.
		if !ok || int64(pr.ContextTokens) != envelope.PerSlotContextTokens ||
			int64(q.ContextTokens) < envelope.PerSlotContextTokens || int64(q.MaxOutputTokens) < envelope.MaxOutputTokens ||
			int64(m.MaxConcurrent) != envelope.Slots {
			return 0, false
		}
	}
	return pr.ContextTokens, true
}

// Validate the whole desired set before reserving any member. Use the same
// resident set and memory owner as inference; residency never plans eviction.
func (a *admissionController) residentBatchLocked() nodepolicy.ModelConfigurationReason {
	set := a.residentLocked()
	desired := map[admKey]bool{}
	for _, ref := range a.policy.ResidentModels {
		k := admKey{engine: ref.Engine, model: a.normalize(ref.Engine, ref.Model)}
		if desired[k] || !a.validStrictModelKey(k) {
			return nodepolicy.ModelReasonUnqualified
		}
		desired[k] = true
		if reason := a.residentPermissionLocked(k); reason != "" {
			return reason
		}
		set[k] = true
	}
	if max := a.policy.Admission.MaxResidentModels; max > 0 && len(set) > max {
		return nodepolicy.ModelReasonNoFit
	}
	if budget := a.policy.Admission.MemoryBudgetBytes; budget > 0 {
		remaining := budget
		for k := range set {
			memory := a.memoryLocked(k)
			if memory <= 0 || memory > remaining {
				return nodepolicy.ModelReasonNoFit
			}
			remaining -= memory
		}
	}
	return ""
}

func (a *admissionController) prepareResident(run string, in nodepolicy.PrepareResidentParams) (nodepolicy.PrepareResidentResult, error) {
	if err := in.Validate(); err != nil {
		return nodepolicy.PrepareResidentResult{}, err
	}
	sequence, err := nodepolicy.ParseResidentOperationID(in.OperationID)
	if err != nil {
		return nodepolicy.PrepareResidentResult{}, err
	}
	a.mu.Lock()
	defer func() { st := a.noteChangeLocked(); a.mu.Unlock(); a.sendState(st) }()
	if in.ProxyRunID != run {
		return nodepolicy.PrepareResidentResult{}, errors.New("resident proxy run conflicts")
	}
	a.invalidateResidentHoldsLocked()
	if a.resident.ops == nil {
		a.resident.ops = map[string]*residentHold{}
		a.resident.generations = map[string]uint64{}
	}
	if old := a.resident.ops[in.OperationID]; old != nil {
		if old.request != in {
			return nodepolicy.PrepareResidentResult{}, errors.New("resident operation conflicts")
		}
		return old.result, nil
	}
	if sequence <= a.resident.highestSequence {
		return nodepolicy.PrepareResidentResult{}, errors.New("resident operation retired")
	}
	if a.resident.highestSequence == ^uint64(0) || sequence != a.resident.highestSequence+1 {
		return nodepolicy.PrepareResidentResult{}, errors.New("resident operation sequence gap")
	}
	// Consume every valid next evaluation, including refusals. No retry can
	// turn a retired or unrecorded refusal into fresh permission later.
	a.resident.highestSequence = sequence
	a.retireResidentOperationsLocked()
	a.resident.generation++
	result := nodepolicy.PrepareResidentResult{EvidenceGeneration: a.resident.generation}
	k := admKey{engine: in.Engine, model: a.normalize(in.Engine, in.Model)}
	reason := nodepolicy.ModelConfigurationReason("")
	switch {
	case in.PolicyRevision != a.policy.Revision || in.ManagerIncarnation == "":
		reason = nodepolicy.ModelReasonUnqualified
	case in.Model != k.model:
		reason = nodepolicy.ModelReasonUnqualified
	case len(a.resident.ops) >= maxResidentOperations:
		reason = nodepolicy.ModelReasonNoFit
	default:
		reason = a.residentPermissionLocked(k)
		if reason == "" && a.strict.models[k].Identity != in.Identity {
			reason = nodepolicy.ModelReasonUnqualified
		}
		if reason == "" {
			reason = a.residentBatchLocked()
		}
		if reason == "" && a.inflight[k] > 0 {
			reason = nodepolicy.ModelReasonNoFit
		}
	}
	budget := time.Duration(in.TimeoutMs) * time.Millisecond
	switchBudget := time.Duration(a.policy.Admission.SwitchTimeoutSeconds) * time.Second
	if switchBudget <= 0 {
		reason = nodepolicy.ModelReasonOperationTimeout
	} else if switchBudget < budget {
		budget = switchBudget
	}
	hold := &residentHold{request: in, sequence: sequence, result: result, key: k, generation: a.strict.generation[k], memory: a.memoryLocked(k), deadline: a.strict.now().Add(budget)}
	if reason != "" {
		hold.result.Refusals = []nodepolicy.ResidentRefusal{{Engine: in.Engine, Model: in.Model, Reason: reason}}
	} else {
		hold.context, _ = a.residentContextLocked(k)
		id := a.strict.identity[in.Engine]
		hold.identity = id
		hold.identity.ModelDigests = make(map[string]string, len(id.ModelDigests))
		for name, digest := range id.ModelDigests {
			hold.identity.ModelDigests[name] = digest
		}
		hold.result.ReservationID = fmt.Sprintf("resident-%d", result.EvidenceGeneration)
		hold.counted = true
		a.inflight[k]++
		a.active++
		a.activeByEngine[k.engine]++
		// This is bookkeeping for the broker-supplied source, not an
		// independent proxy authority for manager incarnation.
		a.resident.latestSource = in.ManagerIncarnation
		if _, known := a.resident.generations[in.ManagerIncarnation]; !known {
			a.resident.generations[in.ManagerIncarnation] = 0
		}
	}
	if len(a.resident.ops) < maxResidentOperations {
		a.resident.ops[in.OperationID] = hold
	}
	a.trimResidentSourcesLocked()
	a.signalLocked()
	return hold.result, nil
}

func (a *admissionController) residentBindingCurrentLocked(h *residentHold) bool {
	context, ok := a.residentContextLocked(h.key)
	return ok && context == h.context && h.request.PolicyRevision == a.policy.Revision && a.strict.generation[h.key] == h.generation &&
		reflect.DeepEqual(h.identity, a.strict.identity[h.key.engine]) && a.residentPermissionLocked(h.key) == ""
}

// Every state-change signal runs this while holding the admission mutex. A
// dispatched operation can lose permission, but cannot lose its capacity.
func (a *admissionController) invalidateResidentHoldsLocked() bool {
	released := false
	for _, h := range a.resident.ops {
		if !h.counted || h.revoked {
			continue
		}
		if !a.strict.now().Before(h.deadline) || !a.residentBindingCurrentLocked(h) {
			h.revoked = true
			if !h.dispatched {
				a.releaseResidentLocked(h)
				released = true
			}
		}
	}
	return released
}

func (a *admissionController) dispatchResident(run string, in nodepolicy.DispatchResidentParams) (nodepolicy.DispatchResidentResult, error) {
	if err := in.Validate(); err != nil {
		return nodepolicy.DispatchResidentResult{}, err
	}
	a.mu.Lock()
	defer func() { st := a.noteChangeLocked(); a.mu.Unlock(); a.sendState(st) }()
	a.invalidateResidentHoldsLocked()
	h := a.resident.ops[in.OperationID]
	if in.ProxyRunID != run || h == nil || h.result.ReservationID != in.ReservationID || h.request.ProxyRunID != in.ProxyRunID {
		return nodepolicy.DispatchResidentResult{Reason: nodepolicy.DispatchReasonReservationUnknown}, nil
	}
	p := h.request
	if in.ManagerIncarnation != p.ManagerIncarnation || in.PolicyRevision != p.PolicyRevision || in.Engine != p.Engine || in.Model != p.Model || in.Identity != p.Identity {
		return nodepolicy.DispatchResidentResult{Reason: nodepolicy.DispatchReasonReservationRevoked}, nil
	}
	if !h.counted || h.revoked || h.finish != nil {
		return nodepolicy.DispatchResidentResult{Reason: nodepolicy.DispatchReasonReservationRevoked}, nil
	}
	if h.dispatched {
		// One operation gets one final grant. A retry cannot authorize a
		// second engine call after an uncertain response.
		return nodepolicy.DispatchResidentResult{Reason: nodepolicy.DispatchReasonReservationRevoked}, nil
	}
	if reason := a.residentBatchLocked(); reason != "" {
		h.revoked = true
		a.releaseResidentLocked(h)
		a.signalLocked()
		return nodepolicy.DispatchResidentResult{Reason: nodepolicy.DispatchResidentReason(reason)}, nil
	}
	h.observation = a.resident.generations[p.ManagerIncarnation]
	h.dispatched = true
	return nodepolicy.DispatchResidentResult{Dispatched: true}, nil
}

func (a *admissionController) finishResident(run string, in nodepolicy.FinishResidentParams) (nodepolicy.FinishResidentResult, error) {
	if err := in.Validate(); err != nil {
		return nodepolicy.FinishResidentResult{}, err
	}
	a.mu.Lock()
	defer func() { st := a.noteChangeLocked(); a.mu.Unlock(); a.sendState(st) }()
	h, err := a.residentReservationLocked(run, in.ProxyRunID, in.OperationID, in.ReservationID)
	if err != nil {
		return nodepolicy.FinishResidentResult{}, err
	}
	if h.finish != nil {
		if !reflect.DeepEqual(*h.finish, in) {
			return nodepolicy.FinishResidentResult{}, errors.New("resident finish conflicts")
		}
		return h.finishResult, nil
	}
	if o := in.Observation; o != nil {
		// Only the matching post-dispatch readback is proof. Legacy loaded
		// notifications, old manager sources and old generations are not.
		if !h.dispatched || o.ManagerIncarnation != h.request.ManagerIncarnation || o.Engine != h.key.engine || o.Model != h.key.model || o.Identity != h.request.Identity ||
			o.Generation == 0 || o.Generation <= h.observation ||
			!reflect.DeepEqual(h.identity, a.strict.identity[h.key.engine]) {
			return nodepolicy.FinishResidentResult{}, errors.New("resident observation is stale or unrelated")
		}
		if o.Generation > a.resident.generations[o.ManagerIncarnation] {
			a.resident.generations[o.ManagerIncarnation] = o.Generation
		}
		a.settleResidentLocked(h, o.Loaded)
	} else if !h.dispatched {
		a.releaseResidentLocked(h)
	}
	// Even not-dispatched after a final grant cannot prove that the engine
	// was untouched: the grant response or subsequent result may have been lost.
	copy := in
	if in.Observation != nil {
		o := *in.Observation
		copy.Observation = &o
	}
	h.finish = &copy
	h.finishResult = nodepolicy.FinishResidentResult{ReservationID: in.ReservationID, Outcome: in.Outcome}
	if h.counted && h.dispatched {
		h.finishResult.RecoveryID = fmt.Sprintf("recovery-%d", h.sequence)
	}
	a.signalLocked()
	return h.finishResult, nil
}

func (a *admissionController) residentReservationLocked(run, proxyRun, operation, reservation string) (*residentHold, error) {
	if proxyRun != run {
		return nil, errors.New("resident proxy run conflicts")
	}
	h := a.resident.ops[operation]
	if h == nil {
		if sequence, err := nodepolicy.ParseResidentOperationID(operation); err == nil && sequence <= a.resident.highestSequence {
			return nil, errors.New("resident operation retired")
		}
		return nil, errors.New("resident reservation unknown")
	}
	if h.result.ReservationID != reservation || h.request.ProxyRunID != proxyRun {
		return nil, errors.New("resident reservation conflicts")
	}
	return h, nil
}

func (a *admissionController) observeResident(run string, in nodepolicy.ObserveResidentParams) (nodepolicy.ObserveResidentResult, error) {
	if err := in.Validate(); err != nil {
		return nodepolicy.ObserveResidentResult{}, err
	}
	a.mu.Lock()
	defer func() { st := a.noteChangeLocked(); a.mu.Unlock(); a.sendState(st) }()
	h, err := a.residentReservationLocked(run, in.ProxyRunID, in.OperationID, in.ReservationID)
	if err != nil {
		return nodepolicy.ObserveResidentResult{}, err
	}
	result := nodepolicy.ObserveResidentResult{ReservationID: in.ReservationID, Settled: true}
	if h.recovery != nil {
		if *h.recovery != in {
			return nodepolicy.ObserveResidentResult{}, errors.New("resident recovery conflicts")
		}
		return result, nil
	}
	o := in.Observation
	if h.finish == nil || !h.dispatched || !h.counted || h.finishResult.RecoveryID == "" || h.finishResult.RecoveryID != in.RecoveryID ||
		o.ManagerIncarnation != h.request.ManagerIncarnation || o.Engine != h.key.engine || o.Model != h.key.model || o.Identity != h.request.Identity ||
		!reflect.DeepEqual(h.identity, a.strict.identity[h.key.engine]) {
		return nodepolicy.ObserveResidentResult{}, errors.New("resident recovery is stale or unrelated")
	}
	// OperationComplete and the per-RecoveryID sequence are supplied by SR5
	// only after its completion barrier and fresh read. Legacy reports cannot
	// assert this ordering. The first accepted observation settles this hold;
	// exact duplicates retain its receipt, and changes cannot settle it twice.
	a.settleResidentLocked(h, o.Loaded)
	copy := in
	h.recovery = &copy
	a.signalLocked()
	return result, nil
}

func (a *admissionController) settleResidentLocked(h *residentHold, loaded bool) {
	// Settled observations rejoin ordinary inventory; no permanent overlay.
	if loaded {
		if a.loaded[h.key.engine] == nil {
			a.loaded[h.key.engine] = map[string]bool{}
		}
		a.loaded[h.key.engine][h.key.model] = true
		a.residencyKnown = true
	} else {
		delete(a.loaded[h.key.engine], h.key.model)
		delete(a.finishedAt, h.key)
	}
	a.releaseResidentLocked(h)
}

func (a *admissionController) retireResidentOperationsLocked() {
	for len(a.resident.ops) >= maxResidentOperations {
		var oldest *residentHold
		for _, h := range a.resident.ops {
			if !h.counted && (oldest == nil || h.sequence < oldest.sequence) {
				oldest = h
			}
		}
		if oldest == nil {
			break
		}
		delete(a.resident.ops, oldest.request.OperationID)
	}
}

func (a *admissionController) trimResidentSourcesLocked() {
	// Retain the latest broker-supplied floor even with no live operations,
	// plus every older source whose unresolved capacity still depends on it.
	used := map[string]bool{a.resident.latestSource: true}
	for _, h := range a.resident.ops {
		if h.counted {
			used[h.request.ManagerIncarnation] = true
		}
	}
	for source := range a.resident.generations {
		if !used[source] {
			delete(a.resident.generations, source)
		}
	}
}

func (a *admissionController) releaseResidentLocked(h *residentHold) {
	if !h.counted {
		return
	}
	h.counted = false
	a.inflight[h.key]--
	if a.inflight[h.key] == 0 {
		delete(a.inflight, h.key)
	}
	a.active--
	a.activeByEngine[h.key.engine]--
	if a.activeByEngine[h.key.engine] == 0 {
		delete(a.activeByEngine, h.key.engine)
	}
	a.trimResidentSourcesLocked()
}

func (a *admissionController) protectedResidentLocked(k admKey) bool {
	return a.residentPermissionLocked(k) == ""
}

func (a *admissionController) engineHasProtectedResidentLocked(engine string) bool {
	for _, h := range a.resident.ops {
		if h.counted && h.key.engine == engine {
			return true
		}
	}
	for k := range a.residentLocked() {
		if k.engine == engine && a.protectedResidentLocked(k) {
			return true
		}
	}
	return false
}
