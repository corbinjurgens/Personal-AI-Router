// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"

	"nvpair-shared/buildapply"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

// Admission is local, before namespace forwarding. Maintenance exclusion also
// belongs at the settings owner, including internal preview/apply callers.
func (b *Broker) handleBuildApply(msg *Message) {
	var request any
	switch msg.Method {
	case "build:apply-impact":
		request = &buildapply.ImpactRequest{}
	case "build:apply-prepare":
		request = &buildapply.PrepareRequest{}
	case "build:apply-finish":
		request = &buildapply.FinishRequest{}
	}
	if err := buildapply.DecodeRequest(msg.Params, request); err != nil {
		_ = b.codec.RespondError(msg.ID, -32602, err.Error())
		return
	}
	if msg.Method == "build:apply-impact" {
		impact, ok := b.cachedBuildApplyImpact()
		if !ok {
			_ = b.codec.Respond(msg.ID, buildapply.Unavailable[buildapply.Impact]())
			return
		}
		_ = b.codec.Respond(msg.ID, buildapply.Result[buildapply.Impact]{Available: true, Value: &impact})
	} else if prepare, ok := request.(*buildapply.PrepareRequest); ok {
		go func() { _ = b.codec.Respond(msg.ID, b.prepareBuildApply(*prepare)) }()
	} else if finish, ok := request.(*buildapply.FinishRequest); ok {
		go func() { _ = b.codec.Respond(msg.ID, b.finishBuildApply(*finish)) }()
	}
}

// WithLocked supplies a detached capture and runs the callback while every
// contributing external owner is pinned against mutation. The caller already
// holds engineConfigMu -> policy mu; implementations must not reenter either
// lock or perform transport I/O, and must return the callback's result without
// failing after an accepted callback.
// There is deliberately no production implementation: manager intent/lock
// revisions, complete roots/inventory, scoped stop and copy/download reservation
// evidence still lack safe read APIs. A cache or a worker probe cannot fill in.
type buildApplyOwners interface {
	WithLocked(func(buildApplyCapturedOwners) error) error
}

type buildApplyCapturedOwners struct {
	Impact                        buildapply.Impact
	Saved, Enforced, PhysicalStop buildapply.Observation[bool]
}

type buildApplyMaintenance struct {
	generation uint64
	active     *buildApplyReservation
	last       *buildApplyReservation
	settings   map[*buildApplySettingsLease]struct{}
}

type buildApplyReservation struct {
	request   buildapply.PrepareRequest
	expected  buildapply.ExpectedSettings
	operation buildapply.BuildApplyOperation
	preparing bool
}

type buildApplyBusyError struct{}

func (*buildApplyBusyError) Error() string {
	return "Build apply and model settings maintenance are mutually exclusive."
}
func (*buildApplyBusyError) Code() int { return buildapply.CodeMaintenanceBusy }

// Reserve before previewEngineSettings/applyEngineSettings enter either their
// ordinary or model-change branch; defer Release across their entire call,
// including native work with engineConfigMu released. A client-only guard
// cannot protect internal callers.
type buildApplySettingsLease struct{ broker *Broker }

func (b *Broker) reserveBuildApplySettings() (*buildApplySettingsLease, error) {
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if b.buildApplyMaintenance.active != nil {
		return nil, &buildApplyBusyError{}
	}
	lease := &buildApplySettingsLease{broker: b}
	if b.buildApplyMaintenance.settings == nil {
		b.buildApplyMaintenance.settings = make(map[*buildApplySettingsLease]struct{})
	}
	b.buildApplyMaintenance.settings[lease] = struct{}{}
	b.buildApplyMaintenance.generation++
	return lease, nil
}

func (lease *buildApplySettingsLease) Release() {
	b := lease.broker
	b.engineConfigMu.Lock()
	if _, exists := b.buildApplyMaintenance.settings[lease]; exists {
		delete(b.buildApplyMaintenance.settings, lease)
		b.buildApplyMaintenance.generation++
	}
	b.engineConfigMu.Unlock()
}

type buildApplyRefusalError struct{ code, detail string }

func (e *buildApplyRefusalError) Error() string { return e.detail }

func buildApplyRefused(code, detail string) buildapply.Result[buildapply.BuildApplyOperation] {
	return buildapply.Result[buildapply.BuildApplyOperation]{Refusal: &buildapply.Refusal{Code: code, Detail: detail}}
}

func buildApplyOwnerRefusal(err error) buildapply.Result[buildapply.BuildApplyOperation] {
	var refusal *buildApplyRefusalError
	if errors.As(err, &refusal) {
		return buildApplyRefused(refusal.code, refusal.detail)
	}
	// Do not export arbitrary owner errors or settings contents.
	return buildapply.Unavailable[buildapply.BuildApplyOperation]()
}

func buildApplyOperationResult(op buildapply.BuildApplyOperation) buildapply.Result[buildapply.BuildApplyOperation] {
	return buildapply.Result[buildapply.BuildApplyOperation]{Available: true, Value: &op}
}

func buildApplyTrue(value buildapply.Observation[bool]) bool {
	return value.Known && value.Value != nil && *value.Value
}

func buildApplyFencesKnown(f buildapply.Fences) bool {
	for _, value := range []buildapply.Observation[string]{f.RuntimeGeneration, f.ServiceInstance, f.BrokerGeneration, f.ManagerIncarnation, f.PolicyRevision, f.SettingsRevision, f.IntentRevision, f.LocksRevision, f.AvailabilityRevision} {
		if !value.Known || value.Value == nil || *value.Value == "" {
			return false
		}
	}
	return true
}

// Pause legitimately changes policy, availability and runtime observations.
// It never grants permission to cross a manager lifetime or a new settings,
// intent or lock revision, even if the new values have equal semantic digests.
func buildApplyStableFences(before, after buildapply.Fences) bool {
	return reflect.DeepEqual(before.ServiceInstance, after.ServiceInstance) &&
		reflect.DeepEqual(before.BrokerGeneration, after.BrokerGeneration) &&
		reflect.DeepEqual(before.ManagerIncarnation, after.ManagerIncarnation) &&
		reflect.DeepEqual(before.SettingsRevision, after.SettingsRevision) &&
		reflect.DeepEqual(before.IntentRevision, after.IntentRevision) &&
		reflect.DeepEqual(before.LocksRevision, after.LocksRevision)
}

func buildApplyStoppedEvidence(c buildApplyCapturedOwners) bool {
	if !buildApplyTrue(c.PhysicalStop) || !c.Impact.Availability.Known || c.Impact.Availability.Value == nil || *c.Impact.Availability.Value != nodepolicy.Paused ||
		!c.Impact.Engines.Known || c.Impact.Engines.Value == nil {
		return false
	}
	for _, engine := range *c.Impact.Engines.Value {
		if !engine.Running.Known || engine.Running.Value == nil || *engine.Running.Value ||
			!engine.LoadedModels.Known || engine.LoadedModels.Value == nil || len(*engine.LoadedModels.Value) != 0 {
			return false
		}
	}
	return true
}

// Complete is necessary but never substitutes for the required individual
// facts. In particular, stopped telemetry does not establish scoped authority.
func buildApplyEvidenceComplete(c buildApplyCapturedOwners) bool {
	i := c.Impact
	if !i.Complete || len(i.Refusals) != 0 || !buildApplyFencesKnown(i.Fences) ||
		!i.Availability.Known || i.Availability.Value == nil || !i.PauseOrigin.Known || i.PauseOrigin.Value == nil ||
		!i.Engines.Known || i.Engines.Value == nil || !i.ActiveWork.Known || i.ActiveWork.Value == nil ||
		*i.ActiveWork.Value < 0 || !i.ModelOperations.Known || i.ModelOperations.Value == nil || !i.SettingsDifferences.Known || i.SettingsDifferences.Value == nil ||
		!i.InterruptionPolicy.OnActive.Known || i.InterruptionPolicy.OnActive.Value == nil ||
		!i.InterruptionPolicy.DrainTimeoutSeconds.Known || i.InterruptionPolicy.DrainTimeoutSeconds.Value == nil ||
		!i.Maintenance.BuildApplyActive.Known || i.Maintenance.BuildApplyActive.Value == nil ||
		!i.Maintenance.ModelSettingsActive.Known || i.Maintenance.ModelSettingsActive.Value == nil || *i.Maintenance.ModelSettingsActive.Value ||
		!buildApplyTrue(i.ExpectedSettings.StopEngines) || !buildApplyTrue(i.ExpectedSettings.UnloadModels) {
		return false
	}
	for _, digest := range []buildapply.Observation[string]{i.ExpectedSettings.PolicyDigest, i.ExpectedSettings.EngineSettingsDigest, i.ExpectedSettings.IntentDigest, i.ExpectedSettings.LocksDigest} {
		if !digest.Known || digest.Value == nil || *digest.Value == "" {
			return false
		}
	}
	for _, engine := range *i.Engines.Value {
		if !buildApplyTrue(engine.ScopedStopAuthority) || !engine.Running.Known || engine.Running.Value == nil ||
			!engine.LoadedModels.Known || engine.LoadedModels.Value == nil || !engine.Locked.Known || engine.Locked.Value == nil ||
			!engine.DesiredState.Known || engine.DesiredState.Value == nil || !engine.Active.Known || engine.Active.Value == nil {
			return false
		}
	}
	for _, op := range *i.ModelOperations.Value {
		if op.BlocksApply {
			return false
		}
	}
	return true
}

func (b *Broker) prepareBuildApply(request buildapply.PrepareRequest) buildapply.Result[buildapply.BuildApplyOperation] {
	// No owner initialization, persistence or Pause acceptance before consent.
	if !request.Consent.AllowEngineStop || !request.Consent.AllowInterruption {
		return buildApplyRefused("interruption_required", "Applying a build requires explicit interruption and engine-stop consent.")
	}
	if request.PreviewID == "" {
		return buildApplyRefused("invalid_request", "A service-validated preview ID is required.")
	}
	b.engineConfigMu.Lock()
	m := &b.buildApplyMaintenance
	if active := m.active; active != nil {
		if reflect.DeepEqual(active.request, request) {
			result := buildApplyOperationResult(active.operation)
			b.engineConfigMu.Unlock()
			return result
		}
		b.engineConfigMu.Unlock()
		return buildapply.MaintenanceBusy[buildapply.BuildApplyOperation]()
	}
	if m.last != nil && m.last.request.OperationID == request.OperationID {
		last := m.last
		b.engineConfigMu.Unlock()
		if reflect.DeepEqual(last.request, request) {
			return buildApplyOperationResult(last.operation)
		}
		return buildApplyRefused("operation_conflict", "The operation ID was already used with different facts.")
	}
	if len(m.settings) != 0 {
		b.engineConfigMu.Unlock()
		return buildapply.MaintenanceBusy[buildapply.BuildApplyOperation]()
	}
	// Production remains refused before even initializing policy. The owner
	// evidence task must land before wiring this.
	if b.buildApplyOwners == nil {
		b.engineConfigMu.Unlock()
		return buildapply.Unavailable[buildapply.BuildApplyOperation]()
	}
	r := &b.nodePolicy
	r.mu.Lock()
	var accepted acceptedAvailability
	var reservation *buildApplyReservation
	err := b.buildApplyOwners.WithLocked(func(c buildApplyCapturedOwners) error {
		if !r.ready || r.recoveryRequired || r.pendingPauseRequested || r.live == nodepolicy.Draining || !buildApplyEvidenceComplete(c) {
			return &buildApplyRefusalError{"owner_evidence_unavailable", "Complete current owner evidence is required before accepting Pause."}
		}
		if !reflect.DeepEqual(request.Fences, c.Impact.Fences) || *c.Impact.Fences.PolicyRevision.Value != r.current.Revision ||
			*c.Impact.ExpectedSettings.PolicyDigest.Value != modelPolicyFingerprint(r.current) {
			return &buildApplyRefusalError{"changed", "Owner fences changed; obtain a new preview."}
		}
		if *c.Impact.Maintenance.BuildApplyActive.Value {
			return &buildApplyRefusalError{buildapply.RefusalMaintenanceBusy, "Another build operation holds maintenance."}
		}
		for _, record := range b.engineSettings {
			if record != nil && record.Snapshot.Adopted {
				return &buildApplyRefusalError{"unscoped_engine", "An adopted engine has no verified scoped stop authority for build apply."}
			}
		}
		op := buildapply.BuildApplyOperation{OperationID: request.OperationID, PreviewID: request.PreviewID, Phase: buildapply.PhaseQuiesce, Fences: c.Impact.Fences, ReadbackConfidence: buildapply.ConfidenceIncomplete}
		if _, err := json.Marshal(op); err != nil {
			return &buildApplyRefusalError{"owner_evidence_unavailable", "The owner correlation cannot produce a valid operation reply."}
		}
		origin := nodepolicy.PausedLocally
		if r.current.Availability == nodepolicy.Paused && r.current.PausedBy != "" {
			origin = r.current.PausedBy
		}
		reservation = &buildApplyReservation{request: request, expected: c.Impact.ExpectedSettings, preparing: true, operation: op}
		m.active = reservation
		m.generation++
		r.buildApplyOperation = request.OperationID
		accepted = r.acceptAvailabilityLocked(nodepolicy.Paused, origin, nodepolicy.PauseConfigured)
		return nil
	})
	r.mu.Unlock()
	b.engineConfigMu.Unlock()
	if err != nil {
		return buildApplyOwnerRefusal(err)
	}
	// The transport/caller lifetime cannot abandon accepted maintenance.
	state, pauseErr := b.runAcceptedAvailability(context.Background(), r, accepted)
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	r.mu.Lock()
	defer r.mu.Unlock()
	op := &reservation.operation
	reservation.preparing = false
	op.RemainsPaused = buildapply.Known(r.current.Availability == nodepolicy.Paused && r.live == nodepolicy.Paused)
	readErr := b.buildApplyOwners.WithLocked(func(c buildApplyCapturedOwners) error {
		if !buildApplyEvidenceComplete(c) || !buildApplyTrue(c.Impact.Maintenance.BuildApplyActive) || !buildApplyStableFences(request.Fences, c.Impact.Fences) || !reflect.DeepEqual(c.Impact.ExpectedSettings, reservation.expected) {
			return errors.New("post-pause evidence changed or is incomplete")
		}
		op.Fences = c.Impact.Fences
		op.Saved, op.Enforced, op.PhysicalStop = c.Saved, c.Enforced, c.PhysicalStop
		if !buildApplyStoppedEvidence(c) {
			if buildApplyTrue(op.PhysicalStop) {
				op.PhysicalStop = buildapply.Observation[bool]{}
			}
			return errors.New("physical stop evidence is missing or contradictory")
		}
		return nil
	})
	if pauseErr != nil || state != nodepolicy.Paused || r.transitionGen != accepted.gen || readErr != nil || !buildApplyTrue(op.Saved) || !buildApplyTrue(op.Enforced) || !buildApplyTrue(op.PhysicalStop) || !buildApplyTrue(op.RemainsPaused) {
		outcome := buildapply.OutcomeUncertain
		op.Outcome = &outcome
		op.Failure = &buildapply.BuildApplyFailure{Phase: op.Phase, Reason: "quiesce_unverified", Detail: "Durable Pause, enforcement and physical shutdown were not all verified.", ReadbackConfidence: buildapply.ConfidenceIncomplete, SafeNextAction: "Keep the node paused and explicitly recover this operation."}
	}
	return buildApplyOperationResult(*op)
}

// Finish releases only this broker's reservation. It never restores a saved
// document, resumes engines or acknowledges the service/helper handoff. The
// service owns whole-build readback and durable terminal receipt publication.
// Explicit recovery can settle prior uncertainty using fresh exact fences, but
// only after the canonical owner proves Pause with unchanged protected state.
func (b *Broker) finishBuildApply(request buildapply.FinishRequest) buildapply.Result[buildapply.BuildApplyOperation] {
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	m := &b.buildApplyMaintenance
	reservation := m.active
	if reservation == nil {
		if m.last != nil && m.last.operation.OperationID == request.OperationID && reflect.DeepEqual(m.last.operation.Fences, request.Fences) {
			return buildApplyOperationResult(m.last.operation)
		}
		return buildapply.Unavailable[buildapply.BuildApplyOperation]()
	}
	if request.OperationID != reservation.operation.OperationID || !buildApplyFencesKnown(request.Fences) {
		return buildApplyRefused("operation_conflict", "Finish must match the reserved operation and provide exact current fences.")
	}
	if reservation.preparing {
		return buildapply.MaintenanceBusy[buildapply.BuildApplyOperation]()
	}
	if b.buildApplyOwners == nil {
		return buildapply.Unavailable[buildapply.BuildApplyOperation]()
	}
	r := &b.nodePolicy
	r.mu.Lock()
	defer r.mu.Unlock()
	err := b.buildApplyOwners.WithLocked(func(c buildApplyCapturedOwners) error {
		if r.buildApplyOperation != request.OperationID || !buildApplyEvidenceComplete(c) || !buildApplyTrue(c.Impact.Maintenance.BuildApplyActive) ||
			!reflect.DeepEqual(request.Fences, c.Impact.Fences) || !buildApplyStableFences(reservation.request.Fences, c.Impact.Fences) ||
			!reflect.DeepEqual(reservation.expected, c.Impact.ExpectedSettings) {
			return &buildApplyRefusalError{"changed", "Current owner evidence no longer matches this operation; explicit recovery is required."}
		}
		if r.current.Availability != nodepolicy.Paused || r.live != nodepolicy.Paused || r.pendingPauseRequested || r.recoveryRequired ||
			!buildApplyTrue(c.Saved) || !buildApplyTrue(c.Enforced) || !buildApplyStoppedEvidence(c) {
			return &buildApplyRefusalError{"pause_unverified", "Finish requires a verified durable and physically enforced Pause."}
		}
		reservation.operation.Phase = buildapply.PhaseFinish
		reservation.operation.Fences = c.Impact.Fences
		reservation.operation.Outcome = nil
		reservation.operation.Failure = nil
		reservation.operation.Saved, reservation.operation.Enforced, reservation.operation.PhysicalStop = c.Saved, c.Enforced, c.PhysicalStop
		reservation.operation.RemainsPaused = buildapply.Known(true)
		r.buildApplyOperation = ""
		m.active, m.last = nil, reservation
		m.generation++
		return nil
	})
	if err != nil {
		return buildApplyOwnerRefusal(err)
	}
	return buildApplyOperationResult(reservation.operation)
}

// cachedBuildApplyImpact never initializes an owner, loads a journal, asks a
// worker for engine status, or schedules a policy push. Repeat all local owner
// fences, including those not representable on the public wire.
func (b *Broker) cachedBuildApplyImpact() (buildapply.Impact, bool) {
	return captureCachedBuildApplyImpact(b.readCachedBuildApplyOwners)
}

func captureCachedBuildApplyImpact(read func() (buildApplyOwnerRead, bool)) (buildapply.Impact, bool) {
	first, ok := read()
	if !ok {
		return buildapply.Impact{}, false
	}
	impact := first.impact
	if first.policyPath != "" {
		if differences, ok := readBuildApplyStoredPause(first.policyPath, first.policyDigest); ok {
			impact.SettingsDifferences = buildapply.Known(differences)
		} else {
			impact.Refusals = append(impact.Refusals, buildapply.Refusal{Code: "stored_policy_unavailable", Detail: "The stored policy could not be matched to the decoded policy owner."})
		}
	}
	last, lastOK := read()
	if !lastOK || !sameCachedBuildApplyOwners(first, last) {
		// Do not retain stale known engine/settings observations after any
		// contributing owner changes during this capture.
		return buildapply.Impact{RestorePolicy: impact.RestorePolicy, InterruptionPolicy: buildapply.InterruptionPolicy{StopsEngines: true, UnloadsModels: true, CommittedStreamsMayFail: true, IncludesWorkBeforeFence: true}, Refusals: []buildapply.Refusal{{Code: "changed", Detail: "Local owner evidence changed during capture."}}}, true
	}
	// The current broker cache does not retain lock revisions, per-manager
	// intent provenance, complete installed inventory or download/copy
	// reservations. Those observations stay unknown; never refresh via the
	// engine status methods, which may probe/adopt/launch native engines.
	if !impact.ExpectedSettings.LocksDigest.Known || !impact.ExpectedSettings.IntentDigest.Known || !impact.Fences.LocksRevision.Known || !impact.Maintenance.BuildApplyActive.Known {
		impact.Refusals = append(impact.Refusals, buildapply.Refusal{Code: "owner_evidence_unavailable", Detail: "Current manager lock/intent provenance, complete inventory and download/copy reservations are not cached by this broker."})
	}
	impact.Complete = len(impact.Refusals) == 0 && impact.ActiveWork.Known && impact.Engines.Known && impact.ModelOperations.Known && impact.SettingsDifferences.Known && impact.Fences.ManagerIncarnation.Known && impact.Fences.SettingsRevision.Known && impact.Fences.IntentRevision.Known && impact.Fences.LocksRevision.Known
	return impact, true
}

type buildApplyOwnerRead struct {
	impact                                                                   buildapply.Impact
	worker                                                                   *rpcWorker
	source                                                                   *engineManagerSource
	proxy                                                                    *proxyProcess
	policyProxy                                                              *proxyProcess
	policySource                                                             *engineManagerSource
	policyPath, policyDigest, settingsDigest, operationsDigest, engineDigest string
	policyGeneration, runningGeneration, modelGeneration                     uint64
	maintenanceGeneration                                                    uint64
	recovering                                                               bool
	pendingInstalls                                                          []buildApplyInstallRead
}

type buildApplyInstallRead struct {
	Handle     *pendingInstallStart `json:"-"`
	ID, Engine string
	Superseded bool
}

func sameCachedBuildApplyOwners(a, b buildApplyOwnerRead) bool {
	// Compare live owners by identity, never by reflection into their mutable
	// transports/mutexes. Every other compared value is a detached observation.
	if a.worker != b.worker || a.source != b.source || a.proxy != b.proxy || a.policyProxy != b.policyProxy || a.policySource != b.policySource || a.recovering != b.recovering || a.policyGeneration != b.policyGeneration || a.runningGeneration != b.runningGeneration || a.modelGeneration != b.modelGeneration || a.maintenanceGeneration != b.maintenanceGeneration || a.policyPath != b.policyPath || a.policyDigest != b.policyDigest || a.settingsDigest != b.settingsDigest || a.operationsDigest != b.operationsDigest || a.engineDigest != b.engineDigest || len(a.pendingInstalls) != len(b.pendingInstalls) {
		return false
	}
	for i, install := range a.pendingInstalls {
		if install != b.pendingInstalls[i] {
			return false
		}
	}
	return reflect.DeepEqual(a.impact, b.impact)
}

func (b *Broker) readCachedBuildApplyOwners() (buildApplyOwnerRead, bool) {
	var capture buildApplyOwnerRead
	impact := &capture.impact
	refuse := func(code, detail string) {
		impact.Refusals = append(impact.Refusals, buildapply.Refusal{Code: code, Detail: detail})
	}
	impact.RestorePolicy = buildapply.RestorePolicy{RemainPaused: true, RequireLocalResume: true, PreserveIntentAndLocks: true, PreserveNewerPause: true}
	impact.InterruptionPolicy = buildapply.InterruptionPolicy{StopsEngines: true, UnloadsModels: true, CommittedStreamsMayFail: true, IncludesWorkBeforeFence: true}
	// engineConfigMu is the established settings/switch journal lock. Do not
	// call loadEngineSettingsLocked, switchCoordinator or pendingSwitchLocked.
	if !b.engineConfigMu.TryLock() {
		return buildApplyOwnerRead{}, false
	}
	settingsKnown := b.engineSettingsLoaded && b.engineSettingsError == nil && b.engineSettingsEpoch != ""
	records := make(map[string]*engineSettingsRecord, len(b.engineSettings))
	semanticSettings := map[string]settings.Config{}
	operations := []buildapply.ModelOperation{}
	modelSettingsActive := false
	for engine, record := range b.engineSettings {
		records[engine] = record.clone()
		if record == nil {
			settingsKnown = false
		} else if canonical, err := record.Snapshot.Settings.Canonical(); err != nil {
			settingsKnown = false
		} else {
			semanticSettings[engine] = canonical
		}
		if record != nil {
			if record.Model != nil && record.Model.Stage != "completed" {
				modelSettingsActive = true
				operations = append(operations, buildapply.ModelOperation{Engine: engine, Kind: "model-settings", BlocksApply: true})
			}
			if record.Snapshot.Phase == "applying" || record.Snapshot.Revision != record.Snapshot.AppliedRevision {
				operations = append(operations, buildapply.ModelOperation{Engine: engine, Kind: "settings", BlocksApply: true})
			}
		}
	}
	for _, op := range b.activeSettings {
		modelSettingsActive = true
		operations = append(operations, buildapply.ModelOperation{Engine: op.Engine, Kind: "settings", BlocksApply: true})
	}
	for pending := range b.pendingInstallStarts {
		capture.pendingInstalls = append(capture.pendingInstalls, buildApplyInstallRead{pending, fmt.Sprintf("%p", pending), pending.engine, pending.superseded})
		operations = append(operations, buildapply.ModelOperation{Engine: pending.engine, Kind: "install", BlocksApply: true})
	}
	sort.Slice(capture.pendingInstalls, func(i, j int) bool { return capture.pendingInstalls[i].ID < capture.pendingInstalls[j].ID })
	settingsEpoch := b.engineSettingsEpoch
	capture.maintenanceGeneration = b.buildApplyMaintenance.generation
	impact.Maintenance.BuildApplyActive = buildapply.Known(b.buildApplyMaintenance.active != nil)
	modelSettingsActive = modelSettingsActive || len(b.buildApplyMaintenance.settings) != 0
	if b.buildApplyMaintenance.active != nil || len(b.buildApplyMaintenance.settings) != 0 {
		refuse(buildapply.RefusalMaintenanceBusy, "Build apply or settings maintenance is already reserved.")
	}
	if len(b.buildApplyMaintenance.settings) != 0 {
		impact.Maintenance.ModelSettingsActive = buildapply.Known(true)
	}
	capture.settingsDigest = modelSettingsHash(struct {
		Records  map[string]*engineSettingsRecord
		Active   map[string]activeSettingsOperation
		Installs []buildApplyInstallRead
	}{records, b.activeSettings, capture.pendingInstalls})
	if settingsKnown {
		impact.ExpectedSettings.EngineSettingsDigest = buildapply.Known(modelSettingsHash(semanticSettings))
		impact.Fences.SettingsRevision = buildapply.Known(modelSettingsHash(struct {
			Epoch, State string
		}{settingsEpoch, capture.settingsDigest}))
		impact.Maintenance.ModelSettingsActive = buildapply.Known(modelSettingsActive)
	} else {
		refuse("settings_unavailable", "The settings owner has no complete initialized baseline.")
	}
	c := b.engineSwitch
	if c != nil && c.store != nil && c.store.loaded && c.err == nil && c.recoveryErr == nil && c.store.untrusted == nil && c.store.fenceErr == nil {
		capture.operationsDigest = modelSettingsHash(struct {
			Journal    settings.SwitchJournal
			Checkpoint *settings.SwitchCheckpoint
			Resolved   map[string]bool
		}{c.store.journal, c.recoveryCheckpoint, c.resolved})
		for _, receipt := range c.store.journal.Receipts {
			if !receipt.Terminal || receipt.RecoveryRequired && !c.resolved[receipt.RequestID] {
				operations = append(operations, buildapply.ModelOperation{Engine: receipt.Request.SourceEngine, Kind: "switch", BlocksApply: true})
			}
		}
		if cp := c.recoveryCheckpoint; cp != nil && cp.Phase != settings.SwitchCheckpointReleased {
			operations = append(operations, buildapply.ModelOperation{Engine: cp.SourceEngine, Kind: "switch-recovery", BlocksApply: true})
		}
	} else {
		refuse("switch_unavailable", "The switch/recovery owner has no complete initialized baseline.")
	}
	b.engineConfigMu.Unlock()
	sort.Slice(operations, func(i, j int) bool {
		if operations[i].Engine != operations[j].Engine {
			return operations[i].Engine < operations[j].Engine
		}
		return operations[i].Kind < operations[j].Kind
	})
	if len(operations) > 0 {
		impact.ModelOperations = buildapply.Known(operations)
	}
	if !b.workersMu.TryLock() {
		return buildApplyOwnerRead{}, false
	}
	capture.worker, capture.source, capture.proxy, capture.recovering = b.engineMgr, b.engineMgrSource, b.proxyProc, b.engineMgrRecovering
	b.workersMu.Unlock()
	r := &b.nodePolicy
	if !r.mu.TryLock() {
		return buildApplyOwnerRead{}, false
	}
	if !r.ready {
		r.mu.Unlock()
		return buildApplyOwnerRead{}, false
	}
	capture.policySource, capture.policyProxy = r.engineManagerSource, r.incarnation
	capture.policyPath = r.path
	// Re-parse with the current trusted Go normalization. Nothing is saved.
	policy, parseErr := nodepolicy.Parse(settingsJSON(r.current))
	if parseErr != nil {
		r.mu.Unlock()
		return buildApplyOwnerRead{}, false
	}
	capture.policyDigest = modelSettingsHash(policy)
	capture.policyGeneration, capture.runningGeneration, capture.modelGeneration = r.transitionGen, r.runningGeneration, r.models.sequence
	impact.Fences.PolicyRevision = buildapply.Known(policy.Revision)
	impact.Availability = buildapply.Known(r.live)
	origin := string(policy.PausedBy)
	if policy.Availability == nodepolicy.Paused && origin == "" {
		origin = "local"
	}
	impact.PauseOrigin = buildapply.Known(origin)
	impact.Fences.AvailabilityRevision = buildapply.Known(modelSettingsHash(struct {
		Revision               string
		Generation             uint64
		Live                   nodepolicy.Availability
		Origin                 string
		Local, Pending, Cancel bool
	}{policy.Revision, r.transitionGen, r.live, origin, r.localPauseRequested, r.pendingPauseRequested, r.cancelActive}))
	// Separate semantic expectations from lifetime/revision fences, which
	// legitimately change when the transaction pauses and replaces the broker.
	impact.ExpectedSettings.PolicyDigest = buildapply.Known(modelPolicyFingerprint(policy))
	impact.ExpectedSettings.StopEngines = buildapply.Known(policy.Pause.StopEngines)
	impact.ExpectedSettings.UnloadModels = buildapply.Known(policy.Pause.UnloadModels)
	impact.InterruptionPolicy.OnActive = buildapply.Known(policy.Pause.OnActive)
	impact.InterruptionPolicy.DrainTimeoutSeconds = buildapply.Known(policy.Pause.DrainTimeoutSeconds)
	if r.recoveryRequired {
		refuse("policy_recovery", "The policy owner requires recovery.")
	}
	if r.pendingPauseRequested || r.live == nodepolicy.Draining {
		refuse("availability_busy", "An availability transition is still in progress.")
	}
	if op := r.models.active; op != nil {
		operations = append(operations, buildapply.ModelOperation{Engine: op.prepare.Engine, Kind: "resident-model", BlocksApply: true})
	}
	for engine, waking := range r.waking {
		if waking {
			operations = append(operations, buildapply.ModelOperation{Engine: engine, Kind: "wake", BlocksApply: true})
		}
	}
	for engine, draining := range r.drains {
		if draining {
			operations = append(operations, buildapply.ModelOperation{Engine: engine, Kind: "drain", BlocksApply: true})
		}
	}
	sort.Slice(operations, func(i, j int) bool {
		if operations[i].Engine != operations[j].Engine {
			return operations[i].Engine < operations[j].Engine
		}
		return operations[i].Kind < operations[j].Kind
	})
	if len(operations) > 0 {
		impact.ModelOperations = buildapply.Known(operations)
		refuse("operation_busy", "An observed local settings, install, switch or model operation blocks apply.")
	}
	if r.models.active != nil || r.models.busy {
		refuse("resident_busy", "The resident-model owner has pending work.")
	}
	activation := r.activationLocked(capture.proxy, policy.Revision)
	activationCurrent := activation.Activation == activationActive && activation.ActiveRevision == policy.Revision && r.runID != ""
	if !activationCurrent {
		refuse("activation_unavailable", "Current policy and availability activation are unverified.")
	}
	managerCurrent := capture.worker != nil && capture.source != nil && capture.source == r.engineManagerSource && !capture.recovering
	if managerCurrent {
		// A broker-assigned worker source token is not the manager's own
		// incarnation. Only reuse an actual native acknowledgement bound to
		// this still-current source; never relabel source.settingsEpoch.
		if op := r.models.active; op != nil && op.evidence.Worker == capture.worker && op.evidence.Source == capture.source && op.prepare.ManagerIncarnation != "" {
			impact.Fences.ManagerIncarnation = buildapply.Known(op.prepare.ManagerIncarnation)
		}
	}
	if !impact.Fences.ManagerIncarnation.Known {
		refuse("manager_unavailable", "The manager's own incarnation acknowledgement is missing or mixed.")
	}
	if activationCurrent && r.haveAdmission && r.admission.Active >= 0 && r.admission.Queued >= 0 && r.admission.Active <= int(^uint(0)>>1)-r.admission.Queued {
		impact.ActiveWork = buildapply.Known(r.admission.Active + r.admission.Queued)
	}
	loadedCurrent := managerCurrent && r.models.loadedSource == capture.source && r.models.loadedProxy == capture.proxy
	names := map[string]bool{}
	for name := range r.running {
		names[name] = true
	}
	for name := range r.loaded {
		names[name] = true
	}
	for name := range r.intent {
		names[name] = true
	}
	for name := range records {
		names[name] = true
	}
	facts := []buildapply.EngineFact{}
	orderedNames := make([]string, 0, len(names))
	for name := range names {
		orderedNames = append(orderedNames, name)
	}
	sort.Strings(orderedNames)
	settingsCurrent := settingsKnown && managerCurrent && len(orderedNames) > 0
	for _, name := range orderedNames {
		if !engines.ValidID(name) {
			continue
		}
		fact := buildapply.EngineFact{Engine: name}
		if record := records[name]; record == nil || record.Snapshot.Epoch != settingsEpoch {
			settingsCurrent = false
		} else if fingerprint, ok := settingsFingerprint(record.Snapshot); !ok || r.settingsIdentity[name] != fingerprint {
			settingsCurrent = false
		}
		if managerCurrent {
			if running, known := r.running[name]; known && r.runningObserved[name] > 0 {
				fact.Running = buildapply.Known(running)
			}
			if loaded, known := r.loaded[name]; known && loaded != nil && loadedCurrent {
				models := append([]string{}, loaded...)
				sort.Strings(models)
				fact.LoadedModels = buildapply.Known(models)
			}
			// Worker startup clears intent before binding this source. Only
			// reported entries are known. Off provenance is not retained, so a
			// complete protected intent digest remains unavailable.
			if enabled, known := r.intent[name]; known {
				desired := "off"
				if enabled {
					desired = "on"
				}
				fact.DesiredState = buildapply.Known(desired)
			}
			if record := records[name]; record != nil && record.Snapshot.Adopted {
				refuse("unscoped_engine", "Cached engine "+name+" has no verified scoped stop authority.")
			}
		}
		if activationCurrent && r.haveAdmission {
			if active, known := r.admission.ActiveByEngine[name]; known && active >= 0 {
				fact.Active = buildapply.Known(active > 0)
			}
		}
		facts = append(facts, fact)
	}
	if !settingsCurrent {
		impact.ExpectedSettings.EngineSettingsDigest = buildapply.Observation[string]{}
		refuse("settings_baseline_unavailable", "A current applied settings baseline is missing for an observed engine.")
	}
	sort.Slice(facts, func(i, j int) bool { return facts[i].Engine < facts[j].Engine })
	if len(facts) > 0 {
		impact.Engines = buildapply.Known(facts)
	}
	// Include raw cached intent privately to detect a change, but do not label
	// its hash as a complete protected intent expectation or authoritative
	// intent revision: this cache does not retain the owner revision/provenance.
	capture.engineDigest = modelSettingsHash(struct {
		Running                            map[string]bool
		Observed                           map[string]uint64
		Loaded                             map[string][]string
		Intent                             map[string]bool
		Identity                           map[string]engineManagerIdentity
		SettingsIdentity                   map[string]string
		Drains, Waking                     map[string]bool
		Admission                          nodepolicy.AdmissionState
		HaveAdmission                      bool
		Run, ActiveRevision                string
		Recovery                           bool
		ActivationError, AvailabilityError string
	}{r.running, r.runningObserved, r.loaded, r.intent, r.engineIdentity, r.settingsIdentity, r.drains, r.waking, r.admission, r.haveAdmission, r.runID, r.activeRevision, r.recoveryRequired, r.activationErr, r.availabilityErr})
	if managerCurrent {
		impact.Fences.RuntimeGeneration = buildapply.Known(modelSettingsHash(struct {
			ManagerSource, ProxyRun                 string
			RunningGeneration, TransitionGeneration uint64
			Running                                 map[string]bool
			Loaded                                  map[string][]string
			Identity                                map[string]engineManagerIdentity
			SettingsIdentity                        map[string]string
		}{capture.source.settingsEpoch, r.runID, r.runningGeneration, r.transitionGen, r.running, r.loaded, r.engineIdentity, r.settingsIdentity}))
	}
	r.mu.Unlock()
	if !b.workersMu.TryLock() {
		return buildApplyOwnerRead{}, false
	}
	current := capture.worker == b.engineMgr && capture.source == b.engineMgrSource && capture.proxy == b.proxyProc && capture.recovering == b.engineMgrRecovering
	b.workersMu.Unlock()
	if !current {
		refuse("changed", "Manager or proxy generation changed during capture.")
	}
	return capture, true
}

// Raw pause flags are read only to explain existing normalization. The trusted
// Parse result must equal the current owner, and file identity is fenced. No
// canonical write is claimed or performed by preview.
func readBuildApplyStoredPause(path, digest string) ([]buildapply.SettingsDifference, bool) {
	data, err := buildapply.ReadLocalFile(path, 2<<20)
	if err != nil {
		return nil, false
	}
	policy, err := nodepolicy.Parse(data)
	if err != nil || modelSettingsHash(policy) != digest {
		return nil, false
	}
	var raw struct {
		Pause struct {
			StopEngines  *bool `json:"stopEngines"`
			UnloadModels *bool `json:"unloadModels"`
		} `json:"pause"`
	}
	if json.Unmarshal(data, &raw) != nil {
		return nil, false
	}
	differences := []buildapply.SettingsDifference{}
	for _, field := range []struct {
		name   string
		stored *bool
	}{{"pause.stopEngines", raw.Pause.StopEngines}, {"pause.unloadModels", raw.Pause.UnloadModels}} {
		entry := buildapply.SettingsDifference{Field: field.name, Effective: buildapply.Known(true), Persisted: buildapply.Known(false)}
		if field.stored != nil {
			entry.Stored = buildapply.Known(*field.stored)
			entry.Persisted = buildapply.Known(*field.stored)
		}
		differences = append(differences, entry)
	}
	return differences, true
}

// The broker admits these locally and does not forward them to a worker.
func isBuildApplyMethod(method string) bool {
	switch method {
	case "build:apply-impact", "build:apply-prepare", "build:apply-finish":
		return true
	default:
		return false
	}
}
