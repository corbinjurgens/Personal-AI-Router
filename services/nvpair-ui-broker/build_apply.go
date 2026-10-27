// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"

	"nvpair-shared/buildapply"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

// Admission is local, before namespace forwarding. This is not the settings
// maintenance exclusion: Task 3 must wire it at the authoritative settings owner.
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
	} else {
		_ = b.codec.Respond(msg.ID, buildapply.Unavailable[buildapply.BuildApplyOperation]())
	}
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
	if a.worker != b.worker || a.source != b.source || a.proxy != b.proxy || a.policyProxy != b.policyProxy || a.policySource != b.policySource || a.recovering != b.recovering || a.policyGeneration != b.policyGeneration || a.runningGeneration != b.runningGeneration || a.modelGeneration != b.modelGeneration || a.policyPath != b.policyPath || a.policyDigest != b.policyDigest || a.settingsDigest != b.settingsDigest || a.operationsDigest != b.operationsDigest || a.engineDigest != b.engineDigest || len(a.pendingInstalls) != len(b.pendingInstalls) {
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
