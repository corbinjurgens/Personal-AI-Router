// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"slices"

	"nvpair-shared/durablefile"
	"nvpair-shared/engineintent"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

type settingsWorkerError struct {
	code    int
	message string
}

func (e *settingsWorkerError) Error() string { return e.message }

func engineSettingsErrorCode(err error) int {
	var workerError *settingsWorkerError
	if errors.As(err, &workerError) && workerError.code != 0 {
		return workerError.code
	}
	return policyErrorCode(err)
}

// validateModelSettingsEntry runs before journal initialization or worker IO.
// The settings entry points reserve maintenance after validation, before either
// the ordinary or model-change branch can enter an owner or worker.
func validateModelSettingsEntry(p settings.Request) error {
	if err := p.Validate(); err != nil {
		return &invalidParamsError{err: err}
	}
	if p.ModelChange != nil && (p.Settings.ServerPort < 1 || p.Settings.ServerPort > 65535 || p.Settings.ProxyPort < 1 || p.Settings.ProxyPort > 65535 || p.Settings.ServerPort == p.Settings.ProxyPort) {
		return &invalidParamsError{err: fmt.Errorf("modelChange requires a complete target settings baseline")}
	}
	return nil
}

// This is a subrecord of the settings journal, including for load-only work.
// Only a selected policy patch reserves the whole policy. Failed and uncertain
// operations keep their withdrawal until a fresh local supersession is accepted.
type modelSettingsOperation struct {
	Caller         string                  `json:"caller,omitempty"`
	Change         modelconfig.ModelChange `json:"change"`
	Review         settings.Preview        `json:"review"`
	NativeBefore   settings.Config         `json:"nativeBefore"`
	Before         string                  `json:"before"`
	After          string                  `json:"after"`
	Implications   string                  `json:"implications"`
	Stage          string                  `json:"stage"`
	Attempted      bool                    `json:"attempted"`
	PolicyRevision string                  `json:"policyRevision,omitempty"`
}

func validateStoredModelSettings(record *engineSettingsRecord) error {
	op := record.Model
	if op == nil {
		return nil
	}
	if err := op.Change.Validate(); err != nil {
		return err
	}
	if !slices.Contains([]string{"accepted", "acceptance-uncertain", "drained", "native-applying", "native-persisted", "policy-persisted", "activation-pending", "releasing", "completed", "completion-uncertain"}, op.Stage) {
		return fmt.Errorf("invalid model settings stage")
	}
	if _, err := op.NativeBefore.Canonical(); err != nil {
		return err
	}
	if _, err := record.Snapshot.Settings.Canonical(); err != nil {
		return err
	}
	if !op.Review.Settings.Equal(record.Snapshot.Settings) || record.Snapshot.RequestID == "" {
		return fmt.Errorf("model settings journal lost its accepted review")
	}
	for _, fingerprint := range []string{op.Before, op.After, op.Implications} {
		if value, err := hex.DecodeString(fingerprint); err != nil || len(value) != sha256.Size {
			return fmt.Errorf("invalid model settings policy fingerprint")
		}
	}
	return nil
}

func modelSettingsConflict(message string) error {
	return &settingsWorkerError{code: codePolicyConflict, message: message}
}

func modelSettingsHash(value any) string {
	// Canonical object ordering includes opaque nested request-option objects;
	// decoding with UseNumber preserves every numeric digit.
	raw := settingsJSON(value)
	var canonical any
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if decoder.Decode(&canonical) == nil {
		raw = settingsJSON(canonical)
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func modelPolicyFingerprint(policy nodepolicy.Policy) string {
	var fields map[string]json.RawMessage
	if json.Unmarshal(settingsJSON(policy), &fields) != nil {
		return ""
	}
	dropFields(fields, "revision", "availability", "pausedBy")
	return modelSettingsHash(fields)
}

// A load-only apply does not reserve unrelated policy. It does depend on the
// complete profile (including qualification/bounds), residency and serving
// implications of the selected model. Availability is checked separately.
func modelPolicyImplications(policy nodepolicy.Policy, engine, model string) string {
	var profiles []nodepolicy.Profile
	for _, profile := range policy.Profiles {
		if profile.Engine == engine && profile.Model == model {
			profiles = append(profiles, profile)
		}
	}
	ref := nodepolicy.Role{Engine: engine, Model: model}
	return modelSettingsHash(struct {
		Profiles                       []nodepolicy.Profile
		Resident, Served, Unrestricted bool
	}{profiles, slices.Contains(policy.ResidentModels, ref), slices.Contains(policy.ServedModels, ref), policy.ServedModels == nil})
}

func (b *Broker) modelSettingsPendingLocked(engine string) bool {
	record := b.engineSettings[engine]
	return record != nil && record.Model != nil && record.Model.Stage != "completed"
}

// Pending snapshots bypass live readback. Their running observation belongs to
// the old incarnation when loading a journal or replacing engine-manager, even
// though the accepted intent and reconciliation state remain authoritative.
// The caller holds engineConfigMu; same-incarnation acceptance uncertainty
// deliberately does not invalidate the running observation.
func (b *Broker) invalidatePendingModelSettingsRunningLocked() {
	for engine, record := range b.engineSettings {
		if b.modelSettingsPendingLocked(engine) {
			record.Snapshot.Running = false
		}
	}
}

func modelSettingsNeedsReconciliation(record *engineSettingsRecord) bool {
	return record.Model != nil && record.Model.Stage != "completed" && (record.Model.Stage == "activation-pending" || record.Model.Stage == "acceptance-uncertain" || record.Receipts[record.Snapshot.RequestID].Phase == "failed")
}

// An unreadable journal cannot establish whether unrecognizable bytes once
// contained model work. Keep ordinary restoration behavior, but never discard
// a model reservation already recognizable in the file or known in memory.
func (b *Broker) knownModelSettingsFenceLocked() bool {
	for _, record := range b.engineSettings {
		if record != nil && record.Model != nil && (record.Model.Stage != "completed" || b.engineSettingsError != nil && validateStoredModelSettings(record) != nil) {
			return true
		}
	}
	return false
}

func markModelAcceptanceUncertain(record *engineSettingsRecord) {
	record.Model.Stage, record.Snapshot.CoupledStage = "acceptance-uncertain", "acceptance-uncertain"
	record.Snapshot.Phase, record.Snapshot.Editable = "applying", false
	record.Snapshot.Error, record.Snapshot.UnresolvedError = "Interrupted before changes; preview again on this device.", "Interrupted before changes; preview again on this device."
	record.Snapshot.Reason = record.Snapshot.UnresolvedError
	receipt := record.Receipts[record.Snapshot.RequestID]
	receipt.Phase = "failed"
	record.Receipts[record.Snapshot.RequestID] = receipt
}

// Must run before taking policyRuntime.mu. Availability writers deliberately
// do not use this fence; they retain their existing independent owner.
func (b *Broker) modelPolicyReservationLocked() error {
	if err := b.loadEngineSettingsLocked(); err != nil {
		if b.knownModelSettingsFenceLocked() {
			return err
		}
		return nil
	}
	for _, record := range b.engineSettings {
		if record.Model != nil && record.Model.Change.Policy != nil && record.Model.Stage != "completed" {
			return &policyConflictError{expected: "pending model settings " + record.Snapshot.RequestID, current: record.Model.PolicyRevision}
		}
	}
	return nil
}

// The journal is the reservation authority. Rebuild volatile admission and
// identity fences before replaying proxy state, advertisements or engine intent.
func (b *Broker) reconstructModelSettingsLocked() {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	for engine, record := range b.engineSettings {
		if record == nil {
			continue
		}
		if b.modelSettingsPendingLocked(engine) {
			record.Snapshot.Phase = "applying"
			if record.Model.Change.Load != nil {
				r.drains[engine] = true
				delete(r.settingsIdentity, engine)
			}
		}
		if record.Model != nil {
			record.Snapshot.CoupledStage = record.Model.Stage
		}
	}
}

// Merge only selected leaves. Raw request-option subtrees retain opaque JSON
// numbers rather than making a float64 round trip. No bounds or qualification
// are synthesized, and a peer uses exactly the existing inference-lock check.
func (b *Broker) mergeModelPolicy(stored nodepolicy.Policy, engine string, change modelconfig.ModelChange, caller string) (nodepolicy.Policy, error) {
	var policy nodepolicy.Policy
	if err := json.Unmarshal(settingsJSON(stored), &policy); err != nil {
		return policy, err
	}
	patch := change.Policy
	if patch == nil {
		return policy, nil
	}
	ref := nodepolicy.Role{Engine: engine, Model: change.ModelID}
	if len(patch.Generation) > 0 || patch.RequestOptionsMode != nil {
		i := slices.IndexFunc(policy.Profiles, func(p nodepolicy.Profile) bool { return p.Engine == engine && p.Model == change.ModelID })
		if i < 0 {
			name := "model-" + modelSettingsHash(ref)[:16]
			for _, p := range policy.Profiles {
				if p.Name == name {
					return policy, &invalidParamsError{err: fmt.Errorf("generated profile name is already in use")}
				}
			}
			policy.Profiles = append(policy.Profiles, nodepolicy.Profile{Name: name, Engine: engine, Model: change.ModelID})
			i = len(policy.Profiles) - 1
		}
		profile := &policy.Profiles[i]
		options := map[string]json.RawMessage{}
		if len(profile.RequestOptions) != 0 {
			if err := json.Unmarshal(profile.RequestOptions, &options); err != nil {
				return policy, &invalidParamsError{err: err}
			}
			if options == nil {
				options = map[string]json.RawMessage{}
			}
		}
		for key, value := range patch.Generation {
			if value == nil {
				delete(options, key)
			} else {
				options[key] = json.RawMessage(value.String())
			}
		}
		if len(options) == 0 {
			profile.RequestOptions = nil
		} else {
			profile.RequestOptions = settingsJSON(options)
		}
		if patch.RequestOptionsMode != nil {
			profile.RequestOptionsMode = nodepolicy.RequestOptionsMode(*patch.RequestOptionsMode)
		}
	}
	if len(patch.Roles) > 0 && policy.Roles == nil {
		policy.Roles = map[string]nodepolicy.Role{}
	}
	for role, selected := range patch.Roles {
		if selected {
			policy.Roles[role] = ref
		} else if policy.Roles[role] == ref {
			delete(policy.Roles, role)
		}
	}
	member := func(list []nodepolicy.Role, selected bool) []nodepolicy.Role {
		out := slices.Clone(list)
		if selected && !slices.Contains(out, ref) {
			out = append(out, ref)
		}
		if !selected {
			out = slices.DeleteFunc(out, func(r nodepolicy.Role) bool { return r == ref })
		}
		if out == nil {
			out = []nodepolicy.Role{}
		}
		return out
	}
	if patch.Served != nil {
		if policy.ServedModels == nil && !*patch.Served {
			return policy, &invalidParamsError{err: fmt.Errorf("cannot exclude one model from unrestricted serving; select an explicit served list first")}
		}
		// Selecting an already-unrestricted model preserves unrestricted scope.
		if policy.ServedModels != nil {
			policy.ServedModels = member(policy.ServedModels, *patch.Served)
		}
	}
	if patch.Resident != nil {
		policy.ResidentModels = member(policy.ResidentModels, *patch.Resident)
	}
	if patch.Binding != nil {
		policy.EngineBindings = member(policy.EngineBindings, *patch.Binding)
	}
	// Use the policy owner's normalization as well as validation so the
	// recorded after-fingerprint survives its normal disk parse on restart.
	canonical, err := nodepolicy.Parse(settingsJSON(policy))
	if err != nil {
		return policy, &invalidParamsError{err: err}
	}
	policy = canonical
	if err := policy.ValidateEngineBindings(b.engineModelNaming); err != nil {
		return policy, &invalidParamsError{err: err}
	}
	if caller != "" {
		merged, err := nodepolicy.MergePeerSet(stored, withoutRevision(settingsJSON(policy.PeerProjection())))
		if err != nil {
			return policy, err
		}
		policy = merged
	}
	return policy, nil
}

func modelSettingsRequest(p settings.Request, caller string) settings.Request {
	p = p.Clone()
	p.NodeID = ""
	p.PreserveCORS = caller != ""
	if p.ModelChange != nil && p.ModelChange.Load != nil {
		slices.Sort(p.ModelChange.Load.Clear)
	}
	return p
}

// Calls are pinned to one worker and never run under the journal/policy mutex.
func modelSettingsCall(ctx context.Context, worker *rpcWorker, method string, params, target any) error {
	if worker == nil {
		return fmt.Errorf("engine manager unavailable")
	}
	raw, rpcErr, err := worker.CallNoTimeout(ctx, method, settingsJSON(params))
	if err != nil {
		return err
	}
	if rpcErr != nil {
		return &settingsWorkerError{code: rpcErr.Code, message: rpcErr.Message}
	}
	if target != nil {
		return json.Unmarshal(raw, target)
	}
	return nil
}

func (b *Broker) modelWorkerCurrent(worker *rpcWorker, source *engineManagerSource) bool {
	if worker == nil || source == nil || b.getEngineMgr() != worker || !b.engineMgrPairCurrent(worker, source) {
		return false
	}
	select {
	case <-worker.Done():
		return false
	default:
		return true
	}
}

// Preview reads native facts outside owner locks, then validates the captured
// journal/policy/source again. Apply performs this same read and compares the
// entire review digest; it cannot silently substitute a new resolution.
type modelSettingsReviewSource struct {
	Launch settings.LaunchState
	Worker *rpcWorker
	Source *engineManagerSource
}

func (b *Broker) previewModelSettings(ctx context.Context, input settings.Request, caller string) (settings.Preview, error) {
	return b.reviewModelSettings(ctx, input, caller, nil)
}

func (b *Broker) reviewModelSettings(ctx context.Context, input settings.Request, caller string, capture *modelSettingsReviewSource) (settings.Preview, error) {
	ctx, cancel := context.WithTimeout(ctx, settings.CallBudget)
	defer cancel()
	p := modelSettingsRequest(input, caller)
	// No journal initialization or worker I/O before the target caller check.
	b.engineConfigMu.Lock()
	err := b.authorizeSettingsCallerLocked(caller)
	b.engineConfigMu.Unlock()
	if err != nil {
		return settings.Preview{}, err
	}
	worker := b.getEngineMgr()
	b.workersMu.Lock()
	source := b.engineMgrSource
	b.workersMu.Unlock()
	var launch settings.LaunchState
	if err := modelSettingsCall(ctx, worker, "engine:get-launch", map[string]string{"engine": p.Engine}, &launch); err != nil {
		return settings.Preview{}, err
	}
	b.engineConfigMu.Lock()
	base, policy, err := b.modelSettingsBaselineLocked(p, caller, launch)
	b.engineConfigMu.Unlock()
	if err != nil {
		return settings.Preview{}, err
	}
	nextPolicy, err := b.mergeModelPolicy(policy, p.Engine, *p.ModelChange, caller)
	if err != nil {
		return settings.Preview{}, err
	}
	preview := settings.Preview{Settings: base.Snapshot.Settings.Clone(), Revision: base.Snapshot.Revision, PolicyRevision: policy.Revision}
	if p.ModelChange.Load != nil {
		request := p.Clone()
		change := request.ModelChange
		change.Policy, change.PreviewDigest = nil, ""
		// During local supersession, the native store may differ from desired.
		request.Settings = settings.ConfigFromLaunch(launch, p.Settings.ProxyPort)
		if err := modelSettingsCall(ctx, worker, "engine:preview-launch", request, &preview); err != nil {
			return preview, err
		}
		if len(preview.Errors) > 0 || preview.Conflict != nil {
			return preview, nil
		}
		if len(preview.PreviewDigest) != sha256.Size*2 {
			return settings.Preview{}, fmt.Errorf("native preview did not bind its source identity")
		}
		if _, err := preview.Settings.Canonical(); err != nil {
			return settings.Preview{}, err
		}
		if preview.Settings.ServerPort != p.Settings.ServerPort || preview.Settings.ProxyPort != p.Settings.ProxyPort {
			return settings.Preview{}, fmt.Errorf("model preview changed an unrelated port")
		}
		if len(preview.Settings.ModelLoads) == 0 {
			return settings.Preview{}, fmt.Errorf("model preview did not resolve native load settings")
		}
		for _, load := range preview.Settings.ModelLoads {
			if load.Resolved == nil {
				return settings.Preview{}, fmt.Errorf("model preview returned an unresolved load")
			}
		}
	}
	nativeDigest := preview.PreviewDigest
	preview.Revision, preview.PolicyRevision = base.Snapshot.Revision, policy.Revision
	preview.Rebind = preview.Settings.ProxyPort != base.Snapshot.Settings.ProxyPort
	preview.Restart = launch.Running && !preview.Settings.Equal(settings.ConfigFromLaunch(launch, p.Settings.ProxyPort))
	// Supersession reviews the unresolved running intent as well as current
	// observation: a replacement manager may have persisted settings but stopped.
	if p.ModelChange.SupersedesRequestID != "" && p.ModelChange.Load != nil && base.Resume {
		preview.Restart = true
	}
	preview.Impact = b.modelSettingsImpact(p.Engine, launch, preview.Settings, preview.Restart)
	if p.ModelChange.Policy != nil {
		preview.Diff = append(preview.Diff, modelconfig.FieldDiff{Field: "policy", Before: settingsJSON(policy.PeerProjection()), After: settingsJSON(nextPolicy.PeerProjection())})
	}
	// Re-read source state after worker inspection, before taking owner locks.
	var after settings.LaunchState
	if err := modelSettingsCall(ctx, worker, "engine:get-launch", map[string]string{"engine": p.Engine}, &after); err != nil {
		return settings.Preview{}, err
	}
	if !reflect.DeepEqual(launch, after) || !b.modelWorkerCurrent(worker, source) {
		return settings.Preview{}, modelSettingsConflict("native source changed during preview; preview again")
	}
	b.engineConfigMu.Lock()
	checked, latest, err := b.modelSettingsBaselineLocked(p, caller, after)
	if err == nil && (checked.Snapshot.Revision != base.Snapshot.Revision || latest.Revision != policy.Revision) {
		err = modelSettingsConflict("model settings changed during preview")
	}
	if err == nil {
		err = b.boundPolicyChangeLocked(policy, nextPolicy)
	}
	b.engineConfigMu.Unlock()
	if err != nil {
		return settings.Preview{}, err
	}
	request := p.Clone()
	request.RequestID, request.ModelChange.PreviewDigest = "", ""
	preview.PreviewDigest = ""
	preview.PreviewDigest = modelSettingsHash(struct {
		Caller                 string
		Request                settings.Request
		Policy, Native, Source string
		Preview                settings.Preview
	}{caller, request, modelPolicyFingerprint(policy), nativeDigest, source.settingsEpoch, preview})
	if err := modelconfig.CheckPayloadSize(preview); err != nil {
		return settings.Preview{}, &invalidParamsError{err: err}
	}
	if capture != nil {
		*capture = modelSettingsReviewSource{Launch: launch, Worker: worker, Source: source}
	}
	return preview, nil
}

// Caller holds engineConfigMu; this only inspects local authority, never a worker.
func (b *Broker) modelSettingsBaselineLocked(p settings.Request, caller string, launch settings.LaunchState) (*engineSettingsRecord, nodepolicy.Policy, error) {
	fail := func(err error) (*engineSettingsRecord, nodepolicy.Policy, error) {
		return nil, nodepolicy.Policy{}, err
	}
	if err := b.authorizeSettingsCallerLocked(caller); err != nil {
		return fail(err)
	}
	if err := b.loadEngineSettingsLocked(); err != nil {
		return fail(err)
	}
	if err := b.pendingSwitchLocked(p.Engine); err != nil {
		return fail(err)
	}
	b.retireReplacedBoundOperationLocked()
	if op := b.boundCoordinator().active; op != nil && (op.request.Engine == p.Engine || op.sourceEngine == p.Engine) {
		return fail(modelSettingsConflict("automatic engine preparation reserves this engine"))
	}
	if launch.Engine != p.Engine || !launch.Editable {
		return fail(modelSettingsConflict("engine is not editable"))
	}
	proxy := b.settingsProxy(p.Engine)
	if proxy == nil {
		return fail(fmt.Errorf("proxy unavailable"))
	}
	ready, port := proxy.Status(p.Engine)
	if !ready {
		return fail(fmt.Errorf("proxy unavailable"))
	}
	native := settings.ConfigFromLaunch(launch, port)
	record := b.engineSettings[p.Engine].clone()
	if record == nil {
		record = &engineSettingsRecord{Snapshot: settings.Snapshot{NodeID: b.nodeID, Engine: p.Engine, Epoch: b.engineSettingsEpoch, Revision: 1, AppliedRevision: 1, Phase: "idle", Settings: native.Clone()}, Receipts: map[string]settingsReceipt{}}
	} else if !b.modelSettingsPendingLocked(p.Engine) && record.Snapshot.Phase != "applying" && record.Snapshot.Phase != "failed" && !record.Snapshot.Settings.Equal(native) {
		record.Snapshot.Settings = native.Clone()
		record.Snapshot.Revision++
		record.Snapshot.AppliedRevision = record.Snapshot.Revision
	}
	if record.Snapshot.Phase == "applying" && !modelSettingsNeedsReconciliation(record) {
		return fail(modelSettingsConflict("settings operation is still applying"))
	}
	if b.modelSettingsPendingLocked(p.Engine) {
		if caller != "" || p.ModelChange.SupersedesRequestID != record.Snapshot.RequestID || p.RequestID == record.Snapshot.RequestID {
			return fail(modelSettingsConflict("local fresh preview with supersedesRequestId required for " + record.Snapshot.RequestID))
		}
		if record.Model.Change.Load != nil && p.ModelChange.Load == nil {
			return fail(modelSettingsConflict("native reconciliation requires a fresh load preview"))
		}
	} else if p.ModelChange.SupersedesRequestID != "" {
		return fail(modelSettingsConflict("superseded operation is no longer pending"))
	}
	for engine, pending := range b.engineSettings {
		if engine != p.Engine && b.modelSettingsPendingLocked(engine) && pending.Model.Change.Policy != nil && p.ModelChange.Policy != nil {
			return fail(modelSettingsConflict("policy is reserved by " + pending.Snapshot.RequestID))
		}
	}
	if p.ExpectedRevision != record.Snapshot.Revision {
		return fail(modelSettingsConflict("settings revision changed; reload and preview again"))
	}
	if err := p.ValidateModelChange(record.Snapshot.Settings, false); err != nil {
		return fail(&invalidParamsError{err: err})
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := checkExpectedRevision(r.current, p.ModelChange.ExpectedPolicyRevision); err != nil {
		return fail(err)
	}
	policy := r.current
	record.Snapshot.Running, record.Snapshot.Adopted = launch.Running, launch.Adopted
	record.Snapshot.EffectiveServerPort, record.Snapshot.EffectiveProxyPort = launch.EffectivePort, port
	record.Snapshot.Editable, record.Snapshot.Format = true, launch.Format
	record.Snapshot.EffectiveSettings = &native
	return record, policy, nil
}

func (b *Broker) modelSettingsImpact(engine string, launch settings.LaunchState, desired settings.Config, restart bool) *modelconfig.Impact {
	models := map[string]bool{}
	for _, load := range launch.ModelLoads {
		models[load.ModelID] = true
	}
	for _, load := range desired.ModelLoads {
		models[load.ModelID] = true
	}
	for _, config := range []settings.Config{settings.ConfigFromLaunch(launch, 0), desired} {
		if config.Serving != nil {
			for _, model := range config.Serving.Models {
				models[model.ModelID] = true
			}
		}
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	_, known := r.loaded[engine]
	known = known && r.models.loadedSource == r.engineManagerSource && r.models.loadedProxy == b.policyProxy()
	if known {
		for _, model := range r.loaded[engine] {
			models[model] = true
		}
	}
	paused := r.managementBlockedLocked()
	r.mu.Unlock()
	scope := "none"
	if restart {
		scope = "whole-router"
	}
	interrupt, off := restart, !launch.Running && !restart || paused
	impact := &modelconfig.Impact{RestartScope: scope, MayInterruptActive: &interrupt, StaysOff: &off, ActiveWorkPolicy: "none"}
	if restart {
		seconds := int32(60)
		impact.DrainTimeoutSeconds, impact.ActiveWorkPolicy = &seconds, "drain-then-restart"
	}
	for model := range models {
		impact.AffectedModels = append(impact.AffectedModels, model)
	}
	slices.Sort(impact.AffectedModels)
	if !known {
		impact.Unknowns = []string{"Loaded router models have no current readback."}
	}
	return impact
}

func (b *Broker) applyModelSettings(ctx context.Context, input settings.Request, caller string) (settingsReceipt, error) {
	p := modelSettingsRequest(input, caller)
	if p.RequestID == "" || p.ModelChange.PreviewDigest == "" {
		return settingsReceipt{}, &invalidParamsError{err: fmt.Errorf("requestId and reviewed previewDigest are required")}
	}
	ctx, cancel := context.WithTimeout(ctx, settings.CallBudget)
	defer cancel()
	b.settingsApplyMu.Lock()
	defer b.settingsApplyMu.Unlock()
	hash := modelSettingsHash(struct {
		Caller  string
		Request settings.Request
	}{caller, p})
	// Receipt lookup precedes CAS, but never the fresh pin/grant/Pause checks.
	b.engineConfigMu.Lock()
	err := b.authorizeSettingsCallerLocked(caller)
	if err == nil {
		err = b.authorizeRemoteManagement(caller, "settings", p.RequestID)
	}
	if err == nil {
		err = b.loadEngineSettingsLocked()
	}
	if err == nil {
		if record := b.engineSettings[p.Engine]; record != nil {
			if receipt, exists := record.Receipts[p.RequestID]; exists {
				b.engineConfigMu.Unlock()
				if receipt.Hash != hash {
					return settingsReceipt{}, &invalidParamsError{err: fmt.Errorf("request identifier was already used for different content or caller")}
				}
				return receipt, nil
			}
		}
	}
	b.engineConfigMu.Unlock()
	if err != nil {
		return settingsReceipt{}, err
	}
	var source modelSettingsReviewSource
	preview, err := b.reviewModelSettings(ctx, p, caller, &source)
	if err != nil {
		return settingsReceipt{}, err
	}
	if len(preview.Errors) != 0 || preview.Conflict != nil {
		return settingsReceipt{}, &invalidParamsError{err: fmt.Errorf("resolve model preview errors before applying")}
	}
	if preview.PreviewDigest != p.ModelChange.PreviewDigest {
		return settingsReceipt{}, modelSettingsConflict("review changed; obtain a fresh model settings preview")
	}
	worker := source.Worker
	var launch settings.LaunchState
	if err := modelSettingsCall(ctx, worker, "engine:get-launch", map[string]string{"engine": p.Engine}, &launch); err != nil {
		return settingsReceipt{}, err
	}
	if !reflect.DeepEqual(launch, source.Launch) {
		return settingsReceipt{}, modelSettingsConflict("native source changed before acceptance; preview again")
	}
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	record, policy, err := b.modelSettingsBaselineLocked(p, caller, launch)
	if err != nil {
		return settingsReceipt{}, err
	}
	if !b.modelWorkerCurrent(worker, source.Source) || ctx.Err() != nil {
		return settingsReceipt{}, modelSettingsConflict("engine source changed before acceptance")
	}
	next, err := b.mergeModelPolicy(policy, p.Engine, *p.ModelChange, caller)
	if err != nil {
		return settingsReceipt{}, err
	}
	if err := b.boundPolicyChangeLocked(policy, next); err != nil {
		return settingsReceipt{}, err
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	// Last admission decision shares the policy/availability ordering with the
	// durable journal write, with no worker I/O inside this section.
	if err := b.authorizeSettingsCallerLocked(caller); err != nil {
		r.mu.Unlock()
		return settingsReceipt{}, err
	}
	if err := b.authorizeRemoteManagementLocked(r, caller, "settings", p.RequestID); err != nil {
		r.mu.Unlock()
		return settingsReceipt{}, err
	}
	if err := checkExpectedRevision(r.current, p.ModelChange.ExpectedPolicyRevision); err != nil {
		r.mu.Unlock()
		return settingsReceipt{}, err
	}
	previous := b.engineSettings[p.Engine]
	if previous != nil && previous.Model != nil && previous.Model.Stage != "completed" {
		// Retire the previous receipt only in the same write that accepts the
		// replacement; native readback is carried as the new operation's baseline.
		receipt := record.Receipts[previous.Snapshot.RequestID]
		receipt.Phase = "superseded"
		record.Receipts[previous.Snapshot.RequestID] = receipt
	}
	record.Previous = record.Snapshot.Settings.Clone()
	record.Resume = preview.Restart && !r.managementBlockedLocked()
	record.Explicit = true
	record.Model = &modelSettingsOperation{Caller: caller, Change: p.ModelChange.Clone(), Review: preview.Clone(), NativeBefore: settings.ConfigFromLaunch(launch, preview.Settings.ProxyPort), Before: modelPolicyFingerprint(policy), After: modelPolicyFingerprint(next), Implications: modelPolicyImplications(policy, p.Engine, p.ModelChange.ModelID), Stage: "accepted"}
	record.Snapshot.Settings = preview.Settings.Clone()
	record.Snapshot.Revision++
	if record.Snapshot.Revision > settings.MaxSafeCounter {
		r.mu.Unlock()
		return settingsReceipt{}, fmt.Errorf("settings revision is exhausted")
	}
	record.Snapshot.RequestID, record.Snapshot.Phase = p.RequestID, "applying"
	record.Snapshot.Editable, record.Snapshot.Reason = false, "Model settings are being applied."
	record.Snapshot.CoupledStage, record.Snapshot.PolicyRevision = "accepted", policy.Revision
	record.Snapshot.Error, record.Snapshot.UnresolvedError = "", ""
	record.Receipts[p.RequestID] = settingsReceipt{Hash: hash, Revision: record.Snapshot.Revision, Phase: "applying"}
	b.engineSettings[p.Engine] = record
	if err := b.saveEngineSettingsLocked(); err != nil {
		if errors.Is(err, durablefile.ErrReplaced) {
			acceptedHash := modelSettingsHash(b.engineSettings)
			// Update only the exact replacement that just landed. If this
			// second write also fails, loading "accepted" remains non-replayable.
			markModelAcceptanceUncertain(record)
			path, pathErr := b.engineSettingsPath()
			var data []byte
			if pathErr == nil {
				data, pathErr = os.ReadFile(path)
			}
			var landed map[string]*engineSettingsRecord
			if pathErr == nil {
				pathErr = json.Unmarshal(data, &landed)
			}
			if pathErr == nil && modelSettingsHash(landed) != acceptedHash {
				pathErr = fmt.Errorf("acceptance readback differs from the landed journal")
			}
			if pathErr != nil {
				b.engineSettingsError = fmt.Errorf("acceptance reconciliation cannot replace changed or unreadable journal: %w", pathErr)
			} else if saveErr := b.saveEngineSettingsLocked(); saveErr != nil {
				record.Snapshot.UnresolvedError += "; uncertainty record durability could not be confirmed"
			}
			if record.Model.Change.Load != nil {
				r.drains[p.Engine] = true
				delete(r.settingsIdentity, p.Engine)
			}
		} else if previous == nil {
			delete(b.engineSettings, p.Engine)
		} else {
			b.engineSettings[p.Engine] = previous
		}
		r.mu.Unlock()
		return settingsReceipt{}, fmt.Errorf("model settings acceptance could not be confirmed; no native or policy effects were made: %w", err)
	}
	if record.Model.Change.Load != nil {
		r.drains[p.Engine] = true
		delete(r.settingsIdentity, p.Engine)
	}
	r.mu.Unlock()
	b.signalModelConfig()
	b.publishSettingsLocked()
	operationCtx, operationCancel := context.WithTimeout(context.Background(), settings.OperationBudget)
	defer operationCancel()
	_ = b.runModelSettingsOperationLocked(operationCtx, p.Engine, record)
	return record.Receipts[p.RequestID], nil
}

// Caller holds settingsApplyMu and engineConfigMu. Save before every effect;
// all native/proxy calls release engineConfigMu. Failed writes never release
// the journal-derived fence, including a terminal-result write failure.
func (b *Broker) saveModelStageLocked(record *engineSettingsRecord, stage string) error {
	previous := record.Model.Stage
	record.Model.Stage, record.Snapshot.CoupledStage = stage, stage
	if err := b.saveEngineSettingsLocked(); err != nil {
		if !errors.Is(err, durablefile.ErrReplaced) {
			record.Model.Stage, record.Snapshot.CoupledStage = previous, previous
		}
		return err
	}
	return nil
}

func (b *Broker) failModelSettingsLocked(record *engineSettingsRecord, err error) error {
	if record.Model.Stage == "completed" {
		record.Model.Stage, record.Snapshot.CoupledStage = "completion-uncertain", "completion-uncertain"
	}
	// The operation itself remains nonterminal. Existing switch, cold binding
	// and legacy mutation owners already fence applying settings rows; the
	// failed attempt is the receipt, not permission to discard that reservation.
	record.Snapshot.Phase = "applying"
	record.Snapshot.Error, record.Snapshot.UnresolvedError = err.Error(), err.Error()
	record.Snapshot.Editable = false
	record.Snapshot.Reason = "Local fresh preview and supersession are required."
	receipt := record.Receipts[record.Snapshot.RequestID]
	receipt.Phase = "failed"
	record.Receipts[record.Snapshot.RequestID] = receipt
	if saveErr := b.saveEngineSettingsLocked(); saveErr != nil {
		record.Snapshot.UnresolvedError += "; reconciliation result could not be saved"
	}
	b.publishSettingsLocked()
	return err
}

func (b *Broker) checkModelPolicyLocked(record *engineSettingsRecord) error {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	op := record.Model
	if op.Change.Policy == nil {
		if modelPolicyImplications(r.current, record.Snapshot.Engine, op.Change.ModelID) != op.Implications {
			return modelSettingsConflict("model profile or residency changed; local fresh preview required")
		}
	} else {
		fingerprint := modelPolicyFingerprint(r.current)
		if fingerprint != op.Before && fingerprint != op.After {
			return modelSettingsConflict("policy content conflicts with the accepted operation; local fresh preview required")
		}
	}
	return nil
}

// The private same-operation writer is reachable only from a journal record.
// It preserves current availability/provenance and refuses any third content.
func (b *Broker) commitModelPolicyLocked(record *engineSettingsRecord) error {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	op := record.Model
	if op.Change.Policy == nil {
		if modelPolicyImplications(r.current, record.Snapshot.Engine, op.Change.ModelID) != op.Implications {
			return modelSettingsConflict("model policy implications changed")
		}
		op.PolicyRevision = r.current.Revision
		record.Snapshot.PolicyRevision = op.PolicyRevision
		return nil
	}
	fingerprint := modelPolicyFingerprint(r.current)
	if fingerprint == op.After {
		op.PolicyRevision = r.current.Revision
		record.Snapshot.PolicyRevision = op.PolicyRevision
		return nil
	}
	if fingerprint != op.Before {
		return modelSettingsConflict("a third policy state cannot be overwritten")
	}
	policy, err := b.mergeModelPolicy(r.current, record.Snapshot.Engine, op.Change, op.Caller)
	if err != nil {
		return err
	}
	if modelPolicyFingerprint(policy) != op.After {
		return modelSettingsConflict("accepted policy patch no longer produces the reviewed content")
	}
	if err := b.boundPolicyChangeLocked(r.current, policy); err != nil {
		return err
	}
	policy, err = r.publishLocked(policy)
	if err != nil && !errors.Is(err, durablefile.ErrReplaced) {
		return &policySaveError{err: err}
	}
	r.current, r.recoveryRequired, r.activationErr = policy, false, ""
	op.PolicyRevision = policy.Revision
	record.Snapshot.PolicyRevision = op.PolicyRevision
	if err != nil {
		return &policySaveError{err: err, revision: policy.Revision}
	}
	return nil
}

func observeModelSettingsLaunch(record *engineSettingsRecord, launch settings.LaunchState) {
	if launch.Engine != record.Snapshot.Engine {
		return
	}
	actual := settings.ConfigFromLaunch(launch, record.Snapshot.EffectiveProxyPort)
	record.Snapshot.EffectiveSettings = &actual
	record.Snapshot.Serving = launch.ServingReadback.Clone()
	record.Snapshot.Running, record.Snapshot.Adopted = launch.Running, launch.Adopted
	record.Snapshot.EffectiveServerPort = launch.EffectivePort
}

// ConfigureLaunch owns activation and its Pause/lock/saved-Off guards. If its
// resume converges to stopped, require fresh explicit veto evidence from that
// same manager before treating the stopped result as a successful saved apply.
func modelSettingsResumeVeto(ctx context.Context, worker *rpcWorker, engine string) (bool, error) {
	var intent engineintent.Snapshot
	if err := modelSettingsCall(ctx, worker, engineIntentMethod, nil, &intent); err != nil {
		return false, err
	}
	if err := intent.Validate(); err != nil {
		return false, err
	}
	if enabled, known := intent.EnabledByEngine[engine]; known && !enabled && intent.OffByEngine[engine] != "" {
		return true, nil
	}
	var installed struct {
		Engines []struct {
			Engine string `json:"engine"`
			Locked *bool  `json:"locked"`
		} `json:"engines"`
	}
	if err := modelSettingsCall(ctx, worker, "engine:get-installed", nil, &installed); err != nil {
		return false, err
	}
	for _, entry := range installed.Engines {
		if entry.Engine == engine && entry.Locked != nil {
			return *entry.Locked, nil
		}
	}
	return false, fmt.Errorf("engine activation veto readback is unavailable")
}

func (b *Broker) runModelSettingsOperationLocked(ctx context.Context, engine string, record *engineSettingsRecord) error {
	fail := func(err error) error { return b.failModelSettingsLocked(record, err) }
	if err := b.checkModelPolicyLocked(record); err != nil {
		return fail(err)
	}
	if err := b.pendingSwitchLocked(engine); err != nil {
		return fail(err)
	}
	op := record.Model
	worker := b.getEngineMgr()
	b.workersMu.Lock()
	source := b.engineMgrSource
	b.workersMu.Unlock()
	if worker == nil {
		return fail(fmt.Errorf("engine manager unavailable; operation remains withdrawn"))
	}
	if op.Change.Load != nil {
		// A running restart has a disclosed 60-second wait after an acknowledged
		// fence. A lost drain ACK never authorizes a native effect.
		b.engineConfigMu.Unlock()
		err := b.drainModelSettings(ctx, engine, record.Resume)
		b.engineConfigMu.Lock()
		if err != nil {
			return fail(err)
		}
		if err := b.checkModelPolicyLocked(record); err != nil {
			return fail(err)
		}
		if err := b.saveModelStageLocked(record, "drained"); err != nil {
			return fail(err)
		}
		var launch settings.LaunchState
		var validated settings.Preview
		b.engineConfigMu.Unlock()
		err = modelSettingsCall(ctx, worker, "engine:get-launch", map[string]string{"engine": engine}, &launch)
		if err == nil {
			// Full frozen settings revalidation checks bindings/build without
			// running the portable patch or selecting a fresh Auto result.
			err = modelSettingsCall(ctx, worker, "engine:preview-launch", settings.Request{Engine: engine, Settings: record.Snapshot.Settings.Clone()}, &validated)
		}
		if err == nil {
			var rechecked settings.LaunchState
			err = modelSettingsCall(ctx, worker, "engine:get-launch", map[string]string{"engine": engine}, &rechecked)
			if err == nil && !settings.ConfigFromLaunch(launch, record.Snapshot.Settings.ProxyPort).Equal(settings.ConfigFromLaunch(rechecked, record.Snapshot.Settings.ProxyPort)) {
				err = modelSettingsConflict("native overrides changed during frozen validation")
			}
			if err == nil {
				launch = rechecked
			}
		}
		b.engineConfigMu.Lock()
		if err != nil {
			return fail(err)
		}
		if !b.modelWorkerCurrent(worker, source) || launch.Engine != engine {
			return fail(modelSettingsConflict("native source changed during reconciliation"))
		}
		observeModelSettingsLaunch(record, launch)
		if len(validated.Errors) != 0 || validated.Conflict != nil || !validated.Settings.Equal(record.Snapshot.Settings) {
			return fail(modelSettingsConflict("native bindings or build changed; local fresh preview required"))
		}
		actual := settings.ConfigFromLaunch(launch, record.Snapshot.Settings.ProxyPort)
		paused, _ := b.settingsPause()
		resumeNeeded := record.Resume && !launch.Running && !paused
		if !actual.Equal(record.Snapshot.Settings) || resumeNeeded {
			if !actual.Equal(record.Snapshot.Settings) && !actual.Equal(op.NativeBefore) {
				return fail(modelSettingsConflict("native overrides changed to a third state; no replacement was attempted"))
			}
			// A replay may converge an exact unchanged before-state. An exact
			// desired readback skips persistence, but saved resume must still
			// converge through ConfigureLaunch's current activation guards.
			op.Attempted = true
			if err := b.saveModelStageLocked(record, "native-applying"); err != nil {
				if !errors.Is(err, durablefile.ErrReplaced) {
					op.Attempted = false
				}
				return fail(err)
			}
			id := record.Snapshot.RequestID
			b.settingsRelayMu.Lock()
			if b.activeSettings == nil {
				b.activeSettings = map[string]activeSettingsOperation{}
			}
			b.activeSettings[id] = activeSettingsOperation{Engine: engine, Port: record.Snapshot.Settings.ProxyPort}
			b.settingsRelayMu.Unlock()
			paused, _ := b.settingsPause()
			configure := settings.Configure{Engine: engine, Settings: record.Snapshot.Settings.Clone(), OperationID: id, Resume: record.Resume && !paused}
			b.engineConfigMu.Unlock()
			configureErr := modelSettingsCall(ctx, worker, "engine:configure-launch", configure, nil)
			// A response is not persistence evidence, and loss is not no effect.
			err = modelSettingsCall(ctx, worker, "engine:get-launch", map[string]string{"engine": engine}, &launch)
			b.settingsRelayMu.Lock()
			delete(b.activeSettings, id)
			b.settingsRelayMu.Unlock()
			b.engineConfigMu.Lock()
			if err != nil {
				record.Snapshot.EffectiveSettings = nil
				return fail(fmt.Errorf("native outcome uncertain: %w", err))
			}
			if !b.modelWorkerCurrent(worker, source) {
				record.Snapshot.EffectiveSettings = nil
				return fail(modelSettingsConflict("native outcome came from a replaced engine manager"))
			}
			observeModelSettingsLaunch(record, launch)
			if !settings.ConfigFromLaunch(launch, record.Snapshot.Settings.ProxyPort).Equal(record.Snapshot.Settings) {
				if configureErr != nil {
					return fail(configureErr)
				}
				return fail(modelSettingsConflict("native response did not persist the exact reviewed settings"))
			}
		}
		if !b.modelWorkerCurrent(worker, source) {
			return fail(modelSettingsConflict("engine manager changed before native persistence could be confirmed"))
		}
		// Persist desired/applied/observed separately, even if runtime startup
		// failed after exact persistence. Readiness still needs live evidence.
		if err := b.saveModelStageLocked(record, "native-persisted"); err != nil {
			return fail(err)
		}
	}
	if !b.modelWorkerCurrent(worker, source) {
		record.Snapshot.EffectiveSettings = nil
		return fail(modelSettingsConflict("engine manager changed before policy publication"))
	}
	if err := b.commitModelPolicyLocked(record); err != nil {
		return fail(err)
	}
	record.Snapshot.PolicyRevision = op.PolicyRevision
	if err := b.saveModelStageLocked(record, "policy-persisted"); err != nil {
		return fail(err)
	}
	// The policy owner publishes/activates while the admission drain stays set.
	b.engineConfigMu.Unlock()
	b.publishNodeProjection()
	b.activatePolicyNow()
	b.notifyPolicyChanged()
	var final settings.LaunchState
	err := modelSettingsCall(ctx, worker, "engine:get-launch", map[string]string{"engine": engine}, &final)
	b.engineConfigMu.Lock()
	if err != nil {
		record.Snapshot.EffectiveSettings = nil
		return fail(err)
	}
	if b.modelWorkerCurrent(worker, source) {
		observeModelSettingsLaunch(record, final)
	}
	if !b.modelWorkerCurrent(worker, source) || final.Engine != engine || !settings.ConfigFromLaunch(final, record.Snapshot.Settings.ProxyPort).Equal(record.Snapshot.Settings) {
		return fail(modelSettingsConflict("current-source native readback no longer matches the accepted settings"))
	}
	if err := b.checkModelPolicyLocked(record); err != nil {
		return fail(err)
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	paused := r.managementBlockedLocked()
	activation := r.activationLocked(b.policyProxy(), r.current.Revision)
	r.mu.Unlock()
	if activation.Activation != activationActive {
		record.Snapshot.Error, record.Snapshot.UnresolvedError = "Policy activation is pending; engine remains withdrawn.", "Policy activation is pending; engine remains withdrawn."
		if err := b.saveModelStageLocked(record, "activation-pending"); err != nil {
			return fail(err)
		}
		b.publishSettingsLocked()
		return fmt.Errorf("policy activation is pending")
	}
	if paused && final.Running {
		return fail(modelSettingsConflict("Pause is pending native shutdown; engine remains withdrawn"))
	}
	if record.Resume && !paused && !final.Running {
		b.engineConfigMu.Unlock()
		veto, vetoErr := modelSettingsResumeVeto(ctx, worker, engine)
		b.engineConfigMu.Lock()
		paused, _ = b.settingsPause()
		if !b.modelWorkerCurrent(worker, source) {
			return fail(modelSettingsConflict("engine manager changed during activation veto readback"))
		}
		if !paused && (vetoErr != nil || !veto) {
			if vetoErr != nil {
				return fail(vetoErr)
			}
			return fail(fmt.Errorf("settings are persisted but the requested running engine is not ready"))
		}
	}
	if final.ServingReadback != nil && (final.ServingReadback.Status == "mismatch" || final.ServingReadback.Status == "unknown") {
		return fail(modelSettingsConflict("native serving readback is unresolved"))
	}
	if !b.modelWorkerCurrent(worker, source) {
		return fail(modelSettingsConflict("native source changed before completion"))
	}
	prior := record.clone()
	// The final receipt is durable before drain release. "releasing" retains
	// the reservation through an interrupted or unacknowledged release.
	record.Model.Stage, record.Snapshot.CoupledStage = "releasing", "releasing"
	record.Snapshot.Phase, record.Snapshot.Error, record.Snapshot.UnresolvedError = "applying", "", ""
	record.Snapshot.AppliedRevision = record.Snapshot.Revision
	record.Snapshot.Running, record.Snapshot.Serving = final.Running, final.ServingReadback.Clone()
	record.Snapshot.Editable, record.Snapshot.Reason = final.Editable, final.Reason
	receipt := record.Receipts[record.Snapshot.RequestID]
	receipt.Phase = "succeeded"
	record.Receipts[record.Snapshot.RequestID] = receipt
	if err := b.saveEngineSettingsLocked(); err != nil {
		*record = *prior
		return fail(err)
	}
	if op.Change.Load != nil {
		b.engineConfigMu.Unlock()
		_, err = b.pushSwitchDrain(ctx, engine, false)
		b.engineConfigMu.Lock()
		if err != nil {
			return fail(err)
		}
	}
	if !b.modelWorkerCurrent(worker, source) {
		return fail(modelSettingsConflict("native source changed during release"))
	}
	resume := record.Resume
	record.Resume = false
	record.Snapshot.Phase = "succeeded"
	if err := b.saveModelStageLocked(record, "completed"); err != nil {
		record.Resume = resume
		if op.Change.Load != nil {
			b.engineConfigMu.Unlock()
			_, _ = b.pushSwitchDrain(ctx, engine, true)
			b.engineConfigMu.Lock()
		}
		return fail(err)
	}
	r.mu.Lock()
	if op.Change.Load != nil {
		delete(r.drains, engine)
	}
	r.mu.Unlock()
	b.publishSettingsLocked()
	b.signalModelConfig()
	// Periodic advertising will probe current runtime; a receipt never turns a
	// backend healthy or manufactures strict model identity.
	return nil
}

// inspectModelSettings is a read, never a settings snapshot initialization or
// apply. Authorization is fresh at the existing target caller boundary.
func (b *Broker) inspectModelSettings(ctx context.Context, p modelconfig.InspectRequest, caller string) (modelconfig.InspectResult, error) {
	if err := p.Validate(); err != nil {
		return modelconfig.InspectResult{}, &invalidParamsError{err: err}
	}
	ctx, cancel := context.WithTimeout(ctx, settings.CallBudget)
	defer cancel()
	b.engineConfigMu.Lock()
	err := b.authorizeSettingsCallerLocked(caller)
	b.engineConfigMu.Unlock()
	if err != nil {
		return modelconfig.InspectResult{}, err
	}
	p.NodeID = ""
	var raw json.RawMessage
	if err := b.settingsWorkerCall(ctx, "engine:inspect-model", p, &raw); err != nil {
		return modelconfig.InspectResult{}, err
	}
	result, err := modelconfig.DecodeInspectResult(raw)
	if err != nil {
		return modelconfig.InspectResult{}, err
	}
	if result.Engine != p.Engine || result.ModelID != p.ModelID {
		return modelconfig.InspectResult{}, fmt.Errorf("model inspection: worker reply identity mismatch")
	}
	result.NodeID = b.nodeID
	if err := result.Validate(); err != nil {
		return modelconfig.InspectResult{}, err
	}
	return result, nil
}

func (b *Broker) handleModelInspection(msg *Message) {
	p, err := modelconfig.DecodeInspectRequest(msg.Params)
	if err != nil {
		_ = b.codec.RespondError(msg.ID, -32602, err.Error())
		return
	}
	if p.NodeID != "" && p.NodeID != b.nodeID {
		worker := b.getEngineMgr()
		if worker == nil {
			_ = b.codec.RespondError(msg.ID, -32000, "engine manager unavailable")
			return
		}
		err = worker.RelayRequest("engine:remote-inspect-model", msg.Params, func(result json.RawMessage, rpcErr *RPCError, err error) {
			if err != nil {
				_ = b.codec.RespondError(msg.ID, -32000, err.Error())
			} else if rpcErr != nil {
				_ = b.codec.RespondError(msg.ID, rpcErr.Code, rpcErr.Message)
			} else {
				_ = b.codec.Respond(msg.ID, result)
			}
		})
		if err != nil {
			_ = b.codec.RespondError(msg.ID, -32000, err.Error())
		}
		return
	}
	result, err := b.inspectModelSettings(context.Background(), p, "")
	if err != nil {
		_ = b.codec.RespondError(msg.ID, engineSettingsErrorCode(err), err.Error())
	} else {
		_ = b.codec.Respond(msg.ID, result)
	}
}
