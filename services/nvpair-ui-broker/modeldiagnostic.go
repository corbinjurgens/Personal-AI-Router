// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"sync"
	"time"

	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

const diagnosticPreviewTTL = 5 * time.Minute
const diagnosticRunLimit = 20 * time.Minute
const diagnosticSampleInterval = 2 * time.Second

// Manager cleanup has a two-minute physical-stop budget. Leave time for the
// joined reply, durable record and release instead of timing it out at 5 s.
const diagnosticCleanupLimit = 3 * time.Minute

// Includes acceptance reconciliation and the owner's joined cleanup, and fits
// inside the desktop's 210-second cancellation envelope.
const diagnosticCancelLimit = 200 * time.Second

// This private observation is deliberately outside the public engine:* relay.
// The manager must observe the same effects used by its compare-and-reserve,
// without waking an engine or running a native helper. Unsupported observation
// fails closed; the broker never manufactures an effects fence from its cache.
const diagnosticObserveMethod = "internal:diagnostic-observe"

type diagnosticObservation struct {
	Identity              modelconfig.DiagnosticIdentity   `json:"identity"`
	PlannedPerSlotContext int32                            `json:"plannedPerSlotContext"`
	PlannedSlots          int32                            `json:"plannedSlots"`
	EffectsFence          string                           `json:"effectsFence"`
	UnloadModels          []modelconfig.DiagnosticModelRef `json:"unloadModels"`
	ReloadedAfter         []modelconfig.DiagnosticModelRef `json:"reloadedAfter"`
	DrainEngines          []string                         `json:"drainEngines"`
	StartsEngine          bool                             `json:"startsEngine"`
	Supported             bool                             `json:"supported"`
}

type diagnosticReserveRequest struct {
	RunID            string                    `json:"runId"`
	Engine           string                    `json:"engine"`
	ModelID          string                    `json:"modelId"`
	ProfileKey       string                    `json:"profileKey"`
	ConfigurationKey string                    `json:"configurationKey,omitempty"`
	ExecutionKey     string                    `json:"executionKey,omitempty"`
	EffectsFence     string                    `json:"effectsFence"`
	TargetMemory     *modelconfig.TargetMemory `json:"targetMemory,omitempty"`
	Hardware         *diagnosticHardware       `json:"hardware,omitempty"`
}

type diagnosticAuthority struct {
	RunID string `json:"runId"`
	Token string `json:"token"`
}

type diagnosticRunRequest struct {
	diagnosticAuthority
	Engine           string `json:"engine"`
	ModelID          string `json:"modelId"`
	ProfileKey       string `json:"profileKey"`
	ConfigurationKey string `json:"configurationKey,omitempty"`
	ExecutionKey     string `json:"executionKey,omitempty"`
	Low              int32  `json:"low,omitempty"`
	High             int32  `json:"high,omitempty"`
	Generate         int32  `json:"generate"`
	Deadline         string `json:"deadline"`
}

// The operation/checkpoint owner supplies affirmative WorkStopped evidence.
// A missing field, timeout, "none", or terminal-looking phase is not that proof.
type diagnosticManagerStatus struct {
	State       string                          `json:"state"`
	Token       string                          `json:"token"`
	Run         *modelconfig.DiagnosticRun      `json:"run,omitempty"`
	WorkStopped bool                            `json:"workStopped"`
	Identity    *modelconfig.DiagnosticIdentity `json:"identity,omitempty"`
	Markers     map[string]time.Time            `json:"markers,omitempty"`
	Low         *modelconfig.MeasuredPhase      `json:"low,omitempty"`
	High        *modelconfig.MeasuredPhase      `json:"high,omitempty"`
	Marker      string
	MarkerAt    time.Time
	Observation *diagnosticRuntimeObservation
}

type diagnosticRuntimeObservation struct {
	PerSlotContext int32  `json:"perSlotContext"`
	Slots          int32  `json:"slots"`
	BuildInfo      string `json:"buildInfo,omitempty"`
	NativeBuildID  string `json:"nativeBuildId,omitempty"`
	// Requires a target-owned mapping, never inference from requested switches.
	PoolPlacement []modelconfig.DiagnosticPlacement `json:"poolPlacement,omitempty"`
}

func (s *diagnosticManagerStatus) UnmarshalJSON(raw []byte) error {
	var wire struct {
		Run         *modelconfig.DiagnosticRun      `json:"run"`
		State       string                          `json:"state"`
		Token       string                          `json:"token"`
		WorkStopped bool                            `json:"workStopped"`
		Identity    *modelconfig.DiagnosticIdentity `json:"identity,omitempty"`
		Markers     map[string]time.Time            `json:"markers,omitempty"`
		Marker      string                          `json:"marker,omitempty"`
		MarkerAt    string                          `json:"markerAt,omitempty"`
		Observation *diagnosticRuntimeObservation   `json:"observation,omitempty"`
		Low         *modelconfig.MeasuredPhase      `json:"low,omitempty"`
		High        *modelconfig.MeasuredPhase      `json:"high,omitempty"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return err
	}
	*s = diagnosticManagerStatus{State: wire.State, Token: wire.Token, WorkStopped: wire.WorkStopped, Identity: wire.Identity, Markers: wire.Markers, Marker: wire.Marker, Observation: wire.Observation, Low: wire.Low, High: wire.High}
	s.Run = wire.Run
	if wire.MarkerAt != "" {
		at, err := time.Parse(time.RFC3339Nano, wire.MarkerAt)
		if err != nil {
			return err
		}
		s.MarkerAt = at
	}
	return nil
}

type diagnosticReview struct {
	preview    modelconfig.DiagnosticPreview
	effects    string
	generation uint64
	policy     string
	worker     *rpcWorker
	expires    time.Time
	caller     string
	source     *engineManagerSource
}

type modelDiagnosticRuntime struct {
	mu           sync.Mutex
	reconcileMu  sync.Mutex
	previews     map[string]diagnosticReview
	active       *brokerDiagnostic
	last         *modelconfig.DiagnosticRun
	now          func() time.Time
	relayCancels map[diagnosticRelayKey]context.CancelFunc
}

// One goroutine owns the ledger. It is published before the first acquisition.
// Cleanup retries run after done closes, serialized by cleanupMu. The manager's
// checkpoint is the only durable owner; this structure is never journalled.
type brokerDiagnostic struct {
	runID               string
	started             time.Time
	cleanupMu           sync.Mutex
	run                 modelconfig.DiagnosticRun // diagnostics.mu
	review              diagnosticReview
	worker              *rpcWorker
	cancel              context.CancelFunc
	done                chan struct{}
	wake                chan struct{}
	lease               *buildApplySettingsLease
	drain               *nodeDrainLease
	token               string
	reserveAttempted    bool
	reserveUncertain    bool
	reserveRequest      diagnosticReserveRequest
	runAttempted        bool
	runUncertain        bool
	runRequest          diagnosticRunRequest
	accepted            bool
	recorded            bool
	stopped             bool
	reservationReleased bool
	releaseAttempted    bool
	markers             map[string]time.Time // diagnostics.mu; manager marker notifications
	samples             diagnosticSamples
	status              diagnosticManagerStatus
}

func (b *Broker) diagnosticNow() time.Time {
	if b.diagnostics.now != nil {
		return b.diagnostics.now()
	}
	return time.Now()
}

func diagnosticError(reason string) error { return modelSettingsConflict(reason) }

// Saved pause, not managementBlockedLocked, is the consent gate. Configured
// pause permits measurement; its availability and saved pause remain untouched.
func (b *Broker) diagnosticGate() (uint64, string, error) {
	return b.diagnosticRequestGate(context.Background())
}

func (b *Broker) diagnosticRequestGate(ctx context.Context) (generation uint64, policy string, err error) {
	err = b.withDiagnosticGate(ctx, func(g uint64, p string) error {
		generation, policy = g, p
		return nil
	})
	return
}

// No I/O or lifecycle wait belongs in accept. Holding the policy and worker
// pair through publication prevents a retired relay from accepting new work.
func (b *Broker) withDiagnosticGate(ctx context.Context, accept func(uint64, string) error) error {
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if err := b.loadEngineSettingsLocked(); err != nil {
		return err
	}
	if b.buildApplyMaintenance.active != nil || len(b.buildApplyMaintenance.settings) != 0 || len(b.activeSettings) != 0 {
		return diagnosticError("node maintenance is active")
	}
	for _, record := range b.engineSettings {
		if record != nil && (record.Snapshot.Phase == "applying" || record.Model != nil && record.Model.Stage != "completed") {
			return diagnosticError("model settings are being applied")
		}
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	if err := b.diagnosticAuthorityLocked(ctx, r); err != nil {
		return err
	}
	if r.pendingPauseRequested || r.buildApplyOperation != "" {
		return diagnosticError("node availability or maintenance is changing")
	}
	if r.current.PauseMode == nodepolicy.PauseImmediate {
		return diagnosticError("This node is halted. Resume it before measuring memory.")
	}
	if r.recoveryRequired {
		return diagnosticError("node policy recovery is pending")
	}
	return accept(b.buildApplyMaintenance.generation, r.current.Revision)
}

func (b *Broker) observeDiagnostic(ctx context.Context, p modelconfig.DiagnosticPreviewRequest) (diagnosticObservation, error) {
	var observation diagnosticObservation
	// The private target is already pinned to this node. Do not forward the
	// public self alias as if it were a peer selection to the manager.
	p.NodeID = ""
	memory := b.modelTargetMemory()
	hardware, err := b.diagnosticTargetHardware(ctx, memory)
	if err != nil {
		return observation, err
	}
	err = modelSettingsCall(ctx, b.getEngineMgr(), diagnosticObserveMethod, struct {
		Request  modelconfig.DiagnosticPreviewRequest `json:"request"`
		Memory   *modelconfig.TargetMemory            `json:"memory"`
		Hardware *diagnosticHardware                  `json:"hardware"`
	}{p, memory, hardware}, &observation)
	if err != nil {
		return observation, err
	}
	if !observation.Supported || observation.EffectsFence == "" || len(observation.DrainEngines) == 0 {
		return observation, diagnosticError("diagnostic observation unavailable for this engine family")
	}
	if p.Engine != "llamacpp" {
		return observation, diagnosticError("unsupported diagnostic engine")
	}
	if !reflect.DeepEqual(diagnosticIdentityHardware(observation.Identity), hardware) {
		return observation, diagnosticError("manager observed different target hardware")
	}
	// Consent includes the exact broker-owned proxy drain catalog as well as
	// the manager's execution hold, including descriptor-backed facades.
	for _, profile := range b.frontedEngineProfiles() {
		observation.DrainEngines = append(observation.DrainEngines, profile.Name)
	}
	slices.Sort(observation.DrainEngines)
	observation.DrainEngines = slices.Compact(observation.DrainEngines)
	observation.ReloadedAfter = b.diagnosticResidentReloads()
	return observation, nil
}

func (b *Broker) previewDiagnostic(ctx context.Context, p modelconfig.DiagnosticPreviewRequest) (modelconfig.DiagnosticPreview, error) {
	if err := p.Validate(); err != nil {
		return modelconfig.DiagnosticPreview{}, err
	}
	if p.NodeID != "" && p.NodeID != b.nodeID {
		return modelconfig.DiagnosticPreview{}, diagnosticError("unsupported: local diagnostics only")
	}
	generation, policy, err := b.diagnosticRequestGate(ctx)
	if err != nil {
		return modelconfig.DiagnosticPreview{}, err
	}
	b.diagnostics.mu.Lock()
	busy := b.diagnostics.active != nil
	b.diagnostics.mu.Unlock()
	if busy {
		return modelconfig.DiagnosticPreview{}, diagnosticError("a diagnostic is active")
	}
	if _, err := b.diagnosticMemory(); err != nil {
		return modelconfig.DiagnosticPreview{}, err
	}
	worker := b.getEngineMgr()
	observation, err := b.observeDiagnostic(ctx, p)
	if err != nil {
		return modelconfig.DiagnosticPreview{}, err
	}
	now := b.diagnosticNow()
	preview := modelconfig.DiagnosticPreview{DiagnosticPreviewRequest: p, DiagnosticIdentity: observation.Identity.Clone(), PreviewID: settingsID(), GenerateTokens: 128, TimeLimitSeconds: 1200,
		PlannedPerSlotContext: observation.PlannedPerSlotContext, PlannedSlots: observation.PlannedSlots,
		UnloadModels: observation.UnloadModels, ReloadedAfter: observation.ReloadedAfter, DrainEngines: slices.Clone(observation.DrainEngines), StartsEngine: observation.StartsEngine, ExpiresAt: now.Add(diagnosticPreviewTTL).Format(time.RFC3339Nano)}
	if preview.PlannedPerSlotContext != 0 {
		preview.HighPromptTokens = preview.PlannedPerSlotContext - 144
		if preview.HighPromptTokens < 64 {
			return modelconfig.DiagnosticPreview{}, diagnosticError("context-too-small")
		}
		preview.LowPromptTokens = min(int32(512), preview.HighPromptTokens/2)
	} else {
		preview.Warnings = append(preview.Warnings, "Prompt targets and realized allocation will be determined after the consented load.")
	}
	if preview.PlannedSlots > 8 {
		return modelconfig.DiagnosticPreview{}, diagnosticError("more than eight slots are unsupported")
	}
	preview.Warnings = append(preview.Warnings, b.diagnosticMemoryWarnings()...)
	preview.Warnings = append(preview.Warnings, "Other programs may change memory use; their activity is included in these samples.", "This fills the configuration's context and can stop the engine if it does not fit.")
	if err := preview.Validate(); err != nil {
		return modelconfig.DiagnosticPreview{}, err
	}
	err = b.withDiagnosticGate(ctx, func(currentGeneration uint64, currentPolicy string) error {
		if generation != currentGeneration || policy != currentPolicy || worker != b.engineMgr {
			return diagnosticError("node changed during preview")
		}
		b.diagnostics.mu.Lock()
		defer b.diagnostics.mu.Unlock()
		if b.diagnostics.active != nil {
			return diagnosticError("a diagnostic is active")
		}
		if b.diagnostics.previews == nil {
			b.diagnostics.previews = make(map[string]diagnosticReview)
		}
		for id, old := range b.diagnostics.previews {
			if !now.Before(old.expires) {
				delete(b.diagnostics.previews, id)
			}
		}
		if len(b.diagnostics.previews) >= 16 {
			oldest := ""
			for id, old := range b.diagnostics.previews {
				if oldest == "" || old.expires.Before(b.diagnostics.previews[oldest].expires) {
					oldest = id
				}
			}
			delete(b.diagnostics.previews, oldest)
		}
		var source *engineManagerSource
		if peer := diagnosticPeer(ctx); peer != nil {
			source = peer.source
		}
		b.diagnostics.previews[preview.PreviewID] = diagnosticReview{preview: preview.Clone(), effects: observation.EffectsFence, generation: generation, policy: policy, worker: worker, expires: now.Add(diagnosticPreviewTTL), caller: diagnosticCaller(ctx), source: source}
		return nil
	})
	return preview, err
}

func (b *Broker) startDiagnostic(ctx context.Context, p modelconfig.DiagnosticStartRequest) (modelconfig.DiagnosticRun, error) {
	if err := p.Validate(); err != nil {
		return modelconfig.DiagnosticRun{}, err
	}
	if p.NodeID != "" && p.NodeID != b.nodeID {
		return modelconfig.DiagnosticRun{}, diagnosticError("unsupported: local diagnostics only")
	}
	generation, policy, err := b.diagnosticRequestGate(ctx)
	if err != nil {
		return modelconfig.DiagnosticRun{}, err
	}
	b.diagnostics.mu.Lock()
	review, found := b.diagnostics.previews[p.PreviewID]
	b.diagnostics.mu.Unlock()
	var source *engineManagerSource
	if peer := diagnosticPeer(ctx); peer != nil {
		source = peer.source
	}
	if !found || review.source != source || review.caller != diagnosticCaller(ctx) || !b.diagnosticNow().Before(review.expires) || review.preview.DiagnosticPreviewRequest != p.DiagnosticPreviewRequest || generation != review.generation || policy != review.policy || b.getEngineMgr() != review.worker {
		return modelconfig.DiagnosticRun{}, diagnosticError("stale diagnostic preview; preview again")
	}
	observation, err := b.observeDiagnostic(ctx, p.DiagnosticPreviewRequest)
	if err != nil {
		return modelconfig.DiagnosticRun{}, err
	}
	if observation.EffectsFence != review.effects || !reflect.DeepEqual(observation.Identity, review.preview.DiagnosticIdentity) || observation.PlannedPerSlotContext != review.preview.PlannedPerSlotContext || observation.PlannedSlots != review.preview.PlannedSlots || !reflect.DeepEqual(observation.UnloadModels, review.preview.UnloadModels) || !reflect.DeepEqual(observation.ReloadedAfter, review.preview.ReloadedAfter) || !reflect.DeepEqual(observation.DrainEngines, review.preview.DrainEngines) || observation.StartsEngine != review.preview.StartsEngine {
		return modelconfig.DiagnosticRun{}, diagnosticError("diagnostic effects changed; preview again")
	}
	if _, err := b.diagnosticMemory(); err != nil {
		return modelconfig.DiagnosticRun{}, err
	}
	// A caller disappearing after acceptance must not abandon cleanup.
	runCtx, cancel := context.WithTimeout(context.Background(), diagnosticRunLimit)
	op := &brokerDiagnostic{review: review, worker: review.worker, cancel: cancel, done: make(chan struct{}), wake: make(chan struct{}, 1)}
	op.started = b.diagnosticNow()
	op.run = modelconfig.DiagnosticRun{RunID: settingsID(), Engine: p.Engine, ModelID: p.ModelID, ProfileKey: p.ProfileKey, ConfigurationKey: review.preview.ConfigurationKey, ExecutionKey: review.preview.ExecutionKey, Phase: "reserving", Outcome: "running", StartedAt: op.started.Format(time.RFC3339Nano)}
	op.runID = op.run.RunID
	var result modelconfig.DiagnosticRun
	err = b.withDiagnosticGate(ctx, func(currentGeneration uint64, currentPolicy string) error {
		if currentGeneration != generation || currentPolicy != policy || b.engineMgr != review.worker {
			return diagnosticError("node changed before diagnostic acceptance")
		}
		b.diagnostics.mu.Lock()
		defer b.diagnostics.mu.Unlock()
		if b.diagnostics.active != nil {
			return diagnosticError("a diagnostic is active")
		}
		delete(b.diagnostics.previews, p.PreviewID)
		b.diagnostics.active = op
		result = op.run.Clone()
		return nil
	})
	if err != nil {
		cancel()
		return modelconfig.DiagnosticRun{}, err
	}
	b.publishDiagnostic(result)
	go b.executeDiagnostic(runCtx, op)
	return result, nil
}

func (b *Broker) diagnosticStatus(runID string) (*modelconfig.DiagnosticRun, error) {
	b.diagnostics.mu.Lock()
	defer b.diagnostics.mu.Unlock()
	if op := b.diagnostics.active; op != nil && (runID == "" || runID == op.run.RunID) {
		b.updateDiagnosticElapsedLocked(op)
		run := op.run.Clone()
		return &run, nil
	}
	if b.diagnostics.last != nil && (runID == "" || runID == b.diagnostics.last.RunID) {
		run := b.diagnostics.last.Clone()
		return &run, nil
	}
	return nil, nil
}

func (b *Broker) setDiagnosticRun(op *brokerDiagnostic, run modelconfig.DiagnosticRun) {
	b.diagnostics.mu.Lock()
	if b.diagnostics.active != op {
		b.diagnostics.mu.Unlock()
		return
	}
	if run.RunID != op.runID || run.Engine != op.run.Engine || run.ModelID != op.run.ModelID || run.ProfileKey != op.run.ProfileKey || run.ConfigurationKey != op.run.ConfigurationKey || run.ExecutionKey != op.run.ExecutionKey {
		b.diagnostics.mu.Unlock()
		return
	}
	// The public run starts before manager acceptance. Preserve that clock
	// origin and never regress when adopting the manager's later-started clock.
	run.StartedAt = op.run.StartedAt
	run.ElapsedMS = max(run.ElapsedMS, op.run.ElapsedMS)
	if run.Phase == "reserving" && op.run.Phase == "draining" {
		// Manager acceptance starts its operation in reserving, after the
		// broker has already completed draining under the same public run.
		run.Phase = "draining"
	}
	if op.run.Result != nil && op.run.Result.RunID == run.RunID && op.run.Result.Outcome == run.Outcome {
		run.Result = op.run.Result
	}
	op.run = run.Clone()
	b.updateDiagnosticElapsedLocked(op)
	run = op.run.Clone()
	b.diagnostics.mu.Unlock()
	b.publishDiagnostic(run)
}

func (b *Broker) updateDiagnosticElapsedLocked(op *brokerDiagnostic) {
	if op.started.IsZero() {
		op.started, _ = time.Parse(time.RFC3339Nano, op.run.StartedAt)
	}
	if !op.started.IsZero() {
		op.run.ElapsedMS = max(op.run.ElapsedMS, b.diagnosticNow().Sub(op.started).Milliseconds())
	}
}

func (b *Broker) publishDiagnostic(run modelconfig.DiagnosticRun) {
	b.engineMu.Lock()
	subscribed := b.engineSubscribed
	b.engineMu.Unlock()
	if !subscribed {
		return
	}
	b.workersMu.Lock()
	source := b.engineMgrSource
	b.workersMu.Unlock()
	if source != nil {
		b.queueEngineNotification(source, "engine:diagnostic-changed", settingsJSON(run))
	}
}

func (b *Broker) diagnosticCall(ctx context.Context, op *brokerDiagnostic, method string, params, result any) error {
	// Never accept stop evidence from a retired manager incarnation.
	if !b.diagnosticWorkerCurrent(op.worker) {
		return diagnosticError("engine manager replaced; reconciliation required")
	}
	bound := 5 * time.Second
	if method == "engine:diagnostic-abort" {
		bound = diagnosticCleanupLimit - 15*time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, bound)
	defer cancel()
	if err := modelSettingsCall(ctx, op.worker, method, params, result); err != nil {
		return err
	}
	if !b.diagnosticWorkerCurrent(op.worker) {
		return diagnosticError("engine manager replaced during diagnostic call")
	}
	return nil
}

func (b *Broker) diagnosticWorkerCurrent(worker *rpcWorker) bool {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	return worker != nil && b.engineMgr == worker
}

func (b *Broker) readDiagnosticManager(ctx context.Context, op *brokerDiagnostic) (diagnosticManagerStatus, error) {
	var status diagnosticManagerStatus
	err := b.diagnosticCall(ctx, op, "engine:diagnostic-run-status", modelconfig.DiagnosticRunRef{RunID: op.runID}, &status)
	if err != nil {
		return status, err
	}
	if err := validateDiagnosticManagerStatus(status, op.runID); err != nil {
		return status, err
	}
	b.diagnostics.mu.Lock()
	expected := op.run.Clone()
	b.diagnostics.mu.Unlock()
	if run := status.Run; status.State != "none" && run != nil && (run.Engine != expected.Engine || run.ModelID != expected.ModelID || run.ProfileKey != expected.ProfileKey || run.ConfigurationKey != expected.ConfigurationKey || run.ExecutionKey != expected.ExecutionKey) {
		return status, diagnosticError("manager diagnostic target mismatch")
	}
	b.observeDiagnosticMarker(op, status)
	return status, nil
}

func validateDiagnosticManagerStatus(status diagnosticManagerStatus, runID string) error {
	if !slices.Contains([]string{"none", "reserved", "accepted", "running", "cleanup-failed", "cleanup-complete", "terminal", "pending"}, status.State) {
		return diagnosticError("unknown manager diagnostic state")
	}
	if status.State == "none" {
		// The manager's tagged embedded value emits a zero run for none.
		if status.WorkStopped || status.Run != nil && !reflect.DeepEqual(*status.Run, modelconfig.DiagnosticRun{}) {
			return diagnosticError("inconsistent empty diagnostic status")
		}
		return nil
	}
	if status.Run == nil || status.Run.RunID == "" || runID != "" && status.Run.RunID != runID {
		return diagnosticError("manager diagnostic identity mismatch")
	}
	return nil
}

func (b *Broker) observeDiagnosticMarker(op *brokerDiagnostic, status diagnosticManagerStatus) {
	b.diagnostics.mu.Lock()
	defer b.diagnostics.mu.Unlock()
	if op.markers == nil {
		op.markers = make(map[string]time.Time)
	}
	for marker, at := range status.Markers {
		if !at.IsZero() {
			if _, exists := op.markers[marker]; !exists {
				op.markers[marker] = at
			}
		}
	}
	if status.Marker != "" && !status.MarkerAt.IsZero() {
		if _, exists := op.markers[status.Marker]; !exists {
			op.markers[status.Marker] = status.MarkerAt
		}
	}
}

func (b *Broker) diagnosticMarkers(op *brokerDiagnostic) map[string]time.Time {
	b.diagnostics.mu.Lock()
	defer b.diagnostics.mu.Unlock()
	out := make(map[string]time.Time, len(op.markers))
	for marker, at := range op.markers {
		out[marker] = at
	}
	return out
}

func (b *Broker) executeDiagnostic(ctx context.Context, op *brokerDiagnostic) {
	op.cleanupMu.Lock()
	defer op.cleanupMu.Unlock()
	defer close(op.done)
	defer op.cancel()
	err := b.acquireDiagnostic(ctx, op)
	if err == nil {
		err = b.sampleDiagnostic(ctx, op)
	}
	if err != nil {
		b.diagnosticFailure(op, err)
	}
	cleanup, cancel := context.WithTimeout(context.Background(), diagnosticCleanupLimit)
	defer cancel()
	if err := b.cleanupDiagnostic(cleanup, op); err != nil {
		b.retainDiagnostic(op, err)
	}
}

func (b *Broker) acquireDiagnostic(ctx context.Context, op *brokerDiagnostic) error {
	generation, policy, err := b.diagnosticGate()
	if err != nil {
		return err
	}
	if generation != op.review.generation || policy != op.review.policy {
		return diagnosticError("diagnostic preview changed before acquisition")
	}
	lease, err := b.reserveBuildApplySettings()
	if err != nil {
		return err
	}
	op.lease = lease
	if err := ctx.Err(); err != nil {
		return err
	}
	preview := op.review.preview
	params := diagnosticReserveRequest{RunID: op.runID, Engine: preview.Engine, ModelID: preview.ModelID, ProfileKey: preview.ProfileKey, ConfigurationKey: preview.ConfigurationKey, ExecutionKey: preview.ExecutionKey, EffectsFence: op.review.effects, TargetMemory: b.modelTargetMemory(), Hardware: diagnosticIdentityHardware(preview.DiagnosticIdentity)}
	op.reserveRequest = params
	op.reserveAttempted = true // Before dispatch, including an uncertain reply.
	op.reserveUncertain = true
	var reserved struct {
		RunID string `json:"runId"`
		Token string `json:"token"`
		State string `json:"state"`
	}
	err = b.diagnosticCall(ctx, op, "engine:diagnostic-reserve", params, &reserved)
	var refusal *settingsWorkerError
	if err == nil || errors.As(err, &refusal) {
		op.reserveUncertain = false
	}
	if err != nil {
		if refusal != nil {
			return err // A definitive refusal has no delayed acceptance to settle.
		}
		// Repeating the same run ID is the manager's idempotent acceptance
		// barrier. A cancelled request uses a fresh bounded cleanup context.
		reconcile, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		status, statusErr := b.readDiagnosticManager(reconcile, op)
		if statusErr != nil {
			return errors.Join(err, statusErr)
		}
		if status.State == "none" {
			if retryErr := b.diagnosticCall(reconcile, op, "engine:diagnostic-reserve", params, &reserved); retryErr != nil {
				return errors.Join(err, retryErr)
			}
		} else if status.State == "reserved" {
			// The Task2 response is idempotent and is the current source of the
			// token even when status does not expose that private authority.
			if retryErr := b.diagnosticCall(reconcile, op, "engine:diagnostic-reserve", params, &reserved); retryErr != nil {
				return errors.Join(err, retryErr)
			}
		} else {
			reserved.RunID, reserved.Token = op.runID, status.Token
		}
	}
	if reserved.RunID != op.runID || reserved.Token == "" {
		return diagnosticError("reservation reply lacks authority")
	}
	op.reserveUncertain = false
	op.token = reserved.Token
	if err := ctx.Err(); err != nil {
		return err
	}
	b.diagnostics.mu.Lock()
	draining := op.run.Clone()
	b.diagnostics.mu.Unlock()
	draining.Phase = "draining"
	b.setDiagnosticRun(op, draining)
	op.drain, err = b.drainNode(ctx)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	var accepted diagnosticManagerStatus
	op.runRequest = diagnosticRunRequest{diagnosticAuthority{op.runID, op.token}, preview.Engine, preview.ModelID, preview.ProfileKey, preview.ConfigurationKey, preview.ExecutionKey, preview.LowPromptTokens, preview.HighPromptTokens, 128, b.diagnosticNow().Add(diagnosticRunLimit - time.Second).Format(time.RFC3339Nano)}
	op.runAttempted, op.runUncertain = true, true
	err = b.diagnosticCall(ctx, op, "engine:diagnostic-run", op.runRequest, &accepted)
	if err == nil || errors.As(err, &refusal) {
		op.runUncertain = false
	}
	if err != nil {
		reconcile, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		accepted, err = b.readDiagnosticManager(reconcile, op)
		if err != nil {
			return err
		}
	}
	if accepted.Run == nil || accepted.Run.RunID != op.runID || accepted.State == "reserved" || accepted.State == "none" {
		return diagnosticError("diagnostic run was not accepted")
	}
	if err := validateDiagnosticManagerStatus(accepted, op.runID); err != nil {
		return err
	}
	op.accepted, op.status = true, accepted
	op.runUncertain = false
	b.observeDiagnosticMarker(op, accepted)
	b.setDiagnosticRun(op, *accepted.Run)
	return nil
}

func (b *Broker) diagnosticFailure(op *brokerDiagnostic, err error) {
	b.diagnostics.mu.Lock()
	defer b.diagnostics.mu.Unlock()
	if op.run.Outcome == "running" {
		op.run.FailedPhase, op.run.Outcome, op.run.Reason = op.run.Phase, "failed", "Diagnostic interrupted; cleanup is required."
	}
	if errors.Is(err, context.Canceled) {
		op.run.Outcome, op.run.Reason = "cancelled", "Measurement cancelled."
	}
}

func (b *Broker) retainDiagnostic(op *brokerDiagnostic, err error) {
	b.diagnostics.mu.Lock()
	op.run.Outcome, op.run.Reason = "cleanup-failed", "Cleanup could not be confirmed. Retry cleanup."
	// A successfully stored record keeps its own original outcome. Do not
	// publish a run/result pair whose outcomes disagree during release retry.
	if op.run.Result != nil && op.run.Result.Outcome != op.run.Outcome {
		op.run.Result = nil
	}
	run := op.run.Clone()
	b.diagnostics.mu.Unlock()
	b.publishDiagnostic(run)
}

// Even a pre-run failure reconciles reservation acceptance before releasing.
// Abort is a join, never a fire-and-forget notification. Record failures retain
// the ledger for retry, so evidence is persisted before admission is reopened.
func (b *Broker) cleanupDiagnostic(ctx context.Context, op *brokerDiagnostic) error {
	if op.reserveAttempted {
		status, err := b.readDiagnosticManager(ctx, op)
		if err != nil {
			return err
		}
		if status.State == "none" && op.releaseAttempted && op.stopped && (op.recorded || !op.accepted) {
			// The explicit release's reply was lost, after physical cleanup and
			// recording. The same manager now confirms no reservation remains.
			op.reservationReleased = true
		}
		if op.reserveUncertain || status.State == "reserved" && op.token == "" {
			var reserved struct {
				RunID string `json:"runId"`
				Token string `json:"token"`
			}
			err := b.diagnosticCall(ctx, op, "engine:diagnostic-reserve", op.reserveRequest, &reserved)
			var refusal *settingsWorkerError
			if err != nil && !(errors.As(err, &refusal) && status.State == "none") {
				return err
			}
			if err == nil {
				if reserved.RunID != op.runID || reserved.Token == "" {
					return diagnosticError("uncertain reservation has no authority")
				}
				op.token = reserved.Token
			}
			op.reserveUncertain = false
			status, err = b.readDiagnosticManager(ctx, op)
			if err != nil {
				return err
			}
		}
		if op.runUncertain && status.State == "reserved" {
			// Resolve a queued/lost acceptance with the exact idempotent request
			// before interpreting reserved as permission to release.
			var accepted diagnosticManagerStatus
			if err := b.diagnosticCall(ctx, op, "engine:diagnostic-run", op.runRequest, &accepted); err != nil {
				var refusal *settingsWorkerError
				if !errors.As(err, &refusal) {
					return err
				}
				// A definitive refusal settles the delayed acceptance barrier.
				accepted, err = b.readDiagnosticManager(ctx, op)
				if err != nil {
					return err
				}
			}
			status = accepted
			if status.Run == nil || status.Run.RunID != op.runID {
				return diagnosticError("uncertain run acceptance identity mismatch")
			}
			op.accepted, op.runUncertain = status.State != "reserved", false
		}
		if op.runAttempted && status.State == "none" && !op.reservationReleased {
			return diagnosticError("attempted run disappeared without cleanup proof")
		}
		if op.runAttempted && status.State != "reserved" && status.State != "none" {
			op.accepted, op.runUncertain = true, false
		}
		if status.State != "none" {
			if status.Token != "" {
				op.token = status.Token
			}
			if op.token == "" {
				return diagnosticError("pending diagnostic has no cleanup authority")
			}
			if !status.WorkStopped && status.State != "reserved" || status.State == "cleanup-failed" {
				if err := b.diagnosticCall(ctx, op, "engine:diagnostic-abort", diagnosticAuthority{op.runID, op.token}, &status); err != nil {
					return err
				}
				// A join response must still identify this operation.
				if err := validateDiagnosticManagerStatus(status, op.runID); err != nil || status.State == "none" {
					return diagnosticError("abort reply identity mismatch")
				}
			}
			if !status.WorkStopped || status.State == "cleanup-failed" {
				return diagnosticError("diagnostic work is not confirmed stopped")
			}
			op.stopped, op.status = true, status
			if op.accepted && status.Run != nil {
				b.setDiagnosticRun(op, *status.Run)
			}
		} else if op.accepted && !op.reservationReleased {
			return diagnosticError("accepted diagnostic disappeared without cleanup proof")
		}
	}
	if op.accepted && !op.recorded {
		if err := b.recordDiagnostic(ctx, op); err != nil {
			return err
		}
		op.recorded = true
	}
	if op.drain != nil {
		if err := op.drain.clear(ctx); err != nil {
			return err
		}
	}
	if op.token != "" && !op.reservationReleased {
		op.releaseAttempted = true
		if err := b.diagnosticCall(ctx, op, "engine:diagnostic-release", diagnosticAuthority{op.runID, op.token}, nil); err != nil {
			// Physical cleanup is proved, but a failed release must still keep
			// the node drained until reconciliation settles its authority.
			hold, cancel := context.WithTimeout(context.Background(), proxyCallTimeout)
			defer cancel()
			return errors.Join(err, b.holdDiagnosticDrains(hold, op))
		}
		op.reservationReleased = true
		op.token = ""
	}
	if op.drain != nil {
		op.drain.releaseSettings()
		op.drain = nil
	}
	if op.lease != nil {
		op.lease.Release()
		op.lease = nil
	}
	b.diagnostics.mu.Lock()
	var completed *modelconfig.DiagnosticRun
	if b.diagnostics.active == op {
		b.updateDiagnosticElapsedLocked(op)
		result := op.run.Clone()
		if result.Outcome == "running" {
			result.Outcome = "cancelled"
		}
		result.Phase = "done"
		b.diagnostics.last = &result
		completed = &result
		b.diagnostics.active = nil
	}
	b.diagnostics.mu.Unlock()
	if completed != nil {
		b.publishDiagnostic(*completed)
	}
	b.signalModelConfig()
	return nil
}

func (b *Broker) cancelDiagnostic(ctx context.Context, runID string) (*modelconfig.DiagnosticRun, error) {
	var op *brokerDiagnostic
	err := b.withDiagnosticAuthority(ctx, true, func() error {
		b.diagnostics.mu.Lock()
		defer b.diagnostics.mu.Unlock()
		op = b.diagnostics.active
		if op != nil && (runID == "" || op.run.RunID == runID) {
			op.cancel()
		} else {
			op = nil
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if op == nil {
		return b.diagnosticStatus(runID)
	}
	select {
	case <-op.done:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	op.cleanupMu.Lock()
	err = b.cleanupDiagnostic(ctx, op)
	if err != nil {
		b.retainDiagnostic(op, err)
	}
	op.cleanupMu.Unlock()
	result, _ := b.diagnosticStatus(runID)
	return result, err
}

func (b *Broker) handleModelDiagnostic(msg *Message) {
	bound := diagnosticCleanupLimit
	if msg.Method == "engine:diagnostic-cancel" {
		bound = diagnosticCancelLimit
		var target modelconfig.DiagnosticRunRef
		if modelconfig.DecodeStrict(msg.Params, &target) == nil && target.NodeID != "" && target.NodeID != b.nodeID {
			// Target 200 s, peer relay/transport 205 s, origin 208 s, desktop
			// 210 s. Leave each owner time to return its joined outcome.
			bound = 208 * time.Second
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), bound)
	defer cancel()
	result, err := b.dispatchDiagnostic(ctx, msg.Method, msg.Params, true)
	if err != nil {
		_ = b.codec.RespondError(msg.ID, engineSettingsErrorCode(err), err.Error())
	} else {
		_ = b.codec.Respond(msg.ID, result)
	}
}

func (b *Broker) diagnosticNotification(source *engineManagerSource, params json.RawMessage) {
	if !b.engineManagerSourceCurrent(source) {
		return
	}
	// Capture markers in the source-fenced reader before phases can advance.
	// Completion and stop proof still require an authoritative private read.
	var status diagnosticManagerStatus
	if json.Unmarshal(params, &status) != nil || status.Run == nil {
		return
	}
	b.diagnostics.mu.Lock()
	op := b.diagnostics.active
	b.diagnostics.mu.Unlock()
	if op != nil {
		b.diagnostics.mu.Lock()
		matches := op.run.RunID == status.Run.RunID
		b.diagnostics.mu.Unlock()
		if matches {
			b.observeDiagnosticMarker(op, status)
			b.setDiagnosticRun(op, *status.Run)
			select {
			case op.wake <- struct{}{}:
			default:
			}
			select {
			case <-op.done:
				b.workersMu.Lock()
				worker := b.engineMgr
				b.workersMu.Unlock()
				go func() { _ = b.reconcileDiagnostics(context.Background(), worker) }()
			default:
			}
		}
	}
}
