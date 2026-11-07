// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

// This private observation is deliberately outside the public engine:* relay.
// The manager must observe the same effects used by its compare-and-reserve,
// without waking an engine or running a native helper. Unsupported observation
// fails closed; the broker never manufactures an effects fence from its cache.
const diagnosticObserveMethod = "internal:diagnostic-observe"

type diagnosticObservation struct {
	Identity      modelconfig.DiagnosticIdentity   `json:"identity"`
	EffectsFence  string                           `json:"effectsFence"`
	UnloadModels  []modelconfig.DiagnosticModelRef `json:"unloadModels"`
	ReloadedAfter []modelconfig.DiagnosticModelRef `json:"reloadedAfter"`
	DrainEngines  []string                         `json:"drainEngines"`
	StartsEngine  bool                             `json:"startsEngine"`
	Supported     bool                             `json:"supported"`
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
}

type modelDiagnosticRuntime struct {
	mu          sync.Mutex
	reconcileMu sync.Mutex
	previews    map[string]diagnosticReview
	active      *brokerDiagnostic
	last        *modelconfig.DiagnosticRun
	now         func() time.Time
}

// One goroutine owns the ledger. It is published before the first acquisition.
// Cleanup retries run after done closes, serialized by cleanupMu. The manager's
// checkpoint is the only durable owner; this structure is never journalled.
type brokerDiagnostic struct {
	runID               string
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
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if err := b.loadEngineSettingsLocked(); err != nil {
		return 0, "", err
	}
	if b.buildApplyMaintenance.active != nil || len(b.activeSettings) != 0 {
		return 0, "", diagnosticError("node maintenance is active")
	}
	for _, record := range b.engineSettings {
		if record != nil && (record.Snapshot.Phase == "applying" || record.Model != nil && record.Model.Stage != "completed") {
			return 0, "", diagnosticError("model settings are being applied")
		}
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.current.PauseMode == nodepolicy.PauseImmediate {
		return 0, "", diagnosticError("This node is halted. Resume it before measuring memory.")
	}
	if r.recoveryRequired {
		return 0, "", diagnosticError("node policy recovery is pending")
	}
	return b.buildApplyMaintenance.generation, r.current.Revision, nil
}

func (b *Broker) observeDiagnostic(ctx context.Context, p modelconfig.DiagnosticPreviewRequest) (diagnosticObservation, error) {
	var observation diagnosticObservation
	err := modelSettingsCall(ctx, b.getEngineMgr(), diagnosticObserveMethod, struct {
		Request modelconfig.DiagnosticPreviewRequest `json:"request"`
		Memory  *modelconfig.TargetMemory            `json:"memory"`
	}{p, b.modelTargetMemory()}, &observation)
	if err != nil {
		return observation, err
	}
	if !observation.Supported || observation.EffectsFence == "" || len(observation.DrainEngines) == 0 {
		return observation, diagnosticError("diagnostic observation unavailable for this engine family")
	}
	return observation, nil
}

func (b *Broker) previewDiagnostic(ctx context.Context, p modelconfig.DiagnosticPreviewRequest) (modelconfig.DiagnosticPreview, error) {
	if err := p.Validate(); err != nil {
		return modelconfig.DiagnosticPreview{}, err
	}
	if p.NodeID != "" && p.NodeID != b.nodeID {
		return modelconfig.DiagnosticPreview{}, diagnosticError("unsupported: local diagnostics only")
	}
	generation, policy, err := b.diagnosticGate()
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
		UnloadModels: observation.UnloadModels, ReloadedAfter: observation.ReloadedAfter, DrainEngines: slices.Clone(observation.DrainEngines), StartsEngine: observation.StartsEngine, ExpiresAt: now.Add(diagnosticPreviewTTL).Format(time.RFC3339Nano)}
	if preview.PerSlotContext != 0 {
		preview.HighPromptTokens = preview.PerSlotContext - 144
		if preview.HighPromptTokens < 64 {
			return modelconfig.DiagnosticPreview{}, diagnosticError("context-too-small")
		}
		preview.LowPromptTokens = min(int32(512), preview.HighPromptTokens/2)
	} else {
		preview.Warnings = append(preview.Warnings, "Prompt targets and realized allocation will be determined after the consented load.")
	}
	if preview.Slots > 8 {
		return modelconfig.DiagnosticPreview{}, diagnosticError("more than eight slots are unsupported")
	}
	preview.Warnings = append(preview.Warnings, b.diagnosticMemoryWarnings()...)
	preview.Warnings = append(preview.Warnings, "Other programs may change memory use; their activity is included in these samples.", "This fills the configuration's context and can stop the engine if it does not fit.")
	if err := preview.Validate(); err != nil {
		return modelconfig.DiagnosticPreview{}, err
	}
	currentGeneration, currentPolicy, err := b.diagnosticGate()
	if err != nil || generation != currentGeneration || policy != currentPolicy || worker != b.getEngineMgr() {
		return modelconfig.DiagnosticPreview{}, diagnosticError("node changed during preview")
	}
	b.diagnostics.mu.Lock()
	defer b.diagnostics.mu.Unlock()
	if b.diagnostics.active != nil {
		return modelconfig.DiagnosticPreview{}, diagnosticError("a diagnostic is active")
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
	b.diagnostics.previews[preview.PreviewID] = diagnosticReview{preview.Clone(), observation.EffectsFence, generation, policy, worker, now.Add(diagnosticPreviewTTL)}
	return preview, nil
}

func (b *Broker) startDiagnostic(ctx context.Context, p modelconfig.DiagnosticStartRequest) (modelconfig.DiagnosticRun, error) {
	if err := p.Validate(); err != nil {
		return modelconfig.DiagnosticRun{}, err
	}
	if p.NodeID != "" && p.NodeID != b.nodeID {
		return modelconfig.DiagnosticRun{}, diagnosticError("unsupported: local diagnostics only")
	}
	generation, policy, err := b.diagnosticGate()
	if err != nil {
		return modelconfig.DiagnosticRun{}, err
	}
	b.diagnostics.mu.Lock()
	review, found := b.diagnostics.previews[p.PreviewID]
	b.diagnostics.mu.Unlock()
	if !found || !b.diagnosticNow().Before(review.expires) || review.preview.DiagnosticPreviewRequest != p.DiagnosticPreviewRequest || generation != review.generation || policy != review.policy || b.getEngineMgr() != review.worker {
		return modelconfig.DiagnosticRun{}, diagnosticError("stale diagnostic preview; preview again")
	}
	observation, err := b.observeDiagnostic(ctx, p.DiagnosticPreviewRequest)
	if err != nil {
		return modelconfig.DiagnosticRun{}, err
	}
	if observation.EffectsFence != review.effects || !reflect.DeepEqual(observation.Identity, review.preview.DiagnosticIdentity) || !reflect.DeepEqual(observation.UnloadModels, review.preview.UnloadModels) || !reflect.DeepEqual(observation.ReloadedAfter, review.preview.ReloadedAfter) || !reflect.DeepEqual(observation.DrainEngines, review.preview.DrainEngines) || observation.StartsEngine != review.preview.StartsEngine {
		return modelconfig.DiagnosticRun{}, diagnosticError("diagnostic effects changed; preview again")
	}
	if _, err := b.diagnosticMemory(); err != nil {
		return modelconfig.DiagnosticRun{}, err
	}
	// A caller disappearing after acceptance must not abandon cleanup.
	runCtx, cancel := context.WithTimeout(context.Background(), diagnosticRunLimit)
	op := &brokerDiagnostic{review: review, worker: review.worker, cancel: cancel, done: make(chan struct{}), wake: make(chan struct{}, 1)}
	op.run = modelconfig.DiagnosticRun{RunID: settingsID(), Engine: p.Engine, ModelID: p.ModelID, ProfileKey: p.ProfileKey, ConfigurationKey: review.preview.ConfigurationKey, ExecutionKey: review.preview.ExecutionKey, Phase: "reserving", Outcome: "running", StartedAt: b.diagnosticNow().Format(time.RFC3339Nano)}
	op.runID = op.run.RunID
	b.diagnostics.mu.Lock()
	if b.diagnostics.active != nil {
		b.diagnostics.mu.Unlock()
		cancel()
		return modelconfig.DiagnosticRun{}, diagnosticError("a diagnostic is active")
	}
	delete(b.diagnostics.previews, p.PreviewID)
	b.diagnostics.active = op
	result := op.run.Clone()
	b.diagnostics.mu.Unlock()
	go b.executeDiagnostic(runCtx, op)
	return result, nil
}

func (b *Broker) diagnosticStatus(runID string) (*modelconfig.DiagnosticRun, error) {
	b.diagnostics.mu.Lock()
	defer b.diagnostics.mu.Unlock()
	if op := b.diagnostics.active; op != nil && (runID == "" || runID == op.run.RunID) {
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
	if b.diagnostics.active == op {
		op.run = run.Clone()
	}
	b.diagnostics.mu.Unlock()
	b.publishDiagnostic(run)
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
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
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
	if status.State != "none" && (status.Run == nil || status.Run.RunID != op.runID) {
		return status, diagnosticError("manager diagnostic identity mismatch")
	}
	b.observeDiagnosticMarker(op, status)
	return status, nil
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
	cleanup, cancel := context.WithTimeout(context.Background(), time.Minute)
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
	params := diagnosticReserveRequest{op.runID, preview.Engine, preview.ModelID, preview.ProfileKey, preview.ConfigurationKey, preview.ExecutionKey, op.review.effects, b.modelTargetMemory()}
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
	op.run.Phase = "draining"
	b.diagnostics.mu.Unlock()
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
	defer b.diagnostics.mu.Unlock()
	op.run.Outcome, op.run.Reason = "cleanup-failed", "Cleanup could not be confirmed. Retry cleanup."
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
				return err
			}
			status = accepted
			if status.Run == nil || status.Run.RunID != op.runID {
				return diagnosticError("uncertain run acceptance identity mismatch")
			}
			op.accepted, op.runUncertain = true, false
		}
		if op.runAttempted && status.State == "none" {
			return diagnosticError("attempted run disappeared without cleanup proof")
		}
		if op.runAttempted && status.State != "reserved" && status.State != "none" {
			op.accepted, op.runUncertain = true, false
		}
		if status.State != "none" {
			if status.Token != "" {
				op.token = status.Token
			}
			if !status.WorkStopped && status.State != "reserved" {
				if err := b.diagnosticCall(ctx, op, "engine:diagnostic-abort", diagnosticAuthority{op.runID, op.token}, &status); err != nil {
					return err
				}
				// A join response must still identify this operation.
				if status.Run == nil || status.Run.RunID != op.runID {
					return diagnosticError("abort reply identity mismatch")
				}
			}
			if (!status.WorkStopped && status.State != "reserved") || status.State == "cleanup-failed" {
				return diagnosticError("diagnostic work is not confirmed stopped")
			}
			if status.State == "terminal" || status.State == "cleanup-complete" {
				op.reservationReleased = true
			}
			op.stopped, op.status = true, status
			if op.accepted && status.Run != nil {
				b.setDiagnosticRun(op, *status.Run)
			}
		} else if op.accepted {
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
		if err := op.drain.release(ctx); err != nil {
			return err
		}
		op.drain = nil
	}
	if op.token != "" && !op.reservationReleased {
		if err := b.diagnosticCall(ctx, op, "engine:diagnostic-release", diagnosticAuthority{op.runID, op.token}, nil); err != nil {
			return err
		}
		op.token = ""
	}
	if op.lease != nil {
		op.lease.Release()
		op.lease = nil
	}
	b.diagnostics.mu.Lock()
	if b.diagnostics.active == op {
		result := op.run.Clone()
		if result.Outcome == "running" {
			result.Outcome = "cancelled"
		}
		result.Phase = "done"
		b.diagnostics.last = &result
		b.diagnostics.active = nil
	}
	b.diagnostics.mu.Unlock()
	b.signalModelConfig()
	return nil
}

func (b *Broker) cancelDiagnostic(ctx context.Context, runID string) (*modelconfig.DiagnosticRun, error) {
	b.diagnostics.mu.Lock()
	op := b.diagnostics.active
	if op == nil || runID != "" && op.run.RunID != runID {
		b.diagnostics.mu.Unlock()
		return b.diagnosticStatus(runID)
	}
	op.cancel()
	b.diagnostics.mu.Unlock()
	select {
	case <-op.done:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	op.cleanupMu.Lock()
	err := b.cleanupDiagnostic(ctx, op)
	if err != nil {
		b.retainDiagnostic(op, err)
	}
	op.cleanupMu.Unlock()
	result, _ := b.diagnosticStatus(runID)
	return result, err
}

func (b *Broker) handleModelDiagnostic(msg *Message) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	var result any
	var err error
	switch msg.Method {
	case "engine:diagnostic-preview":
		var p modelconfig.DiagnosticPreviewRequest
		err = modelconfig.DecodeStrict(msg.Params, &p)
		if err == nil {
			result, err = b.previewDiagnostic(ctx, p)
		}
	case "engine:diagnostic-start":
		var p modelconfig.DiagnosticStartRequest
		err = modelconfig.DecodeStrict(msg.Params, &p)
		if err == nil {
			result, err = b.startDiagnostic(ctx, p)
		}
	case "engine:diagnostic-status", "engine:diagnostic-cancel":
		var p modelconfig.DiagnosticRunRef
		err = modelconfig.DecodeStrict(msg.Params, &p)
		if err == nil && p.NodeID != "" && p.NodeID != b.nodeID {
			err = diagnosticError("unsupported: local diagnostics only")
		}
		if err == nil {
			if msg.Method == "engine:diagnostic-cancel" {
				result, err = b.cancelDiagnostic(ctx, p.RunID)
			} else {
				result, err = b.diagnosticStatus(p.RunID)
			}
		}
	}
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
				go func() { _ = b.reconcileDiagnostics(context.Background(), op.worker) }()
			default:
			}
		}
	}
}
