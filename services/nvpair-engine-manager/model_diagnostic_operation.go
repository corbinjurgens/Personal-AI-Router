// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"reflect"
	"slices"
	"time"

	"nvpair-shared/modelconfig"
)

const (
	diagnosticRunMethod      = "engine:diagnostic-run"
	diagnosticStatusMethod   = "engine:diagnostic-run-status"
	diagnosticAbortMethod    = "engine:diagnostic-abort"
	diagnosticBaselineMethod = "engine:diagnostic-baseline-done"
	diagnosticChangedMethod  = "engine:diagnostic-changed"
	diagnosticBaselineBudget = 30 * time.Second
	diagnosticUnloadBudget   = 30 * time.Second
	diagnosticCleanupBudget  = 2 * time.Minute
	diagnosticRetainedRuns   = 16
)

type diagnosticRunParams struct {
	diagnosticReleaseParams
	Engine           string `json:"engine"`
	ModelID          string `json:"modelId"`
	ProfileKey       string `json:"profileKey,omitempty"`
	ConfigurationKey string `json:"configurationKey,omitempty"`
	ExecutionKey     string `json:"executionKey,omitempty"`
	// Unknown native-fit targets are zero. Actual targets always come from
	// identity-fenced /props after load, never a guessed finite allocation.
	Low      int32  `json:"low,omitempty"`
	High     int32  `json:"high,omitempty"`
	Generate int32  `json:"generate"`
	Deadline string `json:"deadline"`
}

type diagnosticBaselineParams struct {
	diagnosticReleaseParams
	Phase string `json:"phase"`
}

// Partial observations are useful even when load or readback fails. They are
// deliberately separate from MeasuredProfile, whose complete identity is the
// later record owner's responsibility. Placement is observed buffer evidence,
// never synthesized from a finite estimate or the requested GPU switches.
type diagnosticLoadObservation struct {
	PerSlotContext int32                             `json:"perSlotContext"`
	Slots          int32                             `json:"slots"`
	BuildInfo      string                            `json:"buildInfo,omitempty"`
	NativeBuildID  string                            `json:"nativeBuildId,omitempty"`
	PoolPlacement  []modelconfig.DiagnosticPlacement `json:"poolPlacement,omitempty"`
	bufferIdentity string
}

type diagnosticOperationStatus struct {
	modelconfig.DiagnosticRun `json:"run"`
	State                     string                          `json:"state"`
	Token                     string                          `json:"token,omitempty"`
	WorkStopped               bool                            `json:"workStopped"`
	Markers                   map[string]time.Time            `json:"markers,omitempty"`
	Marker                    string                          `json:"marker,omitempty"`
	MarkerAt                  string                          `json:"markerAt,omitempty"`
	Observation               *diagnosticLoadObservation      `json:"observation,omitempty"`
	Low                       *modelconfig.MeasuredPhase      `json:"low,omitempty"`
	High                      *modelconfig.MeasuredPhase      `json:"high,omitempty"`
	Identity                  *modelconfig.DiagnosticIdentity `json:"identity,omitempty"`
}

func (s diagnosticOperationStatus) clone() diagnosticOperationStatus {
	s.DiagnosticRun = s.DiagnosticRun.Clone()
	if s.Identity != nil {
		identity := s.Identity.Clone()
		s.Identity = &identity
	}
	if s.Markers != nil {
		markers := make(map[string]time.Time, len(s.Markers))
		for name, at := range s.Markers {
			markers[name] = at
		}
		s.Markers = markers
	}
	if s.Observation != nil {
		value := *s.Observation
		value.PoolPlacement = slices.Clone(value.PoolPlacement)
		s.Observation = &value
	}
	if s.Low != nil {
		value := s.Low.Clone()
		s.Low = &value
	}
	if s.High != nil {
		value := s.High.Clone()
		s.High = &value
	}
	return s
}

type diagnosticEngineBefore struct {
	engine  string
	running bool
	runID   string
	proc    *managedProc
	touched bool
}

// All mutable operation fields are guarded by mutationAdmission, except the
// engine snapshots and inspection which only the joined worker writes.
type modelDiagnosticOperation struct {
	params                       diagnosticRunParams
	reservation                  diagnosticReserveParams
	ctx                          context.Context
	cancel                       context.CancelFunc
	done                         chan struct{}
	baseline                     chan struct{}
	baselineAccepted             bool
	afterLoad                    chan struct{}
	afterLoadAccepted            bool
	started                      time.Time
	loadStarted                  time.Time
	status                       diagnosticOperationStatus
	engines                      []diagnosticEngineBefore
	inspection                   modelconfig.InspectResult
	configurationPools           []string
	outcome, failedPhase, reason string
	checkpoint                   *diagnosticCheckpoint
}

// The record owner authenticates broker samples against this completed run,
// including failed/partial evidence. A detached copy survives the next run.
type diagnosticCompletedRun struct {
	token       string
	Reservation diagnosticReserveParams
	Status      diagnosticOperationStatus
	Inspection  modelconfig.InspectResult
}

func (e *Executor) completedDiagnosticRun(runID string) (diagnosticCompletedRun, bool) {
	e.mutationAdmission.Lock()
	defer e.mutationAdmission.Unlock()
	r, ok := e.diagnosticCompleted[runID]
	r.Reservation.TargetMemory = r.Reservation.TargetMemory.Clone()
	r.Reservation.Hardware = r.Reservation.Hardware.clone()
	r.Status = r.Status.clone()
	r.Inspection = r.Inspection.Clone()
	return r, ok
}

func (e *Executor) diagnosticRunStatus(runID string) diagnosticOperationStatus {
	status, err := e.diagnosticStatusWithCheckpoint(runID)
	if err != nil {
		return diagnosticOperationStatus{State: "cleanup-failed", DiagnosticRun: modelconfig.DiagnosticRun{RunID: runID, Outcome: "cleanup-failed", Reason: "checkpoint-unreadable"}}
	}
	return status
}

func (e *Executor) diagnosticStatusWithCheckpoint(runID string) (diagnosticOperationStatus, error) {
	// Durable I/O precedes admission locking, never recursively takes it.
	checkpoint, known, err := e.diagnosticCheckpointStatus(runID)
	if err != nil {
		return diagnosticOperationStatus{}, err
	}
	e.mutationAdmission.Lock()
	defer e.mutationAdmission.Unlock()
	if op := e.diagnosticOperation; op != nil && (runID == "" || op.params.RunID == runID) {
		s := op.status.clone()
		s.ElapsedMS = time.Since(op.started).Milliseconds()
		return s, nil
	}
	if done, ok := e.diagnosticCompleted[runID]; ok {
		return done.Status.clone(), nil
	}
	if r := e.diagnostic; r != nil && r.ready && (runID == "" || r.params.RunID == runID) {
		if known && r.workUnconfirmed {
			return checkpoint, nil
		}
		if r.workUnconfirmed && r.checkpoint != nil {
			status := diagnosticOperationStatus{DiagnosticRun: r.checkpoint.Run.Clone(), State: r.checkpoint.State, Token: r.token, WorkStopped: false}
			if r.checkpoint.Identity != nil {
				identity := r.checkpoint.Identity.Clone()
				status.Identity = &identity
			}
			return status, nil
		}
		status := diagnosticOperationStatus{State: "reserved", Token: r.token, WorkStopped: !r.workUnconfirmed, DiagnosticRun: modelconfig.DiagnosticRun{
			RunID: r.params.RunID, Engine: r.params.Engine, ModelID: r.params.ModelID, ProfileKey: r.params.ProfileKey,
			ConfigurationKey: r.params.ConfigurationKey, ExecutionKey: r.params.ExecutionKey, Phase: "reserving", Outcome: "running",
		}}
		if r.checkpoint == nil {
			return diagnosticOperationStatus{}, diagnosticConflict("reservation checkpoint unavailable")
		}
		status.StartedAt = r.checkpoint.Run.StartedAt
		if started, err := time.Parse(time.RFC3339Nano, status.StartedAt); err == nil {
			status.ElapsedMS = max(0, time.Since(started).Milliseconds())
		}
		if r.identity != nil {
			identity := r.identity.Clone()
			status.Identity = &identity
		}
		return status, nil
	}
	if known {
		return checkpoint, nil
	}
	if runID == "" && len(e.diagnosticCompletedOrder) > 0 {
		return e.diagnosticCompleted[e.diagnosticCompletedOrder[len(e.diagnosticCompletedOrder)-1]].Status.clone(), nil
	}
	return diagnosticOperationStatus{State: "none"}, nil
}

func (e *Executor) acceptDiagnostic(p diagnosticRunParams) (*modelDiagnosticOperation, bool, error) {
	e.mutationAdmission.Lock()
	defer e.mutationAdmission.Unlock()
	r := e.diagnostic
	if r == nil || !r.ready || r.params.RunID != p.RunID || r.token != p.Token {
		return nil, false, diagnosticConflict("invalid reservation authority")
	}
	// The reservation already binds the consented profile. An authority-only
	// call cannot substitute a different model or guessed native allocation.
	if p.Engine == "" {
		p.Engine = r.params.Engine
	}
	if p.ModelID == "" {
		p.ModelID = r.params.ModelID
	}
	if p.ProfileKey == "" {
		p.ProfileKey = r.params.ProfileKey
	}
	if p.ConfigurationKey == "" && p.ExecutionKey == "" {
		p.ConfigurationKey, p.ExecutionKey = r.params.ConfigurationKey, r.params.ExecutionKey
	}
	if p.Generate == 0 {
		p.Generate = modelDiagnosticGenerateTokens
	}
	if p.Deadline == "" {
		if current := e.diagnosticOperation; current != nil {
			p.Deadline = current.params.Deadline
		} else {
			p.Deadline = time.Now().Add(modelDiagnosticExerciseLimit).UTC().Format(time.RFC3339Nano)
		}
	}
	if current := e.diagnosticOperation; current != nil {
		if current.params == p {
			return current, false, nil
		}
		return nil, false, diagnosticConflict("another operation or different parameters for this runId")
	}
	if _, exists := e.diagnosticCompleted[p.RunID]; exists {
		return nil, false, diagnosticConflict("runId already completed")
	}
	deadline, err := time.Parse(time.RFC3339Nano, p.Deadline)
	if err != nil || time.Until(deadline) <= 0 || time.Until(deadline) > modelDiagnosticExerciseLimit ||
		p.Generate != modelDiagnosticGenerateTokens || p.Low < 0 || p.High < 0 {
		return nil, false, modelSettingsParamsError(fmt.Errorf("generate must be 128; deadline must be within 20 minutes; targets must be nonnegative"))
	}
	if p.Engine != r.params.Engine ||
		p.ModelID != r.params.ModelID || p.ConfigurationKey != r.params.ConfigurationKey || p.ExecutionKey != r.params.ExecutionKey ||
		p.ProfileKey != "" && p.ProfileKey != r.params.ProfileKey || r.workUnconfirmed {
		return nil, false, diagnosticConflict("run does not match the reserved profile and token")
	}
	if e.shuttingDown.Load() {
		return nil, false, diagnosticConflict("manager is shutting down")
	}
	if err := e.diagnosticPauseVeto(); err != nil {
		return nil, false, err
	}
	e.nodeProjection.mu.Lock()
	pauseChanged := r.pauseGeneration != e.nodeProjection.pauseGeneration
	e.nodeProjection.mu.Unlock()
	if pauseChanged {
		return nil, false, diagnosticConflict("pause changed after reservation")
	}
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	op := &modelDiagnosticOperation{params: p, reservation: r.params, ctx: ctx, cancel: cancel,
		done: make(chan struct{}), baseline: make(chan struct{}), afterLoad: make(chan struct{}), started: time.Now(), outcome: "completed"}
	op.reservation.TargetMemory = r.params.TargetMemory.Clone()
	op.reservation.Hardware = r.params.Hardware.clone()
	op.configurationPools = slices.Clone(r.configurationPools)
	op.status = diagnosticOperationStatus{State: "accepted", Token: p.Token, Markers: make(map[string]time.Time), DiagnosticRun: modelconfig.DiagnosticRun{
		RunID: p.RunID, Engine: p.Engine, ModelID: p.ModelID, ProfileKey: r.params.ProfileKey,
		ConfigurationKey: p.ConfigurationKey, ExecutionKey: p.ExecutionKey, Phase: "reserving", Outcome: "running",
		StartedAt: op.started.UTC().Format(time.RFC3339Nano),
	}}
	if r.identity != nil {
		identity := r.identity.Clone()
		op.status.Identity = &identity
	}
	r.workUnconfirmed = true
	e.diagnosticOperation = op
	return op, true, nil
}

func (e *Executor) diagnosticBaselineDone(p diagnosticBaselineParams) error {
	e.mutationAdmission.Lock()
	defer e.mutationAdmission.Unlock()
	op := e.diagnosticOperation
	if op == nil || op.params.RunID != p.RunID || op.params.Token != p.Token ||
		op.status.Outcome != "running" || op.ctx.Err() != nil {
		return diagnosticConflict("baseline acknowledgement does not match the active token, run and phase")
	}
	switch {
	case p.Phase == "baseline" && op.status.Phase == "baseline":
		if !op.baselineAccepted {
			close(op.baseline)
			op.baselineAccepted = true
		}
	case p.Phase == "after-load" && op.status.Phase == "loading" && !op.status.Markers["loaded"].IsZero():
		if !op.afterLoadAccepted {
			close(op.afterLoad)
			op.afterLoadAccepted = true
		}
	default:
		return diagnosticConflict("sample acknowledgement does not match the active phase")
	}
	return nil
}

func waitDiagnosticSamples(ctx context.Context, ready <-chan struct{}) error {
	timer := time.NewTimer(diagnosticBaselineBudget)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return context.DeadlineExceeded
	case <-ready:
		return ctx.Err()
	}
}

func (e *Executor) cancelDiagnosticForPause() {
	e.mutationAdmission.Lock()
	defer e.mutationAdmission.Unlock()
	if op := e.diagnosticOperation; op != nil {
		op.cancel()
	}
}

// The notification reader only cancels. Internal Sleep and prepare-shutdown
// join before replying, keeping the reader free to consume in-flight replies.
func (e *Executor) abortDiagnosticAndJoin() {
	e.mutationAdmission.Lock()
	op := e.diagnosticOperation
	if op != nil {
		op.cancel()
	}
	var done <-chan struct{}
	if op != nil {
		done = op.done
	}
	e.mutationAdmission.Unlock()
	if done != nil {
		<-done
	}
}

func (e *Executor) diagnosticPhase(op *modelDiagnosticOperation, phase, marker string) time.Time {
	at := time.Now()
	e.mutationAdmission.Lock()
	op.status.Phase, op.status.Marker = phase, marker
	op.status.State = "running"
	op.status.MarkerAt = at.UTC().Format(time.RFC3339Nano)
	if marker != "" {
		if op.status.Markers == nil {
			op.status.Markers = make(map[string]time.Time)
		}
		op.status.Markers[marker] = at
	}
	op.status.ElapsedMS = at.Sub(op.started).Milliseconds()
	status := op.status.clone()
	e.mutationAdmission.Unlock()
	e.notify(diagnosticChangedMethod, status)
	return at
}

func (m *Manager) startDiagnostic(msg *Message) {
	var p diagnosticRunParams
	if err := modelconfig.DecodeStrict(msg.Params, &p); err != nil {
		m.respondOrErr(msg, nil, modelSettingsParamsError(err))
		return
	}
	op, start, err := m.exec.acceptDiagnostic(p)
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	m.exec.mutationAdmission.Lock()
	status := op.status.clone()
	m.exec.mutationAdmission.Unlock()
	if start {
		m.exec.notify(diagnosticChangedMethod, status)
	}
	m.respondOrErr(msg, status, nil)
	if start {
		go m.runDiagnosticOperation(op, false)
	}
}

func (m *Manager) handleDiagnosticOperationRequest(msg *Message) bool {
	if !msg.IsRequest() {
		return false
	}
	switch msg.Method {
	case diagnosticRunMethod:
		// Accept in the lifecycle dispatcher; the long worker has its own bound.
		m.runOp(context.Background(), msg)
		return true
	case diagnosticStatusMethod:
		var p diagnosticReleaseParams
		if err := modelconfig.DecodeStrict(msg.Params, &p); err != nil {
			m.respondOrErr(msg, nil, modelSettingsParamsError(err))
			return true
		}
		status, err := m.exec.diagnosticStatusWithCheckpoint(p.RunID)
		m.respondOrErr(msg, status, err)
		return true
	case diagnosticBaselineMethod:
		var p diagnosticBaselineParams
		if err := modelconfig.DecodeStrict(msg.Params, &p); err != nil {
			m.respondOrErr(msg, nil, modelSettingsParamsError(err))
			return true
		}
		m.respondOrErr(msg, map[string]bool{"ok": true}, m.exec.diagnosticBaselineDone(p))
		return true
	case diagnosticAbortMethod:
		var p diagnosticReleaseParams
		if err := modelconfig.DecodeStrict(msg.Params, &p); err != nil {
			m.respondOrErr(msg, nil, modelSettingsParamsError(err))
			return true
		}
		go func() { status, err := m.abortDiagnostic(p); m.respondOrErr(msg, status, err) }()
		return true
	}
	return false
}

func (m *Manager) abortDiagnostic(p diagnosticReleaseParams) (diagnosticOperationStatus, error) {
	e := m.exec
	e.mutationAdmission.Lock()
	op := e.diagnosticOperation
	if op == nil {
		done, ok := e.diagnosticCompleted[p.RunID]
		if ok && done.Reservation.RunID == p.RunID && done.token == p.Token {
			e.mutationAdmission.Unlock()
			return done.Status.clone(), nil
		}
		e.mutationAdmission.Unlock()
		return m.abortRecoveredDiagnostic(p)
	}
	if op.params.RunID != p.RunID || op.params.Token != p.Token {
		e.mutationAdmission.Unlock()
		return diagnosticOperationStatus{}, diagnosticConflict("invalid operation token")
	}
	op.cancel()
	retry := op.status.Outcome == "cleanup-failed"
	if retry {
		select {
		case <-op.done:
			op.done = make(chan struct{})
			op.status.Outcome = "running"
		default:
			retry = false
		}
	}
	done := op.done
	e.mutationAdmission.Unlock()
	if retry {
		go m.runDiagnosticOperation(op, true)
	}
	<-done
	return e.diagnosticStatusWithCheckpoint(p.RunID)
}

// This is the accepted runOp lifecycle worker. It calls the existing Wake,
// load, unload and Sleep owners synchronously; it never launches/kills a
// process itself. Cleanup uses a fresh bounded context and is always joined.
func (m *Manager) runDiagnosticOperation(op *modelDiagnosticOperation, cleanupOnly bool) {
	e := m.exec
	if !cleanupOnly {
		if err := m.measureDiagnostic(op); err != nil {
			e.mutationAdmission.Lock()
			op.failedPhase = op.status.Phase
			op.outcome, op.reason = "failed", diagnosticFailureReason(err)
			if errors.Is(err, context.Canceled) || errors.Is(context.Cause(op.ctx), context.Canceled) {
				op.outcome = "cancelled"
			}
			e.mutationAdmission.Unlock()
		}
	}
	op.cancel()
	e.diagnosticPhase(op, "restoring", "")
	ctx, cancel := context.WithTimeout(context.Background(), diagnosticCleanupBudget)
	var checkpointErr error
	cleanupProved := false
	if cleanupOnly {
		checkpoint, readErr := e.desired.pendingDiagnostic()
		checkpointErr = readErr
		cleanupProved = readErr == nil && checkpoint != nil && checkpoint.State == "cleanup-complete"
	}
	if op.checkpoint != nil && !cleanupProved && checkpointErr == nil {
		op.checkpoint.State = "pending"
		checkpointErr = e.saveDiagnosticOperation(op)
	}
	err := checkpointErr
	if !cleanupProved {
		err = errors.Join(err, m.restoreDiagnostic(ctx, op))
	}
	cancel()
	if err == nil {
		err = e.confirmDiagnosticWorkStopped(op.params.RunID, op.params.Token)
	}
	if err == nil {
		e.mutationAdmission.Lock()
		terminal := op.status.DiagnosticRun.Clone()
		terminal.Phase = "done"
		terminal.Outcome, terminal.Reason, terminal.FailedPhase = op.outcome, op.reason, op.failedPhase
		terminal.ElapsedMS = time.Since(op.started).Milliseconds()
		e.mutationAdmission.Unlock()
		err = e.completeDiagnosticCheckpoint(op.params.RunID, op.params.Token, terminal)
	}
	e.mutationAdmission.Lock()
	op.status.Phase, op.status.State = "done", "cleanup-complete"
	op.status.Outcome, op.status.Reason, op.status.FailedPhase = op.outcome, op.reason, op.failedPhase
	op.status.Marker, op.status.MarkerAt = "", ""
	op.status.ElapsedMS = time.Since(op.started).Milliseconds()
	if err != nil {
		op.status.State, op.status.Outcome = "cleanup-failed", "cleanup-failed"
		op.status.Reason, op.status.FailedPhase = "cleanup-unconfirmed", "restoring"
	} else {
		op.status.WorkStopped = true
		if e.diagnosticCompleted == nil {
			e.diagnosticCompleted = make(map[string]diagnosticCompletedRun)
		}
		e.diagnosticCompleted[op.params.RunID] = diagnosticCompletedRun{op.params.Token, op.reservation, op.status.clone(), op.inspection.Clone()}
		e.diagnosticCompletedOrder = append(e.diagnosticCompletedOrder, op.params.RunID)
		if len(e.diagnosticCompletedOrder) > diagnosticRetainedRuns {
			delete(e.diagnosticCompleted, e.diagnosticCompletedOrder[0])
			e.diagnosticCompletedOrder = e.diagnosticCompletedOrder[1:]
		}
		e.diagnosticOperation = nil
	}
	status := op.status.clone()
	e.mutationAdmission.Unlock()
	if err != nil {
		if checkpoint, loadErr := e.desired.pendingDiagnostic(); loadErr == nil && checkpoint != nil {
			checkpoint.Run = status.DiagnosticRun.Clone()
			checkpoint.State = "cleanup-failed"
			op.checkpoint = checkpoint
			_ = e.updateDiagnosticCheckpoint(op.params.RunID, op.params.Token, *checkpoint)
		}
	}
	e.notify(diagnosticChangedMethod, status)
	e.mutationAdmission.Lock()
	close(op.done)
	e.mutationAdmission.Unlock()
}

func diagnosticFailureReason(err error) string {
	switch {
	case errors.Is(err, context.Canceled):
		return "cancelled"
	case errors.Is(err, context.DeadlineExceeded):
		return "deadline-exceeded"
	case errors.Is(err, errModelDiagnosticEngineExited):
		return "engine-exited"
	case errors.Is(err, errModelDiagnosticIdentity):
		return "identity-changed"
	default:
		return "diagnostic-failed"
	}
}

func (m *Manager) measureDiagnostic(op *modelDiagnosticOperation) error {
	e := m.exec
	ctx := withDiagnosticHardware(e.modelMemoryContext(op.ctx, op.params.Engine, op.reservation.TargetMemory, false), op.reservation.Hardware)
	ctx, err := e.diagnosticContext(ctx, op.params.Engine, op.params.RunID, op.params.Token)
	if err != nil {
		return err
	}
	ctx = context.WithValue(ctx, diagnosticConfigurationPoolsKey{}, slices.Clone(op.configurationPools))
	ctx = context.WithValue(ctx, diagnosticCapturedHardwareKey{}, true)
	if err := e.compareDiagnosticReservation(ctx, op.reservation); err != nil {
		return err
	}
	// Static accepted hardware capacity is not free-memory telemetry. The cold
	// baseline outlives a sample's TTL, without renewing any sample freshness.
	inspection, err := e.diagnosticInspectIdentity(ctx, op)
	op.inspection = inspection
	if err != nil {
		return err
	}
	op.checkpoint, err = e.desired.pendingDiagnostic()
	if err != nil {
		return err
	}
	if op.checkpoint == nil {
		return diagnosticConflict("reservation checkpoint missing")
	}
	for _, engine := range e.reg.Names() {
		st, err := e.state(engine)
		if err != nil {
			return err
		}
		st.mu.Lock()
		before := diagnosticEngineBefore{engine: engine, running: st.running, runID: st.runID, proc: st.proc}
		unowned := engine == op.params.Engine && st.running && (st.adopted || st.proc == nil)
		st.mu.Unlock()
		if unowned {
			return errModelDiagnosticIdentity
		}
		op.engines = append(op.engines, before)
		op.checkpoint.Engines = append(op.checkpoint.Engines, diagnosticCheckpointEngine{Engine: engine, RunningBefore: before.running, ProcessRunID: before.runID, AffectedModels: []string{}, LoadedModels: []string{}})
	}
	if err := e.saveDiagnosticOperation(op); err != nil {
		return err
	}
	e.diagnosticPhase(op, "unloading", "")
	for i := range op.engines {
		before := &op.engines[i]
		if !before.running {
			continue
		}
		before.touched = true
		loaded, err := e.diagnosticLoaded(ctx, before.engine)
		if err != nil {
			return err
		}
		op.checkpoint.Engines[i].AffectedModels = slices.Clone(loaded)
		if err := e.saveDiagnosticOperation(op); err != nil {
			return err
		}
		if err := e.diagnosticUnloadAll(ctx, op, before.engine); err != nil {
			return err
		}
	}
	e.diagnosticPhase(op, "baseline", "baseline-ready")
	if err := e.saveDiagnosticOperation(op); err != nil {
		return err
	}
	if err := waitDiagnosticSamples(ctx, op.baseline); err != nil {
		return err
	}
	engineCtx, err := e.diagnosticContext(ctx, op.params.Engine, op.params.RunID, op.params.Token)
	if err != nil {
		return err
	}
	for i := range op.engines {
		if op.engines[i].engine == op.params.Engine {
			op.engines[i].touched = true
		}
	}
	e.diagnosticPhase(op, "starting", "")
	for i := range op.checkpoint.Engines {
		if op.checkpoint.Engines[i].Engine == op.params.Engine {
			op.checkpoint.Engines[i].Started = !op.checkpoint.Engines[i].RunningBefore
		}
	}
	if err := e.saveDiagnosticOperation(op); err != nil {
		return err
	}
	wakeErr := m.runDiagnosticLifecycle(engineCtx, op.params.Engine, true)
	st, err := e.state(op.params.Engine)
	if err != nil {
		return err
	}
	// Capture the process even if a subsequent load fails, so cleanup owns the
	// exact run started by this operation and never signals a replacement.
	st.mu.Lock()
	for i := range op.engines {
		if op.engines[i].engine == op.params.Engine {
			op.engines[i].proc, op.engines[i].runID = st.proc, st.runID
		}
	}
	st.mu.Unlock()
	if wakeErr != nil {
		return wakeErr
	}
	e.diagnosticPhase(op, "loading", "")
	st.mu.Lock()
	processRunID := st.runID
	st.mu.Unlock()
	for i := range op.checkpoint.Engines {
		if op.checkpoint.Engines[i].Engine == op.params.Engine {
			op.checkpoint.Engines[i].ProcessRunID = processRunID
			op.checkpoint.Engines[i].LoadedModels = []string{op.reservation.ProfileKey}
		}
	}
	if err := e.saveDiagnosticOperation(op); err != nil {
		return err
	}
	op.loadStarted = time.Now()
	if _, err := e.ModelLoad(engineCtx, op.params.Engine, op.reservation.ProfileKey); err != nil {
		return err
	}
	child, observed, err := e.diagnosticLoadedChild(engineCtx, op, st)
	if err != nil {
		return err
	}
	inspection, err = e.diagnosticInspectIdentity(ctx, op)
	op.inspection = inspection
	if err != nil {
		return err
	}
	observed.NativeBuildID = inspection.NativeBuildID
	e.mutationAdmission.Lock()
	op.status.Observation = &observed
	e.mutationAdmission.Unlock()
	if observed.BuildInfo == "" {
		return errModelDiagnosticIdentity
	}
	identity, err := e.diagnosticRequestedIdentity(ctx, modelconfig.InspectRequest{Engine: op.params.Engine, ModelID: op.params.ModelID, ProfileKey: op.reservation.ProfileKey}, inspection)
	if err != nil {
		return err
	}
	identity.BuildInfo, identity.PerSlotContext, identity.Slots, identity.Placement = observed.BuildInfo, observed.PerSlotContext, observed.Slots, slices.Clone(observed.PoolPlacement)
	if err := identity.Validate(); err != nil {
		return err
	}
	e.mutationAdmission.Lock()
	op.status.Identity = &identity
	e.mutationAdmission.Unlock()
	if err := e.saveDiagnosticOperation(op); err != nil {
		return err
	}
	if op.params.ConfigurationKey != "" {
		st.mu.Lock()
		matches := false
		for _, load := range st.plat.Runtime.ModelLoads {
			if load.ProfileKey == op.reservation.ProfileKey && load.Resolved != nil {
				matches = load.Resolved.PerSlotContext == observed.PerSlotContext && load.Resolved.Slots == observed.Slots
			}
		}
		st.mu.Unlock()
		if !matches {
			return errModelDiagnosticIdentity
		}
	}
	low, high, err := modelDiagnosticTargets(observed.PerSlotContext, observed.Slots)
	if err != nil {
		return err
	}
	if op.params.ConfigurationKey != "" && (op.params.Low != 0 && op.params.Low != low || op.params.High != 0 && op.params.High != high) {
		return errModelDiagnosticIdentity
	}
	e.diagnosticPhase(op, "loading", "loaded")
	if err := e.saveDiagnosticOperation(op); err != nil {
		return err
	}
	if err := waitDiagnosticSamples(ctx, op.afterLoad); err != nil {
		return err
	}
	for _, phase := range []modelDiagnosticExercisePhase{
		{Phase: "low", TargetPromptTokens: low, Requests: 1},
		{Phase: "high", TargetPromptTokens: high, Requests: observed.Slots},
	} {
		phase.StartedAt = e.diagnosticPhase(op, phase.Phase, phase.Phase+"-start")
		if err := e.saveDiagnosticOperation(op); err != nil {
			return err
		}
		result, err := e.modelDiagnosticExercise(engineCtx, child, phase)
		e.mutationAdmission.Lock()
		if phase.Phase == "low" {
			op.status.Low = &result
		} else {
			op.status.High = &result
		}
		e.mutationAdmission.Unlock()
		e.diagnosticPhase(op, phase.Phase, phase.Phase+"-end")
		if err != nil {
			return err
		}
	}
	if err := child.Check(engineCtx); err != nil {
		return err
	}
	op.inspection, err = e.diagnosticInspectIdentity(ctx, op)
	return err
}

func (e *Executor) diagnosticInspectIdentity(ctx context.Context, op *modelDiagnosticOperation) (modelconfig.InspectResult, error) {
	result, err := e.inspectModelArtifacts(ctx, modelconfig.InspectRequest{Engine: op.params.Engine, ModelID: op.params.ModelID, ProfileKey: op.reservation.ProfileKey})
	if err != nil {
		return result, err
	}
	if result.ConfigurationKey != op.params.ConfigurationKey || result.ExecutionKey != op.params.ExecutionKey || result.NativeBuildID == "" {
		return result, errModelDiagnosticIdentity
	}
	return result, nil
}

// Use the loaded sweep's extractor and llama status semantics, while avoiding
// finite-envelope publication for native-fit/explicit GPU-selection profiles.
func (e *Executor) diagnosticLoaded(ctx context.Context, engine string) ([]string, error) {
	st, err := e.state(engine)
	if err != nil {
		return nil, err
	}
	st.mu.Lock()
	running, run, proc := st.running, st.runID, st.proc
	st.mu.Unlock()
	if !running {
		if e.terminationVerifiedContext(ctx, st) {
			return []string{}, nil
		}
		return nil, errModelDiagnosticIdentity
	}
	action, ok := st.manifest.Actions["loaded_models"]
	if !ok || action.Result == nil {
		return nil, errors.New("loaded-model-evidence-unavailable")
	}
	raw, err := e.Action(ctx, engine, "loaded_models", nil)
	if err != nil {
		return nil, err
	}
	loaded, ok := extractStringsResult(raw, action.Result)
	if !ok {
		return nil, errModelDiagnosticIdentity
	}
	if st.manifest.FamilyOrDefault() == "llamacpp" {
		if !llamaLoadedSemantics(raw, loaded) {
			return nil, errModelDiagnosticIdentity
		}
		var rows struct {
			Data []struct {
				ID     string `json:"id"`
				Status struct {
					Value string `json:"value"`
				} `json:"status"`
			} `json:"data"`
		}
		if err := json.Unmarshal(raw, &rows); err != nil {
			return nil, err
		}
		for _, row := range rows.Data {
			if row.Status.Value == "loading" {
				return nil, errors.New("model-still-loading")
			}
			// Sleeping is not affirmative cold evidence. Explicitly unload it
			// as well rather than accepting the read endpoint's loaded filter.
			if row.Status.Value == "sleeping" {
				loaded = append(loaded, row.ID)
			}
		}
	}
	st.mu.Lock()
	current := st.running && st.runID == run && st.proc == proc
	st.mu.Unlock()
	if !current {
		return nil, errModelDiagnosticIdentity
	}
	return loaded, nil
}

func (e *Executor) diagnosticUnloadAll(ctx context.Context, op *modelDiagnosticOperation, engine string) error {
	ctx, cancel := context.WithTimeout(ctx, diagnosticUnloadBudget)
	defer cancel()
	ctx, err := e.diagnosticContext(ctx, engine, op.params.RunID, op.params.Token)
	if err != nil {
		return err
	}
	loaded, err := e.diagnosticLoaded(ctx, engine)
	if err != nil {
		return err
	}
	for _, model := range loaded {
		if _, err := e.ModelUnload(ctx, engine, model); err != nil {
			return err
		}
	}
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		loaded, err = e.diagnosticLoaded(ctx, engine)
		if err == nil && len(loaded) == 0 {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func (m *Manager) restoreDiagnostic(ctx context.Context, op *modelDiagnosticOperation) error {
	e := m.exec
	var failures []error
	for i := range op.engines {
		before := &op.engines[i]
		if !before.touched {
			continue
		}
		st, err := e.state(before.engine)
		if err != nil {
			failures = append(failures, err)
			continue
		}
		st.mu.Lock()
		same := st.proc == before.proc && st.runID == before.runID
		st.mu.Unlock()
		if !same {
			if e.terminationVerifiedContext(ctx, st) {
				continue
			}
			failures = append(failures, errModelDiagnosticIdentity)
			continue
		}
		err = e.diagnosticUnloadAll(ctx, op, before.engine)
		// A run-started engine must return to stopped. Unconfirmed unloads
		// escalate only through the existing ownership-checked Sleep owner.
		if err != nil || !before.running {
			scoped, scopeErr := e.diagnosticContext(ctx, before.engine, op.params.RunID, op.params.Token)
			if scopeErr == nil {
				scopeErr = m.runDiagnosticLifecycle(scoped, before.engine, false)
			}
			if scopeErr != nil || !e.terminationVerifiedContext(ctx, st) {
				failures = append(failures, errors.New("diagnostic-cleanup-unconfirmed"))
			}
		}
	}
	return errors.Join(failures...)
}

func (e *Executor) diagnosticLoadedChild(ctx context.Context, op *modelDiagnosticOperation, st *engineState) (modelDiagnosticExerciseChild, diagnosticLoadObservation, error) {
	st.mu.Lock()
	proc, run, port, digest, config := st.proc, st.runID, st.port, st.launchDigest, st.ownerConfigLocked()
	owned := st.running && !st.adopted && proc != nil && run != "" && digest != ""
	st.mu.Unlock()
	if !owned {
		return modelDiagnosticExerciseChild{}, diagnosticLoadObservation{}, errModelDiagnosticIdentity
	}
	child := modelDiagnosticExerciseChild{Port: port, ModelID: op.reservation.ProfileKey, RunID: run, Done: proc.done}
	var expected diagnosticLoadObservation
	check := func(ctx context.Context) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := e.diagnosticActivationVeto(ctx, op.params.Engine); err != nil {
			return err
		}
		select {
		case <-proc.done:
			return errModelDiagnosticEngineExited
		default:
		}
		st.mu.Lock()
		current := st.running && !st.adopted && st.proc == proc && st.runID == run && st.port == port && st.launchDigest == digest && st.ownerConfigLocked() == config
		st.mu.Unlock()
		if !current {
			return errModelDiagnosticIdentity
		}
		loaded, err := e.diagnosticLoaded(ctx, op.params.Engine)
		if err != nil {
			return err
		}
		if len(loaded) != 1 || loaded[0] != child.ModelID {
			return errModelDiagnosticEngineExited
		}
		return nil
	}
	if err := check(ctx); err != nil {
		return child, expected, err
	}
	// HTTP load completion and the owned stdout/stderr pumps are independent.
	// Wait only for pending capture frames, keeping all identity refusals and
	// the same process/load boundary on every attempt.
	loadCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	ticker := time.NewTicker(25 * time.Millisecond)
	defer ticker.Stop()
	var observed diagnosticLoadObservation
	var err error
	for {
		observed, err = e.diagnosticReadLoad(loadCtx, st, port, child.ModelID, op.reservation.TargetMemory, op.loadStarted)
		if !errors.Is(err, errModelBufferRealizationPending) {
			break
		}
		select {
		case <-loadCtx.Done():
			if err := ctx.Err(); err != nil {
				return child, expected, err
			}
			return child, expected, errModelDiagnosticIdentity
		case <-ticker.C:
		}
		if err := check(loadCtx); err != nil {
			if loadCtx.Err() != nil && ctx.Err() == nil {
				return child, expected, errModelDiagnosticIdentity
			}
			return child, expected, err
		}
	}
	if err != nil {
		if loadCtx.Err() != nil && ctx.Err() == nil {
			return child, expected, errModelDiagnosticIdentity
		}
		return child, expected, err
	}
	expected = observed
	child.Check = func(ctx context.Context) error {
		if err := check(ctx); err != nil {
			return err
		}
		props, err := e.diagnosticReadLoad(ctx, st, port, child.ModelID, op.reservation.TargetMemory, op.loadStarted)
		if err != nil {
			return err
		}
		if !reflect.DeepEqual(props, expected) {
			return errModelDiagnosticIdentity
		}
		return check(ctx)
	}
	if err := child.Check(ctx); err != nil {
		return child, observed, err
	}
	return child, observed, nil
}

func (e *Executor) diagnosticReadProps(ctx context.Context, port int, model string) (diagnosticLoadObservation, error) {
	ctx, cancel := context.WithTimeout(ctx, modelsTimeout)
	defer cancel()
	query := url.Values{"model": {model}, "autoload": {"false"}}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/props?%s", port, query.Encode()), nil)
	if err != nil {
		return diagnosticLoadObservation{}, err
	}
	req.Header.Set(engineIdentityProbeHeader, "1")
	client := *e.clientFor(0)
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	resp, err := client.Do(req)
	if err != nil {
		return diagnosticLoadObservation{}, err
	}
	defer resp.Body.Close()
	const limit = 64 << 10
	raw, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil || len(raw) > limit || resp.StatusCode != http.StatusOK {
		return diagnosticLoadObservation{}, errModelDiagnosticIdentity
	}
	props := modelBufferReadProps(raw)
	if props.Context < 208 || props.Context > math.MaxInt32 || props.Slots < 1 || props.Slots > 8 {
		return diagnosticLoadObservation{}, errModelDiagnosticIdentity
	}
	return diagnosticLoadObservation{PerSlotContext: int32(props.Context), Slots: int32(props.Slots), BuildInfo: props.Build}, nil
}

func (e *Executor) diagnosticReadLoad(ctx context.Context, st *engineState, port int, model string, memory *modelconfig.TargetMemory, notBefore time.Time) (diagnosticLoadObservation, error) {
	st.mu.Lock()
	capture := st.bufferCapture
	current := capture != nil && capture.currentLocked(st)
	st.mu.Unlock()
	if !current {
		return diagnosticLoadObservation{}, errModelDiagnosticIdentity
	}
	boundary := capture.captureBoundary()
	observed, err := e.diagnosticReadProps(ctx, port, model)
	if err != nil {
		return observed, err
	}
	identity, rows, devices, err := e.modelBufferRealization(ctx, st, model, notBefore)
	if err != nil {
		return observed, err
	}
	observed.bufferIdentity = identity
	observed.PoolPlacement, err = modelBufferPoolPlacement(memory, rows, devices)
	st.mu.Lock()
	current = capture.currentLocked(st)
	st.mu.Unlock()
	if !current {
		return observed, errModelDiagnosticIdentity
	}
	if boundary != capture.captureBoundary() {
		return observed, errModelBufferRealizationPending
	}
	return observed, err
}
