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
	"strings"
	"time"
	"unicode"

	"nvpair-shared/engines"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

const desiredStateMaxBytes = modelconfig.MaxPayloadBytes

// This is cleanup evidence in the existing desired-state owner, not resumable
// measurement work. Started is saved BEFORE wake, and LoadedModels BEFORE load.
// AffectedModels records the pre-run unload set, which residents restore through
// their existing owner after the broker has released its drains.
type diagnosticCheckpointEngine struct {
	Engine          string   `json:"engine"`
	RunningBefore   bool     `json:"runningBefore"`
	ProcessRunID    string   `json:"processRunId,omitempty"`
	Started         bool     `json:"started"`
	AffectedModels  []string `json:"affectedModels"`
	LoadedModels    []string `json:"loadedModels"`
	CleanupComplete bool     `json:"cleanupComplete"`
}

type diagnosticCheckpoint struct {
	Run          modelconfig.DiagnosticRun       `json:"run"`
	EffectsFence string                          `json:"effectsFence"`
	Identity     *modelconfig.DiagnosticIdentity `json:"identity,omitempty"`
	BuildPin     string                          `json:"buildPin,omitempty"`
	Engines      []diagnosticCheckpointEngine    `json:"engines"`
	State        string                          `json:"state"`
}

func (c diagnosticCheckpoint) clone() diagnosticCheckpoint {
	c.Run = c.Run.Clone()
	if c.Identity != nil {
		identity := c.Identity.Clone()
		c.Identity = &identity
	}
	c.Engines = slices.Clone(c.Engines)
	for i := range c.Engines {
		c.Engines[i].AffectedModels = slices.Clone(c.Engines[i].AffectedModels)
		c.Engines[i].LoadedModels = slices.Clone(c.Engines[i].LoadedModels)
	}
	return c
}

func diagnosticCheckpointText(text string) bool {
	return len(text) <= 256 && strings.TrimSpace(text) == text && strings.IndexFunc(text, unicode.IsControl) < 0
}

func (c diagnosticCheckpoint) validate() error {
	if err := c.Run.Validate(); err != nil {
		return err
	}
	if c.Run.Engine != "llamacpp" || c.Run.Result != nil || c.EffectsFence == "" ||
		!diagnosticCheckpointText(c.EffectsFence) || !diagnosticCheckpointText(c.BuildPin) ||
		!slices.Contains([]string{"pending", "cleanup-failed", "cleanup-complete"}, c.State) ||
		c.State == "cleanup-failed" && c.Run.Outcome != "cleanup-failed" || len(c.Engines) > 64 {
		return errors.New("invalid diagnostic checkpoint")
	}
	if c.Identity != nil {
		if err := c.Identity.ValidateRequested(); err != nil {
			return err
		}
		if c.Identity.ConfigurationKey != c.Run.ConfigurationKey || c.Identity.ExecutionKey != c.Run.ExecutionKey {
			return errors.New("diagnostic checkpoint identity mismatch")
		}
	}
	// The configuration/execution key already binds the accepted inputs. A
	// complete hardware/realized identity may not exist until after native load;
	// never fabricate one just to persist an interrupted failed measurement.
	seen := map[string]bool{}
	for _, engine := range c.Engines {
		if !engines.ValidID(engine.Engine) || seen[engine.Engine] || !diagnosticCheckpointText(engine.ProcessRunID) ||
			engine.Started && (engine.RunningBefore || engine.Engine != c.Run.Engine) ||
			len(engine.LoadedModels) > 0 && (engine.Engine != c.Run.Engine || !engine.Started && !engine.RunningBefore) ||
			len(engine.AffectedModels) > 64 || len(engine.LoadedModels) > 64 ||
			c.State == "cleanup-complete" && !engine.CleanupComplete {
			return errors.New("invalid diagnostic engine cleanup evidence")
		}
		seen[engine.Engine] = true
		for _, models := range [][]string{engine.AffectedModels, engine.LoadedModels} {
			modelSeen := map[string]bool{}
			for _, model := range models {
				if !diagnosticCheckpointText(model) || nodepolicy.ValidateModelIdentifier(model) != nil || modelSeen[model] {
					return errors.New("invalid diagnostic model cleanup evidence")
				}
				modelSeen[model] = true
			}
		}
	}
	return modelconfig.CheckPayloadSize(c)
}

func (s *desiredStateStore) pendingDiagnostic() (*diagnosticCheckpoint, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.load()
	if err != nil || state.PendingDiagnostic == nil {
		return nil, err
	}
	checkpoint := state.PendingDiagnostic.clone()
	return &checkpoint, nil
}

func (s *desiredStateStore) setPendingDiagnostic(checkpoint diagnosticCheckpoint) error {
	if err := checkpoint.validate(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.load()
	if err != nil {
		return err
	}
	if s.path == "" {
		return errors.New("no user data directory is available")
	}
	if previous := state.PendingDiagnostic; previous != nil {
		before, after := previous.Run, checkpoint.Run
		if before.RunID != after.RunID || before.Engine != after.Engine || before.ModelID != after.ModelID ||
			before.ProfileKey != after.ProfileKey || before.ConfigurationKey != after.ConfigurationKey ||
			before.ExecutionKey != after.ExecutionKey ||
			previous.EffectsFence != checkpoint.EffectsFence ||
			previous.Identity != nil && (checkpoint.Identity == nil || !diagnosticRequestedMatches(*previous.Identity, *checkpoint.Identity)) ||
			previous.State == "cleanup-complete" && checkpoint.State != "cleanup-failed" && !reflect.DeepEqual(*previous, checkpoint) {
			return diagnosticConflict("pending cleanup identity cannot be replaced")
		}
		for _, old := range previous.Engines {
			index := slices.IndexFunc(checkpoint.Engines, func(next diagnosticCheckpointEngine) bool { return next.Engine == old.Engine })
			if index < 0 {
				return diagnosticConflict("pending engine cleanup cannot be discarded")
			}
			next := checkpoint.Engines[index]
			if old.RunningBefore != next.RunningBefore || old.Started && !next.Started ||
				old.CleanupComplete && !next.CleanupComplete ||
				old.ProcessRunID != "" && old.ProcessRunID != next.ProcessRunID ||
				!diagnosticModelsContain(next.AffectedModels, old.AffectedModels) || !diagnosticModelsContain(next.LoadedModels, old.LoadedModels) {
				return diagnosticConflict("pending process or model cleanup cannot be discarded")
			}
		}
	}
	copy := checkpoint.clone()
	state.PendingDiagnostic = &copy
	return s.save(state)
}

func diagnosticModelsContain(models, previous []string) bool {
	for _, model := range previous {
		if !slices.Contains(models, model) {
			return false
		}
	}
	return true
}

func (s *desiredStateStore) clearPendingDiagnostic(runID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.load()
	if err != nil {
		return err
	}
	if state.PendingDiagnostic != nil && (state.PendingDiagnostic.Run.RunID != runID || state.PendingDiagnostic.State != "cleanup-complete") {
		return diagnosticConflict("pending cleanup is not confirmed complete")
	}
	state.PendingDiagnostic = nil
	// Clearing is part of cleanup proof too. Do not downgrade directory-sync
	// failure to success through the ordinary non-checkpoint save path.
	data, err := json.Marshal(state)
	if err != nil {
		return err
	}
	return engineFileReplacer.WriteFile(s.path, data, 0o600)
}

// Reservation acceptance calls this under mutationAdmission after compare, but
// before publishing a usable reservation token or replying. It performs no effect.
func (e *Executor) checkpointDiagnosticReservation(p diagnosticReserveParams) error {
	if err := p.validate(); err != nil {
		return err
	}
	checkpoint := diagnosticCheckpoint{
		Run: modelconfig.DiagnosticRun{RunID: p.RunID, Engine: p.Engine, ModelID: p.ModelID, ProfileKey: p.ProfileKey,
			ConfigurationKey: p.ConfigurationKey, ExecutionKey: p.ExecutionKey, Phase: "reserving", Outcome: "running",
			StartedAt: time.Now().UTC().Format(time.RFC3339Nano)},
		EffectsFence: p.EffectsFence, State: "pending", Engines: []diagnosticCheckpointEngine{},
	}
	if e.diagnostic != nil && e.diagnostic.identity != nil {
		identity := e.diagnostic.identity.Clone()
		checkpoint.Identity = &identity
		checkpoint.BuildPin = identity.NativeBuildID
	}
	if e.diagnostic != nil {
		copy := checkpoint.clone()
		e.diagnostic.checkpoint = &copy
	}
	return e.desired.setPendingDiagnostic(checkpoint)
}

// The operation calls this before every effect/phase and on cleanup failure,
// carrying forward the full engine ledger. Failure forbids the effect.
func (e *Executor) updateDiagnosticCheckpoint(runID, token string, checkpoint diagnosticCheckpoint) error {
	e.mutationAdmission.Lock()
	defer e.mutationAdmission.Unlock()
	r := e.diagnostic
	if r == nil || !r.ready || r.params.RunID != runID || r.token != token || checkpoint.Run.RunID != runID {
		return diagnosticConflict("invalid checkpoint reservation authority")
	}
	if checkpoint.State != "pending" && checkpoint.State != "cleanup-failed" {
		return diagnosticConflict("only cleanup confirmation may finish a checkpoint")
	}
	r.workUnconfirmed = true
	copy := checkpoint.clone()
	r.checkpoint = &copy
	return e.desired.setPendingDiagnostic(checkpoint)
}

// Physical cleanup is durable before the broker records the measurement. Keep
// the same reservation until its explicit release after record publication.
func (e *Executor) completeDiagnosticCheckpoint(runID, token string, run modelconfig.DiagnosticRun) (err error) {
	e.mutationAdmission.Lock()
	defer e.mutationAdmission.Unlock()
	r := e.diagnostic
	if r == nil || !r.ready || r.params.RunID != runID || r.token != token || r.workUnconfirmed || e.diagnosticOperationsActiveLocked() || run.RunID != runID {
		return diagnosticConflict("diagnostic work is not confirmed stopped")
	}
	defer func() {
		if err != nil {
			r.workUnconfirmed = true
		}
	}()
	checkpoint, err := e.desired.pendingDiagnostic()
	if err != nil {
		return err
	}
	if checkpoint == nil || checkpoint.Run.RunID != runID {
		return diagnosticConflict("reservation checkpoint unavailable")
	}
	checkpoint.Run = run.Clone()
	checkpoint.State = "cleanup-complete"
	for i := range checkpoint.Engines {
		checkpoint.Engines[i].CleanupComplete = true
	}
	copy := checkpoint.clone()
	r.checkpoint = &copy
	return e.desired.setPendingDiagnostic(*checkpoint)
}

// Call after confirmDiagnosticWorkStopped and before releaseDiagnostic. A save
// failure re-latches the existing hold; callers must not release on that error.
func (e *Executor) clearDiagnosticCheckpoint(runID, token string) (err error) {
	e.mutationAdmission.Lock()
	defer e.mutationAdmission.Unlock()
	return e.clearDiagnosticCheckpointLocked(runID, token)
}

func (e *Executor) clearDiagnosticCheckpointLocked(runID, token string) (err error) {
	r := e.diagnostic
	if r == nil || !r.ready || r.params.RunID != runID || r.token != token || r.workUnconfirmed || e.diagnosticOperationsActiveLocked() {
		return diagnosticConflict("diagnostic work is not confirmed stopped")
	}
	defer func() {
		if err != nil {
			r.workUnconfirmed = true
		}
	}()
	checkpoint, err := e.desired.pendingDiagnostic()
	if err != nil {
		return err
	}
	if checkpoint == nil {
		return e.desired.clearPendingDiagnostic(runID)
	}
	if checkpoint.Run.RunID != runID {
		return diagnosticConflict("different pending cleanup")
	}
	if checkpoint.State != "cleanup-complete" {
		checkpoint.State = "cleanup-complete"
		for i := range checkpoint.Engines {
			checkpoint.Engines[i].CleanupComplete = true
		}
		copy := checkpoint.clone()
		r.checkpoint = &copy
		if err := e.desired.setPendingDiagnostic(*checkpoint); err != nil {
			return err
		}
	}
	return e.desired.clearPendingDiagnostic(runID)
}

// Manager startup calls this before dispatch/RestoreInstalled. It restores the
// SAME global admission owner with fresh authority, never a measurement runner.
// The operation owner serializes retry cleanup with accepted live diagnostics.
func (e *Executor) restoreDiagnosticCheckpoint() (*diagnosticCheckpoint, error) {
	e.mutationAdmission.Lock()
	defer e.mutationAdmission.Unlock()
	// A startup/installed-restore goroutine may run after a fresh broker
	// reservation. Its accepted checkpoint is not interrupted cleanup work.
	if r := e.diagnostic; r != nil && !r.workUnconfirmed {
		return nil, nil
	}
	checkpoint, err := e.desired.pendingDiagnostic()
	if err != nil {
		return checkpoint, err
	}
	if checkpoint == nil {
		if r := e.diagnostic; r != nil && r.workUnconfirmed && r.checkpoint != nil {
			copy := r.checkpoint.clone()
			checkpoint = &copy
			if err := e.desired.setPendingDiagnostic(copy); err != nil {
				return nil, err
			}
		} else {
			return nil, nil
		}
	}
	if e.diagnosticOperation != nil {
		return nil, diagnosticConflict("a live diagnostic operation owns cleanup")
	}
	if r := e.diagnostic; r != nil {
		if r.params.RunID != checkpoint.Run.RunID {
			return nil, diagnosticConflict("another diagnostic owns execution")
		}
		return checkpoint, nil
	}
	if err := e.diagnosticIdleLocked(); err != nil {
		return nil, err
	}
	release, err := e.trackMutationLocked("")
	if err != nil {
		return nil, err
	}
	run := checkpoint.Run
	r := &diagnosticReservation{
		params: diagnosticReserveParams{RunID: run.RunID, Engine: run.Engine, ModelID: run.ModelID, ProfileKey: run.ProfileKey,
			ConfigurationKey: run.ConfigurationKey, ExecutionKey: run.ExecutionKey, EffectsFence: checkpoint.EffectsFence},
		token: newOpID(), ready: true, settled: make(chan struct{}), release: release, workUnconfirmed: true,
	}
	if checkpoint.Identity != nil {
		identity := checkpoint.Identity.Clone()
		r.identity = &identity
	}
	copy := checkpoint.clone()
	r.checkpoint = &copy
	close(r.settled)
	e.diagnostic = r
	return checkpoint, nil
}

// stop is the Manager.runOp diagnostic sleep dispatch. Passing its scoped
// context into the existing sleepContext reuses exact-image orphan reclaim and
// termination verification; this core never launches or kills a process itself.
func (e *Executor) recoverDiagnosticCheckpoint(ctx context.Context, stop func(context.Context, string) error) error {
	checkpoint, err := e.restoreDiagnosticCheckpoint()
	if err != nil || checkpoint == nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, diagnosticCleanupBudget)
	defer cancel()
	e.mutationAdmission.Lock()
	r := e.diagnostic
	runID, token := r.params.RunID, r.token
	e.mutationAdmission.Unlock()
	failedPhase := checkpoint.Run.FailedPhase
	if failedPhase == "" {
		failedPhase = checkpoint.Run.Phase
	}
	checkpoint.Run.FailedPhase = failedPhase
	checkpoint.Run.Outcome = "failed"
	checkpoint.Run.Reason = "interrupted"
	checkpoint.Run.Phase = "restoring"
	var cleanupErrors []error
	// An interrupted terminal-clear write already contains conclusive cleanup.
	// No process or model effect is repeated in that case.
	if checkpoint.State != "cleanup-complete" {
		checkpoint.State = "pending"
		if err := e.desired.setPendingDiagnostic(*checkpoint); err != nil {
			return err
		}
		for i := range checkpoint.Engines {
			engine := &checkpoint.Engines[i]
			if engine.CleanupComplete {
				continue
			}
			if err := e.recoverDiagnosticEngine(ctx, *engine, runID, token, stop); err != nil {
				cleanupErrors = append(cleanupErrors, fmt.Errorf("recover diagnostic engine %s: %w", engine.Engine, err))
				continue
			}
			engine.CleanupComplete = true
			if err := e.desired.setPendingDiagnostic(*checkpoint); err != nil {
				cleanupErrors = append(cleanupErrors, err)
				break
			}
		}
	}
	// Failed history remains bound to the captured run identity even when
	// the profile/build has changed since the crash. It is never current proof.
	if checkpoint.Identity != nil {
		measured := modelconfig.MeasuredProfile{DiagnosticIdentity: checkpoint.Identity.Clone(), RunID: runID,
			RecordedAt: time.Now().UTC().Format(time.RFC3339Nano), Outcome: "failed", FailedPhase: failedPhase,
			Reason: "interrupted", BuildPin: checkpoint.BuildPin, Notes: []string{"Measurement interrupted; no measurement resumed."}}
		if err := e.modelMemoryProfiles.recordMeasured(checkpoint.Run.Engine, checkpoint.Run.ModelID, checkpoint.Run.ProfileKey, measured); err != nil {
			cleanupErrors = append(cleanupErrors, err)
		}
	}
	if failure := errors.Join(cleanupErrors...); failure != nil {
		checkpoint.State = "cleanup-failed"
		checkpoint.Run.Outcome = "cleanup-failed"
		checkpoint.Run.Reason = "interrupted; cleanup or failed-record persistence is unverified"
		return errors.Join(failure, e.desired.setPendingDiagnostic(*checkpoint))
	}
	if err := e.confirmDiagnosticWorkStopped(runID, token); err != nil {
		return err
	}
	// Release durably confirms/clears the checkpoint before retiring admission.
	// Retain terminal status only after that succeeds, so its idempotent release
	// acknowledgement cannot bypass retirement of the recovered live hold.
	if err := e.releaseDiagnostic(diagnosticReleaseParams{RunID: runID, Token: token}); err != nil {
		return err
	}
	e.retainRecoveredDiagnostic(*checkpoint, r)
	return nil
}

// The status handler consults this when no live operation matches. Read errors
// are returned, never converted to "none", so reconnect cannot release drains
// while the cleanup obligation is unreadable. Call without mutationAdmission.
func (e *Executor) diagnosticCheckpointStatus(runID string) (diagnosticOperationStatus, bool, error) {
	checkpoint, err := e.desired.pendingDiagnostic()
	if err != nil || checkpoint == nil {
		return diagnosticOperationStatus{}, false, err
	}
	if runID != "" && checkpoint.Run.RunID != runID {
		return diagnosticOperationStatus{}, false, nil
	}
	status := diagnosticOperationStatus{DiagnosticRun: checkpoint.Run.Clone(), State: checkpoint.State,
		WorkStopped: false}
	if checkpoint.Identity != nil {
		identity := checkpoint.Identity.Clone()
		status.Identity = &identity
	}
	e.mutationAdmission.Lock()
	if r := e.diagnostic; r != nil && r.ready && r.params.RunID == checkpoint.Run.RunID {
		status.Token = r.token
	}
	e.mutationAdmission.Unlock()
	return status, true, nil
}

func (e *Executor) saveDiagnosticOperation(op *modelDiagnosticOperation) error {
	if op.checkpoint == nil {
		return diagnosticConflict("diagnostic checkpoint unavailable")
	}
	e.mutationAdmission.Lock()
	op.checkpoint.Run = op.status.DiagnosticRun.Clone()
	if op.status.Identity != nil {
		identity := op.status.Identity.Clone()
		op.checkpoint.Identity = &identity
		op.checkpoint.BuildPin = identity.NativeBuildID
	}
	e.mutationAdmission.Unlock()
	return e.updateDiagnosticCheckpoint(op.params.RunID, op.params.Token, *op.checkpoint)
}

func (m *Manager) runDiagnosticLifecycle(ctx context.Context, engine string, wake bool) error {
	if err := m.exec.diagnosticMutationAllowed(ctx, engine); err != nil {
		return err
	}
	raw, err := json.Marshal(opParam{Engine: engine})
	if err != nil {
		return err
	}
	method := sleepMethod
	if wake {
		method = wakeMethod
	}
	return m.runOp(ctx, &Message{Method: method, Params: raw})
}

func (m *Manager) recoverDiagnosticBeforeRestore(ctx context.Context) error {
	m.diagnosticRecoveryMu.Lock()
	defer m.diagnosticRecoveryMu.Unlock()
	m.exec.mutationAdmission.Lock()
	live := m.exec.diagnosticOperation != nil
	m.exec.mutationAdmission.Unlock()
	if live {
		return diagnosticConflict("live diagnostic owns cleanup")
	}
	return m.exec.recoverDiagnosticCheckpoint(ctx, func(ctx context.Context, engine string) error { return m.runDiagnosticLifecycle(ctx, engine, false) })
}

func (m *Manager) abortRecoveredDiagnostic(p diagnosticReleaseParams) (diagnosticOperationStatus, error) {
	e := m.exec
	e.mutationAdmission.Lock()
	r := e.diagnostic
	authorized := r != nil && r.ready && r.params.RunID == p.RunID && r.token == p.Token
	e.mutationAdmission.Unlock()
	if !authorized {
		return diagnosticOperationStatus{}, diagnosticConflict("no matching recovered operation")
	}
	if err := m.recoverDiagnosticBeforeRestore(context.Background()); err != nil {
		return diagnosticOperationStatus{}, err
	}
	return e.diagnosticStatusWithCheckpoint(p.RunID)
}

func (e *Executor) retainRecoveredDiagnostic(checkpoint diagnosticCheckpoint, reservation *diagnosticReservation) {
	e.mutationAdmission.Lock()
	status := diagnosticOperationStatus{DiagnosticRun: checkpoint.Run.Clone(), State: "cleanup-complete", Token: reservation.token, WorkStopped: true}
	if checkpoint.Identity != nil {
		identity := checkpoint.Identity.Clone()
		status.Identity = &identity
	}
	status.Phase = "done"
	if e.diagnosticCompleted == nil {
		e.diagnosticCompleted = make(map[string]diagnosticCompletedRun)
	}
	if _, exists := e.diagnosticCompleted[checkpoint.Run.RunID]; !exists {
		e.diagnosticCompletedOrder = append(e.diagnosticCompletedOrder, checkpoint.Run.RunID)
	}
	e.diagnosticCompleted[checkpoint.Run.RunID] = diagnosticCompletedRun{
		token: reservation.token, Reservation: reservation.params, Status: status,
	}
	if len(e.diagnosticCompletedOrder) > diagnosticRetainedRuns {
		delete(e.diagnosticCompleted, e.diagnosticCompletedOrder[0])
		e.diagnosticCompletedOrder = e.diagnosticCompletedOrder[1:]
	}
	e.mutationAdmission.Unlock()
	// Startup recovery can finish after the broker read the pending checkpoint.
	// Wake its existing source-fenced reconciliation after retaining stop proof.
	e.notify(diagnosticChangedMethod, status)
}

func (e *Executor) recoverDiagnosticEngine(ctx context.Context, engine diagnosticCheckpointEngine, runID, token string, stop func(context.Context, string) error) error {
	if !engine.Started && len(engine.LoadedModels) == 0 {
		// Original models were unloaded, never loaded by this run. Reload belongs
		// to saved residents, after drains and the reservation have been released.
		return nil
	}
	scoped, err := e.diagnosticContext(ctx, engine.Engine, runID, token)
	if err != nil {
		return err
	}
	if !engine.Started && e.diagnosticCheckpointOwnedRun(engine) {
		if err := e.unloadDiagnosticCheckpointModels(scoped, engine); err == nil {
			return nil
		}
	}
	// Across manager death the owned handle is lost. Unload cannot authenticate
	// the old HTTP child, so use ordinary owned sleep/reclaim as the escalation.
	if stop == nil {
		return errors.New("diagnostic lifecycle sleep dispatch unavailable")
	}
	if err := stop(scoped, engine.Engine); err != nil {
		return err
	}
	st, err := e.state(engine.Engine)
	if err != nil {
		st, err = e.unsupportedStopState(engine.Engine)
	}
	if err != nil {
		return err
	}
	if !e.terminationVerifiedContext(ctx, st) {
		return errors.New("diagnostic process termination is unverified")
	}
	return nil
}

func (e *Executor) diagnosticCheckpointOwnedRun(engine diagnosticCheckpointEngine) bool {
	st, err := e.state(engine.Engine)
	if err != nil {
		return false
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	return engine.ProcessRunID != "" && st.runID == engine.ProcessRunID && st.proc != nil && st.running && !st.adopted && !st.stopping
}

func (e *Executor) unloadDiagnosticCheckpointModels(ctx context.Context, engine diagnosticCheckpointEngine) error {
	done, err := e.admitMutationContext(ctx, engine.Engine)
	if err != nil {
		return err
	}
	defer done()
	for _, model := range engine.LoadedModels {
		if _, err := e.ModelUnload(ctx, engine.Engine, model); err != nil {
			return err
		}
	}
	loaded, err := e.diagnosticLoaded(ctx, engine.Engine)
	if err != nil {
		return err
	}
	if !e.diagnosticCheckpointOwnedRun(engine) {
		return errors.New("loaded-model cleanup identity is unverified")
	}
	for _, model := range engine.LoadedModels {
		if slices.Contains(loaded, model) {
			return errors.New("diagnostic model remains loaded")
		}
	}
	return nil
}
