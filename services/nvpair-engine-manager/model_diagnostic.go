// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

const (
	diagnosticReserveMethod = "engine:diagnostic-reserve"
	diagnosticReleaseMethod = "engine:diagnostic-release"
)

// These are broker-only reservation contracts. TargetMemory is input evidence
// for the existing configuration-key owner, never activation authority. Exactly
// one of ConfigurationKey and ExecutionKey must identify the applied profile.
type diagnosticReserveParams struct {
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

type diagnosticReleaseParams struct {
	RunID string `json:"runId"`
	Token string `json:"token"`
}

type diagnosticReservationResult struct {
	RunID string `json:"runId"`
	Token string `json:"token"`
	State string `json:"state"`
}

type diagnosticReservation struct {
	params             diagnosticReserveParams
	token              string
	ready              bool
	settled            chan struct{}
	release            func()
	workUnconfirmed    bool
	pauseGeneration    uint64
	identity           *modelconfig.DiagnosticIdentity
	checkpoint         *diagnosticCheckpoint
	configurationPools []string
}

type diagnosticAuthorityKey struct{}
type diagnosticAuthority struct {
	executor        *Executor
	reservation     *diagnosticReservation
	engine          string
	measured        bool
	pauseGeneration uint64
}

func diagnosticConflict(reason string) error {
	return &codedError{code: -32010, message: "Memory measurement conflict: " + reason}
}

func (p diagnosticReserveParams) validate() error {
	for _, text := range []string{p.RunID, p.Engine, p.ModelID, p.ProfileKey, p.ConfigurationKey, p.ExecutionKey, p.EffectsFence} {
		if len(text) > 256 || strings.TrimSpace(text) != text || strings.ContainsAny(text, "\x00\r\n") {
			return fmt.Errorf("invalid diagnostic reservation text")
		}
	}
	if p.RunID == "" || p.Engine != "llamacpp" || p.ModelID == "" || p.ProfileKey == "" || p.EffectsFence == "" {
		return fmt.Errorf("runId, llama.cpp engine, modelId, profileKey and effectsFence are required")
	}
	if (p.ConfigurationKey == "") == (p.ExecutionKey == "") {
		return fmt.Errorf("exactly one configurationKey or executionKey is required")
	}
	return nil
}

func (p diagnosticReserveParams) sameRequest(other diagnosticReserveParams) bool {
	// A refreshed telemetry observation is not a different requested effect.
	return p.RunID == other.RunID && p.Engine == other.Engine && p.ModelID == other.ModelID &&
		p.ProfileKey == other.ProfileKey && p.ConfigurationKey == other.ConfigurationKey &&
		p.ExecutionKey == other.ExecutionKey && p.EffectsFence == other.EffectsFence && reflect.DeepEqual(p.Hardware.clone(), other.Hardware.clone())
}

func (r *diagnosticReservation) result() diagnosticReservationResult {
	return diagnosticReservationResult{RunID: r.params.RunID, Token: r.token, State: "reserved"}
}

// reserveDiagnostic closes admission before doing bounded observation I/O. The
// provisional global ticket has no usable token. A mismatch retires that ticket
// without starting, stopping, loading or writing policy. Observation I/O holds
// no admission/lifecycle lock; durable acceptance is serialized with admission.
func (e *Executor) reserveDiagnostic(ctx context.Context, p diagnosticReserveParams) (diagnosticReservationResult, error) {
	if err := p.validate(); err != nil {
		return diagnosticReservationResult{}, modelSettingsParamsError(err)
	}
	if err := ctx.Err(); err != nil {
		return diagnosticReservationResult{}, err
	}
	ctx, cancel := context.WithTimeout(ctx, modelInspectionBudget+modelsTimeout)
	defer cancel()
	e.mutationAdmission.Lock()
	if current := e.diagnostic; current != nil {
		if !current.params.sameRequest(p) {
			e.mutationAdmission.Unlock()
			return diagnosticReservationResult{}, diagnosticConflict("another reservation or different parameters for this runId")
		}
		if current.ready {
			result := current.result()
			e.mutationAdmission.Unlock()
			return result, nil
		}
		settled := current.settled
		e.mutationAdmission.Unlock()
		select {
		case <-ctx.Done():
			return diagnosticReservationResult{}, ctx.Err()
		case <-settled:
			return e.reserveDiagnostic(ctx, p)
		}
	}
	if err := e.diagnosticIdleLocked(); err != nil {
		e.mutationAdmission.Unlock()
		return diagnosticReservationResult{}, err
	}
	if err := e.mutationAllowedLocked(p.Engine); err != nil {
		e.mutationAdmission.Unlock()
		return diagnosticReservationResult{}, err
	}
	release, err := e.trackMutationLocked("")
	if err != nil {
		e.mutationAdmission.Unlock()
		return diagnosticReservationResult{}, err
	}
	reservation := &diagnosticReservation{params: p, token: newOpID(), settled: make(chan struct{}), release: release}
	e.nodeProjection.mu.Lock()
	reservation.pauseGeneration = e.nodeProjection.pauseGeneration
	e.nodeProjection.mu.Unlock()
	reservation.params.TargetMemory = p.TargetMemory.Clone()
	reservation.params.Hardware = p.Hardware.clone()
	e.diagnostic = reservation
	e.mutationAdmission.Unlock()

	err = e.compareDiagnosticReservation(ctx, p)
	var identity modelconfig.DiagnosticIdentity
	var configurationPools []string
	if err == nil {
		identityCtx := withDiagnosticHardware(e.modelMemoryContext(ctx, p.Engine, p.TargetMemory, false), p.Hardware)
		request := modelconfig.InspectRequest{Engine: p.Engine, ModelID: p.ModelID, ProfileKey: p.ProfileKey}
		var inspection modelconfig.InspectResult
		inspection, err = e.inspectModelArtifacts(identityCtx, request)
		if err == nil {
			identity, err = e.diagnosticRequestedIdentity(identityCtx, request, inspection)
			configurationPools = slices.Clone(inspection.ConfigurationPools)
		}
	}
	e.mutationAdmission.Lock()
	if err == nil {
		err = ctx.Err()
	}
	if err == nil && e.shuttingDown.Load() {
		err = fmt.Errorf("engine-manager is shutting down")
	}
	if err == nil {
		err = e.diagnosticPauseVeto()
	}
	if err == nil {
		e.nodeProjection.mu.Lock()
		changed := reservation.pauseGeneration != e.nodeProjection.pauseGeneration
		e.nodeProjection.mu.Unlock()
		if changed {
			err = diagnosticConflict("pause changed during reservation")
		}
	}
	if err == nil {
		reservation.identity = &identity
		reservation.configurationPools = configurationPools
		// A failed publication may already be visible after rename. Retain the
		// global hold, expose its authority through status, and recover it.
		err = e.checkpointDiagnosticReservation(p)
		reservation.ready = true
		if err != nil {
			reservation.workUnconfirmed = true
		}
	} else {
		e.diagnostic = nil
	}
	close(reservation.settled)
	e.mutationAdmission.Unlock()
	if err != nil {
		if !reservation.ready {
			release()
		}
		return diagnosticReservationResult{}, err
	}
	return reservation.result(), nil
}

// Caller holds mutationAdmission. Registrations are checked as well as mutation
// tickets, including loads already waiting for the lifecycle lock.
func (e *Executor) diagnosticIdleLocked() error {
	if e.diagnosticOperation != nil {
		return diagnosticConflict("diagnostic completion is in flight")
	}
	if e.shuttingDown.Load() {
		return fmt.Errorf("engine-manager is shutting down")
	}
	for _, active := range e.activeMutations {
		if active != 0 {
			return diagnosticConflict("an admitted mutation is in flight")
		}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, st := range e.engines {
		st.mu.Lock()
		busy := len(st.loads.active) != 0 || st.loads.stops != 0 || st.startCancel != nil || st.relaunch != nil
		st.mu.Unlock()
		if busy {
			return diagnosticConflict("engine execution is in flight")
		}
	}
	return nil
}

func (e *Executor) compareDiagnosticReservation(ctx context.Context, p diagnosticReserveParams) error {
	if err := e.diagnosticPauseVeto(); err != nil {
		return err
	}
	ctx = e.modelMemoryContext(ctx, p.Engine, p.TargetMemory, false)
	// Reservation needs identity, not an estimate or native-help discovery.
	result, err := e.inspectModelArtifacts(ctx, modelconfig.InspectRequest{Engine: p.Engine, ModelID: p.ModelID, ProfileKey: p.ProfileKey})
	if err != nil {
		return err
	}
	if result.Status == "busy" {
		return diagnosticConflict(errModelInspectionBusy.Error())
	}
	if p.ConfigurationKey != "" && result.ConfigurationKey != p.ConfigurationKey ||
		p.ExecutionKey != "" && result.ExecutionKey != p.ExecutionKey {
		return diagnosticConflict("applied profile identity changed; obtain a fresh preview")
	}
	fence, err := e.diagnosticEffectsFence(ctx)
	if err != nil {
		return err
	}
	if fence != p.EffectsFence {
		return diagnosticConflict("loaded models or applied configuration changed; obtain a fresh preview")
	}
	return e.diagnosticPauseVeto()
}

type diagnosticEngineEffects struct {
	Engine, Configuration, RunID string
	Running                      bool
	Loaded                       []string
}

// diagnosticEffectsFence is shared with the later broker preview owner. Missing
// loaded-model evidence for a running engine fails closed; stopped engines have
// an explicitly empty loaded set. The incarnation prevents cross-restart reuse.
func (e *Executor) diagnosticEffectsFence(ctx context.Context) (string, error) {
	// Reservation comparison and the pre-exercise check call this without an
	// admission/lifecycle lock or content lease held. Retry only this read;
	// callers still compare the fresh digest with the accepted effects fence.
	var fence string
	err := retryModelRead(ctx, func(retry *bool) error {
		var err error
		fence, err = e.diagnosticEffectsFenceOnce(ctx, retry)
		return err
	})
	return fence, err
}

// Lock acquisition here never waits: the enclosing whole-read retry owns the
// single contention deadline.
func (e *Executor) diagnosticEffectsFenceOnce(ctx context.Context, retry *bool) (string, error) {
	effects := make([]diagnosticEngineEffects, 0)
	for _, engine := range e.reg.Names() {
		st, err := e.state(engine)
		if err != nil {
			return "", diagnosticConflict("engine state unavailable")
		}
		if !st.opMu.TryLock() {
			*retry = true
			return "", diagnosticConflict("engine lifecycle is busy")
		}
		st.mu.Lock()
		entry := diagnosticEngineEffects{Engine: engine, Configuration: st.ownerConfigLocked(), RunID: st.runID, Running: st.running, Loaded: []string{}}
		// markStopped retains stopping to prevent passive re-adoption. With
		// the lifecycle lock held, that terminal flag is not an in-flight stop.
		uncertain := st.reclaimFailed || st.stopping && st.running || st.proc != nil && !st.running
		st.mu.Unlock()
		st.opMu.Unlock()
		if uncertain || entry.Configuration == "" {
			return "", diagnosticConflict("engine process state is unverified")
		}
		if entry.Running {
			action, ok := st.manifest.Actions["loaded_models"]
			if !ok || action.Result == nil {
				return "", diagnosticConflict("loaded-model evidence unavailable")
			}
			raw, err := e.Action(ctx, engine, "loaded_models", nil)
			if err != nil {
				return "", diagnosticConflict("loaded-model observation failed")
			}
			models, ok := extractStringsResult(raw, action.Result)
			if !ok {
				return "", diagnosticConflict("loaded-model observation is invalid")
			}
			entry.Loaded = append(entry.Loaded, models...)
			slices.Sort(entry.Loaded)
			entry.Loaded = slices.Compact(entry.Loaded)
		}
		if !st.opMu.TryLock() {
			*retry = true
			return "", diagnosticConflict("engine lifecycle changed during observation")
		}
		st.mu.Lock()
		changed := st.ownerConfigLocked() != entry.Configuration || st.runID != entry.RunID || st.running != entry.Running
		st.mu.Unlock()
		st.opMu.Unlock()
		if changed {
			*retry = true
			return "", diagnosticConflict("engine changed during loaded-model observation")
		}
		effects = append(effects, entry)
	}
	return ownerDigest(struct {
		Incarnation string
		Engines     []diagnosticEngineEffects
	}{e.managerIncarnation, effects})
}

// diagnosticContext grants cleanup authority for one named engine only. It
// cannot be manufactured by JSON or reused by a different Executor/reservation.
func (e *Executor) diagnosticContext(ctx context.Context, engine, runID, token string) (context.Context, error) {
	e.mutationAdmission.Lock()
	defer e.mutationAdmission.Unlock()
	r := e.diagnostic
	if r == nil || !r.ready || engine == "" || r.params.RunID != runID || r.token != token {
		return nil, diagnosticConflict("invalid reservation authority")
	}
	return context.WithValue(ctx, diagnosticAuthorityKey{}, diagnosticAuthority{
		executor: e, reservation: r, engine: engine, measured: engine == r.params.Engine,
		pauseGeneration: r.pauseGeneration,
	}), nil
}

func (e *Executor) diagnosticAuthorityLocked(ctx context.Context, engine string) bool {
	authority, ok := ctx.Value(diagnosticAuthorityKey{}).(diagnosticAuthority)
	return ok && authority.executor == e && authority.engine == engine && e.diagnostic != nil &&
		authority.reservation == e.diagnostic && e.diagnostic.ready
}

func (e *Executor) diagnosticMutationAllowed(ctx context.Context, engine string) error {
	e.mutationAdmission.Lock()
	defer e.mutationAdmission.Unlock()
	return e.mutationAllowedContextLocked(ctx, engine)
}

func (e *Executor) diagnosticPauseVeto() error {
	e.nodeProjection.mu.Lock()
	defer e.nodeProjection.mu.Unlock()
	if e.nodeProjection.projection == nil {
		return errNoProjection
	}
	if e.nodeProjection.projection.Policy.PauseMode == nodepolicy.PauseImmediate {
		return &localVetoError{code: codeLocalVeto, reason: "This node is halted. Resume it before measuring memory."}
	}
	return nil
}

// Used by activation owners that already carry reservation authority. Ordinary
// callers keep the normal pause gate; cleanup authority for another engine is
// never activation authority. Final process/HTTP dispatch must use this same
// scoped rule at its existing projection gate.
func (e *Executor) diagnosticActivationVeto(ctx context.Context, engine string) error {
	e.mutationAdmission.Lock()
	authorized := e.diagnosticAuthorityLocked(ctx, engine)
	measured := authorized && e.diagnostic.params.Engine == engine
	e.mutationAdmission.Unlock()
	if !authorized {
		if _, scoped := ctx.Value(diagnosticAuthorityKey{}).(diagnosticAuthority); scoped {
			return diagnosticConflict("expired or incorrectly scoped reservation authority")
		}
		return e.activationVeto(engine)
	}
	if !measured {
		return diagnosticConflict("only the measured engine may activate")
	}
	e.nodeProjection.mu.Lock()
	defer e.nodeProjection.mu.Unlock()
	return e.activationProjectionVetoLocked(ctx, engine)
}

// The operation owner calls begin before accepting work and confirms only after
// joining all work and proving physical cleanup. Release never accepts a caller's
// claim that work stopped. Any scoped mutation also marks work unconfirmed.
func (e *Executor) beginDiagnosticWork(runID, token string) error {
	e.mutationAdmission.Lock()
	defer e.mutationAdmission.Unlock()
	r := e.diagnostic
	if r == nil || !r.ready || r.params.RunID != runID || r.token != token {
		return diagnosticConflict("invalid reservation authority")
	}
	r.workUnconfirmed = true
	return nil
}

func (e *Executor) confirmDiagnosticWorkStopped(runID, token string) error {
	e.mutationAdmission.Lock()
	defer e.mutationAdmission.Unlock()
	r := e.diagnostic
	if r == nil || !r.ready || r.params.RunID != runID || r.token != token {
		return diagnosticConflict("invalid reservation authority")
	}
	if e.diagnosticOperationsActiveLocked() {
		return diagnosticConflict("diagnostic operations have not joined")
	}
	r.workUnconfirmed = false
	return nil
}

func (e *Executor) diagnosticOperationsActiveLocked() bool {
	for engine, active := range e.activeMutations {
		if engine != "" && active != 0 || engine == "" && active != 1 {
			return true
		}
	}
	return false
}

func (e *Executor) releaseDiagnostic(p diagnosticReleaseParams) error {
	e.mutationAdmission.Lock()
	r := e.diagnostic
	if r == nil || !r.ready || r.params.RunID != p.RunID || r.token != p.Token {
		if done, ok := e.diagnosticCompleted[p.RunID]; ok && done.token == p.Token && done.Status.WorkStopped {
			e.mutationAdmission.Unlock()
			return nil
		}
		e.mutationAdmission.Unlock()
		return diagnosticConflict("invalid reservation authority")
	}
	if r.workUnconfirmed || e.diagnosticOperationsActiveLocked() {
		e.mutationAdmission.Unlock()
		return diagnosticConflict("diagnostic work is not confirmed stopped")
	}
	if err := e.clearDiagnosticCheckpointLocked(p.RunID, p.Token); err != nil {
		e.mutationAdmission.Unlock()
		return err
	}
	e.diagnostic = nil
	e.mutationAdmission.Unlock()
	r.release()
	return nil
}

func (m *Manager) handleDiagnosticReservationRequest(ctx context.Context, msg *Message) bool {
	if m.handleDiagnosticEvidenceRequest(ctx, msg) {
		return true
	}
	if !msg.IsRequest() {
		return false
	}
	switch msg.Method {
	case diagnosticReserveMethod:
		go func() {
			var p diagnosticReserveParams
			if !m.parse(msg, &p) {
				return
			}
			result, err := m.exec.reserveDiagnostic(ctx, p)
			m.respondOrErr(msg, result, err)
		}()
		return true
	case diagnosticReleaseMethod:
		var p diagnosticReleaseParams
		if !m.parse(msg, &p) {
			return true
		}
		err := m.exec.releaseDiagnostic(p)
		m.respondOrErr(msg, nil, err)
		return true
	}
	return false
}
