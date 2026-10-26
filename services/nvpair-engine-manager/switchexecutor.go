// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	settings "nvpair-shared/enginesettings"
)

// Configuration is the manager portion of the broker's configuration token.
// The broker supplies node identity, journal generation and settings revisions.
type switchExecutorResult struct {
	settings.SwitchPrepareResult
	Configuration      settings.SwitchConfigurationSnapshot `json:"configuration"`
	ConfigurationFence string                               `json:"configurationFence"`
}

type switchBusyError struct{ engine string }

func (err *switchBusyError) Error() string {
	return fmt.Sprintf("engine %q is reserved by an engine switch (busy)", err.engine)
}

// admitMutation covers waiting for opMu as well as the operation itself. Prepare
// never waits for an admitted mutation while holding the short admission gate.
func (e *Executor) admitMutation(engine string) (func(), error) {
	e.switchAdmission.Lock()
	defer e.switchAdmission.Unlock()
	return e.admitMutationLocked(engine)
}

func (e *Executor) admitMutationLocked(engine string) (func(), error) {
	if e.switchMutations[""] > 0 {
		return nil, &switchBusyError{engine}
	}
	e.desired.mu.Lock()
	state, err := e.desired.load()
	e.desired.mu.Unlock()
	if err != nil {
		return nil, err
	}
	if cp := state.PendingSwitch; cp != nil && cp.Phase != settings.SwitchCheckpointReleased && (cp.Automatic || cp.SourceEngine == engine || cp.TargetEngine == engine) {
		return nil, &switchBusyError{engine}
	}
	return e.trackMutationLocked(engine), nil
}

func (e *Executor) trackMutationLocked(engine string) func() {
	if e.switchMutations == nil {
		e.switchMutations = make(map[string]int)
	}
	e.switchMutations[engine]++
	return func() { e.switchAdmission.Lock(); e.switchMutations[engine]--; e.switchAdmission.Unlock() }
}

// An explicit local Off is still a veto during a remote reservation. Persist
// it before waiting for the reserved lifecycle lock. Track the stop so a newly
// released reservation cannot be replaced while this veto finishes cleanup.
func (e *Executor) admitStop(engine string, origin intentOrigin, operationID string) (func(), error) {
	return e.admitStopScoped(engine, origin, operationID, false)
}

func (e *Executor) admitStopScoped(engine string, origin intentOrigin, operationID string, boundOnly bool) (func(), error) {
	e.switchAdmission.Lock()
	defer e.switchAdmission.Unlock()
	e.desired.mu.Lock()
	state, err := e.desired.load()
	if cold := e.boundActivation; err == nil && cold != nil && cold.engine == engine && origin == intentLocal && (operationID == "" || boundOnly && operationID == cold.operationID) {
		state.setIntent(engine, false, intentLocal)
		err = e.desired.save(state)
		e.desired.mu.Unlock()
		e.remoteWork.cancelEngine(engine, &localVetoError{code: codeLocalVeto, reason: engineOffVeto(engine)})
		if err != nil {
			return nil, err
		}
		e.emitIntent()
		e.switchMutations[""]++ // retain exclusion until physical Stop completes
		done := e.trackMutationLocked(engine)
		return func() { e.switchAdmission.Lock(); e.switchMutations[""]--; e.switchAdmission.Unlock(); done() }, nil
	}
	if boundOnly {
		e.desired.mu.Unlock()
		return nil, err
	}
	if err == nil && operationID != "" {
		cp := state.PendingSwitch
		if origin != intentLocal || cp == nil || cp.OperationID != operationID || cp.Phase == settings.SwitchCheckpointReleased || !cp.YieldsToLocalOff() || (cp.SourceEngine != engine && cp.TargetEngine != engine) {
			// The broker admitted this Off against a particular reservation.
			// Once that reservation is gone, neither its veto nor its physical
			// stop may affect a later switch or an explicit local On.
			e.desired.mu.Unlock()
			return nil, nil
		}
	}
	if err == nil && origin == intentLocal {
		if cp := state.PendingSwitch; cp != nil && cp.Phase != settings.SwitchCheckpointReleased && cp.YieldsToLocalOff() && (cp.SourceEngine == engine || cp.TargetEngine == engine) {
			state.setIntent(engine, false, intentLocal)
			if engine == cp.SourceEngine {
				cp.SourceLocalOff = true
				cp.Source.TerminationVerified = false
			} else {
				cp.TargetLocalOff = true
				cp.Target.TerminationVerified = false
			}
			cp.Error = switchFailure(codeLocalVeto, settings.SwitchErrorRemoteVeto, &localVetoError{code: codeLocalVeto, reason: engineOffVeto(engine)})
			err = e.desired.save(state)
			e.desired.mu.Unlock()
			e.remoteWork.cancelEngine(cp.TargetEngine, &localVetoError{code: codeLocalVeto, reason: engineOffVeto(engine)})
			if err != nil {
				return nil, err
			}
			e.emitIntent()
			return e.trackMutationLocked(engine), nil
		}
	}
	e.desired.mu.Unlock()
	if err != nil {
		return nil, err
	}
	return e.admitMutationLocked(engine)
}

func switchCode(err error) int {
	var veto *localVetoError
	if errors.As(err, &veto) {
		return veto.code
	}
	return rpcCode(err)
}

func switchFailure(code int, kind string, err error) *settings.SwitchError {
	if code == codeEngineLocked {
		kind = settings.SwitchErrorEngineLocked
	} else if code == codeLocalVeto {
		kind = settings.SwitchErrorRemoteVeto
	} else if kind == settings.SwitchErrorRemoteVeto {
		kind = settings.SwitchErrorValidation
	}
	message := err.Error()
	if len(message) > settings.MaxSwitchErrorMessageBytes {
		message = message[:settings.MaxSwitchErrorMessageBytes]
	}
	return &settings.SwitchError{Code: code, Kind: kind, Message: message}
}

func switchIntent(state desiredStateFile, engine string) string {
	on, known := state.Engines[engine]
	if !known {
		return settings.SwitchIntentUnknown
	}
	if on {
		return settings.SwitchIntentOn
	}
	return settings.SwitchIntentOff
}

func switchDigest(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

// Caller holds switchAdmission; admitted mutators cannot change the snapshot.
func (e *Executor) switchConfiguration(state desiredStateFile) (settings.SwitchConfigurationSnapshot, error) {
	config := settings.SwitchConfigurationSnapshot{ManagerIncarnation: e.switchIncarnation}
	ch, closeSub, ok := e.settingsHub.Subscribe()
	if !ok {
		return config, errors.New("settings snapshot unavailable")
	}
	snapshots := <-ch
	closeSub()
	revisions := make(map[string]uint64)
	for _, snapshot := range snapshots {
		revisions[snapshot.Engine] = snapshot.Revision
	}
	var fingerprints []string
	for _, name := range e.reg.Names() {
		if e.switchMutations[name] != 0 {
			return config, &switchBusyError{name}
		}
		manifest, _ := e.reg.Get(name)
		digest, err := switchDigest(manifest)
		if err != nil {
			return config, err
		}
		if st, err := e.state(name); err == nil {
			// The admission gate and zero mutation count protect configuration.
			// A switch step may own opMu, but changes only runtime observations.
			st.mu.Lock()
			digest, err = switchDigest(struct {
				Manifest              *Manifest
				Runtime               Runtime
				InstallDir, ModelsDir string
			}{manifest, st.plat.Runtime, st.installDir, st.modelsDir})
			st.mu.Unlock()
			if err != nil {
				return config, err
			}
		}
		fingerprints = append(fingerprints, digest)
		config.Engines = append(config.Engines, settings.SwitchConfigurationEngine{Engine: name, SettingsRevision: revisions[name], ConfigDigest: digest, Intent: switchIntent(state, name), Provenance: string(state.OffBy[name])})
	}
	var err error
	config.RegistryRuntimeFingerprint, err = switchDigest(fingerprints)
	return config, err
}

func validSwitchEnvelope(p settings.SwitchPrepareRequest) error {
	r := settings.SwitchRequest{RequestID: p.OperationID, SourceEngine: p.SourceEngine, TargetEngine: p.TargetEngine, ExpectedRevision: p.ConfigurationFence}
	if err := r.Validate(); err != nil {
		return err
	}
	if p.Origin != settings.SwitchOriginLocal && p.Origin != settings.SwitchOriginRemote {
		return errors.New("invalid switch origin")
	}
	if p.Caller == "" || p.PayloadHash == "" || p.Deadline.IsZero() {
		return errors.New("caller, payload hash and deadline are required")
	}
	if len(p.PayloadHash) != len("sha256:")+64 || p.PayloadHash[:7] != "sha256:" {
		return errors.New("invalid switch payload hash")
	}
	if _, err := hex.DecodeString(p.PayloadHash[7:]); err != nil {
		return errors.New("invalid switch payload hash")
	}
	return nil
}

// SwitchPrepare also supports a read-only configuration snapshot when the
// operation ID is empty. No reservation is created by that request.
func (e *Executor) SwitchPrepare(ctx context.Context, p settings.SwitchPrepareRequest) switchExecutorResult {
	if p.OperationID != "" && p.Automatic {
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(ctx, p.Deadline)
		defer cancel()
	}
	if p.OperationID != "" {
		e.switchMu.Lock()
		defer e.switchMu.Unlock()
	}
	e.switchAdmission.Lock()
	admissionHeld := true
	defer func() {
		if admissionHeld {
			e.switchAdmission.Unlock()
		}
	}()
	result := switchExecutorResult{}
	reserved := false
	fail := func(code int, kind string, err error) switchExecutorResult {
		result.Error = switchFailure(code, kind, err)
		if reserved {
			// A remote prepare refusal still needs the broker's final veto
			// reconciliation before releasing its provisional reservation.
			if !p.YieldsToLocalOff() {
				result.Checkpoint.Phase = settings.SwitchCheckpointReleased
			}
			result.Checkpoint.Error = result.Error
			if saveErr := e.saveSwitch(result.Checkpoint, "", false); saveErr != nil {
				result.Error = switchFailure(-32000, settings.SwitchErrorStorage, saveErr)
			}
		}
		if result.Error.Kind == settings.SwitchErrorStorage {
			e.desired.mu.Lock()
			state, loadErr := e.desired.load()
			e.desired.mu.Unlock()
			result.Checkpoint = nil
			if loadErr == nil {
				result.Checkpoint = state.PendingSwitch
			}
		}
		return result
	}
	e.desired.mu.Lock()
	state, err := e.desired.load()
	e.desired.mu.Unlock()
	if err != nil {
		return fail(-32000, settings.SwitchErrorStorage, err)
	}
	result.Checkpoint = state.PendingSwitch
	if p.OperationID != "" && p.ManagerIncarnation != e.switchIncarnation {
		return fail(-32000, settings.SwitchErrorInterrupted, errors.New("switch manager incarnation changed; reconcile before continuing"))
	}
	if cp := state.PendingSwitch; p.OperationID != "" && cp != nil && cp.OperationID == p.OperationID {
		if cp.SwitchPrepareRequest != p {
			return fail(-32602, settings.SwitchErrorReusedID, errors.New("operation ID reused with different switch parameters"))
		}
		result.Observed = settings.SwitchObserved{Source: cp.Source, Target: cp.Target}
		result.Error = cp.Error
		return result
	}
	config, err := e.switchConfiguration(state)
	if err != nil {
		return fail(-32000, settings.SwitchErrorBusy, err)
	}
	result.Configuration = config
	result.ConfigurationFence = settings.SwitchConfigurationToken(config)
	if p.OperationID == "" {
		return result
	}
	if err := validSwitchEnvelope(p); err != nil {
		return fail(-32602, settings.SwitchErrorInvalidRequest, err)
	}
	if cp := state.PendingSwitch; cp != nil && cp.Phase != settings.SwitchCheckpointReleased {
		return fail(-32000, settings.SwitchErrorBusy, errors.New("another switch requires reconciliation"))
	}
	if e.shuttingDown.Load() || ctx.Err() != nil || !p.Deadline.After(time.Now()) {
		return fail(-32000, settings.SwitchErrorInterrupted, errors.New("switch deadline expired or manager stopping"))
	}
	if p.ConfigurationFence != result.ConfigurationFence {
		return fail(-32010, settings.SwitchErrorConfigurationConflict, errors.New("manager configuration changed"))
	}
	if p.Automatic && p.Deadline.After(time.Now().Add(settings.AutomaticSwitchBudget)) {
		return fail(-32602, settings.SwitchErrorInvalidRequest, errors.New("automatic deadline exceeds preparation budget"))
	}
	source, err := e.state(p.SourceEngine)
	if err != nil {
		return fail(-32000, settings.SwitchErrorValidation, err)
	}
	target, err := e.state(p.TargetEngine)
	if err != nil {
		return fail(-32000, settings.SwitchErrorValidation, err)
	}
	for _, st := range []*engineState{source, target} {
		st.mu.Lock()
		port := st.port
		st.mu.Unlock()
		if port < 0 || port > 65535 {
			return fail(-32000, settings.SwitchErrorValidation, errors.New("engine server port is invalid"))
		}
	}
	if err := e.switchTargetVeto(p.Origin, p.TargetEngine); err != nil {
		return fail(switchCode(err), settings.SwitchErrorRemoteVeto, err)
	}
	if err := e.automaticSwitchSource(p); err != nil {
		return fail(-32000, settings.SwitchErrorValidation, err)
	}
	cp := &settings.SwitchCheckpoint{SwitchPrepareRequest: p, Phase: settings.SwitchCheckpointReserved, SourceProvenance: string(state.OffBy[p.SourceEngine]), TargetProvenance: string(state.OffBy[p.TargetEngine])}
	cp.Source = switchRuntimeObserved(source, switchIntent(state, p.SourceEngine), false)
	cp.Target = switchRuntimeObserved(target, switchIntent(state, p.TargetEngine), false)
	cp.PreSource, cp.PreTarget = cp.Source, cp.Target
	if err := e.saveSwitch(cp, "", false); err != nil {
		return fail(-32000, settings.SwitchErrorStorage, err)
	}
	result.Checkpoint = cp
	reserved = true
	// Provisional reservation is durable before probing. Long validation work
	// never holds the mutation admission gate; a refusal releases it durably.
	e.switchAdmission.Unlock()
	admissionHeld = false
	for _, pair := range []struct {
		name string
		st   *engineState
	}{{p.SourceEngine, source}, {p.TargetEngine, target}} {
		pair.st.opMu.Lock()
		installed, detectErr := e.Detect(pair.name)
		pair.st.mu.Lock()
		port := pair.st.port
		pair.st.mu.Unlock()
		presence := e.reconcilePresence(ctx, pair.name, pair.st, installed, port, false)
		pair.st.opMu.Unlock()
		if detectErr != nil {
			return fail(-32000, settings.SwitchErrorValidation, detectErr)
		}
		if pair.name == p.TargetEngine && !installed {
			return fail(-32000, settings.SwitchErrorUnprepared, errors.New("target engine is not installed; prepare it locally first"))
		}
		if pair.name == p.TargetEngine && pair.st.plat.Runtime.Ready == nil {
			return fail(-32000, settings.SwitchErrorValidation, errors.New("target engine has no server readiness probe"))
		}
		if err := e.reservedPortError(port); err != nil {
			return fail(-32000, settings.SwitchErrorValidation, err)
		}
		pair.st.mu.Lock()
		running := pair.st.running
		pair.st.mu.Unlock()
		if pair.name == p.TargetEngine && (running || presence.Identified) {
			return fail(-32000, settings.SwitchErrorTargetRunning, errors.New("target engine is already running"))
		}
		if pair.name == p.TargetEngine && presence.Occupied {
			return fail(-32000, settings.SwitchErrorValidation, errors.New("target port is occupied or ownership is unverified"))
		}
	}
	source.mu.Lock()
	sourcePort := source.port
	source.mu.Unlock()
	target.mu.Lock()
	targetPort := target.port
	target.mu.Unlock()
	if sourcePort > 0 && sourcePort == targetPort {
		return fail(-32000, settings.SwitchErrorValidation, errors.New("source and target server ports conflict"))
	}
	cp.Source = e.switchObserved(source, switchIntent(state, p.SourceEngine), false)
	cp.Target = e.switchObserved(target, switchIntent(state, p.TargetEngine), false)
	cp.PreSource, cp.PreTarget = cp.Source, cp.Target
	e.switchAdmission.Lock()
	admissionHeld = true
	if err := e.automaticSwitchSource(p); err != nil {
		return fail(-32000, settings.SwitchErrorValidation, err)
	}
	if err := e.switchTargetVeto(p.Origin, p.TargetEngine); err != nil {
		return fail(switchCode(err), settings.SwitchErrorRemoteVeto, err)
	}
	if err := e.saveSwitch(cp, "", false); err != nil {
		return fail(-32000, settings.SwitchErrorStorage, err)
	}
	if cp.Error != nil {
		return fail(cp.Error.Code, cp.Error.Kind, errors.New(cp.Error.Message))
	}
	result.Observed = settings.SwitchObserved{Source: cp.Source, Target: cp.Target}
	return result
}

func (e *Executor) switchObserved(st *engineState, intent string, terminated bool) settings.SwitchEngineState {
	st.mu.Lock()
	running, healthy, port := st.running, st.healthy, st.port
	st.mu.Unlock()
	now := time.Now().UTC()
	if !running && port > 0 && !terminated {
		ctx, cancel := context.WithTimeout(context.Background(), presenceRefusalWindow)
		listener := probeListener(ctx, st.plat.Runtime.Ready, port)
		cancel()
		if listener == listenerProbeReachable {
			running, healthy = true, false
		}
		if listener == listenerProbeIndeterminate {
			return settings.SwitchEngineState{Intent: intent, ObservedAt: &now}
		}
	}
	return settings.SwitchEngineState{Intent: intent, Running: &running, Healthy: &healthy, ObservedAt: &now, TerminationVerified: terminated && !running}
}

func switchRuntimeObserved(st *engineState, intent string, terminated bool) settings.SwitchEngineState {
	st.mu.Lock()
	running, healthy := st.running, st.healthy
	st.mu.Unlock()
	now := time.Now().UTC()
	return settings.SwitchEngineState{Intent: intent, Running: &running, Healthy: &healthy, ObservedAt: &now, TerminationVerified: terminated && !running}
}

// saveSwitch changes intent and checkpoint in one durable replacement.
func (e *Executor) saveSwitch(cp *settings.SwitchCheckpoint, engine string, on bool) error {
	e.desired.mu.Lock()
	defer e.desired.mu.Unlock()
	if e.desired.path == "" {
		return errors.New("no user data directory is available")
	}
	state, err := e.desired.load()
	if err != nil {
		return err
	}
	if engine != "" {
		if on && state.Locked[engine] {
			return &localVetoError{code: codeEngineLocked, reason: engineLockedVeto(engine)}
		}
		origin := intentLocal
		if cp.Origin == settings.SwitchOriginRemote {
			origin = intentPeer
		}
		state.setIntent(engine, on, origin)
	}
	// A veto may arrive while a step owns its private decoded checkpoint.
	// Never overwrite that durable cancellation with the step's older copy.
	if previous := state.PendingSwitch; previous != nil && previous.OperationID == cp.OperationID && previous.Error != nil && (previous.Error.Code == codeLocalVeto || previous.Error.Code == codeEngineLocked) {
		cp.Error = previous.Error
		if previous.SourceLocalOff && !cp.SourceLocalOff {
			cp.SourceLocalOff, cp.Source = true, previous.Source
		}
		if previous.TargetLocalOff && !cp.TargetLocalOff {
			cp.TargetLocalOff, cp.Target = true, previous.Target
		}
	}
	ports := e.switchPorts(cp)
	if cp.Automatic && state.PendingSwitch != nil && state.PendingSwitch.OperationID == cp.OperationID && state.PendingSwitchPorts != nil {
		ports.SourcePID, ports.SourceRunID = state.PendingSwitchPorts.SourcePID, state.PendingSwitchPorts.SourceRunID
	}
	state.PendingSwitch = cp
	state.PendingSwitchPorts = ports
	return e.desired.save(state)
}

func (e *Executor) switchPorts(cp *settings.SwitchCheckpoint) *switchCheckpointPorts {
	ports := &switchCheckpointPorts{}
	if st, err := e.state(cp.SourceEngine); err == nil {
		st.mu.Lock()
		ports.Source = st.port
		if cp.Automatic {
			ports.SourceRunID = st.runID
			if st.proc != nil && st.proc.cmd != nil && st.proc.cmd.Process != nil {
				ports.SourcePID = st.proc.cmd.Process.Pid
			}
		}
		st.mu.Unlock()
	}
	if st, err := e.state(cp.TargetEngine); err == nil {
		st.mu.Lock()
		ports.Target = st.port
		st.mu.Unlock()
	}
	return ports
}

func (e *Executor) SwitchStep(ctx context.Context, p settings.SwitchStepRequest) settings.SwitchStepResult {
	e.switchMu.Lock()
	defer e.switchMu.Unlock()
	return e.switchStepLocked(ctx, p, false, nil)
}

func (e *Executor) SwitchReconcile(ctx context.Context, p settings.SwitchReconcileRequest) settings.SwitchReconcileResult {
	if p.Automatic {
		// The stdio reader supplies the manager's lifetime context. Carry the
		// broker's bounded cleanup window separately from immutable identity.
		deadline := time.Now().Add(settings.SwitchDrainBudget)
		limit := p.Deadline
		if p.CleanupDeadline != nil {
			limit = *p.CleanupDeadline
		}
		if limit.Before(deadline) {
			deadline = limit
		}
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(ctx, deadline)
		defer cancel()
	}
	e.switchMu.Lock()
	defer e.switchMu.Unlock()
	return e.switchStepLocked(ctx, settings.SwitchStepRequest{SwitchPrepareRequest: p.SwitchPrepareRequest}, true, p.LocalVeto)
}

func (e *Executor) switchStepLocked(ctx context.Context, p settings.SwitchStepRequest, reconcile bool, localVeto *settings.SwitchLocalVeto) settings.SwitchStepResult {
	// Release must read and replace under the same gate as durable local Off
	// admission. A veto written after an earlier reconcile cannot slip through.
	if p.Step == settings.SwitchStepRelease {
		e.switchAdmission.Lock()
		defer e.switchAdmission.Unlock()
	}
	result := settings.SwitchStepResult{}
	fail := func(kind string, err error) settings.SwitchStepResult {
		code := switchCode(err)
		if code == codeLocalVeto {
			kind = settings.SwitchErrorRemoteVeto
		}
		result.Error = switchFailure(code, kind, err)
		if kind == settings.SwitchErrorStorage {
			e.desired.mu.Lock()
			state, loadErr := e.desired.load()
			e.desired.mu.Unlock()
			result.Checkpoint = nil
			if loadErr == nil {
				result.Checkpoint = state.PendingSwitch
				for _, pair := range []struct {
					name        string
					observation *settings.SwitchEngineState
				}{{p.SourceEngine, &result.Observed.Source}, {p.TargetEngine, &result.Observed.Target}} {
					if st, stateErr := e.state(pair.name); stateErr == nil {
						*pair.observation = e.switchObserved(st, switchIntent(state, pair.name), e.switchTerminated(st))
					}
				}
			}
		}
		return result
	}
	e.desired.mu.Lock()
	state, err := e.desired.load()
	e.desired.mu.Unlock()
	if err != nil {
		return fail(settings.SwitchErrorStorage, err)
	}
	cp := state.PendingSwitch
	result.Checkpoint = cp
	if cp == nil || cp.SwitchPrepareRequest != p.SwitchPrepareRequest {
		return fail(settings.SwitchErrorInvalidRequest, &codedError{code: -32602, message: "switch checkpoint identity mismatch"})
	}
	if localVeto != nil && cp.Phase != settings.SwitchCheckpointReleased {
		if !cp.YieldsToLocalOff() || (localVeto.Pause && localVeto.Engine != "") || (!localVeto.Pause && localVeto.Engine != cp.SourceEngine && localVeto.Engine != cp.TargetEngine) {
			return fail(settings.SwitchErrorInvalidRequest, &codedError{code: -32602, message: "invalid operation-specific local veto"})
		}
		// Reload while holding desired.mu so a concurrent local Off's intent
		// cannot be lost when the broker's earlier latch is delivered.
		e.desired.mu.Lock()
		state, err = e.desired.load()
		if err == nil {
			cp = state.PendingSwitch
			reason := nodePausedVeto
			if !localVeto.Pause {
				reason = engineOffVeto(localVeto.Engine)
				state.setIntent(localVeto.Engine, false, intentLocal)
				if localVeto.Engine == cp.SourceEngine {
					cp.SourceLocalOff = true
				} else {
					cp.TargetLocalOff = true
				}
			}
			cp.Error = switchFailure(codeLocalVeto, settings.SwitchErrorRemoteVeto, &localVetoError{code: codeLocalVeto, reason: reason})
			err = e.desired.save(state)
		}
		e.desired.mu.Unlock()
		if err != nil {
			return fail(settings.SwitchErrorStorage, err)
		}
		result.Checkpoint = cp
		e.emitIntent()
	}
	if !reconcile && (p.Step == settings.SwitchStepSourceStop || p.Step == settings.SwitchStepTargetStart) && p.ManagerIncarnation != e.switchIncarnation {
		return fail(settings.SwitchErrorInterrupted, &codedError{code: -32000, message: "switch manager incarnation changed; reconcile before continuing"})
	}
	if reconcile && cp.Phase != settings.SwitchCheckpointReleased && state.PendingSwitchPorts != nil {
		for _, pair := range []struct {
			name string
			port int
		}{{cp.SourceEngine, state.PendingSwitchPorts.Source}, {cp.TargetEngine, state.PendingSwitchPorts.Target}} {
			if st, err := e.state(pair.name); err == nil {
				st.mu.Lock()
				if pair.port > 0 {
					st.port = pair.port
				}
				st.mu.Unlock()
			}
		}
	}
	result.Observed = settings.SwitchObserved{Source: cp.Source, Target: cp.Target}
	if cp.Phase == settings.SwitchCheckpointReleased {
		if p.Step != settings.SwitchStepRelease {
			result.Error = cp.Error
		}
		return result
	}
	if p.Step == settings.SwitchStepRelease {
		paused, _ := e.nodeProjection.managementBlocked()
		if cp.YieldsToLocalOff() {
			for _, pair := range []struct {
				veto        bool
				observation settings.SwitchEngineState
			}{{cp.SourceLocalOff, cp.Source}, {cp.TargetLocalOff, cp.Target}} {
				if pair.veto && !pair.observation.TerminationVerified {
					return fail(settings.SwitchErrorBusy, errors.New("reconcile local Off before release; reservation retained"))
				}
			}
		}
		if cp.TargetCommitted && ((cp.YieldsToLocalOff() && cp.Error != nil && cp.Error.Code == codeLocalVeto) || paused) {
			return fail(settings.SwitchErrorBusy, errors.New("reconcile vetoed committed target before release; reservation retained"))
		}
		if cp.Phase == settings.SwitchCheckpointStopPending || cp.Phase == settings.SwitchCheckpointStartPending {
			return fail(settings.SwitchErrorBusy, errors.New("reconcile incomplete switch step before release"))
		}
		cp.Phase = settings.SwitchCheckpointReleased
		if err := e.saveSwitch(cp, "", false); err != nil {
			return fail(settings.SwitchErrorStorage, err)
		}
		return result
	}
	// An explicit veto while the broker still owns a remote reservation also
	// owns physical termination even if drain failed before source-stop.
	if reconcile && cp.YieldsToLocalOff() {
		for _, pair := range []struct {
			name        string
			veto        bool
			observation *settings.SwitchEngineState
		}{{cp.SourceEngine, cp.SourceLocalOff, &cp.Source}, {cp.TargetEngine, cp.TargetLocalOff, &cp.Target}} {
			if !pair.veto {
				continue
			}
			st, stopErr := e.state(pair.name)
			if stopErr != nil {
				return fail(settings.SwitchErrorInterrupted, stopErr)
			}
			st.opMu.Lock()
			stopErr = e.cleanupSwitchRecovery(ctx, st, pair.name)
			terminated := e.switchTerminated(st)
			*pair.observation = e.switchObserved(st, settings.SwitchIntentOff, terminated)
			st.opMu.Unlock()
			if saveErr := e.saveSwitch(cp, "", false); saveErr != nil {
				return fail(settings.SwitchErrorStorage, saveErr)
			}
			result.Observed = settings.SwitchObserved{Source: cp.Source, Target: cp.Target}
			if stopErr != nil || !terminated {
				return fail(settings.SwitchErrorInterrupted, errors.Join(stopErr, errors.New("local Off termination must be verified; reservation retained")))
			}
		}
	}
	// An explicit local Off during a remote reservation, or either-origin pause
	// during any reservation, cancels an already-committed target. Ordinary
	// restart recovery otherwise preserves committed On.
	vetoedCommittedTarget := false
	if reconcile && cp.TargetCommitted {
		paused, _ := e.nodeProjection.managementBlocked()
		if (cp.YieldsToLocalOff() && cp.Error != nil && cp.Error.Code == codeLocalVeto) || paused {
			vetoedCommittedTarget = true
			cp.TargetCommitted = false
			cp.Target.Intent = settings.SwitchIntentOff
			cp.Phase = settings.SwitchCheckpointStartPending
			if cp.Error == nil || cp.Error.Code != codeLocalVeto {
				cp.Error = switchFailure(codeLocalVeto, settings.SwitchErrorRemoteVeto, &localVetoError{code: codeLocalVeto, reason: nodePausedVeto})
			}
		}
	}
	recoverCommitted := reconcile && cp.TargetCommitted && switchIntent(state, cp.TargetEngine) == settings.SwitchIntentOn
	if reconcile && cp.Phase != settings.SwitchCheckpointStopPending && cp.Phase != settings.SwitchCheckpointStartPending {
		for _, pair := range []struct {
			name        string
			observation *settings.SwitchEngineState
		}{{cp.SourceEngine, &result.Observed.Source}, {cp.TargetEngine, &result.Observed.Target}} {
			if st, err := e.state(pair.name); err == nil {
				if pair.name == cp.TargetEngine && recoverCommitted {
					continue
				}
				st.opMu.Lock()
				installed, _ := e.Detect(pair.name)
				st.mu.Lock()
				port := st.port
				st.mu.Unlock()
				e.reconcilePresence(ctx, pair.name, st, installed, port, false)
				*pair.observation = e.switchObserved(st, switchIntent(state, pair.name), e.switchTerminated(st))
				st.opMu.Unlock()
			}
		}
		if !recoverCommitted {
			result.Error = cp.Error
			return result
		}
	}
	if !reconcile && cp.Error != nil {
		result.Error = cp.Error
		return result
	}
	if p.Step == settings.SwitchStepSourceStop && (cp.Phase == settings.SwitchCheckpointSourceOff || cp.TargetCommitted || cp.Phase == settings.SwitchCheckpointStartPending) {
		return result
	}
	if p.Step == settings.SwitchStepTargetStart && cp.TargetCommitted {
		return result
	}
	if !reconcile && (ctx.Err() != nil || !cp.Deadline.After(time.Now())) {
		return fail(settings.SwitchErrorInterrupted, errors.New("switch step deadline expired or caller cancelled"))
	}
	if !reconcile && cp.Automatic {
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(ctx, cp.Deadline)
		defer cancel()
	}
	stopSource := p.Step == settings.SwitchStepSourceStop || (reconcile && cp.Phase == settings.SwitchCheckpointStopPending)
	startTarget := recoverCommitted || p.Step == settings.SwitchStepTargetStart || (reconcile && cp.Phase == settings.SwitchCheckpointStartPending)
	if !stopSource && !startTarget {
		return fail(settings.SwitchErrorInvalidRequest, &codedError{code: -32602, message: "unknown switch step"})
	}
	if startTarget && !reconcile && cp.Phase != settings.SwitchCheckpointSourceOff {
		return fail(settings.SwitchErrorSourceStop, errors.New("source termination is not verified"))
	}
	engine, kind := cp.TargetEngine, settings.SwitchErrorTargetStart
	if stopSource {
		engine, kind = cp.SourceEngine, settings.SwitchErrorSourceStop
	}
	st, err := e.state(engine)
	if err != nil {
		return fail(kind, err)
	}
	startCtx := ctx
	if startTarget && (!reconcile || recoverCommitted) {
		if !reconcile {
			var cancel context.CancelFunc
			startCtx, cancel = context.WithDeadline(ctx, cp.Deadline)
			defer cancel()
		}
		if cp.YieldsToLocalOff() {
			// Register and read the reservation under the same gate as source
			// Off, which also cancels work registered for the target engine.
			e.switchAdmission.Lock()
			remoteCtx, done, beginErr := e.beginRemote(startCtx, engine)
			err = beginErr
			if err == nil {
				startCtx = remoteCtx
				defer done()
				err = e.switchStartAllowed(startCtx, cp, recoverCommitted)
			}
			e.switchAdmission.Unlock()
		}
		if err == nil {
			err = vetoCause(startCtx, prepareStrataState(startCtx, st))
		}
		if err != nil && !recoverCommitted {
			return fail(kind, err)
		}
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	if stopSource {
		var ownership *stopOwnership
		if !reconcile {
			if err := e.switchTargetVeto(cp.Origin, cp.TargetEngine); err != nil {
				return fail(settings.SwitchErrorRemoteVeto, err)
			}
			// Refresh adoption under the lifecycle lock before saving Off or
			// invoking the stop owner. Listener refusal alone proves no ownership.
			if cp.Automatic {
				installed, detectErr := e.Detect(engine)
				if detectErr != nil {
					return fail(kind, detectErr)
				}
				st.mu.Lock()
				port := st.port
				st.mu.Unlock()
				e.reconcilePresence(ctx, engine, st, installed, port, false)
				if err := e.automaticSwitchSource(cp.SwitchPrepareRequest); err != nil {
					return fail(kind, err)
				}
				st.mu.Lock()
				ownership = &stopOwnership{process: st.proc, runID: st.runID}
				st.mu.Unlock()
				ports := state.PendingSwitchPorts
				if ports == nil || ports.SourceRunID == "" || ports.SourceRunID != ownership.runID || (ownership.process != nil && (ownership.process.cmd == nil || ownership.process.cmd.Process == nil || ports.SourcePID != ownership.process.cmd.Process.Pid)) {
					return fail(kind, errors.New("automatic source incarnation changed before stop"))
				}
			}
		}
		cp.Phase = settings.SwitchCheckpointStopPending
		cp.Source.Intent = settings.SwitchIntentOff
		if err := e.saveSwitch(cp, engine, false); err != nil {
			return fail(settings.SwitchErrorStorage, err)
		}
		e.emitIntent()
		if reconcile {
			paused, _ := e.nodeProjection.managementBlocked()
			explicitPause := paused || (localVeto != nil && localVeto.Pause)
			if cp.Automatic && !cp.SourceLocalOff && !explicitPause {
				err = e.cleanupAutomaticSwitchSource(ctx, st, cp, state.PendingSwitchPorts)
			} else {
				err = e.cleanupSwitchRecovery(ctx, st, engine)
			}
		} else if cp.Automatic {
			err = e.doStopOwnedContext(ctx, st, engine, ownership)
			var changed *stopOwnershipChangedError
			if errors.As(err, &changed) {
				// The stop owner positively refused before signalling or invoking
				// a CLI. Recovery must not turn that refusal into orphan reclaim.
				cp.Phase = settings.SwitchCheckpointReserved
			}
		} else {
			installed, _ := e.Detect(engine)
			st.mu.Lock()
			port := st.port
			st.mu.Unlock()
			e.reconcilePresence(ctx, engine, st, installed, port, true)
			err = e.doStop(st, engine)
		}
		terminated := false
		if cp.Automatic {
			terminated = e.switchTerminatedContext(ctx, st)
			if ports := state.PendingSwitchPorts; ports != nil && ports.SourcePID > 0 {
				terminated = terminated && orphanPIDExited(ports.SourcePID)
			}
			cp.Source = switchRuntimeObserved(st, settings.SwitchIntentOff, terminated)
			if err != nil && !terminated && cp.Source.Running != nil && !*cp.Source.Running {
				cp.Source.Running, cp.Source.Healthy = nil, nil
			}
		} else {
			terminated = e.switchTerminated(st)
			cp.Source = e.switchObserved(st, settings.SwitchIntentOff, terminated)
		}
		if !terminated && err == nil {
			err = errors.New("source termination could not be verified")
		}
		if err == nil {
			cp.Phase = settings.SwitchCheckpointSourceOff
			cp.Error = nil
		}
	} else {
		st.mu.Lock()
		port := st.port
		st.mu.Unlock()
		if port == 0 && !reconcile {
			port, err = freePort()
			if err != nil {
				return fail(settings.SwitchErrorTargetStart, err)
			}
			st.mu.Lock()
			st.port = port
			st.mu.Unlock()
		}
		if !recoverCommitted {
			cp.Phase = settings.SwitchCheckpointStartPending
		}
		intentEngine := ""
		if vetoedCommittedTarget {
			intentEngine = engine
		}
		if !recoverCommitted {
			if err := e.saveSwitch(cp, intentEngine, false); err != nil {
				return fail(settings.SwitchErrorStorage, err)
			}
		}
		launchSucceeded := false
		if reconcile && !recoverCommitted {
			if port == 0 {
				return fail(settings.SwitchErrorInterrupted, errors.New("interrupted target has no durable listener endpoint; reservation retained"))
			}
			err = e.cleanupSwitchRecovery(ctx, st, engine)
			if err == nil {
				err = errors.New("target startup interrupted; source is not restored")
			}
		} else {
			e.switchAdmission.Lock()
			if err == nil {
				err = e.switchStartAllowed(startCtx, cp, recoverCommitted)
			}
			e.switchAdmission.Unlock()
			if err == nil {
				installed, detectErr := e.Detect(engine)
				if detectErr != nil {
					err = detectErr
				} else if !installed {
					err = errors.New("target engine is no longer prepared")
				} else {
					err = e.doStart(startCtx, st, engine, startOpts{RequireOwned: true, ReclaimOwned: true, BoundedCleanup: cp.Automatic, Bind: "127.0.0.1"})
					launchSucceeded = err == nil
				}
				err = vetoCause(startCtx, err)
			}
		}
		// Serialize final veto/readiness commit against a local Off admission.
		e.switchAdmission.Lock()
		if err == nil {
			err = e.switchStartAllowed(startCtx, cp, recoverCommitted)
		}
		if err == nil && e.shuttingDown.Load() {
			err = context.Canceled
		}
		if err == nil {
			cp.TargetCommitted = true
			cp.Phase = settings.SwitchCheckpointTargetCommitted
			cp.Target = e.switchObserved(st, settings.SwitchIntentOn, false)
			err = e.saveSwitch(cp, engine, true)
			e.switchAdmission.Unlock()
			if err != nil {
				return fail(settings.SwitchErrorStorage, err)
			}
			e.emitIntent()
			result.Observed = settings.SwitchObserved{Source: cp.Source, Target: cp.Target}
			return result
		}
		e.switchAdmission.Unlock()
		// A failed or vetoed launch never restores the source.
		st.mu.Lock()
		reclaimFailed := st.reclaimFailed
		owned := st.proc != nil
		st.mu.Unlock()
		var cleanupErr error
		if !reclaimFailed && (!recoverCommitted || launchSucceeded || owned) && (!reconcile || recoverCommitted) {
			if !reconcile && !owned && !launchSucceeded {
				cleanupErr = e.cleanupSwitchRecovery(startCtx, st, engine)
			} else if cp.Automatic {
				cleanupErr = e.doStopContext(startCtx, st, engine)
			} else {
				cleanupErr = e.doStop(st, engine)
			}
		}
		err = errors.Join(err, cleanupErr)
		if recoverCommitted && switchCode(err) != codeLocalVeto {
			return fail(kind, err)
		}
		if recoverCommitted {
			cp.TargetCommitted = false
			cp.Phase = settings.SwitchCheckpointStartPending
		}
		if cp.Automatic {
			terminated := e.switchTerminatedContext(startCtx, st)
			cp.Target = switchRuntimeObserved(st, settings.SwitchIntentOff, terminated)
			if !terminated && cp.Target.Running != nil && !*cp.Target.Running {
				cp.Target.Running, cp.Target.Healthy = nil, nil
			}
		} else {
			cp.Target = e.switchObserved(st, settings.SwitchIntentOff, e.switchTerminated(st))
		}
	}
	if err != nil {
		cp.Error = switchFailure(switchCode(err), kind, err)
		if switchCode(err) == codeLocalVeto {
			cp.Error.Kind = settings.SwitchErrorRemoteVeto
		}
	}
	intentEngine := ""
	if startTarget {
		intentEngine = engine
		if cp.Target.TerminationVerified {
			cp.Phase = settings.SwitchCheckpointSourceOff
		}
	}
	if recoverCommitted {
		e.switchAdmission.Lock()
		// The terminal target observation was made after recovery/cleanup. Acknowledge
		// a concurrent target Off under its admission gate so saveSwitch keeps
		// this fresh observation instead of merging the admission's older On.
		// Source Off still needs its own termination reconciliation.
		e.desired.mu.Lock()
		current, loadErr := e.desired.load()
		e.desired.mu.Unlock()
		if loadErr != nil {
			e.switchAdmission.Unlock()
			return fail(settings.SwitchErrorStorage, loadErr)
		}
		pending := current.PendingSwitch
		if pending == nil || pending.SwitchPrepareRequest != cp.SwitchPrepareRequest {
			e.switchAdmission.Unlock()
			return fail(settings.SwitchErrorInterrupted, errors.New("switch checkpoint changed before recovery terminal"))
		}
		if pending.TargetLocalOff && cp.Target.Intent == settings.SwitchIntentOff {
			cp.TargetLocalOff = true
		}
	}
	saveErr := e.saveSwitch(cp, intentEngine, false)
	if recoverCommitted {
		e.switchAdmission.Unlock()
	}
	if saveErr != nil {
		return fail(settings.SwitchErrorStorage, saveErr)
	}
	if startTarget {
		e.emitIntent()
	}
	result.Error = cp.Error
	result.Observed = settings.SwitchObserved{Source: cp.Source, Target: cp.Target}
	return result
}

// Caller holds switchAdmission, shared with local Off admission. Check both
// before launch and before accepting readiness, using the current durable intent.
func (e *Executor) switchStartAllowed(ctx context.Context, cp *settings.SwitchCheckpoint, recoverCommitted bool) error {
	if err := vetoCause(ctx, ctx.Err()); err != nil {
		return err
	}
	if err := e.switchTargetVeto(cp.Origin, cp.TargetEngine); err != nil {
		return err
	}
	e.desired.mu.Lock()
	current, err := e.desired.load()
	e.desired.mu.Unlock()
	if err != nil {
		return err
	}
	pending := current.PendingSwitch
	if pending == nil || pending.SwitchPrepareRequest != cp.SwitchPrepareRequest || pending.Phase != cp.Phase || pending.TargetCommitted != cp.TargetCommitted {
		return errors.New("switch checkpoint changed before target readiness")
	}
	if pending.Error != nil && (pending.Error.Code == codeLocalVeto || pending.Error.Code == codeEngineLocked) {
		return &localVetoError{code: pending.Error.Code, reason: pending.Error.Message}
	}
	if recoverCommitted && switchIntent(current, cp.TargetEngine) != settings.SwitchIntentOn {
		return errors.New("committed target no longer has durable On intent")
	}
	return nil
}

func (e *Executor) switchTargetVeto(origin, engine string) error {
	if origin == settings.SwitchOriginRemote {
		return e.remoteVeto(engine)
	}
	return e.activationVeto(engine)
}

// Automatic replacement requires a sole live source with affirmative PAIR
// ownership. This deliberately does not invoke or relax orphan reclamation.
func (e *Executor) automaticSwitchSource(p settings.SwitchPrepareRequest) error {
	if !p.Automatic {
		return nil
	}
	for _, name := range e.reg.Names() {
		st, err := e.state(name)
		if err != nil {
			// Unsupported host entries cannot be running in this executor.
			if name == p.SourceEngine {
				return err
			}
			continue
		}
		st.mu.Lock()
		running, adopted, uncertain := st.running, st.adopted, st.reclaimFailed || st.reclaimPID != 0 || (st.running && st.stopping)
		controlled := (st.proc != nil && st.proc.cmd != nil && st.proc.cmd.Process != nil) || (st.plat.Runtime.modeOrDefault() == "command" && st.runID != "")
		st.mu.Unlock()
		if name == p.SourceEngine {
			if !running || adopted || uncertain || !controlled {
				return errors.New("automatic source is not a running PAIR-controlled engine with certain ownership")
			}
		} else if running || uncertain {
			return errors.New("automatic replacement requires a sole conflicting source")
		}
	}
	return nil
}

// A pending automatic source stop may reclaim only the original controlled
// child. A new listener, or a command server with lost control identity, stays
// fenced. Explicit local Off keeps its existing independent cleanup authority.
func (e *Executor) cleanupAutomaticSwitchSource(ctx context.Context, st *engineState, cp *settings.SwitchCheckpoint, ports *switchCheckpointPorts) error {
	if ports == nil || ports.SourceRunID == "" {
		return errors.New("automatic source recovery lacks original execution evidence")
	}
	st.mu.Lock()
	proc, runID, adopted, port := st.proc, st.runID, st.adopted, st.port
	rt, previousPID := st.plat.Runtime, st.reclaimPID
	st.mu.Unlock()
	if adopted {
		return errors.New("automatic source recovery refuses an adopted server")
	}
	if proc != nil || rt.modeOrDefault() == "command" {
		if runID != ports.SourceRunID || (proc != nil && (proc.cmd == nil || proc.cmd.Process == nil || proc.cmd.Process.Pid != ports.SourcePID)) {
			return errors.New("automatic source recovery control identity changed")
		}
		return e.doStopOwnedContext(ctx, st, cp.SourceEngine, &stopOwnership{process: proc, runID: runID})
	}
	if ports.SourcePID <= 0 || (previousPID != 0 && previousPID != ports.SourcePID && !orphanPIDExited(previousPID)) {
		return errors.New("automatic source recovery process ownership is uncertain")
	}
	cleanupCtx, finish := e.startContext(ctx, st)
	defer finish()
	if _, err := e.Detect(cp.SourceEngine); err != nil {
		return err
	}
	reclaimed, err := e.reclaimOrphanForPID(cleanupCtx, st, cp.SourceEngine, port, ports.SourcePID)
	if err != nil || reclaimed {
		return err
	}
	// Refused listeners alone cannot prove that the original child exited.
	if err := e.confirmOrphanStoppedContext(cleanupCtx, cp.SourceEngine, ports.SourcePID, port, rt); err != nil {
		return err
	}
	e.markStopped(st, cp.SourceEngine)
	if rt.Port == 0 {
		return e.clearAllocatedPort(st, cp.SourceEngine)
	}
	return nil
}

// Caller holds opMu. All interrupted switch cleanup uses the same precise
// reclaim owner before readiness can adopt (or overlook) a managed orphan.
func (e *Executor) cleanupSwitchRecovery(ctx context.Context, st *engineState, engine string) error {
	cleanupCtx, finish := e.startContext(ctx, st)
	defer finish()
	if err := cleanupCtx.Err(); err != nil {
		return err
	}
	installed, _ := e.Detect(engine)
	st.mu.Lock()
	port := st.port
	st.mu.Unlock()
	reclaimed, err := e.reclaimOrphan(cleanupCtx, st, engine, port)
	if err != nil || reclaimed {
		return err
	}
	if err := cleanupCtx.Err(); err != nil {
		return err
	}
	e.reconcilePresence(cleanupCtx, engine, st, installed, port, true)
	return e.doStopContext(cleanupCtx, st, engine)
}

func (e *Executor) switchTerminated(st *engineState) bool {
	return e.switchTerminatedContext(context.Background(), st)
}

func (e *Executor) switchTerminatedContext(ctx context.Context, st *engineState) bool {
	st.mu.Lock()
	proc, running, port := st.proc, st.running, st.port
	fenced, reclaimPID := st.reclaimFailed, st.reclaimPID
	st.mu.Unlock()
	if proc != nil || running || (fenced && !orphanPIDExited(reclaimPID)) {
		return false
	}
	if port == 0 {
		return true
	} // no allocated endpoint and no owned process
	return e.waitUnavailableContext(ctx, st.plat.Runtime.Ready, port, time.Second)
}
