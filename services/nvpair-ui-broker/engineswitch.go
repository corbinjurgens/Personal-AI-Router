// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/noderec"
)

// Store, error and changed are guarded by engineConfigMu. The separate waiter
// lock lets a local pause veto remote work without waiting for either operation
// lock. Accepted work belongs to this node, never to a relay waiter.
type engineSwitchCoordinator struct {
	store              *engineSwitchStore
	err                error
	recoveryErr        error // retryable cleanup failure; engineConfigMu
	changed            chan struct{}
	resolved           map[string]bool
	recoveryCheckpoint *settings.SwitchCheckpoint // engineConfigMu; includes orphan reservations
	waitMu             sync.Mutex
	waiters            map[string]context.CancelFunc
	remoteCancel       context.CancelFunc
	remoteSource       string
	remoteTarget       string
	// Active execution is fenced under workersMu, together with manager replacement.
	activeManager *rpcWorker
	activeCancel  context.CancelFunc
	interrupted   bool
	activeID      string
	localVeto     *settings.SwitchLocalVeto // workersMu; immutable operation-specific cleanup input
	vetoID        string                    // retains the veto through failed cleanup/recovery
}

type switchRPCError struct{ detail settings.SwitchError }

func (e *switchRPCError) Error() string { return e.detail.Message }

func switchError(kind, message string) error {
	code := settings.SwitchOperationErrorCode
	switch kind {
	case settings.SwitchErrorInvalidRequest, settings.SwitchErrorReusedID:
		code = settings.SwitchInvalidParamsCode
	case settings.SwitchErrorConfigurationConflict:
		code = settings.SwitchConfigurationConflictCode
	case settings.SwitchErrorRemoteVeto:
		code = settings.SwitchRemoteVetoCode
	}
	if len(message) > settings.MaxSwitchErrorMessageBytes {
		message = message[:settings.MaxSwitchErrorMessageBytes]
	}
	return &switchRPCError{settings.SwitchError{Code: code, Kind: kind, Message: message}}
}

func switchErrorDetail(err error, kind string) *settings.SwitchError {
	var coded *switchRPCError
	if errors.As(err, &coded) {
		value := coded.detail
		return &value
	}
	var storeErr *engineSwitchStoreError
	if errors.As(err, &storeErr) {
		kind = storeErr.Kind
	}
	value := switchError(kind, err.Error()).(*switchRPCError).detail
	return &value
}

// switchTargetRoutable is the single target routability predicate. The Mac's
// dynamic-facade routability predicate replaces this body at the join.
func (b *Broker) switchTargetRoutable(engine string) error {
	if _, ok := engineProxyProfileFor(engine); !ok {
		return switchError(settings.SwitchErrorUnroutable, "The target has no supported inference facade.")
	}
	p := b.settingsProxy(engine)
	if p == nil {
		return switchError(settings.SwitchErrorUnroutable, "The target inference facade is unavailable.")
	}
	if ready, port := p.Status(engine); !ready || port < 1 {
		return switchError(settings.SwitchErrorUnroutable, "The target inference facade is not ready.")
	}
	return nil
}

func (b *Broker) switchCoordinator() *engineSwitchCoordinator {
	b.switchInit.Do(func() {
		b.engineSwitch = &engineSwitchCoordinator{changed: make(chan struct{}), waiters: make(map[string]context.CancelFunc), resolved: make(map[string]bool)}
	})
	return b.engineSwitch
}

func (b *Broker) loadEngineSwitchLocked() error {
	c := b.switchCoordinator()
	if c.store == nil {
		path, err := b.engineSwitchPath()
		if err != nil {
			c.err = err
			return err
		}
		c.store = newEngineSwitchStore(path)
	}
	if err := c.store.Load(); err != nil {
		c.err = err
		return err
	}
	return nil
}

func (b *Broker) switchChangedLocked(r settings.SwitchReceipt) {
	c := b.switchCoordinator()
	close(c.changed)
	c.changed = make(chan struct{})
	b.notifyClients(settings.SwitchChangedNotification, settings.SwitchChanged{NodeID: r.NodeID, RequestID: r.RequestID, Sequence: r.Sequence})
}

// pendingSwitchLocked also fences unresolved recovery. Call before any public
// settings/lifecycle mutation and while the advertiser owns engineConfigMu.
func (b *Broker) pendingSwitchLocked(engine string) error {
	if err := b.loadEngineSwitchLocked(); err != nil {
		return err
	}
	c := b.switchCoordinator()
	if cp := c.recoveryCheckpoint; cp != nil && cp.Phase != settings.SwitchCheckpointReleased && (engine == "" || engine == cp.SourceEngine || engine == cp.TargetEngine) {
		return switchError(settings.SwitchErrorBusy, "Switch recovery reserves this engine.")
	}
	if c.err != nil {
		return c.err
	}
	if c.recoveryErr != nil {
		return c.recoveryErr
	}
	for _, r := range c.store.journal.Receipts {
		if (!r.Terminal || (r.RecoveryRequired && !c.resolved[r.RequestID])) && (engine == "" || engine == r.Request.SourceEngine || engine == r.Request.TargetEngine) {
			return switchError(settings.SwitchErrorBusy, "An engine switch reserves this engine.")
		}
	}
	return nil
}

// Local Off alone may bypass the operation lock for a remote reservation,
// including an unresolved recovery row. The manager remains the veto owner.
func (b *Broker) switchLocalOffReservationLocked(engine string) (remote, reserved bool, operationID string) {
	if b.loadEngineSwitchLocked() != nil {
		return false, false, ""
	}
	c := b.switchCoordinator()
	if cp := c.recoveryCheckpoint; cp != nil && cp.Phase != settings.SwitchCheckpointReleased && (engine == cp.SourceEngine || engine == cp.TargetEngine) {
		remote = cp.Origin == settings.SwitchOriginRemote
		if remote {
			b.workersMu.Lock()
			c.vetoID, c.localVeto = cp.OperationID, &settings.SwitchLocalVeto{Engine: engine}
			b.workersMu.Unlock()
		}
		return remote, true, cp.OperationID
	}
	for _, r := range c.store.journal.Receipts {
		if (!r.Terminal || (r.RecoveryRequired && !c.resolved[r.RequestID])) && (engine == r.Request.SourceEngine || engine == r.Request.TargetEngine) {
			remote = r.Caller != settings.SwitchOriginLocal
			if remote {
				// Admission and completion share workersMu. Latch before the
				// lifecycle RPC can wait for physical stop or its response.
				b.workersMu.Lock()
				c.vetoID, c.localVeto = r.RequestID, &settings.SwitchLocalVeto{Engine: engine}
				b.workersMu.Unlock()
			}
			return remote, true, r.RequestID
		}
	}
	return false, false, ""
}

// Recheck reservations while waiting for ordinary settings work. Lookup and
// TryLock share engineConfigMu with acceptance, so a negative lookup can never
// lead to a blocking Lock behind a newly accepted switch.
func (b *Broker) admitLocalEngineOff(engine string) (remote, reserved bool, operationID string) {
	for {
		b.engineConfigMu.Lock()
		remote, reserved, operationID = b.switchLocalOffReservationLocked(engine)
		if reserved {
			b.engineConfigMu.Unlock()
			return remote, reserved, operationID
		}
		if b.settingsApplyMu.TryLock() {
			b.engineConfigMu.Unlock()
			return false, false, "" // caller owns settingsApplyMu
		}
		changed := b.switchCoordinator().changed
		b.engineConfigMu.Unlock()
		timer := time.NewTimer(10 * time.Millisecond)
		select {
		case <-changed:
		case <-timer.C:
		}
		timer.Stop()
	}
}

// Preserve manager error envelopes and numeric RPC errors; settingsWorkerCall
// deliberately flattens errors and must not be used by this transaction.
func (b *Broker) switchManagerCall(ctx context.Context, method string, params any, out any) error {
	w := b.getEngineMgr()
	cleanup := method == settings.SwitchReconcileMethod
	if p, ok := params.(settings.SwitchPrepareRequest); ok && method == settings.SwitchPrepareMethod && p.OperationID == "" {
		cleanup = true
	}
	if p, ok := params.(settings.SwitchStepRequest); ok && method == settings.SwitchStepMethod && p.Step == settings.SwitchStepRelease {
		cleanup = true
	}
	if cleanup {
		b.workersMu.Lock()
		w = b.engineMgr
		b.workersMu.Unlock()
	}
	return switchManagerCallOn(ctx, w, method, params, out)
}

func switchManagerCallOn(ctx context.Context, w *rpcWorker, method string, params any, out any) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if w == nil {
		return switchError(settings.SwitchErrorBusy, "Engine manager unavailable.")
	}
	raw, rpcErr, err := w.CallNoTimeout(ctx, method, settingsJSON(params))
	if err != nil {
		return err
	}
	if rpcErr != nil {
		kind := settings.SwitchErrorValidation
		switch rpcErr.Code {
		case settings.SwitchInvalidParamsCode:
			kind = settings.SwitchErrorInvalidRequest
		case settings.SwitchConfigurationConflictCode:
			kind = settings.SwitchErrorConfigurationConflict
		case settings.SwitchRemoteVetoCode:
			kind = settings.SwitchErrorRemoteVeto
		}
		return &switchRPCError{settings.SwitchError{Code: rpcErr.Code, Kind: kind, Message: rpcErr.Message}}
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return err
	}
	return nil
}

type switchManagerSnapshot struct {
	settings.SwitchPrepareResult
	Configuration      settings.SwitchConfigurationSnapshot `json:"configuration"`
	ConfigurationFence string                               `json:"configurationFence"`
}

func (b *Broker) switchSnapshotLocked(ctx context.Context) (switchManagerSnapshot, string, error) {
	var result switchManagerSnapshot
	if err := b.loadEngineSwitchLocked(); err != nil {
		return result, "", err
	}
	if err := b.loadEngineSettingsLocked(); err != nil {
		return result, "", err
	}
	if err := b.switchManagerCall(ctx, settings.SwitchPrepareMethod, settings.SwitchPrepareRequest{}, &result); err != nil {
		return result, "", err
	}
	if result.Error != nil {
		return result, "", &switchRPCError{*result.Error}
	}
	if result.ConfigurationFence == "" || result.Configuration.ManagerIncarnation == "" {
		return result, "", fmt.Errorf("manager configuration snapshot is incomplete")
	}
	config := result.Configuration
	config.Engines = append([]settings.SwitchConfigurationEngine(nil), config.Engines...)
	config.NodeID, config.JournalGeneration = b.nodeID, b.engineSwitch.store.Generation()
	for i := range config.Engines {
		if record := b.engineSettings[config.Engines[i].Engine]; record != nil {
			config.Engines[i].SettingsRevision = record.Snapshot.Revision
		}
	}
	return result, settings.SwitchConfigurationToken(config), nil
}

func (b *Broker) acceptEngineSwitch(ctx context.Context, p settings.SwitchRequest, caller string) (settings.SwitchResult, error) {
	ctx, cancelAcceptance := context.WithTimeout(ctx, settings.SwitchStatusWaitBudget)
	defer cancelAcceptance()
	if err := p.Validate(); err != nil {
		return settings.SwitchResult{}, switchError(settings.SwitchErrorInvalidRequest, err.Error())
	}
	// The live operation retains settingsApplyMu, so refuse contention promptly;
	// exact retries still read the journal before trying that lock.
	b.engineConfigMu.Lock()
	if err := b.authorizeSettingsCallerLocked(caller); err != nil {
		b.engineConfigMu.Unlock()
		return settings.SwitchResult{}, err
	}
	if err := b.loadEngineSwitchLocked(); err != nil {
		b.engineConfigMu.Unlock()
		return settings.SwitchResult{}, err
	}
	p.NodeID = b.nodeID
	identity := caller
	if identity == "" {
		identity = settings.SwitchOriginLocal
	}
	hash := settings.SwitchPayloadHash(identity, b.nodeID, p)
	if old, ok := b.engineSwitch.store.Lookup(p.RequestID); ok {
		b.engineConfigMu.Unlock()
		if old.Caller != identity || old.PayloadHash != hash {
			return settings.SwitchResult{}, switchError(settings.SwitchErrorReusedID, "Request ID was already used by another caller or payload.")
		}
		return settings.SwitchResult{Receipt: old}, nil
	}
	b.engineConfigMu.Unlock()
	if !b.settingsApplyMu.TryLock() {
		return settings.SwitchResult{}, switchError(settings.SwitchErrorBusy, "An engine configuration operation is active.")
	}
	releaseOperation := true
	defer func() {
		if releaseOperation {
			b.settingsApplyMu.Unlock()
		}
	}()
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if err := ctx.Err(); err != nil {
		return settings.SwitchResult{}, err
	}
	if err := b.authorizeSettingsCallerLocked(caller); err != nil {
		return settings.SwitchResult{}, err
	}
	if old, ok := b.engineSwitch.store.Lookup(p.RequestID); ok {
		if old.Caller != identity || old.PayloadHash != hash {
			return settings.SwitchResult{}, switchError(settings.SwitchErrorReusedID, "Request ID was already used by another caller or payload.")
		}
		return settings.SwitchResult{Receipt: old}, nil
	}
	if err := b.pendingSwitchLocked(""); err != nil {
		return settings.SwitchResult{}, err
	}
	snapshot, token, err := b.switchSnapshotLocked(ctx)
	if err != nil {
		return settings.SwitchResult{}, err
	}
	now := time.Now().UTC()
	r := settings.SwitchReceipt{Version: settings.SwitchVersion, NodeID: b.nodeID, RequestID: p.RequestID, Caller: identity, PayloadHash: hash, Request: p, ExpectedRevision: p.ExpectedRevision, AcceptedRevision: token, Sequence: 1, Phase: settings.SwitchAccepted, AcceptedAt: now, UpdatedAt: now, Deadline: now.Add(settings.SwitchOperationBudget), Source: settings.SwitchEngineState{Intent: settings.SwitchIntentUnknown}, Target: settings.SwitchEngineState{Intent: settings.SwitchIntentUnknown}}
	refuse := func(err error) (settings.SwitchResult, error) {
		r.Phase, r.Terminal, r.FinishedAt, r.Error = settings.SwitchRejected, true, &now, switchErrorDetail(err, settings.SwitchErrorValidation)
		stored, saveErr := b.engineSwitch.store.Reject(r)
		if stored.Sequence > 0 {
			b.switchChangedLocked(stored)
		}
		if saveErr != nil {
			b.latchSwitchStorageLocked(saveErr)
			return settings.SwitchResult{}, saveErr
		}
		return settings.SwitchResult{Receipt: stored}, err
	}
	if token != p.ExpectedRevision {
		return refuse(switchError(settings.SwitchErrorConfigurationConflict, "Engine configuration changed; reload before switching."))
	}
	if snapshot.Checkpoint != nil && snapshot.Checkpoint.Phase != settings.SwitchCheckpointReleased {
		return refuse(switchError(settings.SwitchErrorBusy, "The manager has an unresolved switch reservation."))
	}
	if err := b.switchTargetRoutable(p.TargetEngine); err != nil {
		return refuse(err)
	}
	if caller != "" {
		if _, local := b.settingsPause(); local {
			return refuse(switchError(settings.SwitchErrorRemoteVeto, "A local pause vetoes remote switching."))
		}
	}
	for _, record := range b.engineSettings {
		if record.Snapshot.Phase == "applying" {
			return refuse(switchError(settings.SwitchErrorBusy, "Pending settings require recovery."))
		}
	}
	stored, err := b.engineSwitch.store.Accept(r)
	if stored.Sequence > 0 {
		b.switchChangedLocked(stored)
	}
	if err != nil {
		b.latchSwitchStorageLocked(err)
		return settings.SwitchResult{}, err
	}
	origin := settings.SwitchOriginLocal
	if caller != "" {
		origin = settings.SwitchOriginRemote
	}
	envelope := settings.SwitchPrepareRequest{OperationID: p.RequestID, Caller: identity, Origin: origin, SourceEngine: p.SourceEngine, TargetEngine: p.TargetEngine, PayloadHash: hash, ConfigurationFence: snapshot.ConfigurationFence, ManagerIncarnation: snapshot.Configuration.ManagerIncarnation, Deadline: r.Deadline}
	operationCtx, cancel := context.WithDeadline(context.Background(), r.Deadline)
	c := b.engineSwitch
	b.workersMu.Lock()
	w := b.engineMgr
	c.activeManager, c.activeCancel, c.interrupted = w, cancel, b.engineMgrRecovering || w == nil
	c.activeID, c.vetoID = p.RequestID, p.RequestID
	c.localVeto = nil
	if c.interrupted {
		cancel()
	}
	b.workersMu.Unlock()
	c.waitMu.Lock()
	if caller != "" {
		c.remoteCancel = cancel
		c.remoteSource, c.remoteTarget = p.SourceEngine, p.TargetEngine
	}
	c.waitMu.Unlock()
	// Recheck after installing cancellation: a pause accepted between the first
	// veto check and registration cannot be missed.
	if caller != "" {
		if _, local := b.settingsPause(); local {
			cancel()
		}
	}
	releaseOperation = false
	go func() {
		defer b.settingsApplyMu.Unlock()
		defer cancel()
		defer func() {
			b.workersMu.Lock()
			c.activeManager, c.activeCancel = nil, nil
			c.activeID = ""
			b.workersMu.Unlock()
			c.waitMu.Lock()
			c.remoteCancel = nil
			c.remoteSource, c.remoteTarget = "", ""
			c.waitMu.Unlock()
		}()
		b.runEngineSwitch(operationCtx, envelope)
	}()
	return settings.SwitchResult{Receipt: stored}, nil
}

// advance persists the next phase before its effect. Any write uncertainty
// latches recovery and halts forward effects, even when ErrReplaced landed.
func (b *Broker) advanceEngineSwitch(id, phase string, observed *settings.SwitchObserved, failure error, recovery bool, reconciledVeto *settings.SwitchLocalVeto) error {
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	c := b.engineSwitch
	if c.err != nil {
		return c.err
	}
	// Completion and manager replacement have one ordering point. Hold the
	// same gate while persisting completion so an already-observed restart
	// cannot race a completed receipt into the journal.
	interruptedTerminal := false
	if settings.IsTerminal(phase) {
		b.workersMu.Lock()
		defer b.workersMu.Unlock()
		if c.activeManager != nil {
			select {
			case <-c.activeManager.Done():
				c.interrupted = true
				if b.engineMgr == c.activeManager {
					b.engineMgrRecovering = true
				}
			default:
			}
		}
		if c.interrupted {
			phase, recovery = settings.SwitchFailedInterrupted, true
			failure = switchError(settings.SwitchErrorInterrupted, "Engine manager restarted during the switch.")
			interruptedTerminal = true
		} else if phase == settings.SwitchCompleted && c.activeID == id && c.localVeto != nil {
			return switchError(settings.SwitchErrorRemoteVeto, "Local Off or pause vetoed the switch.")
		} else if !recovery && c.activeID == id && c.localVeto != nil && (reconciledVeto == nil || *c.localVeto != *reconciledVeto) {
			// A veto admitted after cleanup's snapshot must reach the manager
			// before a failure terminal can relinquish recovery ownership.
			return switchError(settings.SwitchErrorRemoteVeto, "Local veto requires conservative cleanup.")
		}
	}
	r, ok := c.store.Lookup(id)
	if !ok {
		return fmt.Errorf("switch receipt missing")
	}
	now := time.Now().UTC()
	if now.Before(r.UpdatedAt) {
		now = r.UpdatedAt
	}
	r.Phase, r.Sequence, r.UpdatedAt = phase, r.Sequence+1, now
	if observed != nil {
		r.Source, r.Target = observed.Source, observed.Target
	}
	r.Terminal, r.RecoveryRequired = settings.IsTerminal(phase), recovery
	if failure != nil {
		r.Error = switchErrorDetail(failure, switchPhaseError(phase))
	}
	if r.Terminal {
		r.FinishedAt = &r.UpdatedAt
	}
	err := c.store.Advance(id, r)
	if landed, _ := c.store.Lookup(id); landed.Sequence == r.Sequence {
		b.switchChangedLocked(landed)
	}
	if err != nil {
		c.err = err
	}
	if err == nil && interruptedTerminal {
		return failure
	}
	return err
}

func (b *Broker) latchSwitchStorageLocked(err error) {
	var storeErr *engineSwitchStoreError
	if errors.As(err, &storeErr) && storeErr.Kind != settings.SwitchErrorStorage {
		return
	}
	b.engineSwitch.err = err
}

func switchPhaseError(phase string) string {
	switch phase {
	case settings.SwitchFailedDrain:
		return settings.SwitchErrorDrain
	case settings.SwitchFailedSourceStop:
		return settings.SwitchErrorSourceStop
	case settings.SwitchFailedTargetStart:
		return settings.SwitchErrorTargetStart
	case settings.SwitchFailedPublication:
		return settings.SwitchErrorPublication
	case settings.SwitchFailedInterrupted:
		return settings.SwitchErrorInterrupted
	default:
		return settings.SwitchErrorValidation
	}
}

func (b *Broker) switchStep(ctx context.Context, p settings.SwitchPrepareRequest, step string) (settings.SwitchStepResult, error) {
	var result settings.SwitchStepResult
	err := b.switchManagerCall(ctx, settings.SwitchStepMethod, settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: step}, &result)
	if err == nil && result.Error != nil {
		err = &switchRPCError{*result.Error}
	}
	return result, err
}

func (b *Broker) runEngineSwitch(ctx context.Context, p settings.SwitchPrepareRequest) {
	b.workersMu.Lock()
	w := b.switchCoordinator().activeManager
	b.workersMu.Unlock()
	if b.advanceEngineSwitch(p.OperationID, settings.SwitchValidating, nil, nil, false, nil) != nil {
		return
	}
	var prepared switchManagerSnapshot
	err := switchManagerCallOn(ctx, w, settings.SwitchPrepareMethod, p, &prepared)
	if err == nil && prepared.Error != nil {
		err = &switchRPCError{*prepared.Error}
	}
	if err == nil && (prepared.Checkpoint == nil || prepared.Checkpoint.SwitchPrepareRequest != p) {
		err = fmt.Errorf("manager did not acknowledge the switch reservation")
	}
	if err != nil {
		b.finishEngineSwitch(p, settings.SwitchFailedValidation, nil, err)
		return
	}
	if err := b.advanceEngineSwitch(p.OperationID, settings.SwitchDraining, &prepared.Observed, nil, false, nil); err != nil {
		return
	}
	drainedProxy := b.policyProxy()
	if err := b.drainEngineForSwitch(ctx, p); err != nil {
		b.finishEngineSwitch(p, settings.SwitchFailedDrain, nil, err)
		return
	}
	if err := ctx.Err(); err != nil {
		b.finishEngineSwitch(p, settings.SwitchFailedDrain, nil, err)
		return
	}
	if b.policyProxy() != drainedProxy {
		b.finishEngineSwitch(p, settings.SwitchFailedDrain, nil, fmt.Errorf("proxy restarted after drain evidence"))
		return
	}
	if b.advanceEngineSwitch(p.OperationID, settings.SwitchSourceStopping, nil, nil, false, nil) != nil {
		return
	}
	stopped, err := switchForwardStep(ctx, w, p, settings.SwitchStepSourceStop)
	if err == nil && !stopped.Observed.Source.TerminationVerified {
		err = fmt.Errorf("source termination is unverified")
	}
	if err != nil {
		b.finishEngineSwitch(p, settings.SwitchFailedSourceStop, &stopped.Observed, err)
		return
	}
	if b.advanceEngineSwitch(p.OperationID, settings.SwitchSourceOff, &stopped.Observed, nil, false, nil) != nil {
		return
	}
	if b.advanceEngineSwitch(p.OperationID, settings.SwitchTargetStarting, nil, nil, false, nil) != nil {
		return
	}
	started, err := switchForwardStep(ctx, w, p, settings.SwitchStepTargetStart)
	if err == nil && (started.Checkpoint == nil || !started.Checkpoint.TargetCommitted || started.Observed.Target.Running == nil || !*started.Observed.Target.Running || started.Observed.Target.Healthy == nil || !*started.Observed.Target.Healthy) {
		err = fmt.Errorf("target readiness/intent commit is unverified")
	}
	if err != nil {
		b.finishEngineSwitch(p, settings.SwitchFailedTargetStart, &started.Observed, err)
		return
	}
	if b.advanceEngineSwitch(p.OperationID, settings.SwitchTargetReady, &started.Observed, nil, false, nil) != nil {
		return
	}
	if err := b.publishSwitchTarget(ctx, p.TargetEngine); err != nil {
		b.finishEngineSwitch(p, settings.SwitchFailedPublication, &started.Observed, err)
		return
	}
	if err := b.releaseSwitchDrain(ctx, p.SourceEngine); err != nil {
		b.finishEngineSwitch(p, settings.SwitchFailedPublication, &started.Observed, err)
		return
	}
	if err := ctx.Err(); err != nil {
		b.finishEngineSwitch(p, settings.SwitchFailedPublication, &started.Observed, err)
		return
	}
	if err := b.advanceEngineSwitch(p.OperationID, settings.SwitchCompleted, &started.Observed, nil, false, nil); err != nil {
		if switchErrorDetail(err, settings.SwitchErrorStorage).Kind == settings.SwitchErrorRemoteVeto {
			b.finishEngineSwitch(p, settings.SwitchFailedPublication, &started.Observed, err)
		}
		return
	}
	// The manager's checkpoint outlives the broker's durable terminal receipt.
	if _, err := b.switchStep(ctx, p, settings.SwitchStepRelease); err != nil {
		b.workersMu.Lock()
		interrupted := b.switchCoordinator().interrupted
		b.workersMu.Unlock()
		if !interrupted {
			b.blockEngineSwitch(err)
		}
	}
}

func (b *Broker) blockEngineSwitch(err error) {
	b.engineConfigMu.Lock()
	permanent := switchRecoveryPermanent(err)
	if permanent && b.switchCoordinator().err == nil {
		b.switchCoordinator().err = err
	} else if !permanent {
		b.switchCoordinator().recoveryErr = err
	}
	b.engineConfigMu.Unlock()
}

func (b *Broker) finishEngineSwitch(p settings.SwitchPrepareRequest, phase string, observed *settings.SwitchObserved, failure error) {
	if b.finishInterruptedEngineSwitch(p, observed) {
		return
	}
	// Cancellation never prevents bounded conservative cleanup or the durable
	// terminal write. Reconcile never starts a target or restores a source.
	cleanupDeadline := time.Now().Add(settings.SwitchDrainBudget)
	if p.Deadline.Before(cleanupDeadline) {
		cleanupDeadline = p.Deadline
	}
	ctx, cancel := context.WithDeadline(context.Background(), cleanupDeadline)
	defer cancel()
	for {
		b.workersMu.Lock()
		var veto *settings.SwitchLocalVeto
		if c := b.switchCoordinator(); c.activeID == p.OperationID {
			veto = c.localVeto
		}
		b.workersMu.Unlock()
		if veto != nil {
			failure = switchError(settings.SwitchErrorRemoteVeto, "Local Off or pause vetoed the switch.")
		}
		var result settings.SwitchReconcileResult
		err := b.switchManagerCall(ctx, settings.SwitchReconcileMethod, settings.SwitchReconcileRequest{SwitchPrepareRequest: p, LocalVeto: veto}, &result)
		// Only an authoritative reply can prove that prepare left no reservation.
		// A missing checkpoint in a transport/storage failure proves nothing.
		noReservation := err == nil && result.Checkpoint == nil && phase == settings.SwitchFailedValidation && (result.Error == nil || result.Error.Kind == settings.SwitchErrorInvalidRequest)
		if err == nil && result.Error != nil && !switchCleanupSettled(result) {
			err = &switchRPCError{*result.Error}
		}
		if err == nil && result.Observed.Source.Intent != "" && result.Observed.Target.Intent != "" {
			observed = &result.Observed
		}
		// A prepare rejection may have already released its own provisional row.
		if noReservation {
			err = nil
		}
		if drainErr := b.releaseSwitchDrain(ctx, p.SourceEngine); drainErr != nil {
			err = errors.Join(err, drainErr)
		}
		// Preserve storage uncertainty even if interruption or terminal persistence
		// prevents the normal cleanup-error path below from running.
		if err != nil {
			defer b.blockEngineSwitch(err)
		}
		if b.finishInterruptedEngineSwitch(p, observed) {
			return
		}
		if advanceErr := b.advanceEngineSwitch(p.OperationID, phase, observed, failure, err != nil, veto); advanceErr != nil {
			if switchErrorDetail(advanceErr, settings.SwitchErrorStorage).Kind == settings.SwitchErrorRemoteVeto {
				failure = advanceErr
				continue
			}
			return
		}
		if err != nil {
			return
		}
		if result.Checkpoint != nil && result.Checkpoint.Phase != settings.SwitchCheckpointReleased {
			if _, releaseErr := b.switchStep(ctx, p, settings.SwitchStepRelease); releaseErr != nil {
				b.blockEngineSwitch(releaseErr)
			}
		}
		return
	}
}

func (b *Broker) finishInterruptedEngineSwitch(p settings.SwitchPrepareRequest, observed *settings.SwitchObserved) bool {
	b.workersMu.Lock()
	c := b.switchCoordinator()
	if c.activeManager != nil {
		select {
		case <-c.activeManager.Done():
			c.interrupted = true
		default:
		}
	}
	interrupted := c.interrupted
	b.workersMu.Unlock()
	if interrupted {
		// Replacement recovery must own cleanup after this operation releases
		// settingsApplyMu. Never wait for the replacement from inside that lock.
		if observed != nil && (observed.Source.Intent == "" || observed.Target.Intent == "") {
			observed = nil
		}
		_ = b.advanceEngineSwitch(p.OperationID, settings.SwitchFailedInterrupted, observed,
			switchError(settings.SwitchErrorInterrupted, "Engine manager restarted during the switch."), true, nil)
		return true
	}
	return false
}

func switchForwardStep(ctx context.Context, w *rpcWorker, p settings.SwitchPrepareRequest, step string) (settings.SwitchStepResult, error) {
	var result settings.SwitchStepResult
	err := switchManagerCallOn(ctx, w, settings.SwitchStepMethod, settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: step}, &result)
	if err == nil && result.Error != nil {
		err = &switchRPCError{*result.Error}
	}
	return result, err
}

// Reconcile may return the checkpoint's historical operation error even when
// no cleanup step remains. Inspect it without confusing it with a new cleanup
// failure; incomplete/persistence checkpoints always retain their reservation.
func switchCleanupSettled(result settings.SwitchReconcileResult) bool {
	cp := result.Checkpoint
	if cp == nil {
		return false
	}
	if cp.Phase == settings.SwitchCheckpointStopPending || cp.Phase == settings.SwitchCheckpointStartPending {
		return false
	}
	if result.Error == nil {
		return true
	}
	return cp.Error != nil && *cp.Error == *result.Error && result.Error.Kind != settings.SwitchErrorStorage
}

func (b *Broker) publishSwitchTarget(ctx context.Context, engine string) error {
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if err := b.switchTargetRoutable(engine); err != nil {
		return err
	}
	var launch settings.LaunchState
	if err := b.switchManagerCall(ctx, "engine:get-launch", map[string]string{"engine": engine}, &launch); err != nil {
		return err
	}
	p := b.settingsProxy(engine)
	_, port := p.Status(engine)
	if !launch.Running || launch.EffectivePort < 1 || launch.EffectivePort == port {
		return fmt.Errorf("target server endpoint is not ready")
	}
	if err := b.switchProxySetLocal(ctx, p, engine, launch.EffectivePort, true); err != nil {
		return err
	}
	profile, _ := engineProxyProfileFor(engine)
	b.engineProxy(profile).backendPort.Store(int32(launch.EffectivePort))
	if b.regCache != nil {
		b.registerService(noderec.RegisterParams{Service: profile.DiscoveryService, Port: port})
	}
	return nil
}

func (b *Broker) engineSwitchStatus(ctx context.Context, p settings.SwitchStatusRequest, caller string) (settings.SwitchStatus, error) {
	ctx, cancel := context.WithTimeout(ctx, settings.SwitchStatusWaitBudget)
	defer cancel()
	for {
		b.engineConfigMu.Lock()
		if err := b.authorizeSettingsCallerLocked(caller); err != nil {
			b.engineConfigMu.Unlock()
			return settings.SwitchStatus{}, err
		}
		snapshot, token, err := b.switchSnapshotLocked(ctx)
		if err != nil {
			b.engineConfigMu.Unlock()
			return settings.SwitchStatus{}, err
		}
		status := settings.SwitchStatus{ConfigurationRevision: token, ActiveRequestID: b.engineSwitch.store.journal.ActiveRequestID}
		if p.RequestID != "" {
			r, ok := b.engineSwitch.store.Lookup(p.RequestID)
			identity := caller
			if identity == "" {
				identity = settings.SwitchOriginLocal
			}
			if !ok || r.Caller != identity {
				b.engineConfigMu.Unlock()
				return status, switchError(settings.SwitchErrorUnknownReceipt, "Unknown switch receipt for this caller.")
			}
			status.Receipt = &r
			if snapshot.Checkpoint != nil && snapshot.Checkpoint.OperationID == p.RequestID {
				status.Observed = &snapshot.Observed
			}
		}
		changed := b.engineSwitch.changed
		b.engineConfigMu.Unlock()
		if p.AfterSequence == nil || status.Receipt == nil || status.Receipt.Terminal || status.Receipt.Sequence > *p.AfterSequence {
			return status, nil
		}
		select {
		case <-changed:
			continue
		case <-ctx.Done():
			return status, nil
		}
	}
}

func (b *Broker) cancelRemoteSwitch() {
	b.cancelRemoteEngineSwitch("")
}

func (b *Broker) cancelRemoteEngineSwitch(engine string) {
	b.vetoRemoteEngineSwitch(engine, true)
}

// A local stop response can arrive after its reservation has been released.
// Its cancellation must not apply to a subsequently accepted operation.
func (b *Broker) cancelRemoteEngineSwitchOperation(engine, operationID string) {
	c := b.switchCoordinator()
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	c.waitMu.Lock()
	defer c.waitMu.Unlock()
	if operationID != "" && c.activeID == operationID && c.remoteCancel != nil && (engine == c.remoteSource || engine == c.remoteTarget) {
		c.remoteCancel()
	}
}

func (b *Broker) vetoRemoteEngineSwitch(engine string, cancel bool) {
	// Serialize recovery-time pause admission with its final reconcile/release
	// decision, just like local Off. Never wait for physical cleanup here.
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	c := b.switchCoordinator()
	b.workersMu.Lock()
	c.waitMu.Lock()
	if cp := c.recoveryCheckpoint; cp != nil && cp.Phase != settings.SwitchCheckpointReleased && cp.Origin == settings.SwitchOriginRemote && (engine == "" || engine == cp.SourceEngine || engine == cp.TargetEngine) {
		c.vetoID, c.localVeto = cp.OperationID, &settings.SwitchLocalVeto{Engine: engine, Pause: engine == ""}
	}
	if c.remoteCancel != nil && (engine == "" || engine == c.remoteSource || engine == c.remoteTarget) {
		c.localVeto = &settings.SwitchLocalVeto{Engine: engine, Pause: engine == ""}
		if cancel {
			c.remoteCancel()
		}
	}
	c.waitMu.Unlock()
	b.workersMu.Unlock()
}

func (b *Broker) dispatchEngineSwitch(ctx context.Context, method string, raw json.RawMessage, caller string) (any, error) {
	switch method {
	case settings.SwitchMethod:
		p, err := settings.DecodeSwitchRequest(raw)
		if err != nil {
			return nil, switchError(settings.SwitchErrorInvalidRequest, err.Error())
		}
		if caller != "" && p.NodeID != "" && p.NodeID != b.nodeID {
			return nil, switchError(settings.SwitchErrorInvalidRequest, "A switch cannot forward another hop.")
		}
		return b.acceptEngineSwitch(ctx, p, caller)
	case settings.SwitchStatusMethod:
		p, err := settings.DecodeSwitchStatusRequest(raw)
		if err != nil {
			return nil, switchError(settings.SwitchErrorInvalidRequest, err.Error())
		}
		if caller != "" && p.NodeID != "" && p.NodeID != b.nodeID {
			return nil, switchError(settings.SwitchErrorInvalidRequest, "Status cannot forward another hop.")
		}
		return b.engineSwitchStatus(ctx, p, caller)
	default:
		return nil, switchError(settings.SwitchErrorInvalidRequest, "Unsupported switch method.")
	}
}

func (b *Broker) handleEngineSwitch(msg *Message) {
	var route struct {
		NodeID string `json:"nodeId"`
	}
	if json.Unmarshal(msg.Params, &route) == nil && route.NodeID != "" && route.NodeID != b.nodeID {
		w := b.getEngineMgr()
		if w == nil {
			_ = b.codec.RespondError(msg.ID, -32000, "Engine manager unavailable.")
			return
		}
		method := settings.RemoteSwitchMethod
		if msg.Method == settings.SwitchStatusMethod {
			method = settings.RemoteSwitchStatusMethod
		}
		raw, rpcErr, err := w.CallNoTimeout(context.Background(), method, msg.Params)
		if err != nil {
			_ = b.codec.RespondError(msg.ID, -32000, err.Error())
		} else if rpcErr != nil {
			_ = b.codec.RespondError(msg.ID, rpcErr.Code, rpcErr.Message)
		} else {
			_ = b.codec.Respond(msg.ID, raw)
		}
		return
	}
	result, err := b.dispatchEngineSwitch(context.Background(), msg.Method, msg.Params, "")
	if err != nil {
		detail := switchErrorDetail(err, settings.SwitchErrorStorage)
		_ = b.codec.RespondError(msg.ID, detail.Code, detail.Message)
	} else {
		_ = b.codec.Respond(msg.ID, result)
	}
}

func (b *Broker) handleSwitchEngineNotification(method string, raw json.RawMessage) bool {
	switch method {
	case settings.SwitchChangedNotification:
		var changed settings.SwitchChanged
		if json.Unmarshal(raw, &changed) == nil && changed.NodeID != "" && changed.RequestID != "" && changed.Sequence > 0 {
			b.notifyClients(method, raw)
		}
		return true
	case "engine:switch-cancel":
		var p struct {
			ID string `json:"id"`
		}
		if json.Unmarshal(raw, &p) == nil {
			c := b.switchCoordinator()
			c.waitMu.Lock()
			if cancel := c.waiters[p.ID]; cancel != nil {
				cancel()
			}
			c.waitMu.Unlock()
		}
		return true
	case "engine:switch-request":
		var p struct {
			ID     string          `json:"id"`
			Method string          `json:"method"`
			Caller string          `json:"caller"`
			Params json.RawMessage `json:"params"`
		}
		if json.Unmarshal(raw, &p) != nil || p.ID == "" || p.Caller == "" {
			return true
		}
		c := b.switchCoordinator()
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		c.waitMu.Lock()
		if len(c.waiters) >= 32 || c.waiters[p.ID] != nil {
			c.waitMu.Unlock()
			cancel()
			return true
		}
		c.waiters[p.ID] = cancel
		c.waitMu.Unlock()
		w := b.getEngineMgr()
		go func() {
			defer cancel()
			defer func() { c.waitMu.Lock(); delete(c.waiters, p.ID); c.waitMu.Unlock() }()
			result, err := b.dispatchEngineSwitch(ctx, p.Method, p.Params, p.Caller)
			reply := struct {
				ID        string `json:"id"`
				Result    any    `json:"result"`
				Error     string `json:"error,omitempty"`
				ErrorCode int    `json:"errorCode,omitempty"`
			}{ID: p.ID, Result: result}
			if err != nil {
				detail := switchErrorDetail(err, settings.SwitchErrorStorage)
				reply.Error, reply.ErrorCode = detail.Message, detail.Code
			}
			if w != nil {
				_ = w.Notify("engine:switch-reply", reply)
			}
		}()
		return true
	}
	return false
}
