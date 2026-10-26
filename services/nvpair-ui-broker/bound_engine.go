// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"reflect"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

const boundEngineDwell = 60 * time.Second
const boundEngineMaxWaiters = 1024 // same bound as proxy buffer/queue waiters

type boundEngineCoordinator struct {
	mu      sync.Mutex // reader-pump correlation only; never held during IPC
	waiters map[boundWaiterKey]context.CancelFunc
	active  *boundEngineOperation // engineConfigMu
}

type boundWaiterKey struct {
	source  *proxyProcess
	run, id string
}

type boundEngineOperation struct {
	operationID        string
	coldAccepted       bool // engineConfigMu
	localOff           bool // engineConfigMu
	offSignal          chan struct{}
	offResult          json.RawMessage // immutable after done
	offError           error           // immutable after done
	uncertain          bool            // engineConfigMu; retains the original cold ID
	fenceError         error           // engineConfigMu
	request            nodepolicy.PrepareBoundEngineRequest
	proxy              *proxyProcess
	manager            *rpcWorker
	managerSource      *engineManagerSource
	managerIncarnation string
	sourceEngine       string
	deadline           time.Time
	cancel             context.CancelFunc
	done               chan struct{}
	result             nodepolicy.BoundEnginePrepared // immutable after done
	canonicalModels    map[string]bool                // immutable after done, fresh target inventory
	waiters            int                            // engineConfigMu
	waiterContexts     map[context.Context]bool       // engineConfigMu
	accepted           bool                           // engineConfigMu; disconnect cannot cancel accepted work
}

type boundEngineError struct {
	reason string
	code   int
}

func (e *boundEngineError) Error() string {
	return "Bound engine preparation refused: " + e.reason + "."
}
func boundRefusal(reason string, code int) error { return &boundEngineError{reason, code} }

func boundEngineResult(err error) nodepolicy.BoundEnginePrepared {
	if err == nil {
		return nodepolicy.BoundEnginePrepared{Success: true}
	}
	r := nodepolicy.BoundEnginePrepared{Reason: "unavailable", Code: -32000, Message: "Could not verify bound engine readiness."}
	var refusal *boundEngineError
	if errors.As(err, &refusal) {
		r.Reason, r.Code = refusal.reason, refusal.code
	}
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
		r.Reason = "engine-switch-timeout"
	}
	detail := switchErrorDetail(err, settings.SwitchErrorValidation)
	switch detail.Code {
	case -32010:
		r.Reason, r.Code = "stale-revision", detail.Code
	case -32011:
		r.Reason, r.Code = "paused", detail.Code
	case -32012:
		r.Reason, r.Code = "management-denied", detail.Code
	case -32013:
		r.Reason, r.Code = "engine-locked", detail.Code
	}
	if detail.Kind == settings.SwitchErrorBusy {
		r.Reason = "engine-switch-busy"
	}
	// Native errors can contain paths or credentials. Only fixed copy crosses
	// the proxy seam; numeric codes retain their established meaning.
	r.Message = "Bound engine preparation refused: " + r.Reason + "."
	return r
}

func (b *Broker) boundCoordinator() *boundEngineCoordinator {
	b.boundInit.Do(func() { b.boundEngines = &boundEngineCoordinator{waiters: make(map[boundWaiterKey]context.CancelFunc)} })
	return b.boundEngines
}

func (b *Broker) handleBoundEngineNotification(source *proxyProcess, method string, raw json.RawMessage) bool {
	if method != nodepolicy.MethodPrepareBoundEngine && method != nodepolicy.MethodCancelBoundEngine {
		return false
	}
	c := b.boundCoordinator()
	if method == nodepolicy.MethodCancelBoundEngine {
		var in nodepolicy.CancelBoundEngineRequest
		if json.Unmarshal(raw, &in) == nil {
			c.mu.Lock()
			cancel := c.waiters[boundWaiterKey{source, in.ProxyRunID, in.RequestID}]
			c.mu.Unlock()
			if cancel != nil {
				cancel()
			}
		}
		return true
	}
	var in nodepolicy.PrepareBoundEngineRequest
	if json.Unmarshal(raw, &in) != nil {
		return true
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(in.BudgetMs)*time.Millisecond)
	key := boundWaiterKey{source, in.ProxyRunID, in.RequestID}
	c.mu.Lock()
	if len(c.waiters) >= boundEngineMaxWaiters || c.waiters[key] != nil {
		c.mu.Unlock()
		cancel()
		return true
	}
	c.waiters[key] = cancel
	c.mu.Unlock()
	go func() {
		defer cancel()
		defer func() { c.mu.Lock(); delete(c.waiters, key); c.mu.Unlock() }()
		manager := b.getEngineMgr()
		b.workersMu.Lock()
		managerSource := b.engineMgrSource
		b.workersMu.Unlock()
		result := b.prepareBoundEngine(ctx, source, in)
		result.RequestID, result.ProxyRunID = in.RequestID, in.ProxyRunID
		if ctx.Err() == nil && managerSource != nil && b.engineMgrPairCurrent(manager, managerSource) && b.engineManagerSourceCurrent(managerSource) && b.getEngineMgr() == manager && b.boundProxyCurrent(source, in.ProxyRunID) {
			_ = source.peer.NotifyContext(ctx, nodepolicy.NotifyBoundEnginePrepared, result)
		}
	}()
	return true
}

func (b *Broker) boundProxyCurrent(source *proxyProcess, run string) bool {
	b.workersMu.Lock()
	current := source != nil && b.proxyProc == source
	if current {
		select {
		case <-source.Done():
			current = false
		default:
		}
	}
	b.workersMu.Unlock()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	current = current && r.incarnation == source && r.runID == run
	r.mu.Unlock()
	return current
}

// Caller owns engineConfigMu and r.mu. Configuration publication and final
// automatic acceptance use this same short authority boundary.
func (b *Broker) boundAuthorityLocked(r *policyRuntime, source *proxyProcess, in nodepolicy.PrepareBoundEngineRequest, mutate bool) error {
	b.workersMu.Lock()
	current := source != nil && b.proxyProc == source
	if current {
		select {
		case <-source.Done():
			current = false
		default:
		}
	}
	b.workersMu.Unlock()
	if !current {
		return boundRefusal("unavailable", -32000)
	}
	if r.incarnation != source || r.runID != in.ProxyRunID || r.recoveryRequired || r.activationLocked(source, r.current.Revision).Activation != activationActive {
		return boundRefusal("unavailable", -32000)
	}
	if r.current.Revision != in.PolicyRevision {
		return boundRefusal("stale-revision", -32010)
	}
	if r.live != nodepolicy.Available || r.managementBlockedLocked() {
		return boundRefusal("paused", -32011)
	}
	if r.drains[in.Engine] {
		return boundRefusal("draining", -32000)
	}
	if in.PeerID != "" {
		mesh := clustertrust.Open(b.clusterDir)
		incarnation, ok, diagnostic := mesh.PeerServiceIncarnationWithDiagnostic(in.PeerID)
		if !ok || incarnation != in.PeerIncarnation {
			if ok {
				diagnostic.Predicate = "incarnation_mismatch"
			}
			diagnostic.ExpectedIncarnation = in.PeerIncarnation
			diagnostic.ObservedIncarnation = incarnation
			logBoundManagementRefusal("broker.peer_incarnation", in.PeerID, "not_checked", in.ProxyRunID, diagnostic)
			return boundRefusal("management-denied", -32012)
		}
		if mutate {
			if err := b.authorizeRemoteManagementLocked(r, in.PeerID, "bound-engine", in.RequestID); err != nil {
				return err
			}
		}
	}
	naming, known := b.engineModelNaming(in.Engine)
	if !known {
		return boundRefusal("model-unavailable", -32000)
	}
	canonical := engines.NormalizeModelName(in.Model, naming)
	bound := false
	for _, ref := range r.current.EngineBindings {
		if ref.Engine == in.Engine && engines.NormalizeModelName(ref.Model, naming) == canonical {
			bound = true
		}
	}
	if !bound {
		return boundRefusal("model-unavailable", -32000)
	}
	if r.current.ServedModels != nil {
		served := false
		for _, ref := range r.current.ServedModels {
			if ref.Engine == in.Engine && engines.NormalizeModelName(ref.Model, naming) == canonical {
				served = true
			}
		}
		if !served {
			return boundRefusal("model-unavailable", -32000)
		}
	}
	return nil
}

func (b *Broker) prepareBoundEngine(ctx context.Context, source *proxyProcess, in nodepolicy.PrepareBoundEngineRequest) nodepolicy.BoundEnginePrepared {
	if in.Validate() != nil {
		return boundEngineResult(boundRefusal("unavailable", -32602))
	}
	if !b.boundProxyCurrent(source, in.ProxyRunID) {
		return boundEngineResult(boundRefusal("unavailable", -32000))
	}
	c := b.boundCoordinator()
	b.engineConfigMu.Lock()
	b.retireReplacedBoundOperationLocked()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	err := b.boundAuthorityLocked(r, source, in, false)
	r.mu.Unlock()
	if err != nil {
		b.engineConfigMu.Unlock()
		return boundEngineResult(err)
	}
	op := c.active
	if op != nil {
		if op.proxy != source || op.request.Engine != in.Engine {
			b.engineConfigMu.Unlock()
			return boundEngineResult(boundRefusal("engine-switch-busy", -32000))
		}
	} else {
		deadline := time.Now().Add(time.Duration(in.BudgetMs) * time.Millisecond)
		if earlier, ok := ctx.Deadline(); ok && earlier.Before(deadline) {
			deadline = earlier
		}
		opCtx, cancel := context.WithDeadline(context.Background(), deadline)
		op = &boundEngineOperation{request: in, proxy: source, deadline: deadline, cancel: cancel, done: make(chan struct{}), operationID: rand.Text(), offSignal: make(chan struct{})}
		c.active = op
		go b.runBoundEngine(opCtx, op)
	}
	op.waiters++
	if op.waiterContexts == nil {
		op.waiterContexts = make(map[context.Context]bool)
	}
	op.waiterContexts[ctx] = true
	b.engineConfigMu.Unlock()
	defer func() {
		b.engineConfigMu.Lock()
		op.waiters--
		delete(op.waiterContexts, ctx)
		if op.waiters == 0 && !op.accepted {
			op.cancel()
		}
		b.engineConfigMu.Unlock()
	}()
	select {
	case <-ctx.Done():
		return boundEngineResult(ctx.Err())
	case <-op.done:
		result := op.result
		if result.Success && (!b.boundManagerCurrent(op) || !b.boundProxyCurrent(source, in.ProxyRunID)) {
			refused := boundEngineResult(boundRefusal("unavailable", -32000))
			refused.SwitchRequestID = result.SwitchRequestID
			return refused
		}
		// Joining a ready operation does not borrow another peer's authority.
		// Every waiter retains its original verified relationship on readback.
		b.engineConfigMu.Lock()
		r.mu.Lock()
		err := b.boundAuthorityLocked(r, source, in, false)
		r.mu.Unlock()
		b.engineConfigMu.Unlock()
		if err != nil {
			refused := boundEngineResult(err)
			refused.SwitchRequestID = result.SwitchRequestID
			return refused
		}
		if result.Success && !op.canonicalModels[b.modelRef(in.Engine, in.Model).Model] {
			refused := boundEngineResult(boundRefusal("model-unavailable", -32000))
			refused.SwitchRequestID = result.SwitchRequestID
			return refused
		}
		return result
	}
}

type boundEngineStatus struct {
	Engine    string `json:"engine"`
	Installed *bool  `json:"installed"`
	Running   *bool  `json:"running"`
	Healthy   *bool  `json:"healthy"`
	Locked    *bool  `json:"locked"`
}

func (b *Broker) boundStatus(ctx context.Context, op *boundEngineOperation) ([]boundEngineStatus, error) {
	var result struct {
		Engines []boundEngineStatus `json:"engines"`
	}
	err := switchManagerCallOn(ctx, op.manager, "engine:get-installed", nil, &result)
	if err != nil {
		return nil, err
	}
	if len(result.Engines) == 0 {
		return nil, boundRefusal("unavailable", -32000)
	}
	seen := map[string]bool{}
	for _, st := range result.Engines {
		if seen[st.Engine] || st.Installed == nil || st.Running == nil || st.Healthy == nil || st.Locked == nil {
			return nil, boundRefusal("unavailable", -32000)
		}
		seen[st.Engine] = true
	}
	return result.Engines, nil
}

func (b *Broker) boundManagerCurrent(op *boundEngineOperation) bool {
	return op.managerSource != nil && b.engineMgrPairCurrent(op.manager, op.managerSource) && b.engineManagerSourceCurrent(op.managerSource) && b.getEngineMgr() == op.manager
}

func (b *Broker) checkBoundOperation(ctx context.Context, op *boundEngineOperation, mutate bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !b.boundProxyCurrent(op.proxy, op.request.ProxyRunID) {
		return boundRefusal("unavailable", -32000)
	}
	if op.manager != nil && !b.boundManagerCurrent(op) {
		return boundRefusal("unavailable", -32000)
	}
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	return b.boundAuthorityLocked(r, op.proxy, op.request, mutate)
}

func (b *Broker) runBoundEngine(ctx context.Context, op *boundEngineOperation) {
	var err error
	var switchID string
	defer func() {
		op.cancel()
		b.engineConfigMu.Lock()
		op.result = boundEngineResult(err)
		op.result.SwitchRequestID = switchID
		if b.boundCoordinator().active == op && !op.uncertain {
			b.boundEngines.active = nil
		}
		close(op.done)
		b.engineConfigMu.Unlock()
	}()
	// TryLock refuses competing configuration work without waiting behind a
	// potentially long drain. Transient stale snapshots are retried freshly.
	for {
		if err = b.checkBoundOperation(ctx, op, false); err != nil {
			return
		}
		if !b.settingsApplyMu.TryLock() {
			err = boundRefusal("engine-switch-busy", -32000)
			return
		}
		var snapshot switchManagerSnapshot
		var token string
		b.engineConfigMu.Lock()
		op.manager = b.getEngineMgr()
		b.workersMu.Lock()
		op.managerSource = b.engineMgrSource
		b.workersMu.Unlock()
		if !b.boundManagerCurrent(op) {
			err = boundRefusal("unavailable", -32000)
		} else {
			snapshot, token, err = b.switchSnapshotLocked(ctx)
		}
		if err == nil {
			err = b.pendingSwitchLocked("")
		}
		if err == nil {
			for _, record := range b.engineSettings {
				if record.Snapshot.Phase == "applying" {
					err = boundRefusal("engine-switch-busy", -32000)
				}
			}
		}
		b.engineConfigMu.Unlock()
		if err != nil {
			b.settingsApplyMu.Unlock()
			return
		}
		op.managerIncarnation = snapshot.Configuration.ManagerIncarnation
		var statuses []boundEngineStatus
		statuses, err = b.boundStatus(ctx, op)
		if err != nil {
			b.settingsApplyMu.Unlock()
			return
		}
		var target *boundEngineStatus
		var running []string
		for i := range statuses {
			st := &statuses[i]
			if st.Engine == op.request.Engine {
				target = st
			}
			if *st.Running && st.Engine != op.request.Engine {
				running = append(running, st.Engine)
			}
		}
		if target == nil || !*target.Installed {
			err = boundRefusal("model-unavailable", -32000)
		} else if *target.Locked {
			err = boundRefusal("engine-locked", -32013)
		}
		if err == nil {
			err = b.switchTargetRoutable(op.request.Engine)
		}
		if err != nil {
			b.settingsApplyMu.Unlock()
			return
		}
		if *target.Running {
			if !*target.Healthy {
				err = boundRefusal("model-unavailable", -32000)
			} else {
				err = b.finishBoundEngine(ctx, op)
			}
			b.settingsApplyMu.Unlock()
			return
		}
		if err = b.checkBoundOperation(ctx, op, true); err != nil {
			b.settingsApplyMu.Unlock()
			return
		}
		if len(running) > 1 {
			b.settingsApplyMu.Unlock()
			err = boundRefusal("engine-switch-busy", -32000)
			return
		}
		if len(running) == 0 {
			b.engineConfigMu.Lock()
			r := b.nodePolicyRuntime()
			r.mu.Lock()
			err = b.boundAuthorityLocked(r, op.proxy, op.request, true)
			if err == nil && (ctx.Err() != nil || !liveBoundWaiters(op) || !b.engineMgrPairCurrent(op.manager, op.managerSource) || r.engineManagerSource != op.managerSource || b.getEngineMgr() != op.manager) {
				err = boundRefusal("unavailable", -32000)
			}
			if err == nil {
				op.accepted, op.coldAccepted = true, true
			}
			r.mu.Unlock()
			b.engineConfigMu.Unlock()
			if err == nil {
				p := nodepolicy.ActivateBoundEngineRequest{OperationID: op.operationID, Engine: op.request.Engine, Model: op.request.Model, PolicyRevision: op.request.PolicyRevision, ManagerIncarnation: op.managerIncarnation, Deadline: op.deadline.UTC(), PeerID: op.request.PeerID, PeerIncarnation: op.request.PeerIncarnation}
				err = b.activateBoundCold(ctx, op, p)
			}
			if err == nil {
				err = b.finishBoundEngine(ctx, op)
			}
			b.settingsApplyMu.Unlock()
			return
		}
		b.engineConfigMu.Lock()
		op.sourceEngine = running[0]
		r := b.nodePolicyRuntime()
		r.mu.Lock()
		protected := b.protectedResidentLocked(op.sourceEngine, "")
		r.mu.Unlock()
		until := b.boundDwellUntilLocked()
		b.engineConfigMu.Unlock()
		b.settingsApplyMu.Unlock()
		if protected {
			err = boundRefusal("no-fit", -32000)
			return
		}
		if time.Now().Before(until) {
			if !until.Before(op.deadline) {
				err = boundRefusal("engine-switch-cooldown", -32000)
				return
			}
			timer := time.NewTimer(time.Until(until))
			select {
			case <-ctx.Done():
				timer.Stop()
				err = ctx.Err()
				return
			case <-timer.C:
			}
			continue // No reservation/token survives the dwell readback.
		}
		p := settings.SwitchRequest{SourceEngine: running[0], TargetEngine: op.request.Engine, ExpectedRevision: token}
		var result settings.SwitchResult
		result, err = b.acceptAutomaticEngineSwitch(context.WithValue(ctx, boundSwitchKey{}, op), p, op.request.PeerID, op.proxy, op.deadline)
		switchID = result.Receipt.RequestID
		if err != nil {
			// Known IDs, including uncertain acceptance, never become a new
			// submission. Their original journal row owns readback/recovery.
			if switchID == "" && ctx.Err() == nil && switchErrorDetail(err, "").Kind == settings.SwitchErrorConfigurationConflict {
				continue
			}
			return
		}
		b.engineConfigMu.Lock()
		op.accepted = true
		b.engineConfigMu.Unlock()
		err = b.waitBoundSwitch(ctx, op, switchID)
		if err == nil {
			for !b.settingsApplyMu.TryLock() {
				timer := time.NewTimer(10 * time.Millisecond)
				select {
				case <-ctx.Done():
					timer.Stop()
					err = ctx.Err()
					return
				case <-timer.C:
				}
			}
			err = b.finishBoundEngine(ctx, op)
			b.settingsApplyMu.Unlock()
		}
		return
	}
}

func (b *Broker) boundDwellUntilLocked() time.Time {
	until := b.startedAt.Add(boundEngineDwell)
	if c := b.switchCoordinator(); c.store != nil {
		for _, receipt := range c.store.journal.Receipts {
			if receipt.Phase == settings.SwitchCompleted && receipt.FinishedAt != nil && receipt.FinishedAt.Add(boundEngineDwell).After(until) {
				until = receipt.FinishedAt.Add(boundEngineDwell)
			}
		}
	}
	return until
}

// The original settings operation lock excludes a later local On until its
// admitted Off is physically discharged. Private scoped deliveries become
// inert after the manager's cold reservation changes, just like E4 Off.
func (b *Broker) handleBoundEngineOff(msg *Message, engine string) bool {
	b.engineConfigMu.Lock()
	b.retireReplacedBoundOperationLocked()
	op := b.boundCoordinator().active
	if op == nil || (op.request.Engine != engine && op.sourceEngine != engine) {
		b.engineConfigMu.Unlock()
		return false
	}
	if op.uncertain {
		b.engineConfigMu.Unlock()
		b.resolveUncertainBoundOff(msg, op)
		return true
	}
	if !op.accepted {
		op.cancel()
	}
	if !op.coldAccepted {
		b.engineConfigMu.Unlock()
		return false
	}
	if !op.localOff {
		op.localOff = true
		close(op.offSignal)
	}
	b.engineConfigMu.Unlock()
	<-op.done
	if op.offError != nil {
		detail := switchErrorDetail(op.offError, settings.SwitchErrorInterrupted)
		_ = b.codec.RespondError(msg.ID, detail.Code, "Bound engine Off could not be verified.")
	} else {
		_ = b.codec.Respond(msg.ID, op.offResult)
	}
	return true
}

func (b *Broker) activateBoundCold(ctx context.Context, op *boundEngineOperation, p nodepolicy.ActivateBoundEngineRequest) error {
	done := make(chan error, 1)
	go func() {
		var result struct {
			Success bool `json:"success"`
		}
		err := switchManagerCallOn(ctx, op.manager, nodepolicy.MethodActivateBoundEngine, p, &result)
		if err == nil && !result.Success {
			err = boundRefusal("model-unavailable", -32000)
		}
		done <- err
	}()
	offSignal := op.offSignal
	applied := false
	var activationErr error
	for {
		select {
		case activationErr = <-done:
			var coded *switchRPCError
			unknown := activationErr != nil && !errors.As(activationErr, &coded)
			b.engineConfigMu.Lock()
			off := op.localOff
			op.coldAccepted = false
			if !off && unknown {
				b.retainBoundUncertaintyLocked(op)
			}
			b.engineConfigMu.Unlock()
			if !off {
				return activationErr
			}
			cleanup, cancel := context.WithTimeout(context.Background(), settings.SwitchDrainBudget)
			defer cancel()
			boundColdCleanupMethod := "engine:status"
			if !applied {
				boundColdCleanupMethod = "engine:stop"
			} // discharge before relinquishing original lock
			var rpcErr *RPCError
			op.offResult, rpcErr, op.offError = op.manager.CallNoTimeout(cleanup, boundColdCleanupMethod, settingsJSON(map[string]string{"engine": p.Engine}))
			if rpcErr != nil {
				op.offError = &switchRPCError{settings.SwitchError{Code: rpcErr.Code, Kind: settings.SwitchErrorInterrupted, Message: "Bound engine Off failed."}}
			}
			if op.offError != nil {
				b.engineConfigMu.Lock()
				b.retainBoundUncertaintyLocked(op)
				b.engineConfigMu.Unlock()
			}
			return boundRefusal("paused", -32011)
		case <-offSignal:
			// Delivery may beat manager startup admission. A rejected scope
			// has no effects and is retried only for this original operation.
			var result struct {
				Applied bool `json:"applied"`
			}
			err := switchManagerCallOn(ctx, op.manager, "engine:stop", struct {
				Engine      string `json:"engine"`
				OperationID string `json:"boundOperationId"`
			}{p.Engine, p.OperationID}, &result)
			if err == nil && result.Applied {
				applied = true
				offSignal = nil
				continue
			}
			timer := time.NewTimer(10 * time.Millisecond)
			select {
			case activationErr = <-done:
				timer.Stop()
				done <- activationErr
			case <-ctx.Done():
				timer.Stop()
				offSignal = nil
			case <-timer.C:
			}
		}
	}
}

func (b *Broker) resolveUncertainBoundOff(msg *Message, op *boundEngineOperation) {
	b.settingsApplyMu.Lock()
	defer b.settingsApplyMu.Unlock()
	b.engineConfigMu.Lock()
	current := b.boundCoordinator().active == op && b.boundManagerCurrent(op)
	b.engineConfigMu.Unlock()
	if !current {
		_ = b.codec.RespondError(msg.ID, -32000, "Bound activation requires current-manager recovery.")
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), settings.SwitchDrainBudget)
	defer cancel()
	var observed settings.SwitchEngineState
	err := switchManagerCallOn(ctx, op.manager, "engine:stop", struct {
		Engine     string `json:"engine"`
		VerifyStop bool   `json:"verifyStop"`
	}{op.request.Engine, true}, &observed)
	if err == nil && (!observed.TerminationVerified || observed.Intent != settings.SwitchIntentOff) {
		err = boundRefusal("unavailable", -32000)
	}
	if err != nil {
		_ = b.codec.RespondError(msg.ID, -32000, "Bound activation Off could not be verified.")
		return
	}
	// Keep the operation lock until original startup and Off have settled.
	b.engineConfigMu.Lock()
	if b.boundCoordinator().active == op {
		b.boundEngines.active = nil
	}
	if b.switchCoordinator().recoveryErr == op.fenceError {
		b.switchCoordinator().recoveryErr = nil
	}
	b.engineConfigMu.Unlock()
	status, rpcErr, callErr := op.manager.CallNoTimeout(ctx, "engine:status", settingsJSON(map[string]string{"engine": op.request.Engine}))
	if callErr != nil || rpcErr != nil {
		_ = b.codec.RespondError(msg.ID, -32000, "Bound activation stopped; status is unavailable.")
		return
	}
	_ = b.codec.Respond(msg.ID, status)
}

func (b *Broker) retainBoundUncertaintyLocked(op *boundEngineOperation) {
	op.uncertain = true
	if op.fenceError == nil {
		op.fenceError = boundRefusal("engine-switch-busy", -32000)
	}
	if b.switchCoordinator().recoveryErr == nil {
		b.switchCoordinator().recoveryErr = op.fenceError
	}
}

func (b *Broker) retireReplacedBoundOperationLocked() {
	c := b.boundCoordinator()
	op := c.active
	if op == nil || !op.uncertain {
		return
	}
	manager := b.getEngineMgr()
	if manager == nil || manager == op.manager {
		return
	}
	// Publication of a replacement follows the existing conservative manager
	// recovery. Old forward work cannot execute on that different worker.
	if b.switchCoordinator().recoveryErr == op.fenceError {
		b.switchCoordinator().recoveryErr = nil
	}
	c.active = nil
}

func (b *Broker) waitBoundSwitch(ctx context.Context, op *boundEngineOperation, id string) error {
	for {
		if err := b.checkBoundOperation(ctx, op, false); err != nil {
			return err
		}
		b.engineConfigMu.Lock()
		c := b.switchCoordinator()
		receipt, ok := c.store.Lookup(id)
		changed := c.changed
		settled := ok && receipt.Terminal && !receipt.RecoveryRequired && c.store.journal.ActiveRequestID != id && c.store.journal.AutomaticSettled[id] != 0
		recoveryErr := c.recoveryErr
		if c.err != nil {
			recoveryErr = c.err
		}
		b.engineConfigMu.Unlock()
		if recoveryErr != nil {
			return recoveryErr
		}
		if settled {
			if receipt.Error != nil {
				return &switchRPCError{*receipt.Error}
			}
			if receipt.Phase != settings.SwitchCompleted {
				return boundRefusal("model-unavailable", -32000)
			}
			return nil
		}
		// Release may not advance a manual-style terminal row. A short bounded
		// observation timer also rechecks relationship/grant changes while waiting.
		timer := time.NewTimer(20 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-changed:
			timer.Stop()
		case <-timer.C:
		}
	}
}

func (b *Broker) finishBoundEngine(ctx context.Context, op *boundEngineOperation) error {
	if err := b.checkBoundOperation(ctx, op, false); err != nil {
		return err
	}
	statuses, err := b.boundStatus(ctx, op)
	if err != nil {
		return err
	}
	ready := false
	for _, st := range statuses {
		if st.Engine == op.request.Engine && *st.Installed && *st.Running && *st.Healthy && !*st.Locked {
			ready = true
		}
	}
	if !ready {
		return boundRefusal("model-unavailable", -32000)
	}
	var launch settings.LaunchState
	if err := switchManagerCallOn(ctx, op.manager, "engine:get-launch", map[string]string{"engine": op.request.Engine}, &launch); err != nil {
		return err
	}
	_, port := op.proxy.Status(op.request.Engine)
	if !launch.Running || launch.EffectivePort < 1 || port < 1 || launch.EffectivePort == port {
		return boundRefusal("model-unavailable", -32000)
	}
	if err := b.checkBoundOperation(ctx, op, false); err != nil {
		return err
	}
	if err := b.switchProxySetLocal(ctx, op.proxy, op.request.Engine, launch.EffectivePort, true); err != nil {
		return err
	}
	var inventory struct {
		ByEngine map[string][]string `json:"modelsByEngine"`
	}
	if err := switchManagerCallOn(ctx, op.manager, "engine:models", nil, &inventory); err != nil {
		return boundRefusal("model-unavailable", -32000)
	}
	models := make(map[string]bool)
	for _, model := range inventory.ByEngine[op.request.Engine] {
		if nodepolicy.ValidateModelIdentifier(model) == nil {
			models[b.modelRef(op.request.Engine, model).Model] = true
		}
	}
	if err := b.checkBoundOperation(ctx, op, false); err != nil {
		return err
	}
	profile, _ := b.frontedEngineProfile(op.request.Engine)
	b.engineProxy(profile).backendPort.Store(int32(launch.EffectivePort))
	if b.regCache != nil {
		b.registerService(noderec.RegisterParams{Service: profile.DiscoveryService, Port: port})
	}
	op.canonicalModels = models
	return nil
}

type boundSwitchKey struct{}

func liveBoundWaiters(op *boundEngineOperation) bool {
	for ctx := range op.waiterContexts {
		if ctx.Err() == nil {
			return true
		}
	}
	return false
}

// Called by E4 under engineConfigMu and r.mu, immediately before journaling.
func (b *Broker) admitBoundSwitchLocked(ctx context.Context, r *policyRuntime) error {
	op, _ := ctx.Value(boundSwitchKey{}).(*boundEngineOperation)
	if op == nil {
		return nil
	} // Task 2's existing internal/manual paths
	if !liveBoundWaiters(op) || ctx.Err() != nil || op.managerSource == nil || !b.engineMgrPairCurrent(op.manager, op.managerSource) || r.engineManagerSource != op.managerSource || b.getEngineMgr() != op.manager {
		return boundRefusal("unavailable", -32000)
	}
	if err := b.boundAuthorityLocked(r, op.proxy, op.request, true); err != nil {
		return err
	}
	if b.protectedResidentLocked(op.sourceEngine, "") {
		return boundRefusal("no-fit", -32000)
	}
	if time.Now().Before(b.boundDwellUntilLocked()) {
		return boundRefusal("engine-switch-cooldown", -32000)
	}
	op.accepted = true
	return nil
}

// Relevant policy writes and automatic reservations share engineConfigMu then
// r.mu. No policy mutex is held across startup, drain, IPC or output.
func (b *Broker) boundPolicyChangeLocked(before, after nodepolicy.Policy) error {
	b.retireReplacedBoundOperationLocked()
	reserved := map[string]bool{}
	if c := b.boundCoordinator(); c.active != nil {
		reserved[c.active.request.Engine] = true
		if c.active.sourceEngine != "" {
			reserved[c.active.sourceEngine] = true
		}
	}
	if c := b.switchCoordinator(); c.store != nil {
		for id, receipt := range c.store.journal.Receipts {
			if receipt.Automatic && (!receipt.Terminal || receipt.RecoveryRequired && !c.resolved[id] || c.store.journal.AutomaticSettled[id] == 0) {
				reserved[receipt.Request.SourceEngine] = true
				reserved[receipt.Request.TargetEngine] = true
			}
		}
		if cp := c.recoveryCheckpoint; cp != nil && cp.Automatic && cp.Phase != settings.SwitchCheckpointReleased {
			reserved[cp.SourceEngine] = true
			reserved[cp.TargetEngine] = true
		}
	}
	filter := func(refs []nodepolicy.Role, engine string) []nodepolicy.Role {
		var out []nodepolicy.Role
		for _, ref := range refs {
			if ref.Engine == engine {
				out = append(out, ref)
			}
		}
		return out
	}
	for engine := range reserved {
		if !reflect.DeepEqual(filter(before.EngineBindings, engine), filter(after.EngineBindings, engine)) ||
			(before.ServedModels == nil) != (after.ServedModels == nil) ||
			!reflect.DeepEqual(filter(before.ServedModels, engine), filter(after.ServedModels, engine)) ||
			!reflect.DeepEqual(filter(before.ResidentModels, engine), filter(after.ResidentModels, engine)) {
			return switchError(settings.SwitchErrorBusy, "Automatic engine preparation reserves this policy reference.")
		}
	}
	return nil
}
