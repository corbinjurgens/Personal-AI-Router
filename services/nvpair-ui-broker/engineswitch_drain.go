// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

// nodeDrainLease owns only registry entries newly added by drainNode. Task 6
// keeps this private lease until its manager operation has conclusively stopped.
// It is not a reservation and does not own engine lifecycle or admission state.
// The existing policy push slot serializes access to added, including concurrent
// releases; an entry leaves this lease only after confirmed cleanup.
type nodeDrainLease struct {
	broker       *Broker
	added        []string
	settingsHeld bool
}

// drainNode retains settingsApplyMu until cleanup completes, excluding other
// node drains and switch/settings owners of the same registry entries. The
// caller still owns manager reservation, catalog stability and its operation
// ledger. This primitive acknowledges the current catalog and proves it idle.
// On failure, a non-nil lease means rollback is incomplete: retain it and the
// reservation until release succeeds. A nil lease means no cleanup remains.
func (b *Broker) drainNode(parent context.Context) (lease *nodeDrainLease, err error) {
	ctx, cancel := context.WithTimeout(parent, settings.SwitchDrainBudget)
	defer cancel()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !b.settingsApplyMu.TryLock() {
		return nil, fmt.Errorf("an engine configuration operation is active")
	}
	settingsHeld := true
	defer func() {
		if settingsHeld {
			b.settingsApplyMu.Unlock()
		}
	}()
	r := b.nodePolicyRuntime()
	if !r.acquirePush(ctx) {
		return nil, ctx.Err()
	}
	defer r.releasePush()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	pp := b.policyProxy()
	r.mu.Lock()
	runID := r.runID
	r.mu.Unlock()
	if err := b.nodeDrainProxyCurrent(pp, runID); err != nil {
		return nil, err
	}
	profiles := b.frontedEngineProfiles()
	owned := &nodeDrainLease{broker: b, settingsHeld: true}
	settingsHeld = false
	r.mu.Lock()
	for _, profile := range profiles {
		if !r.drains[profile.Name] {
			r.drains[profile.Name] = true
			owned.added = append(owned.added, profile.Name)
		}
	}
	r.mu.Unlock()
	defer func() {
		if err == nil {
			return
		}
		// Keep the push slot through rollback. Reacquiring it here would wait
		// on ourselves, or on replacement replay waiting for this acquisition.
		cleanup, cancelCleanup := context.WithTimeout(context.Background(), proxyCallTimeout)
		defer cancelCleanup()
		if cleanupErr := owned.releaseUnderPush(cleanup, pp); cleanupErr != nil {
			lease = owned
			err = errors.Join(err, fmt.Errorf("node drain rollback incomplete: %w", cleanupErr))
		}
	}()
	for _, profile := range profiles {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := b.nodeDrainProxyCurrent(pp, runID); err != nil {
			return nil, err
		}
		if err := switchDrainAck(ctx, pp, profile.Name, true); err != nil {
			return nil, err
		}
		if err := b.nodeDrainProxyCurrent(pp, runID); err != nil {
			return nil, err
		}
	}
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := b.nodeDrainProxyCurrent(pp, runID); err != nil {
			return nil, err
		}
		raw, rpcErr, err := pp.Call(ctx, nodepolicy.MethodAdmissionSnapshot, settingsJSON(nodepolicy.AdmissionSnapshotParams{}))
		if err != nil {
			return nil, err
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := b.nodeDrainProxyCurrent(pp, runID); err != nil {
			return nil, err
		}
		if rpcErr != nil {
			return nil, fmt.Errorf("admission snapshot refused: %s", rpcErr.Message)
		}
		idle, err := nodeDrainSnapshotIdle(raw, runID)
		if err != nil {
			return nil, err
		}
		if idle {
			return owned, nil
		}
		timer := time.NewTimer(100 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
}

// release uses a caller-owned cleanup context, independent of acquisition
// cancellation. Failed or uncertain undrains keep their registry entries and
// lease ownership for retry. A successful concurrent/repeated release is a no-op.
func (l *nodeDrainLease) release(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	r := l.broker.nodePolicyRuntime()
	if !r.acquirePush(ctx) {
		return ctx.Err()
	}
	defer r.releasePush()
	return l.releaseUnderPush(ctx, nil)
}

// acquiringProxy is non-nil only during rollback with the original push slot
// still held. No intervening replay or push could have sent our additions to
// an un-replayed replacement, so rollback can remove them before letting replay
// proceed. Ordinary release returns retryable failure for an unverified live
// proxy. With no live proxy, only registry cleanup remains.
func (l *nodeDrainLease) releaseUnderPush(ctx context.Context, acquiringProxy *proxyProcess) error {
	b := l.broker
	r := b.nodePolicyRuntime()
	for len(l.added) != 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		pp := b.policyProxy()
		r.mu.Lock()
		incarnation, runID := r.incarnation, r.runID
		r.mu.Unlock()
		if pp != nil && pp != incarnation && (acquiringProxy == nil || pp == acquiringProxy) {
			return fmt.Errorf("node drain cleanup proxy run is unverified")
		}
		if pp != nil && pp == incarnation {
			if err := b.nodeDrainProxyCurrent(pp, runID); err != nil {
				return err
			}
			if err := switchDrainAck(ctx, pp, l.added[0], false); err != nil {
				return fmt.Errorf("release drain for %s: %w", l.added[0], err)
			}
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := b.nodeDrainProxyCurrent(pp, runID); err != nil {
				return err
			}
		} else if b.policyProxy() != pp {
			return fmt.Errorf("proxy incarnation changed during node drain cleanup")
		}
		r.mu.Lock()
		delete(r.drains, l.added[0])
		r.mu.Unlock()
		l.added = l.added[1:]
	}
	if l.settingsHeld {
		l.settingsHeld = false
		b.settingsApplyMu.Unlock()
	}
	return nil
}

func (b *Broker) nodeDrainProxyCurrent(pp *proxyProcess, runID string) error {
	if pp == nil || b.policyProxy() != pp {
		return fmt.Errorf("proxy incarnation changed during node drain")
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	current := r.incarnation == pp && runID != "" && r.runID == runID
	r.mu.Unlock()
	if !current {
		return fmt.Errorf("node drain proxy run is unverified or changed")
	}
	return nil
}

func nodeDrainSnapshotIdle(raw json.RawMessage, runID string) (bool, error) {
	// Presence matters: missing/null counts are not evidence of zero. Pointer
	// map values also prevent a null per-engine count from decoding as zero.
	var snapshot struct {
		RunID          *string         `json:"runId"`
		Active         *int            `json:"active"`
		Queued         *int            `json:"queued"`
		ActiveByEngine map[string]*int `json:"activeByEngine"`
	}
	if err := json.Unmarshal(raw, &snapshot); err != nil {
		return false, fmt.Errorf("invalid node admission snapshot: %w", err)
	}
	if snapshot.RunID == nil || *snapshot.RunID != runID || snapshot.Active == nil || snapshot.Queued == nil || snapshot.ActiveByEngine == nil {
		return false, fmt.Errorf("missing or wrong-incarnation node admission evidence")
	}
	if *snapshot.Active < 0 || *snapshot.Queued < 0 {
		return false, fmt.Errorf("invalid node admission count")
	}
	idle := *snapshot.Active == 0 && *snapshot.Queued == 0
	for _, count := range snapshot.ActiveByEngine {
		if count == nil || *count < 0 {
			return false, fmt.Errorf("invalid engine admission count")
		}
		idle = idle && *count == 0
	}
	return idle, nil
}

// Unlike the settings drain, a switch has no timeout-and-stop path. The bound
// includes acquiring the push slot and every acknowledgement/snapshot request.
func (b *Broker) drainEngineForSwitch(parent context.Context, p settings.SwitchPrepareRequest) error {
	ctx, cancel := context.WithTimeout(parent, settings.SwitchDrainBudget)
	defer cancel()
	b.engineConfigMu.Lock()
	profile, ok := b.frontedEngineProfile(p.SourceEngine)
	if !ok {
		b.engineConfigMu.Unlock()
		return fmt.Errorf("source has no inference facade")
	}
	if b.regCache != nil {
		b.unregisterService(profile.DiscoveryService)
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.drains[p.SourceEngine] = true
	r.mu.Unlock()
	b.engineConfigMu.Unlock()
	pp := b.policyProxy()
	if pp == nil {
		return fmt.Errorf("proxy unavailable; drain is unverified")
	}
	if err := b.switchProxySetLocal(ctx, pp, p.SourceEngine, 0, false); err != nil {
		return err
	}
	if !r.acquirePush(ctx) {
		return ctx.Err()
	}
	defer r.releasePush()
	if err := switchDrainAck(ctx, pp, p.SourceEngine, true); err != nil {
		return err
	}
	runID, err := b.invalidateSwitchEngines(ctx, pp, p)
	if err != nil {
		return err
	}
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if pp != b.policyProxy() {
			return fmt.Errorf("proxy incarnation changed during drain")
		}
		var snapshot nodepolicy.AdmissionSnapshotResult
		raw, rpcErr, err := pp.Call(ctx, nodepolicy.MethodAdmissionSnapshot, settingsJSON(nodepolicy.AdmissionSnapshotParams{}))
		if err != nil {
			return err
		}
		if rpcErr != nil {
			return fmt.Errorf("admission snapshot refused: %s", rpcErr.Message)
		}
		if err := json.Unmarshal(raw, &snapshot); err != nil {
			return err
		}
		if snapshot.RunID != runID || snapshot.ActiveByEngine == nil || snapshot.Active < 0 {
			return fmt.Errorf("missing or wrong-incarnation admission evidence")
		}
		if snapshot.ActiveByEngine[p.SourceEngine] < 0 {
			return fmt.Errorf("invalid source admission count")
		}
		if snapshot.ActiveByEngine[p.SourceEngine] == 0 {
			return nil
		}
		timer := time.NewTimer(100 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
}

func switchDrainAck(ctx context.Context, pp *proxyProcess, engine string, drain bool) error {
	if pp == nil {
		return fmt.Errorf("proxy unavailable")
	}
	raw, rpcErr, err := pp.Call(ctx, nodepolicy.MethodSetEngineDrain, settingsJSON(nodepolicy.SetEngineDrainParams{Engine: engine, Drain: drain}))
	if err != nil {
		return err
	}
	if rpcErr != nil {
		return fmt.Errorf("drain refused: %s", rpcErr.Message)
	}
	var ack struct {
		OK bool `json:"ok"`
	}
	if json.Unmarshal(raw, &ack) != nil || !ack.OK {
		return fmt.Errorf("proxy did not acknowledge the engine drain")
	}
	return nil
}

func (b *Broker) invalidateSwitchEngines(ctx context.Context, pp *proxyProcess, p settings.SwitchPrepareRequest) (string, error) {
	runID := ""
	for _, engine := range []string{p.SourceEngine, p.TargetEngine} {
		raw, rpcErr, err := pp.Call(ctx, nodepolicy.MethodInvalidateEngine, settingsJSON(nodepolicy.InvalidateEngineParams{Engine: engine, OperationID: p.OperationID, Reason: "engine-switch"}))
		if err != nil {
			return "", err
		}
		if rpcErr != nil {
			return "", fmt.Errorf("invalidation refused: %s", rpcErr.Message)
		}
		var ack nodepolicy.InvalidateEngineResult
		if json.Unmarshal(raw, &ack) != nil || !ack.OK || ack.RunID == "" {
			return "", fmt.Errorf("engine invalidation was not acknowledged")
		}
		if runID != "" && ack.RunID != runID {
			return "", fmt.Errorf("proxy changed incarnation between invalidations")
		}
		runID = ack.RunID
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.incarnation != pp || r.runID == "" || r.runID != runID {
		return "", fmt.Errorf("invalidation does not match the active proxy run")
	}
	return runID, nil
}

func (b *Broker) releaseSwitchDrain(ctx context.Context, engine string) error {
	defer b.refreshBindingAdvertisement()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	draining := r.drains[engine]
	r.mu.Unlock()
	if !draining {
		return nil
	}
	if _, err := b.pushSwitchDrain(ctx, engine, false); err != nil {
		return err
	}
	r.mu.Lock()
	delete(r.drains, engine)
	r.mu.Unlock()
	return nil
}

func (b *Broker) switchProxySetLocal(ctx context.Context, pp *proxyProcess, engine string, port int, healthy bool) error {
	if pp == nil {
		return fmt.Errorf("proxy unavailable")
	}
	credential := ""
	if healthy && port > 0 {
		credential = b.localEngineCredential(engine, port)
	}
	raw, rpcErr, err := pp.Call(ctx, engines.AddressMethod(engine, "node/set-local-backend"), settingsJSON(proxyLocalBackend{Engine: engine, Host: "127.0.0.1", Port: port, Healthy: healthy, Credential: credential}))
	if err != nil {
		return err
	}
	if rpcErr != nil {
		return fmt.Errorf("local inference endpoint refused: %s", rpcErr.Message)
	}
	var ack struct {
		OK bool `json:"ok"`
	}
	if json.Unmarshal(raw, &ack) != nil || !ack.OK {
		return fmt.Errorf("local inference endpoint was not acknowledged")
	}
	if healthy {
		if profile, ok := b.frontedEngineProfile(engine); ok {
			b.engineProxy(profile).bindingRegistered.Store(false)
		}
	}
	return nil
}

// Withdrawal is acknowledged before any switch-owned lifecycle mutation,
// including conservative reconciliation after failure or restart.
func (b *Broker) withdrawSwitchIdentities(ctx context.Context, p settings.SwitchPrepareRequest) error {
	pp := b.policyProxy()
	for _, engine := range []string{p.SourceEngine, p.TargetEngine} {
		if err := b.withdrawSettingsIdentity(ctx, engine); err != nil {
			return err
		}
	}
	if b.policyProxy() != pp {
		return fmt.Errorf("proxy changed during identity withdrawal")
	}
	b.engineConfigMu.Lock()
	c := b.switchCoordinator()
	c.withdrawnIdentityID, c.withdrawnIdentityProxy = p.OperationID, pp
	b.engineConfigMu.Unlock()
	return nil
}

// Called under the policy push slot before facade/enable. A journal error or
// failed replay must abort proxy bring-up, never leave a briefly open facade.
func (b *Broker) replaySwitchInvalidations(ctx context.Context, pp *proxyProcess) error {
	b.engineConfigMu.Lock()
	if err := b.loadEngineSwitchLocked(); err != nil {
		b.engineConfigMu.Unlock()
		return err
	}
	var pending []settings.SwitchReceipt
	for _, receipt := range b.engineSwitch.store.journal.Receipts {
		if !receipt.Terminal || (receipt.RecoveryRequired && !b.engineSwitch.resolved[receipt.RequestID]) {
			pending = append(pending, receipt)
		}
	}
	b.engineConfigMu.Unlock()
	for _, receipt := range pending {
		if receipt.Phase == settings.SwitchAccepted || receipt.Phase == settings.SwitchValidating {
			continue
		}
		r := b.nodePolicyRuntime()
		r.mu.Lock()
		r.drains[receipt.Request.SourceEngine] = true
		r.mu.Unlock()
		if err := switchDrainAck(ctx, pp, receipt.Request.SourceEngine, true); err != nil {
			return err
		}
		if _, err := b.invalidateSwitchEngines(ctx, pp, settings.SwitchPrepareRequest{OperationID: receipt.RequestID, SourceEngine: receipt.Request.SourceEngine, TargetEngine: receipt.Request.TargetEngine}); err != nil {
			return err
		}
	}
	return nil
}
