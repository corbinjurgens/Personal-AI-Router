// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

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
