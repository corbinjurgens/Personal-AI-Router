// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// settingsdrain.go coordinates an engine settings apply that restarts a
// running engine with the proxy (FORK_DESIGN.md §3.6): admission stops sending
// that engine new work, the apply waits for its running requests to finish,
// and the drain is cleared once the engine is back.

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

func engineDrainAck(ctx context.Context, pp *proxyProcess, engine string, drain bool) error {
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

func (b *Broker) proxySetLocalEndpoint(ctx context.Context, pp *proxyProcess, engine string, port int, healthy bool) error {
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
	return nil
}

// pushEngineDrain returns the actual acknowledgement and validates the live
// incarnation. Its caller's deadline covers push-slot contention as well.
func (b *Broker) pushEngineDrain(ctx context.Context, engine string, drain bool) (string, error) {
	r := b.nodePolicyRuntime()
	if !r.acquirePush(ctx) {
		return "", ctx.Err()
	}
	defer r.releasePush()
	pp := b.policyProxy()
	r.mu.Lock()
	runID, incarnation := r.runID, r.incarnation
	r.mu.Unlock()
	if pp == nil || pp != incarnation || runID == "" {
		return "", fmt.Errorf("proxy incarnation is unverified")
	}
	if err := engineDrainAck(ctx, pp, engine, drain); err != nil {
		return "", err
	}
	r.mu.Lock()
	current := r.incarnation == pp && r.runID == runID
	r.mu.Unlock()
	if !current || b.policyProxy() != pp {
		return "", fmt.Errorf("proxy incarnation changed during drain push")
	}
	return runID, nil
}

// Typed settings retain this fence after every uncertain or failed operation.
// Only a successful operation releases it. A timeout may interrupt active work
// only after the live proxy acknowledged the drain; a failed ACK is a failure.
func (b *Broker) drainModelSettings(ctx context.Context, engine string, wait bool) error {
	profile, ok := engineProxyProfileFor(engine)
	if !ok {
		return fmt.Errorf("engine has no inference facade")
	}
	if b.regCache != nil {
		b.unregisterService(profile.DiscoveryService)
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.drains[engine] = true
	r.mu.Unlock()
	pp := b.policyProxy()
	if err := b.proxySetLocalEndpoint(ctx, pp, engine, 0, false); err != nil {
		return err
	}
	run, err := b.pushEngineDrain(ctx, engine, true)
	if err != nil {
		return err
	}
	if err := b.withdrawSettingsIdentity(ctx, engine); err != nil {
		return err
	}
	if wait {
		var snapshot nodepolicy.AdmissionSnapshotResult
		raw, rpcErr, err := pp.Call(ctx, nodepolicy.MethodAdmissionSnapshot, settingsJSON(nodepolicy.AdmissionSnapshotParams{}))
		if err != nil {
			return err
		}
		if rpcErr != nil || json.Unmarshal(raw, &snapshot) != nil || snapshot.RunID != run || snapshot.ActiveByEngine == nil || snapshot.ActiveByEngine[engine] < 0 {
			return fmt.Errorf("current admission readback is unavailable after drain")
		}
		if b.policyProxy() != pp {
			return fmt.Errorf("proxy changed during admission readback")
		}
		b.observeAdmission(snapshot.AdmissionState)
		b.waitAdmission(ctx, settingsDrainTimeout, func(st nodepolicy.AdmissionState) bool {
			return st.ActiveByEngine[engine] == 0
		})
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	r.mu.Lock()
	current := r.incarnation == pp && r.runID == run
	r.mu.Unlock()
	if !current || b.policyProxy() != pp {
		return fmt.Errorf("proxy changed after drain acknowledgement")
	}
	return nil
}

// settingsDrainTimeout bounds how long an apply waits for the engine's running
// requests. After it the restart goes ahead and those requests fail, exactly as
// they did before the drain existed.
var settingsDrainTimeout = 60 * time.Second

// drainEngineForSettingsLocked marks engine draining, waits for its active
// count to reach zero, and returns the function that clears the drain. The
// caller holds settingsApplyMu and engineConfigMu; engineConfigMu is released
// while waiting so other settings readers are not frozen for up to a minute,
// which settingsApplyMu makes safe (no other apply can start meanwhile).
func (b *Broker) drainEngineForSettingsLocked(ctx context.Context, engine string) (release func()) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.drains[engine] = true
	r.mu.Unlock()
	kind := pushDrainPrefix + engine

	b.engineConfigMu.Unlock()
	b.pushPolicyNow(ctx, kind)
	drained := b.waitAdmission(ctx, settingsDrainTimeout, func(st nodepolicy.AdmissionState) bool {
		return st.ActiveByEngine[engine] == 0
	})
	b.engineConfigMu.Lock()
	if !drained {
		slog.Warn("engine still had running requests when its settings restart began", "engine", engine)
	}
	return func() {
		r.mu.Lock()
		delete(r.drains, engine)
		r.mu.Unlock()
		b.schedulePolicyPush(kind)
	}
}
