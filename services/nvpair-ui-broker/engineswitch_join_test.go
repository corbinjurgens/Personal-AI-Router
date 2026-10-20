// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrusttest"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

func TestSwitchWaitsForIdentityWithdrawalBeforeLifecycle(t *testing.T) {
	h := newSwitchHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", true)
	baseProxy, baseEngine := h.proxy.answer, h.engine.answer
	var mu sync.Mutex
	withdrawn := map[string]bool{}
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetModelIdentity {
			var p nodepolicy.SetModelIdentityParams
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			mu.Lock()
			for _, engine := range []string{"llamacpp", "lmstudio"} {
				_, present := p.Engines[engine]
				withdrawn[engine] = !present
			}
			mu.Unlock()
		}
		return baseProxy(method, raw)
	})
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == settings.SwitchStepMethod {
			var p settings.SwitchStepRequest
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if p.Step == settings.SwitchStepSourceStop || p.Step == settings.SwitchStepTargetStart {
				mu.Lock()
				ready := withdrawn["llamacpp"] && withdrawn["lmstudio"]
				mu.Unlock()
				if !ready {
					return nil, "lifecycle preceded identity withdrawal acknowledgement"
				}
			}
		}
		return baseEngine(method, raw)
	})
	p := h.request(t, "identity-order")
	runtime := h.b.nodePolicyRuntime()
	runtime.mu.Lock()
	runtime.settingsIdentity = map[string]string{"llamacpp": modelDigestB, "lmstudio": modelDigestB}
	runtime.engineIdentity = map[string]engineManagerIdentity{
		"llamacpp": {EngineRunID: "source-run", LaunchDigest: launchDigestC, ModelDigests: map[string]string{"m": modelDigestB}},
		"lmstudio": {EngineRunID: "target-run", LaunchDigest: launchDigestC, ModelDigests: map[string]string{"m": modelDigestB}},
	}
	runtime.mu.Unlock()
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
		t.Fatal(err)
	}
	if r := h.receipt(t, p.RequestID); r.Phase != settings.SwitchCompleted {
		t.Fatalf("receipt = %+v", r)
	}
}

func TestSwitchIdentityWithdrawalFailurePreventsMutationAndCleanup(t *testing.T) {
	h := newSwitchHarness(t)
	base := h.proxy.answer
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetModelIdentity {
			return nil, "withdrawal refused"
		}
		return base(method, raw)
	})
	p := h.request(t, "identity-refused")
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	r := h.receipt(t, p.RequestID)
	if r.Phase != settings.SwitchFailedDrain || !r.RecoveryRequired {
		t.Fatalf("receipt = %+v", r)
	}
	if got := switchSteps(t, h.engine); len(got) != 0 {
		t.Fatalf("steps after refusal = %v", got)
	}
	if calls := h.engine.methods(settings.SwitchReconcileMethod); len(calls) != 0 {
		t.Fatal("cleanup touched engines despite refused withdrawal")
	}
}

func TestSwitchSettingsPublicationKeepsReservedIdentitiesWithdrawn(t *testing.T) {
	h := newSwitchHarness(t)
	p := h.request(t, "identity-refresh")
	h.active.Store(1)
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	// Keep the coordinator in drain while a periodic settings publication runs.
	h.b.engineConfigMu.Lock()
	h.b.engineSettings = map[string]*engineSettingsRecord{
		"llamacpp": {Snapshot: appliedSnapshot("llamacpp", 1)},
		"lmstudio": {Snapshot: appliedSnapshot("lmstudio", 1)},
	}
	h.b.publishSettingsLocked()
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	count := len(r.settingsIdentity)
	r.mu.Unlock()
	h.b.engineConfigMu.Unlock()
	h.active.Store(0)
	if count != 0 {
		t.Fatalf("settings refresh restored %d reserved identities", count)
	}
	if receipt := h.receipt(t, p.RequestID); receipt.Phase != settings.SwitchCompleted {
		t.Fatalf("receipt = %+v", receipt)
	}
	r.mu.Lock()
	count = len(r.settingsIdentity)
	r.mu.Unlock()
	if count != 2 {
		t.Fatalf("settlement did not re-observe unchanged settings: %d identities", count)
	}
}

func TestSwitchRoutabilityUsesDescriptorProtocolsAndFacadeReadiness(t *testing.T) {
	for _, tc := range []struct {
		name      string
		protocols []string
		ready     bool
		wantOK    bool
	}{
		{"OpenAI chat", []string{engines.ProtocolOpenAIChat}, true, true},
		{"OpenAI v1", []string{engines.ProtocolOpenAIV1}, true, true},
		{"Ollama", []string{engines.ProtocolOllamaAPI}, true, true},
		{"list only", []string{engines.ProtocolLlamaCPPRouter}, true, false},
		{"messages only", []string{engines.ProtocolAnthropicMessages}, true, false},
		{"unsupported", []string{"grpc"}, true, false},
		{"facade unavailable", []string{engines.ProtocolOpenAIChat}, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newSwitchHarness(t)
			d := testDescriptor("custom", tc.protocols...)
			h.b.setDynamicEngines([]engines.Descriptor{d}, func(int) bool { return true })
			h.b.setEngineProxyHandle(h.b.dynamicEngineProfiles()[0], h.pp)
			h.pp.readyMu.Lock()
			h.pp.facadeState["custom"] = proxyFacadeState{ready: tc.ready, port: 18080}
			h.pp.readyMu.Unlock()
			if err := h.b.switchTargetRoutable("custom"); (err == nil) != tc.wantOK {
				t.Fatalf("routability = %v, want allowed %v", err, tc.wantOK)
			}
		})
	}
}

func TestSwitchDynamicFacadeUsesCanonicalAddress(t *testing.T) {
	h := newSwitchHarness(t)
	if err := h.b.switchProxySetLocal(context.Background(), h.pp, "custom", 12345, true); err != nil {
		t.Fatal(err)
	}
	if calls := h.proxy.methods("e-custom:node/set-local-backend"); len(calls) != 1 {
		t.Fatalf("dynamic endpoint calls = %+v", h.proxy.snapshot())
	}
}

func TestSwitchCompletesWithDynamicSourceOrTarget(t *testing.T) {
	for _, engine := range []string{"source", "target"} {
		t.Run(engine, func(t *testing.T) {
			h := newSwitchHarness(t)
			h.b.setDynamicEngines([]engines.Descriptor{customEngineDescriptor()}, func(int) bool { return true })
			h.b.setEngineProxyHandle(h.b.dynamicEngineProfiles()[0], h.pp)
			h.pp.readyMu.Lock()
			h.pp.facadeState["llamacpp-custom"] = proxyFacadeState{ready: true, port: 18180}
			h.pp.readyMu.Unlock()
			h.mu.Lock()
			if engine == "source" {
				h.config.Engines[0].Engine = "llamacpp-custom"
			} else {
				h.config.Engines[1].Engine = "llamacpp-custom"
			}
			h.mu.Unlock()
			p := h.request(t, "dynamic-"+engine)
			if engine == "source" {
				p.SourceEngine = "llamacpp-custom"
			} else {
				p.TargetEngine = "llamacpp-custom"
			}
			if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err != nil {
				t.Fatal(err)
			}
			if r := h.receipt(t, p.RequestID); r.Phase != settings.SwitchCompleted {
				t.Fatalf("dynamic switch = %+v", r)
			}
			if calls := h.proxy.methods("e-llamacpp-custom:node/set-local-backend"); len(calls) != 1 {
				t.Fatalf("dynamic facade calls = %d, want 1", len(calls))
			}
		})
	}
}

func TestSwitchRecoveryWithdrawalRefusalRetainsReservation(t *testing.T) {
	h := newSwitchHarness(t)
	r := switchStoreReceipt("withdrawal-recovery")
	switchStoreAccept(t, h.b.engineSwitch.store, r)
	h.mu.Lock()
	h.cp = &settings.SwitchCheckpoint{
		SwitchPrepareRequest: settings.SwitchPrepareRequest{OperationID: r.RequestID, SourceEngine: r.Request.SourceEngine, TargetEngine: r.Request.TargetEngine},
		Phase:                settings.SwitchCheckpointStartPending,
	}
	h.mu.Unlock()
	base := h.proxy.answer
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetModelIdentity {
			return nil, "withdrawal refused"
		}
		return base(method, raw)
	})
	if h.b.recoverEngineSwitch() {
		t.Fatal("recovery succeeded without withdrawing strict identity")
	}
	if len(h.engine.methods(settings.SwitchReconcileMethod)) != 0 || len(switchSteps(t, h.engine)) != 0 {
		t.Fatal("recovery touched engines after withdrawal refusal")
	}
	h.b.engineConfigMu.Lock()
	err := h.b.pendingSwitchLocked(r.Request.TargetEngine)
	h.b.engineConfigMu.Unlock()
	if err == nil {
		t.Fatal("failed recovery released the target reservation")
	}
}

func TestSwitchLocalOffWithdrawalRefusalPreventsStop(t *testing.T) {
	h := newSwitchHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", true)
	p := h.request(t, "local-off-withdrawal")
	reached, resume := make(chan struct{}), make(chan struct{})
	var once sync.Once
	release := func() { once.Do(func() { close(resume) }) }
	t.Cleanup(release)
	baseEngine := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			var prepare settings.SwitchPrepareRequest
			if err := json.Unmarshal(raw, &prepare); err != nil {
				return nil, err.Error()
			}
			if prepare.OperationID != "" {
				close(reached)
				<-resume
			}
		}
		return baseEngine(method, raw)
	})
	base := h.proxy.answer
	var refuse atomic.Bool
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetModelIdentity && refuse.Load() {
			return nil, "withdrawal refused"
		}
		return base(method, raw)
	})
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
		t.Fatal(err)
	}
	select {
	case <-reached:
	case <-time.After(5 * time.Second):
		t.Fatal("switch did not reach prepare")
	}
	refuse.Store(true)
	id := json.RawMessage(`90`)
	h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:stop", Params: settingsJSON(map[string]string{"engine": "lmstudio"})})
	if len(h.engine.methods("engine:stop")) != 0 {
		t.Fatal("local Off touched the engine despite refused withdrawal")
	}
	var rejected bool
	for _, frame := range h.output.frames() {
		if frame.ID != nil && string(*frame.ID) == string(id) && frame.Error != nil {
			rejected = true
		}
	}
	if !rejected {
		t.Fatal("local Off withdrawal refusal was not returned to the client")
	}
	release()
	if r := h.receipt(t, p.RequestID); !r.RecoveryRequired {
		t.Fatalf("failed cleanup did not retain the durable Off: %+v", r)
	}
}
