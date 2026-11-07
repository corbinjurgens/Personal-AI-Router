// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"testing"

	"nvpair-shared/nodepolicy"
)

func TestClientLifecycleControlsNeverReachManager(t *testing.T) {
	for _, prefix := range []string{"", "llamacpp-proxy:", "nvpair-engine-manager:"} {
		for _, method := range []string{
			"engine:start", "engine:stop", "engine:restart", "engine:wake", "engine:sleep",
			"engine:remote-start", "engine:remote-stop",
		} {
			t.Run(prefix+method, func(t *testing.T) {
				b, out := newPolicyBroker(t)
				engine := attachFakeEngine(t, b)
				b.handleMessage(clientRequest(1, prefix+method, map[string]string{"engine": "llamacpp", "nodeId": "peer"}))
				response := out.waitFrame(t, "lifecycle refusal", isResponse(1))
				if response.Error == nil || response.Error.Code != -32601 || len(response.Result) != 0 {
					t.Fatalf("lifecycle response = %+v, want method-not-found", response)
				}
				notification := clientRequest(2, prefix+method, map[string]string{"engine": "llamacpp"})
				notification.ID = nil
				b.handleMessage(notification)
				if calls := engine.snapshot(); len(calls) != 0 {
					t.Fatalf("client lifecycle reached manager: %v", calls)
				}
			})
		}
	}
}

func TestInstallUsesOneExecutorInstallPath(t *testing.T) {
	b, out := newPolicyBroker(t)
	engine := attachFakeEngine(t, b)
	params := map[string]string{"engine": "llamacpp"}
	b.relayToEngineNow(clientRequest(1, "engine:install", params))
	response := out.waitFrame(t, "install response", isResponse(1))
	if response.Error != nil {
		t.Fatalf("install response: %+v", response.Error)
	}
	calls := engine.snapshot()
	if len(calls) != 1 {
		t.Fatalf("install calls = %v, want one executor install", calls)
	}
	for i, call := range calls {
		var request struct {
			Engine string `json:"engine"`
		}
		if err := json.Unmarshal(call.params, &request); err != nil {
			t.Fatalf("decode install %d: %v", i, err)
		}
		if call.method != "engine:install" || request.Engine != "llamacpp" {
			t.Fatalf("install %d = %s %+v", i, call.method, request)
		}
	}
}

func TestPauseDuringInstallHasNoBrokerStartContinuation(t *testing.T) {
	b, out := newPolicyBroker(t)
	engine := attachFakeEngine(t, b)
	engine.mu.Lock()
	answer := engine.answer
	engine.mu.Unlock()
	engine.setAnswer(func(method string, params json.RawMessage) (any, string) {
		if method == "engine:install" {
			return map[string]bool{"ok": true}, holdSuccessReply
		}
		return answer(method, params)
	})
	done := make(chan struct{})
	go func() {
		defer close(done)
		b.relayToEngineNow(clientRequest(1, "engine:install", map[string]string{"engine": "llamacpp"}))
	}()
	t.Cleanup(func() { engine.releaseHeld(); requireGateOpens(t, done, "install cleanup") })
	engine.waitFor("install preparation", func(calls []policyCall) bool { return len(calls) == 1 })
	if _, err := b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil {
		t.Fatalf("pause during preparation: %v", err)
	}
	engine.releaseHeld()
	requireGateOpens(t, done, "superseded install")
	response := out.waitFrame(t, "manager install completion", isResponse(1))
	if response.Error != nil {
		t.Fatalf("manager completion was replaced: %+v", response)
	}
	if calls := engine.methods("engine:install"); len(calls) != 1 {
		t.Fatalf("install continued after pause: %v", calls)
	}
}
