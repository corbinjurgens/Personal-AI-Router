// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

var (
	launchDigestC = strings.Repeat("c", 64)
	modelDigestB  = strings.Repeat("b", 64)
)

// engineManagerSnapshot is engine:model-identity naming each engine with run
// id run-<engine> and one model m:latest.
func engineManagerSnapshot(engines ...string) json.RawMessage {
	p := struct {
		Engines map[string]engineManagerIdentity `json:"engines"`
	}{Engines: map[string]engineManagerIdentity{}}
	for _, e := range engines {
		p.Engines[e] = engineManagerIdentity{EngineRunID: "run-" + e, LaunchDigest: launchDigestC, ModelDigests: map[string]string{"m:latest": modelDigestB}}
	}
	data, _ := json.Marshal(p)
	return data
}

func appliedSnapshot(engine string, revision uint64) settings.Snapshot {
	return settings.Snapshot{Engine: engine, Epoch: "epoch-1", Revision: revision, AppliedRevision: revision, Phase: "succeeded",
		Settings: settings.Config{ServerPort: 11434, ProxyPort: 11435, LaunchText: "serve"}}
}

// identityCalls decodes every node/set-model-identity request in calls.
func identityCalls(t *testing.T, calls []policyCall) []nodepolicy.SetModelIdentityParams {
	t.Helper()
	var out []nodepolicy.SetModelIdentityParams
	for _, c := range calls {
		if c.method != nodepolicy.MethodSetModelIdentity {
			continue
		}
		var p nodepolicy.SetModelIdentityParams
		if err := json.Unmarshal(c.params, &p); err != nil {
			t.Fatalf("decode %s: %v", c.method, err)
		}
		if err := p.Validate(); err != nil {
			t.Fatalf("the broker sent an invalid identity %s: %v", c.params, err)
		}
		out = append(out, p)
	}
	return out
}

// waitIdentity waits until the newest identity the fake proxy received
// satisfies want, and returns it.
func waitIdentity(t *testing.T, proxy *fakeWorker, what string, want func(nodepolicy.SetModelIdentityParams) bool) nodepolicy.SetModelIdentityParams {
	t.Helper()
	var last nodepolicy.SetModelIdentityParams
	proxy.waitFor(what, func(calls []policyCall) bool {
		got := identityCalls(t, calls)
		if len(got) == 0 {
			return false
		}
		last = got[len(got)-1]
		return want(last)
	})
	return last
}

func TestModelIdentityJoinsEngineManagerAndSettings(t *testing.T) {
	ab := newActivatedBroker(t)
	b := ab.b
	if !b.handlePolicyEngineNotification(engineModelIdentityMethod, engineManagerSnapshot("ollama", "lmstudio")) {
		t.Fatal("engine:model-identity was forwarded instead of consumed")
	}
	b.observeSettingsIdentity([]settings.Snapshot{appliedSnapshot("ollama", 3)})
	fp, _ := settingsFingerprint(appliedSnapshot("ollama", 3))
	got := waitIdentity(t, ab.proxy, "the joined identity", func(p nodepolicy.SetModelIdentityParams) bool { return len(p.Engines) == 1 })
	want := nodepolicy.EngineIdentity{EngineRunID: "run-ollama", SettingsFingerprint: fp, LaunchDigest: launchDigestC, ModelDigests: map[string]string{"m:latest": modelDigestB}}
	if id := got.Engines["ollama"]; id.EngineRunID != want.EngineRunID || id.SettingsFingerprint != want.SettingsFingerprint ||
		id.LaunchDigest != want.LaunchDigest || id.ModelDigests["m:latest"] != modelDigestB {
		t.Fatalf("identity = %+v, want %+v (lmstudio has no applied settings)", got, want)
	}

	t.Run("a stop withdraws the engine", func(t *testing.T) {
		b.observeEngineRunning("ollama", false)
		waitIdentity(t, ab.proxy, "the stopped engine withdrawn", func(p nodepolicy.SetModelIdentityParams) bool { return len(p.Engines) == 0 })
	})
	t.Run("engine-manager exit withdraws every engine", func(t *testing.T) {
		b.observeModelIdentity(engineManagerSnapshot("ollama"))
		waitIdentity(t, ab.proxy, "the identity restored", func(p nodepolicy.SetModelIdentityParams) bool { return len(p.Engines) == 1 })
		b.clearEngineManagerIdentity()
		waitIdentity(t, ab.proxy, "every engine withdrawn", func(p nodepolicy.SetModelIdentityParams) bool { return len(p.Engines) == 0 })
	})
	t.Run("settings being applied withdraw the engine", func(t *testing.T) {
		b.observeModelIdentity(engineManagerSnapshot("ollama"))
		waitIdentity(t, ab.proxy, "the identity restored", func(p nodepolicy.SetModelIdentityParams) bool { return len(p.Engines) == 1 })
		applying := appliedSnapshot("ollama", 4)
		applying.Phase = "applying"
		b.observeSettingsIdentity([]settings.Snapshot{applying})
		waitIdentity(t, ab.proxy, "the engine withdrawn", func(p nodepolicy.SetModelIdentityParams) bool { return len(p.Engines) == 0 })
	})
	t.Run("a proxy restart is replayed the identity", func(t *testing.T) {
		b.observeSettingsIdentity([]settings.Snapshot{appliedSnapshot("ollama", 5)})
		pp, proxy, _ := attachAckingProxy(t, b)
		b.replayPolicyToProxy(context.Background(), pp)
		// A push scheduled before the restart may reach the new process too.
		if got := identityCalls(t, proxy.snapshot()); len(got) == 0 || len(got[len(got)-1].Engines) != 1 {
			t.Fatalf("replayed identities = %+v, want the current one", got)
		}
	})
}

func TestSettingsFingerprintCoversOnlyAppliedRevisions(t *testing.T) {
	base, ok := settingsFingerprint(appliedSnapshot("ollama", 3))
	if !ok || len(base) != 64 {
		t.Fatalf("fingerprint of an applied revision = %q, %v", base, ok)
	}
	for _, tc := range []struct {
		name string
		edit func(*settings.Snapshot)
	}{
		{"being applied", func(s *settings.Snapshot) { s.Phase = "applying" }},
		{"failed", func(s *settings.Snapshot) { s.Phase = "failed" }},
		{"accepted but not applied", func(s *settings.Snapshot) { s.AppliedRevision = 2 }},
		{"no epoch", func(s *settings.Snapshot) { s.Epoch = "" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := appliedSnapshot("ollama", 3)
			tc.edit(&s)
			if fp, ok := settingsFingerprint(s); ok {
				t.Fatalf("fingerprint = %q, want none", fp)
			}
		})
	}
	for _, tc := range []struct {
		name string
		edit func(*settings.Snapshot)
	}{
		{"another epoch", func(s *settings.Snapshot) { s.Epoch = "epoch-2" }},
		{"another revision", func(s *settings.Snapshot) { s.Revision, s.AppliedRevision = 4, 4 }},
		{"other launch text", func(s *settings.Snapshot) { s.Settings.LaunchText = "serve --ctx 2048" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := appliedSnapshot("ollama", 3)
			tc.edit(&s)
			if fp, ok := settingsFingerprint(s); !ok || fp == base {
				t.Fatalf("fingerprint = %q, %v; want a different one", fp, ok)
			}
		})
	}
	s := appliedSnapshot("ollama", 3)
	s.Sequence = 9
	if fp, _ := settingsFingerprint(s); fp != base {
		t.Fatal("a republication changed the fingerprint")
	}
}

func TestStrictReadinessWithdrawnWhenIdentityChanges(t *testing.T) {
	capability := strictCapability(t, "ollama", "m:latest")
	for _, tc := range []struct {
		name   string
		change func(*Broker)
	}{
		{"a new engine run", func(b *Broker) {
			r := b.nodePolicyRuntime()
			r.mu.Lock()
			id := r.engineIdentity["ollama"]
			id.EngineRunID = "engine-run-2"
			r.engineIdentity = map[string]engineManagerIdentity{"ollama": id}
			r.mu.Unlock()
			b.identityChanged()
		}},
		{"new settings", func(b *Broker) {
			b.observeSettingsIdentity([]settings.Snapshot{appliedSnapshot("ollama", 7)})
		}},
		{"a replaced model", func(b *Broker) {
			r := b.nodePolicyRuntime()
			r.mu.Lock()
			id := r.engineIdentity["ollama"]
			id.ModelDigests = map[string]string{"m:latest": strings.Repeat("e", 64)}
			r.engineIdentity = map[string]engineManagerIdentity{"ollama": id}
			r.mu.Unlock()
			b.identityChanged()
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newStrictFixture(t)
			f.advertise(t, capability)
			tc.change(f.b)
			waitWithdrawn(t, f)
			requireNotReady(t, f.b, "after the identity changed")
		})
	}
}

// replaceSettingsProxy publishes a fake proxy in place of the settings
// harness's, keeping its facade ports so previews still pass.
func replaceSettingsProxy(t *testing.T, h *settingsHarness, answer func(string, json.RawMessage) (any, string)) *fakeWorker {
	t.Helper()
	old := h.b.settingsProxy("ollama")
	old.readyMu.Lock()
	state := make(map[string]proxyFacadeState, len(old.facadeState))
	for k, v := range old.facadeState {
		state[k] = v
	}
	old.readyMu.Unlock()
	peer, f := newFakeWorkerPipe(t)
	f.setAnswer(answer)
	publishTestProxy(h.b, &proxyProcess{peer: peer, facadeState: state})
	return f
}

func TestSettingsApplyWithdrawsStrictIdentityFirst(t *testing.T) {
	h := newSettingsHarness(t)
	h.b.observeModelIdentity(engineManagerSnapshot("ollama"))
	p := h.request(t)
	identities := func() []nodepolicy.SetModelIdentityParams { return identityCalls(t, h.unaddressedProxyCalls()) }
	var before string
	deadline := time.Now().Add(3 * time.Second)
	for before == "" {
		if got := identities(); len(got) > 0 {
			before = got[len(got)-1].Engines["ollama"].SettingsFingerprint
		}
		if time.Now().After(deadline) {
			t.Fatal("the proxy never received the engine's identity")
		}
		time.Sleep(10 * time.Millisecond)
	}

	h.entered, h.release = make(chan struct{}), make(chan struct{})
	p.Settings.LaunchText += " --parallel 2"
	done := make(chan settingsReceipt, 1)
	go func() {
		receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
		if err != nil {
			t.Error(err)
		}
		done <- receipt
	}()
	<-h.entered
	// engine:configure-launch has arrived: the proxy has already
	// acknowledged an identity without the engine.
	got := identities()
	if _, present := got[len(got)-1].Engines["ollama"]; present {
		t.Fatalf("identity when the engine changes = %+v, want it withdrawn", got[len(got)-1])
	}
	close(h.release)
	if receipt := <-done; receipt.Phase != "succeeded" {
		t.Fatalf("receipt = %+v, want succeeded", receipt)
	}
	deadline = time.Now().Add(3 * time.Second)
	for {
		got := identities()
		if id, ok := got[len(got)-1].Engines["ollama"]; ok && id.SettingsFingerprint != before {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("no identity with the applied settings after the apply; last %+v", got[len(got)-1])
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestSettingsApplyFailsWithoutConfirmedWithdrawal(t *testing.T) {
	for _, tc := range []struct {
		name   string
		answer string
	}{
		{"the proxy refuses", "identity rejected"},
		{"the proxy does not answer", holdReply},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newSettingsHarness(t)
			setActivationWait(h.b, 200*time.Millisecond)
			p := h.request(t)
			replaceSettingsProxy(t, h, func(method string, _ json.RawMessage) (any, string) {
				if method == nodepolicy.MethodSetModelIdentity {
					return nil, tc.answer
				}
				return map[string]bool{"ok": true}, ""
			})
			p.Settings.LaunchText += " --parallel 2"
			receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
			if err != nil {
				t.Fatal(err)
			}
			if receipt.Phase != "failed" || h.applies.Load() != 0 {
				t.Fatalf("receipt = %+v with %d configure calls, want failed before the engine was touched", receipt, h.applies.Load())
			}
			snap, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "ollama"}, "")
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(snap.Error, "strict admission") {
				t.Fatalf("snapshot error = %q, want the withdrawal failure", snap.Error)
			}
		})
	}
}

func TestSettingsWithdrawalNeedsNoProxyWhenNoneRuns(t *testing.T) {
	b, _ := newPolicyBroker(t)
	if err := b.withdrawSettingsIdentity(context.Background(), "ollama"); err != nil {
		t.Fatalf("withdrawal with no proxy = %v, want nil", err)
	}
}
