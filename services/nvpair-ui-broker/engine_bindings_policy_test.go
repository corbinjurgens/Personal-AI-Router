// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

func TestEngineBindingsPolicySaveRestartWithoutLifecycleEffects(t *testing.T) {
	ab := newActivatedBroker(t)
	waitPushesSettled(t, ab.b)
	recorder := attachFakeEngine(t, ab.b)
	p := ab.b.localPolicy().Policy
	p.EngineBindings = nodepolicy.EngineBindings{{Engine: "ollama", Model: "Native/Model"}}
	raw, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	saved, err := ab.b.setLocalPolicy(raw, mustJSON(p.Revision))
	if err != nil {
		t.Fatal(err)
	}
	waitPushesSettled(t, ab.b)
	if !reflect.DeepEqual(saved.EngineBindings, p.EngineBindings) {
		t.Fatalf("saved bindings=%+v", saved.EngineBindings)
	}
	r := ab.b.nodePolicyRuntime()
	r.mu.Lock()
	intent := len(r.intent)
	r.mu.Unlock()
	if intent != 0 {
		t.Fatal("binding save changed engine intent")
	}
	// The fake records RPCs separately from policy projection notifications.
	for _, call := range recorder.snapshot() {
		if call.method != "engine:get-installed" {
			t.Fatalf("binding-only save called engine lifecycle: %s", call.method)
		}
	}
	restarted, _ := newPolicyBroker(t)
	restarted.loadNodePolicy(ab.path)
	got := restarted.localPolicy()
	if got.RecoveryRequired || !reflect.DeepEqual(got.Policy.EngineBindings, p.EngineBindings) {
		t.Fatalf("restart=%+v", got)
	}
	saved.EngineBindings[0].Model = "caller mutation"
	readback := ab.b.localPolicy()
	readback.Policy.EngineBindings[0].Model = "readback mutation"
	if ab.b.localPolicy().Policy.EngineBindings[0].Model != "Native/Model" {
		t.Fatal("caller owns stored binding slice")
	}
}

func TestEngineBindingsPolicyRejectsTargetOverlapBeforePersistence(t *testing.T) {
	test := func(name string, bindings nodepolicy.EngineBindings, served nodepolicy.ServedModels) {
		t.Run(name, func(t *testing.T) {
			ab := newActivatedBroker(t)
			d := testDescriptor("custom-tagged", engines.ProtocolOpenAIV1)
			d.ModelNaming = engines.ModelNamingImpliedLatestTag
			ab.b.setDynamicEngines([]engines.Descriptor{d}, func(int) bool { return true })
			p := ab.b.localPolicy().Policy
			p.EngineBindings, p.ServedModels = bindings, served
			if _, err := ab.b.setLocalPolicy(mustJSON(p), mustJSON(p.Revision)); err == nil {
				t.Fatal("invalid binding saved")
			}
			if ab.b.localPolicy().Policy.Revision != p.Revision {
				t.Fatal("rejected binding advanced revision")
			}
			if _, err := os.Stat(ab.path); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("rejected save created policy file: %v", err)
			}
		})
	}
	test("builtin overlap", nodepolicy.EngineBindings{{Engine: "ollama", Model: "m:latest"}, {Engine: "llamacpp", Model: "m"}}, nil)
	test("descriptor overlap", nodepolicy.EngineBindings{{Engine: "custom-tagged", Model: "m:latest"}, {Engine: "llamacpp", Model: "m"}}, nil)
	test("deny all", nodepolicy.EngineBindings{{Engine: "ollama", Model: "m"}}, nodepolicy.ServedModels{})
}

func TestEngineBindingsPolicyPeerPermissionCASAndPrivateSections(t *testing.T) {
	rb := newRoutedBroker(t)
	waitPushesSettled(t, rb.b)
	stored := rb.b.localPolicy().Policy
	p := stored.PeerProjection()
	p.EngineBindings = nodepolicy.EngineBindings{{Engine: "ollama", Model: "m:latest"}}
	p.ServedModels = nodepolicy.ServedModels{{Engine: "ollama", Model: "m"}}
	request := policyRelayRequest{ID: "bindings", Method: "set", Caller: "pinned-peer", Params: rmJSON(t, policyParams{Policy: rmJSON(t, p), ExpectedPersistedRevision: rmJSON(t, stored.Revision)})}
	grantRemoteManagement(t, rb.b.clusterDir, "cluster-1", "self-node", "pinned-peer", false)
	_, err := rb.b.servePolicyRelay(context.Background(), request)
	rmDeny(t, err)
	if rb.b.localPolicy().Policy.Revision != stored.Revision {
		t.Fatal("denied edit persisted")
	}
	grantRemoteManagement(t, rb.b.clusterDir, "cluster-1", "self-node", "pinned-peer", true)
	if _, err := rb.b.servePolicyRelay(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	got := rb.b.localPolicy().Policy
	if !reflect.DeepEqual(got.EngineBindings, p.EngineBindings) || !reflect.DeepEqual(got.Routing, stored.Routing) || !reflect.DeepEqual(got.Tiers, stored.Tiers) || got.TierPolicy != stored.TierPolicy || got.Availability != stored.Availability || got.PausedBy != stored.PausedBy {
		t.Fatal("peer edit lost bindings or changed protected policy")
	}
	_, err = rb.b.servePolicyRelay(context.Background(), request)
	if policyErrorCode(err) != codePolicyConflict {
		t.Fatalf("stale edit=%v", err)
	}
}

func TestEngineBindingsPolicyMalformedDiskRequiresPausedRecovery(t *testing.T) {
	for _, tc := range []struct{ name, raw string }{
		{"null list", `{"engineBindings":null}`},
		{"control token", `{"engineBindings":[{"engine":"ollama","model":"bad\u0000"}]}`},
		{"invalid UTF8", "{\"engineBindings\":[{\"engine\":\"ollama\",\"model\":\"\xff\"}]}"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := tc.raw
			path := filepath.Join(t.TempDir(), nodepolicy.FileName)
			if err := os.WriteFile(path, []byte(raw), 0600); err != nil {
				t.Fatal(err)
			}
			b, _ := newPolicyBroker(t)
			b.loadNodePolicy(path)
			got := b.localPolicy()
			if !got.RecoveryRequired || got.Availability != nodepolicy.Draining || got.Policy.Availability != nodepolicy.Paused || len(got.Policy.EngineBindings) != 0 {
				t.Fatalf("recovery=%+v", got)
			}
			for _, file := range []string{path, path + ".bad"} {
				data, err := os.ReadFile(file)
				if err != nil {
					t.Fatal(err)
				}
				if string(data) != raw {
					t.Fatal("invalid evidence was rewritten")
				}
			}
		})
	}
}

func TestEngineBindingsPolicyReadbackSupport(t *testing.T) {
	for _, tc := range []struct {
		name, request, result string
		refused               bool
	}{
		{"dropped binding", `{"engineBindings":[{"engine":"ollama","model":"m"}]}`, `{"policy":{}}`, true},
		{"dropped explicit clearing", `{"engineBindings":[]}`, `{"policy":{}}`, true},
		{"null clearing", `{"engineBindings":[]}`, `{"policy":{"engineBindings":null}}`, true},
		{"preserved clearing", `{"engineBindings":[]}`, `{"policy":{"engineBindings":[]}}`, false},
		{"preserved binding", `{"engineBindings":[{"engine":"ollama","model":"m"}]}`, `{"policy":{"engineBindings":[{"engine":"ollama","model":"m"}]}}`, false},
		{"old absent input", `{}`, `{"policy":{}}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := modelControlReadbackSupported(json.RawMessage(tc.request), json.RawMessage(tc.result)); (err != nil) != tc.refused {
				t.Fatalf("refused=%v: %v", tc.refused, err)
			}
		})
	}
}
