// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"nvpair-shared/durablefile"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

func profileContextLoads() map[string][]modelconfig.ModelLoad {
	return map[string][]modelconfig.ModelLoad{
		"llamacpp": {
			{ProfileKey: "small-profile", ModelID: "shared.gguf", Requested: modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "fixed", Tokens: 4096}}, Resolved: &modelconfig.ResolvedLoad{PerSlotContext: 8192, TotalContext: 32768, Slots: 4}},
			{ProfileKey: "large-profile", ModelID: "shared.gguf", Resolved: &modelconfig.ResolvedLoad{PerSlotContext: 32768, TotalContext: 65536, Slots: 2}},
		},
	}
}

func profileContextPolicy() nodepolicy.Policy {
	p := nodepolicy.Default()
	p.Profiles = []nodepolicy.Profile{
		{Key: "small-profile", DisplayName: "Small", Engine: "llamacpp", Model: "small-profile", ContextTokens: 1024,
			ModelIdentity: &engines.ModelIdentity{Strength: "declared", GeneralName: "Stored model", Files: []engines.ModelIdentityFile{{Role: "model", Name: "shared.gguf", SizeBytes: 256}}}},
		{Key: "large-profile", DisplayName: "Large", Engine: "llamacpp", Model: "large-profile", ContextTokens: 2048},
		{Key: "descriptor-profile", DisplayName: "Descriptor", Engine: "strata", Model: "descriptor-native", ContextTokens: 65536},
	}
	return p
}

func TestDeriveProfileContextSiblingContexts(t *testing.T) {
	p := profileContextPolicy()
	p.Pause.UnloadModels, p.Pause.StopEngines = false, false
	loads := profileContextLoads()
	// Neither a matching artifact id nor a key on another engine is a match.
	loads["strata"] = []modelconfig.ModelLoad{{ProfileKey: "other-profile", ModelID: "descriptor-profile", Resolved: &modelconfig.ResolvedLoad{PerSlotContext: 17}}}
	loads["other-engine"] = []modelconfig.ModelLoad{{ProfileKey: "descriptor-profile", Resolved: &modelconfig.ResolvedLoad{PerSlotContext: 19}}}
	got, err := deriveProfileContext(p, loads)
	if err != nil {
		t.Fatalf("derive: %v", err)
	}
	if got.Profiles[0].ContextTokens != 8192 || got.Profiles[1].ContextTokens != 32768 || got.Profiles[2].ContextTokens != 65536 {
		t.Fatalf("derived contexts = %d, %d, %d; want 8192, 32768, 65536", got.Profiles[0].ContextTokens, got.Profiles[1].ContextTokens, got.Profiles[2].ContextTokens)
	}
	if p.Profiles[0].ContextTokens != 1024 || p.Profiles[1].ContextTokens != 2048 {
		t.Fatal("derivation changed the source policy")
	}
	if got.Pause != p.Pause {
		t.Fatal("context derivation changed unrelated pause fields")
	}
}

func TestDeriveProfileContextDetached(t *testing.T) {
	p := profileContextPolicy()
	p.Profiles[0].Capabilities = []nodepolicy.Capability{nodepolicy.CapTools}
	p.Profiles[0].RequestOptions = json.RawMessage(`{"temperature":0.1}`)
	p.Profiles[0].Qualification = &nodepolicy.QualificationRef{ID: "fixture", Revision: "1"}
	p.Roles = []nodepolicy.Role{{Name: "weak", Profiles: []string{"small-profile"}}}
	routed, _, err := nodepolicy.ParseStored([]byte(routedPolicyDocument))
	if err != nil {
		t.Fatalf("decode API fixture: %v", err)
	}
	p.Tiers, p.Routing = routed.Tiers, routed.Routing
	before := rmJSON(t, p)
	got, err := deriveProfileContext(p, profileContextLoads())
	if err != nil {
		t.Fatal(err)
	}
	got.Profiles[0].Capabilities[0] = nodepolicy.CapEmbeddings
	got.Profiles[0].RequestOptions[2] = 'X'
	got.Profiles[0].Qualification.ID = "changed"
	got.Profiles[0].ModelIdentity.Files[0].Name = "changed.gguf"
	got.Roles[0].Profiles[0] = "changed"
	got.Tiers["weak"][0].Deployment = "changed"
	got.Routing.Tiers["detached-only"] = nodepolicy.TierRouting{}
	if !reflect.DeepEqual(rmJSON(t, p), before) {
		t.Fatal("mutating derived policy changed its source")
	}
}

func TestDeriveProfileContextRefusesUnresolvedTypedLoad(t *testing.T) {
	test := func(name string, resolved *modelconfig.ResolvedLoad) {
		t.Run(name, func(t *testing.T) {
			p := profileContextPolicy()
			loads := profileContextLoads()
			loads["llamacpp"][0].Resolved = resolved
			before := rmJSON(t, p)
			_, err := deriveProfileContext(p, loads)
			if err == nil || err.Error() != `profile "small-profile" has no resolved per-slot context` {
				t.Fatalf("unresolved derive = %v", err)
			}
			if !reflect.DeepEqual(rmJSON(t, p), before) {
				t.Fatal("failed derivation changed the source")
			}
		})
	}
	test("requested context without resolution", nil)
	test("total context without per-slot context", &modelconfig.ResolvedLoad{TotalContext: 32768, Slots: 4})
	test("negative per-slot context", &modelconfig.ResolvedLoad{PerSlotContext: -1, TotalContext: 32768, Slots: 4})
}

func TestProfileLoadsSnapshotDetached(t *testing.T) {
	b, _ := newPolicyBroker(t)
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	b.engineSettings = map[string]*engineSettingsRecord{"llamacpp": {Snapshot: settings.Snapshot{Settings: settings.Config{ModelLoads: profileContextLoads()["llamacpp"]}}}}
	got := b.profileLoadsByEngineLocked()
	got["llamacpp"][0].Requested.Context.Tokens = 7
	got["llamacpp"][0].Resolved.PerSlotContext = 9
	delete(got, "llamacpp")
	source := b.engineSettings["llamacpp"].Snapshot.Settings.ModelLoads[0]
	if source.Requested.Context.Tokens != 4096 || source.Resolved.PerSlotContext != 8192 {
		t.Fatalf("snapshot alias changed source = %+v", source)
	}
}

func newRawProfileBroker(t *testing.T) *routedBroker {
	t.Helper()
	rb := newBrokerOnFile(t, string(rmJSON(t, profileContextPolicy())))
	strata := testDescriptor("strata", engines.ProtocolOpenAIChat)
	strata.Family = "strata"
	rb.b.setDynamicEngines([]engines.Descriptor{customEngineDescriptor(), strata}, func(int) bool { return true })
	if got := rb.b.localPolicy().Policy; len(got.Profiles) != 3 || got.Profiles[0].ModelIdentity == nil {
		t.Fatal("profile fixture did not survive stored decoding")
	}
	rb.b.engineConfigMu.Lock()
	rb.b.engineSettingsLoaded = true
	rb.b.engineSettings = map[string]*engineSettingsRecord{"llamacpp": {Snapshot: settings.Snapshot{
		Engine: "llamacpp", Revision: 1, AppliedRevision: 1, Phase: "idle",
		Settings: settings.Config{ModelLoads: []modelconfig.ModelLoad{rawSnapshotLoad("small-profile", 8192), rawSnapshotLoad("large-profile", 32768)}},
	}}}
	rb.b.engineConfigMu.Unlock()
	return rb
}

func rawProfileDraft(t *testing.T, b *Broker, peer bool) (map[string]json.RawMessage, []map[string]json.RawMessage) {
	t.Helper()
	var raw json.RawMessage
	if peer {
		raw = rmJSON(t, b.peerPolicy().Policy)
	} else {
		raw = rmJSON(t, b.localPolicy().Policy)
	}
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	var profiles []map[string]json.RawMessage
	if err := json.Unmarshal(doc["profiles"], &profiles); err != nil {
		t.Fatal(err)
	}
	return doc, profiles
}

func writeRawProfileDraft(t *testing.T, b *Broker, peer bool, doc map[string]json.RawMessage, profiles []map[string]json.RawMessage) (nodepolicy.Policy, error) {
	t.Helper()
	doc["profiles"] = rmJSON(t, profiles)
	if peer {
		return b.setPeerPolicyAs(rmJSON(t, doc), doc["revision"], "pinned-peer")
	}
	return b.setLocalPolicy(rmJSON(t, doc), doc["revision"])
}

func TestRawPolicyProfileContextNormalization(t *testing.T) {
	test := func(name string, peer bool, context json.RawMessage) {
		t.Run(name, func(t *testing.T) {
			rb := newRawProfileBroker(t)
			doc, profiles := rawProfileDraft(t, rb.b, peer)
			profiles[0]["contextTokens"] = context
			profiles[1]["contextTokens"] = json.RawMessage(`999999`)
			profiles[2]["contextTokens"] = json.RawMessage(`49152`)
			got, err := writeRawProfileDraft(t, rb.b, peer, doc, profiles)
			if err != nil {
				t.Fatalf("raw set: %v", err)
			}
			if got.Profiles[0].ContextTokens != 8192 || got.Profiles[1].ContextTokens != 32768 || got.Profiles[2].ContextTokens != 49152 {
				t.Fatalf("set contexts = %+v", got.Profiles)
			}
			var saved nodepolicy.Policy
			if err := json.Unmarshal(rb.fileBytes(t), &saved); err != nil {
				t.Fatal(err)
			}
			if saved.Profiles[0].ContextTokens != 8192 || saved.Profiles[1].ContextTokens != 32768 || saved.Profiles[2].ContextTokens != 49152 || saved.Revision != got.Revision {
				t.Fatalf("saved contexts/revision = %+v", saved)
			}
			if readback := rb.b.localPolicy().Policy; !reflect.DeepEqual(readback.Profiles, saved.Profiles) {
				t.Fatal("readback differs from saved profile contexts")
			}
		})
	}
	test("local negative context", false, json.RawMessage(`-1`))
	test("peer negative context", true, json.RawMessage(`-1`))
	test("local malformed managed context", false, json.RawMessage(`"caller-value"`))
	test("peer malformed managed context", true, json.RawMessage(`"caller-value"`))
}

func TestRawPolicyProfileIdentityProtected(t *testing.T) {
	test := func(name string, peer bool) {
		t.Run(name, func(t *testing.T) {
			rb := newRawProfileBroker(t)
			stored := rb.b.localPolicy().Policy
			wantIdentity := stored.Profiles[0].ModelIdentity
			doc, profiles := rawProfileDraft(t, rb.b, peer)
			// All casing is managed. Invalid identities cannot fail validation or
			// supply assertions for a new key, even when they are not objects.
			delete(profiles[0], "modelIdentity")
			profiles[0]["ModelIdentity"] = json.RawMessage(`{"strength":"forged","files":"invalid"}`)
			profiles[1]["modelIdentity"] = json.RawMessage(`42`)
			profiles = append(profiles, map[string]json.RawMessage{
				"key": rmJSON(t, "new-profile"), "name": rmJSON(t, "new"), "engine": rmJSON(t, "strata"), "model": rmJSON(t, "another-native"),
				"modelIdentity": json.RawMessage(`{"strength":"verified","contentId":"forged"}`),
			})
			got, err := writeRawProfileDraft(t, rb.b, peer, doc, profiles)
			if err != nil {
				t.Fatalf("raw identity set: %v", err)
			}
			if !reflect.DeepEqual(got.Profiles[0].ModelIdentity, wantIdentity) || got.Profiles[1].ModelIdentity != nil || got.Profiles[3].ModelIdentity != nil {
				t.Fatalf("stored identity not protected: %+v", got.Profiles)
			}
			var saved nodepolicy.Policy
			if err := json.Unmarshal(rb.fileBytes(t), &saved); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(saved.Profiles, got.Profiles) {
				t.Fatal("saved identities differ from returned policy")
			}
			got.Profiles[0].ModelIdentity.Files[0].Name = "changed.gguf"
			if stored.Profiles[0].ModelIdentity.Files[0].Name != "shared.gguf" {
				t.Fatal("published identity aliases the previous stored policy")
			}
		})
	}
	test("local", false)
	test("peer", true)
}

func TestRawPolicyProfileRetargetRefusedWithoutEffects(t *testing.T) {
	test := func(name string, peer bool, index int, field, target string) {
		t.Run(name, func(t *testing.T) {
			rb := newRawProfileBroker(t)
			doc, profiles := rawProfileDraft(t, rb.b, peer)
			profiles[index][field] = rmJSON(t, target)
			doc["idle"] = rmJSON(t, nodepolicy.IdleConfig{UnloadAfterMinutes: 13})
			before := rb.baseline(t)
			_, err := writeRawProfileDraft(t, rb.b, peer, doc, profiles)
			if err == nil || err.Error() != "A profile keeps its model; create a new profile." {
				t.Fatalf("retarget refusal = %v", err)
			}
			if peer && policyErrorCode(err) != -32602 {
				t.Fatalf("peer retarget code = %d, want -32602", policyErrorCode(err))
			}
			before.requireUnchanged(t)
		})
	}
	test("local typed model", false, 0, "model", "other-model")
	test("peer typed model", true, 0, "model", "other-model")
	test("local typed engine", false, 0, "engine", "llamacpp-custom")
	test("peer typed engine", true, 0, "engine", "llamacpp-custom")
	test("local descriptor engine", false, 2, "engine", "llamacpp-custom")
	test("peer descriptor engine", true, 2, "engine", "llamacpp-custom")
}

func TestRawPolicyDescriptorModelCanChange(t *testing.T) {
	test := func(name string, peer bool) {
		t.Run(name, func(t *testing.T) {
			rb := newRawProfileBroker(t)
			doc, profiles := rawProfileDraft(t, rb.b, peer)
			profiles[2]["model"] = rmJSON(t, "changed-native")
			got, err := writeRawProfileDraft(t, rb.b, peer, doc, profiles)
			if err != nil {
				t.Fatal(err)
			}
			if got.Profiles[2].Model != "changed-native" || got.Profiles[2].ContextTokens != 65536 {
				t.Fatalf("descriptor edit = %+v", got.Profiles[2])
			}
		})
	}
	test("local", false)
	test("peer", true)
}

func TestRawPolicyUnresolvedProfileRefusesWithoutEffects(t *testing.T) {
	test := func(name string, peer bool) {
		t.Run(name, func(t *testing.T) {
			rb := newRawProfileBroker(t)
			rb.b.engineConfigMu.Lock()
			rb.b.engineSettings["llamacpp"].Snapshot.Settings.ModelLoads[0].Resolved = nil
			rb.b.engineConfigMu.Unlock()
			doc, profiles := rawProfileDraft(t, rb.b, peer)
			profiles[0]["contextTokens"] = json.RawMessage(`8192`)
			before := rb.baseline(t)
			_, err := writeRawProfileDraft(t, rb.b, peer, doc, profiles)
			if err == nil || !strings.Contains(err.Error(), "no resolved per-slot context") {
				t.Fatalf("unresolved raw set = %v", err)
			}
			before.requireUnchanged(t)
		})
	}
	test("local", false)
	test("peer", true)
}

func TestRawPolicyProfileRevisionBeforeNormalization(t *testing.T) {
	test := func(name string, peer bool) {
		t.Run(name, func(t *testing.T) {
			rb := newRawProfileBroker(t)
			doc, profiles := rawProfileDraft(t, rb.b, peer)
			doc["revision"] = rmJSON(t, "999")
			profiles[0]["model"] = rmJSON(t, "other-model")
			rb.b.engineConfigMu.Lock()
			rb.b.engineSettings["llamacpp"].Snapshot.Settings.ModelLoads[0].Resolved = nil
			rb.b.engineConfigMu.Unlock()
			before := rb.baseline(t)
			_, err := writeRawProfileDraft(t, rb.b, peer, doc, profiles)
			if policyErrorCode(err) != -32010 {
				t.Fatalf("stale profile edit = %v, want revision conflict", err)
			}
			before.requireUnchanged(t)
		})
	}
	test("local", false)
	test("peer", true)
}

func TestRawPeerProfileNormalizationKeepsInferenceLock(t *testing.T) {
	rb := newRawProfileBroker(t)
	r := rb.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Profiles[0].LocalOnly = true
	r.mu.Unlock()
	doc, profiles := rawProfileDraft(t, rb.b, true)
	profiles[0]["contextTokens"] = json.RawMessage(`-1`)
	profiles[0]["modelIdentity"] = json.RawMessage(`"ignored"`)
	got, err := writeRawProfileDraft(t, rb.b, true, doc, profiles)
	if err != nil {
		t.Fatalf("normalizing managed fields of locked profile: %v", err)
	}
	if !got.Profiles[0].LocalOnly || got.Profiles[0].ContextTokens != 8192 || got.Profiles[0].ModelIdentity.GeneralName != "Stored model" {
		t.Fatalf("locked profile readback = %+v", got.Profiles[0])
	}
	doc, profiles = rawProfileDraft(t, rb.b, true)
	profiles[0]["model"] = rmJSON(t, "changed-model")
	before := rb.baseline(t)
	_, err = writeRawProfileDraft(t, rb.b, true, doc, profiles)
	var locked *nodepolicy.RemoteSettingLockedError
	if !errors.As(err, &locked) || policyErrorCode(err) != -32014 {
		t.Fatalf("locked retarget = %v, want inference lock before raw target refusal", err)
	}
	before.requireUnchanged(t)
}

func TestRawPolicyProfileUsesNewestSnapshotUnderWriterLock(t *testing.T) {
	rb := newRawProfileBroker(t)
	doc, profiles := rawProfileDraft(t, rb.b, true)
	profiles[0]["contextTokens"] = json.RawMessage(`-1`)
	doc["profiles"] = rmJSON(t, profiles)
	raw, expected := rmJSON(t, doc), doc["revision"]
	unlock := rmHoldLock(t, &rb.b.engineConfigMu)
	done := rmBlockedCall(t, "persistPolicy", func() error {
		_, err := rb.b.setPeerPolicyAs(raw, expected, "pinned-peer")
		return err
	})
	load := &rb.b.engineSettings["llamacpp"].Snapshot.Settings.ModelLoads[0]
	load.Requested.Context.Tokens = 16384
	load.Resolved.PerSlotContext, load.Resolved.TotalContext = 16384, 32768
	r := rb.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Profiles[0].ModelIdentity.GeneralName = "Newest stored identity"
	r.mu.Unlock()
	unlock()
	if err := rmResult(t, done); err != nil {
		t.Fatalf("queued raw policy: %v", err)
	}
	got := rb.b.localPolicy().Policy.Profiles[0]
	if got.ContextTokens != 16384 || got.ModelIdentity.GeneralName != "Newest stored identity" {
		t.Fatalf("queued write used stale managed fields: %+v", got)
	}
}

func TestRawPolicyProfileRefusalDoesNotAllocateRevision(t *testing.T) {
	rb := newRawProfileBroker(t)
	doc, profiles := rawProfileDraft(t, rb.b, false)
	profiles[0]["model"] = rmJSON(t, "changed-model")
	r := rb.b.nodePolicyRuntime()
	r.mu.Lock()
	issued := r.issued
	r.mu.Unlock()
	_, err := writeRawProfileDraft(t, rb.b, false, doc, profiles)
	if err == nil {
		t.Fatal("retarget succeeded")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.issued != issued {
		t.Fatalf("refusal allocated revision %d, want %d", r.issued, issued)
	}
}

func rawSnapshotLoad(key string, context int32) modelconfig.ModelLoad {
	return modelconfig.ModelLoad{
		ProfileKey: key, ModelID: "shared.gguf",
		Requested: modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "fixed", Tokens: context}},
		Resolved: &modelconfig.ResolvedLoad{
			PerSlotContext: context, TotalContext: context * 2, Slots: 2,
			GPU: modelconfig.GPU{Mode: "cpu"}, NativeBuildID: "fixture-build",
			SourceInputDigest: strings.Repeat("a", 64), PresetDigest: strings.Repeat("b", 64),
			Artifacts: []modelconfig.ArtifactBinding{{Role: "model", Path: "shared.gguf", ResolvedTarget: "shared.gguf", Size: "256", MtimeNS: "1", SampleSHA256: strings.Repeat("c", 64)}},
		},
	}
}

func rawSnapshotLaunch(engine string) settings.LaunchState {
	launch := settings.LaunchState{Engine: engine, ServerPort: 18080, Format: "pair-arguments-v1"}
	if engine == "llamacpp" {
		launch.ModelLoads = []modelconfig.ModelLoad{rawSnapshotLoad("small-profile", 8192)}
	}
	return launch
}

// The policy exists, but no settings read or accepted operation has ever
// initialized a journal record. Every native fact comes through the fake worker.
func newRawSnapshotBroker(t *testing.T) *routedBroker {
	t.Helper()
	p := nodepolicy.Default()
	p.Profiles = []nodepolicy.Profile{{Key: "small-profile", DisplayName: "Small", Engine: "llamacpp", Model: "small-profile", ContextTokens: 1024}}
	rb := newBrokerOnFile(t, string(rmJSON(t, p)))
	rb.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:get-launch" {
			var request struct {
				Engine string `json:"engine"`
			}
			if err := json.Unmarshal(raw, &request); err != nil {
				return nil, err.Error()
			}
			return rawSnapshotLaunch(request.Engine), ""
		}
		return map[string]bool{"ok": true}, ""
	})
	waitPushesSettled(t, rb.b)
	requireRawSnapshotPrivate(t, rb.b)
	return rb
}

func requireRawSnapshotPrivate(t *testing.T, b *Broker) {
	t.Helper()
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if len(b.engineSettings) != 0 {
		t.Fatalf("private preparation attached settings records: %+v", b.engineSettings)
	}
	path, err := b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("settings journal exists or could not be inspected: %v", err)
	}
}

// Fail the first durable-writer step and count it, so a refusal that tries a
// write cannot pass by merely preserving the old bytes after an injected error.
func forbidRawSnapshotWrites(t *testing.T, b *Broker) func() {
	t.Helper()
	var policyWrites, settingsWrites atomic.Int32
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.replacer.CreateTemp = func(string, string, os.FileMode) (durablefile.File, error) {
		policyWrites.Add(1)
		return nil, errors.New("unexpected raw policy write")
	}
	issued := r.issued
	r.mu.Unlock()
	b.engineConfigMu.Lock()
	b.engineSettingsReplacer.CreateTemp = func(string, string, os.FileMode) (durablefile.File, error) {
		settingsWrites.Add(1)
		return nil, errors.New("unexpected settings journal write")
	}
	b.engineConfigMu.Unlock()
	return func() {
		t.Helper()
		if policyWrites.Load() != 0 || settingsWrites.Load() != 0 {
			t.Errorf("refused preparation entered durable writers: policy=%d settings=%d", policyWrites.Load(), settingsWrites.Load())
		}
		r.mu.Lock()
		defer r.mu.Unlock()
		if r.issued != issued {
			t.Errorf("refused preparation allocated revision %d, want %d", r.issued, issued)
		}
	}
}

func TestRawPolicyFreshSnapshotDerivesContext(t *testing.T) {
	test := func(name string, peer bool) {
		t.Run(name, func(t *testing.T) {
			rb := newRawSnapshotBroker(t)
			doc, profiles := rawProfileDraft(t, rb.b, peer)
			profiles[0]["contextTokens"] = json.RawMessage(`"ignored managed value"`)
			got, err := writeRawProfileDraft(t, rb.b, peer, doc, profiles)
			if err != nil {
				t.Fatalf("fresh raw set: %v", err)
			}
			if got.Profiles[0].ContextTokens != 8192 {
				t.Fatalf("fresh context = %d, want native per-slot 8192", got.Profiles[0].ContextTokens)
			}
			var saved nodepolicy.Policy
			if err := json.Unmarshal(rb.fileBytes(t), &saved); err != nil {
				t.Fatal(err)
			}
			if saved.Profiles[0].ContextTokens != 8192 || saved.Revision != got.Revision {
				t.Fatalf("saved fresh policy = %+v", saved)
			}
			if len(rb.engine.methods("engine:get-launch")) != 2 {
				t.Fatalf("native reads = %v, want initial and source recheck", rb.engine.methods("engine:get-launch"))
			}
			requireRawSnapshotPrivate(t, rb.b)
		})
	}
	test("local", false)
	test("peer", true)
}

func TestRawPolicyFreshSnapshotProtectsTypedTarget(t *testing.T) {
	test := func(name string, peer bool) {
		t.Run(name, func(t *testing.T) {
			rb := newRawSnapshotBroker(t)
			doc, profiles := rawProfileDraft(t, rb.b, peer)
			profiles[0]["model"] = rmJSON(t, "other-model")
			before := rb.baseline(t)
			assertNoWrites := forbidRawSnapshotWrites(t, rb.b)
			_, err := writeRawProfileDraft(t, rb.b, peer, doc, profiles)
			if err == nil || err.Error() != "A profile keeps its model; create a new profile." {
				t.Fatalf("fresh typed target refusal = %v", err)
			}
			if len(rb.engine.methods("engine:get-launch")) != 2 {
				t.Fatal("retarget did not consult fresh native authority")
			}
			assertNoWrites()
			before.requireUnchanged(t)
			requireRawSnapshotPrivate(t, rb.b)
		})
	}
	test("local", false)
	test("peer", true)
}

func TestRawPolicyEmptyJournalSnapshotProtectsTypedTarget(t *testing.T) {
	test := func(name string, peer bool) {
		t.Run(name, func(t *testing.T) {
			rb := newRawSnapshotBroker(t)
			path, err := rb.b.engineSettingsPath()
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, []byte(`{}`), 0o600); err != nil {
				t.Fatal(err)
			}
			doc, profiles := rawProfileDraft(t, rb.b, peer)
			profiles[0]["model"] = rmJSON(t, "other-model")
			before := rb.baseline(t)
			assertNoWrites := forbidRawSnapshotWrites(t, rb.b)
			_, err = writeRawProfileDraft(t, rb.b, peer, doc, profiles)
			if err == nil || err.Error() != "A profile keeps its model; create a new profile." {
				t.Fatalf("empty-journal typed target refusal = %v", err)
			}
			assertNoWrites()
			before.requireUnchanged(t)
			data, err := os.ReadFile(path)
			if err != nil || string(data) != `{}` {
				t.Fatalf("private preparation changed existing journal: %q (%v)", data, err)
			}
			rb.b.engineConfigMu.Lock()
			defer rb.b.engineConfigMu.Unlock()
			if len(rb.b.engineSettings) != 0 {
				t.Fatalf("refused preparation attached an initial record: %+v", rb.b.engineSettings)
			}
		})
	}
	test("local", false)
	test("peer", true)
}

func TestRawPolicyFreshSnapshotIncludesIncomingEngine(t *testing.T) {
	rb := newRawSnapshotBroker(t)
	rb.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method != "engine:get-launch" {
			return map[string]bool{"ok": true}, ""
		}
		var request struct {
			Engine string `json:"engine"`
		}
		if err := json.Unmarshal(raw, &request); err != nil {
			return nil, err.Error()
		}
		launch := rawSnapshotLaunch(request.Engine)
		if request.Engine == "llamacpp-custom" {
			launch.ModelLoads = []modelconfig.ModelLoad{rawSnapshotLoad("large-profile", 32768)}
		}
		return launch, ""
	})
	doc, profiles := rawProfileDraft(t, rb.b, false)
	profiles = append(profiles, map[string]json.RawMessage{
		"key": rmJSON(t, "large-profile"), "displayName": rmJSON(t, "Large"), "name": rmJSON(t, "large"),
		"engine": rmJSON(t, "llamacpp-custom"), "model": rmJSON(t, "large-profile"), "contextTokens": json.RawMessage(`-1`),
	})
	got, err := writeRawProfileDraft(t, rb.b, false, doc, profiles)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Profiles) != 2 || got.Profiles[0].ContextTokens != 8192 || got.Profiles[1].ContextTokens != 32768 {
		t.Fatalf("incoming engine contexts = %+v", got.Profiles)
	}
	requireRawSnapshotPrivate(t, rb.b)
}

func TestRawPolicySnapshotUnknownAuthorityRefusesWithoutWrites(t *testing.T) {
	test := func(name string, peer bool, setup func(*testing.T, *routedBroker), want string) {
		t.Run(name, func(t *testing.T) {
			rb := newRawSnapshotBroker(t)
			setup(t, rb)
			doc, profiles := rawProfileDraft(t, rb.b, peer)
			profiles[0]["contextTokens"] = json.RawMessage(`999999`)
			before := rb.baseline(t)
			assertNoWrites := forbidRawSnapshotWrites(t, rb.b)
			_, err := writeRawProfileDraft(t, rb.b, peer, doc, profiles)
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("authority refusal = %v, want %q", err, want)
			}
			assertNoWrites()
			before.requireUnchanged(t)
		})
	}
	missing := func(t *testing.T, rb *routedBroker) { rb.b.setEngineMgr(nil) }
	unreadable := func(t *testing.T, rb *routedBroker) {
		rb.engine.setAnswer(func(string, json.RawMessage) (any, string) { return nil, "native settings unreadable" })
	}
	malformed := func(t *testing.T, rb *routedBroker) {
		rb.engine.setAnswer(func(string, json.RawMessage) (any, string) { return map[string]bool{"ok": true}, "" })
	}
	journal := func(t *testing.T, rb *routedBroker) {
		path, err := rb.b.engineSettingsPath()
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(`{"broken":`), 0o600); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			data, err := os.ReadFile(path)
			if err != nil || string(data) != `{"broken":` {
				t.Errorf("unreadable journal changed: %q (%v)", data, err)
			}
		})
	}
	invalidRecord := func(t *testing.T, rb *routedBroker) {
		path, err := rb.b.engineSettingsPath()
		if err != nil {
			t.Fatal(err)
		}
		data := []byte(`{"llamacpp":{}}`)
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			saved, err := os.ReadFile(path)
			if err != nil || string(saved) != string(data) {
				t.Errorf("invalid snapshot journal changed: %q (%v)", saved, err)
			}
		})
	}
	test("local missing manager", false, missing, "engine manager unavailable")
	test("peer missing manager", true, missing, "engine manager unavailable")
	test("local unreadable native settings", false, unreadable, "native settings unreadable")
	test("peer unreadable native settings", true, unreadable, "native settings unreadable")
	test("local unidentified native reply", false, malformed, "readback did not identify")
	test("peer unidentified native reply", true, malformed, "readback did not identify")
	test("local unreadable journal", false, journal, "journal could not be read")
	test("peer unreadable journal", true, journal, "journal could not be read")
	test("local unidentified saved snapshot", false, invalidRecord, "snapshot authority is invalid")
	test("peer unidentified saved snapshot", true, invalidRecord, "snapshot authority is invalid")
}

func TestRawPolicySnapshotReadsOutsidePolicyMutex(t *testing.T) {
	rb := newRawSnapshotBroker(t)
	var reads atomic.Int32
	rb.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method != "engine:get-launch" {
			return map[string]bool{"ok": true}, ""
		}
		r := &rb.b.nodePolicy
		if !r.mu.TryLock() {
			return nil, "worker read held policy mutex"
		}
		r.mu.Unlock()
		if rb.b.engineConfigMu.TryLock() {
			rb.b.engineConfigMu.Unlock()
			return nil, "worker read released settings CAS mutex"
		}
		reads.Add(1)
		return rawSnapshotLaunch("llamacpp"), ""
	})
	doc, profiles := rawProfileDraft(t, rb.b, false)
	if _, err := writeRawProfileDraft(t, rb.b, false, doc, profiles); err != nil {
		t.Fatalf("worker lock contract: %v", err)
	}
	if reads.Load() != 2 {
		t.Fatalf("worker lock checks = %d, want 2", reads.Load())
	}
}

func TestRawPolicySnapshotSourceChangeRefusesWithoutWrites(t *testing.T) {
	rb := newRawSnapshotBroker(t)
	var reads atomic.Int32
	rb.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:get-launch" {
			if reads.Add(1) == 2 {
				rb.b.clearEngineMgrBinding(rb.source)
			}
			return rawSnapshotLaunch("llamacpp"), ""
		}
		return map[string]bool{"ok": true}, ""
	})
	doc, profiles := rawProfileDraft(t, rb.b, true)
	before := rb.baseline(t)
	assertNoWrites := forbidRawSnapshotWrites(t, rb.b)
	_, err := writeRawProfileDraft(t, rb.b, true, doc, profiles)
	if policyErrorCode(err) != codePolicyConflict || !strings.Contains(err.Error(), "engine source changed") {
		t.Fatalf("source change = %v, want conflict", err)
	}
	assertNoWrites()
	before.requireUnchanged(t)
	requireRawSnapshotPrivate(t, rb.b)
}

func TestRawPolicySnapshotRetiredPolicySourceRefusesWithoutWrites(t *testing.T) {
	rb := newRawSnapshotBroker(t)
	var reads atomic.Int32
	rb.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:get-launch" {
			if reads.Add(1) == 2 {
				r := &rb.b.nodePolicy
				r.mu.Lock()
				r.engineManagerSource = &engineManagerSource{settingsEpoch: "replacement-source"}
				r.mu.Unlock()
			}
			return rawSnapshotLaunch("llamacpp"), ""
		}
		return map[string]bool{"ok": true}, ""
	})
	doc, profiles := rawProfileDraft(t, rb.b, false)
	before := rb.baseline(t)
	assertNoWrites := forbidRawSnapshotWrites(t, rb.b)
	_, err := writeRawProfileDraft(t, rb.b, false, doc, profiles)
	if policyErrorCode(err) != codePolicyConflict || !strings.Contains(err.Error(), "engine source changed") {
		t.Fatalf("retired policy source = %v, want conflict", err)
	}
	assertNoWrites()
	before.requireUnchanged(t)
	requireRawSnapshotPrivate(t, rb.b)
}

func TestRawPolicySnapshotNativeDriftRefusesWithoutWrites(t *testing.T) {
	test := func(name string, mutate func(*settings.LaunchState)) {
		t.Run(name, func(t *testing.T) {
			rb := newRawSnapshotBroker(t)
			var reads atomic.Int32
			rb.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method != "engine:get-launch" {
					return map[string]bool{"ok": true}, ""
				}
				launch := rawSnapshotLaunch("llamacpp")
				if reads.Add(1) == 2 {
					mutate(&launch)
				}
				return launch, ""
			})
			doc, profiles := rawProfileDraft(t, rb.b, false)
			before := rb.baseline(t)
			assertNoWrites := forbidRawSnapshotWrites(t, rb.b)
			_, err := writeRawProfileDraft(t, rb.b, false, doc, profiles)
			if policyErrorCode(err) != codePolicyConflict || !strings.Contains(err.Error(), "native source changed") {
				t.Fatalf("native drift = %v, want conflict", err)
			}
			assertNoWrites()
			before.requireUnchanged(t)
			requireRawSnapshotPrivate(t, rb.b)
		})
	}
	test("typed load changed", func(launch *settings.LaunchState) {
		launch.ModelLoads = []modelconfig.ModelLoad{rawSnapshotLoad("small-profile", 16384)}
	})
	test("other saved settings changed", func(launch *settings.LaunchState) { launch.LaunchText = "--changed" })
}

func TestRawPolicySnapshotPolicyCASChangeRefusesWithoutWrites(t *testing.T) {
	rb := newRawSnapshotBroker(t)
	var changed atomic.Bool
	rb.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:get-launch" {
			if changed.CompareAndSwap(false, true) {
				r := &rb.b.nodePolicy
				r.mu.Lock()
				r.current.Revision = "987654321"
				r.mu.Unlock()
			}
			return rawSnapshotLaunch("llamacpp"), ""
		}
		return map[string]bool{"ok": true}, ""
	})
	doc, profiles := rawProfileDraft(t, rb.b, true)
	before := rb.baseline(t)
	assertNoWrites := forbidRawSnapshotWrites(t, rb.b)
	_, err := writeRawProfileDraft(t, rb.b, true, doc, profiles)
	var conflict *policyConflictError
	if !errors.As(err, &conflict) || conflict.current != "987654321" {
		t.Fatalf("policy CAS change = %v, want current revision conflict", err)
	}
	// Only the independently changed owner revision may differ.
	before.policy.Revision, before.revision = "987654321", "987654321"
	assertNoWrites()
	before.requireUnchanged(t)
	requireRawSnapshotPrivate(t, rb.b)
}

func TestRawPolicySnapshotStoredEngineRequiresAuthorityOnRemoval(t *testing.T) {
	rb := newRawSnapshotBroker(t)
	rb.engine.setAnswer(func(string, json.RawMessage) (any, string) { return nil, "stored engine authority unavailable" })
	doc, _ := rawProfileDraft(t, rb.b, false)
	before := rb.baseline(t)
	assertNoWrites := forbidRawSnapshotWrites(t, rb.b)
	_, err := writeRawProfileDraft(t, rb.b, false, doc, nil)
	if err == nil || !strings.Contains(err.Error(), "stored engine authority unavailable") {
		t.Fatalf("removing unhydrated stored profile = %v", err)
	}
	assertNoWrites()
	before.requireUnchanged(t)
	requireRawSnapshotPrivate(t, rb.b)
}

func TestRawPolicyPreparedSnapshotDetached(t *testing.T) {
	rb := newRawSnapshotBroker(t)
	p := rb.b.localPolicy().Policy
	raw := rmJSON(t, p)
	rb.b.engineConfigMu.Lock()
	loads, current, err := rb.b.prepareProfileLoadsLocked(context.Background(), raw, p)
	if err == nil {
		rb.b.nodePolicy.mu.Lock()
		err = current()
		rb.b.nodePolicy.mu.Unlock()
	}
	rb.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	loads["llamacpp"][0].Requested.Context.Tokens = 256
	loads["llamacpp"][0].Resolved.PerSlotContext = 256
	loads["llamacpp"][0].Resolved.Artifacts[0].Path = "changed.gguf"
	got, err := rb.b.setLocalPolicy(raw, rmJSON(t, p.Revision))
	if err != nil {
		t.Fatal(err)
	}
	if got.Profiles[0].ContextTokens != 8192 {
		t.Fatalf("mutated private read changed next raw write: %+v", got.Profiles[0])
	}
	requireRawSnapshotPrivate(t, rb.b)
}
