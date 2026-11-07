// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

const profileConversionPreset = `[*]
fit = off
load-on-startup = false
[medium-model]
model = medium.gguf
ctx-size = 4096
parallel = 1
gpu-layers = 0
[weak-model]
model = weak.gguf
ctx-size = 4096
parallel = 1
gpu-layers = 0
fit = on
fit-target = 1024
fit-ctx = 2048
spec-draft-n-max = 4
`

func newProfileConversionProcessFixture(t *testing.T) *modelSettingsProcessFixture {
	t.Helper()
	f := prepareModelSettingsProcessFixture(t)
	modelSettingsWrite(t, f.preset, []byte(profileConversionPreset))
	policy := nodepolicy.Default()
	policy.Idle.UnloadAfterMinutes, policy.Idle.StopEngineAfterMinutes = 0, 0
	policy.Profiles = []nodepolicy.Profile{{
		Key: "medium-model", DisplayName: "My custom profile", Icon: "code",
		Engine: "llamacpp", Model: "medium-model", ContextTokens: 4096,
		Capabilities: []nodepolicy.Capability{nodepolicy.CapTools, nodepolicy.CapStructured},
		MemoryBytes:  1048576, MaxConcurrent: 2, MaxOutputTokens: 512,
		RequestOptions:     json.RawMessage(`{"temperature":0.2,"top_p":0.9,"opaque":{"integer":9007199254740993,"items":[2,1]}}`),
		RequestOptionsMode: nodepolicy.RequestOptionsEnforced,
		Qualification:      &nodepolicy.QualificationRef{ID: "saved-conversion-reference", Revision: "7"},
	}}
	// Keep declaration order, an occupied role and empty roles in the oracle.
	// The newly converted sibling must not acquire membership or create a role.
	policy.Roles = []nodepolicy.Role{
		{Name: "strong", Profiles: []string{}},
		{Name: "weak", Profiles: []string{"medium-model"}},
		{Name: "medium", Profiles: []string{}},
	}
	jointWriteJSON(t, filepath.Join(jointAppDir(f.base), nodepolicy.FileName), policy)
	f.startModelSettings(t)
	state, stored := f.snapshot(t), f.policy(t)
	if state.Running || len(state.Settings.ModelLoads) != 0 || len(stored.Profiles) != 1 || !reflect.DeepEqual(stored.Roles, policy.Roles) {
		t.Fatal("conversion fixture must begin Off with an external preset, one policy profile and the declared roles")
	}
	return f
}

func profileConversionIdentity(name string, context int64) engines.ModelIdentity {
	// These fixtures have no repository, quantization or verified content hash.
	// Identity expectations come from their known GGUF bytes, not the preview.
	size := int64(len(modelSettingsGGUF("blk.0.weight")))
	return engines.ModelIdentity{
		Architecture: "llama", TrainedContextTokens: 32768, ConfiguredContextTokens: context,
		Files:      []engines.ModelIdentityFile{{Role: "model", Name: name, SizeBytes: size}},
		TotalBytes: size, Strength: "declared",
	}
}

func assertProfileConversionSettings(t *testing.T, f *modelSettingsProcessFixture, config settings.Config) {
	t.Helper()
	if len(config.ModelLoads) != 2 || config.Serving != nil {
		t.Fatal("whole-engine conversion must retain both loads without manufacturing a finite serving envelope")
	}
	for _, want := range []struct {
		key, file, tensor, fit string
		context, total, slots  int32
	}{
		{"medium-model", "medium.gguf", "blk.0.weight", "off", 8192, 16384, 2},
		{"weak-model", "weak.gguf", "blk.1.weight", "on", 4096, 4096, 1},
	} {
		index := slices.IndexFunc(config.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == want.key })
		if index < 0 {
			t.Fatalf("conversion omitted profile %s", want.key)
		}
		load := config.ModelLoads[index]
		if load.ModelID != want.key || load.Requested.Context == nil || load.Requested.Context.Mode != "fixed" || load.Requested.Context.Tokens != want.context || load.Requested.Slots == nil || *load.Requested.Slots != want.slots || load.Requested.GPU == nil || load.Requested.GPU.Mode != "cpu" || load.Requested.Fit == nil || *load.Requested.Fit != want.fit {
			t.Fatalf("converted profile %s lost its independent requested settings: %+v", want.key, load.Requested)
		}
		r := load.Resolved
		if r == nil || r.PerSlotContext != want.context || r.TotalContext != want.total || r.Slots != want.slots || r.GPU.Mode != "cpu" || r.NativeBuildID == "" || len(r.SourceInputDigest) != 64 || len(r.PresetDigest) != 64 || r.ResolverRevision != "external-conversion/llama" || len(r.ReviewedInputID) != 64 {
			t.Fatalf("converted profile %s lost resolved settings or source/native fences: %+v", want.key, r)
		}
		binding := modelSettingsExpectedBinding(t, f.root, "model", want.file, modelSettingsGGUF(want.tensor))
		assertModelSettingsBindings(t, f.root, r.Artifacts, []modelconfig.ArtifactBinding{binding})
		if want.key == "weak-model" {
			if load.Requested.FitTarget == nil || *load.Requested.FitTarget != "1024" || load.Requested.FitCtx == nil || *load.Requested.FitCtx != 2048 || !reflect.DeepEqual(load.Requested.ExtraOptions, map[string]string{"spec-draft-n-max": "4"}) || r.Options.Fit == nil || *r.Options.Fit != "on" || !reflect.DeepEqual(r.Options.ExtraOptions, map[string]string{"spec-draft-n-max": "4"}) {
				t.Fatal("unselected native-fit section lost its fit controls or supported extra option")
			}
		} else if len(load.Requested.ExtraOptions) != 0 {
			t.Fatal("conversion copied sibling extra options onto the finite profile")
		}
	}
}

func assertProfileConversionPolicy(t *testing.T, before, after nodepolicy.Policy) {
	t.Helper()
	want := before
	want.Profiles = slices.Clone(before.Profiles)
	if len(want.Profiles) != 1 || want.Profiles[0].Key != "medium-model" || after.Revision == before.Revision {
		t.Fatal("conversion did not publish a new policy revision from the seeded profile")
	}
	medium := profileConversionIdentity("medium.gguf", 8192)
	weak := profileConversionIdentity("weak.gguf", 4096)
	want.Profiles[0].ContextTokens, want.Profiles[0].ModelIdentity = 8192, &medium
	want.Profiles = append(want.Profiles, nodepolicy.Profile{
		Key: "weak-model", DisplayName: "weak-model", Engine: "llamacpp", Model: "weak-model",
		ContextTokens: 4096, ModelIdentity: &weak,
	})
	want.Revision = after.Revision
	if len(after.Profiles) != len(want.Profiles) {
		t.Fatalf("conversion policy profiles=%d, want both external sections", len(after.Profiles))
	}
	for i := range want.Profiles {
		if !nodepolicy.EqualRequestOptions(want.Profiles[i].RequestOptions, after.Profiles[i].RequestOptions) {
			t.Fatalf("conversion changed request options for %s", want.Profiles[i].Key)
		}
		// Ignore JSON object ordering only, retaining exact numeric precision.
		want.Profiles[i].RequestOptions = after.Profiles[i].RequestOptions
	}
	if !bytes.Equal(mustMarshal(t, want), mustMarshal(t, after)) {
		t.Fatal("conversion changed user-owned profile fields, role declarations/memberships or unrelated policy state")
	}
}

func TestProfileConversionPublishesWholeEngineAcrossProcesses(t *testing.T) {
	f := newProfileConversionProcessFixture(t)
	baseline := f.metadataBaseline(t)
	request, preview := f.review(t, f.request(t, "whole-engine-conversion", nil))
	assertProfileConversionSettings(t, f, preview.Settings)
	wantIdentities := map[string]engines.ModelIdentity{
		"medium-model": profileConversionIdentity("medium.gguf", 8192),
		"weak-model":   profileConversionIdentity("weak.gguf", 4096),
	}
	if !reflect.DeepEqual(preview.Identities, wantIdentities) {
		t.Fatal("public conversion preview did not return both exact keyed identities")
	}
	if preview.Serving == nil || preview.Serving.Status != "unverified" || len(preview.Serving.Configured) != 0 || len(preview.Serving.LaunchValidated) != 0 || len(preview.Serving.Observed) != 0 {
		t.Fatal("public conversion preview manufactured finite serving capacity or loaded-runtime evidence")
	}
	f.noMetadataEffects(t, baseline)
	if !bytes.Equal(readLiveDurableFile(t, f.preset), []byte(profileConversionPreset)) {
		t.Fatal("preview rewrote the external preset")
	}
	f.rpcOK(t, "engine:apply-settings", request)
	after := f.settled(t, request.RequestID, "succeeded")
	assertModelSettingsSavedReadback(t, after, preview.Settings)
	assertProfileConversionSettings(t, f, after.Settings)
	policy := f.policy(t)
	assertProfileConversionPolicy(t, baseline.policy, policy)
	var inventory struct {
		Policy *nodepolicy.Advertisement `json:"policy"`
	}
	modelSettingsDecode(t, f.rpcOK(t, "engine:models", nil), &inventory)
	medium, weak := wantIdentities["medium-model"], wantIdentities["weak-model"]
	wantAdvertised := []nodepolicy.AdvertisedProfile{
		{
			Key: "medium-model", DisplayName: "My custom profile", Icon: "code", Engine: "llamacpp", Model: "medium-model",
			ContextTokens: 8192, ModelIdentity: &medium,
			Capabilities: []nodepolicy.Capability{nodepolicy.CapTools, nodepolicy.CapStructured}, MaxOutputTokens: 512,
			Qualification: &nodepolicy.QualificationRef{ID: "saved-conversion-reference", Revision: "7"},
		},
		{Key: "weak-model", DisplayName: "weak-model", Engine: "llamacpp", Model: "weak-model", ContextTokens: 4096, ModelIdentity: &weak},
	}
	if inventory.Policy == nil || inventory.Policy.Revision != policy.Revision || !reflect.DeepEqual(inventory.Policy.Roles, baseline.policy.Roles) || !reflect.DeepEqual(inventory.Policy.Profiles, wantAdvertised) {
		t.Fatalf("public inventory omitted converted profiles or changed ordered roles: got %s, want profiles %s and roles %s", mustMarshal(t, inventory), mustMarshal(t, wantAdvertised), mustMarshal(t, baseline.policy.Roles))
	}
	var native modelconfig.InspectResult
	modelSettingsDecode(t, f.rpcOK(t, "engine:inspect-model", modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "weak-model", ProfileKey: "weak-model"}), &native)
	// A saved requested context does not price native fit or establish its
	// realized allocation. Only its unpriced execution identity is available.
	if native.Status != "resolved" || native.ConfigurationKeyReason != "native-fit" || native.ConfigurationKey != "" || native.ExecutionKey == "" || len(native.ConfigurationPools) != 0 || native.Estimate.Status != "unknown" {
		t.Fatal("converted native-fit profile acquired finite pricing or lost its applied execution identity")
	}
	if !bytes.Equal(readLiveDurableFile(t, f.preset), []byte(profileConversionPreset)) {
		t.Fatal("apply rewrote the operator's external source preset")
	}
	f.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
	f.engineIdentity(t, 1)
	sections := modelSettingsPreset(t, readLiveDurableFile(t, f.capture))
	if len(sections) != 2 || sections["medium-model"]["ctx-size"] != "16384" || sections["medium-model"]["parallel"] != "2" || sections["medium-model"]["fit"] != "off" || sections["weak-model"]["ctx-size"] != "4096" || sections["weak-model"]["parallel"] != "1" || sections["weak-model"]["fit"] != "on" || sections["weak-model"]["fit-target"] != "1024" || sections["weak-model"]["fit-ctx"] != "2048" || sections["weak-model"]["spec-draft-n-max"] != "4" {
		t.Fatal("synthetic router did not consume both converted sections and their independent native options")
	}
	if !modelSettingsSamePath(sections["medium-model"]["model"], filepath.Join(f.root, "medium.gguf")) || !modelSettingsSamePath(sections["weak-model"]["model"], filepath.Join(f.root, "weak.gguf")) {
		t.Fatal("generated conversion sections do not select the private fixture artifacts")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.configureCalls != 1 || f.mutations != 0 || f.inferences != 0 {
		t.Fatalf("Off conversion effects: configure=%d mutation=%d inference=%d", f.configureCalls, f.mutations, f.inferences)
	}
}

func TestProfileConversionRecoversEveryProfileAfterNativePersistence(t *testing.T) {
	f := newProfileConversionProcessFixture(t)
	before := f.policy(t)
	request, preview := f.review(t, f.request(t, "conversion-native-persisted", nil))
	gate := newModelSettingsGate()
	f.mu.Lock()
	f.readbackGate = gate
	f.mu.Unlock()
	f.begin(t, "engine:apply-settings", request)
	modelSettingsWait(t, gate)
	if f.disk(t).Model.Stage != "native-applying" || !reflect.DeepEqual(before, f.policy(t)) {
		t.Fatal("conversion published profiles before accepted native readback")
	}
	var override struct {
		Platforms map[string]struct {
			Runtime struct {
				ModelLoads []modelconfig.ModelLoad `json:"modelLoads"`
			} `json:"runtime"`
		} `json:"platforms"`
	}
	modelSettingsDecode(t, readLiveDurableFile(t, filepath.Join(jointAppDir(f.base), "engines", "llamacpp.json")), &override)
	native := override.Platforms[runtime.GOOS+"/"+runtime.GOARCH].Runtime.ModelLoads
	assertProfileConversionSettings(t, f, settings.Config{ModelLoads: native})
	// This existing bridge barrier holds the readback after the real manager's
	// persistence, before broker policy publication. Recovery uses the journal.
	f.crashBroker(t, gate)
	f.startModelSettings(t)
	after := f.settled(t, request.RequestID, "succeeded")
	assertModelSettingsSavedReadback(t, after, preview.Settings)
	assertProfileConversionSettings(t, f, after.Settings)
	assertProfileConversionPolicy(t, before, f.policy(t))
	var replay struct {
		Revision uint64 `json:"revision"`
		Phase    string `json:"phase"`
	}
	modelSettingsDecode(t, f.rpcOK(t, "engine:apply-settings", request), &replay)
	if replay.Revision != after.Revision || replay.Phase != "succeeded" || f.disk(t).Model.Stage != "completed" {
		t.Fatal("recovered conversion lost its accepted receipt or retained an incomplete journal")
	}
}

func TestProfileConversionChangedExternalSourceFailsWithoutPublishing(t *testing.T) {
	f := newProfileConversionProcessFixture(t)
	baseline := f.metadataBaseline(t)
	nativePath := filepath.Join(jointAppDir(f.base), "engines", "llamacpp.json")
	nativeBefore := readLiveDurableFile(t, nativePath)
	request, preview := f.review(t, f.request(t, "conversion-stale-source", nil))
	changedPreset := append([]byte(profileConversionPreset), []byte("threads = 3\n")...)
	modelSettingsWrite(t, f.preset, changedPreset)
	var receipt struct {
		Revision uint64 `json:"revision"`
		Phase    string `json:"phase"`
	}
	modelSettingsDecode(t, f.rpcOK(t, "engine:apply-settings", request), &receipt)
	// The accepted operation reports its failure through the retained receipt.
	// Native publication refuses the changed source and keeps the repair fence.
	after := f.snapshot(t)
	if receipt.Phase != "failed" || after.RequestID != request.RequestID || after.Phase != "applying" || after.Editable || after.Running || !strings.Contains(after.UnresolvedError, "External router source changed") {
		t.Fatalf("changed source lost its failed receipt or public repair fence: receipt %s, snapshot %s", mustMarshal(t, receipt), mustMarshal(t, after))
	}
	if !after.Settings.Equal(preview.Settings) || after.AppliedRevision != baseline.snapshot.AppliedRevision || after.EffectiveSettings == nil || !after.EffectiveSettings.Equal(baseline.snapshot.Settings) || !reflect.DeepEqual(f.policy(t), baseline.policy) || !bytes.Equal(readLiveDurableFile(t, nativePath), nativeBefore) || !bytes.Equal(readLiveDurableFile(t, f.preset), changedPreset) {
		t.Fatal("failed source validation published native settings/policy or lost desired versus applied authority")
	}
	stored, exists := f.disk(t).Receipts[request.RequestID]
	if !exists || stored.Phase != "failed" || stored.Revision != receipt.Revision {
		t.Fatal("source failure did not retain the exact failed receipt")
	}
	var replay struct {
		Revision uint64 `json:"revision"`
		Phase    string `json:"phase"`
	}
	modelSettingsDecode(t, f.rpcOK(t, "engine:apply-settings", request), &replay)
	if replay != receipt {
		t.Fatal("replaying the source-failed operation changed its receipt")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.configureCalls != baseline.configureCalls+1 || f.mutations != baseline.mutations || f.inferences != baseline.inferences {
		t.Fatalf("source failure/replay effects: configure=%d mutation=%d inference=%d", f.configureCalls, f.mutations, f.inferences)
	}
	f.engineProcessFixture.mu.Lock()
	defer f.engineProcessFixture.mu.Unlock()
	for engine, starts := range f.starts {
		if len(starts) != 0 {
			t.Fatalf("source-failed conversion started synthetic engine %s", engine)
		}
	}
}

func TestProfileConversionManagerReplacementKeepsPolicyFenced(t *testing.T) {
	f := newProfileConversionProcessFixture(t)
	before := f.policy(t)
	request, _ := f.review(t, f.request(t, "conversion-replaced-manager", nil))
	gate := newModelSettingsGate()
	f.mu.Lock()
	f.configureGate = gate
	f.mu.Unlock()
	applyID := f.begin(t, "engine:apply-settings", request)
	modelSettingsWait(t, gate)
	f.replaceManager(t)
	gate.unblock()
	f.finishApply(t, applyID)
	task1Eventually(t, "conversion source replacement failure receipt", 90*time.Second, func() bool {
		return f.disk(t).Receipts[request.RequestID].Phase == "failed"
	})
	state := f.snapshot(t)
	if f.disk(t).Model.Stage == "completed" || !reflect.DeepEqual(before, f.policy(t)) || state.Phase != "applying" || state.Editable || state.UnresolvedError == "" {
		t.Fatal("obsolete native source published converted profiles or erased the unresolved operation fence")
	}
	response := f.rpc(t, "policy:set", map[string]any{"policy": before, "expectedPersistedRevision": before.Revision})
	if response.Error == nil || response.Error.Code != -32010 {
		t.Fatal("ordinary policy write bypassed unresolved conversion reservation")
	}
}
