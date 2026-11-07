// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

func newProfileRolesProcessFixture(t *testing.T) *modelSettingsProcessFixture {
	t.Helper()
	f := prepareModelSettingsProcessFixture(t)
	for _, name := range []string{"medium.gguf", "weak.gguf"} {
		if err := os.Remove(filepath.Join(f.root, name)); err != nil {
			t.Fatal(err)
		}
	}
	modelSettingsWrite(t, filepath.Join(f.root, "lfm.gguf"), modelSettingsGGUF("blk.0.weight"))
	// The external artifact selector is converted by the first coupled create.
	// Neither a policy profile nor a typed load is seeded here.
	modelSettingsWrite(t, f.preset, []byte("[*]\nfit = off\nload-on-startup = false\n[lfm-8k]\nmodel = lfm.gguf\nctx-size = 4096\nparallel = 1\ngpu-layers = 0\n"))
	policy := nodepolicy.Default()
	policy.Idle.UnloadAfterMinutes, policy.Idle.StopEngineAfterMinutes = 0, 0
	jointWriteJSON(t, filepath.Join(jointAppDir(f.base), nodepolicy.FileName), policy)
	f.startModelSettings(t)
	if state := f.snapshot(t); len(state.Settings.ModelLoads) != 0 || state.Running || len(f.policy(t).Profiles) != 0 {
		t.Fatal("profile creation fixture already has typed loads, profiles or a running engine")
	}
	return f
}

func profileRolesCreate(t *testing.T, f *modelSettingsProcessFixture, key string, tokens int32) settings.Preview {
	t.Helper()
	state, policy := f.snapshot(t), f.policy(t)
	before := f.metadataBaseline(t)
	request := settings.Request{
		Engine: "llamacpp", RequestID: "create-" + key, ExpectedRevision: state.Revision, Settings: state.Settings.Clone(),
		ModelChange: &modelconfig.ModelChange{
			ProfileKey: key, ModelID: "lfm-8k", ExpectedPolicyRevision: policy.Revision,
			Load: &modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{
				Context: &modelconfig.Context{Mode: "fixed", Tokens: tokens}, Slots: modelSettingsInt(1), GPU: &modelconfig.GPU{Mode: "cpu"},
			}},
		},
	}
	request, preview := f.review(t, request)
	f.noMetadataEffects(t, before)
	f.rpcOK(t, "engine:apply-settings", request)
	after := f.settled(t, request.RequestID, "succeeded")
	assertModelSettingsSavedReadback(t, after, preview.Settings)
	profile, ok := f.policy(t).ProfileByKey(key)
	if !ok || profile.Engine != "llamacpp" || profile.Model != key || profile.DisplayName != key || profile.ContextTokens != int(tokens) || profile.ModelIdentity == nil || profile.ModelIdentity.ConfiguredContextTokens != int64(tokens) {
		t.Fatalf("coupled creation did not publish the selected profile and derived metadata: %+v", profile)
	}
	return preview
}

func TestProfileRolesKeyedSiblingsAcrossProcesses(t *testing.T) {
	f := newProfileRolesProcessFixture(t)
	profileRolesCreate(t, f, "lfm-8k", 8192)
	profileRolesCreate(t, f, "lfm-32k", 32768)
	state := f.snapshot(t)
	if len(state.Settings.ModelLoads) != 2 {
		t.Fatalf("saved typed loads=%d, want two sibling profiles", len(state.Settings.ModelLoads))
	}
	artifact := modelSettingsExpectedBinding(t, f.root, "model", "lfm.gguf", modelSettingsGGUF("blk.0.weight"))
	for _, want := range []struct {
		key    string
		tokens int32
	}{{"lfm-8k", 8192}, {"lfm-32k", 32768}} {
		index := slices.IndexFunc(state.Settings.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == want.key })
		if index < 0 {
			t.Fatalf("missing saved profile %s", want.key)
		}
		load := state.Settings.ModelLoads[index]
		if load.ModelID != "lfm-8k" || load.Requested.Context == nil || load.Requested.Context.Mode != "fixed" || load.Requested.Context.Tokens != want.tokens || load.Resolved == nil || load.Resolved.PerSlotContext != want.tokens || load.Resolved.TotalContext != want.tokens || load.Resolved.Slots != 1 {
			t.Fatalf("saved sibling %s lost its artifact selector or independent context: %+v", want.key, load)
		}
		assertModelSettingsBindings(t, f.root, load.Resolved.Artifacts, []modelconfig.ArtifactBinding{artifact})
		var inspection modelconfig.InspectResult
		modelSettingsDecode(t, f.rpcOK(t, "engine:inspect-model", modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "lfm-8k", ProfileKey: want.key}), &inspection)
		if inspection.Status != "resolved" || inspection.ModelID != "lfm-8k" || len(inspection.Artifacts) != 1 {
			t.Fatalf("keyed sibling inspection %s=%+v", want.key, inspection)
		}
		assertModelSettingsInspectionBinding(t, inspection.Artifacts[0], artifact)
		if inspection.ConfigurationKeyReason == "not-applied" {
			t.Fatalf("keyed inspection %s treated its saved load as a new profile", want.key)
		}
	}
	var fresh modelconfig.InspectResult
	modelSettingsDecode(t, f.rpcOK(t, "engine:inspect-model", modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "lfm-8k"}), &fresh)
	if fresh.Status != "resolved" || fresh.ConfigurationKeyReason != "not-applied" || fresh.ConfigurationKey != "" || fresh.LoadObservation != nil {
		t.Fatal("inspection without a profile key inherited a saved sibling")
	}
	wrongArtifact := f.rpc(t, "engine:inspect-model", modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "other-artifact", ProfileKey: "lfm-32k"})
	if wrongArtifact.Error == nil || wrongArtifact.Error.Code != -32602 {
		t.Fatal("keyed inspection accepted a different artifact for a saved profile")
	}
	policy := f.policy(t)
	if len(policy.Profiles) != 2 {
		t.Fatal("coupled creation did not retain both policy profiles")
	}
	wantRoles := []nodepolicy.Role{{Name: "weak", Profiles: []string{"lfm-8k", "lfm-32k"}}}
	policy.Roles = wantRoles
	policy = f.setPolicy(t, policy)
	if !reflect.DeepEqual(policy.Roles, wantRoles) || !reflect.DeepEqual(f.policy(t).Roles, wantRoles) {
		t.Fatal("saved role membership lost declared profile order")
	}
	var inventory struct {
		Policy *nodepolicy.Advertisement `json:"policy"`
	}
	modelSettingsDecode(t, f.rpcOK(t, "engine:models", nil), &inventory)
	if inventory.Policy == nil || inventory.Policy.Revision != policy.Revision || !reflect.DeepEqual(inventory.Policy.Roles, wantRoles) || len(inventory.Policy.Profiles) != 2 {
		t.Fatal("real broker advertisement lost sibling profiles or ordered role membership")
	}
	for _, want := range []struct {
		key    string
		tokens int
	}{{"lfm-8k", 8192}, {"lfm-32k", 32768}} {
		index := slices.IndexFunc(inventory.Policy.Profiles, func(profile nodepolicy.AdvertisedProfile) bool { return profile.Key == want.key })
		if index < 0 {
			t.Fatalf("advertisement omitted %s", want.key)
		}
		profile := inventory.Policy.Profiles[index]
		if profile.Engine != "llamacpp" || profile.Model != want.key || profile.DisplayName != want.key || profile.ContextTokens != want.tokens || profile.ModelIdentity == nil || profile.ModelIdentity.ConfiguredContextTokens != int64(want.tokens) || len(profile.ModelIdentity.Files) != 1 || profile.ModelIdentity.Files[0].Name != "lfm.gguf" {
			t.Fatalf("advertised sibling metadata differs: %+v", profile)
		}
	}
	f.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
	f.engineIdentity(t, 1)
	sections := modelSettingsPreset(t, readLiveDurableFile(t, f.capture))
	if len(sections) != 2 || sections["lfm-8k"]["ctx-size"] != "8192" || sections["lfm-32k"]["ctx-size"] != "32768" || sections["lfm-8k"]["parallel"] != "1" || sections["lfm-32k"]["parallel"] != "1" {
		t.Fatal("real manager did not generate independent keyed preset sections")
	}
	for _, key := range []string{"lfm-8k", "lfm-32k"} {
		if !modelSettingsSamePath(sections[key]["model"], filepath.Join(f.root, "lfm.gguf")) {
			t.Fatal("sibling preset sections do not reference the single fixture GGUF")
		}
	}
	client := &http.Client{Timeout: 15 * time.Second, Transport: &http.Transport{Proxy: nil}}
	t.Cleanup(client.CloseIdleConnections)
	var cold struct {
		Data []struct {
			ID     string `json:"id"`
			Status struct {
				Value string `json:"value"`
			} `json:"status"`
		} `json:"data"`
	}
	modelSettingsDecode(t, egressProcessHTTP(t, client, http.MethodGet, f.targetPort, "/models", nil, http.StatusOK), &cold)
	if len(cold.Data) != 2 || cold.Data[0].ID != "lfm-32k" || cold.Data[1].ID != "lfm-8k" || cold.Data[0].Status.Value != "unloaded" || cold.Data[1].Status.Value != "unloaded" {
		t.Fatal("synthetic router did not expose both siblings as cold")
	}
	// The private manual overlay identifies this facade as self. Its policy and
	// ordered role candidates still come from the real broker publication.
	f.rpcOK(t, "llamacpp-proxy:node/add-manual", map[string]any{"id": "profile-roles-self", "host": "127.0.0.1", "port": f.targetFacadePort, "addresses": []string{"127.0.0.1"}, "models": []string{"lfm-8k", "lfm-32k"}})
	f.rpcOK(t, "llamacpp-proxy:node/select", map[string]string{"id": "profile-roles-self"})
	task1Eventually(t, "ordered role catalog with derived maximum context", 15*time.Second, func() bool {
		response, err := client.Get("http://127.0.0.1:" + strconv.Itoa(f.targetFacadePort) + "/v1/models")
		if err != nil {
			return false
		}
		var catalog struct {
			Data []struct {
				ID            string `json:"id"`
				ContextLength *int   `json:"context_length"`
			} `json:"data"`
		}
		decodeErr := json.NewDecoder(response.Body).Decode(&catalog)
		if err := response.Body.Close(); err != nil {
			t.Fatal(err)
		}
		if response.StatusCode != http.StatusOK || decodeErr != nil {
			return false
		}
		for _, model := range catalog.Data {
			if model.ID == "weak" {
				return model.ContextLength != nil && *model.ContextLength == 32768
			}
		}
		return false
	})
	// Engine start completes before the broker's asynchronous healthy-backend
	// publication reaches the facade. Await its real read-only forwarding path.
	task1Eventually(t, "healthy synthetic engine through facade", 15*time.Second, func() bool {
		response, err := client.Get("http://127.0.0.1:" + strconv.Itoa(f.targetFacadePort) + "/props")
		if err != nil {
			return false
		}
		var props struct {
			Role string `json:"role"`
		}
		decodeErr := json.NewDecoder(response.Body).Decode(&props)
		if err := response.Body.Close(); err != nil {
			t.Fatal(err)
		}
		return response.StatusCode == http.StatusOK && decodeErr == nil && props.Role == "router"
	})
	egressProcessHTTP(t, client, http.MethodPost, f.targetFacadePort, "/v1/chat/completions", []byte(`{"model":"weak","stream":false,"messages":[{"role":"user","content":"synthetic"}]}`), http.StatusOK)
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.inferences != 1 || !slices.Equal(f.servedModels, []string{"lfm-8k"}) || f.mutations != 0 {
		t.Fatalf("cold role selection effects: served=%v inference=%d mutation=%d", f.servedModels, f.inferences, f.mutations)
	}
}

func TestModelSettingsLastTypedRemovalRecoversThenStartsSourcePreset(t *testing.T) {
	f := newProfileRolesProcessFixture(t)
	profileRolesCreate(t, f, "lfm-8k", 8192)
	policy := f.policy(t)
	policy.Profiles[0].MaxOutputTokens = 512
	policy.Profiles[0].MaxConcurrent = 1
	policy.Profiles[0].MemoryBytes = 1048576
	policy.Profiles[0].Qualification = &nodepolicy.QualificationRef{ID: "saved-removal-reference", Revision: "1"}
	policy.Roles = []nodepolicy.Role{{Name: "weak", Profiles: []string{"lfm-8k"}}}
	policy.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: "lfm-8k"}}
	policy.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "lfm-8k"}}
	policy = f.setPolicy(t, policy)
	state := f.snapshot(t)
	request, preview := f.review(t, settings.Request{
		Engine: "llamacpp", RequestID: "remove-last-profile", ExpectedRevision: state.Revision, Settings: state.Settings.Clone(),
		ModelChange: &modelconfig.ModelChange{ProfileKey: "lfm-8k", ModelID: "lfm-8k", Remove: true, ExpectedPolicyRevision: policy.Revision},
	})
	if len(preview.Settings.ModelLoads) != 0 || preview.Settings.Serving != nil || preview.Serving != nil || len(preview.Identities) != 0 {
		t.Fatal("last-remove review retained a typed managed declaration")
	}
	gate := newModelSettingsGate()
	f.mu.Lock()
	f.readbackGate = gate
	f.mu.Unlock()
	f.begin(t, "engine:apply-settings", request)
	modelSettingsWait(t, gate)
	if f.disk(t).Model.Stage != "native-applying" || !reflect.DeepEqual(policy, f.policy(t)) {
		t.Fatal("last removal published policy before accepted native readback")
	}
	// Read the real manager's persisted override while its reply is held. The
	// broker is then killed between native persistence and policy publication.
	assertProfileRolesNoManagedLoads(t, f)
	f.crashBroker(t, gate)
	f.startModelSettings(t)
	after := f.settled(t, request.RequestID, "succeeded")
	assertModelSettingsSavedReadback(t, after, preview.Settings)
	want := policy
	want.Profiles = []nodepolicy.Profile{}
	want.Roles = []nodepolicy.Role{{Name: "weak", Profiles: []string{}}}
	want.ServedModels, want.ResidentModels = nodepolicy.ServedModels{}, nil
	removed := f.policy(t)
	want.Revision = removed.Revision
	if !reflect.DeepEqual(want, removed) || removed.Revision == policy.Revision || f.disk(t).Model.Stage != "completed" {
		t.Fatal("recovered typed removal retained a profile, membership, reference or unfinished journal")
	}
	var replay struct {
		Revision uint64 `json:"revision"`
		Phase    string `json:"phase"`
	}
	modelSettingsDecode(t, f.rpcOK(t, "engine:apply-settings", request), &replay)
	if replay.Revision != after.Revision || replay.Phase != "succeeded" {
		t.Fatal("recovered removal lost its accepted idempotent receipt")
	}
	assertProfileRolesNoManagedLoads(t, f)
	// The next start must use the ordinary model-source owner. The source is
	// private HOME content, never an installed engine or model directory.
	source := filepath.Join(f.base, ".lmstudio", "models", "source.gguf")
	modelSettingsWrite(t, source, modelSettingsGGUF("blk.0.weight"))
	f.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
	f.engineIdentity(t, 1)
	wantPreset := filepath.Join(jointAppDir(f.base), "engine-bin", "model-sources", "llamacpp.ini")
	capturedPath := string(readLiveDurableFile(t, f.capture+".path"))
	if !modelSettingsSamePath(capturedPath, wantPreset) {
		t.Fatalf("next start selected %s instead of private model-sources preset %s", capturedPath, wantPreset)
	}
	sections := modelSettingsPreset(t, readLiveDurableFile(t, f.capture))
	if len(sections) != 1 || !modelSettingsSamePath(sections["lmstudio/source"]["model"], source) || len(sections["lfm-8k"]) != 0 {
		t.Fatal("next start retained an empty managed preset or lost the private source model")
	}
	if current := f.snapshot(t); !current.Running || len(current.Settings.ModelLoads) != 0 || current.Settings.Serving != nil {
		t.Fatal("default source start recreated removed typed settings")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.configureCalls != 2 || f.mutations != 0 || f.inferences != 0 {
		t.Fatalf("create/removal/recovery effects: configure=%d mutation=%d inference=%d", f.configureCalls, f.mutations, f.inferences)
	}
}

func assertProfileRolesNoManagedLoads(t *testing.T, f *modelSettingsProcessFixture) {
	t.Helper()
	var override struct {
		Platforms map[string]struct {
			Runtime struct {
				ModelLoads []modelconfig.ModelLoad `json:"modelLoads"`
				Serving    json.RawMessage         `json:"serving"`
				LaunchArgs []string                `json:"launch_args"`
			} `json:"runtime"`
		} `json:"platforms"`
	}
	modelSettingsDecode(t, readLiveDurableFile(t, filepath.Join(jointAppDir(f.base), "engines", "llamacpp.json")), &override)
	platform, ok := override.Platforms[runtime.GOOS+"/"+runtime.GOARCH]
	if !ok || platform.Runtime.ModelLoads == nil || len(platform.Runtime.ModelLoads) != 0 || string(platform.Runtime.Serving) != "null" {
		t.Fatal("persisted final removal lacks an explicit empty load mask and cleared serving declaration")
	}
	for _, arg := range platform.Runtime.LaunchArgs {
		if arg == "--models-preset" || arg == "--models-max" {
			t.Fatal("persisted default launch retained a managed preset marker")
		}
	}
	if len(platform.Runtime.LaunchArgs) != 0 {
		t.Fatalf("default launch retained arguments: %v", platform.Runtime.LaunchArgs)
	}
}
