// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
)

// Registered before TestMain's normal builds. The existing fixed package
// executable is the child; no new executable path needs firewall registration.
func runModelLoadChild() bool {
	if os.Getenv("PAIR_TYPED_LOAD_CHILD") == "uuid" {
		if err := json.NewEncoder(os.Stdout).Encode(map[string]string{"uuid": os.Getenv("CUDA_VISIBLE_DEVICES"), "mps": os.Getenv("CUDA_MPS_PIPE_DIRECTORY")}); err != nil {
			os.Exit(2)
		}
		return true
	}
	if os.Getenv("PAIR_TYPED_LOAD_CHILD") != "1" {
		return false
	}
	index := slices.Index(os.Args, "--models-preset")
	if index < 0 || index+1 >= len(os.Args) {
		os.Exit(2)
	}
	data, err := os.ReadFile(os.Args[index+1])
	if err != nil {
		os.Exit(2)
	}
	sections, err := parseLlamaServingPreset(data)
	if err != nil {
		os.Exit(2)
	}
	if sections["medium-model"]["ctx-size"] != "16384" || sections["medium-model"]["parallel"] != "2" {
		os.Exit(2)
	}
	for _, options := range sections {
		model := options["model"]
		if !filepath.IsAbs(model) {
			model = filepath.Join(filepath.Dir(os.Args[index+1]), model)
		}
		if _, err := os.Stat(model); err != nil {
			os.Exit(2)
		}
	}
	capture, err := json.Marshal(struct {
		Args   []string
		Preset []byte
	}{os.Args[1:], data})
	if err != nil {
		os.Exit(2)
	}
	if err := os.WriteFile(os.Getenv("PAIR_TYPED_CAPTURE"), capture, 0o600); err != nil {
		os.Exit(2)
	}
	port := slices.Index(os.Args, "--port")
	if port < 0 || port+1 >= len(os.Args) {
		os.Exit(2)
	}
	address := net.JoinHostPort("127.0.0.1", os.Args[port+1])
	listener, err := net.Listen("tcp", address)
	if err != nil {
		os.Exit(2)
	}
	server := http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/" {
			http.Error(w, "unexpected fixture request", http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusOK)
	})}
	if err := server.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
		os.Exit(2)
	}
	return true
}

type modelLoadFixture struct {
	ex                    *Executor
	st                    *engineState
	rt                    Runtime
	command               launchCommand
	root, source, capture string
}

func newModelLoadFixture(t *testing.T) modelLoadFixture {
	t.Helper()
	f := newLlamaServingFixture(t)
	st := settingsState(t, f.ex)
	st.modelsDir = f.root
	bin, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	st.binPath = bin
	st.plat.Detect = []string{bin}
	st.plat.DetectRequired = nil
	rt := st.plat.Runtime
	rt.Bin = bin
	rt.Env = map[string]string{"LLAMA_CACHE": f.root, "PAIR_TYPED_LOAD_CHILD": "1", "PAIR_TYPED_CAPTURE": filepath.Join(f.root, "capture.json")}
	for _, id := range []string{"medium", "weak"} {
		writeInspectionGGUF(t, filepath.Join(f.root, id+".gguf"), "llama", 0.25, 4096)
	}
	writeTestFile(t, f.source, []byte("[*]\nfit = off\nload-on-startup = false\n[weak-model]\nmodel = weak.gguf\nctx-size = 4096\nparallel = 1\ngpu-layers = 0\n[medium-model]\nmodel = medium.gguf\nctx-size = 16384\nparallel = 2\ngpu-layers = 0\n"))
	args := []string{"--models-preset", f.source, "--models-max", "1", "--log-disable"}
	rt.LaunchArgs, rt.Serving, rt.modelLoadOwner = &args, nil, f.ex
	st.plat.Runtime = rt
	command, err := launchForState(st, rt.Port)
	if err != nil {
		t.Fatal(err)
	}
	return modelLoadFixture{f.ex, st, rt, command, f.root, f.source, filepath.Join(f.root, "capture.json")}
}

func convertModelLoadFixture(t *testing.T, f modelLoadFixture) Runtime {
	t.Helper()
	conversion, err := f.ex.previewTypedLlamaConversion(context.Background(), f.st, f.command, f.rt)
	if err != nil {
		t.Fatal(err)
	}
	if len(conversion.Runtime.ModelLoads) != 2 || len(conversion.SourceDigest) != 64 || len(conversion.Diff) == 0 {
		t.Fatalf("incomplete conversion: %+v", conversion)
	}
	for _, load := range conversion.Runtime.ModelLoads {
		if load.ProfileKey != load.ModelID {
			t.Fatalf("conversion renamed section %q to %q", load.ModelID, load.ProfileKey)
		}
	}
	return conversion.Runtime
}

func twoProfileModelLoadFixture(t *testing.T) (modelLoadFixture, Runtime) {
	t.Helper()
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	model := modelconfig.CloneLoads(rt.ModelLoads[:1])[0]
	if model.ModelID != "medium-model" {
		t.Fatalf("fixture artifact selector = %q, want medium-model", model.ModelID)
	}
	rt.ModelLoads = nil
	for _, profile := range []struct {
		key     string
		context int32
	}{{"lfm-8k", 8192}, {"lfm-32k", 32768}} {
		load := modelconfig.CloneLoads([]modelconfig.ModelLoad{model})[0]
		load.ProfileKey = profile.key
		load.Requested.Context = &modelconfig.Context{Mode: "fixed", Tokens: profile.context}
		load.Requested.Slots = keyTestPointer(int32(1))
		rt.ModelLoads = append(rt.ModelLoads, load)
	}
	prepared, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, rt), rt, false)
	if err != nil {
		t.Fatalf("compile sibling profiles: %v", err)
	}
	rt.ModelLoads = prepared.ModelLoads
	return f, rt
}

func declareTwoProfileLoads(t *testing.T, f modelLoadFixture, rt *Runtime) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(f.root, "medium.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	rt.Serving = &engines.ServingConfig{Version: 1}
	// Declaration order intentionally differs from canonical profile order.
	for _, profile := range []struct {
		key    string
		memory int64
	}{{"lfm-8k", 4096}, {"lfm-32k", 8192}} {
		host, vram := profile.memory, int64(0)
		rt.Serving.Models = append(rt.Serving.Models, engines.ServingModel{
			ModelID: profile.key, MaxOutputTokens: 128,
			Memory:    engines.ServingMemory{HostRAMBytes: &host, VRAMBytes: &vram},
			Artifacts: []engines.ServingArtifact{{Role: "model", SHA256: sha256Hex(data), Provenance: engines.ServingProvenance{Kind: "supplied-sha256"}}},
		})
	}
}

func TestTypedPresetTwoProfilesOneModel(t *testing.T) {
	f, rt := twoProfileModelLoadFixture(t)
	declareTwoProfileLoads(t, f, &rt)
	prepared, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, rt), rt, false)
	if err != nil {
		t.Fatal(err)
	}
	sections, err := parseLlamaServingPreset(prepared.Snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(sortedKeys(sections), []string{"lfm-32k", "lfm-8k"}) {
		t.Fatalf("preset sections = %v", sortedKeys(sections))
	}
	path := filepath.Join(f.root, "medium.gguf")
	for _, profile := range []struct {
		key     string
		context int32
		memory  int64
	}{{"lfm-8k", 8192, 4096}, {"lfm-32k", 32768, 8192}} {
		options := sections[profile.key]
		if options["model"] != path || options["ctx-size"] != strconv.Itoa(int(profile.context)) || options["parallel"] != "1" {
			t.Fatalf("profile %s options = %v", profile.key, options)
		}
		resolved := rtLoadResolution(prepared.ModelLoads, profile.key)
		if resolved == nil || resolved.PerSlotContext != profile.context || resolved.TotalContext != profile.context || resolved.Slots != 1 || len(resolved.Artifacts) != 1 || resolved.Artifacts[0].Path != path {
			t.Fatalf("profile %s resolution = %+v", profile.key, resolved)
		}
		list, exists := prepared.evidence.typed.models[profile.key]
		if !exists || list.Model != "medium-model" || len(list.Files) != 1 || list.Files[0].configured != path {
			t.Fatalf("profile %s evidence = %+v, exists=%t", profile.key, list, exists)
		}
		index := slices.IndexFunc(prepared.Envelopes, func(envelope engines.ServingEnvelope) bool { return envelope.ModelID == profile.key })
		if index < 0 {
			t.Fatalf("profile %s envelope missing: %+v", profile.key, prepared.Envelopes)
		}
		envelope := prepared.Envelopes[index]
		if envelope.PerSlotContextTokens != int64(profile.context) || envelope.TotalContextTokens != int64(profile.context) || envelope.Memory.HostRAMBytes == nil || *envelope.Memory.HostRAMBytes != profile.memory {
			t.Fatalf("profile %s envelope = %+v", profile.key, envelope)
		}
	}
	if len(prepared.ModelLoads) != 2 || len(prepared.Envelopes) != 2 || len(prepared.evidence.typed.models) != 2 {
		t.Fatalf("lost sibling profiles: loads=%+v envelopes=%+v evidence=%v", prepared.ModelLoads, prepared.Envelopes, sortedKeys(prepared.evidence.typed.models))
	}
	for _, load := range prepared.ModelLoads {
		if load.ModelID != "medium-model" {
			t.Fatalf("profile %s artifact selector = %q", load.ProfileKey, load.ModelID)
		}
	}
	if !reflect.DeepEqual(prepared.ModelLoads[0].Resolved.Artifacts, prepared.ModelLoads[1].Resolved.Artifacts) {
		t.Fatal("siblings did not retain identical artifact assertions")
	}
	rt.ModelLoads, rt.Serving, rt.servingLaunch = prepared.ModelLoads, prepared.Config, prepared
	f.st.plat.Runtime = rt
	f.st.runID, f.st.launchDigest = "profiles-run", prepared.Digest
	identity := servingIdentityLocked(f.st)
	if !slices.Equal(sortedKeys(identity.Serving), []string{"lfm-32k", "lfm-8k"}) || identity.Serving["lfm-8k"].PerSlotContextTokens != 8192 || identity.Serving["lfm-32k"].PerSlotContextTokens != 32768 || identity.ModelDigests["lfm-8k"] == "" || identity.ModelDigests["lfm-8k"] != identity.ModelDigests["lfm-32k"] {
		t.Fatalf("served profile identities = %+v", identity)
	}
	if _, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, rt), rt, true); err != nil {
		t.Fatalf("frozen sibling evidence refused: %v", err)
	}
}

func TestTypedProfileUnpricedSiblingKeepsFiniteEnvelope(t *testing.T) {
	test := func(name string, reverse bool) {
		t.Run(name, func(t *testing.T) {
			f, rt := twoProfileModelLoadFixture(t)
			declareTwoProfileLoads(t, f, &rt)
			if reverse {
				slices.Reverse(rt.Serving.Models)
			}
			for i := range rt.ModelLoads {
				if rt.ModelLoads[i].ProfileKey == "lfm-8k" {
					rt.ModelLoads[i].Requested.Fit = keyTestPointer("on")
				}
			}
			prepared, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, rt), rt, false)
			if err != nil {
				t.Fatal(err)
			}
			envelopes := servingEnvelopes(prepared)
			if len(envelopes) != 1 || envelopes[0].ModelID != "lfm-32k" || envelopes[0].PerSlotContextTokens != 32768 || envelopes[0].Memory.HostRAMBytes == nil || *envelopes[0].Memory.HostRAMBytes != 8192 || envelopes[0].ContentDigest == "" {
				t.Fatalf("finite sibling envelope = %+v", envelopes)
			}
		})
	}
	test("unpriced declaration first", false)
	test("finite declaration first", true)
}

func TestTypedProfilesRefuseArtifactDisagreement(t *testing.T) {
	test := func(name string, edit func(*testing.T, modelLoadFixture, *Runtime)) {
		t.Run(name, func(t *testing.T) {
			f, rt := twoProfileModelLoadFixture(t)
			edit(t, f, &rt)
			_, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, rt), rt, false)
			var typed *ServingFieldError
			if !errors.As(err, &typed) || typed.Reason != "Profiles of one model disagree on artifacts." {
				t.Fatalf("disagreeing sibling assertions error = %v", err)
			}
		})
	}
	test("stored sampled bindings differ", func(t *testing.T, f modelLoadFixture, rt *Runtime) {
		rt.ModelLoads[1].Resolved.Artifacts[0].SampleSHA256 = strings.Repeat("f", 64)
	})
	test("declared content assertions differ", func(t *testing.T, f modelLoadFixture, rt *Runtime) {
		declareTwoProfileLoads(t, f, rt)
		rt.Serving.Models[1].Artifacts[0].SHA256 = strings.Repeat("f", 64)
	})
}

func TestTypedProfileGPUSelectionRefusesSiblingAutoBeforeNative(t *testing.T) {
	f, rt := twoProfileModelLoadFixture(t)
	for i := range rt.ModelLoads {
		load := &rt.ModelLoads[i]
		if load.ProfileKey == "lfm-8k" {
			load.Requested.GPU, load.Resolved = &modelconfig.GPU{Mode: "auto"}, nil
		} else {
			load.Requested.ExtraOptions = map[string]string{"device": "CPU"}
		}
	}
	memory := &autoTestMemoryOwner{}
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: memory})
	command := typedFixtureCommand(t, f, rt)
	command.Bin = filepath.Join(f.root, "missing-native-executable")
	_, err := compileTypedLlamaInputs(ctx, f.st, command, rt, false)
	var typed *ServingFieldError
	if !errors.As(err, &typed) || typed.Reason != "lfm-32k selects GPUs and cannot share an engine with reviewed Auto profile lfm-8k; set that profile's GPU mode to layers." || memory.calls != 0 {
		t.Fatalf("sibling GPU refusal = %v, memory calls=%d", err, memory.calls)
	}
}

func TestTypedLoadArtifactIDIsNotASection(t *testing.T) {
	test := func(name, model string) {
		t.Run(name, func(t *testing.T) {
			f, rt := twoProfileModelLoadFixture(t)
			index := slices.IndexFunc(rt.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == "lfm-8k" })
			rt.ModelLoads = modelconfig.CloneLoads(rt.ModelLoads[index : index+1])
			rt.ModelLoads[0].ModelID = model
			prepared, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, rt), rt, false)
			if err != nil {
				t.Fatalf("artifact selector %q refused: %v", model, err)
			}
			sections, err := parseLlamaServingPreset(prepared.Snapshot)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(sortedKeys(sections), []string{"lfm-8k"}) || sections["lfm-8k"]["model"] != filepath.Join(f.root, "medium.gguf") || prepared.ModelLoads[0].ModelID != model || prepared.evidence.typed.models["lfm-8k"].Model != model {
				t.Fatalf("artifact selector changed section or resolution: sections=%v loads=%+v evidence=%+v", sections, prepared.ModelLoads, prepared.evidence.typed.models)
			}
		})
	}
	test("reserved alias is an artifact selector", "coder:local")
	test("artifact tag is kept exact", "org/model:UD-Q4_K_XL")
}

func TestTypedLlamaExportAllowsIdenticalArtifactSiblings(t *testing.T) {
	_, rt := twoProfileModelLoadFixture(t)
	if err := validateTypedLlamaExport(rt, "medium-model"); err != nil {
		t.Fatalf("identical artifact siblings refused: %v", err)
	}
	rt.ModelLoads[1].Resolved.Artifacts[0].SampleSHA256 = strings.Repeat("f", 64)
	if err := validateTypedLlamaExport(rt, "medium-model"); !errors.Is(err, errNoArtifactBinding) {
		t.Fatalf("disagreeing export assertions error = %v", err)
	}
}

func TestModelLoadPatchKeepsProfileModel(t *testing.T) {
	loads := []modelconfig.ModelLoad{
		{ProfileKey: "lfm-8k", ModelID: "medium-model", Requested: modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "fixed", Tokens: 8192}}},
		{ProfileKey: "lfm-32k", ModelID: "medium-model", Requested: modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "fixed", Tokens: 32768}}},
	}
	before := modelconfig.CloneLoads(loads)
	threads := int32(4)
	patched, err := patchModelLoad(loads, "lfm-8k", "medium-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Threads: &threads}})
	if err != nil {
		t.Fatal(err)
	}
	if len(patched) != 2 || patched[0].ProfileKey != "lfm-32k" || patched[1].ProfileKey != "lfm-8k" || patched[1].ModelID != "medium-model" || patched[1].Requested.Threads == nil || *patched[1].Requested.Threads != 4 || patched[1].Requested.Context.Tokens != 8192 || !reflect.DeepEqual(patched[0], before[1]) {
		t.Fatalf("patch changed profile identity or sibling: %+v", patched)
	}
	if !reflect.DeepEqual(loads, before) {
		t.Fatal("patch mutated saved inputs")
	}
	refused, err := patchModelLoad(loads, "lfm-8k", "other-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Threads: &threads}})
	var typed *ServingFieldError
	if refused != nil || !errors.As(err, &typed) || typed.Reason != "A profile keeps its model; create a new profile." || !reflect.DeepEqual(loads, before) {
		t.Fatalf("retarget result = %+v, error = %v", refused, err)
	}
	created, err := patchModelLoad(loads, "lfm-64k", "coder:local", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "fixed", Tokens: 65536}}})
	if err != nil {
		t.Fatalf("create missing load: %v", err)
	}
	if len(created) != 3 || created[1].ProfileKey != "lfm-64k" || created[1].ModelID != "coder:local" || created[1].Requested.Context.Tokens != 65536 || created[1].Resolved != nil {
		t.Fatalf("created profile target = %+v", created)
	}
}

func TestModelLoadRemoveProfile(t *testing.T) {
	test := func(name string, last bool) {
		t.Run(name, func(t *testing.T) {
			f, rt := twoProfileModelLoadFixture(t)
			declareTwoProfileLoads(t, f, &rt)
			if last {
				index := slices.IndexFunc(rt.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == "lfm-8k" })
				rt.ModelLoads = modelconfig.CloneLoads(rt.ModelLoads[index : index+1])
				rt.Serving.Models = slices.DeleteFunc(rt.Serving.Models, func(model engines.ServingModel) bool { return model.ModelID != "lfm-8k" })
			}
			prepared, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, rt), rt, false)
			if err != nil {
				t.Fatal(err)
			}
			rt.ModelLoads, rt.Serving = prepared.ModelLoads, prepared.Config
			f.st.plat.Runtime = rt
			before := modelconfig.CloneLoads(rt.ModelLoads)
			beforeServing := rt.Serving.Clone()
			request := settingsRequest(t, f.ex)
			request.ModelChange = &modelconfig.ModelChange{ProfileKey: "lfm-8k", ModelID: "medium-model", Remove: true, ExpectedPolicyRevision: "0"}
			preview, err := f.ex.PreviewLaunch(request)
			if err != nil || len(preview.Errors) != 0 {
				t.Fatalf("remove preview errors=%v, err=%v", preview.Errors, err)
			}
			if len(preview.PreviewDigest) != 64 || preview.Impact == nil || preview.Impact.RestartScope != "router" || preview.Restart || !modelconfig.EqualLoads(before, f.st.plat.Runtime.ModelLoads) || !beforeServing.Equal(f.st.plat.Runtime.Serving) {
				t.Fatalf("removal preview or read-only state = %+v", preview)
			}
			if _, exists := preview.Identities["lfm-8k"]; exists {
				t.Fatal("removed profile retained an identity")
			}
			if last {
				proposed := rt
				proposed.ModelLoads, proposed.Serving = preview.Settings.ModelLoads, preview.Settings.Serving
				if len(proposed.ModelLoads) != 0 || proposed.Serving != nil || preview.Serving != nil || preparedServingRuntime(f.st, proposed) || len(preview.Identities) != 0 || !slices.Equal(preview.Impact.AffectedModels, []string{"lfm-8k"}) {
					t.Fatalf("last removal did not return to default mode: %+v", preview)
				}
			} else if len(preview.Settings.ModelLoads) != 1 || preview.Settings.ModelLoads[0].ProfileKey != "lfm-32k" || preview.Settings.ModelLoads[0].ModelID != "medium-model" || preview.Settings.ModelLoads[0].Resolved.PerSlotContext != 32768 || preview.Settings.Serving == nil || len(preview.Settings.Serving.Models) != 1 || preview.Settings.Serving.Models[0].ModelID != "lfm-32k" || len(preview.Identities) != 0 || !slices.Equal(preview.Impact.AffectedModels, []string{"lfm-32k", "lfm-8k"}) {
				t.Fatalf("removal changed sibling: %+v", preview)
			}
			found := false
			for _, diff := range preview.Diff {
				if diff.Field != "models.lfm-8k.ctx-size" {
					continue
				}
				var old string
				var next *string
				if err := json.Unmarshal(diff.Before, &old); err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(diff.After, &next); err != nil {
					t.Fatal(err)
				}
				if old != "8192" || next != nil {
					t.Fatalf("removal context diff = %q -> %v", old, next)
				}
				found = true
			}
			if !found {
				t.Fatal("removed profile has no field diff")
			}
		})
	}
	test("keeps sibling of same artifact", false)
	test("last profile returns to default mode", true)
	t.Run("mismatched artifact is refused", func(t *testing.T) {
		loads := []modelconfig.ModelLoad{{ProfileKey: "lfm-8k", ModelID: "medium-model"}}
		before := modelconfig.CloneLoads(loads)
		removed, err := removeModelLoad(loads, "lfm-8k", "other-model")
		var typed *ServingFieldError
		if removed != nil || !errors.As(err, &typed) || typed.Reason != "A profile keeps its model; create a new profile." || !reflect.DeepEqual(loads, before) {
			t.Fatalf("mismatched removal = %+v, err=%v", removed, err)
		}
	})
}

func TestModelLoadPreviewUsesSelectedProfileIdentity(t *testing.T) {
	test := func(name, key string, tokens int32, create bool) {
		t.Run(name, func(t *testing.T) {
			f, rt := twoProfileModelLoadFixture(t)
			if create {
				index := slices.IndexFunc(rt.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == "lfm-8k" })
				rt.ModelLoads = modelconfig.CloneLoads(rt.ModelLoads[index : index+1])
			}
			f.st.plat.Runtime = rt
			before := modelconfig.CloneLoads(rt.ModelLoads)
			request := settingsRequest(t, f.ex)
			request.ModelChange = &modelconfig.ModelChange{ProfileKey: key, ModelID: "medium-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "fixed", Tokens: tokens}}}}
			preview, err := f.ex.PreviewLaunch(request)
			if err != nil || len(preview.Errors) != 0 {
				t.Fatalf("selected preview errors=%v, err=%v", preview.Errors, err)
			}
			identity := preview.Identities[key]
			if !slices.Equal(sortedKeys(preview.Identities), []string{key}) || identity.ConfiguredContextTokens != int64(tokens) || len(preview.Settings.ModelLoads) != 2 || len(identity.Files) != 1 || identity.Files[0].Name != "medium.gguf" {
				t.Fatalf("profile identity map = %+v, loads=%+v", preview.Identities, preview.Settings.ModelLoads)
			}
			if !modelconfig.EqualLoads(before, f.st.plat.Runtime.ModelLoads) {
				t.Fatal("preview persisted the selected change")
			}
		})
	}
	test("existing profile has its own context", "lfm-8k", 8192, false)
	test("existing sibling has its own context", "lfm-32k", 32768, false)
	test("missing sibling uses artifact selector", "lfm-32k", 32768, true)
}

func TestModelLoadCreatesInheritedProfileWithoutBorrowingSibling(t *testing.T) {
	f, rt := twoProfileModelLoadFixture(t)
	f.st.plat.Runtime = rt
	before := modelconfig.CloneLoads(rt.ModelLoads)
	request := settingsRequest(t, f.ex)
	request.ModelChange = &modelconfig.ModelChange{ProfileKey: "lfm-inherited", ModelID: "medium-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Clear: []string{"context"}}}
	preview, err := f.ex.PreviewLaunch(request)
	if err != nil || len(preview.Errors) != 0 {
		t.Fatalf("inherited profile preview errors=%v, err=%v", preview.Errors, err)
	}
	index := slices.IndexFunc(preview.Settings.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == "lfm-inherited" })
	if index < 0 {
		t.Fatal("inherited profile was not compiled")
	}
	load := preview.Settings.ModelLoads[index]
	if len(preview.Settings.ModelLoads) != 3 || load.ModelID != "medium-model" || load.Requested.Context != nil || load.Requested.Slots != nil || load.Resolved == nil || load.Resolved.PerSlotContext != 8192 || load.Resolved.Slots != 1 || preview.Identities["lfm-inherited"].ConfiguredContextTokens != 8192 {
		t.Fatalf("new profile did not inherit canonical defaults: %+v", load)
	}
	if !modelconfig.EqualLoads(before, f.st.plat.Runtime.ModelLoads) {
		t.Fatal("inherited preview persisted changes")
	}
}

func TestModelLoadConversionPreviewIncludesEveryProfileIdentity(t *testing.T) {
	f := newModelLoadFixture(t)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "fixed", Tokens: 6144}}})
	if !slices.Equal(sortedKeys(preview.Identities), []string{"medium-model", "weak-model"}) {
		t.Fatalf("converted identities = %+v", preview.Identities)
	}
	if len(preview.Settings.ModelLoads) != 2 {
		t.Fatalf("converted loads = %+v", preview.Settings.ModelLoads)
	}
	for _, want := range []struct {
		key, file string
		context   int64
	}{
		{"medium-model", "medium.gguf", 8192},
		{"weak-model", "weak.gguf", 6144},
	} {
		t.Run(want.key, func(t *testing.T) {
			identity := preview.Identities[want.key]
			if identity.ConfiguredContextTokens != want.context || len(identity.Files) != 1 || identity.Files[0].Role != "model" || identity.Files[0].Name != want.file {
				t.Fatalf("converted identity = %+v, want file %s and context %d", identity, want.file, want.context)
			}
			index := slices.IndexFunc(preview.Settings.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == want.key })
			if index < 0 {
				t.Fatal("converted profile missing")
			}
			load := preview.Settings.ModelLoads[index]
			if load.ModelID != want.key || load.Resolved == nil || int64(load.Resolved.PerSlotContext) != want.context {
				t.Fatalf("converted load = %+v", load)
			}
		})
	}
	if len(f.st.plat.Runtime.ModelLoads) != 0 {
		t.Fatal("conversion preview persisted typed loads")
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestTypedLlamaNativeFit(t *testing.T) {
	f := newAppliedKeyFixture(t)
	rt := f.st.plat.Runtime
	on, margin, floor := "on", "512", int32(4096)
	rt.ModelLoads[0].Requested.Fit = &on
	rt.ModelLoads[0].Requested.FitTarget = &margin
	rt.ModelLoads[0].Requested.FitCtx = &floor
	command, err := launchForState(f.st, rt.Port)
	if err != nil {
		t.Fatal(err)
	}
	var inputs map[string]modelEstimateInput
	prepared, err := compileTypedLlamaInputs(context.Background(), f.st, command, rt, false, &inputs)
	if err != nil {
		t.Fatal(err)
	}
	sections, err := parseLlamaServingPresetAllowing(prepared.Snapshot, modelLoadAllowedPresetKeys(prepared.ModelLoads))
	if err != nil {
		t.Fatal(err)
	}
	section := sections[f.request.ModelID]
	if section["fit"] != "on" || section["fit-target"] != "512" || section["fit-ctx"] != "4096" {
		t.Fatalf("native fit options=%v", section)
	}
	rt.ModelLoads = prepared.ModelLoads
	f.st.plat.Runtime = rt
	result := inspectAppliedKeyFixture(t, f)
	if result.ConfigurationKeyReason != "native-fit" || result.ConfigurationKey != "" || result.ExecutionKey == "" || len(result.ConfigurationPools) != 0 {
		t.Fatalf("native fit identity=%+v", result)
	}
	if result.Estimate.Status != "unknown" || !slices.Contains(result.Estimate.Unknowns, "Native fit chooses placement when the model loads.") {
		t.Fatalf("native fit estimate=%+v", result.Estimate)
	}
	if err := result.Validate(); err != nil {
		t.Fatal(err)
	}
	capture := f.ex.newModelBufferCapture(f.st, "llamacpp", prepared, inputs, command, keyTestMemoryContext())
	if capture != nil {
		defer capture.retire()
		if capture.frozen[f.request.ModelID] != nil {
			t.Fatal("native fit received finite buffer capture")
		}
	}
	changedMargin := "1024"
	rt.ModelLoads[0].Requested.FitTarget = &changedMargin
	changedPrepared, err := compileTypedLlamaInputs(context.Background(), f.st, command, rt, false)
	if err != nil {
		t.Fatal(err)
	}
	rt.ModelLoads = changedPrepared.ModelLoads
	f.st.plat.Runtime = rt
	changed := inspectAppliedKeyFixture(t, f)
	if changed.ExecutionKey == "" || changed.ExecutionKey == result.ExecutionKey {
		t.Fatalf("changed margin did not change execution identity: before=%s after=%s", result.ExecutionKey, changed.ExecutionKey)
	}
}

func TestNativeOptionsGPUSelectionInheritedPreviewRefusesBeforeHelp(t *testing.T) {
	for _, inherited := range []string{"Auto", "selection"} {
		t.Run(inherited, func(t *testing.T) {
			f := newModelLoadFixture(t)
			rt := convertModelLoadFixture(t, f)
			load := &rt.ModelLoads[0]
			load.Resolved = nil
			selected := modelconfig.RequestedLoad{}
			if inherited == "Auto" {
				load.Requested.GPU = &modelconfig.GPU{Mode: "auto"}
				selected.ExtraOptions = map[string]string{"tensor-split": "1,1"}
			} else {
				load.Requested.ExtraOptions = map[string]string{"tensor-split": "1,1"}
				selected.GPU = &modelconfig.GPU{Mode: "auto"}
			}
			f.st.plat.Runtime = rt
			marker, _ := nativeOptionsFixture(t, f.ex, f.st)
			request := settingsRequest(t, f.ex)
			request.ModelChange = &modelconfig.ModelChange{ModelID: load.ModelID, ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: selected}}
			memory := &autoTestMemoryOwner{}
			ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: memory})
			preview, err := f.ex.PreviewLaunch(request, ctx)
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, message := range preview.Errors {
				found = found || strings.Contains(message, "Reviewed Auto chooses the GPU; remove tensor-split or choose another GPU mode.")
			}
			if !found || memory.calls != 0 {
				t.Fatalf("preview errors=%v memory reads=%d", preview.Errors, memory.calls)
			}
			requireNativeHelpSpawns(t, marker, 0)
		})
	}
}

func TestNativeOptionsInspectPublishesLaunchWarnings(t *testing.T) {
	var marker, library string
	f := newAppliedKeyFixture(t, func(f inspectionFixture) {
		marker, library = nativeOptionsFixture(t, f.ex, f.st)
		f.st.plat.Runtime.EditableLaunch.FixedArgs = []string{"--cors-origins", "http://fixture.invalid"}
		f.st.plat.Runtime.Args = append(f.st.plat.Runtime.Args, "--cors-origins", "http://fixture.invalid")
	})
	// Observed running metadata is deliberately unrelated to the selected file.
	f.st.buildID = "running-build-is-not-the-selected-file"
	result := inspectAppliedKeyFixture(t, f)
	if len(result.NativeOptions) != 4 || result.NativeOptions[0].Name != "cache-prompt" || result.NativeOptions[3].Name != "spec-type" || !reflect.DeepEqual(result.NativeOptions[3].Choices, []string{"none", "draft-mtp", "ngram-mod"}) {
		t.Fatalf("public native hints = %+v", result.NativeOptions)
	}
	for _, key := range []string{"cache-ram", "cors-origins", "ctx-size", "fit", "kv-unified", "load-on-startup", "model", "models-preset", "parallel", "sleep-idle-seconds"} {
		if !slices.Contains(result.UnreportedLaunchOptions, key) {
			t.Fatalf("public warning omitted %s: %v", key, result.UnreportedLaunchOptions)
		}
	}
	if !slices.IsSorted(result.UnreportedLaunchOptions) {
		t.Fatalf("warnings not sorted: %v", result.UnreportedLaunchOptions)
	}
	if result.ConfigurationKey == "" {
		t.Fatalf("missing flags refused a finite typed launch: %+v", result)
	}
	if err := result.Validate(); err != nil {
		t.Fatal(err)
	}
	requireNativeHelpSpawns(t, marker, 1)
	again := inspectAppliedKeyFixture(t, f)
	if !reflect.DeepEqual(again.NativeOptions, result.NativeOptions) || !reflect.DeepEqual(again.UnreportedLaunchOptions, result.UnreportedLaunchOptions) {
		t.Fatal("inspection did not reuse the selected build's published evidence")
	}
	requireNativeHelpSpawns(t, marker, 1)
	data, err := os.ReadFile(library)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, library, data)
	info, err := os.Stat(library)
	if err != nil {
		t.Fatal(err)
	}
	touch := info.ModTime().Add(time.Hour)
	if err := os.Chtimes(library, touch, touch); err != nil {
		t.Fatal(err)
	}
	touched := inspectAppliedKeyFixture(t, f)
	if !reflect.DeepEqual(touched.NativeOptions, result.NativeOptions) {
		t.Fatal("same-byte/touched inspection lost native hints")
	}
	requireNativeHelpSpawns(t, marker, 1)
	writeTestFile(t, library, append(data, []byte("new selected library content")...))
	changed := inspectAppliedKeyFixture(t, f)
	if !reflect.DeepEqual(changed.NativeOptions, result.NativeOptions) || !slices.Contains(changed.UnreportedLaunchOptions, "cache-ram") {
		t.Fatalf("changed-build inspection failed to refresh hints and warnings: %+v", changed)
	}
	requireNativeHelpSpawns(t, marker, 2)
	f.ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	paused := inspectAppliedKeyFixture(t, f)
	if paused.NativeOptions != nil || len(paused.UnreportedLaunchOptions) != 0 {
		t.Fatalf("paused inspection published native hints: %+v", paused)
	}
	requireNativeHelpSpawns(t, marker, 2)
}

func nativeOptionsPreviewRequest(t *testing.T, f inspectionFixture, extra map[string]string) settings.Request {
	t.Helper()
	launch, err := f.ex.LaunchSettings("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	return settings.Request{Engine: "llamacpp", Settings: settings.ConfigFromLaunch(launch, 54301), ModelChange: &modelconfig.ModelChange{ModelID: f.request.ModelID, ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{ExtraOptions: extra}}}}
}

func TestNativeOptionsPreviewDiscoversAndValidatesSelectedBuild(t *testing.T) {
	var marker string
	f := newAppliedKeyFixture(t, func(f inspectionFixture) {
		marker, _ = nativeOptionsFixture(t, f.ex, f.st)
	})
	preview, err := f.ex.PreviewLaunch(nativeOptionsPreviewRequest(t, f, map[string]string{"spec-type": "draft-mtp"}))
	if err != nil || len(preview.Errors) != 0 {
		t.Fatalf("reported extra preview = %+v, err=%v", preview, err)
	}
	requireNativeHelpSpawns(t, marker, 1)
	preview, err = f.ex.PreviewLaunch(nativeOptionsPreviewRequest(t, f, map[string]string{"expert-cache": "auto"}))
	if err != nil {
		t.Fatal(err)
	}
	if got := preview.Errors["modelLoads.extraOptions"]; !strings.Contains(got, "This llama.cpp build does not report --expert-cache.") {
		t.Fatalf("unsupported extra preview errors = %v", preview.Errors)
	}
	requireNativeHelpSpawns(t, marker, 1)
	f.ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	preview, err = f.ex.PreviewLaunch(nativeOptionsPreviewRequest(t, f, map[string]string{"expert-cache": "auto"}))
	if err != nil || len(preview.Errors) != 0 {
		t.Fatalf("paused manual preview reused withheld hints: %+v, err=%v", preview, err)
	}
	requireNativeHelpSpawns(t, marker, 1)
	if f.st.running || f.st.proc != nil {
		t.Fatal("help preview activated an engine")
	}
}

func TestNativeOptionsPaddedAliasesPreview(t *testing.T) {
	var marker string
	f := newAppliedKeyFixture(t, func(f inspectionFixture) {
		marker, _ = nativeOptionsFixture(t, f.ex, f.st)
		writeTestFile(t, os.Getenv("PAIR_NATIVE_HELP_FILE"), []byte(nativeHelpPaddedAliases))
	})
	for _, extra := range []map[string]string{
		{"device": "CPU"},
		{"split-mode": "layer"},
		{"tensor-split": "1,1"},
		{"main-gpu": "0"},
	} {
		preview, err := f.ex.PreviewLaunch(nativeOptionsPreviewRequest(t, f, extra))
		if err != nil || len(preview.Errors) != 0 {
			t.Fatalf("padded alias preview for %v = %+v, err=%v", extra, preview, err)
		}
	}
	requireNativeHelpSpawns(t, marker, 1)
}

func TestNativeOptionsCompilerUsesOnlyCompletedSameBuildEvidence(t *testing.T) {
	var marker, library string
	f := newAppliedKeyFixture(t, func(f inspectionFixture) {
		marker, library = nativeOptionsFixture(t, f.ex, f.st)
	})
	rt := f.st.plat.Runtime
	rt.ModelLoads = modelconfig.CloneLoads(rt.ModelLoads)
	rt.ModelLoads[0].Requested.ExtraOptions = map[string]string{"expert-cache": "auto"}
	command, err := launchForState(f.st, rt.Port)
	if err != nil {
		t.Fatal(err)
	}
	compile := func() error {
		t.Helper()
		var inputs map[string]modelEstimateInput
		_, err := compileTypedLlamaInputs(context.Background(), f.st, command, rt, false, &inputs)
		if err == nil && inputs[f.request.ModelID].SWAFull != nil {
			t.Fatal("help spelling established an inherited SWA allocation default")
		}
		return err
	}
	if err := compile(); err != nil {
		t.Fatalf("unknown build refused manual entry: %v", err)
	}
	requireNativeHelpSpawns(t, marker, 0)
	if _, known := f.ex.nativeOptions(context.Background(), f.st); !known {
		t.Fatal("fixture help unavailable")
	}
	if err := compile(); err == nil || !strings.Contains(err.Error(), "This llama.cpp build does not report --expert-cache.") {
		t.Fatalf("known build accepted an unreported extra: %v", err)
	}
	requireNativeHelpSpawns(t, marker, 1)
	// A different selected build is unknown even if running metadata still
	// names the build whose spelling was discovered previously.
	f.st.buildID = requireNativeBuildID(t, f.st)
	writeTestFile(t, library, []byte("a different selected build"))
	if err := compile(); err != nil {
		t.Fatalf("compiler reused another build's help: %v", err)
	}
	requireNativeHelpSpawns(t, marker, 1)
}

func TestNativeOptionsUnknownPreviewAllowsManualEntry(t *testing.T) {
	test := func(name string, configure func(*Executor), fail bool, spawns int) {
		t.Run(name, func(t *testing.T) {
			var marker string
			f := newAppliedKeyFixture(t, func(f inspectionFixture) {
				marker, _ = nativeOptionsFixture(t, f.ex, f.st)
			})
			configure(f.ex)
			if fail {
				t.Setenv("PAIR_NATIVE_HELP_FAILURE", "1")
			}
			preview, err := f.ex.PreviewLaunch(nativeOptionsPreviewRequest(t, f, map[string]string{"expert-cache": "auto"}))
			if err != nil || len(preview.Errors) != 0 {
				t.Fatalf("unknown-help manual preview = %+v, err=%v", preview, err)
			}
			requireNativeHelpSpawns(t, marker, spawns)
			result := inspectAppliedKeyFixture(t, f)
			if result.NativeOptions != nil || len(result.UnreportedLaunchOptions) != 0 {
				t.Fatalf("unknown-help inspection = %+v", result)
			}
			requireNativeHelpSpawns(t, marker, spawns)
		})
	}
	test("paused", func(e *Executor) { e.nodeProjection.set(nodeProjection{ManagementBlocked: true}) }, false, 0)
	test("unknown projection", func(e *Executor) { e.nodeProjection = nodeProjectionStore{} }, false, 0)
	test("failed help", func(*Executor) {}, true, 1)
}

func TestTypedLlamaExtraOptions(t *testing.T) {
	layers := int32(4)
	request := modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "cpu"}, ExtraOptions: map[string]string{"expert-cache": "auto", "spec-type": "draft-mtp", "moe-cache-mib": "2048"}}
	options, _, err := typedLlamaModelOptions("model", request, &layers)
	if err != nil {
		t.Fatal(err)
	}
	data, err := typedLlamaPreset(map[string]map[string]string{"model": options})
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := parseLlamaServingPresetAllowing(data, modelLoadAllowedPresetKeys([]modelconfig.ModelLoad{{ProfileKey: "model", ModelID: "model", Requested: request}}))
	if err != nil {
		t.Fatal(err)
	}
	for key, want := range request.ExtraOptions {
		if parsed["model"][key] != want {
			t.Fatalf("%s=%q, want %q", key, parsed["model"][key], want)
		}
	}
	if strings.Index(string(data), "expert-cache = auto") > strings.Index(string(data), "moe-cache-mib = 2048") {
		t.Fatal("extras not sorted")
	}
	if _, err := parseLlamaServingPreset(data); err == nil {
		t.Fatal("operator parser accepted unknown fork option")
	}
	for _, tc := range []struct{ key, value, reason string }{
		{"ctx-size", "4096", "Use the typed field for ctx-size."},
		{"alias", "x", "PAIR owns alias."},
		{"mmproj-auto", "true", "Automatic projector selection must be disabled"},
		{"moe-cache-mib", "invalid", "moe-cache-mib"},
		{"tensor-split", "1,1", "Reviewed Auto chooses the GPU; remove tensor-split or choose another GPU mode."},
	} {
		t.Run(tc.key, func(t *testing.T) {
			r := modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "cpu"}, ExtraOptions: map[string]string{tc.key: tc.value}}
			if tc.key == "tensor-split" {
				r.GPU.Mode = "auto"
			}
			_, _, err := typedLlamaModelOptions("model", r, &layers)
			if err == nil || !strings.Contains(err.Error(), tc.reason) {
				t.Fatalf("refusal=%v, want %s", err, tc.reason)
			}
		})
	}
}

func TestConvertTypedLlamaExtraOptions(t *testing.T) {
	extras := map[string]string{"spec-type": "draft-mtp", "moe-cache-mib": "2048", "expert-cache": "auto", "device": "CUDA0,CUDA1", "split-mode": "layer", "mmproj-auto": "false"}
	r, err := convertTypedLlamaOptions("model", extras)
	if err != nil || !reflect.DeepEqual(r.ExtraOptions, extras) {
		t.Fatalf("converted=%+v err=%v", r, err)
	}
	if _, err := convertTypedLlamaOptions("model", map[string]string{"alias": "false"}); err == nil || !strings.Contains(err.Error(), "alias") {
		t.Fatalf("alias refusal=%v", err)
	}
	if _, err := convertTypedLlamaOptions("model", map[string]string{"mmproj-auto": "true"}); err == nil {
		t.Fatal("conversion accepted automatic projector selection")
	}
}

func TestConvertTypedLlamaExtraOptionsPreset(t *testing.T) {
	f := newModelLoadFixture(t)
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	data = append(data, []byte("expert-cache = auto\nspec-type = draft-mtp\nmoe-cache-mib = 2048\ndevice = CUDA0,CUDA1\nsplit-mode = layer\n")...)
	writeTestFile(t, f.source, data)
	rt := convertModelLoadFixture(t, f)
	for _, load := range rt.ModelLoads {
		if load.ModelID != "medium-model" {
			continue
		}
		want := map[string]string{"expert-cache": "auto", "spec-type": "draft-mtp", "moe-cache-mib": "2048", "device": "CUDA0,CUDA1", "split-mode": "layer"}
		if !reflect.DeepEqual(load.Requested.ExtraOptions, want) {
			t.Fatalf("converted extras=%v", load.Requested.ExtraOptions)
		}
		return
	}
	t.Fatal("converted model missing")
}

func TestTypedLlamaGPUSelection(t *testing.T) {
	f := newAppliedKeyFixture(t)
	rt := f.st.plat.Runtime
	layers := int32(1)
	rt.ModelLoads[0].Requested.GPU = &modelconfig.GPU{Mode: "layers", Layers: &layers}
	rt.ModelLoads[0].Requested.ExtraOptions = map[string]string{"device": "CUDA1"}
	command, err := launchForState(f.st, rt.Port)
	if err != nil {
		t.Fatal(err)
	}
	var inputs map[string]modelEstimateInput
	prepared, err := compileTypedLlamaInputs(context.Background(), f.st, command, rt, false, &inputs)
	if err != nil {
		t.Fatal(err)
	}
	sections, err := parseLlamaServingPresetAllowing(prepared.Snapshot, modelLoadAllowedPresetKeys(prepared.ModelLoads))
	if err != nil || sections[f.request.ModelID]["device"] != "CUDA1" {
		t.Fatalf("sections=%v err=%v", sections, err)
	}
	rt.ModelLoads = prepared.ModelLoads
	f.st.plat.Runtime = rt
	result := inspectAppliedKeyFixture(t, f)
	if result.ConfigurationKeyReason != "gpu-selection" || result.ConfigurationKey != "" || result.ExecutionKey == "" || len(result.ConfigurationPools) != 0 {
		t.Fatalf("identity=%+v", result)
	}
	if result.Estimate.Status != "unknown" || !slices.Contains(result.Estimate.Unknowns, "GPU selection is not covered by the estimate: device.") {
		t.Fatalf("estimate=%+v", result.Estimate)
	}
	if err := result.Validate(); err != nil {
		t.Fatal(err)
	}
	capture := f.ex.newModelBufferCapture(f.st, "llamacpp", prepared, inputs, command, keyTestMemoryContext())
	if capture != nil {
		defer capture.retire()
		if capture.frozen[f.request.ModelID] != nil {
			t.Fatal("GPU selection received finite capture")
		}
	}
	descriptors, err := modelInspectionOptions(nil)
	if err != nil {
		t.Fatal(err)
	}
	index := slices.IndexFunc(descriptors, func(d modelconfig.OptionDescriptor) bool { return d.Field == "extraOptions" })
	if index < 0 || descriptors[index].Kind != "options" || !reflect.DeepEqual(descriptors[index].NonPortableKeys, []string{"device", "main-gpu", "split-mode", "tensor-split"}) {
		t.Fatal("GPU portability descriptor missing")
	}
}

func TestExtraOptionUnknownNeverFits(t *testing.T) {
	now := time.Now()
	in := autoTestInput()
	in.Memory = autoTestMemory(now)
	state := modelEstimateState{now: func() time.Time { return now }, run: autoTestEstimate}
	control := state.estimate(context.Background(), in)
	if control.CapacityFit != "fits" || control.FreeFit != "fits" {
		t.Fatalf("finite control does not fit: %+v", control)
	}
	in.Allocation.Options.ExtraOptions = map[string]string{"expert-cache": "auto"}
	estimate := state.estimate(context.Background(), in)
	if estimate.Status != "unknown" || estimate.CapacityFit != "unknown" || estimate.FreeFit != "unknown" || !slices.Contains(estimate.Unknowns, "Options not priced: expert-cache.") {
		t.Fatalf("estimate=%+v", estimate)
	}
}

func TestGPUSelectionSharedAutoPreviewNamesProfiles(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	rt.ModelLoads[0].Requested.ExtraOptions = map[string]string{"main-gpu": "0"}
	rt.ModelLoads[1].Requested.GPU = &modelconfig.GPU{Mode: "auto"}
	other := rt.ModelLoads[1].Resolved
	layers := int32(1)
	other.GPU = modelconfig.GPU{Mode: "layers", Layers: &layers}
	other.Options.GPU = &other.GPU
	other.ResolverRevision, other.ReviewedInputID = "auto/test", "synthetic-reviewed-input"
	other.DeviceBinding = autoTestDeviceBinding()
	memory := &autoTestMemoryOwner{}
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: memory})
	_, err := compileTypedLlamaInputs(ctx, f.st, typedFixtureCommand(t, f, rt), rt, false)
	want := rt.ModelLoads[0].ModelID + " selects GPUs and cannot share an engine with reviewed Auto profile " + rt.ModelLoads[1].ModelID + "; set that profile's GPU mode to layers."
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("refusal=%v, want %s", err, want)
	}
	if memory.calls != 0 {
		t.Fatalf("memory read before refusal: %d", memory.calls)
	}
}

func TestTypedLlamaNativeFitValidation(t *testing.T) {
	on, floor := "on", int32(4096)
	for _, tc := range []struct {
		name string
		load modelconfig.RequestedLoad
	}{
		{"reviewed Auto", modelconfig.RequestedLoad{Fit: &on, GPU: &modelconfig.GPU{Mode: "auto"}}},
		{"fit context without fit", modelconfig.RequestedLoad{FitCtx: &floor}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := typedLlamaModelOptions("model", tc.load, nil); err == nil {
				t.Fatal("accepted invalid native fit")
			}
		})
	}
}

func TestTypedLlamaNativeFitDescriptors(t *testing.T) {
	options, err := llamaModelSettingsOptions()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		field    string
		portable bool
	}{{"fit", true}, {"fitTarget", false}, {"fitCtx", true}} {
		t.Run(tc.field, func(t *testing.T) {
			for _, option := range options {
				if option.Descriptor.Field == tc.field {
					if !option.Descriptor.Supported || option.Descriptor.Portable != tc.portable || option.Descriptor.Owner != "model-load" {
						t.Fatalf("descriptor=%+v", option.Descriptor)
					}
					if err := option.Descriptor.Validate(); err != nil {
						t.Fatal(err)
					}
					return
				}
			}
			t.Fatal("fit descriptor absent")
		})
	}
}

func TestTypedLlamaNativeFitOffPreservesFiniteOptions(t *testing.T) {
	off := "off"
	for _, requested := range []modelconfig.RequestedLoad{{GPU: &modelconfig.GPU{Mode: "cpu"}}, {Fit: &off, GPU: &modelconfig.GPU{Mode: "cpu"}}} {
		options, _, err := typedLlamaModelOptions("model", requested, nil)
		if err != nil {
			t.Fatal(err)
		}
		if options["fit"] != "off" || options["kv-unified"] != "false" || options["load-on-startup"] != "false" || options["ctx-size"] != "8192" || options["gpu-layers"] != "0" {
			t.Fatalf("finite defaults changed: %v", options)
		}
		if _, ok := options["fit-target"]; ok {
			t.Fatal("finite load acquired fit target")
		}
	}
}

func TestTypedLlamaNativeFitReparseScope(t *testing.T) {
	data := []byte("[*]\nfit = off\n[model]\nfit = on\nfit-target = 512\n")
	if _, err := parseLlamaServingPreset(data); err == nil {
		t.Fatal("strict operator parser accepted native fit")
	}
	for _, tc := range []struct{ name, data string }{
		{"wrong section", "[other]\nfit = on\n"},
		{"unvalidated key", "[model]\nfit = on\nctx-checkpoints = 2\n"},
		{"duplicate alias", "[model]\nfit = on\nfit = off\n"},
		{"inline comment", "[model]\nfit = on # comment\n"},
		{"substitution", "[model]\nfit-target = ${MARGIN}\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := parseLlamaServingPresetAllowing([]byte(tc.data), map[string]map[string]bool{"model": {"fit": true, "fit-target": true}}); err == nil {
				t.Fatal("allowing parser lost strict syntax or scope")
			}
		})
	}
}

func TestConvertTypedLlamaNativeFit(t *testing.T) {
	f := newModelLoadFixture(t)
	writeTestFile(t, f.source, []byte("[*]\nfit = off\nload-on-startup = false\n[weak-model]\nmodel = weak.gguf\nctx-size = 4096\nparallel = 1\ngpu-layers = 0\nfit = on\nfit-target = 1024\nfit-ctx = 2048\n[medium-model]\nmodel = medium.gguf\nctx-size = 16384\nparallel = 2\ngpu-layers = 0\n"))
	rt := convertModelLoadFixture(t, f)
	for _, load := range rt.ModelLoads {
		if load.ModelID == "weak-model" {
			if load.Requested.Fit == nil || *load.Requested.Fit != "on" || load.Requested.FitTarget == nil || *load.Requested.FitTarget != "1024" || load.Requested.FitCtx == nil || *load.Requested.FitCtx != 2048 {
				t.Fatalf("conversion lost native fit: %+v", load.Requested)
			}
			return
		}
	}
	t.Fatal("conversion omitted weak-model")
}

func TestConvertTypedLlamaNativeFitAliases(t *testing.T) {
	for _, tc := range []struct{ value, want string }{
		{"on", "on"}, {"true", "on"}, {"1", "on"}, {"off", "off"}, {"false", "off"}, {"0", "off"},
	} {
		t.Run(tc.value, func(t *testing.T) {
			got, err := convertTypedLlamaOptions("model", map[string]string{"fit": tc.value})
			if err != nil || got.Fit == nil || *got.Fit != tc.want {
				t.Fatalf("fit=%v, err=%v, want=%s", got.Fit, err, tc.want)
			}
		})
	}
}

func typedFixtureCommand(t *testing.T, f modelLoadFixture, rt Runtime) launchCommand {
	t.Helper()
	vars := f.st.pathVars()
	vars["host"] = "127.0.0.1"
	vars["port"] = strconv.Itoa(rt.Port)
	command, err := resolveProcessLaunch(llamaServingBase(rt), rt.Bin, vars)
	if err != nil {
		t.Fatal(err)
	}
	return command
}

type typedBindingCloseFile struct {
	servingArtifactFile
	afterClose func()
}

func (f *typedBindingCloseFile) Close() error {
	err := f.servingArtifactFile.Close()
	f.afterClose()
	return err
}

func TestLinkedModelTypedBinding(t *testing.T) {
	nativeFixture := func(t *testing.T) (modelLoadFixture, Runtime, launchCommand) {
		t.Helper()
		f := newModelLoadFixture(t)
		model := "owner/repository:Q4_K_M"
		commit := strings.Repeat("a", 40)
		repo := llamaRepoDir(f.root, "owner/repository")
		writeTestFile(t, filepath.Join(repo, "refs", "main"), []byte(commit))
		writeInspectionGGUF(t, filepath.Join(repo, "snapshots", commit, "model-Q4_K_M.gguf"), "llama", 0.25, 4096)
		rt := f.rt
		args := []string{"--models-max", "1"}
		rt.LaunchArgs, rt.Serving = &args, nil
		rt.ModelLoads = []modelconfig.ModelLoad{{ModelID: model}}
		return f, rt, typedFixtureCommand(t, f, rt)
	}
	t.Run("native callback reaches the selected owner", func(t *testing.T) {
		f, rt, command := nativeFixture(t)
		calls := copyTestNativeModels(t, f.ex, "fake", rt.ModelLoads[0].ModelID)
		prepared, err := compileTypedLlamaInputs(context.Background(), f.st, command, rt, false)
		if err != nil || *calls != 2 || prepared == nil || prepared.Config != nil || len(prepared.ModelLoads) != 1 {
			t.Fatalf("native compile: result=%+v calls=%d err=%v", prepared, *calls, err)
		}
	})
	t.Run("native compilation completes while settings owns opMu", func(t *testing.T) {
		f, rt, command := nativeFixture(t)
		calls := copyTestNativeModels(t, f.ex, "fake", rt.ModelLoads[0].ModelID)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		finished := make(chan error, 1)
		f.st.opMu.Lock()
		go func() {
			_, err := compileTypedLlamaInputs(ctx, f.st, command, rt, false)
			finished <- err
		}()
		select {
		case err := <-finished:
			f.st.opMu.Unlock()
			if err != nil || *calls != 2 {
				t.Fatalf("compile under settings lock: calls=%d err=%v", *calls, err)
			}
		case <-ctx.Done():
			// Release the lock before waiting for cleanup, even for the RED case.
			f.st.opMu.Unlock()
			<-finished
			t.Fatal("native compilation blocked on the settings operation lock")
		}
	})
	t.Run("missing owner refuses native fallback", func(t *testing.T) {
		f, rt, command := nativeFixture(t)
		rt.modelLoadOwner = nil
		usage := servingReadUsage{}
		if _, err := compileTypedLlamaInputsExpected(context.Background(), f.st, command, rt, false, servingFileAccess{usage: &usage}, nil); err == nil || usage.HeaderBytes != 0 || usage.SourceBytes != 0 {
			t.Fatal("native cache fallback bypassed its recognition owner")
		}
	})
	t.Run("recognition withdrawal precedes final cache reads", func(t *testing.T) {
		f, rt, command := nativeFixture(t)
		copyTestNativeModels(t, f.ex, "fake", rt.ModelLoads[0].ModelID)
		native := f.ex.client.Transport
		calls, forbidden, opened, closed := 0, 0, 0, 0
		withdrawn := false
		f.ex.client.Transport = memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
			if request.Method != http.MethodGet || request.URL.Path != "/copy-test/fake/models" {
				t.Fatalf("unexpected native request: %s %s", request.Method, request.URL.Path)
			}
			calls++
			if calls == 2 {
				withdrawn = true
				return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"models":[]}`))}, nil
			}
			return native.RoundTrip(request)
		})
		var reads []inspectionRead
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			file, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			opened++
			return &inspectionCountFile{File: file, reads: &reads, opened: &opened, closed: &closed, onRead: func(int64, int) {
				if withdrawn {
					forbidden++
				}
			}}, nil
		}}
		if _, err := compileTypedLlamaInputsExpected(context.Background(), f.st, command, rt, false, access, nil); err == nil || !withdrawn || forbidden != 0 || opened != closed {
			t.Fatalf("withdrawal: err=%v calls=%d forbidden=%d opened=%d closed=%d", err, calls, forbidden, opened, closed)
		}
	})

	replacement := func(name, operation, kind string, atOpen int) {
		t.Run(name, func(t *testing.T) {
			f := newModelLoadFixture(t)
			rt := f.rt
			command := f.command
			if operation == "native" {
				f, rt, command = nativeFixture(t)
				copyTestNativeModels(t, f.ex, "fake", rt.ModelLoads[0].ModelID)
			}
			if operation == "compile" {
				rt = convertModelLoadFixture(t, f)
				command = typedFixtureCommand(t, f, rt)
			}
			target := filepath.Join(f.root, "medium.gguf")
			if kind == "selector" {
				target = f.source
				if operation == "native" {
					target = filepath.Join(llamaRepoDir(f.root, "owner/repository"), "refs", "main")
				}
			}
			if kind == "build" {
				// Compilation reads build identity but never executes this file.
				target = filepath.Join(t.TempDir(), "native-build")
				writeTestFile(t, target, []byte("synthetic native build"))
				command.Bin = target
			}
			var reads []inspectionRead
			opens, opened, closed, forbidden := 0, 0, 0, 0
			replaced := false
			access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
				if path == target {
					opens++
					if opens == atOpen {
						info, err := os.Stat(target)
						if err != nil {
							t.Fatal(err)
						}
						data, err := os.ReadFile(target)
						if err != nil {
							t.Fatal(err)
						}
						if err := os.Rename(target, target+".old"); err != nil {
							t.Fatal(err)
						}
						writeTestFile(t, target, data)
						if err := os.Chtimes(target, info.ModTime(), info.ModTime()); err != nil {
							t.Fatal(err)
						}
						replaced = true
					}
				}
				file, err := os.Open(path)
				if err != nil {
					return nil, err
				}
				opened++
				return &inspectionCountFile{File: file, reads: &reads, opened: &opened, closed: &closed, onRead: func(int64, int) {
					if path == target && replaced {
						forbidden++
					}
				}}, nil
			}}
			var err error
			if operation == "conversion" {
				_, err = f.ex.previewTypedLlamaConversionExpected(context.Background(), f.st, command, rt, access, nil)
			} else {
				_, err = compileTypedLlamaInputsExpected(context.Background(), f.st, command, rt, false, access, nil)
			}
			if err == nil || !replaced || forbidden != 0 || opened != closed {
				t.Fatalf("replacement: err=%v replaced=%v target opens=%d forbidden=%d opened=%d closed=%d", err, replaced, opens, forbidden, opened, closed)
			}
		})
	}
	replacement("final artifact sampling keeps initial identity", "compile", "artifact", 4)
	replacement("final HF ref read keeps initial identity", "native", "selector", 7)
	replacement("final native build sampling keeps initial identity", "compile", "build", 2)
	replacement("conversion compilation keeps external artifact identity", "conversion", "artifact", 3)
	replacement("conversion first resolution keeps initial source identity", "conversion", "selector", 4)
	replacement("conversion final source sampling keeps initial identity", "conversion", "selector", 12)
	t.Run("conversion retains selector after first complete source read", func(t *testing.T) {
		f := newModelLoadFixture(t)
		data, err := os.ReadFile(f.source)
		if err != nil {
			t.Fatal(err)
		}
		info, err := os.Stat(f.source)
		if err != nil {
			t.Fatal(err)
		}
		var reads []inspectionRead
		opens, opened, closed, forbidden := 0, 0, 0, 0
		replaced := false
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			file, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			opened++
			counted := &inspectionCountFile{File: file, reads: &reads, opened: &opened, closed: &closed, onRead: func(int64, int) {
				if path == f.source && replaced {
					forbidden++
				}
			}}
			if path != f.source {
				return counted, nil
			}
			opens++
			if opens != 2 {
				return counted, nil
			}
			// The retained source reader closes after the final sampler and
			// identity check. Replace before resolver admission starts.
			return &typedBindingCloseFile{servingArtifactFile: counted, afterClose: func() {
				if err := os.Rename(f.source, f.source+".old"); err != nil {
					t.Fatal(err)
				}
				writeTestFile(t, f.source, data)
				if err := os.Chtimes(f.source, info.ModTime(), info.ModTime()); err != nil {
					t.Fatal(err)
				}
				replaced = true
			}}, nil
		}}
		_, err = f.ex.previewTypedLlamaConversionExpected(context.Background(), f.st, f.command, f.rt, access, nil)
		if err == nil || !replaced || forbidden != 0 || opened != closed {
			t.Fatalf("initial selector replacement: err=%v replaced=%v forbidden=%d opened=%d closed=%d", err, replaced, forbidden, opened, closed)
		}
	})

	t.Run("saved artifact metadata checked before sampling", func(t *testing.T) {
		f := newModelLoadFixture(t)
		rt := convertModelLoadFixture(t, f)
		target := rt.ModelLoads[0].Resolved.Artifacts[0].Path
		rt.ModelLoads[0].Resolved.Artifacts[0].Size = "1"
		opens := 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			if path == target {
				opens++
			}
			return os.Open(path)
		}}
		if _, err := compileTypedLlamaInputsExpected(context.Background(), f.st, typedFixtureCommand(t, f, rt), rt, true, access, nil); err == nil || opens != 0 {
			t.Fatalf("saved metadata: err=%v artifact opens=%d", err, opens)
		}
	})
	t.Run("exact outside custom artifacts remain usable while off", func(t *testing.T) {
		f := newModelLoadFixture(t)
		outside := filepath.Join(t.TempDir(), "outside.gguf")
		writeInspectionGGUF(t, outside, "llama", 0.25, 4096)
		writeTestFile(t, f.source, []byte("[external-alias]\nmodel = "+outside+"\ngpu-layers = 0\n"))
		conversion, err := f.ex.previewTypedLlamaConversion(context.Background(), f.st, f.command, f.rt)
		if err != nil {
			t.Fatal(err)
		}
		rt := conversion.Runtime
		rt.modelLoadOwner = nil
		prepared, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, rt), rt, true)
		if err != nil || prepared == nil || prepared.ModelLoads[0].Resolved.Artifacts[0].Path != outside || f.st.running {
			t.Fatalf("outside exact custom artifacts: result=%+v err=%v", prepared, err)
		}
	})
}

func captureModelLoadChild(t *testing.T, path string) ([]string, []byte) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var captured struct {
		Args   []string
		Preset []byte
	}
	if err := json.Unmarshal(data, &captured); err != nil {
		t.Fatal(err)
	}
	return captured.Args, captured.Preset
}

func TestModelSettingsLoadDeterministicCompleteCompile(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	command := typedFixtureCommand(t, f, rt)
	first, err := compilePreparedServing(f.st, command, rt, true)
	if err != nil {
		t.Fatal(err)
	}
	slices.Reverse(rt.ModelLoads)
	second, err := compilePreparedServing(f.st, command, rt, true)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first.Snapshot, second.Snapshot) || first.Digest != second.Digest {
		t.Fatal("input order changed deterministic output or identity")
	}
	sections, err := parseLlamaServingPreset(first.Snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if len(sections) != 2 || sections["medium-model"]["ctx-size"] != "16384" || sections["medium-model"]["parallel"] != "2" || sections["weak-model"]["ctx-size"] != "4096" {
		t.Fatalf("router allocation differs: %+v", sections)
	}
	if first.Config != nil || len(first.Envelopes) != 0 {
		t.Fatal("typed settings fabricated declarations")
	}
	if first.ModelLoads[0].ModelID != "medium-model" || first.ModelLoads[0].Resolved.PerSlotContext != 8192 || first.ModelLoads[0].Resolved.TotalContext != 16384 {
		t.Fatalf("incorrect requested/resolved arithmetic: %+v", first.ModelLoads)
	}
}

func TestModelSettingsLoadTypedOnlyLifecycleConsumesPreset(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	f.st.plat.Runtime = rt
	if rt.Serving != nil || !preparedServingRuntime(f.st, rt) || !llamaServingRuntime(f.st, rt) {
		t.Fatal("typed-only runtime missed lifecycle predicate")
	}
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	args, preset := captureModelLoadChild(t, f.capture)
	index := slices.Index(args, "--models-preset")
	if index < 0 || index+1 >= len(args) || args[index+1] == f.source {
		t.Fatal("child did not receive private generation")
	}
	for _, flag := range []string{"--ctx-size", "--parallel", "--sleep-idle-seconds", "--cache-ram"} {
		if slices.Contains(args, flag) {
			t.Fatalf("router argv contains per-model allocation %s", flag)
		}
	}
	sections, err := parseLlamaServingPreset(preset)
	if err != nil {
		t.Fatal(err)
	}
	if sections["medium-model"]["ctx-size"] != "16384" || sections["weak-model"]["ctx-size"] != "4096" {
		t.Fatal("child did not consume complete generated bytes")
	}
	state, err := f.ex.LaunchSettings("fake")
	if err != nil {
		t.Fatal(err)
	}
	if state.Serving != nil || len(state.ModelLoads) != 2 || len(state.ServingReadback.LaunchValidated) != 0 {
		t.Fatal("readback lost loads or manufactured strict evidence")
	}
	if _, err := os.Stat(args[index+1]); err != nil {
		t.Fatal("readback removed active generation", err)
	}
}

func TestModelSettingsLoadFreshExecutorAndCrashRegeneration(t *testing.T) {
	test := func(name string, automatic bool) {
		t.Run(name, func(t *testing.T) {
			f := newModelLoadFixture(t)
			if automatic {
				f.rt.Port = 0
				f.st.plat.Runtime = f.rt
				var err error
				f.command, err = launchForState(f.st, 0)
				if err != nil {
					t.Fatal(err)
				}
			}
			rt := convertModelLoadFixture(t, f)
			if err := f.ex.persistRuntimeServing("fake", rt.Port, rt.LaunchArgs, rt.LaunchEnv, &rt.Serving, &rt.ModelLoads); err != nil {
				t.Fatal(err)
			}
			manifest := *f.st.manifest
			platform := *f.st.plat
			platform.ModelsDir = f.root
			manifest.Platforms = map[string]Platform{runtime.GOOS + "/" + runtime.GOARCH: platform}
			base, err := json.Marshal(manifest)
			if err != nil {
				t.Fatal(err)
			}
			reg := NewRegistry()
			reg.bundledRaw["fake"] = base
			if err := reg.LoadOverrideDir(f.ex.overrideDir); err != nil {
				t.Fatal(err)
			}
			ex := NewExecutor(reg, NewReporter(nil), func(string, any) {}, t.TempDir())
			acceptModelMutations(ex)
			running := make(chan EngineStatus, 8)
			ex.emit = func(method string, value any) {
				if status, ok := value.(EngineStatus); ok && method == "engine:state-changed" && status.Running {
					running <- status
				}
			}
			t.Cleanup(func() {
				if err := ex.Stop("fake"); err != nil {
					t.Error(err)
				}
			})
			st, err := ex.state("fake")
			if err != nil {
				t.Fatal(err)
			}
			st.modelsDir = f.root
			if len(st.plat.Runtime.ModelLoads) != 2 || st.plat.Runtime.Serving != nil {
				t.Fatal("fresh executor lost persisted typed input")
			}
			if err := ex.Start(context.Background(), "fake"); err != nil {
				t.Fatal(err)
			}
			<-running
			if automatic && (st.port <= 0 || st.plat.Runtime.Port != 0) {
				t.Fatal("automatic port was not allocated from unchanged zero-port intent")
			}
			_, before := captureModelLoadChild(t, f.capture)
			st.mu.Lock()
			proc := st.proc
			st.mu.Unlock()
			if proc == nil {
				t.Fatal("fresh start has no owned child")
			}
			if err := proc.cmd.Process.Kill(); err != nil {
				t.Fatal(err)
			}
			select {
			case <-proc.done:
			case <-time.After(5 * time.Second):
				t.Fatal("synthetic crash did not exit")
			}
			select {
			case <-running:
			case <-time.After(15 * time.Second):
				t.Fatal("saved-On crash recovery did not return to readiness")
			}
			st.mu.Lock()
			replacement := st.proc
			st.mu.Unlock()
			if replacement == nil || replacement == proc {
				t.Fatal("crash recovery did not replace the owned child")
			}
			_, after := captureModelLoadChild(t, f.capture)
			if !bytes.Equal(before, after) {
				t.Fatal("crash relaunch changed generated bytes")
			}
		})
	}
	test("configured port", false)
	test("automatic port", true)
}

func TestModelSettingsLoadPreSpawnDriftRefuses(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	f.st.plat.Runtime = rt
	vars := f.st.pathVars()
	vars["host"] = "127.0.0.1"
	vars["port"] = strconv.Itoa(rt.Port)
	err := f.ex.bringUpProcessWithSnapshot(context.Background(), f.st, "fake", rt, rt.Port, vars, func(dir string, data []byte) (string, func(), error) {
		path, release, err := prepareServingSnapshot(dir, data)
		if err != nil {
			return "", nil, err
		}
		writeInspectionGGUF(t, filepath.Join(f.root, "medium.gguf"), "llama", 0.75, 4096)
		return path, release, nil
	})
	if err == nil {
		t.Fatal("pre-spawn drift accepted")
	}
	if _, err := os.Stat(f.capture); !os.IsNotExist(err) {
		t.Fatalf("drift spawned child: %v", err)
	}
	if f.st.proc != nil {
		t.Fatal("drift installed a child")
	}
}

func TestModelSettingsLoadConflictingSourcesRefused(t *testing.T) {
	test := func(name string, edit func(*testing.T, modelLoadFixture, *launchCommand)) {
		t.Run(name, func(t *testing.T) {
			f := newModelLoadFixture(t)
			rt := convertModelLoadFixture(t, f)
			command := typedFixtureCommand(t, f, rt)
			edit(t, f, &command)
			if _, err := compilePreparedServing(f.st, command, rt, false); err == nil {
				t.Fatal("conflicting allocation source accepted")
			}
		})
	}
	test("CLI context", func(t *testing.T, f modelLoadFixture, c *launchCommand) {
		c.Args = append(c.Args, "--ctx_size", "8192")
	})
	test("explicit environment", func(t *testing.T, f modelLoadFixture, c *launchCommand) { c.Env["LLAMA_ARG_CTX_SIZE"] = "16384" })
	test("inherited environment", func(t *testing.T, f modelLoadFixture, c *launchCommand) { t.Setenv("LLAMA_ARG_N_PARALLEL", "2") })
	test("external preset", func(t *testing.T, f modelLoadFixture, c *launchCommand) {
		c.Args = append(c.Args, "--models-preset", f.source)
	})
	test("CLI threads", func(t *testing.T, f modelLoadFixture, c *launchCommand) { c.Args = append(c.Args, "--threads", "8") })
	test("CLI CPU experts", func(t *testing.T, f modelLoadFixture, c *launchCommand) { c.Args = append(c.Args, "--cpu-moe") })
	test("CLI CPU expert layers", func(t *testing.T, f modelLoadFixture, c *launchCommand) { c.Args = append(c.Args, "--n-cpu-moe", "2") })
	test("explicit CPU expert environment", func(t *testing.T, f modelLoadFixture, c *launchCommand) { c.Env["LLAMA_ARG_CPU_MOE"] = "true" })
	test("inherited CPU expert environment", func(t *testing.T, f modelLoadFixture, c *launchCommand) { t.Setenv("LLAMA_ARG_N_CPU_MOE", "2") })
}

func TestModelSettingsLoadPreSpawnContentSourceReplacementRefuses(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	f.st.plat.Runtime = rt
	vars := f.st.pathVars()
	vars["host"] = "127.0.0.1"
	vars["port"] = strconv.Itoa(rt.Port)
	err := f.ex.bringUpProcessWithSnapshot(context.Background(), f.st, "fake", rt, rt.Port, vars, func(dir string, data []byte) (string, func(), error) {
		path, release, err := prepareServingSnapshot(dir, data)
		if err != nil {
			return "", nil, err
		}
		f.ex.downloads.mu.Lock()
		f.ex.downloads.source = "replacement-source"
		f.ex.downloads.mu.Unlock()
		return path, release, nil
	})
	if err == nil {
		t.Fatal("content source replacement accepted before spawn")
	}
	if _, err := os.Stat(f.capture); !os.IsNotExist(err) {
		t.Fatal("source replacement spawned child", err)
	}
}

func TestModelSettingsLoadPreSpawnNativeBuildDriftRefuses(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	native := filepath.Join(f.root, "synthetic-native-build")
	writeTestFile(t, native, []byte("first native binding"))
	rt.Bin = native
	f.st.binPath = native
	prepared, err := compilePreparedServing(f.st, typedFixtureCommand(t, f, rt), rt, false)
	if err != nil {
		t.Fatal(err)
	}
	rt.ModelLoads = prepared.ModelLoads
	f.st.plat.Runtime = rt
	vars := f.st.pathVars()
	vars["host"], vars["port"] = "127.0.0.1", strconv.Itoa(rt.Port)
	err = f.ex.bringUpProcessWithSnapshot(context.Background(), f.st, "fake", rt, rt.Port, vars, func(dir string, data []byte) (string, func(), error) {
		path, release, err := prepareServingSnapshot(dir, data)
		if err != nil {
			return "", nil, err
		}
		writeTestFile(t, native, []byte("replacement native binding"))
		return path, release, nil
	})
	if err == nil || !strings.Contains(err.Error(), "nativeBuildId") && !strings.Contains(err.Error(), "resolved") {
		t.Fatalf("native binding drift reached spawn: %v", err)
	}
	if f.st.proc != nil {
		t.Fatal("changed native build installed a child")
	}
}

func TestModelSettingsLoadResetAndDefaultSemantics(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	slots, threads := int32(3), int32(6)
	patched, err := patchModelLoad(rt.ModelLoads, "medium-model", "medium-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &slots, Threads: &threads}})
	if err != nil {
		t.Fatal(err)
	}
	reset, err := patchModelLoad(patched, "medium-model", "medium-model", modelconfig.LoadPatch{Clear: []string{"slots", "threads", "context", "gpu"}})
	if err != nil {
		t.Fatal(err)
	}
	rt.ModelLoads = reset
	prepared, err := compilePreparedServing(f.st, typedFixtureCommand(t, f, rt), rt, false)
	if err != nil {
		t.Fatal(err)
	}
	load := prepared.ModelLoads[0]
	if load.Requested.Context != nil || load.Requested.Slots != nil || load.Requested.Threads != nil || load.Requested.GPU != nil || load.Resolved.PerSlotContext != 8192 || load.Resolved.Slots != 1 {
		t.Fatalf("reset retained old intent: %+v", load)
	}
	sections, err := parseLlamaServingPreset(prepared.Snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if _, exists := sections["medium-model"]["threads"]; exists {
		t.Fatal("reset retained hidden thread override")
	}
	if !reflect.DeepEqual(prepared.ModelLoads[1].Requested, rt.ModelLoads[1].Requested) {
		t.Fatal("reset changed another model")
	}
}

func previewModelLoadFixture(t *testing.T, f modelLoadFixture, model string, patch modelconfig.LoadPatch) settings.Preview {
	t.Helper()
	state, err := f.ex.LaunchSettings("fake")
	if err != nil {
		t.Fatal(err)
	}
	preview, err := f.ex.PreviewLaunch(settings.Request{Engine: "fake", Settings: settings.ConfigFromLaunch(state, 54301), ModelChange: &modelconfig.ModelChange{ModelID: model, ExpectedPolicyRevision: "0", Load: &patch}})
	if err != nil || len(preview.Errors) != 0 || preview.Conflict != nil {
		t.Fatalf("model load preview: errors=%v conflict=%v err=%v", preview.Errors, preview.Conflict, err)
	}
	return preview
}

func readModelLoadRuntimeOverride(t *testing.T, e *Executor) Runtime {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(e.overrideDir, "fake.json"))
	if err != nil {
		t.Fatal(err)
	}
	data, err = e.reg.mergeOntoBundled(data)
	if err != nil {
		t.Fatal(err)
	}
	data, err = applyPlatformDefaults(data)
	if err != nil {
		t.Fatal(err)
	}
	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	platform, ok := manifest.HostPlatform()
	if !ok {
		t.Fatal("persisted host platform missing")
	}
	return platform.Runtime
}

func reviewedRemovalConversionFixture(t *testing.T, last bool) (modelLoadFixture, settings.Preview, *modelconfig.ModelChange) {
	t.Helper()
	f := newModelLoadFixture(t)
	if last {
		writeTestFile(t, f.source, []byte("[*]\nfit = off\nload-on-startup = false\n[weak-model]\nmodel = weak.gguf\nctx-size = 4096\nparallel = 1\ngpu-layers = 0\n"))
	}
	request := settingsRequest(t, f.ex)
	request.ModelChange = &modelconfig.ModelChange{ProfileKey: "weak-model", ModelID: "weak-model", Remove: true, ExpectedPolicyRevision: "0"}
	preview, err := f.ex.PreviewLaunch(request)
	if err != nil || len(preview.Errors) != 0 || len(preview.PreviewDigest) != 64 {
		t.Fatalf("external removal preview=%+v error=%v", preview, err)
	}
	change := request.ModelChange.Clone()
	change.PreviewDigest = preview.PreviewDigest
	return f, preview, &change
}

func TestModelSettingsLoadConversionAcceptsReviewedRemoval(t *testing.T) {
	test := func(name string, last bool) {
		t.Run(name, func(t *testing.T) {
			f, preview, change := reviewedRemovalConversionFixture(t, last)
			if err := f.ex.desired.set("fake", false, intentPeer); err != nil {
				t.Fatal(err)
			}
			frozen := settings.Request{Engine: "fake", Settings: preview.Settings}
			if err := f.ex.validateReviewedModelRemoval(context.Background(), frozen, change); err != nil {
				t.Fatalf("frozen removal refused: %v", err)
			}
			assertNoSettingsOverride(t, f.ex)
			if len(f.st.plat.Runtime.ModelLoads) != 0 || f.st.running || f.st.proc != nil {
				t.Fatal("frozen removal validation changed the external router")
			}
			rebinds := 0
			result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings, ReviewedModelChange: change}, func() error { rebinds++; return nil })
			if err != nil {
				t.Fatalf("reviewed removal refused: %v", err)
			}
			if rebinds != 1 || result.Running || f.st.proc != nil || result.Serving != nil || !settings.ConfigFromLaunch(result, preview.Settings.ProxyPort).Equal(preview.Settings) {
				t.Fatalf("accepted removal result=%+v rebinds=%d", result, rebinds)
			}
			stored := readModelLoadRuntimeOverride(t, f.ex)
			if !modelconfig.EqualLoads(stored.ModelLoads, preview.Settings.ModelLoads) || stored.Serving != nil {
				t.Fatalf("persisted removal changed reviewed settings: %+v", stored)
			}
			if last {
				if len(result.ModelLoads) != 0 || preparedServingRuntime(f.st, f.st.plat.Runtime) || result.ServingReadback != nil || f.st.plat.Runtime.servingSnapshot != nil {
					t.Fatalf("last external removal did not return to default mode: %+v", result)
				}
			} else if len(result.ModelLoads) != 1 || result.ModelLoads[0].ProfileKey != "medium-model" || result.ModelLoads[0].ModelID != "medium-model" || result.ModelLoads[0].Resolved == nil || result.ModelLoads[0].Resolved.PerSlotContext != 8192 || result.ModelLoads[0].Resolved.Slots != 2 || result.ModelLoads[0].Resolved.Artifacts[0].Path != filepath.Join(f.root, "medium.gguf") {
				t.Fatalf("selected removal changed unselected profile: %+v", result.ModelLoads)
			}
			if _, err := os.Stat(f.capture); !os.IsNotExist(err) {
				t.Fatalf("cold removal activated native work: %v", err)
			}
		})
	}
	test("preserves unselected external section", false)
	test("final external section returns to default mode", true)
}

func TestModelSettingsLoadConversionRemovalKeepsUnselectedDeclaration(t *testing.T) {
	f := newModelLoadFixture(t)
	f.rt.Serving = &engines.ServingConfig{Version: 1}
	for _, model := range []struct{ key, file string }{{"weak-model", "weak.gguf"}, {"medium-model", "medium.gguf"}} {
		data, err := os.ReadFile(filepath.Join(f.root, model.file))
		if err != nil {
			t.Fatal(err)
		}
		host, vram := int64(4096), int64(0)
		f.rt.Serving.Models = append(f.rt.Serving.Models, engines.ServingModel{
			ModelID: model.key, MaxOutputTokens: 128,
			Memory:    engines.ServingMemory{HostRAMBytes: &host, VRAMBytes: &vram},
			Artifacts: []engines.ServingArtifact{{Role: "model", SHA256: sha256Hex(data), Provenance: engines.ServingProvenance{Kind: "supplied-sha256"}}},
		})
	}
	f.st.plat.Runtime = f.rt
	request := settingsRequest(t, f.ex)
	request.ModelChange = &modelconfig.ModelChange{ProfileKey: "weak-model", ModelID: "weak-model", Remove: true, ExpectedPolicyRevision: "0"}
	preview, err := f.ex.PreviewLaunch(request)
	if err != nil || len(preview.Errors) != 0 {
		t.Fatalf("declared external removal preview=%+v error=%v", preview, err)
	}
	change := request.ModelChange.Clone()
	change.PreviewDigest = preview.PreviewDigest
	result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings, ReviewedModelChange: &change}, func() error { return nil })
	if err != nil {
		t.Fatalf("declared external removal refused: %v", err)
	}
	if result.Serving == nil || len(result.Serving.Models) != 1 || len(result.ModelLoads) != 1 || result.Serving.Models[0].ModelID != "medium-model" || result.Serving.Models[0].MaxOutputTokens != 128 || result.Serving.Models[0].Memory.HostRAMBytes == nil || *result.Serving.Models[0].Memory.HostRAMBytes != 4096 || result.Serving.Models[0].ExpectedNativeSourceDigest != result.ModelLoads[0].Resolved.SourceInputDigest || !result.Serving.Equal(preview.Settings.Serving) {
		t.Fatalf("unselected serving declaration changed: %+v", result)
	}
}

func TestModelSettingsLoadConversionRemovalRefusesUnreviewedSubset(t *testing.T) {
	test := func(name string, last bool, edit func(*settings.Preview, **modelconfig.ModelChange)) {
		t.Run(name, func(t *testing.T) {
			f, preview, change := reviewedRemovalConversionFixture(t, last)
			edit(&preview, &change)
			_, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings, ReviewedModelChange: change}, func() error { t.Fatal("unreviewed subset reached rebind"); return nil })
			if err == nil {
				t.Fatal("unreviewed external subset accepted")
			}
			assertNoSettingsOverride(t, f.ex)
			if len(f.st.plat.Runtime.ModelLoads) != 0 || f.st.running || f.st.proc != nil {
				t.Fatal("refused subset changed the external router")
			}
		})
	}
	test("no accepted selection", false, func(_ *settings.Preview, change **modelconfig.ModelChange) { *change = nil })
	test("final removal without accepted selection", true, func(_ *settings.Preview, change **modelconfig.ModelChange) { *change = nil })
	test("different selected profile", false, func(_ *settings.Preview, change **modelconfig.ModelChange) { (*change).ProfileKey = "medium-model" })
	test("different selected artifact", false, func(_ *settings.Preview, change **modelconfig.ModelChange) { (*change).ModelID = "medium-model" })
	test("unrelated section also missing", false, func(preview *settings.Preview, _ **modelconfig.ModelChange) { preview.Settings.ModelLoads = nil })
	test("native review digest replaced", false, func(_ *settings.Preview, change **modelconfig.ModelChange) {
		(*change).PreviewDigest = strings.Repeat("f", 64)
	})
	test("final native review digest replaced", true, func(_ *settings.Preview, change **modelconfig.ModelChange) {
		(*change).PreviewDigest = strings.Repeat("f", 64)
	})
	test("final removal has no native review", true, func(_ *settings.Preview, change **modelconfig.ModelChange) { (*change).PreviewDigest = "" })
	test("survivor retargeted", false, func(preview *settings.Preview, _ **modelconfig.ModelChange) {
		preview.Settings.ModelLoads[0].ModelID = "other-artifact"
	})
	test("survivor edited", false, func(preview *settings.Preview, _ **modelconfig.ModelChange) {
		preview.Settings.ModelLoads[0].Requested.Threads = keyTestPointer(int32(4))
	})
}

func TestModelSettingsLoadOrdinaryExternalSettingsKeepSource(t *testing.T) {
	f := newModelLoadFixture(t)
	request := settingsRequest(t, f.ex)
	recheck, err := f.ex.validateTypedLlamaConversion(context.Background(), f.st, f.rt, request.Settings, nil)
	if err != nil || recheck != nil {
		t.Fatalf("ordinary external settings were treated as a conversion: recheck present=%t error=%v", recheck != nil, err)
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsLoadConversionRemovalRefusesStaleSource(t *testing.T) {
	test := func(name string, last bool, edit func(*testing.T, modelLoadFixture)) {
		t.Run(name, func(t *testing.T) {
			f, preview, change := reviewedRemovalConversionFixture(t, last)
			edit(t, f)
			frozen := settings.Request{Engine: "fake", Settings: preview.Settings}
			if err := f.ex.validateReviewedModelRemoval(context.Background(), frozen, change); err == nil {
				t.Fatal("frozen preview accepted stale removal evidence")
			}
			_, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings, ReviewedModelChange: change}, func() error { t.Fatal("stale removal reached rebind"); return nil })
			if err == nil {
				t.Fatal("stale removal accepted")
			}
			assertNoSettingsOverride(t, f.ex)
			if len(f.st.plat.Runtime.ModelLoads) != 0 || f.st.running || f.st.proc != nil {
				t.Fatal("stale removal changed the external router")
			}
		})
	}
	source := func(t *testing.T, f modelLoadFixture) {
		t.Helper()
		data, err := os.ReadFile(f.source)
		if err != nil {
			t.Fatal(err)
		}
		writeTestFile(t, f.source, append(data, []byte("\n# source changed since review\n")...))
	}
	artifact := func(name string) func(*testing.T, modelLoadFixture) {
		return func(t *testing.T, f modelLoadFixture) {
			t.Helper()
			path := filepath.Join(f.root, name+".gguf")
			info, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			writeInspectionGGUF(t, path, "llama", 0.75, 4096)
			if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
				t.Fatal(err)
			}
		}
	}
	test("partial removal source drift", false, source)
	test("final removal source drift", true, source)
	test("removed artifact content changed", false, artifact("weak"))
	test("final removed artifact content changed", true, artifact("weak"))
	test("surviving artifact content changed", false, artifact("medium"))
}

func TestModelSettingsLoadConversionRemovalRetainsCompleteSourceCAS(t *testing.T) {
	f, preview, change := reviewedRemovalConversionFixture(t, true)
	recheck, err := f.ex.validateTypedLlamaConversion(context.Background(), f.st, f.rt, preview.Settings, change)
	if err != nil || recheck == nil {
		t.Fatalf("final removal did not retain complete source evidence: %v", err)
	}
	if err := recheck(); err != nil {
		t.Fatalf("unchanged removal source refused: %v", err)
	}
	replaceNativeConversionLibrary(t, filepath.Join(f.root, "weak.gguf"))
	if err := recheck(); err == nil {
		t.Fatal("removed artifact object replacement passed the final source CAS")
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsLoadConversionFinalRemovalRefusesNativeBuildDrift(t *testing.T) {
	f := nativeModelLoadConversionFixture(t)
	writeTestFile(t, f.source, []byte("[weak-model]\nmodel = weak.gguf\nctx-size = 4096\nparallel = 1\ngpu-layers = 0\n"))
	request := settingsRequest(t, f.ex)
	request.ModelChange = &modelconfig.ModelChange{ModelID: "weak-model", Remove: true, ExpectedPolicyRevision: "0"}
	preview, err := f.ex.PreviewLaunch(request)
	if err != nil || len(preview.Errors) != 0 || len(preview.Settings.ModelLoads) != 0 {
		t.Fatalf("final native-fixture removal preview=%+v error=%v", preview, err)
	}
	change := request.ModelChange.Clone()
	change.PreviewDigest = preview.PreviewDigest
	writeTestFile(t, filepath.Join(f.root, "libggml-base.dylib"), []byte("changed native library content after review"))
	_, err = f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings, ReviewedModelChange: &change}, func() error { t.Fatal("changed build reached rebind"); return nil })
	if err == nil {
		t.Fatal("final removal accepted changed native build without a surviving load binding")
	}
	assertNoSettingsOverride(t, f.ex)
	if f.st.running || f.st.proc != nil || len(f.st.plat.Runtime.ModelLoads) != 0 {
		t.Fatal("build refusal changed or activated the external router")
	}
}

func TestModelSettingsLoadConversionPersistsWhileOff(t *testing.T) {
	f := newModelLoadFixture(t)
	if err := f.ex.desired.set("fake", false, intentPeer); err != nil {
		t.Fatal(err)
	}
	before, err := f.ex.desired.snapshot([]string{"fake"})
	if err != nil {
		t.Fatal(err)
	}
	two := int32(2)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
	rebinds := 0
	result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings, Resume: true}, func() error { rebinds++; return nil })
	if err != nil {
		t.Fatal(err)
	}
	if result.Running || f.st.proc != nil || rebinds != 1 || !settings.ConfigFromLaunch(result, preview.Settings.ProxyPort).Equal(preview.Settings) {
		t.Fatalf("Off conversion result=%+v rebinds=%d", result, rebinds)
	}
	stored := readModelLoadRuntimeOverride(t, f.ex)
	if !modelconfig.EqualLoads(stored.ModelLoads, preview.Settings.ModelLoads) || stored.Serving != nil || stored.Port != f.rt.Port {
		t.Fatal("conversion lost typed settings, endpoint or serving independence")
	}
	after, err := f.ex.desired.snapshot([]string{"fake"})
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatalf("Off conversion changed lifecycle intent: before=%+v after=%+v err=%v", before, after, err)
	}
	if _, err := os.Stat(f.capture); !os.IsNotExist(err) {
		t.Fatal("Off conversion spawned native work", err)
	}
	if err := os.Remove(f.source); err != nil {
		t.Fatal(err)
	}
	readback, err := f.ex.LaunchSettings("fake")
	if err != nil || !modelconfig.EqualLoads(readback.ModelLoads, preview.Settings.ModelLoads) || readback.ServingReadback == nil || readback.ServingReadback.Status == "mismatch" {
		t.Fatalf("converted readback depends on removed external source: readback=%+v err=%v", readback.ServingReadback, err)
	}
}

func TestModelSettingsLoadConversionRejectsSourceChangedSincePreviewBeforeStop(t *testing.T) {
	f := newModelLoadFixture(t)
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	f.st.mu.Lock()
	previous := f.st.proc
	f.st.mu.Unlock()
	two := int32(2)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	changed := bytes.Replace(data, []byte("ctx-size = 16384"), []byte("ctx-size = 32768"), 1)
	if bytes.Equal(data, changed) {
		t.Fatal("fixture did not change the unrelated model context")
	}
	writeTestFile(t, f.source, changed)
	_, err = f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { t.Fatal("stale conversion reached rebind"); return nil })
	if err == nil || !strings.Contains(err.Error(), "changed since preview; preview again") {
		t.Fatalf("stale conversion refusal=%v", err)
	}
	f.st.mu.Lock()
	current, running := f.st.proc, f.st.running
	f.st.mu.Unlock()
	if !running || current != previous || len(f.st.plat.Runtime.ModelLoads) != 0 {
		t.Fatal("stale conversion stopped or changed the current router")
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsLoadConfigureRestartsOwnedRouter(t *testing.T) {
	f := newModelLoadFixture(t)
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	f.st.mu.Lock()
	previous := f.st.proc
	f.st.mu.Unlock()
	if previous == nil {
		t.Fatal("start did not install an owned child")
	}
	two := int32(2)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
	if !preview.Restart || preview.Impact == nil || preview.Impact.RestartScope != "router" || !slices.Equal(preview.Impact.AffectedModels, []string{"medium-model", "weak-model"}) || preview.Impact.DrainTimeoutSeconds == nil || *preview.Impact.DrainTimeoutSeconds != 60 || preview.Impact.MayInterruptActive == nil || !*preview.Impact.MayInterruptActive || preview.Impact.StaysOff == nil || *preview.Impact.StaysOff || preview.Impact.ActiveWorkPolicy != "proceed-after-timeout" {
		t.Fatalf("incomplete running impact: %+v", preview.Impact)
	}
	rebinds := 0
	result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error {
		rebinds++
		select {
		case <-previous.done:
		default:
			t.Fatal("rebind preceded owned termination")
		}
		if !modelconfig.EqualLoads(readSettingsOverride(t, f.ex).ModelLoads, preview.Settings.ModelLoads) {
			t.Fatal("rebind preceded typed persistence")
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	f.st.mu.Lock()
	replacement := f.st.proc
	f.st.mu.Unlock()
	if !result.Running || replacement == nil || replacement == previous || rebinds != 1 || result.Serving != nil || !modelconfig.EqualLoads(result.ModelLoads, preview.Settings.ModelLoads) {
		t.Fatalf("typed restart result=%+v replacement=%p previous=%p rebinds=%d", result, replacement, previous, rebinds)
	}
	_, data := captureModelLoadChild(t, f.capture)
	sections, err := parseLlamaServingPreset(data)
	if err != nil {
		t.Fatal(err)
	}
	if sections["weak-model"]["ctx-size"] != "8192" || sections["weak-model"]["parallel"] != "2" || sections["medium-model"]["ctx-size"] != "16384" {
		t.Fatal("replacement did not consume the complete reviewed allocation")
	}
	if enabled, known, err := f.ex.desired.get("fake"); err != nil || !known || !enabled {
		t.Fatalf("restart changed saved On: enabled=%v known=%v err=%v", enabled, known, err)
	}
}

func TestModelSettingsLoadConfigureNoOpPreservesOwnedProcess(t *testing.T) {
	f := newModelLoadFixture(t)
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	request := settingsRequest(t, f.ex)
	if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: request.Settings}, func() error { return nil }); err != nil {
		t.Fatal(err)
	}
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	f.st.mu.Lock()
	previous := f.st.proc
	f.st.mu.Unlock()
	path := filepath.Join(f.ex.overrideDir, "fake.json")
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	one := int32(1)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &one}})
	if preview.Restart || preview.Impact == nil || preview.Impact.RestartScope != "none" || preview.Impact.MayInterruptActive == nil || *preview.Impact.MayInterruptActive {
		t.Fatalf("no-op proposed interruption: %+v", preview.Impact)
	}
	useEngineFileReplacer(t, durablefile.Replacer{Replace: func(string, string) error { t.Fatal("no-op attempted persistence"); return nil }})
	result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { return nil })
	if err != nil || !result.Running {
		t.Fatalf("no-op apply: result=%+v err=%v", result, err)
	}
	f.st.mu.Lock()
	current := f.st.proc
	f.st.mu.Unlock()
	after, err := os.ReadFile(path)
	if err != nil || current != previous || !bytes.Equal(before, after) {
		t.Fatalf("no-op changed owned process or disk: err=%v", err)
	}
}

func TestModelSettingsLoadConversionRestartsOwnedExternalRouter(t *testing.T) {
	f := newModelLoadFixture(t)
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	f.st.mu.Lock()
	previous := f.st.proc
	f.st.mu.Unlock()
	two := int32(2)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
	result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	f.st.mu.Lock()
	replacement := f.st.proc
	f.st.mu.Unlock()
	if !result.Running || previous == nil || replacement == nil || replacement == previous || !modelconfig.EqualLoads(result.ModelLoads, preview.Settings.ModelLoads) {
		t.Fatalf("external conversion did not replace the owned router: %+v", result)
	}
	args, data := captureModelLoadChild(t, f.capture)
	index := slices.Index(args, "--models-preset")
	if index < 0 || index+1 >= len(args) || args[index+1] == f.source {
		t.Fatal("converted router still consumes the mutable external source")
	}
	sections, err := parseLlamaServingPreset(data)
	if err != nil {
		t.Fatal(err)
	}
	if sections["weak-model"]["parallel"] != "2" || sections["medium-model"]["ctx-size"] != "16384" {
		t.Fatal("converted router lost reviewed or unrelated model settings")
	}
}

func TestModelSettingsLoadConfigureVetoKeepsSavedConfigurationCold(t *testing.T) {
	test := func(name string, veto func(*testing.T, modelLoadFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newModelLoadFixture(t)
			veto(t, f)
			before, err := f.ex.desired.snapshot([]string{"fake"})
			if err != nil {
				t.Fatal(err)
			}
			two := int32(2)
			preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
			if preview.Impact == nil || preview.Impact.StaysOff == nil || !*preview.Impact.StaysOff {
				t.Fatalf("cold veto impact: %+v", preview.Impact)
			}
			result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings, Resume: true}, func() error { return nil })
			if err != nil || result.Running || f.st.proc != nil || !modelconfig.EqualLoads(result.ModelLoads, preview.Settings.ModelLoads) {
				t.Fatalf("vetoed configuration result=%+v err=%v", result, err)
			}
			after, err := f.ex.desired.snapshot([]string{"fake"})
			if err != nil || !reflect.DeepEqual(before, after) {
				t.Fatalf("configuration changed intent: before=%+v after=%+v err=%v", before, after, err)
			}
			if _, err := os.Stat(f.capture); !os.IsNotExist(err) {
				t.Fatal("configuration bypassed activation veto", err)
			}
		})
	}
	test("paused by either origin", func(t *testing.T, f modelLoadFixture) {
		f.ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	})
}

func TestModelSettingsLoadConfigureRejectsDriftBeforeStop(t *testing.T) {
	test := func(name string, drift func(*testing.T, modelLoadFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newModelLoadFixture(t)
			f.st.plat.Runtime = convertModelLoadFixture(t, f)
			if err := f.ex.Start(context.Background(), "fake"); err != nil {
				t.Fatal(err)
			}
			f.st.mu.Lock()
			previous := f.st.proc
			f.st.mu.Unlock()
			threads := int32(6)
			preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Threads: &threads}})
			drift(t, f)
			_, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { t.Fatal("drift reached rebind"); return nil })
			if err == nil {
				t.Fatal("stale typed resolution was accepted")
			}
			f.st.mu.Lock()
			current, running := f.st.proc, f.st.running
			f.st.mu.Unlock()
			if !running || current != previous {
				t.Fatal("source refusal stopped or replaced the owned child")
			}
			assertNoSettingsOverride(t, f.ex)
		})
	}
	test("artifact binding", func(t *testing.T, f modelLoadFixture) {
		writeInspectionGGUF(t, filepath.Join(f.root, "weak.gguf"), "llama", 0.75, 4096)
	})
	test("native build binding", func(t *testing.T, f modelLoadFixture) {
		path := filepath.Join(f.root, "different-native-build")
		writeTestFile(t, path, []byte("different synthetic native build"))
		f.st.mu.Lock()
		f.st.binPath = path
		f.st.mu.Unlock()
	})
}

func TestModelSettingsLoadConfigureAdoptedRouterHasNoEffects(t *testing.T) {
	f := newModelLoadFixture(t)
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	threads := int32(6)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Threads: &threads}})
	before := snapshotLlamaState(f.st)
	f.st.mu.Lock()
	f.st.running, f.st.adopted = true, true
	f.st.mu.Unlock()
	t.Cleanup(func() {
		f.st.mu.Lock()
		f.st.running, f.st.adopted = false, false
		f.st.mu.Unlock()
	})
	if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { t.Fatal("adopted router reached rebind"); return nil }); err == nil {
		t.Fatal("adopted router accepted a load change")
	}
	if !modelconfig.EqualLoads(before.plat.Runtime.ModelLoads, f.st.plat.Runtime.ModelLoads) || !f.st.running || !f.st.adopted || f.st.proc != nil {
		t.Fatal("refused apply changed adopted ownership or settings")
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsLoadConversionRechecksCompleteExternalSource(t *testing.T) {
	f := newModelLoadFixture(t)
	two := int32(2)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
	recheck, err := f.ex.validateTypedLlamaConversion(context.Background(), f.st, f.rt, preview.Settings, nil)
	if err != nil || recheck == nil {
		t.Fatalf("conversion source not captured: %v", err)
	}
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, f.source, append(data, []byte("\n# changed after preparation\n")...))
	if err := recheck(); err == nil {
		t.Fatal("external bytes changed without invalidating conversion")
	}
	assertNoSettingsOverride(t, f.ex)
	if f.st.running || f.st.proc != nil {
		t.Fatal("conversion revalidation activated the router")
	}
}

func nativeModelLoadConversionFixture(t *testing.T) modelLoadFixture {
	t.Helper()
	f := newModelLoadFixture(t)
	autoTestNativeSource(t, &f)
	f.rt = f.st.plat.Runtime
	command, err := launchForState(f.st, f.rt.Port)
	if err != nil {
		t.Fatal(err)
	}
	f.command = command
	return f
}

// A replacement with identical bytes and metadata must still fail an existing
// operation's object-evidence fence, even though a fresh build ID would agree.
func replaceNativeConversionLibrary(t *testing.T, path string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(path, filepath.Join(filepath.Dir(path), "saved-original")); err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, path, data)
	if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
}

func TestModelLoadConversionCapturesLibrariesBeforeArtifactReads(t *testing.T) {
	f := nativeModelLoadConversionFixture(t)
	library := filepath.Join(f.root, "libggml-base.dylib")
	replace, replaced, replacementOpens := false, false, 0
	access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
		if replace && !replaced && path == filepath.Join(f.root, "weak.gguf") {
			replaceNativeConversionLibrary(t, library)
			replaced = true
		}
		if replaced && path == library {
			replacementOpens++
		}
		return os.Open(path)
	}}
	positive, err := f.ex.previewTypedLlamaConversionExpected(context.Background(), f.st, f.command, f.rt, access, nil)
	if err != nil || len(positive.Runtime.ModelLoads) == 0 || positive.inputs == nil {
		t.Fatalf("positive native conversion=%+v error=%v", positive, err)
	}
	if _, ok := positive.inputs.libraries["libggml-base.dylib"]; !ok {
		t.Fatal("conversion omitted native library evidence")
	}
	replace = true
	if _, err := f.ex.previewTypedLlamaConversionExpected(context.Background(), f.st, f.command, f.rt, access, nil); err == nil {
		t.Fatal("conversion recaptured a replacement library after reading artifacts")
	}
	if !replaced || replacementOpens != 0 {
		t.Fatalf("replacement triggered=%v opens=%d", replaced, replacementOpens)
	}
}

func TestModelLoadConversionApplyRetainsLibraryEvidence(t *testing.T) {
	f := nativeModelLoadConversionFixture(t)
	two := int32(2)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
	if len(preview.Settings.ModelLoads) == 0 {
		t.Fatal("preview has no accepted loads")
	}
	recheck, err := f.ex.validateTypedLlamaConversion(context.Background(), f.st, f.rt, preview.Settings, nil)
	if err != nil || recheck == nil {
		t.Fatalf("conversion apply validation=%v", err)
	}
	if err := recheck(); err != nil {
		t.Fatal("unchanged conversion replay failed", err)
	}
	replaceNativeConversionLibrary(t, filepath.Join(f.root, "libggml-base.dylib"))
	if err := recheck(); err == nil {
		t.Fatal("apply replay adopted a replacement native library")
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelLoadConversionRejectsMissingExpectedLibraryEvidence(t *testing.T) {
	f := nativeModelLoadConversionFixture(t)
	conversion, err := f.ex.previewTypedLlamaConversion(context.Background(), f.st, f.command, f.rt)
	if err != nil || conversion.inputs == nil || len(conversion.Runtime.ModelLoads) == 0 {
		t.Fatalf("conversion=%+v error=%v", conversion, err)
	}
	conversion.inputs.libraries = nil
	opens := 0
	access := servingFileAccess{open: func(path string) (servingArtifactFile, error) { opens++; return os.Open(path) }}
	if _, err := f.ex.previewTypedLlamaConversionExpected(context.Background(), f.st, f.command, f.rt, access, conversion.inputs); err == nil {
		t.Fatal("incomplete expected evidence was rediscovered")
	}
	if opens != 0 {
		t.Fatalf("incomplete expected evidence opened %d files", opens)
	}
}

func TestModelSettingsLoadConversionRefusesPartialRouterBeforeEffects(t *testing.T) {
	f := newModelLoadFixture(t)
	two := int32(2)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
	preview.Settings.ModelLoads = slices.Delete(preview.Settings.ModelLoads, 0, 1)
	if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { t.Fatal("partial conversion reached rebind"); return nil }); err == nil {
		t.Fatal("conversion dropped an existing router model")
	}
	if f.st.running || f.st.proc != nil || len(f.st.plat.Runtime.ModelLoads) != 0 {
		t.Fatal("partial conversion changed native state")
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsLoadConfigurePersistenceFailureRetainsOldRuntime(t *testing.T) {
	f := newModelLoadFixture(t)
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	request := settingsRequest(t, f.ex)
	if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: request.Settings}, func() error { return nil }); err != nil {
		t.Fatal(err)
	}
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	before := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
	path := filepath.Join(f.ex.overrideDir, "fake.json")
	disk, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	threads := int32(6)
	// Include the nested placement option in the existing transaction failure
	// fixture: no new requested/effective state may survive a failed save.
	experts := &modelconfig.CPUExperts{Mode: "layers", Layers: keyTestPointer(int32(2))}
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Threads: &threads, CPUExperts: experts}})
	failure := errors.New("synthetic override preparation failure")
	useEngineFileReplacer(t, durablefile.Replacer{CreateTemp: func(dir, base string, _ os.FileMode) (durablefile.File, error) {
		if dir == f.ex.overrideDir {
			return nil, failure
		}
		return os.CreateTemp(dir, base)
	}})
	result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { t.Fatal("failed persistence reached rebind"); return nil })
	if !errors.Is(err, failure) || result.Running || !modelconfig.EqualLoads(result.ModelLoads, before) {
		t.Fatalf("failed persistence adopted the new configuration: result=%+v err=%v", result, err)
	}
	after, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(disk, after) || !modelconfig.EqualLoads(f.st.plat.Runtime.ModelLoads, before) {
		t.Fatalf("failed persistence changed disk or runtime: %v", err)
	}
}

func TestModelSettingsLoadConversionSourceDriftBeforeRestartRefuses(t *testing.T) {
	f := newModelLoadFixture(t)
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	two := int32(2)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
	result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error {
		writeTestFile(t, f.source, []byte("external source replaced after native persistence"))
		if err := os.Remove(f.capture); err != nil {
			t.Fatal(err)
		}
		return nil
	})
	if err == nil || result.Running || f.st.proc != nil || !modelconfig.EqualLoads(result.ModelLoads, preview.Settings.ModelLoads) {
		t.Fatalf("source drift was not retained as a cold accepted failure: result=%+v err=%v", result, err)
	}
	if _, err := os.Stat(f.capture); !os.IsNotExist(err) {
		t.Fatal("changed conversion source spawned a replacement", err)
	}
}

func TestModelSettingsLoadConfigureRechecksSourcesAfterRebind(t *testing.T) {
	test := func(name string, drift func(*testing.T, modelLoadFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newModelLoadFixture(t)
			f.st.plat.Runtime = convertModelLoadFixture(t, f)
			if err := f.ex.Start(context.Background(), "fake"); err != nil {
				t.Fatal(err)
			}
			threads := int32(6)
			preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Threads: &threads}})
			result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error {
				drift(t, f)
				if err := os.Remove(f.capture); err != nil {
					t.Fatal(err)
				}
				return nil
			})
			if err == nil || result.Running || f.st.proc != nil || !modelconfig.EqualLoads(result.ModelLoads, preview.Settings.ModelLoads) {
				t.Fatalf("post-rebind drift result=%+v err=%v", result, err)
			}
			if _, err := os.Stat(f.capture); !os.IsNotExist(err) {
				t.Fatal("changed source spawned a replacement", err)
			}
		})
	}
	test("artifact binding", func(t *testing.T, f modelLoadFixture) {
		writeInspectionGGUF(t, filepath.Join(f.root, "weak.gguf"), "llama", 0.75, 4096)
	})
	test("content source generation", func(t *testing.T, f modelLoadFixture) {
		f.ex.downloads.mu.Lock()
		f.ex.downloads.source = "replaced-during-rebind"
		f.ex.downloads.mu.Unlock()
	})
	test("native build binding", func(t *testing.T, f modelLoadFixture) {
		path := filepath.Join(f.root, "different-native-build")
		writeTestFile(t, path, []byte("different synthetic native build"))
		f.st.mu.Lock()
		f.st.binPath = path
		f.st.plat.Detect = []string{path}
		f.st.mu.Unlock()
	})
}

func TestModelSettingsLoadSetPortUsesTypedConfigurationOwner(t *testing.T) {
	f := newModelLoadFixture(t)
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	before := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	result, err := f.ex.SetPort(context.Background(), "fake", port)
	if err != nil {
		t.Fatal(err)
	}
	stored := readModelLoadRuntimeOverride(t, f.ex)
	if result.Running || f.st.proc != nil || stored.Port != port || len(stored.ModelLoads) != len(before) || stored.Serving != nil {
		t.Fatalf("typed port result=%+v stored=%+v", result, stored)
	}
	for i, load := range stored.ModelLoads {
		if load.ModelID != before[i].ModelID || !reflect.DeepEqual(load.Requested, before[i].Requested) || !reflect.DeepEqual(load.Resolved.Artifacts, before[i].Resolved.Artifacts) {
			t.Fatal("port change altered models, requested allocation or artifacts")
		}
	}
	if _, err := compilePreparedServing(f.st, typedFixtureCommand(t, f, f.st.plat.Runtime), f.st.plat.Runtime, true); err != nil {
		t.Fatal("port path persisted a stale native resolution", err)
	}
}

func TestModelSettingsLoadPinsAutomaticPortWithRestartableIdentity(t *testing.T) {
	f := newModelLoadFixture(t)
	f.rt.Port = 0
	f.st.plat.Runtime = f.rt
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	request := settingsRequest(t, f.ex)
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	request.Settings.ServerPort = port
	command, err := launchForState(f.st, port)
	if err != nil {
		t.Fatal(err)
	}
	request.Settings.LaunchText, err = command.argumentText(llamaServingBase(f.st.plat.Runtime))
	if err != nil {
		t.Fatal(err)
	}
	preview, err := f.ex.PreviewLaunch(request)
	if err != nil || len(preview.Errors) != 0 || preview.Conflict != nil {
		t.Fatalf("pin automatic port preview: errors=%v conflict=%v err=%v", preview.Errors, preview.Conflict, err)
	}
	result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { return nil })
	if err != nil || !result.Running || result.EffectivePort != port {
		t.Fatalf("pin automatic port: result=%+v err=%v", result, err)
	}
	stored := readModelLoadRuntimeOverride(t, f.ex)
	if stored.Port != port || !modelconfig.EqualLoads(stored.ModelLoads, result.ModelLoads) {
		t.Fatal("pinned endpoint and native identity were not persisted together")
	}
	if err := f.ex.Stop("fake"); err != nil {
		t.Fatal(err)
	}
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal("pinned native identity failed next start", err)
	}
}

func TestModelSettingsLoadSetPortPinsAutomaticEndpoint(t *testing.T) {
	f := newModelLoadFixture(t)
	f.rt.Port = 0
	f.st.plat.Runtime = f.rt
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	result, err := f.ex.SetPort(context.Background(), "fake", port)
	if err != nil || result.Running || result.Port != port {
		t.Fatalf("pin automatic endpoint: result=%+v err=%v", result, err)
	}
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal("pinned port configuration failed start", err)
	}
}

func TestModelSettingsLoadPersistencePreservesOpaqueFields(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	host := runtime.GOOS + "/" + runtime.GOARCH
	path := filepath.Join(f.ex.overrideDir, "fake.json")
	writeTestFile(t, path, []byte(fmt.Sprintf(`{"engine":"fake","opaque":900719925474099312345,"runtime":{"opaque":1.0000000000000000001},"platforms":{"%s":{"runtime":{"opaque":9007199254740993}},"other/platform":{"runtime":{"port":42}}}}`, host)))
	serving := servingSettingsFixture()
	if err := f.ex.persistRuntimeServing("fake", rt.Port, rt.LaunchArgs, rt.LaunchEnv, &serving, &rt.ModelLoads); err != nil {
		t.Fatal(err)
	}
	if err := f.ex.persistPort("fake", rt.Port+1); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var record struct {
		Opaque    json.Number
		Runtime   struct{ Opaque json.Number }
		Platforms map[string]struct {
			Runtime struct {
				Opaque     json.Number
				Port       int
				ModelLoads []modelconfig.ModelLoad
				Serving    *engines.ServingConfig
				LaunchArgs []string `json:"launch_args"`
			}
		}
	}
	if err := json.Unmarshal(data, &record); err != nil {
		t.Fatal(err)
	}
	if record.Opaque.String() != "900719925474099312345" || record.Runtime.Opaque.String() != "1.0000000000000000001" || record.Platforms[host].Runtime.Opaque.String() != "9007199254740993" {
		t.Fatal("opaque numeric precision changed")
	}
	persisted := record.Platforms[host].Runtime
	if !modelconfig.EqualLoads(persisted.ModelLoads, rt.ModelLoads) || !persisted.Serving.Equal(serving) || record.Platforms["other/platform"].Runtime.Port != 42 || !slices.Equal(persisted.LaunchArgs, *rt.LaunchArgs) {
		t.Fatal("persistence lost unrelated model/platform/declaration/launch state")
	}
	if f.st.running || f.st.proc != nil {
		t.Fatal("private persistence activated stopped engine")
	}
	cleared := []modelconfig.ModelLoad{}
	if err := f.ex.persistRuntimeServing("fake", rt.Port, nil, nil, nil, &cleared); err != nil {
		t.Fatal(err)
	}
	if got := readSettingsOverride(t, f.ex); len(got.ModelLoads) != 0 || !got.Serving.Equal(serving) {
		t.Fatal("reset failed to clear loads or changed declaration")
	}
}

func TestModelSettingsLoadPersistenceFailurePreservesState(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	path := filepath.Join(f.ex.overrideDir, "fake.json")
	invalid := []byte(`{"engine":"fake","runtime":{"opaque":9007199254740993}}`)
	writeTestFile(t, path, invalid)
	failure := errors.New("synthetic override replacement failure")
	useEngineFileReplacer(t, durablefile.Replacer{Replace: func(string, string) error { return failure }})
	before := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
	if err := f.ex.persistRuntimeServing("fake", rt.Port, rt.LaunchArgs, rt.LaunchEnv, &rt.Serving, &rt.ModelLoads); !errors.Is(err, failure) {
		t.Fatalf("expected replacement failure, got %v", err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(invalid, after) || !modelconfig.EqualLoads(before, f.st.plat.Runtime.ModelLoads) || f.st.running {
		t.Fatal("failed persistence changed disk/runtime")
	}
}

func TestModelSettingsLoadConversionRejectsCompleteRouter(t *testing.T) {
	test := func(name, extra string) {
		t.Run(name, func(t *testing.T) {
			f := newModelLoadFixture(t)
			data, err := os.ReadFile(f.source)
			if err != nil {
				t.Fatal(err)
			}
			data = append(data, []byte(extra)...)
			writeTestFile(t, f.source, data)
			before := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
			_, refusal := f.ex.previewTypedLlamaConversion(context.Background(), f.st, f.command, f.rt)
			if refusal == nil {
				t.Fatal("unsupported router member partially converted")
			}
			after, err := os.ReadFile(f.source)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(data, after) || !modelconfig.EqualLoads(before, f.st.plat.Runtime.ModelLoads) {
				t.Fatal("refused conversion changed authority")
			}
		})
	}
	test("sampling field", "temp = 0.4\n")
	test("third model", "[third]\nmodel = third.gguf\ncache-prompt = true\n")
	test("unified KV", "kv-unified = true\n")
}

func TestModelSettingsLoadConversionCPUExpertsRoundTrip(t *testing.T) {
	f := newModelLoadFixture(t)
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, f.source, append(data, []byte("\nn-cpu-moe = 2\n")...))
	conversion, err := f.ex.previewTypedLlamaConversion(context.Background(), f.st, f.command, f.rt)
	if err != nil {
		t.Fatal(err)
	}
	rt := conversion.Runtime
	load := rt.ModelLoads[0]
	if load.ModelID != "medium-model" || load.Requested.CPUExperts == nil || load.Requested.CPUExperts.Mode != "layers" || load.Requested.CPUExperts.Layers == nil || *load.Requested.CPUExperts.Layers != 2 {
		t.Fatalf("conversion lost expert layers: %+v", load)
	}
	prepared, err := compilePreparedServing(f.st, typedFixtureCommand(t, f, rt), rt, false)
	if err != nil {
		t.Fatal(err)
	}
	sections, err := parseLlamaServingPreset(prepared.Snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if sections["medium-model"]["n-cpu-moe"] != "2" || sections["medium-model"]["cpu-moe"] != "" || sections["weak-model"]["n-cpu-moe"] != "" {
		t.Fatalf("converted native options changed: %+v", sections)
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsLoadCPUExpertsFiniteOptions(t *testing.T) {
	blocks := int32(24)
	for _, tc := range []struct {
		name        string
		experts     *modelconfig.CPUExperts
		all, layers string
	}{
		{"inherited", nil, "", ""},
		{"none", &modelconfig.CPUExperts{Mode: "none"}, "", ""},
		{"all", &modelconfig.CPUExperts{Mode: "all"}, "true", ""},
		{"layers twelve", &modelconfig.CPUExperts{Mode: "layers", Layers: keyTestPointer(int32(12))}, "", "12"},
		{"layers zero", &modelconfig.CPUExperts{Mode: "layers", Layers: keyTestPointer(int32(0))}, "", "0"},
		{"layers full", &modelconfig.CPUExperts{Mode: "layers", Layers: &blocks}, "", "24"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			requested := modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "cpu"}, CPUExperts: tc.experts}
			options, resolved, err := typedLlamaModelOptions("exact", requested, &blocks)
			if err != nil {
				t.Fatal(err)
			}
			if options["cpu-moe"] != tc.all || options["n-cpu-moe"] != tc.layers || !reflect.DeepEqual(resolved.Options.CPUExperts, tc.experts) {
				t.Fatalf("native=%v resolved=%+v", options, resolved.Options)
			}
			if tc.experts != nil && resolved.Options.CPUExperts == tc.experts {
				t.Fatal("finite resolution retains caller's CPU expert pointer")
			}
			effective := modelEffectiveLoadOptions(resolved.Options)
			if !reflect.DeepEqual(effective.CPUExperts, tc.experts) || tc.experts != nil && effective.CPUExperts == resolved.Options.CPUExperts {
				t.Fatal("effective options dropped CPU experts or retained an alias")
			}
		})
	}
	for _, tc := range []struct {
		name    string
		experts *modelconfig.CPUExperts
		blocks  *int32
	}{
		{"over block count", &modelconfig.CPUExperts{Mode: "layers", Layers: keyTestPointer(int32(25))}, &blocks},
		{"output layer is not a block", &modelconfig.CPUExperts{Mode: "layers", Layers: keyTestPointer(int32(24))}, keyTestPointer(int32(23))},
		{"missing block count", &modelconfig.CPUExperts{Mode: "layers", Layers: keyTestPointer(int32(0))}, nil},
		{"negative layers", &modelconfig.CPUExperts{Mode: "layers", Layers: keyTestPointer(int32(-1))}, &blocks},
		{"missing layers", &modelconfig.CPUExperts{Mode: "layers"}, &blocks},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := typedLlamaModelOptions("exact", modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "cpu"}, CPUExperts: tc.experts}, tc.blocks)
			var field *ServingFieldError
			if !errors.As(err, &field) || field.Field != "modelLoads.cpuExperts.layers" {
				t.Fatalf("want CPU expert layers field error, got %v", err)
			}
		})
	}
}

func TestModelSettingsLoadCPUExpertsSaveReloadClear(t *testing.T) {
	for _, tc := range []struct {
		name  string
		patch modelconfig.LoadPatch
		want  *modelconfig.CPUExperts
	}{
		{"save layers", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{CPUExperts: &modelconfig.CPUExperts{Mode: "layers", Layers: keyTestPointer(int32(2))}}}, &modelconfig.CPUExperts{Mode: "layers", Layers: keyTestPointer(int32(2))}},
		{"explicit none", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{CPUExperts: &modelconfig.CPUExperts{Mode: "none"}}}, &modelconfig.CPUExperts{Mode: "none"}},
		{"clear to inheritance", modelconfig.LoadPatch{Clear: []string{"cpuExperts"}}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newModelLoadFixture(t)
			f.st.plat.Runtime = convertModelLoadFixture(t, f)
			initial := previewModelLoadFixture(t, f, "medium-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{CPUExperts: &modelconfig.CPUExperts{Mode: "all"}}})
			if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: initial.Settings}, func() error { return nil }); err != nil {
				t.Fatal(err)
			}
			preview := previewModelLoadFixture(t, f, "medium-model", tc.patch)
			result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { return nil })
			if err != nil {
				t.Fatal(err)
			}
			reloaded := readModelLoadRuntimeOverride(t, f.ex)
			if result.Running || !modelconfig.EqualLoads(result.ModelLoads, reloaded.ModelLoads) || !reflect.DeepEqual(reloaded.ModelLoads[0].Requested.CPUExperts, tc.want) || !reflect.DeepEqual(reloaded.ModelLoads[0].Resolved.Options.CPUExperts, tc.want) {
				t.Fatalf("CPU expert intent or effective state changed: result=%+v reload=%+v", result, reloaded.ModelLoads)
			}
			if reloaded.ModelLoads[1].Requested.CPUExperts != nil {
				t.Fatal("CPU expert edit changed sibling model")
			}
		})
	}
}

func TestModelSettingsLoadCPUExpertsConversionRefusesOutOfRange(t *testing.T) {
	f := newModelLoadFixture(t)
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, f.source, append(data, []byte("\nn-cpu-moe = 38\n")...))
	_, err = f.ex.previewTypedLlamaConversion(context.Background(), f.st, f.command, f.rt)
	var field *ServingFieldError
	if !errors.As(err, &field) || field.Field != "modelLoads.cpuExperts.layers" {
		t.Fatalf("want bounded CPU expert layer error, got %v", err)
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsLoadCPUExpertsInheritanceAndDetachment(t *testing.T) {
	f := newModelLoadFixture(t)
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, f.source, []byte(strings.Replace(string(data), "[*]\n", "[*]\ncpu-moe = true\n", 1)))
	rt := convertModelLoadFixture(t, f)
	for _, load := range rt.ModelLoads {
		if load.Requested.CPUExperts == nil || load.Requested.CPUExperts.Mode != "all" || load.Resolved.Options.CPUExperts == nil || load.Resolved.Options.CPUExperts.Mode != "all" {
			t.Fatalf("lost inherited external expert placement: %+v", load)
		}
	}
	patched, err := patchModelLoad(rt.ModelLoads, "medium-model", "medium-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{CPUExperts: &modelconfig.CPUExperts{Mode: "layers", Layers: keyTestPointer(int32(2))}}})
	if err != nil {
		t.Fatal(err)
	}
	*patched[0].Requested.CPUExperts.Layers = 1
	patched[0].Resolved.Options.CPUExperts.Mode = "none"
	if rt.ModelLoads[0].Requested.CPUExperts.Mode != "all" || rt.ModelLoads[0].Resolved.Options.CPUExperts.Mode != "all" || patched[1].Requested.CPUExperts.Mode != "all" {
		t.Fatal("patch or resolved clone changed inherited state")
	}
}

func TestModelSettingsLoadCPUExpertsDescriptors(t *testing.T) {
	options, err := llamaModelSettingsOptions()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ native, field string }{{"cpu-moe", "cpuExperts"}, {"n-cpu-moe", "cpuExperts.layers"}} {
		t.Run(tc.native, func(t *testing.T) {
			i := slices.IndexFunc(options, func(o modelSettingsOption) bool { return o.Native.key == tc.native })
			if i < 0 {
				t.Fatal("missing expert placement descriptor")
			}
			d := options[i].Descriptor
			if d.Field != tc.field || !d.Supported || !d.Portable || d.Owner != "model-load" || options[i].Native.unsupported {
				t.Fatalf("unmapped descriptor: %+v", options[i])
			}
			if tc.native == "cpu-moe" && !slices.Equal(d.Values, []string{"none", "all", "layers"}) {
				t.Fatalf("expert modes=%v", d.Values)
			}
			if tc.native == "n-cpu-moe" && (d.Minimum == nil || d.Minimum.String() != "0") {
				t.Fatalf("expert layer minimum=%v", d.Minimum)
			}
		})
	}
}

func TestModelSettingsLoadFiniteValidation(t *testing.T) {
	layers := int32(4)
	test := func(name string, r modelconfig.RequestedLoad, field string) {
		t.Run(name, func(t *testing.T) {
			_, _, err := typedLlamaModelOptions("exact", r, &layers)
			var typed *ServingFieldError
			if !errors.As(err, &typed) || !strings.Contains(typed.Field, field) {
				t.Fatalf("expected %s refusal, got %v", field, err)
			}
		})
	}
	zero, large := int32(0), int32(6)
	batch, ubatch := int32(128), int32(256)
	precision := "q8_0"
	test("zero slots", modelconfig.RequestedLoad{Slots: &zero}, "requested")
	test("overflow", modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "fixed", Tokens: math.MaxInt32}}, "requested")
	test("too many layers", modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "layers", Layers: &large}}, "gpu.layers")
	test("batch inversion", modelconfig.RequestedLoad{BatchSize: &batch, UBatchSize: &ubatch}, "requested")
	test("unmapped precision", modelconfig.RequestedLoad{CacheTypeV: &precision}, "cacheTypeV")
	test("Auto unavailable", modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "auto"}}, "gpu")
}

func TestModelSettingsLoadPreviewPreservesCompleteRouterAndSources(t *testing.T) {
	f := newModelLoadFixture(t)
	f.command.Env["LLAMA_ARG_LOG_COLORS"] = "off"
	f.command.Args = append(f.command.Args, "--log-file", "model")
	conversion, err := f.ex.previewTypedLlamaConversion(context.Background(), f.st, f.command, f.rt)
	if err != nil {
		t.Fatal(err)
	}
	rt := conversion.Runtime
	if !slices.Contains(*rt.LaunchArgs, "--log-disable") || slices.Contains(*rt.LaunchArgs, "--models-preset") {
		t.Fatalf("conversion lost unrelated options or retained source: %v", *rt.LaunchArgs)
	}
	logFile := slices.Index(*rt.LaunchArgs, "--log-file")
	if logFile < 0 || logFile+1 >= len(*rt.LaunchArgs) || (*rt.LaunchArgs)[logFile+1] != "model" {
		t.Fatalf("conversion consumed an unrelated flag value: %v", *rt.LaunchArgs)
	}
	env, err := literalEnvironment(*rt.LaunchEnv)
	if err != nil {
		t.Fatal(err)
	}
	if env["LLAMA_ARG_LOG_COLORS"] != "off" {
		t.Fatal("conversion dropped unrelated native environment")
	}
	if f.st.plat.Runtime.ModelLoads != nil || f.st.running {
		t.Fatal("conversion preview changed settings or activation")
	}
	f.st.plat.Runtime = rt
	state, err := f.ex.LaunchSettings("fake")
	if err != nil {
		t.Fatal(err)
	}
	preview, err := f.ex.PreviewLaunch(settings.Request{Engine: "fake", Settings: settings.ConfigFromLaunch(state, 54301)})
	if err != nil || len(preview.Errors) != 0 {
		t.Fatalf("typed preview refused valid input: %+v %v", preview.Errors, err)
	}
	if preview.Settings.ServerPort != rt.Port || preview.Settings.ProxyPort != 54301 || !modelconfig.EqualLoads(preview.Settings.ModelLoads, rt.ModelLoads) || preview.Settings.Serving != nil {
		t.Fatal("typed preview changed unrelated settings or accepted resolution")
	}
	if preview.Restart {
		t.Fatal("unchanged Off preview requested a restart")
	}
	if _, err := os.Stat(filepath.Join(f.ex.overrideDir, "fake.json")); !os.IsNotExist(err) {
		t.Fatal("preview persisted settings", err)
	}
}

func TestModelSettingsLoadDeclaredServingValidatesGeneratedBytes(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	host, vram := int64(4096), int64(0)
	rt.Serving = &engines.ServingConfig{Version: 1}
	for _, load := range rt.ModelLoads {
		data, err := os.ReadFile(load.Resolved.Artifacts[0].Path)
		if err != nil {
			t.Fatal(err)
		}
		rt.Serving.Models = append(rt.Serving.Models, engines.ServingModel{ModelID: load.ModelID, MaxOutputTokens: 128, Memory: engines.ServingMemory{HostRAMBytes: &host, VRAMBytes: &vram}, Artifacts: []engines.ServingArtifact{{Role: "model", SHA256: sha256Hex(data), Provenance: engines.ServingProvenance{Kind: "supplied-sha256"}}}})
	}
	command := typedFixtureCommand(t, f, rt)
	prepared, err := compilePreparedServing(f.st, command, rt, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(prepared.Envelopes) != len(rt.Serving.Models) || !reflect.DeepEqual(prepared.Config.Models[0].Memory, rt.Serving.Models[0].Memory) || prepared.Config.Models[0].Artifacts[0].SHA256 != rt.Serving.Models[0].Artifacts[0].SHA256 {
		t.Fatal("typed compilation altered declaration memory or asserted content")
	}
	rt.Serving = prepared.Config
	rt.ModelLoads = prepared.ModelLoads
	if _, err := compilePreparedServing(f.st, command, rt, true); err != nil {
		t.Fatal("reviewed declaration cannot validate generated bytes", err)
	}
	rt.Serving.Models[0].MaxOutputTokens = 100000
	if _, err := compilePreparedServing(f.st, command, rt, false); err == nil {
		t.Fatal("generated preset bypassed declaration output bound")
	}
}

func TestModelSettingsLoadInheritedCapacityChangesApproval(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	command := typedFixtureCommand(t, f, rt)
	index := slices.Index(command.Args, "--models-max")
	if index < 0 || index+1 >= len(command.Args) {
		t.Fatal("fixture has no explicit capacity to remove")
	}
	command.Args = append(slices.Clone(command.Args[:index]), command.Args[index+2:]...)
	t.Setenv("LLAMA_ARG_MODELS_MAX", "1")
	first, err := compilePreparedServing(f.st, command, rt, false)
	if err != nil {
		t.Fatal(err)
	}
	rt.ModelLoads = first.ModelLoads
	t.Setenv("LLAMA_ARG_MODELS_MAX", "2")
	if _, err := compilePreparedServing(f.st, command, rt, true); err == nil {
		t.Fatal("changed inherited router capacity retained approval")
	}
	second, err := compilePreparedServing(f.st, command, rt, false)
	if err != nil {
		t.Fatal(err)
	}
	if first.Digest == second.Digest {
		t.Fatal("effective router capacity is absent from typed identity")
	}
}

func TestModelSettingsLoadWorkerConversionPreviewAndApply(t *testing.T) {
	f := newModelLoadFixture(t)
	state, err := f.ex.LaunchSettings("fake")
	if err != nil {
		t.Fatal(err)
	}
	before := snapshotLlamaState(f.st)
	three := int32(3)
	request := settings.Request{Engine: "fake", Settings: settings.ConfigFromLaunch(state, 54301), ModelChange: &modelconfig.ModelChange{ModelID: "medium-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &three}}}}
	server, client := net.Pipe()
	t.Cleanup(func() { _ = server.Close(); _ = client.Close() })
	if err := client.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	manager := &Manager{exec: f.ex, codec: NewCodec(server)}
	rebinds := 0
	manager.settingsRelay.send = func(_ context.Context, method string, value any) error {
		relay, ok := value.(settings.Relay)
		if !ok || method != "engine:settings-request" || relay.Method != "rebind" || !modelconfig.EqualLoads(relay.Request.Settings.ModelLoads, f.st.plat.Runtime.ModelLoads) {
			return fmt.Errorf("unexpected rebind: %s", method)
		}
		rebinds++
		raw, err := json.Marshal(settingsReply{ID: relay.ID, Result: json.RawMessage(`{}`)})
		if err != nil {
			return err
		}
		manager.settingsRelay.reply(raw)
		return nil
	}
	call := func(method string, params any) Message {
		t.Helper()
		if method == "engine:preview-launch" {
			params = settings.WorkerRequest[any]{Request: params}
		}
		raw, err := json.Marshal(params)
		if err != nil {
			t.Fatal(err)
		}
		id := json.RawMessage(`"typed-preview"`)
		if !manager.handleSettingsMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: method, Params: raw}) {
			t.Fatalf("worker did not handle %s", method)
		}
		var reply Message
		if err := json.NewDecoder(client).Decode(&reply); err != nil {
			t.Fatal(err)
		}
		return reply
	}
	reply := call("engine:preview-launch", request)
	if reply.Error != nil {
		t.Fatalf("conversion preview refused: %+v", reply.Error)
	}
	var preview settings.Preview
	if err := modelconfig.DecodeStrict(reply.Result, &preview); err != nil {
		t.Fatal(err)
	}
	if len(preview.Errors) != 0 || len(preview.Settings.ModelLoads) != 2 || len(preview.PreviewDigest) != 64 || preview.Restart || preview.Impact == nil || preview.Impact.RestartScope != "router" || preview.Impact.DrainTimeoutSeconds == nil || *preview.Impact.DrainTimeoutSeconds != 60 || len(preview.Impact.AffectedModels) != 2 {
		t.Fatalf("incomplete read-only preview: errors=%v digest=%s", preview.Errors, preview.PreviewDigest)
	}
	if preview.Settings.ModelLoads[0].ModelID != "medium-model" || preview.Settings.ModelLoads[0].Resolved.TotalContext != 24576 || preview.Settings.ModelLoads[1].Resolved.TotalContext != 4096 || preview.Settings.Serving != nil {
		t.Fatal("conversion did not preserve complete router and selected allocation")
	}
	var found bool
	for _, diff := range preview.Diff {
		if diff.Field != "models.medium-model.ctx-size" {
			continue
		}
		var old, next string
		if err := json.Unmarshal(diff.Before, &old); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(diff.After, &next); err != nil {
			t.Fatal(err)
		}
		if old != "16384" || next != "24576" {
			t.Fatalf("context diff %q -> %q", old, next)
		}
		found = true
	}
	if !found {
		t.Fatal("conversion preview omitted changed native context")
	}
	if !reflect.DeepEqual(before.plat.Runtime, f.st.plat.Runtime) || f.st.running || f.st.proc != nil {
		t.Fatal("preview changed runtime")
	}
	if _, err := os.Stat(f.capture); !os.IsNotExist(err) {
		t.Fatal("preview spawned native work", err)
	}
	if _, err := os.Stat(filepath.Join(f.ex.overrideDir, "fake.json")); !os.IsNotExist(err) {
		t.Fatal("preview persisted settings", err)
	}
	applied := call("engine:configure-launch", settings.Configure{Engine: "fake", OperationID: "reviewed-conversion", Settings: preview.Settings})
	if applied.Error != nil {
		t.Fatalf("accepted conversion RPC refused: %+v", applied.Error)
	}
	var result settings.LaunchState
	if err := modelconfig.DecodeStrict(applied.Result, &result); err != nil {
		t.Fatal(err)
	}
	if !settings.ConfigFromLaunch(result, preview.Settings.ProxyPort).Equal(preview.Settings) || result.Running || f.st.proc != nil || rebinds != 1 || len(manager.settingsRelay.pending) != 0 {
		t.Fatalf("conversion RPC lost reviewed settings or activated: %+v rebinds=%d", result, rebinds)
	}
	if !modelconfig.EqualLoads(readSettingsOverride(t, f.ex).ModelLoads, preview.Settings.ModelLoads) {
		t.Fatal("conversion RPC did not persist accepted loads")
	}
	request.Settings.ServerPort++
	stale := call("engine:preview-launch", request)
	if stale.Error == nil {
		var invalid settings.Preview
		if err := modelconfig.DecodeStrict(stale.Result, &invalid); err != nil {
			t.Fatal(err)
		}
		if len(invalid.Errors) == 0 || invalid.PreviewDigest != "" {
			t.Fatal("stale baseline produced reviewed conversion")
		}
	}
}

func TestModelSettingsLoadContentFenceAndNativeBuildIdentity(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	command := typedFixtureCommand(t, f, rt)
	release, err := f.ex.downloads.acquireWrite("fake")
	if err != nil {
		t.Fatal(err)
	}
	_, compileErr := compilePreparedServing(f.st, command, rt, false)
	release()
	if compileErr == nil {
		t.Fatal("compiler bypassed content writer fence")
	}
	// A synthetic executable binding changes identity without running it.
	executable := filepath.Join(f.root, "synthetic-native")
	writeTestFile(t, executable, []byte("first synthetic native build"))
	command.Bin = executable
	first, err := compilePreparedServing(f.st, command, rt, false)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, executable, []byte("second synthetic native build"))
	second, err := compilePreparedServing(f.st, command, rt, false)
	if err != nil {
		t.Fatal(err)
	}
	if first.Digest == second.Digest || first.ModelLoads[0].Resolved.NativeBuildID == second.ModelLoads[0].Resolved.NativeBuildID {
		t.Fatal("native build binding omitted from typed identity")
	}
}

func TestModelLoadManualNativeBuildID(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	nativeBuildTestRuntime(t, f.st, t.TempDir())
	rt.Bin = f.st.binPath
	f.st.plat.Install = &Install{Fetch: &Fetch{URL: "https://fixture.invalid/native.tar.gz", SHA256: strings.Repeat("a", 64)}}
	prepared, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, rt), rt, false)
	if err != nil {
		t.Fatal(err)
	}
	if prepared == nil || len(prepared.ModelLoads) == 0 || len(prepared.ModelLoads) != len(rt.ModelLoads) {
		t.Fatalf("manual compilation lost model loads: %+v", prepared)
	}
	want := requireNativeBuildID(t, f.st)
	for _, load := range prepared.ModelLoads {
		if load.Resolved == nil || load.Resolved.NativeBuildID != want {
			t.Fatalf("manual load build = %+v, want %q", load.Resolved, want)
		}
	}
}

func TestModelLoadReviewedAutoNativeBuildID(t *testing.T) {
	f, frozen, owner := autoTestFrozenLaunch(t)
	want := requireNativeBuildID(t, f.st)
	rt := f.st.plat.Runtime
	rt.ModelLoads = modelconfig.CloneLoads(frozen.Settings.ModelLoads)
	if len(rt.ModelLoads) == 0 {
		t.Fatal("Auto fixture has no frozen model loads")
	}
	command, err := modelConstrainCommand(typedFixtureCommand(t, f, rt), rt.ModelLoads)
	if err != nil {
		t.Fatal(err)
	}
	wantSource, err := modelReviewedNativeSource(context.Background(), f.st, command)
	if err != nil {
		t.Fatal(err)
	}
	if wantSource == "" {
		t.Fatal("Auto fixture has no native source identity")
	}
	hasAuto := false
	for _, load := range rt.ModelLoads {
		if load.Resolved == nil || load.Resolved.NativeBuildID != want {
			t.Fatalf("Auto load build = %+v, want %q", load.Resolved, want)
		}
		if modelLoadHasAuto(load.Requested) {
			hasAuto = true
			source := modelAutoNativeSource(load.Resolved)
			if source != wantSource {
				t.Fatalf("Auto native source = %q, want %q", source, wantSource)
			}
		}
	}
	if !hasAuto {
		t.Fatal("Auto fixture has no reviewed Auto load")
	}
	owner.calls = 0
	prepared, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, rt), rt, true)
	if err != nil {
		t.Fatal("frozen Auto replay failed", err)
	}
	if prepared == nil || !modelconfig.EqualLoads(prepared.ModelLoads, rt.ModelLoads) {
		t.Fatalf("frozen replay lost the reviewed model loads: %+v", prepared)
	}
	if owner.calls != 0 {
		t.Fatal("frozen replay consulted telemetry")
	}
}

func TestModelSettingsLoadAutoFrozenRestartCompilation(t *testing.T) {
	f, request, owner := autoTestFrozenLaunch(t)
	rt := f.st.plat.Runtime
	rt.ModelLoads = modelconfig.CloneLoads(request.Settings.ModelLoads)
	command := typedFixtureCommand(t, f, rt)
	before := modelconfig.CloneLoads(rt.ModelLoads)
	owner.calls = 0
	owner.memory = nil
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: owner})
	prepared, err := compileTypedLlamaInputs(ctx, f.st, command, rt, true)
	if err != nil {
		t.Fatal("frozen restart/recovery failed without telemetry", err)
	}
	if owner.calls != 0 || !modelconfig.EqualLoads(before, prepared.ModelLoads) || prepared.Config != nil {
		t.Fatal("finite compilation retuned, consulted telemetry or created serving authority")
	}
	sections, err := parseLlamaServingPreset(prepared.Snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if sections["weak-model"]["fit"] != "off" || sections["weak-model"]["load-on-startup"] != "false" || sections["weak-model"]["ctx-size"] != "4096" {
		t.Fatalf("unexpected native result: %v", sections["weak-model"])
	}
	// Repeated cold compilation models recovery: only persisted settings survive,
	// and a fresh estimator must not be called for a saved allocation.
	f.ex.modelEstimates = modelEstimateState{run: func(context.Context, modelEstimateInput) modelconfig.Estimate {
		t.Fatal("recovery called estimator")
		return modelconfig.Estimate{}
	}}
	again, err := compileTypedLlamaInputs(context.Background(), f.st, command, rt, true)
	if err != nil || again.Digest != prepared.Digest || !bytes.Equal(again.Snapshot, prepared.Snapshot) {
		t.Fatalf("cold compilation changed frozen result: %v", err)
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsLoadAutoPersistedManagerRestartKeepsUUID(t *testing.T) {
	f, frozen, owner := autoTestFrozenLaunch(t)
	if err := f.ex.desired.set("fake", false, intentPeer); err != nil {
		t.Fatal(err)
	}
	if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: frozen.Settings}, func() error { return nil }); err != nil {
		t.Fatal(err)
	}
	manifest := *f.st.manifest
	platform := *f.st.plat
	platform.ModelsDir = f.root
	manifest.Platforms = map[string]Platform{runtime.GOOS + "/" + runtime.GOARCH: platform}
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	registry := NewRegistry()
	registry.bundledRaw["fake"] = data
	if err := registry.LoadOverrideDir(f.ex.overrideDir); err != nil {
		t.Fatal(err)
	}
	recovered := NewExecutor(registry, NewReporter(nil), func(string, any) {}, t.TempDir())
	recovered.overrideDir = f.ex.overrideDir
	recovered.modelEstimates.run = func(context.Context, modelEstimateInput) modelconfig.Estimate {
		t.Fatal("restart consulted cost provider")
		return modelconfig.Estimate{}
	}
	st, err := recovered.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	// Recreate the fixture's selected installed root, never its operation/map.
	st.installDir = f.root
	st.binPath = f.st.binPath
	st.modelsDir = f.root
	command, err := launchForState(st, st.plat.Runtime.Port)
	if err != nil {
		t.Fatal(err)
	}
	owner.calls = 0
	owner.memory = nil
	prepared, err := compileTypedLlamaInputs(context.Background(), st, command, st.plat.Runtime, true)
	if err != nil {
		t.Fatal("persisted restart failed without ephemeral state", err)
	}
	if !modelconfig.EqualLoads(prepared.ModelLoads, frozen.Settings.ModelLoads) || owner.calls != 0 {
		t.Fatal("restart changed reviewed finite settings")
	}
	child := preparedSnapshotCommand(st, command, "private-test-snapshot")
	if child.Env["CUDA_VISIBLE_DEVICES"] != autoTestDeviceBinding().DeviceID {
		t.Fatal("persisted child lost reviewed full UUID")
	}
	writeTestFile(t, filepath.Join(f.root, "libggml-base.dylib"), []byte("replaced after manager restart"))
	if _, err := compileTypedLlamaInputs(context.Background(), st, command, st.plat.Runtime, true); err == nil {
		t.Fatal("persisted replay accepted native library replacement")
	}
}

func TestModelSettingsLoadAutoFrozenRejectsBuildRootDriverLibrary(t *testing.T) {
	for _, name := range []string{"libcuda.so", "LIBCUDA.SO.1", "nvcuda.dll", "NVCUDA.DLL"} {
		t.Run(name, func(t *testing.T) {
			f, request, owner := autoTestFrozenLaunch(t)
			rt := f.st.plat.Runtime
			rt.ModelLoads = modelconfig.CloneLoads(request.Settings.ModelLoads)
			command := typedFixtureCommand(t, f, rt)
			if _, err := compileTypedLlamaInputs(context.Background(), f.st, command, rt, true); err != nil {
				t.Fatal("unchanged frozen compilation failed", err)
			}
			owner.calls = 0
			writeTestFile(t, filepath.Join(f.root, name), []byte("synthetic driver file, never loaded"))
			if _, err := compileTypedLlamaInputs(context.Background(), f.st, command, rt, true); err == nil {
				t.Fatal("frozen compilation accepted build-root driver library")
			}
			if owner.calls != 0 || f.st.running || f.st.proc != nil {
				t.Fatal("driver source refusal resolved or activated an engine")
			}
			assertNoSettingsOverride(t, f.ex)
		})
	}
}

func TestModelSettingsLoadAutoFrozenIdentityDriftRefuses(t *testing.T) {
	test := func(name string, mutate func(*testing.T, modelLoadFixture, *Runtime, *launchCommand)) {
		t.Run(name, func(t *testing.T) {
			f, request, _ := autoTestFrozenLaunch(t)
			rt := f.st.plat.Runtime
			rt.ModelLoads = modelconfig.CloneLoads(request.Settings.ModelLoads)
			command := typedFixtureCommand(t, f, rt)
			mutate(t, f, &rt, &command)
			if _, err := compileTypedLlamaInputs(context.Background(), f.st, command, rt, true); err == nil {
				t.Fatal("frozen identity drift accepted")
			}
			assertNoSettingsOverride(t, f.ex)
			if f.st.running || f.st.proc != nil {
				t.Fatal("identity refusal activated engine")
			}
		})
	}
	test("artifact replacement", func(t *testing.T, _ modelLoadFixture, rt *Runtime, _ *launchCommand) {
		writeInspectionGGUF(t, rt.ModelLoads[1].Resolved.Artifacts[0].Path, "llama", 0.75, 4096)
	})
	test("native build replacement", func(t *testing.T, f modelLoadFixture, _ *Runtime, command *launchCommand) {
		command.Bin = filepath.Join(f.root, "changed-native")
		writeTestFile(t, command.Bin, []byte("changed native build"))
	})
	test("stale source preview digest", func(_ *testing.T, _ modelLoadFixture, rt *Runtime, _ *launchCommand) {
		rt.ModelLoads[1].Resolved.SourceInputDigest = strings.Repeat("f", 64)
	})
	test("stale preset preview digest", func(_ *testing.T, _ modelLoadFixture, rt *Runtime, _ *launchCommand) {
		rt.ModelLoads[1].Resolved.PresetDigest = strings.Repeat("f", 64)
	})
}

func TestModelSettingsLoadAutoConversionSourceRemainsBound(t *testing.T) {
	f := newModelLoadFixture(t)
	autoTestNativeSource(t, &f)
	f.rt.Bin = f.st.binPath
	f.command.Bin = f.st.binPath
	now := time.Now()
	owner := &autoTestMemoryOwner{memory: autoTestMemory(now)}
	f.ex.modelEstimates = modelEstimateState{now: func() time.Time { return now }, run: autoTestEstimate}
	r := autoTestRequest()
	r.Context = &modelconfig.Context{Mode: "auto", Floor: 4096, Cap: 8192}
	request := settingsRequest(t, f.ex)
	request.ModelChange = &modelconfig.ModelChange{ModelID: "weak-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: r}}
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: owner})
	preview, err := f.ex.previewModelLoadChange(ctx, f.st, request)
	if err != nil || len(preview.Errors) != 0 || preview.Auto == nil {
		t.Fatalf("conversion Auto preview: %+v %v", preview, err)
	}
	recheck, err := f.ex.validateTypedLlamaConversion(ctx, f.st, f.rt, preview.Settings, nil)
	if err != nil || recheck == nil {
		t.Fatalf("Auto conversion was not accepted against exact source: %v", err)
	}
	if err := f.ex.validateUnacceptedAutoMemory(ctx, settings.Request{Engine: "fake", Settings: preview.Settings}, "weak-model"); err != nil {
		t.Fatal("frozen conversion resource validation failed", err)
	}
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, f.source, append(data, []byte("\n# source drift\n")...))
	if err := recheck(); err == nil {
		t.Fatal("source drift did not invalidate Auto conversion")
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsLoadAutoRecalculateAfterNativeBuildDrift(t *testing.T) {
	f, frozen, owner := autoTestFrozenLaunch(t)
	f.st.plat.Runtime.ModelLoads = modelconfig.CloneLoads(frozen.Settings.ModelLoads)
	before := modelconfig.CloneLoads(frozen.Settings.ModelLoads)
	writeTestFile(t, f.st.plat.Runtime.Bin, []byte("replacement synthetic Auto native build"))
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: owner})
	owner.calls = 0
	if _, err := f.ex.ConfigureLaunch(ctx, settings.Configure{Engine: "fake", Settings: frozen.Settings}, func() error { t.Fatal("stale Auto reached rebind"); return nil }); err == nil {
		t.Fatal("old Apply accepted native build drift")
	}
	command := typedFixtureCommand(t, f, f.st.plat.Runtime)
	if _, err := compileTypedLlamaInputs(ctx, f.st, command, f.st.plat.Runtime, true); err == nil {
		t.Fatal("old restart accepted native build drift")
	}
	if owner.calls != 0 {
		t.Fatal("Apply/restart consulted capacity after drift")
	}
	request := settingsRequest(t, f.ex)
	request.ModelChange = &modelconfig.ModelChange{ModelID: "weak-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: autoTestRequest()}}
	if err := f.ex.validateModelSettingsLaunch(ctx, request, false); err != nil {
		t.Fatal(err)
	}
	preview, err := f.ex.PreviewLaunch(request, ctx)
	if err != nil || len(preview.Errors) != 0 || preview.Auto == nil || preview.Auto.Resolved == nil {
		t.Fatalf("explicit Recalculate could not recover build drift: %+v %v", preview, err)
	}
	previous := rtLoadResolution(before, "weak-model")
	if owner.calls != 1 || preview.Auto.Resolved.NativeBuildID == previous.NativeBuildID || preview.Auto.ReviewedInputID == previous.ReviewedInputID {
		t.Fatal("Recalculate did not review the new build exactly once")
	}
	if modelAutoConversionSource(preview.Auto.Resolved) != modelAutoConversionSource(previous) {
		t.Fatal("Recalculate lost conversion source")
	}
	if !modelconfig.EqualLoads(before, f.st.plat.Runtime.ModelLoads) || f.st.running || f.st.proc != nil {
		t.Fatal("new preview changed accepted runtime")
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsLoadAutoRecalculateStillChecksFrozenSibling(t *testing.T) {
	f, frozen, owner := autoTestFrozenLaunch(t)
	f.st.plat.Runtime.ModelLoads = modelconfig.CloneLoads(frozen.Settings.ModelLoads)
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: owner})
	request := settingsRequest(t, f.ex)
	r := autoTestRequest()
	r.Context = &modelconfig.Context{Mode: "auto", Floor: 4096, Cap: 8192}
	request.ModelChange = &modelconfig.ModelChange{ModelID: "medium-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: r}}
	if err := f.ex.validateModelSettingsLaunch(ctx, request, false); err != nil {
		t.Fatal(err)
	}
	preview, err := f.ex.PreviewLaunch(request, ctx)
	if err != nil || len(preview.Errors) != 0 || preview.Auto == nil {
		t.Fatalf("frozen sibling setup: %+v %v", preview, err)
	}
	for _, model := range []string{"weak-model", "medium-model"} {
		found := false
		for _, load := range preview.Settings.ModelLoads {
			if load.ModelID == model {
				found = modelLoadHasAuto(load.Requested) && load.Resolved != nil
			}
		}
		if !found {
			t.Fatalf("frozen sibling setup did not retain reviewed Auto for %s", model)
		}
	}
	f.st.plat.Runtime.ModelLoads = modelconfig.CloneLoads(preview.Settings.ModelLoads)
	before := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
	writeTestFile(t, f.st.plat.Runtime.Bin, []byte("replacement build shared with frozen sibling"))
	owner.calls = 0
	request = settingsRequest(t, f.ex)
	request.ModelChange = &modelconfig.ModelChange{ModelID: "weak-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: autoTestRequest()}}
	if err := f.ex.validateModelSettingsLaunch(ctx, request, false); err != nil {
		t.Fatal(err)
	}
	refused, err := f.ex.PreviewLaunch(request, ctx)
	if err != nil {
		t.Fatal(err)
	}
	buildRefusal := false
	for _, reason := range refused.Errors {
		if strings.Contains(reason, "native build changed") {
			buildRefusal = true
		}
	}
	if !buildRefusal || refused.PreviewDigest != "" || owner.calls != 0 {
		t.Fatalf("Recalculate bypassed frozen sibling: errors=%v calls=%d", refused.Errors, owner.calls)
	}
	if !modelconfig.EqualLoads(before, f.st.plat.Runtime.ModelLoads) {
		t.Fatal("refused sibling preview changed accepted settings")
	}
	assertNoSettingsOverride(t, f.ex)
}
