// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelconfig

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/nodepolicy"
)

func testPointer[T any](value T) *T { return &value }

func TestDecodeModelChangeNativeFitSparsePatch(t *testing.T) {
	test := func(name, selection string, want LoadPatch) {
		t.Run(name, func(t *testing.T) {
			change, err := DecodeModelChange([]byte(`{"modelId":"model","expectedPolicyRevision":"0","load":` + selection + `}`))
			if err != nil {
				t.Fatalf("decode sparse patch: %v", err)
			}
			if change.Load == nil || !reflect.DeepEqual(*change.Load, want) {
				t.Fatalf("patch = %+v, want %+v", change.Load, want)
			}
			again, err := DecodeModelChange(mustJSON(t, change))
			if err != nil || !reflect.DeepEqual(again, change) {
				t.Fatalf("sparse patch round trip = %+v, %v", again, err)
			}
		})
	}
	test("target only", `{"set":{"fitTarget":"1024"}}`, LoadPatch{Set: RequestedLoad{FitTarget: testPointer("1024")}})
	test("context only", `{"set":{"fitCtx":8192}}`, LoadPatch{Set: RequestedLoad{FitCtx: testPointer(int32(8192))}})
	test("off and explicit dependent clears", `{"set":{"fit":"off"},"clear":["fitTarget","fitCtx"]}`, LoadPatch{Set: RequestedLoad{Fit: testPointer("off")}, Clear: []string{"fitCtx", "fitTarget"}})
	test("clear fit and dependents", `{"set":{},"clear":["fitTarget","fit","fitCtx"]}`, LoadPatch{Clear: []string{"fit", "fitCtx", "fitTarget"}})
}

func TestDecodeModelChangeNativeFitRejectsMalformedLeaves(t *testing.T) {
	for name, selection := range map[string]string{
		"null target":        `{"fitTarget":null}`,
		"target type":        `{"fitTarget":512}`,
		"target syntax":      `{"fitTarget":"512,"}`,
		"target bound":       `{"fitTarget":"10000000"}`,
		"fractional context": `{"fitCtx":1.5}`,
		"negative context":   `{"fitCtx":-1}`,
		"context bound":      `{"fitCtx":16777217}`,
		"context overflow":   `{"fitCtx":2147483648}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := DecodeModelChange([]byte(`{"modelId":"model","expectedPolicyRevision":"0","load":{"set":` + selection + `}}`)); err == nil {
				t.Fatal("accepted malformed sparse native-fit leaf")
			}
		})
	}
}

func TestCanonicalLoadsKeysByProfile(t *testing.T) {
	t.Run("identity repair", func(t *testing.T) {
		input := []ModelLoad{{ModelID: "org/model:Q4"}}
		got, err := CanonicalLoads(input)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 1 || got[0].ProfileKey != "org/model:Q4" || input[0].ProfileKey != "" {
			t.Fatalf("identity repair = %+v, input = %+v", got, input)
		}
		if err := input[0].Validate(); err == nil {
			t.Fatal("uncanonicalized load accepted an absent profile key")
		}
	})
	t.Run("shared artifact and key ordering", func(t *testing.T) {
		input := []ModelLoad{{ProfileKey: "z-small", ModelID: "a-model"}, {ProfileKey: "a-large", ModelID: "a-model"}}
		got, err := CanonicalLoads(input)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 2 || got[0].ProfileKey != "a-large" || got[1].ProfileKey != "z-small" || got[0].ModelID != "a-model" || got[1].ModelID != "a-model" || input[0].ProfileKey != "z-small" {
			t.Fatalf("canonical loads = %+v, input = %+v", got, input)
		}
		if !EqualLoads(got, input) {
			t.Fatal("load equality depends on profile order")
		}
	})
	t.Run("duplicate key across different models", func(t *testing.T) {
		_, err := CanonicalLoads([]ModelLoad{{ProfileKey: "same", ModelID: "one"}, {ProfileKey: "same", ModelID: "two"}})
		if err == nil || !strings.Contains(err.Error(), "profileKey: duplicate") {
			t.Fatalf("duplicate key error = %v", err)
		}
	})
	t.Run("repair collides with explicit key", func(t *testing.T) {
		if _, err := CanonicalLoads([]ModelLoad{{ModelID: "one"}, {ProfileKey: "one", ModelID: "two"}}); err == nil {
			t.Fatal("accepted duplicate repaired key")
		}
	})
	t.Run("invalid explicit key", func(t *testing.T) {
		if _, err := CanonicalLoads([]ModelLoad{{ProfileKey: "coder:fast", ModelID: "model"}}); err == nil {
			t.Fatal("accepted a reserved profile key on a valid model")
		}
	})
	t.Run("identity repair refuses reserved key", func(t *testing.T) {
		_, err := CanonicalLoads([]ModelLoad{{ModelID: "qwen3:8b"}})
		if err == nil || !strings.Contains(err.Error(), "profileKey: profile key qwen3:8b is reserved: the text before its colon is a valid role name") {
			t.Fatalf("reserved repaired key error = %v", err)
		}
	})
}

func TestLoadArtifactIDIsNotAKey(t *testing.T) {
	load := ModelLoad{ProfileKey: "lfm-8k", ModelID: "medium:local"}
	if err := load.Validate(); err != nil {
		t.Fatalf("valid profile with reserved-alias artifact ID: %v", err)
	}
	load.ProfileKey = ""
	if err := load.Validate(); err == nil || !strings.Contains(err.Error(), "profileKey:") {
		t.Fatalf("missing profile key error = %v", err)
	}
}

func TestCoderProfileKeysRemainValid(t *testing.T) {
	for _, key := range []string{"weak", "coder-api", "coder-local"} {
		t.Run(key, func(t *testing.T) {
			load := ModelLoad{ProfileKey: key, ModelID: "medium:local"}
			if err := load.Validate(); err != nil {
				t.Fatalf("valid coder profile key: %v", err)
			}
		})
	}
}

func TestModelChangeProfileValidation(t *testing.T) {
	accept := func(name, selection string) {
		t.Run(name, func(t *testing.T) {
			input := []byte(`{"modelId":"model","expectedPolicyRevision":"0",` + selection + `}`)
			got, err := DecodeModelChange(input)
			if err != nil {
				t.Fatal(err)
			}
			again, err := DecodeModelChange(mustJSON(t, got))
			if err != nil || !reflect.DeepEqual(again, got) {
				t.Fatalf("change round trip = %+v, %v", again, err)
			}
		})
	}
	accept("implicit model key", `"policy":{"displayName":"My model"}`)
	accept("explicit key", `"profileKey":"model-8k","policy":{"displayName":"My model"}`)
	accept("icon only", `"policy":{"icon":"code"}`)
	accept("clear icon only", `"policy":{"icon":""}`)
	accept("remove only", `"profileKey":"model-8k","remove":true`)
	for name, selection := range map[string]string{
		"reserved key":       `"profileKey":"coder:fast","remove":true`,
		"invalid key":        `"profileKey":"bad\nkey","remove":true`,
		"empty display":      `"policy":{"displayName":""}`,
		"oversized display":  `"policy":{"displayName":"` + strings.Repeat("x", 129) + `"}`,
		"control display":    `"policy":{"displayName":"bad\nname"}`,
		"null display":       `"policy":{"displayName":null}`,
		"unknown icon":       `"policy":{"icon":"unknown"}`,
		"null icon":          `"policy":{"icon":null}`,
		"remove and load":    `"remove":true,"load":{"set":{"slots":1}}`,
		"remove and policy":  `"remove":true,"policy":{"icon":""}`,
		"false remove alone": `"remove":false`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := DecodeModelChange([]byte(`{"modelId":"model","expectedPolicyRevision":"0",` + selection + `}`)); err == nil {
				t.Fatal("accepted invalid profile edit")
			}
		})
	}
}

func TestModelChangeCloneDetachesDisplayPatches(t *testing.T) {
	c := ModelChange{ProfileKey: "model-8k", ModelID: "model", Policy: &PolicyPatch{DisplayName: testPointer("Original"), Icon: testPointer("")}}
	clone := c.Clone()
	*clone.Policy.DisplayName, *clone.Policy.Icon = "Changed", "code"
	if *c.Policy.DisplayName != "Original" || *c.Policy.Icon != "" || c.ProfileKey != "model-8k" {
		t.Fatal("clone aliases display patches")
	}
}

func TestModelChangePreservesProfileSelections(t *testing.T) {
	change, err := DecodeModelChange([]byte(`{"profileKey":"model-8k","modelId":"artifact","expectedPolicyRevision":"0","policy":{"displayName":"Small context","icon":""}}`))
	if err != nil {
		t.Fatal(err)
	}
	if change.ProfileKey != "model-8k" || change.ModelID != "artifact" || change.Remove || change.Policy.DisplayName == nil || *change.Policy.DisplayName != "Small context" || change.Policy.Icon == nil || *change.Policy.Icon != "" {
		t.Fatalf("selected profile metadata = %+v", change)
	}
	remove, err := DecodeModelChange([]byte(`{"profileKey":"model-8k","modelId":"artifact","expectedPolicyRevision":"0","remove":true}`))
	if err != nil {
		t.Fatal(err)
	}
	if !remove.Remove || remove.ProfileKey != "model-8k" || remove.Load != nil || remove.Policy != nil {
		t.Fatalf("remove selection = %+v", remove)
	}
}

func TestInspectRequestProfileKey(t *testing.T) {
	r, err := DecodeInspectRequest([]byte(`{"engine":"llamacpp","modelId":"model","profileKey":"model-8k"}`))
	if err != nil || r.ProfileKey != "model-8k" || r.ModelID != "model" {
		t.Fatalf("profile inspection = %+v, %v", r, err)
	}
	if _, err := DecodeInspectRequest([]byte(`{"engine":"llamacpp","modelId":"model","profileKey":"coder:fast"}`)); err == nil {
		t.Fatal("inspection accepted a reserved profile key")
	}
}

func TestInspectGPUSelectionTransport(t *testing.T) {
	r := InspectResult{Engine: "llamacpp", ModelID: "model", Status: "unknown", Estimate: Estimate{Status: "unknown"}, ConfigurationKeyReason: "gpu-selection", ExecutionKey: strings.Repeat("a", 64)}
	got, err := DecodeInspectResult(mustJSON(t, r))
	if err != nil || !reflect.DeepEqual(got, r) {
		t.Fatalf("transport=%+v err=%v", got, err)
	}
	r.ConfigurationPools = []string{"host"}
	if r.Validate() == nil {
		t.Fatal("GPU selection accepted a finite scope")
	}
}

func TestInspectNativeFitAndNativeOptionsTransport(t *testing.T) {
	result := InspectResult{Engine: "llamacpp", ModelID: "model", Status: "unknown", Estimate: Estimate{Status: "unknown"}, ConfigurationKeyReason: "native-fit", ExecutionKey: strings.Repeat("a", 64), NativeOptions: []NativeOption{{Name: "fit", Kind: "choice", Choices: []string{"on", "off"}}}, UnreportedLaunchOptions: []string{"cache-ram"}}
	got, err := DecodeInspectResult(mustJSON(t, result))
	if err != nil || !reflect.DeepEqual(got, result) {
		t.Fatalf("transport=%+v err=%v", got, err)
	}
	clone := result.Clone()
	clone.NativeOptions[0].Choices[0] = "changed"
	clone.UnreportedLaunchOptions[0] = "changed"
	if result.NativeOptions[0].Choices[0] != "on" || result.UnreportedLaunchOptions[0] != "cache-ram" {
		t.Fatal("native option clone aliases source")
	}
	for _, tc := range []struct {
		name   string
		mutate func(*InspectResult)
	}{
		{"finite and execution keys", func(r *InspectResult) {
			r.ConfigurationKey = strings.Repeat("b", 64)
			r.ConfigurationPools = []string{"host"}
		}},
		{"execution without unpriced reason", func(r *InspectResult) { r.ConfigurationKeyReason = "not-applied" }},
		{"invalid execution digest", func(r *InspectResult) { r.ExecutionKey = "bad" }},
		{"unknown kind", func(r *InspectResult) { r.NativeOptions[0].Kind = "unknown" }},
		{"invalid flag name", func(r *InspectResult) { r.NativeOptions[0].Name = "--fit" }},
		{"duplicate hints", func(r *InspectResult) { r.NativeOptions = append(r.NativeOptions, r.NativeOptions[0]) }},
		{"switch with choices", func(r *InspectResult) { r.NativeOptions[0].Kind = "switch" }},
		{"warning without known help", func(r *InspectResult) { r.NativeOptions = nil }},
		{"warning for reported option", func(r *InspectResult) { r.UnreportedLaunchOptions = []string{"fit"} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			invalid := result.Clone()
			tc.mutate(&invalid)
			if _, err := DecodeInspectResult(mustJSON(t, invalid)); err == nil {
				t.Fatal("accepted invalid native inspection")
			}
		})
	}
}

func TestInspectNativeOptionsUnknownAndKnownEmpty(t *testing.T) {
	for _, options := range [][]NativeOption{nil, {}} {
		result := InspectResult{Engine: "llamacpp", ModelID: "model", Status: "unknown", Estimate: Estimate{Status: "unknown"}, NativeOptions: options}
		got, err := DecodeInspectResult(mustJSON(t, result))
		if err != nil || (got.NativeOptions == nil) != (options == nil) {
			t.Fatalf("known/unknown help changed: %+v, %v", got.NativeOptions, err)
		}
	}
}

func TestDecodeModelChangeCPUExpertsSelection(t *testing.T) {
	for _, selection := range []string{
		`"set":{"cpuExperts":{"mode":"none"}}`,
		`"set":{"cpuExperts":{"mode":"all"}}`,
		`"set":{"cpuExperts":{"mode":"layers","layers":12}}`,
		`"set":{},"clear":["cpuExperts"]`,
	} {
		t.Run(selection, func(t *testing.T) {
			data := []byte(`{"modelId":"model","expectedPolicyRevision":"0","load":{` + selection + `}}`)
			change, err := DecodeModelChange(data)
			if err != nil {
				t.Fatal(err)
			}
			again, err := DecodeModelChange(mustJSON(t, change))
			if err != nil || !reflect.DeepEqual(again, change) {
				t.Fatalf("selection changed across transport: %+v, %v", again, err)
			}
			if change.Load.Set.CPUExperts == nil && !reflect.DeepEqual(change.Load.Clear, []string{"cpuExperts"}) {
				t.Fatal("lost CPU expert selection")
			}
		})
	}
	for _, selection := range []string{
		`"set":{"cpuExperts":{"mode":"all"}},"clear":["cpuExperts"]`,
		`"set":{"cpuExperts":null}`,
		`"set":{"cpuExperts":{"mode":"layers","layers":1.5}}`,
		`"set":{"cpuExperts":{"mode":"layers","layers":2147483648}}`,
		`"set":{"cpuExperts":{"mode":"all","other":1}}`,
	} {
		t.Run("reject "+selection, func(t *testing.T) {
			if _, err := DecodeModelChange([]byte(`{"modelId":"model","expectedPolicyRevision":"0","load":{` + selection + `}}`)); err == nil {
				t.Fatal("accepted ambiguous CPU expert selection")
			}
		})
	}
}

func TestPublicMemoryAuthorityRejected(t *testing.T) {
	for _, field := range []string{"memory", "Memory", "deviceBinding", "nativeMap", "hop", "trusted", "settings"} {
		t.Run(field, func(t *testing.T) {
			data := []byte(`{"engine":"llamacpp","modelId":"model","` + field + `":{}}`)
			if _, err := DecodeInspectRequest(data); err == nil {
				t.Fatal("public inspection accepted worker authority")
			}
			data = []byte(`{"modelId":"model","expectedPolicyRevision":"0","load":{"set":{"slots":1,"` + field + `":{}}}}`)
			if _, err := DecodeModelChange(data); err == nil {
				t.Fatal("public change accepted worker authority")
			}
		})
	}
}

func testResolvedLoad() ModelLoad {
	return ModelLoad{
		ProfileKey: "exact/Model:Q4",
		ModelID:    "exact/Model:Q4",
		Requested:  RequestedLoad{Context: &Context{Mode: "fixed", Tokens: 8193}, Slots: testPointer(int32(2)), GPU: &GPU{Mode: "layers", Layers: testPointer(int32(12))}, KVOffload: testPointer(false)},
		Resolved: &ResolvedLoad{
			PerSlotContext: 8448, TotalContext: 16896, Slots: 2,
			GPU:           GPU{Mode: "layers", Layers: testPointer(int32(12))},
			Options:       RequestedLoad{KVOffload: testPointer(false)},
			Artifacts:     []ArtifactBinding{{Role: "model", Path: "/models/first.gguf", ResolvedTarget: "/blobs/first", Size: "9007199254740993", MtimeNS: "1780000000000000001", SampleSHA256: strings.Repeat("a", 64)}},
			NativeBuildID: "pinned-build", SourceInputDigest: strings.Repeat("b", 64), PresetDigest: strings.Repeat("c", 64),
		},
	}
}

func TestModelChangeRejectsRemovedBindingSelection(t *testing.T) {
	const valid = `{"modelId":"model","expectedPolicyRevision":"0","policy":{"served":true}}`
	if _, err := DecodeModelChange([]byte(valid)); err != nil {
		t.Fatalf("retained served selection: %v", err)
	}
	for _, field := range []string{`"binding":true`, `"Binding":false`, `"binding":null`} {
		t.Run(field, func(t *testing.T) {
			input := strings.Replace(valid, `"served":true`, `"served":true,`+field, 1)
			if _, err := DecodeModelChange([]byte(input)); err == nil || !strings.Contains(err.Error(), "unknown field") {
				t.Fatalf("removed binding selection error = %v, want unknown field", err)
			}
		})
	}
}

func TestDecodeModelChangeRejectsAmbiguousPayloads(t *testing.T) {
	for name, input := range map[string]string{
		"unknown":               `{"modelId":"x","unknown":1}`,
		"wrong case":            `{"ModelId":"x"}`,
		"duplicate nested":      `{"load":{"set":{"slots":1,"slots":2}}}`,
		"case folded duplicate": `{"modelId":"x","MODELID":"y"}`,
		"escaped duplicate":     `{"modelId":"x","model\u0049d":"y"}`,
		"duplicate generation":  `{"policy":{"generation":{"seed":1,"seed":2}}}`,
		"unknown load":          `{"load":{"set":{"tensorSplit":[1]}}}`,
		"resolved authority":    `{"load":{"set":{"resolved":{"slots":1}}}}`,
		"unknown policy":        `{"policy":{"localOnly":false}}`,
		"null selected load":    `{"load":{"set":{"slots":null}}}`,
		"null selected policy":  `{"policy":{"served":null}}`,
		"null role":             `{"policy":{"roles":{"weak":null}}}`,
		"arbitrary role value":  `{"policy":{"roles":{"weak":"another-model"}}}`,
		"root null":             `null`,
		"trailing":              `{} {}`,
		"oversize":              `{"modelId":"` + strings.Repeat("x", MaxPayloadBytes) + `"}`,
		"deep nesting":          `{"policy":{"generation":{"seed":` + strings.Repeat("[", 34) + `0` + strings.Repeat("]", 34) + `}}}`,
		"invalid UTF-8":         "{\"modelId\":\"\xff\"}",
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := DecodeModelChange([]byte(input)); err == nil {
				t.Fatal("accepted ambiguous or unauthorized shape")
			}
		})
	}
}

func TestDecodeModelChangePreservesSelectedClearsAndPrecision(t *testing.T) {
	input := `{"modelId":" exact/Model:Q4 ","expectedPolicyRevision":"9223372036854775807","load":{"set":{"kvOffload":false},"clear":["threads","slots"]},"policy":{"generation":{"seed":9007199254740991,"temperature":0.100000000000000001,"top_p":null},"roles":{"weak":false,"strong":true},"served":false}}`
	got, err := DecodeModelChange([]byte(input))
	if err != nil {
		t.Fatal(err)
	}
	if got.ModelID != " exact/Model:Q4 " || got.ExpectedPolicyRevision != "9223372036854775807" || got.Policy.Generation["temperature"].String() != "0.100000000000000001" || got.Policy.Generation["seed"].String() != "9007199254740991" {
		t.Fatalf("changed exact identity or numeric lexemes: %+v", got)
	}
	clear, selected := got.Policy.Generation["top_p"]
	if !selected || clear != nil || got.Policy.Served == nil || *got.Policy.Served || got.Load.Set.KVOffload == nil || *got.Load.Set.KVOffload || !reflect.DeepEqual(got.Load.Clear, []string{"slots", "threads"}) {
		t.Fatal("lost selected clear, false or canonical clear order")
	}
	data, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	again, err := DecodeModelChange(data)
	if err != nil || !reflect.DeepEqual(got, again) {
		t.Fatalf("round trip differs: %v", err)
	}
}

func TestModelChangeRejectsIncompleteAndConflictingSelections(t *testing.T) {
	for name, selection := range map[string]string{
		"no selection":   ``,
		"empty load":     `,"load":{"set":{}}`,
		"set and clear":  `,"load":{"set":{"slots":1},"clear":["slots"]}`,
		"repeated clear": `,"load":{"set":{},"clear":["slots","slots"]}`,
		"unknown clear":  `,"load":{"set":{},"clear":["device"]}`,
		"invalid mode":   `,"policy":{"requestOptionsMode":"force"}`,
		"empty policy":   `,"policy":{}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := DecodeModelChange([]byte(`{"modelId":"m","expectedPolicyRevision":"0"` + selection + `}`)); err == nil {
				t.Fatal("accepted incomplete or conflicting selection")
			}
		})
	}
}

func TestPolicyRevisionIsCanonicalSignedInt64Text(t *testing.T) {
	for _, revision := range []string{"", "01", "-1", "+1", "1e2", "9223372036854775808"} {
		t.Run(revision, func(t *testing.T) {
			c := ModelChange{ModelID: "m", ExpectedPolicyRevision: revision, Policy: &PolicyPatch{Served: testPointer(true)}}
			if _, err := DecodeModelChange(mustJSON(t, c)); err == nil {
				t.Fatal("accepted noncanonical or overflowing policy revision")
			}
		})
	}
}

func TestModelIdentifiersMatchNodePolicy(t *testing.T) {
	for _, id := range []string{"", "model", "Model", " model ", "a:b/c", "日本語", strings.Repeat("x", 512), strings.Repeat("x", 513), "bad\nmodel", "bad\u0085model", string([]byte{0xff}), "weak", "medium:api", "medium:local", "strong:local", "coder-api", "coder-local", "Weak", "weak:local-api", "weak:api-local", "weak:latest"} {
		t.Run(id, func(t *testing.T) {
			wantValid := nodepolicy.ValidateModelIdentifier(id) == nil
			_, err := DecodeInspectRequest(mustJSON(t, InspectRequest{Engine: "llamacpp", ModelID: id}))
			// Invalid UTF-8 must be checked before encoding replaces it.
			if id == string([]byte{0xff}) {
				err = validateModelID(id)
			}
			if (err == nil) != wantValid {
				t.Fatalf("model validity differs from nodepolicy: err=%v, wantValid=%v", err, wantValid)
			}
		})
	}
}

func mustJSON(t *testing.T, value any) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestInspectRequestBoundaryIsReadOnly(t *testing.T) {
	for name, input := range map[string]string{
		"settings":       `{"engine":"llamacpp","modelId":"m","settings":{}}`,
		"telemetry":      `{"engine":"llamacpp","modelId":"m","memoryBytes":"1"}`,
		"native":         `{"engine":"llamacpp","modelId":"m","nativeBuildId":"mine"}`,
		"invalid engine": `{"engine":"../llamacpp","modelId":"m"}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := DecodeInspectRequest([]byte(input)); err == nil {
				t.Fatal("accepted non-inspect input")
			}
		})
	}
}

func TestCanonicalLoadsCopiesAndOrdersWithoutSortingArtifacts(t *testing.T) {
	m := testResolvedLoad()
	m.Resolved.Artifacts = append(m.Resolved.Artifacts,
		ArtifactBinding{Role: "model", Path: "/models/second.gguf", ResolvedTarget: "/blobs/second", Size: "1", MtimeNS: "-1", SampleSHA256: strings.Repeat("d", 64)},
		ArtifactBinding{Role: "draft", Path: "/models/draft.gguf", ResolvedTarget: "/blobs/draft", Size: "2", MtimeNS: "0", SampleSHA256: strings.Repeat("e", 64)})
	input := []ModelLoad{{ProfileKey: "z", ModelID: "z"}, m}
	got, err := DecodeLoads(mustJSON(t, input))
	if err != nil {
		t.Fatal(err)
	}
	if got[0].ModelID != m.ModelID || !EqualLoads(input, got) || got[0].Resolved.Artifacts[0].Path != "/models/first.gguf" || got[0].Resolved.Artifacts[2].Role != "draft" {
		t.Fatal("canonical order changed identity or artifact semantics")
	}
	if input[0].ModelID != "z" {
		t.Fatal("canonicalization mutated its input")
	}
}

func TestCloneLoadsDetachesAllRequestedFieldsAndResolution(t *testing.T) {
	m := testResolvedLoad()
	m.Requested = RequestedLoad{
		Context: &Context{Mode: "fixed", Tokens: 8193}, Slots: testPointer(int32(2)), GPU: &GPU{Mode: "layers", Layers: testPointer(int32(12))},
		CacheTypeK: testPointer("f16"), CacheTypeV: testPointer("f16"), FlashAttention: testPointer("on"), BatchSize: testPointer(int32(8)), UBatchSize: testPointer(int32(4)),
		Threads: testPointer(int32(3)), ThreadsBatch: testPointer(int32(4)), KVOffload: testPointer(false), OpOffload: testPointer(true), CacheRAM: testPointer(int32(512)),
	}
	input := []ModelLoad{m}
	before := mustJSON(t, input)
	got := CloneLoads(input)
	got[0].Requested.Context.Tokens++
	*got[0].Requested.Slots++
	*got[0].Requested.GPU.Layers++
	*got[0].Requested.CacheTypeK = "q8_0"
	*got[0].Requested.CacheTypeV = "q4_0"
	*got[0].Requested.FlashAttention = "off"
	*got[0].Requested.BatchSize++
	*got[0].Requested.UBatchSize++
	*got[0].Requested.Threads++
	*got[0].Requested.ThreadsBatch++
	*got[0].Requested.KVOffload = true
	*got[0].Requested.OpOffload = false
	*got[0].Requested.CacheRAM++
	*got[0].Resolved.GPU.Layers++
	*got[0].Resolved.Options.KVOffload = true
	got[0].Resolved.Artifacts[0].Path = "changed"
	if !reflect.DeepEqual(before, mustJSON(t, input)) || EqualLoads(input, got) {
		t.Fatal("clone aliases requested/resolved input")
	}
	if CloneLoads(nil) != nil || CloneLoads([]ModelLoad{}) == nil {
		t.Fatal("clone lost nil versus explicit empty")
	}
}

func TestCloneDoesNotDiscardInvalidRecords(t *testing.T) {
	invalid := []ModelLoad{{ModelID: "m", Resolved: &ResolvedLoad{Artifacts: []ArtifactBinding{{Size: "invalid"}}}}}
	if !reflect.DeepEqual(invalid, CloneLoads(invalid)) {
		t.Fatal("clone silently changed invalid readback")
	}
	if _, err := CanonicalLoads(invalid); err == nil {
		t.Fatal("canonicalization accepted invalid readback")
	}
}

func TestModelChangeCloneDetachesSelectionsAndInvalidNumbers(t *testing.T) {
	c := ModelChange{ModelID: "m", ExpectedPolicyRevision: "0", Load: &LoadPatch{Set: RequestedLoad{Slots: testPointer(int32(1))}, Clear: []string{"threads"}}, Policy: &PolicyPatch{Generation: map[string]*json.Number{"seed": testPointer(json.Number("not-a-number")), "top_p": nil}, Roles: map[string]bool{"weak": true}, Served: testPointer(false)}}
	copy := c.Clone()
	if copy.Policy.Generation["seed"].String() != "not-a-number" {
		t.Fatal("clone silently discarded invalid numeric input")
	}
	*copy.Load.Set.Slots = 2
	copy.Load.Clear[0] = "gpu"
	*copy.Policy.Generation["seed"] = "1"
	copy.Policy.Roles["weak"] = false
	*copy.Policy.Served = true
	if *c.Load.Set.Slots != 1 || c.Load.Clear[0] != "threads" || c.Policy.Generation["seed"].String() != "not-a-number" || !c.Policy.Roles["weak"] || *c.Policy.Served {
		t.Fatal("model change clone aliases selected fields")
	}
	if err := c.Validate(); err == nil {
		t.Fatal("invalid numeric input accepted by validation")
	}
}

func TestInspectCloneDetachesFactsDescriptorsAndEstimates(t *testing.T) {
	r := InspectResult{Facts: &Facts{ContextTokens: testPointer(int32(8192)), Dimensions: map[string]string{"embedding": "4096"}, Unknowns: []string{"companion"}}, Options: []OptionDescriptor{{Field: "gpu", Values: []string{"cpu"}, Minimum: testPointer(json.Number("0")), Default: json.RawMessage(`"cpu"`)}}, Estimate: Estimate{Assumptions: []string{"uncalibrated"}, Components: []EstimateComponent{{Name: "weights"}}}}
	r.ConfigurationPools = []string{"host"}
	detached := r.Clone()
	detached.ConfigurationPools[0] = "shared"
	if r.ConfigurationPools[0] != "host" {
		t.Fatal("clone aliases pools")
	}
	before := mustJSON(t, r)
	copy := r.Clone()
	*copy.Facts.ContextTokens = 4096
	copy.Facts.Dimensions["embedding"] = "2048"
	copy.Facts.Unknowns[0] = "changed"
	copy.Options[0].Values[0] = "layers"
	*copy.Options[0].Minimum = "1"
	copy.Options[0].Default[1] = 'x'
	copy.Estimate.Assumptions[0] = "changed"
	copy.Estimate.Components[0].Name = "changed"
	if !reflect.DeepEqual(before, mustJSON(t, r)) {
		t.Fatal("inspect clone aliases derived fields")
	}
}

func TestInspectResultPreservesBytePrecisionAndUnknownImpact(t *testing.T) {
	input := InspectResult{Engine: "llamacpp", ModelID: "m", Status: "resolved", Artifacts: []ArtifactSummary{{Role: "model", Identity: "opaque-sampled-binding", Size: "9007199254740993", Assurance: "sampled-local-binding"}}, Estimate: Estimate{Status: "partial", HostBytes: "18446744073709551615", Unknowns: []string{"device mapping"}, Domains: []MemoryDomain{{ID: "uma", Kind: "shared", CapacityBytes: "9007199254740993"}}, Components: []EstimateComponent{{Name: "weights", DomainID: "uma", Status: "estimated", Bytes: "9007199254740993", LowBytes: "9007199254740993", HighBytes: "9007199254740993", Basis: "derived"}}}}
	got, err := DecodeInspectResult(mustJSON(t, input))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, input) {
		t.Fatal("inspect byte precision changed")
	}
	impact := Impact{RestartScope: "unknown", Unknowns: []string{"owner decision pending"}}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(mustJSON(t, impact), &wire); err != nil {
		t.Fatal(err)
	}
	if wire["drainTimeoutSeconds"] != nil || wire["mayInterruptActive"] != nil || wire["staysOff"] != nil {
		t.Fatal("unresolved impact manufactured owner defaults")
	}
}

func TestInspectArtifactSummariesAllowOrderedCompanions(t *testing.T) {
	artifact := func(role string) ArtifactSummary {
		return ArtifactSummary{Role: role, Identity: "opaque-identity", Size: "1", Assurance: "sampled-local-binding"}
	}
	for name, artifacts := range map[string][]ArtifactSummary{
		"model draft companion": {artifact("model"), artifact("draft"), artifact("companion")},
		"model companion":       {artifact("model"), artifact("companion")},
	} {
		t.Run(name, func(t *testing.T) {
			r := InspectResult{Engine: "llamacpp", ModelID: "m", Status: "resolved", Artifacts: artifacts, Estimate: Estimate{Status: "unknown"}}
			if _, err := DecodeInspectResult(mustJSON(t, r)); err != nil {
				t.Fatalf("rejected ordered read-only summaries: %v", err)
			}
		})
	}
	for name, artifacts := range map[string][]ArtifactSummary{
		"model after draft":      {artifact("model"), artifact("draft"), artifact("model")},
		"draft after companion":  {artifact("model"), artifact("companion"), artifact("draft")},
		"companion before model": {artifact("companion"), artifact("model")},
		"missing model":          {artifact("draft")},
		"unknown role":           {artifact("model"), artifact("sidecar")},
	} {
		t.Run(name, func(t *testing.T) {
			r := InspectResult{Engine: "llamacpp", ModelID: "m", Status: "resolved", Artifacts: artifacts, Estimate: Estimate{Status: "unknown"}}
			if _, err := DecodeInspectResult(mustJSON(t, r)); err == nil {
				t.Fatal("accepted invalid read-only artifact role order")
			}
		})
	}
}

func TestResolvedLoadRejectsCompanionArtifact(t *testing.T) {
	load := testResolvedLoad()
	load.Resolved.Artifacts = append(load.Resolved.Artifacts, ArtifactBinding{
		Role: "companion", Path: "/models/projector.gguf", ResolvedTarget: "/blobs/projector",
		Size: "1", MtimeNS: "0", SampleSHA256: strings.Repeat("f", 64),
	})
	if _, err := CanonicalLoads([]ModelLoad{load}); err == nil {
		t.Fatal("accepted companion artifact in resolved serving load")
	}
}

func TestEstimateRetainsMemoryPlanWithSelectedOverride(t *testing.T) {
	for _, kind := range []string{"engine-report", "user-stated", "measured", "engine-report"} {
		e := memoryPlanEstimateFixture()
		e.EstimateSource.Kind = kind
		if kind != "engine-report" {
			e.EstimateSource.Provider, e.EstimateSource.Revision = "selected-figure", "recorded-at"
		}
		if err := e.Validate(); err != nil {
			t.Fatalf("source %s: %v", kind, err)
		}
	}
}

func TestEstimateRejectsInvalidRanges(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*EstimateComponent)
	}{
		{"missing low", func(c *EstimateComponent) { c.LowBytes = "" }},
		{"missing high", func(c *EstimateComponent) { c.HighBytes = "" }},
		{"low above expected", func(c *EstimateComponent) { c.LowBytes = "101" }},
		{"high below expected", func(c *EstimateComponent) { c.HighBytes = "99" }},
		{"noncanonical", func(c *EstimateComponent) { c.HighBytes = "0100" }},
		{"overflow", func(c *EstimateComponent) { c.HighBytes = "18446744073709551616" }},
		{"invalid basis", func(c *EstimateComponent) { c.Basis = "guess" }},
		{"unknown with bytes", func(c *EstimateComponent) { c.Status = "unknown" }},
		{"partial without residual", func(c *EstimateComponent) { c.Status = "partial"; c.Reason = "" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := completeMemoryArithmeticFixture()
			tc.edit(&e.Components[0])
			if e.Validate() == nil {
				t.Fatal("accepted invalid range")
			}
		})
	}
}

func TestInspectNativeBuildReadback(t *testing.T) {
	r := InspectResult{Engine: "llamacpp", ModelID: "m", Status: "unknown", Estimate: Estimate{Status: "unknown"}, NativeBuild: "b12345-example", NativeBuildID: "sha256:" + strings.Repeat("a", 64)}
	got, err := DecodeInspectResult(mustJSON(t, r))
	if err != nil {
		t.Fatal(err)
	}
	if got.NativeBuild != "b12345-example" || got.NativeBuildID != "sha256:"+strings.Repeat("a", 64) {
		t.Fatalf("native build readback = %+v", got)
	}
	if _, err := DecodeInspectResult([]byte(`{"engine":"llamacpp","modelId":"m","status":"unknown","estimate":{"status":"unknown"},"nativeBuild":"example","nativeBuildId":"fingerprint","surprise":true}`)); err == nil {
		t.Fatal("native build support weakened unknown-field rejection")
	}
}

func TestInspectRejectsInvalidArtifactAndEstimateBytes(t *testing.T) {
	test := func(name string, edit func(*InspectResult)) {
		t.Run(name, func(t *testing.T) {
			r := InspectResult{Engine: "llamacpp", ModelID: "m", Status: "resolved", Artifacts: []ArtifactSummary{{Role: "model", Size: "1", Identity: "opaque", Assurance: "sampled"}}, Estimate: Estimate{Status: "unknown"}}
			edit(&r)
			if _, err := DecodeInspectResult(mustJSON(t, r)); err == nil {
				t.Fatal("accepted invalid inspected size or estimate")
			}
		})
	}
	test("artifact negative", func(r *InspectResult) { r.Artifacts[0].Size = "-1" })
	test("artifact noncanonical", func(r *InspectResult) { r.Artifacts[0].Size = "01" })
	test("artifact signed overflow", func(r *InspectResult) { r.Artifacts[0].Size = "9223372036854775808" })
	test("estimate uint overflow", func(r *InspectResult) { r.Estimate.HostBytes = "18446744073709551616" })
	test("estimate exponent", func(r *InspectResult) { r.Estimate.DeviceBytes = "1e3" })
	test("false fits state", func(r *InspectResult) { r.Estimate.Status = "fits" })
	test("missing resolved artifacts", func(r *InspectResult) { r.Artifacts = nil })
	test("oversized build claim", func(r *InspectResult) { r.NativeBuild = strings.Repeat("x", 257) })
	test("build identity controls", func(r *InspectResult) { r.NativeBuildID = "bad\nidentity" })
}
