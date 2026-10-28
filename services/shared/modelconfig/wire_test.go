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
		ModelID:   "exact/Model:Q4",
		Requested: RequestedLoad{Context: &Context{Mode: "fixed", Tokens: 8193}, Slots: testPointer(int32(2)), GPU: &GPU{Mode: "layers", Layers: testPointer(int32(12))}, KVOffload: testPointer(false)},
		Resolved: &ResolvedLoad{
			PerSlotContext: 8448, TotalContext: 16896, Slots: 2,
			GPU:           GPU{Mode: "layers", Layers: testPointer(int32(12))},
			Options:       RequestedLoad{KVOffload: testPointer(false)},
			Artifacts:     []ArtifactBinding{{Role: "model", Path: "/models/first.gguf", ResolvedTarget: "/blobs/first", Size: "9007199254740993", MtimeNS: "1780000000000000001", SampleSHA256: strings.Repeat("a", 64)}},
			NativeBuildID: "pinned-build", SourceInputDigest: strings.Repeat("b", 64), PresetDigest: strings.Repeat("c", 64),
		},
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
	for _, id := range []string{"", "model", "Model", " model ", "a:b/c", "日本語", strings.Repeat("x", 512), strings.Repeat("x", 513), "bad\nmodel", "bad\u0085model", string([]byte{0xff}), "weak", "medium-api", "strong-local", "Weak", "weak-local-api", "weak-api-local", "weak:latest"} {
		t.Run(id, func(t *testing.T) {
			wantValid := nodepolicy.ValidateModelIdentifier(id) == nil && !nodepolicy.IsReservedAlias(id)
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
	input := []ModelLoad{{ModelID: "z"}, m}
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
		Threads: testPointer(int32(3)), ThreadsBatch: testPointer(int32(4)), KVOffload: testPointer(false), OpOffload: testPointer(true), SleepIdleSeconds: testPointer(int32(-1)), CacheRAM: testPointer(int32(512)),
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
	*got[0].Requested.SleepIdleSeconds = 300
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
	input := InspectResult{Engine: "llamacpp", ModelID: "m", Status: "resolved", Artifacts: []ArtifactSummary{{Role: "model", Identity: "opaque-sampled-binding", Size: "9007199254740993", Assurance: "sampled-local-binding"}}, Estimate: Estimate{Status: "partial", HostBytes: "18446744073709551615", Unknowns: []string{"device mapping"}, Domains: []MemoryDomain{{ID: "uma", Kind: "shared", CapacityBytes: "9007199254740993"}}, Components: []EstimateComponent{{Name: "weights", DomainID: "uma", Status: "estimated", Bytes: "9007199254740993"}}}}
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
}
