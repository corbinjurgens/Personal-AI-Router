// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelconfig

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestTotalContextPaddingAndOverflow(t *testing.T) {
	per, total, err := TotalContext(8193, 2)
	if err != nil || per != 8448 || total != 16896 {
		t.Fatalf("got %d, %d, %v", per, total, err)
	}
	if _, _, err := TotalContext(2147483647, 2); err == nil {
		t.Fatal("overflow accepted")
	}
}

func TestResolvedUnifiedKVPreservesLargerSharedPool(t *testing.T) {
	load := testResolvedLoad()
	load.Requested = RequestedLoad{}
	load.Resolved.PerSlotContext = 4096
	load.Resolved.TotalContext = 16384
	load.Resolved.Slots = 2
	load.Resolved.KVUnified = testPointer(true)
	if err := load.Validate(); err != nil {
		t.Fatalf("valid shared pool refused: %v", err)
	}
	load.Resolved.TotalContext = 4096
	if err := load.Validate(); err == nil {
		t.Fatal("per-slot caps exceed unified total")
	}
	load.Resolved.TotalContext = 16384
	load.Resolved.KVUnified = testPointer(false)
	if err := load.Validate(); err == nil {
		t.Fatal("non-unified pool ignored exact slot arithmetic")
	}
}

func TestRequestedLoadValidation(t *testing.T) {
	zero, batch, ubatch := int32(0), int32(32), int32(64)
	for name, load := range map[string]RequestedLoad{
		"zero slots":           {Slots: &zero},
		"invalid auto range":   {Context: &Context{Mode: "auto", Floor: 8192, Cap: 4096}},
		"ubatch exceeds batch": {BatchSize: &batch, UBatchSize: &ubatch},
		"missing layers":       {GPU: &GPU{Mode: "layers"}},
	} {
		t.Run(name, func(t *testing.T) {
			if err := load.Validate(); err == nil {
				t.Fatal("accepted invalid load")
			}
		})
	}
}

func TestPolicyPatchGenerationUsesExistingRanges(t *testing.T) {
	test := func(name, key, number string, valid bool) {
		t.Run(name, func(t *testing.T) {
			input := `{"modelId":"m","expectedPolicyRevision":"0","policy":{"generation":{"` + key + `":` + number + `}}}`
			_, err := DecodeModelChange([]byte(input))
			if (err == nil) != valid {
				t.Fatalf("decode err=%v, want valid=%v", err, valid)
			}
		})
	}
	test("zero temperature", "temperature", "0", true)
	test("negative temperature", "temperature", "-0.1", false)
	test("top p upper bound", "top_p", "1", true)
	test("top p precise overflow", "top_p", "1.000000000000000001", false)
	test("min p negative", "min_p", "-0.0001", false)
	test("min p clear", "min_p", "null", true)
	test("zero top k", "top_k", "0", true)
	test("fractional top k", "top_k", "1.5", false)
	test("top k integer exponent", "top_k", "1e2", true)
	test("top k safe integer limit", "top_k", "9007199254740991", true)
	test("top k unsafe integer", "top_k", "9007199254740992", false)
	test("seed negative limit", "seed", "-9007199254740991", true)
	test("seed precise noninteger", "seed", "1.000000000000000001", false)
	test("seed unsafe negative", "seed", "-9007199254740992", false)
	test("zero repeat penalty", "repeat_penalty", "0", false)
	test("positive repeat penalty", "repeat_penalty", "0.1", true)
	test("frequency lower bound", "frequency_penalty", "-2", true)
	test("frequency below bound", "frequency_penalty", "-2.00000000000000001", false)
	test("presence upper bound", "presence_penalty", "2", true)
	test("presence above bound", "presence_penalty", "2.00000000000000001", false)
	test("unsupported sampler", "typical_p", "0.5", false)
	test("huge exponent", "temperature", "1e999999999", false)
	test("huge negative exponent", "temperature", "1e-99999999", false)
	test("not a numeric value", "temperature", `"0.5"`, false)
}

func TestPolicyPatchRolesMatchClosedNodePolicyTiers(t *testing.T) {
	for name, roles := range map[string]string{
		"valid explicit assignment and removal": `{"weak":true,"medium":false,"strong":true}`,
		"unknown tier":                          `{"fast":true}`,
		"case folded tier":                      `{"Weak":true}`,
		"unrelated model assignment":            `{"weak":"other/model"}`,
	} {
		t.Run(name, func(t *testing.T) {
			_, err := DecodeModelChange([]byte(`{"modelId":"m","expectedPolicyRevision":"0","policy":{"roles":` + roles + `}}`))
			valid := name == "valid explicit assignment and removal"
			if (err == nil) != valid {
				t.Fatalf("decode err=%v, want valid=%v", err, valid)
			}
		})
	}
}

func TestResolvedLoadValidationAtStoredDecodeBoundary(t *testing.T) {
	test := func(name string, edit func(*ModelLoad)) {
		t.Run(name, func(t *testing.T) {
			m := testResolvedLoad()
			edit(&m)
			if _, err := DecodeLoads(mustJSON(t, []ModelLoad{m})); err == nil {
				t.Fatal("accepted invalid resolved authority")
			}
		})
	}
	test("unpadded context", func(m *ModelLoad) { m.Resolved.PerSlotContext = 8193 })
	test("wrong total", func(m *ModelLoad) { m.Resolved.TotalContext++ })
	test("overflow total", func(m *ModelLoad) { m.Resolved.Slots = math.MaxInt32 })
	test("Auto placement unresolved", func(m *ModelLoad) { m.Resolved.GPU = GPU{Mode: "auto"} })
	test("native default unresolved", func(m *ModelLoad) { m.Resolved.GPU = GPU{Mode: "native-default"} })
	test("negative layers", func(m *ModelLoad) { m.Resolved.GPU.Layers = testPointer(int32(-1)) })
	test("missing layers", func(m *ModelLoad) { m.Resolved.GPU.Layers = nil })
	test("missing artifacts", func(m *ModelLoad) { m.Resolved.Artifacts = nil })
	test("zero artifact size", func(m *ModelLoad) { m.Resolved.Artifacts[0].Size = "0" })
	test("artifact size noncanonical", func(m *ModelLoad) { m.Resolved.Artifacts[0].Size = "01" })
	test("artifact size overflow", func(m *ModelLoad) { m.Resolved.Artifacts[0].Size = "9223372036854775808" })
	test("artifact mtime noncanonical", func(m *ModelLoad) { m.Resolved.Artifacts[0].MtimeNS = "-0" })
	test("missing sampled identity", func(m *ModelLoad) { m.Resolved.Artifacts[0].SampleSHA256 = "" })
	test("missing target path", func(m *ModelLoad) { m.Resolved.Artifacts[0].ResolvedTarget = "" })
	test("wrong role", func(m *ModelLoad) { m.Resolved.Artifacts[0].Role = "main" })
	test("duplicate artifacts", func(m *ModelLoad) { m.Resolved.Artifacts = append(m.Resolved.Artifacts, m.Resolved.Artifacts[0]) })
	test("missing build", func(m *ModelLoad) { m.Resolved.NativeBuildID = "" })
	test("invalid source digest", func(m *ModelLoad) { m.Resolved.SourceInputDigest = "not-a-digest" })
	test("invalid preset digest", func(m *ModelLoad) { m.Resolved.PresetDigest = strings.Repeat("A", 64) })
	test("changed explicit slots", func(m *ModelLoad) { m.Requested.Slots = testPointer(int32(1)) })
	test("changed explicit context", func(m *ModelLoad) { m.Requested.Context.Tokens = 4096 })
	test("changed explicit placement", func(m *ModelLoad) { m.Requested.GPU.Layers = testPointer(int32(3)) })
	test("changed explicit option", func(m *ModelLoad) { m.Resolved.Options.KVOffload = testPointer(true) })
	test("unresolved option Auto", func(m *ModelLoad) { m.Resolved.Options.Context = &Context{Mode: "auto", Floor: 4096, Cap: 8192} })
	test("Auto missing reviewed identity", func(m *ModelLoad) { m.Requested.Context = &Context{Mode: "auto", Floor: 4096, Cap: 16384} })
	test("half review provenance", func(m *ModelLoad) { m.Resolved.ResolverRevision = "resolver-source-revision" })
}

func TestResolvedAutoRequiresOnlyExplicitRangeAndReviewedIdentity(t *testing.T) {
	m := testResolvedLoad()
	m.Requested.Context = &Context{Mode: "auto", Floor: 8192, Cap: 16384}
	m.Resolved.ResolverRevision = "source-revision"
	m.Resolved.ReviewedInputID = "reviewed-memory-and-artifact-input"
	got, err := DecodeLoads(mustJSON(t, []ModelLoad{m}))
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Requested.Context.Floor != 8192 || got[0].Resolved.PerSlotContext != 8448 {
		t.Fatal("schema selected different Auto defaults")
	}
	m.Requested.Context.Cap = 8192
	if _, err := DecodeLoads(mustJSON(t, []ModelLoad{m})); err == nil {
		t.Fatal("accepted resolution outside explicit Auto cap")
	}
}

func TestModelLoadLimitsAndDuplicateExactIDs(t *testing.T) {
	if _, err := CanonicalLoads(make([]ModelLoad, MaxModels+1)); err == nil {
		t.Fatal("accepted unbounded model list")
	}
	if _, err := DecodeLoads([]byte(`[{"modelId":"same","requested":{}},{"modelId":"same","requested":{}}]`)); err == nil {
		t.Fatal("accepted duplicate exact IDs")
	}
	if _, err := DecodeLoads([]byte(`[{"modelId":"Same","requested":{}},{"modelId":"same","requested":{}}]`)); err != nil {
		t.Fatalf("case-distinct IDs must remain distinct: %v", err)
	}
}

func TestDescriptorCarriesFractionalBoundsAndDefaultProvenance(t *testing.T) {
	o := OptionDescriptor{Field: "top_p", Label: "Top P", Kind: "number", Minimum: testPointer(json.Number("0.0")), Maximum: testPointer(json.Number("1.0")), Default: json.RawMessage(`0.95`), DefaultSource: "embedded recommendation", Supported: true, Portable: true}
	if err := o.Validate(); err != nil {
		t.Fatal(err)
	}
	copy := o.Clone()
	*copy.Minimum = "0.5"
	copy.Default[0] = '1'
	if o.Minimum.String() != "0.0" || string(o.Default) != "0.95" {
		t.Fatal("descriptor clone aliases defaults or bounds")
	}
	o.Supported = false
	if err := o.Validate(); err == nil {
		t.Fatal("unsupported descriptor lacks reason")
	}
	o.Reason = "No validated build mapping"
	o.DefaultSource = ""
	if err := o.Validate(); err == nil {
		t.Fatal("projected default lacks provenance")
	}
}
