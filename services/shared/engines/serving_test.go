// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engines

import (
	"encoding/json"
	"strings"
	"testing"
)

func servingFixture() *ServingConfig {
	ram, vram := int64(4096), int64(0)
	return &ServingConfig{Version: 1, Models: []ServingModel{{ModelID: "exact/model:Q4", MaxOutputTokens: 256,
		Memory:                     ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram},
		ExpectedNativeSourceDigest: strings.Repeat("a", 64),
		Artifacts:                  []ServingArtifact{{Role: "model-shard", SHA256: strings.Repeat("b", 64), Provenance: ServingProvenance{Kind: "hf-lfs-sha256", Reference: "pinned metadata"}, Binding: &ServingBinding{SourcePath: "fixture-link", ResolvedTarget: "fixture-blob", SizeBytes: 123, MtimeNS: "1729000000000000001", SampleSHA256: strings.Repeat("c", 64)}}},
	}}}
}

func TestServingValidation(t *testing.T) {
	test := func(name, field string, mutate func(*ServingConfig)) {
		t.Run(name, func(t *testing.T) {
			c := servingFixture()
			mutate(c)
			if errors := c.Validate(); errors[field] == "" {
				t.Fatalf("missing error for %s", field)
			}
		})
	}
	test("unknown version", "serving.version", func(c *ServingConfig) { c.Version = 2 })
	test("empty models", "serving.models", func(c *ServingConfig) { c.Models = nil })
	test("duplicate exact model ID", "serving.models.1.modelId", func(c *ServingConfig) { c.Models = append(c.Models, c.Models[0]) })
	test("empty ID", "serving.models.0.modelId", func(c *ServingConfig) { c.Models[0].ModelID = " " })
	test("zero output", "serving.models.0.maxOutputTokens", func(c *ServingConfig) { c.Models[0].MaxOutputTokens = 0 })
	test("negative output", "serving.models.0.maxOutputTokens", func(c *ServingConfig) { c.Models[0].MaxOutputTokens = -1 })
	test("unsafe output", "serving.models.0.maxOutputTokens", func(c *ServingConfig) { c.Models[0].MaxOutputTokens = 1 << 53 })
	test("missing host RAM", "serving.models.0.memory.hostRamBytes", func(c *ServingConfig) { c.Models[0].Memory.HostRAMBytes = nil })
	test("zero host RAM", "serving.models.0.memory.hostRamBytes", func(c *ServingConfig) { *c.Models[0].Memory.HostRAMBytes = 0 })
	test("missing VRAM", "serving.models.0.memory.vramBytes", func(c *ServingConfig) { c.Models[0].Memory.VRAMBytes = nil })
	test("negative VRAM", "serving.models.0.memory.vramBytes", func(c *ServingConfig) { *c.Models[0].Memory.VRAMBytes = -1 })
	test("unsafe RAM", "serving.models.0.memory.hostRamBytes", func(c *ServingConfig) { *c.Models[0].Memory.HostRAMBytes = 1 << 53 })
	test("malformed source digest", "serving.models.0.expectedNativeSourceDigest", func(c *ServingConfig) { c.Models[0].ExpectedNativeSourceDigest = "bad" })
	test("missing artifacts", "serving.models.0.artifacts", func(c *ServingConfig) { c.Models[0].Artifacts = nil })
	test("malformed artifact digest", "serving.models.0.artifacts.0.sha256", func(c *ServingConfig) { c.Models[0].Artifacts[0].SHA256 = strings.Repeat("g", 64) })
	test("unknown provenance", "serving.models.0.artifacts.0.provenance.kind", func(c *ServingConfig) { c.Models[0].Artifacts[0].Provenance.Kind = "filename" })
	test("missing binding source", "serving.models.0.artifacts.0.binding.sourcePath", func(c *ServingConfig) { c.Models[0].Artifacts[0].Binding.SourcePath = "" })
	test("missing resolved target", "serving.models.0.artifacts.0.binding.resolvedTarget", func(c *ServingConfig) { c.Models[0].Artifacts[0].Binding.ResolvedTarget = "" })
	test("invalid size", "serving.models.0.artifacts.0.binding.sizeBytes", func(c *ServingConfig) { c.Models[0].Artifacts[0].Binding.SizeBytes = -1 })
	test("overflow mtime", "serving.models.0.artifacts.0.binding.mtimeNs", func(c *ServingConfig) { c.Models[0].Artifacts[0].Binding.MtimeNS = "9223372036854775808" })
	test("invalid sample digest", "serving.models.0.artifacts.0.binding.sampleSha256", func(c *ServingConfig) { c.Models[0].Artifacts[0].Binding.SampleSHA256 = "" })
}

func TestServingRoundTripAndClone(t *testing.T) {
	c := servingFixture()
	if errors := c.Validate(); len(errors) != 0 {
		t.Fatalf("valid fixture errors: %v", errors)
	}
	data, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ServingConfig
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if !c.Equal(&decoded) {
		t.Fatal("round trip changed declaration")
	}
	copy := c.Clone()
	*copy.Models[0].Memory.VRAMBytes = 1
	copy.Models[0].Artifacts[0].Binding.MtimeNS = "1"
	copy.Models[0].Artifacts[0].Provenance.Reference = "changed"
	if c.Equal(copy) || *c.Models[0].Memory.VRAMBytes != 0 || c.Models[0].Artifacts[0].Binding.MtimeNS != "1729000000000000001" || c.Models[0].Artifacts[0].Provenance.Reference != "pinned metadata" {
		t.Fatal("clone aliases nested data")
	}
	c.Models[0].Artifacts[0].Binding = nil
	c.Models[0].ExpectedNativeSourceDigest = ""
	if len(c.Validate()) != 0 {
		t.Fatal("structural declaration requires adapter-produced binding/source digest")
	}
	var absent *ServingConfig
	if !absent.Equal(nil) || absent.Clone() != nil || len(absent.Validate()) != 0 {
		t.Fatal("absent serving changed legacy behavior")
	}
}

func TestServingRejectsNonIntegerJSON(t *testing.T) {
	for _, value := range []string{"1.5", "9223372036854775808", "1e100"} {
		t.Run(value, func(t *testing.T) {
			var c ServingConfig
			if err := json.Unmarshal([]byte(`{"version":1,"models":[{"maxOutputTokens":`+value+`}]}`), &c); err == nil {
				t.Fatal("invalid integer decoded")
			}
		})
	}
}

func TestServingReadbackClone(t *testing.T) {
	m := servingFixture().Models[0].Memory
	r := &ServingReadback{Status: "configured", Configured: []ServingEnvelope{{Memory: m, Placement: []string{"opaque"}}}, LaunchValidated: []ServingEnvelope{{Memory: m}}, Observed: []ServingObservation{{Slots: 1}}}
	copy := r.Clone()
	*copy.Configured[0].Memory.HostRAMBytes = 1
	copy.Configured[0].Placement[0] = "changed"
	*copy.LaunchValidated[0].Memory.VRAMBytes = 2
	copy.Observed[0].Slots = 3
	if *r.Configured[0].Memory.HostRAMBytes != 4096 || r.Configured[0].Placement[0] != "opaque" || *r.LaunchValidated[0].Memory.VRAMBytes != 0 || r.Observed[0].Slots != 1 {
		t.Fatal("readback clone aliases nested values")
	}
}
