// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/gguffacts"
	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
)

func estimateFixture(arch string) modelEstimateInput {
	numeric := map[string]gguffacts.Numeric{}
	for field, value := range map[string]uint64{"block_count": 2, "embedding_length": 128, "feed_forward_length": 256, "attention.head_count": 4, "attention.head_count_kv": 2, "vocab_size": 256} {
		numeric[arch+"."+field] = gguffacts.Numeric{Type: 10, Bits: value}
	}
	tensor := func(name string, dims ...uint64) gguffacts.Tensor {
		bytes := uint64(2)
		for _, dim := range dims {
			bytes *= dim
		}
		return gguffacts.Tensor{Name: name, Type: 1, Dimensions: dims, Bytes: bytes}
	}
	facts := gguffacts.Facts{Shards: []gguffacts.Shard{{Role: gguffacts.RoleMain, Architecture: arch, Numeric: numeric, Tensors: []gguffacts.Tensor{
		tensor("token_embd.weight", 128, 256), tensor("output_norm.weight", 128),
		tensor("blk.0.attn_q.weight", 128, 128), tensor("blk.0.ffn_gate.weight", 128, 256),
		tensor("blk.1.attn_q.weight", 128, 128), tensor("blk.1.ffn_gate.weight", 128, 256),
		tensor("output.weight", 128, 256),
	}}}}
	f16, fa := "f16", "off"
	batch, ubatch, cache := int32(64), int32(32), int32(1)
	unified, offload, op := false, false, false
	allocation := &modelconfig.ResolvedLoad{PerSlotContext: 1024, TotalContext: 1024, Slots: 1, KVUnified: &unified, GPU: modelconfig.GPU{Mode: "cpu"}, Options: modelconfig.RequestedLoad{CacheTypeK: &f16, CacheTypeV: &f16, FlashAttention: &fa, BatchSize: &batch, UBatchSize: &ubatch, CacheRAM: &cache, KVOffload: &offload, OpOffload: &op}}
	swaFull := false
	return modelEstimateInput{Facts: facts, Bindings: []modelconfig.ArtifactSummary{{Role: "model", Identity: "synthetic-main", Size: "327936"}}, Allocation: allocation, BuildID: "synthetic-native-build", Support: modelEstimatorSupport, SWAFull: &swaFull}
}

func estimateFixtureResult(t *testing.T, in modelEstimateInput) modelconfig.Estimate {
	t.Helper()
	var state modelEstimateState
	result := state.estimate(context.Background(), in)
	if err := result.Validate(); err != nil {
		t.Fatalf("invalid estimate: %v", err)
	}
	return result
}

func estimateTestComponent(t *testing.T, result modelconfig.Estimate, name string) modelconfig.EstimateComponent {
	t.Helper()
	for _, component := range result.Components {
		if component.Name == name {
			return component
		}
	}
	t.Fatalf("missing %s component: %+v", name, result)
	return modelconfig.EstimateComponent{}
}

func assertEstimateBytes(t *testing.T, result modelconfig.Estimate, name, status, bytes string) {
	t.Helper()
	c := estimateTestComponent(t, result, name)
	if c.Status != status || c.Bytes != bytes {
		t.Fatalf("%s = %+v, want %s %s", name, c, status, bytes)
	}
}

func TestModelSettingsEstimateDenseAndGQA(t *testing.T) {
	for _, tc := range []struct {
		name          string
		heads         uint64
		want, weights string
	}{{"dense MHA", 4, "1048576", "787712"}, {"GQA", 2, "524288", "722176"}} {
		t.Run(tc.name, func(t *testing.T) {
			in := estimateFixture("llama")
			in.Facts.Shards[0].Numeric["llama.attention.head_count_kv"] = gguffacts.Numeric{Type: 10, Bits: tc.heads}
			for layer := 0; layer < 2; layer++ {
				prefix := "blk." + strconv.Itoa(layer) + "."
				for _, spec := range []struct {
					name string
					dims []uint64
				}{
					{"attn_norm.weight", []uint64{128}}, {"attn_k.weight", []uint64{128, 32 * tc.heads}}, {"attn_v.weight", []uint64{128, 32 * tc.heads}}, {"attn_output.weight", []uint64{128, 128}}, {"ffn_norm.weight", []uint64{128}}, {"ffn_up.weight", []uint64{128, 256}}, {"ffn_down.weight", []uint64{256, 128}},
				} {
					bytes := uint64(2)
					for _, dim := range spec.dims {
						bytes *= dim
					}
					in.Facts.Shards[0].Tensors = append(in.Facts.Shards[0].Tensors, gguffacts.Tensor{Name: prefix + spec.name, Type: 1, Dimensions: spec.dims, Bytes: bytes})
				}
			}
			in.Bindings[0].Size = tc.weights
			out := estimateFixtureResult(t, in)
			assertEstimateBytes(t, out, "weights", "estimated", tc.weights)
			assertEstimateBytes(t, out, "attention-cache", "estimated", tc.want)
			assertEstimateBytes(t, out, "recurrent-state", "excluded", "")
			assertEstimateBytes(t, out, "companions", "excluded", "")
			assertEstimateBytes(t, out, "prompt-cache", "estimated", "1048576")
			compute := estimateTestComponent(t, out, "compute-workspace")
			bytes, err := strconv.ParseUint(compute.Bytes, 10, 64)
			if err != nil || compute.Status != "partial" || bytes == 0 {
				t.Fatalf("dense library graph cost unavailable: %+v, %v", compute, err)
			}
			if compute.Basis != "heuristic" || compute.LowBytes != strconv.FormatUint(bytes*3/4, 10) || compute.HighBytes != strconv.FormatUint((bytes*3+1)/2, 10) {
				t.Fatalf("partial CPU portion lacks heuristic band: %+v", compute)
			}
			if out.Status != "partial" || len(out.Unknowns) == 0 || out.BuildID != in.BuildID {
				t.Fatalf("lost partial/build disclosure: %+v", out)
			}
		})
	}
}

func TestModelSettingsEstimateMixedCacheBlocks(t *testing.T) {
	for _, tc := range []struct{ name, k, v, fa, want, status string }{
		{"q8 K with f16 V", "q8_0", "f16", "off", "401408", "estimated"},
		{"f16 K with q4 V", "f16", "q4_0", "on", "335872", "estimated"},
		{"mixed quantized axes", "q8_0", "q4_0", "on", "212992", "estimated"},
		{"bf16 axes", "bf16", "bf16", "on", "524288", "estimated"},
		{"quantized V without FA", "f16", "q4_0", "off", "", "unknown"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := estimateFixture("qwen3moe")
			in.Allocation.Options.CacheTypeK = &tc.k
			in.Allocation.Options.CacheTypeV = &tc.v
			in.Allocation.Options.FlashAttention = &tc.fa
			assertEstimateBytes(t, estimateFixtureResult(t, in), "attention-cache", tc.status, tc.want)
		})
	}
	t.Run("row width must fit each block", func(t *testing.T) {
		in := estimateFixture("qwen3moe")
		quant := "q8_0"
		in.Allocation.Options.CacheTypeK = &quant
		in.Facts.Shards[0].Numeric["qwen3moe.attention.key_length"] = gguffacts.Numeric{Type: 10, Bits: 17}
		assertEstimateBytes(t, estimateFixtureResult(t, in), "attention-cache", "unknown", "")
	})
}

func TestModelSettingsEstimateContextAndSlots(t *testing.T) {
	in := estimateFixture("qwen2")
	in.Allocation.PerSlotContext = 2048
	in.Allocation.TotalContext = 6144
	in.Allocation.Slots = 3
	out := estimateFixtureResult(t, in)
	assertEstimateBytes(t, out, "attention-cache", "estimated", "3145728")
	if c := estimateTestComponent(t, out, "attention-cache"); c.DomainID != "host" {
		t.Fatalf("CPU cache domain = %+v", c)
	}
}

func TestModelSettingsEstimateLFM2RecurrentState(t *testing.T) {
	in := estimateFixture("lfm2")
	in.Facts.Shards[0].Numeric["lfm2.attention.head_count_kv"] = gguffacts.Numeric{Type: 10, Array: []uint64{0, 2}}
	in.Facts.Shards[0].Numeric["lfm2.shortconv.l_cache"] = gguffacts.Numeric{Type: 10, Bits: 3}
	in.Allocation.Slots = 2
	in.Allocation.TotalContext = 2048
	out := estimateFixtureResult(t, in)
	assertEstimateBytes(t, out, "attention-cache", "estimated", "524288")
	assertEstimateBytes(t, out, "recurrent-state", "estimated", "2048")
	in.Allocation.PerSlotContext = 2048
	in.Allocation.TotalContext = 4096
	larger := estimateFixtureResult(t, in)
	assertEstimateBytes(t, larger, "attention-cache", "estimated", "1048576")
	assertEstimateBytes(t, larger, "recurrent-state", "estimated", "2048")
	in.Allocation.Options.CacheTypeK = nil
	independent := estimateFixtureResult(t, in)
	assertEstimateBytes(t, independent, "attention-cache", "unknown", "")
	assertEstimateBytes(t, independent, "recurrent-state", "estimated", "2048")
	f16 := "f16"
	in.Allocation.Options.CacheTypeK = &f16
	delete(in.Facts.Shards[0].Numeric, "lfm2.shortconv.l_cache")
	partial := estimateFixtureResult(t, in)
	assertEstimateBytes(t, partial, "attention-cache", "estimated", "1048576")
	assertEstimateBytes(t, partial, "recurrent-state", "unknown", "")
}

func TestModelSettingsEstimateArchitectureMatrix(t *testing.T) {
	for _, arch := range []string{"qwen2moe", "qwen3moe", "lfm2moe", "qwen35", "qwen35moe", "qwen3next", "deepseek2", "lfm", "unmapped-model"} {
		t.Run(arch, func(t *testing.T) {
			in := estimateFixture(arch)
			if strings.HasPrefix(arch, "qwen3") && (arch == "qwen35" || arch == "qwen35moe" || arch == "qwen3next") {
				in.Facts.Shards[0].Numeric[arch+".full_attention_interval"] = gguffacts.Numeric{Type: 10, Bits: 2}
			}
			if arch == "lfm2moe" {
				in.Facts.Shards[0].Numeric[arch+".attention.head_count_kv"] = gguffacts.Numeric{Type: 10, Array: []uint64{0, 2}}
				in.Facts.Shards[0].Numeric[arch+".shortconv.l_cache"] = gguffacts.Numeric{Type: 10, Bits: 3}
			}
			in.Facts.Shards[0].Numeric[arch+".expert_count"] = gguffacts.Numeric{Type: 10, Bits: 8}
			in.Facts.Shards[0].Numeric[arch+".expert_used_count"] = gguffacts.Numeric{Type: 10, Bits: 2}
			out := estimateFixtureResult(t, in)
			assertEstimateBytes(t, out, "weights", "estimated", "327936")
			assertEstimateBytes(t, out, "compute-workspace", "unknown", "")
			switch arch {
			case "qwen2moe", "qwen3moe":
				assertEstimateBytes(t, out, "attention-cache", "estimated", "524288")
			case "qwen35", "qwen35moe", "qwen3next":
				assertEstimateBytes(t, out, "attention-cache", "estimated", "262144")
				assertEstimateBytes(t, out, "recurrent-state", "unknown", "")
			case "lfm2moe":
				assertEstimateBytes(t, out, "attention-cache", "estimated", "262144")
				assertEstimateBytes(t, out, "recurrent-state", "estimated", "1024")
			default:
				assertEstimateBytes(t, out, "attention-cache", "unknown", "")
			}
		})
	}
	// An MLA marker on a nominally dense model must not use dense math either.
	in := estimateFixture("llama")
	in.Facts.Shards[0].Numeric["llama.attention.kv_lora_rank"] = gguffacts.Numeric{Type: 10, Bits: 32}
	out := estimateFixtureResult(t, in)
	assertEstimateBytes(t, out, "attention-cache", "unknown", "")
	assertEstimateBytes(t, out, "weights", "estimated", "327936")
}

func TestModelSettingsEstimateSWA(t *testing.T) {
	in := estimateFixture("gemma2")
	in.Allocation.PerSlotContext = 4096
	in.Allocation.TotalContext = 4096
	in.Facts.Shards[0].Numeric["gemma2.attention.sliding_window"] = gguffacts.Numeric{Type: 10, Bits: 256}
	in.Facts.Shards[0].Numeric["gemma2.attention.sliding_window_pattern"] = gguffacts.Numeric{Type: 10, Bits: 2}
	// One windowed layer has 512 padded cells, the global layer has 4096.
	out := estimateFixtureResult(t, in)
	assertEstimateBytes(t, out, "attention-cache", "estimated", "1179648")
	assertEstimateBytes(t, out, "compute-workspace", "unknown", "")
	delete(in.Facts.Shards[0].Numeric, "gemma2.attention.sliding_window_pattern")
	assertEstimateBytes(t, estimateFixtureResult(t, in), "attention-cache", "unknown", "")
}

func TestModelSettingsEstimateIncompleteSWAIsUnknown(t *testing.T) {
	for _, tc := range []struct {
		name  string
		field string
		value gguffacts.Numeric
	}{
		{name: "missing Gemma2 window"},
		{name: "zero key width", field: "attention.key_length_swa", value: gguffacts.Numeric{Type: 10}},
		{name: "invalid key width", field: "attention.key_length_swa", value: gguffacts.Numeric{Type: 7, Bits: 1}},
		{name: "invalid value width", field: "attention.value_length_swa", value: gguffacts.Numeric{Type: 10, Array: []uint64{64}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := estimateFixture("gemma2")
			if tc.field != "" {
				in.Facts.Shards[0].Numeric["gemma2.attention.sliding_window"] = gguffacts.Numeric{Type: 10, Bits: 256}
				in.Facts.Shards[0].Numeric["gemma2.attention.sliding_window_pattern"] = gguffacts.Numeric{Type: 10, Bits: 2}
				in.Facts.Shards[0].Numeric["gemma2."+tc.field] = tc.value
			}
			out := estimateFixtureResult(t, in)
			assertEstimateBytes(t, out, "attention-cache", "unknown", "")
			assertEstimateBytes(t, out, "weights", "estimated", "327936")
		})
	}
}

func TestModelSettingsEstimateTensorPlacement(t *testing.T) {
	for _, tc := range []struct {
		name                 string
		layers               int32
		wantHost, wantDevice string
	}{
		{"zero layers", 0, "327936", "0"},
		{"output only", 1, "262144", "65792"},
		{"partial with final block", 2, "163840", "164096"},
		{"all blocks and output", 3, "65536", "262400"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := estimateFixture("qwen3moe")
			in.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: &tc.layers}
			// Isolate tensor placement from known host/cache/workspace costs.
			in.Allocation.Options.CacheRAM = nil
			in.Allocation.Options.CacheTypeK = nil
			out := estimateFixtureResult(t, in)
			if out.HostBytes != tc.wantHost || out.DeviceBytes != tc.wantDevice {
				t.Fatalf("tensor projection host=%s device=%s, want %s/%s", out.HostBytes, out.DeviceBytes, tc.wantHost, tc.wantDevice)
			}
			for _, d := range out.Domains {
				if d.DeviceID != "" || d.CapacityBytes != "" || d.FreeBytes != "" {
					t.Fatalf("invented native mapping/telemetry: %+v", d)
				}
			}
		})
	}
}

// Conditional projections from b11146, not device measurements:
// https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/src/llama-model-loader.cpp#L1363-L1385
func TestModelSettingsEstimateTiedOutputProjection(t *testing.T) {
	for _, tc := range []struct {
		name, arch           string
		untied, cpu          bool
		wantHost, wantDevice string
	}{
		{"Qwen tied different buffer", "qwen2", false, false, "272269312", "272269312"},
		{"Qwen tied CPU same buffer", "qwen2", false, true, "272269312", "0"},
		{"Qwen untied", "qwen2", true, false, "272269312", "272269312"},
		{"Gemma2 tied different buffer", "gemma2", false, false, "272269312", "272269312"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := estimateFixture(tc.arch)
			// F16 embedding: 2 * 896 * 151936 = 272269312 bytes.
			embed := gguffacts.Tensor{Name: "token_embd.weight", Type: 1, Dimensions: []uint64{896, 151936}, Bytes: 272269312}
			in.Facts.Shards[0].Tensors = []gguffacts.Tensor{embed}
			in.Bindings[0].Size = "272269312"
			if tc.untied {
				embed.Name = "output.weight"
				in.Facts.Shards[0].Tensors = append(in.Facts.Shards[0].Tensors, embed)
				in.Bindings[0].Size = "544538624"
			}
			in.Facts.Shards[0].Numeric[tc.arch+".block_count"] = gguffacts.Numeric{Type: 10, Bits: 24}
			in.Facts.Shards[0].Numeric[tc.arch+".embedding_length"] = gguffacts.Numeric{Type: 10, Bits: 896}
			in.Facts.Shards[0].Numeric[tc.arch+".vocab_size"] = gguffacts.Numeric{Type: 10, Bits: 151936}
			if !tc.cpu {
				layers := int32(25)
				in.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: &layers}
			}
			in.Allocation.Options.CacheRAM = nil
			in.Allocation.Options.CacheTypeK = nil
			out := estimateFixtureResult(t, in)
			if out.HostBytes != tc.wantHost || out.DeviceBytes != tc.wantDevice {
				t.Fatalf("host/device = %s/%s, want %s/%s", out.HostBytes, out.DeviceBytes, tc.wantHost, tc.wantDevice)
			}
		})
	}
}

// Normal non-unified iSWA, swa_full=false:
// https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/src/llama-kv-cache-iswa.cpp#L69-L80
func TestModelSettingsEstimateSWAPerStream(t *testing.T) {
	for _, fa := range []string{"off", "on"} {
		t.Run("Gemma2 "+fa, func(t *testing.T) {
			in := estimateFixture("gemma2")
			for key, value := range map[string]uint64{"block_count": 26, "attention.head_count": 8, "attention.head_count_kv": 4, "attention.key_length": 256, "attention.value_length": 256, "attention.sliding_window": 4096, "attention.sliding_window_pattern": 2} {
				in.Facts.Shards[0].Numeric["gemma2."+key] = gguffacts.Numeric{Type: 10, Bits: value}
			}
			in.Allocation.PerSlotContext, in.Allocation.TotalContext, in.Allocation.Slots = 8192, 16384, 2
			batch, ubatch := int32(1024), int32(512)
			in.Allocation.Options.BatchSize, in.Allocation.Options.UBatchSize = &batch, &ubatch
			in.Allocation.Options.FlashAttention = &fa
			// 13 * (67108864 full + 37748736 SWA) = 1363148800.
			assertEstimateBytes(t, estimateFixtureResult(t, in), "attention-cache", "estimated", "1363148800")
		})
	}
	t.Run("padding and slots", func(t *testing.T) {
		in := estimateFixture("gemma2")
		in.Allocation.Slots, in.Allocation.TotalContext = 2, 2048
		in.Facts.Shards[0].Numeric["gemma2.attention.sliding_window"] = gguffacts.Numeric{Type: 10, Bits: 257}
		in.Facts.Shards[0].Numeric["gemma2.attention.sliding_window_pattern"] = gguffacts.Numeric{Type: 10, Bits: 2}
		assertEstimateBytes(t, estimateFixtureResult(t, in), "attention-cache", "estimated", "786432")
	})
	t.Run("cap before padding", func(t *testing.T) {
		in := estimateFixture("gemma2")
		in.Allocation.PerSlotContext, in.Allocation.TotalContext, in.Allocation.Slots = 256, 512, 2
		in.Facts.Shards[0].Numeric["gemma2.attention.sliding_window"] = gguffacts.Numeric{Type: 10, Bits: 4096}
		in.Facts.Shards[0].Numeric["gemma2.attention.sliding_window_pattern"] = gguffacts.Numeric{Type: 10, Bits: 2}
		assertEstimateBytes(t, estimateFixtureResult(t, in), "attention-cache", "estimated", "262144")
	})
	t.Run("missing microbatch remains unknown", func(t *testing.T) {
		in := estimateFixture("gemma2")
		in.Facts.Shards[0].Numeric["gemma2.attention.sliding_window"] = gguffacts.Numeric{Type: 10, Bits: 256}
		in.Facts.Shards[0].Numeric["gemma2.attention.sliding_window_pattern"] = gguffacts.Numeric{Type: 10, Bits: 2}
		in.Allocation.Options.UBatchSize = nil
		out := estimateFixtureResult(t, in)
		assertEstimateBytes(t, out, "attention-cache", "unknown", "")
		if out.Status != "partial" || len(out.Unknowns) == 0 {
			t.Fatalf("missing microbatch must remain partial: %+v", out)
		}
	})
	t.Run("full and unknown modes have distinct cache identities", func(t *testing.T) {
		in := estimateFixture("gemma2")
		in.Facts.Shards[0].Numeric["gemma2.attention.sliding_window"] = gguffacts.Numeric{Type: 10, Bits: 256}
		in.Facts.Shards[0].Numeric["gemma2.attention.sliding_window_pattern"] = gguffacts.Numeric{Type: 10, Bits: 2}
		var state modelEstimateState
		assertEstimateBytes(t, state.estimate(context.Background(), in), "attention-cache", "estimated", "393216")
		full := true
		in.SWAFull = &full
		assertEstimateBytes(t, state.estimate(context.Background(), in), "attention-cache", "estimated", "524288")
		in.SWAFull = nil
		out := state.estimate(context.Background(), in)
		assertEstimateBytes(t, out, "attention-cache", "unknown", "")
		if out.Status != "partial" || len(out.Unknowns) == 0 {
			t.Fatalf("unknown SWA mode must remain partial: %+v", out)
		}
		in.SWAFull = &full
		in.Allocation.Options.UBatchSize = nil
		assertEstimateBytes(t, state.estimate(context.Background(), in), "attention-cache", "estimated", "524288")
	})
}

func TestModelSettingsEstimateCompanions(t *testing.T) {
	in := estimateFixture("llama")
	in.Facts.Shards = append(in.Facts.Shards, gguffacts.Shard{Role: gguffacts.RoleDraft, Tensors: []gguffacts.Tensor{{Name: "draft", Type: 1, Dimensions: []uint64{32}, Bytes: 64}}}, gguffacts.Shard{Role: gguffacts.RoleCompanion, Tensors: []gguffacts.Tensor{{Name: "projector", Type: 1, Dimensions: []uint64{64}, Bytes: 128}}})
	in.Bindings = append(in.Bindings, modelconfig.ArtifactSummary{Role: "draft", Identity: "draft"}, modelconfig.ArtifactSummary{Role: "companion", Identity: "projector"})
	out := estimateFixtureResult(t, in)
	assertEstimateBytes(t, out, "weights", "estimated", "327936")
	assertEstimateBytes(t, out, "companions", "partial", "192")
	if c := estimateTestComponent(t, out, "companions"); c.DomainID != "" {
		t.Fatalf("companion runtime/placement was assumed: %+v", c)
	}
}

func TestModelSettingsEstimateCachePlacementBoundary(t *testing.T) {
	in := estimateFixture("qwen3moe")
	layers := int32(2)
	offload := true
	in.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: &layers}
	in.Allocation.Options.KVOffload = &offload
	partial := estimateFixtureResult(t, in)
	assertEstimateBytes(t, partial, "attention-cache", "estimated", "524288")
	if c := estimateTestComponent(t, partial, "attention-cache"); c.DomainID != "" {
		t.Fatalf("partial cache placement was guessed: %+v", c)
	}
	if partial.DeviceBytes != "164096" {
		t.Fatalf("unmapped partial cache was charged by layer fraction: %s", partial.DeviceBytes)
	}
	offload = false
	host := estimateFixtureResult(t, in)
	if c := estimateTestComponent(t, host, "attention-cache"); c.DomainID != "host" || host.HostBytes != "1736704" {
		t.Fatalf("disabled KV offload: %+v host=%s", c, host.HostBytes)
	}
	offload = true
	layers = 3
	full := estimateFixtureResult(t, in)
	if c := estimateTestComponent(t, full, "attention-cache"); c.DomainID != "offload-unmapped" || full.DeviceBytes != "786688" {
		t.Fatalf("full offload: %+v device=%s", c, full.DeviceBytes)
	}
}

func TestModelSettingsEstimateMalformedInputs(t *testing.T) {
	for _, tc := range []struct {
		name      string
		change    func(*modelEstimateInput)
		component string
	}{
		{"missing heads", func(in *modelEstimateInput) { delete(in.Facts.Shards[0].Numeric, "llama.attention.head_count") }, "attention-cache"},
		{"negative heads", func(in *modelEstimateInput) {
			in.Facts.Shards[0].Numeric["llama.attention.head_count"] = gguffacts.Numeric{Type: 11, Bits: math.MaxUint64}
		}, "attention-cache"},
		{"integer exceeds declared width", func(in *modelEstimateInput) {
			in.Facts.Shards[0].Numeric["llama.attention.head_count"] = gguffacts.Numeric{Type: 0, Bits: 260}
		}, "attention-cache"},
		{"nonfinite key width", func(in *modelEstimateInput) {
			in.Facts.Shards[0].Numeric["llama.attention.key_length"] = gguffacts.Numeric{Type: 12, Bits: math.Float64bits(math.NaN())}
		}, "attention-cache"},
		{"incomplete recurrent mask", func(in *modelEstimateInput) {
			in.Facts.Shards[0].Numeric["llama.attention.head_count_kv"] = gguffacts.Numeric{Type: 10, Array: []uint64{2}}
		}, "attention-cache"},
		{"unknown tensor type", func(in *modelEstimateInput) { in.Facts.Shards[0].Tensors[0].Type = math.MaxUint32 }, "weights"},
		{"zero tensor dimension", func(in *modelEstimateInput) { in.Facts.Shards[0].Tensors[0].Dimensions[0] = 0 }, "weights"},
		{"tensor byte disagreement", func(in *modelEstimateInput) { in.Facts.Shards[0].Tensors[0].Bytes++ }, "weights"},
		{"unknown support mapping", func(in *modelEstimateInput) { in.Support = "unmapped" }, "attention-cache"},
		{"missing precision", func(in *modelEstimateInput) { in.Allocation.Options.CacheTypeV = nil }, "attention-cache"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := estimateFixture("llama")
			tc.change(&in)
			out := estimateFixtureResult(t, in)
			c := estimateTestComponent(t, out, tc.component)
			if c.Status != "unknown" && c.Status != "partial" {
				t.Fatalf("malformed input produced %s", c.Status)
			}
			if out.Status != "partial" || len(out.Unknowns) == 0 {
				t.Fatalf("uncertainty lost: %+v", out)
			}
		})
	}
}

func TestModelSettingsEstimateOperationFailures(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*modelEstimateInput)
		reason string
	}{
		{"no facts", func(in *modelEstimateInput) { in.Facts = gguffacts.Facts{} }, "bounded artifact facts or ordered bindings unavailable"},
		{"tensor limit", func(in *modelEstimateInput) {
			in.Facts.Shards[0].Tensors = make([]gguffacts.Tensor, modelEstimateTensors+1)
		}, "estimator tensor/metadata limit"},
		{"slot arithmetic", func(in *modelEstimateInput) { in.Allocation.Slots = 3 }, "estimator context/slots limit or unsupported unified cache"},
		{"unpadded native context", func(in *modelEstimateInput) { in.Allocation.PerSlotContext = 1025; in.Allocation.TotalContext = 1025 }, "estimator context/slots limit or unsupported unified cache"},
		{"overflow", func(in *modelEstimateInput) { in.Facts.Shards[0].Tensors[0].Dimensions = []uint64{math.MaxUint64, 2} }, "estimator arithmetic overflow"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := estimateFixture("llama")
			tc.change(&in)
			out := estimateFixtureResult(t, in)
			if out.Status != "unknown" || !slices.Equal(out.Unknowns, []string{tc.reason}) {
				t.Fatalf("failure = %+v", out)
			}
		})
	}
	t.Run("panic recovers and releases cache", func(t *testing.T) {
		state := modelEstimateState{run: func(context.Context, modelEstimateInput) modelconfig.Estimate { panic("synthetic library panic") }}
		out := state.estimate(context.Background(), estimateFixture("llama"))
		if out.Status != "unknown" || !slices.Equal(out.Unknowns, []string{"estimator library panic"}) {
			t.Fatalf("panic result = %+v", out)
		}
		state.run = nil
		assertEstimateBytes(t, state.estimate(context.Background(), estimateFixture("llama")), "weights", "estimated", "327936")
	})
	t.Run("cancelled synchronous budget", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		var state modelEstimateState
		out := state.estimate(ctx, estimateFixture("llama"))
		if !slices.Equal(out.Unknowns, []string{"estimator operation budget exhausted"}) || len(state.entries) != 0 {
			t.Fatalf("budget result = %+v", out)
		}
	})
	t.Run("budget expires inside synchronous work", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		state := modelEstimateState{run: func(context.Context, modelEstimateInput) modelconfig.Estimate {
			cancel()
			return modelconfig.Estimate{Status: "estimated"}
		}}
		out := state.estimate(ctx, estimateFixture("llama"))
		if !slices.Equal(out.Unknowns, []string{"estimator operation budget exhausted"}) || len(state.entries) != 0 {
			t.Fatalf("expired work = %+v", out)
		}
	})
	t.Run("busy has no queue", func(t *testing.T) {
		var state modelEstimateState
		state.mu.Lock()
		defer state.mu.Unlock()
		out := state.estimate(context.Background(), estimateFixture("llama"))
		if !slices.Equal(out.Unknowns, []string{"estimator busy"}) {
			t.Fatalf("busy result = %+v", out)
		}
	})
}

func TestModelSettingsEstimateCacheIdentityAndTTL(t *testing.T) {
	now := time.Unix(100, 0)
	calls := 0
	state := modelEstimateState{now: func() time.Time { return now }, run: func(context.Context, modelEstimateInput) modelconfig.Estimate {
		calls++
		return modelconfig.Estimate{Status: "partial", Components: []modelconfig.EstimateComponent{estimateComponent("weights", 1, "synthetic fixture")}}
	}}
	in := estimateFixture("llama")
	first := state.estimate(context.Background(), in)
	first.Components[0].Bytes = "999"
	cached := state.estimate(context.Background(), in)
	if calls != 1 || cached.Components[0].Bytes != "1" {
		t.Fatalf("cache missed or returned aliased data: calls=%d result=%+v", calls, cached)
	}
	changes := []struct {
		name   string
		change func(*modelEstimateInput)
	}{
		{"binding identity", func(in *modelEstimateInput) { in.Bindings[0].Identity = "changed" }},
		{"ordered bindings", func(in *modelEstimateInput) {
			in.Bindings = append(in.Bindings, modelconfig.ArtifactSummary{Role: "draft", Identity: "draft"})
			slices.Reverse(in.Bindings)
		}},
		{"allocation", func(in *modelEstimateInput) { in.Allocation.TotalContext = 2048; in.Allocation.PerSlotContext = 2048 }},
		{"native build", func(in *modelEstimateInput) { in.BuildID = "other-build" }},
		{"support matrix", func(in *modelEstimateInput) { in.Support = "other-support" }},
		{"memory snapshot", func(in *modelEstimateInput) {
			in.Memory = &modelEstimateMemory{Identity: "snapshot", Source: "broker", At: now, Domains: []modelconfig.MemoryDomain{{ID: "ram", Kind: "host"}}}
		}},
	}
	for _, tc := range changes {
		t.Run(tc.name, func(t *testing.T) {
			in := estimateFixture("llama")
			tc.change(&in)
			before := calls
			state.estimate(context.Background(), in)
			if calls != before+1 {
				t.Fatal("different input reused result")
			}
		})
	}
	now = now.Add(modelEstimateTTL)
	before := calls
	state.estimate(context.Background(), estimateFixture("llama"))
	if calls != before+1 {
		t.Fatal("expired result was reused")
	}
}

func TestModelSettingsEstimateCacheWithdrawsExpiredComparisons(t *testing.T) {
	now := time.Now()
	calls := 0
	in := estimateFixture("llama")
	in.Memory = autoTestMemory(now)
	state := modelEstimateState{now: func() time.Time { return now }, run: func(ctx context.Context, in modelEstimateInput) modelconfig.Estimate {
		calls++
		return autoTestEstimate(ctx, in)
	}}
	fresh := state.estimate(context.Background(), in)
	if fresh.FreeFit != "fits" {
		t.Fatalf("synthetic comparison=%+v", fresh)
	}
	now = now.Add(6 * time.Second)
	stale := state.estimate(context.Background(), in)
	if stale.FreeFit != "unknown" || stale.CapacityFit != "unknown" || calls != 1 || len(stale.Components) != len(fresh.Components) {
		t.Fatalf("stale comparison=%+v calls=%d", stale, calls)
	}
	if fresh.Domains[0].FreeFit != "fits" {
		t.Fatal("cache return mutated previous result")
	}
}

func TestModelSettingsEstimateProductionPartialFitsWithWarnings(t *testing.T) {
	in := estimateFixture("llama")
	in.Memory = autoTestMemory(time.Now())
	in.Memory.UMA = true
	in.Memory.Domains = in.Memory.Domains[:1]
	in.Memory.Domains[0].Kind = "shared"
	out := estimateFixtureResult(t, in)
	if out.Status != "partial" || out.FreeFit != "fits" || out.CapacityFit != "fits" || len(out.Domains[0].Unknowns) == 0 {
		t.Fatalf("partial production estimator lost advisory fit/warnings: %+v", out)
	}
}

func TestModelSettingsEstimateUnboundManualGPUComparisonIsUnknown(t *testing.T) {
	in := estimateFixture("llama")
	layers := int32(1)
	in.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: &layers}
	in.Allocation.DeviceBinding = nil
	in.Memory = autoTestMemory(time.Now())
	// Complete synthetic costs isolate physical-placement eligibility from
	// production scratch accounting, which remains partial.
	state := modelEstimateState{run: autoTestEstimate}
	out := state.estimate(context.Background(), in)
	if out.FreeFit != "unknown" || out.CapacityFit != "unknown" {
		t.Fatal("unbound manual GPU acquired a fit verdict")
	}
	for _, domain := range out.Domains {
		if domain.Kind == "device" && (domain.FreeFit != "unknown" || domain.CapacityFit != "unknown") {
			t.Fatal("unbound GPU pool acquired a fit verdict")
		}
	}
}

func TestModelSettingsEstimateTargetCPUSharedPoolNeedsNoGPUMap(t *testing.T) {
	in := estimateFixture("llama")
	fact := noderec.MemoryFact{Bytes: "8589934592", Source: "synthetic-available", SampledAt: time.Now().UTC(), Valid: true}
	in.Memory = &modelEstimateMemory{Target: &modelconfig.TargetMemory{HostID: "synthetic-host", ObservationID: "synthetic-sample", Hop: noderec.NewMemoryHop("synthetic-broker", noderec.MemoryBootClock), Observation: &noderec.MemoryObservation{RunID: "synthetic-run", Domains: []noderec.MemoryDomain{{ID: "shared", Kind: "shared", Total: fact, Free: fact}}}}}
	state := modelEstimateState{run: autoTestEstimate}
	out := state.estimate(context.Background(), in)
	if out.FreeFit != "fits" || len(out.Domains) != 1 || out.Domains[0].Kind != "shared" || out.Domains[0].CapacityTimestamp != fact.SampledAt.Format(time.RFC3339Nano) {
		t.Fatalf("synthetic shared comparison=%+v", out)
	}
	in.Memory.Target.Hop = nil
	unknown := state.estimate(context.Background(), in)
	if unknown.FreeFit != "unknown" || len(unknown.Components) != len(out.Components) {
		t.Fatalf("invalid hop lost components: %+v", unknown)
	}
}

func estimateTargetFixture(shared bool) modelEstimateInput {
	in := estimateFixture("llama")
	layers := int32(3)
	in.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: &layers}
	in.Memory = autoTestMemory(time.Now())
	uuid := autoTestDeviceBinding().DeviceID
	in.Memory.Map.Binding = nil
	in.Memory.Map.Devices = []modelNativeDevice{{Selector: "CUDA0", Kind: 1, UUID: uuid, Pool: "gpu"}}
	fact := noderec.MemoryFact{Bytes: "8589934592", Source: "synthetic", SampledAt: time.Now().UTC(), Valid: true}
	domains := []noderec.MemoryDomain{{ID: "host", Kind: "system", Total: fact, Free: fact}, {ID: "gpu", Kind: "dedicated", DeviceIDs: []string{uuid}, Total: fact, Free: fact}}
	if shared {
		in.Memory.Map.HostPool = "shared"
		in.Memory.Map.Devices = nil
		domains = []noderec.MemoryDomain{{ID: "shared", Kind: "shared", Total: fact, Free: fact}}
	}
	in.Memory.Target = &modelconfig.TargetMemory{HostID: "synthetic-host", ObservationID: "synthetic-observation", MapID: in.Memory.Map.ObservationID, Hop: noderec.NewMemoryHop("synthetic-broker", noderec.MemoryBootClock), Observation: &noderec.MemoryObservation{RunID: "synthetic-run", Domains: domains}}
	return in
}

func TestModelSettingsEstimateRuntimeOverhead(t *testing.T) {
	out := estimateFixtureResult(t, estimateTargetFixture(false))
	c := estimateTestComponent(t, out, "runtime-overhead")
	if c.Status != "unknown" || c.DomainID != "" || c.Bytes != "" || c.Reason != "backend context, driver and allocator costs not yet calibrated or measured" {
		t.Fatalf("runtime overhead=%+v", c)
	}
	for _, d := range out.Domains {
		if !slices.Contains(d.Unknowns, "not priced (any pool): runtime-overhead") {
			t.Fatalf("missing overhead warning: %+v", d)
		}
	}
}

func TestModelSettingsEstimateAdvisorySoleFamilyGPU(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*modelEstimateInput)
		want   string
	}{
		{"sole evidenced CUDA GPU", func(*modelEstimateInput) {}, "fits"},
		{"absent map", func(in *modelEstimateInput) { in.Memory.Map = nil }, "unknown"},
		{"absent family", func(in *modelEstimateInput) { in.Memory.Map.Devices = nil }, "unknown"},
		{"unproved identity", func(in *modelEstimateInput) { in.Memory.Map.Devices[0].UUID = "" }, "unknown"},
		{"unmatched map", func(in *modelEstimateInput) { in.Memory.Target.MapID = "different" }, "unknown"},
		{"ambiguous family inventory", func(in *modelEstimateInput) {
			in.Memory.Map.Devices = append(in.Memory.Map.Devices, in.Memory.Map.Devices[0])
		}, "unknown"},
		{"expired", func(in *modelEstimateInput) {
			for i := range in.Memory.Target.Observation.Domains {
				in.Memory.Target.Observation.Domains[i].Total.AgeMS = 6000
				in.Memory.Target.Observation.Domains[i].Free.AgeMS = 6000
			}
		}, "unknown"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := estimateTargetFixture(false)
			tc.change(&in)
			before, err := json.Marshal(in)
			if err != nil {
				t.Fatal(err)
			}
			out := estimateFixtureResult(t, in)
			if out.FreeFit != tc.want {
				t.Fatalf("comparison=%+v want %s", out, tc.want)
			}
			if tc.want == "fits" && !slices.Contains(out.Domains[1].Unknowns, "device mapping assumed: only one GPU present") {
				t.Fatalf("missing advisory warning: %+v", out)
			}
			after, err := json.Marshal(in)
			if err != nil {
				t.Fatal(err)
			}
			if string(before) != string(after) || in.Allocation.DeviceBinding != nil {
				t.Fatal("advisory comparison changed launch inputs")
			}
		})
	}
}

func TestModelSettingsEstimateTargetUMACombinesRangesOnce(t *testing.T) {
	in := estimateTargetFixture(true)
	out := estimateFixtureResult(t, in)
	if len(out.Domains) != 1 || out.Domains[0].Kind != "shared" || out.FreeFit != "fits" {
		t.Fatalf("UMA comparison=%+v", out)
	}
	var expected, low, high uint64
	for _, c := range out.Components {
		if c.Bytes == "" || c.DomainID == "" {
			continue
		}
		if c.DomainID != "shared" {
			t.Fatalf("uncombined component: %+v", c)
		}
		for _, pair := range []struct {
			text string
			sum  *uint64
		}{{c.Bytes, &expected}, {c.LowBytes, &low}, {c.HighBytes, &high}} {
			n, err := strconv.ParseUint(pair.text, 10, 64)
			if err != nil {
				t.Fatal(err)
			}
			*pair.sum += n
		}
	}
	d := out.Domains[0]
	if d.RequiredBytes != strconv.FormatUint(expected, 10) || d.RequiredLowBytes != strconv.FormatUint(low, 10) || d.RequiredHighBytes != strconv.FormatUint(high, 10) || out.HostBytes != "" || out.DeviceBytes != "" {
		t.Fatalf("UMA charged incorrectly: %+v", out)
	}
	in.Memory.Map = nil
	if got := estimateFixtureResult(t, in); got.FreeFit != "unknown" {
		t.Fatalf("untrusted UMA fit: %+v", got)
	}
}

func TestModelSettingsEstimateCalibratedSplitWeights(t *testing.T) {
	key := modelEstimateCalibrationKey{"weights", "dense", "cuda"}
	modelEstimateCalibrations[key] = modelEstimateCalibration{LowNum: 1, LowDen: 2, HighNum: 2, HighDen: 1, Provenance: []string{"synthetic-fit-a", "synthetic-fit-b", "synthetic-fit-c"}}
	t.Cleanup(func() { delete(modelEstimateCalibrations, key) })
	out := estimateFixtureResult(t, estimateTargetFixture(false))
	parts := 0
	for _, c := range out.Components {
		if c.Name != "weights" {
			continue
		}
		parts++
		n, err := strconv.ParseUint(c.Bytes, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		if c.Basis != "calibrated" || c.LowBytes != strconv.FormatUint(n/2, 10) || c.HighBytes != strconv.FormatUint(n*2, 10) {
			t.Fatalf("split weights=%+v", c)
		}
	}
	if parts != 2 || out.FreeFit != "fits" {
		t.Fatalf("split estimate=%+v", out)
	}
}

func TestModelSettingsEstimateTargetUMAWithdrawsUntrustedEvidence(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*modelEstimateInput)
	}{
		{"missing map", func(in *modelEstimateInput) { in.Memory.Map = nil }},
		{"invalid hop", func(in *modelEstimateInput) { in.Memory.Target.Hop = nil }},
		{"unknown map", func(in *modelEstimateInput) { in.Memory.Map.Unknown = "untrusted topology" }},
		{"expired capacity", func(in *modelEstimateInput) { in.Memory.Target.Observation.Domains[0].Total.AgeMS = 6000 }},
		{"mixed topology", func(in *modelEstimateInput) {
			in.Memory.Target.Observation.Domains = append(in.Memory.Target.Observation.Domains, noderec.MemoryDomain{ID: "gpu", Kind: "dedicated", DeviceIDs: []string{autoTestDeviceBinding().DeviceID}, Total: in.Memory.Target.Observation.Domains[0].Total})
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := estimateTargetFixture(true)
			tc.change(&in)
			out := estimateFixtureResult(t, in)
			if out.CapacityFit != "unknown" || len(out.Components) == 0 {
				t.Fatalf("untrusted UMA comparison=%+v", out)
			}
		})
	}
}

func TestModelSettingsEstimateCalibrationBands(t *testing.T) {
	if len(modelEstimateCalibrations) != 0 {
		t.Fatal("calibration table must start empty")
	}
	t.Run("default rounds outward", func(t *testing.T) {
		low, high, basis := modelEstimateBand("compute-workspace", "dense", "cpu", 5)
		if low != 3 || high != 8 || basis != "heuristic" {
			t.Fatalf("band=(%d,%d,%s)", low, high, basis)
		}
	})
	t.Run("checked overflow", func(t *testing.T) {
		defer func() {
			if r := recover(); r != errModelEstimateOverflow {
				t.Fatalf("overflow=%v", r)
			}
		}()
		modelEstimateBand("compute-workspace", "dense", "cpu", math.MaxUint64)
	})
	t.Run("derived rows only change with matching evidence", func(t *testing.T) {
		key := modelEstimateCalibrationKey{"weights", "dense", "cpu"}
		modelEstimateCalibrations[key] = modelEstimateCalibration{LowNum: 1, LowDen: 2, HighNum: 2, HighDen: 1, Provenance: []string{"synthetic-fit-a", "synthetic-fit-b", "synthetic-fit-c"}}
		t.Cleanup(func() { delete(modelEstimateCalibrations, key) })
		in := estimateFixture("llama")
		out := estimateFixtureResult(t, in)
		weights := estimateTestComponent(t, out, "weights")
		n, err := strconv.ParseUint(weights.Bytes, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		if weights.Basis != "calibrated" || weights.LowBytes != strconv.FormatUint(n/2, 10) || weights.HighBytes != strconv.FormatUint(n*2, 10) {
			t.Fatalf("calibrated weights=%+v", weights)
		}
		cache := estimateTestComponent(t, out, "attention-cache")
		if cache.Basis != "derived" || cache.LowBytes != cache.Bytes || cache.HighBytes != cache.Bytes {
			t.Fatalf("uncalibrated derived row widened: %+v", cache)
		}
		low, high, basis := modelEstimateBand("weights", "dense", "cuda", 5)
		if low != 3 || high != 8 || basis != "heuristic" {
			t.Fatalf("calibration leaked across family=(%d,%d,%s)", low, high, basis)
		}
	})
}

func TestModelSettingsEstimateTargetGPURequiresExactMapAndBinding(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*modelEstimateInput)
		want   string
	}{
		{"synthetic exact binding", func(*modelEstimateInput) {}, "fits"},
		{"map identity changed", func(in *modelEstimateInput) { in.Memory.Target.MapID = "other-map" }, "unknown"},
		{"missing binding", func(in *modelEstimateInput) { in.Allocation.DeviceBinding = nil }, "unknown"},
		{"wrong physical binding", func(in *modelEstimateInput) { in.Allocation.DeviceBinding.DomainID = "other-gpu" }, "unknown"},
		{"missing observation device identity", func(in *modelEstimateInput) { in.Memory.Target.Observation.Domains[1].DeviceIDs = nil }, "unknown"},
		{"mismatched observation device identity", func(in *modelEstimateInput) {
			in.Memory.Target.Observation.Domains[1].DeviceIDs = []string{"GPU-22222222-2222-2222-2222-222222222222"}
		}, "unknown"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := estimateFixture("llama")
			layers := int32(1)
			in.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: &layers}
			in.Allocation.DeviceBinding = autoTestDeviceBinding()
			in.Memory = autoTestMemory(time.Now())
			fact := noderec.MemoryFact{Bytes: "8589934592", Source: "synthetic", SampledAt: time.Now().UTC(), Valid: true}
			in.Memory.Target = &modelconfig.TargetMemory{HostID: "synthetic-host", ObservationID: "synthetic-observation", MapID: in.Memory.Map.ObservationID, Hop: noderec.NewMemoryHop("synthetic-broker", noderec.MemoryBootClock), Observation: &noderec.MemoryObservation{RunID: "synthetic-run", Domains: []noderec.MemoryDomain{{ID: "host", Kind: "system", Total: fact, Free: fact}, {ID: "gpu", Kind: "dedicated", DeviceIDs: []string{in.Allocation.DeviceBinding.DeviceID}, Total: fact, Free: fact}}}}
			tc.mutate(&in)
			state := modelEstimateState{run: autoTestEstimate}
			out := state.estimate(context.Background(), in)
			if out.FreeFit != tc.want || len(out.Components) == 0 {
				t.Fatalf("synthetic mapped comparison=%+v want %s", out, tc.want)
			}
		})
	}
}

func TestModelSettingsEstimateCacheLimitsAndLRU(t *testing.T) {
	state := modelEstimateState{run: func(context.Context, modelEstimateInput) modelconfig.Estimate {
		return modelconfig.Estimate{Status: "unknown"}
	}}
	for i := 0; i < modelEstimateRecords; i++ {
		in := estimateFixture("llama")
		in.BuildID = strconv.Itoa(i)
		state.estimate(context.Background(), in)
	}
	if len(state.entries) != modelEstimateRecords {
		t.Fatalf("cache entries=%d", len(state.entries))
	}
	firstKey := state.entries[0].key
	state.estimate(context.Background(), estimateFixtureWithBuild("0"))
	state.estimate(context.Background(), estimateFixtureWithBuild("new"))
	if len(state.entries) != modelEstimateRecords || state.entries[len(state.entries)-2].key != firstKey {
		t.Fatal("cache did not retain touched LRU entry")
	}
	if state.bytes+modelEstimateSlotBytes > modelEstimateBytes {
		t.Fatal("cache byte ceiling exceeded")
	}
	// A large but contract-valid result forces byte eviction before record limit.
	state = modelEstimateState{run: func(context.Context, modelEstimateInput) modelconfig.Estimate {
		return modelconfig.Estimate{Status: "unknown", Unknowns: []string{strings.Repeat("x", 12000)}}
	}}
	for i := 0; i < modelEstimateRecords; i++ {
		state.estimate(context.Background(), estimateFixtureWithBuild(strconv.Itoa(i)))
	}
	if len(state.entries) >= modelEstimateRecords || state.bytes+modelEstimateSlotBytes > modelEstimateBytes {
		t.Fatalf("byte eviction failed: %d entries, %d bytes", len(state.entries), state.bytes)
	}
}

func estimateFixtureWithBuild(build string) modelEstimateInput {
	in := estimateFixture("llama")
	in.BuildID = build
	return in
}

func TestModelSettingsEstimateAbsentTelemetryPreservesComponents(t *testing.T) {
	in := estimateFixture("qwen3moe")
	if in.Memory != nil {
		t.Fatal("fixture unexpectedly supplies target telemetry")
	}
	out := estimateFixtureResult(t, in)
	if out.Status != "partial" || !slices.Contains(out.Unknowns, "target memory telemetry unavailable") {
		t.Fatalf("absent telemetry must leave comparison unknown: %+v", out)
	}
	if out.MemorySource != "" || out.MemoryTimestamp != "" {
		t.Fatal("absent telemetry acquired a source or timestamp")
	}
	if len(out.Domains) == 0 {
		t.Fatal("component memory domains were lost")
	}
	for _, domain := range out.Domains {
		if domain.CapacityBytes != "" || domain.FreeBytes != "" || domain.DeviceID != "" {
			t.Fatalf("absent telemetry invented capacity, free memory or a device map: %+v", domain)
		}
	}
	assertEstimateBytes(t, out, "weights", "estimated", "327936")
	assertEstimateBytes(t, out, "attention-cache", "estimated", "524288")
	assertEstimateBytes(t, out, "prompt-cache", "estimated", "1048576")
	assertEstimateBytes(t, out, "compute-workspace", "unknown", "")
	assertEstimateBytes(t, out, "companions", "excluded", "")
}

func TestModelSettingsEstimateMemoryDomains(t *testing.T) {
	now := time.Unix(1000, 0)
	for _, tc := range []struct {
		name        string
		memory      modelEstimateMemory
		wantDomains int
		wantSource  bool
	}{
		{"fresh UMA", modelEstimateMemory{Identity: "u", Source: "broker", At: now, UMA: true, Domains: []modelconfig.MemoryDomain{{ID: "uma", Kind: "shared", CapacityBytes: "10000000", FreeBytes: "8000000"}}}, 1, true},
		{"stale telemetry", modelEstimateMemory{Identity: "s", Source: "broker", At: now.Add(-6 * time.Second), Domains: []modelconfig.MemoryDomain{{ID: "ram", Kind: "host"}}}, 2, false},
		{"future telemetry", modelEstimateMemory{Identity: "f", Source: "broker", At: now.Add(time.Second), Domains: []modelconfig.MemoryDomain{{ID: "ram", Kind: "host"}}}, 2, false},
		{"free exceeds capacity", modelEstimateMemory{Identity: "f", Source: "broker", At: now, Domains: []modelconfig.MemoryDomain{{ID: "ram", Kind: "host", CapacityBytes: "1", FreeBytes: "2"}}}, 2, false},
		{"unsigned overflow", modelEstimateMemory{Identity: "o", Source: "broker", At: now, Domains: []modelconfig.MemoryDomain{{ID: "ram", Kind: "host", CapacityBytes: "18446744073709551616"}}}, 2, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := estimateFixture("qwen3moe")
			in.Memory = &tc.memory
			state := modelEstimateState{now: func() time.Time { return now }}
			out := state.estimate(context.Background(), in)
			if err := out.Validate(); err != nil {
				t.Fatal(err)
			}
			if len(out.Domains) != tc.wantDomains || (out.MemorySource != "") != tc.wantSource || out.Status != "partial" {
				t.Fatalf("memory result = %+v", out)
			}
			if tc.memory.UMA && (out.Domains[0].Kind != "shared" || out.HostBytes != "" || out.DeviceBytes != "") {
				t.Fatal("UMA double counted")
			}
			if tc.memory.UMA && out.Domains[0].RequiredBytes != "1900800" {
				t.Fatalf("UMA subtotal = %s", out.Domains[0].RequiredBytes)
			}
		})
	}
}

func TestModelSettingsEstimateMemoryMapping(t *testing.T) {
	now := time.Unix(1000, 0)
	in := estimateFixture("qwen3moe")
	layers := int32(3)
	in.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: &layers}
	on := true
	in.Allocation.Options.KVOffload = &on
	in.Memory = &modelEstimateMemory{Identity: "mapped", Source: "broker", At: now, Domains: []modelconfig.MemoryDomain{{ID: "gpu-physical", Kind: "device", DeviceID: "uuid-1", CapacityBytes: "10000000"}}}
	state := modelEstimateState{now: func() time.Time { return now }}
	unmapped := state.estimate(context.Background(), in)
	if unmapped.Domains[1].DeviceID != "" || unmapped.Domains[1].CapacityBytes != "" {
		t.Fatal("array order was treated as device mapping")
	}
	in.Memory.NativeDeviceID = "uuid-1"
	mapped := state.estimate(context.Background(), in)
	if mapped.Domains[1].ID != "gpu-physical" || mapped.Domains[1].DeviceID != "uuid-1" || mapped.Status != "partial" {
		t.Fatalf("mapped result = %+v", mapped)
	}
	if c := estimateTestComponent(t, mapped, "attention-cache"); c.DomainID != "gpu-physical" {
		t.Fatalf("cache mapping = %+v", c)
	}
}

// Same target/draft, same preview AND launch digest, regardless of estimator
// success, failure or panic. The existing fixture is synthetic and stays Off.
func TestModelSettingsEstimateDigestIndependence(t *testing.T) {
	f := newModelLoadFixture(t)
	f.rt = convertModelLoadFixture(t, f)
	f.st.plat.Runtime = f.rt
	state, err := f.ex.LaunchSettings("fake")
	if err != nil {
		t.Fatal(err)
	}
	three := int32(3)
	request := settings.Request{Engine: "fake", Settings: settings.ConfigFromLaunch(state, 54301), ModelChange: &modelconfig.ModelChange{ModelID: "medium-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &three}}}}
	before := snapshotLlamaState(f.st)
	var baseline settings.Preview
	var launchDigest string
	for _, tc := range []struct {
		name string
		run  func(context.Context, modelEstimateInput) modelconfig.Estimate
		want string
	}{
		{"success", func(context.Context, modelEstimateInput) modelconfig.Estimate {
			return modelconfig.Estimate{Status: "partial", Components: []modelconfig.EstimateComponent{estimateComponent("weights", 1, "synthetic")}}
		}, "partial"},
		{"failure", func(context.Context, modelEstimateInput) modelconfig.Estimate {
			return modelEstimateUnknown("synthetic failure")
		}, "unknown"},
		{"panic", func(context.Context, modelEstimateInput) modelconfig.Estimate { panic("synthetic library panic") }, "unknown"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f.ex.modelEstimates = modelEstimateState{run: tc.run}
			preview, err := f.ex.previewModelLoadChange(context.Background(), f.st, request)
			if err != nil || len(preview.Errors) != 0 || preview.PreviewDigest == "" || preview.Estimate == nil || preview.Estimate.Status != tc.want {
				t.Fatalf("preview failed: %+v %v", preview, err)
			}
			rt := f.rt
			rt.ModelLoads = modelconfig.CloneLoads(preview.Settings.ModelLoads)
			command := typedFixtureCommand(t, f, rt)
			compiled, err := compileTypedLlamaInputs(context.Background(), f.st, command, rt, true)
			if err != nil {
				t.Fatal(err)
			}
			if err := f.ex.validateModelSettingsLaunch(context.Background(), settings.Request{Engine: "fake", Settings: preview.Settings}, true); err != nil {
				t.Fatalf("estimate blocked finite Apply validation: %v", err)
			}
			if tc.name == "success" {
				baseline = preview
				launchDigest = compiled.Digest
			} else {
				if preview.PreviewDigest != baseline.PreviewDigest || compiled.Digest != launchDigest || !preview.Settings.Equal(baseline.Settings) {
					t.Fatal("estimate altered review/native digest or saved proposal")
				}
				preview.Estimate = nil
				expected := baseline
				expected.Estimate = nil
				if !reflect.DeepEqual(preview, expected) {
					t.Fatal("estimator changed non-advisory preview fields")
				}
			}
			if !reflect.DeepEqual(before.plat.Runtime, f.st.plat.Runtime) || f.st.running || f.st.proc != nil {
				t.Fatal("preview changed runtime or effects")
			}
			for _, path := range []string{f.capture, filepath.Join(f.ex.overrideDir, "fake.json")} {
				if _, err := os.Stat(path); !os.IsNotExist(err) {
					t.Fatalf("unexpected preview effect %s: %v", path, err)
				}
			}
		})
	}
}

func TestModelSettingsEstimateInspectionUsesSameFacts(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	var sawFacts bool
	f.ex.modelEstimates.run = func(_ context.Context, in modelEstimateInput) modelconfig.Estimate {
		sawFacts = len(in.Facts.Shards) > 0 && len(in.Bindings) > 0 && strings.HasPrefix(in.BuildID, "unverified-launch-snapshot:")
		return calculateModelEstimate(context.Background(), in)
	}
	out, err := f.ex.InspectModel(context.Background(), f.request)
	if err != nil {
		t.Fatal(err)
	}
	if !sawFacts || out.Estimate.Status != "partial" {
		t.Fatalf("same-read facts handoff unavailable: %+v", out)
	}
	assertEstimateBytes(t, out.Estimate, "weights", "estimated", "128")
	assertEstimateBytes(t, out.Estimate, "attention-cache", "unknown", "")
	for _, option := range out.Options {
		if option.Field == "generation.temperature" && (string(option.Default) != "0.25" || option.DefaultSource != "embedded GGUF general.sampling.temp") {
			t.Fatalf("recommendation provenance changed: %+v", option)
		}
	}
}

type estimateBudgetContext struct {
	context.Context
	firstDeadlineCheck time.Time
}

func (c *estimateBudgetContext) Deadline() (time.Time, bool) {
	if c.firstDeadlineCheck.IsZero() {
		c.firstDeadlineCheck = time.Now()
	}
	return time.Time{}, false
}

func TestModelSettingsEstimateInspectionSharesBudget(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	ctx := &estimateBudgetContext{Context: context.Background()}
	called := false
	f.ex.modelEstimates.run = func(estimateCtx context.Context, in modelEstimateInput) modelconfig.Estimate {
		called = true
		deadline, ok := estimateCtx.Deadline()
		if !ok || deadline.After(ctx.firstDeadlineCheck.Add(modelInspectionBudget)) {
			t.Fatal("estimation received a renewed inspection budget")
		}
		return calculateModelEstimate(estimateCtx, in)
	}
	if _, err := f.ex.InspectModel(ctx, f.request); err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("estimation was not exercised")
	}
}

func TestModelSettingsEstimateInputIsReadOnly(t *testing.T) {
	in := estimateFixture("qwen3moe")
	before, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	estimateFixtureResult(t, in)
	after, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(before, after) {
		t.Fatal("estimator mutated facts, requested/resolved settings or memory input")
	}
}
