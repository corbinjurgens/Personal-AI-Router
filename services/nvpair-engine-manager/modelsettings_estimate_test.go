// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/binary"
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

func recurrentFixture(arch string) modelEstimateInput {
	in := estimateFixture(arch)
	for _, field := range []string{"block_count", "embedding_length"} {
		n := in.Facts.Shards[0].Numeric[arch+"."+field]
		n.Type = 4 // Native scalar hparams load these fields as GGUF uint32.
		in.Facts.Shards[0].Numeric[arch+"."+field] = n
	}
	for key, value := range map[string]uint64{"ssm.conv_kernel": 4, "ssm.inner_size": 256, "ssm.state_size": 16, "ssm.time_step_rank": 8, "ssm.group_count": 2, "wkv.head_size": 32} {
		in.Facts.Shards[0].Numeric[arch+"."+key] = gguffacts.Numeric{Type: 4, Bits: value}
	}
	// Families must not borrow dimensions belonging to another native branch.
	if arch == "rwkv6" || arch == "rwkv7" {
		for key := range in.Facts.Shards[0].Numeric {
			if strings.Contains(key, ".ssm.") {
				delete(in.Facts.Shards[0].Numeric, key)
			}
		}
	} else {
		delete(in.Facts.Shards[0].Numeric, arch+".wkv.head_size")
	}
	if arch == "mamba" {
		delete(in.Facts.Shards[0].Numeric, arch+".ssm.group_count")
	}
	if slices.Contains([]string{"qwen3next", "qwen35", "qwen35moe"}, arch) {
		in.Facts.Shards[0].Numeric[arch+".ssm.time_step_rank"] = gguffacts.Numeric{Type: 4, Bits: 16}
	}
	return in
}

func TestModelSettingsEstimateRecurrentFamilies(t *testing.T) {
	for _, tc := range []struct{ arch, want string }{{"mamba", "38912"}, {"mamba2", "40448"}, {"rwkv6", "34816"}, {"rwkv7", "34816"}, {"qwen3next", "40448"}, {"qwen35", "40448"}, {"qwen35moe", "40448"}} {
		t.Run(tc.arch, func(t *testing.T) {
			c := estimateTestComponent(t, estimateFixtureResult(t, recurrentFixture(tc.arch)), "recurrent-state")
			if c.Status != "estimated" || c.Bytes != tc.want || c.Basis != "derived" {
				t.Fatalf("state=%+v, want %s derived", c, tc.want)
			}
			if slices.Contains([]string{"mamba", "mamba2", "rwkv6", "rwkv7"}, tc.arch) {
				assertEstimateBytes(t, estimateFixtureResult(t, recurrentFixture(tc.arch)), "attention-cache", "excluded", "")
			}
		})
	}
}

func TestModelSettingsEstimateRecurrentSlotsAndPlacement(t *testing.T) {
	for _, tc := range []struct {
		name         string
		layers       int32
		offload      bool
		host, device string
	}{
		{"output only", 1, true, "121344", ""},
		{"trailing dense layer", 2, true, "121344", ""},
		{"trailing recurrent and dense", 3, true, "80896", "40448"},
		{"all blocks", 5, true, "", "121344"},
		{"offload disabled", 3, false, "121344", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := recurrentFixture("qwen35")
			in.Allocation.Slots, in.Allocation.TotalContext = 2, 2048
			in.Facts.Shards[0].Numeric["qwen35.block_count"] = gguffacts.Numeric{Type: 4, Bits: 4}
			in.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: &tc.layers}
			in.Allocation.Options.KVOffload = &tc.offload
			var host, device string
			for _, c := range estimateFixtureResult(t, in).Components {
				if c.Name == "recurrent-state" && c.Bytes != "0" {
					if c.DomainID == "host" {
						host = c.Bytes
					} else if c.DomainID == "offload-unmapped" {
						device = c.Bytes
					}
				}
			}
			if host != tc.host || device != tc.device {
				t.Fatalf("host=%s device=%s, want %s %s", host, device, tc.host, tc.device)
			}
		})
	}
}

func TestModelSettingsEstimateRecurrentMasksAndMainContext(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*modelEstimateInput)
		want string
	}{
		{"slots without context padding", func(in *modelEstimateInput) { in.Allocation.Slots = 3; in.Allocation.TotalContext = 3072 }, "121344"},
		{"boolean mask precedence", func(in *modelEstimateInput) {
			in.Facts.Shards[0].Numeric["qwen35.attention.recurrent_layers"] = gguffacts.Numeric{Type: 7, Array: []uint64{0, 1}}
		}, "20224"},
		{"integer mask precedence", func(in *modelEstimateInput) {
			in.Facts.Shards[0].Numeric["qwen35.attention.recurrent_layers"] = gguffacts.Numeric{Type: 4, Array: []uint64{0, 1}}
		}, "20224"},
		{"scalar integer mask", func(in *modelEstimateInput) {
			in.Facts.Shards[0].Numeric["qwen35.attention.recurrent_layers"] = gguffacts.Numeric{Type: 4, Bits: 1}
		}, "40448"},
		{"nondefault interval", func(in *modelEstimateInput) {
			in.Facts.Shards[0].Numeric["qwen35.full_attention_interval"] = gguffacts.Numeric{Type: 4, Bits: 2}
		}, "20224"},
		{"NextN excluded", func(in *modelEstimateInput) {
			in.Facts.Shards[0].Numeric["qwen35.nextn_predict_layers"] = gguffacts.Numeric{Type: 4, Bits: 1}
		}, "20224"},
		{"NextN explicit mask excluded", func(in *modelEstimateInput) {
			in.Facts.Shards[0].Numeric["qwen35.nextn_predict_layers"] = gguffacts.Numeric{Type: 4, Bits: 1}
			in.Facts.Shards[0].Numeric["qwen35.attention.recurrent_layers"] = gguffacts.Numeric{Type: 7, Array: []uint64{1, 0}}
		}, "20224"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := recurrentFixture("qwen35")
			tc.edit(&in)
			assertEstimateBytes(t, estimateFixtureResult(t, in), "recurrent-state", "estimated", tc.want)
		})
	}
}

func TestModelSettingsEstimateRecurrentExplicitMaskPlacement(t *testing.T) {
	in := recurrentFixture("qwen35")
	in.Facts.Shards[0].Numeric["qwen35.block_count"] = gguffacts.Numeric{Type: 4, Bits: 4}
	in.Facts.Shards[0].Numeric["qwen35.attention.recurrent_layers"] = gguffacts.Numeric{Type: 7, Array: []uint64{0, 1, 0, 1}}
	layers, offload := int32(2), true
	in.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: &layers}
	in.Allocation.Options.KVOffload = &offload
	var domains []string
	for _, c := range estimateFixtureResult(t, in).Components {
		if c.Name == "recurrent-state" {
			if c.Bytes != "20224" || c.Status != "estimated" {
				t.Fatalf("state=%+v", c)
			}
			domains = append(domains, c.DomainID)
		}
	}
	if !reflect.DeepEqual(domains, []string{"host", "offload-unmapped"}) {
		t.Fatalf("domains=%v", domains)
	}
}

func TestModelSettingsEstimateRecurrentRejectsUnsupportedDimensions(t *testing.T) {
	for _, tc := range []struct {
		name, arch, key string
		value           gguffacts.Numeric
	}{
		{"missing state", "mamba2", "ssm.state_size", gguffacts.Numeric{}},
		{"zero conv", "mamba2", "ssm.conv_kernel", gguffacts.Numeric{Type: 4}},
		{"array state", "mamba2", "ssm.state_size", gguffacts.Numeric{Type: 4, Array: []uint64{16}}},
		{"native width overflow", "mamba2", "ssm.inner_size", gguffacts.Numeric{Type: 10, Bits: math.MaxUint32}},
		{"indivisible groups", "mamba2", "ssm.group_count", gguffacts.Numeric{Type: 4, Bits: 3}},
		{"malformed shift", "rwkv6", "token_shift_count", gguffacts.Numeric{Type: 7, Bits: 1}},
		{"signed conv", "mamba2", "ssm.conv_kernel", gguffacts.Numeric{Type: 5, Bits: 4}},
		{"signed inner width", "mamba2", "ssm.inner_size", gguffacts.Numeric{Type: 5, Bits: 256}},
		{"wide state scalar", "qwen35", "ssm.state_size", gguffacts.Numeric{Type: 10, Bits: 16}},
		{"signed interval", "qwen35", "full_attention_interval", gguffacts.Numeric{Type: 5, Bits: 2}},
		{"small head scalar", "rwkv7", "wkv.head_size", gguffacts.Numeric{Type: 0, Bits: 32}},
		{"wide NextN scalar", "qwen35", "nextn_predict_layers", gguffacts.Numeric{Type: 10, Bits: 1}},
		{"zero head", "rwkv7", "wkv.head_size", gguffacts.Numeric{Type: 4}},
		{"incomplete mask", "qwen35", "attention.recurrent_layers", gguffacts.Numeric{Type: 7, Array: []uint64{1}}},
		{"invalid interval", "qwen35", "full_attention_interval", gguffacts.Numeric{Type: 4}},
		{"mismatched value heads", "qwen35", "ssm.time_step_rank", gguffacts.Numeric{Type: 4, Bits: 8}},
		{"NextN recurrent mask", "qwen35", "nextn_predict_layers", gguffacts.Numeric{Type: 4, Bits: 2}},
		{"boolean scalar is not native uint32 mask", "qwen35", "attention.recurrent_layers", gguffacts.Numeric{Type: 7, Bits: 1}},
		{"oversized mask bit", "qwen35", "attention.recurrent_layers", gguffacts.Numeric{Type: 4, Array: []uint64{1, 2}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := recurrentFixture(tc.arch)
			in.Facts.Shards[0].Numeric[tc.arch+"."+tc.key] = tc.value
			c := estimateTestComponent(t, estimateFixtureResult(t, in), "recurrent-state")
			if c.Status != "unknown" || c.Bytes != "" || c.Reason == "" {
				t.Fatalf("state=%+v", c)
			}
		})
	}
	t.Run("unconfirmed family", func(t *testing.T) {
		assertEstimateBytes(t, estimateFixtureResult(t, recurrentFixture("jamba")), "recurrent-state", "unknown", "")
	})
}

func TestModelSettingsEstimateRecurrentDecodedTokenShift(t *testing.T) {
	for _, arch := range []string{"rwkv6", "rwkv7"} {
		t.Run(arch, func(t *testing.T) {
			for _, tc := range []struct {
				name     string
				explicit bool
				typ      gguffacts.ValueType
				status   string
				want     string
			}{
				{"native absent default", false, 4, "estimated", "34816"},
				{"decoded override", true, 4, "estimated", "35840"},
				{"decoded signed override", true, 5, "unknown", ""},
				{"decoded wide override", true, 10, "unknown", ""},
				{"decoded small override", true, 0, "unknown", ""},
			} {
				t.Run(tc.name, func(t *testing.T) {
					var b bytes.Buffer
					write := func(v interface{}) {
						t.Helper()
						if err := binary.Write(&b, binary.LittleEndian, v); err != nil {
							t.Fatal(err)
						}
					}
					str := func(s string) { write(uint64(len(s))); b.WriteString(s) }
					b.WriteString("GGUF")
					write(uint32(3))
					write(uint64(0))
					count := uint64(4)
					if tc.explicit {
						count++
					}
					write(count)
					str("general.architecture")
					write(uint32(8))
					str(arch)
					for _, item := range []struct {
						key string
						v   uint32
					}{{"block_count", 2}, {"embedding_length", 128}, {"wkv.head_size", 32}} {
						str(arch + "." + item.key)
						write(uint32(4))
						write(item.v)
					}
					if tc.explicit {
						str(arch + ".token_shift_count")
						write(tc.typ)
						switch tc.typ {
						case 0:
							write(uint8(3))
						case 5:
							write(int32(3))
						case 10:
							write(uint64(3))
						default:
							write(uint32(3))
						}
					}
					facts, err := gguffacts.Decode(context.Background(), []gguffacts.Artifact{{Role: gguffacts.RoleMain, Reader: bytes.NewReader(b.Bytes()), Size: int64((b.Len() + 31) / 32 * 32)}})
					if err != nil {
						t.Fatal(err)
					}
					if tc.explicit && facts.Shards[0].Numeric[arch+".token_shift_count"].Bits != 3 {
						t.Fatal("decoder lost override")
					}
					if tc.explicit && facts.Shards[0].Numeric[arch+".token_shift_count"].Type != tc.typ {
						t.Fatal("decoder changed numeric type")
					}
					in := estimateFixture(arch)
					in.Facts.Shards[0].Numeric = facts.Shards[0].Numeric
					assertEstimateBytes(t, estimateFixtureResult(t, in), "recurrent-state", tc.status, tc.want)
				})
			}
		})
	}
}

func TestModelSettingsEstimateRecurrentExplicitMaskKeepsAttentionUnknown(t *testing.T) {
	in := recurrentFixture("qwen35")
	in.Facts.Shards[0].Numeric["qwen35.full_attention_interval"] = gguffacts.Numeric{Type: 4, Bits: 2}
	in.Facts.Shards[0].Numeric["qwen35.attention.recurrent_layers"] = gguffacts.Numeric{Type: 4, Array: []uint64{0, 0}}
	out := estimateFixtureResult(t, in)
	c := estimateTestComponent(t, out, "attention-cache")
	// Native qwen35.cpp:17-21 gives this mask priority over the interval.
	// Do not price one attention layer when the explicit mask selects two.
	if c.Status != "unknown" || !strings.Contains(c.Reason, "explicit recurrent") {
		t.Fatalf("attention=%+v, want named explicit-mask unknown", c)
	}
	assertEstimateBytes(t, out, "recurrent-state", "estimated", "0")
}

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

func TestModelSettingsEstimatePinnedSpecialCaches(t *testing.T) {
	test := func(name, arch string, facts map[string]gguffacts.Numeric, fa string, full bool, want string) {
		t.Run(name, func(t *testing.T) {
			in := estimateFixture(arch)
			for key, value := range facts {
				in.Facts.Shards[0].Numeric[arch+"."+key] = value
			}
			in.Allocation.Options.FlashAttention = &fa
			in.SWAFull = &full
			c := estimateTestComponent(t, estimateFixtureResult(t, in), "attention-cache")
			if c.Status != "estimated" || c.Bytes != want || c.LowBytes != want || c.HighBytes != want || c.Basis != "derived" {
				t.Fatalf("cache = %+v, want derived %s", c, want)
			}
		})
	}
	n := func(v uint64) gguffacts.Numeric { return gguffacts.Numeric{Type: 10, Bits: v} }
	mla := map[string]gguffacts.Numeric{"attention.head_count_kv": n(1), "attention.key_length": n(96), "attention.value_length": n(64), "attention.key_length_mla": n(48), "attention.value_length_mla": n(32), "attention.kv_lora_rank": n(64), "rope.dimension_count": n(32)}
	// 7fe450e1 src/llama-kv-cache.cpp:229-234: K only, 2*96*1024*2 = 393216.
	// src/llama-model.cpp:1371-1380 loads the stored width, not the MLA query width.
	test("MLA FA on", "deepseek2", mla, "on", false, "393216")
	test("MLA FA off", "deepseek2", mla, "off", false, "393216")
	mlaNext := make(map[string]gguffacts.Numeric)
	for k, v := range mla {
		mlaNext[k] = v
	}
	mlaNext["nextn_predict_layers"] = n(1)
	// Main context excludes the trailing NextN block: llama-model.cpp:2676-2681.
	test("MLA main context NextN", "deepseek2", mlaNext, "on", false, "196608")
	// Gemma4 first 2 of 3 blocks own caches; full SWA and equal widths make
	// the pattern immaterial to bytes: 2*(32*2+32*2)*1024*2 = 524288.
	// models/gemma4.cpp:7-10, llama-hparams.cpp:320-327, llama-kv-cache-iswa.cpp:74-81.
	test("shared KV full cache equal widths", "gemma4", map[string]gguffacts.Numeric{"block_count": n(3), "attention.shared_kv_layers": n(1), "attention.sliding_window": n(256), "attention.sliding_window_pattern": {Type: 7, Array: []uint64{1, 0, 1}}, "attention.key_length_swa": n(32), "attention.value_length_swa": n(32)}, "on", true, "524288")
}

func TestModelSettingsEstimateSpecialCacheUnknowns(t *testing.T) {
	for _, tc := range []struct {
		arch, field, reason string
		value               uint64
	}{
		{"llama", "attention.shared_kv_layers", "shared-KV", 1},
		{"llama", "nextn_predict_layers", "NextN", 1},
		{"deepseek2", "attention.kv_lora_rank", "MLA", 64},
		{"deepseek3", "attention.kv_lora_rank", "architecture", 64},
		{"gemma4", "attention.shared_kv_layers", "shared-KV", 2},
	} {
		t.Run(tc.arch+" "+tc.field, func(t *testing.T) {
			in := estimateFixture(tc.arch)
			in.Facts.Shards[0].Numeric[tc.arch+"."+tc.field] = gguffacts.Numeric{Type: 10, Bits: tc.value}
			c := estimateTestComponent(t, estimateFixtureResult(t, in), "attention-cache")
			if c.Status != "unknown" || !strings.Contains(c.Reason, tc.reason) {
				t.Fatalf("cache = %+v, want named %s unknown", c, tc.reason)
			}
		})
	}
}

func TestModelSettingsEstimateClass(t *testing.T) {
	for _, tc := range []struct{ arch, want string }{
		{"llama", "dense"}, {"qwen2", "dense"}, {"qwen3moe", "moe"},
		{"deepseek2", "mla"}, {"deepseek32", "mla"}, {"gemma4", "swa"},
		{"lfm2moe", "hybrid"}, {"qwen35", "hybrid"}, {"unmapped", "unmapped"},
	} {
		t.Run(tc.arch, func(t *testing.T) {
			m, reason := modelEstimateAdapt(context.Background(), estimateFixture(tc.arch).Facts)
			if reason != "" {
				t.Fatal(reason)
			}
			if got := modelEstimateClass(m); got != tc.want {
				t.Fatalf("class = %q, want %q", got, tc.want)
			}
		})
	}
	t.Run("window takes precedence over experts", func(t *testing.T) {
		in := estimateFixture("qwen3moe")
		in.Facts.Shards[0].Numeric["qwen3moe.attention.sliding_window"] = gguffacts.Numeric{Type: 10, Bits: 256}
		m, reason := modelEstimateAdapt(context.Background(), in.Facts)
		if reason != "" {
			t.Fatal(reason)
		}
		if got := modelEstimateClass(m); got != "swa" {
			t.Fatalf("class = %q, want swa", got)
		}
	})
}

func TestModelSettingsEstimateClassSelectsBand(t *testing.T) {
	key := modelEstimateCalibrationKey{"weights", "moe", "cpu"}
	modelEstimateCalibrations[key] = modelEstimateCalibration{LowNum: 1, LowDen: 2, HighNum: 2, HighDen: 1}
	t.Cleanup(func() { delete(modelEstimateCalibrations, key) })
	moe := estimateTestComponent(t, estimateFixtureResult(t, estimateFixture("qwen3moe")), "weights")
	if moe.Basis != "calibrated" || moe.LowBytes != "163968" || moe.HighBytes != "655872" {
		t.Fatalf("MoE band = %+v", moe)
	}
	dense := estimateTestComponent(t, estimateFixtureResult(t, estimateFixture("llama")), "weights")
	if dense.Basis != "derived" || dense.LowBytes != dense.Bytes || dense.HighBytes != dense.Bytes {
		t.Fatalf("MoE band leaked to dense = %+v", dense)
	}
}

func TestModelSettingsEstimateMLARejectsUnconfirmedLayout(t *testing.T) {
	for _, tc := range []struct {
		name, field string
		value       uint64
	}{
		{"query width used as stored key", "attention.key_length", 48},
		{"multiple latent heads", "attention.head_count_kv", 2},
		{"missing paired MLA width", "attention.value_length_mla", 0},
		{"missing rotary width", "rope.dimension_count", 0},
		{"all NextN context", "nextn_predict_layers", 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := estimateFixture("deepseek2")
			for k, v := range map[string]uint64{"attention.head_count_kv": 1, "attention.key_length": 96, "attention.value_length": 64, "attention.key_length_mla": 48, "attention.value_length_mla": 32, "attention.kv_lora_rank": 64, "rope.dimension_count": 32} {
				in.Facts.Shards[0].Numeric["deepseek2."+k] = gguffacts.Numeric{Type: 10, Bits: v}
			}
			in.Facts.Shards[0].Numeric["deepseek2."+tc.field] = gguffacts.Numeric{Type: 10, Bits: tc.value}
			c := estimateTestComponent(t, estimateFixtureResult(t, in), "attention-cache")
			if c.Status != "unknown" || c.Reason == "" {
				t.Fatalf("cache = %+v, want named unknown", c)
			}
		})
	}
}

func TestModelSettingsEstimateSpecialCacheNativeGuards(t *testing.T) {
	shared := func() modelEstimateInput {
		in := estimateFixture("gemma4")
		for field, value := range map[string]uint64{"block_count": 3, "attention.shared_kv_layers": 1, "attention.sliding_window": 256, "attention.key_length_swa": 32, "attention.value_length_swa": 32} {
			in.Facts.Shards[0].Numeric["gemma4."+field] = gguffacts.Numeric{Type: 4, Bits: value}
		}
		in.Facts.Shards[0].Numeric["gemma4.attention.sliding_window_pattern"] = gguffacts.Numeric{Type: 7, Array: []uint64{1, 0, 1}}
		full, fa := true, "on"
		in.SWAFull, in.Allocation.Options.FlashAttention = &full, &fa
		return in
	}
	for _, tc := range []struct {
		name, fa, status, bytes, reason string
	}{
		{"FA on variable heads", "on", "estimated", "393216", ""},
		{"FA off variable heads", "off", "unknown", "", "transposed V"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := shared()
			in.Facts.Shards[0].Numeric["gemma4.attention.head_count_kv"] = gguffacts.Numeric{Type: 4, Array: []uint64{1, 2, 1}}
			in.Allocation.Options.FlashAttention = &tc.fa
			c := estimateTestComponent(t, estimateFixtureResult(t, in), "attention-cache")
			// 7fe450e1 llama-kv-cache.cpp:209-234 pads every transposed V
			// to the global maximum width, even when K widths vary by layer.
			if c.Status != tc.status || c.Bytes != tc.bytes || !strings.Contains(c.Reason, tc.reason) {
				t.Fatalf("cache=%+v, want %s %s (%s)", c, tc.status, tc.bytes, tc.reason)
			}
		})
	}
	t.Run("explicit recurrent NextN mask", func(t *testing.T) {
		in := estimateFixture("qwen35")
		for field, value := range map[string]uint64{"block_count": 4, "nextn_predict_layers": 1, "full_attention_interval": 2} {
			in.Facts.Shards[0].Numeric["qwen35."+field] = gguffacts.Numeric{Type: 4, Bits: value}
		}
		in.Facts.Shards[0].Numeric["qwen35.attention.recurrent_layers"] = gguffacts.Numeric{Type: 7, Array: []uint64{0, 1, 0, 0}}
		c := estimateTestComponent(t, estimateFixtureResult(t, in), "attention-cache")
		// models/qwen35.cpp:17-21 gives this mask precedence over interval.
		if c.Status != "unknown" || !strings.Contains(c.Reason, "explicit recurrent") {
			t.Fatalf("cache=%+v, want named explicit-mask unknown", c)
		}
	})
	for _, tc := range []struct {
		name, status, bytes string
		latent, rope        uint64
	}{
		{"aligned MLA precision", "estimated", "208896", 64, 32},
		{"MLA V precision rejected", "unknown", "", 48, 16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := estimateFixture("deepseek2")
			for field, value := range map[string]uint64{"attention.head_count_kv": 1, "attention.key_length": tc.latent + tc.rope, "attention.value_length": tc.latent, "attention.key_length_mla": tc.rope + 16, "attention.value_length_mla": 32, "attention.kv_lora_rank": tc.latent, "rope.dimension_count": tc.rope} {
				in.Facts.Shards[0].Numeric["deepseek2."+field] = gguffacts.Numeric{Type: 4, Bits: value}
			}
			fa, quant := "on", "q8_0"
			in.Allocation.Options.FlashAttention = &fa
			in.Allocation.Options.CacheTypeK, in.Allocation.Options.CacheTypeV = &quant, &quant
			c := estimateTestComponent(t, estimateFixtureResult(t, in), "attention-cache")
			// llama-context.cpp:3803-3809 still validates V precision for MLA.
			if c.Status != tc.status || c.Bytes != tc.bytes {
				t.Fatalf("cache=%+v, want %s %s", c, tc.status, tc.bytes)
			}
		})
	}
	t.Run("floating shared pattern", func(t *testing.T) {
		in := shared()
		in.Facts.Shards[0].Numeric["gemma4.attention.sliding_window_pattern"] = gguffacts.Numeric{Type: 6, Array: []uint64{1, 0, 1}}
		c := estimateTestComponent(t, estimateFixtureResult(t, in), "attention-cache")
		// llama-model-loader.cpp:365-380 rejects float arrays for this field.
		if c.Status != "unknown" || !strings.Contains(c.Reason, "pattern") {
			t.Fatalf("cache=%+v, want named pattern unknown", c)
		}
	})
	t.Run("shared quantized head precision", func(t *testing.T) {
		in := shared()
		for _, field := range []string{"attention.key_length", "attention.value_length", "attention.key_length_swa", "attention.value_length_swa"} {
			in.Facts.Shards[0].Numeric["gemma4."+field] = gguffacts.Numeric{Type: 4, Bits: 16}
		}
		quant := "q8_0"
		in.Allocation.Options.CacheTypeK, in.Allocation.Options.CacheTypeV = &quant, &quant
		c := estimateTestComponent(t, estimateFixtureResult(t, in), "attention-cache")
		// Native precision validation is per head, not the combined GQA row.
		if c.Status != "unknown" || !strings.Contains(c.Reason, "precision") {
			t.Fatalf("cache=%+v, want named precision unknown", c)
		}
	})
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

func TestModelSettingsEstimateSWANativeLayerPatterns(t *testing.T) {
	for _, arch := range []string{"gemma3", "gpt-oss", "llama4"} {
		t.Run(arch, func(t *testing.T) {
			in := estimateFixture(arch)
			in.Allocation.PerSlotContext, in.Allocation.TotalContext = 16384, 16384
			in.Facts.Shards[0].Numeric[arch+".attention.sliding_window"] = gguffacts.Numeric{Type: 4, Bits: 256}
			in.Facts.Shards[0].Numeric[arch+".attention.sliding_window_pattern"] = gguffacts.Numeric{Type: 7, Array: []uint64{0, 1}}
			// Native arrays mark global then SWA. Each F16 K+V cell is
			// 2 * 32 * 2 * 2 = 256 bytes. Llama4 overrides the window to
			// 8192, so its chunked layer has pad(8192+32,256)=8448 cells.
			want := "4325376" // (16384 + 512) * 256
			if arch == "llama4" {
				want = "6356992" // (16384 + 8448) * 256
			}
			assertEstimateBytes(t, estimateFixtureResult(t, in), "attention-cache", "estimated", want)
			if arch == "gpt-oss" {
				// Sinks are learned F32 per-head weights, not extra KV cells:
				// models/openai-moe.cpp:42 and llama-kv-cache.cpp:229-234.
				in.Facts.Shards[0].Tensors = append(in.Facts.Shards[0].Tensors, gguffacts.Tensor{Name: "blk.0.attn_sinks.weight", Type: 0, Dimensions: []uint64{4}, Bytes: 16})
				out := estimateFixtureResult(t, in)
				assertEstimateBytes(t, out, "weights", "estimated", "327952")
				assertEstimateBytes(t, out, "attention-cache", "estimated", want)
			}
		})
	}
}

func TestModelSettingsEstimateSWANativePatternCountercases(t *testing.T) {
	for _, pattern := range []gguffacts.Numeric{
		{Type: 6, Array: []uint64{0, 1}}, // float array is not a native bool/integer array
		{Type: 7, Array: []uint64{1}},
		{Type: 4, Array: []uint64{0, 2}},
		{Type: 4, Array: []uint64{0, 0}},
		{Type: 4, Bits: 1}, // no SWA layers would fail the native assertion
	} {
		in := estimateFixture("gemma3")
		in.Facts.Shards[0].Numeric["gemma3.attention.sliding_window"] = gguffacts.Numeric{Type: 4, Bits: 256}
		in.Facts.Shards[0].Numeric["gemma3.attention.sliding_window_pattern"] = pattern
		assertEstimateBytes(t, estimateFixtureResult(t, in), "attention-cache", "unknown", "")
	}
	in := estimateFixture("gemma3")
	// The pin has a scalar n_swa, not per-layer window lengths. Do not
	// reinterpret a window array as a pattern or use parser defaults.
	in.Facts.Shards[0].Numeric["gemma3.attention.sliding_window"] = gguffacts.Numeric{Type: 4, Array: []uint64{256, 512}}
	assertEstimateBytes(t, estimateFixtureResult(t, in), "attention-cache", "unknown", "")
}

func TestModelSettingsEstimateSWANativeArchitectureGuards(t *testing.T) {
	t.Run("noncanonical architecture", func(t *testing.T) {
		in := estimateFixture("openai-moe")
		in.Facts.Shards[0].Numeric["openai-moe.attention.sliding_window"] = gguffacts.Numeric{Type: 4, Bits: 256}
		in.Facts.Shards[0].Numeric["openai-moe.attention.sliding_window_pattern"] = gguffacts.Numeric{Type: 7, Array: []uint64{0, 1}}
		assertEstimateBytes(t, estimateFixtureResult(t, in), "attention-cache", "unknown", "")
	})
	for _, tc := range []struct {
		name    string
		present bool
	}{
		{"GPT-OSS missing window", false},
		{"GPT-OSS zero window", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := estimateFixture("gpt-oss")
			if tc.present {
				in.Facts.Shards[0].Numeric["gpt-oss.attention.sliding_window"] = gguffacts.Numeric{Type: 4, Bits: 0}
			}
			c := estimateTestComponent(t, estimateFixtureResult(t, in), "attention-cache")
			if c.Status != "unknown" || c.Reason == "" {
				t.Fatalf("cache=%+v, want named window unknown", c)
			}
		})
	}
	t.Run("LFM2-MoE does not select SWA", func(t *testing.T) {
		in := estimateFixture("lfm2moe")
		in.Facts.Shards[0].Numeric["lfm2moe.attention.head_count_kv"] = gguffacts.Numeric{Type: 4, Array: []uint64{0, 2}}
		in.Facts.Shards[0].Numeric["lfm2moe.attention.sliding_window"] = gguffacts.Numeric{Type: 4, Bits: 256}
		c := estimateTestComponent(t, estimateFixtureResult(t, in), "attention-cache")
		// models/lfm2moe.cpp does not enable SWA, unlike models/lfm2.cpp.
		if c.Status != "unknown" || !strings.Contains(c.Reason, "SWA") {
			t.Fatalf("cache=%+v, want named SWA unknown", c)
		}
	})
}

func TestModelSettingsEstimateSWANativeWidthsAndPlacement(t *testing.T) {
	in := estimateFixture("gemma3")
	in.Facts.Shards[0].Numeric["gemma3.attention.sliding_window"] = gguffacts.Numeric{Type: 4, Bits: 256}
	in.Facts.Shards[0].Numeric["gemma3.attention.sliding_window_pattern"] = gguffacts.Numeric{Type: 4, Array: []uint64{0, 1}}
	layers, offload := int32(2), true
	in.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: &layers}
	in.Allocation.Options.KVOffload = &offload
	out := estimateFixtureResult(t, in)
	var rows []modelconfig.EstimateComponent
	for _, c := range out.Components {
		if c.Name == "attention-cache" {
			rows = append(rows, c)
		}
	}
	// Layer 0 is global on host, layer 1 is SWA on device. Splitting by a
	// layer fraction would incorrectly charge 196608 bytes to each pool.
	if len(rows) != 2 || rows[0].DomainID != "host" || rows[0].Bytes != "262144" || rows[1].DomainID != "offload-unmapped" || rows[1].Bytes != "131072" {
		t.Fatalf("unequal native per-layer split: %+v", rows)
	}
	in.Allocation.GPU = modelconfig.GPU{Mode: "cpu"}
	in.Facts.Shards[0].Numeric["gemma3.attention.key_length_swa"] = gguffacts.Numeric{Type: 4, Bits: 64}
	in.Facts.Shards[0].Numeric["gemma3.attention.value_length_swa"] = gguffacts.Numeric{Type: 4, Bits: 96}
	fa := "on"
	in.Allocation.Options.FlashAttention = &fa
	// 1024*(32+32)*2*2 + 512*(64+96)*2*2 = 589824.
	assertEstimateBytes(t, estimateFixtureResult(t, in), "attention-cache", "estimated", "589824")
	fa = "off"
	assertEstimateBytes(t, estimateFixtureResult(t, in), "attention-cache", "unknown", "")
}

func TestModelSettingsEstimateCachePartialLayerSplit(t *testing.T) {
	for _, tc := range []struct {
		name                 string
		layers               int32
		offload              bool
		wantHost, wantDevice string
	}{
		{"output only", 1, true, "524288", "0"},
		{"last block", 2, true, "262144", "262144"},
		{"all blocks", 3, true, "0", "524288"},
		{"disabled KV offload", 2, false, "524288", "0"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := estimateFixture("llama")
			in.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: &tc.layers}
			in.Allocation.Options.KVOffload = &tc.offload
			out := estimateFixtureResult(t, in)
			amounts := map[string]string{"host": "0", "offload-unmapped": "0"}
			for _, c := range out.Components {
				if c.Name == "attention-cache" {
					if c.Basis != "derived" || c.LowBytes != c.Bytes || c.HighBytes != c.Bytes || c.DomainID == "" {
						t.Fatalf("cache placement/range: %+v", c)
					}
					amounts[c.DomainID] = c.Bytes
				}
			}
			if amounts["host"] != tc.wantHost || amounts["offload-unmapped"] != tc.wantDevice {
				t.Fatalf("cache amounts=%v, want %s/%s", amounts, tc.wantHost, tc.wantDevice)
			}
		})
	}
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
	assertEstimateBytes(t, partial, "attention-cache", "estimated", "262144")
	if c := estimateTestComponent(t, partial, "attention-cache"); c.DomainID != "host" {
		t.Fatalf("first layer cache belongs on host: %+v", c)
	}
	if partial.DeviceBytes != "426240" {
		t.Fatalf("last layer cache plus weights: %s", partial.DeviceBytes)
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
