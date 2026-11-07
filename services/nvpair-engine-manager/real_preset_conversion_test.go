// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"nvpair-shared/modelconfig"
)

func TestLlamaQ8PrecisionDescriptorConsistency(t *testing.T) {
	options, err := llamaModelSettingsOptions()
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"cacheTypeK", "cacheTypeV"} {
		t.Run(field, func(t *testing.T) {
			index := slices.IndexFunc(options, func(option modelSettingsOption) bool { return option.Descriptor.Field == field })
			if index < 0 || !slices.Contains(options[index].Descriptor.Values, "q8_0") {
				t.Fatal("main descriptor omitted supported q8_0 precision")
			}
			for _, descriptor := range modelSettingsExtraOptions() {
				if descriptor.Field == field+".alternatives" && slices.Contains(descriptor.Values, "q8_0") {
					t.Fatal("supported q8_0 precision also advertised as unsupported")
				}
			}
		})
	}
}

func TestLlamaZeroPromptCache(t *testing.T) {
	zero := int32(0)
	request := modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "cpu"}, CacheRAM: &zero}
	options, resolved, err := typedLlamaModelOptions("model", request, nil)
	if err != nil {
		t.Fatal(err)
	}
	if options["cache-ram"] != "0" || resolved.Options.CacheRAM == nil || *resolved.Options.CacheRAM != 0 {
		t.Fatalf("zero cache lost: options=%v resolved=%+v", options, resolved)
	}
	data, err := typedLlamaPreset(map[string]map[string]string{"model": options})
	if err != nil {
		t.Fatal(err)
	}
	sections, err := parseLlamaServingPreset(data)
	if err != nil {
		t.Fatal(err)
	}
	// The conversion owner removes PAIR-owned idle sleep before translating
	// the remaining native model options.
	delete(sections["model"], "sleep-idle-seconds")
	converted, err := convertTypedLlamaOptions("model", sections["model"])
	if err != nil {
		t.Fatal(err)
	}
	if converted.CacheRAM == nil || *converted.CacheRAM != 0 {
		t.Fatalf("conversion lost zero cache: %+v", converted)
	}
}

func TestLlamaConversionNativeDefaults(t *testing.T) {
	f := newModelLoadFixture(t)
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Replace(string(data), "[*]\n", "[*]\nflash-attn = auto\nmmproj-auto = false\n", 1)
	text += "cache-ram = 0\ntemp = 0.1\ntop-k = 50\ntop-p = 0.95\nrepeat-penalty = 1.05\nreasoning-budget = 512\nn-predict = 2048\nload-mode = none\nlazy-mode = off\ncache-type-k = q8_0\ncache-type-v = q8_0\n"
	writeTestFile(t, f.source, []byte(text))
	rt := convertModelLoadFixture(t, f)
	prepared, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, rt), rt, false)
	if err != nil {
		t.Fatal(err)
	}
	sections, err := parseLlamaServingPresetAllowing(prepared.Snapshot, modelLoadAllowedPresetKeys(prepared.ModelLoads))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"cache-ram": "0", "temp": "0.1", "top-k": "50", "top-p": "0.95", "repeat-penalty": "1.05", "reasoning-budget": "512", "n-predict": "2048", "load-mode": "none", "lazy-mode": "off", "cache-type-k": "q8_0", "cache-type-v": "q8_0", "mmproj-auto": "false"}
	for key, value := range want {
		if sections["medium-model"][key] != value {
			t.Fatalf("native default %s=%q, want %q", key, sections["medium-model"][key], value)
		}
	}
	for _, load := range rt.ModelLoads {
		if load.Requested.FlashAttention == nil || *load.Requested.FlashAttention != "inherit" {
			t.Fatalf("auto flash attention not inherited: %+v", load.Requested)
		}
	}
}

func TestLlamaConversionProjectorControlsStayRefused(t *testing.T) {
	for _, key := range []string{"mmproj", "mmproj-url", "mmproj-offload", "mmproj-device", "image-min-tokens", "image-max-tokens", "mtmd-batch-max-tokens"} {
		t.Run(key, func(t *testing.T) {
			if _, err := convertTypedLlamaOptions("model", map[string]string{key: "false"}); err == nil {
				t.Fatal("conversion accepted projector control as an extra")
			}
		})
	}
}

func TestModelAutoZeroPromptCacheEstimateInput(t *testing.T) {
	now := time.Now()
	request := autoTestRequest()
	zero := int32(0)
	request.CacheRAM = &zero
	calls := 0
	state := modelEstimateState{now: func() time.Time { return now }, run: func(ctx context.Context, input modelEstimateInput) modelconfig.Estimate {
		calls++
		if input.Allocation.Options.CacheRAM == nil || *input.Allocation.Options.CacheRAM != 0 {
			t.Fatalf("Auto candidate lost zero prompt cache: %+v", input.Allocation.Options)
		}
		return autoTestEstimate(ctx, input)
	}}
	proposal, err := autoTestResolveInput(context.Background(), "model", request, autoTestInput(), modelAutoOperation{Memory: &autoTestMemoryOwner{memory: autoTestMemory(now)}, LayerCosts: autoTestLayerCosts{}}, &state)
	if err != nil {
		t.Fatal(err)
	}
	if calls == 0 || proposal.Resolved == nil || !reflect.DeepEqual(proposal.Resolved.Options.CacheRAM, &zero) {
		t.Fatalf("Auto lost zero cache: calls=%d proposal=%+v", calls, proposal)
	}
}

func TestLlamaNegativePromptCacheRefused(t *testing.T) {
	negative := int32(-1)
	if err := (modelconfig.RequestedLoad{CacheRAM: &negative}).Validate(); err == nil {
		t.Fatal("typed load accepted negative cache RAM")
	}
	if _, err := parseLlamaServingPreset([]byte("[model]\ncache-ram = -1\n")); err == nil {
		t.Fatal("preset accepted negative cache RAM")
	}
}

func TestModelEstimateZeroPromptCache(t *testing.T) {
	in := estimateFixture("llama")
	zero := int32(0)
	in.Allocation.Options.CacheRAM = &zero
	result := estimateFixtureResult(t, in)
	cache := estimateTestComponent(t, result, "prompt-cache")
	if cache.Status != "estimated" || cache.Bytes != "0" || cache.LowBytes != "0" || cache.HighBytes != "0" {
		t.Fatalf("disabled prompt cache was not priced as zero: %+v", cache)
	}
}

func TestLlamaConversionNativeGPUPlacement(t *testing.T) {
	t.Run("native auto fit", func(t *testing.T) {
		request, err := convertTypedLlamaOptions("model", map[string]string{"fit": "on", "gpu-layers": "auto"})
		if err != nil {
			t.Fatal(err)
		}
		layers := int32(4)
		options, _, err := typedLlamaModelOptions("model", request, &layers)
		if err != nil {
			t.Fatal(err)
		}
		if options["fit"] != "on" || options["gpu-layers"] != "auto" || !modelLoadUnpriced(request) {
			t.Fatalf("native fitting changed or acquired finite pricing: %v", options)
		}
	})
	t.Run("auto without fitting refuses", func(t *testing.T) {
		if _, err := convertTypedLlamaOptions("model", map[string]string{"fit": "off", "gpu-layers": "auto"}); err == nil {
			t.Fatal("native auto accepted without native fit")
		}
	})
	t.Run("all layers resolves finite", func(t *testing.T) {
		request, err := convertTypedLlamaOptions("model", map[string]string{"gpu-layers": "all"})
		if err != nil {
			t.Fatal(err)
		}
		layers := int32(4)
		options, _, err := typedLlamaModelOptions("model", request, &layers)
		if err != nil {
			t.Fatal(err)
		}
		if options["gpu-layers"] != "5" {
			t.Fatalf("all layers lost output layer: %v", options)
		}
	})
}

func TestLlamaConversionZeroCacheEstimateInput(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	zero := int32(0)
	rt.ModelLoads[0].Requested.CacheRAM = &zero
	var estimates map[string]modelEstimateInput
	prepared, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, rt), rt, false, &estimates)
	if err != nil {
		t.Fatal(err)
	}
	profile := prepared.ModelLoads[0].ProfileKey
	input := estimates[profile]
	if input.Allocation == nil || input.Allocation.Options.CacheRAM == nil || *input.Allocation.Options.CacheRAM != 0 {
		t.Fatalf("compiler dropped zero cache from estimate: %+v", input.Allocation)
	}
}
