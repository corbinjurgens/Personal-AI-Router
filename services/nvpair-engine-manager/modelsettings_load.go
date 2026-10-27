// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"slices"

	"nvpair-shared/modelconfig"
)

// Native spellings, aliases, negative switches and environment names belong to
// llamaServingOptions. This projection adds presentation and model ownership,
// never another native parser/flag table. Preset describes the future B compiler
// mapping; Supported describes the B field vocabulary, not apply readiness.
type modelSettingsOption struct {
	Native     llamaServingOption
	Descriptor modelconfig.OptionDescriptor
	Preset     string
}

// Deliberately explicit: adding an actual adapter option without an ownership
// decision fails coverage instead of silently inventing a supported control.
func modelSettingsNativeLabels() map[string]string {
	return map[string]string{
		"models-preset": "Generated model preset", "models-max": "Router model capacity",
		"ctx-size": "Tokens per request", "parallel": "Native slots",
		"kv-unified-per-slot": "Unified KV per-slot cap", "kv-unified": "Unified KV allocation",
		"n-predict": "Native output token limit", "gpu-layers": "GPU placement",
		"split-mode": "GPU split mode", "tensor-split": "GPU tensor proportions",
		"main-gpu": "Main native GPU", "device": "Native devices",
		"backend-sampling": "Engine-side sampling", "sleep-idle-seconds": "Unload after idle (seconds; -1 or positive)",
		"cache-ram": "Host prompt cache (MiB)", "fit": "Native automatic fitting",
		"cache-prompt": "Prompt caching", "cache-idle-slots": "Idle slot caching",
		"batch-size": "Logical prompt batch", "ubatch-size": "Physical prompt batch",
		"cont-batching": "Continuous batching", "cache-type-k": "K cache precision",
		"cache-type-v": "V cache precision", "swa-full": "Full sliding-window cache",
		"flash-attn": "Flash Attention", "context-shift": "Context shifting",
		"rope-scaling": "RoPE scaling mode", "rope-scale": "RoPE context scale",
		"rope-freq-base": "RoPE frequency base", "rope-freq-scale": "RoPE frequency scale",
		"yarn-orig-ctx": "YaRN original context", "yarn-ext-factor": "YaRN extension factor",
		"yarn-attn-factor": "YaRN attention factor", "yarn-beta-slow": "YaRN slow correction",
		"yarn-beta-fast": "YaRN fast correction", "override-kv": "Metadata overrides",
		"override-tensor": "Tensor placement overrides", "cpu-moe": "CPU expert placement",
		"n-cpu-moe": "CPU expert layers", "n-cpu-ffn": "CPU feed-forward layers",
		"kv-offload": "Offload KV cache", "op-offload": "Offload operations",
		"repack": "Weight repacking", "no-host": "Disable host buffers",
		"load-mode": "Model loading mode", "lazy-mode": "Lazy loading mode",
		"numa": "NUMA placement", "rpc": "Native RPC devices",
		"models-autoload": "Router automatic model loading", "warmup": "Native warmup",
		"model": "Resolved model artifacts", "model-draft": "Resolved draft artifacts",
		"alias": "Native model alias", "mmproj": "Multimodal projector path",
		"mmproj-url": "Multimodal projector URL", "mmproj-auto": "Automatic projector selection",
		"mmproj-offload": "Projector offload", "mmproj-device": "Projector device",
		"image-min-tokens": "Minimum image tokens", "image-max-tokens": "Maximum image tokens",
		"mtmd-batch-max-tokens": "Multimodal batch tokens", "lora": "LoRA adapters",
		"lora-scaled": "Scaled LoRA adapters", "control-vector": "Control vectors",
		"control-vector-scaled": "Scaled control vectors", "control-vector-layer-range": "Control vector layers",
		"models-dir": "Native model directory", "model-url": "Native model URL",
		"hf-repo": "Native repository", "hf-file": "Native repository file",
		"docker-repo": "Native container repository", "hf-repo-draft": "Native draft repository",
		"spec-draft-type-k": "Draft K cache precision", "spec-draft-type-v": "Draft V cache precision",
		"spec-draft-override-tensor": "Draft tensor overrides", "spec-draft-cpu-moe": "Draft CPU experts",
		"spec-draft-n-cpu-moe": "Draft CPU expert layers", "spec-draft-device": "Draft devices",
		"spec-draft-ngl": "Draft GPU layers", "spec-draft-backend-sampling": "Draft engine-side sampling",
		"spec-draft-n-max": "Maximum speculative draft tokens", "spec-type": "Speculation type",
		"spec-default": "Default speculation preset", "mtp": "Multi-token prediction preset",
		"dflash": "DFlash preset", "eagle3": "Eagle3 preset",
		"embd-gemma-default": "Gemma embedding preset", "fim-qwen-1.5b-default": "Qwen 1.5B fill-in-middle preset",
		"fim-qwen-3b-default": "Qwen 3B fill-in-middle preset", "fim-qwen-7b-default": "Qwen 7B fill-in-middle preset",
		"fim-qwen-7b-spec": "Qwen 7B speculative preset", "fim-qwen-14b-spec": "Qwen 14B speculative preset",
		"fim-qwen-30b-default": "Qwen 30B fill-in-middle preset", "gpt-oss-20b-default": "GPT OSS 20B preset",
		"gpt-oss-120b-default": "GPT OSS 120B preset", "vision-gemma-4b-default": "Gemma 4B vision preset",
		"vision-gemma-12b-default": "Gemma 12B vision preset", "fit-target": "Native fit memory target",
		"fit-ctx": "Native fit context target", "cache-reuse": "Prompt cache reuse",
		"ctx-checkpoints": "Context checkpoints", "checkpoint-min-step": "Minimum checkpoint spacing",
		"prompt-cache": "Prompt cache file", "prompt-cache-all": "Cache all prompts",
		"prompt-cache-ro": "Read-only prompt cache", "slot-save-path": "Slot save directory",
		"load-on-startup": "Startup model loading",
	}
}

func modelSettingsNumber(value string) *json.Number {
	n := json.Number(value)
	return &n
}

func modelSettingsInteger(field, label, minimum, defaultValue, defaultSource, effect string) modelconfig.OptionDescriptor {
	d := modelconfig.OptionDescriptor{Field: field, Label: label, Kind: "integer", Minimum: modelSettingsNumber(minimum), Maximum: modelSettingsNumber("2147483647"), Supported: true, Portable: true, Owner: "model-load", Phase: "B", DefaultSource: defaultSource, IdentityEffect: "native allocation", EstimateEffect: effect}
	if defaultValue != "" {
		d.Default = json.RawMessage(defaultValue)
	}
	return d
}

func llamaModelSettingsOptions() ([]modelSettingsOption, error) {
	labels := modelSettingsNativeLabels()
	options := make([]modelSettingsOption, 0, len(llamaServingOptions))
	for _, native := range llamaServingOptions {
		label, ok := labels[native.key]
		if !ok {
			return nil, fmt.Errorf("model options: missing metadata for %s", native.key)
		}
		d := modelconfig.OptionDescriptor{Field: "native." + native.key, Label: label, Kind: "string", Supported: false, Reason: "No typed override; requires a mapped native build and identity/estimate support.", DefaultSource: "inherited native/model (not inspected)", Owner: "model-load", Phase: "D", IdentityEffect: "native semantics", EstimateEffect: "unknown when selected"}
		if native.bool {
			d.Kind = "boolean"
		}
		if native.unsupported {
			d.Reason = "Native finite adapter rejects this control; typed override unavailable."
		}
		preset := "none: typed override refused"
		switch native.key {
		case "ctx-size":
			d = modelSettingsInteger("context", label, "1", "8192", "finite adapter inherited baseline (not Auto)", "KV/state/workspace")
			d.Maximum = modelSettingsNumber("2147483392")
			preset = "ctx-size = checked padded tokens per request * slots; non-unified KV"
		case "parallel":
			d = modelSettingsInteger("slots", label, "1", "1", "finite adapter inherited baseline", "KV/state/workspace")
			preset = "parallel = slots"
		case "gpu-layers":
			d = modelconfig.OptionDescriptor{Field: "gpu", Label: label, Kind: "enum", Values: []string{"native-default", "cpu", "layers"}, Supported: true, Portable: true, Owner: "model-load", Phase: "B", DefaultSource: "inherited native placement; unresolved", IdentityEffect: "placement", EstimateEffect: "host/device weights and KV"}
			preset = "gpu-layers = 0 for CPU, checked finite layers; inherited native-default must resolve; Auto unavailable"
		case "cache-type-k", "cache-type-v":
			field := "cacheTypeK"
			if native.key == "cache-type-v" {
				field = "cacheTypeV"
			}
			d = modelconfig.OptionDescriptor{Field: field, Label: label, Kind: "enum", Values: []string{"f16"}, Supported: true, Portable: true, Owner: "model-load", Phase: "B", DefaultSource: "inherited native precision; unresolved", IdentityEffect: "KV precision/quality", EstimateEffect: "block-aware KV row sizes"}
			preset = native.key + " = explicit precision; bf16/q8_0/q4_0 need validated build/architecture support"
		case "flash-attn":
			d = modelconfig.OptionDescriptor{Field: "flashAttention", Label: label, Kind: "enum", Values: []string{"inherit", "on", "off"}, Supported: true, Portable: true, Owner: "model-load", Phase: "B", DefaultSource: "inherited native policy; unresolved", IdentityEffect: "attention implementation", EstimateEffect: "KV/workspace support"}
			preset = "flash-attn = on/off when explicit; inherit emits no override; validate target combination"
		case "batch-size", "ubatch-size":
			field := "batchSize"
			if native.key == "ubatch-size" {
				field = "ubatchSize"
			}
			d = modelSettingsInteger(field, label, "1", "", "inherited native default; unresolved", "workspace; ubatch must not exceed batch")
			preset = native.key + " = explicit positive int32"
		case "kv-offload", "op-offload":
			field := "kvOffload"
			if native.key == "op-offload" {
				field = "opOffload"
			}
			d = modelconfig.OptionDescriptor{Field: field, Label: label, Kind: "boolean", Supported: true, Portable: true, Owner: "model-load", Phase: "B", DefaultSource: "inherited native policy; unresolved", IdentityEffect: "placement", EstimateEffect: "host/device allocation"}
			preset = native.key + " = explicit true/false"
		case "sleep-idle-seconds":
			d = modelSettingsInteger("sleepIdleSeconds", label, "-1", "300", "finite adapter inherited baseline", "retention; keep-loaded policy takes precedence")
			preset = "sleep-idle-seconds = -1 or positive int32 (zero invalid)"
		case "cache-ram":
			d = modelSettingsInteger("cacheRAM", label, "1", "512", "finite adapter inherited baseline (MiB)", "host prompt cache; not admission memory")
			preset = "cache-ram = explicit positive int32 MiB"
		case "models-preset", "models-max", "models-autoload", "fit", "fit-target", "fit-ctx", "load-on-startup":
			d.Owner, d.Phase = "manager/router", "B"
			d.Reason = "Manager-owned output/router control; no model-local override."
			d.IdentityEffect, d.EstimateEffect = "router startup", "router capacity and startup effects"
			if native.key == "fit" {
				d.Kind, d.Default, d.DefaultSource = "enum", json.RawMessage(`"off"`), "finite adapter enforced output"
				d.Values, preset = []string{"off"}, "fit = off; reviewed Auto resolves before compilation"
			} else if native.key == "load-on-startup" {
				d.Kind, d.Default, d.DefaultSource = "boolean", json.RawMessage(`false`), "finite adapter enforced output"
				preset = "load-on-startup = false"
			} else if native.key == "models-preset" {
				preset = "manager-generated immutable preset path"
			} else if native.key == "models-max" {
				preset = "preserve existing finite router capacity"
			}
		case "model", "model-draft", "alias", "models-dir", "model-url", "hf-repo", "hf-file", "docker-repo", "hf-repo-draft":
			d.Owner = "artifacts"
			d.IdentityEffect, d.EstimateEffect = "ordered artifact identity", "weights and companions"
			d.Reason = "Target resolver owns paths and exact identity; no portable path override."
			if native.key == "model" || native.key == "model-draft" {
				d.Phase, preset = "B", "target-resolved ordered model/draft shards; preserve supported draft declarations"
			}
		case "kv-unified", "kv-unified-per-slot":
			d.Reason = "Existing unified records remain readable; typed edits require D support."
			d.EstimateEffect = "KV sharing and slot arithmetic"
		case "n-predict":
			d.Owner, d.IdentityEffect, d.EstimateEffect = "request/native", "request semantics", "output work; not KV context"
		}
		if native.unsupported && d.Supported {
			d.Supported, d.Portable = false, false
			d.Reason = "Native finite adapter rejects this control; typed override unavailable."
		}
		if err := d.Validate(); err != nil {
			return nil, err
		}
		options = append(options, modelSettingsOption{Native: native, Descriptor: d, Preset: preset})
	}
	if err := validateModelSettingsOptionCoverage(llamaServingOptions, options); err != nil {
		return nil, err
	}
	return options, nil
}

// Compare against actual native entries, including every alias/env/negative flag
// and rejection bit. Tests also pass a projection with a removed actual row.
func validateModelSettingsOptionCoverage(native []llamaServingOption, options []modelSettingsOption) error {
	byKey := make(map[string]llamaServingOption, len(options))
	for _, option := range options {
		if _, duplicate := byKey[option.Native.key]; duplicate {
			return fmt.Errorf("model options: duplicate native key %s", option.Native.key)
		}
		byKey[option.Native.key] = option.Native
	}
	for _, want := range native {
		got, ok := byKey[want.key]
		if !ok || !slices.Equal(got.flags, want.flags) || !slices.Equal(got.negativeFlags, want.negativeFlags) || got.env != want.env || got.negEnv != want.negEnv || got.bool != want.bool || got.unsupported != want.unsupported {
			return fmt.Errorf("model options: missing or mismatched native option %s", want.key)
		}
	}
	if len(byKey) != len(native) {
		return fmt.Errorf("model options: unexpected native option")
	}
	return nil
}

// Controls outside the allocation parser have request/template ownership. Their
// descriptors add no native spelling table and never materialize recommendations.
func modelSettingsExtraOptions() []modelconfig.OptionDescriptor {
	options := []modelconfig.OptionDescriptor{
		modelSettingsInteger("threads", "Generation threads", "1", "", "inherited native default; unresolved", "CPU work/workspace; no core-count tuning"),
		modelSettingsInteger("threadsBatch", "Prompt-processing threads", "1", "", "inherited native threads default; unresolved", "CPU work/workspace"),
		modelSettingsInteger("gpu.layers", "Finite GPU layers (target facts required)", "0", "", "inherited native placement; unresolved", "host/device weights; target layer count bounds this value"),
	}
	for _, field := range []struct{ key, label string }{{"cacheTypeK.alternatives", "Additional K cache precisions"}, {"cacheTypeV.alternatives", "Additional V cache precisions"}} {
		options = append(options, modelconfig.OptionDescriptor{Field: field.key, Label: field.label, Kind: "enum", Values: []string{"bf16", "q8_0", "q4_0"}, Supported: false, Reason: "Validated native build/architecture support unavailable; quantized V also requires validated Flash Attention.", Owner: "model-load", Phase: "B conditional", DefaultSource: "inherited native precision; unresolved", IdentityEffect: "KV precision/quality", EstimateEffect: "block-aware KV sizes"})
	}
	for _, field := range []struct{ key, label string }{{"context.auto", "Auto context"}, {"gpu.auto", "Auto placement"}} {
		options = append(options, modelconfig.OptionDescriptor{Field: field.key, Label: field.label, Kind: "enum", Values: []string{"auto"}, Supported: false, Reason: "Reviewed Auto resolver and owner trade choices are pending; no defaults selected.", Owner: "model-load", Phase: "C pending O2", DefaultSource: "none selected", IdentityEffect: "frozen reviewed native resolution", EstimateEffect: "requires bounded facts and mapped target memory"})
	}
	for _, field := range []struct{ key, label, kind, min, max string }{
		{"temperature", "Temperature", "number", "0", ""},
		{"top_p", "Top P", "number", "0", "1"},
		{"top_k", "Top K", "integer", "0", "9007199254740991"},
		{"min_p", "Min P", "number", "0", "1"},
		{"repeat_penalty", "Repeat penalty (positive)", "number", "0", ""},
		{"frequency_penalty", "Frequency penalty", "number", "-2", "2"},
		{"presence_penalty", "Presence penalty", "number", "-2", "2"},
		{"seed", "Seed", "integer", "-9007199254740991", "9007199254740991"},
	} {
		d := modelconfig.OptionDescriptor{Field: "generation." + field.key, Label: field.label, Kind: field.kind, Minimum: modelSettingsNumber(field.min), Supported: true, Portable: true, Owner: "request-policy", Phase: "A", DefaultSource: "inherited native/model recommendation; absent remains absent", IdentityEffect: "request/qualification fingerprint", EstimateEffect: "no native load allocation"}
		if field.max != "" {
			d.Maximum = modelSettingsNumber(field.max)
		}
		options = append(options, d)
	}
	options = append(options, modelconfig.OptionDescriptor{Field: "requestOptionsMode", Label: "Generation preference mode", Kind: "enum", Values: []string{"defaults", "enforced"}, Supported: true, Portable: true, Owner: "request-policy", Phase: "A", DefaultSource: "existing policy", IdentityEffect: "request/qualification fingerprint", EstimateEffect: "no native load allocation"})
	for _, field := range []struct{ key, label string }{
		{"samplers", "Additional samplers"}, {"output", "Output/stop options"},
		{"jinja", "Jinja rendering"}, {"chatTemplate", "Chat template"}, {"chatTemplateFile", "Chat template file"},
		{"chatTemplateKwargs", "Template arguments / thinking"}, {"skipChatParsing", "Chat parsing"},
		{"reasoning", "Reasoning mode"}, {"reasoningFormat", "Reasoning format"}, {"reasoningEffort", "Reasoning effort"},
		{"reasoningBudget", "Reasoning token budget"}, {"reasoningBudgetMessage", "Reasoning budget message"},
		{"reasoningPreserve", "Preserve reasoning history"}, {"grammar", "Grammar / structured output"},
	} {
		options = append(options, modelconfig.OptionDescriptor{Field: field.key, Label: field.label, Kind: "string", Supported: false, Reason: "Inherited behavior only; typed override requires pinned request/native mapping and identity support.", Owner: "request/template", Phase: "D", DefaultSource: "embedded/native behavior; not inspected", IdentityEffect: "prompt/request semantics", EstimateEffect: "unknown extra token/work cost"})
	}
	return options
}
