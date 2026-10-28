// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"

	"nvpair-shared/engines"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/secretenv"
)

// Merge before filtering, matching exec's last-assignment semantics while
// supplying the metadata helper a unique, canonical effective environment.
func modelEffectiveEnvironment(overrides map[string]string) []string {
	values := map[string]string{}
	for _, assignment := range os.Environ() {
		k, v, ok := strings.Cut(assignment, "=")
		if ok {
			values[environmentKey(k)] = v
		}
	}
	for k, v := range overrides {
		values[environmentKey(k)] = v
	}
	return secretenv.Remove(literalAssignments(values))
}

func modelNativeEnvironment(root string, overrides map[string]string) ([]string, error) {
	environment := modelEffectiveEnvironment(overrides)
	for _, assignment := range environment {
		key, value, _ := strings.Cut(assignment, "=")
		if key != "LD_LIBRARY_PATH" || value == "" {
			continue
		}
		paths := filepath.SplitList(value)
		if len(paths) != 1 || paths[0] == "" {
			return nil, servingError("Reviewed Auto requires native library search confined to its selected build.")
		}
		resolved, err := filepath.EvalSymlinks(paths[0])
		if err != nil || resolved != root {
			return nil, servingError("Reviewed Auto native library search escapes its selected build.")
		}
	}
	return environment, nil
}

func modelReviewedUUID(loads []modelconfig.ModelLoad) (string, error) {
	uuid := ""
	for _, load := range loads {
		r := load.Resolved
		if r == nil || !strings.HasPrefix(r.ResolverRevision, "auto/") || r.GPU.Mode == "cpu" || r.GPU.Layers != nil && *r.GPU.Layers == 0 {
			continue
		}
		if r.DeviceBinding == nil || r.DeviceBinding.Validate() != nil {
			return "", servingError("Reviewed Auto GPU requires an exact physical binding.")
		}
		if uuid != "" && uuid != r.DeviceBinding.DeviceID {
			return "", servingError("GPU loads must share one reviewed UUID.")
		}
		uuid = r.DeviceBinding.DeviceID
	}
	if uuid != "" {
		for _, load := range loads {
			r := load.Resolved
			if r == nil || r.GPU.Mode == "cpu" || r.GPU.Layers != nil && *r.GPU.Layers == 0 {
				continue
			}
			if r.DeviceBinding == nil || r.DeviceBinding.Validate() != nil || r.DeviceBinding.DeviceID != uuid {
				return "", servingError("Every GPU load in the shared process must bind the same full UUID.")
			}
		}
	}
	return uuid, nil
}

func modelConstrainCommand(command launchCommand, loads []modelconfig.ModelLoad) (launchCommand, error) {
	uuid, err := modelReviewedUUID(loads)
	if err != nil || uuid == "" {
		return command, err
	}
	for _, assignment := range modelEffectiveEnvironment(command.Env) {
		key, value, _ := strings.Cut(assignment, "=")
		if environmentKey(key) == "CUDA_VISIBLE_DEVICES" && value != uuid {
			return command, servingError("CUDA visibility conflicts with the reviewed full UUID.")
		}
	}
	command.Env = maps.Clone(command.Env)
	if command.Env == nil {
		command.Env = map[string]string{}
	}
	command.Env["CUDA_VISIBLE_DEVICES"] = uuid
	effective, err := modelDeviceEnvironment(modelEffectiveEnvironment(command.Env), uuid)
	if err != nil {
		return command, servingError("Reviewed native environment is unsupported.")
	}
	for _, assignment := range effective {
		key, value, _ := strings.Cut(assignment, "=")
		if key == "CUDA_MPS_PIPE_DIRECTORY" {
			command.Env[key] = value
		}
	}
	return command, nil
}

// File-only source validation, also used during accepted replay and immediately
// before spawn. It never opens native libraries or enumerates devices.
func modelReviewedNativeSource(ctx context.Context, st *engineState, command launchCommand) (string, error) {
	runtime, err := modelDeviceRuntimeSnapshot(ctx, st, modelDevicesBuild)
	if err != nil {
		return "", servingError("Reviewed Auto native library source unavailable.")
	}
	entries, err := os.ReadDir(runtime.Root)
	if err != nil {
		return "", servingError("Reviewed Auto native library source unavailable.")
	}
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		if strings.HasPrefix(strings.ToLower(entry.Name()), "libcuda.so") || strings.EqualFold(entry.Name(), "nvcuda.dll") {
			return "", servingError("Reviewed Auto build root contains an unsupported driver library.")
		}
	}
	executable, err := servingFileBinding(command.Bin)
	if err != nil {
		return "", err
	}
	env, err := modelNativeEnvironment(runtime.Root, command.Env)
	if err != nil {
		return "", err
	}
	_, err = modelDeviceEnvironment(env, "")
	if err != nil {
		return "", err
	}
	return modelDeviceDigest(struct {
		Executable  *engines.ServingBinding
		Runtime     modelDeviceRuntime
		Environment []string
	}{executable, runtime, env})
}

// ServingFieldError deliberately contains field/model identifiers, never source
// paths, native source text or launch environment values.
type ServingFieldError struct {
	Field   string
	ModelID string
	Reason  string
}

func (e *ServingFieldError) Error() string {
	if e.ModelID != "" {
		return fmt.Sprintf("%s: model %q: %s", e.Field, e.ModelID, e.Reason)
	}
	return e.Field + ": " + e.Reason
}

func servingError(reason string) error {
	return &ServingFieldError{Field: "serving", Reason: reason}
}

type llamaServingLaunch struct {
	ModelLoads []modelconfig.ModelLoad
	Config     *engines.ServingConfig
	Envelopes  []engines.ServingEnvelope
	Snapshot   []byte
	Source     string
	Digest     string
	// Credential is the native API key of a Strata launch. It reaches the
	// server only through the child environment, never the snapshot, and is
	// never serialized.
	Credential string `json:"-"`
}

type llamaServingSnapshot struct {
	Prepared *llamaServingLaunch
	Path     string
	Release  func()
}

type llamaServingOption struct {
	key           string
	flags         []string
	negativeFlags []string
	env           string
	negEnv        string
	bool          bool
	unsupported   bool
}

// Envelope boundary derived from llama.cpp 7fe450e1 common/arg.cpp. Every
// allocation/placement/startup control is interpreted here or rejected, including
// the parser's LLAMA_ARG_NO_ compatibility names for paired boolean switches.
// Other native options pass through unchanged. Unsupported controls also fail
// in presets; local model/draft paths and disabled startup loading are exceptions.
// Batch sizes, K/V cache types, Flash Attention and KV/operation offload are
// accepted and surfaced as placement; memory bounds stay operator declarations.
var llamaServingOptions = []llamaServingOption{
	{key: "models-preset", flags: []string{"--models-preset"}, env: "LLAMA_ARG_MODELS_PRESET"},
	{key: "models-max", flags: []string{"--models-max"}, env: "LLAMA_ARG_MODELS_MAX"},
	{key: "ctx-size", flags: []string{"--ctx-size", "-c"}, env: "LLAMA_ARG_CTX_SIZE"},
	{key: "parallel", flags: []string{"--parallel", "-np"}, env: "LLAMA_ARG_N_PARALLEL"},
	{key: "kv-unified-per-slot", flags: []string{"--kv-unified-per-slot"}, env: "LLAMA_ARG_KV_UNIFIED_PER_SLOT"},
	{key: "kv-unified", flags: []string{"--kv-unified", "-kvu", "--no-kv-unified", "-no-kvu"}, negativeFlags: []string{"--no-kv-unified", "-no-kvu"}, env: "LLAMA_ARG_KV_UNIFIED", negEnv: "LLAMA_ARG_NO_KV_UNIFIED", bool: true},
	{key: "n-predict", flags: []string{"--predict", "--n-predict", "-n"}, env: "LLAMA_ARG_N_PREDICT"},
	{key: "gpu-layers", flags: []string{"--gpu-layers", "--n-gpu-layers", "-ngl"}, env: "LLAMA_ARG_N_GPU_LAYERS"},
	{key: "split-mode", flags: []string{"--split-mode", "-sm"}, env: "LLAMA_ARG_SPLIT_MODE"},
	{key: "tensor-split", flags: []string{"--tensor-split", "-ts"}, env: "LLAMA_ARG_TENSOR_SPLIT"},
	{key: "main-gpu", flags: []string{"--main-gpu", "-mg"}, env: "LLAMA_ARG_MAIN_GPU"},
	{key: "device", flags: []string{"--device", "-dev"}, env: "LLAMA_ARG_DEVICE"},
	{key: "backend-sampling", flags: []string{"-bs", "--backend-sampling"}, env: "LLAMA_ARG_BACKEND_SAMPLING", bool: true, unsupported: true},
	{key: "sleep-idle-seconds", flags: []string{"--sleep-idle-seconds"}, env: "LLAMA_ARG_SLEEP_IDLE_SECONDS"},
	{key: "cache-ram", flags: []string{"--cache-ram", "-cram"}, env: "LLAMA_ARG_CACHE_RAM"},
	{key: "fit", flags: []string{"--fit", "-fit"}, env: "LLAMA_ARG_FIT"},
	{key: "cache-prompt", flags: []string{"--cache-prompt", "--no-cache-prompt"}, negativeFlags: []string{"--no-cache-prompt"}, env: "LLAMA_ARG_CACHE_PROMPT", negEnv: "LLAMA_ARG_NO_CACHE_PROMPT", bool: true},
	{key: "cache-idle-slots", flags: []string{"--cache-idle-slots", "--no-cache-idle-slots"}, negativeFlags: []string{"--no-cache-idle-slots"}, env: "LLAMA_ARG_CACHE_IDLE_SLOTS", negEnv: "LLAMA_ARG_NO_CACHE_IDLE_SLOTS", bool: true},
	{key: "batch-size", flags: []string{"-b", "--batch-size"}, env: "LLAMA_ARG_BATCH"},
	{key: "ubatch-size", flags: []string{"-ub", "--ubatch-size"}, env: "LLAMA_ARG_UBATCH"},
	{key: "cont-batching", flags: []string{"-cb", "--cont-batching", "-nocb", "--no-cont-batching"}, env: "LLAMA_ARG_CONT_BATCHING", negEnv: "LLAMA_ARG_NO_CONT_BATCHING", bool: true, unsupported: true},
	{key: "cache-type-k", flags: []string{"-ctk", "--cache-type-k"}, env: "LLAMA_ARG_CACHE_TYPE_K"},
	{key: "cache-type-v", flags: []string{"-ctv", "--cache-type-v"}, env: "LLAMA_ARG_CACHE_TYPE_V"},
	{key: "swa-full", flags: []string{"--swa-full"}, env: "LLAMA_ARG_SWA_FULL", bool: true, unsupported: true},
	{key: "flash-attn", flags: []string{"-fa", "--flash-attn"}, env: "LLAMA_ARG_FLASH_ATTN"},
	{key: "context-shift", flags: []string{"--context-shift", "--no-context-shift"}, env: "LLAMA_ARG_CONTEXT_SHIFT", negEnv: "LLAMA_ARG_NO_CONTEXT_SHIFT", bool: true, unsupported: true},
	{key: "rope-scaling", flags: []string{"--rope-scaling"}, env: "LLAMA_ARG_ROPE_SCALING_TYPE", unsupported: true},
	{key: "rope-scale", flags: []string{"--rope-scale"}, env: "LLAMA_ARG_ROPE_SCALE", unsupported: true},
	{key: "rope-freq-base", flags: []string{"--rope-freq-base"}, env: "LLAMA_ARG_ROPE_FREQ_BASE", unsupported: true},
	{key: "rope-freq-scale", flags: []string{"--rope-freq-scale"}, env: "LLAMA_ARG_ROPE_FREQ_SCALE", unsupported: true},
	{key: "yarn-orig-ctx", flags: []string{"--yarn-orig-ctx"}, env: "LLAMA_ARG_YARN_ORIG_CTX", unsupported: true},
	{key: "yarn-ext-factor", flags: []string{"--yarn-ext-factor"}, env: "LLAMA_ARG_YARN_EXT_FACTOR", unsupported: true},
	{key: "yarn-attn-factor", flags: []string{"--yarn-attn-factor"}, env: "LLAMA_ARG_YARN_ATTN_FACTOR", unsupported: true},
	{key: "yarn-beta-slow", flags: []string{"--yarn-beta-slow"}, env: "LLAMA_ARG_YARN_BETA_SLOW", unsupported: true},
	{key: "yarn-beta-fast", flags: []string{"--yarn-beta-fast"}, env: "LLAMA_ARG_YARN_BETA_FAST", unsupported: true},
	{key: "override-kv", flags: []string{"--override-kv"}, unsupported: true},
	{key: "override-tensor", flags: []string{"-ot", "--override-tensor"}, env: "LLAMA_ARG_OVERRIDE_TENSOR", unsupported: true},
	{key: "cpu-moe", flags: []string{"-cmoe", "--cpu-moe"}, env: "LLAMA_ARG_CPU_MOE", bool: true, unsupported: true},
	{key: "n-cpu-moe", flags: []string{"-ncmoe", "--n-cpu-moe"}, env: "LLAMA_ARG_N_CPU_MOE", unsupported: true},
	{key: "n-cpu-ffn", flags: []string{"-ncffn", "--n-cpu-ffn"}, env: "LLAMA_ARG_N_CPU_FFN", unsupported: true},
	{key: "kv-offload", flags: []string{"-kvo", "--kv-offload", "-nkvo", "--no-kv-offload"}, negativeFlags: []string{"-nkvo", "--no-kv-offload"}, env: "LLAMA_ARG_KV_OFFLOAD", negEnv: "LLAMA_ARG_NO_KV_OFFLOAD", bool: true},
	{key: "op-offload", flags: []string{"--op-offload", "--no-op-offload"}, negativeFlags: []string{"--no-op-offload"}, bool: true},
	{key: "repack", flags: []string{"--repack", "-nr", "--no-repack"}, env: "LLAMA_ARG_REPACK", negEnv: "LLAMA_ARG_NO_REPACK", bool: true, unsupported: true},
	{key: "no-host", flags: []string{"--no-host"}, env: "LLAMA_ARG_NO_HOST", bool: true, unsupported: true},
	{key: "load-mode", flags: []string{"-lm", "--load-mode"}, env: "LLAMA_ARG_LOAD_MODE", unsupported: true},
	{key: "lazy-mode", flags: []string{"-lzm", "--lazy-mode"}, env: "LLAMA_ARG_LAZY_MODE", unsupported: true},
	{key: "numa", flags: []string{"--numa"}, env: "LLAMA_ARG_NUMA", unsupported: true},
	{key: "rpc", flags: []string{"--rpc"}, env: "LLAMA_ARG_RPC", unsupported: true},
	{key: "models-autoload", flags: []string{"--models-autoload", "--no-models-autoload"}, env: "LLAMA_ARG_MODELS_AUTOLOAD", negEnv: "LLAMA_ARG_NO_MODELS_AUTOLOAD", bool: true, unsupported: true},
	{key: "warmup", flags: []string{"--warmup", "--no-warmup"}, bool: true, unsupported: true},
	{key: "model", flags: []string{"--model", "-m"}, env: "LLAMA_ARG_MODEL", unsupported: true},
	{key: "model-draft", flags: []string{"--spec-draft-model", "--model-draft", "-md"}, env: "LLAMA_ARG_SPEC_DRAFT_MODEL", unsupported: true},
	{key: "alias", flags: []string{"--alias", "-a"}, env: "LLAMA_ARG_ALIAS", unsupported: true},
	{key: "mmproj", flags: []string{"--mmproj", "-mm"}, env: "LLAMA_ARG_MMPROJ", unsupported: true},
	{key: "mmproj-url", flags: []string{"--mmproj-url", "-mmu"}, env: "LLAMA_ARG_MMPROJ_URL", unsupported: true},
	{key: "mmproj-auto", flags: []string{"--mmproj-auto", "--no-mmproj", "--no-mmproj-auto"}, env: "LLAMA_ARG_MMPROJ_AUTO", negEnv: "LLAMA_ARG_NO_MMPROJ_AUTO", bool: true, unsupported: true},
	{key: "mmproj-offload", flags: []string{"--mmproj-offload", "--no-mmproj-offload"}, env: "LLAMA_ARG_MMPROJ_OFFLOAD", negEnv: "LLAMA_ARG_NO_MMPROJ_OFFLOAD", bool: true, unsupported: true},
	{key: "mmproj-device", flags: []string{"-mmdev", "--mmproj-device"}, env: "MTMD_BACKEND_DEVICE", unsupported: true},
	{key: "image-min-tokens", flags: []string{"--image-min-tokens"}, env: "LLAMA_ARG_IMAGE_MIN_TOKENS", unsupported: true},
	{key: "image-max-tokens", flags: []string{"--image-max-tokens"}, env: "LLAMA_ARG_IMAGE_MAX_TOKENS", unsupported: true},
	{key: "mtmd-batch-max-tokens", flags: []string{"--mtmd-batch-max-tokens"}, env: "LLAMA_ARG_MTMD_BATCH_MAX_TOKENS", unsupported: true},
	{key: "lora", flags: []string{"--lora"}, unsupported: true},
	{key: "lora-scaled", flags: []string{"--lora-scaled"}, unsupported: true},
	{key: "control-vector", flags: []string{"--control-vector"}, unsupported: true},
	{key: "control-vector-scaled", flags: []string{"--control-vector-scaled"}, unsupported: true},
	{key: "control-vector-layer-range", flags: []string{"--control-vector-layer-range"}, unsupported: true},
	{key: "models-dir", flags: []string{"--models-dir"}, env: "LLAMA_ARG_MODELS_DIR", unsupported: true},
	{key: "model-url", flags: []string{"--model-url", "-mu"}, env: "LLAMA_ARG_MODEL_URL", unsupported: true},
	{key: "hf-repo", flags: []string{"--hf-repo", "-hf", "-hfr"}, env: "LLAMA_ARG_HF_REPO", unsupported: true},
	{key: "hf-file", flags: []string{"--hf-file", "-hff"}, env: "LLAMA_ARG_HF_FILE", unsupported: true},
	{key: "docker-repo", flags: []string{"--docker-repo", "-dr"}, env: "LLAMA_ARG_DOCKER_REPO", unsupported: true},
	{key: "hf-repo-draft", flags: []string{"--spec-draft-hf", "--hf-repo-draft", "-hfd", "-hfrd"}, env: "LLAMA_ARG_SPEC_DRAFT_HF_REPO", unsupported: true},
	{key: "spec-draft-type-k", flags: []string{"--spec-draft-type-k", "-ctkd", "--cache-type-k-draft"}, env: "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K", unsupported: true},
	{key: "spec-draft-type-v", flags: []string{"--spec-draft-type-v", "-ctvd", "--cache-type-v-draft"}, env: "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V", unsupported: true},
	{key: "spec-draft-override-tensor", flags: []string{"--spec-draft-override-tensor", "-otd", "--override-tensor-draft"}, unsupported: true},
	{key: "spec-draft-cpu-moe", flags: []string{"--spec-draft-cpu-moe", "-cmoed", "--cpu-moe-draft"}, env: "LLAMA_ARG_SPEC_DRAFT_CPU_MOE", bool: true, unsupported: true},
	{key: "spec-draft-n-cpu-moe", flags: []string{"--spec-draft-n-cpu-moe", "--spec-draft-ncmoe", "-ncmoed", "--n-cpu-moe-draft"}, env: "LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE", unsupported: true},
	{key: "spec-draft-device", flags: []string{"--spec-draft-device", "-devd", "--device-draft"}, unsupported: true},
	{key: "spec-draft-ngl", flags: []string{"--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"}, env: "LLAMA_ARG_N_GPU_LAYERS_DRAFT", unsupported: true},
	{key: "spec-draft-backend-sampling", flags: []string{"--spec-draft-backend-sampling", "--no-spec-draft-backend-sampling"}, env: "LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING", negEnv: "LLAMA_ARG_NO_SPEC_DRAFT_BACKEND_SAMPLING", bool: true, unsupported: true},
	{key: "spec-draft-n-max", flags: []string{"--spec-draft-n-max"}, env: "LLAMA_ARG_SPEC_DRAFT_N_MAX", unsupported: true},
	{key: "spec-type", flags: []string{"--spec-type"}, env: "LLAMA_ARG_SPEC_TYPE", unsupported: true},
	{key: "spec-default", flags: []string{"--spec-default"}, bool: true, unsupported: true},
	{key: "mtp", flags: []string{"--mtp"}, bool: true, unsupported: true},
	{key: "dflash", flags: []string{"--dflash"}, bool: true, unsupported: true},
	{key: "eagle3", flags: []string{"--eagle3"}, bool: true, unsupported: true},
	{key: "embd-gemma-default", flags: []string{"--embd-gemma-default"}, bool: true, unsupported: true},
	{key: "fim-qwen-1.5b-default", flags: []string{"--fim-qwen-1.5b-default"}, bool: true, unsupported: true},
	{key: "fim-qwen-3b-default", flags: []string{"--fim-qwen-3b-default"}, bool: true, unsupported: true},
	{key: "fim-qwen-7b-default", flags: []string{"--fim-qwen-7b-default"}, bool: true, unsupported: true},
	{key: "fim-qwen-7b-spec", flags: []string{"--fim-qwen-7b-spec"}, bool: true, unsupported: true},
	{key: "fim-qwen-14b-spec", flags: []string{"--fim-qwen-14b-spec"}, bool: true, unsupported: true},
	{key: "fim-qwen-30b-default", flags: []string{"--fim-qwen-30b-default"}, bool: true, unsupported: true},
	{key: "gpt-oss-20b-default", flags: []string{"--gpt-oss-20b-default"}, bool: true, unsupported: true},
	{key: "gpt-oss-120b-default", flags: []string{"--gpt-oss-120b-default"}, bool: true, unsupported: true},
	{key: "vision-gemma-4b-default", flags: []string{"--vision-gemma-4b-default"}, bool: true, unsupported: true},
	{key: "vision-gemma-12b-default", flags: []string{"--vision-gemma-12b-default"}, bool: true, unsupported: true},
	{key: "fit-target", flags: []string{"--fit-target", "-fitt"}, env: "LLAMA_ARG_FIT_TARGET", unsupported: true},
	{key: "fit-ctx", flags: []string{"--fit-ctx", "-fitc"}, env: "LLAMA_ARG_FIT_CTX", unsupported: true},
	{key: "cache-reuse", flags: []string{"--cache-reuse"}, env: "LLAMA_ARG_CACHE_REUSE", unsupported: true},
	{key: "ctx-checkpoints", flags: []string{"--ctx-checkpoints", "--swa-checkpoints", "-ctxcp"}, env: "LLAMA_ARG_CTX_CHECKPOINTS", unsupported: true},
	{key: "checkpoint-min-step", flags: []string{"--checkpoint-min-step", "-cms"}, env: "LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT", unsupported: true},
	{key: "prompt-cache", flags: []string{"--prompt-cache"}, unsupported: true},
	{key: "prompt-cache-all", flags: []string{"--prompt-cache-all"}, unsupported: true},
	{key: "prompt-cache-ro", flags: []string{"--prompt-cache-ro"}, unsupported: true},
	{key: "slot-save-path", flags: []string{"--slot-save-path"}, unsupported: true},
	{key: "load-on-startup", flags: []string{"--load-on-startup"}, env: "__PRESET_LOAD_ON_STARTUP", unsupported: true},
}

func llamaServingRuntime(st *engineState, rt Runtime) bool {
	return (rt.Serving != nil || len(rt.ModelLoads) != 0) && st.manifest.FamilyOrDefault() == "llamacpp"
}

// Remove only PAIR-owned defaults, before adding literal user arguments. The
// manifest stays unchanged so the unconfigured path remains byte-for-byte.
func llamaServingBase(rt Runtime) Runtime {
	if (rt.Serving == nil && len(rt.ModelLoads) == 0) || rt.EditableLaunch == nil {
		return rt
	}
	p := *rt.EditableLaunch
	p.FixedArgs = nil
	rt.EditableLaunch = &p
	rt.Args = nil
	for _, control := range p.Controls {
		if control.Value == "{server.host}" {
			rt.Args = append(rt.Args, control.flag(), "{host}")
		}
		if control.Value == "{server.port}" {
			rt.Args = append(rt.Args, control.flag(), "{port}")
		}
		if control.Value == "{cors.origins}" {
			rt.Args = append(rt.Args, control.flag(), "")
		}
	}
	return rt
}

func llamaPositive(value string) (int64, error) {
	n, err := strconv.ParseInt(value, 10, 32)
	if err != nil || n <= 0 || strconv.FormatInt(n, 10) != value {
		return 0, servingError("Use explicit positive native int32 values; automatic, zero and unlimited values are unsupported.")
	}
	return n, nil
}

func validateLlamaValue(key, value string) error {
	if key == "sleep-idle-seconds" && value == "-1" {
		return nil
	}
	switch key {
	case "ctx-size", "parallel", "kv-unified-per-slot", "n-predict", "models-max", "sleep-idle-seconds", "cache-ram", "batch-size", "ubatch-size":
		_, err := llamaPositive(value)
		return err
	case "kv-unified", "cache-prompt", "cache-idle-slots":
		_, err := llamaBool(value)
		return err
	case "fit":
		if value != "off" && value != "false" && value != "0" {
			return servingError("Automatic fitting cannot establish a finite allocation.")
		}
	}
	return nil
}

// Match both pinned arg.cpp long-option lookup paths; short flags stay exact.
func llamaLongOption(name string) string {
	if strings.HasPrefix(name, "--") {
		return strings.ReplaceAll(name, "_", "-")
	}
	return name
}

func llamaOptionKey(name string) string {
	name = llamaLongOption(name)
	for _, opt := range llamaServingOptions {
		if name == opt.key || name == opt.env || name == opt.negEnv || slices.Contains(opt.flags, name) || slices.Contains(opt.flags, "--"+name) {
			return opt.key
		}
	}
	switch name {
	case "temp", "top-k", "top-p", "min-p", "seed", "threads", "threads-batch":
		return name
	}
	return ""
}

func llamaBool(value string) (string, error) {
	switch value {
	case "true", "1", "on":
		return "true", nil
	case "false", "0", "off":
		return "false", nil
	}
	return "", servingError("Use an explicit supported boolean value.")
}

var llamaTagSuffix = regexp.MustCompile(`(?i)[-.]([A-Z0-9_]+)$`)

// Match pinned preset.cpp canonical_tag and its last-colon section rewrite.
func llamaCanonicalModelID(id string) string {
	colon := strings.LastIndexByte(id, ':')
	if colon < 0 {
		return id
	}
	tag := id[colon+1:]
	if suffix := llamaTagSuffix.FindStringSubmatch(tag); suffix != nil {
		tag = suffix[1]
	}
	upper := []byte(tag)
	for i, c := range upper {
		if c >= 'a' && c <= 'z' {
			upper[i] = c - ('a' - 'A')
		}
	}
	return id[:colon+1] + string(upper)
}

func validateLlamaModelID(id string) error {
	if err := nodepolicy.ValidateModelIdentifier(id); err != nil {
		return servingError(err.Error())
	}
	if id == "*" || id != strings.TrimSpace(id) || strings.ContainsAny(id, "[]#;\r\n\x00") || llamaCanonicalModelID(id) != id {
		return &ServingFieldError{Field: "serving.models.modelId", ModelID: id, Reason: "Use an exact canonical native preset model ID."}
	}
	return nil
}

func validateLlamaPadding(key string, value int64, model string) error {
	// Use int64 for the native rounding and check the result against int32,
	// including values whose padding would cross its maximum.
	padded := (value + 255) / 256 * 256
	if padded > math.MaxInt32 || padded != value {
		return &ServingFieldError{Field: "serving.models." + key, ModelID: model, Reason: "Native 256-token padding must preserve the approved allocation within int32 capacity."}
	}
	return nil
}

func validateLlamaPresetValue(key, value, model string) error {
	// Pinned INI values end before comments, newlines and trailing spaces/tabs;
	// leading spaces/tabs are consumed after '='. NUL cannot reach native argv.
	if strings.ContainsAny(value, "#;\r\n\x00") || strings.Trim(value, " \t") != value {
		return &ServingFieldError{Field: "serving.models." + key, ModelID: model, Reason: "Native preset cannot represent this value exactly."}
	}
	return nil
}

// Native child CLI values override its inherited environment. The router's
// base preset contains explicit CLI options only. Environment therefore fills
// settings missing from both model and global presets, rather than replacing
// a preset that is rendered as a child CLI option.
func llamaLaunchOptions(command launchCommand, policy *EditableLaunch) (map[string]string, map[string]string, error) {
	cli, env := map[string]string{}, map[string]string{}
	spellings := map[string]string{}
	accept := func(key, spelling, value string) error {
		if err := validateLlamaValue(key, value); err != nil {
			return err
		}
		if previous, exists := cli[key]; exists && (spellings[key] != spelling || previous != value) {
			return servingError("Duplicate native option aliases or conflicting native option values.")
		}
		cli[key], spellings[key] = value, spelling
		return nil
	}
	for i := 0; i < len(command.Args); i++ {
		token := command.Args[i]
		originalName, _, _ := strings.Cut(token, "=")
		if token == "--" {
			return nil, nil, servingError("Native option terminators are unsupported in serving mode.")
		}
		if policy != nil {
			control, _, last, err := policy.readControl(command.Args, i)
			if err != nil {
				return nil, nil, err
			}
			if control != nil {
				i = last
				continue
			}
		}
		for _, option := range llamaServingOptions {
			for _, flag := range option.flags {
				name, value, attached := strings.Cut(token, "=")
				name = llamaLongOption(name)
				if name != flag {
					// Native multi-letter short flags have exact spellings; reject
					// attached/bundled envelope flags instead of guessing arities.
					if !strings.HasPrefix(token, "--") && strings.HasPrefix(token, flag) && len(token) > len(flag) && strings.ContainsAny(token[len(flag):len(flag)+1], "0123456789+-=") {
						return nil, nil, servingError("Write native short envelope options separately.")
					}
					continue
				}
				if option.unsupported {
					return nil, nil, servingError("Unsupported envelope-affecting native launch option; select local model paths in the preset.")
				}
				if option.bool {
					if attached {
						return nil, nil, servingError("Native boolean switches do not take values.")
					}
					value = "true"
					if slices.Contains(option.negativeFlags, flag) {
						value = "false"
					}
				} else if !attached {
					i++
					if i >= len(command.Args) {
						return nil, nil, servingError("Native envelope option is missing its value.")
					}
					value = command.Args[i]
				}
				if err := accept(option.key, originalName, value); err != nil {
					return nil, nil, err
				}
				break
			}
		}
		// Unrelated native options retain their exact tokens in command.Args.
		// They are not projected into model presets or interpreted here.
	}
	effectiveEnv := map[string]string{}
	for _, assignment := range os.Environ() {
		key, value, _ := strings.Cut(assignment, "=")
		effectiveEnv[environmentKey(key)] = value
	}
	for key, value := range command.Env {
		effectiveEnv[environmentKey(key)] = value
	}
	for name, value := range effectiveEnv {
		key := ""
		for _, option := range llamaServingOptions {
			if (option.env != "" && name == option.env) || (option.negEnv != "" && name == option.negEnv) {
				if option.unsupported {
					return nil, nil, servingError("Unsupported envelope-affecting native environment option.")
				}
				key = option.key
				if name == option.negEnv {
					value = "false" // native compatibility variables act by presence
				}
				break
			}
		}
		if key == "" {
			continue
		}
		if _, exists := cli[key]; exists {
			return nil, nil, servingError("The same native envelope option is configured in CLI and environment.")
		}
		if _, exists := env[key]; exists {
			return nil, nil, servingError("Duplicate native envelope environment aliases.")
		}
		env[key] = value
		if err := validateLlamaValue(key, value); err != nil {
			return nil, nil, err
		}
	}
	return cli, env, nil
}

func parseLlamaServingPreset(data []byte) (map[string]map[string]string, error) {
	sections := map[string]map[string]string{}
	section := ""
	scanner := bufio.NewScanner(strings.NewReader(string(data)))
	scanner.Buffer(make([]byte, 4096), servingSmallFileLimit)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, ";") {
			continue
		}
		if strings.ContainsAny(line, ";#") {
			return nil, servingError("Inline preset comments are unsupported; use separate comment lines.")
		}
		if strings.ContainsAny(line, "$%{}\x00") {
			return nil, servingError("Preset substitution is unsupported.")
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			section = line[1 : len(line)-1]
			if section == "" || strings.ContainsAny(section, "[]\r\n") || sections[section] != nil {
				return nil, servingError("Preset sections must have unique exact model IDs.")
			}
			if section != "*" {
				if err := validateLlamaModelID(section); err != nil {
					return nil, err
				}
			}
			sections[section] = map[string]string{}
			continue
		}
		name, value, ok := strings.Cut(line, "=")
		// The pinned native parser skips the reserved version key in every
		// section and drops an otherwise-empty default section.
		if ok && strings.TrimSpace(name) == "version" {
			continue
		}
		if !ok || section == "" {
			return nil, servingError("Preset requires explicit sections and key=value options.")
		}
		name, value = strings.TrimSpace(name), strings.TrimSpace(value)
		key := llamaOptionKey(name)
		if key == "" || key == "models-preset" || key == "models-max" {
			field := "option"
			if len(name) <= 64 && regexp.MustCompile(`^[a-zA-Z0-9_-]+$`).MatchString(name) {
				field = name
			}
			return nil, &ServingFieldError{Field: "serving.models." + field, ModelID: section, Reason: "Unknown preset option, include or inheritance is unsupported."}
		}
		negative := false
		for _, option := range llamaServingOptions {
			if key == option.key && option.unsupported && key != "model" && key != "model-draft" && key != "load-on-startup" {
				return nil, &ServingFieldError{Field: "serving.models." + key, ModelID: section, Reason: "Unsupported envelope-affecting preset option."}
			}
			if key == option.key {
				negative = slices.Contains(option.negativeFlags, "--"+name)
			}
		}
		if _, duplicate := sections[section][key]; duplicate {
			return nil, servingError("Duplicate preset aliases or options are unsupported.")
		}
		if key == "kv-unified" || key == "cache-prompt" || key == "cache-idle-slots" {
			var err error
			value, err = llamaBool(value)
			if err != nil {
				return nil, err
			}
			if negative {
				value = strconv.FormatBool(value != "true")
			}
		}
		if negative && (key == "kv-offload" || key == "op-offload") {
			// Pinned preset.cpp reverses args_neg using is_truthy; every other
			// value becomes true. Positive offload values remain native-owned.
			value = strconv.FormatBool(value != "on" && value != "enabled" && value != "true" && value != "1")
		}
		if key == "load-on-startup" {
			boolean, err := llamaBool(value)
			if err != nil || boolean != "false" {
				return nil, servingError("Preset startup loading is forbidden in serving mode.")
			}
		}
		if err := validateLlamaValue(key, value); err != nil {
			return nil, err
		}
		sections[section][key] = value
	}
	if scanner.Err() != nil {
		return nil, servingError("Cannot parse bounded native preset.")
	}
	return sections, nil
}

var llamaShardName = regexp.MustCompile(`^(.*)-([0-9]{5})-of-([0-9]{5})(\.gguf)$`)

func llamaArtifactPaths(path string) ([]string, error) {
	parts := llamaShardName.FindStringSubmatch(path)
	if parts == nil {
		return []string{path}, nil
	}
	count, err := strconv.Atoi(parts[3])
	if err != nil || parts[2] != "00001" || count < 1 || count > 99999 {
		return nil, servingError("Split GGUF must select the first shard with a finite shard count.")
	}
	paths := make([]string, count)
	for i := range paths {
		paths[i] = fmt.Sprintf("%s-%05d-of-%05d%s", parts[1], i+1, count, parts[4])
	}
	return paths, nil
}

func compileLlamaServing(command launchCommand, rt Runtime, requireApproved bool) (*llamaServingLaunch, error) {
	if rt.Serving == nil {
		return nil, servingError("A serving declaration is required by the external preset compiler.")
	}
	if len(servingDeclarationErrors(rt.Serving)) != 0 {
		return nil, servingError("Invalid serving declaration.")
	}
	canonicalIDs := map[string]bool{}
	for _, model := range rt.Serving.Models {
		canonical := llamaCanonicalModelID(model.ModelID)
		if canonicalIDs[canonical] {
			return nil, &ServingFieldError{Field: "serving.models.modelId", ModelID: model.ModelID, Reason: "Native model IDs collide after canonicalization."}
		}
		canonicalIDs[canonical] = true
	}
	for _, model := range rt.Serving.Models {
		if err := validateLlamaModelID(model.ModelID); err != nil {
			return nil, err
		}
	}
	cli, env, err := llamaLaunchOptions(command, rt.EditableLaunch)
	if err != nil {
		return nil, err
	}
	value := func(key string) string {
		if v, ok := cli[key]; ok {
			return v
		}
		return env[key]
	}
	source := value("models-preset")
	if !filepath.IsAbs(source) {
		return nil, servingError("Select one absolute native preset path.")
	}
	capacity, err := llamaPositive(value("models-max"))
	if err != nil {
		return nil, err
	}
	data, binding, err := servingSource(source)
	if err != nil {
		return nil, servingError("Native preset is missing, unstable or exceeds the small-file limit.")
	}
	sections, err := parseLlamaServingPreset(data)
	if err != nil {
		return nil, err
	}
	return compileLlamaServingSections(rt, requireApproved, source, data, binding, sections, cli, env, capacity)
}

// The declaration validator consumes either external source bytes or typed
// compiler bytes. The latter are input bytes, never a generated output path.
func compileLlamaServingSections(rt Runtime, requireApproved bool, source string, data []byte, binding *engines.ServingBinding, sections map[string]map[string]string, cli, env map[string]string, capacity int64) (*llamaServingLaunch, error) {
	if len(servingDeclarationErrors(rt.Serving)) != 0 || rt.Serving == nil {
		return nil, servingError("A valid existing serving declaration is required.")
	}
	if len(sections)-boolInt(sections["*"] != nil) != len(rt.Serving.Models) {
		return nil, servingError("Preset models must exactly match the serving declaration.")
	}
	result := &llamaServingLaunch{Config: rt.Serving.Clone(), Source: source}
	var snapshot strings.Builder
	// Finite adapter defaults are presets, never router envelope arguments.
	snapshot.WriteString("[*]\nctx-size = 8192\nparallel = 1\nkv-unified = false\nfit = off\nsleep-idle-seconds = 300\ncache-ram = 512\nload-on-startup = false\n")
	for mi := range result.Config.Models {
		model := &result.Config.Models[mi]
		preset, exists := sections[model.ModelID]
		if !exists || model.ModelID == "*" || strings.ContainsAny(model.ModelID, "[]\r\n") {
			return nil, servingError("Declared model has no exact native preset.")
		}
		options := map[string]string{"ctx-size": "8192", "parallel": "1", "kv-unified": "false", "fit": "off", "sleep-idle-seconds": "300", "cache-ram": "512", "load-on-startup": "false"}
		options["n-predict"] = strconv.FormatInt(model.MaxOutputTokens, 10)
		// Environment is lowest precedence: snapshot defaults must not mask an
		// explicit environment setting absent from the source presets.
		maps.Copy(options, env)
		maps.Copy(options, sections["*"])
		maps.Copy(options, preset)
		maps.Copy(options, cli)
		_, cliContext := cli["ctx-size"]
		_, modelContext := preset["ctx-size"]
		_, globalContext := sections["*"]["ctx-size"]
		_, envContext := env["ctx-size"]
		if capText, capped := options["kv-unified-per-slot"]; capped && !cliContext && !modelContext && !globalContext && !envContext {
			cap, capErr := llamaPositive(capText)
			slots, slotErr := llamaPositive(options["parallel"])
			if capErr != nil || slotErr != nil || cap > math.MaxInt32/slots {
				return nil, servingError("Per-slot KV allocation overflows native context capacity.")
			}
			options["ctx-size"] = strconv.FormatInt(cap*slots, 10)
		}
		delete(options, "models-preset")
		delete(options, "models-max")
		total, err := llamaPositive(options["ctx-size"])
		if err != nil {
			return nil, err
		}
		if err := validateLlamaPadding("ctx-size", total, model.ModelID); err != nil {
			return nil, err
		}
		slots, err := llamaPositive(options["parallel"])
		if err != nil {
			return nil, err
		}
		unified, err := llamaBool(options["kv-unified"])
		if err != nil {
			return nil, err
		}
		perSlot := total / slots
		if unified == "true" {
			perSlot, err = llamaPositive(options["kv-unified-per-slot"])
			if err != nil || perSlot > math.MaxInt32/slots || perSlot*slots > total {
				return nil, servingError("Unified KV requires an explicit per-slot cap backed by the total allocation.")
			}
		} else if total%slots != 0 {
			return nil, servingError("Non-unified context must divide exactly across slots.")
		} else if err := validateLlamaPadding("per-slot-context", perSlot, model.ModelID); err != nil {
			return nil, err
		}
		if capText, ok := options["kv-unified-per-slot"]; ok {
			cap, err := llamaPositive(capText)
			if err != nil {
				return nil, err
			}
			if err := validateLlamaPadding("kv-unified-per-slot", cap, model.ModelID); err != nil {
				return nil, err
			}
			perSlot = min(perSlot, cap)
		}
		if perSlot <= model.MaxOutputTokens {
			return nil, servingError("Output bound must be strictly below the derived per-slot context.")
		}
		paths, _, err := llamaConsumedPaths(source, options, model.ModelID, 0)
		if err != nil {
			return nil, err
		}
		if len(paths) != len(model.Artifacts) {
			return nil, servingError("Declare every consumed model and draft shard exactly once, in native order.")
		}
		for ai, path := range paths {
			current, err := servingFileBinding(path)
			if err != nil {
				return nil, servingError("A declared model or draft shard is missing or unstable.")
			}
			approved := model.Artifacts[ai].Binding
			if approved != nil && *approved != *current {
				return nil, servingError("Serving artifact binding changed; renewed verification is required.")
			}
			if requireApproved && approved == nil {
				return nil, servingError("Apply requires the reviewed artifact binding.")
			}
			model.Artifacts[ai].Binding = current
		}
		placement := []string{"kv-unified=" + unified}
		for _, key := range []string{"kv-unified-per-slot", "gpu-layers", "split-mode", "tensor-split", "main-gpu", "device", "fit", "cache-ram", "batch-size", "ubatch-size", "cache-type-k", "cache-type-v", "flash-attn", "kv-offload", "op-offload"} {
			if v, ok := options[key]; ok {
				placement = append(placement, key+"="+v)
			}
		}
		result.Envelopes = append(result.Envelopes, engines.ServingEnvelope{ModelID: model.ModelID,
			PerSlotContextTokens: perSlot, TotalContextTokens: total, Slots: slots,
			MaxOutputTokens: model.MaxOutputTokens, Memory: model.Memory,
			Placement: placement, Retention: options["sleep-idle-seconds"], RouterCapacity: capacity})
		snapshot.WriteString("\n[" + model.ModelID + "]\n")
		for _, key := range sortedKeys(options) {
			if err := validateLlamaPresetValue(key, options[key], model.ModelID); err != nil {
				return nil, err
			}
			snapshot.WriteString(key + " = " + options[key] + "\n")
		}
	}
	result.Snapshot = []byte(snapshot.String())
	if len(result.Snapshot) > servingSmallFileLimit {
		return nil, servingError("Compiled native preset exceeds the small snapshot limit.")
	}
	for i := range result.Config.Models {
		result.Config.Models[i].ExpectedNativeSourceDigest = ""
	}
	// Source binding, full small source bytes and derived snapshot are all part
	// of the approval digest. No attested content digest is published here.
	encoded, err := json.Marshal(struct {
		Version  int
		Source   *engines.ServingBinding
		Bytes    []byte
		Snapshot []byte
		Models   []engines.ServingModel
		Capacity int64
	}{1, binding, data, result.Snapshot, result.Config.Models, capacity})
	if err != nil {
		return nil, servingError("Cannot bind native serving inputs.")
	}
	digest := sha256.Sum256(encoded)
	result.Digest = hex.EncodeToString(digest[:])
	for i := range result.Config.Models {
		expected := rt.Serving.Models[i].ExpectedNativeSourceDigest
		if (requireApproved && expected == "") || (expected != "" && expected != result.Digest) {
			return nil, servingError("Native source conflict; obtain a fresh preview before applying.")
		}
		result.Config.Models[i].ExpectedNativeSourceDigest = result.Digest
		result.Envelopes[i].NativeSourceDigest = result.Digest
		result.Envelopes[i].ContentDigest = servingContentDigest(result.Config.Models[i])
	}
	return result, nil
}

func boolInt(value bool) int {
	if value {
		return 1
	}
	return 0
}

func llamaServingReadback(prepared *llamaServingLaunch) *engines.ServingReadback {
	return (&engines.ServingReadback{Status: "unverified", Reason: "Native launch configured; loaded runtime limits and content qualification remain unverified.", Configured: prepared.Envelopes}).Clone()
}

// Protected by engineState.mu. The stop fence closes the registration race
// between cancelling existing loads and acquiring the lifecycle operation lock.
type servingLoadControl struct {
	stops  int
	cause  error
	active map[*servingLoad]struct{}
}

type servingLoad struct {
	cancel context.CancelCauseFunc
}

func stopServingLoads(st *engineState) func() {
	return stopServingLoadsWithCause(st, nil)
}

func stopServingLoadsWithCause(st *engineState, cause error) func() {
	st.mu.Lock()
	st.loads.stops++
	if cause != nil {
		st.loads.cause = cause
	}
	for load := range st.loads.active {
		load.cancel(st.loads.cause)
	}
	st.mu.Unlock()
	return func() {
		st.mu.Lock()
		st.loads.stops--
		if st.loads.stops == 0 {
			st.loads.cause = nil
		}
		st.mu.Unlock()
	}
}

func (e *Executor) guardServingLoad(ctx context.Context, st *engineState, action string, params json.RawMessage) (context.Context, func(), error) {
	noop := func() {}
	binding, load := st.manifest.operationBinding(engines.OperationLoad)
	if !load || action != binding.Action {
		return ctx, noop, nil
	}
	if err := e.activationVeto(st.manifest.Engine); err != nil {
		return ctx, noop, err
	}
	reader, err := e.acquireLoadRead(ctx, st, action, params)
	if err != nil {
		return ctx, noop, err
	}
	ctx, cleanup := registerServingLoad(ctx, st)
	// Register before the lifecycle wait so settings cancellation sees waiters.
	if !residentLock(ctx, &st.opMu) {
		cleanup()
		reader()
		return ctx, noop, vetoCause(ctx, ctx.Err())
	}
	release := sync.OnceFunc(func() { st.opMu.Unlock(); cleanup(); reader() })
	if err := e.activationVeto(st.manifest.Engine); err != nil {
		release()
		return ctx, noop, err
	}
	if err := e.checkServingLoad(st, binding.Action, params); err != nil {
		release()
		return ctx, noop, err
	}
	return e.activationDispatchContext(ctx, st.manifest.Engine), release, nil
}

func registerServingLoad(ctx context.Context, st *engineState) (context.Context, func()) {
	ctx, cancel := context.WithCancelCause(ctx)
	entry := &servingLoad{cancel: cancel}
	st.mu.Lock()
	if st.loads.active == nil {
		st.loads.active = make(map[*servingLoad]struct{})
	}
	st.loads.active[entry] = struct{}{}
	if st.loads.stops != 0 {
		cancel(st.loads.cause)
	}
	st.mu.Unlock()
	cleanup := sync.OnceFunc(func() {
		cancel(nil)
		st.mu.Lock()
		delete(st.loads.active, entry)
		st.mu.Unlock()
	})
	return ctx, cleanup
}

// Caller already holds opMu, under switch admission. Share the same stop fence,
// registry and declaration check without recursively taking the lifecycle lock.
func (e *Executor) guardServingLoadLocked(ctx context.Context, st *engineState, action string, params json.RawMessage) (context.Context, func(), error) {
	reader, err := e.acquireLoadRead(ctx, st, action, params)
	if err != nil {
		return ctx, func() {}, err
	}
	ctx, cleanup := registerServingLoad(ctx, st)
	release := sync.OnceFunc(func() { cleanup(); reader() })
	if err := e.activationVeto(st.manifest.Engine); err != nil {
		release()
		return ctx, func() {}, err
	}
	if err := e.checkServingLoad(st, action, params); err != nil {
		release()
		return ctx, func() {}, err
	}
	return e.activationDispatchContext(ctx, st.manifest.Engine), release, nil
}

// Reject undeclared loads before mutation admission. guardServingLoad repeats
// this check under opMu after admission, keeping authorization and dispatch on
// one applied generation without reversing switchAdmission -> opMu lock order.
func (e *Executor) checkServingLoad(st *engineState, action string, params json.RawMessage) error {
	binding, load := st.manifest.operationBinding(engines.OperationLoad)
	if !load || action != binding.Action {
		return nil
	}
	st.mu.Lock()
	config := st.plat.Runtime.Serving.Clone()
	st.mu.Unlock()
	if config == nil {
		return nil
	}
	return rejectUndeclaredServingLoad(config, binding.modelParam(), params)
}

func rejectUndeclaredServingLoad(config *engines.ServingConfig, field string, params json.RawMessage) error {
	var values map[string]json.RawMessage
	if err := json.Unmarshal(params, &values); err != nil {
		return &ServingFieldError{Field: field, Reason: "A declared model ID is required."}
	}
	var model string
	if err := json.Unmarshal(values[field], &model); err != nil {
		return &ServingFieldError{Field: field, Reason: "A declared model ID is required."}
	}
	for _, declared := range config.Models {
		if model == declared.ModelID {
			return nil
		}
	}
	return &ServingFieldError{Field: field, ModelID: model, Reason: "Model is not declared in the applied serving configuration."}
}

// recheckLlamaServing runs immediately before spawn. Do not reuse the preview's
// stat cache: sample the artifacts and re-read the source on every check.
func recheckLlamaServing(launch launchCommand, rt Runtime, prepared *llamaServingLaunch) error {
	current, err := compileLlamaServing(launch, rt, true)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(current, prepared) {
		return servingError("Native serving inputs changed before spawn.")
	}
	return nil
}

// The pre-spawn compiler has already validated these exact finite loads and
// their visibility. Replay the reviewed bytes, never discover another adapter.
func llamaSnapshotCommand(command launchCommand, snapshot string, finite ...[]modelconfig.ModelLoad) launchCommand {
	args := []string{}
	for i := 0; i < len(command.Args); i++ {
		name, value, attached := strings.Cut(command.Args[i], "=")
		name = llamaLongOption(name)
		if name == "--models-preset" {
			if !attached {
				i++
			}
			continue
		}
		if attached && llamaOptionKey(name) != "" {
			args = append(args, name, value)
		} else {
			args = append(args, command.Args[i])
		}
	}
	command.Args = append(args, "--models-preset", snapshot)
	command.Env = maps.Clone(command.Env)
	if command.Env == nil {
		command.Env = map[string]string{}
	}
	for _, loads := range finite {
		for _, load := range loads {
			if r := load.Resolved; r != nil && strings.HasPrefix(r.ResolverRevision, "auto/") && r.DeviceBinding != nil {
				command.Env["CUDA_VISIBLE_DEVICES"] = r.DeviceBinding.DeviceID
			}
		}
	}
	for key := range command.Env {
		if environmentKey(key) == environmentKey("LLAMA_ARG_MODELS_PRESET") {
			delete(command.Env, key)
		}
	}
	// Neutralize an inherited source-path assignment as well. The snapshot CLI
	// is authoritative, so no child can reopen the original mutable preset.
	command.Env["LLAMA_ARG_MODELS_PRESET"] = snapshot
	return command
}
