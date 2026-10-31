<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Engine Manifest Reference

An engine manifest is a single JSON file that tells `nvpair-engine-manager`
how to detect, install, run, probe, and act on one inference engine. The
runner is engine-agnostic: **adding an engine is a manifest, not code.**

## Private memory planning capability

A manifest may select `"memory_plan": {"adapter": "strata"}`. Each supported
platform declares its own `memory_plan` object with `command` and
`working_directory`. There is no inherited invocation or cross-platform
fallback. An invocation without an adapter is invalid; an absent invocation on
the selected platform leaves planning unavailable.

The command is exactly an absolute interpreter followed by
`-B -s -m serve.server --estimate-memory --memory-plan-request {memory_plan_request}`.
The working directory is an absolute, bound package root. Engine-manager alone
substitutes the private request file. This capability is internal to model
inspection, not an action, public executable interface or launch prerequisite.
No bundled engine enables it yet. The provider interface consumes a detached
prepared request and returns a neutral plan. Failed reports remain Unknown;
they do not select another estimation source.

Requests are at most 1 MiB. Combined stdout/stderr is at most 8 MiB and stdout
must contain exactly one JSON value. Errors exclude raw output, argv and paths.
One subprocess is admitted per executor through the inspection gate, without
queueing or holding lifecycle locks. The earlier of the parent deadline and
five seconds bounds execution, with 500 ms reserved inside it for cleanup.
Cancellation, overflow or timeout refuses the result without retry.

The request is written in a private OS temporary directory (Unix 0700, request
0600; Windows protected current-user/SYSTEM ACL). HOME, TEMP, TMP and TMPDIR
refer to that directory. The exact environment otherwise retains only PATH and
SYSTEMROOT. Credentials, proxies, PYTHON variables, preload hooks, STRATA
variables and CUDA controls are excluded. This is environment isolation, not
an OS network sandbox; reviewed interpreter/system site hooks remain trusted.

Query containment extends the platform process owner: a retained Unix process
group, or a Windows kill-on-close Job assigned while the child is suspended,
before resuming it. Cleanup also runs after normal parent exit to terminate
pipe-holding descendants. Unconfirmed cleanup returns a distinct failure and
retains admission and temporary files, rather than declaring success or
removing files while a child may still use them. A manager crash in the Windows
Start-to-assignment interval can leave a suspended process; no query code has
run in that interval. Windows Job and ACL execution acceptance remains Task 6;
cross-vet alone does not establish it.

## Typed model load contract and option matrix

`runtime.modelLoads` is an ordered collection of exact `modelId` records with
portable `requested` preferences and optional target-owned finite `resolved`
readback. Registry validation canonicalizes model order and rejects duplicate
IDs, unknown typed fields and invalid allocations. Shared and host-platform
runtime values use the existing deep merge: an explicitly supplied collection,
including `[]`, replaces the inherited collection. Port-only overrides preserve
it. Both JSON mergers retain decimal number lexemes for unrelated fields too.
The complete settings/request transport remains bounded to 64 KiB.

Typed-only llama process launches consume manager-generated immutable presets,
including fresh starts and crash relaunches. Direct previews compile and
readback clones the stored collection. The existing atomic override owner can
replace it alongside port/launch/serving fields, including an explicit empty
host-platform mask. No second store or compatibility mechanism is introduced.

ConfigureLaunch accepts frozen reviewed typed settings through the existing
stop, persist, rebind and conditional restart owner. Typed SetPort shares this
owner. Worker load-only modelChange previews return complete target resolution
and conversion diffs; the broker owns patch acceptance and policy application.
Conversion validates the complete current external router and reviewed finite
identity, refuses unsupported members without dropping models, and rechecks its
source before stopping and restarting. A generated output pathname is never
desired configuration. Unchanged typed settings do not restart the process;
Off, Pause, engine locks and adopted ownership retain their existing behavior.

The projection in `modelsettings_load.go` reads flags, aliases, environment,
negative switches and native rejection status directly from
`llamaServingOptions` in `serving_llamacpp.go`. Metadata contains no parallel
native flag parser. The coverage test compares against the actual entries and
must reject deletion of any actual row or addition of an unprojected native
entry. The table below documents those entries, including rejected controls.
Underscore-normalized long aliases follow the existing native reader. Native
negative environment aliases retain the actual table's spelling.

Source references: bundled release b12020; adapter/native parser
[0ec4027a2 common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2/common/arg.cpp)
and [common/preset.cpp](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2/common/preset.cpp).
These identify source, not the installed build. Unsupported means no typed editor
values, even when the native parser or existing Advanced launch accepts a value.
No raw extras field bypasses this matrix.

B descriptors describe the manual vocabulary. Target preview validates artifacts, native build,
architecture and interactions. Defaults shown here are inherited source labels,
never newly saved intent or Auto defaults. Absence clears to inheritance; it is
distinct from explicit false/zero, embedded recommendations and finite resolution.
Conditional K/V alternatives have separate unsupported descriptors with reasons;
Auto context/placement likewise report unavailable without selecting a floor/cap.
The scalar GPU layer descriptor has a nonnegative int32 transport range, further
bounded by target model facts. Shared validation remains authoritative for cross-
field context/slot products, ubatch versus batch, positive repeat penalty and the
idle-time sentinel (-1 or positive, excluding zero).
Manual load changes use whole-router restart with the broker's 60-second drain
bound disclosed in preview. Generation locks retain their generation-only
scope; engine locks still veto activation. Copy defaults to tuning only at its
owner. Auto remains unavailable, and configuration makes no qualified-readiness
claim.

| Native key | Actual flags/aliases and environment | UI label | Owner | Valid range/enums or refusal | Default source | B/D status | Portability | Generated preset mapping | Estimate/identity effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `models-preset` | `--models-preset`; env `LLAMA_ARG_MODELS_PRESET` | Generated model preset | Manager/router | No model-local override | Existing native/router policy, uninspected | B owner control; no editor | No | Manager-generated immutable preset path | Router identity/capacity/startup; no automatic retuning |
| `models-max` | `--models-max`; env `LLAMA_ARG_MODELS_MAX` | Router model capacity | Manager/router | Existing positive finite int32 router capacity | Existing native/router policy, uninspected | B owner control; no editor | No | Preserve configured router capacity | Router identity/capacity/startup; no automatic retuning |
| `ctx-size` | `--ctx-size`, `-c`; env `LLAMA_ARG_CTX_SIZE` | Tokens per request | Model load | Per request 1..2147483392; pad up to 256; checked total <= int32 | Finite adapter: 8192 per request | B | Requested only | ctx-size = padded per-request * slots; kv-unified = false | Native context identity; KV/state/workspace |
| `parallel` | `--parallel`, `-np`; env `LLAMA_ARG_N_PARALLEL` | Native slots | Model load | 1..2147483647; context product bounded | Finite adapter: 1 | B | Requested only | parallel = slots | Native slots identity; KV/state/workspace |
| `kv-unified-per-slot` | `--kv-unified-per-slot`; env `LLAMA_ARG_KV_UNIFIED_PER_SLOT` | Unified KV per-slot cap | Model load | No typed edit; preserve/read existing records | Existing finite adapter/source record | D typed override | No | B compiler requires non-unified; do not reinterpret saved unified loads | KV/slot identity; sharing arithmetic |
| `kv-unified` | `--kv-unified`, `-kvu`, `--no-kv-unified`, `-no-kvu`; env `LLAMA_ARG_KV_UNIFIED`, `LLAMA_ARG_NO_KV_UNIFIED` | Unified KV allocation | Model load | No typed edit; preserve/read existing records | Existing finite adapter/source record | D typed override | No | B compiler requires non-unified; do not reinterpret saved unified loads | KV/slot identity; sharing arithmetic |
| `n-predict` | `--predict`, `--n-predict`, `-n`; env `LLAMA_ARG_N_PREDICT` | Native output token limit | Request/native | Existing finite adapter: positive int32; typed override unavailable | Native/source behavior, uninspected | D | No typed copy | No generated B override | Request/native output identity; not context or admission limit |
| `gpu-layers` | `--gpu-layers`, `--n-gpu-layers`, `-ngl`; env `LLAMA_ARG_N_GPU_LAYERS` | GPU placement | Model placement | native-default, cpu, finite nonnegative layers; facts bound layers | Inherited native placement, unresolved | B; Auto waits for C/O2 | Requested modes/layers only | 0 for CPU; finite validated layers; resolve inherited placement | Placement identity; host/device weights and KV; no ordinal mapping |
| `split-mode` | `--split-mode`, `-sm`; env `LLAMA_ARG_SPLIT_MODE` | GPU split mode | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `tensor-split` | `--tensor-split`, `-ts`; env `LLAMA_ARG_TENSOR_SPLIT` | GPU tensor proportions | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `main-gpu` | `--main-gpu`, `-mg`; env `LLAMA_ARG_MAIN_GPU` | Main native GPU | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `device` | `--device`, `-dev`; env `LLAMA_ARG_DEVICE` | Native devices | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `backend-sampling` | `-bs`, `--backend-sampling`; env `LLAMA_ARG_BACKEND_SAMPLING` | Engine-side sampling | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `sleep-idle-seconds` | `--sleep-idle-seconds`; env `LLAMA_ARG_SLEEP_IDLE_SECONDS` | Unload after idle (seconds; -1 or positive) | Model load | -1 or positive int32; zero invalid | Finite adapter: 300 seconds | B advanced | Requested only | sleep-idle-seconds = explicit value | Retention identity; residency/keep-loaded protection wins |
| `cache-ram` | `--cache-ram`, `-cram`; env `LLAMA_ARG_CACHE_RAM` | Host prompt cache (MiB) | Model load | 1..2147483647 MiB | Finite adapter: 512 MiB | B advanced | Requested only | cache-ram = explicit MiB | Native identity; host prompt cache, no admission bound |
| `fit` | `--fit`, `-fit`; env `LLAMA_ARG_FIT` | Native automatic fitting | Manager/router | off only | Finite adapter enforced: off | B owner control; no editor | No | fit = off | Router identity/capacity/startup; no automatic retuning |
| `cache-prompt` | `--cache-prompt`, `--no-cache-prompt`; env `LLAMA_ARG_CACHE_PROMPT`, `LLAMA_ARG_NO_CACHE_PROMPT` | Prompt caching | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `cache-idle-slots` | `--cache-idle-slots`, `--no-cache-idle-slots`; env `LLAMA_ARG_CACHE_IDLE_SLOTS`, `LLAMA_ARG_NO_CACHE_IDLE_SLOTS` | Idle slot caching | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `batch-size` | `-b`, `--batch-size`; env `LLAMA_ARG_BATCH` | Logical prompt batch | Model load | Positive int32; ubatch <= batch | Inherited native default, unresolved | B advanced | Requested only | batch-size = explicit value | Native allocation identity; workspace |
| `ubatch-size` | `-ub`, `--ubatch-size`; env `LLAMA_ARG_UBATCH` | Physical prompt batch | Model load | Positive int32; ubatch <= batch | Inherited native default, unresolved | B advanced | Requested only | ubatch-size = explicit value | Native allocation identity; workspace |
| `cont-batching` | `-cb`, `--cont-batching`, `-nocb`, `--no-cont-batching`; env `LLAMA_ARG_CONT_BATCHING`, `LLAMA_ARG_NO_CONT_BATCHING` | Continuous batching | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `cache-type-k` | `-ctk`, `--cache-type-k`; env `LLAMA_ARG_CACHE_TYPE_K` | K cache precision | Model load | B: inherited or f16; bf16/q8_0/q4_0 require validated build/architecture | Inherited native precision, unresolved | B constrained | Requested only | cache-type-k = explicit precision; otherwise omit | KV quality/identity; block-aware KV sizes; quantized V requires validated Flash Attention |
| `cache-type-v` | `-ctv`, `--cache-type-v`; env `LLAMA_ARG_CACHE_TYPE_V` | V cache precision | Model load | B: inherited or f16; bf16/q8_0/q4_0 require validated build/architecture | Inherited native precision, unresolved | B constrained | Requested only | cache-type-v = explicit precision; otherwise omit | KV quality/identity; block-aware KV sizes; quantized V requires validated Flash Attention |
| `swa-full` | `--swa-full`; env `LLAMA_ARG_SWA_FULL` | Full sliding-window cache | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `flash-attn` | `-fa`, `--flash-attn`; env `LLAMA_ARG_FLASH_ATTN` | Flash Attention | Model load | inherit, on, off; validate target combination | Inherited native policy, unresolved | B | Requested only | flash-attn = on/off; inherit omits | Attention identity; KV/workspace support |
| `context-shift` | `--context-shift`, `--no-context-shift`; env `LLAMA_ARG_CONTEXT_SHIFT`, `LLAMA_ARG_NO_CONTEXT_SHIFT` | Context shifting | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `rope-scaling` | `--rope-scaling`; env `LLAMA_ARG_ROPE_SCALING_TYPE` | RoPE scaling mode | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `rope-scale` | `--rope-scale`; env `LLAMA_ARG_ROPE_SCALE` | RoPE context scale | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `rope-freq-base` | `--rope-freq-base`; env `LLAMA_ARG_ROPE_FREQ_BASE` | RoPE frequency base | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `rope-freq-scale` | `--rope-freq-scale`; env `LLAMA_ARG_ROPE_FREQ_SCALE` | RoPE frequency scale | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `yarn-orig-ctx` | `--yarn-orig-ctx`; env `LLAMA_ARG_YARN_ORIG_CTX` | YaRN original context | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `yarn-ext-factor` | `--yarn-ext-factor`; env `LLAMA_ARG_YARN_EXT_FACTOR` | YaRN extension factor | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `yarn-attn-factor` | `--yarn-attn-factor`; env `LLAMA_ARG_YARN_ATTN_FACTOR` | YaRN attention factor | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `yarn-beta-slow` | `--yarn-beta-slow`; env `LLAMA_ARG_YARN_BETA_SLOW` | YaRN slow correction | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `yarn-beta-fast` | `--yarn-beta-fast`; env `LLAMA_ARG_YARN_BETA_FAST` | YaRN fast correction | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `override-kv` | `--override-kv`; env None | Metadata overrides | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `override-tensor` | `-ot`, `--override-tensor`; env `LLAMA_ARG_OVERRIDE_TENSOR` | Tensor placement overrides | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `cpu-moe` | `-cmoe`, `--cpu-moe`; env `LLAMA_ARG_CPU_MOE` | CPU expert placement | Model load | Serving: true/1/on only; omit to leave unpinned; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `n-cpu-moe` | `-ncmoe`, `--n-cpu-moe`; env `LLAMA_ARG_N_CPU_MOE` | CPU expert layers | Model load | Serving: canonical decimal int32 >= 0, no sign or leading zeros; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `moe-cache-mib` | `--moe-cache-mib`; env `LLAMA_ARG_MOE_CACHE_MIB` | GPU expert cache (MiB) | Model load | Serving: canonical decimal int32 >= 0; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native allocation identity; unknown allocation cost |
| `n-cpu-ffn` | `-ncffn`, `--n-cpu-ffn`; env `LLAMA_ARG_N_CPU_FFN` | CPU feed-forward layers | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `kv-offload` | `-kvo`, `--kv-offload`, `-nkvo`, `--no-kv-offload`; env `LLAMA_ARG_KV_OFFLOAD`, `LLAMA_ARG_NO_KV_OFFLOAD` | Offload KV cache | Model load | Explicit true/false or absent | Inherited native policy, unresolved | B advanced | Requested only | kv-offload = explicit boolean | Placement identity; host/device allocation |
| `op-offload` | `--op-offload`, `--no-op-offload`; env None | Offload operations | Model load | Explicit true/false or absent | Inherited native policy, unresolved | B advanced | Requested only | op-offload = explicit boolean | Placement identity; host/device allocation |
| `repack` | `--repack`, `-nr`, `--no-repack`; env `LLAMA_ARG_REPACK`, `LLAMA_ARG_NO_REPACK` | Weight repacking | Model load | Serving: explicit boolean, negative preset keys invert native truthiness; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `no-host` | `--no-host`; env `LLAMA_ARG_NO_HOST` | Disable host buffers | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `load-mode` | `-lm`, `--load-mode`; env `LLAMA_ARG_LOAD_MODE` | Model loading mode | Model load | Serving: auto, none, mmap, mlock, mmap+mlock, dio; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `lazy-mode` | `-lzm`, `--lazy-mode`; env `LLAMA_ARG_LAZY_MODE` | Lazy loading mode | Model load | Serving: auto, on or off; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `numa` | `--numa`; env `LLAMA_ARG_NUMA` | NUMA placement | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `rpc` | `--rpc`; env `LLAMA_ARG_RPC` | Native RPC devices | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `models-autoload` | `--models-autoload`, `--no-models-autoload`; env `LLAMA_ARG_MODELS_AUTOLOAD`, `LLAMA_ARG_NO_MODELS_AUTOLOAD` | Router automatic model loading | Manager/router | No model-local override | Existing native/router policy, uninspected | B owner control; no editor | No | None: model-local override refused | Router identity/capacity/startup; no automatic retuning |
| `warmup` | `--warmup`, `--no-warmup`; env None | Native warmup | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `model` | `--model`, `-m`; env `LLAMA_ARG_MODEL` | Resolved model artifacts | Target artifacts | No caller path/repository override | Saved preset/receipt/cache binding, unresolved | B resolver only; no path editor | No | Exact ordered local model/draft shards; preserve validated draft declarations | Ordered artifact identity; weights/companions or unknown extra cost |
| `model-draft` | `--spec-draft-model`, `--model-draft`, `-md`; env `LLAMA_ARG_SPEC_DRAFT_MODEL` | Resolved draft artifacts | Target artifacts | No caller path/repository override | Saved preset/receipt/cache binding, unresolved | B resolver only; no path editor | No | Exact ordered local model/draft shards; preserve validated draft declarations | Ordered artifact identity; weights/companions or unknown extra cost |
| `alias` | `--alias`, `-a`; env `LLAMA_ARG_ALIAS` | Native model alias | Target artifacts | No caller path/repository override | Saved preset/receipt/cache binding, unresolved | D override unsupported | No | None: target resolver owns identity | Ordered artifact identity; weights/companions or unknown extra cost |
| `mmproj` | `--mmproj`, `-mm`; env `LLAMA_ARG_MMPROJ` | Multimodal projector path | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `mmproj-url` | `--mmproj-url`, `-mmu`; env `LLAMA_ARG_MMPROJ_URL` | Multimodal projector URL | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `mmproj-auto` | `--mmproj-auto`, `--no-mmproj`, `--no-mmproj-auto`; env `LLAMA_ARG_MMPROJ_AUTO`, `LLAMA_ARG_NO_MMPROJ_AUTO` | Automatic projector selection | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `mmproj-offload` | `--mmproj-offload`, `--no-mmproj-offload`; env `LLAMA_ARG_MMPROJ_OFFLOAD`, `LLAMA_ARG_NO_MMPROJ_OFFLOAD` | Projector offload | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `mmproj-device` | `-mmdev`, `--mmproj-device`; env `MTMD_BACKEND_DEVICE` | Projector device | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `image-min-tokens` | `--image-min-tokens`; env `LLAMA_ARG_IMAGE_MIN_TOKENS` | Minimum image tokens | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `image-max-tokens` | `--image-max-tokens`; env `LLAMA_ARG_IMAGE_MAX_TOKENS` | Maximum image tokens | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `mtmd-batch-max-tokens` | `--mtmd-batch-max-tokens`; env `LLAMA_ARG_MTMD_BATCH_MAX_TOKENS` | Multimodal batch tokens | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `lora` | `--lora`; env None | LoRA adapters | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `lora-scaled` | `--lora-scaled`; env None | Scaled LoRA adapters | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `control-vector` | `--control-vector`; env None | Control vectors | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `control-vector-scaled` | `--control-vector-scaled`; env None | Scaled control vectors | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `control-vector-layer-range` | `--control-vector-layer-range`; env None | Control vector layers | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `models-dir` | `--models-dir`; env `LLAMA_ARG_MODELS_DIR` | Native model directory | Target artifacts | No caller path/repository override | Saved preset/receipt/cache binding, unresolved | D override unsupported | No | None: target resolver owns identity | Ordered artifact identity; weights/companions or unknown extra cost |
| `model-url` | `--model-url`, `-mu`; env `LLAMA_ARG_MODEL_URL` | Native model URL | Target artifacts | No caller path/repository override | Saved preset/receipt/cache binding, unresolved | D override unsupported | No | None: target resolver owns identity | Ordered artifact identity; weights/companions or unknown extra cost |
| `hf-repo` | `--hf-repo`, `-hf`, `-hfr`; env `LLAMA_ARG_HF_REPO` | Native repository | Target artifacts | No caller path/repository override | Saved preset/receipt/cache binding, unresolved | D override unsupported | No | None: target resolver owns identity | Ordered artifact identity; weights/companions or unknown extra cost |
| `hf-file` | `--hf-file`, `-hff`; env `LLAMA_ARG_HF_FILE` | Native repository file | Target artifacts | No caller path/repository override | Saved preset/receipt/cache binding, unresolved | D override unsupported | No | None: target resolver owns identity | Ordered artifact identity; weights/companions or unknown extra cost |
| `docker-repo` | `--docker-repo`, `-dr`; env `LLAMA_ARG_DOCKER_REPO` | Native container repository | Target artifacts | No caller path/repository override | Saved preset/receipt/cache binding, unresolved | D override unsupported | No | None: target resolver owns identity | Ordered artifact identity; weights/companions or unknown extra cost |
| `hf-repo-draft` | `--spec-draft-hf`, `--hf-repo-draft`, `-hfd`, `-hfrd`; env `LLAMA_ARG_SPEC_DRAFT_HF_REPO` | Native draft repository | Target artifacts | No caller path/repository override | Saved preset/receipt/cache binding, unresolved | D override unsupported | No | None: target resolver owns identity | Ordered artifact identity; weights/companions or unknown extra cost |
| `spec-draft-type-k` | `--spec-draft-type-k`, `-ctkd`, `--cache-type-k-draft`; env `LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K` | Draft K cache precision | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `spec-draft-type-v` | `--spec-draft-type-v`, `-ctvd`, `--cache-type-v-draft`; env `LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V` | Draft V cache precision | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `spec-draft-override-tensor` | `--spec-draft-override-tensor`, `-otd`, `--override-tensor-draft`; env None | Draft tensor overrides | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `spec-draft-cpu-moe` | `--spec-draft-cpu-moe`, `-cmoed`, `--cpu-moe-draft`; env `LLAMA_ARG_SPEC_DRAFT_CPU_MOE` | Draft CPU experts | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `spec-draft-n-cpu-moe` | `--spec-draft-n-cpu-moe`, `--spec-draft-ncmoe`, `-ncmoed`, `--n-cpu-moe-draft`; env `LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE` | Draft CPU expert layers | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `spec-draft-device` | `--spec-draft-device`, `-devd`, `--device-draft`; env None | Draft devices | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `spec-draft-ngl` | `--spec-draft-ngl`, `-ngld`, `--gpu-layers-draft`, `--n-gpu-layers-draft`; env `LLAMA_ARG_N_GPU_LAYERS_DRAFT` | Draft GPU layers | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `spec-draft-backend-sampling` | `--spec-draft-backend-sampling`, `--no-spec-draft-backend-sampling`; env `LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING`, `LLAMA_ARG_NO_SPEC_DRAFT_BACKEND_SAMPLING` | Draft engine-side sampling | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `spec-draft-n-max` | `--spec-draft-n-max`; env `LLAMA_ARG_SPEC_DRAFT_N_MAX` | Maximum speculative draft tokens | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `spec-type` | `--spec-type`; env `LLAMA_ARG_SPEC_TYPE` | Speculation type | Model load | Serving: none or draft-mtp with the built-in head; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `spec-default` | `--spec-default`; env None | Default speculation preset | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `mtp` | `--mtp`; env None | Multi-token prediction preset | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `dflash` | `--dflash`; env None | DFlash preset | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `eagle3` | `--eagle3`; env None | Eagle3 preset | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `embd-gemma-default` | `--embd-gemma-default`; env None | Gemma embedding preset | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `fim-qwen-1.5b-default` | `--fim-qwen-1.5b-default`; env None | Qwen 1.5B fill-in-middle preset | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `fim-qwen-3b-default` | `--fim-qwen-3b-default`; env None | Qwen 3B fill-in-middle preset | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `fim-qwen-7b-default` | `--fim-qwen-7b-default`; env None | Qwen 7B fill-in-middle preset | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `fim-qwen-7b-spec` | `--fim-qwen-7b-spec`; env None | Qwen 7B speculative preset | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `fim-qwen-14b-spec` | `--fim-qwen-14b-spec`; env None | Qwen 14B speculative preset | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `fim-qwen-30b-default` | `--fim-qwen-30b-default`; env None | Qwen 30B fill-in-middle preset | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `gpt-oss-20b-default` | `--gpt-oss-20b-default`; env None | GPT OSS 20B preset | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `gpt-oss-120b-default` | `--gpt-oss-120b-default`; env None | GPT OSS 120B preset | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `vision-gemma-4b-default` | `--vision-gemma-4b-default`; env None | Gemma 4B vision preset | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `vision-gemma-12b-default` | `--vision-gemma-12b-default`; env None | Gemma 12B vision preset | Model load | Native boolean; no typed override | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `fit-target` | `--fit-target`, `-fitt`; env `LLAMA_ARG_FIT_TARGET` | Native fit memory target | Manager/router | No model-local override | Existing native/router policy, uninspected | B owner control; no editor | No | None: model-local override refused | Router identity/capacity/startup; no automatic retuning |
| `fit-ctx` | `--fit-ctx`, `-fitc`; env `LLAMA_ARG_FIT_CTX` | Native fit context target | Manager/router | No model-local override | Existing native/router policy, uninspected | B owner control; no editor | No | None: model-local override refused | Router identity/capacity/startup; no automatic retuning |
| `cache-reuse` | `--cache-reuse`; env `LLAMA_ARG_CACHE_REUSE` | Prompt cache reuse | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `ctx-checkpoints` | `--ctx-checkpoints`, `--swa-checkpoints`, `-ctxcp`; env `LLAMA_ARG_CTX_CHECKPOINTS` | Context checkpoints | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `checkpoint-min-step` | `--checkpoint-min-step`, `-cms`; env `LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT` | Minimum checkpoint spacing | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `prompt-cache` | `--prompt-cache`; env None | Prompt cache file | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `prompt-cache-all` | `--prompt-cache-all`; env None | Cache all prompts | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `prompt-cache-ro` | `--prompt-cache-ro`; env None | Read-only prompt cache | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `slot-save-path` | `--slot-save-path`; env None | Slot save directory | Model load | No typed values accepted | Inherited native/model behavior, uninspected | D unsupported | No typed copy | None: typed override refused | Native semantics identity; unknown allocation/work cost |
| `load-on-startup` | `--load-on-startup`; env `__PRESET_LOAD_ON_STARTUP` | Startup model loading | Manager/router | false only | Finite adapter enforced: false | B owner control; no editor | No | load-on-startup = false | Router identity/capacity/startup; no automatic retuning |

### Generation, template, reasoning and other owners

These controls are outside `llamaServingOptions`. Request keys are not aliases
for native launch flags. Their source mapping is documented here for coverage,
without adding another launch parser. Descriptors expose only existing policy
fields or an explicit unsupported reason.

| Native/request spelling | Owner and UI label | Valid range/enums | Default source | Delivery | Portable | Generated preset mapping | Estimate/identity effect |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `--threads/-t`, `LLAMA_ARG_THREADS` | Model load: Generation threads | Positive int32 | Native, unresolved | B advanced | Requested only | `threads` explicit; no core-count tuning | CPU/workspace and native identity |
| `--threads-batch/-tb` (no environment name in the pinned definition) | Model load: Prompt-processing threads | Positive int32 | Native generation threads, unresolved | B advanced | Requested only | `threads-batch` explicit | CPU/workspace and native identity |
| `temperature`; native `--temp` | Request policy: Temperature | Finite >= 0 | Existing policy or native/model recommendation; absent stays absent | A | Selected leaf | None | Request/qualification fingerprint; no native allocation |
| `top_p`; native `--top-p` | Request policy: Top P | 0..1 | Same inherited source | A | Selected leaf | None | Request/qualification fingerprint |
| `top_k`; native `--top-k` | Request policy: Top K | Integer 0..9007199254740991 | Same inherited source | A | Selected leaf | None | Request/qualification fingerprint |
| `min_p`; native `--min-p` | Request policy: Min P | 0..1; llama.cpp only | Same inherited source | A | Selected leaf | None | Request/qualification fingerprint |
| `repeat_penalty`; native `--repeat-penalty` | Request policy: Repeat penalty | Finite strictly > 0; descriptor label states exclusivity at zero | Same inherited source | A | Selected leaf | None | Request/qualification fingerprint |
| `frequency_penalty`; native `--frequency-penalty` | Request policy: Frequency penalty | -2..2 | Same inherited source | A | Selected leaf | None | Request/qualification fingerprint |
| `presence_penalty`; native `--presence-penalty` | Request policy: Presence penalty | -2..2 | Same inherited source | A | Selected leaf | None | Request/qualification fingerprint |
| `seed`; native `--seed/-s` | Request policy: Seed | Integer -9007199254740991..9007199254740991 | Same inherited source | A | Selected leaf | None | Request/qualification fingerprint |
| `requestOptionsMode` | Request policy: Generation preference mode | defaults/enforced; enforced requires options | Existing policy | A | Selected value | None | Request merge semantics; no restart |
| Other samplers, output limits, stop strings | Request/native: Additional samplers and Output/stop options | No new typed leaves; preserve opaque Advanced policy fields | Existing request/native | D typed controls | No new copy fields | None | Request semantics; no context/admission inference |
| `--jinja/--no-jinja`, `LLAMA_ARG_JINJA` | Template: Jinja rendering | Native boolean; typed override unavailable | Embedded/native, uninspected | D | No | No B override | Prompt semantics and request compatibility |
| `--chat-template`, `LLAMA_ARG_CHAT_TEMPLATE` | Template: Chat template | Native template text; typed override unavailable | Model metadata/native | D | No | No B override | Template digest/prompt identity; no model-family heuristics |
| `--chat-template-file`, `LLAMA_ARG_CHAT_TEMPLATE_FILE` | Template: Chat template file | Target-local file; typed override unavailable | Model metadata/native | D | No paths | No B override | Local file binding and prompt identity |
| `--chat-template-kwargs`, `LLAMA_ARG_CHAT_TEMPLATE_KWARGS` | Template: Template arguments / thinking | Native JSON object; typed override unavailable | Embedded/native | D | No | No B override | Thinking/template semantics, not sampling recommendations |
| `--skip-chat-parsing/--no-skip-chat-parsing` | Template: Chat parsing | Native boolean; typed override unavailable | Native, uninspected | D | No | No B override | Reasoning/tool response parsing |
| `--reasoning/-rea`, `LLAMA_ARG_REASONING` | Request/template: Reasoning mode | Native auto/on/off; typed override unavailable | Native/template | D | No | No B override | Prompt/reasoning semantics and token cost |
| `--reasoning-format`, `LLAMA_ARG_THINK` | Request/template: Reasoning format | Native none/deepseek/deepseek-legacy/auto; typed override unavailable | Native parser/template | D | No | No B override | Output parsing and request compatibility |
| `--reasoning-effort`, `LLAMA_ARG_REASONING_EFFORT` | Request/template: Reasoning effort | Native template effort string; typed override unavailable | Template default | D | No | No B override | Prompt semantics; unknown token cost |
| `--reasoning-budget`, `LLAMA_ARG_THINK_BUDGET` | Request/template: Reasoning token budget | Native -1, 0 or positive tokens; typed override unavailable | Native, uninspected | D | No | No B override | Token/work cost; no automatic context change |
| `--reasoning-budget-message`, `LLAMA_ARG_THINK_BUDGET_MESSAGE` | Request/template: Reasoning budget message | Native text; typed override unavailable | Native, uninspected | D | No | No B override | Prompt/output semantics |
| `--reasoning-preserve/--no-reasoning-preserve`, `LLAMA_ARG_REASONING_PRESERVE` | Request/template: Preserve reasoning history | Native boolean with template support; typed override unavailable | Native/template, uninspected | D | No | No B override | History token cost and template compatibility |
| `--grammar`, `--grammar-file`, `--json-schema/-j`, `--json-schema-file/-jf` | Request/template: Grammar / structured output | Native grammar/schema/text/file; typed override unavailable | Existing request/native | D | No files or new typed leaves | No B override | Request compatibility and generation work |
| Ports, bind address, CORS | Router/target settings | Existing managed launch-control validation | Target manifest/settings | Existing engine-wide path | No model copy | Keep existing manager controls | Whole-router/local network policy |
| Roles, served/resident membership, bindings | Broker node policy | Existing target policy validation and qualification prerequisites | Target policy | B coupled seam pending O3/O4/O5 | Explicit selected membership only; no lock/evidence/bounds | None | Routing/admission/qualification owner, never native allocation |
| Admission context/output/memory/concurrency and qualification | Broker policy/evidence | Existing validation; not new model copy fields | Existing target records | No new authoring | No | None | Authoritative admission/evidence; estimates cannot manufacture them |

## Where manifests live

- **Bundled** defaults are compiled into the binary from
  `nvpair-engine-manager/manifests/*.json` (e.g. `ollama.json`).
- **User / third-party** manifests are read at runtime from
  the per-user data dir's `engines/*.json`
  (`%LocalAppData%\Nvidia Corporation\Personal AI Router\engines\` on Windows,
  `~/.config/Nvidia Corporation/Personal AI Router/engines/` on Linux,
  `~/Library/Application Support/Nvidia Corporation/Personal AI Router/engines/` on macOS).

A user manifest **overrides** a bundled one with the same `engine` name,
so a vendor or operator can ship or tweak an engine without rebuilding. The
override is a **deep-merge onto the bundled manifest** (override keys win, the
same key-by-key rules as the shared-defaults merge below), not a wholesale
replace — so a partial file pins only the fields it sets and keeps inheriting
everything else (install URLs, actions, fixes) as the bundled manifest is
upgraded. A user file for an engine with **no** bundled base of the same name
is loaded standalone (it must then be a complete manifest). This is how
`engine:set-port` persists a chosen port: it writes the minimal delta
`{ "engine": "<name>", "runtime": { "port": <n> } }`, which merges onto the
bundled manifest so only `runtime.port` is pinned. (A standalone
`LoadRegistry(dirs…)` load still replaces wholesale; the deep-merge applies to
the per-user `engines/` override layer that the service overlays at startup.)

Port changes merge only the port into an existing override. Returning to the
bundled default removes shared and host-platform port overrides while keeping
arguments, environment and unrelated settings. The file is removed only when
the engine identifier is all that remains. A host-platform port override is
updated too if needed, so it cannot shadow the saved shared port at startup.
An invalid existing override is left intact and the save fails.

### Serving declaration override

`runtime.serving` is an optional version 1 object on the authorized engine
settings channel. Its `models` array is keyed by unique, exact `modelId` values.
Each model declares positive `maxOutputTokens`, `memory.hostRamBytes` and an
explicit nonnegative `memory.vramBytes`; zero VRAM deliberately means CPU-only,
while omission means unknown and fails validation. These are conservative
operator admission declarations, not measured usage or OS allocation limits.
Integer bounds and artifact sizes must fit JSON's safe integer range.

Each `artifacts` entry records `role`, full-file `sha256`, and `provenance.kind`:
`hf-lfs-sha256`, `strata-source-verification`, `verified-manifest`,
`one-time-sha256`, or `supplied-sha256`. An optional provenance `reference`
identifies the receipt or metadata source. A filename shaped like a digest is
not proof of content. No assertion ID or separate acceptance permission exists.
An optional local `binding` records `sourcePath`, `resolvedTarget`, `sizeBytes`,
`mtimeNs` (canonical decimal int64 text to preserve nanoseconds in JSON clients),
and `sampleSha256`. Structural validation does no file I/O. The llama adapter
checks bounded samples and resolved targets during native preview, returning
candidate bindings and `expectedNativeSourceDigest`; authoritative apply
requires the exact reviewed values. A changed binding requires renewed
verification rather than silently rebinding an existing digest.

Serving settings use the existing revisioned preview/apply/journal route.
Port-only persistence preserves serving and unrelated override fields. Settings
apply writes a complete host-platform serving value; JSON null explicitly
removes an inherited declaration. Empty/missing serving retains legacy launch
behavior. Structural validation rejects unsupported versions, duplicate IDs,
invalid digests, missing resource domains and invalid bounds before mutation.
Any declaration change changes the launch record and requires a restart of a
running engine. The llama adapter supplies `configured` envelopes and matching
owned-process `launchValidated` envelopes. The existing model sweep also supplies
`observed` context/slots for freshly loaded llama models via model-specific
`/props` with `autoload=false`; cold models receive no props request. Matching
observations never imply measured memory or content qualification. Unknown or
mismatched loaded allocation withdraws identity through the run-bound drift
reason. Unload/action pokes, input withdrawal and run/config replacement
invalidate observed proof. See [loaded observation](spec.md#loaded-llama-allocation-observation)
for freshness, cancellation and the SR lookup contract. Further Strata observation
remains deferred.

Owned current launches additionally publish aggregate `contentDigest` values
and path-free envelopes in `engine:model-identity.serving`, keyed by exact model
ID. The aggregate hashes artifact role, size and asserted full SHA-256 in native
order; local paths, samples and provenance labels are excluded. Launch identity
binds the original pre-snapshot command plus the approved compiler digest, so
new snapshot generation directories do not invalidate an unchanged launch.
Inventory cannot attest NE content. Full inventory reconciliation rechecks
bounded bindings and native inputs outside locks; drift withdraws model digests
and envelopes and latches `mismatch` readback until successful explicit
apply/start. See [spec.md](spec.md#owned-serving-identity) for the wire and cold
validation contract and [qualification procedure](../../docs/local-qualification.mdx#ne-artifact-attestation).

In the pinned llama build, `sleep-idle-seconds` accepts positive integers or
exactly `-1` (disabled retention) in global/model presets, CLI and environment.
Zero and values below -1 are refused. `Retention` remains a string, including
`"-1"`. Router capacity is not compared with declaration count; SR 5b owns its
revision-bound resident-set comparison and retention enforcement.

Every action bound to `operations.load` holds the engine operation lock through
the current declaration check and dispatch, even when `serving` is absent.
Without a record, request bytes, response and dispatch error precedence retain
their existing semantics, including an already-cancelled caller once the lock
is acquired; lifecycle/settings operations serialize with the load. Ordinary
lock order is `switchAdmission` (admit, then release the gate), `opMu`, `mu`.
The admission-free declaration pre-check still rejects undeclared targets
before admission. The dispatch path acquires neither `opMu` nor `contentMu`;
any content-mutation wrapper runs before the load lock, never inside it.
The load releases its lock before restart and admission cleanup. Caller
cancellation interrupts both a queued load and held dispatch. Stop/StopAll
cancel registered loads and fence new registrations before waiting for `opMu`,
independently of the existing startup cancellation slot.
Stop wins over racing loads on the same engine with or without NE, including
command loads while the server is Off. Command actions use exec's default
cancellation to kill the command and the default five-second stop grace as
their output-pipe WaitDelay. Pipes are released after that bound; descendants
are not guaranteed to be terminated.
Paths and raw launch material belong only on the authorized settings channel.

For finite llama serving, `launch_args` must select an absolute native INI
`--models-preset` and finite positive `--models-max` (or their native environment
forms). The preset contains `[*]` defaults and exact declared model sections.
Only the finite subset in [LAUNCH_TEXT.md](LAUNCH_TEXT.md#finite-llama-router-launch)
is supported. Native CLI overrides model preset, which overrides global preset;
child environment fills otherwise unspecified source settings. PAIR's implicit
8192/1 arguments become finite preset defaults in this mode. Unsupported or
ambiguous envelope options, CLI/environment duplication, startup loading and
unprovable context allocation fail before launch. Existing models and HF
symlinks stay in place; only small immutable launch snapshots are materialized.
Recovery regenerates matching snapshot bytes and active generations survive
until their owning process exits. Manual loads use the manifest's semantic
load binding and reject undeclared models before admission/dispatch. Removing
`serving` restores the unchanged legacy launch behavior.

## Strata user-manifest template

[testdata/strata/strata.json](testdata/strata/strata.json) is a complete user
manifest template, kept as testdata rather than bundled. Copy it to `strata.json`
in the per-user `engines/` directory listed above. Set `$schema` to the absolute
location of this checkout's `manifest.schema.json` for editor validation after
copying it.

Before enabling it, replace every `/REPLACE_WITH_ABSOLUTE_PATH/` value:

- Set `runtime.bin` to the absolute Python interpreter path for the Strata
  environment (including `python.exe` on Windows).
- Set the first `runtime.args` entry to the absolute Strata `server.py` path,
  and the matching `runtime.editable_launch.fixed_args` entry to the same path,
  and the argument after `--config` to the absolute configuration file path.
  Arguments are literal strings, not shell commands; do not rely on environment
  variable or tilde expansion in them. Escape backslashes in JSON on Windows.
- Set `detect` to the absolute interpreter path, matching `runtime.bin`, and
  `detect_required` to the absolute `server.py` and configuration file paths.
  Keep `manifest_version: 2`. Detection requires the interpreter and both regular
  files. It does not import Python dependencies or validate configuration contents.
  Keep only the platform entries you configure, and select an unused runtime port.

The process command is Python plus `server.py --engine strata --config` followed
by the configuration path, then `--host 127.0.0.1 --port {port} --lazy`.
Readiness and health require `/health` to return HTTP 200 and `service:"strata"`.
Inventory uses `/v1/models` and `data[].id`; only `status.value == "loaded"`
counts as resident. Load and unload POST `{"model":"<model-id>"}` to `/v1/load`
and `/v1/unload`, with response budgets of 600 and 30 seconds respectively.
Pull, delete and model copy are unsupported. Strata is chat-only: `openai-chat`
advertises `/v1/chat/completions` and `/v1/models`, without legacy completions,
embeddings, Messages or Responses.

The fixture verifies the manifest contract against loopback HTTP fakes. Real
Strata launch, GPU behavior and cleanup still require owner-approved runtime
qualification; platform entries in this template do not establish that evidence.

An optional `runtime.serving` record enables the finite prepared adapter
described in [spec.md](spec.md#finite-prepared-strata-launch). It requires one
exact model, explicit finite context/parallel/placement, lazy native startup and
absolute existing dependencies. Every explicit file must match a declared
artifact: runtime Python/server/config/native executable/profile/tokenizer and
attested libraries are fully hashed with the existing cache; selected model
shards, pack and draft files use operator-supplied full SHA256 plus bounded local
binding checks. No full model hashing or model copying occurs during preparation.
Pack/draft enumeration is fixed; a main pack may use `native_experts.txt` instead of `experts.bin` (the table and its shards are bound; both together are ambiguous), and a draft pack may not.
The hashed config binds `lib_dirs`, whose entries must be absolute existing
directories. Directories and Python imports are not crawled. Undeclared files
found via library directories or inherited PATH/LD_LIBRARY_PATH, and Python
imports beyond declared files, remain unattested in readback. Loaded bounds and
strict content qualification remain unverified by this adapter.

## Top-level fields

For editable launch settings, `runtime.editable_launch` declares optional
`fixed_args`, command-mode `start_index` (default zero), and `controls` that
bind an engine's syntax to PAIR's policy fields. The bundled manifests reference
[manifest.schema.json](manifest.schema.json) for editor validation. Engine-manager
also validates bindings at load, including cross-control constraints that JSON
Schema does not express. Unknown properties inside `editable_launch` are errors.

```json
"editable_launch": {
  "fixed_args": ["server", "start"],
  "controls": [
    { "flags": ["--port", "-p"], "value": "{server.port}" },
    { "flags": ["--bind"], "env": ["LMS_SERVER_HOST"], "value": "{server.host}" },
    { "flags": ["--cors"], "value": "{cors.enabled}", "implicit": "true" }
  ]
}
```

| Control property | Meaning |
| --- | --- |
| `flags` | Supported flag spellings. The first is the canonical output name; the rest are aliases. |
| `env` | Supported environment names. All share the same binding and validation. |
| `value` | Format binding policy fields, with literal text between them. For example, `{server.host}:{server.port}` or `tcp://{server.host}:{server.port}`. |
| `implicit` | Value supplied by flag presence, such as `true` for `--cors` or `false` for `--no-cors`. Omit for flags that consume a value. Environment sources always consume an explicit value. |

PAIR has a small, fixed policy surface:

| Field | Policy |
| --- | --- |
| `server.port` | Integer 1–65535, synchronized with the server-port field. |
| `server.host` | Managed loopback host, keeping direct engine access behind the proxy. |
| `cors.origins` | Comma-separated origin list, normalized and validated; changes are local-only. |
| `cors.enabled` | Explicit boolean, normalized to `true` or `false`; changes are local-only. |

The **proxy port belongs to the broker**, which creates that listener and validates
port collisions. It is not an engine launch argument or an engine-manifest binding.
The presence of a CORS field identifies a browser-policy control. Engines without
one still work; proxies follow the engine's actual HTTP CORS response.

The tokenizer reads flag/environment syntax, the binding parser extracts fields,
and the field validators enforce policy. All input sources use the same conflict
checks. Launch construction renders those same bindings, with no engine names,
option-kind switches, or separate host:port parser. For example, Ollama uses
`{"env":["OLLAMA_HOST"],"value":"{server.host}:{server.port}"}`.
Bindings can reorder fields or add literal prefixes/brackets without Go changes.
Templates are trusted manifest syntax; user values are never expanded as templates.

Controls must collectively bind every managed server field. Fields within a value
must be unique, separated by literal text, and share ownership: managed server
fields cannot be combined with editable CORS fields. Unknown fields, malformed
formats, duplicate source names and invalid implicit values fail loading.
Implicit values cannot replace managed server fields. Every captured value still
passes the field validator; a format cannot weaken port ranges or loopback policy.

Managed values set every declared environment alias and, when assembling edited
arguments, emit the canonical flag. This prevents inherited network aliases from
overriding the managed values. Short value options accept separate, attached and
equals forms. Ambiguous bundles containing a declared short control are rejected.
A new engine using this grammar needs bindings and tests; a new CLI grammar needs
a shared grammar extension, not an engine-specific branch.

There is no catalog of unrelated vendor options or editable environment names.
Those values pass through literally. Explicit environment assignments override
manifest defaults; omitted defaults remain in effect. The final effective launch
is validated again before execution, including settings saved by earlier versions.

The editor saves literal extra arguments in
host-platform `runtime.launch_args` and explicit environment assignments in
`runtime.launch_env`, after trusted template expansion. Empty arrays clear the
saved arguments or environment assignments. See [LAUNCH_TEXT.md](LAUNCH_TEXT.md) for validation
and [the broker protocol](../nvpair-ui-broker/ENGINE_SETTINGS.md) for application
and recovery. Editing `args`/`start` directly remains trusted manifest authoring.

| Field | Type | Required | Notes |
|---|---|---|---|
| `engine` | string | yes | Unique id (the key used in `engine:*` calls). |
| `display_name` | string | yes | Human-friendly name. |
| `manifest_version` | int | yes | `1` or `2`. Nonempty `detect_required` requires `2`, so older managers reject the manifest instead of ignoring mandatory evidence. Higher versions are rejected. Unknown optional fields within a supported version are ignored. |
| `platforms` | object | yes | Map of `"<goos>/<goarch>"` → platform block (e.g. `"windows/amd64"`, `"darwin/arm64"`, `"linux/amd64"`). At least one entry. The runner selects the block matching the host. |
| `actions` | object | no | Map of action name → action (see below). |
| `family` | string | no | The engine's lineage, same syntax as `engine`; defaults to the engine id. It selects the coded adapters the family shares: the model-store layout used by model copy and the download catalogue (`ollama`, `lmstudio` and `llamacpp` have both). A fork with its own id declares the family it came from. |
| `protocols` | string[] | no | Protocol profiles the engine serves: `ollama-api`, `openai-v1`, `openai-chat`, `anthropic-messages`, `llamacpp-router`. `openai-chat` claims only `/v1/chat/completions` and `/v1/models`, without legacy completions or embeddings; `openai-v1` retains all of those routes. Published in the engine's descriptor; none means no inference surface is claimed. |
| `model_naming` | string | no | `exact` (default) or `implied-latest-tag` (an untagged name means `:latest`, as in Ollama). |
| `operations` | object | no | Semantic model operation bindings; see Model operations. |
| `detect` / `detect_required` / `install` / `uninstall` / `models_dir` / `runtime` | — | no | Optional **shared defaults** inherited by every platform (see below). |

**Shared defaults & per-platform overrides.** The platform-level fields `detect`, `detect_required`, `install`, `uninstall`, `models_dir`, and `runtime` may also be given once at the top level as shared defaults; each `platforms` entry is then merged onto them. Nested objects (e.g. `runtime`, `runtime.env`) merge key-by-key with the platform value winning, while arrays and scalars are replaced wholesale. So a runtime that's identical across platforms except `cli` is declared once at the top level, and each platform sets only `"runtime": { "cli": "…" }`. Omitting a key inherits the default; setting it (even to a zero value like `"port": 0`) overrides it. A manifest that fully specifies each platform with no top-level defaults behaves exactly as before.

## Platform block

| Field | Type | Required | Notes |
|---|---|---|---|
| `detect` | string[] | required with nonempty `detect_required` | Ordered executable alternatives. The first existing path becomes the launch binary. Supports OS env refs (`%VAR%`, `$VAR`), a leading `~`, and directory placeholders such as `{install_dir}`. Without mandatory evidence, existing detection and external-service adoption behavior is unchanged. |
| `detect_required` | string[] | no | All listed paths must resolve to existing regular files, together with a regular-file executable alternative from `detect`. Supports the same path expansions (including `${VAR}`). Missing or empty environment references, unresolved placeholders, directories and blank paths fail closed. These files never select the launch binary. Nonempty arrays require manifest version `2`; an omitted or empty array keeps version `1` semantics. A platform's empty array clears an inherited requirement. |
| `install` | object | no | How to obtain the engine (see Install). Omit for engines that are only ever detected/launched. |
| `uninstall` | object | no | How to remove a user-mode install (see Uninstall). Backs `engine:uninstall`. |
| `models_dir` | string | required with `uninstall.remove` | The engine's on-disk model store, available as `{models_dir}`. **Must be a literal path** — a template is rejected, so the store can never be derived from `{install_dir}`. See Model stores. |
| `runtime` | object | yes | How to launch + probe the engine (see Runtime). |

Mandatory evidence is checked on every detection/status refresh and before
start, including an already-running engine. Readiness may still establish that
a service is running, but cannot make an incomplete installation installed or
make start/install succeed. A live listener continues to block overwrite or
spawn, including for command-mode engines with mandatory evidence. An owned
process remains running and stoppable when a required file disappears.
Descriptor `available` continues to mean platform support, independently of
local installation completeness. Detection checks file presence and type only,
not file contents, dependencies or runtime qualification.

### Uninstall

| Field | Type | Required | Notes |
|---|---|---|---|
| `run` | string[] | at least one of the two | argv for work only a command can do: stopping the engine's own daemon, or invoking a vendor uninstaller. Placeholders resolved, OS env refs expanded. |
| `remove` | string[] | at least one of the two | Trees engine-manager deletes itself, skipping `models_dir`. Prefer this over an `rm -rf` in `run`. |

Both may be set, and `lmstudio.json` sets both: `run` executes first, then the
`remove` targets in order, and the pair is retried as a unit.

Use `remove` for deleting files. The runner knows what `remove` targets, so it
can keep the model store out of it; it cannot inspect a shell string, so a
deletion in `run` would bypass that guarantee. The tests scan each `run`
argument for deleting constructs (`rm -r`, `rmdir`, `rd /s`, `Remove-Item`,
`del /`) rather than just its executable name, because the uninstall this rule
exists for was an entire shell script in one `sh -c` argument.

A `remove` entry may **contain** the model store — LM Studio's engine and its
downloads both live under `~/.lmstudio` — in which case everything except the
store is removed. Rejected at load: a target that is the store, a target inside
it, a target spelled `{models_dir}`, and `remove` without a `models_dir` to
preserve.

Removal is best-effort within a tree. An engine still holding one file open
leaves that remnant behind rather than stopping the walk, and the uninstall's
verdict comes from whether a `detect` executable remains, independently of
missing mandatory files. Half-removing an
engine and then reporting failure would leave the user retrying something that
already worked. A symlinked target is unlinked, never descended into.

### Model stores

Model weights are the most expensive thing on a user's disk and the only thing
PAIR cannot replace for them. Two separate removals have to leave them alone:

1. **`engine:uninstall`**, which removes the engine. The runner skips
   `models_dir` in every `uninstall.remove` target, so this is handled as long
   as file deletion goes through `remove` rather than a command. The same
   applies to the app uninstaller's `--uninstall-managed` pass, which removes
   the engines PAIR installed and preserves every store.
2. **The app-level "remove all data" uninstall**, which deletes the entire app
   data directory. Nothing can exempt a subdirectory of it, so the only defense
   is for `models_dir` to resolve somewhere else entirely.

That second one is why `models_dir` must be a literal path: `{install_dir}` is
inside the app data root, so any store derived from it is destroyed when a user
uninstalls the app — even though the prompt promises their models are kept, and
even though no engine uninstall ever touched them.

Declare the engine's own default where it has one, so PAIR and a user's own copy
of that engine share a library rather than downloading the same weights twice.
Ollama (`~/.ollama`) and LM Studio (`~/.lmstudio/models`) both work that way:
PAIR leaves their location alone and simply names it. `~/.llamacpp` is **PAIR's
own choice** — llama.cpp defaults to a platform cache directory instead — which
is why the manifest has to set `LLAMA_CACHE` explicitly. What matters for either
kind is a stable path in the user's home, outside the app data root.

Note that a store may be broader than the weights alone. `~/.ollama` also holds
Ollama's keys and history; naming the parent protects those too, which is the
safer side to err on.

`TestBundledModelStoresOutliveTheAppDataRoot` checks every bundled manifest's
platform blocks. Its app-data-root comparison is textual, so it holds for every
target platform; the additional check against the resolved data directory
applies to the host the test runs on.

### Install

| Field | Type | Required | Notes |
|---|---|---|---|
| `fetch.url` | string | when `fetch` present | Download URL — **HTTPS** (plain `http` only from loopback). |
| `fetch.sha256` | string | no | Hex SHA-256. When set, the download is verified against it **before** `run` executes; when omitted, the fetch is HTTPS-only and runs with a loud "unpinned" warning (the same weaker guarantee as `script`). Pin it for any real release. |
| `run` | string[] | when `fetch` or nonempty `artifacts` present | Nonempty argv to execute after download (e.g. run the installer, extract the archive). Placeholders resolved; OS env refs expanded. The child also receives exact paths in `NVPAIR_INSTALL_DIR`, `NVPAIR_INSTALL_DOWNLOAD`, and `NVPAIR_INSTALL_DOWNLOAD_<ARTIFACT_NAME>` so commands that reparse argv can avoid shell quoting. Requires a `fetch` or `artifacts`. |
| `script` | string[] | no | **Escape hatch** for vendors that only ship a script installer. Runs **without** checksum verification (logged as unpinned) and replaces `fetch`+`run`. Prefer `fetch`+`run` whenever the vendor publishes a script or artifact: download it first, then execute the local file. **Make failures loud:** a piped bootstrap such as `curl … \| bash` can mask a failed fetch, while a separate fetch prevents the run and reports the error. |
| `mode` | string | no | `"user"` (default) or `"admin"`. The runner **refuses** `"admin"` (engine-manager is user-mode only); it is a deliberate, flagged exception, not a default. |

### Runtime

| Field | Type | Required | Notes |
|---|---|---|---|
| `mode` | string | no | `"process"` (default) — the engine is a foreground process this service spawns and **owns** (liveness = process alive) — or `"command"` — a daemon driven by start/stop commands (liveness = the probe). |
| `bin` | string | process mode | Path to the engine binary (required in `process` mode). Placeholders + OS env refs resolved. |
| `args` | string[] | no | Arguments (process mode). |
| `env` | object | no | Extra environment (merged over the inherited env; the `NVPAIR_PROVIDER_SECRET_` namespace is removed after the merge). Use `{host}`/`{port}` for the listen address, e.g. `"OLLAMA_HOST": "{host}:{port}"`. |
| `port` | int | no | Fixed port, or `0` to auto-assign a free loopback port. |
| `bind` | string | no | Listen address, substituted as `{host}`. Empty ⇒ `127.0.0.1` (the safe default); an inference engine that serves the cluster sets `"0.0.0.0"` (Ollama does), overridable per call via `engine:start {bind}`. Probes always target loopback. |
| `start` | string[][] | command mode | Ordered bring-up commands (each an argv) for `command` mode, e.g. `[["{cli}","server","start","--port","{port}"]]`. |
| `cli` | string | no | The engine's control-CLI path for this platform, referenced as `{cli}` by start/stop/actions so the manifest's **global** actions resolve to the correct per-OS binary. |
| `ready` | probe | no | Readiness probe; `start` waits for it before reporting Running. |
| `stop` | object | no | Process mode: `signal` (`"term"` default, or `"kill"`) + `grace_s` (seconds before force-kill). Command mode: `cmd` (argv) to bring the engine down. |
| `health` | probe | no | Periodic liveness probe while Running; `interval_s` between checks. |
| `user_credential` | bool | no | Declares that the engine accepts a user-entered API key. Defaults to false. This flag alone does not implement credential storage or RPC handling. |
| `auth_statuses` | int[] | no | HTTP statuses meaning a listener is present but requires a credential. Applies to HTTP `ready` and `health` probes; requires `ready.http`. Only `401` and `403` are allowed, without duplicates. Diagnosis only; never makes an engine ready. |
| `app_processes` | string[] | no | App or daemon process image base names, matched case-insensitively. Entries must be nonempty after trimming whitespace, contain no `/` or `\`, and have no case-insensitive duplicates. Diagnosis only; process presence never makes an engine ready. |

For process matching, the observed image's directory is removed using either
path separator, a trailing `.exe` is removed case-insensitively, and whitespace
is trimmed. Configured names use the same normalization before case-insensitive
comparison. For example, `LM Studio` matches `LM Studio.exe`. Readiness remains
the configured probe, regardless of either diagnosis field.

A **probe** is `{ "http": "<url>", "status": <int>, "timeout_s": <int>, "interval_s": <int> }`
or `{ "tcp": "<host:port>", ... }`. `status` defaults to `200`. Prefer
loopback URLs/addresses. An HTTP probe may also set
`"json_match":{"field":"service.role","value":"router"}` to require a JSON
string at that dotted object path. Missing, malformed, oversized, or
wrong-typed response data fails the probe. `json_match` is invalid on TCP
probes.

### Actions

Each action is a config-declared operation exposed over `engine:action`.
Exactly one of `http`, `cmd`, or `remove_path`:

- **`http`** — call the engine's loopback control API. The caller's
  `params` are sent as the JSON request body by default. Set
  `params_in: "query"` to require a JSON object of string values and URL-encode
  them into the query string instead. `body_schema` is informational. Requires
  the engine to be running.
- **`cmd`** — run a CLI command (e.g. `lms get`). The caller's `params`
  become placeholders (e.g. `{model}`); stdout is returned (parsed as
  JSON when it is valid JSON). Does **not** require the engine to be
  running.
- **`remove_path`** — delete a filesystem path declared in the manifest.
  `remove_path.root` and `remove_path.path` are templated (including
  action `params` as placeholders); the resolved target must stay under
  the declared root. Symlink escapes and missing targets are rejected.
  Does **not** require the engine to be running.

An optional **`result`** declares how `engine:models` extracts a normalized
name list from the action's JSON response: `array` is the top-level array
field to iterate and `field` is the string field to pull from each element.
It lets one engine-agnostic path turn each engine's `list_models` shape into a
flat `[]string` with no per-engine code — Ollama's `/api/tags`
(`{"models":[{"name":...}]}`) uses `{"array":"models","field":"name"}`, while
LM Studio's native `/api/v1/models` (`{"models":[{"key":...}]}`) uses
`{"array":"models","field":"key"}`. A required array that is present but empty
is an authoritative empty inventory; a missing, null, or wrong-typed array is
unknown, as is a non-empty unfiltered inventory with no usable names.
`engine:models` runs every running engine's `list_models` and unions the
extracted names; it also backs the node's `em` HTTP endpoint
(`GET /v1/models`) that peers fetch during discovery enrichment.

`result` also takes an optional **`match`** row filter that keeps only array
elements that pass the filter. Exactly one of `in` or `nonempty` is required:

- `{"field":...,"in":[...]}` — keep elements whose `field` (a JSON string)
  equals one of `in`.
- `{"field":...,"nonempty":true}` — keep elements whose `field` is a JSON
  array with length > 0.

This lets a `loaded_models` action reuse the same extractor as `list_models`
even when an engine's list endpoint returns *all* models tagged with residency
rather than a presence-only list. The bundled `loaded_models` actions report
the models resident in memory (surfaced as `engine:models`'s `loadedByEngine`
and pushed via `engine:models-changed`): Ollama's `GET /api/ps` is presence-only
so it needs no `match` (`{"array":"models","field":"name"}`), while LM Studio's
native `GET /api/v1/models` uses
`{"array":"models","field":"key","match":{"field":"loaded_instances","nonempty":true}}`.
A row whose `match.field` is missing or the wrong JSON type fails the match (it's
excluded). The filter shape lives in the manifest so it can be updated without a
code change if an engine's API drifts. An engine that declares no `loaded_models`
action simply contributes no `loadedByEngine` key. The bundled LM Studio list
and loaded actions require LM Studio 0.4.0 or newer, where the native
`/api/v1/models` endpoint is available. All of this stays additive:
`manifest_version` is still `1`.

An optional **`model_resolution`** expands or resolves the `{model}`
placeholder before the action runs:

- **`"lms-get"`** (cmd actions only) — try the value as given (so an
  explicit `huggingface.co`/`lmstudio.ai` URL is honored first), then the
  LM Studio Hub artifact id (`owner/name`), then the Hugging Face repo
  URL. This works around `lms get` routing a bare `owner/name` to the Hub
  registry, which 404s the Hugging-Face-hosted community models LM
  Studio's own Discover tab downloads. A **transient** download failure
  (a stalled or timed-out transfer) is *not* a resolution failure:
  rather than advancing to the next source, `lms-get` retries the same
  command in place (a bounded number of times) — `lms get` resumes a
  partial download on re-run — so a momentary network hiccup recovers
  instead of failing the pull. A hard error still surfaces immediately.
- **`"lms-disk-path"`** (remove_path actions only) — map a logical LM
  Studio model id (the OpenAI-compatible `/v1/models` id, an
  `owner/repo` pull key, or an on-disk path) to concrete files under the
  declared `remove_path.root` via `lms ls --json`. When no indexed file
  matches, the runner falls back to `{root}/{model}` if that path exists.

An optional **`restart_after`** restarts the engine once the action succeeds.
Declare it on an action that changes state the engine only reads at startup:
LM Studio's `delete_model` removes the files, but its server answers
`/v1/models` from an index built at startup and exposes no rescan operation
(no CLI command, REST route, or SDK call), so clients keep being offered a
deleted model until it restarts. It is opt-in per action and deliberately rare —
Ollama reflects a deletion immediately, so its `delete_model` omits the flag and
its engine is never bounced.

The rules the runner enforces:

- **A stopped engine stays stopped.** There is no serving state to reconcile,
  and starting one behind the user's back would be a surprise.
- **The user's ON/OFF intent is never rewritten.** The bounce goes through
  `doStop`/`doStart`, not `Restart`, precisely because `Restart` also persists
  desired-enabled — an action must not turn "delete this model" into "and leave
  this engine switched on".
- **The running check and the bounce share `opMu`.** A concurrent `Stop` either
  wins the lock first (the action then sees a stopped engine and leaves it down)
  or waits and stops the engine we just brought back. It can never be silently
  undone.
- **A failed restart fails the action**, because the action's effect is only
  observable once the engine is back. The error names the restart
  (`… failed to restart: …`) so a caller can distinguish "nothing happened" from
  "the destructive half ran and the engine did not return" — the destructive
  half is *not* rolled back, so re-issuing the action is the wrong response.
- **The platform must declare `runtime.ready`.** `Validate` rejects
  `restart_after` without one: readiness is what makes "the effect is visible by
  the time this replies" true rather than a race with the engine's own startup.
- **A remote caller waits on the readiness budget.** Driving such an action on a
  peer means the peer withholds its response headers until the restart is ready,
  so `remoteclient.go` routes that path through the readiness-sized client
  (`waitsForEngineReadiness`) rather than the ordinary 30s one. Cutting the call
  off would cancel the peer's in-flight handler and let `bringUpCommand` tear the
  engine back down — for a delete, files gone and engine left stopped.

```json
"list_models": { "http": { "method": "GET", "path": "/api/v1/models" }, "result": { "array": "models", "field": "key" } },
"pull_model":  { "cmd": ["{cli}", "get", "{model}", "--yes"], "model_resolution": "lms-get" },
"delete_model": {
  "model_resolution": "lms-disk-path",
  "remove_path": { "root": "{models_dir}", "path": "{models_dir}/{model}" },
  "restart_after": true
}
```

Actions target the engine's control plane — **not** inference endpoints
(those stay with the proxy).

## Model operations

`operations` binds the semantic model operations `load`, `unload`, `pull` and
`delete` to actions. Clients call `engine:load-model`, `engine:unload-model`,
`engine:delete-model` or a remote model endpoint with only an engine and a
model, and the binding decides how the engine is asked:

| Field | Required | Notes |
|---|---|---|
| `action` | yes | A declared action. `pull` must bind `pull_model`, which owns the streaming pull path. |
| `model_param` | no | Parameter carrying the model, default `model`: the JSON body or query key of an HTTP action, or the `{placeholder}` a `cmd` or `remove_path` action must template. Actions with `model_resolution` or `progress_protocol` read `model` and require it. |
| `params` | no | Fixed scalar parameters sent with the model, such as Ollama's `"stream": false`. Query-parameter actions take strings only. Runner placeholders and the model param are rejected. |
| `response_timeout_s` | no | How long, 1–1800 seconds, the engine may take to begin answering an HTTP action. It applies to every call of that action, `engine:action` included, and two operations sharing an action must agree on it. Not allowed on `pull`, which its progress watchdog bounds. A remote peer's load uses the readiness budget when this node binds a load budget above 30 seconds for the same engine id. |

An operation the manifest does not bind falls back to the conventional action
(`load_model`, `unload_model`, `pull_model`, `delete_model`) with the model
under `model`, when that action exists. A pull that names only a model gets its
params from the pull binding.

```json
"operations": {
  "load": { "action": "run_model", "params": { "stream": false }, "response_timeout_s": 600 },
  "unload": { "action": "unload_model", "params": { "keep_alive": 0 } },
  "pull": { "action": "pull_model", "model_param": "name" },
  "delete": { "action": "delete_model", "model_param": "name" }
}
```

## Descriptors

`engine:descriptors` publishes a sanitized projection of every registered
manifest: `engine`, `family`, `label`, `platforms`, `available`, `protocols`,
`modelNaming`, `modelStore`, `catalog`, `operations` (each with an optional
`responseTimeoutS`) and `version`. Its type, vocabulary and validation live in
`nvpair-shared/engines`. Binaries, commands, environment values, paths, URLs,
action definitions and operation parameters never reach it.

A local fork is therefore a complete user manifest with a new id, its own
`runtime.port` and `models_dir`, and the `family` it was forked from. It gets
that family's model-store layout and catalogue with no code change.

## Placeholders

Resolved by the runner at execution time; any other `{token}` fails
validation at load:

| Placeholder | Meaning | Available in |
|---|---|---|
| `{host}` | Listen address from `runtime.bind` (or a per-call `bind`); `127.0.0.1` when unset | runtime args/env |
| `{port}` | The chosen runtime port | runtime args/env, start, probes, action paths |
| `{bin}` | Resolved binary path (process mode) | runtime args/env |
| `{cli}` | The platform's `runtime.cli` path | runtime start/stop, action `cmd` |
| `{download}` | Path of the verified download | `install.run` |
| `{download_<name>}` | Path of a verified member of `install.artifacts` | `install.run` |
| `{install_dir}` | Per-engine user-scoped install dir | `detect`, `install`, runtime |
| `{models_dir}` | The platform's `models_dir`, expanded | `detect`, runtime args/env, `uninstall.run`, actions. **Not** `install`, which resolves only `{install_dir}` and the download paths — and not `uninstall.remove`, where naming the store is rejected at load. Unset when the platform declares no store, so a template referencing it then fails to resolve rather than silently becoming a path under `/` |

A `cmd` action additionally templates the action's own `params` as
placeholders (e.g. `{model}`), resolved at call time. HTTP actions send
`params` as the JSON request **body** by default; `params_in: "query"` sends
their string fields as URL-encoded query parameters with no body. They are not
substituted into `http.path`, which templates only `{port}`.

## Validation

A manifest is rejected at load (with a specific message) when: a required
field is missing, `manifest_version` is unsupported, a platform key isn't
`"<goos>/<goarch>"`, `runtime.bin` is empty in process mode (or
`runtime.start` is empty in command mode), `install.run` has neither `fetch`
nor nonempty `artifacts`, `fetch` or nonempty `artifacts` is present without
nonempty `install.run`, `install.script` is combined with
`fetch`/`artifacts`/`run`, `install.mode` or
`runtime.mode` is invalid, `models_dir` is a template rather than a literal
path, an `uninstall` block sets neither `run` nor `remove`, `uninstall.remove`
is present without a `models_dir` to preserve, an `uninstall.remove` target is
the model store or inside it (whether spelled literally or as `{models_dir}`),
an action sets none or more than one of
`http`/`cmd`/`remove_path`, a `remove_path` action omits `root` or
`path`, `http.params_in` is not `body` or `query`, a `result` is set without
both `array` and `field` (or a
`result.match` without both `match.field` and a non-empty `match.in`),
`family` is not a valid id, `protocols` or `model_naming` name an unknown
value, an `operations` binding breaks a rule in Model operations, or
a non-action templated string uses an unknown placeholder.

---

## Worked example 1 — Ollama (Linux, user-scoped, no sudo)

A conservative Ollama manifest — user-scoped install (never the official
`curl | sh`, which would `sudo`-install to `/usr` + systemd), checksum
**pinned**, server kept on loopback. (The bundled `ollama.json` is shaped
the same way but ships **unpinned** and defaults `runtime.bind` to
`0.0.0.0`; see the note after the example.)

```json
{
  "engine": "ollama",
  "display_name": "Ollama",
  "manifest_version": 1,
  "platforms": {
    "linux/amd64": {
      "detect": ["{install_dir}/bin/ollama"],
      "install": {
        "fetch": { "url": "https://ollama.com/download/ollama-linux-amd64.tgz", "sha256": "‹release sha256›" },
        "run": ["tar", "-xzf", "{download}", "-C", "{install_dir}"],
        "mode": "user"
      },
      "runtime": {
        "bin": "{install_dir}/bin/ollama",
        "args": ["serve"],
        "env": { "OLLAMA_HOST": "{host}:{port}", "LD_LIBRARY_PATH": "{install_dir}/lib/ollama" },
        "port": 11434,
        "ready":  { "http": "http://127.0.0.1:{port}/", "status": 200, "timeout_s": 30 },
        "stop":   { "signal": "term", "grace_s": 5 },
        "health": { "http": "http://127.0.0.1:{port}/", "status": 200, "interval_s": 5 }
      }
    }
  },
  "actions": {
    "list_models": { "http": { "method": "GET",  "path": "/api/tags" } },
    "pull_model":  { "http": { "method": "POST", "path": "/api/pull", "body_schema": { "name": "string" } } }
  }
}
```

> This example **pins** `sha256` (best practice). The bundled `ollama.json`
> ships **unpinned** — `fetch` has no `sha256`, so install is HTTPS-only and
> logs a loud "unpinned" warning. Pin it to the SHA-256 of a *versioned*
> release URL for reproducible, integrity-verified installs ("latest" URLs
> change checksum per release).

## Worked example 2 — LM Studio (a daemon + control-CLI engine)

LM Studio is shaped very differently from Ollama: its server runs in a
background daemon driven by the `lms` CLI, model downloads are a CLI
command (no HTTP endpoint), and its headless installer is a vendor
script. It is still added with **no code** — using `mode: "command"`, a
`cmd` action, `{cli}`, and a downloaded installer script:

```json
{
  "engine": "lmstudio",
  "display_name": "LM Studio",
  "manifest_version": 1,
  "platforms": {
    "linux/amd64": {
      "detect": ["~/.lmstudio/bin/lms"],
      "install": {
        "fetch": { "url": "https://lmstudio.ai/install.sh" },
        "run": ["bash", "{download}"],
        "mode": "user"
      },
      "models_dir": "~/.lmstudio/models",
      "uninstall": {
        "run": ["sh", "-c", "pkill -x lms 2>/dev/null; sleep 2; exit 0"],
        "remove": ["~/.lmstudio"]
      },
      "runtime": {
        "mode": "command",
        "cli": "~/.lmstudio/bin/lms",
        "port": 1235,
        "start": [["{cli}", "server", "start", "--port", "{port}"]],
        "stop":  { "cmd": ["{cli}", "server", "stop"] },
        "ready":  { "http": "http://127.0.0.1:{port}/v1/models", "status": 200, "timeout_s": 60 },
        "health": { "http": "http://127.0.0.1:{port}/v1/models", "status": 200, "interval_s": 10 }
      }
    }
  },
  "actions": {
    "list_models": { "http": { "method": "GET", "path": "/api/v1/models" }, "result": { "array": "models", "field": "key" } },
    "pull_model":  { "cmd": ["{cli}", "get", "{model}", "--yes"], "model_resolution": "lms-get" },
    "load_model":  { "cmd": ["{cli}", "load", "{model}"] }
  }
}
```

What each piece does here:
- **`mode: "command"`** — no owned process; liveness is the `/v1/models` probe, and `stop.cmd` (`lms server stop`) brings the daemon down.
- **`cmd` action** — `pull_model` shells out to `lms get {model}` (no HTTP equivalent exists), with `{model}` filled from the call's params.
- **`model_resolution: "lms-get"`** — `pull_model` tries the model as given, then the Hub artifact id, then the Hugging Face repo URL, so a bare `owner/name` for a Hugging-Face-hosted model (most of LM Studio's catalog) downloads instead of 404ing against the Hub registry.
- **`{cli}`** — the manifest-global `pull_model` action resolves to the right per-OS binary (`lms.exe` on Windows, `lms` elsewhere).
- **Installer script** — LM Studio's headless installer is vendor-provided and unpinned. Engine-manager downloads it over HTTPS and executes the saved file as a separate argument, so fetch failures stay visible and paths containing spaces cannot be split by a shell. Windows executes the saved `.ps1` with PowerShell `-File`; it never pipes remote content through `Invoke-Expression`. The vendor script checks the downloaded archive against its published SHA-512 when one is available.

---

## Engine config reference

A condensed map of each engine's manifest-relevant surface, to guide
authoring. The recurring, high-value controls (and the right targets for a
future "declared tunables" layer) are: bind host/port, model dir,
concurrency/parallelism, max-loaded models, context length, keep-alive/TTL,
GPU selection, and auth.

| Engine | Fit | Headless launch | Config surface | Control |
|---|---|---|---|---|
| **Ollama** | strong (env-first) | `ollama serve` (foreground) | env: `OLLAMA_HOST`, `OLLAMA_MODELS`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_MAX_QUEUE`, `OLLAMA_CONTEXT_LENGTH`, `OLLAMA_FLASH_ATTENTION` | HTTP `/api/tags`, `/api/pull`; CLI `ollama pull/ls/ps/stop` |
| **llama.cpp** | strong (env+flags) — **reference design** | `llama-server --host 127.0.0.1 --port {port}` (foreground) | flags + `LLAMA_ARG_*` (host/port, ctx-size, n-parallel, cont-batching, flash-attn, device, n-gpu-layers, tensor-split, main-gpu, api-key, models-dir/max) | HTTP `/v1/models`, `/models` download/delete, `/models/load`, `/models/unload`, `/health`, `/slots` |
| **LM Studio** | command / daemon | `lms daemon up` → `lms server start --port {port}` | small env (`LMS_SERVER_HOST`, `LM_API_TOKEN`); most config is flags/API/settings (`lms load --context-length/--gpu/--ttl`) | HTTP `/api/v1/models[/download\|load\|unload]`; CLI `lms ls/get/load/unload/ps` |
| **vLLM** | flags-first — **Linux/WSL only** | `vllm serve <model> --host 127.0.0.1 --port {port}` | flags (host/port/api-key); `HF_HOME` for cache. `VLLM_PORT`/`VLLM_HOST_IP` are **not** the API bind | OpenAI `/v1/models`; one model per process (unload = restart) |
| **Jan** | hybrid (on llama.cpp router) | `jan serve <model> --port {port}` (CLI) | forwards `LLAMA_ARG_*`; perf settings are router-preset-driven | `jan serve` auto-downloads HF repos |
| **GPT4All** | weak (GUI app) | none (Local API Server toggled in the GUI, `:4891`) | desktop settings / Python SDK — not env | OpenAI `/v1` once enabled in-app |

Caveats worth encoding when authoring these:
- **Model store location**: leave each engine's model cache at the engine's own default — `~/.ollama`, `~/.llamacpp`, `~/.lmstudio/models` — and declare that path as `models_dir`. Never point one inside `{install_dir}` or anywhere else under the app data root: `uninstall` skips `models_dir`, but the app-level "remove all data" uninstall deletes the entire app data root and nothing can exempt a subdirectory of it. See Model stores.
- **llama.cpp** ships prebuilt archives with **published SHA-256s**, so it's an ideal checksum-pinned `fetch`+`run` target (no placeholder SHA needed).
- **LM Studio** model dir is settings-controlled (no documented relocation env); surface its config as flags/API, not env.
- **vLLM**: native Windows is unsupported (WSL only); the bind is a flag, not env.
- **GPT4All**: no first-class headless CLI today — out of scope for a config-only manager unless targeting its Python SDK.
