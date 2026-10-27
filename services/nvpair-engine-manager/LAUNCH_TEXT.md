<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PAIR engine arguments

Typed per-model loads are declared as `runtime.modelLoads`, separately from
argument text and serving qualifications. Their `requested` leaves are portable
intent; `resolved` is target-owned finite readback, never copy input. The
[option matrix](MANIFEST.md#typed-model-load-contract-and-option-matrix) records
every native allocation entry plus generation/template/reasoning controls.
Descriptors derive native spellings from the existing adapter table. A blank
field inherits; showing a source-labelled default never saves an override.

Until the native compiler/application bodies land, RPC launch preview/configure
refuse typed loads and model changes before preparation, rebind or persistence.
Ordinary RPC writes also refuse stored typed loads while preserving direct
executor readback/persistence integration remains pending. These are temporary
bodies, not persisted feature flags. Ordinary settings with no typed load state
keep the existing path. Inspection is a read and reports unavailable artifact
facts/estimates with descriptors; Auto selects no candidate or owner defaults.
Typed requests expose no raw-extras field. Advanced launch retains its existing
supported validation.

The parent-only launch decoder accepts the broker's internal PreserveCORS guard;
public and pinned callers cannot supply it as authority. New fields are strictly
decoded within the existing 64 KiB settings budget. Direct `PreviewLaunch` and
`ConfigureLaunch` plus `launchStateLocked` and `persistRuntimeServing` still need
the same guard and modelLoads preservation at their existing owners. No restart
or drain choice is inferred while O1 remains unanswered.

The Settings editor uses this notation for local and paired-device engine
configuration. Engine-manager assembles the manifest-owned command; the broker
coordinates its saved revision, proxy port and lifecycle as one operation.

## `pair-arguments-v1` notation

The same notation is used on every target OS. It represents tokens passed
directly to the engine, not a command to paste into a system shell.

- ASCII spaces, tabs and line breaks separate unquoted tokens.
- Single quotes enclose literal text. Double quotes enclose literal text with
  only two escapes: `\"` for a quote and `\\` for a backslash. Other backslashes
  remain literal, including those in a path such as `C:\new\tools`.
- Quoted and unquoted segments may form one token: `--name="two words"`.
  Empty quoted strings represent empty arguments. Quoted line breaks are literal.
- `$HOME`, `%USERPROFILE%`, `{port}` and `~` are not expanded by the parser.
  There are no shell comments, pipelines, redirects or command substitution.
  Unquoted `|`, `&`, `;`, `<`, `>`, backticks and `$(` are rejected. Such values
  can be quoted when they are literal engine arguments.
- Input must be valid UTF-8, at most 16 KiB, with at most 256 tokens.
  Control characters other than tab, CR and LF are rejected, including NUL.
  Errors describe the syntax problem without echoing the submitted text.
- Formatting sorts explicitly configured environment assignments by name, then
  writes the arguments. The executable and startup subcommand are never editable.
  Values are quoted deterministically; inherited environment variables are never
  included. Empty text selects managed defaults. Normalized text has the same
  size bound as input.

For example:

```text
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_ORIGINS="https://example.com"
--port 1235 --bind 127.0.0.1 --future-option "two words"
```

The settings validator checks valid environment assignments, declared loopback
bind/CORS controls, managed port options and conflicts before accepting settings.
The executable and optional startup subcommand come from the trusted manifest.
The `--` terminator and vendor flags retain their token positions for that layer.
Unknown vendor option semantics are not decided by the tokenizer.

## Serving records

The settings `serving` object is independent of argument text; port or text
edits carry it unchanged. Its versioned declaration, artifact provenance and
local binding fields are described in [MANIFEST.md](MANIFEST.md#serving-declaration-override).
Preview checks structure without probing an engine, writing settings or
stopping a process. The llama adapter reads the small native preset and bounded
model samples. Invalid fields use keys such as
`serving.models.0.memory.vramBytes`. Missing serving keeps existing launches.
Changing or explicitly removing a declaration changes the launch record and
restarts a running engine through the existing settings operation. Persistence,
receipts and recovery share that operation; no second journal is introduced.

Serving readback separates `configured`, `launchValidated` and `observed`
envelopes, with `unconfigured`, `unverified` and `mismatch` status/reasons. A
llama declaration reports derived configured bounds and, for a matching owned
process, launch-validated bounds. Loaded observations, content qualification and
residency remain unverified. The desktop and terminal preserve this object
without an editor.

### Finite llama router launch

With `serving` configured, select an absolute `--models-preset` INI path and a
positive `--models-max` child capacity. Each declared model needs an exact
section and an existing local `model` path. Relative model paths resolve against
the source preset directory; snapshot paths are absolute and preserve symlinks.
Declare artifacts in main-model shard order followed by draft-model shard order.
Split GGUFs select the `-00001-of-` shard and require every remaining shard.

NE removes the bundled router's implicit context, parallelism, retention and
cache arguments. Finite defaults live in the generated preset instead:
context 8192, one slot, non-unified KV, fitting off, 300-second idle sleep and
512 MiB prompt cache. Model options override global `[*]` options; explicit
router CLI options override model options. A native environment assignment
fills a setting absent from both model and global source presets, because the
child receives preset options as CLI arguments. Preview shows the resulting
per-model allocation. Explicit user options remain in the launch text.
Supported non-envelope CLI options and `LLAMA_ARG_` environment assignments pass
through unchanged in the actual launch, including `--log-disable` and inherited
`LLAMA_ARG_LOG_COLORS`. They are not copied into generated per-model presets.

The supported subset includes context/parallelism, KV mode/per-slot cap,
prediction default, GPU layers/device/split settings, idle sleep, prompt cache,
logical/physical batch sizes, K/V cache types, Flash Attention, KV/operation
offload, local model/draft paths and disabled startup loading. A reserved
top-level `version` preset key is skipped, as the pinned native parser does.
`threads` and `threads-batch` pass through presets unchanged, like the sampling
keys. Placement stays opaque;
memory declarations are operator bounds, not inferred allocations. The adapter's
explicit envelope-option set is `llamaServingOptions` in `serving_llamacpp.go`:

- Context (`--ctx-size`/`-c`), slots (`--parallel`/`-np`), KV mode and per-slot
  cap, prediction limit, router preset and capacity.
- GPU layers, split mode, tensor split, main GPU and device (opaque placement).
- Idle sleep, cache RAM, prompt caching, idle-slot caching and fitting off.
- Logical/physical batch sizes (`--batch-size`/`-b`, `--ubatch-size`/`-ub`),
  K/V cache types (`--cache-type-k`/`-ctk`, `--cache-type-v`/`-ctv`), Flash
  Attention (`--flash-attn`/`-fa`), KV offload (`--kv-offload`/`-kvo`,
  `--no-kv-offload`/`-nkvo`) and operation offload (`--op-offload`,
  `--no-op-offload`), retained in placement entries. Batch sizes must be
  explicit positive integers; the other preset values reach the native parser
  unvalidated.
- Unsupported allocation controls: continuous batching, full SWA cache,
  context shift, RoPE/YaRN scaling and metadata overrides. The adapter does not
  interpret their allocation semantics.
- Unsupported placement controls: tensor overrides, CPU MoE/FFN,
  main-model sampling offload, weight repacking, host buffers, load/lazy modes, NUMA and RPC devices;
  draft K/V types, tensor/MoE/device/GPU/sampling placement and draft batch size;
  projector auto-selection/offload/device and image token/batch allocation.
- Unsupported startup controls: both router autoload switches, warmup switches,
  speculative-type/default/sidecar selection and built-in model presets that also set
  context, parallelism or batch sizes.
- Unsupported launch selections: model/draft paths (use local preset entries),
  model aliases, projector/LoRA/control-vector paths and control-vector layer
  placement, models directory, model URL, HF repo/file,
  Docker repo, draft HF repo, automatic-fit target/context, cache reuse,
  context checkpoints/spacing, prompt-cache files/all/read-only, slot-save path
  and startup loading.
  Every native alias and associated environment name in that set is checked.
  The table is pinned to llama.cpp `7fe450e1` `common/arg.cpp`, including its
  paired-switch `LLAMA_ARG_NO_*` compatibility names. Such negative environment
  names act by presence, even with an empty value. Positive/negative environment
  aliases together are ambiguous. Projector placement's native
  `MTMD_BACKEND_DEVICE` and internal preset startup control are also rejected.

Only envelope controls receive duplicate-alias, conflicting-value and
CLI-plus-environment rejection. Envelope forms, including underscore and mixed
underscore/hyphen aliases, are classified after
the pinned native long-option normalization. Supported normalized forms have
canonical semantics; distinct spellings for the same setting are ambiguous.
Unsupported forms,
include/inheritance/substitution, startup loads, automatic fitting and ambiguous
attached short envelope forms fail validation. Context, slots, router
capacity and supported finite numeric settings reject auto, zero, negative and
native integer overflow. An explicit CLI/preset difference follows native
precedence; an insufficient resulting allocation is rejected.

Non-unified total context must divide exactly across slots. Unified KV requires
an explicit per-slot cap and enough total capacity for all slots. With a cap
and no source context setting, total context is derived as slots times cap.
Total context, non-unified per-slot allocation and any KV cap must already be
256-token aligned, with rounding still within int32 capacity. Output must be
strictly smaller than per-slot context. KV mode and cap are
retained in the envelope's placement entries. These are cold configured bounds;
native training-context caps and actual loaded properties still require runtime
validation before qualification.

Preview returns a source approval digest and candidate artifact bindings.
Apply requires those exact reviewed values and prepares a private small
snapshot before stopping or changing settings. Start rechecks source content,
resolved targets, size, mtime and first/last 64 KiB samples immediately before
spawn. A conflict requires renewed verification and preview. No GGUF copy or
full-model hash occurs. Recovery rebuilds the same snapshot bytes from matching
inputs. An active process owns its generation until exit; replacing or reading
settings cannot remove it. Snapshot generation names are not content identities.
Pending generations are released when superseded and at orderly shutdown.
Model IDs/sections must already match native tag canonicalization; normalized
collisions are rejected. Resolved absolute model/draft paths and all generated
values must survive the native INI grammar exactly: `#`, `;`, CR/LF, NUL and
leading/trailing spaces or tabs cannot be emitted as literal values. Errors name
the model and field without exposing the path.
Undeclared manual load targets fail at the manifest-bound load action before
mutation admission or HTTP dispatch, including generic `engine:action` calls.
Authorization and dispatch serialize with settings and lifecycle replacement.
An unlocked declaration check rejects undeclared models before admission.
Restart-after actions then enter mutation admission before taking the operation
lock; the declaration is checked again under that lock and retained through
dispatch. A replacement that wins before the second check causes rejection and
releases admission. Operation-lock cleanup precedes mutation cleanup and the
restart owner, preserving `switchAdmission` before `opMu` ordering.
Without a serving declaration, custom native preset launches follow the
[custom preset rule](spec.md#custom-llama-preset-launch); launches without a
preset keep the existing 8192/1 defaults.

## Launch resolution boundary

`resolveProcessLaunch` returns the executable, argv and explicit environment
that `bringUpProcess` uses. `resolveCommandLaunch` returns the executable and
argv for each ordered command-mode start step. Existing trusted manifest
placeholder/path expansion is preserved. The builders do not mutate the
caller's resolution context.

User arguments are combined **after** trusted manifest expansion;
storing user text in `runtime.args` or `runtime.start` would interpret literal
placeholders and environment references. Settings persist literal extra tokens
in host-platform `runtime.launch_args` alongside the managed port. Explicit
editable environment assignments are stored as a replacement array in `runtime.launch_env`,
so editing or removing an assignment cannot leave an old value behind through
manifest merging. These values reach both process and command-mode launches
literally, overriding inherited values. An empty array clears prior explicit
editable assignments (inherited OS environment still applies). Environment names
are not allowlisted, but a name in the reserved provider-secret namespace
(`NVPAIR_PROVIDER_SECRET_` in any case) is rejected with an error that names the
variable and omits its value. Every process engine-manager starts, including
engines, control and install commands and port-owner lookups, gets the
inherited environment without that namespace, removed after any manifest or
settings values are merged. Explicit assignments override manifest defaults, including
values such as Linux Ollama's `LD_LIBRARY_PATH`; omitted defaults are restored
from the manifest. Existing trusted `args`,
`start`, `env`, install and other-platform overrides are preserved.

## Managed fields and preview

`runtime.editable_launch` declares fixed startup arguments, an optional
command-mode `start_index`, and field-binding `controls`. A control describes
its flag/environment names, a `value` format and an optional `implicit` value
for flag presence. Binding syntax and policy are separate: the same parser reads
all formats, while four PAIR fields enforce port, loopback and CORS policy.
See [the schema and examples](MANIFEST.md#top-level-fields).

Ollama binds `OLLAMA_HOST` to `{server.host}:{server.port}` and
`OLLAMA_ORIGINS` to `{cors.origins}`. LM Studio binds `--port`/`-p` to
`{server.port}`, `--bind`/`LMS_SERVER_HOST` to `{server.host}`, and `--cors`
to `{cors.enabled}` with an implicit value of `true`. Managed network values
replace every declared environment alias and use the first flag as canonical
output for edited arguments. All CORS fields are local-only. Proxy ports are
owned and validated by the broker, independently of engine argument syntax.
The proxy always follows standard HTTP CORS responses, including absence of
permission, independently of how an engine configures that policy.
Environment assignments such as `OLLAMA_ORIGINS="http://localhost"` can be added,
edited or removed at the beginning of the text, before engine arguments. Names
use letters, digits and underscores and start with a letter or underscore; duplicate names are invalid
(case-insensitive on Windows). Values support spaces, empty strings and the
same literal quoting as arguments. Origin lists are normalized before validation
and storage, including extra surrounding quotes, scheme/host case and whitespace.
Each nonempty origin requires a scheme and specific host; wildcard subdomains
and ports are allowed, but universal origins, credentials, paths and remaining
quotes are rejected. Conflicting repeated origin lists are invalid. Environment values remain
quoted in normalized commands. `OLLAMA_HOST` continues to synchronize with
the server-port field. The editor contains no executable/subcommand. Removing
a managed option reinserts it. Equals-form ports and repeated identical ports
normalize to one managed value; conflicting duplicate ports are invalid.
Declared short aliases accept `-p 1235`, `-p1235` and `-p=1235`. Ambiguous
short-option bundles containing managed controls are rejected without guessing
unknown vendor options' arities. CORS switches accept no attached value; `--cors=false`
is rejected rather than interpreted differently from the engine.
Disagreement with the numeric field returns an explicit conflict unless the
caller supplies `resolution: "server"` or `"launch"`.
The editor supplies that resolution automatically: a changed server-port field
updates the command; otherwise the command's port updates the field. A final
server/proxy collision is rejected with an explicit collision error dialog.

Other vendor options and environment values pass through literally, including
unknown options, short options, configuration-file options and values that the
engine may reject. PAIR maintains no vendor-option catalog and performs no
type, range or compatibility checks for them. Only the adapter-declared networking and CORS controls
receive semantic validation; vendor support is determined at startup. There are
no guessed network aliases or nested-assignment checks. An adapter must identify
the controls PAIR manages; arbitrary vendor options remain opaque.
Changes to declared CORS origin lists, switches and explicit booleans are local to the engine's device; other
environment assignments can be edited through the same local or pinned-peer API.
On startup failure, PAIR returns a bounded tail of the engine output alongside
the exit/readiness failure. It does not classify vendor messages or interpret
options. Supplied positional/assignment values and explicit environment values
are redacted by exact match; engine-transformed values cannot be guaranteed to
match. Do not put secrets in launch values. Custom output is still excluded from
ongoing exported engine logs, and desktop subprocess logs redact launch text,
argument arrays and environment maps. Successful startup discards the capture.

The effective argv and explicit environment are revalidated immediately before
execution, so saved settings from older versions cannot bypass mandatory checks
on a later start.

The wire key remains `launchText`; snapshots identify the new notation with
`format: "pair-arguments-v1"`. Older clients stay read-only for an unknown format.
Existing component `launch_args`/`launch_env` need no migration. The broker
converts saved `pair-launch-v1` journal commands through the internal preview
request before displaying or replaying them, preserving pending arguments and
ports. The legacy executable check exists only on that conversion path.

The broker-facing internal methods are `engine:get-launch`,
`engine:preview-launch`, `engine:configured-ports` and
`engine:configure-launch`. UI/headless clients use the combined broker API
documented in [ENGINE_SETTINGS.md](../nvpair-ui-broker/ENGINE_SETTINGS.md).

Tests cover grammar and normalization, exact argv received by a real isolated
child process, both lifecycle modes, preservation of override fields, reset to
defaults, platform precedence and replacement of an existing settings file.
The existing engine-manager suite also exercises the JSON-RPC stdio path.

## Networking adapter coverage

The bundled engines are Ollama, LM Studio, and llama.cpp. The shared parser
contains no engine name checks. Their mandatory networking declarations and all
platform variants are pinned by TestBundledNetworkingControls; adding an engine
requires extending that inventory and reviewing its networking alternatives.
Other settings have no catalog.

| Engine | Port/bind controls | CORS controls | Vendor reference |
| --- | --- | --- | --- |
| Ollama | OLLAMA_HOST | OLLAMA_ORIGINS, including vendor quote stripping | [Environment source](https://github.com/ollama/ollama/blob/main/envconfig/config.go) |
| LM Studio | --port / -p; --bind / LMS_SERVER_HOST | --cors (no short alias) | [Server command source](https://github.com/lmstudio-ai/lms/blob/main/src/subcommands/server.ts) |
| llama.cpp | --host; --port | --cors-origins | [Server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) |

Remote CORS comparisons use normalized policy. The broker derives the internal
preserveCORS preview guard from the authenticated caller and repeats validation
after acquiring its apply lock, before recording a new desired revision. A peer
cannot bypass this check using a stale ingress check or a client-supplied false flag.

Vendor option semantics can change independently of PAIR. New aliases or networking
controls require a manifest/test update; this is the deliberately bounded portion
of engine behavior PAIR owns.
