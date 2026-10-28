<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvpair-engine-manager

## Local engine API keys

`engine:set-credential` takes `{engine, credential}`; an empty credential
deletes it. `engine:credential-state` takes `{engine}`. Both return only
`{supported, configured}`. Support comes from the host runtime's
`user_credential` declaration (LM Studio). Unsupported state returns both
booleans false; unsupported Set, invalid names/keys and any supplied `node` or
`nodeId` field are invalid params (-32602, `Invalid engine or credential.`).
Keys contain at most 512 printable non-space ASCII bytes.

The executor lazily uses `shared/secretstore` with purpose `engine-api-key` and
the validated engine id, backed by the current user's native credential store.
There is no plaintext or environment fallback. An unavailable or locked store
returns -32000 with the fixed unavailable message documented in the spec,
without claiming the key is absent. The next
operation or health/presence sweep retries unknown or unavailable reads. A
confirmed value or absence stays cached for this executor's lifetime; external
keychain edits require restarting the service. Successful Set/Clear replaces
the cached reference only after native completion. Failed mutations retain the
previous confirmed entry but require Get before further eligibility; a failed
refresh never serves its old key. Teardown drops references without writing to
the store. Go memory is not guaranteed to be securely erased.

Candidate selection works before start and adoption, using the operation's
effective port, including a one-shot `StatusAtPort` port. An owned prepared
Strata key takes priority over a saved user key. HTTP actions and broker-only
`internal:local-engine-credential` retrieval additionally require a running
engine on its current port; adopted command-mode LM Studio may use its saved
key. Stopped engines export no key. CLI actions/start commands receive no saved
key in arguments or environment; Strata's existing private launch contract is
unchanged.

Authenticated probes and HTTP actions share endpoint validation: initial URL
must be explicit `http://127.0.0.1:<effective-port>` without userinfo, fragment
or opaque form. The private client clones a supported HTTP transport, disables
proxies and refuses every redirect. Failures carry fixed diagnostics or HTTP
status, never an engine response body or transport diagnostic. Successful
keyed action JSON field names/string values and non-JSON text redact the active
key. HTTP status and JSON identity still establish readiness; process presence
is only diagnosis. A declared auth response after attaching the saved key says
the saved key was rejected, while a known-absent key keeps the no-key diagnosis.
With no key attached, existing port handling, HTTP transport and errors remain
unchanged; credential endpoint eligibility is not applied.

Trust follows `nvpair-service`'s private same-user socket/pipe, broker stdio,
then manager stdio. Set/state have no ec route, remote method, settings/policy
relay or notification. The broker refuses client retrieval before its generic
engine relay; its own existing retrieval intentionally supplies the local
proxy. No key enters public status, inventory, diagnostics or peer payloads.
Same-user processes remain inside this trust boundary. Fake-store/HTTP and
pinned direct/reverse peer tests do not establish native store acceptance;
macOS, Windows and Linux native/session/persistence/prompt checks remain
untested.

`engine:inspect-model` calls `Executor.InspectModel` for an exact local engine/model.
It returns bounded target-local artifacts/facts plus service-owned llama.cpp
option descriptors. Inspection uses the canonical artifact resolver and GGUF
metadata reader, without engine start, HTTP inventory, download, full hashing,
native probe or settings writes. Advisory estimates use those same bounded
facts: tensor weights can be sized without an effective allocation; unresolved
cache, workspace, placement and memory telemetry stay explicit Unknowns.
`engine:remote-inspect-model` uses the existing pinned POST
`/v1/engine-settings/inspect` read route; the target broker supplies node identity.
The remote reply is strictly decoded and checked against the requested identity.
See the [option matrix](MANIFEST.md#typed-model-load-contract-and-option-matrix)
for supported B vocabulary, inherited defaults, explicit unsupported rows and
generation/template/reasoning ownership. A descriptor is not apply readiness.

The correlated shared `settings.Relay` union carries `Inspect: *InspectRequest`
for `method: inspect`, with the value `Request` empty and omitted by MarshalJSON.
Settings methods retain their value `Request` and no `Inspect`. Broker consumers
must use `settings.DecodeRelay` before selecting the branch. The executor's
existing `settingsParent(ctx, method, settings.Request, caller)` signature is
unchanged: the pinned inspect adapter carries only engine and model ID in its
private `Resolution` slot, then converts that into the actual `Inspect` pointer
before emitting `engine:settings-request`. The model ID is not serialized as a
settings resolution, and no mutation payload travels with the read. Caller
identity comes from the pinned peer transport; target forwarding is cleared.

`engine:preview-launch` accepts load-only model changes; `engine:configure-launch`
accepts parent-owned frozen settings. Unknown,
duplicate, incorrectly cased and oversized fields are invalid params (-32602).
The parent-only launch decoder still permits the broker-owned PreserveCORS
guard; public settings decoders refuse it. Existing engine counters remain exact
numeric JSON within the shared safe-integer bound; new policy revisions and byte
quantities are decimal strings. Public callers cannot supply resolved authority.

Typed llama presets compile through `compilePreparedServing` from the full
Runtime, including typed-only configurations with no serving declaration.
Fresh/manual starts and crash relaunches consume immutable generated presets.
Canonical sorted model entries, checked context arithmetic, ordered sampled
artifact bindings, native executable binding and generated bytes form the input
identity. Existing serving declarations are validated separately against those
bytes; ordinary typed loads create no memory declarations or qualifications.

Direct `PreviewLaunch` and launch readback preserve typed records. The worker's
`engine:preview-launch` accepts load-only `modelChange` edits against a fresh
target baseline. It returns selected edits or a complete-router conversion,
per-field/source diffs and a digest, refusing unsupported members without takeover.
Public preview validation admits owner-injected Auto to that compiler; it does
not invoke the no-facts resolver or read a capacity snapshot itself. With no
owner, the exact `target memory telemetry unavailable` refusal remains.
`persistRuntimeServing` accepts an optional typed collection replacement in the
same atomic port/launch/serving override write, preserving platform fields and
opaque numeric precision. These helpers do not activate a stopped engine.

The direct `ConfigureLaunch` path detects typed changes and uses the existing
stop, persist, rebind and conditional restart sequence after revalidating typed
inputs. An accepted complete-router conversion preserves the current external
artifact set and unrelated launch settings, validates the reviewed finite
resolution, and rechecks the source before stopping and restarting. `SetPort`
uses the same configuration owner for typed settings. Worker RPC accepts frozen
reviewed settings; selected-patch acceptance and coupled policy remain broker-owned.
Previews disclose whole-router scope, affected configured models, the broker's
60-second drain bound and possible interruption after timeout. Loaded-model
observations remain an explicit unknown for the broker to resolve. Off, Pause,
engine locks and adopted-process ownership keep their existing behavior.
Apply-build must add model-settings apply to its maintenance exclusion when
implemented; no exclusion symbol exists yet. Production Auto has no trusted memory
producer and refuses with `target memory telemetry unavailable`. Native
configuration does not create qualification or establish strict readiness.

Advisory memory estimates use the private MIT `gguf-parser-go` v0.27.0 pin
(`9257743a3c1775b882ba1b58ec7f5a856c2cc385`) through in-memory structs only.
Selected typed previews carry the compiler's already-decoded facts to a separate
estimator after constructing the ordinary review digest. Estimator failure,
overflow, panic, contention or budget exhaustion does not change the preview,
launch identity, requested/resolved settings or Apply behavior. No estimate
starts an engine, parses another file, writes policy, fills admission memory or
creates qualification. Embedded sampling recommendations keep their original
GGUF provenance and remain separate from settings overrides.

The [estimate support matrix](spec.md#advisory-memory-estimate-support) describes
dense/GQA, LFM2 recurrence, Qwen variants, SWA, mixed cache precision and tensor
placement. Unsupported components stay independently Unknown. Companions have
separate known storage and unsized runtime costs. Totals remain partial and
uncalibrated, with no Fits claim. CPU and unmapped offload byte projections are
not native device allocation readback. Task 10 has no runtime trusted-memory
producer, telemetry wire input or native device map. Capacity, free memory and
fit comparison remain Unknown with reason `target memory telemetry unavailable`.
The engine's private owner-injected Auto selector implements bounded comparison
and freezing with synthetic fixtures; it does not establish runtime telemetry support.
Useful component estimates remain available. Simultaneous residents are unsized
and no release credit is assumed. Auto's no-facts helper remains Unknown.

Auto searches only the existing validated bounded facts, at most 128 candidate
evaluations: highest context first, then greatest finite GPU offload. Its default
range is 8192 down to 4096 tokens with one slot; explicit ranges, fixed context,
slots, precision and other manual choices remain constraints. Context candidates
use the 256-token grid and known model context, without automatic RoPE extension.
Dense weight transfer alone does not prove total cost monotonicity. Bisection
also requires the cost owner's complete layer-independent non-weight host cost
for the exact context/options, checked against every evaluated allocation.
No production estimator supplies this proof. Without it, or on a disagreement,
the selector enumerates from greatest offload through CPU within the same budget.
Unknown costs, stale telemetry, unknown device maps
and incomplete allocations cannot fit. Exhausted searches report an unsearched
range separately from insufficient memory. The disclosed heuristic reserves
10 percent of each pool's capacity plus 512 MiB. CPU-only results disclose their
speed cost. Current estimator graph costs remain partial, including CPU scratch,
so capacity alone cannot create a production Fits result.

Reviewed Auto uses `auto/finite-llama/1`, binding requested preferences, finite
options, artifact/build identities, memory snapshot hash and any external-router
conversion source. Apply and cold restart/recovery compilation reuse that exact
finite result without consulting telemetry or searching. Artifact/build drift
refuses; native fit remains off, with no OOM retry ladder. Explicitly reselecting
Auto in a new preview is Recalculate. It can review a replacement native build
for the selected model; old Apply/restart and frozen siblings still reject drift.
A manual edit cannot silently retune saved
Auto. Conversion labels alone never authorize Auto. Manual preview and Task 10
advisory digest behavior are preserved. Broker end-to-end frozen forwarding and
recovery integration remain deferred until a trusted memory producer exists;
engine unit support does not establish end-to-end Auto Apply.

Each executor owns a derived LRU with ceilings of 128 records, 1 MiB retained
accounting and 30 seconds. Its key includes ordered sampled artifact identities,
effective allocation, estimator revision, native build/support and memory
snapshot identity (including an explicit unavailable identity). This cache is
separate from artifact inspection's cache and never replaces source rechecks.

The broker-only parent request `engine:activate-bound-engine` supplies a
zero-source bound activation with operation ID, engine/model, accepted policy
revision, manager incarnation, absolute preparation deadline and optional
verified peer ID/relationship incarnation. It has no ec endpoint. The manager
rechecks current peer relationship and management permission, policy projection,
pause and lock before admission and after waiting for the lifecycle lock.
Revocation or cancellation can end that wait before effects. Cold admission
uses the existing mutation gate to exclude competing starts, refuses a running
source, and delegates startup, precise owned reclaim and durable intent to
`StartWith`. Automatic local and peer demand both yield to local Off and pause
through existing work cancellation; persisted intent keeps its actual origin.
The wrapper never stops a source, changes launch settings or loads a model.
Local Off enters the existing Stop admission while cold startup is reserved,
persists Off before cancelling startup and retains exclusion through physical
stop. Broker retries use the private `boundOperationId` option on `engine:stop`;
the response is `{applied}` and a scope whose cold operation has ended has no
effects against a later manual On or an E4 reservation. The broker retains its
original operation lock through Off discharge and returns the ordinary status
to the local client. A transport-uncertain cold result retains its original
operation ID and blocks new activation until verified Off or manager recovery.
Server readiness does not certify the requested model; the broker verifies
fresh canonical inventory before replying to the proxy.

Internal automatic replacement reuses the E4 switch transaction. Optional
`automatic: true` in prepare/checkpoint and receipt contracts describes cause,
independently of authenticated local/remote origin. The public request has no
automatic flag. Automatic work retains its initial deadline, capped at 90 seconds.
It requires a sole running PAIR-controlled, non-adopted source with no reclaim
uncertainty, rechecked before source effects. The reservation fences other
activation. Local Off can preempt automatic work of either origin through the
same durable operation-scoped veto as remote E4. Manual local switching, locks,
pause and termination proof retain their guards. Readiness proves the server,
not model inventory. The broker owns bounded automatic receipt retention after
manager release; unresolved checkpoints remain fenced through recovery.

A config-driven control plane for local inference engines. The bundled
manifests support Ollama, LM Studio, and llama.cpp. It manages everything about
an engine **except serving inference**: detect, user-mode install,
start/stop/restart, health, and config-declared actions. Adding an engine
is a JSON manifest, not code.

The bundled manifests under `manifests/` are the working reference for manifest
authoring.

An unexpected owned-process exit with saved On intent retries through restore
after 5 seconds, 30 seconds and 2 minutes. Readiness resets that failure episode;
three unsuccessful attempts leave the existing unexpected-exit error standing.
Explicit stop, pause/sleep and shutdown cancel recovery; pending switches retain
their recovery owner and failed orphan reclaim still fences adoption.

Process-mode manifests with `runtime.port: 0` save their allocated endpoint in
the desired-state file before spawn. After a manager crash, restore uses that
port with the same managed-path and executable-image checks as fixed-port
orphan reclaim, stopping a matching orphan before launching cold. Foreign
processes are preserved. A clean stop clears the endpoint while shutdown
preserves saved On/Off intent. Older files without endpoint metadata keep their
existing restore behavior.

Reclaim also checks a saved positive PID for the same endpoint, revalidating
both its recorded executable and its current OS image against the managed
install. A matching child is stopped and its exit confirmed even before it
binds, so restore cannot race it with a cold child. Missing PID records and the
spawn-to-PID-save crash window retain listener-based reclaim. Linux image
matching resolves available symlinks and compares case-sensitively; macOS
retains canonical paths and filesystem-verified case aliases.

### llama.cpp backend checkpoint

The `llamacpp` manifest runs `llama-server` in router mode on loopback port
`8081`. It can list, download, load, unload, and delete exact model ids such as
`owner/repository:Q4_K_M`. Downloads use `/models/sse` for progress; deletion
uses the router's native `DELETE /models` cache operation and does not restart
the router. `LLAMA_CACHE` points at the manifest's `models_dir`, `~/.llamacpp`,
which no removal this service performs will delete.
Readiness requires `/props` to report `role:"router"`, so model-selection
arguments that switch `llama-server` to single-model mode and incompatible
listeners already occupying the port are rejected rather than adopted.
After five minutes without inference work, a loaded model enters llama.cpp sleep
mode and releases its model and KV-cache memory; the next request wakes it. The
router child remains alive and can retain a residual backend GPU context.

Windows and Linux installs download checksum-pinned server and CUDA-runtime
archive pairs (CUDA 12.x for x64 and CUDA 13.4 for arm64); macOS uses the
standard Metal-capable archive. These on-demand downloads are roughly
0.6–0.8 GiB and do not enlarge the PAIR installer. llama.cpp keeps its `auto`
GPU-layer policy: supported NVIDIA/Metal devices can accelerate, while the
dynamic CPU backend remains the fallback. Hardware acceptance, not `/health`
alone, is required to claim GPU activation.

## Communication

Bidirectional newline-delimited JSON-RPC 2.0 — the same conventions as
every other NVPAIR subprocess. Stdio by default; `--ipc <path>` dials a Unix
domain socket or Windows named pipe instead. The service is
**parent-agnostic**: it speaks to whatever owns its pipe (in practice
`nvpair-ui-broker`) and carries no front-end dependency, so the
same binary runs under any supervisor with zero code change. The `engine:*`
namespace is the surface the orchestrator/broker forwards from the UI.

Parent notifications have a 60-second write budget, including waiting for the
shared write gate and writing the complete frame. Policy, settings, switch,
model-mutation and resident relays use their caller's earlier cancellation or
deadline for the initial send and retain their existing reply budgets. A queued
caller cancellation emits no frame and leaves another active writer alone.
Policy/settings/switch reply cancellation removes the pending entry first,
then synchronously sends its cancellation frame with a fresh one-second budget;
cleanup failure is logged without replacing the original caller error.

An active write failure or interruption permanently retires the stream. Any
default publication failure, including a queued default timeout, logs a terminal
parent transport failure, cancels root work and closes both transport directions.
The resulting read-loop exit follows `Manager.Run`'s existing `StopAll` cleanup;
it can abort inference on PAIR-owned engines, including forced process-tree
termination on Windows. Existing workload failure/interruption paths report
interrupted inference; publication failure does not make it completed or roll
back successful engine changes. Saved intent and recovery semantics remain.
The budget bounds one send, not a whole lifecycle operation or shutdown.

State and identity publication retain their ordering lock, and content/lifecycle
serialization remains in place. Transport closure releases stalled writes so
these locks and relay slots can be released. Unix stdio owns nonblocking,
close-on-exec duplicates prepared before protocol I/O and requires successful
deadline capability probes; unsupported descriptors fail startup. Windows uses
the actual inherited pipe file owners. Closing output first interrupts the active
writer, then closing input wakes the reader. Stderr stays independent. Root
cancellation (including signals) closes the transport and joins its owner watcher.
IPC sockets/named pipes use the same terminal-close policy. No queue, retry,
delivery acknowledgement or runtime budget setting is introduced.

## JSON-RPC surface

### Automatic resident loads (internal)

`engine:reconcile-resident` is a parent-IPC-only broker operation. It consumes
the strict `nodepolicy.ReconcileResidentParams` DTO, including the current
manager incarnation, process-bound reservation, identity, policy revision,
engine-specific `declaredResidentCount` (integer 1..256, from that revision),
absolute operation budget (`timeoutMs`, at most 60 seconds), and finite
`load.contextLength` / `load.keepAliveSeconds`. Clients and peers must not
invoke it. The broker owns policy, qualification and capacity; the manager
does not infer a grant from saved intent.

The manifest must explicitly bind the semantic load operation to a `POST`
action with model parameter `model`, body schema exactly `{model:string}`,
and no fixed parameters. Loaded inventory must use a safe `GET` action with
a result extraction spec and loaded-status filter. Neither action may use
parameter placement overrides, restart, commands, remove paths, model
resolution, progress protocols or content mutation. The load sends exactly
`{model}` using the normalized installed model. `keepAliveSeconds` retains its
1..86400 validation but is not sent to the engine; approved native settings
own retention. No prompt, download, delete, copy, engine start, restore or
inference is performed. This path takes E3 admission and the lifecycle lock,
and registers with the shared load guard so settings/stop cancellation applies.

The current NE serving envelope is validated before the final grant and again
immediately before loading. Requested context must match the approved finite
per-slot allocation, native idle retention must be disabled (`-1`), and router
capacity must cover `declaredResidentCount`. That count describes this engine's
desired residents at `policyRevision`, not all serving declarations. Missing or
invalid counts fail closed before effects; automatic resident loads remain
refused until the SR 5c broker producer supplies the field. Prepare/Dispatch
DTOs and keep-alive validation are unchanged.

Before and after admission, unknown projection, either-origin pause, engine lock, switch busy,
engine Off, missing installed model, stale/withdrawn identity or unsupported
load refuse execution. Immediately before loading the manager emits
`engine:dispatch-resident-request` with `{id, params}`; the broker relays the
`node/dispatch-resident` params to the proxy and returns an internal
`engine:dispatch-resident-reply` notification with `{id, result}` or
`{id, error}`. Only a correlated live `dispatched:true` grants permission.
See [spec.md](spec.md#automatic-residency-internal-parent-ipc) for the exact hop.

The result is `not-dispatched`, `loaded`, `failed`, `timeout`, `cancelled` or
`unknown`. This DTO has no reason field; local refusal conditions and typed
proxy reasons are documented in the spec. `loaded` requires a new authoritative
post-call inventory observation with the current manager/model identity and
fresh model-specific context/slot readback matching its approved envelope;
even a lost load response can converge through that readback. Missing inventory
is unknown, while an explicit empty list is observed absence. Successful settings
re-apply alone does not prove that an already loaded child's allocation changed.
Every dispatched call pokes the loaded watcher. Desired residents never become
observed inventory or new desired-state-file fields. Duplicate operation IDs
coalesce for their original lifetime; changed payloads conflict. Idle unload protection belongs to
the broker.

Requests (caller → service):

| Method | Params | Result |
|---|---|---|
| `engine:get-installed` | — | `{ engines: [EngineStatus] }` |
| `engine:describe` | `{ engine }` | the engine's manifest (local only: it contains commands and paths) |
| `engine:descriptors` | — | `{ engines: [Descriptor] }` — every registered engine's sanitized, versioned descriptor (`nvpair-shared/engines`): `{ version, engine, family, label, platforms, available, protocols, modelNaming, modelStore?, catalog?, operations: { <load\|unload\|pull\|delete>: { responseTimeoutS? } } }`. It never carries binaries, commands, environment values, paths, URLs or credentials. See [MANIFEST.md](MANIFEST.md#descriptors) |
| `engine:status` | `{ engine }` | `EngineStatus` |
| `engine:install` | `{ engine, start?, port?, bind? }` | `EngineStatus` (after install; also starts it if `start:true`) |
| `engine:uninstall` | `{ engine }` | `EngineStatus` (after removal) |
| `engine:start` | `{ engine, port?, bind? }` | `EngineStatus` (after readiness) |
| `engine:stop` | `{ engine }` | `EngineStatus` |
| `engine:restart` | `{ engine }` | `EngineStatus` |
| `engine:set-port` | `{ engine, port }` | `EngineStatus` (after rebind) |
| `engine:set-lock` | `{ engine, locked }` | `null` acknowledgement after durable lock+Off and verified stop, or unlock without start. Local only: see [Engine lock](#engine-lock) |
| `engine:action` | `{ engine, action, params }` | the engine's raw response. `action:"pull_model"` is streamed: it emits live `engine:pull-progress` notifications and returns the pull's terminal result (see below). An action whose manifest declares `restart_after` (LM Studio's `delete_model`) restarts a running engine before replying, so the response also means the engine is back and healthy |
| `engine:logs` | `{ engine }` | `{ lines: [LogLine] }` |
| `engine:errors` | — | `{ errors: [ServiceError] }` |
| `engine:catalog` | `{ engine, platform?, arch?, query? }` | `{ models: [{ id, name, author, url, size?, downloads, likes, updatedAt, tags, family?, parameterSize?, appleOnly? }], source, platform?, arch?, fetchedAt?, searchable?, query? }` — the models an engine can **download**, as opposed to `engine:models`, which reports what is already installed. Each engine family has one curated source, selected by the manifest's `family` (a fork with `family: "llamacpp"` gets llama.cpp's); the client spellings `lm-studio`, `llama-cpp` and `llama.cpp` are accepted. Ollama's has no public API, so it is a locked list compiled into this binary from `catalog/ollama-models.json`, regenerated on demand by a developer (`npm run scrape:ollama-models` in `desktop/`) who reviews the diff; serving it needs no network. LM Studio's catalog *is* the `lmstudio-community` Hugging Face org, whose repo ids are exactly what `lms get` accepts, so it is fetched live and cached for six hours, with concurrent callers coalesced onto one request, a failed attempt — with or without a list to fall back on — followed by a backoff before the next. llama.cpp downloads any GGUF repo as `repo:quantization`; its browse list is the GGUF repos of `ggml-org`, `bartowski`, and `unsloth`, fetched and cached the same way, keeping only open, text-generating repos that ship a `Q4_K_M` file, each offered as `repo:Q4_K_M`. It is the one `searchable` source: a `query` searches all of Hugging Face under the same rules, each query cached on its own (the 20 most recently used are kept), and the reply echoes it. A query longer than 100 characters is refused rather than shortened, since a shortened one searches for something the operator did not type. Other sources ignore a query and return their whole list for the client to filter. `platform` and `arch` are the GOOS and GOARCH the models will be **installed on**, which is not always this host — a client driving a peer should name that peer. Omitting both means this host; naming a platform without an arch leaves the arch unknown rather than borrowing this host's. MLX quantizations only install on Apple Silicon, so they are marked `appleOnly` and kept only for a `darwin`/`arm64` target — an Intel Mac does not get them; the reply echoes the `platform` and `arch` it filtered for. `name` is pull-ready: it can be handed to `pull_model` verbatim. An engine with no curated source is an error, not an empty list. The Ollama reply is a single multi-megabyte frame, so every hop on its path must allow `jsonrpc.WorkerFrameBytes`. |
| `engine:models` | — | `{ models: [string], modelsByEngine: { <engine>: [string] }, loadedByEngine: { <engine>: [string] } }` — the flat de-duplicated union of every running engine's models, the per-engine breakdown keyed by engine name, and the per-engine set of models currently **loaded in memory** (all normalized from each engine's `list_models` / `loaded_models` action `result` spec). `modelsByEngine` carries a key for every running engine whose inventory was successfully queried, including an empty list = "running, no models available"; a missing key means not running / not queryable / invalid response. `loadedByEngine` uses the same known-empty distinction for residency and also omits engines with no loaded endpoint. `policy` is this node's policy advertisement (`{ revision, roles, pausedBy? }`) as the broker last pushed it with `engine:node-projection`, omitted until the first push. The `/v1/models` HTTP surface returns the same shape. |
| `engine:remote-get-installed` | `{ node }` | `{ engines: [EngineStatus] }` fetched from the remote node over `ec` mTLS |
| `engine:remote-install` | `{ node, engine, start? }` | `{ opId, status: EngineStatus }` after the remote install (live progress via `engine:remote-progress`) |
| `engine:remote-pull-model` | `{ node, engine, model?, params? }` | `{ opId, result }` after the remote pull (live progress via `engine:remote-progress`) |
| `engine:remote-copy-model` | `{ node, engine, model }` | `{ opId, result: { engine, model, files, filesSkipped, bytesTotal, bytesCopied } }` after copying the model from `node`'s engine store into this node's (live progress via `engine:remote-progress` with `op:"copy"`). See "Model copy between nodes" below |
| `engine:copy-model` | `{ opId, sourceNodeId, destinationNodeId, engine, model, expectedExportId? }` | `{ opId, result: copyModelResult }` after destination-owned verified publication |
| `engine:cancel-model-copy` | `{ opId }` | `{ cancelled: boolean }`; local channel only, exact active operation |
| `engine:active-model-copies` | `{}` | `{ operations: [{ opId, sourceNodeId, destinationNodeId, engine, model, expectedExportId?, progress? }] }`; memory-only active snapshot |
| `engine:model-copy-capabilities` | `{ nodeId, engine }` | Caller-specific versioned copy capabilities described below |
| `engine:remote-start` | `{ node, engine, port? }` | `EngineStatus` from the remote node (always the manifest's `runtime.bind`; no per-call bind override on the remote path) |
| `engine:remote-stop` | `{ node, engine }` | `EngineStatus` from the remote node |
| `engine:wake` | `{ engine }` | `EngineStatus` — starts the engine only if its saved intent is On (an engine with no saved intent counts as Off and is refused). Never changes the saved intent. Refused once shutdown has begun |
| `engine:sleep` | `{ engine }` | `EngineStatus` — stops the engine the way `engine:restart` and shutdown do, without recording Off intent, so `engine:wake` or the next launch's restore can start it again |
| `engine:intent` | — | `{ enabledByEngine: { <engine>: bool }, offByEngine: { <engine>: "local" \| "peer" } }` — one saved-intent snapshot for registered engines, without taking any engine's lifecycle lock |
| `engine:load-model` | `{ engine, model }` | the engine's raw response — loads one model through the manifest's `load` binding, the same one the remote load uses (Ollama: `run_model` with `stream:false` and a 10-minute response budget) |
| `engine:unload-model` | `{ engine, model }` | the engine's raw response — unloads one model through the manifest's `unload` binding, the same one the remote unload uses (Ollama: `keep_alive:0`) |
| `engine:delete-model` | `{ engine, model }` | the engine's raw response — deletes one model through the manifest's `delete` binding, the same one the remote delete uses. LM Studio's restarts a running server before replying |
| `engine:remote-policy-get` | `{ nodeId }` | `{ policy, availability, pausedBy? }` — the target broker's peer projection of its policy: no routing section, and tiers keep only their local entries |
| `engine:remote-policy-set` | `{ nodeId, policy, expectedPersistedRevision }` | `{ policy }` — the peer projection after the target broker merges the change into its stored policy; the revision is forwarded unchanged and compared by the broker |
| `engine:remote-availability-set` | `{ nodeId, state, pauseMode? }` | the target broker's `{ availability, pausedBy? }` result (configured pause waits for the target's drain, on the readiness-sized response budget); immediate is refused by the target |
| `shutdown` | — | `null` |
| `log/set-level` | `{ level }` | `{ level }` |

`EngineStatus` = `{ engine, display_name, installed, running, healthy, port, locked?, terminationVerified }`.
`terminationVerified:true` is positive physical-stop evidence for an engine whose
host-platform block is absent. `running:false` and descriptor `available:false`
alone carry no such evidence. Missing or false evidence is unverified. Before
each process launch the manager saves the actual endpoint and executable, then
the PID after spawn, in `engine-state.json` (without arguments or environment).
If the host block later disappears, Sleep uses that cleanup-only record and the
existing managed-install containment / exact-image reclaim predicate, never
adoption. Both endpoint refusal and recorded process exit are required; a failed
reclaim retains its PID fence even if the listener closes. Missing legacy records,
an unresolved spawn-to-PID-save crash window, foreign listeners and indeterminate
probes cannot establish stop evidence, so node pause remains draining and refuses
success. The record cannot authorize starting an unsupported engine.
`locked` is the engine's durable lock on its own node: present `true` or `false`
is authoritative, and an absent field means lock state is unavailable, as from a
target that predates it. A client never reads an absent `locked` as unlocked.

### Engine lock

`engine:set-lock {engine, locked}` is the local-only setter for an engine's
lock. `engine` must name a registered engine and `locked` must be a JSON
boolean; any other member, `node`, `nodeId`, `caller` and `origin` included, a
repeated member or a case variant such as `Locked` is `-32602`, so a request
cannot address another node, claim an origin or carry an ambiguous value. No `ec`
route, broker peer relay or `engine:remote-*` method reaches it, so a paired
node, even one with remote-management permission, can neither set nor clear a
lock. Locking atomically saves `locked[engine]:true`, saved Off and local Off
provenance in `engine-state.json`, then cancels starts, serving loads and crash
relaunch before waiting for lifecycle ownership to stop the engine. Success
requires verified termination. A stop failure returns `-32000` while retaining
the durable lock and Off and publishing actual runtime state; repeated Lock
retries the stop. A failure before file replacement changes neither intent nor
runtime. The existing atomic writer's directory-sync warning does not establish
power-loss durability. Unlock removes only the lock, leaves Off and its origin,
publishes state and requires a separate Start.

Lock/unlock commands serialize per engine through their acknowledgements. An
engine reserved by an unreleased switch refuses either command before writing,
using the existing `-32000` busy message. Checkpoints and unknown engine keys
survive lock updates. Missing lock maps in old files mean unlocked; null or
non-boolean values and locked+On contradictions fail closed. An older writer
may discard the new map; downgrading a writer after adopting locks is unsupported.
Every registered engine's status and state pushes carry authoritative `locked`
when the store is readable; a read failure leaves it absent and refuses activation.

Locked local and peer start/load/switch targets refuse with `-32013`, wire name
`engine_locked`, and `This engine is locked on this node. Unlock it on this node before starting or loading a model.` Restart, install-with-start, restore, wake,
crash relaunch, settings recovery and action restart-after use the same activation
guard; loads recheck after lifecycle waiting and late readiness cannot save On
over a lock. Metadata-only settings remain available and cannot restart a locked
engine. Sleep, stop and unload never unlock it. Switching away from a locked
source retains ordinary source ownership checks; the target must be unlocked.

`engine:wake`, `engine:sleep` and `engine:intent` exist for the broker's node
policy (pause, idle stop, start on demand). `engine:start` and `engine:stop`
remain the only operations that record what the user wants; wake and sleep take
the engine's lifecycle lock like `engine:restart` but only act on that record.

`engine:intent` and `engine:intent-changed` share the `engineintent.Snapshot`
wire type. Both maps are always present and derived from one desired-store read.
`enabledByEngine` includes every registered engine; never-persisted intent is
`false`. `offByEngine` includes only registered engines explicitly saved Off:
`local` means this device's client saved Off, and `peer` means a paired device
did. Legacy saved Off without stored provenance projects `local`; unknown and
On engines have no origin entry. Stale origins for other engines are excluded.
Invalid saved Off origins and unreadable state fail the read; no guessed
snapshot is pushed. Reading does not write or migrate the desired-state file
(whose private provenance field remains `offBy`). Origins are display data,
independent of runtime status and proxy admission booleans.

Notifications (service → caller): `engine:ready{version}`,
`engine:state-changed{EngineStatus}`,
`engine:model-identity{engines}` (a whole snapshot keyed by engine, each entry
`{engineRunId, launchDigest, modelDigests}` with `modelDigests` keyed by model
name; emitted after the loaded-model watcher's first sweep and whenever the
snapshot changes; absent engines have no current identity. Only running owned
processes with a known launch digest are included, excluding adopted services
and command-mode daemons; model digests must come from an inventory of the same
run and contain lowercase SHA-256 hex),
`engine:models-changed{engine, models}` — pushed when an engine's set of
loaded (in-memory) models changes (explicit load/unload, JIT auto-load, or
TTL/idle eviction); `models` is the full `engine:models` shape (incl.
`loadedByEngine`) so a consumer swaps its whole snapshot,
`engine:install-progress{engine, stage, percent?, error?}` (`error` on the
terminal `failed` frame),
`engine:pull-progress{engine, op, stage, percent?, message}` (live progress for a
local model pull driven via `engine:action{action:"pull_model"}` — the local
counterpart of `engine:remote-progress`; frames are coalesced to changes in
stage/percent, the engine's terminal success surfaces as `stage:"success"`, and
a failed pull emits a terminal `stage:"error", percent:-1, message` frame so a
UI converges even if its synchronous call already timed out),
`engine:remote-progress{opId, node, engine, op, stage, percent?, message}`
(relayed live progress for a remote install/pull; a model copy adds
`file?, bytesDone?, bytesTotal?`),
`engine:intent-changed{enabledByEngine, offByEngine}` (pushed after an explicit start, stop,
restart or install-and-start changes saved intent; wake and sleep never do), and
`policy:request{id, method, caller, params}` / `policy:cancel{id}` (a paired
node's node-policy call relayed to the broker, answered by the broker's
`policy:reply{id, result, error?, errorCode?}`; see below), and — for the error
pipeline — `errors:report` / `errors:clear` (consumed by `nvpair-errors`
via the broker; see below).

Notifications (caller → service): `engine:node-projection{policy, pausedLocally, managementBlocked}`
is this node's policy advertisement (declared roles with their profile bounds,
policy revision, pause origin), whether the node is paused locally, and
whether a pause of either origin, pending, draining or saved, fences peer
management writes. An older broker omits `managementBlocked`, which then reads
as false; it is never derived from `pausedBy` or `pausedLocally`. The
broker sends it after every persisted policy or availability change and on
every `engine:ready`. engine-manager keeps the last one in memory only and
serves `policy` on `engine:models` and `GET /v1/models`, so paired peers learn
this node's roles from the pinned inventory fetch they already make. It is
never put in mDNS TXT or in the peer-services descriptor. `pausedLocally` remains
provenance. `managementBlocked` fences all peer mutations and local activation,
and either-origin pause cancels accepted starts, loads and crash relaunch.

Streaming HTTP pulls have a 30-minute **inactivity** watchdog. Each increase in
an Ollama layer's or llama.cpp file's completed byte count refreshes that
watchdog, so an active download may run longer than 30 minutes; duplicate
progress frames and heartbeats do not extend a stalled pull. CLI-driven pulls
without structured byte progress retain the fixed 30-minute action timeout.

For llama.cpp, an accepted pull that ends before a matching `download_finished`
or `download_failed` event stops the active download. This includes caller
cancellation, remote caller disconnect, inactivity timeout, and premature SSE
termination. Cleanup checks `GET /models` and sends `POST /models/unload` only
when the exact model is still `downloading`; cached files are retained, and
models that have already completed are left alone.

The initial `POST /models` handshake has a separate 30-second total timeout and
continues through caller cancellation so its acceptance can still be read.
The pull then waits for cleanup, which has a separate five-second budget for
the inventory check and stop request together. A failed cleanup reports that
the download could not be confirmed stopped alongside the original pull error.
If the start acknowledgement is lost or unreadable, acceptance and cancellation
are reported as unconfirmed without unloading an unowned download. The monitored
model must match `params.model`; mismatches are rejected before subscribing.

`percent` on all three progress notifications is present only when the step is
measurable (download bytes, byte-progress pulls) or terminal (`100` on install
`done` / `already-installed`, `-1` on install `failed` and pull `error`). An
indeterminate step omits it — the install `verified` and `installing` stages
always do — so a UI renders the stage alone.

The `engine:remote-*` methods are the client half of remote engine
management: engine-manager resolves the target `node` in an `ec` peer
directory (fed by its own `discovery:subscribe{services:[ec]}` to the broker
relay), dials that peer's `ec` surface over pin-based cluster mTLS, and — for
`remote-install` / `remote-pull-model` — mints an `opId`, relays each streamed
progress frame up as `engine:remote-progress`, and settles the request with the
terminal result. They fail if this node isn't clustered or the target isn't a
pinned cluster peer. See "Remote engine management" below.

`engine:install`, `engine:start`, `engine:stop`, `engine:restart`,
`engine:set-port`, and `engine:action` run in their own goroutine on the
service side, so the read loop never blocks and their responses arrive when
the op finishes.

`engine:set-port` is the **persistent** port setter (distinct from the
one-shot `engine:start {port}` override, which reverts on the next restart).
It validates `1-65535`, persists the choice as a manifest override (a
`{ engine, runtime: { port } }` delta written
to the per-user `engines/` dir that deep-merges onto the bundled manifest, so
`runtime.port` becomes the single source of truth and the port is restored on
the next start with no separate store), and applies it — bouncing the engine
onto the new port if it was running. Setting the port back to the bundled
default removes only the shared and host-platform port overrides; the file is
removed only when no other overrides remain. Arguments, environment, install
settings and other platform overrides are preserved. A host-platform port is
updated when necessary so it cannot shadow the saved value on restart.
Malformed override files fail the save instead of being replaced.
Because the chosen port lives in the
effective manifest, restore is automatic: a normal `engine:start` (no explicit
port) and the adopt-on-fixed-port path both come up on the retained port.
Moving a **running, adopted** engine is **refused** with an error (nothing is
persisted) — NVPAIR can't relocate a process it didn't start; see Adoption below.

## Lifecycle

Combined launch/server/proxy edits use the broker's
[engine settings protocol](../nvpair-ui-broker/ENGINE_SETTINGS.md). The worker
checks basic argument syntax and adapter-declared networking/CORS controls
using [`pair-arguments-v1`](LAUNCH_TEXT.md), persists host-platform `launch_args` and `launch_env`
with the port, and holds its operation lock through stop/rebind/start. The
paired engine-control surface relays settings to its local broker and streams
full authoritative snapshots to pinned peers. Outbound settings subscriptions
require current peer-service admission, which excludes this node's own cluster
principal even when discovery gives it a different host UUID. The discovery
directory retains self for model-copy identity resolution; mesh self-trust is
not permission to start a peer subscription. Failed peer streams retry after
three seconds.

For custom llama.cpp launches using native `--models-preset` (including the
equals form), model presets own context, parallelism, idle retention and cache
RAM. Engine-manager omits the manifest's fixed `--ctx-size`, `--parallel`,
`--sleep-idle-seconds` and `--cache-ram` defaults. Explicit user options still
override the preset without a duplicate manifest default. Other fixed startup
options remain, unless explicitly supplied by the user; launches without a
preset retain their existing fixed arguments.

```
NotInstalled --engine:install--> (HTTPS download + verify-if-pinned + user-mode run) --> Stopped
Stopped      --engine:start----> (adopt if already serving the port, else spawn) --> Running --health--> Running
Running      --engine:stop-----> (stop signal, bounded grace, force if needed) --> Stopped
```

Detect uses the manifest's `detect` paths. Install is one-shot and
user-mode — an HTTPS download, verified against the manifest's `sha256`
when one is pinned (an unpinned fetch runs with a loud warning). Start
waits for the readiness probe, then runs a periodic health probe; an
unexpected exit is reported. The bundled Ollama manifest allows up to ten
minutes for startup because GPU discovery can exceed the previous 30-second
allowance on supported Windows systems. The deadline remains finite: if Ollama
never serves its readiness endpoint, engine-manager stops the owned process and
reports the failed start. On Unix, stopping an owned process sends SIGTERM to
its process group, waits the manifest's `stop.grace_s` (five seconds by
default), then escalates to SIGKILL; `signal:"kill"` skips the grace. Failed
startup cleanup uses the same policy. On Windows, the windowless engines we
spawn cannot receive a graceful (non-`/F`) close, so stopping uses immediate
`taskkill /T /F`.

Unix signals target the whole process group only when the target PID is its
group leader; a nonleader is signalled alone. This also applies to orphan
reclaim, so a shared group retains its unrelated processes.

For command-mode engines, any attempted start command may have launched a
detached daemon before returning. If startup fails or is cancelled, including
during shutdown, cleanup runs the manifest's bounded stop command and confirms
that the listener closes using the configured readiness endpoint before
releasing the lifecycle lock. This also applies when the start CLI itself
fails or is cancelled before returning.

### Adoption — start may attach to an engine it didn't launch

Before spawning, `engine:start` **probes the chosen port's readiness
endpoint**, including any manifest-declared JSON identity. If a compatible
service answers there — the engine's own desktop
app (e.g. the Ollama tray app on `11434`), or an instance left running from a
previous session — the service **adopts** that instance: it marks the engine
`running` without launching its own, rather than spawning a duplicate that
would only collide on the port. Consequences worth knowing:

- **An adopted engine has no child process the service owns**, so `engine:stop`
  (and `engine:restart`'s stop phase, and shutdown's cleanup) resolves the PID
  bound to the engine's port and terminates it **only when that process is
  running the very binary NVPAIR manages for the engine** — reclaiming an orphan
  a prior run left on our own managed port (e.g. an `ollama serve` on `11435`
  whose handle was lost after a crash). This is precise to the port, so a
  genuine third-party listener on a *different* port (Ollama's own desktop app
  on `11434` while NVPAIR manages `11435`) is never touched. A listener whose
  image is **not** our managed binary is declined with an actionable error
  naming the offending PID and image path — the user / desktop app owns that
  process, and NVPAIR won't terminate it out from under them.
- **Executable inspection fails closed.** Linux reads `/proc/<pid>/exe`;
  macOS uses the kernel's `proc_pidpath` with a fixed-size buffer, without
  trusting command-line text. Port lookup is bounded and refuses ambiguous
  owners. An unavailable or empty first lookup permits another provider;
  malformed, incomplete or conflicting ownership remains unknown even if that
  provider reports one PID. macOS resolves filesystem aliases such as `/var`
  and `/private/var` before matching; an executable symlink escaping the managed
  install directory does not confer ownership. Unknown images are refused before signaling or
  uninstalling files.
  Configured filenames are literal. A ` (deleted)` suffix is interpreted as
  Linux process-image metadata only when that full spelling does not exist;
  an existing executable with that literal suffix retains its distinct identity.
  Linux confirms an existing suffixed file against the `/proc` executable's
  file identity; a different file at that spelling makes inspection unknown.
- **`engine:stop` may return an error while still saving OFF.** When stop
  declines a foreign listener it returns an actionable error, but the user's
  OFF choice is persisted anyway — UI layers should treat the saved desired
  state as authoritative (the engine will not restore on restart) and surface
  the error as guidance, not as proof the OFF intent was lost. The same applies
  to cluster `POST /v1/engines/stop`, which may answer HTTP 500 even though OFF
  was recorded. A local stop records local provenance and cancels accepted peer
  activation; a cluster stop records peer provenance. Either ordinary Off permits
  a fresh explicitly authorized peer start or switch.
- **`engine:set-port` on a running adopted engine is refused** (returns an
  error, persists nothing). Moving it would mean killing the old listener and
  spawning a new one; since NVPAIR can't kill what it didn't start, it errors
  rather than leaving a duplicate serving the new port while the original keeps
  serving the old one. Stop the engine in its own app first, then set the port.
- **`engine:install` short-circuits the same way.** `detect` honors existing
  system installs (Ollama's detect paths include `/Applications/Ollama.app`,
  `%LOCALAPPDATA%\Programs\Ollama`, etc.), so "installing" an engine that's
  already present downloads nothing and reports `installed: true` — and a
  following `start:true` then adopts the running instance.
- **Liveness reconciliation uses the same probe.** A fixed-port engine found
  already serving is reported `running: true` even though NVPAIR never started it.

To get a **NVPAIR-owned, stoppable** instance, start it on a port nothing is
already serving — the probe misses, so the service spawns and owns the child
(tracked in `st.proc`), and a later `engine:stop` / `engine:restart` actually
terminates it. `engine:set-port` to a free port does exactly this; quitting the
external app first and re-starting on its usual port works too. Auto-assigned
ports (manifest `runtime.port: 0`) never adopt — there's no fixed address to
probe — so they always spawn an owned process.

## Remote engine management

When the parent passes `--control-port`, engine-manager serves the **`ec`
surface** — a cluster-scoped remote-control endpoint over **pin-based mutual
TLS** (`nvpair-shared/clustertrust`, the same identity + per-peer pins
`nvpair-cluster-manager` mints). It presents this node's cluster leaf, requires a
client cert, and rejects any caller that isn't a byte-for-byte pinned cluster
peer with a `403`. It differs from `em` (`--http-port`) only in what it permits:
`ec` performs privileged operations, while `em` is a read-only inventory. Both are
locked to pinned cluster peers on the LAN — a node's model list tells a caller
which models that machine holds, which is cluster data like any other. `em` also
keeps a plaintext personality on loopback, because this node's own scanner reads
its own inventory that way and must be able to while unclustered.

Membership is evaluated **live**, per handshake and per request, from
`--cluster-dir`. While this node is not a cluster member it presents no leaf, so
every handshake is refused and nothing privileged is reachable; the moment it
becomes a member the same listener serves pinned peers. The listener is therefore
bound for the life of the process rather than only when clustered at startup — a
node joins and leaves a cluster while engine-manager runs, and a surface chosen
at bind time would stay dark until the process was restarted.

`--control-host` selects the literal IPv4 or IPv6 bind address for `ec` only;
its default remains `0.0.0.0`. Isolated fixtures can pass `127.0.0.1` (or `::1`
where available). Empty values, DNS names, bracketed addresses, zone suffixes
and host/port pairs are rejected before startup, without a fallback address.
The broker forwards `--engine-manager-control-host` to this flag. This does
not isolate the whole process: `em` (`--http-port`) still binds independently
on `0.0.0.0`, and fixtures must disable or independently isolate it and all
other listeners. Control mTLS, membership, pin and management checks apply
unchanged on loopback.

Pinning authenticates the caller; it does not authorize changes. The `ec`
registry explicitly classifies every route as read or mutation, including POST
reads such as policy get and settings get/preview. Before dispatching any
mutation, the common gate freshly reads
`clustertrust.Mesh.RemoteManagement` for the verified mTLS UUID. Install, pull,
start/stop, load/unload/delete, settings apply, switch, policy edits and
pause/resume all require an effective allow. Unclassified routes fail closed.
Missing, corrupt, pending or changing permission state denies; an earlier UI
read never authorizes a request. Broker-owned writes recheck at their locked
acceptance point after queueing. Direct operations accept at this gate; later
lifecycle/content lock waits are part of that accepted operation.

Permission refusal is HTTP 409 with `code:"remote_management_denied"`, preserved
as JSON-RPC `-32012` by ordinary and streaming clients, before progress headers
or side effects. It is distinct from an unpinned 403, local veto
`-32011/local_veto`, engine lock `-32013/engine_locked` and conflict `-32010`.
Every one of these typed codes reaches the origin's JSON-RPC client with its own
number, through the 409 body, a stream error frame and the broker relays alike,
with the target's message and no JSON-RPC error `data`. Reads, status, model exports and
permitted inference stay available without management permission. Pull-copy
reads the source and writes the local requester; it grants no remote destination
write authority. No inbound destination-copy route exists.

`GET /v1/remote-management` returns only this authenticated caller's
`{version:1, allowed, revision, state}`, without a peer selector or grant list.
The local client method is `engine:remote-management-get {nodeId}` using the
existing pinned direct/reverse client. States are `allowed`, `denied`,
`unreviewed`, `unavailable`, or client-reported `unsupported` for an old target's
404/405. Only a fresh `allowed` result reports permission; it is not a promise
that a later mutation will pass. There is no remote setter. Grants and the
one-time existing-pin migration belong to
[cluster-manager](../nvpair-cluster-manager/README.md#remote-management-permission).

Revocation blocks new acceptance, not already accepted target work, which can
finish/reconcile and remains readable through existing progress/receipts. Local
pause, lock and operation-scoped local-stop cancellation still apply; grants cannot override them or private,
CORS, served/resident and admission protections. Downgrading the target to
pre-PG software restores that executable's old pinned-peer behavior. The gate
logs caller/target UUID, bounded operation family/class, decision, permission
revision and reason, without paths, queries, bodies or secrets.

Endpoints (all under `/v1`):

| Route | Shape | Purpose |
|---|---|---|
| `GET /v1/engines` | JSON | remote `engine:get-installed` |
| `GET /v1/remote-management` | JSON | caller-only effective permission, no grant list or setter |
| `POST /v1/engines/install` | NDJSON stream | remote install (+ optional start) with live progress |
| `POST /v1/models/pull` | NDJSON stream | remote model pull with live progress |
| `POST /v1/engines/start` | JSON | remote start → `EngineStatus` |
| `POST /v1/engines/stop` | JSON | remote stop → `EngineStatus` |
| `GET /v1/models/files?engine=&model=` | NDJSON stream | the model's files with size and sha256, for a peer copying it |
| `GET /v1/models/file?engine=&model=&path=` | file bytes | one listed file, with HTTP Range support |
| `POST /v1/node-policy/get` | JSON | the target broker's peer projection of its policy, as for `engine:remote-policy-get` |
| `POST /v1/node-policy/set` | JSON `{ policy, expectedPersistedRevision }` | the peer projection after the target broker merges the change; the revision is forwarded unchanged |
| `POST /v1/node-availability/set` | JSON `{ state, pauseMode? }` | the target broker's `node:set-availability`, answered once the configured state is reached; peer immediate requests are refused |

The node-policy routes carry the broker's policy, which engine-manager does not
interpret. Each is gated on the caller's pin like the settings routes, accepts at
most 256 KiB, refuses unknown fields, clears any `nodeId` so the target acts on
itself and never forwards a further hop, and relays to the broker as
`policy:request` stamped with the authenticated caller. The broker re-checks the
pin, acts, and answers with `policy:reply {id, result, error?, errorCode?}`.
`pauseMode` is carried unchanged, including explicit null, for target-side
validation after fresh permission and pause-fence checks. Omission means
`configured`; `immediate` is target-local only, so an authorized unfenced peer
receives `-32602`, `Immediate pause must be requested on this node.` No payload
`nodeId` can impersonate locality. Any mode on Resume is invalid. Local
immediate calls start cancellation and controlled shutdown without the drain
wait; this peer relay cannot invoke them. Timing is never saved. Results preserve actual target
`pausedBy` (`local`/`peer`, unknown empty/omitted, cleared while available),
without exposing local activation or origin-private routing fields.
Every error carries a JSON-RPC `errorCode`: `-32602` for invalid parameters,
including missing `expectedPersistedRevision`, invalid policy or availability;
`-32010` for a revision conflict; `-32011` for a resume the local veto refuses;
`-32012` for remote-management denial; `-32013` for an engine locked on this
node; `-32014` for target-owned profile inference settings;
`-32000` for a save failure or other operational failure. The route maps
`-32602` to HTTP 400, `-32010`, `-32011`, `-32012`, `-32013` and `-32014` to HTTP 409 with the body
`{"error": <diagnostic>, "code": "policy_conflict" | "local_veto" | "remote_management_denied" | "engine_locked" | "remote_setting_locked"}`, and other
failures to a plain-text HTTP 409. The origin's remote-policy methods map HTTP
400 to JSON-RPC `-32602`, a typed 409 to its code, and other HTTP failures to
`-32000`, preserving the diagnostic. No classification depends
on diagnostic text. A stale policy draft is a conflict before its policy is
validated, including a draft that tries to enable remote or paid execution.
The route abandons its wait (and sends
`policy:cancel`) if the caller loses its pin or 15 minutes pass; a pause the broker
already accepted still finishes.

`remote_setting_locked` preserves the target broker's safe bounded explanation
and maps back to `-32014` at the origin. It names only the stored profile's
engine/model; a new-profile lock attempt receives a fixed generic message.
There is no identifier payload or error data, and classification never parses
the message. The broker, not this relay, owns the latest stored `localOnly`
check after permission and revision checks. A refused write has no partial
policy, persistence, activation, push or notification effects. See the
[broker contract](../nvpair-ui-broker/README.md#model-inference-settings-and-target-owned-locks).

**Off, lock and pause.** Ordinary Off, whether local, peer, legacy or unknown,
does not prohibit a new granted peer start, load or switch. Load never implicitly
starts an Off engine: its real stopped-runtime error remains an ordinary error.
Automatic restore/wake/residency still require saved On. `offBy` retains diagnostic
and operation provenance: a peer Off never downgrades existing Off provenance,
and On clears it. A new local Stop can cancel an already accepted operation with
`-32011/local_veto`, including its durable switch cleanup obligations.

After a fresh grant check, the common `ec` registry gate returns HTTP 503 for
every mutation until the first projection. A known `managementBlocked` returns
HTTP 409 `{"code":"local_veto","error":"This node is paused. Resume it on this node before managing it remotely."}`.
This covers install, start, stop, load, unload, delete, pull, settings apply,
switch apply, policy set and availability set, including peer Resume and repeated
Pause by the initiating peer. Reads, read-only POSTs, events, model files and HEAD
remain admitted. Grant denial keeps `-32012` precedence; pause precedes lock for
granted peers. Only a local Resume clears pause.

Local start/load/switch also refuse known pause; local reads, configuration,
lock/unlock, stop and unload remain usable. Accepted starts/loads are canceled
for either pause origin, and settings/recovery cannot reactivate an engine.
Accepted downloads may finish without activation. While available, lock permits
granted stop/unload/delete/pull/install-without-start and settings edits, but any
activation side effect still refuses. HTTP coded bodies, stream terminal errors,
JSON-RPC relays and switch receipts preserve `-32011` versus `-32013` and the
target's message; lock switch receipts use kind `engine_locked`.

The streaming routes emit zero or more `{"type":"progress",...}` frames followed
by exactly one terminal `{"type":"result",...}` or `{"type":"error",...}` frame.
The initiating node's engine-manager consumes that stream, relays each progress
frame up as `engine:remote-progress`, and settles the originating
`engine:remote-*` request on the terminal frame — so a UI gets one synchronous
response plus a live progress feed keyed by `opId`.

The broker wires both directions: it passes `--control-port`/`--cluster-dir`,
registers `ec` with the discovery daemon whenever a cluster dir is configured, and
relays engine-manager's `discovery:subscribe{services:[ec]}` into the relay
directory so the peer directory stays current. It does **not** restart
engine-manager on `cluster:identity-changed` — the surface follows membership on
its own.

## Model copy between nodes

`engine:copy-model {opId, sourceNodeId, destinationNodeId, engine, model, expectedExportId?}` asks the selected destination to fetch directly from the
selected source. Device selectors are stable host UUIDs; `engine` is the exact
manager ID, including custom IDs. The controller carries only control and
progress. Its model store is never opened for a copy between other devices.
`engine:remote-copy-model {node, engine, model}` mints an operation ID and
invokes this same coordinator with `node` as source and this device as destination.
The wrapper retains its original field-validation order and drops unrelated
parameters when constructing the canonical tuple.

Canonical RPC and inbound HTTP copy requests accept exact, unique field names
and valid model identifiers, with a complete 16 KiB control-body limit. Operation IDs are 32 lowercase hex
characters. Identical endpoints, unknown fields (including caller, URL and
path overrides), duplicate active IDs and excess active work are refused before
peer lookup or transfer. Each Manager owns at most eight active reservations,
including inbound copies; there is no queue or terminal history. Cancellation
and shutdown close the exact request context and retain resumable partials.
Closing a GUI subscriber alone does not cancel a copy. A reconnect snapshot
that no longer contains an operation establishes an unknown outcome, not success.

The destination route is pinned `POST /v1/models/copy`, classified as a
mutation. Its body contains only `opId`, `sourceNodeId`, `engine`, `model` and
optional `expectedExportId`. The destination rechecks the authenticated
controller's grant and its pause/recovery gates while holding a reservation.
The destination independently needs a live source pin; controller trust cannot
substitute for that relationship. Copy does not install or activate an engine,
and does not require Running or override saved Off/locks. A dispatched copy
POST is never replayed through an alternate route.

`engine:model-copy-capabilities {nodeId, engine}` reads the actual node's
`{version:1, engine, available, installed, modelStore, exportLayouts, importLayouts, writable, reason, identityBoundRetry, receiptDurability}`.
The peer route is pinned `GET /v1/models/copy-capabilities?engine=...`;
`writable` reflects readiness and the actual caller's target-local permission.
Adapter names are `ollama`, `llamacpp`, and `lmstudio`; llama.cpp also supports
`llama-artifacts-v1`. No filesystem paths or grant lists are returned. Missing
routes/versions are unsupported. Source and destination layouts must intersect,
the selected export must be supported by both, and the destination must be installed.
The capabilities GET has a 30-second context budget covering both response
headers and the complete bounded body, shortened by an earlier caller deadline.

An explicit retry can bind `expectedExportId` to a learned custom export.
After destination authorization/admission, a matching committed receipt is
fully rehashed before any source lookup or capability request. A verified
receipt can finish with the source offline, reporting `durability:"uncertain"`.
Otherwise the source must be live and its export identity must still match
before writes. Without an export ID, retry resolves the alias afresh and cannot
guarantee the same content after a lost response. Timeouts are never success.

The shared pinned peer/service adapter owns both listing and file transport.
It resolves the current admitted source and advertised `ec` endpoint, prefers
direct mTLS, and can use the existing authenticated reverse session after a
bounded failure before HTTP dispatch. Inner mTLS still reaches the source's
ordinary pin-gated control handlers. Listing frames and file bytes stream with
backpressure and cancellation; the adapter adds no listener, full-file buffer,
or replay after dispatch or response exposure. A broken file response ends that
HTTP attempt. The copy owner below issues a new Range request from the retained
partial's current length and checks the completed file's hash; transport does
not invent a second resume policy.

On the source, `GET /v1/models/files` resolves the model's files from the
engine's on-disk layout and lists them as paths relative to the engine's store,
with size and sha256. Hashes are computed when first asked for, by streaming
the file, and cached by absolute path, size and modification time. Files whose
name is already their sha256 (Ollama blobs, llama.cpp blobs behind snapshot
links) are not re-read on the source; the destination still verifies them. The
listing is an NDJSON stream in the shape of the other streaming routes:
`{"type":"progress","stage":"hashing","message":<path>,"percent"?}` frames
while files are hashed, then one `{"type":"result","result":{engine, model,
revision?, files:[{path, size, sha256}]}}` or `{"type":"error"}` frame, so the
first hash of a large model does not run into the 30-second response-header
budget. `GET /v1/models/file` serves one file with `http.ServeContent`, so
`Range` requests work. It only serves a path that is exactly one of the
model's listed files; a malformed path (`..`, absolute, backslashes, drive
letters) is a `400`, anything else not in the list is a `404`. For native
layouts, every listed file must resolve, after symlinks, inside the engine's
store; custom bindings use the exact local allowlist described below.

Custom llama.cpp names are resolved separately from catalog identifiers. An
exact configured native preset section supplies the local model and optional
draft GGUF shards, in native order. Applied serving declarations must compile
against their approved bindings and their asserted full SHA256 values must
match streamed full-file hashes. Without a declaration, the same bounded preset
parser and effective launch options select the exact section; relative paths
are resolved against that preset. Projectors, adapters, includes and source URLs
are explicitly refused on this custom path. Only these exact configured files
may be exported outside `LLAMA_CACHE`; peers cannot provide host paths. A custom
name with no unique binding is refused without guessing a Hugging Face repo.
Committed custom-copy receipts can also supply an exact local binding. Different
configured and receipt artifact sets for the same alias are ambiguous.

Custom listings add `layout:"llama-artifacts-v1"`, `exportId`, and `role` on
each file. Paths are synthetic `model/<GGUF basename>` and
`draft/<GGUF basename>` entries; host paths and serving settings are absent.
The export ID binds the exact model name and ordered role/path/size/full-SHA256
manifest (canonical versioned JSON is specified beside `artifactExportID`). Each
file GET must carry this `exportId`. The source re-resolves and checks the issued
configuration and regular-file identity before opening the file. Source-local
issued bindings are bounded; restart or eviction requires a fresh listing.
Listing results are bounded to 1 MiB, artifact sets to 1024 entries and progress
frames to 64 KiB independently of the stream's total length. Shared hashing
waiters cancel independently; cancellation of a producer lets a live waiter
retry under its own context.

On the destination, the copy:

1. Refuses an engine with no known layout, an engine that is not installed
   here, and a listing whose paths or hashes do not fit the engine's layout.
2. Skips files already present with the listed size and sha256. Different
   completed bytes are a conflict for every layout and are never overwritten.
   Unsafe destination symlinks and Windows case/path collisions are refused.
3. Refuses to start when the volume holding the store has less free space than
   the bytes still missing (net of partial downloads) plus 256 MiB.
4. Downloads each missing file into `<final>.part` beside its final name,
   resuming an existing partial with an HTTP `Range` request. Partials are kept
   across retries and restarts. A download that receives nothing for 60
   seconds, or drops, is retried from where it stopped, up to five attempts.
   Every attempt opens the partial through the store's rooted filesystem and
   checks its regular-file identity before truncation or writes. Its pathname
   and the renamed result must still identify the file whose bytes were hashed.
5. Verifies the sha256 of the complete file, then renames it into place. A
   mismatch deletes the partial and fails the copy (after one fresh retry when
   the attempt had resumed an older partial).
6. Writes the file that makes the model visible to the engine last.

Progress is `engine:remote-progress` with `opId`, `op:"copy"`, `node` naming the
source, `sourceNodeId`, `destinationNodeId`, `engine`, `model`, and `exportId`
as soon as a custom listing resolves. Stages are `listing`, `hashing` (relayed from the source, `file` set),
`checking`, `downloading` (`file`, and `bytesDone`/`bytesTotal`/`percent` over
the whole model, about every 500 ms), `finalizing`, then a terminal `done`
(`percent:100`), `error`, `cancelled` or `unknown` (`percent:-1`, `message`). Each accepted
operation emits exactly one terminal outcome; done follows verified publication.
Lost or invalid copy acknowledgements carry a typed unknown outcome and emit
`stage:"unknown"`, even if the destination already published the model.
Definite remote refusals remain errors; explicit cancellation remains cancelled.
An unknown outcome permits explicit idempotent retry and never establishes failure
or success from the transport error text. Copy stream error frames preserve this
distinction in `stage` independently of their diagnostic message.
Copy NDJSON uses the per-operation callback, bounded coalescing and checked
write/flush errors that cancel the transfer. It does not subscribe to another
operation's engine-wide progress. Terminal result byte counts and the custom
receipt qualification are retained separately from inventory.

| Engine | Store | What is copied | How the engine sees it |
|---|---|---|---|
| Ollama | `OLLAMA_MODELS` from the launch environment or this process, else `{models_dir}/models` (`~/.ollama/models`) | `blobs/sha256-*` named by the manifest, then `manifests/<host>/<namespace>/<name>/<tag>` | The manifest is written last, after checking every blob it names is in place. Ollama reads manifests on each listing, so no restart is needed |
| llama.cpp | `LLAMA_CACHE` (the manifest sets `{models_dir}`, `~/.llamacpp`) | For `owner/repo:TAG`, the snapshot GGUF matching `TAG` with all its splits, plus the multimodal projector beside it, chosen as llama.cpp chooses them | Files go under the destination's existing snapshot for the repo, or the source's revision with `refs/main` written last. The first split goes last. A running router is asked to rescan (`GET /models?reload=1`) |
| llama.cpp custom artifacts | `.pair-copies/<SHA256 of exact model ID>/<exportId>/` inside the destination's own store | Exact configured model and draft GGUF shards | Only the versioned alias-level `copy.json` receipt marks a completed bundle; native presets and inventory are not changed |
| LM Studio | `{models_dir}` (`~/.lmstudio/models`) | The path `lms ls --json` gives for the model (a GGUF file plus any `mmproj` GGUF beside it, or an MLX directory). An id that matches several models by path prefix is refused | Same relative path. A running server is restarted afterwards, as for `delete_model`, because LM Studio indexes models at startup |

Limits: one copy of a given model at a time per node. Ollama models named with
a registry port (`host:port/...`) are refused. An adopted Ollama instance
started outside PAIR may use a different `OLLAMA_MODELS` than this service
resolves, and LM Studio's models root is the manifest's `models_dir`, not a
custom folder chosen in LM Studio's settings. The destination's sha256 check
catches a source file that changes during a copy.

Custom bundles use the same identity-withdrawal guard, file locks, free-space,
resume and full-hash checks. The receipt is published only after every artifact
is verified. Interrupted bundles without a receipt remain resumable and are not
completed copies. A different committed export for the same alias is a conflict;
identical receipts are no-ops after file verification. No automatic deletion,
native rescan, engine start, load or serving configuration follows custom COPY.

The additive Go result carries `layout`, `exportId`, `requiresConfiguration:true`
and `durability:"confirmed"|"uncertain"`. A replacement that landed but whose
durability could not be confirmed is qualified completion, preserving the
receipt and verified artifacts. Reading a previous receipt alone cannot establish
durability, so an identical retry reports `uncertain`. Custom bundles need an
explicit destination-owned serving setup before loading; receipts do not appear
in `ModelsResult` or native `list_models`. Dedicated bridge/UI presentation of
the setup requirement and durability qualification is separate work; these Go
fields do not establish that presentation or serving readiness.

### Controller COPY process verification

The isolated `services/tests` selector `^TestModelCopyControllerProcess$` drives
the typed broker RPC with separate synthetic controller/source/destination
identities and real engine-manager handlers. Its test source covers custom and
canonical layouts, authenticated direct/reverse transfer, positive-byte
interruption and explicit Range retry, controller cancellation, target permission
and pin failures, changed source content, destination conflicts, failed mutation
acknowledgement and lost-response no-replay behavior. See the
[fixture prerequisites and native checklist](../readme.md#controller-selected-model-copy).

On Windows at base `44e76c6b` (2026-10-26), the complete focused parent passed
plain and race with 14 named terminals, no failures or skips. Review then
strengthened the lost-acknowledgement fault to drop the downstream connection
before response headers; that changed leaf and its parent passed plain and race
(two terminals each). Module vet exited zero. Commands used the registered
fixed-path wrapper, private profiles and offline Go, from `services/tests`:

```sh
go test -p 1 -parallel 1 -count=1 -json -timeout=30m -run '^TestModelCopyControllerProcess$' .
go test -race -p 1 -parallel 1 -count=1 -json -timeout=30m -run '^TestModelCopyControllerProcess$' .
go test -p 1 -parallel 1 -count=1 -json -timeout=15m -run '^TestModelCopyControllerProcess$/^dispatched_POST_is_not_replayed_on_alternate_reverse_route$' .
go test -race -p 1 -parallel 1 -count=1 -json -timeout=15m -run '^TestModelCopyControllerProcess$/^dispatched_POST_is_not_replayed_on_alternate_reverse_route$' .
```

Raw output remains in the machine's temporary
`download-transfer-process-checks-20261026` directory. The combined whole-suite
run is separate from these focused results.
The fixture records/refuses controller artifact transport and checks that its
model store stays absent. Production artifact filesystem calls have no
process-level attempt observer, so these checks alone do not prove zero attempted
controller filesystem IO. Installed builds, actual source bindings, native UI
behavior, strict controller artifact IO instrumentation and destination serving
setup require separate acceptance.

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--ipc <path>` | _(stdio)_ | IPC endpoint: Unix socket or Windows named pipe |
| `--http-port <port>` | `0` (off) | Serve the plain LAN model-list surface (`GET /v1/models`, the `em` service) on this port; the broker passes `:14322` |
| `--control-port <port>` | `0` (off) | Serve the cluster-scoped mTLS remote-control surface (the `ec` service) on this port; callers are admitted only while this node is a cluster member. The broker passes `:14323` |
| `--control-host <ip>` | `0.0.0.0` | Literal IPv4 or IPv6 bind address for `ec` only; `127.0.0.1` confines that listener to loopback. Does not change the `em` listener or authentication |
| `--reserved-port <port>` | `0` (off) | Refuse local or remote engine starts and persisted port changes on a parent-owned proxy alias; the broker configures this from `OLLAMA_HOST` |
| `--cluster-dir <dir>` | _(none)_ | Cluster identity/pin directory; gates the `ec` surface on and supplies the leaf/pins used to serve it and to dial peers |
| `--loaded-poll-interval <sec>` | `5` | Seconds between loaded-model polls that drive `engine:models-changed`; `0` disables the watcher |
| `--log-level <level>` | _(env `NVPAIR_LOG_LEVEL` or `info`)_ | `debug` \| `info` \| `warn` \| `error` |
| `--uninstall-managed` | | Remove every engine PAIR installed, then exit; for the platform uninstallers, which have no broker to call `engine:uninstall-managed` through. Same selection and safety path as that method, loading only the manifests compiled into this binary. Always exits 0 so an uninstaller cannot stall |
| `--version` | | Print version and exit |

Logs go to **stderr** via the shared `nvpair-shared/applog` format; stdout is
reserved for JSON-RPC frames in stdio mode.

## Logging & errors

Each managed engine's stdout/stderr is captured into a bounded ring
(queryable via `engine:logs`); serving output redacts path-bearing tokens, quoted
paths and the configured models directory before capture or diagnostics, and
oversized output is drained without retention.
Operational failures (install/start/health)
are recorded and surfaced as `errors:report` / `errors:clear`
notifications using the `nvpair-shared/errors` wire shape. The broker
(`nvpair-ui-broker`) forwards them to the `nvpair-errors` registry.
Ids follow `engine-manager:<class>:<engine>`.

Command-mode engines declaring `runtime.auth_statuses` and
`runtime.app_processes` (currently LM Studio) diagnose failed start commands,
failed readiness, and health-probe failures through the existing `errors:report`
message. Start failures retain `engine-manager:start-failed:<engine>` with
severity `error` and action `retry`; health failures retain
`engine-manager:unhealthy:<engine>` with severity `warning` and action `none`.
`engine:state-changed{EngineStatus}` and the error payloads are unchanged; no
payload field was added. With `<Name>` replaced by the manifest's `display_name`,
the messages are:

- API-key refusal (a declared HTTP 401 or 403): `<Name> is running but requires an API key, and PAIR does not have one. Turn off API authentication in <Name>'s server settings, then try again.`
- App or daemon process present but no connection: `Can't connect to <Name>. Does it have an API key set, or is its server off?`
- No matching process: `Can't connect to <Name>, and no <Name> process is running. Open <Name>, then try again.`

Auth evidence takes precedence over process inspection. Without diagnosis fields,
or if process listing fails without auth evidence, the original failure wording
is retained. Before a command-mode start, a declared auth response from the
effective ready endpoint refuses the start without running either the start or
cleanup stop command, leaving the listener unadopted. Explicit Stop and shutdown
retain their existing behavior. After a start command is attempted, three
consecutive readiness responses with a declared auth
status fail readiness early (about 0.6 seconds for immediate responses); any
other result resets that count. Other failures retain the existing readiness
timeout (60 seconds for LM Studio), and failed-start stop-command cleanup is
unchanged. This diagnosis does not add LM Studio API-key storage or use.

LM Studio readiness remains its HTTP probe: process presence never makes an
engine running, healthy, adopted, or routable. Diagnosis runs on a start attempt
(including automatic relaunch) or in the health loop of an engine PAIR started
or adopted through Start. An engine adopted only by a passive status read has
no health loop and gets no health diagnosis. A health failure is reported even
if a status read already changed `healthy` to false, and the report clears when
the health probe recovers. Merely opening LM Studio does not trigger diagnosis.

## Security posture

Runs **user mode only** — no admin/sudo at runtime (privilege escalation
is reserved for NVPAIR's own install time). It has two optional LAN listeners, and
**both are pin-based cluster mutual TLS with a live membership check** — every
caller is authorized against a per-peer pin, a non-member is refused with a `403`,
and while this node belongs to no cluster it presents no leaf so no handshake
completes:

- **`--http-port`** — the model-list surface (`em`, `GET /v1/models`; the broker
  passes `:14322`). A node's model inventory is cluster data, so LAN callers must
  be pinned peers. This port additionally serves **plaintext on loopback only**,
  which is how this node's own scanner enriches its own card — including when the
  node belongs to no cluster, so a standalone machine still shows its own models.
- **`--control-port`** — the remote-control surface (`ec`, the `engine:remote-*`
  targets; the broker passes `:14323`). mTLS only, no plaintext personality, since
  every route performs a privileged operation.

Unlike the rest of NVPAIR, engine-manager therefore **does terminate inter-node
mTLS itself** (and dials peers' `ec` surfaces with the same pinned identity),
matching how `nvpair-errors` and `nvpair-workload-manager` handle their own
cluster traffic. Managed engines bind **loopback by default**, but a manifest's
`runtime.bind` may open an inference engine to the LAN — Ollama ships
`0.0.0.0` to serve the cluster — overridable per call via
`engine:start {bind}`; readiness/health probes always target loopback.
Downloads are **HTTPS-only** (plain `http` only from loopback) and verified
against the manifest's `sha256` when one is pinned; an unpinned fetch runs
with a loud warning.

## Cross-platform

One binary compiles and runs on Windows, Linux, and macOS × amd64/arm64.
Per-OS variance lives in the manifest first; OS primitives (process
termination, console hiding) are the only build-tagged Go
(`proc_windows.go` / `proc_unix.go`).

## Target-owned engine switching

The broker owns switch acceptance, drain, publication and the operation journal.
The manager executes only broker-local `engine:switch-prepare`,
`engine:switch-step` and `engine:switch-reconcile` requests using the shared
`enginesettings` envelopes. These executor methods have no ec route. Results
carry a durable `checkpoint`, current `observed` states and a coded `error`;
callers must inspect that error rather than treating a JSON-RPC result as success.

Prepare atomically reserves source and target, validates preparation and port
ownership, and rejects a running target. An empty prepare request reads the
manager configuration snapshot without reserving. Its additional `configuration`
and `configurationFence` fields supply configuration digests, settings revisions,
intent/provenance, registry/runtime fingerprint and manager incarnation. The
broker combines these with node identity and journal generation for its public
configuration token; it passes the manager fence back in the executor envelope.
The envelope also carries `managerIncarnation` from the snapshot. Prepare and
forward steps (`source-stop`, `target-start`) reject a different incarnation
with a coded `interrupted` error, including inherited checkpoints after restart.
Reconcile and release accept the original envelope to finish conservative cleanup.

A broker-managed manager starts recovery-pending before binding its ec listener.
All mutating authenticated ec requests return HTTP 503 until the parent sends
`engine:switch-recovery-ready` with `{managerIncarnation}` after successful
conservative recovery. The parent pins that request to the recovered worker;
the manager checks its incarnation and refuses readiness while a checkpoint is
unreleased. This gate is independent of local pause. Inventory, read-only
queries, event streams, authoritative state deliveries and parent-only
inspect/reconcile/release remain available during recovery.

The optional `pendingSwitch` in `engine-state.json` shares the durable replacement
with `engines` and `offBy`. Source Off and `stop-pending` are saved before stopping.
Adjacent `pendingSwitchPorts` records source/target execution endpoints; a
dynamic target port is allocated and checkpointed before launching its process.
Target On and `target-committed` are saved together only after readiness. A failed
start saves target Off, preserves a stronger local Off, and never restores the
source. Termination evidence requires listener refusal as well as stopped owned
state. Reconciliation cleans up uncommitted interrupted work and never resumes
that target startup. Cleanup checks the durable positive port, managed install
containment and executable image before reclaiming a child, even if unhealthy.
A committed target still saved On recovers through the same reclaim owner and
an owned cold launch; a later fresh switch target start uses that owner too.
Foreign listeners are never killed and retain the existing port-ownership error.
Reclaim observes caller and shutdown cancellation before signalling and while
waiting for termination. Windows windowless orphans use `taskkill /T /F` without
consuming the Unix graceful interval. After a forced signal, reclaim polls for
positive PID-exit evidence for up to the default five-second stop grace, even
without a readiness probe; caller cancellation interrupts that wait.
Failed or cancelled reclaim reports a start failure and fences adoption,
including explicit Start, until a verified stop or successful owned cold start.
Passive status remains not running and unhealthy.
Listener refusal alone cannot clear that fence: the recorded failed-reclaim PID
must be verified exited. Live/reused PIDs and inspection errors retain the fence;
the retained PID never authorizes signalling without the current reclaim predicate.
The matching checkpoint remains reserved until the broker releases it after its
terminal receipt is durable; a released checkpoint retains the last operation's
retry identity until replaced by a new reservation.

Reserved engines reject lifecycle/configuration/install/restart-after mutations
with JSON-RPC `-32000` and `data.kind: busy`; automatic restore skips them.
Wake/Sleep retain their intent-neutral behavior. Shutdown cancels starts and stops
owned engines even during a reservation, without rewriting intent. A local Off
during a remote reservation persists local provenance and cancels target startup
before waiting for cleanup and returning the stop result. New local and remote
switches check target lock (`-32013`) and either-origin pause (`-32011`) before
source stop. Historical saved Off does not refuse a new switch.
The veto takes neither the switch lock nor the engine operation lock, including
for a remote reservation retained for recovery. Local Off of a reserved engine
during a local switch remains a busy refusal.
The parent may scope `engine:stop` with internal `switchOperationId` for an Off
admitted during a remote reservation. Under the same admission lock, a settled
or different reservation makes the scoped stop a no-op against newer work.
Missing checkpoints, including rejected prepares, also make scoped stops no-ops.
The broker durably journals the admitted Off before latching it, then discharges
it through an ordinary unscoped stop before releasing its operation lock. Internal
`verifyStop: true` on that parent stop returns `SwitchEngineState` with verified
termination and saved Off, or an error; ordinary callers retain the status reply.
This discharge replays after either process restarts, including without a manager
checkpoint, before restoration or readiness. An older direct peer start loses to
the subsequently admitted local Off; a later explicit local On waits for discharge.
A provisional remote prepare refusal retains its reservation until final cleanup.
Reconcile physically stops only engines whose local Off is recorded for this
operation (`sourceLocalOff`/`targetLocalOff`), including the reserved phase. Old
local-Off provenance on an adopted running source does not authorize physical
cleanup after failed drain. Release refuses unverified operation-specific Off.
Source is never restarted by reconciliation.
While a remote reservation remains unreleased, a new local Off of either engine
also vetoes an already-committed target. Either-origin pause fences committed
targets of local and remote reservations too. Reconciliation records
target Off and pending cleanup before stopping it; source is never restored.
Restart reconciliation without an explicit veto preserves committed target On.
Reconcile can carry an internal operation-specific `localVeto` (the local Off
engine or a pause). The manager records it durably before inspecting/cleaning
the target, independent of a pending lifecycle stop request. Release shares the
durable Off admission gate and refuses a committed target with an uncleaned
veto, retaining its reservation until reconciliation verifies target Off.

The existing pinned ec listener exposes `/v1/engine-switch/apply`, `/status`
and `/fence`, relaying to the target broker with
`engine:switch-request/reply/cancel`. The
cancel notification retires only the wait, never accepted broker-owned work.
Pending waits are bounded; settings relay correlation is separate.
`engine:remote-switch`, `engine:remote-switch-status` and
`engine:remote-switch-fence` use the existing pinned peer client. These routes
do not provision engines or fall back to remote start/stop.

Fence is a management mutation, requiring the fresh pinned caller's grant at
the HTTP gate and again at broker admission. Its strict request is
`{nodeId?, requestId}`; the target replaces the node alias and relays only
`engine:switch-fence` with certificate-derived caller identity. It returns
`{nodeId, requestId, fenced:true}` after the broker's durable absence fence, or
`{nodeId, requestId, fenced:false, receipt}` for an existing owned receipt.
It never cancels accepted work, calls an executor or forwards another hop.
The broker bounds caller/request-ID tombstones globally to 32, evicting the
oldest insertion, while retaining the existing generation-based revision fence
after eviction. A foreign tombstone does not reserve another caller's ID.
Receipts remain unchanged and non-evicting; callers read a fresh configuration
baseline after acknowledgement. Failure, including uncertain persistence, is
not acknowledgement. See the broker README for durability and capacity rules.

Fence acknowledgements use shared `DecodeSwitchFenceResult`: field names are
case-sensitive and unique, `fenced` is an explicit boolean, and the exact union
rejects a null receipt or trailing JSON. Nested receipt objects reject unknown,
duplicate and case-variant fields before the existing receipt validator checks
values and identities. Ordinary switch request/journal decoding is unchanged.

Authoritative broker operation errors travel through the existing coded HTTP
409 envelope as `operation_error` (`-32000`). The switch client preserves their
exact target message, including `Unknown switch receipt for this caller.`;
absence still establishes no fence by itself. The relay carries only numeric
code and message, so it does not supply a distinct `unknown_receipt` wire kind.
Uncoded transport/coordinator failures remain uncertain diagnostics. Existing
policy/conflict/veto/permission/lock aliases retain their meanings.

## Target-owned model downloads

`PullModelStream` owns local, CLI and paired catalog pulls. The existing
`engine:models`, `/v1/models` and `engine:models-changed` inventory carries
`downloadState` (version 1): `sourceId`, decimal-string `revision`,
`recoveryRequired` and `entries`. Each entry preserves the exact request model
and engine, a target-minted ID, `queued`, `in_progress`, `failed`, `complete` or
`unknown` phase, required `blocksReadiness`, optional measured integer percent
and a fixed reason code. `persistence:"uncertain"` qualifies an outcome whose
marker removal landed without confirmed directory durability. This does not
certify vendor-file durability. Pull progress carries model, ID and the full
snapshot; remote request `opId` remains a separate correlation identity.

Operation completion, current content inventory and observed loaded residency
are independent. Ready inventory and identity digests exclude queued and unsafe
content. The named llama.cpp SSE adapter projects only stable `unloaded`,
`loading`, `loaded` and `sleeping` inventory statuses. Name membership, transient
`downloaded`, percent 100 and accepted POST do not prove native completion.
NDJSON pulls require `status:"success"`; llama.cpp requires matching
`download_finished`; CLI pulls require successful exit without an explicit error.

`model-download-state.json` is a private, bounded unresolved-content journal,
separate from desired engine intent. It retains unsafe PAIR attempts across
restart and store changes without automatic effects. Its engine/content keys
use descriptor-selected naming; raw dispatch spelling is preserved. The manager
admits at most 32 accepted/unsafe keys, retains the latest 32 other outcomes,
publishes at most 64 entries, and bounds journal/snapshot bytes to 64/256 KiB.
Markers have no TTL or eviction. Malformed/unreadable recovery fences ready
publication and Load/export/pull effects. A same-key explicit successful retry
can resolve its marker. Successful Delete also requires a fresh same-run
unmasked absence observation; llama.cpp requires a forced cache rescan. Failed
cleanup, ordinary list absence and restart cannot clear a marker.

Content writers and Load/export readers have nonblocking admission beside the
existing content semaphore. An admitted reader makes a writer fail busy before
withdrawal or effects; an active writer refuses new reads. Load guards cover
typed/raw/remote actions and automatic residency. Export leases span hashing
and response-body completion. Existing in-flight inference is not cancelled.

Scanner observers keep last-good lifecycle data with `downloadStateCurrent`:
false means stale, absent means unsupported. A successful same-revision refresh
can restore currentness; older revisions cannot overwrite newer observations.
Origin and target notifications are immediate; other observers converge through
the existing refresh cycle and its work/request budgets. Updated clients require
current support and canonical ready inventory for ready-dependent actions. This
requires a coordinated fleet update. Files predating this journal are not
retroactively certified. Process, native UI and real-target acceptance remain
separate from component tests.

`services/tests/model_download_state_test.go::TestModelDownloadStateProcess`
contains process scenarios for target-local service subscribers, a newly attached
client, pinned observer inventory and production scanner self-enrichment. Its
synthetic SSE router exposes first-shard, delayed-terminal and disconnect
barriers. The assertions distinguish completion, stable ready inventory and
loaded observation; reject typed/raw Load and existing export reads without
native effects; isolate a cancelled queued model; and check direct/reverse POST
counts, failure recovery, failed journal removal and bounded retained state.
Service restarts use the existing private bundle's process-identity cleanup.
On Windows at base `44e76c6b` (2026-10-26), the final plain run below passed
17 named terminals with no failures or skips. The race run had 15 passes and
two failed terminals (the queued-cancellation leaf and its parent): Windows
refused replacement of the fixed scanner image during setup. That leaf alone
then passed under race (two terminals, no failures or skips). No data race was
reported. The failed aggregate run remains failed; no second full focused race
pass is claimed. `go vet ./...` exited zero.
The commands ran from `services/tests` with the fixed-path wrapper, private
profile and offline Go bindings. Raw output remains in the machine's temporary
`download-transfer-process-checks-20261026` directory.

```sh
go test -p 1 -parallel 1 -count=1 -json -timeout=30m -run '^TestModelDownloadStateProcess$' .
go test -race -p 1 -parallel 1 -count=1 -json -timeout=30m -run '^TestModelDownloadStateProcess$' .
go test -race -p 1 -parallel 1 -count=1 -json -timeout=15m -run '^TestModelDownloadStateProcess$/^queued_cancellation_has_no_native_effects_or_unsafe_marker$' .
```

These runs include bounded second-response NDJSON capture, replacement
reverse-session/read checks, control-surface POST counters and exact terminal
notification convergence assertions.
The control observer reuses a registered fixture image and the existing owned
port adapter; it forwards pinned HTTP streams to the real manager unchanged.

The fixture does not prove independent scanner peer refresh: authenticated peer
descriptors reject loopback addresses. Its production scanner uses only the
target's registered loopback self-enrichment route, with mDNS disabled. Automatic
resident dispatch is an internal broker/proxy boundary and is not exercised by
this process fixture. The compiled manager has no short progress-watchdog seam
or crash barrier inside durable journal replacement; watchdog and exact atomic
replacement failures retain component-level coverage. No lower-revision or
retired-source notification is injected by this fixture. The ec listener uses
`--control-host=127.0.0.1`; em and cluster-manager retain their production
wildcard binding on private fixture ports. No installed-machine or transfer-join
acceptance follows from these scenarios.

The separate `TestModelDownloadStateTransferJoinProcess` source regression joins
downloads to the accepted COPY implementation on base `efda20f7691a01980dd4740789d5d4829f6ba43b`.
It first requires a real custom-alias list and full artifact GETs with independently
checked hashes. Active and failed pulls then refuse that previously reachable
export and destination COPY: no artifact GET is dispatched by COPY, destination
store contents and mutation-withdrawal counts stay unchanged, and an explicit
successful same-key catalog retry restores export and COPY. Another scenario
creates a real COPY receipt, fails a catalog pull for the same exact key, and
performs an identity-bound receipt retry. Successful local rehash and skipped
files must leave the failed download snapshot, durable unsafe marker, receipt
and readiness fence unchanged. Assertions require absent controller artifact
stores, stopped controller engines and source/destination certificates on peer
reads. As in the COPY fixture, they do not observe attempted local artifact syscalls.

On Windows (2026-10-26), the joined selector passed plain and race with three
named terminals each, no failures or skips; module vet exited zero. Commands
ran from `services/tests` under the same fixed-path/private/offline bindings:

```sh
go test -p 1 -parallel 1 -count=1 -json -timeout=20m -run '^TestModelDownloadStateTransferJoinProcess$' .
go test -race -p 1 -parallel 1 -count=1 -json -timeout=20m -run '^TestModelDownloadStateTransferJoinProcess$' .
```

The merged whole-suite run remains separate. These scenarios reuse
the controller COPY fixture and its cleanup, with a fixture-only adopted SSE
router and Windows-safe aliases holding a 128-byte model and 4 KiB draft. They
do not extend the unsupported Task 5 injection or native-machine claims above.

## Shutdown behavior

Shuts down on stdin EOF (parent closed the pipe), `SIGINT`/`SIGTERM`, or a
`shutdown` JSON-RPC request — stopping any running engines first so none
are orphaned.
