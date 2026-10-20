<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvpair-engine-manager

A config-driven control plane for local inference engines. The bundled
manifests support Ollama, LM Studio, and llama.cpp. It manages everything about
an engine **except serving inference**: detect, user-mode install,
start/stop/restart, health, and config-declared actions. Adding an engine
is a JSON manifest, not code.

The bundled manifests under `manifests/` are the working reference for manifest
authoring.

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
absolute operation budget (`timeoutMs`, at most 60 seconds), and finite
`load.contextLength` / `load.keepAliveSeconds`. Clients and peers must not
invoke it. The broker owns policy, qualification and capacity; the manager
does not infer a grant from saved intent.

Only an explicit Ollama-family semantic load binding to prompt-free
`POST /api/generate` is supported. The manager sends the normalized installed
model, `stream:false`, `options.num_ctx = contextLength` and numeric
`keep_alive = keepAliveSeconds` (seconds). Other fixed load parameters,
restart-after bindings, implicit bindings and other engine families are
`unsupported-load`. No prompt, download, delete, copy, engine start, restore,
generic action fallback or inference is performed. This path takes E3
admission and the lifecycle lock, never B2's content guard.

Before and after admission, unknown projection, local pause/Off, switch busy,
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
post-call inventory observation with the current manager/model identity;
even a lost load response can converge through that readback. Missing inventory
is unknown, while an explicit empty list is observed absence. Every dispatched
call pokes the loaded watcher. Desired residents never become observed inventory
or new desired-state-file fields. Duplicate operation IDs coalesce for their
original lifetime; changed payloads conflict. Idle unload protection belongs to
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
| `engine:action` | `{ engine, action, params }` | the engine's raw response. `action:"pull_model"` is streamed: it emits live `engine:pull-progress` notifications and returns the pull's terminal result (see below). An action whose manifest declares `restart_after` (LM Studio's `delete_model`) restarts a running engine before replying, so the response also means the engine is back and healthy |
| `engine:logs` | `{ engine }` | `{ lines: [LogLine] }` |
| `engine:errors` | — | `{ errors: [ServiceError] }` |
| `engine:catalog` | `{ engine, platform?, arch?, query? }` | `{ models: [{ id, name, author, url, size?, downloads, likes, updatedAt, tags, family?, parameterSize?, appleOnly? }], source, platform?, arch?, fetchedAt?, searchable?, query? }` — the models an engine can **download**, as opposed to `engine:models`, which reports what is already installed. Each engine family has one curated source, selected by the manifest's `family` (a fork with `family: "llamacpp"` gets llama.cpp's); the client spellings `lm-studio`, `llama-cpp` and `llama.cpp` are accepted. Ollama's has no public API, so it is a locked list compiled into this binary from `catalog/ollama-models.json`, regenerated on demand by a developer (`npm run scrape:ollama-models` in `desktop/`) who reviews the diff; serving it needs no network. LM Studio's catalog *is* the `lmstudio-community` Hugging Face org, whose repo ids are exactly what `lms get` accepts, so it is fetched live and cached for six hours, with concurrent callers coalesced onto one request, a failed attempt — with or without a list to fall back on — followed by a backoff before the next. llama.cpp downloads any GGUF repo as `repo:quantization`; its browse list is the GGUF repos of `ggml-org`, `bartowski`, and `unsloth`, fetched and cached the same way, keeping only open, text-generating repos that ship a `Q4_K_M` file, each offered as `repo:Q4_K_M`. It is the one `searchable` source: a `query` searches all of Hugging Face under the same rules, each query cached on its own (the 20 most recently used are kept), and the reply echoes it. A query longer than 100 characters is refused rather than shortened, since a shortened one searches for something the operator did not type. Other sources ignore a query and return their whole list for the client to filter. `platform` and `arch` are the GOOS and GOARCH the models will be **installed on**, which is not always this host — a client driving a peer should name that peer. Omitting both means this host; naming a platform without an arch leaves the arch unknown rather than borrowing this host's. MLX quantizations only install on Apple Silicon, so they are marked `appleOnly` and kept only for a `darwin`/`arm64` target — an Intel Mac does not get them; the reply echoes the `platform` and `arch` it filtered for. `name` is pull-ready: it can be handed to `pull_model` verbatim. An engine with no curated source is an error, not an empty list. The Ollama reply is a single multi-megabyte frame, so every hop on its path must allow `jsonrpc.WorkerFrameBytes`. |
| `engine:models` | — | `{ models: [string], modelsByEngine: { <engine>: [string] }, loadedByEngine: { <engine>: [string] } }` — the flat de-duplicated union of every running engine's models, the per-engine breakdown keyed by engine name, and the per-engine set of models currently **loaded in memory** (all normalized from each engine's `list_models` / `loaded_models` action `result` spec). `modelsByEngine` carries a key for every running engine whose inventory was successfully queried, including an empty list = "running, no models available"; a missing key means not running / not queryable / invalid response. `loadedByEngine` uses the same known-empty distinction for residency and also omits engines with no loaded endpoint. `policy` is this node's policy advertisement (`{ revision, roles, pausedBy? }`) as the broker last pushed it with `engine:node-projection`, omitted until the first push. The `/v1/models` HTTP surface returns the same shape. |
| `engine:remote-get-installed` | `{ node }` | `{ engines: [EngineStatus] }` fetched from the remote node over `ec` mTLS |
| `engine:remote-install` | `{ node, engine, start? }` | `{ opId, status: EngineStatus }` after the remote install (live progress via `engine:remote-progress`) |
| `engine:remote-pull-model` | `{ node, engine, model?, params? }` | `{ opId, result }` after the remote pull (live progress via `engine:remote-progress`) |
| `engine:remote-copy-model` | `{ node, engine, model }` | `{ opId, result: { engine, model, files, filesSkipped, bytesTotal, bytesCopied } }` after copying the model from `node`'s engine store into this node's (live progress via `engine:remote-progress` with `op:"copy"`). See "Model copy between nodes" below |
| `engine:remote-start` | `{ node, engine, port? }` | `EngineStatus` from the remote node (always the manifest's `runtime.bind`; no per-call bind override on the remote path) |
| `engine:remote-stop` | `{ node, engine }` | `EngineStatus` from the remote node |
| `engine:wake` | `{ engine }` | `EngineStatus` — starts the engine only if its saved intent is On (an engine with no saved intent counts as Off and is refused). Never changes the saved intent. Refused once shutdown has begun |
| `engine:sleep` | `{ engine }` | `EngineStatus` — stops the engine the way `engine:restart` and shutdown do, without recording Off intent, so `engine:wake` or the next launch's restore can start it again |
| `engine:intent` | — | `{ enabledByEngine: { <engine>: bool }, offByEngine: { <engine>: "local" \| "peer" } }` — one saved-intent snapshot for registered engines, without taking any engine's lifecycle lock |
| `engine:load-model` | `{ engine, model }` | the engine's raw response — loads one model through the manifest's `load` binding, the same one the remote load uses (Ollama: `run_model` with `stream:false` and a 10-minute response budget) |
| `engine:unload-model` | `{ engine, model }` | the engine's raw response — unloads one model through the manifest's `unload` binding, the same one the remote unload uses (Ollama: `keep_alive:0`) |
| `engine:delete-model` | `{ engine, model }` | the engine's raw response — deletes one model through the manifest's `delete` binding, the same one the remote delete uses. LM Studio's restarts a running server before replying |
| `engine:remote-policy-get` | `{ nodeId }` | `{ policy, availability }` — the target broker's peer projection of its policy: no routing section, and tiers keep only their local entries |
| `engine:remote-policy-set` | `{ nodeId, policy, expectedPersistedRevision }` | `{ policy }` — the peer projection after the target broker merges the change into its stored policy; the revision is forwarded unchanged and compared by the broker |
| `engine:remote-availability-set` | `{ nodeId, state }` | the target broker's `node:set-availability` result (waits for the target's drain, on the readiness-sized response budget) |
| `shutdown` | — | `null` |
| `log/set-level` | `{ level }` | `{ level }` |

`EngineStatus` = `{ engine, display_name, installed, running, healthy, port }`.

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

Notifications (caller → service): `engine:node-projection{policy, pausedLocally}`
is this node's policy advertisement (declared roles with their profile bounds,
policy revision, pause origin) and whether the node is paused locally. The
broker sends it after every persisted policy or availability change and on
every `engine:ready`. engine-manager keeps the last one in memory only and
serves `policy` on `engine:models` and `GET /v1/models`, so paired peers learn
this node's roles from the pinned inventory fetch they already make. It is
never put in mDNS TXT or in the peer-services descriptor. `pausedLocally`
drives the local veto (Remote engine management).

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
full authoritative snapshots to pinned peers.

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
  was recorded. A local stop records the OFF as local, which vetoes paired
  nodes' starts; a cluster stop records it as peer's.
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
`-32011/local_veto` and conflict `-32010`. Reads, status, model exports and
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
pause/Off cancellation still applies; grants cannot override it or private,
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
| `POST /v1/node-availability/set` | JSON `{ state }` | the target broker's `node:set-availability`, answered once the state is reached |

The node-policy routes carry the broker's policy, which engine-manager does not
interpret. Each is gated on the caller's pin like the settings routes, accepts at
most 256 KiB, refuses unknown fields, clears any `nodeId` so the target acts on
itself and never forwards a further hop, and relays to the broker as
`policy:request` stamped with the authenticated caller. The broker re-checks the
pin, acts, and answers with `policy:reply {id, result, error?, errorCode?}`.
Every error carries a JSON-RPC `errorCode`: `-32602` for invalid parameters,
including missing `expectedPersistedRevision`, invalid policy or availability;
`-32010` for a revision conflict; `-32011` for a resume the local veto refuses;
`-32012` for remote-management denial;
`-32000` for a save failure or other operational failure. The route maps
`-32602` to HTTP 400, `-32010`, `-32011` and `-32012` to HTTP 409 with the body
`{"error": <diagnostic>, "code": "policy_conflict" | "local_veto" | "remote_management_denied"}`, and other
failures to a plain-text HTTP 409. The origin's remote-policy methods map HTTP
400 to JSON-RPC `-32602`, a typed 409 to its code, and other HTTP failures to
`-32000`, preserving the diagnostic. No classification depends
on diagnostic text. A stale policy draft is a conflict before its policy is
validated, including a draft that tries to enable remote or paid execution.
The route abandons its wait (and sends
`policy:cancel`) if the caller loses its pin or 15 minutes pass; a pause the broker
already accepted still finishes.

**Local veto.** A node's own pause or Off wins over paired nodes until it is
lifted on the node itself. `POST /v1/engines/start`, `POST /v1/engines/install`
with `start`, and `POST /v1/models/load` answer HTTP 409
`{"error": <reason>, "code": "local_veto"}` while the broker's last projection
says `pausedLocally`, or while the engine is saved Off locally, and HTTP 503
before the first projection arrives. `engine-state.json` keeps the Off's origin
in `offBy` (`local` or `peer`): a local `engine:stop` or uninstall writes
`local`, a cluster stop writes `peer` unless the engine is already Off, and any
start clears it. An Off with no `offBy` entry predates the field and counts as
local; an engine with no saved intent is not vetoed. A remote start rechecks the
veto under the engine's lifecycle lock, and remote starts and loads already
running when a local pause or a local stop of that engine arrives are canceled
and answer the same 409; during an install with `start` only the start is
canceled, reported as an error frame with `"code": "local_veto"`. Stop, unload,
delete, pull, model copy and settings edits are not vetoed, but a settings
apply that would resume a stopped engine while the node is paused locally is:
the broker refuses it, its `engine:settings-reply` carries `errorCode: -32011`,
and the settings route answers the same 409. At the origin, the 409 and the
coded error frame become JSON-RPC `-32011` on `engine:remote-start`,
`engine:remote-install`, `engine:remote-load-model` and
`engine:remote-apply-settings`.

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

`engine:remote-copy-model {node, engine, model}` copies a model that a pinned
peer (`node`) already holds into this node's store for the same engine, over
the peer's `ec` surface, instead of downloading it from the internet again.

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
letters) is a `400`, anything else not in the list is a `404`, and every listed
file must resolve, after symlinks, inside the engine's store.

On the destination, the copy:

1. Refuses an engine with no known layout, an engine that is not installed
   here, and a listing whose paths or hashes do not fit the engine's layout.
2. Skips files already present with the listed size and sha256.
3. Refuses to start when the volume holding the store has less free space than
   the bytes still missing (net of partial downloads) plus 256 MiB.
4. Downloads each missing file into `<final>.part` beside its final name,
   resuming an existing partial with an HTTP `Range` request. Partials are kept
   across retries and restarts. A download that receives nothing for 60
   seconds, or drops, is retried from where it stopped, up to five attempts.
5. Verifies the sha256 of the complete file, then renames it into place. A
   mismatch deletes the partial and fails the copy (after one fresh retry when
   the attempt had resumed an older partial).
6. Writes the file that makes the model visible to the engine last.

Progress is `engine:remote-progress` with `op:"copy"` and `node` naming the
source. Stages are `listing`, `hashing` (relayed from the source, `file` set),
`checking`, `downloading` (`file`, and `bytesDone`/`bytesTotal`/`percent` over
the whole model, about every 500 ms), `finalizing`, then a terminal `done`
(`percent:100`) or `error` (`percent:-1`, `message`). The desktop app does not
render copy progress yet and ignores these frames.

| Engine | Store | What is copied | How the engine sees it |
|---|---|---|---|
| Ollama | `OLLAMA_MODELS` from the launch environment or this process, else `{models_dir}/models` (`~/.ollama/models`) | `blobs/sha256-*` named by the manifest, then `manifests/<host>/<namespace>/<name>/<tag>` | The manifest is written last, after checking every blob it names is in place. Ollama reads manifests on each listing, so no restart is needed |
| llama.cpp | `LLAMA_CACHE` (the manifest sets `{models_dir}`, `~/.llamacpp`) | For `owner/repo:TAG`, the snapshot GGUF matching `TAG` with all its splits, plus the multimodal projector beside it, chosen as llama.cpp chooses them | Files go under the destination's existing snapshot for the repo, or the source's revision with `refs/main` written last. The first split goes last. A running router is asked to rescan (`GET /models?reload=1`) |
| LM Studio | `{models_dir}` (`~/.lmstudio/models`) | The path `lms ls --json` gives for the model (a GGUF file plus any `mmproj` GGUF beside it, or an MLX directory). An id that matches several models by path prefix is refused | Same relative path. A running server is restarted afterwards, as for `delete_model`, because LM Studio indexes models at startup |

Limits: one copy of a given model at a time per node. Ollama models named with
a registry port (`host:port/...`) are refused. An adopted Ollama instance
started outside PAIR may use a different `OLLAMA_MODELS` than this service
resolves, and LM Studio's models root is the manifest's `models_dir`, not a
custom folder chosen in LM Studio's settings. The destination's sha256 check
catches a source file that changes during a copy.

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--ipc <path>` | _(stdio)_ | IPC endpoint: Unix socket or Windows named pipe |
| `--http-port <port>` | `0` (off) | Serve the plain LAN model-list surface (`GET /v1/models`, the `em` service) on this port; the broker passes `:14322` |
| `--control-port <port>` | `0` (off) | Serve the cluster-scoped mTLS remote-control surface (the `ec` service) on this port; callers are admitted only while this node is a cluster member. The broker passes `:14323` |
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
(queryable via `engine:logs`). Operational failures (install/start/health)
are recorded and surfaced as `errors:report` / `errors:clear`
notifications using the `nvpair-shared/errors` wire shape. The broker
(`nvpair-ui-broker`) forwards them to the `nvpair-errors` registry.
Ids follow `engine-manager:<class>:<engine>`.

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
state. Reconciliation cleans up interrupted work and never resumes target startup.
The matching checkpoint remains reserved until the broker releases it after its
terminal receipt is durable; a released checkpoint retains the last operation's
retry identity until replaced by a new reservation.

Reserved engines reject lifecycle/configuration/install/restart-after mutations
with JSON-RPC `-32000` and `data.kind: busy`; automatic restore skips them.
Wake/Sleep retain their intent-neutral behavior. Shutdown cancels starts and stops
owned engines even during a reservation, without rewriting intent. A local Off
during a remote reservation persists local provenance and cancels target startup
before waiting for cleanup and returning the stop result. Remote switches honor
local Off and local pause with `-32011`; local switches remain allowed.
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
While a remote reservation remains unreleased, local Off of either engine or
local pause also vetoes an already-committed target. Reconciliation records
target Off and pending cleanup before stopping it; source is never restored.
Restart reconciliation without an explicit veto preserves committed target On.
Reconcile can carry an internal operation-specific `localVeto` (the local Off
engine or a pause). The manager records it durably before inspecting/cleaning
the target, independent of a pending lifecycle stop request. Release shares the
durable Off admission gate and refuses a committed target with an uncleaned
veto, retaining its reservation until reconciliation verifies target Off.

The existing pinned ec listener exposes `/v1/engine-switch/apply` and `/status`,
relaying to the target broker with `engine:switch-request/reply/cancel`. The
cancel notification retires only the wait, never accepted broker-owned work.
Pending waits are bounded; settings relay correlation is separate.
`engine:remote-switch` and `engine:remote-switch-status` use the existing pinned
peer client. Neither route provisions engines or falls back to remote start/stop.

## Shutdown behavior

Shuts down on stdin EOF (parent closed the pipe), `SIGINT`/`SIGTERM`, or a
`shutdown` JSON-RPC request — stopping any running engines first so none
are orphaned.
