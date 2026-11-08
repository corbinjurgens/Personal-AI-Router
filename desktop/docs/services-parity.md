<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Services Capability Status

Build-apply preview consumes the broker's private owner-evidence envelope and
excludes manager-known model, data and install roots before package inspection.
Unknown root coverage and changed manager/source evidence refuse. Public impact
fields remain the desktop contract; private manager evidence and paths are not
part of preview, status or receipts. Production start remains unavailable until
atomic maintenance acceptance and the separate identity, worker, saved Pause,
enforcement and physical-stop proofs are supplied. Existing historical copy
safety and external data-root unknowns also block eligibility.

Model configuration contract: inspection is connected through the typed bridge,
existing broker settings relay and pinned engine-control read route. Portable
load edits, target resolutions, selected policy patches and preview impact are
represented; ordinary settings clients preserve model loads. Bounded inspection
and reviewed finite llama.cpp load application use the manager and broker's
coupled recovery path. Advisory components are connected and remain partial or
Unknown when costs, mapping or fresh evidence are missing. Production Fits and
Auto refuse until complete-cost specification and implementation acceptance;
synthetic complete fixtures are not production support. Apply forwards the
retained frozen result without re-resolving Auto.
Qualification and hardware-fit acceptance are not implied by a settings receipt.
Load-plus-policy changes reserve raw policy writers through native persistence
and target policy activation. Pause remains independently available, generation
locks retain their generation-only scope, and destination copy preserves
unselected fields. Interrupted bare acceptance requires fresh local review;
later durable stages recover the frozen resolution.

Inspect also carries optional `loadObservation` through the same authenticated
local/peer relay. Strict shared and desktop codecs preserve its bounded summary;
absence maps to Unknown/`observation-omitted`. The panel compares rounded native
buffers with the frozen load baseline, independently of draft estimates. Reports
and recovered history grant no qualification, physical-occupancy or complete-cost
authority. See [native buffer observations](../../docs/model-buffer-observations.mdx).
Windows buffer-process execution and directory/firewall/survivor/lock closure
remain pending PC3; native accuracy and calibration remain unverified. Copy
process families await the separate copy fix, rather than inheriting unrelated
synthetic passes.

This document records the current integration status between Personal AI Router
and the `services/` backend. It is a present-state reference, not a release
history.

- Backend versions: `services/versions.json`
- Method-level contract: `docs/services-api.md`
- Exceptions: `docs/service-contract-exceptions.json`
- Integration runbook: `docs/services-backend.md`

## Status summary

Desktop peer COPY now exposes an editable source, its actual installed-model
inventory, and an independent mandatory destination in **Copy model**. Local
and remote ready rows offer **Copy to device…** independently of upstream pull
support. Existing destination-local Copy to this PC callers share the canonical
coordinator. Progress/cancel/retry use complete tuples and operation IDs; the
controller receives no model bytes or presets. Ordinary success refreshes the
destination; custom success retains setup-required and durability qualification
without fabricating native inventory. Unknown outcomes require verification.
Component/bridge checks do not establish installed builds or native transfer
acceptance. Keyboard/focus, narrow-window layout, actual peers, direct/reverse
transfer and destination serving setup remain external acceptance work.

The synthetic controller COPY process fixture is implemented under
`services/tests/model_copy_controller_test.go`; Windows plain/race runs passed
on 2026-10-26 at base `44e76c6b` (14 named terminals each, no failures or skips).
The review-hardened transport-drop leaf then passed plain/race (two terminals
each), and module vet exited zero. It checks destination-owned transfer, receipts, authenticated
reader identity, interruption/retry/cancel and refusal paths. Its controller
transport/store assertions do not instrument every attempted local artifact
filesystem operation. See the [process fixture limits and native checklist](../../services/readme.md#controller-selected-model-copy)
before using its eventual results as acceptance evidence.

A profile names its engine. Concrete model requests still route through their
entry engine. Installed engines run while the node is available; private
on-demand wake repairs a stopped engine. Native N23, visual,
multi-engine, Windows and assembled-source acceptance remain pending.

| Domain                     | Status                          | Current state                                                                                                                                   |
| -------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Process supervision        | Complete                        | Electron attaches to `nvpair-service`, which owns `nvpair-ui-broker`; the broker supervises all workers                                                                   |
| Discovery                  | Complete                        | Broker discovery snapshots drive available nodes and node state                                                                                 |
| Node telemetry             | Integrated                      | Paired remotes use the broker-backed fresh getter; self and unpaired discovery retain direct `/v1/node-info`; remote OS remains backend-limited |
| Manual nodes               | Complete with local persistence | Broker owns probing, proxy registration, and persistence; Electron's old entries are migrated once                                              |
| llama.cpp backend          | Integrated                      | Desktop and TUI expose install, lifecycle, catalog pull, inventory, load/unload, endpoints, routing, and demo traffic                            |
| Local engine lifecycle     | Complete                        | Install, uninstall, update, state, and port configuration; no start, stop, restart or On/Off control |
| Remote engine lifecycle    | Partial                         | Remote install, status, and model pull are supported; no start or stop control |
| Engine models              | Partial                         | Core list, pull, load, unload, and supported delete actions are wired                                                                           |
| Engine version metadata    | Backend-limited                 | The backend status surface does not provide a complete version/install-owner/update contract                                                    |
| Errors                     | Complete                        | Broker-relayed registry with peer synchronization                                                                                               |
| Workloads                  | Complete after subscription     | Local and peer workload events feed one Electron catalog                                                                                        |
| Routing history            | Desktop and TUI presentation integrated | Desktop Settings → Routing and the TUI Routing tab use origin-local bounded history with matching state, attempt, money and wait semantics |
| Cluster pairing            | Complete                        | PIN pairing, identity, membership, leave, and removal                                                                                           |
| Cluster transport security | Backend-owned                   | Node-to-node transport security, including the proxies' cluster-mTLS inference ingress, is entirely backend; Personal AI Router implements none |
| Settings                   | Partial                         | Cluster identity plus per-engine ports and engine arguments, local and remote; inert backend settings are not surfaced                            |
| Model catalog search       | Backend-owned                   | `engine:catalog` serves the cached llama.cpp catalog and searches Hugging Face for it         |

## Role editing and API availability

The node settings RoleEditor is the sole role lifecycle editor: create, rename,
remove and reorder user-created names, including empty roles. Roles have no
built-ins or defaults and no cluster-wide store. Memberships reference ordered
node-local profile keys; deployment pickers select existing target roles only.
Local tier entries are retired. Widening uses the requesting node's role order.

Local rename/removal couples API settings through the same revision-checked
policy save. A nonempty `tiers[name]` or any `routing.tiers[name]` is configured;
an empty `tiers` list alone is absent. Rename refuses atomically when both names
are configured, including cross-map collisions. Otherwise configuration ends
under the new name, preserving destination-only configuration and deleting all
old-name keys, even empty ones. Local removal deletes the role's API settings.
Remote edits leave private API maps unseen and unchanged, send no configuration
intent and infer none from role changes. Every remote rename/removal confirmation
says API entries on the named node can't be seen from here and stay as they are.

Any declared role has exact `:api` and `:local` aliases in the catalogue, even
without an API entry. API selection requires a configured deployment and
`routing.tiers` remote permission, but public dispatch remains disabled in this
build: the API alias is listed and refused, and enabled deployments and remote
or paid permissions cannot be saved. Role edits do not activate dispatch or
coordinate configuration across the cluster. See
[Hybrid routing](../../docs/hybrid-routing.mdx) and
[model configuration](../../docs/model-configuration.mdx).

## Routing history availability and limits

Production connects proxy facts, broker accepted windows, the service relay and
desktop/TUI consumer banks. Public `routing:subscribe`, `routing:unsubscribe`
and `routing:get-initial` use empty-object params and the initial reply shape
`{requests: RoutingHistoryRecord[]}`; subscribed clients receive full-record
`routing:changed`. Private window/request/end/accepted methods never reach
clients, peers, managers or scheduler. The desktop already relays typed
`routing:active`/`routing:cancel`; history adds no cancel action.

History is private to the origin machine and memory-only. Broker restart clears
it; proxy death can lose unsent/partial windows while preserving accepted rows.
Clients clear banks on disconnect and resubscribe/refetch on reconnect. The
newest 256 requests retain at most 32 attempts with original indexes and
`droppedAttempts`. A model-less local destination keeps its node/engine and null
model (`model not specified`); API models remain required. Money stays exact
decimal nano-USD strings, waits are measured, and detached workload associations
remain historical. Completed delivery can have failed accounting. See
[the contract](../../services/nvpair-proxy/spec.md#1252-origin-local-routing-history)
for amount meanings and merge/association rules.

Eight service writer leases and one global initial transfer bound simultaneous
history writes without a subscriber cap. Busy keeps desktop/TUI loading with
validated rows and one bounded retry timer; ordinary RPC and inference continue.
Bank/frame capacities are representation limits, not RSS claims. Electron's
ready/pull/consumed loans are separate from the service relay.

Desktop Settings → Routing includes expandable history and identity-safe workload
summaries. The TUI Routing tab follows Jobs, with `r` refresh, `enter` attempts
and `esc` back. It drains coalesced ready tokens on the existing UI tick, showing
changes within one second. Narrow tables keep Model/State/Attempts; scrollable
detail retains destination/account IDs, fixed reasons, admission/first-byte waits
and HTTP status. Both clients label Ceiling, Cumulative granted, Current held,
Charged and Accounting separately, distinguish unavailable amounts from zero,
and retain Completed for delivered requests whose accounting failed.

Streaming paused, waiting to resume requires owner-reported recoverability;
Interrupted, resumed requires an actual linked successor. SC-b continuation is
not implemented, so a terminal exhausted interruption says Stream interrupted.
History has no cancel action and does not activate public providers. Component
fixtures do not establish installed/native acceptance or whole-J318 completion.

## Supervision

Electron starts only `nvpair-service` when needed; the service owns
`nvpair-ui-broker`. The broker owns:

- `nvpair-proxy`, one process hosting a facade per enabled engine, under one
  supervisor;
- `nvpair-node-scanner`;
- `nvpair-node-info`;
- `nvpair-manual-nodes`;
- `nvpair-workload-manager`;
- `nvpair-cluster-manager`;
- `nvpair-node-settings`;
- `nvpair-engine-manager`;
- `nvpair-errors`;
- `nvpair-job-scheduler`.

Worker crashes are handled by the broker; broker restarts belong to the service.
Electron reports service connection state and exposes explicit service controls.

## Discovery and nodes

Personal AI Router consumes `discovery:get-nodes` and `discovery:nodes-changed` from the
broker. The bridge:

- keys every node by the backend's stable per-host UUID (`AvailableNode.hostUuid`),
  the same identity workloads (`originatedFrom`/`scheduledOn`), errors (`nodeId`),
  proxy routing (`Node.ID`), `/v1/node-info` (`hostUuid`), and cluster membership
  (`ClusterNode.nodeUuid`) all use; the hostname is display only;
- preserves the backend's canonical reachable address;
- reflects the discovery `trusted` (locally pinned peer) and `clustered` (belongs
  to some cluster) flags, and uses `clustered` to mark an already-clustered
  discovered node non-invitable. `trusted` is a display annotation the scanner
  re-derives whenever the cluster-manager reports a pin-set change, and again on
  its periodic sweep of each peer's reported membership; it is not the routing
  gate. Each proxy answers "do I hold a pin for this peer?" from its own
  mesh when it resolves candidates, so PAIR must never treat `trusted` as
  authoritative for whether work can reach a node;
- merges per-engine proxy presence;
- carries flat and per-engine remote model inventories;
- emits node and discovery pushes to the renderer.

`selfId` is the cluster-manager's `nodeUuid` (from `cluster:get-node-id`), minted
at startup and resolved as soon as the broker is ready, so self is identified by
the same UUID key rather than a hostname guess.

The discovery snapshot does not include full dynamic telemetry. Electron
schedules CPU, memory, GPU, and VRAM refreshes separately. Paired remote nodes
use `discovery:get-node-info {hostUuid}` through the local service and broker;
the scanner owns admitted identity and direct/reverse route selection. Self and
unpaired discovery targets retain direct `/v1/node-info` polling.

Two limitations follow from the current discovery and node-info contract:

- **Remote telemetry.** An admitted paired peer can provide its full sample over
  the authenticated reverse session when inbound direct HTTP fails. If neither
  route works, Personal AI Router keeps its last-good or unavailable presentation
  without changing membership. A paired getter never falls back to an unpaired
  direct request.
- **Remote OS.** The contract does not report a remote node's operating system.
  The local node's OS is known from the running process; remote nodes fall back
  to a placeholder until the backend reports OS on discovery or node-info.

Manual nodes use the broker's `node/add`, `node/remove`, and `nodes/list`
surface. `nvpair-manual-nodes` persists the list itself. Electron's former
`configs/manual-nodes.json` is sent once as `node/add` per entry, then deleted.

## Routing and inference

All engine facades are broker-owned and cluster-aware. They live in one
`nvpair-proxy` process, each enabled after spawn on its own port, and each
serves its engine's dialect:

- the llama.cpp facade serves OpenAI-compatible routes and remaps
  `GET /v1/models` to the router's `GET /models`.

Sharing a process is what lets facades share the burst reservations the
scheduler depends on: facades bursting at once compete for the same node's GPU,
so a dispatch through any one has to be visible to the others.

Routing precedence is manual selection, scheduler priority, then deterministic
proxy ordering. Personal AI Router leaves proxies in automatic mode.

`nvpair-job-scheduler` combines total queued and running workload across all
enabled engines with a smoothed 0–3 GPU-pressure signal. The backend scanner and
manual node worker provide maximum-GPU utilization, while invalid, missing, or
older-than-10-second samples receive neutral pressure. The scheduler emits
order, pending count, and pressure; the broker forwards each `schedule:priority`
snapshot to the matching facade through `node/set-priority`. Each facade adds
local reservations, so its estimate is
`pending + gpuPressure + localReservations` during concurrent bursts.

Alongside `node/set-priority`, the broker sends the private process-scoped
`node/set-routing-costs` full replacement to every live proxy and replays its
current revision after spawn, independently of desktop subscriptions. It caches
newest usable destination-attested current speed evidence for tier scoring;
scoring does not consume it yet. PAIR consumes routing results, not this feed.

Current limitation: pressure does not represent GPU capability or available
VRAM, and a multi-GPU node is represented by its busiest device rather than
engine-to-device affinity.

### Secure inference transport

An NVPAIR-launched engine binds to loopback only; it is never directly
LAN-reachable. Each node's proxy exposes two personalities on one listener: a
loopback-only plaintext path for local clients, and a LAN ingress gated by
cluster mTLS that forwards trusted-peer requests to the loopback engine. Because
the engine port is private, discovery advertises the **promoted proxy port** for
`ol`/`lm`/`lc`, and the peer's real engine port is knowable only from
authoritative `engine:remote-get-installed` facts.

Admitted peer dispatch prefers that advertised direct facade and can carry the
same request through the existing authenticated reverse session after a bounded
pre-dispatch failure. The reverse stream still enters the ordinary mTLS facade,
so receiver admission and the no-onward-routing rule remain unchanged.

Personal AI Router consequences (all reflection, no security implementation):

- The proxy `node/*` presence port is the peer's promoted proxy port. Personal AI Router
  surfaces it as `EngineStatusData.proxyPort` and never labels it as the engine
  port; a remote engine's port comes only from `ec` facts.
- Because both peers must speak the mTLS channel, mixed-version clusters cannot
  run inference across the version boundary. Local use and the shared
  nearby-model list are unaffected.

### llama.cpp support

The bundled backend manifest can install and start `llama-server`, list exact
router model ids, stream model downloads over SSE, load or unload a model, and
delete native cache entries by exact id. `LLAMA_CACHE` points at the manifest's
declared `models_dir`, `~/.llamacpp`, so downloaded models survive both an
engine uninstall and the app-level "remove all data" uninstall — the latter
deletes the whole app data root, so a model store inside it would not.
Loaded models enter llama.cpp sleep mode after five idle minutes, release model
and KV-cache memory, and wake on the next request. The router child remains alive
and can retain a residual backend GPU context.

Windows and Linux installs use checksum-pinned server and CUDA-runtime archive
pairs; macOS uses the standard Metal-capable archive. The on-demand download is
roughly 0.6–0.8 GiB and is not part of the application installer. GPU layers
remain `auto`, allowing supported NVIDIA/Metal acceleration and dynamic CPU
fallback; hardware acceptance is still required to confirm acceleration.

The facade is in the default broker and TUI set: local OpenAI-compatible clients
use the broker-reported listener (normally `8080`) while the managed router runs
on `8081`. Desktop and TUI expose install, lifecycle, download progress,
inventory, load/unload, endpoint, routed state, and inference-demo workflows.
Both also expose model deletion.
Desktop browsing populates a six-hour cache from the 50 most-downloaded GGUF
repositories for each approved publisher and can explicitly search up to 50
public Hugging Face matches. Results are limited to pull-ready `Q4_K_M` IDs.
The TUI keeps its direct exact-ID download prompt.

Current limits are explicit: the manual-node worker does not probe llama.cpp,
the catalog offers no alternate quantizations, and package selection is fixed
rather than driver-aware or offline-repacked.

## Engine lifecycle

`nvpair-engine-manager` is authoritative for installed, running, healthy, and
port state.

Personal AI Router supports local:

- install and uninstall;
- managed update;
- engine and proxy port changes;
- desired-state restoration across app restarts;
- engine and model progress.

On shutdown the broker stops the proxy, then calls `engine:prepare-shutdown`,
then waits for each worker to exit without force-killing it, so engines are not
orphaned during teardown. `engine:prepare-shutdown` stops managed engine
processes without changing the persisted desired state; the broker restores
enabled engines on the next launch. Personal AI Router leaves that ordering to the broker
rather than stopping the engines itself first. Stopping a managed engine itself sends one stop signal and
waits for it to exit with no timeout: SIGTERM to the process group on Unix
(never escalated to SIGKILL) and `taskkill /T /F` on Windows (its windowless
engines cannot receive a graceful close).

Start, stop and restart RPCs and peer start/stop routes are removed.
`engine:wake` and `engine:sleep` are broker-private and refused from clients
and peers. Settings application may still restart an engine internally.

Desktop renders reported engine state and availability. There is no saved
engine On/Off state. Installed engines start with the available service, after
installation and Resume, and recover after a crash or admission wake. PAIR
releases idle memory per model and applies one capacity limit to inference and
manual loads. Pause and Halt remain the availability controls.

Install still sends `engine:install` with `start: true`; the install start
option remains. First-open desktop startup does not start installed engines.
`nvpair-engine-manager` retains desired-state persistence and restoration until
phase 15. This does not yet make every installed engine run while available.
Phase 11 must not be installed without phase 15.

Personal AI Router synthesizes one centralized pending-action state while waiting for
terminal backend notifications. It never treats optimistic state as
authoritative.

Remote cluster support includes status, install, model pull, and
remote model load, unload (eject), and delete via the `ec` surface. Uninstall
and update remain local-only. A refusal of any of these with `-32011`
is reported with the neutral veto plus the
target's own message, and offers no retry.

Server port, proxy port, and engine arguments are one authoritative record owned
by the node running the engine, read and written through the broker's
`engines:{get,preview,apply}-settings` channels. A request naming another node
is relayed to that peer, so a clustered device is edited like the local one. The
one exception is managed CORS origin settings: the owning node rejects a change to
browser access policy relayed from a peer. Other environment assignments pass
through literally, with no maintained catalog of engine options or variables.

## Models

Local model state comes from canonical `engine:models`. Remote model state comes
from node-scanner enrichment and is attributed per engine.

The desktop copy control API accepts independent source and destination UUIDs,
an exact model, manager engine ID and operation ID. It exposes typed capability,
start, cancel and active-snapshot calls; Copy-to-this-PC delegates to the same
coordinator. Backend progress/result fields preserve export identity, byte
counts, serving-setup requirements and uncertain receipt durability. Progress
is keyed by the full tuple at the destination, active state hydrates from Go,
and lost results remain unknown. Destination status and scanner-owned remote
inventory refresh without controller artifact I/O or synthetic model rows.
The independent source/destination picker is not part of this bridge contract.

Personal AI Router uses:

- `engine:models`, selecting the requested engine's `modelsByEngine` inventory;
- `pull_model`, whose body is `{ model }`;
- the typed `engine:load-model`, `engine:unload-model` and
  `engine:delete-model` operations, which engine-manager maps onto each
  manifest's `operations` bindings: llama.cpp `load_model`, `unload_model`
  and `delete_model` (`DELETE /models` with URL-encoded query params).

llama.cpp exposes Load, Eject, and Delete in the model manager. Any
other engine engine-manager registers, such as a locally configured fork, is
listed once installed, with its descriptor's label and only the operations its
`engine:descriptors` entry declares. An engine without a descriptor here (a
paired node's variant) shows status without start/stop controls.
User-configurable keep-alive / expiry controls remain unsupported; managed
llama.cpp uses its fixed five-minute idle sleep.

The bundled llama.cpp manifest declares no `restart_after`: llama.cpp reflects a
deletion in-process, so there is no engine bounce and no restart confirmation.
llama.cpp may stop the selected model while removing its cache entry, but the
router and its other models stay up. `tests/modular/delete-model-restart.test.ts`
reads the shipped manifest and asserts it omits `restart_after`. The desktop
carries no capability flag for a restart-backed delete.

Several model-action timeouts have to be ordered correctly.
`MODULAR_MODEL_ACTION_TIMEOUT_MS` anchors the model-action observers, and a load
whose descriptor declares a longer response budget gets a separate allowance:

| Budget                                                        | Value | Why                                                                                                                                                                                                                           |
| ------------------------------------------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MODULAR_MODEL_ACTION_TIMEOUT_MS` (`modular-runtime.ts`)      | 120s  | The RPC must outlast a full stop + readiness-probed start.                                                                                                                                                                    |
| `engineResponseHeaderTimeout` (`executor.go`)                 | 30s   | Downloads and ordinary local HTTP actions retain a prompt response-header bound; probe contexts stay shorter.                                                                                                                 |
| `pullProgressTimeout` (`executor.go`)                         | 30min | Streaming llama.cpp pulls may run indefinitely while file completed bytes advance; 30 minutes without advancing bytes cancels the stalled pull.                                                              |
| `PENDING_TIMEOUT_MS`                                          | 60s   | Unchanged for every other command, including a delete.                                                                                                                                                                |
| `remoteReadyResponseHeaderTimeout` (`remoteclient.go`)        | 11min | Remote start/delete responses can wait on engine readiness, and a cold model load can withhold headers for minutes, so those calls use the readiness-sized client. Other model actions retain the ordinary 30s budget. |

The net is a backstop, not the mechanism: a delete now ends the spinner on real
backend truth either way. On success the refreshed canonical inventory drops the model;
on failure the reported error carries `nodeId` + `engineType` + `modelName`, which
is what lets the store attribute it and clear that row.

Local Load also observes its eventual JSON-RPC rejection and reports the
same model context; the resulting attributed `errors:update` clears `Loading…`
without waiting for the net.

A failed restart is reported as `Deleted <model>, but <engine> failed to
restart: …`, never as a failed delete — the files really are gone, and telling
the user otherwise invites a retry that hits `model … not found on disk`. The
bounce itself raises no crash or unhealthy alert: `markStopped` cancels the
health loop and clears `unhealthy`/`exited`, and `doStart` clears them again, so
the engine simply shows as stopped and then running.

A local `pull_model` streams target-owned lifecycle snapshots through
`engine:pull-progress` (`{engine,op,stage,percent?,id,model,downloadState}`).
`engine:remote-progress` adds `node` and the distinct request `opId`. No
single-model-per-engine initiation map is needed: target-local, origin and
freshly attached clients can project queued, in-progress, failed, complete and
unknown operations from the same exact target/model/download identity. Unknown
percent remains indeterminate; RPC completion/timeout does not clear a newer
authoritative operation. Copy and installation keep their existing ownership.

`downloadState` is a full bounded version-1 snapshot with `sourceId`, decimal
uint64 `revision`, `recoveryRequired`, and `entries`. Entry fields are `id`,
`engine`, exact `model`, `phase`, backend-derived `blocksReadiness`, optional
integer `percent`, bounded `reason` and optional `persistence:"uncertain"`.
That qualification reports landed marker removal without confirmed directory
durability. Snapshots allow at most 64 entries and 256 KiB, with up to 32 unsafe
attempts and 32 recent outcomes; the separate unresolved-content journal is
bounded to 64 KiB. Recovery preserves unsafe markers as unknown without
automatic retry, cleanup or model effects.

The bridge rejects malformed extensions, lower revisions and retired sources.
Notifications alone cannot choose or replace a source baseline, even for an
unseen retired worker. A mismatching local notification triggers a coalesced
current `engine:models` read; remote notifications wait for authenticated
discovery. Worker-ready and broker-session generations retire outstanding reads.
It retains last-good names separately from current inventory. Discovery carries
optional `downloadStateCurrent`: false on failed fetch/source retirement, absent
for unsupported data, and true after a successful current fetch (including a
same-revision refresh). Unsupported/stale/recovery-required data never grants
readiness. Other observers use the existing scanner refresh cycle, normally
15 seconds plus preceding refresh work and request budget; origin/target
lifecycle pushes are immediate. These guarantees require a coordinated fleet
update. Installed-machine, native UI, and isolated multi-process acceptance
remain separate validation obligations.

The source regression `TestModelDownloadStateProcess` in `services/tests` covers
real service/broker/manager relays, fresh-client hydration, pinned observer reads,
scanner loopback self-enrichment, queued-model isolation, native failure and
disconnect outcomes, restart markers, readiness refusals and bounded retention.
Its fake router uses synthetic catalog IDs and tiny target-only shards; the
controller has no model store or model-serving listener. On Windows at base
`44e76c6b` (2026-10-26), its final plain run passed 17 named terminals with no
failures or skips. Race had 15 passes and two failed terminals after a fixed
scanner image replacement was refused during setup; the affected leaf and its
parent then passed a targeted race rerun. No data race was reported, and module
vet exited zero. The failed aggregate race run is preserved. These runs include
second-response NDJSON, replacement-route reads, control-POST counts and terminal
subscriber convergence.
Independent scanner peer refresh, automatic resident dispatch,
the full progress-watchdog interval, lower-revision/retired-source notification
injection, and exact crashes inside journal replacement
are not established by this fixture. Its ec bind is loopback; existing em and
cluster-manager listeners still bind wildcard addresses on private ports. Joined
peer-transfer receipt assertions and native UI acceptance remain separate.

`TestModelDownloadStateTransferJoinProcess` adds source assertions on the joined
download/COPY implementation at `efda20f7691a01980dd4740789d5d4829f6ba43b`.
Windows plain and race runs passed on 2026-10-26 with three named terminals each,
no failures or skips, and module vet exited zero. The whole-suite run is separate.
Real custom export list/GET positive controls precede active
and failed catalog-pull fences, with blocked COPY required to dispatch no
artifact GET and leave destination artifacts and mutation effects unchanged.
A real, independently hashed COPY receipt is then kept separate from a failed
same-key catalog pull: an identity-bound verified retry may report COPY success
while the catalog snapshot/journal and ready-dependent export fence remain
unchanged. Explicit successful catalog retry is tested separately as the source
readiness restoration path. This does not broaden the Task 5 or native UI proof.

`engine:models` (and the `em` `GET /v1/models` surface) returns the flat model
union, the per-engine breakdown (`modelsByEngine`), the per-engine set of
models loaded in memory (`loadedByEngine`), and `downloadState`. The engine-manager
watches inventory/readiness and loaded state and pushes `engine:models-changed`
(`{ engine, models }`, the full `engine:models` shape) on explicit load/unload,
JIT auto-load, and TTL/idle eviction. Discovery carries
`loadedByEngine` onto each node's `AvailableNode` so remote cards reflect
residency too.

Personal AI Router consumes both: `parseBrokerNode` records `loadedByEngine` on every node
(local via loopback self-enrichment, remote via the peer's enrichment), and
`applyLocalModelsResult` (`modular-supervisor.ts` → `modular-state.ts`) applies
the `engine:models-changed` snapshot to the local node immediately. Model rows
are stamped `ModelItem.status: 'loaded'` when a name is in that engine's loaded
set, which drives the loaded dot, disables **Load** for a resident model, and
gates **Eject** to loaded models only (`ModelRow.tsx`). The optimistic
load/eject pending-action clears on the resulting model patch instead of its
safety-net timeout (`pending-actions.store.ts`). Loaded state carries no
`sizeVram`/`expiresAt` — the backend delivers the simpler `loadedByEngine`
name-set, not structured details. Rows include lifecycle-only and loaded-only
names. `downloaded:true` requires current canonical inventory and current
supported lifecycle state with no recovery requirement or matching readiness
block; even a complete operation waits for a current inventory read. Loaded
observations remain independent, including a loaded model whose content is
currently unsafe.

Desktop retains the per-engine list and Add model catalog. Exact target/engine/model
rows show queued, downloading, failed, complete, or unknown download status, with
an accessible small determinate bar only for a measured current percentage and an
indeterminate bar otherwise. Stale observations explain disconnected/unavailable
status and never show completed progress. Load and Copy require authoritative
current readiness; active pulls additionally disable Eject and Delete. Both menu
flags and selection handlers enforce those conditions, including open-menu races
and delete confirmation. The Go admission gate remains the final authority.
Pending pull optimism hands off only to a matching new active download ID and
never treats a listed name as completion. Failed partial models stay in the
catalog for an explicit download retry; no automatic retry or engine start occurs.

The model catalogue is owned by the backend: `engine:catalog` on
`nvpair-engine-manager` serves the llama.cpp
list, and searches Hugging Face for llama.cpp, and both the desktop app and
the terminal interface browse it. Electron only relays the call and maps rows
for the renderer.

## Errors

`nvpair-errors` is broker-owned and is the authoritative error registry. Personal AI Router:

- fetches `errors:get-initial`;
- consumes full `errors:update` snapshots;
- forwards clear and report actions through the broker;
- uses engine operation metadata to clear matching pending actions;
- renders supported retry hints.

Peer error synchronization and authentication are backend-owned.

## Workloads

Personal AI Router subscribes to the broker workload stream and maintains an Electron-local
catalog keyed by workload origin and ID.

`scheduledOn` identifies the execution node. Workloads that have not yet been
scheduled are not attributed to a node.

The broker backs workloads with a durable, order-independent store and answers
`workloads:get-initial` with an authoritative baseline (current plus
recently-terminal jobs), and the workload-manager backfills a joining node with
peers' in-flight work and periodically re-syncs. This makes cluster job counts
converge across restarts and network hiccups on the backend side. Personal AI Router's
`workloads:get-initial` handler fetches that broker baseline (falling back to its
own subscription-built catalog if the call fails), and the supervisor seeds the
catalog from it right after `workloads:subscribe`, so a freshly started or
restarted app immediately shows cluster-wide in-flight jobs. The renderer store
subscribes before fetching the baseline so a transition during init is never
dropped.

## Cluster identity and membership

`nvpair-cluster-manager` owns:

- node crypto identity;
- interactive six-digit PIN pairing;
- bounded, service-owned ask-to-join admission requests and their separate
  eligibility/snapshot API;
- trusted certificate pins;
- membership snapshots;
- cluster leave and member removal;
- trust endorsements, roster reconciliation, and durable admission-bound
  removal proofs.

Personal AI Router creates a cluster automatically when the first outbound invite requires
one. A failed invite can abandon that unused solo cluster. The UI displays the
inviter's PIN and requires the invitee to enter it.

The cluster-manager emits `cluster:invite-received` exactly once per inbound
invite (always `pending`) and exposes no list-pending-invites RPC. To give the
receiver a durable, recoverable set of pending invites, Electron main accumulates
them (`ModularState`), prunes them as they resolve — via membership
(`nodes:changed`), the local accept/decline result, the receiver-side
`cluster:invite-canceled` and `cluster:invite-expired` pushes (both consumed
directly so the PIN prompt dismisses at once when the inviter cancels or the
invite times out), and a per-invite `cluster:invite-status` sweep (pruning
terminal or evicted sessions). The authoritative set is served by
`cluster:get-initial` (`pendingInvites`) and broadcast on every change via
`cluster:pending-invites-changed`.

The cluster-manager expires unanswered invites after a TTL (default 5m) on
**both** sides: it expires its own **outbound** invite and, on the receiver,
expires the **inbound** invite and signals the inviter for immediate teardown. It
emits `cluster:invite-expired` on the receiver — which Personal AI Router
consumes to prune the inbound invite — and on the inviter, alongside
`cluster:invite-declined`. Personal AI Router reconciles the **outbound** side
through the `cluster:invite-status` poll in `useInvitePairing` and dissolves any
solo cluster it auto-created only to back the invite via
`cluster:abandon-if-solo`, so `cluster:invite-declined` stays an inviter-side
latency optimization it does not consume (recorded in
`docs/service-contract-exceptions.json`). Because the backend owns receiver-side
expiry, Personal AI Router keeps no client-side inbound TTL; the
`cluster:invite-status` sweep is the sole backstop for a missed push.

Personal AI Router treats the `canceled` and `rejected` invite states as terminal non-paired
outcomes (coerced by `parseInvite`, which also carries the backend `reason`). The
outbound Cancel action calls the `cluster:cancel-invite` RPC, so canceling an
in-flight invite invalidates the PIN backend-side and best-effort notifies the
joiner to drop its prompt (the receiver reflects this via the already-consumed
`cluster:invite-canceled` push). A `rejected` result surfaces its reason
distinctly — `already-clustered` renders as "That node is already in a cluster" —
and Personal AI Router pre-empts the invite entirely by disabling an already-`clustered`
discovered node.

A wrong PIN is a terminal, non-retryable failure: the invitation ends on both
sides and the user must request a fresh one. The cluster-manager mirrors the
failure — the joiner fails the completion with `reason: 'incorrect-pin'`, clears
its `pending-inbound` member (`nodes:changed`), and best-effort notifies the
inviter over the pairing channel (`phase: "fail"`), which flips its outbound
invite to `failed` (same reason), tears down the EAP session (invalidating the
PIN), and pushes `cluster:invite-failed`. Personal AI Router consumes this end to end: the
inviter surfaces the "Incorrect PIN" error from the `cluster:invite-status` poll
returning `failed` with `reason: 'incorrect-pin'` (rendered by
`InvitePairingPanel`), and `cluster:invite-failed` is consumed
(`modular-supervisor.ts`) to prune any matching inbound invite. The joiner's
`pending-inbound` row clears via `nodes:changed`, and both surfaces normalize
`-32001` / "invite session evicted" to friendly copy (`cluster-invite-error.ts`)
as a backstop. Other `failed` causes (peer unreachable, malformed) carry an empty
`reason`, so Personal AI Router falls back to generic "Pairing failed" copy.

The PIN is a convenience pairing code and should not be described as a strong
authenticator.

### Ask-to-join service surface

The service contract supports requesting admission from one selected existing
member. This section describes the API and its ownership boundaries.
Ordinary Invite keeps its existing clustered-target rejection. There is no
cluster merge, automatic join, broadcast approval or outbound-only requester
transport. The existing broker/service generic relays remain unchanged.

`cluster:get-join-requests` and `cluster:join-requests-changed` provide the same
full manager-owned eligibility/request snapshot, separate from
`ClusterInitialSnapshot`. `cluster:ask-to-join` requires explicit
`replaceSoloCluster`; `cluster:respond-to-join-request` requires an admitting-
device `allowRemoteManagement` boolean on approval; `cluster:cancel-join-request`
cancels an outgoing request. Shared types, strict parsing, invoke registration
and notification demux carry these contracts without a new ignored-method
exemption. Claimed identities are unverified; snapshots contain no PIN/security
material. Fetch the dedicated getter on attach/reconnect, including requests
received with no GUI attached.

The member approves locally and displays the existing PIN through local
`cluster:invite-status`; the requester enters it through ordinary acceptance
with its own grant choice. The exact child invite and authenticated
`PairingInfo.joinRequestId` correlate the transaction. Approval/status alone is
not Joined: canonical confirmed durable pairing/receipt state owns success.
Wrong-PIN, pending confirmation, rollback, compensation and grant independence
remain with the existing transaction.

Go alone owns source/rate/capacity bounds, five-minute request expiry,
composition eligibility and consented durable empty-solo replacement after
approval. Retained peer relationships (offline included) and uncertain stores
refuse safely. Client detach preserves pending work; manager restart invalidates
requests, so clients read actual membership before retrying. An old local
method-not-found disables this feature; initial peer 404 means `unsupported-peer`,
later status 404 means lost status. No fallback Invite or pre-approval teardown.
Broker code/message survive relay; eligibility reasons come from the snapshot
because error data is dropped. Full limits and shapes:
[cluster-manager spec §7.0.1](../../services/nvpair-cluster-manager/spec.md#701-ask-to-join-admission-requests).

## Security

Cluster identity, pairing, trust, membership, and all node-to-node transport
security are owned entirely by the backend. Personal AI Router implements none of it: it does
not read private keys, build TLS contexts, or authenticate peers. This includes
cross-node inference: the proxies terminate the cluster-mTLS ingress and gate it
on cluster pins, and NVPAIR-launched engines bind to loopback so they are
reachable off-box only through that ingress.

The one Personal AI Router-facing constraint is that trust begins with a low-entropy human
PIN, so UI copy must not present the PIN as a strong authenticator.

## Settings

Personal AI Router uses node settings for persisted cluster identity and friendly name
synchronization.

Per-engine settings are separate and owned by `nvpair-engine-manager`: the
server port, the proxy port, and the engine arguments, persisted as a manifest
override that survives a restart. The broker owns the combined operation —
journal, validation, stop, proxy rebind, restart, and crash recovery — so both
the full editor and the port-only `engine:set-port` RPC used by the TUI go
through the same path.

Backend settings with no active behavior are intentionally not exposed in the
UI. A setting should be surfaced only when a backend component consumes it and
its effect is observable.

## Current client-side responsibilities

The following responsibilities remain in Personal AI Router because the backend does not
provide an equivalent client-facing contract:

| Responsibility                                              | Location                                         |
| ----------------------------------------------------------- | ------------------------------------------------ |
| Schedule and merge node telemetry; call the paired getter and directly poll self/unpaired nodes | `node-info-poller.ts` |
| Migrate Electron's old manual node file once               | `manual-nodes-migration.ts`, `modular-supervisor.ts` |
| Bridge the local node into engine proxies                   | `modular-supervisor.ts`                          |
| Present optimistic engine transition state                  | `pending-actions.store.ts`, bridge state         |
| Relay the backend model catalogue to the renderer           | `service-bridge/model-catalog.ts`                |
| Accumulate and reconcile receiver-side pending invites      | `modular-state.ts`, `modular-supervisor.ts`      |
| Mirror backend-coupled runtime defaults not yet reported    | `modular-runtime.ts`                             |
| Collapse a superseded node row before the scanner proves it | `modular-state.ts`, `modular-runtime.ts`         |

These are current integration boundaries. Do not add compatibility shims or
duplicate backend services around them.

The last row is a display-side collapse, not a directory decision. The scanner
treats a matching address, hostname, and `lastSeen` gap as grounds to ask the
machine who it is, and evicts only on a node-info identity mismatch; the bridge
has no such proof and hides the row on the suspicion alone. It must never be
extended into anything a consumer reads as an eviction.

## Unused backend methods

The generated `docs/services-api.md` is authoritative for methods handled by
the backend but not called by Personal AI Router. Typical optional opportunities include:

- engine manifest description;
- engine log and error-ring views;
- scheduler status and interval controls;
- proxy selected-node observability;
- explicit unsubscribe calls for process-lifetime subscriptions.

Unused methods are not automatically product gaps. Adopt them only when they
serve a current Personal AI Router feature.

## Maintenance

On every backend update:

1. run `npm run service-contracts`;
2. resolve missing notifications or document why they are internal;
3. verify meaningful payload fields are consumed;
4. update this present-state status;
5. run `npm run service-contracts:write`;
6. run `npm run service-contracts:check` and `npm run typecheck`.

Do not append release diaries or prior-state narratives. Git history is the
record of how this integration changed.
