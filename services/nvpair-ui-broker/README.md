<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvpair-ui-broker

The [engine settings protocol](ENGINE_SETTINGS.md) documents local/paired
launch and port editing, revision checks, serialized application and recovery.
Its [engine switching contract](ENGINE_SETTINGS.md#engine-switching) covers
`engine:switch`, status/progress, acknowledged source drain, durable receipts
and interrupted-operation reconciliation. The broker owns transaction order;
the engine manager owns process lifecycle and intent. Switch recovery precedes
settings replay and enabled-engine restoration. Local Off reaches the manager's
durable veto path during remote reservations without waiting for the transaction
lock. Forward steps are pinned to the preparing manager incarnation; replacement
interrupts the operation and gates the new manager until reconciliation finishes.
The gate applies to forward operations, including on the initial spawn. The
manager's discovery subscription and node/settings projections still reach the
current worker during recovery, including authoritative local-pause state and
log-level changes. Mutating direct peer control also remains fenced inside the
manager until the broker acknowledges successful recovery for its incarnation.
Off/pause admission and terminal completion share a gate, so an admitted veto
cannot produce a successful receipt while a stop response is pending. Broker
cleanup carries the operation-specific Off/pause veto to the manager before
reconciling a committed target. A veto admitted during cleanup requires another
reconcile or retained recovery ownership; checkpoint release refuses an
uncleaned committed target with a durable veto. A delayed Off response cancels
only its original operation. The internal local stop carries that operation ID;
manager admission makes the entire delayed request a no-op once the reservation
is released or belongs to another operation, preserving later explicit On intent.
Off/pause during retained recovery latches against its checkpoint identity;
recovery rechecks after reconciliation and drain release, repeats cleanup for a
new veto, and orders checkpoint release with that admission gate.
Admission first persists an internal operation-keyed local obligation in the
switch journal; failure visibly refuses the local request. Before terminal release
or recovery resolution, the broker holds the operation lock while discharging
every obligated engine through a verified ordinary stop on the current manager.
Pause must be persisted and applied too; its operation-specific completion proof
is saved before a later local Resume can run. Cleanup preserves that Resume and
uses the saved proof without re-pausing. Older journals without the proof retain
the current local-pause check. Discharge is durably recorded; failure
retains recovery ownership. Restart replays undischarged obligations even with no
manager checkpoint, before restoration/readiness. An older direct peer start loses
to the subsequently admitted local Off; later local On waits for this discharge.
Explicit local start, restart and install with `start: true` share the operation
lock and recovery reservation check, including installs without a port override
and the desktop install, installAll and update commands that send them.
Install preparation runs without either global lock; only its separate start
uses lifecycle admission. A local Off of that engine, or a local pause, admitted during preparation supersedes
that older start even after switch discharge. The install remains completed,
but its start returns busy. A later On retains normal lifecycle ordering.
The journal lock is released during lifecycle worker waits.
Clients cannot relay the manager's private switch prepare, step or reconcile
methods; only the coordinator dispatches those under its operation lock.
Worker exit is fenced before supervisor
notification, with already-closed worker
channels checked again at completion. Transient conservative recovery failures
retry with bounded backoff on the same replacement; exhausted recovery remains
hidden and reports a sticky error. Journal/storage uncertainty stays blocked.
Each of the three recovery attempts has a 60-second cleanup budget, plus
one-/two-second backoffs: RPC budgets can occupy the synchronous spawn path for
about 183 seconds. Entry lock waits precede those budgets, so this is not an
unconditional end-to-end bound. A simultaneous drain-release failure cannot
downgrade a permanent storage failure.

A Go service that exposes the NVIDIA Personal AI Router API to **UI processes**
and other clients (CLIs, dashboards, mobile companions, scripts, tests, etc.).
The shipped graphical UI is bundled alongside the backend services and launches
the broker from the same installation directory. The broker is the parent
process and canonical backend entry point: it supervises the worker subprocesses
on the UI's behalf, speaking JSON-RPC over stdio.

The broker supervises **ten** worker subprocesses, so a client gets the whole
backend behind one endpoint. Each is spawned at startup and relayed under its own
namespace:

| Worker | Role | Client namespace |
| --- | --- | --- |
| `nvpair-node-scanner` | Discovery daemon: advertises this host's one `_nvpair-node._tcp` record and browses the LAN | `discovery:*` |
| `nvpair-node-info` | Local GPU / CPU / memory inventory over HTTP at `/v1/node-info` | — (HTTP only) |
| `nvpair-proxy` | One process hosting an inference proxy and router facade per enabled engine | `ollama-proxy:*`, `lmstudio-proxy:*`, `llamacpp-proxy:*` |
| `nvpair-engine-manager` | Local engine and model control plane; also serves `GET /v1/models` to peers | `engine:*` |
| `nvpair-cluster-manager` | Node identity, trusted-node store, PIN pairing | `cluster:*`, `nodes:*` |
| `nvpair-workload-manager` | Cluster workload relay between this node and peers | `workloads:*` |
| `nvpair-job-scheduler` | Ranks nodes by pending work and GPU pressure for the proxies | _internal_ |
| `nvpair-errors` | Service-error datastore and cross-node sync | `errors:*` |
| `nvpair-node-settings` | Typed per-node settings store | `settings/*` |
| `nvpair-manual-nodes` | User-added nodes, merged into the discovery snapshot | `node/*`, `nodes/list` |

Only the scanner is required. Every other worker is optional: a missing binary
leaves the broker running without that capability rather than failing to start.
The [worker subprocesses](#worker-subprocesses) section covers each one's link,
lifecycle, and relay rules.

These responsibilities live in the broker itself rather than in a worker:

- **Engine advertising.** The broker polls local Ollama, LM Studio, and
  llama.cpp every 5 s and registers each running engine's promoted facade port
  (`ol` / `lm` / `lc`) with the discovery daemon, so they are carried in this
  host's single `_nvpair-node` record. The
  model list is not part of that record — it is served over HTTP by
  `nvpair-engine-manager` on the `em` service and fetched by a peer's daemon
  during discovery enrichment.
- **Workload brokering.** The broker stamps the local node id onto the
  `workload:*` events a proxy emits, applies them to its authoritative store,
  feeds the scheduler, and forwards them to the workload manager for broadcast.
  Peer-origin updates come back the other way as `workloads:upsert` /
  `workloads:remove` for subscribed clients.
- **Scheduling telemetry.** Scanner and manual-node observations populate a
  source-aware cache keyed by `hostUuid`; scanner data takes precedence. The
  broker advances sample age locally, feeds compact maximum-GPU telemetry to the
  scheduler, and replays it after a scheduler restart.

### Supervision & recovery

Every worker runs under a supervisor that auto-restarts it on an unexpected exit: exponential backoff (~1 s → 16 s), a budget of 5 attempts per unhealthy streak, and a 60 s "healthy reset" window after which a restarted worker is considered recovered. A crash is surfaced as a sticky `supervisor:subprocess-crashed:<name>` entry in the error registry (so connected tools can show "this helper is down"); recovery clears it, and a worker that exhausts its restart budget leaves the entry up. The scanner is the one worker whose **initial** spawn is fatal (the broker has no job without discovery); every other worker is optional and a failed first spawn just runs the broker without it. `nvpair-errors` is a special case: it can't report its own death into itself, so its crash logs to stderr (rather than the pipeline) while still being auto-restarted.

There is **one supervisor per process, not per engine**. `nvpair-proxy` hosts a facade for every enabled engine, so its crash entry names the process (`nvpair-proxy`) and its restart brings every facade back together; a terminal failure releases each engine's port-ownership gate individually, so a caller waiting on one engine is never held by another's state.

The proxy also runs for the **routing listener**, which belongs to the process
and needs no engine. A spawn in which no facade comes up still fails and is
retried as before, unless the persisted policy enables `routing.listener`: then
the process is kept and serves the listener, and any facade that failed is
terminal for it, exactly as when another facade is serving.

- **Startup.** Serve creates the one supervisor when a facade is configured or
  the saved policy enables the listener (and a proxy binary is resolved).
- **Policy changes.** Only a change to `routing.listener.enabled` or
  `routing.listener.port` requests lifecycle work. Other policy edits never
  start, stop, replace or renew supervision, including after restart-budget
  exhaustion.
- **No serving facade or recovery in progress.** A `policy:set` that enables
  the listener starts a missing supervisor, including after initial facade
  startup failed. Disabling the listener stops a process whose completed
  bring-up yielded no facades, including when every configured facade failed:
  no idle proxy remains. A supervisor whose restart budget is exhausted may
  likewise be retired and replaced for an enabled listener when a listener
  change is pending. Exhaustion alone never starts a fresh supervisor or renews
  its restart budget. Neither start nor stop runs on the request: `policy:set`
  returns within its own activation bound (5 s) with the new revision saved, and
  a background pass starts or stops the proxy. Until the new process has
  acknowledged the policy it is replayed, the revision is `pending`
  (`policy:changed` announces the activation then), and after a disable there is
  no proxy to acknowledge anything. Toggles made while a pass runs are folded
  into one more pass that reads the policy afresh, so the last change wins.
  A change deferred during bring-up or recovery stays pending; completion of
  bring-up or publication of exhaustion wakes that pending change.
  The start can take as long as a proxy start does, a stop as long as the proxy
  takes to exit, and neither can delay a `policy:set`.
- **A serving facade or recovery in progress.** A listener change never starts,
  stops or replaces a supervisor while a facade is serving, while facade
  bring-up is unfinished, or while a crashed process is awaiting a restart
  with budget remaining. Configuration alone does not preserve it. The
  running proxy serves or closes the listener as the pushed policy says.
- **Ordering.** A policy change cannot start the proxy before startup has
  decided whether it runs. Once the Serve context is cancelled or proxy
  shutdown is entered, pending and new reconciliation cannot begin a spawn;
  shutdown publishes its closing state before waiting for lifecycle work. A
  spawn that had already passed that check can still create its process, and
  shutdown then stops it. A respawn refused because shutdown began does not
  count against the restart budget.
- **Port overlap.** The listener and a facade cannot share a port. `policy:set`
  refuses (`-32602`, message starting `routing.listener.port`) an enabled
  listener whose port a configured facade uses: its compatibility port or the
  one persisted in its place, the port planned for its next spawn, the port it
  bound, or the inherited Ollama alias. `<engine>-proxy:set-port` and
  `engine:set-port` refuse (`-32000`) the port of an enabled listener, like the
  alias port. The node policy, the scheduler ranking and `log/set-level`
  reach the proxy through the process, so they also reach one that serves no
  facade.

## Communication

Bidirectional newline-delimited JSON-RPC 2.0 — same conventions as every other NVPAIR subprocess. By default the broker reads stdin and writes stdout; with `--ipc <path>` it dials a Unix domain socket or Windows named pipe instead. The parent UI process owns the endpoint in both cases: it either spawns the broker with piped stdio, or sets up the pipe / socket beforehand and points the broker at it via `--ipc`. The broker always talks to exactly one peer.

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--ipc <path>` | _(stdio)_ | IPC endpoint to dial: Unix socket path or Windows named pipe (e.g. `\\.\pipe\nvpair-ui-broker`) |
| `--scanner-path <path>` | `./nvpair-node-scanner[.exe]` in the CWD | Explicit path to the `nvpair-node-scanner` binary the broker should spawn |
| `--node-info-path <path>` | `./nvpair-node-info[.exe]` in the CWD | Explicit path to the `nvpair-node-info` binary the broker should spawn. When omitted and no default sibling exists, the broker runs without the local inventory server (non-fatal); when set to an invalid path, the broker exits with an error |
| `--proxy-path <path>` | `./nvpair-proxy[.exe]` in the CWD | Explicit path to the `nvpair-proxy` binary. One process fronts every engine: the broker spawns it once and then sends a `facade/enable` per entry in `--proxy-engines`. Same optional semantics as `--node-info-path`: an absent default sibling means no local proxies (non-fatal); an invalid explicit path exits with an error |
| `--proxy-engines <csv>` | `ollama,lmstudio,llamacpp` | Which engines to front with a proxy. An unrecognized name exits with an error rather than being skipped, so a typo cannot look like it worked. An engine left out is not started **and not prepared** — the broker will not relocate an engine whose facade nothing is going to claim |
| `--workload-manager-path <path>` | `./nvpair-workload-manager[.exe]` in the CWD | Explicit path to the `nvpair-workload-manager` binary the broker spawns for the cluster workload relay. Same optional semantics as `--node-info-path`: an absent default sibling means no workload relay (non-fatal); an invalid explicit path exits with an error |
| `--errors-path <path>` | `./nvpair-errors[.exe]` in the CWD | Explicit path to the `nvpair-errors` binary the broker spawns (with `--peer-sync`) for the service-error pipeline. Same optional semantics as `--node-info-path`: an absent default sibling means the error pipeline is disabled — producers' errors are dropped (non-fatal); an invalid explicit path exits with an error |
| `--engine-manager-path <path>` | `./nvpair-engine-manager[.exe]` in the CWD | Explicit path to the `nvpair-engine-manager` binary the broker spawns for engine management. Same optional semantics as `--node-info-path` |
| `--engine-manager-http-port <port>` | `14322` | Model-list HTTP listener (`em`) passed to engine-manager and registered with discovery |
| `--engine-manager-control-port <port>` | `14323` | Remote-control listener (`ec`) passed to engine-manager and registered with discovery when a cluster dir is configured |
| `--node-info-port <port>` | `14318` | Node-info HTTP listener (`ni`) passed to `nvpair-node-info` as `--port` and registered with discovery |
| `--errors-port <port>` | `14319` | Errors peer-sync listener (`er`) passed to `nvpair-errors` as `--port` and registered with discovery |
| `--workload-manager-port <port>` | `14320` | Workload relay listener (`wl`) passed to `nvpair-workload-manager` as `--port` and registered with discovery |
| `--cluster-manager-port <port>` | `14321` | Cluster pairing listener (`cl`) passed to `nvpair-cluster-manager` as `--port` and registered with discovery |
| `--manual-nodes-path <path>` | `./nvpair-manual-nodes[.exe]` in the CWD | Explicit path to the `nvpair-manual-nodes` binary the broker spawns for user-added nodes. Same optional semantics as `--node-info-path` |
| `--settings-path <path>` | `./nvpair-node-settings[.exe]` in the CWD | Explicit path to the `nvpair-node-settings` binary the broker spawns for the typed settings store. Same optional semantics as `--node-info-path` |
| `--cluster-manager-path <path>` | `./nvpair-cluster-manager[.exe]` in the CWD | Explicit path to the `nvpair-cluster-manager` binary the broker spawns for cluster pairing / membership. Same optional semantics as `--node-info-path` |
| `--scheduler-path <path>` | `./nvpair-job-scheduler[.exe]` in the CWD | Explicit path to the `nvpair-job-scheduler` binary the broker spawns for responsive, node-wide workload and GPU-pressure ranking. Same optional semantics as `--node-info-path` |
| `--cluster-dir <path>` | `cluster/` in the per-user data dir (`%LocalAppData%\Nvidia Corporation\Personal AI Router` on Windows, `~/.config/Nvidia Corporation/Personal AI Router` on Linux) | Cluster config dir (`node.crt`/`node.key` + `trusted/`, minted by `nvpair-cluster-manager`). Threaded to the cluster-scoped workers (`nvpair-errors`, `nvpair-workload-manager`, `nvpair-node-scanner`, `nvpair-manual-nodes`, and `nvpair-engine-manager`), each of which derives its membership from it continuously — so a create, join, or leave takes effect in place and the broker does **not** restart them. The broker also passes the parent of this path to `nvpair-cluster-manager` as `--config-dir`, so the only writer of the cluster dir and the workers reading it cannot resolve different directories. `nvpair-node-info` is excluded — it stays plain HTTP even when clustered (see the repository-root `SECURITY.md`). Defaults so cluster mTLS auto-activates with nothing to pass; with an empty or cert-less dir this node is not a member, so it serves and dials no inter-node cluster traffic at all |
| `--log-level <lvl>` | _(env `NVPAIR_LOG_LEVEL` or `info`)_ | `debug` \| `info` \| `warn` \| `error` |
| `--version` | | Print version and exit |

The `--*-port` flags above form one immutable startup plan. Before any worker starts, the broker rejects a plan in which a port is outside 1 to 65535, two services share a port, or a service takes an engine's compatibility facade port or managed backend base port from the engine table (reserved even for an engine left out of `--proxy-engines`). Each worker receives, registers and replays exactly its planned port, including across supervised restarts and scanner respawns; a port at its default is left to the worker's identical flag default, so the default plan spawns workers with their historical argv. The planned ports stay reserved against engine settings and the inherited `OLLAMA_HOST` alias, and a default port that the plan moved elsewhere is released. These flags separate listeners only: a second instance on one host still needs its own app data, cluster identity, IPC endpoint and engine facades.

Logs go to **stderr** (shared `applog` format, same as every other NVPAIR binary). stdout is reserved for JSON-RPC frames in stdio mode, so logging there would corrupt the protocol. Every spawned worker's stderr is forwarded to the broker's stderr unmodified, so `[nvpair-node-scanner]`, `[nvpair-node-info]`, `[nvpair-proxy]`, `[nvpair-workload-manager]`, and `[nvpair-cluster-manager]` lines interleave with the broker's `[nvpair-ui-broker]` lines on a single stream. The proxy logs under the process name because that is what it is; a line about one engine's facade carries an `engine` field instead of a distinct prefix.

## Worker subprocesses

On startup — **before** emitting `app:ready` — the broker spawns the scanner and (when available) node-info, `nvpair-proxy`, the workload-manager, and the cluster-manager as child processes over stdio. The proxy is spawned up front but doesn't gate `app:ready` — each of its facades announces its listen port asynchronously (see below). None of the auxiliary workers gate `app:ready`.

**`nvpair-node-scanner`** (the consolidated discovery daemon) is spawned first. It pushes `discovery:node-discovered`, `discovery:node-updated`, and `discovery:node-removed` notifications into the broker, which maintains them in an in-memory map keyed by `id`. Clients query that map via `discovery:get-nodes` and — once they've opted in via `discovery:subscribe` — receive a `discovery:nodes-changed` notification on every store mutation. The raw `discovery:node-*` notifications are never forwarded as-is. The scanner polls healthy node-info endpoints on a staggered two-second cadence, backs consecutive remote failures off to a 30-second cap, and emits compact `discovery:node-telemetry` observations containing maximum GPU utilization, validity, and age; these remain internal to broker scheduling. The broker registers this node's local service ports (`ni`/`er`/`wl`/`cl`/`em`, plus `ol`/`lm`/`lc` from the engine poller) with the daemon over the same link, so the daemon can advertise them all in one `_nvpair-node` record.

**`nvpair-node-info`** is spawned next. It's a server, not an event source: it stands up the local `/v1/node-info` HTTP endpoint (GPU/CPU/memory inventory). It does not advertise itself — the broker registers its `ni` port with the scanner daemon, which carries it in the node record, and a peer's daemon fetches `/v1/node-info` over plain HTTP to enrich the node. The broker doesn't read anything back from node-info's stdout (drained and discarded). Spawning it is **optional**: if the binary can't be resolved (and no `--node-info-path` override was given) the broker logs a warning and continues serving discovery without it.

**Engine advertising.** The broker runs an internal 5 s poll loop against local
Ollama (`GET /`), LM Studio (`GET /v1/models`), and llama.cpp (`GET /health`)
at their configured backend ports and reconciles this node's engine registration
with the scanner daemon:

- engine **up** and facade **ready** → register `ol` / `lm` / `lc` at the
  promoted facade port, never the private backend port;
- engine **down** → unregister it.

The daemon folds those registrations into this host's single `_nvpair-node` record, so a peer discovers the engine through the shared channel. The model list is not part of that registration — it's served over HTTP by `nvpair-engine-manager` (the `em` service, `GET /v1/models`) and enriched onto each node by the peer's daemon. There is no separate advertiser subprocess and no manual-advertise RPC.

**Descriptor engines.** After engine-manager starts and before facades are prepared, the broker reads `engine:descriptors` once and fronts every descriptor engine (any engine ID that is not built in, such as a `llamacpp-custom` fork manifest or a Strata user manifest) that is available on this host and that `engines.ProxyCanFront` accepts, at most `engines.MaxDynamicEngines` of them in ID order (`enginecatalog.go`). Each gets a facade in the one proxy process, enabled with its descriptor so the proxy can build its routes; a facade port planned from `engines.DynamicFacadePortBase` clear of every built-in engine port and every other descriptor engine's persisted or planned port (the proxy then persists where it landed); and the default advertisement loop, which reads its backend port from engine-manager's `engine:status` and probes its OpenAI or Ollama model list. While it is up the broker registers its `e-<id>` service: cluster-manager publishes it on the authenticated peer-services path and admits it on the reverse transport, and the scanner adds it to this node's own directory entry but keeps it out of the mDNS TXT record. When it stops, the registration is withdrawn from both. `--proxy-engines` names built-in engines only. The set is fixed for the broker's lifetime; a new manifest takes effect at the next start. Unknown, malformed, unavailable or unfrontable descriptors are not fronted, and descriptor engines have no engine settings or `<id>-proxy:` client relay yet.

**`nvpair-proxy`** is one process that fronts every enabled engine. It starts with no engine and no listener; the broker then sends it a `facade/enable` per engine, carrying that engine's port and any alias addresses. A flag could not express this, because the broker plans a different port for each engine. Each facade forwards inference to a node it discovers on the network and speaks its own engine's dialect.

One process for all of them is deliberate. Between scheduler snapshots a facade
takes short-lived reservations for work it has dispatched, and those live in
the process; separate processes would each hold only part of that picture, so
simultaneous bursts across engines could select the same node believing it idle.
The cost is **shared fate**: a crash takes every facade down and the supervisor
brings them all back together, reported once as
`supervisor:subprocess-crashed:nvpair-proxy` rather than against one engine.
Within the process the boundaries are finer — a facade that loses its bind
race, cannot be moved to a free port, or panics while handling a request is
withdrawn or answered with an error on its own, leaving the others serving.

Ollama's standalone default is `:11435`; with managed port ownership enabled (the default), the broker starts settings and engine-manager first, claims `:11434` with that facade, and only then moves a stopped default-port Ollama backend to a free port. Custom backend ports are preserved. When the inherited `OLLAMA_HOST` names a distinct local plaintext port, the broker also gives the facade that normalized loopback-only alias so clients already using the variable enter the same routing path; `localhost` reserves both canonical loopback families atomically, while remote and HTTPS targets are ignored. The alias port is reserved against every configured engine, local or remote engine start override, every facade's control plane, and the managed Ollama and LM Studio backend port plans, so a backend that has to move can never land on the alias. A running Ollama or unknown owner on either requested port is never stopped or moved: the primary uses a safe fallback when needed, and an occupied alias remains with its owner while the broker reports a warning.

llama.cpp is a managed engine: engine-manager owns its lifecycle and starts it
on its configured loopback port, `:8081` by default. The broker places its
default-enabled facade on `:8080` or a safe fallback, without automatically
relocating the engine or adding a compatibility-port ownership gate. Fallback
selection excludes the default engine port and other engines' reserved ports.
Explicit engine and proxy settings are preserved; a bind failure on an explicitly
chosen proxy port is reported instead of selecting a fallback.

For automatic model-bearing inference, every facade combines scheduler pending counts and GPU pressure with the process-wide reservation map under one lock before forwarding, so concurrent requests distribute without an artificial delay or a round trip through the scheduler. A reservation is released when its request ends and moves with a failover, so a node stops counting as loaded as soon as it stops working. Manual pins, model-owner tiers, and the complete failover list keep their existing precedence. The broker otherwise treats the proxy as **optional and non-fatal**.

Each facade announces its bound port **asynchronously**, via an engine-addressed `ready` notification emitted once its HTTP listener is up. The broker records readiness per engine and exposes it through that engine's `<engine>-proxy:get-status` request — per engine, because the facades bind different ports and a single port for the process would be whichever readied last. Because `ready` arrives after `app:ready` (and the proxy is optional), clients learn a port by **polling** `<engine>-proxy:get-status` rather than assuming it from `app:ready`.

The proxy is a full bidirectional JSON-RPC peer with a control plane (node
selection, manual nodes, ...) and an event stream. The broker acts as a
**generic relay** in both directions: any request a client sends under an
enabled `<engine>-proxy:` namespace (other than reserved broker-local ones) is
forwarded to that engine's facade and the response relayed straight back, and
every facade notification is re-emitted to subscribed clients under the same
component namespace.

Client namespaces are unchanged by the process collapse, but the wire inside is not. Because one process holds every facade, a message on that link carries the engine it concerns: the client's `ollama-proxy:` prefix comes off and a bare `ollama:` facade address goes on. The two are not interchangeable — `ollama-proxy:` is how a client addresses the component, `ollama:` is how a message addresses a facade inside the process — so the relay is a translation between them rather than a strip. Process-scoped methods (`log/set-level`, the scheduler's `node/set-priority`) carry no address, as do the proxy's own `errors:report` / `errors:clear` for the routing listener (`nvpair-proxy:routing-listener-blocked`), which the broker routes into the error pipeline once per process and, like the OLLAMA_HOST alias warning, caches until cleared and replays to a restarted `nvpair-errors`. `facade/enable` names its engine in the payload because it runs before that facade exists.

Two classes of proxy notification are **not** re-emitted under any
`<engine>-proxy:` namespace, because neither is a proxy control-plane event:

- The `workload:*` lifecycle events the proxy fires per inference request. Those are workload-manager traffic — see the workload-manager paragraph below for how they're routed.
- `node/activity`, which a proxy raises while a peer's engine is streaming response bytes back through it. That is discovery input: the broker forwards it to `nvpair-node-scanner` as a `discovery:node-activity` notification, where it counts as proof the peer is alive and cancels the eviction it would otherwise face for failing a liveness probe it had no spare CPU to answer. No client has any use for a per-request liveness frame. The handoff is a bounded queue drained by one goroutine — reports arrive for as long as inference streams, so a wedged scanner must not be able to stall the proxy reader, and a dropped report only means the scanner falls back to probing a node that will very likely answer.

**`nvpair-workload-manager`** is the cluster workload relay, and it's the only worker the broker talks to **bidirectionally over a notification-only link** (no id-bearing request/response). It supervises it the same optional, non-fatal way as node-info / proxy: a missing default sibling (and no `--workload-manager-path`) just means no cluster workload relay. The broker plays the workload **broker** role between the proxy and the manager:

- **Outbound (proxy -> broker -> manager -> peers).** When a facade emits a `workload:started` / `workload:completed` / `workload:errored`, the broker stamps the local stable `hostUuid` onto `params.workloadInfo.originatedFrom`, applies the transition to its authoritative workload store, and fans the accepted update to the scheduler before forwarding the original lifecycle frame to the manager. The manager broadcasts it to peer nodes. With no manager supervised the event still updates local scheduling and subscribed clients, but is not broadcast.
- **Inbound (peers -> manager -> broker).** The manager translates peer-origin lifecycle events into `workloads:upsert` and peer-origin removals into `workloads:remove` on stdout. The broker applies each accepted transition to the same store, fans it to the scheduler, and relays it to clients subscribed via `workloads:subscribe`.
- **Local echo.** Local-origin proxy workloads are also emitted to the same `workloads:*` client stream (lifecycle translated to `workloads:upsert`), so a subscribed client sees a coherent cluster-wide view — its own workloads alongside peers'.

**`nvpair-job-scheduler`** consumes the accepted workload stream, compact GPU
telemetry, and discovery snapshot. It smooths fresh utilization into pressure
0–3, uses neutral pressure 1 for invalid/missing/older-than-10-second samples,
and orders by `pending + gpuPressure`, then pressure, then stable UUID. Load is
node-wide across Ollama, LM Studio, and llama.cpp because they normally contend
for the same resources. Each engine-specific `schedule:priority` carries
`{engine,nodes,ranks}` and refreshes when order, pending counts, or pressure
changes. The broker caches, generation-orders, and replays the full
`{nodes,ranks}` snapshot to the matching facade, where a newly delivered
snapshot resets optimistic reservation deltas. On scheduler spawn/restart the
broker replays active workloads and telemetry before discovery, then resumes all
three live feeds.

`schedule:priority` and `node/set-priority` are internal worker contracts: the broker does not expose either notification to its connected client.

**`nvpair-cluster-manager`** owns the node's stable cryptographic identity (a self-generated UUID + Ed25519 self-signed certificate), the trusted-node store, and the EAP-NOOB PIN-based pairing flow that bootstraps mTLS trust between machines. Like ollama-proxy it's a full bidirectional JSON-RPC peer, and like the rest it's supervised **optionally and non-fatally**: a missing default sibling (and no `--cluster-manager-path`) just means the broker runs without cluster pairing. The broker is a **generic relay** for it in both directions: any `cluster:*` or `nodes:*` request a client sends is forwarded verbatim and the response (result or JSON-RPC error) is relayed straight back, and every notification the manager pushes is forwarded to the client unchanged. Unlike the proxy stream these events are not opt-in — they're low-volume, important pairing/membership pushes (`cluster:invite-received`, `cluster:identity-changed`, `nodes:changed`, ...), so they always flow. `errors:report` / `errors:clear` are the one exception: like the proxy's, they target `nvpair-errors` (which the broker doesn't supervise) and are dropped. Because pairing calls drive multi-round-trip inter-node network exchanges, the relay's per-call timeout is more generous (30 s) than the proxy's local control-plane timeout.

Shared lifecycle for all workers:

- Path resolution: the explicit `--scanner-path` / `--node-info-path` / `--proxy-path` / `--workload-manager-path` / `--cluster-manager-path` (and the analogous flags for errors / engine-manager / manual-nodes / settings; every engine's facade lives in the single process at `--proxy-path`, with `--proxy-engines` choosing which are enabled) if set, otherwise the same-named `./<binary>` (with `.exe` on Windows) in the broker's current working directory. No PATH fallback — a clear "not found" beats a surprising stale binary. A missing scanner is fatal (it's the broker's core job); every other worker is optional.
- Log level: the broker's currently resolved level is forwarded at spawn time via `--log-level <lvl>`. Runtime `log/set-level` requests are fanned out to every running child over its stdin (see `log/set-level` below).
- Hidden console window on Windows (`CREATE_NO_WINDOW`).
- Environment: each worker inherits the broker's environment without the reserved `NVPAIR_PROVIDER_SECRET_` provider-secret namespace. The proxy alone keeps the credential references that the node policy's routing accounts configure, and no other name in that namespace. Names are compared case-insensitively on Windows. The reference set is captured when the proxy is launched, and a `policy:set` that changes it replaces the running proxy once, through the same stop and start that a listener change uses, so the replacement does not count as a crash. Ambiguous duplicate entries for a configured reference fail the proxy launch with an error that names the reference. Neither values nor environment snapshots are logged.
- Shutdown: when the broker exits (signal, peer EOF, or `shutdown` RPC), it first stops the proxy so no new inference arrives, then asks engine-manager to stop its engines (`engine:prepare-shutdown`) while the rest of the tree is still up, then closes each running worker's stdin. Clients leave this ordering to the broker; one that calls `engine:prepare-shutdown` itself stops the engines while the proxy is still routing to them. The worker sees EOF and exits cleanly, and the broker waits for it to exit — no timeout and no force-kill. Each worker owns its own bounded shutdown (engine-manager bounds its engine stop internally; the HTTP workers cancel their context and drain their server on EOF), so the broker never SIGKILLs a worker mid-teardown. A grace-then-kill teardown would orphan engine processes by killing engine-manager partway through stopping them, which is why there is no timeout here.
- **Auto-restart with crash surfacing** for every supervised worker (see [Supervision & recovery](#supervision--recovery)): a crash is reported as `supervisor:subprocess-crashed:<name>` and the worker is restarted with backoff, clearing the entry once it's healthy again and leaving it up if the restart budget is exhausted.

## Node policy

The fork's node policy (see `FORK_DESIGN.md` §3 at the repository root) is owned
by the broker. Its schema, defaults and validation live in
`nvpair-shared/nodepolicy`; the broker is the only writer of the file and the
proxy enforces it.

- **File.** `<appdir>/node-policy.json`, loaded at startup before engine-manager
  and the proxy start. A missing file means `nodepolicy.Default()` (revision
  `"0"`). A version 1 file is migrated in memory and left untouched on disk; the
  first save of any kind whose replacement lands, including a pause or resume,
  rewrites it as version 2, even when the save returns a durability warning.
  A document without a `version` key parses as version 2, not version 1.
- **Revision.** The broker is the only revision assigner. Every persisted
  document, whether from `policy:set`, a relayed peer set or a persisted
  availability change, is stamped with the next revision before it is written:
  the larger of the previous revision plus one and the current Unix time in
  nanoseconds, as a canonical decimal string. The clock term means a revision
  issued before a corrupt file was recovered is never issued again afterwards (a
  clock that moves backwards across a recovery can defeat this). A `revision` a
  client or peer sends inside the policy document is ignored, as `availability`
  is; the base revision a set is conditional on travels separately, as
  `expectedPersistedRevision`. A revision at the
  signed 64-bit maximum cannot advance and the save fails. A restart reloads the
  revision that was stored. After a persisted availability change the broker
  sends `policy:changed` with the new revision, so editors refresh their base.
- **Activation.** A saved revision is not an enforced one. The persisted
  revision is `policy.revision`; `activeRevision` is the newest revision the
  running proxy process acknowledged, empty when it has acknowledged none or no
  proxy runs. `activation` describes the revision of the policy in the same
  result or event: `active` when that revision was acknowledged or is
  superseded by a later acknowledged one (`activeRevision` has reached it, so a
  reply to an earlier set can show a newer `activeRevision` than its own
  `policy.revision`), `error` when the newest push failed (with
  `activationError`, operational text that never holds policy content or
  credential references) and `pending` otherwise, including when no proxy runs,
  before a started proxy has been replayed to, and while a push is still
  waiting to be sent. A proxy the broker stops on purpose stops counting first,
  so a push its closing pipe fails is not an `error`. Only a `node/set-policy`
  acknowledgement (`{revision, runId}`, `nodepolicy.PolicyAck`) names a
  revision, and the proxy ignores the availability inside it, so a revision is
  `active` only while the newest `node/set-availability` push to that proxy
  process was acknowledged as well; there is one activation path. The broker
  accepts a policy acknowledgement only from the proxy process it last
  replayed to, with the run id that process first acknowledged under, and only
  for the exact revision it pushed; anything else leaves `activeRevision` as it
  was. While the newest
  availability push was refused, failed or not answered in time, every revision
  is `error` with an `activationError` naming the availability push and
  `activeRevision` is empty, because the proxy is not known to enforce what the
  policy says; the next availability acknowledgement restores the reporting.
  The push slot is a one-slot lock that serializes what is sent to the proxy.
  `policy:set` saves, then takes the slot and pushes the saved document, and the
  whole wait, slot and acknowledgement together, is bounded by the proxy
  call bound (5 s). If the slot is not free in that time (a replay is still
  talking to a slow proxy), or is got with less than a tenth of the bound left,
  nothing is sent: the result is `pending` with both revisions and a push is
  scheduled in the background that takes the slot when it frees and sends the
  persisted document. That later push can activate the revision after the reply
  was sent, and the broker then announces `policy:changed` for it (this applies
  to any scheduled push whose answer changes `activeRevision` or the recorded
  failure). A request that was sent and is refused, answered wrongly, or not
  answered in time leaves the revision saved and `error`, and nothing is
  rescheduled. A refused or invalid request leaves the proxy on the policy it
  had; after an acknowledgement timeout the proxy may still apply and enforce
  the revision, while the broker keeps reporting `error` until the next push or
  a proxy replay reconciles it, because a late acknowledgement is not
  recovered. The proxy peer is built with a write interrupt, so a proxy that
  stops reading fails the pending call at its bound and is restarted by
  supervision rather than wedging every later push. A persisted pause or resume
  pushes `node/set-availability` first, so admission stops before anything
  else, then `node/set-policy` with the current document, whose revision is the
  availability change's: the same path, so the revision is reported `active`
  only once the proxy acknowledged both pushes, and the push also re-delivers a
  policy whose earlier push was refused. A refused or unanswered availability
  push leaves that revision saved and `error`, and the `node:set-availability`
  result says so (below); the state was still reached on this node. Nothing else
  retries a failed activation, except that every policy push first sends
  availability again when the newest availability push was not acknowledged: the
  next policy or availability change, or a proxy restart, pushes the persisted
  document again. A repeated `node:set-availability` for the state the node is
  already in does nothing, except after the newest availability push was
  refused, failed or not answered in time: then it takes a new revision and
  pushes again. During recovery, where nothing is saved, availability is pushed
  alone and activation is reported against the revision actually pushed (`"0"`),
  which a refused availability push leaves in `error`. The
  public-egress lease that consumes the active revision belongs to hybrid
  dispatch and is not implemented here.
- **Durability.** Every save (`policy:set`, a relayed peer set, a persisted pause
  or resume) goes through `nvpair-shared/durablefile`: a temporary sibling is
  written and synced, replaces the file, and the parent directory is synced
  (Windows uses write-through replacement instead). The in-memory policy,
  proxy supervision reconciliation where needed, proxy push and `policy:changed`
  follow every landed replacement, including persisted pause and resume.
  A failure to create, write, sync or close the temporary file, or to replace the
  file, leaves the old file byte for byte. A directory-sync failure comes after
  the replacement: the complete stamped document is adopted, recovery is cleared,
  and application proceeds exactly as after a successful save. The caller gets
  error `-32000` saying the document with revision N was saved and is being
  applied but may not survive a power loss; the underlying failure is preserved.
  Memory, the file and a fresh broker agree on that document. Before replacement,
  no state is adopted, pushed or announced and the error says the policy could
  not be saved. Every attempted revision is spent and is never assigned again.
  Persisted pause and resume complete their normal availability transition,
  including drain and engine actions, before returning that durability warning.
  During recovery, availability changes remain in memory only and preserve the
  corrupt file and recovery flag until an explicit local `policy:set` lands.
  The tests inject a failure at each step and restart a broker on the resulting
  files, which shows ordering under a crash of the process. They are not proof of
  power-loss durability: that rests on what `fsync`, `rename` and Windows
  `MoveFileEx` write-through guarantee on the underlying filesystem and storage.
- **Recovery.** A file that cannot be read, parsed or validated is kept as
  `node-policy.json.bad` (when it could be read) and the node starts on a
  recovery policy: the defaults, so no tiers or profiles and remote and paid
  execution off, with availability `paused` whatever the file said. `policy:get`
  reports `recoveryRequired: true` until an explicit local `policy:set` lands a
  valid replacement under a fresh revision, which clears recovery even when
  the save returns a durability warning. Until then an availability change
  (local or relayed) takes effect in memory
  and reaches the proxy as usual but is not persisted: it does not overwrite the
  corrupt file, take a revision or announce `policy:changed`, except to report an
  availability push the proxy refused (Activation). A relayed
  `policy:set` from a paired node is refused during recovery.
- **Proxy state.** The broker pushes `node/set-policy`, `node/set-availability`,
  `node/set-residency`, `node/set-engine-intent`, `node/set-engine-drain` and
  `node/set-model-identity` (Engine identity below; methods and payloads in
  `nodepolicy/wire.go`). Residency comes from
  engine-manager's `engine:models-changed` (`loadedByEngine`) and
  `engine:state-changed` (a stopped engine is absent); saved intent from
  `engine:intent` and `engine:intent-changed`. Both are seeded from
  `engine:intent`, `engine:get-installed` and `engine:models` whenever
  engine-manager (re)starts. Each push sends the state current at send time and
  pushes are serialized, so an older state can never overwrite a newer one.
  When the proxy (re)starts, all of them are re-sent from its spawn hook **before
  any facade is enabled**, availability first, so a paused node is never briefly
  admitting work, and `policy:changed` then reports the activation the replay
  reached.
- **Node projection.** After every persisted policy change, pause and resume,
  and on every `engine:ready` (and when a new engine-manager is attached), the
  broker sends engine-manager `engine:node-projection {policy, pausedLocally}`:
  the stored policy's advertisement (`revision`, declared `roles` with the
  bounds of the matching profiles, `pausedBy`), and whether the node is paused
  locally (Local veto). engine-manager serves `policy` to paired peers in its
  pinned `GET /v1/models` document and enforces `pausedLocally`. Each send reads
  the policy current at send time under one lock, so an older projection never
  follows a newer one.
- **Strict admission readiness.** The proxy reports its strict admission
  controller's readiness with the notification `admission/strict-readiness`,
  whose params are `nodepolicy.StrictAdmissionReadiness`
  (`{policyRevision, proxyRunId, capability?}`), after acknowledging each
  `node/set-policy` and whenever that readiness changes (`readiness_contract.go`).
  The broker discards a report from another proxy run than the one that
  acknowledged the policy, and an invalid report withdraws what it held. It
  holds the newest report and adds its capability to the projection's
  `policy.strictAdmission` only while the report is authoritative: the proxy
  that sent it still runs and is the incarnation the policy was replayed to,
  `proxyRunId` is the run of that incarnation's acknowledgements, and
  `policyRevision` is both the active and the persisted revision, with the
  availability push acknowledged and the node `available`. Models of an engine
  saved Off are left out, and so are models whose identity is not the engine
  identity below, and a capability with no model left is withdrawn.
  Anything else withdraws it: proxy exit or stop, a replay to a new proxy run
  (which clears the held report until the new run reports), a saved policy
  pending acknowledgement, and a pause or drain. A report may arrive before
  the acknowledgement it follows; it is advertised once that is recorded.
  Readiness is reconsidered, under the projection lock, after every report,
  acknowledgement, proxy start or exit and intent change, and a projection is
  sent only when the capability changes. `Policy.Advertisement()` never sets
  it. Peers without the capability keep concrete routing.
- **Engine identity** (`model_identity.go`, `docs/local-qualification.mdx`).
  engine-manager's `engine:model-identity` notification (consumed, never
  forwarded to clients) is a whole snapshot of each engine it launched and
  owns: run id, launch digest and model content digests. The broker joins it
  with the SHA-256 fingerprint of each engine's applied settings revision
  (journal epoch, applied revision and settings; none while a revision is
  applying, failed or unapplied) and pushes the result to the proxy with
  `node/set-model-identity`, one more coalesced push kind replayed to a
  restarted proxy. An engine lacking either part is left out. `running:false`
  removes an engine's run identity at once, and engine-manager exit removes
  all of it. Before every settings operation calls `engine:configure-launch`,
  the broker removes that engine's fingerprint and waits for the proxy to
  acknowledge the identity without it; with no proxy running nothing needs
  withdrawing, and a proxy without the method holds no capability. Any other
  failure fails the operation before the engine is touched.
- **Admission.** The proxy's `admission/state` (active, per-engine active, last
  activity) feeds the drains below; `admission/unload` unloads the named models
  through `engine:unload-model`; `admission/wake` calls `engine:wake` unless the
  node is not `available` or the engine is saved Off. None of them reach clients.
- **Pausing** (`node:set-availability {state:"paused"}`): persist `paused`, push
  `draining` (with `cancelActive` when `pause.onActive` is `cancel`), wait for the
  proxy's active count to reach zero for up to `pause.drainTimeoutSeconds`, then
  push again with `cancelActive` and allow 10 s for that work to unwind. Then, as
  the policy says, unload every loaded model and `engine:sleep` every running
  engine, push `paused`, and notify clients. The request returns once paused.
- **Resuming** persists and pushes `available`, then `engine:wake`s the engines
  the pause stopped. Their saved intent was never changed, so only engines saved
  On come back.
- **Ordering.** Transitions run one at a time. A newer `node:set-availability`
  cancels the one in progress, which returns the error `superseded by a later
  node:set-availability request`; the latest request always wins, and a resume
  never waits out a pause's drain. A paired node's resume refused by the local
  veto is not accepted, so it supersedes nothing.
- **Local veto.** A pause records who made it in `policy.pausedBy`: `local` for
  a local client's `node:set-availability`, `peer` for one relayed from a
  paired node; a resume clears it, and `policy:set` (local or relayed) never
  changes it. A node paused locally, including a pause saved before `pausedBy`
  existed and the recovery policy's pause, refuses a paired node's resume with
  error `-32011` (`this node was paused on the node itself; resume it there`),
  also while a local pause is accepted but not yet saved. A paired node's pause
  over a local one stays `local`; a local pause over a peer one becomes
  `local`. Only a local resume lifts it. A paired node may resume a peer pause.
  Configuration edits through `policy:set` stay allowed while vetoed.
  engine-manager applies the same veto to paired nodes' engine starts and model
  loads from the projected `pausedLocally`. While the node is paused, whoever
  paused it, `engine:apply-settings` (and journal recovery) applies the
  settings but never resumes a stopped engine; the local user resumes the node
  first. A paired node's apply that would resume one while the node is paused
  locally is refused with `-32011` before it is accepted, and its
  `engine:settings-reply` carries `errorCode` like `policy:reply`.
- **Startup.** A persisted `paused` reaches the proxy before it can admit
  anything. With `pause.stopEngines` set, `engine:restore-enabled` is skipped
  (also when engine-manager restarts while paused) and sent on resume instead.
- **Idle policy.** Every 30 s the broker looks at each running engine with no
  active requests. Its idle time runs from the latest of its last admitted
  request (`admission/state.lastActivityMs`), the moment it was seen to start,
  and the moment it last gained a loaded model. Past `idle.unloadAfterMinutes`
  its models are unloaded; past `idle.stopEngineAfterMinutes` an engine saved On
  is put to sleep. Each step runs once per idle period, so an engine that refuses
  to stop is not retried every tick. Nothing runs while the node is draining.
- **Start on demand.** `engine:wake` (from `admission/wake` or a client) starts
  only an engine saved On, never while the node is paused or draining; a client's
  `engine:wake` is refused with an error then.
- **Settings restarts.** When `engine:apply-settings` would restart a running
  engine (`preview.restart`), the broker pushes `node/set-engine-drain
  {engine, drain:true}`, waits up to 60 s for that engine's active count to reach
  zero (releasing the settings journal lock meanwhile), applies, and clears the
  drain. See [ENGINE_SETTINGS.md](ENGINE_SETTINGS.md).
- **Other nodes.** A `nodeId` naming another node is relayed through
  engine-manager's pinned control server (`engine:remote-policy-get`,
  `engine:remote-policy-set`, `engine:remote-availability-set`). On the target,
  engine-manager delivers it as `policy:request {id, method, caller, params}`;
  the broker checks the caller is still pinned, acts on itself only (a relayed
  request is never forwarded again), and answers
  `policy:reply {id, result, error?, errorCode?}`. Every failure carries its
  JSON-RPC code: invalid parameters (including a missing or malformed
  `expectedPersistedRevision`, invalid policy or invalid availability) use
  `-32602`; revision conflicts use `-32010`; a resume refused by the local veto
  uses `-32011`; save failures and other operational failures use `-32000`.
  Engine-manager maps `-32602` to HTTP 400, `-32010` and `-32011` to HTTP 409
  with the body `{"error", "code": "policy_conflict" | "local_veto"}`, and other
  failures to a plain-text HTTP 409; the origin maps HTTP 400 back to `-32602`,
  a typed 409 back to its code, and other HTTP failures to `-32000`.
  Classification uses the code, never error text.
  A successful
  remote change is also announced to this node's clients under the target's
  `nodeId`.
- **Peer projection.** A paired node sees and changes a projection of the policy,
  never the owner's document. A relayed `policy:get` and the result of a relayed
  `policy:set` carry `nodepolicy.PeerPolicy`: the device fields, the local tier
  entries (a tier left with none is omitted), `tierPolicy` and the `revision` of
  the protected source document. The whole `routing` section (accounts,
  credential references, deployments, limits, listener) and every API tier entry
  are left out. A relayed `policy:set` must carry the projection's `tiers` and
  `tierPolicy` unchanged and no `routing` key; it is then merged into the stored
  policy, whose tiers (API entries included), `tierPolicy`, `routing`,
  `revision` and availability are kept exactly, and only the device fields
  change. The relayed set also carries `expectedPersistedRevision`, compared
  before the merge, so a peer holding an older projection is refused as a
  conflict rather than merged. Any change to those protected sections, or any `routing` content, is
  rejected with nothing persisted and nothing pushed to the proxy. The merge runs
  against the stored policy inside the same critical section as the save, and
  the revision check precedes it, so a peer holding an older projection cannot
  restore protected content a local `policy:set` has since changed: the stale
  set is a conflict and changes nothing. Local clients keep the full
  document, including in `policy:changed`.

## JSON-RPC Surface

### Notifications (broker → caller)

#### `app:ready`

Emitted once on startup, after the scanner (and, when available, node-info and ollama-proxy) have been spawned and the transport is connected. Seeing `app:ready` means the broker is accepting requests **and** discovery events are flowing into its store (even if no nodes are known yet — mDNS browsing takes a moment). It does not imply node-info came up — that worker is optional and a failure to spawn it is logged but does not block `app:ready`. Nor does it imply the proxy is serving yet: the proxy reports its bound port asynchronously, so `ollama-proxy:get-status` may briefly show `ready:false` right after `app:ready`.

```json
{"jsonrpc":"2.0","method":"app:ready","params":{"version":"0.5.0"}}
```

#### `discovery:nodes-changed`

**Opt-in.** Only delivered to a peer that has called `discovery:subscribe`; until then (and after `discovery:unsubscribe`) the stream is silent. Once subscribed, it's fired every time the discovery store mutates — that is, whenever the daemon reports a `discovery:node-discovered`, `discovery:node-updated`, or `discovery:node-removed` (or a manual node changes). The payload is the full current snapshot as a **bare array** of `AvailableNode` objects (same shape as the array element in `discovery:get-nodes`), sorted by `id`.

```json
{
  "jsonrpc":"2.0",
  "method":"discovery:nodes-changed",
  "params":[
    {
      "id":"MY-PC",
      "name":"MY-PC",
      "ipAddress":"192.168.1.10",
      "port":14318,
      "lastSeen":1748275200
    }
  ]
}
```

A few semantics worth knowing:

- **Always the full snapshot, never a diff.** Replace your local list on every event.
- **A baseline is emitted on subscribe.** The first `discovery:nodes-changed` after a fresh `discovery:subscribe` carries the current snapshot (possibly an empty array), sent right after the subscribe response. Every later event corresponds to a real store mutation. (`discovery:get-nodes` remains available for an on-demand snapshot at any time.)
- **Fires on `lastSeen`-only changes too.** A periodic re-discovery that produces an otherwise byte-identical record still bumps `lastSeen`, which is a genuine freshness signal worth surfacing.
- **Empty array is a valid payload.** If every previously-known node ages out, you'll get a `discovery:nodes-changed` with `"params": []`.
- **Best-effort delivery.** The broker tries the notification once per mutation; transient write errors are logged but not retried.

#### `ollama-proxy:<event>`

**Opt-in.** Only delivered to a peer that has called `ollama-proxy:subscribe`; silent otherwise. Once subscribed, every notification `ollama-proxy` emits is re-emitted to the client as `ollama-proxy:<method>` — the proxy's native method name with the `ollama-proxy:` prefix, and `params` passed through verbatim (the proxy's own snake_case schema). The proxy's current notifications are:

| Forwarded as | Proxy emits | Meaning |
|---|---|---|
| `ollama-proxy:ready` | `ready` | The proxy bound its HTTP port (`{version, port}`). Also replayed as the baseline on a fresh subscribe. |
| `ollama-proxy:node/discovered` | `node/discovered` | A manual node was added to the proxy (relay-sourced routing targets don't emit these). |
| `ollama-proxy:node/updated` | `node/updated` | A manual node's fields changed. |
| `ollama-proxy:node/removed` | `node/removed` | A manual node was removed. |
| `ollama-proxy:node/selection-changed` | `node/selection-changed` | The active upstream changed (`{id}`; empty `id` = auto-select). |
| `ollama-proxy:proxy/request-started` | `proxy/request-started` | An HTTP request began being forwarded. |
| `ollama-proxy:proxy/request` | `proxy/request` | An HTTP request completed (status, durations). |

```json
{"jsonrpc":"2.0","method":"ollama-proxy:node/discovered","params":{"id":"MY-GPU-BOX","host":"my-gpu-box.local.","port":11434,"addresses":["192.168.1.10"],"txt":["models=llama3"]}}
```

Two classes of proxy notification are **not** re-emitted under the `ollama-proxy:` namespace: `errors:report` / `errors:clear` are routed into the `nvpair-errors` pipeline and surface on the `errors:update` stream (see [`errors:update`](#errorsupdate)); the proxy's `workload:*` events go to the workload-manager and surface on the separate `workloads:*` stream below.

#### `workloads:upsert` / `workloads:remove`

**Opt-in.** Only delivered to a peer that has called `workloads:subscribe`; silent otherwise. Once subscribed, the broker pushes a `workloads:upsert` whenever a workload is created or its state changes, and a `workloads:remove` when one is retired. The stream is the union of two sources, in the same shape regardless of origin:

- **Local workloads** — the `workload:*` lifecycle events the enabled proxy
  facades emit per inference request, stamped with this host's stable `hostUuid`
  (`originatedFrom`) and translated to `workloads:upsert`. The proxy also fills
  in `scheduledOn` with the destination node's `hostUuid`; the broker passes
  that through unchanged.
- **Peer workloads** — the `workloads:upsert` / `workloads:remove` the `nvpair-workload-manager` relays from other nodes after validating and de-duplicating their broadcasts.

- **Retired history** — a `workloads:remove` for a completed or failed workload that fell outside the broker's history caps (the newest 10,000 terminal records, none older than seven days). The broker enforces the caps once a minute whether or not history persistence is on, so a subscriber that applies removals stays as bounded as the broker. Active workloads are never retired, and no removal is sent while a newer workload with the same `(originatedFrom, workloadId)` pair is still held. Retirement is local housekeeping and is not broadcast to peers.

- **Inferred workloads** — a `workloads:upsert` transitioning a workload to `failed` that **no origin ever sent**. The broker synthesizes one in two situations: when a node leaves discovery while workloads are pinned to it, and when a remote origin that is still present stops re-asserting a workload this node believes is running (the origin's re-sync heartbeat asserts each of its active workloads indefinitely, so prolonged silence about one means it is finished or the origin is gone). Both are recorded as *inferred*, so the origin's next authoritative event overrides them; a client should treat a `failed` as the broker's best current answer rather than proof the origin reported a failure, and its `error` text names the reason. Workloads this node originated or is itself executing are never inferred about.

`workloads:upsert` carries `params.workloadInfo` (a full `Workload`); `workloads:remove` carries `params.workloadId` and the origin `params.originatedFrom`. See [`nvpair-workload-manager`](../nvpair-workload-manager/README.md) for the `Workload` shape.

```json
{"jsonrpc":"2.0","method":"workloads:upsert","params":{"workloadInfo":{"id":"42","model":"llama-3-70b","engine":"ollama","state":"running","originatedFrom":"MY-PC","scheduledOn":"GPU-RIG","createdAt":1716998400000,"startedAt":1716998400000,"completedAt":null,"error":null,"requesterId":null}}}
```

A few semantics worth knowing:

- **Baseline is explicit.** `workloads:subscribe` itself replays nothing; subscribe first, then call `workloads:get-initial` and merge the snapshot with any overlapping transitions by workload key.
- **`upsert`, not a method replay.** A local `workload:completed` and a peer's running-state update both arrive as `workloads:upsert`; key on `(originatedFrom, engine, runId, id)` so concurrent engines and proxy restarts cannot collide.
- **Best-effort delivery**, like the other streams: one write attempt per event, transient errors logged not retried.

#### `errors:update`

The full service-error snapshot from `nvpair-errors`, relayed verbatim on every change (a bare `ServiceError[]`). **Not opt-in:** following the established error-reporting protocol the broker forwards it unconditionally. Every supervised worker's `errors:report` / `errors:clear` — proxy upstream-unreachable, engine-manager install/exit failures, manual-node probe failures, the node scanner's `node-scanner:node-info-unreachable:<hostUuid>` for a paired peer it cannot reach (the broker clears the ones still standing if the scanner crashes), and the supervisor's own `supervisor:subprocess-crashed:<name>` — is forwarded into `nvpair-errors`, deduped there, and re-emitted here. A client can also inject its own `errors:report` (see Requests below); it is forwarded into `nvpair-errors` the same way a worker's is.

```
ServiceError = { id, message, timestamp, nodeId?, severity?, action?, engineType?, operation?, modelName? }
```

#### `engine:<event>`

**Opt-in** (`engine:subscribe`). The notifications `nvpair-engine-manager` emits — `engine:ready`, `engine:state-changed` (full `EngineStatus`), `engine:models-changed` (`{engine, models}`, pushed when an engine's set of models loaded in memory changes), `engine:install-progress`, `engine:pull-progress` (live progress for a local model pull), `engine:remote-progress` (live progress for a remote install/pull) — re-emitted verbatim (their method names are already `engine:`-prefixed). Off by default.

#### `connection/cluster-identity` / `connection/cluster-auto-sync`

The change-only cluster signals `nvpair-node-settings` pushes when `cluster_id` / `cluster_auto_sync` change. Relayed verbatim and unconditionally — the same way the UI receives them from the settings subprocess directly.

#### `policy:changed` / `node:availability-changed`

**Not opt-in**: sent to every client. `policy:changed {nodeId, policy,
activeRevision, activation, activationError?}` follows a successful `policy:set`
and each persisted availability change (not one made during policy recovery),
once the push that activates it has been answered or has timed out. Each
announcement carries the policy persisted when it is sent and whether the proxy
enforces that revision (see Activation above): announcements of this node are
sent one at a time, each reading the state as it is sent, so their revisions
never go backwards and the last one matches `policy:get`. The same holds for
`node:availability-changed`. A push scheduled in the background announces again when its
answer changes the activation, and so does a pause's later availability push or
an availability push made during policy recovery. It is also
sent when a proxy start has been replayed the policy. `node:availability-changed {nodeId, availability,
active}` follows each availability transition (`draining`, then `paused`; or
`available`), and while draining it repeats whenever the active count changes.
Both are also sent with another node's `nodeId` after this node changed that
node's policy or availability; the remote form carries no `active`.

```json
{"jsonrpc":"2.0","method":"node:availability-changed","params":{"nodeId":"<hostUuid>","availability":"draining","active":2}}
```

### Requests (caller → broker)

#### `policy:get` / `policy:set` / `node:set-availability`

| Method | Params | Result |
|---|---|---|
| `policy:get` | `{ nodeId? }` | `{ policy, availability, recoveryRequired, activeRevision, activation, activationError? }` — `policy` is the persisted `nodepolicy.Policy` (`policy.revision` is the broker-assigned revision of the stored document); `availability` is live and may be `draining`; `recoveryRequired` is true while the node runs on the paused recovery policy because its file was unusable, until an explicit `policy:set` succeeds; `activeRevision` is the revision the running proxy acknowledged (empty when none, and after the newest availability push was refused, failed or not answered in time; a push still in flight leaves it as it was) and `activation` is `active`, `pending` or `error` (Activation above). Relayed from a paired node, `policy` is the peer projection (`nodepolicy.PeerPolicy`: no `routing`, no API tier entries) and the activation fields are absent |
| `policy:set` | `{ nodeId?, policy, expectedPersistedRevision }` | `{ policy, activeRevision, activation, activationError? }` after the precondition, validation, persisting and the bounded wait for the proxy to acknowledge that exact revision; `activation` describes `policy.revision` and is `pending` or `error` when it was saved but not activated; `policy.revision` against `activeRevision` shows both revisions. The wait is bounded, so a busy push slot yields `pending`. Fields the caller omits take their defaults. `policy.availability`, `policy.pausedBy` and `policy.revision` are ignored (the persisted availability and its provenance are kept: only `node:set-availability` changes them; the broker assigns the next revision). `expectedPersistedRevision` is required: the canonical decimal `policy.revision` of the `policy:get` the draft is based on. A missing, null, empty or non-canonical value is error `-32602` before anything else, and there is no unconditional write. It is compared with the persisted `policy.revision` under the same lock as the save, so two writers holding the same revision cannot both succeed; a mismatch is error `-32010` whose message starts with `policy conflict:` and names the current persisted revision, with nothing validated, persisted, pushed or announced. The caller reloads and reviews the policy rather than resubmitting the old draft. A failure before replacement is error `-32000` (`the policy could not be saved: ...`), with the file and running state unchanged. A landed replacement whose durability failed is also error `-32000`, saying the document with revision N was saved and is being applied but may not survive a power loss. That complete document is adopted, recovery is cleared, proxy supervision is reconciled, and the proxy push and `policy:changed` proceed as after success (Durability above). An invalid policy is error `-32602` and nothing changes, as is an enabled `routing.listener.port` that a configured facade uses (Supervision & recovery). Enabling or disabling the listener on a node with no facade starts or stops the proxy in the background, so that revision is `pending` until the new process acknowledges the replay. Relayed from a paired node, `expectedPersistedRevision` is required in the same way (the `revision` of the projection it read) and a mismatch is returned with `errorCode: -32010` and `policy conflict:` text in `policy:reply`, while invalid parameters carry `errorCode: -32602`; only device fields may change: `tiers` and `tierPolicy` must equal the projection and `routing` must be absent, or the set is rejected; the result is the peer projection |
| `node:set-availability` | `{ nodeId?, state: "available" \| "paused" }` | `{ availability, revision, activeRevision, activation, activationError? }` once the state is reached; pausing waits for the drain. `revision` is the persisted revision the activation fields describe (the one this change took, unless it was made during policy recovery); `activation` is `error`, with `activeRevision` empty, when the proxy refused or did not answer the availability push, even though the state was reached on this node (Activation). A request replaced by a newer one fails with `superseded by a later node:set-availability request`. Relayed from a paired node the result is `{ availability }` alone: like the relayed `policy:get`, it shows that peer no activation fields. A paired node's resume of a node paused locally is error `-32011` and changes nothing (Local veto) |

An omitted `nodeId`, or this node's own, acts locally; any other is relayed to
that node (see [Node policy](#node-policy)).

#### `workloads:cancel`

Params `{ originatedFrom, workloadId, regenerate?, engine?, runId? }`, result
`{ ok }`. Workload ids are per-engine counters, so `engine` (and `runId`, from the
workload record) pick the job when two share an id; the proxy refuses an
ambiguous id. Only the node a job originated on can cancel it. When
`originatedFrom` is this node the broker calls its proxy's
`workload/cancel {workloadId, regenerate, engine?, runId?}` and `ok` is the
proxy's `found`. Otherwise the cancel goes to `nvpair-workload-manager`, which
carries it to the origin over the pinned peer channel, and `ok:true` only means it
was handed on; the outcome arrives on the `workloads:*` stream. A peer's relayed
`workloads:cancel` reaches the broker from the workload-manager and is acted on
only by the origin. `-32602` without `originatedFrom` or `workloadId`.

#### `discovery:get-node-info`

```json
{"jsonrpc":"2.0","id":17,"method":"discovery:get-node-info","params":{"hostUuid":"physical-host"}}
```

Relays an asynchronous fresh read to the scanner over its existing stdio peer.
Desktop paired-node polling and TUI paired detail polling call it through
`nvpair-service`'s existing local control connection. The scanner owns current
directory/admission/pin resolution and direct/reverse `ni` transport; clients
supply only the host UUID. Self and unpaired client polling stays direct, and
manual TLS telemetry remains supplied by the manual-node worker.

The result is the full existing node-info shape, directly in `result`: `GPUs`,
optional `cpu`/`memory`, `telemetryValid`, `msSince`, `hostUuid` and nullable
`clusterUuid`. This is neither cached enrichment nor compact scheduler telemetry.
The scanner validates host and reported principal before success. Invalid or
extra params yield `-32602`; scanner failures retain their RPC code/message/data.
A missing scanner yields `-32000` (`node scanner unavailable`); deadline/transport
failure yields `-32000` (`node information unavailable`) without a fresh result.
These failures do not remove cluster membership.

Each broker/scanner read has a 1500 ms backend budget. Broker write-gate wait,
active scanner-input write and response wait are bounded. A response-only or
queued timeout leaves the scanner usable; an interrupted active frame retires
that scanner and existing supervision replaces it. Client timeout/cancellation
disowns late results locally; there is no wire cancellation method. The scanner
may retry this bodyless GET once over reverse after a direct first-data reset,
before response exposure. No remote-control renderer method or listener is added.

#### `discovery:get-nodes`

Returns the broker's current snapshot of every node `nvpair-node-scanner` has reported as live on the network. The list is sorted by `id` for stable rendering.

```json
{"jsonrpc":"2.0","id":1,"method":"discovery:get-nodes"}
```

Response:

```json
{
  "jsonrpc":"2.0",
  "id":1,
  "result":{
    "nodes":[
      {
        "id":"MY-PC",
        "name":"MY-PC",
        "ipAddress":"192.168.1.10",
        "port":14318,
        "lastSeen":1748275200
      }
    ]
  }
}
```

The per-node payload uses **camelCase** keys at this external boundary, in contrast to the snake_case the internal discovery wire uses elsewhere in the codebase.

Field-by-field:

| Field | Type | Source |
|---|---|---|
| `id` | string | the node's mDNS instance name (its hostname) from the `_nvpair-node` record |
| `name` | string | Currently mirrors `id`; will diverge once a node can advertise a human-readable name. |
| `ipAddress` | string | the node's authoritative `ip=` (else the best-ranked advertised address); empty string when none is usable. |
| `port` | number | the node's `ni` (node-info) port from its `_nvpair-node` TXT, typically `14318`. |
| `lastSeen` | number | **Unix seconds**. Updated on every `discovery:node-discovered` / `discovery:node-updated` event the broker receives from the daemon; not refreshed by `discovery:get-nodes` itself (a pure read). |
| `trusted` | bool | whether this node is a paired cluster peer (the daemon holds a pin for its `cluster-uuid`); false for non-cluster/unknown nodes. |
| `clustered` | bool | whether this node belongs to some cluster (it advertises a `cluster-uuid`), independent of whether we're paired with it (`trusted`). A client uses it to suppress a cluster invite that an already-clustered peer would reject. Omitted (false) for standalone/unknown nodes. |
| `models` | string[] | the node's available model names, enriched by the daemon from the node's engine-manager `em` endpoint (`GET /v1/models`). Omitted when the node advertises no engine-manager or no engine is running. |
| `modelsByEngine` | object | the same models attributed to the engine that serves each (keyed by engine name, e.g. `ollama` / `lmstudio`). Additive alongside the flat `models` union. A present engine key with `[]` means its inventory was successfully queried and is empty; a missing key means it was not running/queryable. Omitted when no engine inventory was successfully reported. |
| `loadedByEngine` | object | the models currently **loaded in memory** for each engine (normally a subset of `modelsByEngine`), keyed by engine name. Enriched from the peer's `loadedByEngine`. An engine key with an empty list means "running, nothing loaded"; a missing key means loaded state wasn't reported. Omitted when no engine reports loaded state. |
| `policy` | object | the node's own policy advertisement, enriched from the same `GET /v1/models` fetch: `revision` (its persisted policy revision), `roles` (tier name to `{engine, model, contextTokens?, capabilities?}`) and `pausedBy?` (`local` or `peer`). Omitted when the node publishes none (an older peer, or before its broker's first push), which means the node declares nothing. |

A node that the scanner reports as removed is deleted from the snapshot; `lastSeen` is not preserved for removed nodes. The response is wrapped in `{nodes: [...]}` (rather than a bare array) so we can grow summary fields later without breaking clients. An empty list is a normal early-startup state, not an error.

The broker also stores the scanner's richer per-node payload internally (`host`, all `addresses`, `txt`, `gpus`, `cpu`, `memory`); that data isn't currently exposed but will back future RPCs without re-querying the scanner.

#### `discovery:subscribe`

Opts the peer into the `discovery:nodes-changed` notification stream. The stream is **off by default** — a freshly connected peer receives no `discovery:nodes-changed` until it calls this. On a fresh subscription the broker immediately pushes one `discovery:nodes-changed` with the current snapshot (a baseline) right after the response, so a subscriber doesn't need a separate `discovery:get-nodes` to seed its view.

```json
{"jsonrpc":"2.0","id":6,"method":"discovery:subscribe"}
```

Response:

```json
{"jsonrpc":"2.0","id":6,"result":{"subscribed":true}}
```

Idempotent: subscribing while already subscribed returns `{"subscribed":true}` and does **not** re-send a baseline (the stream is already flowing). The subscription is per-connection state — it does not persist across reconnects.

#### `discovery:unsubscribe`

Stops the `discovery:nodes-changed` stream for this peer. The discovery store keeps tracking nodes; the peer simply stops being notified. `discovery:get-nodes` still works while unsubscribed.

```json
{"jsonrpc":"2.0","id":7,"method":"discovery:unsubscribe"}
```

Response:

```json
{"jsonrpc":"2.0","id":7,"result":{"subscribed":false}}
```

Idempotent: unsubscribing while not subscribed returns `{"subscribed":false}` with no error.

#### `ping`

Liveness check. Returns the broker's version and how long it has been running.

```json
{"jsonrpc":"2.0","id":2,"method":"ping"}
```

Response:

```json
{"jsonrpc":"2.0","id":2,"result":{"pong":true,"version":"0.5.0","uptime_ms":1234}}
```

#### `version`

Returns the broker's stamped version.

```json
{"jsonrpc":"2.0","id":3,"method":"version"}
```

Response:

```json
{"jsonrpc":"2.0","id":3,"result":{"version":"0.5.0"}}
```

#### `ollama-proxy:get-status`

Reports whether the supervised `ollama-proxy` is up and, if so, the HTTP port it bound. `ready` is `false` and `port` is `0` until the proxy emits its `ready` notification — and stays that way if no proxy is being supervised or the proxy has crashed — so the call is always safe and never errors. Answered locally from the broker's captured state (no round-trip to the proxy).

```json
{"jsonrpc":"2.0","id":8,"method":"ollama-proxy:get-status"}
```

Response (once the proxy is serving):

```json
{"jsonrpc":"2.0","id":8,"result":{"ready":true,"port":11435}}
```

`port` is the proxy's HTTP reverse-proxy listener — point an Ollama client at `http://localhost:<port>` to route inference traffic through it. The proxy comes up asynchronously, so a client typically polls this shortly after `app:ready` until `ready` is `true`.

#### `ollama-proxy:<method>` (generic relay)

Any request whose method starts with `ollama-proxy:` — except the reserved broker-local methods (`ollama-proxy:get-status`, `ollama-proxy:set-port`, and the subscription methods below) — is **forwarded to `ollama-proxy`** with the `ollama-proxy:` prefix stripped, and the proxy's result or error is relayed straight back to the caller unchanged. The `ollama-proxy:` prefix is the only translation: everything after it is the proxy's own native method name, and `params` pass through verbatim. This means the proxy's whole control plane is reachable through the broker without the broker enumerating individual methods. Current proxy methods include `nodes/list`, `node/select`, `node/selected`, `node/add-manual`, and `node/remove-manual` (see the `ollama-proxy` README for their params/results).

For example, to list the Ollama nodes the proxy currently knows and pin traffic to one:

```json
{"jsonrpc":"2.0","id":10,"method":"ollama-proxy:nodes/list"}
{"jsonrpc":"2.0","id":11,"method":"ollama-proxy:node/select","params":{"id":"MY-GPU-BOX"}}
```

The responses are exactly what the proxy returns (its payloads use the proxy's own snake_case `Node` schema — `id` / `host` / `port` / `addresses` / `txt` — not the camelCase `discovery:*` shape; the `ollama-proxy:` namespace is a deliberate exception kept thin to avoid coupling).

Two relay-specific error cases:

- If no proxy is being supervised (or it has exited), the broker replies with error `-32000` `"ollama-proxy not available"`.
- `ollama-proxy:shutdown` is **refused** with error `-32601` — the broker owns the proxy's lifecycle, so a client can't terminate it independently. Shut the broker down instead (which tears the proxy down with it).

#### `ollama-proxy:set-port` / `lmstudio-proxy:set-port` / `llamacpp-proxy:set-port`

**Intercepted, not relayed verbatim.** For every engine profile, a port-only caller — `nvpair-tui` is the one in tree — gets to move a single port without rendering the whole launch settings form, but the change still runs through the same authoritative settings operation the desktop editor uses, so a port set from the terminal cannot diverge from one set from the UI. The namespace selects the engine. The broker reads that engine's current settings, substitutes the requested proxy port, and applies the result while preserving the server port and launch arguments. Accepted settings are saved in the settings journal.

A malformed request, missing port, or port outside `1`–`65535` is refused with error `-32602 "port must be between 1 and 65535"`. A **requested port that is already in use or reserved** by a PAIR service, another configured engine/proxy, or the engine's own server port is refused with error `-32000 "resolve settings errors and port conflicts before applying"`. The broker does not pick a different port on the caller's behalf. Retry with a free port. A request that collides with an inherited `OLLAMA_HOST` alias is refused with its own message naming that alias. The response echoes the requested port (`{"port": <requested>}`) once it is bound.

```json
{"jsonrpc":"2.0","id":9,"method":"ollama-proxy:set-port","params":{"port":11500}}
{"jsonrpc":"2.0","id":10,"method":"llamacpp-proxy:set-port","params":{"port":8082}}
```

Automatic conflict resolution still exists, but only for a port the user did not just choose: when the Ollama proxy announces a (re)bound port on startup and a running engine has since taken it, the broker steers the proxy to a free port and surfaces a sticky `warning` into the errors pipeline (id `ollama-proxy:port-bumped`, `action:"none"`) explaining the move. That path **never changes an engine's port** — only the proxy is moved. Port setters also return `-32000` if the settings operation cannot run, including when the engine manager or proxy is unavailable.

#### `lmstudio-proxy:get-status` / `lmstudio-proxy:subscribe` / `lmstudio-proxy:unsubscribe` / `lmstudio-proxy:<method>` (generic relay)

The LM Studio counterpart of the `ollama-proxy:*` surface runs the supervised `lmstudio-proxy` on compatibility port `:1234` and tracks the managed LM Studio backend on `:1235`. With managed port ownership enabled (the default), the broker identifies and moves an existing LM Studio server through engine-manager before allowing the proxy to claim `1234`; unknown owners are left untouched and force a warned proxy fallback. Disabling managed ownership preserves explicit custom backend and proxy ports. `lmstudio-proxy:get-status` reports the actual bound port; `lmstudio-proxy:subscribe` / `lmstudio-proxy:unsubscribe` opt into / out of its `lmstudio-proxy:<event>` stream; `lmstudio-proxy:set-port` uses the settings operation described above; and other `lmstudio-proxy:<method>` requests are relayed with the prefix translated (`nodes/list`, `node/select`, `node/add-manual`, `node/remove-manual`, ...). `lmstudio-proxy:shutdown` is refused because the broker owns lifecycle ordering. Workload and error events feed the shared streams exactly as Ollama's do.

The same broker-local status, subscription, and port-setting methods apply to `llamacpp-proxy:*`. llama.cpp port changes use the same validation, journal, and application path; remaining methods use the generic facade relay, with `llamacpp-proxy:shutdown` refused.

#### `ollama-proxy:subscribe`

Opts the peer into the `ollama-proxy:<event>` stream (off by default). On a fresh subscription the broker immediately replays the proxy's last `ready` payload as a baseline `ollama-proxy:ready` (if the proxy has come up), so a subscriber learns the port without a separate `ollama-proxy:get-status`.

```json
{"jsonrpc":"2.0","id":12,"method":"ollama-proxy:subscribe"}
```

Response:

```json
{"jsonrpc":"2.0","id":12,"result":{"subscribed":true}}
```

Idempotent: re-subscribing returns `{"subscribed":true}` and does not re-send a baseline. This is a reserved broker-local method — it is not forwarded to the proxy.

#### `ollama-proxy:unsubscribe`

Stops the `ollama-proxy:<event>` stream for this peer. Reserved broker-local; not forwarded. Idempotent.

```json
{"jsonrpc":"2.0","id":13,"method":"ollama-proxy:unsubscribe"}
```

Response:

```json
{"jsonrpc":"2.0","id":13,"result":{"subscribed":false}}
```

#### `workloads:subscribe`

Opts the peer into the `workloads:upsert` / `workloads:remove` stream (off by default). No baseline is pushed automatically; subscribe first, then call `workloads:get-initial` and merge by workload key so no transition can fall into a snapshot/subscribe gap.

```json
{"jsonrpc":"2.0","id":14,"method":"workloads:subscribe"}
```

Response:

```json
{"jsonrpc":"2.0","id":14,"result":{"subscribed":true}}
```

Idempotent: re-subscribing returns `{"subscribed":true}`. Per-connection state; not persisted across reconnects.

#### `workloads:unsubscribe`

Stops the `workloads:*` stream for this peer. Idempotent.

```json
{"jsonrpc":"2.0","id":15,"method":"workloads:unsubscribe"}
```

Response:

```json
{"jsonrpc":"2.0","id":15,"result":{"subscribed":false}}
```

#### `workloads:get-initial`

Returns `{ "workloads": [Workload, ...] }`, the broker's full-fidelity current and historic workload snapshot. Active records are kept in memory; completed/failed history is bounded by the caps described under `workloads:remove` and persisted across broker restarts. The scheduler uses an internal active-only replay of the same catalog when its subprocess restarts.

#### `errors:get-initial`

Returns the current `ServiceError[]` snapshot from `nvpair-errors` (an empty array when no datastore is supervised, so it never errors). Relayed to `nvpair-errors`' own `errors:get-initial`.

#### `errors:clear`

Clears a service error by id. Params: `{ "id": "<error id>" }`. The broker stamps the `clearedBy` attribution and forwards the clear to `nvpair-errors`; it acks `null` immediately, with the authoritative state change arriving on the next `errors:update`.

#### `errors:report`

Forwards a client-originated service error into `nvpair-errors`, letting a client surface its own operational errors (e.g. a model-pull failure the engine itself doesn't report) through the same registry that drives `errors:update`. Params are a `ServiceError` (`{ id, message, severity?, action?, ... }`); the broker fills in the origin `nodeId` and `timestamp` when omitted, exactly as it does for a supervised worker's report. Accepted in two forms: as a **notification** (fire-and-forget, mirroring how a worker emits one on its stdout) or as a **request** (acked `null` once received — like `errors:clear`, the authoritative state change arrives on the next `errors:update`; an empty `id` is rejected with `-32602`).

#### `engine:subscribe` / `engine:unsubscribe`

Opt into / out of the `engine:<event>` stream (off by default). Acks `{ subscribed: bool }`. No baseline is replayed — call `engine:get-installed` for the initial snapshot.

#### `engine:<method>` (generic relay)

Any other `engine:*` request is forwarded to `nvpair-engine-manager` verbatim and its response relayed straight back. This covers the whole engine control plane: `engine:get-installed`, `engine:describe`, `engine:status`, `engine:install`, `engine:uninstall`, `engine:start`, `engine:stop`, `engine:restart`, `engine:action`, `engine:logs`, `engine:errors`, `engine:models`, and the node
policy's `engine:wake`, `engine:sleep`, `engine:intent` and
`engine:unload-model`. `engine:wake` is refused by the broker while this node is
paused or draining. Lifecycle ops run for minutes (reporting progress via the `engine:install-progress` / `engine:state-changed` push events), so the relay imposes **no broker-side timeout** — fire the request and watch the event stream for the outcome. Error `-32000 "engine-manager not available"` when no engine-manager is supervised.

`engine:set-port` is **not** in that generic set. Like `<engine>-proxy:set-port` it is intercepted and run through the authoritative settings operation, so moving an engine's server port from a port-only caller validates and restarts exactly as the full editor does, and persists as a manifest override that survives a restart. Its response is the engine's `engine:status` result.

#### `settings/<method>` (generic relay)

Any `settings/*` request is forwarded to `nvpair-node-settings` and its response relayed back: `settings/get-force-ports`, `settings/set-force-ports`, `settings/get-cluster-id`, `settings/set-cluster-id`, `settings/get-cluster-auto-sync`, `settings/set-cluster-auto-sync`, `settings/get-cluster-friendly-name`, `settings/set-cluster-friendly-name`. Error `-32000 "node-settings not available"` when no settings worker is supervised.

#### `node/add` / `node/remove` / `nodes/list` (manual nodes)

Relayed to `nvpair-manual-nodes`. `node/add` (`{ address, name?, tls_port?, mtls? }`) registers a user-added node and probes it; `node/remove` (`{ id }`) drops it; `nodes/list` returns the tracked manual nodes. Manually added nodes also surface in the shared `discovery:get-nodes` / `discovery:nodes-changed` snapshot — the broker merges `nvpair-manual-nodes`' `node/discovered|updated|removed` into the same store the scanner feeds. `nvpair-manual-nodes` persists its list to `manual-nodes.json` in the per-user data dir and replays it when it starts, so manual nodes survive a restart of the worker, the broker, or the machine without any client re-adding them: on a worker crash-and-restart the broker first evicts the orphaned entries, then merges each replayed entry back in as its `node/discovered` arrives. Error `-32000 "manual-nodes not available"` when no manual-nodes worker is supervised.

**Manual → proxy bridge.** When the broker supervises both `nvpair-manual-nodes` and a proxy, it also bridges a manual node whose engine is reachable into that proxy via `node/add-manual` (host/port from the node's per-engine status), so inference can route to it through `ollama-proxy:node/select` / `lmstudio-proxy:node/select` just like a relay-discovered node. This is per-engine: a node whose `ollama_*` status is up is bridged into `ollama-proxy` (host/port from `ollama_port`), and one whose `lmstudio_*` status is up into `lmstudio-proxy` (from `lmstudio_port`) — a node running both is bridged into both. Manual nodes are by definition the ones that never appear via the daemon's `_nvpair-node` discovery, so this explicit add is what makes them routable. The bridge tracks reachability: an engine that goes down (or a node that is removed, or whose prober crashes) is pulled back out with `node/remove-manual`. A proxy that isn't supervised → that leg is a no-op; manual nodes still appear in the discovery snapshot as before.

#### `cluster:<method>` / `nodes:<method>` (generic relay)

Requests whose method starts with `cluster:` or `nodes:` are **forwarded to `nvpair-cluster-manager`** verbatim (no prefix stripping — these are the manager's own native namespaces), and the manager's result or JSON-RPC error is relayed straight back to the caller unchanged. The private `cluster:set-local-services` method is excluded: client requests receive method-not-found and client notifications are discarded. Current public methods include `cluster:get-node-id`, `cluster:set-identity`, `cluster:create`, `cluster:invite-node`, `cluster:respond-to-invite`, `cluster:cancel-invite`, `cluster:invite-status`, `nodes:get-initial`, and `nodes:remove` (see the [`nvpair-cluster-manager` README](../nvpair-cluster-manager/README.md) for their params/results and app-defined error codes).

The broker sends `cluster:set-local-services` only down the child's stdio link,
with `{"generation":UINT64,"services":[noderec.RegisterParams,...]}`. This is an
authoritative full snapshot of the broker's current registration cache, including
removals and empty arrays. Engine entries (`ol`, `lm`, `lc`) contain registered
proxy listener ports, never the raw engine ports. Each real cache mutation
advances the generation; every cluster-manager spawn or respawn receives the
current baseline even when no registration changes. Snapshot capture and queueing
are serialized, and a blocked old child cannot delay its replacement or broker
registration/shutdown. Cluster-manager owns the listener protocol classification;
the broker supplies no user-selected TLS policy and does not infer it from
discovery transport labels. This metadata has no desktop consumer.

The cluster-manager also pushes notifications (`cluster:invite-received`, `cluster:invite-canceled`, `cluster:identity-changed`, `nodes:changed`, ...); the broker forwards them to the caller verbatim. Unlike `ollama-proxy:<event>` / `workloads:*` these are **not opt-in** — there is no `cluster:subscribe`; the events always flow.

If no cluster-manager is being supervised (or it has exited), the broker replies with error `-32000` `"nvpair-cluster-manager not available"`. Because pairing calls drive multi-round-trip inter-node network exchanges, the relay's per-call timeout is 30 s (vs. the proxy's 5 s).

#### `shutdown`

Acknowledges, then terminates the broker. The proxy is stopped first, then engine-manager is asked to stop its engines (`engine:prepare-shutdown`), then every running worker subprocess is torn down by closing its stdin and waiting for it to exit — no grace timer, no force-kill (each worker bounds its own shutdown).

```json
{"jsonrpc":"2.0","id":4,"method":"shutdown"}
```

#### `log/set-level`

Standard NVPAIR log-level toggle (see `shared/applog`). Accepts `debug` \| `info` \| `warn` \| `error`. The broker applies the level to itself and then **fans it out to every child subprocess it supervises** (the scanner, and — when running — node-info, the proxy, the workload-manager, the cluster-manager, and the job-scheduler) by forwarding a `log/set-level` notification down each child's stdin, so a single call adjusts the whole process tree. The response reflects the broker's own resolved level; child forwarding is best-effort (a child that's mid-teardown or never started is skipped and logged, never failing the call).

```json
{"jsonrpc":"2.0","id":5,"method":"log/set-level","params":{"level":"debug"}}
```

Response:

```json
{"jsonrpc":"2.0","id":5,"result":{"level":"debug"}}
```

## Shutdown

The broker shuts down on:

- stdin EOF (parent closed the pipe) or IPC peer disconnect
- `SIGINT` / `SIGTERM`
- A `shutdown` JSON-RPC request

## Running

The broker needs `nvpair-node-scanner` reachable as a sibling file in its working directory (or via `--scanner-path`); it also looks for `nvpair-node-info` (or via `--node-info-path`), `nvpair-proxy` (or via `--proxy-path`), `nvpair-workload-manager` (or via `--workload-manager-path`), and `nvpair-cluster-manager` (or via `--cluster-manager-path`) the same way (plus errors / engine-manager / manual-nodes / settings), though all but the scanner are optional. The simplest local layout is to put all the built `.exe`s next to each other — which is exactly how the installer lays them out.

Quick smoke test in stdio mode (PowerShell, run from a directory that has the worker binaries — at minimum `nvpair-node-scanner`, plus `nvpair-node-info` if you want the local inventory server, `nvpair-proxy` if you want the local engine proxies, `nvpair-workload-manager` if you want the cluster workload relay, and `nvpair-cluster-manager` if you want cluster pairing):

```powershell
'{"jsonrpc":"2.0","id":1,"method":"discovery:get-nodes"}' | .\nvpair-ui-broker.exe
```

You should see an `app:ready` notification followed by a `discovery:get-nodes` response. The `nodes` array will likely be empty on the first request — mDNS browsing needs a few seconds to populate the store. Issue the call again after a short wait to see real entries (assuming an `_nvpair-node._tcp` record is live on the LAN). To watch live updates, subscribe to the push stream first — every store change then pushes a `discovery:nodes-changed` notification:

```powershell
'{"jsonrpc":"2.0","id":1,"method":"discovery:subscribe"}' | .\nvpair-ui-broker.exe
```

Pass explicit worker paths:

```powershell
.\nvpair-ui-broker.exe --scanner-path C:\path\to\nvpair-node-scanner.exe --node-info-path C:\path\to\nvpair-node-info.exe --proxy-path C:\path\to\nvpair-proxy.exe --workload-manager-path C:\path\to\nvpair-workload-manager.exe --errors-path C:\path\to\nvpair-errors.exe --engine-manager-path C:\path\to\nvpair-engine-manager.exe --manual-nodes-path C:\path\to\nvpair-manual-nodes.exe --settings-path C:\path\to\nvpair-node-settings.exe --cluster-manager-path C:\path\to\nvpair-cluster-manager.exe
```

Attach to a pre-existing endpoint:

```powershell
# Windows named pipe (the UI process must already be listening on it)
.\nvpair-ui-broker.exe --ipc \\.\pipe\nvpair-ui-broker
```

```bash
# Unix domain socket (the UI process must already be listening on it)
./nvpair-ui-broker --ipc /tmp/nvpair-ui-broker.sock
```

## Performance history and projection

The broker owns destination-local inference performance aggregation and the
dedicated application-data file `performance-history.json`. It loads history
before accepting samples. Dirty samples coalesce into a durable save on a
five-second timer, with a final flush joined for at most five seconds after
worker teardown. Startup and each distinct workload-manager incarnation force
a durable `Store.Checkpoint()` before publication, including when no sample
changed. A current local admission epoch change also forces a checkpoint.
These restart/authority checkpoints are separate from the dirty-save timer.

Only a successful durable save/checkpoint becomes eligible for publication.
A failed write keeps the live local view available and sets persistence
degraded; even a replacement adopted by the store with a durability error is
withheld until a later clean checkpoint. A blocked disk, manager stdin or
client output does not hold the sample-ingestion path. Disk persistence, the
coalesced manager sender and the client change sender run independently.

Corrupt, oversize or unreadable regular history is moved to a unique sibling
`performance-history.json.corrupt-<unix-ms>` without replacing prior quarantines.
The store warns with the quarantine path only, starts empty and exposes
`persistence.stats.HistoryReset`. If quarantine fails, the original remains
untouched, ingestion/display continue in memory and publication stays blocked.
Losing or quarantining the file also loses its revision floor. Revisions use
the existing larger-of-previous-plus-one/current-wall-milliseconds rule; if the
clock is behind a peer's retained floor, that peer rejects lower revisions until
a later successful save or checkpoint issues a revision above the floor.

Public JSON-RPC methods accept absent, null or empty-object params:

| Method | Result and behavior |
| --- | --- |
| `performance:get` | Current full projection, without subscribing. |
| `performance:subscribe` | `{subscribed:true}`, then an asynchronous `performance:changed` baseline and subsequent full replacements. Repeated subscriptions are idempotent. |
| `performance:unsubscribe` | `{subscribed:false}`; disables subsequent changes. A notification already being written can finish. |

The `performance:changed` payload is identical to the read result:

```text
{
  local: {measuringHost, status, lastSeenAgeMs, summary?, expiredEntries},
  peers: [{measuringHost, status, lastSeenAgeMs, summary?, expiredEntries}],
  persistence: {degraded, publicationBlocked, error?, stats},
  expiredPeers
}
```

`status` is `fresh`, `stale`, `unknown` or `missing`. Local data is live broker
aggregation, including unsaved samples, and uses projection-only revision zero
with an empty admission incarnation. An empty local history is `missing`.
An admitted peer without an accepted summary is `missing`; revoked/retired data
is `unknown` with no summary. `stats` is `performancestore.Stats`: `Keys`,
`Samples`, `Dirty`, `PublishedRevision`, `HistoryReset`, `EvictedKeys`,
`EvictedSamples`, `ExpiredSamples` and `Duplicates`. `summary.omittedEntries`
reports source keys omitted by the wire cap. `expiredEntries` counts projected
keys whose newest sample is beyond the source retention window; `expiredPeers`
counts payloads retired after an hour without a heartbeat. Unknown metric
values remain absent, preserving the shared wire's nullable measurements.

`Broker.consumePerformanceNotification` privately consumes the manager's
`performance:received` and `performance:status` contracts from workload-manager
specification section 7.4. The receiver-local opaque admission token and the
canonical positive decimal sender epoch must both match the current shared
`Mesh.PeerServiceIncarnation` and `Mesh.PeerAdmissionEpoch`. Checks bracket cache
mutation and projection; discovery names and aliases grant no authority.
Unknown fields, missing/null required metadata, invalid numbers, oversized
summaries, old revisions and changed-body equal revisions are rejected. Each
peer holds a latest full summary, never additive totals, bounded by the shared
128 KiB summary cap. The cache retains at most 128 peer records including
revision/digest tombstones; at capacity, new peers are withheld rather than
evicting a floor. A newly validated sender epoch resets its revision floor.
Certificate or receiver-local admission changes retire payloads while keeping
the same sender epoch's floor. Equal revisions and delayed fresh statuses cannot
revive retired payloads; a higher revision from current authority can.

Source wire bytes and ages remain immutable per revision. Reads add
`sampleElapsedMs` and receiver-monotonic elapsed time with saturating arithmetic.
Greater duplicate/status offsets raise the age floor; smaller offsets and clock
rollback cannot lower it. A heartbeat refreshes reachability independently of
sample age. Silence makes the view stale at 90 seconds and retires its payload
at one hour; reads revalidate immediately and periodic sweeps run every five
seconds independently of client, disk-save and worker writes. Status offsets can
exceed the seven-day retention window; initial received offsets are bounded by it.

Own-host publication is a private `performance:publish` notification containing
the shared summary. It requires the current `Mesh.LocalAdmissionEpoch` and this
host's canonical UUID. The sender selects the latest successfully saved revision
and freezes its body as late as possible, immediately before the first stdin
write; retries reuse identical bytes. Successful unchanged revisions are not
resent; the manager owns peer heartbeats. Broker/manager restarts checkpoint and
rebase saved ages without rejuvenating samples. The first manager shares the
broker startup checkpoint. A blocked local stdio write's delay is not represented
in ages: the immutable local publish contract has no
elapsed header, and the manager anchors elapsed time when it receives the
publication. The same stdin still carries ordinary lifecycle traffic; this is
not a separate worker transport.

The future T2 boundary is `Broker.ingestPerformanceSample(json.RawMessage) error`,
reached exactly once through the proxy's unaddressed, process-scoped
`performance:sample` notification. It strictly decodes and validates
`performancewire.Sample`, checks the measuring host through the store and
deduplicates the producer/sequence pair. It never writes disk or worker stdin.
No proxy extractor emits these samples yet. This projection has no scheduler or
ranking input, and desktop/TUI bridge consumers and UI are later work. Public
contract drift for these methods must be addressed by that integration; the
private worker exceptions do not exempt the public surface. Unsubscribed clients
receive neither performance changes nor private manager frames.

## What this version intentionally does NOT do (yet)

- **Engine-advertise control surface.** Engine registration is auto-driven only:
  the broker tracks local Ollama, LM Studio, and llama.cpp and registers `ol`,
  `lm`, or `lc` with the daemon while each is up. There's no manual-advertise
  RPC (custom service, port, name, or TXT), and no way to advertise anything
  other than the detected engines.
- **node-info control surface.** node-info is spawned and torn down with the broker, and the broker pushes it only two things over stdin: the log level, and this node's cluster principal (`nodeinfo:set-cluster-identity`, sent on spawn and on every membership or pin-set change, because node-info holds no cluster dir and so cannot read membership itself). The broker registers its plain port with the daemon and passes no TLS material; its only port flag is the `--port` override from `--node-info-port`. `discovery:get-node-info` queries admitted peers through the scanner; it does not reconfigure the node-info worker.
- **Per-event push semantics.** `discovery:nodes-changed` always carries the full current snapshot, not a delta. For small N this is fine and lets the client treat the payload as authoritative without state reconciliation. `errors:update` is likewise a full snapshot.
- **BYO-TLS node-info discovery.** The daemon enriches peers over plain HTTP per the consolidated transport policy (node-info is plain on the broker path). Reading a node-info served over operator-configured BYO-TLS / mTLS is not supported through the broker; discovery works for the plain-HTTP case (the common one).
- **Per-worker log-level granularity.** `log/set-level` fans the broker's single level out to every running child, but there's no way to set a different level per worker — it's one level for the whole process tree.
- **AuthN / AuthZ.** Transport security relies on the parent's pipe / socket ACL. There is no per-message token.
- **HTTP / WebSocket transport.** JSON-RPC over stdio or pipe only — no web bridge yet. Likely lives in a separate component if/when added.
- **Automatic workload baseline on subscribe.** `workloads:subscribe` starts only the live stream; clients explicitly request `workloads:get-initial` after subscribing and merge overlapping records. The broker persists bounded terminal history, while active state is rebuilt from live workload reconciliation rather than treated as durable across a full process-tree restart.
