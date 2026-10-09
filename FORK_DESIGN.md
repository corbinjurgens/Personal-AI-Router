<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fork Design

How the fork's features fit into PAIR's process model. [FORK.md](FORK.md) is
the roadmap and [WORKLOG.md](WORKLOG.md) the history; this file is the
technical contract the implementation follows. Where it names a JSON-RPC
method or field, the Go source is authoritative once implemented.

## 1. Process model after the fork

```text
Electron app ─┐                       ┌─ nvpair-ui-broker ─┬─ proxy (admission, tiers)
nvpair-tui ───┼─ local socket/pipe ── nvpair-service       ├─ engine-manager
              │  (attach / detach)    (owns broker stdio)  ├─ node-settings, manual-nodes, …
another client┘                                            └─ scheduler, workload-manager, …
```

- `nvpair-service` is a new, per-user, long-running process. It owns the
  broker exactly as Electron used to, and it outlives every client.
- Electron and the TUI become clients: they connect to the service and start
  it (detached) if it is not running. Quitting either one detaches. Stopping
  the service is an explicit action.
- The broker is unchanged in shape. It still speaks newline-delimited JSON-RPC
  over stdio to a single parent, now `nvpair-service`.

## 2. nvpair-service

**Endpoint.** On Linux and macOS this is `<appdir>/service.sock`, a Unix socket
with mode 0600. On Windows it is the named pipe
`\\.\pipe\nvpair-service-<username>`, restricted to the current user.
`<appdir>` is `nvpair-shared/appdir.Dir()`. `shared/ipc.Listen` and `ipc.Dial`
already cover both platforms.

**Single instance.** At startup the service dials the endpoint. If something
answers, it exits with status 0 and the message "already running". A stale
Unix socket file is removed.

**Broker supervision.** The service spawns `nvpair-ui-broker` from its own
directory, with the working directory set to that directory and broker
arguments passed through after `--`. If the broker exits unexpectedly, the
service restarts it with backoff and sends `service/broker-restarted` to
clients.

**Multiplexing.**
- Each client request's `id` is rewritten to a service-unique id, and the
  response is routed back with the original id.
- Every broker notification is broadcast to every attached client. There are a
  few exceptions:
  - The service caches the last `app:ready` and replays it to clients that
    attach later.
  - `*:subscribe` requests are forwarded, since the broker treats them as
    idempotent.
  - `*:unsubscribe` requests are answered locally with success and never
    forwarded, because another client may still depend on the stream.
  - A client's `shutdown` request is answered locally and then that client is
    detached. The broker is not stopped.

**Service methods.** These are handled by the service itself:
- `service/status` returns `{pid, brokerPid, clients, startedAt, version}`.
- `service/stop` stops the broker cleanly (the same `shutdown` sequence the
  TUI used: request, close stdin, then wait up to 18 s before killing it), then
  exits.

**Logs.** Broker stderr lines go to `<appdir>/logs/broker.log`, rotated at
10 MiB with one old generation kept. They are also broadcast as
`service/log {source, stream:"stderr", text}`. A client must redact before
storing or showing them, exactly as it redacts child stderr today.

**Autostart.**
- `nvpair-service autostart enable|disable|status` registers the service to
  start at login:
  - Windows: an `HKCU\…\Run` value.
  - macOS: a `~/Library/LaunchAgents` plist.
  - Linux: an XDG autostart `.desktop` entry.
- Autostart is off unless enabled.

## 3. Node policy (phases 3–5)

The policy is one JSON file per node, `<appdir>/node-policy.json`. Its schema,
defaults and validation live in `services/shared/nodepolicy`. The broker owns
the file and is its only writer. The proxy enforces it.

### 3.1 Client-facing broker methods

| Method | Params | Result |
|--------|--------|--------|
| `policy:get` | `{nodeId?}` | `{policy, availability}`; `availability` is live and may be `draining` |
| `policy:set` | `{nodeId?, policy}` | `{policy}` after validation and persisting |
| `node:set-availability` | `{nodeId?, state: "available" \| "paused"}` | `{availability}`; returns once the state is reached (pausing waits for the drain) |
| `workloads:cancel` | `{originatedFrom, workloadId, regenerate?}` | `{ok}` |

Notifications:
- `policy:changed {nodeId, policy}` and
  `node:availability-changed {nodeId, availability, active}`.
- Both are sent to every client without a subscription.

Behavior when `nodeId` names another node:
- The broker forwards the call through engine-manager's pinned-mTLS control
  server. This uses the same pattern as remote engine settings
  (`settingsremote.go`): routes `POST /v1/node-policy/{get,set}` and
  `/v1/node-availability/set`, relayed to the target's broker as
  `policy:request`.
- A remote caller cannot forward a further hop.

### 3.2 Broker ↔ proxy

The methods and payload types are defined in `nodepolicy/wire.go`.

Broker → proxy requests:
- `node/set-policy`, `node/set-availability`, `node/set-engine-drain`,
  `node/set-residency`, `node/set-engine-intent`.
- All five are re-sent whenever the proxy restarts.

Proxy → broker notifications:
- `admission/state`, `admission/unload`, `admission/wake`.

### 3.3 Admission (proxy, destination side)

Admission is one node-wide controller in the proxy process. All engine
facades share it. It runs in two places:
- In `handleClusterIngress`, before `reverseProxyToLocal`, for work that peers
  send here.
- In `handleHTTP`, before dispatching to the self candidate, for local work.

Both places read the body's `model` (within the existing size cap). A
rejection is a `503` carrying `X-PAIR-Admission: <reason>` and `Retry-After`.
On the routing node, a rejected self candidate moves on to the next candidate
the same way a peer's 503 does.

Decision order for `(engine, model)`:
1. **Availability.** If it is not `available`, reject with `paused` or
   `draining`. If the engine is draining for a settings restart, reject with
   `draining`.
2. **Engine running?** If the engine is stopped:
   - If its saved intent is Off, reject with `engine-off`.
   - Else if `idle.startOnDemand` is set, send `admission/wake` and wait for
     `node/set-local-backend` to report the engine healthy, up to
     `idle.wakeTimeoutSeconds`. On timeout, reject with `wake-timeout`.
3. **Profile.** Look up the profile by `(engine, normalized model)`. Merge its
   `requestOptions` into the body, with keys already in the request winning.
4. **Concurrency.** The limit is `profile.maxConcurrent`, or else
   `admission.maxConcurrentPerModel`. If the model is at its limit, wait.
5. **Residency.** Count the resident set: models loaded on any engine
   (`node/set-residency`) plus models with in-flight requests. If adding the
   requested model would exceed `admission.maxResidentModels`:
   - Evict idle models (loaded, nothing in flight) by sending
     `admission/unload`, if `switchModels` is set.
   - Otherwise wait for in-flight work on other models to finish.
6. **Memory.** If `memoryBudgetBytes > 0`, add up the `memoryBytes` of resident
   and in-flight profiles (unknown counts as 0). If adding this model exceeds
   the budget, evict idle models or wait, as in step 5.
7. **Waiting.** Every wait is bounded:
   - The bound is `min(admission.queueTimeoutSeconds, X-PAIR-Admission-Wait)`.
     The router sends `X-PAIR-Admission-Wait: 0` while it still has other
     candidates, so a busy node hands the request back at once.
   - When the wait expires, reject with `busy` or `no-fit`.
   - An eviction wait is bounded by `switchTimeoutSeconds`. Once that passes,
     the request is admitted anyway and the engine's own memory handling takes
     over.
8. **Admit.** Increment the in-flight count, release it when the response
   finishes, and record the last activity per engine. Report the totals in
   `admission/state`.

### 3.4 Availability (broker orchestration)

`node:set-availability {state:"paused"}` runs these steps:
1. Persist `paused`, then push `node/set-availability {state:"draining"}`. If
   `pause.onActive` is `cancel`, set `cancelActive`.
2. Wait for `admission/state.active == 0`, up to
   `pause.drainTimeoutSeconds`. If the timeout passes, push again with
   `cancelActive`.
3. If `pause.unloadModels` is set, unload every loaded model through
   engine-manager.
4. If `pause.stopEngines` is set, call `engine:sleep` on every running managed
   engine.
5. Push `node/set-availability {state:"paused"}` and notify clients.

Resuming:
- Persist `available`, push it, and send `engine:wake` for engines that were
  put to sleep.

At broker startup:
- A persisted `paused` is pushed to the proxy before any request can be
  admitted.
- If `pause.stopEngines` is set, `engine:restore-enabled` is skipped.

Pausing only affects work executed on this machine. Requests entering this
node are still routed to other machines.

### 3.5 Idle policy and start-on-demand (broker)

- New engine-manager methods change no saved intent:
  - `engine:wake {engine}` starts the engine only if its saved intent is On.
  - `engine:sleep {engine}` stops it, the same way `doStop` does.
- Every 30 s the broker checks `admission/state.lastActivityMs` per engine:
  - If `idle.unloadAfterMinutes` has passed, it unloads that engine's loaded
    models.
  - If `idle.stopEngineAfterMinutes` has passed, it calls `engine:sleep`.
- `admission/wake` maps to `engine:wake`, except while the node is paused.
- An engine saved Off is never started by any of this.

### 3.6 Settings restarts

When `engine:apply-settings` would restart a running engine (`preview.restart`):
1. The broker pushes `node/set-engine-drain {engine, drain:true}`.
2. It waits for that engine's active count to reach 0, up to 60 s.
3. It applies the settings.
4. It clears the drain.

### 3.7 Tiers (proxy, routing side)

A request whose `model` is `weak`, `medium` or `strong` is resolved by the
node it enters.

**Eligible routes.**
- OpenAI-compatible inference routes: `/v1/chat/completions`,
  `/v1/completions`, `/v1/embeddings`.
- Each of these can be served by any engine, so tier candidates may cross
  engines.
- A native route (`/api/chat`, ...) resolves only to candidates of its own
  engine.

**What the request needs.**
- `tools`: a non-empty `tools` or `functions` array.
- `vision`: an `image_url` or `image` content part, or a non-empty Ollama
  `images` array.
- `structured`: `response_format.type` set to `json_schema` or `json_object`,
  or an Ollama `format`.
- `embeddings`: an embeddings route.
- Context: about body bytes / 4, plus `max_tokens`,
  `max_completion_tokens` or `options.num_predict`.

**Candidates.**
- Start with the requested tier's entries in order. Append stronger tiers if
  `allowStronger` is set, then weaker tiers if `allowWeaker` is set.
- Drop entries lacking a needed capability, or whose `contextTokens` (when
  set) is below the estimate.
- Expand each entry to every node whose `modelsByEngine[engine]` contains the
  model. A node whose `loadedByEngine[engine]` contains it is warm.

**Ordering.**
- `preferLoaded: "tier"`: tier rank first, then warm before cold, then
  scheduler priority.
- `"any"`: warm first, then tier rank, then priority.

**Forwarding.**
- Each candidate is a `(node, engine, model)`. A self candidate targets that
  engine's local backend. A peer candidate targets that peer's facade port for
  that engine.
- The body's `model` is rewritten to the concrete model.

**When nothing is eligible.** The response is `404` with
`{"error":"no eligible model for tier <tier>","needs":[...]}`.

**Model lists.** `/v1/models` responses gain one synthetic entry per
configured tier (`owned_by: "pair"`), so clients can pick a tier from their
model list.

### 3.8 Routing metadata

- Every routed inference response carries `X-PAIR-Model`, `X-PAIR-Engine` and
  `X-PAIR-Node`, plus `X-PAIR-Tier` when a tier was requested.
- `Workload.model` becomes the concrete model.
- A new optional `Workload.requestedModel` carries the tier when it differs.

## 4. Jobs: cancel and regenerate (phase 6)

`workloads:cancel {originatedFrom, workloadId, regenerate}` goes to the broker.

**When the job originated on this node**, the broker calls the proxy's
`workload/cancel`:
- **Before the response commits:**
  - With `regenerate`, the current attempt is aborted, its node is excluded,
    and dispatch continues with the remaining candidates.
  - Without `regenerate`, the request ends with `cancelled`.
- **After it commits:** the stream is aborted and the job is `cancelled`. Two
  models' output is never spliced together.

**When the job originated on a peer:**
- The cancel is relayed to the origin through the workload-manager's existing
  pinned-mTLS peer channel, as a new `workloads:cancel` method that peers act
  on only for workloads they originated.

## 5. Model transfer (phase 6)

Engine-manager's control server gains these routes:
- `GET /v1/models/files?engine=&model=` returns the model's files with size
  and sha256. Hashes are computed lazily and cached by path, size and mtime.
- `GET /v1/models/file?engine=&model=&path=` serves one file, with HTTP Range
  support.

The destination's new `engine:remote-copy-model {node, engine, model}`:
- Downloads each file into the engine's model storage, as `.part` files with
  resume.
- Verifies the sha256 of each file before renaming it into place.
- Reports progress as `engine:remote-progress`.

Which engines are supported depends on their storage format:
- **Ollama:** the manifest plus its blobs.
- **llama.cpp:** the cached GGUF files.
- **LM Studio:** the model's directory under its models root.

An engine whose layout cannot be copied safely returns a clear error.

## 6. Manual peers

`nvpair-manual-nodes` persists its node list to `<appdir>/manual-nodes.json`:
- `node/add` saves the entry and `node/remove` deletes it.
- Entries are replayed at startup.
- Addresses may be hostnames, such as Tailscale MagicDNS names. They are
  re-resolved on every probe.

Electron's old `configs/manual-nodes.json` is migrated once:
1. Electron sends each entry to the service as `node/add`.
2. It then deletes the file.
3. The replay code is removed.
