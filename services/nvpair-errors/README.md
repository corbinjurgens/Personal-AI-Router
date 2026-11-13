<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvpair-errors

> **Status: cross-node.** This subprocess owns the per-node in-memory
> list of `ServiceError`s and broadcasts changes to the broker. The
> wire protocol matches the UI side's spec
> (`errors:get-initial` / `errors:clear` / `errors:update`); the
> internal request shapes between the broker and nvpair-errors mirror
> them so the broker can be lifted out into a dedicated `nvpair-broker`
> subprocess later without changing nvpair-errors.
>
> **Cross-node sync is implemented (`--peer-sync`).** Every node sees
> every other node's errors. nvpair-errors runs no mDNS itself — it
> discovers peers through the broker's discovery relay (the broker
> registers its `er` port with the `nvpair-node-scanner` daemon on its
> behalf, and it subscribes for `er` nodes), streams its local-origin
> errors to those peers over pin-based cluster mTLS, and subscribes to
> theirs. See
> [Cross-node propagation](#cross-node-propagation) for the full
> design. Without the flag, nvpair-errors is a stdio-only local
> datastore with no network surface at all.
>
> **Severity / action values are placeholders.** The canonical enum
> value set is not finalized yet; producers ship with placeholder
> values (`info`/`warning`/`error` and `dismiss`/`retry`/`none`).
> Replacing them is a one-line change in each producer — no wire-shape
> change.

## Purpose

A typed in-memory error registry for the local node. Producers
(subprocesses or broker-side handlers) report errors as JSON-RPC
notifications/requests; consumers (the Debug panel today, a future
header badge) read the list via `errors:get-initial` and subscribe to
`errors:update` push notifications for live changes.

Pure datastore. No business logic, no severity inference, no
escalation. The producer is the authority on what each error means;
nvpair-errors just stores and broadcasts.

## Communication

Two channels:

- **Broker channel (always on):** bidirectional newline-delimited
  JSON-RPC 2.0. Stdio by default; `--ipc <path>` switches to a Unix
  domain socket or Windows named pipe (same flag shape as every other
  NVPAIR subprocess). This is how local producers' reports reach the store
  and how `errors:update` reaches the UI.
- **Peer channel (`--peer-sync` only):** cross-node sync over pin-based
  cluster mTLS, always. Peers are discovered through the broker's
  discovery relay (not a self-run mDNS browse), and only a paired cluster
  member is served or subscribed to (`403` otherwise). A node that belongs to
  no cluster exchanges nothing on this channel — there is no plain-HTTP
  personality. Membership comes from `--cluster-dir` and is re-read
  continuously, so a node that joins or leaves converges without a
  restart. See [Cross-node propagation](#cross-node-propagation).

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--ipc <path>` | _(stdio)_ | IPC endpoint: Unix socket or Windows named pipe |
| `--peer-sync` | _(off)_ | Enable cross-node sync: serve this node's error stream and subscribe to the streams of peers discovered via the broker's discovery relay |
| `--port <n>` | `14319` | HTTP port for the cross-node errors endpoint (only used with `--peer-sync`) |
| `--cluster-dir <path>` | _(off)_ | Cluster config dir (the `.../cluster` dir with `node.crt`/`node.key` + `trusted/*.json`); when set, peer-sync runs over pin-based mTLS scoped to paired cluster members |
| `--node-id <uuid>` | _(resolved locally)_ | This node's origin id, stamped as `nodeId` on local-origin entries and sync snapshots. The broker passes its resolved UUID; an explicit value takes precedence. When omitted (with or without `--peer-sync`), the binary resolves the node UUID with `nodeid.Resolve` from the parent of `--cluster-dir` (or the shared default config root): `cluster/identity.json`, then `node-id.json`, otherwise mint and persist `node-id.json`. The broker always passes `--node-id` |
| `--log-level <level>` | `info` | One of `error`/`warn`/`info`/`debug`; falls back to `$NVPAIR_LOG_LEVEL` |
| `--version` | | Print version and exit |

## ServiceError shape

```ts
{
  id:         string   // required, stable, see "ID namespace convention"
  message:    string   // required, user-facing
  timestamp:  number   // required, Unix millis; upsert tie-break
  nodeId?:    string   // ORIGIN node id — see "Cross-node propagation"
  severity?:  string   // "info" | "warning" | "error" (placeholder; see status block)
  action?:    string   // "dismiss" | "retry" | "none" (placeholder; see status block)
  engineType?: string  // set by nvpair-engine-manager on its engine-manager:* ids
  operation?:  string  // set where meaningful (pull, install)
  modelName?:  string  // set for engine-manager:pull-failed:<engine>:<model>
}
```

## ID namespace convention

`<producer-shortname>:<error-class>[:<context>]`. Convention only —
nvpair-errors keys by composite `(nodeId, id)` and
**highest-timestamp-wins on upsert**. A later report with the same id
but an OLDER timestamp is dropped.
Equal-timestamp reports overwrite (a producer re-emitting a
steady-state error refreshes the message).

Starter ids:

| id | emitter | meaning |
|---|---|---|
| `llamacpp-proxy:upstream-unreachable:<node-id>` | llamacpp-proxy | an upstream node dropped out of discovery |
| `node-scanner:node-info-unreachable:<host-uuid>` | nvpair-node-scanner | a paired peer's node-info stopped answering, or its reverse-session claims lapsed or never arrived after scanner start; cleared when it answers or renews, stops being paired, or is evicted |
| `manual-nodes:probe-failed:<node-id>` | nvpair-manual-nodes | a manual node has failed N consecutive probes |
| `engine-manager:install-failed:<engine>` | nvpair-engine-manager | installing the engine failed |
| `engine-manager:pull-failed:<engine>:<model>` | nvpair-engine-manager | pulling a model for the engine failed |
| `engine-manager:start-failed:<engine>` | nvpair-engine-manager | starting the engine failed |
| `supervisor:subprocess-crashed:<subprocess-name>` | broker | any spawned subprocess exited unexpectedly |

## JSON-RPC methods (broker → nvpair-errors)

### `errors:get-initial` (request)

Returns the current full list, sorted by id.

```json
// → {"jsonrpc":"2.0","id":1,"method":"errors:get-initial"}
// ← {"jsonrpc":"2.0","id":1,"result":[
//      {"id":"engine-manager:start-failed:llamacpp","message":"...","timestamp":1700000000000,"nodeId":"node-a"}
//   ]}
```

### `errors:report` (request OR notification)

Upserts the ServiceError by id. Highest-timestamp-wins; equal
timestamps overwrite. Returns `null` for the request form; the
notification form is fire-and-forget.

```json
// notification form (most producers use this):
{"jsonrpc":"2.0","method":"errors:report","params":{
  "id":"llamacpp-proxy:upstream-unreachable:peer-b",
  "message":"peer-b unreachable for 5 ticks",
  "timestamp":1700000123456,
  "nodeId":"node-a"
}}
```

### `errors:clear` (request OR notification)

Removes the entry with the given id. No-op (no `errors:update` push)
if the id is not present. ClearedBy is broker-stamped and ignored by
nvpair-errors today; see [Origin tracking and loop prevention via `nodeId`](#origin-tracking-and-loop-prevention-via-nodeid).

```json
// notification form:
{"jsonrpc":"2.0","method":"errors:clear","params":{
  "id":"llamacpp-proxy:upstream-unreachable:peer-b",
  "clearedBy":"node-a"
}}
```

### `shutdown` (request)

Graceful shutdown. Response is sent before `cancel()` fires.

### `log/set-level` (request OR notification)

Standard. Params: `{"level":"<error|warn|info|debug>"}`.

## JSON-RPC notifications (nvpair-errors → broker)

### `ready`

Emitted once on startup with the binary's version.

```json
{"jsonrpc":"2.0","method":"ready","params":{"version":"0.1.0"}}
```

### `errors:update`

Emitted on every state change (a successful `errors:report` upsert or
a successful `errors:clear`). Carries the CURRENT FULL list, sorted by
id, so consumers can render-on-event without re-fetching.

Importantly: dropped reports (older-timestamp races) and no-op clears
(id not present) do NOT emit `errors:update`. The UI would re-render
on every dropped frame otherwise.

```json
{"jsonrpc":"2.0","method":"errors:update","params":[
  {"id":"engine-manager:start-failed:llamacpp","message":"...","timestamp":...,"nodeId":"node-a"},
  {"id":"supervisor:subprocess-crashed:nvpair-node-info","message":"...","timestamp":...,"nodeId":"node-a"}
]}
```

## How a future producer wires in

Two notification emit points on the producer's existing stdio. No new
flags, no new sockets, no new client libraries.

```go
// On failure:
codec.Notify("errors:report", ServiceError{
    ID:        "my-producer:my-error-class:context",
    Message:   "user-facing description",
    Timestamp: time.Now().UnixMilli(),
    NodeID:    localNodeID,   // origin; see "Cross-node propagation"
})

// On subsequent success (where applicable):
codec.Notify("errors:clear", map[string]string{
    "id": "my-producer:my-error-class:context",
})
```

The broker (`nvpair-ui-broker`) demuxes
`errors:report` and `errors:clear` notifications from any subprocess's
stdio stream and forwards them to nvpair-errors. The producer never talks
to nvpair-errors directly.

## Cross-node propagation

With `--peer-sync`, nvpair-errors propagates errors peer-to-peer so every
node's UI shows the whole network's errors. The design is a **snapshot
stream**: each node streams its full local-origin list to every pinned
peer that subscribes, and the subscriber reconciles that list
authoritatively for the peer's `nodeId`. Peer discovery comes from the broker's discovery relay:
nvpair-errors subscribes for `er` nodes and consumes the node
snapshots the broker pushes down.

```
producer --errors:report--> nvpair-errors A <--GET /v1/errors/events-- nvpair-errors B --errors:update--> UI B
                            (serves its snapshot)   (stream frames)    (subscribes, reconciles)
```

### Discovery (via the relay)

- nvpair-errors runs no mDNS itself. It sends `discovery:subscribe`
  with `{"services":["er"]}`, and the broker relays back
  `discovery:nodes` snapshots — the full filtered `er` set, re-sent on
  every change — which nvpair-errors diffs into peer events. The broker
  registers its `er` port with the `nvpair-node-scanner` discovery
  daemon on its behalf, so it rides this node's one `_nvpair-node`
  record.
- A peer's directory entry maps directly to an origin `nodeId` (the
  peer's stable `hostUuid`), so no per-service TXT lookup is
  needed. Self is ignored.

### Transport (cluster mTLS, unconditionally)

`GET /v1/errors/events` — stream this node's local-origin snapshot as
NDJSON: one `SyncEnvelope` (`{"nodeId": "...", "errors": [...]}`) per
line, sent in full at once on connect and again after every local
change, with a `null` keepalive line every 10 s. `GET /v1/errors` —
return the same snapshot once, as a local probe (handy for manual
verification). Neither route accepts a POST: peers no longer push
snapshots, and any method other than GET answers `405`.

Both routes require **pin-based cluster mTLS**. The caller must present
a certificate this node currently pins; anything else is refused with
`403`, and the stream re-checks the pin before every line, so removing a
peer's pin ends its open stream. There is no plain-HTTP personality:
this is a cluster data plane, so a node that belongs to no cluster
refuses every handshake and serves nothing here.

The port stays bound either way. The server certificate is resolved per
handshake from `--cluster-dir`, so a node that joins a cluster starts
serving on the same listener, and one that leaves stops — neither needs a
restart.

### Subscriptions

- A discovered or updated pinned peer gets one stream. An event that
  changes nothing leaves it running; a changed address list or
  principal restarts it. A peer removed by discovery has its stream
  cancelled and its entries evicted.
- Each stream picks a reachable address from the peer's published
  list, connects directly or over a reverse session, and applies every
  frame through the same reconcile. A frame whose `nodeId` is not the
  peer's pinned principal ends the stream and changes nothing.
- A stream ends when no line arrives for 25 s, when the peer's
  admission, certificate or endpoint changes, or when the connection
  drops. It reconnects after 1 s, doubling to 30 s with 20 % jitter,
  and the delay resets after a stream that delivered a frame. When this
  node joins or leaves a cluster, every discovered peer is checked
  against the pins again and every admitted peer's stream restarts, so
  peers discovered before the join are picked up. A peer paired, or
  unpaired, while this node is already a member is gated again when the
  relay reports its changed `trusted` annotation: the cluster-manager's
  `cluster:trust-changed` makes the scanner re-derive it after the pin
  is written, so its stream starts (or stops) without a membership
  change. A joiner that does not yet advertise its cluster principal
  is picked up when it starts to, which changes its record. An unpaired
  peer keeps its entries until discovery removes it.
- A peer's entries stay when its stream ends, as they always did when
  updates stopped; only discovery removal evicts them. Every reconnect
  starts with the peer's full snapshot, so a change missed while a
  stream was down cannot persist, and nothing is resent periodically.

### Reconcile = replace-by-origin (handles clears + initial sync)

Because each frame carries the peer's **complete** local-origin set,
the subscriber treats it as authoritative for that `nodeId`: upsert every
entry present, evict any stored entry for the same `nodeId` that is
absent. A cleared error simply drops out of the next frame and the
subscriber removes it; a late-joining node is fully synced by the first
frame of its stream.
Report, clear, and initial-sync therefore collapse into one idempotent
operation — no separate clear or sync endpoint.

### Origin tracking and loop prevention via `nodeId`

`nodeId` is the ORIGIN node id. Producers set it (or the broker does,
via `ReportError`); nvpair-errors preserves it verbatim. A node only ever
streams entries whose `nodeId == local` — foreign entries learned from a
peer are stored but never re-served, so they can't bounce. The subscriber
stamps the envelope's `nodeId` onto every reconciled entry, so a buggy or
hostile peer can only mutate its own slice of the keyspace, never a
third node's or ours.

`errors:clear` carries an optional `clearedBy` field the broker stamps
with the local node id. Clear is scoped to `(local-node-id, id)`: a
user clears errors their own node owns; a peer's error clears when the
peer recovers and drops it from its stream.

### Conflict resolution: highest-timestamp-wins

Upsert and reconcile both use **highest `timestamp` wins** rather than
arrival order. Under cross-node delivery, out-of-order arrival from
different network paths can flip "last arrival" without flipping "most
recent emit" — the timestamp rule keeps state convergent. Equal
timestamps overwrite (a producer re-emitting a steady-state error
refreshes the message).

### Keying

The store keys by composite `(nodeId, id)` because ids are only unique
within a node (two nodes both emit `engine-manager:start-failed:llamacpp`). The
on-wire shapes are unchanged — `errors:update` / `errors:get-initial`
still carry a flat `[]ServiceError`, sorted by `id` then `nodeId`.

### What does NOT change

nvpair-errors's broker (stdio) wire protocol. The `ServiceError` shape.
Producer call sites. The broker ↔ UI bindings. Enabling
`--peer-sync` only adds the peer channel; the local path is identical.

## Validation errors

- Missing required field (`id` or `message` on `errors:report`,
  `id` on `errors:clear`): `-32602`.
- Malformed params JSON: `-32602`.
- Unknown method: `-32601`.

Notification-form messages with the same problems are logged
(`slog.Warn`) and dropped; no response or `errors:update` push fires.

## Shutdown

The manager shuts down on:

- stdin EOF (parent closed the pipe).
- `SIGINT` / `SIGTERM`.
- `shutdown` JSON-RPC request.
