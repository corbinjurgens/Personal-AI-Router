<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvpair-workload-manager

## Purpose

Propagates inference-workload lifecycle events across the cluster, so every node
has the same view of what work is queued, running, or finished — and on which
node. That shared view is what makes the Jobs list meaningful on any member and
what supplies the pending-work side of `nvpair-job-scheduler`'s GPU-aware
ranking.

The manager is a **relay and deduplicator**, not the source of truth. The broker
and proxies decide a workload's state; this component passes `workloadInfo`
through opaquely and forwards it.

## Communication

Two channels:

- **Local** — bidirectional newline-delimited JSON-RPC 2.0 with the supervising
  `nvpair-ui-broker`. Stdio by default; `--ipc <path>` switches to a Unix domain
  socket or Windows named pipe.
- **Inter-node** — a cluster-mTLS server on port `14320`
  (`POST /v1/workloads/events`) that accepts the same JSON-RPC frames from pinned
  peers, plus an outbound broadcaster that pushes local events to those peers with
  bounded retry. The same server streams this node's performance summary to
  pinned peers (`GET /v1/workloads/performance/events`), and the manager reads
  each peer's stream.

Peers are discovered through the broker's node records rather than by this
component browsing the network itself.

Each outbound route uses the peer's advertised `wl` port independently of this
node's listener. Missing, zero or out-of-range advertised ports are ignored,
without substituting this node's port or the production default.

## CLI flags

| Flag | Default | Description |
| --- | --- | --- |
| `--port <n>` | `14320` | Inter-node port to listen on and advertise (1 to 65535; zero is rejected before startup) |
| `--ipc <path>` | _(stdio)_ | IPC endpoint: Unix socket or Windows named pipe |
| `--cluster-dir <path>` | _(none)_ | Cluster config dir (`node.crt` / `node.key` + `trusted/`) supplying the identity and pins the inter-node mTLS channel requires. Without it the node has no cluster identity and exchanges no inter-node traffic |
| `--log-level <level>` | `info` | One of `error` / `warn` / `info` / `debug` |
| `--version` | | Print version and exit |

## Transport security

The inter-node interface is **always cluster mTLS**. There is no plaintext
personality on it. The listener presents this node's cluster certificate,
requires a client certificate, and refuses any caller that is not a pinned
cluster peer with `403`. Pins are re-read per request, so removing a member takes
effect immediately without a restart.

A node that is not a current cluster member has no identity to present, so it
neither serves nor broadcasts inter-node workload traffic at all. The port stays
bound: the certificate is resolved per handshake, so a node converges to serving
after it joins without a rebind or a restart.

Workload state is therefore only ever exchanged between paired members. See the
repository [`SECURITY.md`](../../SECURITY.md) for the surrounding trust
boundaries.

## Lifecycle events

Inbound lifecycle notifications, on either channel:

| Method | Resulting state |
| --- | --- |
| `workload:submitted` | `queued` |
| `workload:started` | `running` |
| `workload:completed` | `completed` |
| `workload:errored` | `failed` |

The broker may synthesize `workload:errored` with state `failed` and error
`proxy exited before the request finished` to retire open or inferred local-origin
outcomes of a departed proxy run; once received, the manager broadcasts it like
any local terminal and retains it for two heartbeat intervals for best-effort
re-assertion.

All proxy lifecycle frames, as well as broker-synthesized retirement terminals,
reach this manager only after the broker's store accepts them.

Each carries `params.workloadInfo`. Removal uses `workloads:remove` with
`params.workloadId` and the origin `params.originatedFrom`.
The re-sync set identifies a workload by `(originatedFrom, engine, runId, id)`;
removal carries only origin and ID and untracks all matching identities.
Inbound lifecycle dedup adds `state`, `scheduledOn`, and `seq` to that identity;
removal dedup uses origin and workload ID.

## Cancelling a job on its origin

Only the node a job entered on can cancel it, because only its proxy holds the
request. When a user cancels a job another node originated, the broker writes
`workloads:cancel` with `{ originatedFrom, workloadId, regenerate?, engine, runId }` to this
component's stdin. The manager adds a fresh `cancelId` and posts the cancel over
the same pinned channel: to the origin if it is a current peer, otherwise to
every peer. It is sent directly, not through the ordered lifecycle queue, so a
slow broadcast round cannot hold it up.

A receiving manager validates the workload identity and `cancelId` (`400` on
invalid input), collapses retried deliveries by `(originatedFrom, engine, runId, workloadId, cancelId)`,
and hands it to its broker as `workloads:cancel`. The broker acts
only if it is the origin.

## Workload shape

Defined in [`workload.go`](workload.go):

| Field | Notes |
| --- | --- |
| `id` | Stable workload identifier |
| `model`, `engine` | What was requested and by which engine |
| `runId` | Producing process's nonce, minted at proxy startup. Part of the dedup key: `id` is a per-process counter that both engine proxies start at 1 and that resets on restart, so `runId` keeps a reused id from colliding with an older workload |
| `state` | `queued`, `running`, `completed`, `failed`, or `cancelled` |
| `originatedFrom` | Node the request entered the cluster on |
| `scheduledOn` | Node it was routed to; absent until a target is chosen |
| `createdAt`, `startedAt`, `completedAt` | Epoch milliseconds; the last two are nullable |
| `error` | Normalized failure text, nullable |
| `requesterId` | Optional client attribution, nullable |
| `requestedModel` | Optional, additive: the user-created role a request asked for, when `model` is the concrete model chosen for it. Absent when the client named a concrete model. Passed through opaquely; not part of the dedup key |
| `seq` | Producer's event counter, from 1. Part of the dedup key, so a workload that returns to a placement it already had is not mistaken for a redelivery |

Optional and nullable fields use pointers so a peer's payload round-trips without
inventing zero values.

Prompts, messages, and response bodies are **not** part of this contract and must
never be added to it.

## Output to the broker

Translated remote events are forwarded to the broker as:

| Notification | Params |
| --- | --- |
| `workloads:upsert` | `{ workloadInfo }` |
| `workloads:remove` | `{ workloadId, originatedFrom }` |
| `workloads:cancel` | `{ originatedFrom, workloadId, regenerate?, engine, runId, cancelId }` — a peer's cancel; the broker acts only if it is the origin |
| `ready` | `{ version }` — startup handshake |

Duplicate events arriving from more than one peer are collapsed before they reach
the broker, so a workload observed over several paths is reported once.

## Performance summaries

Summary entries carry the destination's bounded optional measurement `provenance`
(`configuration:{digest,complete}`, `engineFamily`, `api`, `extractorSchema`) and
`applicability` (`current`, `historical`, `unknown`). Both local publish and peer
receipt use the shared strict decoder. Missing old provenance remains unknown;
malformed or private metadata is refused. The relay preserves the destination's
attestation and its revision ordering, without recomputing configuration identity
or changing sample ages. This extends the existing summary format in place.

The broker can publish its latest complete own `performancewire.Summary` on
stdin as `performance:publish`, with the summary directly in `params`. An
optional JSON-RPC request receives `null` on acceptance or `-32602` for invalid
data. A notification has no reply. This manager validates the measuring host
against both its configured host UUID and current certificate principal, and
requires `admissionIncarnation` to be the canonical positive decimal value of
`Mesh.LocalAdmissionEpoch()`. It accepts increasing revisions and identical
retransmissions; changing a revision's contents is invalid. Empty `entries: []`
is a complete replacement, including removal of the previous entries.
The own-summary slot belongs to the broker's persisted host/epoch revision;
same-epoch certificate replacement preserves it while each actual transport
dispatch is fenced by the current shared Mesh admission identity.

Performance summaries travel on `GET /v1/workloads/performance/events`, an
NDJSON snapshot stream (`clustertrust.ServeSnapshotStream`) on the workload
service, behind the same pin gate as the events endpoint, rechecked before
every line. Each frame is `{"summary": Summary, "elapsedMs": int64}`: the full
latest local summary, sent at once on connect and again whenever the broker
republishes. `elapsedMs` is the milliseconds elapsed on the sender's monotonic
clock since it received that revision's broker publication, computed when the
line is written. Identical same-revision publications preserve this anchor; a
new revision resets it. The sender withholds a summary whose local admission
changed or whose elapsed time exceeds the seven-day `performancewire.RetentionMs`
window: it writes no frame. Before the stream sent any frame it stays open;
after one, the sender ends it at the next line while the summary is withheld,
so the receiver's stale rule applies (keepalives would otherwise keep the
earlier frame fresh). A `null` keepalive follows every 10 seconds and asserts
nothing about the summary. The receiver
refuses a frame with a missing, negative, non-integer or out-of-window
`elapsedMs`, unknown fields or trailing data; zero and the exact retention
boundary are valid. The frame leaves the shared summary wire schema and its
source ages unchanged.

Each manager subscribes to every current target whose host ID equals its
cluster principal and that has a current peer admission, one stream per peer,
up to 128. Subscriptions are re-evaluated on discovery and membership changes
and with the 30-second sweep. A stream uses its own `PeerServiceClient` on the
pinned direct/reverse workload service with no total timeout; connect and
response headers are bounded to 10 seconds, and a stream with no line for 25
seconds is dropped. Reconnects back off from 1 second, doubling to 30 seconds,
and start again from the full summary. A stream is bound to the peer's admission
identity and route generation at connect; either changing ends it. A stream
that ends on a refused frame keeps backing off. Streams share only the peer set
and address chooser with the lifecycle sender, and no locks, queue or
connection slots with the lifecycle FIFO or direct cancel delivery, whose
ordering and behaviour are unchanged. The former summary
POST and its capability probe are gone; every machine runs
the same build, so there is no version probe.

Inbound summaries require a current pin and durable peer admission, not merely
discovery presence. The stream's principal is the pinned peer it was opened to,
whose server certificate the service client verified. `measuringHost` must equal
that principal; the authenticated peer-services contract already binds the host
UUID to it. Legacy discovery host aliases cannot assert a measuring identity, and
no subscription is opened for an alias whose host ID differs from the principal.
The epoch must equal `Mesh.PeerAdmissionEpoch(principal)`. The manager
uses `Mesh.PeerServiceIncarnation` and repeats pin checks around processing and
broker emission. Forwarded summaries are rejected and accepted summaries are
never gossiped. The receiver rejects unknown fields, malformed/trailing data and
params over 128 KiB via `performancewire.DecodeSummary`; a stream line is limited
to 128 KiB plus 1 KiB. Any refused frame ends the stream, and the reconnect
re-sends the full summary.

The stdout contracts, intended for the broker's subsequent performance handler,
are:

| Notification | Exact params |
| --- | --- |
| `performance:received` | `{ summary: Summary, admissionIdentity: string, sampleElapsedMs: int64 }` — full replacement for a higher sender revision |
| `performance:status` | `{ measuringHost: string, admissionIncarnation: string, admissionIdentity: string, status: "fresh" \| "stale" \| "unknown", revision: int64, sampleElapsedMs: int64, lastSeenAgeMs: int64 }` |

`admissionIdentity` is the receiver-local opaque Mesh hash, not a wire epoch.
The broker must revalidate it against current shared admission authority before
consuming either notification. An authority change during a pipe write refuses
the frame and emits an `unknown` status for the just-written identity; a
completed write cannot be retracted. A failed broker write refuses the frame,
keeping a new revision retryable on the reconnect; if only its status write
failed, the retry emits status without re-emitting the already accepted
replacement.

A lower revision is ignored without refreshing reachability. An identical
same-revision frame emits freshness only; a changed same-revision summary is
refused. Only a newly verified sender epoch resets the revision floor.
Receiver-local admission/certificate changes retire cached payload bytes while
preserving that sender epoch's revision floor and content digest, including
across pin loss. A missing local pin cannot prove a new sender epoch. Receiver
state is bounded to 128 current or retired peer records; additional records
are refused until the process restarts. This manager owns no durable history.

An accepted summary is fresh until 90 seconds of receiver-monotonic silence;
the independent 30-second sweep emits `stale` at the first sweep at or after that
boundary. After a stream's frame is accepted, each keepalive on that stream
refreshes `lastSeen` while the peer's admission still holds, and re-asserts
`fresh` at most every 30 seconds, the cadence the broker saw from the former
heartbeat. An ignored older-revision frame does not count as accepted. So a
peer goes `stale` 90 seconds after its stream drops, including a stream the
sender ended because its summary became withheld, not while it is live.
`sampleElapsedMs` starts with the frame's `elapsedMs` and advances using
receiver-monotonic time from this revision's first accepted receipt. A duplicate
can raise the elapsed offset but cannot make samples younger. Both notifications
carry this elapsed metadata; the broker adds it to source entry ages. Keepalives
never rewrite source ages or observation timestamps. Authority loss emits `unknown` and retires payloads; failed status writes retry on a later
sweep. No aggregation, persistence, scheduling or UI projection is implemented
here. T3b must stamp the canonical current local epoch, publish full snapshots,
consume these notifications and apply their admission/freshness fences. On a
broker or manager restart, it must also rebase saved source ages by elapsed time
since persistence, using monotonic time when available and otherwise wall-clock
time clamped to zero or greater. Restart must not rejuvenate persisted samples.

## Testing

```bash
go test ./...
```

## See also

- [`../nvpair-ui-broker/README.md`](../nvpair-ui-broker/README.md) — the
  supervisor and relay
- [`../nvpair-job-scheduler/README.md`](../nvpair-job-scheduler/README.md) — the
  primary consumer of workload counts
- [`../VERSIONING.md`](../VERSIONING.md) — SemVer bump rules
