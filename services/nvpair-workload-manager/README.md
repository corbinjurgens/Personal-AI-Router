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
  bounded retry.

Peers are discovered through the broker's node records rather than by this
component browsing the network itself.

## CLI flags

| Flag | Default | Description |
| --- | --- | --- |
| `--port <n>` | `14320` | Inter-node port to listen on and advertise |
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

Each carries `params.workloadInfo`. Removal uses `workloads:remove` with
`params.workloadId` and the origin `params.originatedFrom`.

## Cancelling a job on its origin

Only the node a job entered on can cancel it, because only its proxy holds the
request. When a user cancels a job another node originated, the broker writes
`workloads:cancel` with `{ originatedFrom, workloadId, regenerate?, engine?, runId? }` to this
component's stdin. The manager adds a fresh `cancelId` and posts the cancel over
the same pinned channel: to the origin if it is a current peer, otherwise to
every peer. It is sent directly, not through the ordered lifecycle queue, so a
slow broadcast round cannot hold it up.

A receiving manager rejects a cancel without `originatedFrom` or `workloadId`
(`400`), collapses retried deliveries by `(originatedFrom, workloadId,
cancelId)`, and hands it to its broker as `workloads:cancel`. The broker acts
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
| `requestedModel` | Optional, additive: the tier (`weak`, `medium`, `strong`) a request asked for, when `model` is the concrete model chosen for it. Absent when the client named a model |
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
| `workloads:cancel` | `{ originatedFrom, workloadId, regenerate?, engine?, runId?, cancelId }` — a peer's cancel; the broker acts only if it is the origin |
| `ready` | `{ version }` — startup handshake |

Duplicate events arriving from more than one peer are collapsed before they reach
the broker, so a workload observed over several paths is reported once.

## Performance summaries

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

Each summary POST carries exactly one `X-PAIR-Performance-Elapsed-Ms` header:
canonical nonnegative decimal milliseconds elapsed on the sender's monotonic
clock since it received that revision's broker publication. Identical
same-revision publications preserve this anchor; a new revision resets it.
The sender withholds a publication once its elapsed time exceeds the seven-day
`performancewire.RetentionMs` window. The receiver returns `400` for a missing,
duplicate, noncanonical, negative or out-of-window elapsed header, including
signs, leading zeroes, spaces and alternate number formats. Zero and the exact
retention boundary are valid. The header leaves the shared summary wire schema
and its source ages unchanged.

Performance uses the existing pinned direct/reverse workload service and
`POST /v1/workloads/events`, with `performance:summary` and the shared version 2
summary schema. It has its own connection pool, a latest-summary slot and a
coalesced wake token. A worker round covers all current admitted targets,
using at most four simultaneous requests. No performance frame enters
the lifecycle FIFO or holds its locks. Each peer gets one summary attempt per
round; the next publish, discovery backfill or 30-second heartbeat sends the
newest whole snapshot. Snapshot selection happens after capability negotiation,
immediately before constructing that peer's summary POST. In-flight requests
finish independently; pending intermediate snapshots are replaced. Slow peers
can delay subsequent performance rounds, while the independent timer continues
freshness sweeps and coalesces heartbeat wakes. Lifecycle ordering and direct
cancel delivery retain their existing behavior.

Support is negotiated once per authenticated admission and endpoint generation
with `performance:capabilities`, `params: {}`, on the same endpoint. A compatible
peer replies `200` with exactly `{"version":2}`. Version 1 capability replies
cannot receive summaries; the telemetry deployment updates all machines together
without a dual protocol. An older peer's `400` unknown
method or `501` disables further probes for that route/admission. Other failures,
including malformed or unsupported-version replies, permit at most three probe
attempts, one per round. Negotiation failures produce no repeated warnings or
summary POSTs. A changed route/admission permits a new negotiation. Absence of an
accepted summary remains unknown at the broker, including for unsupported peers.
Negotiation state is retained only for the current peer set; it does not impose
the receiver's 128-record cap on outbound targets.

Inbound summaries require a current pin and durable peer admission, not merely
discovery presence. `measuringHost` must equal the authenticated certificate
principal; the authenticated peer-services contract already binds the host UUID
to that principal. Legacy discovery host aliases cannot assert a measuring
identity, and outbound performance skips aliases whose host ID differs from the
principal. The epoch must equal `Mesh.PeerAdmissionEpoch(principal)`. The manager
uses `Mesh.PeerServiceIncarnation` and repeats pin checks around processing and
broker emission. Forwarded summaries are rejected and accepted summaries are
never gossiped. The receiver rejects unknown fields, malformed/trailing data and
params over 128 KiB via `performancewire.DecodeSummary`; the outer HTTP frame is
limited to 1 MiB with explicit overflow detection.

The stdout contracts, intended for the broker's subsequent performance handler,
are:

| Notification | Exact params |
| --- | --- |
| `performance:received` | `{ summary: Summary, admissionIdentity: string, sampleElapsedMs: int64 }` — full replacement for a higher sender revision |
| `performance:status` | `{ measuringHost: string, admissionIncarnation: string, admissionIdentity: string, status: "fresh" \| "stale" \| "unknown", revision: int64, sampleElapsedMs: int64, lastSeenAgeMs: int64 }` |

`admissionIdentity` is the receiver-local opaque Mesh hash, not a wire epoch.
The broker must revalidate it against current shared admission authority before
consuming either notification. An authority change during a pipe write returns
`403` and emits an `unknown` status for the just-written identity; a completed
write cannot be retracted. Failed broker writes return `500`, keeping a new
revision retryable; if only its status write failed, retrying emits status without
re-emitting the already accepted replacement.

A lower revision is ignored without refreshing reachability. An identical
same-revision heartbeat emits freshness only; a changed same-revision summary
returns `400`. Only a newly verified sender epoch resets the revision floor.
Receiver-local admission/certificate changes retire cached payload bytes while
preserving that sender epoch's revision floor and content digest, including
across pin loss. A missing local pin cannot prove a new sender epoch. Receiver
state is bounded to 128 current or retired peer records; additional records
receive `503` until the process restarts. This manager owns no durable history.

An accepted summary is fresh until 90 seconds of receiver-monotonic silence;
the independent 30-second sweep emits `stale` at the first sweep at or after that
boundary. A verified heartbeat refreshes reachability to `fresh` and resets
`lastSeenAgeMs`, while `sampleElapsedMs` starts with the sender's elapsed header
and advances using receiver-monotonic time from this revision's first accepted
receipt. A duplicate can raise the elapsed offset but cannot make samples
younger. Both notifications carry this elapsed metadata; the broker adds it to
source entry ages. Heartbeats never rewrite source ages or observation
timestamps. Authority loss emits `unknown` and retires payloads; failed status writes retry on a later
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
