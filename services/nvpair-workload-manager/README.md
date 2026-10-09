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
`workloads:cancel` with `{ originatedFrom, workloadId, regenerate? }` to this
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
| `workloads:cancel` | `{ originatedFrom, workloadId, regenerate?, cancelId }` — a peer's cancel; the broker acts only if it is the origin |
| `ready` | `{ version }` — startup handshake |

Duplicate events arriving from more than one peer are collapsed before they reach
the broker, so a workload observed over several paths is reported once.

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
