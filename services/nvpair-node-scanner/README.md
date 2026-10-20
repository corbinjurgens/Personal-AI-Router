<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvpair-node-scanner

**The consolidated discovery daemon.** Promoted from a plain node-info browser: it is now the single place this host advertises itself on and discovers the LAN through. Every other service registers its port with this daemon instead of advertising its own mDNS service, and consumers query/subscribe to it instead of browsing themselves.

Concretely, the daemon:

- **Advertises** this node's one `_nvpair-node._tcp` record, built from a registry of the local services its parent (the broker) registers — `ni`/`ol`/`lm`/`er`/`wl`/`cl`/`em`/`ec` ports plus the node identity (`uuid=`, `cluster-uuid=` when clustered, `ip=`). One record per node covers every service.
- **Browses** `_nvpair-node._tcp` for every node on the LAN (including itself) and maintains a queryable directory keyed by host UUID.
- **Bootstraps admitted peers without mDNS:** cluster-manager reads the peer's
  complete self-description through its pinned direct/reverse `cl` route. The
  broker privately relays `discovery:peer-services` notifications to the scanner,
  which checks the original shared Mesh admission incarnation and installs the
  services through the same directory, enrichment and discovery events. This
  notification is worker IPC, not a public client method. Descriptors carry only
  identity, name, public candidate addresses and allowlisted service ports;
  trust, transport policy, membership and liveness remain locally derived.
  Authenticated snapshots take precedence over browse claims, including complete
  service withdrawal. Browse updates and removals affect only their own claim.
  Leave, removal, certificate/admission replacement and same-certificate re-pair
  retire authenticated metadata on trust reload; a one-second reconciliation also
  expires observations after 30 seconds. A surviving browse claim remains visible.
  Sequence watermarks are scoped to the cluster-manager process source, and stale
  responses and delayed enrichment cannot replace a newer snapshot. Endpoint or
  incarnation changes clear cached enrichment and advance the route generation.
  Independently observed inference activity survives a source change; final
  removal clears it only when neither claim remains.
- **Enriches** each discovered node: GPU/CPU/memory inventory from its `ni` (node-info) port over plain HTTP, and the model list from its `em` (engine-manager) port (`GET /v1/models`) over cluster mTLS — a peer's model inventory is cluster data, so it is fetched only when this node holds a pin for the principal that peer advertises, while this node's own list is read over loopback in plaintext — the flat union, the per-engine breakdown (`modelsByEngine`), and the per-engine set of models loaded in memory (`loadedByEngine`), all carried onto the node's directory entry so remote cards can show loaded state. The same document carries the node's policy advertisement (`policy`: declared roles, policy revision, pause origin), which becomes the entry's `policy`; a peer known only from its authenticated peer-services claim is enriched through the same fetch. A node whose answer has no `policy` declares nothing. The policy's strict admission capability (`strictAdmission`) only affects hybrid eligibility, so a malformed one is dropped while the advertisement and model list are kept, and a case-variant `strictAdmission` key drops the advertisement but keeps the model list. Each enrichment has a last-good cache so a transient fetch miss doesn't blank the node's card. Node-info enrichment can optionally be moved onto HTTPS (see the TLS flags below) — off by default and gated only on operator flags, never inferred per-node.
- **Samples scheduling telemetry** from healthy nodes every two seconds with
  stable per-node jitter. Consecutive remote failures back off to 4, 8, 16, then
  30 seconds and reset on success; local loopback sampling stays at the healthy
  cadence. It emits only host UUID, maximum GPU utilization, validity, and
  source-reported sample age; it does not mutate the public directory for each
  telemetry sample.
- **Annotates trust:** with `--cluster-dir`, a browsed peer whose `cluster-uuid=` this node holds a pin for is marked `trusted` (via `nvpair-shared/clustertrust`). The annotation is re-derived on `discovery:reload-trust`, which the broker sends when `nvpair-cluster-manager` reports a pin-set change, as well as when a peer's mDNS record changes. Both triggers are needed: a peer that was already advertising when this node joined never moves its record again, so the browse path alone would leave it annotated from before this node's own pins existed. A change emits one `discovery:node-updated`; a reload that finds nothing to change is silent.
- **Advertises live membership:** the same reload re-derives this node's own `cluster-uuid=` from the cluster dir and re-advertises when it differs from what is being published, so joining or leaving a cluster takes effect without a restart.
- **Anti-flap:** a node has to be absent from every scan for the browser's full miss threshold (twelve scans at 5s, a full minute) before it is evicted, and from ~15s in it is TCP-probed on its advertised service ports on *every* scan — any answer resets the counter and recovers it outright. So neither a momentary multicast gap nor a sustained one drops a healthy node, and eviction rests on a run of ~10 consecutive failures rather than one dial on the deadline. The window is sized deliberately: a node saturated by its own inference load starves its control plane, so it stops answering mDNS *and* stops accepting probe connections for as long as the load runs, while still serving requests. Two signals can also vouch for a node without any probe — a node-info enrichment in the last 10s, and inference response bytes from it in the last 60s (`discovery:node-activity`, relayed by the broker from whichever proxy saw the bytes). The second is the only liveness evidence that gets stronger the busier a node is.
- **A scan that loses everything is treated as our own fault:** when more than one node is known and a scan returns *none* of them, no node is penalized for it — for up to six consecutive scans (~30s), after which they count normally. Independent machines do not leave the network in the same five-second window, but a process too starved to drain its multicast socket within `scanTimeout` hears silence from all of them at once; without this, one saturated machine dropped its entire cluster, idle peers included, in a single reconcile. The suppression needs two or more known nodes to engage: losing a lone peer looks the same whether it left or we stopped listening, so that case falls through to the ordinary threshold.
- **Converges without mDNS:** a 15s sweep re-reads each peer over HTTP and folds any change into the directory, independent of browse events. It covers facts that change without moving the mDNS record (a model pull/delete, a late engine start) and facts that did move it but whose announcement was never received. The second case is why a peer's cluster membership is re-read from its `ni` port (`clusterUuid` on `/v1/node-info`, which under the broker is served plain — so it answers whether or not the two nodes share a cluster; a node-info run standalone with `--cluster-dir` gates the endpoint while clustered and will not answer a non-pinned peer, which the sweep treats as no answer): membership otherwise arrives solely as the `cluster-uuid=` TXT key, read once per record change, and the anti-flap probe above deliberately keeps a still-reachable peer's entry alive — so a missed re-advertisement would leave a departed peer's principal in place for the life of the process, suppressing the invite that would bring it back. Membership is refreshed before models on each tick, since the model fetch is pinned mTLS keyed on that principal. Only a peer's explicit report is acted on: a failed fetch, or a peer too old to report the field, leaves the annotation untouched rather than inferring "unclustered" from silence.

### Policy advertisement refresh

The model refresh also compares the policy advertisement, including its revision
and `pausedBy` provenance (`local` or `peer`). A policy-only change updates the
directory and last-good cache and emits one `discovery:node-updated` with the full
node, even when its model inventory is unchanged. An identical answer emits no
update. A successful resume advertisement clears the pause hint.

Refresh has a nominal 15-second cadence, with other refresh work and bounded
fetch batches contributing to completion time. Observing a pause or resume
requires a successful fetch; there is no one-sweep convergence guarantee. Failed
fetches retain the last successful inventory and policy indefinitely, so a failed
refresh can retain a pause hint until a later successful resume fetch.

Non-loopback model fetches require an admitted, pinned cluster peer and select
HTTPS through the cluster transport; an unpinned LAN peer is refused without a
plaintext fallback. Loopback addresses select plaintext regardless of host UUID,
including a non-self UUID. Loopback fixtures therefore prove propagation, not
authenticated peer fetching. Availability is carried in the inventory policy,
never in mDNS TXT data; policy changes require no mDNS record change.

## Communication

Bidirectional newline-delimited JSON-RPC 2.0. Stdio by default; `--ipc <path>` switches to a Unix domain socket or Windows named pipe.

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--ipc <path>` | _(stdio)_ | IPC endpoint: Unix socket or Windows named pipe |
| `--cluster-dir <path>` | _(none)_ | Cluster config dir (`node.crt`/`node.key` + `trusted/`). Controls peer trust annotation, local membership advertisement, pinned direct/reverse peer transport for node information and model inventory, and the parent directory used to resolve this node's identity |
| `--ca-bundle <path>` | _(none)_ | PEM CA bundle to trust for verifying node-info server certs (additive to the system trust store). Setting any TLS flag moves node-info enrichment onto HTTPS |
| `--client-cert <path>` | _(none)_ | PEM client certificate to present when fetching `/v1/node-info` from TLS-enabled nodes (mTLS; requires `--client-key`) |
| `--client-key <path>` | _(none)_ | PEM client private key matching `--client-cert` |
| `--log-level <level>` | `info` | `error` / `warn` / `info` / `debug` |
| `--version` | | Print version and exit |

## JSON-RPC surface

### Requests (caller → daemon)

| Method | Params | Result |
|---|---|---|
| `discovery:register` | `{service, port, txt?}` | `{ok:true}` — record a local service so it's folded into this node's `_nvpair-node` record |
| `discovery:unregister` | `{service}` | `{ok:true}` — drop a local service |
| `discovery:update-txt` | `{service, port, txt?}` | `{ok:true}` — same as register with a new TXT |
| `discovery:get-nodes` | `{service?}` | `{nodes:[DirectoryNode]}` — the directory, optionally filtered to nodes advertising one service |
| `discovery:get-node-info` | `{hostUuid}` | Fresh full node-info sample for an admitted peer: `GPUs`, optional `cpu`/`memory`, `telemetryValid`, `msSince`, `hostUuid`, and nullable `clusterUuid` |
| `discovery:reload-identity` | — | `{ok:true}` — re-resolve and re-advertise this node's identity |
| `discovery:reload-trust` | — | `{ok:true}` — re-derive `trusted` on every directory entry from the current pin set |
| `discovery:set-observed-addresses` | `{addresses}` | `{ok:true}` — the local addresses peers have actually reached this node on, so address selection can rank a peer-proven address above an inferred one |
| `shutdown` | — | `null` |

`log/set-level` is also accepted.

`discovery:get-node-info` is a read for paired remote telemetry. The scanner
resolves the host identity to its current directory endpoint and admitted
cluster principal; callers cannot supply a URL, port or principal. It uses the
shared pinned peer/service adapter for `ni`, preferring direct plain HTTP and
using plain HTTP inside the authenticated reverse session when direct fails.
This bodyless GET may retry once over reverse when direct TCP connects but
first data resets, only before a response is exposed. The full response must
contain one JSON value with a clean body end, match the requested host UUID,
and match the directory principal if `clusterUuid` is non-null. Authority is
checked again before success.

The fresh read has a 1500 ms backend budget covering queue wait, address
attempts and HTTP/body work. It does not return the last-good enrichment cache
or compact scheduling telemetry. Invalid params yield `-32602`; missing
directory/admission/pin, unavailable transport, timeout or an invalid sample yields
`-32000` (`node information unavailable`) without a successful result or a
membership change. Recovery fetches a new sample through the same adapter.
There is no unpaired or operator-TLS fallback for a failed paired getter.
Existing self and unpaired enrichment paths remain direct. The broker exposes
this method over its existing local service connection; no listener is added.

The following notifications are accepted rather than requests, because the daemon has
nothing to answer and the caller must not wait:

- `discovery:node-activity` `{hostUuid, msSince}` — a peer's engine returned
  inference response bytes `msSince` ms ago. Counts as proof of life and cancels
  that node's eviction. The broker sends this while inference is streaming, so an
  id-bearing call would cost it a pending entry and a blocked goroutine per
  report in exchange for an ack nobody reads.
- `log/set-level` (above).

### Notifications (daemon → caller)

Emitted as the directory changes. Each carries the affected `DirectoryNode` (identity, advertised service ports, enrichment, `trusted`).

- `discovery:node-discovered` — a new node entered the directory
- `discovery:node-updated` — a known node's record or enrichment changed. A re-observation that changes nothing emits no event, so `lastSeen` is a last-*change* stamp rather than a liveness heartbeat
- `discovery:node-removed` — a node aged out (identity + last-known fields)
- `discovery:node-telemetry` — compact maximum-GPU utilization, validity, and
  sample age for broker-internal scheduling

- `errors:report` / `errors:clear` — a **paired** (`trusted`) remote peer this node cannot reach is reported as a `warning` (`action:"none"`) under one id per peer, `node-scanner:node-info-unreachable:<hostUuid>`. Two observations raise it, and the message states which:
  - **node-info not answering.** Checked when the peer's mDNS record is discovered or changes, at the point the daemon logs `node-info is not answering` (no published address answered). A peer already down when first discovered is reported. A peer that goes quiet behind an unchanged record is not noticed by this path, because nothing re-enriches it between record changes.
  - **reverse-session claims lapsing.** A peer known through an authenticated peer-services claim is not enriched over HTTP at all. When its claim passes the 30 s age limit while it is still admitted, the peer is reported; the claim age is the debounce.
  - **admitted peer that never spoke.** A claim is only sent after cluster-manager reads a peer's descriptor, so a scanner that starts while a reverse-session peer is down is never handed one. 30 s after start the scanner looks once at its admitted peers and reports any that has no directory entry (no claim and no mDNS record), with the same id. A peer with a directory entry is left to the claim or node-info path, whether or not it also reaches this node directly, and the single look means a peer evicted later is not reported straight back after its eviction cleared it. This is what makes a restarted scanner (or one restarted after the broker cleared its errors on a crash) report what is still true. Known limits: after a cold start or broker restart cluster-manager is new too, and its reverse session plus descriptor read can take longer than 30 s, so the look can briefly report a reachable peer until its first claim clears it; the look can also run between a browse's enrichment and its directory upsert and report a peer that is about to appear; and a peer missed at the look (its pin file mid-rewrite, or the mesh not yet clustered) is not looked at again until the scanner restarts.

  The clear follows when node-info answers again, the next claim for the peer is accepted, the peer stops being paired, or the node is evicted. Unpaired LAN nodes and this node's own loopback node-info are not reported. The scanner leaves `nodeId` and `timestamp` unset; the broker stamps them and forwards both notifications to `nvpair-errors`, and clears the ids still standing if the scanner crashes (a restarted scanner re-reports what is still true).

A one-shot `ready` notification is sent on startup carrying the daemon version. The broker replays its service registrations on every scanner spawn, so a restarted daemon is repopulated without needing to compare an epoch.

## Shutdown

The daemon shuts down on:
- stdin EOF (parent process closed the pipe)
- `SIGINT` / `SIGTERM`
- `shutdown` JSON-RPC request
