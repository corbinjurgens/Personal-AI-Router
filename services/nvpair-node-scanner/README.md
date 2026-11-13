<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvpair-node-scanner

Memory observations travel with compact node telemetry. Each fetch adds its
suspend-aware duration to producer age and stamps the local hop before codec
serialization. Directory/route retirement prevents memory emission from an
obsolete response. The hop is local to scanner and broker, never a remote
clock comparison. Existing utilization telemetry retains its own semantics.

Model enrichment accepts only nonempty valid UTF-8 identifiers of at most 512
UTF-8 bytes, with no C0 (U+0000–U+001F), DEL (U+007F), or C1 (U+0080–U+009F)
characters. Flat, per-engine and loaded model lists are filtered entry by entry
before cache/directory updates; raw model tokens are checked before JSON string
decoding can replace malformed UTF-8. Valid siblings retain their order and
engine attribution. A successful all-invalid list replaces last-good data with
an empty list; malformed documents and failed fetches retain last-good data.
Policy advertisements reuse the shared raw model-entry filter, including its
explicit empty served permission and opaque unsupported-capability behavior.

The same inventory carries optional `modelIdentities`, keyed by engine and
served model name. Each entry must pass `engines.ModelIdentity.Validate` and
belong to that response's `modelsByEngine`. Invalid entries and malformed engine
or identity blocks are dropped independently, preserving valid routing models.
The existing HTTP inventory body limit also bounds identity metadata. Identities
are display metadata and grant no routing or strict-admission authority.
Last-good identities stay paired with their last-good inventory on failed or
stale reads; a successful inventory removes identities for models no longer
listed. Identity-only changes publish `discovery:node-updated`. Directory and
cache copies detach the identity maps and their `files` slices.
Individual events and aggregate directory responses omit identity metadata when
its encoded JSON-RPC envelope would exceed the ordinary discovery frame limit.
Routing inventory and the stored identity snapshot remain intact; no transport
or HTTP limit is raised.

New policy writes remain strict. Only the broker's stored-policy loader removes
invalid old model entries and dependent references in memory, without changing
the stored file, revision or surviving policy. No identifier is trimmed,
truncated or replaced by a fallback name to pass validation. Ordinary Unicode,
spaces, tags and slashes keep their existing meaning; public methods and
payload fields are unchanged.

**The consolidated discovery daemon.** Promoted from a plain node-info browser: it is now the single place this host advertises itself on and discovers the LAN through. Every other service registers its port with this daemon instead of advertising its own mDNS service, and consumers query/subscribe to it instead of browsing themselves.

Concretely, the daemon:

- **Advertises** this node's one `_nvpair-node._tcp` record, built from a registry of the local services its parent (the broker) registers — `ni`/`ol`/`lm`/`er`/`wl`/`cl`/`em`/`ec` ports plus the node identity (`uuid=`, `cluster-uuid=` when clustered, `ip=`) and the display name (`name=`, once set). One record per node covers every service. The instance label and the SRV/A host are `nvpair-<uuid>` (host `nvpair-<uuid>.local.`), built from the node UUID, so generic Bonjour browsers show `nvpair-<uuid>`. When a re-stamp changes the UUID, the responder is replaced and the old instance withdrawn.
- **Carries the display name in `name=`.** The name is chosen by the node's cluster manager and is not identity; names need not be unique. The broker passes this node's name to the daemon with `discovery:set-self-name`; until then the record and the self entry carry no name. TXT strings travel byte for byte (backslashes, quotes and UTF-8 included; see `mdns.EscapeTXT`/`mdns.UnescapeTXT`).
- **Browses** `_nvpair-node._tcp` for every node on the LAN (including itself) and maintains a queryable directory keyed by host UUID. A browsed node's directory name comes from its `name=` value, validated on receipt with `nodename.Normalize` (missing or invalid gives an empty name); the instance label is never used as a name.
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
  If the first node-info fetch fails, later authenticated claims retry full
  enrichment until a node-info response succeeds, with one fetch in flight per
  peer. A valid response need not include CPU data to count as successful.
  Independently observed inference activity survives a source change; final
  removal clears it only when neither claim remains.
- **Enriches** each discovered node: GPU/CPU/memory inventory from its `ni` (node-info) port over plain HTTP, and the model list from its `em` (engine-manager) port (`GET /v1/models`) over cluster mTLS — a peer's model inventory is cluster data, so it is fetched only when this node holds a pin for the principal that peer advertises, while this node's own list is read over loopback in plaintext — the flat union, the per-engine breakdown (`modelsByEngine`), and the per-engine set of models loaded in memory (`loadedByEngine`), all carried onto the node's directory entry so remote cards can show loaded state. The same document carries the node's policy advertisement (`policy`: declared roles, policy revision, pause origin), which becomes the entry's `policy`; a peer known only from its authenticated peer-services claim is enriched through the same fetch. A node whose answer has no `policy` declares nothing. The policy's strict admission capability (`strictAdmission`) only affects hybrid eligibility, so a malformed one is dropped while the advertisement and model list are kept, and a case-variant `strictAdmission` key drops the advertisement but keeps the model list. Each enrichment has a last-good cache so a transient fetch miss doesn't blank the node's card. Node-info enrichment can optionally be moved onto HTTPS (see the TLS flags below) — off by default and gated only on operator flags, never inferred per-node.
- **Samples scheduling telemetry** from self and unpinned nodes every two
  seconds with stable per-node jitter. Consecutive remote failures back off to
  4, 8, 16, then 30 seconds and reset on success; local loopback sampling stays
  at the healthy cadence. It emits only host UUID, maximum GPU utilization,
  validity, and source-reported sample age; it does not mutate the public
  directory for each telemetry sample.
- **Streams pinned peers' node-info** instead of polling it. A stream peer is a
  directory node other than self that advertises `ni`, has a cluster principal,
  and whose principal this node holds a pin for. On every telemetry tick the
  scanner starts one `GET /v1/node-info/events` stream per stream peer (at most
  64) and cancels streams of nodes that stopped being stream peers. The stream
  uses the same pinned direct/reverse plain route and admission as paired
  `ni` reads, with no total timeout and without the per-origin read gate;
  connect and headers are bounded by a 10 s cancel timer and the stream drops
  after 6 s without a line. Node-info sends the full body on connect and on
  threshold changes, and `null` every 2 s while the sample stays within
  thresholds. A frame goes through the polled sample's path (host check,
  service build, memory observation, telemetry) and its reported `clusterUuid`
  is applied at once. A keepalive re-emits the last frame's maximum GPU
  utilization as a fresh sample (`msSince` 0, no memory fields), so the
  scheduler still gets a sample every 2 s. That understates the age by up to
  5 s at the start of a collector stall, within the scheduler's 10 s rule:
  once a valid GPU sample is 5 s old node-info publishes a frame on every
  `msSince` change, re-emissions keep ageing from that frame, and the frame
  node-info sends when the sample is fresh again ends the ageing.
  Every line rechecks that the route captured at connect is current and that
  the node is still a stream peer. A reconnect waits 2 s after a stream that
  delivered a frame and stayed open 5 s, and otherwise 4 s doubling to 30 s
  over consecutive failures.
- **Streams pinned peers' model inventory** instead of fetching it every 15 s.
  The same supervisor starts one `GET /v1/models/events` stream (engine-manager's
  em surface, pinned mTLS over the same direct/reverse route as the `/v1/models`
  fetch) for each stream peer that also advertises `em`, with the same connect
  bound, line-by-line route recheck, limit and reconnect delays as the node-info
  stream; it drops after 25 s without a line (engine-manager sends `null` every
  10 s). Each frame is a full `/v1/models` document and goes through the
  fetch's decode and apply path, so `discovery:node-updated` is emitted only
  when the inventory changed. When a stream ends, the node takes the
  failed-fetch path once: the last inventory is kept and download freshness is
  withdrawn. The 15 s sweep below skips stream peers; this node's own list
  (over loopback) and every other node keep it, and browse enrichment on an
  mDNS record change still fetches `/v1/models` once.
- **Annotates trust:** with `--cluster-dir`, a browsed peer whose `cluster-uuid=` this node holds a pin for is marked `trusted` (via `nvpair-shared/clustertrust`). The annotation is re-derived on `discovery:reload-trust`, which the broker sends when `nvpair-cluster-manager` reports a pin-set change, as well as when a peer's mDNS record changes. Both triggers are needed: a peer that was already advertising when this node joined never moves its record again, so the browse path alone would leave it annotated from before this node's own pins existed. A change emits one `discovery:node-updated`; a reload that finds nothing to change is silent.
- **Advertises live membership:** the same reload re-derives this node's own `cluster-uuid=` from the cluster dir and re-advertises when it differs from what is being published, so joining or leaving a cluster takes effect without a restart.
- **Anti-flap:** a node has to be absent from every scan for the browser's full miss threshold (twelve scans at 5s, a full minute) before it is evicted, and from ~15s in it is TCP-probed on its advertised service ports on *every* scan — any answer resets the counter and recovers it outright. So neither a momentary multicast gap nor a sustained one drops a healthy node, and eviction rests on a run of ~10 consecutive failures rather than one dial on the deadline. The window is sized deliberately: a node saturated by its own inference load starves its control plane, so it stops answering mDNS *and* stops accepting probe connections for as long as the load runs, while still serving requests. Two signals can also vouch for a node without any probe — a node-info enrichment in the last 10s, and inference response bytes from it in the last 60s (`discovery:node-activity`, relayed by the broker from whichever proxy saw the bytes). The second is the only liveness evidence that gets stronger the busier a node is.
- **A scan that loses everything is treated as our own fault:** when more than one node is known and a scan returns *none* of them, no node is penalized for it — for up to six consecutive scans (~30s), after which they count normally. Independent machines do not leave the network in the same five-second window, but a process too starved to drain its multicast socket within `scanTimeout` hears silence from all of them at once; without this, one saturated machine dropped its entire cluster, idle peers included, in a single reconcile. The suppression needs two or more known nodes to engage: losing a lone peer looks the same whether it left or we stopped listening, so that case falls through to the ordinary threshold.
- **Converges without mDNS:** a 15s sweep re-reads each peer over HTTP and folds any change into the directory, independent of browse events. It covers facts that change without moving the mDNS record (a model pull/delete, a late engine start) and facts that did move it but whose announcement was never received. The second case is why a peer's cluster membership is re-read from its `ni` port (`clusterUuid` on `/v1/node-info`, which under the broker is served plain — so it answers whether or not the two nodes share a cluster; a node-info run standalone with `--cluster-dir` gates the endpoint while clustered and will not answer a non-pinned peer, which the sweep treats as no answer): membership otherwise arrives solely as the `cluster-uuid=` TXT key, read once per record change, and the anti-flap probe above deliberately keeps a still-reachable peer's entry alive — so a missed re-advertisement would leave a departed peer's principal in place for the life of the process, suppressing the invite that would bring it back. Membership is refreshed before models on each tick, since the model fetch is pinned mTLS keyed on that principal. Stream peers' models arrive on their models stream instead (above). Peers with a live node-info stream are skipped: the stream reports a `clusterUuid` change as a frame at once. Only a peer's explicit report is acted on: a failed fetch, or a peer too old to report the field, leaves the annotation untouched rather than inferring "unclustered" from silence.

### Policy advertisement refresh

Model enrichment also carries the target's optional `downloadState` snapshot:
`{version:1,sourceId,revision,recoveryRequired,entries}`. Each entry preserves
`id`, `engine`, exact `model`, `phase`, `blocksReadiness`, optional measured
integer `percent`, bounded `reason`, and optional `persistence:"uncertain"`.
The shared decoder bounds snapshots to 256 KiB and 64 entries, validates source
and operation IDs, decimal uint64 revisions and lifecycle fields, and rejects
duplicate IDs/model keys. The target retains unresolved unsafe entries and a
bounded recent outcome set; scanner does not reconstruct operation history.

`downloadStateCurrent` is observation metadata: true follows a successful
validated current read, false follows failure or malformed lifecycle data.
Last-good operation details remain visible with false currentness; a successful
same-revision read can restore true. Absent support is unknown. Explicit current
`entries:[]` is known empty, while missing, malformed or stale state never
certifies readiness. The canonical model inventory and observed loaded set
remain independent; scanner neither infers completion from a model name nor
infers loaded state from completion. Updated clients require current support
before enabling ready-dependent actions, so the guarantee requires a coordinated
fleet update.

The HTTP inventory envelope is bounded before JSON decoding by the existing
large-worker reply limit. A malformed, aliased, repeated or oversize lifecycle
extension withdraws flat/per-engine ready names while retaining real loaded
observations and any valid prior snapshot with false currentness. This prevents
inventory-only routing consumers from treating those names as ready. An absent
extension keeps historical inventory semantics while updated clients report
readiness as unknown. The lifecycle extension's own 256 KiB limit remains
independent of the outer envelope cap.

Lifecycle-only changes and freshness loss emit directory updates through the
existing refresh path. Lower revisions from the same source are ignored; a
current replacement source retires reads begun against the prior baseline.
Host rekey retains stale display details and final eviction clears the cache.
Refresh uses the existing nominal 15-second sweep, its preceding refresh work
and bounded request batches; this is eventual convergence, with no universal
wall-clock deadline. Lifecycle state is not advertised in mDNS and introduces
no per-model polling.

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

The scanner relays a pause declaration rather than deciding availability. The
broker sends its policy advertisement to engine-manager, whose
[`ModelsResult`](../nvpair-engine-manager/models.go) includes it in `/v1/models`.
[`fetchModelsForPeer` and `refreshNodeModelsCandidates`](model_enrichment.go)
read that document; [`directory.applyModels`](directory.go) compares the policy
as well as the model lists, and `modelInventory.applyTo` in
[`enrich_hosts.go`](enrich_hosts.go) carries it as `DirectoryNode.Policy`.
Consequently, a pause-only change can reach subscribers without removing the
node or changing its mDNS record. A missing policy supplies no pause hint; a
failed fetch is not evidence of either pause or resume. The broker and proxy
consume this snapshot as described in the
[broker README](../nvpair-ui-broker/README.md#paused-node-discovery-propagation).

## Communication

Bidirectional newline-delimited JSON-RPC 2.0. Stdio by default; `--ipc <path>` switches to a Unix domain socket or Windows named pipe.

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--ipc <path>` | _(stdio)_ | IPC endpoint: Unix socket or Windows named pipe |
| `--mdns` | `true` | Browse and advertise nodes over mDNS; `--mdns=false` constructs neither the browser nor the responder |
| `--mdns-service <name>` | `_nvpair-node._tcp` | DNS-SD service used by both browser and responder (`_name._tcp`) |
| `--test-scan-interval <duration>` | `5s` | Fixture-only mDNS scan cadence |
| `--test-scan-timeout <duration>` | `3s` | Fixture-only timeout for each mDNS browse |
| `--test-peer-refresh-interval <duration>` | `15s` | Fixture-only cadence for the mDNS-independent inventory and membership sweep |
| `--cluster-dir <path>` | _(none)_ | Cluster config dir (`node.crt`/`node.key` + `trusted/`). Controls peer trust annotation, local membership advertisement, pinned direct/reverse peer transport for node information and model inventory, and the parent directory used to resolve this node's identity |
| `--ca-bundle <path>` | _(none)_ | PEM CA bundle to trust for verifying node-info server certs (additive to the system trust store). Setting any TLS flag moves node-info enrichment onto HTTPS |
| `--client-cert <path>` | _(none)_ | PEM client certificate to present when fetching `/v1/node-info` from TLS-enabled nodes (mTLS; requires `--client-key`) |
| `--client-key <path>` | _(none)_ | PEM client private key matching `--client-cert` |
| `--log-level <level>` | `info` | `error` / `warn` / `info` / `debug` |
| `--version` | | Print version and exit |

The `--test-*` durations are startup overrides for isolated process fixtures.
They accept Go duration strings, must be positive, and the scan timeout must not
exceed the scan interval. Invalid timings fail before trust-store access, IPC
connection or discovery construction, including when `--mdns=false`. With no
overrides, production scans, browse timeouts and peer refreshes remain 5s, 3s and
15s. Each scanner owns its timings; the shared browser's miss threshold,
empty-scan grace, liveness probes and eviction rules remain unchanged. Shorter
fixture cadences compress the time between scans without changing those rules.

Use `--mdns=false` for isolated tests or air-gapped deployments that do not use
multicast discovery. The scanner still publishes its local directory entry,
accepts the broker's private authenticated peer-services notifications, and runs
periodic enrichment and maintenance. Authenticated and explicitly configured peer
HTTP remains active under its existing transport and admission rules. This flag
provides neither general network isolation nor an authentication bypass.

For genuine multicast fixtures, give all participants the same private service
under `local`, for example `_pair-example._tcp`. Only `local` is supported:
zeroconf v1.0.0 caches its receive-matching service name under that domain.
The responder uses `nvpair-<uuid>.local.` as its SRV/A target; sharing a host
name cannot match another service's PTR, SRV or TXT records. Service names are
lowercased and accept one optional trailing root dot. Empty names, invalid DNS
labels and service types other than `_name._tcp` fail before
identity, IPC or discovery setup, even with `--mdns=false`. Production defaults
are unchanged. Names separate discovery records; configured and authenticated
peer HTTP still requires independently owned fixture endpoints.

## JSON-RPC surface

### Requests (caller → daemon)

| Method | Params | Result |
|---|---|---|
| `discovery:register` | `{service, port, txt?}` | `{ok:true}` — record a local service so it's folded into this node's `_nvpair-node` record |
| `discovery:unregister` | `{service}` | `{ok:true}` — drop a local service |
| `discovery:update-txt` | `{service, port, txt?}` | `{ok:true}` — same as register with a new TXT |
| `discovery:get-nodes` | `{service?}` | `{nodes:[DirectoryNode]}` — the directory, optionally filtered to nodes advertising one service |
| `discovery:get-node-info` | `{hostUuid}` | Full node-info sample for this scanner's registered self or an admitted peer (a pinned peer's from its live node-info stream): `GPUs` (including optional `physicalId`), optional `cpu`/`memory`/`memoryObservation`, `serviceBuildId`, `telemetryValid`, `msSince`, `hostUuid`, and nullable `clusterUuid` |
| `discovery:reload-identity` | — | `{ok:true}` — re-resolve and re-advertise this node's identity |
| `discovery:set-self-name` | `{name}` | `{ok:true}` — advertise `name` as this node's `name=` and on its self entry; a name `nodename.Normalize` refuses is rejected with `-32602` |
| `discovery:reload-trust` | — | `{ok:true}` — re-derive `trusted` on every directory entry from the current pin set |
| `discovery:set-observed-addresses` | `{addresses}` | `{ok:true}` — the local addresses peers have actually reached this node on, so address selection can rank a peer-proven address above an inferred one |
| `shutdown` | — | `null` |

`log/set-level` is also accepted.

`discovery:set-self-name` is private worker IPC: the broker sends it with the
display name from this node's cluster manager, which owns the name. The daemon
stores the normalized value, re-announces the TXT record and republishes its
self entry when the name changed.

`discovery:get-node-info` reads fresh local or paired remote node information.
Callers supply only a host identity, never a URL, port or principal. For this
scanner's exact stamped host UUID, the local registry owns authority: it must
have a valid `ni` service registration from the broker. The scanner reads that
registered port exclusively over `127.0.0.1` using its existing plain HTTP
client, independently of advertised LAN addresses, directory claims, peer pins
and operator TLS. Redirects are refused. The full response must identify the
same host and explicitly report the registered cluster principal (an empty
string for unclustered self). Unknown or conflicting identity is unavailable.

The registry's local node-info incarnation retires on every `ni` registration,
including same-port replacement, unregister, or stamped host/principal change.
Retirement cancels queued and in-flight reads; an already buffered response is
also rejected. Address changes and unrelated service updates do not retire the
loopback authority. Directory or mDNS entries, including entries advertising a
loopback address, cannot establish self authority.

For a stream peer (a pinned peer, see "Streams pinned peers' node-info"
above) the scanner answers from the latest frame of its node-info stream, with
`msSince` aged as for keepalive re-emissions (time since the last line, or a
stale frame's age plus the time since it), and only while a line arrived
within 6 s and the route the stream was opened on is still current at answer
time. Otherwise the answer is unavailable; there is no peer request for a
stream peer and nothing stale is returned after revocation or a route outage.

For any other remote host the scanner resolves its current directory endpoint
and admitted cluster principal. It uses the
shared pinned peer/service adapter for `ni`, preferring direct plain HTTP and
using plain HTTP inside the authenticated reverse session when direct fails.
This bodyless GET may retry once over reverse when direct TCP connects but
first data resets, only before a response is exposed. The full response must
contain one JSON value with a clean body end, match the requested host UUID,
and match the directory principal if `clusterUuid` is non-null. Authority is
checked again before success.

The local and remote reads have a 1500 ms backend budget covering queue wait, address
attempts and HTTP/body work. Neither returns the last-good enrichment cache
or compact scheduling telemetry. Full GPU rows, including observed physical
aliases, and the memory observation pass through without reconstruction.
Invalid params yield `-32602`; missing local registration or remote
directory/admission/pin, retired authority, unavailable transport, timeout or an
invalid sample yields
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
  - **reverse-session claims lapsing.** A peer known through an authenticated peer-services claim is enriched through its advertised services, but claim receipt itself does not establish node-info liveness. When its claim passes the 30 s age limit while it is still admitted, the peer is reported; the claim age is the debounce.
  - **admitted peer that never spoke.** A claim is only sent after cluster-manager reads a peer's descriptor, so a scanner that starts while a reverse-session peer is down is never handed one. 30 s after start the scanner looks once at its admitted peers and reports any that has no directory entry (no claim and no mDNS record), with the same id. A peer with a directory entry is left to the claim or node-info path, whether or not it also reaches this node directly, and the single look means a peer evicted later is not reported straight back after its eviction cleared it. This is what makes a restarted scanner (or one restarted after the broker cleared its errors on a crash) report what is still true. Known limits: after a cold start or broker restart cluster-manager is new too, and its reverse session plus descriptor read can take longer than 30 s, so the look can briefly report a reachable peer until its first claim clears it; the look can also run between a browse's enrichment and its directory upsert and report a peer that is about to appear; and a peer missed at the look (its pin file mid-rewrite, or the mesh not yet clustered) is not looked at again until the scanner restarts.

  The clear follows when node-info answers again, the next claim for the peer is accepted, the peer stops being paired, or the node is evicted. Unpaired LAN nodes and this node's own loopback node-info are not reported. The scanner leaves `nodeId` and `timestamp` unset; the broker stamps them and forwards both notifications to `nvpair-errors`, and clears the ids still standing if the scanner crashes (a restarted scanner re-reports what is still true).

A one-shot `ready` notification is sent on startup carrying the daemon version. The broker replays its service registrations on every scanner spawn, so a restarted daemon is repopulated without needing to compare an epoch.

## Shutdown

The daemon shuts down on:
- stdin EOF (parent process closed the pipe)
- `SIGINT` / `SIGTERM`
- `shutdown` JSON-RPC request
