<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvpair-cluster-manager

> This README is the **integrator's guide**: how a parent process drives
> cluster membership and node pairing over this component's JSON-RPC
> protocol. The binary is built by `build.sh` / `build.bat` alongside the
> other eleven components and is spawned by `nvpair-ui-broker` at startup,
> as an optional worker — a missing binary leaves the broker running without
> cluster pairing.

## Purpose

`nvpair-cluster-manager` owns, for one node: its **cluster membership** (who
its peers are), its **cryptographic identity** (a stable UUID + keypair +
self-signed cert), and the **trusted-node store** (the pinned peer certs
that back mTLS on every inter-node interface). You drive it to *found* a
cluster, *invite* other nodes, *respond* to invites, and *remove* nodes.
Trust between nodes is bootstrapped by an **EAP-NOOB pairing authenticated
by a six-digit PIN** the user carries out of band.

You talk to it over **local JSON-RPC** (this README). It talks to other
nodes' cluster managers over **HTTP/mTLS** on TCP `14321` — that part is
internal; you only need to make sure the port is reachable (see
[Networking](#inter-node-networking)).

## Communication

An Initial Exchange transport failure returns `state:"failed"`, its original
`error` diagnostic, and a stable `reason` (`connection-reset`,
`connection-refused`, `connection-timeout`, or `connection-unreachable`).
The reason is retained in invite status. Desktop and terminal clients suggest
opening PAIR on the other device and inviting this computer for these failures;
wrong PIN, protocol failure, and explicit refusal retain their own remedies.

Newline-delimited **JSON-RPC 2.0**, one object per `\n`. Stdio by default
(`stdin` = requests in, `stdout` = results + notifications out); `--ipc
<path>` switches to a Unix domain socket or Windows named pipe. The local
interface is **request/response** plus a few manager → caller
notifications. The manager never sends you a *request*, so you do not need
an inbound-request handler — only a notification handler.

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--ipc <path>` | _(stdio)_ | IPC endpoint: Unix socket or Windows named pipe instead of stdio |
| `--config-dir <path>` | shared per-user config dir | Where the `cluster/` identity + trusted-store subtree lives (§7.4) |
| `--port <n>` | `14321` | Inter-node HTTP listener port |
| `--log-level <level>` | `info` | One of `error`/`warn`/`info`/`debug`; mutable at runtime via `log/set-level` |
| `--version` | | Print version and exit |

No flag carries the node identity (self-generated) or the cluster identity
(supplied via `cluster:set-identity`) — the binary is self-bootstrapping
under any supervisor.

## Lifecycle

1. **Spawn** the binary as a child, wired to its stdio (or `--ipc`). On
   Windows, hide the console window the same way the other subprocesses are
   spawned (`syscall.SysProcAttr{HideWindow: true, CreationFlags:
   0x08000000}`).
2. **Readiness:** there is no `ready` banner. The node identity is minted
   at startup, so a first successful `cluster:get-node-id` is the natural
   readiness probe.
3. **Hand it the cluster identity:** right after spawn, call
   `cluster:set-identity` with the `cluster_id` / `cluster_friendly_name`
   you hold in `nvpair-node-settings` (empty strings if none). Call it again
   whenever node-settings reports a change.
4. **Shutdown:** the manager exits cleanly on `stdin` EOF (close its pipe)
   or `SIGINT`/`SIGTERM`. There is no reconnect or buffering — a fresh
   manager is a fresh spawn.

## Your responsibilities (the integration contract)

The manager owns membership and trust; **you own the human and the
settings store**. Concretely, the parent must:

- **Persist cluster identity.** When you receive `cluster:identity-changed`
  (after a `cluster:create` or a successful join), write the new
  `cluster_id` / `cluster_friendly_name` to `nvpair-node-settings`. On the
  next startup, feed it back via `cluster:set-identity`. The manager
  *reports* the value; it never writes node-settings itself.
- **Display and collect the PIN.** On `cluster:invite-node` you get back a
  six-digit `pin` — show it to the inviting user. On the joining node you
  get `cluster:invite-received` (no PIN) — prompt that user to type the PIN
  shown on the inviting node, then send `cluster:respond-to-invite`. The
  PIN never crosses the network; the human carries it. If the same node
  sends a newer invite before the first is answered, the manager emits
  `cluster:invite-canceled` for the old ID before `cluster:invite-received`
  for the replacement; dismiss the old prompt and keep the newest one.
- **Resolve invite targets to an address.** `cluster:invite-node` takes a
  reachable `address` (and optional `port`/`nodeId`). You pick the target
  (from `nvpair-node-scanner`, manual nodes, etc.) and pass a resolved
  address; the manager does not do UUID→address resolution for you.
- **Don't re-invite an already-clustered node.** If the target is already
  in a cluster it **rejects** the pairing: `cluster:invite-node` returns
  `state:"rejected"` (`reason:"already-clustered"`) with no PIN. Skip the
  invite up front when discovery already shows the target as clustered
  (its `cluster-uuid=`/`trusted`) — surface it as "Connected" / "Already
  paired" — and only allow a re-pair after the existing relationship is
  removed (`nodes:remove` / `cluster:leave`).
- **Drive the join/leave UI** off `nodes:get-initial` (seed) plus
  `nodes:changed` (live updates, if enabled) — see the [flows](#typical-flows).

## Methods (caller → manager)

All requests carry a JSON-RPC `id` and get exactly one `result` or `error`
back. Negative *outcomes* (failed pairing, wrong PIN, non-member removal)
come back as a **successful `result`** with a terminal `state`/flag — not
as a JSON-RPC `error`. See [Errors](#errors).

```jsonc
// --- identity ---
{"jsonrpc":"2.0","id":1,"method":"cluster:get-node-id"}
//   -> {"nodeUuid":"b3d4...","nodeId":"NODE-A","name":"Lab desk A","certFingerprint":"sha256:...","clusterId":""}

{"jsonrpc":"2.0","id":2,"method":"cluster:set-identity","params":{"clusterId":"cluster-xyz","clusterFriendlyName":"Lab 3 desks"}}
//   -> {"clusterId":"cluster-xyz","clusterFriendlyName":"Lab 3 desks"}   (reflects node-settings; no identity-changed push)

// --- found a cluster (only when clusterId == "") ---
{"jsonrpc":"2.0","id":3,"method":"cluster:create","params":{"clusterFriendlyName":"Lab 3 desks"}}
//   -> {"clusterId":"<new-uuid-v4>","clusterFriendlyName":"Lab 3 desks"}  (also emits cluster:identity-changed)
//   The manager ALWAYS mints a fresh, globally-unique clusterId (UUID v4) — you only pass a friendly name.
//   An explicit create is intentional and is never auto-dissolved by a declined invite.
//   To replay a previously-known id (restore/startup) use cluster:set-identity instead, not create.
//   Called while already clustered -> error -32004 {reason:"already-clustered"}.
//   OPTIONAL: you only need create to name the cluster up front. Inviting while
//   unclustered auto-founds a cluster of one for you (see cluster:invite-node).

// --- membership ---
{"jsonrpc":"2.0","id":4,"method":"nodes:get-initial"}
//   -> {"nodes":[ <ClusterNode>, ... ]}   (members + pending invites; empty list is normal)

{"jsonrpc":"2.0","id":5,"method":"nodes:remove","params":{"nodeId":"NODE-B"}}   // nodeUuid also accepted
//   -> {"nodeId":"NODE-B","removed":true}   (removed:false if it wasn't a member)
//   removing self -> error -32602 {field:"nodeUuid"} — use cluster:leave instead.

{"jsonrpc":"2.0","id":10,"method":"cluster:leave"}   // this node unjoins its own cluster
//   -> {"left":true}   (left:false if already unclustered)
//   announces departure to all members, drops every pin/member, resets to unclustered
//   (emits cluster:identity-changed with clusterId:"" and an empty nodes:changed).

// --- pairing (inviter side) ---
{"jsonrpc":"2.0","id":6,"method":"cluster:invite-node","params":{"address":"192.168.1.22","port":14321,"nodeId":"NODE-B","allowRemoteManagement":false}}
//   -> {"inviteId":"inv-9f3a1c","state":"pending","pin":"402199"}   (show the PIN; "failed" + no pin if unreachable)
//   -> {"inviteId":"inv-9f3a1c","state":"rejected","reason":"already-clustered"}   (no pin; target is already in a cluster - remove/leave first)
//   if this node is unclustered, it auto-founds a cluster of one first (same as
//   cluster:create, emitting cluster:identity-changed) — no separate create needed.

{"jsonrpc":"2.0","id":7,"method":"cluster:invite-status","params":{"inviteId":"inv-9f3a1c"}}
//   -> <Invite>   (poll the inviter's invite until state flips to "paired"/"failed"/"expired")

// --- pairing (joiner side) ---
{"jsonrpc":"2.0","id":8,"method":"cluster:respond-to-invite","params":{"inviteId":"inv-9f3a1c","accept":true,"pin":"402199","allowRemoteManagement":false}}
//   -> <Invite with state:"paired" or "pending">   (state:"failed" + reason:"incorrect-pin" on a wrong PIN; state:"declined" if accept:false)
//   With negotiated inviter-v1 completion, the inviter polls the joiner and drives
//   Completion plus authenticated confirmation; the joiner makes no pairing dial.
//   On the legacy transport, decline best-effort notifies the inviter (phase "decline"),
//   so the inviter's invite-status flips to "declined" and it pushes cluster:invite-declined.
//   In legacy mode, a wrong PIN likewise notifies the inviter (pairing phase "fail"), so the inviter's
//   invite-status flips to "failed" (reason:"incorrect-pin") and it pushes cluster:invite-failed.
//   If the user never responds, the invite times out on both sides after the invite TTL
//   (default 5m): the receiver expires it and clears the PIN prompt. Legacy mode notifies
//   the inviter (phase "expire"); negotiated mode exposes expiry through readiness polling.

// --- logging ---
{"jsonrpc":"2.0","id":9,"method":"log/set-level","params":{"level":"debug"}}
```

`cluster:invite-node` runs the Initial Exchange. When both transcript objects
advertise `completion:"inviter-v1"`, PIN submission makes the joiner ready and
waits for a local terminal outcome for up to 30 seconds; a pending response can
be followed through `cluster:invite-status`. The inviter drives every pairing
network round trip, allowing it to pair while new inbound connections are
blocked. An older peer retains the joiner-driven Completion transport. Match
responses by JSON-RPC `id` and allow the PIN submission wait in client timeouts.

If PIN submission returns pending, continue polling `cluster:invite-status`
with that same inviteId through its matching terminal or unknown/expired outcome. The receiver
updates `nodes:changed` when a wrong PIN or receipt-write failure clears its
pending member; there is no separate new receiver terminal notification.
Either role compensating a failed commit stays pending with
`reason:"rollback-pending"` until its durable pin/member cleanup succeeds.
Unconfirmed restart recovery stays pending with `reason:"awaiting-confirmation"`
until it authenticates the exact invite outcome or the existing invite TTL
expires; expiry/unknown does not undo the durable admission. A rollback receipt is replayed
before startup serves and preserves a newer admission; failed durable replay
aborts startup. Silence never authorizes rollback or a new paired success claim.

## Notifications (manager → caller)

No `id`; never carry an error; handle them off your read loop.

```jsonc
// A pairing arrived and this node was invited. Prompt the user for the PIN, then respond-to-invite.
{"jsonrpc":"2.0","method":"cluster:invite-received","params": <Invite with state:"pending", pin:null> }

// Inviter-side: the joiner declined. Clear the PIN prompt. If this cluster was
// auto-created by invite-node and no sibling
// pending outbound invite remains, the manager also leaves (cluster:identity-changed
// with empty id) so both sides end standalone. An intentional solo cluster is kept.
{"jsonrpc":"2.0","method":"cluster:invite-declined","params": <Invite with state:"declined", pin:null> }

// Inviter-side: the Completion Exchange failed (today only a wrong PIN). Show an
// "Incorrect PIN" error keyed on reason:"incorrect-pin" (empty reason for other causes).
// Same provenance-safe cleanup as decline (a throwaway auto-created solo cluster is left).
{"jsonrpc":"2.0","method":"cluster:invite-failed","params": <Invite with state:"failed", reason:"incorrect-pin", pin:null> }

// A pending invite hit its TTL without accept/decline. Emitted on the INVITER
// (outbound: lost decline / abandoned PIN / offline joiner; same provenance-safe
// cleanup as decline; a purely local fallback that does not signal the joiner)
// and on the RECEIVER (inbound: the local user never entered the PIN — clears the
// PIN prompt and drops the tentative pending-inbound member). The receiver
// signals the inviter (pairing phase "expire") for immediate teardown; each
// side's own TTL is the fallback if the signal is lost.
{"jsonrpc":"2.0","method":"cluster:invite-expired","params": <Invite with state:"expired", pin:null> }

// Joiner-side: the inviter canceled this invite, its TTL expired, or a newer
// invite from the same sender superseded it. Dismiss this invite's PIN prompt.
{"jsonrpc":"2.0","method":"cluster:invite-canceled","params": <Invite with state:"canceled", pin:null> }

// The local clusterId originated here — i.e. after cluster:create, or after this node joined on accept.
// Persist {clusterId, clusterFriendlyName} to nvpair-node-settings. NOT emitted for changes you drove via set-identity.
{"jsonrpc":"2.0","method":"cluster:identity-changed","params":{"clusterId":"...","clusterFriendlyName":"..."}}

// (PROPOSED, spec §4) Full membership snapshot on every change, so you can avoid polling nodes:get-initial.
{"jsonrpc":"2.0","method":"nodes:changed","params":{"nodes":[ <ClusterNode>, ... ]}}
```

## Data shapes

`ClusterNode` (a member or pending invitee, from `nodes:get-initial` /
`nodes:changed`):

```jsonc
{
  "id":        "NODE-B",            // logical id (hostname); the membership key, display only
  "nodeUuid":  "7a1c...",           // stable cryptographic id; the trust principal
  "name":      "Lab desk B",        // display name
  "ipAddress": "192.168.1.22",
  "advertisedAddresses": ["192.168.1.22", "100.64.0.10"], // optional authenticated locator hints
  "port":      14321,
  "clusterId": "cluster-xyz",
  "state":     "member",            // "member" | "pending-outbound" | "pending-inbound"
  "joinedAt":  1716998460000,       // epoch ms; null while pending
  "lastSeen":  1716998500000        // epoch ms; null if never contacted
}
```

`advertisedAddresses` is optional and omitted when empty. Confirmed members
retain these descriptor-derived hints in `members.json`; old records without
the field still load. The manager clones the slice on member inputs, reads and
snapshots. Removing membership or completing teardown removes its hints; a
fresh admission does not inherit the removed member's hints. Hints neither
replace `ipAddress` nor grant membership, trust, liveness or service presence.

`Invite` (the pairing session, from invite-node/-status/-received/respond):

```jsonc
{
  "inviteId":  "inv-9f3a1c",        // correlation / dedup key
  "fromNodeId":   "NODE-A",
  "fromNodeUuid": "b3d4...",
  "fromNodeName": "Lab desk A",
  "toNodeId":  "NODE-B",            // null if unknown to the inviter
  "clusterId": "cluster-xyz",
  "clusterFriendlyName": "Lab 3 desks",
  "pin":       "402199",            // inviter-local invite-node or pending outbound invite-status only; never logged
  "state":     "pending",           // "pending" | "paired" | "declined" | "canceled" | "expired" | "failed" | "rejected"
  "createdAt": 1716998400000,
  "respondedAt": null
}
```

## Typical flows

**Found a new cluster (first node).** `cluster:get-node-id` → `clusterId:
""` → `cluster:create` (the manager mints a unique UUID v4 `clusterId`) →
persist the `cluster:identity-changed` payload to node-settings. The node is
now a cluster of one and can invite. This explicit step is **optional** — its
only value over the auto-found below is naming the cluster (`clusterFriendlyName`)
before the first invite.

**Invite a node (inviter).** `cluster:invite-node {address, allowRemoteManagement:false}` → show the
returned `pin` → tell the user to read it to the other node's operator →
watch for `state:"paired"` via `nodes:changed` (or poll
`cluster:invite-status`). If this node isn't clustered yet the invite
auto-founds a cluster of one first (same mint-and-record as `cluster:create`,
emitting `cluster:identity-changed`), so a UI need not orchestrate a separate
create step.

**Receive & accept an invite (joiner).** Handle `cluster:invite-received` →
prompt "enter the PIN shown on `fromNodeName`" →
`cluster:respond-to-invite {inviteId, accept:true, pin, allowRemoteManagement:false}`.
A `result` with `state:"paired"` means
joined (the manager also pinned the peer and emitted
`cluster:identity-changed` with the adopted cluster — persist it). Decline
with `accept:false`. A replacement invite from the same `fromNodeUuid`
automatically cancels the older ID; handle its `cluster:invite-canceled`
notification before presenting the replacement.

**Remove a node.** `nodes:remove {nodeId}` → `{removed:true}`. The manager
prepares an exact-admission removal proof, selects a current endpoint, persists
the proof only once that selection and the original admissions are still
current, and attempts one authenticated notification while the pin still
exists. A stale selection leaves no durable removal state. It then
rechecks the original admissions, deletes the pin and drops the member. An
unavailable notification does not prevent this best-effort local removal;
durable proofs support later convergence, without replaying the notification.

## Ask to join an existing cluster

A standalone requester addresses one existing member. That member's local user
approves, then displays the ordinary pairing PIN; the requester enters it using
`cluster:respond-to-invite` and chooses its own `allowRemoteManagement` value.
The member remains EAP Server/inviter and the requester EAP Peer/joiner.
Approval is consent to start this ceremony, not membership or remote-management
authority over the requester. Normal invites and their already-clustered
joiner rejection retain their existing contract.

| Local method | Params | Result |
| --- | --- | --- |
| `cluster:get-join-requests` | `{}` | `{supported:true, canAskToJoin, requiresSoloReplacement, blockedReason, requests:JoinRequest[]}` |
| `cluster:ask-to-join` | `{address, port?, nodeId?, replaceSoloCluster:boolean}` | `JoinRequest`; returns promptly, manager owns progress |
| `cluster:respond-to-join-request` | `{requestId, accept:true, allowRemoteManagement:boolean}` or `{requestId, accept:false}` | Updated incoming `JoinRequest` |
| `cluster:cancel-join-request` | `{requestId}` | Updated outgoing `JoinRequest` |
| `cluster:join-requests-changed` | Notification | Same full snapshot as the getter |

`JoinRequest` contains `requestId`, `direction` (`incoming`/`outgoing`),
`peerNodeUuid`, `peerName`, `peerAddress`, `peerPort`, `state`, nullable
`inviteId`, `reason`, `createdAt` and `expiresAt` (epoch ms). States are
`awaiting-approval`, `awaiting-requester`, `pairing`, `paired`, `declined`,
`canceled`, `expired`, `failed` and `rejected`. IDs have at least 128 random bits;
they correlate operations and never authorize membership. Names/UUIDs before
pairing are unverified claims; names are bounded and sanitized. Request records
and snapshots contain no PIN, certificate, key or grant. Member clients read
the pending inviter PIN locally through `cluster:invite-status` once the exact
child `inviteId` exists. Only the existing per-user local endpoint can approve.

The getter reconciles authoritative composition; `blockedReason` is empty when
eligible, otherwise `already-clustered`, `pairing-busy` or `state-unavailable`.
Refuse self-targets (`self-target`), retained peer relationships (offline peers
included), competing pairing, unreadable stores and incomplete publication,
receipt/rollback or teardown. Discovery absence cannot prove an empty cluster.
An intentional empty solo requires explicit consent: “Joining replaces this
device's empty cluster. Other devices will not be moved.” It stays intact until
approval, a fresh safe-solo check and durable teardown. Preserve models,
engine/settings and node UUID/keypair; failure after replacement leaves the
device unclustered, without resurrecting its retired admission.

The existing plain split-listener route is `POST /v1/cluster/join-requests`.
Bodies are strict `{version:1, operation, requestId}`, with operations `request`,
`status`, `ready`, `cancel`; `request` additionally requires
`requester:{nodeUuid,name}` and `port`. HTTP 200 returns
`{version:1, requestId, state, reason, createdAt, expiresAt, inviteId:string|null}`.
No PIN, grant, roster, certificate or eligibility snapshot is returned. Bind
follow-ups to the observed source IP and ID, ignore forwarded-address headers,
and callback only to that IP and validated pairing port, without redirects.

Allowed sources are private unicast IPv4 (`10.0.0.0/8`, `172.16.0.0/12`,
`192.168.0.0/16`), IPv6 ULA (`fc00::/7`), Tailscale CGNAT
(`100.64.0.0/10`) and fixture loopback; reject public, multicast, unspecified
and link-local sources. Canonicalize
IPv4-mapped IPv6 before source accounting. Bounds: 4 KiB strict bodies; one
outgoing request; eight live incoming globally and one per IP; creates per IP
one/30 seconds (burst one), globally one/second (burst four); follow-ups per IP
five/second (burst ten). The source table holds 128 entries with five-minute
idle expiry. Terminal records last five minutes, capped at 64; do not evict live
cancellation fences to accept new work. Eight nonblocking HTTP handler slots
and bounded body/operation timeouts refuse excess traffic before allocation.
These constants add no settings, listeners or firewall permissions.

HTTP errors use `{version:1, reason}`: 400 malformed/version errors
(`unsupported-version`), 404 unknown status/ready or wrong source, 409 conflicting
reuse or `target-not-clustered`, 429 capacity/rate refusal with `Retry-After`.
Duplicate creates do not refresh expiry or prompt again. Source-bound
cancel-before-create tombstones prevent reordered resurrection within the same
bounds. Requests create passive review data, never automatic focus/modal changes.

Request creation captures the member cluster/admission/generation. Approval sets
`awaiting-requester`; it starts no EAP. The requester polls every two seconds,
one call in flight. After eligibility/consent recheck and any durable solo
replacement, it sends `ready`. This starts exactly one canonical member-to-
requester invite with the saved member grant choice; leave/rejoin invalidates
approval. Optional authenticated `PairingInfo.joinRequestId` binds the exact
live request and child invite. Check authenticated peer UUID when discovery
supplied one. Unrelated invites do not satisfy requests; Ask never auto-accepts
or enters a PIN. A requester blocking inbound handoff reports connection failure.

Only the exact child's confirmed durable receipt outcome yields local `paired`;
HTTP status is advisory. Keep existing wrong-PIN, confirmation, rollback and
compensation behavior. Uncertain durable outcomes block another attempt. Request
lifetime is `effectiveInviteTTL()` from local creation (production five minutes);
the child keeps its own invite TTL. Cancel/expiry fences late approval/ready and
cancels only the exact pending child. Durable completion wins a late cancel;
never unpair to honor it. Members decline before handoff and cancel the exact
child with `cluster:cancel-invite` afterward.

Client detach preserves service-owned work. Manager restart loses memory-only
requests; receipt recovery still runs. Show “Request status is no longer
available. Check the cluster on both devices before trying again.” Read actual
membership before retrying. Local errors: `-32001` unknown ID, `-32002` invalid
state, `-32602` invalid params, `-32004` eligibility refusal with stable reason.
The broker preserves code/message but drops error data, so read eligibility
reasons from the snapshot. Old local method-not-found means unsupported;
initial peer HTTP 404 means `unsupported-peer`, later status 404 means lost
attempt. Network failure does not imply eligibility. No fallback Invite or
pre-approval solo teardown. Full transaction and abuse contract: spec §7.0.1.

## Remote-management permission

`allowRemoteManagement` is a required boolean on invite initiation and on
acceptance (`accept:true`); omission or null returns `-32602`. Declines need no
permission choice. The inviter chooses whether the joiner may manage the
inviter's device; the joiner independently chooses whether the inviter may
manage the joiner's device. The choice stays local to that pairing transaction.
Both completion transports persist its admission-bound decision before paired
success; receipt replay cannot overwrite a later local decision. A failed or
compensated pairing does not leave an effective grant.

Only clients attached to this user's local `nvpair-service` endpoint can edit
this device's grants. This relies on OS endpoint access, not physical presence;
another process running as that user has the same access. No peer HTTP setter
or peer grant-list endpoint exists. Local RPCs are:

| Method | Params / result |
| --- | --- |
| `cluster:get-remote-management` | `{}` → `{version:1, revision, grants:[{peerUuid, allowed, state}]}` for current pinned members |
| `cluster:set-remote-management` | `{peerUuid, allowed, expectedRevision}` → the fresh snapshot above |
| `cluster:remote-management-changed` | Local notification carrying the same snapshot after a committed update or uncertain landed replacement |

`allowed` must be a real boolean; `expectedRevision` is the snapshot's opaque
canonical decimal string, never a JavaScript number. Malformed params return
`-32602`; a stale revision returns `-32010`. States are `allowed`, `denied`,
`unreviewed` or `unavailable`; only `allowed` grants authority. Fetch on attach
and consume local pushes. An unavailable snapshot can have an empty revision;
do not construct a CAS write from it. After a failed or uncertain save, fetch
again rather than retrying a stale write or assuming rollback. Before-replacement failure
retains the previous document; an uncertain landed denial already denies.

Cluster-manager alone writes `cluster/remote-management.json`. On first PG
startup it automatically captures existing pin/admission authorities in
`remote-management-migration.json`, including an empty snapshot for a fresh
profile. A pending migration resumes only that immutable snapshot; only still
matching admissions are grandfathered. The completed marker survives leave and
prevents resnapshotting or recreating grants after grant-file loss. The
`remote-management-initializing.json` publication guard keeps incomplete or
uncertain allows unavailable until durably confirmed. Missing, corrupt,
unsupported, inconsistent or changing authority fails closed. If neither
authority file nor publication guard survives, PG cannot distinguish profile
loss from a first installation; this is not a supported automatic repair.

Grants match both nodes' current admissions and certificates. Same-admission
re-pinning preserves a choice; removal/re-pair, readmission, certificate
replacement or local leave/rejoin invalidates it. Endorsements, roster gossip,
pin presence and the inert `cluster_auto_sync` preference never grant it.
Revocation prevents new acceptance; already accepted work can finish or
reconcile, with its existing progress/receipts still readable. It does not stop,
pause or unpair the node. Local pause/Off, private policy, CORS and served/resident
restrictions remain independent. Reads and permitted inference remain available.
An authorization overlapping revocation may already hold a coherent allow
snapshot; authorization started after a successful revoke denies. Revocation
does not claim that earlier subprocess or filesystem effects were undone.

Downgrading the target to pre-PG software restores that executable's unrestricted
pinned-peer management behavior. Old pairing clients without the required
boolean must upgrade; an old target's missing permission read means unsupported,
not protected. Local decision logs contain actor kind, peer UUID, attempted and
previous revision, incarnation, old/new choice and outcome; no PIN, certificate,
policy body, credential or inference content is part of this audit.

## Errors

`result` vs `error` is the "did it apply?" vs "could I even process this?"
signal. Use the standard JSON-RPC `error` object `{code, message, data?}`.

| Code | Meaning |
|---|---|
| `-32700` / `-32600` / `-32601` / `-32602` | parse error / invalid request / method not found / invalid params |
| `-32603` | internal error; a permission replacement may have landed with uncertain durability — read back before retrying |
| `-32001` | unknown invite or local join-request ID |
| `-32002` | invalid invite or join-request state |
| `-32010` | remote-management revision conflict — fetch the local snapshot and retry with a current revision |
| `-32004` | precondition not met — `create` while clustered (`{reason:"already-clustered"}`). `invite-node` while unclustered no longer errors: it auto-founds a cluster of one instead. |

Note the deliberate splits: a **malformed** PIN (not six digits) is
`-32602` (no attempt spent), but a **well-formed-but-wrong** PIN is a
`result` with `state:"failed"` and `reason:"incorrect-pin"` (the joiner also
mirrors this to the inviter, which pushes `cluster:invite-failed`). An
unreachable peer is likewise a `failed` *result* but with an empty `reason`.
A target that refuses because it is **already clustered** is a `result` with
`state:"rejected"` (`reason:"already-clustered"`), also not an error. Full table
and per-method surface in spec §7.6.

## Identity, certificates & storage

Generated on first run and reused forever after, under
`<config-dir>/cluster/` (dirs `0700`, key/pin files `0600`):

```
<config-dir>/cluster/
  identity.json            # { "node_uuid": "...", "created_at": <epoch ms> }
  node.key                 # this node's private key (PEM) — never leaves the host
  node.crt                 # this node's self-signed leaf (PEM)
  trusted/                 # the trusted-node store: ONE <peerUuid>.json per pinned peer
  members.json             # confirmed members, including optional advertisedAddresses
  pairing-receipts.json     # bounded public transcript/terminal confirmation receipts
```

The **node UUID is the trust principal** (cert subject `CN`/URI SAN), not
the hostname — hostnames are mutable, non-unique, and spoofable. Removing a
node deletes its `trusted/<uuid>.json`, which is the only revocation
mechanism (no CRL/OCSP). This keypair + `trusted/` store is the shared
trust fabric other inter-node services (`nvpair-workload-manager`,
`nvpair-errors` peer-sync) are intended to reuse. Full format in spec §7.4.

## Inter-node networking

- One HTTP listener on TCP **`14321`** serves the source-limited plain-HTTP
  admission prelude (`POST /v1/cluster/join-requests`), the **pairing
  channel** (`POST /v1/cluster/pairing`, used only during a pairing, before
  any pin exists, authenticated by the PIN) and the **mTLS trusted
  endpoints** (`POST /v1/cluster/members/remove` and `POST /v1/cluster/roster`,
  reachable only by already-paired peers). Open `14321/tcp` in the firewall
  / installer rules.
- The manager no longer runs its own `_nvpair-cluster-manager` mDNS
  advertise/browse. Its `cl` port is registered with the
  `nvpair-node-scanner` discovery daemon (carried on this node's one
  `_nvpair-node` record), and it resolves peer addresses from the
  daemon's `discovery:node-*` events (relayed by the broker); a
  Broker-supplied address always wins, so manual-only nodes are still
  invitable and discovery-less networks still work.

## Cluster trust fan-out

Pairing is point-to-point, but trust is **not** left point-to-point: every
member transitively trusts every other member, so pairing A↔B then A↔C also
leaves B and C able to mTLS each other — no manual B↔C pairing needed. On
each pairing both sides sign an **endorsement** of the peer's cert (Ed25519,
stored with the pin); members then exchange **rosters** over the mTLS
`POST /v1/cluster/roster` endpoint and pin any peer endorsed by a node they
already trust, iterating to a fixpoint.

Pairing also assigns each node a durable, monotonic **admission epoch**. A
removal is an admission-targeted proof containing the original remover's
signature/cert plus endorsements from relays. This lets a machine that was
offline verify its removal through an older trusted member even if it never
pinned the remover. Proof is persisted before de-pinning, survives restart,
has no time-based expiry, and is deleted only after a strictly newer
authenticated admission of that machine is pinned. A unanimous `403` without
such proof is not enough to make a machine leave—another member may simply
have left and dropped its pins. A bare authenticated rejector is instead
de-pinned as a departed peer after the local cluster generation is rechecked,
so the surviving machine converges to the correct solo roster.

Proofs are accepted only for a target admission this machine has actually
authenticated; a signer cannot invent a very large epoch to block future
readmission. Endorsements must also name the endorser's current admission, and
legacy timestamp-only tombstones never revoke admission-aware membership.
Pre-v2 pins are deterministically migrated to their first admission so an
offline machine from an older install remains removable, including older files
whose cluster id was absent. Pairing grants a peer's mTLS pin only after its
membership is durable, and create/join activates admission only after the
provisional self/member/pin set is durable; startup rolls back a crash-left
provisional set. Local teardown is journaled and completed on restart before
stale settings can be applied; while that journal is pending, old pins cannot
authenticate and no new admission can activate. Startup fails closed if removal
replay or cleanup cannot be persisted. Unauthenticated pre-invite sessions must
begin with a valid Type-1 EAP message, are capped at 64, and expire on the invite
TTL when they never reach a user-visible invite.

Reconciles fire on pair success and removal, with a ~30 s heartbeat backstop.
The human PIN stays the only root of trust — nothing is pinned without an
endorsement chain leading back to a real pairing. Full model in spec §7.7.

## Private peer-service bootstrap

After admission, the manager periodically reads the peer's complete current
service descriptor from `GET /v1/cluster/services` on the existing trusted
`cl` mux. Membership supplies only the locator. The response uses the serving
scanner's self identity/addresses and current broker service registrations;
neither roster entries nor reverse sessions establish service presence.
The pinned direct/reverse client retains its normal transport policy.

The descriptor keeps the scanner's canonical address first and appends missing,
sorted IPv4 overlay hints from `netpick.SharedOverlayAddresses`. Its public
address cap remains four: a full scanner list gives its last position to the
first missing overlay hint, never replacing the canonical address. Overlay
hints cannot rescue a missing or invalid scanner self row. They are interface
heuristics, not evidence that Tailscale is configured or reachable. This adds no
mDNS publication; existing publication of peer-proven overlay addresses remains.

After full decoding, the original local/peer admissions and certificate pins
are rechecked before committing the hints. All member writers share one
snapshot-through-replacement storage boundary. Hint updates hold it through
save or selective rollback, preserving unrelated address/name changes; landed
replacements retain the existing adoption policy. The resolver assembles the
recorded member endpoint, Browser endpoints with their own port, then bounded,
deduplicated persisted hints with the member port. This internal union is not
truncated to the public descriptor cap. Restart can therefore recover locators
without a browse record, but must read a fresh authenticated descriptor again.

TCP chooser confirmation is not authentication. Descriptor reads share an
eight-second overall budget and divide remaining time across remaining whole
attempts, including setup, headers and complete body/EOF. Cancellation stops
the chooser wait without caching an abandoned result; already-running finite
timeout-only probes may finish later and close their connections. It does not
promise immediate cancellation of an OS connect.

The original admission incarnation accompanies the validated result through
private `cluster:peer-services` and broker-to-scanner `discovery:peer-services`
notifications. They are consumed internally and never forwarded to clients or
installed as local reverse forwarding registrations. Scanner owns directory
mutation, normal enrichment, trust and existing discovery events. Its separate
authenticated and browse claims prevent multicast updates or expiry from
overwriting authenticated services. Valid service omission withdraws that
service. Admission changes immediately invalidate old authority; unavailable
routes do not renew the bounded metadata lifetime or remove membership.

Refresh retries cover startup, restored admission, reverse availability and
worker replacement. Requests have bounded concurrency, duration and response
size; complete validation precedes atomic scanner application. Details and
bounds are in the specification's authenticated operational-directory section.

Membership results and `nodes:changed` snapshots retain the optional hint field
through the broker's verbatim relay. Desktop and TUI membership parsers project
their existing fields and do not need these hints. The private descriptor uses
the separate strict `noderec` decoder; the service-contract checker is not a
test of unknown-field handling. Scanner directory consumers for inventory,
inference and control retain their existing freshness, eligibility and access
checks. A saved locator alone creates none of those capabilities. Synthetic
restart/TLS tests do not establish native tailnet or away-from-home operation.

## Security note

The six-digit pairing PIN is **temporary, low-entropy security debt**: it
protects against passive eavesdroppers and accidental joins but **not an
active man-in-the-middle** on the LAN, and must be replaced by a
high-entropy pairing code before cluster trust is relied on in production.
The repository [`SECURITY.md`](../../SECURITY.md) records the same boundary.

Trust is also **transitive** now (see fan-out above): a compromised member
can endorse certs the whole cluster will pin, widening the blast radius of
one bad node. This deliberate trade-off is why pairing should occur only on
a controlled network.

## See also

- [`../../SECURITY.md`](../../SECURITY.md) — pairing, trust, and mTLS
  boundaries as documented for the product.
- [`../VERSIONING.md`](../VERSIONING.md) — SemVer bump rules and
  version-stamp workflow.
- [`../readme.md`](../readme.md) — product overview and repository layout.
