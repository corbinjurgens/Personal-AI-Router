<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Engine settings protocol

The broker owns combined settings operations for Ollama, LM Studio, and
llama.cpp. Each node owns its own configuration. `nodeId` selects a discovered,
currently pinned peer; omission or the local host ID selects this node. Bulk
propagation is not part of this API.

| Method | Request | Result |
| --- | --- | --- |
| `engine:get-settings` | `{engine, nodeId?}` | Full snapshot |
| `engine:preview-settings` | `{engine, nodeId?, expectedRevision, settings, resolution?}` | Normalized settings, errors, conflict, restart/rebind summary |
| `engine:apply-settings` | Preview request plus `requestId` | `{revision, phase}` acknowledgement |

`engine` is `ollama`, `lmstudio`, or `llamacpp`. `settings` contains all three
fields: `serverPort`, `proxyPort`, `launchText`. The last field contains
arguments and leading environment assignments, without the executable or
startup subcommand. The argument grammar is
[`pair-arguments-v1`](../nvpair-engine-manager/LAUNCH_TEXT.md). Preview does not
change component configuration or runtime. A snapshot read may persist the
initial revision baseline or reconcile an external component change.

Each full snapshot includes desired `settings`, `revision`, `appliedRevision`,
`phase` (`idle`, `applying`, `succeeded`, `failed`), `requestId`, `error`,
`effectiveServerPort`, `effectiveProxyPort`, `running`, `adopted`, `editable`,
`reason`, `format`, `epoch`, and `sequence`. A configured runtime server port
does not imply a listening server: consult `running`. The operation receipt is
not a state update. Consume `engine:settings-changed` or fetch the full snapshot
after a response loss. A renderer must retain dirty drafts and reject stale
baseline revisions, even when only arguments changed.

Declared CORS origin lists, switches and explicit booleans can only change on
the engine's owning node. The broker derives an internal `preserveCORS` preview
guard from the authenticated caller, overriding any client-supplied value. It
rechecks canonical CORS policy under the apply lock before accepting a revision;
the early remote ingress check alone is insufficient when operations overlap.

## Ownership, ordering and failures

The node configuration mutex precedes engine operation locks. Worker dispatch
is asynchronous; neither stdio reader waits for an operation. The broker holds
the node lock across validation, journal acceptance and application. The worker
holds its engine lock across override persistence, stopping with the old launch
context, installing the new context, the narrow proxy rebind callback and one
start/readiness wait. The callback validates an active operation token and
acquires no node configuration lock. Legacy public port setters enter this same
operation and retain their response shape. Legacy lifecycle port overrides and
automatic port reconciliation share the node lock and revision reconciliation.

Validate both ports together against the unchanged counterpart, registered
PAIR services, aliases, other configured engines/proxies and occupied listeners.
Only this engine's current running port and its proxy listener can be reused
for a swap. Revalidation happens at Apply; OS bind remains the final arbiter of
a competing process. Adopted process engines are read-only; command-mode
engines require their official stop path.

A normalized no-op changes no runtime. Proxy-only edits do not restart the
engine. A stopped engine remains stopped. Running launch/server changes stop
and start once without rewriting explicit enabled intent. While the node is
paused, an Apply or a recovered operation never resumes a stopped engine, even
one a failed Apply left with a pending resume; the settings are still applied.
A paired node's Apply that would resume one while the node is paused on the
node itself is refused with the local veto (`-32011`) before acceptance. Advertisements and
proxy upstream eligibility are disabled during application and restored only
after readiness. Polling advertisers share the node lock.

An accepted Apply that restarts a running engine (`preview.restart`) first
drains it at the proxy: the broker pushes `node/set-engine-drain {engine,
drain:true}` so admission sends that engine no new work, waits up to 60 s for
the proxy's active count for the engine to reach zero, then applies and clears
the drain. The node configuration lock is released during the wait and the
operation lock is not, so readers stay responsive and no second Apply can start.
Proxy-only edits and edits to a stopped engine do not drain.

Validation, stale revisions and journal-write failures cause no runtime
mutation. Once accepted, desired settings remain saved if the engine or proxy
fails. The failed snapshot exposes actual runtime facts and the prior applied
revision; correction/retry is another explicit Apply. Component writes and OS
listeners are recoverable steps, not an ACID transaction. The failed snapshot
preserves the worker's error message. Settings-driven start failures use this
result instead of also raising global lifecycle errors. Normal start failures
still raise a lifecycle error; the exit watcher begins after readiness to avoid
reporting the same startup failure twice.

`engine-settings-operations.json` in the app data directory holds previous and
desired settings, revision, resume intent and operation receipts. Component
engine overrides and proxy-port files remain their owners' configuration.
Acceptance is written and synced before component changes. Interrupted applying
records replay before enabled-engine restoration. Explicit settings override
automatic facade defaults on later startup. Unreadable journal data is preserved
and suppresses automatic component rewrites. On first upgrade, a valid existing
proxy-port choice is preserved as explicit when it differs from the engine port
(except LM Studio's obsolete 1235 default). Old stores lack choice provenance,
so a historical automatic move is conservatively preserved too. Unreadable data
blocks automatic recovery. The journal retains up to 256 terminal receipts
per engine; evicted retries must still pass the original expected revision.

## Paired transport

The existing engine-control listener exposes pinned-mTLS POST endpoints
`/v1/engine-settings/get`, `/preview`, `/apply` and GET `/events`. Requests are
limited to 64 KiB and cannot forward another hop. The authenticated caller ID is
carried through correlated `engine:settings-request` / `engine:settings-reply`
notifications to the local broker. A failed reply carries `errorCode`; the
route answers `-32011` as HTTP 409 `{"error", "code": "local_veto"}`, which the
origin returns as `-32011`, and any other failure as a plain-text 409. Cancellation and pin removal cancel queued
work; the broker rechecks the caller after acquiring the node lock. Once
durably accepted, the target completes under its own eleven-minute deadline even
if the initiating peer disconnects.

`engine:settings-projection` seeds the worker's subscription hub with full local
snapshots. A subscription registers and receives its baseline under one lock.
Slow readers coalesce full snapshots, avoiding missing-field patches. There are
at most 64 subscribers and 64 outbound peer streams. Each stream checks trust
on every frame/heartbeat, bounds writes to five seconds and emits one-second
heartbeats. Clients reconnect after disconnection and discard older sequences
within an epoch; a fresh authority epoch replaces the baseline. Settings never
enter mDNS/public discovery metadata. Older/offline/unpaired peers are read-only
with an explicit unavailable state in the editor.

## Validation

Tests cover literal argv, parser fuzz/table cases, managed-field validation,
real process no-op/proxy-only/launch changes, ownership rejection, persistence
failure, journal recovery, revision conflicts/deduplication, queued cancellation,
three-node pinned TLS subscriptions, revocation/reconnect, bounded slow readers,
proxy persistence failure preserving its listener, renderer ordering and log
redaction. The rendered editor is also exercised with isolated browser fixtures.
Native vendor and Linux/macOS runtime verification remain release-platform
checks; Windows fixture results do not certify vendor-option compatibility.

## Engine switching

The target broker coordinates `engine:switch` independently of settings Apply.
Submit `{nodeId?, requestId, sourceEngine, targetEngine, expectedRevision}`;
the result is `{receipt}` after durable acceptance. The target must be prepared,
stopped and supported by a ready inference facade. A running target fails
validation before drain. No download, provisioning, model loading or role/tier
change is implied. Completion means server readiness and facade publication.

Read `engine:switch-status {nodeId?, requestId?, afterSequence?}` for
`{configurationRevision, activeRequestId?, receipt?, observed?}`. Omit requestId
to obtain the opaque `e3v1:` configuration token for a new request. It combines
node identity, journal generation, manager incarnation, registry/runtime and
engine configuration digests, settings revisions and desired intent/provenance.
Runtime health and idle Wake/Sleep do not change the token. The manager's own
configuration fence is separate and passes unchanged in internal envelopes.
The same caller and payload under the same request ID retrieve the original
operation before revision checking; another caller or payload cannot reuse it.

`afterSequence` requires requestId and waits at most 25 seconds for a newer
receipt, without holding operation/configuration locks. A timeout returns the
unchanged receipt. Each durable sequence emits
`engine:switch-changed {nodeId, requestId, sequence}`. Paired requests use pinned
engine-control routes and authenticated caller identity. Manager-originated progress notifications
retain the remote target node ID when forwarded to local clients. Relay
`engine:switch-cancel` retires only its waiter. Accepted execution belongs to
the target and continues after caller disconnect, with a 15-minute total bound.

The normal persisted phases are `accepted`, `validating`, `draining`,
`source-stopping`, `source-off`, `target-starting`, `target-ready`, `completed`.
Each phase is saved before its effect and manager observations are saved after
the effect. Source Off is committed before stopping; target On is committed by
the manager only after readiness. A failed target start never restores source.
One switch runs per node under the shared settings operation lock. Settings and
legacy lifecycle/port mutations respect switch reservations. Local pause cancels
remote execution before waiting on transition locks; a switch never resumes the
node. The pending local pause projection reaches the manager before cancellation
or transition-lock waiting. Local Off of either involved engine bypasses the
broker operation lock during a remote switch, including recovery reservations,
and reaches the manager's existing stop admission. That owner durably saves
local Off and cancels remote startup before waiting for lifecycle locks. The
switch fails with target Off and never restores source. Local Off during a local
switch remains a busy refusal; unrelated engines do not veto the switch.
The broker latches the veto for that accepted operation before waiting for the
stop response. Its terminal-completion gate checks the latch, including after
facade publication, and conservative cleanup produces a failure receipt.
The internal reconcile envelope carries that operation's local Off engine or
pause veto, so the manager durably admits it before committed-target cleanup
even when the local lifecycle request has not arrived. A veto latched during
reconcile requires another cleanup pass before terminal release, or retained
recovery ownership if cleanup cannot finish within its budget. Manager release
refuses a committed target with a durable veto recorded after reconcile read
its checkpoint. A delayed local Off response cancels only its original switch.
Failed veto delivery retains the exact operation veto for conservative recovery.
After broker restart, a durable remote-veto failure receipt also requires target
cleanup; it does not change node availability or invent local Off provenance.
Off admission rechecks reservations while waiting for ordinary settings work;
negative lookup and operation-lock acquisition are ordered with acceptance.

Switch drain is strict and separate from the settings restart drain. Withdraw
source discovery and its local inference endpoint, acknowledge source drain,
then invalidate source and target through `node/invalidate-engine` with the
operation ID. Both `{ok, runId}` acknowledgements must match the live proxy.
Only a fresh `node/admission-snapshot` requested afterward, from that same run,
with zero source active/held work permits source-stop. Missing evidence, lost
acknowledgement, proxy replacement, push-slot contention or the 60-second drain
deadline fails without stopping source. Pending invalidations replay before any
facade enables after proxy restart. Invalidation does not grant strict readiness
or erase uncertain capacity markers.

The sibling `engine-switch-operations.json` journal keeps immutable terminal
receipts with no automatic eviction. A full journal visibly rejects new IDs;
reserved space permits progress, terminal writes, recovery and identical retries.
Write failure halts effects; a landed but uncertain replacement is adopted for
readback while fresh mutations remain fenced. Terminal receipts are durable
before manager reservation release. Release/cleanup failure retains a recovery
fence; completion receipts are never rewritten to describe later runtime.

Failures terminate as `failed-validation`, `failed-drain`, `failed-source-stop`,
`failed-target-start`, `failed-publication` or `failed-interrupted`; pre-acceptance
refusals may have a durable `rejected` receipt. Errors carry a stable kind and
numeric code: invalid/reused ID `-32602`, configuration conflict `-32010`, local
remote veto `-32011`, and other operational/storage/full/busy errors `-32000`.
Manager result envelopes are inspected even when JSON-RPC itself succeeds.

Startup loads the switch journal, marks nonterminal operations
`failed-interrupted`, and conservatively reconciles manager reservations before
settings replay and enabled-intent restoration. It never resumes switch steps
or starts target as recovery. Conflicting pending settings/switch journals fail
closed. A committed target On is preserved unless a local Off/pause veto applies
to the unreleased remote reservation; that veto cleans up target to Off too.
An uncommitted partial target launch is cleaned up. Unknown cleanup remains
visibly recovery-required. Terminal receipts remain historical facts; current
observations are reported separately.

Forward prepare/source-stop/target-start calls use the original worker handle
and carry its explicit `managerIncarnation` in the internal operation envelope.
A manager rejects inherited forward steps with a coded `interrupted` error,
including retries of previously acknowledged steps; reconcile and release still
accept the original envelope. Manager exit or replacement synchronously cancels
active forward work. The broker persists `failed-interrupted`, releases its
operation lock, then reconciles/releases authoritative checkpoint evidence on
the replacement before exposing it to restore or other forward callers. Recovery
never waits behind a live switch that is itself waiting for replacement recovery.
Older checkpoints without the incarnation remain readable for cleanup only.
The worker exit owner fences execution before closing its peer/Done channel or
notifying the supervisor. Terminal persistence shares that fence's gate and
also checks the pinned worker's already-closed Done channel. This orders observed
exit against completion, rather than claiming atomicity with an unobservable
physical crash.

Transient inspection, reconcile and drain-release errors are separate from
permanent journal/storage uncertainty. Replacement recovery retries conservative
cleanup on the same still-current worker with bounded backoff; the replacement
stays hidden throughout. Success clears only the retryable error and permits the
remaining bring-up continuation. Exhaustion keeps the manager unavailable and
reports sticky `engine-switch:recovery-blocked`; it never resumes switch steps or
silently treats the blocked replacement as usable. Corruption and uncertain
durability remain fenced.
Every constituent cleanup error is classified: simultaneous manager storage
and proxy drain-release failures retain the permanent storage fence. Recovery
makes at most three attempts, each with a 60-second cleanup context, separated
by one-/two-second backoffs. Their RPC budgets can occupy synchronous manager
spawn for approximately 183 seconds. Operation/configuration lock waits happen
before each context, so there is no unconditional end-to-end spawn-time bound.
