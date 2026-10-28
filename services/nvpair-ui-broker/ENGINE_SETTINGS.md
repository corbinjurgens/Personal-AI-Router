<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Engine settings protocol

## Model configuration contract

`engine:inspect-model` accepts `{engine,nodeId?,modelId}` and is a read through
the existing settings relay and pinned `POST /v1/engine-settings/inspect` route.
It does not initialize a settings journal, start an engine, download content,
change trust or authorize a load. Artifact inspection reports its bounded source
assurance separately from advisory estimates and service-owned option descriptors.
The target supplies its own fresh self-memory evidence through the strict
parent-only worker envelope; public and peer requests cannot supply capacity or
native-map authority. Explicit inspection/Recalculate can use the target's
existing permission-gated helper. Observation projection adds no native action.

Optional `InspectResult.loadObservation` follows the same authenticated read
path. Shared decoding/cloning and target stamping preserve a bounded selected
summary and frozen same-load baseline, without raw output or settings authority.
Unknown/partial/unsupported summaries remain data. The complete 64 KiB reply
limit still applies, so the optional field can be omitted under response pressure;
the desktop maps absence to Unknown/`observation-omitted`. Observations never
enter settings Preview, portable patches, review digests or accepted journals.
See [native buffer observations](../../docs/model-buffer-observations.mdx).

`settings.modelLoads` is an exact-model-ID ordered collection of `requested`
portable preferences and optional target-owned finite `resolved` records.
Bindings, native build/source/preset digests and reviewed Auto inputs are target
authority, not copyable preferences. Ordinary port/launch constructors preserve
the collection. The complete settings request, not just this collection, must
fit the existing 64 KiB transport limit. Unknown/ambiguous fields are rejected.
The expanded complete preview must also fit; duplicating a large launch argument
into parsed readback produces a field/size refusal before Apply acceptance, not
a larger transport limit or a silently truncated preview.

Existing engine revision, applied-revision and sequence counters remain numeric
JSON, with marshal/decode refusal above `2^53-1`. New policy revisions, byte
quantities and nanosecond file identities use decimal strings. A consumer must
never round either representation.

Preview/apply can carry `modelChange` with `modelId`, decimal
`expectedPolicyRevision`, selected `load` set/clear leaves, selected `policy`
leaves, `previewDigest` and optional `supersedesRequestId`. The full settings
payload must echo the target baseline beside that patch. Generation numbers
retain decimal precision; locks, qualification and admission bounds are not
selected-copy fields. The target merge preserves unrestricted
`servedModels` versus an explicit empty list and validate enforced-empty options.

Typed Preview performs worker reads without the configuration or policy mutex,
then rechecks both CAS revisions and the worker source. Its digest binds the
authenticated caller, selected patch, complete target baseline, policy content,
native bindings/build, finite resolution, diff and impact. The bounded review
owner retains that exact result, original request and caller for ten minutes.
Explicit Recalculate supersedes the caller/model's previous recommendation;
expired or lost unaccepted reviews require another explicit review. Apply checks
receipt authentication/idempotency first, then validates retained full finite
settings against current baseline, source and permissions. It does not repeat
the portable patch, helper or Auto search. Fresh resource eligibility compares
the same unaccepted Auto allocation, never selects a replacement. Manual Apply
and accepted recovery do not require advisory telemetry. A running load change discloses a
whole-router restart, affected configured/observed models and the 60-second
drain before Apply. Missing loaded-model readback remains an explicit unknown.
Off stays Off; Apply never resumes a paused node. Policy-only edits use the
policy CAS and publication owner and perform no native configure or restart.

Snapshots distinguish desired settings, effective readback, coupled stage,
policy revision and unresolved error. A receipt is never evidence of runtime
readiness. CAS/pending conflicts use `-32010`, Pause `-32011`, grant denial
`-32012`, inference lock `-32014`; malformed fields use `-32602`.

The existing build-maintenance settings-entry exclusion rejects conflicting
model-settings work. It is separate from the remaining owner-evidence conditions
that keep production apply-build start unavailable; no client bypass exists.

Production memory costs remain partial. Fits and production Auto refuse until
complete-cost specification and implementation acceptance. Telemetry, native
reports and synthetic complete fixtures do not remove that prerequisite. Manual
finite configuration remains available. Copy process families remain pending the
separate copy fix, and Windows buffer-process acceptance is pending. These are
limitations, not passing validation claims; native accuracy, occupancy and
calibration remain unverified.

The existing `engine-settings-operations.json` journal contains the accepted
model operation as a subrecord, including previous/desired native settings,
frozen review, selected policy patch, before/after policy-content fingerprints,
load-only implications, caller, original resume intent and execution stage.
The stages are `accepted`, `drained`, `native-applying`, `native-persisted`,
`policy-persisted`, `activation-pending`, `releasing` and `completed`.
`acceptance-uncertain` and `completion-uncertain` retain withdrawal after
uncertain acceptance or terminal persistence.
The journal uses the existing durable file replacement helper. A failed
acceptance has no native or policy effects; a landed-but-uncertain acceptance
remains fenced for local reconciliation. Receipts fingerprint the complete
normalized request and caller. Reused IDs with different content refuse;
identical retries still check the current peer pin, grant and Pause first.

Acceptance uncertainty is saved back only after exact readback of the replacement
that landed. A different or unreadable journal is never blindly replaced. If
recording uncertainty also fails, a restart treats a bare `accepted` stage as
ambiguous and requires fresh local reconciliation before any native effect.
This also conservatively covers a crash after successful acceptance but before
its first confirmed progress stage; that window cannot prove durable acceptance
from the journal alone.
The attempt receipt becomes `failed`, with snapshot phase `applying`, coupled
stage `acceptance-uncertain` and an interrupted-before-changes explanation.
Recovery performs no native configure or policy publication from that bare stage.
A fresh local superseding preview is required; an identical request-ID retry
reads the historical failed attempt rather than replaying it.

A load-plus-policy acceptance reserves the complete policy document. Ordinary
local and peer `policy:set` calls, including Advanced, role, copy and TUI
writers, receive `-32010` naming the pending request. Availability transitions
retain their independent owner. The private journal-driven policy commit
compares content excluding only `revision`, `availability` and `pausedBy`,
preserving their current values. Before content applies the accepted patch;
after content adopts the landed revision; any third content refuses replacement.
A landed policy replacement with uncertain durability is adopted and remains
unresolved. No blind retry or rollback overwrites a newer policy.

Load-only operations validate the policy revision used by Preview, reserve no
whole policy document, and recheck the selected model's profile, serving and
residency implications before publication. Unrelated policy edits remain usable.
The existing inference lock protects generation options, mode and identity;
it does not prevent load-only configuration. Selected edits retain opaque
request-option numbers, target locks, unrelated model settings and existing
qualification/admission bounds. They create no qualification evidence. Excluding
one model from an unrestricted served list requires first selecting an explicit
list; the model patch cannot silently turn unrestricted permission into empty.

Before native effects, the current proxy must acknowledge backend withdrawal,
admission drain and strict identity withdrawal. A running apply waits up to
60 seconds for active work; acknowledged active work may be interrupted at that
bound. Missing or failed acknowledgement prevents restart. The manager receives
the frozen configuration once. Response loss is reconciled by exact current-source
native readback, not interpreted as no effect. Frozen full-settings validation
rechecks artifacts/build on replay without resolving portable Auto inputs again.
Changed overrides, bindings/build or selected policy implications require a
fresh review. A successful configure acknowledgement alone cannot complete Apply.
Exact persisted native settings do not discharge saved resume intent after a
manager restart. Resume converges through the existing ConfigureLaunch owner,
which retains current Pause, engine-lock and saved-Off vetoes. A stopped result
requires current veto evidence, rather than silently claiming running readiness.

The operation keeps discovery, local backend eligibility, resident reconciliation
and strict settings identity withdrawn through native persistence, policy
publication/activation and exact final readback. A stopped save claims no loaded
readiness. The final receipt is durable before drain release, and `releasing`
retains the reservation until release acknowledgement and final journal save.
Failed or uncertain stages retain their fences. An unresolved model operation
keeps snapshot phase `applying`, including after an attempt's receipt becomes
`failed`; `unresolvedError` explains the reconciliation requirement. This retains
the existing switch and cold-binding exclusions until the operation is settled.
Startup reconstructs them before
proxy replay/facade enablement, advertising and enabled-intent restoration.
Progressed applying records without failed receipts replay automatically.
This includes `activation-pending` after restart. Fresh local supersession
also reviews and carries any unresolved resume obligation through the existing
manager activation vetoes, even when the replacement manager is stopped.
`activation-pending` permits an explicit fresh local Preview/Apply after a repaired
proxy acknowledgement; callers need not restart the broker or invoke recovery.
Failed/conflicted attempts and ambiguous acceptance require a local
fresh Preview/Apply with a new request ID, both current revisions and
`supersedesRequestId`. Supersession retires the old reservation only in the same
durable acceptance as its replacement, carries native readback forward and
preserves unrelated newer policy content. A peer cannot clear that conflict.
Pause of either origin always wins over completion and recovery. This recoverable
sequence makes no ACID, qualification or native-hardware acceptance claim.

An unreadable ordinary settings journal retains the existing enabled-intent
restoration and proxy-policy replay behavior, with its error reported. A
recognizable unresolved or invalid model subrecord, or unresolved model reservation
already known in memory, still blocks that restoration/replay on a load error.
Valid completed model history does not turn an ordinary journal error into a new
reservation. Fully unrecognizable
bytes cannot establish whether model work once existed; this fundamental ambiguity
is reported as unknown, rather than asserted to be a model-free journal.

The broker owns combined settings operations for Ollama, LM Studio, and
llama.cpp. Each node owns its own configuration. `nodeId` selects a discovered,
currently pinned peer; omission or the local host ID selects this node. Bulk
propagation is not part of this API.

| Method | Request | Result |
| --- | --- | --- |
| `engine:get-settings` | `{engine, nodeId?}` | Full snapshot |
| `engine:preview-settings` | `{engine, nodeId?, expectedRevision, settings, modelChange?, resolution?}` | Normalized settings, errors, conflict, selected diff, review digest and runtime impact |
| `engine:apply-settings` | Preview request plus `requestId`, and reviewed `modelChange.previewDigest` for selected edits | `{revision, phase}` attempt receipt after target-owned execution; lost replies require readback |

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
is asynchronous; neither stdio reader waits for an operation. The broker serializes
whole operations but releases the configuration mutex during typed worker reads,
drain, native application and policy activation. It holds configuration then policy
ordering for durable acceptance and policy commit. Settings notifications carry
detached sequenced snapshots and perform I/O without the configuration mutex.
Each publication registers its predecessor while assigning sequences, then waits
and delivers outside that mutex. Delivery completes before reacquiring it, so
concurrent publications cannot reverse Hub/TUI baselines or invert the locks. The worker
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
paused, a local Apply or a recovered operation never resumes a stopped engine,
even one a failed Apply left with a pending resume; local settings are still
applied. Every paired Apply is refused while a pause of either origin is
pending, stored or live, including a no-op or a request for an existing receipt.
The refusal is `-32011` with
`This node is paused. Resume it on this node before managing it remotely.`
No settings revision or receipt is written. Only local Resume lifts the fence;
Get, preview, events and switch status remain readable. Advertisements and
proxy upstream eligibility are disabled during application and restored only
after readiness. Polling advertisers share the node lock.

An accepted Apply that restarts a running engine (`preview.restart`) first
drains it at the proxy: the broker pushes `node/set-engine-drain {engine,
drain:true}` so admission sends that engine no new work, waits up to 60 s for
the proxy's active count for the engine to reach zero, then applies and clears
the drain. The node configuration lock is released during the wait and the
operation lock is not, so readers stay responsive and no second Apply can start.
Ordinary proxy-only edits and ordinary edits to a stopped engine do not drain.
Typed load edits establish an acknowledged fence even while stopped so their
native/policy sequence and recovery cannot expose stale authority.

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
notifications to the local broker. Get, preview and events are reads; Apply
requires the target's remote-management grant at the `ec` gate and again under
the broker's journal locks immediately before acceptance, followed by the
common pause decision. Worker validation and lock waits cannot bypass that
decision. Apply prepares initialization, drift reconciliation and legacy
argument conversion on a private record, then rechecks authorization before
saving any of them or accepting a receipt. A refused delayed Apply leaves the
journal unchanged. A failed reply
carries `errorCode`; `-32011` and `-32012` become HTTP 409 with code
`local_veto` and `remote_management_denied`, respectively, and retain those RPC
codes at the origin. Permission denial means change permission locally on the
target, independently of conflict (`-32010`) and local veto. Other failures use
a plain-text 409. Cancellation and pin removal cancel queued
work; the broker rechecks the caller after acquiring the node lock. Once
durably accepted, the target completes under its own eleven-minute deadline even
if the initiating peer disconnects. Revocation blocks new acceptance, including
queued broker writes, but accepted Apply/switch work retains target-owned
completion and reconciliation. Progress and receipts remain readable; revocation
does not roll back effects or replace pause/local-Off cancellation. The same
fresh acceptance checks apply to switch receipt acceptance, and grants never
override private/CORS, served/resident or target-admission restrictions.

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

An unknown receipt is only a point-in-time absence observation. To prevent a
late original request, call `engine:switch-fence {nodeId?, requestId}`. Under
the same admission lock, the broker returns either an unchanged owned receipt
or `{nodeId, requestId, fenced:true}` after durable tombstone persistence. The
receipt variant is `{nodeId, requestId, fenced:false, receipt}`. Clients must
validate this exclusive union and identity; lost, malformed or failed replies
do not establish a fence. Remote calls use `engine:remote-switch-fence` and
POST `/v1/engine-switch/fence`, with authenticated caller identity and a fresh
management grant. Fencing does not cancel or start engine work.

The journal retains at most 32 caller/request-ID pairs, evicting the oldest
insertion. Duplicate fencing neither writes nor refreshes insertion order.
Retained pairs reject that caller's switch admission as `fenced` (`-32010`);
other callers retain their own namespace. Persistence advances the existing
journal generation, so the original request's expected revision stays stale
after eviction or restart. This does not reserve an evicted ID against a new
payload with a fresh revision. Read a fresh baseline after a confirmed fence
before deliberately submitting a new ID. Storage uncertainty never becomes a
fence acknowledgement merely by retrying within the same process.

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
legacy lifecycle/port mutations respect switch reservations. A pause of either origin cancels
remote execution before waiting on transition locks; a switch never resumes the
node. The pending `managementBlocked` projection reaches the manager before cancellation
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
its checkpoint. The broker-to-manager `engine:stop` request includes internal
`switchOperationId` when Off is admitted against a remote switch. Under the
manager's admission lock, a settled, released or different reservation makes
that entire scoped stop a no-op, including physical stop and intent writes.
Thus an earlier Off cannot stop a later explicit On or veto a newer switch.
Ordinary unscoped Off retains its serialized local stop semantics. A delayed
local Off response cancels only its original switch.
Failed veto delivery retains the exact operation veto for conservative recovery.
After broker restart, a durable remote-veto failure receipt also requires target
cleanup; it does not change node availability or invent local Off provenance.
Off admission rechecks reservations while waiting for ordinary settings work;
negative lookup and operation-lock acquisition are ordered with acceptance.
Off/pause admitted during retained recovery latches against that checkpoint's
operation ID. Recovery rechecks the latch after reconcile and drain release,
repeating cleanup for any new veto within the cleanup budget. Its final
metadata-only release and clearing recovery ownership share the local admission
lock; long physical cleanup runs outside that lock. Failure retains recovery.

The broker durably owns every admitted local Off and either-origin pause. Internal journal
`localObligations[operationId]` records `sourceOff`, `targetOff`, `pause`, optional
`pauseApplied`, and `discharged`; public receipts are unchanged. Admission persists before latching
or returning to the local handler, and a failed write visibly refuses the request.
After reconcile, while still holding `settingsApplyMu`, cleanup sends ordinary
unscoped `engine:stop` with internal `verifyStop: true` to the current worker for
each obligated engine. Its `SwitchEngineState` reply must prove saved Off and
termination. A successful Pause of either origin durably records `pauseApplied` for its
pending obligations before a later Resume can run. Cleanup relies on that proof
independently of current availability, preserving the owner's later Resume while
still reconciling the operation's target veto. An unapplied Pause remains
unresolved. After conservative reconciliation, the manager reservation may be
released behind the authoritative pause fence, retaining the durable recovery
receipt and undischarged obligation. Intent-neutral sleep then runs without
reservation contention; fresh readback from that same manager must verify
shutdown before `pauseApplied` and discharge are saved. Startup and replacement
perform this shutdown before acknowledging manager readiness. A later local
Resume may supersede an unapplied pause after conservative switch cleanup:
discharge records that explicit override without setting `pauseApplied`.
Discharge precedes recovery resolution; failure keeps recovery ownership.
Broker restart and manager replacement replay undischarged obligations even
when inspection returns no checkpoint, before restore/readiness. An older direct
peer start loses to a subsequently admitted Off at this settlement boundary.
Later explicit local On takes the same operation lock and runs after discharge;
the delayed scoped handler remains a no-op against later work. Historical local
Off alone never authorizes a physical reconcile stop after failed drain: manager
cleanup and release guards use operation-specific checkpoint veto flags.

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

The sibling `engine-switch-operations.json` journal keeps manual terminal
receipts with no eviction. Manual receipt capacity visibly rejects new IDs;
reserved space permits progress, terminal writes, recovery and identical retries.
Write failure halts effects; a landed but uncertain replacement is adopted for
readback while fresh mutations remain fenced. Terminal receipts are durable
before manager reservation release. Release/cleanup failure retains a recovery
fence; completion receipts are never rewritten to describe later runtime.

Broker-internal binding preparation uses the same transaction with optional
`automatic: true` on receipts, prepare envelopes and checkpoints. The public
switch request has no automatic flag. Automatic work retains its initial
absolute deadline, at most 90 seconds including acceptance, drain and activation;
each step uses the earlier existing bound. Manual work keeps its 15-minute
budget. Server readiness does not establish model readiness.
The private reconcile envelope carries `cleanupDeadline` from the broker's
bounded cleanup context without changing operation identity. Live cleanup uses
the original remaining window; later recovery gets a fresh bounded cleanup
window and never resumes forward execution.

Automatic IDs are broker-generated `binding:<brokerRunId>:<sequence>` values.
New public switch/fence submissions cannot use that namespace. Existing legacy
manual receipts and tombstones keep their exact retry/fence semantics. Status
reads retain caller authorization. Internal acceptance consumes a single-use
current-run sequence and rechecks its producing proxy session at acceptance;
pruning or restart cannot make a delayed submission fresh.

The journal retains a ring of 32 settled terminal automatic receipts, in durable
settlement order, plus unresolved work. Manual receipts and tombstones do not
consume that ring. Automatic rows become eligible after obligations are
discharged and manager release is verified. Settlement and pruning share one
durable write with monotonic generation. Recovery, failed cleanup, byte limits
and uncertain writes remain fences. Old journals read as manual without a write.

Automatic replacement requires the sole running conflicting engine to be
PAIR-controlled, non-adopted and free of reclaim uncertainty. The manager
rechecks this before reservation and source effects. Automatic reservations
prevent another engine from starting during replacement. Local Off may preempt
automatic work of either origin through the existing durable operation-specific
obligation. Delayed scoped Off cannot affect newer work. Manual local switches
retain busy behavior; lock and pause guards remain authoritative. Desktop and
TUI show `Automatic engine switch` with existing phases, request identities,
observations and refusal codes.

Failures terminate as `failed-validation`, `failed-drain`, `failed-source-stop`,
`failed-target-start`, `failed-publication` or `failed-interrupted`; pre-acceptance
refusals may have a durable `rejected` receipt. Errors carry a stable kind and
numeric code: invalid/reused ID `-32602`, configuration conflict `-32010`, local
remote veto `-32011`, remote-management denial `-32012`, a target engine locked on
its node `-32013` (kind `engine_locked`), and other operational/storage/full/busy errors `-32000`.
Manager result envelopes are inspected even when JSON-RPC itself succeeds.

Startup loads the switch journal, marks nonterminal operations
`failed-interrupted`, and conservatively reconciles manager reservations before
settings replay and enabled-intent restoration. It never resumes switch steps
or starts target as recovery. Conflicting pending settings/switch journals fail
closed. A committed target On is preserved unless a local-Off or either-origin pause veto applies
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
remaining bring-up continuation. This gate also applies on the initial spawn;
the manager's discovery subscription and node/settings state projections bypass
it so recovery receives the peer directory and authoritative pause state.
Log-level changes also reach the current worker during this interval. The
manager starts with its mutating authenticated peer-control surface fenced;
read-only queries and state deliveries remain available. Only the broker's
successful recovery sends internal `engine:switch-recovery-ready` with
`managerIncarnation`. The manager verifies its incarnation and the absence of
an unreleased checkpoint before acknowledging that same incarnation. Broker
publication still rechecks the current worker and its exit fence after the ACK.
Exhaustion keeps the manager unavailable and
reports sticky `engine-switch:recovery-blocked`; it never resumes switch steps or
silently treats the blocked replacement as usable. Corruption and uncertain
durability remain fenced.
Every constituent cleanup error is classified: simultaneous manager storage
and proxy drain-release failures retain the permanent storage fence. Recovery
makes at most three attempts, each with a 60-second cleanup context, separated
by one-/two-second backoffs. Their RPC budgets can occupy synchronous manager
spawn for approximately 183 seconds. Operation/configuration lock waits happen
before each context, so there is no unconditional end-to-end spawn-time bound.
