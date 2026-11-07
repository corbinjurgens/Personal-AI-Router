<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Engine settings protocol

## Model configuration contract

`engine:set-stated-memory` accepts `{engine,nodeId?,modelId,configurationKey,
hostBytes?,deviceBytes?,sharedBytes?}`. `engine:clear-stated-memory` accepts the
same identity without byte fields. Both return an inspection reply. Byte fields
are canonical unsigned decimal strings and must cover exactly the applied
`configurationPools`. The inspected configuration key is independent of estimate
availability; stale keys return conflict `-32010`, asking the client to reinspect.

The manager owns private `model-memory-profiles.json` beside its buffer history,
through the existing durable-file replacer (0600). Records are bounded by 256
entries and 1 MiB. An unreadable or corrupt store is preserved and reports
`memoryProfilesUnavailable`; writes refuse. Inspection returns `memoryProfiles`
with the current stated record first, followed by bounded history (at most 8),
dropping non-current rows to respect the complete reply budget. A replacement
that landed with uncertain durability is adopted in memory and reports that
uncertain outcome; reinspect before retrying. Unsupported file entries are
preserved, and the complete mutation reply is validated before publication.
Statements do not overlay estimates or change policy, budgets, allocation or
launch state.

Remote set and clear use the strict settings relay union and authenticated
mutation routes `POST /v1/engine-settings/set-stated-memory` and
`POST /v1/engine-settings/clear-stated-memory`, through the existing direct or
reverse transport. The destination broker rechecks remote-management permission and
Pause; only the destination store changes. Local recording is
allowed while paused. Public requests cannot provide target-memory authority;
the private worker envelope supplies target evidence and disables native helper
initialization for recording.

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

`settings.modelLoads` is a profile-keyed ordered collection. Each load carries
`profileKey`, its exact artifact `modelId`, `requested` portable preferences and
optional target-owned finite `resolved` records. Several profiles can use one
artifact with different load settings; llama.cpp preset sections and served ids
use the profile key. An existing profile keeps its engine and artifact selection.
Artifact bindings, native build/source/preset digests and reviewed Auto inputs
are target authority, not copyable preferences. Ordinary port/launch constructors preserve
the collection. The complete settings request, not just this collection, must
fit the existing 64 KiB transport limit. Unknown/ambiguous fields are rejected.
The expanded complete preview must also fit; duplicating a large launch argument
into parsed readback produces a field/size refusal before Apply acceptance, not
a larger transport limit or a silently truncated preview.

Existing engine revision, applied-revision and sequence counters remain numeric
JSON, with marshal/decode refusal above `2^53-1`. New policy revisions, byte
quantities and nanosecond file identities use decimal strings. A consumer must
never round either representation.

Preview/apply can carry `modelChange` with `profileKey`, artifact `modelId`, decimal
`expectedPolicyRevision`, selected `load` set/clear leaves, selected `policy`
leaves, `previewDigest` and optional `supersedesRequestId`. The full settings
payload must echo the target baseline beside that patch. Generation numbers
retain decimal precision; locks, qualification and admission bounds are not
selected-copy fields. The target merge preserves unrestricted
`servedModels` versus an explicit empty list and validate enforced-empty options.
Omitting `profileKey` selects the identity key equal to `modelId`. A Remove uses
`remove: true` and no load or policy patch. It removes the selected typed native
load when present, and the policy profile and role memberships. With a typed
load, served and resident references match the engine and profile key.
Without a typed load, only the resident reference matching the engine and the
profile's native model is removed; served entries stay even when
that native model equals the key. Empty role declarations remain. Removal
without a typed load uses the policy-only operation and leaves native settings
unchanged. Removing the final load leaves
default engine mode, without an empty managed-router marker.

The coupled policy stage creates or updates the keyed profile and derives
`contextTokens` from that load's resolved per-slot context and `modelIdentity`
from the accepted keyed preview identities. These fields are target-owned;
they are not selected portable policy leaves. Raw policy writes preserve the
stored identity and derive typed-profile context from the broker's authoritative
load snapshot. Role membership edits address the node's declared roles and keep
their ordered profile-key lists; undeclared roles are refused.

Typed Preview performs worker reads without the configuration or policy mutex,
then rechecks both CAS revisions and the worker source. Its digest binds the
authenticated caller, selected patch, complete target baseline, policy content,
native bindings/build, finite resolution, diff and impact. The bounded review
owner retains that exact result, original request and caller for ten minutes.
Explicit Recalculate supersedes the caller/engine/profile's previous recommendation;
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
profile implications, caller, original resume intent and execution stage. Typed Remove
also retains `reviewedModelChange` in this same subrecord: the original native
worker's Remove-only selection, profile key, artifact id, expected policy
revision and native preview digest. The public broker digest remains in the
accepted request and review; it cannot substitute for native omission evidence.
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

Every coupled model acceptance reserves the complete policy document, including
load-only changes that publish derived profile metadata and Remove. Ordinary
local and peer `policy:set` calls, including Advanced, role, copy and TUI
writers, receive `-32010` naming the pending request. Availability transitions
retain their independent owner. The private journal-driven policy commit
compares content excluding only `revision`, `availability`, `pausedBy` and `pauseMode`,
preserving their current values. Before content applies the accepted patch;
after content adopts the landed revision; any third content refuses replacement.
A landed policy replacement with uncertain durability is adopted and remains
unresolved. No blind retry or rollback overwrites a newer policy.

Load-only operations validate the policy revision used by Preview and recheck
the selected profile's serving and residency implications before publication.
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
For typed Remove, every frozen `engine:preview-launch` uses the existing private
`WorkerRequest[Request].reviewedModelChange`; its `request.modelChange` is absent
and `request.settings` is the already compiled accepted configuration. The direct
`engine:configure-launch` payload remains `Configure`, with the same evidence in
`reviewedModelChange`. The private retained source saves `NativeDigest` before
public wrapping and includes it and the selection in retention-byte accounting
and source checks. Public requests cannot supply this parent-only field.

Initial Apply, journal recovery and same-key local supersession after native
cleanup has landed forward the original native proof unchanged. They never
reapply the portable Remove to the frozen settings or reconstruct proof from
the wrapped public digest or current state. Missing, invalid or wrapped native
proof refuses; the manager validates the original digest against the complete
native source before accepting the selected omission. Evidence travels only as
transient worker metadata and the existing accepted journal subrecord, never in
settings `Config` or runtime overrides. This protocol description does not claim
installed-engine or platform acceptance.
Changed overrides, bindings/build or selected policy implications require a
fresh review. A successful configure acknowledgement alone cannot complete Apply.
Exact persisted native settings do not discharge saved resume intent after a
manager restart. Resume converges through the existing ConfigureLaunch owner,
which retains current Pause and saved-Off vetoes. A stopped result
requires current veto evidence, rather than silently claiming running readiness.

The operation keeps discovery, local backend eligibility, resident reconciliation
and strict settings identity withdrawn through native persistence, policy
publication/activation and exact final readback. A stopped save claims no loaded
readiness. The final receipt is durable before drain release, and `releasing`
retains the reservation until release acknowledgement and final journal save.
Failed or uncertain stages retain their fences. An unresolved model operation
keeps snapshot phase `applying`, including after an attempt's receipt becomes
`failed`; `unresolvedError` explains the reconciliation requirement.
If policy persistence fails after a typed Remove persisted native settings,
the removed load stays absent while the previous policy stays saved. The failed
attempt remains fenced and requires a fresh local superseding review; it does
not roll the native change back or release the old policy for ordinary writes.
Startup reconstructs them before
proxy replay/facade enablement, advertising and installed-engine restoration.
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

An unreadable ordinary settings journal retains the existing installed-engine
restoration and proxy-policy replay behavior, with its error reported. A
recognizable unresolved or invalid model subrecord, or unresolved model reservation
already known in memory, still blocks that restoration/replay on a load error.
Valid completed model history does not turn an ordinary journal error into a new
reservation. Fully unrecognizable
bytes cannot establish whether model work once existed; this fundamental ambiguity
is reported as unknown, rather than asserted to be a model-free journal.

The broker owns combined settings operations for built-in and fronted descriptor engines.
Each node owns its own configuration. `nodeId` selects a discovered,
currently pinned peer; omission or the local host ID selects this node. Bulk
propagation is not part of this API.

| Method | Request | Result |
| --- | --- | --- |
| `engine:get-settings` | `{engine, nodeId?}` | Full snapshot |
| `engine:preview-settings` | `{engine, nodeId?, expectedRevision, settings, modelChange?, resolution?}` | Normalized settings, errors, conflict, selected diff, review digest and runtime impact |
| `engine:apply-settings` | Preview request plus `requestId`, and reviewed `modelChange.previewDigest` for selected edits | `{revision, phase}` attempt receipt after target-owned execution; lost replies require readback |

`engine` is a built-in engine ID, such as `llamacpp`, or a registered descriptor
engine ID that this broker fronts, such as `llamacpp-custom`. Both use the same
settings journal, revision checks, apply owner and snapshot publication.
After a broker restart, facade preparation restores the journal's explicit
settings flag and server/proxy ports for both engine kinds. The explicit proxy
port overrides the proxy's persisted port file; a bind failure is reported
without automatically moving the chosen port. Descriptor engines without
explicit settings retain their ordinary planned/persisted facade port selection.
`settings` contains
`serverPort`, `proxyPort`, `launchText`, and optional `workingSetLimitMiB`.
The limit is an integer from 256 through 1048576, accepted only by Windows
llama.cpp engines in process mode. Absence or null removes the limit. Preview
returns field errors under `workingSetLimitMiB`, includes changes in `diff`,
and marks a running engine for restart. Apply persists `working_set_limit_mib`
through the manager's runtime write. Snapshots project the manager's
`workingSetLimitStatus`: `applied`, `unverified`, `not-running`, or `unenforced-adopted`,
omitted when no limit is saved. Adopted processes require restart for descendant
enforcement. Root cap failure fails start. Job notifications trigger immediate
member sweeps; a 250 ms target sweep cadence covers lost notifications. Processes
can fault in pages before their first cap, and a late cap trims the working set.
An unopenable member terminates the engine job; a live member whose cap fails is
terminated (see the manager spec). The
`launchText` field contains
arguments and leading environment assignments, without the executable or
startup subcommand. The argument grammar is
[`pair-arguments-v1`](../nvpair-engine-manager/LAUNCH_TEXT.md). Preview does not
change component configuration or runtime. A snapshot read may persist the
initial revision baseline or reconcile an external component change.

`unverified` means this manager started the running engine with the saved value,
but its containment has not verified every live member: a member could not be
capped (it is terminated, and the status returns to `applied` after the next clean
sweep), or the job could not be read (the engine job is terminated). `applied`
means a full verification (the root cap before resume, then each sweep on a
250 ms target cadence) completed within the last second and found every live
member capped. A stalled or stopped watcher also reports `unverified`.

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
PAIR services, other configured engines/proxies and occupied listeners.
Sibling checks include fronted descriptor engines' live facade ports and saved
ports in the settings journal or proxy-port files, even while their proxy is
stopped. Automatic facade fallback also excludes descriptor siblings' saved,
planned and live facade ports.
Only this engine's current running port and its proxy listener can be reused
for a swap. Revalidation happens at Apply; OS bind remains the final arbiter of
a competing process. Adopted process engines are read-only; command-mode
engines require their official stop path.

A normalized no-op changes no runtime. Proxy-only edits do not restart the
engine. A stopped engine remains stopped. Running launch/server changes stop
and start once when the node is available. While the node is
paused, a local Apply or a recovered operation never resumes a stopped engine,
even one a failed Apply left with a pending resume; local settings are still
applied. Every paired Apply is refused while a pause of either origin is
pending, stored or live, including a no-op or a request for an existing receipt.
The refusal is `-32011` with
`This node is paused. Resume it on this node before managing it remotely.`
No settings revision or receipt is written. Only local Resume lifts the fence;
Get, preview and events remain readable. Advertisements and
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
`settingsdrain.go` requires the drain acknowledgement from the current proxy
run. When waiting for active work before a running engine resumes, it also
requires fresh `node/admission-snapshot {}` readback from that same run. Missing
acknowledgement, source replacement or failed required readback refuses progress.
The push-slot wait shares the caller deadline; the proxy/source fence is rechecked
before effects and drain release.

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
and suppresses automatic component rewrites. Unreadable data blocks automatic
recovery. The journal retains up to 256 terminal receipts
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
queued broker writes, but accepted Apply work retains target-owned
completion and reconciliation. Progress and receipts remain readable; revocation
does not roll back effects or replace pause/local-Off cancellation. Grants never
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
