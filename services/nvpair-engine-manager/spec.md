<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Microservice: Engine Manager (`nvpair-engine-manager`)

## Host-local credentials

Only broker stdio accepts `engine:set-credential` (`{engine,credential}`) and
`engine:credential-state` (`{engine}`). They return
`{supported:boolean,configured:boolean}` and never a key. The host runtime's
`user_credential` flag owns support. State for an unsupported known engine is
`{supported:false,configured:false}`; unsupported Set and malformed names,
keys or any `node`/`nodeId` selector return -32602 with fixed text
`Invalid engine or credential.` Empty Set means Delete. Credentials are empty
or at most 512 printable non-space ASCII bytes. Native failures return -32000
with fixed text:
`Credential store is unavailable. Unlock the operating system credential store and try again.`
No false absence is returned on failure.

The executor owns a lazy shared native secret store adapter, fixed purpose
`engine-api-key` and validated engine id. Its credential mutex serializes load,
Set and Clear through native completion, separately from engine state. Snapshot
`st.mu` first, release it before credential I/O, and never take `st.mu` while
holding the credential mutex. Successful mutations publish only after native
completion; failed mutations retain the previous confirmed entry and force a
Get reconciliation before eligibility. Refresh failure is unavailable, never
stale key use. Unknown/unavailable reads retry at the next operation and
health/presence sweep. Confirmed presence or absence is cached until successful
Set/Clear or executor teardown. External store edits require service restart.
In-flight snapshots can finish with their previous key. Teardown drops cache
and adapter references without a native write; no secure-erasure claim applies.
There is no secondary plaintext/settings/override store or retry goroutine.

`candidateCredential` binds selection to a valid operation port independently
of `running` and `proc`, so cold command readiness and first adoption use saved
keys. An owned prepared Strata launch credential takes priority. A saved key is
eligible only when the host runtime declares support. Broker retrieval and HTTP
actions additionally require running on the exact current port; stopped engines
export nothing, and Strata export retains owned-process eligibility. CLI actions
receive no user credential. Probe helpers take an explicit immutable candidate
snapshot; they do not select a credential. Readiness, health, presence and
diagnosis use the same selection owner, preserve HTTP/JSON identity and bounded
readiness, and propagate unavailable instead of a misleading no-key/app/process
diagnosis. Only an attached key receiving a declared auth status yields
`The API key saved in PAIR was rejected by <Name>. Enter its current key, or turn off API authentication in <Name>'s server settings.`

Probes and actions attach keys through the same helper. Before network I/O the
initial URL must have exactly scheme `http`, hostname `127.0.0.1`, explicit
parsed port equal to the snapshot port, and no userinfo, fragment or opaque
form. A private client clones `*http.Transport`, clears Proxy and refuses
redirects (including same-origin); unsupported injected transports fail closed.
Credential-bearing failures expose fixed operation/status text, never response
bodies or raw transport errors. Successful action JSON is decoded and the
active key replaced in field names/string values before re-encoding; non-JSON
text is redacted before wrapping. Unauthenticated engine behavior is preserved.

Local authorization is the existing `ipc.ListenPrivate` same-user 0600 Unix
socket or user-SID Windows pipe, then service multiplexing, broker stdio and
manager stdio. The ec pinned mTLS direct/reverse registry exposes no credential
or generic JSON-RPC route, even to peers allowed to manage engines. Settings,
policy and switch relays do not carry credentials. The separate broker-only
`internal:local-engine-credential` path is intentionally retained for the local
proxy, with the broker refusing public client retrieval before generic relay.
No credential state notification is emitted. Public status/diagnostics/models
and peer snapshots carry no keys. Same-user code is inside the trust boundary.
Native OS/session availability and persistence acceptance remain untested by
the fake and mTLS fixture checks.

## Model settings contract

`engine:inspect-model` accepts `{engine,modelId}` at the target-local worker.
The public broker request may also select `nodeId`; the broker resolves that
target and supplies authenticated node identity in the reply. The worker rejects
a nonempty node selector, rather than labelling local results as a remote node.
`Executor.InspectModel(ctx, request)` validates exact identity and cancellation,
then composes the artifact, descriptor and advisory-estimate seams. The temporary
artifact/facts and estimate bodies return `unavailable` with reasons and no
paths, bindings, sizes, native build, device mapping, memory or allocation claim.
They do not call lifecycle, inventory, download, model hashing or native helpers.

`engine:remote-inspect-model` accepts `{engine,nodeId,modelId}` with a required
remote target. It uses POST `/v1/engine-settings/inspect`, a `controlRead` route on
the existing pinned listener. Incoming fields use the strict inspect decoder,
the body limit remains 64 KiB, and any supplied node selector is cleared before
the target broker relay. Pin admission and active-call cancellation retain their
existing owners. Reads require no remote-management mutation grant. The reply
must pass the shared strict decoder and match engine/model/target node exactly.
Grant, Pause, revision and inference-lock refusals retain -32012, -32011, -32010
and -32014 through the existing coded transport. Invalid local ingress is
-32602; target validation is HTTP 400 on the pinned engine-settings endpoint,
which the remote client preserves as invalid parameters (-32602).

The shared correlated relay has two branches: inspect has `Inspect` non-nil and
an empty value `Request`, omitted from JSON; get/preview/apply/rebind have their
value `Request` and nil `Inspect`. `settings.DecodeRelay` validates this union.
The existing executor `settingsParent` function type remains unchanged. Only its
private inspect adapter uses `Request.Resolution` to carry the model ID to
`settingsRelay.callInspect`; the emitted wire record contains the real inspect
shape, not a settings request. This avoids a second parent callback or endpoint.

Llama descriptors are derived from every actual `llamaServingOptions` entry,
including native rejection bits and all aliases/environment names. Model field
metadata adds labels, portable ranges/enums, default provenance, ownership,
phase and identity/estimate effects. No second native flag table lives in shared
code or the UI. The [full matrix](MANIFEST.md#typed-model-load-contract-and-option-matrix)
also covers generation, threads, template, reasoning, grammar and router owners.
Unsupported rows are explicit refusals, not permission for raw extras. B support
describes the manual field vocabulary; acceptance remains broker-owned.
Inherited finite-adapter defaults are labelled and never materialized as intent.

Worker load-only `modelChange` preview traffic reaches target-baseline validation
and bounded conversion/compilation, returning diffs and a review digest without
persistence or activation. Parent Configure traffic carries frozen reviewed
native settings and validates them before effects. Selected-patch acceptance,
policy writes and preservation of ordinary public baseline echoes belong to the
broker. Worker policy-edit traffic remains refused. Auto validates preferences
and returns no resolution, review identity, candidate or default trades.
Estimates and Auto remain unavailable until their respective owners implement
them. Missing or changed artifacts and unreviewed resolutions remain refusals.

Direct previews compile typed records without persistence or activation.
`launchStateLocked` clones stored requested/resolved records; configured native
readback is distinct from accepted settings and strict declaration evidence.
`persistRuntimeServing` can replace modelLoads with ports/launch/serving in the
existing atomic override, including the host-platform precedence mask. Omitted
replacement preserves the collection; an explicit empty array clears it.
Unrelated fields and numeric lexemes survive the replacement.

The full Runtime selects typed compilation at the existing prepared-serving
owner, including Serving=nil. Exact Task 5 artifact resolution, bounded GGUF
facts and repeated binding checks run under engine content-read admission.
The compiler emits sorted complete model sections, checked finite padded context
(per request times native slots), non-unified KV, fitting off and disabled
startup loading. Reset removes requested leaves; inherited finite adapter
defaults are regenerated, never retained in a hidden old source. Explicit CPU
does not require estimate facts; finite layers/native-default resolution needs
bounded layer facts. Only f16 KV precision is currently mapped; unmapped
precision and Auto are field refusals. Batch compatibility is checked against
the pinned effective native defaults. Metadata context is a fact, not a
universal training hard cap or permission to add unsupported scaling controls.

Typed launch conflicts with external presets and overlapping CLI, explicit env
or inherited native allocation sources. Router capacity and unrelated router
options remain at their existing owners. The complete-router conversion
preview parses the bounded subset, returns field/source diffs and source digest,
and refuses unrepresentable sections/options without partial takeover. Inherited
allocation sources that settings cannot remove are refused. Generated preset
paths are outputs, never approval inputs. Native executable bindings, canonical
typed inputs, ordered sampled artifact bindings and generated bytes participate
in identity. Pre-spawn rechecks and the existing immutable generation lease
apply on fresh/manual starts and crash relaunches. Existing declarations retain
their exact model/artifact/output/memory checks against generated bytes; typed
settings do not create serving declarations, content hashes or qualifications.

Direct ConfigureLaunch detects typed changes and revalidates their inputs before
the existing stop, persistence, rebind and conditional restart sequence. Accepted
external-preset conversion reconstructs the complete current router, preserves
its artifact set and unrelated launch settings, and validates the exact reviewed
native resolution. Source bindings are rechecked before stop and restart;
native build and artifact identity are rechecked immediately before spawn.
Conversion records the complete reviewed external-source identity in the
resolved provenance pair, included in the native source-input digest. A source
changed since preview is refused before stopping. Persisted conversion provenance
does not require the removed external preset for subsequent readback or starts,
and cannot authorize an Auto resolution.
SetPort uses the same owner for typed settings. Worker RPC accepts frozen native
settings, while combined-policy requests and Auto remain guarded. Broker
acceptance, drain and coupled policy application stay at their existing owner.
Load previews describe whole-router scope, affected configured models, the
60-second drain bound, possible interruption and whether the engine stays Off.
Unobserved loaded models remain explicit unknowns for broker reconciliation.
Unchanged typed settings do not restart an owned process. Saving Off, paused or
locked settings never activates it; adopted processes remain read-only.
No typed settings create qualification. Apply-build must
add this mutation to its future maintenance exclusion; no symbol exists today.

## Installation completeness

Manifest version 2 adds optional shared/per-platform `detect_required` paths.
With nonempty mandatory evidence, `EngineStatus.Installed` requires every path
to resolve to a regular file and at least one regular-file executable alternative
from ordered `detect`. Only the executable alternative selects the launch binary.
Blank paths, missing environment values, unresolved placeholders and directories
fail closed. Nonempty requirements in version 1 are rejected; omitted or empty
requirements preserve existing detection and adoption semantics. Platform arrays
replace shared arrays, so `[]` explicitly clears inherited requirements.

Status may truthfully report a running service whose installation is incomplete.
Readiness and adoption cannot restore `Installed` or bypass start/install checks
in that case. Occupied listeners still prevent spawn and installation, and an
existing owned process remains stoppable. Uninstall decides absence and success
from executable presence independently of mandatory files, retaining ownership
and failure reporting while an executable survives. Descriptors continue to
report platform support through `available`; no status or descriptor fields are
added. [MANIFEST.md](MANIFEST.md#platform-block) defines the authoring contract.

## Model identifiers

Models accepted for operations, serving declarations and selectable metadata
must be nonempty valid UTF-8, at most 512 UTF-8 bytes, with no C0
(U+0000–U+001F), DEL (U+007F), or C1 (U+0080–U+009F) characters. Validation
precedes trimming, normalization, fallback naming and effects; malformed raw
UTF-8 must be checked before JSON string decoding can replace it. Catalog and
installed/loaded inventory filtering removes individual invalid entries while
keeping valid siblings and successful empty inventories. Identifiers are never
truncated; ordinary Unicode, spaces, tags and slashes keep their existing rules.

New policy writes remain strict. Only the broker's disk-policy loader removes
old invalid model entries and dependent references in memory, without rewriting
the file, advancing its revision or discarding valid policy siblings. Explicit
empty served permission remains deny-all. This load-only sanitation does not
relax operation, serving, peer-update or activation validation, and adds no
method or payload field.

## Unexpected owned-process exit

An owned process exiting without an intentional stop reports the existing
unexpected-exit error, then retries saved-On intent through `restoreEnabled`
after 5 seconds, 30 seconds and 2 minutes (delays between failed attempts).
Successful readiness resets the consecutive-failure episode. After three failed
attempts no further automatic start is scheduled and the unexpected-exit error
remains. Unknown/Off intent never launches. Explicit stop, pause/sleep and
shutdown cancel both pending backoff and its registered start. A pending switch
defers automatic restore to switch reconciliation; the failed-reclaim latch and
precise cold-reclaim predicate remain authoritative. There is no retry RPC or
configuration setting, and recovery does not resume interrupted inference.

Before spawning a process-mode engine configured with `runtime.port: 0`, the
desired-state owner durably saves its selected endpoint in the optional
`allocatedPorts` map in `engine-state.json`. Restore reads that endpoint before
allocating a new port or probing readiness, then applies the existing positive
port, managed-install-path and current executable-image predicate. A matching
orphan is stopped and verified before a cold launch; a foreign listener is
never signalled. The endpoint is execution evidence, not authority to terminate
a process or a change to saved On/Off intent. Clean stop, including shutdown,
clears it; failed termination retains it. If endpoint removal fails after a
verified stop, later Stop, Sleep or shutdown retries that write without changing
saved intent. Files without this map retain their previous semantics, and
fixed-port engines continue using their configured port.

Before listener inspection, reclaim checks any positive `processStops` PID
whose saved port and executable match the current endpoint and managed binary.
The current OS executable image must independently match that binary inside
the managed install. This allows reclaim before bind, with the same PID-exit
and listener-refusal confirmation and failed-reclaim fence as listener reclaim.
A retained PID alone never authorizes signalling. Missing or zero PID records
retain listener-based reclaim, including the unresolved spawn-to-PID-save
window. Linux resolves executable symlinks when possible and compares paths
case-sensitively. macOS still requires canonical resolution, accepting case
aliases only when the filesystem confirms the same file. Windows image
comparison is unchanged.

## Custom llama preset launch

When custom `runtime.launch_args` uses native `--models-preset` (or the preset
is in `editable_launch.fixed_args`), omit the fixed model defaults for
`ctx-size`, `parallel`, `sleep-idle-seconds` and `cache-ram`. These settings
belong to each model preset; router CLI defaults would override every model.
Preserve other fixed options, except where the user explicitly supplies the
same option. Native aliases, long-option underscore normalization and equals
forms identify the same setting. Explicit user model options pass through and
override preset values without a duplicate fixed default. Preview, editable
argument readback and pre-spawn validation use the same effective fixed
prefix. Launches without a preset preserve the existing fixed arguments.

## Finite llama serving launch

An optional versioned `runtime.serving` declaration activates the finite llama
adapter for the llama.cpp family. It derives per-model total/per-slot context,
slots, opaque placement, idle retention and finite router child capacity from
the supported local preset subset. The shared declaration supplies the output
and memory bounds and artifact digest provenance. It does not enable automatic
residency. Owned validated runs publish declaration-derived content identities.

NE builds a canonical router launch with no PAIR-implicit envelope flags. The
legacy context 8192/parallel 1 and retention/cache defaults become finite preset
defaults. Explicit CLI options override model presets, which override global
`[*]` presets; native child environment supplies otherwise absent source
settings. Preview reports the resulting effective envelopes. Supported native
CLI and `LLAMA_ARG_` environment options outside the envelope controls pass
through unchanged, including inherited logging preferences. Router-only options
are not materialized into model presets. [LAUNCH_TEXT.md](LAUNCH_TEXT.md#finite-llama-router-launch)
lists the envelope controls and unsupported selections. For those controls,
duplicate aliases, conflicting repeated values, CLI plus environment for one
setting and unsupported forms fail closed. Preset include/inheritance/substitution,
startup loads and automatic fitting are unsupported. Finite numeric fields
reject auto/zero/negative/overflow, except `sleep-idle-seconds = -1`, which
disables idle retention in the pinned build. Positive retention remains valid;
zero and values below -1 are refused. Retention stays string-valued (`"-1"`).
The single `llamaServingOptions` table derives native spellings/environment names
from pinned llama.cpp `7fe450e1` `common/arg.cpp`. Logical/physical batch sizes,
K/V cache types, Flash Attention and KV/operation offload are supported and
retained in placement entries. Unsupported SWA, context-shift and scaling
controls, tensor/MoE/loading placement, draft/projector allocation, router
autoload and built-in model preset controls are rejected on CLI, in environment
and in source presets. A reserved `version` preset key is skipped; `threads` and
`threads-batch` pass through presets unchanged. Main-model sampling offload,
continuous batching, control-vector artifacts/layer placement and the default
speculative preset are unsupported too. Long CLI options normalize underscores
to hyphens before envelope lookup, exactly as the pinned native parser does;
the original spelling is retained for duplicate-alias rejection. Paired switches include native
`LLAMA_ARG_NO_*` compatibility variables, whose presence (even empty) matters;
configuring both environment aliases is ambiguous. Logging controls still pass
through. See the launch-text document for the supported/rejected boundary.
Non-unified context divides exactly across slots; unified KV requires an
explicit cap backed by total capacity. A cap without explicit source context
derives total as slots times cap with checked arithmetic. Output must remain
strictly below per-slot context. Total context, non-unified sequence context and
any per-slot cap must preserve the pinned runtime's 256-token padding, including
the int32 limit after rounding. Model IDs and preset sections must already be
native-canonical, with no normalized collisions. Generated values must survive
the native INI grammar exactly; paths containing comment markers or line breaks
are rejected after absolute resolution, with a model field error and no path.
Actual loaded training-context caps and props
are separate runtime evidence, not inferred from cold configuration.

Preview performs no probe, spawn, load, persistence or snapshot creation. It
reads bounded source/configuration data and model samples only, returning a
native-source approval digest and candidate local bindings. Apply requires the
exact reviewed digest/bindings, prepares an immutable small snapshot before
stopping a process or saving settings, and rejects changed inputs. Snapshots
reference absolute existing source model paths, preserving HF symlinks and
requiring all selected shards; model files are neither copied nor fully hashed.
Start re-reads source bytes, resolved targets, stat and bounded samples directly
before spawn, without a manager/state/content lock over file I/O. Recovery
regenerates identical bytes from matching approved inputs. Each owned process
retains its generation until exit. Same-stat middle-only artifact changes
outside the samples remain outside this cheap local binding's assurance.
Superseded pending generations and unconsumed cold generations at orderly
executor shutdown are released independently of running-process leases.

Serving process output enters the bounded engine log ring and startup diagnostics
only after the configured models directory, quoted spans holding a path and
path-bearing tokens are replaced with `[redacted]`. Failures report the NE
validation/launch stage, native early-exit code or readiness-wait duration with
that redacted output. Unconfigured launches retain their existing output
behavior. After a scanner error, remaining process output is drained and
discarded so oversized lines cannot block process exit.

Readback distinguishes configured, owned-launch-validated and loaded-observed
values. The existing loaded-model sweep supplies llama observations as described
below; strict qualification remains a separate gate. All manual load entry points
converge at the
action bound to `operations.load`: an undeclared exact model ID returns a typed
field error before mutation admission and dispatch, with zero engine calls.
An unlocked pre-check precedes mutation admission; restart-after actions acquire
admission before the operation lock. A second declaration check and load dispatch
share the per-engine operation lock for every manifest-bound load, including
when the current record is absent, so
settings/lifecycle replacement cannot send a load authorized under an older
configuration to a newer process. If replacement wins before that lock, rejection
releases admission without dispatch or saved-settings changes. Operation-lock
cleanup runs before mutation cleanup reacquires admission and before the restart
owner reacquires the operation lock. This preserves `switchAdmission` before
`opMu` ordering. Declaration rejection is active for every engine with an
applied serving record; engines without one retain request bytes, responses and
errors while serializing loads with their lifecycle. Once the operation lock is
acquired, dispatch owns validation and error precedence, including for an
already-cancelled caller. Queued loads honor caller cancellation without waiting
for the operation holder; held dispatches honor caller cancellation. Stop and
StopAll cancel queued/held loads on the same engine before waiting for the
operation lock, independently of startup cancellation. This applies with or
without NE, including a command load while its server is Off: Stop wins over
the racing load. A stop fence also cancels loads arriving during that wait.
Command actions use exec's default cancellation to kill the command and bound
output-pipe draining with the default five-second stop grace as WaitDelay.
Pipes are released after that bound; descendants are not guaranteed to be
terminated.
The dispatch path does not acquire the operation or content lock; any outer
content mutation is entered before the
load operation lock and the load releases it before restart/admission cleanup.
Other legacy launch behavior is unchanged. See
[LAUNCH_TEXT.md](LAUNCH_TEXT.md#finite-llama-router-launch) for supported options
and precedence, and [MANIFEST.md](MANIFEST.md#serving-declaration-override) for
the declaration and artifact ordering.

## Owned serving identity

`engine:model-identity` carries whole engine snapshots. Each owned current run
has `engineRunId`, `launchDigest`, `modelDigests`, and optional `serving`, a map
from exact model ID to the shared path-free `ServingEnvelope`. Each envelope's
`contentDigest` equals that model's `modelDigests` entry. Adopted and external
runs publish no identity. Unconfigured engines retain inventory-based digests.

NE content is SHA-256 of canonical JSON with version
`pair-serving-content-v1` and ordered `artifacts` entries containing `role`,
`sizeBytes`, and lowercase `sha256`. Order is the validated declaration/native
shard order. Paths, stat timestamps, samples, model IDs and provenance labels
are excluded. Explicit relocation/re-attestation preserves this aggregate
content digest while changing the local launch/config identity. NE launch is
SHA-256 of canonical JSON with version `pair-serving-launch-v1`, `launchDigest`
(the existing executable-stat/argv/explicit-environment digest captured before
snapshot rewriting), and `preparedDigest` (the approved compiler digest).
Random snapshot generation directories never enter this hash.

Only the owned launch-validated declaration produces these identities; native
inventory cannot supply or restore NE digests. Full inventory watcher sweeps
(60-second reconciliation and existing action pokes) validate the compiler and
bounded file bindings outside state/content locks. On drift they latch a
run-bound reason, publish an explicit empty `modelDigests` and omit `serving`;
settings readback reports `mismatch`. Restored bytes or later inventory do not
clear withdrawal. Successful explicit apply/start can clear it. A run/config
replacement fences validation results from the preceding snapshot.

The cold internal seam is
`currentServingEnvelope(ctx context.Context, engine, model string) (engines.ServingEnvelope, engineModelIdentity, error)`.
It requires the current owned run, exact declared model, matching approved
compiler result and no latched drift. It performs bounded file validation
outside locks and rechecks run/config before returning detached values. A
`*ServingIdentityRefusal` reports refusal; it never probes, loads, or edits
configuration. SR owns requested context/output/slot comparisons. The helper
`servingRetentionDisabled` is true only for exact retention `"-1"`; SR 5b owns
retention enforcement and capacity comparison against the engine-specific
`declaredResidentCount` bound to `policyRevision`. Declaration count is not the
resident set.

This identity alone does not establish loaded observations, qualification, strict
readiness envelope enforcement, or plain direct-model output enforcement.
Those remain separate gates. See [Local qualification](../../docs/local-qualification.mdx)
for the operator digest and binding trust contract.

### Loaded llama allocation observation

For a current owned llama launch, each existing model sweep captures the run and
configuration before reading `loaded_models`. Only freshly reported loaded exact
IDs are queried with `GET /props?model=<URL-encoded-ID>&autoload=false`. There is
no generic manifest action, router-placeholder observation, cold-model probe,
load or wake. Requests use the existing HTTP client, the five-second sweep
deadline, an 8 MiB response bound and no redirects. Network and bounded local
input validation run outside state locks.

Model-specific `default_generation_settings.n_ctx` is already per-slot context;
it is never divided by `total_slots`. Both positive integer values must equal
the current launch envelope. Semantically unknown residency, undeclared loaded
IDs, invalid props in a fully read 200 response and allocation mismatch clear
proof and latch the owned-run drift reason, withdrawing strict model identity
until successful explicit apply/start. An authoritative cold/unloaded set clears
observations without withdrawing launch validation. Strata has no observation
producer here.

Connection errors, sweep/action deadlines, non-200 responses and response body
read failures make loaded observation unknown for that sweep. They clear all
previous observed proof without writing serving drift. Identity, readiness and
resident eligibility that require observed proof are withheld for that sweep;
the next successful sweep restores observation without apply or restart.
Only exactly HTTP 200 succeeds for HTTP `list_models` and `loaded_models`
inventory actions (including a shared response) and `/props`; other 2xx statuses
are transport failures too. These observation reads never follow redirects:
success must come from the requested URL, and every 3xx is transport unknown.
Other actions retain their existing redirect policy and general 2xx success rule.
Semantic disagreement from any model in a sweep latches even when another
model's props read fails transport, independent of inventory ordering. A sweep
with any transport failure publishes no observed proof.

Settings readback uses the existing `observed` array and status only for matching
fresh loaded values; configured and launch-validated cold bounds remain distinct.
Observations are fenced against process/run/configuration replacement, later
sweeps and residency-changing action pokes. They expire after one configured
poll interval plus the sweep budget (at least five seconds). A failed sweep
cannot retain last-good observed proof. Source withdrawal hides proof immediately;
every loaded observation also revalidates the bounded approved inputs. Cold sets
retain the full-inventory cadence for local input drift checks.

SR's internal non-loading lookup contract is
`currentServingObservation(ctx context.Context, engine, model string) (engines.ServingObservation, error)`.
After requesting a fresh model sweep following load, SR can use this helper to
revalidate current local inputs and retrieve the exact model's fresh observation.
It returns `*ServingIdentityRefusal` when proof is absent, expired, withdrawn or
stale. It never performs HTTP I/O, loads, edits settings or establishes residency
grants. These observations establish context and slots only, not measured memory,
content qualification or live-host acceptance.

## Finite prepared Strata launch

For family `strata`, `runtime.serving` activates a narrow adapter for the native
config and server argument semantics inspected at `corbinjurgens/Strata`
`f6726098`, extended to the profile its current `main` (`0e70753c`) generates
(see the extension below). It requires one exact model, an owned Python process, absolute Python,
server and config paths, `--engine strata`, loopback host, explicit port and
`--lazy`. Python switches, abbreviated/unknown/duplicate server options and
inherited or explicit `STRATA_*`/`PYTHON*` overrides are rejected.

The flat JSON config accepts `exe`, `args`, `cwd`, `tokenizer`, `model_name`,
`parallel`, `lib_dirs` and `idle_unload_s`, and the fields the fork generates:
`log`, `env`, `gpu` and `fit_max_tokens` are validated and forwarded in the
snapshot; `build_metadata`, `gpus_asked`, `port`, `host` and `open_browser` are
validated and not forwarded, because the server reads none of them or takes the
command line, which PAIR owns. `host` must be `127.0.0.1`, `port` in 1..65535,
`open_browser` false, `gpu` a small non-negative index. `log` is an absolute
clean path in an existing directory, a new or regular existing file (not a link
or device), and never the same file as a declared input, judged by identity. `env` is a closed
allow-list of `STRATA_RESIDENT_HEADROOM_GIB`, `STRATA_FILE_CACHE_MIB`,
`STRATA_FILE_CACHE_COMPACT` and `STRATA_OWNED_PRICE`, each with its value syntax,
listed in the approved placement as `env:NAME=value`. Duplicate keys (compared
ignoring case, also inside `env`), nulls and unknown fields fail closed;
`memory_policy` is an unknown field, so the automatic memory policy stays
rejected, and so is an inline `api_key`: config bytes are hashed into advertised
fields. `api_key_file` is the absolute path of a regular, undeclared file read
with a bounded read, holding one key of 16 to 512 printable ASCII characters
without spaces or commas (one trailing line end allowed). The file is bound into
the approval digest by identity only (path, size, modification time, file ID),
never by content; it is never in the snapshot, the server receives the key only as
`STRATA_API_KEY` in its child environment, engine-manager's own requests to the
engine carry it (without following redirects, and redacted from error bodies),
and `internal:local-engine-credential` gives it, with the launch's port, only
to the broker for its health probe and its own proxy. Context is an explicit positive
`--max-context` per slot; `parallel` must be in 1..8. Explicit `--batch` or
`--slots` must agree with parallel and be at least two; otherwise the server
supplies parallelism. Total context is context times slots, checked against int32
overflow. Output must fit strictly below per-slot context. `--native`, `--pack`,
fixed positive `--prefill`/`--expert-cache` and `--spec` in 2..8 are required.
`--serve` may appear once or be omitted, because the server prepends it. Optional
placement is `--vram-reserve-mib` and `--kv-resident` (positive), `--kv` (one of
`fp16`, `int8`, `q4_0`, `q4`, `k8v4`), `--resident-budget-gib` (a positive plain
decimal up to 4096), the value-less `--no-prefill-borrow`, and `--suffix-draft`
or `--lookup-chain` only as an explicit `0`. Automatic fitting (`auto` cache or
prefill), clamping, other native options, hooks and alternate Python/native modes
are unsupported.

Every selected GGUF shard, pack/draft file, profile, tokenizer file, Python
executable, server script, config and native executable must have a matching
declared artifact with an absolute unambiguous existing file binding. Runtime
roles are `runtime:python`, `runtime:server`, `runtime:config`,
`runtime:native-exe`, `runtime:profile`, `runtime:tokenizer:<filename>` and
operator-selected `runtime:lib:<label>`. These files use cached full SHA256
verification plus binding checks. Model/shard/pack and `draft` artifacts retain
operator-supplied full SHA256 and bounded first/last samples, size, mtime and
resolved-target checks. Runtime declarations cannot resolve to any declared
model target or use GGUF paths (including split GGUF names). Full hashing of
other declared runtime files occurs before the lifecycle lock with cancellation;
locked compilation only rechecks bounded bindings and cached size/mtime/file
identity, rejecting files changed during preparation without rehashing.

Pack enumeration is fixed: `index.txt`, `dense.bin`, `experts.bin`; draft uses
`dense.txt`, `dense.bin`, `experts.bin`. Existing `embd.bin`, `extra.bin`,
`manifest.json` and `draft_vocab.bin` are also bound. A main pack that has
`native_experts.txt` and no `experts.bin` (the fork packer's usual output) binds
the table as a pack file instead: a `v3` or `v4` header, then layer lines from
zero with eight unsigned numbers and an optional shard column (one name, or for
`v4` a gate,up,down list in which an empty component means the first `--native`
shard). Every version comment in the table must be `v3` or `v4` and agree with the
header. Each named shard must be a plain `.gguf` name resolved
beside the first `--native` shard and a declared model artifact. Both the table
and `experts.bin` together are rejected as an ambiguous expert source, and a
table in the draft pack is unsupported. Tokenizer enumeration requires
`vocab.json`, `merges.txt`, `token_type.json` and `chat_template.jinja`.
`cwd`, tokenizer and each `lib_dirs` entry must be absolute existing directories.
The hashed config binds the library directory list; directories are never
crawled or hashed. Undeclared files reached through those directories or
inherited PATH/LD_LIBRARY_PATH, and Python imports beyond declared files, remain
unattested and are identified in readback. This is not a closed runtime closure.

Preview, apply and start reuse the small immutable snapshot and input recheck
lifecycle above. Only the config argument is replaced with a snapshot path;
original model targets remain in the snapshot. Cold readback never calls load.
Configured and owned-launch-validated bounds do not establish observed loaded
limits or strict content qualification. Public source inspection does not
establish the selected MSI native build; Windows/GPU/runtime acceptance remains
an owner gate.

## Automatic residency (internal parent IPC)

`engine:reconcile-resident` is an internal parent-codec request, with shared
`nodepolicy.ReconcileResidentParams` / `ReconcileResidentResult`. There is no
ec/HTTP route. The broker must reject client-originated requests before its
generic engine relay. Shared decoding rejects duplicate, case-variant, unknown
and null fields; invalid params return `-32602`. The reservation's incarnation
must exactly equal this manager's incarnation, including when omitted on wire.
`declaredResidentCount` is required: an integer in 1..256 describing this
engine's desired resident set at the request's existing `policyRevision`.
The bound reuses nodepolicy's resident-model limit. Missing, zero, negative,
non-integer and over-bound counts are invalid params before effects. Broker and
manager ship together; there is no compatibility default. Until SR 5c's broker
producer supplies the field, automatic resident loads are refused. The field
does not alter Prepare/Dispatch DTOs or keep-alive semantics.

The first request establishes an absolute deadline of `timeoutMs` (1..60000),
including lifecycle-lock waiting, dispatch relay, load and readback. Identical
operation IDs share the result and original deadline, including completion
until that deadline. Changed payloads conflict with `-32010`. Completed entries
expire lazily; the bounded operation table refuses new work when full. No
persistent residency queue or restart replay is introduced.

Checks before effects, after E3 admission and the lifecycle lock, and after
the dispatch relay refuse these conditions with `outcome:"not-dispatched"`:

- `projection-unknown`: no broker projection since startup.
- `paused`, `engine-locked`: either-origin pause or durable engine lock.
  `beginRemote` registers accepted work for cancellation by a later veto.
- `switch-busy`: an unreleased E3 checkpoint reserves this engine.
- `engine-off`: no saved On intent or running engine, regardless of Off origin.
- `not-installed`: normalized model absent from authoritative installed inventory.
  Older sweeps cannot replace newer installed inventory.
- `unqualified`: incarnation/run/requested-model digest mismatch, absent owned
  launch identity, withdrawn identity, ambiguous normalized identities, stale
  serving envelope or mismatched finite context.
- `unsupported-load`: implicit/restart-after/content-mutating load binding,
  unsupported action shape, fixed parameters, enabled native idle retention,
  or router capacity smaller than the revision-bound declared resident count.
  Readback must be a
  side-effect-free `GET` action with a result extraction spec and loaded-status
  filter.

These are internal refusal conditions: `ReconcileResidentResult` has no reason
field. No raw engine bodies, prompts or digest dumps are returned as explanations.

### Final-grant hop for the broker's SR 5 relay

Under E3 mutation admission and the engine lifecycle lock, immediately before
the load, the manager synchronously sends an **internal notification**:

```json
{"jsonrpc":"2.0","method":"engine:dispatch-resident-request","params":{"id":"correlation-id","params":{"reservationId":"reservation-id","proxyRunId":"proxy-run","managerIncarnation":"manager-incarnation","operationId":"operation-id","policyRevision":"1","engine":"llamacpp","model":"m","identity":{"engineRunId":"engine-run","settingsFingerprint":"sha256-hex","modelDigest":"sha256-hex"}}}}
```

`params.params` is exactly `nodepolicy.DispatchResidentParams`, built from the
reconcile request. The broker routes `node/dispatch-resident` to the bound proxy
run, preserving every field. The proxy atomically validates the grant and marks
it dispatched under its admission mutex. The broker must return its result as an
**internal notification**, with the same correlation ID:

```json
{"jsonrpc":"2.0","method":"engine:dispatch-resident-reply","params":{"id":"correlation-id","result":{"dispatched":true}}}
```

A refusal is `result:{"dispatched":false,"reason":"reservation-revoked"}`.
Reasons are `reservation-revoked`, `reservation-unknown`, or a shared resident
refusal reason: `engine-off`, `no-fit`, `paused`, `engine-locked`,
`paused-local`, `engine-off-local` (older inbound reasons), `not-served`, `not-installed`, `unqualified`,
`unsupported-load`, `switch-busy`, `operation-timeout`, `observation-unknown`,
`projection-unknown`. The prepare and dispatch validators reject `load-failed`:
it describes a completed load in model-configuration status, not refusal of
permission to load. A relay error instead has
`error:{"code":-32000,"message":"bounded explanation"}`, with no result.
Exactly one of result/error is required. Null, duplicate/case-variant/unknown
envelope keys and malformed results cannot grant permission. A request named
`engine:dispatch-resident-reply` cannot satisfy the pending notification hop.

Only a correlated, valid `dispatched:true` from a live context authorizes a load.
Refusal, timeout, cancellation, missing parent, send error and malformed or
mismatched reply all mean `not-dispatched` with zero engine calls. The pending
map is bounded and entries are removed on every terminal path. A queued grant
is checked against `ctx.Err()` at authorization and again before execution.
Sending is synchronous and bounded by the caller's remaining operation deadline
and the parent codec's 60-second ceiling, including write-gate waiting. A queued
cancellation emits nothing and leaves an active writer alone; an interrupted
active write closes the parent session and finishes before returning. Send
failure removes the pending entry and cannot authorize a load or accept a late
grant. Default publication failure also closes the session and reaches the
existing `Manager.Run`/`StopAll` engine cleanup path.
No inventory, publication or content lock is held across the relay or load.

Manager compares its own current run and normalized requested-model digest and
requires an owned nonempty launch digest. `StrictModelIdentity` has no separate
launch field; `settingsFingerprint` is broker-owned applied-settings identity.
The proxy owns qualification against the full engine identity, including launch
digest, and the final grant revalidates it. Manager does
not synthesize it or treat request fields alone as proof of a current grant.

### Load mapping and observation evidence

An explicit manifest `OperationLoad` binding must name a `POST` action with
`modelParam:"model"`, `BodySchema` exactly `{model:string}`, and no fixed
parameters. Its payload is exactly `{model}` with the normalized installed
model. The `loaded_models` action must be a safe `GET` with result extraction
and a loaded-status filter. Neither action may use `ParamsIn`, restart,
commands, remove paths, model resolution, progress protocols or content
mutation. Compatible manifests share this contract; no family-specific body
fields are invented. `load.keepAliveSeconds` remains validated in 1..86400 but
is not sent; native launch settings own retention. No prompt is sent.
There is no pull/install/copy/delete/start/restore, inference, arbitrary-action
fallback, eviction or unload protection implementation here. Loads never take
B2's content guard or withdraw identity.

The manager calls NE's current-envelope validator before final grant and again
immediately before the load. Requested context must equal the current approved
finite per-slot context, retention must be exactly `-1` (disabled), and
`RouterCapacity` must be at least `DeclaredResidentCount`. Settings/stop use the
same registered-load cancellation guard as ordinary loads; residency does not
introduce a separate cancellation registry.

Every dispatched call, success or failure, pokes the loaded watcher. A fresh
readback begins after the call, under the same deadline. `loaded` requires the
requested normalized model, still-current identity and fresh model-specific
context/slot readback matching the approved envelope. Successful settings
re-apply may clear withdrawal but cannot prove that an already loaded child's
allocation changed. The observation contains
`{managerIncarnation,generation,engine,model,identity,loaded}`; generation is a
monotonic counter scoped to this manager's resident readbacks. The settings
fingerprint is the one authorized by the proxy. No last-good watcher baseline
or desired resident list supplies observation evidence.

An explicit empty loaded array yields an observation with `loaded:false` and
outcome `failed`; missing/null/malformed inventory yields no observation and
`unknown`. Lost response followed by current authoritative loaded readback can
return `loaded`; a lost response alone never proves absence. An explicit HTTP
failure can return `failed` without an observation. Deadline/cancellation after
dispatch returns `timeout`/`cancelled`; identity withdrawal yields `unknown`.
Pending switch checkpoints/ports, Off provenance and intents stay untouched.
No resident fields are added to `engine-state.json`; broker owns unload protection.

The broker maps a definite terminal `failed` result to derived model status
`unknown` with reason `load-failed`, including finish/recovery and retained-hold
readbacks. It preserves the observation independently: failure without an
observation does not prove absence or release capacity. Missing/unknown results
and transport uncertainty retain `operation-timeout`; no raw engine error is
published as a model-configuration reason. This mapping does not change grants,
holds, recovery evidence, retry deadlines or terminal-hold expiry.

## 1. Purpose
A declarative, config-driven control plane for **local inference engines**
(currently Ollama, LM Studio, and llama.cpp). It owns an engine's entire
lifecycle *except serving inference* — locate, install, launch, stop, restart,
health, and config-declared actions — so one uniform API manages any engine
across OSes with no per-engine code. A third party drops in a JSON manifest and
their engine's install/launch/controls "just appear" over the same API: the core
extensibility story for an open-source product.

## 2. Scope
**In scope**
- Broker-coordinated argument/server/proxy settings, literal argument
  persistence, ownership validation and pinned peer settings synchronization.
  The normative operation and failure contract is
  [ENGINE_SETTINGS.md](../nvpair-ui-broker/ENGINE_SETTINGS.md); launch grammar is
  [LAUNCH_TEXT.md](LAUNCH_TEXT.md).
- Detect installation; **user-mode** install (HTTPS download + checksum-verify-when-pinned + run) per OS/arch.
- Lifecycle: start (with readiness probe), stop, restart, status, periodic health.
- Persistent server-port changes preserve arguments, environment and unrelated
  manifest overrides. Resetting to the bundled port removes only shared and
  host-platform port overrides, deleting the file only when it has no other
  settings. Host-platform precedence must not override a successfully saved port
  on reload; malformed existing overrides fail the save and remain intact.
- Config-declared **actions** mapped to each engine's local control API:
  Ollama declares `list_models`, `loaded_models`, `pull_model`, `run_model`,
  `unload_model`, and `delete_model`; LM Studio declares
  `list_models`/`list_downloaded`, `loaded_models`, `pull_model`, `load_model`,
  `chat`, `unload_model`, and `delete_model` (`remove_path` with
  `lms-disk-path` resolution);
  llama.cpp declares list, loaded-list, pull, load, unload, and native cache
  delete with exact model ids. `loaded_models` reports
  models currently resident in memory, name-extracted via the same declarative
  `result` spec with optional nested-path and row filters: Ollama uses
  `GET /api/ps`, LM Studio filters nonempty `loaded_instances` from
  `GET /api/v1/models`, and llama.cpp matches nested `status.value == loaded`
  records from `GET /models`.
- Semantic model operations: each manifest binds `load`, `unload`, `pull` and
  `delete` to its actions, with the model parameter name, fixed scalar
  parameters and an optional bounded response budget
  ([MANIFEST.md](MANIFEST.md#model-operations)). Local typed RPCs and the
  remote `ec` model endpoints resolve through the same bindings, so clients
  name only an engine and a model. Ollama binds `load` to `run_model` with
  `stream:false` and a 600-second budget, `unload` with `keep_alive:0`, and
  `pull`/`delete` with `name`; LM Studio and llama.cpp use the conventional
  actions with `model`.
- Engine identity and family: a manifest's `engine` is the stable ID and its
  optional `family` selects the coded model-store and catalogue adapters, so a
  locally defined fork with its own ID, port and store gets its family's
  behavior with no code change. `engine:descriptors` publishes the sanitized
  projection.
- The bundled llama.cpp launch fixes `--sleep-idle-seconds 300`: after five
  minutes without inference work, a loaded model releases its model and
  KV-cache memory and wakes on the next request. The router child remains alive
  and can retain a residual backend GPU context. Its readiness probe requires
  `/props` to report `role:"router"`; a single-model listener is incompatible
  and must not be started or adopted because its `/models` contract does not
  expose router residency state.
- The same bundled llama.cpp launch also fixes
  `--ctx-size 8192 --parallel 1 --cache-ram 512`
  in both `runtime.args` and `runtime.editable_launch.fixed_args`, so fresh and
  previously saved editable launches start with one parallel slot, an
  8,192-token total context and a 512 MiB prompt-cache cap. This bounds default KV-cache memory (the router
  propagates parent options to model children and would otherwise derive a much
  larger context from the model), at the cost of one-slot concurrency and a
  shorter maximum prompt; a loaded child resolving `--parallel auto` to four
  slots multiplied that memory. The tradeoff is deliberate: low default memory
  over default concurrency and maximum prompt length. Opaque user arguments
  remain the explicit override lane: an advanced user whose hardware and
  workload need different values supplies later llama.cpp arguments in the
  launch text. The manifest carries no parsing, range validation, migration or
  per-model logic for these values. `--cache-ram 512` caps llama-server's
  host-memory prompt cache at 512 MiB; the upstream default of 8,192 MiB lets a
  loaded model's process grow by up to that much as distinct prompts accumulate,
  cleared only by restarting the engine. Measured with Qwen3-0.6B over 150
  distinct streamed prompts, the working set plateaued after about 10 requests
  at roughly 466 MiB above its start with the cap, while it grew about 136 MiB
  per request without it. The cap limits only the host-memory prompt cache; for
  that model one cached entry for a long prompt is about 140 MiB, so 512 MiB
  holds a few recent prompts, and a full 8,192-token state (about 896 MiB in
  f16) does not fit, so long contexts are not kept in the host cache.
- Per-engine stdout/stderr log capture and structured operational error records, surfaced via the errors pipeline.
- A normalized node-level model list (`engine:models`): union of every running engine's `list_models`, name-extracted via each action's declarative `result` spec, plus the per-engine set of models loaded in memory (`loadedByEngine`, from each engine's `loaded_models` action). A successful explicit empty inventory remains an engine key with `[]`; a missing/malformed/failed inventory omits that engine key instead of being mislabeled as authoritative empty. A watcher polls the loaded set and pushes `engine:models-changed` when it changes (explicit load/unload, JIT auto-load, TTL/idle eviction).
- Expose all of the above over the `engine:*` JSON-RPC surface to whatever orchestrates the service, plus an optional plain-HTTP LAN endpoint (`--http-port`, `GET /v1/models`) that serves the model list to a peer's discovery daemon (the list moved off the size-limited mDNS TXT onto HTTP), together with this node's policy advertisement (`policy`, §7.0) as the broker last pushed it.

**Out of scope**
- **Inference traffic** — stays with `nvpair-proxy`; this service never proxies `/api/chat` etc.
- **Multi-instance per engine and an MCP server** — future-additive, not v1.
- **The node's error list** — owned by `nvpair-errors`, which holds it as in-memory session state; this service only emits `errors:report` / `errors:clear`.
- Automatic deletion of persistent llama.cpp model-cache files.

## 3. Key Use Cases
- **Install an engine, user-mode**: `engine:install {engine:"ollama"}` downloads the per-OS user-scoped package (Windows/Linux standalone archive extracted into a user dir; macOS app bundle — never an elevated `Setup.exe` or `curl | sh`), checksum-verifies, extracts, re-detects.
- **Run lifecycle**: `engine:start` / `engine:stop` / `engine:restart` / `engine:status`, with readiness and health probes against the engine's loopback port.
- **Run a declared action**: `engine:action {engine, action, params}` → the manifest-declared HTTP call to the engine's loopback control API (e.g. `127.0.0.1:{port}/api/pull`). (Methods, notifications, and UI events all use the colon form `engine:*`, matching the POC UI and the `errors:*` notifications.)
- **Onboard a new engine (no code)**: a vendor adds `engines/<vendor>.json`; the generic runner exposes their lifecycle + actions immediately.
- **Edge case — already installed**: detect short-circuits install (idempotent).
- **Edge case — checksum mismatch**: install fails before `run`, is reported, and never executes an unverified payload.
- **Edge case — readiness timeout / crash**: failed readiness returns to Stopped; an unexpected owned-process exit is reported and enters bounded saved-On relaunch through `restoreEnabled` as described above.

## 4. Open Questions / Risks
- **`engine:*` method stability**: the Broker forwards the namespace verbatim; evolve it additively and preserve `engine:get-installed` as the installed-engine query.
- **`severity` / `action` enums**: current wire values are `info|warning|error` and `dismiss|retry|none`; coordinate any future shared-enum expansion without changing existing values.
- **Bounded crash relaunch**: the existing owner retries an unexpected owned-process exit after 5 seconds, 30 seconds and 2 minutes while current saved intent remains known On. It rechecks intent under the lifecycle lock; Off/unknown intent, pause, shutdown and pending-switch ownership prevent automatic activation. Successful readiness ends the episode. There is no opt-in manifest flag or unbounded retry policy.
- **Risk — dynamic port assignment**: manifest `port: 0` reserves the option; range/policy TBD (may move to a future supervisor).
- **Risk — admin-only installers**: `mode: "admin"` is a flagged, refused exception, not the default; product direction is strictly user-mode.
- **Risk — LAN-open inference bind (interim)**: inference engines default `runtime.bind` to `0.0.0.0` (ordinary engines stay loopback; a per-call `bind` re-pins). This is a deliberate, temporary exception to the loopback-only posture; narrow it once authenticated inference transport exists. The engine-manager `ec` control surface is already protected independently by pin-based mTLS.
- **Future — declared/tunable env layer**: `runtime.env` is static today. A "declared tunables" layer (manifest-declared knobs, UI/broker-overridable per start) is worth adding; by env-richness the priority is Ollama → llama.cpp/Jan → vLLM (LM Studio / GPT4All are flags/settings-driven, not env). Related: `runtime.env` is process-mode-only — extending it to command-mode start commands is a deliberate, still-open choice. See `MANIFEST.md` → "Engine config reference".
- **Model deletion where the vendor has no command (LM Studio)**: implemented via the generic **`remove_path`** action kind — a manifest-declared, param-templated path the runner removes with safety rails (must resolve under a declared allowed root, reject `..`/symlink escapes). LM Studio's `delete_model` uses `model_resolution: "lms-disk-path"` to map logical ids to on-disk files via `lms ls --json` before deleting under `{models_dir}`, then `restart_after` to bounce a running server: LM Studio answers `/v1/models` from an index built at startup and exposes no rescan, so clients keep being offered the deleted model until it restarts. Ollama deletes via `DELETE /api/delete` (no restart needed) and ejects via `unload_model` (`POST /api/generate` with `keep_alive: 0`). llama.cpp deletes native cache entries via `DELETE /models` with the exact model id in a URL-encoded query parameter; the router updates its inventory in-process, so it also needs no restart.

## 5. Requirements

**Functional**
- Load + validate per-engine JSON manifests (bundled + user dir); select the host `<goos>/<goarch>` block; resolve placeholders (`{bin}`, `{cli}`, `{port}`, `{download}`, `{install_dir}`). Install commands also receive resolved download and destination paths in child-scoped `NVPAIR_INSTALL_*` environment variables so shell reparsing cannot corrupt them.
- Support both `process` (owned foreground) and `command` (daemon + control-CLI) runtimes; execute detect / install / uninstall / start / stop / restart / status / health and HTTP **or** CLI actions; emit `engine:*` results and notifications.
- For command-mode startup failures or cancellation after any start-command
  attempt, run the bounded official stop command and use the configured
  readiness endpoint to confirm listener closure before releasing the lifecycle
  lock. A control CLI may detach a daemon before it returns, so cleanup must
  also run when that CLI fails or is cancelled.
- Bound process-mode stops: on Unix send SIGTERM to the owned process group, then SIGKILL after `runtime.stop.grace_s` (five seconds by default); `signal:"kill"` skips the grace. On Windows, windowless managed engines require immediate `taskkill /T /F`.
  Unix stop and orphan-reclaim signals target the group only when the PID is
  its group leader (`pgid == pid`); otherwise they target the PID alone.
- Reclaim an adopted managed orphan only after authoritative executable-image
  inspection: Linux `/proc/<pid>/exe`, macOS kernel `proc_pidpath` with a
  fixed-size buffer, or the Windows process-image API. Refuse unavailable,
  malformed or ambiguous ownership inspection. An unavailable or empty port
  lookup may fall back to another provider; known malformed, incomplete or
  conflicting output must not. Resolve macOS filesystem aliases for image
  comparison, while canonical install containment rejects symlinks
  to external executables. Refusal must preserve external listeners and files.
  Configured paths remain literal; only Linux process-image metadata may remove
  one trailing ` (deleted)`, and an existing literal suffixed file takes
  precedence over that interpretation.
  On Linux, an existing suffixed spelling must identify the actual `/proc`
  executable file; a metadata collision with another file fails closed.
- Emit `errors:report` / `errors:clear` on its stdio for the Broker to forward to `nvpair-errors`.

**Non-functional**
- Compiles and runs on Windows / macOS / Linux × amd64 / arm64.
- **User mode, no runtime admin/sudo**; stdio / `--ipc` is the primary control channel, with an optional plain-HTTP model-list listener and a clustered pin-gated mTLS control listener; engines bind loopback by default, but a manifest's `runtime.bind` can open an inference engine to the LAN (Ollama ships `0.0.0.0` to serve the cluster), overridable per-call via `engine:start {bind}`; HTTPS-only downloads, checksum-verified when the manifest pins a `sha256`.
- Fully testable end-to-end with no real engines, network, or installers in the test path.
- Small, readable, minimal-dependency.

## 6. Inputs and Outputs

**Inputs** — from the orchestrator (`nvpair-ui-broker`) over stdio / `--ipc`, plus engine manifests from disk. Newline-delimited JSON-RPC 2.0 `engine:*` requests; manifest JSON (`engine`, `display_name`, `manifest_version`, `platforms{<os/arch>{detect, install, runtime}}`, `actions`). Every manifest is validated on load and rejected with a specific, human-readable error if malformed; manifests are read-only to this service.

`EngineStatus` object (returned by `engine:status` and `engine:get-installed`):
```json
{
  engine: string
  display_name: string
  installed: boolean
  running: boolean
  healthy: boolean
  port: number
  locked?: boolean
}
```

`locked` is the engine's durable lock on its own node. Present `true` or `false`
is authoritative; absent means lock state is unavailable, as from an older
target, and is never evidence of unlocked. Every registered engine's status,
get-installed shell and state push reads the desired-state store before taking
the runtime state mutex. A store read error leaves `locked` unavailable and
fails activation closed.

`LogLine` object (entry in `engine:logs`):
```json
{ time: string, stream: "stdout" | "stderr", text: string }
```

Errors are emitted with the `nvpair-shared/errors.ServiceError` wire shape (`id`, `message`, `severity`, `action`, …), mirrored locally in `reporter.go` with identical JSON tags. `id` follows `engine-manager:<class>:<engine>`; the Broker stamps `nodeId` / `timestamp`.

Example request:
```json
{"jsonrpc":"2.0","id":1,"method":"engine:install","params":{"engine":"ollama"}}
```

**Outputs** — JSON-RPC responses/notifications to the orchestrator; spawned engine OS processes; loopback HTTP to the engine's control API; `errors:report` / `errors:clear` notifications the Broker forwards to `nvpair-errors`. Lifecycle is not a hot path; error emits are best-effort (no-op if `nvpair-errors` is down). stdout writes are serialized so frames never interleave.

Example error notification:
```json
{"jsonrpc":"2.0","method":"errors:report","params":{"id":"engine-manager:install-failed:ollama","message":"sha256 mismatch","severity":"error"}}
```

## 7. API / Interface Contract
The stdio interface toward the local orchestrator (`nvpair-ui-broker`) is the primary surface. In addition, a clustered node serves a cluster-scoped remote-control surface (the `ec` service) directly over pin-based mTLS — engine-manager terminates that mTLS itself (see §7.2), the one place inter-node crypto lives here rather than in the orchestrator.

### 7.0 Methods and notifications
All traffic is JSON-RPC 2.0, newline-delimited over stdio (default) or `--ipc` (Unix domain socket / Windows named pipe).

Requests (caller → service):

| Method | Params | Result |
|--------|--------|--------|
| `engine:get-installed` | — | `{ engines: [EngineStatus] }` |
| `engine:describe` | `{ engine }` | the engine's manifest; local only, since it contains commands and paths |
| `engine:descriptors` | — | `{ engines: [Descriptor] }` — the sanitized, versioned projection of every registered manifest (`nvpair-shared/engines.Descriptor`): `version`, `engine`, `family`, `label`, `platforms`, `available`, `protocols`, `modelNaming`, `modelStore?`, `catalog?`, `operations` (`load`/`unload`/`pull`/`delete`, each `{ responseTimeoutS? }`). It must never contain executables, commands, environment values, private paths, URLs or credentials; the descriptor tests check a manifest containing all of them. Consumers validate it with `Descriptor.Validate` and reject unknown versions, protocols and operations |
| `engine:status` | `{ engine }` | `EngineStatus` |
| `engine:install` | `{ engine }` | `EngineStatus` (after install) |
| `engine:uninstall` | `{ engine }` | `EngineStatus` (after removal) |
| `engine:uninstall-managed` | — | `{ engines: [{ engine, removed, error? }] }` — removes the engines this service installed, selected by their install marker, each through the `engine:uninstall` path. Backs both clients' "reset all data"; the platform uninstallers reach the same code through `--uninstall-managed` because they have no broker. Neither `removed` nor `error` means the engine was left alone as not ours. |
| `engine:start` | `{ engine }` | `EngineStatus` (after readiness) |
| `engine:stop` | `{ engine }` | `EngineStatus` |
| `engine:restart` | `{ engine }` | `EngineStatus` |
| `engine:set-lock` | `{ engine, locked }` | `null` acknowledgement after persistence and verified stop for Lock, or removal of only the lock for Unlock. Local only: decoded strictly, with exact, unique member names, so `engine` must be registered, `locked` must be a JSON boolean and any other member, `node`, `nodeId`, `caller` and `origin` included, is `-32602`. No `ec` route or peer relay reaches it. See Lock storage and execution below |
| `engine:action` | `{ engine, action, params }` | the engine's raw response |
| `engine:logs` | `{ engine }` | `{ lines: [LogLine] }` |
| `engine:errors` | — | `{ errors: [ServiceError] }` |
| `engine:catalog` | `{ engine, platform?, arch?, query? }` | `{ models: [CatalogModel], source, platform?, arch?, fetchedAt?, searchable?, query? }` — the models an engine can **download**. One curated source per engine family, selected by the manifest's `family`: Ollama's is compiled in (`catalog/ollama-models.json`, no network), LM Studio's is the `lmstudio-community` Hugging Face org fetched live and cached, and llama.cpp's is the GGUF repos of `ggml-org`, `bartowski`, and `unsloth`, fetched live and cached, offered as `repo:Q4_K_M`. Only llama.cpp's is `searchable`: a `query` searches all of Hugging Face for GGUF repos, each query cached on its own, and is echoed back; a query over 100 characters is an error rather than being shortened; other sources ignore it and return their whole list. `platform` and `arch` are the GOOS and GOARCH the models will be installed on; omitting both means this host, and naming a platform without an arch leaves the arch unknown rather than borrowing this host's. `appleOnly` rows (MLX) are kept only for a `darwin`/`arm64` target. An engine with no curated source is an error, not an empty list. The Ollama reply is a single multi-megabyte frame — every hop on its path must allow `jsonrpc.WorkerFrameBytes`. |
| `engine:remote-get-installed` | `{ node }` | `{ engines: [EngineStatus] }` from the remote node |
| `engine:remote-install` | `{ node, engine, start? }` | `{ opId, status }` after the remote install |
| `engine:remote-pull-model` | `{ node, engine, model?, params? }` | `{ opId, result }` after the remote pull |
| `engine:remote-load-model` | `{ node, engine, model }` | the remote action result |
| `engine:remote-unload-model` | `{ node, engine, model }` | the remote action result |
| `engine:remote-delete-model` | `{ node, engine, model }` | the remote action result |
| `engine:remote-copy-model` | `{ node, engine, model }` | `{ opId, result: { engine, model, files, filesSkipped, bytesTotal, bytesCopied } }` after copying the model from `node` into this node's engine store (§7.3) |
| `engine:copy-model` | `{ opId, sourceNodeId, destinationNodeId, engine, model, expectedExportId? }` | `{ opId, result: copyModelResult }` after verified destination publication (§7.3) |
| `engine:cancel-model-copy` | `{ opId }` | `{ cancelled: boolean }`; local channel only |
| `engine:active-model-copies` | `{}` | `{ operations: [{ opId, sourceNodeId, destinationNodeId, engine, model, expectedExportId?, progress? }] }` |
| `engine:model-copy-capabilities` | `{ nodeId, engine }` | `{ version:1, engine, available, installed, modelStore, exportLayouts, importLayouts, writable, reason, identityBoundRetry, receiptDurability }` |
| `engine:remote-start` | `{ node, engine, port? }` | `EngineStatus` from the remote node (manifest `runtime.bind`; no per-call bind on the remote path) |
| `engine:remote-stop` | `{ node, engine }` | `EngineStatus` from the remote node |
| `engine:wake` | `{ engine }` | `EngineStatus` — starts the engine only if its saved intent is On; intent-neutral; holds the engine's lifecycle lock like `engine:restart`; refused once shutdown has begun |
| `engine:sleep` | `{ engine }` | `EngineStatus` — the `doStop` that restart and shutdown use; intent-neutral |
| `engine:intent` | — | `{ enabledByEngine, offByEngine }` — one saved-intent snapshot per registered engine; no saved intent reads as `false` without an origin |
| `engine:load-model` | `{ engine, model }` | the engine's raw response, via the manifest's `load` operation binding |
| `engine:unload-model` | `{ engine, model }` | the engine's raw response, via the manifest's `unload` operation binding |
| `engine:delete-model` | `{ engine, model }` | the engine's raw response, via the manifest's `delete` operation binding |
| `engine:remote-policy-get` | `{ nodeId }` | `{ policy, availability, pausedBy? }` — the target broker's peer projection of its policy: no routing section, and tiers keep only their local entries |
| `engine:remote-policy-set` | `{ nodeId, policy, expectedPersistedRevision }` | `{ policy }` — the peer projection after the target broker merges the change into its stored policy; the revision is forwarded unchanged and compared by the broker |
| `engine:remote-availability-set` | `{ nodeId, state, pauseMode? }` | the target broker's `{ availability, pausedBy? }` result; configured waits for drain, immediate is refused |
| `shutdown` | — | `null` |
| `log/set-level` | `{ level }` | `{ level }` |

The availability relay carries request-scoped `pauseMode` unchanged, including
explicit null for target validation. Omission selects `configured`; explicit
values are `configured` or `immediate`, valid only with `state:"paused"`.
After fresh target permission (`-32012`) and pause-fence (`-32011`) checks,
an authorized peer's immediate request is refused with `-32602`,
`Immediate pause must be requested on this node.` The authenticated caller,
not payload `nodeId`, determines locality. Valid local immediate requests are
currently unavailable with no persistence, supersession or execution effects;
the existing configured transition is never used for them. Timing is never
persisted. The target's policy readback and availability result preserve actual
`pausedBy` (`local`/`peer`, empty/omitted when unknown, cleared while available).
Peer responses omit activation and origin-private routing fields.

The `engineintent.Snapshot` shared wire type defines both intent payloads.
`enabledByEngine` is a required object mapping every registered engine to a
boolean. `offByEngine` is a required object mapping only registered engines
explicitly saved Off to `"local"` or `"peer"`, relative to the engine-owning
device. Legacy saved Off without provenance projects `"local"`; On and unknown
intent have no origin entry. Both maps derive from one desired-store read under
its mutex, without lifecycle locks or file writes. Unregistered stale origins
are excluded. Invalid saved Off origins or unreadable state fail the read and
suppress intent publication. The private disk field remains `offBy`; origins
are display data and do not change proxy admission or lifecycle behavior.

Notifications (service → caller): `engine:intent-changed{enabledByEngine, offByEngine}` (after an explicit start/stop changes saved intent), `policy:request{id, method, caller, params}` / `policy:cancel{id}` (a paired node's node-policy call for the broker, which answers with the `policy:reply{id, result, error?, errorCode?}` notification), `ready{version}`, `engine:state-changed{EngineStatus}`, `engine:models-changed{engine, models}` (pushed when an engine's loaded-in-memory model set changes; `models` is the full `engine:models` shape incl. `loadedByEngine`), `engine:install-progress{engine, stage, percent?, error?}` (`error` on the terminal `failed` frame), `engine:pull-progress{engine, op, stage, percent?, message}` (live progress for a local model pull driven via `engine:action{action:"pull_model"}` — the local counterpart of `engine:remote-progress`), `engine:remote-progress{opId, node, engine, op, stage, percent?, message}` (live progress relayed from a remote install/pull; a model copy reports `op:"copy"` with `file?, bytesDone?, bytesTotal?`, §7.3), and `errors:report` / `errors:clear` (consumed by `nvpair-errors` via the Broker). `install` / `start` / `stop` / `restart` / `action` / `remote-*` each run in their own goroutine so the read loop never blocks; their responses arrive when the op completes.

Notifications (caller → service): `engine:node-projection{policy, pausedLocally, managementBlocked}` carries this node's policy advertisement (`nvpair-shared/nodepolicy.Advertisement`: `revision`, `roles` per tier with the bounds of the matching profile, `pausedBy?`), local-pause provenance, and `managementBlocked`: whether a pause of either origin, accepted but not yet saved, draining or saved, fences peer management writes. The common mutation gate requires a known projection and enforces this bit after the current grant check. Local activation also refuses known blocked state; trusted local startup may proceed before the first projection. An older broker omits the bit, which reads as false; it is never derived from `pausedBy` or `pausedLocally`. The broker sends it after every persisted policy or availability change and again on every `engine:ready`, reading its current policy each time under one lock, so a later projection never carries older state. engine-manager keeps the last one in memory, replaced whole, never persisted; an undecodable one leaves the previous in place. `engine:models` and `GET /v1/models` return it as `policy`, omitted until the first push. `engine:models-changed` does not carry it.

`engine:model-identity{engines}` is also a service → caller notification: a whole snapshot keyed by engine, each entry `{engineRunId, launchDigest, modelDigests}` with `modelDigests` keyed by model name. The loaded-model watcher emits it after its first sweep and whenever the snapshot changes; an absent engine has no current identity. Only running owned processes with a known launch digest are included, excluding adopted services and command-mode daemons. Model digests must come from an inventory of the same run and contain lowercase SHA-256 hex. Publication is ordered with `engine:state-changed`, so a running engine's identity cannot arrive after its stop. The broker adds its settings fingerprint before publishing the strict admission identity to the proxy.

Before changing installed model content, the executor sends the broker the internal notification `engine:model-mutation-request{requestId, engine}` and waits for `engine:model-mutation-reply{requestId, engine, ok, error, code}`. Only a matching accepted reply permits the change. The broker conservatively withdraws the whole engine, retaining its applied settings fingerprint, and confirms destination withdrawal through the same proxy barrier as settings changes. These messages are never client RPCs or client events.

The guard covers manifest-bound semantic `delete` and `pull` actions in `Action` (including custom action names), streaming `PullModelStream`, and destination `CopyModelFrom`, including `runCopyPlan` and finalization. Typed delete and direct `ec` delete/pull inherit the guard through these owners. Copy source reads, load/unload residency operations, and opaque actions without a model-operation binding do not acquire it. A per-engine content semaphore serializes the guard and mutation separately from lifecycle `opMu`, so restart-after-action does not nest that lock around confirmation. Waiting for the semaphore observes cancellation and deadlines without beginning suppression or advancing the inventory epoch. Joining earlier publications also observes cancellation and deadlines; any abandoned join releases the publication mutex when it eventually acquires it.

The relay has a 15-second confirmation budget, beginning before its synchronous parent send. Refusal, timeout, cancellation, no parent, and send errors fail closed before mutation effects. An accepted reply authorizes mutation only while the context remains live, including when cancellation and a queued reply are both ready. The deadline bounds the reply wait after the send returns; a blocked send cannot authorize a mutation but can outlast the budget. Pending entries are removed on every reached terminal path.

Before requesting withdrawal, the inventory owner suppresses the engine, advances its epoch, and joins earlier identity publications. Inventory captures that epoch before I/O; suppressed or stale-epoch observations cannot restore identity. Every guarded exit, including refusal, cancellation and partial failure, advances the epoch again, clears cached digests, releases suppression and requests a fresh sweep. Only a fresh post-completion observation of the owned run can restore identity. Local epoch metadata participates in publication deduplication, so unchanged final content is republished too; it is not sent on the wire. Withdrawal does not cancel inference already dispatched.

`percent` on all three progress notifications follows one rule: it is present only when the step is measurable (download bytes, byte-progress pulls) or terminal (`100` on install `done` / `already-installed`, `-1` on install `failed` and pull `error`). An indeterminate step omits it — the install `verified` and `installing` stages always do, because the install command is opaque — and a consumer renders the stage alone rather than a percentage.

Example `engine:install-progress` (stdout):
```json
{"jsonrpc":"2.0","method":"engine:install-progress","params":{"engine":"ollama","stage":"downloading","percent":42}}
{"jsonrpc":"2.0","method":"engine:install-progress","params":{"engine":"ollama","stage":"installing"}}
```

Example `engine:pull-progress` (stdout):
```json
{"jsonrpc":"2.0","method":"engine:pull-progress","params":{"engine":"ollama","op":"pull","stage":"pulling","percent":62,"message":"pulling"}}
```

### 7.1 Versioning
- `engine:*` method names are namespaced and additive — new methods, never breaking changes to existing ones.
- `manifest_version` gates manifest-schema evolution: unknown optional fields are ignored (backward-compatible growth); a version higher than supported is rejected.

### 7.2 Remote engine management (the `ec` surface)

**Management authorization.** `controlServer.routes` is the single route/method
classification registry. `controlServer.authorize` runs after pinned mTLS
identity verification and requires a fresh
`clustertrust.Mesh.RemoteManagement(caller)` allow for every mutation, including
stop, unload, delete, pull, policy/role/served/resident edits and pause/resume.
Read-only POST policy get and settings get/preview, GET inventories, model
exports, events and switch status remain reads. Unclassified routes fail closed;
denial precedes mutation handlers, streaming headers and side effects. The
broker rechecks the same predicate at its owning locked acceptance points for
policy, availability, settings and switch; relayed caller identity comes only
from the current authenticated worker notification.

HTTP 409 `{"error": "bounded explanation", "code":"remote_management_denied"}`
maps to RPC `-32012`, independently of `-32010/policy_conflict`,
`-32011/local_veto`, `-32013/engine_locked` and unpinned 403. Each typed code
keeps its number through the 409 body, stream error frames and broker relays,
with the target's message and no JSON-RPC error `data`. `-32013` is the refusal
of a start, load or switch into an engine locked on its own node; a switch
receipt carries it as kind `engine_locked`, which is valid only with `-32013`. Missing, invalid, unsupported, changing or
incomplete authority denies without a cached allow. A grant does not override
local veto, private/CORS policy, served/resident protection or target admission.
Permitted inference is unchanged. Direct operations accept at the fresh `ec`
gate; subsequent lifecycle/content lock waits belong to that accepted operation.
Revocation blocks new acceptance, including queued broker writes, but does not
cancel or roll back already accepted work; existing progress/receipt reads
remain available. R4 local veto cancellation remains independent.
An overlapping authorization may already hold a coherent allow snapshot before
the denial replacement; authorization begun after successful revoke must deny.

Pinned `GET /v1/remote-management` returns only the verified caller's
`{version:1, allowed, revision, state}`; it cannot select a peer or set a grant.
`engine:remote-management-get {nodeId}` uses the existing pinned direct/reverse
client. Effective states are `allowed|denied|unreviewed|unavailable`; a target
404/405 is reported as `unsupported`, never enforced permission. Only
cluster-manager writes local grants and migration state. New targets enforce
the gate against old clients; old pairing clients missing the required local
boolean receive `-32602`. A pre-PG target or downgrade cannot enforce PG.
Decision logs contain authenticated caller/target UUIDs, bounded operation
family/class, decision, permission revision and reason, without request content.

With `--control-port` and a clustered `--cluster-dir`, engine-manager serves a cluster-scoped remote-control surface over pin-based mTLS (`nvpair-shared/clustertrust`): it presents this node's cluster leaf, requires a client cert, and `403`s any caller that isn't a byte-for-byte pinned cluster peer. The listener is bound whenever `--control-port` is set and admits callers by live membership — its leaf is resolved per handshake, so an unclustered node presents none and every handshake is refused — so no restart is needed on `cluster:identity-changed`; the broker registers `ec` whenever a cluster dir is configured. Routes under `/v1`: `GET /engines`, streaming `POST /engines/install` and `/models/pull` (chunked NDJSON — zero+ `{"type":"progress"}` frames then one terminal `{"type":"result"}`/`{"type":"error"}` frame), non-streaming `POST /models/{load,unload,delete}`, non-streaming `POST /engines/{start,stop}` returning `EngineStatus`, and the model-copy source routes `GET /models/files?engine=&model=` (NDJSON: `hashing` progress frames, then a result frame carrying `{engine, model, revision?, files:[{path, size, sha256}]}`) and `GET /models/file?engine=&model=&path=` (one listed file, Range supported). The node-policy routes `POST /node-policy/get`, `POST /node-policy/set` and `POST /node-availability/set` (bodies `{}`, `{policy, expectedPersistedRevision}`, `{state}`; at most 256 KiB; unknown fields refused; the revision is the compare-and-set precondition of a policy set, forwarded unchanged for the broker to enforce) are pin-gated with the caller's identity, clear any `nodeId` so no further hop is possible, and relay to the local broker as `policy:request` carrying the authenticated `caller`; the broker owns and re-authorizes the call. `engine:remote-policy-get` / `-set` and `engine:remote-availability-set` are their client half, addressed by `nodeId`.

The broker's `policy:reply {id, result, error?, errorCode?}` carries a JSON-RPC
`errorCode` on every failure. Invalid parameters, including missing or malformed
`expectedPersistedRevision`, invalid policy and invalid availability, use
`-32602`; revision conflicts use `-32010`; a resume the local veto refuses uses
`-32011`; remote-management denial uses `-32012`; an engine lock uses `-32013`;
target-owned profile inference settings use `-32014`; save failures and other operational errors use `-32000`. Inbound
policy routes return HTTP 400 for `-32602`, HTTP 409 with the JSON body
`{"error": <diagnostic>, "code": "policy_conflict" | "local_veto" | "remote_management_denied" | "engine_locked" | "remote_setting_locked"}`
for `-32010`, `-32011`, `-32012`, `-32013` and `-32014`, and a plain-text HTTP 409 for other broker or relay errors. At the
origin, HTTP 400 becomes JSON-RPC `-32602`, a typed 409 becomes its code, and
other HTTP failures become `-32000`; diagnostics are preserved without inferring
a code from their text. A stale policy draft conflicts before
policy validation, even if it tries to enable remote or paid execution.
Successful replies have no error code. Body/shape validation still returns 400,
unpaired callers 403, non-POST calls 405 and a missing coordinator 503.

The `-32014` / `remote_setting_locked` case is a target broker refusal to
change a stored profile's protected inference settings or identity, or to
create a remote local-only lock. The shared merge supplies the safe diagnostic:
bounded stored engine/model only, or a fixed generic message when no stored
identity exists. Engine-manager carries the code/message without inspecting
policy options, adding error data or classifying by text. Permission and
expected revision are checked before the target lock under the existing
writer lock. Rejection leaves the entire policy and its publication unchanged.

**Off, lock and pause.** Saved Off is ordinary intent, never a prohibition on a
new granted peer start/load/switch. This includes local, peer and legacy Off and
missing intent. Explicit model load never starts a stopped engine, and its real
runtime failure must not become a lock/pause refusal. Saved-On restoration,
wake and residency retain their existing intent requirements. A local stop or
uninstall records `offBy:local`; an ec stop records `peer` unless already Off,
preserving its earlier provenance. On clears the origin. Legacy Off without
`offBy` projects local provenance. A local Stop admitted during existing peer
work still cancels that operation with `-32011/local_veto` and retains all
operation-specific durable switch cleanup; it does not veto a later operation.

After a successful fresh grant check, `controlServer.authorize` uses the common
route registry classification: every incoming mutation returns HTTP 503 until
projection is known, then HTTP 409 with `code:local_veto` while
`managementBlocked` is true. The exact message is `This node is paused. Resume it on this node before managing it remotely.` Grant denial retains `-32012`
precedence, and pause precedes lock for a granted peer. Install, start, stop,
load, unload, delete, pull, settings apply, switch apply, policy set and
availability set are fenced. This includes the pause initiator's Resume and
repeated Pause; only the owning node may Resume. Reads return before this check,
including GET/HEAD, read-only POST preview/status/get, event streams and model
files. No second production route registry is introduced.

Either-origin pause projection cancels accepted starts, loads, crash relaunch
and pending activation. Local start/load/switch also refuse known pause; local
reads, configuration, lock/unlock, stop and unload remain available. Settings
apply/recovery can change metadata but cannot restart while paused or locked.
Accepted content installation/download may finish, without starting an engine.
When available, a locked engine still admits granted stop/unload/delete/pull,
install-without-start and settings edits; restart-after cannot activate it.
Lock refusal is `-32013/engine_locked` with the exact message `This engine is locked on this node. Unlock it on this node before starting or loading a model.`
The HTTP body, stream terminal error, JSON-RPC relay and switch receipt retain
the numeric code and bounded target message. Switch kind `engine_locked` is
valid only with `-32013`; pause and operation-scoped Stop retain `-32011`.

**Lock storage and execution.** `engine-state.json` extends the existing document
with optional `locked: {engine:true}`; it has no schema version bump or separate
store. Missing maps mean unlocked. Reject null maps, null/non-boolean entries
and locked+On contradictions rather than repairing unreadable state. Preserve
unknown engine keys and checkpoints. An older binary rewriting this private
file may discard locks; downgrade to an old writer is unsupported.

Each complete command holds the engine's `lockMu` before mutation admission
through acknowledgement. Ordinary lifecycle commands do not take it. Unreleased
switch involvement refuses lock/unlock before any write with `-32000` and the
existing `engine "<engine>" is reserved by an engine switch (busy)` message.
Lock atomically replaces lock=true, saved Off and local Off provenance, emits
intent/state, then cancels starts, serving loads and crash relaunch before
waiting on `opMu`. Never hold `desired.mu` while waiting on `opMu`. The existing
process owner stops/unloads and verifies termination before success. Pre-replace
failure leaves disk/runtime unchanged. Post-replace stop failure retains durable
locked+Off, publishes actual runtime facts and returns operational `-32000`;
idempotent Lock retries unresolved stop. Use the existing atomic writer warning
semantics; a directory-sync warning is not proof of power-loss durability.
Unlock removes only the lock and emits state, leaving Off and provenance without
starting. Lock-only changes emit state too.

Common start/load guards read the durable store under `desired.mu`; loads check
before admission and after lifecycle waiting, and starts check before accepting
readiness. No late success, intent setter or switch commit may save On over a
lock. These guards cover explicit activation, restart, restore, wake, crash
relaunch, settings recovery, switch recovery and action restart-after. Neither
sleep nor idle unload/stop changes lock. Source ownership validation remains
unchanged when switching away from a locked engine; target lock and either-origin
pause are checked before source stop for local and remote switches.

The `engine:remote-*` methods are the client half: engine-manager resolves the target `node` in an `ec` peer directory (fed by its own `discovery:subscribe{services:[ec]}` to the broker relay), dials the peer's `ec` surface with the same pinned identity, relays each streamed progress frame up as `engine:remote-progress` (keyed by a minted `opId`), and settles the request on the terminal frame. Remote install/pull run for the operation's full duration (no broker-imposed timeout); remote stop and model unload/delete are fast request/response, remote Ollama model load uses the readiness-sized response budget, and remote start waits for the engine's bounded readiness result. They error if this node isn't clustered or the target isn't a pinned peer.

### 7.3 Model copy between nodes
`engine:copy-model {opId, sourceNodeId, destinationNodeId, engine, model, expectedExportId?}` dispatches to the destination's existing `CopyModelFrom`.
The controller owns no artifact I/O for a copy between other devices.
`engine:remote-copy-model {node, engine, model}` mints an opId and delegates to
the same coordinator with this node as destination.
The wrapper retains its original parsing and missing-field error precedence,
including shared raw model validation, and forwards only the canonical tuple.

Requirements:

- **Identity and admission.** Device selectors are host UUID keys, independently
  associated with certificate principals by the live directory. Resolve this
  node as the unique directory entry whose cluster UUID matches the live mesh
  principal; unknown or ambiguous associations fail closed. Never guess from a
  hostname or treat the certificate UUID as a host selector. New canonical RPC
  and inbound HTTP control objects are at most 16 KiB, with exact unique keys, valid Unicode/model
  identifiers and no trailing body. Op IDs are 32 lowercase hex characters;
  expected export IDs are lowercase SHA256 hex. Identical endpoints, malformed
  requests, duplicate active IDs and capacity overflow fail before lookup/dial
  or progress. Manager's copy-only state is bounded to eight active entries,
  including incoming destination requests, with tuple/context/latest progress.
  Every exit releases its reservation. Exact cancellation and shutdown cancel
  contexts, including lock waits; completed entries are removed, not journaled.
- **Destination authority.** Pinned `POST /v1/models/copy` is a registered
  mutation. Its only fields are `opId`, `sourceNodeId`, `engine`, `model` and
  optional `expectedExportId`; the recipient always writes locally. A narrow
  Manager callback rechecks the real caller's fresh grant and pause/recovery
  gates under its reservation before effects. Previously accepted work is not
  undone by later grant revocation. The source independently pin-gates the
  destination as the actual reader. Preserve `-32012/remote_management_denied`,
  `-32011/local_veto`, and `-32010/policy_conflict` end to end. There is no
  install/start, activation gate or controller staging fallback.
- **Capabilities and retry.** Pinned read `GET /v1/models/copy-capabilities`
  accepts only `?engine=...` and returns the local/remote RPC capability shape
  above. `modelStore` and native layout names are adapter IDs (`ollama`,
  `llamacpp`, `lmstudio`), plus `llama-artifacts-v1` for custom llama.cpp.
  Fields describe actual installation/support and caller write readiness,
  without paths, grants or other-peer permission state. Unsupported versions
  or missing routes are never allows. Require an installed destination and a
  shared layout, then verify the actual export against both endpoints' lists.
  The outbound capabilities GET has the ordinary 30-second remote-control budget
  as a context deadline covering headers and the whole bounded response body;
  an earlier caller deadline or cancellation still wins.
  After destination authorization/admission, an identity-bound retry first
  verifies a matching committed receipt and fully hashes every confined file
  before source lookup/dial/capability checks. Verified completion may therefore
  survive an offline source; reading an old receipt retains uncertain durability.
  Otherwise source authority must be live and its export ID must equal
  `expectedExportId` before writes; alias rebinding is a typed conflict.
  Alias-only retry is fresh resolution, not proof of identical content. Active
  snapshots carry no terminal history: disappearance while disconnected means
  unknown outcome and permits explicit idempotent retry, never inferred success.

- **Transport owner.** The shared pinned peer/service adapter resolves the
  admitted source and current advertised `ec` endpoint, prefers direct mTLS,
  and uses the existing authenticated reverse session after a bounded
  pre-dispatch direct failure. Inner mTLS retains the source handler's pin and
  membership checks. Listing frames and file bytes stream with backpressure
  and cancellation, without whole-file buffering or a new listener. The
  adapter does not replay dispatched requests or exposed responses. An
  interrupted file attempt ends; the copy owner issues a new Range request and
  verifies the final hash under the transfer rules below.

- **Source listing.** `GET /v1/models/files` resolves exactly the model's files from the engine's on-disk layout, as slash-separated paths relative to the engine's store, with size and sha256. Hashes are computed lazily by streaming the file and cached by absolute path, size and mtime; a file named by its sha256 (Ollama blobs, llama.cpp LFS blobs behind snapshot links) uses that name. Native-layout files must resolve, after symlinks, to regular files inside the store; custom artifacts use the exact local bindings below.
- **Source serving.** `GET /v1/models/file` serves a path only when it is exactly one of the model's listed files (`400` for a malformed path, `404` otherwise), through `http.ServeContent` for Range support.
- **Custom llama.cpp identity.** Before interpreting an ID as an HF locator,
  resolve exact configured preset/serving candidates and committed copy receipts.
  Applied declarations must compile against the snapshot of the effective launch
  configuration and approved artifact bindings; compilation and binding I/O run
  outside lifecycle/state locks, followed by a snapshot recheck. Undeclared
  presets use the same parser and effective launch-option precedence, resolving
  relative paths against the preset. Export all consumed model then draft GGUF
  shards, in native order. Unknown/ambiguous bindings, declaration/preset drift,
  missing shards, projectors, adapters, includes and remote-source options are
  refused without falling back to catalog interpretation. External files are
  authorized only by these exact local bindings, never peer-provided paths.
- **Custom export protocol.** A custom listing adds
  `layout:"llama-artifacts-v1"`, `exportId` and per-file `role:"model"|"draft"`.
  Synthetic paths are `model/<basename>` or `draft/<basename>`. The export ID is
  SHA256 of version-1 canonical JSON containing the exact requested ModelID and
  ordered role/path/size/full-SHA256 tuples; the serialization is normative in
  `artifactExportID`. Sampled serving-binding hashes never substitute for full
  file hashes, and asserted full serving hashes must match. GET adds
  `exportId` and checks the source's issued binding snapshot, current manifest,
  symlink-resolved identity and the opened regular file. A source restart or
  bounded issuance-cache eviction requires relisting; unchanged listings retain
  their binding. Neither host paths nor launch/serving configuration is exported.
- **Bounds and cancellation.** Listing manifests are at most 1 MiB; progress
  frames are at most 64 KiB each, with no cumulative hash-stream byte limit.
  Result frames exceeding 64 KiB use the canonical stream envelope with `type`
  first. Artifact arrays are decoded incrementally and limited to 1024 entries;
  source shard enumeration refuses excessive counts before allocation. Sizes
  and totals are nonnegative and checked for overflow. Duplicate, Windows-unsafe
  and case-fold-colliding paths are refused. A shared hash waiter observes its own
  cancellation immediately; a producer cancellation causes a live waiter to
  re-stat and retry in the same cache under its own context, while ordinary I/O
  failures are propagated. Successful full hashes remain reusable.
- **Refusals before downloading.** An engine without a known layout, an engine not installed on the destination, a listing whose paths, hashes or shape do not fit the layout, and a volume with less free space than the missing bytes (net of partials) plus 256 MiB.
- **Transfer.** Each missing file is downloaded to `<final>.part`, resumed with `Range` after an interruption or stall (60 s without data; up to five attempts per run), and partials survive across runs. The complete file's sha256 must match before an atomic rename; a mismatch deletes the partial and fails, after one fresh retry if the attempt resumed an older partial. Files already present with a matching hash are skipped.
- **Conflicts and containment.** A differing completed destination file is a
  conflict in every layout. Check all files before writes and recheck under the
  writer locks, including before rename. Destination lexical and resolved
  containment, pre-existing symlinks and case collisions are checked; escaping
  locations and unsafe partial paths are refused. The model content mutation
  guard and broker identity-withdrawal acknowledgement remain mandatory.
  Each partial open, including retries, is confined through `os.Root` and
  validated as the same regular file before truncation/writes. Check the verified
  partial identity before and after the rooted rename; path swaps cannot qualify
  a completed file or custom receipt.
- **Custom publication.** Import into the destination-owned
  `.pair-copies/<SHA256-of-exact-ModelID>/<exportId>/` tree. The raw alias is
  metadata, not a filesystem path. Publish versioned alias-level `copy.json`
  only after all artifact hashes verify, through the durable file replacer. The
  receipt records engine, exact alias, layout, export identity and relative
  artifact metadata; it contains no source paths or serving claims. A different
  committed alias receipt is a conflict. An identical receipt is a no-op after
  verification; reading it cannot establish earlier durability. Uncommitted
  partial bundles are resumable but unavailable as completed copies. No garbage
  collection, native preset mutation, rescan/start/load or fake native inventory
  entry is added.
- **Custom terminal result.** Add `layout`, `exportId`,
  `requiresConfiguration:true` and typed `durability` (`confirmed` or
  `uncertain`) to `copyModelResult`. Adopt a landed-but-uncertain receipt as
  qualified completion; pre-replacement failures remain failures. An identical
  verified receipt reports `uncertain` because its earlier durability is unknown.
  Legacy layout results retain their original meaning. This Go contract alone
  does not provide dedicated bridge/UI qualification display; custom artifact
  completion is separate from destination serving configuration and readiness.
- **Visibility.** The file that makes the model visible is written last: Ollama's manifest (after checking that every blob it names is present), llama.cpp's first GGUF split and then `refs/main` when the repo was not cached before, followed by `GET /models?reload=1` on a running router; LM Studio's model GGUF, followed by a restart of a running server (the `restart_after` path, since LM Studio indexes at startup).
- **Progress.** `engine:remote-progress` retains `node=source`, `opId`, `engine`,
  `op:"copy"`, `stage`, `percent?`, `message` and adds `sourceNodeId`,
  `destinationNodeId`, `model`, `exportId?`, `file?`, `bytesDone?`, `bytesTotal?`.
  Per-operation callbacks stream coalesced listing/hashing/checking/downloading/
  finalizing progress, followed by exactly one done/error/cancelled/unknown outcome.
  Done follows verified publication and carries total byte counts. The remote
  NDJSON endpoint emits one terminal result/error frame, checks actual writes
  and flushes, and cancels on failure or request loss. Progress frames are
  bounded individually; inactivity and write deadlines bound a stalled stream.
  Missing or invalid mutation acknowledgements carry a copy-specific typed
  uncertainty error and emit terminal `stage:"unknown"` with `percent:-1`,
  including when destination publication completed before acknowledgement loss.
  Definite refusal/error and explicit cancellation retain their distinct stages.
  The copy stream's terminal error frame preserves `stage:"unknown"` as
  structured state; diagnostic message text never selects the outcome. Controller
  transport loss never replays the POST or fabricates a definite failed copy.

Stores: Ollama uses `OLLAMA_MODELS` from the engine's launch environment or this process, else `{models_dir}/models`; llama.cpp uses `LLAMA_CACHE` (the manifest sets `{models_dir}`) in the Hugging Face hub cache layout, selecting `owner/repo:TAG` files as llama.cpp does; LM Studio uses `{models_dir}` and the paths `lms ls --json` reports. The layout is selected by the manifest's `family`, so a fork with its own id and `models_dir` uses its family's layout rooted at its own store; an engine whose family has no layout is refused. One copy of a given model runs at a time per node.

## 8. Dependencies
- **Upstream**: the orchestrator that spawns it and forwards `engine:*` (`nvpair-ui-broker`).
- **Downstream**: `nvpair-errors` (via the Broker's `errors:report` / `errors:clear` forwarding); the managed engine processes.
- **External**: engine vendors' download URLs (HTTPS; checksum-pinned when a `sha256` is set, else HTTPS-only with a warning); first-party `nvpair-shared/applog` (logging + `log/set-level`) and `nvpair-shared/clustertrust` (pin-based mTLS). The `nvpair-shared/errors` wire shape is mirrored locally with identical JSON tags. No third-party runtime services.

## 9. Data Ownership
- **Owned**: the in-memory engine registry (parsed manifests + per-engine runtime state) and per-engine log/error ring buffers — transient only.
- **Source of truth**: no — `nvpair-errors` owns the node's error list (in memory, for the session); model inventories belong to the engines; manifests on disk are authored elsewhere.
- **Storage**: in-memory; manifests read from the per-user data dir's
  `engines/*.json` (`%LocalAppData%\Nvidia Corporation\Personal AI Router` on
  Windows, `~/.config/Nvidia Corporation/Personal AI Router` on Linux, and
  `~/Library/Application Support/Nvidia Corporation/Personal AI Router` on
  macOS) plus bundled `manifests/*.json`. Each engine's model store is declared
  as `models_dir` outside that data dir (`~/.ollama`, `~/.llamacpp`,
  `~/.lmstudio/models`) so neither an engine uninstall nor the app-level data
  purge removes downloaded models; clearing one is the user's own choice, made
  per model through `delete_model`.

  Removing an engine deletes only the paths its manifest's `uninstall.remove`
  names, and never its `models_dir`. The store is recognised by file identity,
  not by path: a store reached through a symlink keeps both the link and the
  directory it leads to, and a spelling that differs only in case is the store
  wherever the filesystem says so. Removal unlinks a symlink or junction rather
  than following it, and refuses a target that is the store or inside it.

  One persisted file: `engine-bin/<engine>/installed-by-pair.json`, written after
  an install this service performed. It is the only record distinguishing an
  install PAIR made from one the user made themselves, which an engine whose
  vendor script chooses its own destination cannot be told apart any other way,
  and it gates both `engine:uninstall` for such an engine and
  `engine:uninstall-managed`. No database.

## 10. Design Constraints
- **Performance**: control plane, not inference; sub-second RPCs except install (network-bound) and start (bounded by the readiness timeout).
- **Scalability**: a handful of engines per node; one managed instance per engine in v1.
- **Reliability**: best-effort readiness and health probes, with bounded saved-On crash relaunch through the existing `restoreEnabled` owner (§4). Explicit stop, pause and shutdown cancel recovery; exhausted attempts leave the unexpected-exit error visible. Install is one-shot and idempotent (detect short-circuits).
- **Security**: **user mode only — no admin/sudo at runtime** (escalation reserved for product install and uninstall time: the Windows uninstaller is elevated and runs `--uninstall-managed`, which for that reason loads only the manifests compiled into the binary, never the user-writable override directory); both optional LAN listeners terminate pin-based mTLS and reject unpinned peers — the read-only model-list listener (`em`) because a node's model inventory is cluster data, and the `ec` control listener because its routes are privileged; `em` additionally serves plaintext on loopback only, for this node's own scanner; engines bind loopback by default, but a manifest's `runtime.bind` may open an inference engine to the LAN (Ollama defaults to `0.0.0.0`, overridable per-call); downloads are HTTPS-only (plain HTTP only from loopback) and checksum-verified before execution when the manifest pins a `sha256` (an unpinned fetch is HTTPS-only with a loud warning, like a `script` install).
- **Compliance**: no PII; payloads carry engine/model identifiers and error messages only.

## 11. Assumptions
- The orchestrator owns the stdio/IPC pipe and is the only peer; on parent exit (stdin EOF) the service shuts down cleanly — no reconnect or buffering.
- Manifests are either well-formed (validated) or rejected; authors keep `engine:*` and manifest shapes stable per the versioning rules.
- Engines bind loopback unless the manifest's `runtime.bind` opts otherwise (inference engines default to `0.0.0.0`; a per-call `bind` overrides); the engine's own startup banner is not trusted — the readiness probe confirms the actual bind.
- `nvpair-errors` is best-effort; its absence degrades to local logs only.

## 12. Failure Modes and Mitigations
- **Download / checksum failure on install**: engine stays NotInstalled. → Fail fast before `run`; `engine:install-progress` error + `errors:report` (`engine-manager:install-failed:<engine>`); never execute an unverified payload.
- **Engine fails its readiness probe**: start never reaches Running. → Bounded ready timeout (or early failure after three consecutive declared auth responses, below) → back to Stopped; structured error; no half-started state.
- **Engine process crashes**: engine unavailable. → A watcher flips state, emits `engine:state-changed` + `errors:report` (`engine-manager:exited:<engine>`), then schedules the bounded saved-On relaunch episode (§4). Current On intent, pause/shutdown cancellation, switch ownership and reclaim fencing still govern every attempt; inference is not resumed. The parent's supervisor separately reports if the *manager itself* dies.
- **`nvpair-errors` unavailable**: failures absent from the node's error list. → The Broker no-ops the forward; local applog + ring buffers still hold everything.
- **Invalid / unknown-OS manifest**: that engine is unusable. → Reject at load with a specific error; other engines unaffected.

Command-mode engines declaring `runtime.auth_statuses` and
`runtime.app_processes` (currently LM Studio) diagnose failed start commands,
failed readiness, and health-probe failures through the existing `errors:report`
message. Start failures retain `engine-manager:start-failed:<engine>` with
severity `error` and action `retry`; health failures retain
`engine-manager:unhealthy:<engine>` with severity `warning` and action `none`.
`engine:state-changed{EngineStatus}` and the error payloads are unchanged; no
payload field was added. With `<Name>` replaced by the manifest's `display_name`,
the messages are:

- API-key refusal (a declared HTTP 401 or 403): `<Name> is running but requires an API key, and PAIR does not have one. Turn off API authentication in <Name>'s server settings, then try again.`
- App or daemon process present but no connection: `Can't connect to <Name>. Does it have an API key set, or is its server off?`
- No matching process: `Can't connect to <Name>, and no <Name> process is running. Open <Name>, then try again.`

Auth evidence takes precedence over process inspection. Without diagnosis fields,
or if process listing fails without auth evidence, the original failure wording
is retained. Before a command-mode start, a declared auth response from the
effective ready endpoint refuses the start without running either the start or
cleanup stop command, leaving the listener unadopted. Explicit Stop and shutdown
retain their existing behavior. After a start command is attempted, three
consecutive readiness responses with a declared auth
status fail readiness early (about 0.6 seconds for immediate responses); any
other result resets that count. Other failures retain the existing readiness
timeout (60 seconds for LM Studio), and failed-start stop-command cleanup is
unchanged. This diagnosis does not add LM Studio API-key storage or use.

LM Studio readiness remains its HTTP probe: process presence never makes an
engine running, healthy, adopted, or routable. Diagnosis runs on a start attempt
(including automatic relaunch) or in the health loop of an engine PAIR started
or adopted through Start. An engine adopted only by a passive status read has
no health loop and gets no health diagnosis. A health failure is reported even
if a status read already changed `healthy` to false, and the report clears when
the health probe recovers. Merely opening LM Studio does not trigger diagnosis.

## 13. Observability
- **Logging**: structured `slog` to stderr via `nvpair-shared/applog` (captured by the parent's debug panel); per-engine stdout/stderr ring buffers; key events — install fetch/verify/run outcomes, start/ready/stop transitions, health flips, action invocations, every failure.
- **Metrics**: lightweight in-process counters (installs, starts, failures, health transitions per engine), surfaced via `engine:status` and `engine:errors`; no separate metrics endpoint in v1.
- **Alerts**: surfaced through the errors pipeline — `errors:report` for install/start/health failures and manager crashes; the UI's error list is the alerting surface for v1.

## 14. Sample usage
The operator opens the UI; the Broker calls `engine:get-installed`, which returns each manifest's `EngineStatus` (e.g. Ollama `installed:true, running:false`). To set up a missing engine, the Broker sends `engine:install {engine:"ollama"}`: the service downloads the per-OS user-scoped package, checksum-verifies it, runs it user-mode, and re-detects — progress streaming as `engine:install-progress` notifications.

The operator starts it: `engine:start {engine:"ollama"}` resolves the manifest runtime (`OLLAMA_HOST` = `{host}:{port}`, `{host}` from `runtime.bind` defaulting to `0.0.0.0`; per-call `port`/`bind` override either), spawns the process, and waits for the readiness probe before reporting `running:true, healthy:true` in the response and an `engine:state-changed` notification. A periodic health probe keeps the state current; an unexpected exit emits `engine:state-changed` + `errors:report`. If something is already serving the port, `start` **adopts** it instead of spawning a duplicate.

The operator stops it: `engine:stop {engine:"ollama"}` signals a process the service owns. For an **adopted** engine (no owned process), it resolves the PID bound to the port and terminates it only when that process is running the binary we manage — reclaiming an orphan a prior run left on our own managed port; a genuinely foreign listener (a different image on a different port) is declined with an error naming its PID and image. A user-initiated `stop` records the OFF intent regardless (even when the RPC returns an error), so the health loop and restore-on-restart don't flip the engine back on — clients must not treat a stop error as proof the OFF choice was discarded. The cluster `ec` stop endpoint shares this semantics and may return HTTP 500 while OFF is persisted. Local and peer Off retain their provenance; either permits a later granted peer activation when unlocked and available.

The operator pulls a model with `engine:action {engine:"ollama", action:"pull_model", params:{name:"llama3.2"}}`, which issues the manifest-declared `POST 127.0.0.1:{port}/api/pull` through the streaming pull owner. Local `engine:pull-progress` and remote `engine:remote-progress` frames carry the target model, download ID and lifecycle snapshot. Unchanged lifecycle/percentage updates are coalesced. Streaming Ollama and llama.cpp pulls use a 30-minute inactivity watchdog refreshed only when a layer/file completed-byte count advances; duplicate progress and heartbeats do not keep a stalled pull alive. The NDJSON adapter requires `{"status":"success"}` before emitting `stage:"success"`; EOF or 100 percent alone is unknown. Failed or unknown pulls emit `stage:"error", percent:-1` with the fixed reason in their lifecycle snapshot, in addition to the JSON-RPC error. A CLI-driven pull (LM Studio's `lms get`) has indeterminate progress and retains the fixed 30-minute action timeout. Its successful command exit must not contain an explicit error result. On `shutdown` (or stdin EOF) the service stops every running engine first, so none are orphaned.

For the `llamacpp-models-sse` adapter, the monitored model must match
`params.model`. Subscribe before starting, and bound the `POST /models`
acknowledgement by a separate 30-second total timeout that survives caller
cancellation. Once accepted, any exit before a matching `download_finished` or
`download_failed` event performs cleanup before returning: caller cancellation,
remote disconnect, inactivity timeout, SSE read failure, and premature SSE EOF
all take this path. Using the same captured router URL and a fresh five-second
context, query `GET /models` and send `POST /models/unload` with the exact model
only if its status is still `downloading`. Missing or completed models need no
stop request, and persistent cache files are retained. Validate the unload
acknowledgement (`success:true`); join cleanup failures to the original error
without losing cancellation or inactivity causes. A lost or malformed start
acknowledgement means acceptance and cancellation cannot be confirmed: do not
unload a download without confirmed ownership. Neither terminal SSE event
triggers cleanup. These operations retain the existing JSON-RPC and progress
payloads and do not terminate the entire engine when cleanup fails.

## 15. Current integration / wiring

### Broker-local switch executor

The broker owns ordering, strict drain and publication; engine-manager owns
process lifecycle, intent, ownership checks and local veto. Only the parent
stdio/IPC channel exposes `engine:switch-prepare`, `engine:switch-step` and
`engine:switch-reconcile`. Internal execution results contain the shared durable
checkpoint, observations and coded error; no settings-worker error flattening is
permitted. Apply/status/fence ec routes relay to the broker, never to those executor
methods, and never provision or forward another hop.

Construction of the broker-managed Manager closes its independent direct ec
mutation gate before the listener starts. The broker opens it only after
successful conservative recovery, using the parent-only
`engine:switch-recovery-ready` request with `{managerIncarnation}`. The manager
checks the current incarnation and absence of an unreleased checkpoint under
the mutation-admission lock, then acknowledges the same incarnation. Mutating
ec routes return HTTP 503 until then, including stop, install, settings apply,
policy/availability changes and switch apply. Read-only queries and event
streams remain available. This is separate from local pause, and neither blocks
authoritative state deliveries nor conservative parent inspect/reconcile/release.

Prepare's short mutation-admission gate snapshots configuration and durably
reserves both IDs before probes. Competing admitted mutations refuse prepare;
reservations refuse lifecycle, port/configuration, install/uninstall and
restart-after operations with busy (`-32000`). Validation failure durably
releases a manual local provisional reservation; remote and automatic reservations
remain until broker reconciliation/release. Automatic restore skips reserved IDs;
Wake/Sleep do not rewrite intent. Shutdown remains terminal cleanup and may stop
reserved owned processes after cancelling their starts.

An empty prepare envelope returns a read-only manager configuration snapshot.
Prepare adds `configuration` (`SwitchConfigurationSnapshot`) and
`configurationFence` to the shared result. The manager fence excludes dynamic
running/health/allocated ports and contains its incarnation. The broker supplies
its authoritative settings revisions, node identity and journal generation when
computing the public `e3v1` revision; the manager fence goes unchanged into every
matching executor envelope. Settings projections also supply last-published
per-engine revisions in the manager snapshot.
Every nonempty prepare envelope includes `managerIncarnation` copied from that
snapshot. Prepare and forward steps (`source-stop`, `target-start`) compare it
with the current executor incarnation before returning an idempotent success or
performing effects. A mismatch returns `-32000` with error kind `interrupted`,
even when the envelope matches an inherited checkpoint. Reconcile and release
still accept that inherited envelope; restart never resumes forward execution.

`engine-state.json` optionally contains `pendingSwitch` with the shared
`SwitchCheckpoint` fields: exact operation envelope, phase, pre-intents and
provenance, observations, target commit flag and latest error. The file
also carries optional adjacent `pendingSwitchPorts: {source, target}` execution
metadata. A port-zero target receives its port before `start-pending` is saved,
and reconciliation restores those endpoints before probing or orphan cleanup.
Missing interrupted-target endpoint evidence retains the reservation and refuses
cleanup success. This metadata is excluded from configuration tokens. Old files
without it and legacy local Off provenance retain their semantics. Source Off
and `stop-pending` use one durable write before `doStop`; target On and
`target-committed` use one durable write after `doStart` readiness. A failed stop
never starts the target. A failed start never restores the source and saves
target Off without downgrading a concurrent local Off. Other engines' intent is
untouched. Runtime state and termination evidence remain separate from intent.

Steps compare the whole operation envelope, serialize on the existing engine
operation lock, and retain successful or failed checkpoints for idempotent
retries. Uncommitted interrupted work is reconciled toward cleanup, never forward
startup. Target cleanup, committed target On recovery and later fresh switch
target starts share the positive-port, managed-install and executable-image
reclaim predicate before readiness or adoption. Committed target On recovery
reclaims a matching orphan and launches cold; foreign listeners are preserved
and produce the existing port-ownership error. Reclaim verifies PID exit and
listener refusal, honours caller/shutdown cancellation, and uses immediate
`taskkill /T /F` for Windows console orphans while retaining Unix graceful stop.
A failed or cancelled reclaim reports `StartFailed` and fences passive and
explicit-start adoption until verified stop or successful owned cold start;
failed or no-op stops do not clear that fence. Listener refusal alone is
insufficient: the recorded failed-reclaim PID must be verified exited. A live
or reused PID, or an inspection error, retains the fence; retained identity never
authorizes signalling without the current precise reclaim predicate.
The broker must persist its terminal receipt before release. A pending stop or
start cannot be released until cleanup is reconciled. Released checkpoints retain
the last identity until a subsequent prepare replaces them; the broker's journal
owns older receipt retention. Unreadable checkpoint metadata fails mutation and
restore admission closed.

Local and remote prepare/start recheck lock (`-32013`) and either-origin pause
(`-32011`); target refusal precedes source stop. Explicit local Off
cancels remote target work and durably records local provenance before waiting
on the operation lock for cleanup; that vetoing stop returns its actual result.
This admission does not acquire the switch lock or engine operation lock and
also applies to recovery reservations and automatic operations of either origin.
A local Off during a manual local switch is
refused busy without changing intent.
For parent-only `engine:stop`, optional `switchOperationId` binds a broker-admitted
Off to its yielding reservation. Stop admission checks the operation ID, cause/origin,
engine membership and unreleased phase atomically under the admission lock.
If that scope has settled or changed, the entire request succeeds as a no-op,
with no cancellation, physical stop or intent write against later work. Missing
checkpoints (including rejected prepares) also make scoped delivery a no-op.
There is no process-local rejection eligibility or fallback. The broker journals
the admitted obligation and discharges it through an ordinary unscoped stop on
the current manager while holding its operation lock, before release/resolution.
Internal parent `verifyStop: true` requests a `SwitchEngineState` reply requiring
saved Off and timestamped verified termination; ordinary stop replies remain status.
This also works without a checkpoint after manager replacement and precedes
restore/readiness. An older direct peer start loses to the later admitted Off;
a later local On cannot interleave with broker discharge.
A remote prepare that reserved keeps its provisional checkpoint on rejection
until final reconciliation/release. Reconcile physically stops only engines with
operation-specific `sourceLocalOff`/`targetLocalOff` checkpoint flags, set by scoped
or ordinary stop admission and the reconcile veto envelope. Historical local Off
on an adopted running source is observation-only after a failed drain without
a new veto. Release requires termination evidence for those flags, rather than
all saved Off provenance. Cleanup failure retains the checkpoint.
The existing projection
path cancels activation on either-origin pause. Local switching also requires
an unlocked target and an available node.
Before a remote reservation is released, new local Off of either engine also
cancels an already-committed target. Either-origin pause fences committed targets
of local and remote reservations. Reconcile durably changes the target to
target Off with pending cleanup, then stops it without restoring source. In the
absence of an explicit veto, restart reconciliation preserves committed On.
The broker-local reconcile envelope may include `localVeto: {engine, pause}`
for the matching remote or automatic operation. `pause: true` requires an empty engine;
otherwise engine must be the reserved source or target. Reconcile durably saves
the veto (and local Off provenance for an engine veto) before cleanup. Release
serializes its checkpoint read/write with durable Off admission and refuses a
vetoed committed target until reconciliation cleans it to verified Off.

Automatic prepare/checkpoint and receipt contracts optionally include
`automatic: true`. This is execution cause, never caller authority. Forward
execution shares the initial deadline, capped at 90 seconds, including prepare,
source stop and failed target startup cleanup. An expired cleanup retains the
owned handle, endpoint and checkpoint until termination can be verified.
Manual E4 retains its defaults. Automatic replacement requires
the selected sole running source to be PAIR-controlled, non-adopted and without
reclaim uncertainty, rechecked inside reservation and before source effects.
Automatic reservations fence mutations of other engines to preserve that fact.
The source stop retains its observed owned process/run identity through the
durable Off write and into the existing stop owner. Losing that identity refuses
automatic stop without reclaiming an adopted replacement. Owned-process grace,
forced-signal exit waiting and command stop use the remaining automatic deadline;
an expired wait is not termination proof. Manual stop wrappers retain their
existing background-context behavior. Verified stopped engines remain eligible
as later targets, including a switch back to the previous source.
Adjacent `pendingSwitchPorts` also retains the original automatic source's PID
and control run ID. These are execution fences, never signalling authority.
Automatic source recovery narrows the unchanged managed-install/image reclaim
predicate to that original child, refuses adopted replacements, and requires
original PID exit as well as endpoint refusal. Lost command-server control
identity remains fenced. An ownership refusal before physical stop is saved
back as reserved, so cleanup cannot turn that refusal into orphan reclamation.
Explicit local Off and Pause keep their existing independent cleanup authority.
Automatic reconcile calls carry optional private `cleanupDeadline` separately
from the immutable prepare envelope. The broker supplies its cleanup context's
absolute deadline: live failure cleanup remains inside the original operation
window; later conservative recovery supplies its fresh cleanup window. The
manager clamps this to its existing drain/cleanup bound. This field grants no
forward execution or caller authority and is absent on manual calls.
No reclaim predicate is weakened, and server readiness does not prove a model
is available. Automatic checkpoints retain the existing conservative recovery
and release rules. The broker owns automatic ID generation and settled receipt
pruning; unresolved or undisclosed cleanup never expires into safety.

Switch relay requests use independent IDs and a bounded pending map. Replies
preserve numeric error codes; waiter cancellation sends `engine:switch-cancel`
and does not cancel an accepted operation. The existing pin gate supplies caller
identity and the target replaces the incoming node alias.

`engine:remote-switch-fence` resolves the peer through the existing pinned
client and sends POST `/v1/engine-switch/fence`. The route is a management
mutation, requiring a fresh grant, and relays local `engine:switch-fence` with
the authenticated caller. Validated shared `SwitchFenceRequest` is
`{nodeId?, requestId}`. The target normalizes node identity; no onward hop,
executor effect or payload-supplied caller is allowed. `SwitchFenceResult`
validates exactly `{nodeId, requestId, fenced:true}` or
`{nodeId, requestId, fenced:false, receipt}` with matching receipt identities.
An owned accepted receipt wins unchanged; a foreign receipt is not revealed.
Absent fences persist before acknowledgement. Broker journal `tombstones`
holds at most 32 insertion-ordered `{caller, requestId}` pairs, with oldest
insertion eviction and nonempty ASCII identities of at most 128 bytes each.
Duplicate pairs do not write; other callers may use the same ID with a fresh
revision. Receipts keep their global ID ownership rules and never evict.
Retained owned tombstones refuse switch mutation with kind `fenced` and code
`-32010`. The unchanged journal generation increment makes the original
payload's expected revision stale after eviction/restart as well. No new token
is returned: clients obtain a fresh baseline after acknowledgement. Byte
capacity preserves active terminal reservations. Capacity, cancellation or
storage failure is not acknowledgement, including an uncertain replacement
retried within the same process; trusted restart may read the durable journal.

Shared `DecodeSwitchFenceResult` is the acknowledgement decoder: exact unique
field names, an explicit boolean `fenced`, the validated exclusive union, no
explicit null receipt and no trailing JSON. The receipt and its nested objects
reject duplicate, unknown and case-variant keys; existing `SwitchReceipt`
validation owns values and identity. Other switch decoders remain unchanged.

An authoritative broker RPC operation error uses existing `writeCodedError`
and the HTTP 409 envelope with `code:"operation_error"`, preserving numeric
`-32000` and the exact target message at the outbound switch client. This keeps
`Unknown switch receipt for this caller.` distinguishable from generic
transport/coordinator failure without matching diagnostic suffixes. The relay
does not carry `SwitchError.Kind`, so no distinct unknown-receipt kind is
inferred. Existing aliases and other control families' classification remain
unchanged. Plain failures remain uncertain and cannot establish absence.

### Download lifecycle and content admission

The canonical model inventory includes optional version-1 `downloadState`:
`{version,sourceId,revision,recoveryRequired,entries}`. Source is the manager's
fresh switch incarnation; revision is a canonical uint64 decimal string and
never wraps. Entries are `{id,engine,model,phase,blocksReadiness,percent?,reason?,persistence?}`.
ID is 32 lowercase hexadecimal characters; engine/model retain
the shared identifier limits. Phase is `queued|in_progress|failed|complete|unknown`.
Reasons are `busy|capacity|download_failed|interrupted|progress_timeout|acceptance_unknown|cleanup_unknown|persistence_error|unsupported`.
Ordinary progress and success omit reason. Only measured integer percentages
in 0–100 are present. Persistence qualification is `uncertain`, for a landed
marker removal whose durability was not confirmed. No arbitrary native text or
filesystem path is part of this contract.

Snapshots are full, bounded to 256 KiB and 64 entries, with unique IDs and
engine/content keys. Missing or malformed support is unknown; an explicit
current empty entries array is known empty. Receiver currentness is separate
`downloadStateCurrent` metadata on discovery observations. Queue/progress/
terminal notifications carry exact model, ID and snapshot, while remote streams
also retain request `opId`. The stream decoder validates both correlations.

One Executor gate owns reservations, active writers, reader leases and unsafe
memory. It is never held during another owner's wait, I/O or notification.
The journal writer mutex is acquired outside that gate and engine/inventory/
publication locks, ordering capture/save/adoption across engines. Journal
publication precedes native effects. Pre-replacement failure retains old bytes;
landed uncertain addition refuses effects and retains unsafe content; landed
uncertain removal is adopted with explicit qualification. Recovery retains
unsafe keys under the new source as unknown without starting network/model work.
Naming-contract changes require conservative recovery. Limits are 32 accepted/
unsafe keys, 32 recent outcomes and 64 KiB journal bytes; unresolved keys are
never evicted. A pre-effect failure does not invalidate previously ready content,
but never clears an earlier unsafe marker.

Success resolves a marker only through its same-key native terminal from the
current dispatch run and journal removal. Delete requires successful mutation
plus current unmasked absence with every row identified; ambiguous identities
retain the marker. llama.cpp must reconstruct its cache using `/models?reload=1`
without redirects. Revision capacity is reserved before durable Delete resolution.
Inventory/digests
are fenced by engine run and content epoch and re-masked at publication. Loaded
observations are independent and remain visible even for unsafe content. New
Load/export readers and content writers exclude each other without waiting for
content under `opMu`. Existing load dispatches and inference are not cancelled.
No journal mechanism certifies historical vendor files, resumes downloads or
implements a general job/history service.

### Product integration

The engine manager builds standalone (`go build ./...`) and is tested end to
end over stdio. Current product wiring is:

1. `build.bat` and `build.sh` build the version from `versions.json` and stage
   the binary in the repository-root `build/bin/` bundle.
2. `installer/nvpair-setup.nsi` bundles the Windows binary and owns the local
   subnet firewall rules for its model-list and cluster-control listeners.
3. `nvpair-ui-broker` supervises the process, relays its `engine:*` JSON-RPC
   surface, forwards engine events and errors, and registers its applicable
   discovery services.
4. The bundled graphical UI, `nvpair-tui`, and other clients use the broker
   contract rather than owning engine-manager process wiring. Product packaging
   places the graphical UI alongside the broker and workers in the same
   installation directory.
5. `nvpair-engine-manager/README.md`, `nvpair-ui-broker/README.md`, and the
   cross-process tests are the current architecture, protocol, and
   integration references.

The retired desktop wiring is intentionally omitted; its implementation record
remains available in Git history and is not part of the current architecture.
