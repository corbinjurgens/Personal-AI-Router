<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Microservice: Engine Manager (`nvpair-engine-manager`)

## Native build identity and configuration keys

`nativeBuildID(ctx, st)` returns `sha256:<hex>` over JSON containing
`version: "pair-native-build-v1"`, `platform: "GOOS/GOARCH"`, `executable`
(the full-file SHA256) and `libraries` (objects with lower-case JSON keys `name`
and `sha256`, sorted by name). The library set contains the executable
directory's files accepted by `modelDeviceLibraryName`. No executable name,
path, mtime or manifest pin enters this content identity. It applies to managed
and override runtimes, including `install.fetch` installs. Full-file hashing
retains cancellation, the existing file size bound, regular-file checks and
library confinement. The process-local hash memo uses resolved path, size and
mtime and rechecks object evidence before reuse; nothing is persisted.

Every typed manual or Auto `ResolvedLoad.NativeBuildID` holds that fingerprint.
Auto's reviewed input envelope separately binds its native source digest;
frozen compilation and conversion compare the appropriate identity without
consulting telemetry or repeating native enumeration. The native source digest
contains only `{BuildID, Environment}`. Environment includes explicit launch
values and effective `CUDA_*`, `GGML_*` and `LLAMA_*` controls, with the checked
build root replaced by `{build_root}`. Unrelated ambient values are excluded.
Finite configuration keys bind this source alongside artifacts and allocation
options, so unchanged build bytes retain keys across touches and moves.
Expected-source replay checks still reject executable or library replacement.

Target-owned stated-memory records also retain `nativeBuildId`, copied from the
compiled applied configuration rather than accepted from a caller. Inspection
projects it with the record's configuration key and currentness. An earlier key
remains readable history, labelled with its build as an estimate, and never
becomes current measured or validated evidence. A record without retained build
identity reports that absence. GPU-selection loads remain unpriced and have no
finite key for a statement or buffer overlay.
This read-only calculation adds no persistent mutation or native initialization;
device helper gates and pinned advisory defaults are separate checks.

Conversion captures the executable and complete native library evidence before
reading external model artifacts. Its expected-input whitelist includes those
original objects and rejects a missing library snapshot; replay does not
rediscover files to replace missing or changed evidence. Accepted conversion
validation captures the same complete evidence at the start of its operation.

The inherited SWA allocation default remains conditional on the known managed
b12020 artifact declaration and checked native files. This assurance belongs to
the load estimator, independently of device report recognition. A recognized
device report from another build does not establish an inherited SWA default;
it stays unknown until separate semantic evidence is available. This does not
gate content identity or configuration-key availability.

## Running native build readback

After owned llama.cpp readiness, a bounded empty-model `/props?autoload=false`
read supplies the optional `EngineStatus.buildInfo` claim, while file-only
content hashing supplies `buildId`. Publication is fenced by the owned run and
process; stop, exit, replacement and external adoption clear the values.
Inspection exposes those saved observations as `nativeBuild` and `nativeBuildId`.
An external listener is not assigned the detected installation's fingerprint.

Buffer capture attempts every build. Recognized grammar yields `recognized-output`
assurance; no recognized allocation output yields a partial `unrecognized-build-output`
record with no rows. Records bind `buildId` separately from the `reportedBuild`
claim. Valid other-build records never disable writes. Inspection prefers the
newest matching-build record, otherwise retains earlier-build history with
currentness withdrawn. The existing stored decode folds former known-pin records
into `reportedBuild` with an empty build identity; no migration pass is added.

## Model identity

The executor's in-memory model identity owner reads the latest stored inventory
for llama.cpp-family engines only. It resolves checked local artifacts under
download read admission and the inspection budget. Each background scan attempt
tries the engine's content semaphore without waiting, yielding to reserved mutations;
a mutation waits behind an admitted scan with its existing context. Snapshots
perform no I/O and detach file slices. Runtime contention can retry the complete
read within the bounded read-observation window described below. Busy reads
after that bound retain the previous entry; resolution failures
remove it. Explicit refresh, changed run/configuration and a ten-minute interval
invalidate resolution. Each full inventory sweep requests a coalesced refresh
without waiting for artifact I/O.

`engine:models`, `GET /v1/models` and `engine:models-changed` carry optional
`modelIdentities: { <engine>: { <model>: ModelIdentity } }`, using the shared
`engines.ModelIdentity` JSON fields. Publication attaches a detached owner
snapshot after readiness masking, restricted to exact engine/model keys still
present in `modelsByEngine`. Unlisted, unavailable and readiness-blocked models
have no attached identity, even when the identity owner retains metadata for
them. Every attached file slice remains detached from the owner. The loaded
watcher compares attached per-engine snapshots as JSON and emits the existing
`engine:models-changed` notification for identity-only changes and withdrawals,
so consumers update without polling. This does not change inventory, readiness
or observed residency.

HTTP inventory omits optional identities when the complete encoded body exceeds
the existing peer inventory limit. JSON-RPC responses include their actual
correlation ID in the frame budget, and model notifications include their full
envelope. If identities exceed either publication budget they are omitted while
routing names and the identity owner's stored snapshot remain intact. The loaded
watcher compares the publishable snapshot, including metadata withdrawals.

Identity records base file names, roles and sizes; model-role files alone supply
total bytes and the serving-content formula's content ID. SHA256 values come
only from existing free blob-link names, never whole-file hashing or a persisted
hash store. Repository and revision come from the HF cache ID/snapshot path,
then configured cache path segments, then a Hugging Face GGUF repository URL.
Quantization prefers the HF tag, then the model file name, then GGUF file type.
Architecture, trained context, general name and size label use bounded GGUF
facts. Configured context is PAIR's reading: typed per-slot context when present,
otherwise context divided by parallel from env, wildcard, section and CLI in
that precedence order. It is not a native runtime observation.

These identities support display and grouping only. The `verified` strength
means every model-role file supplied a free digest; it grants no strict
admission, readiness, residency or hybrid eligibility authority.

## Windows engine working-set limit

Engine settings accept optional integer `workingSetLimitMiB`, from 256 through
1048576 inclusive. Absent or null means no limit. The target refuses a non-null
value on non-Windows hosts, non-llama.cpp engine families, or runtime modes other
than `process`, with a field error keyed `workingSetLimitMiB`. Preview reports
the value change in `diff` and requires a restart when the engine is running.
Apply validates before stopping or writing, then uses the existing stop,
persist, rebind and conditional restart operation. The host runtime override
stores `working_set_limit_mib`; clearing it masks an inherited limit with null.

Windows applies the same hard maximum, in bytes, to each owned engine process
and descendant. The root starts suspended, is assigned to a limit-free job,
capped through `SetProcessWorkingSetSizeEx` and verified by readback before
resuming. Failure kills and reaps the root and fails start. Job notifications
trigger an immediate sweep of the engine job's members, and the manager also
sweeps on a 250 ms target cadence, so a lost notification delays a descendant's
cap by about one interval rather than indefinitely. A process can fault in pages
before its first cap; the late cap trims its working set to the maximum. A member
that cannot be opened terminates the engine job; a live member whose cap fails
is terminated. A notification read or job enumeration failure terminates the
engine job. This limits each process
working set, not aggregate tree memory or committed memory. The engine job has
no kill-on-close lifetime policy.

`LaunchState` includes the saved `workingSetLimitMiB` and
`workingSetLimitStatus`; the broker projects the status onto its settings
snapshot. Status is omitted when no limit is saved. `not-running` means stopped;
`unenforced-adopted` means the running engine is adopted or was started
without the current value. Adoption cannot enforce new descendants after a
manager restart, so an engine restart is required. A failed spawn leaves the
saved limit in place and reports the failed start, as with other runtime keys.

`unverified` means this manager started the running engine with the saved value,
but its containment has not verified every live member: a member could not be
capped (it is terminated, and the status returns to `applied` after the next clean
sweep), or the job could not be read (the engine job is terminated). `applied`
means a full verification (the root cap before resume, then each sweep on a
250 ms target cadence) completed within the last second and found every live
member capped. A stalled or stopped watcher also reports `unverified`.

## Private model memory device metadata

`mapModelSettingsDevices(ctx, operation, runner)` consumes an operation snapshot,
not public settings or an uploaded capacity claim. Inputs are bounded host/run/
topology identity, physical pools, selected native selector, complete effective
environment, content-identified library hashes, and an explicit native-init gate.
The runtime snapshot resolver accepts managed, fetch and override llama.cpp
builds using the selected executable's directory; it does not require a manifest
pin or managed path and does not accept a public library path.
The adapter is review-only and has no startup, Apply or recovery caller.

CPU maps to one host/shared pool without native initialization. GPU identity
requires dedicated physical PCI/full UUID agreement; Windows additionally
requires the driver's LUID to agree with supplied DXGI LUID. A second helper
enumeration under full UUID-only visibility must expose exactly one GPU,
`CUDA0`, with that identity. The binding returns host, pool, UUID, PCI, LUID,
selector, constrained child environment and a digest of physical/native/build/
library/environment identity. No memory bytes, sample times or producer run
enter that digest. A separate bounded observation digest binds host, run and
the exact supplied topology observation. Pool ordering is canonicalized.
Failed reports clear native inventory and binding; they confer no replay
or capacity authority. Missing identities, duplicate aliases, ambiguous joins,
unsupported exports/ABI, MIG/MPS, mixed UMA/discrete and split placements refuse
a GPU binding. Unsupported Metal/Vulkan/HIP remain visible as Unknown.

The existing executable recognizes `--private-model-devices` before flags,
registry, transport or service startup. Its stdin is an internal selected-runtime
snapshot capped at 64 KiB. It verifies the selected library set/hashes and reports
`ggml_commit()` as bounded metadata before CUDA registration or property calls.
It checks
64-bit `ggml_backend_dev_props` layout and required exports, and enumerates at
most 64 devices. The ggml description must match the CUDA driver name and total
memory must agree within 1 MiB for the same PCI identity. Those values are
recognition checks, not capacity authority. Missing exports, loader failures,
undecodable values or disagreement return `device report not recognized for this build`
with no partial report. Recognition alone is cached by content build ID; each
operation still checks topology and UUID-constrained visibility. Several dedicated
pools run no helper and report `several GPUs: reviewed Auto supports one GPU; choose GPUs with extra options`.
`ggml_backend_load` loads
only the selected CUDA module and checks the native engine ABI; no `load_all`
or ambient engine search is used. Linux requires the effective library search
path to be exactly the selected install directory. Windows uses DLL-load-dir
plus System32 dependency resolution and System32 for the CUDA driver.

The parent uses a three-second context budget (including reserved pipe drain),
combined 64 KiB stdout/stderr limit, reserved provider-secret filtering, hidden
Windows child, command cancellation and `Wait`/`WaitDelay` reaping. It never
logs raw native diagnostics or environment. The mode performs metadata reads
only, with no engine launch, model loading, listener or network request. Those
reads can initialize a driver context, so operation permission must be supplied
by the existing activation/Pause owner. CPU remains available when prohibited.
An independent helper-local hard exit is armed before input parsing, bounding
native calls even if the manager dies. Windows cmd.exe pseudo-variables (names
beginning with `=`) are excluded before effective-environment merging, validation
and identity calculation. Empty names and NUL, CR or LF still refuse. Unix
environment keys keep their exact case. Linux review and replay set
`CUDA_MPS_PIPE_DIRECTORY=/dev/null` after verifying the character device,
excluding implicit default-directory MPS.

Primary ABI/selection sources:

- [Pinned ggml metadata declarations](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/include/ggml-backend.h)
- [Pinned ggml module ABI check](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-backend-reg.cpp)
- [Pinned ggml revision export](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/include/ggml.h)
- [MPS pipe directory](https://docs.nvidia.com/deploy/mps/latest/appendix-environment-variables.html#cuda-mps-pipe-directory)
- [CUDA driver UUID/LUID APIs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html)
- [CUDA visibility and enumeration contract](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)

Task 5 must join target telemetry/topology, operation permissions, public
inspection/preview and target-owned frozen settings to this adapter, and compile
the returned environment/selector through serving. Full topology is not present
in the current memory-domain wire. No hook into held settings/inspection/serving
files is part of this foundation. Unit sources use injected runners and the
existing test executable's TestMain sentinel, never real libraries/drivers or a
new fixture path. Source formatting is not compilation, test or native evidence;
native acceptance remains UNTESTED.

## Parent-only build-apply owner read

`engine:owner-evidence` is registered only in Manager's parent request path.
Its sole request is `{}`. Reject null, omitted params, arrays, extra/duplicate
members, case variants, engine/node selectors, caller, operation ID and mutation
flags with -32602 and fixed text `Invalid engine owner evidence request.` It
adds no ec/control handler, ready notification, subscription or mutation.

The shared wire contract is `buildapply.EngineOwnerEvidence`, with exactly:

| Member | Meaning |
| --- | --- |
| `managerIncarnation` | Nonempty existing Executor `managerIncarnation`, even with no instantiated engines. |
| `captureToken` | Canonical digest of incarnation and every detached section, knowledge state, reason and derived completeness. |
| `universe` | Observation of the sorted complete engine union. Each entry has `engine`, `registered`, `supported`, `installed`, `running`, `engineRunId`, `terminationVerified`, `modelRoots`, `dataRoots`, `rootsComplete`. |
| `stopAuthority` | Observation of the same sorted engine keys, each with `engine`, `scoped`, `adopted`, `reclaimPending`, `processLifetime`, `reason`. |
| `reservations` | Observation of `{revision,blocksApply,unsafeCopiesComplete,operations}` from the joint copy/download/content owner. |
| `complete` | Derived knowledge/coverage, never an acceptance or mutation permission bit. |
| `reasons` | Sorted bounded `{section,engine,code}` identifiers; no native errors, commands, credentials or private transport tokens. |

Every potentially unknown fact uses `Observation` (`known`, `value`); unknown
requires null, known requires a value. Known-empty lists are arrays, never null.
Registered membership is known from the captured registry; support of an
unregistered retained key is unknown. Unobserved runtime booleans cannot become
known false. `terminationVerified` retains positive owner stop knowledge.
Unsupported engines with retained cleanup evidence remain in the union.

`processLifetime` is an Observation of `{engineRunId,pid}`, where `pid` is an
Observation retained from the owned process object. Neither that object nor its
pointer is serialized; saved cleanup PIDs and OS lookups confer no authority.
Adopted/external engines retain the existing scoped refusal. Closed or replaced
lifetimes, pending reclaim, changed configuration and admitted mutations withdraw
the dependent knowledge.

Roots contain only resolved owner-known paths. `modelRoots` represents configured
model roots; `dataRoots` includes the managed install root. These lists do not
claim exhaustive coverage unless `rootsComplete` is true. Unresolved paths stay
unknown. The current manifest does not declare exhaustive data roots, including
for custom/external engines, so that coverage remains incomplete. Empty complete
universe/authority lists remain valid. Consumers must refuse unknown root
coverage before package inspection and merge represented roots with their own
protected profile roots; no filesystem scan or argument/environment inference
is allowed.

Each reservation operation has `id`, `engine`, `kind`, `phase`,
`sourceIncarnation`, `retained`, `unsafe`. Kinds are `copy`, `download`,
`content_reader`, `content_writer`; phases are bounded owner states. Source is
an Observation, explicitly unknown for historical downloads without retained
origin. A live operation and its unsafe marker share one identity. Content gate
entries receive deterministic opaque IDs. The reservation revision includes
native/download source, copy and download revisions, content authority generation
and engine admission generation. Source mismatch withdraws the entire section.
Missing historical copy baseline, journal recovery, uncertain retention or
unknown origin never implies safe empty reservations.

The handler repeats detached captures from existing saved-state, universe,
admission and joint reservation owners. Changed captures withdraw all sections;
contention leaves affected sections unknown with bounded reasons. It performs
no state creation, Status/GetInstalled, detection, reclaim,
adoption, OS lookup, native HTTP, model read, readiness check, recovery or write.
No new cache, lifecycle owner or polling mechanism is introduced. Saved semantic
revisions do not certify acknowledged durable writes or absence of ABA edits.

Strict decoding validates exact nested fields, knowledge/value consistency,
nonempty identities and revisions, unique sorted engine and operation identities,
universe/authority correspondence, roots coverage and derived completeness.
`owner_evidence.go` retains `ownerDigest`, SHA256 over `json.Marshal` bytes
encoded as `sha256:` plus lowercase hexadecimal. Canonical capture validation
binds every serialized section. The existing
`buildapply.MaxRequestBytes` and `MaxStatusBytes` apply. Invalid or oversized
response evidence produces -32000 with fixed text
`Engine owner evidence is unavailable.` No authority-bearing array is truncated.

Captured-worker resident reconciliation also uses this read's native
`managerIncarnation`, even when unrelated evidence is incomplete. It obtains no
mutation permission or reservation from the read.

This is evidence only. Matching captures do not pin a remote owner or prove
physical liveness after the read. Complete evidence may contain scoped false
authority or blocking work. Production `WithLocked` acceptance remains absent;
real starts additionally need separately owned identity/membership preservation,
worker-exit reservations, saved/enforced state and physical-stop proof. Unknown
evidence refuses, and this method supplies no production start adapter.

## Host-local credentials

The manager holds no user-entered key and has no method that sets or reads
one. The only key it attaches is the one it generates for a process it
launched. Credentials are at most 512 printable non-space ASCII bytes.

A key-bearing `candidateCredential` binds selection to a valid operation port.
The owned prepared Strata launch credential is selected only for the process
that serves that port. Broker retrieval and HTTP actions additionally require
running on the exact current port; stopped and adopted engines export nothing,
and Strata export retains owned-process eligibility. CLI actions receive no
credential. Probe helpers take an explicit immutable candidate snapshot; they
do not select a credential. Readiness, health, presence and diagnosis use the
same selection owner and preserve HTTP/JSON identity and bounded readiness.
Only an attached key receiving a declared auth status yields
`The API key saved in PAIR was rejected by <Name>. Enter its current key, or turn off API authentication in <Name>'s server settings.`

Probes and actions attach keys through the same helper. Before network I/O the
initial URL must have exactly scheme `http`, hostname `127.0.0.1`, explicit
parsed port equal to the snapshot port, and no userinfo, fragment or opaque
form. A private client clones `*http.Transport`, clears Proxy and refuses
redirects (including same-origin); unsupported injected transports fail closed.
Credential-bearing failures expose fixed operation/status text, never response
bodies or raw transport errors. Successful action JSON is decoded and the
active key replaced in field names/string values before re-encoding; non-JSON
text is redacted before wrapping. Without a key, the existing port handling,
HTTP transport and errors are preserved; credential eligibility is not applied.

Local authorization is the existing `ipc.ListenPrivate` same-user 0600 Unix
socket or user-SID Windows pipe, then service multiplexing, broker stdio and
manager stdio. The ec pinned mTLS direct/reverse registry exposes no credential
or generic JSON-RPC route, even to peers allowed to manage engines. Settings
and policy relays do not carry credentials. The broker-only
`internal:local-engine-credential` path is intentionally retained for the local
proxy, with the broker refusing public client retrieval before generic relay.
No credential state notification is emitted. Public status/diagnostics/models
and peer snapshots carry no keys. Same-user code is inside the trust boundary.

## Model settings contract

`engine:inspect-model` at the target-local worker accepts only the parent
broker's worker envelope, whose `request` is `{engine,modelId}`.
The public broker request may also select `nodeId`; the broker resolves that
target and supplies authenticated node identity in the reply. The worker rejects
a nonempty node selector, rather than labelling local results as a remote node.
`Executor.InspectModel(ctx, request)` validates exact identity and cancellation,
then composes bounded artifact facts, descriptors and an advisory estimate.
The estimate consumes the same validated facts through a separate read-only
handoff, without another read, parser or resolver. Inspect has no effective
allocation and identifies its native launch snapshot as unverified; it can size
tensor storage while cache/workspace/placement remain Unknown. Artifact/estimate
inspection starts no lifecycle, inventory read, download or full model hashing.
The explicit parent-authorized operation can invoke the existing bounded
device-map helper under its Pause, activation, ownership and maintenance fences;
an observation summary neither authorizes that helper nor adds another call.

`InspectResult.loadObservation` is an optional detached, path-free summary of
the newest relevant native buffer sample, not a settings Preview field, request
field or Auto input. Actual-start compilation supplies the frozen
artifact/options/estimator baseline. The process-output and existing model-sweep
owners establish child/load lifetime, recognized build and effective context/slots;
ordinary typed loads need no `ServingConfig`. Inspect projects retained history
without another props/model-list request or history write. Missing, unsupported,
stale, ambiguous and incomplete observations remain Unknown/partial. Recovered
history cannot restore currentness.

The b12020 mapping carries recognized self-reported pin assurance only, not
proof of an undisclosed clean build. Rows retain reported decimal MiB, precision,
conservative intervals, phase and coverage; repeated buffers are occurrences,
and later scheduler reservations are not additive. Compatible comparisons use
the captured load estimate, never the current draft. Incomplete estimate scope
and unproved device mapping remain Unknown. Observations cannot become Fits,
Tight, Exceeds, physical occupancy, qualification, admission or calibration input.

Projection bounds the optional summary to 4 KiB inside the unchanged 64 KiB
complete reply limit, reserving existing advisory/target space. It reduces
optional rows with omitted coverage, then uses a compact structured
`unknown/response-budget` result when possible. If even that cannot fit, only
`loadObservation` is omitted; original artifact facts and the estimate fallback
remain available. The bridge maps absence to
`{status:"unknown",current:false,reason:"observation-omitted"}`. Invalid supplied
summaries still fail strict decoding. Derived history is separate from settings
and journals; corrupt history is preserved with writes disabled, and landed
`ErrReplaced` publication is adopted with uncertainty. See the full
[buffer-observation contract](../../docs/model-buffer-observations.mdx).

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
broker. Worker policy-edit traffic remains refused. The broker injects target-local
memory through a strict private worker envelope. Missing evidence returns
`target memory telemetry unavailable`; incomplete costs also refuse Auto.
Its private owner-injected selector and frozen compiler
are specified below. Missing or changed
artifacts and unreviewed resolutions remain refusals. Advisory estimation errors
are data, never an additional manual-preview or Apply refusal.

Direct previews compile typed records without persistence or activation.
Read-only model previews, inspection and diagnostic observation absorb brief
runtime-lock contention inside the manager. An observation that needs restarting
gets at most 21 attempts, with retries started within one second of the first
contention, within the caller's context and the existing help/metadata budgets.
Lock acquisitions inside such a whole-read retry never wait on their own, so one
contention window governs the entire observation, including diagnostic
lifecycle and effects captures. A reader holding a content lease releases that
lease and the inspection slot before retrying its entire observation; it never
waits for a runtime owner that may need the same content guard. A standalone
native help or device snapshot, which holds no inverse lock and is not inside a
whole-read retry, may wait up to one second for its lock. A completed runtime
change requires a fresh revalidated snapshot,
not reuse of earlier facts or effects. Continuous drift or contention past the
bound retains the explicit busy/conflict/unknown response without current proof.
Mutation acceptance, reservation comparisons, source/ownership checks, consent,
revision checks and stored-measurement publication keep their existing fences.
These are read retries, never automatic replay of an accepted or uncertain action.
Diagnostic reservation and exercise identity rechecks use the same inspection
owner. Busy inspection after its bound remains an explicit observation conflict;
its absent key is never classified as proof that the accepted identity changed.
A manual load Preview attaches its fresh advisory estimate only after the
authoritative read has completed and released its content lease. The estimate
uses the remaining metadata budget; its expiration yields an Unknown estimate
and never turns a completed Preview into a refusal.

Typed loads are keyed by `profileKey`. The preset section name, profile key
and served id are identical; `modelId` separately selects artifact files.
Distinct profiles may select the same artifact with different launch settings.
Launch settings live only in the engine manager, while node policy owns profile
metadata. The broker derives identity and context in its coupled policy stage.
The existing stored decode/repair pass assigns missing keys from exact model
ids before canonical validation; subsequent settings persistence records those
keys without a separate migration or schema boundary.
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
(per request times native slots), non-unified KV and disabled startup loading.
Host prompt cache accepts a nonnegative MiB ceiling; zero disables it and is
priced as zero. Missing effective cache information remains Unknown.
Fitting defaults to off, including the global preset defaults. A typed load may
set `fit = on` with optional model-local `fit-target` margins and `fit-ctx`
minimum context. Native-fit default GPU placement emits native `auto`; its
resolved layer count is a request ceiling, not observed placement. Conversion
maps `gpu-layers = auto` only with fit on, and finite `all` to the bounded
model-plus-output layer count. Native-fit loads are unpriced: applied inspection has a
diagnostic execution key, no finite configuration key or pools, an unknown
estimate and no buffer capture or stated memory record. External operator
presets retain the finite parser; conversion validates native fit through the
typed fields first. Reset removes requested leaves; inherited finite adapter
defaults are regenerated, never retained in a hidden old source. Explicit CPU
does not require estimate facts; finite layers/native-default resolution needs
bounded layer facts. f16 and q8_0 KV precision are accepted; unmapped
precision and unreviewed Auto are field refusals. Batch compatibility is checked against
the pinned effective native defaults. Metadata context is a fact, not a
universal training hard cap or permission to add unsupported scaling controls.

Conversion maps native `flash-attn = auto` to inherited policy (the pinned native
default). Native sampler defaults, reasoning budget and output token limits
survive as native extra options, below per-request overrides. Thread controls
remain typed. Projector controls remain refused except the disable forms below.

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
settings, including validated frozen Auto; combined-policy requests remain guarded. Broker
acceptance, drain and coupled policy application stay at their existing owner.
Load previews describe whole-router scope, affected configured models, the
60-second drain bound, possible interruption and whether the engine stays Off.
Unobserved loaded models remain explicit unknowns for broker reconciliation.
Unchanged typed settings do not restart an owned process. Saving Off or paused
settings never activates it; adopted processes remain read-only.
No typed settings create qualification. The broker's existing build-maintenance
settings-entry exclusion also covers this mutation; it is not a client bypass
of the independent owner-evidence requirements for production build apply.

### Reviewed Auto engine unit

The private `modelAutoMemoryOwner` supplies a target-owned snapshot through the
operation context, populated from the broker's trusted self evidence on the
parent-only worker envelope. Public callers cannot upload this authority.
The selector consumes the compiler's already validated bounded facts and ordered
artifact bindings; it performs no additional file/parser/network/native reads.
Capacity alone is insufficient: every runtime component and independent pool
must be fully priced, fresh and mapped. Task 10's incomplete graph costs remain
Unknown; synthetic complete costs in tests do not extend its support matrix.
Public launch validation does not invoke the no-facts resolver for owner-injected
Auto. The normal PreviewLaunch compiler owns baseline/artifact checks and reads
the owner snapshot once. Missing-owner validation retains the exact production
refusal; frozen Apply never reads the owner.

Default Auto context is 8192 cap and 4096 floor, with one slot. Explicit context
ranges use the valid 256-token grid; a smaller explicit cap can also be its floor.
Fixed context uses existing native padding and never shrinks. Known model context
limits Auto without enabling RoPE extension; unknown supported context refuses.
Slots, manual placement, precision and explicit options are constraints, never
search dimensions. Candidates are CPU, native-default resolved to finite layers,
or supported finite layer counts. Search orders highest context then greatest
offload, with a hard 128-evaluation budget across the entire preview. Exhaustion
reports `budget-exhausted` and the unsearched context/layer range, independently
of `insufficient-memory`. Any unpriced searched candidate prevents an insufficient
memory verdict. CPU-only can use host inputs and discloses slower execution.

Dense llama/qwen2/qwen3 output-first weight transfer with explicit host KV and
op-offload disabled proves only weight transfer: host weights decrease and device
weights increase. It does not prove complete total costs. The optional private
`modelAutoLayerCostOwner` must derive a complete layer-independent non-weight host
cost from the exact context/options, not from sampled estimates. Each evaluated
allocation must match that fixed cost and the bounded weight transfer. Only then
can bisection use the resulting per-pool bounds, never a boolean Fits predicate.
No production estimator implements this proof. Without it, or if an estimate
disagrees, discard bisection bounds and enumerate greatest layers through CPU
under the same budget. Synthetic fixed-cost formulas test total monotonicity
and the floor-only search; layer-dependent workspace fixtures test fallback.
Partial costs and unknown mapping
cannot form bounds or Fits. Independent pools reserve ceiling(capacity/10) plus
512 MiB each against snapshot free memory; UMA is one pool. This is a disclosed
uncalibrated heuristic, never admission or qualification authority.

Preview returns requested intent plus finite resolution, estimate assumptions,
memory source/timestamp and `auto/finite-llama/1` review identity. ReviewedInputID
binds model, requested preferences, finite allocation/options, sampled artifacts,
native build, estimator revision, memory snapshot hash and external conversion
source. External conversion provenance alone cannot authorize Auto. Every Apply
validates frozen identity without invoking the resolver. Before acceptance, Auto
also requires fresh sufficient target facts for that same finite allocation.
Accepted replay does not require telemetry. Cold start/recovery compilation revalidates artifact/build/source and
preset digests and emits the accepted finite options with fit off. Drift refuses;
native failure cannot trigger smaller settings or an OOM retry ladder.

Recalculate explicitly reselects Auto in a fresh selected-patch preview and
produces a separately reviewable identity. Ordinary manual edits cannot silently
recalculate saved Auto. Its baseline diff renders the selected model's previously
reviewed finite options without requiring its old native build to remain present.
The new preview binds the current build. Old Apply/restart still reject build
drift, and every unselected frozen sibling is checked normally. Copy carries
portable preferences only. Broker source,
authority, preview/revision checks retain their existing ownership. Broker
retains finite reviews and forwards full settings for frozen validation and recovery.
This does not establish native or production Auto acceptance. Manual compilation and
Task 10's advisory-only digest independence remain unchanged.

### Advisory memory estimate support

The estimator uses MIT `github.com/gpustack/gguf-parser-go` v0.27.0, commit
`9257743a3c1775b882ba1b58ec7f5a856c2cc385`. Its only library operations are
in-memory architecture projection, checked GGML row sizing and the bounded dense
CPU graph estimator. It never calls a library file/URL parser, mmap, network or
native API. Native field semantics are the finite typed compiler contract
`finite-llama/non-unified/1`; the executable's sampled build binding is included
in the result/cache identity. This identifies the inputs, not native calibration
or proof that an arbitrary executable implements the reference allocator.

Reference mappings come from the pinned library's `ggml.go`,
`file_architecture.go` and `file_estimate__llamacpp.go`, including the linked
GGML type traits and llama.cpp `16d222fc5f2c0fdc3d0180e0b772516ec6e2eddd`
hparams/LFM2 state contracts. No Unsloth implementation, defaults or constants
are used. The matrix is an advisory projection, not extra native option support:
the manual compiler continues to admit only its existing f16 KV mapping.

| Architecture or option | Estimated when mapped | Unknown or partial |
| --- | --- | --- |
| Dense/GQA: llama, qwen2, qwen3 | All tensor storage; per-layer K/V with scalar or complete KV-head array, positive/divisible embedding and attention heads, explicit widths or the native embedding/head quotient | Missing/malformed dimensions or unresolved precision/FA; zero heads without recurrent mapping |
| Qwen2/Qwen3 MoE | All resident tensors, including every expert; ordinary attention cache | MoE workspace; active experts never discount resident weights |
| LFM2/LFM2MoE | Declared zero-KV-head recurrent mask; F32 short-convolution state per layer and slot when `shortconv.l_cache` is present; attention layers alone hold KV | Missing recurrent mask/state dimensions; recurrent workspace |
| LFM and unmapped recurrent families | Tensor storage | Attention/state/workspace need a distinct native mapping; no dense substitution |
| Qwen3Next/Qwen3.5/Qwen3.5MoE | Attention KV for a complete head mask or explicit full-attention interval | Delta-net/SSM recurrent state dimensions are not fully retained/mapped; workspace Unknown |
| MLA (including DeepSeek) | Tensor storage | Compressed/shared/NextN cache and workspace need their own mapping; no dense formula |
| SWA: mapped dense families and LFM2 | Explicit window plus scalar layer pattern (LFM2 uses its declared attention mask), effective logical batch and attention alignment; separate SWA widths when supplied | Missing/malformed pattern or unresolved batch; boolean-array patterns are not mapped; SWA workspace |
| Mixed K/V | Independent f16, bf16, q8_0 and q4_0 GGML block traits; row divisibility checked for each axis | Unresolved precision; quantized V without explicit FA; no silent f16 fallback |
| CPU graph | Partial pinned library CPU graph portion for dense llama/Qwen2/Qwen3 with complete conventional tensor shapes, scalar dimensions, explicit batches/FA/precision and `op-offload=false` | Complete CPU scratch remains unsized: the library reserves hypothetical GPU scratch even at zero GPU layers and cannot disable op offload; that buffer is not charged as host. GPU graph and special architectures remain Unknown |
| CPU/full/partial placement | CPU domain or output-first tensor-based host/offload projection; full offload cache with explicit KV offload; disabled KV offload stays host | Native device identity absent without broker mapping; partial KV/state placement stays unsized; no proportional layer-fraction split |
| Draft/projector/other companions | Separate bounded tensor storage subtotal | Runtime cache/workspace and independent placement; absent is explicitly excluded, not Unknown |
| Prompt cache and exclusions | Emitted `cache-ram` ceiling, explicitly demand-dependent and distinct from actual usage | Driver/runtime, alignment/fragmentation, mmap residency, other processes and simultaneous residents remain unsized |

Components use `weights`, `attention-cache`, `recurrent-state`,
`compute-workspace`, `prompt-cache` and `companions`, with the existing
`estimated`, `partial`, `unknown` and `excluded` states. An excluded absent
component has an explicit absence reason and no byte field; a mapped zero-cost state
has `estimated` and `"0"`. Known per-domain totals are partial component sums,
never a complete peak or Fits statement. Selected-model storage is separate
from simultaneous resident scenarios; no reclaimed memory is credited.

The adapter checks scalar/array/tensor dimensions and products before library
arithmetic, restricts facts to 16,384 tensors, 512 layers/array values and 8 MiB
retained accounting, context to 1,048,576 allocated cells and slots to 256.
The dense graph has additional batch/dimension/element bounds before its
synchronous library call. Checks honor the existing five-second operation
deadline before/after bounded work; no abandoned timeout goroutine exists.
Library panic, overflow and budget exhaustion produce specific Unknown reasons.
The zero-ready executor-owned estimator uses nonblocking cache admission; a
contended cache yields the advisory Unknown reason `estimator busy`.
Its derived LRU has 128-record, 1-MiB and 30-second limits, including charged
record/string/slice overhead and detached backing storage. Keys bind ordered
artifacts, effective allocation, revision, native build/support and the complete
bounded memory snapshot; unavailable telemetry has an explicit identity.
Neither the cache nor preview handoff alters Task 5 admission/cache/rechecks.

Inspection and explicit load preview consume the target broker's existing self
memory observation. Only these explicit operations may invoke the bounded native
map helper, subject to target permission, activation, ownership and Pause.
Apply, recovery and start never enumerate devices or search Auto. Absent input yields Unknown with the reason
`target memory telemetry unavailable`. Useful component sizing and its independent
uncertainty are preserved. Real estimator results remain partial even when a
synthetic caller supplies memory: complete native cost accounting is unresolved.

Reviewed Auto GPU loads share a full UUID, child-only `CUDA_VISIBLE_DEVICES`
and selector `CUDA0`. Finite settings and ordinary launch assignments persist
these bytes. NativeBuildID additionally binds the executable, relevant library
names/content hashes and non-credential effective child environment. Cold replay
requires library search to remain confined to the selected build root and
validates that source without reconstructing a private map or requiring a live
producer run. Source drift refuses activation. Public requests cannot supply the
worker envelope or replace a target-owned resolution. Manual/conversion GPU
settings remain launchable without reviewed bindings and keep Unknown comparisons.
The [advisory estimate support contract](#advisory-memory-estimate-support)
describes the unresolved accounting limits; MANIFEST covers option vocabulary,
not complete-cost eligibility. Production Fits and Auto remain unavailable.

The shared Go comparison owner produces `capacityFit` and `freeFit` as `unknown`,
`exceeds`, `tight` or `fits`. Each charged domain must equal the checked sum of
its sized components; every component category needs a charge or explicit
exclusion. Required bytes greater than the compared capacity or available bytes
are Exceeds. Otherwise, checked required bytes plus the pool's reserve greater
than that value are Tight; equality is Fits. The reserve is capacity divided by
ten, rounded up, plus 512 MiB. Overflow, partial costs, overlap or an unmatched
physical pool makes the comparison Unknown. Every charged pool must pass;
capacity on another GPU cannot compensate for a shortfall.

Worker-only inspection and preview envelopes keep memory facts outside public
`InspectRequest` and portable changes. They reuse node-info's independent fact
sources, sample times and carried ages, plus a broker dispatch boot-clock stamp.
The manager adds measured same-host transit before comparison and rechecks age
on cache hits and before returning. Invalid clocks or hop stamps withdraw
comparisons. Measurement timestamps are distinct from estimator timestamps;
domain replies include source/sample identities and bounded remaining validity.
Public strict decoders reject worker-only fields. Frozen finite validation is
separate from a new preview and does not require historical memory to recur.
Reviewed Auto GPU allocations carry one target-owned physical `DeviceBinding`
(host, domain, device, exact selector and binding digest); CPU carries none.
Public `ModelChange` cannot supply that binding, and no measured bytes,
timestamps or producer-run identity becomes native settings authority.
Manual and external-conversion finite GPU allocations remain readable and
launchable without a binding. Their unproven GPU-domain comparisons are Unknown.
External-conversion provenance cannot authorize an Auto allocation.
Invalid byte ranges, free greater than capacity and invalid domains are
rejected as advisory Unknowns.
No GPU utilization percentage is interpreted as capacity. Device identity
requires an explicit native-to-physical mapping; arrays never imply identity.
UMA accepts one shared pool and removes host/device aggregate double counting.
Snapshot free memory is timestamped and unreserved. No estimate creates memory
admission, ME2 input, observation, qualification or automatic retuning.

The compiler emits a separate optional read-only handoff only after its existing
artifact/build rechecks. Preview consumes it after its ordinary digest and base
reply-size check. Actual-start compilation also requests this handoff and freezes
detached per-model estimate inputs for buffer capture. Ordinary validation and
readback callers do not request it. A failing/panicking estimate
or oversized advisory attachment leaves manual settings and review/native
digests unchanged. Sampling recommendation provenance remains descriptor-owned.
Synthetic `TestModelSettingsEstimate` cases specify arithmetic, support/Unknown
boundaries, cache limits/identity, memory domains and digest independence. Source
fixtures do not establish numerical accuracy on native builds or real hardware.

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

## Model sources

Model sources are read-only folders other applications keep models in. The
built-in sources are Ollama's store (`OLLAMA_MODELS`, else `~/.ollama/models`,
plus `/usr/share/ollama/.ollama/models` on Linux) and LM Studio's model folders
(`~/.lmstudio/models`, `~/.cache/lm-studio/models`); there is no setting to add
more yet. Sources are engine-neutral: each found model carries its format, and
IDs are `ollama/<name>@<tag>` or `lmstudio/<path>`, never containing `:`. The
earliest folder keeps a duplicate ID.

At each start of a llama.cpp-family engine without `runtime.serving`, typed
model loads or a user preset (`--models-preset` in any spelling, or
`LLAMA_ARG_MODELS_PRESET`), engine-manager writes the GGUF source models as a
native preset under its install base (`model-sources/<engine>.ini`, written
atomically) and appends `--models-preset` with that path after launch
validation. Models with invalid identifiers or preset-unsafe paths are left
out; with none left the file is removed and nothing is appended. Sources are
rescanned only at start. Peer copy, model settings inspection, serving presets
and typed loads do not cover source models. See the README's Model sources
section for the folder layouts.

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
reject auto/zero/negative/overflow, except nonnegative `cache-ram` and
`sleep-idle-seconds = -1`, which
disables idle retention in the pinned build. Positive retention remains valid;
zero and values below -1 are refused. Retention stays string-valued (`"-1"`).
The single `llamaServingOptions` table derives native spellings/environment names
from pinned llama.cpp `0ec4027a` `common/arg.cpp`. Logical/physical batch sizes,
K/V cache types, Flash Attention, KV/operation offload, CPU MoE placement,
weight repacking and load mode are supported and retained in placement entries.
`cpu-moe` accepts only `true`, `1` or `on`; omit it to leave experts unpinned.
False is refused because native drops the positive-only child switch while
the child can inherit `LLAMA_ARG_CPU_MOE`. `n-cpu-moe` requires a canonical
nonnegative decimal int32, without a sign or leading zeros (`0` is allowed).
`repack` uses explicit booleans; negative preset keys (`nr`, `no-repack`)
invert native truthiness (`on`, `enabled`, `true`, `1`). `load-mode` accepts
exactly `auto`, `none`, `mmap`, `mlock`, `mmap+mlock` or `dio`. The b12020
adapter also accepts canonical nonnegative int32 `moe-cache-mib`,
`lazy-mode` (`auto`, `on`, `off`) and `spec-type` (`none`, `draft-mtp`).
Built-in MTP uses the main model's head, without a separate draft artifact.
Automatic projector selection accepts only disable forms: `--no-mmproj`,
`--no-mmproj-auto`, presence of `LLAMA_ARG_NO_MMPROJ_AUTO` (even empty), or
`mmproj-auto` set to `false`, `0` or `off` in environment, presets or typed
extra options. Negative preset keys `no-mmproj` and `no-mmproj-auto` invert
native truthiness (`on`, `enabled`, `true`, `1`). Wildcard preset disables are
preserved in every converted profile. Enabling automatic selection and explicit
`mmproj`/`mmproj-url` selections remain refused because projector allocation
is unsupported.
Each control is retained in placement readback. These controls
have no typed override and do not change memory estimation. Unsupported SWA,
context-shift and scaling controls, tensor overrides, CPU FFN, host buffers,
NUMA/RPC placement, draft/projector allocation, router
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

Artifact lookup for custom-alias copy export and inspection reads only
`models-preset`, `model`, `model-draft`, `ctx-size` and `parallel`. Other preset
options are ignored, and structural refusals apply to the affected section
(including defaults inherited from `[*]`). Serving compilation keeps the strict
parser. Both native option readers use the shared effective environment,
including Windows pseudo-variable exclusion and explicit override precedence.
An empty option name never denotes an optional alias. Selected artifacts still
require their existing checked bindings; native controls grant no copy
permission and add no separate draft artifact.

Preview performs no probe, spawn, load, persistence or snapshot creation. It
reads bounded source/configuration data and model samples only, returning a
native-source approval digest and candidate local bindings. The broker clears
the digest on a client preview, so unchanged settings can be re-approved after a
build changes compiled launch defaults. The broker compiles every plain request
from the target's own model loads, and the request may send back only those
loads or that compile's exact result, which is what the preview returned. Apply
requires the exact reviewed digest/bindings, prepares an immutable small snapshot before stopping a process
or saving settings, and rejects changed inputs. Apply and startup restore/start
still refuse saved typed loads or a saved digest that the inputs they are about
to launch did not produce, and the settings readback reports either as a
mismatch, whether or not the configuration has a declaration. Snapshots
reference absolute existing source model paths, preserving HF symlinks and
requiring all selected shards; model files are neither copied nor fully hashed.
Start re-reads source bytes, resolved targets, stat and bounded samples directly
before spawn, without a manager/state/content lock over file I/O. Recovery
regenerates identical bytes from matching approved inputs. Each owned process
retains its generation until exit. Same-stat middle-only artifact changes
outside the samples remain outside this cheap local binding's assurance.
Typed loads hold no full hash, so a new modified time is the only signal for a
change between the samples, and matching samples do not show the content is the
same. A saved binding whose file has a new time with the same target, size and
samples is refused as stale approval, and the readback reports it as a mismatch
that names every such file. A no-edit plain Apply accepts the new time, or Recalculate
on a frozen Auto load; Recalculate also accepts moved files of finite profiles,
which plain Apply cannot pass while the Auto load is refused. Only the settings
preview request accepts, and its diff names every file it accepts in a
`models.<profile>.fileModifiedTime` entry. Accepting is its own Apply: a plain
preview with another edit is refused, an apply must send back the loads the
preview returned, and a port change, finite model change or other internal
recompile refuses until it has happened. A file
with another target, size or samples is refused with no acceptance; restoring
the original file recovers.
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
owner reacquires the operation lock. The admission owner in
`mutation_admission.go` preserves `mutationAdmission` before `opMu` ordering.
Declaration rejection is active for every engine with an
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
engine carry it (without following redirects, with fixed failure diagnostics
instead of response bodies),
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

Checks before effects, after ordinary mutation admission and the lifecycle lock, and after
the dispatch relay refuse these conditions with `outcome:"not-dispatched"`:

- `projection-unknown`: no broker projection since startup.
- `paused`: either-origin pause.
  `beginRemote` registers accepted work for cancellation by a later veto.
- `engine-off`: no installation or running engine, regardless of retired saved intent.
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

Under ordinary mutation admission and the engine lifecycle lock, immediately before
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
refusal reason: `engine-off`, `no-fit`, `paused`,
`paused-local`, `engine-off-local` (older inbound reasons), `not-served`, `not-installed`, `unqualified`,
`operation-busy`, `unsupported-load`, `operation-timeout`, `observation-unknown`,
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
(currently llama.cpp). It owns an engine's entire
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
  llama.cpp declares list, loaded-list, pull, load, unload, and native cache
  delete with exact model ids. `loaded_models` reports
  models currently resident in memory, name-extracted via the same declarative
  `result` spec with optional nested-path and row filters: llama.cpp matches
  nested `status.value == loaded` records from `GET /models`.
- Semantic model operations: each manifest binds `load`, `unload`, `pull` and
  `delete` to its actions, with the model parameter name, fixed scalar
  parameters and an optional bounded response budget
  ([MANIFEST.md](MANIFEST.md#model-operations)). Local typed RPCs and the
  remote `ec` model endpoints resolve through the same bindings, so clients
  name only an engine and a model. llama.cpp uses the conventional
  actions with `model`.
- Engine identity and family: a manifest's `engine` is the stable ID and its
  optional `family` selects the coded model-store and catalogue adapters, so a
  locally defined fork with its own ID, port and store gets its family's
  behavior with no code change. `engine:descriptors` publishes the sanitized
  projection.
- The bundled llama.cpp launch fixes `--sleep-idle-seconds -1`; PAIR alone releases idle model memory.
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
- A normalized node-level model list (`engine:models`): union of every running engine's `list_models`, name-extracted via each action's declarative `result` spec, plus the per-engine set of models loaded in memory (`loadedByEngine`, from each engine's `loaded_models` action) and optional display metadata (`modelIdentities`, keyed by engine and ready model). A successful explicit empty inventory remains an engine key with `[]`; a missing/malformed/failed inventory omits that engine key instead of being mislabeled as authoritative empty. A watcher polls the loaded set and pushes `engine:models-changed` for inventory, residency, download lifecycle, memory evidence and attached identity changes.
- Expose all of the above over the `engine:*` JSON-RPC surface to whatever orchestrates the service, plus an optional plain-HTTP LAN endpoint (`--http-port`, `GET /v1/models`) that serves the model list to a peer's discovery daemon (the list moved off the size-limited mDNS TXT onto HTTP), together with this node's policy advertisement (`policy`, §7.0) as the broker last pushed it.

**Out of scope**
- **Inference traffic** — stays with `nvpair-proxy`; this service never proxies `/api/chat` etc.
- **Multi-instance per engine and an MCP server** — future-additive, not v1.
- **The node's error list** — owned by `nvpair-errors`, which holds it as in-memory session state; this service only emits `errors:report` / `errors:clear`.
- Automatic deletion of persistent llama.cpp model-cache files.

## 3. Key Use Cases
- **Install an engine, user-mode**: `engine:install {engine:"llamacpp"}` downloads the per-OS user-scoped package (Windows/Linux standalone archive extracted into a user dir; macOS app bundle — never an elevated `Setup.exe` or `curl | sh`), checksum-verifies, extracts, re-detects.
- **Run lifecycle internally**: the broker uses availability-driven `engine:wake` and `engine:sleep`; internal Executor operations retain readiness and health probes against the engine's loopback port. Clients query `engine:status` and have no start, stop or restart control.
- **Run a declared non-loading action**: `engine:action {engine, action, params}` accepts known inventory reads and manifest-bound unload, pull and delete operations. Load bindings, inference and unclassified custom actions are refused. Allocation enters admitted model load or proxy inference.
- **Onboard a new engine (no code)**: a vendor adds `engines/<vendor>.json`; the generic runner exposes their lifecycle + actions immediately.
- **Edge case — already installed**: detect short-circuits install (idempotent).
- **Edge case — checksum mismatch**: install fails before `run`, is reported, and never executes an unverified payload.
- **Edge case, readiness timeout or crash**: failed readiness returns to Stopped; an unexpected owned-process exit is reported and enters bounded installed-engine relaunch while the node is available.

## 5. Requirements

**Functional**
- Load + validate per-engine JSON manifests (bundled + user dir); select the host `<goos>/<goarch>` block; resolve placeholders (`{bin}`, `{port}`, `{download}`, `{install_dir}`). Install commands also receive resolved download and destination paths in child-scoped `NVPAIR_INSTALL_*` environment variables so shell reparsing cannot corrupt them.
- Run each engine as an owned foreground process; execute detect / install / uninstall / start / stop / restart / status / health and HTTP **or** CLI actions; emit `engine:*` results and notifications.
- Bound stops: on Unix send SIGTERM to the owned process group, then SIGKILL after `runtime.stop.grace_s` (five seconds by default); `signal:"kill"` skips the grace. On Windows, windowless managed engines require immediate `taskkill /T /F`.
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
- **User mode, no runtime admin/sudo**; stdio / `--ipc` is the primary control channel, with an optional plain-HTTP model-list listener and a clustered pin-gated mTLS control listener; engines bind loopback by default, but a manifest's `runtime.bind` can open an inference engine to the LAN (the cluster serves inference over it); HTTPS-only downloads, checksum-verified when the manifest pins a `sha256`.
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
  terminationVerified: boolean
  healthy: boolean
  port: number
}
```

`terminationVerified` is positive physical-stop evidence. `running:false` alone
does not prove termination, including when orphan reclamation failed. A completed
physical Pause requires verified shutdown rather than an inferred stopped state.

`LogLine` object (entry in `engine:logs`):
```json
{ time: string, stream: "stdout" | "stderr", text: string }
```

Errors are emitted with the `nvpair-shared/errors.ServiceError` wire shape (`id`, `message`, `severity`, `action`, …), mirrored locally in `reporter.go` with identical JSON tags. `id` follows `engine-manager:<class>:<engine>`; the Broker stamps `nodeId` / `timestamp`.

An invalid or unreadable engine override disables that engine instead of falling
back to its bundled manifest. The engine remains listed as unavailable, with a
false installed/running/healthy status and a validation diagnostic through the
existing `engine:errors` / `errors:report` path. Normal engine-manager admission
does not start or probe that engine.
Override filenames must match their declared engine; a mismatch disables both
identities, and conflicting files cannot restore an invalid entry. The separate
privileged `--uninstall-managed` path uses only bundled commands and location
overrides; it intentionally does not depend on validity of user override files.
If the override directory exists but cannot be read as a directory, all known
engines are unavailable rather than falling back. Runtime override reload is
not supported: corrected overrides take effect in a fresh manager, which clears
the stable validation error for each valid engine. Shutdown still cleans up
processes the executor already owns.

Example request:
```json
{"jsonrpc":"2.0","id":1,"method":"engine:install","params":{"engine":"llamacpp"}}
```

**Outputs** — JSON-RPC responses/notifications to the orchestrator; spawned engine OS processes; loopback HTTP to the engine's control API; `errors:report` / `errors:clear` notifications the Broker forwards to `nvpair-errors`. Lifecycle is not a hot path; error emits are best-effort (no-op if `nvpair-errors` is down). stdout writes are serialized so frames never interleave.

Example error notification:
```json
{"jsonrpc":"2.0","method":"errors:report","params":{"id":"engine-manager:install-failed:llamacpp","message":"sha256 mismatch","severity":"error"}}
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
| `engine:install` | `{ engine }` | `EngineStatus` after install and start when the node is available |
| `engine:uninstall` | `{ engine }` | `EngineStatus` (after removal) |
| `engine:uninstall-managed` | — | `{ engines: [{ engine, removed, error? }] }` — removes the engines this service installed, selected by their install marker, each through the `engine:uninstall` path. Backs both clients' "reset all data"; the platform uninstallers reach the same code through `--uninstall-managed` because they have no broker. Neither `removed` nor `error` means the engine was left alone as not ours. |
| `engine:action` | `{ engine, action, params }` | the engine's raw response |
| `engine:logs` | `{ engine }` | `{ lines: [LogLine] }` |
| `engine:errors` | — | `{ errors: [ServiceError] }` |
| `engine:catalog` | `{ engine, platform?, arch?, query? }` | `{ models: [CatalogModel], source, platform?, arch?, fetchedAt?, searchable?, query? }` — the models an engine can **download**. One curated source per engine family, selected by the manifest's `family`; only llama.cpp has one: the GGUF repos of `ggml-org`, `bartowski`, and `unsloth`, fetched live from Hugging Face and cached, offered as `repo:Q4_K_M`. It is `searchable`: a `query` searches all of Hugging Face for GGUF repos, each query cached on its own, and is echoed back; a query over 100 characters is an error rather than being shortened. `platform` and `arch` are the GOOS and GOARCH the models will be installed on; omitting both means this host, and naming a platform without an arch leaves the arch unknown rather than borrowing this host's. The llama.cpp list is not machine-locked, so nothing is filtered for the target and `appleOnly` is never set. An engine with no curated source is an error, not an empty list. The reply is a single JSON-RPC frame — every hop on its path must allow `jsonrpc.WorkerFrameBytes`. |
| `engine:remote-get-installed` | `{ node }` | `{ engines: [EngineStatus] }` from the remote node |
| `engine:remote-install` | `{ node, engine }` | `{ opId, status }` after the remote install |
| `engine:remote-pull-model` | `{ node, engine, model?, params? }` | `{ opId, result }` after the remote pull |
| `engine:remote-load-model` | `{ node, engine, model }` | the remote action result |
| `engine:remote-unload-model` | `{ node, engine, model }` | the remote action result |
| `engine:remote-delete-model` | `{ node, engine, model }` | the remote action result |
| `engine:remote-copy-model` | `{ node, engine, model }` | `{ opId, result: { engine, model, files, filesSkipped, bytesTotal, bytesCopied } }` after copying the model from `node` into this node's engine store (§7.3) |
| `engine:copy-model` | `{ opId, sourceNodeId, destinationNodeId, engine, model, expectedExportId? }` | `{ opId, result: copyModelResult }` after verified destination publication (§7.3) |
| `engine:cancel-model-copy` | `{ opId }` | `{ cancelled: boolean }`; local channel only |
| `engine:active-model-copies` | `{}` | `{ operations: [{ opId, sourceNodeId, destinationNodeId, engine, model, expectedExportId?, progress? }] }` |
| `engine:model-copy-capabilities` | `{ nodeId, engine }` | `{ version:1, engine, available, installed, modelStore, exportLayouts, importLayouts, writable, reason, identityBoundRetry, receiptDurability }` |
| `engine:wake` | `{ engine }` | Broker-internal `EngineStatus`: starts an installed engine when available; refused during shutdown |
| `engine:sleep` | `{ engine }` | Broker-internal `EngineStatus`: uses internal `doStop` |
| `engine:load-model` | `{ engine, model }` | the engine's raw response, via the manifest's `load` operation binding |
| `engine:unload-model` | `{ engine, model }` | the engine's raw response, via the manifest's `unload` operation binding |
| `engine:delete-model` | `{ engine, model }` | the engine's raw response, via the manifest's `delete` operation binding |
| `engine:remote-policy-get` | `{ nodeId }` | `{ policy, availability, pausedBy?, pauseMode? }` — the target broker's peer projection of its policy: no routing section, and tiers keep only their local entries |
| `engine:remote-policy-set` | `{ nodeId, policy, expectedPersistedRevision }` | `{ policy }` — the peer projection after the target broker merges the change into its stored policy; the revision is forwarded unchanged and compared by the broker |
| `engine:remote-availability-set` | `{ nodeId, state, pauseMode? }` | the target broker's `{ availability, pausedBy?, pauseMode? }` result; configured waits for drain, immediate is refused |
| `shutdown` | — | `null` |
| `log/set-level` | `{ level }` | `{ level }` |

The availability relay carries request-scoped `pauseMode` unchanged, including
explicit null for target validation. Omission selects `configured`; explicit
values are `configured` or `immediate`, valid only with `state:"paused"`.
After fresh target permission (`-32012`) and pause-fence (`-32011`) checks,
an authorized peer's immediate request is refused with `-32602`,
`Immediate pause must be requested on this node.` The authenticated caller,
not payload `nodeId`, determines locality. An accepted target-local immediate
request fences execution and requests cancellation before the persistence barrier,
supersedes the previous transition, and persists Pause. It starts controlled
shutdown without the configured drain or cancel-settle wait; missing physical
shutdown proof leaves the node draining and fenced. The broker persists the
accepted `pauseMode` with Pause and clears it on Resume. Stored pauses without
a mode are repaired to `immediate`, preserving local-only Resume protection.
The target's policy readback and availability result preserve actual
`pausedBy` (`local`/`peer`, empty/omitted when unknown, cleared while available)
and saved `pauseMode` (`configured`/`immediate`, omitted while available).
Peer responses omit activation and origin-private routing fields.



Notifications (service → caller): `policy:request{id, method, caller, params}` / `policy:cancel{id}` (a paired node's node-policy call for the broker, which answers with the `policy:reply{id, result, error?, errorCode?}` notification), `ready{version}`, `engine:state-changed{EngineStatus}`, `engine:models-changed{engine, models}` (pushed for ready inventory, residency, download lifecycle, memory evidence and attached identity changes; `models` carries the `engine:models` inventory shape including `loadedByEngine` and `modelIdentities`, without `policy`), `engine:install-progress{engine, stage, percent?, error?}` (`error` on the terminal `failed` frame), `engine:pull-progress{engine, op, stage, percent?, message}` (live progress for a local model pull driven via `engine:action{action:"pull_model"}` - the local counterpart of `engine:remote-progress`), `engine:remote-progress{opId, node, engine, op, stage, percent?, message}` (live progress relayed from a remote install/pull; a model copy reports `op:"copy"` with `file?, bytesDone?, bytesTotal?`, section 7.3), and `errors:report` / `errors:clear` (consumed by `nvpair-errors` via the Broker). `install` / `uninstall` / `action` / `remote-*` and internal wake/sleep each run in their own goroutine so the read loop never blocks; their responses arrive when the op completes.

Notifications (caller → service): `engine:node-projection{policy, pausedLocally, managementBlocked}` carries this node's policy advertisement (`nvpair-shared/nodepolicy.Advertisement`: `revision`, `roles` per tier with the bounds of the matching profile, `pausedBy?`), local-pause provenance, and `managementBlocked`: whether a pause of either origin, accepted but not yet saved, draining or saved, fences peer management writes. The common mutation gate requires a known projection and enforces this bit after the current grant check. Local activation also refuses known blocked state; trusted local startup may proceed before the first projection. An older broker omits the bit, which reads as false; it is never derived from `pausedBy` or `pausedLocally`. The broker sends it after every persisted policy or availability change and again on every `engine:ready`, reading its current policy each time under one lock, so a later projection never carries older state. engine-manager keeps the last one in memory, replaced whole, never persisted; an undecodable one leaves the previous in place. `engine:models` and `GET /v1/models` return it as `policy`, omitted until the first push. `engine:models-changed` does not carry it.

`engine:model-identity{engines}` is also a service → caller notification: a whole snapshot keyed by engine, each entry `{engineRunId, launchDigest, modelDigests}` with `modelDigests` keyed by model name. The loaded-model watcher emits it after its first sweep and whenever the snapshot changes; an absent engine has no current identity. Only running owned processes with a known launch digest are included, excluding adopted services. Model digests must come from an inventory of the same run and contain lowercase SHA-256 hex. Publication is ordered with `engine:state-changed`, so a running engine's identity cannot arrive after its stop. The broker adds its settings fingerprint before publishing the strict admission identity to the proxy.

Before changing installed model content, the executor sends the broker the internal notification `engine:model-mutation-request{requestId, engine}` and waits for `engine:model-mutation-reply{requestId, engine, ok, error, code}`. Only a matching accepted reply permits the change. The broker conservatively withdraws the whole engine, retaining its applied settings fingerprint, and confirms destination withdrawal through the same proxy barrier as settings changes. These messages are never client RPCs or client events.

The guard covers manifest-bound semantic `delete` and `pull` actions in `Action` (including custom action names), streaming `PullModelStream`, and destination `CopyModelFrom`, including `runCopyPlan` and finalization. Typed delete and direct `ec` delete/pull inherit the guard through these owners. Copy source reads, load/unload residency operations, and opaque actions without a model-operation binding do not acquire it. A per-engine content semaphore serializes the guard and mutation separately from lifecycle `opMu`, so restart-after-action does not nest that lock around confirmation. Waiting for the semaphore observes cancellation and deadlines without beginning suppression or advancing the inventory epoch. Joining earlier publications also observes cancellation and deadlines; any abandoned join releases the publication mutex when it eventually acquires it.

The relay has a 15-second confirmation budget, beginning before its synchronous parent send. Refusal, timeout, cancellation, no parent, and send errors fail closed before mutation effects. An accepted reply authorizes mutation only while the context remains live, including when cancellation and a queued reply are both ready. The deadline bounds the reply wait after the send returns; a blocked send cannot authorize a mutation but can outlast the budget. Pending entries are removed on every reached terminal path.

Before requesting withdrawal, the inventory owner suppresses the engine, advances its epoch, and joins earlier identity publications. Inventory captures that epoch before I/O; suppressed or stale-epoch observations cannot restore identity. Every guarded exit, including refusal, cancellation and partial failure, advances the epoch again, clears cached digests, releases suppression and requests a fresh sweep. Only a fresh post-completion observation of the owned run can restore identity. Local epoch metadata participates in publication deduplication, so unchanged final content is republished too; it is not sent on the wire. Withdrawal does not cancel inference already dispatched.

`percent` on all three progress notifications follows one rule: it is present only when the step is measurable (download bytes, byte-progress pulls) or terminal (`100` on install `done` / `already-installed`, `-1` on install `failed` and pull `error`). An indeterminate step omits it — the install `verified` and `installing` stages always do, because the install command is opaque — and a consumer renders the stage alone rather than a percentage.

Example `engine:install-progress` (stdout):
```json
{"jsonrpc":"2.0","method":"engine:install-progress","params":{"engine":"llamacpp","stage":"downloading","percent":42}}
{"jsonrpc":"2.0","method":"engine:install-progress","params":{"engine":"llamacpp","stage":"installing"}}
```

Example `engine:pull-progress` (stdout):
```json
{"jsonrpc":"2.0","method":"engine:pull-progress","params":{"engine":"llamacpp","op":"pull","stage":"pulling","percent":62,"message":"pulling"}}
```

### 7.1 Versioning
- `engine:*` method names are namespaced and additive — new methods, never breaking changes to existing ones.
- `manifest_version` gates manifest-schema evolution: unknown optional fields are ignored (backward-compatible growth); a version higher than supported is rejected.

### 7.2 Remote engine management (the `ec` surface)

**Management authorization.** `controlServer.routes` is the single route/method
classification registry. `controlServer.authorize` runs after pinned mTLS
identity verification and requires a fresh
`clustertrust.Mesh.RemoteManagement(caller)` allow for every mutation, including
install, unload, delete, pull, policy/role/served/resident edits and pause/resume.
Read-only POST policy get and settings get/preview, GET inventories, model
exports and events remain reads. Unclassified routes fail closed;
denial precedes mutation handlers, streaming headers and side effects. The
broker rechecks the same predicate at its owning locked acceptance points for
policy, availability and settings; relayed caller identity comes only
from the current authenticated worker notification.

HTTP 409 `{"error": "bounded explanation", "code":"remote_management_denied"}`
maps to RPC `-32012`, independently of `-32010/policy_conflict`,
`-32011/local_veto` and unpinned 403. Each typed code
keeps its number through the 409 body, stream error frames and broker relays,
with the target's message and no JSON-RPC error `data`. Missing, invalid, unsupported, changing or
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

With `--control-port` and a clustered `--cluster-dir`, engine-manager serves a cluster-scoped remote-control surface over pin-based mTLS (`nvpair-shared/clustertrust`): it presents this node's cluster leaf, requires a client cert, and `403`s any caller that isn't a byte-for-byte pinned cluster peer. The listener is bound whenever `--control-port` is set and admits callers by live membership: its leaf is resolved per handshake, so an unclustered node presents none and every handshake is refused. No restart is needed on `cluster:identity-changed`; the broker registers `ec` whenever a cluster dir is configured. Routes under `/v1`: `GET /engines`, streaming `POST /engines/install` and `/models/pull` (chunked NDJSON: zero or more `{"type":"progress"}` frames then one terminal `{"type":"result"}`/`{"type":"error"}` frame), non-streaming `POST /models/{load,unload,delete}`, and the model-copy source routes `GET /models/files?engine=&model=` (NDJSON: `hashing` progress frames, then a result frame carrying `{engine, model, revision?, files:[{path, size, sha256}]}`) and `GET /models/file?engine=&model=&path=` (one listed file, Range supported). There are no engine start, stop, restart, wake or sleep routes. The node-policy routes `POST /node-policy/get`, `POST /node-policy/set` and `POST /node-availability/set` (bodies `{}`, `{policy, expectedPersistedRevision}`, `{state}`; at most 256 KiB; unknown fields refused; the revision is the compare-and-set precondition of a policy set, forwarded unchanged for the broker to enforce) are pin-gated with the caller's identity, clear any `nodeId` so no further hop is possible, and relay to the local broker as `policy:request` carrying the authenticated `caller`; the broker owns and re-authorizes the call. `engine:remote-policy-get` / `-set` and `engine:remote-availability-set` are their client half, addressed by `nodeId`.

The model-file listing result includes the issued `exportId`; file GET or HEAD
also supplies it as a query parameter as specified in section 7.3.

The broker's `policy:reply {id, result, error?, errorCode?}` carries a JSON-RPC
`errorCode` on every failure. Invalid parameters, including missing or malformed
`expectedPersistedRevision`, invalid policy and invalid availability, use
`-32602`; revision conflicts use `-32010`; a resume the local veto refuses uses
`-32011`; remote-management denial uses `-32012`;
target-owned profile inference settings use `-32014`; save failures and other operational errors use `-32000`. Inbound
policy routes return HTTP 400 for `-32602`, HTTP 409 with the JSON body
`{"error": <diagnostic>, "code": "policy_conflict" | "local_veto" | "remote_management_denied" | "remote_setting_locked"}`
for `-32010`, `-32011`, `-32012` and `-32014`, and a plain-text HTTP 409 for other broker or relay errors. At the
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

After a successful fresh grant check, `controlServer.authorize` uses the common
route registry classification: every incoming mutation returns HTTP 503 until
projection is known, then HTTP 409 with `code:local_veto` while
`managementBlocked` is true. The exact message is `This node is paused. Resume it on this node before managing it remotely.` Grant denial retains `-32012`
precedence. Install,
load, unload, delete, pull, settings apply and policy set are fenced.
Availability set alone delegates the pause decision to the broker, after pin,
fresh grant, projection hydration. The broker allows peers
to control configured pauses of either origin; Halt Resume, pending/draining
Halt and build apply remain refused. Repeated peer Pause of completed local
Halt succeeds without effects. Unknown or legacy pause mode remains local-only
until local Resume. Reads return before this check,
including GET/HEAD, read-only POST preview/status/get, event streams and model
files. No second production route registry is introduced.

Either-origin pause projection cancels accepted starts, loads, crash relaunch
and pending activation. Local start/load also refuse known pause; local
reads, configuration, internal stop and unload remain available. Accepted
file-only transfers can continue, but new destination copy admission still
refuses known Pause, and peer mutations remain fenced. Sleep protects external
ownership from automatic unload or unscoped stop.
Unverified ownership or termination prevents a completed physical Pause; the
broker retains its draining fence until proof or local retry. Settings
apply/recovery can change metadata but cannot restart while paused.
Accepted content installation/download may finish, without starting an engine.
The HTTP body, stream terminal error and JSON-RPC relay retain
the numeric code and bounded target message. Pause and local Stop retain `-32011`.

Common start/load guards preserve either-origin pause.
Source ownership validation remains authoritative.

The `engine:remote-*` methods are the client half: engine-manager resolves the target `node` in an `ec` peer directory (fed by its own `discovery:subscribe{services:[ec]}` to the broker relay), dials the peer's `ec` surface with the same pinned identity, relays each streamed progress frame up as `engine:remote-progress` (keyed by a minted `opId`), and settles the request on the terminal frame. Remote install/pull run for the operation's full duration (no broker-imposed timeout); model unload/delete are request/response, and a remote model load whose binding declares a long response budget uses the readiness-sized response budget. They error if this node isn't clustered or the target isn't a pinned peer.

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
  Manager callback rechecks the real caller's fresh grant and pause gate
  under its reservation before effects. Previously accepted work is not
  undone by later grant revocation. The source independently pin-gates the
  destination as the actual reader. Preserve `-32012/remote_management_denied`,
  `-32011/local_veto`, and `-32010/policy_conflict` end to end. There is no
  install/start, activation gate or controller staging fallback.
- **Capabilities and retry.** Pinned read `GET /v1/models/copy-capabilities`
  accepts only `?engine=...` and returns the local/remote RPC capability shape
  above. `modelStore` and native layout names are adapter IDs (`llamacpp`),
  plus `llama-artifacts-v1` for custom llama.cpp.
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

- **Source listing.** `GET /v1/models/files` binds the exact selected model's
  ordered files as logical slash-separated names with size and sha256. Native
  sources require engine recognition: llama.cpp HF ref/snapshot and selected
  GGUF shards/projector. Supported file and directory links can resolve outside the store;
  no link grants access to arbitrary descendants or a resolved parent. Hashes
  are computed lazily and cached. HF snapshot files use the checked full hashing/cache owner rather than local
  LFS blob filenames. The destination verifies the complete bytes.
- **Source serving.** Each list issues an `exportId` for its frozen selection.
  GET and HEAD, including Range, require that ID and exact engine, model and
  listed logical name (`400` for malformed paths, `404` for unlisted names).
  Rechecks retain the old selection and use a checked opened handle through
  `http.ServeContent`. Missing, stale, evicted or restarted issued state
  requires relisting; unchanged content can receive the same ID. Native IDs
  are transfer tokens, not durable native receipts or native result fields.
- **Custom llama.cpp identity.** Before interpreting an ID as an HF locator,
  resolve exact configured preset, persisted resolved typed-load and committed
  copy-receipt candidates. Typed bindings can select artifacts without
  `Runtime.Serving` or a running engine. Disagreeing candidates are ambiguous.
  Export checks the selected effective configuration, artifact bindings and
  applicable full hash assertions, followed by a snapshot recheck. It does not
  recompile or approve the whole launch. Full launch and apply retain their
  separate review checks. Undeclared
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
  file hashes, and asserted full serving hashes must match. GET or HEAD adds
  `exportId` and checks the source's issued binding snapshot, current selection,
  symlink-resolved identity and the opened regular file. A source restart or
  bounded issuance-cache eviction requires relisting; unchanged listings retain
  their binding. Neither host paths nor launch/serving configuration is exported.
- **Link and evidence limits.** Ordinary file hard links share changes with
  their other names; directory hard links are unsupported. Filesystem permissions
  apply. Identity checks and samples do not prove detection of every same-stat
  middle mutation or prevent a later native engine open from racing. Existing
  peer settings administration, caller grants, removal and strict destination
  containment are unchanged. Selected-copy validation does not establish whole
  launch approval or publisher authenticity.
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
- **Visibility.** The file that makes the model visible is written last: llama.cpp's first GGUF split and then `refs/main` when the repo was not cached before, followed by `GET /models?reload=1` on a running router.
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

Stores: llama.cpp uses `LLAMA_CACHE` (the manifest sets `{models_dir}`) in the Hugging Face hub cache layout, selecting `owner/repo:TAG` files as llama.cpp does. The layout is selected by the manifest's `family`, so a fork with its own id and `models_dir` uses its family's layout rooted at its own store; an engine whose family has no layout is refused. One copy of a given model runs at a time per node.

#### Publisher comparison observations

`copyModelResult` optionally adds `publisherComparison:{status,files,replayed?}`.
Each `files` entry has exact fields:

| Field | Meaning |
| --- | --- |
| `path` | Exact copied logical relative path |
| `status` | `verified`, `mismatch`, or `unavailable` |
| `provider`, `repository`, `revision`, `file` | Optional complete selector: `huggingface`, exact owner/repository, full lowercase 40-hex immutable commit, exact repository-relative file |
| `algorithm` | `sha256` |
| `expectedDigest` | Full lowercase provider SHA256, present only for an acquired statement |
| `actualDigest` | Full lowercase SHA256 of bytes checked by the existing transfer/cache owner |
| `acquisitionBasis` | `provider`, `cache`, or `unavailable` |
| `acquiredAt` | Original successful UTC RFC3339 acquisition time; absent for unavailable |

Verified requires exact independently acquired statement agreement. It conveys
no model safety, author trust or signature assertion. Mismatch remains visible,
never verified, and introduces no publication veto. Existing full declared
artifact assertions and full transfer hash failures remain hard failures.
Missing comparison fields never imply verified. The aggregate is mismatch if
any file mismatches, verified only if every copied file verifies, otherwise
unavailable. Include all copied files, with individual mixed states preserved;
reject partial coverage, duplicate paths and inconsistent aggregate claims.

Only the native llama.cpp HF snapshot resolver initially supplies a mapping.
Source listings may include a per-file `publisher` record with the same entry
shape, initially unavailable with a checked full digest. It is solely a selector
at the destination. The destination replaces every peer observation, regardless
of claimed status/basis/time, using its own existing catalogue HTTP/cache owner.
The fixed HF `/api/models/OWNER/REPO/revision/COMMIT?blobs=true` response must
identify the exact repository and immutable revision, explicitly report public
and ungated, and name the exact file with a full LFS SHA256. Ignore Git SHA1
blob identity as a SHA256 statement. Do not follow provider redirects. Refuse
invalid UTF-8, malformed, duplicate, truncated or
oversize metadata as unavailable without affecting authorized transfer.
Never accept a peer URL, mutable revision, alias/name similarity or local LFS
filename as observation authority. Other providers/layouts, custom aliases,
private/gated repositories, non-LFS files and unproven/missing mappings are
unavailable, with transfer allowed under the existing rules.

The catalogue owner bounds revision metadata by its existing search capacity,
TTL, retry backoff, deadline and response-body cap. Cached statements retain the
original acquisition time; expired data is unavailable on refresh failure.
Optional observation records require exact unique keys, valid Unicode, bounded
safe paths, full lowercase digests, supported statuses, coherent equality and
bounded valid UTC timestamps. Result metadata uses the existing 8 MiB envelope;
source manifests/receipts keep their existing 1 MiB cap and progress its 64 KiB
cap. Tests inject synthetic response transports into the existing HTTP owner.

Receipt replay is destination-local evidence: first rehash every committed
file, then preserve recorded statement fields and acquisition basis/time with
`replayed:true`. No provider call, invented freshness or upgraded peer assertion
is allowed. Missing receipt observations stay unavailable. Native copies do not
gain durable receipts. Strip observational fields from native ExportID hashing;
the custom canonical identity already excludes them. No new request fields,
URL, settings, grant/trust flag, RPC version/capability, serving/launch/apply/Auto
behavior or permissions arise from these observations. Pinned source reads,
destination grants, Pause and recovery/admission retain their existing owners.

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
  as `models_dir` outside that data dir (`~/.llamacpp`) so neither an engine uninstall nor the app-level data
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

## 11. Assumptions
- The orchestrator owns the stdio/IPC pipe and is the only peer; on parent exit (stdin EOF) the service shuts down cleanly — no reconnect or buffering.
- Manifests are either well-formed (validated) or rejected; authors keep `engine:*` and manifest shapes stable per the versioning rules.
- Engines bind loopback unless the manifest's `runtime.bind` opts otherwise (inference engines default to `0.0.0.0`; a per-call `bind` overrides); the engine's own startup banner is not trusted — the readiness probe confirms the actual bind.
- `nvpair-errors` is best-effort; its absence degrades to local logs only.

## 13. Observability
- **Logging**: structured `slog` to stderr via `nvpair-shared/applog` (captured by the parent's debug panel); per-engine stdout/stderr ring buffers; key events — install fetch/verify/run outcomes, start/ready/stop transitions, health flips, action invocations, every failure.
- **Metrics**: lightweight in-process counters (installs, starts, failures, health transitions per engine), surfaced via `engine:status` and `engine:errors`; no separate metrics endpoint in v1.
- **Alerts**: surfaced through the errors pipeline — `errors:report` for install/start/health failures and manager crashes; the UI's error list is the alerting surface for v1.

## 14. Sample usage
The operator opens the UI; the Broker calls `engine:get-installed`, which returns each manifest's `EngineStatus` (e.g. llama.cpp `installed:true, running:false`). To set up a missing engine, the Broker sends `engine:install {engine:"llamacpp"}`: the service downloads the per-OS user-scoped package, checksum-verifies it, runs it user-mode, and re-detects — progress streaming as `engine:install-progress` notifications.

The operator pulls a model with `engine:action {engine:"llamacpp", action:"pull_model", params:{model:"owner/repo:Q4_K_M"}}`, which issues the manifest-declared `POST 127.0.0.1:{port}/models` through the streaming pull owner. Local `engine:pull-progress` and remote `engine:remote-progress` frames carry the target model, download ID and lifecycle snapshot. Unchanged lifecycle/percentage updates are coalesced. Streaming pulls use a 30-minute inactivity watchdog refreshed only when a layer/file completed-byte count advances; duplicate progress and heartbeats do not keep a stalled pull alive. An HTTP `pull_model` action must declare a `progress_protocol` (only `llamacpp-models-sse` exists), and the manifest is refused at load without one; the adapter requires the matching `download_finished` event before emitting `stage:"success"`, so a closed stream or 100 percent alone is unknown. Failed or unknown pulls emit `stage:"error", percent:-1` with the fixed reason in their lifecycle snapshot, in addition to the JSON-RPC error. A CLI-driven pull has indeterminate progress and retains the fixed 30-minute action timeout. Its successful command exit must not contain an explicit error result. On `shutdown` (or stdin EOF) the service stops every running engine first, so none are orphaned.

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

### Intentional memory diagnostics

Peer diagnostics use private parent IPC
`engine:remote-diagnostic-preview|start|status|cancel` and POST
`/v1/model-diagnostic/preview|start|status|cancel` on the existing classified ec
listener. Preview, start and cancel require the verified caller's management
grant; status is a pinned read. `remoteClient` and `NewPeerServiceClient` own
direct/reverse delivery. No additional listener, polling worker or transport is
introduced. The peer cannot forward through this target: the incoming node
selector is stripped before the local broker call.

`diagnosticRelay` emits `engine:diagnostic-request{id,method,caller,params}` and
accepts only a correlated parent `engine:diagnostic-reply{id,result,error?,errorCode?}`.
The broker binds that request and its consent to the current manager source and
exact worker pair, rechecking grant, maintenance and Halt at acceptance.
Configured pause is permitted. `engine:diagnostic-request-cancel{id}` retires a
waiter, not accepted work. The relay is bounded and cancellation-safe; late
replies cannot recreate authority. Cancel uses the readiness HTTP client and a
205-second wait outside the target's 200-second cleanup join. Public responses
are strictly validated and projected, retaining exact run and target identity.
An empty status is JSON null; it is handled at this domain boundary without
broadening shared strict decoding.

The manager owns the diagnostic execution reservation and accepted operation.

The broker acknowledges sampled barriers through
`engine:diagnostic-baseline-done`, bound to the run and token: `phase: baseline`
after three fresh cold samples, and `phase: after-load` after two fresh samples
from every pool following `loaded`. The latter is accepted only while loading
with a confirmed loaded marker. Each wait is bounded to thirty seconds and
joins operation cancellation before any later native exercise.
`internal:diagnostic-observe` accepts `{request,memory,hardware}` on the private
parent connection. `request` is the diagnostic preview request, `memory` is the
target broker's `TargetMemory`, and `hardware` is
`{cpuModel,totalRamBytes,gpus:[{uuid,name,totalMemoryBytes}]}` from target node-info
facts. Every usable GPU, including an unused GPU, must have a matching physical
alias and capacity in the memory observation. A shared pool is system RAM and
GPU memory together. Missing or ambiguous hardware fails observation closed.
This read performs no wake, native enumeration, policy write or model load.
Its effects fence uses the same owner as compare-and-reserve. Runtime build
claims, context, slots and placement remain absent until the authorized load.
The observation carries separately named planned context and slots from the
saved finite profile, without treating them as realization. Diagnostic consent
refuses explicit requested or resolved slot counts above eight before effects.

`engine:diagnostic-reserve` carries the same private `hardware` object alongside
`targetMemory`. Acceptance checkpoints the run/profile/key and requested identity
before returning usable authority. The operation saves its engine ledger before
unloading, its Started obligation before wake, and its LoadedModels obligation
before load. Wake and cleanup sleep synchronously dispatch through `Manager.runOp`
with reservation-scoped contexts. Configured pause is preserved; Halt vetoes
activation. Only post-load fenced props supply realized context, slots and native
build claims. Managed native `/props` has no `devices` or `n_gpu_layers` fields.
Realized placement requires the existing buffer parser's complete model,
KV/recurrent and completed compute-reservation reports from the exact loaded
child generation. Capture validates the current process/run, source, artifacts,
build and load boundary. Missing, partial, stale or other-load reports refuse
identity before exercise, including on a host-only or shared-memory machine.
CPU/CPU-mapped/CUDA-host allocations map to the host pool; Metal allocations map
to the sole shared pool. CUDA labels join through the existing selected-runtime
device helper's PCI/full UUID (and Windows LUID) identity in the captured child
environment, independently of pool or native enumeration order. A diagnostic
may map several GPUs without creating a reviewed-Auto binding. Native ordinals
and requested GPU options never prove physical placement. The helper runs only
with the accepted diagnostic's activation authority. Its per-load identity
evidence retires with that child generation; ordinary inspection never starts
native enumeration to fill a gap.
The live reservation retains accepted pool kinds for configuration-key checks
through broker drain and the cold baseline. This is captured identity only;
expired capacity/free-memory facts are never refreshed or reused as fit evidence.

`engine:record-measured-memory` accepts `{runId,token,profile}` only for a retained,
joined producing run. It checks the reservation key, complete captured identity,
outcome, native exercise evidence and any linked buffer sample before persistence.
The broker may downgrade exercise coverage using its sample overlap evidence.
It cannot promote native partial coverage or replace native timings and token counts.
ExecutionKey results can be semantically current and never overlay estimates or
feed Auto. Unpriced loads retain only bounded private parser/device realization
evidence, without a finite/public/persisted buffer sample. GPU-selection
inspection also excludes buffer history from another configuration.

The private `engine:inspect-model` worker envelope also accepts `hardware`.
Inspection revalidates files, requested settings, native build and hardware before
using authenticated stored realization, and uses fenced live realization when the
profile is loaded. Matching finite and ExecutionKey records retain applicable
realization after an unchanged manager/store restart. Native fit may retain its
authenticated realized context, which can differ from its pre-load plan; fixed
contexts and slots still match the saved resolution. Changed requested inputs,
artifacts, build, filtered native launch environment, CPU/RAM/GPU identity, context or physical placement withdraw
currentness. A present but uncertain live load cannot fall back to retained
realization. The same identity reaches record currentness and the estimate
overlay. Missing hardware keeps measured history visible with currentness withdrawn.
The ordinary shared worker envelope and its strict decoder are unchanged.

`InspectResult.measurementIdentity` optionally carries only `{digest,complete}`.
Complete measurement applicability binds the configuration owner's semantic
sampled-artifact, process working-set limit and effective launch/load inputs to this same diagnostic
hardware and realization proof. It excludes run/epoch/revision, file paths and
mtimes, routing endpoints and available-memory samples. Artifact assurance
remains sampled local binding, not full model-content verification. Finite
`ConfigurationKey`, unpriced `ExecutionKey`, measured-overlay and Auto semantics
are unchanged. Missing source/build/hardware/realization returns
`{digest:"",complete:false}`. In particular, cold native fit cannot predict its
next realization from retained history. After live readback, inspection rechecks
the canonical artifact/configuration owner and the original memory lifetime.
An envelope with `nativeInitAllowed:false` performs no native help discovery or
device initialization. Measurement metadata never starts, loads or unloads.

`PendingDiagnostic` belongs to the existing desired-state store. Startup restores
its admission hold before manager readiness and recovers through the lifecycle
owner before installed-engine restore. Recovery never resumes measurement.
Status exposes the nested run, token, identity, markers, optional observation with
`poolPlacement`, and low/high exercise evidence. Pending checkpoint read errors are
errors, not an empty status. A terminal checkpoint alone does not grant
`workStopped`: durable cleanup confirmation and durable checkpoint clearing must
finish before the reservation is released. Uncertain writes retain the hold;
recovered abort retries cleanup with fresh authority. Saved pause is never changed.

### Ordinary mutation admission and verified stop

`mutation_admission.go` is the canonical ordinary mutation-admission owner.
Admission covers the wait for the engine lifecycle lock and the operation.
The executor retains native `managerIncarnation` and `activeMutations`;
admission entry and retirement advance owner-evidence fences.

The existing desired-state decode drops retired transaction fields in place
without writing on load. The strict raw owner-evidence decoder can refuse such
fields until a normal save replaces the document; no automatic repair is
promised.

### Download lifecycle and content admission

The canonical model inventory includes optional version-1 `downloadState`:
`{version,sourceId,revision,recoveryRequired,entries}`. Source is the executor's
native manager incarnation; revision is a canonical uint64 decimal string and
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

### Offline copy baseline

The operator contract and ordered interval procedure are in
[Offline copy baseline](README.md#offline-copy-baseline). Root coordinates as
the profile owner. The standalone commands are:

```text
nvpair-engine-manager --initialize-fresh-profile
nvpair-engine-manager --amend-copy-baseline --expected-journal-sha256=HEX --copy-baseline-disposition=complete-empty-history --attest-profile-quiescent
```

`HEX` is the lowercase SHA-256 of the exact reviewed current bytes. Amendment
also requires `--previous-profile-reviewed` whenever the previous app exists,
even without a journal. `--allow-empty-old-journal` applies only to a current
version-1 object missing only `copyRecoveryRequired`, with empty arrays and a
valid `copiesComplete` boolean. Current input instead requires strict empty
arrays, `copiesComplete:false` and `copyRecoveryRequired:false`. The current
version-1 journal requires both booleans; missing journals remain historically
incomplete, old ordinary loads unreadable, and no ordinary constructor
provisions or repairs them. The old amendment is one-off, not compatibility.

The current app is `BASE/Nvidia Corporation/Personal AI Router`, the previous
app `BASE/NVIDIA Corporation/PAIR`; organization parents can alias, so the
app-level paths matter. On Windows both modes require a set, nonempty, absolute
`LOCALAPPDATA` naming the profile owner's Local base before mutation. Fresh
allocation refuses either existing app, imports no history and is deliberately
non-idempotent. First launch may already create the profile; installation,
empty inventories, deleted journals and recreated children do not prove it
fresh. Failure preserves allocated and landed state. Unix success confirms file
publication/readback and each newly created directory entry through the first
pre-existing ancestor. Windows has no supported equivalent here: success is
bounded to allocation and file publication/readback and reports
`fresh-directory durability: UNVERIFIED (unsupported)`, not whole-profile
durability. Keep this limitation in completion reports.

Previous inspection accepts a definitely absent app or journal, a validated
empty current journal, or the precisely empty old shape without the current-input
old opt-in. Markers, unreadability, unsafe paths and inspection uncertainty
refuse even with review attestation. Current retained downloads or copies,
naming mismatch, recovery true, unreadable or unsupported old data remain
blocked for separate review; replacement documents are unsupported. Actual
machine coverage remains UNVERIFIED. Empty current state does not license
ignoring retained previous markers after partial migration.

Root reserves the interval, prevents restart and other users, closes GUI/TUI
and stops service, broker and standalone owners by separately authorized
procedures, then verifies controllers, destinations and downloads stopped. Root
reviews exact bytes and historical evidence across both namespaces, records the
digest and disposition, invokes the command once, checks backup bytes, hashes
and strict state, and resolves failures before release. Empty live lists or
markers do not establish history. The tool does not stop owners. Service locks,
sockets, optional ports and process snapshots cannot prove exclusion; snapshots
race startup and may miss renamed processes. Attestation is mandatory and a
nonempty `NVPAIR_SERVICE_ENDPOINT` is refused. Rechecks are not atomic CAS.

Amendment reserves a deterministic digest-named backup, refuses an existing
backup and confirms exact bytes before publication. It rechecks hashes and
reports observed landed state; refusal or uncertainty exits non-zero. There is
no automatic cleanup, retry, restore, resume or persistent recovery protocol.
Complete-looking bytes after a crash or uncertain write cannot reveal a prior
durability error on ordinary read; preserve evidence and resolve it before
restart. Root enforces exclusion after tool exit. After a separately authorized
interval is released, ordinary startup may reread the confirmed journal and
read-only evidence must show expected baseline/source and remaining blockers.
Neither command establishes full owner evidence or build-apply eligibility; O1
still refuses real starts, O2 external-root coverage remains separate, and O3
does not certify predecessors or infer historical completion.

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

### Installed engine lifecycle and model admission

Engine installation and node availability determine activation. There is no saved
engine On/Off state or intent RPC. The internal `engine:restore-installed`
notification restores installed engines at startup and Resume. `engine:wake`
repairs a stopped installed engine, subject to pause and Halt. Installation starts
an engine only while available; install requests have no `start` option. Crash
relaunch follows the same availability guards. Settings restarts only an engine
that settings stopped (`resume`), or the running engine it changed. It also starts
an engine whose last restore, at startup, on resume or in a crash relaunch, refused
a stale approval, when the approval the apply replaces still reads as `mismatch`,
unless the node is paused or the engine was since stopped. That record is in memory
only. A failed start, before or after spawn, keeps it, but that apply has already
saved its approval, so a later apply starts the engine only if that approval has
gone stale too; a request or resume starts it otherwise. A refused wake does not
count: the next request retries it.

`engine-state.json` retains allocated ports and process-stop evidence. Its old
`engines` and `offBy` fields are ignored on decode and omitted on the next write.

Typed llama.cpp presets force `sleep-idle-seconds = -1`; PAIR owns per-model idle
release. The typed idle field is removed and extra options cannot override it.
Conversion drops native sleep values and includes the changed value in its diff.
Typed router launches force `--models-max 0`; PAIR admission owns every load's
capacity. External presets keep their own positive native capacity until converted.
Resident checks accept native capacity zero as unlimited.

Peer `POST /v1/models/load` retains the target authorization and availability gate,
then emits private `engine:admitted-load-request{id,params:{engine,model}}` to the
broker. The broker admits and executes through its existing reservation owner and
answers `engine:admitted-load-reply{id,result?,error?:{code,message}}`. Caller
cancellation removes the manager waiter; broker execution has its own bound.
