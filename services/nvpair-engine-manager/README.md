<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvpair-engine-manager

## Native build identity

`nativeBuildID` identifies llama.cpp builds as `sha256:<hex>` from the platform,
the full executable hash and the sorted names and full hashes of accepted
runtime libraries in the executable's directory. Managed and override builds
use the same calculation. Paths, modification times and manifest pins do not
enter the fingerprint. File hashes are memoized in memory with object checks;
there is no persisted hash store.

Manual and reviewed Auto loads store this content fingerprint in `nativeBuildId`.
Finite configuration keys bind it through a native source digest that also
includes explicit launch environment values and effective `CUDA_*`, `GGML_*`
and `LLAMA_*` controls. Validated build-root paths are normalized in that digest;
unrelated ambient variables are excluded. Touching or moving an unchanged build
preserves its key, including installs described by `install.fetch`. Expected
file evidence still rejects replacement during preparation and replay. Auto
retains its environment-bound source separately in the reviewed input envelope.
Native initialization remains subject to activation and Pause gates.

The inherited SWA allocation default has its own known-source assurance for
the managed b12020 artifact declaration. Recognizing another build's device
report does not establish its SWA default. Fetch installs and other builds can
receive configuration keys while that advisory default remains unknown.

## Operation-scoped memory device map

`mapModelSettingsDevices` is an internal, independently callable review adapter.
It accepts an executable-directory content-identified library snapshot, effective selection and
environment, explicit native-initialization permission, and supplied target
host/run/topology identity. It owns no telemetry cache and makes no capacity or
Fits decision. CPU placement uses only the supplied host/shared pool.

CUDA matching requires canonical PCI plus a full physical GPU UUID, and on
Windows the same driver's LUID matching the supplied DXGI identity. Review runs
the private metadata helper again with UUID-only `CUDA_VISIBLE_DEVICES` and
requires the same physical device at `CUDA0`. The returned binding retains that
UUID, pool, constrained environment and stable digest. Observation identity
includes producer run; stable binding identity excludes run and observation.
The ggml description must match the CUDA driver name, and total memory must
agree within 1 MiB. These cross-checks do not replace physical identity.
List position never proves a match. Unsupported exports,
ambiguous aliases, MIG/MPS environments, shared/discrete mixtures, Metal,
Vulkan, HIP and split-device selection remain Unknown.

The existing executable's private `--private-model-devices` mode exits before
registry/transport startup. It reads a bounded internal runtime snapshot from
stdin, verifies selected library hashes and reports `ggml_commit()` as metadata.
It checks the property layout against CUDA driver name and total memory for
the same PCI identity before accepting the report. A failed check returns
`device report not recognized for this build` without a partial inventory.
Recognition results alone are cached in memory by content build ID; topology,
visibility and bindings are checked anew for each operation. Several dedicated
GPU pools run no helper and report `several GPUs: reviewed Auto supports one GPU; choose GPUs with extra options`.
It loads no model and starts no engine or network listener. The
parent enforces a three-second ceiling, combined stdout/stderr ceiling of
64 KiB, device bound, credential-environment filtering, cancellation and reaping.
Unix uses purego; Windows uses x/sys restricted DLL loading and purego calls.
The helper also arms its own hard process-exit deadline before reading input,
so a stuck native call remains bounded after manager exit. Unix environment key
spelling is preserved. Linux forces the MPS pipe directory to `/dev/null`, a
verified character device that cannot contain MPS control pipes; this constraint
is retained in the replay environment instead of trusting absent MPS variables.
Native initialization remains subject to the target's Pause/activation owner.

Production joins belong to Task 5: obtain target-owned topology and freshness,
resolve the managed runtime with `modelDeviceRuntimeSnapshot`, supply effective
launch selection/environment and current native permission at explicit review,
then persist/compile the returned UUID constraint and `--device CUDA0` through
the existing settings/serving owner. Apply/recovery/start must replay the frozen
binding without invoking this adapter. These joins are not enabled here.
Task 1's memory domain wire alone does not supply the full PCI/UUID/LUID topology;
missing identity must stay Unknown at that join.

Source fixtures under `TestModelSettingsDevices` inject enumeration results or
use a TestMain sentinel before normal startup. They never load real libraries.
These fixtures do not establish native acceptance, which remains UNTESTED.

## Build-apply owner evidence (parent only)

`engine:owner-evidence` accepts exactly `{}` on the parent request channel.
Null, selectors, caller/operation fields, mutation flags, extra fields and
duplicate or case-variant fields are invalid params. There is no ec/control
route or notification. The broker consumes the private
`shared/buildapply.EngineOwnerEvidence` contract; clients must not receive this
document or its roots and operation provenance. Captured-worker resident
reconciliation also reads its native `managerIncarnation`; unrelated evidence
sections may remain incomplete. This read grants no mutation or reservation.

The exact response sections are `managerIncarnation`, `captureToken`,
`universe`, `stopAuthority`, `reservations`, `complete`, `reasons`.
Incarnation is the Executor's native `managerIncarnation`, including at idle.
Revisions describe canonical saved semantics, not durability or
mutation history. Universe and authority include sorted registered,
unsupported, instantiated, cleanup, durable and reservation engine names.
Runtime facts remain unknown until their existing owner observed them. A
retained owned process lifetime carries its run ID and, when retained, PID;
external/adopted processes receive no owned stop authority.

The read composes detached existing owner captures without engine initialization,
detection, process lookup, native contact, model inventory, journal recovery,
notifications or writes. Contention and observed changes withdraw evidence.
Download source mismatch withdraws reservations while preserving native manager
identity. Reservation revision binds copy/download/content authority and
admission generations; retained historical origin is explicitly unknown when
the owner has no origin. Historical copy baselines and uncertain outcomes stay
incomplete and block eligibility.

Known empty arrays differ from unknown observations. `modelRoots` and
`dataRoots` contain represented owner paths, with managed install roots in
`dataRoots`; `rootsComplete` qualifies exhaustive coverage. The current manifest
has no exhaustive data-root declaration, so coverage remains unknown for those
engines. No arbitrary command or environment path is interpreted as a root.
Request and response use the existing build-apply byte limits; overflow returns
fixed unavailable failure, never a truncated list. Reasons are bounded codes.

`owner_evidence.go` owns `ownerDigest`: SHA256 of `json.Marshal` bytes,
encoded as `sha256:` followed by lowercase hexadecimal. `captureToken` binds
all detached sections, including knowledge state, authority revisions and
incarnation. `complete` is derived knowledge, including known false authority
and known blocking work, not permission to apply. Matching reads
are not an atomic reservation. This RPC supplies no production `WithLocked`
adapter; build-apply starts remain refused until separately owned atomic
acceptance and the other transaction proofs exist.

## Serving credential

PAIR accepts, stores and relays no user-entered engine API key. The only key
the manager attaches is the one it generates for a process it launched
(Strata and other serving launches). Keys contain at most 512 printable
non-space ASCII bytes.

Candidate selection works before start and adoption, using the operation's
effective port, including a one-shot `StatusAtPort` port. The owned prepared
key is selected only for the process that serves that port. HTTP actions and
broker-only `internal:local-engine-credential` retrieval additionally require
a running engine on its current port; adopted and stopped engines export no
key. CLI actions receive no key in arguments or environment;
Strata's existing private launch contract is unchanged.

Authenticated probes and HTTP actions share endpoint validation: initial URL
must be explicit `http://127.0.0.1:<effective-port>` without userinfo, fragment
or opaque form. The private client clones a supported HTTP transport, disables
proxies and refuses every redirect. Failures carry fixed diagnostics or HTTP
status, never an engine response body or transport diagnostic. Successful
keyed action JSON field names/string values and non-JSON text redact the active
key. HTTP status and JSON identity still establish readiness; process presence
is only diagnosis. A declared auth response after attaching the key says the
key was rejected, while no key keeps the no-key diagnosis. With no key
attached, existing port handling, HTTP transport and errors remain unchanged;
credential endpoint eligibility is not applied.

Trust follows `nvpair-service`'s private same-user socket/pipe, broker stdio,
then manager stdio. Retrieval has no ec route, remote method, settings/policy
relay or notification. The broker refuses client retrieval before its generic
engine relay; its own existing retrieval intentionally supplies the local
proxy. No key enters public status, inventory, diagnostics or peer payloads.
Same-user processes remain inside this trust boundary.

`engine:inspect-model` accepts only the parent broker's strict worker envelope,
with an exact local engine/model in `request`. It calls `Executor.InspectModel`
and returns bounded target-local artifacts/facts and service-owned descriptors.
Artifact inspection uses the canonical resolver and GGUF metadata reader,
without engine start, HTTP inventory, download, full hashing or settings writes.
The surrounding explicit inspection/Recalculate operation can use the existing
permission-gated device-map helper; Pause, activation, ownership and maintenance
vetoes still apply. Advisory estimates use the same bounded
facts: tensor weights can be sized without an effective allocation; unresolved
cache, workspace, placement and memory telemetry stay explicit Unknowns.
The optional `measurementIdentity:{digest,complete}` is a separate applicability
projection over applied sampled artifacts, semantic launch/load options, content
build, physical hardware and authenticated realized context/slots/placement.
Incomplete inputs return an empty digest with `complete:false`. Run IDs,
settings revisions, paths, mtimes, routing endpoints and free memory are excluded.
Cold native fit stays unknown even when diagnostic history remains current.
Loaded profiles use the diagnostic owner's live readback, with no fallback to
history after failed readback. The worker envelope's `nativeInitAllowed:false`
also disables native help discovery for background inspection. This metadata
starts no engine, loads no model and grants no admission or capacity.
`engine:remote-inspect-model` uses the existing pinned POST
`/v1/engine-settings/inspect` read route; the target broker supplies node identity.
The remote reply is strictly decoded and checked against the requested identity.
See the [option matrix](MANIFEST.md#typed-model-load-contract-and-option-matrix)
for supported B vocabulary, inherited defaults, explicit unsupported rows and
generation/template/reasoning ownership. A descriptor is not apply readiness.

The correlated shared `settings.Relay` union carries `Inspect: *InspectRequest`
for `method: inspect`, with the value `Request` empty and omitted by MarshalJSON.
Settings methods retain their value `Request` and no `Inspect`. Broker consumers
must use `settings.DecodeRelay` before selecting the branch. The executor's
existing `settingsParent(ctx, method, settings.Request, caller)` signature is
unchanged: the pinned inspect adapter carries only engine and model ID in its
private `Resolution` slot, then converts that into the actual `Inspect` pointer
before emitting `engine:settings-request`. The model ID is not serialized as a
settings resolution, and no mutation payload travels with the read. Caller
identity comes from the pinned peer transport; target forwarding is cleared.

`engine:preview-launch` accepts load-only model changes; `engine:configure-launch`
accepts parent-owned frozen settings. Unknown,
duplicate, incorrectly cased and oversized fields are invalid params (-32602).
The parent-only launch decoder still permits the broker-owned PreserveCORS
guard; public settings decoders refuse it. Existing engine counters remain exact
numeric JSON within the shared safe-integer bound; new policy revisions and byte
quantities are decimal strings. Public callers cannot supply resolved authority.

Typed llama presets compile through `compilePreparedServing` from the full
Runtime, including typed-only configurations with no serving declaration.
Typed loads are keyed by `profileKey`: the profile key, preset section name
and served id are the same string. `modelId` selects the artifact files;
profiles may share an artifact model with different launch settings. Launch
settings live only in the engine manager. Node policy holds profile metadata
and the broker derives identity and context in the coupled policy stage.
Stored loads missing a key take their exact `modelId` during the existing
decode/repair pass; the next settings save persists `profileKey`.
Fresh/manual starts and crash relaunches consume immutable generated presets.
Canonical sorted model entries, checked context arithmetic, ordered sampled
artifact bindings, native executable binding and generated bytes form the input
identity. Existing serving declarations are validated separately against those
bytes; ordinary typed loads create no memory declarations or qualifications.

Direct `PreviewLaunch` and launch readback preserve typed records. The worker's
`engine:preview-launch` accepts load-only `modelChange` edits against a fresh
target baseline. It returns selected edits or a complete-router conversion,
per-field/source diffs and a digest, refusing unsupported members without takeover.
Public preview validation admits owner-injected Auto to that compiler; it does
not invoke the no-facts resolver or read a capacity snapshot itself. With no
owner, the exact `target memory telemetry unavailable` refusal remains.
`persistRuntimeServing` accepts an optional typed collection replacement in the
same atomic port/launch/serving override write, preserving platform fields and
opaque numeric precision. These helpers do not activate a stopped engine.

The direct `ConfigureLaunch` path detects typed changes and uses the existing
stop, persist, rebind and conditional restart sequence after revalidating typed
inputs. An accepted complete-router conversion preserves the current external
artifact set and unrelated launch settings, validates the reviewed finite
resolution, and rechecks the source before stopping and restarting. `SetPort`
uses the same configuration owner for typed settings. Worker RPC accepts frozen
reviewed settings; selected-patch acceptance and coupled policy remain broker-owned.
Previews disclose whole-router scope, affected configured models, the broker's
60-second drain bound and possible interruption after timeout. Loaded-model
observations remain an explicit unknown for the broker to resolve. Off, Pause and
adopted-process ownership keep their existing behavior.
Build maintenance uses the broker's existing settings-entry exclusion.
Target-owned memory and native-map
inputs travel only on the broker's strict worker envelope. Production Auto still
refuses incomplete cost accounting. Native
configuration does not create qualification or establish strict readiness.

Advisory memory estimates use the private MIT `gguf-parser-go` v0.27.0 pin
(`9257743a3c1775b882ba1b58ec7f5a856c2cc385`) through in-memory structs only.
Selected typed previews carry the compiler's already-decoded facts to a separate
estimator after constructing the ordinary review digest. Estimator failure,
overflow, panic, contention or budget exhaustion does not change the preview,
launch identity, requested/resolved settings or Apply behavior. No estimate
starts an engine, parses another file, writes policy, fills admission memory or
creates qualification. Embedded sampling recommendations keep their original
GGUF provenance and remain separate from settings overrides.

The [estimate support matrix](spec.md#advisory-memory-estimate-support) describes
dense/GQA, LFM2 recurrence, Qwen variants, SWA, mixed cache precision and tensor
placement. Unsupported components stay independently Unknown. Companions have
separate known storage and unsized runtime costs. Totals remain partial and
uncalibrated rather than complete peak-allocation bounds. CPU and unmapped offload byte projections are
not native device allocation readback. Inspection and explicit Recalculate use
the target broker's existing self telemetry cache, including source retirement
and suspend-aware aging. Missing input leaves comparisons Unknown with reason
`target memory telemetry unavailable`. With fresh mapped capacity and priced
weights, partial costs can produce advisory Fits or Tight with named omission
warnings; these verdicts do not establish a complete peak-memory bound.
The private owner-injected Auto selector consumes target-owned self evidence
through the existing worker envelope. Its complete-cost cases remain synthetic;
connected telemetry does not establish complete accounting.

The following matrix records the current estimator implementation. Derived rows describe
bounded tensor or native layout arithmetic. Compute rows are partial heuristic
projections from the pinned parser, with an uncalibrated outward band of
three quarters to three halves of the priced portion. They are neither native
allocation bounds nor complete costs. No calibration evidence has been added.

| Architecture class | Weights | Attention cache | Recurrent state | Compute workspace |
| --- | --- | --- | --- | --- |
| Dense llama/Qwen2/Qwen3 | Derived directory bytes | Derived mapped widths | Excluded | Heuristic CPU, full or partial single GPU, with complete conventional graph tensors |
| Qwen2/Qwen3 MoE | Derived, all resident experts | Derived mapped widths | Excluded | Heuristic CPU/full/partial single GPU; validated routed and shared expert widths, active expert count |
| DeepSeek2 MLA | Derived directory bytes | Derived only for the confirmed latent/rotary native layout | Excluded | Heuristic CPU/full/partial single GPU for the confirmed low-rank query and KV tensor layout, routed/shared experts |
| LFM2 hybrid | Derived directory bytes | Derived attention layers | Derived mapped short-convolution state | Heuristic CPU/full/partial single GPU with explicit per-layer KV heads and short-convolution tensors |
| LFM2MoE, Qwen3Next, Qwen3.5 variants | Derived directory bytes | Derived mapped attention layers; explicit Qwen recurrent masks remain Unknown | Derived for mapped short-convolution or R/S state layouts; otherwise Unknown | Unknown: graph workspace mapping deferred |
| SWA/shared-KV/NextN variants | Derived directory bytes | Derived only for the supported native layouts | Independently mapped | Unknown: workspace layer/window mapping deferred |
| Other or incomplete layouts | Known tensor bytes, partial if incomplete | Unknown unless independently mapped | Unknown unless independently mapped | Unknown with the missing graph input named |

Compute uses only each library device's `Computation` fields, with compute and
output sharing a maximum as in `Computation.Sum()`. Host portions stay on host;
the single device portion stays on the device domain. At zero GPU layers the
parser's hypothetical device scratch is excluded, never transferred to host.
CPU requires explicit `op-offload=false`. GPU pricing requires one native
CUDA or Metal inventory entry; unresolved inventory and multiple-device splits
stay Unknown. Device byte projections do not establish a launch binding.
Library panics become Unknown. Runtime/driver/allocator costs, native scratch
residuals, mmap residency and transient peaks remain unpriced warnings.

The estimator's historical source basis at llama.cpp `7fe450e19305b828c199d602c23a8337aaa1f03b` is
`src/models/qwen2moe.cpp` and `qwen3moe.cpp` (expert tensor widths and native
`n_ff/n_expert_used` fallback), `deepseek2.cpp` (query/KV low-rank and shared
expert shapes), `lfm2.cpp` (attention versus convolution layers), and
`src/llama-model.cpp` (output-first offload placement). Parser v0.27.0 owners
are `file_architecture.go:transformerArchitecture` (retained metadata),
`file_estimate__llamacpp.go:EstimateLLaMACppRun` (placement and computation),
its flash/recurrent compute helpers (active-expert intermediate widths), and
`LLaMACppComputationMemoryUsage.Sum`. Scalar dimensions and complete tensor
shapes are required; unsupported per-layer expert widths, leading dense MoE
blocks and query-projection variants remain deferred rather than inferred.
Useful component estimates remain available. The
[estimate support contract](spec.md#advisory-memory-estimate-support) records
current accounting limits; synthetic costs do not enable production. MANIFEST
documents native option vocabulary, not complete-cost eligibility.
Simultaneous residents are unsized
and no release credit is assumed. Auto's no-facts helper remains Unknown.

Go owns advisory `capacityFit` and `freeFit` comparisons per physical pool.
Complete synthetic cost fixtures exercise Fits, Tight and Exceeds, with checked
component sums and a reserve of 10 percent of capacity plus 512 MiB per pool.
Missing, expired, incomplete or unmapped evidence stays Unknown. UMA is one
shared pool; another device's free memory cannot cover a shortfall. Replies
carry measurement provenance and bounded remaining validity separately from
the estimate timestamp. Cached arithmetic never renews a measurement.
Private worker envelopes reuse timestamped node-info facts and account for
the same-host dispatch hop; public requests cannot upload this authority.
Reviewed Auto GPU results retain an exact physical device binding through cloning,
finite recompilation and readback; CPU results have none. This plumbing does
not complete the unresolved native cost accounting or enable production Auto.
Manual and external-conversion GPU settings remain usable without a binding;
their unproven GPU-domain comparisons stay Unknown.

Auto searches only the existing validated bounded facts, at most 128 candidate
evaluations: highest context first, then greatest finite GPU offload. Its default
range is 8192 down to 4096 tokens with one slot; explicit ranges, fixed context,
slots, precision and other manual choices remain constraints. Context candidates
use the 256-token grid and known model context, without automatic RoPE extension.
Dense weight transfer alone does not prove total cost monotonicity. Bisection
also requires the cost owner's complete layer-independent non-weight host cost
for the exact context/options, checked against every evaluated allocation.
No production estimator supplies this proof. Without it, or on a disagreement,
the selector enumerates from greatest offload through CPU within the same budget.
Unknown costs, stale telemetry, unknown device maps
and incomplete allocations cannot fit. Exhausted searches report an unsearched
range separately from insufficient memory. The disclosed heuristic reserves
10 percent of each pool's capacity plus 512 MiB. CPU-only results disclose their
speed cost. Current estimator graph costs remain partial, including CPU scratch,
so capacity alone cannot create a production Fits result.

Reviewed Auto uses `auto/finite-llama/1`, binding requested preferences, finite
options, artifact/build identities, memory snapshot hash and any external-router
conversion source. Apply and cold restart/recovery compilation reuse that exact
finite result without Auto search. Unaccepted Auto additionally checks fresh
resource eligibility; accepted replay needs no advisory telemetry. Artifact/build drift
refuses; native fit remains off, with no OOM retry ladder. Explicitly reselecting
Auto in a new preview is Recalculate. It can review a replacement native build
for the selected model; old Apply/restart and frozen siblings still reject drift.
A manual edit cannot silently retune saved
Auto. Conversion labels alone never authorize Auto. Manual preview and Task 10
advisory digest behavior are preserved. The broker now retains a bounded exact
review and forwards frozen full settings at Apply without another portable
patch, Auto resolver/search or helper call. Only an unaccepted Auto allocation needs fresh
complete resource eligibility; manual Apply and accepted recovery do not require
advisory telemetry. Production Auto and Fits remain unavailable until the
complete-cost specification and implementation are supplied and accepted.

Each executor owns a derived LRU with ceilings of 128 records, 1 MiB retained
accounting and 30 seconds. Its key includes ordered sampled artifact identities,
effective allocation, estimator revision, native build/support and memory
snapshot identity (including an explicit unavailable identity). This cache is
separate from artifact inspection's cache and never replaces source rechecks.

Owned llama.cpp loads also support optional `InspectResult.loadObservation`.
Actual-start compilation freezes the estimate inputs for that load; bounded
output capture and the existing loaded-model sweep correlate rounded reports
with child/load, artifact, source and effective-option identity. No serving
declaration or qualification is required. Inspect reads a detached selected
summary from retained history, adding no model-list/props request, load,
inference or initialization beyond the surrounding operation's existing behavior.

Capture attempts every build and recognizes allocation output by its grammar.
An unrecognized output is retained as `partial` with `unrecognized-build-output`
and no rows. Samples carry `buildId`, `reportedBuild` and, when output is
recognized, `buildAssurance: recognized-output`, without claiming clean-source
provenance. Two-decimal MiB reports and conservative byte intervals
are allocation reports, not exact bytes, physical occupancy or a complete-process
peak. Incomplete role/device/component coverage remains Unknown. The comparison
uses the frozen captured-load baseline, never a newly edited draft, and cannot
update estimates, Auto, admission, qualification or margins.

Derived samples use `model-buffer-samples.json`, separate from settings and
operation journals. Corruption disables history writes; landed `ErrReplaced`
publication is adopted with durability uncertainty. Recovered history never
proves a current process. Other-build records remain readable and never disable
writes. Inspection prefers matching-build history, otherwise labels the newest
retained sample `earlier build`. The existing decode folds old known-pin records
into a reported build with no content identity; those never match the current build.
The optional Inspect summary is bounded to 4 KiB within
the existing complete reply limit; projection reduces rows, returns structured
Unknown, or omits that field while preserving artifact facts. The desktop maps
absence to Unknown/`observation-omitted`. See
[native buffer observations](../../docs/model-buffer-observations.mdx) for
formats, privacy, retention and coverage limits. Windows process acceptance is
pending; native accuracy, occupancy and calibration remain unverified.

A config-driven control plane for local inference engines. The bundled
manifest supports llama.cpp. It manages everything about
an engine **except serving inference**: detect, user-mode install,
internal lifecycle, health, and config-declared actions. Adding an engine
is a JSON manifest, not code.

The bundled manifests under `manifests/` are the working reference for manifest
authoring.

An unexpected owned-process exit with installation retries through restore
after 5 seconds, 30 seconds and 2 minutes. Readiness resets that failure episode;
three unsuccessful attempts leave the existing unexpected-exit error standing.
Explicit stop, pause/sleep and shutdown cancel recovery; pending switches retain
their recovery owner and failed orphan reclaim still fences adoption.

Reclaim also checks a saved positive PID for the same endpoint, revalidating
both its recorded executable and its current OS image against the managed
install. A matching child is stopped and its exit confirmed even before it
binds, so restore cannot race it with a cold child. Missing PID records and the
spawn-to-PID-save crash window retain listener-based reclaim. Linux image
matching resolves available symlinks and compares case-sensitively; macOS
retains canonical paths and filesystem-verified case aliases.

### llama.cpp backend checkpoint

The `llamacpp` manifest runs `llama-server` in router mode on loopback port
`8081`. It can list, download, load, unload, and delete exact model ids such as
`owner/repository:Q4_K_M`. Downloads use `/models/sse` for progress; deletion
uses the router's native `DELETE /models` cache operation and does not restart
the router. `LLAMA_CACHE` points at the manifest's `models_dir`, `~/.llamacpp`,
which no removal this service performs will delete.
Readiness requires `/props` to report `role:"router"`, so model-selection
arguments that switch `llama-server` to single-model mode and incompatible
listeners already occupying the port are rejected rather than adopted.
After five minutes without inference work, a loaded model enters llama.cpp sleep
mode and releases its model and KV-cache memory; the next request wakes it. The
router child remains alive and can retain a residual backend GPU context.

Windows and Linux installs download checksum-pinned server and CUDA-runtime
archive pairs (CUDA 12.x for x64 and CUDA 13.4 for arm64); macOS uses the
standard Metal-capable archive. These on-demand downloads are roughly
0.6–0.8 GiB and do not enlarge the PAIR installer. llama.cpp keeps its `auto`
GPU-layer policy: supported NVIDIA/Metal devices can accelerate, while the
dynamic CPU backend remains the fallback. Hardware acceptance, not `/health`
alone, is required to claim GPU activation.

## Communication

Bidirectional newline-delimited JSON-RPC 2.0 — the same conventions as
every other NVPAIR subprocess. Stdio by default; `--ipc <path>` dials a Unix
domain socket or Windows named pipe instead. The service is
**parent-agnostic**: it speaks to whatever owns its pipe (in practice
`nvpair-ui-broker`) and carries no front-end dependency, so the
same binary runs under any supervisor with zero code change. The `engine:*`
namespace is the surface the orchestrator/broker forwards from the UI.

Parent notifications have a 60-second write budget, including waiting for the
shared write gate and writing the complete frame. Policy, settings,
model-mutation and resident relays use their caller's earlier cancellation or
deadline for the initial send and retain their existing reply budgets. A queued
caller cancellation emits no frame and leaves another active writer alone.
Policy/settings reply cancellation removes the pending entry first,
then synchronously sends its cancellation frame with a fresh one-second budget;
cleanup failure is logged without replacing the original caller error.

State and identity publication retain their ordering lock, and content/lifecycle
serialization remains in place. Transport closure releases stalled writes so
these locks and relay slots can be released. Unix stdio owns nonblocking,
close-on-exec duplicates prepared before protocol I/O and requires successful
deadline capability probes; unsupported descriptors fail startup. Windows uses
the actual inherited pipe file owners. Closing output first interrupts the active
writer, then closing input wakes the reader. Stderr stays independent. Root
cancellation (including signals) closes the transport and joins its owner watcher.
IPC sockets/named pipes use the same terminal-close policy. No queue, retry,
delivery acknowledgement or runtime budget setting is introduced.

## JSON-RPC surface

### Diagnostic execution reservation (internal)

`engine:diagnostic-reserve` accepts `{runId, engine, modelId, profileKey,
configurationKey?, executionKey?, effectsFence, targetMemory?}` and returns
`{runId, token, state: "reserved"}`. Exactly one identity key is required. The
applied llama.cpp profile is inspected through the existing file-only owner,
with target memory evidence for finite configuration keys. `effectsFence` is
the manager's `diagnosticEffectsFence`: it binds the manager incarnation, each
engine's applied configuration and process incarnation, and fresh loaded-model
sets. Missing loaded-model evidence refuses reservation.

Reservation compares the preview identity and effects under node-wide mutation
exclusion. An admitted mutation or registered serving load, including one
waiting for a lifecycle lock, refuses it. Repeating the same run ID, target,
identity and effects returns the same token; different parameters conflict.
Ordinary lifecycle operations, resident reconciliation and residency-affecting
manifest actions refuse with `reserved by a memory measurement (busy)`.

`engine:diagnostic-release` accepts `{runId, token}`. It refuses while any
diagnostic mutation is in flight or its work is not confirmed stopped by the
manager operation owner. It never trusts a broker-supplied cleanup flag.
Reservations live in the executor and are lost on manager restart; durable
recovery and the diagnostic operation loop are separate owners.

The private `diagnosticContext` carries authority for one executor, reservation
and engine. It bypasses only that reservation's global admission ticket and
preserves other admission checks. `WakeDiagnostic` and `sleepContext` reuse the
existing start and stop owners without writing pause policy. Diagnostic
activation is restricted to the measured engine; configured pause permits
scoped activation and Halt refuses it. The lifecycle and final dispatch owners
must propagate this context and apply the scoped pause rule before executing a
diagnostic under configured pause.

The broker denies public client relay for `engine:diagnostic-reserve`,
`engine:diagnostic-release`, `engine:diagnostic-run`,
`engine:diagnostic-run-status`, `engine:diagnostic-abort`,
`engine:diagnostic-baseline-done` and `engine:record-measured-memory`.
Only reserve and release are implemented by the reservation handler. The
distinct public `engine:diagnostic-status` is reserved for broker orchestration
and is not in this denial list.

### Automatic resident loads (internal)

The manifest must explicitly bind the semantic load operation to a `POST`
action with model parameter `model`, body schema exactly `{model:string}`,
and no fixed parameters. Loaded inventory must use a safe `GET` action with
a result extraction spec and loaded-status filter. Neither action may use
parameter placement overrides, restart, commands, remove paths, model
resolution, progress protocols or content mutation. The load sends exactly
`{model}` using the normalized installed model. `keepAliveSeconds` retains its
1..86400 validation but is not sent to the engine; approved native settings
own retention. No prompt, download, delete, copy, engine start, restore or
inference is performed. This path takes ordinary mutation admission and the
lifecycle lock, and registers with the shared load guard so settings/stop
cancellation applies.

The current NE serving envelope is validated before the final grant and again
immediately before loading. Requested context must match the approved finite
per-slot allocation, native idle retention must be disabled (`-1`), and router
capacity must cover `declaredResidentCount`. That count describes this engine's
desired residents at `policyRevision`, not all serving declarations. Missing or
invalid counts fail closed before effects; automatic resident loads remain
refused until the SR 5c broker producer supplies the field. Prepare/Dispatch
DTOs and keep-alive validation are unchanged.

Before and after admission, unknown projection, either-origin pause,
engine Off, missing installed model, stale/withdrawn identity or unsupported
load refuse execution. Immediately before loading the manager emits
`engine:dispatch-resident-request` with `{id, params}`; the broker relays the
`node/dispatch-resident` params to the proxy and returns an internal
`engine:dispatch-resident-reply` notification with `{id, result}` or
`{id, error}`. Only a correlated live `dispatched:true` grants permission.
See [spec.md](spec.md#automatic-residency-internal-parent-ipc) for the exact hop.

The result is `not-dispatched`, `loaded`, `failed`, `timeout`, `cancelled` or
`unknown`. This DTO has no reason field; local refusal conditions and typed
proxy reasons are documented in the spec. `loaded` requires a new authoritative
post-call inventory observation with the current manager/model identity and
fresh model-specific context/slot readback matching its approved envelope;
even a lost load response can converge through that readback. Missing inventory
is unknown, while an explicit empty list is observed absence. Successful settings
re-apply alone does not prove that an already loaded child's allocation changed.
Every dispatched call pokes the loaded watcher. Desired residents never become
observed inventory or new desired-state-file fields. Duplicate operation IDs
coalesce for their original lifetime; changed payloads conflict. Idle unload protection belongs to
the broker.

Requests (caller → service):

| Method | Params | Result |
|---|---|---|
| `engine:get-installed` | — | `{ engines: [EngineStatus] }` |
| `engine:describe` | `{ engine }` | the engine's manifest (local only: it contains commands and paths) |
| `engine:descriptors` | — | `{ engines: [Descriptor] }` — every registered engine's sanitized, versioned descriptor (`nvpair-shared/engines`): `{ version, engine, family, label, platforms, available, protocols, modelNaming, modelStore?, catalog?, operations: { <load\|unload\|pull\|delete>: { responseTimeoutS? } } }`. It never carries binaries, commands, environment values, paths, URLs or credentials. See [MANIFEST.md](MANIFEST.md#descriptors) |
| `engine:status` | `{ engine }` | `EngineStatus`, including optional `buildInfo` (running engine claim) and `buildId` (content fingerprint), cleared on stop; inspection exposes the same observed values as `nativeBuild` and `nativeBuildId` |
| `engine:install` | `{ engine }` | `EngineStatus` (starts when available) |
| `engine:uninstall` | `{ engine }` | `EngineStatus` (after removal) |
| `engine:set-port` | `{ engine, port }` | `EngineStatus` (after rebind) |
| `engine:action` | `{ engine, action, params }` | the engine's raw response for known inventory reads or manifest-bound unload, pull and delete operations. Load bindings, inference and unclassified custom actions are refused; model allocation must enter admitted load or proxy inference. `action:"pull_model"` streams `engine:pull-progress` and returns its terminal result. A permitted action with `restart_after` restarts a running engine before replying |
| `engine:logs` | `{ engine }` | `{ lines: [LogLine] }` |
| `engine:errors` | — | `{ errors: [ServiceError] }` |
| `engine:catalog` | `{ engine, platform?, arch?, query? }` | `{ models: [{ id, name, author, url, size?, downloads, likes, updatedAt, tags, family?, parameterSize?, appleOnly? }], source, platform?, arch?, fetchedAt?, searchable?, query? }` — the models an engine can **download**, as opposed to `engine:models`, which reports what is already installed. Each engine family has one curated source, selected by the manifest's `family`; only llama.cpp has one (a fork with `family: "llamacpp"` gets it), and the client spellings `llama-cpp` and `llama.cpp` are accepted. It is fetched live from Hugging Face and cached for six hours, with concurrent callers coalesced onto one request, a failed attempt — with or without a list to fall back on — followed by a backoff before the next. llama.cpp downloads any GGUF repo as `repo:quantization`; its browse list is the GGUF repos of `ggml-org`, `bartowski`, and `unsloth`, keeping only open, text-generating repos that ship a `Q4_K_M` file, each offered as `repo:Q4_K_M`. It is the one `searchable` source: a `query` searches all of Hugging Face under the same rules, each query cached on its own (the 20 most recently used are kept), and the reply echoes it. A query longer than 100 characters is refused rather than shortened, since a shortened one searches for something the operator did not type. `platform` and `arch` are the GOOS and GOARCH the models will be **installed on**, which is not always this host — a client driving a peer should name that peer. Omitting both means this host; naming a platform without an arch leaves the arch unknown rather than borrowing this host's. The llama.cpp list is not machine-locked, so nothing is filtered for the target, and `appleOnly` is never set; the reply echoes the `platform` and `arch` it was asked for. `name` is pull-ready: it can be handed to `pull_model` verbatim. An engine with no curated source is an error, not an empty list. The reply is a single JSON-RPC frame, so every hop on its path must allow `jsonrpc.WorkerFrameBytes`. |
| `engine:models` | — | `{ models: [string], modelsByEngine: { <engine>: [string] }, loadedByEngine: { <engine>: [string] }, modelIdentities?: { <engine>: { <model>: ModelIdentity } } }` — the flat de-duplicated union of every running engine's models, the per-engine breakdown keyed by engine name, and the per-engine set of models currently **loaded in memory** (all normalized from each engine's `list_models` / `loaded_models` action `result` spec). `modelsByEngine` carries a key for every running engine whose inventory was successfully queried, including an empty list = "running, no models available"; a missing key means not running / not queryable / invalid response. `loadedByEngine` uses the same known-empty distinction for residency and also omits engines with no loaded endpoint. `modelIdentities` is optional display metadata for llama.cpp-family models present in the readiness-masked `modelsByEngine`; each `ModelIdentity` follows `shared/engines.ModelIdentity` and grants no admission or residency authority. `policy` is this node's policy advertisement (`{ revision, roles, pausedBy? }`) as the broker last pushed it with `engine:node-projection`, omitted until the first push. The `/v1/models` HTTP surface returns the same shape. |
| `engine:remote-get-installed` | `{ node }` | `{ engines: [EngineStatus] }` fetched from the remote node over `ec` mTLS |
| `engine:remote-install` | `{ node, engine }` | `{ opId, status: EngineStatus }` after the remote install (live progress via `engine:remote-progress`) |
| `engine:remote-pull-model` | `{ node, engine, model?, params? }` | `{ opId, result }` after the remote pull (live progress via `engine:remote-progress`) |
| `engine:remote-copy-model` | `{ node, engine, model }` | `{ opId, result: { engine, model, files, filesSkipped, bytesTotal, bytesCopied } }` after copying the model from `node`'s engine store into this node's (live progress via `engine:remote-progress` with `op:"copy"`). See "Model copy between nodes" below |
| `engine:copy-model` | `{ opId, sourceNodeId, destinationNodeId, engine, model, expectedExportId? }` | `{ opId, result: copyModelResult }` after destination-owned verified publication |
| `engine:cancel-model-copy` | `{ opId }` | `{ cancelled: boolean }`; local channel only, exact active operation |
| `engine:active-model-copies` | `{}` | `{ operations: [{ opId, sourceNodeId, destinationNodeId, engine, model, expectedExportId?, progress? }] }`; memory-only active snapshot |
| `engine:model-copy-capabilities` | `{ nodeId, engine }` | Caller-specific versioned copy capabilities described below |
| `engine:wake` | `{ engine }` | Broker-internal `EngineStatus`: starts an installed engine when available; refused during shutdown |
| `engine:sleep` | `{ engine }` | Broker-internal `EngineStatus`: stops for availability, allowing a later wake or restore |
| `engine:load-model` | `{ engine, model }` | the engine's raw response — loads one model through the manifest's `load` binding, the same one the remote load uses |
| `engine:unload-model` | `{ engine, model }` | the engine's raw response — unloads one model through the manifest's `unload` binding, the same one the remote unload uses |
| `engine:delete-model` | `{ engine, model }` | the engine's raw response — deletes one model through the manifest's `delete` binding, the same one the remote delete uses. A binding whose action declares `restart_after` restarts a running server before replying |
| `engine:remote-policy-get` | `{ nodeId }` | `{ policy, availability, pausedBy?, activeRequestPolicy }` — the target broker's peer projection of its policy: no routing section, and tiers keep only their local entries. `activeRequestPolicy` is `{ revision, proxyRunId }` when the target's live proxy run has acknowledged exactly that revision, else `null`; it is relayed opaquely |
| `engine:remote-policy-set` | `{ nodeId, policy, expectedPersistedRevision }` | `{ policy }` — the peer projection after the target broker merges the change into its stored policy; the revision is forwarded unchanged and compared by the broker |
| `engine:remote-availability-set` | `{ nodeId, state, pauseMode? }` | the target broker's `{ availability, pausedBy? }` result (configured pause waits for the target's drain, on the readiness-sized response budget); immediate is refused by the target |
| `shutdown` | — | `null` |
| `log/set-level` | `{ level }` | `{ level }` |

`EngineStatus` = `{ engine, display_name, installed, running, healthy, port, terminationVerified }`.
`terminationVerified:true` is positive physical-stop evidence for an engine whose
host-platform block is absent. `running:false` and descriptor `available:false`
alone carry no such evidence. Missing or false evidence is unverified. Before
each process launch the manager saves the actual endpoint and executable, then
the PID after spawn, in `engine-state.json` (without arguments or environment).
If the host block later disappears, Sleep uses that cleanup-only record and the
existing managed-install containment / exact-image reclaim predicate, never
adoption. Both endpoint refusal and recorded process exit are required; a failed
reclaim retains its PID fence even if the listener closes. Missing legacy records,
an unresolved spawn-to-PID-save crash window, foreign listeners and indeterminate
probes cannot establish stop evidence, so node pause remains draining and refuses
success. The record cannot authorize starting an unsupported engine.



Notifications (caller → service): `engine:node-projection{policy, pausedLocally, managementBlocked}`
is this node's policy advertisement (declared roles with their profile bounds,
policy revision, pause origin), whether the node is paused locally, and
whether a pause of either origin, pending, draining or saved, fences peer
management writes. An older broker omits `managementBlocked`, which then reads
as false; it is never derived from `pausedBy` or `pausedLocally`. The
broker sends it after every persisted policy or availability change and on
every `engine:ready`. engine-manager keeps the last one in memory only and
serves `policy` on `engine:models` and `GET /v1/models`, so paired peers learn
this node's roles from the pinned inventory fetch they already make. It is
never put in mDNS TXT or in the peer-services descriptor. `pausedLocally` remains
provenance. `managementBlocked` fences all peer mutations and local activation,
and either-origin pause cancels accepted starts, loads and crash relaunch.

Streaming HTTP pulls have a 30-minute **inactivity** watchdog. Each increase in
a llama.cpp file's completed byte count refreshes that
watchdog, so an active download may run longer than 30 minutes; duplicate
progress frames and heartbeats do not extend a stalled pull. CLI-driven pulls
without structured byte progress retain the fixed 30-minute action timeout.

For llama.cpp, an accepted pull that ends before a matching `download_finished`
or `download_failed` event stops the active download. This includes caller
cancellation, remote caller disconnect, inactivity timeout, and premature SSE
termination. Cleanup checks `GET /models` and sends `POST /models/unload` only
when the exact model is still `downloading`; cached files are retained, and
models that have already completed are left alone.

The initial `POST /models` handshake has a separate 30-second total timeout and
continues through caller cancellation so its acceptance can still be read.
The pull then waits for cleanup, which has a separate five-second budget for
the inventory check and stop request together. A failed cleanup reports that
the download could not be confirmed stopped alongside the original pull error.
If the start acknowledgement is lost or unreadable, acceptance and cancellation
are reported as unconfirmed without unloading an unowned download. The monitored
model must match `params.model`; mismatches are rejected before subscribing.

`percent` on all three progress notifications is present only when the step is
measurable (download bytes, byte-progress pulls) or terminal (`100` on install
`done` / `already-installed`, `-1` on install `failed` and pull `error`). An
indeterminate step omits it — the install `verified` and `installing` stages
always do — so a UI renders the stage alone.

The `engine:remote-*` methods are the client half of remote engine
management: engine-manager resolves the target `node` in an `ec` peer
directory (fed by its own `discovery:subscribe{services:[ec]}` to the broker
relay), dials that peer's `ec` surface over pin-based cluster mTLS, and — for
`remote-install` / `remote-pull-model` — mints an `opId`, relays each streamed
progress frame up as `engine:remote-progress`, and settles the request with the
terminal result. They fail if this node isn't clustered or the target isn't a
pinned cluster peer. See "Remote engine management" below.

`engine:install`, `engine:uninstall`,
`engine:set-port`, and `engine:action` run in their own goroutine on the
service side, so the read loop never blocks and their responses arrive when
the op finishes.

`engine:set-port` is the **persistent** port setter.
It validates `1-65535`, persists the choice as a manifest override (a
`{ engine, runtime: { port } }` delta written
to the per-user `engines/` dir that deep-merges onto the bundled manifest, so
`runtime.port` becomes the single source of truth and the port is restored on
the next start with no separate store), and applies it — bouncing the engine
onto the new port if it was running. Setting the port back to the bundled
default removes only the shared and host-platform port overrides; the file is
removed only when no other overrides remain. Arguments, environment, install
settings and other platform overrides are preserved. A host-platform port is
updated when necessary so it cannot shadow the saved value on restart.
Malformed override files fail the save instead of being replaced.
Because the chosen port lives in the
effective manifest, restore is automatic: an internal start with no explicit
port and the adopt-on-fixed-port path both come up on the retained port.
Moving a **running, adopted** engine is **refused** with an error (nothing is
persisted) — NVPAIR can't relocate a process it didn't start; see Adoption below.

## Lifecycle

Combined launch/server/proxy edits use the broker's
[engine settings protocol](../nvpair-ui-broker/ENGINE_SETTINGS.md). The worker
accepts optional `workingSetLimitMiB` (256 through 1048576) alongside
`serverPort`, `proxyPort` and `launchText`. This setting is available only for
Windows llama.cpp engines in process mode, and persists as
`working_set_limit_mib`. Each owned engine process and descendant receives the
same hard working-set maximum. Job notifications trigger an immediate member
sweep, with periodic sweeps on a 250 ms target cadence covering lost notifications.
Processes may fault in pages before their first cap; a late cap trims the working
set. An unopenable member terminates the job; a live member whose cap fails is
terminated (see [spec.md](spec.md)). Changing or clearing it restarts a running engine.
Readback `workingSetLimitStatus` is `applied`, `unverified`, `not-running`, or
`unenforced-adopted`, and is omitted when no limit is saved. An adopted process
requires a restart for enforcement of new descendants.

`unverified` means this manager started the running engine with the saved value,
but its containment has not verified every live member: a member could not be
capped (it is terminated, and the status returns to `applied` after the next clean
sweep), or the job could not be read (the engine job is terminated). `applied`
means a full verification (the root cap before resume, then each sweep on a
250 ms target cadence) completed within the last second and found every live
member capped. A stalled or stopped watcher also reports `unverified`.

The worker
checks basic argument syntax and adapter-declared networking/CORS controls
using [`pair-arguments-v1`](LAUNCH_TEXT.md), persists host-platform `launch_args` and `launch_env`
with the port, and holds its operation lock through stop/rebind/start. The
paired engine-control surface relays settings to its local broker and streams
full authoritative snapshots to pinned peers. Outbound settings subscriptions
require current peer-service admission, which excludes this node's own cluster
principal even when discovery gives it a different host UUID. The discovery
directory retains self for model-copy identity resolution; mesh self-trust is
not permission to start a peer subscription. Failed peer streams retry after
three seconds.

Preset conversion preserves native sampler defaults, reasoning budget and output
limits as extra options. Native `flash-attn = auto` becomes inherited policy;
f16 and q8_0 cache precision remain explicit. Native-fit `gpu-layers = auto`
stays unpriced, while finite `all` resolves to the model plus output layer.
`cache-ram = 0` disables the host prompt cache and is priced as zero; negative
values are refused. None of these conversions proves native memory fit.

The finite llama.cpp adapter accepts projector-disable switches `--no-mmproj`
and `--no-mmproj-auto`, the presence of `LLAMA_ARG_NO_MMPROJ_AUTO`, and
`mmproj-auto = false` (also `0` or `off`) in environment, presets and typed
extra options. Negative preset keys such as `no-mmproj = true` disable
automatic selection and survive whole-preset conversion, including in `[*]`.
Enabling automatic selection, `--mmproj` and `--mmproj-url` remain refused.

For custom llama.cpp launches using native `--models-preset` (including the
equals form), model presets own context, parallelism, idle retention and cache
RAM. Engine-manager omits the manifest's fixed `--ctx-size`, `--parallel`,
`--sleep-idle-seconds` and `--cache-ram` defaults. Explicit user options still
override the preset without a duplicate manifest default. Other fixed startup
options remain, unless explicitly supplied by the user; launches without a
preset retain their existing fixed arguments.

```
NotInstalled --engine:install--> (HTTPS download + verify-if-pinned + user-mode run) --> Stopped
Stopped      --internal wake---> (adopt if already serving the port, else spawn) --> Running --health--> Running
Running      --internal sleep--> (stop signal, bounded grace, force if needed) --> Stopped
```

Detect uses the manifest's `detect` paths. Install is one-shot and
user-mode — an HTTPS download, verified against the manifest's `sha256`
when one is pinned (an unpinned fetch runs with a loud warning). Start
waits for the readiness probe, then runs a periodic health probe; an
unexpected exit is reported. The deadline is finite: if an engine never serves
its readiness endpoint, engine-manager stops the owned process and reports the
failed start. On Unix, stopping an owned process sends SIGTERM to
its process group, waits the manifest's `stop.grace_s` (five seconds by
default), then escalates to SIGKILL; `signal:"kill"` skips the grace. Failed
startup cleanup uses the same policy. On Windows, the windowless engines we
spawn cannot receive a graceful (non-`/F`) close, so stopping uses immediate
`taskkill /T /F`.

Unix signals target the whole process group only when the target PID is its
group leader; a nonleader is signalled alone. This also applies to orphan
reclaim, so a shared group retains its unrelated processes.

### Adoption — start may attach to an engine it didn't launch

Before spawning, the Executor **probes the chosen port's readiness
endpoint**, including any manifest-declared JSON identity. If a compatible
service answers there — the engine's own desktop
app, or an instance left running from a previous session — the service **adopts** that instance: it marks the engine
`running` without launching its own, rather than spawning a duplicate that
would only collide on the port. Consequences worth knowing:

To get a **NVPAIR-owned, stoppable** instance, start it on a port nothing is
already serving — the probe misses, so the service spawns and owns the child
(tracked in `st.proc`), so internal lifecycle cleanup can terminate it.
`engine:set-port` to a free port does exactly this; quitting the
external app first and re-starting on its usual port works too. Auto-assigned
ports (manifest `runtime.port: 0`) never adopt — there's no fixed address to
probe — so they always spawn an owned process.

## Model sources

Model sources are folders where other applications keep models on this node.
Engine-manager reads them in place and never writes to them. The built-in
sources are:

| Source | Folders | Model ID |
|---|---|---|
| `ollama` | `OLLAMA_MODELS` when set, else `~/.ollama/models` (Linux also `/usr/share/ollama/.ollama/models`) | `ollama/<name>@<tag>`; `<name>` gains its namespace outside `library` and its registry host outside `registry.ollama.ai` |
| `lmstudio` | `~/.lmstudio/models`, `~/.cache/lm-studio/models` | `lmstudio/<path>`: the file's folder path and name without `.gguf` or a `-00001-of-NNNNN` shard suffix |

An Ollama model is its manifest's model layer, with the projector layer as its
multimodal projector; a model with an adapter layer is skipped. In LM Studio
folders, GGUF files nested up to four folders deep are models, split models
count once by their first shard, and files whose name contains `mmproj` are
projectors. A folder's only projector pairs with every model in it. Files that
are not GGUF are skipped. When two folders produce the same ID, the one listed
first keeps it.
IDs use `@` rather than `:` because llama.cpp rewrites a tag after a colon.

Sources are engine-neutral: each found model carries its format, and an engine
offers the formats it can run. llama.cpp is currently the only consumer. At
each start of a llama.cpp-family engine, engine-manager scans the sources and
writes a native preset to `model-sources/<engine>.ini` under its engine
install base, one section per GGUF model naming its `model` (and `mmproj`)
path, then adds `--models-preset` with that file. The router lists those models
beside its own store. The manifest's fixed launch defaults stay, since the
preset sets only paths. Models whose ID fails [model identifier](spec.md#model-identifiers)
validation, or whose path contains `#`, `;`, `$`, `%`, `{`, `}`, a line break or
NUL, are left out. With nothing to offer, the file is removed and no flag is
added.

The preset is skipped when the launch declares `runtime.serving`, carries typed
model loads, or already names a preset through `--models-preset` or
`LLAMA_ARG_MODELS_PRESET`; those launches own their model list. Sources are
scanned only at start, so a model added later appears after a restart. Peer
copy, model settings inspection and serving presets cover only an engine's own
store, not source models. Adding sources by hand is not supported yet.

## Remote engine management

When the parent passes `--control-port`, engine-manager serves the **`ec`
surface** — a cluster-scoped remote-control endpoint over **pin-based mutual
TLS** (`nvpair-shared/clustertrust`, the same identity + per-peer pins
`nvpair-cluster-manager` mints). It presents this node's cluster leaf, requires a
client cert, and rejects any caller that isn't a byte-for-byte pinned cluster
peer with a `403`. It differs from `em` (`--http-port`) only in what it permits:
`ec` performs privileged operations, while `em` is a read-only inventory. Both are
locked to pinned cluster peers on the LAN — a node's model list tells a caller
which models that machine holds, which is cluster data like any other. `em` also
keeps a plaintext personality on loopback, because this node's own scanner reads
its own inventory that way and must be able to while unclustered.

Membership is evaluated **live**, per handshake and per request, from
`--cluster-dir`. While this node is not a cluster member it presents no leaf, so
every handshake is refused and nothing privileged is reachable; the moment it
becomes a member the same listener serves pinned peers. The listener is therefore
bound for the life of the process rather than only when clustered at startup — a
node joins and leaves a cluster while engine-manager runs, and a surface chosen
at bind time would stay dark until the process was restarted.

`--control-host` selects the literal IPv4 or IPv6 bind address for `ec` only;
its default remains `0.0.0.0`. Isolated fixtures can pass `127.0.0.1` (or `::1`
where available). Empty values, DNS names, bracketed addresses, zone suffixes
and host/port pairs are rejected before startup, without a fallback address.
The broker forwards `--engine-manager-control-host` to this flag. This does
not isolate the whole process: `em` (`--http-port`) still binds independently
on `0.0.0.0`, and fixtures must disable or independently isolate it and all
other listeners. Control mTLS, membership, pin and management checks apply
unchanged on loopback.

Pinning authenticates the caller; it does not authorize changes. The `ec`
registry explicitly classifies every route as read or mutation, including POST
reads such as policy get and settings get/preview. Before dispatching any
mutation, the common gate freshly reads
`clustertrust.Mesh.RemoteManagement` for the verified mTLS UUID. Install, pull,
load/unload/delete, settings apply, policy edits and
pause/resume all require an effective allow. Unclassified routes fail closed.
Missing, corrupt, pending or changing permission state denies; an earlier UI
read never authorizes a request. Broker-owned writes recheck at their locked
acceptance point after queueing. Direct operations accept at this gate; later
lifecycle/content lock waits are part of that accepted operation.

Permission refusal is HTTP 409 with `{"code":"remote_management_denied","error":...}`
(`clustertrust.WriteRemoteManagementDenied`, which owns the code, text and body
shape shared with the cluster manager and the broker), preserved
as JSON-RPC `-32012` by ordinary and streaming clients, before progress headers
or side effects. It is distinct from an unpinned 403, local veto
`-32011/local_veto` and conflict `-32010`.
Every one of these typed codes reaches the origin's JSON-RPC client with its own
number, through the 409 body, a stream error frame and the broker relays alike,
with the target's message and no JSON-RPC error `data`. Reads, status, model exports and
permitted inference stay available without management permission. Pull-copy
reads the source and writes the local requester; it grants no remote destination
write authority. No inbound destination-copy route exists.

`GET /v1/remote-management` returns only this authenticated caller's
`{version:1, allowed, revision, state}`, without a peer selector or grant list.
The local client method is `engine:remote-management-get {nodeId}` using the
existing pinned direct/reverse client. States are `allowed`, `denied`,
`unreviewed`, `unavailable`, or client-reported `unsupported` for an old target's
404/405. Only a fresh `allowed` result reports permission; it is not a promise
that a later mutation will pass. There is no remote setter. Grants and the
one-time existing-pin migration belong to
[cluster-manager](../nvpair-cluster-manager/README.md#remote-management-permission).

Revocation blocks new acceptance, not already accepted target work, which can
finish/reconcile and remains readable through existing progress/receipts. Local
pause, lock and operation-scoped local-stop cancellation still apply; grants cannot override them or private,
CORS, served/resident and admission protections. Downgrading the target to
pre-PG software restores that executable's old pinned-peer behavior. The gate
logs caller/target UUID, bounded operation family/class, decision, permission
revision and reason, without paths, queries, bodies or secrets.

Endpoints (all under `/v1`):

| Route | Shape | Purpose |
|---|---|---|
| `GET /v1/engines` | JSON | remote `engine:get-installed` |
| `GET /v1/remote-management` | JSON | caller-only effective permission, no grant list or setter |
| `POST /v1/engines/install` | NDJSON stream | remote install (+ optional start) with live progress |
| `POST /v1/models/pull` | NDJSON stream | remote model pull with live progress |
| `GET /v1/models/files?engine=&model=` | NDJSON stream | the model's files with size and sha256, for a peer copying it |
| `GET /v1/models/file?engine=&model=&path=` | file bytes | one listed file, with HTTP Range support |
| `POST /v1/node-policy/get` | JSON | the target broker's peer projection of its policy, as for `engine:remote-policy-get` |
| `POST /v1/node-policy/set` | JSON `{ policy, expectedPersistedRevision }` | the peer projection after the target broker merges the change; the revision is forwarded unchanged |
| `POST /v1/node-availability/set` | JSON `{ state, pauseMode? }` | the target broker's `node:set-availability`, answered once the configured state is reached; peer immediate requests are refused |

The node-policy routes carry the broker's policy, which engine-manager does not
interpret. Each is gated on the caller's pin like the settings routes, accepts at
most 256 KiB, refuses unknown fields, clears any `nodeId` so the target acts on
itself and never forwards a further hop, and relays to the broker as
`policy:request` stamped with the authenticated caller. The broker re-checks the
pin, acts, and answers with `policy:reply {id, result, error?, errorCode?}`.
`pauseMode` is carried unchanged, including explicit null, for target-side
validation after fresh permission and pause-fence checks. Omission means
`configured`; `immediate` is target-local only, so an authorized unfenced peer
receives `-32602`, `Immediate pause must be requested on this node.` No payload
`nodeId` can impersonate locality. Any mode on Resume is invalid. Local
immediate calls start cancellation and controlled shutdown without the drain
wait; this peer relay cannot invoke them. Timing is never saved. Results preserve actual target
`pausedBy` (`local`/`peer`, unknown empty/omitted, cleared while available),
without exposing local activation or origin-private routing fields.
Every error carries a JSON-RPC `errorCode`: `-32602` for invalid parameters,
including missing `expectedPersistedRevision`, invalid policy or availability;
`-32010` for a revision conflict; `-32011` for a resume the local veto refuses;
`-32012` for remote-management denial; `-32014` for target-owned profile inference settings;
`-32000` for a save failure or other operational failure. The route maps
`-32602` to HTTP 400, `-32010`, `-32011`, `-32012` and `-32014` to HTTP 409 with the body
`{"error": <diagnostic>, "code": "policy_conflict" | "local_veto" | "remote_management_denied" | "remote_setting_locked"}`, and other
failures to a plain-text HTTP 409. The origin's remote-policy methods map HTTP
400 to JSON-RPC `-32602`, a typed 409 to its code, and other HTTP failures to
`-32000`, preserving the diagnostic. No classification depends
on diagnostic text. A stale policy draft is a conflict before its policy is
validated, including a draft that tries to enable remote or paid execution.
The route abandons its wait (and sends
`policy:cancel`) if the caller loses its pin or 15 minutes pass; a pause the broker
already accepted still finishes.

`remote_setting_locked` preserves the target broker's safe bounded explanation
and maps back to `-32014` at the origin. It names only the stored profile's
engine/model; a new-profile lock attempt receives a fixed generic message.
There is no identifier payload or error data, and classification never parses
the message. The broker, not this relay, owns the latest stored `localOnly`
check after permission and revision checks. A refused write has no partial
policy, persistence, activation, push or notification effects. See the
[broker contract](../nvpair-ui-broker/README.md#model-inference-settings-and-target-owned-locks).

After a fresh grant check, the common `ec` registry gate returns HTTP 503 for
every mutation until the first projection. A known `managementBlocked` returns
HTTP 409 `{"code":"local_veto","error":"This node is paused. Resume it on this node before managing it remotely."}`.
This covers install, load, unload, delete, pull, settings apply
and policy set. Availability set alone passes the pause check to
the broker, which permits configured Pause/Resume of either origin, refuses
Resume of Halt, and accepts repeated Pause of completed local Halt as a no-op.
Pin validation, the fresh grant and projection hydration still
apply before availability reaches the broker. Reads, read-only POSTs, events, model files and HEAD
remain admitted. Grant denial keeps `-32012` precedence.
Only a local Resume clears Halt or a pause whose mode is unknown.

Local start/load also refuse known pause; local reads, configuration,
internal stop and unload remain usable. Accepted starts/loads are canceled
for either pause origin, and settings/recovery cannot reactivate an engine.
Accepted downloads may finish without activation. HTTP coded bodies, stream terminal errors,
JSON-RPC relays preserve `-32011` and the target's message.

The streaming routes emit zero or more `{"type":"progress",...}` frames followed
by exactly one terminal `{"type":"result",...}` or `{"type":"error",...}` frame.
The initiating node's engine-manager consumes that stream, relays each progress
frame up as `engine:remote-progress`, and settles the originating
`engine:remote-*` request on the terminal frame — so a UI gets one synchronous
response plus a live progress feed keyed by `opId`.

The broker wires both directions: it passes `--control-port`/`--cluster-dir`,
registers `ec` with the discovery daemon whenever a cluster dir is configured, and
relays engine-manager's `discovery:subscribe{services:[ec]}` into the relay
directory so the peer directory stays current. It does **not** restart
engine-manager on `cluster:identity-changed` — the surface follows membership on
its own.

## Model copy between nodes

`engine:copy-model {opId, sourceNodeId, destinationNodeId, engine, model, expectedExportId?}` asks the selected destination to fetch directly from the
selected source. Device selectors are stable host UUIDs; `engine` is the exact
manager ID, including custom IDs. The controller carries only control and
progress. Its model store is never opened for a copy between other devices.
`engine:remote-copy-model {node, engine, model}` mints an operation ID and
invokes this same coordinator with `node` as source and this device as destination.
The wrapper retains its original field-validation order and drops unrelated
parameters when constructing the canonical tuple.

Canonical RPC and inbound HTTP copy requests accept exact, unique field names
and valid model identifiers, with a complete 16 KiB control-body limit. Operation IDs are 32 lowercase hex
characters. Identical endpoints, unknown fields (including caller, URL and
path overrides), duplicate active IDs and excess active work are refused before
peer lookup or transfer. Each Manager owns at most eight active reservations,
including inbound copies; there is no queue or terminal history. Cancellation
and shutdown close the exact request context and retain resumable partials.
Closing a GUI subscriber alone does not cancel a copy. A reconnect snapshot
that no longer contains an operation establishes an unknown outcome, not success.

The destination route is pinned `POST /v1/models/copy`, classified as a
mutation. Its body contains only `opId`, `sourceNodeId`, `engine`, `model` and
optional `expectedExportId`. The destination rechecks the authenticated
controller's grant and its pause gate while holding a reservation.
The destination independently needs a live source pin; controller trust cannot
substitute for that relationship. Copy does not install or activate an engine,
and does not require Running. A dispatched copy
POST is never replayed through an alternate route.

`engine:model-copy-capabilities {nodeId, engine}` reads the actual node's
`{version:1, engine, available, installed, modelStore, exportLayouts, importLayouts, writable, reason, identityBoundRetry, receiptDurability}`.
The peer route is pinned `GET /v1/models/copy-capabilities?engine=...`;
`writable` reflects readiness and the actual caller's target-local permission.
The adapter name is `llamacpp`; llama.cpp also supports
`llama-artifacts-v1`. No filesystem paths or grant lists are returned. Missing
routes/versions are unsupported. Source and destination layouts must intersect,
the selected export must be supported by both, and the destination must be installed.
The capabilities GET has a 30-second context budget covering both response
headers and the complete bounded body, shortened by an earlier caller deadline.

An explicit retry can bind `expectedExportId` to a learned custom export.
After destination authorization/admission, a matching committed receipt is
fully rehashed before any source lookup or capability request. A verified
receipt can finish with the source offline, reporting `durability:"uncertain"`.
Otherwise the source must be live and its export identity must still match
before writes. Without an export ID, retry resolves the alias afresh and cannot
guarantee the same content after a lost response. Timeouts are never success.

The shared pinned peer/service adapter owns both listing and file transport.
It resolves the current admitted source and advertised `ec` endpoint, prefers
direct mTLS, and can use the existing authenticated reverse session after a
bounded failure before HTTP dispatch. Inner mTLS still reaches the source's
ordinary pin-gated control handlers. Listing frames and file bytes stream with
backpressure and cancellation; the adapter adds no listener, full-file buffer,
or replay after dispatch or response exposure. A broken file response ends that
HTTP attempt. The copy owner below issues a new Range request from the retained
partial's current length and checks the completed file's hash; transport does
not invent a second resume policy.

On the source, `GET /v1/models/files` resolves the model's files from the
engine's on-disk layout and lists logical paths relative to that layout's root,
with size and sha256. Hashes are computed when first asked for, by streaming
the file, and cached by absolute path, size and modification time. Files whose
name is already their sha256 retain their native declaration;
HF snapshot files use the checked full hashing/cache owner even behind local
LFS blob names. The destination independently verifies every complete file. The
listing is an NDJSON stream in the shape of the other streaming routes:
`{"type":"progress","stage":"hashing","message":<path>,"percent"?}` frames
while files are hashed, then one `{"type":"result","result":{engine, model,
revision?, exportId, files:[{path, size, sha256}]}}` or `{"type":"error"}` frame, so the
first hash of a large model does not run into the 30-second response-header
budget. `GET /v1/models/file` serves one file with `http.ServeContent`, so
`Range` requests work. It only serves a path that is exactly one of the
model's listed files; a malformed path (`..`, absolute, backslashes, drive
letters) is a `400`, anything else not in the list is a `404`. Native sources
can resolve outside the store when they belong to the exact selected model
binding described below. A peer never supplies a host path.

Custom llama.cpp names are resolved separately from catalog identifiers. An
exact configured native preset section supplies the local model and optional
draft GGUF shards, in native order. Persisted resolved typed model loads can
also select exact artifacts without `Runtime.Serving` or a running engine.
Export checks the selected configuration,
bindings and applicable asserted full SHA256 values against streamed full-file
hashes. It does not recompile the whole engine for copy or grant launch
approval. Full launch and apply retain their separate review checks. Without a declaration, the same bounded preset
parser and effective launch options select the exact section; relative paths
are resolved against that preset. Projectors, adapters, includes and source URLs
are explicitly refused on this custom path. Only these exact configured files
may be exported outside `LLAMA_CACHE`; peers cannot provide host paths. A custom
name with no unique binding is refused without guessing a Hugging Face repo.
Committed custom-copy receipts can also supply an exact local binding. Different
preset, typed or receipt artifact sets for the same alias are ambiguous.

Custom listings add `layout:"llama-artifacts-v1"`, `exportId`, and `role` on
each file. Paths are synthetic `model/<GGUF basename>` and
`draft/<GGUF basename>` entries; host paths and serving settings are absent.
The export ID binds the exact model name and ordered role/path/size/full-SHA256
manifest (canonical versioned JSON is specified beside `artifactExportID`). Each
file GET must carry this `exportId`. Native listings also issue an `exportId`
for the transfer, while retaining their native layout and result fields; it is
not a durable native receipt. Every GET or HEAD, including Range, requires a
prior listing and its issued ID. The source rechecks the stored selection and
opened regular-file identity before serving. Source-local issued bindings are
bounded; restart or eviction requires a fresh listing. Relisting unchanged bytes
may issue the same ID with newly checked evidence.
Listing results are bounded to 1 MiB, artifact sets to 1024 entries and progress
frames to 64 KiB independently of the stream's total length. Shared hashing
waiters cancel independently; cancellation of a producer lets a live waiter
retry under its own context.

### Transfer integrity and publisher comparison

Successful transfer integrity means the destination's complete bytes matched
the source manifest, including resumed files and checked existing files.
Optional `copyModelResult.publisherComparison` reports a separate observation:
`{status, files, replayed?}`. Each file contains `path`, `status`, `algorithm`
(`sha256`), `actualDigest` (the full lowercase SHA256), and `acquisitionBasis`.
A proven selector also carries `provider:"huggingface"`, `repository`, immutable
`revision` (full lowercase commit SHA), and exact repository-relative `file`.
Acquired statements add `expectedDigest` and `acquiredAt` (UTC RFC3339).

Per-file states are `verified`, `mismatch`, or `unavailable`. Verified is exact
agreement with a provider statement independently acquired by this destination;
it establishes neither model safety, author trust nor a signature. Mismatch is
displayed and never labelled verified; it does not veto an authorized copy.
Declared artifact assertions and complete transfer hash mismatches still fail
through their existing owners. Missing optional comparison fields mean
unavailable. Aggregate mismatch wins over other states; aggregate verified
requires every listed file, including shards and companions, to be verified.
A verified subset with an unavailable file is unavailable, with each file's
individual status retained in expandable provenance.

The first supported mapping is a native llama.cpp HF snapshot's exact public
repository, immutable revision and selected relative file. The source listing
optionally adds per-file `publisher` selectors. Peer status, digests, acquisition
time and cache claims never authorize a destination observation. The destination
uses its existing catalogue HTTP owner to GET the fixed HF model revision API
with `blobs=true`, requires the exact returned repository/revision and explicit
public/ungated facts, and compares the exact sibling's LFS SHA256. No peer URL,
mutable alias, filename similarity or local LFS blob filename supplies authority.
Custom aliases, private/gated repositories, non-LFS files,
missing mappings and provider failures are unavailable; authorized transfer
remains available. No credentials, settings, request fields or capabilities are
added. Executor-only callers without a catalogue owner report unavailable.

Revision metadata is acquired from the catalogue owner's configured endpoint
without redirects. Malformed provider encodings are
unavailable. Optional source selectors are omitted as a complete set if they
would exceed the existing copy-manifest bound; the byte identity and authorized
copy remain available, with publisher comparison unavailable.

The cache uses the catalogue's existing search capacity, six-hour TTL,
failure backoff, HTTP deadline and
8 MiB body limit. Expired statements are unavailable when refresh fails.
`acquisitionBasis:"provider"|"cache"` distinguishes the read, while `acquiredAt`
always records the original successful acquisition. Unavailable records carry
`acquisitionBasis:"unavailable"` and no expected digest or acquisition time.
Optional records reject malformed/duplicate fields, unsafe paths, non-full
digests, inconsistent status and invalid times. Result observations use the
existing 8 MiB metadata envelope; source manifests and receipts retain their
1 MiB limit, artifact count limit and separately bounded progress frames.

Local receipt replay rehashes every committed artifact first, sets `replayed:true`
and preserves its recorded acquisition basis/time. It makes no provider call
and invents no online freshness. Receipts without observations replay as
unavailable. Native copies gain no new durable receipt. Observations are excluded
from native/custom ExportID calculations and do not alter pinned reads, grants,
Pause/recovery, serving configuration, launch/apply or Auto behavior. The copy
modal labels Transfer integrity and Publisher comparison separately and expands
the complete per-file tuple, digests and acquisition history.

Provider tests inject the catalogue client's response transport with synthetic
statements. Existing copy control fixtures explicitly return provider-unavailable;
executor-only tests have no catalogue owner. These tests do not contact HF.

### Linked model sources

PAIR uses the engine's recognized model inventory, rather than importing loose
files. For a recognized llama.cpp Hugging Face cache model it binds the selected
ref, snapshot, GGUF shards and chosen projector. Supported source file and
directory symlinks, Windows junctions, and ordinary file hard links may point
outside the configured store. Filesystem permissions still apply. Directory
hard links are unsupported.

The listing freezes the ordered logical file set and its selector, directory
and opened-object evidence. Reads come from a checked open handle: every read
rechecks that handle's identity, size and modification time, and the configured
path is resolved again at open, by any read that covers a 64 MiB boundary
(always the first read) and after the last read. A replaced or retargeted
source requires a fresh listing. A linked directory does not authorize later
unlisted descendants. Peer file requests use only engine, model, logical
name and issued ID. For llama.cpp, peer inspection exposes path-free facts for
the selected model under existing authorization. Existing settings administration and
removal behavior are unchanged; destination paths remain confined. File hard
links share mutable bytes with their other names.

Identity and sampled bindings do not detect every same-stat middle-of-file
mutation, and a native engine can open a path later after PAIR's check. Local
writers are trusted; these checks do not isolate hostile writers. Copy validates
the selected export, not the whole launch, and does not establish publisher
authenticity. Destination full-hash verification checks transfer integrity.

The composed linked-model selector passed on macOS at integration source
`3300897f`, with 441 test and subtest results passing and no skips or failures
in each plain and race run. These are synthetic fixtures: fake native rows and
responses exercise PAIR's selection, checks and transport, not actual vendor
recognition or inference. The Windows plain component selector at `43ab0f45`
passed with 413 passing and three skipped test records. The skips cover an
unconfigured isolated UNC share, privileged symlink creation and an HF hardlink
fixture setup defect; they are unverified, not passing coverage. The junction
matrix's separate ordinary-file hardlink case passed. Synthetic metadata
wrappers do not isolate type rejection or actual vendor-specific reparse
behavior. Native acceptance remains pending. Component reader
counters prove the instrumented read boundaries, not the absence of every
process-wide filesystem attempt. See the existing
[controller COPY process limits](#controller-copy-process-verification).

`TestLinkedModelsProcess` in `services/tests` adds isolated synthetic process
coverage on base `10b8802d`. On macOS arm64/APFS (2026-10-30), the plain selector
passed without skips. It exercises independent private controller/source/target
fixtures for external preset file links and a file hardlink, selected typed
export without `Runtime.Serving` despite unrelated model drift, and HF cache
links. Native results retain empty
layout and durable receipt identity fields; custom results retain their receipt
and fresh target settings review. Destination file sets and SHA256 values are
checked against synthetic inputs.

Pinned direct and reverse lanes cover logical-name and identity refusals,
GET/HEAD/Range, source retargeting, fresh listing after manager restart, copy
failure/retry, destination escape refusal, offline completed custom receipts,
and preservation of a failed catalog attempt's journal and readiness fence.
Successful HTTP response counters observe no sibling sentinel bytes. They do
not observe process-wide filesystem attempts. Fixtures use private profiles,
registered test images, fake native inventory and loopback endpoints; they do
not establish vendor recognition, model loading, inference, GPU behavior or
installed-profile acceptance. The independent source review's refusal-status
and retarget-hook assertions were strengthened; affected plain cases passed,
and the full race selector passed without skips. Plain evidence composes the
unchanged earlier cases with the repaired cases, not a new aggregate run.
An initial author-state receipt-locality assertion failure remains unexplained
despite later diagnostic and final race passes. At frozen source `39a61c4b`,
Windows x64 plain passed all 12 terminal test records (including the parent),
with no failures or skips. Existing listener/product image coverage was
verified; the fixed model-settings-manager bridge used the source-proven
listenerless allowance and was not counted as firewall-covered. No owned
survivors or wrapper locks remained. UDP endpoint changes were observed with
unknown causality; full listener equality is not established. Windows race and
the component Windows skips above are not covered by this process run.

On the destination, the copy:

1. Refuses an engine with no known layout, an engine that is not installed
   here, and a listing whose paths or hashes do not fit the engine's layout.
2. Skips files already present with the listed size and sha256. Different
   completed bytes are a conflict for every layout and are never overwritten.
   Unsafe destination symlinks and Windows case/path collisions are refused.
3. Refuses to start when the volume holding the store has less free space than
   the bytes still missing (net of partial downloads) plus 256 MiB.
4. Downloads each missing file into `<final>.part` beside its final name,
   resuming an existing partial with an HTTP `Range` request. Partials are kept
   across retries and restarts. A download that receives nothing for 60
   seconds, or drops, is retried from where it stopped, up to five attempts.
   Every attempt opens the partial through the store's rooted filesystem and
   checks its regular-file identity before truncation or writes. Its pathname
   and the renamed result must still identify the file whose bytes were hashed.
5. Verifies the sha256 of the complete file, then renames it into place. A
   mismatch deletes the partial and fails the copy (after one fresh retry when
   the attempt had resumed an older partial).
6. Writes the file that makes the model visible to the engine last.

Progress is `engine:remote-progress` with `opId`, `op:"copy"`, `node` naming the
source, `sourceNodeId`, `destinationNodeId`, `engine`, `model`, and `exportId`
as soon as a custom listing resolves. Stages are `listing`, `hashing` (relayed from the source, `file` set),
`checking`, `downloading` (`file`, and `bytesDone`/`bytesTotal`/`percent` over
the whole model, about every 500 ms), `finalizing`, then a terminal `done`
(`percent:100`), `error`, `cancelled` or `unknown` (`percent:-1`, `message`). Each accepted
operation emits exactly one terminal outcome; done follows verified publication.
Lost or invalid copy acknowledgements carry a typed unknown outcome and emit
`stage:"unknown"`, even if the destination already published the model.
Definite remote refusals remain errors; explicit cancellation remains cancelled.
An unknown outcome permits explicit idempotent retry and never establishes failure
or success from the transport error text. Copy stream error frames preserve this
distinction in `stage` independently of their diagnostic message.
Copy NDJSON uses the per-operation callback, bounded coalescing and checked
write/flush errors that cancel the transfer. It does not subscribe to another
operation's engine-wide progress. Terminal result byte counts and the custom
receipt qualification are retained separately from inventory.

| Engine | Store | What is copied | How the engine sees it |
|---|---|---|---|
| llama.cpp | `LLAMA_CACHE` (the manifest sets `{models_dir}`, `~/.llamacpp`) | For `owner/repo:TAG`, the snapshot GGUF matching `TAG` with all its splits, plus the multimodal projector beside it, chosen as llama.cpp chooses them | Files go under the destination's existing snapshot for the repo, or the source's revision with `refs/main` written last. The first split goes last. A running router is asked to rescan (`GET /models?reload=1`) |
| llama.cpp custom artifacts | `.pair-copies/<SHA256 of exact model ID>/<exportId>/` inside the destination's own store | Exact configured model and draft GGUF shards | Only the versioned alias-level `copy.json` receipt marks a completed bundle; native presets and inventory are not changed |

Limits: one copy of a given model at a time per node. An adopted llama.cpp
instance started outside PAIR may use a different `LLAMA_CACHE` than this
service resolves. The destination's sha256 check catches a source file that
changes during a copy.

Custom bundles use the same identity-withdrawal guard, file locks, free-space,
resume and full-hash checks. The receipt is published only after every artifact
is verified. Interrupted bundles without a receipt remain resumable and are not
completed copies. A different committed export for the same alias is a conflict;
identical receipts are no-ops after file verification. No automatic deletion,
native rescan, engine start, load or serving configuration follows custom COPY.

The additive Go result carries `layout`, `exportId`, `requiresConfiguration:true`
and `durability:"confirmed"|"uncertain"`. A replacement that landed but whose
durability could not be confirmed is qualified completion, preserving the
receipt and verified artifacts. Reading a previous receipt alone cannot establish
durability, so an identical retry reports `uncertain`. Custom bundles need an
explicit destination-owned serving setup before loading; receipts do not appear
in `ModelsResult` or native `list_models`. Dedicated bridge/UI presentation of
the setup requirement and durability qualification is separate work; these Go
fields do not establish that presentation or serving readiness.

### Controller COPY process verification

The isolated `services/tests` selector `^TestModelCopyControllerProcess$` drives
the typed broker RPC with separate synthetic controller/source/destination
identities and real engine-manager handlers. Its test source covers custom and
canonical layouts, authenticated direct/reverse transfer, positive-byte
interruption and explicit Range retry, controller cancellation, target permission
and pin failures, changed source content, destination conflicts, failed mutation
acknowledgement and lost-response no-replay behavior. See the
[fixture prerequisites and native checklist](../readme.md#controller-selected-model-copy).

On Windows at base `44e76c6b` (2026-10-26), the complete focused parent passed
plain and race with 14 named terminals, no failures or skips. Review then
strengthened the lost-acknowledgement fault to drop the downstream connection
before response headers; that changed leaf and its parent passed plain and race
(two terminals each). Module vet exited zero. Commands used the registered
fixed-path wrapper, private profiles and offline Go, from `services/tests`:

```sh
go test -p 1 -parallel 1 -count=1 -json -timeout=30m -run '^TestModelCopyControllerProcess$' .
go test -race -p 1 -parallel 1 -count=1 -json -timeout=30m -run '^TestModelCopyControllerProcess$' .
go test -p 1 -parallel 1 -count=1 -json -timeout=15m -run '^TestModelCopyControllerProcess$/^dispatched_POST_is_not_replayed_on_alternate_reverse_route$' .
go test -race -p 1 -parallel 1 -count=1 -json -timeout=15m -run '^TestModelCopyControllerProcess$/^dispatched_POST_is_not_replayed_on_alternate_reverse_route$' .
```

Raw output remains in the machine's temporary
`download-transfer-process-checks-20261026` directory. The combined whole-suite
run is separate from these focused results.
The fixture records/refuses controller artifact transport and checks that its
model store stays absent. Production artifact filesystem calls have no
process-level attempt observer, so these checks alone do not prove zero attempted
controller filesystem IO. Installed builds, actual source bindings, native UI
behavior, strict controller artifact IO instrumentation and destination serving
setup require separate acceptance.

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--ipc <path>` | _(stdio)_ | IPC endpoint: Unix socket or Windows named pipe |
| `--http-port <port>` | `0` (off) | Serve the plain LAN model-list surface (`GET /v1/models`, the `em` service) on this port; the broker passes `:14322` |
| `--control-port <port>` | `0` (off) | Serve the cluster-scoped mTLS remote-control surface (the `ec` service) on this port; callers are admitted only while this node is a cluster member. The broker passes `:14323` |
| `--control-host <ip>` | `0.0.0.0` | Literal IPv4 or IPv6 bind address for `ec` only; `127.0.0.1` confines that listener to loopback. Does not change the `em` listener or authentication |
| `--reserved-port <port>` | `0` (off) | Refuse local or remote engine starts and persisted port changes on a parent-owned proxy port |
| `--cluster-dir <dir>` | _(none)_ | Cluster identity/pin directory; gates the `ec` surface on and supplies the leaf/pins used to serve it and to dial peers |
| `--loaded-poll-interval <sec>` | `5` | Seconds between loaded-model polls that drive `engine:models-changed`; `0` disables the watcher |
| `--log-level <level>` | _(env `NVPAIR_LOG_LEVEL` or `info`)_ | `debug` \| `info` \| `warn` \| `error` |
| `--uninstall-managed` | | Remove every engine PAIR installed, then exit; for the platform uninstallers, which have no broker to call `engine:uninstall-managed` through. Same selection and safety path as that method, loading only the manifests compiled into this binary. Always exits 0 so an uninstaller cannot stall |
| `--version` | | Print version and exit |

Logs go to **stderr** via the shared `nvpair-shared/applog` format; stdout is
reserved for JSON-RPC frames in stdio mode.

## Logging & errors

Each managed engine's stdout/stderr is captured into a bounded ring
(queryable via `engine:logs`); serving output redacts path-bearing tokens, quoted
paths and the configured models directory before capture or diagnostics, and
oversized output is drained without retention.
Operational failures (install/start/health)
are recorded and surfaced as `errors:report` / `errors:clear`
notifications using the `nvpair-shared/errors` wire shape. The broker
(`nvpair-ui-broker`) forwards them to the `nvpair-errors` registry.
Ids follow `engine-manager:<class>:<engine>`.

## Security posture

Runs **user mode only** — no admin/sudo at runtime (privilege escalation
is reserved for NVPAIR's own install time). It has two optional LAN listeners, and
**both are pin-based cluster mutual TLS with a live membership check** — every
caller is authorized against a per-peer pin, a non-member is refused with a `403`,
and while this node belongs to no cluster it presents no leaf so no handshake
completes:

- **`--http-port`** — the model-list surface (`em`, `GET /v1/models`; the broker
  passes `:14322`). A node's model inventory is cluster data, so LAN callers must
  be pinned peers. This port additionally serves **plaintext on loopback only**,
  which is how this node's own scanner enriches its own card — including when the
  node belongs to no cluster, so a standalone machine still shows its own models.
- **`--control-port`** — the remote-control surface (`ec`, the `engine:remote-*`
  targets; the broker passes `:14323`). mTLS only, no plaintext personality, since
  every route performs a privileged operation.

Unlike the rest of NVPAIR, engine-manager therefore **does terminate inter-node
mTLS itself** (and dials peers' `ec` surfaces with the same pinned identity),
matching how `nvpair-errors` and `nvpair-workload-manager` handle their own
cluster traffic. Managed engines bind **loopback by default**, but a manifest's
`runtime.bind` may open an inference engine to the LAN; readiness/health probes
always target loopback. Clients configure networking through engine settings.
Downloads are **HTTPS-only** (plain `http` only from loopback) and verified
against the manifest's `sha256` when one is pinned; an unpinned fetch runs
with a loud warning.

## Cross-platform

One binary compiles and runs on Windows, Linux, and macOS × amd64/arm64.
Per-OS variance lives in the manifest first; OS primitives (process
termination, console hiding) are the only build-tagged Go
(`proc_windows.go` / `proc_unix.go`).

## Mutation admission and stop evidence

`mutation_admission.go` owns ordinary mutation admission, including waiting for
the lifecycle lock. Native `managerIncarnation` and `activeMutations` continue
to fence owner evidence and concurrent work. Lifecycle, settings, installation,
model actions, residency, wake/sleep and restore use their existing executor
owners; a profile names its engine.

Broker sleep and shutdown preserve cleanup evidence. Pause requires positive `terminationVerified` evidence.
Precise managed image/endpoint checks, failed-reclaim fences and foreign-listener
refusal remain. There is no public stop response or stop-verification option.

The existing desired-state decoder ignores retired transaction fields in place
without writing on load. The strict raw saved-state read for build-apply owner
evidence may refuse those fields until a normal save replaces the document.
This does not promise a repair or authorize a runtime mutation.

## Target-owned model downloads

`PullModelStream` owns local, CLI and paired catalog pulls. The existing
`engine:models`, `/v1/models` and `engine:models-changed` inventory carries
`downloadState` (version 1): `sourceId`, decimal-string `revision`,
`recoveryRequired` and `entries`. Each entry preserves the exact request model
and engine, a target-minted ID, `queued`, `in_progress`, `failed`, `complete` or
`unknown` phase, required `blocksReadiness`, optional measured integer percent
and a fixed reason code. `persistence:"uncertain"` qualifies an outcome whose
marker removal landed without confirmed directory durability. This does not
certify vendor-file durability. Pull progress carries model, ID and the full
snapshot; remote request `opId` remains a separate correlation identity.

Operation completion, current content inventory and observed loaded residency
are independent. Ready inventory and identity digests exclude queued and unsafe
content. The named llama.cpp SSE adapter projects only stable `unloaded`,
`loading`, `loaded` and `sleeping` inventory statuses. Name membership, transient
`downloaded`, percent 100 and accepted POST do not prove native completion.
An HTTP pull requires a declared `progress_protocol` and completes only on
llama.cpp's matching `download_finished`; CLI pulls require successful exit
without an explicit error.

`model-download-state.json` is a private, bounded unresolved-content journal,
separate from engine process cleanup evidence. It retains unsafe PAIR attempts across
restart and store changes without automatic effects. Its engine/content keys
use descriptor-selected naming; raw dispatch spelling is preserved. The manager
admits at most 32 accepted/unsafe keys, retains the latest 32 other outcomes,
publishes at most 64 entries, and bounds journal/snapshot bytes to 64/256 KiB.
Markers have no TTL or eviction. Malformed/unreadable recovery fences ready
publication and Load/export/pull effects. A same-key explicit successful retry
can resolve its marker. Successful Delete also requires a fresh same-run
unmasked absence observation; llama.cpp requires a forced cache rescan. Failed
cleanup, ordinary list absence and restart cannot clear a marker.

The same file and journal writer retain unresolved controller and destination
copies. The current version-1 shape requires `entries`, `copies`,
`copiesComplete` and `copyRecoveryRequired`; old shapes fail recovery rather
than being interpreted as an empty copy history. A missing file preserves ordinary download admission, but
copy safety remains unknown. Download writes preserve that unknown baseline
with `copiesComplete:false`. ROOT must coordinate an explicitly quiescent
current-format baseline or a fresh private profile with an explicit complete
baseline. Startup does no peer contact, artifact scan, cleanup or expiry to
manufacture one, and there is no copy recovery API. Ordinary constructors do
not provision missing journals or repair old ones; the explicit empty-old
amendment below is not runtime compatibility.

### Offline copy baseline

These standalone operator commands run as the profile owner, with root acting
as coordinator. They are not first-launch hooks. Reserve an exclusive offline
interval for either command. On Windows, `LOCALAPPDATA` must be set, nonempty
and absolute, and must be the profile owner's Local base; both modes refuse
before mutation otherwise. The current app is `BASE/Nvidia Corporation/Personal AI Router`
and the previous app is `BASE/NVIDIA Corporation/PAIR`, where `BASE` is the
resolved per-user configuration base. Their organization parents can alias on
case-insensitive filesystems: inspect the app directories, not the parent.

For a genuinely new whole profile, before any GUI or headless launch:

```text
nvpair-engine-manager --initialize-fresh-profile
```

The command exclusively allocates the current profile only if both app paths
are definitely absent, imports no history, and writes an empty complete
baseline. It is deliberately non-idempotent. A first launch may already create
the profile; installation, reinstallation, empty inventories, deleted journals
or recreated children do not establish freshness. A failure leaves allocated
directories and any landed bytes for review, exits non-zero and is not retried
or cleaned up automatically. Unix success confirms file publication and strict
readback, and every newly created directory entry through the first pre-existing
ancestor. Boundary errors leave allocations and return non-zero. Windows has no
supported equivalent for new-directory confirmation in this codebase: bounded
success confirms allocation and file publication/readback and reports
`fresh-directory durability: UNVERIFIED (unsupported)`. It does not confirm
whole-profile durability; retain that limitation in completion reports until
support is separately verified.

For an existing profile, after the review below, the separate amendment is:

```text
nvpair-engine-manager --amend-copy-baseline --expected-journal-sha256=HEX --copy-baseline-disposition=complete-empty-history --attest-profile-quiescent
```

`HEX` is the 64-character lowercase SHA-256 of the exact reviewed current
journal bytes. Add `--previous-profile-reviewed` whenever the previous app
directory exists, even without a journal. Add `--allow-empty-old-journal` only
when the current input is the precisely empty version-1 old object missing only
`copyRecoveryRequired`, with a valid boolean `copiesComplete`. The ordinary
current input must be strictly valid, with empty `entries` and `copies`,
`copiesComplete:false` and `copyRecoveryRequired:false`. The old opt-in is
refused for current input. No other runtime flags, positional arguments, false
offline booleans, replacement document, force or profile override are accepted.

Previous inspection accepts a definitely absent app or journal, a strictly
valid empty current journal, or the precisely empty old shape. Current-format
previous baseline flags do not certify history. The old opt-in concerns the
current input, not previous-only inspection. Any previous marker, unreadability,
unsafe path or inspection uncertainty refuses the amendment; the review flag
never overrides refusal. Retained downloads or copies, naming mismatch, current
recovery true, unreadable data and unsupported older shapes require separate
review. Do not delete, merge or ignore previous markers, including after partial
migration. General replacement documents are unsupported; actual machine
journal coverage remains UNVERIFIED pending separate inventory and review.

In order, root must:

1. Reserve the machine interval, prevent autostart and restart, and exclude all
   other profile users. Close GUI and TUI and stop service, broker and standalone
   owners using separately authorized procedures. The tool does not do this.
2. Verify controllers, destinations and downloads have stopped. Empty live
   lists do not prove historical completion. Review exact journal bytes and
   explicit history evidence for both current and known previous namespaces,
   including partial migration. Inspect the previous app and journal before
   attesting; preserve unsupported state for separate review.
3. Record the expected digest and disposition, then invoke the appropriate
   command once. For amendment, check the exact backup bytes, reported hashes
   and strict resulting state. Resolve any failure before releasing the interval.

Marker absence qualifies input only; it does not establish complete history.
The amendment requires the operator's quiescence attestation. Service sockets
and locks cover neither all profile users nor startup before listen, optional
ports may be disabled, and process snapshots can miss renamed processes or race
with startup. The tool refuses a nonempty `NVPAIR_SERVICE_ENDPOINT`: review and
stop custom-endpoint instances before clearing it. Its conservative process and
endpoint checks do not create a lease; repeated checks and hash comparisons are
not atomic CAS against ordinary writers. The exclusive interval must remain in
force after the tool exits until root releases it.

The amendment exclusively reserves the deterministic backup basename
`model-download-state.json.before-copy-baseline-SHA256.bak`, where `SHA256` is
the reviewed digest, and confirms an exact-byte backup before publication. An
existing backup, including a dangling link, refuses. It rechecks paths, previous
state, owners and current bytes/hash before backup and publication. Refusals and
uncertain writes exit non-zero; summaries report observed hashes, load and
landed state when available, with unavailable values otherwise. There is no
automatic cleanup, retry, restore, resume or persistent recovery protocol. A
crash or uncertain write after backup can leave complete-looking new bytes; a
later ordinary read cannot recover the earlier durability error. Preserve the
evidence and resolve state and durability before restart.

After a separately authorized real interval is released, ordinary startup may
reread the confirmed journal; read-only evidence must show the expected baseline
and source and remaining blockers. This guidance does not authorize that
startup. Neither command establishes full owner evidence or build-apply
eligibility: O1 still refuses real starts, O2 external-root coverage remains
separate, and O3 does not certify predecessors or infer historical completion.

Each retained copy binds its operation, engine, exact model and normalized
content key/naming, expected export identity when supplied, controller or
destination role, owner/source/destination node identities and original manager
incarnation. Only unsafe/uncertain provenance is stored, with no URLs,
credentials or cancellation functions. Downloads and copies share the journal's
aggregate entry and byte bounds and one serialized capture/save/adopt boundary.
Their short admission locks are released during durable I/O and transport.

Manager admission exposes pending publication before either role saves its
unsafe marker. No forwarding, source export/import or destination filesystem
effect starts until that save is confirmed. A failed save refuses effects; a
landed but uncertain addition remains unsafe and withdraws copy completeness.
Release, cancellation and shutdown retire live ownership only. Matching positive
terminal identity and synchronous local completion can clear the current
incarnation's exact marker. A verified receipt retry is positive local completion
even when its receipt durability remains qualified. Unknown outcomes, mismatches,
cancellation and successor reload retain the marker. A successor preserves the
original incarnation and cannot certify its predecessor's completion. Retained
records block build apply even when live operations are empty.

Content writers and Load/export readers have nonblocking admission beside the
existing content semaphore. An admitted reader makes a writer fail busy before
withdrawal or effects; an active writer refuses new reads. Load guards cover
typed/raw/remote actions and automatic residency. Export leases span hashing
and response-body completion. Existing in-flight inference is not cancelled.

Scanner observers keep last-good lifecycle data with `downloadStateCurrent`:
false means stale, absent means unsupported. A successful same-revision refresh
can restore currentness; older revisions cannot overwrite newer observations.
Origin and target notifications are immediate; other observers converge through
the existing refresh cycle and its work/request budgets. Updated clients require
current support and canonical ready inventory for ready-dependent actions. This
requires a coordinated fleet update. Files predating this journal are not
retroactively certified. Process, native UI and real-target acceptance remain
separate from component tests.

`services/tests/model_download_state_test.go::TestModelDownloadStateProcess`
contains process scenarios for target-local service subscribers, a newly attached
client, pinned observer inventory and production scanner self-enrichment. Its
synthetic SSE router exposes first-shard, delayed-terminal and disconnect
barriers. The assertions distinguish completion, stable ready inventory and
loaded observation; reject typed/raw Load and existing export reads without
native effects; isolate a cancelled queued model; and check direct/reverse POST
counts, failure recovery, failed journal removal and bounded retained state.
Service restarts use the existing private bundle's process-identity cleanup.
On Windows at base `44e76c6b` (2026-10-26), the final plain run below passed
17 named terminals with no failures or skips. The race run had 15 passes and
two failed terminals (the queued-cancellation leaf and its parent): Windows
refused replacement of the fixed scanner image during setup. That leaf alone
then passed under race (two terminals, no failures or skips). No data race was
reported. The failed aggregate run remains failed; no second full focused race
pass is claimed. `go vet ./...` exited zero.
The commands ran from `services/tests` with the fixed-path wrapper, private
profile and offline Go bindings. Raw output remains in the machine's temporary
`download-transfer-process-checks-20261026` directory.

```sh
go test -p 1 -parallel 1 -count=1 -json -timeout=30m -run '^TestModelDownloadStateProcess$' .
go test -race -p 1 -parallel 1 -count=1 -json -timeout=30m -run '^TestModelDownloadStateProcess$' .
go test -race -p 1 -parallel 1 -count=1 -json -timeout=15m -run '^TestModelDownloadStateProcess$/^queued_cancellation_has_no_native_effects_or_unsafe_marker$' .
```

These runs include bounded second-response NDJSON capture, replacement
reverse-session/read checks, control-surface POST counters and exact terminal
notification convergence assertions.
The control observer reuses a registered fixture image and the existing owned
port adapter; it forwards pinned HTTP streams to the real manager unchanged.

The fixture does not prove independent scanner peer refresh: authenticated peer
descriptors reject loopback addresses. Its production scanner uses only the
target's registered loopback self-enrichment route, with mDNS disabled. Automatic
resident dispatch is an internal broker/proxy boundary and is not exercised by
this process fixture. The compiled manager has no short progress-watchdog seam
or crash barrier inside durable journal replacement; watchdog and exact atomic
replacement failures retain component-level coverage. No lower-revision or
retired-source notification is injected by this fixture. The ec listener uses
`--control-host=127.0.0.1`; em and cluster-manager retain their production
wildcard binding on private fixture ports. No installed-machine or transfer-join
acceptance follows from these scenarios.

The separate `TestModelDownloadStateTransferJoinProcess` source regression joins
downloads to the accepted COPY implementation on base `efda20f7691a01980dd4740789d5d4829f6ba43b`.
It first requires a real custom-alias list and full artifact GETs with independently
checked hashes. Active and failed pulls then refuse that previously reachable
export and destination COPY: no artifact GET is dispatched by COPY, destination
store contents and mutation-withdrawal counts stay unchanged, and an explicit
successful same-key catalog retry restores export and COPY. Another scenario
creates a real COPY receipt, fails a catalog pull for the same exact key, and
performs an identity-bound receipt retry. Successful local rehash and skipped
files must leave the failed download snapshot, durable unsafe marker, receipt
and readiness fence unchanged. Assertions require absent controller artifact
stores, stopped controller engines and source/destination certificates on peer
reads. As in the COPY fixture, they do not observe attempted local artifact syscalls.

On Windows (2026-10-26), the joined selector passed plain and race with three
named terminals each, no failures or skips; module vet exited zero. Commands
ran from `services/tests` under the same fixed-path/private/offline bindings:

```sh
go test -p 1 -parallel 1 -count=1 -json -timeout=20m -run '^TestModelDownloadStateTransferJoinProcess$' .
go test -race -p 1 -parallel 1 -count=1 -json -timeout=20m -run '^TestModelDownloadStateTransferJoinProcess$' .
```

The merged whole-suite run remains separate. These scenarios reuse
the controller COPY fixture and its cleanup, with a fixture-only adopted SSE
router and Windows-safe aliases holding a 128-byte model and 4 KiB draft. They
do not extend the unsupported Task 5 injection or native-machine claims above.

## Shutdown behavior

Shuts down on stdin EOF (parent closed the pipe), `SIGINT`/`SIGTERM`, or a
`shutdown` JSON-RPC request — stopping any running engines first so none
are orphaned.

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

## Target memory telemetry

Inspection and preview consume the target broker's scanner-backed memory,
including its independent capacity/available provenance and remaining validity.
UMA is one shared pool. Device names and array ordinals never select capacity;
several dedicated GPUs leave placement unproved. GPU-selection extra options
remain accepted and unpriced. These estimates are advisory, never admission or
qualification. The process fixtures exercise fake observations through real
workers; they establish no native allocation accuracy. Production Auto and its
accepted-result recovery acceptance still require complete-cost accounting.

Stated figures are stored by the target for an applied finite configuration key,
with its content-based `nativeBuildId`. Local and pinned direct/reverse writes
use the same broker relay and manager store. A settings or native-build change
keeps the earlier record readable but makes it non-current. Earlier-build
history is labelled as an estimate, never validated or selected as current
measured evidence. A record with no retained build identity says so explicitly.
GPU-selection extra options have reason `gpu-selection` and no finite key, so
they cannot save a stated figure or use a buffer overlay.

Peer inspection preserves `profileKey` through the typed inspection relay, so
the target reports the selected applied configuration instead of inspecting
only its artifacts. Stated-memory writes from an origin broker use the target
`nodeId`; their replies and durable records belong only to that target.

## Internal memory diagnostic operation

The broker reserves execution with `engine:diagnostic-reserve`, then calls
`engine:diagnostic-run` with `{runId, token}`. The reservation binds the applied
model, profile, configuration/execution key and consented effects. Optional
explicit run fields must match that reservation. `generate` defaults to 128;
`deadline` is an RFC3339 timestamp bounded to twenty minutes. Unknown native-fit
targets remain unspecified until identity-checked `/props` reports the realized
per-slot context, slot count and build. Managed `/props` supplies no device or
GPU-layer fields. Placement requires complete buffer reports from that exact
load through the existing capture/parser owner. Host allocations map to the
host/shared pool; CUDA allocations join the native device identity owner's
PCI/full UUID (and Windows LUID) report in the child's effective environment.
Missing, incomplete, stale or unmapped evidence refuses identity before exercise.
Requested GPU options never manufacture observed placement. Unpriced loads may
retain private realization evidence without a public or persisted buffer sample.

Ordinary inspection can mark matching authenticated finite or ExecutionKey
measurements current, including retained valid realization after an unchanged
restart. Live uncertain realization supersedes past records. Native fit's
realized context can differ from its plan; changed inputs, artifacts, build, launch environment,
CPU/RAM/GPU identity, context or placement withdraw currentness. ExecutionKey
currentness grants neither an estimate overlay nor Auto eligibility.

`engine:diagnostic-run-status` accepts `{runId}` (empty selects the active run).
Its reply contains `state`, `token`, nested `run`, `workStopped`, retained
`markers`, optional `observation`, and provisional `low`/`high` request results.
`engine:diagnostic-changed` carries the same snapshot on phase/outcome changes.
Markers are `baseline-ready`, `loaded`, `low-start`, `low-end`, `high-start` and
`high-end`, with UTC times retained for the broker's memory samples. The broker
acknowledges the cold baseline with
`engine:diagnostic-baseline-done{runId,token,phase:"baseline"}`, which returns
`{ok:true}` only at that run's baseline barrier. The barrier expires after
thirty seconds.

`engine:diagnostic-abort{runId,token}` cancels and joins the accepted work before
replying. A new Pause/Halt cancels the run; the ordinary internal Sleep response
joins cleanup before completing the transition. An unchanged configured pause
permits the explicitly scoped measured engine and is preserved. Halt never
permits activation. Cleanup unloads all touched engines, stops an engine started
by the run, and escalates unconfirmed unloads through the existing owned-process
Sleep owner. `cleanup-failed` retains exclusion and abort retries cleanup only.
`workStopped:true` is affirmative cleanup evidence; a terminal phase alone is
not. Completed key-bound results remain available to the later record owner.
`engine:diagnostic-release` accepts repeated release of a completed matching
token without affecting a newer reservation. These methods are broker-only;
the public broker status method is the distinct `engine:diagnostic-status`.
