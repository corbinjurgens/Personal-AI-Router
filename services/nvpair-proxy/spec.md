<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Microservice: Engine Proxy (`nvpair-proxy`)

> **Status: implemented.** House spec style (`nvpair-cluster-manager`,
> `nvpair-workload-manager`, `nvpair-engine-manager`, `nvpair-job-scheduler`).
> Where this file and `README.md` disagree, this file wins.

## 1. Purpose

### Model identifiers

Accepted model metadata is nonempty valid UTF-8, at most 512 UTF-8 bytes, with
no C0 (U+0000–U+001F), DEL (U+007F), or C1 (U+0080–U+009F) characters. The
shared `nodepolicy` validator checks original identifiers before normalization
and selectable normalized results afterward. Identifiers are never truncated
or repaired into fallback names; ordinary Unicode, spaces, tags and slashes
retain their existing meaning. Request identifiers keep the stricter 256-byte
ingress cap. A short tier alias can select a valid 512-byte concrete model.

Invalid inventory/catalog entries are skipped individually. New policy writes
are rejected atomically; only the broker's stored-policy loader sanitizes old
invalid model entries and their dependent references in memory, preserving
valid siblings and explicit deny-all permissions without rewriting the file or
advancing its revision. These rules add no routing method or payload field.

### Local execution observations (T2)

Each actual engine-facing inference attempt may emit one `performance:sample`
with `scope: localExecution`, the process `runID` as producer and a positive
sequence assigned under the process-wide FIFO lock at finalization. A failed
local dial is an attempt. Admission refusal and control traffic are not.
Ordinary self routing, ordinary peer ingress and strict local dispatch share
the producer. Strict self dispatch is instrumented once. Direct and reverse
peer transports reach the destination producer; their origin creates no local
execution sample.

The profile key uses the measuring host, actual engine, normalized dispatched
model and a hash from the existing joined `EngineIdentity` (engine run,
settings fingerprint, launch digest and model-content digest), normalized
effective request options, API and extractor schema. Request content is excluded.
Unknown top-level overrides or missing required identity suppress observations;
there is no placeholder digest. Finalization revalidates the identity/key and
drops stale executions rather than relabeling them. Descriptor `Family` selects
the pinned llama.cpp or Strata schema. Strengthened non-Ollama identity and
native acceptance remain later gates owned by their respective units.

The existing SC SSE/NDJSON reader invokes a synchronous numeric extractor only
for validated genuine upstream JSON objects, including empty-choices usage
objects. Neither `[DONE]` nor synthetic errors are extractor input. The callback
does no IO, clock access, publication or external locking and retains no payload.
The reader stores copied typed numbers; independent evidence merges, repeated
counters never add, conflicts become unknown and Strata MCP disqualification
is sticky. Result snapshots copy nullable fields. SC alone classifies stream
completion. The same installation adapter supplies one reader per response at
local dispatch and keeps origin peer normalization free of numeric production.
On validated pinned peer hops it preserves an exact canonical interruption
terminator and bounded reason without passing synthetic metadata to extraction.
Existing media/encoding, integrity-header, allocation and cancellation rules
apply. Nonstream responses are unobserved; pure nonstream extraction remains
available without a production observation mode.

Monotonic elapsed times measure admission entry to grant, local `RoundTrip` to
headers, first genuine body byte (before any peek), and finalization. Unreached
milestones and `FirstTokenMs` remain nil. Daily adapters preserve native rate
and duration in `EngineDecodeTokensPerSecond` and `EngineDecodeNs`, leaving
legacy `DecodeNs` nil. Ollama keeps its original count/duration mapping.
Aggregate `Entry.outputTokens` (`Entry.OutputTokens *int64`, JSON
`outputTokens,omitempty`) sums independently known output counts only from
successful, SC-complete local executions retained in the entry. Count
qualification is independent of rate qualification: usage-only completion
contributes its count even when all decode rates are unknown. Native engine
decode intervals and rates do not redefine this visible-output total or imply
a count. If no qualifying count is known, the field is absent, not zero;
explicit zero counts remain measured zero. Failed, interrupted, cancelled and
protocol-unobserved executions do not contribute. Existing token and retained
sample bounds apply.
Qualified completion requires successful delivery and SC completion. Errors,
interruption or cancellation cannot qualify a terminal rate; unobserved
protocol completion remains unknown. These results do not release strict
capacity, change routing or alter workload reconciliation.
An unobserved response counts as completed only when that attempt's response
was accepted, its body reached upstream EOF and delivery had no write error.
Discarded attempts and body read errors remain failed.
Requester cancellation records `cancelled`. Expiry of the routing owner's
pre-output deadline records `failed`, preserving the deadline cancellation cause
on the dispatch context and the existing HTTP 503 response.

The bounded FIFO has capacity 128, shared across facades. Its single sender
performs codec IO outside the lock; producers never await codec writes or queue
space. Saturating counters cover full/sequence-exhausted, missing identity or
unsupported options, stale identity and invalid-sample drops. There is no
durable queue, resend or retained request content. Normal shutdown cancellation
stops the sender between writes or while waiting for the codec write gate.
The parent session interrupts an active write when its caller is cancelled or
its 60-second notification budget expires, then closes both transport
directions. Shutdown does not join the telemetry sender.

### Strict destination admission (A1a)

Strict requests carry exactly one shared-parser-validated
`X-PAIR-Strict-Admission` envelope. Both paired ingress and local self dispatch
use the existing process-wide admission controller, including its mutex,
in-flight counters, residency, notifications and cancellation registry. Engine
IDs are built-in names or valid descriptor engine IDs (`engines.ValidRoutingID`).
Model keys use the engine's facade naming: Ollama's omitted tag becomes
`:latest`, a descriptor engine follows its declared naming, and an engine with
no facade here keeps its model ID exact. Equivalent normalized capability entries reject.

Finite `maxConcurrent` and the matching authoritative model identity are
required. The capability is derived (local qualification, B, below) with the
finite effective policy/profile concurrency limit for the normalized
engine/model, which is also enforced for concrete traffic; an unlimited limit
installs nothing. Caller fields never establish destination capability. Strict
dispatch is terminal-local and never enters concrete route selection or retries.
Headerless concrete admission retains its prior semantics, including its switch
timeout behavior.

Queue and cold deadlines derive from remaining durations on a monotonic clock,
share the pre-output cap, and never reset on wake, switch or state changes.
Zero cold allowance forbids wake/unload and excludes unloaded models and models
with an unload pending authoritative residency acknowledgement. Tickets admitted
warm-only (including already-warm requests with nonzero cold allowance) revalidate
loaded residency and absence of pending unload before dispatch; failure returns
the typed cold-residency error and releases the undispatched slot. In-flight
models, including admitted/dispatched strict tickets, are excluded from eviction.
Recovery of a pending unload requires authoritative unload completion; timeout clearing is deliberately absent.
Once cold work is required, its original deadline also bounds subsequent capacity
or other waits and is checked before any ticket allocation. Budget expiry
rejects, including when capacity becomes available at the deadline. Context
cancellation removes queued waiters without allocating a slot. The same deadline
covers connection, headers and first response body; active generation after the
first byte is outside the selection timer. Typed errors support `errors.Is/As`:
unavailable/off and cold residency map to HTTP 503, occupied capacity to 429;
malformed/duplicate envelopes map to 400. A context deadline maps to 504.
The ticket deadline watcher uses the injected timer; `awaitFirstBody` also retains
its existing real-time timer, so first-body timing is not wholly clock-injected.

Tickets are allocated/released under the controller mutex. Before dispatch,
`BeforeStrictDispatch` revalidates identity/availability and durably writes a
version-1 JSON capacity document through `shared/durablefile`. The canonical
`appdir` file `proxy-strict-capacity.json` holds slots with numeric ticket ID,
engine and normalized model only. Startup restores slots into the same in-flight
counters. Slot IDs continue above recovered IDs. No request content is persisted.
A change to a key's installed entry or qualification, including withdrawal and
reinstallation, advances its per-key generation, invalidating undispatched
tickets even if the same identity is reinstalled; a rederivation that leaves the
entry unchanged keeps the generation. Dispatched work keeps its accounting. Installation, marker writing and recovery share one model
key contract: known engine, nonempty normalized key, no reserved alias or control
characters; internal spaces are supported. The writer and loader share a total
slot bound, enforced before adding a dispatch marker.

`Finish` is idempotent. Not-dispatched and qualified-complete evidence, or
authoritative-idle/verified-replacement evidence, release a ticket only after a
needed durable clear succeeds. Uncertain terminal paths retain the slot and its
marker. HTTP EOF supplies no qualified termination evidence in A1a and remains
uncertain. The destination-owned reconciliation seam clears recovered or
quarantined slots only on authoritative-idle/verified-replacement evidence; it
cannot remove a live HTTP ticket. Elapsed time does not clear capacity.

Failed dispatch persistence prevents inference. `ErrReplaced` means marker bytes
landed and are retained, but uncertain durability still prevents dispatch.
Pre-replacement failures retain quarantine in memory and are logged. Failed
clears retain capacity in memory. Corrupt/unreadable state disables strict
admission. Reconciliation returns the recovery error without writing while state
is untrusted, and skips persistence when no entries were removed. Marker loading
requires nonnull version and slots, positive unique slot IDs, built-in engine IDs
and valid normalized model keys. The complete
document is validated before any counters or markers are adopted; invalid later
slots cannot leave partially restored state. A1b/A2 must provide actual
authoritative reconciliation and the qualification owner must prove terminal
evidence before activation.

### Strict readiness and peer capability (A2)

After answering each successful `node/set-policy` with its `PolicyAck`, and each
successful `node/set-model-identity`, the proxy sends the broker
`admission/strict-readiness` (`nodepolicy.NotifyStrictReadiness`) with
`nodepolicy.StrictAdmissionReadiness`: the active revision, the proxy's `runId`,
and the installed capability, its models under their normalized keys sorted by
engine and model. The capability is omitted when nothing is installed or when the
strict capacity document is untrusted. Nothing is sent before a policy is active
or after a refused request. Reports leave from the control read loop only, so a
report computed from older state never follows a newer one. The broker owns the
fences and the publication (its README).

A peer's capability reaches routing from its pinned inventory document through
the scanner and the broker relay: `subscribedToNode` copies
`DirectoryNode.Policy.StrictAdmission` onto the internal `Node.StrictAdmission`,
which `nodeEqual` ignores, so readiness changes announce no `node/updated`.
`Node.strictModel(engine, model)` is the hybrid exclusion rule: a peer is a hybrid
local candidate for a model only when it advertises a valid capability of the
supported version naming that engine and the model as the engine's facade
normalizes it. A peer with no policy, no capability or an unsupported version is
excluded from hybrid candidates only. Concrete routing never consults it, so such
peers stay concrete candidates. The capability is a hint; the destination
revalidates every strict request. `hybridLocalCandidates(tier, needs,
crossEngine)` is the role eligibility built on it: of `resolveTierCandidates`'
candidates, in order, only those offered by their node's own declared role,
whose advertised role carries finite qualification
(`AdvertisedRole.HybridQualified`), and whose node has a strict entry for the
role's normalized model (`Node.strictModel`; this node's own installed
capability for itself). Router tier entries never make a node hybrid eligible.
The hybrid branch of `routingSpend.dispatch` calls this filter. The HTTP facade
supplies `concreteDispatchPolicy`; the unified listener supplies the hybrid
policy through the same HTTP request lifecycle (§5.1, §12.6).

### Local qualification and identity (B)

`node/set-model-identity` (`nodepolicy.SetModelIdentityParams`) replaces the
engine identity: per engine the run id, applied-settings fingerprint, launch
digest and model content digests. It is decoded strictly, validated, answered
`{"ok":true}`, and rederives the capability; an invalid one is refused and the
previous identity kept. The capability is derived from the active policy, that
identity and the qualification registry read once at start from the `appdir`
artifact `local-qualification.json` (absent: empty; invalid: empty and logged).
The accepted artifact schema integer is `2`: every entry requires non-null
`shapes` with boolean `tools`, an array `structuredOutput` of distinct
`json_object`/`json_schema` names, and boolean `reasoningReplay`. Replay is
currently unsupported, so `reasoningReplay` must be `false`; `true` rejects the
document. Missing/partial/null declarations, wrong types, unknown fields or modes,
repeated keys/modes and unsupported versions reject the whole artifact. Version-1
operator evidence needs explicit requalification against current identities,
defaults, tokenizer/template, shapes and measured bounds, not a version-number
edit or automatic migration. Shipped synthetic fixtures qualify no real target;
they do not establish that operators have no evidence.

A profile with `qualification {id, revision}` installs its normalized model only
when the registry entry exists for the same engine and model, its
`launchDigests` and `modelDigests` contain the identity's launch digest and the
model's digest (looked up under every name normalizing to it; disagreeing names
give none), the profile's `requestOptions` equal the qualified ones, its
`contextTokens` and `maxOutputTokens` do not exceed the qualified ones, and the
effective concurrency is finite. Capability coverage fails closed: `tools`
requires `shapes.tools:true`, `structured` requires both JSON modes, and vision,
embeddings or unknown capabilities refuse installation. Empty capabilities add
no shape requirement; one-mode evidence can serve that mode without advertising
the broad structured capability. Two profiles normalizing to one model install
neither. The installed identity is `{engineRunId, settingsFingerprint,
modelDigest}`. Coverage withdrawal uses the existing generation fence.

For an NE identity with `serving`, readiness also requires the normalized model's
valid, unambiguous content-bound envelope: qualification context/output must fit
its per-slot/output ceilings, effective concurrency must fit slots, and profile
memory must cover declared VRAM. RAM is a separate domain. A nil serving map
retains the non-NE inventory/digest rules. NE owns content/config/template identity:
`modelDigests` consumes its canonical `pair-serving-content-v1` manifest SHA-256
over ordered artifact role, size and trusted full-file SHA-256; `launchDigests`
consumes `pair-serving-launch-v1`, binding the pre-snapshot launch digest and
approved prepared digest, excluding random generation directories. Existing
artifact attestation provenance and explicit renewed verification/apply remain
required; B adds no provenance acceptance field or alternate identity store.
Changed content, launch inputs, declarations/envelopes or defaults invalidate
stale evidence. Inventory digest absence alone does not permanently exclude an
approved NE llama.cpp launch. See [identity and attestation](../../docs/local-qualification.mdx#ne-artifact-attestation).

`serveStrictLocal` calls `qualifyStrict` before `AdmitStrict`: the model must be
installed under the request's identity (else 503 `strict-admission`), and the
body is validated against the installed qualification (else 400
`strict-qualification`, naming fields and bounds only). Only the
`/v1/chat/completions` shape is supported: allowlisted top-level fields, text
content, function tools, `json_object`/`json_schema` formats, no repeated or
case-variant keys, nesting at most 64. The profile's `requestOptions` are merged
as defaults for absent fields. Incoming keys and every post-merge effective key
are checked; defaults cannot inject unknown keys or context overrides.
`reasoning_effort` and `parallel_tool_calls` are refused whenever present,
including null, false and defaults-injected values. The effective body is what
is accounted and forwarded, preserving accepted fields except the sole
validated no-op removal below. One finite output bound is required
(`max_tokens`/`max_completion_tokens`, equal if both) and may not exceed the
lower of the profile's and the qualification's ceilings. The input bound is
computed by the qualification's accounting method (`utf8-bytes`: request,
message, tool and schema overheads plus the UTF-8 bytes of role, name, tool call
id and text and of the compact tool call, tool, tool choice and schema JSON; the
qualified context is at most 2^24 tokens); string choices contribute decoded
UTF-8 bytes, object choices compact JSON bytes. `json_object` charges schema
overhead; `json_schema` also charges its name, description and compact schema.
Input plus output must fit the smaller profile/qualification context.
`AdmitStrict` takes the generation `qualifyStrict` validated against and refuses a model reinstalled
since. All of this happens before any wake, unload, slot or inference.

Present tools must be a non-null array; omitted tools and `tools:[]` have no
active definitions and zero definition overhead. Choice omitted/`none`/`auto`
permits either; `required` and named choices require nonempty definitions and
tools evidence. Named functions may omit `parameters`; supplied parameters must
be an object. Definitions, assistant calls and tool-role history require tools
evidence, including history with empty/omitted definitions. Assistant null content
still requires nonempty calls. Null tool arrays, calls, choices, supplied
parameters and formats, malformed members and new shapes are refused. JSON
modes are independent evidence entries; omission/text grants no
JSON mode and needs none. Nested decode failures retain generic diagnostics such
as `unsupported shape`, not a specialized policy denial for every malformed
member. Existing non-shape scalar, sampling, streaming and output rules remain.

PAIR accepts `store:false` as a no-op, removes it before forwarding, and applies no accounting charge; `store:true` is explicitly refused because PAIR does not store requests. Omitting `store` preserves existing behavior.

Both raw and merged store values are validated with normal defaults precedence;
null and non-boolean values are refused. This local exception does not add
`store` to the public-account adapter's allowlist. Pinned OpenCode 2.0.32 generic
provider lowering emits false by default; `compatibility.supportsStore:false`
was a pre-fix workaround, not a requirement after Task 7. The historical 2.0.32
and 2.1.1 inventories, immutable native-source links, configuration conditions and
synthetic examples are in [Pinned OpenCode compatibility](../../docs/local-qualification.mdx#pinned-opencode-compatibility);
they make no installed-2.1.4 claim.

Both inventoried releases automatically replay present assistant
`reasoning_content`, with no established client replay-off switch. Current local
validation refuses that member, including empty strings/null and every role;
replay is never silently stripped. Future assistant-only replay would require
approved exact NE-bound content/launch/template/reasoning-format evidence,
unchanged forwarding and UTF-8 accounting (empty string zero bytes) plus existing
message/template overheads. Schema `2` does not enable that future contract.
Unqualified local reasoning turns remain incompatible, and exact-template
qualification cannot alone bypass the current unsupported implementation.
The named `replay-with-tool-roundtrip-refused` and `empty-replay-refused` fixtures
record refusal; synthetic native streaming response forwarding is not real
provider/model or successful replay qualification. None of this activates provider
egress or waives the common HTTP entry's existing permission/acceptance gates.

Before forwarding to an engine, ingress and self dispatch strip every
`X-PAIR-*` header (strict, admission wait/result and routing model/engine/node/tier
metadata) and `X-NVPAIR-Engine-Identity-Probe`. Other engine headers are preserved.

Carries inference **to** the node that should serve it. One process hosts a
**facade** per enabled engine: each facade is an HTTP reverse proxy speaking its
engine's own dialect, filtering to nodes that advertise the requested model,
choosing among them, and failing over when one refuses.

It is a data plane, not a policy component. Ranking belongs to
`nvpair-job-scheduler`, and cluster membership and trust to
`nvpair-cluster-manager`. `nvpair-engine-manager` owns an engine's *install and
process* lifecycle; the per-request *workload* lifecycle events in §5.5 are this
process's own. What it owns beyond those events is the choice of *which
advertised owner receives this request*, and the local accounting that makes
that choice good under bursts.

### 1.1 What a caller is promised

The proxy aims to get every inference request **started** on some node that can
serve it, and spends a bounded amount of effort doing so. Stated precisely:

> While budget remains (§5.1), a failure that happens *before* the engine begins
> producing output is retried rather than returned. Once output begins, the
> request is committed to that node. Supported streams then finish with valid
> protocol completion or an explicit protocol error; no post-commit dispatch
> occurs in SC-a.

Failures absorbed that way include a retryable status, a transport error, an
engine that answers its headers and then produces nothing, and a target that
leaves discovery mid-request. Waiting for an owner to *exist* costs no attempt
at all, so a node blinking out does not spend the budget.

The following limits apply.

**Effort is bounded, and which bound ran out changes what the caller gets.**
Absorption holds only while budget remains. Past that there are two shapes, both
detailed in §5.4:

- **The dispatch budget ran out.** The final permitted attempt is never
  retried, so its own failure is the answer: the upstream's status and body
  under §5.2's stream-integrity rules, or a `502` if it failed at the transport
  and a `504` if it answered and produced nothing.
- **The deadline passed between attempts, or no owner came back.** Nothing is
  in flight to answer, so the proxy generates a `503` with `Retry-After` that
  names which of those it was.

Either way this is a persistent, bounded attempt to start — not an assurance
that every submission eventually runs.

**Commitment and successful completion are distinct.** The first body byte
commits the destination (§5.2), including a byte of an incomplete frame.
Supported SSE/NDJSON streams are then read in bounded volatile memory, with
complete frames preserved and torn tails withheld. `completed` requires valid
protocol completion and successful delivery. An unrecovered interruption is
`failed` with an operational reason even though HTTP status remains committed,
usually at 200. Client cancellation or failed delivery is `cancelled` (§5.4).

**Protocol coverage is bounded.** SC-a manages OpenAI chat/completions SSE and
native Ollama chat/generate NDJSON under §5.2's stream rules. Other protocols and
explicit non-streaming responses retain raw forwarding; a copy abort fails
their workload but receives no synthetic protocol repair. Qualified same-model
continuation, private attempt history and its UI belong to pending SC-b.

**A non-streaming request is the weakest case.** Its headers arrive only when
the response is complete, so "queued inside the engine" and "generating right
now" look identical from outside. The attempt cap therefore has to cover whole
generation, and when it expires the proxy cannot tell whether it abandoned a
stalled request or a working one — so retrying may discard real progress. That
is a genuine limitation, not a safety margin (§5.6).

## 2. Scope

**In scope**

- One listener per enabled engine, serving loopback plaintext for local clients
  and pin-gated cluster mTLS for peers on the same port.
- Model-owner filtering, target precedence, and retryable failover.
- Per-request workload lifecycle events.
- The process-wide reservation map that makes concurrent dispatch spread.
- The persisted per-engine port a user chose through `set-port`.
- The fork's node policy (§12): node-wide admission for work executed here,
  weak/medium/strong tier resolution, routing metadata headers, and workload
  cancel and regenerate.

**Out of scope**

- Deciding node rank (scheduler), membership or pinning (cluster-manager),
  engine install/start/stop (engine-manager).
- Any cryptography of its own: it consumes the trust fabric, never builds it.
- Model search or catalog browsing.

## 3. Process model

The process starts with **no engine and no listener**. The broker then sends one
`facade/enable` per engine, carrying that engine's port and any alias addresses,
and for a descriptor engine its descriptor (README, "Descriptor engines").

A flag cannot express this. The broker plans a different port for each engine —
Ollama's managed facade wants `:11434`, LM Studio's wants `:1234`, and
llama.cpp's managed facade wants `:8080`; any may be absent so the child keeps
its own persisted port — and a single-valued flag carries only one plan.

### 3.1 Why one process

Between scheduler snapshots a facade takes short-lived **reservations** for work
it has dispatched but that the scheduler has not yet observed. Those live in the
process, and every facade shares them. If each engine had its own proxy
process, none could see the others' reservations, so simultaneous bursts across
Ollama, LM Studio, and llama.cpp could select the same node, each incorrectly
believing it was idle.
Sharing the map is the reason the engines share a process.

### 3.2 What sharing costs

**Shared fate on the process.** A crash takes every facade down; the supervisor
restarts them together and the crash is reported once, against the process.

Inside the process the boundaries are finer, and deliberately so:

| Failure | Blast radius |
| --- | --- |
| Bind race lost on enable | That engine only; reported so the broker can retry elsewhere |
| Port cannot be re-listened | That engine only; its ownership gate is released |
| Panic handling a request | That request only; logged with stack, answered as an internal error |
| Process crash or exit | Every facade |

The first three failures are contained within their listed boundaries. Only a
process crash or exit takes down every facade. Containment matters here in a way
it would not for a single-engine process: one engine's bind race must not take
another engine's working listener with it.

## 4. Addressing

Every facade-scoped message carries the engine it concerns, in both directions.
Without it the engine would be implied by which process a message travelled
through, which is precisely the property one process destroys.

The address is a bare built-in engine id (`ollama:ready`), deliberately **not**
the component name (`ollama-proxy:ready`). A descriptor engine is addressed by
its service key, `e-<id>:ready`, because its user-chosen id could otherwise
spell a process-scoped namespace such as `errors`. The component form is how a client
addresses the component through the broker; the two meet in the broker's relay,
which takes one off and puts the other on.

| Class | Addressed? | Examples |
| --- | --- | --- |
| Facade-scoped | yes | `ready`, `error`, `node/*`, `errors:*`, `set-port`, `discovery:subscribe`, `discovery:nodes` |
| Process-scoped | no | `log/set-level`, `node/set-priority`, `node/set-policy`, `node/set-availability`, `node/set-engine-drain`, `node/set-residency`, `node/set-engine-intent`, `node/invalidate-engine`, `node/admission-snapshot`, `workload/cancel`, `workload:*`, `admission/*`, `discovery:node-activity` |
| Bootstrap | names its engine in the payload | `facade/enable` |

`facade/enable` is unaddressed because it runs *before* the facade it names
exists. Only a built-in engine id or a well-formed `e-<id>` counts as an address: plenty of unaddressed
methods contain a colon, so treating `errors:report` as engine `errors` would
strip a real method down to `report`.

The routing listener (§12.6) belongs to the process, so its `errors:report` and
`errors:clear` are sent unaddressed; the broker relays them to the error
pipeline like a facade's.

An unaddressed facade-scoped message resolves to **no** facade. Falling back to
"the only facade" would work with one engine enabled and misroute silently with
two.

## 5. Request path

Cluster peer dispatch uses `nvpair-shared/clustertrust.PeerServiceClient`,
addressed by the admitted peer UUID and the facade's fixed service identifier
(`ol`, `lm`, or `lc`). Its route resolves the current advertised endpoint and
generation, including runtime ports. Direct mTLS is preferred; bounded failure
before HTTP dispatch can select the existing authenticated reverse session.
The reverse byte stream retains inner mTLS to the receiving facade, its pin
checks and local admission policy, and its prohibition on onward routing.
Direct probe failure does not by itself exclude a pinned model owner from the
discovery overlay. Self and manual candidates keep their existing transports.

The adapter streams status, headers and ordered bodies with backpressure and
cancellation. It adds neither a listener nor whole-response buffering, and
does not replay dispatched requests or resume exposed response bodies. A lost
session after output has been exposed truncates that attempt; a restored
session serves new requests. Retry decisions before the proxy's commitment
point remain owned by the existing dispatch policy below.

For a model-bearing inference request:

1. Filter a request-local discovery snapshot to nodes whose per-engine inventory
   advertises the requested model. Ollama normalizes the implicit `:latest` tag;
   LM Studio and llama.cpp ids match exactly. An empty owner set returns a local
   `502` without contacting an engine.
2. Order the eligible owners: explicit `node/select` pin, then the scheduler's
   priority list, then deterministic default ordering.
3. Reserve the least estimated-loaded scheduler-listed candidate and move it to
   the front (§6).
4. Forward, failing over through the remaining candidates on a retryable status
   or transport error. An upstream model `404` counts as retryable, because
   positive inventory can be stale.
5. When a round's candidates are exhausted, back off and re-resolve (§5.1),
   until the dispatch budget or the deadline runs out.

An ineligible manual selection cannot override the capability gate, and failover
never broadens to an excluded node.

**Paused remote peers.** `resolveRouteCandidates(model, inference)` in `proxy.go`
owns the pause rule. `resolveCandidates` calls it with `inference` true and
`resolvePassthroughCandidates` with false. For an inference request, a remote
node is not added to the result set when its `Node.PausedBy` is non-empty. Any
non-empty value excludes, including an origin this build does not know. The hint
is advisory (§12.2): it decides routing eligibility only, never what the peer
will accept.

Ordering (explicit `node/select` pin, scheduler priority, stable ID fallback) is
computed over the full owner list. The exclusion is applied at the step that adds
each node to the result set, before host deduplication, not before ordering: an
excluded node does not claim its resolved host, so another record for that host
stays eligible. The rule covers a concrete model, a request without a `model`
member (which has no model-owner filter) and a tier request, because
`resolveTierCandidates` obtains candidates for applicable declared roles and
eligible local tier entries through `resolveCandidates`; API entries are
skipped. `hybridLocalCandidates` takes its candidates from
`resolveTierCandidates` and so inherits the rule, including inference entering
through the unified routing listener (§12.6).

The hint does not remove:

- Self, including a dormant self candidate (§12.2). Local admission judges it,
  and a paused self candidate is refused there with `503`.
- A manual node without a pause hint. A manually added node carries no policy
  advertisement, so it normally has none.
- Model lists, CORS preflights and other non-inference requests, which
  enumerate candidates through `resolvePassthroughCandidates` without the
  inference pause filter.

Each resolution reads the current discovery snapshot, so a hint takes effect on
the next resolution after the proxy holds it. A pause-hint update alone does not
alter already-resolved candidates; §5.1 re-resolves between rounds, and a local
policy generation change can also rebuild the current round. A hint-only change
emits no `node/updated`. The hint is a snapshot of the peer's saved policy, not
live state. Its scanner refresh has a nominal 15-second cadence plus other
refresh work, observing a pause or resume needs a successful fetch, and a failed
fetch keeps the last successful policy indefinitely, so no convergence time is
guaranteed. A peer fetched from a non-loopback address must be an admitted,
pinned cluster peer reached over HTTPS through the cluster transport; an unpinned
LAN peer is refused with no plaintext fallback. Loopback addresses use plaintext,
so a loopback fixture shows propagation, not authenticated fetching. The
[scanner README](../nvpair-node-scanner/README.md) is the owner of that fetch.

The destination wins over a stale hint. A peer that is paused while the proxy
still holds an empty hint for it receives the request and refuses it at admission
with `503`, `X-PAIR-Admission: paused` and `Retry-After: 30`. The router treats
the rejection as retryable while another dispatch is permitted. On the final
permitted attempt it relays the peer's `503`, even if other candidates remain. A hint that outlives a resume only keeps the node out of inference routing
until a later fetch replaces it. Management changes never consult the hint; the
target's own state decides them.

When hints exclude every remote candidate and no self candidate remains, a
concrete model request is answered with the local `502`
`{"error":"no available node advertises the requested model"}`, a request without
a `model` member with `502` `{"error":"no active node selected or available"}`,
and a tier request with `404` `no eligible model for tier <tier>` (§12.3). None
carries `X-PAIR-Admission` or `Retry-After`, and no engine is contacted.

The hint changes neither a node's advertised inventory nor its declared roles,
so the combined model list and the tier aliases keep their declared contents
while dispatch skips paused peers. This section makes no claim about whether a
paused node stays in or leaves any inventory; that follows ordinary discovery.

The llama.cpp facade exposes `GET /models` and `GET /v1/models` as aliases for
the fleet's llama.cpp inventory. Both query every candidate's `GET /models`
concurrently and merge duplicate model ids into an OpenAI list envelope
(`{"object":"list","data":[...]}`). A selected node affects candidate ordering,
not inventory scope. Its inference routes remain `/v1/*`.

### 5.1 Retry bounds

Concrete HTTP inference uses `concreteDispatchPolicy(start)` on the common
`routingSpend.dispatch` owner, with the bounds below. The same driver contains
a hybrid policy branch used by the unified listener (§12.6). Both enter through
`facade.serveInference` for context, cancellation, owned buffering, registration
and the response sink. The facade retains concrete and legacy bare-tier
behavior (§12.3), including registration before its early validation exits.

| Bound | Value | Governs |
| --- | --- | --- |
| `maxDispatchAttempts` | 5 | how many dispatches may fail |
| `jobDeadline` | 10 minutes from request creation | how long new attempts may keep being started, including while waiting for an owner to exist |
| `firstBodyTimeout` | `proxyResponseTimeout` (120s) | first-content wait in a round with alternatives, clipped to the remaining job deadline; a sole candidate gets the remaining deadline |
| `defaultStreamFrameIdle` | 60 seconds | after commit, wait for the next complete valid event on an SC-managed stream; independent of dispatch bounds |
| `idleClientWriteTimeout` | 30 seconds | a single blocked client write or flush; independent of upstream read idle |

The retry budget applies only to inference requests. Routes not classified in
the engine's route table are still forwarded verbatim, and some perform
state-changing operations; for example, `POST /api/pull` starts a model
download. Since any `5xx` is retryable regardless of route, re-resolving and
redispatching such a request could start the operation several times, which is
both non-obvious and wasteful. For a non-inference request the proxy therefore
resolves candidates once and tries each at most once. It does not re-resolve
candidates, back off between rounds, or apply the inference deadline.

**An attempt is consumed by a dispatch, never by a resolution.** Finding no
eligible owner sends nothing, so a job whose only owner briefly drops out does
not burn its budget while the node comes back. A `busy` admission refusal from
the round's sole candidate also costs no attempt: it queues again with the
existing backoff until capacity is available or the job deadline expires.
Other refusals and failures retain their dispatch accounting.

**Elapsed time per attempt is the sum of three budgets, not one.** They run in
sequence: the dial (`proxyDialTimeout`, 10s), then headers
(`proxyResponseTimeout`, 120s, enforced by the transport), then first content
(`firstBodyTimeout`, 120s, which starts only once headers arrive). First content
is also bounded by the remaining job deadline. If the round has a single
candidate, only the remaining job deadline bounds this first-content wait,
allowing long cache-miss prefill without resending it to the same owner.

The dispatch-count and job-deadline bounds are checked before every dispatch.
The deadline also bounds preparation, admission and waiting for the first
content; it does not cut off a response that has already committed. Committed
streams and client writes have their own idle limits.

Which bound ends a job depends on how its attempts fail, and all three cases
are real:

- **Fast failures** — a refused connection, a prompt non-busy `503` — cost about a
  second, so all five attempts fit well inside the deadline and the dispatch
  budget is what ends the job. This is the common case.
- **Slow failures** — an engine that accepts and then withholds everything —
  can reach the deadline with budget unspent. With one candidate, waiting for
  its first content uses the remaining deadline without a timeout redispatch.
- **No owner at all** costs no attempt, so the deadline is the only thing that
  can end the wait.

The last case is why the deadline cannot be dropped as redundant with the
attempt count, and the middle one is why five attempts is a ceiling rather than
a promise. **Do not remove either bound as unreachable.**

#### Hybrid policy on the common driver

`hybridDispatchPolicy(start, limits, rules)` uses `Limits.MaxAttempts` and
`start + PreOutputBudgetSeconds`, with total `QueueSeconds` and
`ColdStartSeconds` allowances across attempts. It does not multiply these by
the concrete five-attempt/ten-minute bounds. Preparation, admission, peer-slot
waits and first response compete for the same pre-output deadline. Wait charges
conservatively reduce both queue and cold allowances across attempts; they do
not restart at each destination.

Hybrid local/peer resolution filters `resolveTierCandidates` through
`hybridLocalCandidates`, preserving its policy rank/warmth order and eligible
declared roles. `preferLoaded:any` puts warmth before tier rank; otherwise rank
comes first. Reservations select within the leading rank/warmth class rather
than promoting a fallback. Router entries alone cannot qualify a node. Self
attempts qualify the body and enter `AdmitStrict`; peer attempts carry the
canonical request key, strict model identity and remaining budgets in
`X-PAIR-Strict-Admission`. Destination ingress revalidates them and stays
local-only; an origin different from the authenticated peer is refused with
403 before body buffering.

A hybrid peer-service slot waits for at most the smaller of remaining queue
allowance and half the remaining pre-output time (`peerSlotContext`). Zero
means immediate refusal when occupied; `ErrPeerSlotOccupied` consumes an
attempt. The slot allowance is reserved out of the destination's queue, cold
and pre-output budgets before sending, so origin and destination cannot each
spend the full allowance. Concrete peer-slot acquisition keeps its existing
blocking behavior, bounded by the request context.

The driver's public branch considers the requested tier's API entries in
configured order after the first local/peer round, sharing the request's
attempt ceiling and deadline. It requires remote permission and build
availability, then the per-candidate advisory guard and G's wrapper (§12.9).
Refused candidates consume no attempt and trigger no credential lookup,
durable reservation or send. In this build `PublicDispatchAvailable()` is
false, policy validation refuses remote/paid permissions and enabled
deployments, and no production public qualification source is installed.
The listener entry supplies no public activation or permission.

The first accepted successful raw response byte commits through the same
owner and sink (§5.2), stopping the pre-output watch. A committed stream is
governed by stream/read and client-write idle limits rather than the selection
deadline; it cannot fall back. Strict capacity quarantine, cumulative spend,
buffer ownership and both cancellation surfaces keep their existing contracts.

#### Waiting for an owner

There is no memory of nodes seen in the past: every resolution runs against the
live discovery snapshot, so "eligible owner" always means "advertising this
model right now". What differs is the position in the request's life.

- **No owner at admission** — nothing has been dispatched, so there is nothing
  to wait for being interrupted. Return `502` immediately (§5.4). A model
  nobody advertises will not become servable by waiting.
- **No owner after at least one dispatch** — an owner existed a moment ago and
  went away, so the proxy waits for one to reappear, re-resolving on the backoff
  schedule. This costs no attempt, so `jobDeadline` is the only thing that ends
  it: the deadline *is* the grace period for a node coming back.

The asymmetry is deliberate. Waiting is justified by evidence that a capable
owner existed within this request's lifetime, and nothing weaker.

Candidates are re-resolved every round, so a node that recovered or a model that
finished pulling becomes eligible mid-retry. The same node may be retried — with
one owner, all five attempts go to it.

A requester that has gone away ends the retry immediately: nobody is left to
receive the answer, so the remaining attempts belong to work someone is waiting
for. Its end is reported as `cancelled` and is never attributed to a missing
node.

### 5.2 The commit point

**Commit is the first raw byte of an accepted successful response body, not
the response headers or the first complete frame.** One request owner
(`routingSpend`) arbitrates acceptance, commitment, cancellation and attempts
across destinations. Its one `responseSink` owns client delivery, counting
genuine and synthetic output and recording protocol completion separately from
successful delivery. Neither the storage account nor the frame reader is a
second lifecycle owner. Synthetic heartbeats, local errors and SC-a endings do
not commit a destination.

Headers are not a valid commit signal: they may show only that the engine
accepted the request. Treating them as commitment would disable the header
timeout and prevent failover while the request remains queued inside the
engine. The first content byte is the first engine-independent evidence that
generation has begun.

The difference is engine-specific and large. Measured against LM Studio 0.4.x
with a model loaded at `--parallel 1`, probing while a long generation held the
single slot:

| Request | Headers | First body byte |
| --- | --- | --- |
| streaming | 12ms, on accept | 35.5s, when the slot freed |
| non-streaming | 38.9s, withheld until complete | with the headers |

Nothing arrived during that measured wait — no SSE keepalive or empty role
delta. The commit gate itself also accepts heartbeat bytes; it does not require
meaningful model output.

**Only the existence of the byte matters to the claim, never its value.** The
gate does not parse it. It is spliced back onto the front of the body before
the origin's frame reader is installed, so supported stream parsing does not
move the existing commit/retry boundary.

`EOF` with no bytes is accepted as an empty answer without retry. This is
defensive rather than expected
— an inference route answering `200` with an empty body is an engine
misbehaving — but treating it as a stall would spend the whole budget waiting
for content that is never coming and then fail a request the engine considered
answered. No raw byte exists to set the owner's byte-commit flag. An empty
declared stream then fails protocol validation without retrying; an empty
non-streaming response retains its raw transport behavior.

A first-content timeout before any accepted raw byte **never** commits,
including on the final permitted attempt, where it is terminal with `504`.
The owner arbitrates timeout and the raw read under the same mutex. If the raw
byte already won but peek publication or reader construction subsequently
fails, the accepted response ends through its sink; it cannot dispatch again
or replace the status with a new error response.

The gate is scoped to inference, and costs a non-streaming response nothing
when its headers and body arrive together. It remains engine-independent.
Request framing settings and peer metadata are validated before the raw gate;
response frame parsing and reader allocation happen after the byte claim. A
byte commits even if later parsing rejects it or allocation fails.

#### Managed streaming after commit (SC-a)

| Route | Managed response | Non-streaming override |
| --- | --- | --- |
| `/v1/chat/completions`, `/v1/completions` | `stream:true`, or response `text/event-stream` when the flag is omitted | Explicit `stream:false`; omission otherwise defaults false |
| Native Ollama `/api/chat`, `/api/generate` | `stream:true` or omitted flag (native default true); `application/json` or `application/x-ndjson` | Explicit `stream:false` |

The stream member must be a single boolean or null when present; duplicate
members return 400 before defaults or dispatch, including duplicate nulls.
Null has the same framing meaning as omission: OpenAI media selects management,
while native Ollama declares streaming by default. Validation does not
re-encode unrelated request bytes. SSE content type does not enable parsing for
Responses, Anthropic or other routes. A declared/default stream with an empty
body, wrong media type or malformed framing fails rather than becoming a raw
success. Management applies to committed successful HTTP responses; upstream
error statuses retain §5.4's handling.

Framing settings (route protocol, declared/disabled state and requested choice
count) are derived from the effective body dispatched by each attempt, after
model rewriting and profile defaults. Defaults that add `n:2` require both
choices at finish-reason-only EOF; defaults that set native `stream:false` keep
even a single JSON response larger than the stream frame limit on raw copying.
An existing null member remains unchanged and prevents a profile default from
replacing that member, as with any existing request member.

**Peer defaults and version compatibility.** A destination ingress can apply
its own profile defaults, which the origin cannot infer. The destination sets
`X-PAIR-Stream-Framing` from the effective body it actually sends to its local
engine. The single header value is `v1;<mode>;<choices>`: mode is `declared`,
`disabled` or `auto` (the last is valid only for OpenAI routes); choices is a
positive canonical decimal integer, defaulting to 1. Protocol comes from the
unchanged route. Both concrete and strict paired ingress use this contract;
the header grants no engine-termination evidence or admission release.
Engine-supplied header values are overwritten, and unsupported/non-inference
routes never gain metadata. Local strict dispatch never emits peer metadata.

On a pinned peer's successful supported inference response, the origin validates
this metadata before the first-body gate and commit claim, then uses those
settings for framing. Missing metadata returns the typed attempt error
`peer framing metadata missing (version-incompatible peer)`; malformed,
duplicate, unknown-version or invalid values return
`peer framing metadata invalid (version-incompatible peer)`. No peer body
bytes or started notification are exposed for that attempt. Existing pre-commit
failover applies, and exhaustion returns the existing explicit upstream error
with the reason recorded in `proxy/request` and the failed workload. There is
no legacy inference branch: older peers must be upgraded for supported
inference routes. Non-2xx/admission responses retain existing status handling.
The origin strips this internal header from every client response and trailer,
including non-peer engine responses; direct engines cannot supply authoritative
framing metadata. Unsupported routes and non-inference forwarding are otherwise
unaffected.

The pinned destination also normalizes supported response framing and publishes
an EOF disposition on the same internal `X-PAIR-Stream-Framing` trailer channel:
`v1;completed` or `v1;interrupted;<reason>`, with a reason from the closed set
below. This trailer is separate from the pre-body `v1;<mode>;<choices>` header.
The origin requires one valid disposition for a normalized peer response and
strips it before client delivery, including for supported non-streaming
responses. Missing, duplicate, malformed or unknown-version dispositions do
not establish completion; missing disposition is a peer interruption, while
malformed or inconsistent stream metadata can be `malformed upstream stream`.
A destination's synthetic error and `[DONE]`
cannot turn an interruption into a completed workload at the origin. Engines
cannot supply trusted disposition: ingress removes their values and publishes
its own after body close at EOF. The trailer supplies no engine-termination or
strict-admission release evidence.

Origin and destination dispatch explicitly set `Accept-Encoding: identity` on
the supported inference routes, even when the client requests compression.
The explicit value prevents Go's transport from negotiating gzip itself. A
managed response with non-identity Content-Encoding still fails as malformed.

The reader preserves complete raw frames and their delimiters across arbitrary
read splits, including CRLF, multiline SSE and UTF-8. It withholds an incomplete
tail. OpenAI `[DONE]` and Ollama `done:true` definitively terminate a stream.
Finish-reason-only OpenAI completion requires every requested choice to finish
and clean EOF. A valid final NDJSON object without a newline is accepted at clean
EOF. An upstream protocol error is a failure, including one followed by a normal
terminal marker.

The 60-second complete-event idle interval starts after commit and refreshes on
complete valid events, including SSE heartbeat comments. Partial byte drips do
not refresh it. It is separate from the first-body, dispatch and client-write
bounds. Each client write/flush retains its 30-second deadline, refreshed for
each operation; it bounds a stuck client, not time spent waiting for generation.
Frames are held in volatile memory only: at most 1 MiB per raw frame and
64 MiB of live frame-buffer allocations process-wide across facades, reserved
before allocation/growth. Exhaustion terminates explicitly rather than waiting
for memory. The category is subordinate to the inference aggregate budget.
Before dispatch, an effectively enabled managed stream reserves separate response
working grants for metadata, choice tracking, read scratch, its parser,
initial raw frame storage and, for SSE, multiline assembly. Local numeric
streams reserve a 96 KiB parser; local syntax-only and peer streams reserve
16 KiB. OpenAI `/v1` routes default to non-streaming when `stream` is omitted
or null; native Ollama NDJSON routes default to streaming. An unexpected
auto-SSE response to an omitted-flag OpenAI request acquires fresh reader
grants as at base and is not protected against a prior capacity decrease.
Local requests also retain their existing fixed and fingerprint
preparation, including non-streaming requests. Peers reserve no fingerprint
storage, and peer non-streaming requests reserve no preparation grants.
These grants do not share the request parser or performance fingerprint charge.
Each pre-dispatch aggregate acquisition may wait up to the configured I8
`waitSeconds` (zero for immediate refusal). Each acquisition joins the FIFO
tail with a fresh deadline and holds earlier grants while queued; a full
stream category refuses at once. Exhausted capacity returns 503, while a request or
choice count that cannot fit the applicable limit returns 413, including an
impossible OpenAI choice count on a non-streaming request. Native NDJSON routes
ignore `n`. Reader growth does not wait.
The response reader claims the admitted grants without another charge even
if aggregate capacity decreases before the first output. Live readers retain
their working allocations between frames, so an admitted stream can drain
through its original terminal frame while usage exceeds a reduced aggregate
cap. New admissions and larger replacement allocations still require capacity;
growth charges both arrays during copying. Unclaimed preparation grants and
reader-owned grants are released by their respective owners. Drain requires
subsequent raw frames, joined SSE payloads and choice tracking to fit the
reader's existing grants or retained high-water storage. Larger replacement
storage still needs capacity and may interrupt the stream after a decrease.
A pre-output retry is a new dispatch admission: it releases the failed attempt's
preparation and must satisfy current policy and capacity, including a changed
policy generation. Same-attempt drain does not authorize that retry.
A peer request sent with `stream:false` has no prepaid frame grants even if
the destination profile enables streaming, so a decrease before reader
construction can interrupt that response as at base.

The following byte-accounting ceilings are measured with one-choice SSE,
minimal `model`/`stream` bodies, their retained body/model bytes, a 96 KiB
request parser and fully constructed live readers. They exclude rewrites and
live registry metadata, routing snapshots and other concurrent allocations.
Registry metadata competes for the same root capacity; admission limits can
also bind first. These isolated preparation ceilings are not throughput guarantees.

| Preparation class | 192 MiB | 256 MiB | 1024 MiB |
| --- | ---: | ---: | ---: |
| Self numeric stream | 564 | 564 | 564 |
| Self syntax stream | 754 | 1005 | 1820 |
| Self non-stream | 874 | 1166 | 4665 |
| Self omitted-flag OpenAI non-stream | 875 | 1167 | 4671 |
| Peer stream | 1489 | 1820 | 1820 |
| Peer non-stream | 2047 | 2729 | 10919 |

The omitted-flag row uses `{"model":"m"}`; the explicit non-stream row
includes `"stream":false`, so its retained request and fingerprint are larger.
Each ceiling is the smaller of the aggregate capacity divided by the request
charge and the existing 64 MiB stream category divided by the frame charge
(the latter is absent for non-streams). The 564 self numeric bound equals the
base live-reader ceiling: admission now pays that existing charge before
dispatch. Metadata-only preparation counts do not include reader storage.
The stream category does not scale with I8 capacity. No whole-response prefix
is retained. Real slow-model acceptance of the idle default remains pending.

On premature EOF, read failure, idle expiry, malformed framing or a memory bound,
the old body closes and the torn tail is discarded. OpenAI receives
`data: {"error":{"message":"stream interrupted: <reason>","type":"server_error","code":"stream_interrupted"}}`
as an SSE event, then a `[DONE]` event; Ollama receives
`{"error":"stream interrupted: <reason>"}` as a newline-terminated object.
Reasons are a closed set of operational categories, never copied engine error
text:

- `upstream closed before the stream completed`
- `upstream read failed`
- `upstream stopped sending events`
- `upstream frame exceeded the frame limit`
- `stream frame buffer exhausted`
- `malformed upstream stream`
- `upstream reported an error`
- `peer response interrupted`

An already forwarded upstream error is retained without another error frame.
Failure never fabricates a successful finish reason or `done:true`.

The managed body returns normal EOF so ReverseProxy finishes HTTP framing.
Before forwarding headers the origin removes Content-Length and body integrity
metadata invalidated by possible error extension, leaving transfer framing to
net/http. It preserves endpoint content type, concrete routing headers and
engine-owned CORS. HTTP status is already committed and is never rewritten to
502 for a later protocol failure.

**After commit there is no retry or continuation in SC-a.** A peer or engine
loss mid-stream ends a managed OpenAI stream with an error frame plus `[DONE]`,
not a pause, until SC-b; managed Ollama streams use the error object above.
Delivery remains best effort if the client has disconnected or writes fail.
The sink records continuation as unavailable. SC-b's qualified same-model
continuation remains pending and must use this request owner and sink rather
than introduce another dispatch or delivery owner. Destination ingress remains
local-only, with its own local request/attempt arbitration and response sink,
but no origin registry key or onward route; parsing a terminal marker
does not establish engine termination or release an uncertain strict hold.

### 5.3 Per-attempt cancellation

A watcher cancels an attempt promptly when its target leaves discovery.
Otherwise the attempt could stay blocked until the first-content timeout while
the broker's node-loss sweep had already marked the workload failed, so the UI
would report failure while the client was still waiting.

Each attempt runs on a context derived from the request's, and that child
relationship is load-bearing: the disconnect watcher and the outcome
classification both read the *parent*, so cancelling a child cannot be mistaken
for the requester leaving. Confusing the two would either report a job that went
on to succeed as `cancelled`, or let a real disconnect keep burning attempts.

**The request mutex decides commitment versus pre-output abandonment.** The
raw read, discovery watcher, deadline and regenerate operation all transition
the same active attempt under the request owner's mutex. An abandoned,
cancelled, replaced or ended attempt cannot accept a late byte. If raw
commitment wins first, discovery disappearance cannot abandon it into another
dispatch; a later transport interruption ends through the same sink. If the
watcher wins before commitment, the attempt is abandoned and ordinary
pre-output retry bounds apply. There is no separate compare-and-swap owner.

Whole-request cancellation remains possible after commitment. A disconnect,
failed client write or qualified `workload/cancel` ends the request and its
active child; it never starts a post-commit attempt. Cancellation is recorded
under the request mutex before callbacks and I/O run outside it. SC-a ending
planning is arbitrated against the same owner; synthetic output cannot revive
an ended request. The complete workload cancellation key is in §12.5.

### 5.4 Status handling and outcome classification

Whether a status is retried depends on *when* it arrives, not only on its class.
Before commit the proxy may try elsewhere; at commit the status becomes the
client's answer.

| Upstream status | Pre-commit, budget remaining | Workload state if it is the answer |
| --- | --- | --- |
| `2xx` | commits (§5.2) | `completed` only after managed protocol completion or successful raw copy; interruption is `failed` |
| `3xx` | not retried; not expected from an engine, and forwarded verbatim if one sends it | `failed` |
| `4xx` except `404`/`408`/`429` | not retried — the same request fails identically everywhere, so retrying only spends budget | `failed` |
| `404` | retried on an inference route only: positive model inventory can be stale | `failed` |
| `408`, `429` | retried | `failed` |
| `5xx` | retried | `failed` |

Anything not retried commits and becomes the client's answer immediately, with
no further dispatch, whatever budget is left.

These are the statuses the proxy itself returns:

| Condition | Status |
| --- | --- |
| No owner advertises the model, nothing dispatched | `502` |
| Final permitted attempt returned a status | that status; body follows §5.2's managed/raw rules |
| Final permitted attempt failed at the transport | `502` |
| Final permitted attempt answered but produced no content | `504` |
| The loop ended between attempts, nothing in flight | `503` with `Retry-After` |

The final permitted attempt is never retried, so it answers for itself. That is
why exhausting the dispatch budget does not produce a `503`: the fifth attempt
is the answer. The `503` is for a loop that ended with no attempt to answer,
which means time ran out.

Its body names which, because the two call for different responses from the
caller:

- **the deadline passed while attempts were still failing** — an owner was
  present throughout, and slow attempts reached 10 minutes before they reached
  five dispatches (§5.1);
- **no owner became available** — the candidate set was empty and the wait for
  a node to come back timed out.

A third reason, **every dispatch attempt failed**, guards the case where the
loop ends on the count with nothing having answered. The paths above make that
unreachable today; it is cheap to keep and wrong to rely on.

The resulting workload state:

| Outcome | State |
| --- | --- |
| committed 2xx, managed stream reaches protocol completion and delivery succeeds | `completed` |
| committed 2xx, managed stream interrupted or upstream reports a protocol error | `failed`, with operational node/reason |
| committed 2xx, unmanaged raw copy completes | `completed`; protocol completeness is not inspected |
| committed 2xx, unmanaged inference copy aborts | `failed`; no synthetic protocol repair |
| committed non-2xx | `failed` |
| client disconnected, or dead-client write deadline tripped | `cancelled` |
| cancelled by our own shutdown | `cancelled` |
| budget or deadline exhausted pre-commit | `failed` |

`cancelled` distinguishes requester-initiated termination from an actionable
engine, node, or routing failure. Client disconnects and shutdown cancellation
therefore do not appear in the failed bucket.

**Reporting is deferred.** A managed stream reports its explicit result to the
finalizer separately from the successful HTTP body copy: clean EOF after a
synthetic error must not become workload success. Failures use the existing
`workload:errored` payload with state `failed` and
`error = "stream interrupted after output began on <node>: <reason>"`.
`proxy/request.Error` also carries the sanitized operational reason. Neither
the protocol nor the workload schema gains a terminal state.

An unmanaged mid-copy error is not returned to the handler:
`httputil.ReverseProxy` converts it into `panic(http.ErrAbortHandler)` under a
real server. Deferred reporting records failure and re-panics so `http.Server`
closes that response. Client write/flush failure and parent cancellation take
precedence as `cancelled`, including when an upstream failure races them.
Managed idle cancellation is private to the reader and cannot cancel the parent
request into a false `cancelled` result. Reservation and ticket releases remain
deferred under their existing certainty rules.

That panic is not the failure mode §3.2 describes, despite the shared word.
`http.ErrAbortHandler` is a sentinel `net/http` defines to mean "abandon this
response without logging a stack", and `http.Server` recovers it silently by
contract; re-panicking is how the connection gets closed, and swallowing it
would leave the client waiting on a response that will never be finished.
§3.2's entry covers an *unexpected* panic — a bug — which is logged with its
stack and answered as an internal error.

### 5.5 Workload lifecycle

| Event | When | State | `scheduledOn` |
| --- | --- | --- | --- |
| `workload:submitted` | admitted | `queued` | empty |
| `workload:submitted` | each dispatch | `queued` | the target |
| `workload:submitted` | between attempts | `queued` | empty |
| `workload:started` | commit | `running` | the node that served |
| `workload:completed` / `workload:errored` | terminal | per §5.4 | unchanged |
| `workload:errored` synthesized by the broker | proxy process exit (crash or replacement) | `failed` | unchanged |

On process exit, the proxy's own `cancelled` terminals come only from a graceful
stop. Unread tail frames can be lost when stdout closes, so even a request that
actually drained can appear `failed`. A delivered `completed` or `cancelled`
tail may still be accepted after process exit until the exact source handle is
withdrawn; once accepted, that authoritative outcome is preserved. The broker
retires open or inferred local-origin outcomes of a departed run as authoritative
`failed` with error `proxy exited before the request finished`, excluding
authoritative terminals, the live replacement run and foreign-origin workloads.
Every synthesized terminal sets `seq` to the current stored sequence plus one
and `completedAt` to the broker's retirement time.
For an inferred outcome, retirement replaces its prior `error` and `completedAt`
with that fixed error and the broker's retirement time, preserving other identity
and placement facts. The best-effort emission join is capped at the smaller of
`workerSignalGrace` and the remaining shared teardown budget; an unarmed teardown
clock still allows `workerSignalGrace`. Fanout need not finish before that wait
expires, and the cap does not bound an underlying client write. Retirement
supplies no engine-idle evidence and does not trigger residency changes.

`submitted` covers the whole pre-commit life; `started` means the engine is
producing content. That split is forced as well as honest: the broker's store
merges by state rank and rejects a lower one, so a job that claimed `running` at
dispatch time could not return to `queued` for its next attempt and the retry
would be invisible. Keeping the loop inside one state, with `scheduledOn`
carrying the detail, is the only shape that store can represent.

The proxy emits at most one terminal per request, including disconnect/copy-unwind
races; this does not guarantee delivery of its terminal frame. An interrupted
stream cannot later resurrect as `completed`. Its committed placement and
workload identity are retained, with the bounded reason exposed
through the existing failure display. The private continuation timeline and
placement changes planned for SC-b are not shipped here.

Every event carries `seq`, the producer's event counter from 1. The
workload-manager's inter-node dedup is a permanent set, so a retry returning to
a placement it already used — `queued` on A, cleared, `queued` on A again, which
this loop produces routinely — would otherwise be indistinguishable from a
redelivery and dropped by every peer.

The broker forwards all proxy lifecycle frames only after its store accepts
them. For an authoritative queued workload of the same generation and placement,
changing only model metadata (even with a new `seq`) is a store no-op and is not
forwarded to peers; peer broker stores already reject the same update.

A job admitted but not yet dispatched has no execution node. Consumers must
treat an absent `scheduledOn` as "not placed" rather than assuming a node.

**Model fields.** `model` is the concrete model the job asks for. For a tier
request (§12.3) it is the model of the candidate being tried, updated on each
dispatch and fixed at commit, and the optional, additive `requestedModel`
carries the tier (`weak`, `medium`, `strong`). `requestedModel` is absent when
the client named a model. `engine` stays the facade the request entered on even
when a tier candidate is served by another engine, because `engine` is part of
the workload's store and dedup identity and changing it mid-life would fork the
record; the engine that answered is in the `X-PAIR-Engine` response header.

Inference without a `model` member still has a routing request identity but emits
no workload lifecycle events. The broker's workload projection and the
workload-manager's lifecycle validation require a nonempty concrete `model`.
The routing key identifies the request but does not supply the concrete model.

### 5.6 Coverage gap: non-streaming requests

A non-streaming request is the case the retry policy serves worst, and the
weakness is the opposite of too little retrying.

Its headers arrive only when the response is complete, so from outside the
engine "queued behind other work" and "generating right now" are
indistinguishable. The 120s header cap therefore has to cover whole generation
— the first-content wait adds nothing here, since the body arrives with the
headers — and when it expires the proxy retries without being able to tell
whether it just abandoned a stalled request or a working one. A generation
legitimately slower than that cap is restarted elsewhere, discarding real
progress, and the restart faces the same cap.

Nothing in the current engine surface fixes this: there is no progress signal
before the response is complete. A caller that cares should prefer a streaming
request, where the first content byte gives the proxy the evidence it lacks
here.

SC-a does not buffer a whole non-streaming response or recover it after commit.
Explicit `stream:false` and unsupported protocols keep raw copying and lack the
managed complete-event idle bound and protocol-completion check. A transport
abort is still reported as failed; a clean raw EOF cannot prove semantic
completion for those protocols.

### 5.7 Accepted behavior

**An abandoned attempt may still run inside the engine.** Neither engine PAIR
supports today exposes a way to withdraw a request it has accepted, and both
queue internally, so a retried job can have more than one copy generating while
the abandoned copy keeps its node busy.

Whether closing the connection cancels anything upstream is **unverified** —
plausible, engine-specific, and never measured. The behavior above is stated as
the pessimistic case for that reason.

SC-a closes an interrupted stream's body and bounds its own reader lifetime;
this is not proof that the engine stopped computing. Destination strict holds
therefore retain their existing uncertain-termination handling. Neither a
synthetic error nor a parsed model end marker creates new release authority.

This is accepted rather than overlooked: those internal queues are invisible and
un-retargetable, which is the problem the retry policy works around rather than
a mechanism it can build on. Do not reach for engine environment bounds or
occupancy probes to "fix" it — LM Studio exposes no such signal at all, and an
adopted engine keeps its own environment regardless. An engine that did offer a
real cancellation API would change this calculus, and the claim should be
revisited per engine rather than assumed to hold forever.

### 5.8 Browser policy

Each facade preserves its engine's CORS policy, including missing headers and
permission denials. Ingress gates apply before preflight forwarding. Proxy errors
grant no CORS permissions. Multi-target preflights intersect permissions from all
responding engines without forwarding caller credentials; unavailable engines
are skipped, and no responders yields 502. Preflights take no reservations.

Model inventories with an Origin require agreement from every responding engine:
a denial yields 403, an invalid inventory yields 502 without a partial list. An
invalid-inventory error retains CORS headers only when all responding engines
approve. See README.md for the complete forwarding and intersection rules.

## 6. Reservations

A reservation is one in-flight dispatch this process has made since the last
scheduler snapshot. Estimated load for a node is
`pending + gpuPressure + reservations`, the first two from the scheduler.

- **Taken** when a candidate is chosen, unless an eligible manual pin applies,
  and re-taken through the same path on each retry round (§5.1).
- **Moved** to the node an attempt actually targets, because the node that
  refused the request is not doing the work. An attempt can hold a node for the
  whole first-content budget, so the claim follows each dispatch rather than
  waiting for the commit.
- **Released** when the request ends, so a node stops counting as loaded as soon
  as it stops working. Without release a trickle of short requests makes an idle
  node read as busy for the rest of the snapshot interval.
- **Released before every retry backoff**, for the same reason applied to a
  waiting job: between attempts nothing is running, and because the map is
  process-wide a stale claim would push the *other* engine's traffic away from a
  node that is in fact free. This mirrors clearing `scheduledOn`, one being the
  proxy's own estimate and the other the scheduler's.

Each is stamped with the snapshot **generation** it was counted against. A
snapshot supersedes every reservation taken before it — its pending counts
already include that work — so a release naming an older generation is dropped.
Applying it would double-count the completion and drive the node's estimated
load below zero, which reads as permanently idle and attracts every subsequent
dispatch.

`node/set-priority` therefore carries a generation and is applied at most once;
a redelivered snapshot must not clear live reservations.

## 7. Listener personalities

One port per facade, demultiplexed on the connection's first byte:

- **Loopback plaintext** for local clients. A LAN caller is refused.
- **Cluster mTLS** when `--cluster-dir` shows this node is a member: a peer
  whose client certificate matches a local pin is forwarded straight to the
  local engine reported by `node/set-local-backend`, never re-routed onward.
  Inference is admitted first (§12.2).

Separately from the facades, the process may hold one **routing listener**
(§12.6): loopback only, no cluster mTLS personality, owned by the process rather
than an engine.

Membership and pins are re-derived per request and on a watch, so joining or
leaving a cluster needs no restart.

An engine with an inherited host variable (today Ollama alone) may also be given
loopback-only **alias** addresses, so clients already using that variable enter
the same routing path. An occupied alias is non-fatal: the existing owner is
untouched, the primary listener stays up, and a warning is reported.

## 8. Ports

| | Ollama | LM Studio | llama.cpp |
| --- | --- | --- | --- |
| Engine's own client-facing port | 11434 | 1234 | 8080 |
| Where PAIR relocates the engine | 11435 | 1235 | 8081 |
| Standalone port, when `port` is omitted | 11435 | 1234 | 8080 |
| Persisted-port file | `proxy-port.json` | `lmstudio-proxy-port.json` | `llamacpp-proxy-port.json` |

A port chosen at runtime via `set-port` is persisted per engine and restored
when that facade is enabled, taking precedence over the requested port, so the
facade returns where the user left it. `ignorePersistedPort` bypasses that for a
broker-coordinated start.

Port persistence uses `durablefile`: failure before replacement leaves the old
listener and file intact; `ErrReplaced` keeps the new listener and saved port,
emits `ready`, and reports the durability error rather than reverting memory.

`port` 0 means "the engine's standalone default". Any other out-of-range value
is rejected rather than honoured as ephemeral: the facade announces the
requested port in `ready`, so binding an ephemeral one would leave the broker
unable to locate it.

## 9. Logging constraints

Never log prompts, chat messages, request bodies, stream chunks, full responses,
credentials, or pairing data. Operational metadata only — engine, model, job id,
node id, path, stream flag, normalized error text — and keep per-request routing
detail below info level.

Bounded volatile reads for first-body gating and SC-a frame validation are
permitted (§5.2); persistence and logging of that content are prohibited. Frame
buffers are released when their stream ends. Upstream protocol error text may
remain in its forwarded frame but never enters logs or workload/request failure
metadata; those use bounded operational reasons.

The log component is the **process** (`nvpair-proxy`), because at process init
there is no engine to name. Facade-scoped records carry an `engine` field, which
is what keeps a line attributable when several facades share the sink.

## 10. Adding an engine

Everything engine-specific is one entry in `engines.go` plus the shared identity
in `nvpair-shared/engines`. A new engine needs: its identity and discovery
service key, its ports and persisted-port filename, its route table and model
normalization, and its empty model-list envelope. It needs no new process, no
new supervisor, and no new relay wiring.

## 11. Failure modes

| Mode | Behavior |
| --- | --- |
| Requested port taken at enable | Tagged bind failure; broker retries on a fallback port |
| No advertised owner for the model | Local `502`, no engine contacted |
| All owners refuse retryably | Last upstream status surfaced |
| Transport error with candidates left | Forget the node's confirmed address, fail over |
| Supported stream cuts off after commit | Discard torn tail, send protocol error and clean HTTP EOF; committed status unchanged, workload `failed` with node/reason; no redispatch |
| Supported stream stops completing events for 60 seconds | Close upstream, send protocol error, workload `failed`; partial drips do not extend the interval |
| Declared/default stream empty, wrong media type, malformed, over frame limit or frame budget | Protocol error and workload `failed`, with bounded operational reason |
| Upstream sends a protocol error | Preserve the error frame without a duplicate error; workload `failed` |
| Unmanaged inference response copy aborts | Workload `failed`; HTTP connection aborts without protocol repair |
| Client disconnects mid-stream | Terminal workload event emitted at once; upstream cancelled |
| Broker link closed | Facades stop serving, then the shared transport pool closes |
| Admission refuses work here | `503` with `X-PAIR-Admission` and `Retry-After`; a router fails over (§12.2) |
| Tier with no eligible model | `404` `{"error":"no eligible model for tier <tier>","needs":[...]}` (§12.3) |
| Invalid node policy request | JSON-RPC error; the previous state is kept (§12.1) |

## 12. Node policy (fork)

The fork's node policy is defined in `nvpair-shared/nodepolicy` (`policy.go`
for the schema, `wire.go` for the methods and payloads) and described in
`FORK_DESIGN.md` §3–§4 at the repository root. The broker owns the policy file;
this process enforces it. Where this section and the Go source disagree, the
source wins.

### 12.1 Broker → proxy requests

All are process-scoped (§4) and answered `{"ok":true}` or with a JSON-RPC error,
except the typed acknowledgements and snapshot described below. Input is
validated before anything is applied, and an invalid request leaves the previous
state in force. The broker re-sends policy state whenever the proxy starts; until it does, the proxy runs on
`nodepolicy.Default()` with availability `available` and no active revision.

| Method | Params | Effect |
| --- | --- | --- |
| `node/set-policy` | private `{originNodeId, policy}` | the policy is validated with `nodepolicy.Parse`; `originNodeId` binds the process's origin identity; the policy's `availability` field is ignored; its `revision` becomes the active revision, and a revision older than the active one is refused with the previous policy kept |
| `node/set-availability` | `{state, cancelActive?}` | `available`, `draining` or `paused`; `cancelActive` cancels every admitted execution on this node; it never changes the active revision (§12.1.1) |
| `node/set-engine-drain` | `{engine, drain}` | refuse new work for one engine during a settings restart |
| `node/set-residency` | `{loadedByEngine}` | the models loaded per engine on this node, normalized per engine |
| `node/set-engine-intent` | `{enabledByEngine}` | each managed engine's saved On/Off intent |
| `node/invalidate-engine` | `{engine, operationId, reason:"engine-switch"}` | answers `{ok:true, runId}`; withdraws engine strict eligibility and fences held tickets (§12.1.2) |
| `node/admission-snapshot` | `{}` | answers `{runId, active, activeByEngine, queued, lastActivityMs}` from current admission state (§12.1.2) |
| `node/renew-egress-lease` | `{runId, revision, sequence, notAfterNanos}` | trusted parent only; answers `{runId, revision, sequence, nowNanos}`; zero deadline samples the receiver clock without changing authority; a nonzero deadline is a run/revision-bound grant (§12.9) |
| `node/revoke-egress-lease` | `{runId, sequence}` | trusted parent only; answers `{runId, sequence}`; a higher sequence clears the current lease without activating policy (§12.9) |
| `workload/cancel` | `{originatedFrom, runId, engine, workloadId, regenerate?}` | §12.5; answers `{found}` |
| `routing/cancel` | `{request: {originNodeId, proxyRunId, requestId}, regenerate}` | §12.5.1; answers `{status}`; private parent IPC |
| `routing/active` | `{}` | §12.5.1; answers `{requests: RoutingRequestKey[]}`; private parent IPC |

Unknown fields and engine IDs that are not `engines.ValidRoutingID` are errors. A
valid descriptor engine ID with no facade here is accepted state.
Invalidation accepts engine ID syntax independently of registry membership.

#### 12.1.1 Activation acknowledgement

The broker persists a policy under a revision (a canonical decimal string) and
asks the proxy to activate it. The proxy holds the active revision and the
policy it names as one value, so a reader of the admission controller's policy
never sees a revision without its policy, or the reverse. It starts with no
revision, older than any.

`node/set-policy` is the only request that activates a revision. It answers
`{originNodeId, revision, runId}`: the bound origin identity, the revision applied,
and `runId`, this process's run nonce
(the one stamped on workloads). A revision lower than the active one is out of
order: the request is refused with `-32602` and the previous policy and
revision stay. The same revision again is an idempotent replay.

A persisted availability change also takes a revision. The broker sends
`node/set-availability` first, so admission stops before anything else, and then
`node/set-policy` with the current document, whose revision is the availability
change's; `node/set-availability` itself carries no revision and does not move
the active one. That second request is also what re-delivers a policy whose
earlier push was refused. During policy recovery, where nothing is saved, only
`node/set-availability` is sent.

The broker accepts an acknowledgement as activation only from the proxy process
it last replayed to, with the run id that process first acknowledged under, and
only for the exact revision it pushed, and it bounds its whole wait for one
(`nvpair-ui-broker` README). Public egress additionally requires the separate
run/revision-bound lease (§12.9). A lease acknowledgement never substitutes for
policy or availability acknowledgement and cannot enable public dispatch.

#### Inference buffer policy

`routing.buffers.capacityMiB` is an integer from 192 through 1024 (default
256). `routing.buffers.waitSeconds` is an integer from 0 through 600 (default
5); zero disables waiting. The broker persists both in node policy and
activates them live through the existing `node/set-policy` acknowledgement
path. Buffer activation additionally requires
`capacityBytes >= 2 * effectiveIndividualLimitBytes + 1`, covering overlapping
old/new body storage and the overflow probe. The individual body limit is
64 MiB by default. A policy failing this requirement is refused without
changing the active policy or buffer settings.

One process-wide byte root covers owned inference storage across facades and
paired ingress: retained bodies, copies/rewrites, parsing/preparation and
response buffers. It charges allocation capacity before allocation, including
both arrays during growth, and retains a charge until the last borrower
releases it. It is not a process RSS bound, an engine/GPU memory limit, or a
promise about allocator/transport overhead. The stream category remains
subordinate to this root (§5.2). Live registry metadata and routing snapshot
copy/encoding storage also use this root (§12.5.1); cancellation of an existing
owner requires no additional capacity and remains reachable under byte pressure.

Waiting aggregate acquisitions join one FIFO queue with at most 1,024 waiter
records. Only the head can receive capacity: a large head blocks smaller
requests behind it. Queue saturation, aggregate contention with waiting
disabled, and wait expiry return 503. Each acquisition snapshots its own wait
allowance, shortened by any earlier context deadline; request cancellation
removes the waiter and releases its pending reservation. Pre-output waits
share the request's overall pre-output deadline, rather than extending it.
A body larger than the individual limit returns 413 without queueing for
capacity (when its size is unknown, as soon as overflow is detected). An
individual/category allocation or a request's own retained-plus-replacement
footprint that cannot fit its applicable limit also returns 413. Response
growth and some bounded category acquisitions refuse immediately (§5.2).

A live capacity decrease never revokes storage already held. Usage can remain
above the new cap while owners drain; new grants cannot increase that excess.
Queued proposals made impossible by the decrease are removed with 413;
remaining waiters retain their original deadlines. Admitted streams drain
within retained grants, but larger replacement allocations and pre-output
retries require current capacity; see §5.2 for the reader-specific conditions.
The desktop's Node policy JSON editor can change these settings; the TUI has
no buffer controls. See [Hybrid routing](../../docs/hybrid-routing.mdx) for the
desktop edit example.

#### Automatic residency control

The broker alone calls `node/prepare-resident`, `node/dispatch-resident`,
`node/finish-resident` and `node/observe-resident`. `handlePolicyMethod` decodes
the shared nodepolicy DTOs and delegates to the process-wide admission controller.
These are internal parent IPC, with no renderer or peer command surface.
`engine:reconcile-resident` belongs to engine-manager. Immediately before load,
its correlated `engine:dispatch-resident-request` notification asks the broker
to obtain the final proxy grant; only a valid matching notification reply
`engine:dispatch-resident-reply` can authorize the manager's load.

Prepare checks the whole revision-bound desired resident set against the same
resident/in-flight memory and model limits as inference, without eviction.
Served permission, desired membership, healthy On intent, local availability,
drain state and unique qualified identity are checked again at dispatch. NE
resident context equals native per-slot context; qualification covers native
context/output and effective concurrency equals native slots. Native retention
must be `"-1"`, router capacity must cover this engine's desired resident count,
and explicit load/loaded-read bindings must satisfy engine-manager's model-only
POST and safe GET contract. No prompt or request keep-alive override is sent.

Finish/observe reconcile operation-bound evidence; failed load, stale generation,
source replacement or unknown observation cannot invent loaded satisfaction.
A definite terminal manager `failed` result is reported by the broker as
`unknown`/`load-failed`, including recovery and retained-hold readbacks. It is a
bounded reason, not a raw engine error or proof of unloaded/free capacity;
observation, hold cleanup, terminal expiry and retry deadlines retain their
existing meaning. A stale proxy `engine-off` refusal remains `pending` without
a reason while the current manager source reports the engine running.

An interrupted load without a terminal manager reply retains its resident hold
and reports `unknown`/`operation-timeout` until a policy edit invalidates the
declaration or the proxy restarts. Elapsed observation time and manager
replacement alone do not release this nonterminal hold. After proxy or manager
replacement, fresh inventory and the manager-owned observed serving envelope
from the current source can report the resident `satisfied`; the hold remains.
Fresh manager-owned observed context/slots are required for restart confirmation;
an engine adopted from a previous manager run is never confirmed and remains
`unknown`. There is no persistent load queue replay. This protocol does not
clear durable strict-inference quarantine from HTTP completion or elapsed time.
Local pause/Off remains a veto, and PG retains route-policy ownership. See
[Local qualification](../../docs/local-qualification.mdx#daily-adapter-and-residency-contract)
for the on-demand/resident distinction and pending real-host acceptance.

#### 12.1.2 Engine switch invalidation and fresh drain evidence

These are internal broker requests. Invalidation requires every field, rejects
nulls and unknown fields, validates `engine` with `engines.ValidID`, and requires
`operationId` to match the switch request-ID rule: 1–128 ASCII bytes, first
alphanumeric, remaining alphanumeric or `.`, `_`, `:`, `-`. The only reason is
`engine-switch`. Under the admission mutex it removes every installed strict
model for that engine, advances each generation, and wakes queued admissions.
Held tickets fail `BeforeStrictDispatch`. Dispatched work is not cancelled;
capacity markers and quarantine are not cleared. Other engines are unaffected.
`StrictReadiness` remains withdrawn until the authoritative qualification feed
exists; invalidation never grants readiness.

Deduplication is per engine and operation ID for the lifetime of the proxy.
Repeating an invalidation has no effect even if fresh capabilities have since
been installed. Source and target may be invalidated independently using the
same operation ID. The broker must replay pending invalidations from its switch
journal before enabling facades after proxy restart; `runId` identifies the
incarnation acknowledging the invalidation. The proxy does not persist switch
journal state.

The snapshot request requires an explicit empty object, rejects other fields,
and copies current admission counts and activity maps under the admission mutex
at request time. Active counts include held tickets, dispatched work and
quarantined capacity, matching `admission/state`. The broker must request a fresh
snapshot after the drain acknowledgement and validate the returned `runId`
against that proxy incarnation. A cached zero, missing snapshot or changed
incarnation cannot prove a drained engine. The response is a snapshot, not a
subscription or a capacity-release operation.

Proxy → broker notifications, also process-scoped:

| Notification | When |
| --- | --- |
| `admission/state` | `{active, activeByEngine, queued, lastActivityMs}`; at most once a second while it changes, and immediately when `active` reaches 0 |
| `admission/unload` | `{engine, models}`: idle models admission wants unloaded to make room; the result arrives as the next `node/set-residency` |
| `admission/wake` | `{engine}`: a request arrived for a stopped engine saved On; the result arrives as `node/set-local-backend` reporting it healthy |

### 12.2 Admission

One controller per process, shared by every facade, because the limits are the
machine's (the same reasoning as §3.1). It runs in two places:

- the cluster ingress (§7), for inference a peer sends here: the body is read
  within `--max-request-bytes` (`413` beyond it), admitted, the model profile's
  `requestOptions` merged in, and only then forwarded. Non-inference routes are
  not admitted.
- `handleHTTP`, before dispatching to this node's own engine (the self
  candidate). A rejection is a failed candidate, exactly like a peer's `503`,
  and writes nothing to the client unless it is the request's answer.

Decision order for `(engine, model)`, model normalized per engine (§5 step 1):

Before routing resolution, admission, performance preparation or workload
placement, request model and tier-alias identifiers must fit 256 decoded UTF-8
bytes. Measure before whitespace trimming or normalization, including body
model/name strings, model query parameters and model-detail path identifiers.
Escaped Unicode counts as decoded UTF-8 bytes; an identifier above the limit
returns typed HTTP `400` without allocating its oversized decoded form. Existing
field/type and malformed-prefix behavior remains unchanged within this bound.
This identifier limit is independent of `--max-request-bytes` and the inference
root's `413` storage-impossibility / `503` contention distinction. Bounded
normalization (including `:latest`), workload IPC and retained control identifiers
are outside the inference byte root, except live canonical registry keys and
`routing/active` copy/encoding storage (§12.5.1). Performance queue entries own scalar and
identifier copies; finished-model timestamps own small keys and are created only
for resolved inventory or known resident identities, not arbitrary terminal
ingress names. Residency updates reconcile those timestamps at the existing
admission owner.

1. **Availability**: `paused` → `paused`, `draining` → `draining`; an engine
   drain → `draining`.
2. **Engine running**, as `node/set-local-backend` last reported. A stopped
   engine whose saved intent is Off, unknown, or with `idle.startOnDemand` off
   is refused `engine-off` without a wake. Otherwise `admission/wake` is sent
   (at most once per 10 s per engine) and the request waits for the engine to
   report healthy, up to `idle.wakeTimeoutSeconds` under the wait cap, then
   `wake-timeout`.
3. **Profile** by `(engine, normalized model)`. Its `requestOptions` merge into
   the body recursively for objects, with keys in the request winning.
4. **Concurrency**: `profile.maxConcurrent`, else
   `admission.maxConcurrentPerModel`; at the limit the request queues, then
   `busy`.
5. **Residency**: models reported loaded on any engine, plus models in flight,
   plus models that finished after the latest residency report (presumed still
   loaded). Before the first `node/set-residency` only in-flight models count.
   If the model would exceed `maxResidentModels`, idle models are unloaded least
   recently used first via `admission/unload` when `switchModels` is set; the
   eviction waits up to `switchTimeoutSeconds` for the residency to change and
   then admits anyway. Without `switchModels`, or when unloading every idle
   model would not be enough, the request queues for in-flight work, then
   `no-fit`.
6. **Memory**: with `memoryBudgetBytes > 0`, the resident and in-flight
   profiles' `memoryBytes` (unknown counts 0) plus this model's must fit,
   evicting or queueing as in step 5. A model larger than the whole budget is
   `no-fit` at once.
7. **Bounds**: queue and wake waits are bounded by
   `min(policy timeout, X-PAIR-Admission-Wait)`; the eviction wait by
   `switchTimeoutSeconds` alone, since it ends in admission. A caller that
   leaves while waiting is dropped.
8. **Admit**: the in-flight count is taken and released when the execution
   ends, including a panic unwind; `lastActivityMs` is stamped per engine on
   admit and release.

A rejection is `503` with `X-PAIR-Admission: <reason>`, `Retry-After`, and a
JSON body naming the reason.

**The router's wait header.** Each inference attempt carries
`X-PAIR-Admission-Wait`: `0` while the round has other candidates, so a busy
node hands the request straight back, and `admission.queueTimeoutSeconds` for
the round's last candidate. The self candidate is admitted with the same value
and the header is not passed to an engine. With a single candidate, a `busy`
refusal requeues with backoff without spending a dispatch attempt, bounded by
the job deadline.

**A paused node and its own requests.** Pausing affects only work executed
here. A self candidate refused while the node is not `available` is excluded
for the rest of the request; when nothing else can take it, the request is
answered at once with the admission `503` instead of spending its retry budget.

**Peers that are paused.** The same admission runs on the cluster ingress (§7),
so a request that reaches a peer that is paused or draining is refused there
with `503` and `X-PAIR-Admission: paused` (`Retry-After: 30`) or `draining`
(`Retry-After: 5`). This is what a router receives when its pause hint for that
peer was stale (§5); it fails over like any other failed candidate. The router's
own filter is described in §5 and is only a routing preference.

**Pause hint metadata.** `subscribedToNode` copies `DirectoryNode.Policy.PausedBy`
onto the internal `Node.PausedBy`, together with `Roles`, `ServedModels`,
`PolicyRevision` and `StrictAdmission`. `PausedBy` is `local` or `peer` for a
pause with recorded provenance and empty otherwise, so a stored pause without
provenance is not advertised as a pause. `nodeEqual` ignores these
fields, so a revision or pause-hint change announces no `node/updated`.
Availability itself is never carried in discovery TXT data; the hint arrives in
the peer's inventory policy. A node without a policy advertisement, such as a
manual node, has an empty hint.

**Waking a stopped engine.** The broker advertises an engine only while it is
healthy, so a stopped engine drops this node out of that engine's discovery
overlay. The proxy remembers its own node id and each engine's last advertised
models from discovery, and offers a *dormant* self candidate — last in the
round — when the engine is saved On, start-on-demand is set, the node is
available, and the model was last advertised by that engine or is covered by a
profile. Admission wakes it; the dispatch then targets the backend's reported
address.

**cancelActive** cancels every admitted execution: an ingress request's context
is cancelled; a self dispatch that has not committed moves on to another
candidate (this node excluded), and one that has committed ends as `cancelled`.

### 12.3 Tiers

A request whose `model` is `weak`, `medium` or `strong` on an inference route
is resolved by the node it enters.

- **Entry kinds**: a policy (schema version 2) tier entry has a `kind`. A
  `local` entry names an engine and model. An `api` entry names a deployment in
  the policy's origin-only `routing` section. The reachable concrete/legacy
  tier resolver considers `local` entries only and skips `api` entries without
  error: an API-only tier has no dispatch candidate there. Alias lists still
  include API-only tiers (below). The unified listener selects the common
  driver's hybrid branch (§5.1, §12.6). Migrating a
  version 1 file tags every entry `local`, so it resolves the candidates it
  resolved before.

- **Needs**, read from the buffered body in OpenAI and Ollama shapes: `tools`
  (non-empty `tools` or `functions`), `vision` (an `image_url`, `image` or
  `input_image` content part, or non-empty Ollama `images`), `structured`
  (`response_format.type` `json_schema`/`json_object`, or an Ollama `format`),
  `embeddings` (`/v1/embeddings`, `/api/embed`, `/api/embeddings`), and a
  context estimate of body bytes / 4 plus `max_tokens`,
  `max_completion_tokens` or a positive `options.num_predict`.
- **Search order**: the requested tier, then stronger tiers nearest first with
  `allowStronger`, then weaker tiers nearest first with `allowWeaker`, all from
  this router's `tierPolicy`.
- **Declared roles**: a node's `roles` (for a peer, the `roles` of the `policy`
  object in its inventory document as carried by the discovery overlay; for
  this node, its own policy's `roles` with the bounds of its matching profile,
  never its published document) name one engine and model per tier. A node
  that declares at least one role is authoritative for itself: for each tier in
  the search order it offers only the role it declared for that tier, and for a
  tier it omits it offers nothing. A node that declares no role (no `policy`
  object, or an empty `roles`) receives this router's tier entries as below.
  There is no migration from entries to roles.
- **Entries**: for each tier in the search order, first the declared roles,
  then the tier's entries for the nodes that declare nothing. A role or entry
  lacking a needed capability, or whose `contextTokens` is set and below the
  estimate, is dropped; a role is judged by the bounds its node advertises.
- **Candidates**: each role and entry expands through its engine's facade in this
  process with the ordinary owner filter and order (§5): node/select pin,
  scheduler priority, ID, self mapped to that engine's local backend, peers over
  cluster mTLS to the port that facade's discovery overlay carries — the peer's
  facade port for that engine. A role keeps only the nodes that declare it, and
  each is sent the model as its node spells it. The owner filter applies to
  roles too, so a declared model missing from the node's inventory is no
  candidate. A node whose `loadedByEngine` (or, for this node,
  the reported residency) contains the model is warm.
- **Engines**: `/v1/chat/completions`, `/v1/completions` and `/v1/embeddings`
  may cross engines. Every other inference route resolves only to its own
  engine. An engine without a facade enabled in this process contributes no
  candidates, because the process has no routing view of it.
- **Order**: `preferLoaded: "tier"` sorts by tier rank, then warm before cold;
  `"any"` by warm first, then tier rank. Declared roles (ordered by engine,
  then model) before entries, entry order, and the per-role or per-entry owner
  order break ties. Reservations (§6) are taken only among the candidates
  sharing the leading tier rank and warmth.
- **Body**: the top-level `model` is replaced by splicing its value's bytes;
  every other byte is forwarded as received.
- **Nothing eligible** before any dispatch: `404` with
  `{"error":"no eligible model for tier <tier>","needs":[...]}`.
- **Reserved aliases**: `weak`, `medium`, `strong` and each with a `-api` or
  `-local` suffix are reserved exact model names. Matching is byte-for-byte
  after trimming surrounding whitespace, so `<alias>:latest`, case variants and
  qualified names such as `org/weak-api` are ordinary model names.
  `normalizeModel` returns a reserved name unchanged, so Ollama's implied
  `:latest` is never appended to one.
- **Refusals**: a request is answered `400`
  `{"error":"...","code":"feature_unavailable","reason":"alias_not_inference_ready"|"unsupported_alias_use","model":"<alias>"}`
  before the CORS preflight, candidate resolution, admission, wake, reservation,
  workload events or any engine request, and only the `proxy/request` report is
  emitted, when it names:
  - a `-api` or `-local` alias in the body `model` on any route
    (`alias_not_inference_ready`);
  - a `-api` or `-local` alias as the string value of any top-level JSON key
    equal under `strings.EqualFold` to `model` or `name`, repeated keys
    included, so a case-variant key the engine may read instead of the one the
    router read is covered (Ollama `/api/pull`, `/api/show`, `/api/delete`,
    `/api/create` use `name`); as any `model` query parameter value (llama.cpp
    `GET /props?model=`); or as the whole remainder of a `/v1/models/` or
    `/api/v0/models/` path (LM Studio model detail)
    (`alias_not_inference_ready`);
  - a bare tier in the body `model` on a route that is not a classified
    inference route (Responses, an unclassified native or passthrough path), or
    a bare tier in `model` with surrounding whitespace
    (`unsupported_alias_use`).
  Nested keys, other fields and other paths are not checked. A bare tier in
  `name`, a query parameter or a model path keeps today's behavior. A bare tier
  spelled exactly in `model` on a classified inference route (including native
  and Messages routes) keeps the tier path above on facade entry; unified
  listener inference uses §12.6. The cluster ingress (§7) is not
  covered; refusing aliases there is Phase 4 work.
- **Model lists**: an OpenAI-shape model list (`/v1/models`, llama.cpp's
  `/models`) gains `{"id":"<alias>","object":"model","created":0,"owned_by":"pair"}`
  for the bare tier and its `-api` and `-local` variants, for each tier with at
  least one entry of either kind (including a tier whose entries are all
  `api`), each tier this node's own `roles` declare, and each tier a node in
  any facade overlay of this process declares. The records are built from the
  policy and those declarations (`configuredAliases`, `hybridAliasRecords`) and
  carry no credential, account, deployment or endpoint. An upstream model already named like an alias is left alone. When no
  upstream yields a list and at least one alias is configured, the aliases are
  returned with `200` instead of `503`; without a configured tier the statuses
  are unchanged, and the browser `Origin` checks of §5.8 still apply. The
  native Ollama list does not carry aliases.
- **Readiness**: listed `-api` and `-local` aliases are not inference-ready
  until Phase 4 of the hybrid routing plan; they are refused as above. Declared
  roles are local destinations: when dispatch exists, `-api` excludes them and
  `-local` and the bare tier include them.

### 12.4 Routing metadata

Every routed inference response that commits carries `X-PAIR-Model` (the
concrete model sent), `X-PAIR-Engine` (the engine that answered), `X-PAIR-Node`
(the node id that answered), and `X-PAIR-Tier` when a tier was requested. They
are set on the committed response only, so a failed-over attempt's headers can
never reach the client. Workload fields are in §5.5.

### 12.5 Cancel and regenerate

`workload/cancel {originatedFrom, runId, engine, workloadId, regenerate?}` acts
on an in-flight workload this process originated and answers `{found}`.
Every hop requires the complete key: the originating node, exact proxy run,
valid engine ID and workload ID. Missing or malformed fields are invalid
params (`-32602`); a valid key for another origin/run or an ended workload
finds nothing. Workload counters can repeat across engines and proxy restarts,
so neither engine nor run may be inferred from a numeric ID. Records missing
`runId` remain displayable but are not cancellable; clients must not fill in
the current run. For a departed run, the broker retires open or inferred local-origin
outcomes as `failed`, preserving authoritative terminals already accepted before
source withdrawal.

- **Before commit, with `regenerate`**: the current attempt is abandoned
  through the same claim the discovery watcher uses (§5.3), its node is excluded
  for the rest of the request, and dispatch continues with the remaining
  candidates, re-resolving as usual and emitting the usual `workload:submitted`
  placements. The aborted attempt costs no dispatch. If the exclusion leaves no
  candidate the request ends with `503`. Between attempts, the node last tried
  is excluded.
- **Without `regenerate`, or after commit**: the request is cancelled and its
  workload ends `cancelled` with the error `cancelled by request`. A committed
  stream is never re-dispatched, so two models' output is never spliced.

#### 12.5.1 Private request control

The process-scoped parent IPC methods `routing/cancel` and `routing/active`
use the existing live request registry, including requests without a concrete
workload or enabled facade. They have no peer HTTP adapter and no per-engine
addressed form. The broker exposes them to local clients as `routing:cancel`
and `routing:active`; it never relays these methods to another origin.

`routing/cancel` requires
`{request: {originNodeId, proxyRunId, requestId}, regenerate: bool}`
and returns `{status}`. Every field is required, including
the boolean. Identity strings use the shared strict key contract; `requestId`
is a canonical positive signed-64-bit decimal string, never a JSON number.
Unknown, duplicate, missing, null or malformed fields return invalid params
(`-32602`). A complete key cannot select another origin or proxy incarnation.

| Status | Meaning |
| --- | --- |
| `not_found` | No live owner for this complete key, including an ended request |
| `conflict_committed` | Regenerate requested after commitment; no state, callback or exclusion changes |
| `regenerating` | An eligible pre-commit regeneration uses the existing attempt transition and remaining bounds |
| `cancelled` | Ordinary cancellation, including a repeat while the cancelled owner remains live |

Regeneration abandons an eligible active concrete destination and excludes its
node. Between attempts it excludes the last node only when one exists. Before
any destination has been tried it still returns `regenerating`, consumes no
attempt and fabricates no exclusion. An uncertain hold is retained and cannot
authorize regeneration. Both request and workload cancellation enter one
mutex-protected owner transition and run cancellation callbacks after unlock.
The workload surface retains its legacy post-commit cancellation behavior;
the request surface instead refuses committed regeneration without side effects.
Cancellation retains the first accepted cancellation state/reason and uses
`cancelled by request`; it adds no second-request policy.

`routing/active {}` returns `{requests: [...]}` with the complete live keys.
The empty array is `[]`, never null. IDs sort by decimal length then lexically
(`9` before `10`, through `9223372036854775807`). Keys are copied under the
identity mutex; sorting and encoding run outside it. One snapshot copy through
response encoding is allowed per proxy instance; an overlapping snapshot is
refused with bounded control error `-32000` before copying. The broker similarly
allows one forward across both methods, using the existing five-second proxy
call bound and releasing its gate on every error. No routing history is added.

Every live registry entry reserves 512 bytes of metadata representation
headroom (linkage, key headers, lease and closure), plus the actual origin,
run and decimal-ID byte lengths, from the existing process byte root before
allocating owned key data or publishing/advancing the counter. The nonwaiting
grant never queues behind inference. Refusal gives new inference `503` before
body or other effects, with no counter advance. There is no independent live
entry cap; the allocator's separate 1,024 FIFO waiter bound is unchanged.
Cancellation does not acquire a new grant and does not release an entry's
charge. Final registry cleanup releases that row/key charge exactly once.
Map backing has separate root-funded ownership in the same registry owner:
both the canonical request map and workload secondary index retain a grant of
256 fixed bytes plus 256 bytes per explicit reserved slot. Partial deletion
releases row data but retains the backing grant; removal does not rebuild or
acquire capacity. An empty map is discarded and its backing grant released,
including an empty workload index whose canonical owners are still live.

Insertions reuse paid spare slots. The backing owner also counts cumulative
insertions since allocation, so deleted-slot/tombstone churn cannot increase
retained storage without bound. When that allowance is exhausted, insertion
nonblockingly reserves an entire fresh backing allocation while the old one
remains charged, copies only live rows, swaps maps, then releases the discarded
backing grant. Fresh backing has at least eight slots and room for twice the
new live count, rounded to a power of two. A failed overlap grant preserves
the old map/funding and publishes no new row or identity. No terminal rows or
per-departed-row leases are retained, and churn can compact backing on insertion.

Before any snapshot key copy, the proxy counts the complete reply footprint
under the identity mutex. It reserves 512 bytes of fixed metadata headroom,
128 bytes per copied key for headers/rounding, the actual cloned key bytes,
and four times each result JSON and complete framed-message upper bound for
encoding copies, scratch and allocation rounding. The envelope includes a
conservative sixfold escaped caller-ID bound and its newline. This covers the
result RawMessage and separately marshalled outer message while both coexist.
An ID beyond the conservative echo bound receives a bounded `-32000` error
with null ID, before gate admission, so the refusal itself cannot overflow IPC.
The complete framed reply must also fit the existing 1 MiB codec capability;
oversized replies and exhausted root capacity return bounded control error
`-32000` before copying, including for an empty snapshot. No uncharged fallback
or partial list is returned. The asynchronous worker holds its snapshot lease
through sorting, IPC encoding and successful or failed response I/O, then
releases it before the instance gate. Snapshot strings are independently
owned clones, so concurrent entry removal cannot release their storage early.
Snapshots are observations, not reservations: an owner may end before a
subsequent cancellation arrives.

### 12.6 Routing listener

An engine-independent loopback listener that serves the hybrid aliases (§12.3)
with no engine running. Its configuration is the active policy's
`routing.listener` (`enabled`, `port`, `allowedOrigins`), off by default. It has
no independently enabled facade, discovery state or cluster personality, and
is never advertised to peers. Bind state uses the unaddressed `errors:report`
and `errors:clear` below; no `ready`, discovery or node notification names its
port. Inference uses the existing request/workload notification contract.

- **Lifecycle**: `node/set-policy` reconciles it after the revision activates,
  from the policy the controller then holds. Enabled with a port, it binds
  `127.0.0.1:<port>` and nothing else. A changed port binds first, starts
  serving the new socket with the same `http.Server`, installs it as the active
  listener, then closes the old acceptance socket. Existing connections retain
  their server and context; rebind does not cancel them or keep a retired
  server. Disable cancels the server's base context, closes acceptance and
  calls `Shutdown` with a three-second maximum (`routingShutdownGrace`),
  forcing `Close` on timeout. Re-enable creates a fresh server. The server has
  a 10-second header timeout, 90-second keep-alive idle timeout and 64 KiB
  header limit, with no whole-response `WriteTimeout`.
  A request that declares a body is still answered with `Connection: close`,
  so net/http does not drain an unread body before a gate refusal. Accepted
  inference reads through the common owned buffering path. Only the listener this code owns is touched;
  facades are unaffected. Process shutdown closes it before the facades using
  the shared three-second shutdown context and permanently stops reconciliation.
  This bounds HTTP teardown after the control read loop returns; cancelling a
  context alone does not interrupt `codec.Read` or prove bounded parent I/O.
  A reconcile
  that finds the configured port already served is a no-op, so the broker's
  replay after a proxy restart re-establishes the listener without a new push.
- **Port exclusivity**: the listener does not bind a port another listener
  holds. Go sets `SO_REUSEADDR` on listeners, and on macOS a bind of
  `127.0.0.1:<port>` then succeeds beside a socket bound to `0.0.0.0:<port>`,
  `[::]:<port>` or `[::1]:<port>`, leaving two listeners on one port, with
  the other holder's clients (or a client dialling `localhost`) possibly
  answered by this one. So `listenLoopbackExclusive` binds and releases
  `0.0.0.0:<port>`, then `[::1]:<port>`, then binds `127.0.0.1:<port>`. The
  IPv4 wildcard probe fails while any wildcard holder exists (IPv4 or
  dual-stack, foreign or one of this process's facades); a holder on
  `127.0.0.1` fails the loopback bind itself. A `[::1]` bind failure counts only
  when an ephemeral `[::1]` bind succeeds, so a host with no IPv6 loopback
  address does not block the listener. The probes are released before the
  loopback bind, because holding them could make a platform that refuses
  overlapping binds collide with the listener's own probe.
  - **Residual window**: between the probe's release and the loopback bind (two
    system calls) a foreign process can bind the wildcard or `[::1]` and, on
    macOS, share the port with the listener without either side noticing. The
    same sharing results when a foreign process binds a wildcard address after
    the listener is up; the listener cannot prevent that. Only this process's
    own facades are refused (below).
  - **Verified**: macOS only (IPv4 wildcard, IPv6 wildcard, `[::1]` and a
    facade as holders). Linux and Windows have not been run.
  - **Facades**: the reverse order is closed from the other side. A facade's
    `listen` and `setPort` bind through `listenWildcard`, which refuses the
    port the listener holds under the listener's lock, so the broker retries
    the facade on another port.
  - **When both meet**: whichever binds first keeps the port. If a facade
    already holds the port when the policy is activated, the facade keeps it and
    the listener is reported blocked (retried on any later reconcile). If the
    listener holds it when `facade/enable` arrives, the facade is refused with
    the bind failure. After a proxy restart the broker replays the policy
    first, so the listener takes the port and the facade is refused.
- **Collision**: a failed bind or probe preserves the old working listener during
  rebind; an initial failure leaves none. It never picks another port,
  logs a warning and sends the unaddressed `errors:report`
  `nvpair-proxy:routing-listener-blocked` (severity `error`, action `none`).
  Each failed reconcile reports; there is no retry timer. The broker caches the
  report until the matching `errors:clear` and replays it to a restarted
  `nvpair-errors`, as it does for the OLLAMA_HOST alias warning.
  The revision still activates and acknowledges, so an unrelated setting is not
  held back by a port clash. Any later reconcile retries the bind. A bind,
  a no-op reconcile or a disable sends `errors:clear` for the same id the first
  time it is not blocked in this process, which also clears an error a previous
  process left in the pipeline.
- **Request order**: `Host`, then `Origin`, then route and method, then (for an
  inference route) the query check, then the shared inference entry. The first
  failing gate answers before body reads, registration, admission, discovery
  or scheduler effects. Preflight is handled before inference entry.
  - `Host` must be `127.0.0.1` or `localhost` (case-insensitive) with exactly
    the port captured from the accepted connection's local address by
    `ConnContext`, else `403` `host_not_allowed`. After a rebind, an old
    connection still requires its old port; it cannot use the new port in
    `Host`. This resists DNS rebinding.
  - Any `Origin` header (including `null`, an empty value or repeats) is `403`
    `origin_not_allowed` unless there is exactly one and it equals an entry of
    `allowedOrigins` byte for byte. The entries are read from the active policy
    per request, so a changed list needs no rebind. An allowed origin gets
    `Access-Control-Allow-Origin: <origin>` on every response; `Vary: Origin` is
    always set. Engine CORS is not consulted.
  - `GET /v1/models` answers `200` `{"object":"list","data":[...]}` from
    `hybridAliasRecords` over the active policy (its entries and this node's
    own `roles`; peers' declarations are not seen), `[]` when none; a query string
    on it is ignored.
  - A route is an inference route when any engine profile classifies the path as
    a `POST` inference route (`isInferenceRequest`), so the set is the facades'
    union. Another method on a known path is `405` with `Allow`; an unknown path
    is `404`. A query string or a bare `?` on an inference route is `400`
    `query_not_allowed`. Otherwise `serveHybridInference` in `routing.go`
    delegates to the common HTTP inference lifecycle with
    `hybridDispatchPolicy` (§5.1). Limits bound owned body preparation and
    dispatch before output; commitment stops that deadline, so a healthy
    accepted stream survives activation changes and rebind.
  - `OPTIONS` with `Origin` and `Access-Control-Request-Method` on a known path
    is a preflight: `204` with `Access-Control-Allow-Methods` (the route's
    method and `OPTIONS`), `Access-Control-Allow-Headers: Content-Type,
    Authorization` and `Access-Control-Max-Age`, provided the requested method is
    the route's own and every requested header is one of those two; else `403`
    `preflight_not_allowed`. An `OPTIONS` that is not a preflight is `405`.
- **Alias parsing**: `ownInferenceBuffers` and `bufferInferenceBody` retain the
  same storage account and effective individual body limit as the facade.
  `validateRequestIdentifiers` checks every top-level model/name occurrence;
  `ambiguousModelMember` refuses duplicates (including escaped keys) and
  case-folded model spellings with `400` `ambiguous-model` before selection.
  Exactly one top-level model naming an exact supported alias is required;
  malformed JSON, concrete models, missing models and alias lookalikes return
  `400` `unsupported-alias`. A single escaped spelling of the exact key or
  alias is accepted. Nested model values are payload, not selectors.
  Bare aliases follow the tier's `bareMode`; `-local` excludes public
  destinations and `-api` excludes local destinations. Public remains disabled
  by nodepolicy and the egress permission owner, so API-only requests return
  `404` without falling through to local candidates. Other aliases use the
  existing hybrid qualified-role selector, and only the selected concrete model
  replaces the top-level alias. Facade alias refusal and strict/concrete
  behavior (§12.3) retain their existing precedence. A strict admission header
  on the unified listener is refused with `400` `strict-admission`: destination
  admission remains local to the existing strict entry.
- **Entry ownership**: the compatible enabled facade with the lexically first
  engine name lends its dialect and notification ID sequence; OpenAI requests
  still resolve across enabled engines. No new socket, subscription, persistent
  state or request lifecycle owner is created. Without a compatible facade,
  a request-local empty adapter remains excluded from local selection even if
  a facade is enabled during body preparation. With public unavailable this
  path emits no workload and cannot collide with a facade workload ID.
  Both entries call `facade.serveInference`, which installs the request context,
  cancellation reason, `newRoutingRequest`, storage owner, canonical registry,
  response sink and common dispatch. Its defers unwind every early return and
  retain registry discovery until outstanding buffer borrowers join. Disable
  and shutdown cancel this same owner and force-close stalled sockets within
  the listener grace; no detached body reader is introduced. Parent transport
  writes remain bounded by `newOwnedCodec`/`NotifyContext` and transport close,
  independently of HTTP grace. Context cancellation alone does not interrupt
  `codec.Read` or prove broader parent-I/O acceptance.

### 12.7 Provider account ledger (D)

The proxy owns provider account state (`accounts.go`, `accountstore.go`,
`accountlock_*.go`). The private `runChatAccountAttempt` library runner
uses `reserveChatAccount` before sending and independently calls `reconcile`,
`recordOutcome` and qualification classification (`recordQualification` or
`quarantine`) afterwards (§12.8). Its sole intended production caller is G's
`runPublicChatAttempt` wrapper (§12.9), which remains unwired from HTTP ingress.

- **Owner and lock.** One `accountLedger` per process, one mutex for every
  operation from lazy open to adoption. `main` places the store in the app data
  directory; a proxy built without it (in-process tests) reports the ledger
  unavailable. The store opens on first use, which is the activation of a
  policy naming at least one account, or a reservation. Opening takes an
  exclusive, nonblocking lock on `proxy-accounts.lock` (`flock`
  `LOCK_EX|LOCK_NB`; `LockFileEx` with `LOCKFILE_FAIL_IMMEDIATELY` on Windows),
  held until the process ends; the OS releases it on exit or crash. A lock taken
  before a failed load is released, and a failed open is retried by the next
  operation, so a replacement proxy takes the store once its predecessor exits.
- **Store.** `proxy-accounts.json` is `{"version":1,"checksum":"<sha256 hex of the state bytes>","state":{...}}`,
  at most 8 MiB, replaced whole through
  `durablefile.WriteFile` on every change after an encoded-size check. Loading
  refuses another version, a checksum mismatch, unknown fields, trailing data,
  any invalid record (canonical money, identifiers, credential reference
  syntax, enums, a hold's amount recomputed from its bounds and prices, record
  caps, cross references) and leaves the file untouched; the ledger is then
  unavailable. State: `lastObservedMs`, `bindings` (credential reference to
  account, cap 1024), `accounts` (spend per period key `d:YYYY-MM-DD` and
  `m:YYYY-MM`, `generation`, `lastConfiguredCredential`, `cooldown`,
  `credentialFailure`, `frozen`,
  `probe`; cap 1024), `holds` (at most 4096, of which at most 256 fully completed),
  `quarantine` (cap 1024). Records hold names, never secret values.
- **Writes.** A mutation is applied to a copy, written, then adopted. A write
  that fails before replacement changes nothing. One that lands without
  confirmed durability (`durablefile.ErrReplaced`) is adopted, its new hold is
  `uncertain`, the call fails, and no reservation succeeds until a complete
  rewrite is confirmed. A restrictive transition (rate limit, exhaustion,
  credential failure, freeze, quarantine) that is not confirmed durable stops
  all reservations for the process lifetime.
  Each commit samples the clock once for both the rollback fence and period
  pruning, so a clock change between reads cannot erase current-period spend.
- **Arithmetic.** Money is integer nano-USD. A charge is
  `ceil(tokens * perMillion / 1,000,000)` with a 128-bit product; a hold is the
  input charge at the maximum uncached input rate plus the output charge, with
  checked addition. Overflow refuses.
- **reserve(deployment, inputTokens, outputTokens)** returns only after the hold
  is durable, with a ledger-allocated attempt ID (`<per-open nonce>-<n>`). It
  refuses: an unavailable store, an unconfirmed write that still
  cannot be confirmed, a clock behind `lastObservedMs`, an unknown deployment
  or account, an account whose reference is bound to another ID, a deployment
  without endpoint, protocol, qualification or USD prices in nano-USD per
  million tokens (no conversion), a quarantined identity, token bounds outside
  `0 <= input`, `1 <= output <= maxOutputTokens`, `input + output <= contextTokens`,
  a frozen account, a credential failure under the configured
  reference, an active cooldown, a half-open account whose single probe is
  taken, the per-request ceiling, `spent(today) + outstanding + hold > daily`
  or the same for the month, and a full ledger. Outstanding holds (`reserved`
  and `uncertain`) count in every period until resolved. A half-open account
  (expired cooldown, or credential failure under a changed reference) admits
  exactly one reservation as its probe. The hold records the account,
  deployment, endpoint, model, qualification, credential reference name,
  policy revision, currency, unit, rates, token bounds, amount, time and the
  account generation.
- **reconcile(attempt, usage)** resolves a `reserved` hold of this process with
  trusted usage priced at the hold's own rates, adding the actual to the spend
  of the hold's own day and month; nil usage charges the bound. A repeat with
  the same usage is a no-op, different usage a conflict; recovered, unknown or
  evicted attempts are refused without change. Negative usage is refused
  without changing the hold. Usage above a token bound
  freezes the account and quarantines the identity in one write and reports
  the failure; when the cost or the spend sum overflows, the hold stays
  outstanding.
- **recordOutcome(attempt, outcome)** once per attempt: `ok` clears cooldown and
  credential failure only for the probe and only if the account generation has
  not moved since its reservation; `rate-limited` (Retry-After clamped to
  1 s - 1 h, default 60 s) and `exhausted` (reset within 32 days) merge to the
  later deadline; `credential-failure` suppresses the account until its
  reference changes; a delayed failure for an obsolete reference cannot replace
  an existing failure for the configured reference. `transient` only releases
  the probe. Money is untouched.
- **recordQualification(attempt)** records successful completion of all identity
  and bound checks, independently of reconciliation and the account outcome.
  Call only when classification is complete; a violation calls `quarantine`
  instead. The hold persists `qualificationChecked`; an absent marker means
  classification is unfinished.
  Unfinished-classification holds from a previous process cannot currently be
  reclaimed and require a future owner recovery operation before dispatch activation.
- **quarantine(attempt)** records the identity the attempt was reserved under:
  `(endpoint, model, qualification id, revision)`. Deployment names, deletion,
  re-addition and restart do not clear it; a new qualification revision is a
  different identity. Quarantine also records completion of qualification
  classification, including when the same key was already quarantined.
  Above-bound reconciliation records that classification in its atomic write.
- **Configuration.** After each activated `node/set-policy` the ledger adopts the
  routing section. A configured account's reference is bound to its ID if
  unbound (written before any reservation); a reference bound to another ID
  makes that account refuse reservations, and activation still succeeds. An
  account absent from the policy is retired: its holds, spend, cooldown,
  freeze, bindings and quarantines persist and resume when the same ID returns.
  Its last configured credential reference also persists: delayed obsolete-reference
  failures cannot replace that reference's failure, while that reference's own
  failure supersedes an obsolete failure during retirement. Re-adding with a
  different reference is an ordinary reference edit. `lastConfiguredCredential`
  is optional for existing stores and is recorded when routing next configures
  the account.
  Price and ceiling edits apply to new holds only.
- **Recovery.** On open, `reserved` holds of an earlier process become
  `uncertain` and stay outstanding until an owner releases them; a probe left
  taken becomes a `probe-lost` cooldown of 60 s or the existing later deadline.
  Releasing uncertain holds and unfreezing an account are owner controls not
  wired yet. A hold becomes eligible for eviction only after reconciliation,
  its account outcome, and its qualification classification are all recorded.
  Only the latest 256 fully completed holds retain idempotency records; older
  completed records are pruned. Reservations stop at 3840 unfinished holds,
  including reconciled holds awaiting outcome or qualification classification,
  leaving room for completed history within the store's hold cap.
- **Not covered here.** Native lock and crash-window evidence on macOS and
  Windows, power-loss durability and the broker-driven proxy restart on a
  credential-reference change are activation gates; equivalent provider
  credentials under different references are outside the local guarantee.

### 12.8 Strict Chat adapter library (E; provider dispatch gated)

`chat_adapter.go`, `api_transport.go` and `runChatAccountAttempt` in `accounts.go`
provide a private single-attempt library behind G's `runPublicChatAttempt` wrapper
(§12.9). Public HTTP inference enters the accepted `facade.serveInference`
common entry, including API selection and bounded diagnostic refusals. Its
provider attempt path uses this wrapper subject to the production provider
gate; no real provider qualification is supplied here.
`remoteDispatchAvailable` in `shared/nodepolicy/routing.go` remains false:
enabled deployments and remote/paid tier permissions are refused. This section
does not disable that common HTTP entry or concrete engine routing. Provider
dispatch and its diagnostic refusal remain distinct from ingress acceptance.

#### Request and qualification contract

`chatQualification.prepare` requires `ProtocolOpenAIChatCompletions`, POST and
exactly `/v1/chat/completions`. The supplied `chatTarget` must match the profile's
endpoint, model and qualification ID/revision. Profile callbacks are trusted
in-memory inputs, not policy-configurable code or a production registry.

| Form | Implemented acceptance boundary | Owner |
| --- | --- | --- |
| Envelope | `chatRequestKeys` plus `parallel_tool_calls`, intersected with `chatQualification.Fields`; unknown keys fail rather than being dropped. Defaults fill absent fields, then undergo the same validation. | `prepare`, `chatObject` |
| Messages | Nonempty ordered history; profile-approved `system`, `developer`, `user`, `assistant`, `tool` roles and string content. Null content is allowed for an assistant with tool calls. Names and tool IDs are validated; every pending call needs one matching result before another non-tool message or history end. | `requestGrammar`, `chatName` |
| Function tools | Function declarations, unique names, validated schemas; `tool_choice` is `auto`, `none`, `required` or a declared function. `parallel_tool_calls` requires tools. Historical calls require unique nonempty IDs and JSON-object argument strings. | `requestGrammar`, `chatFunction` |
| Structured output | Profile-approved `text`, `json_object`, `json_schema`. Schemas additionally require the profile validator and the finite keyword subset: `type`, `properties`, `required`, `additionalProperties`, `items`, `enum`, `description`; references/recursive schemas are unsupported. | `requestGrammar`, `schema`, `chatSchema` |
| Scalars | Profile-approved booleans for streaming/parallel tools; temperature 0–2, top-p 0–1, penalties −2–2, integer seed, string stop or an array of 1–4 strings. `stream_options` contains only boolean `include_usage` and requires streaming. | `chatScalars` |
| Output/context | Positive integer `max_tokens` or `max_completion_tokens`; if both appear they must agree. Omission requires a positive qualified default. Output and input-plus-output must fit both deployment and qualification bounds. | `prepare`, `chatAdd` |

`chatObject`, `chatExactUnique` and `chatDecodeStorage` reject malformed JSON/UTF-8,
exact duplicate keys, noncanonical protocol field spelling, trailing JSON and
whole-document nesting beyond `maxStrictBodyDepth` (64). Case-distinct property
and argument data keys remain distinct. Supported message/tool/schema values
and array order are preserved; top-level serialization uses canonical key order.
Request body and profile defaults each have a 16 MiB input ceiling.

`chatQualification.fresh` requires nonreasoning billing evidence, named accounting,
tokenizer and template proof, accepted reported-model mappings to one canonical
model, positive framing overheads, finite limits and `finish-then-done` termination.
Evidence must have `CheckedAt <= now < ValidUntil` and age strictly below `MaxAge`,
which cannot exceed seven days. `InputBound` covers the entire final serialized
body, including UTF-8, history, arguments/results, tools and schemas; request,
message, tool and schema overheads are added with checked arithmetic. There is no
bytes-per-token guess or unknown/unlimited output fallback.

Responses, Anthropic Messages, legacy Completions, native engine protocols,
embeddings, image/audio content and opaque reasoning are unsupported.
On the public-account (chat adapter) path, `n` (even one), `store`, metadata/user
tracking, previous-response/conversation state, logprobs, reasoning fields,
context overrides and unknown extensions are rejected even if a profile lists
them. Stateful SSE fields are unsupported.

On the strict local path, `store:false` is accepted as a no-op and stripped
after profile defaults are merged, before accounting and forwarding to the
engine. Both the raw request and the effective request with defaults are
validated: `store:true` is refused with HTTP 400 `strict-qualification`, as is
any `store` value other than the JSON boolean `false`.

#### Public transport and resource ownership

`newPublicClient` uses `nodepolicy.CheckEndpoint`: a full HTTPS URL with a
lower-case DNS hostname and nonempty unreserved path segments. It refuses IP
literals, local-name suffixes, userinfo, query (including bare `?`), fragments,
percent encoding, dot/empty path segments and explicit default `:443`. Other
valid explicit ports are allowed. The configured URL is used unchanged; ingress
URLs and headers are not inputs.

`pinPublic` resolves before final permission acceptance, rejects empty or mixed
public/nonpublic answer sets, and pins one checked numeric address and the
connect deadline into the request context. `dialPublic` consumes that pin without
re-resolution or retry. `publicAddress` unmapped-address validation and `publicDeniedPrefixes`
exclude private, loopback, link-local/metadata, unspecified, multicast, CGNAT,
documentation, benchmarking and reserved ranges; IPv6 must also lie in `2000::/3`
outside the denied prefixes. Host and TLS identity remain the configured origin.
The fresh transport uses system roots, hostname/certificate verification and TLS
1.2 minimum, with no client certificate, cluster trust or cookie jar. Environment
proxies, redirects, connection reuse, automatic compression and HTTP/2 attempts
are disabled. Requests specify `Accept-Encoding: identity`; nonidentity or multiple
Content-Encoding declarations are refused without decompression.

`publicClient.postObserved` (and the synthetic `post` convenience seam) builds
only protocol headers and the selected bearer credential;
it accepts no caller request/header object. Raw provider headers/trailers and error
bodies are never downstream material. Returned metadata is status, synthesized
content type, timing and an initial Retry-After value bounded to 128 bytes for
internal classification. Errors do not echo credentials or provider content.

| Limit | Enforced boundary | Owner |
| --- | --- | --- |
| Network phases | Connect/DNS and TLS handshake 1–60 s; response headers and body idle 1–600 s; response headers 64 KiB. Caller context must supply the overall pre-output deadline; there is no whole-generation client timeout. | `newPublicClient`, `dialPublic`, `publicBody` |
| Body bytes | Explicit positive successful-response ceiling up to 64 MiB and independent error-body ceiling up to 1 MiB; declared overflow refuses before reading, otherwise reads probe at most one overflow byte. No implicit library default is installed. | `newPublicClient`, `post`, `publicBody.Read` |
| Parser | Positive profile body ceiling up to 64 MiB; event up to 1 MiB and no larger than body; line no larger than event; reconstructed tool bytes no larger than body; tool-call cap 1–128. Runner also requires profile body/event limits within transport limits. | `responseLimits`, `runChatAccountAttempt` |
| Admission | Nonblocking `publicBudget`/`chatByteBudget` with idempotent releases; absent/refused hooks fail closed. Transport reserves `3 * (max(success cap, error cap) + 1)`; parser admits fixed storage, raw growth and actual decode/token storage before allocation. | `post`, `chatAcquire`, `chatLease`, `chatDecodeStorage`, `parseResponse` |

Reservations bound live adapter-owned representations, not Go process heap or
cumulative allocations. Conservative decode headroom can refuse a body below its
wire ceiling. `publicBody.Close` cancels upstream and releases headroom once;
EOF alone retains its lease until Close. Prepared bodies retain their lease until
`chatPrepared.close`. Parser leases release on return, failure and panic unwinding.
The `emit` callback borrows bytes only during the call; retained output and
qualification callback storage require caller-owned budgeting. I must supply the
aggregate process-wide owner; no unlimited production budget is provided.

#### Response and ledger completion

`chatParseResult` separates headers, first body byte, first meaningful text/function
delta, protocol terminal, exposure and trusted usage. `responseEvent` checks every
envelope's reported model against `AcceptedModels` and requires stable response ID,
creation time and reported model across the stream. This validates profile-declared
identity, not model attestation. Only choice index zero is supported. Nonstream
completion requires a qualified finish reason; SSE requires a finish followed by
`[DONE]`, terminated events and clean EOF. EOF/truncation cannot substitute for
terminal completion. Validated SSE metadata/comments can set `Exposed` before any
meaningful delta; exposure forbids replay, and the library performs no retries.

`chatUsage` accepts only nonnegative integer prompt/completion/total token fields
with checked, equal total. Usage is trusted only when the profile permits it and
identity and protocol completion succeed. Missing usage on a complete response is
distinct from malformed, partial or mismatched usage. Complete above-bound usage
returns `errChatBound` with trusted usage for ledger enforcement.

`prepareChatAccount` copies configuration and revision;
`reserveChatAccountAllowance` compares the entire snapshot under the ledger mutex
before durable reservation against the request's remaining allowance.
`runChatAccountAttempt` validates/prepares and admits parser/transport headroom
before invoking the resolver for only the selected canonical credential reference.
There is no built-in environment lookup, secret cache or credential-value identity.
Empty/control-character credentials refuse. It then reserves, associates the
typed reservation with I's lifecycle even if cancellation won during the write,
and invokes the mandatory advisory post-reserve guard. Freshness and cancellation
are checked again at the final guard in `postObserved`, immediately before
`http.Client.Do`. G rechecks permission, lease and revision and invokes
`lifecycle.acceptedByGuard` under the permission locks. All request construction,
DNS validation and headroom acquisition precede final acceptance; no blocking
work occurs between its successful return and `Do`, and locks are released before
I/O. A successful synthetic guard is not an egress lease.

A post-reserve refusal retains the hold. E preserves `errPublicFinalGuard`
instead of collapsing it into `errChatAttempt`; this proven-unsent branch performs
no provider-outcome, money or qualification completion. G maps it to the typed
`final-guard` refusal (§12.9). D reservation/accounting failures are distinct from
these permission stages and can also leave a durable or uncertain liability.

| Observation | Sanitized runner category / ledger outcome |
| --- | --- |
| Validated HTTP 200 completion | Success / `outcomeOK` |
| 401/403 | `errChatCredential` / `outcomeCredentialFailure` |
| 400/404/422 | `errChatConfiguration` / nonrestrictive `outcomeTransient` |
| 429 | `errChatRateLimit` / `outcomeRateLimited` |
| Connection/TLS/redirect failure, 408/5xx or unsupported status | `errChatTransient` / `outcomeTransient` |

Observed 401/403/429 restrictions survive error-body overflow, encoding refusal or
drain failure. `chatRetryAfter` parses initial header seconds/date only; invalid,
past or overflowing values use D's 60 s default, valid delays clamp to 1 s–1 h.
Generic 429 never becomes `outcomeExhausted`; provider bodies cannot invent account
state. Classification grants no retry permission or zero-cost conclusion.

The runner attempts `recordOutcome`, money reconciliation and qualification
classification independently, even after a sibling write fails. Only complete,
identity-valid results reconcile: missing usage charges the full reserved bound;
trusted above-bound usage goes through `reconcile` to freeze/quarantine atomically.
Identity violations call `quarantine` using the reserved immutable identity, never
reconcile untrusted usage. `recordQualification` marks finished classification,
including classified failed attempts, not successful inference. Budget refusal
leaves classification unfinished; panic unwinding does not fabricate completion.
Cancellation, timeout, interruption, malformed usage and failed persistence do not
refund uncertain money. Ledger failures surface as `errChatAccounting`; restrictive
write failures preserve D's admission latch.

#### Activation gates

F must supply real provider qualification artifacts, tokenizer/billing/model and
terminal evidence, and evidence acquisition/freshness. G supplies the lease and
guarded wrapper (§12.9); B still requires finite local qualification. I must
complete public ingress/attempt/cancellation integration using its existing
request owner, overall pre-output deadline and aggregate buffer owner. D's native lock/crash-window acceptance,
uncertain-hold/unfreeze controls and previous-process unfinished-classification
recovery remain pending gates (§12.7); broker credential-reference restart also
needs acceptance. Synthetic profiles/tests prove library boundaries only and
cannot enable providers, establish vendor-wide ceilings or close native/live
acceptance. No default fake profile or activation path is supplied.

### 12.9 Public egress lease and stable consumer (G; provider dispatch disabled)

`shared/nodepolicy/routing.go` owns the single `remoteDispatchAvailable=false`
constant. Policy validation refuses enabled deployments and remote/paid tier
permissions; G's early and final production gates read the same value through
`PublicDispatchAvailable()`. No environment variable, flag, policy field, mutable
test switch or alternate enable bit exists. Production provider egress remains
zero even if private lease messages are acknowledged; the accepted public HTTP
common entry and its diagnostic responses remain available. Provider activation requires a later
reviewed source change after F's real OpenCode qualification, D's native durability
and recovery gates, B's finite qualification, I's complete ingress/attempt/cancel
path and final acceptance evidence. Qualification never flips availability.

#### Receiver-clock authority

The typed wire contract is in `shared/nodepolicy/wire.go`. `runId` names this
proxy process and `revision` its exact active policy. `sequence` is a canonical
positive decimal string within signed 64-bit range; `notAfterNanos` and reply
`nowNanos` are canonical nonnegative decimal strings in this proxy's monotonic
clock domain, never wall-clock timestamps or JSON numbers. Invalid payloads or
bindings return `-32602` without changing the lease.

- An anchor-only renewal (`notAfterNanos: "0"`) validates the run, ready policy
  revision and sequence, advances the high-water, and samples `nowNanos` without
  changing any existing deadline. It neither grants nor revokes authority.
- The broker records returned proxy sample `p0` at broker monotonic receipt time
  `b0`. At later issuance/request-start time `b`, before codec I/O, it computes
  `notAfter = p0 + (b - b0) + 5s`, with checked elapsed time and signed overflow.
  Sampling precedes receipt, so under the shared host monotonic clock rate the
  deadline is no later than issuance plus five seconds. Pipe and acknowledgement
  delay consume validity rather than restart it. A delayed anchor can only shorten
  or deny a grant. Grant expiry does not cause re-anchoring.
- The receiver accepts a nonzero deadline only when `now < notAfter` and
  `notAfter - now <= 5s`. Permission refuses at `now >= notAfter`, with no grace.
  A renewal also requires the exact run/revision, ready ledger activation, no
  activation transition and `sequence > high-water`.
- Renewals and revokes share one high-water across policy changes and
  withdrawals for the whole run. Duplicate/reordered messages cannot restore
  authority. A higher-sequence revoke for the current run clears deadline and
  revision without requiring a policy revision. Stale-run messages are refused;
  a replacement starts without any inherited lease, anchor or sequence authority.

#### Broker eligibility, withdrawal and status

One context-bound renewal job belongs to `Broker.Serve`; there is no startup
grant or per-facade job. Each one-second tick requires `ready`, no policy recovery,
the live exact `proxyProcess` equal to both runtime incarnation and lease process,
a nonempty acknowledged run, `activeRevision == current.Revision`, no policy or
availability error, no pending policy/availability push and no withdrawal latch.
It acquires the policy push slot with a one-second context, then rechecks the
same process/run/revision and eligibility before issuing. Missed ticks do not
create a retry backlog, and the runtime mutex is never held across codec I/O.

Adopting a new desired revision or starting an unacknowledged push stops renewal.
Failed, lost, unreadable, wrong-origin/run/revision activation acknowledgements,
failed availability pushes, process exit/replacement and withdrawal cannot restore
eligibility. A renewal transport/RPC error skips that tick and is re-evaluated
next tick without latching activation closed or discarding a valid anchor. A
successful renewal RPC with an undecodable reply or a decoded reply with an
invalid run/revision/sequence binding invalidates the anchor and records both
activation and availability errors; fresh exact acknowledgements
of both are required before a new anchor/grant. Negative elapsed time or arithmetic
or sequence exhaustion denies further issuance.

`persistPolicy` adopts complete replaced bytes on `durablefile.ErrReplaced`,
pushes that revision and still returns the save error. Matching activation of
that adopted document remains lease-eligible; there is no second durability
tracker. A failure before replacement leaves the old desired document intact;
invalid input/stale CAS cannot manufacture a revocation revision.

Policy results and `policy:changed` retain desired versus `activeRevision` and
`activation` (`pending`, `error`, `active`) with bounded operational error text.
A failed restrictive push does not report the new revision enforced. The old
lease can authorize no new attempt after its issuance-plus-five-second bound.
These policy status fields are not a public-dispatch permission signal. Scheduled
binding-error announcements coalesce to at most one sender plus one waiter;
the underlying parent client-write lifetime remains unbounded by G.

Credential-reference restart and intentional shutdown latch withdrawal before
retiring the old supervisor, attempt a higher-sequence revoke on the exact old
pipe within one second, then stop/join the old proxy even if withdrawal fails.
`Serve` cancels/joins renewal before teardown. Crash/replacement invalidates old
eligibility; replay re-establishes policy/availability, not an inherited grant.
Proxy broker-stdin EOF cancels its request context. This supplies no new
engine-idle, completed-fanout or transport-write lifetime guarantee; §5.5 and
§12.5 retain the departed-run rules.

#### I's single-attempt contract

The exact signature and owner trace are in
[Public egress lease](../../docs/public-egress-lease.mdx). I may call
`publicAttemptPermitted(tier, snapshot) error` to skip a candidate before body
construction/qualification. Its success is provisional; every production attempt
must call `(*Proxy).runPublicChatAttempt` with the immutable D snapshot, E
qualification, selected transport/resolver, one request context, byte-budget and
emit hooks, and a request-bound `*publicAttemptLifecycle`.

G observes the current validated account (including refused candidates) to tighten
I's allowance and checks build/lease/permission early. D/E own preparation,
selected-reference lookup, durable reserve, response parsing and independent
completion. The unchanged E runner takes `publicAttemptGuard`, which accepts a
nil continuation for its advisory post-reserve check and I's acceptance hook for
the final handoff. `publicPermissionAccept` holds G's lease mutex, then admission,
then invokes I's request transition; it checks exact active snapshot/deployment/
account, enabled API membership, remote/family permission and `PaidAllowed` for
`BillingPaid`. Trial credit is not permanently free eligibility.

`publicAttemptRefusal` unwraps to `errPublicPermission` and distinguishes
`before-reserve` (no new hold), `after-reserve` (hold retained) and `final-guard`
(proven unsent, hold retained). Reasons are `permission-refused`,
`cancelled-before-send` and `request-lifecycle-refused` where applicable. Other E/D
errors keep their categories; they are not proof of no liability. No guard refund
or release API exists. Revocation prevents the next acceptance; already accepted
in-flight work retains accounting and may complete.

I retains the trusted local-origin tier/mode and qualification decision,
`RoutingRequestKey` (origin node, proxy run, request ID), total attempts and one
pre-output deadline, cumulative paid allowance across destinations/regeneration,
cancellation, aggregate process byte root and output sink. Lifecycle association
retains D's durable/uncertain hold even when cancellation races reservation;
settlement never refills granted allowance. The first raw positive successful
response read drives commitment through `firstSuccessfulByte`, even if parsing
later refuses it; discarded provider error bytes do not. An owned sink supplies
actual delivery facts and replaces the synthetic `emit` seam. I must not call raw
`runChatAccountAttempt`/`post` in production, cache guard success, reserve twice,
retry within the guard, enter G while holding its request mutex or turn a lease
ACK into permission. The wrapper has no production ingress caller yet.
