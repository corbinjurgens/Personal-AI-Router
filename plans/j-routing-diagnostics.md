<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution plan: J routing diagnostics (J0/J1/J2, P5.1)

## Goal

Expose bounded, redacted routing history on the origin machine in desktop Settings → Routing and a terminal Routing tab. Project I's immutable request identity, attempt facts, waits, account state, spend and winning terminal disposition. Keep private diagnostics off workload, scheduler and cluster wires. Preserve the owner's rule: show delay/pause-resume while qualified same-model recovery remains possible, and an error only when that possibility is exhausted.

## Change scope

Intent: private routing observability and method-aware workload privacy projection.
Scope: Bounded improvement at existing service/bridge/client owners, with focused validation, tests and documentation. No routing-policy, retry, admission, billing, ledger, HTTP compatibility or continuation-execution changes.
Basis: package-J handoff, R1–R8/LOW/UNVERIFIED in `../reviews/j-plan-review-r1.md`, and coordinator-adopted answers below.
Compatibility: preserve I's `routing:active`, `routing:cancel`, `routing/active`, `routing/cancel`, complete keys and workload cancellation identity. Add local history methods only. Strip well-formed unknown workload fields; reject malformed envelopes before effects. Preserve all legitimate workload fields.
Non-goals: persistence, peer history access, a new settings page, history cancel controls, SC-b implementation, public-provider activation, generic harness work, configurable history caps and unrelated refactoring.

## Context the executor needs

### Verified source and provenance

Amendment inspected `origin/integration/join12` at **81960f6201e296d7f78274b6e8515ed0c6a8d8d5** with git show/grep/ls-tree, without checkout. Join12 is being repaired for I3a services/tests identity fixtures; its product shape stands. Record the accepted repaired successor before dispatch. Original brief: `git show origin/plan/cloud-handoffs:plans/handoffs/j-routing-diagnostics.md`; historical branch/commit/push instructions there are superseded by this in-place, plan-only amendment request. No proposed tests/builds/native checks were executed.

Prose shorthand: P = services/nvpair-proxy, B = services/nvpair-ui-broker, N = services/shared/nodepolicy, T = services/nvpair-tui/ui, X = services/tests, D = desktop/src. Files fields use repository paths.

- `P/request_registry.go:26–43` allocates the checked process-wide counter and immutable RoutingRequestKey under identityMu; `P/http_handler.go:540–549` registers before inference effects and defers cleanup. `P/routing_identity.go` binds origin/random incarnation before readiness. J uses that exact key, never discovery identity, facade counters, empty origin or broker stamping.
- `P/request_spend.go::routingSpend`, `publicAttemptLifecycle.tryAccount`, `associated`, `finished` own ceiling/grants/uncertain liabilities/delivery/accounting. A refused paid candidate can tighten the ceiling; settlement never refills allowance. I4b's successor owns the common request/attempt/sink; consume immutable snapshots there.
- `P/proxy.go:178–215` defines Workload and `workloadParams{workloadInfo: Workload}`. `B/workload_projection.go:164–224` handles nested lifecycle records and top-level removals, with replay/rehydration earlier in the file. Filtering params by Workload tags destroys the lifecycle envelope.
- `B/client_requests.go:178–187` returns -32601 for internalClientMethods before dispatch. History client methods belong in the dispatch switch, never that denylist.
- `N/routing_wire.go::decodeStrictContract` rejects null required fields and takes a required-key list. A focused nullable-aware history decoder is required; do not weaken existing strict admission/key decoding.
- `X/workload_interop_test.go::TestWorkloadManagerOutboundBroadcast` (558 onward) uses newInterNodeCluster, startDirectClusterPeer, copied fixed-path workload bridge, explicit settings journal, pinned peer, readiness and process inventory. Its fake Ollama POST /api/chat returns `{"done":true}`; it does not use newServiceFixture or Chat Completions. `temporaryClusterPort` in cluster_outbound_pairing_test.go:33 does not exclude protected ports.
- D/shared/types/json.ts and bridge json-rpc-client.ts define concrete JsonValue types. I6 supplies routing types/bridge: extend them, do not unconditionally recreate them. Renderer never imports Electron.
- D/ui/components/Settings.tsx mounts the existing Routing settings. Load repository KUI guidance at UI execution. T/view.go:21–38 requires Title, Init, SetSize, Update, View, `Help() []key.Binding`. Default views are Nodes, Jobs, Service, Errors, Logs; maxTabs is 9. E4 Task 2 is integrated in node detail, with no new top-level tab.

### External prerequisites and reservation transfers

Read siblings i-unified-dispatch.md, amended pa-paused-advertisement.md, sr5-rescope.md, xa-remote-config-acceptance.md and e4-engine-switch-controls.md before execution. Coordinator records accepted SHAs/ancestry, exact releases and generator-writer interval before dependent dispatch.

- I2/I3a are present subject to the fixture repair. Task 3 waits for integrated **I3b, I8, I4a, I4b, I5 and I6** plus exact releases. I4b supplies sole request/attempt/sink disposition; I5 supplies real hybrid selection. No unreachable API placeholder. I7 native proof is not needed for disjoint source authoring and is not J acceptance evidence.
- Tasks 2/3 wait for I6 contract join/release: N routing wire/tests; D routing types/bridge, ws-channels, empty-handlers, pair-api; B broker/client_requests; proxy/broker README/spec; exceptions/generated API. Preserve active/cancel methods and independent history subscription flag.
- I3b/8/4a/4b/5/6 proxy reservations include http_handler.go, cancel.go, proxy.go, request_registry.go, request_spend.go, routing.go, response_sink.go, ingress.go, accounts.go, api_transport.go, chat_adapter.go, performance.go, streamframe.go, tiers.go, routing_listener.go and focused tests. This inventory grants no blanket edit permission. Before Task 3 dispatch inspect accepted successor, name exact hooks/files, refresh Files and rerun graph audit. No abortReason on a deleted controller or competing recorder transition authority.
- Serialize N wire tests with SR5a; B broker/README with G3/SR5c; proxy README/spec and generated outputs with SR6a. Preserve **I3 → G3 → SR5c**, **I6 → SR6a**. NE4 overlaps HTTP/ingress/admission/shared wire; NE/PG/SR/I generator work needs a coordinated writer. Do not require unrelated resident implementation.
- PA's amendment supersedes stale PA-before-I with **I owner cutover/release → PA**. Coordinator reconciles I's stale gate before dispatch; do not introduce a cycle. PA reserves metadata/candidates and subsequent proxy/broker docs; transfer exact overlaps after the current owner without editing PA here.
- E4 requires no pending central-bridge rewrite or Routing mount gate. Real shared paths are TUI README, terminal docs and frontend/parity docs. Require their exact release. Recount views and release ui.go if another lane has since changed navigation.
- Task 6 waits for repaired I3a fixtures, XA/I7 fixture-owner release, separately reserved isolation/port helper repairs and PC3 interval. Shared helper repair needs its own exact Files/dependency/accepted SHA, never silent expansion into their harnesses.

## Decisions (owner and coordinator-adopted answers)

- Owner: exact decimal nano-USD strings end to end. I owns tightest paid-account request ceiling and cumulative granted allowance, never refilled by settlement; J observes.
- Owner: show delay/pause-resume while qualified same-model recovery remains possible; error only when exhausted. Consume I4b disposition. SC-b is absent at I's endpoint, so terminal continuation-unavailable failure can truthfully say Stream interrupted. Cancellation stays cancellation.
- Adopted answer 1 (owner may veto): memory-only broker history; **broker restart clears history**. No persistence file/database/setting. Proxy restart retains prior broker history, without inference replay.
- Adopted answer 2 (owner may veto): show validated local account/deployment policy IDs without hashes on origin machine. Never project credentials, endpoint secrets, Billing/config objects or raw account structs.
- Adopted answer 3 (owner may veto): fixed presentation bounds 256 records/32 attempts, oldest dropped, plus metadata/copy bounds below. No configuration. Trimming cannot affect I's complete lifecycle/accounting.
- Adopted answer 4 (owner may veto): strip well-formed unknown fields at correct nested projection; reject malformed envelopes. Diagnostics use fixed categories/counts, never raw key names.
- Adopted answer 5 (owner may veto): pause only while I says recoverable; Interrupted, resumed only after an actual linked successor. Nullable continuation is not proof of recovery. No terminal inference from HTTP status, raw first byte, workload state or handler return.
- Existing KUI layout; no history cancel action. Private routing data never rides Workload or reaches manager/scheduler/peers. Never diagnostics/log prompts, messages, response bodies, PINs, credentials or keys. Tests use synthetic sentinels, never real/private content.

## Decisions made while writing this amendment

- Full-record snapshots and accepted-upsert monotonic merges at every async boundary. Broker process-exit retirement is a presentation event, separate from I's inference terminal authority.
- Task 3 is one atomic accepted RPC change: producer, broker relay, every existing consumer, desktop bridge, TUI decoder, tests and docs. Task 2 schema-only work emits no method. Partial producer/receiver commits cannot independently claim RPC delivery.
- A small versioned semantic JSON compatibility corpus is justified; it is not an immutable frozen interface. Same-schema additive unknown fields are stripped at declared projections; changed meaning/type requires a schema bump/coordinated migration. No byte-order comparisons.
- Detached local workload associations remain historical; later API attempts cannot make those local jobs appear live. Card summaries describe associated local attempt facts after detachment.

## Executable view contract

Task 2 Go structs and Task 3 TS types have these exact JSON members. All keys required/non-null unless `?null` marks present-but-nullable; never omit those keys. Arrays are non-null. Destination's kind-exclusive fields are required but nullable as below.

```text
key = {originNodeId, proxyRunId, requestId} (I's existing type/validation)
destination = {kind, model, nodeId, engine, deploymentId, accountId}
  local: nodeId/engine nonempty; deploymentId/accountId null
  api: deploymentId/accountId nonempty; nodeId/engine null
workload = {originNodeId, proxyRunId, engine, id, attemptIndex, attached}
requested = {model, tier, mode}
waits = {admissionMs, firstByteMs?null}
account = {state, heldNanoUsd?null, chargedNanoUsd?null, accounting}
attempt = {index, startedAt, endedAt?null, destination, outcome, reason,
           status?null, waits, continuationOf?null, account?null}
spend = {currency, ceilingNanoUsd?null, grantedNanoUsd?null,
         heldNanoUsd?null, chargedNanoUsd?null, accounting}
record = {schema, request, workload?null, seq, requested, createdAt,
          endedAt?null, state, reason, status?null, served?null,
          recoverable, spend, droppedAttempts, attempts}
initial = {requests: record[]}
```

- Schema 1; seq is projection revision from 1 through 9007199254740991, never wrapping. requestId remains canonical positive signed-64 decimal string. Origin/run use I validator; other IDs/engine keys use `[A-Za-z0-9][A-Za-z0-9._:-]{0,127}`. Model is nonempty valid UTF-8, at most 512 bytes, no control characters. Source IDs/model only from validated owner metadata, never bodies/errors/headers.
- Times/durations/indexes/counts are nonnegative safe integers; timestamps Unix ms, ends not before starts. Status null or integer 100–599. firstByteMs is null until I's successful raw-body event, which does not imply downstream delivery. Booleans: recoverable/attached.
- Money passes ParseNanoUSD: canonical nonnegative signed-64 decimal digits, strings only. TS bounds by decimal length/lexical comparison to `9223372036854775807`; no Number/parseFloat. Currency USD; null unavailable/not applicable, zero known zero.
- State: routing/streaming/completed/failed/cancelled. Outcome: pending/committed/rejected/failed/redispatched/interrupted/cancelled. Account state: ok/cooldown/frozen/credential-failure/exhausted/probe/conflict/unknown. Accounting: none/pending/recorded/retained/failed/unknown. Mode: chat/completion/embedding/other. Tier: empty/weak/medium/strong.
- Closed reasons at BOTH request/attempt: empty, unknown, client-gone, cancel, regenerate, node-paused, proxy-shutdown, proxy-exited, target-left, transport, upstream-status, no-content, stream-interrupted, upstream-error, admission-paused, admission-draining, admission-busy, admission-no-fit, admission-engine-off, admission-wake-timeout, engine-not-running, request-invalid, no-owner, regenerate-exhausted, attempts-exhausted, deadline, request-ceiling, account-ceiling, account-cooldown, account-frozen, account-credential, account-exhausted, account-conflict, ledger-unavailable, deployment-quarantined, peer-framing. Map only landed typed I/admission/account facts; unmatched typed facts project literal unknown. No speculative hooks to emit unused codes. Invalid supplied enum strings fail validation; defensive display returns fixed Unknown without echoing input.
- At most 32 attempts, original indexes strictly increasing; droppedAttempts counts removed prefix. continuationOf must identify a real interrupted predecessor in I facts and be smaller than successor index. Within retained window validate predecessor outcome/link; reference into trimmed prefix must be below first retained index and below droppedAttempts, producer-verified against complete facts. Never guess links. Local account null; API account present. Served null until successful delivery, required for completed and equal to its delivered attempt destination. Terminal requires endedAt/recoverable=false; recoverable requires streaming and an interrupted attempt awaiting qualified continuation. Historical interrupted predecessors can remain in completed records.
- Workload null before association/API-only. Present origin/run equal request key; attemptIndex is owner-established local attempt; attached follows I's live registry index. Preserve historical detached references but never use them for cancellation.
- Ceiling = I tightest paid ceiling including refused candidates; granted = cumulative granted liability upper bounds (settlement does not reduce); held = current outstanding liability including uncertain/retained holds; charged = known settled charge. Accounting separately describes uncertainty/failure. Null when unavailable. Attempt amounts refer to that attempt's current hold/known charge. Immutable snapshots only; no sums from trimmed history, repeated-update addition, inferred refunds or unchecked arithmetic. Successful delivery with failed accounting remains completed delivery.
- One raw/decoded record at most 128 KiB, initial at most 32 MiB, raw depth at most 16 and member count at most 2048 per record including unknown members. Check before retaining/copying. Clone only bounded typed metadata, discard unknowns. These are presentation/decoder bounds, not inference body settings.

Decoder: focused nullable-aware history decoder token-walks raw JSON before map decoding; rejects trailing data, recursive duplicate keys including unknown subtrees, non-object envelopes, missing required keys, wrong types and size/depth/member overflow. At EACH known nested projection strip well-formed unknown members and validate known ones; allow null only at declared locations. Do not change decodeStrictContract or canonicalize null away. TS uses concrete JsonValue and same validation. Verify raw transport rejects duplicates before JSON parsing loses them; if not, reserve/add its exact raw ingress file to Task 3 before dispatch. Decode diagnostics: method plus literal category invalid-shape/invalid-value/duplicate-field/limit-exceeded/unknown-fields-stripped and count only, never raw decoder errors, keys or values.

Ordering: key is origin/run/request. Assign seq/snapshot under projection lock after immutable I fact read; notify outside I locks. Broker upsert returns accepted plus acceptedRecord; lower seq ignored, equal seq idempotent, terminal→active rejected. Push only acceptedRecord. Bridge/renderer/TUI apply same merge including rAF queue and delayed baseline. Subscribe before initial; merge baseline rather than replace pushes. Disconnect retires callbacks by connection generation, clears client rows; reconnect resubscribes/refetches. Broker history survives proxy restart but not broker restart. Accept private updates only from bound process/origin/run. Fence departed run before retirement; reject its late frames/snapshots; retire only its open presentation records failed/proxy-exited with incremented revision. Never infer delivery/cancellation/accounting from retirement. Replacement keys disjoint. Sort createdAt descending then origin/run lexical and numeric requestId descending (length then lexical); keep newest 256 everywhere. Immutable key/createdAt cannot change. Delayed stale snapshots cannot resurrect trimmed older rows.

Metadata clarifications: requested.model is empty only before I has validated a model; an early rejected request can retain that empty value, never raw invalid input. Destination models remain nonempty. Attempt indexes are contiguous from droppedAttempts. When an associated local attempt has been trimmed, its index must be in that dropped prefix; omit the card summary rather than guess its disposition. On seq overflow stop that record's projection with fixed limit-exceeded diagnostics, never alter inference.

Snapshot acquisition must not invert I locks: read owner facts from one serialized projection read path outside owner callbacks/locks, then assign seq; an older owner snapshot must not receive a newer seq. Use I's monotonic fact revision when available, discarding stale snapshots before assignment. Keep at most one bounded pending snapshot per retained key with a single IPC writer; coalesce superseded revisions. Never block requests on IPC, keep an unbounded queue or synthesize terminal state on diagnostic overflow. Terminal supersedes pending active snapshot.

Fixed reason copy for both clients (empty maps to empty; unknown/unrecognized display input maps to Unknown):

```text
client-gone: Client disconnected
cancel: Cancelled by user
regenerate: Regenerated
node-paused: Node paused
proxy-shutdown: Proxy shutting down
proxy-exited: Proxy exited
target-left: Node left the cluster
transport: Connection error
upstream-status: Engine returned an error status
no-content: No output before timeout
stream-interrupted: Stream interrupted
upstream-error: Engine error
admission-paused: Node paused
admission-draining: Node draining
admission-busy: Node busy
admission-no-fit: Model does not fit
admission-engine-off: Engine off
admission-wake-timeout: Engine wake timed out
engine-not-running: Engine not running
request-invalid: Request could not be adapted
no-owner: No node serves this model
regenerate-exhausted: Too many regenerations
attempts-exhausted: All attempts used
deadline: Request deadline reached
request-ceiling: Request spend ceiling reached
account-ceiling: Account ceiling reached
account-cooldown: Account cooling down
account-frozen: Account frozen
account-credential: Account credential failed
account-exhausted: Account exhausted
account-conflict: Account conflict
ledger-unavailable: Spend ledger unavailable
deployment-quarantined: Deployment quarantined
peer-framing: Peer sent a malformed frame
```

Recoverable interruptions suppress terminal stream-interrupted reason copy on request and attempt rows; show Streaming paused, waiting to resume. Actual linked predecessors show Interrupted, resumed. Terminal interruption copy follows only winning I4b disposition.

## Tasks

### Task 1 — Method-aware workload privacy projection

**Files:** `services/nvpair-ui-broker/workload_projection.go`, `services/nvpair-ui-broker/workload_projection_test.go`.
**Depends on:** none
**External prerequisites:** exact broker projection/test release against I/G/SR/NE, accepted repaired I3a identity schema. No I4b/I6 prerequisite.
**Complexity:** judgment
**Do:**

1. Add projectWorkloadFrame(method, params), returning accepted projection and fixed diagnostic category/count. Lifecycle methods workload:submitted/started/completed/errored require object params.workloadInfo; outer allowlist only workloadInfo. Nested Workload allowlist at baseline: id, model, engine, runId, state, originatedFrom, scheduledOn, createdAt, startedAt, completedAt, error, requesterId, requestedModel, seq. Compare against accepted I successor tags/consumers; amend before dispatch for legitimately added identity fields. Validate current required/nullable types and preserve optional absence. Strip unknowns at both levels.
2. workloads:remove uses separate top-level workloadId/originatedFrom shape at baseline. Include engine/run disambiguators only when part of the accepted I removal contract, with its exact optionality. No lifecycle envelope for removal. Reject null/wrong shapes/types, recursive duplicate fields, trailing JSON and invalid known identity fields. Malformed frames must not flow through stampNodeID's unchanged fallback.
3. Project before store/scheduler/client/manager effects, including activeLocalReplayFrames and scheduler replay, so old unknown persisted data cannot leak on rehydration. Preserve all legitimate identity/cancellation fields. No Workload/manager protocol expansion. Existing legitimate error/model strings are not made confidential by an allowlist; J never fills them with private diagnostics/upstream content.
4. Add TestWorkloadPrivacyProjection subtests for every lifecycle method, removal, outer/nested unknowns including secret-bearing synthetic key names, all legitimate optional/null values, identity preservation, malformed/duplicate known/unknown fields, restart/rehydration. Independent literal structural allowlists; malformed input yields zero store/scheduler/client/manager effects. Capture logs: no key names/values/raw errors.

**Verify:** Inspect broker TestMain/setup first; safe broker `go test -count=1 -v -run '^(TestWorkloadPrivacyProjection|TestRouteProxyWorkload|Test.*Replay|Test.*Rehydrat)' .`, supported race repeat, then vet from services/nvpair-ui-broker. New subtests execute without skips and positive legitimate-field controls pass. No services/tests invocation; cluster proof belongs to Task 6.

### Task 2 — Nullable validated history schema and compatibility corpus

**Files:** `services/shared/nodepolicy/routing_wire.go`, `services/shared/nodepolicy/routing_wire_test.go`, `services/shared/nodepolicy/testdata/routing-request.json` (new).
**Depends on:** none
**External prerequisites:** I6 contract join and exact wire/test release; SR5a/NE shared-wire release.
**Complexity:** judgment
**Do:** Implement Executable view contract with focused decoder, preserving I active/cancel/key types. No RequestCeiling helper, unstamped transport or undefined ValidateStamped. Corpus schema 1 uses synthetic IDs/timestamps, runtime engine key llamacpp, explicit nullable fields: completed rejected→served local (350 ms admission/1200 ms first byte); terminal interrupted API with charge 123456789, ceiling 1000000000, granted/held 250000000 and retained accounting; cancelled with droppedAttempts 2/retained index 2; recoverable interruption; actual linked successor. Recovery examples are contract fixtures, not claims SC-b exists.

TestRoutingHistoryContract covers semantic round-trip, every required/null position, IDs/metadata/numeric/money bounds, all enums at request/attempt/account levels, mutually exclusive destinations/account kinds, invalid continuation including trimmed prefix, recursive duplicate/unknown fields, valid unknown stripping and fixed-category errors without raw text. Keep prior admission/key strictness tests unchanged. No JSON byte ordering assertions.

**Verify:** From services/shared, `go test -count=1 -v -run '^TestRoutingHistoryContract' ./nodepolicy`, supported race repeat, full nodepolicy tests/vet and SPDX checks pass. Every malformed fixture rejected; semantic nulls/values preserved. Schema-only commit introduces no method/emitter and does not claim J0 completion.

### Task 3 — Atomic private history RPC integration from I facts through consumers

**Files:** `services/nvpair-proxy/routing_events.go` (new), `services/nvpair-proxy/routing_events_test.go` (new), `services/nvpair-proxy/routing.go`, `services/nvpair-proxy/request_spend.go`, `services/nvpair-proxy/http_handler.go`, `services/nvpair-proxy/proxy.go`, `services/nvpair-ui-broker/routing_events.go` (new), `services/nvpair-ui-broker/routing_events_test.go` (new), `services/nvpair-ui-broker/broker.go`, `services/nvpair-ui-broker/client_requests.go`, `services/nvpair-ui-broker/engineproxy.go`, `services/nvpair-ui-broker/proxy.go`, `desktop/src/shared/types/routing.ts`, `desktop/src/shared/types/ws-channels.ts`, `desktop/src/electron/service-bridge/routing.ts`, `desktop/src/electron/service-bridge/modular-state.ts`, `desktop/src/electron/service-bridge/modular-supervisor.ts`, `desktop/src/electron/service-bridge/empty-handlers.ts`, `desktop/src/ui/api/pair-api.ts`, `desktop/tests/modular/routing-projection.test.ts` (new), `services/nvpair-tui/ui/routingwire.go` (new), `services/nvpair-tui/ui/routingwire_test.go` (new), `services/nvpair-tui/README.md`, `services/nvpair-proxy/README.md`, `services/nvpair-proxy/spec.md`, `services/nvpair-ui-broker/README.md`, `desktop/docs/service-contract-exceptions.json`, `desktop/docs/services-api.md` (generated), `desktop/docs/services-backend.md`, `desktop/docs/frontend-api.md`, `desktop/docs/architecture.md`, `desktop/docs/services-parity.md`.
  `desktop/src/electron/service-bridge/json-rpc-client.ts`, `desktop/src/electron/service-bridge/routing-json.ts` (new), `desktop/tests/modular/json-rpc-client.test.ts`.
**Depends on:** 1, 2
**External prerequisites:** integrated I3b/8/4a/4b/5/6 and exact releases; G3/SR5c/SR6a/PA/NE/E4 shared paths/generator transferred per table. No I7 proof prerequisite.
**Complexity:** judgment
**Do:**

1. Mandatory pre-dispatch reconciliation: inspect accepted I successor with git grep on routingRequest/routingSpend/publicAttemptLifecycle and I4b cutover report. Record sole request/attempt/sink snapshot symbols/paths. Listed proxy routing.go/request_spend.go/http_handler.go/proxy.go are candidate observation sites: remove unused paths, replace moved paths with actual owners/focused tests, name exact observation hooks. Resolve raw JSON duplicate accepting boundary and process-exit fence. Reserve each changed path, amend Files and rerun graph before code. Missing typed fact blocks dispatch; do not invent another owner. No inferred permission to edit cancellation/ingress/adapter/transport/sink authority.
2. Add a small typed allowlisted read-only projection alongside owners. Cover EVERY registered inference request from registration through no-owner/adaptation/admission/budget/cancel/success/hybrid exits; pre-registration identity/capacity refusals explicitly outside history. Workload null until I association, detached according to I index. Inputs are bounded validated IDs/model, typed codes, waits, terminal/recoverability/delivery/accounting facts, never raw errors/request-response maps/accounts. Expose missing immutable facts at I owner without another state machine. No ID counter, ceiling calculator, reservation accumulator, abortReason or workload finalizer. Diagnostics copies/emits outside inference locks with bounded storage; inference control never waits for diagnostics IPC.
3. Emit full record as private routing/request params, with Ordering rules. Instrument real I5 hybrid facts, no 501/unreachable placeholder. Repeated snapshots replace displayed accounting; no double addition. I4b disposition controls terminal state. Delivery-success/accounting-failure stays completed plus failed accounting, never resend or rewrite into inference failure.
4. Broker accepts only exact local bound process/origin/run. Route before generic proxy echo; never manager/scheduler/peer. Bounded history upsert publishes accepted snapshots only. Add public local dispatch switch cases routing:subscribe/unsubscribe/get-initial returning initial shape; routing:changed push to subscribers. **Never put these three public methods in internalClientMethods.** Keep history flag separate from active discovery. Deny routing/request and private worker methods from client injection before generic relay; history has no direct/reverse peer route. Test real request dispatch, not helper-only invocation.
5. Apply departed-run fencing and client generation/baseline reconciliation exactly as Ordering. Capture old handle, never resample replacement. No global failOpen across runs or inferred accounting/delivery terminal authority.
6. Extend I6 TS routing types/bridge/API/channels/empty handlers, preserving active/cancel consumers/tests. History API subscribe/unsubscribe/getInitial/onChanged follows existing transport conventions; parser uses concrete JsonValue, closed unions, nullable validation and safe fixed diagnostic categories. At baseline json-rpc-client.ts:104–107 parseMessage calls JSON.parse, losing duplicate keys. In this exact raw ingress, before dispatch/response resolution, validate history raw frames with focused routing-json.ts token scanning (escaped names compared after decoding, duplicate rejection recursively including unknown members, bounded size/depth/member count). Recognize routing:changed by method and initial results by pending routing:get-initial request ID; retain pending method metadata for that check, without a new transport. Other method parsing stays unchanged. Malformed history drops with fixed category; initial rejection resolves through existing bounded error handling, never hangs. Add raw notification/result tests in json-rpc-client.test.ts and projection tests, covering escaped duplicate names, nested unknown duplicates and safe log capture. State/supervisor push only accepted-upsert results. No unused exports/routine dead-code omissions, casts/any/unknown signatures or renderer Electron imports.
7. TUI routingwire.go reuses shared Go decoder/types, method constants, initial decoder, merge/format helpers. No duplicate handwritten wire schema. Fixed-category decode logging, never decodeStrictContract raw error. Corpus semantic tests; rpccmd.go remains unchanged because existing bounded calls/notifications suffice.
8. Add TestRoutingHistoryProjection (proxy), TestRoutingHistoryRPC (broker), TestRoutingHistoryOrdering in affected Go packages, TestRoutingHistoryWire (TUI), routing-projection.test.ts. Actual subscribe/unsubscribe/get-initial dispatch, private/client/peer rejection with zero effects, no manager/scheduler routing, all early exits, repeated accounting, rejected paid tightening, multiple attempts, cancel during durable reserve, uncertain holds, completed delivery/accounting failure. Barrier tests force lower/equal seq, stale initial after push, late active after terminal, old run after replacement, reconnect stale callback and cap eviction. Preserve I cancel/active regressions.
9. Synthetic sentinel tests put distinct markers in request messages, upstream response/error bodies, credentials/endpoint secrets, PINs, key-like values and unknown key names. Capture proxy/broker/client/cluster diagnostic emissions plus logs across success/rejection/cancel/malformed decode/ledger failure. Require absence everywhere in these captured surfaces with positive path controls. Raw inference traffic is not a diagnostic surface and naturally contains its own synthetic payload. Policy ID fields intentionally remain visible; no raw account objects or real secrets.
10. Same accepted change updates producer spec/README, broker method docs, desktop API/transport/architecture/parity and TUI wire-consumer README. Describe local-only visibility, memory/restart loss, caps, amount semantics, early coverage and SC-b limitation. Exact private exception for routing/request: proxy to its origin-local broker only; projected as routing:changed, never client invoke or cluster method. Inventory tracked proxy/broker MANIFEST files with `rg --files services | rg 'MANIFEST[.]md$'`; reserve/add only a matching method-inventory path before dispatch. No engine-manager manifest edit. One coordinated writer runs `npm --prefix desktop run build:modular-binaries`, `service-contracts`, `service-contracts:write`, `typecheck`, `service-contracts:check` (each named npm script); regenerate again with no further diff. Never hand-edit services-api. Producer/relay/all consumers/bridge/tests/docs land as one accepted contract change; partial implementation commits are not independently complete delivery.

**Verify:** Inspect affected module fixtures, run focused new tests `-count=1 -v`, supported race repeat/vet in shared/proxy/broker/TUI, then full isolated affected-module suites. Desktop lint/typecheck/unit/dead-code/contracts and modular binary build pass. Actual broker methods reachable; private/peer injection zero effects; all redaction captures and ordering barriers pass; I active/cancel regressions remain green. No services/tests execution on Mac/MSI. Unavailable native/race gates remain unverified; missing extra Files requires reservation/amendment.

### Task 4 — Desktop history and identity-safe workload summary

**Files:** `desktop/src/ui/stores/routing.store.ts` (new), `desktop/src/ui/stores/init.ts`, `desktop/src/ui/utils/routing-labels.ts` (new), `desktop/src/ui/components/RoutingSettings/RoutingHistory.tsx` (new), `desktop/src/ui/components/RoutingSettings/RoutingAttempt.tsx` (new), `desktop/src/ui/components/RoutingSettings/RoutingSettings.tsx`, `desktop/src/ui/components/Workloads/WorkloadItemCard.tsx`, `desktop/tests/modular/routing-view.test.ts` (new).
**Depends on:** 3
**External prerequisites:** exact UI/store/card releases; load repository kaizen-ui guidance.
**Complexity:** judgment
**Do:**

1. Use workloads.store/init lifecycle precedent with subscribe-before-initial, generation fencing, Ordering merges including rAF queue and 256 cap. Full request key helper is renderer-safe. Workload matching compares originatedFrom/runId/engine/id against owner association; missing run never matches. Detached association shows only its associated local attempt history, not later API liveness/destination. Test reused IDs across origins/runs/engines and local→API detachment.
2. Labels: Routing/Streaming/Completed/Failed/Cancelled; outcomes In progress/Served/Rejected/Failed/Re-dispatched/Cancelled. Interrupted predecessor shows Interrupted, resumed only with actual successor; otherwise Interrupted. Stream interrupted only for authoritative terminal no-recovery failure; recoverable streaming summary Streaming paused, waiting to resume. Completed summary `Routing: <n> attempt(s), served on <destination>`; failed `Routing failed: <fixed label>`; cancelled `Routing cancelled: <fixed label>`; routing `Routing: attempt <n>`. Use the exact Fixed reason copy table above, never raw text. No pause inferred just from last attempt outcome.
3. Money string arithmetic: 0→$0, 1→$0.000000001, 123456789→$0.123456789, 1000000000→$1, null→—. Wait null→—, 350→350 ms, 1234→1.2 s. Separately label ceiling/cumulative granted/current held/charged and accounting status; never call granted held. Show dropped attempts `(+<n> older)`.
4. Mount last in existing RoutingSettings: Routing history; subtitle `Private to this machine. Kept in memory; broker restart clears history. Newest 256 requests.` Empty `No routed requests yet.` Disconnected explicit. KUI Table/Collapsible supported expansion and keyboard-focusable Show/Hide attempts, existing settings tokens/classes, no new page/CSS system. Columns When/Model/State/Attempts/Served/Spend/Reason. At narrow widths use inner horizontal scroll/minimum readable widths, wrap expanded rows, ellipsis long IDs with full accessible text; never page overflow. Attempt detail shows destination IDs/model, outcome/fixed reason, admission/first-byte waits, HTTP status, account state and separate held/charged amounts.
5. Card adds plain summary for exact association only; associated local attempt authoritative failure controls error text, no recoverable interruption error styling, no workload-state rewrite/cancel action. Render corpus, long IDs, empty/disconnected, keyboard expansion and completed delivery/accounting failure.

**Verify:** Desktop lint/typecheck/unit/dead-code pass; routing-view.test.ts covers labels/formatters, stale initial/push/reconnect/rAF, cap/identity/detachment. Fixture render inspection at 360/1200 CSS pixels shows accessible expansion/readable scroll/no page overflow. Static check: no Electron renderer imports, casts/any/unknown signatures, float money. No npm start/OpenCode/engine operation is required.

### Task 5 — TUI Routing tab and parity documentation

**Files:** `services/nvpair-tui/ui/routingview.go` (new), `services/nvpair-tui/ui/routingview_test.go` (new), `services/nvpair-tui/ui/ui.go`, `services/nvpair-tui/README.md`, `docs/terminal-interface.mdx`, `desktop/docs/services-parity.md`.
**Depends on:** 3, 4
**External prerequisites:** Task 4 settles display semantics before parity publication. E4/OB exact doc release, ui.go navigation release/recount; no pending E4 tab prerequisite.
**Complexity:** judgment
**Do:** Implement actual View including SetSize/Help bindings. Add Routing after Jobs, use jobs call/subscription/nodeNamer/table conventions, r refresh/enter attempts. Subscribe then fetch, Ordering merge/generation fence, disconnect clear/Disconnected, reconnect resubscribe. Bound rows/attempts/width via fitTable/joinLines; at narrow widths retain Model/State/Attempts and hide lower-priority columns while detail remains available. Task 4 wording/formatting exactly, explicit account IDs/amount meanings, pause only from recoverability. Footer `Private to this machine · memory only; broker restart clears history · newest 256 requests`. Recount maxTabs. No rpccmd rewrite. Docs explain parity/SC-b limit and preserve E4/OB content.

TestRoutingView subtests cover corpus, expand, 256 cap, disconnect/reconnect stale results, stale baseline/lower seq, terminal no-reopen, real successor/recoverable/no-recovery wording, long IDs/narrow widths, accounting-failed completed delivery. Shared decoder owns validation.

**Verify:** From services/nvpair-tui run `go test -count=1 -v -run '^TestRouting' ./ui`, full isolated tests/vet and supported race repeat, no skips. Width 40/80/120 fixtures stay bounded and enter expands selected row. SPDX/docs checks pass. Fixtures do not establish native TUI acceptance.

### Task 6 — PC3-only outbound privacy proof with the actual fixture

**Files:** `services/tests/workload_interop_test.go`.
**Depends on:** 1, 3
**External prerequisites:** accepted join12 I3a repair; XA/I7 exact fixture transfer; separately reserved isolation/port helper repairs and fixed-path PC3 interval.
**Complexity:** judgment
**Do:**

1. Add TestWorkloadBroadcastCarriesNoRoutingFields beside TestWorkloadManagerOutboundBroadcast, using actual newInterNodeCluster/pinned direct peer/fixed-path bridge/settings journal/inventory pattern. Keep fake Ollama dialect POST /api/chat with synthetic model/messages and upstream `{"done":true}`. No installed Ollama; compatibility fixture is not daily-engine acceptance. Do not substitute Chat Completions without its descriptor/route; no substitution planned.
2. Readiness barriers before sending/capture. Require nonempty capture and submitted/terminal positive controls for exact origin/run/engine/workload, status 200/correct model/state. Drain through terminal/barrier, not timing-only absence. Structural envelope/nested allowlist assertions independent of implementation, no arbitrary substring blacklist over legitimate model/error values. No private routing method in peer capture. Exercise malformed/unknown injection and restart replay through released existing seams with positive replay control; exhaustive lifecycle/removal matrix belongs to Task 1. Missing process injection/restart seam requires separately reserved exact repair, never fabricated coverage.
3. XA isolation: no production discovery/advertisement on owner's LAN; explicit absolute temp app/journal/data paths outside repository/vault; pinned synthetic peer, registered empty scanner where discovery unnecessary. Loopback/temp state alone insufficient. EVERY fake engine/facade/peer/worker/observer listener rejects **8080, 11434, 1234, 14318–14323** before serving/dialing, including httptest/child defaults. Unchecked temporaryClusterPort insufficient; coordinator supplies accepted exact helper repair/dependency before dispatch.
4. PC3 ONLY, inspect TestMain builds before invocation; no Mac/MSI casual list/run. Reserve serialized interval free of GPU/measurement work. Read TESTING_ON_PC section 10; GOFLAGS includes nvpair-test-exec, absolute NVPAIR_TEST_BIN_DIR, exact executable/rule coverage from scripts/windows-test-bin.ps1 list including bridge/children. Missing elevated rule leaves gate pending, no temporary-path bypass. Record before/after firewall/survivor/listener cleanup, names/skips/provenance. Screen observation UNMONITORED unless attended; unchanged rules do not prove no prompt. No owner prompt-report request.

**Verify:** PC3 reserved interval only, services/tests: `go test -p 1 -parallel 1 -count=1 -v -run '^(TestWorkloadBroadcastCarriesNoRoutingFields|TestWorkloadManagerOutboundBroadcast)$' .`; then coordinator-approved whole `go test -p 1 -parallel 1 -count=1 -v ./...` after whole-suite isolation audit. New test executes without skip, both positive controls/structural assertions pass, no installed service/model contact or survivors. PC3 no -race; MSI never services/tests for J. Unavailable whole-suite isolation stays pending, not waived.

## Verification (whole-plan)

Record integrated SHA, OS/arch, command, executed names/skips, fixture provenance/results. Read rules/specs, project-home orientation/blockers and TESTING_ON_PC before implementation/native tests. Coordinator owns worklog/project-note updates outside this design's single-file scope.

- Mac: inspect module setup; safe affected shared/proxy/broker/TUI unit/race/vet plus desktop lint/typecheck/unit/dead-code. Retain make check (desktop tests, not services/tests; inspect current targets first). Task 3 modular binary build/contracts generation/check, SPDX/diff whitespace. Never make test, services/tests cases/TestMain discovery, model start, npm start or live inference on Mac.
- PC3 owns Task 6 services/tests runtime/whole-suite acceptance and serialized affected-module plain native tests under fixed-path/firewall preflight; no race. MSI supplies supported affected-module race in reserved interval, never J services/tests.
- Native Electron/TUI interaction, actual OpenCode llama.cpp/Strata routing, billing, GPU/memory, multi-machine timing and pause-resume require separately scheduled owner-authorized TESTING_ON_PC gates. Fake tests establish none. I4b supplies no SC-b implementation; J cannot close continuation acceptance.
- Eventual requested PR declares binary bumps for proxy/broker/TUI (confirm compiled dependency reach) and user-visible history release intent. No PR merely for this plan. Never hand-edit versions/changelog/generated services-api.

## Do not

- Do not execute implementation, edit code/sibling plans/vault, commit or push during this amendment.
- Do not duplicate I identity/registry/index/cancellation/accounting/terminal/sink authority, speculative hooks, discovery-origin repair, unchecked money addition or completion from HTTP 200/handler/workload state.
- Do not add private Workload fields or forward history to manager/scheduler/peer; no raw errors/unknown keys/secret content in diagnostics/logs. Allowlist does not sanitize legitimate arbitrary workload strings.
- Do not weaken admission/key strictness, hand-edit generated docs, change activation, use TS casts/any/unknown signatures/dynamic imports/renderer Electron imports. Use static @/ cross-directory imports and concrete JsonValue.
- Do not refactor beyond tasks, bypass reservations/isolation or let presentation caps truncate I authority.

## If stuck

Missing typed facts, incompatible I successor, unbound identity, absent raw duplicate enforcement, extra Files/shared fixture repair: stop dependent task and report exact symbol/path/SHA for coordinator release/plan amendment; continue released disjoint work. No raw error/workload-state/second-accumulator fallback. Reconcile contradictory PA/I gates before overlapping dispatch. Skips/missing machine/toolchain/firewall/live authorization are unverified gates. Fix readiness/isolation rather than weaken assertions; diagnostics refresh/reconnect never authorizes inference replay.

## Amendment 2026-10-21

Applied R1–R8, LOW and UNVERIFIED handling. Replaced obsolete I assumptions with owner snapshots/reservation transfers, corrected method dispatch/nested sanitizer, defined nullable schema/redaction/accounting/ordering/correlation and consolidated contract delivery in Task 3. Recorded coordinator-adopted owner answers; retained versioned semantic corpus and PC3-only actual-fixture proof, separated native/live acceptance. Numbering replaces earlier decomposition. Only Task 1 can precede I4b/I6, subject to exact release and repaired identity schema. No product tests/builds/native gates ran. Missing Note lifecycle is intentional for this repository run package under the amendment request. Audit fork candidates are prohibitions, enumerations, owner-veto rights and objective evidence conditions, not delegated implementation alternatives; count candidates are absent. The graph audit must remain clean after the required post-I Files reconciliation.
