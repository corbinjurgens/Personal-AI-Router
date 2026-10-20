<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution plan: PA — paused-node authenticated advertisement

Planning baseline: `origin/integration/join12`,
`81960f6201e296d7f78274b6e8515ed0c6a8d8d5`. Source verified with `git show`
and `git grep`, without a checkout. References below use file/symbol names,
not historical line offsets. Re-resolve them at the coordinator's integrated
I/G landing before dependent implementation. Path shorthand: `P/` =
`services/nvpair-proxy/`, `S/` = `services/nvpair-node-scanner/`, `B/` =
`services/nvpair-ui-broker/`, `E/` = `services/nvpair-engine-manager/`,
`N/` = `services/shared/`, `X/` = `services/tests/`.

## Goal

Carry the existing revisioned `policy.pausedBy` advertisement into proxy
routing metadata. When inference candidate resolution observes a paused remote
peer, exclude that peer before host deduplication and admission to the result
set, with zero dispatch attempts to it. A later available advertisement makes
the same peer eligible again. Preserve destination admission and local veto
for stale/missing hints, stable model catalogue and declared tier aliases,
self/dormant-self, manual-no-hint, model-list and pass-through behavior.
No availability data enters mDNS. Tests prove each hop and the dynamic proxy
replacement; a PC3-only process gate proves scanner-to-broker propagation.

## Change scope

Intent / Scope: **Bounded improvement**. Product edits add internal metadata
to proxy `Node`, copy it in `subscribedToNode`, and exclude paused remote
inference candidates at I's final common candidate-selection owner. At join12
that owner is `facade.resolveCandidates` in `P/proxy.go`; do not assume I leaves
it there. All other PA changes are focused tests and documentation.
Basis: the handoff and R1 review retain the existing advertisement pipeline;
I owns dispatch, attempts, retry, cancellation and hybrid activation.
Compatibility: `Node.PausedBy` is `json:"-"`, ignored by `nodeEqual`, like other
routing metadata. No JSON-RPC, HTTP, persisted-policy, descriptor or TXT shape
changes. Peers without a policy/pause hint retain existing behavior. Concrete
empty candidates retain 502; empty tiers retain 404. A PA-only patch requires
no generated service-contract change.
Non-goals: scanner push/cache/revision ordering, new all-paused 503, retry or
cancellation redesign, destination veto changes, engine-manager/shared-wire
product edits, UI work, SR residency, XA's remote-configuration matrix, OB
Off-origin display, NE launch behavior, live engines in mandatory tests.

## Context the executor needs

Read the full `reviews/pa-plan-review-r1.md` and handoff
`git show origin/plan/cloud-handoffs:plans/handoffs/pa-paused-advertisement.md`.
The current coordinator instruction supersedes the handoff's historical
branch/push instructions: this amendment changes only this package plan.
Read sibling `i-unified-dispatch.md`, `sr5-rescope.md`, amended
`xa-remote-config-acceptance.md` and `ob-off-origin.md` in `packages/`.

Verified facts at join12:

- `N/nodepolicy/advertise.go`, `Policy.Advertisement`, already carries
  `Revision`, `PausedBy`, roles and served models. Broker
  `persistAvailabilityLocked`, `pauseNode`, `resumeNode` and
  `publishNodeProjection` publish the adopted saved revision/provenance.
  `B/availability.go::setAvailabilityAs` also publishes a pending local-pause
  veto projection **before** persistence. A first notification/count threshold
  alone does not prove the completed transition's advertisement.
- `B/policy.go::pausedLocally` and `E/nodeprojection.go` retain local veto;
  remote configuration edits remain permitted while paused. Reuse
  `TestPauseProvenanceIsNeverDowngraded` and
  `TestConfigurationEditsStayAllowedWhileLocallyPaused`; PA does not duplicate
  OB/XA acceptance.
- `E/models.go` serves the whole node projection as `policy` in `/v1/models`.
  `TestNodeProjectionServedWithTheModelInventory` already reads a paused
  projection back over HTTP. No engine-manager edit is needed.
- `S/model_enrichment.go::modelsClient` selects pinned cluster mTLS for
  non-loopback peers; unpinned LAN peers are refused. **Any loopback address**
  selects the plaintext `modelsHTTP` client without checking self UUID.
  `TestModelFetchTransportFollowsMembership` proves transport selection,
  not an end-to-end authenticated advertisement fetch. The loopback process
  case below proves propagation, not authentication.
- `decodeInventoryPolicy`, `directory.applyModels` (`S/directory.go`) and
  `refreshNodeModelsCandidates` compare policy with deep equality, update
  caches/directory and emit `discovery:node-updated` for policy-only changes.
  Failed fetches retain last good inventory/policy indefinitely. Existing
  `TestRefreshPolicyAdvertisement` covers this retention.
- `S/daemon.go::peerRefreshInterval` is 15 seconds. `refreshPeersLoop` does
  other work before `refreshModelsOnce`; that function waits for its bounded
  batches to finish. Periodic sweeps cannot overlap. Asynchronous
  `enrichModelsCandidates` and peer-services-triggered enrichment can interact
  with refresh; existing route-generation/incarnation guards remain. No hard
  end-to-end propagation deadline follows from the ticker.
- `B/discovery.go::directoryToEnriched` relays policy intact;
  `P/facade.go::replaceSubscribed` decodes discovery snapshots.
  `P/discovery.go::Node` has routing-only `PolicyRevision`, roles, served
  models and strict capability but no `PausedBy`. `nodeEqual` ignores routing
  metadata. `P/proxy.go::subscribedToNode` currently drops the pause hint.
- `facade.resolveCandidates` handles concrete and tier eligibility; its `add`
  closure classifies self, pinned peer, manual node, then excludes unsafe
  self-targets and deduplicates hosts. Selected/priority ordering already
  determines calls to `add`. `P/facade.go::targetURL` uses `targets.Prefer`,
  not a blocking TCP probe; correct the stale nearby comment when editing
  the owner. Empty-model enumeration must not apply the pause exclusion.
- `P/tiers.go::declaredRoles` and catalogue construction preserve configured
  intent. `hybridLocalCandidates` uses tier eligibility, but hybrid dispatch
  is feature-unavailable at join12. Verify shared eligibility after I's hybrid
  cutover; do not claim join12 exercises active hybrid dispatch.
- `P/http_handler.go::handleHTTP` retains concrete-empty 502
  `no available node advertises the requested model` and tier-empty 404
  `no eligible model for tier`. `P/admission.go` retains destination 503,
  `X-PAIR-Admission: paused`, `Retry-After: 30`. I3a's bound origin/run
  identity must survive fixtures; reuse `newTestProxy`/`newBoundProxy`, not a
  raw identity-less `NewProxy` handler.
- `N/noderec/noderec.go::NodeRecord` and `S/registry.go` have no availability
  TXT field. Desktop decoding/types know `pausedBy`; join12 does not establish
  desktop node-card `Paused (local)` / `Paused (peer)` labels. TUI availability
  copy does not justify a desktop claim.
- `X/broker_test.go::startBroker` starts a production scanner without service
  port overrides. `TestModelsRefreshConvergesWithoutMDNSChange` uses real mDNS.
  Temporary state directories and stopping installed PAIR do not isolate LAN
  discovery. `X/fixture_ports_test.go::reserveFixtureServicePorts` and its
  `.args()` provide checked broker service ports, not discovery isolation.

No Go, desktop, process, native Windows or live engine tests were executed
while amending. Post-I owner layout, fixture isolation and Windows exact-path
rule coverage remain execution gates, not verified facts.

### External reservations and transfer protocol

I2/I3a and E4 Task 2 are in join12. I3b/4a/4b/5/6/8 are in flight on MSI.
Before PA Tasks 1/2 write any proxy file, the coordinator must record the
integrated successor SHA and release **all** of these reservations:

- I3b and I8/4a finish their proxy identity/buffer writes; I4b lands the sole
  request owner and releases `P/proxy.go` and its final candidate owner
  (`P/routing.go` when moved); I5 lands hybrid selection and releases
  `P/tiers.go`, `P/routing.go` and candidate fixtures. I6 finishes any further
  `P/proxy.go` writes and releases them. An explicit coordinator transfer of
  these exact files/seams can substitute for a remaining task landing only
  when that task's amended reservation excludes PA's write interval.
- G3's integrated broker relay and I's completed broker contract join must be
  recorded, following sibling order I3 -> G3 -> SR5c, with I8/6 overlapping
  broker writes completed before transfer. PA does not reverse this order.
- Reconcile I's stale prerequisite “PA before I3–5” before dispatch: it is
  superseded by **I owner cutover/release -> PA**. The coordinator must remove
  that contradictory gate in I's reservation ledger/plan. Do not dispatch
  either side with a cross-package cycle. This amendment does not edit I.
- PA Tasks 1/2 reserve metadata propagation and pause eligibility as one small
  serial patch at the final owner; no second candidate/attempt path. Re-read
  that owner, fixtures, identity envelope and newly occupied test filenames
  at the recorded SHA. If I moved files, amend exact Files below and rerun
  the graph audit **before** writing; no guessed extra-file permission.
- Task 3 waits for I6 and SR6a to publish and release `P/README.md` and
  `P/spec.md`. Task 5's `B/README.md` waits for I6/G3/SR documentation release.
  Task 6 reserves `X/broker_test.go` against I7/XA/shared fixture lanes.
  Tasks 4/5 and independent product-document authoring proceed only with
  exact-file release; Task 7 finalizes after Task 2 semantics land.

## Decisions (settled — do not revisit)

- Local Off/pause veto and destination admission remain authoritative.
  Remote configuration edits remain permitted; PA adds no permission path.
- Mac never serves models. `services/tests` case execution is **PC3 only**,
  never Mac, Linux or MSI. Mac can vet/compile/list that module; PC3 never
  uses `-race`. MSI/Mac unit modules supply race evidence.
- No availability in mDNS, no new wire format, no push, cache or configuration.
  LAN hints use the existing authenticated path with the documented loopback
  exception. Revisioned means persisted policy revision beside provenance.
- Snapshot semantics: exclusion applies when the resolver observes the hint.
  It cannot revoke a dispatch selected before an update. I retains attempt,
  retry and cancellation ownership; stale available hints meet destination
  rejection, and stale paused hints delay re-entry until successful refresh.

Coordinator-adopted owner answers (2026-10-21; owner may veto):

1. Keep existing empty-candidate responses: concrete 502, tier 404; no new
   all-paused 503/header/body in PA.
2. Accept advisory polling: nominal 15-second refresh cadence under healthy
   successful refresh, with preceding work/batching/fetch time; no hard
   one-sweep bound. Failed refresh retains last good state indefinitely.
3. Keep a stable catalogue: paused peers stay in model lists and declared tier
   aliases; pause changes dispatch eligibility only.
4. Add no revision-order guard. Periodic sweeps are serialized; asynchronous
   enrichment is the relevant interaction. A bare monotonic revision check
   would need reset/incarnation semantics and is outside PA.

## Decisions made while writing this plan

- Add `PausedBy nodepolicy.PauseOrigin` with `json:"-"`; do not compare it in
  `nodeEqual`. Any nonempty advertised pause provenance excludes a remote
  inference peer; both local and peer origin count.
- Apply the guard after self/pin/manual classification and defensive self
  checks, before host deduplication/result append. At join12 the condition is
  `model != "" && !self && n.PausedBy != ""`. Production manual nodes carry no
  advertisement. Do not manufacture paused manual records as a substitute
  for the required pinned-peer test.
- New focused tests live in the listed availability files. Reuse existing
  helpers without modifying unrelated tests. Helper-created listeners used
  by PA need the same checked-port discipline as direct listeners.
- Per-candidate debug metadata (`node_id`, `paused_by`, `policy_revision`) is
  optional diagnostics, not an acceptance oracle. No new metrics are needed.
- Task 6 remains gated on coordinator-owned network/child-listener isolation;
  PA does not invent scanner product flags or a new generic harness to unblock
  it. Live engine evidence is a separate optional coordinated gate below.

## Tasks

### Task 1 — The proxy's routable Node carries the pause hint

**Files:** `services/nvpair-proxy/discovery.go`, `services/nvpair-proxy/proxy.go`, `services/nvpair-proxy/availability_test.go`

**Depends on:** none

**Complexity:** mechanical

**Do:** After the proxy reservation transfer above, add
`PausedBy nodepolicy.PauseOrigin` with tag `json:"-"` beside `PolicyRevision`
in `Node`. Extend the routing-metadata comment on `nodeEqual`, leaving its
body unchanged. In `subscribedToNode`'s nonnil policy block copy
`node.PausedBy = n.Policy.PausedBy`; describe it as an advisory advertisement
with a LAN authentication path and loopback exception, not universally pinned.
Create `availability_test.go` with SPDX header and `package main`:

- `TestSubscribedToNodeCarriesPausedBy`: independent engine-profile cases
  for nil policy, revision without pause, local pause, peer pause; assert
  complete expected provenance/revision and unchanged other metadata.
- `TestPauseHintAnnouncesNoNodeUpdate`: `Discovery.SetSubscribed` transitions
  available -> paused -> available, changing only revision/provenance.
  Assert no discovered/updated/removed events on replacements and assert the
  stored snapshot changes each time. Reuse
  `TestSubscribedCarriesPolicyWithoutChurn` as precedent.

Check setup/conversion/decode errors. Inspect filenames after integration;
extend an existing PA-owned file instead of overwriting another owner's work.

**Verify:** Mac: `go -C services/nvpair-proxy vet ./...` and
`go -C services/nvpair-proxy test -race -count=1 -v -run '^(TestSubscribedToNodeCarriesPausedBy|TestPauseHintAnnouncesNoNodeUpdate)$' .`.
Both named tests must execute and PASS without skips/races. Root
`node scripts/spdx-headers.mjs` passes. Diff is confined to declared Files.

### Task 2 — Shared inference eligibility excludes paused peers dynamically

**Files:** `services/nvpair-proxy/proxy.go`, `services/nvpair-proxy/availability_test.go`

**Depends on:** 1

**Complexity:** judgment

**Do:** Reconfirm I's final common owner/release and exact Files as above.
Insert the agreed pause exclusion at that owner before host deduplication
and result append, preserving self admission and empty-model enumeration.
Correct the nearby stale “TCP-probed” targetURL comment to match
`targets.Prefer`; do not introduce a probe. Update the function comment to
say paused **remote peer inference** candidates are excluded, with self,
manual-no-hint and non-inference exceptions. Preserve I's identity, response,
request/attempt, retry, cancellation and hybrid mechanisms.

In `availability_test.go`, use `forEachEngine`, bound-origin fixtures and
`clustertrusttest.Join`/`clustertrust.Open` from the final subscribed-peer
tests. Maintain pinned-peer fixtures; no manual-node fallback. All new HTTP,
TLS, fake-engine and self-facade listeners bind `127.0.0.1:0` and reject ports
8080, 11434, 1234, 14318–14323 before use. Build local checked listeners and
start unstarted httptest servers on them. For helper-created listeners that
cannot accept such a listener, keep a PA-local fixture in this file; do not
alter unrelated helpers. Use the allocated facade port consistently for
self identity, selfNode and proxy construction, never profile defaults.

Add top-level tests with these mandatory assertions:

- `TestPAPausedPinnedPeerEligibility`: selected/priority paused pinned peer
  plus available pinned peer; concrete and tier resolution omit paused local
  and peer origins. HTTP request counters show zero attempts to paused peer
  and success through the eligible peer. Preserve bound origin/strict
  capability setup so identity refusal cannot masquerade as pause exclusion.
- `TestPAPausedPeerDynamicResume`: on the **same pinned peer**, replace the
  subscribed advertisement available -> paused -> available with new
  revisions, using the discovery replacement path. Next concrete and tier
  requests skip while paused (zero counter delta) and dispatch after resume
  (positive counter delta); metadata replacements emit no outward
  `node/updated`. Compare catalogue models and declared tier aliases before,
  during and after; they stay present. Run final I hybrid eligibility through
  the same owner after cutover, without adding an alias implementation.
- `TestPAPauseRouteExceptions`: manual node without hint remains eligible;
  paused self and dormant-self still use local admission; model-list and
  pass-through still reach paused peers. Assert route response and request
  counters, not just resolver IDs. Self admission still rejects when locally
  paused. An available duplicate destination remains eligible when an earlier
  paused record has the same host: exclusion must precede deduplication.
- `TestPAAllPausedResponses`: pinned paused owners, no self/dormant/manual
  alternative; concrete 502 and tier 404 with their existing bodies and zero
  upstream requests. No new terminal all-paused 503.
- `TestPAStaleHintDestinationRejection`: use a pinned peer whose snapshot
  still says available while its controlled destination returns admission
  503/paused, then an available peer. Counters/barriers establish the stale
  snapshot before the request and prove the failed candidate and successful
  failover under I's actual retry contract. No timing race against a sweep;
  reuse existing destination admission tests for the authoritative veto.

Check fixture setup, transitions, reads and JSON decoding explicitly; use
descriptive subtests and structured policy assertions. No prompts/body logs.

**Verify:** Mac:
`go -C services/nvpair-proxy test -race -count=1 -v -run '^TestPA' .`
must execute every new named regression without skips, with required counter
and replacement assertions passing. Then
`go -C services/nvpair-proxy vet ./...` and
`go -C services/nvpair-proxy test -race -count=1 ./...` pass. Run
`npm --prefix desktop run service-contracts:check`; no PA wire diff is
expected. An unrelated integrated I contract difference goes to I's owner
for generation/check, not a hand edit or rejection of PA's metadata design.

### Task 3 — Proxy specification and README publish the final eligibility rule

**Files:** `services/nvpair-proxy/spec.md`, `services/nvpair-proxy/README.md`

**Depends on:** 2

**Complexity:** mechanical

**Do:** External gate: I6 and SR6a documentation publication/release recorded.
Re-read their integrated spec. Update Request path, Admission and the internal
policy metadata description to name `Node.PausedBy` and the final owner symbol.
Say exclusion is before deduplication/admission to the result set, not before
ordering. Name the remote-peer inference restriction, self/dormant-self,
manual-no-hint, model-list and pass-through exceptions. Describe snapshot
semantics, 502 concrete/404 tier empty results, nominal healthy polling and
indefinite last-good retention on failure; destination admission/local veto
win over stale hints. Explain pinned LAN fetch and the loopback exception.
Model catalogue and tier aliases retain declared intent while dispatch skips
paused peers. Describe hybrid only as implemented at the final I join.
README gets a concise link to those spec sections with the same exceptions.

**Verify:** Root SPDX check passes; diff touches only Files. Read the changed
paragraphs against Task 2's tests and final owner: every exception, response,
freshness condition and hybrid claim agrees. `rg -n 'PausedBy|paused peer' services/nvpair-proxy/spec.md services/nvpair-proxy/README.md`
finds the new explanation, without a hard convergence bound or a claim that
all paused nodes cease to be catalogue owners.

### Task 4 — Scanner tests prove dynamic policy retention and transport refusal

**Files:** `services/nvpair-node-scanner/availability_test.go`, `services/nvpair-node-scanner/README.md`

**Depends on:** none

**Complexity:** judgment

**Do:** External gate: exact Files released. Reuse `newRefreshTestDaemon`,
`seedEMNode`, `refreshNodeModels`, `dir.get` and cache inspection. A PA-local
stub uses a checked ephemeral loopback listener with the protected-port
exclusions in Task 2; do not reuse an unchecked `httptest.NewServer` listener.

- `TestPauseHintFollowsTheAdvertisement`: one explicit transition sequence
  (not order-dependent nominally independent subtests): available revision 1,
  local pause 2 with unchanged models, identical pause produces no change,
  acknowledged failed fetch retains pause 2 and cached policy, resume 3
  clears it, peer pause 4 preserves peer provenance. Check policy nonnil
  before dereference and compare inventory/policy fields structurally.
- `TestPauseHintEmitsOneNodeUpdated`: capture codec frames with synchronized
  output as in `peerServicesWithCapturedCodec`; after initial population,
  one pause-only refresh emits exactly one update with expected node and
  policy, identical refresh emits none, resume emits the cleared hint.
  Check every frame decode error; join concurrent writers before reading.
- `TestPauseHintIsNeverFetchedWithoutAPin`: reuse membership fixture and
  existing `peerServicesTestTransport` RoundTripper seam. Install a
  counting/refusing transport on `d.modelsHTTP`, assert loopback selection
  (including a non-self UUID) is plaintext, and assert an unpinned nonloopback
  `modelsClient` refuses transport. Call `fetchModelsForPeer` synchronously
  for that unpinned peer and require false plus **zero** plain transport calls
  and no peer-client cache creation; nonloopback selection cannot fall back
  to that client. A controlled loopback fetch is a positive control that
  increments the counter. With joined/admitted pin setup, assert HTTPS
  selection using `TestModelFetchTransportFollowsMembership` precedent.
  No actual external dial, elapsed-time proof, orphan goroutine, production
  transport abstraction or claim of end-to-end mTLS from this unit test.

README explains policy-only refresh, nominal cadence/success requirement,
last-good retention, pinned LAN transport/loopback exception and no mDNS
availability. It must not promise one-sweep convergence.

**Verify:** Mac:
`go -C services/nvpair-node-scanner test -race -count=1 -v -run '^TestPauseHint' .`,
`go -C services/nvpair-node-scanner vet ./...` and
`go -C services/nvpair-node-scanner test -race -count=1 ./...` pass.
All named tests execute without skips/races; transport counter's positive
control executes and unpinned refusal stays zero. Root SPDX check passes.

### Task 5 — Broker evidence distinguishes persisted advertisement from pending veto

**Files:** `services/nvpair-ui-broker/availability_test.go`, `services/nvpair-ui-broker/README.md`

**Depends on:** none

**Complexity:** judgment

**Do:** External gate: broker test filename and README released after I/G/SR
overlaps. Use `newActivatedBroker`, `attachFakeEngine`, `nodeProjections` and
existing provenance/veto coverage. Add
`TestPauseProjectionCarriesProvenanceAndRevision` with independent fixtures
for emission cases: local pause, local resume, peer pause/resume. Check
`setAvailability` result and error before reading stored policy. Snapshot
the expected saved advertisement after completion, require revision change
where persistence occurs, and use the fake worker's bounded `waitFor`
predicate to wait for that exact revision/provenance/roles, not `before+1`.
Pending local-veto notifications do not satisfy the saved-advertisement
predicate. Assert the matched advertisement equals stored policy.

For peer resume veto of a local pause, first complete pause and drain pending
frames via a request/reply barrier on the same fake engine reader. Capture
baseline only after that barrier. Require typed `localVetoError`, unchanged
stored policy/revision, then a second bounded reader barrier and no additional
projection. Never wait for a projection on this no-emission path. Keep barrier
helpers PA-local and use the fake worker's existing request/reply protocol;
if the final fixture lacks a drain seam, stop for an exact-file reservation
amendment rather than substitute an immediate count check. Reuse existing
provenance tests for peer-pause-over-local behavior instead of repeating the
whole OB/XA veto matrix.

README's Node projection paragraph distinguishes pending veto from adopted
saved policy; remote routers skip advertised paused peers for inference,
subject to the exceptions and advisory freshness in Task 3.

**Verify:** Mac:
`go -C services/nvpair-ui-broker test -race -count=1 -v -run '^(TestPauseProjectionCarriesProvenanceAndRevision|TestNodeProjectionPublishedOnPersistAvailabilityAndReady|TestPauseProvenanceIsNeverDowngraded|TestConfigurationEditsStayAllowedWhileLocallyPaused)$' .`,
`go -C services/nvpair-ui-broker vet ./...` and
`go -C services/nvpair-ui-broker test -race -count=1 ./...` pass.
New cases execute without skips; emission matches the persisted advertisement
and veto proves no emission after drained baseline. Root SPDX check passes.

### Task 6 — PC3 process acceptance proves policy propagation without TXT changes

**Files:** `services/tests/paused_advertisement_test.go`, `services/tests/broker_test.go`

**Depends on:** none

**Complexity:** judgment

**Do:** External gates: coordinator reserves Files against I7/XA and supplies
integrated fixture isolation with exact helper Files/owning dependency recorded
before dispatch. Read amended XA's external fixture/preflight gates. At join12
the discovery isolation prerequisite is **not satisfied**. Do not copy
`startBroker`/real-mDNS baseline unchanged or declare it hermetic. A harness
repair outside these Files is a coordinator-owned prerequisite amendment,
not PA permission to edit product flags/shared harnesses. The supplied seam
must isolate the production scanner's discovery environment and prevent
contact with installed/LAN nodes while retaining the real refresh path.
Record its proof and listener inventory. If absent, leave process acceptance
pending while disjoint unit/doc tasks proceed.

After that gate, add `Policy *nodepolicy.Advertisement` with
`json:"policy,omitempty"` and import to `availableNode` only. In the new file
add `TestPAPausedAdvertisementConvergesWithoutMDNSChange`, using the released
isolation seam and `reserveFixtureServicePorts(t).args()` for all six broker
service overrides. Keep PA-specific startup/stub glue in this file and reuse
existing message readers/cleanup. Inventory every child service, facade,
engine, observer, scanner discovery listener and helper listener. All new
listeners must be loopback on checked ephemeral ports outside
8080, 11434, 1234, 14318–14323. Unisolated wildcard discovery or child default
listeners leave this gate pending; stopping installed PAIR is insufficient.
Do not start unrelated workers/real engines to satisfy a test fixture.

`pausedAdvertisementPlatform` skips **all non-Windows** platforms before
starting a fixture. On Windows require `GOFLAGS` containing `nvpair-test-exec`
and absolute `NVPAIR_TEST_BIN_DIR`. OS gating is not machine authorization:
the coordinator must reserve actual PC3 and verify the preflight below.

Keep identity/TXT fixed throughout one explicit sequence, with checked status
and JSON errors, bounded polling and request barriers:

1. Available revision 5 with fixed model inventory reaches broker directory.
2. Pause revision 6 (`local`) arrives with the same inventory and TXT.
3. Switch stub to HTTP 500; wait for a **new failed request acknowledged after
   the switch**, then read directory/cache evidence that pause 6 remains.
   A sleep followed by unchanged data is not failure-path evidence.
4. Resume revision 7 clears provenance through successful refresh without a
   TXT update. Retained last-good pause clears only after this success.

Every polling predicate checks `Policy != nil` before dereference. Use bounded
test deadlines as harness timeouts, not product latency promises. Log each
phase completion and fixture cleanup with owned PIDs. Comment explicitly that
this loopback plaintext fixture proves propagation, not pinned authentication.

**Verify:** Mac only:
`go -C services/tests vet ./...` and
`go -C services/tests test -list '^TestPAPausedAdvertisementConvergesWithoutMDNSChange$' .`
compile/list the named test without case execution. PC3 only, after all gates:
`go -C services/tests test -count=1 -v -run '^TestPAPausedAdvertisementConvergesWithoutMDNSChange$' .`.
Require PASS, no skip, all phase logs including acknowledged failure, fixed
TXT and successful owned-process cleanup. Run existing
`TestModelsRefreshConvergesWithoutMDNSChange` with `-count=1 -v` only after
the coordinator separately certifies its discovery/listener isolation; its
additive decode compatibility can be vet/compiled meanwhile. Unsafe baseline
execution is not a regression prerequisite. Root SPDX check passes.

### Task 7 — Product documentation explains availability without timing/UI promises

**Files:** `docs/node-availability.mdx`, `fern/docs.yml`, `README.md`

**Depends on:** 2

**Complexity:** judgment

**Do:** External gate: exact documentation files released. Create the focused
MDX page with the existing SPDX style; index under Fern Reference and add a
concise user-facing link after Architecture in the root README reading order.
Explain pause/provenance, draining, local veto, retained remote configuration,
advertisement, stable catalogue/tier aliases and router exceptions. A pause
blocks new admissions; existing work can still drain according to pause
policy. Do not say a paused/draining node immediately runs no inference.
Describe nominal healthy 15-second polling, successful-refresh requirement,
preceding work/batching and indefinite last-good retention on failure.
Concrete empty candidates return 502, tiers 404; stale available hints still
meet destination 503/paused. Name the LAN authentication/loopback exception
and snapshot semantics. No desktop node-card label claim; link established
TUI behavior only after verifying its current symbols. Debug logs are optional
operational diagnostics, never proof of zero dispatch. No live engine tutorial
that implicitly authorizes a machine run.

**Verify:** Root `node scripts/spdx-headers.mjs` passes.
`rg -n 'node-availability' fern/docs.yml README.md` shows each intended link;
resolve every relative link from its document directory. The new page's
`rg -n '\bbackend\b' docs/node-availability.mdx` has no matches (scope is the
page alone, not this plan's prohibition). Read claims against Tasks 2–6 and
the final integrated source; no hard deadline or unsupported desktop label.
When Fern CLI is installed, `fern check` from `fern/` passes; when absent,
record that check as unrun, without claiming documentation-build evidence.

## Verification (whole-plan)

Mandatory unit/source gate on Mac, after file joins (no real engines):

```bash
for module in services/nvpair-proxy services/nvpair-node-scanner services/nvpair-ui-broker; do
  go -C "$module" vet ./... || exit 1
  go -C "$module" test -race -count=1 ./... || exit 1
done
go -C services/tests vet ./...
go -C services/tests test -list '^TestPAPausedAdvertisementConvergesWithoutMDNSChange$' .
node scripts/spdx-headers.mjs
npm --prefix desktop run service-contracts:check
```

Run the named `-count=1 -v` selectors in Tasks 1/2/4/5 to record actual
PASS/skip status, not a cached/silent selector match. All required names
execute without skips/races. The unit loop fails on the first failure.
Never use `make test-services`, `make test` or a module wildcard on Mac:
they include `services/tests`. Mac process checks stop at vet/compile/list.
PA changes no desktop code; integration desktop gates belong to the coordinator.
I owns regeneration of unrelated integrated contract changes through
`npm --prefix desktop run service-contracts:write`; rerun check afterward.

Mandatory PC3-only process gate (can remain external-gated separately from
the bounded patch): complete Task 6 at the exact integrated SHA. Import
amended XA preflight and current `TESTING_ON_PC.md`: reserve PC3 free of
other GPU/measurement sessions; serialize all listener tests machine-wide;
inspect `scripts/windows-test-bin.ps1 list`, fixed executable destinations,
user `GOFLAGS`, absolute `NVPAIR_TEST_BIN_DIR` and exact-path inbound rule
coverage for every package image/copied child. Existing manifest membership
does not prove rule coverage. No new manifest role is expected; if required,
stop for reservation/Files amendment. Missing elevated setup leaves execution
pending; never use temporary executable paths or bypass rules. Record
before/after firewall and survivor checks, cleanup logs, actual names/skips,
and screen observation **UNMONITORED** unless attended. Do not request a
routine owner prompt report. PC3 uses no `-race`; MSI can supply serialized
proxy/scanner/broker unit race evidence, never this package's process cases.

### Separate optional live acceptance gate — XA/I coordinated, MSI and PC3

This is not a prerequisite for the bounded PA patch or its mandatory fixture
assertions. It requires a separate coordinator reservation **and explicit
owner authorization** for actual MSI/PC3 engine/model operations, with engine,
model, source SHA, machine roles, memory/launch envelope, observation limits
and cleanup recorded before execution. Use an approved llama.cpp engine;
Strata is optional and no Ollama/live Mac runs are authorized here. Coordinate
with XA/I, SR/NE engine readiness and existing GPU/measurement reservations.

After successful hint refresh is observed, exercise pause/resume on MSI with
PC3 as router and record actual dispatch/result evidence and refresh times.
Do not require a live request to beat an asynchronous sweep or impose a
20-second success bound. Quiet default-level logs prove no zero-attempt
claim; mandatory zero-dispatch/stale-hint assertions come from fixture
counters/barriers in Task 2. Passing process fixtures does not close this live
gate; absent authorization/reservation leaves it unperformed.

Final diff must match Files and scope. Proxy is the only compiled product
binary expected to change; eventual PR release intent declares proxy only,
subject to actual diff. Do not hand-edit versions/changelog. Implementation
commits follow repository policy with `git commit --no-signoff`; untracked
paths need explicit staging authorization. This amendment makes no commits.

## Do not

- Do not add mDNS availability, wire/persistence fields, a second routing
  owner, push, revision-order guard, pause cache, metrics or configuration.
- Do not alter I's retry/cancellation/attempt ownership or SR/XA/OB/E4/NE
  mechanisms; do not refactor beyond the specified tasks.
- Do not edit destination veto/admission, engine-manager/shared-wire product
  code, desktop UI or generated APIs for PA's internal metadata.
- Do not remove self/dormant-self/manual-no-hint/non-inference exceptions,
  catalogue entries or declared tier aliases; do not add an all-paused 503.
- Do not bind/dial installed/protected ports, infer discovery isolation from
  temporary state, run process cases outside PC3, run PC3 race, overlap
  machine listener/measurement intervals, or start a real engine via this plan.
- Do not log prompts/messages/bodies/PINs/keys or use debug-log absence as a
  request-count assertion. Say engine in new prose/comments.
- Do not edit other packages, vault notes or product code during this plan
  amendment. No branch, checkout, commit or push is part of this amendment.

## If stuck

Use Go 1.25 or newer without lowering module directives. If an external
reservation, integrated owner, discovery isolation seam, fixed-path/rule
coverage or machine reservation is absent, report that exact gate as pending;
continue only released disjoint work. Reconcile contradictory PA/I ordering
before dispatch, never by silently choosing one side.

If I moves candidate ownership, identify the landed file/symbol, amend exact
Files/dependencies and audit before edits. Never restore the old closure as
a second path. Pinned-peer fixture failures require fixing pin/admission,
identity and transport setup under the final I contract; do not replace or
drop the test. A new fixture helper/file or non-loopback listener requirement
needs coordinator scope/reservation amendment before implementation.

For notification failures, inspect decoded frames and use bounded predicates
plus the reader barrier; do not weaken to counts/sleeps. For process failure,
check post-switch stub acknowledgement and actual isolated scanner route;
do not contact installed PAIR. Stop and report self exclusion, catalogue loss,
policy comparisons added to `nodeEqual`, unexplained contract drift or a
failure requiring writes outside declared Files. Preserve external evidence
as unverified until its gate runs.

## Amendment 2026-10-21

Applied review R1–R7 and LOW corrections. Re-anchored to join12 with symbol
references; repaired dependency syntax and task complexity; recorded the
coordinator-adopted owner answers for veto. Replaced stale PA-before-I gates
with explicit I/G integration and file-reservation transfer, including I6/SR6a
documentation serialization and cross-package cycle reconciliation. Added
dynamic pinned-peer resume/counter evidence, route exceptions, checked fixture
ports, transport refusal instrumentation, persisted-projection synchronization
and acknowledged failed-refresh retention. Process acceptance is PC3-only
and externally gated on real isolation/fixed-path readiness; live MSI/PC3
engine validation is a separate owner-authorized XA/I gate. Corrected cadence,
loopback authentication, draining, catalogue, hybrid and desktop claims.
No implementation or runtime validation is claimed by this amendment.
