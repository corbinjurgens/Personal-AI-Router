<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution plan: PA — paused-node authenticated advertisement

Planned against `fork/groundwork` at commit
`584e99242562df1cb19d2c7a318b7342e03cb7d2`. Every `path:line` below refers to
that commit. Path shorthand: `B/` = `services/nvpair-ui-broker/`,
`P/` = `services/nvpair-proxy/`, `E/` = `services/nvpair-engine-manager/`,
`S/` = `services/nvpair-node-scanner/`, `N/` = `services/shared/`,
`X/` = `services/tests/`.

## Goal

A routing node never dispatches an inference request to a paired node that has
advertised itself as paused, and still fails over correctly when that
advertisement is stale:

- The pause hint a node already publishes — `pausedBy` in its policy
  advertisement, served only in its pinned-mTLS `GET /v1/models` document and
  stamped with the policy revision that recorded the pause — reaches the
  proxy's routable `Node` (today it is dropped at `subscribedToNode`,
  `P/proxy.go:1230`).
- `resolveCandidates` (`P/proxy.go:629`), which concrete and tier routing both
  use, skips a non-self node carrying a pause hint whenever it is filtering
  owners for a model. A paused peer therefore costs no dispatch, no 503 round
  trip and no retry-budget. The self candidate, manual nodes and non-inference
  routes (model list, pass-through) are unaffected.
- A stale hint is harmless in both directions: a peer that paused after the
  last fetch still answers `503 X-PAIR-Admission: paused` from its own
  admission (`P/admission.go:382-383`), which the router treats as a failed
  candidate; a peer that resumed after the last fetch is routed to again within
  one scanner sweep (`peerRefreshInterval`, `S/daemon.go:218`, 15 s).
- The destination's local veto is untouched: a local pause or Off still refuses
  a paired node's resume, start, load and switch (`B/availability.go:59-65`,
  `B/policy.go:772-774`, `E/nodeprojection.go:67-89`).
- Nothing about the pause enters mDNS: the `_nvpair-node` TXT record keeps only
  `v`, `uuid`, `cluster-uuid`, `ip`, `ips` and the service keys
  (`N/noderec/noderec.go:59-69`, `S/registry.go:139-147`).
- The behaviour is covered by proxy, scanner and broker unit tests, one
  cross-process acceptance test, the proxy spec, the component READMEs and a
  new product document `docs/node-availability.mdx`.

## Change scope

- **Intent / Scope:** Bounded improvement. Product code changes are confined to
  two functions and one struct in the proxy (`P/discovery.go` `Node`,
  `P/proxy.go` `subscribedToNode` and `resolveCandidates`). Every other task is
  tests or documentation. No JSON-RPC method, payload field, HTTP route, mDNS
  key or persisted file changes; `Node.PausedBy` is `json:"-"` like
  `Node.Roles`, so the proxy's outward `node/*` contract is unchanged and
  `desktop/docs/services-api.md` needs no regeneration.
- **Basis:** the advertisement pipeline that already exists end to end:
  `Policy.Advertisement()` carries `PausedBy` and `Revision`
  (`N/nodepolicy/advertise.go:275`); the broker publishes it to engine-manager
  on every pause and resume (`B/availability.go:207`, `:322`,
  `B/policy_notifications.go:88-113`); engine-manager serves it as `policy` in
  `/v1/models` (`E/models.go:105-109`, `E/nodeprojection.go:55-65`); the
  scanner fetches it over pinned mTLS only (`S/model_enrichment.go:278-284`),
  compares the advertisement exactly (`S/directory.go:277`) and emits
  `discovery:node-updated` on a change; the broker relays `DirectoryNode.Policy`
  intact (`B/discovery.go:80`, `:94-122`); the proxy decodes the snapshot in
  `replaceSubscribed` (`P/facade.go:419`).
- **Compatibility:** an older peer that publishes no `policy`, or a policy
  without `pausedBy`, keeps today's behaviour (no hint, dispatched, rejected by
  its own admission if paused). No new listener roles, no Windows firewall
  rules, no manifest (`services/shared/testbin/programs/tests.txt`) changes:
  the cross-process test reuses the broker fixture and an `httptest` stub.
- **Non-goals:** no faster propagation than the scanner's sweep (no new relay
  message, no reverse-session payload change); no change to what the federated
  model list or the hybrid tier aliases show for a paused peer; no change to
  the 502/503 bodies the router answers with when no candidate remains; no
  change to the self candidate, the dormant self candidate, manual nodes or the
  destination veto; no desktop or TUI code (the desktop already reads
  `policy.pausedBy` from discovery nodes,
  `desktop/src/shared/utils/node-roles.ts:55`); no mDNS TXT key; no Ollama or
  live-engine runs.

## Context the executor needs

Verified in the Go source at the SHA above. The local Go toolchain used while
writing this plan was 1.24 and the modules require 1.25 (`go 1.25.0` in each
`go.mod`), so nothing below was compiled or run; every statement is from
reading the source. Line numbers are exact where a single line is cited and
approximate (±2) for ranges.

**The hint and its provenance.**
- `nodepolicy.Availability` is `available`, `draining` (never persisted) or
  `paused` (`N/nodepolicy/policy.go:48-58`). `Policy.PausedBy` is `local` or
  `peer`, empty while available, written only by `node:set-availability`
  (`N/nodepolicy/policy.go:124-128`, `:149-156`); `Validate` rejects a
  `pausedBy` on an available policy (`:483-486`).
- `Policy.Advertisement()` returns `{Revision, Roles, PausedBy, ServedModels}`
  (`N/nodepolicy/advertise.go:68-79`, `:258-276`). `PausedBy` is documented
  there as "lets a remote UI show paused on that node before it tries"
  (`:73`). The strict capability is added by the broker afterwards
  (`B/policy_notifications.go:102`).
- `persistAvailabilityLocked` saves the pause with `PausedBy = origin` under a
  new revision before anything else happens (`B/availability.go:154-167`), and
  `pauseNode` publishes the node projection right after adopting it
  (`:174-207`): the hint therefore carries the revision of the pause itself and
  is visible to peers while the node is still draining. `resumeNode` clears it
  under the next revision (`:290-322`). A peer pause over a local pause stays
  `local` (`:59-65`); `pausedLocally` counts a pause without provenance as local
  (`B/policy.go:772-774`). These are covered by
  `TestNodeProjectionPublishedOnPersistAvailabilityAndReady`
  (`B/policy_projection_test.go:168`) and
  `TestPauseProvenanceIsNeverDowngraded` (`B/policy_veto_test.go:96`).
- engine-manager keeps the last projection whole (`E/nodeprojection.go:41-65`)
  and serves it as `policy` in `/v1/models` and `engine:models`
  (`E/models.go:105-109`); `TestNodeProjectionServedWithTheModelInventory`
  already pushes a `PausedBy: local` projection and reads it back over HTTP
  (`E/nodeprojection_test.go:67`, `:101`). `engine:models-changed` never
  carries the policy (`E/models.go:46-50`). No engine-manager change is needed.

**Authenticated, revisioned, dynamic at the scanner.**
- A peer's `/v1/models` is fetched only through `peerServiceClient`, pinned to
  the cluster principal the peer advertises; there is no plaintext fallback for
  a peer (`S/model_enrichment.go:278-284`, `:119-164`). Self is read over
  loopback. `TestModelFetchTransportFollowsMembership`
  (`S/cluster_membership_test.go:39-66`, daemon from
  `newMembershipTestDaemon`, `:17`) proves an unpinned peer is never fetched.
- The answer's `policy` is decoded by `decodeInventoryPolicy`
  (`S/model_enrichment.go:186`); the whole `modelInventory` (`S/enrich_hosts.go:17-30`)
  replaces the node's inventory fields in `applyModels`, which reports a change
  when `reflect.DeepEqual(n.Policy, inventory.policy)` is false
  (`S/directory.go:265-284`, `:277`), so a `pausedBy` flip alone emits one
  `discovery:node-updated` (`S/model_enrichment.go:383-385`). A failed fetch
  keeps the last-good advertisement (`:60-71`, `:350-352`, covered by
  `TestRefreshPolicyAdvertisement`, `S/refresh_test.go:445`).
- Peers are re-fetched every `peerRefreshInterval = 15 * time.Second`
  (`S/daemon.go:218`) by `refreshPeersLoop` → `refreshModelsOnce`
  (`S/daemon.go:1428-1444`, `S/model_enrichment.go:293-311`), and once more
  when an authenticated peer-services claim first arrives or its route changes
  (`S/peer_services.go:153-157`, `:161-220`). Stale-result guards:
  `routeGeneration` (`S/model_enrichment.go:341`, `:355`) and the claim
  incarnation check (`:358-366`). There is no revision-order guard: a reordered
  pair of fetches for one peer is corrected by the next sweep.
- The self record's TXT is built from `NodeRecord` fields only
  (`S/registry.go:115-131`, `:139-147`; keys at `N/noderec/noderec.go:59-69`);
  `NodeRecord` has no availability field, so the hint cannot enter mDNS without
  a type change. PA adds none.

**Broker relay.** `directoryToEnriched` copies `DirectoryNode.Policy` onto
`EnrichedNode.Policy` (`B/discovery.go:80`, `:94-122`), and the broker's
`discovery:get-nodes` row documents `policy.pausedBy?` (`B/README.md:884`).
The `discovery:nodes` snapshot the proxy subscribes to carries
`noderec.DirectoryNode` values (`P/facade.go:419-431`).

**Proxy, where the hint is dropped today.**
- `Node` (`P/discovery.go:34-78`) carries `Roles`, `ServedModels`,
  `PolicyRevision` and `StrictAdmission` as `json:"-"` routing metadata
  (`:71-77`); `nodeEqual` (`:191-195`) ignores them so a policy edit announces no
  `node/updated`. There is no `PausedBy` field.
- `subscribedToNode` copies `Roles`, `ServedModels`, `Revision` and
  `StrictAdmission` from `n.Policy` (`P/proxy.go:1262-1267`) and nothing else.
- `resolveCandidates(model)` (`P/proxy.go:629`): with a model it first filters
  the snapshot to owners (`:645-653`, `nodeAdvertisesModel` at `:936`); the
  `add` closure (`:672`) resolves a dialable URL, then classifies the node as
  self (`isSelfTarget`, `:684-695`, debug log at `:691`), a pinned cluster peer
  (`p.mesh.HasPin(n.ClusterUUID)`, `:696-707`), a manual node (`:708-710`) or an
  unpinned relay peer, which is dropped (`:711-718`, log at `:715`); the
  self-target guard follows (`:720-724`, log at `:721`), then host dedup
  (`:725-728`) and the append (`:734-743`). Selection, priority and ID order
  call `add` (`:748-763`); the dormant self candidate comes last (`:768-771`);
  the final debug line logs `selected`, `priority`, `candidates`, `eligible`
  and `known` (`:772`).
- Tier routing builds every candidate through `resolveCandidates(role.Model)`
  and `resolveCandidates(e.Model)` (`P/tiers.go:304`, `:321`) inside
  `resolveTierCandidates` (`:281`); `declaredRoles` (`:197-210`) reads
  `Node.Roles` only. Hybrid eligibility is built on `resolveTierCandidates`
  (`P/spec.md:170-172`). So one filter in `resolveCandidates` covers concrete,
  tier and hybrid routing.
- The model-list routes (`matchedRoute.Role.isModelList()`,
  `P/http_handler.go:718-724`) also call `resolve()` → `resolveCandidates("")`
  (`:658-676`), with an empty model. The filter must not apply there.
- Empty candidate sets: a tier request answers `404 no eligible model for tier`
  (`P/http_handler.go:734-759`); a model-bearing request answers
  `502 {"error":"no available node advertises the requested model"}` (`:761-767`).
  Candidates are re-resolved every retry round (`:1162`, `P/spec.md:499-501`),
  so a hint that changes mid-request takes effect on the next round.
- Destination fast rejection: admission rejects `paused` before anything else
  (`P/admission.go:378-385`), answered as `503` with
  `X-PAIR-Admission: paused` and `Retry-After: 30` (`writeAdmissionRejection`,
  `P/admission.go:852`; `retryAfterSeconds`, `:838-847`; header constant
  `N/nodepolicy/wire.go:1100-1101`, reason `:1116-1123`). A routing node treats
  that 503 as a failed candidate and moves on (`P/spec.md:1090-1092`,
  `TestHandleHTTP_FailoverOn503`, `P/failover_test.go:294`). The self candidate,
  when rejected while this node is not available, is excluded for the rest of
  the request (`P/http_handler.go:1297-1306`,
  `TestHandleHTTP_SelfCandidateRejectedFailsOver`,
  `P/admission_http_test.go:586`).

**Test fixtures to reuse.**
- Proxy: `forEachEngine` / `engineCase` (`P/enginecase_test.go:26`, fields
  `profile`, `inferencePath`, `modelListPath`, `advertisedModel`,
  `inferenceRequest()`), `newTestProxy`/`testProxy` (`P/failover_test.go:36`,
  `:63`), `nodeFor`/`nodeForModel` (`:67`, `:84`), `newRecordingEngine`
  (`P/admission_http_test.go:280`, with `hits()`), `selfNode`/`setBackend`
  (`:258`, `:262`), `newTierFixture`/`ollamaTierPolicy`/`ollamaCase`
  (`P/tiers_test.go:133`, `:173`), the pinned subscribed-peer pattern with
  `clustertrusttest.Join` + `p.mesh = clustertrust.Open(clusterDir)`
  (`P/subscribed_test.go:368-380`), `TestSubscribedCarriesPolicyWithoutChurn`
  (`:214`) as the model for the projection test.
- Scanner: `newRefreshTestDaemon` (`S/refresh_test.go:93`), `startModelsStub` /
  `modelsStub.set` (`:65`, `:37`), `seedEMNode` (`:108`),
  `d.refreshNodeModels` (`S/model_enrichment.go:329`), `d.dir.get`.
- Broker: `newActivatedBroker` (`B/policy_test.go:396`), `attachFakeEngine`
  (`:227`), `nodeProjections`/`waitProjections`/`requireLastProjectionIsStored`
  (`B/policy_projection_test.go:129`, `:150`, `:158`), `b.setAvailability(ctx,
  state, origin)` (`B/availability.go:43`).
- Cross-process: `startBroker`, `sendReq`, `availableNode`,
  `availableNodesResult` (`X/broker_test.go:66`, `:99`, `:47-60`);
  `swapModels` stub, `pollForNode`, `findNode`, the mDNS responder setup and
  `TestModelsRefreshConvergesWithoutMDNSChange` (`X/models_refresh_test.go:31`,
  `:153`, `:57-80`). `availableNode` has no `Policy` field today (`:47-56`).
  Platform gating pattern: `strataContractPlatform`
  (`X/strata_contract_test.go:276-286`): skip on darwin; on Windows require
  `GOFLAGS` to contain `nvpair-test-exec` and an absolute
  `NVPAIR_TEST_BIN_DIR` (`services/readme.md:215-246`).

**Seams that will move under other plans (rebase notes).**
- I Tasks 3-5 (unified dispatch) edit `P/proxy.go`, `P/policy.go`,
  `B/policy.go`, `B/proxy.go`, `N/nodepolicy/wire.go`. PA's only product edits
  in `P/proxy.go` are the four-line block in `subscribedToNode`
  (`:1262-1267`) and one guarded `return` inside the `add` closure of
  `resolveCandidates` (between the self-target guard at `:720-724` and the
  host dedup at `:725-728`), plus one field on the final debug line (`:772`).
  If I has moved candidate construction into another file or type by the time
  PA rebases, apply the same guard at the point where a non-self peer
  candidate is appended, keeping the rule word for word from Task 2. PA does
  not touch `P/policy.go`, `P/http_handler.go`, `B/policy.go`, `B/proxy.go` or
  `N/nodepolicy/wire.go`.
- G Task 3 (egress lease broker relay) edits broker files. PA's broker changes
  are a new test file and one sentence in `B/README.md` under "Node
  projection" (`B/README.md:496-504`); if that bullet moved, add the sentence
  wherever `engine:node-projection` is described.
- J (routing diagnostics) may restructure the rejection bodies in
  `P/http_handler.go`; PA deliberately leaves them alone (Owner question 1).

**Could not verify.** Compilation and tests (Go toolchain mismatch): no
command in this plan was run. The `engineCase` struct was read only to
`P/enginecase_test.go:48`; `tc.advertisedModel` is taken from its uses at
`P/failover_test.go:315` and `P/admission_http_test.go:592`. Whether the
scanner's `peerServiceClient` returns `ok == false` for an unpinned peer
without touching the network was inferred from `modelsClient`
(`S/model_enrichment.go:278-284`) and the test at
`S/cluster_membership_test.go:50-66`, not executed.

## Decisions (settled — do not revisit)

- **Local Off/pause wins over remote start/resume/residency.** The hint is
  advisory for routers; the destination's admission and `pausedLocally` veto
  remain authoritative and are not changed. Not "trust the hint and skip
  admission at the destination", because a stale hint must never admit work a
  local pause forbids.
- **Remote configuration edits remain permitted** while paused
  (`B/README.md:591-592`, `TestConfigurationEditsStayAllowedWhileLocallyPaused`,
  `B/policy_veto_test.go:181`). PA changes nothing on the policy relay.
- **Mac never serves models.** Mac runs proxy/scanner/broker unit tests (with
  `-race`), never `services/tests`.
- **No mDNS identity mutation.** The hint travels only in the pinned
  `/v1/models` document. Not "a `paused=` TXT key", because TXT is unsigned,
  size-limited and readable by any LAN host, and a pause must not re-announce
  the mDNS record.
- **Authenticated** means the existing pinned-mTLS fetch (`S/model_enrichment.go:278-284`);
  PA adds no new authentication and no plaintext path.
- **Revisioned** means the hint is the `pausedBy` of the advertisement whose
  `revision` is the revision the pause or resume persisted; the proxy keeps
  `Node.PolicyRevision` beside `Node.PausedBy` and logs it when it skips a
  node. Not "a separate availability sequence number", because the policy
  revision already moves on every pause and resume
  (`X/service_policy_test.go:176-184`).
- **Dynamic** means the hint follows the scanner's sweep: a pause or resume is
  visible to peers within `peerRefreshInterval` (15 s) plus one fetch, and
  reaches the proxy on the next `discovery:nodes` push. Not a push from the
  pausing node to its peers (Owner question 2).

## Decisions made while writing this plan

1. **Where the filter lives.** Inside `resolveCandidates`' `add` closure,
   after the self/peer/manual classification and the self-target guard, before
   `seenHost` dedup: `if model != "" && !self && n.PausedBy != "" { skip }`.
   After the switch, so the self candidate (which carries the same hint when
   this node is paused) keeps its admission path and its 503-with-reason answer
   (`TestHandleHTTP_SelfOnlyRejectionAnswersWithAdmission503`,
   `P/admission_http_test.go:624`); before dedup, so a manual duplicate of a
   paused relay peer is still tried (manual nodes are a deliberate bypass and
   never carry a hint). Alternative rejected: filtering in the owner loop
   (`:645-653`) would also drop a paused self before it is recognized as self.
2. **Only when a model is being resolved** (`model != ""`). Model-list and
   pass-through routes keep querying paused peers: a paused node still answers
   non-inference routes (they are not admitted, `P/spec.md:1088-1089`), and the
   federated model list should not shrink and grow with pauses.
3. **Any non-empty `pausedBy` counts** (`local` and `peer` alike); the proxy
   never distinguishes them. A peer pause is as futile to dispatch to as a
   local one.
4. **`Node.PausedBy` is `json:"-"` and ignored by `nodeEqual`**, like `Roles`
   and `PolicyRevision`, so a pause announces no `node/updated` from the
   proxy. The desktop learns pauses from the broker's discovery feed, not from
   `node/*`.
5. **No revision-order guard in the scanner or proxy** (Owner question 4):
   the last fetch wins, as today for roles and strict capability.
6. **Logging.** One `slog.Debug` per skipped node with `node_id`, `paused_by`
   and `policy_revision`, plus a `paused` count on the existing "resolved"
   debug line. No new client-visible event; J owns diagnostics.
7. **Tests live in new `availability_test.go` files** in `P/`, `S/` and `B/`
   as the catalogue names them, rather than appended to the existing
   `subscribed_test.go`, `refresh_test.go` and `policy_projection_test.go`, so
   PA's tests rebase cleanly over I/G/J.
8. **The cross-process test proves the directory hop, not a two-proxy
   dispatch.** It reuses the plaintext loopback stub pattern of
   `TestModelsRefreshConvergesWithoutMDNSChange`; the pinned-mTLS fetch, the
   proxy filter and the destination veto are each proven by unit tests. A
   two-node live dispatch acceptance is left to the existing Windows gates.
9. **Documentation.** The proxy spec gets the rule (§5 step 1 and §12.2); the
   scanner and broker READMEs each get one sentence; a new product page
   `docs/node-availability.mdx` explains pausing, provenance, the veto and the
   hint for operators, indexed from `fern/docs.yml` (Reference section) and the
   root `README.md` reading order after Architecture.

## Tasks

### Task 1 — The proxy's routable `Node` carries the pause hint

**Files:** `services/nvpair-proxy/discovery.go` (modify),
`services/nvpair-proxy/proxy.go` (modify, `subscribedToNode` only),
`services/nvpair-proxy/availability_test.go` (create).

**Depends on:** none.

**Complexity:** mechanical.

**Do:**

1. In `services/nvpair-proxy/discovery.go`, inside `type Node struct`
   (`:34-78`), after the `PolicyRevision string` field (`:73`) and before
   `StrictAdmission` (`:74-77`), add:

   ```go
   	// PausedBy is the node's advertised pause provenance (DirectoryNode.Policy
   	// .PausedBy): "local" or "peer" while the node says it is paused, empty
   	// otherwise. It is an authenticated hint stamped with PolicyRevision:
   	// resolveCandidates skips a peer that carries one when it resolves owners
   	// for a model, and the destination's own admission remains the veto for a
   	// hint that is stale. Internal routing metadata like Roles: not compared by
   	// nodeEqual, so a pause announces no node/updated.
   	PausedBy nodepolicy.PauseOrigin `json:"-"`
   ```

   Update the `nodeEqual` doc comment (`:187-190`) so its list reads
   "Loaded, ClusterUUID, Roles, PolicyRevision, PausedBy and StrictAdmission
   are internal routing metadata and deliberately not compared." Do not change
   the function body.

2. In `services/nvpair-proxy/proxy.go`, `subscribedToNode` (`:1230`), extend
   the `if n.Policy != nil {` block (`:1262-1267`) with one line after
   `node.PolicyRevision = n.Policy.Revision`:

   ```go
   		node.PausedBy = n.Policy.PausedBy
   ```

   Extend the comment above the block (`:1260-1261`) to: "The node's own
   declaration, read by tier routing (declaredRoles), its pause hint, read by
   resolveCandidates, and its strict admission capability, read only by
   Node.strictModel."

3. Create `services/nvpair-proxy/availability_test.go` with the two-line
   SPDX comment header used by every Go file, `package main`, and:

   - `TestSubscribedToNodeCarriesPausedBy`: with `forEachEngine`, build a
     `noderec.DirectoryNode{HostUUID: "host-uuid", Name: "host-a", IP:
     "10.0.0.5", Services: {tc.profile.DiscoveryService: {Port:
     tc.profile.FacadePort}}}`. Subtests: `no advertisement` → `PausedBy == ""`;
     `advertisement without a pause` (`Policy: &nodepolicy.Advertisement{Revision:
     "40"}`) → `PausedBy == ""` and `PolicyRevision == "40"`; `paused locally`
     (`PausedBy: nodepolicy.PausedLocally`, `Revision: "41"`) → `PausedBy ==
     nodepolicy.PausedLocally` and `PolicyRevision == "41"`; `paused by a peer`
     → `nodepolicy.PausedByPeer`.
   - `TestPauseHintAnnouncesNoNodeUpdate`: `d := NewDiscovery()`;
     `d.SetSubscribed([]Node{available})` then `d.SetSubscribed([]Node{paused})`
     where `paused` differs only in `PausedBy` and `PolicyRevision`; assert the
     second call returns `len(discovered)+len(updated)+len(removed) == 0` and
     `d.Nodes()[0].PausedBy == nodepolicy.PausedLocally` (the overlay took the
     hint silently). Model it on `TestSubscribedCarriesPolicyWithoutChurn`
     (`P/subscribed_test.go:214`).

**Verify:**

- Mac: `cd services/nvpair-proxy && go vet ./... && go test -race -run
  'TestSubscribedToNodeCarriesPausedBy|TestPauseHintAnnouncesNoNodeUpdate|TestSubscribed|TestStrictModel'
  ./...` passes (exit 0, every named test reported `PASS`, none skipped).
- Mac: `cd services/nvpair-proxy && go test -race ./...` passes.
- `node scripts/spdx-headers.mjs` from the repository root reports no missing
  header.
- Windows (PC3 or MSI): not required for this task.

### Task 2 — `resolveCandidates` skips peers that advertise a pause

**Files:** `services/nvpair-proxy/proxy.go` (modify, `resolveCandidates`
only), `services/nvpair-proxy/availability_test.go` (modify).

**Depends on:** Task 1.

**Complexity:** judgment.

**Do:**

1. In `resolveCandidates` (`P/proxy.go:629`), declare `pausedSkipped := 0`
   next to `known := len(nodes)` (`:644`).

2. Inside the `add` closure, immediately after the defensive self-target guard
   (`:720-724`, the block ending `return` after the
   `"resolveCandidates: skipping self-target node"` log) and before
   `if seenHost[u.Host] {` (`:725`), insert exactly:

   ```go
   		// The node's own policy advertisement says it is paused. It reached us
   		// through the scanner's pinned fetch with the revision that recorded
   		// the pause, so dispatching to it would only buy a 503 from its
   		// admission. Only owner resolution for a model is affected: model-list
   		// and pass-through routes still reach a paused peer, and this node's own
   		// engine keeps its admission path, whose rejection is the request's
   		// answer when nothing else can take it. A hint that is stale in the
   		// other direction is caught by the destination's admission.
   		if model != "" && !self && n.PausedBy != "" {
   			pausedSkipped++
   			slog.Debug("resolveCandidates: skipping paused peer",
   				"node_id", n.ID, "paused_by", n.PausedBy, "policy_revision", n.PolicyRevision)
   			return
   		}
   ```

3. Extend the final debug line (`:772-774`) with `"paused", pausedSkipped`
   after `"known", known`.

4. Update the `resolveCandidates` doc comment (`:617-628`): after the sentence
   ending "advertised owners.", add "A peer whose advertisement says it is
   paused (Node.PausedBy) is not an owner for a model; this node's own engine
   is, and admission answers for it."

5. Append to `services/nvpair-proxy/availability_test.go`:

   - `TestResolveCandidatesSkipsPausedPeersForAModel`: `forEachEngine`; two
     `newRecordingEngine(t, http.StatusOK)` upstreams; `disc := NewDiscovery()`;
     `paused := nodeForModel(t, "paused", pausedEngine.URL, tc.advertisedModel)`
     with `paused.PausedBy = nodepolicy.PausedLocally` and
     `paused.PolicyRevision = "7"`; `good := nodeForModel(t, "good",
     goodEngine.URL, tc.advertisedModel)`; `disc.AddManual(paused)`;
     `disc.AddManual(good)`; `p := testProxy(tc.profile, disc,
     tc.profile.FacadePort)`; `p.soleFacade().SetSelected("paused")`.
     Subtests:
     - `model owners exclude the paused peer`: ids of
       `p.soleFacade().resolveCandidates(tc.advertisedModel)` equal `["good"]`.
     - `model list still reaches it`: ids of
       `p.soleFacade().resolveCandidates("")` contain both `paused` and `good`.
     - `an inference request never dials it`: `rec := httptest.NewRecorder();
       p.soleFacade().handleHTTP(rec, tc.inferenceRequest())`; assert
       `rec.Code == http.StatusOK`, `goodEngine.hits() == 1`,
       `pausedEngine.hits() == 0`, and `rec.Header().Get(nodepolicy.NodeHeader)
       == "good"`.
     - `a peer pause counts too`: set `paused.PausedBy =
       nodepolicy.PausedByPeer`, `disc.AddManual(paused)` again (replaces), and
       assert `resolveCandidates(tc.advertisedModel)` ids equal `["good"]`.
     (Manual nodes never carry a hint in production; they are used here
     because the rule reads the `Node` field and a manual node is dialable
     without a cluster pin.)
   - `TestResolveCandidatesSkipsPausedPinnedPeer`: copy the pinned fixture of
     `TestStrictModelExcludesPeersWithoutAValidCapability`
     (`P/subscribed_test.go:322`, lines `:368-380`: `profile :=
     ollamaOnlyProfile(t)`, `subscribedToNode` from `noderec.DirectoryNode`
     values with `ClusterUUID` principals, `clustertrusttest.Join`, `p.mesh =
     clustertrust.Open(clusterDir)`), with two peers: `peer-0` whose `Policy` is
     `&nodepolicy.Advertisement{Revision: "3", PausedBy:
     nodepolicy.PausedLocally}` and `peer-1` with `Revision: "3"` and no pause.
     Assert `resolveCandidates(model)` ids equal `["peer-1"]` and
     `resolveCandidates("")` ids equal `["peer-0", "peer-1"]` (sorted).
   - `TestAllOwnersPausedAnswersNoOwner`: as the first test but with only the
     paused manual node; `handleHTTP` on `tc.inferenceRequest()` returns
     `http.StatusBadGateway` with body
     `{"error":"no available node advertises the requested model"}` followed
     by a newline (`http.Error` appends one), and the paused engine has
     `hits() == 0`.
   - `TestPausedSelfKeepsItsAdmissionPath`: copy the setup of
     `TestHandleHTTP_SelfCandidateRejectedFailsOver`
     (`P/admission_http_test.go:586-600`: `selfNode("self", port,
     tc.advertisedModel)` plus a manual `peer`, `setBackend`,
     `p.SetPriority([]string{"self", "peer"})`), but give the self node
     `PausedBy = nodepolicy.PausedLocally` while leaving
     `p.admission` available. Assert the first request is served by the local
     engine (`local.hits() == 1`): the hint never removes self. Then
     `p.admission.setAvailability(nodepolicy.Paused, false)` and assert the next
     request is served by `peer` with `rec.Code == http.StatusOK` (the existing
     self-exclusion path, unchanged).
   - `TestTierRoutingSkipsPausedPeer`: `tc := ollamaCase(t)`; `pol :=
     ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"weak": {{Kind:
     nodepolicy.EntryLocal, Engine: "ollama", Model: "small:1b"}}})`; `fx :=
     newTierFixture(t, tc, pol, map[string][]string{"a": {"small:1b"}, "b":
     {"small:1b"}}, nil)`; then mark `a` paused: read `n := fx.p.soleFacade().
     discovery` nodes, find id `a`, set `PausedBy = nodepolicy.PausedLocally`
     and `AddManual` it back; `fx.p.SetPriority([]string{"a", "b"})`; `rec :=
     fx.do(t, "/v1/chat/completions", `{"model":"weak"}`)`; assert `rec.Code ==
     http.StatusOK`, `fx.servedBy(t) == "b"`, `fx.eng["a"].hits() == 0`.
   - `TestStaleAvailableHintStillFailsOverOnDestinationRejection`: two manual
     nodes with no hint; the first upstream answers like a paused destination:
     `w.Header().Set(nodepolicy.AdmissionHeader, string(nodepolicy.RejectPaused));
     w.Header().Set("Retry-After", "30"); w.WriteHeader(503)` with body
     `{"error":"this node is not accepting the request (paused)","code":"admission"}`;
     the second answers 200. `SetSelected` the first. Assert `rec.Code == 200`,
     the first upstream was hit exactly once and the second once, and
     `nodepolicy.NodeHeader` names the second. This pins the "stale metadata
     still meets destination fast rejection" half of the outcome.

**Verify:**

- Mac: `cd services/nvpair-proxy && go vet ./... && go test -race -run
  'TestResolveCandidates|TestAllOwnersPaused|TestPausedSelf|TestTierRoutingSkipsPausedPeer|TestStaleAvailableHint|TestSubscribedToNodeCarriesPausedBy|TestPauseHintAnnouncesNoNodeUpdate'
  ./...` passes with every named test `PASS` and none skipped.
- Mac: `cd services/nvpair-proxy && go test -race ./...` passes (no
  regression in `TestTier_*`, `TestHandleHTTP_*`, `TestStrictModel*`,
  `TestHybrid*`, `TestAlias*`).
- Mac: `cd desktop && npm run service-contracts:check` passes unchanged (no
  wire change; if it reports a diff, the change is wrong, not the check).
- `git diff --stat` for this task touches only the two files listed.
- Windows (PC3 or MSI): not required for this task.

### Task 3 — Proxy specification and README name the rule

**Files:** `services/nvpair-proxy/spec.md` (modify),
`services/nvpair-proxy/README.md` (modify).

**Depends on:** Task 2.

**Complexity:** mechanical.

**Do:**

1. `services/nvpair-proxy/spec.md`, §5 "Request path", step 1 of the
   model-bearing list (`:402-405`). After "An empty owner set returns a local
   `502` without contacting an engine." add the sentence: "An owner whose own
   policy advertisement says it is paused (`pausedBy` set, carried on the
   routable node as `Node.PausedBy` with the revision that recorded it) is
   dropped from the owner set before ordering, for concrete, tier and hybrid
   resolution alike; this node's own engine is never dropped by the hint, and
   model-list and pass-through routes still reach a paused peer."

2. Same file, §12.2 "Admission", the paragraph "**A paused node and its own
   requests.**" (`:1139-1142`). Append a new paragraph after it:

   > **A paused peer and this node's requests.** A peer's pause reaches this
   > node as the `pausedBy` of the policy advertisement in its pinned
   > `/v1/models` document, fetched by the scanner over cluster mTLS and
   > relayed in the `discovery:nodes` snapshot; it is never read from mDNS.
   > `resolveCandidates` skips such a peer while resolving owners for a model
   > (§5 step 1), so a paused peer costs no dispatch. The hint is advisory and
   > bounded by the scanner's sweep: a peer that paused since the last fetch
   > still answers its admission `503` (`X-PAIR-Admission: paused`), which is
   > a failed candidate like any other, and a peer that resumed since the last
   > fetch is routed to again after the next sweep. The destination's own
   > admission and local veto always win over the hint.

3. Same file, §1 "Strict readiness and peer capability (A2)" paragraph that
   describes `subscribedToNode` copying `StrictAdmission` (`:160-163`): extend
   the parenthetical list so it reads "`subscribedToNode` copies
   `DirectoryNode.Policy.StrictAdmission` onto the internal
   `Node.StrictAdmission` and `DirectoryNode.Policy.PausedBy` onto
   `Node.PausedBy`, which `nodeEqual` ignores, so readiness and pause changes
   announce no `node/updated`."

4. `services/nvpair-proxy/README.md`: in the opening paragraph that says the
   proxy "replaces its routing overlay from each pushed `discovery:nodes`"
   (`:9-13`), add one sentence at its end: "A node whose advertised policy
   says it is paused stays in the overlay but is not an inference owner until
   its advertisement changes; see the spec, §5 and §12.2."

**Verify:**

- `node scripts/spdx-headers.mjs` passes (no header change expected).
- `git diff --stat` touches only the two files listed; `grep -n "A paused
  peer and this node's requests" services/nvpair-proxy/spec.md` finds exactly
  one line; `grep -c "Node.PausedBy" services/nvpair-proxy/spec.md` is at
  least 2.
- Mac: `cd desktop && npm run service-contracts:check` still passes (spec
  prose is not part of the generated API, so this confirms nothing else was
  touched).
- Windows: not required.

### Task 4 — Scanner: prove the hint is authenticated, revisioned and dynamic

**Files:** `services/nvpair-node-scanner/availability_test.go` (create),
`services/nvpair-node-scanner/README.md` (modify).

**Depends on:** none (tests existing scanner behaviour; no product change).

**Complexity:** mechanical.

**Do:**

1. Create `services/nvpair-node-scanner/availability_test.go` (SPDX header,
   `package main`) using `newRefreshTestDaemon`, `startModelsStub`,
   `modelsStub.set`, `seedEMNode`, `d.refreshNodeModels` and `d.dir.get`
   (`S/refresh_test.go:93`, `:65`, `:37`, `:108`):

   - `TestPauseHintFollowsTheAdvertisement`: an `inventory(revision, pausedBy
     string)` helper returning `{"models":["a"],"policy":{"revision":<rev>,
     "roles":{}}}` plus `"pausedBy":<pausedBy>` when non-empty. Subtests, in
     order on one daemon and stub for `peer-P` at `127.0.0.1`:
     - `available at first`: stub `inventory("1", "")`; first refresh returns
       true; `got.Policy.PausedBy == ""`, `got.Policy.Revision == "1"`.
     - `a pause alone is a change`: stub `inventory("2", "local")`; refresh
       returns true; `PausedBy == nodepolicy.PausedLocally`, `Revision == "2"`;
       models unchanged (`[]string{"a"}`).
     - `an identical paused answer is not`: refresh returns false.
     - `a failed fetch keeps the pause`: stub `set(http.StatusInternalServerError,
       nil)`; refresh returns false; `PausedBy` still `local`, `Revision` still
       `"2"`; `d.lastPolicy["peer-P"]` (under `d.infoMu`) equals the stored
       advertisement.
     - `a resume clears it under a newer revision`: stub `inventory("3", "")`;
       refresh returns true; `PausedBy == ""`, `Revision == "3"`.
     - `a peer pause is carried as peer`: stub `inventory("4", "peer")`; refresh
       returns true; `PausedBy == nodepolicy.PausedByPeer`.
   - `TestPauseHintEmitsOneNodeUpdated`: `d := newRefreshTestDaemon()`, then
     capture the codec output the way `peerServicesWithCapturedCodec` does
     (`S/reachability_report_test.go:28-48`): `var out bytes.Buffer; d.codec =
     NewCodec(&out)`, and a local `sent()` that decodes `out.Bytes()` with
     `json.NewDecoder` into `[]Message`, keeping the frames whose `Method ==
     noderec.NotifyNodeUpdated`. `emit` is nil-codec-safe
     (`S/daemon.go:579-586`), so the codec must be set before the refresh.
     Seed `peer-P`, populate with `inventory("1", "")` (one refresh), reset
     `out`, then flip to `inventory("2", "local")` with one refresh and assert
     `sent()` holds exactly one frame; decode its `Params` as
     `noderec.NodeEvent` and assert `Node.HostUUID == "peer-P"` and
     `Node.Policy.PausedBy == nodepolicy.PausedLocally`. A third refresh with
     the same body must add no frame.
   - `TestPauseHintIsNeverFetchedWithoutAPin`: `clusterDir :=
     filepath.Join(t.TempDir(), "cluster"); d := newMembershipTestDaemon(clusterDir)`
     (`S/cluster_membership_test.go:17`, the unclustered state of
     `TestModelFetchTransportFollowsMembership`, `:39-55`); assert
     `d.modelsClient("", "192.0.2.10", 14317, "principal-stranger", true)`
     returns `ok == false`, and that `d.fetchModelsForPeer("peer-X",
     "192.0.2.10", 14317, "principal-stranger", true)` returns `ok == false`
     within one second (run it in a goroutine, `select` on its result and
     `time.After(time.Second)`, fail on the timeout): the address is in
     TEST-NET-1, so a return inside the deadline shows no dial was attempted.

2. `services/nvpair-node-scanner/README.md:32`, the bullet beginning
   "**Enriches** each discovered node": after "The same document carries the
   node's policy advertisement (`policy`: declared roles, policy revision,
   pause origin), which becomes the entry's `policy`;" insert "the pause origin
   is the authenticated, revisioned hint a peer's proxy uses to stop dispatching
   to a paused node, and it changes only through this fetch, never through the
   mDNS record;" keeping the rest of the sentence.

**Verify:**

- Mac: `cd services/nvpair-node-scanner && go vet ./... && go test -race -run
  'TestPauseHint' ./...` passes with the three tests `PASS` and none skipped.
- Mac: `cd services/nvpair-node-scanner && go test -race ./...` passes.
- `node scripts/spdx-headers.mjs` passes.
- Windows: not required (the scanner tests listen on `127.0.0.1:0` through
  `startModelsStub`; if PC3 runs the module anyway, run it serialized as
  `services/readme.md:237` says).

### Task 5 — Broker: pin the provenance and revision of the published hint

**Files:** `services/nvpair-ui-broker/availability_test.go` (create),
`services/nvpair-ui-broker/README.md` (modify).

**Depends on:** none (tests existing broker behaviour; no product change).

**Complexity:** mechanical.

**Do:**

1. Create `services/nvpair-ui-broker/availability_test.go` (SPDX header,
   `package main`) with `TestPauseProjectionCarriesProvenanceAndRevision`:
   `ab := newActivatedBroker(t)`; `engine := attachFakeEngine(t, ab.b)`;
   `ctx := context.Background()`. Each subtest records `before :=
   len(nodeProjections(t, engine))` and `prev := ab.b.localPolicy().Policy.Revision`,
   performs one transition, then `got := waitProjections(t, engine, before+1)`,
   `requireLastProjectionIsStored(t, ab.b, got)` and asserts on
   `last := got[len(got)-1]`:
   - `local pause advertises local under a new revision`:
     `ab.b.setAvailability(ctx, nodepolicy.Paused, nodepolicy.PausedLocally)`
     → `last.PausedBy == nodepolicy.PausedLocally`, `last.Revision != prev`,
     `last.Revision == ab.b.localPolicy().Policy.Revision`.
   - `a peer pause over a local pause stays local`:
     `ab.b.setAvailability(ctx, nodepolicy.Paused, nodepolicy.PausedByPeer)`
     returns no error and `nodepolicy.Paused`; the stored policy's `PausedBy`
     is still `local` (`ab.b.localPolicy().Policy.PausedBy`) and its revision
     equals `prev` (an already-paused node is not paused again when the
     provenance class is unchanged, `B/availability.go:174-181`). The origin
     is rewritten to local before acceptance (`:59-65`), so the local-pause
     branch publishes the authoritative projection (`:87-93`):
     `waitProjections(t, engine, before+1)` and assert the last one has
     `PausedBy == nodepolicy.PausedLocally` and `Revision == prev`.
   - `a peer resume of a local pause is vetoed and publishes nothing new`:
     `_, err := ab.b.setAvailability(ctx, nodepolicy.Available,
     nodepolicy.PausedByPeer)`; `var veto *localVetoError; errors.As(err,
     &veto)` is true; `len(nodeProjections(t, engine)) == before`.
   - `local resume clears the hint under a newer revision`:
     `ab.b.setAvailability(ctx, nodepolicy.Available, nodepolicy.PausedLocally)`
     → `last.PausedBy == ""`, `last.Revision != prev`.
   - `peer pause advertises peer`: `ab.b.setAvailability(ctx, nodepolicy.Paused,
     nodepolicy.PausedByPeer)` → `last.PausedBy == nodepolicy.PausedByPeer`;
     then `ab.b.setAvailability(ctx, nodepolicy.Available,
     nodepolicy.PausedByPeer)` succeeds and the next projection has
     `PausedBy == ""`.
   Use `t.Helper()` on any local assertion helper and `t.Fatalf` with the
   projection printed on mismatch.

2. `services/nvpair-ui-broker/README.md`, the "**Node projection.**" bullet
   (`:496-504`): after "engine-manager serves `policy` to paired peers in its
   pinned `GET /v1/models` document and enforces `pausedLocally`." add: "A
   peer's proxy reads that `pausedBy` from its discovery snapshot and stops
   offering this node as an inference owner while it is set; the hint is
   advisory, and this node's admission and local veto still decide."

**Verify:**

- Mac: `cd services/nvpair-ui-broker && go vet ./... && go test -race -run
  'TestPauseProjectionCarriesProvenanceAndRevision|TestNodeProjection|TestPause|TestPeer'
  ./...` passes with the new test `PASS` and none skipped.
- Mac: `cd services/nvpair-ui-broker && go test -race ./...` passes.
- `node scripts/spdx-headers.mjs` passes.
- Windows: not required.

### Task 6 — Cross-process acceptance: a pause reaches the broker's directory without an mDNS change

**Files:** `services/tests/paused_advertisement_test.go` (create),
`services/tests/broker_test.go` (modify, `availableNode` only).

**Depends on:** none (acceptance of existing scanner→broker behaviour).
External prerequisite: a Windows machine configured per
`services/readme.md:215-246` (PC3).

**Complexity:** mechanical.

**Do:**

1. In `services/tests/broker_test.go`, add to `type availableNode struct`
   (`:47-56`) the field

   ```go
   	Policy *nodepolicy.Advertisement `json:"policy,omitempty"`
   ```

   and the import `"nvpair-shared/nodepolicy"` (already required by the
   module, `X/go.mod:8`). No other change to the file.

2. Create `services/tests/paused_advertisement_test.go` (SPDX header,
   `package tests`) with `TestPAPausedAdvertisementConvergesWithoutMDNSChange`.
   Add a local gate `pausedAdvertisementPlatform(t)` identical in effect to
   `strataContractPlatform` (`X/strata_contract_test.go:276-286`): skip on
   darwin with the message "PA cross-process acceptance runs on Windows PC3;
   the Mac runs no PAIR processes"; on Windows fail unless `GOFLAGS` contains
   `nvpair-test-exec` and `NVPAIR_TEST_BIN_DIR` is absolute. Then copy the
   structure of `TestModelsRefreshConvergesWithoutMDNSChange`
   (`X/models_refresh_test.go:57-150`) with instance name
   `xproc-paused-advertisement-1` and TXT
   `{"v=1", "uuid=xproc-paused-advertisement-1-uuid", "ip=127.0.0.1", "em=<port>"}`:
   - Phase 1: stub body `{"models":["llama3:8b"],"modelsByEngine":{"ollama":
     ["llama3:8b"]},"policy":{"revision":"5","roles":{}}}`; `pollForNode(...,
     30*time.Second, func(n availableNode) bool { return n.Policy != nil &&
     n.Policy.Revision == "5" && n.Policy.PausedBy == "" })`.
   - Phase 2 (pause, no mDNS change): swap the body to the same with
     `"revision":"6","pausedBy":"local"`; poll 45 s until `n.Policy.Revision ==
     "6" && n.Policy.PausedBy == nodepolicy.PausedLocally` and
     `len(n.Models) == 1` (the inventory survives the pause).
   - Phase 3 (resume): swap to `"revision":"7"` with no `pausedBy`; poll 45 s
     until `n.Policy.Revision == "7" && n.Policy.PausedBy == ""`.
   - Phase 4 (stale fetch keeps the last hint): make the stub answer
     `http.StatusInternalServerError` (define a sibling stub type in this
     file whose handler answers a settable status and body; do not modify
     `models_refresh_test.go`), wait `20 * time.Second` (more than one
     `peerRefreshInterval`), then one `discovery:get-nodes` must still show
     `Revision == "7"` and `PausedBy == ""` for the node.
   Record in the test's doc comment that the stub is dialed over loopback in
   plaintext, which is the scanner's self path; the pinned-mTLS peer path is
   covered by `S/cluster_membership_test.go` and Task 4.

**Verify:**

- Mac: `cd services/tests && go vet ./... && go test -run
  TestPAPausedAdvertisementConvergesWithoutMDNSChange -list . ./...` prints the
  test name (compiles and lists; the Mac never runs it).
- PC3 (Windows, serialized, with the fixed-path wrapper and firewall rules
  from `services/readme.md:215-246`): `cd services/tests && go test -run
  TestPAPausedAdvertisementConvergesWithoutMDNSChange -count=1 -v ./...`
  passes with `--- PASS` for the test and all four phases logged; the run
  must not report `SKIP` on Windows.
- PC3: `go test -run TestModelsRefreshConvergesWithoutMDNSChange -count=1
  ./...` still passes (the shared `availableNode` change is additive).
- `node scripts/spdx-headers.mjs` passes.

### Task 7 — Product documentation: node availability

**Files:** `docs/node-availability.mdx` (create), `fern/docs.yml` (modify),
`README.md` (modify).

**Depends on:** Task 2 (the page describes the shipped routing rule).

**Complexity:** judgment.

**Do:**

1. Create `docs/node-availability.mdx` with the SPDX HTML comment header used
   by the other `.mdx` files (copy the first four lines of
   `docs/strict-local-admission.mdx`). Title: `# Node availability`. Sections
   and the facts each must state (write in the voice of
   `docs/strict-local-admission.mdx`; say "engine", never "backend"; no
   host names, addresses or user names):
   - **What pausing does**: a paused node runs no inference work and never
     wakes an engine; requests entering a paused node are still routed to
     other machines (`N/nodepolicy/policy.go:44-47`). Pause options: what to do
     with running work, drain timeout, unload models, stop engines
     (`policy.pause`, `B/availability.go:174-279`). `draining` is the
     transition state and is never saved.
   - **Who paused it (provenance)**: `pausedBy` is `local` for a pause made on
     the node itself, `peer` for one relayed from a paired node; a resume clears
     it; configuration edits never change it. A pause saved before the field
     existed counts as local.
   - **Local veto**: a node paused locally refuses a paired node's resume
     (`-32011`, "this node was paused on the node itself; resume it there"),
     and engine-manager refuses paired nodes' engine starts and model loads
     while the node is paused locally; a paired node may resume a peer pause;
     a local pause over a peer pause becomes local and is never downgraded;
     remote configuration edits stay allowed. Point to
     `services/nvpair-ui-broker/README.md` "Local veto" for the exact rules.
   - **How paired nodes learn about a pause (the advertisement)**: the node's
     engine-manager serves `policy.pausedBy` and `policy.revision` in its
     pinned `GET /v1/models` document; a paired node's scanner fetches it over
     cluster mTLS (only for nodes it holds a pin for), re-fetches every 15
     seconds and on reconnect, keeps the last good answer across a failed
     fetch, and never reads it from mDNS. State the latency bound plainly: a
     pause or resume is visible to other nodes within about 15 seconds plus
     one fetch.
   - **What the router does with it**: a node that advertises a pause is not
     offered as an owner for a model, so no request is sent to it; model lists
     still include it; the router's own engine is governed by its own
     admission, not the hint. When every owner of a model is paused the router
     answers `502` "no available node advertises the requested model". A node
     that paused since the last fetch still answers `503` with
     `X-PAIR-Admission: paused` and the router moves to the next node; a node
     that resumed since the last fetch is used again after the next fetch. The
     destination's own admission and local veto always win.
   - **Seeing it**: the desktop node card and the terminal interface show
     "Paused (local)" / "Paused (peer)" from the same `pausedBy`
     (`services/nvpair-tui/ui/availability.go:96-98`); the proxy logs
     `resolveCandidates: skipping paused peer` at debug level with the node id,
     provenance and policy revision.
   - **Limits**: no push faster than the sweep; no revision-order guard; a
     manual (user-added) node carries no advertisement and is always tried.

2. `fern/docs.yml`: in the `Reference` section, after the `Local Qualification`
   entry (`:64-66`), add

   ```yaml
         - page: Node Availability
           path: ../docs/node-availability.mdx
           slug: node-availability
   ```

3. `README.md`, the numbered reading order (`:226-262`): extend item 8
   (Architecture) with a second sentence: "**[Node availability](docs/node-availability.mdx)**
   explains pausing a node, who may resume it, and how paired nodes stop
   routing to it." Keep the numbering unchanged. This follows
   `.cursor/rules/readme-ownership.mdc` (a link, no implementation notes).

**Verify:**

- `node scripts/spdx-headers.mjs` passes (the new `.mdx` carries the header).
- `grep -n "node-availability" fern/docs.yml README.md` shows exactly one hit
  in each file.
- `grep -c "backend" docs/node-availability.mdx` prints `0`.
- Every relative link in the new page resolves: for each `](../services/...)`
  or `](...)` target, `test -e docs/<target>` (or the repository-relative
  path) succeeds.
- If the Fern CLI is installed locally, `cd fern && fern check` passes;
  otherwise record "fern check not run" in the task's commit message body.
- Windows: not required.

## Verification (whole-plan)

Run after every task has landed on the branch, from the repository root:

- `node scripts/spdx-headers.mjs` — no missing header.
- Mac: `make test-services` (or `for m in services/nvpair-proxy
  services/nvpair-node-scanner services/nvpair-ui-broker; do (cd $m && go
  test -race ./...); done`) passes; Mac also compiles `services/tests` with
  `go vet ./...` and `go test -list . ./...` but never runs its cases.
- Mac: `cd desktop && npm run service-contracts:check && npm run lint && npm
  run typecheck && npm run test:unit && npm run dead-code:check` pass with no
  desktop diff (PA changes no desktop file).
- PC3 (Windows, serialized): `cd services/tests && go test -run
  'TestPAPausedAdvertisementConvergesWithoutMDNSChange|TestModelsRefreshConvergesWithoutMDNSChange'
  -count=1 ./...` passes; `go test ./...` in `services/nvpair-proxy`,
  `services/nvpair-node-scanner` and `services/nvpair-ui-broker` passes without
  `-race`.
- MSI (Windows + RTX): the three Go modules above with `-race` where the
  toolchain allows it; no live engine is needed for PA.
- Manual two-node check (MSI as destination, PC3 as router, both clustered;
  no Mac): pause MSI from its own desktop; within 20 s a request on PC3 for a
  model only MSI serves answers `502 no available node advertises the
  requested model` without a request appearing in MSI's proxy log; resume MSI;
  within 20 s the same request is served by MSI. Then pause MSI and, before
  PC3's next sweep (within 15 s), send the request from PC3: it is answered
  by MSI's `503 X-PAIR-Admission: paused` or by another owner, never hangs,
  and PC3's proxy log shows the failed candidate. Record both observations in
  the pull request description's `pair-release-intent:v1` block as the
  user-facing change ("a paused node is no longer tried by paired routers")
  and declare a bump for `nvpair-proxy` only.
- `git log --oneline fork/groundwork..HEAD` shows one commit per task, each
  made with `git commit --no-signoff` and a conventional subject
  (`feat(proxy): ...`, `test(node-scanner): ...`, `test(broker): ...`,
  `test(services): ...`, `docs(proxy): ...`, `docs: ...`).

## Do not

- Do not add a `paused`, `availability` or `pausedBy` key to the `_nvpair-node`
  TXT record, to `noderec.NodeRecord`, or to the peer-services descriptor.
- Do not make the proxy skip its own (self) candidate, the dormant self
  candidate, or a manual node on the hint; do not consult the hint for
  model-list or pass-through routes.
- Do not change the destination veto (`B/availability.go`, `B/policy.go`,
  `E/nodeprojection.go`), the policy relay, or any JSON-RPC method or payload.
- Do not edit `P/http_handler.go`, `P/policy.go`, `B/policy.go`, `B/proxy.go`
  or `N/nodepolicy/wire.go` (I and G own those seams right now).
- Do not add a revision-order guard, a push notification, or a cache of
  pause state anywhere; the scanner's last-good rule and sweep are the
  freshness model.
- Do not edit `desktop/docs/services-api.md`, `CHANGELOG.md` or
  `services/versions.json` by hand; do not regenerate the services API (no
  wire change).
- Do not write the word "backend" in new copy or comments; do not log request
  bodies, prompts, PINs or key material (the new debug line logs node id,
  provenance and revision only).
- Do not run `services/tests` on the Mac; do not run Windows listener tests in
  parallel with another machine-wide invocation.
- Do not fold tests into `P/subscribed_test.go`, `S/refresh_test.go` or
  `B/policy_projection_test.go`; keep PA's tests in the three new
  `availability_test.go` files.

## If stuck

- **Go toolchain**: the modules need Go 1.25 (`go 1.25.0` in `go.mod`). If
  `go test` refuses to build, run with `GOTOOLCHAIN=go1.25.0` set in the
  environment (the Go command downloads that release); do not lower the `go`
  directive.
- **Task 1/2 compile errors around `nodepolicy.PauseOrigin`**: `P/discovery.go`
  already imports `nvpair-shared/nodepolicy` (`:24`); if the type is reported
  missing the import list was edited by mistake.
- **`TestResolveCandidatesSkipsPausedPinnedPeer` yields no candidates at
  all**: the pinned fixture needs the principals joined with
  `clustertrusttest.Join` and `p.mesh` opened before `resolveCandidates`; copy
  `P/subscribed_test.go:368-380` exactly, then re-run. If it still fails,
  reduce the test to `TestResolveCandidatesSkipsPausedPeersForAModel` (manual
  nodes) and report which assertion failed together with the
  `resolveCandidates: ...` debug lines (add a temporary
  `slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})))`
  at the top of the failing test while diagnosing; remove it before
  committing).
- **`TestPausedSelfKeepsItsAdmissionPath` routes to the peer first**: the
  priority list is `["self","peer"]` and the self node must resolve to the
  facade's own port; confirm the fixture uses `tc.profile.FacadePort` for both
  `selfNode` and `testProxy`.
- **Scanner notification capture decodes nothing** (Task 4, second test):
  the captured frames are `jsonrpc.Message` values (`Params json.RawMessage`,
  `N/jsonrpc/jsonrpc.go:28-32`), written by `NewCodec(&out)`; confirm the
  codec was assigned before the refresh that flips the hint, and that `out`
  was reset after the populating refresh so the first `node-updated` is not
  counted. If `Message` is not visible in `package main`, use
  `jsonrpc.Message` with the import `"nvpair-shared/jsonrpc"`.
- **Task 6 phase 2 never converges on PC3**: confirm the broker's scanner
  logs a `/v1/models` fetch against the stub port every 15 s; if fetches
  stop, the mDNS responder was not kept running (check the `zsrv.Run`
  goroutine and cleanup ordering copied from `models_refresh_test.go`). If the
  broker's row lacks `policy` entirely, verify `availableNode.Policy` was
  added with the exact JSON tag `policy`.
- **Rebase conflicts in `P/proxy.go`** after I Tasks 3-5 land: re-apply the
  Task 2 guard at the point where a non-self peer candidate is appended, with
  the same three conditions (`model != ""`, not self, `PausedBy != ""`), and
  re-run the Task 2 verify line. If `resolveCandidates` no longer exists under
  that name, stop and report the new function and file instead of guessing.
- **Stop and report** (do not improvise) if: the Task 2 tests show the
  self candidate being skipped; `service-contracts:check` reports a diff;
  `nodeEqual` has gained a comparison of policy fields; or any Windows gate
  fails for a reason outside the files this plan names.

## Owner questions

1. **Answer when every owner of a model is paused.** With the hint, a request
   for a model served only by paused nodes gets the existing immediate `502`
   `no available node advertises the requested model` (`P/http_handler.go:761-767`)
   instead of today's slower path (dispatch, `503 paused`, back-off,
   re-resolve, final `503`). Keep the `502` (plan default, no change to
   `P/http_handler.go` while I reworks dispatch), or add a distinct
   `503 X-PAIR-Admission: paused` reply naming that every owner is paused?
   The second is more informative to a client but is a new response shape and
   touches I's seam.
2. **Propagation latency.** The hint reaches peers within one scanner sweep
   (`peerRefreshInterval`, 15 s) plus one fetch. Acceptable, or should PA add a
   push (for example a `discovery:refresh-peer` nudge after a pause, or the
   pause carried on the cluster-manager reverse session)? A push is a new wire
   message across scanner, broker and cluster-manager and is out of PA's scope
   as planned.
3. **Model listing while paused.** The plan keeps a paused peer's models in
   the federated model list and keeps its declared tiers in the hybrid alias
   list, so clients see a stable catalogue; only dispatch skips it. Confirm,
   or should a paused peer also disappear from the model list?
4. **Revision-order guard.** The scanner applies the last fetch that lands;
   two overlapping fetches for one peer could briefly re-apply an older
   advertisement until the next sweep. The plan adds no guard, because a node
   whose policy file is reset legitimately moves to a lower revision and a
   guard would stick until eviction. Confirm no guard, or accept that risk for
   a "revision never decreases" rule in `applyModels`?
