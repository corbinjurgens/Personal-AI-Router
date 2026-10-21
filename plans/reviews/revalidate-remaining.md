<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Review: revalidate remaining plan tasks against current code (slug: revalidate-remaining)

Kind: plan freshness review. Investigation only; no product change, no pull request.

Question: can an executor that follows each remaining B, J, PA and SR task literally still succeed at the current `fork/groundwork`, and in what order should the remaining tasks run across the fleet alongside the remaining I tasks?

## Verdict

`RED 0 UNVERIFIED 4 LOW 6`

Task tally (19 tasks in scope): **READY as written 0 · READY with listed edits 8 · BLOCKED 11 · OBSOLETE 0.**

| Package | READY with edits | BLOCKED (on) |
|---|---|---|
| B | T2, T3, T4, T5 | T6 (I T6 docs join) |
| J | - | T2 (I T6 `routing_wire.go`), T3 (I T4b/T5/T6), T4 (J3), T5 (J3/J4), T6 (J3; PC3) |
| PA | T1 (split: `discovery.go` now, `proxy.go` line after I T4b), T7 (after T2 semantics) | T2 (I T4b landing fixes the candidate owner), T3 (I T6 then SR 6a docs) |
| SR | 5a-ii, 5b | 5c (G T3; I T6 broker files), 6a (5c), 6b (5b, 5c; PC3) |

No task is OBSOLETE: nothing landed since the plans were written already does any remaining task's work. Every remaining task's premise still holds at the inspected head except where listed under "Stale references" below, and all stale items are line numbers, moved symbols, landed-gate statements or superseded helper claims; none changes a task's design.

## Inspection base and environment

- The common brief names `fork/groundwork` at `210babf`. The branch has moved: `origin/fork/groundwork` = **`8a63866c1cbadd70f40732c84b04efb72d6e2fd4`**, two first-parent merges later (`8739691` NE Task 5 Part B: strict readiness bounded by serving envelopes; `8a63866` I public-lifecycle test fix). Every `path:line` below is at `8a63866`.
- Landed units since the plans' own baselines, from `git log --first-parent`: I T3b/T8 `9681775`, XA `721c9ef` and `d5a636a`, PA T5 `26606f5`, J T1 `4bb1d8d`, E4 desktop `d8772c8`, SR 5a-i `e6a8f73` (+ `aebf553`), NE T4 Part A `75861b7`, scanner `--mdns` `fe68c81`, NE T6 Part C `8f6dd0f`, I T4a `d0fb979`, B T7 `0468344`, PA T6 `210babf`, NE T5 Part B `8739691`. I T4b is reported as nearly done on MSI; its branch is not fetched here, so its exact file set is INFERRED from the I plan's Files list only.
- Plans read (all via `git show`, no checkout): `plan/b-opencode-shapes:plans/b-opencode-shapes.md` (amendment r4), `plan/j-routing-diagnostics:plans/j-routing-diagnostics.md`, `plan/pa-paused-advertisement:plans/pa-paused-advertisement.md`, `plan/cloud-investigations:plans/context/{sr5-rescope,i-unified-dispatch,g-egress-lease}.md`, `plan/ne-serving-envelope:plans/ne-serving-envelope.md` (Files only), and the finished audits `plan/inv-audit-ne-sr:plans/reviews/audit-ne-sr.md`, `plan/inv-audit-pa-j-b:plans/reviews/audit-pa-j-b.md`, which are cited where they already prove a point.
- Toolchain: container Go 1.24.7; with `GOTOOLCHAIN=auto` Go 1.25.0 resolved. Executed here (VERIFIED): `go test -list` in `services/nvpair-proxy` for every test name the B, PA and SR plans select (all resolve; see per-task tables). Not run: any test case, `-race`, desktop npm checks, `services/tests`, engines, models. Where the two audits already executed a selection at this same SHA, that result is cited rather than repeated.
- VERIFIED means the file/symbol/line was read here or a command ran; INFERRED is marked.

## Findings

### UNVERIFIED-1: B Tasks 2-6 still wait on the replay owner answer, which no public branch records

- Claimed invariant (B plan, Owner questions): "Should B accept and forward assistant reasoning_content unchanged when exact NE-bound template evidence qualifies it, while refusing unqualified replay? ... Tasks 2-6 wait for this answer and their existing external gates."
- Probe: the B plan on `plan/b-opencode-shapes` (through amendment r4) and `plans/context/*` on `plan/cloud-investigations` carry no answer. Every other B T2 gate is now satisfied by code (NE Part A `75861b7`; I T3a fixtures `0d9d8cf`; I T4a `d0fb979` released `local_qualification.go`; B T7 `0468344`).
- Observed result: B T2-T5 are code-ready but gated on one owner sentence. The `shapes.reasoningReplay` member (T2 Do 2) and the replay positives in T3 Do 4 / T5 Do 2 are the only parts that depend on it.
- Affected state: the whole B lane, which is otherwise disjoint from every remaining I file (see overlap table).
- Disposition: owner question. If the owner wants B moving before deciding, amend T2 to land `shapes.tools` and `shapes.structuredOutput` with `reasoningReplay` present but required-false until the answer; the plan currently forbids that split, so it needs an explicit amendment, not an executor choice.

### UNVERIFIED-2: PA T2 and J T3 cannot name their final proxy files until I T4b lands

- Claimed invariant (PA plan, Change scope): "At join12 that owner is `facade.resolveCandidates` in `P/proxy.go`; do not assume I leaves it there." (J plan, T3 Do 1): "Listed proxy routing.go/request_spend.go/http_handler.go/proxy.go are candidate observation sites ... replace moved paths with actual owners."
- Probe (VERIFIED at `8a63866`): `resolveCandidates` is still `services/nvpair-proxy/proxy.go:631`; `services/nvpair-proxy/routing.go` and `response_sink.go` do not exist; I T4b's Files create both and edit `proxy.go`, `http_handler.go`, `cancel.go`, `request_registry.go`, `request_spend.go`. The I T4b branch is not visible here.
- Observed result: PA T1's `proxy.go` edit (`subscribedToNode`, `proxy.go:1324-1362`) and PA T2's exclusion site, and J T3's observation hooks, are each inside an in-flight I T4b file. Whether `resolveCandidates` moves into `routing.go` (which I T5 then also owns) decides PA T2's Files.
- Disposition: defer PA T1's `proxy.go` line and all of PA T2 until the coordinator records the accepted I T4b SHA, then amend PA Files once (see "Replacement text", PA T2). PA T1's `discovery.go` and new test need no wait.

### UNVERIFIED-3: the G3 / SR 5c ordering text is still contradictory on the plan branches read here

- Claimed invariant (SR plan, Context): "Governing cross-package broker order: I3 (3a then 3b) -> G3 -> SR5c ... Coordinator must reconcile G's stale SR-first text before dispatch." (G plan, Task 3 Depends on): "then SR 5 broker relay integrated/released".
- Probe: `plans/context/g-egress-lease.md` still states the SR-first prerequisite; `services/nvpair-ui-broker/` has no `egresslease.go` (VERIFIED by directory listing), so G T3 is unlanded; I T6 (broker `broker.go`, `client_requests.go`) is unstarted.
- Observed result: SR 5c is blocked twice over (G T3 by the governing order; I T6 by the "overlapping I8/I6 broker reservations completed" clause). The coordinator ledger that would record the reconciliation is private and not visible.
- Disposition: amend plan (G): replace G T3's SR-first prerequisite with the I-amendment order so that G T3 can dispatch right after I T6; otherwise neither G T3 nor SR 5c has a literal green light.

### UNVERIFIED-4: race, desktop, Windows and process gates not executed here

Every Verify block in scope requires at least one of `-race` on Mac, `npm --prefix desktop run ...`, PC3 `services/tests` or MSI native runs. This container executed none of them. The plain-run results the two audits recorded at `8a63866` stand; anything beyond is pending the fleet.

### LOW-1: B plan line references are stale throughout its Context section

All B Context citations predate I T4a's rewrite of `local_qualification.go` (`d0fb979` moved validation to borrowed views) and B T7. None of the described behaviour changed except where B T7 added `store`. Current locations are in the B table below. No executor effect beyond lookup time, provided the Do items are read against the table.

### LOW-2: J plan Context carries three superseded facts

`temporaryClusterPort` now rejects protected ports (XA `d5a636a`: `services/tests/cluster_outbound_pairing_test.go:33-43` -> `fixture_ports_test.go:40-55`); `B/workload_projection.go:164-224` describes pre-J1 code (J1 landed `4bb1d8d`); `maxTabs` is in `services/nvpair-tui/ui/keys.go:119`, not `ui.go`. J T6's "separately reserved isolation/port helper repairs" prerequisite is therefore already satisfied.

### LOW-3: PA plan Context says I T3b/T4a/T8 are in flight; they landed

`9681775` (3b/8) and `d0fb979` (4a) are first-parent ancestors of `8a63866`. Only I T4b is in flight; I T5/T6 are unstarted. PA Tasks 4-6 lack "done" marks although they landed (`b00f597`, `26606f5`, `210babf`). Audit RED-1 (`audit-pa-j-b.md`, broker README promising router-side exclusion) is still open at this head and closes only when PA T2 lands.

### LOW-4: SR plan Context predates NE Task 4 Parts A/B/C and I T4a

"NE T4 is unstarted", "`serving_identity.go`/`serving_readback.go` do not exist", "held behind MSI I 4a's proxy `http_handler.go`/`ingress.go` reservation" are all false at `8a63866` (VERIFIED: `services/nvpair-engine-manager/serving_identity.go:146`, `serving_readback.go:105`, `services/nvpair-proxy/readiness_contract.go:152-158`, `d0fb979` merged). Coordinator amendment r2's two deferrals now have a contract to bind to (see 5b). Task 501 lacks its "done" mark (`e6a8f73`).

### LOW-5: B Task 7 and J Task 1 headings carry no landed mark

B T7 landed at `0468344`; J T1 at `4bb1d8d`. The B plan has no lifecycle section by design; still, an executor reading "Task 7 can proceed after its I 4a gate" (Task 1 Status) will look for work that is done. Add the marks when the plans are next touched.

### LOW-6: B's "coordinated schema integer" instruction resolves to a single value and should say so

B T2 Do 1/Decisions: "use NE's agreed combined incompatible revision when it already includes B's exact contract; otherwise advance the accepted NE artifact version by one." NE Part A touched no proxy file and no qualification document version (NE plan Task 4 Part A Files; `services/nvpair-proxy/local_qualification.go:43` `qualificationDocumentVersion = 1`; `docs/local-qualification.mdx:30` `"version": 1`; `testdata/local-qualification.json` version 1). The coordinated integer is therefore **2**, owned by B. Record it in the plan rather than at execution time.

## Per-task revalidation

Status key: READY = ready as written; READY+ = ready with the listed edits; BLOCKED (on what); OBSOLETE.

### B Tasks 2-6 (`plan/b-opencode-shapes`)

Premise check (VERIFIED): `validateRequest` still checks raw keys before the merge and never re-checks the allowlist after it (`services/nvpair-proxy/local_qualification.go:324-345` raw loop; `:352-365` post-merge does only unique-key, object and `store` strip), so T3's general effective-key gap is real at this head. `accountTools` accepts `tools: []` and `tool_choice` without tools (`:732-771`); `accountResponseFormat` accepts text/json_object/named json_schema (`:773-799`); roles/null-content rule at `:644-708`. `Profile.Capabilities` exists (`services/shared/nodepolicy/policy.go:344`, kinds `:70-81`) and reaches `qualifiedModelLocked(k, pr)` (`services/nvpair-proxy/readiness_contract.go:127`), so T4's predicate has its inputs. NE Part B's envelope guard is at `readiness_contract.go:152-158`; the B T4 gate "accepted NE Task 4 Part B integrated" is satisfied by `8739691`.

Stale references (B Context; old -> current at `8a63866`):

| Old | Current |
|---|---|
| `local_qualification.go:39` version | `:43` (`qualificationDocumentVersion = 1`) |
| `:98-190` decoding/registry validation | `:102-163` `validate`, `:165-197` `parseQualificationDocument` |
| `validateRequest:251-309` | `:299-399` |
| `chatRequestKeys:242-245` | `:290-293` |
| `chat_adapter.go:446` appends `parallel_tool_calls` | `:492` |
| `:378-469` message roles/content | `accountMessages` `:644-708`, `contentText` `:710-730` |
| `accountTools:471-516` | `:732-771` |
| `accountResponseFormat:520-545` | `:773-799` |
| `tiers.go:86-88` tools needs from nonempty arrays | `tiers.go:76` |
| `readiness_contract.go:125-172,212-227` | `qualifiedModelLocked` `:127-163`; `qualifyStrict` `:257-282` |
| `ingress.go:310-350` qualifies before admission | `:367-402` (`qualifyStrict` `:367`, 400 `strict-qualification` `:372-374`, `AdmitStrict` `:402`) |
| `local_qualification_test.go:25-75` helpers; `:460-519` | `testQualification` `:28`, `testIdentity` `:41`, `installQualified` `:53`; `TestStrictHTTPRejectsUnqualifiedRequestsBeforeSideEffects` `:738`, `TestStrictHTTPForwardsTheEffectiveBody` `:799` |
| `admission_http_test.go:280-300` recording engine | `recordingEngine` `:272-310`; `strictHTTPFixture` `:28-38` (ollama profile) |
| `cors_test.go:433-450` | `:437-438` |
| "I Task 4a on MSI currently also holds `local_qualification.go` and `readiness_contract.go`" | landed `d0fb979`; both files released |
| "JOIN15 I3b/I8 remains pending" | landed `9681775` |
| Task 7 (store:false) | landed `0468344`; `checkQualificationStore` `:401-411`, `stripQualificationStore` `:413-453`, `TestQualificationStoreNoOp` `:371` |

Test names in each Verify resolve at `8a63866` (VERIFIED by `go test -list`): `TestQualificationFixtureCases`, `TestQualificationContextBoundary`, `TestQualificationAppliesProfileDefaults`, `TestQualificationDocumentRejects`, `TestLoadQualificationRegistry`, `TestStrictCapabilityDerivation`, `TestStrictQualifyThenAdmitIsFencedByReinstallation`, `TestStrictHTTPRejectsUnqualifiedRequestsBeforeSideEffects`, `TestStrictHTTPForwardsTheEffectiveBody`, `TestChatRequest`, `TestChatUnsupported`, `TestChatHistory`, `TestChatAccounting`. New names (`TestQualificationShapesDocument`, `TestQualificationShapeMatrix`, `TestQualificationShapeDefaultsAndBounds`, `TestOpenCodeShapeFixtures`, `TestStrictHTTPRefusesUnqualifiedShapesBeforeSideEffects`, `TestStrictHTTPOpenCodeQualifiedForwarding`) are absent, as expected. A third test file, `local_qualification_merged_test.go`, exists since I T4a; it constructs no artifact literals, so T2 Do 3's `rg` scope needs no widening.

| Task | Status | Notes |
|---|---|---|
| T2 schema | READY+ (gated UNVERIFIED-1) | Version is 2 (LOW-6). Files have no remaining-I overlap. |
| T3 effective-shape validation | READY+ | Depends on T2. Post-merge allowlist site is `:352-365`. |
| T4 capability coverage | READY+ | Part B landed; insert the predicate after `readiness_contract.go:152-158`, before `return m, q, true` at `:163`. |
| T5 synthetic forwarding | READY+ | New files. `strictHTTPFixture` is ollama-based (`admission_http_test.go:30`); a llama.cpp fixture must use `profileFor("llamacpp")` (`engines.go:238`) and `configureStrict`. I T4b's fixture discovery (`rg -l 'attemptClaim|attemptCtl|workloadCtl|installResponseStream|newStreamFrameReader'`) returns `cancel_test.go`, `owned_buffers_test.go`, `performance_test.go`, `retry_test.go`, `streamframe_test.go` only (VERIFIED), none of B's. |
| T6 docs | BLOCKED on I T6 | `services/nvpair-proxy/spec.md` is shared with I T6, PA T3, SR 6a, G T4. `docs/local-qualification.mdx` changed in NE Part B (`8739691`); re-read before editing. The `spec.md:1572` "store rejected" sentence scoped to the public path (audit LOW-7) belongs here. |

### J Tasks 2-6 (`plan/j-routing-diagnostics`)

Premise check (VERIFIED): `RoutingRequestKey` and `decodeStrictContract(data, dst, required...)` exist (`services/shared/nodepolicy/routing_wire.go:27`, `:155`); the registry allocates the key under `identityMu` (`services/nvpair-proxy/request_registry.go:26-46`) and `http_handler.go:536-541` registers before inference and defers cleanup; `routingSpend`, `publicAttemptLifecycle.tryAccount/associated/acceptedByGuard/firstSuccessfulByte/finished` exist (`request_spend.go:21-198`). `Workload`/`workloadParams` are `proxy.go:178-215`. `internalClientMethods` is `services/nvpair-ui-broker/client_requests.go:24-52`, checked at `:178-183` with `-32601`. No `routing:active`/`routing:cancel`/`routing/*` method, no desktop `routing.ts`, no TUI `routingwire.go` exists: I T6 is unlanded, so J2/J3's "extend, do not recreate" instruction has nothing to extend yet. `json-rpc-client.ts:104-107` `parseMessage` still calls `JSON.parse`. TUI `View` interface `services/nvpair-tui/ui/view.go:21-38`; five default views `ui.go:69-73`; `maxTabs = 9` at `keys.go:119`. Only one tracked manifest exists (`services/nvpair-engine-manager/MANIFEST.md`), so J3 Do 10 adds no inventory path. The broker codec does no raw duplicate-key rejection (no such check in `services/nvpair-ui-broker/codec.go`); J T1 resolved the same question for workload frames by token-walking the raw params inside the projection (`workload_projection.go:124-153`), which is the precedent J3 should follow instead of reserving `codec.go`.

Stale references (old -> current):

| Old | Current |
|---|---|
| `P/request_registry.go:26-43` | `:26-46` |
| `P/http_handler.go:540-549` | `:536-541` (workload association `:919`, `:942`) |
| `B/workload_projection.go:164-224` | J1 landed: `projectWorkloadFrame` `:22-99`, `projectedWorkloadInfo` `:157`, snapshot `:173` |
| `B/client_requests.go:178-187` | `:178-185` |
| "`temporaryClusterPort` ... does not exclude protected ports" | now backed by `allocateFixtureLoopback` (`services/tests/fixture_ports_test.go:40-71`; XA `d5a636a`) |
| "maxTabs is 9" (ui.go) | `services/nvpair-tui/ui/keys.go:117-119` |
| "I2/I3a are present subject to the fixture repair" | I2/3a/3b/4a/8 landed; 4b in flight; 5/6/7 unstarted |
| T6 "accepted join12 I3a repair; XA/I7 exact fixture transfer; separately reserved isolation/port helper repairs" | I3a repair and XA landed; port helper landed; only I7 (different files) remains |

| Task | Status | Notes |
|---|---|---|
| T2 schema + corpus | BLOCKED on I T6 | Both edit `services/shared/nodepolicy/routing_wire.go`; I T6 publishes the cancel/active types J2 must preserve. Everything else in T2 is valid as written. |
| T3 atomic RPC | BLOCKED on I T4b, T5, T6 | Files overlap I T4b (`proxy.go`, `http_handler.go`, `request_spend.go`, new `routing.go`), I T6 (`broker.go`, `client_requests.go`, desktop bridge/types, docs, exceptions). Do 1 reconciliation stands. |
| T4 desktop UI | BLOCKED on J3 | `RoutingSettings.tsx`, `WorkloadItemCard.tsx`, `workloads.store.ts`, `init.ts` all exist; new files absent. |
| T5 TUI tab | BLOCKED on J3/J4 | `rpccmd.go`, `README.md`, `docs/terminal-interface.mdx`, `desktop/docs/services-parity.md` exist. |
| T6 PC3 privacy proof | BLOCKED on J3 | `TestWorkloadManagerOutboundBroadcast` `services/tests/workload_interop_test.go:558`, `newInterNodeCluster` `cluster_data_plane_test.go:48`, `startDirectClusterPeer` `:133`, fake Ollama `{"done":true}` all present. Port-helper prerequisite satisfied. |

### PA Tasks 1-3 and 7 (`plan/pa-paused-advertisement`)

Premise check (VERIFIED): proxy `Node` (`services/nvpair-proxy/discovery.go:34-78`) has `Roles`, `ServedModels`, `PolicyRevision`, `StrictAdmission`, no `PausedBy`; `nodeEqual` `:191`; `SetSubscribed` `:165`. `subscribedToNode` (`proxy.go:1324`) copies roles/served/revision/strict at `:1356-1361` and drops `PausedBy`. `resolveCandidates` `proxy.go:631`, stale "TCP-probed" comment `:677`, `targetURL` uses `targets.Prefer` (`facade.go:404-409`). Concrete-empty 502 body `http_handler.go:836`; tier-empty 404 `:821`; `writeAdmissionRejection` sets `X-PAIR-Admission` and `Retry-After` 30 for paused/off (`admission.go:872-890`). `hybridLocalCandidates` `tiers.go:445`, `declaredRoles` `:279`. `TestSubscribedCarriesPolicyWithoutChurn` `subscribed_test.go:214`; `forEachEngine` `enginecase_test.go:115`; `newTestProxy`/`newBoundProxy` `failover_test.go:36,44`; `clustertrusttest.Join`/`clustertrust.Open` used in `admission_http_test.go:525-566`. `services/nvpair-proxy/availability_test.go` does not exist (scanner and broker have their own landed ones), so T1 creates it without collision. Spec sections "5. Request path" `spec.md:382`, "12.2 Admission" `:1081`, "12.4 Routing metadata" `:1280`; README "Node selection" `README.md:352`, "Node policy" `:379`. `docs/node-availability.mdx` absent; `fern/docs.yml:52` has the Reference section; root `README.md:252` is the Architecture entry. The TUI already renders `pausedBy` (`services/nvpair-tui/ui/availability.go`), so T7's "link established TUI behavior" has a target.

Stale references: "I3b/4a/4b/5/6/8 are in flight on MSI" -> 3b/4a/8 landed, 4b in flight, 5/6 unstarted; "Before PA Tasks 1/2 write any proxy file ... I4b lands the sole request owner and releases `P/proxy.go`" is the live gate (UNVERIFIED-2). Tasks 4, 5, 6 landed (`b00f597`, `26606f5`, `210babf`); T6's "loopback plaintext fixture" wording is stale (audit LOW-6: the landed fixture is pinned).

| Task | Status | Notes |
|---|---|---|
| T1 Node carries `PausedBy` | READY+ (split) | `discovery.go` and the new test have no I overlap and can be written now; the one-line `subscribedToNode` copy is in `proxy.go` (I T4b in flight). |
| T2 exclusion at the owner | BLOCKED on I T4b landing | Then READY+ with Files amended to the actual owner file (`proxy.go` or `routing.go`). If `resolveCandidates` moves to `routing.go`, I T5 owns that file next: schedule T2 in the window between I T4b acceptance and I T5 dispatch, or after I T5. |
| T3 spec/README | BLOCKED on I T6 then SR 6a | Shared `spec.md`/`README.md` chain (see order). |
| T7 docs page | READY+ after T2 semantics | Files (`docs/node-availability.mdx`, `fern/docs.yml`, `README.md`) overlap nothing in I/J/SR/B; draft can start now and publish after T2. |

### SR 5a-ii, 5b, 5c, 6a, 6b (`plans/context/sr5-rescope.md`)

Premise check (VERIFIED): NE Task 4 Parts A/B/C are landed. Manager: `currentServingEnvelope(ctx, engine, model) (engines.ServingEnvelope, engineModelIdentity, error)` `services/nvpair-engine-manager/serving_identity.go:146`; `servingRetentionDisabled` (`Retention == "-1"`) `:56`; `observeLlamaServing` `serving_readback.go:105` with the SR lookup contract documented at `:172-193`; `ServingEnvelope{PerSlotContextTokens, Slots, Retention, RouterCapacity, ...}` `services/shared/engines/serving.go:123-135`. `residentLoadCall` is still Ollama-pinned (`residency.go:218-249`, `/api/generate`, `num_ctx`/`keep_alive` body) and `residentCheck` hard-codes `/api/ps` (`:350`); the 5a-i cold bound reads `ServingReadback.Configured` (`:340-344`, `residentConfiguredContext` `:357-376`). Guards to reuse exist: `checkServingLoad` `serving_llamacpp.go:813`, `guardServingLoad` `:765`, `rejectUndeclaredServingLoad` `:827`, `stopServingLoads` `:751`, `st.loads`, `Manifest.operationBinding(op)` `modelops.go:93`, `OperationBinding.modelParam` `:64`. `manifests/llamacpp.json:92` binds `load` -> `load_model` (`POST /models/load`, `body_schema {model: string}`, `:115`) and `loaded_models` is `GET /models` filtered on `status.value == loaded` (`:103-107`). Proxy: `residentContextLocked` (`services/nvpair-proxy/residency.go:100-122`) routes through `qualifiedModelLocked`, so Part B's envelope guard (`readiness_contract.go:152-158`) already binds resident prepare and `residentBindingCurrentLocked` (`residency.go:249-253`) at dispatch. Broker: every symbol 5c names exists (`policyRuntime` `policy.go:99`, `persistPolicy` `:640`, `setPeerPolicyAs` `:574`, `recordActivation` `:1454`, `observeLoaded` `:1073`, `observeIntent` `:1098`, `observeAdmission` `:1530`, `unloadModels` `:1600-1611` with no resident recheck, `idleTick` `idlepolicy.go:51`, `notifyPolicyChanged` `policy_notifications.go:31`, `strictReadinessLocked` `readiness.go:77`, `forwardEngineNotificationFrom` `broker.go:1111`, `spawnEngineMgr` `worker_lifecycle.go:481`); `modelconfig.go`, `modelconfig_test.go`, `idlepolicy_test.go` do not exist (new, as planned); `internalClientMethods` already lists `MethodReconcileResident` and `engine:dispatch-resident-request/reply` (`client_requests.go:49-51`). Shared wire: `ModelConfiguration` with the four states `wire.go:658-705`, `ReconcileResidentParams` `:473`, `ResidentLoadOptions.Validate` still requires `KeepAliveSeconds` in `1..86400` `:184-193`, `ObserveResidentParams`/`RecoveryID` `:582-637`; Prepare/Dispatch DTOs (`:318`, `:216`) carry no `Load`. Desktop: `policySnapshot` `node-policy.ts:408` (unexported), `parsePolicyDocument` `:422`, `parsePolicySetResult` `:438`, `parsePolicyChange` `:462`; no `modelConfiguration` in `desktop/src/shared/types/node-policy.ts`. Process tests: `reserveFixtureServicePorts(t).args()` `services/tests/fixture_ports_test.go:91`, `installedServicePorts` `:26`, `llamacpp-proxy:set-port` used at `llamacpp_interop_test.go:50`, `TestServicePolicy` `service_policy_test.go:21`, `TestPolicyRelayJoint` `policy_relay_joint_test.go:36`, `TestEngineSwitchProcess*` (seven) `engine_switch_test.go:326-779`; `served_resident_test.go` absent.

Stale references: see LOW-4; plus "`notification_lifetime_test.go`" exists (OK), "`TestResident|TestStrict`" proxy selectors resolve (VERIFIED, 40 names).

| Task | Status | Notes |
|---|---|---|
| 5a-ii attested envelopes | READY+ | Gate satisfied by NE A/B/C. Edits: bind to the real signatures; add the profile == native per-slot equality the proxy still lacks (Part B enforces `q <= envelope` only, 5a-i enforces `pr <= q`); take audit-ne-sr LOW-1 (manager pre-load bound from `currentServingEnvelope`, not `ServingReadback.Configured`) and LOW-2 (owned-identity test case). |
| 5b manifest-generic adapter | READY+ after 5a-ii | r2 deferral (1) now has its contract (`servingRetentionDisabled`); deferral (2) still needs 5b to supply the revision-bound resident set against `RouterCapacity`. Files are manager-only: no I overlap. |
| 5c broker reconciler | BLOCKED on G T3 (and I T6 broker files) | UNVERIFIED-3. Everything it must reuse exists; `unloadModels`/`idleTick` still have no resident protection. |
| 6a public status consumers | BLOCKED on 5c | Also serialized on `spec.md`/`README.md`/`service-contract-exceptions.json`/`services-api.md` with I T6 and B T6. |
| 6b PC3 process proof | BLOCKED on 5b/5c; PC3 only | Harness seams named in Do exist. |

## File overlap with the remaining I tasks

I T4b/T5/T6/T7 Files are taken from `plans/context/i-unified-dispatch.md` (INFERRED for T4b's in-flight branch). "x" marks a file both tasks declare.

| Remaining task | Files shared with I T4b | with I T5 | with I T6 | with I T7 |
|---|---|---|---|---|
| B T2-T5 | none | none | none | none |
| B T6 | - | - | `services/nvpair-proxy/spec.md`, `README.md` (I T6 owns both) | - |
| J T2 | - | - | `services/shared/nodepolicy/routing_wire.go`, `routing_wire_test.go` | - |
| J T3 | `proxy.go`, `http_handler.go`, `request_spend.go`, `routing.go` (new in T4b) | `http_handler.go`, `routing.go` | broker `broker.go`, `client_requests.go`; desktop `routing.ts`, `ws-channels.ts`, service-bridge `routing.ts`, `empty-handlers.ts`, `pair-api.ts`; proxy `README.md`/`spec.md`; broker `README.md`; `service-contract-exceptions.json`; `services-api.md` | - |
| J T4, T5 | none | none | none | none |
| J T6 | - | - | - | same module (`services/tests`), different files |
| PA T1 | `services/nvpair-proxy/proxy.go` | - | `proxy.go` | - |
| PA T2 | `proxy.go` (or `routing.go` if the owner moves) | `routing.go` | `proxy.go` | - |
| PA T3 | - | - | proxy `spec.md`, `README.md` | - |
| PA T7 | none | none | none | none |
| SR 5a-ii | none | none | none | none |
| SR 5b | none | none | none | none |
| SR 5c | - | - | broker `broker.go` | - |
| SR 6a | - | - | proxy `README.md`, `spec.md`, `service-contract-exceptions.json`, `services-api.md` | - |
| SR 6b | - | - | - | same module, different file |

Cross-package overlaps outside I: G T3 shares broker `policy.go`, `policy_test.go`, `broker.go`, `proxy.go` with SR 5c and J T3; G T4 shares proxy `spec.md`/`README.md`/exceptions/`services-api.md` with B T6, PA T3, SR 6a. `docs/local-qualification.mdx` is shared by B T6 and SR 6a.

Serialisation chains that follow:

1. Proxy inference owner: I T4b -> (PA T1 `proxy.go` line + PA T2 as one patch) -> I T5 -> I T6 -> J T3. If `resolveCandidates` ends up in `routing.go`, PA T2 runs after I T5 instead.
2. Broker: I T6 -> G T3 -> SR 5c -> J T3 (broker part) -> SR 6a.
3. Shared routing wire: I T6 -> J T2 -> J T3.
4. Proxy `spec.md`/`README.md`: I T6 -> B T6 -> SR 6a -> PA T3 -> G T4 (PA's own gate names I6 and SR6a; B's names NE B and I6; SR 6a consumes B text).
5. `docs/local-qualification.mdx`: B T6 -> SR 6a.

## Proposed execution order

Machines: Mac authors and runs Mac unit/vet gates (never serves a model, never runs `services/tests`); MSI executes I (per the I plan) and supplies Windows race evidence; PC3 runs process cases serially. Columns are concurrent lanes; rows are phases that wait on the row above only where an arrow says so.

| Phase | Mac author lane | MSI lane | PC3 lane |
|---|---|---|---|
| 0 (now) | B T2 -> T3 -> T4 -> T5 (gated only on UNVERIFIED-1); in parallel SR 5a-ii -> 5b (manager/proxy files, disjoint from I and B); PA T1 part a (`discovery.go` + `availability_test.go` minus the `proxy.go` line); PA T7 draft | finish and land I T4b | record the PA T6 and NE Part B (`TestNEServingEnvelope`) runs that audit-pa-j-b UNV-1/audit-ne-sr item 8 found missing; B PC3 hermetic selection once B T5 lands |
| 1 (after I T4b accepted) | PA T1 part b + PA T2 as one small patch at the recorded owner (before I T5 dispatch, or after I T5 if the owner moved to `routing.go`); amend PA Files first | I T5 | SR 5b unskipped manager module run (`PersistPort`); unit race for B/SR selections if MSI is busy |
| 2 (after I T5) | nothing new; keep B/SR lanes finishing | I T6 (desktop bridge, shared wire, proxy/broker docs, generated contracts) | - |
| 3 (after I T6) | B T6 docs -> SR 6a docs part; J T2 against the released `routing_wire.go`; G T3 (after UNVERIFIED-3 is reconciled) | I T7 process evidence prep | J T6 and I T7 process cases need J T3 / I T6 respectively; nothing yet |
| 4 (after G T3) | SR 5c -> SR 6a code part -> J T3 (proxy/broker/desktop/TUI atomic change) -> PA T3 -> J T4 -> J T5 | race repeats for 5c/J3 | SR 6b (after 5b/5c), J T6 (after J3), I T7, PA live gate (owner-authorized) |

Parallelism gained: phase 0 runs four disjoint lanes (B, SR 5a-ii/5b, PA T1a/T7, I T4b) with no shared file. The only remaining tasks that cannot start before I T6 are those that touch the shared wire, broker or proxy documents.

## Replacement text for stale Do/Verify items

Quotes are verbatim from the plan; "new" is the proposed replacement. Context-section line references are covered by the tables above and are not repeated.

**B T2, Do 1.** Old: "External gates: owner approval of r2 replay behavior, accepted NE Task 4 Part A identity/provenance/bounds join and released shared files. Before editing, record accepted NE integration SHA, ancestry/file release and the exact coordinated schema integer in execution-time run state". New: "External gate: owner approval of r2 replay behavior. NE Task 4 Part A is integrated at `75861b7` and touched no proxy file; I T4a (`d0fb979`) and B T7 (`0468344`) released `local_qualification.go`. NE made no combined qualification revision, so the coordinated schema integer is 2 (`qualificationDocumentVersion` 1 -> 2). Re-read the borrowed-view validator at `local_qualification.go:299-399` before editing."

**B T2, Do 3.** Old: "Update all local qualification test artifact constructors/literals via `rg -n 'qualificationDocumentVersion|localQualification\{|"version"|"qualifications"' services/nvpair-proxy/local_qualification_test.go services/nvpair-proxy/testdata/local-qualification.json`." New: same command plus `docs/local-qualification.mdx` in the path list so the `"version": 1` example (`:30`) is caught for T6; `local_qualification_merged_test.go` needs no change.

**B T3, Do 1.** Old: "`d8772c8` does not recheck the allowlist after merge". New: "`8a63866` does not recheck the allowlist after merge: the raw loop is `local_qualification.go:324-345`; after `transformBody` only `uniqueKeysContext`, `isJSONObject` and `stripQualificationStore` run (`:352-365`). Add the effective-key check between `:361` and `:362`, before the store strip, so a defaults-injected unknown key fails before any strip or accounting."

**B T4, Do 1.** Old: "External gate: accepted NE Task 4 Part B is integrated, and the coordinator has serially released `readiness_contract.go` after its Part B changes. In the integrated `qualifiedModelLocked` path add one coverage predicate after the existing identity/default/bound joins". New: "NE Task 4 Part B is integrated at `8739691`; `readiness_contract.go` has no other pending owner (I T4b/T5/T6 do not list it). In `qualifiedModelLocked` (`readiness_contract.go:127-163`) add the coverage predicate after the serving-envelope guard (`:152-158`) and before `return m, q, true` (`:163`), reading `pr.Capabilities`; a profile whose capabilities lose coverage then drops out of `models` and `refreshStrictCapabilityLocked` bumps its generation (`:106-110`), which is the fence T4 Do 2 relies on."

**B T5, Do 1.** Old: "Recheck NE and I T3a/T4a/T4b file releases and current handler/helper ownership at the accepted integration base." New: "I T3a (`0d9d8cf`) and T4a (`d0fb979`) are landed. I T4b's focused-fixture discovery at `8a63866` returns `cancel_test.go`, `owned_buffers_test.go`, `performance_test.go`, `retry_test.go`, `streamframe_test.go`; the new `opencode_shapes_test.go` must reuse `strictHTTPFixture`/`recordingEngine` (`admission_http_test.go:28-38`, `:272-310`) through a llama.cpp variant (`profileFor("llamacpp")`, `engines.go:238`) rather than copying them, and must not edit those five files."

**B T6, Do 1.** Old: "External documentation gate: accepted NE Task 4 Part B and I T6 text integrated/released". New: "NE Task 4 Part B text is integrated (`docs/local-qualification.mdx` at `8739691`); I T6 remains the gate for `services/nvpair-proxy/spec.md`. Also scope `spec.md:1572` ('store' rejected) to the public-account path when adding the contract line."

**B Task 1 Status / Task 7 heading.** Old: "Task 7 can proceed after its I 4a gate". New: "Task 7 landed at `0468344` (verified by `audit-pa-j-b.md`); Task 2 may now serialize behind it."

**J T2, External prerequisites.** Old: "I6 contract join and exact wire/test release; SR5a/NE shared-wire release." New: "I6 contract join and exact release of `services/shared/nodepolicy/routing_wire.go`/`routing_wire_test.go` (I6 adds the cancel/active types there). SR 5a-i (`e6a8f73`) and NE Part A (`75861b7`) touched `policy.go`/`wire.go`, not `routing_wire.go`; no further shared-wire release is pending."

**J T3, Do 1.** Old: "Listed proxy routing.go/request_spend.go/http_handler.go/proxy.go are candidate observation sites". New: "At `8a63866` the registration site is `http_handler.go:536-541` and the key allocator `request_registry.go:26-46`; `request_spend.go:63-198` holds the attempt facts; `routing.go` does not exist until I T4b lands. Resolve the raw duplicate-key question as J T1 did: token-walk the raw params inside `routing_events.go` (precedent `workload_projection.go:124-153`); do not reserve the broker codec."

**J T3, Do 10.** Old: "Inventory tracked proxy/broker MANIFEST files with `rg --files services | rg 'MANIFEST[.]md$'`; reserve/add only a matching method-inventory path before dispatch." New: "At `8a63866` that command returns only `services/nvpair-engine-manager/MANIFEST.md`; no proxy/broker manifest path is added."

**J T5, Do.** Old: "Recount maxTabs." New: "`maxTabs = 9` is `services/nvpair-tui/ui/keys.go:119` and the shell has five views (`ui.go:69-73`); adding Routing makes six, within the bound."

**J T6, External prerequisites.** Old: "accepted join12 I3a repair; XA/I7 exact fixture transfer; separately reserved isolation/port helper repairs and fixed-path PC3 interval." New: "I3a repair and XA (`d5a636a`) are landed; `temporaryClusterPort` and every fixture listener now go through `listenFixtureLoopback`/`allocateFixtureLoopback` (`services/tests/fixture_ports_test.go:40-71`), which reject 8080, 11434, 1234 and 14318-14323. Remaining gates: J T3 landed, I T7 not editing `workload_interop_test.go`, and the fixed-path PC3 interval."

**PA T1, Do.** Old: "After the proxy reservation transfer above, add `PausedBy nodepolicy.PauseOrigin` with tag `json:"-"` beside `PolicyRevision` in `Node`." New: "Add `PausedBy nodepolicy.PauseOrigin` with tag `json:"-"` beside `PolicyRevision` at `services/nvpair-proxy/discovery.go:73` and create `availability_test.go` now; these files have no pending owner. Make the `subscribedToNode` copy (`proxy.go:1356-1361`) only after the coordinator records the accepted I T4b SHA, as part of the T2 patch."

**PA T2, Do.** Old: "Reconfirm I's final common owner/release and exact Files as above." New: "At `8a63866` the owner is `facade.resolveCandidates` (`proxy.go:631`, `add` closure `:671-735`, stale 'TCP-probed' comment `:677`). After I T4b lands, re-resolve: if the owner moved to `routing.go`, amend Files to `services/nvpair-proxy/routing.go` and schedule after I T5, which also owns it; otherwise run between I T4b and I T5."

**PA T3, Do.** Old: "External gate: I6 and SR6a documentation publication/release recorded." New: unchanged in substance, add: "Sections to edit at `8a63866`: `spec.md` '5. Request path' (`:382`), '12.2 Admission' (`:1081`), '12.4 Routing metadata' (`:1280`); `README.md` 'Node selection' (`:352`). Landing this also closes `audit-pa-j-b.md` RED-1 (broker README `services/nvpair-ui-broker/README.md:510-518`)."

**PA T7, Do.** Old: "link established TUI behavior only after verifying its current symbols." New: "the TUI renders `pausedBy` in `services/nvpair-tui/ui/availability.go` (`:31-95` per audit); link that. Root README insertion point is after item 8 Architecture (`README.md:252`); Fern index under the Reference section (`fern/docs.yml:52`)."

**SR 5a-ii, Depends on.** Old: "external prerequisite: NE T4 attested identity, proxy envelope projection, current-envelope validator and non-loading observed readback integrated, with exact signatures and file release recorded. NE T4 rebases over 501." New: "NE Task 4 Parts A (`75861b7`), B (`8739691`) and C (`8f6dd0f`) are integrated. Signatures: manager `func (e *Executor) currentServingEnvelope(ctx context.Context, engine, model string) (engines.ServingEnvelope, engineModelIdentity, error)` (`serving_identity.go:146`); observed readback lookup per `serving_readback.go:172-193`; proxy envelope guard inside `qualifiedModelLocked` (`readiness_contract.go:152-158`), which `residentContextLocked` (`residency.go:100-122`) and `residentBindingCurrentLocked` (`:249-253`) already call."

**SR 5a-ii, Do.** Old: "At proxy prepare/final dispatch require profile context equal to current native per-slot allocation". New: add "The proxy currently enforces `pr.ContextTokens <= q.ContextTokens` (5a-i, `residency.go:115-116`) and `q.ContextTokens <= envelope.PerSlotContextTokens` (Part B); add the equality `pr.ContextTokens == envelope.PerSlotContextTokens` in `residentContextLocked` using `servingEnvelopeLocked` (`readiness_contract.go:165-184`). In the manager, replace the `ServingReadback.Configured` read (`residency.go:340-344`) with `currentServingEnvelope` so the bound is launch-validated, and add the owned-identity test case from `audit-ne-sr.md` LOW-2."

**SR 5b, Do (retention/capacity).** Old (r2): "no disabled-retention contract exists (serving_llamacpp.go rejects 0/negative sleep-idle-seconds, defaults 300); NE/5b must define it at the compiler owner". New: "NE Part A defined it: `sleep-idle-seconds = -1` is accepted through preset/global/CLI/env and `servingRetentionDisabled(e)` is `e.Retention == "-1"` (`serving_identity.go:56`; tests `serving_llamacpp_test.go:43-88`). 5b requires `servingRetentionDisabled` on the resident's envelope and `envelope.RouterCapacity >= len(resident set for that engine)`, where 5b supplies the revision-bound set. The load binding is `manifests/llamacpp.json:92` -> `load_model` `POST /models/load` `{model}` (`:115`); `loaded_models` is `GET /models` with `status.value in [loaded]` (`:103-107`); replace the hard-coded `/api/generate` and `/api/ps` checks at `residency.go:228`, `:350`."

**SR 5c, Depends on.** Old: "the governing broker order/file releases in Context satisfied." New: "I3a/3b/8 are landed; I T6 (`broker.go`, `client_requests.go`) and G T3 (broker `policy.go`, `policy_test.go`, `broker.go`, `proxy.go`, new `egresslease.go`) are not. 5c dispatches after G T3 is accepted; the coordinator must first replace G T3's 'SR 5 broker relay integrated' prerequisite with the I-amendment order (UNVERIFIED-3)."

**SR 5c, Do (internal methods).** Old: "Existing internalClientMethods already blocks public reconcile/dispatch methods; add regression assertions in modelconfig_test without editing that production allowlist." New: unchanged; the entries are `nodepolicy.MethodReconcileResident`, `engine:dispatch-resident-request`, `engine:dispatch-resident-reply` at `client_requests.go:49-51`.

**SR 6a, Do.** Old: "The exact 6a/I6 document overlap is `services/nvpair-proxy/README.md`, `services/nvpair-proxy/spec.md`, `desktop/docs/service-contract-exceptions.json` and generated `desktop/docs/services-api.md`". New: add "`docs/local-qualification.mdx` is shared with B T6 and was edited by NE Part B at `8739691`; B T6 writes first, 6a rebases."

**SR Task 501 heading.** Old: "### Task 501 — 5a-i: enforce finite resident context now". New: append " — done e6a8f73" (plus `aebf553` test follow-up), per the plan's Note lifecycle.

## Owner challenges

- B's replay gate holds four code-ready tasks on one question (UNVERIFIED-1). Letting T2 land the schema with `reasoningReplay` required-false would not change behaviour and would unblock T3-T5's tools/structured work; only the replay positives would wait. The plan forbids this today; the owner can authorize the split with one sentence.
- PA T2 is a small patch sitting behind the largest in-flight I unit. If I T4b slips, consider letting PA land first on `proxy.go:631-735` with I T4b rebasing; the exclusion is one `if` before deduplication and does not touch the inference loop I T4b extracts. The plans currently say the opposite (I owner cutover -> PA); this is a scheduling choice, not a design change.

## Questions for the owner

1. Replay (UNVERIFIED-1): accept qualified, unchanged `reasoning_content` forwarding as proposed in B r2, or return it for redesign? And may B T2 land `shapes` with `reasoningReplay` required-false meanwhile?
2. I T4b: where will `resolveCandidates` live after cutover (`proxy.go` or `routing.go`)? PA T2's Files and its slot (before or after I T5) depend on it (UNVERIFIED-2).
3. G T3 prerequisite text: confirm the coordinator ledger adopts I3 -> G3 -> SR5c and drops G's SR-first clause (UNVERIFIED-3), so SR 5c has a literal dispatch condition.
4. Are PC3 run records for PA T6 and NE Part B available to attach, so phase 0's PC3 lane can start with them rather than re-running?
