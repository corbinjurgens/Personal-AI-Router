<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution plan: Phase 06: speed- and context-aware routing on profile identity, apps classification, Strata acceptance

## Goal
Overall goal of the batch (`README.md` in this folder, sections 2 to 5): PAIR matches the owner's model of their cluster. Models are known by their true identity, profiles are the unit of configuration, roles are user-defined names, the desktop is node- and role-centric, estimates are visible and validated, and requests go where they finish soonest. Phases 01 to 05 deliver identity (`contentId`), profiles, role names with fallback lists, the desktop reorganisation and estimate/validate.

This phase's goal (README section 3, themes T-B/T-F and T-G; section 5, row 06): a role request entering any node goes to the (node, profile) candidate expected to finish it soonest. The estimate uses measured speed keyed by profile and `contentId`, the request's size and current load, and routing history says why that candidate won and which role and mode were requested. Clients see the largest context each role offers as `context_length` in `/v1/models`. Concrete-model requests are context-gated like role requests, and an oversized request gets a 404 that names the context it needs and the largest available. A node that serves several roles from one memory slot prefers its loaded model by default, so it does not swap models between roles. Ollama and LM Studio appear as one "Apps" group. A Strata profile gets its engine-reported memory plan, and its user manifest is a copyable recipe in the docs. The owner's acceptance case: from the Mac, a `medium` request goes to Joes while Joes is idle and warm, and to MSI while Joes is busy. Routing history shows the selection terms, and `/v1/models` shows `context_length` for each role.

## Change scope
Intent: routing decides by expected completion time and declared context, says why, and advertises context per role. The desktop groups apps. Strata profiles price memory from the engine's own plan.
Scope: Bounded improvement — it executes four existing vault plans with amendments for the profile, `contentId` and role model of phases 01 to 03. Those plans are the smart-tier-routing Todo, the engines-and-apps Todo and engine-memory-plan Tasks 4 to 6, which Strata needs. It adds four things: a context field on role aliases, context gating for concrete requests, an explaining 404, and a `preferLoaded` default. Leave unchanged: the job scheduler and its `schedule:priority` / `node/set-priority` contract; `services/shared/schedulerwire`; strict admission and hybrid qualification beyond the identity they already read; engine lifecycle and role-driven activation (phase 04); the estimator and diagnostic run (phase 05).
Basis: README section 5 row 06 and section 3 T-B/T-F/T-G; owner decision 2026-10-31 in vault `decided.md` ("Cluster tier routing should be speed- and availability-aware"); owner statement 2026-10-29 on apps (quoted in the engines-and-apps Todo); owner expectations 5, 6, 12 in vault `Expectation Drift.md`.
Compatibility: none to preserve. This is a pre-release personal fork, and every machine runs the same build. Change formats in place, with no version bumps, capability markers, release-intent blocks or migrations. Saved node policies are not rewritten (see Task 7).
Non-goals: energy or power terms; memory-headroom terms; a displacement-cost term in the scorer (replaced by Task 7's default); feeding the launched (`/props`) context into routing; a `llama-server --help` option read (README T-G; not in row 06); removing Ollama or LM Studio support; Strata budget selection; any native Strata start on the Mac; queue-duration telemetry; live multi-machine acceptance by the executing session (owner-run, listed under Verification).

## Entry conditions
Check each before starting. Evidence is recorded in `plans/2026-10-31-expectation-gaps/00-index.md`: a ` — done SHA` suffix on the phase row and task headings, plus the phase file's close-out. The vault `WORKLOG.md` also carries it. If a condition is absent, do not implement the missing prerequisite here. Stop, and record in `00-index.md` under phase 06 which condition failed and which tasks are blocked. Tasks that do not need the missing phase may proceed, as stated per condition.

1. **Phase 01 landed (needed by Tasks 1 to 7).** Inventory carries a structured model identity with `contentId`, computed by the `servingContentDigest` formula (`services/nvpair-engine-manager/serving_identity.go` `servingContentDigest`: role, size and sha256 per file). PAIR records the engine's build identity (`/props` `build_info`) for each engine. Observable: `git grep -n contentId services/shared` finds the identity type; phase 01's close-out names where `build_info` is stored.
2. **Phase 02 landed (needed by Tasks 1 to 7 and 10 to 12).** Every PAIR-configured model is a profile with a stable id. The generated preset section name and the served model id are the profile id. `contextTokens` is derived from the profile's `load.context`. The node advertisement maps profile id to identity (including `contentId`), served name, engine, `contextTokens` and capabilities. Roles map to ordered profile lists. Typed loads take the serving-envelope identity path, so `nodepolicy.EngineIdentity.ModelDigests[profileId]` holds the serving content digest (verify-A A5, corrected detail 1). Observable: phase 02's close-out names the advertisement type and the accessor from served id to profile.
3. **Phase 03 landed (needed by Tasks 3 to 7).** Role names are strings validated by a shared grammar, and `TierOrder` is gone. Each role has a fallback list, which replaces `allowStronger`/`allowWeaker` (phase 03 Task 3). Router local tier entries are retired (phase 03 Task 4). The proxy routes any declared role name (phase 03 Task 5). Routing history stores the requested role as a string (phase 03 Task 7). A candidate's rank is its role's position in the fallback chain: `candidate.tierRank` at baseline, possibly renamed by phase 03. Observable: phase 03's close-out names the renamed symbols. Use those names wherever this plan cites a baseline `tier` symbol.
4. **Phase 04 landed (needed only by Task 7's desktop control and Task 9).** One node settings page exists, with an engines section that shows install, update, running state, errors and process settings, plus a routing section. `NodeEnginesInline`, `EngineLockControl`, `EngineSwitchControls` and the per-engine `BackendRow` tree on the card are removed. Observable: `git grep -n NodeEnginesInline desktop/src` prints nothing, and phase 04's close-out names the node settings page file and its engines-section file. Tasks 1 to 6, 8 and 10 to 12 do not wait for phase 04. Task 7's Go part and its desktop part are one task, so Task 7 waits.
5. **Phase 05 is optional.** If it landed, diagnostic-run speeds are in the performance store under the profile's `contentId`. The scorer must work from request samples alone. Do not run Task 11 while a phase 05 session is still open, because they share `services/nvpair-engine-manager/modelsettings_estimate.go` and `desktop/src/ui/components/ModelSettings/ModelMemoryEstimate.tsx`.
6. **Engine-memory-plan Todo Tasks 1 to 3 and memory-estimation-direction Tasks 1 to 8 landed (needed by Tasks 10 to 12).** Verified at baseline: `services/nvpair-engine-manager/memoryplan.go`, `memoryplan_process.go` and `memoryplan_strata.go` exist. `services/shared/modelconfig/memory.go` reads `EstimateSource` and ranged `RequiredLowBytes`/`RequiredHighBytes`. The vault status of the estimation-direction Todo (2026-10-31) says Tasks 1 to 8 are integrated. `memoryplan_strata_source.go` does not exist (memory-plan Task 4 not landed).

## Context the executor needs
Repository: the owner's PAIR fork. Base your branch on current `origin/integration/join33`, which must contain phases 01 to 03. This plan was written against `4eccd05d` (2026-10-31) plus the phase plans in this folder. Source at your base is authoritative over this section. Re-check every symbol below with `git grep` before editing, because phases 01 to 03 rename `tier` symbols to `role` symbols.

Repo rules to read first: `.cursor/rules/services-change-propagation.mdc`, `service-contracts.mdc`, `renderer-stores.mdc`, `type-discipline.mdc`, `no-legacy-fallbacks.mdc` and `engines-vs-backends-naming.mdc` (user-facing copy says "engine"; the group label "Apps" is the owner's word). A JSON-RPC change touches the Go producer, the broker relay, the desktop channel maps and bridge, the TUI when it decodes the payload, tests and docs in one change. `desktop/docs/services-api.md` is generated by `npm run service-contracts:write` and never hand-edited.

Checks available: per Go module, `go test ./...` and `go vet ./...` (the cross-process suite is in `services/tests`). In `desktop/`: `npm run lint`, `npm run typecheck`, `npm run test:unit`, `npm run dead-code:check` and `npm run service-contracts:check`. At the repo root: `make check` and `node scripts/spdx-headers.mjs`. On Windows, run Go tests only through the fixed-path wrapper (user `GOFLAGS` contains `nvpair-test-exec`), serialized, with firewall and survivor checks before and after. Never create firewall rules. Run the full engine-manager suite with `-timeout 60m`. Known baseline failures are listed in vault `bugs.md` ("integration/join33 8fa02206 engine-manager, broker and desktop test failures on Windows"). The Strata cross-process tests skip on darwin (`services/tests/strata_contract_test.go`, about line 278) and run on Windows (PC3, with MSI as the alternative).

Vault plans this phase executes. They live in `/home/user/wasabi/4. Projects/Personal-AI-Router/Todo/`. Read each one in full and follow its task text except where this plan states an amendment:
- `2026-10-31-pair-smart-tier-routing.md` (the smart-routing Todo): planned and debated (five review findings D1 to D5 accepted), not executed. Four tasks: 1 measurement identity, 2 broker feed, 3 scorer, 4 routing-history `selection`. This phase's Tasks 1 to 4 execute them.
- `2026-10-29-pair-engines-and-apps-ui.md` (the engines-and-apps Todo): planned, debated twice, not executed. Task 1 adds the classification and Task 2 the desktop grouping. This phase's Tasks 8 and 9 execute them.
- `2026-10-29-pair-engine-memory-plan.md` (the memory-plan Todo) and its "Amendment 2026-10-29" for Tasks 4 to 6: Tasks 1 to 3 landed, Tasks 4 to 6 did not. This phase's Tasks 10 to 12 execute Tasks 4 to 6.
- `2026-10-23-pair-strata-integration.md` (the Strata Todo): native acceptance on MSI stays an owner-run item (Verification, external). Nothing in this phase starts Strata natively.

Verified facts at `4eccd05d` (claim ids from `investigation/`):
- **Chooser.** `services/nvpair-proxy/tiers.go` `facade.resolveTierCandidates` sorts by `preferLoaded` (warmth first only under `"any"`), then tier rank, then warmth, then the `node/select` pin among cold preparation ranks, then preparation (L7-C20). `facade.reserveRoundPrepared` passes only the leading class to `Proxy.reserveCandidatePrepared`, which uses `Proxy.candidateChoiceLocked`: lowest pending plus GPU pressure plus reservations, ties to scheduler order (L7-C21). Equal loads fall to the lowest node ID (L7-C22). `preferLoaded` is read only in the tier comparator, and its default is `"tier"` (`services/shared/nodepolicy/policy.go` `TierPolicy`, `Default`; L7-C27). `Admission.MaxResidentModels` defaults to 1 (L7-C24). A full cap evicts idle least-recently-used models or waits (L7-C25).
- **Context.** `requestNeeds.metBy` (tiers.go) drops a candidate when its declared `contextTokens` is nonzero and below `len(body)/4` plus the first positive `max_tokens`, `max_completion_tokens` or `options.num_predict` (L7-C1, C2). Its only callers are inside `resolveTierCandidates`. The gate also applies to hybrid alias requests through `hybridLocalCandidates` (verify-B B5). Concrete requests are not gated: `routing.go` parses `needs` only inside `if tier != ""` (about line 687; L7-C3). The concrete resolver is `services/nvpair-proxy/bound_engine.go` `facade.resolveConcreteCandidates`. The no-eligible 404 lists only capabilities: `routing.go` near lines 893 and 1399 call `writeProxyError(w, http.StatusNotFound, []string{"no eligible model for tier ", tier}, needs.names(), true)`, and `writeProxyError` is in `services/nvpair-proxy/http_handler.go` (L7-C14). A concrete request with no candidate gets 502 `{"error":"no available node advertises the requested model"}` (routing.go, about line 911). Strict admission (`local_qualification.go` about line 460), strict readiness (`readiness_contract.go` about lines 140 and 154) and the broker's resident entry (`services/nvpair-ui-broker/modelconfig.go` about line 302) also compare `Profile.ContextTokens` with the launch envelope (B5). They keep working unchanged, because phase 02 derives `contextTokens` from the load.
- **Model list.** `services/nvpair-proxy/model_list.go` `facade.serveModelList` appends the alias records from `facade.tierModelEntries` on the OpenAI dialect only (L7-C16). `tiers.go` `hybridAliasRecords` marshals exactly `{id, object:"model", created:0, owned_by:"pair"}` per alias from `configuredAliases` (bare, `-api` and `-local`; L7-C15). `services/nvpair-proxy/routing_listener.go` (about line 358) calls `hybridAliasRecords(policy, nil, nil)` for the listener without facades. The test `services/nvpair-proxy/hybrid_alias_test.go` `TestHybridAliasRecordsNameOnlyTheAlias` (line 339) pins the four-key set.
- **Performance identity.** `services/nvpair-proxy/performance.go` `facade.performanceCurrentIdentity` needs a strict engine identity and a model digest from `admissionController.modelDigestLocked` (`readiness_contract.go` line 200). Without a digest no sample is produced (A5). `facade.performanceFingerprintOwned` hashes `Run` (`EngineRunID`), `Settings`, `Launch`, `Model` (the digest), `API`, `Schema` and `Options`. `st.runID = newEngineRunID()` runs on every launch, so every restart starts a new `ProfileKey` (B8). `performancewire.Profile` carries `engineFamily`, `quantization` and `contextTokens`, and the proxy fills only `engineFamily` (B8). `LoadNs` comes only from Ollama (B8). The broker store (`services/nvpair-ui-broker/performancestore/store.go`) keeps at most `performancewire.MaxSamplesPerKey` (64) samples per key. Each sample expires after `performancewire.RetentionMs` (seven days). The desktop decodes `performancewire.Profile` with an explicit key list (`desktop/src/electron/service-bridge/performance.ts`, about line 123).
- **Engine identity.** `nodepolicy.EngineIdentity` (`services/shared/nodepolicy/wire.go`) has `engineRunId`, `settingsFingerprint`, `launchDigest`, `modelDigests` and `serving`. It has no build field at baseline. The engine manager builds it in `services/nvpair-engine-manager/model_identity.go` `Executor.modelIdentity`. The broker assembles it in `services/nvpair-ui-broker/model_identity.go` (about line 203).
- **Routing history.** `RoutingHistoryRequested{Model, Tier, Mode}` is in `services/shared/nodepolicy/routing_wire.go` (about line 324). The desktop type in `desktop/src/shared/types/routing.ts` is `requested: {model, tier, mode}`. `desktop/src/ui/components/RoutingSettings/RoutingHistory.tsx` renders only `requested.model` (L7-C32).
- **Policy text in the desktop.** The desktop edits node policy as JSON text (`desktop/src/shared/types/node-policy.ts` `NodePolicySnapshot`) and saves through `policy:set` with `expectedPersistedRevision`. Lexeme-preserving edit helpers (`objectFields`, `replaceField`, `policyDraft`) are module-private in `desktop/src/shared/utils/node-policy-model-settings.ts`. No desktop control for `tierPolicy` exists (grep: no `preferLoaded` in `desktop/src`).
- **Apps.** No `kind` field and no `AppsRow` exist (L8-C39). Only `ollama.json` and `lmstudio.json` are apps (owner, engines-and-apps Todo).
- **Strata.** Strata is a dynamic descriptor engine and is not bundled. Its template is `services/nvpair-engine-manager/testdata/strata/strata.json`, and `docs/strata.mdx` "Procedure" (about line 225) tells the user to copy that file (L8-C7, C14). The memory-plan code has no production caller, and `Estimate.MemoryPlan` is never assigned (L8-C17). Native acceptance is open (L8-C18).

## Decisions (settled — do not revisit)
- Role routing chooses by expected completion, using measured speed, whether the model is loaded, current usage and reported availability. Owner decision 2026-10-31 (vault `decided.md`).
- The proxy owns routing, the broker owns evidence, and nothing in the desktop decides. The desktop only displays and edits stored values.
- No scheduler contract change: `schedulerwire.Priority`, `schedule:priority` and `node/set-priority` stay as they are.
- These decisions from the smart-routing Todo's plan-time section are settled for this phase:
  - The broker feeds the proxy through one private, process-scoped method, `node/set-routing-costs`, with a contract-exception entry. Not `schedulerwire.Priority`, because that couples speed to reservation generations.
  - Warmth, preparation and rank remain hard classes, and the scorer ranks only inside the leading class.
  - No energy term now.
  - Score weights are all 1, in one table in `tier_costs.go`.
  - With no evidence, the chooser is the unchanged `candidateChoiceLocked`.
  - Unknown rates are imputed as the slowest measured rate in the class.
- Owner decisions D1 (user-defined roles with fallback lists), D2 (profiles in node policy) and D6 (any llama.cpp build; `build_info` recorded) from README section 6 are in force.
- Apps: the label is "Apps". The classification is service-owned: manifests declare it, descriptors carry it and the desktop renders it, with no desktop list of app names. Existing Ollama and LM Studio controls stay inside the Apps item (QC1, owner 2026-10-29).
- Strata: MSI is the first native target. Preserve native API authentication, and never forward a key. No Mac native engine. Memory-plan owner answers O1 to O7 are settled (memory-plan Todo).
- No compatibility work (fork rule "Pre-release fork: no compatibility work").

## Decisions made while writing this plan
Edit or delete a line to veto it.
- **The smart-routing Todo's four open choices keep its defaults (listed for veto):**
  1. A warm slow node beats a cold fast node, because warmth is a hard class.
  2. The score's output budget is 512 tokens when a request names no limit.
  3. Every 16th committed choice in a class probes an unmeasured idle candidate.
  4. The cold cost is 120,000 ms when no load time is measured.
- **Amendment to smart-routing Task 1 (identity).** Its name-based measurement identity is dropped. After phases 01 and 02, every profile-served model has `ModelDigests[profileId]` equal to its serving content digest, which is `contentId`. That digest is the performance sample's model digest. What remains of Task 1: confirm that wiring with tests, carry `contentId` on samples, and apply the fingerprint amendment below. Models with no digest (LM Studio, any model not served as a profile) produce no samples and route by the baseline chooser. Not a name-based path, because `contentId` makes it unnecessary (README T-A "Replaces").
- **Fingerprint amendment (verify-B B8).** When the sample's served model is a phase 02 profile and the engine's `build_info` is known, the fingerprint hashes `Profile` (profile id) and `Build` (`build_info`) in place of `Run`. Otherwise it keeps today's sequence. Justification: a restart of the same profile on the same build with the same settings, launch, `contentId` and options serves at the same speed. Hashing the run id splits that evidence into a new key on every restart, so the newest key starts empty after each restart and old keys crowd the 64-key cap. Retention must still do four things:
  - Expire each sample after `RetentionMs`.
  - Keep at most `MaxSamplesPerKey` samples per key, newest kept.
  - Start a new key on any build, settings, launch, `contentId` or options change, so evidence from a different build never mixes.
  - Drop a sample whose engine run changed during the attempt (the run id stays in the identity tuple and in the end-of-attempt recheck; it is only removed from the hash).
- **Evidence key amendment (smart-routing Task 2).** The key (host, engine, model) becomes (host, profile id). The profile id is the served model id, `ProfileKey.Model`. For app models such as Ollama, the served model id stands in, so the same rule covers them. Each feed entry carries the `contentId` it was measured under, so the scorer uses only evidence measured on the content the candidate serves now, and `selection` shows whether two nodes serve the same file. To carry it, `performancewire.Profile` gains `contentId`. This changes the replicated summary format in place, and the desktop decoder accepts the field. Not proxy-only lookup from the advertisement, because the advertisement does not say which content the evidence was measured on.
- **Rank and role amendment (smart-routing Tasks 3 and 4).** The scorer's rank is the position in the requested role's fallback chain (phase 03). `selection` carries the role name string of the winning class and the rank, plus profile id and `contentId` per candidate, in place of `tierRank` and model.
- **Context advertisement.** `context_length` on a role alias is the largest declared `contextTokens` among the profiles every visible node declares for that role. It does not include fallback roles, and it is not the launched `/props` value. It is present only when that maximum is positive. The bare role alias and its `-local` variant carry it; the `-api` variant does not. Rationale: the declared value is what routing gates on, and the owner's wording ("configured for 128k") names the configured value (L7 open question). A fallback role answers only when the role itself has no eligible candidate, so the role's own maximum is what a client can rely on.
- **Concrete gating is context-only.** A concrete request names its model, so capabilities are not gated, as today. The bound is the candidate node's advertised `contextTokens` for the profile it serves. A model with no advertised bound (apps, unprofiled models) has bound 0, which means unbounded, as for roles today.
- **The 404 text.** When a request's candidates were all dropped and at least one was dropped only for context, the 404 error string appends `: request needs about N tokens of context; the largest available is M`. M is the largest bound among the context-dropped candidates. A concrete request in that state gets this 404 in place of today's 502, because waiting cannot fix it.
- **Residency across roles on one node (README T-G).** The scorer's displacement cost (the smart-routing Todo's deferred item) is not built.
  - `TierPolicy.PreferLoaded` gains the value `"auto"`, and `nodepolicy.Default()` sets it. Under `"auto"`, the router behaves as `"any"` when at least one candidate for the request is on a node that declares two or more role names, and as `"tier"` otherwise. `preferLoaded` is the ingress router's policy, so "nodes that declare more than one role" is evaluated over the request's candidate nodes, not over the ingress node.
  - Trade-off: on such a node, a loaded model from a later role in the fallback chain answers before a cold model of the requested role, which avoids an eviction and a cold load at the cost of answering from another role. A role with no fallback entries never borrows.
  - The smart-routing Todo's "no new `preferLoaded` value" is overridden for this need.
  - Saved policies that already hold `"tier"` keep it (no migration). The owner switches them in node settings. The acceptance list names this step.
- **Where the Apps group renders.** In the phase 04 node settings page's engines section and in `desktop/src/ui/components/Welcome/WelcomeEnginesStep.tsx`. The engines-and-apps Todo's other surfaces are gone after phase 04 (`NodeEnginesInline`, `NodeEngineSettings`, `EngineSwitchControls`) or are left unchanged by that Todo, so they need no grouping. Lock controls are gone (D4), so the Todo's lock assertions are dropped.
- **Strata recipe.** `docs/strata.mdx` embeds the full user-manifest template as a JSON block, so a user needs no checkout. A test keeps the block equal to `testdata/strata/strata.json`.
- **Memory-plan ordering amendment.** The memory-plan Todo's amendment orders its Task 5 after estimation-direction Task 8 and before estimation-direction Task 9. If phase 05 already landed estimation-direction Task 9 or 12, Task 11 here builds its source dispatch on the estimate input phase 05 left. It also keeps the `engine-report` domain-amount rule of the amendment. The ordering existed to avoid churn in shared files, not for a semantic dependency (inference from the amendment text).
- **Packaging.** Twelve tasks. The proxy tasks form one chain because they share `tiers.go`, `routing.go` and `services/nvpair-proxy/spec.md`. Task 8 follows Task 4 only because both regenerate `desktop/docs/services-api.md`. Task 9 follows Task 7 because both edit phase 04 node settings files, whose layout is not known here.

## Tasks

### Task 1 — Performance samples key on the profile's contentId and survive engine restarts
**Files:** `services/nvpair-proxy/performance.go`, `services/nvpair-proxy/performance_test.go`, `services/shared/performancewire/wire.go`, `services/shared/performancewire/wire_test.go`, `services/shared/nodepolicy/wire.go`, `services/shared/nodepolicy/wire_test.go`, `services/nvpair-engine-manager/model_identity.go`, `services/nvpair-engine-manager/model_identity_test.go`, `services/nvpair-ui-broker/model_identity.go`, `services/nvpair-ui-broker/model_identity_test.go`, `services/tests/performance_observations_test.go`, `services/nvpair-proxy/testdata/performance/README.md`, `services/nvpair-proxy/spec.md`, `docs/performance-telemetry.mdx`, `desktop/src/shared/types/performance.ts`, `desktop/src/electron/service-bridge/performance.ts`, `desktop/tests/modular/performance.test.ts`
**Depends on:** none — needs entry conditions 1 and 2
**Complexity:** judgment
**Do:** This executes smart-routing Task 1 with the identity and fingerprint amendments in "Decisions made while writing this plan". Do not build its name-based identity, identity-kind field or `"Kind"` hash.
1. Wiring check. Write a test in `performance_test.go`: an engine identity whose `ModelDigests` maps a profile id to a 64-hex digest, and a llama.cpp facade replaying `testdata/performance/llamacpp-chat-stream-usage.txt`. `performanceCurrentIdentity` must return `tuple.model` equal to that digest, and exactly one `performance:sample` must result. Then confirm on the real chain: in `services/tests/performance_observations_test.go`, add a case where a typed-load (profile) fake llama.cpp engine produces a broker-stored entry after one completed streamed request. If at your base the digest in `ModelDigests` for a profile-served model is not the `contentId` that phase 01 publishes for that profile, stop and report both values. Do not add a translation or a second identity source.
2. Build identity. Run `git grep -n "BuildInfo" services/shared/nodepolicy/wire.go`. If it prints a field on `EngineIdentity`, use it. If it prints nothing, take these steps:
   - Add `BuildInfo string` with JSON name `buildInfo,omitempty` to `nodepolicy.EngineIdentity`.
   - In `EngineIdentity.Validate`, accept it when it is empty or at most 128 printable bytes.
   - Fill it in `services/nvpair-engine-manager/model_identity.go` `Executor.modelIdentity` from the build identity phase 01 records (its close-out names the symbol).
   - Carry it in `services/nvpair-ui-broker/model_identity.go` where the `nodepolicy.EngineIdentity{...}` literal is built.
   - Add round-trip tests in the three modules' named test files.
   If the field exists, leave those six files untouched.
3. Fingerprint. Add `profile` and `build` to `performanceIdentityTuple`.
   - Set `profile` to the normalized served model id when the current policy has a phase 02 profile with that id for this engine (use the accessor phase 02's close-out names). Set `build` from `EngineIdentity.BuildInfo`.
   - When both are non-empty, `performanceFingerprintOwned` hashes the sequence `Profile, Build, Settings, Launch, Model, API, Schema`, then `Options`, written with the same `performanceHashString` framing. Otherwise it keeps today's `Run, Settings, Launch, Model, API, Schema` sequence exactly.
   - Keep `run` in the tuple and keep the end-of-attempt recheck comparing the whole tuple.
4. Tests in `performance_test.go`:
   - A digest-backed identity with no profile produces the same fingerprint as at the base commit. Before editing, run the base code once to get the expected hex, and pin it.
   - With profile and build known, two different run ids give the same fingerprint. A different build, settings fingerprint, launch digest, digest or option set each gives a different fingerprint.
   - A run id change during an attempt still drops the sample.
5. Content id on samples. Add `ContentID string` with JSON name `contentId,omitempty` to `performancewire.Profile`. Validate it as empty or 64 lowercase hex in `Profile.Validate`. The proxy fills it from `tuple.model` where it builds `performancewire.Profile{EngineFamily: ...}` (`performance.go`, about line 930). Add the key `contentId` to the desktop decoder's allowed list in `desktop/src/electron/service-bridge/performance.ts`, and to the type in `desktop/src/shared/types/performance.ts`. Add a decoder test in `desktop/tests/modular/performance.test.ts`, and a wire test in `wire_test.go`.
6. Never write `ModelDigests`, `a.strict.identity` or readiness state, and never make up a digest.
7. Docs:
   - In `docs/performance-telemetry.mdx` and the performance section of `services/nvpair-proxy/spec.md`: samples are keyed by host and served profile id, and the fingerprint covers profile, build, settings, launch, `contentId`, API, schema and options (no run id when profile and build are known). Models without a content digest are not measured.
   - In `testdata/performance/README.md`: the fixtures prove extraction, not installed-runtime qualification.
**Verify:**
- In `services/nvpair-proxy`, `go test -count=1 -run 'Performance' ./...` exits 0. In `services/shared`, `go test -count=1 ./performancewire ./nodepolicy` exits 0.
- In `services/tests`, `go test -count=1 -run 'TestPerformanceObservation' .` exits 0.
- If step 2 added the field, `go test -count=1 -run 'ModelIdentity' .` exits 0 in `services/nvpair-engine-manager` and in `services/nvpair-ui-broker`.
- From `desktop/`, `npx vitest run --project unit tests/modular/performance.test.ts tests/modular/node-performance.test.tsx` passes and `npm run typecheck` exits 0.
- The full `go test -count=1 ./...` in `services/nvpair-proxy` shows no failure outside the `bugs.md` baseline.

### Task 2 — The broker feeds per-profile speed evidence to each proxy
**Files:** `services/shared/performancewire/routing.go` (new), `services/shared/performancewire/routing_test.go` (new), `services/nvpair-ui-broker/performance.go`, `services/nvpair-ui-broker/performance_test.go`, `services/nvpair-ui-broker/broker.go`, `services/nvpair-ui-broker/client_requests.go`, `services/nvpair-ui-broker/proxyaddressing_test.go`, `services/nvpair-proxy/tier_costs.go` (new), `services/nvpair-proxy/tier_costs_test.go` (new), `services/nvpair-proxy/proxy.go`, `services/nvpair-proxy/spec.md`, `services/nvpair-proxy/README.md`, `services/nvpair-ui-broker/README.md`, `docs/performance-telemetry.mdx`, `desktop/docs/service-contract-exceptions.json`, `desktop/docs/services-parity.md`, `desktop/docs/services-api.md` (regenerated only)
**Depends on:** 1 — consumes `performancewire.Profile.ContentID`; shares `spec.md` and `docs/performance-telemetry.mdx`
**Complexity:** judgment
**Do:** Execute smart-routing Task 2 steps 1 to 7 as written, with these amendments:
1. In `services/shared/performancewire/routing.go`, `RoutingProfile` has these fields in place of `Engine` and `Model`:
   - `Profile string` with JSON name `profile`: the served model id, which is the profile id for PAIR-configured models.
   - `ContentID string` with JSON name `contentId,omitempty`: from the entry's `performancewire.Profile.ContentID`.
   `Host`, `Fingerprint`, `AgeMs`, `Prefill`, `Decode`, `LoadP95Ms`, `LoadSamples` and `Extra` stay as the Todo specifies. Uniqueness is on (host, profile). `Validate` checks `ContentID` with the same rule as `Profile.Validate`.
2. Broker selection in `performanceRuntime.routingCosts`: for each (`ProfileKey.Host`, `ProfileKey.Model`), apply the Todo's usable-phase rule, then newest-entry rule, then fingerprint tie-break. Never pool fingerprints.
3. Proxy cache lookup in `tier_costs.go`: `routingEvidence(host, profile, contentID string) (performancewire.RoutingProfile, int64, bool)`.
   - It keys by the candidate's node ID and `p.normalizeEngineModel(candidate.engine, candidate.model)` (`services/nvpair-proxy/policy.go`), as the Todo's D4 says.
   - When `contentID` is non-empty and the entry's `ContentID` differs, it returns false.
   - Keep the Todo's test that this key equals the `ProfileKey` the proxy's own sample producer writes, including a candidate from a different engine than the ingress facade.
4. The contract-exception entry text for `"node/set-routing-costs"` in `desktop/docs/service-contract-exceptions.json` is: "Broker-internal. nvpair-ui-broker sends each local proxy the newest per-(host, profile) speed evidence, with the content id it was measured on, from its performance projection for role scoring; PAIR consumes the routing result, not this feed."
5. Do not touch the `priority*` maps, reservations, `schedulerwire` or the scheduler.
**Verify:**
- `go test -count=1 ./...` exits 0 in `services/shared/performancewire`, `services/nvpair-ui-broker` and `services/nvpair-proxy`, apart from failures in the `bugs.md` baseline.
- From `desktop/`, `npm run service-contracts:check` exits 0.
- `git diff --stat -- services/nvpair-job-scheduler services/shared/schedulerwire` prints nothing.

### Task 3 — Role choice minimises expected completion time inside the leading class
**Files:** `services/nvpair-proxy/tiers.go`, `services/nvpair-proxy/tier_costs.go`, `services/nvpair-proxy/tier_costs_test.go`, `services/nvpair-proxy/proxy.go`, `services/nvpair-proxy/routing.go`, `services/nvpair-proxy/tiers_test.go`, `services/nvpair-proxy/reservation_test.go`, `services/shared/nodepolicy/policy.go`, `services/nvpair-proxy/spec.md`, `services/nvpair-proxy/README.md`, `docs/hybrid-routing.mdx`
**Depends on:** 2 — reads the evidence cache; shares `tier_costs.go`, `proxy.go`, `README.md` and `spec.md`
**Complexity:** judgment
**Do:** Execute smart-routing Task 3 steps 1 to 6 as written, with these amendments:
1. "Tier round" means a role round: the `reserveRoundPrepared` callers in `routing.go` where a role was requested (phase 03 Task 5). The rank in the class key is the candidate's position in the requested role's fallback chain, which phase 03 assigns to `candidate.tierRank` or its renamed field.
2. Evidence: call `routingEvidence(nodeID, profile, contentID)`, where `contentID` is the `contentId` the candidate's node advertises for that profile (the phase 02 advertisement). Pass an empty string when none is advertised.
3. Acceptance tests use the role `medium`, the default name kept by phase 03, and one profile id served on both `msi` and `joes` with the same `contentId`. Keep every number in the Todo's acceptance, busy, request-size, baseline, cold and probe cases.
4. Replace the Todo's "Tier order" case with a fallback-order case. Give the role a fallback list with a second role. A faster candidate of the second role never wins while an eligible candidate of the requested role exists, and the existing phase 03 fallback tests still pass.
5. The four plan-time values (warm slow beats cold fast, 512, 16, 120000) are those in "Decisions made while writing this plan". Keep them as named constants in `tier_costs.go`, each with a one-line comment that names this plan's decision section.
6. Docs: update the `TierPolicy.PreferLoaded` comment in `services/shared/nodepolicy/policy.go`, `services/nvpair-proxy/spec.md` (role ordering section), `README.md` and `docs/hybrid-routing.mdx`. They state that within the leading class the candidate with the lowest expected completion time wins, that warmth, preparation and fallback rank keep their precedence, and that a warm slow node is not compared with a cold fast one.
**Verify:**
- From `services/nvpair-proxy`, `go test -count=1 -run 'TierCost|Tier|Role|Reserv|ChooseReachable|Failover' ./...` exits 0.
- `go test -count=1 ./...` in `services/nvpair-proxy` and in `services/tests` shows no failure outside the `bugs.md` baseline.
- `go vet ./...` in `services/nvpair-proxy` exits 0.

### Task 4 — Routing history shows the requested role, the mode and why the destination won
**Files:** `services/shared/nodepolicy/routing_wire.go`, `services/shared/nodepolicy/routing_wire_test.go`, `services/shared/nodepolicy/routing_history_window_test.go`, `services/shared/nodepolicy/testdata/routing-request.json`, `services/nvpair-proxy/routing.go`, `services/nvpair-proxy/routing_facts.go`, `services/nvpair-proxy/routing_facts_test.go`, `services/nvpair-proxy/routing_events.go`, `services/nvpair-proxy/routing_events_test.go`, `services/nvpair-ui-broker/routing_events_test.go`, `services/nvpair-service/history_test.go`, `services/nvpair-tui/ui/routingwire_test.go`, `services/tests/service_routing_listener_test.go`, `desktop/src/shared/types/routing.ts`, `desktop/src/electron/service-bridge/routing-json.ts`, `desktop/src/ui/components/RoutingSettings/RoutingHistory.tsx`, `desktop/tests/modular/routing-projection.test.ts`, `desktop/tests/modular/routing-delivery.test.ts`, `desktop/tests/modular/routing-view.test.ts`, `desktop/docs/frontend-api.md`, `services/nvpair-proxy/spec.md`, `services/nvpair-ui-broker/README.md`, `desktop/docs/services-parity.md`, `desktop/docs/services-api.md` (regenerated only)
**Depends on:** 3 — records the `tierDecision` it returns; shares `routing.go` and `spec.md`; Task 2 shares `services-parity.md`, `services-api.md` and the broker `README.md`
**Complexity:** judgment
**Do:** Execute smart-routing Task 4 steps 1 to 5 as written, with these amendments:
1. The selection types, in place of the Todo's `TierRank` field and candidate `Model` field:
   ```go
   type RoutingHistorySelection struct {
       Decision    string                       `json:"decision"` // measured | imputed | probe | pinned | baseline
       Role        string                       `json:"role"`     // role name of the winning class (fallback chain entry)
       Rank        int64                        `json:"rank"`     // its position in the requested role's fallback chain
       Warm        bool                         `json:"warm"`
       Preparation bool                         `json:"preparation"`
       Winner      RoutingHistorySelectionCand  `json:"winner"`
       RunnerUp    *RoutingHistorySelectionCand `json:"runnerUp"`
   }
   type RoutingHistorySelectionCand struct {
       NodeID        string                        `json:"nodeId"`
       Profile       string                        `json:"profile"`   // served model id (profile id)
       Engine        string                        `json:"engine"`
       ContentID     string                        `json:"contentId"` // "" when the node advertises none
       TotalMs       *int64                        `json:"totalMs"`
       Terms         []RoutingHistorySelectionTerm `json:"terms"`
       EvidenceAgeMs *int64                        `json:"evidenceAgeMs"`
       Samples       int64                         `json:"samples"`
       Imputed       bool                          `json:"imputed"`
   }
   ```
   `Role` must pass phase 03's role-name grammar. `ContentID` is empty or 64 lowercase hex. The 2048-byte selection cap, the runner-up omission rule and the key-set test stay as the Todo specifies.
2. `RoutingHistory.tsx`, table row: the requested-model cell becomes a stack of two lines. The first line is the existing model text. The second line is `Role: ROLE · Mode: MODE`, where ROLE is the requested role string from the record (`requested.tier` at baseline, renamed by phase 03 Task 7) and MODE is `requested.mode`. When the role is empty, the second line is `Mode: MODE`. Use the existing `Stack` and text components.
3. `RoutingHistory.tsx`, expanded row: render the Todo's selection content. Show the decision, the role and rank, then the winner and runner-up, each with node, profile, the first 12 characters of `contentId` (the text "no content id" when it is empty), terms in milliseconds, evidence age and sample count. Use the existing formatters.
4. `desktop/tests/modular/routing-view.test.ts` asserts:
   - A record with role `medium` and mode `chat` renders `Role: medium · Mode: chat`.
   - A record with an empty role renders `Mode: chat`.
   - A record with a selection renders its decision and the winner's profile.
**Verify:**
- `go test -count=1 ./...` exits 0 in `services/shared` (nodepolicy), `services/nvpair-proxy`, `services/nvpair-ui-broker`, `services/nvpair-service`, `services/nvpair-tui` and `services/tests`, apart from failures in the `bugs.md` baseline.
- From `desktop/`, `npm run lint`, `npm run typecheck`, `npm run dead-code:check` and `npm run service-contracts:check` exit 0, and `npm run test:unit` shows no failure beyond the `bugs.md` baseline.
- `node scripts/spdx-headers.mjs` reports no missing header.

### Task 5 — Role aliases in /v1/models advertise context_length
**Files:** `services/nvpair-proxy/tiers.go`, `services/nvpair-proxy/routing_listener.go`, `services/nvpair-proxy/hybrid_alias_test.go`, `services/nvpair-proxy/tiers_test.go`, `services/nvpair-proxy/spec.md`, `services/nvpair-proxy/README.md`, `docs/hybrid-routing.mdx`
**Depends on:** 4 — shares `spec.md` and `README.md`; `tiers.go`, `tiers_test.go` and `docs/hybrid-routing.mdx` change in Task 3
**Complexity:** judgment
**Do:**
1. In `tiers.go`, add `func (d declaredRoles) maxContext(role string) int`. It returns the largest positive `contextTokens` among the profiles that `d.self` and every `d.peers` entry declare for `role`, counting only profiles whose served model id passes `nodepolicy.ValidateModelIdentifier`. It returns 0 when none is positive. Use the phase 02 shape of `AdvertisedRole`: a role maps to profiles, each with `contextTokens`. Use the per-role lookup phase 03 left in place of `declaredRoles.forTier`.
2. Change the alias-record builder (`hybridAliasRecords` at baseline) to take a `contextLength func(role string) int`. For the bare role alias and its `-local` variant, add `"context_length": N` to the marshalled map when N > 0. The `-api` variant never carries it. Keep `id`, `object`, `created` and `owned_by` unchanged.
   - `facade.tierModelEntries` passes `f.declaredRoles(pol).maxContext`.
   - `services/nvpair-proxy/routing_listener.go`, at the call that passes `nil` overlays, passes a function computed from `policy.Advertisement().Roles` only (this node's own declarations).
3. Rename `TestHybridAliasRecordsNameOnlyTheAlias` to `TestHybridAliasRecordsNameTheAliasAndContext` and change its assertion deliberately:
   - A role whose profiles declare 16384 and 32768 on two nodes yields `context_length` 32768 on the bare and `-local` records, with key set `{context_length, created, id, object, owned_by}`.
   - The `-api` record and a role with no positive bound keep the key set `{created, id, object, owned_by}`.
   - Keep the skip assertion.
   Add a `tiers_test.go` case that lists `/v1/models` through the facade fixture and decodes `context_length` for each configured role.
4. Docs: `services/nvpair-proxy/spec.md` (model list section), `README.md` and `docs/hybrid-routing.mdx` state the field. It is the largest declared context among the role's profiles across visible nodes, it is absent when unknown, and it is never on `-api` aliases.
**Verify:** From `services/nvpair-proxy`, `go test -count=1 -run 'HybridAlias|TierModel|ModelList|RoutingListener' ./...` exits 0, and the output of `go test -count=1 -run 'TestHybridAliasRecordsNameTheAliasAndContext' -v .` contains `--- PASS`. `git grep -n "TestHybridAliasRecordsNameOnlyTheAlias" services` prints nothing. The full `go test -count=1 ./...` in `services/nvpair-proxy` shows no failure outside the `bugs.md` baseline.

### Task 6 — Concrete-model requests are context-gated and an oversized request gets a 404 that names the context
**Files:** `services/nvpair-proxy/routing.go`, `services/nvpair-proxy/tiers.go`, `services/nvpair-proxy/bound_engine.go`, `services/nvpair-proxy/tiers_test.go`, `services/nvpair-proxy/routing_test.go`, `services/nvpair-proxy/spec.md`, `services/nvpair-proxy/README.md`
**Depends on:** 5 — shares `tiers.go`, `tiers_test.go`, `spec.md` and `README.md`; `routing.go` changes in Tasks 3 and 4
**Complexity:** judgment
**Do:**
1. Shortfall record. In `tiers.go`, add `type contextShortfall struct{ needed, largest int }`. `resolveTierCandidates` (or its phase 03 name) returns it beside the candidates. A candidate that has every needed capability but fails only the context bound sets `largest = max(largest, bound)`, and `needed` is `needs.contextTokens`. Update its callers, including `hybridLocalCandidates`.
2. Parse needs for concrete requests. In `routing.go`, where `needs` is parsed only for a tier (about line 687), also parse when `tier == ""`, `model != ""` and `isInferenceRequest(f.profile, r.Method, r.URL.Path)`. Use `parseRequestNeedsOwned` with the same error response. `crossEngine` stays tier-only.
3. Gate concrete candidates. In the `resolve` closure's concrete branch, filter the result of `f.resolveConcreteCandidates(routingModel, r.URL.Path)` and record a `contextShortfall`.
   - Get each candidate's bound from a new `facade.advertisedContext(c candidate) int`. It returns the `contextTokens` that the candidate's node advertises for the profile whose served id normalizes to `c.model`: this node's policy advertisement when `c.self`, the discovery node's advertisement otherwise. It returns 0 when there is none.
   - Keep a candidate when `requestNeeds{contextTokens: needs.contextTokens}.metBy(nil, bound)` is true.
   - Put `advertisedContext` in `bound_engine.go` next to `resolveConcreteCandidates`.
4. 404 text. When the candidate list is empty and the recorded shortfall has `largest > 0`, add these parts after the existing prefix parts: `": request needs about "`, `strconv.Itoa(needed)`, `" tokens of context; the largest available is "`, `strconv.Itoa(largest)`.
   - Apply this at both role sites (`routing.go` near lines 893 and 1399), and keep `needs.names()` and `includeNeeds` there.
   - For a concrete request, answer the same way with status 404, prefix parts `"no eligible model for "` and the model, and `needs.names()` with `includeNeeds` true. This replaces the 502 path only when the shortfall is set. Record the ingress fact with `factNoOwner` and 404, as the role path does.
   - With no shortfall, every response stays as today.
5. Tests:
   - Role: a body of 120000 bytes (30000 tokens) against role profiles declaring 16384 on one node and 32768 on another goes to the 32768 node.
   - Role: against only 16384, the response is 404 with error `no eligible model for tier medium: request needs about 30000 tokens of context; the largest available is 16384`. Adapt the prefix to phase 03's wording if it changed.
   - Concrete: the same body to a concrete profile id served only with 16384 gets that 404 with prefix `no eligible model for` and the model id.
   - Concrete: a concrete model with no advertised bound is still routed.
   - Concrete: a request without a body over the bound is unchanged.
   - Name the new tests `TestConcreteRequestContextGate` and `TestOversizedRequestNamesContext`.
6. Docs: `services/nvpair-proxy/spec.md` and `README.md` state that concrete requests are context-gated by the serving node's advertised profile context (0 is unbounded), and give the 404 text.
**Verify:** From `services/nvpair-proxy`, `go test -count=1 -run 'TestConcreteRequestContextGate|TestOversizedRequestNamesContext' -v .` shows `--- PASS` for both. `go test -count=1 -run 'Tier|Role|Concrete|Binding|Hybrid' ./...` exits 0. The full suite shows no failure outside the `bugs.md` baseline, and `go vet ./...` exits 0.

### Task 7 — preferLoaded auto keeps a one-slot node with several roles warm
**Files:** `services/shared/nodepolicy/policy.go`, `services/shared/nodepolicy/policy_test.go`, `services/nvpair-proxy/tiers.go`, `services/nvpair-proxy/tiers_test.go`, `services/nvpair-proxy/engine_binding_candidates_test.go`, `services/nvpair-proxy/spec.md`, `services/nvpair-proxy/README.md`, `services/nvpair-service/README.md`, `docs/model-engine-bindings.mdx`, `docs/hybrid-routing.mdx`, `docs/architecture.mdx`, `desktop/src/shared/utils/node-policy-model-settings.ts`, `desktop/tests/modular/node-policy-prefer-loaded.test.ts` (new), `desktop/src/ui/components/RoutingPreferenceControl.tsx` (new), `desktop/tests/modular/routing-preference-control.test.tsx` (new), and the phase 04 node settings page component (its path is in phase 04's close-out in `00-index.md`)
**Depends on:** 6 — shares `tiers.go`, `tiers_test.go`, `spec.md` and `README.md`; `policy.go` and `docs/hybrid-routing.mdx` change in Tasks 3 and 5; also needs entry condition 4 (phase 04)
**Complexity:** judgment
**Do:**
1. `services/shared/nodepolicy/policy.go`:
   - `TierPolicy.PreferLoaded` accepts `"auto"`, `"tier"` and `"any"`. `Default()` sets `"auto"`. The `Validate` error becomes ``tierPolicy.preferLoaded must be "auto", "tier" or "any"``.
   - Rewrite the field comment: "auto" behaves as "any" when a candidate for the request is on a node that declares two or more roles, and as "tier" otherwise.
   - Tests in `policy_test.go` cover the default value, all three accepted values and the rejection of `"sometimes"`.
2. `services/nvpair-proxy/tiers.go`, in the role resolver after `out` is built: `anyLoaded` is true for `"any"` and false for `"tier"`. For `"auto"`, it is true when some `c` in `out` has `len(declared.of(c)) >= 2`, which counts role names. Change nothing else in the comparator.
3. Tests in `tiers_test.go`:
   - `"auto"` with a candidate node declaring `weak` and `medium`, where `medium` is warm and the `weak` fallback chain is `[weak, medium]`: a `weak` request goes to the warm `medium` profile.
   - The same policy where every candidate node declares one role: the cold `weak` candidate wins.
   - Explicit `"tier"` and `"any"` behave as before.
   - Update `engine_binding_candidates_test.go` only where a case fails because `Default()` now says `"auto"`. In that case set its intended value explicitly.
4. Desktop helper. In `desktop/src/shared/utils/node-policy-model-settings.ts`, export `readPreferLoaded(text: string)` and `applyPreferLoaded(text: string, value: 'auto' | 'tier' | 'any')`. Both use the module's lexeme-preserving `objectFields`/`replaceField` approach, and both return the module's existing `Result` type. `applyPreferLoaded` replaces or inserts only `tierPolicy.preferLoaded` and leaves every other byte of the text unchanged. If phase 02 or 04 moved these helpers, put the functions beside them (`git grep -n "function replaceField" desktop/src`). Unit tests in `desktop/tests/modular/node-policy-prefer-loaded.test.ts` cover these cases:
   - read the value;
   - insert into a policy without `tierPolicy`;
   - replace an existing value;
   - a large integer elsewhere in the text survives byte for byte.
5. Desktop control. `RoutingPreferenceControl.tsx` renders a select labelled "Prefer loaded models" with three options:
   - "Automatic" (`auto`)
   - "Role order first" (`tier`)
   - "Loaded model first" (`any`)

   Below the select, render this help text exactly: "Automatic prefers a model that is already loaded when a node serves more than one role, so one memory slot is not swapped between roles. A loaded model can then answer for a later role in the fallback list."

   Mount it in the routing section of the phase 04 node settings page. It saves through the existing `policy:set` path with `expectedPersistedRevision`, as the page's other policy edits do. It never computes the effective value. Its test covers rendering the current value, saving each option, and the refusal path on a revision conflict.
6. Docs: `services/nvpair-proxy/spec.md`, `README.md`, `docs/model-engine-bindings.mdx`, `docs/hybrid-routing.mdx` and `docs/architecture.mdx` (the `preferLoaded` sentence) describe `"auto"` and the trade-off. `services/nvpair-service/README.md` lists the three accepted values. State that saved policies holding `"tier"` keep it until changed.
**Verify:**
- In `services/shared`, `go test -count=1 ./nodepolicy` exits 0. In `services/nvpair-proxy`, `go test -count=1 -run 'PreferLoaded|Tier|Role|EngineBinding' ./...` exits 0.
- In `services/tests`, `go test -count=1 -run 'Policy' .` exits 0.
- From `desktop/`, `npx vitest run --project unit tests/modular/node-policy-prefer-loaded.test.ts tests/modular/routing-preference-control.test.tsx` passes, and `npm run typecheck`, `npm run lint` and `npm run dead-code:check` exit 0.

### Task 8 — Engine descriptors carry an optional engine-or-app classification
**Files:** `services/shared/engines/descriptor.go`, `services/shared/engines/descriptor_test.go`, `services/nvpair-engine-manager/registry.go`, `services/nvpair-engine-manager/registry_test.go`, `services/nvpair-engine-manager/descriptor.go`, `services/nvpair-engine-manager/descriptor_test.go`, `services/nvpair-engine-manager/manifest.schema.json`, `services/nvpair-engine-manager/manifests/ollama.json`, `services/nvpair-engine-manager/manifests/lmstudio.json`, `services/nvpair-engine-manager/MANIFEST.md`, `services/nvpair-engine-manager/README.md`, `services/nvpair-engine-manager/spec.md`, `services/nvpair-tui/ui/engineswire.go`, `services/nvpair-tui/ui/engineswire_test.go`, `desktop/src/shared/types/engines.ts`, `desktop/src/electron/service-bridge/engine-descriptors.ts`, `desktop/tests/modular/engine-descriptors.test.ts`, `docs/strata.mdx`, `desktop/docs/services-api.md` (regenerated only)
**Depends on:** 4 — both regenerate `desktop/docs/services-api.md`
**Complexity:** mechanical
**Do:** Execute engines-and-apps Task 1 Do steps 1 to 5 exactly as written: optional `kind` set to `engine` or `app` with default `engine`; `Descriptor.Kind` with `IsApp()`; `"kind": "app"` only in `ollama.json` and `lmstudio.json`; decoders accept the field; docs; `npm run service-contracts:write`. No amendment.
**Verify:** Run the engines-and-apps Task 1 Verify commands on the Mac: the shared engines package, the manager selector, `services/nvpair-tui` `./ui`, the broker selector, the proxy selector and the desktop checks. Run on PC3 (on MSI when PC3 is unavailable): `go test -count=1 -run '^TestDescriptorEngineIsFrontedFromAUserManifest$' .` from `services/tests`. Pass conditions are as the Todo states:
- the `engine:descriptors` reply reports `"kind":"app"` for Ollama and LM Studio and omits `kind` for llama.cpp and Strata;
- `"kind":"other"` is rejected;
- existing selectors pass.

### Task 9 — Apps render as one group in node settings and in the Welcome step
**Files:** `desktop/src/ui/components/AppsRow.tsx` (new), `desktop/tests/modular/apps-row.test.tsx` (new), `desktop/src/ui/components/Welcome/WelcomeEnginesStep.tsx`, `desktop/tests/modular/lmstudio-credential.test.ts`, `docs/getting-started.mdx`, the phase 04 node settings engines-section component and its test (paths in phase 04's close-out in `00-index.md`)
**Depends on:** 7, 8 — uses `kind` from Task 8; Task 7 also edits phase 04 node settings files; also needs entry condition 4 (phase 04)
**Complexity:** judgment
**Do:** Execute engines-and-apps Task 2 with these amendments for phase 04's surfaces:
1. Grouping surfaces are only the phase 04 node settings engines section and `WelcomeEnginesStep.tsx`.
   - In the engines section, `AppsRow` renders the section's existing per-engine row for the selected app. That row shows install, update, running state, errors and process settings, plus LM Studio's credential control where phase 04 placed it.
   - In Welcome, `AppsRow` renders the existing `WelcomeEngineRow`, and keeps the Todo's persistent list of apps selected for install.
2. Do not touch `NodeEnginesInline`, `NodeEngineSettings` or `EngineSwitchControls`. Phase 04 removed them. If any still exists at your base, stop: entry condition 4 failed.
3. Drop the Todo's lock-control assertions (locks are removed under D4). Keep every other assertion in its Verify:
   - one llama.cpp row and one Apps row;
   - choosing an app sends no lifecycle, install, settings or switch call;
   - the selected app's actions carry its real engine id;
   - LM Studio's credential control works inside the group;
   - the Welcome selections, defaults and Install targets;
   - one app shows directly without a chooser;
   - a remote engine without a descriptor renders individually;
   - a node with no apps shows no Apps row.
4. `AppsRow` takes app descriptors and a render function from its host, and contains no app names.
**Verify:**
- From `desktop/`, `npx vitest run --project unit tests/modular/apps-row.test.tsx tests/modular/lmstudio-credential.test.ts` passes, together with phase 04's engines-section test file.
- `npm run typecheck`, `npm run lint` and `npm run dead-code:check` exit 0.
- `git grep -n -i "ollama\|lmstudio\|lm studio" desktop/src/ui/components/AppsRow.tsx` prints nothing.

### Task 10 — A Strata profile's planning input is prepared credential-free and the manifest recipe is in the docs
**Files:** `services/nvpair-engine-manager/memoryplan_strata_source.go` (new), `services/nvpair-engine-manager/memoryplan_strata_source_test.go` (new), `services/nvpair-engine-manager/serving_strata.go`, `services/nvpair-engine-manager/serving_strata_test.go`, `services/nvpair-engine-manager/strata_auth_test.go`, `services/nvpair-engine-manager/testdata/strata/strata.json`, `services/nvpair-engine-manager/strata_manifest_test.go`, `docs/strata.mdx`
**Depends on:** 8 — shares `docs/strata.mdx`
**Complexity:** judgment
**Do:**
1. Execute memory-plan Task 4 as written; the Todo's amendment leaves Task 4 unchanged.
   - Build `memoryPlanInput` (`memoryplan.go`) and `strataMemoryPlanBinding` (`memoryplan_strata.go`) from the credential-free derivation stage of `serving_strata.go`.
   - Bind the selected module, native, manifest and model closure.
   - Add the platform-scoped `memory_plan` invocation, with absolute placeholders, to `testdata/strata/strata.json`.
   - Never read the API key for pricing.
   The input is per profile: after phase 02 a Strata model is a profile whose engine is the Strata descriptor engine. Take the model and settings from that profile's typed load, the same source the launch path uses.
2. Recipe. In `docs/strata.mdx` "Procedure", replace the instruction to copy the testdata file with a self-contained recipe:
   - where the per-user `engines/` directory is on Windows, macOS and Linux;
   - the file name `strata.json`;
   - the full template as one fenced `json` block, byte-identical to `testdata/strata/strata.json`;
   - which `/REPLACE_WITH_ABSOLUTE_PATH/` values to replace;
   - how to choose a free `runtime.port`.
   State that `kind` is optional and Strata is an engine. Keep the existing sections on the API key, unsupported actions and finite prepared serving.
3. In `strata_manifest_test.go`, add `TestStrataRecipeMatchesTemplate`. It reads `../../docs/strata.mdx`, extracts the first fenced `json` block under "Procedure", and requires it to equal `testdata/strata/strata.json` byte for byte.
**Verify:** On the Mac, from `services/nvpair-engine-manager`, `go test -count=1 -run 'TestMemoryPlanStrataSource|TestStrataServing|TestStrataManifest|TestStrataRecipeMatchesTemplate' .` exits 0. Run the same selector with `-race`, and it exits 0. Pass conditions are those of memory-plan Task 4's Verify; in particular, a key-read hook that fails on every call is never reached by pricing. No real Python or native binary runs.

### Task 11 — A Strata profile's estimate carries its engine-reported memory plan
**Files:** `services/nvpair-engine-manager/modelsettings_estimate.go`, `services/nvpair-engine-manager/modelsettings_estimate_test.go`, `services/nvpair-engine-manager/modelsettings_rpc.go`, `services/nvpair-engine-manager/modelsettings_contract_test.go`, `services/nvpair-engine-manager/modelsettings_artifacts.go`, `services/nvpair-engine-manager/modelsettings_artifacts_test.go`, `services/nvpair-engine-manager/modelsettings_load.go`, `services/nvpair-engine-manager/modelsettings_load_test.go`, `services/nvpair-engine-manager/modelsettings_auto_test.go`, `services/nvpair-engine-manager/model_buffer_capture_test.go`, `services/nvpair-engine-manager/README.md`, `services/nvpair-engine-manager/spec.md`, `services/shared/modelconfig/memory.go`, `services/shared/modelconfig/memory_test.go`, `desktop/src/ui/components/ModelSettings/ModelMemoryEstimate.tsx`, `desktop/tests/modular/model-memory-estimate.test.tsx`, `desktop/tests/modular/model-settings-auto.test.tsx`
**Depends on:** 10 — uses the prepared input; Task 8 edits the engine-manager `README.md` and `spec.md` first; also needs entry condition 5 (no open phase 05 session)
**Complexity:** judgment
**Do:**
1. Execute memory-plan Task 5 with its 2026-10-29 amendment:
   - select the provider from the target's selected manifest and platform declaration;
   - assign `Estimate.MemoryPlan` and `EstimateSource` (`engine-report`) for a declared report;
   - a failed report returns Unknown without calling the PAIR estimator;
   - fill each device and host-RAM domain's `requiredBytes` with `max(pricedTotalBytes, minimumSelectionCapacityBytes)`, never their sum;
   - put unpriced costs in `EstimateSource.UnpricedCosts`;
   - Auto selection stays as it is.
   Change `services/shared/modelconfig/memory.go` and `memory_test.go` only if a report-specific slot helper is needed. Otherwise leave them untouched.
2. Ordering amendment (see decisions). If phase 05 changed the estimate input or the `ModelMemoryEstimate.tsx` rendering, dispatch on what phase 05 left. Keep its ranges, warnings and "Run anyway" behaviour intact. Add only the "Source: Engine report" and "Source: PAIR estimate" line and the unpriced-costs warning from memory-plan Task 5, unless phase 05 already renders a source line. In that case use phase 05's line.
3. The estimate is requested per profile, as phase 02 made inspection profile-keyed. A Strata profile's inspection reaches the fake planning CLI through `InspectModel`.
**Verify:** On the Mac, from `services/nvpair-engine-manager`, `go test -count=1 -run 'TestModel(Estimate|Auto|Settings|Inspection|Buffer)|TestMemoryPlan' .` exits 0. Run it again with `-race`, and it exits 0. `go vet ./...` exits 0. From `desktop/`, `npx vitest run --project unit tests/modular/model-memory-estimate.test.tsx tests/modular/model-settings-auto.test.tsx` passes, and `npm run typecheck` exits 0. Pass conditions are those of memory-plan Task 5's Verify plus its amendment:
- an `engine-report` estimate with known capacity yields a verdict from its domain amounts, with its unpriced costs as warnings;
- a report whose threshold exceeds its priced total uses the threshold.

### Task 12 — Target-local Strata planning is proven end to end with fake processes
**Files:** `services/tests/model_memory_plan_test.go` (new), `services/tests/model_settings_fixture_test.go`, `services/tests/model_settings_test.go`, `services/nvpair-engine-manager/memoryplan_process_windows_test.go`, `services/nvpair-engine-manager/MANIFEST.md`, `docs/model-configuration.mdx`, `docs/strata.mdx`, `desktop/docs/services-api.md` (regenerated only)
**Depends on:** 11 — proves its dispatch; `docs/strata.mdx`, `MANIFEST.md` and `services-api.md` change earlier in Tasks 8 and 10
**Complexity:** judgment
**Do:** Execute memory-plan Task 6 with its amendment: the process test also asserts that the domain amounts and `EstimateSource` survive the real broker-to-manager relay, and that the verdict carries the named unpriced costs. Use a Strata profile, not a bare model id, for the target-local request. Windows execution follows the Todo:
- run on MSI if it is free, otherwise PC3;
- use the fixed-path wrapper;
- check firewall rules and survivors before and after;
- a missing interval or rule leaves the run pending and is recorded as pending, never bypassed.
Finish the docs and run `npm run service-contracts:write`.
**Verify:** Windows fixed-path environment:
- From `services/nvpair-engine-manager`, `go test -count=1 -timeout=30m -run 'TestMemoryPlan|TestModel(Estimate|Auto|Settings|Inspection|Buffer)' .` exits 0. Run the same command with `-race`, and it exits 0.
- From `services/tests`, `go test -count=1 -timeout=30m -run '^TestModelMemoryPlanProcess$' .` exits 0. Run it with `-race`, and it exits 0.
- `go vet ./...` exits 0 in both modules.

Mac:
- From `desktop/`, `npm run service-contracts:check` exits 0.
- `node scripts/spdx-headers.mjs` reports no missing header, and `git diff --check` prints nothing.

If the Windows interval is unavailable, record the run as pending in `00-index.md`. Task 12 is then not done.

## Verification (whole-plan)
Local (the executing session, on its base and final head; compare each failure with a run at the base commit in a separate scratch worktree before calling it pre-existing):
- Go, `go test -count=1 ./...` and `go vet ./...`, each exiting 0 apart from failures listed in the `bugs.md` baseline:
  - `services/nvpair-proxy`;
  - `services/nvpair-ui-broker`;
  - `services/nvpair-job-scheduler` (unchanged; proves no regression);
  - `services/shared`, which covers `nodepolicy`, `performancewire`, `engines` and `modelconfig`;
  - `services/nvpair-tui` and `services/nvpair-service`;
  - `services/nvpair-engine-manager`, with `-timeout 60m` on Windows.
- `services/tests`: `go test -count=1 -run 'TestUnifiedDispatchProcess|TestStrata|TestPerformanceObservation|TestRoutingHistoryProductionChain|TestServiceRoutingListenerWithoutFacades|TestDescriptorEngineIsFrontedFromAUserManifest|TestModelMemoryPlanProcess' .` exits 0. The `TestStrata*` contract tests skip on darwin, so run this selector on Windows through the fixed-path wrapper, on PC3 (on MSI when PC3 is unavailable). A skip is untested, not a pass.
- `git diff --name-only BASE..HEAD -- services/nvpair-job-scheduler services/shared/schedulerwire` prints nothing (BASE is your recorded starting commit).
- From `desktop/`, `npm run lint`, `npm run typecheck`, `npm run dead-code:check` and `npm run service-contracts:check` exit 0, and `npm run test:unit` has no failure beyond the baseline. At the repo root, `node scripts/spdx-headers.mjs` reports nothing missing and `make check` exits 0.
- Keep the commands, exit codes and the last 50 lines of any failing output in the phase 06 close-out.

External (owner-run on the owner's machines; the executing session does not run these):
1. Install the build on the Mac, MSI and Joes.
2. In node settings on MSI and Joes, set "Prefer loaded models" to "Automatic" if the saved value is "Role order first".
3. Send one pinned `medium` request to each of MSI and Joes to seed evidence.
4. From the Mac, send an unpinned `medium` request while Joes is idle and its `medium` profile is warm. `X-PAIR-Node` names Joes, and the routing-history row shows `Role: medium · Mode: chat` with a `selection` naming Joes, its terms, evidence age and samples.
5. Make Joes busy (three or more concurrent long requests) and send another `medium` request from the Mac. It goes to MSI, and the `selection` shows MSI winning on a lower total.
6. `curl` the Mac's `/v1/models`. Each role alias (`weak`, `medium`, `strong` and any user-defined role with profiles) carries `context_length` equal to the largest context among that role's profiles across nodes.
7. A request whose body exceeds every profile's context for its role gets the 404 naming the needed and largest context.
8. The Apps group appears in node settings and in the Welcome step on the Mac.
9. Strata native acceptance on MSI, per the Strata Todo's "Verification (whole-plan)":
   - an owner-approved profile and a released interval;
   - owned start with authentication;
   - cold versus loaded inventory, bounded load, one short chat request, cancellation, unload and verified stop and restore;
   - one admitted peer route from the Mac;
   - the profile's estimate shows "Source: Engine report".
   This item keeps the Strata Todo open until it is recorded.

## Close-out
When every task's Verify passes (Task 12's Windows run included), do these steps:
- In `plans/2026-10-31-expectation-gaps/00-index.md`, append ` — done SHA` to the phase 06 row, with SHA the last commit of the phase. Append ` — done SHA` to each task heading in this file, with that task's commit, as the generated lifecycle section below describes. Record any pending item, such as a Windows run without an interval, beside the row instead of a done mark.
- Write a vault `WORKLOG.md` entry with:
  - the commit range;
  - the gate commands and their exit codes;
  - remaining limitations: warm slow beats cold fast; context is declared, not launched; saved `"tier"` policies keep their value; models without a content digest are not measured; Strata native acceptance is open;
  - the hand-over: phase 06 is the last phase; the index's whole-plan acceptance (items 3 and 6 of its list depend on this phase) is owner-run.
- In the vault, record the executed tasks of the smart-routing Todo (1 to 4), the engines-and-apps Todo (1 and 2) and the memory-plan Todo (4 to 6) through the project-notes lifecycle (`project-note todo archive` with the inspected hash, commits and verify evidence as notes). Keep the Strata Todo open until the external item 9 is recorded.
- Commit each task with explicit pathspecs. Follow the commit convention the earlier phases of this batch used (check `git log` for phase 01 commits): sign-off trailer or none, and message style. Never stage `.claude/`.

## Do not
- Do not refactor beyond what the tasks specify.
- Do not change the job scheduler, `schedulerwire`, `schedule:priority`, `node/set-priority`, strict admission or hybrid qualification.
- Do not write to `ModelDigests`, manufacture a digest, or add a name-based measurement identity.
- Do not add an energy term, a displacement-cost term or a policy field other than the `"auto"` value of `preferLoaded`.
- Do not feed `/props` launched context into routing or the model list in this phase.
- Do not decide routing, effective preferences or app membership in the desktop.
- Do not hard-code app names in the desktop or classify apps by family or vendor. Do not remove Ollama or LM Studio controls.
- Do not start Strata, Python or any native engine on the Mac, read real API keys, or fetch models.
- Do not log or store prompts, bodies, request options, PINs or key material.
- Do not hand-edit `desktop/docs/services-api.md`, `CHANGELOG.md` or `services/versions.json`. Add no version bumps, release-intent blocks or migrations.
- Do not touch installed PAIR, firewall, trust or network settings on any machine.

## If stuck
- A flaky test listed in `bugs.md` may be re-run up to twice; record each run.
- If the key from Task 2's `routingEvidence` does not equal the producer's `ProfileKey` (host or served-id naming differs), stop and report both values. Do not add a translation table.
- If the profile digest is not the published `contentId` (Task 1 step 1), stop and report. Phase 01 or 02 must reconcile it.
- If the selection or desktop loan bound in Task 4 needs more than 34 KiB, stop and report the measured size (smart-routing Todo rule).
- If phase 03 left no per-role lookup for Task 5, or no rank field for Task 3, stop and report the missing interface. Do not reintroduce `TierOrder`.
- If a phase 04 surface named in Tasks 7 or 9 does not exist, stop those tasks and describe the needed layout instead of redesigning the page.
- If a Windows package needs a firewall rule that does not exist, leave that run pending and continue. Never create rules.
- If `npm run service-contracts:check` reports a mismatch that `npm run service-contracts:write` does not fix, fix the producing README or exception entry, never the generated file.

## Note lifecycle

This file is working state in a repository, not documentation.

- When an item lands, append ` — done <sha>` to its heading, where
  `<sha>` is the commit that landed it. An item consciously dropped gets
  ` — dropped` and a `project-note deferred` entry.
- When every item carries one of those marks, delete this file in the
  commit that lands the last item. Nothing in it is kept: the commits and
  the run archive are the record, and `project-note done` records the
  completion.
- In an execute run the orchestrator does this after each unit lands,
  not the unit. A plain session does it as each item lands.
