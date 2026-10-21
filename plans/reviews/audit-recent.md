<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Delivery audit: recent units (B 2-5, SR 5a-ii/5b, NE readback transport, manual node references)

Verdict: **RED 0 UNVERIFIED 3 LOW 7**

| Unit | Verdict |
| --- | --- |
| B Tasks 2-4 (merges `96f1290`, `59ad85a`) | DELIVERED. Every requirement sentence FULFILLED; the reasoning-replay sentences are superseded by the required-false ruling and are implemented as refusals. |
| B Task 5 (merge `8ac96ad`, in `integration/join25`) | DELIVERED. Fixture, refusal-before-effects, forwarding and SSE mechanics present and passing; the plan's replay-positive sentences are superseded by the ruling (refusal tests instead). |
| SR 5a-ii (merge `d9f2276`) | DELIVERED. Proxy and manager join NE's validator and observed allocation with equality; replacement revokes holds; post-load mismatch withdraws eligibility. |
| SR 5b (merge `40046f8`) | DELIVERED, one ruling clause UNVERIFIED: `declaredResidentCount` is required and bounded 1..256, but its binding to the policy revision is only enforceable by the unlanded 5c broker reconciler (UNV-2). |
| NE readback transport classification (`8b2707e` + `02f8ad6`) | DELIVERED. Every transport class named by the ruling leaves the sweep UNKNOWN without a drift latch; only semantic mismatches latch; the resident observation caller inherits the classification. |
| Manual node reference validation (`2a18bd3`) | DELIVERED at the consumer (bounded string kind rejecting control and Cf). Residual producer gap from UNV-3 remains (UNV-1). |

## Scope, inspection base and method

- Brief: `plans/handoffs/inv-audit-recent.md` and `plans/handoffs/inv-common.md` on `plan/cloud-investigations`; plans `plans/b-opencode-shapes.md` (`plan/b-opencode-shapes`), `plans/context/sr5-rescope.md`, the prior report `plans/reviews/audit-pa-j-b.md` (UNV-3), and the OpenCode inventories on `plan/b-opencode-inventory-msi` / `plan/b-opencode-inventory-pc3`.
- Code inspected at `integration/join25` = `02f8ad60851cf95a448724c0b92ba277753f6974` (checked out detached; all module tests below ran at this SHA). All `path:line` citations below are at `02f8ad6` unless marked otherwise.
- `fork/groundwork` was `7e270f10ced402325dc9e1efece54991e9bbbc08` at the start of this run (as the common brief states). It moved during the run: the remote-tracking ref now resolves to `15695aed7bce23bec01089c6f6def68e7890d074`, which is `02f8ad6` (all of `integration/join25`, fast-forwarded) plus one merge `15695ae` ("switch-fixture-baseline") touching only `services/tests/engine_switch_test.go` and `services/tests/remote_management_test.go` (+12 lines). `git diff --stat 02f8ad6 15695ae -- services/nvpair-engine-manager services/nvpair-proxy services/nvpair-ui-broker services/shared` is empty, so the citations at `02f8ad6` hold for the audited modules at the new head. This report is saved on a branch created from `15695ae`.
- Unit merges mapped from `git log --first-parent`: `96f1290` (B Tasks 2-3), `59ad85a` (B Task 4), `d9f2276` (SR 5a-ii), `40046f8` (SR 5b), `8ac96ad` (B Task 5), `8b2707e` (`roadmap/ne-readback-transport`), `02f8ad6` (readback follow-up in the resident caller), `2a18bd3` (`roadmap/j-manual-id`).
- VERIFIED means I read the code or ran the check; INFERRED is marked explicitly. No engine, model, product binary or `services/tests` case was started; `services/tests` received `go vet ./...` only.

## Module unit tests run (at `02f8ad6`, Linux amd64, go1.24.7, sandboxed container)

| Module | Command | Result |
| --- | --- | --- |
| `services/shared` | `go vet ./...`; `go test -count=1 ./...` | vet 0. All packages ok except one first-run failure in `clustertrust` (`TestReverseDialPrivateEndpointPreservesSaturatedUnlockedLiveOwner`: unix-socket i/o timeout after 120 queued connections); the package passed on an immediate re-run. `nodepolicy`: 142 top-level tests PASS, 0 FAIL, 0 SKIP (2159 `=== RUN` lines including subtests). |
| `services/nvpair-proxy` | `go vet ./...`; `go test -count=1 -v ./...` with `PAIR_CORS_PARITY_URL`, `PAIR_CORS_OLLAMA_EXECUTABLE` and all proxy env vars unset (the B plan's sanitized shape) | vet 0. 710 top-level PASS, **4 FAIL**, 6 SKIP (3194 RUN lines). Failures: `TestPublicTransportBounds`, `TestPublicTransportErrorBodyBounds`, `TestPublicTransportFinalGuard` (all `public DNS refused`: `api_transport.go:115` refuses when the sandbox resolver returns nothing; these are I/G public-transport tests, outside the audited units) and `TestHandleHTTP_RealSocketFlushDeadline` (`zombie_test.go:516`), which fails identically when run alone at the pre-join25 base `7e270f1`, so it is not a join25 regression. Skips: `TestLoopbackAliasOwnsBothLocalhostFamilies`, `TestDualAliasBindIsAtomic`, `TestCORSBrowserFixture/{ollama,lmstudio}`, `TestCORSExternalEngineParity`, `TestRoutingListener_ReportsAnotherHolderAsACollision/IPv6_loopback`. Audited prefixes: `TestQualification*`/`TestStrict*`/`TestOpenCode*` 68 top-level PASS, 0 FAIL/SKIP; `TestResident*`/`TestServedResident*` 49 top-level PASS, 0 FAIL/SKIP. |
| `services/nvpair-engine-manager` | `go vet ./...`; `go test -count=1 -v -skip PersistPort ./...` (the SR plan's named Mac exclusion) | vet 0. 758 top-level PASS, **19 FAIL**, 6 SKIP (2711 RUN lines). Failures: 17 `TestStrata*` (every one: "Strata/Python environment overrides are unsupported in prepared mode"; the container exports a `PYTHON*` variable and the compiler fails closed on it; Strata is owner-deferred and outside this audit) and `TestStopDeclinesLiteralDeletedSuffixListener`, `TestStopReclaimsManagedLiteralDeletedSuffix` (procfs listener inspection returns nothing in this sandbox). Skips are Windows-only or layout tests. Audited prefixes: `TestResident*` 37 PASS/0 FAIL, `TestLoadedServing*` 17 PASS/0 FAIL, `TestAction*` 10 PASS, `TestLlamaServing*` 46 PASS/0 FAIL, `TestDesired*`+`TestResidentPreservesDesiredStateAndFailedWrites`+`TestResidentFailedDesiredReadDoesNotRepairState` 8 PASS. `PersistPort` was skipped as the plan requires on a non-Windows host and is untested here. |
| `services/nvpair-ui-broker` | `go vet ./...`; `go test -count=1 -v -skip EngineManagerPortsSpawn ./...` | vet 0. 666 top-level PASS, 0 FAIL, 0 SKIP (1515 RUN lines). `TestWorkloadManualDestinationReachesConsumers`, `TestWorkloadManualDestinationRejectsBeforeEffects`, `TestWorkloadStrictIdentitiesRemainStrict`, `TestWorkloadProjectionDiagnostics` PASS. `EngineManagerPortsSpawn` untested here. |
| `services/tests` | `go vet ./...` only | exit 0. No case listed or run. |

No Windows (PC3/MSI) result exists in this run; the Windows gates named by the plans stay pending (UNV-3).

## Unit 1: B Tasks 2-4 (`96f1290`, `59ad85a`) and Task 5 (`8ac96ad`)

Ruling applied: `reasoningReplay` stays required-false. Plan sentences that presuppose the approved r2 qualify-and-forward policy are marked SUPERSEDED where the code implements the refusal instead.

### Task 2: one NE-compatible explicit shape evidence schema

| Requirement (plan) | Status | Evidence |
| --- | --- | --- |
| "Add `shapes` with ... explicit required non-null boolean tools; explicit required non-null array structuredOutput, unique allowed names only; required non-null reasoningReplay boolean." | FULFILLED | `local_qualification.go:89-93` (pointer/nil-distinguishing members), `:95-120` validate; tests `local_qualification_test.go:815` `TestQualificationShapesDocument` (PASS). |
| "Reject absent object, null object/members, partial declaration, wrong types, duplicates, unknown shape fields, unknown/duplicate mode names and unsupported document versions." | FULFILLED | nil object `:96-98`; nil members `:99-101,:111-113`; unknown modes `:103-105`; duplicate modes `:106-108`; unknown fields via `dec.DisallowUnknownFields()` `:212`; duplicate keys via `uniqueKeys` `:219-221`; version `:222-224` (`qualificationDocumentVersion = 2`, `:43`). `TestQualificationShapesDocument` (`local_qualification_test.go:815`) rejects version 1 ("old evidence") and version 3. |
| "Validate required members in declared order, not map iteration." | FULFILLED | `:99-116` checks tools, structuredOutput, reasoningReplay in that order. |
| "No effort/parallel fields, no old-version compatibility path." | FULFILLED | struct `:89-93` has three members; `:222-224` accepts exactly version 2. |
| "Shared `testQualification` declares tools true, structuredOutput empty and reasoningReplay false; fixtures exercising JSON modes/replay explicitly declare their evidence." | FULFILLED (INFERRED for the helper body) | `testdata/local-qualification.json` is version 2 with `shapes` (`tools:true`, both modes, `reasoningReplay:false`); `opencode-shapes.json` declares per-case shapes. |
| "`reasoningReplay` stays required-false" (ruling) | FULFILLED | `:114-117` rejects `reasoningReplay:true` with "must be false while reasoning replay is unsupported". |

### Task 3: validate and account each retained shape on the effective request

| Requirement | Status | Evidence |
| --- | --- | --- |
| "Validate every effective top-level key against `chatRequestKeys` after defaults merge ... before scalar/accounting/admission work." | FULFILLED | raw check `:363`, merge `:366`, effective re-check `:388`, store strip `:391`, then scalars/accounting `:398-425`. `TestQualificationShapeMatrix/unrelated_unknown_*` and `TestQualificationShapeDefaultsAndBounds/default_shapes/unrelated_unknown_*` PASS (defaults-injected unknown key refused). |
| "Preserve Task 7's explicit store:false-only normalization on raw and merged bodies ... never a forwarded key." | FULFILLED | `checkQualificationKeys` `:432-450` treats `store` specially; `checkQualificationStore` `:452-460` (true refused with the storage message, non-false refused); `stripQualificationStore` `:464+` removes the effective false after merge. 30+ `store_request_and_default_controls/*` subtests PASS. |
| "Reject effort/parallel presence whatever the value." | FULFILLED | not in `chatRequestKeys` `:332-335`; `reasoning_effort_{null,false,"high",{}}` and `parallel_tool_calls_*` subtests PASS on raw and default paths. |
| "Gate nonempty tool definitions, requiring tool_choice, assistant tool_calls and tool-role history on tools evidence. Present tools must be a non-null array; accept `[]` unchanged with zero definition overhead and no tools-evidence requirement of its own." | FULFILLED | null tools `:796`; definitions gate `:818` (only when `definitions > 0`); `[]` adds nothing to `total`; history gate `:719-720`; choice gate `:842-845` (`required`/named need evidence and nonempty definitions; `none`/`auto` do not). Fixture `empty-tools` (plain qualification, accepted, 18 tokens) and `required-without-definitions` PASS. |
| "Qualified history needs no definitions in that request, including native `tools:[]`." | FULFILLED | fixture `tool-roundtrip-no-active-tools` (tools qualification, `tools:[]`, accepted) and `tool-history-definitions-omitted`; matrix `native_history_with_empty_definitions/{qualified,undeclared}`. |
| "Preserve named function checks, optional parameters/object-when-present behavior, choice accounting, role restrictions and null-content-with-nonempty-tool-calls rule. Reject null/wrong-type gated members." | FULFILLED | `:803-813` (named function, `parameters` object only when present), `:728` null content rule, `:716` calls only on assistant; matrix `null_tools`, `null_choice`, `null_calls`, `assistant_null_without_calls` etc. PASS. |
| "Gate each `response_format` JSON mode on its exact structuredOutput entry; text and omission stay accepted ... An evidence entry for json_object does not authorize json_schema and vice versa." | FULFILLED | `:852-884` with per-mode checks at `:865-870`; fixtures `unqualified-json-object` (schema-only evidence) and `unqualified-json-schema` (object-only evidence) refused. |
| "Under the approved r2 qualify-and-forward policy, retain raw reasoning_content ... require ... reasoningReplay true" | SUPERSEDED by ruling | `reasoning_content` is not an allowed message member (`strictObject(m, "role","content","name","tool_calls","tool_call_id")` `:699`), so every replay form (empty, non-empty, null, any role) is refused with the generic "unsupported shape" error. Matrix `replay_refused_{assistant,user,tool}_*` and fixtures `*-replay-refused` PASS. See LOW-4 for the diagnostic wording. |
| "Add `TestQualificationShapeMatrix` ... separate minimal triggering requests ... qualified positive and undeclared negative differ only in the evidence tested." | FULFILLED | `local_qualification_test.go:167`; 80+ subtests PASS, each shape with `qualified`/`undeclared` pairs. |
| "Add `TestQualificationShapeDefaultsAndBounds`: same matrix via requestOptions injection ... exact input+output=context and one-byte-over refusal, output ceiling/equal-alias limits ... prove the unknown-defaults regression fails without the post-merge allowlist check." | FULFILLED | `local_qualification_test.go:344`; subtests `default_shapes/*`, `exact_context`, `one_byte_over_context`, `output_ceiling`, `equal_maximum_aliases`, `unequal_maximum_aliases`, `raw_store_*`, `store_request_and_default_controls/*` PASS. The "fails without the check" sensitivity claim is INFERRED from the `default_shapes/unrelated_unknown_*` cases (not re-run against a mutated validator). |

### Task 4: refuse installation of profiles with uncovered routing capabilities

| Requirement | Status | Evidence |
| --- | --- | --- |
| "In the integrated `qualifiedModelLocked` path add one coverage predicate after the existing identity/default/bound joins: tools requires tools evidence; structured requires both existing JSON mode entries; vision, embeddings and unknown capabilities fail; empty capabilities add no requirement." | FULFILLED | `readiness_contract.go:160-172`: `CapTools` → `*q.Shapes.Tools`; `CapStructured` → both `json_object` and `json_schema`; `default` → refuse; loop is a no-op for empty capabilities. Placed after the identity/digest/default/bound/envelope joins (`:128-159`). |
| "Extend `TestStrictCapabilityDerivation` with tools missing/covered, structured missing/one-mode-only/both-covered, vision refused, embeddings refused and empty capabilities accepted." | FULFILLED (PASS; subtest names not individually read) | `local_qualification_test.go:932`. |
| "policy/evidence changes that remove required coverage withdraw readiness and fence an earlier qualified validation ... Extend `TestStrictQualifyThenAdmitIsFencedByReinstallation`." | FULFILLED | `local_qualification_test.go:1114`: cases "policy requires uncovered structured output" and "evidence loses tools coverage"; asserts no advertised capability and ticket unavailable after coverage loss. |
| "Update the derivation comment to include coverage." | FULFILLED | `readiness_contract.go:81-82` ("Its routing capabilities must be covered by the qualification's shapes."). |

### Task 5: synthetic OpenCode forwarding and refusal before effects

| Requirement | Status | Evidence |
| --- | --- | --- |
| "Create synthetic fixture qualifications for engine `llamacpp`, model `fixture-model`, context 4096/output ceiling 1024, request/message/tool/schema overheads 8/4/16/16. Use integrated NE synthetic attested identity/envelope constructors." | FULFILLED | `opencode_shapes_test.go:67-75` (shapes from fixture, overheads 16/16 on top of `testQualification`'s 8/4), profile `:101-105` (4096/1024); NE fixture via `newReadinessEnvelopeFixture(t, "llamacpp")` `:240-245`. |
| "`testdata/opencode-shapes.json` contains named cases with qualification ID, synthetic body, expected acceptance/bounds or stable error category and an arithmetic note ... Pin MSI CLI 2.1.1/native `ba546ec`/inventory `dd0b69e3` and PC3 2.0.32/native `843a6c5`/inventory `a88a4f7`." | FULFILLED | fixture file `sources` block carries both pins and conditions; test asserts them `:52-61`. Every accepted case has `arithmetic`. |
| "Include PC3 generic store:false accepted/stripped, store:true refused and store-absent controls with identical accounting." | FULFILLED | `plain-text` (18), `pc3-store-false` (18), `pc3-store-true` refused. |
| "Include `tool-roundtrip-no-active-tools` ...; first-turn choices required/auto/none/named; no-active-tools omission; generic `max_completion_tokens:16`, maximum omission/defaults and the existing max_tokens alias; plain text and native streaming usage." | FULFILLED | all named cases present; the required-name list is enforced in the test `:93-98`. |
| "Include replay-with-tool-roundtrip and empty/nonempty replay positives with exact synthetic evidence, plus undeclared/wrong-role/null refusals." | SUPERSEDED by ruling | implemented as `replay-with-tool-roundtrip-refused`, `empty-replay-refused`, `wrong-role-replay-refused`, `null-replay-refused` (all `unsupported shape`); fixture `evidence` text says replay "remains refused pending the owner's answer". |
| "Existing JSON modes and raw-overlay response_format/parallel cases are regression/overlay coverage ...; effort remains refused. Add image/provider-option refusals." | FULFILLED | `overlay-json-object`, `overlay-json-schema`, `overlay-parallel-refused`, `effort-refused`, `image-refused`, `provider-option-refused`. |
| "Add `TestOpenCodeShapeFixtures` ... Require unique nonempty case names and the named required cases; derive coverage from names, never a hard-coded changing count. Compute expected accounting with explicit hand arithmetic and independently compacted synthetic JSON." | FULFILLED | `:209`; uniqueness `:83-91`; required names `:93-98`; independent oracle `openCodeInputBound` (`:109-127`) uses `encoding/json` + `json.Compact`, not the validator. |
| "Add `TestStrictHTTPRefusesUnqualifiedShapesBeforeSideEffects`, table-driven on ingress and self ... synthetic engine stopped/cold, enabled start-on-demand and no residency. Assert 400 `strict-qualification`, zero fake-engine hits, active slots, tickets and markers, and no admission/wake or admission/unload notifications." | FULFILLED | `:287-325`; fixture with `warm=false` leaves residency empty and the backend cold, `StartOnDemand = true` (`:257,:263-269`). |
| "Add separate `TestStrictHTTPOpenCodeQualifiedForwarding` success subtests with a fresh harness each time ... loaded/resident, healthy and intended on ... each get 200 and exactly one fake-engine hit; assert synthetic tool IDs/arguments/results, reasoning_content ..., tools:[], maximum spelling and defaults unchanged, with expected response headers/body forwarded. Use stream false and ordinary synthetic Chat Completions JSON here." | FULFILLED (replay part SUPERSEDED) | `:326-366`; `assertOpenCodeEffective` compares every decoded field of the forwarded body against an independent effective-body oracle; response body and headers asserted. |
| "Add a native streaming usage success subtest, using a loopback fake that emits valid Chat SSE deltas/usage/terminal data ... Include synthetic reasoning_content response data and a subsequent assistant-history request replaying it ... New listeners bind `127.0.0.1:0`, check excluded ports." | FULFILLED for SSE and port checks; replay second turn SUPERSEDED (asserted refused, one engine hit total) | `testOpenCodeStreaming` `:404-443`; `openCodeEngine` binds `127.0.0.1:0` and refuses 8080/11434/1234/14318-14323 (`:394`). |

Edge probes against the inventories (MSI `ba546ec`, PC3 `843a6c5`): history-only `tools:[]` accepted only with tools evidence (matches both inventories' `tool-roundtrip-no-active-tools`); generic `max_completion_tokens` and omission-with-defaults accepted; `stream:true`+`stream_options.include_usage:true` accepted with zero charge; `tool_choice` emitted only with active tools in both releases, and PAIR additionally accepts `none`/`auto` without tools; PC3's default `store:false` accepted and stripped; both releases' automatic `reasoning_content` replay is refused (so a second native turn after a local reasoning model returns reasoning will fail 400 until the owner answers; this is the known, ruled state, not a defect); camel `parallelToolCalls` is never emitted and snake `parallel_tool_calls`/`reasoning_effort`/raw-overlay unknown keys are refused on raw and effective bodies.

## Unit 2: SR 5a-ii (`d9f2276`) and 5b (`40046f8`)

Rulings applied: `declaredResidentCount` required, 1..256, bound to the policy revision, fail closed; `Load.ContextLength` equals the policy's context tokens exactly.

### 5a-ii

| Requirement | Status | Evidence |
| --- | --- | --- |
| "At proxy prepare/final dispatch require profile context equal to current native per-slot allocation and qualification covering actual context/output/slots." | FULFILLED | `nvpair-proxy/residency.go:116` (shared helper), `:120-129` (`pr.ContextTokens == envelope.PerSlotContextTokens`, `q.ContextTokens >= envelope`, `q.MaxOutputTokens >= envelope.MaxOutputTokens`, `m.MaxConcurrent == envelope.Slots`); captured in the hold `:233`; revalidated at dispatch through `residentBindingCurrentLocked` `:260-264` from `:274`. Tests `TestResidentNativeAllocation`, `TestResidentNativeAllocationRevalidation` (`residency_test.go:20,58`) PASS. |
| "Envelope/run/config replacement revokes undispatched holds through existing identity/generation invalidation." | FULFILLED | `:263` compares the hold's captured identity with `a.strict.identity[engine]`; revalidation cases "envelope replacement", "run replacement", "qualification narrows without generation refresh" PASS. |
| "Manager uses NE's validator before final grant and immediately before load, without a network probe that autoloads." | FULFILLED | `residentCheck` (`nvpair-engine-manager/residency.go:331`) calls `currentServingEnvelope` `:344` (local validation only, `serving_identity.go:145-173`); invoked before the grant relay `:458` and again after the grant and before `dispatchAction` `:490,:494`. `TestResidentDailyAdapterNativeRequirements` asserts zero HTTP effects from preflight. |
| "After load require fresh model-specific observed context/slots equal to the validated envelope; mismatch withdraws eligibility through NE's existing path and cannot report loaded/satisfied." | FULFILLED | `residentObservation` `:524-571`: fresh `loaded_models` read, `observeLlamaServing`, then `currentServingObservation` must equal the envelope `:564-567`. `TestResidentObservedAllocation` (`residency_test.go:285`) "native context mismatch"/"native slot mismatch"/"unknown allocation" → unknown and identity withdrawn. |
| "Preserve the distinction between cold configured, launch-validated and observed evidence." | FULFILLED | `residentConfiguredContext` consumes a cold `ServingReadback.Configured` (`:347-351`); observation comes only from `servingObservationsLocked` with freshness budget (`serving_readback.go:177-190`). |
| "No Prepare/Dispatch DTO changes." | FULFILLED | `PrepareResidentParams`/`DispatchResidentParams` untouched in the merges (diff stats). |

### 5b

| Requirement | Status | Evidence |
| --- | --- | --- |
| "Replace the Ollama-pinned resident adapter with one manifest-shape rule ... Accept only HTTP POST, model parameter `model`, `body_schema` exactly `{model: string}`, no fixed binding params, no `ParamsIn` override, `restart_after`, `cmd`, `remove_path`, `model_resolution`, `progress_protocol` or content mutation." | FULFILLED | `residentLoadCall` `:218-240` and `residentSafeHTTP` `:242-246`. `TestResidentDailyAdapterUnsafeShapes` (`residency_test.go:214`) covers each named restriction. |
| "Require safe `loaded_models` HTTP GET with a declared Result loaded filter and the same no-mutation/no-command restrictions." | FULFILLED | `:360-363` (`Result.Match`, `Array`, `Field`, GET, no body schema). |
| "Send exactly the normalized `{model}` body, ignoring validated keepAliveSeconds." | FULFILLED | `:236-239`; `TestResidentLoadMappingAndObservation` PASS. README `:99-101` and spec `:371-372` say so. |
| "Reuse `checkServingLoad` before mutation admission and the machinery of `guardServingLoad` after admission ... extract a helper for that caller ... never call the lock-taking guard recursively." | FULFILLED | `checkServingLoad` at `:435` before `admitMutation` `:439`; `guardServingLoadLocked` (`serving_llamacpp.go:808`) after `residentLock(ctx, &st.opMu)` `:447-452`; shared `registerServingLoad` `:785`. |
| "Load dispatch must use the registered cancellable context ... Settings-driven `stopServingLoads` must cancel an in-flight resident load; cancellation/unknown must never become loaded merely because later inventory contains the name." | FULFILLED | `dispatchAction(ctx, ...)` `:494` uses the guard's ctx; `TestResidentDailyAdapterSettingsStopCancelsLoad` (`residency_test.go:157`) asserts `ResidentCancelled`, no readback, no leaked registration. |
| "Before final grant and again immediately before load ... require exact current run/config/content, finite effective per-slot context/output/slots, matching profile allocation through revision/identity binding, and native retention/protection support." | FULFILLED | `:344-358`: envelope identity equality, `residentConfiguredContext` equality `:366-383`, `servingRetentionDisabled(envelope)` and `RouterCapacity >= DeclaredResidentCount` `:352-354`. Retention-disabled spelling is NE's exact `"-1"` (`serving_identity.go:56`, `TestServingRetentionDisabledIsExact`, `TestLlamaServingRetentionContract`). |
| "Unknown/stale/ambiguous bounds refuse without effects." | FULFILLED | `TestResidentDailyAdapterNativeRequirements` asserts zero loads/models/gets and no grant for "native idle expiry" and "exceeds router capacity". |
| "Preserve the original absolute deadline (maximum 60 seconds and finite switch timeout); a manifest timeout cannot extend it." | FULFILLED | `reconcileResident` `:193-196` derives the deadline from `TimeoutMs`; wire rejects `timeoutMs` 60_001 (`wire_test.go:305`); readback runs under the same ctx `:499`. |
| "Add `TestResidentDailyAdapter...` for manifest-derived endpoint/body/filter, unsafe-shape refusal, finite options, effective mismatch, unsupported retention/capacity, veto during grant/lock wait, stale source/config, duplicate ID, timeout and response loss. Add undeclared-model zero-effect refusal and settings-stop cancellation." | FULFILLED | `TestResidentDailyAdapter{FinalGrantRevalidatesNativeLaunch,NativeRequirements,SettingsStopCancelsLoad,UndeclaredLoadBeforeAdmission,UnsafeShapes,ReapplyRequiresFreshObservation}` plus the pre-existing `TestResident*` veto/duplicate/timeout/cancellation cases; all PASS. |
| "Extend desired-state round-trip tests to prove no residency operation changes checkpoints, allocated ports, intents or OffBy, including failed reads/writes." | FULFILLED | `desiredstate_test.go:150` and `:203` PASS. |
| Ruling: "`declaredResidentCount` is required, 1..256, ... fail closed." | FULFILLED | `wire.go:476-486` field; `:488-491` range; required/non-null/unknown-field rejection via `decodeResident` `:794`, `:943-964`; tests `wire_test.go:293-313` (0, -1, 257, 1.5, "1", overflow, exact 256). Manager rejects invalid params with `-32602` (`residency.go:201-206`). |
| Ruling: "... bound to the policy revision" | UNVERIFIED (UNV-2) | The manager carries the count beside `PolicyRevision` and documents the binding (README `:87-88`, spec `:272`), but has no policy to bind against; the derivation from the revision's resident list is 5c (`roadmap/sr-5c`, unlanded). |
| Ruling: "`Load.ContextLength` equals the policy's context tokens exactly." | FULFILLED at the two enforcing owners | manager: `int64(p.Load.ContextLength) == bound` against the cold per-slot envelope `:382`; proxy: `pr.ContextTokens == envelope.PerSlotContextTokens` `:124` and shared `ValidateResidentContext` (`shared/nodepolicy/policy.go:639-644`). Equality to the *policy profile* itself is transitive through these two checks only once the broker derives `Load` from the same revision's profile (5c). |

Fail-closed bounds probed: `contextLength` 0/negative rejected (`wire.go:190-192`); `ValidateResidentContext` rejects nonpositive request or bound and request above bound without arithmetic (`policy.go:640`); `residentConfiguredContext` requires exactly one envelope for the normalized model and a representable positive bound (`residency.go:374-381`); proxy refuses when the profile is not unique for the normalized model (`:104-113`). `TestResidentContextBound{Prepare,PerModel,Revalidation}` (proxy) and `TestResidentContextBound{Configured,ColdPreflight}` (manager) PASS.

## Unit 3: NE readback transport classification (`8b2707e`, `02f8ad6`)

Ruling: "transport failures (connection error, timeout, body read error, any non-200 including 3xx; redirects not followed on observation reads) make that sweep's observation UNKNOWN with no drift latch; only semantic mismatches latch."

| Clause | Status | Evidence |
| --- | --- | --- |
| Connection error → unknown, no latch | FULFILLED | `models.go:205` sets `loadedTransportFailed` on any `Action` error; `observeLlamaServing` clears proof and returns before any latch `serving_readback.go:115-121`. `/props` dial error is a plain error (not `ServingIdentityRefusal`) → `transportFailed = true; continue` `:140-147`, then `:155-157` return. Tests `TestLoadedServingTransportFailureRecoversNextSweep/{\/models,\/props}/connection`. |
| Timeout (action or sweep deadline) → unknown | FULFILLED | ctx errors surface as transport; `ctx.Err() != nil` short-circuit `:119`, `:159-161`; cases `timeout`, `sweep timeout` PASS. |
| Body read error → unknown | FULFILLED | `actions.go:227-229` returns an error (previously discarded); `llamaServingProps` `:88-91`; `TestActionBodyReadFailureIsTransportError` (not an `ActionHTTPError`), case `truncated body` PASS. |
| Any non-200 including 3xx and other 2xx → unknown | FULFILLED | `actions.go:234-235` requires exactly 200 for `list_models`/`loaded_models`; `/props` `:89` requires 200; cases `404`, `500`, `201 valid body`, `204 empty body`, `redirect` PASS; `TestActionObservationRequiresHTTP200` shows mutation actions keep the 2xx contract. |
| Redirects not followed on observation reads | FULFILLED | `actions.go:214-220` copies the client with `ErrUseLastResponse` for the two inventory actions only; `/props` `:79-81`; `TestLoadedServingRedirectIsTransportUnknown` asserts zero redirected requests for 301/302/307 and `TestActionRedirectPolicyPreservesMutationBehavior` that `load_model` still follows. |
| Only semantic mismatches latch | FULFILLED | unknown residency semantics (200 body that does not parse to the pinned status shape) `:124-126`; undeclared loaded ID `:136-138`; `/props` 200 with invalid/oversized/nonpositive values is a `ServingIdentityRefusal` `:92-103` → mismatch latch `:148-151`; `validateServingSnapshot` failure (non-ctx) `:158-163`. `TestLoadedServingSemanticEvidenceSurvivesTransportFailure` proves a semantic disagreement in one model latches even when another model's props read fails transport, in either order. |
| Resident observation caller interaction | FULFILLED | `residency.go:542-546` passes `readErr != nil` as the transport flag; `known` false → `nil` observation → `ResidentUnknown` (`:495-520`); a `/props` transport failure leaves `currentServingObservation` empty → `nil` → unknown. `TestResidentPostLoadTransportFailureRemainsUnknown` (`residency_test.go:316`) covers inventory connection failure, 201, 204 and props connection failure, asserts no drift and no retained proof, and that a fresh retry restores `ResidentLoaded`. |
| Spec text | FULFILLED | `spec.md:172-191` matches the code (exact-200, redirects, shared response, semantic-vs-transport, "a sweep with any transport failure publishes no observed proof"). |

Edges probed beyond the tests (all VERIFIED from code): a 200 inventory with an empty body becomes `null` (`actions.go:237-238`) and then fails extraction → semantic latch, consistent with "invalid ... in a fully read 200 response"; a sweep where the engine stopped between the snapshot and `Status` (`models.go:173`) returns with `known=false`, and `observeLlamaServing` exits at `s.current(st)` because `st.running` is false (`serving_identity.go:124-127`) → no latch; `reconcilePresence` performs no probe for an owned run (`status.go:161`), so `Status` cannot itself misclassify. See LOW-5 for the 8 MB truncation nuance.

## Unit 4: manual node reference validation (`2a18bd3`)

Ruling: "Manual node references: a bounded string kind that rejects control and format (Unicode Cf) characters." Input: UNV-3 of `audit-pa-j-b.md` (J Task 1 silently drops frames whose `scheduledOn` falls outside `strictID`).

| Clause | Status | Evidence |
| --- | --- | --- |
| Bounded string kind for the manual destination | FULFILLED | `workload_projection.go:101-106` (`scheduledOn: "node-reference"`), bound 256 UTF-8 bytes `:110`, validator `:125-135` (non-empty, `len <= 256`, valid UTF-8, no `unicode.IsControl`, no `unicode.Is(unicode.Cf)`). |
| Rejects control characters | FULFILLED | tests `TestWorkloadManualDestinationRejectsBeforeEffects` (`workload_projection_test.go:248`): newline, NUL, DEL, C1 U+0085. |
| Rejects format (Cf) characters | FULFILLED | same test: U+202A-U+202E, U+2066-U+2069, U+200E/200F, U+061C, U+200B-U+200D, U+2060, U+00AD, U+FEFF; plus over-bound (257 bytes, 86x3-byte), invalid UTF-8, empty. |
| Legitimate free-form IDs reach consumers (UNV-3 repair) | FULFILLED | `TestWorkloadManualDestinationReachesConsumers` (`:212`): space, slash, CJK, accented Latin, exact 256-byte bound reach store, client, scheduler, manager, projection and replay. |
| Strict identities stay strict | FULFILLED | `id`, `engine`, `runId`, `originatedFrom` remain `identity` (`:102-103`), `TestWorkloadStrictIdentitiesRemainStrict` (`:289`). |
| A drop is observable (prior LOW-3) | FULFILLED | fixed-category debug logs at every projection site `:180`, `:256`, `:355`, `:499`; `TestWorkloadProjectionDiagnostics` PASS. |

Edges probed. Producers of the reference: the proxy stamps `scheduledOn` from the routing candidate id (`nvpair-proxy/http_handler.go:1737`) with `omitempty` (`proxy.go:185`), so the new "empty is malformed" rule (`:126`) is not reachable from the proxy (LOW-6). The candidate id for a manual node is the broker's `key` (`nvpair-ui-broker/manualnodes.go:176-191`): the host UUID once node-info reports it, else the manual id. The manual id is `entry.Name` when the user typed one, else `"manual:" + address` (`nvpair-manual-nodes/manager.go:736-741`); `node/add` validates only `address != ""` (`:680-692`) and the proxy's `node/add-manual` only `id != ""` (`nvpair-proxy/proxy.go:1581-1594`). So user-typed names are a real production path (answering the prior audit's owner question 4), and names longer than 256 bytes or containing a control/Cf character are still accepted at creation and dropped at the broker (UNV-1). Other RPC fields carrying the same reference: `proxy/request-started.nodeId` (`http_handler.go:1707-1713`), `node/activity.hostUuid` (`proxy.go:448`, consumed at `broker.go:2054-2064` with a non-empty check only), `node/discovered`/`node/updated` `id` (`proxy.go:1598-1603`), and the manual-nodes `errors:report` id (`manager.go:392-402`). None applies the bounded kind (LOW-1); none drops frames.

## Findings

No RED finding.

### UNV-1: manual node names are bounded only at the consumer; oversize or control-bearing names still vanish silently

- Claimed invariant: ruling "Manual node references: a bounded string kind that rejects control and format (Unicode Cf) characters"; prior UNV-3 disposition "either validate `node/add-manual` ids ... so refusal is explicit, or relax `scheduledOn` to a bounded string kind. At minimum add the fixed-category debug log".
- Code path: name accepted unvalidated at `nvpair-manual-nodes/manager.go:680-692` and used as the id `:736-741`; bridged as the proxy candidate id `nvpair-ui-broker/manualnodes.go:176-191`; stamped into `scheduledOn` `nvpair-proxy/http_handler.go:1737`; rejected as `malformed` at `workload_projection.go:125-135` with only a debug log.
- Observed result (from code): a workload routed to a manual node whose typed name exceeds 256 UTF-8 bytes or contains any Cc/Cf rune never appears in the desktop, TUI or scheduler; the user receives no refusal when creating the node. Not reproduced against a real client.
- Affected state: workload visibility and scheduler pending counts for such manual nodes; not inference.
- Disposition: repair now at the producer (apply the same bound and character rule to `node/add` `name`, returning `-32602`), or owner question: are names over 256 bytes or with invisible characters ever legitimate? The consumer-side rule delivered by `2a18bd3` is correct and should stay.

### UNV-2: `declaredResidentCount` revision binding is not yet enforceable

- Claimed invariant: ruling "`declaredResidentCount` is required, 1..256, bound to the policy revision, fail closed".
- Code path: `wire.go:476-491` validates range and presence; the manager compares it with `RouterCapacity` (`residency.go:352-354`) and carries `PolicyRevision` alongside, but nothing in the landed code derives the count from the revision's resident list; the broker reconciler that must do so is 5c (`roadmap/sr-5c`, under review).
- Observed result: the required/bounded/fail-closed parts are proven (`wire_test.go:293-313`, `TestResidentDailyAdapterNativeRequirements`); the binding is documented (README `:87-88`, spec `:272`) and unprovable until 5c lands.
- Affected state: none today (no producer of `engine:reconcile-resident` exists outside tests; `internalClientMethods` blocks public callers).
- Disposition: defer to the 5c review; require a 5c test that the count sent equals the revision's resident set for that engine.

### UNV-3: full-module and Windows acceptance not evidenced here

- Claimed invariant: SR plan Verification "run full affected module plain/race tests once after focused checks"; B plan "PC3 focused hermetic gate ... runs the same selection without race under the fixed-path wrapper".
- Observed result: in this container the proxy full module has 4 failures and the engine-manager 19, all outside the audited units and attributable to the sandbox (public DNS refused; Python environment override present; procfs listener inspection; a flush-deadline test that also fails at the pre-join25 base). Audited prefixes are all green. No `-race`, no Windows run.
- Affected state: acceptance status of `integration/join25` ("pending Windows gate" in the common brief).
- Disposition: owner/coordinator question: attach the Mac `-race` and PC3 plain logs for the selections named in the plans; this report cannot substitute for them.

### LOW-1: the bounded node-reference kind is applied to `scheduledOn` only

`proxy/request-started.nodeId`, `node/activity.hostUuid`, `node/discovered`/`node/updated` `id` and the manual-nodes `errors:report` id carry the same manual reference without the bound (citations in Unit 4). No frame is dropped and no consumer misbehaves from code reading; the TUI renders the name through its namer. Informational; consider sharing one `nodepolicy` helper if the producer fix in UNV-1 lands.

### LOW-2: non-Cc/Cf invisible characters pass the node-reference kind

U+2028/U+2029 (Zl/Zp), U+3164 and other non-control, non-format invisibles are accepted (`workload_projection.go:130-133` checks only `IsControl` and `Cf`). This matches the ruling's wording exactly; recorded so the rule is not mistaken for a general "printable only" filter.

### LOW-3: generated `desktop/docs/services-api.md` cannot see the touched SR payload

The resident methods are shared constants (`shared/nodepolicy/wire.go:166-170`) and the extractor builds its constant map from the component's own Go files (`desktop/scripts/verify-service-contracts.ts:104-133`), so `engine:reconcile-resident`, `node/{prepare,dispatch,finish,observe}-resident` and the dispatch request/reply notifications appear neither in `services-api.md` nor in `service-contract-exceptions.json` (zero matches for "resident" in both). There is therefore nothing inconsistent with `ReconcileResidentParams.declaredResidentCount`; there is a coverage gap that SR 6a ("audit internal methods/notifications against contract exceptions") already owns. `proxy/request-started` is listed as ignored (`services-api.md:240`), unaffected by these units.

### LOW-4: replay refusal diagnostic is generic

Under the required-false ruling every `reasoning_content` member is refused as "messages[i] has an unsupported shape" (`local_qualification.go:699`). The plan's sentence "explicitly reporting that local multi-turn reasoning compatibility remains pending real-template qualification" belongs to the unapproved r2 branch, so this is not a gap against the ruling; a dedicated message would help operators diagnose the second-turn 400 that both inventoried OpenCode releases will produce.

### LOW-5: oversized 200 inventory bodies are truncated, then classified semantic

`dispatchAction` reads at most 8 MiB without error (`actions.go:227`), so a larger 200 `loaded_models` body is silently truncated and fails extraction → semantic latch, whereas `/props` detects overflow explicitly (`serving_readback.go:92-94`) → semantic refusal. Both outcomes are "semantic" and match the ruling; the asymmetry is only in how the condition is detected. No realistic inventory approaches the bound.

### LOW-6: an explicit empty `scheduledOn` is now malformed

Before `2a18bd3` an empty placement string was accepted; now `""` is rejected (`workload_projection.go:126`). Every in-tree producer omits the field when empty (`nvpair-proxy/proxy.go:185`, `nvpair-workload-manager/workload.go:75`, both `omitempty`), so no frame changes category. INFERRED: an out-of-tree or older peer that emits `"scheduledOn":""` would be dropped.

### LOW-7: manager identity lookups use the un-normalized model name as the digest key

`residentCheck` and `residentObservation` index `id.ModelDigests[p.Model]` (`residency.go:345`, `:560`) while `residentIdentity` normalizes names (`:253-262`). A request naming the model in a non-canonical alias fails closed as `unqualified` rather than unsafely; the broker (5c) will send the normalized name. No effect on safety.

## Generated docs and contracts

- `desktop/docs/services-api.md`: read only. No entry exists for any resident method (see LOW-3); the workload notifications it lists (`workloads:upsert`/`workloads:remove`) carry no field-level schema, so the `scheduledOn` kind change is invisible to it; `proxy/request-started` is marked ignored. No inconsistency with the Go payloads touched by these units was found; coverage, not correctness, is the gap.
- `services/nvpair-engine-manager/spec.md` and `README.md` (5b and readback text) agree with the code on: exact-200 and no-redirect observation reads, transport-vs-semantic classification, `keepAliveSeconds` validated but not sent, retention exactly `"-1"`, `declaredResidentCount` 1..256 and router-capacity comparison.

## Owner challenges

- The manual-reference rule now lives at the consumer. If free-form names are meant to be supported (they are, via `node/add` `name`), the producer should enforce the same bound so a user sees the refusal when typing the name rather than losing workload cards later (UNV-1).
- Both inventoried OpenCode releases replay `reasoning_content` automatically. With the required-false ruling, any local reasoning model that returns reasoning will break the user's second turn with a 400. This is the ruled state; the report records it so the pending owner answer is not forgotten.

## Questions for the owner / coordinator

1. May manual node names be bounded at creation (256 UTF-8 bytes, no Cc/Cf) with an explicit `-32602`? (UNV-1)
2. Where is the Mac `-race` / PC3 evidence for the selections named in the B and SR plans for `integration/join25`? (UNV-3)
3. Should the 5c review be required to prove `declaredResidentCount` equals the revision's resident set for the engine before it lands? (UNV-2)
