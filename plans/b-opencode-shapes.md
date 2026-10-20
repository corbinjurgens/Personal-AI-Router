<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution plan: B extension — qualification for the OpenCode request shapes actually used

## Revision basis

This revision supersedes the original five-task plan. It applies review `../reviews/b-plan-review-r1.md` (AMEND), including R1–R8, ownership/factual corrections and LOW items. The handoff is `origin/plan/cloud-handoffs:plans/handoffs/b-opencode-shapes.md`; its historical branch/publish instructions are superseded by this assignment. Amend this package only: no product implementation, commits, checkouts, branch creation or vault writes during plan amendment.

Source baseline verified using `git show` and `git grep`: `origin/integration/join10` = `da2d6a3cd042942b743e6cf253640b5cd557e043`. Line references below are navigation aids at that SHA. No product tests, installed-client inspection, engines, models or native acceptance were run for this amendment. Installed OpenCode behavior remains UNVERIFIED until Task 1.

## Goal

Qualify only the OpenCode Chat Completions shapes the installed client needs, starting with tools and conditionally reasoning replay. A strict request using an undeclared shape fails with 400 `strict-qualification` before admission, wake, unload, capacity markers or inference. Accepted effective requests include profile defaults, are conservatively accounted and preserve their synthetic fields when forwarded. Profiles advertising tools/structured without matching qualification do not install. Synthetic tests establish the contract without a real client/model run. NE Task 4 supplies trustworthy model/content/config identity; B consumes that identity and owns request shape validation/accounting.

## Change scope

**Intent.** Bind the actually used client shapes to evidence for the exact engine/model/template, without broad provider-option support.

**Scope: Bounded improvement.** Change the proxy's local qualification schema, request validator/accounting, install coverage check, focused fixtures/tests and qualification documentation. First inventory installed-client metadata and configuration read-only on MSI/PC3. Product changes are limited to the Files below after external file release.

**Basis.** The owner milestone is tools plus reasoning replay if enabled. The review proves the previous optional-field surface and test prescriptions exceed that evidence. The coordinator's adopted answers below resolve the product choices, subject to owner veto.

**Compatibility.** Tools and the already supported `json_object`/`json_schema` modes become explicit evidence boundaries. Preserve existing text, sampling, defaults, streaming and output-bound behavior except the named qualification tightening. Coordinate one incompatible artifact revision with NE Task 4; reject old evidence and require requalification, without a fallback. Do not assume the fleet has no version-1 evidence: shipping no evidence does not prove operators have created none. No new RPC, shared wire, advertisement or desktop contract is required by B.

**Non-goals.** `reasoning_effort` and `parallel_tool_calls` support; new structured modes/provider options; vision, audio, attachments, files, `user`, `verbosity`, `n`, logprobs or context overrides; digest production; another identity store, dispatcher, resident loader or launch compiler; I/public/hybrid activation; real-client acceptance; Strata work (owner deferred it on 2026-10-21).

## Context the executor needs

Verified at `da2d6a3`:

- `services/nvpair-proxy/local_qualification.go:39,98-190` has artifact version 1, strict document decoding and registry validation. `validateRequest:251-309` checks raw keys, merges defaults, checks scalars/output, accounts messages/tools/formats, then compares input+output with the smaller profile/qualification context. Qualification must examine the effective body, not just the original request.
- `chatRequestKeys:242-245` excludes effort and parallel calls. `services/nvpair-proxy/chat_adapter.go:446` independently appends `parallel_tool_calls` and iterates the resulting list. Keep that list and the adapter unchanged; B needs no added top-level request key for nested replay.
- `local_qualification.go:378-469` accepts system/user/assistant/tool roles, string/text-part content and assistant null content only with nonempty tool calls. Tool calls belong on assistant messages. Non-null `tool_call_id` is checked against the tool role; its pointer does not distinguish absent from null. Several existing scalar decoders likewise accept null. Do not describe all current primitive validation as strict typing.
- `accountTools:471-516` accepts an empty tools array and tool choice without tools today. A function must be named; `parameters` is optional, and must be an object only when supplied. `accountResponseFormat:520-545` already accepts text, json_object and named object json_schema. B gates this existing surface; it adds no structured breadth.
- `services/nvpair-proxy/tiers.go:86-88` derives tools needs from nonempty tools/functions arrays, not mere field presence. `services/shared/nodepolicy/policy.go` defines tools, structured, vision and embeddings capabilities. The current qualification install path does not join them to shape evidence.
- `services/nvpair-proxy/readiness_contract.go:125-172,212-227` joins engine/model, launch/model digests, defaults and finite bounds, and fences installations by generation. `services/nvpair-proxy/ingress.go:310-350` qualifies before admission and checks body/model agreement. Preserve these joins and their post-NE equivalents.
- `services/nvpair-proxy/local_qualification_test.go:25-75` creates synthetic ollama/m fixtures at context 4096/output 1024 through `testQualification`, `testIdentity`, `installQualified`; `:460-519` tests ingress/self refusal before effects. `services/nvpair-proxy/admission_http_test.go:280-300`'s recording engine returns ordinary JSON, not SSE. These are synthetic helpers, not a requirement for installed Ollama.
- `services/nvpair-engine-manager/serving_files.go:97-113` creates private `generation-*` directories, explicitly not content identities. `serving_llamacpp.go:699-713` hashes source/snapshot approval inputs but publishes no attested content digest. `model_identity.go:64-90`'s old launch hash covers executable metadata, arguments and explicit environment. Native list-model content digests are missing at this baseline. This current limitation is assigned to NE Task 4; a shapes record alone cannot establish freshness.
- `services/nvpair-proxy/cors_test.go:433-450` can contact `PAIR_CORS_PARITY_URL` or start `PAIR_CORS_OLLAMA_EXECUTABLE`. Unqualified full-suite commands are unsafe. `services/tests/broker_management_test.go:128,196` uses protected product ports. B never runs `services/tests`, including test listing on Mac.

The old unpinned SDK-main table is withdrawn. It establishes neither installed OpenCode behavior nor configuration. No durable product documentation shall repeat it as fact. Task 1 replaces it with pinned sources and synthetic evidence, distinguishing SDK capability, OpenCode call construction and active configuration.

## External ownership and execution ordering

These are file-release gates, not satisfaction by the existence of another plan. Before each affected task, the coordinator records accepted integration SHAs, proves ancestry at its execution base and releases the files. Re-read the integrated source; never overwrite another package's changes. No same-file authors run concurrently.

| Owner | Reservation and required B action |
| --- | --- |
| NE Task 4 (`ne-serving-envelope.md`) | External prerequisite for B Tasks 2–6: integrate its exact attested content/launch identity, provenance acceptance and effective bounds first. Shared files include `services/nvpair-proxy/local_qualification.go`, `readiness_contract.go`, their focused tests including `local_qualification_test.go`, `admission.go`, `http_handler.go`, `ingress.go` and `docs/local-qualification.mdx`. Consume NE's actual schema and `operator-attested-sha256-v1` acceptance; do not invent an independent digest/migration. NE owns native identity/bounds/provenance; B owns shape/accounting. NE Task 3 alone does not satisfy this prerequisite. The coordinator must resolve NE's historical Strata dependency without commissioning Strata work in B. |
| I T3a | Reserve constructor/handler/common fixtures discovered by `rg -l 'NewProxy|&Proxy|handleHTTP|serveIngress' services/nvpair-proxy --glob '*_test.go'`; this can include B's current helper owners. Run that discovery at the execution base before claiming the lane. Consume the accepted identity migration or obtain an explicit serial file release; preserve missing-identity negatives and acknowledged run/source identity. |
| I T4a/T4b | Own ingress/body/handler buffers and sole dispatch/sink cutover. B does not edit those production owners. B's HTTP fixtures consume the accepted handler/body ownership contract, preserving lease lifetime, cancellation and response ownership. At each accepted I join, rebase B through the coordinator and rerun its focused tests; adapt B tests at their owner, never resurrect removed entry points or create a second dispatcher. |
| I T6, NE T5, SR T6a | Reserve proxy spec and qualification docs. Documentation order is accepted NE T4/T5 and I T6 -> B Task 6 -> SR T6a consumes B text. Coordinator must release the docs before B writes, without requiring successor SR T6a first and creating a cycle. B test/schema execution does not require I activation. |
| SR T5a | Consumes the joined NE/B qualification and bounds contract. B does not implement resident loading. Publish B's accepted schema/requalification boundary to this successor before it freezes fixtures. |
| XA / SRR T3 / E4 T2 | XA owns `services/tests/remote_configuration_test.go`, switch/harness extraction and `services/readme.md`; SRR owns restart/reconnect process tests/module files and procedure docs; E4 T2 owns TUI switch/detail. No direct B source dependency. Coordinate Windows machine isolation intervals; B does not duplicate their acceptance. |

For I T3a/T4a/T4b, a serial release means the coordinator records which accepted implementation currently owns the seam and orders its next overlapping author after B, with mandatory rebase/test on integration. It never means concurrent edits or pretending an unlanded implementation is present. If no such reservation is available, affected B tests wait. No reverse ordering of NE T4/schema is authorized by this plan.

## Decisions (settled — coordinator adopted, owner may veto)

- Keep reasoning replay disabled/refused until Task 1 proves the installed client/configuration replays it. If it does, qualify the exact llama.cpp template and reasoning format under NE-bound content/config identity, with synthetic tool-history replay cases. A synthetic pass is not real model qualification; absent trustworthy template evidence leaves that model's replay false.
- Defer `reasoning_effort` and `parallel_tool_calls`. Reject their presence explicitly while unsupported, including null, false and values inserted through defaults. Inventory evidence that the client uses them produces a compatibility blocker for the coordinator; it does not authorize generic support here.
- Fail closed at installation: tools/structured capability without matching shape qualification means that profile does not install. Correct evidence/profile declarations rather than advertise an unusable capability.
- Digest reporting belongs to NE Task 4. No Strata implementation or acceptance in B; the owner deferred Strata on 2026-10-21.
- Keep tools in scope, as required by the owner milestone.
- Synthetic fixtures only; no real body capture, prompt logging, secrets, native engine starts, model loads or API calls to installed inference endpoints. Metadata inspection and contract fixtures suffice for implementation proof.

## Decisions made while writing this plan

- Preserve the existing structured modes under explicit, mode-specific qualification rather than remove current behavior or add modes based on assumed OpenCode use. Inventory controls which client fixtures claim to represent emitted structured requests. This is a plan-time interpretation of the review's preserve-existing-behavior requirement, not an additional owner answer.
- Use a required `shapes` object with required `tools` boolean and required `structuredOutput` array of qualified mode names (`json_object`, `json_schema`). An empty array qualifies neither; each retained mode needs its own evidence. No never-used effort/parallel booleans. Only if inventory proves replay use, add a required `reasoningReplay` boolean in that same incompatible revision; otherwise omit it and reject replay. This keeps unused replay support out of the implementation.
- Coordinate the final schema revision once after NE's accepted join: use NE's agreed combined incompatible revision when it already includes B's exact contract; otherwise advance the accepted NE artifact version by one. Record the resulting exact integer and schema in Task 1's report before Task 2 starts. Do not hard-code an independent version 2 migration over NE.
- Require nonempty tools when the tools member is present and require tool_choice to have nonempty tools. Tool history remains valid without re-sending tool definitions when tools are qualified. Preserve optional function parameters and existing non-shape semantics. Explicitly reject null/newly gated shape members; do not perform a general scalar/null cleanup.
- A structured routing capability requires evidence for both existing JSON modes, because the shared capability has no per-mode granularity. A profile with fewer qualified modes can serve those modes only when it does not advertise the broad structured capability. Empty capabilities impose no extra shape requirement; vision/embeddings never install through this text-only contract.

## Verification safety contract

All task Verify commands below inherit this contract. Audit the selected tests, TestMain/init and transitive fixture helpers at the actual integrated base before execution. Search `os.Getenv`, `os.LookupEnv`, `exec.Command`, `net.Listen`, `ListenAndServe`, `httptest`, dial targets and fixed ports. Include only tests whose effects are synthetic/local and whose listeners use `127.0.0.1:0`. No contact, probe, bind or start on 8080, 11434, 1234 or 14318–14323. Temporary outputs go to an explicit absolute temporary directory outside repositories/synced vaults; use `t.TempDir()` for test data. Check assigned ephemeral addresses against the protected set before serving/dialing; close and retry allocation if one is protected. Do not bind a product port to test its occupancy.

On Mac use the following sanitized focused command shape, substituting each task's anchored TESTS expression. This runs the proxy package only, never `services/tests`:

```sh
env -u PAIR_CORS_PARITY_URL -u PAIR_CORS_OLLAMA_EXECUTABLE \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  go -C services/nvpair-proxy test -count=1 -v -run "$TESTS" .
```

Task `TESTS` values below are explicit selections, not permission for full `./...` runs. Audit any additional live opt-in variables discovered at the integrated base and unset them too. The CORS parity test is excluded, even with its variables unset. Use the same selection with `-race` for the whole-plan Mac race gate. `go -C services/nvpair-proxy vet ./...` compiles/analyzes only. If a selected inherited helper starts processes, contacts installed state or binds a forbidden address, stop that selection and report pending isolation; do not silently treat a skipped check as a pass or broaden B to repair unrelated fixtures.

Windows is a separate coordinator-scheduled hermetic gate on PC3, never `-race`. Read project `TESTING_ON_PC.md` section 10 first. Require a serialized machine reservation with no GPU/measurement session; `GOFLAGS` must include `nvpair-test-exec`, and `scripts/windows-test-bin.ps1 list` destinations must have the fixed-path rules. Use the documented wrapper, sanitized environment and the same audited focused selections, with before/after firewall and survivor checks. Missing elevated rules leave execution pending. Record every skip and screen observation as unmonitored unless someone actually watched. No unconditional Windows full-proxy or `services/tests` gate belongs to B. MSI Task 1 is read-only metadata inspection, not Windows test acceptance.

## Tasks

### Task 1 — Pin the installed OpenCode field matrix without side effects

**Files:** `.claude/plan-runs/pair-roadmap-20261019/reports/b-opencode-client-inventory.md` (new, execution-time metadata report only; no product or installed-client changes)

**Depends on:** none

**Complexity:** judgment

**Do:**

1. On MSI and PC3 where OpenCode is installed, inspect executable/package metadata, installation manifest/lockfile and the effective configuration read-only. Record the installed OpenCode version/build/revision, installation channel, exact provider package (`npm` selection), installed `@ai-sdk/openai-compatible` and relevant AI SDK/core package versions and lock integrity. Resolve OpenCode's actual provider and per-model/variant settings with their configuration precedence; inspect only allowlisted compatibility settings. Never print whole config/environment files, tokens, personal paths, prompt history or credentials. Prefer package metadata; do not execute the interactive client, hooks, package installation, upgrade, network inference, engine or model start. If a binary-only installation lacks resolvable SDK metadata, mark that source relationship unverified, not inferred from latest upstream.
2. Pin OpenCode and SDK source to the exact installed releases/commits using release/package metadata and immutable source URLs (`blob/<commit>/...` or versioned registry artifacts with integrity). Locate actual files from those pinned trees; do not reuse guessed raw paths that returned 404. Cite package version, immutable revision, source path/symbol and stable URL for each matrix row. A moving main/dev URL and provider documentation alone do not prove client emission. Reading source/release metadata over the network is allowed; no request is sent to inference providers.
3. Build a host-specific matrix with columns: request field/message member; SDK construction condition; OpenCode call/config condition; installed source pin; active setting (safe value only); synthetic evidence case; current B accept/reject/ignore; planned qualification/rejection; unresolved evidence. Cover model/messages, max_tokens versus max_completion_tokens and omission/defaults, stream/stream_options/include_usage, tools/tool_choice/tool history, null assistant content, text parts, reasoning_content replay, reasoning_effort, parallel_tool_calls, response_format modes, sampling/stop, extra provider options and attachments. Distinguish source-proven behavior from configuration assumptions; do not label optional SDK mechanics as observed wire traffic.
4. Use the pinned converter/provider contract with exclusively synthetic turns/results to establish first turn, tool round trip and replay construction; no real-body capture. Inspect existing fleet qualification documents by allowlisted metadata only (version, safe evidence ID/revision, shape presence, installed model/template identity references); no contents, local paths or secrets in this report. Record whether requalification is required; never claim nothing deployed is invalidated merely because no evidence ships in the repository.
5. Record replay branch as `used`, `not used` or `unverified`. Only `used` permits the replay implementation in Tasks 2/3/5; the other states leave it refused and omit its artifact flag. Inventory the exact llama.cpp template/reasoning-format/jinja configuration references read-only and identify the NE-bound evidence needed; never certify the template by SDK inspection. Mark unsupported effort/parallel options as a coordinator compatibility blocker. Missing host access/source pins blocks completion of this task and dependent shape work; no engine run substitutes for it.
6. Before Task 2, append accepted NE T4 integration/file-release evidence and the exact coordinated artifact revision/schema to this report, once the external prerequisite is satisfied. Task 1's client inventory can finish independently; later schema work remains externally gated. Keep this private metadata report out of product commits; its creation is execution-time run state, not a product implementation unit.

**Verify:** Read-only evidence review: every client-emission row has an installed version, immutable source URL and reproducible synthetic case, or an explicit unverified blocker; no row substitutes current main for installed source. Both installed hosts are covered; inaccessible hosts remain pending. No config was changed and no engine/model/client inference session was started. Tasks 2–6 cannot run on an incomplete matrix or missing schema/file-release record. An explicitly unverified replay-use row is complete only as a documented refusal branch; it does not authorize implementation. Inventory resolves version/configuration questions without asking the owner to recall them.

### Task 2 — Install one NE-compatible explicit shape evidence schema

**Files:** `services/nvpair-proxy/local_qualification.go`, `services/nvpair-proxy/local_qualification_test.go`, `services/nvpair-proxy/testdata/local-qualification.json`

**Depends on:** 1

**Complexity:** judgment

**Do:**

1. External gate: accepted NE Task 4 identity/provenance/bounds join and released shared files; record schema integer from Task 1. Apply the I common-fixture reservation above before touching helpers. Re-read the integrated structs/parser/tests; preserve NE provenance and exact digest acceptance and generation invalidation.
2. Add `shapes` with the exact narrowed contract in Decisions: explicit required non-null boolean tools; explicit required non-null array structuredOutput, unique allowed names only; conditional required non-null reasoningReplay boolean only for inventory branch `used`. Reject absent object, null object/members, partial declaration, wrong types, duplicates, unknown shape fields, unknown/duplicate mode names and unsupported document versions. Validate required members in declared order, not map iteration, for stable diagnostics. No effort/parallel fields, no old-version compatibility path.
3. Update all local qualification test artifact constructors/literals via `rg -n 'qualificationDocumentVersion|localQualification\{|"version"|"qualifications"' services/nvpair-proxy/local_qualification_test.go services/nvpair-proxy/testdata/local-qualification.json`. Shared `testQualification` declares tools true, structuredOutput empty and conditional replay false; fixtures exercising JSON modes explicitly declare each one. Keep invalid registry tests intentionally invalid and update assertions to the actual accepted schema version. Reuse existing pointer helpers after discovery; do not add duplicate generic helpers. New Go files are not required in this task.
4. Add `TestQualificationShapesDocument` for the above valid/invalid document matrix, exact coordinated version, NE attestation acceptance and old-evidence rejection. Verify independent mode declarations do not implicitly grant the other mode. Existing accounting/request behavior is unchanged until Task 3, leaving schema tests green independently.

**Verify:** Safety-audited Mac selection `TESTS='^(TestQualificationDocumentRejects|TestQualificationShapesDocument|TestLoadQualificationRegistry|TestQualificationFixtureCases|TestQualificationContextBoundary|TestQualificationAppliesProfileDefaults)$'` through the sanitized command. Each named test exists and executes; no selected case fails/skips. `gofmt -l` on this task's Go files prints nothing; `go -C services/nvpair-proxy vet ./...` exits zero. Inspect the diff: NE's identity/provenance acceptance is preserved, and the fixture version matches the agreed revision. Requalification of existing fleet metadata is recorded as a rollout gate, not presumed complete.

### Task 3 — Validate and account each retained shape on the effective request

**Files:** `services/nvpair-proxy/local_qualification.go`, `services/nvpair-proxy/local_qualification_test.go`, `services/nvpair-proxy/testdata/local-qualification.json`

**Depends on:** 2

**Complexity:** judgment

**Do:**

1. Gate on effective fields after defaults merge, before any admission/accounting side effect. Preserve the raw request allowlist, duplicated-key rejection and NE envelope/output enforcement. Keep `chatRequestKeys` unchanged. Explicitly reject effort/parallel members as unsupported in both original and effective bodies, whatever their values; defaults must not bypass the boundary. Do not add a generic provider-options mechanism.
2. Gate tool definitions, tool_choice, assistant tool_calls and tool-role history on tools evidence. Present tools must be a non-null nonempty array; tool_choice requires nonempty tools, while qualified tool history does not require definitions in that same request. Preserve existing named function checks, optional parameters/object-when-present behavior, tool choice values, role restrictions and null-content-with-nonempty-tool-calls rule. Reject null/wrong-type gated members rather than silently forwarding them. Do not tighten unrelated scalar decoding.
3. Gate each `response_format` JSON mode on its exact structuredOutput entry; text and omission stay accepted. Preserve validation/accounting for both existing modes. An evidence entry for json_object does not authorize json_schema and vice versa. Reject malformed/null formats with the existing generic unsupported-shape error when decoding fails; field-only diagnostic promises apply to policy denials, not all nested parse failures.
4. Replay: when inventory is not `used`, reject any reasoning_content member explicitly, including null, and add no positive replay implementation. When inventory is `used`, retain raw member presence in each message, reject JSON null before decoding, then require a JSON string, assistant role and reasoningReplay true. Absent differs from null; valid empty string contributes zero bytes, nonempty string contributes its UTF-8 byte length. Never infer absence from a nil pointer. Keep other message checks; reasoning alone cannot authorize null assistant content. Refuse replay when template qualification is absent even though the SDK emits it.
5. Add `TestQualificationShapeMatrix`, with separate minimal triggering requests for each retained shape: tools definitions, tool_choice, assistant tool_calls, tool role, json_object, json_schema and conditional replay. For each, a qualified positive and undeclared negative must differ only in the evidence being tested. Include absence, null, wrong type, empty/nonempty values, role restrictions, malformed tool definitions/choices/schemas, missing-tools dependency and valid tool history without definitions. Do not toggle structured coverage against a body lacking response_format. Add unconditional negative tests for effort/parallel null and ordinary values. Existing text/stream/scalar paths stay covered.
6. Add `TestQualificationShapeDefaultsAndBounds`: same matrix via requestOptions injection, matching evidence/default equality, exact input+output=context and one-byte-over refusal, output ceiling/equal-alias limits, input byte lengths including UTF-8, compact JSON tool calls/schema/choice and conditional replay. Add defaults containing unsupported effort/parallel and undeclared tools/formats. Assert effective bytes/decoded fields remain unchanged except documented merged defaults. Derive expected bounds independently using the documented arithmetic, not the function under test; no fixed fixture/subtest total.

**Verify:** `TESTS='^(TestQualificationShapeMatrix|TestQualificationShapeDefaultsAndBounds|TestQualificationFixtureCases|TestQualificationContextBoundary|TestQualificationAppliesProfileDefaults|TestChatRequest|TestChatUnsupported|TestChatHistory|TestChatAccounting)$'` through the sanitized, helper-audited Mac command; all named cases execute and pass without skips. Each retained shape has its own positive/negative trigger. Replay branch `not used`/`unverified` has only refusal cases. `gofmt -l` on changed Go files prints nothing; vet exits zero. A Chat adapter regression is investigated at its actual cause; never assume every failure means the allowlist changed.

### Task 4 — Refuse installation of profiles with uncovered routing capabilities

**Files:** `services/nvpair-proxy/readiness_contract.go`, `services/nvpair-proxy/local_qualification_test.go`

**Depends on:** 3

**Complexity:** mechanical

**Do:**

1. In the integrated `qualifiedModelLocked` path add one coverage predicate after the existing identity/default/bound joins: tools requires tools evidence; structured requires both existing JSON mode entries; vision, embeddings and unknown capabilities fail; empty capabilities add no requirement. Keep NE provenance/envelope checks. Do not add a router/advertisement field or downgrade the profile silently.
2. Extend `TestStrictCapabilityDerivation` with tools missing/covered, structured missing/one-mode-only/both-covered, vision refused, embeddings refused and empty capabilities accepted under otherwise matching identity. For a previously installed profile, policy/evidence changes that remove required coverage withdraw readiness and fence an earlier qualified validation via existing generation semantics. Extend `TestStrictQualifyThenAdmitIsFencedByReinstallation` for this transition, without a second identity cache.
3. Update the derivation comment to include coverage among the conditions for installation. Preserve the existing model/default/digest mismatch tests.

**Verify:** `TESTS='^(TestStrictCapabilityDerivation|TestStrictQualifyThenAdmitIsFencedByReinstallation)$'` through the sanitized, helper-audited Mac command. Missing coverage produces no installed model/readiness entry; covered/empty profiles install only with all other joins satisfied; removed coverage invalidates earlier validation. No skips. Vet and gofmt checks pass.

### Task 5 — Prove synthetic OpenCode forwarding and refusal before effects

**Files:** `services/nvpair-proxy/testdata/opencode-shapes.json` (new), `services/nvpair-proxy/opencode_shapes_test.go` (new)

**Depends on:** 3, 4

**Complexity:** judgment

**Do:**

1. Recheck NE and I T3a/T4a/T4b file releases and current handler/helper ownership. Create synthetic fixture qualifications for engine `llamacpp`, model `fixture-model`, context 4096/output ceiling 1024, request/message/tool/schema overheads 8/4/16/16. Use the integrated NE synthetic attested identity/envelope constructors with matching model/content/launch/provenance/config references and accepted schema. Define plain/tools/per-mode qualifications; include replay evidence only in inventory branch `used`. Evidence text must say synthetic and qualifying no real model/template.
2. `testdata/opencode-shapes.json` contains named cases with qualification ID, synthetic body, expected acceptance/bounds or stable error category and an arithmetic note for each accepted bound. Add source pins/version context from Task 1 without private configuration. Include first turn with tools, tool-choice variants used by the client, assistant-null tool round trip with tool result, plain text, usage streaming shape when source/config warrants it, JSON modes already supported, image/provider-option refusals and unconditional effort/parallel refusal. Distinguish client-emission cases from broader existing-validator regression cases. Conditional replay cases include replay plus tool calls and later tool results, empty/nonempty strings and undeclared/wrong-role/null refusal. If the inventory does not prove replay, its fixtures only demonstrate refusal. Never copy real body text.
3. Add `TestOpenCodeShapeFixtures` to parse artifacts and call validateRequest with exactly matching profile/defaults/bounds. Require unique nonempty case names and the named required cases; derive coverage from names/declared shapes, never a hard-coded changing count. Compute expected accounting with explicit hand arithmetic and independently compacted synthetic JSON; do not bless the validator's returned number as its own oracle.
4. Add `TestStrictHTTPRefusesUnqualifiedShapesBeforeSideEffects`, table-driven on ingress and self using fresh explicitly installed synthetic llama.cpp fixtures. Match facade engine, body model, profile, qualification ref, NE identities/envelope, strict header and I run/source identity. For each unqualified definitions/history/tool-choice/JSON-mode/replay request and unsupported effort/parallel request, set the synthetic engine stopped/cold, enabled start-on-demand and no residency. Assert 400 `strict-qualification`, zero fake-engine hits, active slots, tickets and markers, and no admission/wake or admission/unload notifications. Cover each retained shape; valid bodies must reach shape refusal rather than fail an unrelated model/output/header check.
5. Add separate `TestStrictHTTPOpenCodeQualifiedForwarding` success subtests with a fresh harness each time. Install matching `fixture-model`/llamacpp evidence and policy at context 4096/output 1024; use max_tokens 16 in the body. Explicitly report that model loaded/resident, healthy and intended on, with matching NE observed envelope and I identity. Do not attempt to repair a cold rejection fixture merely by toggling health. A qualified tool round trip gets 200 and exactly one fake-engine hit; assert all synthetic tool IDs/arguments/results and defaulted fields received unchanged, plus expected response headers/body forwarded to the client. Use stream false with an ordinary synthetic Chat Completions JSON response for this case.
6. Add a streaming success subtest only for the inventory's streaming shape, using a loopback fake that emits valid Chat SSE deltas/usage/terminal data and clean EOF under the integrated SC/I parser contract. Assert the forwarded synthetic response and successful terminal handling; an ordinary JSON recording-engine response is never used as SSE proof. No native engine/process runs. New listeners explicitly bind `127.0.0.1:0`, check excluded port numbers and clean up. New files carry SPDX headers. Use focused descriptive subtests/assertions following repository test style.

**Verify:** `TESTS='^(TestOpenCodeShapeFixtures|TestStrictHTTPRefusesUnqualifiedShapesBeforeSideEffects|TestStrictHTTPOpenCodeQualifiedForwarding|TestStrictHTTPRejectsUnqualifiedRequestsBeforeSideEffects|TestStrictHTTPForwardsTheEffectiveBody)$'` through the sanitized, helper-audited Mac command. Every required named case runs and passes without skips, both paths prove no effects on rejection, and the separate warm harness proves exactly one forwarding call plus response fidelity. Check test log case names against fixture names, not a predicted count. `gofmt -l` on the new Go file prints nothing; vet and `node scripts/spdx-headers.mjs` exit zero.

### Task 6 — Document the joined contract and requalification boundary

**Files:** `docs/local-qualification.mdx`, `services/nvpair-proxy/spec.md`

**Depends on:** 2, 3, 4, 5

**Complexity:** mechanical

**Do:**

1. External documentation gate: accepted NE T4/T5 and I T6 text integrated/released; SR T6a follows B for these sections. Update the artifact example, field table and validation/accounting/install descriptions to the exact accepted schema integer and narrowed shape contract. Include one example for tools and independently declared JSON modes; replay appears only for inventory branch `used`. Explain old-evidence rejection and explicit requalification; never claim the fleet has no operator evidence.
2. Explain unsupported effort/parallel rejection, tools nonempty/tool_choice dependency, optional parameters, tool history without definitions, null/new-shape rejection, conditional assistant-only replay and byte accounting, exact structured mode evidence and capability fail-closed installation. Preserve generic malformed nested-shape diagnostics and existing non-shape behavior; do not promise field-specific messages for every decode error.
3. Add a short pinned OpenCode compatibility subsection using Task 1's immutable sources, installed versions and configuration conditions. Label unverified assumptions explicitly. Keep only nonprivate synthetic examples; never publish private host/config/paths or real bodies. No moving SDK-main table as durable truth.
4. Describe content/config/template identity as supplied by accepted NE T4, including attestation provenance and stale evidence invalidation. Do not leave a permanent statement that llama.cpp cannot qualify because list_models lacks digests. No Strata implementation/acceptance claims. Preserve accurate current activation gates; do not activate I/hybrid/public paths in prose.
5. Run service-contracts:check. If the integrated NE/I contract requires generated documentation, route regeneration via the reserved contract owner with `npm --prefix desktop run service-contracts:write`; B does not hand-edit it or absorb that lane. Eventual PR metadata must declare proxy binary release intent in `pair-release-intent:v1`; do not edit versions.json/CHANGELOG.md. No PR or commit is created for this amendment.

**Verify:** Read both files against the exact schema and named fixtures; all example fields/version values parse under that schema, unsupported fields are not presented as accepted, mode/replay conditions agree, and NE digest ownership is current. `node scripts/spdx-headers.mjs` and `npm --prefix desktop run service-contracts:check` exit zero. Inspect this task's diff, not a whole-tree two-file count; unrelated/integration changes remain untouched. Generated contract differences block release to the reserved owner rather than prove B introduced a new RPC.

## Verification (whole-plan)

- On the integrated Mac execution base, rerun the union of Tasks 2–5's anchored test selections through the sanitized command with `-race`; all named tests/cases must execute and pass, no unexplained skips. Re-audit helpers/env after each accepted NE/I join before running them. Vet, SPDX and service-contracts:check pass. No full proxy suite, engine process, protected port or `services/tests` execution is required for B proof.
- PC3 focused hermetic gate stays pending until the Windows preflight and coordinator isolation interval are satisfied, then runs the same selection without race under the fixed-path wrapper. No result on Mac establishes Windows/native/model acceptance.
- Task 1's read-only installed metadata and pinned synthetic conversion are implementation prerequisites. A later real OpenCode/llama.cpp acceptance is separately scheduled by the coordinator under exact NE-bound model/template evidence; no capture/run is a prerequisite for these implementation tests. A synthetic fixture never qualifies a real template by itself.
- Check the task graph with `bash ~/.claude/skills/execution-plan/scripts/audit-plan.sh .claude/plan-runs/pair-roadmap-20261019/packages/b-opencode-shapes.md`. Require `graph: ok`; a missing lifecycle section is acceptable for this repository package under this assignment. Review lexical fork candidates as objective conditions, not open design choices.

## Do not

- Do not implement or commit code during this amendment; edit this plan only. Do not write vault notes. Do not stage this untracked run directory without an explicit request.
- During execution, do not edit the broker, engine-manager, shared wire, desktop, chat adapter, services/tests, launch/residency owners or other packages' reserved files. Do not refactor beyond the tasks. Adapt B's fixtures after accepted joins, preserving I identity/cancellation/buffer ownership.
- Do not implement effort/parallel support, generic provider options, new structured modes, attachments, Strata or duplicate NE digest/identity logic. Do not strip undeclared members and forward them.
- Never hand-edit `desktop/docs/services-api.md`, `services/versions.json` or `CHANGELOG.md`. No real prompts, requests, responses, secrets, private paths, hostnames or user configuration in tests/logs/public docs. Synthetic fixture diagnostics only.
- Never start native engines/models, test installed endpoints, bind/contact protected ports or run services/tests from B. No Windows race run. Missing wrapper/rules/isolation evidence is a pending gate, not authorization to use temporary executable paths.

## If stuck

- Installed version/config/source cannot be pinned: mark inventory blocked with the exact missing metadata/access; keep replay refused and halt dependent shape work. Do not infer from latest upstream or require a model run.
- Inventory proves client effort/parallel/provider options outside scope: report the exact safe field/config fact to the coordinator; retain explicit refusal. A decision to change client configuration versus expand scope needs the owner only when inventory/source cannot settle the intended compatibility choice.
- NE T4 is absent, schema version conflicts or I files are reserved: wait for accepted integration/release. Do not use join10's old launch hash as template freshness evidence, fabricate attestation or overwrite shared seams. The coordinator resolves NE's deferred-Strata ordering.
- Replay use is proven but exact NE-bound llama.cpp template qualification is missing: keep that model's replay disabled. Synthetic tool-history cases can still prove validator mechanics, explicitly without real qualification. Schedule any model acceptance separately.
- A byte sum disagrees: recompute UTF-8 and compact JSON accounting independently, preserving existing formula/NE caps; fix wrong fixture arithmetic, not product accounting just to fit a number. An unrelated inherited helper safety failure leaves that verification pending; do not widen execution to full suites.
- A selected test disappears after I integration: locate its accepted successor at the shared owner and update B's test selection/fixture against the new contract. Do not recreate obsolete dispatch paths. A missing selected test is not a pass.
- After two failed correction attempts, stop and report the exact blocker and current plan/task state; preserve unrelated work.

## Owner questions

None requires an answer before the read-only inventory. Version, provider selection, replay configuration, SDK emission and existing evidence metadata are inventory questions, not requests for owner recollection. The adopted answers above remain defaults unless vetoed.

Only if inventory cannot answer them, escalate these concrete choices with its evidence: which installed client/provider configuration is the intended daily target when host configurations differ; whether to change a configuration that requires deferred/unsupported fields versus authorize a separately amended scope; and, if source/config plus existing qualification cannot establish replay use/template qualification, whether to keep the disabled default or schedule a separate exact-template acceptance. No digest-ownership, capability fail-closed or Strata question remains open.

## Amendment 2026-10-21

R1 is applied through the external NE Task 4 identity/schema/file-release prerequisite and NE/I/SR documentation reservations. R2 narrows implementation behind the installed-client, source-pinned inventory. R3 replaces the combined negative test with per-shape positives/refusals, existing-mode positives, defaults/boundary cases and forwarding assertions. R4 uses separate, matching warm llama.cpp success fixtures and genuine synthetic SSE for streaming proof. R5 preserves raw presence and refuses null for gated members. R6 defers effort entirely and gives existing structured modes separate evidence. R7 replaces full-suite/live-capture commands with audited hermetic selections and conditional Windows preflight. R8 uses numeric dependencies and expanded repository paths; the audit graph is clean.

Ownership/factual and LOW corrections remove claims of no deployed evidence and previously passing suites, preserve optional tool parameters and actual tier/null semantics, coordinate one incompatible revision with NE, avoid unused shape booleans and map-ordered diagnostics, derive fixture coverage without fixed counts, retain generic decode errors and pin durable client claims. The coordinator-adopted replay/refusal, fail-closed installation, NE digest ownership and deferred-Strata decisions are recorded above; owner escalation is limited to unresolved post-inventory compatibility choices. This amendment changes this plan only.
