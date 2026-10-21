<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Review: delivery audit of landed PA, J, B and XA work (slug: audit-pa-j-b)

Verdict: `RED 1 UNVERIFIED 4 LOW 7`

## Base inspected

- Product base: `fork/groundwork` at **`8a63866c1cbadd70f40732c84b04efb72d6e2fd4`**. The common brief
  names `210babf`; the branch had moved by two first-parent merges (`8739691` NE Task 5 Part B,
  `8a63866` I public-lifecycle test fix). Neither touches the files audited here; all `path:line`
  citations below are at `8a63866`.
- Landed units mapped from `git log --first-parent`:
  - PA T4 `b00f597` (scanner availability tests + README), PA T5 `26606f5` (broker availability test
    + README), PA T6 prerequisite `fe68c81` (scanner `--mdns` flag), PA T6 `210babf`
    (`services/tests/paused_advertisement_test.go`, `broker_test.go`).
  - J T1 `4bb1d8d` (`workload_projection.go`, `client_requests.go`, new test).
  - B T7 `0468344` (`local_qualification.go` + test).
  - XA `721c9ef` (Tasks 1-4) and `d5a636a` (Tasks 5-10 plus fixture/port repairs).
- PA Tasks 1-3 and 7 are **not** landed: `services/nvpair-proxy` contains no `PausedBy`
  (`rg PausedBy services/nvpair-proxy` is empty), `resolveCandidates` has no pause exclusion
  (`services/nvpair-proxy/proxy.go:631-735`) and `subscribedToNode` still drops the pause hint
  (`proxy.go:1356-1361` copies roles, served models, revision and strict capability only). There is no
  `docs/node-availability.mdx`.

## Toolchain and checks run

- Container Go is 1.24.7; the modules require `go 1.25.0`. With `GOTOOLCHAIN=auto` the 1.25.0
  toolchain downloaded, so vet and unit tests did run. Results (all VERIFIED by execution):
  - `nvpair-proxy`: `go vet ./...` clean; `TestQualificationStoreNoOp`, `TestQualificationFixtureCases`,
    `TestQualificationContextBoundary`, `TestQualificationAppliesProfileDefaults` PASS, no skips
    (run with the CORS/proxy environment variables unset, as the B plan prescribes).
  - `nvpair-node-scanner`: vet clean; `^TestPauseHint` (three tests) PASS.
  - `nvpair-ui-broker`: vet clean; `TestPauseProjectionCarriesProvenanceAndRevision` (five subtests),
    `TestNodeProjectionPublishedOnPersistAvailabilityAndReady`, `TestPauseProvenanceIsNeverDowngraded`,
    `TestConfigurationEditsStayAllowedWhileLocallyPaused`, `TestWorkloadPrivacyProjection*`,
    `TestRouteProxyWorkload*`, `Test.*Replay.*`, `Test.*Rehydrat.*` PASS.
  - `services/tests`: `go vet ./...` clean; `go test -list` (TestMain builds the binaries, no case ran)
    lists `TestPAPausedAdvertisementConvergesWithoutMDNSChange`, all seven
    `TestXARemoteConfiguration*`, both `TestEngineSwitchProcessPinned*`,
    `TestRemoteManagementPersistsAcrossSeparateProcessRestarts` and four `TestFixtureLoopback*`.
  - `gofmt -l` on every landed Go file prints nothing; `node scripts/spdx-headers.mjs` reports 0 missing.
- Not run (per brief): any `services/tests` case, engines, models, desktop builds. No `-race`.

## Obligation checklist

Status key: FULFILLED (code/tests read and, where possible, executed), FAILED (plan obligation not met
at this SHA), UNVERIFIED (only provable on a real machine or evidence absent).

### PA Task 4 - scanner tests prove retention and transport refusal

| Obligation (plan) | Status | Evidence |
|---|---|---|
| `TestPauseHintFollowsTheAdvertisement`: one explicit sequence available 1 -> local pause 2 -> identical no change -> acknowledged failed fetch retains pause 2 and cache -> resume 3 clears -> peer pause 4 | FULFILLED | `services/nvpair-node-scanner/availability_test.go:120-152`; every phase asserts request count `before+1` (acknowledged fetch) and compares directory and `lastPolicy` cache structurally (`:102-118`). Failed-fetch retention is the product behaviour at `model_enrichment.go:329-338` (no cache write on failure). |
| `TestPauseHintEmitsOneNodeUpdated`: captured codec frames, exactly one `node-updated` for pause-only change, none for identical, one for resume | FULFILLED | `availability_test.go:185-224`; synchronized writer (`:154-183`), decode errors checked. |
| `TestPauseHintIsNeverFetchedWithoutAPin`: counting/refusing transport on `d.modelsHTTP`; loopback with non-self UUID selects plaintext; unpinned non-loopback refuses in both discovery and periodic modes with zero plaintext calls and no peer-client cache; positive control increments once; admitted pin selects HTTPS | FULFILLED | `availability_test.go:226-281`; matches `model_enrichment.go:278-284` (loopback -> plain client regardless of UUID; otherwise `peerServiceClient`, no fallback). |
| Checked ephemeral loopback port, protected ports rejected | FULFILLED | `availability_test.go:48-67` rejects 8080, 11434, 1234, 14318-14323. |
| README: policy-only refresh, nominal cadence, success requirement, last-good retention, pinned LAN / loopback exception, no mDNS availability, no one-sweep promise | FULFILLED | `services/nvpair-node-scanner/README.md:45-64` (section "Policy advertisement refresh"); wording matches code. |

### PA Task 5 - broker evidence distinguishes persisted advertisement from pending veto

| Obligation | Status | Evidence |
|---|---|---|
| `TestPauseProjectionCarriesProvenanceAndRevision` with independent fixtures for local pause/resume, peer pause/resume; `setAvailability` result and error checked; wait for the exact saved advertisement (not `before+1`); matched advertisement equals stored policy | FULFILLED | `services/nvpair-ui-broker/availability_test.go:38-104`; `waitFor` predicate compares full `Advertisement` (`:76-87`), roles unchanged (`:73-75`). |
| Peer resume veto of a local pause: complete pause, drain via request/reply barrier on the fake engine reader, baseline after barrier, typed `localVetoError`, unchanged stored policy, second barrier, no new projection | FULFILLED | `availability_test.go:106-124`; barrier helper `:19-36`. Product path: `availability.go:59-63` returns `&localVetoError{}` before any transition (VERIFIED: "local pause beats remote resume"). |
| README Node-projection paragraph distinguishes pending veto from adopted saved policy | FULFILLED (text present) | `services/nvpair-ui-broker/README.md:505-518`. Product: `availability.go:88-93` publishes the pending projection before persistence. |
| README: "remote routers skip advertised paused peers for inference, subject to the exceptions and advisory freshness in Task 3" | **FAILED** (see RED-1) | README `:510-512` describes PA Task 2/3 semantics that are not in the proxy at this SHA. |

### PA Task 6 - PC3 process acceptance

| Obligation | Status | Evidence |
|---|---|---|
| `availableNode` gains `Policy *nodepolicy.Advertisement` only | FULFILLED | `services/tests/broker_test.go:46-58`; the merge also adds a guarded `NVPAIR_TEST_PA_SCANNER` branch in `runSubscriptionScanner` (`:121-124`), inside the declared file. |
| Isolation seam: production scanner with discovery isolated, real refresh path retained | FULFILLED by `--mdns=false` | `fe68c81` adds the flag (`services/nvpair-node-scanner/main.go:28`, `daemon.go:231,284-294,317-325`); `publishSelf` and the 15 s refresh loop still run without a browser. The fixture wrapper appends `--mdns=false` (`paused_advertisement_test.go:57`). The README for the flag states it is not general network isolation (`README.md:83-88`). |
| `pausedAdvertisementPlatform` skips all non-Windows; Windows requires `GOFLAGS` `nvpair-test-exec` and absolute `NVPAIR_TEST_BIN_DIR` | FULFILLED | `paused_advertisement_test.go:36-44`. |
| Sequence: available 5 -> pause 6 (same inventory, fixed descriptor) -> HTTP 500 acknowledged after switch, pause 6 retained -> resume 7 clears only after success; `Policy != nil` before deref; bounded deadlines; phase logs; owned PIDs | FULFILLED (source) | `:520-578`; the resume response is gated behind a barrier so the retained pause is read from the broker directory after the acknowledged 500 and before the next successful fetch (`:556-574`), which satisfies "no stale window" as far as the scanner-to-broker hop is concerned. |
| All listeners loopback on checked ports, six broker service overrides | FULFILLED | `reserveFixtureServicePorts(t).args()` (`:305,417`), `listenFixtureLoopback` for the inventory server (`:346`). The advertised peer address is a TEST-NET documentation address, probed once and required to fail promptly (`:306-332`). |
| Comment that fixture proves propagation, not authentication | FULFILLED, stronger than planned | The landed fixture serves the inventory over the pinned cluster TLS config and verifies the client pin (`:347-349,394`) through a private reverse stub (`:161-265`); comments say outer reverse session is a stub (`:157-160`). |
| Actual PC3 execution | UNVERIFIED (see UNV-1) | No run record in the repository. |

### J Task 1 - method-aware workload privacy projection

| Obligation | Status | Evidence |
|---|---|---|
| `projectWorkloadFrame(method, params)` returns projection + fixed category/count; lifecycle methods require object `workloadInfo`; outer allowlist only `workloadInfo`; nested allowlist exactly the 14 baseline fields | FULFILLED | `services/nvpair-ui-broker/workload_projection.go:22-99`, allowlist `:101-107`. Compared with the proxy producer (`services/nvpair-proxy/proxy.go:179-208`) and the manager (`services/nvpair-workload-manager/workload.go:69-79`): no legitimately added identity field is missing. |
| `workloads:remove` top-level `workloadId`/`originatedFrom` only; no lifecycle envelope | FULFILLED | `:36-52,61-64`; manager removal contract is `(workloadId, originatedFrom)` (`nvpair-workload-manager/manager.go:381`). |
| Reject null/wrong shapes/types, recursive duplicate keys (including unknown subtrees and escaped names), trailing JSON, invalid identities; malformed must not reach `stampNodeID` fallback | FULFILLED | token walk `:122-153`, type table `:109-120`, early returns at `routeProxyWorkload` `:335-338` before `stampNodeID` `:340`. Test matrix `workload_projection_test.go:107-183`. |
| Project before store/scheduler/client/manager effects, including `activeLocalReplayFrames` and scheduler replay, and the `workloads:get-initial` snapshot | FULFILLED | `:237-240`, `:268`, `:478-481`, `client_requests.go:272`; retirement projects persisted removals `:530-537`. |
| Preserve legitimate identity/cancellation fields; no Workload protocol expansion | FULFILLED | nullable fields and `scheduledOn`/`runId` absence survive (`workload_projection_test.go:76-96`); cancel frames bypass the projection by method (`:425-439`). |
| Tests for every lifecycle method, removal, secret-bearing synthetic keys, nulls, identity, malformed, restart/rehydration; zero effects on malformed; logs carry no keys/values/raw errors | FULFILLED | `TestWorkloadPrivacyProjection*` (`workload_projection_test.go:57-409`), executed PASS. |
| Diagnostics "fixed category and count" surfaced | PARTIAL (LOW-3) | Only `retireWorkloadHistory` logs the category (`:534`); every other caller discards it (`:162,237,335,478`). |

### B Task 7 - `store:false` no-op

| Obligation | Status | Evidence |
|---|---|---|
| Raw-key validation recognizes `store` without adding it to `chatRequestKeys`; only JSON `false` accepted; `true` refused with a storage-specific message; null/string/number/array/object refused | FULFILLED | `services/nvpair-proxy/local_qualification.go:326-329,401-409`; `chatRequestKeys` unchanged (`:290`). |
| Unique-key/object checks precede normalization so duplicate `store` cannot vanish | FULFILLED | duplicate detection precedes the key loop (`validateRequest` `:299-322`); tests `duplicate raw/escaped/default store refused` PASS. |
| Merge precedence preserved: validate raw, merge defaults, validate effective `store` with the same rule, strip before scalar validation/accounting and before returning the forwarded bytes; defaults-injected true/null/wrong type fail; absent path returns identical bytes | FULFILLED | strip at `:362-365` after `mergeDefaults`, before `newBodyWorkspace`; `stripQualificationStore` `:413-453` re-validates (`:419-421`) and returns the original slice when no member was found (`:433-435`). Tests `absent preserves exact bytes`/`merged bytes` and bound equality to an omitted-store control PASS. |
| Before side effects, typed error | FULFILLED | `validateRequest` runs inside `qualifyStrict` (`readiness_contract.go:257-267`) which `serveStrictLocal` calls at `ingress.go:367` and maps `*strictQualificationError` to HTTP 400 `strict-qualification` (`ingress.go:372-374`) before `AdmitStrict` (`ingress.go:402`). `serveStrictLocal` serves both the local facade strict branch (`http_handler.go:605-620`) and cluster ingress. |
| `TestQualificationStoreNoOp` covers the listed cases | FULFILLED | `local_qualification_test.go:371-504`; also covers buffer-lease accounting of the stripped copy. |
| Diff confined to the two Files | FULFILLED | `git diff --stat 0468344^1 0468344`. |

### XA Tasks 1-10

| Obligation | Status | Evidence |
|---|---|---|
| T1 `xaPlatform` copies `strataContractPlatform` (skip darwin, Windows wrapper fatal) and `TestXARemoteConfigurationStatus` with four subtests | FULFILLED | `services/tests/remote_configuration_test.go:24-34,180-247`. |
| T2 conflict: HTTP 409 `policy_conflict` + exact text, origin `-32010`, unchanged bytes, positive control, stale retry | FULFILLED | `:249-299`; text produced by the relay (verified exact strings exist in product: `nvpair-shared/nodepolicy/projection.go:174`, `policy.go:562`). |
| T3 veto: relayed resume `-32011` with exact text; remote start vetoed while paused; local resume then local Off vetoes remote start; stored intent `local` | FULFILLED | `:301-342`; veto text `nvpair-engine-manager/nodeprojection.go:95,98`; `codeLocalVeto = -32011` (`remoteclient.go:41`). |
| T4 roles: allowed peer sets, target advertises in `engine:models`, denied peer 409 management-denied, tiers protected | FULFILLED | `:510-563`; `xaPeerDraft` detached copy `:497-508`. |
| T5 serving: explicit empty / subset+resident / advertised / resident-outside-served refused / denied peer / absent restores unrestricted; proxy facade enforcement positive control, empty, other, match, no llamacpp start | FULFILLED (source) | `:565-839`; admission order `modelServed` first (`nvpair-proxy/admission.go:385`), body/header strings at `admission.go:893`. |
| T6 extract `pinnedSwitchFixture`, `remoteStatus`, `request` | FULFILLED | `services/tests/engine_switch_test.go:783-838`; the fixture now uses `task2WithoutScanner` (`:797`), i.e. no production scanner or mDNS responder (`reverse_peer_control_task2_test.go:53-61,301-325`). |
| T7 switch: stale conflict + rejected receipt, local pause veto `-32011`, veto replay, resume + fresh completion, identical retry (journal answer, `ec` count +1), reused ID `-32602` with exact text, journal holds exactly three IDs | FULFILLED (source) | `:344-480`; strings at `nvpair-ui-broker/engineswitch.go:491,520,546,556`. |
| T8 `newRemoteServiceFixture` extraction, restart test updated only as described | FULFILLED | `service_harness_test.go:155-225`, `remote_management_test.go:1110-1123`. |
| T9 service: grants, local policy edit (activation `pending`), peers see policy (denied 409), engine intent + allowed-peer start vetoed `local_veto`, clean stop | FULFILLED (source) | `:36-178`. |
| T10 README paragraph, seven selectors | FULFILLED | `services/readme.md:262-291`; `grep -c '^func TestXARemoteConfiguration'` = 7. |
| Fixture gate: every XA listener rejects protected ports | FULFILLED | `listenFixtureLoopback`/`allocateFixtureLoopback`/`newFixtureHTTPServer` (`fixture_ports_test.go:40-71`) now back `freePort` (`tls_test.go:158-167`), `temporaryClusterPort` (`cluster_outbound_pairing_test.go:33-43`), the switch control server and the task2 fake/node-info servers; unit tests `TestFixtureLoopback*` cover rejection paths. |
| Target-local pairing permission and local veto respected | FULFILLED (source) | denied peer: `managementHTTP` 409 bodies for policy set (`:548-553,672-680`), service denied set (`:120-127`); grant toggling through `cluster:set-remote-management` (`:58-91`); vetoes above. The matrix does not include a denied peer attempting `engine:remote-switch`/`engine:remote-start` (plan did not require it; covered by existing `TestRemoteManagement*` only if those cases exist there, not re-audited). |
| Actual PC3 execution | UNVERIFIED (UNV-2) | No run record in the repository. |

## Findings

### RED-1 - Broker README advertises router-side pause exclusion that the proxy does not implement

- Claimed invariant (PA plan, Task 5): "README's Node projection paragraph distinguishes pending veto
  from adopted saved policy; remote routers skip advertised paused peers for inference, subject to the
  exceptions and advisory freshness in Task 3." Handoff risk: "PA paused advertisement must never let a
  peer route to a paused node and must clear on resume without a stale window."
- Code path: `services/nvpair-ui-broker/README.md:510-512` ("Remote routers can use advertised pauses to
  skip remote peers for inference; self/dormant-self admission, manual nodes without hints, model lists
  and pass-through retain their own behavior") and `:515-518` (snapshot semantics, re-entry on refresh).
  At `8a63866` the proxy has no `Node.PausedBy` (`services/nvpair-proxy/discovery.go:34-80`),
  `subscribedToNode` discards `n.Policy.PausedBy` (`proxy.go:1356-1361`), and `resolveCandidates`
  classifies self / pinned peer / manual / unpinned and deduplicates by host with no availability check
  (`proxy.go:671-735`).
- Observed result: a router that has already observed `pausedBy` for a pinned peer still selects it as a
  candidate and dials it. The only guard is the destination: the paused node's own admission returns 503
  `X-PAIR-Admission: paused` (`admission.go:387-393`), which the origin treats as retryable and fails
  over (`http_handler.go:858-870,1588-1596`). So no inference runs on a paused node, but the "zero
  dispatch attempts" property the README describes is false, and with a single eligible peer the user
  sees the 503 pass through rather than a 502/404 empty-candidate response.
- Affected state: documentation only; router behaviour is unchanged from before PA. The README paragraph
  was written to Task 2/3 semantics but merged before Task 2.
- Disposition: amend plan/README now. Either reword `README.md:510-518` to "will skip once PA Task 2
  lands" / move the sentence to PA Task 3's deliverable, or land PA Tasks 1-2 (the I owner cutover gate
  in the plan's "External reservations" must be recorded first). Also correct the stale "TCP-probed"
  comment (`proxy.go:677`) when Task 2 edits that owner, as the plan already requires.

### UNV-1 - PA Task 6 process acceptance has no execution evidence

- Claimed invariant: "PC3 only, after all gates: `go -C services/tests test -count=1 -v -run
  '^TestPAPausedAdvertisementConvergesWithoutMDNSChange$' .` Require PASS, no skip, all phase logs
  including acknowledged failure, fixed TXT and successful owned-process cleanup."
- Probe: the case skips on every non-Windows OS (`paused_advertisement_test.go:38-40`), so this
  container can only vet/compile/list it (done, listed). The source satisfies every structural
  obligation in the checklist above.
- Observed result: no PC3 run log, firewall before/after record or survivor check is in the repository
  or the plan branches read here.
- Affected state: PA Task 6 completion claim; the plan's mandatory PC3 gate.
- Disposition: owner question / defer: record the PC3 run (names executed, skips, `pa6-propagation`
  phase log lines, `fixture_worker_exited`, cleanup) in the coordinator's run state before calling Task
  6 landed. Note the fixture uses the production scanner's real refresh loop, so the observed cadence
  will be the 15 s sweep plus batching; the 30 s setup and 90 s whole-case bounds are harness limits.

### UNV-2 - XA suite has no execution evidence and one residual real-mDNS neighbour

- Claimed invariant: "PC3 (Windows, wrapper configured, PAIR stopped, one `go test` at a time) ... Pass:
  both commands print `ok`, all selected named subtests execute with no skips, cleanup logs show no
  owned survivors."
- Probe: all seven cases list; `xaPlatform` skips only darwin and gates Windows (`:24-34`), so Linux
  would run them unguarded. The fixtures XA uses no longer start a production scanner or mDNS responder
  (`newPinnedSwitchFixture` -> `task2WithoutScanner`; joint, management and switch fixtures use the
  empty-directory subscription scanner: `policy_relay_joint_test.go:318-326`,
  `engine_switch_test.go:157-161`). `TestModelsRefreshConvergesWithoutMDNSChange` (`broker_test.go:70`)
  still launches the production scanner with mDNS; it is not an XA selector.
- Observed result: no PC3 run record; isolation of the XA selectors is satisfied by source reading.
- Affected state: XA "complete" status in the common brief.
- Disposition: owner question: attach the PC3 `-v` log for the two whole-plan selectors (XA plan
  "Verification (whole-plan)") to the run state. Until then XA is complete as code, unverified as
  acceptance.

### UNV-3 - J Task 1 silently drops a lifecycle frame whose identity falls outside `strictID`

- Claimed invariant (J plan, Task 1): "Preserve all legitimate identity/cancellation fields ... Existing
  legitimate error/model strings are not made confidential by an allowlist."
- Code path: identity-kind fields (`id`, `engine`, `runId`, `originatedFrom`, `scheduledOn`) must match
  `^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$` (`workload_projection.go:117-119` ->
  `nvpair-shared/nodepolicy/routing_wire.go:33-42`). `scheduledOn` is set from the candidate id
  (`nvpair-proxy/http_handler.go:1737`), which for a manual node is the caller-supplied `id` of
  `node/add-manual`, validated only for non-emptiness (`nvpair-proxy/proxy.go:1586-1594`). The desktop
  bridge passes the host UUID (`desktop/src/electron/service-bridge/modular-supervisor.ts:2638-2643`)
  and the manual-nodes worker keys on `HostUUID`, so production ids are in range; a user-typed id with
  a space, slash or non-ASCII character is not.
- Observed result (from code): such a `workload:started` frame is rejected as `malformed` and dropped
  before store/client/scheduler/manager with no log at all (`:335-338`); the workload never appears in
  the desktop or TUI, and the scheduler never sees it. Not reproduced against a real client because
  no production path was found that emits such an id.
- Affected state: workload visibility for manual nodes with free-form ids; not inference.
- Disposition: amend plan or repair now: either validate `node/add-manual` ids against the same
  pattern at the proxy (or broker) so refusal is explicit, or relax `scheduledOn` to a bounded string
  kind. At minimum add the fixed-category debug log (see LOW-3) so a drop is observable.

### UNV-4 - `store:false` outside strict qualification is untouched; public adapter still refuses it

- Claimed invariant (B plan): "Ordinary use of this generic OpenCode path against a PAIR local target
  therefore fails even for plain text unless the user sets `compatibility.supportsStore:false` ...
  No client configuration workaround is required after this fix."
- Code path: the fix lives in `validateRequest`, reached only through `qualifyStrict`
  (`readiness_contract.go:257-267`) from `serveStrictLocal` (`ingress.go:367`). A request without the
  strict envelope goes through the ordinary facade/relay path, which does not inspect top-level keys,
  so `store:false` is forwarded verbatim to the engine; the performance sampler treats `store` as an
  unknown option (`performance.go:378-383,551-556`) and declines to sample. The public-account adapter
  keeps the shared key list without `store` (`chat_adapter.go:492`), as the plan directs.
- Observed result: for the daily strict llama.cpp target the plan's claim holds (verified by unit
  tests). Whether llama.cpp's router mode tolerates an unknown `store` member on the non-strict path,
  and whether OpenCode requests reach the strict path on the real hosts, can only be shown on MSI/PC3.
- Affected state: compatibility claim scope; telemetry "unknown" classification on non-strict requests.
- Disposition: owner question: confirm the daily OpenCode path is strict-only (then the claim is
  complete) and whether public deployments should accept `store:false` once I public dispatch activates
  (B Task 6 docs line should state the public-path refusal explicitly).

### LOW-1 - Stale "TCP-probed" comment on `targetURL`

`nvpair-proxy/proxy.go:676-678` still says the address is "cached, TCP-probed"; `targetURL` uses
`targets.Prefer` (`facade.go:409`). PA Task 2 owns the correction; no behaviour effect.

### LOW-2 - Broker README workload text predates J Task 1

`nvpair-ui-broker/README.md:292` says the broker forwards "the original lifecycle frame to the
manager"; it now forwards the projected frame (`workload_projection.go:335-340`).
`README.md:1079` calls `workloads:get-initial` a "full-fidelity" snapshot; it is the projected snapshot
(`client_requests.go:272`, comment at `workload_projection.go:570-572` already updated). Persisted
unknown members are stripped on the way out, which the README should say. Amend docs.

### LOW-3 - Projection diagnostics are discarded on the hot paths

`projectWorkloadFrame` returns `(category, count)` but `routeProxyWorkload` (`:335`),
`emitWorkloadEventProvenance` (`:478`), `activeLocalReplayFrames` (`:237`) and
`projectedWorkloadInfo` (`:162`) ignore them; only retirement logs (`:534`). The plan asked for fixed
category/count diagnostics; a malformed or stripped frame is otherwise invisible. Add the same
`slog.Debug("...", "category", c, "count", n)` at those sites (no keys/values).

### LOW-4 - XA README omits that Linux runs the cases unguarded

`services/readme.md:270` ("It skips on macOS and, on Windows, requires the fixed-path wrapper")
matches `xaPlatform`, but a Linux checkout runs all seven cases with real binaries. The XA plan chose
this ("Linux runs unguarded"); the README should say so because the fleet rule "process cases are PC3
only" is otherwise enforced by convention alone.

### LOW-5 - PA Task 5 barrier is answered by the fake worker's catch-all

`pa:projection-barrier` (`availability_test.go:23`) is a synthetic method; the fake engine answers
every unknown method `{"ok":true}` (`policy_test.go:280-291`). The barrier is sound only because the
fake worker serves one connection in order; the test comment says so. No product seam exists, which the
plan accepted. Nothing to change; recorded so a future fake-worker refactor keeps ordered serving.

### LOW-6 - PA Task 6 comment claims a plaintext fixture; the landed fixture is pinned

The plan asked to "comment explicitly that this loopback plaintext fixture proves propagation, not
pinned authentication". The fixture actually serves the inventory over the peer mesh's server TLS and
verifies the caller's pin (`paused_advertisement_test.go:347-349,394`), with a private reverse stub.
Comments at `:46-48` (scanner module) and `:157-160` are accurate for what each proves; the plan text
is the stale side. Amend the plan's Task 6 wording when it is next touched.

### LOW-7 - `services/nvpair-proxy/spec.md:1572` lists `store` as rejected

That paragraph describes the public-account (`chatQualification`) path, where `store` is still
rejected (`chat_adapter.go:492`), so it is not wrong; but with B Task 7 landed and B Task 6 (local
qualification docs) pending, `docs/local-qualification.mdx` says nothing about `store` and a reader of
`spec.md` can conclude PAIR rejects `store` everywhere. B Task 6 should add the planned contract line
and scope the spec sentence to the public path.

## Documentation contradictions (summary)

| Document | Line(s) | Says | Code |
|---|---|---|---|
| `services/nvpair-ui-broker/README.md` | 510-518 | remote routers skip advertised paused peers; snapshot/re-entry semantics | no `PausedBy` in proxy; no exclusion (RED-1) |
| `services/nvpair-ui-broker/README.md` | 292 | forwards the original lifecycle frame to the manager | forwards the projected frame (LOW-2) |
| `services/nvpair-ui-broker/README.md` | 1079 | `workloads:get-initial` is full-fidelity | projected, unknown members stripped (LOW-2) |
| `services/nvpair-proxy/proxy.go` | 676-678 | TCP-probed address choice | `targets.Prefer` (LOW-1) |
| `services/readme.md` | 270 | skips on macOS, wrapper on Windows | Linux unguarded (LOW-4) |
| `services/nvpair-proxy/spec.md` | 1572 | `store` rejected | rejected on public path only; accepted-and-stripped `false` on strict local path (LOW-7) |

No contradiction found in `services/nvpair-node-scanner/README.md:45-64,83-88` (matches
`model_enrichment.go` and `daemon.go`), `services/readme.md` XA selectors (match the seven functions),
or `desktop/docs/services-api.md` (no RPC shape changed by these units; J Task 1 strips fields rather
than adding any, and `desktop/src/shared/types/workloads.ts` lists only allowlisted members). No
desktop or TUI copy was added by these units; the TUI already shows `pausedBy`
(`services/nvpair-tui/ui/availability.go:31-95`), unchanged.

## Real-machine acceptance checklist (what code reading cannot show)

Run on PC3 only, serialized, wrapper configured, no `-race`; MSI for unit race evidence only.

1. PA T6: `go test -v -count=1 -run '^TestPAPausedAdvertisementConvergesWithoutMDNSChange$' .` in
   `services/tests`. Require PASS, no skip, the log lines `pa6-propagation available5`, `pause6`,
   `acknowledged500 ... retained=pause6`, `resume7`, `fixture_worker_started`/`fixture_worker_exited`,
   and `owned_processes` showing the broker, adapter and real scanner only. Record the TEST-NET probe
   outcome (refused or timeout under 3 s) because a LAN that routes documentation addresses would
   change scanner timing.
2. PA unit race (MSI): `go -C services/nvpair-node-scanner test -race -count=1 -run '^TestPauseHint' .`
   and the broker selector from PA Task 5 Verify.
3. XA: the two whole-plan selectors from the XA plan, `-v`, with before/after firewall rule list and
   survivor scan; require seven `ok` areas, no `SKIP`, and the switch journal assertion in
   `TestXARemoteConfigurationSwitch` passing (three request IDs).
4. XA residual: run `TestEngineSwitchProcessPinned(Direct|Reverse)` and `TestReversePeerControlTask2`
   once after the `task2WithoutScanner` change to prove the scanner-bearing default path is unchanged.
5. B T7 on a real OpenCode client (MSI, strict profile installed, llama.cpp On, owner-authorized):
   send one plain-text turn from the unmodified generic OpenCode provider (no `supportsStore` setting)
   and confirm 200 with the engine receiving no `store` member (engine request log or a loopback
   capture); then confirm a hand-crafted `store:true` returns 400 `strict-qualification` with the
   storage message and zero engine hits. Also send the same turn to a non-strict facade route and
   record whether llama.cpp tolerates the forwarded `store` member (UNV-4).
6. PA end-to-end (after PA Task 2 lands; not before): pause MSI locally, route from PC3; expect the
   router to omit MSI with zero attempts. Until Task 2 lands, the observable behaviour is one 503
   `paused` attempt followed by failover, which is what RED-1 documents.
7. J T1 on a real cluster: add a manual node whose id contains a space on the TUI/RPC path, route one
   request to it and confirm whether the workload card appears (UNV-3). If it does not, the silent
   drop is reproduced.

## Owner challenges

- The broker README paragraph (RED-1) is the only landed PA text that promises router behaviour.
  Rather than wait for PA Task 2, consider having PA Task 3 own all routing-eligibility prose and
  keep Task 5's README strictly to broker projection semantics; that keeps every doc sentence tied to
  the task that lands its code.
- J Task 1's identity regex is stricter than anything the product enforces on manual node ids.
  Enforcing it at the ingestion boundary without a matching producer-side rule turns a cosmetic id into
  a silent loss of workload visibility; the plan's "strip well-formed unknown fields, reject malformed
  envelopes" decision did not anticipate legitimate free-form ids. Either validate ids where they are
  created or accept bounded strings for `scheduledOn`.

## Questions for the owner

1. Is there a PC3 run record for PA Task 6 and the XA whole-plan selectors that should be cited by this
   audit (UNV-1, UNV-2)? None is reachable from the public branches.
2. Should the broker README paragraph be reworded now (doc-only change) or left until PA Task 2 lands
   (RED-1)?
3. Is the daily OpenCode path strict-only on MSI and PC3, so B Task 7's compatibility claim is complete
   as landed (UNV-4)?
4. Are free-form manual node ids (typed by a user, not host UUIDs) a supported input anywhere in the
   fleet (UNV-3)?
