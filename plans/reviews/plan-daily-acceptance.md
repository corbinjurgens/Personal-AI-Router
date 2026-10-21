<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution plan: daily local/peer milestone acceptance on the real fleet

Kind: new execution plan (investigation output; no product change, no pull request). Most tasks are real-machine procedures. Every `path:line` is at `origin/fork/groundwork` = `8a63866c1cbadd70f40732c84b04efb72d6e2fd4`.

## Goal

The owner can show, on MSI and PC3 (Windows, llama.cpp router mode with one approved per-model preset each) with the Mac as an origin-only host and OpenCode as the client, that every clause of the daily local/peer milestone holds or is honestly reported as not yet landed:

1. one approved serving configuration per serving host, and a named origin-only host;
2. tier (weak/medium) and peer routing with real models, roles consumed from the declaring node;
3. desktop remote configuration and engine/model switching with target-local "my own device" permission, locked-peer refusal and local veto;
4. a headless `nvpair-service` on Windows that restarts its broker and survives its own restart with attached clients reconnecting;
5. completed/failed/cancelled outcomes that are honest, including interrupted streams;
6. telemetry that is accurate or reports unknown;
7. the OpenCode request shapes actually used (tools, `store:false`).

Each check below is marked **READY-NOW** (provable at `8a63866`) or **WAITS-FOR <package task>** (the product behaviour the milestone asks for is not landed; the procedure records the current behaviour instead). Tally: 14 tasks, 10 READY-NOW, 4 WAITS-FOR (one of them with a READY-NOW interim observation).

## Change scope

Intent / Scope: **Minimal**. Procedure only. The only repository artifacts a task may create are run-evidence files outside the repository and, in Task 14, nothing in the repository at all. No product code, test, doc or generated file changes. Basis: `plans/handoffs/inv-plan-daily-acceptance.md` and `inv-common.md` on `plan/cloud-investigations`; the landed audits `plans/reviews/audit-ne-sr.md` (`plan/inv-audit-ne-sr`) and `plans/reviews/audit-pa-j-b.md` (`plan/inv-audit-pa-j-b`), whose real-machine checklists are reused by citation. Compatibility: no wire, policy or settings schema is touched; approved host records are frozen before any load. Non-goals: landing I/SR/PA/J/B/NE tasks; Strata; Ollama; the strong tier (owner-deferred 2026-10-21); public egress (G stays disabled); performance benchmarking.

## Context the executor needs

### Base and drift from the brief

- `inv-common.md` names `fork/groundwork` at `210babf`. The branch has moved two first-parent merges: `8739691` "Merge roadmap/ne-t5b-readiness (NE Task 5 Part B)" and `8a63866` "Merge roadmap/i-public-lifecycle (test: deterministic borrowed alias assertion)". This plan is written against `8a63866`. Consequence: **NE Part B is landed** (the brief says "under review"): the strict readiness envelope guard is in `services/nvpair-proxy/readiness_contract.go:153-158` and the PC3 process gate `TestNEServingEnvelope` exists (`services/tests/serving_envelope_test.go:761`). VERIFIED by `git log --first-parent` and `git show --stat 8739691`.
- Not landed at `8a63866` (VERIFIED by absence): I Tasks 4b/5/6/7 (no `services/nvpair-proxy/routing.go`, `response_sink.go`, `routing_cancel.go`; `rg routing:cancel services` is empty), PA Tasks 1-3/7 (no `PausedBy` in `services/nvpair-proxy/discovery.go`; `subscribedToNode` copies roles, served models, revision and strict capability only, `services/nvpair-proxy/proxy.go:1356-1361`), SR 5a-ii/5b/5c/6a/6b, B Tasks 2-6, J Tasks 2-6, G Tasks 3-4, SC-b.

### Inventory of what already exists for live validation (handoff item 1)

Documents (`docs/*.mdx`):

| Document | What it gives this plan |
|---|---|
| `docs/service-restart-reconnect.mdx:53-90` | the only written real-machine procedure in the repository: "Owner-run Windows installed-service acceptance: NOT RUN", six numbered steps; reused verbatim by Task 7 |
| `docs/local-qualification.mdx:108-160` | how a strict model is installed (identity, digests, envelope fit) and what a strict request must satisfy; Task 3 reads `strictAdmission` from it |
| `docs/strict-local-admission.mdx:14-40` | the `/v1/models` `policy.strictAdmission` shape Task 3 polls |
| `docs/performance-telemetry.mdx` | the accurate-or-unknown contract: nonstream samples have unknown protocol evidence, `--` means unknown, failed/cancelled/interrupted samples do not contribute to decode coverage; Task 10 asserts against it |
| `docs/engine-settings.mdx`, `docs/engine-lifecycle.mdx:163-196` | settings apply/CAS and managing engines on other machines; Task 6 background |
| `docs/inference-dispatcher.mdx` + `scripts/inference-dispatcher.{sh,ps1}` | a Go synthetic client that behaves like any third-party client (`scripts/inference-dispatcher.sh:5-8`); used only as a control when OpenCode's behaviour needs isolating |

Cross-process tests (`services/tests`, module `go 1.25.0`; `go vet ./...` exit 0 and `go test -list . .` listed 180 top-level cases in this container without running one; no case was executed here):

| Selector | Gate | Covers |
|---|---|---|
| `TestXARemoteConfiguration{Status,Conflict,Veto,Roles,Serving,Switch,Service}` | skips on darwin; Windows requires `GOFLAGS` with `nvpair-test-exec` and absolute `NVPAIR_TEST_BIN_DIR` (`services/tests/remote_configuration_test.go:26-31`); Linux unguarded | remote config matrix with fake engines; documented in `services/readme.md:262-291` |
| `TestNEServingEnvelope` | Windows only (`services/tests/serving_envelope_test.go:762-763`); the NE fixture image must be the fixed-path copy and `NVPAIR_NE_SERVING_EXECUTABLE` must name it (`:39,:54-60`) | E apply -> generated launch -> cold readback -> broker join -> strict destination Chat with fake llama |
| `TestPAPausedAdvertisementConvergesWithoutMDNSChange` | Windows only (`services/tests/paused_advertisement_test.go:38-41`) | scanner-to-broker pause propagation (fake inventory) |
| `TestServiceRestartReconnect` (`services/tests/service_restart_reconnect_test.go:33`), `TestServicePolicy` (`service_policy_test.go:21`) | platform rules in `services/tests/service_harness_test.go:137`; Windows fixed path | real `nvpair-service` restart with a retained TUI RPC client; native policy edit survives restart |
| `TestEngineSwitchProcess*` incl. `DrainsInflightBeforeStop` (`engine_switch_test.go:478`), `PinnedDirect`/`PinnedReverse` (`:592,:779`), `TestE4Switch*` (`:651,:715`) | darwin skips at `engine_switch_test.go` fixtures | switch drains in-flight work; remote switch over pinned mesh |
| `TestRemoteManagement*` (12 cases), `TestReversePeerControlTask2*`, `TestSecureInferenceClusterMTLS` (`secure_inference_test.go:263`) | darwin skips | target-local permission, denied peer can still infer, pinned mTLS inference |
| `TestPerformanceObservation*` (11 cases incl. `Interruption` `:351`, `Cancellation` `:423`) | process fixtures | telemetry outcomes with synthetic engines |
| `TestLlamaCPPFacadeUsesRouterInventoryAndExactModelIDs` (`llamacpp_interop_test.go:64`) | process fixture | llama.cpp facade uses router inventory ids |

Gates to respect when selecting from the above: `broker_management_test.go:130,198`, `ollama_host_alias_test.go:24`, `model_routing_interop_test.go:43`, `scheduler_interop_test.go:559` and `workload_identity_interop_test.go:49,161,289` need product default ports (11434/11435/1234/14318) and skip when occupied; `ghost_node_test.go:41,91` and `broker_test.go:70` (`TestModelsRefreshConvergesWithoutMDNSChange`) use real multicast. Only `fixture_ports_test.go:251` style reservations are hermetic. Nothing in `services/tests` starts a real engine.

Live-tagged unit tests: `services/nvpair-engine-manager/live_test.go:14-15,42,189` and `live_pound_test.go:11,25` run only under `-tags live` with `NVPAIR_LIVE_OLLAMA`/`NVPAIR_LIVE_LMSTUDIO`; `installlayout_test.go:99-102` under `NVPAIR_LIVE_LAYOUT`. None applies to llama.cpp; none is used by this plan. `services/nvpair-proxy/cors_test.go:437-438` contacts `PAIR_CORS_PARITY_URL`/`PAIR_CORS_OLLAMA_EXECUTABLE` when set; unset them for every run.

Windows harness: `scripts/windows-test-bin.ps1` (`install`/`list`/`rules`/`run`, `:5-13`; requires absolute `NVPAIR_TEST_BIN_DIR`, `:16-19`), documented at `services/readme.md:215-246`. Binaries the process suite builds and copies: `services/tests/main_test.go:56-67` (proxy, errors, node-info, scanner, broker, service, node-settings, workload-manager, engine-manager, manual-nodes, cluster-manager, job-scheduler); fixture bundles in `services/shared/testbin/programs/tests.txt`.

Existing acceptance checklists reused by citation: `audit-ne-sr.md` "Real-machine acceptance checklist" items 1-8 (router `id`/`status.value`, child `/props`, 10-minute sweep stability, Windows binding semantics, same-stat mutation, `sleep-idle-seconds = -1`, resident preflight refusals, `TestNEServingEnvelope`) and `audit-pa-j-b.md` items 1-7 (PA T6 log lines, XA selectors with firewall before/after, `store:false` on a real OpenCode client, PA end-to-end after T2, J manual-node id). Where this plan's task repeats one, it names the audit item instead of restating it.

### Verified code facts the checks observe

Tier routing. A request whose `model` is `weak`, `medium` or `strong` resolves on the node it enters, from the roles nodes declare for themselves (`services/nvpair-proxy/tiers.go:6-9`; `declaredRoles` `:279`, `resolveTierCandidates` `:363`). Empty tier answers 404 `no eligible model for tier <tier>` (`services/nvpair-proxy/http_handler.go:820-828`); an unadvertised concrete model answers 502 `{"error":"no available node advertises the requested model"}` (`:836`). `Role` is `{engine, model}` (`services/shared/nodepolicy/policy.go:160-163`); the desktop editor labels Weak/Medium/Strong (`desktop/src/ui/components/RoutingSettings/RoleEditor.tsx:29-31`). The llama.cpp facade listens on 8080 and the engine on 8081+ (`services/shared/engines/engines.go:163-167`; manifest `services/nvpair-engine-manager/manifests/llamacpp.json:24`).

Strict path is not the daily path yet. `X-PAIR-Strict-Admission` (`services/shared/nodepolicy/routing_wire.go:20`) is parsed at `http_handler.go:605-620` and only then does `serveStrictLocal` run; the only producer of a strict candidate is `hybridLocalCandidates` (`tiers.go:445-473`, `node.strictModel` at `:468`), and hybrid selection is I Task 5 (unlanded). INFERRED: an OpenCode request at the facade today takes the concrete/legacy-tier path, so B qualification (`validateRequest`, `store` handling at `services/nvpair-proxy/local_qualification.go:326,403-406`) and strict admission tickets (`readiness_contract.go:700-726`) are not exercised by daily traffic until I Task 5 lands. This is `audit-pa-j-b.md` UNV-4 restated for the whole milestone.

Admission refusal. A destination refuses with 503, `X-PAIR-Admission: <reason>` and `Retry-After` (`services/nvpair-proxy/admission.go:886-897`); reasons `paused|draining|busy|no-fit|engine-off|wake-timeout` (`services/shared/nodepolicy/wire.go:1119-1124`), plus `not-served` (`admission.go:385-393` per `audit-pa-j-b.md`). The origin treats 503 as retryable and fails over.

Pause/veto/permission. Remote resume of a locally paused node returns `localVetoError` before any transition (`services/nvpair-ui-broker/availability.go:59-63`), code `-32011` (`services/nvpair-ui-broker/policy.go:443-446`); remote start/load of a locally paused node or an engine saved Off is refused by engine-manager with `this node was paused on the node itself; resume it there` / `<engine> was turned off on the node itself; start it there` (`services/nvpair-engine-manager/nodeprojection.go:96-100`), HTTP 409 code `local_veto` on the `ec` route (`controllifecycle.go:76-82`). Target-local permission is re-read at every acceptance (`services/nvpair-ui-broker/remote_management.go:58-67`), denial is `-32012` (`:21-24`), applied to availability (`availability.go:55`), settings (`enginesettings.go:553`), switch (`engineswitch.go:474,515`) and policy (`policy.go:582`). The desktop pairing choice reads `Allow this peer to manage this device (my own devices)` and the node card reads `may manage` / `is blocked from changing this device` (`desktop/src/ui/components/InvitePairingPanel.tsx:38,131`). Remote settings/switch requests carry `nodeId` and are relayed as `engine:remote-<method>` (`enginesettings.go:663-667`).

Headless service. `nvpair-service` is a user-session process, not a Windows SCM service: autostart is the `HKCU\...\Run\NVPAIRService` entry (`services/nvpair-service/README.md:95-108`); the Windows endpoint is a per-user named pipe (`README.md:23-30`). It restarts an exited broker with 1 s doubling to 30 s backoff (`README.md:139-146`; `service.go:70-71`), broadcasts `service/broker-restarted` and replays `*:subscribe` after `app:ready` (`service.go:552-557`). The desktop bridge listens for that notification (`desktop/src/electron/service-bridge/json-rpc-client.ts:88,430`; `modular-supervisor.ts:724`). The TUI reconnect contract is `docs/service-restart-reconnect.mdx:6-20`.

Interrupted streams. The proxy terminates an interrupted stream with one bounded frame: SSE `data: {"error":{"message":"stream interrupted: <reason>","type":"server_error","code":"stream_interrupted"}}` then `data: [DONE]` (`services/nvpair-proxy/streamframe.go:228-236`), reasons at `:195-201` (`upstream closed before the stream completed`, `upstream read failed`, `upstream stopped sending events`, ...), chosen by `finishInterruptedLocked` (`:993-1003`). The request log carries `stream interrupted after output began on <node>: <reason>` (`http_handler.go:1020-1027`). No same-model continuation exists (SC-b absent; I plan Decisions: "At I's endpoint no qualified continuation implementation exists"), so the milestone's "delay or pause-resume" cannot be observed; the honest error can.

Cancellation. The admission ticket is released by `defer` when the handler returns (`http_handler.go:547`); a client that disconnects before completion emits terminal `cancelled` with `client disconnected before completion` (`:983`); workload states are `completed|failed|cancelled` (`services/nvpair-workload-manager/workload.go:23-30`). `workloads:cancel` requires `{originatedFrom, workloadId, engine, runId}` on every hop (`services/nvpair-ui-broker/workloadcancel.go:26-33`, I 3b landed); the proxy answers `{found}` (`services/nvpair-proxy/cancel.go:170-180`). Strict tickets keep capacity on `uncertain` evidence (`readiness_contract.go:709-726`) but, per the strict-path fact above, no daily request holds one today.

Telemetry. Measurements carry `Outcome` completed/failed/cancelled (`services/nvpair-proxy/performance.go:798-817`); present-but-unusable engine fields mark the sample unknown rather than guessing (`:884-918`); the schema and display rules are in `docs/performance-telemetry.mdx`.

NE/E settings. The serving record is `services/shared/engines/serving.go:21-22,85-89` (`modelId`, `maxOutputTokens`, `memory.{hostRamBytes,vramBytes}`, `artifacts[]`, `expectedNativeSourceDigest`); the derived envelope `:124-134`; readback `status`/`configured`/`launchValidated`/`observed` `:167-171`. The llama preset keys the compiler reads are `models-preset`, `models-max`, `ctx-size`, `parallel`, `kv-unified-per-slot`, `kv-unified`, `sleep-idle-seconds` (`services/nvpair-engine-manager/serving_llamacpp.go:75-88`); `-1` retention is accepted at `:221`; PAIR's NE-mode preset defaults are `ctx-size = 8192, parallel = 1, kv-unified = false, fit = off, sleep-idle-seconds = 300, cache-ram = 512, load-on-startup = false` (`:569`). Strict readiness requires `q.ContextTokens <= PerSlotContextTokens`, `q.MaxOutputTokens <= MaxOutputTokens`, `MaxConcurrent <= Slots`, `MemoryBytes >= VRAMBytes` (`readiness_contract.go:153-158`). Manager loads use manifest `POST /models/load {model}` and `GET /models` filtered by `status.value == loaded` (`manifests/llamacpp.json:103-115`).

What could not be verified here: nothing Windows-native, no engine or model, no OpenCode binary, no desktop run (`npm` was not executed), no `-race`. The owner's frozen NE host records, OpenCode configuration and the private `TESTING_ON_PC.md` are outside this repository; where a step depends on them it says so.

## Decisions (settled — do not revisit)

- Each node declares its own weak/medium/strong roles; a router consumes the declaring node's roles. Not router-side tier tables, because `declaredRoles` is the landed owner.
- Local Off/pause wins over remote start, resume and residency; remote configuration edits stay permitted while paused (`TestConfigurationEditsStayAllowedWhileLocallyPaused` exists in the broker).
- Pairing marks a peer as "my own device" or locked; enforced target-locally at acceptance time, never cached.
- Interrupted streams: the user sees only delay or pause-resume; an error only if no same-model continuation exists. Because no continuation exists at `8a63866`, the acceptable observation today is exactly one `stream_interrupted` error frame followed by `[DONE]`, never a silent truncation and never a 200 that pretends completion.
- Strata deferred; strong tier deferred; plain direct-model output caps deferred (NE Part D after I 4b); no Ollama in the fleet; G stays disabled; Mac never serves a model, not even for troubleshooting.
- No test touches production ports or a running candidate without the owner present. Process tests run on PC3 only, serialized, under the fixed-path wrapper; MSI supplies unit `-race` evidence only.

## Decisions made while writing this plan

- Checks are ordered so the owner can start today: freeze and preflight (T1), automated Windows gates (T2), per-host approved configuration (T3), then the live clauses (T4-T11), with the WAITS-FOR tasks (T8 pause-resume, T11 strict shapes, T12 PA skip, T13 residency) placed last and each carrying an interim READY-NOW observation that records current behaviour.
- The llama.cpp facade port 8080 is treated as the OpenCode base URL on each serving host and on the Mac; the owner confirms the actual configured port in T1 (`engine:set-port` readback). This is the only port the plan names for a client.
- `inference-dispatcher` is used only as a control when an OpenCode observation is ambiguous; OpenCode is the client of record.
- Every check records: exact command or UI action, the JSON-RPC method and the fields read (never bodies), the HTTP status/headers/first error object, and the log line category. Prompts are fixed synthetic sentences chosen by the owner and never copied into evidence.
- Evidence lives in one run directory outside any repository or synced vault, named by date and host role; Task 14 summarises it. This plan does not create repository files.
- Because the daily OpenCode path is non-strict until I Task 5, T11 observes `store:false` and tools on the non-strict path (what llama-server does with them) and marks strict-path verification WAITS-FOR I Task 5 plus B Tasks 2-6.

## Tasks

Conventions: "serving host" = MSI or PC3; "origin" = the Mac; `OC>` = an action in OpenCode; `D>` = desktop; `T>` = TUI; `RPC` = a JSON-RPC call through `nvpair-service call <method> '<json>'` (`services/nvpair-service/README.md:41-52`). "fail" conditions are exhaustive for that check; anything else is "unverified, record and continue".

### Task 1 — Freeze inputs and preflight every machine — READY-NOW

**Files:** none (procedure). Reads `services/readme.md:215-246`, `scripts/windows-test-bin.ps1`, `docs/service-restart-reconnect.mdx:53-90`.

**Depends on:** none. External: owner present; the owner's frozen NE host records for MSI and PC3 (SR Q3 OWNER-CONFIRM: exact llama build, model artifact digests, per-model `ctx-size`/`parallel`/`sleep-idle-seconds`/`models-max`, host RAM/VRAM bounds); `TESTING_ON_PC.md` section 10.

**Complexity:** judgment

**Do:**
1. Record `git rev-parse origin/fork/groundwork` on every machine; all three must print `8a63866c1cbadd70f40732c84b04efb72d6e2fd4` or the same newer SHA, noted once.
2. On MSI and PC3 record the OpenCode version (`opencode --version`) and the provider block of its config with secrets redacted: base URL, model ids offered, `compatibility.supportsStore` if set. Expected from `plans/b-opencode-shapes.md` (branch `plan/b-opencode-shapes`): MSI 2.1.1, PC3 2.0.32; a different version is recorded, not a fail.
3. On MSI and PC3 verify `GOFLAGS` contains `nvpair-test-exec` and `NVPAIR_TEST_BIN_DIR` is absolute; run `scripts/windows-test-bin.ps1 list` and compare every printed path with the inbound firewall rule list (exported before and after the whole run). Missing rules: stop; elevated `rules` is an owner action, never done by the procedure.
4. List every executable the procedure will start, and confirm a firewall rule or an owner decision for each: `nvpair-service.exe`, `nvpair-ui-broker.exe`, `nvpair-proxy.exe`, `nvpair-engine-manager.exe`, `nvpair-node-scanner.exe`, `nvpair-cluster-manager.exe`, `nvpair-node-info.exe`, `nvpair-node-settings.exe`, `nvpair-manual-nodes.exe`, `nvpair-job-scheduler.exe`, `nvpair-workload-manager.exe`, `nvpair-errors.exe`, `nvpair-tui.exe`, the desktop application executable, the pinned `llama-server.exe` (router process and, INFERRED from router mode, each model child it spawns from the same image), the OpenCode executable and its node runtime if separate, `scripts/inference-dispatcher.ps1`'s built client, every `go-test\<module>\*.test.exe` and `tests\products\*`, `tests\service-bundle-1\*`, `tests\policy-joint\*`, `tests\subscription-scanner\*`, `tests\port-bridge\*`, `tests\stalled-worker\*`, `tests\workload-bridge\*`, `tests\performance-observations\*` under `NVPAIR_TEST_BIN_DIR` (`services/shared/testbin/programs/tests.txt`). A Windows Firewall prompt during the run is recorded with the executable path; the run continues only if the owner answered it.
5. Record listeners before the interval (`netstat -ano` filtered to 8080, 8081-8090, 11434, 11435, 1234, 14318-14323 and the cluster `ec` port the broker reports). The installed PAIR candidate must be stopped by its reviewed controller before T2; for T3 onward it is started again deliberately, and no `services/tests` case runs while it is up.
6. Freeze the NE host record per serving host: for each model, `modelId` (must equal the preset section name exactly), artifact `sha256` with provenance kind, `ctx-size`, `parallel`, `sleep-idle-seconds`, `models-max`, `maxOutputTokens`, `memory.hostRamBytes`, `memory.vramBytes`. Default selection from the SR plan: MSI weak = LFM2.5-24B-A2B UD-Q2_K_XL, medium = Tiel-Coder-35B-A3B UD-Q4_K_XL; PC3 weak = LFM2.5-24B-A2B UD-Q2_K_XL; no strong. Without the owner's confirmed numbers, T3 does not apply anything.
7. On the Mac: `engine:intent` readback shows every engine Off (`OffBy` local) and no local model; record it. The Mac desktop and TUI are the only clients used from the origin.

**Verify:** A preflight sheet exists with items 1-7 filled; firewall export "before" saved; installed candidate state recorded. Fail: any machine on a different SHA, missing wrapper/rules, or no confirmed host record.

### Task 2 — Automated Windows gates that exist today — READY-NOW

**Files:** none (procedure). Runs existing tests only.

**Depends on:** 1. External: PC3 reserved interval, installed PAIR stopped; MSI reserved interval for `-race`.

**Complexity:** mechanical

**Do:** On PC3, from `services/tests`, serialized, one command at a time, with `PAIR_CORS_PARITY_URL`, `PAIR_CORS_OLLAMA_EXECUTABLE`, `HTTP(S)_PROXY`, `ALL_PROXY` unset and `NVPAIR_NE_SERVING_EXECUTABLE` pointing at the fixed-path fake image the NE fixture expects (`serving_envelope_test.go:39-72`):

```
go test -v -count=1 -timeout 30m -run '^TestXARemoteConfiguration(Status|Conflict|Veto|Roles|Serving|Switch|Service)$' .
go test -v -count=1 -timeout 20m -run '^TestNEServingEnvelope$' .
go test -v -count=1 -timeout 10m -run '^TestPAPausedAdvertisementConvergesWithoutMDNSChange$' .
go test -v -count=1 -timeout 20m -run '^(TestServiceRestartReconnect|TestServicePolicy)$' .
go test -v -count=1 -timeout 30m -run '^(TestEngineSwitchProcess(CompletesAndRetries|RejectsStaleRevision|RejectsRunningTargetBeforeDrain|DrainsInflightBeforeStop|BrokerDeathInterruptsWithoutRestore|PinnedDirect|PinnedReverse)|TestE4Switch.*)$' .
go test -v -count=1 -timeout 30m -run '^(TestRemoteManagement.*|TestReversePeerControlTask2.*|TestSecureInferenceClusterMTLS|TestLlamaCPPFacadeUsesRouterInventoryAndExactModelIDs)$' .
go test -v -count=1 -timeout 30m -run '^TestPerformanceObservation.*$' .
```

Do not run `./...`, `make test` or the selectors in the "Gates to respect" paragraph (they need product default ports or multicast). On MSI, from each of `services/shared`, `services/nvpair-proxy`, `services/nvpair-ui-broker`, `services/nvpair-engine-manager`, `services/nvpair-node-scanner`: `go vet ./...` then `go test -race -count=1 -p 1 -parallel 1 ./...` with the engine-manager `-skip PersistPort` and broker `-skip EngineManagerPortsSpawn` exclusions recorded as untested.

**Verify:** Every command prints `ok`; `-v` output shows each named case ran (`=== RUN` and `--- PASS`, no `--- SKIP`); each fixture's cleanup line reports no owned survivors (`owned_running=0` for service fixtures, `fixture_worker_exited` for PA); no `WARNING: DATA RACE` on MSI. Fail: any `SKIP` of a named case, any `FAIL`, any survivor, any firewall prompt not answered by the owner. This closes `audit-pa-j-b.md` UNV-1/UNV-2 and `audit-ne-sr.md` checklist item 8.

### Task 3 — One approved serving configuration per host — READY-NOW

**Files:** none (procedure). Reads `docs/local-qualification.mdx:108-160`, `docs/strict-local-admission.mdx:14-40`, `services/shared/engines/serving.go:85-89,124-134,167-171`.

**Depends on:** 1 (frozen host record). External: owner's separate live-engine approval per host (NE plan "MSI — owner-approved actual llama gate", "PC3 — owner-approved actual llama gate").

**Complexity:** judgment

**Do:** On each serving host with the installed PAIR running and the owner present:
1. `D>` or `RPC engine:preview-settings` with the llama.cpp launch text pointing `--models-preset` at the frozen preset file and the `serving` record from the host record, `expectedNativeSourceDigest` absent. Expected: preview returns the derived digest and one `configured` envelope per model with `perSlotContextTokens`, `slots`, `maxOutputTokens`, `retention`, `routerCapacity` equal to the frozen values. Fail: a field error, or a derived value that differs from the preset (do not edit the preset to match; stop and report).
2. `RPC engine:apply-settings` with the previewed digest copied into `expectedNativeSourceDigest`. Expected: receipt accepted; `engine:get-settings` readback `serving.status` is `launchValidated` (cold) with `launchValidated[]` equal to the configured envelopes and `observed` absent. Fail: `unverified`/`mismatch`, or a changed launch argv that still contains `--ctx-size`/`--parallel` router overrides (NE Task 2 contract; inspect the engine-manager debug log's argv line, not the preset).
3. Start the engine (`engine:start`). Expected: `GET /models` on the router (engine port 8081) lists each preset `id` exactly with `status.value` in `unloaded|loaded|loading|sleeping` (`audit-ne-sr.md` item 1); the broker's `engine:model-identity` notification carries `serving` keyed by both model ids with non-empty `contentDigest` (`services/shared/nodepolicy/wire.go:1244-1250` per audit). Fail: identity absent, or `serving` missing for a declared model.
4. Install B qualification evidence for each model: copy the `contentDigest` and launch digest from the identity into the qualification record at the frozen per-slot context (`audit-ne-sr.md` LOW-6: context must equal the preset per-slot value exactly), with `shapes` absent (B Task 2 unlanded; version 1 schema). Expected: `/v1/models` on the facade (port 8080) shows `policy.strictAdmission.models[]` containing each model with `maxConcurrent` equal to `parallel` and an `identity` whose `modelDigest` equals the `contentDigest`. Fail: the model is absent from `strictAdmission` after 60 s (one full inventory sweep); record the readiness reason from the broker log category `strict-readiness`.
5. Send one small Chat Completions request with the concrete model id (`OC>` or inference-dispatcher) so the child loads; then `GET /props?model=<url-encoded id>&autoload=false` on the router. Expected: `default_generation_settings.n_ctx` equals the per-slot value and `total_slots` equals `parallel` (`audit-ne-sr.md` item 2); readback `serving.status` becomes `observed` within 60 s. Fail: `mismatch` (record `reason`), or a cold model that this GET loads (watch the process list).
6. Leave the engine running for 10 minutes with one model loaded and one cold (MSI) and confirm readback stays `observed` and identity keeps `serving` (`audit-ne-sr.md` item 3, UNVERIFIED-1 probe). Record any flip to `mismatch` with its reason text; that is evidence for the owner question, not a fail of this task.
7. On the Mac nothing is applied; record `engine:intent` again (all Off).

**Verify:** Per host: steps 1-6 recorded; final `engine:get-settings` readback JSON (no paths) saved; `/v1/models` `strictAdmission` saved. Pass: `observed` status, strict entries for every frozen model, zero `mismatch` in the 10-minute window. The NE "MSI/PC3 owner-approved actual llama gate" wording in `plans/ne-serving-envelope.md` is satisfied by this task except the "above-bound strict output request rejected before forwarding" item, which WAITS-FOR I Task 5 (no strict daily path; see T11).

### Task 4 — Tier routing with declared roles on each serving host — READY-NOW

**Files:** none (procedure). Reads `services/nvpair-proxy/tiers.go:6-9,279,363`, `http_handler.go:820-836`.

**Depends on:** 3.

**Complexity:** mechanical

**Do:** On each serving host:
1. `D>` Settings -> Routing -> roles: weak = llama.cpp/LFM2.5 model id; medium (MSI only) = llama.cpp/Tiel-Coder model id; strong left empty. `RPC policy:get` afterwards must show `roles.weak.engine == "llamacpp"` and the exact model id. Fail: a role saved with a model id that is not a preset `id`.
2. `OC>` with base URL the facade (port 8080) and model `weak`: one short turn. Expected: HTTP 200 SSE, the response `model` field equals the weak preset id (llama.cpp echoes the concrete id), broker `workloads:get-initial` shows one workload with `model` = concrete id, `scheduledOn` = this node, state `completed`. Fail: 404 `no eligible model for tier`, or a response from a different model.
3. MSI only: repeat with `medium`. Expected: 200 and the medium preset id; the cold child loads on demand (record the time to first byte as information, not a bound).
4. `OC>` model `strong`. Expected on both hosts: HTTP 404 body containing `no eligible model for tier strong` (`http_handler.go:820-828`). Fail: 200 (would mean a strong role exists) or 502.
5. `OC>` a concrete model id that no node serves. Expected: 502 `{"error":"no available node advertises the requested model"}` (`:836`).
6. Change the weak role to the medium model (MSI) and resend `weak`: the response `model` must follow the role change on the next request without restart. Restore.

**Verify:** Per host: six request records with status, response `model`, `X-PAIR-Admission` header if any, and the workload row. Pass: 2-3 return 200 on the declared model; 4-5 return the exact 404/502 bodies; 6 follows. Nothing in this task contacts a peer.

### Task 5 — Peer routing from the origin-only Mac — READY-NOW

**Files:** none (procedure). Reads `services/nvpair-proxy/proxy.go:631-735` (candidates), `tiers.go:279-300` (declared roles of peers), `docs/performance-telemetry.mdx` (destination-only samples).

**Depends on:** 3, 4. External: Mac, MSI and PC3 paired with each other, MSI and PC3 marked "my own device" from the Mac's point of view where T6 needs it (T6 toggles this).

**Complexity:** judgment

**Do:**
1. Mac: all engines Off (T1 item 7). The Mac's llama.cpp facade must still be enabled so the proxy has a routing view (`tiers.go:15-18`: an engine with no facade enabled contributes no candidates). Record `engine:set-port` readback for the Mac facade port.
2. `D>` on the Mac Nodes list: MSI and PC3 appear with `Declared roles` (`desktop/src/ui/components/NodeList/NodeDeclaredRoles.tsx`) weak (both) and medium (MSI). Fail: roles missing after 60 s (scanner policy refresh is nominal 15 s, `audit-pa-j-b.md` UNV-1 note).
3. `OC>` on the Mac with base URL the Mac facade, model `weak`: Expected 200; the Mac proxy's `proxy/request` event carries `node_id` of MSI or PC3 (stable ID order after the scheduler priority, `proxy.go:660`); the workload appears on the Mac with `originatedFrom` = Mac and `scheduledOn` = the peer; the peer's `workloads` shows the same `workloadId`. Fail: 404 tier (roles not consumed from peers), 502, or a `node_id` of the Mac.
4. `OC>` model `medium`: Expected 200 from MSI only. Then pause PC3 locally (`T>` P on PC3) and send `weak` five times: every request must land on MSI; PC3's proxy log may show one 503 `paused` per request until PA Task 2 lands (T12 records this). Resume PC3.
5. Concrete routing: `OC>` the PC3 weak model id from the Mac: 200 from PC3. Then turn PC3's llama.cpp Off locally and resend: Expected 503 `X-PAIR-Admission: engine-off` from PC3 observed at the Mac proxy as a failover attempt, final answer from MSI if MSI serves that id, else the 502/503 passes through (record which). Turn PC3 On.
6. Telemetry: on the Mac the performance view for the peer's profile must stay empty (origin does not count another local execution, `docs/performance-telemetry.mdx`); on the serving peer the sample appears.
7. Security control: temporarily unpair PC3 from the Mac (or revoke the pin) and resend `weak`: PC3 must never be a candidate (`proxy.go:643` refreshes membership per request). Re-pair.

**Verify:** Seven records with `node_id`, status, admission header and the workload pair. Pass: 3-5 route to a declared peer, veto/Off produce the named 503 reason and a failover, 7 excludes the unpinned peer. This is the SR plan's "Mac-origin peer routing through llama.cpp with all Mac engines Off".

### Task 6 — Remote configuration and switching from the desktop with permission and veto — READY-NOW

**Files:** none (procedure). Reads `services/nvpair-ui-broker/remote_management.go:21-24,58-67`, `availability.go:55-63`, `enginesettings.go:663-667`, `engineswitch.go:474,515`, `nodeprojection.go:96-100`, `InvitePairingPanel.tsx:38,131`.

**Depends on:** 3. External: XA fake-process selectors passed in T2 (same matrix with fake engines).

**Complexity:** judgment

**Do:** From the Mac desktop against MSI (target), then once against PC3 for the permission matrix only:
1. Permission on: MSI's node card shows `may manage this device` for the Mac (set at pairing or via the card's `RemoteManagementChoice`). `D>` edit MSI's roles (swap weak/medium models) and save. Expected: MSI `policy:get` locally shows the new roles and a higher `revision`; the Mac card refreshes. Fail: `-32012`.
2. Stale CAS: resend the previous edit with the old `expectedPersistedRevision` through `RPC policy:set` with `nodeId` = MSI. Expected `-32010` and unchanged bytes (`policy.go:443`).
3. Engine settings remotely: `D>` MSI engine settings, change only the llama.cpp port text (not the serving record), apply. Expected: receipt OK; MSI `engine:get-settings` still carries the `serving` record unchanged (NE Task 1 semantic equality; `audit-ne-sr.md` composition check 4). Fail: `serving` lost or `status` regressed from `launchValidated`/`observed`.
4. Engine switch remotely: `D>` switch MSI's served engine llama.cpp -> Off -> llama.cpp (E4 control; `engine:switch` with `nodeId`). Expected: `engine:switch-status` on MSI reports completion; an identical retry returns the stored receipt without a second switch (XA Task 7 contract). Fail: a second start observed in MSI's engine-manager log.
5. Local veto, pause: `T>` on MSI press P (pause). From the Mac `D>` resume MSI. Expected `-32011` with `this node was paused on the node itself; resume it there`; MSI stays paused; from the Mac, editing MSI's roles still succeeds (edits allowed while paused). Resume on MSI itself.
6. Local veto, Off: on MSI set llama.cpp Off locally (`engine:stop`, `OffBy` local). From the Mac `D>` start it. Expected `-32011` `llamacpp was turned off on the node itself; start it there`. Start it on MSI.
7. Permission off (locked): on MSI flip the Mac's choice to `is blocked from changing this device`. From the Mac repeat 1, 3, 4 and a resume. Expected: every write answers `-32012` (HTTP 409 `management-denied` body on the `ec` route); reads (`policy:get`, `engine:get-settings`, model list) still succeed; a `weak` inference from the Mac still routes to MSI (`TestRemoteManagementDeniedPeerCanInfer` contract). Restore permission.
8. Repeat 7 once against PC3.

**Verify:** A matrix table permission × {roles, settings, switch, resume, start, inference} × {MSI, PC3} with the observed code. Pass: exactly `-32010`/`-32011`/`-32012` where named, success elsewhere, `serving` preserved, no second engine start. Fail: any widened code set.

### Task 7 — Headless `nvpair-service` on Windows: broker restart, service restart, client reconnect — READY-NOW

**Files:** none (procedure). Executes `docs/service-restart-reconnect.mdx:53-90` verbatim plus two additions. Reads `services/nvpair-service/README.md:95-108,139-146`, `service.go:552-557`, `json-rpc-client.ts:88,430`.

**Depends on:** 3. External: owner present; no inference, download or test running.

**Complexity:** judgment

**Do:** On MSI (then PC3):
1. Headless start: `nvpair-service autostart enable` then sign out and in; confirm `nvpair-service status` answers and the Run key `NVPAIRService` exists (`README.md:108`). Attach the TUI and the desktop. Record PIDs/creation times of service, broker and workers (procedure step 1).
2. Broker-process restart: with both clients attached and an `engine:subscribe` active, terminate only the broker process by PID (owner action). Expected within 35 s: service log shows the restart (backoff 1 s on first attempt), both clients receive `service/broker-restarted` (desktop bridge logs `broker-restarted`; TUI shows reconnecting then connected), subscriptions resume without client action (service replays `*:subscribe`), `service/status` `brokerRestarts` increments by 1 (`service.go:363`). Fail: a client that needs manual reconnect, or a subscription that stays silent after a policy change on the node.
3. Service-process restart: steps 3-6 of `docs/service-restart-reconnect.mdx` exactly: stop the exact service through its reviewed controller, observe disconnected state in both clients, restart the same profile at the same endpoint, confirm the original TUI and desktop PIDs survive and refresh, submit one owner-approved `weak` request from OpenCode on that host and confirm the replacement proxy serves it (200, workload row on the replacement broker). Fail: either client process restarts, or the request is served by a leftover old proxy (check PIDs).
4. Engine survival: record whether the llama-server router survived the broker restart (adopted run) or was restarted, and whether `engine:model-identity` after the restart carries `serving` again. An adopted/external run publishes no identity by design (`audit-ne-sr.md` NE Task 4 Part A "Adopted/external runs publish none"); record which case occurred and, if adopted, apply `engine:restart` and confirm identity returns. This is information for the owner question on headless identity, not a fail.
5. Firewall and listener export after the interval; survivor scan.

**Verify:** The six-step record from the document plus items 2 and 4, with PIDs, times and the `brokerRestarts` counter. Pass: both restarts recovered with retained clients and one served request. Fail conditions as stated. Note for the owner: `nvpair-service` is a user-session process with a Run-key autostart, not a Windows SCM service; a headless machine with no interactive login has no service (see Owner questions).

### Task 8 — Interrupted streams: peer loss and engine switch mid-stream — WAITS-FOR I Task 4b and SC-b (interim observation READY-NOW)

**Files:** none (procedure). Reads `streamframe.go:195-201,228-236,993-1003`, `http_handler.go:1020-1027`, `engine_switch_test.go:478`.

**Depends on:** 5. External: owner present on both machines.

**Complexity:** judgment

**Do:** Interim READY-NOW observation (what the user sees today):
1. From the Mac, `OC>` a `weak` turn with a prompt that produces a long answer, routed to PC3 (pause MSI first so PC3 is the only candidate). Two seconds into streaming, disconnect PC3's network (or stop its llama-server child by PID with the owner). Expected at OpenCode: output stops, then exactly one error frame `{"error":{"message":"stream interrupted: upstream closed before the stream completed" | "upstream read failed" | "upstream stopped sending events", "type":"server_error","code":"stream_interrupted"}}` followed by `[DONE]`; the Mac workload row ends `failed`; the Mac proxy debug log carries `stream interrupted after output began on <node>: <reason>`; the performance sample on PC3 (if any) is `failed`. Fail: a stream that ends with `[DONE]` and no error (silent truncation), a 200 with `finish_reason: stop` after the cut, or a hang beyond the idle bound (`streamInterruptReasonIdle`).
2. Engine switch mid-stream: on MSI, start a long `medium` stream from OpenCode on MSI itself, then `D>` from the Mac switch MSI llama.cpp -> Off. Expected: the stream completes normally (switch drains in-flight work, `TestEngineSwitchProcessDrainsInflightBeforeStop`), then the engine stops; a request sent after the switch gets 503 `X-PAIR-Admission: engine-off`. Fail: the stream is cut with `stream_interrupted`, or the engine stops before the stream ends.
3. Client disconnect mid-stream is T9.

WAITS-FOR part: the milestone sentence "the user sees only delay or pause-resume" requires a same-model continuation. At `8a63866` no continuation executor exists (I plan: "interruption is classified as continuation-unavailable and produces SC-a's clean terminal error"; SC-b is not in any package on the listed branches). After I 4b lands the error path moves into the request-owned sink but the observable stays the one clean error; after SC-b (unplanned here) repeat step 1 expecting a pause then same-model resume from MSI. Record the gate: **WAITS-FOR I Task 4b (sink) and SC-b (continuation; no plan branch)**.

**Verify:** Two records with the exact error object (message, type, code), the workload terminal state and the log category. Pass today: honest single error, honest `failed`, no truncation. The milestone clause is marked WAITS-FOR until SC-b.

### Task 9 — Cancelled requests release reservations — READY-NOW (strict-capacity part WAITS-FOR I Task 5)

**Files:** none (procedure). Reads `http_handler.go:547,983`, `workloadcancel.go:26-33`, `cancel.go:170-180`, `workload.go:23-30`, `performance_observations_test.go:423`.

**Depends on:** 4. External: MSI preset with `parallel = 1` for the weak model (frozen record), so one slot makes the release observable.

**Complexity:** judgment

**Do:** On MSI, OpenCode on MSI against the local facade:
1. Client disconnect: start a long `weak` stream, press OpenCode's cancel (or kill the OpenCode process) 2 s in. Expected: MSI proxy emits the terminal `cancelled` event with reason `client disconnected before completion`; the workload row shows `cancelled` (not `failed`); the performance sample is `cancelled` and excluded from decode coverage. Within 1 s send a second `weak` request: Expected 200, not 503 `busy` (the admission ticket was released by the deferred `release()`). Fail: `busy` for more than one retry, a `failed` row, or no terminal row at all.
2. Desktop cancel: start a long stream from the Mac routed to MSI; on the Mac desktop press the workload card's cancel. Expected: the broker answers `{ok}`, the proxy `workload/cancel` answers `{found:true}`, OpenCode receives the stream end (record the exact terminator), the row is `cancelled` on both Mac and MSI, and the next request is admitted immediately. Repeat the cancel once more: `{found:false}` after cleanup. Fail: `found:false` on the first cancel (identity mismatch: compare `originatedFrom`, `engine`, `runId`, `workloadId` on the card with the proxy registry, I 3b contract).
3. Regenerate: trigger OpenCode's regenerate on a completed turn. Expected: a new workload, the old one untouched (post-commit regenerate is a conflict with no cancellation side effect, I Decisions). Record only.
4. Peer-side release: from the Mac start a `weak` stream to PC3 and cancel from the Mac; on PC3 the row ends `cancelled` and the next PC3 request is admitted.

WAITS-FOR part: "cancelled requests releasing reservations" for strict tickets (`strictUncertain` retains capacity until authoritative reconciliation, `readiness_contract.go:709-726`) is only reachable through the strict path, which daily traffic does not take until I Task 5. Mark **WAITS-FOR I Task 5** and, after it lands, add: cancel mid-stream on a strict route, then confirm the slot is released only after `TestPerformanceObservationCancellation`-style evidence (engine idle or qualified completion), never by elapsed time.

**Verify:** Records for 1, 2, 4 with the terminal event, `found` values and the follow-up request status. Pass: `cancelled` rows, immediate re-admission. Fail conditions as stated.

### Task 10 — Telemetry is accurate or reports unknown — READY-NOW

**Files:** none (procedure). Reads `docs/performance-telemetry.mdx`, `performance.go:798-817,884-918`.

**Depends on:** 4, 8, 9.

**Complexity:** judgment

**Do:** On MSI (and once on PC3):
1. Streaming accuracy: send five `weak` streaming turns with `stream_options.include_usage` on (OpenCode default per `plans/b-opencode-shapes.md`). From each raw SSE capture (OpenCode debug log or inference-dispatcher control) note llama-server's final `usage.completion_tokens` and `timings.predicted_per_second`. Open the performance view (`T>` node detail, `v`). Expected: sample count 5, `outputTokens` sum equal to the sum of `completion_tokens`, "Engine-reported decode" within rounding of the duration-weighted mean of the five `predicted_per_second` values (formula in the document), coverage `5/5`. Fail: any count that is not the engine's count, or a rate derived from wall clock (the document forbids it).
2. Unknown: send two non-streaming turns (`stream:false` via inference-dispatcher). Expected: samples recorded with `--` for prefill/decode rates (nonstream has unknown protocol evidence), transport `completed`, protocol incomplete not asserted. Fail: a numeric rate shown for a nonstream sample.
3. Honest outcomes: after T8 step 1 and T9 step 1 the view shows one `failed` and one `cancelled` in the outcome counts, and neither contributes to decode coverage (coverage denominator grows, numerator does not).
4. Origin: the Mac shows no sample for peer executions (T5 step 6).
5. Freshness: stop PC3's service; on the Mac the PC3 summary moves from Fresh to Stale/Unknown with the last-seen age, and never shows a synthetic value.

**Verify:** One table per host: sample count, token sum match, rate match, unknown markers, outcome counts. Pass: exact token sums, rate within 1 token/s of the computed mean, `--` on nonstream, outcomes present. This supplies the "Native MSI llama.cpp ... PC3 llama.cpp evidence ... remain pending" items named in `docs/performance-telemetry.mdx`.

### Task 11 — OpenCode shapes: tools and `store:false` — interim READY-NOW; strict qualification WAITS-FOR I Task 5 and B Tasks 2-6

**Files:** none (procedure). Reads `local_qualification.go:290-293,326,403-406,732`, `tiers.go:76` (tools need), `plans/b-opencode-shapes.md` Tasks 3, 5, 7.

**Depends on:** 4. External: the owner's OpenCode config as frozen in T1.

**Complexity:** judgment

**Do:** READY-NOW observations on the non-strict daily path (MSI, OpenCode on MSI, medium model):
1. Plain text: one turn with the unmodified generic provider (no `compatibility.supportsStore`). Capture the request OpenCode sent (its debug log) and record whether it contains `store:false`. Expected on the daily path: the proxy forwards the body unchanged (no strict validation; `audit-pa-j-b.md` UNV-4) and llama-server answers 200 or a 400 naming `store`. Record which. A 400 here is a compatibility blocker for the owner, not a procedure fail.
2. Tool call: a turn that makes OpenCode call one tool (for example a file read in a scratch directory). Expected: the first request carries `tools:[...]`, llama-server answers a `tool_calls` delta, the second request carries the assistant `tool_calls` message plus a `tool` role message and (per the B inventory) `tools:[]` or the definitions again; both get 200 and the turn completes. The proxy's `requestNeeds` derives the tools capability only from a non-empty `tools` array (`tiers.go:76`), so a `tools:[]` history request must still route to the same node. Fail: 404 tier on the second request (capability mismatch) or a 400 from llama-server (template lacks tool support; record the template name from `/props`).
3. Reasoning replay: if the medium model emits `reasoning_content`, the next turn carries it back; record whether llama-server accepts it (B owner question).

WAITS-FOR part (do not attempt before the gates): after I Task 5 lands (strict path for daily hybrid aliases) and B Tasks 2-6 land (shape evidence), run `audit-pa-j-b.md` checklist item 5 exactly: plain turn -> 200 with no `store` member reaching the engine; hand-crafted `store:true` -> 400 `strict-qualification` with the storage message and zero engine hits; an undeclared tools request against a profile without tools evidence -> 400 before any engine effect. Mark **WAITS-FOR I Task 5, B Tasks 2-6**.

**Verify:** Three records with the request top-level key list (never message content), the engine status and the proxy status. Pass today: turns complete, blockers recorded. The milestone clause closes only when the WAITS-FOR part passes.

### Task 12 — Paused peer skipped by the router — WAITS-FOR PA Task 2 (interim observation READY-NOW)

**Files:** none (procedure). Reads `proxy.go:1356-1361`, `admission.go:886-897`, `audit-pa-j-b.md` RED-1 and checklist item 6.

**Depends on:** 5.

**Complexity:** mechanical

**Do:** Today: pause PC3 locally, route five `weak` requests from the Mac. Expected at `8a63866`: each request may first dial PC3 and receive 503 `X-PAIR-Admission: paused`, `Retry-After: 30`, then fail over to MSI (RED-1); the TUI on the Mac shows PC3 `Paused (local)`. Record the PC3 proxy's admission log count. After PA Task 2 lands, repeat expecting zero requests at PC3 (PA Task 2 `TestPAPausedPinnedPeerEligibility` counters contract) and re-entry within one successful refresh after resume. Fail (today): an inference that runs on the paused node (admission bypass), or a request that fails instead of failing over when MSI is eligible.

**Verify:** Two counts (attempts at PC3, successes at MSI) before and after PA Task 2. Clause status: WAITS-FOR PA Task 2; the safety property (no inference on a paused node) is READY-NOW.

### Task 13 — Residency: on-demand serving now, automatic residency later — WAITS-FOR SR 5a-ii/5b/5c (on-demand part READY-NOW)

**Files:** none (procedure). Reads `plans/context/sr5-rescope.md` (branch `plan/cloud-investigations`) Decisions and Q1-Q3; `audit-ne-sr.md` item 7.

**Depends on:** 3.

**Complexity:** mechanical

**Do:** READY-NOW: on MSI with both models cold, send `medium`: the child loads on demand (router autoload), 200; `GET /models` now shows it `loaded`; readback `observed` carries its context/slots. Send `weak`: second child loads; if `models-max` is 1 the first child is evicted (record which). Declare `residentModels` = [weak model] through `policy:set` and confirm it persists and peer-projects (XA Serving contract) and that `audit-ne-sr.md` item 7 holds (reconcile-resident with exact context -> `unsupported-load`, over/under -> `unqualified`, zero `/models/load` calls). WAITS-FOR SR 5a-ii/5b/5c: automatic load of the declared resident after restart, hard protection from idle unload and the four `modelConfiguration` states; SR 6b owns its process test. Mark **WAITS-FOR SR 5a-ii, 5b, 5c (and 6b for the process gate)**.

**Verify:** Load/unload sequence record with `GET /models` snapshots. Pass today: on-demand first use works, declaration persists, resident execution refuses as documented.

### Task 14 — Evidence record and milestone verdict — READY-NOW

**Files:** none (procedure; the evidence directory is outside any repository).

**Depends on:** 2-13.

**Complexity:** mechanical

**Do:** Produce one summary table: clause -> task -> READY-NOW/WAITS-FOR -> PASS/FAIL/UNVERIFIED -> evidence file. Include the firewall before/after diffs and the survivor scans from T1, T2, T7. State the SHA and the OpenCode versions. Record every Windows Firewall prompt with its executable. Hand the owner questions below with the observations that inform them (T3 item 6, T7 item 4, T11 items 1-3). Nothing is committed; if the owner wants the summary in the repository it goes to a `plan/` branch by a separate instruction.

**Verify:** Every task 2-13 has a row; no row says PASS without an evidence file; no prompt, body, PIN, key, host name, address or user name appears in any file.

## Verification (whole-plan)

- Order: T1 -> T2 (installed PAIR stopped) -> T3 (installed PAIR running, owner present) -> T4 -> T5 -> T6 -> T7 -> T9 -> T8 -> T10 -> T11 -> T12 -> T13 -> T14. T6 and T7 can swap; T12/T13 can run any time after T5/T3.
- Machine placement: PC3 runs all `services/tests` selectors in T2 and its own live checks; MSI runs unit `-race` and its live checks; the Mac is origin/client only and never runs `services/tests` (not even `-list`, because `TestMain` builds binaries, `services/tests/main_test.go:41` per XA plan) and never starts an engine.
- Milestone verdict rule: the milestone is "proven" only when every READY-NOW task passes and each WAITS-FOR task's interim observation passes; the WAITS-FOR clauses stay open with their named package tasks: I 4b + SC-b (pause-resume), I 5 + B 2-6 (strict OpenCode shapes), PA 2 (zero-attempt skip), SR 5a-ii/5b/5c (automatic residency).
- Container evidence used while writing this plan (no cases run): `services/tests` `go vet ./...` exit 0; `go test -list . .` printed 180 names including every selector in T2; the audit reports' module unit runs at the same SHA are cited, not repeated.

## Do not

- Do not start any engine, load any model or change any preset on a serving host before T1 item 6 is confirmed by the owner and the host's live approval is recorded; never on the Mac.
- Do not run `services/tests` with `./...`, `make test`, or any selector from the "Gates to respect" list; do not run `-race` on PC3; do not run two `go test` invocations on one machine at once.
- Do not bind, probe or free 8080, 8081+, 11434, 11435, 1234 or 14318-14323 to make a check pass; do not create firewall rules from the procedure; do not run from temporary executable paths.
- Do not edit presets, qualification records, policy or settings to make a readback match; a mismatch is a finding.
- Do not record prompts, messages, response bodies, PINs, keys, host names, addresses or user names; record key names, statuses, headers, codes and log categories only.
- Do not treat a passing fake-process selector as closure of a live clause, or a live pass as closure of a WAITS-FOR clause.
- Do not reword the milestone: "delay or pause-resume" is not satisfied by a clean error, and "accurate-or-unknown" is not satisfied by a plausible estimate.

## If stuck

- A Windows Firewall prompt appears: record the executable path; continue only if the owner answered; otherwise stop that task and report the path for the elevated `rules` run.
- `strictAdmission` never lists a frozen model (T3 step 4): read the broker's strict-readiness reason; check `contentDigest` equality and that the qualification `contextTokens` equals the per-slot value exactly (`audit-ne-sr.md` LOW-6). Do not lower the preset.
- Readback flips to `mismatch` during T3 step 6: capture the reason text and the engine-manager sweep log; this is `audit-ne-sr.md` UNVERIFIED-1 evidence, re-apply settings to clear it and continue.
- Tier 404 from the Mac in T5: confirm the peer's policy `revision` on the Mac's `engine:models` projection matches the peer's local `policy:get`; wait one more refresh; if roles still differ, record the scanner log category and stop T5.
- A client (TUI/desktop) does not reconnect in T7: do not restart the client; capture its log and the service `service/status`; apply the installation's reviewed recovery with the owner.
- OpenCode gets a 400 from llama-server in T11: record the top-level keys and the engine's error `code`, not the message content; that is an owner/B blocker, not a procedure fail.
- After two failed attempts at any step, stop that task, keep the evidence, and continue with tasks that do not depend on it.

## Owner questions

1. Headless Windows: `nvpair-service` autostarts from the per-user Run key and needs an interactive login (`services/nvpair-service/README.md:95-108`). Does "headless service with Windows service-process restart" require a session-less (SCM/boot) service, or is a logged-in locked session acceptable for the daily milestone?
2. Identity after a service-side restart: an adopted (not owned) llama-server run publishes no `serving` identity by NE design. Should the acceptance require a controlled `engine:restart` after every service/broker restart, or should NE re-own a surviving router it launched before (a plan amendment)?
3. Daily path and B: until I Task 5 lands, OpenCode traffic never enters strict qualification, so B Task 7's `store:false` fix and future tools/replay evidence do not apply to daily requests. Is the owner content to accept the non-strict path for the daily milestone (with T11's observation of what llama-server does with `store:false`), or should I Task 5 be pulled forward as a milestone prerequisite?
4. Pause-resume: with SC-b absent from every planned package, is "one honest `stream_interrupted` error then `[DONE]`" acceptable as the day-one interrupted-stream behaviour, or does the milestone wait for SC-b?
5. T3 step 6 / `audit-ne-sr.md` UNVERIFIED-1: if a 10-minute run shows transient `mismatch` latches, should NE Part C be amended so transport-level read failures expire proof instead of latching withdrawal?
6. Frozen records (SR Q3): confirm per host the preset values (`parallel`, `sleep-idle-seconds`, `models-max`), the resident selection (Q1 default: PC3 weak resident; MSI weak resident, medium on demand) and that B qualification `contextTokens` will be recorded exactly at the preset per-slot value.
