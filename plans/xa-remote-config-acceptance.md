<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution plan: XA — remote configuration cross-process acceptance harness

Planned against `fork/groundwork` at commit `17727f1c4786ece2524a4ac8d50c132f71da8388`.
Every `path:line` below refers to that commit.

## Goal

`services/tests` carries one acceptance suite, `TestXARemoteConfiguration*`,
that drives real PAIR binaries (broker, engine-manager, cluster-manager, proxy,
nvpair-service) through the same JSON-RPC methods the desktop and terminal
clients use for remote configuration, and proves with fake engines only:

- **status**: a paired origin reads the target's policy projection, its
  effective remote-management grant, its switch status and its engine intent,
  and each read equals what the target's own broker reports;
- **conflict**: a stale `expectedPersistedRevision` on a remote policy set and a
  stale `expectedRevision` on a remote engine switch are refused with the typed
  conflict (`-32010`), nothing is persisted, and the switch refusal is readable
  as a `rejected` receipt;
- **veto**: a target paused on the node itself refuses a relayed resume and a
  remote switch with `-32011`, and an engine saved Off on the node itself
  refuses a remote start with `-32011`;
- **roles**: an allowed peer can set `roles` (a device field), the target
  persists and advertises them in its `engine:models` projection, and a denied
  peer cannot (`-32012`, bytes unchanged);
- **serving/residency**: an allowed peer can set `servedModels` (absent, empty
  and subset are distinct) and `residentModels`; the target persists, projects
  and advertises them, and a running proxy enforces the served set at its
  facade (`503`, `X-PAIR-Admission: not-served`);
- **idempotent switch**: an identical remote switch retry returns the stored
  receipt without a second attempt, a reused request ID with a different payload
  is refused, and a vetoed request replays as its stored `rejected` receipt;
- **native path**: the same `policy:set`, `cluster:*` and `engine:intent` calls
  reach the broker unchanged through `nvpair-service`, the process the terminal
  interface attaches to.

Residency *execution* (automatic resident loads) is explicitly deferred until
SR 5a–5c land; the suite covers residency *declaration* now. Live engines are
never started by this suite.

## Change scope

- **Intent / Scope:** Bounded improvement. Test code and one documentation
  section only; no product code changes. New file
  `services/tests/remote_configuration_test.go`; one fixture extraction in
  `services/tests/engine_switch_test.go` and one in
  `services/tests/service_harness_test.go` (with its single existing caller in
  `services/tests/remote_management_test.go` updated); one paragraph in
  `services/readme.md`.
- **Basis:** the fixtures already in `services/tests` — `policyJointFixture`
  (`services/tests/policy_relay_joint_test.go:232`), `managementFixture`
  (`services/tests/remote_management_test.go:118`), `switchProcessFixture`
  (`services/tests/engine_switch_test.go:88`), the pinned two-node mesh
  `newTask2Fixture` (`services/tests/reverse_peer_control_task2_test.go:91`)
  and `serviceFixture` (`services/tests/service_harness_test.go:81`).
- **Compatibility:** no wire change, no manifest change
  (`services/shared/testbin/programs/tests.txt` already registers every
  program these fixtures copy: `policy-joint/*`, `products/*`,
  `service-bundle-1/*`), no new listener roles, so no new Windows firewall
  rules. Existing tests keep passing; two of them are refactored without
  behaviour change.
- **Non-goals:** no live Ollama/LM Studio/llama.cpp/Strata runs; no OpenCode
  driving; no desktop or TUI code; no new JSON-RPC methods; no residency
  execution coverage before SR 5a–5c; no mDNS-dependent discovery assertions;
  no `-race` on PC3.

## Context the executor needs

Verified in the Go source at the SHA above.

**Module and gating.**
- `services/tests` is its own module (`services/tests/go.mod:1`, `go 1.25.0`).
  `TestMain` (`services/tests/main_test.go:41`) builds every product binary
  into a `testbin.MainDir` and redirects `HOME`/`XDG_CONFIG_HOME`/`APPDATA`/
  `LOCALAPPDATA`/`USERPROFILE` (`main_test.go:169-175`), so children never
  touch a real install. `go test -list` still runs `TestMain`, so a Mac
  `-list` compiles the binaries and lists names without starting fixtures.
- Platform gate pattern to copy: `strataContractPlatform`
  (`services/tests/strata_contract_test.go:276-286`) skips on darwin and on
  Windows fails unless `GOFLAGS` contains `nvpair-test-exec` and
  `NVPAIR_TEST_BIN_DIR` is absolute. `TestOBOffOrigin`
  (`services/tests/off_origin_test.go:20-21`) documents the same policy in a
  comment only.
- Windows fixed-path wrapper and serialized runs: `services/readme.md:215-246`.

**Joint fixture (origin engine-manager ↔ target broker+engine-manager).**
- `newPolicyJointFixture` (`policy_relay_joint_test.go:242-345`): target node
  `policy-joint-target`, origin node `policy-joint-origin`, mutual pins,
  origin granted remote management through a real cluster-manager
  (`managementGrantFixture`, `remote_management_test.go:71-113`). Every
  bundled engine is isolated to an inert loopback port
  (`jointIsolateEngines`, `policy_relay_joint_test.go:476-489`). Target broker
  runs with `--proxy-engines=` (no proxy, `policy_relay_joint_test.go:365`),
  so `policy:get` activation there is `pending` and **remote switches are
  refused as `unroutable` before any pause check**
  (`services/nvpair-ui-broker/engineswitch.go:88-91`, called at
  `engineswitch.go:551`): do not use this fixture for switch cases.
- Seeded policy: version 2, revision `"12"`, availability `paused` with no
  `pausedBy` (counts as a local pause), one local tier entry
  (`policy_relay_joint_test.go:293-316`). `jointExpectedPeer`
  (`policy_relay_joint_test.go:445-451`) spells the projection with
  `Roles: map[string]nodepolicy.Role{}`.
- Helpers: `f.originCall(method, params)` adds `nodeId` and calls the origin
  (`:378-386`); `f.rpc` fails on error (`:388-395`); `f.http(path, params,
  status)` posts to the target's `ec` port with the origin's pinned TLS
  (`:397-413`); `f.assertPolicy` compares the broker's `policy:get` and the
  stored file with `f.want` (`:415-425`); `f.assertUnchanged(before)`
  (`:427-433`); `f.startTarget`/`f.startOrigin` restart halves (`:347-376`).
- `newManagementFixture` (`remote_management_test.go:125-149`) wraps a joint
  fixture, adds a second pinned peer `management-denied` whose grant is
  `false`, exposes `f.denied *http.Client`, `f.joint`, `f.cm` (real
  cluster-manager). `managementHTTP(t, client, port, method, path, params,
  status)` (`:202-207`) asserts the 409 body is exactly
  `{"code":"remote_management_denied","error":"Remote management is blocked on this device. Change permission locally on that device."}`
  (`:180-200`). `f.permission(t, client, allowed)` reads
  `GET /v1/remote-management?...&local=true` (`:209-230`).

**Typed refusals and codes.**
- Policy relay: HTTP 400 ↔ `-32602`; 409 JSON `{"error","code"}` with
  `policy_conflict` ↔ `-32010`, `local_veto` ↔ `-32011`,
  `remote_management_denied` ↔ `-32012` (`services/nvpair-engine-manager/spec.md:531-545`).
  Origin message format: `remote node policy: HTTP <status>: <text>`
  (`policy_relay_joint_test.go:195`). Conflict text:
  `policy conflict: the persisted revision is <rev>, not the expected 0; reload the policy and review it before saving`
  (`policy_relay_joint_test.go:149`). Local-pause resume text:
  `this node was paused on the node itself; resume it there`
  (`policy_relay_joint_test.go:207`, origin code `-32011` at `:215`).
- Switch over `ec`: `-32602` → 400; `-32010` → 409 JSON
  `{"error","code":"configuration_conflict"}`; `-32011` and `-32012` →
  `writeCodedError` 409 (`services/nvpair-engine-manager/switchremote.go:178-200`).
  Broker refusal strings: `Engine configuration changed; reload before switching.`
  (`engineswitch.go:544`), `A local pause vetoes remote switching.`
  (`engineswitch.go:556`). `settings.SwitchConfigurationConflictCode = -32010`,
  `SwitchRemoteVetoCode = -32011`, kinds `configuration_conflict`,
  `remote_veto`, `reused_id` (`services/shared/enginesettings/switch.go:85-104`).
- Switch idempotency: `acceptEngineSwitch`
  (`engineswitch.go:461`) looks the request ID up **before** any refusal
  (`:487-492`, again at `:517-522`): same caller and payload hash → returns the
  stored receipt with no error, whatever its phase; different caller or
  payload → `reused_id`. A refusal stores a `rejected`, terminal receipt
  (`:533-534`, `engineswitch_store.go:142-146`).
- Remote-start veto: `engine:remote-start` params are `{node, engine}`
  (`services/nvpair-engine-manager/remote.go:22-29`, field `node`, not
  `nodeId`); a start vetoed by local pause or local Off answers HTTP 409
  `local_veto` and reaches the origin as `-32011` (`spec.md:547-575`).
  `codeLocalVeto = -32011` (`services/nvpair-engine-manager/remoteclient.go:41`).
- Remote management read: `engine:remote-management-get {nodeId}` →
  `{version:1, allowed, revision, state}` (`remote_management.go:102-131`).

**Policy shapes.**
- `nodepolicy.Policy` fields incl. `ServedModels` (`servedModels,omitzero`: nil =
  unrestricted, non-nil empty = serves nothing), `ResidentModels`
  (`residentModels,omitempty`), `Roles map[string]Role`
  (`services/shared/nodepolicy/policy.go:94-131`). `ServedModels.MarshalJSON`
  errors on nil (`:169-177`), so build drafts with `omitzero` in effect (plain
  `json.Marshal` of a struct) and never marshal a bare nil `ServedModels`.
- Validation: roles need a known tier and a valid engine/model only
  (`policy.go:546-553`); `residentModels` must be a subset of a non-nil
  `servedModels` and each entry needs a `Profile` with positive
  `contextTokens`, `maxOutputTokens`, `maxConcurrent`, `memoryBytes` and a
  `qualification {id, revision}` (`policy.go:557-566`, `Profile` at
  `:337-363`, `QualificationRef` at `:367-370`).
- `MergePeerSet` (`projection.go:145-189`): a peer may change device fields,
  roles included; `tiers`/`tierPolicy` must equal the projection; `routing`
  refused; `availability`/`pausedBy`/`revision` restored from stored.
- Advertisement: `nodepolicy.Advertisement{servedModels,omitzero; revision;
  roles; pausedBy}` (`advertise.go:67-79`). The broker pushes it to
  engine-manager as `engine:node-projection {policy, pausedLocally}` after
  every persisted policy or availability change (`spec.md:463`;
  `services/nvpair-engine-manager/nodeprojection.go:31-37`), and
  `engine:models` returns it as `policy` (`services/nvpair-engine-manager/models.go:54`).
  The push is asynchronous relative to the `policy:set` reply, so poll.
- Broker `policy:get`/`policy:set`/`node:set-availability` contract:
  `services/nvpair-ui-broker/README.md:785-792` (`policy:set` result carries
  `activation` `active|pending|error` and `activeRevision`; activation
  constants at `services/nvpair-ui-broker/policy.go:85-87`).

**Proxy served enforcement.**
- Admission checks `modelServed` first (`services/nvpair-proxy/admission.go:374-377`);
  both sides are normalized (`services/nvpair-proxy/policy.go:258-269`), so
  `switch-model` and `switch-model:latest` are equivalent for Ollama.
- Admission runs only for the **self** candidate (`services/nvpair-proxy/http_handler.go:1262-1270`);
  a node whose address is the proxy's own facade port is self
  (`services/nvpair-proxy/proxy.go:684-693`). A manual node at the engine's
  own port (as `holdRequest` adds, `engine_switch_test.go:417`) bypasses
  admission. Rejection: 503, header `X-PAIR-Admission` (`nodepolicy.AdmissionHeader`)
  = reason, body `{"error":"this node is not accepting the request (not-served)","code":"admission","reason":"not-served"}`
  (`admission.go:851-868`).

**Switch fixture (fake engines, real broker+manager+proxy).**
- `newSwitchProcessFixture` (`engine_switch_test.go:106-186`): the test binary
  is the engine image (`init` at `:31-39`, `runSwitchEngine` `:41-86`: answers
  `/api/generate` with `{"model":"switch-model","done":true,...}` after
  notifying `/inflight`, which **blocks until `f.unblock()`**
  (`:136-145`, `:188`)). `ollama` is the running source on `f.sourcePort`
  with its facade on `f.facadePort`; `llamacpp` is the target. Helpers:
  `f.rpc`/`f.rpcOK` (keeps `engine:switch-changed` pushes, `:210-246`),
  `f.status` (`:248`), `f.request(id)` (`:257`), `f.terminal(id)` (`:262`),
  `f.pid(engine)` (exactly one start, `:283`), `assertSwitchProcessStopped`
  (`:293`). No policy file is seeded, so the broker's policy is
  `nodepolicy.Default()` with revision `"0"`.
- Pinned remote switch: `testEngineSwitchPinned` (`engine_switch_test.go:588-690`)
  composes `newTask2Fixture` (two real cluster-managers, observers, fake
  Ollama HTTP engine, reverse session) with `newSwitchProcessFixtureAt` on
  node B's base and ports (`:590-611`), then drives
  `settings.RemoteSwitchMethod` / `RemoteSwitchStatusMethod` from node A's
  engine-manager (`origin.call`). Its lines 590-611 are the fixture; the rest
  is one scenario. The fixture is what Tasks 6 and 7 reuse.

**Engine intent.** `engine:subscribe`, `engine:intent`,
`engine:intent-changed`; helpers `subscribeOBIntent`, `obIntent`,
`decodeOBIntent`, `assertOBIntent`, `assertOBStoredIntent`
(`off_origin_test.go:135-238`). Local `engine:stop` records `local` origin;
`ec` stop records `peer` (`off_origin_test.go:22-38`).

**Service bundle.** `newServiceFixture` (`service_harness_test.go:98-149`)
copies `nvpair-service`, broker and scanner into bundle slot 1 (one slot:
`service_harness_test.go:36-39`, callers are serial); `f.copyBinary`
(`:151`), `f.inventory.paths`, `f.env`, `f.ports` (`reserveFixtureServicePorts`,
`fixture_ports_test.go:54`), `f.start()`/`f.readyClient()` (`:226`, `:258`),
`c.mustCall(t, id, method, params)` (`:547`), `f.policyPath()` (`:661`).
`TestRemoteManagementPersistsAcrossSeparateProcessRestarts`
(`remote_management_test.go:1110-1288`) already turns that fixture into a
three-peer remote-configuration target (lines `1110-1159`: empty-directory
scanner, engine-manager and cluster-manager copies, inert engine isolation,
minted peer/allowed/target identities, pins, members, seeded policy, two
pinned HTTP clients) before `f.start()` at `:1160`.

**Residency at this SHA.** The engine-manager side (`engine:reconcile-resident`,
`engine:dispatch-resident-request/-reply`, `spec.md:112-227`) and the proxy
side (`node/prepare-resident`, `node/dispatch-resident`, `node/finish-resident`,
`node/observe-resident`, `node/set-residency`, `services/nvpair-proxy/policy.go:43-71`)
are landed. The broker only rejects client-originated
`engine:reconcile-resident` (`services/nvpair-ui-broker/client_requests.go:49`)
and pushes `node/set-residency` with loaded inventory
(`services/nvpair-ui-broker/policy.go:1374`); **no broker code calls
`node/prepare-resident`, `node/finish-resident` or `engine:reconcile-resident`**
(verified by `rg -n -F` over `services/nvpair-ui-broker`), so a
`residentModels` declaration produces no load. That is the SR 5a–5c gap.

**Not verified.** Whether the local `go` toolchain on the executor's machine
is ≥ 1.25 (required by the module); whether PC3's firewall rule set already
covers the `policy-joint`, `products` and `service-bundle-1` programs (it must,
since the reused tests run there today — confirm with
`scripts/windows-test-bin.ps1 list` before the first run).

## Decisions (settled — do not revisit)

- Routine XA cases use fake engines only. Not live Ollama/LM Studio/llama.cpp,
  because the suite is a Windows gate that must run unattended and must never
  change real engine state.
- Reuse the existing `services/tests` fixtures; extract, do not duplicate.
  Not a new harness, because `policyJointFixture`, `switchProcessFixture`,
  `newTask2Fixture` and `serviceFixture` already own cleanup, port
  reservation and process inventory for exactly these processes.
- Mac only vets and lists; every execution is a PC3 Windows gate. Not Mac
  execution, because the Mac never runs `services/tests` cases (fleet rule).
- Residency execution waits on SR 5a–5c. Not driving `node/prepare-resident`
  on the proxy directly from the test, because that would test a path the
  product does not take and would be deleted when the broker relay lands.
- Desktop and TUI are not touched. Not adding bridge or TUI assertions,
  because XA is the services slice; the clients relay these methods unchanged.

## Decisions made while writing this plan

- Gate: `xaPlatform(t)` copies `strataContractPlatform` exactly (skip on
  darwin; Windows requires the fixed-path wrapper). Linux runs unguarded.
- Test naming: one top-level function per outcome,
  `TestXARemoteConfiguration{Status,Conflict,Veto,Roles,Serving,Switch,Service}`,
  so PC3 selectors are stable and a failing area is dispatchable alone.
- Switch cases (conflict, veto, idempotency) run on the pinned two-node
  fixture extracted from `testEngineSwitchPinned`, direct route only, because
  the joint fixture has no proxy and refuses switches as `unroutable`.
- Served enforcement is proven on the local `switchProcessFixture` by adding a
  manual node at the proxy's own facade port (the `self` candidate), after a
  positive control; no mDNS or node-info discovery is relied on.
- The service-bundle setup of `TestRemoteManagementPersistsAcrossSeparateProcessRestarts`
  is extracted into `newRemoteServiceFixture` in `service_harness_test.go`
  and that test is updated to call it (touches
  `remote_management_test.go`, outside the catalogue's expected list, to avoid
  a 50-line duplicate). `main_test.go` and `service_processes_test.go` need
  no change: every binary XA uses is already built and inventoried there.
- `services/readme.md` gets one XA paragraph in "Cross-process tests"; no new
  doc under `docs/`.

## Tasks

### Task 1 — Platform gate and status acceptance land in a new XA file

**Files:** `services/tests/remote_configuration_test.go` (new)

**Depends on:** none

**Complexity:** mechanical

**Do:**
1. Create the file with the two-line SPDX header, `package tests`.
2. Add `xaPlatform(t *testing.T)` with the body of `strataContractPlatform`
   (`strata_contract_test.go:276-286`), skip message
   `"XA cross-process acceptance is reserved for Windows PC3; no Mac processes"`,
   Windows fatal text
   `"Windows XA fixture requires the configured fixed-path wrapper and NVPAIR_TEST_BIN_DIR"`.
3. Add `TestXARemoteConfigurationStatus`: call `xaPlatform(t)`, then
   `f := newPolicyJointFixture(t)` and four subtests, each `t.Run`:
   - `"policy projection"`: `body := f.rpc(t, "engine:remote-policy-get", map[string]any{}).Result`;
     `assertJointPeer(t, body, f.want)`; decode `struct{ Availability nodepolicy.Availability `json:"availability"` }`
     from `body` and require `== nodepolicy.Paused`; `f.assertPolicy(t)`.
   - `"management grant"`: `m := f.originCall(t, "engine:remote-management-get", map[string]any{})`;
     require `m.Error == nil`; decode
     `struct{ Version int; Allowed bool; Revision string; State string }`
     (json tags `version`, `allowed`, `revision`, `state`) and require
     `Version == 1 && Allowed && State == "allowed" && Revision != ""`.
   - `"switch status"`: `m := f.originCall(t, settings.RemoteSwitchStatusMethod, map[string]any{})`;
     decode `settings.SwitchStatus` as `remote`; call
     `callRPC(t, f.target.in, f.target.msgs, settings.SwitchStatusMethod, map[string]any{}, 10*time.Second)`
     and decode as `local`; require `remote.ConfigurationRevision != "" &&
     remote.ConfigurationRevision == local.ConfigurationRevision &&
     remote.ActiveRequestID == "" && remote.Receipt == nil`.
   - `"engine intent"`: `c := subscribeOBIntent(t, f.target)`;
     `assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), obIntent(false, map[string]engineintent.Origin{}))`.
4. Imports: `encoding/json`, `testing`, `time`, `nvpair-shared/engineintent`,
   `settings "nvpair-shared/enginesettings"`, `nvpair-shared/nodepolicy`.

**Verify:**
- Mac: `cd services/tests && go vet ./... && go test -list 'TestXARemoteConfiguration' .`
  — pass condition: vet prints nothing and the list contains
  `TestXARemoteConfigurationStatus`.
- PC3 (Windows, serialized, wrapper configured):
  `cd services/tests && go test -count=1 -run 'TestXARemoteConfigurationStatus$' .`
  — pass condition: `ok` with all four subtests passed, no skip line.

### Task 2 — Policy conflict acceptance

**Files:** `services/tests/remote_configuration_test.go`

**Depends on:** Task 1

**Complexity:** mechanical

**Do:**
1. Add `TestXARemoteConfigurationConflict` (gate, `f := newPolicyJointFixture(t)`):
   - `before := assertStoredPolicy(t, f.path, f.want)`.
   - `params := map[string]any{"policy": jointExpectedPeer(f.want), "expectedPersistedRevision": "0"}`.
   - HTTP: `body := f.http(t, "/v1/node-policy/set", params, http.StatusConflict)`;
     decode `struct{ Error, Code string }` and require `Code == "policy_conflict"` and
     `Error == "policy conflict: the persisted revision is " + f.want.Revision + ", not the expected 0; reload the policy and review it before saving"`.
   - Origin RPC: `m := f.originCall(t, "engine:remote-policy-set", params)`;
     require `m.Error != nil && m.Error.Code == -32010 && len(m.Result) == 0 &&
     m.Error.Message == "remote node policy: HTTP 409: " + <the same text>`.
   - `f.assertUnchanged(t, before)`.
   - Positive control, so a conflict cannot pass vacuously:
     `draft := jointExpectedPeer(f.want); draft.Idle.UnloadAfterMinutes++`;
     `res := f.rpc(t, "engine:remote-policy-set", map[string]any{"policy": draft, "expectedPersistedRevision": f.want.Revision})`;
     decode `struct{ Policy nodepolicy.PeerPolicy `json:"policy"` }`; require
     `Policy.Revision != "" && Policy.Revision != f.want.Revision`; then
     `f.want.Idle = draft.Idle; f.want.Revision = Policy.Revision; f.assertPolicy(t)`.
   - Stale retry after the control: resend the control params unchanged
     (old revision) through `f.originCall` and require `-32010` again and
     `f.assertPolicy(t)`.
2. Add `net/http` to imports.

**Verify:**
- Mac: `go vet ./...` clean; `go test -list 'TestXARemoteConfiguration' .` lists `TestXARemoteConfigurationConflict`.
- PC3: `go test -count=1 -run 'TestXARemoteConfigurationConflict$' .` — `ok`.

### Task 3 — Local veto acceptance (pause and saved Off)

**Files:** `services/tests/remote_configuration_test.go`

**Depends on:** Task 2

**Complexity:** judgment

**Do:**
1. Add `TestXARemoteConfigurationVeto` (gate, `f := newPolicyJointFixture(t)`).
   The fixture's target is paused without provenance, which counts as a local
   pause (`nodepolicy.Policy.PausedBy` comment, `policy.go:124-128`).
   - `"relayed resume is vetoed"`: `before := assertStoredPolicy(t, f.path, f.want)`;
     `m := f.originCall(t, "engine:remote-availability-set", map[string]any{"state": "available"})`;
     require `m.Error != nil && m.Error.Code == -32011 &&
     m.Error.Message == "remote node policy: HTTP 409: this node was paused on the node itself; resume it there"`;
     `f.assertUnchanged(t, before)`.
   - `"remote start is vetoed while paused"`:
     `task1Eventually(t, "remote start refused by local pause", 15*time.Second, func() bool {`
     `m := callRPC(t, f.origin.in, f.origin.msgs, "engine:remote-start", map[string]any{"node": "policy-joint-target", "engine": "ollama"}, 30*time.Second)`;
     `return m.Error != nil && m.Error.Code == -32011 })`. The poll covers the
     fail-closed 503 the target answers before its first projection
     (`spec.md:552-553`). (`engine:remote-start` takes `node`, not `nodeId`;
     do not use `f.originCall`.)
   - `"local resume then local Off vetoes remote start"`:
     `local := callRPC(t, f.target.in, f.target.msgs, "node:set-availability", map[string]any{"state": "available"}, 15*time.Second)`;
     require `local.Error == nil`; `c := subscribeOBIntent(t, f.target)`;
     `c.call(t, "engine:stop", map[string]string{"engine": "ollama"})`;
     `c.assertIntent(t, obIntent(false, map[string]engineintent.Origin{"ollama": engineintent.Local}))`;
     then `task1Eventually(t, "remote start refused by local Off", 15*time.Second, func() bool {`
     send the same `engine:remote-start` as above and return
     `m.Error != nil && m.Error.Code == -32011 })`;
     finally `assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), obIntent(false, map[string]engineintent.Origin{"ollama": engineintent.Local}))`
     and `assertOBStoredIntent(t, filepath.Join(jointAppDir(f.base), "engine-bin", "engine-state.json"), false, engineintent.Local)`.
   - `"peer Off cannot clear a local Off"`: `f.http(t, "/v1/engines/stop", map[string]any{"engine": "ollama"}, http.StatusOK)`;
     `c.assertIntent(t, obIntent(false, map[string]engineintent.Origin{"ollama": engineintent.Local}))`.
   Order the subtests exactly as listed; each depends on the previous state.
2. Add `path/filepath` to imports.

**Verify:**
- Mac: vet clean; `-list` shows `TestXARemoteConfigurationVeto`.
- PC3: `go test -count=1 -run 'TestXARemoteConfigurationVeto$' .` — `ok`,
  four subtests passed. Also rerun the neighbour it shares semantics with:
  `go test -count=1 -run 'TestOBOffOrigin$' .` — `ok`.

### Task 4 — Roles acceptance (allowed peer sets, target advertises, denied peer cannot)

**Files:** `services/tests/remote_configuration_test.go`

**Depends on:** Task 3

**Complexity:** mechanical

**Do:**
1. Add helper `xaEngineModelsPolicy(t, p *interopProcess) *nodepolicy.Advertisement`:
   `callRPC(t, p.in, p.msgs, "engine:models", map[string]any{}, 10*time.Second)`,
   fail on error, decode `struct{ Policy *nodepolicy.Advertisement `json:"policy"` }`
   and return `Policy` (nil before the first projection push).
2. Add `TestXARemoteConfigurationRoles` (gate, `f := newManagementFixture(t)`,
   `j := f.joint`):
   - `"allowed peer sets roles"`: `draft := jointExpectedPeer(j.want)`;
     `draft.Roles = map[string]nodepolicy.Role{"weak": {Engine: "ollama", Model: "joint-local"}}`;
     `res := j.rpc(t, "engine:remote-policy-set", map[string]any{"policy": draft, "expectedPersistedRevision": j.want.Revision}).Result`;
     decode `struct{ Policy nodepolicy.PeerPolicy `json:"policy"` }`; require
     `reflect.DeepEqual(Policy.Roles, draft.Roles)` and a new revision;
     `j.want.Roles = draft.Roles; j.want.Revision = Policy.Revision; j.assertPolicy(t)`.
   - `"target advertises roles"`: `task1Eventually(t, "roles projection", 10*time.Second, func() bool {`
     `adv := xaEngineModelsPolicy(t, j.target); return adv != nil && adv.Revision == j.want.Revision && adv.Roles["weak"].Engine == "ollama" && adv.Roles["weak"].Model == "joint-local" })`.
   - `"denied peer cannot change roles"`: `before := readPolicyBytes(t, j.path)`;
     `draft.Roles = map[string]nodepolicy.Role{"strong": {Engine: "ollama", Model: "joint-local"}}`;
     `managementHTTP(t, f.denied, j.ports[1], http.MethodPost, "/v1/node-policy/set", map[string]any{"policy": draft, "expectedPersistedRevision": j.want.Revision}, http.StatusConflict)`;
     `j.assertUnchanged(t, before)`.
   - `"peer cannot change tiers alongside roles"`: `draft := jointExpectedPeer(j.want); draft.Tiers["weak"][0].Model = "different-local"`;
     `m := j.originCall(t, "engine:remote-policy-set", map[string]any{"policy": draft, "expectedPersistedRevision": j.want.Revision})`;
     require `m.Error != nil && m.Error.Code == -32602 && m.Error.Message == "remote node policy: HTTP 400: a paired node cannot change the tiers"`;
     `j.assertPolicy(t)`.
3. Add `reflect` to imports.

**Verify:**
- Mac: vet clean; `-list` shows `TestXARemoteConfigurationRoles`.
- PC3: `go test -count=1 -run 'TestXARemoteConfigurationRoles$' .` — `ok`.

### Task 5 — Serving and residency declaration, and proxy served enforcement

**Files:** `services/tests/remote_configuration_test.go`

**Depends on:** Task 4

**Complexity:** judgment

**Do:**
1. Add `TestXARemoteConfigurationServing` (gate) with two independent halves.
2. Half A, declaration (`f := newManagementFixture(t)`, `j := f.joint`):
   - `"explicit empty served set persists"`: `draft := jointExpectedPeer(j.want)`;
     `draft.ServedModels = nodepolicy.ServedModels{}`; set through
     `j.rpc(t, "engine:remote-policy-set", ...)`; decode the projection and
     require `Policy.ServedModels != nil && len(Policy.ServedModels) == 0`;
     update `j.want.ServedModels = nodepolicy.ServedModels{}` and revision;
     `j.assertPolicy(t)`; require the stored bytes
     `readPolicyBytes(t, j.path)` contain `"servedModels":[]`
     (`bytes.Contains`).
   - `"served subset with resident declaration"`: `draft := jointExpectedPeer(j.want)`;
     `draft.Profiles = append(draft.Profiles, nodepolicy.Profile{Name: "xa-resident", Engine: "ollama", Model: "joint-local", ContextTokens: 8192, MaxOutputTokens: 1024, MaxConcurrent: 1, MemoryBytes: 1, Qualification: &nodepolicy.QualificationRef{ID: "xa-qual", Revision: "1"}})`;
     `draft.ServedModels = nodepolicy.ServedModels{{Engine: "ollama", Model: "joint-local"}}`;
     `draft.ResidentModels = nodepolicy.ResidentModels{{Engine: "ollama", Model: "joint-local"}}`;
     set via `j.rpc`; require the returned projection's `ServedModels` and
     `ResidentModels` deep-equal the draft's; update `j.want` (Profiles,
     ServedModels, ResidentModels, Revision); `j.assertPolicy(t)`.
   - `"target advertises the served set"`: `task1Eventually(..., 10*time.Second, func() bool {`
     `adv := xaEngineModelsPolicy(t, j.target); return adv != nil && adv.Revision == j.want.Revision && reflect.DeepEqual(adv.ServedModels, j.want.ServedModels) })`.
   - `"resident outside served is refused"`: `draft := jointExpectedPeer(j.want)`;
     `draft.ServedModels = nodepolicy.ServedModels{{Engine: "ollama", Model: "other"}}`
     (resident stays `joint-local`); `m := j.originCall(t, "engine:remote-policy-set", ...)`;
     require `m.Error != nil && m.Error.Code == -32602 && m.Error.Message == "remote node policy: HTTP 400: residentModels must be a subset of servedModels"`;
     `j.assertPolicy(t)`.
   - `"denied peer cannot change the served set"`: `before := readPolicyBytes(t, j.path)`;
     a draft clearing `ServedModels` to `nodepolicy.ServedModels{}` and
     `ResidentModels` to nil, posted with `managementHTTP(t, f.denied, j.ports[1], http.MethodPost, "/v1/node-policy/set", ..., http.StatusConflict)`;
     `j.assertUnchanged(t, before)`.
3. Half B, enforcement (`s := newSwitchProcessFixture(t)`), as subtest
   `"proxy enforces served set at its facade"`:
   - `s.unblock()` first (the fake engine's `/api/generate` blocks on the
     controller otherwise).
   - Register the proxy itself as a manual node so the request resolves to the
     self candidate:
     `s.rpcOK(t, "ollama-proxy:node/add-manual", map[string]any{"id": "xa-self", "host": "127.0.0.1", "port": s.facadePort, "addresses": []string{"127.0.0.1"}, "models": []string{"switch-model"}})`;
     `s.rpcOK(t, "ollama-proxy:node/select", map[string]string{"id": "xa-self"})`.
   - Helper closure `generate() (*http.Response, []byte)`: POST
     `http://127.0.0.1:<s.facadePort>/api/generate` with body
     `{"model":"switch-model","stream":false,"prompt":"fixture"}`, 30 s
     client timeout, read ≤ 64 KiB.
   - Helper closure `setServed(served nodepolicy.ServedModels)`: `get := s.rpcOK(t, "policy:get", map[string]any{})`;
     decode `struct{ Policy nodepolicy.Policy `json:"policy"` }`;
     `p := Policy; p.ServedModels = served`;
     `set := s.rpcOK(t, "policy:set", map[string]any{"policy": p, "expectedPersistedRevision": Policy.Revision})`;
     decode `struct{ Policy nodepolicy.Policy; ActiveRevision string `json:"activeRevision"`; Activation string `json:"activation"` }`;
     then `task1Eventually(t, "policy activation", 15*time.Second, func() bool {`
     re-read `policy:get`, decode the same struct and return
     `Activation == "active" && ActiveRevision == <set revision> })`.
   - Sequence and assertions:
     1. positive control: `resp, body := generate()`; require `resp.StatusCode == 200`
        and body decodes with `done == true`.
     2. `setServed(nodepolicy.ServedModels{})`; `resp, body := generate()`;
        require `resp.StatusCode == http.StatusServiceUnavailable`,
        `resp.Header.Get(nodepolicy.AdmissionHeader) == "not-served"`,
        `resp.Header.Get("Retry-After") != ""`, and body decodes to
        `{Error: "this node is not accepting the request (not-served)", Code: "admission", Reason: "not-served"}`.
     3. `setServed(nodepolicy.ServedModels{{Engine: "ollama", Model: "other"}})`;
        `generate()` → 503 with `X-PAIR-Admission: not-served`.
     4. `setServed(nodepolicy.ServedModels{{Engine: "ollama", Model: "switch-model"}})`;
        `generate()` → 200, `done == true`.
     5. `s.pid(t, "ollama")` still reports exactly one start and
        `s.mu.Lock(); len(s.starts["llamacpp"]) == 0; s.mu.Unlock()`.
4. Add `bytes`, `io`, `strings` to imports as used.

**Verify:**
- Mac: vet clean; `-list` shows `TestXARemoteConfigurationServing`.
- PC3: `go test -count=1 -run 'TestXARemoteConfigurationServing$' .` — `ok`,
  all subtests passed; then `go test -count=1 -run 'TestEngineSwitchProcessCompletesAndRetries$' .`
  — `ok` (proves the switch fixture is unchanged by this task).

### Task 6 — Extract the pinned remote-switch fixture

**Files:** `services/tests/engine_switch_test.go`

**Depends on:** none

**Complexity:** mechanical

**Do:**
1. In `engine_switch_test.go`, add:
   ```go
   // pinnedSwitchFixture is node B's broker-supervised switch stack on the
   // task2 mesh, driven from node A's engine-manager over the pinned ec route.
   type pinnedSwitchFixture struct {
       mesh   *task2Fixture
       f      *switchProcessFixture
       origin *task1Process
       ec     *task1Observer
   }
   ```
   and `func newPinnedSwitchFixture(t *testing.T, reverse bool) *pinnedSwitchFixture`
   whose body is exactly lines `590-611` of `testEngineSwitchPinned` (mesh,
   stop B's manager, ports, `newSwitchProcessFixtureAt`, origin, the
   `t.Cleanup` that stops origin/scanner/f.p, directory, `ec`, the
   `if !reverse` interrupt and the `NotifyNodes` send), returning the struct.
2. Replace those lines in `testEngineSwitchPinned` with
   `p := newPinnedSwitchFixture(t, reverse); mesh, f, origin, ec := p.mesh, p.f, p.origin, p.ec`
   so the remaining body (`:612-689`) is unchanged.
3. Add `func (p *pinnedSwitchFixture) remoteStatus(t *testing.T, requestID string) settings.SwitchStatus`:
   `origin.call(t, settings.RemoteSwitchStatusMethod, settings.SwitchStatusRequest{NodeID: p.mesh.b.uuid, RequestID: requestID})`,
   fail on `Error != nil`, decode and return.
4. Add `func (p *pinnedSwitchFixture) request(t *testing.T, id string) settings.SwitchRequest`:
   `settings.SwitchRequest{NodeID: p.mesh.b.uuid, RequestID: id, SourceEngine: "ollama", TargetEngine: "llamacpp", ExpectedRevision: p.remoteStatus(t, "").ConfigurationRevision}`.

**Verify:**
- Mac: `go vet ./...` clean; `go test -list 'TestEngineSwitchProcessPinned' .` lists both pinned tests.
- PC3: `go test -count=1 -run 'TestEngineSwitchProcessPinned(Direct|Reverse)$' .` — `ok` for both (behaviour unchanged).

### Task 7 — Remote switch conflict, veto and idempotency acceptance

**Files:** `services/tests/remote_configuration_test.go`

**Depends on:** Task 5, Task 6

**Complexity:** judgment

**Do:**
1. Add `TestXARemoteConfigurationSwitch` (gate; `p := newPinnedSwitchFixture(t, false)`;
   `f, origin := p.f, p.origin`). Run the subtests in this order (shared
   state):
   - `"stale revision is a typed conflict and a rejected receipt"`:
     `req := p.request(t, "xa-switch-stale"); req.ExpectedRevision = "e3v1:" + strings.Repeat("0", 64)`;
     `m := origin.call(t, settings.RemoteSwitchMethod, req)` — **note**
     `task1Process.call` (`reverse_peer_control_test.go:145`) fatals on an RPC
     error; use `id := origin.send(t, settings.RemoteSwitchMethod, req, true); m := task2Response(t, origin, id)`
     instead. Require `m.Error != nil && m.Error.Code == settings.SwitchConfigurationConflictCode`.
     Then `st := p.remoteStatus(t, "xa-switch-stale")`; require
     `st.Receipt != nil && st.Receipt.Phase == settings.SwitchRejected && st.Receipt.Terminal && st.Receipt.Error != nil && st.Receipt.Error.Kind == settings.SwitchErrorConfigurationConflict && st.Receipt.Caller == p.mesh.a.uuid && st.Receipt.NodeID == p.mesh.b.uuid`.
     `f.pid(t, "ollama")`; require no `llamacpp` start.
   - `"local pause vetoes a remote switch"`:
     `f.rpcOK(t, "node:set-availability", map[string]string{"state": "paused"})`;
     `req := p.request(t, "xa-switch-veto")`; send as above; require
     `m.Error.Code == settings.SwitchRemoteVetoCode` (`-32011`);
     `st := p.remoteStatus(t, "xa-switch-veto")`; require phase `rejected`,
     terminal, `Error.Kind == settings.SwitchErrorRemoteVeto`,
     `Error.Code == -32011`. Require no `llamacpp` start.
   - `"vetoed request replays as its stored receipt"`: resend the identical
     `req` with `origin.call` (no error expected); decode
     `settings.SwitchResult`; require `reflect.DeepEqual(result.Receipt, *st.Receipt)`.
   - `"local resume then a fresh request completes"`:
     `f.rpcOK(t, "node:set-availability", map[string]string{"state": "available"})`;
     `req := p.request(t, "xa-switch-ok")`; `origin.call(t, settings.RemoteSwitchMethod, req)`
     and decode `settings.SwitchResult` as `accepted`; require
     `accepted.Receipt.Caller == p.mesh.a.uuid && accepted.Receipt.NodeID == p.mesh.b.uuid`;
     `var terminal settings.SwitchStatus; task1Eventually(t, "remote terminal receipt", 45*time.Second, func() bool { terminal = p.remoteStatus(t, "xa-switch-ok"); return terminal.Receipt != nil && terminal.Receipt.Terminal })`;
     require `terminal.Receipt.Phase == settings.SwitchCompleted && terminal.Receipt.Error == nil`;
     `assertSwitchProcessStopped(t, f.pid(t, "ollama")); f.pid(t, "llamacpp")`.
   - `"identical retry returns the terminal receipt"`: `origin.call` the same
     `req`; decode `settings.SwitchResult`; require
     `reflect.DeepEqual(replay.Receipt, *terminal.Receipt)`; require
     `p.ec.count(p.mesh.a.uuid, http.MethodPost, settings.SwitchApplyPath)` equals
     the count taken before this subtest plus 1 (the retry traversed `ec` once
     and was answered from the journal), and `f.pid(t, "llamacpp")` still
     exactly one start.
   - `"reused request ID with another payload is refused"`:
     `bad := req; bad.SourceEngine, bad.TargetEngine = "llamacpp", "ollama"`
     (the engines are part of the payload hash; `ExpectedRevision` alone would
     also change the hash, but swapping engines keeps the request otherwise
     valid); send with `origin.send`/`task2Response`; require `m.Error != nil`;
     `after := p.remoteStatus(t, "xa-switch-ok")`; require
     `reflect.DeepEqual(after.Receipt, terminal.Receipt)`; require
     `f.pid(t, "llamacpp")` still one start and `f.mu`-guarded
     `len(f.starts["ollama"]) == 1`.
2. Add `strings` to imports if Task 5 did not.

**Verify:**
- Mac: vet clean; `-list` shows `TestXARemoteConfigurationSwitch`.
- PC3: `go test -count=1 -run 'TestXARemoteConfigurationSwitch$' .` — `ok`,
  six subtests passed, and the broker journal
  (`filepath.Join(p.mesh.b.base, "engine-switch-operations.json")`, read with
  `settings.DecodeSwitchJournal(readLiveDurableFile(...))` inside the last
  subtest) holds exactly the three request IDs `xa-switch-stale`,
  `xa-switch-veto`, `xa-switch-ok`.

### Task 8 — Extract the remote-configuration service bundle fixture

**Files:** `services/tests/service_harness_test.go`, `services/tests/remote_management_test.go`

**Depends on:** none

**Complexity:** mechanical

**Do:**
1. In `service_harness_test.go`, add:
   ```go
   // remoteServiceFixture is a serviceFixture whose bundle also runs
   // engine-manager and cluster-manager, with an empty-directory scanner, inert
   // engines, a pinned target identity and two pinned peers: one denied and one
   // allowed remote management. Start it with f.start().
   type remoteServiceFixture struct {
       *serviceFixture
       enginePath, clusterPath string
       peer, allowed           *http.Client
       permissionPath          string
   }
   ```
   and `func newRemoteServiceFixture(t *testing.T) *remoteServiceFixture`
   whose body is exactly `remote_management_test.go:1111-1159` (from
   `f := newServiceFixture(t)` through the `allowedClient` cleanup), returning
   `&remoteServiceFixture{serviceFixture: f, enginePath: enginePath, clusterPath: clusterPath, peer: client, allowed: allowedClient, permissionPath: filepath.Join(f.clusterDir, clustertrust.RemoteManagementFile)}`.
   Keep the identity names `management-service-target`,
   `management-service-peer`, `management-service-allowed` and the seeded
   policy revision `"12"` unchanged.
2. In `TestRemoteManagementPersistsAcrossSeparateProcessRestarts`, replace
   lines `1111-1159` with
   `rf := newRemoteServiceFixture(t); f, client, allowedClient, enginePath, clusterPath := rf.serviceFixture, rf.peer, rf.allowed, rf.enginePath, rf.clusterPath`
   and delete the local `permissionPath :=` at `:1163` in favour of
   `permissionPath := rf.permissionPath`. Nothing else in the test changes.
3. Move the imports the extracted block needs (`net`, `net/http`,
   `net/http/httptest`, `strconv`, `nvpair-shared/clustertrust`,
   `nvpair-shared/nodepolicy`) into `service_harness_test.go`; remove any that
   become unused in `remote_management_test.go` (`go vet` reports them).

**Verify:**
- Mac: `go vet ./...` clean; `go test -list 'TestRemoteManagementPersistsAcrossSeparateProcessRestarts' .` lists it.
- PC3: `go test -count=1 -run 'TestRemoteManagementPersistsAcrossSeparateProcessRestarts$' .` — `ok` (unchanged behaviour).

### Task 9 — Native (nvpair-service) remote configuration acceptance

**Files:** `services/tests/remote_configuration_test.go`

**Depends on:** Task 7, Task 8

**Complexity:** mechanical

**Do:**
1. Add `TestXARemoteConfigurationService` (gate; `rf := newRemoteServiceFixture(t); rf.start(); c := rf.readyClient()`).
   Request IDs for `c.mustCall` are strings; use `"xa1"`, `"xa2"`, … in order.
   - `"local grants through the service"`:
     `view := decodeResult[managementSnapshot](t, *c.mustCall(t, "xa1", "cluster:get-remote-management", mustMarshal(t, map[string]any{})))`;
     `c.mustCall(t, "xa2", "cluster:set-remote-management", mustMarshal(t, map[string]any{"peerUuid": "management-service-peer", "allowed": false, "expectedRevision": view.Revision}))`;
     re-read and set `management-service-allowed` to `true` with the fresh
     revision (`"xa3"`, `"xa4"`).
   - `"local policy edit with roles and served set"`:
     `before := decodeServicePolicy(t, c.mustCall(t, "xa5", "policy:get", nil))`;
     `draft := before; draft.Roles = map[string]nodepolicy.Role{"weak": {Engine: "ollama", Model: "xa-model"}}; draft.ServedModels = nodepolicy.ServedModels{{Engine: "ollama", Model: "xa-model"}}`;
     `m := c.mustCall(t, "xa6", "policy:set", mustMarshal(t, map[string]any{"policy": draft, "expectedPersistedRevision": before.Revision}))`;
     `snap := decodeServicePolicySnapshot(t, m)`; require
     `snap.Policy.Revision != before.Revision && reflect.DeepEqual(snap.Policy.Roles, draft.Roles) && reflect.DeepEqual(snap.Policy.ServedModels, draft.ServedModels) && snap.Activation == "pending"`
     (no proxy in this bundle); `assertStoredPolicy(t, rf.policyPath(), snap.Policy)`.
   - `"peers see the service-made policy"`: allowed peer
     `data := managementHTTP(t, rf.allowed, rf.ports.engineControl, http.MethodPost, "/v1/node-policy/get", map[string]any{}, http.StatusOK)`;
     decode `struct{ Policy nodepolicy.PeerPolicy `json:"policy"` }` and
     require `reflect.DeepEqual(Policy, snap.Policy.PeerProjection())`; denied
     peer (`rf.peer`) reads the same `/v1/node-policy/get` with 200 and gets
     409 on `/v1/node-policy/set` with
     `{"policy": snap.Policy.PeerProjection(), "expectedPersistedRevision": snap.Policy.Revision}`;
     `assertStoredPolicy(t, rf.policyPath(), snap.Policy)` afterwards.
   - `"engine intent through the service"`:
     `m := c.mustCall(t, "xa7", "engine:intent", nil)`; `decodeOBIntent(t, m.Result)`
     must validate; `c.mustCall(t, "xa8", "engine:stop", mustMarshal(t, map[string]string{"engine": "ollama"}))`;
     `task1Eventually(t, "service intent push", 10*time.Second, func() bool {`
     decode `c.mustCall(t, "xa9", "engine:intent", nil).Result` and return
     `OffByEngine["ollama"] == engineintent.Local })`;
     then the allowed peer's start is vetoed:
     `status, _, body := managementHTTPResponse(t, rf.allowed, rf.ports.engineControl, http.MethodPost, "/v1/engines/start", map[string]any{"engine": "ollama"})`;
     require `status == http.StatusConflict` and the body decodes to
     `struct{ Code string `json:"code"` }` with `Code == "local_veto"`.
     (`managementHTTP` is not used here: its 409 assertion expects the
     management-denied body.)
   - `"clean stop"`: `assertServiceStoppedReply(t, c.mustCall(t, "xa10", "service/stop", nil))`;
     `rf.waitChild(rf.service, serviceStopTimeout)`; `rf.assertStopped()`.
2. No new imports beyond those already present.

**Verify:**
- Mac: vet clean; `-list` shows `TestXARemoteConfigurationService`.
- PC3: `go test -count=1 -run 'TestXARemoteConfigurationService$' .` — `ok`,
  and the log contains `fixture_cleanup endpoint=... owned_running=0 endpoint_rejected=true`.

### Task 10 — Document the XA suite in the services README

**Files:** `services/readme.md`

**Depends on:** Task 9

**Complexity:** mechanical

**Do:**
1. In `services/readme.md`, after the paragraph ending `after any change to a
   JSON-RPC method, payload, or notification.` (`services/readme.md:259-260`)
   and before `### Tests that skip themselves` (`:262`), insert:

   ```markdown
   #### Remote configuration acceptance (XA)

   `tests/remote_configuration_test.go` is the acceptance suite for remote
   configuration between paired nodes: status reads, revision conflicts,
   local veto, roles, served and resident model declarations, served-set
   enforcement at a proxy facade, idempotent engine switches, and the same
   calls through `nvpair-service`. It drives real binaries with fake engines
   only and never starts an installed engine. It skips on macOS and, on
   Windows, requires the fixed-path wrapper above. Run one area at a time on a
   Windows gate machine:

   ```bash
   cd tests
   go test -count=1 -run 'TestXARemoteConfigurationStatus$' .
   go test -count=1 -run 'TestXARemoteConfigurationConflict$' .
   go test -count=1 -run 'TestXARemoteConfigurationVeto$' .
   go test -count=1 -run 'TestXARemoteConfigurationRoles$' .
   go test -count=1 -run 'TestXARemoteConfigurationServing$' .
   go test -count=1 -run 'TestXARemoteConfigurationSwitch$' .
   go test -count=1 -run 'TestXARemoteConfigurationService$' .
   ```

   `go test -run 'TestXARemoteConfiguration' .` runs them all. On macOS,
   `go vet ./...` and `go test -list 'TestXARemoteConfiguration' .` are the
   only checks; the cases themselves do not run there. Automatic resident
   loads are not covered until the broker's resident relay lands.
   ```
2. Do not edit `README.md` at the repository root (public scope rule) or any
   generated file.

**Verify:**
- Mac: `node scripts/spdx-headers.mjs` from the repository root exits 0;
  `grep -n 'TestXARemoteConfigurationService' services/readme.md` prints one
  line; the seven `-run` selectors in the README match the seven top-level
  functions in `services/tests/remote_configuration_test.go`
  (`grep -c '^func TestXARemoteConfiguration' services/tests/remote_configuration_test.go` prints `7`).

## Deferred until SR 5a–5c land (not a task now)

Residency *execution* acceptance: an allowed peer declares `residentModels`,
the broker relays `node/prepare-resident` → `engine:reconcile-resident` →
`engine:dispatch-resident-request/-reply` → `node/finish-resident`, and the
fake engine observes exactly one `POST /api/generate` with
`{model, stream:false, keep_alive, options:{num_ctx}}` and no prompt
(`services/nvpair-engine-manager/spec.md:199-212`), followed by a `GET /api/ps`
readback. Known-now constraints the future task must honour: the switch
fixture's fake engine (`engine_switch_test.go:41-86`) needs an `/api/ps`
route and a loaded flag toggled by the keep-alive load; `residentModels`
requires a qualified `Profile` (Task 5 half A shows the shape); the
engine-manager refuses loads for a non-Ollama family, so the case must target
`ollama`; the proxy's qualification registry must hold the profile's
`qualification {id, revision}` for the same engine identity, which is
proxy-side setup that SR 5a–5c define. Re-plan that task from the SR 5 spec
sections once they are in `fork/groundwork`; its file is
`services/tests/remote_configuration_test.go` and it depends on Task 5.

## Verification (whole-plan)

Mac (vet and list only):
```bash
cd services/tests && go vet ./... && go test -list 'TestXARemoteConfiguration|TestEngineSwitchProcessPinned|TestRemoteManagementPersistsAcrossSeparateProcessRestarts' .
node scripts/spdx-headers.mjs        # from the repository root
```
Pass: vet silent; the list prints the seven `TestXARemoteConfiguration*`
names plus the three refactored neighbours; SPDX check exits 0.

PC3 (Windows, wrapper configured, PAIR stopped, one `go test` at a time):
```bash
cd services/tests
go test -count=1 -run 'TestXARemoteConfiguration' .
go test -count=1 -run 'TestEngineSwitchProcessPinned(Direct|Reverse)$|TestRemoteManagementPersistsAcrossSeparateProcessRestarts$|TestOBOffOrigin$|TestPolicyRelayJoint$|TestRemoteManagement' .
```
Pass: both commands print `ok`, no `--- SKIP` line for any `TestXA*` case
(a skip is not a pass), and `scripts/windows-test-bin.ps1 list` shows no new
program paths beyond those already ruled.

MSI (race evidence, optional for XA): `go test -race -count=1 -run 'TestXARemoteConfiguration' .`
— pass: `ok` with no `WARNING: DATA RACE`.

Desktop checks are unaffected (no desktop file changes);
`npm --prefix desktop run service-contracts:check` is not required because no
JSON-RPC contract changed.

## Do not

- Do not start, install or pull a real engine; do not set `NVPAIR_LIVE_*`.
- Do not add test-only branches, environment flags or new RPC methods to any
  product binary.
- Do not add programs to `services/shared/testbin/programs/tests.txt` or new
  listener roles; reuse the registered bundles.
- Do not run `services/tests` cases on the Mac; do not run `-race` on PC3.
- Do not use `f.originCall` for `engine:remote-*` engine methods: they take
  `node`, not `nodeId`.
- Do not use the joint fixture for engine switches (no proxy → `unroutable`).
- Do not add a manual node at the engine port to test served enforcement; it
  bypasses admission.
- Do not change `TestRemoteManagementPersistsAcrossSeparateProcessRestarts`
  or `testEngineSwitchPinned` beyond the extraction described.
- Do not edit the root `README.md`, `desktop/docs/services-api.md`,
  `CHANGELOG.md` or `services/versions.json`.
- Do not log policy documents, prompts, PINs or key material in test output
  beyond what the existing helpers already log.

## If stuck

- A fixture refuses to start on Windows with `a previous process may still be
  running`: stop every `nvpair-*` process the previous run left, remove the
  wrapper's stale `.lock` files only after confirming no process uses the
  destination (`services/readme.md:240-246`), rerun once. If it recurs,
  report with the `fixture_started`/`fixture_cleanup` log lines.
- Task 5 half B positive control fails (not a 200 with `done:true`): check
  that `s.unblock()` ran before the request and that the manual node's port is
  `s.facadePort`, not `s.sourcePort`. If the response is a 503 with an
  `X-PAIR-Admission` reason other than `not-served`, record the reason and
  stop; the served assertion must not be weakened to any 503.
- Task 7 stale-revision subtest fails with an `unroutable` or `busy` kind:
  the pinned fixture's proxy did not front both engines; confirm
  `waitEngineProxyReady` passed in `newSwitchProcessFixtureAt` logs and
  report rather than relaxing the expected kind.
- Task 4/5 advertisement polls time out: read `engine:models` once more and
  report the `policy.revision` seen versus expected; do not extend beyond
  30 s.
- Any `-32012` where `-32011`/`-32010` is expected means the fixture's grant
  is missing: verify `managementGrantFixture` ran (joint) or
  `managementSet(..., true)` ran (mesh) and report; never widen the accepted
  code set.
- After two failed attempts at any task, stop and report the task number,
  the exact command, and the first failing assertion with its log context.

## Owner questions

1. Residency execution: the plan defers the automatic-load acceptance until
   SR 5a–5c land (no broker relay exists at this SHA). Do you want an interim
   case that drives the proxy's `node/prepare-resident` → `engine:reconcile-resident`
   hops directly from the test, knowing it exercises a path the product does
   not take and will be replaced? Default: no.
2. Live evidence: the daily milestone asks for real OpenCode/XA evidence on
   MSI and PC3 with intended engines On. This plan never starts a real
   engine. Do you want an opt-in `NVPAIR_XA_LIVE=1` variant of
   `TestXARemoteConfigurationSwitch` that runs against the installed PAIR
   bundle and real llama.cpp/Strata engines (it would stop and start engines
   on that machine), or does the live evidence stay a manual checklist?
   Default: manual checklist, no live test.
3. Pause provenance in the native case: Task 9 pauses nothing; a relayed
   resume of a service-made local pause is covered by the joint fixture only.
   Is one service-path pause/resume cycle (which restarts no process but
   drains the node) worth adding to the PC3 gate time? Default: not added.
