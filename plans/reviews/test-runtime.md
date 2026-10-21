<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Review: services/tests and module suite wall-clock cost (test-runtime)

Inspected: `fork/groundwork` at `7e270f10ced402325dc9e1efece54991e9bbbc08` (the branch had not moved). Every `path:line` below is at that SHA; paths are relative to `services/` unless they start with `tests/` or name another top-level directory.

Verdict: **RED 0 UNVERIFIED 3 LOW 15** (LOW = the twelve ranked speedups in section 4 plus the three LOW notes in section 6.2). No test was found that passes while the behaviour it names is broken. The known switch-revision flake did not reproduce (25/25 passes); its most likely mechanism is named below. The gate time is dominated by four mechanisms, in order: (1) the broker suite rebuilding Go fixture programs with an empty build cache, (2) negatives proved by waiting out product constants (30 s detect timeout, 5 s stop grace, 10 s teardown budget, 10 s "still clustered"), (3) fixed `time.Sleep` barriers in the cross-process suite, and (4) the irreducible per-test cost of starting and stopping real binaries with no parallelism anywhere (`t.Parallel()` count is zero in all four suites).

## 1. Platform and method

- Platform: Linux x86-64 container, 4 vCPU, 15 GiB, Go toolchain 1.25.0 (selected by `go.mod` `go 1.25.0` through `GOTOOLCHAIN=auto`; host Go 1.24.7). Not Windows: timings below are relative evidence of mechanisms, not Windows predictions.
- Runs: `go test -count=1 -json ./...` in `nvpair-engine-manager`, `nvpair-proxy`, `nvpair-ui-broker`, then `tests/`, serially (one at a time, mirroring the Windows gate). Per-test elapsed from the `pass`/`fail`/`skip` events; wall clock from a timeline file.
- The handoff's exception was used: `tests/` ran in this isolated container only. No engine or model was started; every engine in these suites is a fake process or the test binary re-executed.
- Linux failures (environmental, see section 7) inflate some `tests/` durations to their timeout. Where a test failed here, the table marks it and the mechanism comes from code, not from the measurement.

Wall clock (Linux, warm module/build cache):

| suite | wall | package elapsed | top-level tests | entries incl. subtests | pass / fail / skip (top-level) |
|---|---:|---:|---:|---:|---|
| nvpair-engine-manager | 252 s | 236.9 s | 778 | 2661 | 753 / 19 / 6 |
| nvpair-proxy | 57 s | 50.0 s | 714 | 3091 | 707 / 4 / 3 |
| nvpair-ui-broker | 136 s | 131.2 s | 663 | 1462 | 663 / 0 / 0 |
| tests/ | 708 s | 705.1 s | 180 | 444 | 158 / 18 / 4 |
| total | 1153 s (19.2 min) | | | | |

The handoff's Windows figures (6 + 1 + 2.5 + 14 = 23.5 min) are consistent with this ratio; the Windows machine is roughly 1.2-1.4x slower on the process-heavy suites.

Of the `tests/` 705 s, passing tests account for 337 s, Linux-failing tests (mDNS/IPv6 and DNS, timing out) for 326 s, and one 30 s skip. The passing-only distribution is the one to reason from: 62 tests under 0.1 s, 37 tests at 1-2 s (one broker fixture start plus teardown each), and a long tail of 10 tests above 10 s that hold 55% of the passing time.

## 2. Where the time goes, per suite (top 10/40 shares)

Full top-40 tables are in the appendix.

**nvpair-engine-manager** (leaf tests, sum 230 s): top 10 = 71.0 s (30.8%), top 40 = 113.0 s (49.1%). 571 of 778 top-level tests finish under 0.1 s (4.6 s total). The suite is one package, so everything is serial.

**nvpair-ui-broker** (top-level, sum 130.5 s): top 10 = 85.4 s (65.5%), top 40 = 118.3 s (90.7%). Four tests that compile a Go fixture inside the test hold 52 s.

**nvpair-proxy** (leaf, sum 48.5 s): top 10 = 28.1 s (57.9%), top 40 = 45.1 s (93%). 680 of 714 top-level tests are under 0.1 s. Lowest priority; mechanisms are product timeouts (`idleClientWriteTimeout`, transport cancel bounds) exercised at real time.

**tests/** (top-level, sum 693.6 s; passing-only 337 s): top 10 = 411.6 s (59.4%) but six of those ten failed here on environment; among passing tests the top 10 are 243 s = 72% of passing time.

## 3. Mechanisms behind the expensive tests (VERIFIED unless marked)

### 3.1 nvpair-ui-broker: `go build` inside tests with an empty build cache (52 s of 131 s)

- `engine_manager_ports_test.go:315-327` runs `go build -modcacherw -o <bin> <source>` per call; `service_ports_test.go:64-70` writes a `main.go` into `t.TempDir()` and calls the same helper. Seven call sites (3 in `engine_manager_ports_test.go`, 4 in `service_ports_test.go`).
- `testmain_test.go:27-28` sets `HOME`, `XDG_CONFIG_HOME`, `APPDATA`, `LOCALAPPDATA`, `USERPROFILE` to a fresh temp dir **before** those builds run. Go derives `GOCACHE` from `os.UserCacheDir()` (`$XDG_CACHE_HOME`, then `$HOME/.cache`) at `cmd/go/internal/cache/default.go:93-100`, and `GOMODCACHE` from `GOPATH` which defaults under `HOME`. Unless the gate shell exports `GOCACHE`/`GOMODCACHE` explicitly, each in-test build recompiles the standard library into an empty cache. Measured: `TestEngineManagerPortsSpawnAndScannerReplay` 17.1 s, `TestEngineManagerPortsCLIRejectsInvalidBeforeWorkerResolution` 14.1 s, `TestServicePortsDefaultPlanKeepsWorkerArgvAndRegistrations` 10.4 s, `TestServicePortsInvalidPlansRejectedBeforeAnyChild` 6.1 s, `TestServicePortsAlternatePlanFlowsThroughSpawnRestartAndReplay` 5.9 s; their leaf subtests are each under 0.5 s, so the parent time is the build. Contrast `tests/main_test.go:160-168`, which sets the same variables **after** its builds precisely for this reason ("go derives its module and build caches from these variables").
- INFERRED: on the Windows gate the same seven builds run with Defender real-time scanning of fresh object files; this is likely a larger share of the 2.5 min there than 40% is here.

### 3.2 nvpair-engine-manager: waiting out product timeouts

| test | measured | mechanism |
|---|---:|---|
| `TestUninstallRetries` | 30.15 s | `remediation_test.go:102-134` lowers `uninstallRetries`/`uninstallBackoff` but not `detectTimeout`; `install.go:274` then calls `waitDetect(engine, false, e.detectTimeout)` with the 30 s default from `executor.go:211`, polling every 500 ms (`lifecycle.go:57`), to prove the binary is still detected. Other tests already shorten it (`install_linux_test.go:65`, `install_windows_test.go:58` set 2 s). |
| `TestLlamaServingColdCommandLoadCancelledByStop` (3 subtests) and `TestLlamaServingCommandExitedParentPipeBound` (3 subtests) | 6 x 5.01 s = 30 s | `actions.go:397` sets `cmd.WaitDelay = stopGrace(Runtime{})`, a hard-coded 5 s (`lifecycle.go:733-744`); the fixture child deliberately holds the output pipe, so every `Stop`/`StopAll`/cancel waits the full `WaitDelay`. The tests bound at `stopGrace(Runtime{}) + 3 s` (`serving_llamacpp_test.go:1354`, `:1477`). No manifest or test seam shortens it because it is not derived from the engine runtime. |
| `TestRemoteSettingsEventsReconnectThroughExistingOwner` | 5.02 s | `watchPeerSettings` scans on a fixed 2 s ticker (`settingsremote.go:173`); the test (`remote_reverse_test.go:681-704`) needs two scans (connect, reconnect). |
| `TestRemoteSettingsEventsRetireChangedAuthority` (5 subtests) | 5 x 1.02 s | the authority re-check goroutine ticks every 1 s (`settingsremote.go:270`). |
| `TestCommandStopFailureRestartsHealthMonitoring` 3.0 s, `TestHealthFlipReportsAndClears` 2.2 s, `TestStopReclaimsOurOrphanOnManagedPort` 2.7 s, `TestStopClearsUnhealthy` 1.3 s, `TestCommandStopRejectsSuccessWhileEndpointIs*` 2 x 2.0 s | ~13 s | `Probe.IntervalS` is an integer number of seconds (`registry.go:252`) converted at `lifecycle.go:1001-1002`, so the smallest health interval a test can request is 1 s; `lifecycle_stop_test.go:123` sleeps 1.5 s "to outlast a health interval". |
| `TestModelIdentityIsNeverSentAfterTheStopItPrecedes` | 2.4 s | `model_identity_test.go:355-358` waits a fixed 2 s for a stop that is expected not to complete (a held publish), then releases it. The later `<-stopped` at `:370` makes a hung stop fail the test, so this is not a vacuous pass. |

Together the rows above are ~84 s of the suite's 237 s.

### 3.3 nvpair-ui-broker: const budgets waited out at real time (another ~45 s)

`proxyCallTimeout = 5 s` (`proxy.go:29`, aliased by `policyActivationTimeout` at `policy.go:75`), `workerStopGrace = 3 s` (`stopwait.go:20`), `teardownBudget = 10 s` (`stopwait.go:56`), `minWorkerGrace` (`stopwait.go:61`), `localServicesWriteTimeout = 5 s` (`reverse_services.go:15`) are `const`, so these tests must spend them: `TestProxySupervisionKeepsPolicySetWithinItsBoundAgainstAnUnresponsiveProxy` 5.0 s (`engineproxy_test.go:1283` bounds at `policyActivationTimeout + 2 s`), `TestPolicyActivationAvailabilityPushFaults` 10.7 s (two 5 s subtests; `policy_test.go:3882`; `setActivationWait` at `policy_test.go:3166-3171` exists but these subtests exercise the hung-push path), `TestWaitForStdinCloseKillsWorkerThatIgnoresShutdown` 5.0 s and `TestWaitForStdinCloseGivesUpWithoutProcess` 3.0 s (`stopwait_test.go:97`, `:74-90`), `TestTeardownBudgetCapsAggregateWait` 6.3 s (`stopwait_test.go:273-295`), `TestLocalServicesWriteTimeoutClosesUnresponsiveChild` 5.0 s (`reverse_services_test.go:209-219`). Switch recovery retries sleep `time.Second << attempt` (`broker.go:682-721`): `TestSwitchRecoveryReplaysJournalOffWithoutCheckpoint` 3.0 s, `TestSwitchRecoveryExhaustionKeepsManagerHiddenAndReportsBlocked` 3.0 s, three `TestSwitchRecoveryRetries*` at 1.0 s.

### 3.4 tests/: fixed sleeps, negatives by waiting, and process start/stop

| test (Linux status) | measured | mechanism |
|---|---:|---|
| `TestPairedDirectoryBootstrap/.../both_routes_unavailable_preserve_authority_and_restoration_recovers` (pass) | 32.0 s of 37.9 s | `paired_directory_bootstrap_test.go:590` waits up to 40 s for "unrefreshed authenticated claim expires during outage": it waits out a real peer-claim expiry. INFERRED: the expiry is a product constant in the engine-manager peer directory; the test has no shorter seam. |
| `TestXARemoteConfigurationServing/proxy_enforces_served_set_at_its_facade` (pass) | 36.4 s | `remote_configuration_test.go:702-760`: generate requests through the facade with a 30 s client timeout and two `task1Eventually` waits (15 s policy activation, 10 s inventory). INFERRED: most of the 36 s is one facade dispatch that waits a product owner/dispatch budget before refusing with 503 `not-served`; `go test -json` buffers non-verbose output so the intra-test split could not be read here. A `-v` run with the time-stamped fixture logs would show it. |
| `TestClusterNoSelfRemoveWhenPeer{Accepts,Unreachable,Left}` (pass) | 3 x 11.1 s | `assertStillClustered` is `time.Sleep(wait)` then one RPC (`cluster_manager_test.go:1338-1344`), called with 10 s at `:1382` and the two siblings. A negative proved by waiting. |
| 20 cluster-manager tests (pass) | 20 x 1.0 s | `time.Sleep(time.Second)` "give both inter-node listeners a moment to bind" at `cluster_manager_test.go:182` and 19 more sites (20 occurrences of that exact call), plus `:692` 600 ms. |
| `TestPerformanceObservationFailedDispatchThenSuccess` (pass) | 17.7 s | `performance_observations_test.go:473-479`: five refused dials with the proxy's real `defaultRetryBackoff` (15 s total) and jitter; the comment documents the 68.75 s bound. |
| `TestDescriptorEngineIsFrontedFromAUserManifest` (pass) | 19.6 s | `enginecatalog_process_test.go:95-129`: `app:ready` then two poll loops (20 s and 30 s deadlines at 100/250 ms). INFERRED: the 14 s before the first fixture log is the broker spawning engine-manager and the catalog/detect pass (`detectTimeout` 30 s default again, section 3.2). |
| `TestReversePeerControlTask1/two_way_workloads...` (pass) | 6.9 s | `reverse_peer_control_test.go:695` sleeps 6 s for the workload-manager discovery tick (`nvpair-workload-manager/manager.go:24` `discoveryInterval = 5 s`); `:789` sleeps 4 s "exceeds the production push timeout" to prove no delivery. |
| `TestLogSetLevelViaRPC` 8.2 s, `TestLogLevelEnvFallback` 5.0 s (pass) | 13.2 s | `log_level_test.go:164` (2 s), `:199-201` (0.2 s + 6 s, "confirm that no debug/info lines show up"), `:281` (5 s, "the first mDNS scan browse runs with a 3s timeout"). Negatives by waiting on the scanner's mDNS cadence. |
| `TestPortInteropBridgeBrokerTeardownKillsStalledWorker` (pass) | 5.3 s | waits the broker's real `workerStopGrace` (3 s) plus bundle teardown; by design. |
| `TestFixtureBrokerLeavesDefaultServicePortsAlone` (pass) | 3.2 s | `fixture_ports_test.go:446` sleeps 3 s "let the scanner enrich self". |
| `TestReversePeerControlTask2` (fail here) | 90.9 s | two 1.5 s sleeps after settings projections (`reverse_peer_control_task2_test.go:592`, `:641`); the rest of the 90 s here is Linux mDNS timeouts, not representative. |
| `TestClusterOutboundPairingAuthenticatedJoinerPinFailure` (pass) | 30.4 s | `cluster_outbound_pairing_test.go:396-398` pumps up to 40 s for the bounded PIN response of an incomplete compensation. INFERRED: the wait is a product bounded submission budget in `nvpair-cluster-manager/outbound_pairing.go`; `pairingHTTPTimeout = 10 s` (`invite.go:26`) and `inviteExpiryInterval = 15 s` (`invite_expiry.go:21`) are the candidate inputs. |

Structural costs, both VERIFIED:

- `tests/main_test.go:75-158` builds twelve product binaries one after another with `go build` (`:183`). Here, with a warm cache, the first test started 14.4 s after `go test` began (3.8 s compiling the test binary, 11 s for the twelve builds). The builds are independent and the Go build cache is concurrency-safe.
- 236 fixture-constructor call sites across the suite start at least one product binary (broker fixtures start 6-8). `jointCopyExecutable` (`policy_relay_joint_test.go:524-533`) copies the whole test binary with `os.ReadFile`/`os.WriteFile` at 28 call sites so fixtures can re-exec it under a fixed path. No `t.Parallel()` anywhere; on Windows `testbin.Dir` (`shared/testbin/testbin.go:114-129`) returns one fixed directory per (module, role) when `NVPAIR_TEST_BIN_DIR` is set, so two tests of the same role cannot overlap even inside one package (the copy helper's error text says so: "a previous process may still be running").

## 4. Ranked speedups (seconds saved here per unit of risk)

Risk scale: A = test-only edit, cannot mask anything; B = adds a package-level variable in product code used only by tests (no behaviour change at default); C = replaces a wait-out negative with a shorter wait or a barrier (could mask a slow failure). "Windows-serial safe" means the change does not need package or test parallelism.

| rank | change | saves here | masks | Windows-serial safe |
|---:|---|---:|---|---|
| 1 | **Broker fixture builds: keep the real build cache and build each fixture program once per package.** In `nvpair-ui-broker/testmain_test.go`, capture `go env GOCACHE`/`GOMODCACHE` before the loop at `:27-28` and pass them to the build command's `Env` in `engine_manager_ports_test.go:322`; wrap `buildEngineManagerPortTestBinary` per `program` in a `sync.Once` map (the scanner fixture source is a package constant at `engine_manager_ports_test.go:329`, `service_ports_test.go:64` can write its `main.go` once). Diff sketch: `command.Env = append(os.Environ(), "GOCACHE="+realGoCache, "GOMODCACHE="+realModCache)` and `fixtureBuilds.Do(program, build)`. | ~45-50 s of 131 s (risk A) | nothing; the fixture binaries are byte-identical per run. Note the `-modcacherw` comment at `:320-321` wanted the temporary GOPATH removable; with the real cache nothing is written under the temp HOME. | yes |
| 2 | **`TestUninstallRetries`: shorten `detectTimeout`** as `install_linux_test.go:65` already does: add `ex.detectTimeout = 2 * time.Second` after `newTestExecutor` at `remediation_test.go:120`. The assertion is on the attempt count and error text, not on the 30 s. | 28 s (risk A) | nothing | yes |
| 3 | **Command `WaitDelay` seam.** `actions.go:397`: `cmd.WaitDelay = commandWaitDelay` with `var commandWaitDelay = stopGrace(Runtime{})`; the two pipe-bound tests set it to 200 ms and bound at `commandWaitDelay + 3 s`. | ~29 s (risk B) | the tests still prove the delay is honoured and bounded, at a different value; a regression that ignores `WaitDelay` is still caught because the child holds the pipe forever. | yes |
| 4 | **Broker const budgets to vars** (`proxy.go:29`, `stopwait.go:20`, `:56`, `reverse_services.go:15`, and the `time.Second << attempt` base at `broker.go:721`), overridden to 100-300 ms by the eleven tests in section 3.3 (with `t.Cleanup` restore; the suite is serial so no race). | ~38 s of 131 s (risk B) | the tests assert "within bound" relationships, which survive scaling; what they lose is the assurance that the production value itself is not absurd, which a one-line const test keeps. | yes |
| 5 | **Peer-settings tickers to vars** (`settingsremote.go:173` and `:270`), set to 50 ms in `remote_reverse_test.go`. | ~8.5 s (risk B) | nothing: the tests wait on channels, not on the tick count. | yes |
| 6 | **`tests/main_test.go`: build the twelve binaries concurrently** (`errgroup`, `GOMAXPROCS` limit). | ~6 s here (risk A); INFERRED larger on Windows where the serial 12-build pass is cold more often | nothing | yes |
| 7 | **Replace the twenty 1 s "let listeners bind" sleeps** in `cluster_manager_test.go` with a barrier: `startCM` already pumps stderr; wait for the cluster-manager's inter-node listener log line or `waitTCP` on the inter-node port (`cluster_data_plane_test.go:209-221` has the helper). | ~18 s (risk A, needs the listener's bound port or log line to be observable; it is logged, see the `splitlisten accept` line in the fixture logs) | nothing; a barrier is stronger than a sleep | yes |
| 8 | **Health probe sub-second interval seam**: either a `var minHealthInterval = time.Second` floor applied at `lifecycle.go:1001-1002` that tests lower, or an unexported `healthIntervalOverride`. Six EM tests use `IntervalS: 1`. | ~8-10 s (risk B) | nothing if the manifest field stays integer seconds for users | yes |
| 9 | **`assertStillClustered` 10 s to a product-informed bound.** The self-remove decision follows a reconcile that the logs show at ~4 s (`cluster_manager_test.go:1338-1344`); waiting two reconcile periods (say 5 s) instead of 10 s keeps the negative meaningful. Better: expose the reconcile count via `cluster:get-node-id` or a log line and wait for "two reconciles observed, still clustered". | 15 s (sleep shortening, risk C) or ~20 s (barrier, risk A) | the sleep version could pass if self-removal became slower than 5 s; the barrier version cannot | yes |
| 10 | **Log-level negatives** (`log_level_test.go:201` 6 s, `:281` 5 s): the scanner has a browse timeout of 3 s and a scan cadence the test waits on. A `--scan-interval` test flag or env on the scanner fixture would cut both waits to ~1.5 s. | ~8 s (risk B/C) | the 6 s window exists to see at least one scan; with a faster cadence the window can shrink without weakening the check | yes |
| 11 | **Avoid re-copying the test binary** (`jointCopyExecutable`, 28 sites): on Windows the fixed role path is stable per package run, so copy once per role (`sync.Once` keyed by target path) and let later fixtures reuse it; the "previous process may still be running" check moves to the first copy. | small here (page cache), INFERRED seconds on Windows per copy | nothing | yes, this is specifically a fixed-path win |
| 12 | **`t.Parallel()` in the module suites** for in-process tests that touch no package-level var and spawn no process (most of the 571 sub-0.1 s EM tests and 575 broker tests). | little: those tests already sum to ~5 s per suite | the risk is in misclassifying a test that mutates `uninstallRetries`-style globals | yes, but low value; the time is in the long tail, not the fast tests |

Not recommended: lowering `-timeout`, skipping on Windows, or sharding `tests/` across packages (fixture roles are shared, see section 5.3).

Sum of ranks 1-9 on this platform: about 190-200 s of the 1153 s gate (17%), with ranks 1-5 alone about 150 s and all of risk A/B. Rank 1 is the only one whose Windows effect is probably larger than measured here.

## 5. Test-impact selection for joins

### 5.1 Map: changed module to `tests/` files

Derived from which `*Bin` each file references directly and through the fixture constructors it calls (`service_harness_test.go` fixtures start service+broker+scanner+proxy+cluster-manager+engine-manager; `startBroker*` fixtures start broker+scanner and whichever workers the broker is pointed at; `startInteropProcess` is generic). Method prefixes are the JSON-RPC namespaces each file drives.

| changed module | `tests/` files that exercise it (binary or RPC surface) |
|---|---|
| `nvpair-ui-broker` | every broker-fixture file: `broker_test`, `broker_management_test`, `broker_supervision_test`, `cluster_data_plane_test`, `cluster_restore_test`, `discovery_relay_test`, `engine_switch_test`, `enginecatalog_process_test`, `fixture_ports_test`, `llamacpp_interop_test`, `model_routing_interop_test`, `models_http_test`, `models_refresh_test`, `off_origin_test`, `ollama_host_alias_test`, `paused_advertisement_test`, `performance_observations_test`, `policy_relay_joint_test`, `port_bridge_test`, `remote_configuration_test`, `remote_management_test`, `scheduler_interop_test`, `serving_envelope_test`, `workload_identity_interop_test`, `workload_interop_test`, plus every service-harness file (`service_*_test`, `paired_directory_bootstrap_test`, `reverse_node_info_test`, `strata_*_test` via the proxy/service fixtures). In practice: the whole suite minus `cluster_manager_test`, `cluster_outbound_pairing_test`, `node_settings_test`, `tls_test`, `errors_test`, `discovery_test`, `remote_engine_test`, `durable_file_test`, `mdns_fixture_*`. |
| `nvpair-engine-manager` | `engine_switch_test`, `off_origin_test` (15 `engine:` calls), `remote_configuration_test` (20), `remote_management_test` (13), `reverse_peer_control_task2_test` (18), `policy_relay_joint_test`, `serving_envelope_test`, `enginecatalog_process_test`, `strata_*_test`, `remote_engine_test`, `paired_directory_bootstrap_test`, `reverse_inference_test`, `reverse_modelcopy_test`, `cluster_data_plane_test`, `fixture_ports_test`, `ollama_host_alias_test`, `service_restart_reconnect_test`, `broker_supervision_test` |
| `nvpair-proxy` | `llamacpp_interop_test`, `model_routing_interop_test`, `performance_observations_test` (11 tests), `secure_inference_test`, `log_level_test`, `workload_identity_interop_test`, `scheduler_interop_test`, `engine_switch_test` (facade dispatch), `remote_configuration_test` (served-set at facade), `serving_envelope_test`, `strata_contract_test`, `strata_facade_test`, `broker_management_test`, `ollama_host_alias_test`, `reverse_inference_test`, `service_routing_listener_test`, `service_policy_test` |
| `nvpair-cluster-manager` | `cluster_manager_test` (20 tests, 134 `cluster:` calls), `cluster_outbound_pairing_test` (8), `cluster_restore_test`, `remote_management_test` (31 `cluster:` calls), `reverse_*_test`, `secure_inference_test`, `paired_directory_bootstrap_test`, `engine_switch_test` (pinned fixture), `fixture_ports_test`, `remote_configuration_test` |
| `nvpair-workload-manager` | `workload_interop_test`, `workload_identity_interop_test`, `cluster_data_plane_test`, `reverse_peer_control_test`, `scheduler_interop_test`, `fixture_ports_test`, `broker_supervision_test`, `performance_observations_test` |
| `nvpair-job-scheduler` | `scheduler_interop_test` |
| `nvpair-node-scanner` | `broker_test`, `ghost_node_test`, `models_http_test`, `models_refresh_test`, `paused_advertisement_test`, `discovery_relay_test`, `port_bridge_test`, `reverse_peer_control_task2_test`, and every broker fixture indirectly |
| `nvpair-node-settings` | `node_settings_test`, `ollama_host_alias_test`, `cluster_restore_test`, `broker_supervision_test` |
| `nvpair-node-info` | `tls_test`, `discovery_relay_test`, `reverse_inference_test`, `reverse_peer_control_task2_test`, `fixture_ports_test`, `broker_supervision_test` |
| `nvpair-errors` | `errors_test`, `reverse_peer_control_test` (10 `errors:` calls), `broker_management_test`, `cluster_data_plane_test`, `fixture_ports_test` |
| `nvpair-manual-nodes` | `broker_management_test`, `broker_supervision_test` |
| `nvpair-service` | all `service_*_test`, `paired_directory_bootstrap_test`, `reverse_node_info_test`, `remote_configuration_test` (Service case), `remote_management_test` |
| `shared/*` | any package under `shared/` that a module imports: run the full suite (section 5.3). `shared/testbin`, `shared/jsonrpc`, `shared/enginesettings`, `shared/nodepolicy` are imported by `tests/` itself. |
| `desktop/` only | none of `tests/` (desktop is a relay); run desktop lint/typecheck/unit tests only. |

### 5.2 How a join chooses

1. `git diff --name-only <base>..<join>` and map each path to its module (first path component under `services/`).
2. Always run: `go vet ./...` and the module unit suite of every changed module; `tests/` files for every changed module from the table; and the **always-on core** regardless of change: `broker_test`, `broker_supervision_test`, `engine_switch_test`, `remote_configuration_test` (the XA acceptance surface), `service_lifecycle_test`, `fixture_ports_test` (the port-isolation guard), and `policy_relay_joint_test`. These cover the broker's supervision of every worker and the JSON-RPC contracts most joins touch; on this platform they are ~55 s.
3. Run the selection with `go test -count=1 -run '^(TestA|TestB|...)$' .` in `tests/`; `TestMain` still builds all twelve binaries (rank 6 above matters more here).
4. Any change under `shared/`, to `programs/tests.txt` or the testbin manifest, to `tests/*_test.go` helpers (`main_test`, `service_harness_test`, `workload_interop_test` `startInteropProcess`, `fixture_ports_test`), or to a JSON-RPC method name/payload in any producer, runs the full `tests/` suite. The common brief's rule that a JSON-RPC change updates producer, relay, consumers, bridge and tests together is exactly the case where the map is unsafe.

Expected effect: a proxy-only or scheduler-only join runs roughly 35-45% of the `tests/` time; a cluster-manager-only join about 45%; a broker join still runs nearly everything (the broker fronts all workers), so broker-lane work (SR 5c, PA 1-2, G3) gains mostly from section 4, not from selection.

### 5.3 Where the map is unsafe

- Shared fixtures: on Windows every test using a given `testbin` role shares one fixed directory; selection does not change that, but a selected subset must not be run in two shells at once (readme: keep machine-wide invocations serialized).
- Cross-module contracts: `engine:settings-projection`, `engine:switch-*`, `policy:*` and `cluster:*` are produced in one module and consumed in two or three others. A change in `nvpair-engine-manager` that alters a notification the broker relays is only caught by `tests/` files in the broker row, which the engine-manager row includes only partially. Use the union of the rows of every module that imports the changed package, not the changed module alone.
- `shared/` changes have no safe subset.
- The map was derived from string occurrences of method prefixes and `*Bin` variables; files that drive a binary only through a helper defined elsewhere (for example `strata_*_test` through `newProxyServiceFixture`) are listed from the constructor-call graph, which is correct today but drifts silently when a helper changes. A generated map (a `go test -list`-time registration of binaries per test, or a `//nvpair:drives` comment convention) would hold better.

## 6. Timing- and order-sensitive tests

### 6.1 The known flake: `TestEngineSwitchProcessRejectsRunningTargetBeforeDrain` (UNVERIFIED)

Reproduction attempt: 25 consecutive runs here, 25 passes (0.7 s each). Not reproduced, so UNVERIFIED; the mechanism below is INFERRED from code.

- Claimed invariant (handoff): the switch request built right after `engine:start` carries the current `ConfigurationRevision`; the observed failure was "Engine configuration changed; reload before switching".
- Code path: `engine_switch_test.go:391-393` starts the target, then `f.request` (`:268-271`) reads `ConfigurationRevision` from `engine:switch-status` and immediately submits. Both the status read (`nvpair-ui-broker/engineswitch.go:1004-1009`) and acceptance (`:527`, refusal at `:545-546`) compute the token in `switchSnapshotLocked` (`:433-458`), which hashes the engine-manager's `SwitchPrepare` configuration (`nvpair-engine-manager/switchexecutor.go:149-191`: intent from desired state, manifest and runtime digest, settings revisions) with the broker's own `record.Snapshot.Revision` per engine substituted at `:452-456`.
- Of those inputs, the only one that moves without a test action is the broker's settings revision. `refreshEngineSettings` runs every 2 s (`enginesettings.go:779-781`) and calls `settingsSnapshotLocked`, which increments `Revision` whenever the launch-derived config differs from the stored record (`enginesettings.go:214-217`, comparing `ServerPort`, `ProxyPort`, `LaunchText`, `Serving` at `shared/enginesettings/settings.go:18-20`). The fixture journal is hand-written (`engine_switch_test.go:177-192`); the first tick after the fixture's `engine:start ollama` and the test's `engine:start llamacpp` is the first observation of the running launch. If that tick lands between the status read and the accept, the token differs and the broker refuses with exactly the observed message. The window is the few ms between two RPCs, which matches "once, then 10/10".
- Alternatives considered and ruled out from code: `Intent` is persisted before `engine:start` replies (`lifecycle.go:123` `setDesiredEnabled` is the last statement of `Start`); the runtime digest only hashes exported fields and the post-start writes at `lifecycle.go:299-370` are to unexported serving fields; `JournalGeneration` only changes on switch-journal writes.
- Probe that would settle it: include the per-engine `SettingsRevision` values (not just the hash) in the `SwitchErrorConfigurationConflict` detail, or run the test with the refresh ticker forced to 50 ms (make `refreshEngineSettings`'s interval a var) and `-count=50`; a reproduction under that change confirms the mechanism.
- Same pattern elsewhere: every caller of `f.request`/`p.remoteRequest` (`engine_switch_test.go:329`, `:375`, `:482`, `:510`, `:654`, `:722`, `:767-769`, `:835`, `:844`), `remote_configuration_test.go` (3 `ExpectedRevision` captures), `remote_management_test.go` (7), `serving_envelope_test.go` (2), `reverse_peer_control_task2_test.go` (2) capture a revision and submit on it with no fence against the 2 s refresh. `:375` ("switch-stale") intentionally stales the revision and is unaffected. The exposure is the same as the known one; only the first tick after a start is a plausible mover, so tests that start an engine and immediately switch are the ones at risk (`:391`, the pinned fixtures at `:835-844` after `engine:start` in `newPinnedSwitchFixture`).
- Disposition: amend the tests, not the product. After any `engine:start` in a switch fixture, read `engine:settings-status` (or `engine:switch-status`) until two consecutive reads agree, or make the refresh interval a test-settable var and set it long in these fixtures. Either removes the race without weakening what the test proves. Repair now (cheap) rather than defer, because it gates every join.

### 6.2 Other candidates

- UNVERIFIED (observed here, cause environmental): one failed service-harness test retains its bundle slot by design (`service_harness_test.go:113-122`: "Retain the slot when ownership cleanup could not be verified") and the manifest has one slot (`:55`), so `TestServicePolicy`, `TestServiceRestartReconnect` and `TestServiceRoutingListenerWithoutFacades` then fail with "service fixture slots exhausted" (`service_restart_reconnect_test.go:82`). Order-dependent amplification: one genuine failure in `TestServiceLifecycle` shows as four red tests and costs a full rerun. Disposition: report the retained slot once and `t.Skip` the dependants with an explicit reason, or add a second slot to the manifest. Owner question: is one slot a firewall-rule constraint on the gate machine?
- UNVERIFIED (not reproduced): `node_info_rpc_test.go:241-245` asserts `elapsed` between 1 s and 2.5 s around a 1 s relay deadline; under heavy load (another suite's teardown, Defender) a 2.6 s elapsed fails a correct implementation. `policy_test.go:3777-3787` sweeps release offsets of 97-101 ms in 200 us steps against a 100 ms activation wait by design, which is a deliberately tight timing probe and the most load-sensitive test in the broker suite. Disposition: widen the upper bound at `:242` to 5 s (the lower bound is the real assertion); leave the policy sweep but run it last or mark it with a retry-once wrapper.
- LOW (negatives by waiting, could pass if the failure became slower than the wait): `assertStillClustered` 10 s (`cluster_manager_test.go:1338-1344`), `log_level_test.go:201` 6 s, `reverse_peer_control_test.go:789` 4 s, `fixture_ports_test.go:446` 3 s, `reverse_peer_control_task2_test.go:592` and `:641` 1.5 s. None can pass while the behaviour is broken today because each is paired with a positive observation earlier in the test; they are speed items, not RED.
- LOW: `cluster_manager_test.go` 1 s bind sleeps (20) and `:692` 600 ms are fixed-delay barriers; under load a listener that binds later than 1 s makes the next `cluster:*` call fail with a connection error. Not seen here; a barrier (rank 7) removes the exposure.
- LOW: `TestPolicyActivationSlotReleasedAtTheDeadlineStillDeliversThePush` (2.4 s) is a 21-iteration timing sweep (`policy_test.go:3777`) that is also the broker test most likely to flake under CPU starvation; it passed here.

## 7. Platform notes (Linux-only observations, not findings about the gate)

- `tests/`: 18 top-level failures, all environmental: the container has no IPv6 multicast (`discovery_test.go:143` "listen udp6 [ff02::]:5353: socket: address family not supported by protocol"), so every mDNS-dependent test ran to its 30-45 s timeout; the service-harness cascade of section 6.2 followed one such failure.
- `nvpair-engine-manager`: 19 failures in two groups, both platform-specific here: `/proc`-based "(deleted)" listener inspection (`lifecycle_stop_test.go:219`, `:254`) and the Strata serving family failing with "Strata/Python environment overrides are unsupported in prepared mode" (`serving_strata_test.go:293` and 13 more). Strata is deferred by the owner; the suite's Windows result for these is the one that counts. Not investigated further (outside scope).
- `nvpair-proxy`: 4 failures where `TestPublicTransport*` got "public DNS refused" (`api_transport_test.go:485`, `egresslease_test.go:612`): the container's egress proxy, not the product.

## 8. Top 5 changes and expected gate time

1. Broker in-test builds with the real `GOCACHE`, built once per fixture program (rank 1): -45 s here; on Windows likely -60 to -90 s of the 2.5 min broker suite.
2. `TestUninstallRetries` `detectTimeout` 2 s (rank 2): -28 s in the engine-manager suite, Windows about the same (it is a pure wait).
3. `commandWaitDelay` var (rank 3): -29 s engine-manager, Windows about the same.
4. Broker const budgets to vars (rank 4): -38 s broker.
5. `tests/` barriers for the cluster-manager bind sleeps and `assertStillClustered` (ranks 7 and 9, barrier form), plus the twelve-binary build in parallel (rank 6): -45 s here, more on Windows.

Expected gate after these five: on this platform 1153 s to about 965 s (about 16%). Mapped to the handoff's Windows figures (6 + 1 + 2.5 + 14 min), the module suites fall from 9.5 to about 7 min and `tests/` from 14 to about 12.5 min: a full gate of roughly **20 min instead of 25**. With the test-impact selection of section 5 applied to non-broker joins, a typical proxy, scheduler or cluster-manager join runs `tests/` in 5-7 min, for a gate of **12-14 min**; broker-lane joins stay near the full 20 min because the broker fronts every worker. The floor is set by process start/stop per test (about 1-2 s each for 180 tests) and by product budgets the suite deliberately exercises at real time (section 3.4 rows for the paired-directory expiry, XA facade enforcement and the performance-observation backoff, about 90 s together); taking those below the floor would need product-side test seams and is a separate decision.

## Appendix: measured tables

**nvpair-engine-manager, top 40** (leaf tests; sum 230.2s)

| # | s | cum % | status | test |
|---:|---:|---:|---|---|
| 1 | 30.15 | 13.1 | pass | `TestUninstallRetries` |
| 2 | 5.02 | 15.3 | pass | `TestRemoteSettingsEventsReconnectThroughExistingOwner` |
| 3 | 5.01 | 17.5 | pass | `TestLlamaServingColdCommandLoadCancelledByStop/user_Stop` |
| 4 | 5.01 | 19.6 | pass | `TestLlamaServingColdCommandLoadCancelledByStop/StopAll` |
| 5 | 5.01 | 21.8 | pass | `TestLlamaServingColdCommandLoadCancelledByStop/caller_cancel` |
| 6 | 5.01 | 24.0 | pass | `TestLlamaServingCommandExitedParentPipeBound/user_Stop` |
| 7 | 5.01 | 26.2 | pass | `TestLlamaServingCommandExitedParentPipeBound/StopAll` |
| 8 | 5.01 | 28.3 | pass | `TestLlamaServingCommandExitedParentPipeBound/caller_cancel` |
| 9 | 3.03 | 29.7 | pass | `TestCommandStopFailureRestartsHealthMonitoring` |
| 10 | 2.73 | 30.8 | pass | `TestStopReclaimsOurOrphanOnManagedPort` |
| 11 | 2.47 | 31.9 | pass | `TestLoadedServingInvalidationConcurrentLifecycle/start` |
| 12 | 2.41 | 33.0 | pass | `TestModelIdentityIsNeverSentAfterTheStopItPrecedes` |
| 13 | 2.22 | 33.9 | pass | `TestHealthFlipReportsAndClears` |
| 14 | 2.07 | 34.8 | pass | `TestCommandStopRejectsSuccessWhileEndpointIsUnhealthy` |
| 15 | 2.04 | 35.7 | pass | `TestModelCopyReverseResumesDroppedBodyWithNewRange` |
| 16 | 2.01 | 36.6 | pass | `TestCommandStopRejectsSuccessWhileEndpointIsLive` |
| 17 | 1.82 | 37.4 | pass | `TestFailedRestoreRetainsDesiredOnForRetry` |
| 18 | 1.61 | 38.1 | pass | `TestReadinessTimeoutNoSpuriousExit` |
| 19 | 1.56 | 38.8 | pass | `TestSwitchDelayedScopedStopPreservesLaterReservation/target` |
| 20 | 1.36 | 39.3 | pass | `TestSwitchDelayedScopedStopPreservesLaterReservation/source` |
| 21 | 1.34 | 39.9 | pass | `TestUninstallTerminatesRunningInstance` |
| 22 | 1.32 | 40.5 | pass | `TestStopClearsUnhealthy` |
| 23 | 1.26 | 41.0 | pass | `TestUninstallDoesNotKillExternalSameNameProcess` |
| 24 | 1.24 | 41.6 | pass | `TestStopDeclinesForeignListenerWithActionableError` |
| 25 | 1.23 | 42.1 | pass | `TestStopDoesNotReclaimSameExternalImage` |
| 26 | 1.22 | 42.6 | pass | `TestLlamaServingSafeStartFailures/readiness_timeout_reports_stage_and_duration` |
| 27 | 1.19 | 43.2 | pass | `TestProcImageUnavailableOwnerRefused/uninstall` |
| 28 | 1.18 | 43.7 | pass | `TestProcImageUnavailableOwnerRefused/stop` |
| 29 | 1.11 | 44.2 | pass | `TestModelIdentityIgnoresAListingAcrossARunChange` |
| 30 | 1.06 | 44.6 | pass | `TestDeclinedAdoptedStopStillPersistsOffIntent` |
| 31 | 1.06 | 45.1 | pass | `TestStopHonorsProcessSignalPolicy/term_escalates_after_configured_grace` |
| 32 | 1.04 | 45.5 | pass | `TestStopRejectsAdoptedProcessEngine` |
| 33 | 1.03 | 46.0 | pass | `TestStopKeepsAdoptedProcessRunningWhileEndpointIsUnhealthy` |
| 34 | 1.02 | 46.4 | pass | `TestRemoteSettingsEventsRetireChangedAuthority/peer_pin_removed` |
| 35 | 1.02 | 46.9 | pass | `TestRemoteSettingsEventsRetireChangedAuthority/peer_certificate_replaced` |
| 36 | 1.02 | 47.3 | pass | `TestRemoteSettingsEventsRetireChangedAuthority/admission_epoch_changed` |
| 37 | 1.02 | 47.8 | pass | `TestSettingsPairedMutationPushObserversAndRevocation` |
| 38 | 1.01 | 48.2 | pass | `TestCommandStopFailureKeepsLiveEngineRunning` |
| 39 | 1.01 | 48.6 | pass | `TestRemoteSettingsEventsRetireChangedAuthority/local_certificate_replaced` |
| 40 | 1.01 | 49.1 | pass | `TestRemoteSettingsEventsRetireChangedAuthority/endpoint_generation_changed` |

Top 10: 71.0s = 30.8%
Top 20: 90.6s = 39.3%
Top 40: 113.0s = 49.1%

**nvpair-ui-broker, top 25** (top-level functions; sum 130.5s)

| # | s | cum % | status | test |
|---:|---:|---:|---|---|
| 1 | 17.08 | 13.1 | pass | `TestEngineManagerPortsSpawnAndScannerReplay` |
| 2 | 14.05 | 23.9 | pass | `TestEngineManagerPortsCLIRejectsInvalidBeforeWorkerResolution` |
| 3 | 10.67 | 32.0 | pass | `TestPolicyActivationAvailabilityPushFaults` |
| 4 | 10.42 | 40.0 | pass | `TestServicePortsDefaultPlanKeepsWorkerArgvAndRegistrations` |
| 5 | 6.26 | 44.8 | pass | `TestTeardownBudgetCapsAggregateWait` |
| 6 | 6.05 | 49.5 | pass | `TestServicePortsInvalidPlansRejectedBeforeAnyChild` |
| 7 | 5.85 | 53.9 | pass | `TestServicePortsAlternatePlanFlowsThroughSpawnRestartAndReplay` |
| 8 | 5.02 | 57.8 | pass | `TestProxySupervisionKeepsPolicySetWithinItsBoundAgainstAnUnresponsiveProxy` |
| 9 | 5.01 | 61.6 | pass | `TestWaitForStdinCloseKillsWorkerThatIgnoresShutdown` |
| 10 | 5.00 | 65.5 | pass | `TestLocalServicesWriteTimeoutClosesUnresponsiveChild` |
| 11 | 3.03 | 67.8 | pass | `TestSwitchRecoveryReplaysJournalOffWithoutCheckpoint` |
| 12 | 3.01 | 70.1 | pass | `TestSwitchRecoveryExhaustionKeepsManagerHiddenAndReportsBlocked` |
| 13 | 3.00 | 72.4 | pass | `TestWaitForStdinCloseGivesUpWithoutProcess` |
| 14 | 2.38 | 74.2 | pass | `TestPolicyActivationSlotReleasedAtTheDeadlineStillDeliversThePush` |
| 15 | 2.03 | 75.8 | pass | `TestNotificationLifetimeWorkerWrappers` |
| 16 | 2.02 | 77.3 | pass | `TestStderrSinkNeverBlocksAndKeepsOverflow` |
| 17 | 2.00 | 78.8 | pass | `TestWorkloadPrivacyProjectionPersistedRetirement` |
| 18 | 1.50 | 80.0 | pass | `TestNodeInfoRPCRelayDeadlineDisownsLateReply` |
| 19 | 1.50 | 81.1 | pass | `TestNodeInfoRPCStalledInputRetiresScannerAndUnwindsWrite` |
| 20 | 1.44 | 82.2 | pass | `TestAvailabilityPublicationFailureLeavesOneWholeDocument` |
| 21 | 1.04 | 83.0 | pass | `TestNotificationLifetimeWorkerActiveDeadline` |
| 22 | 1.02 | 83.8 | pass | `TestProxySupervisionKeepsTheBackoffOfAFacadeNodeAcrossListenerChanges` |
| 23 | 1.02 | 84.6 | pass | `TestNotificationLifetimeBoundSourceFailure` |
| 24 | 1.01 | 85.4 | pass | `TestSwitchRecoveryRetriesInspectionBeforePublishingManager` |
| 25 | 1.01 | 86.2 | pass | `TestSwitchRecoveryRetriesDrainReleaseWithoutRestart` |

Top 10: 85.4s = 65.5%
Top 20: 107.3s = 82.2%
Top 40: 118.3s = 90.7%

**nvpair-proxy, top 15** (leaf tests; sum 48.5s)

| # | s | cum % | status | test |
|---:|---:|---:|---|---|
| 1 | 5.08 | 10.5 | pass | `TestChatAccountAttemptDelayedClassification` |
| 2 | 5.01 | 20.8 | fail | `TestHandleHTTP_RealSocketFlushDeadline` |
| 3 | 2.50 | 26.0 | pass | `TestPublicTransportCancel/idle` |
| 4 | 2.50 | 31.1 | pass | `TestPublicTransportCancel/cancel` |
| 5 | 2.26 | 35.8 | pass | `TestPendingOutcomeSurvivesCompletedHoldRetention/exhaustion` |
| 6 | 2.20 | 40.3 | pass | `TestE2EFailoverOverRealBinary/ollama` |
| 7 | 2.16 | 44.8 | pass | `TestRetainedReconcilesAreBoundedAndEvictedAttemptsAreUnknown` |
| 8 | 2.15 | 49.2 | pass | `TestChatAccountAttemptRetentionAndPriorProcess` |
| 9 | 2.14 | 53.6 | pass | `TestPendingOutcomeSurvivesCompletedHoldRetention/rate_limit` |
| 10 | 2.06 | 57.9 | pass | `TestPendingOutcomeSurvivesCompletedHoldRetention/credential_failure` |
| 11 | 2.00 | 62.0 | pass | `TestPublicTransportBounds/slow_headers` |
| 12 | 1.87 | 65.9 | pass | `TestPendingQualificationSurvivesCompletedHoldRetention` |
| 13 | 1.50 | 69.0 | pass | `TestHandleHTTP_WaitsForAnOwnerWithoutSpendingAttempts` |
| 14 | 1.21 | 71.5 | pass | `TestPublicTransportCancel/bytes_reset_idle` |
| 15 | 1.05 | 73.6 | pass | `TestResidentCapacitySuccessfulLoadsRetireWithoutReplay` |

Top 10: 28.1s = 57.9%
Top 20: 40.1s = 82.7%
Top 40: 45.1s = 93.0%

**services/tests, top 40** (top-level functions; sum 693.6s)

| # | s | cum % | status | test |
|---:|---:|---:|---|---|
| 1 | 90.89 | 13.1 | fail | `TestReversePeerControlTask2` |
| 2 | 45.09 | 19.6 | fail | `TestWorkloadManagerRehydratesRecentTerminalOnRestart` |
| 3 | 45.06 | 26.1 | fail | `TestWorkloadManagerRehydratesActiveWorkloadOnRestart` |
| 4 | 37.85 | 31.6 | pass | `TestPairedDirectoryBootstrap` |
| 5 | 36.88 | 36.9 | pass | `TestXARemoteConfigurationServing` |
| 6 | 35.36 | 42.0 | fail | `TestReverseNodeInfoPhase04` |
| 7 | 30.42 | 46.4 | pass | `TestClusterOutboundPairingAuthenticatedJoinerPinFailure` |
| 8 | 30.05 | 50.7 | fail | `TestWorkloadFailedOnNodeLoss` |
| 9 | 30.02 | 55.0 | skip | `TestScannerEvictsRecordSupersededAtItsAddress` |
| 10 | 30.02 | 59.4 | fail | `TestModelsHTTPEnrichment` |
| 11 | 30.01 | 63.7 | fail | `TestModelsRefreshConvergesWithoutMDNSChange` |
| 12 | 19.58 | 66.5 | pass | `TestDescriptorEngineIsFrontedFromAUserManifest` |
| 13 | 17.71 | 69.1 | pass | `TestPerformanceObservationFailedDispatchThenSuccess` |
| 14 | 17.43 | 71.6 | pass | `TestReversePeerControlTask1` |
| 15 | 11.29 | 73.2 | pass | `TestClusterNoSelfRemoveWhenPeerAccepts` |
| 16 | 11.28 | 74.8 | pass | `TestClusterNoSelfRemoveWhenPeerUnreachable` |
| 17 | 11.07 | 76.4 | pass | `TestClusterNoSelfRemoveWhenPeerLeft` |
| 18 | 10.69 | 78.0 | pass | `TestClusterCancelInviteRace` |
| 19 | 8.22 | 79.1 | pass | `TestLogSetLevelViaRPC` |
| 20 | 7.73 | 80.3 | fail | `TestServiceLifecycle` |
| 21 | 7.58 | 81.4 | fail | `TestReverseModelCopyPhase04` |
| 22 | 6.32 | 82.3 | pass | `TestClusterSelfRemoveWhenRemovedOffline` |
| 23 | 5.75 | 83.1 | pass | `TestPerformanceObservationBrokerPersistenceAndPublication` |
| 24 | 5.38 | 83.9 | pass | `TestPerformanceObservationDestinationAndAuthenticatedReplacement` |
| 25 | 5.29 | 84.6 | pass | `TestPerformanceObservationOldCapabilityDoesNotAffectInference` |
| 26 | 5.28 | 85.4 | pass | `TestPortInteropBridgeBrokerTeardownKillsStalledWorker` |
| 27 | 5.01 | 86.1 | pass | `TestLogLevelEnvFallback` |
| 28 | 3.34 | 86.6 | fail | `TestRemoteManagementPersistsAcrossSeparateProcessRestarts` |
| 29 | 3.34 | 87.1 | pass | `TestWorkloadOutOfOrderSuppressed` |
| 30 | 3.22 | 87.5 | pass | `TestFixtureBrokerLeavesDefaultServicePortsAlone` |
| 31 | 3.05 | 88.0 | pass | `TestClusterInviteExpiresBothSides` |
| 32 | 3.02 | 88.4 | pass | `TestRemoteManagementDeniedPeerCanReadModelExports` |
| 33 | 2.68 | 88.8 | pass | `TestStrataContractModelsAndOperations` |
| 34 | 2.36 | 89.1 | pass | `TestOBOffOrigin` |
| 35 | 2.03 | 89.4 | pass | `TestAllDefaultEnginesAreServedByOneProxyProcess` |
| 36 | 1.96 | 89.7 | pass | `TestRemoteManagementReverseMutationBoundary` |
| 37 | 1.94 | 90.0 | pass | `TestRemoteManagementReverseRevocationBoundary` |
| 38 | 1.92 | 90.3 | pass | `TestXARemoteConfigurationSwitch` |
| 39 | 1.87 | 90.5 | pass | `TestPolicyRelayJoint` |
| 40 | 1.85 | 90.8 | pass | `TestStrataContractUnsupportedManagement` |

Top 10: 411.6s = 59.4%
Top 20: 556.6s = 80.3%
Top 40: 629.8s = 90.8%
