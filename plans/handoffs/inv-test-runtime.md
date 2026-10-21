<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Investigation: services/tests and module suite wall-clock cost (slug: test-runtime, branch plan/inv-test-runtime)

Read `plans/handoffs/inv-common.md` on branch `plan/cloud-investigations` first.

Kind: performance investigation of the test suites themselves. No product change.

Why: every integration join is gated on one Windows machine running, serially, the engine-manager module suite (about 6 minutes), the proxy suite (about 1 minute), the broker suite (about 2.5 minutes) and the full cross-process `services/tests` suite (about 14 minutes, about 500 tests). One join gate takes about 25 minutes, and one failure sends the whole join back. The goal is to cut that wall-clock time without weakening coverage.

Exception to the common brief, for this investigation only: you MAY run `services/tests` in your own isolated cloud container to measure durations (`go test -json ./...` from `services/tests`, after `./build.sh` from `services/` if the tests need staged binaries; read `services/readme.md#testing`). Never run it on any real host. Never start real engines or models. Record your platform; Linux timings differ from Windows, so report relative cost and mechanisms, not absolute Windows predictions.

Constraints you must design within:
- Windows runs are serialized per machine. Test binaries run from a fixed path for firewall rules; fixture paths are shared between packages, so naive package-level parallelism across modules is not available on Windows without changing fixture paths.
- A skipped test is untested, not passing. Coverage must not shrink silently.
- Tests listen on `127.0.0.1:0` unless a peer must dial a LAN address.

Do:
1. Measure per-test and per-package durations (top 40 tests by elapsed, and the cumulative share of the top 10 and top 40). Do the same for `services/nvpair-engine-manager` and `services/nvpair-ui-broker` module suites.
2. For each expensive test, name the mechanism with `path:line`: fixed `time.Sleep`, polling intervals, waiting out a timeout to prove a negative, repeated product-binary startup per test, repeated fixture builds, serial tests that could be `t.Parallel()`, broad retries.
3. Propose concrete changes ranked by seconds saved per unit of risk. For each: the exact change (diff sketch), seconds saved on your platform, what it could mask (for example a negative proved by waiting is replaced by an event barrier), and whether it is safe under Windows serialization.
4. Propose a test-impact selection for joins: a map from changed service module/package to the `services/tests` files that exercise it (derive from which binaries and RPC methods each test file drives). State how a join would choose the minimal sufficient subset, and which tests must always run. Point out where the map would be unsafe (shared fixtures, cross-module contracts).
5. Find any test that is order- or timing-sensitive enough to flake under load (fixed short deadlines, races between an async effect and an assertion). One known intermittent example: `TestEngineSwitchProcessRejectsRunningTargetBeforeDrain` once failed with "Engine configuration changed; reload before switching" at `services/tests/engine_switch_test.go` (the switch revision captured just after `engine:start` was stale), then passed 10/10 on reruns. Report whether you see the same pattern elsewhere.

Report: `plans/reviews/test-runtime.md`. Use RED for a test that can pass while the behaviour it names is broken, UNVERIFIED for a likely flake you could not reproduce, LOW for pure speed opportunities. End with the top 5 changes and the expected gate time after them.
