<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Investigation: SR 5c resident reconcile state machine spec and attack (slug: sr5c-spec, branch plan/inv-sr5c-spec)

Read `plans/handoffs/inv-common.md` on branch `plan/cloud-investigations` first, especially the SR 5c reconcile ruling.

Kind: specification plus adversarial review of a branch under repair. Inputs: `plans/context/sr5-rescope.md` (Task 5c), the coordinator ruling, and the code on `roadmap/sr-5c` (inspect the newest head; record its SHA; at writing it was `74878aa`) compared with `fork/groundwork`. Main code: the broker's resident reconciliation in `services/nvpair-ui-broker/` (find it from `git diff fork/groundwork...roadmap/sr-5c --stat`), plus the desktop consumers and `services/tests/service_policy_test.go`, `policy_relay_joint_test.go` on that branch.

Why: SR 5c keeps the declared llama.cpp residents loaded on a host. It has gone through several review/repair rounds, each finding a new transition bug: a duplicate load on confirmation, transient failure suppression, a duplicate after a capacity change, pause clearing pending state, and a failed finish stuck after proxy replacement. Rounds are finding one edge at a time. A complete transition table would end that.

Do:
1. Write the state machine as a table. State per (engine, model, policy revision): desired, observed-loaded, in-flight load, pending observation, failure count/backoff, gating (pause, Off, drain, capacity). Events: policy revision change/removal, observation (confirming, contradicting, unknown/transport failure per the NE readback ruling), load start/success/failure, observation timeout, pause/Off/drain on and off, capacity change, proxy or engine-manager replacement/restart, broker restart. For every (state, event) cell give the required next state and side effect (load issued or not) derived from the ruling. Mark cells the ruling does not settle as OPEN with your recommended answer.
2. Check every cell against the branch code with `path:line`. Classify mismatches: RED (duplicate load, lost load, stuck state, a gate that a load bypasses, unbounded retry), UNVERIFIED, LOW.
3. Check the branch tests against the table: list cells with no test. Propose one table-driven test that enumerates all cells (sketch its case table and the fake engine-manager it needs). Run the broker module unit tests (`go test ./...` in `services/nvpair-ui-broker`) on the branch and report the result.
4. Invariants that must hold across all sequences (for example: at most one in-flight load per key; no load while gated; every in-flight or pending record has a bounded exit). For each, give the code location that enforces it or say none does.

Report: `plans/reviews/sr5c-spec.md`, with the full table, mismatches, test gaps, and a ranked repair list.
