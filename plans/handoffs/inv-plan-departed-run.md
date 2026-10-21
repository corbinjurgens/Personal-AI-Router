<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Investigation: plan the departed-run workload retirement fix (slug: plan-departed-run, branch plan/inv-plan-departed-run)

Read `plans/handoffs/inv-common.md` on branch `plan/cloud-investigations` first.

Kind: execution plan for a confirmed defect. Input: RED-1 of the first-round debate report (`plans/reviews/debate-i-g.md` on branch `plan/inv-debate-i-g`): when the proxy process crashes or is replaced, workloads it registered as local-origin inference stay "running" in the broker's workload view and cannot be cancelled. This is queued on the broker lane after SR 5c.

Do:
1. Re-verify the defect at the inspected SHA with `path:line`: how the proxy registers workloads, how the broker tracks them, what happens to them when the proxy process exits or a new proxy run starts (proxy-run identity from I's restart-safe identity work), and what cancel does for a workload whose owner run is gone. Include the remote-origin case (a peer forwards a request, then the local proxy dies) and the TUI/desktop projection.
2. Decide the owner of retirement (broker supervisor on worker exit, workload manager, or the proxy's next run) by the repository rule that services own state and desktop only renders. Not a desktop fix.
3. Specify the honest outcome for retired workloads (the owner wants honest completed/failed/cancelled; an interrupted run is failed unless the requester cancelled) and the notification shape. If a JSON-RPC payload changes, list every place the repository rule says must change together.
4. Write the plan in the task format: for each task, Files (exact paths, including tests and docs), Depends on, Complexity (mechanical/judgment), Do, Verify (a command with a pass condition). Include a process-level test in `services/tests` that kills the proxy mid-request and asserts retirement and cancellability (describe it; do not run it).
5. Note interactions with SR 5c residency and I Tasks 5-7, and any ordering trap.

Report: `plans/reviews/plan-departed-run.md` (the plan plus a short verification of the defect).
