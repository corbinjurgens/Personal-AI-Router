<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Investigation: debate the remaining I and G tasks (slug: debate-i-g, branch plan/inv-debate-i-g)

Read `plans/handoffs/inv-common.md` on branch `plan/cloud-investigations` first.

Kind: adversarial design review (debate) of an unbuilt plan. Inputs: `plans/context/i-unified-dispatch.md` (remaining Tasks 5, 6, 7; Task 4b is nearly done) and `plans/context/g-egress-lease.md` (remaining Tasks 3, 4), checked against the code at the inspected SHA.

I is the single attempt owner for inference dispatch: permission, reserve, pre-send, commit; restart-safe origin/proxy-run/request identity; canonical cancellation across local, peer and cloud attempts; process-wide buffer bound; interrupted-stream behaviour (owner: user sees only delay or pause-resume, error only when no same-model continuation exists). G is the public egress lease, landed disabled.

Do:
1. For each remaining task, state the strongest objection: a scenario (concrete request sequence, restart, cancellation race, peer loss, buffer pressure) where the task as written yields a wrong outcome, double send, lost cancellation, stuck reservation, unbounded memory, or a dishonest completed/failed/cancelled status. Ground each in current code paths (`path:line`).
2. For each objection give the minimal plan amendment, and say whether it is corrective (needed to meet an already-settled contract) or scope-expanding (needs owner approval).
3. Identify over-engineering: parts that the daily local/peer milestone does not need and that could be deferred without weakening landed safety boundaries. Public cloud egress stays disabled for the milestone.
4. Check the interaction of I's remaining tasks with SR 5b/6 residency and NE Part D (deferred output caps): any ordering trap.
5. End with a ranked list: top 5 amendments by risk reduced per unit of work.
