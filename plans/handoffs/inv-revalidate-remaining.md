<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Investigation: revalidate remaining plan tasks against current code (slug: revalidate-remaining, branch plan/inv-revalidate-remaining)

Read `plans/handoffs/inv-common.md` on branch `plan/cloud-investigations` first.

Kind: plan freshness review. The remaining tasks were planned against older bases (17727f1 to 9681775). Much has landed since (NE Parts A/C, I 3a/3b/4a/8, SR 5a-i, PA 4-6, J 1, B 7). Question: can an executor that follows each remaining task literally still succeed at 210babf, and in what order should they run?

Scope: B Tasks 2-6 (`plan/b-opencode-shapes`), J Tasks 2-6 (`plan/j-routing-diagnostics`), PA Tasks 1-3 and 7 (`plan/pa-paused-advertisement`), SR 5a-ii, 5b, 5c, 6a, 6b (`plans/context/sr5-rescope.md`).

Do, for each remaining task:
1. Check every cited path, symbol, line reference, wire field and test name still exists with the described behaviour. List each stale reference with its current replacement.
2. Check whether its premise still holds (a landed task may already have done part of it, or changed the interface it consumes). Mark the task: READY as written / READY with listed edits / BLOCKED (on what) / OBSOLETE.
3. Compute file overlaps between remaining tasks and the remaining I tasks (4b, 5, 6, 7 in `plans/context/i-unified-dispatch.md`, which hold large parts of services/nvpair-proxy). Produce a dependency/serialisation table and a proposed execution order across two Windows machines and one Mac author, maximising parallel work with disjoint files.
4. Propose exact replacement text for each stale Do/Verify item (quote old, give new). Do not rewrite whole plans.
