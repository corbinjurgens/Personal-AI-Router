<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Investigation: audit landed PA, J, B and XA work (slug: audit-pa-j-b, branch plan/inv-audit-pa-j-b)

Read `plans/handoffs/inv-common.md` on branch `plan/cloud-investigations` first.

Kind: delivery audit. Question: does fork/groundwork deliver PA Tasks 4-6, J Task 1, B Task 7 (store:false) and XA (all tasks) as their plans state, and are their user-visible claims (desktop, TUI, docs) true of the code?

Do:
1. Read each plan (`plan/pa-paused-advertisement`, `plan/j-routing-diagnostics`, `plan/b-opencode-shapes`, `plan/xa-remote-config-acceptance`) before code. Build an obligation checklist for the landed tasks only.
2. Map obligations to code and tests; FULFILLED / FAILED / UNVERIFIED with evidence.
3. Specific risks: PA paused advertisement must never let a peer route to a paused node and must clear on resume without a stale window; local pause beats remote resume. J T1 retirement path projection is complete (no diagnostic left unprojected or leaking prompts/bodies). B T7: store:false requests are accepted or rejected exactly as the plan says, before side effects, with typed errors. XA: remote config acceptance respects target-local pairing permission ("my own device" vs friend/locked) and local veto.
4. Check docs (docs/*.mdx, services/*/spec.md, component READMEs) touched by these tasks against the code; list contradictions.
5. Produce a short real-machine acceptance checklist for what cannot be shown from code.
