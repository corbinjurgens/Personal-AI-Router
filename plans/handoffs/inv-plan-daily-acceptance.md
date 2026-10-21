<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Investigation: plan the daily local/peer milestone acceptance (slug: plan-daily-acceptance, branch plan/inv-plan-daily-acceptance)

Read `plans/handoffs/inv-common.md` on branch `plan/cloud-investigations` first.

Kind: new execution plan, written to `plans/reviews/plan-daily-acceptance.md` (same saving steps). Use the plan format from `plans/handoffs/b-opencode-shapes.md` on branch `plan/cloud-handoffs` (Goal, Change scope, Context, Decisions, Decisions made while writing, Tasks with Files/Depends on/Complexity/Do/Verify, Verification, Do not, If stuck, Owner questions). Most tasks here are real-machine procedures, not code: Files may be "none (procedure)" plus any test, script or doc the procedure needs.

Goal to plan: prove the daily local/peer milestone (see inv-common.md) on the real fleet: MSI and PC3 (Windows, llama.cpp router mode with per-model presets, each with one approved serving configuration), the Mac as an origin-only host, OpenCode as the client.

Do:
1. Inventory what already exists for live validation: docs/*.mdx procedures, services/tests live or multi-node tests and how they are gated (env vars, skips), scripts/, existing XA acceptance tasks. Cite paths.
2. Map each milestone clause to (a) the landed code that implements it, (b) the remaining package task it still depends on (I, SR, PA, J, B, NE Part B), and (c) an observable acceptance check: exact action, exact expected observation (UI state, JSON-RPC payload, HTTP status/body shape, log line without prompt content), and what counts as fail.
3. Cover: tier routing (weak/medium/strong) with declared roles per node; peer routing; remote config and engine/model switching from the desktop with "my own device" vs locked permission and local veto; headless nvpair-service with Windows service-process restart and client reconnect; streams interrupted by peer loss or engine switch (delay/pause-resume vs error); cancelled requests releasing reservations; telemetry that is accurate or reports unknown; OpenCode tool calls and store:false shapes.
4. Order the procedure so it can start before all packages land: mark each check READY-NOW or WAITS-FOR <package task>.
5. Safety: no test may touch production ports or a running candidate without the owner present; Windows firewall prompts are possible for new executable paths, so list every executable the procedure starts.
