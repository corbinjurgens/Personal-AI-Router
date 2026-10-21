<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Investigation: decision brief for the open owner questions (slug: owner-decisions, branch plan/inv-owner-decisions)

Read `plans/handoffs/inv-common.md` on branch `plan/cloud-investigations` first.

Kind: decision support. The owner must answer these questions; several block the daily local/peer milestone. Produce, for each, the options, what each option costs in code (cite `path:line` of the seam that changes), what it risks, which roadmap tasks it unblocks or adds, and a recommendation. Keep each to about half a page so the owner can answer quickly.

Questions:
1. B: should reasoning replay (re-sending prior reasoning content in requests, `reasoningReplay`) be supported, or stay refused?
2. Should the service run in a headless Windows session (no interactive login), and what does that require of process restart and engine launch?
3. After a service restart, should an already-running llama-server be adopted (identity re-established from readback), or restarted?
4. For the daily milestone, accept a non-strict daily path now, or pull I Task 5 forward?
5. Interrupted streams: accept a day-one honest error (SC-a) where continuation is unavailable, or plan same-model continuation (SC-b) now?
6. Per-host serving presets and resident model selection: who chooses, and where is it configured?
7. Regenerate after a response has started committing: cancel the running attempt, or refuse the regenerate?
8. Manual node IDs: keep free-form bounded strings (current ruling), or restrict to a strict identifier grammar?
9. Does OpenCode point at the facade port (the proxy's public entry) or a per-engine port?

Also list any further owner decision you find the code currently assumes without a recorded answer.

Report: `plans/reviews/owner-decisions.md`.
