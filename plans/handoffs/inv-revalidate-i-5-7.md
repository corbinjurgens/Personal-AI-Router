<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Investigation: revalidate I Tasks 5-7 against Task 4b as built (slug: revalidate-i-5-7, branch plan/inv-revalidate-i-5-7)

Read `plans/handoffs/inv-common.md` on branch `plan/cloud-investigations` first.

Kind: plan revalidation, producing ready-to-dispatch task briefs. Inputs: `plans/context/i-unified-dispatch.md` (Tasks 5, 6, 7) and `plans/reviews/debate-i-g.md` on `plan/inv-debate-i-g`. Task 4b is implemented on `roadmap/i-public-lifecycle` (inspect the newest head; at writing `7c9bfb5`, with a repair in progress for how a committed first-body timeout is classified in performance records). The plan was written before 4b existed, so its Tasks 5-7 likely name stale symbols, files, or seams.

Do:
1. For each of Tasks 5, 6, 7: list every symbol, file, seam and assumption it names and check each against the 4b head with `path:line` (READY, STALE with the correct replacement, or MISSING).
2. Restate each task as a brief an implementer can execute without the orchestrator: Files (exact), Depends on, Complexity, Do, Verify (command with pass condition), and which parts are needed for the daily local/peer milestone versus deferred (the coordinator ruled that I5's hybrid and I6's desktop public surfaces follow the milestone). Split a task where the milestone needs only part of it.
3. Interrupted streams: the owner rule is that the user sees only delay or pause-resume, and an error only when no same-model continuation exists. 4b delivers a synthetic error ending (SC-a) when continuation is unavailable. State precisely what remains for same-model continuation (SC-b) and whether any of Tasks 5-7 owns it; if none does, say so as a gap.
4. Ordering traps with SR 5c, PA 1-2, D3 pinned-origin equality (queued on `roadmap/i-pinned-origin`, not pushed in final form), G3/G4 and NE Part D.

Report: `plans/reviews/revalidate-i-5-7.md`.
