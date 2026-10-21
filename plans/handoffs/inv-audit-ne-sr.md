<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Investigation: audit landed NE and SR work (slug: audit-ne-sr, branch plan/inv-audit-ne-sr)

Read `plans/handoffs/inv-common.md` on branch `plan/cloud-investigations` first.

Kind: delivery audit. Question: does the landed code on fork/groundwork deliver what the NE plan (Tasks 1-4 Part A and Task 6 Part C) and SR Task 5a-i (`plans/context/sr5-rescope.md`) promised, and do the two packages compose correctly?

Do:
1. Read the NE plan on `plan/ne-serving-envelope` and `plans/context/sr5-rescope.md` before reading code or commit messages. Extract every Do/Verify obligation, compatibility promise and decision for the landed tasks into a checklist.
2. Map each obligation to code and tests at the inspected SHA (use first-parent merges to find the commits). Mark each FULFILLED / FAILED / UNVERIFIED with evidence.
3. Composition checks (known hot spots): identity precedes envelope (a cold exact allocation without identity must be "unqualified"); SR residency vs NE readiness ordering; engine-manager lifecycle mutex guards around runtime copies (Part C) - look for any remaining whole-struct copy or unguarded read of mutable runtime state; settings edits preserving NE configuration; no migration that blesses existing processes, files or settings.
4. Look for tests that would still pass if the property broke (assertions on the wrong field, fixtures that bypass the path). Name them.
5. List what can only be proven on a real Windows machine with llama.cpp, as a short acceptance checklist the owner can run.
