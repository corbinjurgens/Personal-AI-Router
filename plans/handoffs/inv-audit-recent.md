<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Investigation: delivery audit of recent units (slug: audit-recent, branch plan/inv-audit-recent)

Read `plans/handoffs/inv-common.md` on branch `plan/cloud-investigations` first.

Kind: delivery audit (did the code deliver what the plan and rulings required). Scope:
- B Tasks 2-4 (landed: merges `96f1290`, `59ad85a` on `fork/groundwork`) and B Task 5 (in `integration/join25`), against `plans/b-opencode-shapes.md` on `plan/b-opencode-shapes` and the ruling that `reasoningReplay` stays required-false.
- SR 5a-ii and 5b (merges `d9f2276`, `40046f8`) against `plans/context/sr5-rescope.md` and the SR rulings.
- NE readback transport classification (`integration/join25`, commits from `roadmap/ne-readback-transport` and the follow-up `02f8ad6`) against the NE readback ruling.
- Manual node reference validation (`integration/join25`, from `roadmap/j-manual-id`) against the manual node reference ruling and UNV-3 of `plans/reviews/audit-pa-j-b.md` on `plan/inv-audit-pa-j-b`.

Do:
1. For each unit, list every DONE-WHEN or requirement sentence in its plan/ruling and mark it FULFILLED (cite code and test `path:line`), FAILED, or UNVERIFIED (needs a real machine or evidence you cannot produce). Read the plan and ruling before the code.
2. Probe edges the tests may miss: for readback, each transport-failure class versus semantic mismatch, and the interaction with the resident observation caller in the engine manager; for manual references, Unicode format characters, length bounds, and every RPC field that carries a node reference (are any missed?); for B, request shapes OpenCode actually sends per the inventories on `plan/b-opencode-inventory-*`; for SR, the fail-closed bounds and context-length equality.
3. Run the module unit tests for the touched modules on `integration/join25` and report counts.
4. Check generated docs and contracts consistency: does `desktop/docs/services-api.md` match the Go payloads touched by these units? (Read only; it is generated.)

Report: `plans/reviews/audit-recent.md`, verdict per unit, findings with dispositions.
