<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution plan index: PAIR expectation gaps batch

Entry point for the batch described in `README.md` (causes, recommendations,
owner decisions). This index routes execution; it holds no executable tasks.
Each phase file is a self-contained, session-sized plan in the execution-plan
format, written against `integration/join33` at `4eccd05d` (2026-10-31).
Executors base their branch on current `origin/integration/join33` and re-read
the source before implementing: the files are authoritative over the plans.

## Status

The owner has not yet approved or ordered the phases. Until they do, this
folder is a proposal. Approval means: the owner edits `README.md` section 6
(decisions) where they disagree, deletes or reorders phase rows below, and
says so in the vault. Each approved phase then follows the owner's existing
convention: debate, execute, audit.

## Phases and order

| Phase | File | Entry conditions | Size | Status |
| --- | --- | --- | --- | --- |
| 01 | `01-identity-and-unblock.md` | none (baseline) | medium | proposed |
| 02a | `02a-profiles-services.md` | 01 landed | large | proposed |
| 02b | `02b-profiles-desktop-and-import.md` | 01 and 02a landed | large | proposed |
| 03 | `03-roles.md` | 02a and 02b landed | medium | proposed |
| 04 | `04-desktop.md` | 03 landed (and through it 01, 02a, 02b); no early start | large | proposed |
| 05 | `05-estimate-and-validate.md` | 01 landed for Tasks 1-4 (visible estimate); 02a and 02b for Task 5 and later; 04 (lock removal, peer-Resume rule) for Task 6 and later | large | proposed |
| 06 | `06-routing-and-apps.md` | 01, 02a, 02b, 03 landed; 04 for Tasks 7 and 9; 05 optional, and Task 11 never while a 05 session is open | medium | proposed |

Dependency graph (acyclic): 01 -> 02a -> 02b -> 03 -> 04; 01 -> 05 (Tasks 1-4);
02b -> 05 (Task 5 onward); 04 -> 05 (Task 6 onward); 03 -> 06; 04 -> 06 (Tasks
7 and 9: routing preference control and Apps group); 05 -> 06 (optional:
measured speeds; Task 11 waits for 05 to close). 03 runs before 04, never in
parallel: 04 rebases onto 03 on the paths both edit and never reverts 03's
change. 02a and 02b ship to the owner's machines together (02a alone leaves the
desktop unable to read the new shapes). Wherever 05 runs beside 02a, 02b, 03,
04 or 06 (shared engine-manager, broker, `modelconfig`, `nodepolicy` and
desktop bridge files), the phase that lands second on a shared path rebases
onto the other and re-runs its task's Verify. Phase 05 names the tasks that may
start early.

## Owner decisions this batch assumes

`README.md` section 6, as amended by the 2026-10-31 debate adjudication
(README section 8): D1 user-defined role names with the defaults kept, custom
roles without fallback, roles declarable before they have profiles; D2 the
profile (user-chosen key, display name, icon, identity, derived context) lives
in node policy and its launch settings stay in the engine manager under the
same key; D3 MSI and Joes presets, including native fit, are imported into
profiles once; D4 engine toggles,
locks and the switch UI are removed in favour of role-driven activation on the
non-exclusive start-on-demand path; D5 Halt is the existing immediate local
pause, promoted and renamed, local-only; D6 any llama.cpp build is accepted and
`build_info` is recorded, with device enumeration limited to verified ggml
layouts. A changed
decision requires amending the affected phase before it is executed.

## Whole-plan acceptance (external; keeps this index open)

All on the owner's machines, after the last phase lands, recorded in the vault
(`WORKLOG.md` and a Validation folder):

1. From the Mac, with no hand-edited file on any node: create a profile for
   `unsloth/LFM2.5-24B-A2B-GGUF` `UD-Q4_K_XL` at 64k context, deploy it to the
   `weak` role on MSI and Joes (model copy included), and see both nodes show
   the identity tags, never the preset section names.
2. Models tab shows Tiel 35B on MSI and Joes as one model with two locations;
   Profiles tab shows which node and role use each profile.
3. A `medium` request from the Mac goes to idle warm Joes and, with Joes busy,
   to MSI; routing history explains the choice; `/v1/models` lists each role
   with its highest `context_length`.
4. Halt from the toolbar on Joes removes it from routing within one request
   retry and cannot be undone from the Mac.
5. Measure a profile on MSI from the Mac: consent preview with the resource
   rundown, then a record bound to the GGUF, `build_info`, GPU, CPU and RAM,
   with the estimate's ranges shown beside the measured values.
6. No engine toggles or padlocks anywhere; node settings reachable from the
   card in one click.

## Retention rule

This index stays until every phase file carries its `done` marks and the
whole-plan acceptance above is recorded. "The last item" in the lifecycle
section below means that full condition, not the last code task. Pending
external acceptance keeps this index open after the code phases finish. When a
phase file is deleted under its own lifecycle, replace its link in the table
with the commit range that landed it.

## Investigation evidence

`investigation/L1.md` to `L8.md` are the finder reports; `verify-A.md` and
`verify-B.md` are the adversarial verification passes; `debate.md` is the
design review whose adjudication README section 8 records. They are snapshots at
`4eccd05d` and are not updated as code changes.

## Note lifecycle

This file is working state in a repository, not documentation.

- When an item lands, append ` — done <sha>` to its heading, where
  `<sha>` is the commit that landed it. An item consciously dropped gets
  ` — dropped` and a `project-note deferred` entry.
- When every item carries one of those marks, delete this file in the
  commit that lands the last item. Nothing in it is kept: the commits and
  the run archive are the record, and `project-note done` records the
  completion.
- In an execute run the orchestrator does this after each unit lands,
  not the unit. A plain session does it as each item lands.
