<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

You are writing ONE execution plan for the PAIR fork (NVIDIA Personal AI Router, personal fork). Planning only: do not implement product code.

REPOSITORY
- GitHub: corbinjurgens/Personal-AI-Router (PUBLIC repository). Base branch: fork/groundwork at commit 17727f1 (or newer; record the exact SHA you planned against).
- Layout: desktop/ (Electron + React client of the services), services/ (Go: broker nvpair-ui-broker, nvpair-proxy, nvpair-engine-manager, nvpair-tui, nvpair-workload-manager, scanner, cluster/settings/manual-nodes workers, shared/ packages, tests/ = cross-process tests driving real binaries), docs/.
- Rules that bind every plan: services/ is the source of truth, desktop only relays and renders. Prefer services/*/spec.md over READMEs; Go source is authoritative. A JSON-RPC change updates producer, broker relay, every consumer, the desktop bridge (desktop/src/electron/service-bridge/), tests and docs in the same change, then `npm --prefix desktop run service-contracts:check`. Never hand-edit desktop/docs/services-api.md (regenerate with `npm --prefix desktop run service-contracts:write`), CHANGELOG.md or services/versions.json. No type casts in TS, no legacy fallbacks, say "engine" not "backend" in new copy, never log prompts/bodies/PINs/keys. Every new file has the two-line SPDX header (see existing files). Commits use `git commit --no-signoff`. In new Go tests listen on 127.0.0.1:0.
- Path shorthand used below: B/=services/nvpair-ui-broker/, P/=services/nvpair-proxy/, E/=services/nvpair-engine-manager/, S/=services/nvpair-node-scanner/, T/=services/nvpair-tui/ui/, N/=services/shared/, X/=services/tests/, D/=desktop/src/.
- PC3 never runs -race (no C toolchain); Mac and MSI supply race evidence. Windows tests that start PAIR listeners run serialized on PC3/MSI with firewall checks.
- Fleet: Mac (orchestrator; never runs a model; may run Go module unit/race tests and desktop tests but NEVER services/tests cases), MSI (Windows + RTX; strong/medium/weak engines), PC3 (Windows; LFM2.5-24B-A2B via llama.cpp; runs Windows gates incl. services/tests). Engines: llama.cpp and the owner's Strata fork only; no Ollama. Client: OpenCode over OpenAI-compatible Chat Completions.

PLAN FORMAT (an executor with no other context, usually Codex gpt-6.1-sol, will follow it literally; tasks are dispatched as independent committed units, in parallel when they do not depend on each other)
# Execution plan: <topic>
## Goal  (finished observable state)
## Change scope  (Intent / Scope: Minimal | Bounded improvement | Structural, and what may change / Basis / Compatibility / Non-goals)
## Context the executor needs  (verified facts from the code: exact paths, symbols, current behaviour; say what you could not verify)
## Decisions (settled — do not revisit)  (from the owner decisions below; rejected alternatives as "Not X, because Y")
## Decisions made while writing this plan  (your own choices, so the owner can veto)
## Tasks  (### Task N — outcome, then unbulleted fields:)
  **Files:** exact repo paths created/modified/deleted incl. tests/docs/generated files
  **Depends on:** task numbers or "none" (+ external prerequisites by package name)
  **Complexity:** mechanical | judgment
  **Do:** exact steps; exact snippets for wire shapes, user-facing strings, commands
  **Verify:** commands with pass conditions; name new top-level tests; state which checks run on Mac and which need PC3/MSI Windows
  Each task = one independently committable outcome that leaves the tree green. No two tasks share a file unless one depends on the other. Pull shared-file edits (manifests, generated docs) into their own task.
## Verification (whole-plan)
## Do not
## If stuck  (what to retry, when to stop and report)
## Owner questions  (only choices that change product behaviour, cost, security or data risk)

Before finishing, re-read the plan as a stranger: every task executable from the file alone, every Verify pass/fail, no unresolved "or"/"consider"/"as needed", dependency graph acyclic, Files disjoint across independent tasks. Cite code as path:line from the SHA you planned against.

SAVING (exactly this)
1. From fork/groundwork, create branch `plan/xa-remote-config-acceptance`.
2. Write the plan to `plans/xa-remote-config-acceptance.md` (new top-level `plans/` folder; this branch only; it is never merged into product branches). Add the SPDX header used by other .md files in the repo.
3. `git add plans/xa-remote-config-acceptance.md && git commit --no-signoff -m "plan: xa-remote-config-acceptance"` and `git push -u origin plan/xa-remote-config-acceptance`. Do not open a pull request. Do not touch any other file.
4. Reply with the branch, commit SHA, and a 10-line summary including your owner questions.
Note: the repository is public, so the plan becomes public. Do not include personal data, secrets, host names, IPs, usernames or local file paths in it.


PACKAGE: XA — remote configuration cross-process acceptance harness. SLUG: xa-remote-config-acceptance

Catalogue entry (owner's roadmap):
Files (expected): X/{remote_configuration_test.go,service_harness_test.go,service_processes_test.go,main_test.go}; services/readme.md. Extend only if the code proves another file is necessary, and say why.
Depends-on: SR (served/resident config; Tasks 1a/1b/2/3a/3b/4/5d landed, 5a/5b/5c pending on NE Task 4), E3 (target-owned engine switch, landed), TC (terminal canonical controls, landed), CH6 (isolated cross-process fixtures, partly landed).
Outcome: real RPC fixtures cover status, conflict, veto, roles, serving/residency and idempotent switch; native/live execution remains pending when unattended.
Daily milestone (owner): requires 'real OpenCode/XA evidence on MSI and PC3, with intended engines On (no automatic remote wake/switch promised)' and 'desktop remote configuration and switching honoring target-local permission and local veto'. XA is the local/peer slice of that.
Already landed and relevant: PG1-PG5 pairing/permission (remote mutations need fresh target-local permission; services/tests has TestRemoteManagement* cases), E3 switch receipts (X/engine_switch_test.go), OB saved-Off origin (X/off_origin_test.go, TestOBOffOrigin), local veto R4, roles R1-R3.
Ask: plan the harness so routine cases use fake engines (no live models) and split cleanly: what can land now versus what waits on SR 5a-5c; reuse existing fixtures in services/tests (read service_harness_test.go, service_processes_test.go, secure cluster/pairing fixtures) instead of new harnesses. Mac may only vet and `go test -list` services/tests; all executions are PC3 Windows gates, so state exact `-run` selectors per task.
