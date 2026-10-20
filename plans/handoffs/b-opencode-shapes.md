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
1. From fork/groundwork, create branch `plan/b-opencode-shapes`.
2. Write the plan to `plans/b-opencode-shapes.md` (new top-level `plans/` folder; this branch only; it is never merged into product branches). Add the SPDX header used by other .md files in the repo.
3. `git add plans/b-opencode-shapes.md && git commit --no-signoff -m "plan: b-opencode-shapes"` and `git push -u origin plan/b-opencode-shapes`. Do not open a pull request. Do not touch any other file.
4. Reply with the branch, commit SHA, and a 10-line summary including your owner questions.
Note: the repository is public, so the plan becomes public. Do not include personal data, secrets, host names, IPs, usernames or local file paths in it.


PACKAGE: B extension — qualification for the OpenCode request shapes actually used. SLUG: b-opencode-shapes

Background: package B (finite qualification and freshness) landed (strict capacity from a node-local evidence file and engine identity; qualification identity/freshness on Profile and Advertisement; effective body/tokenizer/template/context/output/config/content validation before side effects). Main code: P/{local_qualification.go,admission.go,tiers.go,ingress.go}, N/nodepolicy/{policy.go,wire.go}, N/enginesettings/settings.go, B/{policy.go,enginesettings.go}, E/{proc.go,settings.go,models.go}, docs/local-qualification.mdx.
Owner daily milestone item: 'B extended only for the OpenCode request shapes actually used (tools, reasoning replay if enabled).' The client is OpenCode over OpenAI-compatible Chat Completions; engines are llama.cpp (router mode with --models-preset, finite per-model presets; NE Task 2 landed) and the owner's Strata fork (OpenAI-compatible server; NE Task 3 prepared-launch validation in review).
Ask: (1) Determine from OpenCode's documented/known Chat Completions usage which request fields matter (tools/tool_choice, parallel tool calls, streaming with usage, reasoning fields and reasoning replay in assistant messages, response_format, max_tokens vs max_completion_tokens, stop, temperature, etc.) — state clearly which facts come from OpenCode source/docs you could read and which are assumptions to confirm on MSI. (2) Map each shape to what the current admission/qualification code accepts, rejects or ignores, citing code. (3) Plan the minimal extension so those shapes are validated and qualified per engine/model (template supports tools? reasoning format?), without accepting shapes the engine cannot serve, and without generic support for unused shapes. Include fixtures that record real request bodies only as synthetic redacted examples (never log prompts). Owner question candidates: whether reasoning replay is enabled in the owner's OpenCode config.
