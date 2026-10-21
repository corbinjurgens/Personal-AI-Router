<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Common brief for cloud investigation runs (read first)

You are doing ONE investigation for the PAIR fork (NVIDIA Personal AI Router, personal fork). Investigation only: do not change product code, tests or docs, and do not open pull requests. Your output is one Markdown report pushed to your own `plan/` branch.

## Repository

- GitHub: corbinjurgens/Personal-AI-Router (PUBLIC repository). Product base: `fork/groundwork` at `7e270f1` (record the exact SHA you inspected; if the branch moved, say so and inspect the newer head).
- Layout: desktop/ (Electron + React client of the services), services/ (Go: broker nvpair-ui-broker, nvpair-proxy, nvpair-engine-manager, nvpair-tui, nvpair-workload-manager, scanner, cluster/settings/manual-nodes workers, shared/ packages, tests/ = cross-process tests driving real binaries), docs/.
- Path shorthand: B/=services/nvpair-ui-broker/, P/=services/nvpair-proxy/, E/=services/nvpair-engine-manager/, S/=services/nvpair-node-scanner/, T/=services/nvpair-tui/ui/, N/=services/shared/, X/=services/tests/, D/=desktop/src/.
- Rules that bind the code: services/ is the source of truth and desktop only relays and renders. Prefer services/*/spec.md over READMEs; Go source is authoritative. A JSON-RPC change updates producer, broker relay, every consumer, the desktop bridge (desktop/src/electron/service-bridge/), tests and docs together. desktop/docs/services-api.md, CHANGELOG.md and services/versions.json are generated. No TS type casts, no legacy fallbacks, "engine" not "backend" in new copy, never log prompts, bodies, PINs or keys.
- Fleet: Mac (orchestrator; never serves a model), MSI (Windows + RTX), PC3 (Windows; llama.cpp). Engines: llama.cpp (router mode with --models-preset) and the owner's Strata fork (Strata integration is deferred by the owner). Client: OpenCode over OpenAI-compatible Chat Completions. No Ollama in the owner's fleet, though the code still supports it.

## Where the plans are

Package plans live on `plan/*` branches (each holds `plans/<slug>.md`), never on product branches:

- `plan/ne-serving-envelope` — NE: per-engine serving envelope, identity, strict readiness (Tasks 1-6 landed incl. Part B strict readiness; a readback transport-classification fix is pending in `integration/join25`; Part D deferred).
- `plan/pa-paused-advertisement` — PA (T4, T5, T6 landed; T1-T3, T7 remain).
- `plan/j-routing-diagnostics` — J (T1 landed; T2-T6 remain).
- `plan/b-opencode-shapes` — B extension for OpenCode request shapes (T2-T4 and T7 landed; T5 pending in `integration/join25`; T6 docs remain; `reasoningReplay` stays required-false until the owner answers). OpenCode inventories: `plan/b-opencode-inventory-msi`, `plan/b-opencode-inventory-pc3`.
- `plan/xa-remote-config-acceptance` — XA (complete).
- This branch, `plans/context/`: redacted copies of plans that otherwise live only on the orchestrator: `i-unified-dispatch.md` (I: unified dispatch; T2, T3a, T3b, T4a, T8 landed; T4b on `roadmap/i-public-lifecycle` under repair after review; T5-T7 remain), `g-egress-lease.md` (G: T1-T2 landed, public egress disabled; T3 on `roadmap/g-t3` had a review RED and is deferred until after SR 5c and PA 1-2; T4 remains), `sr5-rescope.md` (SR Task 5/6 rescoped into 7 tasks; 5a-i, 5a-ii, 5b, 5d landed; 5c on `roadmap/sr-5c` under repeated review/repair; 6a (now docs only) and 6b remain). Paths in them written as `<run-dir>`, `<repo>`, `<vault>` or `$WASABI_VAULT` refer to private files you cannot see; work without them and say where that limits a conclusion.

Landed history: `git log --first-parent fork/groundwork` shows one merge per landed unit; merge subjects name the package and task (for example "Merge roadmap/ne-t6c ..."). Use them to map plan tasks to commits.

Unlanded product branches you may inspect (all pushed): `integration/join25` (pending Windows gate), `roadmap/i-public-lifecycle` (I 4b), `roadmap/sr-5c`, `roadmap/g-t3`. The first round of cloud reports lives on `plan/inv-*` branches at `plans/reviews/<slug>.md` (audit-pa-j-b, audit-ne-sr, revalidate-remaining, debate-i-g, plan-daily-acceptance); read the one your handoff names.

Coordinator rulings since the first round (treat as settled):
- NE readback: transport failures (connection error, timeout, body read error, any non-200 including 3xx; redirects not followed on observation reads) make that sweep's observation UNKNOWN with no drift latch; only semantic mismatches latch.
- Broker lane order: SR 5c, then PA 1-2, then G3, then G4. G3/G4 and the I5 hybrid / I6 desktop public surfaces come after the daily milestone items.
- SR: `declaredResidentCount` is required, 1..256, bound to the policy revision, fail closed. `Load.ContextLength` equals the policy's context tokens exactly. SR 6a is docs only.
- SR 5c reconcile: level-triggered state per (engine, model, revision). In-flight and pending-observation are facts, cleared only by a fresh confirming observation, a load failure, revision replacement/removal, or a bounded observation timeout. Pause, Off and drain block new loads but preserve those records. Failures retry, bounded.
- Manual node references: a bounded string kind that rejects control and format (Unicode Cf) characters.

## Owner product decisions you must respect

- Each node declares its own optional weak/medium/strong engine/model roles; a router consumes the declaring node's roles.
- Local Off/pause wins over remote start, resume and residency. Remote configuration edits remain permitted.
- Pairing marks a peer as "my own device" (remote changes allowed) or friend/locked (blocked); enforced target-locally.
- Interrupted streams: the user sees only delay or pause-resume; an error only if no same-model continuation exists.
- Plain direct-model output caps are deferred (NE Part D after I 4b). Strata is deferred.
- Daily local/peer milestone (the near-term goal, separate from full completion): one approved serving configuration per host (or a named origin-only host), tier and peer routing with real models, desktop remote configuration and switching with target-local permission and local veto, headless service with Windows service-process restart, honest completed/failed/cancelled outcomes including interrupted streams, accurate-or-unknown telemetry.

## Evidence rules

- Every claim about the code cites `path:line` at the SHA you inspected. Separate VERIFIED (you read the code or ran a check) from INFERRED.
- You may run `go build`, `go vet` and module unit tests (`go test ./...` inside one services module directory), `npm ci` plus desktop lint/typecheck/unit tests if your environment allows. Do NOT run `services/tests` cases (they start real product binaries and listeners); `go vet ./...` and `go test -list . ./...` there are fine. Never start engines or models.
- Classify each finding: RED (behaviour, safety, data-integrity or public-contract violation, reproduced or proven from code), UNVERIFIED (required evidence missing or only provable on a real machine), LOW (no observable effect). For each RED or UNVERIFIED give: claimed invariant (quote the plan sentence), executable probe or code path, observed result, affected state, and a recommended disposition (repair now / amend plan / defer / owner question).
- Report a plan-vs-code gap as a finding, not as something to fix. Do not redesign settled owner decisions; you may argue against one in a clearly separated "Owner challenges" section.

## Saving (exactly this)

1. From `fork/groundwork`, create the branch named in your handoff (`plan/inv-<slug>`).
2. Write the report to `plans/reviews/<slug>.md` with the SPDX header used at the top of this file.
3. `git add plans/reviews/<slug>.md && git commit --no-signoff -m "review: <slug>"` and `git push -u origin plan/inv-<slug>`. Touch no other file. No pull request.
4. Reply with branch, commit SHA, a verdict line `RED n UNVERIFIED n LOW n`, and a 10-line summary.

The repository is public: no personal data, secrets, host names, IP addresses, usernames or local file paths in the report.
