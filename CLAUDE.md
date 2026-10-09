<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CLAUDE.md

This is a **personal fork** of NVIDIA's Personal AI Router (PAIR). It adds
features for running a small, changing group of personal machines as one
inference pool: a backend that runs with no GUI, `weak`/`medium`/`strong` model
tiers, per-machine resource limits, a pause switch, and low idle overhead.

- [WORKLOG.md](WORKLOG.md): the record of all work done on this fork. **Read
  it first, and add an entry for every session**, covering what changed, why,
  how it was verified, and what was left open.
- [FORK.md](FORK.md): the goals and the phased roadmap.
- [research_findings.md](research_findings.md): the original source review the
  roadmap is based on.
- [FORK_DESIGN.md](FORK_DESIGN.md): the technical contract for the fork's
  features (service, node policy, admission, tiers, cancel, model copy).
- [TESTING_ON_PC.md](TESTING_ON_PC.md): the real-machine test checklist.
- [AGENTS.md](AGENTS.md): upstream's contributor guide. Its conventions still
  apply, except that the fork changed the process model (see its notice).

## Where things stand (2026-10-10)

- All work is on branch `fork/groundwork`. All six roadmap phases have a first
  implementation, tested only in a Linux container. Next comes the owner's
  real-PC testing with TESTING_ON_PC.md, then fixes for whatever it finds.
- Key code:
  - `services/nvpair-service` (background service, `call` subcommand)
  - `services/shared/nodepolicy` (policy contract)
  - the proxy's admission and tiers
  - the broker's `policy.go` / `availability.go` / `idlepolicy.go`
  - engine-manager model copy
  - desktop `service-connection.ts`, `RoutingSettings`
- Open items and owner decisions are listed in FORK.md (unchecked boxes and
  "Open decisions") and in WORKLOG.md's latest "Next up".
- Known environment-only test failures (they fail on upstream too):
  - proxy `TestHandleHTTP_RealSocketFlushDeadline`
  - 11 mDNS-dependent tests in `services/tests`
- Agents work in worktrees under `.claude/worktrees/`, listed in
  `.git/info/exclude`. Merge their branches into `fork/groundwork`.

## Commits

Commit as `Corbin Jurgens <corbinjurgens@gmail.com>` with no trailers: no
`Signed-off-by`, no `Co-Authored-By`, no session links. This overrides the
sign-off rule in `AGENTS.md` and `.cursor/rules/commit-sign-off.mdc`.
