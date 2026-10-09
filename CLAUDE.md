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
- [AGENTS.md](AGENTS.md): upstream's contributor guide. Its conventions still
  apply.

## Commits

Commit as `Corbin Jurgens <corbinjurgens@gmail.com>` with no trailers: no
`Signed-off-by`, no `Co-Authored-By`, no session links. This overrides the
sign-off rule in `AGENTS.md` and `.cursor/rules/commit-sign-off.mdc`.
