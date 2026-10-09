<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fork Work Log

A running record of the work done on this fork: what changed, why, how it
was checked, and what was left open. The plan itself is in
[FORK.md](FORK.md); this file is the history.

**Keeping it up to date:** add an entry for each working session, newest at
the top. List the branch, the commits, what changed and why, how it was
verified (including what was *not* verified), and any decision deferred to the
owner. Update the entry in the same commit as the work when practical.

---

## 2026-10-09: Groundwork, phase 1 start

**Branch:** `fork/groundwork`, based on upstream `develop` at `54d2fe33`.
No pull request opened.

**Context:** an earlier source review judged PAIR a good base for a personal
multi-machine inference pool. It found specific idle-overhead problems and
proposed a six-phase plan, now recorded in `FORK.md`. This session began
phase 1.

### Changes

| Commit | Area | Change |
|--------|------|--------|
| `595b3d5` | `nvpair-ui-broker` | **Workload history stays bounded on clients.** The history caps (newest 10,000 terminal records, none older than 7 days) used to be enforced only inside the persistence flusher, and silently. Electron and the renderer drop an entry only on `workloads:remove`, so finished jobs piled up for the whole session, and without a data dir the store had no bound at all. The caps now apply to every store. A once-a-minute sweep (on the stale-workload ticker) prunes and emits `workloads:remove` for each retired record. `Flush`/`Checkpoint` no longer prune. A removal is withheld while a newer workload with the same `(originatedFrom, workloadId)` survives, because proxy restarts reuse ids. Retirement is local and not broadcast to peers. README updated. |
| `99a13f8` | `nvpair-engine-manager` | **One model-list request per poll.** LM Studio and llama.cpp declare `list_models` and `loaded_models` against the same endpoint, so the 5 s watcher fetched it twice. When both actions are the identical side-effect-free HTTP request (`sameReadRequest`), one response feeds both extractors. Ollama (`/api/tags` vs `/api/ps`) is unchanged. |
| `63c7e92` | `nvpair-proxy` | **Request body cap.** The proxy buffers bodies for failover replay through `http.MaxBytesReader`, with a 64 MiB default set by the new `--max-request-bytes` flag. Oversized requests get 413 and are never dispatched. Other read errors keep the old behavior. README flag table updated. |
| `3aaa0d8` | `desktop/docs` | Regenerated `services-api.md` with `npm run service-contracts:write`: `workloads:remove` is now a notification the broker sends. |
| `774753f` | `desktop` | **Protocol logging follows the log level.** Every broker stdout JSON-RPC line went to the log file through `appendFileSync` on the main thread at any level. A new `isServiceLogLevelEnabled` gates file writes on the level chosen in Service Settings: protocol traffic (`verbose`) is written only at `debug`. The in-memory debug panel buffer and redaction are unchanged. |
| `aa9915a`, `9fa8251` | docs | Added `FORK.md` (goals, phased roadmap, open decisions, test caveats). |
| `8772cf3` | docs | Added this log, a short fork introduction, and a fork notice at the top of `AGENTS.md`. |
| `a73862e` | docs | Recorded the commit identity convention in `CLAUDE.md` (see Conventions set). |
| `88652fa` | docs | Restored commit hashes in this log. |
| *(this commit)* | docs | `FORK.md`: commit hashes and verification notes on finished items, a status line, and a recommended order for next steps. |

### Verification

- Desktop: `npm run lint` (0 warnings), `npm run typecheck`, `npm run test:unit`
  (284 passing), `npm run dead-code:check`, and `npm run service-contracts:check`
  all clean. `node scripts/spdx-headers.mjs` clean.
- Go: `go test ./...` passes for `nvpair-ui-broker` and
  `nvpair-engine-manager`. New tests cover retirement and announcement,
  pair suppression, keeping active records, shared-endpoint detection against
  the shipped manifests, and 413 handling for both engine profiles.
- Failing in this container **on unmodified upstream as well** (compared using a
  clean worktree of `develop`), so they say nothing about these changes:
  - `nvpair-proxy` `TestHandleHTTP_RealSocketFlushDeadline` (socket buffer
    sizes).
  - 11 cross-process tests in `services/tests` (need mDNS/multicast).
- **Not verified:** nothing was run on real hardware, and memory and CPU were
  not measured. Run the cross-process tests on a real machine before relying
  on the workload-history change.

### Decisions deferred to the owner

- Whether a stream cut off after a committed 2xx should count as `failed`
  instead of upstream's documented `completed` (`nvpair-proxy/spec.md` §5.4).
  Left unchanged.
- New Markdown files carry the standard NVIDIA SPDX header because the repo's
  header check requires it.

### Conventions set

- The owner rewrote this session's commits under their own identity and
  force-pushed. From now on, commits are authored as
  `Corbin Jurgens <corbinjurgens@gmail.com>` with no trailers (no sign-off,
  co-author, or session link). This is recorded in `CLAUDE.md`.

### Next up

Lazy tray popup and visibility-gated node-info polling (rest of phase 1). Then
measure PAIR's own memory with engines stopped, Overview open and closed, and
under a synthetic job stream.
