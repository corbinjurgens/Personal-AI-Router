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

## 2026-10-10: All six phases, first implementation

**Branch:** `fork/groundwork`. The owner asked for the entire plan, using
subagents economically. Partway through, credits ran low, so the rest runs on
cheaper models, one agent at a time. Agents worked in separate git worktrees
and their branches were merged here.

**How the work was split:**
- Two read-only agents mapped routing and engine control.
- I wrote the shared contract (`services/shared/nodepolicy`) and
  `FORK_DESIGN.md`.
- Build agents each took one area: phase 1 desktop, the service and TUI, the
  proxy, the broker with engine-manager, and model transfer.

### Changes

| Commit | Area | Change |
|--------|------|--------|
| `351a55d` | engine-manager | Residency checked each tick, inventory once a minute or after a PAIR action. |
| `eb3b58e` | shared, docs | `nodepolicy` contract (policy file, defaults, validation, broker↔proxy wire) and `FORK_DESIGN.md`. |
| `978562c` (merge) | desktop | Lazy tray popup and menu-only mode; visibility-gated node poller with metrics-only pushes; bounded async log writer; protocol frames skipped below debug; workload baseline reconciliation; Overview opens on a pairing invite. |
| `7862aca` (merge) | engine-manager | Model copy between paired nodes (`/v1/models/files`, `/v1/models/file`, `engine:remote-copy-model`) for Ollama, llama.cpp and LM Studio, with resume and sha256 checks. |
| `db99d82` (merge) | broker, engine-manager, workload-manager | Policy store and `policy:get/set`, `node:set-availability` with drain, cancel, unload and stop; `engine:wake/sleep/intent/unload-model`; idle policy; settings-restart drain; remote policy through `ec`; `workloads:cancel` relay. Merge conflicts in engine-manager README and spec resolved by hand. |
| `82bbc29` | broker, workload-manager | `workloads:cancel` carries optional `engine` and `runId`, since ids repeat across engines. |
| `32c7820` (merge) | service, TUI, manual-nodes, build | `nvpair-service` (multi-client attach, broker restart, logs, autostart); `servicectl` and `ipc.ListenPrivate`; the TUI attaches to the service; manual nodes persisted; the service added to build, installers and desktop bundling. |
| `e35a283` (merge) | proxy | Destination admission, tiers across nodes and engines, routing headers and `requestedModel`, `workload/cancel` with regenerate. |
| `92ce606` | service | `nvpair-service call <method> [json]`; short socket path fallback when the app data path is too long for a Unix socket. |
| `e5806b4`..`fd95cdf` | desktop | Electron attaches to `nvpair-service` (endpoint mirror of servicectl, connect-or-start, socket JSON-RPC client with reconnect, `service/log` redacted into logs); quit detaches; tray pause toggle and "Stop background service and quit"; one-time manual-node migration, store and replay removed. Sonnet agent; 379 desktop tests pass. |
| *(this commit)* | docs | `TESTING_ON_PC.md`, `FORK.md` status, this entry. |

### Verification

- Go `vet` and `test` pass in shared, proxy, broker, engine-manager,
  workload-manager, manual-nodes, service and TUI. The one exception fails
  on unmodified upstream too: proxy `TestHandleHTTP_RealSocketFlushDeadline`.
- Desktop: lint (0 warnings), typecheck, 323 unit tests, dead-code check,
  contracts check and SPDX check all clean.
- `services/build.sh` stages all 14 binaries.
- **End-to-end in the container** (real binaries, isolated config dir): the
  service started the broker, and through `nvpair-service call` I exercised
  `policy:get`, pause and resume, `policy:set` (persisted, with an invalid
  tier rejected), `workloads:cancel` relay, and `stop`. No processes were left
  behind.
- **Not verified:** Windows and macOS, real engines and GPUs, two machines,
  pipe ACLs, autostart, memory savings. The `services/tests` cross-process
  suite still needs mDNS.

### Decisions and deviations recorded

- Recorded in `FORK_DESIGN.md` and the agents' component docs:
  - `/v1/models/files` streams NDJSON.
  - `engine:intent` and `engine:unload-model` were added to engine-manager.
  - Idle stop applies only to engines saved On.
  - `Workload.engine` stays the facade the request entered on.
  - A paused node rejects its own self candidate once per request.
- Left for the owner:
  - A `versions.json` entry for `nvpair-service`.
  - The stream-abort classification.
  - Docs that still describe the TUI as owning its broker
    (`docs/architecture.mdx`, `.cursor/rules/system-architecture.mdc`,
    `communication-layers.mdc`).

### Next up

- Desktop UI for policy and tiers, cancel, remote pause and model copy.

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
| `e5d9083` | docs | `FORK.md`: commit hashes and verification notes on finished items, a status line, and a recommended order for next steps. |

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

### 2026-10-10 addendum: roadmap checked against the research

The owner pushed `research_findings.md` (`1377cd7`), the original source
review. Comparing it with `FORK.md`: same six phases in the same order, and
phase 1's finished items match findings §5A, B, D and E. Points that were
missing from `FORK.md` and are now added:

- a trial run on each machine before building much further;
- follow-ups to finished items: an async bounded log writer (§5B), Electron's
  add-only baseline seeding (§5A), and separate residency and inventory polling
  (§5D);
- `nvpair-tui` as the interim way to run without Electron;
- one start-on-demand and idle-unload policy across engines; a node-wide
  reservation across engines; no fake GPU-percentage limits;
- capability parsing in the proxy for tiers;
- service-owned peer identity, invitations, and model-file integration;
- job cancel and regenerate on another device, never splicing two answers
  (§6);
- packaging, signing, offline CSS vendoring, and engine and model licences
  (§1).

The one deliberate departure is the stream-abort classification. The research
calls it a correctness issue; the plan keeps it as an open decision because
upstream documents the current behavior on purpose. `research_findings.md`
gained the SPDX header the repo's check requires, and `CLAUDE.md` links it.

### Next up

Lazy tray popup and visibility-gated node-info polling (rest of phase 1). Then
measure PAIR's own memory with engines stopped, Overview open and closed, and
under a synthetic job stream.
