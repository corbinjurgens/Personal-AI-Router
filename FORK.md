<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fork Roadmap

This fork builds on PAIR's Go services to run a small, changing group of
personal machines (an RTX desktop or two and a Mac) as one inference pool. What
it adds on top of upstream:

- a backend that keeps running with no GUI open;
- `weak` / `medium` / `strong` model tiers that resolve to concrete model
  profiles;
- one resident model per machine by default, with RAM/VRAM admission;
- a per-machine pause switch (for gaming, for example) that automatic loading
  cannot override;
- low idle overhead from the desktop app.

Base: upstream `develop` at `54d2fe33` (October 9, 2026), which is ahead of
release v0.1.1. It is the better base because it includes the unified proxy,
llama.cpp support, and remote engine settings.

The plan comes from [research_findings.md](research_findings.md), a source
review of `54d2fe33` done before the fork. Each phase below follows its
recommendations, and the [Open decisions](#open-decisions) say where this plan
departs from it.

A session-by-session record of changes is in [WORKLOG.md](WORKLOG.md).
Each finished item below names its commit; the log has the full detail.

**Status (2026-10-10):** every phase has a first implementation on
`fork/groundwork`, tested in a Linux container and in an end-to-end check
there. None of it has run on Windows, macOS, a GPU or two real machines yet.
[TESTING_ON_PC.md](TESTING_ON_PC.md) is the checklist for that. The technical
contract is [FORK_DESIGN.md](FORK_DESIGN.md). The biggest remaining gap is the
desktop UI: apart from the tray's pause toggle, the new features (policy,
tiers, cancel, model copy) have no desktop screens yet. Use
`nvpair-service call` for them.

## Phase 1: Reduce idle overhead (done, needs measuring)

- [x] Workload history stays bounded on the desktop (`595b3d5`, `3aaa0d8`;
  research §5A).
- [x] One model-list request per poll (`99a13f8`; §5D).
- [x] Proxy request size cap, 64 MiB by default (`63c7e92`; §5E). Spilling
  very large requests to disk is left until it is needed.
- [x] Protocol logging follows the log level (`774753f`); bounded async log
  writer, and protocol frames skipped below `debug` (`f2c3ba6`, `d3f3d3d`;
  §5B).
- [x] Desktop workload baseline reconciles instead of only adding (`49eb7a8`).
- [x] Residency checked every 5 s, inventory once a minute (`351a55d`).
- [x] Lazy tray popup, destroyed after 60 s hidden, loading only the stores it
  shows. Optional menu-only tray. (`2f75db9`, `9c70c89`, `dad9c9b`)
- [x] Node-info poller runs only while a window is visible. Readings-only
  changes push only metrics. (`86e5dab`; §5C)
- [ ] Measure memory before and after on a real PC (TESTING_ON_PC.md §9).

## Phase 2: Make the GUI optional (service done, desktop pending)

- [x] `nvpair-service`: per-user background service that owns the broker,
  with multi-client attach and detach, broker restart, logs, autostart, and
  `status` / `stop` / `call` subcommands (merge `32c7820`, `92ce606`).
- [x] The TUI attaches to the service. Quitting leaves inference running.
- [x] Manual nodes are saved by the backend, and hostnames are re-resolved.
- [x] The desktop app attaches to the service instead of spawning a broker
  (detached start, reconnect, quit detaches, tray "Stop background service
  and quit"), and migrates Electron's old manual-node file (`e5806b4`..`fd95cdf`).

## Phase 3: Reliable node availability (done)

- [x] `available` / `draining` / `paused`, persisted. Pausing drains or
  cancels running work, unloads models, and optionally stops engines without
  changing their saved intent (merge `db99d82`).
- [x] Engines saved Off are never started on demand.
- [x] Idle unload, idle stop, and start-on-demand, all configurable and off by
  default except start-on-demand.
- [x] Settings restarts drain the engine first.
- [ ] Advertise paused nodes to peers. Today peers learn from a fast `503`.

## Phase 4: Resource admission and model profiles (done)

- [x] Node policy file with profiles (`eb3b58e`).
- [x] Destination-side admission in the proxy covering availability, wake,
  concurrency, resident model count across engines with model switching, and
  the memory budget. Bounded queueing; the router hands a busy request
  straight to the next machine (merge `e35a283`).
- [ ] Automatic memory estimates. Today `memoryBytes` is declared per profile.

## Phase 5: Tiered routing (done)

- [x] `weak` / `medium` / `strong` resolve across machines and, on
  OpenAI-compatible routes, across engines. Capability and context checks,
  warm-model preference, and stronger/weaker fallback.
- [x] `X-PAIR-Model` / `-Engine` / `-Node` / `-Tier` headers, and
  `requestedModel` on jobs. Tiers appear in `/v1/models`.

## Phase 6: Remote management (done, UI pending)

- [x] Remote policy and pause editing through paired nodes (`nodeId`).
- [x] Model copy between machines, with resume and sha256 checks, for Ollama,
  llama.cpp and LM Studio (merge `7862aca`).
- [x] Job cancel, and cancel-and-regenerate on another machine, from any node
  (`82bbc29`).
- [x] Manual peers by hostname (for example Tailscale) persist.
- [x] Tray pause/resume toggle for this PC.
- [ ] Desktop UI for policy and tiers, job cancel, remote pause and model copy.

## Open decisions

- **A stream cut off after a 2xx counts as `completed`.** Upstream does this on
  purpose (`services/nvpair-proxy/spec.md` §5.4: "committed 2xx, truncated by
  the node dying"). For success accounting and "regenerate on another device",
  `failed` with a clear reason may suit this fork better. The research (§6)
  calls it a correctness issue to fix before trusting Jobs as success
  accounting. It is a semantic change to the Jobs view and the scheduler
  inputs, so it waits on your decision.
- **Staying in sync with upstream.** Do we merge upstream `develop` regularly, or
  pin a revision and cherry-pick? Fork-only changes are easier to carry if each
  stays small and inside one service.
- **Packaging and distribution.** NVIDIA's signing and update channel are not
  part of the public build, so the fork needs its own if you want installers
  or auto-update. The build also downloads NVIDIA UI CSS at build time; vendor
  or cache it for reproducible offline builds. Engines and models keep their
  own licences (research §1).
- **`versions.json` entry for nvpair-service.** Without one the service
  reports version 0.0.0 and upstream's CI check fails. Upstream forbids hand
  edits to that file; in this fork you may choose to add the entry.
- **Release automation.** Upstream CI rejects hand edits to
  `services/versions.json` and `CHANGELOG.md`, and expects a
  `pair-release-intent:v1` block in each PR description. Decide whether the fork
  keeps that pipeline or replaces it.

## Known local test caveats

These fail identically on unmodified upstream code in a sandboxed container,
and pass in upstream CI:

- `TestHandleHTTP_RealSocketFlushDeadline` (`services/nvpair-proxy`) depends on
  kernel socket buffer sizes.
- Eleven cross-process tests in `services/tests` (discovery, broker
  subscription, and workload-manager rehydration and broadcast) need working
  mDNS/multicast networking.

Run them on a real machine before trusting a change to discovery or workload
broadcast.
