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

**Status (2026-10-09):** phase 1 is partly done on `fork/groundwork`. Nothing
has been measured on real hardware yet. See [Recommended order](#recommended-order)
for what comes next.

The ground rules in [AGENTS.md](AGENTS.md) and `.cursor/rules/` still apply.
Runtime behavior belongs in `services/`; `desktop/` only relays commands and
renders state.

## Phase 1: Reduce idle overhead (in progress)

Done on `fork/groundwork`. Each item has unit tests and passes the repo's
desktop checks and its service's Go tests. None has been measured for memory
or CPU.

- [x] **Workload history stays bounded on the desktop.** The broker used to trim
  its history without telling clients, so Electron and the renderer kept every
  finished job for the whole session. The caps now apply to every broker store,
  and the broker sends `workloads:remove` for each record it drops.
  (`nvpair-ui-broker`, `595b3d5`; generated API doc `3aaa0d8`; research §5A)
  *Still to check:* the cross-process workload tests need mDNS and could not
  run here (see [Known local test caveats](#known-local-test-caveats)).
- [x] **One model-list request per poll.** LM Studio and llama.cpp were queried
  twice every 5 s for the same endpoint. (`nvpair-engine-manager`, `99a13f8`;
  research §5D)
- [x] **Proxy request size cap.** Bodies the proxy buffers for failover are
  limited to 64 MiB by default (`--max-request-bytes`); larger requests get a
  413. (`nvpair-proxy`, `63c7e92`; research §5E) Spilling very large
  replayable requests to a temporary file, which the research mentions as an
  option, is left until a real need appears.
- [x] **Protocol logging follows the log level.** Broker JSON-RPC traffic is
  written to the log file only when the level is `debug`. It is no longer
  written synchronously on every message at the default `warn`. (`desktop`,
  `774753f`; research §5B, partly; see the follow-up below)

Next:

- [ ] Before building much further, run the pinned base plus this branch on
  each machine and confirm the basic workflow works there (research:
  "My recommendation").
- [ ] Create the tray popup only when it is opened, destroy it on dismiss, and
  initialize only the stores it displays. Consider a native-menu-only tray
  mode.
- [ ] Run the 2 s node-info poller only while a window is visible. Keep
  telemetry-only changes from triggering model and discovery notifications.
- [ ] Measure before and after: PAIR's own memory with engines stopped, with
  Overview open, with Overview closed, and under a synthetic job stream.
  Phase 2 adds a fourth case: the service running with no Electron.

Follow-ups to finished items (smaller, can wait):

- [ ] Logging: research §5B also asks for a bounded asynchronous writer
  instead of `appendFileSync`, and for skipping the work of building an entry
  (including redaction) for lines that will not be written.
- [ ] Workload history: Electron's `seedWorkloads` only adds entries from a
  fresh baseline and never drops ones the broker no longer has, so a missed
  removal is never repaired (research §5A).
- [ ] Model polling: check residency often and reconcile the installed
  inventory less often, keeping occasional reconciliation because models can
  change outside PAIR (research §5D).

## Recommended order

1. **Lazy tray popup** (phase 1). Small, desktop only, likely the biggest
   remaining idle-memory saving. The hidden popup window is built at startup
   in `desktop/src/electron/tray.ts`.
2. **Visibility-gated node-info poller** (phase 1). Small, desktop only.
   `node-info-poller.ts` already has start and stop hooks; tie them to window
   visibility instead of the backend connection.
3. **Manual nodes stored in the backend** (phase 2). Small to medium, mostly
   Go, and fully testable here. Needed before the GUI can be optional.
4. Then either **phase 2** (start with Option A, the Go supervisor) or
   **phase 3** (the pause switch), if pausing for gaming matters more day to
   day.

Items 1 and 2 need a before-and-after memory check on a real machine. The
measurement item in phase 1 covers that.

## Phase 2: Make the GUI optional

- [ ] Keep the broker running independently as a per-user service that
  listens on a local socket or Windows named pipe. Electron and the TUI attach
  to it and detach from it.
  - Option A: a small Go supervisor that owns the broker's stdio and exposes
    attach/detach. Less disruptive.
  - Option B: refactor the broker so its runtime does not depend on any client
    session. Cleaner, but more work.
- [ ] Move persistence of manual nodes from Electron into the service (today
  Electron stores the list and replays it into the backend).
- [ ] Make the desktop and TUI reconnect to a running service instead of
  spawning their own broker.

Until then, `nvpair-tui` from the standalone services bundle runs PAIR
without Electron. It must stay open, and must not run alongside the desktop
app, since both try to own the same services and ports.

## Phase 3: Reliable node availability

- [ ] Add node states `Available`, `Draining`, and `Paused`. Pausing stops new
  work, finishes or cancels running jobs, unloads models or stops the managed
  engine, and persists until you turn it off.
- [ ] Keep requests from waking an engine whose saved intent is Off.
- [ ] One PAIR policy for starting models on demand and unloading them when
  idle, across engines. Today this is engine-specific; llama.cpp's launch
  settings include a 300 s idle sleep.
- [ ] Coordinate engine launch-setting restarts with draining.

## Phase 4: Resource admission and model profiles

- [ ] A profile is a concrete model, engine, quantization, context limit,
  launch settings, and capabilities.
- [ ] Each machine has final say over what it accepts. The originating router
  picks a candidate, and the destination accepts, queues, or rejects based on
  its pause state, the resident model, current reservations, memory headroom,
  and whether accepting would need a model switch. Local and peer requests go
  through the same path.
- [ ] Settings for resident model count and concurrent request count are
  separate. The Mac is budgeted as one unified-memory pool; the RTX machines
  track RAM and VRAM separately.
- [ ] A node-wide reservation that covers every engine on the machine, so
  "one resident model" holds across engines and model switches. Use
  conservative memory estimates.
- [ ] Do not present a GPU-utilization percentage as an enforced limit unless
  the engine can actually enforce it.

## Phase 5: Tiered routing

- [ ] `weak` / `medium` / `strong` aliases resolve to eligible profiles. Tools,
  vision, structured output, and context length are hard requirements. The
  proxy reads only `model` today, so it must parse enough of the request to
  check them.
- [ ] A stronger model may answer a weaker tier when allowed.
- [ ] Prefer an already-loaded model that fits. A degradation policy decides
  whether a stronger request may fall back to a weaker tier.
- [ ] Routing metadata always names the concrete model that answered.

## Phase 6: Remote management

- [ ] Edit model profiles remotely, coordinated with running requests.
- [ ] Easier connection setup: reliable manual peers over LAN or Tailscale
  first, with identity and peer configuration owned by the service. Later,
  invitations and address updates built on the existing cluster trust.
  Remote model download and delete already exist and can be reused.
- [ ] Direct model-file transfer between machines, with resume, checksums,
  and integration with the destination's model storage.
- [ ] Jobs: remote cancel controls, and "cancel and regenerate on another
  device". Never splice two models' output into one answer. Moving a
  generation to another machine mid-stream is a separate, much harder feature
  and out of scope (research §6).

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
