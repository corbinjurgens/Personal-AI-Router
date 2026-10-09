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

A session-by-session record of changes is in [WORKLOG.md](WORKLOG.md).

The ground rules in [AGENTS.md](AGENTS.md) and `.cursor/rules/` still apply.
Runtime behavior belongs in `services/`; `desktop/` only relays commands and
renders state.

## Phase 1: Reduce idle overhead (in progress)

Done on `fork/groundwork`:

- [x] **Workload history stays bounded on the desktop.** The broker used to trim
  its history without telling clients, so Electron and the renderer kept every
  finished job for the whole session. The caps now apply to every broker store,
  and the broker sends `workloads:remove` for each record it drops.
  (`nvpair-ui-broker`)
- [x] **One model-list request per poll.** LM Studio and llama.cpp were queried
  twice every 5 s for the same endpoint. (`nvpair-engine-manager`)
- [x] **Proxy request size cap.** Bodies the proxy buffers for failover are
  limited to 64 MiB by default (`--max-request-bytes`); larger requests get a
  413. (`nvpair-proxy`)
- [x] **Protocol logging follows the log level.** Broker JSON-RPC traffic is
  written to the log file only when the level is `debug`. It is no longer
  written synchronously on every message at the default `warn`. (`desktop`)

Next:

- [ ] Create the tray popup only when it is opened, destroy it on dismiss, and
  initialize only the stores it displays. Consider a native-menu-only tray
  mode.
- [ ] Run the 2 s node-info poller only while a window is visible. Keep
  telemetry-only changes from triggering model and discovery notifications.
- [ ] Measure before and after: PAIR's own memory with engines stopped, with
  Overview open, with Overview closed, and under a synthetic job stream.

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

## Phase 3: Reliable node availability

- [ ] Add node states `Available`, `Draining`, and `Paused`. Pausing stops new
  work, finishes or cancels running jobs, unloads models or stops the managed
  engine, and persists until you turn it off.
- [ ] Keep requests from waking an engine whose saved intent is Off.
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

## Phase 5: Tiered routing

- [ ] `weak` / `medium` / `strong` aliases resolve to eligible profiles. Tools,
  vision, structured output, and context length are hard requirements.
- [ ] Prefer an already-loaded model that fits. A degradation policy decides
  whether a stronger request may fall back to a weaker tier.
- [ ] Routing metadata always names the concrete model that answered.

## Phase 6: Remote management

- [ ] Edit profiles remotely, make connection setup easier (manual peers over
  LAN or Tailscale first), then add direct model-file transfer with resume and
  checksums.

## Open decisions

- **A stream cut off after a 2xx counts as `completed`.** Upstream does this on
  purpose (`services/nvpair-proxy/spec.md` §5.4: "committed 2xx, truncated by
  the node dying"). For success accounting and "regenerate on another device",
  `failed` with a clear reason may suit this fork better. It is a semantic
  change to the Jobs view and the scheduler inputs, so it is left as is for now.
- **Staying in sync with upstream.** Do we merge upstream `develop` regularly, or
  pin a revision and cherry-pick? Fork-only changes are easier to carry if each
  stays small and inside one service.
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
