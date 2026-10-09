<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Testing the Fork on Your PCs

A checklist for trying `fork/groundwork` on real machines. Everything here
passed automated tests in a Linux container. None of it has run on Windows or
macOS, on a GPU, or across two real machines yet.

## 1. Build

Prerequisites (see [docs/building.mdx](docs/building.mdx)): Go 1.25+, Node 22,
`jq`.

```bash
git checkout fork/groundwork
cd services && ./build.sh          # Windows: build.bat
# binaries land in services/build/bin, including nvpair-service
cd ../desktop && npm install && npm start   # desktop app (also builds the Go binaries)
```

## 2. The background service (no GUI needed)

All commands below are run from `services/build/bin`.

```bash
./nvpair-service                 # run in the foreground (Ctrl+C stops it)
./nvpair-service status          # from another terminal
./nvpair-service call policy:get # call any broker method
./nvpair-service stop
```

The TUI is now a client. `./nvpair-tui` starts the service if needed:

- `q` quits the TUI and leaves inference running.
- `Q` then `y` stops the service.
- `P` pauses or resumes inference on this PC. The header shows Available,
  Pausing… or Paused.

Check these:

- [ ] Closing the TUI leaves `nvpair-service status` reporting running, and
  your engines keep serving.
- [ ] Reopening the TUI shows the same nodes and jobs.
- [ ] `nvpair-service autostart enable` starts the service at the next login.
  `autostart disable` removes it.
- [ ] **Windows:** no console window appears when the TUI starts the service,
  and the service survives closing the terminal.
- [ ] **macOS:** the service survives closing Terminal.

The desktop app is a client too.

- [ ] It starts the service if it is not running. Quitting the app leaves the
  service and inference running, and reopening it reattaches.
- [ ] The tray menu's "Stop background service and quit" stops the service
  (this can take about 20 s).
- [ ] The tray menu's pause toggle shows "Pausing…" while draining.
- [ ] Killing the service makes the app reconnect, restarting the service if
  needed.
- [ ] An old `configs/manual-nodes.json` from the upstream app is migrated
  once and then deleted.

## 3. Pause for gaming

```bash
./nvpair-service call node:set-availability '{"state":"paused"}'
./nvpair-service call node:set-availability '{"state":"available"}'
```

- [ ] Start a long generation, then pause. The default `pause.onActive` is
  `finish`, so the generation completes before the call returns. Then the
  models unload.
- [ ] While paused, a request to this PC's engine port gets
  `503` with `X-PAIR-Admission: paused`. On a two-PC cluster, requests that
  enter this PC are served by the other PC.
- [ ] Set `"stopEngines": true` under `pause` (see section 4). Pausing then
  also stops the engines. Restarting the service keeps them stopped. Resuming
  starts only the engines that were saved On.

## 4. Policy file

The policy file lives at
`<appdir>/node-policy.json`:

- Windows: `%LOCALAPPDATA%\Nvidia Corporation\Personal AI Router\`
- macOS: `~/Library/Application Support/Nvidia Corporation/Personal AI Router/`
- Linux: `~/.config/Nvidia Corporation/Personal AI Router/`

Change it with `policy:set`, which validates the new policy and applies it
live:

```bash
./nvpair-service call policy:set "$(cat my-policy.json | jq -c '{policy: .}')"
```

Example with tiers and a profile:

```json
{
  "version": 1,
  "admission": { "maxResidentModels": 1, "queueTimeoutSeconds": 30, "switchModels": true },
  "idle": { "unloadAfterMinutes": 15, "stopEngineAfterMinutes": 0, "startOnDemand": true },
  "profiles": [
    { "name": "qwen8", "engine": "ollama", "model": "qwen3:8b", "contextTokens": 16384,
      "capabilities": ["tools"], "requestOptions": { "options": { "num_ctx": 16384 } } }
  ],
  "tiers": {
    "weak":   [{ "engine": "ollama", "model": "qwen3:4b", "capabilities": ["tools"] }],
    "medium": [{ "engine": "ollama", "model": "qwen3:8b", "capabilities": ["tools"], "contextTokens": 16384 }],
    "strong": [{ "engine": "lmstudio", "model": "qwen3-32b", "capabilities": ["tools", "vision"] }]
  },
  "tierPolicy": { "allowStronger": true, "allowWeaker": false, "preferLoaded": "tier" }
}
```

To edit another paired PC's policy, add `"nodeId": "<its hostUuid>"` to
`policy:get` and `policy:set`. The same works for `node:set-availability`.

## 5. Routing checks

- [ ] **Tiers.** POST `{"model":"weak", ...}` to
  `http://localhost:<engine proxy port>/v1/chat/completions`. The response
  headers name what answered: `X-PAIR-Model`, `X-PAIR-Engine`, `X-PAIR-Node`
  and `X-PAIR-Tier`. A request with `tools` skips models that are not tagged
  `tools`.
- [ ] **One resident model.** Alternate requests between two models on one PC.
  The idle model unloads before the other loads, and the broker log shows an
  admission unload.
- [ ] **Busy hand-off.** With two PCs and `maxConcurrentPerModel: 1`, two
  simultaneous requests for one model go to different PCs.
- [ ] **Start on demand.** Set `idle.stopEngineAfterMinutes: 1` and wait. The
  engine stops. The next request starts it again. An engine you turned Off is
  never started this way.
- [ ] **Settings change during a request.** Changing a running engine's launch
  settings mid-request lets the request finish before the restart.
- [ ] **Model list.** `/v1/models` lists `weak`, `medium` and `strong` when
  they are configured.

## 6. Jobs

- [ ] Cancel a running job:
  `./nvpair-service call workloads:cancel '{"originatedFrom":"<host uuid>","workloadId":"<id>","engine":"ollama"}'`
  It ends as `cancelled`.
- [ ] Add `"regenerate": true` while the job is still queued or waiting for its
  first token. It moves to another PC.
- [ ] Cancel a job that started on the other PC, from this PC.

## 7. Copy a model between PCs (paired cluster)

```bash
./nvpair-service call engine:remote-copy-model '{"node":"<source hostUuid>","engine":"ollama","model":"qwen3:8b"}'
```

- [ ] An Ollama model appears in `ollama list` on the destination without a
  restart.
- [ ] Interrupt the copy (stop the service or unplug the network), then run it
  again. It resumes instead of starting over.
- [ ] Try a llama.cpp `owner/repo:Q4_K_M` model, and an LM Studio GGUF.
- [ ] A destination without enough free disk space refuses before
  downloading.

## 8. Manual peers

- [ ] Add a peer by Tailscale name with
  `node/add {"address":"mybox.tailnet.ts.net"}`. It survives a service
  restart (it is saved in `manual-nodes.json`).

## 9. Desktop idle overhead (phase 1)

- [ ] With engines stopped, measure PAIR's memory three ways: Overview open,
  Overview closed (the tray popup is destroyed after 60 s hidden), and the
  service running alone.
- [ ] The tray menu's "Show Status Popup on Click" toggle switches to a
  menu-only tray.
- [ ] A new pairing invite still opens Overview.

## Known gaps

The open items list in [FORK.md](FORK.md) is the authoritative version.

- In the desktop app, policy, tiers and pause are under Settings → Routing &
  resources (live). Cancel is on each job, and "Copy to this PC" is on a
  paired node's model rows. Only one copy per source node and engine runs at a
  time.
- Paused peers are not advertised. Other PCs find out from a fast `503` and
  move on.
- `services/versions.json` has no `nvpair-service` entry, so the service
  reports version `0.0.0`.
