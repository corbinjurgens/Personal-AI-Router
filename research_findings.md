<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

**Yes. After inspecting the source, I think PAIR is a worthwhile foundation for what you want.** Its backend is open, its main responsibilities are reasonably separated, and several features you need already have useful implementation hooks.

The two largest additions would be **a backend that runs independently of the GUI** and **a routing system that manages model choice, residency, and capacity**. I also found concrete opportunities to reduce background work, including a history-retention issue that can make the desktop’s memory usage grow over a long session.

## What I inspected

I cloned the repository and compared these versions:

| Version | Revision | Why it matters |
|---|---|---|
| Latest published release, **v0.1.1** | `13b68115` | What you would currently download from Releases. |
| Development branch, inspected **October 9, 2026** | `54d2fe33` | Contains significant newer work, including llama.cpp integration, a unified proxy, and remote engine launch settings. |

The latest release is still labelled v0.1.1. Some capabilities in the current source are therefore ahead of the downloadable release. [GitHub](https://github.com/NVIDIA/Personal-AI-Router/releases/tag/v0.1.1?utm_source=chatgpt.com)

This was a source and control-flow review. I also checked the inspected commit’s upstream CI: its desktop checks, service build/tests, and platform builds reported success. I did **not** run PAIR on your hardware or measure its RAM consumption or inference speed. I did not make source changes. [Inspected commit](https://github.com/NVIDIA/Personal-AI-Router/commit/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b) · [Upstream CI](https://github.com/NVIDIA/Personal-AI-Router/actions/runs/37831761196)

## 1. How open is it?

### The application and backend are open source

PAIR’s own code is under **Apache License 2.0**. The repository includes the actual implementations of:

- The Electron/React desktop application.
- The Go broker that supervises the services.
- Discovery and node information.
- Cluster identity, pairing, and peer transport.
- Request proxying and scheduling.
- Engine installation, lifecycle, and model operations.
- Settings persistence.
- The terminal interface.
- Public build scripts and tests.

You can modify it, keep a private fork, and redistribute modified versions under Apache’s conditions. You do not need NVIDIA to accept a pull request before using your changes. Upstream contributions require a **Developer Certificate of Origin sign-off; the contribution policy explicitly says there is no CLA**. [Licence](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/LICENSE) · [Contribution policy](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/CONTRIBUTING.md#developer-certificate-of-origin-dco-and-sign-off)

There are a few practical boundaries:

- **Engines and models have their own licences.** PAIR’s licence does not relicense LM Studio, model weights, or other software it downloads.
- **NVIDIA’s signing and update distribution are separate.** The public build configuration disables publishing; NVIDIA layers its release distribution configuration on separately. A fork needs its own packaging/update arrangements if you want automatic distribution.
- **The build fetches some assets externally.** For example, its CSS-vendoring script downloads versioned NVIDIA UI assets. That is something I would cache or vendor properly for reproducible offline builds.

These are manageable dependencies. The routing or scheduling implementation is not hidden in a proprietary component. [Third-party notices](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/THIRD_PARTY_NOTICES.md) · [Public packaging configuration](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/desktop/electron-builder.config.ts#L234-L245) · [CSS build dependency](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/desktop/scripts/vendor-kaizen-ui-foundations-css.ts)

## 2. Does it have an underlying server, and can you close the GUI?

**It has a separate Go backend. However, its lifetime is currently controlled by the application that launches it.**

There are three different behaviours to distinguish:

| Action | What happens now |
|---|---|
| **Close the main Overview window** | That window is destroyed. Electron and the backend keep running. |
| **Choose Exit / quit the entire Electron application** | PAIR shuts down its broker and managed inference stack. |
| **Run the terminal interface instead** | You avoid Electron, but the terminal interface starts its own broker and shuts it down when you quit. |

This behaviour is present in both the release and the inspected development version. [Desktop lifecycle](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/desktop/src/electron/main.ts) · [TUI supervision](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-tui/supervisor.go#L80-L138)

### Closing the main window leaves a hidden web window

This is the first obvious improvement for your memory requirement.

At startup, PAIR creates:

1. The main Overview window.
2. A **separate hidden tray-popup BrowserWindow**.

The tray popup loads the same renderer entry point and initializes a broad set of application stores and subscriptions. It usually hides when dismissed instead of being destroyed. Both windows use `backgroundThrottling: false`.

Consequently, closing Overview leaves Electron and the tray’s renderer/state alive. The exact cost needs measurement, but the extra renderer and subscriptions are visible directly in the code. [Tray initialization](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/desktop/src/electron/tray.ts#L58-L83) · [Window creation and preferences](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/desktop/src/electron/window.ts#L253-L360) · [Renderer initialization](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/desktop/src/ui/stores/init.ts)

**A relatively small first patch would be:**

- Create the tray popup only when clicked.
- Destroy it after dismissal or an idle period.
- Initialize only the state that the tray actually displays.
- Offer a native context-menu-only tray mode.

That would reduce idle GUI overhead while preserving the existing process architecture.

### A fully detachable GUI requires a backend change

The broker currently communicates with one parent through JSON-RPC over standard input/output. Its `--ipc` option does **not** turn it into a server that GUIs can attach to: the broker connects to a socket owned by its parent.

When that parent connection closes, the broker exits its read loop and shuts down the inference stack. Therefore, simply setting a process to “detached” or preventing Electron from explicitly killing it would not be enough. [Broker transport](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-ui-broker/main.go#L219-L262) · [Broker shutdown path](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-ui-broker/broker.go#L2303-L2327)

There is also no existing browser dashboard transport waiting to be enabled. The renderer-only build produces static assets, and the API bootstrap explicitly requires Electron’s preload bridge. [Renderer API bootstrap](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/desktop/src/ui/api/bootstrap.ts#L34-L50)

### How I would make the GUI optional

I would keep the Go services and introduce an independently running, per-user service with a local socket or Windows named pipe. Electron and the TUI would attach to it when opened.

There are two viable implementations:

| Approach | Trade-off |
|---|---|
| **Add a small Go supervisor that permanently owns the existing broker connection** | Less initial disruption. The supervisor exposes an attachable interface to GUI/TUI clients. Adds one small process. |
| **Refactor the broker to own its runtime independently of client sessions** | Cleaner long-term architecture. More changes because the current connection and subscription state are intertwined with the broker. |

Either can let **all Electron processes exit while inference continues**.

Most durable settings already live in Go, which helps. One important exception is the saved manual-node list: Electron currently persists it and replays it into the backend. That responsibility would need to move into the service so remote nodes remain configured when no GUI opens. [Manual-node persistence](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/desktop/src/electron/service-bridge/manual-nodes-store.ts) · [Replay logic](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/desktop/src/electron/service-bridge/modular-supervisor.ts#L829-L849)

**For trying it now without Electron, the existing option is `nvpair-tui` from the standalone services bundle.** It must remain running, and you should not run it alongside the desktop application because both try to own the same local services and ports. [Services-only usage](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/docs/building.mdx#run-without-the-desktop-application)

## 3. Can we add your missing features?

Yes. The source provides a useful base, but the work varies considerably in size.

The following reflects the **inspected development branch**; I flag the particularly relevant release differences.

| Your requirement | Existing foundation | Remaining work |
|---|---|---|
| Low-overhead background operation | Separate Go services; terminal client | Remove unnecessary hidden GUI work; add independent service lifetime and GUI attachment. |
| `weak` / `medium` / `strong` names | Proxy already selects destinations and reads the requested model | Add aliases, concrete-model selection, compatibility filtering, and fallback policy. |
| Prefer an already-loaded suitable model | Installed and loaded model information exists | Make routing depend on the particular request and model profile. |
| One resident model per machine | Load/unload operations exist | Add a node-wide reservation that coordinates all engines and model switches. |
| RAM/VRAM budgets | Resource metadata exists | Add actual admission decisions, reservations, and conservative memory estimates. |
| Start models on demand; unload when idle | Engine-dependent behaviour and explicit load/unload operations; development’s llama.cpp launch includes a 300-second idle sleep setting | Add consistent PAIR policy, including waking eligible stopped engines while respecting explicit Off. |
| Gaming/off switch | Saved engine On/Off controls | Add “stop accepting work”, drain/cancel, unload, and persistent paused state. |
| Remote run settings and context | Development adds revisioned remote engine settings, arguments, and environment variables | Add model-specific profiles and coordinate changes with active requests. |
| Remote model download/delete | Already implemented, including in v0.1.1 | Mostly reuse existing operations. |
| Copy an existing model to another device | Remote download operations exist | Direct file transfer, resumption, checksums, and destination storage integration are additional work. |
| Cancellation and device failover | Cancellation propagation and pre-response retries exist; development improves these | Remote job-cancel controls and explicit handling of interrupted generations. |
| Secure connections despite changing addresses | Cluster identity, pairing, manual nodes, and authenticated peer transport | Improve connection setup and address discovery, especially outside the LAN. |

The useful existing hooks are the [model operations](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-engine-manager/modelops.go), [remote controls](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-engine-manager/remote.go), [engine settings contract](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/shared/enginesettings/settings.go), and [llama.cpp manifest](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-engine-manager/manifests/llamacpp.json).

### The scheduler needs a substantial extension

The current scheduler ranks nodes using **pending work plus a coarse GPU-utilization pressure value**. It computes one node-wide ordering and emits it for each engine.

It does not currently make a decision such as:

> “This request needs tools and 16K context. The Mac already has a suitable model loaded, while the desktop would need to unload another model first.”

The proxy separately filters destinations by the requested concrete model. Its reservations improve the originating router’s ranking, but they do not reserve capacity at the destination. Two different computers can therefore make competing placement decisions based on similar advertised state. [Scheduler implementation](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-job-scheduler/schedule.go#L65-L127) · [Proxy candidate selection and reservations](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-proxy/proxy.go#L1944-L2155)

**My proposed design would give each machine final authority over accepting work.**

The originating router would select a concrete candidate, then the destination would atomically accept, queue, or reject it based on:

- Whether inference is enabled.
- Its current model and runtime profile.
- Existing requests and reserved capacity.
- Memory headroom.
- Whether accepting the request requires a model switch.

This is essential for your changing fleet. Advertised free memory is useful for choosing where to ask; the destination still needs to confirm that capacity is available.

The same admission path must cover local requests and requests from peers.

### Model tiers should resolve to explicit profiles

I would define a profile as something like:

> Concrete model + engine + quantization + context limit + launch settings + supported capabilities.

Then `weak`, `medium`, and `strong` resolve to eligible profiles according to your configured preferences.

A stronger model can satisfy a weaker tier if permitted. A weaker model can satisfy a stronger request only under your chosen degradation policy. Tools, image support, structured output, and sufficient context should remain eligibility requirements.

The current proxy only extracts `model` and forwards the remaining request unchanged. It does not currently check those capabilities. Cross-engine aliases therefore need more than changing the names returned by `/v1/models`. [Request parsing](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-proxy/proxy.go#L216-L237)

I would keep the actual selected model visible in routing metadata, so you can see what answered while you are still evaluating models.

### One loaded model and one active request should be separate settings

Your “one model per consumer device” default is sensible, but it should not automatically mean one request at a time.

The system should independently track:

- Resident model count.
- Active request/context capacity.
- Memory reserved for those requests.
- Whether a loaded model can accept another request.

On the Mac, the memory budget must treat unified memory as one pool. On the RTX machines, it needs separate RAM and VRAM accounting. A displayed GPU-utilization percentage should not be presented as an enforced usage limit unless the underlying engine can actually enforce it.

### The off switch must take precedence over automatic loading

I would use explicit states such as **Available**, **Draining**, and **Paused**.

“Pause for gaming” would first stop admission, then either finish or cancel existing work, then unload models or stop the managed engine. New requests would not wake the machine’s inference engine until you enable it again.

PAIR’s saved engine intent is a useful starting point. Its current remote launch-setting changes can restart engines, so those changes also need coordination with draining and active requests. [Saved engine intent](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-engine-manager/desiredstate.go) · [Applying launch settings](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-engine-manager/settings.go#L387-L443)

### Networking is reusable, but public-address operation needs more work

PAIR already has cryptographic identity and authenticated peer channels. I would build convenient invitations and address updates around that existing trust system.

Its current security architecture is LAN-first: some discovery/metadata traffic can be plaintext, while inference between paired nodes uses authenticated transport. The local plaintext inference endpoint accepts loopback traffic. That means “enter any remote URL and everything works securely through NAT” is not an existing finished feature. [Security architecture](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/SECURITY.md)

For your first version, I would prioritize reliable manual peers over LAN/Tailscale, with persistent identity and configuration owned by the daemon. General Internet discovery and invitation handling can follow.

## 4. How easy is it to modify?

**Small, focused changes look quite approachable. The complete feature set is a substantial project.**

The architecture helps: runtime behaviour lives in Go, while the desktop generally sends commands and renders reported state. Engine behaviour is partly described through manifests, and the registry supports per-user manifest overrides.

That gives us room to change launch settings or adapt an existing engine without redesigning the whole application. However, adding a completely new engine throughout the product also involves proxy profiles, discovery, contracts, and UI support. The manifest mechanism is useful, but it is not a complete universal plugin API. [Developer guide](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/docs/developing.mdx) · [Manifest overrides](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-engine-manager/registry.go#L430-L468)

My rough assessment:

| Change | Relative scope |
|---|---|
| Lazy/native tray, reduced hidden-window work | Small |
| Fix history retirement and routine protocol logging | Small to medium |
| Deduplicate model polling | Small |
| Independently running service with reconnecting GUI | Medium, with lifecycle testing across platforms |
| Aliases within one well-defined API/backend family | Medium |
| Model-specific memory admission and switching across machines | Substantial |
| Consistent cross-engine capability handling | Substantial |
| Resumable model-file transfer | Medium to substantial |
| Seamless mid-generation migration between different models/engines | A separate, much harder feature |

These are engineering assessments from the source, not fixed time estimates.

## 5. Is the code optimized? Concrete findings

There are thoughtful optimizations already present, alongside clear opportunities to improve it. I would address the following before attempting a broad rewrite.

### A. Completed-job metadata can accumulate indefinitely in the desktop

This is the strongest memory-related finding.

The broker limits terminal job history to **10,000 records and seven days**. Its pruning code directly deletes records from its own map.

Electron maintains a separate workload map, and the renderer maintains another. Those maps remove records when they receive explicit removal events, but automatic backend pruning does not emit those events. Fetching another baseline does not repair the Electron map: its seeding method only adds missing entries.

The visible history limit of **30** is applied after collecting and sorting records. It limits displayed rows, not retained state.

**Result:** during an uninterrupted session, completed-job metadata can continue accumulating in Electron and initialized renderers beyond the backend’s retention limits. Map copying and sorting also grow with that history. These are metadata records, not stored copies of full prompts and responses.

The missing retirement path exists in v0.1.1 as well. [Backend pruning](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-ui-broker/workloadstore/persistence.go#L174-L205) · [Electron workload map](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/desktop/src/electron/service-bridge/modular-state.ts#L1268-L1310) · [Renderer history selectors](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/desktop/src/ui/stores/workloads.store.ts#L197-L238)

**Fix:** propagate retirement events consistently, or maintain a deliberately bounded desktop history projection. Active jobs must be preserved.

### B. Routine backend messages are synchronously logged

Every broker stdout JSON-RPC line enters the desktop logging path before parsing. Ordinary protocol traffic is labelled verbose, but the logger does not actually filter it by severity. It serializes the entry and uses `appendFileSync` on Electron’s main thread.

This is a concrete synchronous I/O path. Whether it causes noticeable stalls depends on message volume and storage; I have not measured that.

**Fix:** make complete protocol logging an explicit diagnostic option, filter before generating/storing entries, and use a bounded asynchronous writer where necessary. Preserve the existing sensitive-field redaction.

The logs already have retention bounds; this is a different issue from the workload-history growth. [Protocol logging](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/desktop/src/electron/service-bridge/json-rpc-subprocess.ts#L298-L310) · [Synchronous logger](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/desktop/src/shared/utils/log.ts#L62-L95)

### C. Hidden GUI telemetry causes unnecessary work

Electron polls richer node information every two seconds independently of the backend’s routing telemetry. Its lifetime is tied to the running service connection, rather than visible windows.

Telemetry changes can trigger broader discovery, model, and status notifications. Those messages are broadcast to non-destroyed windows, including hidden ones.

**Fix:** only run the rich GUI feed while a client needs it, and keep telemetry-only changes separate from model/discovery changes. The backend’s liveness and scheduling feed should continue when the GUI is absent.

Some stores already suppress equal updates, so I would not claim every notification produces a React render. [GUI poller](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/desktop/src/electron/service-bridge/node-info-poller.ts) · [Broad node notifications](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/desktop/src/electron/service-bridge/modular-state.ts#L2521-L2545)

### D. Model polling includes a duplicate request

The loaded-model watcher runs every five seconds and fetches both installed and loaded model information.

For LM Studio, the `list_models` and `loaded_models` actions call the **same `/api/v1/models` endpoint twice**, then extract different views.

**Fix:** derive both views from one response. Later, separate frequent residency checks from less frequent inventory reconciliation. Keep occasional reconciliation because users can change models outside PAIR. [Watcher](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-engine-manager/loadedwatch.go#L87-L97) · [Duplicate endpoint definitions](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-engine-manager/manifests/lmstudio.json#L69-L77)

### E. Request buffering needs a size policy

The proxy buffers incoming request bodies with `io.ReadAll` so it can retry them. I did not find a request-body limit around that path.

That is usually modest for short text prompts, but long contexts and embedded image data can be much larger.

**Fix:** impose a configurable limit and return a clear error for oversized requests. If very large replayable requests are necessary, consider bounded buffering with temporary-file backing. The response already streams; it is not buffering the whole generated answer. [Request buffering](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-proxy/proxy.go#L216-L237)

### What I would keep

The source already includes useful measures: overlapping-poll suppression, failure backoff, bounded metrics histories, coalesced updates, streaming responses, and native Windows GPU telemetry.

I would also keep the Go service boundaries initially. Multiple processes impose overhead, but counting processes does not tell us whether merging them would deliver worthwhile savings after accounting for fault isolation and complexity.

**The first measurements should separate PAIR overhead from engine/model memory:** engines stopped with Overview open, Overview closed, and then the independent service with no Electron. A sustained synthetic job stream would separately measure history growth.

## 6. Cancellation and switching devices: one important limit

PAIR can already try another eligible destination before committing a response, and development improves this with bounded retries and refreshed candidates.

Once output is streaming, it does not move the generation state or seamlessly switch to another model. Cancellation propagates through the HTTP request, but the engine must respond to that cancellation.

There is also a correctness issue in development: an upstream stream that aborts after a successful response was committed can be recorded as **completed** in workload state, even though an abort is recorded in the request event. I would fix that before using Jobs as reliable success accounting. [Retry implementation](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-proxy/proxy.go#L1535-L1765) · [Interrupted-stream classification](https://github.com/NVIDIA/Personal-AI-Router/blob/54d2fe33067ce66cc59e1070fb0bc3a3ed87160b/services/nvpair-proxy/proxy.go#L1402-L1476)

For your first version, I would provide **cancel and regenerate on another available device**. Continuing from already displayed text could be a later, explicit feature. It should not silently splice together two different model responses.

## My recommendation

**Give PAIR a trial, and use its newer source as the basis for modifications once the basic workflow is proven.** The development branch is the more useful extension target because of its unified proxy, llama.cpp support, and remote settings. Pin a revision and verify it across your machines before building further on it.

I would implement your version in this order:

1. **Reduce idle desktop overhead:** lazy/native tray, bounded workload history, diagnostic-only protocol logging.
2. **Make the GUI fully optional:** independent Go service, local attachment, reconnecting GUI/TUI, backend-owned manual-node persistence.
3. **Add trustworthy node availability:** pause, drain/cancel, unload, and enforcement of explicit Off.
4. **Add resource admission and model profiles:** one resident model by default, controlled concurrency, memory reservations, safe switching.
5. **Add weak/medium/strong routing:** compatible profiles, warm-model preference, configurable degradation, visible routing decisions.
6. **Extend remote management:** profile editing, better connection setup, then direct model-file transfer.

That preserves the difficult infrastructure PAIR already provides and concentrates our work on the behaviour that would make it useful for your changing collection of machines. **I would build on PAIR’s backend; I would plan the detachable GUI and capacity-aware routing as explicit architectural work, rather than expect configuration alone to deliver them.**
