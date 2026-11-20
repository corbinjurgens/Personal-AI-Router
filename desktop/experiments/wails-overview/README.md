<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared Overview renderer

Production source baseline: `28b822ddc898d337083c5b22dd4821344eb4b743`.
The comparison uses the PAIR Overview lists, cards, and SVG connector under
the production Kaizen theme and Tailwind styles. The native host owns window
and service lifecycle. The renderer consumes the desktop service projection.

## Production data dependencies

`desktop/src/ui/components/MainApp/Content.tsx` places `WorkloadListView`,
`WorkloadNodeConnections`, and `NodeList` together. `desktop/src/ui/main.tsx`
installs `FoundationsThemeProvider`, `setupApis`, and `connectAndInitialize`.
The latter provisions Overview and full-app stores in `src/ui/stores/init.ts`.
`src/ui/api/{pair-api,engine-api}.ts` binds the typed renderer channels in
`src/shared/types/ws-channels.ts`. The main-process bridge registers those
channels in `src/electron/service-bridge/empty-handlers.ts` and projects
broker notifications in `modular-supervisor.ts` and `modular-state.ts`.

| Surface and displayed fields | Renderer read and update path | Electron projection and Go owner | Cadence |
| --- | --- | --- | --- |
| Self, local badge, confirmed members, offline name/IP, live card replacement | `useOverviewNodes` calls `buildOverviewNodes` with connection self ID, node map, membership, and `window.windowApi.platform`. `app:get-initial`, `cluster:get-initial`, `nodes:changed`, `cluster:node-name-changed`, `connection:cluster-identity`. | `handleGetSelfId`, `handleClusterGetInitial`, `cluster-json.ts`; broker relays `cluster:get-node-id` and `nodes:get-initial` to cluster manager `manager.go`. Its `membership.go` emits the roster. Go member UUID equals desktop `ClusterNode.nodeUuid` and `NodeItem.id`. | On baseline, membership or identity change. |
| Live node name/IP, hardware readiness, GPU/CPU and memory charts, build label, performance expansion | Nodes snapshot `nodes:get-initial`, then `nodes:upsert` / `nodes:remove`; sample stream `metrics:update` into `metrics.store.ts`; `NodeCardDetails` and `NodePerformance` use topology, performance and metric histories. | `node-projection.ts` merges broker `discovery:get-nodes` / `discovery:subscribe` directory records with telemetry. Broker `client_requests.go` serves its directory snapshot; `node_info_rpc.go` fronts the scanner and paired target reads. `nvpair-node-info/main.go` owns `/v1/node-info` and `NodeInfoResponse`; unpaired and local Electron poll targets use that HTTP surface directly. | Directory updates on discovery changes; Electron node-info polling at 2 seconds, telemetry updates on accepted samples. |
| Engine pills and model identity on workload cards | `engines:get-initial`, `engines:state-changed`, descriptor and progress pushes into `engine-status.store.ts` and `engine-models.store.ts`; pending lifecycle commands also affect stock pills. | Supervisor seeds engine projection from descriptor/status/model reads and engine notifications; `nvpair-ui-broker/client_requests.go` relays engine commands and subscriptions to `nvpair-engine-manager/manager.go` (`engine:descriptors`, `engine:status`, `engine:models`). Peer engine state also comes from discovery and node-info. | Baseline, lifecycle/inventory changes and download progress. |
| Workload card identity, model/tier, origin, execution target, lifecycle, timestamps and failure | `workloads:get-initial`, `workloads:upsert` and `workloads:remove` in `workloads.store.ts`; keys use origin plus workload ID. `WorkloadListView` applies history filters and order. `WorkloadItemCard` resolves labels using nodes, membership and engine models. | `workload-projection.ts` parses baseline `workloads[]`, pushed `workloadInfo` and removal identity. `nvpair-ui-broker/client_requests.go` serves the durable baseline and handles `workloads:subscribe`; `workload_projection.go` projects/fans out transitions and `nvpair-workload-manager/workload.go` defines the workload wire record. | Each lifecycle transition; renderer commits push batches on an animation frame. |
| Workload routing detail | `routing.store.ts` and `routing-history.ts` use the private `routing:*` / `routing-history:*` protocol, with synchronous decoded row loans to `WorkloadItemCard`. | Electron bridge `routing.ts` and `routing-delivery.ts` carry baselines and deltas from service and broker, preserving lease and generation. `nvpair-service/history.go`, broker routing handlers, and `shared/nodepolicy/wire.go` own the history and wire shapes. | Baseline plus decision/attempt updates; renderer visibility and retirement govern delivery. |
| Connector path, direction, motion and glow | `WorkloadNodeConnections` selects assigned jobs; the DOM provides workload identity/origin, node identity and chart anchors. It watches scroll, resize and content changes. | Same workload and node projection as the lists. There is no connector-specific RPC. | Geometry remeasures while layout moves. SVG motion continues between measurements, with glow removed above 12 rendered running connections. |
| Node availability, permission and profile summaries | `NodeAvailabilityControls` uses `node-configuration.store.ts`; `NodeDeclaredRoles` reads discovery's advertised policy, engine models/status, error state and optional per-engine saved settings. | `policy:get`, `policy:changed`, `node:availability-changed`, `engine:remote-management-get`, `discovery:get-nodes`, `engines:get-settings`, `errors:get-initial`, `errors:update`. Broker `policy.go` and policy wire types own availability and policy; cluster-manager owns grants, engine-manager `remote_management.go` owns target permission; `nvpair-errors/manager.go` owns error snapshots. | Visibility-scoped reads and pushes; summaries load saved engine settings on request. |

Native operations in stock `MainApp` include the title bar, installed service
lifecycle, onboarding, inference demo, updates and window actions. Its
content bar adds settings, models, profiles, discovery, endpoints and pairing.
The experiment mounts the Overview columns and leaves service attachment and
hardware polling to the live host. Filter and disclosure actions stay local to
the renderer. The full app, native bootstrap, pairing flows and other tabs
remain stock surfaces.

| Stock card action | Owner and wire path |
| --- | --- |
| Node settings and profile navigation | `focusNodeSettings` in `overview-ui.store.ts` selects the routing settings tab. |
| Rename or nickname | `NodeNameModal` calls the cluster store, which forwards `cluster:set-node-name`, `cluster:set-peer-name`, or `cluster:set-peer-nickname` through the bridge to the cluster manager's `manager.go`, with `NodeNameResult` and `PeerNicknameResult` projections from `cluster-json.ts`. |
| Install engine | `NodeEnginesInline` calls the renderer's engine API. The bridge translates `engine:command` into `engine:install`, relayed by the broker to the engine manager's `manager.go`. |
| Pause, halt or resume | `NodeAvailabilityControls` calls the node-configuration store with `NodeAvailabilityRequest`. The bridge forwards `node:set-availability` to broker `policy.go`, which applies local availability or dispatches to the paired target. |
| Cancel or reroute a workload | `WorkloadCancelActions` derives the origin, ID, engine and run identity via `workloadCancelRequest`. The bridge passes `workloads:cancel` to broker `workloadcancel.go`; the origin's proxy receives `workload/cancel` using the shared `nodepolicy/wire.go` types. |

## Build and replay

From `desktop/` with Node 26 on PATH:

```sh
npm run build:experiment:overview
npm run build:experiment:overview:electron
npx vitest run --project unit tests/experiments/wails-overview
```

Serve or load only `dist/renderer/` from this experiment directory. The build
writes a separate `dist/asset-manifest.json` with SHA-256 for each file under
`dist/renderer/`, keyed by slash-separated relative path. Package those same
renderer bytes in each host. Neither output is an input to the stock Electron
builder. Vite uses the stock `@/` alias, React and Tailwind plugins, with the
production stylesheet including Kaizen's vendored styles and local font rules.

The entry point takes `mode=replay` (default) and a `scene` query, with optional
`order=nodes-first`. The default order is workloads first. The scene always
starts with a synthetic self placeholder and a confirmed offline peer. At one
second, self and peer receive their first discovery and telemetry events, one
completed job appears, and the requested running jobs start. A discoverable
pending peer remains outside the Overview. Telemetry values are fixed for
each scene, with observations timestamped on receipt every two seconds.

| Scene | Workload list after the first update |
| --- | --- |
| `idle` (default) | Completed job history and no running jobs |
| `shared-node` | Two running jobs with different origins and the same proxy-local job ID, executing on the model host |
| `running-12` | Twelve running connections with glow |
| `running-13` | Thirteen running connections with the production single-bead animation |

Replay data uses fictional nodes and documentation-only addresses. The
renderer invokes the same typed store baselines and push subscriptions as
live. It uses the production roster projection without joining, invite
polling or permission mutations. The routing-attempt history stream and
full-app error modal are outside the renderer's current input surface.

The Electron command builds the shared bundle and packages an unpacked app for
the current platform and architecture with the installed Electron runtime.
It stages the app under `dist/electron-app/` and writes the package under
`dist/electron/`. The build verifies every packaged renderer file against
the manifest. The macOS package uses bundle ID
`com.nvidia.pair.overview-experiment`. The app uses its own
`PAIR Overview Experiment` directory under Electron's app-data root.

On Apple Silicon, run the packaged replay from `desktop/`:

```sh
app="experiments/wails-overview/dist/electron/mac-arm64/PAIR Overview Experiment.app"
env -u ELECTRON_RUN_AS_NODE open -n "$app" --args --mode=replay --scene=running-13
```

Use `--scene=idle`, `--scene=shared-node`, or `--scene=running-12` for
the other fixtures. Add `--order=nodes-first` to change the column order.
Use `--mode=live` to attach to the installed per-user PAIR service. Select
**Overview** in the experiment tray menu to open a new window after closing
it, or **Quit** to exit the experiment.

## Window and tray operations

Stock behavior is from `src/electron/{tray,window,window-visibility,main}.ts`,
`src/electron/connector/index.ts`, and `src/electron/ipc/window.ipc.ts`.
The Wails column gives the comparison contract for that host.

| Operation | Stock Electron | Matched Electron | Wails target |
| --- | --- | --- | --- |
| Normal Quit | Tray Quit calls `app.quit()`. Cleanup destroys the tray and detaches the UI connector. The service keeps running. The stock tray has a separate **Stop background service and quit** action. | Native tray Quit exits the app and detaches any Overview socket. The service keeps running. | Native tray Quit exits the app and detaches Overview. The service keeps running. |
| Close Overview | The title-bar action closes and destroys the Overview window. The tray and service connector remain. | The native close button destroys the window, its IPC handler, and its service attachment. The tray remains available. | Close destroys the Overview webview and dashboard attachment. The tray remains available. |
| Show or hide popup | In popup mode a tray click opens or toggles the status popup. Blur hides an open popup after the initial show grace. The native menu can switch to menu-only mode. | The tray icon opens a native menu with Overview and Quit. | The tray icon exposes a native menu with Overview and Quit. |
| Hidden renderer | A minimized or hidden Overview keeps its renderer. A closed Overview is destroyed. A hidden status popup keeps its renderer for up to 60 seconds, then is destroyed. Menu-only mode creates no popup renderer. | A minimized Overview keeps its renderer. Closing it destroys the renderer immediately. The tray menu creates no renderer. | Closing Overview destroys its renderer. The tray menu creates no renderer. |
| View focus | Overview creation raises the view. Reopening an existing Overview shows and focuses it. The status popup takes focus on macOS and Windows. | Selecting Overview opens and raises a new window, or restores and focuses the current one. | Selecting Overview opens or focuses the view. |
| Subscriptions | The app-wide bridge keeps its broker attachment and subscriptions through window close. Visible Overview or popup windows enable node-info polling. The private routing history has a separate renderer visibility and retirement lifecycle. | A live Overview owns its socket and subscriptions. Minimizing pauses node-info polling. Close detaches the client. Reopening subscribes and fetches fresh baselines. Service-wide streams can remain active for other clients. Replay uses fixture events. | A live Overview owns its dashboard attachment. Close detaches it. Reopen subscribes and resnapshots. Replay uses fixture events. |

## Live host input

Set `mode=live` and expose `window.__PAIR_OVERVIEW_HOST__` before the entry
point runs. Its `platform` is the native display name. Its
`onGeneration(listener)` returns an unsubscribe function and delivers a new
`PreloadServiceTransport` for each ready window/IPC/broker generation. Send
`null` on disconnect or broker retirement. The renderer subscribes to pushes
before starting the baseline reads and calls `destroy()` on retired
transports. Deliver only the current generation's projected events and
initial results. A failed or missing baseline must reject, including when
the last generation succeeded. Deliver a fresh generation for a full state
refresh.

The required desktop baselines are `app:get-initial`, `cluster:get-initial`,
`nodes:get-initial`, `engines:get-initial`, `workloads:get-initial`,
`discovery:get-nodes`, and `errors:get-initial`. Live card reads also use
`policy:get`, `cluster:get-remote-management`,
`engine:remote-management-get`, and `engines:get-settings`. The renderer
invoke allowlist rejects mutations before forwarding to either host. Stock
card action affordances remain visible and their mutations are refused. The
matched Electron main process enforces the same allowlist independently of
the renderer and rejects requests carrying a retired generation.

The Electron attachment uses the installed service socket or named pipe
resolved by the desktop's servicectl-compatible endpoint resolver. The
window starts fresh discovery and proxy reads, engine and model reads,
cluster, workload and error baselines, and the associated dashboard streams.
It uses the production node-info poller for visible telemetry. Missing or
failed baseline reads leave the view unavailable. Stock UI startup and
mutation paths are outside the experiment, including cluster identity setup,
manual-node migration, service startup, startup registration, and updates.
The private routing-attempt history stream is omitted in both experiments.

### Known limit of the matched Electron live path

Live windows share the production projection singleton returned by
`getModularBridgeState()`. Retiring a generation clears projected discovery and
proxy node sources, workload and error snapshots, and performance, and
invalidates model observations. It does not clear every remembered engine fact,
proxy port or download state that the production owner keeps as last-good
values. A later successful response that omits an optional fact can therefore
show a value from the previous generation. Failed baseline reads still leave
the view unavailable.

## Wails live input: not implemented

The Wails host serves replay only; `-mode live` returns an error. The owner
decided on 2026-11-20 to compare the hosts on replay and to record live Wails
as inconclusive rather than change production code.

The missing contract: the desktop's payload projection lives in
`desktop/src/electron/service-bridge/` and cannot run in a webview as written.

- `broadcaster.ts` imports Electron `BrowserWindow` and `push-bus.ts` (Node
  `events`). `modular-state.ts` imports the broadcaster and reaches another
  `BrowserWindow` import through `routing.ts` and `routing-delivery.ts`.
- `node-info-poller.ts` imports `desktop/src/shared/utils/log.ts`, which uses
  `fs`, `path`, the Electron path provider and the bounded log writer, and
  hooks `process.stdout` and `process.stderr` on load.
- `node-projection.ts` reads `process.platform` through `currentPlatform()`.
- The poller reads local and unpaired node-info targets over direct HTTP. The
  experiment webview's CSP and file filter do not allow that. The
  `discovery:get-node-info` RPC covers registered self and admitted peers only
  (`services/nvpair-node-scanner/node_info_rpc.go`, `getPeerNodeInfo`).

Live Wails would need one of these:

1. A browser-safe projection entry point in those modules. It would take a
   typed push sink, a routing-history callback and the native platform as
   inputs, and the Electron wiring would keep supplying the native versions.
   Wails would also need a defined path for unpaired-node telemetry. Both
   hosts would then relay allowlisted raw JSON-RPC, with projection running
   in the shared renderer, fresh for each generation.
2. A Go port of the projection, which would duplicate desktop policy.

Until then, a comparison on replay input measures the shell (Electron versus
Wails displaying identical bytes and events). Compare live end-to-end results
only between stock and matched Electron.
