<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution plan: j-routing-diagnostics (J0 + J1 + J2 — private routing projection and diagnostics, P5.1)

Planned against `fork/groundwork` at commit `584e99242562df1cb19d2c7a318b7342e03cb7d2`. Every `path:line` below refers to that commit. Path shorthand: `B/` = `services/nvpair-ui-broker/`, `P/` = `services/nvpair-proxy/`, `N/` = `services/shared/`, `X/` = `services/tests/`, `T/` = `services/nvpair-tui/ui/`, `D/` = `desktop/src/`, `W/` = `services/nvpair-workload-manager/`.

## Goal

When this plan is finished:

1. The proxy keeps, per inference request it routes, a bounded and redacted **routing request record**: the request identity, each dispatch attempt with its destination kind, outcome, closed-set reason code, admission wait and time to first byte, the account state the attempt saw, the request's cumulative per-request spend ceiling and reserved/charged amounts as exact decimal `NanoUSD` strings, and the terminal state (`completed`, `failed`, `cancelled`) with a reason. It publishes the record to its own broker only, over a new private notification `routing/request`.
2. The broker keeps the newest 256 records in memory, stamps the origin node like it does for workloads, and serves them to local clients over `routing:subscribe` / `routing:unsubscribe` / `routing:get-initial` and the push `routing:changed`. The broker **never** forwards a routing record to the workload-manager, the scheduler, or any peer, and it strips any non-allowlisted field from a proxy workload frame before that frame leaves the node.
3. A cross-process test proves the cluster workload wire carries no account, deployment, credential, spend, attempt or reason field.
4. The desktop client (J1) shows the routing history inside the existing Routing settings tab (KUI components only), with attempt reasons, waits, account states and money as strings, plus a one-line routing summary on each workload card; interrupted attempts are shown as a pause, not an error, unless no same-model continuation is possible.
5. The terminal interface (J2) gets a `Routing` tab rendering the same records with the same wording rules.
6. The TypeScript view types in `D/shared/types/routing.ts` and the Go decoder in `T/routingwire.go` are both validated against one JSON corpus, `services/shared/nodepolicy/testdata/routing-request.json`, so J1 and J2 build on a frozen interface.

## Change scope

**Intent:** Add a private, bounded, redacted diagnostics stream for routing decisions, and harden the cluster workload wire against private-field leakage.

**Scope:** Bounded improvement. What may change: a new shared Go wire package file under `N/nodepolicy/`, a new emitter in the proxy, new history/relay code in the broker, new desktop bridge/types/store/components, a new TUI decoder and view, documentation and the generated `desktop/docs/services-api.md`, `desktop/docs/service-contract-exceptions.json`. What may not change: routing policy semantics (`N/nodepolicy/routing.go`), the `Workload` wire struct (`P/proxy.go:172-211`), the retry loop's decisions, the admission ledger's arithmetic, the cluster workload-manager protocol, the OpenAI-compatible HTTP surface.

**Basis:** `.cursor/rules/service-contracts.mdc`, `.cursor/rules/services-change-propagation.mdc`, `.cursor/rules/ipc-patterns.mdc`, `.cursor/rules/renderer-stores.mdc`, `.cursor/rules/security.mdc`, `.cursor/rules/proxy-inference-routing.mdc`, `P/spec.md` §5 and §12, `desktop/scripts/verify-service-contracts.ts`, and the owner decisions restated under Decisions below.

**Compatibility:** Purely additive on every wire. No existing method, payload field or header changes. Clients that never call `routing:subscribe` see no new traffic. Service binaries whose compiled output changes (`nvpair-proxy`, `nvpair-ui-broker`, `nvpair-tui`) need a bump declared in the pull request's `pair-release-intent:v1` block; do not edit `services/versions.json` or `CHANGELOG.md`.

**Non-goals:** persisting history to disk; exposing routing history to paired peers; any new HTTP endpoint; changing how the proxy chooses candidates, retries, or charges accounts; implementing SC-b same-model continuation (this plan only reserves and renders the field); the `routing:active` / `routing:cancel` / `routing/active` methods owned by the I plan (Task 6 there); cancelling requests from the routing history UI (the workload card's existing cancel action stays the only control).

## Context the executor needs

### Verified facts (code at 584e992)

**Shared policy package**

- `N/nodepolicy/routing_wire.go:25-29` already defines `RoutingRequestKey{OriginNodeID, ProxyRunID, RequestID}`; `Validate()` applies the strict ID regex `^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$` and requires `RequestID` to be a positive int64 decimal string. `decodeStrictContract` (`routing_wire.go:152-177`) rejects unknown, duplicate and null fields. `StrictModelIdentity` and `StrictAdmissionRequest` live in the same file.
- `N/nodepolicy/values.go:18` declares `type NanoUSD string`; `ParseNanoUSD` (`values.go:26-34`) accepts canonical decimal digits only; `Int64()` converts; JSON marshals as a string. There is no float path.
- `N/nodepolicy/routing.go:76-86` `Account{ID, Credential, Billing, PerRequestCeiling, DailyCeiling, MonthlyCeiling NanoUSD}`; `Deployment` (`routing.go:90-121`); `Limits{QueueSeconds, ColdStartSeconds, MaxAttempts, PreOutputBudgetSeconds, MaxOutputTokens}` (`routing.go:171-182`); `maxAttempts = 10` (`routing.go:196`); defaults `MaxAttempts 3`, `PreOutputBudgetSeconds 30` (`routing.go:293-299`); `remoteDispatchAvailable = false` (`routing.go:21`).
- `N/nodepolicy/wire.go:1118-1125` `RejectReason` constants `paused, draining, busy, no-fit, engine-off, wake-timeout`. `MethodWorkloadCancel = "workload/cancel"` (`wire.go:1269`), `WorkloadCancelParams{WorkloadID, Engine, RunID, Regenerate}` (`wire.go:1277-1287`).
- Corpus directory `N/nodepolicy/testdata/` holds `deployment.json`, `served-resident.json`, `strict-admission-raw.json`, `strict-admission.json`, read by `N/nodepolicy/routing_wire_test.go` and by `desktop/tests/modular/node-policy.test.ts:46-96` via `resolve(process.cwd(), '../services/shared/nodepolicy/testdata/<name>')`.

**Proxy request path (`P/http_handler.go`)**

- `handleHTTP` (`:505`) → `handleHTTPWithStreamIdle` (`:510`); `start := time.Now()` at `:512`; `reqID := strconv.FormatUint(f.nextRequestID.Add(1), 10)` at `:520` is a **per-facade** counter, not process-wide.
- `ctl := newWorkloadCtl(cancelRequest)` at `:669`; candidate resolution `:670-686`; no-candidate rejections `:732-779`; the `Workload` is built and `workload:submitted` emitted at `:817-847` (`createdMs := start.UnixMilli()` at `:830`); `emitTerminal := func(state, errMsg string)` at `:865-893` is the single place every terminal `workload:completed` / `workload:errored` is emitted; the finalize `defer` at `:946-1030` classifies `cancelled` / `failed` / `completed` and builds the stream-interruption message `"stream interrupted after output began on <node>: <reason>"`; `repointWorkload` at `:1037-1053`.
- Attempt loop from `:1120`: exhaustion branches `:1148-1189`; attempt start `repointWorkload(cand.id, candModel)` at `:1224`; `attemptCtx, cancelAttempt := context.WithCancel(r.Context())` at `:1235` followed by `claim` and `att := &attemptCtl{...}` (`:1235-1239`); model rewrite failure `:1247-1256`; admission wait `:1260-1263`; self admission `p.admission.admit` `:1274-1279`; `rejectSelf` `:1280-1294`; `decision.err` / redispatched `:1295-1303`; `decision.reject` and candidate exclusion `:1304-1316`; profile-defaults mismatch `:1319-1329`; dormant target `:1330-1341`; stream flag `:1345-1358`; discovery watcher `:1377-1400`.
- `ModifyResponse` `:1443-1612` decides retry (`shouldRetry`, `:784-797`), waits for the first body byte (`awaitFirstBody`, `errFirstBodyTimeout`), commits the claim, sets `X-PAIR-*` headers, records `ttfbMs` and `servedNodeID`, emits `proxy/request-started` and `workload:started`. `ErrorHandler` `:1613-1660`. After `proxy.ServeHTTP(sc, attemptReq)` returns (`:1670-1685`) the loop classifies `att.redispatched()` → `"attempt aborted to re-dispatch the request"` and `claim.abandoned()` → `errTargetGone`. Loop-exhaustion 503 reasons `:1699-1735`.
- `P/cancel.go:35-42` `attemptCtl{claim, cancel, nodeID, redispatch atomic.Bool}`; `abortForRedispatch` `:46-54`; `workloadCtl` `:62-72`; `cancel(regenerate)` `:115-142` (message `"cancelled by request"`); `abortLocalIf` `:153-167` (message `"cancelled: this node stopped accepting work"`); `handleWorkloadCancel` `:185-220`.
- `P/admission.go:289-298` `admissionRequest{engine, model, waitCap, cancel}`, `:302-308` `admissionDecision{ticket, profile, hasProfile, reject, err}`.
- `P/accounts.go:55-72` ledger errors `errLedgerUnavailable, errAccountFrozen, errAccountCredential, errAccountCooldown, errAccountCeiling, errDeploymentQuarantined, errAccountConflict`; `accountLedger` `:78-110`; `reserve` `:387`; `reconcile` `:771`; `recordOutcome` `:873`. `P/accountstore.go:69-76` `accountOutcome` values `ok, rate-limited, exhausted, credential-failure, transient`.
- `P/routing_listener.go:265-305` the hybrid listener answers inference routes with 501 `"hybrid routing is not available for inference yet"`; today only `handleHTTP` runs inference, so J0 instruments one path.
- `P/streamframe.go:112-116` `streamOutcome` values `Streaming, Completed, Interrupted, UpstreamError, Cancelled`; `streamFrameResult{Outcome, Reason, Numeric}` `:208-212`.
- `P/proxy.go`: `workload:*` method constants `:152-157`; `Workload` struct `:172-211` (includes `RequestedModel`, `Seq`); `Proxy` fields `serveCtx` `:280`, `codec`, `runID` `:327`, `routing *routingListener` `:337`, `accounts *accountLedger` `:342`, `selfMu/selfNodeID/selfModels` `:349-351`, `workloadsMu/workloads` `:355-356`; `NewProxy` `:361-382`; `emitWorkload` `:578-582`; `candidate{id, url, peerUUID, fac, engine, model, self, dormant, warm, tierRank, declared}` `:590-615`; `handleMessage` `:1303` with `handlePolicyMethod` at `:1359` and `handleWorkloadCancel` at `:1363`.
- `P/spec.md`: §5.1 retry bounds (`maxDispatchAttempts 5`, `jobDeadline 10 min`, `firstBodyTimeout 120 s`), §5.2 SC-a managed streaming (`:552-650`, "After commit there is no retry or continuation in SC-a. SC-b's qualified same-model continuation remains pending."), §5.4 outcome classification (`:677-765`), §5.5 workload lifecycle (`:765-806`), §12.1 broker→proxy table (`:992-1013`), §12.5 cancel (`:1272-1290`), §12.7 ledger (`:1393-1513`).
- Test helpers: `newTestProxy` `P/failover_test.go:36`, `nodeForModel` `:84`, `notificationsOf` `P/policy_test.go:43`, `rpc` `:57`, `waitForCond` `P/retry_test.go:54`, `forEachEngine` `P/enginecase_test.go:115`, `newRecordingEngine` `P/admission_http_test.go:280`, `stallingEngine` `:323`, `startCancelFixture` `P/cancel_test.go:19-45`, `NewDiscovery` `P/discovery.go:117`, `AddManual` `:197`.

**Broker**

- `B/broker.go:177` `nodeID string`; subscription flag pattern `workloadsMu` / `workloadsSubscribed` `:333-339`; `forwardProxyProcessNotification` `:841-866` sends unaddressed proxy methods to `routeProcessScopedProxyNotification` (`B/engineproxy.go:545-565`), whose `switch` handles `methodPerformanceSample`, `proxyWorkloadMethods` → `routeProxyWorkload`, node activity, errors, then admission notifications.
- `B/workload_projection.go:150-156` `proxyWorkloadMethods`; `routeProxyWorkload` `:164-186` = `stampNodeID` → `emitWorkloadEvent("workloads:upsert")` → `wm.Forward`; `stampNodeID` `:195-224`; `emitWorkloadEventProvenance` `:303-324` calls `b.codec.Notify(method, params)` only when `workloadsSubscribed`; `workloadsInitialResult` `:390-392`.
- `B/client_requests.go:24-66` `internalClientMethods`; `handleMessage` `:130`; `workloads:subscribe` `:247-257`, `workloads:unsubscribe` `:259-265`, `workloads:get-initial` `:267-278`.
- `B/proxy.go:344-349` `clearProxyHandles` runs when the proxy process goes away; `handleNotify` `:225-255`.
- `B/workloadcancel.go` whole file: `methodWorkloadsCancel = "workloads:cancel"`, `cancelLocalWorkload` via `b.policyProxy()`.
- `B/policy.go:40-43` method constants `policy:get`, `policy:set`, `node:set-availability`, `policy:changed`; `notifyClients` `:676-684`.
- Test helpers (`B/policy_test.go`): `attachFakeProxy` `:220`, `syncBuffer` and `waitFrame` `:236-282`, `newPolicyBroker` `:284-289` (`&Broker{codec: NewCodec(out), nodeID: "self-node", ...}`), `clientRequest` `:302`, `isResponse` `:316`; `B/workloadcancel_test.go` shows `b.setWorkloadMgr(&workloadManagerProcess{stdin: writer, done: ...})`.
- `W/manager.go:403` `broadcastFrame(method, params json.RawMessage)` re-marshals the proxy's params **verbatim** into the cluster frame, and `W/workload.go:111` `parseLifecycle` does not reject unknown fields. Any field added to the `Workload` struct therefore reaches every paired node. This is why routing detail must travel on its own method and why `routeProxyWorkload` gets an allowlist.
- `N/jsonrpc/jsonrpc.go`: `Codec.Notify(method, params any)` `:262`, `Respond` `:287`, `RespondError` `:304`. `N/jsonrpc/engines.go:125` `ProxyComponent = "nvpair-proxy"`.

**Desktop**

- `desktop/scripts/verify-service-contracts.ts` detects emitted methods by `notify(` / `emit(` / `publish(` / `broadcast(` followed by a literal or a local `const X = "..."` declared in that binary's non-test Go files; handled methods by `case "literal"`; UI references only inside `D/electron/service-bridge/` and `D/shared/types/ws-channels.ts`. `--check` exits 1 on `missingNotifications` (emitted by a Go binary, referenced nowhere in the desktop) unless listed in `desktop/docs/service-contract-exceptions.json` `ignoredMethods`, and on a stale `desktop/docs/services-api.md`. Consequence: the proxy's new `routing/request` must be in `ignoredMethods` (it is proxy→broker only), and the broker's `routing:changed` must be referenced in the desktop bridge **in the same commit** as the broker emits it.
- `D/shared/types/ws-channels.ts` invoke map `:153-165` (`workloads:get-initial`, `workloads:cancel`, `policy:get`, `policy:set`, `node:set-availability`) and push map `:194-205` (`policy:changed`, `workloads:upsert`, `workloads:remove`).
- `D/electron/service-bridge/workload-projection.ts` parses workloads with `objectValue` / `stringValue` / `numberValue` from `./node-projection`; `modular-state.ts` holds the `workloads` Map (`getWorkloads` `:697`, `seedWorkloads` `:734`, `upsertWorkloadFromInfo` `:762-772`, `removeWorkloadFromParams` `:775-792`); `modular-supervisor.ts` subscribes `workloads:subscribe` at `:1078` then `seedWorkloadBaseline` at `:1080` and routes notifications at `:1484-1528`; `empty-handlers.ts` `handleWorkloadsGetInitial` `:489-500` and the handler map `:782-825`; `D/ui/api/pair-api.ts` `IWorkloadsApi` `:102-113`, `INodePolicyApi` `:115-129`, `IPairApi` `:153-164`, `createPairApi` `:170-254`.
- `D/ui/stores/workloads.store.ts` is the rAF-batched zustand pattern (subscribe before `getInitial`, `pendingOps`, `cleanup`); `D/ui/stores/init.ts` calls `useWorkloadsStore.getState().initialize()` in `initializeAllStores`, and also has `cleanupAllStores` and `resyncAllStores`.
- `D/ui/components/RoutingSettings/RoutingSettings.tsx` (500 lines) is mounted from `D/ui/components/Settings/Settings.tsx:38` when `activeTab === 'routing'`; it renders a node `Dropdown`, availability `Badge`, `RoleEditor`, policy `TextArea`, and `BulkNodeConfiguration` inside `settings-card` sections. `D/ui/components/Workloads/WorkloadItemCard.tsx` uses KUI `Button, Card, Flex, Stack, Text` and shows error text only when `state === 'failed'`.
- KUI components already imported somewhere in `D/ui/`: `Badge Banner Button Card Checkbox Divider Dropdown Flex FormField Modal Panel RadioGroup Select Stack Switch Table Tabs Tag Text TextInput Tooltip VerticalNav` from `@nvidia/foundations-react-core`.
- Desktop tests: `vitest.config.ts` includes `tests/modular/**/*.test.{ts,tsx}`, environment `node`, alias `@`; `desktop/tests/modular/node-policy.test.ts:860-912` shows the jsdom component harness (`vi.stubGlobal`, `useConnectionStore.setState({ connected: true, selfId: SELF })`, `createRoot`, `act`); `workload-card-labels.test.ts` is a pure-function label test; `node-configuration-store.test.ts` shows a typed `ServiceTransport` fake. Scripts: `npm run lint`, `typecheck`, `test:unit`, `dead-code:check`, `service-contracts`, `service-contracts:write`, `service-contracts:check`.
- `desktop/docs/services-backend.md:128-143` notification table; `desktop/docs/frontend-api.md:141,179,194` `pairApi` tables; `B/README.md` JSON-RPC surface from `:657` (`workloads:upsert/remove` `:717-742`, `policy:changed` `:760`, requests `:783+`).

**Terminal interface**

- `T/view.go:21-38` `View` interface (`Title, Init, SetSize, Update, View, Help`); `T/ui.go:67-75` `defaultViews` returns nodes, jobs, service, errors, logs; `T/keys.go:119` `maxTabs = 9` (one slot is free for `Routing`).
- `T/rpccmd.go`: `callTimeout 35s`, `TickMsg`, `NotificationMsg`, `DisconnectedMsg`, `sessionReadyMsg`, `call`, `callWithin`, `waitForNotification`. J2 needs nothing new here (the brief lists `rpccmd.go`; see Decisions made while writing this plan).
- `T/jobs.go` is the model for a subscribed, table-rendered tab: `feedCmd` subscribes then `loadCmd` calls `get-initial`; `Update` handles `DisconnectedMsg`, `sessionReadyMsg`, `TickMsg`, `NotificationMsg` (including `discovery:nodes-changed` and `nodes:changed` for the `nodeNamer`); `maxFinishedJobs = 200`; `View` uses `fitTable` and `joinLines`.
- Helpers: `fixedCol/flexCol/layoutColumns/newTable/fitTable/restoreCursor` `T/table.go:136-200`; `decodeOrLog` `T/decode.go:32`; `toast` `T/toast.go`; `nodeNamer` `T/nodenames.go` (`learnDiscovered :55`, `learnMembers :61`, `setSelf :69`, `name :86`), `nodeIdentityCmd :44`; `ageLabel` `T/errors.go:478`; `engineDisplayName` `T/proxystatus.go:64`; `footerStyle` `T/styles.go:47`; `T/policywire.go` already imports `nvpair-shared/nodepolicy`. Tests: `newReconnectUIFixture` `T/model_test.go:37`; `T/main_test.go` has `TestMain`.
- Docs: `services/nvpair-tui/README.md:36-37` tab table; `docs/terminal-interface.mdx:193` and `:390-400` describe the Jobs tab.

**Cross-process tests**

- `X/workload_interop_test.go:558` `TestWorkloadManagerOutboundBroadcast` drives a real broker + workload-manager and captures the outbound cluster frame; `newServiceFixture` builds the fixture. These run on PC3 only.

### What could not be verified

- No Go 1.25 toolchain was available where this plan was written (modules require 1.25). No Go test or build was run; line numbers were verified by reading the source only. Executors must build and test on Mac/MSI/PC3 as stated in each task.
- Whether `@nvidia/foundations-react-core` exports a collapsible/accordion component was not verified; the plan therefore uses `Button` toggles and conditional rendering, which are known to be in use.
- The I plan is not in the repository. Everything under "I-dependent seams" below is an assumption derived from current code, marked `[I-SEAM]`, for the coordinator to reconcile.

### I-dependent seams (assumptions, all marked `[I-SEAM]`)

- `[I-SEAM-ID]` Request identity. J0 uses `nodepolicy.RoutingRequestKey` as the record key: `OriginNodeID` = `p.selfNodeID` (`P/proxy.go:350`, empty before the broker announced it; the broker then stamps it, Task 3), `ProxyRunID` = `p.runID` (`P/proxy.go:327`), `RequestID` = a **new process-wide** `atomic.Int64` on `Proxy` minted at request start (Task 2). Assumption: I's unified identity will mint the same key in the same place; if I lands first, Task 2 replaces the local counter with I's field and the record shape does not change.
- `[I-SEAM-ATTEMPT]` Attempt records. J0 hooks the existing loop points listed above; it does not introduce an attempt object beyond `attemptCtl`. Assumption: I's attempt records expose at least destination, start/end, outcome and abort reason; J0's `routingAttemptRecorder` interface (Task 2) is the adapter point.
- `[I-SEAM-CANCEL]` Cancellation. J0 distinguishes `cancel` (client/UI cancel), `regenerate`, `node-paused` (`abortLocalIf`), `proxy-shutdown` (`serveCtx` done) by adding an `abortReason` string to `attemptCtl` (Task 2). Assumption: I's cancellation carries an equivalent reason; the closed reason set in Task 1 is the contract, not the field name.
- `[I-SEAM-ACTIVE]` I Task 6 publishes broker `routing:active` / `routing:cancel` and proxy `routing/active`. J0's `routing/request`, `routing:changed`, `routing:subscribe`, `routing:unsubscribe`, `routing:get-initial` are **different** methods and must not be merged into them. If I reserves the `routing:` prefix for its own subscription flag, the broker's `routingSubscribed` flag (Task 3) is the single place to reconcile.
- `[I-SEAM-CONT]` SC-b continuation. `RoutingAttemptView.continuationOf` (Task 1) is reserved now and always `null` in J0; J1/J2 render it. No proxy code sets it in this plan.

## Decisions (settled — do not revisit)

- **PerRequestCeiling is cumulative across one routing request**: the request's ceiling is the tightest `PerRequestCeiling` of any paid account tried during that request, and there is no refill between attempts. Not per-attempt ceilings, because the owner decided the user's exposure per request is a single number. Not "sum of ceilings", because that would let a retry double the spend. Computed by the pure helper in Task 1; the ledger is not changed.
- **Interrupted streams are shown as a delay or pause-then-resume**, never as an error, unless no same-model continuation is possible, in which case the request is `failed` with reason `stream-interrupted`. Not "show the interruption as failed immediately", because SC-b continuation is pending and the UI must not promise an error the proxy may still recover from. Today (`P/spec.md` §5.2) SC-a has no continuation, so after commit an interruption is terminal: the record then reads `failed / stream-interrupted` and the attempt reads `interrupted`.
- **Desktop UI uses the existing KUI component library and the existing Routing settings layout.** Not a new page, route or custom CSS component, because the owner wants no new layouts.
- **Money is a decimal string end to end** (`NanoUSD` in Go, `NanoUsd = string` in TS, string in the TUI). Not `number` anywhere, because exactness is required. Formatting is done by string arithmetic (Task 6, Task 7).
- **Routing detail never rides on the `Workload` struct** and is never forwarded by the broker to the workload-manager, scheduler or peers. Not "add optional fields to `Workload`", because `W/manager.go:403` broadcasts params verbatim cluster-wide.
- **History is bounded**: 256 requests per broker, 32 attempts per request, oldest dropped first; `droppedAttempts` counts what was trimmed. Not unbounded, because the broker and renderer hold it in memory.
- **No prompt, message, body, PIN or key material** appears in any record, log line or test fixture. Reasons are closed-set codes; the only free text is the upstream HTTP status (an integer) and node/model identifiers.

## Decisions made while writing this plan

1. Transport is a whole-record snapshot per change (`routing/request` carries the full `RoutingRequestView`), not per-attempt deltas. Simpler to replay through `routing:get-initial`, and records are at most 32 attempts.
2. Broker holds the history (not the proxy), because the broker outlives proxy restarts within a session and already owns client subscriptions. On proxy exit, non-terminal records are marked `failed / proxy-exited` by the broker.
3. Account and deployment **IDs** (the user's own policy identifiers, not credentials) appear in the local private record so the user can see which account was tried. Credentials, billing and raw provider responses never appear. See Owner question 2.
4. `routing/request` is listed in `desktop/docs/service-contract-exceptions.json` `ignoredMethods` with reason "proxy→broker only; the broker re-emits it as routing:changed".
5. Waits are integers in milliseconds (`admissionMs`, `firstByteMs`), formatted in the clients.
6. The TUI tab is named `Routing`, placed after `Jobs` in `defaultViews`.
7. `T/rpccmd.go` is not modified: the existing `call` and `NotificationMsg` already cover subscribe / get-initial / push. The brief listed it; leaving it untouched is a deliberate narrowing.
8. The desktop workload card shows a single summary line ("Routing: 2 attempts, served on <node>") derived from the routing store by workload correlation `(engine, workload id)`; it does not duplicate the attempt table.
9. One shared corpus file `routing-request.json` is the frozen interface. Go (`N`), desktop (`D`) and TUI (`T`) tests all parse it and assert the same field values, so a change to the view shape fails three suites.

## Tasks

Dependency graph (acyclic): 1 → 2 → 3 → 5; 1 → 4 → 7; 3 → 6; 3 → 7. Tasks 2 and 4 may run in parallel after 1. Tasks 5, 6 may run in parallel after 3. Task 7 needs 3 and 4.

### Task 1 — Shared routing view wire types, closed reason set, ceiling helper, and the frozen JSON corpus

**Files:**
- modify `services/shared/nodepolicy/routing_wire.go`
- modify `services/shared/nodepolicy/routing_wire_test.go`
- create `services/shared/nodepolicy/testdata/routing-request.json`

**Depends on:** none

**Complexity:** judgment

**Do:**

1. Append to `services/shared/nodepolicy/routing_wire.go` (keep the existing content and SPDX header):

```go
// RoutingViewSchema is bumped only when a field changes meaning or type.
const RoutingViewSchema = 1

// RoutingHistoryLimit and RoutingAttemptLimit bound what the broker keeps and
// what any client is expected to render.
const (
	RoutingHistoryLimit = 256
	RoutingAttemptLimit = 32
)

// Request states.
const (
	RoutingStateRouting   = "routing"
	RoutingStateStreaming = "streaming"
	RoutingStateCompleted = "completed"
	RoutingStateFailed    = "failed"
	RoutingStateCancelled = "cancelled"
)

// Attempt outcomes.
const (
	AttemptPending      = "pending"
	AttemptCommitted    = "committed"
	AttemptRejected     = "rejected"
	AttemptFailed       = "failed"
	AttemptRedispatched = "redispatched"
	AttemptInterrupted  = "interrupted"
	AttemptCancelled    = "cancelled"
)

// Destination kinds.
const (
	DestinationLocal = "local" // a node in this cluster (self or peer)
	DestinationAPI   = "api"   // a routing deployment reached through an account
)

// Account states as seen by one attempt.
const (
	AccountStateOK         = "ok"
	AccountStateCooldown   = "cooldown"
	AccountStateFrozen     = "frozen"
	AccountStateCredential = "credential-failure"
	AccountStateExhausted  = "exhausted"
	AccountStateProbe      = "probe"
	AccountStateConflict   = "conflict"
)

// RoutingReason codes. Closed set: clients map each to copy and must treat an
// unknown code as "unknown" without failing.
const (
	ReasonNone                 = ""
	ReasonClientGone           = "client-gone"
	ReasonCancel               = "cancel"
	ReasonRegenerate           = "regenerate"
	ReasonNodePaused           = "node-paused"
	ReasonProxyShutdown        = "proxy-shutdown"
	ReasonProxyExited          = "proxy-exited"
	ReasonTargetLeft           = "target-left"
	ReasonTransport            = "transport"
	ReasonUpstreamStatus       = "upstream-status"
	ReasonNoContent            = "no-content"
	ReasonStreamInterrupted    = "stream-interrupted"
	ReasonUpstreamError        = "upstream-error"
	ReasonAdmissionPaused      = "admission-paused"
	ReasonAdmissionDraining    = "admission-draining"
	ReasonAdmissionBusy        = "admission-busy"
	ReasonAdmissionNoFit       = "admission-no-fit"
	ReasonAdmissionEngineOff   = "admission-engine-off"
	ReasonAdmissionWakeTimeout = "admission-wake-timeout"
	ReasonEngineNotRunning     = "engine-not-running"
	ReasonRequestInvalid       = "request-invalid"
	ReasonNoOwner              = "no-owner"
	ReasonRegenerateExhausted  = "regenerate-exhausted"
	ReasonAttemptsExhausted    = "attempts-exhausted"
	ReasonDeadline             = "deadline"
	ReasonRequestCeiling       = "request-ceiling"
	ReasonAccountCeiling       = "account-ceiling"
	ReasonAccountCooldown      = "account-cooldown"
	ReasonAccountFrozen        = "account-frozen"
	ReasonAccountCredential    = "account-credential"
	ReasonAccountExhausted     = "account-exhausted"
	ReasonAccountConflict      = "account-conflict"
	ReasonLedgerUnavailable    = "ledger-unavailable"
	ReasonDeploymentQuarantine = "deployment-quarantined"
	ReasonPeerFraming          = "peer-framing"
)

// KnownRoutingReasons lists every code above except ReasonNone.
func KnownRoutingReasons() []string { /* return all codes in declaration order */ }

// AdmissionReasonCode maps a RejectReason to its routing reason code.
func AdmissionReasonCode(r RejectReason) string {
	switch r {
	case RejectPaused: return ReasonAdmissionPaused
	case RejectDraining: return ReasonAdmissionDraining
	case RejectBusy: return ReasonAdmissionBusy
	case RejectNoFit: return ReasonAdmissionNoFit
	case RejectEngineOff: return ReasonAdmissionEngineOff
	case RejectWakeTimeout: return ReasonAdmissionWakeTimeout
	}
	return ReasonTransport
}
```

   Use the actual `RejectReason` constant names from `wire.go:1118-1125` in the switch.

2. Add the view structs. Every money field is `*NanoUSD` so `null` is representable; every time field is Unix milliseconds `int64`; pointers for optional ints.

```go
type RoutingDestinationView struct {
	Kind         string `json:"kind"`                   // DestinationLocal | DestinationAPI
	NodeID       string `json:"nodeId,omitempty"`       // local only
	Engine       string `json:"engine,omitempty"`       // local only
	DeploymentID string `json:"deploymentId,omitempty"` // api only
	AccountID    string `json:"accountId,omitempty"`    // api only
	Model        string `json:"model"`
}

type RoutingWaitsView struct {
	AdmissionMs int64  `json:"admissionMs"`
	FirstByteMs *int64 `json:"firstByteMs"` // null until the first body byte
}

type RoutingAccountStateView struct {
	State        string   `json:"state"` // AccountState*
	HeldNanoUsd  *NanoUSD `json:"heldNanoUsd"`
	ChargedNanoUsd *NanoUSD `json:"chargedNanoUsd"`
}

type RoutingAttemptView struct {
	Index          int                      `json:"index"`
	StartedAt      int64                    `json:"startedAt"`
	EndedAt        *int64                   `json:"endedAt"`
	Destination    RoutingDestinationView   `json:"destination"`
	Outcome        string                   `json:"outcome"` // Attempt*
	Reason         string                   `json:"reason"`  // Reason*
	Status         *int                     `json:"status"`  // upstream HTTP status, null if none
	Waits          RoutingWaitsView         `json:"waits"`
	ContinuationOf *int                     `json:"continuationOf"` // [I-SEAM-CONT] index of the interrupted attempt this one continues; always null in J0
	Account        *RoutingAccountStateView `json:"account"`        // null for local destinations
}

type RoutingSpendView struct {
	Currency       string   `json:"currency"` // always "USD"
	CeilingNanoUsd *NanoUSD `json:"ceilingNanoUsd"`
	ReservedNanoUsd *NanoUSD `json:"reservedNanoUsd"`
	ChargedNanoUsd *NanoUSD `json:"chargedNanoUsd"`
}

type RoutingWorkloadRef struct {
	Engine string `json:"engine"`
	ID     string `json:"id"`
}

type RoutingRequestedView struct {
	Model string `json:"model"`
	Tier  string `json:"tier"` // "" when the client sent none
	Mode  string `json:"mode"` // "chat" | "completion" | "embedding" | "other"
}

type RoutingRequestView struct {
	Schema          int                  `json:"schema"`
	Request         RoutingRequestKey    `json:"request"`
	Workload        RoutingWorkloadRef   `json:"workload"`
	Seq             int64                `json:"seq"`
	Requested       RoutingRequestedView `json:"requested"`
	CreatedAt       int64                `json:"createdAt"`
	EndedAt         *int64               `json:"endedAt"`
	State           string               `json:"state"`  // RoutingState*
	Reason          string               `json:"reason"` // Reason*
	Status          *int                 `json:"status"` // HTTP status returned to the client, null while open
	Served          *RoutingDestinationView `json:"served"`
	Spend           RoutingSpendView     `json:"spend"`
	DroppedAttempts int                  `json:"droppedAttempts"`
	Attempts        []RoutingAttemptView `json:"attempts"`
}
```

3. Add `func (v *RoutingRequestView) Validate() error` that: requires `Schema == RoutingViewSchema`; calls `v.Request.Validate()` **except** that an empty `OriginNodeID` is allowed (the broker stamps it; after stamping, callers use `ValidateStamped()` which requires it); requires `State` in the five states; requires `Reason` to be `ReasonNone` or a known code; requires `len(Attempts) <= RoutingAttemptLimit`; requires attempt indexes strictly increasing; requires every money string to pass `ParseNanoUSD` when non-nil; requires `Spend.Currency == "USD"`; requires `Destination.Kind` to be one of the two kinds and the kind-specific fields to be set (`NodeID` and `Engine` for local, `DeploymentID` and `AccountID` for api). Add `func DecodeRoutingRequestView(raw []byte) (RoutingRequestView, error)` using `decodeStrictContract` (`routing_wire.go:152-177`) followed by `Validate()`.

4. Add the pure ceiling helper (owner decision, exact int64 math, no float):

```go
// RequestCeiling implements the cumulative per-request ceiling: the tightest
// PerRequestCeiling of every paid account tried in one request. An account
// with no ceiling does not loosen it. nil means no paid account was tried.
func RequestCeiling(ceilings []NanoUSD) (*NanoUSD, error)
```

   Behaviour: skip empty strings; parse each with `ParseNanoUSD`; keep the minimum by `Int64()`; return the canonical string of the minimum (re-encode from int64 with `strconv.FormatInt`); return `nil, nil` when nothing was paid; return the first parse error otherwise.

5. Create `services/shared/nodepolicy/testdata/routing-request.json`: a JSON **array** of three `RoutingRequestView` objects (schema 1), with `request.originNodeId = "node-a"`, `proxyRunId = "run-1"`, `requestId` `"1"`, `"2"`, `"3"`; `workload.engine = "llama.cpp"`, ids `"wl-1"`, `"wl-2"`, `"wl-3"`; `requested.model = "lfm2.5-24b-a2b"`, tier `""`, mode `"chat"`. Record 1: `completed`, two attempts: index 0 local `node-b` rejected `admission-busy` with `admissionMs 350`, index 1 local `node-a` committed `status 200`, `firstByteMs 1200`; `served` = attempt 1's destination; spend all `null`, currency `"USD"`. Record 2: `failed / stream-interrupted`, one attempt index 0 destination api `deploymentId "dep-paid"`, `accountId "acct-1"`, outcome `interrupted`, reason `stream-interrupted`, `status 200`, account `{state:"ok", heldNanoUsd:"250000000", chargedNanoUsd:"123456789"}`, spend `{currency:"USD", ceilingNanoUsd:"1000000000", reservedNanoUsd:"250000000", chargedNanoUsd:"123456789"}`, `droppedAttempts 0`. Record 3: `cancelled / cancel`, `droppedAttempts 2`, one attempt index 2 local, outcome `cancelled`, reason `cancel`, `status null`, `endedAt` set; request `status 499`. All timestamps are fixed integers (for example `1760000000000`-based). No other identifiers, no prose fields.

6. In `routing_wire_test.go` add:
   - `TestRoutingRequestViewCorpusRoundTrip`: reads the corpus, decodes each element with `DecodeRoutingRequestView`, re-marshals, and compares the re-marshalled JSON to the original after `json.Compact`; asserts the values listed in step 5 (counts, states, reasons, money strings) so the corpus cannot drift silently.
   - `TestRoutingRequestViewRejectsUnknownField`, `TestRoutingRequestViewRejectsFloatMoney` (`"chargedNanoUsd": 1.5` fails), `TestRoutingRequestViewRejectsUnknownReason`, `TestRoutingRequestViewAttemptLimit` (33 attempts fail).
   - `TestRequestCeiling`: table cases: `[]` → nil; `["5", ""]` → `"5"`; `["300", "200", "250"]` → `"200"`; `["x"]` → error; `["0007"]` → error (non-canonical per `ParseNanoUSD`).
   - `TestAdmissionReasonCodeCoversAllRejectReasons`: every `RejectReason` constant maps to a code starting with `admission-`.

**Verify:**
- Mac or MSI: `cd services/shared && go vet ./... && go test -race ./nodepolicy/` passes, including the five new test names above.
- PC3: `cd services/shared && go test ./nodepolicy/` passes (no `-race`).
- `node scripts/spdx-headers.mjs` from the repository root reports no missing header (the JSON corpus carries no header; the script skips `.json`, as the existing corpus files show).
- `git diff --stat` touches only the three files listed.

### Task 2 — Proxy emits a bounded, redacted `routing/request` record for every inference request

**Files:**
- create `services/nvpair-proxy/routing_events.go`
- create `services/nvpair-proxy/routing_events_test.go`
- modify `services/nvpair-proxy/http_handler.go`
- modify `services/nvpair-proxy/cancel.go`
- modify `services/nvpair-proxy/proxy.go`
- modify `services/nvpair-proxy/spec.md`
- modify `services/nvpair-proxy/README.md`
- modify `desktop/docs/service-contract-exceptions.json`
- modify `desktop/docs/services-api.md` (regenerated only)

**Depends on:** 1

**Complexity:** judgment

**Do:**

1. `routing_events.go` (new, SPDX header, `package main` like its neighbours):

```go
// methodRoutingRequest is proxy -> broker only. The broker never forwards it.
const methodRoutingRequest = "routing/request"

type routingRequestParams struct {
	nodepolicy.RoutingRequestView
}

// routingRecorder builds one RoutingRequestView for one inference request and
// emits a full snapshot after every change. Safe for use from the request
// goroutine and the ModifyResponse/ErrorHandler goroutines.
type routingRecorder struct {
	mu       sync.Mutex
	p        *Proxy
	view     nodepolicy.RoutingRequestView
	ceilings []nodepolicy.NanoUSD // every paid account's PerRequestCeiling tried so far
}

func (p *Proxy) newRoutingRecorder(engine, workloadID, model, tier, mode string, createdAt time.Time) *routingRecorder
func (r *routingRecorder) attemptStarted(dest nodepolicy.RoutingDestinationView, startedAt time.Time) (index int)
func (r *routingRecorder) attemptAdmitted(index int, admissionWait time.Duration)
func (r *routingRecorder) attemptFirstByte(index int, firstByte time.Duration, status int)
func (r *routingRecorder) attemptAccount(index int, state string, held, charged *nodepolicy.NanoUSD, perRequestCeiling nodepolicy.NanoUSD)
func (r *routingRecorder) attemptEnded(index int, outcome, reason string, status *int, endedAt time.Time)
func (r *routingRecorder) streaming()
func (r *routingRecorder) finished(state, reason string, status int, endedAt time.Time)
func (r *routingRecorder) emit() // marshals a copy of view, p.codec.Notify(methodRoutingRequest, params)
```

   `attemptStarted` appends; when `len(Attempts) == RoutingAttemptLimit` it drops `Attempts[0]` and increments `DroppedAttempts` before appending (index keeps counting from the request's first attempt, so indexes stay unique). `attemptAccount` appends `perRequestCeiling` to `ceilings` when non-empty and recomputes `view.Spend.CeilingNanoUsd` with `nodepolicy.RequestCeiling`; it sums `held` into `ReservedNanoUsd` and `charged` into `ChargedNanoUsd` using `Int64()` and `strconv.FormatInt`, never float. `emit()` is called at the end of every mutator while **not** holding `mu` across the `Notify` call (copy under lock, notify after). `[I-SEAM-ATTEMPT]` These eight methods are the adapter surface I's attempt records plug into.

   Request identity `[I-SEAM-ID]`: add `routingSeq atomic.Int64` to `Proxy` (`P/proxy.go:265-357`); `newRoutingRecorder` sets `Request = RoutingRequestKey{OriginNodeID: p.currentSelfNodeID(), ProxyRunID: p.runID, RequestID: strconv.FormatInt(p.routingSeq.Add(1), 10)}` where `currentSelfNodeID` reads `selfNodeID` under `selfMu` (`P/proxy.go:349-351`). `Seq` is a per-record counter incremented on every `emit()`.

2. `cancel.go`: add `abortReason string` guarded by the existing struct's usage to `attemptCtl` (`:35-42`) with setter `setAbortReason(reason string)` using a `sync.Once`-protected store (first reason wins); `abortForRedispatch` (`:46-54`) sets `nodepolicy.ReasonRegenerate` when called from `cancel(regenerate=true)` (`:115-142`) and `nodepolicy.ReasonNodePaused` when called from `abortLocalIf` (`:153-167`). `cancel(regenerate=false)` sets `nodepolicy.ReasonCancel`. Add `func (a *attemptCtl) abortReasonOr(fallback string) string`. `[I-SEAM-CANCEL]`

3. `http_handler.go` hook points (keep every existing behaviour; add calls only):
   - At `:830-847` right after the `Workload` is built: `rec := p.newRoutingRecorder(f.engine, wl.ID, model, tierName, mode, start)` where `mode` is `"chat"` for the chat completions route, `"completion"` for completions, `"embedding"` for embeddings, else `"other"` (derive from the route already matched at `:602-668`). When `isInf && model != ""` is false, `rec` stays nil and every later call is nil-safe.
   - At `:1224` (attempt start): `idx := rec.attemptStarted(destinationOf(cand, candModel), time.Now())` where `destinationOf` returns `{Kind: local, NodeID: cand.id, Engine: cand.engine, Model: candModel}`; for a routing deployment candidate (none exists today; `P/routing_listener.go:265-305` returns 501) the function returns `Kind: api` with `DeploymentID`/`AccountID` — leave that branch in place, documented as unreachable until hybrid dispatch lands.
   - At `:1260-1263` after the admission wait is known: `rec.attemptAdmitted(idx, admissionWait)`.
   - `:1247-1256`, `:1319-1329`, `:1345-1358` (model rewrite failure / profile mismatch / stream flag failure): `rec.attemptEnded(idx, AttemptFailed, ReasonRequestInvalid, nil, now)`.
   - `:1295-1303` (`decision.err`): `rec.attemptEnded(idx, AttemptFailed, ReasonTransport, nil, now)`; when `att.redispatched()` is true use `AttemptRedispatched` with `att.abortReasonOr(ReasonRegenerate)`.
   - `:1304-1316` (`decision.reject`): `rec.attemptEnded(idx, AttemptRejected, nodepolicy.AdmissionReasonCode(decision.reject), nil, now)`.
   - `:1330-1341` (dormant target): `rec.attemptEnded(idx, AttemptRejected, ReasonEngineNotRunning, nil, now)`.
   - In `ModifyResponse` (`:1443-1612`): when `shouldRetry` fires → `rec.attemptEnded(idx, AttemptRejected, ReasonUpstreamStatus, &resp.StatusCode, now)`; on `errFirstBodyTimeout` → `attemptEnded(idx, AttemptFailed, ReasonNoContent, &resp.StatusCode, now)`; on commit → `rec.attemptFirstByte(idx, ttfb, resp.StatusCode)` then `rec.streaming()` when the response is a stream.
   - In `ErrorHandler` (`:1613-1660`): `attemptEnded(idx, AttemptFailed, ReasonTransport, nil, now)`.
   - After `proxy.ServeHTTP` returns (`:1670-1685`): if `att.redispatched()` → `attemptEnded(idx, AttemptRedispatched, att.abortReasonOr(ReasonRegenerate), nil, now)`; if `claim.abandoned()` → `attemptEnded(idx, AttemptFailed, ReasonTargetLeft, nil, now)`; if the attempt committed and the finalize classification is a stream interruption → `attemptEnded(idx, AttemptInterrupted, ReasonStreamInterrupted, status, now)`; if committed and the client context ended → `AttemptCancelled` with `att.abortReasonOr(ReasonClientGone)`; if committed and the response completed → `attemptEnded(idx, AttemptCommitted, ReasonNone, status, now)`.
   - Inside `emitTerminal` (`:865-893`), after the existing `emitWorkload` call: `rec.finished(stateFor(state), reasonFor(...), httpStatus, now)` where `stateFor` maps the workload terminal state `completed` → `RoutingStateCompleted`, `cancelled` → `RoutingStateCancelled`, `failed` → `RoutingStateFailed`, and `reasonFor` maps the loop-exhaustion branches (`:1148-1189`, `:1699-1735`) and the finalize classification (`:946-1030`): regenerate exhausted → `ReasonRegenerateExhausted`; attempts exhausted → `ReasonAttemptsExhausted`; job deadline → `ReasonDeadline`; no candidate at `:732-779` → `ReasonNoOwner`; stream interruption → `ReasonStreamInterrupted`; client gone → `ReasonClientGone`; cancel → `att.abortReasonOr(ReasonCancel)`; `p.serveCtx.Done()` closed → `ReasonProxyShutdown`; completed → `ReasonNone`. Ledger errors (`P/accounts.go:55-72`) map one-to-one to `ReasonLedgerUnavailable, ReasonAccountFrozen, ReasonAccountCredential, ReasonAccountCooldown, ReasonAccountCeiling, ReasonDeploymentQuarantine, ReasonAccountConflict`; keep the mapping in one function `ledgerReasonCode(err error) string` in `routing_events.go` so it is testable; it is wired but unreachable today for the same 501 reason.
   - `rec.attemptAccount` is called only on the api destination branch; today it is never reached. Leave the call in the unreachable branch with the comment `// hybrid dispatch: reached once routing_listener serves inference`.

4. `proxy.go`: add the `routingSeq atomic.Int64` field and the `currentSelfNodeID()` accessor; nothing else.

5. `routing_events_test.go` (new): using `newTestProxy` (`P/failover_test.go:36`), `notificationsOf` (`P/policy_test.go:43`), `waitForCond` (`P/retry_test.go:54`) and engines listening on `127.0.0.1:0`:
   - `TestRoutingRecordCompletesWithCommittedAttempt`: one healthy recording engine; drive a chat completion; assert exactly one final `routing/request` with `state completed`, one attempt `committed`, `status 200`, `firstByteMs != null`, `served.nodeId` equals the self node, `spend.ceilingNanoUsd == null`.
   - `TestRoutingRecordRejectedThenCommitted`: first engine answers 503 (`shouldRetry`), second answers 200; assert attempts `[rejected/upstream-status status 503, committed]`.
   - `TestRoutingRecordCancelledByWorkloadCancel`: use `startCancelFixture` (`P/cancel_test.go:19-45`) with a `stallingEngine`; send `workload/cancel` with `regenerate=false`; assert the request ends `cancelled / cancel` and the open attempt reads `cancelled / cancel`.
   - `TestRoutingRecordRegenerateMarksRedispatched`: same fixture, `regenerate=true`, then let the second attempt complete; assert attempt 0 is `redispatched / regenerate`.
   - `TestRoutingRecordStreamInterruptedIsFailed`: engine streams two SSE chunks then closes the connection; assert `state failed`, `reason stream-interrupted`, attempt outcome `interrupted`.
   - `TestRoutingRecordAttemptCapDropsOldest`: unit-test `routingRecorder` directly with 40 `attemptStarted` calls; assert `len(Attempts)==32`, `DroppedAttempts==8`, first index `8`.
   - `TestRoutingRecordNeverCarriesBodies`: marshal the final record from the first test and assert the JSON does not contain the request prompt string used in the test.
   - `TestRoutingRecordValidates`: every emitted record passes `nodepolicy.DecodeRoutingRequestView`.
   - `TestLedgerReasonCode`: table over the seven ledger errors.

6. `spec.md`: add §5.6 "Routing request record" after §5.5 (`:765-806`) describing the recorder, the hook points, the closed reason set (link to `N/nodepolicy/routing_wire.go`), the cap, redaction rules and the statement "emitted to the broker only; never forwarded". Add a row to §12.1's method tables for `routing/request` (proxy → broker notification). `README.md`: add `routing/request` under the notifications list with one sentence.

7. `desktop/docs/service-contract-exceptions.json`: add to `ignoredMethods` an entry for `routing/request` with reason `"proxy to broker only; the broker re-emits it to clients as routing:changed"`, in the same shape as the existing `workload:submitted` entry.

8. From `desktop/`: `npm run service-contracts:write` to regenerate `desktop/docs/services-api.md`; commit the regenerated file.

**Verify:**
- Mac or MSI: `cd services/nvpair-proxy && go vet ./... && go test -race ./...` passes, including the nine new tests.
- PC3: `cd services/nvpair-proxy && go test ./...` passes.
- `cd desktop && npm run service-contracts:check` exits 0 (Mac).
- `grep -n "routing/request" desktop/docs/services-api.md` shows the method under the proxy section (generated, not hand-edited).
- `grep -rn "prompt\|messages" services/nvpair-proxy/routing_events.go` returns no log statement that includes a body or prompt.
- Existing `go test ./...` in `services/nvpair-proxy` shows no new skips compared with the baseline run on the same machine.

### Task 3 — Broker history, client relay, cluster-wire allowlist, and the desktop bridge for the routing stream

**Files:**
- create `services/nvpair-ui-broker/routing_events.go`
- create `services/nvpair-ui-broker/routing_events_test.go`
- modify `services/nvpair-ui-broker/broker.go`
- modify `services/nvpair-ui-broker/proxy.go`
- modify `services/nvpair-ui-broker/engineproxy.go`
- modify `services/nvpair-ui-broker/client_requests.go`
- modify `services/nvpair-ui-broker/workload_projection.go`
- modify `services/nvpair-ui-broker/workload_projection_test.go`
- modify `services/nvpair-ui-broker/README.md`
- create `desktop/src/shared/types/routing.ts`
- modify `desktop/src/shared/types/ws-channels.ts`
- create `desktop/src/electron/service-bridge/routing.ts`
- modify `desktop/src/electron/service-bridge/modular-state.ts`
- modify `desktop/src/electron/service-bridge/modular-supervisor.ts`
- modify `desktop/src/electron/service-bridge/empty-handlers.ts`
- modify `desktop/src/ui/api/pair-api.ts`
- create `desktop/tests/modular/routing-projection.test.ts`
- modify `desktop/docs/services-api.md` (regenerated only)
- modify `desktop/docs/services-backend.md`
- modify `desktop/docs/frontend-api.md`
- modify `desktop/docs/architecture.md`

**Depends on:** 1, 2

**Complexity:** judgment

**Do:**

Broker side:

1. `B/routing_events.go` (new):

```go
const (
	proxyRoutingRequest    = "routing/request"       // proxy -> broker (never forwarded)
	methodRoutingSubscribe = "routing:subscribe"
	methodRoutingUnsubscribe = "routing:unsubscribe"
	methodRoutingGetInitial = "routing:get-initial"
	notifyRoutingChanged   = "routing:changed"
)

type routingInitialResult struct {
	Requests []nodepolicy.RoutingRequestView `json:"requests"`
}

// routingHistory keeps the newest nodepolicy.RoutingHistoryLimit records,
// keyed by RoutingRequestKey, ordered by CreatedAt then RequestID.
type routingHistory struct {
	mu      sync.Mutex
	order   []nodepolicy.RoutingRequestKey
	records map[nodepolicy.RoutingRequestKey]nodepolicy.RoutingRequestView
}

func (h *routingHistory) upsert(v nodepolicy.RoutingRequestView) (evicted []nodepolicy.RoutingRequestKey)
func (h *routingHistory) snapshot() []nodepolicy.RoutingRequestView // newest first
func (h *routingHistory) failOpen(reason string, endedAt int64) []nodepolicy.RoutingRequestView // marks routing/streaming records failed; returns the changed ones
```

   `upsert` ignores a record whose `Seq` is lower than the stored one for the same key (out-of-order protection). Eviction removes the oldest by `CreatedAt`.

   Broker methods in the same file:
   - `func (b *Broker) handleProxyRoutingRequest(params json.RawMessage)`: `stamped := stampNodeID(params, b.nodeID)` (reuse `B/workload_projection.go:195-224`; it stamps a top-level `nodeId`, so instead write a sibling `stampRoutingOrigin(params, nodeID)` that sets `request.originNodeId` only when empty); decode with `nodepolicy.DecodeRoutingRequestView`; on error log the method name and error (no payload) and return; `b.routing.upsert`; if `routingSubscribed`, `b.codec.Notify(notifyRoutingChanged, view)`. **Never** calls `wm.Forward`, the scheduler fan, or any peer path.
   - `func (b *Broker) handleRoutingSubscribe / Unsubscribe / GetInitial(id json.RawMessage)` following `workloads:*` at `B/client_requests.go:247-278`; `get-initial` responds `routingInitialResult{Requests: b.routing.snapshot()}`.
   - `func (b *Broker) failRoutingForProxyExit()`: `for _, v := range b.routing.failOpen(nodepolicy.ReasonProxyExited, now)` notify each when subscribed.

2. `B/broker.go`: add fields `routing routingHistory`, `routingMu sync.Mutex`, `routingSubscribed bool` next to `workloadsSubscribed` (`:333-339`). `[I-SEAM-ACTIVE]` This flag is the reconciliation point with I Task 6's `routing:active`.

3. `B/engineproxy.go` `routeProcessScopedProxyNotification` (`:545-565`): add `case method == proxyRoutingRequest: b.handleProxyRoutingRequest(params); return true` **before** the `proxyWorkloadMethods` case.

4. `B/client_requests.go`: add `methodRoutingSubscribe`, `methodRoutingUnsubscribe`, `methodRoutingGetInitial` to `internalClientMethods` (`:24-66`) and three `case "routing:subscribe":` / `"routing:unsubscribe"` / `"routing:get-initial"` literals in `handleMessage` (`:130+`) next to the `workloads:*` cases (the contract checker needs the literal form).

5. `B/proxy.go`: in `clearProxyHandles` (`:344-349`) call `b.failRoutingForProxyExit()` after the handles are cleared.

6. `B/workload_projection.go` `routeProxyWorkload` (`:164-186`): before `stampNodeID`, pass params through a new `sanitizeWorkloadFrame(params json.RawMessage) (json.RawMessage, []string)` that unmarshals into `map[string]json.RawMessage`, keeps only the keys that correspond to the JSON tags of `Workload` (`P/proxy.go:172-211`) — list them explicitly in a `var workloadWireAllowlist = map[string]struct{}{...}` in this file — and drops the rest, returning the dropped key names; log `"workload frame dropped non-allowlisted fields"` with the key names only (no values) when non-empty. The store, scheduler fan, client echo and `wm.Forward` all receive the sanitized frame.

7. `B/routing_events_test.go` (new), using `newPolicyBroker`, `attachFakeProxy`, `syncBuffer.waitFrame`, `clientRequest`, `isResponse` (`B/policy_test.go:220-316`):
   - `TestRoutingRequestStampedAndRelayedWhenSubscribed`: client sends `routing:subscribe`; fake proxy emits `routing/request` with `originNodeId ""`; assert a `routing:changed` frame whose `request.originNodeId == "self-node"`.
   - `TestRoutingRequestNotRelayedWhenUnsubscribed`: same without subscribe; assert no `routing:changed` frame within 200 ms; then `routing:get-initial` returns the record.
   - `TestRoutingRequestNeverForwardedToWorkloadManager`: install a pipe workload-manager (pattern from `B/workloadcancel_test.go`); emit `routing/request`; assert nothing is written to the manager's stdin within 200 ms.
   - `TestRoutingHistoryEvictsOldest`: 257 upserts → 256 kept, oldest key gone.
   - `TestRoutingHistoryIgnoresStaleSeq`.
   - `TestRoutingFailOpenOnProxyExit`: an open record becomes `failed / proxy-exited` and is pushed.
   - `TestRoutingRequestInvalidPayloadDropped`: unknown field → not stored, no push.
   - In `workload_projection_test.go`: `TestRouteProxyWorkloadStripsUnknownFields`: emit a `workload:started` frame with extra keys `accountId`, `attempts`, `spend`; assert the frame forwarded to the manager pipe and the `workloads:upsert` echo contain only allowlisted keys.

8. `B/README.md`: under the JSON-RPC surface (`:657+`) add `routing:subscribe`, `routing:unsubscribe`, `routing:get-initial` (request rows near `workloads:*` at `:783+`), `routing:changed` (notification row near `:717-742`), and a "Private to this node" sentence stating the broker never forwards routing records.

Desktop side (same commit — the contract checker ties them):

9. `D/shared/types/routing.ts` (new): `export type RoutingState = 'routing' | 'streaming' | 'completed' | 'failed' | 'cancelled'`, `export type RoutingAttemptOutcome = 'pending' | 'committed' | 'rejected' | 'failed' | 'redispatched' | 'interrupted' | 'cancelled'`, `export type RoutingDestinationKind = 'local' | 'api'`, `export type RoutingAccountState = 'ok' | 'cooldown' | 'frozen' | 'credential-failure' | 'exhausted' | 'probe' | 'conflict'`, `export type RoutingReason = string` (closed set documented in a comment; unknown codes render as "unknown"), and interfaces `RoutingDestination`, `RoutingWaits`, `RoutingAccountStateView`, `RoutingAttempt`, `RoutingSpend`, `RoutingWorkloadRef`, `RoutingRequested`, `RoutingRequest` mirroring Task 1 field for field, money as `NanoUsd` imported from `./node-policy`, `null` where Go has pointers, plus `export const ROUTING_HISTORY_LIMIT = 256` and `ROUTING_ATTEMPT_LIMIT = 32`.

10. `ws-channels.ts`: invoke map adds `'routing:subscribe': { params: undefined; result: undefined }`, `'routing:unsubscribe'` likewise, `'routing:get-initial': { params: undefined; result: { requests: RoutingRequest[] } }` next to `workloads:*` (`:153-165`); push map adds `'routing:changed': RoutingRequest` (`:194-205`).

11. `D/electron/service-bridge/routing.ts` (new): `parseRoutingRequest(value: unknown-free input)` using the existing `objectValue` / `stringValue` / `numberValue` helpers from `./node-projection` (same style as `workload-projection.ts`; no casts); returns `RoutingRequest | null`, rejecting a record whose `schema !== 1`, whose money strings fail `/^(0|[1-9][0-9]*)$/`, or whose state/outcome/kind is outside the unions; `parseRoutingInitial(result)` returns `RoutingRequest[]`; `routingKey(r) = \`${r.request.originNodeId}\u0000${r.request.proxyRunId}\u0000${r.request.requestId}\``; `isTerminalRouting(r)`.

12. `modular-state.ts`: add `routing: Map<string, RoutingRequest>` with `getRouting()`, `seedRouting(list)`, `upsertRouting(r)` bounded to 256 by `createdAt`; mirror the workload functions at `:697-792`.

13. `modular-supervisor.ts`: after `workloads:subscribe` / `seedWorkloadBaseline` (`:1078-1080`) add `await subscribe('routing:subscribe', 'subscribe to broker routing stream')` and `await this.seedRoutingBaseline()` (calls `routing:get-initial`, `seedRouting`, emits one `routing:changed` per record through the push bus); in the notification switch (`:1484-1528`) add `case 'routing:changed'` → parse, `upsertRouting`, `emitBridgePush('routing:changed', parsed)`.

14. `empty-handlers.ts`: add `handleRoutingGetInitial` next to `handleWorkloadsGetInitial` (`:489-500`) returning `{ requests: getRouting() }`, and register `'routing:get-initial'`, `'routing:subscribe'`, `'routing:unsubscribe'` in the handler map (`:782-825`).

15. `pair-api.ts`: add `IRoutingApi { subscribe(): Promise<void>; unsubscribe(): Promise<void>; getInitial(): Promise<{ requests: RoutingRequest[] }>; onChanged(cb: (r: RoutingRequest) => void): () => void }` to `IPairApi` (`:153-164`) as `routing`, implemented in `createPairApi` (`:170-254`) in the same way as `workloads`.

16. `desktop/tests/modular/routing-projection.test.ts` (new): reads `../services/shared/nodepolicy/testdata/routing-request.json` via `resolve(process.cwd(), ...)` exactly like `node-policy.test.ts:46-96`; asserts `parseRoutingRequest` accepts all three corpus records and returns values equal to the Go assertions in Task 1 step 6 (state, reasons, attempt counts, `chargedNanoUsd === '123456789'`, `droppedAttempts === 2`); rejects a copy with `schema: 2`, a copy with `chargedNanoUsd: 1.5`, a copy with an unknown `state`; `routingKey` is stable; state `upsertRouting` keeps 256 newest.

17. Docs: `desktop/docs/services-backend.md:128-143` add the `routing:changed` row and the three request rows; `desktop/docs/frontend-api.md` add `pairApi.routing` to the tables at `:141/:179/:194`; `desktop/docs/architecture.md` add one paragraph "Routing diagnostics are origin-private: the proxy reports to its own broker, which never forwards them". Then `cd desktop && npm run service-contracts:write`.

**Verify:**
- Mac or MSI: `cd services/nvpair-ui-broker && go vet ./... && go test -race ./...` passes including the eight new broker tests.
- PC3: `cd services/nvpair-ui-broker && go test ./...` passes.
- Mac: `cd desktop && npm run lint && npm run typecheck && npm run test:unit && npm run dead-code:check && npm run service-contracts:check` all exit 0; `test:unit` lists `routing-projection.test.ts` as passed.
- `git diff --name-only` equals the Files list.
- `grep -n "routing:changed\|routing:get-initial" desktop/docs/services-api.md` shows both under the broker section.

### Task 4 — Terminal interface decoder for the frozen routing view

**Files:**
- create `services/nvpair-tui/ui/routingwire.go`
- create `services/nvpair-tui/ui/routingwire_test.go`

**Depends on:** 1

**Complexity:** mechanical

**Do:**

1. `routingwire.go` (new, `package ui`, SPDX header): import `nvpair-shared/nodepolicy` as `T/policywire.go` already does. Define:

```go
const (
	methodRoutingSubscribe   = "routing:subscribe"
	methodRoutingUnsubscribe = "routing:unsubscribe"
	methodRoutingGetInitial  = "routing:get-initial"
	notifyRoutingChanged     = "routing:changed"
)

type routingInitial struct {
	Requests []nodepolicy.RoutingRequestView `json:"requests"`
}

func decodeRoutingRequest(raw json.RawMessage) (nodepolicy.RoutingRequestView, bool) // DecodeRoutingRequestView + decodeOrLog style logging
func decodeRoutingInitial(raw json.RawMessage) ([]nodepolicy.RoutingRequestView, bool)
func routingKey(v nodepolicy.RoutingRequestView) string // origin + "\x00" + run + "\x00" + id
```

   Also the pure label helpers that J2 and the corpus test share (no Bubble Tea imports):
   - `routingStateLabel(state string) string`: `routing`→`Routing`, `streaming`→`Streaming`, `completed`→`Completed`, `failed`→`Failed`, `cancelled`→`Cancelled`, else `Unknown`.
   - `attemptOutcomeLabel(outcome, reason string, continuation bool) string`: `committed`→`Served`; `rejected`→`Rejected`; `failed`→`Failed`; `redispatched`→`Re-dispatched`; `cancelled`→`Cancelled`; `pending`→`In progress`; `interrupted`→ `Interrupted, resumed` when `continuation` is true, else `Interrupted`; else `Unknown`.
   - `reasonLabel(reason string) string`: table in Task 6 step 2 (same strings, keep them identical).
   - `accountStateLabel(state string) string`: `ok`→`OK`, `cooldown`→`Cooling down`, `frozen`→`Frozen`, `credential-failure`→`Credential failed`, `exhausted`→`Exhausted`, `probe`→`Probing`, `conflict`→`Conflict`, else `Unknown`.
   - `formatNanoUsd(v *nodepolicy.NanoUSD) string`: `nil`→`—`; otherwise `"$" + dollars + "." + nine fractional digits trimmed to at least two` by string arithmetic on the digits: `"123456789"`→`$0.123456789`, `"1000000000"`→`$1.00`, `"250000000"`→`$0.25`, `"0"`→`$0.00`.
   - `formatWait(ms *int64) string`: `nil`→`—`; `< 1000`→`"350 ms"`; otherwise one decimal seconds `"1.2 s"`.

2. `routingwire_test.go`: `TestRoutingCorpusDecodes` reads `../../shared/nodepolicy/testdata/routing-request.json` relative to the package directory (use `filepath.Join("..", "..", "shared", "nodepolicy", "testdata", "routing-request.json")`), decodes all three records and asserts the same values as Task 1 step 6; `TestRoutingLabels` table-tests every helper above with the exact strings listed, including `formatNanoUsd` on the four examples and `formatWait` on `350` and `1234`.

**Verify:**
- Mac or MSI: `cd services/nvpair-tui && go vet ./... && go test -race ./ui/ -run 'TestRouting'` passes; full `go test -race ./...` still passes.
- PC3: `cd services/nvpair-tui && go test ./...` passes.
- `go vet` reports no unused symbol warnings; the helpers are used by Task 7 later, so until then they are referenced only by the test, which Go permits.

### Task 5 — Cross-process proof that the cluster workload wire carries no private routing fields

**Files:**
- modify `services/tests/workload_interop_test.go`

**Depends on:** 2, 3

**Complexity:** mechanical

**Do:**

1. Next to `TestWorkloadManagerOutboundBroadcast` (`X/workload_interop_test.go:558`) add `TestWorkloadBroadcastCarriesNoRoutingFields` using the same `newServiceFixture` and capture path. Drive the real proxy through one completed inference request against an `httptest.NewServer` (it listens on `127.0.0.1:0`) that answers `POST /v1/chat/completions` with a minimal non-streaming chat completion, registered with the fixture's proxy the same way `TestWorkloadManagerOutboundBroadcast` makes the proxy run a workload. Capture every outbound `workload:*` frame the manager broadcasts.
2. For each captured frame, unmarshal `params` into `map[string]json.RawMessage` and assert: no key outside the allowlist in `B/workload_projection.go` (copy the key list into the test as a literal so a drift in either place fails); and the raw frame bytes do not contain any of the substrings `"accountId"`, `"deploymentId"`, `"credential"`, `"attempts"`, `"spend"`, `"reason"`, `"NanoUsd"`, `"routing/request"`.
3. Also assert that no frame with method `routing/request` or `routing:changed` was broadcast by the manager.

**Verify:**
- PC3 only (services/tests never run on Mac): `cd services/tests && go test ./... -run 'TestWorkloadBroadcastCarriesNoRoutingFields|TestWorkloadManagerOutboundBroadcast' -v` passes without a skip line for the new test; the full `go test ./...` in `services/tests` shows no new failures and no new skips compared with the baseline.
- MSI may also run it (Windows gates), never with `-race`.

### Task 6 — Desktop routing history (J1): store, labels, history and attempt components, settings mount, workload card summary

**Files:**
- create `desktop/src/ui/stores/routing.store.ts`
- modify `desktop/src/ui/stores/init.ts`
- create `desktop/src/ui/utils/routing-labels.ts`
- create `desktop/src/ui/components/RoutingSettings/RoutingHistory.tsx`
- create `desktop/src/ui/components/RoutingSettings/RoutingAttempt.tsx`
- modify `desktop/src/ui/components/RoutingSettings/RoutingSettings.tsx`
- modify `desktop/src/ui/components/Workloads/WorkloadItemCard.tsx`
- create `desktop/tests/modular/routing-view.test.ts`

**Depends on:** 3

**Complexity:** judgment

**Do:**

1. `routing.store.ts`: copy the shape of `workloads.store.ts` (zustand, rAF-batched `pendingOps`, `initialize()` subscribes via `pairApi.routing.onChanged` **before** `pairApi.routing.getInitial()`, `refresh()`, `cleanup()`), state `requests: Map<string, RoutingRequest>` keyed by `routingRequestKey` from `@/ui/utils/routing-labels` (defined in step 2 with the same formula as the bridge's `routingKey`; the renderer must not import from `src/electron/`). Keep newest 256 by `createdAt`. Selectors: `selectRoutingNewestFirst`, `selectRoutingForWorkload(engine, id)`. Register `initialize` in `initializeAllStores`, `cleanup` in `cleanupAllStores`, `refresh` in `resyncAllStores` in `init.ts`.


2. `routing-labels.ts` (pure, tested): `routingRequestKey(r: RoutingRequest): string` returning `` `${r.request.originNodeId}\u0000${r.request.proxyRunId}\u0000${r.request.requestId}` `` (same formula as the bridge's `routingKey`; duplicated on purpose so the renderer never imports from `src/electron/`), then the exact copy tables, identical to Task 4:
   - `routingStateLabel`: `Routing`, `Streaming`, `Completed`, `Failed`, `Cancelled`, `Unknown`.
   - `attemptOutcomeLabel(outcome, continuation: boolean)`: `Served`, `Rejected`, `Failed`, `Re-dispatched`, `Cancelled`, `In progress`, interrupted → `Interrupted, resumed` when continuation else `Interrupted`, `Unknown`.
   - `reasonLabel(reason)`: `client-gone`→`Client disconnected`; `cancel`→`Cancelled by user`; `regenerate`→`Regenerated`; `node-paused`→`Node paused`; `proxy-shutdown`→`Proxy shutting down`; `proxy-exited`→`Proxy exited`; `target-left`→`Node left the cluster`; `transport`→`Connection error`; `upstream-status`→`Engine returned an error status`; `no-content`→`No output before timeout`; `stream-interrupted`→`Stream interrupted`; `upstream-error`→`Engine error`; `admission-paused`→`Node paused`; `admission-draining`→`Node draining`; `admission-busy`→`Node busy`; `admission-no-fit`→`Model does not fit`; `admission-engine-off`→`Engine off`; `admission-wake-timeout`→`Engine wake timed out`; `engine-not-running`→`Engine not running`; `request-invalid`→`Request could not be adapted`; `no-owner`→`No node serves this model`; `regenerate-exhausted`→`Too many regenerations`; `attempts-exhausted`→`All attempts used`; `deadline`→`Request deadline reached`; `request-ceiling`→`Request spend ceiling reached`; `account-ceiling`→`Account ceiling reached`; `account-cooldown`→`Account cooling down`; `account-frozen`→`Account frozen`; `account-credential`→`Account credential failed`; `account-exhausted`→`Account exhausted`; `account-conflict`→`Account conflict`; `ledger-unavailable`→`Spend ledger unavailable`; `deployment-quarantined`→`Deployment quarantined`; `peer-framing`→`Peer sent a malformed frame`; `''`→`''`; anything else→`Unknown`.
   - `accountStateLabel`: as Task 4.
   - `formatNanoUsd(v: NanoUsd | null)`: same string arithmetic and the same four examples as Task 4; no `Number()` / `parseFloat`.
   - `formatWait(ms: number | null)`: `—`, `350 ms`, `1.2 s`.
   - `requestSummaryLine(r: RoutingRequest): string`: `completed` → `Routing: ${n} attempt(s), served on ${served.nodeId ?? served.deploymentId}`; `failed` → `Routing failed: ${reasonLabel}`; `cancelled` → `Routing cancelled: ${reasonLabel}`; `routing` → `Routing: attempt ${n}`; `streaming` → when the last attempt is `interrupted` and `continuationOf` is null on every later attempt → `Streaming paused, waiting to resume`, else `Streaming on ${nodeId}`.
   - `requestIsTruthfullyFailed(r)`: `state === 'failed'` — used so the card shows an error only for `failed`; an `interrupted` attempt inside a `streaming` request is never an error (owner decision).

3. `RoutingAttempt.tsx`: props `{ attempt: RoutingAttempt; continuation: boolean }`; renders one KUI `Flex` row: `Text` `#${index}`, destination (`local` → `${nodeId} · ${engine} · ${model}`; `api` → `${deploymentId} · ${accountId} · ${model}`), `Badge` with `attemptOutcomeLabel`, `Text` with `reasonLabel` when non-empty, `Text` `Admission ${formatWait(admissionMs)} · First byte ${formatWait(firstByteMs)}`, and when `account` is non-null a `Tag` `${accountStateLabel} · held ${formatNanoUsd(held)} · charged ${formatNanoUsd(charged)}`. When `attempt.status` is non-null append `· HTTP ${status}`.

4. `RoutingHistory.tsx`: reads `useRoutingStore` with `selectRoutingNewestFirst`; renders inside a `settings-card` section titled `Routing history` with the subtitle `Private to this machine. Newest 256 requests.`; a KUI `Table` with columns `When` (relative age like the workload list), `Model`, `State` (`Badge`, `routingStateLabel`), `Attempts` (`${attempts.length}` plus ` (+${droppedAttempts} older)` when non-zero), `Served` (destination or `—`), `Spend` (`formatNanoUsd(spend.chargedNanoUsd)` and, when `ceilingNanoUsd` is non-null, ` of ${formatNanoUsd(ceiling)} ceiling`), `Reason` (`reasonLabel`); a `Button` `Show attempts` / `Hide attempts` per row toggling a nested `Stack` of `RoutingAttempt` rows, with `continuation` computed as "some later attempt has `continuationOf === this.index`". Empty state: `Text` `No routed requests yet.`. No new CSS files; reuse `settings-card` classes already in `RoutingSettings.tsx`.

5. `RoutingSettings.tsx`: mount `<RoutingHistory />` as the last `settings-card` section (after `BulkNodeConfiguration`). No other change.

6. `WorkloadItemCard.tsx`: select `selectRoutingForWorkload(workload.engine, workload.id)`; when a record exists render one `Text` line with `requestSummaryLine(record)` under the existing status line. Keep the existing rule that error text appears only when the workload is `failed`; do not add error styling for `streaming` records.

7. `desktop/tests/modular/routing-view.test.ts`: pure tests for every function in `routing-labels.ts` with the exact strings above (including all four money examples and the interrupted rules); a jsdom render test modelled on `node-policy.test.ts:860-912` that seeds `useRoutingStore` with the three corpus records (read from `../services/shared/nodepolicy/testdata/routing-request.json`) and asserts `RoutingHistory` renders three rows, the text `$0.123456789`, the text `Stream interrupted`, and `(+2 older)`; a store test with a typed fake `pairApi.routing` asserting subscribe-before-getInitial order and the 256 cap.

**Verify:**
- Mac: `cd desktop && npm run lint && npm run typecheck && npm run test:unit && npm run dead-code:check` exit 0; `routing-view.test.ts` passes.
- `grep -rn "from '@/electron\|from '../../electron" desktop/src/ui/stores/routing.store.ts desktop/src/ui/utils/routing-labels.ts desktop/src/ui/components/RoutingSettings/RoutingHistory.tsx desktop/src/ui/components/RoutingSettings/RoutingAttempt.tsx` returns nothing.
- `grep -n "parseFloat\|Number(" desktop/src/ui/utils/routing-labels.ts` returns nothing.
- `grep -rn "as [A-Z]\|: any\|unknown" desktop/src/ui/utils/routing-labels.ts desktop/src/ui/stores/routing.store.ts` returns nothing.
- Manual (Mac, `npm start` against MSI or PC3 as the engine node): Settings → Routing shows the history after one OpenCode chat request; the workload card shows `Routing: 1 attempt, served on <node>`.

### Task 7 — Terminal interface Routing tab (J2)

**Files:**
- create `services/nvpair-tui/ui/routingview.go`
- create `services/nvpair-tui/ui/routingview_test.go`
- modify `services/nvpair-tui/ui/ui.go`
- modify `services/nvpair-tui/README.md`
- modify `docs/terminal-interface.mdx`

**Depends on:** 3, 4

**Complexity:** judgment

**Do:**

1. `routingview.go`: `routingView` implementing `View` (`T/view.go:21-38`), modelled on `jobsView` in `T/jobs.go`:
   - `Title()` returns `"Routing"`.
   - `Init()` returns `feedCmd` = `call(client, methodRoutingSubscribe, nil)` then `call(client, methodRoutingGetInitial, nil)` decoded with `decodeRoutingInitial`; also `nodeIdentityCmd` so `nodeNamer` can name self.
   - `Update`: `DisconnectedMsg` clears rows and shows the footer `Disconnected`; `sessionReadyMsg` re-runs `feedCmd`; `NotificationMsg` with method `notifyRoutingChanged` → `decodeRoutingRequest`, upsert into `map[string]nodepolicy.RoutingRequestView` keyed by `routingKey`, trim to `nodepolicy.RoutingHistoryLimit` by `CreatedAt`; `discovery:nodes-changed` and `nodes:changed` feed the `nodeNamer` exactly as `jobs.go` does; `TickMsg` refreshes ages; `enter` toggles the attempt detail of the selected row; `r` re-runs `feedCmd`.
   - `View`: a table via `newTable` / `fitTable` (`T/table.go:136-200`) with columns `Age` (`ageLabel`, fixed 6), `Model` (flex), `State` (fixed 10, `routingStateLabel`), `Att` (fixed 5, rendered as `"<len>"` when `DroppedAttempts == 0`, else `"<len>+<dropped>"`, for example `1+2`), `Served` (flex, `nodeNamer.name` for local, `deploymentId` for api), `Spend` (fixed 14, `formatNanoUsd(ChargedNanoUsd)`), `Reason` (flex, `reasonLabel`). When the selected row is expanded, render below the table one line per attempt: `#<index>  <destination>  <attemptOutcomeLabel>  <reasonLabel>  adm <formatWait>  ttfb <formatWait>  [<accountStateLabel> held <money> charged <money>]  [HTTP <status>]`, truncated to the width with the existing `joinLines` helper. Footer uses `footerStyle` with `Private to this machine · newest 256 requests`.
   - `Help()` returns `"enter attempts · r refresh"` in the existing help format.
   - Continuation rule: an `interrupted` attempt renders `Interrupted, resumed` when a later attempt's `ContinuationOf` equals its index, else `Interrupted`; a `streaming` request whose last attempt is `interrupted` shows state `Streaming (paused)`.

2. `ui.go` `defaultViews` (`:67-75`): insert `newRoutingView(client)` after `newJobsView(client)`; confirm the tab count stays `<= maxTabs` (`T/keys.go:119`, 9).

3. `routingview_test.go`: using the pattern from `T/model_test.go:37` (`newReconnectUIFixture`) and the corpus from Task 4: `TestRoutingViewRendersCorpus` feeds the three corpus records as `NotificationMsg{Method: "routing:changed"}` and asserts the rendered `View()` at width 120 contains `Completed`, `Failed`, `Cancelled`, `$0.123456789`, `Stream interrupted`, and `1+2` in the `Att` column; `TestRoutingViewExpandShowsAttempts` sends `enter` and asserts `#0` and `Rejected` and `Node busy` appear; `TestRoutingViewTrimsHistory` feeds 257 records and asserts 256 rows; `TestRoutingViewDisconnectClears`.

4. `services/nvpair-tui/README.md:36-37` tab table: add the `Routing` row: "Routing — private routing history for this node: attempts, reasons, waits, account state and spend; enter expands a request". `docs/terminal-interface.mdx`: add a `### Routing` subsection after the Jobs subsection (`:390-400`) with the same sentence, the key help, and the statement that the data never leaves the node.

**Verify:**
- Mac or MSI: `cd services/nvpair-tui && go vet ./... && go test -race ./...` passes including the four new tests.
- PC3: `cd services/nvpair-tui && go test ./...` passes.
- Manual (PC3 or MSI): run `nvpair-tui` against a running broker, press the `Routing` tab, send one OpenCode request, see one row appear and `enter` expand it.
- `node scripts/spdx-headers.mjs` from the repository root reports nothing missing.

## Verification (whole-plan)

Run after Task 7 on a branch that contains all seven commits:

- Mac: `make check` passes; `cd desktop && npm run dead-code:check` passes; `cd desktop && npm run service-contracts:check` passes; `go test -race ./...` in `services/shared`, `services/nvpair-proxy`, `services/nvpair-ui-broker`, `services/nvpair-tui` passes.
- PC3: `services/build.bat` succeeds; `go test ./...` in each of those modules and in `services/tests` passes; `TestWorkloadBroadcastCarriesNoRoutingFields` ran (no skip).
- MSI: `go test -race ./...` in `services/nvpair-proxy` and `services/nvpair-ui-broker` passes with llama.cpp engines running; the desktop app shows the Routing history after one OpenCode request to a Strata or llama.cpp model and the Jobs card shows the summary line.
- Privacy sweep: `grep -rn "routing/request\|routing:changed" services/nvpair-workload-manager services/nvpair-cluster-manager 2>/dev/null` returns nothing; `grep -rn "NanoUsd" services/nvpair-workload-manager` returns nothing.
- The pull request's `pair-release-intent:v1` block declares bumps for `nvpair-proxy`, `nvpair-ui-broker`, `nvpair-tui` and a user-facing line "Routing history (desktop Settings → Routing, terminal Routing tab)".

## Do not

- Do not add any field to the proxy `Workload` struct or to any `workload:*` payload.
- Do not forward `routing/request` or `routing:changed` to the workload-manager, the scheduler, cluster workers or any peer.
- Do not write floats, `number` money or `parseFloat` anywhere money is handled.
- Do not log, store or test with prompts, messages, response bodies, PINs, credentials or key material; reasons are codes only.
- Do not change retry, admission, ledger or candidate-selection behaviour; this plan observes, it does not decide.
- Do not hand-edit `desktop/docs/services-api.md`, `CHANGELOG.md` or `services/versions.json`.
- Do not use TS casts, `any`, `unknown` in signatures, dynamic imports, or renderer imports from `src/electron/`.
- Do not reuse or rename I Task 6's `routing:active`, `routing:cancel`, `routing/active`.
- Do not touch `T/rpccmd.go`, `N/nodepolicy/routing.go`, `N/nodepolicy/wire.go`, or any `W/` file.
- Do not run `services/tests` on Mac, or `-race` on PC3.
- Do not add a cancel control to the routing history UI.

## If stuck

- Contract checker fails with `missingNotifications` for `routing:changed`: the desktop reference must be in `D/electron/service-bridge/` or `ws-channels.ts` in the same commit as the broker emitter; check the literal is a string constant the regex can see (`notify("routing:changed"` or a local `const notifyRoutingChanged = "routing:changed"` in a non-test broker file).
- `dead-code:check` fails on a new export: first remove `export` when the symbol is used in one file; when it is consumed elsewhere add the import where the plan says it is consumed; when neither applies, add a justified entry to `desktop/dead-code-omissions.json` and say so in the commit message.
- A hook point line in `http_handler.go` has moved because the I plan landed first: locate it by the surrounding symbol named in Context (for example `repointWorkload`, `emitTerminal`, `claim.abandoned()`), not by line; if the I plan replaced `attemptCtl`, implement the `routingAttemptRecorder` calls on I's attempt object and report the seam you reconciled.
- `TestWorkloadBroadcastCarriesNoRoutingFields` skips on PC3: read the skip reason; if it is a firewall or port serialization issue, rerun serialized as the fleet notes say; if the fixture cannot start the proxy, stop and report.
- Retry a flaky network-dependent test twice; if a test fails three times for the same reason, stop and report the test name, the machine and the output, without weakening the assertion.
- Stop and report if any step requires changing the `Workload` struct, forwarding routing data to a peer, or a float in money.

## Owner questions

1. **Persistence.** J0 keeps routing history in broker memory only (newest 256). A broker restart clears it. Do you want it persisted on disk (which creates a new local file containing model names, node IDs and spend figures), or is memory-only acceptable for P5.1?
2. **Identifiers in the private record.** The local record shows the account ID and deployment ID (your own policy identifiers, never credentials) so you can see which account an attempt used. Confirm these may appear in the desktop and terminal views on the origin machine, or ask for them to be redacted to a hash.
3. **Caps.** 256 requests per broker and 32 attempts per request, oldest dropped. Confirm or give other numbers.
4. **Never-forwarded guarantee.** This plan makes the broker strip non-allowlisted fields from every workload frame before it leaves the node (Task 3 step 6) and adds a cross-process test. Stripping is silent apart from a log line naming the dropped keys. Confirm silent stripping is wanted rather than failing the frame.
5. **Interruption wording.** With SC-a only (no continuation today), an interruption after output began shows as `Failed / Stream interrupted` on the request and `Interrupted` on the attempt. Confirm this is the truthful wording you want until SC-b lands, when it becomes `Streaming paused, waiting to resume` and `Interrupted, resumed`.
