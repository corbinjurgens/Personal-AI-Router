<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Plan: retire the workloads of a departed proxy run (slug: plan-departed-run)

Kind: execution plan for a confirmed defect, with a short re-verification of the defect. Investigation only: no product change, no pull request.

Input: RED-1 of `plans/reviews/debate-i-g.md` (branch `plan/inv-debate-i-g`): after the proxy process crashes or is replaced, the local-origin inference workloads it registered stay "running" (or "queued") in the broker's workload view and cannot be cancelled. Queued on the broker lane after SR 5c.

## Verdict

`RED 1 UNVERIFIED 2 LOW 2`

The defect is real at the inspected head and is a lifecycle hole, not a display bug: the only producer of a local-origin terminal workload event is the request handler inside the proxy process, so a process that dies without draining leaves every open record it owned without any terminalizer, and every cancel path then refuses it because its run is not the current run. The owner of the repair is the broker (the dead proxy's parent and the authoritative workload store), through the same `workload:errored` lifecycle frame the proxy would have emitted, applied as an authoritative event and forwarded to the workload-manager so peers converge too. No JSON-RPC payload changes; desktop and TUI need no edit. Five tasks below; the process-level test is described, not run.

## Inspection base and environment

- `origin/fork/groundwork` = `7e270f10ced402325dc9e1efece54991e9bbbc08`, the SHA the common brief names; the branch had not moved. Every `path:line` below is at `7e270f1`. Branch `plan/inv-plan-departed-run` was created from it and holds only this report.
- Unlanded branches inspected for interaction only (no checkout): `origin/roadmap/sr-5c` (SR 5c) and `origin/roadmap/i-public-lifecycle` (I 4b), through `git diff --stat` and targeted `git diff` of the broker and proxy files this plan touches or depends on.
- Path shorthand from the common brief: B/ = `services/nvpair-ui-broker/`, P/ = `services/nvpair-proxy/`, N/ = `services/shared/`, X/ = `services/tests/`, T/ = `services/nvpair-tui/ui/`, D/ = `desktop/src/`, WM/ = `services/nvpair-workload-manager/`.
- Checks run here (host Go 1.24.7 with `GOTOOLCHAIN=auto` resolving the modules' Go 1.25): `go vet ./...` in B/ (exit 0), `go test -count=1 ./workloadstore` in B/ (ok), `go vet ./...` and `go test -list . ./...` in X/ (exit 0; the package's `TestMain` built the product binaries to list, nothing was executed, no listener or engine was started). No desktop check was run. No `services/tests` case was run.
- VERIFIED means the code was read or a check ran; INFERRED means a consequence derived from verified code without execution.

## Part 1: the defect, re-verified at 7e270f1

### 1.1 How the proxy registers a workload (VERIFIED)

- A request gets its own cancellable context, a `workloadCtl`, and a registry entry under the process's pinned origin/run before any inference effect: `P/http_handler.go:514-541` (`registerRoutingRequest`, `defer unregisterRoutingRequest`). The registry requires identity (`P/request_registry.go:25-45`, `routingIdentityLocked` at `:28`), which `applySetPolicy` installs when the broker's `node/set-policy` is accepted (`P/policy.go:136-154`).
- The `Workload` record is minted with `RunID: p.runID` and state `queued`, linked into the registry's workload index, and emitted as `workload:submitted` (`P/http_handler.go:896-924`; the struct is `P/proxy.go:178-209`; the run nonce is 16 random bytes, `P/routing_identity.go:13-18`). `running` is emitted on the first committed byte (`P/http_handler.go:1732-1745`).
- Every terminal event is produced by that handler alone: `emitTerminal` under `terminalOnce` (`P/http_handler.go:937-964`), reached from the disconnect watcher (`:976-987`) and from the finalize defer's classification (`:1063-1095`). `emitWorkload` is a plain `codec.Notify` on the broker pipe (`P/proxy.go:580-584`).

### 1.2 How the broker tracks it (VERIFIED)

- Proxy notifications are forwarded only from the current process handle (`B/proxy.go:199-208`) and the workload ones are routed by `routeProxyWorkload`: projection, origin stamping, authoritative store apply, scheduler fan-out, client echo, then forward to the workload-manager (`B/workload_projection.go:334-362`; accepted methods `:320-326`; the authoritative/inferred apply funnel `:477-500`).
- The store key is `(origin, engine, runId, id)` (`B/workloadstore/store.go:47-52`); `Record` projects origin, id, state, times, placement, terminal and `Inferred`, but not `RunID`, `Engine` or `Seq` (`:58-94`). Active records are never restored from disk (`B/workloadstore/persistence.go:15-19`).
- The scheduler counts queued and running work by `scheduledOn` (`services/nvpair-job-scheduler/schedule.go:105-122`), fed from the same accepted stream (`B/workload_projection.go:557-568`).
- The workload-manager keeps every local-origin lifecycle frame it is handed as its re-sync set (`WM/manager.go:359-371`, `:443-460`) and re-asserts the non-terminal ones to every peer every 30 s (`:42-43`, `:510-536`). A restarted manager is rehydrated from the broker's active local-origin records (`B/workload_projection.go:226-263`, called at `B/worker_lifecycle.go:446-460`).

### 1.3 What happens when the proxy process exits

Graceful stop (VERIFIED): broker stdin EOF ends `readLoop` (`P/proxy.go:1365-1376`); `Run` cancels the serve context and shuts the facades down within three seconds (`:457-492`, `:566-575`), each handler's `r.Context()` fires and `emitTerminal("cancelled", ...)` runs (`P/http_handler.go:976-987`, `:1063-1078`). The broker keeps forwarding during the stop because `stopProxySupervisorLocked` retires the policy incarnation, waits in `sup.Stop()`, and clears the process handle only afterwards (`B/proxy.go:456-470`; the handle compare is `:199-203`).

Crash (VERIFIED absence): `SIGKILL`, an OOM kill, an external kill, or a panic outside `handleMessage`'s recover (`P/proxy.go:1385-1400`) emits nothing. The supervisor's monitor sees `Done()` close (`B/supervisor.go:334-356`, `:243-266`), calls `onCrash`, which only clears the per-engine handles and reports the crash (`B/proxyport.go:157-171`, `B/broker_errors.go:71-80`), then backs off and respawns (`B/supervisor.go:272-302`). `spawnProxy` publishes a fresh handle and replays the policy (`B/worker_lifecycle.go:303-360`); `replayPolicyToProxy` resets the incarnation and acknowledged run (`B/policy.go:1268-1300`) and the new run's first ack is recorded at `B/policy.go:1450-1495` (`:1473-1483`). None of these paths touches the workload store. The only inferred-terminal producers are `failWorkloadsForNode`, fired when a node leaves discovery (`B/workload_projection.go:652-690`, wired at `B/worker_lifecycle.go:32`), and `failStaleForeignWorkloads`, which skips local-origin records by design (`B/workload_projection.go:740-767`; `B/workloadstore/store.go:455-457`). A search for any local-origin terminalizer (`proxy-exited`, `ActiveForNode(b.nodeID`, a departure hook on `proxyProcess.Done`) finds none; the one exit hook that exists withdraws strict readiness only (`B/readiness.go:111-123`).

Deliberate replacement (VERIFIED): a credential-set change (`B/proxy.go:504-512`), a listener change (`:575-604`) and shutdown (`:515-525`) all go through `stopProxySupervisorLocked`. The drain is bounded by the proxy's three-second shutdown context; a request whose handler does not unwind in time, or whose terminal frame is lost on the closing pipe, is left open exactly like a crash.

### 1.4 Why the records can then never be cancelled (VERIFIED)

- The replacement proxy mints a new run (`P/routing_identity.go:13-18`). `workloadCancelControl` requires `target.RunID == p.runID` (`P/request_registry.go:108-118`), so `workload/cancel` answers `found:false` (`P/cancel.go:170-179`) and the broker answers `{ok:false}` (`B/workloadcancel.go:47-77`, `:80-103`). A peer-relayed cancel takes the same path (`:108-136`; `WM/cancel.go:66-100`). While no proxy runs, `policyProxy()` is nil because the handle reports exited (`B/proxy.go:335-348`) and the client gets `-32000 "proxy not available"` instead.
- Nothing else retires the records: the stale sweep skips local origin (above); the broker re-asserts them as active to a restarted workload-manager (`B/workload_projection.go:249-263`), whose heartbeat keeps telling every peer they are running (`WM/manager.go:510-536`), so peers' silence sweep never fires either.
- A whole-broker restart is clean (active records are not restored, `B/workloadstore/persistence.go:15-19`). The hole is exactly the supervised proxy crash/replacement case.

### 1.5 Remote-origin cases

- Case A, this node is the destination (a peer forwarded a request here over strict ingress, then this proxy dies). VERIFIED: the destination registers no workload for ingress traffic (no `registerRoutingRequest`, `registerWorkload` or `Workload{` in `P/ingress.go`); the record lives at the origin, whose proxy observes the broken reverse connection and ends the request through its own finalize path (`P/http_handler.go:997-1011`, `:1063-1095`). INFERRED: the record converges when the origin's terminal reaches this node; if that frame is lost, the remote-origin backstop retires it after five minutes of origin silence (`B/workload_projection.go:705-767`). This case is already honest and needs no change; the retirement below deliberately excludes remote-origin records (`Record.Origin != b.nodeID`).
- Case B, this node is the origin and had forwarded the request to a peer, then this proxy dies. VERIFIED: the local record is local-origin and falls under the defect; the peer's ingress handler loses its client connection (the dead proxy's socket) and cancels its engine request through `r.Context()`; the peer's store holds the record as remote-origin and will only converge when this node broadcasts a terminal, which today never happens, or when its silence sweep fires, which cannot happen while this node's workload-manager keeps re-asserting the record (1.4). The repair must therefore reach the workload-manager, not only the local store (see Part 2).

### 1.6 TUI and desktop projection (VERIFIED)

- Desktop renders what the broker says: the card dims and labels terminal states (`D/ui/components/Workloads/WorkloadItemCard.tsx:130-170`), shows `error` for `failed` (`:225`), and offers Cancel / "Cancel and run on another device" only while `workloadCancelRequest` is non-null, which requires state `queued` or `running` and a complete `{originatedFrom, engine, runId, id}` key (`D/shared/utils/workloads.ts:44-58`). The bridge forwards `workloads:cancel` unchanged and parses `{ok}` (`D/electron/service-bridge/node-policy.ts:359-376`; `D/ui/api/pair-api.ts:111-112`, `:233`); terminal classification mirrors the broker's (`D/electron/service-bridge/workload-projection.ts:77-83`).
- TUI renders the same stream and treats `completed`/`failed`/`cancelled` as finished (`T/jobs.go:28-41`, `:460-466`); it has no cancel action (no `workloads:cancel` producer under `T/`).
- Consequence today (INFERRED from the code above): after a proxy crash mid-stream the origin's own desktop shows the job as running with two live cancel buttons that always fail ("Could not cancel the job ..." from `WorkloadItemCard.tsx:65-76`), the TUI lists it as active indefinitely, every peer shows it running, and the scheduler keeps counting it as pending on `scheduledOn`. Once the records are terminal, both clients hide the cancel affordance and dim the card without any client change.

## Findings

### RED-1: open local-origin workloads of a departed proxy run are never retired and cannot be cancelled (re-verified)

- Claimed invariant: owner milestone, "honest completed/failed/cancelled outcomes including interrupted streams" and "headless service with Windows service-process restart" (common brief); I plan Compatibility, "older displayed workloads lacking a run remain displayable but cannot be cancelled" (which describes pre-`runId` history, not records of a run the broker itself watched die).
- Code path: Part 1.3 and 1.4 above. The absence is proven from code; the consequence is deterministic.
- Observed result: not executed here (process tests are forbidden in this environment); see UNVERIFIED-1 for what a real machine must show.
- Affected state: broker workload store (local-origin active records), scheduler pending counts, the workload-manager's re-sync set and every peer's store, desktop and TUI job lists, persisted history (the lost outcome is never written).
- Disposition: repair now on the broker lane, after SR 5c as the coordinator ordered; the plan in Part 4.

### UNVERIFIED-1: end-to-end timing of crash detection, retirement and replacement cancel

- Claimed invariant (this plan, Task 2 and Task 5): a killed proxy's open local-origin workloads reach `failed` before the supervisor's respawn and a replacement run's workload is cancellable while the old run's key is refused.
- Probe: the described `TestWorkloadDepartedRunRetired` (Task 5), MSI/PC3 only. The supervisor's exit detection (`B/supervisor.go:352-353`) rests on `cmd.Wait` returning (`B/proxy.go:180-194`), which is platform-independent, but Windows process teardown of the proxy's child sockets and the 1 s backoff interplay are only observable on a real machine.
- Disposition: defer to the plan's own Windows gate; nothing here can stand in for it.

### UNVERIFIED-2: engine-side effect of a dead proxy

- Claimed invariant (owner, accurate-or-unknown telemetry; I plan, "a delivered terminal marker is not engine-idle evidence"): retiring the workload must not be read as the engine being idle.
- Probe: whether llama.cpp stops generating when the proxy's upstream socket closes abruptly can only be observed with a real engine, which this investigation must not start. The plan therefore couples retirement to nothing on the residency or engine side (Part 5 item 5).
- Disposition: owner question 2 / defer; no code depends on it.

### LOW-1: a cancel during the crash-to-respawn gap is an RPC error, not `{ok:false}`

- `handleWorkloadsCancel` runs `cancelLocalWorkload`, which returns `proxy not available` when `policyProxy()` is nil (`B/workloadcancel.go:80-83`, `B/proxy.go:335-348`), and the broker answers `-32000` (`B/workloadcancel.go:56-59`); the desktop shows "Could not cancel the job: ..." (`WorkloadItemCard.tsx:72-76`). After Task 2 the window is the few milliseconds between `cmd.Wait` returning and the retirement, and the record is terminal afterwards, so the button is gone. No change proposed; recorded so Task 5 asserts "not `ok:true`" rather than a specific error.

### LOW-2: the first-round disposition (inferred provenance, `failWorkloadsForNode`'s path) would leave peers with ghosts

- `debate-i-g.md` RED-1 recommends "the same provenance path as `failWorkloadsForNode`". That path is inferred: `emitWorkloadEventInferred` never forwards to the workload-manager (`B/workload_projection.go:459-461` applies and fans locally only; the forward lives in `routeProxyWorkload`, `:353-357`), inferred records are never persisted (`B/workloadstore/store.go:515-534`) and are excluded from manager rehydration (`:485-507`). The origin's workload-manager would keep re-asserting `running` to every peer forever (1.4), and the honest outcome would vanish from history on the next broker restart. The plan uses an authoritative `workload:errored` frame through `routeProxyWorkload` instead (Part 2). Plan-vs-plan gap, no code effect yet.

## Part 2: who owns retirement

Decision: the broker, in `nvpair-ui-broker`, at the moment it knows the proxy process is gone, plus a fence on the first event of a replacement run. Reasons, by the repository rule that services own state and the desktop only renders:

1. The broker is the parent of the proxy and the only process that observes its exit (`B/supervisor.go:352-353`, `B/proxy.go:180-194`) and its deliberate stop (`B/proxy.go:456-470`). It already owns the authoritative store and already synthesizes terminals for departed peers (`B/workload_projection.go:652-690`). With its proxy dead, the origin broker is the single writer for local-origin records; a terminal it emits is the truth, not a guess, so it is applied authoritatively and broadcast like any proxy-emitted terminal.
2. Not the proxy's next run: the new process has a fresh run and must not mint events for records it never owned (the I plan's "no automatic resumption/replay" and "old-run requests cannot cancel a reused workload number"); it also does not exist when the supervisor's budget is exhausted.
3. Not the workload-manager: it is a relay with no notion of the proxy process and no store (`WM/manager.go:359-371` only tracks frames it is handed); giving it a timeout would re-introduce the inferred/silence semantics for the one node that has certain knowledge.
4. Not the desktop or TUI: they render the stream; hiding a stale card locally would leave peers, the scheduler and history wrong.
5. Not the stale sweep: its local-origin skip is deliberate (`B/workloadstore/store.go:455-457`), and its provenance is inferred (LOW-2).

## Part 3: the honest outcome and the notification shape

- Outcome: `failed`. The owner's rule is that an interrupted run is `failed` unless the requester cancelled. A requester whose proxy died saw a torn connection, not a cancel; the proxy's own graceful path already reports `cancelled` for the requests it drains (`P/http_handler.go:1063-1078`), so anything still open after the process has exited is lost work. `completedAt` is the broker's retirement time; `error` is the fixed string `proxy exited before the request finished` (reason code `proxy-exited`, the same word J Task 2 reserves for its presentation records); `scheduledOn`, `model`, `requestedModel`, `createdAt`, `startedAt`, `requesterId` are unchanged; `seq` is the stored `seq + 1`, because every event carries the producer's counter and the manager's dedup key includes it (`WM/dedup.go:114-135`; `P/spec.md:788-792`).
- Notification shape: the existing `workload:errored` lifecycle frame with `params.workloadInfo` (the full `Workload`), handed to `routeProxyWorkload` exactly as the proxy's frame would be, so it becomes a `workloads:upsert` to clients and the scheduler and a `workload:errored` broadcast to peers (`B/workload_projection.go:334-362`). No new method, field, state or reason code. The store accepts it as an authoritative same-generation forward transition (`B/workloadstore/store.go:272-300`: rank `failed` 3 over `queued` 1 / `running` 2); the manager marks it terminal and re-asserts it for two heartbeats (`WM/manager.go:443-460`); peers apply it by rank; the broker's history persists it (`B/workloadstore/store.go:530-533`).
- Because no JSON-RPC payload changes, nothing in `desktop/src/electron/service-bridge/`, `desktop/docs/services-api.md` or `desktop/docs/service-contract-exceptions.json` changes. If the owner instead wants a machine-readable reason (for example a `reason: "proxy-exited"` member on `workloadInfo`), the repository rule requires one unit changing together: producer `P/proxy.go:178-209` (`Workload`) and `B/workload_projection.go:99-105` (`workloadFieldKinds`, which strips unknown members) and `WM/workload.go` (the manager's `Workload`), broker relay `B/workload_projection.go` projection and `markWorkloadFailed` (`:797-817`), consumers `D/shared/types/workloads.ts`, `D/electron/service-bridge/workload-projection.ts` (`parseWorkload`, `sameWorkload`), `D/ui/components/Workloads/WorkloadItemCard.tsx`, `T/jobs.go:28-41`, tests in all of them, docs `WM/spec.md` (Workload shape), `WM/README.md:91-107`, `P/spec.md:765-810`, `B/README.md:731-756`, and the generated `desktop/docs/services-api.md` through `npm --prefix desktop run service-contracts:write`. This plan does not do that; see owner question 1.

## Part 4: execution plan

Conventions: all paths repository-relative; new Go files carry the SPDX header; `pair-test-style` for tests; commits `--no-signoff`; no `services/versions.json` or `CHANGELOG.md` edits (the eventual PR declares release intent for the broker binary). Verify commands run from the repository root with `GOTOOLCHAIN=auto` where the host Go is older than the modules' 1.25.

### Task 1 — store query for open local-origin records outside the live run

**Files:** `services/nvpair-ui-broker/workloadstore/store.go`, `services/nvpair-ui-broker/workloadstore/store_test.go`.

**Depends on:** none.

**Complexity:** mechanical.

**Do:** Add `func (s *Store) ActiveLocalOutsideRun(origin, liveRunID string) []Record` beside `ActiveForNode` (`store.go:396-410`): under `s.mu`, return copies of every record with `!Terminal && Origin == origin && Key.RunID != liveRunID`; an empty `liveRunID` therefore selects every open local-origin record. Add `RunID string` and `Seq int64` to `Record` and fill them in `recordFrom` (`:540-557`) from `Incoming`, extending `ParseIncoming`'s header struct (`:148-170`) with `seq`; both are additive and change no merge rule. Do not touch `StaleForeign`'s local-origin skip (`:455-457`) or the inferred rules.

**Verify:** `go -C services/nvpair-ui-broker test -count=1 -run '^Test(ActiveLocalOutsideRun|ActiveForNode|ReplayForNode|StaleForeign|ParseIncoming)' ./workloadstore` passes (exit 0, every listed test executed). New `TestActiveLocalOutsideRun` covers: open local records of run `r1` returned with `liveRunID=""` and with `liveRunID="r2"`; records of `r2` excluded when `liveRunID="r2"`; terminal local records excluded; remote-origin records excluded whatever their run; records executing locally but originated elsewhere excluded; `RunID`/`Seq` populated on returned copies.

### Task 2 — broker retirement producer and the two departure hooks

**Files:** `services/nvpair-ui-broker/departedrun.go` (new), `services/nvpair-ui-broker/departedrun_test.go` (new), `services/nvpair-ui-broker/workload_projection.go`, `services/nvpair-ui-broker/proxy.go`, `services/nvpair-ui-broker/proxyport.go`.

**Depends on:** 1.

**Complexity:** judgment.

**Do:**
- In `departedrun.go`: `const departedRunWorkloadError = "proxy exited before the request finished"` and `func (b *Broker) retireDepartedRunWorkloads(liveRunID, cause string)`. It calls `b.workloads.ActiveLocalOutsideRun(b.nodeID, liveRunID)`, and for each record builds a `workload:errored` frame with `markWorkloadTerminal(rec.Info, "failed", now, departedRunWorkloadError, rec.Seq+1)` and passes it to `b.routeProxyWorkload("workload:errored", params)` so the store apply is authoritative, the scheduler and subscribed clients are fanned, and the workload-manager receives the lifecycle frame for broadcast (`workload_projection.go:334-362`). One `slog.Info` with `cause` and `count` (ids only, never model or request content). Idempotent: a second call finds nothing open.
- In `workload_projection.go`: generalize `markWorkloadFailed` (`:797-817`) into `markWorkloadTerminal(info, state, completedAt, errMsg string, seq int64)` that also sets `seq`; keep `markWorkloadFailed` as the inferred sweeps' thin wrapper that leaves `seq` untouched, so the node-loss and silence sweeps are unchanged.
- Hook (a), deliberate stop: in `stopProxySupervisorLocked` (`proxy.go:456-470`) call `b.retireDepartedRunWorkloads("", "proxy stopped")` after `sup.Stop()` has returned and `clearProxyHandles()` ran; the process has exited by then, so any still-open local-origin record was lost. This covers credential restart (`:504-512`), listener disable (`:575-604`) and broker shutdown (`:515-525`); at shutdown the history flusher is still alive (`workload_projection.go:281-316`), so the outcome is persisted.
- Hook (b), crash: in `configureProxySupervisorCallbacks` (`proxyport.go:157-171`) extend the `clearHandle` closure to call `b.retireDepartedRunWorkloads("", "proxy crashed")` after clearing the engine handles. `onCrash` runs on the monitor goroutine after `Done()` closed and before the backoff and respawn (`supervisor.go:262-302`), so no replacement run can have emitted anything yet; `liveRunID=""` is exact.
- Not hooked, with the reason recorded in a comment: the two `spawnProxy` bring-up failure paths (`worker_lifecycle.go:357-361`, `:383-404`) stop a process that never had an enabled facade; the hybrid listener refuses inference at this head (`P/routing_listener.go:302-305`), so no local-origin workload can exist for it. This also keeps `worker_lifecycle.go` and `policy.go` out of this task (Part 5, SR 5c).
- Lock order: `stopProxySupervisorLocked` runs under `proxySupMu`; retirement takes `workloadEmitMu` then `schedulerFeedMu` (`workload_projection.go:477-500`, `:557-568`), the order `replayActiveWorkloadsToScheduler` already uses (`worker_lifecycle.go:632-644`); nothing under `workloadEmitMu` takes `proxySupMu` (searched). Document the order in `departedrun.go`.

**Verify:** `go -C services/nvpair-ui-broker vet ./...` exit 0, then `go -C services/nvpair-ui-broker test -count=1 -run '^Test(DepartedRun|FailWorkloadsForNodeMatchesByHostUUID|FailStaleForeignWorkloads|ActiveLocalReplayFrames|WorkloadsCancel|WorkloadPrivacyProjection)' .` passes with every listed test executed; repeat the `TestDepartedRun` selector with `-race` where supported. `departedrun_test.go` (top-level `TestDepartedRun` with subtests, using `storeIncoming` from `workloadmanager_test.go:16` and a fake workload-manager stdin writer): run `r1` queued and running records, an `r2` running record and a remote-origin record executing locally; `retireDepartedRunWorkloads("", ...)` marks both `r1` records and the `r2` record `failed`, authoritative (`!Inferred`), `error` equal to the constant, `completedAt` set, `seq` incremented, `scheduledOn` unchanged, remote record untouched; `retireDepartedRunWorkloads("r2", ...)` leaves `r2` running; the fake workload-manager received one `workload:errored` frame per retired record and a subscribed client one `workloads:upsert` each; the scheduler feed saw them; a second call emits nothing; `ReplayForNode` within the terminal window includes the retired records so a restarted manager re-asserts `failed`; a synthetic `onCrash` through `configureProxySupervisorCallbacks` and a `stopProxySupervisorLocked` with a nil supervisor both retire. Whole-module gate before integration: `go -C services/nvpair-ui-broker test -count=1 ./...` exit 0.

### Task 3 — replacement-run fence on the first event of a new run

**Files:** `services/nvpair-ui-broker/workload_projection.go`, `services/nvpair-ui-broker/broker.go`, `services/nvpair-ui-broker/departedrun_test.go`.

**Depends on:** 2.

**Complexity:** judgment.

**Do:** In `routeProxyWorkload`, after projection and stamping (`workload_projection.go:335-339`), parse the frame's `runId` with `workloadstore.ParseIncoming`; when it is non-empty and differs from `b.lastLocalRunID` (a new `atomic.Pointer[string]` or mutex-guarded string on `Broker`, declared beside `workloads` at `broker.go:347-353`), store it and call `b.retireDepartedRunWorkloads(runID, "proxy run replaced")` before applying the new event. Justification to record in a comment: the frame came from the current process handle (`proxy.go:199-203`), the broker runs one proxy process at a time (`stopProxySupervisorLocked` joins the old process before `startProxySupervisorLocked`, `proxy.go:504-512`, `:575-604`), so every open local-origin record of another run belongs to a process that no longer exists. This closes the cases the departure hooks cannot see: a lost or late acknowledgement, a fixture proxy that is not supervised, or any future path that publishes a handle without passing through the two hooks. It is a fence, not the primary mechanism: it fires only when the replacement serves its first request.

**Verify:** `go -C services/nvpair-ui-broker test -count=1 -run '^Test(DepartedRun|WorkloadPrivacyProjection)' .` passes; add subtests: a `workload:submitted` frame for run `r2` retires open `r1` records before `r2`'s record is applied and the client sees `r1` failed before `r2` queued; a second `r2` frame triggers no scan (assert via the fake store's call count or a test hook); a frame without `runId` (rejected by projection at `workload_projection.go:84-86`) changes nothing; remote-origin frames from the workload-manager path never reach the fence (they enter through `forwardWorkloadManagerNotification`, `:425-437`, not `routeProxyWorkload`).

### Task 4 — documentation of the retired outcome

**Files:** `services/nvpair-ui-broker/README.md`, `services/nvpair-workload-manager/spec.md`, `services/nvpair-workload-manager/README.md`, `services/nvpair-proxy/spec.md`.

**Depends on:** 3.

**Complexity:** mechanical.

**Do:**
- `B/README.md`: in the `workloads:upsert` / `workloads:remove` section add a bullet "Departed-run workloads" after the inferred bullet (`:744`): when the proxy process this broker supervises exits or is replaced, the broker itself ends every open local-origin workload of the departed run as authoritative `failed` with error `proxy exited before the request finished`, applies it to its store, feeds the scheduler, and forwards it to the workload-manager so peers converge; this is not inferred and is not overridden. Amend the sentence "Workloads this node originated or is itself executing are never inferred about" to say they are never inferred about, but a departed run's are retired authoritatively. In `workloads:cancel` (`:810-822`) add: a job whose proxy run has exited has already been retired and is not cancellable; `ok:false` from the replacement run, or `-32000 proxy not available` while no proxy runs.
- `WM/spec.md:85-87` and `:171`: note that `workload:errored` with `failed` may be produced by the broker for a departed proxy run, with the fixed error text, and that the manager treats it like any local terminal (re-asserted for two heartbeats).
- `WM/README.md:66-74`: one sentence under the method table with the same fact.
- `P/spec.md` §5.5 (`:765-773`): add a row "proxy process exit (crash or replacement) | `workload:errored` synthesized by the broker | `failed` | unchanged" and one sentence that the proxy's own `cancelled` terminals come only from a graceful stop. §12.5 (`:1288-1304`): after "a `runId` other than this process's finds nothing" add "because the broker has already retired that run's workloads as failed".
- No change to `desktop/docs/services-api.md` (generated; no payload change) or `desktop/docs/service-contract-exceptions.json`; no `docs/*.mdx` describes workload outcomes at this head.

**Verify:** `git diff --check` clean; `npm --prefix desktop run service-contracts:check` still passes unchanged (proves no contract drift was introduced); a reviewer reads the four files for "engine" not "backend" and for the exact error string matching `departedRunWorkloadError`.

### Task 5 — process-level evidence: kill the proxy mid-request

**Files:** `services/tests/workload_departed_run_test.go` (new). Shared helpers reused, not edited: `startBrokerWithDirs` (`X/broker_supervision_test.go:55`), `proxyPidRe`/`waitForStderr`/`mustPid` (`:204`, `:146`, `:260-271`), `waitForMethod`/`waitForResponse` (`X/main_test.go:253`, `:272`), `writeRawFrame`, `waitProxyReady`, `waitForWorkloadEvent` (`X/workload_interop_test.go:336`, `:758`, `:826`), `portOfURL`, `newInterNodeCluster`, `startStubClusterPeer`, `waitStubPeerWorkload`, `drainWorkloadMessages` (`X/workload_identity_interop_test.go:425`, `:376`, `:402`; `X/cluster_data_plane_test.go:48`, `:119`). `scripts/windows-test-bin.ps1` is unaffected (same test package, no new executable); confirm with its `list` before claiming so.

**Depends on:** 3.

**Complexity:** judgment.

**Do (described, not run):** top-level `TestWorkloadDepartedRunRetired`, skipping like its neighbours when port 11435 is busy, with two subtests:
1. `OriginRetiresAndReplacementCancels`. Start a fake engine as in `X/workload_identity_interop_test.go:174-193`: flush one chunk so the proxy commits and the workload is `running`, then block until released or `r.Context()` ends. Start the broker with `--proxy-path`, `--proxy-engines ollama`, `--workload-manager-path`; `workloads:subscribe`; wait `app:ready`, `waitProxyReady`, record `pid1` from `proxyPidRe`. Pin and select the fake engine (`node/add-manual`, `node/select`), fire one `/api/chat` in the background, and wait for `workloads:upsert` id `1` state `running`; record its `runId` as `r1` and `engine`. Kill `pid1` (`os.FindProcess(pid1).Kill()`). Assert within 15 s a `workloads:upsert` for id `1` with `runId == r1`, `state == failed`, `error == "proxy exited before the request finished"`, `completedAt` non-null and `seq` greater than the running frame's. Call `workloads:cancel {originatedFrom, workloadId:"1", engine, runId:r1}` and assert the response is not `{ok:true}` (either `ok:false` or an error; LOW-1). Call `workloads:get-initial` and assert the record is `failed`. Wait for the respawn (`proxyPidRe` again, `pid2 != pid1`) and `waitProxyReady`; fire a second blocking `/api/chat`; assert `workloads:upsert` id `1` `running` with `runId r2 != r1` (the facade counter restarts at 1, which is the reused-number case). Cancel with `runId r1` and assert `ok:false` and no `workloads:upsert` for `(r2, 1)` leaves `running` within 2 s; cancel with `runId r2` and assert `ok:true` followed by `workloads:upsert` `(r2, 1)` `cancelled` with error `cancelled by request`. Release the fake engine in cleanup before closing it (the defer-order note at `X/workload_identity_interop_test.go:191-196`).
2. `PeerHearsFailed` (needs mDNS, so it runs outside the network sandbox like `TestWorkloadManagerRehydratesActiveWorkloadOnRestart`): same start with `newInterNodeCluster` and `startStubClusterPeer`; wait until the stub peer has heard `running`; `drainWorkloadMessages`; kill the proxy; assert the stub peer receives a `workload:errored` frame for the same `(origin, engine, r1, 1)` with `state failed` within 15 s (direct broadcast), and again within 45 s (heartbeat re-assertion of the terminal). This proves the broadcast half of Part 2 and is the assertion the inferred path would have failed.

Readiness barriers only; no sleeps as proof. No real engine, model, provider or installed PAIR; disposable directories only.

**Verify:** MSI or PC3 only, from `services/tests`, after `TESTING_ON_PC.md` preflight (installed candidate stopped, fixture ports free, firewall coverage for the test binary): `go test -count=1 -v -run '^TestWorkloadDepartedRunRetired$' .` passes with both subtests executed (a skip, port collision, firewall prompt or timeout leaves acceptance pending); on MSI also `go test -race -count=1 -v -run '^TestWorkloadDepartedRunRetired$' .` when the Windows race toolchain is available. Also re-run the neighbours this plan's behaviour touches: `go test -count=1 -v -run '^(TestBrokerCrashSurfacingAndRestart|TestWorkloadManagerRehydratesActiveWorkloadOnRestart|TestWorkloadManagerRehydratesRecentTerminalOnRestart|TestWorkloadFailedOnNodeLoss)$' .` (all four exist at `7e270f1`, confirmed by `go test -list`). Record SHA, commands, exit codes and bounded raw output as the plan's evidence; redact request material. Not on the Mac or in a cloud container.

### Plan graph

1 -> 2 -> 3 -> 4, 3 -> 5. No shared Files between concurrent tasks; Tasks 4 and 5 are disjoint and may run in parallel after 3. Five tasks.

## Part 5: interactions and ordering traps

1. SR 5c (`origin/roadmap/sr-5c`) edits `B/policy.go` (157 lines), `B/worker_lifecycle.go` (`spawnProxy`, six lines right after `replayPolicyToProxy`), `B/broker.go` (seven lines in `forwardEngineNotificationFrom`), `B/idlepolicy.go`, new `B/modelconfig.go`, and `B/README.md` (34 lines). This plan deliberately hooks `B/proxy.go` and `B/proxyport.go`, not `replayPolicyToProxy` or `recordActivation` in `B/policy.go` and not `spawnProxy`, so the only overlaps are one field in `B/broker.go` and the README section. Ordering: after SR 5c as the coordinator ruled; if the coordinator wants the milestone hole closed first, the plan can land before SR 5c with the same files, and SR 5c would rebase the README only. Trap: a reviewer must not "simplify" by moving the retirement into `replayPolicyToProxy`, which is both an SR 5c file and too early (the old run may still be draining when a credential restart calls stop then start; the stop hook is the right moment).
2. SR 5c residency semantics: a replaced proxy run loses its in-memory resident holds and SR 5c refreshes the engine inventory after the replay (`sr-5c` hunk in `spawnProxy`); the reconcile is level-triggered per (engine, model, revision) and cleared only by observation, failure, revision change or timeout (common brief). Retiring workloads must not feed that state machine: a retired workload is not evidence the engine is idle (UNVERIFIED-2), and the retirement must not trigger unloads, drains or `dispatch-resident` decisions. The scheduler's pending count drops at retirement while the engine may still be finishing a dead socket's generation for a few seconds; that is the same approximation every terminal already carries.
3. I Task 4b (`origin/roadmap/i-public-lifecycle`) rewrites `P/http_handler.go`, `P/cancel.go` and `P/request_registry.go` and adds `routing.go`/`response_sink.go`; it touches no broker file. This plan touches no proxy file. The process test depends only on the proxy emitting `running` at first committed byte and `cancelled by request` on cancel, which 4b keeps (its registry still refuses a foreign run at `workloadCancelControl`). No ordering constraint beyond the test running on the integrated head.
4. I Task 5: every pre-commit hybrid local attempt has an ordinary local workload (I Decisions), so a departed run's hybrid attempts are retired by the same hooks; public attempts create no cluster workload, so there is nothing to retire and the sink/ledger facts stay with I. The registry entries of the dead process vanish with it, so `routing:active` of the replacement run is naturally empty.
5. I Task 6: its documentation sentence "preserve old workload display while explaining why missing-run records cannot be cancelled" must be reconciled with Task 4: after this plan the only non-cancellable displayed records are history (terminal) and pre-`runId` records; a record of a run the broker watched die no longer stays live. I6's proxy README/spec text and Task 4's `P/spec.md` §12.5 sentence touch the same paragraph; whichever lands second rebases a few lines.
6. I Task 7: `TestUnifiedDispatchProcess`'s restart case ("late old notifications/cancels", "no automatic restart replay") should add the assertion from Task 5 subtest 1 (departed-run records reach `failed`; the replacement run cancels nothing of the old run) rather than duplicating the fixture; propose this as an I plan amendment, since I7 owns `X/unified_dispatch_test.go` and this plan owns `X/workload_departed_run_test.go`. Both are Windows-only acceptance; neither runs on the Mac.
7. J Task 2 retires its own presentation records "failed/proxy-exited" after fencing the departed run (J plan line 102). Those are routing-diagnostic records, not Workload records; J must not assume Workload retirement and this plan must not grow a presentation store. Keep the reason word `proxy-exited` identical so the two views agree.
8. PA Tasks 1-2 (router skips paused peers) and G3/G4 share no file with this plan.
9. Trap, provenance: using the inferred path (LOW-2) would pass a local unit test and still leave peers with permanent ghosts; Task 5 subtest 2 exists to catch exactly that regression.
10. Trap, `seq`: `Record` has no `Seq` today, so a retirement frame built from `Info` alone would repeat the last `seq`; Task 1 projects it and Task 2 increments it. The manager's dedup would still pass (the state differs), but the producer contract (`P/spec.md:788-792`) and J's ordering rules expect monotonic sequences.
11. Trap, `workloads:remove`: never retire by removal; removal is history housekeeping, is not broadcast (`B/workload_projection.go:513-550`) and would delete the honest outcome from clients that already hold the record.
12. Trap, broker restart during the gap: active records are not restored (`B/workloadstore/persistence.go:15-19`), so a broker restart after a proxy crash is already clean; the plan changes nothing there, and Task 5 must not assert persistence of an active record.

## Owner challenges and questions

1. Error text and reason exposure. The plan uses the fixed `error` string `proxy exited before the request finished` and no new field. If the owner wants a machine-readable reason on the wire for J or the desktop, that is a `Workload` payload change with the full propagation list in Part 3 and should be a separate unit after I6/J2 settle the shared contract.
2. `failed` for deliberate replacement. A credential-set change or listener disable that cuts a request past the three-second drain produces `failed`, while the drained requests produce `cancelled` from the proxy itself. This follows the owner's rule (interrupted equals failed unless the requester cancelled) and keeps the two provenances distinct; confirm, or ask for `cancelled` with a distinct reason when the stop was the owner's own action.
3. Lane placement. The coordinator ordered this after SR 5c. The plan's files are disjoint from SR 5c's except `B/README.md` and one `B/broker.go` field, and the daily milestone names the restart case; the owner may prefer it before SR 5c. Either order works with the files as reserved here.

## Verification record (this environment)

- `git rev-parse origin/fork/groundwork` = `7e270f10ced402325dc9e1efece54991e9bbbc08`; branch `plan/inv-plan-departed-run` created from it.
- `go vet ./...` in `services/nvpair-ui-broker`: exit 0. `go test -count=1 ./workloadstore` in `services/nvpair-ui-broker`: ok.
- `go vet ./...` in `services/tests`: exit 0. `go test -list . ./...` in `services/tests`: exit 0; its `TestMain` built the product binaries to produce the listing; no test, listener, service or engine ran. The listing confirms `TestBrokerCrashSurfacingAndRestart`, `TestWorkloadCrossEngineIdentityDistinct`, `TestWorkloadManagerRehydratesActiveWorkloadOnRestart`, `TestWorkloadManagerRehydratesRecentTerminalOnRestart`, `TestWorkloadFailedOnNodeLoss`, `TestWorkloadManagerOutboundBroadcast` exist at this SHA.
- Not run: the broker's whole-module tests, `-race` anywhere, proxy or workload-manager module tests, desktop checks, any `services/tests` case, any engine, model or PAIR service.
- Branch diffs inspected read-only: `origin/fork/groundwork...origin/roadmap/sr-5c` (broker lifecycle, policy, README) and `origin/fork/groundwork...origin/roadmap/i-public-lifecycle` (proxy registry and cancel).
