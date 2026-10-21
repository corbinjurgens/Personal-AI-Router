<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Decision sheet: the nine open owner questions (slug: owner-decisions)

Kind: decision support. Investigation only: no product change, no pull request. Each question gets the options, the code cost with the seam that changes, the risks, the roadmap tasks it unblocks or adds, and one recommendation. A final section lists decisions the code already assumes without a recorded answer.

Inputs: `plans/handoffs/inv-common.md` and `plans/handoffs/inv-owner-decisions.md` (branch `plan/cloud-investigations`), the context plans `i-unified-dispatch.md`, `g-egress-lease.md`, `sr5-rescope.md` on that branch, `plans/b-opencode-shapes.md` (branch `plan/b-opencode-shapes`, read at `fefe7b8`), `plans/reviews/debate-i-g.md` (branch `plan/inv-debate-i-g`), the two OpenCode inventories (`plan/b-opencode-inventory-msi`, `plan/b-opencode-inventory-pc3`), and, because its owner-question list is the origin of questions 2 to 5, `plans/reviews/plan-daily-acceptance.md` (branch `plan/inv-plan-daily-acceptance`) plus `plans/reviews/audit-pa-j-b.md` UNV-3 for question 8.

## Verdict

`RED 0 UNVERIFIED 2 LOW 2`

This sheet asks for decisions rather than reporting defects. The two UNVERIFIED items are facts the owner's machines alone can settle (which port each OpenCode installation targets; whether a llama-server router survives a service restart and is adopted). The two LOW items are code facts with no observable effect today that change how several answers should be read: strict qualification, and therefore every B shape rule, is unreachable from production traffic at the inspected SHA; and free-form manual node names cannot be entered from the desktop or TUI.

## Inspection base and environment

- `origin/fork/groundwork` = `7e270f10ced402325dc9e1efece54991e9bbbc08`, the SHA the common brief names; the branch had not moved. Every `path:line` below is at `7e270f1` unless a plan branch is named. Path shorthand from the common brief: P/ = `services/nvpair-proxy/`, B/ = `services/nvpair-ui-broker/`, E/ = `services/nvpair-engine-manager/`, N/ = `services/shared/`, D/ = `desktop/src/`.
- VERIFIED means the code was read at that SHA or a check ran here; INFERRED means a consequence derived from verified code without execution.
- Checks run: `go vet ./...` in P/, B/, E/, `services/nvpair-service`, `services/nvpair-manual-nodes` and N/ (all exit 0, Go 1.25 toolchain via `GOTOOLCHAIN=auto` on a 1.24 host); `go test -count=1` in `nvpair-shared/nodepolicy`, `nvpair-shared/engines` and `nvpair-manual-nodes` (ok). Nothing under `services/tests` was executed or listed; no engine, model, service or desktop check was started; no `-race`.
- Private inputs (`<run-dir>` `owner-decisions.md`, `decided.md`, the frozen NE host records, OpenCode configuration values, `TESTING_ON_PC.md`) were not available; where an answer depends on them it says so.

## Findings

### UNVERIFIED-1: which port each OpenCode installation targets is not knowable from the repository, and one inventoried configuration cannot be a PAIR facade on another host

- Claimed invariant (common brief): "Client: OpenCode over OpenAI-compatible Chat Completions"; the acceptance plan's convention: "The llama.cpp facade port 8080 is treated as the OpenCode base URL on each serving host and on the Mac" (`plan-daily-acceptance.md` line 103).
- Code path (VERIFIED): the facade's plaintext personality refuses every non-loopback caller with 403 `loopback-only` (`P/ingress.go:93-100`); LAN peers reach a node only through the pinned mTLS cluster ingress (`P/ingress.go:112-131`), which OpenCode cannot present. The llama.cpp facade standalone port is 8080 (`P/engines.go:262-266`; `N/engines/engines.go:163-167`), the engine itself runs from 8081 (`E/manifests/llamacpp.json:24`).
- Evidence: the MSI inventory reports "both loopback and non-loopback endpoints exist; endpoint values and model names are withheld" (`plan/b-opencode-inventory-msi` line 36); the PC3 inventory reports "one canonical `providers` entry ... using ... a non-loopback endpoint category" (`plan/b-opencode-inventory-pc3` line 112).
- Observed result (INFERRED): a non-loopback OpenCode endpoint is either a direct engine port (8081 on another host, bypassing PAIR), a remote non-PAIR provider, or a misconfiguration that the facade will refuse. It cannot be a working cross-host PAIR facade.
- Affected state: the answer to question 9 and every milestone clause observed through OpenCode.
- Disposition: owner question (question 9 below); confirm the configured base URL per host in the frozen record.

### UNVERIFIED-2: whether a llama-server router outlives a broker or service restart on Windows, and is then adopted, is only provable on a real machine

- Claimed invariant (owner milestone): "headless service with Windows service-process restart"; NE: "Adopted/external runs publish none" (identity).
- Code path (VERIFIED): a replacement engine-manager starts saved-On engines (`E/desiredstate.go:212-241`, triggered by `B/broker.go:732-746`); `doStart` probes the port and, when the listener identifies as the engine, adopts it without a process handle (`E/lifecycle.go:144-175`; `E/status.go:146-262`); identity and serving snapshots require an owned run (`E/model_identity.go:164`; `E/serving_identity.go:111-112`). Whether the router process survives the parent's death is a Windows process-tree fact (no job object is created in `E/proc_windows.go:24-31`), not provable here.
- Disposition: owner question (question 3), to be settled by the acceptance plan's T7 item 4 observation.

### LOW-1: strict qualification, and therefore every B shape rule, is unreachable from production traffic at `7e270f1`

- `X-PAIR-Strict-Admission` is parsed at `P/http_handler.go:605-620` and `P/ingress.go:144-150`; only then does `serveStrictLocal` call `qualifyStrict` -> `validateRequest` (`P/ingress.go:359-367`; `P/readiness_contract.go:272-282`). The only code that builds a strict candidate is `hybridLocalCandidates` (`P/tiers.go:445-473`), which has no production caller (search for `hybridLocalCandidates(` outside tests: definition only), and the hybrid listener refuses inference (`P/routing_listener.go:302-305`). The facade tier path merges only the profile's `RequestOptions` (`P/http_handler.go:729`) and forwards the body. No observable effect; recorded because it changes the weight of questions 1 and 4 and of B Task 7.

### LOW-2: free-form manual node names cannot be entered from the desktop or the TUI

- The manual-nodes worker keys an entry by `Name` when present, else `"manual:" + Address` (`services/nvpair-manual-nodes/manager.go:736-741`), and validates only that the address is nonempty (`:681-689`). The desktop sends `node/add` with a name only in the one-time legacy-file migration (`D/electron/service-bridge/manual-nodes-migration.ts:12-16`; `D/electron/service-bridge/modular-supervisor.ts:1032-1035`); the TUI sends the address alone (`services/nvpair-tui/ui/nodes.go:1090-1096`). A free-form id therefore arises only from the migration or a raw RPC. No observable effect in the owner's fleet; it bounds the blast radius of question 8.

## Question 1 — B: support reasoning replay (`reasoningReplay`), or keep it refused?

What the code does (VERIFIED). The qualification artifact carries `shapes.reasoningReplay` (`P/local_qualification.go:88-92`) and refuses any document that sets it true: "shapes.reasoningReplay must be false while reasoning replay is unsupported" (`:113-118`). A strict request whose message carries `reasoning_content` fails `strictObject(m, "role", "content", "name", "tool_calls", "tool_call_id")` with "messages[i] has an unsupported shape" (`:695-701`). Both rules live only on the strict path (LOW-1): daily non-strict traffic forwards `reasoning_content` to llama-server untouched.

What the client does (plan evidence, not inspected here). Both inventoried OpenCode releases (MSI 2.1.1, PC3 2.0.32) automatically lower present unstructured reasoning into assistant `reasoning_content`, even with `requireReasoning:false`; there is no client switch to turn replay off (`plans/b-opencode-shapes.md` lines 51, 57; inventories). Whether the daily local templates return replayable reasoning is unverified on both hosts.

Options.

- A. Keep refused (status quo). Cost 0. Risk: once strict becomes the daily path (question 4), the second turn of any conversation with a reasoning model fails 400 `strict-qualification` before admission, with no client-side workaround. Until then, no effect.
- B. Qualify and forward unchanged (the r2 proposal): allow `reasoningReplay:true` only with exact NE-bound template evidence, accept `reasoning_content` as a JSON string on assistant messages only, count its UTF-8 bytes into the input bound, never strip. Cost: drop the false-only rule (`:113-118`), extend the message allowlist and accounting (`:695-701`, plus a role/type check and byte count), fixtures and `TestQualificationShapeMatrix`; this is B Tasks 2-6 as written, gated on NE Task 4 Part A and I file releases. Risk: forwarding reasoning to a template that does not accept it is exactly what the per-model evidence must exclude; evidence must be re-recorded when a template changes; replay inflates input bytes, so bounds can refuse long sessions earlier (honest).
- C. Strip before forwarding. Rejected by the plan (changes model behavior, bypasses accounting); listed only for completeness.

Roadmap. B unblocks B Tasks 2-6 (all wait on this answer) and SR 6a's documentation consumption. A leaves B at Task 7 and makes strict-path multi-turn use a known incompatibility.

Recommendation: approve B. The non-strict path already forwards replay; B makes the strict path match that reality under evidence instead of refusing it. Nothing changes for daily traffic until question 4 moves it onto strict.

## Question 2 — headless Windows session: should the service run without an interactive login, and what does that require?

What the code does (VERIFIED). `nvpair-service` is a per-user process: autostart is the per-user `Run` registry value "for this user only, with no elevation" (`services/nvpair-service/autostart_windows.go:14-27`; README lines 6-11, 48-52), which runs only at interactive logon. Its endpoint is a per-user named pipe whose DACL grants the current user's SID alone (`N/servicectl/servicectl_windows.go:29-34`; `N/ipc/ipc_private_windows.go:28-36,59-64`); state lives under the user's local application data (`N/appdir/appdir.go:45-55`). The service restarts a dead broker with backoff and replays subscriptions (`services/nvpair-service/broker.go:128-190`, `service.go:70-71,552-557`); the broker respawns dead workers (`B/supervisor.go:239-300`); a replacement engine-manager starts only engines with a saved On intent (`E/desiredstate.go:212-241`) when the broker asks (`B/broker.go:732-746`; `B/availability.go:333-338`). Engines inherit the service's environment (`E/proc.go:35-48`) and run windowless (`E/proc_windows.go:24-31`). Nothing restarts the service itself. The acceptance plan already notes the consequence: "a headless machine with no interactive login has no service" (`plan-daily-acceptance.md` line 247).

Options.

- A. Interactive logon stays required (status quo; automatic logon plus a locked screen covers reboots). Cost 0. Risk: a reboot without automatic logon leaves the host silent; automatic logon stores a credential on the machine; nothing supervises the service process.
- B. Register the service as a Task Scheduler task that runs at startup as the owner's account "whether the user is logged on or not". The process then runs in the non-interactive session under the user's token with the profile loaded, so the pipe name, DACL, application-data paths, model paths and engine environment are unchanged; a later interactive logon starts a second instance that exits "already running" through the pipe's first-instance flag (`services/nvpair-service/platform_windows.go:24-29`). Cost: an alternative writer beside `writeAutostart`/`removeAutostart`/`readAutostart` (`autostart_windows.go:21-66`, roughly 80-150 lines calling the scheduler), a flag like the existing `--systemd` split (`autostart.go:66-80,140`), one Windows test. Task Scheduler's restart-on-failure setting would also supervise the service. Risks (INFERRED, not provable here): CUDA engines in the non-interactive session work in general but must be confirmed on MSI; a batch-logon right is required for the account; per-executable firewall rules must already exist because no prompt can be answered.
- C. A real Windows Service (SCM). Cost: a service control handler in `run`/`serve` (`main.go:41-86,165-235`), install/uninstall commands, an explicit application-data override (the service account's local data is not the owner's), a pipe DACL that also grants the interactive user, engine install and model paths under a non-user profile, Windows tests; several hundred lines across `nvpair-service`, `N/appdir`, `N/ipc` and E/. Risk: largest change on the milestone path; running as the owner's account needs a stored password, running as a system account relocates every path the engines use.

Roadmap. B adds one small SRR follow-up task and leaves the acceptance plan's T7 intact (it already records autostart and restart). C adds a package. Question 3 decides what the engine does across the restart either way.

Recommendation: B. If the owner accepts a logged-in locked session for day one, A costs nothing and T7 proves it as written; B is the cheap path to a genuinely headless host and C should wait until a multi-user or system-account requirement appears.

## Question 3 — after a service restart, adopt an already-running llama-server or restart it?

What the code does (VERIFIED). When the replacement engine-manager restores a saved-On engine, `doStart` first reconciles presence on the configured port (`E/lifecycle.go:144-162` -> `E/status.go:146-262`): a listener that answers the ready probe with the manifest's identity (`E/manifests/llamacpp.json:25`, `role == router`) is adopted ("adopting already-running external engine", `E/lifecycle.go:165-191`): `adopted=true`, no process handle, a fresh run id, health loop only. An adopted run publishes no identity (`E/model_identity.go:164`), has no serving identity ("Serving identity requires a current owned run.", `E/serving_identity.go:111-112`), its settings readback reports mismatch (`E/settings.go:147-163`), finite llama serving refuses to apply ("Finite llama serving requires an owned process launch.", `E/settings.go:323`), and port moves are refused (`E/setport.go:60-68`). Concrete, non-strict serving through the facade continues. The reclaim predicate already exists on the stop path: PID on port, image equals the managed binary, path inside the managed install directory, then terminate (`E/lifecycle.go:620-650`; `E/proc.go:129-147,193-224`; `E/proc_windows.go:97-108`). The acceptance plan's T7 item 4 records which case occurs.

Options.

- A. Adopt (status quo). Cost 0. Keeps loaded models warm and any stream a still-live proxy is serving (manager-only crash). Risk: until someone issues `engine:restart`, the host has no strict readiness, no NE envelope, no SR residency; the desktop shows "running under external management"; the daily non-strict path is unaffected (LOW-1).
- B. Restart owned-image orphans at restore: in the adoption branch, when `pidOnPort` returns a PID whose image is our managed binary (reuse the doStop predicate), terminate with the stop grace and fall through to the owned launch; keep adoption for foreign listeners. Cost: about 40-60 lines at `E/lifecycle.go:165-175` plus a helper shared with `:620-650`, two lifecycle tests, one Windows-only presence test. Risk: every service or broker restart reloads the resident models cold (tens of seconds to minutes for the daily sizes) and cuts streams a surviving proxy was still serving; after a whole-service restart those streams are already dead because the proxy died with the broker.
- C. Re-own from readback: persist at launch a run record (run id, launch digest, prepared preset digest, PID, image, process creation time) and, at adoption, accept the listener as owned when PID/image/creation time match and the current compiled preset digest equals the recorded one. Cost: a new persisted file in E/ (not `desiredstate.go`, which SR fences), a re-own path in `E/serving_identity.go` and `E/model_identity.go`, broker identity replay, an NE plan amendment; 200+ lines. Risk: launch inputs that changed on disk after the launch are detected only through the digest; a stale record on a reused PID is excluded by creation time; the manager still holds no process handle, so stop must use the reclaim path.

Roadmap. B is a small SRR/NE follow-up and makes T7 item 4 deterministic (always "restarted"). C is an NE amendment that also touches SR 5a-ii/5b evidence rules. A adds an operator step ("restart the engine after every service restart") to the acceptance.

Recommendation: B. The identity that strict readiness and residency depend on is worth a cold reload per restart, restarts are rare, and B reuses an existing, tested predicate. Revisit C only if restart frequency makes reloads painful.

## Question 4 — daily milestone: accept the non-strict path now, or pull I Task 5 forward?

What the code does (VERIFIED). OpenCode at the facade takes the concrete tier path: `nodepolicy.IsTier(model)` -> `resolveTierCandidates` (`P/http_handler.go:655-667`; `P/tiers.go:363`), concrete admission, profile defaults merged (`:729`), five dispatch attempts within a ten-minute job deadline (`P/http_handler.go:130-147`). Strict admission requires the header (`:605-620`), its only producer has no caller, and the hybrid listener refuses inference (LOW-1). Strict is what joins B qualification (`P/readiness_contract.go:272-282`), finite context accounting (`P/local_qualification.go:425-430`) and NE strict readiness (`P/readiness_contract.go:299`) to a request. I Task 5 is the planned producer, depends on 4b (I plan Task 5 "Depends on: 4b"; 4b is in flight), and bundles the hybrid listener cutover the debate recommends deferring (`debate-i-g.md` I Task 5 Objection A, amendment 3).

Options.

- A. Accept non-strict for the milestone. Cost 0. What the owner gives up: no shape gating (B Task 7's `store:false` fix does not apply; what matters is what llama-server does with `store`, the acceptance plan T11 observation), no finite context refusal before the engine (llama-server's own behavior on overflow applies), no identity-bound readiness in routing (a tier candidate is eligible on roles and discovery, not on NE envelope). What still holds: tier and peer routing with real models, veto, workloads, honest outcomes, telemetry.
- B. Pull I Task 5 forward as a milestone prerequisite. Cost: I 4b (22 files, in repair) then Task 5 (12 files: `P/tiers.go`, `P/routing.go`, `P/routing_listener.go`, `P/http_handler.go`, `P/ingress.go`, `P/peer_transport.go`, N/ routing) plus B Tasks 2-6 and NE Part B for the evidence to exist. Risk: weeks on the milestone path for a listener the daily client does not use; the listener rework also touches the facade's `setPort`/`swapListener` (`P/facade.go:846-900`) that OpenCode does use.
- C. Strict on the facade: have the tier path consume `hybridLocalCandidates` for local and peer candidates and send the strict envelope (the header the peer ingress already parses, `P/ingress.go:144-150,190-192`), keeping the hybrid listener deferred. Cost: `P/http_handler.go:655-700` (resolve) and the attempt header injection near `:1554`, fallback rules when no candidate is strict-ready, tests; medium, but it must land inside I's single-owner cutover (4b-i) or it creates a second selection authority the I plan forbids.

Roadmap. A keeps the milestone on 4b-i, PA 1-2, SR 5c. C is a scoped replacement for Task 5's selection half (the debate's amendment 3 already proposes "keep from Task 5 only the pinned-origin check"; C would add "strict candidates on the facade"). B reorders the whole I lane.

Recommendation: A for the milestone, with the record that B/NE strict evidence is not exercised by daily traffic, and C scheduled as the first post-milestone I unit (after 4b-i) in place of the full Task 5 listener cutover. Question 1's answer only takes effect when C or B lands.

## Question 5 — interrupted streams: day-one honest error (SC-a) now, or plan same-model continuation (SC-b) now?

What the code does (VERIFIED). An interrupted stream ends with exactly one bounded frame, `{"error":{"message":"stream interrupted: <reason>","type":"server_error","code":"stream_interrupted"}}` then `[DONE]` (`P/streamframe.go:228-236`), reasons bounded at `:193-201`, chosen by `finishInterruptedLocked` (`:993-1003`); a peer destination's canonical record is forwarded intact by the origin (`:670-681`). The workload ends failed. The owner decision reads: "an error only if no same-model continuation exists"; no continuation exists (I plan Decisions: "At I's endpoint no qualified continuation implementation exists ... produces SC-a's clean terminal error"; SC-b has no plan branch). The acceptance plan T8 and the debate's owner challenge 3 both say the same.

Options.

- A. Accept SC-a as day-one behavior. Cost 0. Risk: a peer loss mid-stream shows the user an error in OpenCode; they retry by hand.
- B. Plan SC-b now. What it needs: the request-owned sink (I 4b), retained delivered bytes per request under the byte budget, continuation eligibility (same content digest, engine family, prefill qualification, context fit), prefix reconstruction into a new engine request and a splice that the client cannot see. Chat Completions has no partial-assistant continuation, so the executor must either use an engine-native completion route with the template applied by PAIR, or accept a visible boundary. Cost: a new package after 4b-ii, touching `P/streamframe.go` (termination decision moves to the sink), `P/http_handler.go`, `P/owned_buffers.go`, NE identity consumers; large. Risk: duplicated or garbled text at the splice, retained output bytes competing with the 192 MiB budget, and the "delay" the user sees can exceed the time to simply retry.
- C. Narrow SC-b to same-node restarts only (engine child crash while the router survives). Smaller, but still needs the sink and prefix reconstruction.

Roadmap. A leaves SC-b unscheduled; T8 records "WAITS-FOR" honestly. B adds a package after I 4b-ii and before I 5/6 could reasonably use it.

Recommendation: A now, and write the SC-b plan only after I 4b-i lands and the owner has seen how often interruptions occur in daily use. The milestone wording "honest ... outcomes including interrupted streams" is met by SC-a; "delay or pause-resume" stays an explicit open clause.

## Question 6 — per-host serving presets and resident model selection: who chooses, and where is it configured?

What the code does (VERIFIED). Two documents per host, both target-local and both remotely editable through PG mutation routes: (1) the node policy, written only by that host's broker at `<appdir>/node-policy.json` (`B/policy.go:293`), holding `profiles` (`N/nodepolicy/policy.go:337-367`), `servedModels`/`residentModels` with the subset rule (`:107-110,560-566`), the host's `roles` (`:111-116`), `tiers` (`:117-124`) and `admission` (`:296-312`); remote writes go through `/v1/node-policy/set` (`E/policyrelay.go:63-67`, `controlMutation`) with target-local permission. (2) The engine settings, whose `serving` member is "an operator-approved declaration, not runtime evidence" (`N/engines/serving.go:18-24`; `N/enginesettings/settings.go:9-16`), and whose launch text names one absolute native preset file (`E/serving_llamacpp.go:547-549`) that the compiler validates per model (`:520-700`, adapter defaults `ctx-size 8192`, `parallel 1`, `sleep-idle-seconds 300` at `:569`); edited through `engine:preview-settings`/`engine:apply-settings` with `nodeId` for a peer (`B/ENGINE_SETTINGS.md`). Editors: the desktop role editor (`D/ui/components/RoutingSettings/RoleEditor.tsx`), the generic "Node policy JSON" editor (`RoutingSettings.tsx:421`), the engine settings section (`D/ui/components/BackendRow/EngineSettingsSection.tsx`); no dedicated served/resident control exists (only bridge parsing, `D/electron/service-bridge/node-policy.ts:516-529`). SR records coordinator defaults the owner may veto: Q1 PC3 weak resident; MSI weak resident, medium on demand until NE memory bounds show both fit; Q3 preset values OWNER-CONFIRM (`sr5-rescope.md` lines 56, 212-214).

Options for "who chooses".

- A. The owner, per host, in that host's own documents; the Mac edits them remotely with target-local permission and local veto. Cost 0 (the design as landed). Risk: two documents must agree (profile `contextTokens` equal to the preset per-slot context; `maxResidentModels`/memory budget consistent with the resident list); nothing enforces the join until SR 5a-ii/5b.
- B. The origin (Mac) decides residency for peers from its tier policy. Contradicts the recorded decisions ("each node declares its own roles"; "local Off/pause wins"); no code path; not recommended.
- C. Automatic selection from NE memory bounds. SR 5b/5c compute fit but the plan fixes selection to "the approved host record"; automation would be a later SR amendment.

Recommendation: A, recorded in the frozen host record per host: preset values (`parallel`, `sleep-idle-seconds`, `models-max`), served list, resident list, roles, and the two admission numbers the record currently omits: `admission.maxResidentModels` (default 1, `N/nodepolicy/policy.go:438`) and `admission.memoryBudgetBytes` (default 0). With the SR Q1 default on MSI, a medium on-demand request beside the resident weak model needs `maxResidentModels` of at least 2, otherwise `fitLocked` plans eviction of the resident (`P/admission.go:468-485,623-639`) that SR 5c protection will refuse.

## Question 7 — regenerate after the response has started committing: cancel the running attempt, or refuse?

What the code does (VERIFIED). `workloadCtl.cancel(regenerate=true)` before commit aborts the attempt and excludes the node; after commit it falls through and cancels the whole request with reason "cancelled by request" (`P/cancel.go:115-141`; design comment `:8-14`). The broker answers `{ok: found}` (`B/workloadcancel.go:47-55`); the desktop card sends `regenerate` as a boolean on the same request (`D/ui/components/Workloads/WorkloadItemCard.tsx:55-61`); the wire result carries only `found` (`nodepolicy.WorkloadCancelResult`). The I plan keeps this as settled ("after commit it cancels the request") while its new `routing:cancel` must answer `conflict_committed` with no side effect; the debate records the mismatch (owner challenge 1).

Options.

- A. Keep cancel. Cost 0. Risk: the user who asked for another answer gets the current one truncated, OpenCode receives a cancelled stream, and nothing regenerates; the terminal reason says "cancelled by request", which hides why.
- B. Refuse after commit, one rule for both surfaces. Cost: `P/cancel.go:127-129` returns a distinct outcome; `WorkloadCancelResult` gains a bounded status (JSON-RPC change: proxy producer, `B/workloadcancel.go` relay, `services/nvpair-workload-manager/cancel.go`, `D/shared/utils/workloads.ts` and bridge, tests, generated docs); the card shows "already answering; cancel first". Risk: two clicks to get a fresh answer; small wire change on files I 4b/6 own.
- C. Keep cancel but record the honest reason: change the reason at `P/cancel.go:140` on the regenerate branch to "regenerate refused after commit; request cancelled" (one line plus a test). No wire change.

Roadmap. B belongs in I 4b-i/6 where the single transition and `routing:cancel` land (`debate-i-g.md` amendment 5). C can land any time.

Recommendation: B as the rule, implemented with I 6; C now as the interim so the workload record is honest. This argues against the settled I decision; see Owner challenges.

## Question 8 — manual node IDs: free-form bounded strings (current ruling) or a strict identifier grammar?

What the code does (VERIFIED). A manual node's id is its user name when present, else `"manual:" + address` (`services/nvpair-manual-nodes/manager.go:736-741`); `node/add` validates only a nonempty address (`:681-689`); the broker keys the discovery store by that id until the peer's host UUID is known (`B/broker.go:370-385`); the proxy's `node/add-manual` requires a nonempty id, port and address (`P/proxy.go:1581-1594`); the id becomes a candidate id and `scheduledOn`. J Task 1's projection validates identity fields with `^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$` and drops a frame as malformed otherwise (`B/workload_projection.go:102,120` -> `N/nodepolicy/routing_wire.go:33-39`; `audit-pa-j-b.md` UNV-3). Only the legacy migration and raw RPC can set a name (LOW-2). The coordinator ruling: "a bounded string kind that rejects control and format (Unicode Cf) characters".

Options.

- A. Bounded free-form (the ruling): valid UTF-8, 1-128 bytes, no control or Cf characters; apply at `node/add` (`manager.go:681`), at `node/add-manual` (`P/proxy.go:1590`), and accept the same kind for `scheduledOn` in the projection (`B/workload_projection.go:120`, a second kind beside "identity"). Cost: one shared validator (about 20 lines in N/nodepolicy) and three call sites with tests. Risk: a name shaped like a host UUID could collide with a real store key; always keying manual entries as `"manual:" + name` closes it (`manager.go:736-741`, one line, plus the broker's re-key path which already tolerates alias changes).
- B. Strict identifier grammar (`strictID`): reject names with spaces, slashes or non-ASCII at `node/add`. Cost: the same three call sites with the existing regex; plus a load-time repair for persisted entries that no longer validate (`services/nvpair-manual-nodes/store.go:30-52`). Risk: breaks any user who named a node with a space; IPv6 zone identifiers and bracketed literals in `"manual:" + address` also fail the grammar.

Roadmap. Either closes J UNV-3; A matches the ruling and J Task 2's "model ... no control characters" style; B would need a J amendment to its identity kinds. No milestone task depends on it.

Recommendation: A, as ruled, with the always-prefixed key. It is cheaper than B's migration and the only producers of free-form names are the migration and raw RPC.

## Question 9 — does OpenCode point at the facade port or a per-engine port?

What the code does (VERIFIED). The facade is the proxy's client entry per engine: llama.cpp on 8080 (`P/engines.go:262-266`; `N/engines/engines.go:163-167`), listening on all interfaces (`P/facade.go:486-488`; `P/routing_listener.go:191-197`) but answering plaintext only from loopback (`P/ingress.go:93-100`); peers arrive over pinned mTLS (`:112-131`). It lists the tier aliases on `/v1/models` (`P/tiers.go:529-556`) and resolves `weak`/`medium`/`strong` to concrete candidates on the node the request enters (`P/http_handler.go:655-667`). The engine's own port starts at 8081 (`E/manifests/llamacpp.json:24`); talking to it directly bypasses PAIR entirely: no tiers, no peer routing, no veto, no workloads, no telemetry, no shape or context rules, but native engine behavior for every shape. The hybrid listener port is policy-chosen and refuses inference (`N/nodepolicy/routing.go:219-234`; `P/routing_listener.go:302-305`). Which port each host's OpenCode uses is UNVERIFIED-1.

Options.

- A. Facade 8080 on the host where OpenCode runs: on MSI and PC3 their own facade; on the Mac its own facade with every engine Off, routing tiers to peers over mTLS (the acceptance plan's T5). Cost 0. Risk: the facade adds the tier path's retry bounds and admission; shapes llama-server accepts natively pass through unchanged today (LOW-1).
- B. Per-engine port 8081 directly (or another host's 8081 across the LAN). Cost 0. Risk: none of the milestone's routing, veto, outcome or telemetry clauses can be observed through OpenCode; a LAN engine port is plaintext and unauthenticated.
- C. The hybrid listener. Needs I Task 5; not a day-one option.

Roadmap. A is what every acceptance step assumes; it also decides amendment 3 of the debate ("Confirming the client's base URL settles amendment 3"), that is, whether I Task 5's listener cutover can be deferred.

Recommendation: A, recorded per host in the frozen record with the `engine:set-port` readback; the PC3 non-loopback provider entry (UNVERIFIED-1) needs the owner's explanation before T11 runs on that host, because a non-loopback PAIR facade would be refused.

## Further owner decisions the code assumes without a recorded answer

Each is a default the product ships and daily use will exercise; none has an entry in the recorded decisions available here. Values at `N/nodepolicy/policy.go:428-461` unless noted.

1. Admission per host: `maxResidentModels` 1, `memoryBudgetBytes` 0 (no limit), `switchModels` true (unload idle models to make room, `P/admission.go:468-485`), `queueTimeoutSeconds` 30, `switchTimeoutSeconds` 60. With one resident and one on-demand model on MSI the cap of 1 forces eviction or refusal (question 6).
2. Idle behavior for on-demand models: policy `unloadAfterMinutes` 0 and `stopEngineAfterMinutes` 0 (never), `startOnDemand` true, `wakeTimeoutSeconds` 90; but the llama.cpp adapter and manifest default `sleep-idle-seconds` to 300 (`E/serving_llamacpp.go:569`; `E/manifests/llamacpp.json:21`), so an on-demand model sleeps after five minutes and the next request pays a wake. SR fixes retention only for residents (`sr5-rescope.md` Q3).
3. Tier fallback: `allowStronger` true, `allowWeaker` false, `preferLoaded` "tier". A `weak` request may be answered by the medium model when no weak candidate is eligible, silently; with strong deferred, `medium` never escalates.
4. Concrete retry and wait bounds: five dispatch attempts, a ten-minute job deadline and a 120 s first-content cap (`P/http_handler.go:130-147`) define how long OpenCode waits before an error; never decided by the owner.
5. Pause defaults: `onActive` finish-active, `drainTimeoutSeconds` 120, `unloadModels` true, `stopEngines` false. Pausing a host unloads its models, including the resident ones (SR: "only reason `pause` bypasses this protection"), so resume pays a cold load.
6. Client entry topology: the only client entry is a loopback facade on a cluster host (question 9). A machine that is not a cluster member has no supported way to use the fleet; manual nodes are inference targets, not client entries. If the owner ever runs OpenCode from a non-member machine, that is a new surface (the deferred hybrid listener plus authentication), not a configuration.
7. Qualification evidence authoring: strict evidence is a hand-written `<appdir>/local-qualification.json` (`P/local_qualification.go:12,40,241`); there is no UI, CLI or generator, and the acceptance plan's T3 item 4 asks the owner to copy digests by hand. Whether that is acceptable, or whether a `qualification:record` helper belongs on the roadmap, is undecided; it bites as soon as question 4 moves daily traffic onto strict.
8. Who writes the Mac's `tiers` and `tierPolicy`: peers' roles are consumed automatically (`P/tiers.go:279-340`), so the Mac needs no tier entries for the daily path; but any tier entries present on the Mac still participate (`configuredAliases`, `P/tiers.go:541-556`). The owner has not said whether the Mac keeps an empty `tiers` map.

## Owner challenges (clearly separated; these argue against settled decisions)

1. I decision "Keep concrete regenerate compatibility: ... after commit it cancels the request" (question 7). The same click produces a cancelled stream through `workloads:cancel` and `conflict_committed` with no effect through the planned `routing:cancel`. One rule, refusal, is more honest and removes a test matrix; the interim one-line reason change keeps the record honest if the owner keeps the current rule.
2. The milestone sentence "the user sees only delay or pause-resume" (question 5). With SC-b unplanned and expensive, the owner should either restate the day-one clause as "one honest interruption error, no silent truncation" or schedule SC-b explicitly after I 4b; leaving it implicit makes every acceptance run end with an open clause.

## Verification record (this environment)

- `go vet ./...`: `services/nvpair-proxy`, `services/nvpair-ui-broker`, `services/nvpair-engine-manager`, `services/nvpair-service`, `services/nvpair-manual-nodes`, `services/shared`: exit 0 each (GOTOOLCHAIN=auto resolved Go 1.25 on a 1.24 host).
- `go test -count=1 ./nodepolicy/ ./engines/` in `services/shared`: ok; `go test -count=1 ./...` in `services/nvpair-manual-nodes`: ok.
- Not run: `services/tests` (not even listed), `-race`, desktop checks, anything Windows-native; no engine, model or service started.
- Plan and inventory text was read with `git show` from the named branches; nothing outside `plans/reviews/owner-decisions.md` was written in this worktree.
