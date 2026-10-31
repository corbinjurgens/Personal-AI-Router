<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution plan: Phase 04: node-centric desktop with Models and Profiles tabs, Halt, and role-driven activation

## Goal

Overall goal of the batch (`README.md` in this folder, sections 2 to 5): make
PAIR match the owner's model of role to profile to engine on their three
machines, so that configuring the cluster from one node just works.

This phase's goal (README section 3 T-U, section 5 row 04, owner decisions D4
and D5 at their defaults): the service starts an engine because a profile on
this node needs it, not because a user saved it On, and several engines can run
at once; engine locks and the engine switch disappear from every user-facing
surface; a peer may resume a pause that a peer made, while a local pause or Halt
stays local; peers learn of a pause at once instead of after the 15 s poll. On
the desktop, each node has one settings page, the Overview card shows roles and
engine running state instead of switches and padlocks, the window has the tabs
Overview, Models, Profiles and Settings, and the toolbar has a large local-only
Halt button with a Resume counterpart. Finished state: the three machine checks
in `## Verification (whole-plan)` pass on the Mac and MSI (and Joes for deploy).

## Change scope

Intent: the owner sees nodes, roles, profiles and models, never engine toggles,
padlocks or a switch transaction; engines start when a profile needs them.
Scope: Structural change — activation authority moves from saved engine intent
to profiles (proxy, broker, engine-manager bound path), the engine lock is
deleted end to end, and the desktop information architecture is rebuilt from
existing components; leave routing order, scheduling, the estimate and Validate
(phase 05), smart routing (phase 06) and the engine switch's Go owner unchanged.
Basis: README section 5 row 04 and decisions D4/D5 (owner defaults); the brief's
explicit instruction to remove locks (L4 sized at about 60 source and 35 test
files) and to compose the desktop from existing pieces.
Compatibility: pre-release fork, no compatibility work. Change wire and file
formats in place: no version bumps, capability markers, compatibility shims or
migrations. The `locked` key disappears from `engine-state.json` readers; an
old file that still carries it must load (the decoder ignores the key) — that is
tolerance of a removed key, not a migration. Preserve: `localOnly` on profiles,
the immediate-pause semantics landed by the stop/pause Todo (Tasks 1-7), the
request owner and commitment boundary in the proxy, saved engine intent storage
(re-purposed as Start with PAIR), `engineBindings` in node policy.
Non-goals: deleting the engine switch journal and its Go owner; force-stopping
apps PAIR did not start (owner Q5 stands); a role lock; retiring
`engineBindings`; new routing or scheduling logic in TypeScript; redesigning the
role editor or the profile editor (phases 02 and 03 own them); a tray popup
redesign.

## Entry conditions

Record evidence of every earlier phase in
`plans/2026-10-31-expectation-gaps/00-index.md` (the phase row ends with
` — done <sha>`). If a required condition is absent, do not implement the
missing prerequisite here: stop the dependent tasks and report which condition
failed; tasks without that condition continue.

| Needed by | Condition | How to observe it | If absent |
| --- | --- | --- | --- |
| Tasks 2, 5, 6, 7, 9 | Phase 02 landed (planned in `02-profiles.md`): node-policy profiles carry an `id`, an `engine`, a model identity and a `load`; `Profile.ServedModel()` is the profile id for a profile with a load; `Policy.ProfileByID` and `Policy.ProfileFor(engine, model)` (matching the served model) exist; roles map to ordered lists of profile ids; the advertisement publishes `AdvertisedProfile{ID, Name, Engine, ServedModel, ModelIdentity, ...}`. | Index row 02 ends with ` — done <sha>`; `git grep -nE "func \(pr Profile\) ServedModel|func \(p Policy\) ProfileByID|AdvertisedProfile" services/shared/nodepolicy` shows all three. | Stop Tasks 2, 5, 6, 7, 9 and 10. |
| Tasks 8, 10 | Phase 03 landed: role names are strings (no `ROLE_TIERS`/`TierOrder` closed set); `RoleEditor` edits role name to profile list; `NodeDeclaredRoles` no longer iterates fixed tiers. | Index row 03 ends with ` — done <sha>`; `git grep -n "ROLE_TIERS" desktop/src` prints nothing. | Stop Tasks 8 and 10. |
| Task 9 | Phase 01 landed: inventory rows (`ModelItem` in `desktop/src/shared/types/engines.ts`) carry the structured model identity (repository, file or quant label, content id) and the desktop has one identity-tag renderer. | Index row 01 ends with ` — done <sha>`; `01-identity-and-unblock.md` close-out names the field and the tag component. | Stop Task 9. |
| Task 10 | Phase 02's deploy exists. Phase 02 decided (its P11) that deploy is a desktop orchestration over existing RPCs, not a new JSON-RPC method: `deployProfileChange` and `targetModelFor` in `desktop/src/shared/utils/node-policy-model-copy.ts`, driven by the "Deploy profile" action of `ModelSettingsCopy.tsx` (copy with `engine:copy-model` when the target lacks the `contentId`, then target preview and apply). | `git grep -nE "deployProfileChange|targetModelFor" desktop/src` shows both, and `git grep -n "Deploy profile" desktop/src/ui/components/ModelSettings/ModelSettingsCopy.tsx` prints a line. | Stop Task 10. |

Tasks that may start before phase 02 lands: Tasks 1 and 3 (availability,
discovery advertisement, toolbar, AppBar and tray). Task 1 shares
`services/shared/noderec/noderec_test.go` with phase 02 Task 1; whichever lands
second rebases that test file. Task 4 needs only Tasks 1 and 3, but it edits engine-settings
desktop files that phase 02 also edits, so start it after phase 02 lands. Tasks
2, 5, 6 (activation and lock removal) and 7 (settings page) need only phase 02.
Tasks 8 and 10 also need phase 03. Task 9 needs phase 01 and Task 7.

## Context the executor needs

Repository: base the branch on current `origin/integration/join33` (`4eccd05d`
or later). Every path below was verified with `git ls-files` at `4eccd05d`;
re-check symbols with `git grep` before editing. Follow `AGENTS.md` and
`.cursor/rules/{services-change-propagation,service-contracts,renderer-stores,type-discipline,no-legacy-fallbacks,engines-vs-backends-naming}.mdc`.
Desktop UI uses Kaizen UI (`.agents/skills/kaizen-ui/SKILL.md`: KUI components
over divs, semantic tokens, no emojis). New files start with the SPDX header
(`node scripts/spdx-headers.mjs` checks). `desktop/docs/services-api.md` is
generated by `npm run service-contracts:write` and is never hand-edited.

Verified facts (claim ids from `investigation/`):

- Activation (verify-A A6, L4-C11..C13). `services/nvpair-engine-manager/wakesleep.go`
  `Wake` starts only saved-On engines (`errIntentOff`). The bound path is the only
  path that starts a saved-Off engine: proxy `services/nvpair-proxy/bound_engine.go`
  `boundModel`/`needsBoundPreparation` and `services/nvpair-proxy/self.go`
  `boundSelfCandidate`/`nodeOffersBinding` read only `Policy.EngineBindings`;
  broker `services/nvpair-ui-broker/bound_engine.go` `boundAuthorityLocked` requires
  an `EngineBindings` entry, and `runBoundEngine` refuses with
  `engine-switch-busy` when more than one other engine runs, switches away from
  exactly one (`acceptAutomaticEngineSwitch`, dwell `boundDwellUntilLocked`,
  `waitBoundSwitch`), and cold-activates only when none runs
  (`activateBoundCold`). Engine-manager `services/nvpair-engine-manager/bound_engine.go`
  `activateBoundEngine`'s `guard` returns `switchBusyError` while any other engine
  is running, stopping or reclaiming. `StartWith` (`lifecycle.go`) calls
  `setDesiredEnabled(engine, true, origin)` after every start, so bound activation
  flips saved intent On. `dormantSelfCandidate` (`self.go`) already covers
  profile models but requires saved On and `Idle.StartOnDemand`. Plain Start,
  Wake and `RestoreEnabled` have no cross-engine exclusion.
- Memory guard (L4-C15): `services/shared/nodepolicy/policy.go`
  `Admission.MaxResidentModels` defaults to 1 across all engines and is enforced
  in `services/nvpair-proxy/admission.go` and `residency.go`; with
  `switchModels` the proxy unloads LRU idle models to make room.
- Engine switch callers (grep at `4eccd05d`): broker `engineswitch.go`
  `acceptEngineSwitch` (manual, `engine:switch`), `acceptAutomaticEngineSwitch`
  (only caller `runBoundEngine`); switch admission/snapshot helpers
  (`pendingSwitchLocked`, `switchSnapshotLocked`) are also used by
  `enginesettings.go`, `enginesettings_recovery.go`, `modelsettings.go`,
  `build_apply.go` and `bound_engine_offers.go`; recovery in
  `engineswitch_recovery.go`; engine-manager `switchexecutor.go`
  `automaticSwitchSource`; TUI `services/nvpair-tui/ui/engine_switch.go`;
  desktop `EngineSwitchControls.tsx`, `engine-switch.store.ts`,
  `electron/service-bridge/engine-switch.ts`, channels `engines:switch*`.
- Locks (L4-C10, C12, Q5). Owners: engine-manager `enginelock.go`
  `setEngineLockedCommand`, `desiredstate.go` `locked`/`setLocked`,
  `nodeprojection.go` `activationVeto`/`remoteVeto` lock branch, `status.go`/
  `wakesleep.go` `Locked` status, `owner_evidence.go` lock digests; build-apply
  evidence `services/shared/buildapply/engine_owner.go` `LockedByEngine`,
  `services/nvpair-ui-broker/build_apply.go`, `services/nvpair-service/build_apply_run.go`;
  wire `-32013`/`engine_locked`/`engine-locked` in `services/shared/nodepolicy/wire.go`,
  `nodepolicy/bound_engine.go`, `enginesettings/switch.go`; TUI
  `ui/engineswire.go`, `ui/engine_switch.go`, `ui/nodedetail.go`,
  `ui/settingswire.go`; desktop listed in Task 4. `localOnly` on a profile is a
  different, generation-settings-only lock and stays (L4-C20).
- Pause (verify-B B2, L4-C1..C6). `services/nvpair-ui-broker/availability.go`
  `setAvailabilityRequestAs` refuses every peer-origin request with
  `localVetoError` while `policy.go` `managementBlockedLocked()` is true (any
  pause, either origin), and refuses immediate mode unless caller is empty and
  origin local. Peers learn `pausedBy` only from `/v1/models` `policy`, read by
  the scanner's `refreshPeersLoop` every `defaultPeerRefreshInterval` (15 s,
  `services/nvpair-node-scanner/daemon.go`) or by browse-event enrichment when a
  node's raw mDNS record changes; the proxy drops advertised paused peers
  (`proxy.go`, `inference && !self && n.PausedBy != ""`). The broker's
  `notifyAvailability` reaches local clients only. The scanner registry
  (`registry.go` `register`) ignores per-service TXT today; the broker reaches the
  scanner through `discovery.go` `scannerProcess.pushRegister`.
- Desktop (L5, verify-B B7). `MainApp/ContentBar.tsx` `TAB_ITEMS` holds Overview
  and Settings and the toolbar holds Endpoints and Add node; `AppBar/AppBar.tsx`
  has an unused `slotTitleRight`; `shared/types/overview.ts` `OverviewTab`;
  `ui/stores/overview-ui.store.ts` `activeTab`, `settingsSubTab`, `focusNodeId`,
  `focusNodeEngineSettings` (callers: `ClusterSettings/ActiveNode.tsx` "Edit node",
  `Welcome/WelcomeModal.tsx`, `hooks/useOverviewCommands.ts` tray `focus-node`,
  consumer `NodeList/NodeCardDetails.tsx`). Settings sub-tabs in `Settings.tsx`
  `SETTINGS_NAV_ITEMS` and `ui/types/settings-window.ts` `SettingsWindowTab`
  (`cluster`, `routing`, `service`). "Routing & resources" (`RoutingSettings.tsx`)
  mounts `NodeAvailabilityControls`, `RoleEditor`, `ModelEngineBindingsEditor`,
  `ModelSettingsPanel`, the "Advanced policy JSON" textarea, `BulkNodeConfiguration`,
  `RoutingHistory`. The card (`NodeCardDetails.tsx`) mounts `NodeEnginesInline`
  (switch, `EngineLockControl`, `CredentialControl`, install), `NodeDeclaredRoles`,
  `NodeAvailabilityControls`, `NodePerformance`, and the expandable
  `NodeEngineSettings` (`EngineSwitchControls` plus one `BackendRow` per engine:
  `BackendHeader` with a second switch, `BackendUpdateBanner`, `ModelSection` →
  `ModelManager`, `EngineSettingsSection`, `BackendFooter`). `engine-models.store.ts`
  holds every node's inventory keyed `nodeId:engineType` and replaces its `Map`
  on every change (immutable). Immediate pause in the UI is
  `NodeAvailabilityControls.tsx` "Stop local models now" (local only,
  `requestAvailability(targetId, 'paused', 'immediate')`); the tray item is in
  `electron/tray.ts` `buildMenu`, its label source `electron/service-bridge/node-availability.ts`.
  `shared/types/node-policy.ts` `NodeAvailabilityRequest.pauseMode` comment says
  immediate is "currently unavailable" (stale, B2).
- Desktop checks: `desktop/scripts/verify-service-contracts.ts` parses Go
  sources; it fails on a Go notification the desktop does not reference unless
  `desktop/docs/service-contract-exceptions.json` `ignoredMethods` lists it with a
  reason, and on a stale generated `services-api.md`.

Not verified here (no engines, no machines): every native outcome, mDNS
delivery between the owner's machines, and the timing of peer exclusion.

## Decisions (settled — do not revisit)

- Remove engine locks and the engine switch UI; replace On/Off toggles with
  role-driven activation plus install/update, running state and a per-engine
  "Stop now" in the node settings page (README D4).
- Halt is the existing immediate local pause, renamed and moved to the toolbar,
  local-only; apps PAIR did not start stay "Pause incomplete" rather than being
  force-stopped (README D5; owner Q5 of 2026-10-28). Not force-stop of external
  apps, because the owner kept Q5.
- Push a pause hint to peers instead of waiting for the 15 s poll (README D5).
- A peer-paused node may be resumed by a managing peer; a locally paused or
  halted node only locally (README section 3 T-U; this supersedes vault
  `decided.md` 2026-10-23 "Paused nodes require local Resume" for peer-origin
  pauses only).
- Keep `localOnly` on profiles (README T-U; owner 2026-10-28 "inference lock
  stays generation-only").
- Desktop is a client: no routing or scheduling logic in TypeScript; stores
  follow `renderer-stores.mdc`; no `as` casts (`type-discipline.mdc`); deleted
  components take their exports and tests with them.
- Pre-release fork: change formats in place; no compatibility work.

## Decisions made while writing this plan

- Concurrent engines: yes. Bound activation no longer treats another running
  engine as a conflict; it cold-starts the target beside it. The resident-model
  cap (`admission.maxResidentModels`, default 1, and `memoryBudgetBytes`) stays the
  only memory guard; with the default cap, loading on the second engine unloads
  the first engine's idle model through the existing LRU path, and the idle
  policy later stops the idle engine. Not "stop the other engine first", because
  the owner wants engines to coexist when resources allow and the cap already
  protects memory.
- The automatic engine switch is removed from bound activation. The switch
  transaction's Go owner stays (engine-manager `switchexecutor.go`,
  `switchrelay.go`, `switchremote.go`, `switchwatch.go`; broker `engineswitch*.go`;
  `services/shared/enginesettings/switch.go`) because engine-settings apply,
  model-settings apply, build-apply and bound offers use its admission and
  reservation helpers and `engineswitch_recovery.go` recovers journal rows. The
  TUI switch screen (`services/nvpair-tui/ui/engine_switch.go`) stays as the
  operator and recovery surface; the desktop loses every switch control.
- A profile on this node is activation authority: when the requested model is
  the served name of a profile whose engine is this facade's engine, the proxy
  takes the bound preparation path whatever the saved intent. `engineBindings`
  stays for models that are not profiles and is edited through Advanced policy
  JSON; its editor (`ModelEngineBindingsEditor.tsx`) is deleted. Retiring
  `engineBindings` is left for a later plan.
- Saved engine intent is re-purposed as "Start with PAIR" (autostart): it is read
  by `RestoreEnabled`, `Wake` (Resume and idle wake) and resident reconciliation
  as today, and written only by a new local method `engine:set-autostart` plus
  the existing install/uninstall reset and switch-journal writes. `engine:start`,
  `engine:stop`, `engine:remote-start`, `engine:remote-stop` and bound activation
  stop writing it. Start with PAIR is local-only in this phase; a peer's page
  shows it read-only with "Start with PAIR is set on the device running the
  engine." (the CORS precedent in `EngineSettingsSection.tsx`).
- Pause-hint carrier: the advertisement path. The broker hands the live
  availability to its scanner, which publishes it as a node-level mDNS TXT key
  `avail`; a changed raw record makes each peer's scanner re-enrich that node's
  `/v1/models` (which carries `pausedBy`) at once. Not the workload broadcaster,
  because peers' proxies read pause state from the scanner directory and the
  broadcaster would need a new frame kind plus a receiver-to-proxy path that does
  not exist; the advertisement path adds only a change trigger to data peers
  already consume. Limitation: it depends on multicast reaching peers; the 15 s
  sweep stays as the fallback and the paused node's own 503 stays the
  correctness guarantee.
- Peer Resume rule: a peer-origin request for `available` is accepted when the
  persisted pause origin is `peer`, no local pause or Halt request is pending
  (`localPauseRequested` false) and no build-apply operation runs; every other
  peer write during a pause keeps `localVetoError`. A local Halt over a peer
  pause records origin `local`, which makes it local-only.
- Halt copy and confirm: the toolbar button reads "Halt"; it opens
  `ConfirmModal` titled "Halt this PC?" with body "This PC stops serving models
  now. Requests that have not started answering move to other nodes; answers
  already streaming end with an error. Other nodes cannot undo this." and buttons
  "Halt" and "Cancel". This supersedes the stop/pause Todo Q6 "one click" for the
  renderer only. The tray item "Halt this PC" stays one click (a tray menu is
  already a deliberate two-step action). The AppBar badge shows "Paused on this
  PC" (origin local), "Paused by a peer" (origin peer) or "Pausing"; the broker
  does not report whether a local pause was immediate, so the UI does not claim
  "Halted" as a state.
- Node settings page: an in-window view of the Overview tab, opened by
  `openNodeSettings(nodeId)` in `overview-ui.store.ts` (replaces
  `focusNodeEngineSettings`) from the card's settings icon, Cluster "Edit node",
  the welcome flow and the tray `focus-node` command; KUI `PageHeader` with
  `Breadcrumbs` "Overview" › node name. The Settings tab keeps Cluster and Service;
  "Routing & resources" is removed. Routing history appears on this PC's page only
  (it is this router's history).
- Deploy to node reuses phase 02's desktop deploy orchestration (its P11: no new
  JSON-RPC method), extracted into a hook shared by `ModelSettingsCopy` and the
  Profiles tab, instead of the deploy RPC the batch brief assumed.
- `BulkNodeConfiguration` and the store's bulk state are deleted in Task 7;
  Deploy to node (Task 10) replaces them. Between the two tasks bulk role edits
  are unavailable.
- Models tab grouping: group by repository (model identity), then by quant label;
  each quant row lists the nodes holding it, flags "Different file" when two
  nodes' content ids differ, and rows without an identity go to a final
  "Unidentified" group under their raw name. Grouping is a pure presentation
  utility with its own unit test.
- Per-engine model lists leave the card in Task 7 and stay in the node settings
  page's engine section until Task 9 replaces them with the Models tab.
- Packaging: ten tasks. Go tasks 1, 2 and 6 and desktop tasks 3, 4, 7 to 10 are
  separate; Task 5 (autostart) and Task 6 (lock wire removal) are the contract
  tasks that change Go and desktop together. Shared docs force the order
  1 → 2 → 5 → 6 on the Go side.

## Tasks

### Task 1 — Configured Pause is peer-reversible and peers learn of a pause at once
**Files:** services/nvpair-ui-broker/availability.go, services/nvpair-ui-broker/availability_test.go, services/nvpair-ui-broker/policy.go, services/nvpair-ui-broker/policy_test.go, services/nvpair-ui-broker/discovery.go, services/nvpair-ui-broker/README.md, services/shared/noderec/noderec.go, services/shared/noderec/noderec_test.go, services/nvpair-node-scanner/registry.go, services/nvpair-node-scanner/registry_test.go, services/nvpair-node-scanner/daemon.go, services/nvpair-node-scanner/daemon_test.go, services/nvpair-node-scanner/README.md, services/tests/stop_pause_controls_test.go, docs/node-availability.mdx, desktop/docs/service-contract-exceptions.json, desktop/docs/services-api.md
**Depends on:** none
**Complexity:** judgment
**Do:**
1. Peer Resume. In `availability.go` `setAvailabilityRequestAs`, replace the
   blanket `origin == nodepolicy.PausedByPeer && r.managementBlockedLocked()`
   refusal with: refuse unless the request is `state == Available` and a new
   `policy.go` helper `peerResumableLocked()` returns true. The helper is true
   only when the persisted pause origin (the value advertised as `pausedBy`) is
   `peer`, `r.localPauseRequested` is false and `r.buildApplyOperation == ""`.
   Keep `managementBlockedLocked()` itself unchanged for every other peer
   mutation, and update its comment to state the one exception. An accepted peer
   Resume runs the same transition a local Resume runs (`resumeNode`,
   `wakeAfterResume`). Confirm that an immediate local request accepted over a
   peer pause records origin `local` (`acceptAvailabilityLocked`); fix it if not.
2. Tests in `availability_test.go`: `TestPeerResumesPeerPause` (peer pauses,
   same peer resumes, state `available`), `TestPeerCannotResumeLocalPause`
   (local configured pause, peer Resume gets `localVetoError`),
   `TestPeerCannotResumeAfterLocalHalt` (peer pause, local immediate pause,
   peer Resume refused), `TestPeerRepeatPauseStillRefused`. Extend
   `services/tests/stop_pause_controls_test.go` with one cross-process case
   where a granted peer resumes its own pause.
3. Pause hint. Add a node-level availability field to `noderec.NodeRecord`
   emitted by `TXT()` as key `avail` with the live value (`available`,
   `draining`, `paused`) and read by `ParseTXT`; omit the key when empty. Add an
   internal broker-to-scanner method `discovery:set-availability`
   (constant in `noderec.go` next to `MethodUpdateTXT`, params
   `{"availability": "<value>"}`). The scanner registry stores it, `record()`
   includes it, and the daemon re-advertises with `responder.UpdateTXT` only on a
   real change (the same pattern as `setIdentity`). The broker calls it through
   its scanner process (`discovery.go`, next to `pushRegister`) whenever the live
   availability it pushes to the proxy changes, including at immediate
   acceptance (`runAcceptedAvailability`) and on Resume, and once after the
   scanner (re)starts. Never block an availability transition on this call; log
   and continue on failure.
4. Peer side: confirm in `daemon.go` that a browse event for a node whose raw
   record changed re-runs enrichment that refetches `/v1/models` (and so
   `policy.pausedBy`). If it does not refetch models for a TXT-only change, make
   it refetch models for that node when the `avail` value changed. Keep the 15 s
   `refreshPeersLoop` unchanged as the fallback.
5. Tests: `noderec_test.go` `TestNodeRecordAvailabilityTXTRoundTrip`;
   `registry_test.go` `TestRegistryAvailabilityChangesRecord` (same value → no
   change, new value → change); a broker test `TestAvailabilityChangeSendsScannerHint`
   in `availability_test.go` using the existing scanner test double.
6. Docs: `docs/node-availability.mdx` sections "Local and peer origin",
   "Resume and restart", "Remote management while paused" and "How peers learn a
   node is paused" state the peer-Resume rule, the name Halt for the immediate
   local pause, and the `avail` hint with the 15 s fallback; broker `README.md`
   pause fence and peer-learning paragraphs; scanner `README.md` TXT keys and
   the new internal method.
7. Run `cd desktop && npm run service-contracts:check`. If it reports
   `discovery:set-availability`, add an `ignoredMethods` entry: "Private
   broker-to-scanner availability hint; never a renderer or peer request." Then
   run `npm run service-contracts:write`.
**Verify:** In `services/nvpair-ui-broker`, `go test -run 'TestPeerResumesPeerPause|TestPeerCannotResumeLocalPause|TestPeerCannotResumeAfterLocalHalt|TestPeerRepeatPauseStillRefused|TestAvailabilityChangeSendsScannerHint' ./...` passes, then `go test ./... && go vet ./...` exit 0. In `services/shared`, `go test ./noderec/...` passes. In `services/nvpair-node-scanner`, `go test ./... && go vet ./...` exit 0. In `services/tests`, `go vet ./...` exits 0. In `desktop`, `npm run service-contracts:check` exits 0.

### Task 2 — Profile-backed requests start their engine beside running engines
**Files:** services/nvpair-proxy/bound_engine.go, services/nvpair-proxy/bound_engine_test.go, services/nvpair-proxy/self.go, services/nvpair-proxy/proxy.go, services/nvpair-proxy/tiers_test.go, services/nvpair-proxy/README.md, services/nvpair-proxy/spec.md, services/nvpair-ui-broker/bound_engine.go, services/nvpair-ui-broker/bound_engine_test.go, services/nvpair-ui-broker/bound_engine_offers.go, services/nvpair-ui-broker/bound_engine_offers_test.go, services/nvpair-ui-broker/engineswitch.go, services/nvpair-ui-broker/engineswitch_test.go, services/nvpair-ui-broker/README.md, services/nvpair-ui-broker/ENGINE_SETTINGS.md, services/nvpair-engine-manager/bound_engine.go, services/nvpair-engine-manager/bound_engine_test.go, services/nvpair-engine-manager/switchexecutor.go, services/nvpair-engine-manager/switchexecutor_test.go, services/nvpair-engine-manager/README.md, services/nvpair-engine-manager/spec.md, services/tests/engine_binding_test.go, docs/model-engine-bindings.mdx
**Depends on:** 1 — both edit `services/nvpair-ui-broker/README.md`; also needs the phase 02 entry condition
**Complexity:** judgment
**Do:**
1. Proxy. Make a profile count as a binding: `boundModel` (`bound_engine.go`)
   and `boundSelfCandidate` (`self.go`) return true when the current policy has
   a profile whose engine is `f.profile.Name` and whose served name (phase 02:
   the profile id) equals the requested model after `normalizeModel`, in
   addition to the existing `EngineBindings` match. Use phase 02's
   `Policy.ProfileFor(f.profile.Name, model)` (it matches `Profile.ServedModel()`);
   do not add a second lookup. For peers, `nodeOffersBinding` treats the peer's
   advertised `AdvertisedProfile` whose `Engine` is this engine and whose
   `ServedModel` equals the model as an offer, so a peer's stopped engine is a preparation candidate;
   keep the existing `PausedBy`, `PolicyRevision` and `ServedModels` checks.
   Exclude profile models from `dormantSelfCandidate` so one engine is never
   offered twice for the same request. Keep `resolveConcreteCandidates`
   ordering (warm before preparation).
2. Broker. `boundAuthorityLocked` accepts the request when the policy has such a
   profile or an `EngineBindings` entry (keep the `ServedModels` check).
   `bound_engine_offers.go` offers profile-backed (engine, served name) pairs the
   same way it offers bindings, unless phase 02's advertisement already
   publishes them to peers — then read that and add no second list. In
   `runBoundEngine`, delete the `len(running) > 1` refusal and the whole
   automatic-switch branch (protected-resident `no-fit`, dwell, `acceptAutomaticEngineSwitch`,
   `waitBoundSwitch`); when the target is not running, always take the cold
   activation branch. After the edit, delete every function and type that
   `git grep` shows has no remaining non-test caller (expected:
   `acceptAutomaticEngineSwitch`, `automaticSwitchSubmission`, `boundDwellUntilLocked`,
   `waitBoundSwitch`; in engine-manager, `automaticSwitchSource` and the
   automatic-only branches that call it). Keep `acceptEngineSwitch` and the
   journal and recovery code.
3. Engine-manager. In `activateBoundEngine`'s `guard`, stop treating another
   engine's `running` state as busy. Keep refusing when the target itself is
   `stopping`, and keep the other-engine `reclaimFailed`/`reclaimPID` check only
   when that engine's reclaim targets the same port as the target (otherwise
   drop it). Keep `admitBoundEngineMutation`.
4. Tests (add): proxy `TestProfileModelNeedsBoundPreparation` (profile on this
   node, engine saved Off and stopped → `needsBoundPreparation` true) and
   `TestPeerProfileOfferIsPreparationCandidate` in `bound_engine_test.go`;
   broker `TestBoundEngineColdActivatesBesideRunningEngine` (another engine
   running → cold activation, no switch submitted) in `bound_engine_test.go`;
   engine-manager `TestActivateBoundEngineAdmitsWithOtherEngineRunning` in
   `bound_engine_test.go`. Rewrite each existing test that asserts
   `engine-switch-busy` for two running engines to assert cold activation beside
   the running engine; delete a test only when its sole subject is the removed
   automatic switch.
   Extend `services/tests/engine_binding_test.go` with a profile-backed request
   that starts a stopped engine while another engine runs.
5. Docs: `docs/model-engine-bindings.mdx` opens with "A profile starts its engine
   when a request needs it" and states that bindings remain for models that are
   not profiles and that engines run concurrently under the resident-model cap;
   update the one-running-engine statements in proxy `spec.md`/`README.md`,
   broker `README.md`/`ENGINE_SETTINGS.md` ("Engine switching": manual only, TUI)
   and engine-manager `README.md`/`spec.md`.
**Verify:** `go test -run 'TestProfileModelNeedsBoundPreparation|TestPeerProfileOfferIsPreparationCandidate' ./...` passes in `services/nvpair-proxy`; `go test -run TestBoundEngineColdActivatesBesideRunningEngine ./...` passes in `services/nvpair-ui-broker`; `go test -run TestActivateBoundEngineAdmitsWithOtherEngineRunning ./...` passes in `services/nvpair-engine-manager`; then `go test ./... && go vet ./...` exit 0 in each of those three modules and in `services/shared`; `go vet ./...` exits 0 in `services/tests`; `git grep -n "acceptAutomaticEngineSwitch" -- services` prints nothing.

### Task 3 — Toolbar Halt and Resume with AppBar and tray state
**Files:** desktop/src/ui/components/MainApp/HaltControl.tsx (new), desktop/src/ui/components/MainApp/ContentBar.tsx, desktop/src/ui/components/MainApp/MainApp.tsx, desktop/src/ui/components/AppBar/AppBar.tsx, desktop/src/ui/components/NodeList/NodeAvailabilityControls.tsx, desktop/src/electron/tray.ts, desktop/src/electron/service-bridge/node-availability.ts, desktop/src/shared/types/node-policy.ts, desktop/tests/modular/halt-control.test.tsx (new), desktop/tests/modular/node-availability-controls.test.tsx, desktop/tests/modular/node-availability.test.ts, desktop/tests/modular/stop-pause-contract.test.ts, desktop/docs/architecture.md, desktop/docs/frontend-api.md, README.md, docs/overview.mdx
**Depends on:** none
**Complexity:** judgment
**Do:**
1. `HaltControl.tsx`: a component for this PC only. It reads this PC's entry
   from `useNodeConfigurationStore` (`availability` keyed by the local target id
   used by `NodeAvailabilityControls`) and starts `watchNodeAvailability` for it
   while mounted. Available → a large KUI `Button` (`kind="primary"`,
   `color="danger"`, `size="medium"`) labelled "Halt" that opens
   `ConfirmModal` with the copy in "Decisions made while writing this plan";
   confirming calls `requestAvailability(<local target id>, 'paused', 'immediate')`.
   Paused (either origin) → a `Button` "Resume" calling
   `requestAvailability(<local target id>, 'available')`. Draining → "Halt"
   stays enabled under the same escalation rule as today's `canStop` in
   `NodeAvailabilityControls.tsx` (export that predicate and reuse it; do not
   copy it). Unknown availability or a disconnected service → disabled "Halt".
   Show the request error under the button as `Text` with `role="status"`.
2. `ContentBar.tsx`: mount `HaltControl` as the first item of the right-hand
   `Flex`, before Endpoints and Add node.
3. AppBar: `MainApp.tsx` passes `slotTitleRight` to `AppBar` with a KUI `Badge`
   for this PC: "Paused on this PC" (paused, `pausedBy` local), "Paused by a peer"
   (paused, `pausedBy` peer), "Pausing" (draining); nothing when available.
4. `NodeAvailabilityControls.tsx`: rename the "Stop local models now" button to
   "Halt" and route it through the same `ConfirmModal`; keep it local-only.
5. Tray: in `tray.ts` `buildMenu`, rename "Stop local models now" to
   "Halt this PC"; in `node-availability.ts`, add a disabled first status line
   "This PC is paused" when paused and "This PC is pausing" when draining, and
   keep the Pause/Resume item labels.
6. `shared/types/node-policy.ts`: replace the stale comment on `pauseMode` with
   "Omission uses saved drain options. Immediate (Halt) is accepted only from a
   client on this node."
7. Tests: `halt-control.test.tsx` covers available → confirm → immediate
   request; cancel sends nothing; paused shows Resume and sends `available`;
   unknown is disabled. Update label assertions in
   `node-availability-controls.test.tsx`, `node-availability.test.ts` and
   `stop-pause-contract.test.ts`.
8. Docs: replace "Stop local models now" with "Halt" in `README.md`,
   `docs/overview.mdx`, `desktop/docs/architecture.md`, `desktop/docs/frontend-api.md`,
   describing the toolbar placement and confirm.
**Verify:** In `desktop`: `npx vitest run --project unit tests/modular/halt-control.test.tsx tests/modular/node-availability-controls.test.tsx tests/modular/node-availability.test.ts tests/modular/stop-pause-contract.test.ts` passes; `npm run lint && npm run typecheck && npm run test:unit && npm run dead-code:check` exit 0; `git grep -n "Stop local models now" -- desktop/src desktop/docs docs README.md` prints nothing.

### Task 4 — Desktop drops engine locks and the engine switch UI
**Files:** desktop/src/ui/components/NodeList/NodeEnginesInline.tsx, desktop/src/ui/components/NodeList/EngineSwitchControls.tsx (delete), desktop/src/ui/components/NodeList/NodeEngineSettings.tsx, desktop/src/ui/components/BackendRow/BackendRow.tsx, desktop/src/ui/components/BackendRow/EngineSettingsSection.tsx, desktop/src/ui/stores/engine-switch.store.ts (delete), desktop/src/ui/stores/pending-actions.store.ts, desktop/src/ui/api/engine-api.ts, desktop/src/shared/types/engine-api.ts, desktop/src/shared/types/engine-switch.ts, desktop/src/shared/types/engines.ts, desktop/src/shared/types/node-policy.ts, desktop/src/shared/types/ws-channels.ts, desktop/src/shared/constants/service-errors.ts, desktop/src/electron/service-bridge/empty-handlers.ts, desktop/src/electron/service-bridge/engine-switch.ts, desktop/src/electron/service-bridge/modular-state.ts, desktop/src/electron/service-bridge/modular-supervisor.ts, desktop/src/electron/service-bridge/node-policy.ts, desktop/tests/modular/engine-lock-contract.test.ts (delete), desktop/tests/modular/node-engine-lock-controls.test.tsx (delete), desktop/tests/modular/engine-switch-controls.test.tsx (delete), desktop/tests/modular/engine-switch-store.test.ts (delete), desktop/tests/modular/engine-switch.test.ts (delete), desktop/tests/modular/node-configuration.test.ts, desktop/tests/modular/node-policy.test.ts, desktop/tests/modular/remote-management.test.tsx, desktop/tests/modular/service-error-codes.test.ts, desktop/tests/modular/engine-settings-state.test.ts, desktop/tests/modular/pending-actions-service-down.test.ts, desktop/docs/frontend-api.md, desktop/docs/services-backend.md, desktop/docs/services-parity.md, desktop/docs/service-contract-exceptions.json, desktop/docs/services-api.md
**Depends on:** 1, 3 — task 1 also edits the contract exceptions file and generated `services-api.md`; task 3 also edits `node-policy.ts` and `frontend-api.md`
**Complexity:** judgment
**Do:**
1. Locks: delete `EngineLockControl`, `LOCK_COPY` and lock props from
   `NodeEnginesInline.tsx`; delete `setLock` from `ui/api/engine-api.ts`
   (`IEngineApi`, `createEngineApi`) and `shared/types/engine-api.ts`; delete the
   `engine:set-lock` handler and the `setLock` branch in `empty-handlers.ts`;
   delete `ENGINE_LOCKED_CODE` and `engineLockedMessage` (`service-errors.ts`) and
   their uses in `modular-supervisor.ts`; delete `locked` from
   `shared/types/engines.ts` and `modular-state.ts` (`reportedLock`); delete the
   `engine-locked` reason from `shared/types/node-policy.ts` and
   `electron/service-bridge/node-policy.ts`; delete the `setLock` action and
   `baselineLocked` from `pending-actions.store.ts`; delete the lock refusal
   handling in `EngineSettingsSection.tsx`.
2. Switch: delete `EngineSwitchControls.tsx` and `engine-switch.store.ts` and
   their mount in `NodeEngineSettings.tsx`; delete `switchEngine`,
   `switchStatus`, `fenceSwitch`, `onSwitchChanged` from the renderer API and
   types; delete the `engines:switch*` channels in `ws-channels.ts`, their
   handlers in `empty-handlers.ts`, the `engine:switch-changed` relay in
   `modular-supervisor.ts`, `electron/service-bridge/engine-switch.ts` parsers
   left unused and `shared/types/engine-switch.ts` types left unused; delete
   switch reservation and `isSwitchPending`/`beginSwitch`/`endSwitch` from
   `BackendRow.tsx`, `EngineSettingsSection.tsx` and `pending-actions.store.ts`.
   The broker still refuses a settings apply that collides with a TUI switch;
   show that refusal through the existing apply error path.
3. Run `npm run service-contracts:check`. For each switch notification it now
   reports as missing, add an `ignoredMethods` entry with the reason "Engine
   switch is a TUI-only operator surface since phase 04; the desktop does not
   show switches." Then `npm run service-contracts:write` and update the lock and
   switch passages of `frontend-api.md`, `services-backend.md` and
   `services-parity.md`.
4. Delete the five lock and switch test files; remove lock and switch cases from
   the other listed tests.
**Verify:** In `desktop`: `npm run lint && npm run typecheck && npm run test:unit && npm run dead-code:check && npm run service-contracts:check` exit 0; `git grep -nE "setLock|EngineLockControl|ENGINE_LOCKED|engine-locked|engine_locked|EngineSwitchControls|useEngineSwitchStore|engines:switch" -- desktop/src desktop/tests` prints nothing.

### Task 5 — Engine Start and Stop stop writing saved intent; Start with PAIR becomes a setting
**Files:** services/nvpair-engine-manager/lifecycle.go, services/nvpair-engine-manager/desiredstate.go, services/nvpair-engine-manager/desiredstate_test.go, services/nvpair-engine-manager/wakesleep.go, services/nvpair-engine-manager/wakesleep_test.go, services/nvpair-engine-manager/controllifecycle.go, services/nvpair-engine-manager/manager.go, services/nvpair-engine-manager/manager_test.go, services/nvpair-engine-manager/README.md, services/nvpair-engine-manager/spec.md, services/nvpair-ui-broker/idlepolicy.go, services/nvpair-tui/ui/engineswire.go, services/nvpair-tui/ui/engineswire_test.go, services/tests/off_origin_test.go, desktop/src/shared/types/ws-channels.ts, desktop/src/electron/service-bridge/empty-handlers.ts, desktop/src/ui/api/engine-api.ts, desktop/src/ui/components/BackendRow/EngineSettingsSection.tsx, desktop/tests/modular/engine-autostart.test.tsx (new), desktop/tests/modular/engine-intent.test.ts, desktop/docs/frontend-api.md, desktop/docs/services-api.md, docs/node-availability.mdx, docs/terminal-interface.mdx
**Depends on:** 2, 4 — task 2 also edits the engine-manager README and spec; task 4 also edits the desktop channel map, bridge handlers, renderer API, `EngineSettingsSection.tsx` and generated docs
**Complexity:** judgment
**Do:**
1. Engine-manager: `StartWith` no longer calls `setDesiredEnabled` after a
   start, and the stop path in `lifecycle.go` (the
   `setDesiredEnabled(engine, false, origin)` call) no longer writes intent; keep the cancellation of
   in-flight peer work with `-32011` (`engineOffVeto`) on a local stop. This
   covers local `engine:start`/`engine:stop`, `engine:remote-start`/`engine:remote-stop`
   (`controllifecycle.go`) and bound activation. `RestoreEnabled`, `Wake` and the
   switch executor's journal writes keep their current reads and writes.
2. Add `engine:set-autostart` to the engine-manager request switch
   (`manager.go`): params `{"engine": "<manager name>", "enabled": true|false}`,
   local clients only (refuse a call that arrives through the peer relay), writes
   the saved intent with origin local, emits `engine:intent-changed` through
   `emitIntent`, and replies with the intent snapshot (`engine:intent` shape). It
   does not start or stop the engine. The broker's generic `engine:*` relay
   forwards it; do not add it to `internalClientMethods`.
3. Rename the user-facing wording of saved intent to "Start with PAIR": TUI
   `engineswire.go` labels " (saved Off, local)"/" (saved Off, by peer)"/
   " (saved Off)" become " (not started with PAIR)"; update its tests.
   `idlepolicy.go` header comment: a stopped engine keeps Start with PAIR, and
   profile requests start any engine through the bound path.
4. Desktop: channel `'engines:set-autostart': { request: { engineType: EngineType; enabled: boolean }; response: EngineIntentSnapshot | null }`
   in `ws-channels.ts`; handler in `empty-handlers.ts` calling the broker with
   `engine:set-autostart` (manager name via `engineManagerName`); renderer method
   `setAutostart(engineType, enabled)` in `ui/api/engine-api.ts`. In
   `EngineSettingsSection.tsx` add a KUI `Checkbox` "Start with PAIR" bound to the
   local intent from `engine-status.store` (`intent`), local node only; remote
   nodes show the text "Start with PAIR is set on the device running the engine."
   Add `engine-autostart.test.tsx` (checkbox reflects intent, toggling sends one
   request, remote shows the text) and update `engine-intent.test.ts`.
5. Tests (Go): `TestStartDoesNotChangeSavedIntent` and
   `TestStopDoesNotChangeSavedIntent` in `desiredstate_test.go`;
   `TestSetAutostartWritesIntentOnly` and `TestSetAutostartRefusedFromPeer` in
   `manager_test.go`; update `services/tests/off_origin_test.go` to the new
   meaning (Stop no longer records Off).
6. Docs: `docs/node-availability.mdx` replaces "Saved Off no longer blocks a
   granted peer" with a "Start with PAIR" section; `docs/terminal-interface.mdx`
   saved-intent passage; engine-manager `README.md`/`spec.md` method table;
   `desktop/docs/frontend-api.md`; then `npm run service-contracts:write`.
**Verify:** In `services/nvpair-engine-manager`: `go test -run 'TestStartDoesNotChangeSavedIntent|TestStopDoesNotChangeSavedIntent|TestSetAutostartWritesIntentOnly|TestSetAutostartRefusedFromPeer' ./...` passes, then `go test ./... && go vet ./...` exit 0; the same pair exits 0 in `services/nvpair-ui-broker` and `services/nvpair-tui`; `go vet ./...` exits 0 in `services/tests`; in `desktop`, `npm run lint && npm run typecheck && npm run test:unit && npm run dead-code:check && npm run service-contracts:check` exit 0.

### Task 6 — Engine locks are removed from services, TUI, build-apply evidence and docs
**Files:** services/nvpair-engine-manager/enginelock.go (delete), services/nvpair-engine-manager/enginelock_test.go (delete), services/nvpair-engine-manager/desiredstate.go, services/nvpair-engine-manager/desiredstate_test.go, services/nvpair-engine-manager/nodeprojection.go, services/nvpair-engine-manager/nodeprojection_test.go, services/nvpair-engine-manager/controllifecycle.go, services/nvpair-engine-manager/executor.go, services/nvpair-engine-manager/manager.go, services/nvpair-engine-manager/remoteclient.go, services/nvpair-engine-manager/remoteclient_test.go, services/nvpair-engine-manager/residency.go, services/nvpair-engine-manager/residency_test.go, services/nvpair-engine-manager/status.go, services/nvpair-engine-manager/wakesleep.go, services/nvpair-engine-manager/switchexecutor.go, services/nvpair-engine-manager/switchexecutor_test.go, services/nvpair-engine-manager/owner_evidence.go, services/nvpair-engine-manager/owner_evidence_test.go, services/nvpair-engine-manager/bound_engine_test.go, services/nvpair-engine-manager/modelcopy_control_test.go, services/nvpair-engine-manager/modelsettings_load_test.go, services/nvpair-engine-manager/pausegate_test.go, services/nvpair-engine-manager/policyrelay_test.go, services/nvpair-engine-manager/settingsremote_test.go, services/nvpair-engine-manager/stop_pause_test.go, services/nvpair-engine-manager/README.md, services/nvpair-engine-manager/spec.md, services/nvpair-ui-broker/bound_engine.go, services/nvpair-ui-broker/bound_engine_test.go, services/nvpair-ui-broker/bound_engine_offers.go, services/nvpair-ui-broker/modelsettings.go, services/nvpair-ui-broker/modelsettings_contract_test.go, services/nvpair-ui-broker/build_apply.go, services/nvpair-ui-broker/build_apply_test.go, services/nvpair-ui-broker/engineswitch.go, services/nvpair-ui-broker/engineswitch_test.go, services/nvpair-ui-broker/README.md, services/nvpair-ui-broker/ENGINE_SETTINGS.md, services/nvpair-proxy/bound_engine.go, services/nvpair-proxy/engine_binding_requests_test.go, services/nvpair-proxy/README.md, services/nvpair-proxy/spec.md, services/nvpair-service/build_apply_run.go, services/nvpair-service/build_apply_preview_test.go, services/shared/buildapply/engine_owner.go, services/shared/buildapply/engine_owner_test.go, services/shared/buildapply/wire.go, services/shared/enginesettings/switch.go, services/shared/enginesettings/switch_test.go, services/shared/nodepolicy/bound_engine.go, services/shared/nodepolicy/wire.go, services/shared/nodepolicy/wire_test.go, services/nvpair-tui/ui/engine_switch.go, services/nvpair-tui/ui/engine_switch_test.go, services/nvpair-tui/ui/engineswire.go, services/nvpair-tui/ui/engineswire_test.go, services/nvpair-tui/ui/nodedetail.go, services/nvpair-tui/ui/settingswire.go, services/nvpair-tui/README.md, services/tests/engine_binding_test.go, services/tests/off_origin_test.go, services/tests/remote_configuration_test.go, services/tests/stop_pause_controls_test.go, docs/architecture.mdx, docs/local-qualification.mdx, docs/model-engine-bindings.mdx, docs/node-availability.mdx, docs/terminal-interface.mdx, desktop/src/shared/types/build-apply.ts, desktop/src/electron/service-bridge/build-apply.ts, desktop/src/electron/build-apply/index.ts, desktop/src/ui/components/ServiceSettings/BuildApplyPreview.tsx, desktop/tests/modular/build-apply-contract.test.ts, desktop/tests/modular/build-apply-ui.test.tsx, desktop/tests/modular/build-apply-reconnect.test.ts, desktop/tests/modular/build-apply-status.test.ts, desktop/docs/services-api.md
**Depends on:** 5 — shares engine-manager, TUI, docs and generated-doc paths with task 5 (and through it with tasks 1, 2 and 4)
**Complexity:** judgment
**Do:**
1. Re-run the footprint at your base and compare with Files:
   `git grep -lE "engine:set-lock|setEngineLocked|engineLocked|EngineLocked|engine_locked|-32013|engine-locked|setLocked|lockedStatus|LockedByEngine|locksDigest|locksRevision|preserveIntentAndLocks" -- services docs desktop/src desktop/tests`
   (at `4eccd05d` it listed 86 files, including desktop lock and switch files that
   Task 4 deletes or cleans first).
   A true-positive file not in Files is in scope for this task; record it in
   the commit message. `localOnly` and mutex names ending in `Locked` are not
   engine locks; leave them.
2. Engine-manager: delete `enginelock.go` and its test; remove the
   `engine:set-lock` method, the lock branch of `activationVeto`/`remoteVeto`
   (keep the pause veto first and unchanged), `Locked` in status and the
   `locked` map in `desiredStateFile` (the decoder ignores an existing `locked`
   key in old files; the owner-evidence document stops listing `locked`), lock
   digests in `owner_evidence.go`, the `codeEngineLocked`/`-32013` code and the
   switch error kind `engine_locked`.
3. Broker, proxy, shared: remove the `engine-locked` bound refusal and
   `*target.Locked` checks (`bound_engine.go`, `bound_engine_offers.go`),
   `modelsettings.go` lock refusal, build-apply lock facts (`build_apply.go`,
   `services/shared/buildapply/engine_owner.go` `LockedByEngine`, `wire.go`,
   `services/nvpair-service/build_apply_run.go`), `nodepolicy` wire reason and
   code, `enginesettings/switch.go` lock error kind.
4. TUI: remove lock display and actions in `nodedetail.go`, `engineswire.go`,
   `engine_switch.go`, `settingswire.go`; update `services/nvpair-tui/README.md`.
5. Desktop (same change, because the build-apply wire changes): remove lock
   observations and the `preserveIntentAndLocks` flag (renamed
   `preserveIntent`) from `shared/types/build-apply.ts`,
   `electron/service-bridge/build-apply.ts`, `electron/build-apply/index.ts` and
   `ServiceSettings/BuildApplyPreview.tsx` and their tests; run
   `npm run service-contracts:write`.
6. Docs: remove "The per-engine local lock" and "Which code means what" lock
   rows from `docs/node-availability.mdx` and change its title to "Node
   availability: pause, resume and Halt"; remove lock text from the other listed
   docs, READMEs and specs.
7. Tests: delete lock-only tests; remove lock cases from the listed tests; keep
   pause-veto coverage.
**Verify:** The step 1 command prints nothing; `go test ./... && go vet ./...` exit 0 in `services/nvpair-engine-manager`, `services/nvpair-ui-broker`, `services/nvpair-proxy`, `services/nvpair-service`, `services/nvpair-tui` and `services/shared`; `go vet ./...` exits 0 in `services/tests`; in `desktop`, `npm run lint && npm run typecheck && npm run test:unit && npm run dead-code:check && npm run service-contracts:check` exit 0.

### Task 7 — One settings page per node replaces Routing and resources and the card's engine settings
**Files:** desktop/src/ui/components/NodeSettings/NodeSettingsPage.tsx (new), desktop/src/ui/components/NodeSettings/NodePolicyJsonEditor.tsx (new), desktop/src/ui/components/NodeSettings/NodeEnginesSection.tsx (new), desktop/src/ui/components/RoutingSettings/RoutingSettings.tsx (delete), desktop/src/ui/components/RoutingSettings/BulkNodeConfiguration.tsx (delete), desktop/src/ui/components/RoutingSettings/ModelEngineBindingsEditor.tsx (delete), desktop/src/ui/components/NodeList/NodeEngineSettings.tsx (delete), desktop/src/ui/components/NodeList/NodeCardDetails.tsx, desktop/src/ui/components/NodeList/NodeEnginesInline.tsx, desktop/src/ui/components/NodeList/NodeAvailabilityControls.tsx, desktop/src/ui/components/BackendRow/BackendRow.tsx, desktop/src/ui/components/BackendRow/BackendHeader.tsx, desktop/src/ui/components/BackendRow/EngineSettingsSection.tsx, desktop/src/ui/components/Settings.tsx, desktop/src/ui/types/settings-window.ts, desktop/src/ui/stores/overview-ui.store.ts, desktop/src/ui/stores/node-configuration.store.ts, desktop/src/ui/components/MainApp/Content.tsx, desktop/src/ui/components/ClusterSettings/ActiveNode.tsx, desktop/src/ui/components/Welcome/WelcomeModal.tsx, desktop/src/ui/hooks/useOverviewCommands.ts, desktop/tests/modular/node-settings-page.test.tsx (new), desktop/tests/modular/routing-settings-pause.test.tsx (delete), desktop/tests/modular/model-engine-bindings-editor.test.tsx (delete), desktop/tests/modular/node-configuration-bulk.test.ts (delete), desktop/tests/modular/node-availability-controls.test.tsx, desktop/tests/modular/remote-management.test.tsx, desktop/tests/modular/model-settings-panel.test.tsx, desktop/tests/modular/node-configuration.test.ts, desktop/tests/modular/node-policy-precision.test.ts, desktop/tests/modular/service-build-identity.test.tsx, desktop/tests/modular/lmstudio-credential.test.ts, desktop/tests/modular/engine-settings-state.test.ts, desktop/docs/architecture.md, docs/overview.mdx
**Depends on:** 1, 3, 4, 5 — task 1 provides the peer Resume semantics shown here; tasks 3 and 4 also edit `NodeAvailabilityControls.tsx`, `NodeEnginesInline.tsx`, `BackendRow.tsx` and the docs; task 5 adds Start with PAIR to `EngineSettingsSection.tsx`; also needs the phase 02 entry condition
**Complexity:** judgment
**Do:**
1. Navigation state: in `overview-ui.store.ts` replace `focusNodeId`,
   `focusNodeEngineSettings` and `consumeFocusNode` with
   `nodeSettingsId: string | null`, `openNodeSettings(nodeId)` (sets
   `activeTab: 'overview'` and the id) and `closeNodeSettings()`. Call
   `openNodeSettings` from `ActiveNode.tsx` "Edit node", `WelcomeModal.tsx` and
   `useOverviewCommands.ts` (`focus-node`). Remove `routing` from
   `SettingsWindowTab`/`isSettingsWindowTab` and `SETTINGS_NAV_ITEMS`.
2. `Content.tsx`: on the Overview tab, render `NodeSettingsPage` instead of the
   overview body while `nodeSettingsId` is set.
3. `NodeSettingsPage.tsx`, composed in this order (each existing component as
   it is after phases 02/03 unless stated): KUI `PageHeader` with `Breadcrumbs`
   "Overview" (calls `closeNodeSettings`) › node name; `NodeAvailabilityControls`;
   `RoleEditor`; profiles on this node (`ModelSettingsPanel` with the props
   `RoutingSettings` passed); `NodeEnginesSection`; `NodePolicyJsonEditor`;
   `RoutingHistory` (this PC only). Move the target, management and
   `permissionReadOnly` logic out of `RoutingSettings.tsx` into the page
   unchanged; a node whose management is not allowed shows read-only controls
   as today.
4. `NodePolicyJsonEditor.tsx`: the "Advanced policy JSON" block extracted from
   `RoutingSettings.tsx` unchanged (labels, limits, Save, Reload, status badge).
5. `NodeEnginesSection.tsx`: one `BackendRow` per engine of the node
   (`engineTypesForNode`). `BackendHeader` loses its On/Off `Switch` and shows a
   KUI `Badge` for running state (Running, Stopped, Starting, Error, Managed
   outside PAIR) and a "Stop now" `Button` (enabled when running) that calls
   `window.pairApi.engines.toggle` on a running engine (an intent-neutral stop
   since Task 5). Move `CredentialControl` from `NodeEnginesInline.tsx` into
   `BackendRow` (local node only). Keep `InstallButton`, `BackendUpdateBanner`,
   `ModelSection`, `EngineSettingsSection` (ports, environment in Engine
   arguments, working-set limit, Start with PAIR) and `BackendFooter`
   (install prerequisites, Docs, Uninstall).
6. `NodeAvailabilityControls.tsx`: for a remote node, show "Resume" when the
   reported state is paused with `pausedBy` peer and management is allowed;
   keep "Remote node paused here; resume on that device" text for
   `pausedBy` local.
7. Card: `NodeCardDetails.tsx` removes "Show engine settings" and the
   `NodeEngineSettings` mount and the focus effect; add a KUI icon `Button`
   (aria-label "Open {name} settings") that calls `openNodeSettings(node.id)`.
8. Delete `RoutingSettings.tsx`, `BulkNodeConfiguration.tsx`,
   `ModelEngineBindingsEditor.tsx`, `NodeEngineSettings.tsx`, the bulk state and
   actions in `node-configuration.store.ts` (`bulk`, `applyAll`,
   `BULK_CONCURRENCY` and helpers left unused) and their tests.
9. Tests: `node-settings-page.test.tsx` covers open from card icon and from
   "Edit node", breadcrumb back, section order, read-only for a non-managed peer,
   remote Resume shown only for a peer-origin pause. Update the listed tests.
10. Docs: `desktop/docs/architecture.md` navigation section and
    `docs/overview.mdx` describe the node settings page.
**Verify:** In `desktop`: `npx vitest run --project unit tests/modular/node-settings-page.test.tsx` passes; `npm run lint && npm run typecheck && npm run test:unit && npm run dead-code:check && npm run service-contracts:check` exit 0; `git grep -nE "RoutingSettings/RoutingSettings|<RoutingSettings|BulkNodeConfiguration|ModelEngineBindingsEditor|focusNodeEngineSettings|Routing & resources" -- desktop/src desktop/tests` prints nothing (the `RoutingSettings/` folder keeps `RoleEditor.tsx`, `ModelSettingsEditor.tsx`, `RoutingHistory.tsx` and `RoutingAttempt.tsx`).

### Task 8 — Overview card shows roles as identity tags and engine running state
**Files:** desktop/src/ui/components/NodeList/NodeCardDetails.tsx, desktop/src/ui/components/NodeList/NodeEnginesInline.tsx, desktop/src/ui/components/NodeList/NodeDeclaredRoles.tsx, desktop/tests/modular/node-card-roles.test.tsx (new), desktop/tests/modular/service-build-identity.test.tsx, desktop/tests/modular/node-availability-controls.test.tsx, desktop/tests/modular/lmstudio-credential.test.ts
**Depends on:** 7 — shares the card files and tests; also needs the phase 03 entry condition
**Complexity:** judgment
**Do:**
1. `NodeEnginesInline.tsx` becomes display-only: per reported engine, the engine
   icon and label (`engineLabel`) and a KUI `Badge` with running state; no
   `Switch`, no install button, no credential control (install and credential
   live in the settings page). Keep the component name `NodeEnginesInline`.
2. `NodeDeclaredRoles.tsx`: per role name (phase 03 strings), one row: the role
   name, then each profile in the role's ordered list as its display name plus
   the identity tags rendered by phase 01's tag component (repository and quant);
   a profile whose engine is not running gets a "Starts on request" tag. Read
   from the advertisement in `discovered-nodes.store` exactly as phases 02/03
   left it; no new store.
3. `NodeCardDetails.tsx` keeps: chart, `NodeLabel`, `NodeEnginesInline`,
   `NodeDeclaredRoles`, `NodeAvailabilityControls`, service build, performance,
   the settings icon from Task 7.
4. Tests: `node-card-roles.test.tsx` (role with two profiles renders both
   names and identity tags; no switch or padlock role present in the card DOM);
   update the listed tests.
**Verify:** In `desktop`: `npx vitest run --project unit tests/modular/node-card-roles.test.tsx` passes; `npm run lint && npm run typecheck && npm run test:unit && npm run dead-code:check` exit 0; `git grep -n "<Switch" -- desktop/src/ui/components/NodeList/NodeEnginesInline.tsx` prints nothing.

### Task 9 — Models tab lists cluster models grouped by identity and quant
**Files:** desktop/src/ui/components/Models/ModelsView.tsx (new), desktop/src/ui/components/Models/ModelGroup.tsx (new), desktop/src/ui/hooks/useModelActions.ts (new), desktop/src/ui/utils/group-cluster-models.ts (new), desktop/src/ui/components/ModelManager/ModelManager.tsx (delete), desktop/src/ui/components/ModelManager/ModelSection.tsx (delete), desktop/src/ui/components/ModelManager/ModelRow.tsx, desktop/src/ui/components/BackendRow/BackendRow.tsx, desktop/src/ui/components/MainApp/ContentBar.tsx, desktop/src/ui/components/MainApp/Content.tsx, desktop/src/shared/types/overview.ts, desktop/tests/modular/group-cluster-models.test.ts (new), desktop/tests/modular/models-view.test.tsx (new), desktop/tests/modular/delete-model-restart.test.ts, desktop/tests/modular/engine-descriptors.test.ts, desktop/tests/modular/model-copy-modal.test.tsx, desktop/tests/modular/model-download-readiness.test.tsx, desktop/tests/modular/model-settings-copy.test.tsx, desktop/tests/modular/model-settings-panel.test.tsx, desktop/docs/architecture.md
**Depends on:** 3, 7 — task 3 also edits `ContentBar.tsx`; task 7 also edits `Content.tsx`, `BackendRow.tsx`, tests and `architecture.md`; also needs the phase 01 entry condition
**Complexity:** judgment
**Do:**
1. `group-cluster-models.ts`: a pure function from the `engine-models.store`
   map (all `nodeId:engineType` entries) and node names to groups: repository →
   quant label → per-node entries
   `{nodeId, engineType, modelName, status, size, contentId}`, with the rule in "Decisions made while writing this plan"
   (content-id mismatch flagged, "Unidentified" group last, stable sort by
   label). No routing decisions.
2. `useModelActions.ts`: move `actionAllowed`, `handleAction`, `handleDownload`
   and the copy-modal state out of `ModelManager.tsx` unchanged in behaviour,
   parameterised by node id and engine; use `usePendingActionsStore` as
   `ModelManager` did.
3. `ModelsView.tsx` and `ModelGroup.tsx`: a KUI list with one `Card` per
   repository showing the identity tags (phase 01 component), one row per quant
   with per-node presence (node name, Loaded/On disk badge, size via
   `formatBytes`), and each node entry's menu from `ModelRow` (Load, Eject, Copy
   to device…, Delete; "Configure generation settings" calls
   `openNodeSettings(nodeId)`). "Copy to device…" opens `ModelCopyModal` with
   progress from `engine-progress.store`. A toolbar "Add model" opens a node and
   engine picker (KUI `Select`), then `ModelHubModal` for that engine,
   downloading to the chosen node. Select `state.models` and derive with
   `useMemo` (the store replaces its map on change).
4. Tabs: add `'models'` to `OverviewTab`, to `TAB_ITEMS` and `isOverviewTab`
   in `ContentBar.tsx` (order Overview, Models, Settings until Task 10), and
   render `ModelsView` in `Content.tsx`.
5. Remove `ModelSection` from `BackendRow.tsx`; delete `ModelManager.tsx` and
   `ModelSection.tsx`; move their test coverage to `models-view.test.tsx` and
   the hook.
6. Tests: `group-cluster-models.test.ts` (two nodes with the same repository
   and quant form one group with two entries; differing content ids are
   flagged; unidentified rows last); `models-view.test.tsx` (renders groups,
   copy opens `ModelCopyModal`, Add model downloads to the chosen node).
**Verify:** In `desktop`: `npx vitest run --project unit tests/modular/group-cluster-models.test.ts tests/modular/models-view.test.tsx` passes; `npm run lint && npm run typecheck && npm run test:unit && npm run dead-code:check` exit 0; `git grep -nE "ModelManager/ModelManager|<ModelManager|ModelSection" -- desktop/src desktop/tests` prints nothing (the `ModelManager/` folder keeps `ModelRow*`, `ModelCopyModal` and the transient row components).

### Task 10 — Profiles tab shows profile cards, their usage and Deploy to node
**Files:** desktop/src/ui/components/Profiles/ProfilesView.tsx (new), desktop/src/ui/components/Profiles/ProfileCard.tsx (new), desktop/src/ui/components/Profiles/DeployProfileModal.tsx (new), desktop/src/ui/hooks/useProfileDeploy.ts (new), desktop/src/ui/components/ModelSettings/ModelSettingsCopy.tsx, desktop/tests/modular/model-settings-copy.test.tsx, desktop/src/ui/utils/profile-summary.ts (new), desktop/src/ui/components/MainApp/ContentBar.tsx, desktop/src/ui/components/MainApp/Content.tsx, desktop/src/shared/types/overview.ts, desktop/tests/modular/profile-summary.test.ts (new), desktop/tests/modular/profiles-view.test.tsx (new), desktop/docs/architecture.md, docs/overview.mdx
**Depends on:** 7, 9 — task 9 also edits the tab files and `architecture.md`; task 7 also edits `docs/overview.mdx`; also needs the phase 02 and 03 entry conditions
**Complexity:** judgment
**Do:**
1. Data: read every node's advertised profiles (`AdvertisedProfile`) and roles
   from the projection phases 02/03 left in `discovered-nodes.store` (this PC's
   from its policy read as those phases do). Group cards by profile id. Usage =
   every (node, role name) whose role list contains the profile id. Load
   settings come from node policy reads where readable (this PC, and peers whose
   management is allowed, through the existing `node-configuration.store` policy
   read); for other peers the card shows the advertised `ContextTokens` only. No
   new store; when phase 02 left no renderer projection of advertised profiles,
   stop and report (that projection belongs to phase 02).
2. `profile-summary.ts`: for profiles sharing one model identity, return the
   `load` keys whose values differ (for example context, GPU layers,
   `n-cpu-moe`), formatted as short `key value` strings; pure, unit-tested.
   Profiles without a readable load are left out of the comparison.
3. `ProfileCard.tsx`: KUI `Card` with icon (when set), name, identity tags,
   engine label, the differing-settings summary, measurement state ("Not
   validated" unless the profile carries a measurement record), usage list
   "node → role", and "Deploy to node". Clicking the card opens the owning
   node's settings page at the profiles section (`openNodeSettings`).
4. Move the per-target deploy orchestration out of `ModelSettingsCopy.tsx`
   into `useProfileDeploy.ts` without changing its behaviour (`targetModelFor`,
   copy through `engine:copy-model` with progress from `engine-progress.store`,
   inventory refresh, `deployProfileChange`, target preview and apply), and make
   `ModelSettingsCopy.tsx` call the hook. `DeployProfileModal.tsx`: source node
   `Select` (nodes whose policy holds the profile and is readable), target node
   `Select` (peers with management allowed plus this PC, minus the source),
   target role `Select` (role names on that node plus a new name), then the hook;
   show copy progress (`formatCopyProgressLabel`), the target preview diff, a
   confirm, and the per-target result or refusal.
5. Tabs: add `'profiles'`; final order Overview, Models, Profiles, Settings.
6. Tests: `profile-summary.test.ts`; `profiles-view.test.tsx` (usage list;
   deploy to a target lacking the model starts one copy before preview; a target
   holding the same `contentId` skips the copy; a target without management
   permission is absent from the list); keep `model-settings-copy.test.tsx`
   passing against the hook.
7. Docs: `desktop/docs/architecture.md` and `docs/overview.mdx` describe the
   Models and Profiles tabs.
**Verify:** In `desktop`: `npx vitest run --project unit tests/modular/profile-summary.test.ts tests/modular/profiles-view.test.tsx` passes; `npm run lint && npm run typecheck && npm run test:unit && npm run dead-code:check && npm run service-contracts:check` exit 0; `git grep -n "children: 'Profiles'" -- desktop/src/ui/components/MainApp/ContentBar.tsx` prints one line.

## Verification (whole-plan)

Local gate (all must pass on the final commit of the phase; retain the command,
exit code and the last 50 lines of output of each in the close-out WORKLOG
entry):

- `cd desktop && npm run lint && npm run typecheck && npm run test:unit && npm run dead-code:check && npm run service-contracts:check`
- For each of `services/nvpair-engine-manager`, `services/nvpair-ui-broker`,
  `services/nvpair-proxy`, `services/nvpair-tui`, `services/nvpair-node-scanner`,
  `services/nvpair-service`, `services/shared`: `go test ./... && go vet ./...`
  and `GOOS=windows GOARCH=amd64 go vet ./...`.
- `cd services/tests && go vet ./...`; run the cross-process cases touched by
  Tasks 1, 2, 5 and 6 where the environment supports the suite.
- Repo root: `make check` and `node scripts/spdx-headers.mjs`.
- A failure called pre-existing must fail the same way on the recorded starting
  revision in a separate scratch checkout; otherwise it blocks the phase.

Machine acceptance (owner's Mac and MSI, plus Joes for deploy; record each as
observed-pass, observed-fail or untested in the vault `Validation/` folder):

1. Halt reroute: from the Mac, start a long streaming request routed to the
   Mac's own engine with MSI eligible for the same role; before the first
   token arrives, press Halt in the Mac toolbar and confirm. The request
   completes on MSI (routing history shows a redispatch with node-paused); the
   AppBar shows "Paused on this PC"; MSI's peer view of the Mac shows it paused
   without waiting 15 s (time it); the Mac's Resume brings it back.
2. Models tab on the Mac shows Tiel 35B as one model group whose quant row
   lists MSI and Joes.
3. From the Mac's Profiles tab, deploy a profile to a role on Joes; Joes's card
   then shows the role with that profile's identity tags, and a request for that
   role on Joes starts its engine if it was stopped (Task 2), without any engine
   switch or padlock visible anywhere.

## Close-out

- In `plans/2026-10-31-expectation-gaps/00-index.md`, append ` — done <sha>`
  to the phase 04 row, and append ` — done <sha>` to each task heading in this
  file as it lands (the lifecycle text at the end of this file).
- Add a vault entry to `/home/user/wasabi/4. Projects/Personal-AI-Router/WORKLOG.md`
  listing: commits per task, the local gate results, the machine-acceptance
  rows with their state, the remaining limitations (pause hint depends on
  multicast; Start with PAIR is local-only; `engineBindings` still exists; the
  TUI still shows the engine switch; tray popup shows no pause state), and the
  hand-over: phase 05 may now change the refusal that used `-32013`, and phase
  06's Apps group builds on the node settings page's engine section.
- Any task dropped gets ` — dropped` and a vault `deferred.md` entry.

## Do not

- Do not refactor beyond what the tasks specify; in particular, do not delete
  the engine switch journal or its Go owner, do not retire `engineBindings`, do
  not redesign `RoleEditor` or the profile editor.
- Do not add routing, candidate ordering or scheduling logic to TypeScript;
  Models and Profiles views only group and display.
- Do not force-stop engines PAIR did not start, and do not let a peer resume a
  local pause or a Halt.
- Do not add a second availability owner, a second profile lookup, a second
  model-name formatter or a new renderer store where an existing one holds the
  data.
- Do not use `as` casts, do not hand-edit `desktop/docs/services-api.md` or the
  vendored KUI CSS, do not add compatibility shims, version bumps or migrations.
- Do not edit files of the read-only baseline worktree `/home/user/pair-join33`.

## If stuck

- A failing check after a Do step is safe to retry after fixing the code; a
  flaky cross-process test is retried once and then reported with its output.
- If phase 02's profile or deploy symbols differ from what Task 2 and Task 10
  assume, adapt to phase 02's names; if the concept is missing (no profile id,
  no deploy orchestration), stop the affected task and report.
- If removing the bound path's other-engine guard (Task 2) breaks a switch or
  settings-apply invariant that a test protects, stop and report the test and
  invariant; do not re-add the sole-running refusal.
- If the browse-event path cannot refetch a peer's policy without a larger
  scanner change (Task 1 step 4), land steps 1 to 3 and report the gap; the
  15 s sweep still applies.
- If `service-contracts:check` reports drift that a task's listed exception
  does not cover, stop and report the method name rather than adding a broad
  exception.

## Note lifecycle

This file is working state in a repository, not documentation.

- When an item lands, append ` — done <sha>` to its heading, where
  `<sha>` is the commit that landed it. An item consciously dropped gets
  ` — dropped` and a `project-note deferred` entry.
- When every item carries one of those marks, delete this file in the
  commit that lands the last item. Nothing in it is kept: the commits and
  the run archive are the record, and `project-note done` records the
  completion.
- In an execute run the orchestrator does this after each unit lands,
  not the unit. A plain session does it as each item lands.
