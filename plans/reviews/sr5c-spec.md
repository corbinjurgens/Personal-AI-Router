<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SR 5c resident reconcile: state-machine specification and adversarial review

Verdict: **RED 3 UNVERIFIED 2 LOW 7**

## 0. Scope, inputs and evidence

- Inspected head of `roadmap/sr-5c`: `74878aa3931c1e2943b6ccde6eadf3597213292d` (the handoff's `74878aa`; the branch had not moved). Base `fork/groundwork`: `7e270f10ced402325dc9e1efece54991e9bbbc08` (unchanged from the common brief). Every `path:line` below is at `74878aa` unless marked `(groundwork)`; proxy and manager files are identical on both.
- Branch commits over groundwork: `e9d328e feat(broker): reconcile bounded resident model configuration`, `517d019 fix: confirm resident inventory without duplicate loads`, `1730d6d fix: retry resident reconciliation without duplicate loads`, `74878aa fix: preserve resident completion across temporary vetoes`.
- `git diff origin/fork/groundwork...origin/roadmap/sr-5c --stat`: 15 files, +3143/-43. Broker: `modelconfig.go` (new, 755 lines), `modelconfig_test.go` (new, 1539), `idlepolicy.go`/`idlepolicy_test.go`, `policy.go`, `policy_notifications.go`, `policy_test.go`, `worker_lifecycle.go`, `broker.go`, `README.md`. Desktop: `service-bridge/node-policy.ts`, `shared/types/node-policy.ts`, `tests/modular/node-policy.test.ts`. Cross-process: `services/tests/service_policy_test.go`, `policy_relay_joint_test.go`. The manager's 5b commit (`40c7523`) is already in groundwork, so the manager and proxy resident paths cited below are landed code.
- Runs on `74878aa` (VERIFIED): `go test -count=1 ./...` in `services/nvpair-ui-broker` -> `ok nvpair-ui-broker 132.058s`, `ok .../performancestore`, `ok .../relay`, `ok .../workloadstore`. `go vet ./...` in the broker module -> clean. `go test -count=1 -race -run '^(TestModelConfig|TestResidentProtection|TestServedResident)' .` -> `ok 5.928s`. `services/tests`: `go vet ./...` clean; `go test -list` shows `TestServicePolicyModelConfiguration` and `TestPolicyRelayJointModelConfiguration`; neither was run (common brief).
- Ruling applied (common brief, settled): level-triggered state per (engine, model, revision); in-flight and pending-observation are facts cleared only by a fresh confirming observation, a load failure, revision replacement/removal, or a bounded observation timeout; pause, Off and drain block new loads but preserve those records; failures retry, bounded; `declaredResidentCount` 1..256 revision-bound; `Load.ContextLength` equals the profile's context exactly; NE readback transport failures are UNKNOWN with no latch.

## 1. State model as the branch implements it

Key = normalized `nodepolicy.Role{Engine, Model}` from `residentRefs` (`modelconfig.go:181-198`, deterministic engine/model order, llama.cpp naming is `exact` per `services/nvpair-engine-manager/manifests/llamacpp.json:8`, so normalization is the identity for the daily engines).

| Spec component | Branch representation | Where |
|---|---|---|
| desired | key in `residentRefs(e.Policy)`; `DesiredResident` always true for listed keys | `modelconfig.go:181-198, 215` |
| observed-loaded | `e.Loaded[engine]` (nil = unknown, slice = known) bound to the loaded source/proxy; `e.LoadedCurrent[engine][model]` = identity continuity since that observation AND `loadedRevision == Policy.Revision` | `modelconfig.go:50-51, 119-133`; `policy.go:1121-1163` |
| in-flight load | one global `r.models.active *modelConfigAttempt` with phases prepare -> grant (`grantStarted`, `granted`) -> load (manager RPC) -> finish; `busy` serializes passes | `modelconfig.go:61-70, 81, 347-351, 541` |
| unsettled hold (post in-flight, finish not settled or RecoveryID unresolved) | `r.models.unsettled[key]{op, finish, terminal}` | `modelconfig.go:32-36, 546-560` |
| pending observation | `r.models.pending[key]{evidence, expires = now + 60s}` (`residentObservationTimeout = 2 * idleCheckInterval`) | `modelconfig.go:25-30, 417, 635` |
| confirmed proof | `r.models.confirmed[key] = evidence` | `modelconfig.go:86, 374, 440` |
| failure count / backoff | **none per key**; global `last`/`retryAt = now + 30s` suppresses only identical evidence until the next idle tick | `modelconfig.go:74-76, 347, 378-379` |
| gating | pause/Live, intent (Off), drain, activation/source/worker, running, qualification, envelope (retention `-1`, capacity >= count, per-slot context == profile) | `modelconfig.go:236-267` |
| capacity | not a broker gate; proxy `prepare-resident` refusal becomes `blocked/<reason>` | `modelconfig.go:572-575`; proxy `residency.go:165-262` |
| published status | `ModelConfiguration{PolicyRevision, ObservationGeneration, Models}` with the four states and bounded reasons | `modelconfig.go:309-339`; `wire.go:664-720` |

Composite per-key states used in the table:

- **S0 Undesired** (no key for this revision).
- **S1 Eligible** (`state=pending`, inventory known, gates clear, no record).
- **S2 In-flight** with phases **P** prepare, **G** grant, **L** load, **F** finish.
- **S3 Unsettled hold** (granted, finish unsettled or `RecoveryID` outstanding).
- **S4 Pending observation** (terminal `loaded` + finished `loaded`; awaiting watcher; bounded 60 s).
- **S5 Confirmed** (satisfied with proof).
- **S6 Gated** (`blocked`: paused-local, engine-off-local, switch-busy, engine-off, unqualified, unsupported-load, proxy refusal) with any S3-S5 record preserved.
- **S7 Unknown** (inventory unknown or activation/source/worker not current) with records preserved.
- **S8 Failed, awaiting retry** (identical to S1 but suppressed until the next tick or an evidence change).

Events: **E1** revision replaced, key still desired; **E2** revision replaced with key removed, or key removed; **E3** confirming observation (same source and proxy, engine present, model listed); **E4** contradicting observation (same source and proxy, engine present, model absent); **E5** unknown observation (engine absent, payload without `loadedByEngine`, transport failure, or observation from a retired source/proxy); **E6** load start; **E7** load success (manager terminal `loaded`, finish `loaded`, no `RecoveryID`); **E8** load failure (terminal `failed`/`timeout`/`cancelled`/`not-dispatched`, finish without `RecoveryID`); **E9** load uncertain (manager RPC error/timeout or finish returning `RecoveryID`); **E10** observation timeout; **E11/E12** pause (or Live != available) on/off; **E13/E14** Off on/off; **E15/E16** drain on/off; **E17** capacity change (`ActiveByEngine`) or proxy refusal; **E18** proxy replacement; **E19** engine-manager replacement; **E20** broker restart; **E21** idle tick with unchanged evidence; **E22** `engine:dispatch-resident-request` from the manager.

## 2. Transition table, checked cell by cell

Columns: required next state and side effect derived from the ruling; what the code does with `path:line`; verdict (MATCH, RED, LOW, OPEN, and "untested" when no branch test covers the cell). "load" means a `prepare-resident` plus `engine:reconcile-resident` is issued; "no load" means neither.

### S0 Undesired

| Event | Required | Code | Verdict |
|---|---|---|---|
| E1/E2 adds key | -> S1/S6/S7 by gates and inventory; no load until pass | `residentRefs` of new policy; `reconcileModelConfig` 380-384 | MATCH (`TestModelConfigStatusUnknownEmptyAndRemoval`) |
| any other | nothing | records for removed keys deleted 352-367; no unload (`unloadModels` callers are only pause/idle/admission: `availability.go:242`, `idlepolicy.go:102`, `policy.go:1593`) | MATCH |

### S1 Eligible (no record)

| Event | Required | Code | Verdict |
|---|---|---|---|
| E6 load start | -> S2.P; prepare to captured proxy, reconcile to paired manager, `Load.ContextLength = profile.ContextTokens`, `DeclaredResidentCount` = engine's count, deadline <= min(60 s, switch timeout) | 431, 511-520 (bound), 540 (prepare), 581 (reconcile with `Load`), 214-223 (count) | MATCH (`LoadUsesRevisionProfile`, `AbsolutePolicyDeadlineBoundsLoad`, `InflightStatusRetainsEveryDesiredReference`) |
| E3 confirming (identity continuous, same revision observation) | confirm from readback, no load -> S5 | 423-429 -> `confirmResident` 453-487 (`engine:get-launch`, observed envelope must match) | MATCH (`ConfirmingInventoryDoesNotReload`, `WorkerRestartConfirmsWithoutReload`, `LoadedConfirmationRequiresObservedEnvelope`) |
| E3 confirming but `LoadedCurrent` false only because `loadedRevision != revision` | OPEN by ruling text; recommended: confirm from readback, **no load** (observation is fresh, same source/proxy/identity) | 122 requires `loadedRevision == revision`; falls to 430-431 `reconcileResident` -> **load issued against a positive loaded observation** | **RED-1** |
| E4 contradicting | stay S1, load this pass | 224-232, 250-266, 431 | MATCH |
| E5 unknown | -> S7, no load | 224-232 (`ObservedLoaded` nil), 249 (no `pending`), 399 | MATCH (`NotificationWithoutLoadedInventoryClearsKnowledge`, `ProxyReplacementRequiresFreshInventory`) |
| E11/E13/E15 | -> S6, no load | 237-242 | MATCH (`TemporaryVetoPreservesCompletedLoad`, `AuthorityChangesDuringPreparePreventLoad`) |
| E17 proxy refusal | -> S6 `blocked/<reason>`, no manager call; retry on evidence change | 572-575 | MATCH (`CapacityEvidenceRetriesOneRefusedAttempt`, `ProxyRefusalDoesNotReachManager`, `AdmissionRefusalsHaveNoLoad`) |
| E1 (profile/envelope unchanged) | OPEN; recommended: keep observed-loaded fact, re-confirm, no load | see RED-1 | **RED-1** |
| E18 | -> S7 until fresh inventory bound to new proxy | 119-121, `worker_lifecycle.go:364-368` refresh | MATCH (`ProxyReplacementRequiresFreshInventory`) |
| E19 | -> S7 until fresh inventory from new source | 116-118; `worker_lifecycle.go:507-512` resets `loaded/intent/running`; `policy.go:1124-1127` rejects retired-source observations | MATCH |
| E20 | all in-memory state lost; -> S7, refreshed by spawn | `policyRuntime.models` is not persisted; `refreshEnginePolicySnapshotFrom` at spawn `worker_lifecycle.go:368, 541` | MATCH (INFERRED: proxy and manager are children of the broker, so they restart with it) |
| E21 | no RPC when evidence unchanged and before `retryAt`; one re-evaluation per tick | 347 | MATCH |

### S2 In-flight

| Phase/Event | Required | Code | Verdict |
|---|---|---|---|
| P: E11/E13/E15/E1/E18/E19 during prepare | no manager call; finish on the original proxy; -> S6/S7/S1 per new evidence | `residentCurrent` 565, 580 (`sameModelAuthority` covers Paused, Live, Intent, Drains, Policy, Running, Proxy, Worker, Source); finish 603-611 always to `e.Proxy` | MATCH (`AuthorityChangesDuringPreparePreventLoad` x4, `ManagerReplacementInvalidatesPreparedWork`) |
| G: E22 valid request from current paired source, matching captured params | forward `node/dispatch-resident` to prepared proxy with remaining deadline; one grant; reply through same worker/source pair | 680-726 (strict envelope 728-755, `DisallowUnknownFields`, `op.evidence.Source == source`, `reflect.DeepEqual(p, want)`, `grantStarted` once, reply via `replyPeerRelay` -> `remote_management.go:46-50` pair check) | MATCH (`DispatchRejectsForeignAndDuplicateRequests`, `GrantCompletesOnlyAfterFinish`) |
| G: E22 foreign/duplicate/malformed/no live op | drop silently; manager times out | 685-705 | MATCH (same tests) |
| G: authority changes during grant | reply `reservation-revoked`; no load | 710-716 | MATCH (`AuthorityChangesDuringGrantLoadAndFinish grant/*`, `ManagerReplacementDuringGrant`) |
| L: E7 | -> S4 (pending 60 s); status `satisfied` if still current; no second load | 624-636 | MATCH (`GrantCompletesOnlyAfterFinish`, `CompletedLoadWaitsForWatcher`) |
| L: E8 terminal failure | -> S8; hold cleared; retry bounded | 599-603, 623, 637-640; status stays `unknown/operation-timeout` for `failed`/`timeout`/`cancelled` | MATCH on record; **LOW-1** (reason text); **RED-3** (retry bound) |
| L: E9 non-terminal (RPC error/timeout) | finish `unknown`; keep hold; -> S3 | 583-589, 603-611, 615-622, 546-560 | MATCH (`RecoveryRequiresTerminalManagerBarrier/transport`, `AbsolutePolicyDeadlineBoundsLoad`) |
| L: manager reply attests another incarnation/engine/model/identity | treat as unknown, not terminal | 591-595 | MATCH, untested |
| L: `loaded` without a recorded grant | treat as unknown | 596-602 | MATCH, untested |
| L: E11/E13/E15 during load | load cannot be revoked by the broker; manager-side veto owns it; completion still recorded as pending (fact), status `blocked` | 624-636 (pending set regardless of `residentCurrent`); manager `residency.go:427-431, 488-493` (INFERRED) | MATCH (`TemporaryVetoPreservesCompletedLoad/pause in flight`, `AuthorityChangesDuringGrantLoadAndFinish load/*`) |
| L: E18 during load, grant issued | finish attempted on original proxy; on failure -> S3 | 609-611, 550-554 | MATCH on this pass; the S3 record then has no exit, see **RED-2** |
| L: E19 during load | result cannot arrive; `terminal=false` -> S3 within proxy run | 583-589 | MATCH (`ManagerReplacementDuringLoadOrFinish/load`); duration **UNVERIFIED-2** |
| F: finish transport failure, granted | -> S3; re-finish next pass | 609-611, 550-554, 401-422 | MATCH (`TransientRecoveryRetries/finish transport failure`) |
| F: finish returns `RecoveryID`, terminal, source current | fresh `engine:models` read from the same source then `observe-resident` with `OperationComplete`, `Sequence 1`; -> S4 if loaded else S8 | 615-621, 643-677 | MATCH (`RecoveryRequiresTerminalManagerBarrier/terminal`, `TransientRecoveryRetries`, `RecoveryCannotUseReplacementManager`) |
| F: finish returns `RecoveryID`, not terminal | keep hold, no observe, -> S3 | 618 | MATCH (`RecoveryRequiresTerminalManagerBarrier/transport`) |
| F: finished reservation id mismatch | treat as unsettled | 612-614 | MATCH, untested |
| any: absolute deadline | one deadline for prepare, grant, load; finish uses a separate 5 s cleanup budget that does not extend the load | 517-519, 540 `TimeoutMs`, 581, 606-607 | MATCH (`AbsolutePolicyDeadlineBoundsLoad`) |

### S3 Unsettled hold

| Event | Required | Code | Verdict |
|---|---|---|---|
| E21 tick / any pass with state `pending` | re-send identical finish; if `RecoveryID` and `terminal` and source current -> recover; settled -> S4 (if both outcomes `loaded`) or S8 | 401-422 | MATCH (`TransientRecoveryRetries` x5) |
| E3 confirming | must not settle the proxy hold nor load; status `unknown` | 401 branch precedes 423; `residentProofLocked` leaves state `pending` then the branch sets `unknown/operation-timeout` | MATCH by plan (`ConfirmingInventoryCannotSettleUncertainHold`) |
| E1/E2 | record retired | 357-363 | MATCH (`TransientRecoveryRetries/revision replacement`, `PendingObservationAllowedExits/revision replacement`) |
| E11/E13/E15 | status `blocked`; record preserved | 236-242 give `blocked`, 399 skips the branch, record untouched | MATCH, untested |
| E18 proxy replacement | plan: "a new proxy run loses old in-memory holds ... wait for fresh inventory/identity before preparing there"; ruling: every in-flight fact has a bounded exit -> record dropped, -> S7 then S1 | 357-363 keep the record (only revision/desire checked); 405 re-sends finish to `hold.op.evidence.Proxy`, the retired process, whose `Call` fails (`proxy.go:314-321`); `settled` stays false forever; status `unknown/operation-timeout` and **no load for the rest of the revision**; one dead-peer 5 s call every tick | **RED-2**, untested |
| E19 manager replacement | plan: hold retained within the proxy run, no replacement-manager recovery | 407 `residentSourceCurrent` false -> never settles; proxy also refuses a new prepare while `inflight[k] > 0` (`residency.go` proxy `:218`) and keeps dispatched capacity across revisions (`:267-283`) | MATCH by plan; **UNVERIFIED-2** (duration is the whole proxy run) |
| E10 | n/a (no timer) | - | OPEN: none by plan |

### S4 Pending observation

| Event | Required | Code | Verdict |
|---|---|---|---|
| E3 confirming, same allocation | -> S5, no load | 364-377 (`confirmed[ref] = pending.evidence`, pending deleted) | MATCH (`CompletedLoadWaitsForWatcher`, `PendingObservationAllowedExits/confirming`) |
| E3 confirming, different authority (E18/E19 happened) | re-confirm via readback, no load | 293-295 returns `pending`; 423-429 `confirmResident` | MATCH, untested |
| E4 contradicting | ruling: not a clearer; stay S4, no load; published state OPEN | 291-301: state `satisfied` with `ObservedLoaded` forced true (same allocation) | MATCH on record (`CompletedLoadWaitsForWatcher` empty publication); **LOW-2** on status |
| E5 unknown | stay S4; status `unknown` | 296 | MATCH, untested |
| E10 timeout | record cleared; re-evaluate: loaded and current -> confirm; not loaded -> load; unknown -> S7 | 365-367 then 423/431 | MATCH (`ObservationTimeoutReevaluates`, `PendingObservationAllowedExits/observation timeout`) |
| E7 same key again | impossible: S4 never enters `reconcileResident` | 291-301 leave state != `pending` | MATCH |
| E1/E2 | cleared | 365 | MATCH (`PendingObservationAllowedExits`) |
| E11/E13/E15 then E12/E14/E16 | `blocked` while gated, record preserved, `satisfied` on resume without a load (until timeout) | 236-242; `residentProofLocked` only for `pending` entries; 297-300 | MATCH (`TemporaryVetoPreservesCompletedLoad` pause/Off/drain) |
| E17 | no load | 291-301 | MATCH (`CompletedLoadWaitsForWatcher`) |
| E18/E19 | preserved; `sameResidentAllocation` false -> `unknown` until fresh inventory; then confirm path | 285-287, 293-301 | MATCH, untested |
| E21 | no RPC | 347 | MATCH |

### S5 Confirmed

| Event | Required | Code | Verdict |
|---|---|---|---|
| E3 | stay | 303-305 | MATCH (`ConfirmingInventoryDoesNotReload`) |
| E4 contradicting (model unloaded by engine/pause) | -> S1 and reload (level-triggered) | `policy.go:1132-1141` deletes `confirmed`; 303 requires `ObservedLoaded` true | MATCH, untested |
| E5 unknown | ruling (NE readback): unknown latches nothing; -> S7, no load | `policy.go:1132-1141` deletes the proof even when the engine is merely absent; status `unknown`; later confirming observation needs one `get-launch` | MATCH on loads; **LOW-3** |
| E17 | stay, no load | 303-305, `sameModelAuthority` ignores `Active`/`Loaded` | MATCH (`CompletedLoadWaitsForWatcher`) |
| E1 identity unchanged | OPEN; recommended: proof retired per revision but observed-loaded fact kept; confirm from readback, no load | 352-356 delete proof; 122 drops `LoadedCurrent`; 431 loads | **RED-1** (`ReplacedRevisionReconcilesOnce` only covers a changed envelope, where a load is legitimate) |
| E2 | cleared; no unload | 352-356; no unload call site in `modelconfig.go` | MATCH (`StatusUnknownEmptyAndRemoval`) |
| E11/E13/E15 | `blocked`, proof preserved | 236-242, `residentProofLocked` skipped for non-pending | MATCH (`TemporaryVetoPreservesCompletedLoad`) |
| E18/E19 | proof no longer same authority; re-confirm via `get-launch` once fresh inventory shows loaded; no load | 160-172, 303, 423-429 | MATCH (`WorkerRestartConfirmsWithoutReload`) |
| E21 | no RPC | 347 | MATCH |

### S6 Gated and S7 Unknown

| Event | Required | Code | Verdict |
|---|---|---|---|
| any E6 | never | 236-249 return `blocked`/`unknown`; 399 requires `pending` | MATCH (invariant I2, I4) |
| E12/E14/E16, fresh inventory | re-evaluate with preserved records | records only cleared at 352-367 by revision/desire/expiry | MATCH |
| E13 Off while engine stops | engine stop deletes inventory -> S7 on resume until the manager republishes | `policy.go:1076-1091` | MATCH, untested |
| E5 payload without `loadedByEngine` | all engines unknown | `policy.go:1055-1063, 1157-1162` | MATCH (`NotificationWithoutLoadedInventoryClearsKnowledge`) |
| adopted engine (manager restarted, engine kept) | OPEN; recommended `unknown/observation-unknown` | 458 `launch.Adopted` -> `unknown` every pass with one `get-launch`; no load | **LOW-7** |

### S8 Failed, awaiting retry

| Event | Required | Code | Verdict |
|---|---|---|---|
| E21 tick, evidence unchanged | ruling: "failures retry, bounded" | 347/379: retried every 30 s indefinitely, no count, no backoff | **RED-3** (`FailedAttemptRetriesOnTick` codifies the tick retry) |
| evidence changes | retry once | 347 | MATCH (`FailedAttemptWaitsForChangedEvidence`, `ConcurrentEvidenceChangeIsReevaluated`) |

### Protection side (resident hard protection)

| Event | Required | Code | Verdict |
|---|---|---|---|
| idle unload planned/executed | re-read protected set immediately before each unload; only reason `pause` bypasses | `policy.go:1687-1695`; `idlepolicy.go:124-137` reads `r.current.ResidentModels` | MATCH (`TestResidentProtectionAutomaticUnload`, `RechecksQueuedUnload`, `PauseMayUnload`) |
| idle sleep | recheck at planning and before sleep | `idlepolicy.go:93, 108-112` | MATCH (`IdleSleepPlanning`, `RechecksQueuedIdleSleep`) |
| admission eviction (`unloadModels(..., "admission")`) | protected | `policy.go:1593, 1687` | MATCH, untested |

## 3. Findings

### RED-1: a revision change re-issues a load for an already-loaded resident (duplicate load)

- Claimed invariant: ruling "in-flight and pending-observation are facts, cleared only by a fresh confirming observation ..."; plan 5c "`satisfied` requires current authoritative loaded and effective-envelope evidence"; handoff RED class "duplicate load".
- Code path (VERIFIED): `policy:set` -> `schedulePolicyPush` -> `signalModelConfig` (`policy.go:1276-1277`); after activation `notifyPolicyChanged` -> pass. `reconcileModelConfig` deletes `confirmed[ref]` because the revision differs (`modelconfig.go:352-356`). `captureModelConfig` builds `LoadedCurrent` only when `r.models.loadedRevision == e.Policy.Revision` (`modelconfig.go:122`); `loadedRevision` is set only by an inventory observation (`policy.go:1130`). The manager publishes `engine:models-changed` only when the loaded set or memory status changes (`services/nvpair-engine-manager/loadedwatch.go:46-99`), so after an ordinary save nothing refreshes it. The entry is `pending` with `ObservedLoaded=true` (224-232, 250-266), case 423 fails (`LoadedCurrent` empty), and 431 `reconcileResident` issues prepare + `engine:reconcile-resident` with `Load`. The manager has no already-loaded short circuit: `executeResident` goes straight to `dispatchAction` (manager `residency.go:408-494`).
- Observed: one load RPC against the broker's own positive loaded observation on every policy save for every loaded resident. Worse, after the load the broker records a pending fact (635) which expires after 60 s (365-367) and, with `loadedRevision` still stale, the same path loads again; this repeats until some `engine:models-changed` arrives (UNVERIFIED-1 covers the engine-side effect of the repeated `POST /models/load`).
- Affected state: S5 on E1 (identity unchanged); S1 on E3 when `loadedRevision` is stale.
- Test coverage: `TestModelConfigReplacedRevisionReconcilesOnce` (`modelconfig_test.go:665-713`) changes the serving envelope (8192 -> 8193), so identity differs and a reload is legitimate there; the unchanged-identity case has no test.
- Disposition: repair now. Drop the `loadedRevision == revision` gate from `LoadedCurrent` (keep source/proxy/identity continuity, `modelconfig.go:119-133`) so E1 goes through `confirmResident` (one `get-launch`, no load). Keep `confirmed`/`pending` revision-bound as they are. Add the unchanged-identity test (section 5, row T-R1).

### RED-2: an unsettled hold survives proxy replacement and has no exit within the revision (stuck, lost load)

- Claimed invariant: plan 5c "On proxy replacement invalidate prior permission/status and wait for fresh inventory and activation before preparing a new run"; plan "A new proxy run loses old in-memory holds"; handoff invariant "every in-flight or pending record has a bounded exit".
- Code path (VERIFIED): a granted op whose finish fails or returns `RecoveryID` without a terminal manager result is stored in `unsettled` (`modelconfig.go:546-560`). The only clearer is `residentDeclarationCurrent` (357-363: revision and desire). Each pass re-sends finish to `hold.op.evidence.Proxy` (405); after proxy replacement that is the retired `*proxyProcess`, whose `Call` returns an error on a dead peer (`proxy.go:314-321`), so `settled` is false; for a non-terminal hold `settled` requires `hold.terminal` (407) which is fixed false. The key is published `unknown/operation-timeout` (402) and 399-401 never reach `reconcileResident`, so the new proxy run never loads the resident. The repeated proxy finish also returns the same immutable `RecoveryID` for a live proxy (`services/nvpair-proxy/residency.go:329-333, 359-361`), so even the live-proxy, non-terminal case has no exit except a policy edit.
- Observed: desired resident stays unloaded and `unknown` until the user edits policy; one 5 s dead-peer RPC per 30 s tick.
- Affected state: S3 on E18 (also S3 non-terminal on a live proxy; the latter is plan-accepted within a run, see UNVERIFIED-2).
- Test coverage: none. `TestModelConfigManagerReplacementDuringLoadOrFinish` and `TestModelConfigAuthorityChangesDuringGrantLoadAndFinish` stop at "replacement inherited nothing" and do not run a further pass after a proxy replacement with a hold.
- Disposition: repair now. In the 357-363 sweep also drop `unsettled` entries whose `op.evidence.Proxy != e.Proxy` (and `pending` entries likewise may stay, they already expire). Document that the retired proxy's capacity accounting died with it. Add test rows T-S3-E18.

### RED-3: failed loads retry at a fixed 30 s interval with no bound on count

- Claimed invariant: ruling "Failures retry, bounded"; handoff RED class "unbounded retry". Plan 5c (pre-ruling) wanted changed-evidence-only retry; the branch chose tick retry (`TestModelConfigFailedAttemptRetriesOnTick`, `modelconfig_test.go:757-769`).
- Code path (VERIFIED): suppression is `sameModelEvidence && now.Before(retryAt)` with `retryAt = now + idleCheckInterval` (`modelconfig.go:347, 378-379`); there is no per-key failure counter or growing backoff. Each retry costs `switch-prepare`, `prepare-resident`, `reconcile-resident` and, when the proxy grants, a manager `POST /models/load` that fails again, up to the 60 s deadline.
- Observed: a resident that cannot load (for example an engine that refuses the allocation) is attempted every 30 s for as long as the preference exists.
- Affected state: S8 on E21.
- Disposition: owner question on what "bounded" means, then repair: per-key failure count with exponential backoff capped (for example 30 s, 60 s, ..., 10 min) and reset on any meaningful evidence change; publish `blocked/unsupported-load` after N consecutive identical failures. If the coordinator reads "bounded" as the per-attempt deadline only, downgrade to LOW and record the decision.

### UNVERIFIED-1: engine effect of a repeated `POST /models/load` for a loaded model

- Invariant: plan "No engine call ... is a substitute for a refused load"; RED-1 produces repeated loads.
- Probe: on a real host with llama.cpp router mode, observe `POST /models/load {model}` for an already-loaded model (idempotent, error, or reload with VRAM churn). Not provable here; manager returns `loaded` if the readback lists it (`residency.go:494-501`).
- Disposition: real-machine probe after RED-1 repair; the repair removes the call either way.

### UNVERIFIED-2: a manager crash mid-load blocks the resident for the rest of the proxy run

- Invariant: plan "retain unknown status/hold within the current proxy run"; "Do not ... restart a worker merely to free capacity".
- Path (VERIFIED code, duration unproven): non-terminal hold -> `unsettled` with `terminal=false`; proxy keeps dispatched capacity even across revision changes (`services/nvpair-proxy/residency.go:267-283`) and refuses a new prepare while `inflight[k] > 0` (`:218`, `no-fit`). The broker clears its record on a revision change (357-363) but the proxy then refuses, so the resident is `blocked/no-fit` until the proxy restarts.
- Disposition: owner question (accept and document in 6a, or allow a replacement-manager fresh read under identical identity to settle the hold, which the plan currently forbids). Cross-process evidence belongs to `services/tests` on a real host.

### LOW findings

- **LOW-1** Definite failures publish `unknown/operation-timeout`: `reconcileResident` presets that reason (`modelconfig.go:561`) and only `not-dispatched` is remapped (637-639); `failed`, `timeout`, `cancelled` keep the preset. The bounded reason set (`wire.go:383-394`) has no load-failure reason. Disposition: amend plan (add a reason or map `failed` to `unsupported-load`).
- **LOW-2** During S4 a contradicting same-source observation publishes `satisfied` with `ObservedLoaded` forced true (`modelconfig.go:297-300`) for up to 60 s (for example after a pause unloaded the model and the user resumed within the window). Self-corrects on E10. OPEN recommendation: publish `unknown/observation-unknown` when the fresh observation says not loaded, keep the record.
- **LOW-3** An unknown observation (engine absent from `loadedByEngine`) deletes the confirmed proof (`policy.go:1132-1141`); no load results, one extra `get-launch` on the next confirming observation. Recommendation: skip the deletion when `loadedByEngine[engine]` is absent.
- **LOW-4** `containsString(e.Loaded[engine], ref.Model)` compares raw names while every other comparison normalizes (`modelconfig.go:368` vs 227); only affects `implied-latest-tag` engines (Ollama), where the shortcut never fires and the pending record expires into the `confirmResident` path. No load. llama.cpp is `exact`.
- **LOW-5** `residentCurrent` uses `time.Now()` (`modelconfig.go:503`) while every other clock read is `r.now()`; harmless in production, inconsistent with the plan's fake-clock rule.
- **LOW-6** Every idle tick runs a pass and its deferred `notifyPolicyChanged` (386-392) publishes `policy:changed` with an incremented `ObservationGeneration` (380) to all clients every 30 s even when nothing changed. Observable as traffic only; no state effect. Recommendation: notify only when the published `Models` differ from the previous status.
- **LOW-7** An adopted engine can never be confirmed (`modelconfig.go:458` rejects `launch.Adopted`), so a loaded resident on an adopted engine stays `unknown/observation-unknown` with one `get-launch` per tick; no load. Consistent with NE's no-strict-readiness-for-adopted rule; recommend documenting in 6a.

## 4. Cells with no branch test

Cells marked "untested" above, grouped:

1. S1/S5 on E1 with unchanged identity (RED-1): no test asserts zero loads after an unrelated policy save.
2. S3 on E18 (RED-2): no pass after a proxy replacement with an unsettled hold; no assertion that the new run eventually loads.
3. S3 on E11/E13/E15: record preservation under veto is only tested for S4 (`TemporaryVetoPreservesCompletedLoad`), not for S3.
4. S4 on E18/E19 and on E5: preserved record with changed authority or unknown inventory.
5. S5 on E4 (reload after the engine dropped a confirmed resident) and S5 on E5 (LOW-3).
6. S2.L with a manager reply attesting a different incarnation/engine/model/identity (591-595) and `loaded` without a recorded grant (596-602).
7. S2.F with finished reservation id mismatch (612-614).
8. S6 `Off` while the engine stops (inventory deleted at `policy.go:1091`) and resume.
9. `unloadModels(..., "admission")` protection (`policy.go:1593`).
10. Two residents on one engine where the first load changes `Active`/`Loaded` mid-pass (444-446 continues with stale `e.Loaded` for the second key; INFERRED safe because residents are protected from eviction).
11. Observation-timeout while gated (S4 + S6, then E10 while still gated, then resume): expected one load only if inventory says not loaded; untested.

## 5. One table-driven test for every cell

Sketch for `modelconfig_table_test.go` (broker module), built on the existing `newModelConfigFixture` (`modelconfig_test.go:24-88`), `fakeWorker` with `holdSuccessReply` (`policy_test.go:36-61`), `attachAckingProxy`/`attachFakeEngine` (`policy_test.go:282-311`) and `r.now` fake clock:

```go
type residentState int // S1..S8 as above
type residentEvent int // E1..E22 as above

type cell struct {
    from        residentState
    event       residentEvent
    wantState   nodepolicy.ModelConfigurationState
    wantReason  nodepolicy.ModelConfigurationReason
    wantLoads   int // engine:reconcile-resident calls after the event, cumulative
    wantPrepare int // prepare-resident calls, cumulative
    wantRecord  record // which of pending/unsettled/confirmed must exist afterwards
    needsHost   bool   // documents cells only provable on a real machine
}

func TestModelConfigTransitionTable(t *testing.T) {
    for _, c := range residentCells { // ~90 rows, one per non-trivial cell in section 2
        t.Run(fmt.Sprintf("%s/%s", c.from, c.event), func(t *testing.T) {
            f := newModelConfigFixture(t)
            m := newFakeManager(t, f) // below
            enter(t, f, m, c.from)    // drives the fixture into S1..S8
            apply(t, f, m, c.event)   // one event, including a reconcile pass where the ruling expects one
            assertCell(t, f, m, c)
        })
    }
}
```

`enter` drives: S2 phases by holding `prepare-resident`, `dispatch-resident` or the manager reconcile with `holdSuccessReply`; S3 by returning `RecoveryID` from finish with a non-terminal manager reply; S4 by `startLoadedAttempt` without a watcher publication; S5 by S4 plus `observeLoadedFrom(source, {engine: {model}})`; S6/S7 by `r.localPauseRequested`, `r.intent`, `r.drains`, `r.loaded` deletion; S8 by a terminal `failed` reply.

`apply` implements events: E1 by bumping `r.current.Revision`, `replayPolicyToProxy`, `reportStrict` with unchanged identity; E3/E4/E5 through `handlePolicyEngineNotification(source, engineModelsChangedMethod, ...)` with the three payload shapes; E10 by advancing `r.now` past `residentObservationTimeout`; E18 via `attachAckingProxy` then `observeLoadedFrom`; E19 via `attachFakeEngine` then a fresh `activateEngineManagerSource` and `bindEngineMgr`; E21 by advancing `r.now` by `idleCheckInterval + 1s`; E22 by `dispatchResidentFrom(source, request)`.

Fake engine manager (`newFakeManager`): a `fakeWorker` whose answer function is a small scripted state machine instead of per-test closures:

- `switch-prepare` -> `emptySwitchManagerSnapshot()` with a configurable `ManagerIncarnation`.
- `reconcile-resident` -> decodes `ReconcileResidentParams`, records them (asserting `Load.ContextLength` equals the fixture profile and `DeclaredResidentCount` equals the engine count), optionally emits a correlated `engine:dispatch-resident-request` through `f.b.dispatchResidentFrom` for the current source, then returns one of: `loaded` with a matching `ResidentLoadedObservation`, `failed`, `timeout`, `not-dispatched`, a transport error, or a hold (`holdSuccessReply`) released by the test; a flag makes it attest a foreign incarnation to cover 591-595.
- `engine:models` -> the scripted inventory (`nil`, `{}`, `{model}`) so recovery reads are controllable (`recoverResident` 653-666).
- `engine:get-launch` -> `residentLaunchReadback(e)` or a variant with `Adopted`, mismatched `Observed`, or `Status != observed`.
- A counter per method and a `waitFor` barrier (already on `fakeWorker`) replace timing guesses.

The proxy stays the existing acking fake with a scripted `finish-resident` (`RecoveryID` on/off, transport failure on/off, reservation-id mismatch) and `observe-resident` (`Settled` true/false, failure).

Expected rows that fail on `74878aa` and should be added first: T-R1 (S5, E1 unchanged identity: `wantLoads` 1, `wantState satisfied` after one `get-launch`), T-S3-E18 (S3, E18 then fresh empty inventory then pass: `wantLoads` 2, record `none`), T-R3 (S8, E21 x N: `wantLoads` bounded by the chosen policy).

## 6. Cross-sequence invariants and what enforces them

| # | Invariant | Enforced by | Status |
|---|---|---|---|
| I1 | At most one in-flight resident operation per node (hence per key) | `busy` (`modelconfig.go:347-351, 386-390`), sequential loop 393-447, single `active` 541 | VERIFIED (`ConcurrentTriggersCoalesce`, `BusyRetainsOriginalOwnership`) |
| I2 | No load decided while paused/Off/draining/not running/not active/unqualified/unsupported | `modelConfigEntry` 236-267 yields `pending` only in the default branch; 399 requires `pending`; mid-operation `residentCurrent` 565/580/710/714 via `sameModelAuthority` 160-172 | VERIFIED for decision and prepare/grant; load phase after grant is manager-vetoed (INFERRED, manager `residency.go:427-431, 488-493`) |
| I3 | No load for a key with a pending or unsettled record | `residentProofLocked` 291-301 moves S4 out of `pending`; 401 branch precedes 423/431 | VERIFIED |
| I4 | No load without a known inventory from the current source and proxy | 224-232 (`ObservedLoaded` nil when `Loaded` unknown or `Source` nil), 249; `captureModelConfig` 116-121 | VERIFIED |
| I5 | Every in-flight or pending record has a bounded exit | `pending`: `expires` 365-367; `active`: deadline 517-519 and `residentCurrent` 503; `unsettled`: **none** except revision/desire change (357-363) | VIOLATED (RED-2 for proxy replacement; plan-accepted within a live proxy run, UNVERIFIED-2) |
| I6 | One grant per operation | `grantStarted` 696/706, `granted` 719; proxy `h.dispatched` single-use (`residency.go` proxy `:303-307`) | VERIFIED |
| I7 | Grant reply and any resident work only through the original worker/source pair; a replacement never inherits | `op.evidence.Source == source` 696; `replyPeerRelay` pair check `remote_management.go:46-50`; `residentSourceCurrent` 506-509 | VERIFIED (`DispatchRejectsForeignAndDuplicateRequests`, `RecoveryCannotUseReplacementManager`) |
| I8 | `OperationID` strictly increasing per proxy run | 528-534; proxy gap check `residency.go` proxy `:189-193` | VERIFIED |
| I9 | Finish always on the proxy that prepared, never a replacement | `hold.op.evidence.Proxy` 405, `e.Proxy` 609 | VERIFIED (`AuthorityChangesDuringGrantLoadAndFinish`) |
| I10 | Revision replacement/removal retires every per-key record | 352-367 | VERIFIED |
| I11 | Removing a preference never unloads | no unload call in `modelconfig.go`; `unloadModels` callers `availability.go:242`, `idlepolicy.go:102`, `policy.go:1593` | VERIFIED |
| I12 | Desired residents are never auto-unloaded or idle-slept; only reason `pause` bypasses | `policy.go:1687-1695`; `idlepolicy.go:93, 108-112, 124-137` | VERIFIED (`TestResidentProtection*`) |
| I13 | No load is issued against a positive same-source loaded observation | 423 requires `LoadedCurrent`, which 122 ties to `loadedRevision` | VIOLATED (RED-1) |
| I14 | Retry of a failed attempt is bounded | none (347, 379) | VIOLATED or OPEN (RED-3) |
| I15 | `Load.ContextLength == profile.ContextTokens`; `DeclaredResidentCount` per engine, revision-bound | 581 (`pr.ContextTokens`), 214-223; `params.Validate()` 582 | VERIFIED (`LoadUsesRevisionProfile`, `InflightStatusRetainsEveryDesiredReference`) |
| I16 | Deadline <= min(60 s, switch timeout) and never extended by finish | 512-519, 606-607 | VERIFIED (`AbsolutePolicyDeadlineBoundsLoad`) |
| I17 | `satisfied` only with `ObservedLoaded == true` and no reason | `wire.go:691-706` `Validate`; `confirmResident` 453-487; 297-300 forces the flag (LOW-2) | VERIFIED structurally; honesty gap LOW-2 |
| I18 | Status publication does not loop | pass -> `notifyPolicyChanged` -> `signalModelConfig` -> next pass suppressed at 347 | VERIFIED (no loop); periodic tick publication LOW-6 |

## 7. Ranked repair list

1. **RED-1** Remove the `loadedRevision == revision` condition from `LoadedCurrent` (`modelconfig.go:122`), keeping source/proxy/identity continuity; add the unchanged-identity revision test. Prevents a load (and the 60 s re-load cycle) on every policy save for every loaded resident.
2. **RED-2** In the record sweep (`modelconfig.go:357-363`) drop `unsettled` holds whose `op.evidence.Proxy` is not the current proxy; add the S3/E18 test with a following fresh-inventory pass that must load once. Removes the only unbounded per-key record.
3. **RED-3** Decide the meaning of "bounded" with the owner; implement a per-key failure counter with capped exponential backoff reset on evidence change (or document the 30 s cadence as the bound if accepted). Replace `TestModelConfigFailedAttemptRetriesOnTick` with the chosen bound.
4. **LOW-2 / LOW-1** Publish `unknown/observation-unknown` during S4 when the fresh same-source observation contradicts; map terminal `failed` to a real reason (plan amendment: add a `load-failed` reason or reuse `unsupported-load`).
5. **LOW-3** Do not delete `confirmed` on an unknown observation (`policy.go:1132-1141`): skip when `loadedByEngine[engine]` is absent.
6. **LOW-6** Notify `policy:changed` from a pass only when the published entries changed.
7. **LOW-4 / LOW-5** Normalize in `containsString` at `modelconfig.go:368`; use `r.now()` in `residentCurrent`.
8. **Tests** Land the table-driven test (section 5) so future rounds stop finding one edge at a time; cover the 11 untested cell groups in section 4.
9. **UNVERIFIED-2 / LOW-7** Document in 6a: a manager crash mid-load blocks that resident until the proxy restarts; adopted engines stay `unknown`.

## 8. Owner challenges (separate from findings; the decisions stand)

- "Do not infer completion from ... a replacement manager." A replacement manager that re-adopts or restarts the engine and publishes a fresh same-identity inventory is at least as trustworthy as the original manager's post-crash state, yet the rule makes a manager crash during a load block the resident for the whole proxy run (UNVERIFIED-2). Consider allowing settlement from a fresh read under identical `EngineRunID`/digests from any current source.
- The 60 s observation timeout equals the manager's `inventoryReconcileInterval` (`models.go:117`) and the watcher only notifies on change; a loaded set that does not change never refreshes `loadedRevision`. Even after RED-1 is fixed, consider having the broker request `engine:models` once after a successful load instead of waiting for a change notification.
