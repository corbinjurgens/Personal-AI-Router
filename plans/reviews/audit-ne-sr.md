<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Review: audit of landed NE (Tasks 1-4 Part A, Task 6 Part C) and SR 5a-i

Kind: delivery audit (investigation only; no product change, no pull request).

Question: does the code on `fork/groundwork` deliver what the NE plan (`plan/ne-serving-envelope`, `plans/ne-serving-envelope.md`) promised for Tasks 1-3, Task 4 Part A and Task 6 Part C, and what SR Task 5a-i (`plans/context/sr5-rescope.md`, Task 501 plus coordinator amendment r2) promised, and do the two packages compose correctly?

## Verdict

`RED 0 UNVERIFIED 2 LOW 6`

Every Do obligation of NE Part A, NE Part C and SR 5a-i that can be checked by reading code or running module unit tests is FULFILLED at the inspected head. The two UNVERIFIED items are (1) the intended-but-operationally-severe latch of identity withdrawal on any transient `loaded_models`/`/props` read failure, and (2) the `-race`, desktop and Windows/native verification gates that this environment could not execute. The LOW items are test and reservation hygiene plus composition notes for SR 5a-ii/5b/5c.

## Inspection base and environment

- The common brief names `fork/groundwork` at `210babf`. The branch has moved: `origin/fork/groundwork` = `8a63866c1cbadd70f40732c84b04efb72d6e2fd4` (two first-parent merges later). This report inspects `8a63866`. Every `path:line` below is at that SHA.
- Two landed units sit after `210babf`: `8739691` "Merge roadmap/ne-t5b-readiness (NE Task 5 Part B)" and `8a63866` "Merge roadmap/i-public-lifecycle (test: deterministic borrowed alias assertion)". Part B is therefore landed at the inspected head although the handoff describes it as "under review". Part B is out of this audit's scope; it is referenced only where SR composition depends on it (`services/nvpair-proxy/readiness_contract.go:152-158`).
- Landed-unit map from `git log --first-parent`: NE T1 `97e9b87`/`9b85d52` (merged before T2), T2 `043d290`, T3 `2026d36`..`be09760` (merge `41492d4`), T4 Part A merge `75861b7` (commit `2cfac12`), SR 5a-i merge `e6a8f73` (commit `4606b2a`), follow-up test commit `aebf553`, T6 Part C merge `8f6dd0f` (commits `64384ef`, `515364b`), SR 5d merge `3940dd8` (`a392be0`), T5 Part B merge `8739691`.
- Toolchain: the host Go is 1.24.7 and every module declares `go 1.25.0`; with `GOTOOLCHAIN=auto` the 1.25 toolchain resolved and `go vet` / `go test` ran. `-race`, `npm` checks, `services/tests` and anything Windows-native were not run (see "Verification record"). No engine or model was started; manager tests use the fake engine binaries only.
- Claims are marked VERIFIED (code read or check executed) or INFERRED.

## Findings

### UNVERIFIED-1: any transient loaded-inventory read failure latches identity withdrawal until an explicit apply or restart

- Claimed invariant (NE Task 6 Part C, Do): "Missing/unknown loaded semantics and mismatch withhold verified state and withdraw identity using A's run-bound drift reason; cold/unloaded has no observation and is not a mismatch." NE Task 4 Part A: "Once set, drift remains withdrawn until successful apply/start; an inventory response cannot restore it."
- Code path (VERIFIED): `services/nvpair-engine-manager/models.go:161-168` defers `observeLlamaServing` for every running owned llama engine on every sweep; `models.go:196-212` leaves `loadedOK[i]` false on any `loaded_models` transport error, 5-second sweep timeout (`models.go:22,125`) or shape error; `services/nvpair-engine-manager/serving_readback.go:117-118` maps `!known` to reason "Loaded model residency is unknown." and `serving_readback.go:149-151` writes it into `servingDrift`, which `model_identity.go:166-168` and `serving_identity.go:75,115` treat as withdrawn identity (explicit empty `ModelDigests`, no `Serving`). A `/props` timeout or non-200 (`serving_readback.go:133-136`) latches the same way. The latch clears only in `lifecycle.go:370` (successful owned start) and `settings.go:602-604` (successful `ConfigureLaunch`). `spec.md:171-173` documents this as intended.
- Observed result: the code implements the plan literally. The behaviour this produces on a real host is that one blip of the 5-second sweep (engine busy, slow disk during a child load, a sweep deadline consumed by the inventory read) withdraws strict identity, strict readiness and resident eligibility for that engine until an operator re-applies settings or restarts the engine. Cold/unloaded sets are handled correctly and do not latch (`serving_readback.go:120,153`; `serving_readback_test.go:200-248`).
- Affected state: `Runtime.servingDrift`, published `engine:model-identity`, proxy strict capacity, SR resident permission (`services/nvpair-proxy/residency.go:70-73,86-91`).
- Why UNVERIFIED rather than RED: no plan sentence is violated, and the direction is conservative. Whether sweeps blip under real llama.cpp loads is only provable on MSI/PC3.
- Disposition: owner question (see "Owner challenges"); if accepted, amend NE Part C so that a transport-level failure (error, timeout, non-200) expires proof like a failed sweep ("A failed sweep cannot retain last-good observed proof", `spec.md:181-182`) while only semantic unknowns (unfamiliar status value, undeclared loaded ID, allocation mismatch, invalid shape) latch.

### UNVERIFIED-2: race, desktop and native verification gates not executed here

- Claimed invariants: NE Part A Verify "run `go test -race -count=1 ./...` and `go vet ./...` from each of services/shared, services/nvpair-engine-manager, services/nvpair-ui-broker ... Run `npm --prefix desktop run typecheck` and `npm --prefix desktop run service-contracts:check`"; Part C Verify "`go test -race -count=1 ./...`"; SR 501 Verify "repeat with `-race` on Mac"; NE "Windows-native - separate approval gate".
- Executed here: plain `go vet` and `go test -count=1` per module (record below). Not executed: `-race`, desktop typecheck/contracts, Windows symlink/path/quoting evidence, PC3/MSI live gates.
- Disposition: defer to the owner's Mac/PC3/MSI runs; the acceptance checklist at the end lists what only those machines can prove.

### LOW-1: manager cold bound reads `Configured` regardless of readback status

- `services/nvpair-engine-manager/residency.go:340-344` takes `launchStateLocked(...).ServingReadback.Configured`, which `settings.go:144-146` fills from a fresh cold compile of the preset on disk, before `settings.go:157-160` marks the readback `mismatch` when that compile no longer matches the launched digest. `residentConfiguredContext` (`residency.go:357-374`) ignores `Status`. If the preset on disk changed after launch and the 60-second drift sweep has not yet latched, the bound comes from the edited preset while identity (checked first, `residency.go:326-328,335-337`) still reflects the launched one.
- No observable effect at this head: for every NE-configured engine `residentLoadCall` (`residency.go:218-221`) then returns "unsupported-load" (`residency.go:346-348`), exactly as r2 requires ("Resident execution stays unsupported in 5a-i exactly as at base"). INFERRED: the window closes at the next full-inventory sweep.
- Disposition: amend 5a-ii to bind the manager pre-load bound to `currentServingEnvelope` (`serving_identity.go:146-172`), which validates the launched digest, instead of `ServingReadback.Configured`.

### LOW-2: `TestResidentContextBoundColdPreflight` would pass if `residentCheck` dropped the bound

- `services/nvpair-engine-manager/residency_test.go:229-263` (as rewritten by `aebf553`) expects `"unqualified"` for all three cases, because the fixture has no owned NE launch and the identity check (`residency.go:326-328`) refuses first. The bound itself is asserted only through a direct call of `residentConfiguredContext` (`residency_test.go:243`), so removing the call at `residency.go:340-344` leaves every manager test green. There is no manager case with owned identity where an over-bound request yields `"unqualified"` and an exact request reaches `"unsupported-load"`.
- Disposition: repair now (test only): add a case built on `ownedServingFixture` (`serving_identity_test.go:369`) or `loadedReadbackFixture` with the fixture's per-slot context (8192), asserting `"unqualified"` for 8193 and `"unsupported-load"` for 8192.

### LOW-3: Part C edited files outside its reservation

- Part C Files: `serving_readback.go`, `serving_readback_test.go`, `registry.go`, `loadedwatch.go`, `models.go`, `models_test.go`, `settings.go`, `settings_test.go`, `spec.md`, `MANIFEST.md`. The merge `8f6dd0f` also changed `lifecycle.go` (`doStart`, `doStop`, `reconcileFailedCommandStop` runtime copies), `setport.go` (`canMoveAdoptedEngine` signature) and `serving_identity_test.go`, and did not touch `models_test.go` or `settings_test.go`. The extra edits are the lock-guard fixes the Part C hot spot asks for (correct in substance; see composition check 3). Part A stayed inside its Files (three declared test files were simply unused).
- Disposition: amend the plan's landed marks to record the actual file set; no code change.

### LOW-4: a successful unchanged re-apply clears an observation-mismatch latch without a restart

- `settings.go:596-604`: when `changed` is false and the engine is running, nothing restarts, yet `servingDrift` is cleared. An apply does re-validate approved bindings and the native source (`settings.go:508-519,543-559`), so input drift cannot be cleared this way; but a Part C drift reason set by a loaded-allocation mismatch (`serving_readback.go:134-135`) is cleared while the mismatched child is still loaded. The next sweep (at most one poll interval) re-latches, and `servingObserved` stays nil meanwhile, so no `observed` status or observation proof is published in the window (`serving_readback.go:157-160`).
- Plan text ("until successful apply/start") is satisfied. Disposition: defer; note for 5b that "successful apply" is not proof the loaded child changed.

### LOW-5: two runtime reads rely on `opMu` rather than `st.mu`

- `residency.go:340` reads `st.plat.Runtime.Serving` holding only `opMu`; `serving_strata.go:589` likewise. The sole writer, `ConfigureLaunch` (`settings.go:503,575-584`), holds `opMu` and `st.mu`, so this is serialized and not a data race; it departs from the convention Part C established elsewhere (`settings.go:102-105,137-139,317-319,529-531`; `lifecycle.go:152-155,609-611,889-891`). All remaining `st.plat.Runtime` reads outside `st.mu` touch manifest-static fields only (`actions.go:272,331`, `install.go:69,193,202`, `lifecycle.go:999`, `modellayout.go:78`, `status.go:157`); the whole-struct copy in `switchexecutor.go:174-180` is under `st.mu`.
- Disposition: defer (cosmetic); fold into 5b when `residency.go` is next edited.

### LOW-6: composition forces profile == qualification == native per-slot context for residency

- Proxy 5a-i: `services/nvpair-proxy/residency.go:115-116` requires `ValidateResidentContext(pr.ContextTokens, q.ContextTokens)` (profile at or below qualification). Part B: `readiness_contract.go:152-156` requires `q.ContextTokens <= envelope.PerSlotContextTokens`. Manager 5a-i: `residency.go:373` requires `p.Load.ContextLength == PerSlotContextTokens`. SR 5c will derive `Load.ContextLength` from the same profile. So a resident load is only grantable when profile context, B qualification context and native per-slot context are exactly equal; a conservative qualification below the native allocation refuses residency, by design ("A smaller requested number cannot shrink an already configured native allocation").
- Not a defect. Disposition: owner question (Q3 freeze): record B qualification `contextTokens` exactly at the preset per-slot value for every intended resident, and tell 5c to copy `pr.ContextTokens` into `Load.ContextLength` unchanged.

## Obligation checklist

Status: FULFILLED / FAILED / UNVERIFIED. "Landed history" tasks (T1-T3) were spot-checked, not re-audited line by line.

### NE Task 1 (landed history)

| Obligation | Status | Evidence |
|---|---|---|
| Shared versioned declaration/readback types with strict structural validation | FULFILLED | `services/shared/engines/serving.go:18-24,84-135,177-246` (version 1 only, unique model IDs, digest shape, memory domains, int64 text mtime) |
| Semantic equality and deep copies so a port/text edit cannot erase serving | FULFILLED | `services/shared/enginesettings/settings.go:19-32`; manager `setport.go:128-130,207-209` (nil pointer preserves on-disk serving); desktop `EngineSettingsSection.tsx:50,171`; TUI passes the shared `Config` through, `nodedetail_test.go:285-314` |
| Typed bridge parsing without casts | FULFILLED (spot check) | `desktop/src/electron/service-bridge/engine-settings.ts:124-185` |

### NE Task 2 (landed history)

| Obligation | Status | Evidence |
|---|---|---|
| Finite preset defaults, no router envelope overrides, CLI precedence | FULFILLED | `serving_llamacpp.go:568-583` (env < global < model < CLI), `:596-597` (`models-preset`/`models-max` never per model) |
| Reject auto/zero/unlimited, divide context across slots, unified-KV cap | FULFILLED | `:598-636` |
| Source digest rechecked before spawn, before side effects | FULFILLED | `lifecycle.go:319-325` |
| Router capacity recorded, not enforced | FULFILLED | `:551,686`; MANIFEST.md:99 |

### NE Task 3 (landed history, owner-deferred)

Preserved: `serving_strata.go:233-251,573` still compiles a single-model envelope and shares readback. Not re-audited beyond preservation (owner deferral 2026-10-21).

### NE Task 4 Part A

| Obligation | Status | Evidence |
|---|---|---|
| D1: capture pre-snapshot launch; versioned NE hash over `launchDigest` + `prepared.Digest`; legacy hashing unchanged; generation dirs excluded | FULFILLED | `lifecycle.go:287-289,326-327,361-363`; `serving_identity.go:42-54` (`pair-serving-launch-v1`); regression `serving_identity_test.go:38,79` |
| D6: canonical content manifest per model in native shard order, excluding paths/mtimes/samples; same digest in `ContentDigest` and `ModelDigests` | FULFILLED | `serving_identity.go:18-40,58-68,87-91`; `serving_llamacpp.go:724`; `serving_identity_test.go:116,275` |
| Identity only from owned current run, declared model, matching compiler result, no drift; adopted/external runs publish none | FULFILLED | `model_identity.go:164-168`; `serving_identity.go:72-76,111-116` |
| Copy nested maps/slices (no aliasing) | FULFILLED | `serving_identity.go:65,87-91` (`Clone()`), broker `model_identity.go:82-91,181-190`; `model_identity_test.go:23` mutation checks |
| Optional `Serving map[string]ServingEnvelope` on manager and shared identity; broker decode/pass-through; nil for legacy | FULFILLED | `model_identity.go:41-46`; `services/shared/nodepolicy/wire.go:1244-1250,1285-1289`; broker `model_identity.go:30-36,89,172` |
| Shared validation: key correspondence, matching content digest, finite context/slots/output (output < per-slot), explicit memory, valid source digest | FULFILLED | `services/shared/engines/serving.go:145-158`; `wire.go:1285-1289` (keyed to `ModelDigests[model]`) |
| Broker retains applied E fingerprint and withdrawal ordering | FULFILLED | broker `model_identity.go:161-179,226-270` unchanged in substance |
| No B provenance acceptance/schema revision | FULFILLED | no proxy file in `75861b7`; `engines/serving.go:223-227` kinds unchanged |
| Documentation of aggregate digest and HF-LFS/external shasum procedure, copying digests into B records | FULFILLED | `docs/local-qualification.mdx:197-239`; `spec.md:124-127`; `MANIFEST.md:84-86` |
| D7: exact `-1` retention through preset/CLI/env; zero and < -1 rejected; `Retention` string; `servingRetentionDisabled` | FULFILLED | `serving_llamacpp.go:221-227`; `serving_identity.go:56`; tests `serving_llamacpp_test.go:43-63` (global, per-model, CLI, env), `:84-88` |
| No capacity-vs-declaration comparison (SR 5b owns it) | FULFILLED | none found; `MANIFEST.md:99`, `spec.md:150-152` |
| D8: `currentServingEnvelope(ctx, engine, model) (engines.ServingEnvelope, engineModelIdentity, error)`; snapshot under locks, validate outside, recheck, typed refusal, never loads/probes | FULFILLED | `serving_identity.go:95-172` (exact signature; `ServingIdentityRefusal`); tests `serving_identity_test.go:149,188,314,343` |
| D9: drift check on `fullInventory` sweeps only via existing watcher; sampling even with unchanged stat; run-bound drift stored in `registry.go`; empty `ModelDigests`/no `Serving`; `mismatch` readback; latched until apply/start; replaced run not withdrawn by old sweep | FULFILLED | `loadedwatch.go:170-172`; `serving_identity.go:177-197`; `serving_files.go:50-95` (head/tail 64 KiB, stat before/after); `registry.go:183-184`; `settings.go:133,147-160`; `serving_identity_test.go:368-425` (byte 0 flipped under preserved mtime, restore does not unlatch, restart does) |
| No full model hash in any path; sparse 80 GiB fixture with 10 s deadline; skip with named reason | FULFILLED | `serving_identity_test.go:427-470` (`Truncate(80<<30)`, `fileHashes` inspected); binding reads at most 128 KiB |
| Symlink retarget, unstable read, relocation with renewed binding | FULFILLED (symlink case skips where symlinks are unavailable) | `serving_identity_test.go:205,239,275` |
| Verify: shared/manager/broker `go test -race`, `go vet`; desktop typecheck and contracts unchanged | UNVERIFIED here (plain vet/test ran; see record) | UNVERIFIED-2 |
| Files: no proxy file, none of SR 5a-i's six paths | FULFILLED | `git show --stat 75861b7` |

### NE Task 6 Part C

| Obligation | Status | Evidence |
|---|---|---|
| llama-router-specific manager GET, not a generic manifest action; `/props?model=<encoded>&autoload=false`; no redirect | FULFILLED | `serving_readback.go:71-82`; test `serving_readback_test.go:179,297` |
| Only names freshly reported by `loaded_models` this sweep/run; never a cold model, router placeholder or wake | FULFILLED | `models.go:163-168,196-212`; `serving_readback.go:25-37,103-139`; tests `:200,231,262` |
| Existing bounded client/cancellation; network I/O outside locks | FULFILLED | `serving_readback.go:79-88,114-140` (lock released before HTTP); `:391` cancellation test |
| `n_ctx` already per-slot, `total_slots` as slots, no second division | FULFILLED | `serving_readback.go:91-100`; fixture `n_ctx 8192, total_slots 2` against envelope 8192/2 (`serving_readback_test.go:144,287-288`) |
| Observed published only from fresh matching current-run/config | FULFILLED | `serving_readback.go:146-153,157-170` (freshness budget), `settings.go:150-156`; tests `:405,419` |
| Unknown/mismatch withhold and withdraw via A's drift reason; cold is not a mismatch | FULFILLED (but see UNVERIFIED-1) | `serving_readback.go:117-151`; tests `:200,282-333` |
| Recheck run/config after I/O; invalidate on unload/source/config/process transitions | FULFILLED | `serving_identity.go:123-127`; `serving_readback.go:107,146`; `loadedwatch.go:188-201` (action poke); tests `:335,361,377,440` |
| Record the observation lookup contract for SR 5a-ii | FULFILLED | `serving_readback.go:172-193`; `spec.md:186-190` |
| No Strata readback, no services/tests | FULFILLED | `serving_readback.go:32-34` refuses non-llama |
| Verify: `go test -race`, `go vet` manager | UNVERIFIED here (plain ran) | UNVERIFIED-2 |
| Files reservation | deviates | LOW-3 |

### SR Task 501 (5a-i) with coordinator amendment r2

| Obligation | Status | Evidence |
|---|---|---|
| `ValidateResidentContext(contextLength, qualifiedContextTokens int) error`, direct comparisons, no overflow | FULFILLED | `services/shared/nodepolicy/policy.go:637-645`; `policy_test.go:17-39` (10 cases incl. max int) |
| `Policy.Validate` structural and wire/keep-alive validation unchanged | FULFILLED | `e6a8f73` diff touches only the helper in `policy.go`; `wire.go` untouched |
| Proxy prepare: unique normalized `ProfileFor`, current installed qualification, compare under admission mutex before counting capacity; capture context in hold | FULFILLED | `services/nvpair-proxy/residency.go:86-91,98-120,199-209,218-222` |
| Revalidate profile/qualification at final dispatch; preserve revision/generation/identity invalidation and single-use grant | FULFILLED | `residency.go:249-253,257-272,280,289-305` |
| Refusal allocates no counted capacity, zero engine effects | FULFILLED | `residency.go:219-220` (no `counted`); `residency_test.go:76-82` |
| No `load` on Prepare/Dispatch DTOs | FULFILLED | `wire.go` unchanged in `e6a8f73` |
| Manager `residentCheck`: cold `ServingReadback.Configured` for the unique normalized model via existing helpers; equality with checked int64/int conversion and the shared helper | FULFILLED | `residency.go:332-374` (`count != 1`, `uint64(bound) > maxInt`, `== bound`) |
| r2: no retention or router-capacity check in 5a-i; resident execution stays unsupported | FULFILLED | no such comparison in `residency.go`; `:218-221,346-348` keep Ollama-only |
| No compiler/projection/strict identity created | FULFILLED | reuses `launchStateLocked` |
| Tests: exact, bound+1, absent/nonpositive, max int, below `1<<24` above limit, per-model limits, normalized collision, native/profile mismatch, replacement between prepare and dispatch; zero capacity/effects on refusal | FULFILLED | shared `policy_test.go:17-39`; proxy `residency_test.go:45-146`; manager `residency_test.go:199-263` |
| Verify: shared nodepolicy, proxy `^(TestResident|TestStrict)`, manager `^(TestResident|TestNotification)` plus `-race` on Mac, vet | FULFILLED for plain runs here; `-race` UNVERIFIED | record below |

## Composition checks (handoff item 3)

1. Identity precedes envelope. VERIFIED. Manager: `residentPrecheck` (incarnation, veto, desired, running, installed, `residentIdentity`) runs before the cold bound (`residency.go:335-344`); a cold exact allocation without owned identity returns `"unqualified"` (`residency_test.go:259-260`). Proxy: identity presence, installed digest and strict qualification precede `residentContextLocked` (`residency.go:70-91`), and `qualifiedModelLocked` checks launch/content digests before the Part B envelope guard (`readiness_contract.go:132-139` vs `:152-158`). Broker: an envelope keyed to a model without a content digest fails `EngineIdentity.Validate` (`wire.go:1285-1289`) and the engine is left out of the push (`model_identity.go:173-175`).
2. SR residency vs NE readiness ordering. VERIFIED. Resident permission is derived from the same `qualifiedModelLocked` strict readiness uses, so readiness withdrawal (identity change, envelope change through generation bump) revokes undispatched holds via `invalidateResidentHoldsLocked` (`residency.go:249-272`); a dispatched hold keeps its capacity but loses permission, per the landed contract. Consequence noted in LOW-6: profile, qualification and native per-slot context must coincide.
3. Lifecycle mutex guards around runtime copies. VERIFIED. All whole-`Runtime` copies are now taken under `st.mu` (`settings.go:102-105,137-139,317-319,529-531`; `lifecycle.go:152-155,609-611,889-891`; `switchexecutor.go:174-180`; `serving_identity.go:108-121`). Snapshot pointers (`servingLaunch`, `Serving`) are compared by identity on recheck (`serving_identity.go:124-127`), and `llamaServingLaunch` is never mutated after `compileLlamaServing` returns (only `:723-724` inside the compiler). Residual `opMu`-only reads: LOW-5.
4. Settings edits preserving NE configuration. VERIFIED at manager (`setport.go:128-130,207-209`; `ConfigureLaunch` passes the previewed `Serving` explicitly, `settings.go:571`), TUI (shared `Config.Clone`, test `nodedetail_test.go:285`) and desktop draft (`EngineSettingsSection.tsx:171`). Desktop unit tests were not run here (INFERRED from source).
5. No migration blesses existing processes, files or settings. VERIFIED. Declarations without a launch read back `unverified` (`settings.go:611-616`); apply requires the exact reviewed native digest and bindings (`settings.go:517-519`, `serving_llamacpp.go:672-674,717-721`); adopted or command-mode runs publish no identity (`model_identity.go:164`, `serving_identity.go:111`); legacy launches keep `launchDigest(launch)` (`lifecycle.go:361-363`) and inventory digests (`model_identity.go:174`); drift is not restored by inventory (`serving_identity.go:174-176`, test `:402-407`).

## Tests that would still pass if the property broke (handoff item 4)

- `TestResidentContextBoundColdPreflight` (manager) - LOW-2: the `residentCheck` integration of the cold bound is unobservable because identity refuses first in every case.
- `TestResidentContextBoundPrepare` (proxy, `residency_test.go:45-94`): "bound plus one", "absent bound", "negative bound" and "below protocol ceiling" are already refused by the pre-existing `pr.ContextTokens > q.ContextTokens` check in `qualifiedModelLocked` (`readiness_contract.go:140`); only "absent profile context", "negative profile context" and "normalized collision" prove the new helper call. Adequate, but the case names overstate what they add.
- `TestLoadedServingMissingEndpointWithdrawsProof` (`serving_readback_test.go:250`) removes the whole `Actions` map; it does not cover a transport failure of an existing `loaded_models` action, which is the UNVERIFIED-1 path. Add a case where the fake returns a connection error or 500 once and assert the intended behaviour, whichever way the owner decides.
- `TestServingSymlinkRetargetWithdrawsIdentity` skips where symlinks are unavailable (`serving_identity_test.go:215`); on Windows this is the case that matters (acceptance item 4).

## Real-machine acceptance checklist (handoff item 5)

Only provable on Windows with the pinned llama.cpp build, within the owner's approved intervals, using the frozen NE host record:

1. Router-reported model `id` in `GET /models` equals the preset section name exactly, and `status.value` is one of `loaded|unloaded|loading|sleeping|downloading|downloaded` across a full load/unload/sleep cycle (`serving_readback.go:60-66` latches on any other value).
2. Child `/props?model=<id>&autoload=false` returns `default_generation_settings.n_ctx` equal to the preset per-slot value and `total_slots` equal to `parallel`, for both MSI models and the PC3 model; a cold model is not loaded by that request (watch the router log and process list).
3. Sweep stability: with one model loaded and one cold, run for at least 10 minutes and confirm settings readback stays `observed` and `engine:model-identity` keeps `serving`; if it flips to `mismatch`, capture the reason text (UNVERIFIED-1).
4. Windows binding semantics: HF snapshot symlink targets resolve (`serving_files.go:55`), drive-letter/case normalization gives stable `sourcePath`/`resolvedTarget`, and a second owned start reproduces the same `launchDigest` (`serving_identity_test.go:38` on Windows).
5. Same-stat change detection: flip one byte in the first or last 64 KiB of a GGUF with mtime preserved and confirm withdrawal on the next full sweep; restore and confirm it stays withdrawn until apply/restart.
6. `sleep-idle-seconds = -1` in the preset is accepted by the pinned build and the child never sleeps; `0` and `-2` are refused by preview.
7. Resident preflight: with `[model]` per-slot context C, a `node/reconcile-resident` with `load.contextLength = C` is refused `unsupported-load` (5b not landed) and with `C+1` or `C-1` is refused `unqualified`, with zero `/models/load` calls (`residency.go:340-348`).
8. The PC3 process gate `TestNEServingEnvelope` (`services/tests/serving_envelope_test.go:761`, Part B) and `-race` runs of shared/manager/broker/proxy remain pending per UNVERIFIED-2.

## Owner challenges

- Latch-on-unknown (UNVERIFIED-1). The plan's choice makes every transient read failure an operator incident (re-apply or restart to recover strict routing). The milestone asks for "accurate-or-unknown telemetry"; treating a transport failure as "unknown, proof expired, retry next sweep" would keep that honesty while reserving the latch for semantic disagreements (unfamiliar status, undeclared loaded ID, allocation mismatch). The current code already expires proof on a failed sweep; the extra step is to not write `servingDrift` when `!known` is caused by transport rather than by shape.
- Exact-equality residency (LOW-6). Q3's freeze should state B qualification `contextTokens` per resident model as exactly the preset per-slot value; otherwise 5c will produce only `unqualified` refusals.

## Verification record

Commands run from each module directory at `8a63866` with `GOTOOLCHAIN=auto` (Go 1.25 resolved; host 1.24.7). No `-race`, no `services/tests`, no npm, no engines.

- `services/shared`: `go vet ./...` exit 0. `go test -count=1 ./...`: `nodepolicy`, `engines`, `enginesettings` and all other packages ok except `clustertrust` (`TestReverseDialPrivateEndpointPreservesSaturatedUnlockedLiveOwner`, networking in this container; unrelated to NE/SR). `go test -run '^TestResidentContextBound' -v ./nodepolicy`: 10 subtests pass.
- `services/nvpair-proxy`: `go vet ./...` exit 0. `go test -count=1 -run '^(TestResidentContextBound|TestResident|TestStrict|TestServedResident|TestQualified|TestReadiness)' .` ok; `TestResidentContextBound*` 11 subtests pass. Full `go test -count=1 ./...` has three failures unrelated to NE/SR: `TestPublicTransportBounds`, `TestPublicTransportErrorBodyBounds`, `TestPublicTransportFinalGuard` (public DNS refused by the container egress) and `TestHandleHTTP_RealSocketFlushDeadline` (socket flush timing).
- `services/nvpair-engine-manager`: `go vet ./...` exit 0. `go test -count=1 -skip PersistPort ./...` fails only in two groups unrelated to the audited units: every `TestStrata*` serving case fails with "Strata/Python environment overrides are unsupported in prepared mode" because this container inherits a `PYTHON*` environment variable that `serving_strata.go:302-305` refuses by design (Strata is owner-deferred), and `TestStopDeclinesLiteralDeletedSuffixListener` / `TestStopReclaimsManagedLiteralDeletedSuffix` fail on `/proc` "(deleted)" image inspection inside the container. The audited selection `go test -count=1 -run '^(TestResident|TestNotification|TestServing|TestLoadedServing|TestCurrentServing|TestLlamaServing|TestDesired)' .` is ok (64 s); `TestResidentContextBoundConfigured` (14 subtests), `TestResidentContextBoundColdPreflight` (3), `TestServingSymlinkRetargetWithdrawsIdentity` and `TestServingSparseModelNeverFullyHashed` all PASS without skips on this Linux filesystem.
- `services/nvpair-ui-broker`: `go vet ./...` exit 0. `go test -count=1 -skip EngineManagerPortsSpawn ./...` ok (all packages).
