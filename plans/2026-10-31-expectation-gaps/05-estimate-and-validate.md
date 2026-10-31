<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution plan: Phase 05: a visible estimate and a measured Validate bound to model, build and hardware

## Goal

Overall goal of the batch (`README.md` in this folder, section 4): PAIR knows what
it serves (phase 01), every served configuration is a named, copyable profile
(phase 02), roles and the desktop follow the owner's model (phases 03, 04), and
routing uses speed and context (phase 06).

This phase's goal (README section 3 "T-C and T-D", section 5 row 05; owner
expectation 11 in vault `Expectation Drift.md`): for every llama.cpp profile,
PAIR shows its own memory estimate exactly as the service computes it (per-pool
fits/tight/exceeds verdict, expected and high range, "not priced" warnings),
lets the owner run anyway or state a figure, and offers "Measure this
configuration": a consented, cancellable run at low and full context, startable
from another node unless the target is halted, whose memory and speed result is
recorded against the exact model file, engine build (`build_info`), GPU, CPU and
RAM, and whose speed also lands in the broker performance store for routing.
Each profile load also keeps its native startup output for inspection, and a
small load-time guard stops a load when available RAM falls below a floor.
Nothing here blocks Apply, role assignment or routing: an unmeasured profile
stays assignable and routable.

## Change scope

Intent: the owner can see a usable estimate and validate a profile on the machine it runs on, instead of hand-run agent sweeps.
Scope: bounded improvement — execute the unlanded tasks of three settled vault Todos (cited by task number below) with the stated amendments, extend the configuration key and its provenance, retain per-load native output, add one advisory load guard; leave routing, admission, ME2 qualification, Strata (EMP Tasks 4-6) and calibration bands (estimation Task 14) unchanged.
Basis: owner expectation 11 and T-D (vault `Expectation Drift.md`, `Tasks/2026-10-31 PAIR expectation gaps sweep.md`); README section 5 row 05; the three Todos' accepted decisions (decided.md 2026-10-29).
Compatibility: pre-release fork rule (no version bumps, capability markers, migrations or compatibility shims; wire formats change in place). Existing stated-memory records whose key changes in Task 3 become non-current; that is accepted, not migrated.
Non-goals: measured or estimated figures as admission, qualification, `memoryBytes`, policy or launch authority; automatic or background measurement; searching for the largest context that fits; Ollama, LM Studio or Strata estimates or measurement; real calibration bands (estimation Task 14); routing on measured speed (phase 06 consumes the samples); a hold that stops repeated loads after a guard stop.

## Entry conditions

Check each before Task 1 and record the evidence in the phase's close-out. If one is absent, stop and report; do not implement a missing prerequisite here.

1. **Phase 01 landed** (`plans/2026-10-31-expectation-gaps/00-index.md` row 01 ends with ` — done <sha>`, and that sha is an ancestor of your base: `git merge-base --is-ancestor <sha> HEAD`). This phase uses from it, by the names its close-out records:
   - the structured model identity carried in `ModelsResult` with `contentId` and the quant label;
   - the typed CPU expert option (estimation Task 10 steps 1, 2 and 5: field, option machinery, conversion, copy, editor control) and typed `n-cpu-moe`;
   - the engine build identity read from llama-server `/props` `build_info`, replacing the `b11146` manifest pin in device enumeration (`modelsettings_devices.go`), the reviewed native source (`serving_llamacpp.go modelReviewedNativeSource`) and buffer capture (`model_buffer_parser.go modelBufferBuildSupported`) (verify-A A7).
2. **Phase 02 landed** (row 02 done, ancestor of HEAD): every served llama.cpp model is a typed load keyed by a profile id, the served model id of a profile is its profile id, and the profile carries the phase 01 identity. In this phase every `modelId` of a llama.cpp load is a profile id.
3. **Phase 04, only for Tasks 6, 8 and 12:** per-engine locks are removed (`git grep -n "setEngineLockedCommand" -- services` prints nothing) and Halt (the local-only immediate stop, README D5) is distinguishable from the configured, peer-reversible Pause both in the broker (`services/nvpair-ui-broker/availability.go`; at baseline Halt is `nodepolicy.PauseImmediate` with `PausedLocally`) and in the manager's node projection (`services/nvpair-engine-manager/nodeprojection.go`, at baseline only `PausedLocally` and `ManagementBlocked`). Phase 04's close-out names the field. Tasks 1-5, 7 and 9-11 do not wait for phase 04 except through their dependencies.
4. **Todo state:** the tasks this phase executes are still unlanded on your base. Check with `git log --oneline --first-parent origin/integration/join33 | grep -i -E "diagnostic|stated|telemetry 5b|apply anyway"` and the Todos' status lines. A task that landed meanwhile is skipped and recorded in the close-out.
5. **Desktop paths:** paths under `desktop/src/ui/components/ModelSettings/` are as at baseline `4eccd05d`. Phases 02 and 04 record any rename in their close-outs; apply this plan's edits to the renamed file and record the mapping in this phase's close-out.

## Context the executor needs

Repository: the PAIR fork, base `origin/integration/join33` (planned against `4eccd05d`, 2026-10-31). Re-read each named symbol before editing; line numbers are navigation aids only. Evidence ids refer to `investigation/L6.md` (L6-Cn), `investigation/verify-B.md` (B3, B4, B8) and `investigation/verify-A.md` (A7).

Specifications this phase executes (vault `/home/user/wasabi/4. Projects/Personal-AI-Router/Todo/`; read the cited task text in full before each task; do not copy it):
- `2026-10-29-pair-memory-estimation-direction.md` ("estimation"): Tasks 9, 10 (step 3 and step 4 only), 12, 13 not landed (L6-C2). Tasks 1-8 and 11 landed (L6-C1). Task 14 stays out of this phase.
- `2026-10-28-pair-memory-telemetry.md` ("telemetry"): Tasks 5b, 6, 7 not landed (L6-C3). Its section "Amendment 2026-10-29: memory estimation direction" overrides the task text it names.
- `2026-10-29-pair-memory-diagnostic-run.md` ("diagnostic"): Task 5 landed as `Broker.drainNode` in `services/nvpair-ui-broker/engineswitch_drain.go`, called only from tests (L6-C5, B4). Tasks 1, 2, 3, 13, 4, 12, 6, 7, 8, 9, 10 are executed here in that order (L6 "What a fix would have to touch"). Task 11 (real-machine acceptance) is replaced by this phase's machine acceptance.

Verified current state (baseline):
- Estimator: `modelEstimateState.estimate` → `calculateModelEstimate` in `services/nvpair-engine-manager/modelsettings_estimate.go` prices weights, attention caches, recurrent state, compute workspace and prompt cache as ranges (L6-C8). `calculateModelEstimate` always sets `Status: "partial"`, always adds an unknown `runtime-overhead` row and always appends `Unknowns` "mmap residency unsized" and "transient peaks unsized" (L6-C9, B3). `modelEstimateState.estimate` sets `EstimateSource{Kind: "pair-estimate"}`.
- `modelconfig.CompareEstimate` (`services/shared/modelconfig/memory.go`) is already monotonic: it sums priced and partial rows, names omissions as `not priced: <name>` / `not priced (any pool): <name>` domain warnings, adds `EstimateSource.UnpricedCosts` as warnings, and uses reserve `MemoryReserve(capacity)` = 10% + 512 MiB (L6-C11). It does not read `Status` or `Unknowns`.
- The blockers are the consumers: `modelAutoFit` (`modelsettings_auto.go`, about line 137) returns unknown unless `out.Status == "estimated" && len(out.Unknowns) == 0`, so Auto never selects and the frozen-Auto Apply check in `modelsettings_rpc.go` (about line 252, "Reviewed allocation no longer fits fresh complete target evidence; explicitly Recalculate.") refuses (L6-C10, B3). `desktop/src/ui/components/ModelSettings/ModelMemoryEstimate.tsx` `comparison()` and `complete` recompute an all-or-nothing rule, so the owner always sees "Unknown" (L6-C12, B3).
- Configuration key: `modelConfigurationKey` (`model_configuration_key.go`) hashes artifact bindings, native source, options digest, placement and `modelConfigurationPools`. It returns "" when GPU mode is neither `cpu` nor `layers` with a count (`native-default` gets no key) and when the telemetry-derived pool list is empty (B3). Inspection sets it in `modelsettings_artifacts.go` (about lines 651-679), reasons `not-applied`, `not-finite`, `artifacts-unavailable` (L6-C15). It holds no CPU model, RAM or GPU name (L6-C14).
- Stated figures: `model-memory-profiles.json` store in `model_memory_profiles.go`; `MemoryProfile.Validate` accepts only `Kind == "stated"` (`services/shared/modelconfig/memory_profile.go`); writer at `modelsettings_rpc.go` about line 294. Not overlaid on the estimate yet (estimation Task 9).
- Telemetry: node-info publishes timestamped pools; the broker's `telemetryCache` (`telemetry.go`, `lookupLocalMemory`) ages them; `Broker.modelTargetMemory` (`modelsettings_reviews.go`) builds the envelope's `Memory`. node-info's directory enrichment carries `CPUInfo.Name`, `MemoryInfo.TotalBytes` and `GPUInfo{Name, VramBytes}` (`services/shared/noderec/noderec.go`); the broker holds directory entries in `Broker.store` (`*discoveryStore`, `discovery.go`). The proxy and admission never read memory telemetry (L6-C19, C20). Nothing watches memory during a load; the Windows working-set feature is a hard per-process cap (L6-C21).
- Native output: `startupOutput` (`startup_output.go`) keeps a redacted 8 KiB tail of an engine's startup and discards it on success; `lifecycle.go` writes serving lines to engine logs after `redactServingLine` and passes raw frames to `modelBufferCapture.observe` (`model_buffer_capture.go`), which attributes lines to a child between `modelBufferLoading` and `modelBufferReady`. Inspection attaches the buffer observation through `attachModelBufferObservation`, rendered by `desktop/src/ui/components/ModelSettings/ModelLoadBufferReport.tsx` inside `ModelSettingsPreview.tsx`. Vault `deferred.md` entry "2026-10-31 — Show native startup output for custom llama.cpp launches" asks for the native load summary.
- Speed: proxy samples are `performancewire.Sample{Key: ProfileKey{Host, Engine, Model, Fingerprint}}` (`services/shared/performancewire/wire.go`); the broker records them through `Broker.ingestPerformanceSample` (`performance.go`) into `performancestore`. The fingerprint includes the engine run id, so restarts fragment history, and plain llama.cpp records nothing for lack of a digest (L6-C22, C23, B8).
- Pause: the immediate local pause (`nodepolicy.PauseImmediate`, local only, `availability.go` about line 76) is the owner's Halt (README D5). At baseline both pause kinds set `ManagementBlocked`, which `lifecycle.go` checks before `startManagedProcObserved` (`localVetoError{reason: nodePausedVeto}`).
- Switch relay pattern for peer requests: `switchrelay.go` (`engine:switch-request`), `switchremote.go`, broker `engineswitch.go`; internal method denial list `internalClientMethods` in `services/nvpair-ui-broker/client_requests.go`.

Repository rules: `.cursor/rules/services-change-propagation.mdc`, `service-contracts.mdc`, `renderer-stores.mdc`, `type-discipline.mdc`, `no-legacy-fallbacks.mdc`, `engines-vs-backends-naming.mdc`. A JSON-RPC change touches the Go producer, broker relay, desktop channel maps and bridge, tests and docs in one change; `desktop/docs/services-api.md` is regenerated with `npm run service-contracts:write` from `desktop/`, never hand-edited. Commits: explicit paths, `git commit --no-signoff` (owner instruction in vault `Agent Instructions/README.md`, overriding `.cursor/rules/commit-sign-off.mdc`), no version or release-intent edits. Use the `pair-test-style` and `kaizen-ui` skills where the cited Todo tasks name them, and read the vault `Framework/*.md` notes each Todo task names.

Machine routing (from the cited Todos, with the 2026-10-31 updates): the Mac runs focused Go selectors and desktop checks, never the full engine-manager suite or `services/tests`, and never serves models. PC3 runs Windows selectors, full suites and `services/tests` under vault `TESTING_ON_PC.md` section 10. Joes is back in PAIR use (decided.md 2026-10-31 "Joes serves MSI-like tiers"), superseding the 2026-10-29 retirement the Todos cite. MSI runs real models for PAIR only in an interval the owner authorizes, with no Strata work running there (decided.md 2026-10-28).

Not verified here: every native outcome (no GPU, engines or owner machines were used), the rendered UI, and whether the running builds (b11219, b11985) expose the chat input-token counting endpoint that diagnostic Task 3 uses.

## Decisions (settled — do not revisit)

- Follow each cited Todo's "Decisions (settled - do not revisit)" section and its owner-accepted items: estimation QA1-QA7, diagnostic QB1-QB3, telemetry ownership and frozen-Auto rules (decided.md 2026-10-29 "Memory estimation, diagnostic run and Apps UI: owner accepts all 13 recommendations"). This phase does not reopen them.
- The estimate never blocks Apply, role assignment or routing; uncertainty becomes warnings, and only `exceeds` asks one "Run anyway" confirmation (owner 2026-10-29, QA6; decided.md O2/O7).
- Measurement is not qualification: results are labelled "Measured on this machine" with date, build and configuration; never "guaranteed", "qualified" or "safe" (decided.md 2026-10-29 O7).
- An unmeasured profile stays assignable and routable (owner expectation 11: "Whether the full test run has been ran or not, it should allow setting a profile as a role and just immediately working").
- Validate may be started from another node unless the target is halted; a configured, peer-reversible Pause ("simply off") does not prevent it (owner expectation 11; README D5).
- The measurement result is bound to the exact model file, engine build and node hardware (GPU, RAM, CPU) (owner expectation 11).
- Any llama.cpp build that answers `/props` is accepted and its `build_info` is recorded with every measurement (README D6).
- Estimates, stated figures, measured records and the load guard are advisory: never admission authority, ME2 evidence, policy writes or launch binding (estimation and telemetry Todos' settled decisions).
- Reuse, do not duplicate: `drainNode`, `model-memory-profiles.json`, `modelConfigurationKey`, `CompareEstimate`, the switch relay for peer-initiated runs (QB1), `modelsettings_reviews.go` for preview ids, the broker telemetry cache for every memory read.

## Decisions made while writing this plan

Edit or delete any of these to veto.

- **Packaging.** Twelve tasks group the Todo tasks by outcome: Task 1 = estimation 9 + 10 step 3; Task 2 = telemetry 5b (amended); Task 3 = new key and provenance work + estimation 10 step 4; Task 4 = telemetry 6 + estimation 12; Task 5 = diagnostic 1 + 2; Task 6 = diagnostic 3 + 13; Task 7 = diagnostic 4 + 12 + 8 + telemetry 5b's measured-record step; Task 8 = diagnostic 6 + 7 + speed samples; Task 9 = diagnostic 9; Tasks 10 and 11 are new; Task 12 = telemetry 7 + estimation 13 + diagnostic 10 + the guard's process case.
- **Reservation order.** Estimation Task 9 proceeds without waiting for engine-memory-plan (EMP) Task 5, which is held behind Strata work; EMP Task 5 rebases on this phase. Estimation Task 13 and telemetry Task 7 move into Task 12, so diagnostic Task 4's entry condition "estimation Task 13 landed" is replaced by Tasks 1, 2 and 4 here (the shared files are the estimator, `modelsettings_rpc.go`, the desktop decoder and the generated API). These are file-serialization orders from the estimation Todo's "Shared-file reservation order", not owner decisions.
- **Partial estimates.** `calculateModelEstimate` keeps `Status: "partial"` and stops appending the two fixed `Unknowns`; `modelEstimateState.estimate` lists them as `EstimateSource.UnpricedCosts: ["mmap residency", "transient peaks"]`, so `CompareEstimate` reports them as `not priced (any pool): ...` warnings on every domain. No consumer gates on `Status` or `Unknowns`; consumers use the per-domain verdict. This applies the estimation plan's monotonic-verdict decision to the three sites B3 names.
- **Key composition.** The configuration key no longer hashes the telemetry-derived pool list and no longer refuses without telemetry or without an explicit GPU layer count. Placement is `cpu` for CPU placement, the `DeviceBinding.DeviceID` when a binding exists, otherwise `unbound-gpu` for every other GPU mode including `native-default` (the options digest already carries the mode and count). `ConfigurationPools` stays telemetry-derived and separate.
- **Provenance is recorded, then checked, not hashed.** `build_info`, CPU name, total RAM and GPU names with VRAM are stored with every stated and measured record and returned with the key. A record is current only when its key matches and its provenance matches the current one (build info equal; CPU name equal; RAM and each GPU's VRAM equal after rounding down to whole GiB; same GPU names in order). When the current hardware identity is unknown (node-info entry missing), the record stays current with the warning `hardware identity unavailable; record not rechecked`. Hashing these facts would make the key disappear whenever node-info is down.
- **Hardware identity source.** The broker attaches its own node's node-info enrichment (`CPUInfo.Name`, `MemoryInfo.TotalBytes`, `GPUInfo.Name`, `GPUInfo.VramBytes`) to the model-settings worker envelope beside `Memory`, under the same rule that public and origin callers cannot supply it.
- **Consent thresholds** (owner: "more than low vram usage, and more than half ram usage"). Warn for a dedicated GPU pool when its in-use bytes (capacity minus available) exceed max(1 GiB, 10% of capacity); warn for the system RAM pool, and for a shared (UMA) pool, when in-use bytes exceed half its capacity. Texts: `<GPU name>: <in use> of <capacity> VRAM in use. More than an idle GPU uses; results will include it.` and `RAM: <in use> of <capacity> in use (more than half).` The diagnostic plan's other-GPU-users warning (QB3) stays.
- **Pause during a run.** A configured Pause accepted during a run does not abort it (the node is already drained and reserved); restore honours it by leaving engines stopped, as the diagnostic plan's restore rule already says. Halt aborts the run exactly as the diagnostic plan describes for Pause. The diagnostic token's intent-neutral start passes the configured-Pause spawn veto in `lifecycle.go`, never the Halt veto.
- **`BuildPin` becomes `Provenance`.** `MeasuredProfile.BuildPin` (diagnostic Task 1) is replaced by the Task 3 provenance, because the `b11146` pin no longer exists after phase 01.
- **Token counting.** When the running build answers the chat input-token counting endpoint with 404, the run fails phase `low` with reason `token-counting-unavailable`; there is no plain-tokenization fallback (keeps the diagnostic plan's "If stuck" rule without assuming the pinned build).
- **Speed samples.** One `performancewire.Sample` per measured request with `Key{Host: node id, Engine: <the engine id proxy samples use for this engine>, Model: <phase 01 contentId>, Fingerprint: <profile id>}`, `Profile{EngineFamily: "llamacpp", Quantization: <phase 01 quant label>, ContextTokens: <perSlotContext>}`, producer `pair-diagnostic-<broker incarnation>`. Without a contentId the run records the note `speed not recorded: model identity unavailable` and writes no sample. Phase 06 reads these keys.
- **Startup output retention.** Per profile, the newest load's redacted native output from the loading marker to ready, failure or exit, at most 8 KiB, kept in manager memory only (no file), replaced by the next load of that profile.
- **Load guard.** Owned by the broker (it holds the telemetry cache; the manager never reads OS memory). Floor `loadGuard.minAvailableRamMiB` in node policy, default 256 MiB when omitted, `0` disables, range 0 to 1048576. The default sits below the 0.47 GiB dip the owner accepted on Joes (decided.md 2026-10-31 "Joes Medium restored"). Edited through the node's policy JSON (phase 04's advanced JSON); no new control in this phase. Effective sampling is the producer's refresh (1-2 s) read at 1 Hz. VRAM is sampled and recorded, never acted on. There is no hold after a stop: a later load is guarded again on its own.
- **Out of phase.** Estimation Task 14 (real calibration) and EMP Tasks 4-6 stay open; diagnostic Task 11 is replaced by this phase's machine acceptance.

## Tasks

### Task 1 — Stated figures, native reports and CPU expert placement feed the estimate
**Files:** `services/nvpair-engine-manager/modelsettings_estimate.go`, `services/nvpair-engine-manager/modelsettings_estimate_test.go`, `services/nvpair-engine-manager/modelsettings_rpc.go`, `services/nvpair-engine-manager/model_buffer_store.go`
**Depends on:** none — entry conditions 1 and 2 must hold
**Complexity:** judgment
**Do:**
1. Execute estimation Task 9 (steps 1-3) as written, with one amendment: its reservation "after EMP Task 5" does not apply (plan-time decision "Reservation order").
2. Execute estimation Task 10 step 3 only: in `modelEstimatePlacement`, the expert tensors of the layers that the typed CPU expert option (phase 01) selects are placed on the host domain, using the expert-tensor name pattern of llama.cpp `common/arg.cpp` at the b11146 pin `7fe450e19305b828c199d602c23a8337aaa1f03b` (cite the line in a comment), and Task 6's host compute portion follows them. Steps 1, 2 and 5 landed in phase 01; step 4 is in Task 3 here.
**Verify:** from `services/nvpair-engine-manager`: `go test -count=1 -run '^(TestModelSettingsEstimate|TestModelBuffer|TestModelMemoryProfiles)' .`, the same with `-race`, and `go vet ./...` all exit 0. Pass: estimation Task 9's Verify conditions hold; a MoE fixture with CPU experts `layers: 12` has exactly those layers' expert bytes on the host domain and off the device domain, `all` moves every expert tensor, `none` leaves placement as before.

### Task 2 — A partial estimate gets a verdict and Auto selects on it
**Files:** `services/nvpair-engine-manager/modelsettings_estimate.go`, `services/nvpair-engine-manager/modelsettings_estimate_test.go`, `services/nvpair-engine-manager/modelsettings_auto.go`, `services/nvpair-engine-manager/modelsettings_auto_test.go`, `services/nvpair-engine-manager/modelsettings_rpc.go`, `services/nvpair-engine-manager/MANIFEST.md`, `services/shared/modelconfig/memory_test.go`, `services/shared/enginesettings/settings.go`, `services/shared/enginesettings/settings_test.go`, `services/nvpair-ui-broker/modelsettings.go`, `services/nvpair-ui-broker/modelsettings_estimate_test.go`, `desktop/src/shared/types/engine-settings.ts`, `desktop/src/electron/service-bridge/engine-settings.ts`, `desktop/tests/modular/engine-settings-bridge.test.ts`, `desktop/docs/services-api.md` (generated)
**Depends on:** 1 — same estimator and `modelsettings_rpc.go`
**Complexity:** judgment
**Do:**
1. Execute telemetry Task 5b as replaced by the 5b bullets of that Todo's "Amendment 2026-10-29: memory estimation direction": Auto candidates and the frozen-Auto Apply check call `modelconfig.CompareEstimate`, accept `fits` and `tight` with warnings, never select `exceeds` or `unknown`; delete `modelAutoFit`'s separate arithmetic; add `applyAnyway` (QA7). The amendment supersedes 5b's original complete-cost dispatch gate (decided.md 2026-10-29 O7). Its "measured records" step is Task 7 here.
2. Amendment for this phase (plan-time decision "Partial estimates"): in `calculateModelEstimate` remove the line appending `"mmap residency unsized", "transient peaks unsized"` to `out.Unknowns`; in `modelEstimateState.estimate` set `UnpricedCosts: []string{"mmap residency", "transient peaks"}` on the `pair-estimate` `EstimateSource`. Keep `Status: "partial"`.
3. In `modelsettings_auto.go`, delete the `out.Status != "estimated" || len(out.Unknowns) != 0` condition with `modelAutoFit`'s arithmetic. Derive the bisection direction from which domain's `FreeFit` is `exceeds` (host exceeds: more offload; device exceeds: less), keeping the dense-transfer proof gate the amendment names.
4. In `services/shared/modelconfig/memory_test.go` add `TestCompareEstimatePartialNamesUnpricedCosts`: a `partial` estimate with priced weights and KV, an unknown `runtime-overhead` row and the two unpriced costs yields a non-`unknown` verdict per domain with warnings `not priced (any pool): runtime-overhead`, `not priced (any pool): mmap residency` and `not priced (any pool): transient peaks`.
5. Regenerate `desktop/docs/services-api.md` with `npm run service-contracts:write` from `desktop/`.
**Verify:** run telemetry Task 5b's Verify commands with the amendment's additions (manager, broker and shared selectors plain and `-race`, vet, desktop bridge test, typecheck, lint, dead-code, `npm run service-contracts:check`); all exit 0. Pass: the amendment's Verify conditions hold; `go test -count=1 -run '^TestCompareEstimatePartialNamesUnpricedCosts$' ./modelconfig` from `services/shared` passes; `git grep -n 'out.Status != "estimated"' -- services/nvpair-engine-manager/modelsettings_auto.go` prints nothing; `git grep -n 'mmap residency unsized' -- services/nvpair-engine-manager/modelsettings_estimate.go` prints nothing.

### Task 3 — Every profile has a configuration key with build and hardware provenance
**Files:** `services/nvpair-engine-manager/model_configuration_key.go`, `services/nvpair-engine-manager/model_configuration_key_test.go`, `services/nvpair-engine-manager/modelsettings_artifacts.go`, `services/nvpair-engine-manager/modelsettings_artifacts_test.go`, `services/nvpair-engine-manager/modelsettings_rpc.go`, `services/nvpair-engine-manager/model_memory_profiles.go`, `services/nvpair-engine-manager/model_memory_profiles_test.go`, `services/shared/modelconfig/wire.go`, `services/shared/modelconfig/wire_test.go`, `services/shared/modelconfig/memory_profile.go`, `services/shared/modelconfig/memory_profile_test.go`, `services/shared/enginesettings/settings.go`, `services/shared/enginesettings/settings_test.go`, `services/nvpair-ui-broker/modelsettings.go`, `services/nvpair-ui-broker/modelsettings_estimate_test.go`, `desktop/src/shared/types/model-config.ts`, `desktop/src/electron/service-bridge/engine-settings.ts`, `desktop/tests/modular/engine-settings-bridge.test.ts`, `desktop/tests/modular/model-config-contract.test.ts`, `desktop/docs/services-api.md` (generated)
**Depends on:** 2 — same `modelsettings_rpc.go`, wire, bridge and generated API
**Complexity:** judgment
**Do:**
1. Key (plan-time decision "Key composition"): in `modelConfigurationKey` remove `Pools` from the hashed struct and the `len(pools) == 0` refusal; set placement to `cpu` for CPU placement (mode `cpu`, or `layers` with count 0), to `DeviceBinding.DeviceID` when a binding exists, otherwise `unbound-gpu`. Keep the refusals for a missing allocation, source, options digest, context, slots or bindings. `modelConfigurationPools` and `InspectResult.ConfigurationPools` are unchanged.
2. Estimation Task 10 step 4: a test in `model_configuration_key_test.go` that changing only the CPU expert option changes the key.
3. Types in `services/shared/modelconfig/wire.go`, with `Validate` (text at most 256 bytes, canonical decimal byte strings, at most 16 GPUs) and deep `Clone`:
   ```go
   type HardwareGPU struct {
       Name           string `json:"name"`
       VRAMTotalBytes string `json:"vramTotalBytes,omitempty"`
   }
   type HardwareIdentity struct {
       CPUName       string        `json:"cpuName,omitempty"`
       RAMTotalBytes string        `json:"ramTotalBytes,omitempty"`
       GPUs          []HardwareGPU `json:"gpus,omitempty"`
   }
   type ConfigurationProvenance struct {
       BuildInfo string            `json:"buildInfo,omitempty"`
       DeviceID  string            `json:"deviceId,omitempty"`
       Hardware  *HardwareIdentity `json:"hardware,omitempty"`
   }
   ```
   `InspectResult` gains `ConfigurationProvenance *ConfigurationProvenance` (`json:"configurationProvenance,omitempty"`), set whenever `ConfigurationKey` is set. `MemoryProfile` (`memory_profile.go`) gains `Provenance *ConfigurationProvenance` (`json:"provenance"`), required by `Validate`.
4. `enginesettings.WorkerRequest` gains `Hardware *modelconfig.HardwareIdentity` (`json:"hardware,omitempty"`). In `services/nvpair-ui-broker/modelsettings.go`, wherever an envelope sets `Memory: b.modelTargetMemory()`, also set `Hardware` from the broker's own entry in `b.store` (host UUID equal to `b.nodeID`): `CPU.Name`, `Memory.TotalBytes`, each `GPUs[i].Name` and `VramBytes`; nil when the entry or its enrichment is absent. Public and origin requests cannot carry it (same check as `Memory`).
5. In `modelsettings_artifacts.go` build the provenance: `BuildInfo` from phase 01's engine build identity for the inspected engine, `DeviceID` from the binding, `Hardware` from the envelope. With phase 02, a profile without an applied load keeps reason `not-applied`; every applied profile gets a key.
6. Stated writes (`modelsettings_rpc.go`, the `Kind: "stated"` record) store the inspection's provenance.
7. Currentness in `model_memory_profiles.go` (plan-time decision "Provenance is recorded, then checked, not hashed"): a record whose provenance differs from the current one is returned with `Current: false` and the warning `recorded on different hardware or build`; with nil current hardware the record stays current with the warning `hardware identity unavailable; record not rechecked`. Estimation Task 9's overlay (Task 1) uses only current records.
8. Desktop type and strict decoder for the new fields; extend the bridge and contract tests; regenerate services-api.
**Verify:** from `services/nvpair-engine-manager`: `go test -count=1 -run '^(TestModelConfigurationKey|TestModelSettings|TestModelMemoryProfiles)' .` plain and `-race`, `go vet ./...`; from `services/shared`: `go test -count=1 ./modelconfig ./enginesettings`; from `services/nvpair-ui-broker`: `go test -count=1 -run '^TestModelSettings' .`; from `desktop/`: `npx vitest run --project unit tests/modular/engine-settings-bridge.test.ts tests/modular/model-config-contract.test.ts`, `npm run typecheck`, `npm run service-contracts:check`; all exit 0. Pass (new tests): a `native-default` typed load gets a non-empty key with placement `unbound-gpu`; the key is identical with and without a memory observation; the key changes with the CPU expert option and with a device binding; inspection returns provenance with the injected build info, CPU name, RAM and GPUs; a stated record whose CPU name differs is returned not current with the warning text; nil hardware keeps it current with its warning; a public request carrying `hardware` is refused.

### Task 4 — Model Settings shows the service estimate, Run anyway and the stated figure
**Files:** `desktop/src/ui/components/ModelSettings/ModelMemoryEstimate.tsx`, `desktop/src/ui/components/ModelSettings/ModelMemoryStatedFigure.tsx` (new), `desktop/src/ui/components/ModelSettings/ModelSettingsPreview.tsx`, `desktop/src/ui/components/ModelSettings/ModelSettingsPanel.tsx`, `desktop/src/ui/components/ModelSettings/ModelLoadSettings.tsx`, `desktop/src/ui/stores/engine-settings.store.ts`, `desktop/src/electron/service-bridge/engine-settings.ts`, `desktop/tests/modular/model-memory-estimate.test.tsx`, `desktop/tests/modular/model-memory-stated-figure.test.tsx` (new), `desktop/tests/modular/model-settings-auto.test.tsx`, `desktop/tests/modular/model-settings-apply.test.tsx`, `desktop/tests/modular/engine-settings-bridge.test.ts`, `docs/model-configuration.mdx`
**Depends on:** 3 — provenance fields and the shared bridge file; transitively 2 for `applyAnyway`
**Complexity:** judgment
**Do:** Execute telemetry Task 6 as replaced by the Task 6 bullets of its 2026-10-29 amendment, then estimation Task 12, as one change. Amendments:
1. Delete `comparison()` and `complete` from `ModelMemoryEstimate.tsx`. Each domain shows the service's `freeFit` and `capacityFit`, `requiredBytes` with `requiredLowBytes`/`requiredHighBytes` as `expected (up to high)`, and the domain warnings under "Not included in this estimate". "Unknown" appears only for a domain whose service verdict is `unknown`, with its reason. A `partial` estimate is never shown as "Unknown" on that account.
2. Estimation Task 12 step 2's conditional is resolved: `applyAnyway` exists (Task 2), so a frozen-Auto refusal offers "Apply anyway" and "Recalculate".
3. The stated figure shows its provenance: `Your figure (not verified by PAIR), <date>, <CPU name>, <RAM GiB> GiB RAM, <GPU names>, build <buildInfo>`; a non-current record shows `Recorded on different hardware or build` and its stored provenance.
4. No verdict disables Apply, Save or role assignment.
**Verify:** from `desktop/`: `npx vitest run --project unit tests/modular/model-memory-estimate.test.tsx tests/modular/model-memory-stated-figure.test.tsx tests/modular/model-settings-auto.test.tsx tests/modular/model-settings-apply.test.tsx tests/modular/engine-settings-bridge.test.ts`, `npm run lint`, `npm run typecheck`, `npm run dead-code:check` exit 0. Pass: telemetry Task 6's and estimation Task 12's Verify conditions; a fixture `partial` estimate with `freeFit: "tight"` renders "Tight: may not fit" and the three not-priced warnings; Apply stays enabled for `unknown` and `exceeds` (the latter opens "Run anyway"); `git grep -n "536870912n" -- desktop/src/ui/components/ModelSettings/ModelMemoryEstimate.tsx` prints nothing.

### Task 5 — The manager can reserve the node for a measurement
**Files:** `services/shared/modelconfig/diagnostic.go` (new), `services/shared/modelconfig/diagnostic_test.go` (new), `services/shared/modelconfig/memory_profile.go`, `services/shared/modelconfig/memory_profile_test.go`, `services/nvpair-engine-manager/switchexecutor.go`, `services/nvpair-engine-manager/switchexecutor_test.go`, `services/nvpair-engine-manager/wakesleep.go`, `services/nvpair-engine-manager/wakesleep_test.go`, `services/nvpair-engine-manager/actions.go`, `services/nvpair-engine-manager/serving_llamacpp.go`, `services/nvpair-engine-manager/executor.go`, `services/nvpair-engine-manager/model_diagnostic.go` (new), `services/nvpair-engine-manager/model_diagnostic_reservation_test.go` (new), `services/nvpair-ui-broker/client_requests.go`, `services/nvpair-ui-broker/diagnostic_internal_test.go` (new)
**Depends on:** 3 — provenance type and `memory_profile.go`
**Complexity:** judgment
**Do:** Execute diagnostic Tasks 1 and 2 as written, with these amendments:
1. Task 1: replace `MeasuredProfile.BuildPin` with `Provenance *ConfigurationProvenance` (Task 3 type), required for every measured record.
2. Task 1: `DiagnosticPreview` gains `Resources []DiagnosticResource` (at most 16) with `DiagnosticResource{DomainID, Kind ("system"|"dedicated"|"shared"), Label, CapacityBytes, InUseBytes (canonical decimals, optional), Warning string}`; filled in Task 8.
3. Task 1's entry condition (estimation Task 8 landed) holds at baseline (L6-C1). `ModelID` in every diagnostic type holds the profile id (entry condition 2).
**Verify:** diagnostic Task 1's and Task 2's Verify commands on the Mac (shared `go test -count=1 ./modelconfig` plain and `-race`; manager `go test -count=1 -run '^(TestDiagnosticReservation|TestSwitch|TestWake|TestSleep)' .` plain and `-race`; broker `go test -count=1 -run '^TestDiagnosticInternalMethods' .`; `go vet ./...` in each module) exit 0. Pass: their pass conditions, plus a measured record without provenance and a preview with 17 resources are rejected.

### Task 6 — The manager runs one cancellable measurement at low and full context
**Files:** `services/nvpair-engine-manager/model_diagnostic_exercise.go` (new), `services/nvpair-engine-manager/model_diagnostic_exercise_test.go` (new), `services/nvpair-engine-manager/model_diagnostic.go`, `services/nvpair-engine-manager/model_diagnostic_test.go` (new), `services/nvpair-engine-manager/executor.go`, `services/nvpair-engine-manager/switchexecutor.go`, `services/nvpair-engine-manager/switchexecutor_test.go`, `services/nvpair-engine-manager/manager.go`, `services/nvpair-engine-manager/wakesleep.go`, `services/nvpair-engine-manager/nodeprojection.go`, `services/nvpair-engine-manager/nodeprojection_test.go`, `services/nvpair-engine-manager/lifecycle.go`, `services/nvpair-engine-manager/README.md`
**Depends on:** 5 — reservation and types; entry condition 3 (phase 04 lock removal and Halt) must also hold before this task starts
**Complexity:** judgment
**Do:** Execute diagnostic Tasks 3 and 13 as written, with these amendments:
1. Task 13 step 3: per-engine locks no longer exist; drop every lock clause and drop `enginelock.go`/`enginelock_test.go` from its Files. Halt aborts the run exactly as step 3 describes for Pause. A configured Pause accepted during a run does not abort it; restore honours it under "Restore honours the newest intent" by leaving engines stopped. An explicit local Off keeps step 3's treatment.
2. In `lifecycle.go`, the spawn veto that checks `nodeProjection.projection.ManagementBlocked` before `startManagedProcObserved` lets a start carrying the diagnostic token through a configured Pause, never through Halt (phase 04's field).
3. Task 3: the token counting endpoint is the running build's (phase 01 build identity), not the b11146 pin; a 404 from it fails phase `low` with reason `token-counting-unavailable` (plan-time decision "Token counting").
**Verify:** from `services/nvpair-engine-manager`: `go test -count=1 -run '^(TestModelDiagnostic|TestModelDiagnosticExercise|TestDiagnosticReservation)' .` plain and `-race`, `go vet ./...` exit 0. Pass: diagnostic Task 3's and Task 13's pass conditions without the lock cases, plus: Halt accepted during load, token counting, generation and restore aborts the run and leaves no engine started; a configured Pause accepted during the high phase lets the run complete and restore leaves every engine stopped; a diagnostic-token start succeeds under a configured Pause and is refused under Halt; a fake server answering 404 on token counting ends the run `failed` in phase `low` with `token-counting-unavailable`.

### Task 7 — Measured records are stored, recovered and overlaid on the estimate
**Files:** `services/nvpair-engine-manager/model_memory_profiles.go`, `services/nvpair-engine-manager/model_memory_profiles_test.go`, `services/nvpair-engine-manager/model_diagnostic.go`, `services/nvpair-engine-manager/model_diagnostic_recovery_test.go` (new), `services/nvpair-engine-manager/modelsettings_rpc.go`, `services/nvpair-engine-manager/modelsettings_estimate.go`, `services/nvpair-engine-manager/modelsettings_estimate_test.go`, `services/nvpair-engine-manager/modelsettings_auto.go`, `services/nvpair-engine-manager/modelsettings_auto_test.go`, `services/nvpair-engine-manager/desiredstate.go`, `services/nvpair-engine-manager/desiredstate_test.go`, `services/nvpair-engine-manager/manager.go`, `services/nvpair-ui-broker/client_requests.go`, `services/nvpair-ui-broker/diagnostic_internal_test.go`, `desktop/src/shared/types/model-config.ts`, `desktop/src/electron/service-bridge/engine-settings.ts`, `desktop/tests/modular/engine-settings-bridge.test.ts`, `desktop/docs/services-api.md` (generated)
**Depends on:** 6, 10 — 6 shares `model_diagnostic.go` and `manager.go`; 10 shares the desktop decoder files and generated API (transitively 1, 2, 4)
**Complexity:** judgment
**Do:** Execute diagnostic Task 4, then Task 12, then Task 8, then the "Measured records" step of telemetry Task 5b's 2026-10-29 amendment, with these amendments:
1. Diagnostic Task 4's entry condition is met by Tasks 1, 2 and 4 here; estimation Task 13 runs in Task 12 (plan-time decision "Reservation order"). Diagnostic Task 8's entry condition is met by Task 1.
2. Measured records carry provenance (Task 5) and use Task 3's currentness rule: a record that is not current for provenance is not overlay-eligible and is shown with `measured on different hardware or build`.
3. The 5b measured step in `modelsettings_auto.go`: an Auto candidate whose key matches a current, overlay-eligible measured record uses its pool peaks through the domain slot (`EstimateSource.Kind = measured`); a record that does not cover every pool is ignored with a warning.
**Verify:** from `services/nvpair-engine-manager`: `go test -count=1 -run '^(TestModelMemoryProfiles|TestModelSettings|TestModelDiagnostic|TestModelDiagnosticRecovery|TestDesiredState|TestModelSettingsEstimate)' .` plain and `-race`, `go vet ./...`; broker `go test -count=1 -run '^TestDiagnosticInternalMethods' .`; desktop bridge test, `npm run typecheck`, `npm run lint`, `npm run dead-code:check`, `npm run service-contracts:check`; all exit 0. Pass: diagnostic Tasks 4, 12 and 8 pass conditions and the 5b amendment's measured-step conditions; a completed record with a different CPU name leaves the estimate unchanged and carries the warning text.

### Task 8 — The broker runs a consented measurement locally or for a peer and records its speed
**Files:** `services/nvpair-ui-broker/modeldiagnostic.go` (new), `services/nvpair-ui-broker/modeldiagnostic_test.go` (new), `services/nvpair-ui-broker/broker.go`, `services/nvpair-ui-broker/policy.go`, `services/nvpair-ui-broker/client_requests.go`, `services/nvpair-ui-broker/performance.go`, `services/nvpair-ui-broker/performance_test.go`, `services/nvpair-ui-broker/engineswitch.go`, `services/nvpair-ui-broker/engineswitch_test.go`, `services/nvpair-ui-broker/README.md`, `services/nvpair-ui-broker/ENGINE_SETTINGS.md`, `services/nvpair-engine-manager/switchrelay.go`, `services/nvpair-engine-manager/switchrelay_test.go`, `services/nvpair-engine-manager/switchremote.go`, `services/nvpair-engine-manager/switchremote_test.go`, `desktop/src/shared/types/model-config.ts`, `desktop/src/shared/types/ws-channels.ts`, `desktop/src/electron/service-bridge/engine-settings.ts`, `desktop/src/electron/service-bridge/empty-handlers.ts`, `desktop/src/electron/service-bridge/modular-supervisor.ts`, `desktop/src/ui/api/engine-api.ts`, `desktop/src/ui/components/TrayApp/TrayNodeRow.tsx`, `desktop/tests/modular/tray-node-row.test.tsx` (new), `desktop/tests/modular/engine-settings-bridge.test.ts`, `desktop/docs/frontend-api.md`, `desktop/docs/services-backend.md`, `docs/model-memory-diagnostic.mdx` (new), `desktop/docs/services-api.md` (generated)
**Depends on:** 7 — record method, checkpoint, shared broker and desktop files; entry condition 3 must also hold
**Complexity:** judgment
**Do:** Execute diagnostic Task 6, then Task 7, with these amendments:
1. Task 6 step 2 preconditions: replace "node available and not paused; engine llama.cpp family and not locked" with "node not halted; engine llama.cpp family". A configured Pause does not refuse. Every other precondition stays.
2. Task 6 preview: fill `Resources` from the same fresh local observation `modelTargetMemory` uses (`b.lookupLocalMemory`), labelled with node-info GPU names and "RAM", and add the threshold warnings with the exact texts in plan-time decision "Consent thresholds", beside the other-GPU-users warning (QB3).
3. Task 6 result: the stored record's provenance comes from the manager's inspection provenance for the run's key (Task 3), so it carries `build_info`, GPU, CPU and RAM.
4. Task 7: the remote gates are management permission and the Halt veto only; no Pause or lock veto. Keep the switch relay mirror and the shared fencing helper as written (QB1).
5. Speed samples: extract `func (b *Broker) recordPerformanceSample(sample performancewire.Sample) error` in `performance.go` (Record plus `performanceSignal(b.performance.changeWake)`) and make `ingestPerformanceSample` call it. When a run ends `completed`, or `failed` after the low phase returned requests, `modeldiagnostic.go` records one sample per `MeasuredRequest` per plan-time decision "Speed samples", with `Measurement{Outcome: completed, PromptTokens, CachedPromptTokens, OutputTokens, PrefillNs = PromptMS*1e6, DecodeNs = DecodeMS*1e6, EngineDecodeTokensPerSecond = DecodePerSecond, FirstTokenMs, TotalMs = EndMS}` and nil for each unknown value.
6. Document the thresholds, the Halt-only veto and the speed samples in `docs/model-memory-diagnostic.mdx`; regenerate services-api.
**Verify:** from `services/nvpair-ui-broker`: `go test -count=1 -run '^(TestModelDiagnostic|TestPerformance|TestSwitch)' .` plain and `-race`; from `services/nvpair-engine-manager`: `go test -count=1 -run '^(TestSwitchRelay|TestSwitchRemote|TestRemoteDiagnostic)' .` plain and `-race`; `go vet ./...` in both; from `desktop/`: `npx vitest run --project unit tests/modular/tray-node-row.test.tsx tests/modular/engine-settings-bridge.test.ts`, `npm run typecheck`, `npm run lint`, `npm run dead-code:check`, `npm run service-contracts:check`; all exit 0. Pass: diagnostic Tasks 6 and 7 pass conditions without the lock and Pause refusals, plus: a configured-paused node previews and starts; a halted node refuses with zero effects, locally and from a peer; synthetic telemetry with 2 GiB of 16 GiB VRAM in use and 20 of 32 GiB RAM in use yields both threshold warnings, 0.5 GiB and 10 GiB yield none; a completed fake run adds one performance sample per request under `Model = contentId` and `Fingerprint = profile id`; a run without contentId writes none and carries the note.

### Task 9 — Model Settings offers Measure with a resource rundown, progress and results
**Files:** `desktop/src/ui/components/ModelSettings/ModelMemoryDiagnostic.tsx` (new), `desktop/src/ui/components/ModelSettings/ModelSettingsPanel.tsx`, `desktop/src/ui/stores/engine-settings.store.ts`, `desktop/tests/modular/model-memory-diagnostic.test.tsx` (new)
**Depends on:** 8 — final method set; transitively 4 for the shared panel and store
**Complexity:** judgment
**Do:** Execute diagnostic Task 9 as written, with these amendments:
1. The button is offered for every llama.cpp profile with a configuration key, whatever the estimate says, and is disabled only with the preview's refusal text (for example a halted target).
2. The consent modal lists the preview's `Resources` (label, in use of capacity) with their warnings shown before the effects list, and keeps "Measure" and "Cancel".
3. Results show `Measured on this machine: <date>, build <buildInfo>, <GPU names>, <CPU name>, <RAM GiB> GiB RAM`, and beside each pool's "Measured peak" the estimate's `expected (up to high)` for the same pool.
4. A profile without a record shows "Not measured" and nothing else changes: assignment and Apply stay available.
**Verify:** from `desktop/`: `npx vitest run --project unit tests/modular/model-memory-diagnostic.test.tsx`, `npm run typecheck`, `npm run lint`, `npm run dead-code:check` exit 0. Pass: diagnostic Task 9's pass conditions, plus: resources and threshold warnings render in the modal; results render provenance and estimate ranges beside measured peaks; a halted-target refusal disables the button with its text; "Not measured" renders without disabling Apply.

### Task 10 — Each profile load keeps its native startup output for inspection
**Files:** `services/shared/modelconfig/load_report.go` (new), `services/shared/modelconfig/load_report_test.go` (new), `services/shared/modelconfig/wire.go`, `services/shared/modelconfig/wire_test.go`, `services/nvpair-engine-manager/model_buffer_capture.go`, `services/nvpair-engine-manager/model_buffer_capture_test.go`, `services/nvpair-engine-manager/startup_output.go`, `services/nvpair-engine-manager/startup_output_test.go`, `services/nvpair-engine-manager/modelsettings_artifacts.go`, `desktop/src/shared/types/model-config.ts`, `desktop/src/electron/service-bridge/engine-settings.ts`, `desktop/tests/modular/engine-settings-bridge.test.ts`, `desktop/tests/modular/model-buffer-contract.test.ts`, `desktop/src/ui/components/ModelSettings/ModelLoadBufferReport.tsx`, `desktop/tests/modular/model-buffer-report.test.tsx`, `desktop/docs/services-api.md` (generated)
**Depends on:** 4 — shared desktop decoder files; transitively 3 for `wire.go` and `modelsettings_artifacts.go`
**Complexity:** judgment
**Do:**
1. `load_report.go`: `LoadReport{LoadID string, StartedAt string, EndedAt string (RFC 3339, optional), Outcome ("loading"|"ready"|"failed"), Output string, Truncated bool}` with `Validate` (Output at most 8192 bytes, valid UTF-8, no control characters except newline and tab) and `Clone`. `InspectResult` gains `LoadReport *LoadReport` (`json:"loadReport,omitempty"`).
2. `startup_output.go`: add `func (o *startupOutput) text() (string, bool)` returning the flushed, redacted tail and whether bytes were dropped, without closing; existing failure behaviour unchanged.
3. `model_buffer_capture.go`: each child gets its own `startupOutput` from `newStartupOutput` built with the args and environment of the `command` passed to `newModelBufferCapture` (no `lifecycle.go` change). From the `modelBufferLoading` match until `modelBufferReady`, child exit or poison, append each line `observe` routes to that child after `redactServingLine(line, st.modelsDir)`. At the end, store a `LoadReport` with an opaque `LoadID` (capture token, child port and generation, hashed) in `modelBufferOwner`, one per (engine, model), replacing that model's previous report; keep it across child exit and capture retire, in memory only. Lines outside the loading window are never kept, so no request content is retained.
4. Attach the model's newest report to inspection next to `attachModelBufferObservation` (`modelsettings_artifacts.go`).
5. Desktop type, strict decoder and contract test; `ModelLoadBufferReport.tsx` renders a collapsed "Native load output" block (monospace, started and ended times, outcome) with the caption `As printed by the engine during load; PAIR removed paths and private values.` Regenerate services-api.
**Verify:** from `services/shared`: `go test -count=1 ./modelconfig`; from `services/nvpair-engine-manager`: `go test -count=1 -run '^(TestModelBuffer|TestStartupOutput|TestModelSettings)' .` plain and `-race`, `go vet ./...`; from `desktop/`: `npx vitest run --project unit tests/modular/model-buffer-report.test.tsx tests/modular/model-buffer-contract.test.ts tests/modular/engine-settings-bridge.test.ts`, `npm run typecheck`, `npm run lint`, `npm run dead-code:check`, `npm run service-contracts:check`; all exit 0. Pass (new tests): a fake child's lines between the loading and ready markers appear in inspection's `loadReport.output` with the models directory and an argument value replaced by `[redacted]`; a line printed after ready is absent; a second load replaces the first report; 20 KiB of load output yields at most 8192 bytes with `truncated: true`; the block renders and is collapsed by default.

### Task 11 — A load-time pressure guard stops a load when RAM runs out
**Files:** `services/shared/nodepolicy/policy.go`, `services/shared/nodepolicy/policy_test.go`, `services/shared/modelconfig/load_report.go`, `services/shared/modelconfig/load_report_test.go`, `services/nvpair-engine-manager/model_buffer_capture.go`, `services/nvpair-engine-manager/model_buffer_capture_test.go`, `services/nvpair-engine-manager/model_load_guard.go` (new), `services/nvpair-engine-manager/model_load_guard_test.go` (new), `services/nvpair-engine-manager/model_diagnostic.go`, `services/nvpair-engine-manager/manager.go`, `services/nvpair-engine-manager/README.md`, `services/nvpair-ui-broker/loadguard.go` (new), `services/nvpair-ui-broker/loadguard_test.go` (new), `services/nvpair-ui-broker/broker.go`, `services/nvpair-ui-broker/client_requests.go`, `services/nvpair-ui-broker/README.md`, `desktop/src/shared/types/node-policy.ts`, `desktop/src/electron/service-bridge/node-policy.ts`, `desktop/tests/modular/node-policy.test.ts`, `desktop/src/shared/types/model-config.ts`, `desktop/src/electron/service-bridge/engine-settings.ts`, `desktop/tests/modular/engine-settings-bridge.test.ts`, `desktop/src/ui/components/ModelSettings/ModelLoadBufferReport.tsx`, `desktop/tests/modular/model-buffer-report.test.tsx`, `desktop/docs/services-api.md` (generated)
**Depends on:** 9, 10 — 9 ends the measurement chain (shares `model_diagnostic.go`, `manager.go`, broker and desktop files); 10 owns the load report and capture
**Complexity:** judgment
**Do:** This is advisory safety, never admission: it never refuses a load start, never changes routing, residency policy or saved intent, and never reads OS memory in the broker or manager.
1. Node policy: `Policy` gains `LoadGuard LoadGuard` (`json:"loadGuard"`) with `MinAvailableRAMMiB int` (`json:"minAvailableRamMiB"`), default 256 when omitted through the existing omitted-defaults decoding in `policy.go`, valid 0 to 1048576, 0 disables. Not advertised to peers. Desktop type and decoder.
2. Manager: emit notification `engine:model-load-changed {engine, modelId, loadId, phase: "loading"|"ended"}` from the capture child's loading marker and from its ready marker, failure, exit or poison (send outside `c.mu`), using Task 10's `LoadID`.
3. Broker `loadguard.go`: track active loads from that notification; while at least one is active, tick every second and read the local memory view as `modelTargetMemory` does; take the host pool (kind `system`, or `shared` on UMA) and record the minimum available RAM and the minimum available VRAM per device pool for each active load. When the host pool's available bytes are valid and below the floor, call the internal method `engine:load-guard-abort {engine, modelId, loadId, availableBytes, floorBytes, minDeviceAvailableBytes}` once for that load. A stale or unknown sample does nothing. A load is dropped from the watch on `ended` or after 20 minutes. Add `engine:load-guard-abort` to `internalClientMethods`.
4. Manager `model_load_guard.go`: on `engine:load-guard-abort`, if `loadId` is not the current loading child of that model, reply `stale` and do nothing. If the load belongs to a running measurement (Task 6), abort the measurement through its abort path with reason `load-guard`. Otherwise unload the model through the owner the `unload_model` action uses. Set the model's `LoadReport.Outcome` to `stopped-by-guard` and its new `Guard{AvailableBytes, FloorBytes, MinDeviceAvailableBytes, At}` field (`load_report.go`).
5. `ModelLoadBufferReport.tsx` shows `PAIR stopped this load: available RAM fell to <X> (floor <Y>).` Document the guard in both READMEs; regenerate services-api.
**Verify:** from `services/shared`: `go test -count=1 ./nodepolicy ./modelconfig`; from `services/nvpair-ui-broker`: `go test -count=1 -run '^(TestLoadGuard|TestPolicy)' .` plain and `-race`; from `services/nvpair-engine-manager`: `go test -count=1 -run '^(TestModelLoadGuard|TestModelBuffer|TestModelDiagnostic)' .` plain and `-race`; `go vet ./...` in each; from `desktop/`: `npx vitest run --project unit tests/modular/node-policy.test.ts tests/modular/model-buffer-report.test.tsx tests/modular/engine-settings-bridge.test.ts`, `npm run typecheck`, `npm run lint`, `npm run dead-code:check`, `npm run service-contracts:check`; all exit 0. Pass (new tests, fake clock and injected telemetry): available RAM below the floor during a load produces exactly one abort and the model unloaded with saved intent unchanged; above the floor, floor 0, a stale sample or no active load produce no call; an abort for a superseded `loadId` replies `stale`; an abort during a measurement's load ends the run `failed` with `load-guard`; policy without `loadGuard` decodes to 256 and 1048577 is rejected; the stop message renders.

### Task 12 — Process acceptance covers estimate, Auto, measurement and the guard
**Files:** `services/tests/model_settings_test.go`, `services/tests/model_settings_fixture_test.go`, `services/tests/discovery_relay_test.go`, `services/tests/model_buffer_samples_fixture_test.go`, `services/tests/model_memory_telemetry_test.go` (new), `services/tests/model_memory_estimate_test.go` (new), `services/tests/model_memory_diagnostic_test.go` (new), `services/tests/model_load_guard_test.go` (new), `services/nvpair-engine-manager/README.md`, `services/nvpair-ui-broker/README.md`, `services/nvpair-node-info/README.md`, `docs/model-configuration.mdx`, `docs/performance-telemetry.mdx`, `docs/model-memory-diagnostic.mdx`, `desktop/docs/services-api.md` (generated), `THIRD_PARTY_NOTICES.md` (generated, changed only when telemetry Task 7's license step shows a dependency change)
**Depends on:** 11 — final behaviour of every task
**Complexity:** judgment
**Do:** Execute telemetry Task 7, estimation Task 13 and diagnostic Task 10 as written, with these amendments:
1. Estimation Task 13: drop the "locked engine refuses the remote set" case (locks removed in phase 04).
2. Diagnostic Task 10: drop lock cases; add a configured-paused target that measures (from a peer fixture node) and a halted target that refuses with zero effects; assert the record's provenance carries the fixture's `build_info`, CPU name and RAM, and that one performance sample per request reaches the broker store under contentId and profile id.
3. Add `TestModelLoadGuardProcess`: fake node-info memory through the scanner to broker path drops below the floor during a fake child's load; the model is unloaded, inspection's load report shows `stopped-by-guard`, saved intent and residency policy are unchanged; above the floor nothing happens.
4. Docs as the three tasks state, plus the guard (floor, default, advisory) in `docs/model-configuration.mdx` and the diagnostic speed samples in `docs/performance-telemetry.mdx`; regenerate services-api.
**Verify:** on PC3 in a serialized fixed-path interval (vault `TESTING_ON_PC.md` section 10), from `services/tests`: `go test -count=1 -timeout=90m -run '^(TestModelMemoryTelemetryProcess|TestModelMemoryEstimateProcess|TestModelMemoryDiagnosticProcess|TestModelLoadGuardProcess|TestModelSettings)' .`, then `-race` for the four new families, then `go vet ./...`, with firewall inventory and survivors recorded before and after. Pass: every named family runs with no skip or failure, no new firewall rule, no survivors. Then telemetry Task 7's license and contract commands, and on the Mac `npm run service-contracts:check`, `node scripts/spdx-headers.mjs`, `git diff --check`, all exit 0.

## Verification (whole-plan)

Run after Task 12 on the integrated branch. Keep commands, exit codes and bounded output under a dated directory outside the repository (for example `/tmp/pair-phase05-<date>/`) and cite it in the close-out. A skip, timeout or unrun check leaves the gate pending.

Locally runnable (Mac):
- From `desktop/`: `npm run lint`, `npm run typecheck`, `npm run test:unit`, `npm run dead-code:check`, `npm run service-contracts:check`; each exits 0.
- From the repository root: `node scripts/spdx-headers.mjs` and `git diff --check` exit 0.
- From `services/shared`: `go test -count=1 ./...` and `go vet ./...` exit 0.

PC3 (full suites; the Mac never runs these):
- `services/nvpair-engine-manager`: `go test -count=1 -timeout 60m ./...`, then `go vet ./...`.
- `services/nvpair-ui-broker`, `services/nvpair-proxy`, `services/nvpair-node-info`: `go test -count=1 ./...` and `go vet ./...` in each.
- `services/tests`: Task 12's command. All exit 0 with no feature skip.

External, on the owner's machines, only in intervals the owner authorizes (MSI with no Strata work running, decided.md 2026-10-28). Record in the vault `Validation/` folder under a dated name; anything not observed is marked UNTESTED:
1. MSI: open the LFM2.5-24B-A2B profile configured at 8192 context and a second profile of the same GGUF at the model's trained context. Each shows per-pool ranges and verdicts (not "Unknown"); measure each; the consent shows the resource rundown and any threshold warning; the results show measured peaks beside the estimate's ranges and the record shows `build_info` b11219, the GPU name and VRAM, the CPU name and the RAM size.
2. From the Mac, start a measurement of a Joes profile; Joes's tray shows "Measuring memory" with Cancel; cancel it midway once and confirm saved state is unchanged; requests to Joes route elsewhere during the run.
3. On MSI, the measured profile's native load output appears in its panel and shows the buffer sizes and the CPU expert placement of its `n-cpu-moe` setting.
4. On MSI, in the same authorized interval, set `loadGuard.minAvailableRamMiB` above the machine's available RAM, load a profile, and see the load stopped with the guard message; restore the floor.
5. With no measurement, assign an unmeasured profile to a role and see a request served by it.

## Close-out

- Record verified changes in this repository's `plans/2026-10-31-expectation-gaps/00-index.md`: append ` — done <sha>` to the phase 05 row, and in this file append ` — done <sha>` to each task heading as it lands, per the lifecycle section below.
- Add a vault `WORKLOG.md` entry: the landed shas per task, which vault Todo task numbers each one executed (so their owner can archive them), the entry-condition evidence, the plan-time decisions kept or vetoed, any desktop path renames applied (entry condition 5), the gate results with their evidence directory, and the external acceptance items still UNTESTED.
- Remaining limitations to state: native accuracy unverified until the external items pass; repeated route-triggered loads can be stopped repeatedly by the guard (no hold); estimation Task 14 and EMP Tasks 4-6 open.
- Hand-over to phase 06: name the performance-sample key used (contentId and profile id) and the producer id, so routing reads diagnostic speed.

## Do not

- Do not refactor beyond what the tasks specify. Add no service, telemetry cache, second drain, second reservation or second comparison; the guard's one-second ticker reads the existing broker telemetry cache only.
- Do not let an estimate, stated figure, measured record or guard result block Apply, Save, role assignment or routing, or change admission, ME2 qualification, `memoryBytes`, policy or launch settings.
- Do not refuse to assign or route an unmeasured profile.
- Do not reopen the settled decisions of the three cited Todos or README section 6.
- Do not add a veto on configured Pause, or any engine-lock check, to measurement; only Halt refuses.
- Do not add estimation, option or measurement paths for Ollama, LM Studio or Strata, and do not start estimation Task 14 or EMP Tasks 4-6.
- Do not retain or log prompts, responses, synthetic prompt text or any engine output outside a load window.
- Do not write "guaranteed", "qualified" or "safe" in user-facing text.
- Do not load a real model or touch a GPU in tests, or run real models on any owner machine outside an authorized interval.
- Do not hand-edit `desktop/docs/services-api.md`, `THIRD_PARTY_NOTICES.md`, `services/versions.json` or `CHANGELOG.md`; add no version bumps, release-intent blocks or `Signed-off-by` trailers.

## If stuck

- An entry condition is absent: stop and report which one; do not implement it here.
- A cited Todo task's text conflicts with an amendment here: the amendment wins only for the point it names; for anything else stop and report the conflict.
- A PAIR activation path bypasses the diagnostic reservation, cancellation cannot join an in-flight load, or the token counting endpoint behaves differently from the diagnostic plan: follow the diagnostic Todo's "If stuck" section (stop and report).
- No fresh memory sample exists on a platform: the preview refuses with `memory-telemetry-unavailable` and the guard does nothing; do not read OS memory in the broker or manager.
- A task needs a file outside its Files list: stop and report the file and why.
- Retry only transient tool or setup failures. A failing consent, exclusion, intent, monotonicity or permission assertion is a defect to fix, not to loosen.

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
