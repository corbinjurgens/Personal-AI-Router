<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Verify-A: adversarial source check at integration/join33 4eccd05d

Target: /home/user/pair-join33 (detached, HEAD 4eccd05d "Merge Windows working-set containment sweep"). Read-only; `git status` clean before and after. `go build ./...` in services/nvpair-engine-manager completed with no errors. No tests run.

---

## A1. Public Apply of typed llama loads end to end through the broker

**Verdict: confirmed, with corrected detail.**

Probes: read services/nvpair-ui-broker/enginesettings.go:642-660 and 780-790, modelsettings.go:388-520 and 654-1120, modelsettings_reviews.go:212-252, settingsdrain.go:24-55; services/nvpair-engine-manager/settingsrelay.go:157-220, modelsettings_rpc.go:491-526, settings.go:555-740; services/shared/enginesettings/settings.go:291-296.

Evidence trace:
- Public entry: `dispatchEngineSettings` "apply" -> `applyEngineSettings` (enginesettings.go:787). That calls `applyModelSettings` when `p.ModelChange != nil` (enginesettings.go:651-652).
- Digest check: `applyModelSettings` requires `requestId` and `previewDigest` (modelsettings.go:656-657). It fetches the retained review and refuses on mismatch: `preview.PreviewDigest != p.ModelChange.PreviewDigest` gives "review changed" (697-699). It also refuses when the native launch differs from the review, the policy fingerprint changed, the review is past its TTL, or memory evidence changed (703-751).
- Journal: the accepted record is written by `saveEngineSettingsLocked()` before any effect (768). Then `runModelSettingsOperationLocked` runs.
- Drain: `drainModelSettings(ctx, engine, record.Resume)` (modelsettings.go ~990). It runs only when `op.Change.Load != nil`. A policy-only change does not drain.
- Worker call: `modelSettingsCall(ctx, worker, "engine:configure-launch", configure, nil)` (modelsettings.go:1062). Its payload is `settings.Configure{Engine, Settings, OperationID, Resume}`, and that struct has no `ModelChange` field (shared/enginesettings/settings.go:291-296).
- Worker `ConfigureLaunch` -> `configureLaunchLocked` (settings.go:558-740) runs in this order: `doStop` if `changed && old.Running` (667-671); `persistRuntimeSettings` (672-676); runtime swap and `rebind()`; `doStart` if `!activationBlocked && ((changed && old.Running) || resume)` (718-733). It restarts the single llama-server router process for that engine.
- Policy commit: `commitModelPolicyLocked` runs after stage "native-persisted" (modelsettings.go ~1093-1100).

Corrected detail:
1. The worker's "not ready" refusal fires in more cases than the claim says. modelsettings_rpc.go:497-512:
   ```go
   if p.ModelChange != nil {
       ...
       if !applying && p.ModelChange.Load != nil && p.ModelChange.Policy == nil { return ctx.Err() }
       return modelSettingsParamsError(fmt.Errorf("modelChange: reviewed model settings application is not ready"))
   }
   ```
   It fires for every `ModelChange` on the applying path (load-only too). On the preview path it fires when `Policy != nil` or `Load == nil`. The broker never reaches it:
   - Its preview call clears `change.Policy, change.PreviewDigest = nil, ""` (modelsettings.go reviewModelSettings).
   - Its revalidation and frozen preview send `ModelChange: nil` (modelsettings_reviews.go:237, modelsettings.go:1012).
   - `configure-launch` cannot carry a ModelChange, because settingsrelay.go:208 builds `settings.Request{Engine, RequestID, Settings}`.
2. Several steps are conditional. ConfigureLaunch is skipped when native readback already equals the reviewed settings and no resume is needed (modelsettings.go:1037). The restart is skipped when there is an activation veto (Pause or lock), or when the engine is saved Off and was not running (settings.go:714-718). The drain waits up to 60 s only when `record.Resume` is set.
3. The worker side is llama.cpp only: `previewModelLoadChange` refuses unless the family is llamacpp, `Load != nil` and `Policy == nil` (modelsettings_load.go:941-942).

---

## A2. Profile uniqueness, CanonicalLoads uniqueness, and identifiers

**Verdict: confirmed, with corrected detail.**

Probes: services/shared/nodepolicy/policy.go:406-431, 621, 664-690, 815-825; policy_load.go:520-565; projection.go:99-130; services/shared/modelconfig/wire.go:102-106 and 638-656; grep for readers of `.Profiles` and `.Name`.

Evidence:
- `Policy.Validate` checks `key := pr.Engine + "\x00" + pr.Model; if models[key] { ... "another profile already covers %s %q" }` (policy.go:681-684). It also refuses a duplicate `Name` (674-676). The raw loader independently refuses a "duplicate profile model" (policy_load.go:559-563).
- `CanonicalLoads` sorts by ModelID and refuses `modelLoads.%d.modelId: duplicate` (wire.go:646-653). `ModelLoad` is `{ModelID, Requested, Resolved}` with no other identifier (wire.go:102-106).
- Readers of `Profile.Name` outside validation:
  - The only service reader is the broker's `mergeModelPolicy`. When it generates a profile name `"model-"+hash`, it checks that the name is not already in use (modelsettings.go:244-249).
  - Every other profile lookup keys on (engine, model): proxy `profileLocked` (admission.go:738), `ProfileFor` (policy.go:819), `residentProfile` (broker modelconfig.go:234), `checkPeerInferenceSettings` (projection.go:101), and `modelPolicyImplications`.
  - Every other `.Name` hit is an engine profile (`f.profile.Name`), not a `nodepolicy.Profile`.

Corrected detail:
1. `Profile.Name` is a validated, unique identifier (regex at policy.go:621 plus a duplicate check). It just has no functional reader.
2. The (engine, model) duplicate check compares exact strings without normalization. "llama3" and "llama3:latest" pass `Validate` but are counted as two profiles for the same normalized key by the proxy, which then treats both as ambiguous (readiness_contract.go:90-97, residency.go:104-111).

---

## A3. Typed load option set vs. serving preset path

**Verdict: confirmed.** The requested breakdown follows.

Probes: modelconfig/wire.go:31-60 (`RequestedLoad`, `GPU`) and 83-97 (`ResolvedLoad`); modelsettings_load.go:370-460 (typed compile), 819-912 (`typedLlamaModelOptions`), 1584-1680 (`convertTypedLlamaOptions`), 1690-1860 (option descriptors); serving_llamacpp.go:249-352 (`llamaServingOptions`), 413-449 (`validateLlamaValue`), 659-758 (`parseLlamaServingPreset`), 1035 (placement surfacing). Git: `git log -S` shows 0b7207a7 "Accept MoE expert placement, repacking and load mode in serving mode", merged by c6f414db, which is an ancestor of HEAD.

`RequestedLoad` fields: Context, Slots, GPU{mode,layers}, CacheTypeK/V (only f16 maps, modelsettings_load.go:870), FlashAttention, BatchSize, UBatchSize, Threads, ThreadsBatch, KVOffload, OpOffload, SleepIdleSeconds, CacheRAM.

| option | (a) typed RequestedLoad field | (b) accepted in a parsed preset | (c) refused as unsupported |
|---|---|---|---|
| n-cpu-moe | no | yes; canonical int32 >= 0 (serving_llamacpp.go:306, 432-436) | typed conversion of an external preset refuses it (`convertTypedLlamaOptions` default -> "cannot be represented exactly") |
| cpu-moe | no | yes, but only the value `true`; false gives "Omit cpu-moe..." (305, 424-431) | typed conversion refuses it |
| repack | no | yes; boolean, including the nr/no-repack negative spellings (310, 730-742) | typed conversion refuses it |
| tensor-split | no | yes; no `unsupported` bit and no value validation; surfaced in placement (277, 1035) | typed conversion refuses it |
| split-mode / main-gpu | no | yes, same as tensor-split (276, 278) | typed conversion refuses them |
| device | no portable field. Auto writes `options["device"] = resolved.DeviceBinding.Selector` from the resolved device binding only (modelsettings_load.go:458-460) | yes (279) | typed conversion refuses it |
| model-draft (local draft GGUF) | no field. The target resolver emits `model-draft` from `draft`-role artifacts (modelsettings_load.go:390-404) | yes; explicitly exempt from the unsupported refusal (708) | n/a |
| hf-repo-draft, spec-type, spec-draft-* (n-max, ngl, device, cpu-moe, n-cpu-moe, type-k/v, override-tensor, backend-sampling), spec-default, mtp, dflash, eagle3 | no | no | yes: `unsupported: true` (339-352) -> "Unsupported envelope-affecting preset option." |
| n-cpu-ffn, override-tensor | no | no | yes (304, 307) |

The option descriptors mark every native key other than the mapped B fields as `Supported: false`, with reason "No typed override; requires a mapped native build..." (modelsettings_load.go:1784-1790).

---

## A4. Preset parsing in resolveLlamaArtifactsSelection

**Verdict: confirmed, with corrected detail.**

Probes: modelcopy_export.go:638-760; serving_llamacpp.go:459-486, 659-758; callers via `grep resolveLlamaArtifacts`; modelsettings_artifacts.go:544-553; serving_strata.go:346-374; grep of every `parseLlamaServingPreset(` call site; manifests/llamacpp.json.

Evidence:
- `presetConfigured` becomes true from any of these (modelcopy_export.go:680-695):
  - `rt.Serving != nil && len(rt.ModelLoads)==0`
  - a `--models-preset` argument
  - a non-empty `LLAMA_ARG_MODELS_PRESET` in the command env
  - **the manager process's own** `os.Getenv("LLAMA_ARG_MODELS_PRESET")`
- With a preset configured, the parse error is wrapped at 712-715 as `fmt.Errorf("custom artifact configuration is unsupported: %w", parseErr)`. A launch-options parse error at 698-700 produces the same prefix.
- `parseLlamaServingPreset` refuses:
  - unknown keys: `llamaPresetOptionKey` returns "" -> "Unknown preset option, include or inheritance is unsupported." (698-705). Only the table keys plus temp/top-k/top-p/min-p/seed/threads/threads-batch are known (466-469).
  - unsupported keys other than model, model-draft and load-on-startup -> "Unsupported envelope-affecting preset option." (707-710)
  - inline comments, substitution, duplicate sections and duplicate keys.
- All named examples carry `unsupported: true`: fit-target (366), hf-repo (336), hf-file (337), alias (320), mmproj-auto (323), spec-type (350), override-tensor (304).
- The file is parsed before the model's section is looked up (713 before 717). One bad key in **any** section breaks resolution for **every** model on that engine.
- Callers:
  - Peer-copy export: `llamaExportFilesBound` -> `resolvePreparedExport` -> `resolveLlamaArtifactsSelection` (serving_strata.go:355). The raw error propagates.
  - Inspection: `inspectModelArtifacts` -> `resolveLlamaArtifacts` (modelsettings_artifacts.go:544).
  - Typed preview/compile: modelsettings_load.go:377 and 1437.

Corrected detail: inspection does **not** surface the quoted message. It maps any non-limit, non-ambiguous error to `unavailable("missing", "Configured artifacts are missing, unsupported, partial or changed.")` (modelsettings_artifacts.go:551-552). Typed load preview maps it to "Exact artifacts are missing, ambiguous, changed or exceed the bounded resolver limits." (modelsettings_load.go:379). Only export carries the literal "custom artifact configuration is unsupported: serving.models.<key>: model "<id>": Unsupported envelope-affecting preset option."

Sub-question (does the parsed preset populate the model list or model identity?): **no.**
- Parse call sites: serving_llamacpp.go:900 (serving compile), modelsettings_load.go:1005/1079/1541 (generated snapshots), 1410 (typed conversion), model_buffer_capture.go:1291 (buffer loads from the prepared snapshot), modelcopy_export.go:713.
- The serving compile only cross-checks the preset sections against the declared `rt.Serving.Models` count and IDs (serving_llamacpp.go:921-930).
- The model list comes from manifest `list_models` GET /models `{array:data, field:id}`.
- Identity comes from inventory digests or serving envelopes (see A5).

---

## A5. HF-cache facts and the no-digest/no-sample chain

**Verdict: confirmed, with corrected detail.**

Probes: modellayout.go:1-20, 656-669 (`splitLlamaModel`), 1043-1128 (`cachedRef`), 1158-1300 (`resolveBoundedLlamaCacheArtifacts`); modelfiles.go:113 (`validSHA256`); manifests/llamacpp.json actions; model_identity.go:112-137 (`extractDigests`) and 146-200 (`modelIdentity`); serving_identity.go:135-158; models.go:227; broker model_identity.go:85-103 and 198-215; proxy performance.go:158-186 and 787-793; readiness_contract.go:200-212.

What the HF cache resolution extracts:
- repo and tag from the model ID: `owner/repo[:tag]`, using the last colon (`splitLlamaModel`).
- the snapshot commit: first line of `refs/main`; otherwise the first sorted ref that matches `commitRe` (`cachedRef`).
- the `.gguf` file names under `snapshots/<commit>`, recursively, filtered by tag (`llamaModelFileSet`). Role is `model`, or `companion` (mmproj).
- `f.SHA256 = filepath.Base(real)`, but only `if blobs != "" && filepath.Dir(real) == blobs && validSHA256(filepath.Base(real))` (1260-1263). This is the LFS blob filename (64 lowercase hex), recorded per file and not verified. It is set only when the snapshot entry resolves into `blobs/`.
- the size, and a sampled SHA (`SampleSHA256`) from `bindingExpected`.

No-digest chain:
- `list_models` result is `{"array":"data","field":"id"}` with no `digest`.
- `extractDigests` returns nil when `spec.Digest == ""` (model_identity.go:114). The inventory digests are therefore empty.
- `modelIdentity` fills ModelDigests from those inventory digests only when `!serving` (model_identity.go:179-184). Serving runtimes use `servingIdentityLocked` envelopes instead (serving_identity.go:153).
- The broker forwards `ModelDigests` unchanged (broker model_identity.go:94-99, 207-209).
- `performanceCurrentIdentity`: `digest, found := a.modelDigestLocked(...)`, `valid := exists && found && ...`, and `if !valid ... return ..., false` (performance.go:173-182).
- `startPerformanceAttempt` then does `f.host.dropPerformance(1); return nil, nil` (787-793), so no sample is recorded.

Corrected detail:
1. "Non-serving" means `Runtime.Serving == nil`. A typed ModelLoads apply sets `Settings.Serving = prepared.Config` (modelsettings_load.go:1049), so typed-load engines take the serving envelope identity path and do get digests.
2. The blob SHA is absent for snapshot files that are regular copies rather than links into `blobs/`, which is common for Windows HF caches without symlinks.
3. The blob SHA is per shard. Nothing feeds `modelFile.SHA256` into `ModelDigests`.

---

## A6. Wake, bindings, roles, and concurrent engines

**Verdict: first three sentences confirmed. The last sentence ("Nothing in engine-manager prevents two engines running concurrently") is refuted as stated.**

Probes: wakesleep.go:39-65; engine-manager bound_engine.go:63-187; lifecycle.go:115-167; nodeprojection.go:182-225; desiredstate.go:409-430; switchexecutor.go:1134-1158; broker bound_engine.go:175-245 and 378-520; proxy bound_engine.go:157-161; grep `.Roles` in proxy and engine-manager.

Evidence:
- `Wake` returns `errIntentOff` when `!known || !enabled` (wakesleep.go:57-63).
- The bound path is `activateBoundEngine` -> `StartWith` with Origin local or peer.
  - Its vetoes are `boundEngineAuthority`/`remoteVeto`, which check only Pause and engine lock (nodeprojection.go:182-198). There is no saved-intent check.
  - After start it calls `setDesiredEnabled(engine, true, origin)` (lifecycle.go:160). A saved-Off engine is therefore started and flipped to On.
- Broker authority requires an `EngineBindings` entry (bound_engine.go:224-231). The proxy's `boundModel` checks only `EngineBindings`. `Policy.Roles` is not consulted in either path; it is used by discovery and tiers only.
- Broker `runBoundEngine`:
  - `len(running) > 1` -> "engine-switch-busy"
  - `len(running) == 0` -> cold activation
  - exactly one -> switch away from `running[0]`, subject to protected residency ("no-fit") and dwell cooldown (bound_engine.go:470-520).

Refutation of the last sentence:
- Engine-manager's own bound activation guard refuses while any other engine is running or stopping (or reclaiming), returning `switchBusyError` (bound_engine.go:157-173). The guard runs before and after acquiring the lifecycle lock (lifecycle.go:127-151).
- `automaticSwitchSource` refuses unless the source is the sole running engine: "automatic replacement requires a sole conflicting source" (switchexecutor.go:1155-1157).

What is true: plain Start, Wake and `RestoreEnabled` have no cross-engine exclusion. `RestoreEnabled` starts every saved-On engine concurrently in goroutines (desiredstate.go:409-430). Two engines can run at once through those paths. Exclusivity exists only on the bound and automatic-switch paths.

---

## A7. No binary probing; b11146 gates; consequences on other builds

**Verdict: confirmed, with a material correction.** There is a second build gate that does not read `/props`.

Probes: `git grep` for `--help`, `--version`, `--list-devices`, `build_info`; grep of `exec.Command`/`runCommandOutput`; modelsettings_devices.go:31-36, 175-190, 360-420, 538-600, 618; model_buffer_parser.go:17, 38-40; model_buffer_capture.go:37-39, 240-262, 660-725; serving_llamacpp.go:120-175; modelsettings_load.go:296-345 and 500-525; modelsettings_rpc.go:120-160; modelsettings_auto.go:140-200; modelsettings_artifacts.go:650-675; serving_strata.go:334-341; lifecycle.go:484; shared/engines/serving.go:166-172; manifests/llamacpp.json.

Evidence:
- The help/version/list-devices flags appear only in launch.go:99-101, an arity/skip table.
- The engine-manager executes only these:
  - manifest CLI actions (actions.go:560)
  - install (install.go:498)
  - the engine process
  - `lsof`/`ss`/`ps`/`taskkill`
  - `lms ls --json`
  - the memory-plan helper
  - **itself** as the device helper (`os.Executable()` with `--private-model-devices` and `LD_LIBRARY_PATH` set to the llama install root, modelsettings_devices.go:549-555). This dlopens the ggml libraries but does not run llama-server.
- `/props` `build_info` is read in model_buffer_capture.go:37-39. It is gated by `modelBufferBuildSupported`, which returns `build == "b11146-7fe450e" || build == "b11146-"+modelBufferPin` (model_buffer_parser.go:38-39).
- `ServingReadback` is `{Status, Reason, Configured, LaunchValidated, Observed}`, with no build field. Neither `ServingEnvelope` nor `ServingObservation` has one.

Correction: device enumeration's gate is **not** `/props`. Every caller passes the constant `modelDevicesBuild` ("b11146"). The real check is the **install manifest artifact URL**:

```go
if a.Name == "server" && strings.Contains(a.URL, "/releases/download/"+modelDevicesBuild+"/") && len(a.SHA256) == 64 { pinned = true }
if !pinned { return ..., errors.New("selected runtime is not pinned b11146") }
```
(modelsettings_devices.go:389-396). It also requires the binary to be a managed install path ("native runtime is not managed", 405-407).

- `modelReviewedNativeSource` and `modelReviewedNativeSourceExpected` go through this same snapshot. On failure they return `"Reviewed Auto native library source unavailable."` (serving_llamacpp.go:131-135, 147-150).
- The shipped manifest pins b11146: all 10 download URLs are `releases/download/b11146`.
- Typed `NativeBuildID` for manual loads is a digest of the executable file binding (`modelLoadDigest(json(build))`, modelsettings_load.go:304-313), not a build string.
- `7fe450e1` is the upstream llama.cpp commit constant (`modelDevicesCommit`, `modelBufferPin`), not a commit in this repo (`git log 7fe450e1` -> unknown revision).

Consequences on a machine whose llama-server is b11219 or b11985. This means the install manifest URL is not b11146, or the binary is not a managed install.

| feature | effect | refusal path |
|---|---|---|
| Typed load Preview (manual, non-Auto) | works. No b11146 check; build ID is the executable digest. Device map is degraded (see device enumeration). | n/a |
| Typed Apply (manual) | works by the same trace. The frozen re-preview and ConfigureLaunch do not call the reviewed-source path for non-Auto loads (modelsettings_load.go:330-345 and 506-512 skip `!modelLoadHasAuto`). | n/a |
| Auto (Preview) | **refuses** | `reviewedSource(command)` -> `modelReviewedNativeSourceExpected` -> "Reviewed Auto native library source unavailable." (modelsettings_load.go:506-512, serving_llamacpp.go:147-150). Separately, GPU Auto fit needs `NativeDeviceID`, which is absent, so `modelAutoFit` returns unknown (modelsettings_auto.go:155-159). |
| Already-accepted Auto load at launch | **refuses** (inferred from the code path, not run). Launch `compilePreparedServing` (lifecycle.go:484) -> `compileTypedLlamaExpectedContext` (serving_strata.go:334-336) -> for Auto loads with `Resolved != nil`, `buildID, err = reviewedSource(command)` returns an error (modelsettings_load.go:330-345). The engine would not start until the user explicitly Recalculates or converts the load. | same message |
| Memory estimate | degrades rather than refusing. Estimates gate on `Support == modelEstimatorSupport && BuildID != ""`, where BuildID is the executable digest (modelsettings_estimate.go:664, 711, 875, 1362; the 1362 gate also requires an architecture allowlist). Without a device map, device domains fall to "offload-unmapped". | n/a |
| Inspection ConfigurationKey | degrades. `modelReviewedNativeSource` fails -> `ConfigurationKeyReason = "artifacts-unavailable"` (modelsettings_artifacts.go:664-671). | n/a |
| Device enumeration | **refuses** | `modelDeviceRuntimeSnapshot` -> "selected runtime is not pinned b11146" (devices.go:395). In the memory context, `runtimeErr != nil` leaves `NativeInitAllowed=false`, and `mapModelSettingsDevices` fails with "native initialization prohibited by target activation/Pause" (devices.go:184-186, modelsettings_rpc.go:136-160). The helper also rechecks `selected.Build != modelDevicesBuild` (devices.go:618). |
| Native buffer capture | **refuses at two gates**: (1) the reviewed source is unavailable (stage "runtime-snapshot"), so `sourceID` stays empty and samples become unknown with reason "source-unavailable" (model_buffer_capture.go:250-262, 669-703); (2) `!modelBufferBuildSupported(p.Build)` gives `BufferObservation{Status:"unknown", Reason:"unsupported-build"}` (711-725). | as listed |

---

## A8. gguffacts retained keys

**Verdict: confirmed.**

Probes: services/shared/gguffacts/reader.go:183-200 (`dimensionKeys`, `samplerKeys`, `candidate`) and 370-470 (`metadata`); types.go:51-55, 125-160.

Evidence:
- Numeric retention is `candidate(key)`: a sampler key (`general.sampling.*`), or `dimensionKeys[key after the first '.']`. Dimension keys include context_length, block_count, expert_count, expert_used_count, expert_shared_count, expert feed-forward lengths, attention/ssm/rope dimensions, and others.
- Strings: only `general.architecture` is retained as text (`s.Architecture`). `tokenizer.chat_template*` is retained as {key, SHA256} only. Every other string is skipped (reader.go:405-444).
- Tokens and merges are retained as counts only. `Tensors` keep Name, Type, Dimensions, Offset, Bytes and Layer.
- None of the four keys is retained:
  - `general.name`, `general.basename`, `general.size_label`: strings, not retained.
  - `general.file_type`: uint32; "file_type" is not in `dimensionKeys`, so it is not retained.
- `Shard` has no fields for them.

---

## Claims whose correction would change a plan built on them

- **A7 (most material).** The b11146 gate for device enumeration, Auto and buffer capture is the install manifest's pinned download URL, not `/props`. On a non-b11146 build:
  - Auto Preview refuses.
  - Engines with an already-accepted Auto load would fail to compile at launch (inferred from code).
  - Manual typed Preview and Apply still work.
  - Estimates and inspection degrade instead of refusing.

  A plan that adds `/props`-based build detection alone would not unblock these paths.
- **A6.** Engine-manager does enforce one-running-engine on bound cold activation and automatic switching. Concurrent engines arise only from plain Start, Wake and RestoreEnabled. A plan to "add exclusion in engine-manager" needs to know that exclusion already exists on the bound and automatic paths.
- **A4.** Inspection and typed preview hide the preset-parse error behind generic "missing/unsupported" messages. Only export shows "custom artifact configuration is unsupported...". The manager's own `LLAMA_ARG_MODELS_PRESET` environment variable also triggers parsing, and one bad section breaks every model on the engine. Diagnostics work should target those mappings.
- **A5.** Typed-load engines already get digests through serving envelopes, so the gap is limited to plain (no serving, no typed loads) llama.cpp. The free blob SHA exists only for symlinked or linked HF caches and is per shard.
- **A2.** Profile duplicate detection uses exact strings. Normalized aliases pass Validate but are treated as ambiguous by the proxy. This matters for any plan to allow multiple profiles per model.
- **A1.** The drain and restart are conditional: there is no drain for policy-only changes, and no restart when Paused, locked, or saved Off and not running. The "not ready" guard is unreachable from the broker for apply, not only for policy patches.
