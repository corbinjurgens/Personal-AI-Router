<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# verify-B: adversarial verification at /home/user/pair-join33 (detached, 4eccd05d)

All probes were read-only: git grep, sed and awk on tracked files. Nothing was edited or run.

## B1: role authority, tier validation, unknown roles, role labels. CONFIRMED, with one correction about unknown roles in the proxy

Probe: I read services/nvpair-proxy/tiers.go (`declaredRoles`, `of`, `forTier`, `resolveTierCandidates`), services/nvpair-proxy/discovery.go `filteredModels`, services/shared/nodepolicy/policy.go `Validate` and advertise.go `Advertisement`, desktop/src/shared/utils/node-roles.ts, and desktop/src/ui/stores/node-configuration.store.ts.

- **All tiers are blocked: confirmed.** tiers.go:409-414 reads `for _, e := range pol.Tiers[t] { ... for _, c := range g.resolveCandidates(e.Model) { if len(declared.of(c)) == 0 { add(...) } } }`. The check is the whole map length, not the tier `t`. The type comment at tiers.go:261-264 says so too: "this router's tier entries never apply to it". For self, the declaration comes from `pol.Advertisement().Roles` (tiers.go:281). For peers, it comes from discovery `n.Roles` when `len(n.Roles) > 0` (tiers.go:286).
- **Validation against TierOrder: confirmed.** policy.go:116 defines `TierOrder = {weak, medium, strong}`. policy.go:713-716 rejects unknown keys in `p.Roles` with `roles: unknown tier`, and policy.go:742-745 does the same for `p.Tiers`.
- **"Unknown role names are dropped silently by the proxy's filteredModels": not exactly.**
  - discovery.go:148-178 keeps only TierOrder keys with a valid model.
  - However, `if len(roles) == 0` keeps one deterministic unknown-tier entry as a marker. The comment reads: "A valid declaration of an unknown tier is still authoritative: dropping its presence would enable the router's legacy fallback."
  - Result: a peer that declares only unknown tiers still counts as a declaring node and gets no router tier entries. Unknown keys are dropped only when at least one known tier survives.
  - A peer whose role models all fail `ValidateModelIdentifier` ends with an empty map and does receive tier entries.
  - Neither case logs anything.
- **Desktop parseNodeRoles: confirmed.** node-roles.ts:33-41 iterates only `ROLE_TIERS = ['weak','medium','strong']`, so it does drop unknown tiers.
  - Consequence: a peer that declares only an unknown tier shows no roles in the desktop, yet the proxy treats it as authoritative with no tier entries.
- **Role label: confirmed.** node-configuration.store.ts:111-118 is `roleLabel(role, descriptors)`, which returns `` `${role.model} (${name})` ``, where `name = engineLabel(...)` or the raw `role.engine` when the engine is unknown.
  - `inventoryChoices` (lines 126-140) uses the raw `model.name`.
  - Callers: NodeDeclaredRoles.tsx:23, RoleEditor.tsx:108, BulkNodeConfiguration.tsx:67, plus `roleItems` at store lines 155 and 160.
  - No `formatModelDisplayName` is applied.
- **Profile.Quantization is never shown: confirmed.**
  - `nodepolicy.Profile.Quantization` exists at policy.go:410.
  - `git grep -i quantiz` finds no hit in the three components, in the store, or in electron/service-bridge/node-policy.ts.
  - The only "Quantization" text in the UI is NodePerformance.tsx:52-53, which shows the performance-sample profile and not the policy Profile (see B8: that field is never filled).

## B2: immediate pause. CONFIRMED, with detail

Probe: I read NodeAvailabilityControls.tsx in full, services/nvpair-ui-broker/availability.go (lines 56-460), policy.go:52 and :265, proxy routing.go:120-145 and :407-465, streamframe.go:292-303, proxy.go:741 and :1395, and nvpair-node-scanner/daemon.go:227-234 and :1481-1496.

- **Desktop button: confirmed.**
  - NodeAvailabilityControls.tsx:107-118 renders `{local && <Button ... onClick={() => onRequest({ state: 'paused', pauseMode: 'immediate' })}>Stop local models now</Button>}`.
  - `local = nodeId === undefined` (line 195), and the request carries no `nodeId`.
  - The button is disabled unless `canStop` (lines 40-45).
  - The IPC channel is `node:set-availability` (ui/api/pair-api.ts:299).
- **Local-only: confirmed.** availability.go:76-79: `if mode == PauseImmediate && (caller != "" || origin != PausedLocally)` returns "Immediate pause must be requested on this node."
- **Blocks at acceptance: confirmed.**
  - `acceptAvailabilityLocked` (106-125) sets `r.live = Draining; r.cancelActive = true` under r.mu.
  - `runAcceptedAvailability` (127-) then calls `publishNodeProjection()` and `pushAvailabilityAndActivate(false)`. This sends the availability push, which carries the proxy fence, before it persists and before taking engineConfigMu or transitionMu. The comment there reads "Publish before waiting for engineConfigMu or transitionMu".
- **Cancels without draining: confirmed.**
  - pauseNode:308 sets `r.cancelActive = mode == PauseImmediate || ...`.
  - Line 325 is `if mode != PauseImmediate && !b.waitAdmission(ctx, drainTimeout, idle)`. Immediate skips the drain wait and the `pauseCancelSettle` wait.
- **Parallel engine:sleep with termination proof: confirmed.**
  - `pauseEnginesOn` starts one goroutine per known engine (running, loaded and intent), each calling `callEngineOn(ctx, worker, engineSleepMethod, ...)`. policy.go:52 defines `engineSleepMethod = "engine:sleep"`.
  - Proof means `status.Running != nil && !*status.Running && status.TerminationVerified`.
  - `publishPhysicalPause` sets `r.live = Paused` only when the worker/source pair is unchanged.
  - Before the sleeps, immediate still waits for `prepareSwitchPause` and for `settingsApplyMu`. That is not a drain, but it can delay the stop.
- **Only a local Resume lifts it: confirmed for peers.**
  - availability.go:63-66 refuses peer-origin writes with `localVetoError` while `managementBlockedLocked()`. policy.go:265-267 makes that true for any pending pause, Paused, Draining or live Paused.
  - The desktop hides Resume for remote nodes (`resume = local && ...`).
  - A local Resume is also refused while a build-apply operation runs (`buildApplyBusyError`, availability.go:86-89).
- **Proxy reroute: confirmed.**
  - routing.go:414-434: `if redispatch && a != nil && !r.committed && ...` sets `a.aborted, a.redispatch = true, true` and `r.excluded[a.nodeID] = true`.
  - "committed" means the first accepted raw response byte (routing.go:137-140).
  - Committed requests go to `transitionLocked(routingEnding, streamInterruptReasonNodePaused)`. That reason is `"node paused"` (streamframe.go:303), one of the protocol-correct bounded error terminators.
  - Detail: peer ingress uses `withdrawLocalIf(attempt, false)` (ingress.go:335), so only the origin's driver redispatches. A paused node that is serving a peer's request ends it, and the origin reroutes.
- **Peers learn through a ~15 s poll: confirmed.**
  - Peer `PausedBy` comes from the scanner's DirectoryNode.Policy (proxy.go:1395). That is filled from each peer's /v1/models `policy` field by `refreshModelsOnce` inside `refreshPeersLoop`, which ticks every `defaultPeerRefreshInterval = 15 * time.Second` (daemon.go:234, 1481-1496). mDNS-triggered enrichment can deliver it sooner.
  - The proxy excludes advertised paused peers from inference (proxy.go:741).
  - The broker's `notifyAvailability` reaches local clients only (policy_notifications.go:76-88).
- **Stale comment: confirmed.** desktop/src/shared/types/node-policy.ts:213 reads `/** Omission uses saved drain options. Immediate is local-only and currently unavailable. */`. It is stale.

## B3: memory estimate UI, Auto, configuration key. CONFIRMED; the "always Unknown" has a broader cause than runtime overhead

Probe: I read ModelMemoryEstimate.tsx in full, desktop/src/shared/types/model-config.ts:185-238, engine-manager modelsettings_estimate.go (70-200 and 440-535), modelsettings_auto.go:137-230, 340-480, model_configuration_key.go, modelsettings_artifacts.go:470-680, and model_buffer_capture.go:290-370.

- **The UI ignores service verdicts and ranges: confirmed.**
  - The `Estimate` type has `capacityFit` and `freeFit` (both top-level and per domain), `requiredLowBytes` and `requiredHighBytes`, and per-component `lowBytes` and `highBytes`.
  - `git grep capacityFit|freeFit|lowBytes|highBytes` in desktop/src finds only the type and the electron parser (engine-settings.ts). No UI reads them.
  - ModelMemoryEstimate.tsx has its own `comparison()`: Exceeds / Tight (`capacity/10 + 512 MiB` reserve) / Fits, gated by `complete`.
- **Always "Unknown": confirmed, but for a different reason.**
  - `complete` requires `estimate.status === 'estimated' && !estimate.unknowns?.length`.
  - The only production producer, `calculateModelEstimate` (modelsettings_estimate.go:477), always sets `Status: "partial"`.
  - It always appends `unknownEstimateComponent("runtime-overhead", ...)` (line 506).
  - It always appends `Unknowns: "mmap residency unsized", "transient peaks unsized"` (line 507).
  - It also adds every unknown or partial component to Unknowns (lines 529-532).
  - Every other path returns `modelEstimateUnknown` (status "unknown").
  - So pricing runtime overhead alone would not change the result. The status value and the fixed mmap and transient unknowns would still block it.
- **Auto can never select: confirmed.**
  - modelsettings_auto.go:138: `if memory == nil || ... || out.Status != "estimated" || len(out.Unknowns) != 0 || ...` returns `false, false, 0`.
  - The candidate loop (line 374) sets `selected` only `if fits`.
  - With nothing selected, line 466 returns "unavailable: no complete fitting candidate".
  - `modelEstimateState.run` (the only override hook) has no production assignment.
- **Configuration key: confirmed, with extra conditions.** The key is produced only in two places:
  - **Inspection** (modelsettings_artifacts.go:484, 527-531, 651-679): llama.cpp only (other engines return "unavailable"). It needs a `Runtime.ModelLoads` entry for the model with `Resolved != nil` (otherwise the reason is "not-applied"). `compileTypedLlamaInputs` must yield a finite `Allocation`.
  - **Buffer capture** (model_buffer_capture.go:369), iterating typed loads.
  - `modelConfigurationKey` (model_configuration_key.go:58-88) also returns "" in these cases:
    - GPU mode is neither `cpu` nor `layers` with an explicit count. `native-default` gets no key, even for a typed applied load.
    - There are no memory-topology pools (`in.Memory.Target` is absent).
    - There are no bindings.
  - Preset-only serving and unconfigured llama.cpp get no key, as claimed.

## B4: diagnostic run absent. CONFIRMED

Probe: I grepped for `drainNode(`, `engine:diagnostic`, "Measure", stored-profile kind validation, and the full list of `"engine:..."` method strings.

- **drainNode only from tests: confirmed.** `Broker.drainNode` is defined at engineswitch_drain.go:35. Every call site is in engineswitch_drain_test.go (357, 389, 432, 473, 482, 495, 509, 554, 711).
- **No diagnostic RPC: confirmed.** No `engine:diagnostic-*` string exists in services or desktop. The enumerated `engine:` methods include `engine:sleep`, `engine:wake`, `engine:set-stated-memory` and `engine:clear-stated-memory`, with no diagnostic or measure method.
- **Stated kind only: confirmed.** shared/modelconfig/memory_profile.go:53 is `if !digest(p.ConfigurationKey) || p.Kind != "stated"`, and every writer uses `Kind: "stated"` (modelsettings_rpc.go:294).
  - "measured" exists only as an `EstimateSource.Kind` label (memory_plan.go:545, wire.go:913). It is not a stored profile.
- **No Measure control: confirmed.** No "Measure this configuration" or Measure control exists in desktop/src/ui. The only "measure" hits are layout ResizeObserver code and NodePerformance's "Measured runs".
- Related, not a refutation: a passive per-launch buffer observation does exist. `modelBufferCapture` in lifecycle.go and model_buffer_capture.go keeps it in memory, attaches it to inspection results (modelsettings_artifacts.go:707), and shows it in desktop ModelLoadBufferReport.tsx. It is not a diagnostic run and not a stored profile.

## B5: context gating. CONFIRMED, with corrections on scope and cross-checks

Probe: I read tiers.go:50-110, 249-258, 364-415, 466-500, 548-590, routing.go:680-725, admission.go `admit` (361-575), local_qualification.go:440-462, readiness_contract.go:125-160, residency.go:100-131, nvpair-ui-broker/modelconfig.go:302, nodepolicy/advertise.go:427-442, and hybrid_alias_test.go:339-363.

- **metBy call sites and estimate: confirmed.**
  - `metBy` has two call sites, both inside `resolveTierCandidates` (tiers.go:397 for declared roles and 404 for tier entries).
  - The estimate is `contextTokens: len(body)/4` plus the first positive value of `max_tokens`, `max_completion_tokens` or `options.num_predict` (tiers.go:73, 94-99).
  - `metBy` returns `contextTokens == 0 || contextTokens >= n.contextTokens` (line 258).
- **Scope correction:** `resolveTierCandidates` is also reached by hybrid alias requests through `hybridLocalCandidates` (tiers.go:472, routing.go:714). The gate therefore applies to bare-tier and hybrid alias requests alike.
- **Role context source: confirmed.** advertise.go:431-432 copies `adv.ContextTokens = pr.ContextTokens` from the matching Profile.
  - A role with no Profile advertises ContextTokens 0, which means unbounded.
  - It also advertises no capabilities, so `metBy` rejects that role for any request that needs tools, vision, structured output or embeddings.
- **Ordinary admission: confirmed.** `admit` has no token or context check.
- **Correction on context checks elsewhere:** strict and hybrid admission does gate context. local_qualification.go:460 rejects when `bound.input+bound.output > min(profile.ContextTokens, q.ContextTokens)` using accounted tokens, and chat_adapter.go:592 has a similar check.
- **Alias record shape: confirmed.** hybrid_alias_test.go:355 pins exactly `{created,id,object,owned_by}`. The values are `created: 0`, `object: "model"`, `owned_by: "pair"` (tiers.go:575-580). There are three records per configured tier: `t`, `t-api` and `t-local`.
- **"Never synchronized" is confirmed. "Cross-checked only in residentContextLocked" is refuted (incomplete).**
  - No code writes `Profile.ContextTokens` from a typed load.
  - Besides residency.go:116-127 (`residentContextLocked`, which requires equality with the envelope `PerSlotContextTokens`), there are two more cross-checks:
    1. Strict readiness, `qualifiedModelLocked`, at readiness_contract.go:140 and 154. For any strict-qualified model it requires `pr.ContextTokens <= q.ContextTokens` and `q.ContextTokens <= envelope.PerSlotContextTokens`. That indirectly bounds the profile by the live per-slot context.
    2. The broker's model residency entry, nvpair-ui-broker/modelconfig.go:302, blocks with `UnsupportedLoad` unless `envelope.PerSlotContextTokens == pr.ContextTokens`.
  - Outside strict and resident paths, the two values are still unrelated.

## B6: display-name formatter and unused fields. CONFIRMED

Probe: I read format-model-display-name.ts in full, shared/constants/engines.ts:10 (`EngineTypes = ['ollama','lm-studio','llama-cpp']`), shared/types/engines.ts:176-200, electron/service-bridge/node-projection.ts:460-475, and engine-manager models.go:56-66 and 380.

- **Formatter on `unsloth/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL` with engine `'llama-cpp'`:**
  1. There is no arrow, and `HF_PREFIX` fails, so `isHfModel` is false.
  2. The switch takes the default branch. The name contains '/', so it calls `formatHuggingFaceModelName`.
  3. Split on '/' and drop the first segment: `LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL`.
  4. `.replace(/-GGUF(:[^\s/]*)?$/i,'')` gives `LFM2.5-24B-A2B`.
  5. `[-_]` becomes a space: `LFM2.5 24B A2B`.
  - The quantization is lost entirely; this branch adds no "(quant)" suffix.
  - Call sites: ModelManager.tsx:90 and WorkloadItemCard.tsx:213.
- **ModelItem quantization and digest are unread: confirmed.**
  - The fields exist in shared/types/engines.ts:184 and 186 and in ui/types/engine-info.ts:18 and 20.
  - In ui/, they appear only as writes of `''` (ModelManager.tsx:135 and 137).
- **The service never fills them: confirmed, for every engine.**
  - `ModelsResult` (models.go:56-62) carries only name lists (`models`, `modelsByEngine`, `loadedByEngine`), plus policy, memoryEvidence and downloadState. There is no quantization or digest.
  - The desktop builds every inventory ModelItem through `modelItem()` in node-projection.ts:460-475, with `quantization: ''` and `digest: ''`.
  - So the fields are empty for all engines, not just llama.cpp.

## B7: model inventory and main-app chrome. CONFIRMED, with an intermediate component

Probe: I read engine-models.store.ts, grepped for `<ModelManager`, `<ModelSection`, `<BackendRow` and `<NodeEngineSettings`, and read MainApp/ContentBar.tsx.

- **Store keys: confirmed.** engine-models.store.ts:13-14: `modelsKey = `${nodeId}:${engineType}``, holding all nodes; removal is by `${nodeId}:` prefix.
- **ModelManager mount: confirmed.** `<ModelManager backend nodeId>` is mounted only in ModelSection.tsx:62, inside a collapsed `<details>` "Models" accordion.
  - ModelSection is mounted by BackendRow.tsx:200 when expanded and `canShowAccordions`.
  - BackendRow is mapped per backend in NodeEngineSettings.tsx:140, which NodeCardDetails.tsx:369 mounts per node.
- **Other store readers:** ModelCopyModal, RoleEditor, BulkNodeConfiguration, ModelEngineBindingsEditor and NodeEngineSettings also read the store. They are pickers or derived values, not model lists.
- **ContentBar: confirmed.** ContentBar.tsx:15-18 `TAB_ITEMS` holds only Overview and Settings. The toolbar Buttons are only "Endpoints" (line 50) and "Add node" (line 62).

## B8: performance fingerprint, LoadNs, consumers. CONFIRMED

Probe: I read proxy performance.go:139-212 and 930, performance_extract.go:60-92, engine-manager lifecycle.go:355 and 1371, and grepped every importer of performancewire or performancestore.

- **Fingerprint contents: confirmed.**
  - `performanceFingerprintOwned` hashes Run (`EngineRunID`), Settings, Launch, Model (digest), API and Schema, then `"Options"` via `performanceOptionsWrite(body, ...)` (performance.go:188-211).
  - `ProfileKey` is `{Host, Engine, Model, Fingerprint}`.
  - `st.runID = newEngineRunID()` runs on each launch (lifecycle.go:355 and 1371), so each restart gives a new Fingerprint, and therefore a new key.
- **LoadNs source: confirmed.** LoadNs is set only at performance_extract.go:78, for family `ollama` on streaming `/api/chat` or `/api/generate`. The llama.cpp and Strata path (`performanceExtractDaily`) does not set it.
- **Display-only: confirmed.**
  - Consumers: broker performancestore with `performance:get` and `performance:subscribe` for clients (broker performance.go:583-615), the workload-manager peer relay (`performance:publish`, `performance:summary`, `performance:received`), the TUI, and the desktop NodePerformance.
  - nvpair-job-scheduler has no hit. The proxy only produces samples.
- Extra: the proxy builds the sample profile as `performancewire.Profile{EngineFamily: ...}` only (performance.go:930). `Profile.Quantization` and `Profile.ContextTokens` are never filled, so NodePerformance always shows "--" for both.

## Corrections that would change a plan built on these claims

1. **B5:** context is enforced beyond the tier gate. Strict and hybrid admission enforces `min(profile.ContextTokens, q.ContextTokens)` with accounted tokens (local_qualification.go:460). Strict readiness requires `Profile.ContextTokens <= q.ContextTokens <= envelope.PerSlotContextTokens` (readiness_contract.go:140 and 154). The broker's resident entry requires equality (modelconfig.go:302).
   - A plan to synchronize `Profile.ContextTokens` from typed loads must cover strict readiness and broker residency, not only `residentContextLocked`.
   - The tier gate also applies to hybrid alias requests.
2. **B3:**
   - The desktop's "Unknown" and Auto's refusal are caused by `status: "partial"` plus the fixed "mmap residency unsized" and "transient peaks unsized" unknowns, not only by the unknown runtime-overhead component. Pricing runtime overhead alone will not unblock either.
   - The configuration key also needs explicit GPU placement (`cpu` or `layers` with a count; `native-default` gets no key) and a memory-target pool.
3. **B1:** the proxy keeps an unknown-tier marker, so a peer that declares only unknown tiers is still authoritative and gets no tier entries. The desktop's `parseNodeRoles` hides it, so the two sides disagree.
4. **B6:** quantization and digest are empty for all engines (the desktop's `modelItem()` hard-codes `''`, and `ModelsResult` has names only). Showing quantization needs a new service field, not just a UI change.
5. **B8:** the performance profile's quantization and context fields are never filled by the proxy. Any plan that reuses them for display or keying must first add a producer.
