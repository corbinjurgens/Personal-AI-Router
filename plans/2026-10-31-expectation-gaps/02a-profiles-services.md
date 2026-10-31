<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution plan: Phase 02a: profiles as the join key in node policy, engine manager, broker and proxy

## Goal

Overall goal of the batch (`README.md` sections 2 and 4): PAIR on the owner's
machines (Mac, MSI, Joes, PC3) is configured through roles that point at named
profiles, each profile being one model identity plus engine plus launch
settings, shown by true identity rather than preset section names, so the
owner's daily loop works without hand-edited files.

This phase's goal (`README.md` section 5 row 02a, target model in section 4,
owner decisions D2 and D3 at their defaults as amended by the 2026-10-31 debate
adjudication F1, F3, F4 and F5 in README section 8): a **profile with a
user-chosen key** is the join key between node policy and the engine manager.
A node-policy profile carries `key`, `displayName`, optional `icon`, `engine`,
`model` (the served model id), `modelIdentity` (phase 01) and a derived
`contextTokens`, and keeps its request options, mode, `localOnly`,
capabilities and limits. Launch settings stay in the engine manager: its typed
loads are keyed by the profile key and the generated llama.cpp preset has one
section per key, so the key is the served model id and one GGUF can back
several profiles with different context. `contextTokens` of a profile with a
typed load is written by the broker's coupled operation from the reviewed
per-slot context and cannot be changed by a raw `policy:set`. Roles become
ordered lists of profile keys, possibly empty, so a role can be declared before
any profile exists; the advertisement publishes every declared role name with
its profiles, identity and bounds, and the proxy resolves a role to its
profiles with candidate = (node, profile), ignoring empty roles. The broker's
existing coupled model-change operation creates, updates and removes a profile
and its typed load in one journaled transaction, locally and through the
existing settings relay from a peer. Tier names and routing order stay as they
are (phase 03). The desktop editors, deploy, the hand-written preset import and
user documentation are phase 02b (`02b-profiles-desktop-and-import.md`).

## Change scope

Intent: make "a named configuration of one model on one engine" a first-class,
key-addressed object across node policy, engine manager, broker and proxy, so N
configurations of one GGUF exist without hand-invented native aliases.

Scope: Structural change — introduce the profile key as the join key between
the broker-owned node policy and the engine-manager-owned typed loads, change
the role and advertisement shapes to (possibly empty) lists of profile keys,
and adapt every Go producer and consumer (shared packages, engine manager,
broker, proxy, node scanner and TUI tests, cross-process tests) plus the
generated service contract document; leave tier names, tier search order, the
candidate comparator, every desktop source file (phase 02b), engine toggles and
locks, estimates and validation unchanged.

Basis: owner decisions D2 and D3 (`README.md` section 6 defaults); the debate
adjudication F1, F3, F4, F5 (`README.md` section 8, `investigation/debate.md`);
the storage split of vault `Investigations/2026-10-27-model-config-gui.md`
("Debate conclusions"); verified evidence L3-C1..C18, L1-C7..C9, A1, A2, A3,
B1, B5.

Compatibility: pre-release fork rule — change formats in place, keep
`nodepolicy.SchemaVersion = 2`, add no capability markers and no mixed-build
handling. The only data imports in this phase are the stored policy shape
(Task 1) and stored engine `modelLoads` (Task 2), authorised by D3; the
hand-written preset import is phase 02b. Served model ids on existing nodes
stay byte-identical through the switch except in the one case P3 names.
Because the desktop's strict bridge parsers reject the new shapes until phase
02b lands, a build containing this phase and not 02b is not installed on the
owner's machines; 02a and 02b ship together.

Non-goals: desktop types, parsers and editors, deploy, the preset import and
user docs (phase 02b); user-defined role names, router tier entries and the
history tier enum (phase 03); role-driven engine activation, tabs, node
settings page, Halt, removal of engine toggles and locks (phase 04); estimate
display and Validate (phase 05); context advertisement on `/v1/models`, smart
routing and `kind: app` (phase 06); a name alias for concrete-model requests;
new JSON-RPC methods.

## Entry conditions

Phase 01 (`01-identity-and-unblock.md`) must have landed. Check all of the
following at the branch base before Task 1; record the commands and outputs in
the phase's WORKLOG entry.

1. `plans/2026-10-31-expectation-gaps/00-index.md` row 01 ends with
   ` — done <sha>` and `git merge-base --is-ancestor <sha> HEAD` exits 0.
2. Structured model identity exists in a shared Go package:
   `git grep -n 'type ModelIdentity struct' -- services/shared/engines/model_identity.go`
   prints one line and that struct has `json:"contentId,omitempty"`,
   `json:"repository,omitempty"` and `json:"quantization,omitempty"` fields.
   This plan calls it `engines.ModelIdentity` (package `nvpair-shared/engines`,
   which `nodepolicy` and `enginesettings` already import).
3. The engine inventory carries it and one function computes it:
   `git grep -n 'json:"modelIdentities' -- services/nvpair-engine-manager/models.go`
   prints the `ModelsResult.Identities` field, and
   `git grep -n 'func (e \*Executor) modelIdentityFromArtifacts' -- services/nvpair-engine-manager`
   prints the function phase 01 Task 3 extracted (identity from a resolved
   `modelFileList` and a per-slot context).
4. Typed loads express the MoE, placement and native-fit options:
   `git grep -n -e 'json:"nCpuMoe' -e 'json:"cpuMoe' -e 'json:"repack' -e 'json:"tensorSplit' -e 'json:"device' -e 'json:"fit,' -e 'json:"fitTarget' -e 'json:"fitCtx' -- services/shared/modelconfig/wire.go`
   prints eight `RequestedLoad` fields.
5. Phase 01's preset readers exist:
   `git grep -n -e 'func readLlamaArtifactPreset' -e 'func readLlamaPresetSections' -- services/nvpair-engine-manager/serving_llamacpp.go`
   prints both, and `previewTypedLlamaConversionExpected` in
   `services/nvpair-engine-manager/modelsettings_load.go` calls
   `readLlamaPresetSections`.

If any condition is absent: stop before Task 1, write a WORKLOG entry
"phase 02a blocked: phase 01 entry condition N missing" (N = the number
above) with the failing command, and report. Do not implement phase 01 work in this phase.

## Context the executor needs

Repository and environment:
- Base the branch on current `origin/integration/join33` with phase 01 merged.
  This plan was written against `integration/join33` at `4eccd05d`
  (2026-10-31); line numbers below are navigation aids at that snapshot only.
  Re-read every named symbol before editing it.
- Repo rules: `.cursor/rules/services-change-propagation.mdc`,
  `service-contracts.mdc`, `type-discipline.mdc`, `no-legacy-fallbacks.mdc`,
  `engines-vs-backends-naming.mdc`, `commit-sign-off.mdc` (commit with
  `git commit -s`). There is no root `AGENTS.md` at the snapshot; the
  `.cursor/rules` files are the repository instructions.
  `desktop/docs/services-api.md` is generated
  (`npm run service-contracts:write`), never hand-edited.
- Each Go component is its own module with `replace nvpair-shared => ../shared` (for example `services/nvpair-proxy/go.mod`), so a change to
  `services/shared/nodepolicy` breaks consumer modules until they are adapted.
- Package direction (verified at `4eccd05d`, non-test imports):
  `nodepolicy` imports only `engines` and `jsonrpc`; `modelconfig` imports
  `engines` and `noderec`; `noderec` imports `nodepolicy`; `enginesettings`
  imports `engines` and `modelconfig`. This phase keeps `nodepolicy` free of
  `modelconfig` (the policy profile carries no load, P1) and lets `modelconfig`
  import `nodepolicy` directly for the profile-key grammar, which adds no cycle
  because `modelconfig` already depends on `nodepolicy` through `noderec`.

Node policy today (verified, L3-C1, C2, A2, L1-C7):
- `services/shared/nodepolicy/policy.go` `Profile` has Name, Engine, Model,
  Quantization, ContextTokens, Capabilities, MemoryBytes, MaxConcurrent,
  RequestOptions, RequestOptionsMode, LocalOnly, MaxOutputTokens,
  Qualification. `Policy.Validate` refuses a duplicate `Name` and a second
  profile for the same `(Engine, Model)` ("another profile already covers");
  `policy_load.go` independently refuses a "duplicate profile model".
  `ProfileFor(engine, model)` returns the first exact match. `Quantization` has
  no reader (L3-C4). `Name` has no functional reader beyond the collision check
  (verify-A A2). Profile names match `profileName`
  `^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$`.
- `Policy.Roles map[string]Role`, `Role{Engine, Model}`; the same `Role` type
  is the element of `ServedModels`, `ResidentModels` and `EngineBindings`
  (`engine_bindings.go` `type EngineBindings []Role`). Role keys are validated
  against `TierOrder` (phase 03 changes that, not this phase).
- `advertise.go` `AdvertisedRole{Engine, Model, ContextTokens, Capabilities, MaxOutputTokens, Qualification}` with a tolerant `UnmarshalJSON` and
  `HybridQualified`; `Advertisement.Roles map[string]AdvertisedRole`;
  `Policy.Advertisement` fills bounds from `ProfileFor(role.Engine, role.Model)` (`advertise.go:427-443`); nothing advertises a role name without a
  model (debate F5). `FilterAdvertisementModels` filters raw advertisement JSON.
  Consumers: proxy `tiers.go` and `discovery.go`, node scanner
  `model_enrichment.go` `decodeInventoryPolicy`, `shared/noderec`
  `DirectoryNode.Policy`, broker `policy_notifications.go`, engine manager
  `nodeprojection.go`, desktop `electron/service-bridge/node-projection.ts`
  and `shared/utils/node-roles.ts` `parsePolicyAdvertisement` (the desktop is
  adapted in phase 02b).
- `projection.go` `PeerPolicy` carries `Roles map[string]Role`;
  `checkPeerInferenceSettings` enforces the `localOnly` lock keyed by
  `(engine, model)` (vault Todo `2026-10-23-pair-model-inference-settings.md`,
  Tasks 1-5 landed; error `-32014` `remote_setting_locked`).
- `policy_load.go` `ParseStored` is the disk-load path only; it removes invalid
  model entries and repairs references in memory, never persists the repair,
  and the broker's `loadNodePolicy` (`services/nvpair-ui-broker/policy.go`)
  puts the node into recovery (paused, file kept as `.bad`) when parsing
  fails. Old-shape files therefore need the Task 1 import or the node pauses.

Model-change wire and engine manager today (verified, L3-C6..C14, A1, A3):
- `services/shared/modelconfig/wire.go`: `RequestedLoad` (portable intent),
  `ResolvedLoad` (target-owned: `PerSlotContext`, `TotalContext`, `Slots`,
  `GPU`, `Options`, `Artifacts`, build and review digests, `DeviceBinding`),
  `ModelLoad{ModelID, Requested, Resolved}`, `LoadPatch`, `PolicyPatch {Generation, RequestOptionsMode, Roles map[string]bool, Served, Resident, Binding}`, `ModelChange{ModelID, ExpectedPolicyRevision, Load, Policy, PreviewDigest, SupersedesRequestID}`, `InspectRequest{Engine, NodeID, ModelID}`. `CanonicalLoads` sorts by `ModelID` and refuses duplicates.
- `services/shared/enginesettings/settings.go` `Config.ModelLoads`,
  `Preview` (Settings, Args, Env, Errors, Restart, Impact, Serving, Diff,
  PreviewDigest, Auto, Estimate), `Request.ValidateModelChange`,
  `Request.NodeID` (`settings.go:74`).
- `services/nvpair-engine-manager/modelsettings_load.go`:
  `compileTypedLlamaInputsExpected` resolves artifacts per load with
  `resolveLlamaArtifacts(..., load.ModelID, ...)` and writes
  `sections[load.ModelID]`; `typedLlamaPreset` renders one `[section]` per key;
  `typedLlamaModelOptions` maps the typed fields (ctx-size = per-slot x slots);
  `patchModelLoad` finds or appends by `ModelID`; `rtLoadResolution` looks up
  the previous resolution by `ModelID`; `previewModelLoadChange` converts an
  external `--models-preset` into typed loads implicitly on the first preview
  when the engine has none (`previewTypedLlamaConversion`, each section becomes
  a `ModelLoad` whose `ModelID` is the section name, `externalConversionResolver`).
  Auto proposals (`op.Proposals`), `RecalculateModel` and review hashes
  (`modelsettings_auto.go` `modelAutoReviewedID`, `modelAutoReviewHash`) are
  keyed by model id. `setport.go` `persistRuntimeSettings` writes
  `runtime.modelLoads` to `<appdir>/engines/<engine>.json` through
  `CanonicalLoads`; `registry.go` decodes it. `serving_identity.go`
  `servingIdentityLocked` keys serving envelopes by `envelope.ModelID`, which
  is the preset section name. Other load readers: `model_buffer_capture.go`,
  `modelsettings_artifacts.go`, `modelcopy_export.go`, `settings.go`,
  `modelsettings_rpc.go` (`validateModelSettingsLaunch`, `InspectModel`).
- The engine manager restarts frozen Auto loads from `Requested` plus
  `Resolved` (`modelAutoCompileFrozen`) without the broker; it never reads node
  policy.

Settings relay today (verified by debate F3 at `4eccd05d`): remote editing of
engine-manager loads already exists. `settings.Request.NodeID` addresses a
peer; the broker's `dispatchEngineSettings` (`services/nvpair-ui-broker/enginesettings.go:776-790`) is "the client RPC surface and the
peer relay", `handleSettingsRelay` (`:845`) serves the peer side, and
`applyEngineSettings` dispatches a `ModelChange` to `applyModelSettings`
(`:642`). The desktop keys engine settings by `${nodeId}:${engine}`
(`desktop/src/ui/stores/engine-settings.store.ts:26-27`). This phase routes
remote load edits through that relay; it adds no copy of the load to policy.

Broker coupled operation today (verified, A1, L3-C9, C10, C16, C18):
- `applyModelSettings`, `reviewModelSettings`, `mergeModelPolicy`,
  `validateModelSettingsEntry`, `modelPolicyImplications`,
  `commitModelPolicyLocked` and `runModelSettingsOperationLocked` are in
  `services/nvpair-ui-broker/modelsettings.go`; review revalidation in
  `modelsettings_reviews.go`. Today `reviewModelSettings` calls
  `mergeModelPolicy` before the worker `engine:preview-launch`, strips the
  policy patch from the worker call, journals the accepted record, drains,
  calls worker `engine:configure-launch` and commits policy after stage
  "native-persisted". `mergeModelPolicy` auto-creates a profile named
  `model-<hash16>` and never writes `ContextTokens`.
- `services/nvpair-ui-broker/modelconfig.go` `residentProfile` and
  `modelConfigEntry` block residency unless
  `envelope.PerSlotContextTokens == pr.ContextTokens` (B5).
- `policy.go` `setLocalPolicy` and `setPeerPolicyAs` are the raw `policy:set`
  write paths; `policyErrorCode` maps typed errors (`-32602` invalid params).
  `policy.go` is one of the broker's three `engine:models` callers, so it can
  read this node's inventory.

Proxy today (verified, L1-C6, C9, L7-C1..C4, C11, C12, B1, B5):
- `services/nvpair-proxy/tiers.go` `declaredRoles{self, peers}`, `of`,
  `forTier`, `resolveTierCandidates` (dedup key `node NUL engine NUL normalized model`, comparator quoted in L7-C20), `hybridLocalCandidates`
  (looks up the candidate's role by tier), `requestNeeds.metBy`. A local
  router tier entry applies only to nodes with `len(declared.of(c)) == 0`.
- `discovery.go` `Node.Roles` and `filteredModels` (keeps only `TierOrder`
  keys and one deterministic unknown-tier marker when none survives, B1);
  `proxy.go` copies `n.Policy.Roles` into `Node.Roles` and defines `candidate`.
- Profile lookups by `(engine, normalized model)`: `admission.go`
  `profileLocked`, `readiness_contract.go` `refreshStrictCapabilityLocked`,
  `residency.go` `residentContextLocked`, `self.go` (served check). They keep
  working unchanged because `Profile.Model` stays the served model id (P2).
  The proxy reads only node policy and peer advertisements; it never reads the
  engine-manager record.

Not verifiable here: every native outcome (llama-server behaviour with two
sections sharing one GGUF, MSI's and Joes's preset contents and launch text,
peer copy of a multi-GB GGUF). Those are machine acceptance items in phase 02b.

## Decisions (settled — do not revisit)

- Launch settings stay in the engine manager, keyed by the profile key; node
  policy holds what routing and the UI need (owner D2 as amended by debate
  adjudication F3, keeping the 2026-10-27 storage split). Not the typed load
  inside the policy profile, because that made two authoritative copies and a
  write guard on raw `policy:set` while the settings relay already provides
  remote edit and deploy (debate F3).
- The profile key is a user-chosen slug, immutable after creation, used as the
  preset section name and the served model id; `displayName` is free text for
  the UI; imported sections keep their section name as the key (debate
  adjudication F4). Not generated hex ids, because they would reach clients'
  model lists and routing history (debate F4); not `Name` as the key, because
  the owner wants free-text names and icons beside a stable machine id.
- A node may declare a role with an empty profile list; validation accepts it,
  the advertisement carries the name, routing ignores it (debate adjudication
  F5).
- The proxy never reads the engine-manager record; it learns everything from
  node policy and peer advertisements.
- Keep `requestOptions`, `requestOptionsMode` and `localOnly` on the profile
  with the landed lock semantics (inference-settings Todo Tasks 1-5), now keyed
  by profile key.
- Tier names, `TierOrder`, `tierSearchOrder`, the tier comparator and router
  tier entries are unchanged (phase 03 owns them).
- A native-fit load (phase 01 Task 8, debate adjudication F1) is a typed load
  like any other in this phase: it has a key, a served id and a derived
  `contextTokens`; it has no configuration key.
- Copy and deploy never carry hardware placement, device bindings, resolved
  values, paths, locks, qualification or measurement; the target re-resolves
  (2026-10-27 debate).
- Pre-release fork: formats change in place, `SchemaVersion` stays 2, no
  version bumps, no capability markers; data imports only where a task says so.

## Decisions made while writing this plan

- P1. One copy of launch intent: the engine-manager
  `ModelLoad{ProfileKey, ModelID, Requested, Resolved}`. The policy profile has
  no `load` field. The coupled operation writes the policy profile's `model`,
  `modelIdentity` and `contextTokens` from the reviewed preview at commit. Raw
  `policy:set` (local and peer, including the desktop's Advanced JSON editor)
  stays whole-document, with two refusals by the broker: a change to
  `contextTokens` or `modelIdentity` of a profile whose key has a typed load on
  this node (`-32602`, message `contextTokens and modelIdentity of profile %q are derived from its load settings; change the load settings instead.`), and a new
  key that P4 refuses. There is no `CheckProfileLaunchFields`. A raw edit that
  deletes a profile with a typed load leaves that load served under its key as
  a plain model; phase 02b's "Import as profiles" adopts it again.
- P2. `Profile.Model` keeps today's meaning: the served model id the engine
  lists and the proxy matches. For a typed llama.cpp profile the coupled
  operation sets `model = key`; for an app-engine profile (Ollama, LM Studio)
  and for a llama.cpp model served without a typed load, `model` is the native
  id and the key is a free slug. The source model a typed profile resolves its
  artifacts from is `ModelLoad.ModelID`, held by the engine manager only.
  Uniqueness per node: `key`, and `(engine, model)` as today; the latter no
  longer blocks N profiles of one GGUF because each typed profile serves under
  its own key. `ProfileFor(engine, model)` is unchanged; `ProfileByKey` is
  added. A `ServedModel()` accessor is not needed.
- P3. Stored-shape imports keep served ids byte-identical. Policy: a stored
  profile without `key` gets `key = nodepolicy.ProfileKeyFromModelID(model)`;
  on a collision with an earlier key on the node, append `-2`, `-3`, … ;
  `displayName` = the old `name`; `model` unchanged. A stored role object
  `{engine, model}` becomes `[key]` of the profile matching `(engine, model)`,
  creating `{key, displayName: model, engine, model}` (key by the same rule)
  when none exists. Engine manager: a stored `modelLoads` entry without
  `profileKey` gets `profileKey = nodepolicy.ProfileKeyFromModelID(ModelID)`.
  `ProfileKeyFromModelID` returns the id unchanged when it passes
  `ValidateProfileKey`; otherwise it lowercases it, turns each run of
  characters outside `[a-z0-9._-]` into `-`, drops leading characters until it
  starts with `[a-z0-9]`, truncates to 64 bytes, and appends `-p` (after
  truncating to 62 bytes) when the result ends in `-api` or `-local` or is a
  reserved alias. Both stores use this one function, so a typed load and its
  policy profile get the same key. The one case where a served id changes: a
  typed load whose model id is not a valid key is served under the slug after
  the upgrade while its policy profile still names the old id; that profile is
  not routable until phase 02b's adoption updates it. The owner's machines
  serve hand-written presets (unverified), so the case is expected only on test
  machines. Not fresh keys plus a name alias, because that adds proxy semantics.
- P4. Key rules (debate adjudication F4): `nodepolicy.ValidateProfileKey(key)`
  accepts `^[a-z0-9][a-z0-9._-]{0,63}$` (the role-name grammar phase 03 adopts),
  refuses a name ending in `-api` or `-local` and refuses `IsReservedAlias`.
  `Policy.Validate` refuses a duplicate key and a key equal to a role name
  declared in the same policy ("profile key %q is also a role name on this
  node"). The broker refuses, on a coupled create and on a raw `policy:set` for
  a key not in the stored policy, a key equal to a model id in this node's
  current inventory (any engine) other than that profile's own `model`
  (message `Profile key %q is already a model id on this node.`). Keys are
  immutable: a coupled change names an existing key to edit it; renaming is
  creating a new profile and removing the old one (phase 02b offers
  "Duplicate as"). The client that creates a profile supplies the key; Go code
  never invents one (P3's import is the exception). For llama.cpp the key also
  passes the engine manager's `validateLlamaModelID`, which the grammar implies.
- P5. `displayName` is display text: 1 to 128 bytes of valid UTF-8, no control
  characters, no leading or trailing whitespace; it need not be unique.
  `icon` is optional text of at most 64 bytes under the same character rule;
  rendering it is phase 04.
- P6. For a profile whose key has a typed load, the broker writes
  `contextTokens` = the reviewed `ResolvedLoad.PerSlotContext` of that load at
  commit (fixed, Auto and native fit alike), which equals the serving
  envelope's per-slot context that strict readiness and residency compare
  against (B5). For native fit it is the configured bound; native fit may
  launch with less (phase 01 decision). A profile without a typed load keeps a
  hand-declared `contextTokens`.
- P7. `Profile.Quantization` is removed (no reader, L3-C4; replaced by
  `modelIdentity`) and `Profile.Name` is replaced by `key` plus `displayName`;
  Task 1's import drops both from stored files.
- P8. `modelIdentity` is written only by the coupled operation and phase 02b's
  import, from the worker preview's identity of the profile's resolved
  artifacts (`enginesettings.Preview.Identities`); it is never copied and raw
  `policy:set` cannot change it on a typed profile (P1). App-engine profiles
  carry none in this phase.
- P9. Wire additions: `ModelLoad.ProfileKey`; `ModelChange.ProfileKey`
  (required except for import), `Remove`, `ImportPreset`;
  `PolicyPatch.DisplayName`, `Icon`, `Capabilities`;
  `PolicyPatch.Roles[role] = true` appends the profile to that role's list when
  absent, `false` removes it and leaves the role declared;
  `InspectRequest.ProfileKey` (optional). An existing typed profile keeps its
  source `ModelID`; re-sourcing is a new profile. Removing the last typed
  profile of a PAIR-managed llama.cpp engine is refused. `ImportPreset` is
  validated on the wire here and refused by the engine manager with "Preset
  import is not available yet." until phase 02b Task 4.
- P10 and P11 (preset import, deploy) are phase 02b decisions.
- P12. Role list order is used only through insertion order into the
  existing stable sort: among candidates the unchanged comparator ranks equal,
  an earlier list position comes first. An empty list contributes no
  candidates and does not count as a declaration for the node-wide "declares
  any role" rule that phase 03 retires.
- P13. Withdrawn by debate adjudication F3: `nodepolicy` no longer imports
  `modelconfig`, so `modelconfig/wire_test.go` keeps importing `nodepolicy`
  and no test file moves.
- P14. Packaging: Tasks 1 to 4 change a shared package and its consumers;
  consumer modules do not compile between Task 1 and their own task. Run the
  tasks on one branch and do not push or merge it before the whole-phase gate
  passes. Task 5 carries the cross-process tests and the generated contract.
  No desktop source file changes in this phase.

## Tasks

### Task 1 — Policy and model-change wire key profiles by a user-chosen key
**Files:** services/shared/nodepolicy/policy.go, services/shared/nodepolicy/advertise.go, services/shared/nodepolicy/projection.go, services/shared/nodepolicy/policy_load.go, services/shared/nodepolicy/policy_test.go, services/shared/nodepolicy/profile_key_test.go (new), services/shared/nodepolicy/advertise_test.go, services/shared/nodepolicy/projection_test.go, services/shared/nodepolicy/policy_load_test.go, services/shared/nodepolicy/model_controls_test.go, services/shared/nodepolicy/engine_binding_offers_test.go, services/shared/nodepolicy/model_reference_size_test.go, services/shared/nodepolicy/routing_test.go, services/shared/modelconfig/wire.go, services/shared/modelconfig/wire_test.go, services/shared/enginesettings/settings.go, services/shared/enginesettings/settings_test.go, services/shared/noderec/noderec_test.go, services/nvpair-node-scanner/availability_test.go, services/nvpair-node-scanner/inventory_capability_test.go, services/nvpair-node-scanner/refresh_test.go, services/nvpair-node-scanner/reverse_peer_test.go, services/nvpair-tui/ui/modelsettings_preserve_test.go
**Depends on:** none — the phase entry conditions must hold
**Complexity:** judgment
**Do:**
1. `services/shared/nodepolicy/policy.go`: add `func ValidateProfileKey(key string) error` (P4 grammar, suffix and reserved-alias rules) and `func ProfileKeyFromModelID(id string) string` (P3 slug rule), each with a doc comment naming the decision.
2. `services/shared/modelconfig/wire.go` (imports `nvpair-shared/nodepolicy` for the key grammar):
   - `ModelLoad` becomes `{ProfileKey string json:"profileKey"; ModelID string json:"modelId"; Requested; Resolved}`. `ModelLoad.Validate` requires `nodepolicy.ValidateProfileKey(ProfileKey) == nil`. `CanonicalLoads` sorts by `ProfileKey`, refuses a duplicate `ProfileKey` with `modelLoads.%d.profileKey: duplicate`, and accepts several loads with one `ModelID`. `CloneLoads`/`EqualLoads` cover the new field.
   - `ModelChange` gains `ProfileKey string json:"profileKey,omitempty"`, `Remove bool json:"remove,omitempty"`, `ImportPreset bool json:"importPreset,omitempty"`. `Validate`: with `ImportPreset`, `ProfileKey`, `ModelID`, `Load`, `Policy` must be empty and `Remove` false; with `Remove`, `ProfileKey` valid and `Load`, `Policy` nil; otherwise `ProfileKey` and `ModelID` valid and at least one of `Load`, `Policy`.
   - `PolicyPatch` gains `DisplayName *string json:"displayName,omitempty"`, `Icon *string json:"icon,omitempty"`, `Capabilities *[]string json:"capabilities,omitempty"`; "select an edit" counts them; update the `Roles` doc comment to P9's list semantics (keys still checked by `isTier`).
   - `InspectRequest` gains `ProfileKey string json:"profileKey,omitempty"`, validated with `nodepolicy.ValidateProfileKey` when non-empty.
3. `services/shared/enginesettings/settings.go`: `Preview` gains `Identities map[string]engines.ModelIdentity json:"identities,omitempty"` (phase 01's type), keyed by profile key; include it in `Clone`.
4. `services/shared/nodepolicy/policy.go` `Profile`, fields in this order with these JSON names: `Key key`, `DisplayName displayName`, `Icon icon,omitempty`, `Engine engine`, `Model model`, `ModelIdentity *engines.ModelIdentity modelIdentity,omitempty`, `ContextTokens contextTokens,omitempty`, then the existing Capabilities, MemoryBytes, MaxConcurrent, RequestOptions, RequestOptionsMode, LocalOnly, MaxOutputTokens, Qualification. Delete `Name` and `Quantization` (P7). Document that `Model` is the served model id (equal to `Key` for a typed llama.cpp profile, P2) and that `ContextTokens` is derived for profiles with a typed load (P6).
   - Add `func (p Policy) ProfileByKey(key string) (Profile, bool)`; keep `ProfileFor(engine, model)` unchanged.
   - `Policy.Roles` becomes `map[string][]string` (ordered profile keys); keep `Role` for `ServedModels`, `ResidentModels`, `EngineBindings` and reword its doc comment to "a served model reference". `normalize` keeps a non-nil map, turns a nil list into an empty list and keeps empty lists (a declared role with no profiles yet, F5).
   - `Validate`: replace the `Name` rules with: key valid (`ValidateProfileKey`) and unique; key not a key of `p.Roles` (P4); `displayName` and `icon` valid per P5; keep the `(engine, model)` duplicate rule and `checkModelRef(pr.Engine, pr.Model, ...)`. Roles: key still in `TierOrder`; the list may be empty; no duplicate key in one list; every key names a profile of this policy.
5. `advertise.go`: replace `AdvertisedRole` with `AdvertisedProfile{Key key, DisplayName displayName, Icon icon,omitempty, Engine engine, Model model, ModelIdentity modelIdentity,omitempty, ContextTokens, Capabilities, MaxOutputTokens, Qualification}`; keep the tolerant `UnmarshalJSON` (also drop a malformed `modelIdentity`) and move `HybridQualified` to it. `Advertisement.Roles` becomes `map[string][]AdvertisedProfile`, built by `Policy.Advertisement` in list order from `ProfileByKey`; every declared role appears, an empty one as `[]`. Adapt `FilterAdvertisementModels` and its helpers to lists: drop entries whose `model` fails `ValidateModelIdentifier` or whose `key` fails `ValidateProfileKey`; keep the role name with an empty list when every entry is dropped.
6. `projection.go`: `PeerPolicy.Roles map[string][]string`; `checkPeerInferenceSettings` keys by profile key: a stored `LocalOnly` profile must exist in incoming with the same key, engine, model and equal options, effective mode and flag; setting `LocalOnly` remotely on a new or unlocked profile keeps the existing refusals.
7. `policy_load.go` `ParseStored`: accept the new shape; before strict decode run the one-time import of the old shape (P3, D3): give profiles without `key` their key and `displayName`, drop `name` and `quantization`, rewrite role objects to one-element key lists (creating profiles as P3 says). Adapt `checkStoredModelShapes`, `removedStoredModels` and `repairStoredModelReferences` so removing an invalid profile also removes its key from role lists (the role stays declared). The repair stays in-memory (existing contract); the next save persists the new shape.
8. Tests (add these names): `TestValidateProfileKey` in `profile_key_test.go` (accepts `lfm-64k`, `joes-tiel35b-udq4kxl`, `a.b_c`, a 64-byte key; refuses empty, `LFM`, `p/x`, `x:y`, `-x`, `coder-api`, `coder-local`, `weak`, `medium-local`, a 65-byte key); `TestProfileKeyFromModelID` (`lfm2.5-24b-a2b-msi-ctx131072` unchanged; `unsloth/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL` → `unsloth-lfm2.5-24b-a2b-gguf-ud-q4_k_xl`; `weak` → `weak-p`; a 100-byte id → 64 bytes that pass `ValidateProfileKey`); `TestValidateProfiles` (two `llamacpp` profiles with keys and models `lfm-8k` and `lfm-32k`: valid; duplicate key; a key equal to a declared role name; two `lmstudio` profiles with one `model`; a role naming an unknown key; a duplicate key in one list: each invalid; `medium: []` valid); `TestAdvertisementListsRoleProfiles` (lists in order, an empty declared role advertised as `[]`, filtered entries leave the name with `[]`); `TestParseStoredImportsOldProfileShape` (old file with profiles `{name:"model-0123456789abcdef", engine:"llamacpp", model:"lfm2.5-24b-a2b-msi-ctx131072", quantization:"Q4"}` and `{name:"hf", engine:"llamacpp", model:"unsloth/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL"}`, roles `medium` → the first model and `weak` → `{llamacpp, "joes-x"}`: keys `lfm2.5-24b-a2b-msi-ctx131072`, `unsloth-lfm2.5-24b-a2b-gguf-ud-q4_k_xl` and a created `joes-x`; display names `model-0123456789abcdef`, `hf`, `joes-x`; every `model` unchanged; roles are one-element key lists); `TestCanonicalLoadsKeysByProfile`; `TestModelChangeProfileValidation`. Update the other listed tests, including the node scanner, noderec and TUI literals, to the new shapes (TUI `ModelLoad` literal gains `ProfileKey: "exact-model"`).
**Verify:** From `services/shared`: `go vet ./... && go test ./...` exits 0,
and `go test -run 'TestValidateProfileKey|TestProfileKeyFromModelID|TestValidateProfiles|TestAdvertisementListsRoleProfiles|TestParseStoredImportsOldProfileShape|TestCanonicalLoadsKeysByProfile|TestModelChangeProfileValidation' -v ./...`
prints `--- PASS` for each of the seven names and no `--- FAIL`. From
`services/nvpair-node-scanner` and from `services/nvpair-tui`:
`go vet ./... && go test ./...` exits 0. These print nothing:
`git grep -n 'Quantization' -- services/shared/nodepolicy`,
`git grep -n 'AdvertisedRole' -- services/shared`,
`git grep -n -e CheckProfileLaunchFields -e ValidateProfileID -- services`,
`git grep -n '"nvpair-shared/modelconfig"' -- services/shared/nodepolicy ':!*_test.go'`.

### Task 2 — Engine manager keys typed loads and preset sections by profile key
**Files:** services/nvpair-engine-manager/modelsettings_load.go, services/nvpair-engine-manager/modelsettings_rpc.go, services/nvpair-engine-manager/modelsettings_auto.go, services/nvpair-engine-manager/modelsettings_artifacts.go, services/nvpair-engine-manager/model_buffer_capture.go, services/nvpair-engine-manager/modelcopy_export.go, services/nvpair-engine-manager/setport.go, services/nvpair-engine-manager/registry.go, services/nvpair-engine-manager/settings.go, services/nvpair-engine-manager/serving_identity.go, services/nvpair-engine-manager/modelsettings_load_test.go, services/nvpair-engine-manager/modelsettings_auto_test.go, services/nvpair-engine-manager/modelsettings_contract_test.go, services/nvpair-engine-manager/modelsettings_estimate_test.go, services/nvpair-engine-manager/modelsettings_artifacts_test.go, services/nvpair-engine-manager/serving_llamacpp_test.go, services/nvpair-engine-manager/linked_model_custom_binding_test.go, services/nvpair-engine-manager/linked_model_hf_test.go, services/nvpair-engine-manager/linked_model_recognition_test.go, services/nvpair-engine-manager/registry_test.go, services/nvpair-engine-manager/README.md, services/nvpair-engine-manager/spec.md
**Depends on:** 1 — uses `ModelLoad.ProfileKey`, `ModelChange.ProfileKey`, `Preview.Identities` and `ProfileKeyFromModelID`
**Complexity:** judgment
**Do:**
1. Split keys by meaning across the listed files. Key by `ProfileKey` everything that identifies a configuration: preset `sections`, compiled options, `rtLoadResolution` for frozen options and review provenance, Auto proposals (`op.Proposals`), the recalculation target (rename `RecalculateModel` to `RecalculateProfile`), the `model` argument of `modelAutoReviewedID`/`modelAutoReviewHash`, preview diff field names (`models.<profileKey>.<key>`), impact model sets and buffer-capture selection. Key by `ModelID` everything that identifies files: `resolveLlamaArtifacts`, the artifact assertions handed to the resolver, inspection facts, `validateTypedLlamaExport` and copy export. When several loads share a `ModelID`, refuse compilation unless their previous artifact assertions are identical ("Profiles of one model disagree on artifacts.").
2. `typedLlamaPreset` writes one `[<profileKey>]` section per load, so the serving declaration's `ModelID` and the router's served id are the profile key. Check `serving_identity.go` keys envelopes by that served id; change it only where it reads `ModelLoad.ModelID`. Native-fit loads (phase 01) follow the same rule.
3. `patchModelLoad(loads, profileKey, modelID, patch)`: find by `ProfileKey`; append `{ProfileKey, ModelID}` when absent; refuse when the existing load has a different `ModelID` ("A profile keeps its model; create a new profile."). Add `removeModelLoad(loads, profileKey)`; refuse when it would leave no load (P9).
4. `previewModelLoadChange`: take `ProfileKey` from the change; handle `Remove` through `removeModelLoad`; remove the implicit conversion branch and refuse every change on an engine that has an external `models-preset` and no typed loads with `modelLoadError("", "import", "Import this engine's presets first.")`; refuse `ImportPreset` with "Preset import is not available yet." (phase 02b Task 4 replaces this refusal). Fill `result.Identities[profileKey]` for every compiled load with phase 01's `e.modelIdentityFromArtifacts(engine, load.ModelID, <that load's resolved artifact list>, load.Resolved.PerSlotContext)`; do not compute identity a second way.
5. `modelsettings_rpc.go`: `validateModelSettingsLaunch` uses `ProfileKey`; `InspectModel` with `InspectRequest.ProfileKey` reports that profile's stored requested and resolved values as the current ones; without it, inspection behaves as a new profile of `ModelID`.
6. Stored import (P3, D3): where `registry.go` decodes the override file's `runtime.modelLoads`, set `ProfileKey = nodepolicy.ProfileKeyFromModelID(ModelID)` on entries that lack `profileKey` before `CanonicalLoads`; `setport.go` `persistRuntimeSettings` then writes `profileKey` on the next save.
7. Add tests: `TestTypedPresetTwoProfilesOneModel` (two loads `lfm-8k` and `lfm-32k`, one `ModelID`, contexts 8192 and 32768: the generated preset has two sections named by the keys with the same `model` path and different `ctx-size`, and serving identity has two envelopes keyed by those keys), `TestModelLoadPatchKeepsProfileModel`, `TestModelLoadRemoveLastProfileRefused`, `TestExternalPresetRequiresImport`, `TestStoredModelLoadsImportProfileKey` (a stored entry with `modelId` `lfm2.5-24b-a2b-msi-ctx131072` gets that key; one with `unsloth/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL` gets `unsloth-lfm2.5-24b-a2b-gguf-ud-q4_k_xl`). Update the listed tests and any other `_test.go` in this module that stops compiling.
8. `README.md`/`spec.md`: describe `modelLoads` keyed by `profileKey`, the generated section name = profile key = served id, that launch settings live only here (node policy holds no copy), the import refusal, and `Preview.identities`.
**Verify:** From `services/nvpair-engine-manager`: `go vet ./... && go test ./...`
exits 0, and `go test -run 'TestTypedPresetTwoProfilesOneModel|TestModelLoadPatchKeepsProfileModel|TestModelLoadRemoveLastProfileRefused|TestExternalPresetRequiresImport|TestStoredModelLoadsImportProfileKey' -v ./...`
prints `--- PASS` for each of the five names and no `--- FAIL`.

### Task 3 — Broker coupled operation writes a profile and its typed load together
**Files:** services/nvpair-ui-broker/modelsettings.go, services/nvpair-ui-broker/modelsettings_reviews.go, services/nvpair-ui-broker/modelconfig.go, services/nvpair-ui-broker/policy.go, services/nvpair-ui-broker/policy_profiles.go (new), services/nvpair-ui-broker/policy_profiles_test.go (new), services/nvpair-ui-broker/modelsettings_apply_test.go, services/nvpair-ui-broker/modelsettings_contract_test.go, services/nvpair-ui-broker/modelsettings_estimate_test.go, services/nvpair-ui-broker/modelsettings_recovery_test.go, services/nvpair-ui-broker/modelconfig_test.go, services/nvpair-ui-broker/policy_test.go, services/nvpair-ui-broker/policy_roles_test.go, services/nvpair-ui-broker/policy_projection_test.go, services/nvpair-ui-broker/remote_management_test.go, services/nvpair-ui-broker/build_apply_test.go, services/nvpair-ui-broker/README.md
**Depends on:** 1 — shared types, `ValidateProfileKey` and `ProfileByKey`
**Complexity:** judgment
**Do:**
1. `reviewModelSettings`: for a change with `Load`, a `Remove` of a profile that has a typed load, and an `ImportPreset`, call the worker `engine:preview-launch` first (as today for load changes, stripping the policy patch), then call `mergeModelPolicy` with the worker preview, so the merge can read the reviewed load; a policy-only change keeps today's path without a worker preview. Keep every existing review check, digest, TTL and journal stage; the policy before/after fingerprints and the `policy` diff entry are computed from the post-preview merge.
2. Rewrite `mergeModelPolicy(stored, engine, change, preview, caller, inventory)` keyed by `change.ProfileKey`, where `inventory` is this node's current model ids from `engine:models` (every engine):
   - Absent profile and not `Remove`: refuse without `Policy.DisplayName` ("A new profile needs a display name."); apply P4 (key grammar, not a declared role name, not an inventory model id other than its own `model`); for `llamacpp` refuse without `Load` ("A llama.cpp profile is created with load settings.") and create `{Key, DisplayName, Icon, Engine: engine, Model: Key}`; for an app engine create `{Key, DisplayName, Icon, Engine: engine, Model: change.ModelID}`. Delete the `model-<hash16>` naming path.
   - Present profile: refuse when `engine` differs; for an app engine refuse when `change.ModelID` differs from `Model`; for `llamacpp` the engine manager's `patchModelLoad` refuses a different source model (Task 2 step 3) and the broker returns that refusal.
   - Apply `DisplayName`, `Icon`, `Capabilities`, `Generation`, `RequestOptionsMode`.
   - With `Load`: set `profile.ContextTokens` to the `Resolved.PerSlotContext` of the preview's `Settings.ModelLoads` entry with this `ProfileKey` (P6) and `profile.ModelIdentity` to `preview.Identities[ProfileKey]` (P8). Store no load in policy (P1).
   - `Roles`: append or remove this key in each named role's list (P9); an emptied list stays as a declared role (F5).
   - `Served`, `Resident`, `Binding`: references are `Role{engine, profile.Model}`.
   - `Remove`: delete the profile, remove its key from every role list (the roles stay declared) and its served reference from `ServedModels`, `ResidentModels`, `EngineBindings`.
   - Keep the final `nodepolicy.Parse`, `ValidateEngineBindings` and, for a peer caller, `MergePeerSet` (the `localOnly` lock).
3. `modelPolicyImplications` and `modelReviewMemoryCurrent` select by profile key (the profile and its served reference); `modelconfig.go` `residentProfile`/`modelConfigEntry` keep matching `pr.Model` (it is the served id, P2).
4. `policy_profiles.go` (new): `checkDerivedProfileFields(stored, incoming nodepolicy.Policy, typedKeys, inventory map[string]bool) error` returning `*derivedProfileFieldError` for P1's two refusals and P4's inventory collision. `typedKeys` are the profile keys of the local engine manager's typed loads, read through the same engine-settings call `reviewModelSettings` uses for the current loads; `inventory` comes from the existing `engine:models` read in `policy.go`. When the settings snapshot cannot be read, refuse a change to `contextTokens` or `modelIdentity` of any `llamacpp` profile with "Engine settings are unavailable; try again.". `policy.go` `setLocalPolicy` and `setPeerPolicyAs` call it before persisting; `policyErrorCode` maps `*derivedProfileFieldError` to `-32602` with its message.
5. Remote edits use the existing relay (debate adjudication F3): confirm that `dispatchEngineSettings` and `handleSettingsRelay` forward a `ModelChange` with `profileKey` unchanged (they relay JSON generically at baseline); change nothing there unless a test below fails.
6. Add tests: `TestModelSettingsCreatesProfileWithLoad` (fake worker preview with load `ProfileKey "lfm-64k"`, per-slot 65536, identities: the committed policy has profile key and model `lfm-64k`, `contextTokens` 65536 and `modelIdentity`, its JSON has no `load` member, and the configure call carries the load with that key); `TestModelSettingsSecondProfileSameModel`; `TestModelSettingsRemoveProfileClearsReferences` (the role list becomes `[]` and the role stays declared); `TestModelSettingsRefusesKeyCollision` (a key equal to an inventory model id, to a declared role name, to an existing key on another engine); `TestPolicySetRefusesDerivedFieldEdit` in `policy_profiles_test.go` (local and peer: a `contextTokens` or `modelIdentity` change on a typed profile is refused with `-32602`; a `displayName`, `icon` or `requestOptions` change on the same profile is accepted; deleting the profile is accepted); `TestResidentEntryAcceptsDerivedContext` (`modelConfigEntry` is not `UnsupportedLoad` when the envelope per-slot equals the derived value); `TestPeerModelChangeKeepsProfileKey` (a change relayed from a peer creates the profile with the requested key on the target). Update the listed tests and any other `_test.go` in this module that stops compiling.
7. `README.md`: the model-settings section describes profile-keyed changes, that launch settings live in the engine manager and remote edits use the settings relay, derived `contextTokens` and `modelIdentity`, and the two `policy:set` refusals.
**Verify:** From `services/nvpair-ui-broker`: `go vet ./... && go test ./...`
exits 0, and `go test -run 'TestModelSettingsCreatesProfileWithLoad|TestModelSettingsSecondProfileSameModel|TestModelSettingsRemoveProfileClearsReferences|TestModelSettingsRefusesKeyCollision|TestPolicySetRefusesDerivedFieldEdit|TestResidentEntryAcceptsDerivedContext|TestPeerModelChangeKeepsProfileKey' -v ./...`
prints `--- PASS` for each of the seven names and no `--- FAIL`.
`git grep -n -F '"model-" +' -- services/nvpair-ui-broker` prints nothing.

### Task 4 — Proxy resolves each role to its ordered profiles
**Files:** services/nvpair-proxy/tiers.go, services/nvpair-proxy/discovery.go, services/nvpair-proxy/proxy.go, services/nvpair-proxy/tiers_test.go, services/nvpair-proxy/hybrid_alias_test.go, services/nvpair-proxy/residency_test.go, services/nvpair-proxy/availability_test.go, services/nvpair-proxy/chat_adapter_test.go, services/nvpair-proxy/engine_binding_candidates_test.go, services/nvpair-proxy/model_controls_test.go, services/nvpair-proxy/reverse_transport_test.go, services/nvpair-proxy/routing_listener_test.go, services/nvpair-proxy/routing_test.go, services/nvpair-proxy/served_models_test.go, services/nvpair-proxy/stop_pause_test.go, services/nvpair-proxy/subscribed_test.go, services/nvpair-proxy/spec.md, services/nvpair-proxy/README.md
**Depends on:** 1 — advertisement and policy shapes
**Complexity:** judgment
**Do:**
1. `proxy.go`: `candidate` gains `profileKey string`; the discovery copy of `n.Policy.Roles` clones each list.
2. `discovery.go`: `Node.Roles map[string][]nodepolicy.AdvertisedProfile`; `filteredModels` keeps, for each `TierOrder` tier, the entries with a valid `key` and `model`, keeps a tier whose list is empty, and keeps the B1 unknown-tier marker rule unchanged (phase 03 removes it).
3. `tiers.go`: `declaredRoles` holds lists. `of(c)` and the node-wide "declares any role" test count only non-empty lists (P12). `forTier(t)` returns the distinct `(engine, model)` pairs ordered by their smallest list position across the declaring nodes (self, then peers by node id), then engine, then model. In `resolveTierCandidates`, a resolved candidate is kept when its node's list for `t` has an entry with that engine and normalized model that passes `metBy`; set `c.profileKey` from that entry; the dedup key for role candidates is `node NUL engine NUL profileKey` (tier-entry candidates keep their model key). Leave `tierSearchOrder`, the tier-entry loop, the comparator and `reserveRound` unchanged (P12). `hybridLocalCandidates` reads the role entry whose key is `c.profileKey`.
4. Profile lookups in `admission.go`, `readiness_contract.go`, `residency.go` and `self.go` keep comparing `pr.Model` (P2); if one of those files stops compiling because it read `Profile.Name` or the old `Roles` shape, fix it in this task and name the file in the commit message.
5. Add tests in `tiers_test.go`: `TestTierRoleTwoProfilesOneModel` (self declares `medium: [lfm-8k (contextTokens 8192), lfm-32k (32768)]`, both typed profiles of one GGUF with `model` equal to their keys, inventory lists both keys: a small request yields two candidates in list order with `profileKey` set; a request whose estimate is 20000 tokens yields only `lfm-32k`), `TestDiscoveryKeepsRoleLists` (lists kept, invalid entries dropped, an empty list kept, unknown-tier marker kept), `TestEmptyRoleIsNotADeclaration` (a node whose only declared role is `medium: []` yields no role candidates and is treated as declaring nothing for router tier entries), `TestHybridCandidateUsesProfileEntry`. Update the listed tests and any other `_test.go` in this module that stops compiling.
6. `spec.md`/`README.md`: a declared role is an ordered, possibly empty list of profiles; an empty role contributes no candidates; the served model of a typed profile is its key.
**Verify:** From `services/nvpair-proxy`: `go vet ./... && go test ./...` exits
0, and `go test -run 'TestTierRoleTwoProfilesOneModel|TestDiscoveryKeepsRoleLists|TestEmptyRoleIsNotADeclaration|TestHybridCandidateUsesProfileEntry' -v ./...`
prints `--- PASS` for each of the four names and no `--- FAIL`.
`diff <(git show <base>:services/nvpair-proxy/tiers.go | sed -n '/^func tierSearchOrder/,/^}/p') <(sed -n '/^func tierSearchOrder/,/^}/p' services/nvpair-proxy/tiers.go)`
prints nothing, where `<base>` is the branch base commit.

### Task 5 — Cross-process tests and the generated contract follow profiles
**Files:** services/tests/model_settings_test.go, services/tests/model_settings_fixture_test.go, services/tests/model_routing_interop_test.go, services/tests/linked_models_test.go, services/tests/paused_advertisement_test.go, services/tests/policy_relay_joint_test.go, services/tests/remote_configuration_test.go, services/tests/served_resident_test.go, services/tests/service_policy_test.go, services/tests/unified_dispatch_test.go, services/tests/profile_roles_process_test.go (new), desktop/docs/services-api.md
**Depends on:** 1, 2, 3, 4 — exercises all of them
**Complexity:** judgment
**Do:**
1. Update the listed cross-process tests to profile keys, role lists and profile-keyed model changes; keep each test's scenario.
2. Add `TestProfileRolesAcrossProcesses` in `profile_roles_process_test.go` using the existing fixtures of `model_settings_test.go` and `model_routing_interop_test.go`: one node applies two profiles `fixture-8k` and `fixture-32k` of one fixture model with different context through the coupled operation, assigns both to `medium`, and a `medium` request from a second node is served with `X-PAIR-Model` equal to one of the two keys; a request whose estimate exceeds the smaller context is served by `fixture-32k`. A third case: a node whose policy declares `strong: []` advertises `strong` with an empty list (read the second node's directory entry) and receives no `strong` request.
3. Add `TestRemoteProfileCreateThroughRelay` in the same file: the second node creates `fixture-relay` on the first node through the settings relay (`NodeID` set); the first node's policy then has that profile with derived `contextTokens`, and its engine settings hold the load under that key.
4. From `desktop`: run `npm run service-contracts:write`, then `npm run service-contracts:check`; commit `desktop/docs/services-api.md` only if the write changed it.
**Verify:** From `services/tests`: `go test -run 'TestProfileRolesAcrossProcesses|TestRemoteProfileCreateThroughRelay|TestModelSettings|TestModelControls|TestStrictModelRouting|TestPolicyRelayJoint|TestXARemoteConfiguration|TestServedResident|TestServicePolicy|TestPAPausedAdvertisement|TestUnifiedDispatch|TestLinkedModels' ./...`
exits 0. From `desktop`: `npm run service-contracts:check` exits 0.

## Verification (whole-plan)

Local gate (all must pass on the final branch head; keep commands, exit codes
and the last 50 lines of any failing output in the WORKLOG entry):
1. Every Go module compiles against the new shared package:
   `for m in services/*/go.mod; do (cd "$(dirname "$m")" && go build ./... && go vet ./...) || echo "FAIL $m"; done`
   prints no `FAIL` line.
2. `go test ./...` exits 0 in `services/shared`, `services/nvpair-engine-manager`,
   `services/nvpair-ui-broker`, `services/nvpair-proxy`,
   `services/nvpair-node-scanner` and `services/nvpair-tui`.
3. Task 5's `services/tests` subset exits 0.
4. From `desktop`: `npm run lint`, `npm run typecheck`, `npm run test:unit`,
   `npm run dead-code:check`, `npm run service-contracts:check` each exit 0
   with no desktop source change. If a desktop test fails because it reads a
   Go-owned fixture this phase changed, stop and report it: the fix belongs to
   phase 02b, and the two phases then land together.
5. From the repo root: `node scripts/spdx-headers.mjs` exits 0 and
   `git diff --check <base>..HEAD` prints nothing.
6. `git diff --name-only <base>..HEAD -- desktop/src desktop/tests` prints
   nothing.

A test that fails is pre-existing only when the same command fails the same
way on the branch base in a separate scratch worktree; otherwise it blocks the
phase.

Machine acceptance: none for this phase alone. A build with 02a and without
02b is not installed on the owner's machines; phase 02b's machine acceptance
covers both.

## Close-out

- Record verified changes in `plans/2026-10-31-expectation-gaps/00-index.md`:
  append ` — done <sha>` to each task heading in this file as it lands and to
  the phase 02a row in the index when the whole-phase local gate passes.
- Add a vault entry to `/home/user/wasabi/4. Projects/Personal-AI-Router/WORKLOG.md`
  with the branch, commit range, gate commands and results, any check that did
  not run (marked untested), and remaining limitations: the desktop cannot
  write the new shapes until phase 02b; app-engine profiles carry no
  `modelIdentity`; tier names and order are unchanged; the P3 served-id change
  for typed loads whose model id was not a valid key.
- Hand-over: phase 02b starts from these names; record any deviation in the
  WORKLOG entry: `Profile.Key` (`key`), `Profile.DisplayName` (`displayName`),
  `Profile.Icon`, `Policy.ProfileByKey`, `nodepolicy.ValidateProfileKey`,
  `nodepolicy.ProfileKeyFromModelID`, `Policy.Roles map[string][]string`,
  `AdvertisedProfile` (`key`, `displayName`, `icon`, `engine`, `model`,
  `modelIdentity`, `contextTokens`), `ModelLoad.ProfileKey` (`profileKey`),
  `ModelChange.ProfileKey`/`Remove`/`ImportPreset`,
  `PolicyPatch.DisplayName`/`Icon`/`Capabilities`, `InspectRequest.ProfileKey`,
  `Preview.Identities`, the broker's derived-field refusal text, and the engine
  manager's "Preset import is not available yet." refusal that 02b Task 4
  replaces.

## Do not

- Do not refactor beyond what the tasks specify.
- Do not put launch settings into node policy, add a policy-side copy of the
  load or a launch-field guard on `policy:set` beyond P1's two refusals.
- Do not change tier names, `TierOrder`, `tierSearchOrder`, the tier
  comparator, router tier entries or the routing-history tier enum.
- Do not let the proxy read engine-manager files or RPCs for profiles.
- Do not generate profile keys in Go code outside P3's stored-shape import.
- Do not copy placement, device bindings, resolved values, paths, `localOnly`,
  qualification, bounds or measurements between nodes.
- Do not add a JSON-RPC method, a capability marker, a schema version bump or
  a compatibility branch; the only imports are the two named in Change scope.
- Do not edit desktop sources or tests (phase 02b); do not edit
  `desktop/docs/services-api.md` by hand, `services/versions.json` or
  `CHANGELOG.md`.
- Do not implement the preset import or deploy (phase 02b).

## If stuck

- Compile errors in a consumer module after Task 1 are expected until that
  module's task lands (P14); do not add shims to make them build early.
- Deterministic test, lint or type failures in files a task owns: fix and
  rerun that task's Verify. Safe to retry any Go check.
- If phase 01's identity type cannot be imported by `nodepolicy` or
  `enginesettings` without a cycle, stop and report the package path and the
  cycle; do not move phase 01's type in this phase.
- If a raw `policy:set` path cannot read the engine settings snapshot without
  holding a lock the settings path also takes (deadlock risk), stop and report
  the call path instead of caching the snapshot on your own.
- If an existing node's stored policy cannot be imported by P3 (for example a
  model that is a reserved alias), the node enters recovery as before: report
  the redacted shape of the offending entry, do not widen the import.
- Stop and report when a change would alter the strict-admission or
  qualification contracts beyond the profile lookups named in Task 4.

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
