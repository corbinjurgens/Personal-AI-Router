<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution plan: Phase 02: profiles as the join key for model, engine and launch settings

## Goal

Overall goal of the batch (`README.md` sections 2 and 4): PAIR on the owner's
machines (Mac, MSI, Joes, PC3) is configured through roles that point at named
profiles, each profile being one model identity plus engine plus launch
settings, shown by true identity rather than preset section names, so the
owner's daily loop works without hand-edited files.

This phase's goal (`README.md` section 5 row 02, target model in section 4,
owner decisions D2 and D3 at their defaults): a **profile with its own id** is
the single user-facing configuration object. A node-policy profile carries
`id`, `name`, optional `icon`, `modelIdentity` (phase 01) and a `load` section
holding the typed `RequestedLoad`, which becomes the single source of launch
intent. The engine manager keys its typed loads by profile id and writes the
generated llama.cpp preset with one section per profile id, so the native model
id a profile is served under is its id, and one GGUF can back several profiles
with different context. `contextTokens` of a profile with a load is derived
from its reviewed load, never typed by hand, and is what the advertisement,
strict readiness and broker residency see. The one-profile-per-(engine, model)
rule is replaced by uniqueness of id, of name and of the served model. Roles
become ordered lists of profile ids, the advertisement publishes per role the
profiles with their identity and bounds, and the proxy resolves a role to its
profiles with candidate = (node, profile). The broker's existing coupled
model-change operation creates, updates and removes a profile and its typed
load in one journaled transaction. A profile deploys to another node (copying
the model first when the target lacks it), and a one-time import turns each
hand-written preset section on MSI and Joes into a profile without changing
what clients see. Tier names and routing order stay as they are (phase 03);
the desktop tabs are phase 04, and this phase only keeps the existing editors
working on the new schema.

## Change scope

Intent: make "a named configuration of one model on one engine" a first-class,
id-keyed object across node policy, engine manager, broker, proxy and desktop,
so N configurations of one GGUF exist without inventing native aliases.

Scope: Structural change — introduce the profile id as the join key between
the broker-owned node policy and the engine-manager-owned typed loads, change
the role and advertisement shapes to profile lists, and adapt every producer
and consumer (shared Go packages, engine manager, broker, proxy, node scanner
and TUI tests, desktop types, bridge, stores and the existing editors); leave
tier names, tier search order, the candidate comparator, desktop navigation
and tabs, engine toggles and locks, estimates and validation unchanged.

Basis: owner decisions D2 and D3 (`README.md` section 6 defaults); the earlier
storage split (vault `Investigations/2026-10-27-model-config-gui.md`, "Debate
conclusions"); verified evidence L3-C1..C18, L1-C7..C9, A1, A2, A3, B1, B5.

Compatibility: pre-release fork rule — change formats in place, keep
`nodepolicy.SchemaVersion = 2`, add no capability markers and no mixed-build
handling. The only data imports are the three this plan names (stored policy
shape in Task 1, stored engine `modelLoads` in Task 2, hand-written preset
sections in Task 6); each is authorised by D3. Served model ids on existing
nodes stay byte-identical through the switch (plan-time decision P3).

Non-goals: user-defined role names, fallback lists, router tier entries and
the history tier enum (phase 03); Models and Profiles tabs, node settings
page, Halt, removal of engine toggles and locks (phase 04); estimate display
and Validate (phase 05); context advertisement on `/v1/models`, smart routing
and `kind: app` (phase 06); a name alias for concrete-model requests; new
JSON-RPC methods.

## Entry conditions

Phase 01 (`01-identity-and-unblock.md`) must have landed. Check all of the
following at the branch base before Task 1; record the commands and outputs in
the phase's WORKLOG entry.

1. `plans/2026-10-31-expectation-gaps/00-index.md` row 01 ends with
   ` — done <sha>` and `git merge-base --is-ancestor <sha> HEAD` exits 0.
2. Structured model identity exists in a shared Go package:
   `git grep -n 'json:"contentId' -- services/shared` prints at least one
   struct field whose struct also declares a repository field and a file or
   quantization label field. This plan calls that type `ModelIdentity`; use
   the name and package phase 01 gave it. The package must be importable from
   `services/shared/nodepolicy` and `services/shared/enginesettings` without an
   import cycle (`nvpair-shared/engines` and `nvpair-shared/modelconfig` both
   qualify).
3. The engine inventory carries it: `git grep -n 'ModelIdentity'` (with phase
   01's name) lists a use in `services/nvpair-engine-manager/models.go`, and
   phase 01 provides a function that computes it from a resolved artifact list
   (find it with `git grep -n 'func .*Identit' -- services/nvpair-engine-manager`).
4. Typed loads express the MoE and placement options: `type RequestedLoad struct` in `services/shared/modelconfig/wire.go` has fields for
   `n-cpu-moe`, `cpu-moe`, `repack`, `tensor-split` and `device`.
5. Phase 01's lenient preset reader exists (its task names the function) and
   the external-preset conversion path (`previewTypedLlamaConversion` in
   `services/nvpair-engine-manager/modelsettings_load.go`) no longer fails the
   whole file on one unsupported key.
6. The desktop has phase 01's identity tag formatter
   (`git grep -n -i 'identity' -- desktop/src/shared/utils` lists an exported
   function that renders repository and quantization tags).

If any condition is absent: stop before Task 1, write a WORKLOG entry
"phase 02 blocked: phase 01 entry condition N missing" (N = the number
above) with the failing command, and report. Do not implement phase 01 work in this phase.

## Context the executor needs

Repository and environment:
- Base the branch on current `origin/integration/join33` with phase 01 merged.
  This plan was written against `integration/join33` at `4eccd05d`
  (2026-10-31); line numbers below are navigation aids at that snapshot only.
  Re-read every named symbol before editing it.
- Repo rules: `.cursor/rules/services-change-propagation.mdc`,
  `service-contracts.mdc`, `renderer-stores.mdc`, `type-discipline.mdc`,
  `no-legacy-fallbacks.mdc`, `engines-vs-backends-naming.mdc`,
  `commit-sign-off.mdc` (commit with `git commit -s`). There is no root
  `AGENTS.md` at the snapshot; the `.cursor/rules` files are the repository
  instructions. `desktop/docs/services-api.md` is generated
  (`npm run service-contracts:write`), never hand-edited.
- Each Go component is its own module with `replace nvpair-shared => ../shared` (for example `services/nvpair-proxy/go.mod`), so a change to
  `services/shared/nodepolicy` breaks consumer modules until they are adapted.

Node policy today (verified, L3-C1, C2, A2, L1-C7):
- `services/shared/nodepolicy/policy.go` `Profile` has Name, Engine, Model,
  Quantization, ContextTokens, Capabilities, MemoryBytes, MaxConcurrent,
  RequestOptions, RequestOptionsMode, LocalOnly, MaxOutputTokens,
  Qualification. `Policy.Validate` refuses a duplicate `Name` and a second
  profile for the same `(Engine, Model)` ("another profile already covers");
  `policy_load.go` independently refuses a "duplicate profile model".
  `ProfileFor(engine, model)` returns the first exact match. `Quantization` has
  no reader (L3-C4). Profile names match `profileName`
  `^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$`.
- `Policy.Roles map[string]Role`, `Role{Engine, Model}`; the same `Role` type
  is the element of `ServedModels`, `ResidentModels` and `EngineBindings`
  (`engine_bindings.go` `type EngineBindings []Role`). Role keys are validated
  against `TierOrder` (phase 03 changes that, not this phase).
- `advertise.go` `AdvertisedRole{Engine, Model, ContextTokens, Capabilities, MaxOutputTokens, Qualification}` with a tolerant `UnmarshalJSON` and
  `HybridQualified`; `Advertisement.Roles map[string]AdvertisedRole`;
  `Policy.Advertisement` fills bounds from `ProfileFor(role.Engine, role.Model)`; `FilterAdvertisementModels` filters raw advertisement JSON.
  Consumers: proxy `tiers.go` and `discovery.go`, node scanner
  `model_enrichment.go` `decodeInventoryPolicy`, `shared/noderec`
  `DirectoryNode.Policy`, broker `policy_notifications.go`, engine manager
  `nodeprojection.go`, desktop `electron/service-bridge/node-projection.ts`
  and `shared/utils/node-roles.ts` `parsePolicyAdvertisement`.
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
  `modelconfig/wire_test.go` is `package modelconfig` and imports
  `nvpair-shared/nodepolicy`; once `nodepolicy` imports `modelconfig` that is
  an import cycle in tests.
- `services/shared/enginesettings/settings.go` `Config.ModelLoads`,
  `Preview` (Settings, Args, Env, Errors, Restart, Impact, Serving, Diff,
  PreviewDigest, Auto, Estimate), `Request.ValidateModelChange`.
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

Broker coupled operation today (verified, A1, L3-C9, C10, C16, C18):
- `services/nvpair-ui-broker/enginesettings.go` `applyEngineSettings`
  dispatches to `applyModelSettings` when `ModelChange != nil`; it needs no
  change in this phase. `applyModelSettings`, `reviewModelSettings`,
  `mergeModelPolicy`, `validateModelSettingsEntry`, `modelPolicyImplications`,
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

Proxy today (verified, L1-C6, C9, L7-C1..C4, C11, C12, B1, B5):
- `services/nvpair-proxy/tiers.go` `declaredRoles{self, peers}`, `of`,
  `forTier`, `resolveTierCandidates` (dedup key `node NUL engine NUL normalized model`, comparator quoted in L7-C20), `hybridLocalCandidates`
  (looks up the candidate's role by tier), `requestNeeds.metBy`.
- `discovery.go` `Node.Roles` and `filteredModels` (keeps only `TierOrder`
  keys and one deterministic unknown-tier marker when none survives, B1);
  `proxy.go` copies `n.Policy.Roles` into `Node.Roles` and defines `candidate`.
- Profile lookups by `(engine, normalized model)`: `admission.go`
  `profileLocked`, `readiness_contract.go` `refreshStrictCapabilityLocked`
  (chain `Profile.ContextTokens <= qualification <= envelope per-slot`, about
  lines 140-154), `residency.go` `residentContextLocked` (equality with the
  envelope), `self.go` (served check). The proxy reads only node policy and
  peer advertisements; it never reads the engine-manager record.

Desktop today (verified, L1-C4, C12, C13, L3 Q3, B1):
- `desktop/src/shared/types/node-roles.ts` (`RoleTier`, `NodeRole`,
  `NodeRoles`, `NodePolicyAdvertisement`), `shared/utils/node-roles.ts`
  (`ROLE_TIERS`, `parseNodeRoles`, `parsePolicyAdvertisement`,
  `sameNodeRoles`), `electron/service-bridge/node-projection.ts`,
  `ui/stores/node-configuration.store.ts` (`roleLabel` returns
  `` `${role.model} (${name})` ``, `roleKey`, `inventoryChoices`, `roleItems`,
  `chosenRole`, `suggestRoles`, `roleDifferences`),
  `ui/components/RoutingSettings/RoleEditor.tsx`,
  `RoutingSettings/BulkNodeConfiguration.tsx`,
  `NodeList/NodeDeclaredRoles.tsx`.
- Model settings: `shared/types/model-config.ts` (`ModelChange`,
  `ModelLoad`, `PolicyPatch`, `InspectModelRequest`),
  `shared/types/node-policy.ts` (`NodeModelReference`),
  `shared/utils/node-policy-model-settings.ts` (`parseModelSettings`,
  `createModelSettingsProfile`, `removeModelSettingsProfile`,
  `profileNameValid`), `shared/utils/node-policy-model-copy.ts`
  (`portableLoadPatch`, `modelPolicyPatch`, `copyModelSettings`),
  `electron/service-bridge/engine-settings.ts` (strict response shapes,
  reserved-id check in `modelId`), `ui/stores/engine-settings.store.ts`
  (`modelSettingsKey(target, modelId)`), `ui/components/ModelSettings/`
  `ModelSettingsPanel.tsx`, `ModelProfileControls.tsx`, `ModelLoadSettings.tsx`
  (shows `JSON.stringify(routing.roles[tier])`), `ModelSettingsCopy.tsx`
  (already hands off to `ModelManager/ModelCopyModal.tsx` for content copy),
  `RoutingSettings/ModelSettingsEditor.tsx`.
- Peer model copy: JSON-RPC `engine:copy-model` (desktop `ui/api/engine-api.ts`
  `copyModel`, `electron/service-bridge/model-copy.ts`, engine manager
  `modelcopy_control.go` `Manager.runCopy`), vault Framework `Models.md`
  "Copy a model to a peer".

Not verifiable here: every native outcome (llama-server behaviour with two
sections sharing one GGUF, MSI's actual preset contents and launch text, peer
copy of a multi-GB GGUF). Those are machine acceptance items below.

## Decisions (settled — do not revisit)

- Profiles live in node policy (broker-owned, revision compare-and-set,
  remotely editable, advertised); the engine manager keys its typed loads by
  profile id (owner D2 default). Not engine-manager-owned profiles mirrored
  into policy, because policy is the remotely edited and advertised store.
- Keep the storage split from the 2026-10-27 debate: launch effects (resolved
  loads, the generated preset, serving identity) stay in the engine manager,
  policy stays in the broker, one coupled broker operation writes both, and
  the profile id is the join key. Not one merged record.
- The proxy never reads the engine-manager record; it learns everything from
  node policy and peer advertisements.
- Import each hand-written preset section into a profile once, after which the
  hand-written preset is not used (owner D3 default).
- Keep `requestOptions`, `requestOptionsMode` and `localOnly` on the profile
  with the landed lock semantics (inference-settings Todo Tasks 1-5), now keyed
  by profile id.
- Tier names, `TierOrder`, `tierSearchOrder`, the tier comparator and router
  tier entries are unchanged (phase 03 owns them).
- Copy and deploy never carry hardware placement, device bindings, resolved
  values, paths, locks, qualification or measurement; the target re-resolves
  (2026-10-27 debate).
- Pre-release fork: formats change in place, `SchemaVersion` stays 2, no
  version bumps, no capability markers; data imports only where a task says so.
- Desktop navigation and tabs are phase 04; this phase changes only what the
  existing editors need to keep working.

## Decisions made while writing this plan

- P1. The engine-manager `ModelLoad` keeps `requested` beside `resolved` as
  the applied copy, because engine restarts recompile frozen Auto loads from
  it without the broker. The policy profile's `load` is the authoritative
  intent; the coupled operation writes both from one reviewed patch, and raw
  `policy:set` (local and peer) cannot change `load`, `model`, `engine`,
  `modelIdentity` or `contextTokens` of a profile that has a load, nor add or
  remove such a profile (`nodepolicy.CheckProfileLaunchFields`). Not removing
  `requested` from the engine manager, because the engine manager would then
  have to read policy.
- P2. `Profile.ServedModel()` is the profile id when the profile has a `load`
  and its `model` otherwise. Uniqueness per node: id, name, and
  `(engine, ServedModel())`. App-engine profiles (Ollama, LM Studio) are
  therefore one per native model. A `load` is allowed only for engine
  `llamacpp`.
- P3. Imports keep served ids byte-identical: a stored profile without `id`
  gets `id = model` when that is a valid, unused profile id, else `p-` plus the
  first 20 hex digits of sha256(engine + "\x00" + model); a stored role object
  `{engine, model}` becomes `[id]` of the matching profile, creating
  `{id, name, engine, model}` (name = model when unused, else the id) when none
  exists; a stored engine `modelLoads` entry without `profileId` gets
  `profileId = modelId`; a preset section becomes the profile whose id and name
  are the section name. Not fresh generated ids plus a name alias in concrete
  routing, because that adds proxy semantics and would change what clients and
  `X-PAIR-Model` see during the switch.
- P4. New profile ids are generated by the client that creates the profile:
  `p-` followed by 20 lowercase hex digits from a CSPRNG (desktop
  `newProfileId()`); the broker validates grammar and uniqueness and never
  invents an id. Id grammar (`modelconfig.ValidateProfileID`): the current
  profile-name pattern `^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$`, not a reserved
  tier alias, and for llama.cpp also accepted by the engine manager's
  `validateLlamaModelID`. Deploy keeps the id on the target.
- P5. `name` is display text: 1 to 128 bytes of valid UTF-8, no control
  characters, no leading or trailing whitespace, unique per node. `icon` is
  optional text of at most 64 bytes under the same character rule; rendering it
  is phase 04.
- P6. For a profile with a `load`, the broker writes `contextTokens` = the
  reviewed `ResolvedLoad.PerSlotContext` of that profile at commit (fixed and
  Auto alike), which equals the serving envelope's per-slot context that
  strict readiness and residency compare against (B5). A profile without a
  `load` keeps a hand-declared `contextTokens`.
- P7. `Profile.Quantization` is removed (no reader, L3-C4; replaced by
  `modelIdentity`); Task 1's import drops it from stored files.
- P8. `modelIdentity` is written only by the coupled operation and the import,
  from the worker preview's identity of the profile's resolved artifacts
  (`enginesettings.Preview.Identities`); it is never copied, never accepted
  from `policy:set`. App-engine profiles carry none in this phase.
- P9. Wire additions: `ModelLoad.ProfileID`; `ModelChange.ProfileID`
  (required except for import), `Remove`, `ImportPreset`; `PolicyPatch.Name`,
  `Icon`, `Capabilities`; `PolicyPatch.Roles[role] = true` appends the profile
  to that role's list when absent, `false` removes it; `InspectRequest.ProfileID`
  (optional). An existing profile keeps its `model`; re-sourcing is a new
  profile. Removing the last typed profile of a PAIR-managed llama.cpp engine
  is refused.
- P10. The implicit external-preset conversion on the first preview becomes
  the explicit import (`ImportPreset`). Any other model change on an engine
  that still runs a hand-written preset is refused with "Import this engine's
  presets first." Import is all-or-nothing: one section that cannot be
  represented refuses the whole import with per-section reasons, because a
  partial import would stop serving the excluded sections.
- P11. Deploy is orchestrated by the desktop (`ModelSettingsCopy`) over
  existing RPCs: `engine:copy-model` when the target lacks a model with the
  same `contentId`, then the target's model-settings preview and apply through
  the existing relay. No new JSON-RPC method. Copied: id, name, icon, portable
  load fields (`portableLoadPatch` rules), generation request options as
  `modelPolicyPatch` copies them, `requestOptionsMode`, capabilities and the
  selected role memberships.
- P12. Role list order is used only through insertion order into the
  existing stable sort: among candidates the unchanged comparator ranks equal,
  an earlier list position comes first.
- P13. The `nodepolicy`-using part of `modelconfig/wire_test.go` moves to a
  new external test file (`package modelconfig_test`) to break the test import
  cycle.
- P14. Packaging: Tasks 1 to 4 change a shared package and its consumers;
  consumer modules do not compile between Task 1 and their own task. Run the
  tasks on one branch and do not push or merge it before the whole-phase gate
  passes. The desktop tasks are separate because the TypeScript side has its
  own checks.

## Tasks

### Task 1 — Policy and model-change wire key profiles by id
**Files:** services/shared/nodepolicy/policy.go, services/shared/nodepolicy/advertise.go, services/shared/nodepolicy/projection.go, services/shared/nodepolicy/policy_load.go, services/shared/nodepolicy/policy_test.go, services/shared/nodepolicy/advertise_test.go, services/shared/nodepolicy/projection_test.go, services/shared/nodepolicy/policy_load_test.go, services/shared/nodepolicy/model_controls_test.go, services/shared/nodepolicy/engine_binding_offers_test.go, services/shared/nodepolicy/model_reference_size_test.go, services/shared/nodepolicy/routing_test.go, services/shared/modelconfig/wire.go, services/shared/modelconfig/wire_test.go, services/shared/modelconfig/wire_external_test.go (new), services/shared/enginesettings/settings.go, services/shared/enginesettings/settings_test.go, services/shared/noderec/noderec_test.go, services/nvpair-node-scanner/availability_test.go, services/nvpair-node-scanner/inventory_capability_test.go, services/nvpair-node-scanner/refresh_test.go, services/nvpair-node-scanner/reverse_peer_test.go, services/nvpair-tui/ui/modelsettings_preserve_test.go
**Depends on:** none — the phase entry conditions must hold
**Complexity:** judgment
**Do:**
1. `services/shared/modelconfig/wire.go`:
   - Add `func ValidateProfileID(id string) error` implementing P4's grammar
     (reuse `validateModelID`'s reserved-alias rule).
   - `ModelLoad` becomes `{ProfileID string json:"profileId"; ModelID string json:"modelId"; Requested; Resolved}`. `ModelLoad.Validate` requires a
     valid `ProfileID`. `CanonicalLoads` sorts by `ProfileID`, refuses a
     duplicate `ProfileID` with `modelLoads.%d.profileId: duplicate`, and
     accepts several loads with one `ModelID`. `CloneLoads`/`EqualLoads` cover
     the new field.
   - `ModelChange` gains `ProfileID string json:"profileId,omitempty"`,
     `Remove bool json:"remove,omitempty"`, `ImportPreset bool json:"importPreset,omitempty"`. `Validate`: with `ImportPreset`,
     `ProfileID`, `ModelID`, `Load`, `Policy` must be empty and `Remove`
     false; with `Remove`, `ProfileID` valid and `Load`, `Policy` nil;
     otherwise `ProfileID` and `ModelID` valid and at least one of `Load`,
     `Policy`.
   - `PolicyPatch` gains `Name *string json:"name,omitempty"`, `Icon *string json:"icon,omitempty"`, `Capabilities *[]string json:"capabilities,omitempty"`; "select an edit" counts them; update the
     `Roles` doc comment to P9's list semantics (keys still checked by
     `isTier`).
   - `InspectRequest` gains `ProfileID string json:"profileId,omitempty"`,
     validated with `ValidateProfileID` when non-empty.
2. Apply P13: move every test in `modelconfig/wire_test.go` that references
   `nodepolicy` into `services/shared/modelconfig/wire_external_test.go` with
   `package modelconfig_test`, unchanged in substance.
3. `services/shared/enginesettings/settings.go`: `Preview` gains
   `Identities map[string]ModelIdentity json:"identities,omitempty"` (phase
   01's type), keyed by profile id; include it in `Clone`.
4. `services/shared/nodepolicy/policy.go`:
   - `Profile` fields in this order with these JSON names: `ID id`, `Name name`, `Icon icon,omitempty`, `Engine engine`, `Model model`,
     `ModelIdentity *ModelIdentity modelIdentity,omitempty`, `Load *modelconfig.RequestedLoad load,omitempty`, `ContextTokens contextTokens,omitempty`, then the existing Capabilities, MemoryBytes,
     MaxConcurrent, RequestOptions, RequestOptionsMode, LocalOnly,
     MaxOutputTokens, Qualification. Delete `Quantization` (P7). Document that
     `Model` is the source model the engine resolves artifacts from and that
     `ContextTokens` is derived for profiles with a load (P6).
   - Add `func (pr Profile) ServedModel() string` (P2) and `func (p Policy) ProfileByID(id string) (Profile, bool)`. Change `ProfileFor(engine, model)` to match `pr.ServedModel() == model`.
   - `Policy.Roles` becomes `map[string][]string` (ordered profile ids); keep
     `Role` for `ServedModels`, `ResidentModels`, `EngineBindings` and reword
     its doc comment to "a served model reference". `normalize` drops empty
     role lists and keeps a non-nil map.
   - `Validate`: replace the `(engine, model)` duplicate rule with: id valid
     (`modelconfig.ValidateProfileID`) and unique; name valid per P5 and
     unique; icon valid per P5 when set; `checkModelRef(pr.Engine, pr.Model, ...)`; `Load` only for engine `llamacpp` and `Load.Validate()` passes;
     `(engine, ServedModel())` unique and `ServedModel()` not a reserved
     alias. Roles: key still in `TierOrder`; list non-empty, no duplicate id,
     every id names a profile. Resident checks use `ProfileFor` (served).
   - Add `func CheckProfileLaunchFields(stored, incoming []Profile) error`
     returning `*ProfileLaunchFieldsError` (message names the profile id
     only) when: a stored profile with `Load` is missing from incoming (by
     id); an incoming profile with `Load` is not in stored; for one id where
     either side has `Load`, `Engine`, `Model`, `Load` (structural equality),
     `ModelIdentity` or `ContextTokens` differ; for any id, `ModelIdentity`
     differs; a new incoming profile sets `ModelIdentity`. Do not call it from
     `MergePeerSet` (the broker calls it on raw writes, Task 3).
5. `advertise.go`: replace `AdvertisedRole` with `AdvertisedProfile{ID id, Name name, Engine engine, ServedModel servedModel, ModelIdentity modelIdentity,omitempty, ContextTokens, Capabilities, MaxOutputTokens, Qualification}`; keep the tolerant `UnmarshalJSON` (also drop a malformed
   `modelIdentity`) and move `HybridQualified` to it.
   `Advertisement.Roles` becomes `map[string][]AdvertisedProfile`, built by
   `Policy.Advertisement` in list order from `ProfileByID`. Adapt
   `FilterAdvertisementModels` and its helpers to lists: drop entries whose
   `servedModel` fails `ValidateModelIdentifier` or whose `id` fails
   `ValidateProfileID`; drop a role whose list becomes empty.
6. `projection.go`: `PeerPolicy.Roles map[string][]string`;
   `checkPeerInferenceSettings` keys by profile id: a stored `LocalOnly`
   profile must exist in incoming with the same id, engine, model and
   `ServedModel()` and equal options, effective mode and flag; setting
   `LocalOnly` remotely on a new or unlocked profile keeps the existing
   refusals.
7. `policy_load.go` `ParseStored`: accept the new shape; before strict decode
   run the one-time import of the old shape (P3, D3): give profiles without
   `id` their id, drop `quantization`, rewrite role objects to one-element id
   lists (creating profiles as P3 says). Adapt `checkStoredModelShapes`,
   `removedStoredModels` and `repairStoredModelReferences` so removing an
   invalid profile also removes its id from role lists. The repair stays
   in-memory (existing contract); the next save persists the new shape.
8. Tests (add these names): `TestValidateProfileIdentity` (two `llamacpp`
   profiles with a load and one `model`, different ids: valid; duplicate id,
   duplicate name, two `lmstudio` profiles without load on one model, a load
   on `lmstudio`, a role naming an unknown id, a duplicate id in one list: each
   invalid); `TestAdvertisementListsRoleProfiles`; `TestCheckProfileLaunchFields`
   (each refusal case and the allowed edits: name, icon, request options,
   capabilities); `TestParseStoredImportsOldProfileShape` (old file with
   profile `{name, engine:"llamacpp", model:"lfm2.5-24b-a2b-msi-ctx131072", quantization:"Q4"}` and roles `medium` -> that model, `weak` ->
   `{llamacpp, "joes-x"}` parses to ids equal to the models, role lists, and a
   created `joes-x` profile); `TestCanonicalLoadsKeysByProfile`;
   `TestModelChangeProfileValidation`. Update the other listed tests, including
   the node scanner, noderec and TUI literals, to the new shapes (TUI
   `ModelLoad` literal gains `ProfileID: "exact/model"`).
**Verify:** From `services/shared`: `go vet ./... && go test ./...` exits 0,
and `go test -run 'TestValidateProfileIdentity|TestAdvertisementListsRoleProfiles|TestCheckProfileLaunchFields|TestParseStoredImportsOldProfileShape|TestCanonicalLoadsKeysByProfile|TestModelChangeProfileValidation' -v ./...`
prints `--- PASS` for each of the six names and no `--- FAIL`. From
`services/nvpair-node-scanner` and from `services/nvpair-tui`:
`go vet ./... && go test ./...` exits 0. `git grep -n 'Quantization' -- services/shared/nodepolicy` and `git grep -n 'AdvertisedRole' -- services/shared`
print nothing.

### Task 2 — Engine manager keys typed loads and preset sections by profile id
**Files:** services/nvpair-engine-manager/modelsettings_load.go, services/nvpair-engine-manager/modelsettings_rpc.go, services/nvpair-engine-manager/modelsettings_auto.go, services/nvpair-engine-manager/modelsettings_artifacts.go, services/nvpair-engine-manager/model_buffer_capture.go, services/nvpair-engine-manager/modelcopy_export.go, services/nvpair-engine-manager/setport.go, services/nvpair-engine-manager/registry.go, services/nvpair-engine-manager/settings.go, services/nvpair-engine-manager/serving_identity.go, services/nvpair-engine-manager/modelsettings_load_test.go, services/nvpair-engine-manager/modelsettings_auto_test.go, services/nvpair-engine-manager/modelsettings_contract_test.go, services/nvpair-engine-manager/modelsettings_estimate_test.go, services/nvpair-engine-manager/modelsettings_artifacts_test.go, services/nvpair-engine-manager/serving_llamacpp_test.go, services/nvpair-engine-manager/linked_model_custom_binding_test.go, services/nvpair-engine-manager/linked_model_hf_test.go, services/nvpair-engine-manager/linked_model_recognition_test.go, services/nvpair-engine-manager/registry_test.go, services/nvpair-engine-manager/README.md, services/nvpair-engine-manager/spec.md
**Depends on:** 1 — uses `ModelLoad.ProfileID`, `ModelChange.ProfileID` and `Preview.Identities`
**Complexity:** judgment
**Do:**
1. Split keys by meaning across the listed files. Key by `ProfileID`
   everything that identifies a configuration: preset `sections`, compiled
   options, `rtLoadResolution` for frozen options and review provenance, Auto
   proposals (`op.Proposals`), the recalculation target (rename
   `RecalculateModel` to `RecalculateProfile`), the `model` argument of
   `modelAutoReviewedID`/`modelAutoReviewHash`, preview diff field names
   (`models.<profileId>.<key>`), impact model sets and buffer-capture
   selection. Key by `ModelID` everything that identifies files:
   `resolveLlamaArtifacts`, the artifact assertions handed to the resolver,
   inspection facts, `validateTypedLlamaExport` and copy export. When several
   loads share a `ModelID`, refuse compilation unless their previous artifact
   assertions are identical ("Profiles of one model disagree on artifacts.").
2. `typedLlamaPreset` writes one `[<profileId>]` section per load, so the
   serving declaration's `ModelID` and the router's served id are the profile
   id. Check `serving_identity.go` keys envelopes by that served id; change it
   only where it reads `ModelLoad.ModelID`.
3. `patchModelLoad(loads, profileID, modelID, patch)`: find by `ProfileID`;
   append `{ProfileID, ModelID}` when absent; refuse when the existing load has
   a different `ModelID` ("A profile keeps its model; create a new profile.").
   Add `removeModelLoad(loads, profileID)`; refuse when it would leave no load
   (P9).
4. `previewModelLoadChange`: take `ProfileID` from the change; handle `Remove`
   through `removeModelLoad`; remove the implicit conversion branch and refuse
   every change on an engine that has an external `models-preset` and no typed
   loads with `modelLoadError("", "import", "Import this engine's presets first.")`; refuse `ImportPreset` with "Preset import is not available yet."
   (Task 6 replaces this refusal). Fill `result.Identities[profileID]` for every
   compiled load from phase 01's identity function over that load's resolved
   artifacts; do not compute identity a second way.
5. `modelsettings_rpc.go`: `validateModelSettingsLaunch` uses `ProfileID`;
   `InspectModel` with `InspectRequest.ProfileID` reports that profile's
   stored requested and resolved values as the current ones; without it,
   inspection behaves as a new profile of `ModelID`.
6. Stored import (P3, D3): where `registry.go` decodes the override file's
   `runtime.modelLoads`, set `ProfileID = ModelID` on entries that lack
   `profileId` before `CanonicalLoads`; `setport.go` `persistRuntimeSettings`
   then writes `profileId` on the next save.
7. Add tests: `TestTypedPresetTwoProfilesOneModel` (two loads, one `ModelID`,
   contexts 8192 and 32768: the generated preset has two sections named by the
   profile ids with the same `model` path and different `ctx-size`, and serving
   identity has two envelopes keyed by those ids), `TestModelLoadPatchKeepsProfileModel`,
   `TestModelLoadRemoveLastProfileRefused`, `TestExternalPresetRequiresImport`,
   `TestStoredModelLoadsImportProfileID`. Update the listed tests and any other
   `_test.go` in this module that stops compiling.
8. `README.md`/`spec.md`: describe `modelLoads` keyed by `profileId`, the
   generated section name = profile id, the import refusal, and
   `Preview.identities`.
**Verify:** From `services/nvpair-engine-manager`: `go vet ./... && go test ./...`
exits 0, and `go test -run 'TestTypedPresetTwoProfilesOneModel|TestModelLoadPatchKeepsProfileModel|TestModelLoadRemoveLastProfileRefused|TestExternalPresetRequiresImport|TestStoredModelLoadsImportProfileID' -v ./...`
prints `--- PASS` for each of the five names and no `--- FAIL`.

### Task 3 — Broker coupled operation writes a profile and its typed load together
**Files:** services/nvpair-ui-broker/modelsettings.go, services/nvpair-ui-broker/modelsettings_reviews.go, services/nvpair-ui-broker/modelconfig.go, services/nvpair-ui-broker/policy.go, services/nvpair-ui-broker/modelsettings_apply_test.go, services/nvpair-ui-broker/modelsettings_contract_test.go, services/nvpair-ui-broker/modelsettings_estimate_test.go, services/nvpair-ui-broker/modelsettings_recovery_test.go, services/nvpair-ui-broker/modelconfig_test.go, services/nvpair-ui-broker/policy_test.go, services/nvpair-ui-broker/policy_roles_test.go, services/nvpair-ui-broker/policy_projection_test.go, services/nvpair-ui-broker/remote_management_test.go, services/nvpair-ui-broker/build_apply_test.go, services/nvpair-ui-broker/README.md
**Depends on:** 1 — shared types and `CheckProfileLaunchFields`
**Complexity:** judgment
**Do:**
1. `reviewModelSettings`: for a change with `Load`, a `Remove` of a profile
   that has a load, and an `ImportPreset`, call the worker
   `engine:preview-launch` first (as today for load changes, stripping the
   policy patch), then call `mergeModelPolicy` with the worker preview, so the
   merge can read the reviewed load; a policy-only change keeps today's path
   without a worker preview. Keep every existing review check, digest, TTL and journal
   stage; the policy before/after fingerprints and the `policy` diff entry are
   computed from the post-preview merge.
2. Rewrite `mergeModelPolicy(stored, engine, change, preview, caller)` keyed by
   `change.ProfileID`:
   - Absent profile and not `Remove`: create `{ID, Name: *Policy.Name, Icon, Engine: engine, Model: change.ModelID}`; refuse without a name ("A new
     profile needs a name."); refuse a new `llamacpp` profile without `Load`
     ("A llama.cpp profile is created with load settings."). Delete the
     `model-<hash16>` naming path.
   - Present profile: refuse when `engine` or `change.ModelID` differs from
     the stored profile.
   - Apply `Name`, `Icon`, `Capabilities`, `Generation`,
     `RequestOptionsMode`.
   - With `Load`: set `profile.Load` to the reviewed `Requested` of the
     preview's `Settings.ModelLoads` entry with this `ProfileID`,
     `profile.ContextTokens` to its `Resolved.PerSlotContext` (P6) and
     `profile.ModelIdentity` to `preview.Identities[ProfileID]` (P8).
   - `Roles`: append or remove this id in each named role's list (P9); drop
     an empty list.
   - `Served`, `Resident`, `Binding`: references are `Role{engine, profile.ServedModel()}`.
   - `Remove`: delete the profile, remove its id from every role list and its
     served reference from `ServedModels`, `ResidentModels`, `EngineBindings`.
   - Keep the final `nodepolicy.Parse`, `ValidateEngineBindings` and, for a
     peer caller, `MergePeerSet` (the `localOnly` lock).
3. `modelPolicyImplications` and `modelReviewMemoryCurrent` select by profile
   id (the profile and its served reference); `modelconfig.go`
   `residentProfile`/`modelConfigEntry` match `pr.ServedModel()`.
4. `policy.go`: `setLocalPolicy` and `setPeerPolicyAs` call
   `nodepolicy.CheckProfileLaunchFields(stored.Profiles, incoming.Profiles)`
   before persisting; `policyErrorCode` maps `*ProfileLaunchFieldsError` to
   `-32602` with its message.
5. Add tests: `TestModelSettingsCreatesProfileWithLoad` (fake worker preview
   with load `ProfileID "p-0123456789abcdef0123"`, per-slot 32768, identities:
   committed policy has that profile with `load`, `contextTokens` 32768,
   `modelIdentity`, and the configure call carries the same load),
   `TestModelSettingsSecondProfileSameModel`, `TestModelSettingsRemoveProfileClearsReferences`,
   `TestPolicySetRefusesLaunchFieldEdit` (local and peer),
   `TestResidentEntryAcceptsDerivedContext` (`modelConfigEntry` is not
   `UnsupportedLoad` when the envelope per-slot equals the derived value).
   Update the listed tests and any other `_test.go` in this module that stops
   compiling.
6. `README.md`: model-settings section describes profile-keyed changes,
   derived `contextTokens`, the launch-field guard on `policy:set`.
**Verify:** From `services/nvpair-ui-broker`: `go vet ./... && go test ./...`
exits 0, and `go test -run 'TestModelSettingsCreatesProfileWithLoad|TestModelSettingsSecondProfileSameModel|TestModelSettingsRemoveProfileClearsReferences|TestPolicySetRefusesLaunchFieldEdit|TestResidentEntryAcceptsDerivedContext' -v ./...`
prints `--- PASS` for each of the five names and no `--- FAIL`.

### Task 4 — Proxy resolves each role to its ordered profiles
**Files:** services/nvpair-proxy/tiers.go, services/nvpair-proxy/discovery.go, services/nvpair-proxy/proxy.go, services/nvpair-proxy/admission.go, services/nvpair-proxy/readiness_contract.go, services/nvpair-proxy/residency.go, services/nvpair-proxy/self.go, services/nvpair-proxy/tiers_test.go, services/nvpair-proxy/hybrid_alias_test.go, services/nvpair-proxy/residency_test.go, services/nvpair-proxy/availability_test.go, services/nvpair-proxy/chat_adapter_test.go, services/nvpair-proxy/engine_binding_candidates_test.go, services/nvpair-proxy/model_controls_test.go, services/nvpair-proxy/reverse_transport_test.go, services/nvpair-proxy/routing_listener_test.go, services/nvpair-proxy/routing_test.go, services/nvpair-proxy/served_models_test.go, services/nvpair-proxy/stop_pause_test.go, services/nvpair-proxy/subscribed_test.go, services/nvpair-proxy/spec.md, services/nvpair-proxy/README.md
**Depends on:** 1 — advertisement and policy shapes
**Complexity:** judgment
**Do:**
1. `proxy.go`: `candidate` gains `profileID string`; the discovery copy of
   `n.Policy.Roles` clones each list.
2. `discovery.go`: `Node.Roles map[string][]nodepolicy.AdvertisedProfile`;
   `filteredModels` keeps, for each `TierOrder` tier, the entries with a valid
   `id` and `servedModel`, drops empty tiers, and keeps the B1 unknown-tier
   marker (smallest tier name with a valid entry) when no known tier survives.
3. `tiers.go`: `declaredRoles` holds lists. `forTier(t)` returns the distinct
   `(engine, servedModel)` pairs ordered by their smallest list position across
   the declaring nodes (self, then peers by node id), then engine, then model.
   In `resolveTierCandidates`, a resolved candidate is kept when its node's
   list for `t` has an entry with that engine and normalized served model that
   passes `metBy`; set `c.profileID` from that entry; the dedup key for role
   candidates is `node NUL engine NUL profileID` (tier-entry candidates keep
   their model key). Leave `tierSearchOrder`, the tier-entry loop, the
   comparator and `reserveRound` unchanged (P12). `hybridLocalCandidates`
   reads the role entry whose id is `c.profileID`.
4. `admission.go` `profileLocked`, `readiness_contract.go`
   `refreshStrictCapabilityLocked`, `residency.go` `residentContextLocked`,
   `self.go`: compare `pr.ServedModel()` where they compare `pr.Model`; keep the
   ambiguity counts.
5. Add tests in `tiers_test.go`: `TestTierRoleTwoProfilesOneModel` (self
   declares `medium: [pSmall (contextTokens 8192), pLarge (32768)]`, both one
   `model`, served ids = their ids, inventory lists both ids: a small request
   yields two candidates in list order with `profileID` set; a request whose
   estimate is 20000 tokens yields only `pLarge`), `TestDiscoveryKeepsRoleLists`
   (lists kept, invalid entries dropped, unknown-tier marker kept),
   `TestHybridCandidateUsesProfileEntry`. Update the listed tests and any other
   `_test.go` in this module that stops compiling.
6. `spec.md`/`README.md`: a declared role is an ordered list of profiles; the
   served model of a profile with a load is its id.
**Verify:** From `services/nvpair-proxy`: `go vet ./... && go test ./...` exits
0, and `go test -run 'TestTierRoleTwoProfilesOneModel|TestDiscoveryKeepsRoleLists|TestHybridCandidateUsesProfileEntry' -v ./...`
prints `--- PASS` for each of the three names and no `--- FAIL`.
`diff <(git show <base>:services/nvpair-proxy/tiers.go | sed -n '/^func tierSearchOrder/,/^}/p') <(sed -n '/^func tierSearchOrder/,/^}/p' services/nvpair-proxy/tiers.go)`
prints nothing, where `<base>` is the branch base commit.

### Task 5 — Desktop editors read and write profile-keyed roles
**Files:** desktop/src/shared/types/node-roles.ts, desktop/src/shared/utils/node-roles.ts, desktop/src/electron/service-bridge/node-projection.ts, desktop/src/ui/stores/node-configuration.store.ts, desktop/src/ui/components/RoutingSettings/RoleEditor.tsx, desktop/src/ui/components/RoutingSettings/BulkNodeConfiguration.tsx, desktop/src/ui/components/NodeList/NodeDeclaredRoles.tsx, desktop/tests/modular/node-configuration.test.ts, desktop/tests/modular/node-configuration-store.test.ts, desktop/tests/modular/node-configuration-bulk.test.ts, desktop/tests/modular/node-roles.test.ts (new)
**Depends on:** 1 — wire shapes the parsers mirror
**Complexity:** judgment
**Do:**
1. Types: keep `RoleTier` (phase 03). Add interface `AdvertisedProfile` with
   fields `id`, `name`, `engine`, `servedModel`, optional `modelIdentity`
   (phase 01's desktop identity type), optional `contextTokens`, optional
   `capabilities`. Exact type lines:
   ```ts
   export type NodeRoles = Partial<Record<RoleTier, AdvertisedProfile[]>>
   export type PolicyRoles = Partial<Record<RoleTier, string[]>>
   ```
   Delete `NodeRole` when nothing uses it after the change.
2. `parseNodeRoles`/`parsePolicyAdvertisement`/`sameNodeRoles` and
   `node-projection.ts` decode lists, drop invalid entries, drop empty tiers.
3. Store: add `profileChoices(policyText)` (the node's profiles with id, name,
   engine, served model, identity); `roleLabel(profile, descriptors)` returns
   the profile name, the engine label in parentheses and phase 01's identity
   tags; `roleKey` is the profile id; `roleItems` lists the profiles not yet in
   the tier; `suggestRoles` maps the first matching local tier entry to the
   profile whose `(engine, servedModel)` equals it; `roleDifferences` compares
   lists in order; delete `inventoryChoices` and `chosenRole` when unused
   (`npm run dead-code:check` decides).
4. `RoleEditor.tsx`: per tier, the ordered list of chosen profiles with a
   Remove button each and a Select "Add profile"; writing the draft sets
   `roles[tier]` to the id list and removes the key when the list is empty,
   through the existing draft write and Save.
5. `BulkNodeConfiguration.tsx`: the same per target, using each target's own
   profiles; a profile id the target lacks is shown as "not on this node" and
   cannot be assigned.
6. `NodeDeclaredRoles.tsx`: one tag `${tier}: ${label}` per profile, in list
   order, with identity tags.
7. Tests: `node-roles.test.ts` covers list parsing and invalid-entry dropping;
   update the three store/bulk tests to lists and profile labels.
**Verify:** From `desktop`: `npm run typecheck` and `npm run lint` exit 0;
`npx vitest run --project unit tests/modular/node-roles.test.ts tests/modular/node-configuration.test.ts tests/modular/node-configuration-store.test.ts tests/modular/node-configuration-bulk.test.ts`
exits 0 with no failed test.

### Task 6 — Desktop model settings editors key on profile id
**Files:** desktop/src/shared/types/model-config.ts, desktop/src/shared/types/node-policy.ts, desktop/src/shared/utils/node-policy-model-settings.ts, desktop/src/shared/utils/node-policy-model-copy.ts, desktop/src/electron/service-bridge/engine-settings.ts, desktop/src/ui/stores/engine-settings.store.ts, desktop/src/ui/components/ModelSettings/ModelSettingsPanel.tsx, desktop/src/ui/components/ModelSettings/ModelProfileControls.tsx, desktop/src/ui/components/ModelSettings/ModelLoadSettings.tsx, desktop/src/ui/components/ModelSettings/ModelSettingsCopy.tsx, desktop/src/ui/components/RoutingSettings/ModelSettingsEditor.tsx, desktop/tests/modular/node-policy-model-settings.test.ts, desktop/tests/modular/node-policy-model-copy.test.ts, desktop/tests/modular/engine-settings-bridge.test.ts, desktop/tests/modular/engine-settings-state.test.ts, desktop/tests/modular/model-config-contract.test.ts, desktop/tests/modular/model-load-settings.test.tsx, desktop/tests/modular/model-settings-apply.test.tsx, desktop/tests/modular/model-settings-auto.test.tsx, desktop/tests/modular/model-settings-copy.test.tsx, desktop/tests/modular/model-buffer-report.test.tsx, desktop/tests/modular/remote-management.test.tsx
**Depends on:** 1, 3, 5 — wire shapes (1), the broker create and remove semantics the editors call (3), the `PolicyRoles` type (5)
**Complexity:** judgment
**Do:**
1. `model-config.ts`: mirror Task 1's wire (`ModelLoad.profileId`,
   `ModelChange.profileId/remove/importPreset`, `PolicyPatch.name/icon/capabilities`,
   `InspectModelRequest.profileId`, preview `identities`).
   `engine-settings.ts` bridge accepts and validates them (apply the
   reserved-id check to `profileId`).
2. `node-policy-model-settings.ts`: `ModelSettingsProfile` gains `id`, `icon`,
   `modelIdentity`, `hasLoad`, `servedModel`; `key` is the id;
   `profileNameValid` follows P5; add `newProfileId()` (P4, via
   `crypto.getRandomValues`); `createModelSettingsProfile` writes `{id, name, engine, model}` for app engines and returns an error for `llamacpp` ("Create
   llama.cpp profiles with load settings."); `removeModelSettingsProfile`
   removes by id, refuses a profile with a load ("Remove it from its load
   settings.") and removes the id from role lists.
3. `node-policy-model-copy.ts`: `modelPolicyPatch` and `portableLoadPatch`
   build changes with `profileId`; role membership reads id lists.
4. `engine-settings.store.ts`: `modelSettingsKey(target, profileId)`.
5. `ModelSettingsPanel.tsx`: selection is a profile id from the node's
   profiles for the engine, plus "New profile" from an installed model.
   `ModelProfileControls.tsx`: for `llamacpp`, "Create profile" asks for a
   name, generates the id and opens `ModelLoadSettings` for it (its first Apply
   sends `{profileId, modelId, policy.name, load}`); "Remove profile" for a
   profile with a load sends `{profileId, remove: true}` through
   preview/apply. App engines keep the draft-based create/remove.
6. `ModelLoadSettings.tsx`: role toggles read and write membership of this
   profile id; show profile names instead of `JSON.stringify` (L1-C13).
   `ModelSettingsEditor.tsx`: key by id. `ModelSettingsCopy.tsx`: compile and
   keep its current copy flow working with `profileId` (deploy behaviour is
   Task 7).
7. Update the listed tests; add a test that `newProfileId()` matches
   `^p-[0-9a-f]{20}$` and that creating a llama.cpp profile sends `name` and
   a load in one change.
**Verify:** From `desktop`: `npm run typecheck`, `npm run lint` and
`npm run dead-code:check` exit 0; `npx vitest run --project unit tests/modular/node-policy-model-settings.test.ts tests/modular/node-policy-model-copy.test.ts tests/modular/engine-settings-bridge.test.ts tests/modular/engine-settings-state.test.ts tests/modular/model-config-contract.test.ts tests/modular/model-load-settings.test.tsx tests/modular/model-settings-apply.test.tsx tests/modular/model-settings-auto.test.tsx tests/modular/model-settings-copy.test.tsx tests/modular/model-buffer-report.test.tsx tests/modular/remote-management.test.tsx`
exits 0 with no failed test.

### Task 7 — A profile deploys to another node
**Files:** desktop/src/shared/utils/node-policy-model-copy.ts, desktop/src/ui/components/ModelSettings/ModelSettingsCopy.tsx, desktop/tests/modular/node-policy-model-copy.test.ts, desktop/tests/modular/model-settings-copy.test.tsx
**Depends on:** 3, 6 — target-side create keeps the id (3); same files, profile-keyed copy (6)
**Complexity:** judgment
**Do:**
1. `node-policy-model-copy.ts`: add `deployProfileChange(source, targetModelId, roles, expectedPolicyRevision)` returning a `ModelChange` with
   `profileId: source.id`, `modelId: targetModelId`, `load.set` = the source
   load reduced by the existing `portableLoadPatch` rules, `policy.name`,
   `policy.icon`, `policy.capabilities`, the generation leaves
   `modelPolicyPatch` copies, `requestOptionsMode` and the selected roles
   (P11). Never `localOnly`, qualification, bounds, resolved values or device
   fields. Add `targetModelFor(contentId, targetInventory)` returning the
   target's native model id whose phase 01 identity has an equal `contentId`.
2. `ModelSettingsCopy.tsx`: rename the action to "Deploy profile". Per
   selected target: when `targetModelFor` finds nothing, start
   `engine:copy-model` from this node's `source.model` to the target through
   the existing `ModelCopyModal` path, wait for its completion in the existing
   progress store, refresh the target inventory and resolve again; when it is
   still absent, show "Copy finished but the target reports a different file."
   and skip that target. Then run the existing target preview, show the diff,
   and apply on confirmation. A broker refusal (for example an existing
   profile with that id on another model) is shown as the target's error.
3. Tests: `deployProfileChange` excludes every non-portable field and keeps
   the id; `targetModelFor` matches by `contentId` only; the component invokes
   `engine:copy-model` before preview when the target lacks the model and
   skips it when the target has it.
**Verify:** From `desktop`: `npm run typecheck` and `npm run lint` exit 0;
`npx vitest run --project unit tests/modular/node-policy-model-copy.test.ts tests/modular/model-settings-copy.test.tsx`
exits 0 with no failed test.

### Task 8 — Hand-written preset sections import into profiles
**Files:** services/nvpair-engine-manager/modelsettings_load.go, services/nvpair-engine-manager/modelsettings_load_test.go, services/nvpair-engine-manager/README.md, services/nvpair-ui-broker/modelsettings.go, services/nvpair-ui-broker/modelsettings_apply_test.go, services/nvpair-ui-broker/README.md, desktop/src/ui/components/ModelSettings/ModelSettingsPanel.tsx, desktop/src/ui/stores/engine-settings.store.ts, desktop/tests/modular/model-settings-apply.test.tsx
**Depends on:** 2, 3, 6 — same engine-manager files (2), same broker files (3), same desktop files (6)
**Complexity:** judgment
**Do:**
1. Engine manager, `previewModelLoadChange` with `ImportPreset`:
   - External `models-preset` and no typed loads: read the file with phase
     01's lenient reader, run `previewTypedLlamaConversion`, set each converted
     load's `ProfileID` to its section name (`ModelID` stays the section
     name). Collect per-section failures: conversion refusals and section names
     that fail `modelconfig.ValidateProfileID` ("Rename this section: profile
     ids use letters, digits and ._:/-"). Any failure refuses the whole import
     with `Errors["models.<section>"]` per failed section (P10). Otherwise
     compile, fill `Identities`, and return the preview with diff and restart
     impact as today's conversion does.
   - Typed loads already present: return the current loads unchanged with
     `Restart` false and `Identities` filled (adoption of loads made before
     this phase).
   - Neither: refuse "No hand-written preset or typed loads to import."
2. Broker, review and commit for `ImportPreset`: for every load in the
   preview, the profile with `id == ProfileID` gets `Load`, `ContextTokens`
   (P6) and `ModelIdentity` (P8); create `{id, name: section name, engine "llamacpp", model: ModelID}` when absent (P3); refuse the import when an
   existing profile with that id has a different `model`, or when the name is
   taken by another id. Role lists, served, resident and binding references
   stay byte-identical. Journal, drain, configure and commit use the existing
   coupled stages. A permitted peer can import (same remote-management
   permission as other model changes).
3. Desktop: `ModelSettingsPanel.tsx` shows "Import as profiles" for a
   llama.cpp engine whose settings snapshot has no `modelLoads` and whose launch
   text contains `--models-preset`, and for one whose typed loads include a
   `profileId` with no policy profile carrying a load; it previews `{importPreset: true, expectedPolicyRevision}` with the existing preview component and
   applies on confirmation. `engine-settings.store.ts` exposes the import
   preview and apply through the existing actions.
4. Tests: engine manager `TestImportPresetSectionsBecomeProfiles` (three
   sections, one using `n-cpu-moe`: three loads keyed by the section names,
   generated preset section names equal the old ones),
   `TestImportPresetRefusesWholeOnBadSection`; broker
   `TestImportCreatesProfilesKeepsRoles` (old-shape policy with roles naming
   two sections: after commit both profiles carry `load` and `contextTokens`
   equal to the reviewed per-slot context, and `roles` serializes to the same
   bytes as before); desktop test that the button appears only in the two
   stated conditions and sends `importPreset: true`.
5. READMEs: document the import and that hand-written presets are not used
   afterwards (D3).
**Verify:** From `services/nvpair-engine-manager`: `go test -run 'TestImportPresetSectionsBecomeProfiles|TestImportPresetRefusesWholeOnBadSection' -v ./...`
prints `--- PASS` for both and no `--- FAIL`; from `services/nvpair-ui-broker`:
`go test -run 'TestImportCreatesProfilesKeepsRoles' -v ./...` prints
`--- PASS`; both modules' `go vet ./... && go test ./...` exit 0. From
`desktop`: `npm run typecheck` exits 0 and `npx vitest run --project unit tests/modular/model-settings-apply.test.tsx`
exits 0.

### Task 9 — Cross-process tests, docs and generated contracts follow profiles
**Files:** services/tests/model_settings_test.go, services/tests/model_settings_fixture_test.go, services/tests/model_routing_interop_test.go, services/tests/linked_models_test.go, services/tests/paused_advertisement_test.go, services/tests/policy_relay_joint_test.go, services/tests/remote_configuration_test.go, services/tests/served_resident_test.go, services/tests/service_policy_test.go, services/tests/unified_dispatch_test.go, services/tests/profile_roles_process_test.go (new), docs/model-configuration.mdx, docs/local-qualification.mdx, desktop/docs/services-api.md
**Depends on:** 1, 2, 3, 4, 5, 6, 7, 8 — exercises and documents all of them
**Complexity:** judgment
**Do:**
1. Update the listed cross-process tests to profile ids, role lists and
   profile-keyed model changes; keep each test's scenario.
2. Add `TestProfileRolesAcrossProcesses` in `profile_roles_process_test.go`
   using the existing fixtures of `model_settings_test.go` and
   `model_routing_interop_test.go`: one node applies two profiles of one
   fixture model with different context through the coupled operation, assigns
   both to `medium`, and a `medium` request from a second node is served with
   `X-PAIR-Model` equal to one of the two profile ids; a request whose estimate
   exceeds the smaller context is served by the larger profile.
3. `docs/model-configuration.mdx`: profiles (id, name, icon, identity,
   load, derived context), several profiles per model, roles as profile
   lists, the import, deploy (copy first when needed), what never copies.
   `docs/local-qualification.mdx`: profile examples gain `id` and `name`;
   `contextTokens` is derived for profiles with a load.
4. From `desktop`: run `npm run service-contracts:write`, then
   `npm run service-contracts:check`; commit `desktop/docs/services-api.md`
   only if the write changed it.
**Verify:** From `services/tests`: `go test -run 'TestProfileRolesAcrossProcesses|TestModelSettings|TestModelControls|TestStrictModelRouting|TestPolicyRelayJoint|TestXARemoteConfiguration|TestServedResident|TestServicePolicy|TestPAPausedAdvertisement|TestUnifiedDispatch|TestLinkedModels' ./...`
exits 0. From `desktop`: `npm run service-contracts:check` exits 0. From the
repo root: `node scripts/spdx-headers.mjs` exits 0.

## Verification (whole-plan)

Local gate (all must pass on the final branch head; keep commands, exit codes
and the last 50 lines of any failing output in the WORKLOG entry):
1. Every Go module compiles against the new shared package:
   `for m in services/*/go.mod; do (cd "$(dirname "$m")" && go build ./... && go vet ./...) || echo "FAIL $m"; done`
   prints no `FAIL` line.
2. `go test ./...` exits 0 in `services/shared`, `services/nvpair-engine-manager`,
   `services/nvpair-ui-broker`, `services/nvpair-proxy`,
   `services/nvpair-node-scanner` and `services/nvpair-tui`.
3. Task 9's `services/tests` subset exits 0.
4. From `desktop`: `npm run lint`, `npm run typecheck`, `npm run test:unit`,
   `npm run dead-code:check`, `npm run service-contracts:check` each exit 0.
5. From the repo root: `node scripts/spdx-headers.mjs` exits 0 and
   `git diff --check <base>..HEAD` prints nothing.
6. `git grep -n -F '"model-" +' -- services/nvpair-ui-broker` prints nothing
   (auto-named `model-<hash16>` profiles are gone; at the base it prints
   `modelsettings.go`).

A test that fails is pre-existing only when the same command fails the same
way on the branch base in a separate scratch worktree; otherwise it blocks the
phase.

Machine acceptance (owner's machines, after the code is installed; record in
the vault `Validation/` folder and WORKLOG):
1. MSI starts on the phase build with its current hand-written preset and
   policy: the broker log has no "node policy invalid" line, the node is not
   paused, and every role names a profile whose id equals the section name the
   role named before the upgrade.
2. From the Mac, open MSI's llama.cpp model settings, run "Import as
   profiles", confirm the preview lists MSI's three preset sections, and apply.
   Afterwards `<appdir>/engines/llamacpp.json` `runtime.modelLoads` has three
   entries whose `profileId` equals the section names, `node-policy.json`
   profiles carry `load` and `contextTokens`, and `roles` is unchanged.
3. On MSI create a second profile of the LFM GGUF with a different context
   (for example 32768 beside the imported one), apply, and assign both to
   `medium` in the role editor. MSI's generated `serving/llamacpp/generation-*/models.ini`
   has two sections with the same `model` path and different `ctx-size`.
4. From the Mac, send a small `medium` chat request: `X-PAIR-Node` names MSI
   and `X-PAIR-Model` equals one of the two profile ids. Send one whose body
   exceeds four times the smaller context: `X-PAIR-Model` equals the larger
   profile's id.
5. The desktop shows both profiles under MSI's `medium` role with the
   repository and quantization tags from phase 01.
6. Deploy the 32768 profile from MSI to Joes `medium`; when Joes lacks the GGUF
   the copy runs first with progress, then the target preview and apply; Joes
   then advertises the same profile id.

## Close-out

- Record verified changes in `plans/2026-10-31-expectation-gaps/00-index.md`:
  append ` — done <sha>` to each task heading in this file as it lands and to
  phase 02's row in the index when the whole-phase local gate passes.
- Add a vault entry to `/home/user/wasabi/4. Projects/Personal-AI-Router/WORKLOG.md`
  with the branch, commit range, gate commands and results, any check that did
  not run (marked untested), the machine acceptance status, and remaining
  limitations: app-engine profiles carry no `modelIdentity`; tier names and
  order are unchanged; the desktop editors are the minimal versions.
- Hand-over: phases 03 (roles) and 04 (desktop) start from the shapes this
  phase lands — `Policy.Roles map[string][]string`, `AdvertisedProfile`,
  `Profile.ServedModel()`, `ModelChange.ProfileID` — and phase 05 keys
  measurements by profile id and `contentId`. Name any deviation from those
  names in the WORKLOG entry so the next phase finds it.

## Do not

- Do not refactor beyond what the tasks specify.
- Do not change tier names, `TierOrder`, `tierSearchOrder`, the tier
  comparator, router tier entries or the routing-history tier enum.
- Do not let the proxy read engine-manager files or RPCs for profiles.
- Do not merge the engine-manager load record into node policy and do not
  remove `requested` from it (P1).
- Do not generate profile ids in Go code; the creating client does (P4).
- Do not copy placement, device bindings, resolved values, paths, `localOnly`,
  qualification, bounds or measurements between nodes.
- Do not add a JSON-RPC method, a capability marker, a schema version bump or
  a compatibility branch; the only imports are the three named in Change
  scope.
- Do not import a hand-written preset partially (P10).
- Do not design the Models or Profiles tabs, the node settings page or Halt.
- Do not edit `desktop/docs/services-api.md` by hand, `services/versions.json`
  or `CHANGELOG.md`.

## If stuck

- Compile errors in a consumer module after Task 1 are expected until that
  module's task lands (P14); do not add shims to make them build early.
- Deterministic test, lint or type failures in files a task owns: fix and
  rerun that task's Verify. Safe to retry any Go or desktop check.
- If phase 01's identity type cannot be imported by `nodepolicy` or
  `enginesettings` without a cycle, stop and report the package path and the
  cycle; do not move phase 01's type in this phase.
- If llama-server refuses two preset sections that point at one GGUF on a
  real machine, stop and report the server version and error; do not fall back
  to native aliases.
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
