<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution plan: Phase 02b: profile editors, deploy and preset import

## Goal

Overall goal of the batch (`README.md` sections 2 and 4): PAIR on the owner's
machines (Mac, MSI, Joes, PC3) is configured through roles that point at named
profiles, each profile being one model identity plus engine plus launch
settings, shown by true identity rather than preset section names, so the
owner's daily loop works without hand-edited files.

This phase's goal (`README.md` section 5 row 02b, owner decisions D2 and D3 as
amended by the 2026-10-31 debate adjudication F1, F3, F4 and F5): the desktop
reads and writes the profile shapes phase 02a landed, so the owner can create a
profile with a key and a display name, duplicate it under a new key, remove it,
declare a role before it has profiles, assign profiles to roles in order,
deploy a profile to another node (copying the model first when the target lacks
it), and import each hand-written llama.cpp preset section on MSI and Joes into
a profile once, including Joes's native-fit section (`fit-target=512`) as it
is. Launch settings are edited through the existing engine-settings relay;
nothing is moved into node policy. User documentation describes profiles,
keys, roles as profile lists, deploy and the import. After this phase the
combined 02a and 02b build is what the owner installs. Tier names stay (phase
03); tabs, the node settings page and Halt are phase 04.

## Change scope

Intent: the owner can manage profiles and their role assignments from the
desktop and bring MSI's and Joes's hand-written presets under PAIR without
changing what clients see.

Scope: Structural change — adapt the desktop's types, bridge parsers, stores
and existing editors (role editor, bulk configuration, declared-role tags,
model settings panel, profile controls, load settings, copy) to profile keys
and role lists; add deploy to the copy flow; add the explicit preset import in
the engine manager, broker and desktop; update user docs. Leave tier names,
desktop navigation and tabs, engine toggles and locks, estimate display and
Validate unchanged.

Basis: owner decisions D2 and D3 (`README.md` section 6); the debate
adjudication F1, F3, F4, F5 (`README.md` section 8,
`investigation/debate.md`); verified evidence L1-C4, C12, C13, L3 Q3, B1, A3,
A4 and the debate's F1 evidence lines.

Compatibility: pre-release fork rule — formats change in place, no capability
markers, no mixed-build handling. The one data import is the hand-written
preset import (Task 4), authorised by D3. Served model ids stay byte-identical:
an imported section's name becomes its profile key and served id.

Non-goals: user-defined role names beyond `weak`, `medium` and `strong`
(phase 03); Models and Profiles tabs, node settings page, Halt, removal of
engine toggles and locks, role-driven activation (phase 04); estimate display
and Validate (phase 05); smart routing and context advertisement (phase 06);
new JSON-RPC methods; any Go change outside the import (Task 4).

## Entry conditions

Phase 02a (`02a-profiles-services.md`) and phase 01 must have landed. Check
all of the following at the branch base before Task 1; record the commands and
outputs in the phase's WORKLOG entry.

1. `plans/2026-10-31-expectation-gaps/00-index.md` rows 01 and 02a end with
   ` — done <sha>`, and `git merge-base --is-ancestor <sha> HEAD` exits 0 for
   both shas.
2. Phase 02a's shapes exist:
   `git grep -n -e 'func (p Policy) ProfileByKey' -e 'func ValidateProfileKey' -e 'func ProfileKeyFromModelID' -e 'type AdvertisedProfile struct' -e 'Roles map\[string\]\[\]string' -- services/shared/nodepolicy`
   prints five lines, and
   `git grep -n -e 'json:"profileKey"' -e 'json:"importPreset' -- services/shared/modelconfig/wire.go`
   prints two.
3. The engine manager still refuses the import:
   `git grep -n 'Preset import is not available yet.' -- services/nvpair-engine-manager`
   prints one line (Task 4 replaces it).
4. Phase 01's native-fit fields and conversion exist:
   `git grep -n -e 'json:"fitTarget' -e 'json:"fitCtx' -- services/shared/modelconfig/wire.go`
   prints two lines, and
   `git grep -n 'TestConvertTypedLlamaNativeFit' -- services/nvpair-engine-manager`
   prints one.
5. The desktop has phase 01's identity tag renderer:
   `git grep -n 'export function modelIdentityTags' -- desktop/src/ui/utils/model-identity-label.ts`
   prints one line and `desktop/src/ui/components/ModelIdentity/ModelIdentityTags.tsx`
   exists.

If any condition is absent: stop before Task 1, write a WORKLOG entry
"phase 02b blocked: entry condition N missing" with the failing command, and
report. Do not implement phase 01 or 02a work in this phase.

## Context the executor needs

Repository and environment:
- Base the branch on current `origin/integration/join33` with phases 01 and
  02a merged. This plan was written against `integration/join33` at
  `4eccd05d` (2026-10-31) plus phase 02a's described interfaces; line numbers
  are navigation aids at that snapshot only. Re-read every named symbol before
  editing it.
- Repo rules: `.cursor/rules/services-change-propagation.mdc`,
  `service-contracts.mdc`, `renderer-stores.mdc`, `type-discipline.mdc`,
  `no-legacy-fallbacks.mdc`, `engines-vs-backends-naming.mdc`,
  `commit-sign-off.mdc` (commit with `git commit -s`). Desktop UI uses Kaizen
  UI (`.agents/skills/kaizen-ui/SKILL.md`). `desktop/docs/services-api.md` is
  generated (`npm run service-contracts:write`), never hand-edited.

Phase 02a shapes (read them in `services/shared/nodepolicy/policy.go`,
`advertise.go` and `services/shared/modelconfig/wire.go`):
- `Profile{key, displayName, icon?, engine, model, modelIdentity?, contextTokens?, capabilities, memoryBytes, maxConcurrent, requestOptions, requestOptionsMode, localOnly, maxOutputTokens, qualification}`, where `model` is the served id and equals `key` for a typed llama.cpp profile;
- `roles: Record<tier, string[]>` (lists may be empty);
- `AdvertisedProfile{key, displayName, icon?, engine, model, modelIdentity?, contextTokens?, capabilities?, ...}`;
- `ModelLoad{profileKey, modelId, requested, resolved}`, where `modelId` is the source model;
- `ModelChange{profileKey, modelId, remove?, importPreset?, load?, policy?, ...}`, `PolicyPatch{displayName?, icon?, capabilities?, roles?, ...}`, `InspectRequest{profileKey?}`, and preview `identities` keyed by profile key.

The
key grammar is `nodepolicy.ValidateProfileKey`: `^[a-z0-9][a-z0-9._-]{0,63}$`,
not ending in `-api` or `-local`, not a reserved alias, unique per node, not a
declared role name, not a model id in the node's inventory other than its own
served id (phase 02a P4). The broker refuses a raw `policy:set` that changes
`contextTokens` or `modelIdentity` of a typed profile (02a P1).

Desktop today (verified at `4eccd05d`, L1-C4, C12, C13, L3 Q3, B1):
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
  `electron/service-bridge/engine-settings.ts` (strict response shapes that
  throw on any unknown key; reserved-id check in `modelId`),
  `ui/stores/engine-settings.store.ts` (`modelSettingsKey(target, modelId)`,
  settings keyed by `${nodeId}:${engine}`), `ui/components/ModelSettings/`
  `ModelSettingsPanel.tsx`, `ModelProfileControls.tsx`, `ModelLoadSettings.tsx`
  (shows `JSON.stringify(routing.roles[tier])`), `ModelSettingsCopy.tsx`
  (already hands off to `ModelManager/ModelCopyModal.tsx` for content copy),
  `RoutingSettings/ModelSettingsEditor.tsx`.
- Peer model copy: JSON-RPC `engine:copy-model` (desktop `ui/api/engine-api.ts`
  `copyModel`, `electron/service-bridge/model-copy.ts`, engine manager
  `modelcopy_control.go` `Manager.runCopy`), vault Framework `Models.md`
  "Copy a model to a peer". Remote load edits go through the engine-settings
  relay (`settings.Request.NodeID`; broker `enginesettings.go`
  `dispatchEngineSettings`, `handleSettingsRelay`).

Import facts (verified at `4eccd05d`, A3, A4, debate F1):
- `previewTypedLlamaConversion` turns each external preset section into a
  `ModelLoad` whose `ModelID` is the section name; since phase 01 Task 8 it
  reads the file with the exact `readLlamaPresetSections`, maps non-typed keys
  to `extraFlags`, maps `fit`, `fit-target` and `fit-ctx` to the native-fit
  fields, and artifact lookup inside it uses the lenient
  `readLlamaArtifactPreset`.
- Joes's configuration per the vault: `n-cpu-moe=38`, `fit-target=512`,
  `load-mode mmap`, native GPU auto-fit kept on
  (`Tasks/2026-10-30 Windows planned-model performance tuning.md:291`, `:323`). MSI launches with
  `--fit off` and uses `mmproj-auto` and `spec-type` keys (debate D3 analysis;
  `serving_llamacpp.go:323`, `:349` mark them unsupported for strict parsing),
  which phase 01 turns into extra flags.

Not verifiable here: every native outcome (llama-server behaviour with two
sections sharing one GGUF, MSI's and Joes's actual preset files and launch
text, peer copy of a multi-GB GGUF). Those are the machine acceptance items.

## Decisions (settled — do not revisit)

- Launch settings stay in the engine manager and are edited through the
  existing engine-settings relay; the desktop never writes a load into node
  policy (debate adjudication F3).
- The profile key is user-chosen, immutable and is the served model id;
  `displayName` is free text; an imported section keeps its section name as
  its key (debate adjudication F4).
- A role may be declared with an empty profile list (debate adjudication F5).
- Import each hand-written preset section into a profile once, after which the
  hand-written preset is not used (owner D3). The import carries Joes's
  sections as they are, including `fit-target`; a native-fit profile has no
  configuration key and no estimate verdict but is routable, importable and
  deployable (debate adjudication F1).
- Copy and deploy never carry hardware placement, device bindings, resolved
  values, paths, locks, qualification or measurement; the target re-resolves
  (2026-10-27 debate).
- Desktop navigation and tabs are phase 04; this phase changes only what the
  existing editors need.
- Pre-release fork: formats change in place; no compatibility work.

## Decisions made while writing this plan

- P10. The implicit external-preset conversion became an explicit import
  (`ImportPreset`) in phase 02a. Import is all-or-nothing: one section that
  cannot be represented refuses the whole import with per-section reasons,
  because a partial import would stop serving the excluded sections. Fit keys
  never trigger that refusal: phase 01 types `fit`, `fit-target` and
  `fit-ctx`, so a Joes section with `fit-target=512` imports as a native-fit
  profile. The remaining per-section refusals are a PAIR-owned key in the
  section (phase 01 Task 8's list) and a section name that fails
  `nodepolicy.ValidateProfileKey` ("Rename this section: profile keys use
  lower-case letters, digits, '.', '_' and '-'.").
- P11. Deploy is orchestrated by the desktop (`ModelSettingsCopy`) over
  existing RPCs: `engine:copy-model` when the target lacks a model with the
  same `contentId`, then the target's model-settings preview and apply through
  the existing engine-settings relay. No new JSON-RPC method. Copied: key,
  display name, icon, portable load fields (`portableLoadPatch` rules: for fit
  that is `fit` and `fitCtx`, never `fitTarget`), generation request options as
  `modelPolicyPatch` copies them, `requestOptionsMode`, capabilities and the
  selected role memberships. The target's own broker applies P4's key checks;
  a collision is shown as that target's error.
- P15. Key entry in the desktop: "Create profile" proposes a key built from
  the model identity and context (repository name and quantization, lowercased,
  plus `-<n>k` for the per-slot context, slugged with the same rule as
  `ProfileKeyFromModelID`, for example `lfm2.5-24b-a2b-gguf-ud-q4_k_xl-64k`),
  which the owner can edit before saving. The desktop mirrors
  `ValidateProfileKey` in `profileKeyError(key)` and also refuses a key already
  used on the node, a declared role name and a model id in the node's
  inventory; the broker stays authoritative. "Duplicate as" creates a new
  profile under a new key with the source profile's load and policy fields
  (the way to rename a key); "Rename" edits only `displayName` and `icon`.
- P16. Role editors declare roles with possibly empty lists: removing the last
  profile from a role leaves it declared with `[]`; a "Remove role" control
  deletes the key. Until phase 03 only `weak`, `medium` and `strong` can be
  declared.
- P17. Packaging: five tasks. Tasks 1 and 2 adapt parsers and editors (the
  bridge rejects unknown keys, so each parser change rides with its editor
  change); Task 3 adds deploy; Task 4 adds the import across engine manager,
  broker and desktop; Task 5 updates user docs. Machine acceptance covers the
  combined 02a and 02b build.

## Tasks

### Task 1 — Desktop role parsers and editors use profile-key lists
**Files:** desktop/src/shared/types/node-roles.ts, desktop/src/shared/utils/node-roles.ts, desktop/src/electron/service-bridge/node-projection.ts, desktop/src/ui/stores/node-configuration.store.ts, desktop/src/ui/components/RoutingSettings/RoleEditor.tsx, desktop/src/ui/components/RoutingSettings/BulkNodeConfiguration.tsx, desktop/src/ui/components/NodeList/NodeDeclaredRoles.tsx, desktop/tests/modular/node-configuration.test.ts, desktop/tests/modular/node-configuration-store.test.ts, desktop/tests/modular/node-configuration-bulk.test.ts, desktop/tests/modular/node-roles.test.ts (new)
**Depends on:** none — the phase entry conditions must hold
**Complexity:** judgment
**Do:**
1. Types: keep `RoleTier` (phase 03). Add interface `AdvertisedProfile` with fields `key`, `displayName`, optional `icon`, `engine`, `model`, optional `modelIdentity` (phase 01's desktop identity type), optional `contextTokens`, optional `capabilities`. Exact type lines:
   ```ts
   export type NodeRoles = Partial<Record<RoleTier, AdvertisedProfile[]>>
   export type PolicyRoles = Partial<Record<RoleTier, string[]>>
   ```
   Delete `NodeRole` when nothing uses it after the change.
2. `parseNodeRoles`/`parsePolicyAdvertisement`/`sameNodeRoles` and `node-projection.ts` decode lists, drop invalid entries and keep a declared role whose list is empty (P16).
3. Store: add `profileChoices(policyText)` (the node's profiles with key, display name, engine, served model, identity); `roleLabel(profile, descriptors)` (replacing phase 01's `roleLabel(role, descriptors, identity)`) returns the display name, the engine label in parentheses and phase 01's identity tags (`modelIdentityTags`, rendered with `ModelIdentityTags`); `roleKey` is the profile key; `roleItems` lists the profiles not yet in the tier; `suggestRoles` maps the first matching local tier entry to the profile whose `(engine, model)` equals it; `roleDifferences` compares lists in order; delete `inventoryChoices` and `chosenRole` when unused (`npm run dead-code:check` decides).
4. `RoleEditor.tsx`: per tier, the ordered list of chosen profiles with a Remove button each, a Select "Add profile", and a "Declare role" button for an undeclared tier and "Remove role" for a declared one; writing the draft sets `roles[tier]` to the key list (an empty list stays, P16; "Remove role" deletes the key), through the existing draft write and Save.
5. `BulkNodeConfiguration.tsx`: the same per target, using each target's own profiles; a profile key the target lacks is shown as "not on this node" and cannot be assigned.
6. `NodeDeclaredRoles.tsx`: one tag `${tier}: ${label}` per profile, in list order, with identity tags; a declared empty role shows `${tier}: no profiles`.
7. Tests: `node-roles.test.ts` covers list parsing, invalid-entry dropping and an empty declared role kept; update the three store/bulk tests to lists and profile labels.
**Verify:** From `desktop`: `npm run typecheck` and `npm run lint` exit 0;
`npx vitest run --project unit tests/modular/node-roles.test.ts tests/modular/node-configuration.test.ts tests/modular/node-configuration-store.test.ts tests/modular/node-configuration-bulk.test.ts`
exits 0 with no failed test.

### Task 2 — Desktop model settings editors key on the profile key
**Files:** desktop/src/shared/types/model-config.ts, desktop/src/shared/types/node-policy.ts, desktop/src/shared/utils/node-policy-model-settings.ts, desktop/src/shared/utils/node-policy-model-copy.ts, desktop/src/electron/service-bridge/engine-settings.ts, desktop/src/ui/stores/engine-settings.store.ts, desktop/src/ui/components/ModelSettings/ModelSettingsPanel.tsx, desktop/src/ui/components/ModelSettings/ModelProfileControls.tsx, desktop/src/ui/components/ModelSettings/ModelLoadSettings.tsx, desktop/src/ui/components/ModelSettings/ModelSettingsCopy.tsx, desktop/src/ui/components/RoutingSettings/ModelSettingsEditor.tsx, desktop/tests/modular/node-policy-model-settings.test.ts, desktop/tests/modular/node-policy-model-copy.test.ts, desktop/tests/modular/engine-settings-bridge.test.ts, desktop/tests/modular/engine-settings-state.test.ts, desktop/tests/modular/model-config-contract.test.ts, desktop/tests/modular/model-load-settings.test.tsx, desktop/tests/modular/model-settings-apply.test.tsx, desktop/tests/modular/model-settings-auto.test.tsx, desktop/tests/modular/model-settings-copy.test.tsx, desktop/tests/modular/model-buffer-report.test.tsx, desktop/tests/modular/remote-management.test.tsx
**Depends on:** 1 — uses the `PolicyRoles` type and `profileChoices`
**Complexity:** judgment
**Do:**
1. `model-config.ts`: mirror phase 02a's wire (`ModelLoad.profileKey`, `ModelChange.profileKey/remove/importPreset`, `PolicyPatch.displayName/icon/capabilities`, `InspectModelRequest.profileKey`, preview `identities`). `engine-settings.ts` bridge accepts and validates them (add each key to its `shape` lists; apply `profileKeyError` to `profileKey`).
2. `node-policy-model-settings.ts`: `ModelSettingsProfile` gains `key`, `displayName`, `icon`, `modelIdentity`, `typed` (the node's engine settings hold a load with this key); `key` replaces the old name-based key; replace `profileNameValid` with `displayNameValid` (02a P5) and add `profileKeyError(key)` and `proposeProfileKey(identity, perSlotContext)` (P15); `createModelSettingsProfile` writes `{key, displayName, engine, model}` for app engines and returns an error for `llamacpp` ("Create llama.cpp profiles with load settings."); `removeModelSettingsProfile` removes by key, refuses a typed profile ("Remove it from its load settings.") and removes the key from role lists, leaving those roles declared.
3. `node-policy-model-copy.ts`: `modelPolicyPatch` and `portableLoadPatch` build changes with `profileKey`; role membership reads key lists.
4. `engine-settings.store.ts`: `modelSettingsKey(target, profileKey)`; load settings for a remote node are read and applied through the existing `${nodeId}:${engine}` entries and the relay (no new store).
5. `ModelSettingsPanel.tsx`: selection is a profile key from the node's profiles for the engine, plus "New profile" from an installed model. `ModelProfileControls.tsx`: for `llamacpp`, "Create profile" asks for a display name, a key prefilled by `proposeProfileKey` and an optional icon (text, 02a P5), validates the key (P15) and opens `ModelLoadSettings` for it (its first Apply sends `{profileKey, modelId, policy.displayName, policy.icon, load}`); "Duplicate as" asks for a new key and display name and sends one change with the source load and policy fields under the new key; "Rename" edits `displayName` and `icon` through a policy-only change; "Remove profile" for a typed profile sends `{profileKey, remove: true}` through preview/apply. App engines keep the draft-based create/remove.
6. `ModelLoadSettings.tsx`: role toggles read and write membership of this profile key; show profile display names instead of `JSON.stringify` (L1-C13); the native-fit fields from phase 01 Task 9 stay as they are. `ModelSettingsEditor.tsx`: key by profile key. `ModelSettingsCopy.tsx`: compile and keep its current copy flow working with `profileKey` (deploy behaviour is Task 3).
7. Update the listed tests; add tests that `proposeProfileKey` for repository `unsloth/LFM2.5-24B-A2B-GGUF`, quant `UD-Q4_K_XL` and 65536 yields `lfm2.5-24b-a2b-gguf-ud-q4_k_xl-64k`, that `profileKeyError` agrees with the Go grammar cases of phase 02a's `TestValidateProfileKey`, that creating a llama.cpp profile sends `displayName` and a load in one change, and that "Duplicate as" sends a change with the new key and the source load.
**Verify:** From `desktop`: `npm run typecheck`, `npm run lint` and
`npm run dead-code:check` exit 0; `npx vitest run --project unit tests/modular/node-policy-model-settings.test.ts tests/modular/node-policy-model-copy.test.ts tests/modular/engine-settings-bridge.test.ts tests/modular/engine-settings-state.test.ts tests/modular/model-config-contract.test.ts tests/modular/model-load-settings.test.tsx tests/modular/model-settings-apply.test.tsx tests/modular/model-settings-auto.test.tsx tests/modular/model-settings-copy.test.tsx tests/modular/model-buffer-report.test.tsx tests/modular/remote-management.test.tsx`
exits 0 with no failed test. `git grep -n -e newProfileId -e profileNameValid -- desktop/src` prints nothing.

### Task 3 — A profile deploys to another node
**Files:** desktop/src/shared/utils/node-policy-model-copy.ts, desktop/src/ui/components/ModelSettings/ModelSettingsCopy.tsx, desktop/tests/modular/node-policy-model-copy.test.ts, desktop/tests/modular/model-settings-copy.test.tsx
**Depends on:** 2 — same files, profile-keyed copy
**Complexity:** judgment
**Do:**
1. `node-policy-model-copy.ts`: add `deployProfileChange(source, sourceLoad, targetModelId, roles, expectedPolicyRevision)` returning a `ModelChange` with `profileKey: source.key`, `modelId: targetModelId`, `load.set` = the source profile's load (read from the source node's engine settings through the relay) reduced by the existing `portableLoadPatch` rules, `policy.displayName`, `policy.icon`, `policy.capabilities`, the generation leaves `modelPolicyPatch` copies, `requestOptionsMode` and the selected roles (P11). Never `localOnly`, qualification, bounds, resolved values, device fields or `fitTarget`. Add `targetModelFor(contentId, targetInventory)` returning the target's native model id whose phase 01 identity has an equal `contentId`.
2. `ModelSettingsCopy.tsx`: rename the action to "Deploy profile". Per selected target: when `targetModelFor` finds nothing, start `engine:copy-model` from this node's source model (`ModelLoad.modelId` of the profile's load) to the target through the existing `ModelCopyModal` path, wait for its completion in the existing progress store, refresh the target inventory and resolve again; when it is still absent, show "Copy finished but the target reports a different file." and skip that target. Then run the existing target preview through the relay, show the diff, and apply on confirmation. A broker refusal (for example the key already names another model or a model id on that node) is shown as the target's error.
3. Tests: `deployProfileChange` excludes every non-portable field (including `fitTarget`) and keeps the key; `targetModelFor` matches by `contentId` only; the component invokes `engine:copy-model` before preview when the target lacks the model and skips it when the target has it.
**Verify:** From `desktop`: `npm run typecheck` and `npm run lint` exit 0;
`npx vitest run --project unit tests/modular/node-policy-model-copy.test.ts tests/modular/model-settings-copy.test.tsx`
exits 0 with no failed test.

### Task 4 — Hand-written preset sections import into profiles
**Files:** services/nvpair-engine-manager/modelsettings_load.go, services/nvpair-engine-manager/modelsettings_load_test.go, services/nvpair-engine-manager/README.md, services/nvpair-ui-broker/modelsettings.go, services/nvpair-ui-broker/modelsettings_apply_test.go, services/nvpair-ui-broker/README.md, desktop/src/ui/components/ModelSettings/ModelSettingsPanel.tsx, desktop/src/ui/stores/engine-settings.store.ts, desktop/tests/modular/model-settings-apply.test.tsx, services/tests/profile_import_process_test.go (new)
**Depends on:** 2 — same desktop files (`ModelSettingsPanel.tsx`, `engine-settings.store.ts`, `model-settings-apply.test.tsx`)
**Complexity:** judgment
**Do:**
1. Engine manager, `previewModelLoadChange` with `ImportPreset` (replace phase 02a's "Preset import is not available yet." refusal):
   - External `models-preset` and no typed loads: run `previewTypedLlamaConversion` and set each converted load's `ProfileKey` to its section name (`ModelID` stays the section name). Collect per-section failures (P10): conversion refusals and section names that fail `nodepolicy.ValidateProfileKey`. `fit`, `fit-target` and `fit-ctx` are converted by phase 01 and never fail a section. Any failure refuses the whole import with `Errors["models.<section>"]` per failed section. Otherwise compile, fill `Identities`, and return the preview with diff and restart impact as today's conversion does.
   - Typed loads already present: return the current loads unchanged with `Restart` false and `Identities` filled (adoption of loads whose policy profile is missing, phase 02a P1 and P3).
   - Neither: refuse "No hand-written preset or typed loads to import."
2. Broker, review and commit for `ImportPreset`: for every load in the preview, the profile with `key == ProfileKey` gets `Model = ProfileKey`, `ContextTokens` (02a P6) and `ModelIdentity` (02a P8); create `{key, displayName: section name, engine "llamacpp", model: key}` when absent; when a policy profile exists for the same served id under a different key (02a P3's changed-served-id case), rewrite that profile's key and role references to the load's key. Refuse the import when an existing profile with that key has a different `model`. Role lists, served, resident and binding references otherwise stay byte-identical. Journal, drain, configure and commit use the existing coupled stages. A permitted peer can import (same remote-management permission as other model changes).
3. Desktop: `ModelSettingsPanel.tsx` shows "Import as profiles" for a llama.cpp engine whose settings snapshot has no `modelLoads` and whose launch text contains `--models-preset`, and for one whose typed loads include a `profileKey` with no policy profile; it previews `{importPreset: true, expectedPolicyRevision}` with the existing preview component and applies on confirmation. `engine-settings.store.ts` exposes the import preview and apply through the existing actions.
4. Tests: engine manager `TestImportPresetSectionsBecomeProfiles` (three sections, one with `n-cpu-moe = 38`, `fit = on`, `fit-target = 512` and `load-mode = mmap`: three loads keyed by the section names, the native-fit one with `fit auto`, `fitTarget "512"` and no refusal; generated preset section names equal the old ones), `TestImportPresetRefusesWholeOnBadSection` (a section with `alias = x` and one named `Upper` each produce a per-section error and no load changes); broker `TestImportCreatesProfilesKeepsRoles` (old-shape policy with roles naming two sections: after commit both profiles carry `contextTokens` equal to the reviewed per-slot context and `modelIdentity`, no `load` member, and `roles` serializes to the same bytes as before); desktop test that the button appears only in the two stated conditions and sends `importPreset: true`; `services/tests/profile_import_process_test.go` `TestPresetImportAcrossProcesses` (a fixture engine with a two-section hand-written preset, one native-fit, is imported from a second node through the relay; both sections stay served under their names and the native-fit profile inspects with reason `native-fit`).
5. READMEs: document the import, that it accepts native-fit sections, and that hand-written presets are not used afterwards (D3).
**Verify:** From `services/nvpair-engine-manager`: `go test -run 'TestImportPresetSectionsBecomeProfiles|TestImportPresetRefusesWholeOnBadSection' -v ./...`
prints `--- PASS` for both and no `--- FAIL`; from `services/nvpair-ui-broker`:
`go test -run 'TestImportCreatesProfilesKeepsRoles' -v ./...` prints
`--- PASS`; both modules' `go vet ./... && go test ./...` exit 0; from
`services/tests`: `go test -run 'TestPresetImportAcrossProcesses' ./...` exits 0.
From `desktop`: `npm run typecheck` exits 0 and `npx vitest run --project unit tests/modular/model-settings-apply.test.tsx`
exits 0. `git grep -n 'Preset import is not available yet.' -- services` prints nothing.

### Task 5 — User docs and the generated contract describe profiles
**Files:** docs/model-configuration.mdx, docs/local-qualification.mdx, desktop/docs/services-api.md
**Depends on:** 3, 4 — documents deploy and the import
**Complexity:** mechanical
**Do:**
1. `docs/model-configuration.mdx`: profiles (key, display name, icon, identity, load settings held by the engine manager under the same key, derived context), the key rules and "Duplicate as" for renaming a key, several profiles per model, roles as ordered and possibly empty profile lists, the import (including native-fit sections, which have no configuration key or estimate verdict but are routable and deployable), deploy (copy first when needed), what never copies, and that a raw policy edit cannot change `contextTokens` or `modelIdentity` of a typed profile.
2. `docs/local-qualification.mdx`: profile examples gain `key` and `displayName` and lose `name`; `contextTokens` is derived for profiles with a load.
3. From `desktop`: run `npm run service-contracts:write`, then `npm run service-contracts:check`; commit `desktop/docs/services-api.md` only if the write changed it.
**Verify:** `git grep -n '"quantization"' -- docs/local-qualification.mdx` prints nothing and `git grep -n '"displayName"' -- docs/local-qualification.mdx` prints at least one line; from `desktop`: `npm run service-contracts:check` exits 0; from the repo root: `node scripts/spdx-headers.mjs` exits 0.

## Verification (whole-plan)

Local gate (all must pass on the final branch head; keep commands, exit codes
and the last 50 lines of any failing output in the WORKLOG entry):
1. `go vet ./... && go test ./...` exits 0 in `services/nvpair-engine-manager`
   and `services/nvpair-ui-broker`; `go build ./...` exits 0 in every Go
   module (`for m in services/*/go.mod; do (cd "$(dirname "$m")" && go build ./...) || echo "FAIL $m"; done` prints no `FAIL`).
2. From `services/tests`: `go test -run 'TestPresetImportAcrossProcesses|TestProfileRolesAcrossProcesses|TestRemoteProfileCreateThroughRelay|TestModelSettings' ./...` exits 0.
3. From `desktop`: `npm run lint`, `npm run typecheck`, `npm run test:unit`,
   `npm run dead-code:check`, `npm run service-contracts:check` each exit 0.
4. From the repo root: `node scripts/spdx-headers.mjs` exits 0,
   `git diff --check <base>..HEAD` prints nothing and `make check` exits 0.

A test that fails is pre-existing only when the same command fails the same
way on the branch base in a separate scratch worktree; otherwise it blocks the
phase.

Machine acceptance (owner's machines, after the combined 02a and 02b build is
installed; record in the vault `Validation/` folder and WORKLOG):
1. MSI starts on the build with its current hand-written preset and policy: the
   broker log has no "node policy invalid" line, the node is not paused, and
   every role names a profile whose key equals the section name the role named
   before the upgrade.
2. From the Mac, open MSI's llama.cpp model settings, run "Import as
   profiles", confirm the preview lists MSI's preset sections, and apply.
   Afterwards `<appdir>/engines/llamacpp.json` `runtime.modelLoads` has one
   entry per section whose `profileKey` equals the section name,
   `node-policy.json` profiles carry `contextTokens` and `modelIdentity` and no
   `load`, and `roles` is unchanged.
3. Joes: run "Import as profiles" from the Mac. The import is not refused; the
   MoE profile's load settings show `fit` auto, `fitTarget` 512 and `nCpuMoe`
   38; inspection reports `native-fit`; a request to that profile succeeds.
4. On MSI create a second profile of the LFM GGUF with key `lfm-32k` and a
   32768 context beside the imported one, apply, and assign both to `medium` in
   the role editor. MSI's generated `serving/llamacpp/generation-*/models.ini`
   has two sections with the same `model` path and different `ctx-size`.
5. From the Mac, send a small `medium` chat request: `X-PAIR-Node` names MSI
   and `X-PAIR-Model` equals one of the two keys. Send one whose body exceeds
   four times the smaller context: `X-PAIR-Model` is `lfm-32k`.
6. The desktop shows both profiles under MSI's `medium` role with the
   repository and quantization tags from phase 01.
7. Deploy `lfm-32k` from MSI to Joes `medium`; when Joes lacks the GGUF the
   copy runs first with progress, then the target preview and apply; Joes then
   advertises `lfm-32k`.
8. On the Mac, declare `strong` with no profiles and save: the Mac's
   advertisement lists `strong` with an empty list and `strong` requests are
   not routed to the Mac.

## Close-out

- Record verified changes in `plans/2026-10-31-expectation-gaps/00-index.md`:
  append ` — done <sha>` to each task heading in this file as it lands and to
  the phase 02b row in the index when the whole-phase local gate passes; when
  machine acceptance has not run, annotate the row
  ` — local gate done <sha>; machine acceptance pending` instead.
- Add a vault entry to `/home/user/wasabi/4. Projects/Personal-AI-Router/WORKLOG.md`
  with the branch, commit range, gate commands and results, any check that did
  not run (marked untested), the machine acceptance status, and remaining
  limitations: app-engine profiles carry no `modelIdentity`; tier names and
  order are unchanged; the desktop editors are the minimal versions; native-fit
  profiles have no estimate verdict or memory record.
- Hand-over: phases 03 (roles) and 04 (desktop) start from these shapes —
  `Policy.Roles map[string][]string` (possibly empty lists),
  `AdvertisedProfile`, `Profile.Key`, `ModelChange.ProfileKey`,
  `deployProfileChange`, `targetModelFor`, `profileKeyError` — and phase 05
  keys measurements by profile key and `contentId`. Name any deviation from
  those names in the WORKLOG entry so the next phase finds it.

## Do not

- Do not refactor beyond what the tasks specify.
- Do not write a load into node policy or read launch settings from policy;
  use the engine-settings relay.
- Do not change tier names, `TierOrder`, `tierSearchOrder`, the tier
  comparator, router tier entries or the routing-history tier enum.
- Do not copy placement, device bindings, resolved values, paths, `localOnly`,
  qualification, bounds, measurements or `fitTarget` between nodes.
- Do not add a JSON-RPC method, a capability marker, a schema version bump or
  a compatibility branch.
- Do not import a hand-written preset partially (P10), and do not refuse a
  section because of `fit`, `fit-target` or `fit-ctx`.
- Do not design the Models or Profiles tabs, the node settings page or Halt.
- Do not edit `desktop/docs/services-api.md` by hand, `services/versions.json`
  or `CHANGELOG.md`.

## If stuck

- Deterministic test, lint or type failures in files a task owns: fix and
  rerun that task's Verify. Safe to retry any Go or desktop check.
- If `engine-settings.ts` rejects a response after a change, the field is
  missing from a `shape` list; add it in the same commit rather than relaxing
  `shape`.
- If llama-server refuses two preset sections that point at one GGUF on a
  real machine, stop and report the server version and error; do not fall back
  to native aliases.
- If a Joes section fails the import for a reason other than P10's two, stop
  and report the section's redacted keys; do not drop the section from the
  import.
- Stop and report when a change would need a Go change outside Task 4's files
  or would alter phase 02a's shapes.

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
