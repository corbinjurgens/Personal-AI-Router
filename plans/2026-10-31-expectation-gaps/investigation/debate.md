<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Debate: PAIR expectation-gaps batch (README sections 2-7, 00-index, phases 01 and 02)

Target: `/home/user/Personal-AI-Router/plans/2026-10-31-expectation-gaps/README.md` sections 2-7, `00-index.md`, `01-identity-and-unblock.md`, `02-profiles.md` (03-06 read for context only; their wording is in flux and is not assessed here).
Requirement: `/home/user/wasabi/4. Projects/Personal-AI-Router/Expectation Drift.md` (owner statement) and `Tasks/2026-10-31 PAIR expectation gaps sweep.md`.
Declared scope and basis (quoted): "Structural change: establish a Profile as the organising object joining node policy and engine-manager launch settings; user-defined roles; node-centric desktop; visible estimate and measured validate; speed-aware routing on content identity. Basis: the owner's 2026-10-31 expectation statement; the fork is pre-release with no compatibility obligations; every machine runs the same build." Work kind: feature (restructuring).
Revision for factual premises: read-only worktree `/home/user/pair-join33` at `integration/join33` `4eccd05d` (confirmed with `git rev-parse`, clean tree). Environment: source reading and `git grep`; nothing built or run; no owner machine touched.
Design-first target: no factual claim was supplied, so no claim is numbered and nothing factual is certified. Every objection below names its factual premise and the evidence checked at `4eccd05d`, or is marked `unverified` with the blocker.

## Findings

### F1. REQUIRED CORRECTION — D3's one-time import cannot carry Joes's measured configuration; after phase 02 Joes is either stranded in the mode the batch retires or silently re-placed

Consequence (behaviour, data): the owner's fastest node (Joes, about 79 and 279 tok/s per the sweep register) runs a configuration the vault records as `n-cpu-moe=38`, `fit-target=512`, `load-mode mmap`, with native GPU auto-fit kept on. Under the design, typed llama.cpp loads always emit `fit = off` and refuse `fit-target`; the import is all-or-nothing (P10), and phase 01 Task 8 step 5 refuses PAIR-owned keys (`fit` is in that list) on conversion. So D3's import of Joes refuses as a whole, and until it succeeds every other model change on that engine is refused with "Import this engine's presets first." (02 P10, Task 2 step 4). Joes stays serving but becomes uneditable through PAIR, and the batch's identity, estimate and Validate work never reaches it. If the owner instead rewrites the section to drop `fit-target`, the placement that was tuned by sweeps r1-r15 changes without a measurement.

Factual premises and evidence:
- Typed mode forces `fit = off`: `services/nvpair-engine-manager/modelsettings_load.go:861` and `:1425` (`"fit": "off"` in the generated option map); `services/nvpair-engine-manager/serving_llamacpp.go:927` writes `fit = off` into the `[*]` snapshot defaults.
- Conversion refuses any `fit` other than `off`: `modelsettings_load.go:1654-1657` (`case "fit": if value != "off" { return fail() }`); `fit-target` is `unsupported: true` in the option table: `serving_llamacpp.go:365`.
- Phase 01 Task 8 step 5 (`01-identity-and-unblock.md:339`): "keep today's handling of typed, `fit`, `load-on-startup`, `kv-unified` ...; refuse PAIR-owned keys ... 'PAIR owns this option; remove it from the preset before converting.'"; step 3 lists `fit` as PAIR-owned (line 337 area, descriptor "manager/router" at `modelsettings_load.go:1824-1829`).
- Phase 02 P10 (`02-profiles.md`, Decisions made while writing): import "is all-or-nothing: one section that cannot be represented refuses the whole import"; Task 2 step 4 refuses every change on an engine with an external preset and no typed loads.
- Joes's configuration: vault `Tasks/2026-10-30 Windows planned-model performance tuning.md:291` ("changed C1 to `n-cpu-moe=38` with `fit-target=512`"), `:301`, `:323` ("Joes ... kept native GPU auto-fit with a 1 GiB margin, and serving mode would refuse non-off `fit` and `fit-target`"); `Archive/Todo/2026-10-31-pair-moe-placement-working-set.md:246-248, :300` ("The Joes configuration stays in unconfigured mode, where these already reach native"). `unverified` as to Joes's file on disk today: the preset was not read from the machine; the premise rests on these vault records.
- The design's own text is silent on `fit`: README T-P/T-C and D3 do not mention it; phase 02 contains no occurrence of `fit` (grep).

What an alternative must meet: the owner's "Validate" is a measured run ("test runs the model with those settings ... and record those results", owner statement), which does not need PAIR-resolved finite allocations. Either (a) type `fit` and `fit-target` as profile load fields (the configuration key then hashes them, and the buffer capture already records what native allocated), relaxing the serving envelope's finite-allocation rule for profiles the owner marks as native-fit, or (b) make D3 explicitly say Joes stays in unconfigured mode and that phase 02's "Import first" refusal does not apply to engines the owner leaves unconfigured. (a) is the only one that meets whole-plan acceptance item 5 (measure a profile) on Joes. This is a scope decision for the owner; the plan should not assume it.

### F2. REQUIRED CORRECTION — role-driven activation is built on the one path that enforces one running engine, contradicting the owner's "multiple engines can be activated when needed"

Consequence (behaviour): the owner wrote "if the system has enough resource, then depending on how the roles are set then multiple engines can be activated when needed." README T-U's fix is "reusing the binding path's activation, generalised from 'engine binding' to 'profile needs engine'". That path refuses while any other engine runs, or switches the one running engine away. A node with a llama.cpp profile on `weak` and a Strata (or Ollama) profile on `medium` would therefore never run both; each request for the other role triggers a switch, with its dwell cooldown, instead of a second engine. D4's default text does not say this, so the owner approves "role-driven activation" without being told it is exclusive.

Factual premises and evidence:
- Broker `runBoundEngine`: `services/nvpair-ui-broker/bound_engine.go:378` with `engine-switch-busy` refusals at `:95`, `:269`, `:399`; verify-A A6 (confirmed): "`len(running) > 1` -> engine-switch-busy; `len(running) == 0` -> cold activation; exactly one -> switch away".
- Engine-manager guard: `services/nvpair-engine-manager/bound_engine.go:39, :65, :173` return `switchBusyError` while any other engine is running, stopping or reclaiming; `switchexecutor.go:1156` "automatic replacement requires a sole conflicting source".
- The non-exclusive paths (`Wake`, `RestoreEnabled`) start only saved-On engines: `services/nvpair-engine-manager/wakesleep.go:57-63` (`errIntentOff`); `services/nvpair-ui-broker/idlepolicy.go:183-188` ("not starting engine on demand: it is saved Off").
- Residency cap across engines defaults to 1: `services/shared/nodepolicy/policy.go:350-352, :596` (`MaxResidentModels: 1`), so even co-running engines would not hold two models without a policy change the design does not name.
- README T-U and D4 (section 6) name no exclusivity decision; phase 04's goal line "engines start when a profile needs them" (04-desktop.md:29) inherits the same path (04 wording in flux; the dependency on the bound path is also stated in README T-U, which is in scope).

What an alternative must meet: either an explicit owner decision "one engine at a time per node stays" recorded in section 6, or an activation path that starts a second engine when the resident/memory policy allows it. The smaller mechanism that exists today is the non-exclusive `dormantSelfCandidate` + `wakeOnDemand` path (`services/nvpair-proxy/self.go:80-110`, which already accepts profile models at `:105-107`), which is gated only by saved intent and `Idle.StartOnDemand` (`policy.go:383`): making "assigned to a role" imply intent On (or dropping the intent check for role profiles) gives role-driven, non-exclusive activation without touching the switch machinery, and leaves `engineBindings` for the exclusive case.

### F3. RECOMMENDED CHANGE — moving `RequestedLoad` into node policy creates two authoritative copies and a new write guard, reverses the 2026-10-27 debate split, and gains nothing the existing relay does not already provide

Consequence (ownership, data): phase 02 P1 keeps `ModelLoad.requested` in the engine manager "as the applied copy" and makes `Profile.load` in policy "the authoritative intent", then adds `nodepolicy.CheckProfileLaunchFields` so raw `policy:set` (local and peer, including the desktop's Advanced JSON editor) cannot touch `load`, `model`, `engine`, `modelIdentity` or `contextTokens`. The owner gets two records that must agree, a new error class (`*ProfileLaunchFieldsError`, `-32602`), and a policy document that can no longer be edited as a whole. The README's D2 rationale ("policy is the remotely edited and advertised store") is answered by facilities that already exist.

Factual premises and evidence:
- The 2026-10-27 debate concluded "Keep per-model llama load settings with engine settings (engine-manager owner), request/routing settings in node policy; one composed editor and a broker-coordinated apply, not one merged record" (vault `Investigations/2026-10-27-model-config-gui.md:19`). Phase 02 "Decisions (settled)" says it keeps that split, while P1 moves the intent to policy; README section 4 lists `load (typed RequestedLoad ...)` inside the policy profile.
- Remote editing of engine-manager loads already exists: `settings.Request.NodeID` (`services/shared/enginesettings/settings.go:74`), the broker's peer relay `handleSettingsRelay` (`services/nvpair-ui-broker/enginesettings.go:845`) and `dispatchEngineSettings` "the client RPC surface and the peer relay" (`:776-790`); `applyEngineSettings` dispatches `ModelChange` to `applyModelSettings` (02 context, `enginesettings.go:642`); the desktop keys settings by `${nodeId}:${engine}` (`desktop/src/ui/stores/engine-settings.store.ts:26-27`). Phase 02 P11 itself deploys "through the existing relay".
- Advertisement needs only bounds, not the load: `Policy.Advertisement` publishes `ContextTokens`, `Capabilities`, `MaxOutputTokens`, `Qualification` from the profile (`services/shared/nodepolicy/advertise.go:427-443`); phase 02's `AdvertisedProfile` adds id, name, served model and identity, none of which is the load.
- Copy/deploy carries only portable fields re-resolved per target (2026-10-27 conclusion; 02 "Decisions (settled)"), so the policy copy of the load is never what a peer consumes.

What an alternative must meet: keep the load in the engine manager keyed by profile id (02 Task 2 does this anyway); have the coupled operation write the derived `contextTokens` and `modelIdentity` into the policy profile at commit (02 Task 3 step 2 already does this); drop `Profile.load`, `CheckProfileLaunchFields`, Task 3 step 4 and the matching tests. Same single source of launch intent, no duplicate, raw policy editing stays whole, and the 2026-10-27 decision stands. Nothing in the owner's statement asks for the load to live in policy; it asks for one named, copyable configuration object, which the join key alone provides.

### F4. RECOMMENDED CHANGE — a second generated identifier (`id`) beside the existing unique `Profile.Name`, and served model ids that are opaque hex

Consequence (behaviour, ownership): phase 02 P4 generates `p-` + 20 CSPRNG hex digits in the desktop and makes that string the preset section name and the native served id for every new profile. Clients that list `/v1/models` on a PAIR endpoint (the proxy merges upstream served ids with the tier aliases), the routing history and the `X-PAIR-Model`-style readbacks then show `p-3f9a...` for new profiles while imported ones keep readable names (P3). The owner's statement asks to "confirm what exactly is happening and where my request is going"; a hex id in a client's model list is the opposite of the "true identifier" the owner asked for. Meanwhile `Profile.Name` is already a validated, unique, node-scoped identifier with the same grammar the design proposes for `id`.

Factual premises and evidence:
- `Profile.Name` validated against `profileName` `^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$` and refused on duplicate: `services/shared/nodepolicy/policy.go:621, :671-676`; verify-A A2 (confirmed): "`Profile.Name` is a validated, unique identifier ... It just has no functional reader."
- P4's id grammar is the same pattern plus the reserved-alias and `validateLlamaModelID` rules (02 P4); `validateLlamaModelID` at `serving_llamacpp.go:519-527`.
- Tier aliases are appended to the merged upstream `/v1/models` list: `services/nvpair-proxy/tiers.go:550-556` (`tierModelEntries` "appends the synthetic alias records to a merged /v1/models list").
- P5 adds `name` as display text (1-128 bytes UTF-8) distinct from `id`; P2 makes the served id the profile id.

What an alternative must meet: one identifier. Use `Name` (grammar unchanged) as the profile key, the preset section and the served id; renaming is "create new, delete old" (the design already treats re-sourcing a model that way, P9). Keep a separate free-text display label and icon only if the owner wants Unicode names; the owner's own examples (`unsloth/LFM2.5-24B-A2B-GGUF` + `UD-Q4_K_XL` + role) are within the existing grammar. This removes `newProfileId()`, `ValidateProfileID`, the `p-` + sha256 fallback in P3, and the "id vs name" uniqueness pair.

### F5. RECOMMENDED CHANGE — a role cannot be defined before a profile is assigned to it, so "define roles, then assign" is not met and the Mac cannot define a role at all

Consequence (behaviour): the owner's first expectation is "in a PAIR Cluster, I can define Roles ... allow us to assign a model on each node to them". README T-E makes the cluster's role list "the union of names any node advertises plus the local defaults", and phase 02 Task 1 step 4 makes `Roles` a `map[string][]string` whose `normalize` "drops empty role lists" and whose `Validate` requires "list non-empty". A role therefore exists only once some node has a profile in it; the Mac, which declares no roles (vault `Archive/blocked.md:547` "empty Mac tiers map"), cannot create `coder` and then assign Joes's profile to it; the name must be typed on Joes's editor first. Two nodes spelling a new role differently produce two roles with no place to reconcile them.

Factual premises and evidence:
- Advertisement is built from `p.Roles` only (`advertise.go:427-443`); nothing advertises a role name without a model.
- 02 Task 1 step 4: "`normalize` drops empty role lists"; "Roles: ... list non-empty".
- README T-E alternative (b) rejects a cluster store because "advertisement already propagates names" — it propagates assignments, not definitions.

What an alternative must meet: a declared-but-empty role must survive in the declaring node's policy and advertisement (allow an empty list, or a separate `roleNames` set on the policy), so the desktop's role list is the union of declared names rather than of assignments. This is the smallest change that gives "define, then assign" without a cluster store.

### F6. RECOMMENDED CHANGE — fallback lists replacing the strength order are an unrequested abstraction that also discards the owner's stored `allowStronger`/`allowWeaker`

Consequence (ownership, data): the owner asked for name-based roles "for now ... later we could add more options/tags". D1's default replaces `TierPolicy.AllowStronger/AllowWeaker` with `tierPolicy.roleFallback[role] = [...]`, a new per-role list the owner must learn and maintain, and phase 03 notes a stored change to `allowStronger`/`allowWeaker` "revert[s] to the default fallback lists" (03-roles.md:156-157; 03 wording in flux, but the replacement is README T-E's own text).

Factual premises and evidence: `TierPolicy{AllowStronger, AllowWeaker, PreferLoaded}` at `services/shared/nodepolicy/policy.go:569-579`; the order dependence is `tierSearchOrder(tier, pol.TierPolicy)` (`services/nvpair-proxy/tiers.go:387, :469`); role keys validated against `TierOrder` at `policy.go:714, :743`; the only other enum is `routingHistoryTiers` (`services/nvpair-proxy/routing_events.go:580`).

What an alternative must meet: arbitrary role names routable by exact name, with the three defaults keeping their order and the two booleans; a custom role has no fallback until the owner asks for one. That is: relax the two validations, make the history field a string, and keep `tierSearchOrder` returning `[tier]` for a name outside `TierOrder`. Smaller surface, no data reversion, and the "more options later" door stays open.

### F7. RECOMMENDED CHANGE — D6 "accept any build" leaves the device helper reading a struct layout pinned to b11146 with the commit check downgraded to "recorded, not enforced"

Consequence (behaviour): the helper copies `ggml_backend_dev_props` into a Go struct whose size (56) and `ID` offset (40) are hard-coded from b11146's header; it runs in a subprocess, so a crash is contained, and a CUDA selector that does not match the driver's PCI topology ends as `Unknown`. But a layout change that keeps strings at plausible offsets, or a grown struct that overruns the 56-byte buffer, is not detected by `recover()`; phase 01 Task 10 step 2 turns `checkModelDeviceCommit` into "returns the `ggml_commit` string instead of comparing it". Auto then writes `options["device"]` from the helper's selector into a launch.

Factual premises and evidence: `services/nvpair-engine-manager/modelsettings_devices.go:662-671` (struct and header URL for `7fe450e`), `:683-685` (size/offset check), `:772-782` (`ggml_commit` compared today), `:568-585` (subprocess with bounded output), `:860-885` (`joinModelCUDADevices` cross-checks PCI with the CUDA driver and sets `Unknown` on mismatch); 01 Task 10 step 2 and step 6 ("an ABI difference must end as 'devices unavailable'"). `unverified`: whether b11219 or b11985 changed `ggml_backend_dev_props`; no llama.cpp source is available in this environment (`/home/user/Tools` does not exist).

What an alternative must meet: accept any build for `/props`, `build_info`, buffer capture and the configuration key (those need no ABI), but keep an allow-list of ggml commits (or `GGML_BACKEND_API_VERSION` values) whose props layout is known for device enumeration, returning `devices unavailable: unverified ggml layout` otherwise. The owner's builds are added to the list once their header is read; Auto placement stays blocked only on the one feature that actually depends on the ABI.

### F8. RECOMMENDED CHANGE — phases 03 and 04 are not disjoint, and phases 01 and 02 are far beyond one session

Consequence (cost): `00-index.md` says "03 and 04 may run in parallel after 02 where their files are disjoint". The two file lists overlap in 21 paths, among them `services/nvpair-proxy/proxy.go`, `tiers_test.go`, `desktop/src/ui/stores/node-configuration.store.ts`, `NodeList/NodeDeclaredRoles.tsx` and `ModelSettingsCopy.tsx`; phase 04 deletes `RoutingSettings.tsx` and `BulkNodeConfiguration.tsx` (04-desktop.md:599) that phase 03's role editor work edits. Parallel execution produces conflicts or discarded work. Each phase file is called "session-sized" (00-index), yet phase 01 names 119 distinct files over 11 tasks and phase 02 names 129 over 9, with P14 stating consumer modules "do not compile between Task 1 and their own task".

Factual premises and evidence: distinct-file counts from the `**Files:**` lines of each phase (`sort -u | wc -l`: 01 = 119, 02 = 129, 03 = 93, 04 = 200); overlap via `comm -12` of the 03 and 04 lists; 02 P14. (03/04 lists may shift under reconciliation; the index's "disjoint" premise is what is assessed.)

What an alternative must meet: a compiling tree at every task boundary and one owner-usable cluster at every phase boundary. Split 02 at the module seam (02a: shared types + Go consumers, 02b: desktop editors + import + deploy), and order 03 before 04 or move 03's desktop edits into 04.

### F9. OPTIONAL OPPORTUNITY — hashing and `--help` reads are traced; `pause hint` and `icon` are the only other additions, and both trace

Checked for untraced abstractions: `contentId` and the persisted one-time hash (owner: copy, sameness, speed; README T-A), `extraFlags` and the bounded `--help` read (owner: "introspect ... extra options", "fallback to just input manually"), `icon` (owner: "custom icons / display names"), the D5 pause hint to peers (not asked, but it serves "the router automatically send to different machine"; cost is one notification). Only `id` (F4), `load` in policy (F3) and `roleFallback` (F6) lack a stated requirement. No further untraced state found in sections 2-7, 01 or 02.

### F10. OPTIONAL OPPORTUNITY — retiring router local `tiers` entries loses nothing the owner uses; the API side is untouched

Premise checked: local entries are consulted only for nodes that declare no role (`services/nvpair-proxy/tiers.go:403-414`, verify-B B1 confirmed); API entries drive `dispatchPublic` (`routing.go:1273-1276`), egress leases (`egresslease.go:157-160`) and alias listing (`tiers.go:566-575`). README T-E keeps API entries. Risk is confined to a node that declares nothing (today the Mac), which after phase 03 would need at least one declared role or the defaults to receive work; the design's "declares this role" rule covers it. No correction needed; record the Mac case in phase 03's acceptance.

### F11. OPTIONAL OPPORTUNITY — freeze list (D7) cuts nothing the owner named

Premise checked against the owner statement: hybrid/public routing, strict qualification beyond context, linked-model follow-ups, LM Studio part 3, fixture campaigns, Ollama memory evidence appear nowhere in the owner's list; the owner asked to keep Ollama/LM Studio as apps (met by phase 06 `kind: app`) and named Strata as "eventually" (phase 06 includes Strata acceptance, which is more than the owner asked now and could itself be frozen to shorten the batch). The hybrid Todo's own status shows "Native/live acceptance and public activation remain open" (`Todo/2026-10-11-hybrid-routing/00-index.md:7`), so freezing it strands no landed feature.

## Owner decisions D1-D6: strongest case for each alternative

- D1 (user-defined names, union, fallback lists). Alternative "keep three names, add profiles": zero routing-order change, no history enum change, no data reversion, and the owner's example roles are the three names. Strongest case: F5 and F6 show the default as written neither lets the owner define a role before assigning nor keeps today's two booleans; the alternative keeps a working routing order while profiles land. If names are wanted now, F6's minimal version is the middle path.
- D2 (profiles in node policy, engine manager keyed by profile id). Alternative "engine-manager-owned profiles mirrored into policy": matches the settled 2026-10-27 split, keeps one copy of the load, needs no `CheckProfileLaunchFields`, and remote editing already works through the settings relay (F3 evidence). The policy mirror holds only what routing needs (id, name, served model, identity, derived context, capabilities).
- D3 (import MSI and Joes once). Alternative "keep running unconfigured": the only mode that today carries Joes's `fit-target` placement and MSI's `mmproj-auto`/`spec-type` keys (`serving_llamacpp.go:323, :349` unsupported); F1 shows the import cannot be done for Joes as designed. Until `fit` is typed, the alternative is the honest default for Joes; for MSI (launch args `--fit off`) the import can proceed.
- D4 (remove locks, toggles, switch UI; role-driven activation). Alternative "keep toggles as an advanced control": saved intent is what every non-exclusive start path keys on (`wakesleep.go:57-63`, `idlepolicy.go:183-188`), `RestoreEnabled` restarts saved-On engines after a crash or reboot (verify-A A6), and `owner_evidence.go:893-897` ties lock state to Off intent. Removing the user-facing toggle while keeping intent storage "re-purposed as Start with PAIR" (04 scope) is a rename, not a removal; the stronger case is to keep the control, hide it under the node settings page, and let F2's non-exclusive path make it rarely needed. The lock removal is the owner's explicit ask; the 2026-10-23 decision that introduced locks (vault `decided.md:833-846`) was to let a node forbid an engine from peers; `localOnly` on profiles and the local-only Halt cover the remaining need.
- D5 (Halt = immediate local pause, local-only; keep Q5 "incomplete" for apps). Alternative "force-stop app engines": the owner wrote Halt should "force unload"; an Ollama or LM Studio process PAIR did not start keeps its models resident, so Halt on an app-serving node is not a force unload. The case for the alternative is the owner's literal wording; the case against is Q5, which the owner answered. Unchanged premise: `availability.go:76-79` enforces local-only immediate pause (verify-B B2).
- D6 (accept any build). Alternative "keep the pin and install b11146 everywhere": contradicts the owner's own builds and Strata, so the full alternative is weak; F7's split (any build for `/props`, allow-listed ggml layouts for device enumeration) is the stronger middle.

## Sequencing (question 4)

- Phase 01 independence: holds. Its scope leaves profiles, roles, routing order and strict admission unchanged (01 Change scope); its entry conditions are baseline symbols only. Two coupling points to note: Task 6 touches `services/shared/nodepolicy/wire.go` (`ModelContentIDs`) and the proxy's performance identity, which the smart-routing Todo's Task 1 also targets (01 handles the collision in its entry conditions); Task 10 step 8 requires the owner to delete `model-buffer-samples.json` on machines that captured samples.
- Usable cluster at each boundary: after 01, yes (owner machines Preview only; unconfigured mode untouched). After 02, MSI can import; Joes cannot (F1) and is then refused every PAIR-side model change until it does, so Joes is "usable but frozen". After 03 without 04, the old editors carry role names. After 04 without 05, estimates still read "Unknown" and Validate is absent, so the owner's daily tuning loop is unchanged until phase 05, the fifth of six large phases. After 06, smart routing lands last, although it is the one item the owner has a debated plan for today and whose Task 1 phase 01 already replaces.
- Phase 02 size: F8.

## Simpler mechanism that meets the same constraints (question 2)

Profile = the existing `nodepolicy.Profile` with `Name` as the key (grammar unchanged), gaining `icon`, `modelIdentity` (phase 01) and a derived `contextTokens`; engine-manager `ModelLoad` gains `profileId = Name` and the generated preset uses `[<Name>]` as the section (so N configurations of one GGUF are N sections, served as their names); roles become `map[string][]string` of names; the broker's coupled operation stays the single writer of both records and writes the derived bounds into the policy profile; deploy = existing `engine:copy-model` + relay preview/apply with `portableLoadPatch`. This gives every property README section 4 lists except `load` inside policy and a second id, at the cost of fewer files in phase 02 (no `CheckProfileLaunchFields`, no `ValidateProfileID`, no `newProfileId`, no P3 hash fallback). Evidence for each facility: `policy.go:621, :671-676`; `advertise.go:427-443`; `enginesettings.go:845, :776-790`; `desktop/src/ui/components/ModelSettings/ModelSettingsCopy.tsx` already hands off to `ModelCopyModal` (02 context, verified by the plan's own citation; not re-read here).

## Unverified premises carried by the design (not findings; places the design leans on unchecked facts)

- llama-server accepts two preset sections naming the same `model` path with different `ctx-size` (02 Not verifiable; README T-P). No llama.cpp source available here.
- MSI's actual launch text and the `-ctx131072` cause (README section 8). Note the typed path drops the manifest's fixed `--ctx-size 8192` (`serving_llamacpp.go:388` `p.FixedArgs = nil`), while unconfigured mode keeps it, which is consistent with the README's explanation but not a confirmation.
- Whether Windows HF caches link into `blobs/` (README section 8).
- The ggml props layout on b11219/b11985 (F7).
- Joes's preset on disk today (F1; vault-recorded).

## DESIGN ASSESSMENT

Overall stance: sound with corrections. The diagnosis (section 2) matches the code at `4eccd05d`; the direction (identity, profiles as the join key, name-based roles, node-centric desktop, visible estimate, measured validate) serves the owner's statement. Two defaults fail the requirement as written (F1, F2), and three structural choices add surface the owner did not ask for where an existing facility already serves (F3, F4, F6).

Numbered findings:
1. F1 REQUIRED — D3 import cannot represent Joes's `fit-target` placement; typed mode forces `fit = off`; Joes is stranded or silently re-placed. (behaviour, data)
2. F2 REQUIRED — role-driven activation reuses the exclusive binding path; the owner expects multiple engines when resources allow; D4 does not state the exclusivity. (behaviour)
3. F3 RECOMMENDED — `load` in node policy duplicates the engine-manager record, adds a write guard, reverses 2026-10-27; the relay already provides remote edit and deploy. (ownership, data)
4. F4 RECOMMENDED — a generated `id` beside the already-unique `Profile.Name`; opaque served ids reach clients and history. (behaviour, ownership)
5. F5 RECOMMENDED — roles cannot be defined before assignment; the Mac cannot define one. (behaviour)
6. F6 RECOMMENDED — fallback lists replace the strength order and revert stored `allowStronger`/`allowWeaker`; not requested. (ownership, data)
7. F7 RECOMMENDED — any-build acceptance with a b11146-pinned props layout and an unenforced commit check; allow-list layouts for device enumeration only. (behaviour; layout change unverified)
8. F8 RECOMMENDED — 03/04 overlap in 21 files; 01 and 02 at 119 and 129 files are not session-sized; 02 has non-compiling intermediate states. (cost)
9. F9 OPTIONAL — traceability check: only `id`, policy `load` and `roleFallback` lack a stated requirement.
10. F10 OPTIONAL — retiring local tier entries affects only nodes that declare nothing; API entries untouched.
11. F11 OPTIONAL — freeze list cuts nothing the owner named; Strata acceptance in phase 06 exceeds the owner's "eventually".

Three strongest reasons the owner might reject the batch as a whole:
1. It is six phases of 93 to 200 files each, and the owner's daily loop (estimate visible, Validate, speed-aware routing) only changes at phases 05 and 06; the README's own sixth cause ("optimised for evidence rigour across many subsystems instead of the owner's daily loop") describes this batch's shape as much as the roadmap it replaces.
2. It retires the one mode that carries the owner's hand-tuned, measured configurations (Joes's native fit) before PAIR can express them, so the first visible effect on the fastest node is a refusal, not an improvement (F1).
3. It reopens a settled storage decision and introduces identifiers, policy-held loads and fallback lists the owner never asked for (F3, F4, F6), enlarging the surface the owner must learn in exactly the place they said should be "underlying settings you click through to see".
