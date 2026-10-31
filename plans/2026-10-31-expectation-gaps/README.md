<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PAIR expectation gaps: causes, recommendations and the batch

Written 2026-10-31 for the owner's sweep of places where the fork does not match
their expectations (vault `Tasks/2026-10-31 PAIR expectation gaps sweep.md`,
owner statement in vault `Expectation Drift.md`). Source investigated:
`integration/join33` at `4eccd05d` (2026-10-31), the current integration head.
`fork/groundwork` is 623 commits behind it and was not used as the baseline.

How to read this: section 2 is the short answer. Section 3 is the cause and
recommendation per theme, in the form the sweep asked for (fundamental fix,
alternatives, what it replaces, what it depends on). Section 4 is the target
model the recommendations add up to. Section 5 is the proposed batch, with the
execution plan in `00-index.md` and one file per phase. Section 6 lists the
decisions only the owner can make, each with a recommended default; the plan
assumes the defaults until the owner edits them. Section 7 is what to stop
doing. Section 8 is the verification record. Evidence is in `investigation/`
(eight finder reports L1 to L8 and two verifier reports); each claim cited below
as `Lx-Cn` or `A/Bn` is traceable there with file and symbol.

## 1. Method

Eight parallel read-only finder units (roles and tiers; llama.cpp model
identity; profiles and launch settings; engine lifecycle, locks and Halt;
desktop information architecture; memory estimate and validate; routing and
context; engines, apps, Strata and option introspection) wrote evidence-cited
reports against the worktree. Two independent verifiers then attacked the
sixteen claims the recommendations rest on, with a falsification brief. All
sixteen held in substance; seven gained corrections that changed the plan
(section 8). Nothing was built, run or tested; no machine of the owner's was
touched. Native behaviour on MSI, Joes, PC3 and the Mac is inferred from
source only.

## 2. The short answer: why it feels like going nowhere

The fork has landed a very large amount of machinery since `fork/groundwork`
(278 merges: typed model settings, memory telemetry, ranged estimates, engine
bindings, stop/pause, linked models, peer transfer, LM Studio credentials,
routing history, working-set caps). Almost none of it is visible on the
owner's machines, for five structural reasons. Each one is a cause, not a
symptom, and each explains several of the owner's complaints at once.

1. **There is no profile.** Both per-model records PAIR has (the node-policy
   `Profile` and the engine-manager typed `ModelLoad`) are keyed by the exact
   native model id, and each allows exactly one record per id (A2). A second
   configuration of the same GGUF therefore needs a second native alias. That
   is the whole reason context sizes and machine names ended up inside model
   names (`lfm2.5-24b-a2b-msi-ctx131072`, `joes-tiel35b-udq4kxl`), why roles
   point at model ids rather than configurations, and why a profile cannot be
   copied, named, iconed or assigned twice.
2. **PAIR does not publish what a model is.** The inventory is a bare list of
   names; `quantization` and `digest` are empty for every engine (B6). The
   desktop label strips both the owner and the quant from a Hugging Face name
   (`unsloth/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL` shows as `LFM2.5 24B A2B`, B6).
   Only the serving-envelope mode produces a content digest, so plain
   llama.cpp records no speed samples, cannot tell two nodes hold the same
   file, and peer copy lands the model under the source's section name (A5,
   L2). The HF cache resolver already has repository, commit, file names and
   sha256 for free and publishes none of it (A5).
3. **The owner's machines run the one mode the machinery does not cover.**
   MSI and Joes serve from hand-written llama.cpp presets ("unconfigured"
   mode). In that mode there is no digest, no configuration key, no stated
   memory figure, no native buffer capture and no measurement (B3, L6). Worse,
   the typed mode they would need to move to cannot express `n-cpu-moe`,
   `cpu-moe`, `repack`, `tensor-split` or `device`, although the preset parser
   accepts them and refuses to convert them (A3). And device enumeration,
   reviewed Auto and buffer capture all refuse unless the install manifest pins
   build `b11146`; the owner runs `b11219` and `b11985` (A7). The strict preset
   parser also fails the whole file on one unsupported key in any section, so
   inspection and copy export fail for every model on the engine (A4).
4. **Tiers are a closed, ordered vocabulary and the UI is engine-centric.**
   `weak`/`medium`/`strong` is validated, ordered, aliased and stored as an
   enum in fifteen hand-spelled places (L1). The node card is organised around
   engines with On/Off switches, padlocks and a switch transaction, and
   start-on-demand only wakes engines saved On (A6, L4); roles do not drive
   activation. The owner's model is role to profile to engine.
5. **Estimate is invisible and Validate is absent.** The service-side
   estimator prices weights, KV, MoE, MLA, SWA and recurrent state as ranges
   with a fits/tight/exceeds verdict, but it always reports status `partial`
   with fixed unknowns, the desktop ignores the verdict and applies its own
   all-or-nothing rule, so the owner always sees "Unknown", and Auto can never
   select (B3). The diagnostic run exists only as one drain helper called from
   tests (B4).

A sixth cause is process, not code: the roadmap has optimised for evidence
rigour across many subsystems (hybrid and public routing, strict admission,
linked models, LM Studio credentials, fixture isolation) instead of the
owner's daily loop on three machines. Section 7 proposes what to freeze.

## 3. Themes: cause, fix, alternatives, dependencies

### T-A. PAIR does not know what it is serving

**Observed.** Preset section names shown as models; no repository, file, quant,
size or context shown anywhere; copy export fails on MSI's preset; the
`-ctx131072` label runs at 8K.

**Cause (verified).**
- The llama.cpp model id is whatever llama-server's `/models` returns; in
  preset mode that is the INI section name (L2-C1..C3).
- `ModelsResult` carries names only; the desktop hard-codes empty
  `quantization` and `digest` (B6).
- `formatModelDisplayName` strips owner and quant (B6).
- `gguffacts` keeps architecture, context and tensor types but drops
  `general.name`, `basename`, `size_label` and `file_type` (A8).
- PAIR does parse the preset, but only on demand (inspection, conversion,
  export) with the strict serving-envelope parser; one unsupported key in any
  section fails every model on that engine, and inspection and preview hide the
  real error behind generic messages (A4). This corrects the sweep note's
  "without reading it".
- The `-ctx131072` mismatch is most likely the manifest's fixed
  `--ctx-size 8192` on the command line overriding the section (L7-C9; not
  confirmed without MSI's launch text).

**Fundamental fix.** Give every served model a structured identity that PAIR
computes itself and carries everywhere: `{repository, file or quant label,
contentId}` where `contentId` is the existing `servingContentDigest` formula
(role, size, sha256 per file; no paths or names, so it compares across nodes).
Populate it in `ModelsResult` from the existing artifact resolver (HF cache
blob sha256 when present, a one-time persisted hash otherwise) and from
`gguffacts` extended with the four general keys. Show it as tags in every
model and role label; stop stripping the quant.

**Alternatives.** (a) Read llama-server's `/models` metadata if it carries the
path (unverifiable here; would not give a cross-node identity). (b) Keep names
as identity and only fix the label: cheap, but leaves copy, sameness and speed
telemetry broken.

**Replaces.** Display-only formatting of the raw id; the `digest`-only identity
for speed samples (the smart-routing plan's "name-based identity" becomes
unnecessary once `contentId` exists).

**Depends on.** Nothing. This is the first phase. The lenient preset read is a
prerequisite for anything that inspects MSI or Joes today.

### T-P (new). Profiles: one object for model, engine and launch settings

**Observed.** Context in names; "profile" in PAIR means request-option
preferences per model; launch settings are a separate per-engine record; a
role is `{engine, model}`; nothing can be named, iconed, assigned twice or
deployed.

**Cause (verified).** One `Profile` per (engine, model) and one typed
`ModelLoad` per model id, both keyed by the native name (A2); `Profile.Name`
has no reader beyond a collision check; `Profile.ContextTokens` and the typed
load's `context` are two values that typed Apply never synchronises (B5);
roles are `map[tier]{engine, model}` (L1).

**Fundamental fix.** Introduce a profile id (stable, node-independent, user
visible as a name with optional icon) and make it the join key:
- node policy: `profiles[]` gains `id`, `modelIdentity` (section T-A), a
  `load` section (the typed `RequestedLoad`, moved here as the single source of
  launch intent) and keeps request options, capabilities and limits;
  `contextTokens` becomes derived from `load.context`, never typed by hand;
- engine-manager: typed loads keyed by profile id; the generated preset uses
  the profile id as the section name (so N configurations of one GGUF are N
  sections of one file, which llama-server already supports), and the served
  model id for that section is `<profile id>`; the advertisement maps profile
  id to `{modelIdentity, served name, engine, contextTokens, capabilities}`;
- roles: `roles[roleName] = [profileId, ...]` (ordered list; the router tries
  them in order, warm first under the existing comparator).
The broker's coupled model-change operation already updates both stores in
one journaled transaction (A1) and is the glue.

**Alternatives.** (a) Keep the two records and add a mapping table: less
movement, but the context duplication and the one-per-model rule remain.
(b) Alias-per-configuration by convention (what agents did by hand): no code,
but it is exactly the symptom the owner rejected.

**Replaces.** Hand-written aliases; the separate `ModelSettingsCopy` flow
(copying a profile replaces "copy preferences" plus "copy model" as two
actions); `Profile.contextTokens` as a typed field.

**Depends on.** T-A (identity). Owner decision D2.

### T-E. Roles are hard-coded and the tier editor is opaque

**Observed.** Only weak/medium/strong; Joes looked unset; two overlapping
mechanisms (node roles versus router tier entries).

**Cause (verified).** `TierOrder` and `IsTier` in Go and `ROLE_TIERS` in the
desktop are the sources of truth, with fifteen hand-spelled sites, nine
reserved `-api`/`-local` aliases, a tier enum in routing history, and routing
order (`tierSearchOrder`, `allowStronger`/`allowWeaker`, `tierRank`) that
depends on the order (L1, L7-C28/C29). A node that declares any role gets no
router tier entries for any tier, and the proxy keeps an unknown-tier marker
that the desktop hides, so the two disagree about who is authoritative (B1).
The desktop cannot edit `tiers` entries at all (L1).

**Fundamental fix.** Roles become names declared per node (`roles` map keyed by
string) and the cluster's role list is the union of names any node advertises
plus the local defaults; keep `weak`, `medium`, `strong` as the default names;
replace the implicit strength order with an explicit optional per-role
`fallback` list (a role may name which other roles may answer for it) so
`allowStronger`/`allowWeaker` still have a meaning for the defaults. Routing
history stores the role as a string. Router `tiers` entries (local and `-api`)
are retired from the routing path for local models (API entries stay for the
hybrid feature, frozen).

**Alternatives.** (a) Keep the three names and only add profiles: smallest,
but the owner's expectation 1 is not met and every future role is a code
change. (b) A cluster store for role names: rejected earlier for good reason
(no cluster store exists; advertisement already propagates names).

**Replaces.** `TierOrder`, `IsTier`, `borrowedAlias`, the history tier enum,
`ROLE_TIERS`, `TIER_LABELS`; the node-wide "declares any role" rule becomes
"declares this role".

**Depends on.** T-P (roles point at profiles). Owner decision D1.

### T-U. UI: engine-centric where it should be node- and role-centric

**Observed.** Engine On/Off and padlocks on cards; "show engine settings" and
an engine switch; node settings split across the card and "Routing &
resources"; no Models tab, no Profiles tab; Halt is a small button.

**Cause (verified).** The card renders `NodeEnginesInline` (switch, lock,
install, credential), `NodeDeclaredRoles` (raw ids), `NodeAvailabilityControls`
and an expandable `NodeEngineSettings` with `EngineSwitchControls` and one
`BackendRow` per engine (L5). Model lists are mounted per engine per node
inside a collapsed accordion although the renderer already holds every node's
inventory (B7). `ContentBar` has two tabs and two buttons (B7). Start-on-demand
wakes only saved-On engines; a role implies no binding; the bound-activation
path refuses while another engine runs (A6). The immediate local pause exists
with the right semantics but as "Stop local models now" (B2).

**Fundamental fix.**
- Activation follows roles: when a profile assigned to a role on this node is
  needed and its engine is not running, the engine is started (reusing the
  binding path's activation, generalised from "engine binding" to "profile
  needs engine"); saved On/Off intent and the switch transaction stop being
  user-facing. Install, update, running state and errors remain visible per
  engine, inside the node's settings.
- One settings page per node (opened from the card and from Cluster's "Edit
  node"): availability, roles with profiles, profiles on this node, engine
  process settings (ports, environment, working-set cap, autostart, install),
  advanced JSON, history.
- Tabs: Overview, Models (cluster-wide, grouped by identity then quant, with
  per-node presence, copy with progress from `ModelCopyModal`), Profiles
  (profile cards with usage node to role, deploy to node), Settings.
- Halt: the existing immediate pause renamed and promoted to the toolbar as a
  large local-only control; a peer-controllable On/Off is the existing
  configured Pause/Resume made symmetric (peers may resume what a peer paused;
  a local Halt stays local-only, which the code already enforces). Engine locks
  are removed (medium, about sixty source files, B2/L4); `localOnly` on model
  settings stays.

**Alternatives.** (a) Keep toggles and locks and add the tabs: less work, but
the card keeps asking questions the owner does not want to answer. (b) Remove
toggles before role-driven activation exists: breaks start-on-demand.

**Replaces.** `NodeEnginesInline` switches and `EngineLockControl`,
`EngineSwitchControls`, the per-engine `BackendRow` tree on the card,
`RoutingSettings` as a node picker page, `BulkNodeConfiguration` (bulk becomes
"deploy profile to nodes").

**Depends on.** T-P and T-E for the data; the Halt and the settings page do not
and can go first. Owner decisions D4, D5.

### T-C and T-D. Estimate, Validate and memory safety

**Observed.** Tuning done by hand; PAIR cannot propose or verify settings;
estimates show "Unknown"; no run-time RAM guard; hidden startup logs.

**Cause (verified).** Estimator landed on the service (ranges, verdict,
warnings) but always `partial` with fixed unknowns; the desktop ignores the
verdict; Auto requires a complete estimate (B3). A configuration key exists only
for typed, applied, finite-allocation llama.cpp loads with explicit GPU
placement and memory pools (B3). Device enumeration, Auto and buffer capture
gate on the install manifest pinning `b11146` (A7). Diagnostic run: absent but
fully planned in `Todo/2026-10-29-pair-memory-diagnostic-run.md` (B4, L6).
Nothing watches RAM or VRAM during a load; the Windows working-set feature is a
hard cap (L6). Native startup output is captured but not shown for custom
launches (vault `deferred.md`).

**Fundamental fix.**
- Make the estimate honest and visible: render the service verdict, ranges and
  "not priced" warnings as they are; "Run anyway"; stated figure entry
  (telemetry Task 6, estimation Task 12, both already specified).
- Replace the `b11146` install pin with a build identity read from the engine
  (`/props` `build_info`, already read in two places) plus the ggml library
  directory found by `detect`, so a user-installed build enumerates devices and
  captures buffers; record `build_info` in the measurement binding.
- Give preset-mode and unconfigured models a configuration key once they are
  profiles (T-P makes every served model a typed load, so this falls out).
- Validate: execute the diagnostic-run plan's tasks as written, reusing
  `drainNode`, `model-memory-profiles.json` and `modelConfigurationKey`, with
  two amendments the owner asked for: the result binds CPU and RAM identity as
  well as GPU, and the consent preview shows the owner's thresholds (more than
  low VRAM use, more than half RAM in use) rather than only "other programs use
  the GPU". An unmeasured profile stays assignable and routable (the plan
  already says so).
- Speed from a diagnostic run is written to the performance store under the
  profile's `contentId`, which also feeds the smart-routing plan.

**Alternatives.** (a) Skip the estimator and only measure: cheaper now, but
every new model needs a run before any guidance. (b) Keep the pin and ask the
owner to run `b11146`: contradicts the owner's own builds and Strata later.

**Replaces.** Hand-run agent sweeps and hand-edited presets; the desktop's own
fit rule in `ModelMemoryEstimate.tsx`.

**Depends on.** T-P (keys per profile); T-A (contentId for speed). Owner
decision D6.

### T-B and T-F. Routing is not speed- or capacity-aware; context

**Observed.** Ties go to the lowest node id; no speed data on MSI or Joes;
requests for 30k should go to a 32k or a 128k node; clients should see the
highest context.

**Cause (verified).** Chooser is tier rank, warmth, then scheduler load, then
node id (L7-C20..C22). Speed samples need a model digest that plain llama.cpp
lacks (A5, B8), and the sample key includes the engine run id so every restart
starts a new profile (B8). Context gating applies only to tier requests and
compares the request estimate to hand-declared `contextTokens` (0 means
unbounded); the model list alias carries no context field, pinned by a test;
launched context is read from `/props` but reaches only the desktop
(L7-C7/C8/C15, B5). Context is also enforced at strict readiness and broker
residency, so any sync must cover those (B5).

**Fundamental fix.** Execute the already debated smart-routing plan
(`Todo/2026-10-31-pair-smart-tier-routing.md`) after T-A, substituting
`contentId` for its name-based measurement identity and the profile id for
its (host, engine, model) evidence key; derive `contextTokens` from the
profile's load; advertise per role the maximum `contextTokens` among its
profiles across nodes as a `context_length` field on the alias record; gate
concrete-model requests the same way as tier requests.

**Alternatives.** The plan's own alternatives were debated and recorded.

**Replaces.** The plan's Task 1 (name-based identity) and part of Task 2.

**Depends on.** T-A, T-P. The plan's open choices (warm slow beats cold fast,
512-token default budget, probe every 16th request, 120 s cold cost) remain
the owner's.

### T-G. Backends versus apps; Strata; custom builds

**Observed.** Ollama and LM Studio are not backends in the owner's sense;
Strata matters; custom llama.cpp forks should be introspectable.

**Cause (verified).** Ollama and LM Studio are about 2.6% of the engine
manager's non-test lines plus family branches in shared files (L8-C35..C38);
the planned `kind: app` classification is absent (L8-C39). Strata is a
dynamic descriptor engine with offline tests and an unwired memory-plan path,
no bundled manifest, and open native acceptance on MSI (L8-C14..C18). A custom
build is selected through the override manifest's `detect` path, not
`runtime.bin` (L8-C9/C10). Nothing runs the binary for `--help` or `--version`;
per-engine extra arguments pass through; per-model options are a 107-key
static table, 79 of them refused (L8-C19, C29).

**Fundamental fix.** Land the small `kind: app` plan as written (it only
regroups). Add a bounded `llama-server --help` read at detect time to produce
the list of flags the build supports; show unknown-but-supported flags in the
profile editor as a "raw options" list with manual fallback (validation is the
engine's), and stop refusing presets that use them. Strata: complete the
memory-plan wiring and native acceptance as its Todo says, after T-P, so a
Strata profile is just a profile with a different engine.

**Alternatives.** Remove Ollama and LM Studio support: the owner said keep
them; the cost of keeping is small once they are grouped as apps.

**Depends on.** T-P for the editor; none for `kind`.

## 4. The target model (what the plan implements)

```
Cluster
  roles: union of names declared by nodes (defaults: weak, medium, strong)
  role fallback: optional list per role (replaces strength order)

Model identity (computed, node-independent)
  repository, file/quant label, contentId (= servingContentDigest formula),
  architecture, trained context, size

Profile (per node, copyable)
  id, name, icon?, engine, modelIdentity, load (typed RequestedLoad incl.
  n-cpu-moe/cpu-moe/repack/tensor-split/device/extra flags),
  requestOptions + mode + localOnly, capabilities, limits,
  contextTokens := load.context (derived),
  measurement? (diagnostic result bound to contentId, build_info, GPU, CPU, RAM)

Node
  roles[roleName] = [profileId, ...]
  availability: available | paused (configured, peer-reversible) | halted (local-only)
  engines: installed/running/errors, process settings (ports, env, cap, autostart);
  activation follows roles

Router
  candidate = (node, profile); eligibility by capabilities and contextTokens;
  order: warm, role rank/fallback, then expected completion (smart-routing plan)
  advertisement: per role, max contextTokens across nodes
```

## 5. The batch

Six phases, one file each under this folder, index in `00-index.md`. Sizes are
the finders' estimates; nothing was measured.

| Phase | Outcome | Size | Needs owner decision |
| --- | --- | --- | --- |
| 01 Identity and unblock | Lenient preset read; structured model identity in inventory and labels; gguffacts general keys; typed `n-cpu-moe`, `cpu-moe`, `repack`, `tensor-split`, `device`; build identity from the engine instead of the `b11146` pin | medium | D6 |
| 02 Profiles | Profile id as join key across node policy and engine manager; multiple profiles per model; derived context; roles map to profile lists; advertisement by profile; deploy-to-node | large | D2 |
| 03 Roles | User-defined role names with defaults and fallback lists; retire `TierOrder` and router local entries; history as strings | medium to large | D1 |
| 04 Desktop | Models tab, Profiles tab, per-node settings page, toolbar Halt and symmetric Pause, removal of engine toggles, locks and switch UI, role-driven activation | large | D4, D5 |
| 05 Estimate and Validate | Visible estimate; diagnostic run per its Todo with CPU/RAM binding and the owner's thresholds; speed into the performance store | large | — |
| 06 Routing and apps | Smart-routing plan on `contentId`/profile ids; context advertisement; `kind: app`; Strata acceptance | medium | — |

Order: 01 first (it is what makes MSI and Joes inspectable at all and is
independent), then 02, then 03 and 04 in parallel (disjoint files after 02),
then 05 and 06. Phase 01 can start now. The owner's plan convention (debate,
execute, audit per phase) applies to each file.

## 6. Decisions for the owner

Each has a recommended default that the plan assumes. Edit the plan's
"Decisions made while writing this plan" sections to veto.

- **D1 Role names.** Default: user-defined names, union across nodes, with
  `weak`/`medium`/`strong` pre-created and an optional per-role fallback list
  instead of a global strength order. Alternative: keep the three names and
  add only profiles.
- **D2 Profile placement.** Default: the profile lives in node policy
  (broker-owned, CAS-protected, remotely editable, advertised), and the engine
  manager keys its typed loads by profile id. Alternative: engine-manager-owned
  profiles mirrored into policy.
- **D3 Migrate MSI and Joes to PAIR-owned configuration.** Default: yes, as a
  one-time import of each preset section into a profile (phase 02's last
  task), after which hand-written presets are not used. Alternative: keep
  running unconfigured and accept no identity, estimate or measurement on
  those machines.
- **D4 Engine toggles, locks and switch.** Default: remove locks and the
  switch UI; replace toggles with role-driven activation plus install/update,
  running state and a per-engine "stop now" inside the node settings page.
  Alternative: keep toggles as an advanced control.
- **D5 Halt semantics.** Default: rename the immediate local pause to Halt,
  toolbar placement, local-only (as today); keep "incomplete" for app engines
  PAIR did not start (the owner's 2026-10-28 Q5 answer) rather than
  force-killing them; push a pause hint to peers instead of waiting for the 15
  s poll. Alternative: reverse Q5 and force-stop app engines.
- **D6 Supported builds.** Default: accept any llama.cpp build that answers
  `/props` and whose ggml libraries are found beside the detected binary;
  record `build_info` with every measurement; drop the `b11146` manifest pin
  from the device and buffer gates. Alternative: keep the pin and install
  `b11146` everywhere.
- **D7 Freeze list (section 7).**

## 7. What to stop doing

Freeze, do not delete: hybrid and public (provider) routing beyond what is
landed; strict local qualification beyond what phase 02 needs for context;
linked-model follow-ups; LM Studio part 3; new fixture-isolation campaigns;
Ollama-specific memory evidence. None of these is on the owner's list and each
has consumed multi-day cycles. Anything in the open Todos that phases 01 to 06
do not reference should be moved to `shelved.md` with a one-line reason, so the
board shows the batch and nothing else.

## 8. Verification record

Sixteen claims attacked by two verifiers (`investigation/verify-A.md`,
`investigation/verify-B.md`). Verdicts: 14 confirmed, 2 partly refuted (A6:
engine-manager's bound and automatic-switch paths do enforce one running
engine; B5: context is also enforced at strict readiness and broker
residency). Corrections that changed this document and the plan:

- A7: the `b11146` gate is the install manifest's artifact URL, not only
  `/props`; detecting the build from `/props` alone would not unblock devices,
  Auto or buffer capture (phase 01 task).
- A3: `n-cpu-moe` and friends are accepted by the preset parser but absent
  from typed loads and refused on conversion (phase 01 task).
- A4: one bad preset key breaks every model on the engine, and the error is
  hidden by inspection and preview (phase 01 task).
- A5: the free sha256 is per shard and only for caches that link into `blobs/`
  (phase 01 task uses a persisted one-time hash otherwise).
- B3: `partial` status and the fixed mmap and transient unknowns block the
  display and Auto, not runtime overhead alone (phase 05 task).
- B6: `quantization` and `digest` are empty for every engine; a new service
  field is required, not a label fix (phase 01 task).
- B1: proxy keeps an unknown-tier marker the desktop hides (phase 03 task).

Not verifiable here and left as acceptance items on the owner's machines:
MSI's actual launch text (the `-ctx131072` cause), whether Windows HF caches
link into `blobs/`, whether llama-server's `/models` carries paths, and every
native outcome.
