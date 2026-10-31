<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution plan: Phase 03: user-defined role names with defaults and fallback lists

## Goal
Overall goal of the batch (`README.md` in this folder): PAIR matches the owner's model of
role -> profile -> engine on MSI, Joes, PC3 and the Mac, so the machinery already landed
becomes visible and usable. This phase's goal (README section 3 T-E, section 5 row 03,
owner decision D1): role names are free strings that each node declares; the cluster's role
list is the union of the names any node advertises plus the defaults `weak`, `medium` and
`strong`, which every policy starts with. The implicit strength order is replaced by an
optional per-role fallback list (`tierPolicy.roleFallback[role] = [otherRole, ...]`) on the
router, so the search order for a request is "the role, then its fallback list in order" and
`tierRank` is the position in that list. Router-side local `tiers` entries no longer route
local models: a node is a candidate for a role only when it declares that role. Routing
history stores the requested role as a string. When the phase is done, the owner can add a
role `coder` on Joes from the Mac, send a request with model `coder` from the Mac, see it
routed to Joes and see `coder` in routing history; `weak` falls back to `medium` when the
router's fallback list says so.

## Change scope
Intent: the role vocabulary stops being a closed, ordered, hand-spelled set (`TierOrder`,
`IsTier`, `isTier`, `ROLE_TIERS`, the history enum) and becomes data.
Scope: Structural change — the role vocabulary, its validation, the router's role search
order, router local tier entries, the routing-history role field and the desktop's
role-name handling change; the smart-routing scorer, profiles, hybrid/API routing and the
desktop information architecture stay as they are.
Basis: owner decision D1 (README section 6) and the phase brief; repo rules
`.cursor/rules/no-legacy-fallbacks.mdc`, `services-change-propagation.mdc`,
`type-discipline.mdc`; the fork rule "pre-release fork: no compatibility work".
Compatibility: stored policies that use only the default names keep parsing because
`Default()` fills `roleFallback` and the three default roles; API (`kind: "api"`) tier entries,
`routing.tiers` and the nine reserved aliases keep their exact behaviour; `preferLoaded`
keeps its meaning. No version bump, capability marker or migration is added; the only data
handling is the existing disk-load repair dropping retired local tier entries (Task 4).
Non-goals: the smart-routing scorer (phase 06); context advertisement (phase 06); a role
editor redesign, a fallback-list editor, the Profiles tab or rendering the requested role in
the history table (phase 04); renaming `TierPolicy`, `tierSearchOrder`, `tierRank`, the
`requested.tier` history field, `X-PAIR-Tier` or the "no eligible model for tier" text;
any change to hybrid/public routing beyond restricting it to the default names (frozen, D7).

## Entry conditions
Phase 02 (`02-profiles.md`) must have landed. Check all of these on the branch before Task 1:
1. `plans/2026-10-31-expectation-gaps/00-index.md`: the row for phase 02 ends with
   ` — done <sha>`, and the vault `WORKLOG.md`
   (`/home/user/wasabi/4. Projects/Personal-AI-Router/WORKLOG.md`) has phase 02's close-out entry.
2. `services/shared/nodepolicy/policy.go`: `Policy.Roles` maps a role name to an ordered list
   of profile ids (not to the old `Role{Engine, Model}`), and `Profile` has a stable id field.
   Read phase 02's close-out entry for the exact field names; this plan calls them "the
   profile id" and "the role's profile list".
3. `services/shared/nodepolicy/advertise.go`: `Advertisement.Roles` is keyed by role name and
   its value lists the advertised profiles (phase 02 "advertisement by profile").
4. `services/nvpair-proxy/tiers.go` `facade.resolveTierCandidates` builds role candidates from
   the declared profiles (phase 02 changed the per-role `(engine, model)` match).
If any of these is absent, stop and report: do not implement phase 02's work here.
Phase 04 (`04-desktop.md`) is allowed to run in parallel with this phase and touches some of the same desktop files
(Task 6). Base on the current integration head; if phase 04 has already moved or deleted
`RoleEditor.tsx`, `NodeDeclaredRoles.tsx`, `BulkNodeConfiguration.tsx`,
`ModelSettingsCopy.tsx` or `ModelLoadSettings.tsx`, apply Task 6's change to the file that
now owns that behaviour and record the substitution in the close-out entry.

## Context the executor needs
Repository: the PAIR fork, executors branch from `origin/integration/join33` (`4eccd05d` or
later; this plan was written against `4eccd05d` plus phase 02's described interfaces). Line
numbers below are navigation aids at `4eccd05d`; re-find every symbol with `git grep` first.
Repo rules to follow: `.cursor/rules/services-change-propagation.mdc`,
`service-contracts.mdc`, `renderer-stores.mdc`, `type-discipline.mdc`,
`no-legacy-fallbacks.mdc`, `engines-vs-backends-naming.mdc`. Shared Go code lives in module
`services/shared` (`nvpair-shared`), which every service module consumes through a
`replace nvpair-shared => ../shared` directive, so removing a shared symbol breaks every
consumer in the same commit; each task below therefore updates all callers of what it removes.

Verified facts (investigation `L1`, `L7`, `verify-B` at `4eccd05d`; re-checked with `git grep`):
- Vocabulary source: `services/shared/nodepolicy/policy.go` `TierWeak`/`TierMedium`/`TierStrong`,
  `TierOrder` (about line 110) and `IsTier` (about line 805). Non-test users of these symbols:
  `policy.go` `Validate` (roles loop about line 713, tiers loop about line 742),
  `policy_load.go` (about line 139 `validateTierRefs(TierWeak, ...)`, about line 174
  `!IsTier(tier)`), `routing.go` `defaultRouting` (about line 320) and `Routing.validate`
  (about lines 429-440), `values.go` `ParseAlias`/`IsReservedAlias` (about lines 106-129),
  proxy `discovery.go` `Node.filteredModels` (about line 153), `http_handler.go` (about line
  748), `routing.go` (about line 684), `tiers.go` `tierSearchOrder` (about line 233) and
  `configuredAliases` (about line 567) (L1-C1..C3, L7-C28).
- Test files referencing those symbols (`git grep -lw "TierOrder\|IsTier\|TierWeak\|TierMedium\|TierStrong" -- services`):
  19 test files, all in Task 1's Files (which also adds `values_test.go` for the alias cases).
- `services/shared/modelconfig/wire.go` `isTier` (about line 503) is a literal copy used by
  the `modelId` reserved-alias check (about line 498) and `PolicyPatch.Validate`
  ("roles: unknown tier", about line 621). `nodepolicy` does not import `modelconfig`.
- Fallback: `TierPolicy{AllowStronger, AllowWeaker, PreferLoaded}` (`policy.go` about line
  569), `Default()` sets `AllowStronger: true, AllowWeaker: false, PreferLoaded: "tier"`.
  `tiers.go` `tierSearchOrder` derives order by index in `TierOrder`; `resolveTierCandidates`
  assigns `tierRank` = index in that order and sorts by `preferLoaded`, `tierRank`, warmth, the
  `node/select` pin, preparation (L7-C20, C27). `hybridLocalCandidates` reads
  `order[c.tierRank]`. `projection.go` `MergePeerSet` refuses a peer change with
  `incoming.TierPolicy != stored.TierPolicy` (a struct `!=`, which stops compiling once
  the struct holds a map).
- Policy decoding is not strict: `Parse` decodes onto `Default()` with `json.Unmarshal`, so an
  unknown field such as `allowStronger` is ignored and map fields merge key by key.
- Router local entries: `TierEntry` has `Kind` (`local` or `api`), `Engine`, `Model`,
  `ContextTokens`, `Capabilities` and `Deployment`; `resolveTierCandidates` adds a local entry's candidates only for nodes with
  `len(declared.of(c)) == 0` (node-wide authority, L1-C9, B1). `candidate.declared`
  (`proxy.go` about line 605) is written only by `resolveTierCandidates` and read nowhere else.
  `PeerPolicy.Tiers` carries only local entries (`projection.go` `localTiers`). API entries
  are read by `routing.go` `dispatchPublic` (about line 1273) and `egresslease.go` (about line
  160). `EntryLocal` appears in 29 files (`git grep -lw EntryLocal -- services`).
- Disk load: `ParseStored` repairs invalid model entries in memory (`filterPolicyModels`),
  never persists the repair, and a stored policy that still fails puts the node on the paused
  recovery policy (`recoveryRequired`, broker README `policy:get`). That is why Task 4 drops
  retired local entries through that repair instead of letting them fail validation.
- Unknown roles today: proxy `filteredModels` keeps only `TierOrder` keys but retains one
  unknown-tier "marker" when no known tier survives, so such a node stays authoritative and the
  desktop's `parseNodeRoles` (`desktop/src/shared/utils/node-roles.ts`) drops it (B1).
- History: the wire record field `RoutingHistoryRequested.Tier` is already a `string` in
  `services/shared/nodepolicy/routing_wire.go` (record, `RoutingHistoryRecordHeader` and the
  `Validate` header initializer all carry `Requested` whole); only validation
  (`historyEnum(r.Requested.Tier, "|weak|medium|strong")`, about line 810) restricts it. The
  proxy's native fact `routingRequestedDescriptor{model [512]byte; length uint16; tier, mode uint8}`
  (`routing_facts.go` about line 126) stores the tier as 1/2/3 via a switch in
  `requestedDescriptor` and `routing_events.go` decodes it with `routingHistoryTiers`
  (about line 580). Broker, service and TUI packers walk the schema reflectively and pack
  strings with a 16-bit length (`nvpair-ui-broker/routing_events.go` `historyPackValue`), so
  the smart-routing plan's record/header/mapping/initializer quadruple needs no new field
  here: the field exists and stays a string. The TUI has no tier code (L1-C15; it shows
  `Requested.Model`). The shared fixture `services/shared/nodepolicy/testdata/routing-request.json`
  is consumed by Go (`routing_wire_test.go`), the TUI (`routingwire_test.go`,
  `routingview_test.go`) and the desktop (`routing-projection.test.ts`, `routing-delivery.test.ts`,
  `routing-view.test.ts`).
- Desktop hand-spelled sites (L1-C4): `shared/types/node-roles.ts` `RoleTier`;
  `shared/utils/node-roles.ts` `ROLE_TIERS`; `RoleEditor.tsx` `TIER_LABELS`;
  `shared/types/model-config.ts:75`; `shared/types/routing.ts` (about lines 126, 403, 596);
  `electron/service-bridge/engine-settings.ts:85` `modelId`; `ModelSettingsCopy.tsx`
  (about lines 120, 380, 680-682); `ModelLoadSettings.tsx` (about lines 420, 427);
  `shared/utils/node-policy-model-settings.ts:233`. `node-configuration.store.ts`
  `suggestRoles` reads local tier entries and is the only desktop reader of `tiers`. The
  desktop does not edit `tierPolicy` (L1-C14); RoutingSettings has an "Advanced policy JSON"
  editor that does.
- Pause hints remove a paused remote node from inference candidates
  (`docs/node-availability.mdx` "How peers learn a node is paused").

Test categories that need edits (L1 counted 65 Go test files with 682 quoted tier names; most
keep passing unchanged because the defaults keep the same names):
1. tests asserting rejection of an unknown name such as `huge` (now a valid role name):
   `nodepolicy` `policy_test.go`, `routing_test.go`, `projection_test.go`, `values_test.go`,
   broker `policy_roles_test.go`, `modelconfig` `options_test.go`;
2. alias construction and listing (`values_test.go` `ParseAlias`, proxy `hybrid_alias_test.go`);
3. the history enum (`routing_events_test.go`, `routing_facts_test.go`, `routing_wire_test.go`,
   desktop `routing-projection.test.ts`);
4. `AllowStronger`/`AllowWeaker` fixtures (Task 3's list);
5. router local tier entries and node-wide authority (Task 4's list), including
   `subscribed_test.go` `TestUnknownTierDeclarationPreventsRouterFallback`;
6. mechanical symbol renames (Task 1's list).

Not verifiable while writing: phase 02's exact field names; every native outcome on the
owner's machines; whether the owner's stored policies changed `allowStronger`/`allowWeaker`
from the defaults (Task 3 makes such a change revert to the default fallback lists).

## Decisions (settled — do not revisit)
- Role names are user-defined strings declared per node; the cluster's role list is the union
  of names nodes advertise plus `weak`, `medium`, `strong`, which `Default()` pre-creates (D1).
  Not "keep the three names and add only profiles", because the owner's expectation 1 asks for
  user-defined roles.
- An optional per-role fallback list replaces the global strength order; `tierRank` is the
  position in "the role, then its fallback list" (D1; keeps the smart-routing plan's
  "tier rank is a hard leading class" valid, L7-C34).
- The nine reserved `-api`/`-local` aliases, API tier entries and `routing.tiers` exist only for
  the three default names; hybrid/public routing is frozen (D7, README section 7).
- Router local `tiers` entries are retired from local-model routing; declared roles are the only
  local candidates, and the node-wide "declares any role" rule disappears with them (README T-E).
- `preferLoaded` keeps its two values and meaning; the smart-routing scorer is untouched.
- No compatibility shims, version bumps or capability markers; `Default()` fills the defaults so
  existing policies parse (fork rule, `no-legacy-fallbacks.mdc`).
- Roles map to profile lists as phase 02 left them; this phase does not change that shape.

## Decisions made while writing this plan
- Role-name grammar: `^[a-z0-9][a-z0-9._-]{0,63}$` (1 to 64 bytes, lower case), and the name must
  not end in `-api` or `-local`. Lower case only, so `Coder` and `coder` cannot both exist; no
  `:` so Ollama's `:latest` handling never touches a role name.
- A role name must not equal a profile id of the same node (checked in `Policy.Validate`). Across
  nodes the router cannot check, so a role request wins over a concrete model of the same name
  elsewhere; this is documented in Task 8.
- The router treats a request `model` as a role when it is a default name, or when it passes the
  grammar and this node or any visible peer declares it; otherwise it is a concrete model name.
- `tierPolicy.roleFallback` is router-side (applies to requests entering that node), like
  `TierPolicy` today; peers still cannot change it (`MergePeerSet` keeps refusing). In this phase
  it is edited through the desktop's existing "Advanced policy JSON" editor; a fallback editor is
  phase 04 work.
- Fallback is one level, not transitive; duplicates are removed keeping the first; a list must not
  name its own key; names no node declares are allowed and simply contribute no candidates.
- Default fallback lists mirror today's defaults (`allowStronger: true`, `allowWeaker: false`):
  `weak: [medium, strong]`, `medium: [strong]`, `strong: []`. A stored `allowStronger`/`allowWeaker`
  is ignored after this phase, so a node whose owner had changed them reverts to these lists;
  the acceptance step checks the owner's routers.
- Keep the identifiers `TierPolicy` (JSON `tierPolicy`), `tierSearchOrder`, `tierRank`, the
  history field `requested.tier`, the header `X-PAIR-Tier` and the error "no eligible model for
  tier X": renaming them is churn with no behaviour, and the smart-routing plan cites them.
  Constants are renamed `RoleWeak`/`RoleMedium`/`RoleStrong`, `TierOrder` becomes
  `DefaultRoles`, `IsTier` becomes `IsDefaultRole` plus `ValidateRoleName`.
- Local tier entries leave the format: `TierEntry` keeps `kind` with the single value `"api"` (so
  API entries parse unchanged) and `deployment`; `Parse` refuses `kind: "local"` with a message
  naming the replacement; `ParseStored` drops `kind: "local"` and kind-less entries through the
  existing in-memory repair and counts them under `tiers`; the legacy version 1 migration drops
  tier entries (every version 1 entry was local); `PeerPolicy` loses `tiers`, and a peer write
  carrying a `tiers` member in any spelling is refused like `routing`.
- The routing-history native fact stores the role in a fixed 64-byte buffer
  (`nodepolicy.MaxRoleNameBytes`).
- Desktop: an added role is a draft row and is saved only once it has at least one profile (an
  empty role is omitted, as unset tiers are today); role rows list defaults first, then other
  names alphabetically; the label is the role name itself. The "Suggest" button and its copy are
  removed because they read retired local tier entries.
- Packaging: eight tasks. Tasks 1, 3, 4, 5 are serial because they share `policy.go` and
  `tiers.go`; Task 2 can run after Task 1 in parallel with 3 to 5; Task 6 needs Tasks 1 and 4;
  Task 7 needs Tasks 1 and 6 (it uses the desktop grammar helper); Task 8 (docs) runs last.

## Tasks

### Task 1 — Role-name grammar and default roles replace TierOrder in nodepolicy
**Files:** services/shared/nodepolicy/policy.go, services/shared/nodepolicy/values.go, services/shared/nodepolicy/routing.go, services/shared/nodepolicy/policy_load.go, services/shared/nodepolicy/policy_test.go, services/shared/nodepolicy/values_test.go, services/shared/nodepolicy/routing_test.go, services/shared/nodepolicy/policy_load_test.go, services/shared/nodepolicy/projection_test.go, services/shared/nodepolicy/advertise_test.go, services/shared/nodepolicy/model_controls_test.go, services/shared/nodepolicy/wire_test.go, services/shared/noderec/noderec_test.go, services/nvpair-proxy/discovery.go, services/nvpair-proxy/http_handler.go, services/nvpair-proxy/routing.go, services/nvpair-proxy/tiers.go, services/nvpair-proxy/availability_test.go, services/nvpair-proxy/subscribed_test.go, services/nvpair-ui-broker/discovery_test.go, services/nvpair-ui-broker/policy_projection_test.go, services/nvpair-ui-broker/policy_roles_test.go, services/nvpair-ui-broker/policy_test.go, services/nvpair-engine-manager/nodeprojection_test.go, services/nvpair-node-scanner/availability_test.go, services/nvpair-node-scanner/inventory_capability_test.go, services/nvpair-node-scanner/refresh_test.go, services/nvpair-node-scanner/reverse_peer_test.go
**Depends on:** none — phase 02 is an entry condition, not a task
**Complexity:** judgment
**Do:**
1. In `services/shared/nodepolicy/policy.go`, replace the "Tier names" constant block, `TierOrder`
   and `IsTier` with:
   ```go
   // Default role names. Every policy starts with these roles, and only they have
   // the -api and -local hybrid aliases. Any other valid role name is a plain role.
   const (
   	RoleWeak   = "weak"
   	RoleMedium = "medium"
   	RoleStrong = "strong"
   )

   // DefaultRoles lists the default role names, weakest first. The order is used
   // only for listing; routing order comes from TierPolicy.RoleFallback.
   var DefaultRoles = []string{RoleWeak, RoleMedium, RoleStrong}

   // MaxRoleNameBytes bounds a role name; routing history stores it in a fixed buffer.
   const MaxRoleNameBytes = 64

   var roleNamePattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,63}$`)

   // ValidateRoleName reports why name cannot name a role, or nil.
   func ValidateRoleName(name string) error {
   	if !roleNamePattern.MatchString(name) {
   		return fmt.Errorf("role name %q must be 1 to 64 lower-case letters, digits, '.', '_' or '-', starting with a letter or digit", name)
   	}
   	if strings.HasSuffix(name, "-api") || strings.HasSuffix(name, "-local") {
   		return fmt.Errorf("role name %q must not end in -api or -local", name)
   	}
   	return nil
   }

   // IsDefaultRole reports whether name is one of DefaultRoles.
   func IsDefaultRole(name string) bool { return slices.Contains(DefaultRoles, name) }
   ```
2. `Policy.Validate`: in the roles loop replace the `TierOrder` membership check with
   `ValidateRoleName(name)` (error `roles: %w`), and add a check that the role name is not equal
   to the profile id of any profile in `p.Profiles` (error
   `roles: role name %q is also a profile id on this node`). In the `p.Tiers` loop replace the
   membership check with `IsDefaultRole(tier)` (error
   `tiers: %q is not a default role; only weak, medium and strong have API entries`).
3. `routing.go`: `defaultRouting` iterates `DefaultRoles`; `Routing.validate` checks keys with
   `IsDefaultRole` (error `routing.tiers: %q is not a default role`) and still requires every
   default role to be present.
4. `values.go` `ParseAlias`: use `IsDefaultRole`; update its doc comment to "each default role,
   bare or with the -api or -local suffix". `IsReservedAlias` is unchanged.
5. `policy_load.go`: the single-entry check uses `RoleWeak`; in `checkStoredModelShapes`' member
   loop, `roles` keys are checked with `ValidateRoleName` and `tiers` keys with `IsDefaultRole`.
6. Callers outside `nodepolicy`, behaviour unchanged in this task: proxy `discovery.go` and
   `tiers.go` use `nodepolicy.DefaultRoles` where they used `TierOrder`; `http_handler.go` and
   `routing.go` use `nodepolicy.IsDefaultRole(model)` where they used `IsTier(model)`. Tasks 3
   and 5 change their behaviour.
7. Tests: rename `TierWeak`/`TierMedium`/`TierStrong` to `RoleWeak`/`RoleMedium`/`RoleStrong` and
   `TierOrder` to `DefaultRoles` in every file in Files. Rewrite each case that expects
   `unknown tier` for a grammatical name (`huge`, `fast`) so it uses names the grammar rejects
   (`Huge`, `huge-api`, `a b`, a 65-byte name) and expects the new messages; the case
   "suffixed alias as tier" (`weak-local`) now expects "must not end in -api or -local". Add
   table tests for `ValidateRoleName` (accepts `weak`, `coder`, `r1.fast_v2`, a 64-byte name;
   rejects empty, `Coder`, `-x`, `coder-api`, `coder-local`, `co:der`, a 65-byte name), a
   `Validate` case accepting `roles.coder`, a case rejecting a role named like one of the node's
   profile ids, cases rejecting `tiers.coder` and `routing.tiers.coder`, and a `ParseAlias`
   case where `coder-api` is not an alias.
**Verify:** `git grep -nw "TierOrder\|IsTier\|TierWeak\|TierMedium\|TierStrong" -- services` prints
nothing; `cd services/shared && go vet ./... && go test ./...` passes; in each of
`services/nvpair-proxy`, `services/nvpair-ui-broker`, `services/nvpair-engine-manager`,
`services/nvpair-node-scanner`: `go build ./... && go vet ./... && go test ./...` passes.

### Task 2 — Model-settings patches validate role names with the shared grammar
**Files:** services/shared/modelconfig/wire.go, services/shared/modelconfig/wire_test.go, services/shared/modelconfig/options_test.go
**Depends on:** 1 — uses `nodepolicy.ValidateRoleName` and `nodepolicy.IsReservedAlias`
**Complexity:** mechanical
**Do:**
1. In `services/shared/modelconfig/wire.go` delete `isTier`. Import `nvpair-shared/nodepolicy`.
2. In the `modelId` check, replace the hand-written suffix strip and `isTier` call with
   `if nodepolicy.IsReservedAlias(id) { return fmt.Errorf("modelId: reserved routing alias") }`.
3. In `PolicyPatch.Validate`, replace the `isTier` loop with
   `for role := range p.Roles { if err := nodepolicy.ValidateRoleName(role); err != nil { return fmt.Errorf("roles: %w", err) } }`.
4. `options_test.go`: the case named "unknown tier" (`{"fast":true}`) becomes "invalid role name"
   with input `{"Fast":true}`; add an accepted case `{"coder":true}`. `wire_test.go`: keep the
   reserved-alias cases for `weak`, `weak-api`, `medium-local`; add `coder` accepted as a
   `modelId` (a role name is not reserved).
**Verify:** `git grep -n "func isTier" -- services/shared/modelconfig` prints nothing;
`cd services/shared && go build ./... && go vet ./... && go test ./modelconfig/...` passes (a
build error "import cycle not allowed" means stop and report).

### Task 3 — Per-role fallback lists replace allowStronger and allowWeaker
**Files:** services/shared/nodepolicy/policy.go, services/shared/nodepolicy/projection.go, services/shared/nodepolicy/policy_test.go, services/shared/nodepolicy/projection_test.go, services/nvpair-proxy/tiers.go, services/nvpair-proxy/tiers_test.go, services/nvpair-proxy/routing_test.go, services/nvpair-proxy/routing_listener_test.go, services/nvpair-proxy/served_models_test.go, services/nvpair-proxy/engine_binding_candidates_test.go, services/nvpair-ui-broker/policy_test.go, services/tests/policy_relay_joint_test.go
**Depends on:** 1 — same `policy.go`, `tiers.go` and test files
**Complexity:** judgment
**Do:**
1. In `policy.go` replace `TierPolicy` with:
   ```go
   // TierPolicy controls how this router widens a role request.
   type TierPolicy struct {
   	// RoleFallback lists, per requested role, the other roles that may answer
   	// when no candidate of the requested role is eligible, in the order tried.
   	// It is not transitive. A role absent from the map has no fallback.
   	RoleFallback map[string][]string `json:"roleFallback"`
   	// PreferLoaded ranks candidates already loaded on their node ahead of cold
   	// ones. "tier" keeps fallback order first and prefers loaded models within
   	// each position; "any" prefers any loaded eligible model over a cold one.
   	PreferLoaded string `json:"preferLoaded"`
   }
   ```
   `Default()` sets `RoleFallback: map[string][]string{RoleWeak: {RoleMedium, RoleStrong}, RoleMedium: {RoleStrong}, RoleStrong: {}}`.
   `normalize` makes a nil map an empty map and a nil list an empty list. `Validate` checks each
   key and each listed name with `ValidateRoleName` (error `tierPolicy.roleFallback.%s: %w`),
   refuses a list naming its own key and refuses a duplicate in one list.
2. `projection.go` `MergePeerSet`: replace `incoming.TierPolicy != stored.TierPolicy` with
   `!reflect.DeepEqual(incoming.TierPolicy, stored.TierPolicy)`; keep the refusal text.
3. `services/nvpair-proxy/tiers.go`: rewrite `tierSearchOrder` (keep the name) to return the
   role, then each name of `tp.RoleFallback[role]` not already in the order. It no longer
   returns nil for an unknown name. Update its doc comment. `tierRank` stays the index in that
   order; do not change the sort, `reserveRoundPrepared` or `hybridLocalCandidates`.
4. Tests: replace every `AllowStronger`/`AllowWeaker` setting (Go struct fields and JSON
   `allowStronger`/`allowWeaker` members) with the equivalent lists:
   | allowStronger | allowWeaker | roleFallback |
   | --- | --- | --- |
   | true | false | weak: [medium, strong]; medium: [strong]; strong: [] (the default) |
   | true | true | weak: [medium, strong]; medium: [strong, weak]; strong: [medium, weak] |
   | false | true | weak: []; medium: [weak]; strong: [medium, weak] |
   | false | false | weak: []; medium: []; strong: [] |
   Where a test toggled `allowWeaker` only to make a peer change the tier policy
   (`nvpair-ui-broker/policy_test.go`, `services/tests/policy_relay_joint_test.go`), change
   `roleFallback.weak` instead and keep the expected refusal. Add `tiers_test.go` cases: a
   custom order (`weak: [strong, medium]`) yields ranks 0, 1, 2 in that order; a duplicate-free
   result when the list repeats the role; a fallback to an undeclared name contributes no
   candidates; `preferLoaded: "any"` still lets a warm fallback candidate lead. Add
   `policy_test.go` cases for the defaults after `Parse("{}")`, a self-reference refusal and a
   file containing `allowStronger: false` parsing to the default lists.
**Verify:** `git grep -n "AllowStronger\|AllowWeaker\|allowStronger\|allowWeaker" -- services ':!*.md' ':!services/shared/nodepolicy/policy_test.go'`
prints nothing (the excluded file holds the new case that parses a stored `allowStronger`); `cd services/shared && go test ./nodepolicy/...` passes;
`cd services/nvpair-proxy && go vet ./... && go test ./...` passes;
`cd services/nvpair-ui-broker && go test ./...` passes;
`cd services/tests && go test -run 'TestPolicyRelayJoint' ./...` passes.

### Task 4 — Router local tier entries are retired; declared roles are the only local candidates
**Files:** services/shared/nodepolicy/policy.go, services/shared/nodepolicy/routing.go, services/shared/nodepolicy/policy_load.go, services/shared/nodepolicy/projection.go, services/shared/nodepolicy/policy_test.go, services/shared/nodepolicy/policy_load_test.go, services/shared/nodepolicy/routing_test.go, services/shared/nodepolicy/projection_test.go, services/shared/nodepolicy/model_controls_test.go, services/nvpair-proxy/tiers.go, services/nvpair-proxy/proxy.go, services/nvpair-proxy/buffer_decrease_retry_test.go, services/nvpair-proxy/engine_binding_candidates_test.go, services/nvpair-proxy/engine_binding_requests_test.go, services/nvpair-proxy/hybrid_alias_test.go, services/nvpair-proxy/local_credential_test.go, services/nvpair-proxy/model_controls_test.go, services/nvpair-proxy/owned_buffers_test.go, services/nvpair-proxy/policy_test.go, services/nvpair-proxy/reverse_transport_test.go, services/nvpair-proxy/routing_listener_test.go, services/nvpair-proxy/routing_test.go, services/nvpair-proxy/served_model_body_test.go, services/nvpair-proxy/served_models_test.go, services/nvpair-proxy/subscribed_test.go, services/nvpair-proxy/tiers_test.go, services/nvpair-ui-broker/policy_test.go, services/tests/model_routing_interop_test.go, services/tests/policy_relay_joint_test.go, services/tests/public_egress_lease_test.go, services/tests/service_policy_test.go, services/tests/service_routing_listener_test.go, services/tests/unified_dispatch_test.go
**Depends on:** 3 — same `policy.go`, `projection.go`, `tiers.go` and test files
**Complexity:** judgment
**Do:**
1. `policy.go`: `TierEntry` becomes `{Kind EntryKind \`json:"kind"\`; Deployment string \`json:"deployment,omitempty"\`}`
   with `EntryAPI` the only `EntryKind`; delete `EntryLocal`, `TierEntry.IsLocal`, the local
   fields and the tier-capability normalisation. Rewrite the `Roles` doc comment to: a node is
   offered for a role only when it declares that role, and a role absent from the map is not
   served by this node. Rewrite the `Tiers` doc comment to: the API deployments a default role
   uses for hybrid routing (frozen); local candidates come only from declared roles.
   `migrateV1` no longer converts tier entries: a version 1 file's tiers are dropped (all
   version 1 entries were local); delete `tierEntryV1`.
2. `routing.go` `validateTierRefs`: only `EntryAPI` is valid; for `kind == "local"` return
   `tiers.%s[%d]: local tier entries are retired; declare the role on the serving node`, for any
   other kind `tiers.%s[%d]: kind must be "api"`. Adapt `entryKey`.
3. `policy_load.go`: in `checkStoredModelShapes` and `filterPolicyModels`, a tier entry whose
   `kind` is `"local"` or absent is removed on disk load and counted under the existing `tiers`
   count; remove the version 1 kind defaulting. Do not persist the repair (existing rule).
4. `projection.go`: delete `PeerPolicy.Tiers`, `localTiers` and `nonEmptyTiers` (if no other
   caller remains); in `MergePeerSet` refuse any member spelled `tiers` in any case with
   `a paired node cannot change the tiers`, the same way `routing` is refused; keep
   `merged.Tiers = stored.Tiers`. Update the `PeerProjection` doc comment.
5. `services/nvpair-proxy/tiers.go` `resolveTierCandidates`: delete the `pol.Tiers[t]` entry loop
   and the `fromRole` parameter of `add`; delete `candidate.declared` in `proxy.go`. Rewrite the
   doc comments of the file header, `declaredRoles` and `resolveTierCandidates` so they no longer
   mention router tier entries or node-wide authority. `configuredAliases` keeps reading
   `pol.Tiers` (API entries only now).
6. Tests, by category: (a) a test that used a local tier entry only to make a role request
   resolve declares the role on the serving node instead (phase 02 shape: the role's profile list
   names an existing profile); (b) a test of router-entry behaviour (entries apply only to
   non-declaring nodes, entry context or capability bounds, declared-before-entry order, the
   unknown-tier marker disabling router fallback, `IsLocal`, and the comment "authoritative for
   itself" in `tiers_test.go`) is deleted, and where a bound or
   capability check was the subject it is re-expressed on the declared role's profile;
   (c) parse tests: `Parse` refuses `kind: "local"` with the retired message, `ParseStored` drops
   it with `counts["tiers"]` incremented and returns a valid policy, the version 1 migration test
   expects empty tiers, projection tests expect no `tiers` in `PeerPolicy` and a refused peer
   write that carries `tiers`. In `services/tests/unified_dispatch_test.go` (about line 419) drop
   the local `weak` entry: the advertised role already provides the candidate.
**Verify:** `git grep -nw "EntryLocal\|localTiers\|tierEntryV1" -- services` prints nothing;
`git grep -n "len(declared.of(c)) == 0\|\.IsLocal()" -- services/shared services/nvpair-proxy`
prints nothing; `cd services/shared && go vet ./... && go test ./...` passes; in
`services/nvpair-proxy`, `services/nvpair-ui-broker`, `services/nvpair-service`,
`services/nvpair-engine-manager`: `go build ./... && go vet ./... && go test ./...` passes;
`cd services/tests && go test ./...` passes.

### Task 5 — The proxy routes any declared role name
**Files:** services/nvpair-proxy/tiers.go, services/nvpair-proxy/discovery.go, services/nvpair-proxy/http_handler.go, services/nvpair-proxy/routing.go, services/nvpair-proxy/tiers_test.go, services/nvpair-proxy/subscribed_test.go, services/nvpair-proxy/hybrid_alias_test.go
**Depends on:** 4 — same `tiers.go` and test files; relies on entries being gone
**Complexity:** judgment
**Do:**
1. In `tiers.go` add `func (f *facade) roleRequested(model string) bool`: true when
   `nodepolicy.IsDefaultRole(model)`; otherwise true when `nodepolicy.ValidateRoleName(model) == nil`
   and this node's policy declares `model` with a non-empty profile list, or any node in the
   discovery overlays of `f` and `f.host.enabledFacades()` advertises `model` in its roles (the
   same sources `declaredRoles` reads). Read the declarations without building the whole
   `declaredRoles` value, because this runs for every inference request.
2. `http_handler.go` (the `tier` computed for `observeRequested`, about line 748) and
   `routing.go` (non-hybrid branch, about line 684): replace `nodepolicy.IsDefaultRole(model)`
   with `f.roleRequested(model)`. The hybrid branch keeps `nodepolicy.ParseAlias`.
3. `discovery.go` `Node.filteredModels`: keep every role whose name passes `ValidateRoleName`
   and whose value is valid under phase 02's rules; delete the unknown-tier marker branch and its
   comment.
4. `configuredAliases` (rename its `peerTiers` parameter to `peerRoles`, and
   `declaredRoles.peerTiers` to `peerRoles`): for each default role, unchanged (bare name plus
   `-api` and `-local` when it has API entries, is declared here or by a peer); then each
   non-default name declared here or by a peer, bare only, sorted. Update its doc comment.
5. Tests: replace `TestUnknownTierDeclarationPreventsRouterFallback` in `subscribed_test.go` with
   a test that a peer declaring only `coder` keeps `coder` after `filteredModels` and that a
   declaration of `Coder` is dropped. Add to `tiers_test.go`: a request for `coder` declared by a
   peer resolves to that peer's profile; `coder` declared nowhere is not a role request (it takes
   the concrete-model path); `weak` declared nowhere is still a role request and gets 404
   `no eligible model for tier weak`; `X-PAIR-Tier: coder` is set on the committed attempt. Add
   to `hybrid_alias_test.go`: `/v1/models` lists `coder` (no `coder-api`, no `coder-local`) when
   a peer declares it, after the default aliases.
**Verify:** `git grep -n "marker" -- services/nvpair-proxy/discovery.go` prints nothing;
`cd services/nvpair-proxy && go vet ./... && go test ./...` passes;
`cd services/tests && go test -run 'TestUnifiedDispatchProcess|TestModelControlsRoutingAcrossProcesses|TestStrictModelRoutingAcrossProcesses|TestServiceRoutingListenerWithoutFacades' ./...` passes.

### Task 6 — Desktop role names are data, and the RoleEditor can add a role
**Files:** desktop/src/shared/types/node-roles.ts, desktop/src/shared/utils/node-roles.ts, desktop/src/ui/stores/node-configuration.store.ts, desktop/src/ui/components/RoutingSettings/RoleEditor.tsx, desktop/src/ui/components/RoutingSettings/BulkNodeConfiguration.tsx, desktop/src/ui/components/NodeList/NodeDeclaredRoles.tsx, desktop/src/shared/types/model-config.ts, desktop/src/electron/service-bridge/engine-settings.ts, desktop/src/shared/utils/node-policy-model-settings.ts, desktop/src/ui/components/ModelSettings/ModelSettingsCopy.tsx, desktop/src/ui/components/ModelSettings/ModelLoadSettings.tsx, desktop/tests/modular/node-configuration.test.ts, desktop/tests/modular/node-configuration-bulk.test.ts, desktop/tests/modular/node-policy.test.ts, desktop/tests/modular/node-policy-precision.test.ts, desktop/tests/modular/model-settings-copy.test.tsx, desktop/tests/modular/model-load-settings.test.tsx, desktop/tests/modular/node-policy-model-settings.test.ts, desktop/tests/modular/remote-management.test.tsx
**Depends on:** 1, 4 — mirrors Task 1's grammar; the Suggest flow it removes read the local entries Task 4 retires
**Complexity:** judgment
**Do:**
1. `shared/types/node-roles.ts`: delete `RoleTier`; `NodeRoles` is keyed by `string` (value type
   as phase 02 left it).
2. `shared/utils/node-roles.ts`: delete `ROLE_TIERS`; add, with doc comments naming the Go
   counterparts: `DEFAULT_ROLES = ['weak', 'medium', 'strong'] as const` (`nodepolicy.DefaultRoles`),
   `roleNameError(name: string): string | undefined` implementing exactly Task 1's
   `ValidateRoleName` (same pattern, same suffix rule; messages in plain English),
   `isReservedAlias(id: string): boolean` (a default role, bare or with `-api`/`-local`;
   `nodepolicy.IsReservedAlias`) and `roleNames(...maps: NodeRoles[]): string[]` (the defaults
   in order, then every other key of the maps sorted, no duplicates). `parseNodeRoles` keeps
   every key with no `roleNameError` and a valid value; `sameNodeRoles` compares the union of
   both maps' keys.
3. `node-configuration.store.ts`: `withRoles` iterates the draft's keys and omits roles with no
   profile; `roleDifferences` and `setRole` take `role: string`; delete `suggestRoles` and every
   use of it.
4. `RoleEditor.tsx`: delete `TIER_LABELS`, the Suggest button, `suggesting`/`noSuggestion` state
   and their copy. Rows: `roleNames` over the draft and every discovered node's advertised
   roles, defaults first (read the advertisements from the same store `NodeDeclaredRoles.tsx` reads); each row's
   label is the role name. Add a text field (aria-label `New role name`) and a button `Add role`:
   it shows `roleNameError`, refuses a name already listed, refuses a name equal to one of this
   node's profile ids (message `A profile on this node already uses that name.`), and otherwise adds
   an unset row held in component state until a profile is chosen. Replace the authority copy with:
   `A router sends a role's requests only to nodes that declare it. Roles left unset are not served by this node.`
5. `BulkNodeConfiguration.tsx` and `NodeDeclaredRoles.tsx`: iterate `roleNames(...)` over the maps
   they show instead of `ROLE_TIERS`; a `RoleTier` parameter becomes `string`.
6. `shared/types/model-config.ts:75`: `roles?: Record<string, boolean>`.
   `electron/service-bridge/engine-settings.ts` `modelId` and
   `shared/utils/node-policy-model-settings.ts` (about line 233): use `isReservedAlias`.
7. `ModelLoadSettings.tsx`: list role rows from `roleNames` over the policy's roles (defaults
   first) instead of the literal arrays. `ModelSettingsCopy.tsx` (when the file still exists after
   phase 02): hold role membership in a separate `roles: string[]` selection, not in the field
   union with `served`/`resident`/`binding`, so a role named `served` cannot collide; list roles
   with `roleNames`.
8. Tests: update the listed tests for the type and helper changes; delete `suggestRoles` tests;
   add tests that `parseNodeRoles` keeps `coder` and drops `Coder` and `coder-api`, that
   `roleNameError` matches every case in Task 1's Go table, that `withRoles` omits an empty role,
   and that the RoleEditor adds `coder`, rejects `coder-local` with the suffix message and
   rejects a profile id.
**Verify:** `git grep -n "ROLE_TIERS\|RoleTier\|TIER_LABELS\|suggestRoles\|'weak' | 'medium' | 'strong'" -- desktop/src`
prints nothing; in `desktop/`: `npm run lint && npm run typecheck && npm run test:unit && npm run dead-code:check`
passes.

### Task 7 — Routing history records the requested role as a string
**Files:** services/nvpair-proxy/routing_facts.go, services/nvpair-proxy/routing_events.go, services/nvpair-proxy/routing_facts_test.go, services/nvpair-proxy/routing_events_test.go, services/shared/nodepolicy/routing_wire.go, services/shared/nodepolicy/routing_wire_test.go, services/shared/nodepolicy/testdata/routing-request.json, services/nvpair-tui/ui/routingwire_test.go, services/nvpair-tui/ui/routingview_test.go, desktop/src/shared/types/routing.ts, desktop/tests/modular/routing-projection.test.ts
**Depends on:** 1, 6 — uses `ValidateRoleName` and `MaxRoleNameBytes`; the desktop validator imports `roleNameError` from `node-roles.ts`
**Complexity:** judgment
**Do:**
1. `routing_facts.go`: `routingRequestedDescriptor` becomes
   `{model [512]byte; length uint16; role [nodepolicy.MaxRoleNameBytes]byte; roleLen uint8; mode uint8}`;
   `requestedDescriptor` copies `tier` into `role` when `nodepolicy.ValidateRoleName(tier) == nil`
   and leaves it empty otherwise; delete the `weak`/`medium`/`strong` switch.
2. `routing_events.go`: delete `routingHistoryTiers`; `encodeRecord` writes
   `w.quote(f.requested.role[:f.requested.roleLen])` for `requested.tier`.
3. `routing_wire.go` record validation: replace `historyEnum(r.Requested.Tier, "|weak|medium|strong")`
   with `(r.Requested.Tier == "" || ValidateRoleName(r.Requested.Tier) == nil)`. Do not add
   fields to `RoutingHistoryRecord`, `RoutingHistoryRecordHeader`, its mapping or the header
   initializer: `Requested.Tier` is already a string in all of them.
4. `testdata/routing-request.json`: change the first case's `requested.tier` from `weak` to
   `coder`. If a TUI test in Files fails only because it asserts that fixture value, update the
   assertion; make no other TUI change (the TUI decodes the string reflectively).
5. `desktop/src/shared/types/routing.ts`: `requested.tier` is typed `string`; remove `tier` from
   the enumeration table (about line 403); the validator (about line 596) accepts `''` or a value
   with no `roleNameError`.
6. Tests: Go cases for `requestedDescriptor` with `coder`, a 64-byte role, and an invalid role
   (stored empty); record validation accepts `coder` and rejects `Coder`; extend the existing
   maximum-size record cases (a 512-byte model, as in `routing-projection.test.ts` about line 314)
   with a 64-byte role and confirm they still pass. If a maximum-size case no longer fits
   `jsonrpc.HistoryRecordFrameBytes`, stop and report; do not change frame limits in this phase.
   `routing-projection.test.ts`: the `requested.tier` mutation still produces an invalid record;
   add an accepted `coder` record and a rejected `Coder` record.
**Verify:** `git grep -n "routingHistoryTiers\||weak|medium|strong" -- services` prints nothing;
`cd services/nvpair-proxy && go test ./...`, `cd services/shared && go test ./nodepolicy/...`,
`cd services/nvpair-tui && go vet ./... && go test ./...`,
`cd services/nvpair-ui-broker && go test ./...`, `cd services/nvpair-service && go test ./...` and
`cd services/tests && go test -run 'TestRoutingHistoryProductionChain' ./...` pass; in `desktop/`:
`npm run typecheck && npm run test:unit` passes.

### Task 8 — Docs describe user-defined roles, fallback lists and retired router entries
**Files:** services/nvpair-proxy/README.md, services/nvpair-proxy/spec.md, services/nvpair-proxy/proxy.go, services/nvpair-workload-manager/README.md, services/nvpair-workload-manager/spec.md, services/nvpair-workload-manager/workload.go, services/nvpair-ui-broker/README.md, docs/hybrid-routing.mdx, docs/node-availability.mdx, desktop/docs/services-parity.md, desktop/docs/services-api.md
**Depends on:** 3, 5, 6, 7 — documents their final behaviour
**Complexity:** mechanical
**Do:**
1. `services/nvpair-proxy/README.md` "Tiers" and "Hybrid aliases" bullets (about lines 661-700)
   and `spec.md` tier sections (about lines 1838-1900, history at about 1233): a role request is
   a default name or any role name a node declares; role-name grammar; candidates come only from
   nodes declaring the role; search order is the role then `tierPolicy.roleFallback[role]`; the
   default lists; reserved aliases and API entries exist only for `weak`, `medium`, `strong`; a
   role name wins over a concrete model of the same name on another node; `requested.tier`
   records the role string. Delete the text about router tier entries, node-wide authority,
   `allowStronger` and `allowWeaker`.
2. Comments: `services/nvpair-proxy/proxy.go` (about line 192) and
   `services/nvpair-workload-manager/workload.go` (about line 81) say "the role a request asked
   for" instead of listing the three names; `services/nvpair-workload-manager/README.md`
   (about line 123) and `spec.md` (about line 81) likewise.
3. `services/nvpair-ui-broker/README.md`: the `policy:get` row and the paragraph near line 1267
   say the peer projection has no `tiers` and that a peer write carrying `tiers` is refused.
4. `docs/hybrid-routing.mdx:13`: the aliases exist for the default roles only.
   `docs/node-availability.mdx` (about line 227): the example becomes "a request for a role,
   such as `strong`". `desktop/docs/services-parity.md:189`: "role/fallback order".
5. Run `npm run service-contracts:check` in `desktop/`; only when it reports
   `desktop/docs/services-api.md` stale, run `npm run service-contracts:write` (never edit that
   file by hand).
**Verify:** `git grep -n -i "allowStronger\|allowWeaker\|authoritative for itself" -- services docs desktop/docs ':!services/shared/nodepolicy/policy_test.go'`
prints nothing; in `desktop/`: `npm run service-contracts:check` passes; at the repo root
`node scripts/spdx-headers.mjs` passes; `cd services/nvpair-proxy && go build ./...` and
`cd services/nvpair-workload-manager && go build ./...` pass.

## Verification (whole-plan)
Local gate (all must pass on the final commit of the phase; record each command, its exit code
and the last `ok`/`FAIL` lines per module in the close-out entry):
1. Go, from the repo root:
   `for m in shared nvpair-proxy nvpair-ui-broker nvpair-tui nvpair-service nvpair-engine-manager nvpair-node-scanner nvpair-workload-manager nvpair-manual-nodes tests; do (cd services/$m && go build ./... && go vet ./... && go test ./...) || echo "FAIL $m"; done`
   prints no `FAIL` line. This covers nodepolicy, modelconfig, proxy, broker, TUI and
   `services/tests` (`TestUnifiedDispatchProcess` included).
2. Desktop, in `desktop/`: `npm run lint && npm run typecheck && npm run test:unit && npm run dead-code:check && npm run service-contracts:check` passes.
3. Repo root: `node scripts/spdx-headers.mjs` and `make check` pass.
4. Greps: `git grep -nw "TierOrder\|IsTier\|EntryLocal\|ROLE_TIERS\|RoleTier" -- services desktop/src`
   prints nothing.
A failure called pre-existing counts only when the same command fails the same way on the
recorded starting revision in a separate scratch worktree; otherwise the gate is not passed.

Acceptance on the owner's machines (Mac, Joes, MSI; external, recorded in the vault):
1. Install the phase build on every node. For each node, the desktop's Routing settings shows no
   recovery notice (its `policy:get` reports `recoveryRequired: false`).
2. From the Mac, open Joes's RoleEditor, add role `coder`, assign one of Joes's profiles, save.
   Joes's card shows `coder`.
3. From the Mac, send to the Mac's OpenAI-compatible PAIR endpoint (the client URL the desktop
   shows): `curl -sS -D - <endpoint>/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"coder","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'`.
   Pass: HTTP 200, `X-PAIR-Node` is Joes's node id, `X-PAIR-Tier: coder`, `X-PAIR-Model` is the
   assigned profile's served id. `<endpoint>/v1/models` lists `coder` and no `coder-api`.
4. The Mac's routing history (desktop Routing history table and the TUI routing view) shows the
   request with requested model `coder` served by Joes.
5. Fallback: note which nodes declare `weak` and their profile lists, then remove `weak` from
   each of them through the RoleEditor. In the Mac's Advanced policy JSON set
   `tierPolicy.roleFallback.weak` to `["medium"]` and save. A `weak` request from the Mac returns
   200 with `X-PAIR-Tier: weak` and `X-PAIR-Model` naming a profile some node declares for
   `medium`. Set `roleFallback.weak` to `[]`: the same request returns 404 with
   `no eligible model for tier weak`. Restore the `weak` declarations and set the Mac's
   `roleFallback.weak` back to `["medium", "strong"]`.
6. On each router the owner uses (Mac, MSI, Joes), read `tierPolicy.roleFallback` and confirm it
   matches what the owner wants (the old `allowStronger`/`allowWeaker` values are no longer read).

## Close-out
- In `plans/2026-10-31-expectation-gaps/00-index.md`, append ` — done <sha>` to the phase 03 row,
  and in this file append ` — done <sha>` to each task heading as it lands (see Note lifecycle).
- Add a vault entry to `/home/user/wasabi/4. Projects/Personal-AI-Router/WORKLOG.md` with: the
  commit range, the local gate commands with exit codes and final `ok`/`FAIL` lines, any file
  substitution made because phase 04 moved a desktop file, the acceptance results (steps not yet run are
  marked "pending on machines"), and the remaining limitations: no fallback-list editor in the desktop (phase 04), the
  history table does not render the requested role (phase 04), role names cannot be checked for
  collisions with concrete model ids on other nodes.
- Hand-over: phase 06 (`06-routing-and-apps.md`) relies on `tierRank` meaning "position in the
  role's fallback chain" and on role names being strings; phase 04 builds its role surfaces on
  `roleNames`, `roleNameError` and `DEFAULT_ROLES` from `desktop/src/shared/utils/node-roles.ts`.

## Do not
- Do not refactor beyond what the tasks specify; in particular do not rename `TierPolicy`,
  `tierSearchOrder`, `tierRank`, `requested.tier`, `X-PAIR-Tier` or the 404 text.
- Do not touch the smart-routing scorer, `candidateChoiceLocked`, `reserveRoundPrepared` or the
  scheduler.
- Do not extend hybrid/API routing, reserved aliases or `routing.tiers` to non-default role names.
- Do not add compatibility shims: no reading of `allowStronger`/`allowWeaker`, no local-entry
  routing path, no version bump, no capability marker.
- Do not change the role-to-profile-list shape from phase 02 or implement any missing phase 02 work.
- Do not edit `desktop/docs/services-api.md` by hand, and do not edit anything in `/home/user/pair-join33`.
- Do not change history frame limits in `services/shared/jsonrpc`.

## If stuck
- Compile errors after removing a shared symbol: `git grep -nw <symbol> -- services` and update
  every caller in the same task; this is safe to repeat.
- A test that fails because it encoded the old closed vocabulary or router-entry behaviour:
  classify it under the categories in "Context the executor needs" and edit it as that category
  says. A test that fails for another reason is a real regression: fix the code, not the test.
- Stop and report (do not improvise) when: an entry condition is missing; `modelconfig` cannot
  import `nodepolicy` without a cycle; a maximum-size history record no longer fits the frame;
  phase 02's role or advertisement shape makes a step here impossible as written; or the gate
  still fails after two fix attempts on the same failure.

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
