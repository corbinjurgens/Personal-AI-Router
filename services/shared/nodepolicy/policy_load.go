// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
)

// ParseStored is exclusively for disk load. It removes invalid model entries
// before strict nested decoding and repairs their dependent references in
// memory. Counts have fixed category keys and never contain document text.
// Callers must not persist this repair or assign a new revision on load.
func ParseStored(data []byte) (Policy, map[string]int, error) {
	container, err := readPolicyModelContainer(data)
	if err != nil {
		return Policy{}, nil, errors.New("stored policy is invalid")
	}
	// Retired engine bindings have no authority. Drop them only on disk load,
	// preserving the order and original values of every retained member.
	container.members = slices.DeleteFunc(container.members, func(member modelContainerMember) bool {
		return strings.EqualFold(member.key, "engineBindings")
	})
	data, err = container.marshal()
	if err != nil {
		return Policy{}, nil, errors.New("stored policy is invalid")
	}
	if err := checkModelListKeys(data); err != nil {
		return Policy{}, nil, errors.New("stored policy is invalid")
	}
	// Reserved public keys are fatal even when model sanitation would remove
	// their profiles. Check both explicit keys and disk-only model-key repair.
	if err := checkStoredReservedProfileKeys(data); err != nil {
		var reserved *reservedProfileKeyError
		if errors.As(err, &reserved) {
			return Policy{}, nil, err
		}
		return Policy{}, nil, errors.New("stored policy is invalid")
	}
	data, droppedRoles, err := repairStoredRoles(data)
	if err != nil {
		return Policy{}, nil, errors.New("stored policy is invalid")
	}
	if err := checkStoredModelShapes(data); err != nil {
		return Policy{}, nil, errors.New("stored policy is invalid")
	}
	clean, counts, err := filterPolicyModels(data, true)
	if err == nil {
		counts["roles"] += droppedRoles
		clean, err = repairStoredProfileKeys(clean)
		if err != nil {
			var collision *profileKeyRepairError
			if errors.As(err, &collision) {
				return Policy{}, nil, err
			}
			return Policy{}, nil, errors.New("stored policy is invalid")
		}
		var p Policy
		repairs, repairErr := removedStoredModels(data, counts)
		if repairErr != nil {
			return Policy{}, nil, errors.New("stored policy is invalid")
		}
		p, err = parsePolicy(clean, repairs)
		if err == nil {
			return p, counts, nil
		}
	}
	return Policy{}, nil, errors.New("stored policy is invalid")
}

func checkStoredReservedProfileKeys(data []byte) error {
	container, err := readPolicyModelContainer(data)
	if err != nil {
		return err
	}
	raw := container.fields[modelField(container.fields, "profiles")]
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	var entries []json.RawMessage
	if err := json.Unmarshal(raw, &entries); err != nil {
		return err
	}
	for _, entry := range entries {
		fields, err := modelObject(entry)
		if err != nil {
			return err
		}
		key := fields[modelField(fields, "key")]
		if key == nil {
			key = fields[modelField(fields, "model")]
		}
		var name string
		if err := json.Unmarshal(key, &name); err != nil {
			return err
		}
		if IsReservedProfileKey(name) {
			return ValidateProfileKey(name)
		}
	}
	return nil
}

// repairStoredRoles discards object-shaped declarations before any list-shape
// validation. Roles never create profiles or acquire automatic memberships.
func repairStoredRoles(data []byte) ([]byte, int, error) {
	container, err := readPolicyModelContainer(data)
	if err != nil {
		return nil, 0, err
	}
	key := modelField(container.fields, "roles")
	raw := bytes.TrimSpace(container.fields[key])
	if len(raw) == 0 || raw[0] != '{' {
		return data, 0, nil
	}
	var roles map[string]json.RawMessage
	if err := json.Unmarshal(raw, &roles); err != nil {
		return nil, 0, err
	}
	container.fields[key] = json.RawMessage("[]")
	clean, err := container.marshal()
	return clean, len(roles), err
}

// A removed model must not conceal an unrelated type, engine, bound, or
// reference error. Substitute a harmless model solely to run those existing
// validators; the substitution is never returned or persisted.
func checkStoredModelShapes(data []byte) error {
	container, err := readPolicyModelContainer(data)
	if err != nil {
		return err
	}
	fields := container.fields
	version := SchemaVersion
	if raw := fields[modelField(fields, "version")]; len(raw) > 0 {
		if err := json.Unmarshal(raw, &version); err != nil {
			return err
		}
	}
	routing := defaultRouting()
	if raw := fields[modelField(fields, "routing")]; len(raw) > 0 && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		container, err := readModelContainer(raw, "deployments", "tiers")
		if err != nil {
			return err
		}
		obj := container.fields
		if raw := obj[modelField(obj, "accounts")]; len(raw) > 0 {
			if err := json.Unmarshal(raw, &routing.Accounts); err != nil {
				return err
			}
		}
		if raw := obj[modelField(obj, "deployments")]; len(raw) > 0 {
			if err := json.Unmarshal(raw, &routing.Deployments); err != nil {
				return err
			}
		}
		if raw := obj[modelField(obj, "tiers")]; len(raw) > 0 && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			families := map[string]bool{}
			for _, deployment := range routing.Deployments {
				families[deployment.Family] = true
			}
			tiers, err := modelObject(raw)
			if err != nil {
				return err
			}
			for name, rawTier := range tiers {
				if err := ValidateRoleName(name); err != nil {
					return err
				}
				if _, err := modelObject(rawTier); err != nil {
					return err
				}
				var tr TierRouting
				if err := json.Unmarshal(rawTier, &tr); err != nil {
					return err
				}
				// Validate original permissions before an exhausted-family repair
				// can clear them. Original deployment families still exist here.
				if err := tr.validate(families); err != nil {
					return err
				}
			}
		}
	}
	accounts := map[string]bool{}
	for _, account := range routing.Accounts {
		accounts[account.ID] = true
	}
	check := func(name string, entry json.RawMessage) error {
		ok, err := filterModelEntry(entry, true)
		if err != nil || ok && name != "tiers" {
			return err
		}
		obj, err := modelObject(entry)
		if err != nil {
			return err
		}
		if !ok {
			obj[modelField(obj, "model")] = json.RawMessage(`"stored-model-check"`)
		}
		if name == "profiles" {
			repairStoredProfileFields(obj)
		}
		raw, err := json.Marshal(obj)
		if err != nil {
			return err
		}
		switch name {
		case "profiles":
			var pr Profile
			if err := json.Unmarshal(raw, &pr); err != nil {
				return err
			}
			p := Default()
			p.Profiles = []Profile{pr}
			return p.Validate()
		case "deployments":
			var d Deployment
			if err := json.Unmarshal(raw, &d); err != nil {
				return err
			}
			return validateDeployment(d, accounts)
		case "tiers":
			var e TierEntry
			if err := json.Unmarshal(raw, &e); err != nil {
				return err
			}
			if e.Kind == "" && version == legacySchemaVersion {
				e.Kind = EntryLocal
			}
			// Retired stored entries are discarded, but must not conceal the
			// unrelated malformed fields this shape preflight already refused.
			if e.Kind == EntryLocal {
				if err := checkModelRef(e.Engine, e.Model, e.ContextTokens, e.Capabilities); err != nil {
					return err
				}
				if e.Deployment != "" {
					return errors.New("a local entry cannot name a deployment")
				}
				return nil
			}
			return routing.validateTierRefs("stored-model-check", []TierEntry{e})
		case "servedModels", "residentModels":
			_, err := decodeModelList(name, append(append([]byte{'['}, raw...), ']'))
			return err
		}
		return nil
	}
	for _, name := range []string{"profiles", "servedModels", "residentModels", "deployments", "roles", "tiers"} {
		container := fields
		if name == "deployments" {
			raw := fields[modelField(fields, "routing")]
			if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
				continue
			}
			routing, err := readModelContainer(raw, "deployments", "tiers")
			if err != nil {
				return err
			}
			container = routing.fields
		}
		raw := container[modelField(container, name)]
		if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			continue
		}
		if name == "roles" {
			if _, _, err := filterRoleMemberships(raw, nil); err != nil {
				return err
			}
			continue
		}
		if name == "tiers" {
			members, err := modelObject(raw)
			if err != nil {
				return err
			}
			for tier, entry := range members {
				if err := ValidateRoleName(tier); err != nil {
					return err
				}
				var entries []json.RawMessage
				if err := json.Unmarshal(entry, &entries); err != nil {
					return err
				}
				seen := map[string]bool{}
				for _, e := range entries {
					var decoded TierEntry
					if err := json.Unmarshal(e, &decoded); err != nil {
						return err
					}
					if version != legacySchemaVersion {
						key := entryKey(decoded)
						if seen[key] {
							return errors.New("duplicate tier entry")
						}
						seen[key] = true
					}
					obj, err := modelObject(e)
					if err != nil {
						return err
					}
					var kind string
					if raw := obj[modelField(obj, "kind")]; len(raw) > 0 {
						if err := json.Unmarshal(raw, &kind); err != nil {
							return err
						}
					}
					if kind == "" || kind == string(EntryLocal) {
						if err := check(name, e); err != nil {
							return err
						}
					} else if err := routing.validateTierRefs(tier, []TierEntry{decoded}); err != nil {
						return err
					}
				}
			}
			continue
		}
		var entries []json.RawMessage
		if err := json.Unmarshal(raw, &entries); err != nil {
			return err
		}
		for _, entry := range entries {
			if err := check(name, entry); err != nil {
				return err
			}
		}
	}
	return nil
}

type profileKeyRepairError struct {
	pairs []string
}

func (e *profileKeyRepairError) Error() string {
	return fmt.Sprintf("Profiles share a key after repair: %s. Rename one in this node's policy file before installing.", strings.Join(e.pairs, ", "))
}

// repairStoredProfileFields is part of disk-only shape repair. In particular,
// raw policy writes never acquire a key from their model as a fallback.
func repairStoredProfileFields(fields map[string]json.RawMessage) bool {
	repairedKey := fields[modelField(fields, "key")] == nil
	if repairedKey {
		fields["key"] = fields[modelField(fields, "model")]
	}
	if fields[modelField(fields, "displayName")] == nil {
		if name := fields[modelField(fields, "name")]; name != nil {
			fields["displayName"] = name
		}
	}
	delete(fields, modelField(fields, "name"))
	delete(fields, modelField(fields, "quantization"))
	return repairedKey
}

func repairStoredProfileKeys(data []byte) ([]byte, error) {
	container, err := readPolicyModelContainer(data)
	if err != nil {
		return nil, err
	}
	field := modelField(container.fields, "profiles")
	raw, exists := container.fields[field]
	if !exists || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return data, nil
	}
	var entries []json.RawMessage
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, err
	}
	type identity struct {
		key, pair string
		repaired  bool
	}
	identities := make([]identity, 0, len(entries))
	for i, entry := range entries {
		fields, err := modelObject(entry)
		if err != nil {
			return nil, err
		}
		repaired := repairStoredProfileFields(fields)
		entries[i], err = json.Marshal(fields)
		if err != nil {
			return nil, err
		}
		var pr Profile
		if err := json.Unmarshal(entries[i], &pr); err != nil {
			return nil, err
		}
		identities = append(identities, identity{pr.Key, pr.Engine + "/" + pr.Model, repaired})
	}
	for _, candidate := range identities {
		var pairs []string
		repaired := false
		for _, other := range identities {
			if candidate.key == other.key {
				pairs = append(pairs, other.pair)
				repaired = repaired || other.repaired
			}
		}
		if len(pairs) > 1 && repaired {
			return nil, &profileKeyRepairError{pairs: pairs}
		}
	}
	container.fields[field], err = json.Marshal(entries)
	if err != nil {
		return nil, err
	}
	return container.marshal()
}

// modelObject retains raw values and refuses ambiguous effective keys before
// sanitation can accidentally reveal a shadowed member. It is used only at
// the model containers this prerequisite owns.
func modelObject(data []byte) (map[string]json.RawMessage, error) {
	return peerFields(data)
}

type modelContainerMember struct {
	key   string
	value json.RawMessage
	owned bool
}

// modelContainer keeps legacy members in document order, including repeated
// nonmodel fields. Only model-owned container names reject effective duplicates.
// fields provides a last-wins view for inspection without reordering the output.
type modelContainer struct {
	fields  map[string]json.RawMessage
	members []modelContainerMember
}

func readPolicyModelContainer(data []byte) (*modelContainer, error) {
	return readModelContainer(data, "profiles", "roles", "tiers", "servedModels", "residentModels", "routing")
}

func readModelContainer(data []byte, owned ...string) (*modelContainer, error) {
	d := json.NewDecoder(bytes.NewReader(data))
	if tok, err := d.Token(); err != nil || tok != json.Delim('{') {
		return nil, errors.New("policy must be a JSON object")
	}
	c := &modelContainer{fields: map[string]json.RawMessage{}}
	for d.More() {
		tok, err := d.Token()
		if err != nil {
			return nil, err
		}
		key, ok := tok.(string)
		if !ok {
			return nil, errors.New("invalid policy key")
		}
		modelOwned := slices.ContainsFunc(owned, func(name string) bool { return strings.EqualFold(name, key) })
		for prior := range c.fields {
			if strings.EqualFold(prior, key) {
				if modelOwned {
					return nil, errors.New("policy has a duplicate field")
				}
				delete(c.fields, prior)
			}
		}
		var raw json.RawMessage
		if err := d.Decode(&raw); err != nil {
			return nil, err
		}
		c.fields[key] = raw
		c.members = append(c.members, modelContainerMember{key: key, value: raw, owned: modelOwned})
	}
	if _, err := d.Token(); err != nil {
		return nil, err
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, errors.New("policy must be one JSON object")
	}
	return c, nil
}

func (c *modelContainer) marshal() ([]byte, error) {
	var out bytes.Buffer
	out.WriteByte('{')
	for i, member := range c.members {
		if i > 0 {
			out.WriteByte(',')
		}
		key, err := json.Marshal(member.key)
		if err != nil {
			return nil, err
		}
		out.Write(key)
		out.WriteByte(':')
		value := member.value
		if member.owned {
			value = c.fields[member.key]
		}
		out.Write(value)
	}
	out.WriteByte('}')
	return out.Bytes(), nil
}

func modelField(fields map[string]json.RawMessage, name string) string {
	for key := range fields {
		if strings.EqualFold(key, name) {
			return key
		}
	}
	return name
}

// filterModelEntry checks only model syntax. Other malformed fields continue
// through the ordinary strict parser, even when model sanitation is enabled.
func filterModelEntry(raw json.RawMessage, remove bool) (bool, error) {
	fields, err := modelObject(raw)
	if err != nil {
		return false, err
	}
	key := modelField(fields, "model")
	if fields[key] == nil {
		return false, errors.New("model is required")
	}
	if err := ValidateModelIdentifierJSON(fields[key]); err != nil {
		// Wrong JSON types/missing fields are malformed shapes, not model names.
		value := bytes.TrimSpace(fields[key])
		if len(value) == 0 || value[0] != '"' {
			return false, errors.New("invalid model field")
		}
		if remove {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

func filterModelArray(raw json.RawMessage, remove bool, required ...string) (json.RawMessage, int, error) {
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return raw, 0, nil
	}
	var entries []json.RawMessage
	if json.Unmarshal(raw, &entries) != nil {
		return nil, 0, errors.New("invalid model list")
	}
	kept := make([]json.RawMessage, 0, len(entries))
	for _, entry := range entries {
		if len(required) > 0 {
			fields, err := modelObject(entry)
			if err != nil {
				return nil, 0, err
			}
			if len(fields) != len(required) {
				return nil, 0, errors.New("invalid model entry shape")
			}
			for _, key := range required {
				if fields[key] == nil {
					return nil, 0, errors.New("invalid model entry shape")
				}
			}
		}
		ok, err := filterModelEntry(entry, remove)
		if err != nil {
			return nil, 0, err
		}
		if ok {
			kept = append(kept, entry)
		}
	}
	encoded, err := json.Marshal(kept)
	return encoded, len(entries) - len(kept), err
}

func pruneProfileKeys(keys []string, dropped map[string]bool) ([]string, int) {
	kept := make([]string, 0, len(keys))
	for _, key := range keys {
		if !dropped[key] {
			kept = append(kept, key)
		}
	}
	return kept, len(keys) - len(kept)
}

// filterRoleMemberships removes only references to profiles actually dropped by
// sanitation. Unrelated missing keys still reach ordinary policy validation.
func filterRoleMemberships(raw json.RawMessage, dropped map[string]bool) (json.RawMessage, int, error) {
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return raw, 0, nil
	}
	var entries []json.RawMessage
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, 0, err
	}
	removed := 0
	for i, entry := range entries {
		fields, err := modelObject(entry)
		if err != nil {
			return nil, 0, err
		}
		var role Role
		if err := json.Unmarshal(entry, &role); err != nil {
			return nil, 0, err
		}
		kept, count := pruneProfileKeys(role.Profiles, dropped)
		removed += count
		fields[modelField(fields, "profiles")], err = json.Marshal(kept)
		if err != nil {
			return nil, 0, err
		}
		entries[i], err = json.Marshal(fields)
		if err != nil {
			return nil, 0, err
		}
	}
	clean, err := json.Marshal(entries)
	return clean, removed, err
}

func filterPolicyModels(data []byte, remove bool) ([]byte, map[string]int, error) {
	container, err := readPolicyModelContainer(data)
	if err != nil {
		return nil, nil, err
	}
	fields := container.fields
	counts := map[string]int{}
	version := SchemaVersion
	if raw := fields[modelField(fields, "version")]; len(raw) > 0 {
		if err := json.Unmarshal(raw, &version); err != nil {
			return nil, nil, err
		}
	}
	for _, name := range []string{"profiles", "servedModels", "residentModels"} {
		key := modelField(fields, name)
		if raw, ok := fields[key]; ok {
			var required []string
			if name != "profiles" {
				required = []string{"engine", "model"}
			}
			clean, count, err := filterModelArray(raw, remove, required...)
			if err != nil {
				return nil, nil, err
			}
			fields[key], counts[name] = clean, count
		}
	}
	for _, name := range []string{"roles", "tiers"} {
		key := modelField(fields, name)
		raw, exists := fields[key]
		if !exists || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			continue
		}
		if name == "roles" {
			clean, _, err := filterRoleMemberships(raw, nil)
			if err != nil {
				return nil, nil, err
			}
			fields[key] = clean
			continue
		}
		members, err := modelObject(raw)
		if err != nil {
			return nil, nil, err
		}
		for tier, entry := range members {
			var entries []json.RawMessage
			if json.Unmarshal(entry, &entries) != nil {
				return nil, nil, errors.New("invalid tier list")
			}
			kept := make([]json.RawMessage, 0, len(entries))
			for _, e := range entries {
				obj, err := modelObject(e)
				if err != nil {
					return nil, nil, err
				}
				var kind string
				if raw, ok := obj[modelField(obj, "kind")]; ok && json.Unmarshal(raw, &kind) != nil {
					return nil, nil, errors.New("invalid tier kind")
				}
				if kind == string(EntryLocal) || kind == "" && version == legacySchemaVersion {
					if remove {
						counts[name]++
						continue
					}
				} else if kind == "" {
					ok, err := filterModelEntry(e, remove)
					if err != nil {
						return nil, nil, err
					}
					if !ok {
						counts[name]++
						continue
					}
				}
				kept = append(kept, e)
			}
			if entries != nil {
				members[tier], err = json.Marshal(kept)
			}
			if err != nil {
				return nil, nil, err
			}
		}
		fields[key], err = json.Marshal(members)
		if err != nil {
			return nil, nil, err
		}
	}
	key := modelField(fields, "routing")
	if raw, ok := fields[key]; ok && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		routingContainer, err := readModelContainer(raw, "deployments", "tiers")
		if err != nil {
			return nil, nil, err
		}
		routing := routingContainer.fields
		dkey := modelField(routing, "deployments")
		if entries, ok := routing[dkey]; ok {
			clean, count, err := filterModelArray(entries, remove)
			if err != nil {
				return nil, nil, err
			}
			routing[dkey], counts["deployments"] = clean, count
		}
		fields[key], err = routingContainer.marshal()
		if err != nil {
			return nil, nil, err
		}
	}
	clean, err := container.marshal()
	return clean, counts, err
}

type storedModelRepairs struct {
	counts      map[string]int
	profiles    map[ModelRef]bool
	profileKeys map[string]bool
	served      map[ModelRef]bool
	deployments map[string]bool
	families    map[string]bool
}

func removedStoredModels(data []byte, counts map[string]int) (*storedModelRepairs, error) {
	r := &storedModelRepairs{counts: counts, profiles: map[ModelRef]bool{}, profileKeys: map[string]bool{}, served: map[ModelRef]bool{}, deployments: map[string]bool{}, families: map[string]bool{}}
	container, err := readPolicyModelContainer(data)
	if err != nil {
		return nil, err
	}
	fields := container.fields
	for _, name := range []string{"profiles", "servedModels", "residentModels", "deployments"} {
		container := fields
		if name == "deployments" {
			raw := fields[modelField(fields, "routing")]
			if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
				continue
			}
			routing, err := readModelContainer(raw, "deployments", "tiers")
			if err != nil {
				return nil, err
			}
			container = routing.fields
		}
		raw := container[modelField(container, name)]
		if len(raw) == 0 {
			continue
		}
		var entries []json.RawMessage
		if err := json.Unmarshal(raw, &entries); err != nil {
			return nil, err
		}
		seen := map[string]bool{}
		models := map[ModelRef]bool{}
		for _, entry := range entries {
			if name == "profiles" {
				fields, err := modelObject(entry)
				if err != nil {
					return nil, err
				}
				repairStoredProfileFields(fields)
				entry, err = json.Marshal(fields)
				if err != nil {
					return nil, err
				}
			}
			var ref struct{ Engine, Model, Key, ID, Family string }
			if err := json.Unmarshal(entry, &ref); err != nil {
				return nil, err
			}
			key := ref.Key
			if name == "deployments" {
				key = ref.ID
			}
			if name == "servedModels" || name == "residentModels" {
				key = ref.Engine + "\x00" + ref.Model
			}
			if seen[key] {
				return nil, errors.New("duplicate model entry")
			}
			seen[key] = true
			if name == "profiles" {
				model := ModelRef{Engine: ref.Engine, Model: ref.Model}
				if models[model] {
					return nil, errors.New("duplicate profile model")
				}
				models[model] = true
			}
			ok, err := filterModelEntry(entry, true)
			if err != nil {
				return nil, err
			}
			if ok {
				continue
			}
			switch name {
			case "profiles":
				r.profiles[ModelRef{Engine: ref.Engine, Model: ref.Model}] = true
				r.profileKeys[ref.Key] = true
			case "servedModels":
				r.served[ModelRef{Engine: ref.Engine, Model: ref.Model}] = true
			case "deployments":
				r.deployments[ref.ID], r.families[ref.Family] = true, true
			}
		}
	}
	return r, nil
}

func repairStoredModelReferences(p *Policy, repairs *storedModelRepairs) {
	// Older saved pauses do not distinguish a configured pause from a Halt.
	// Retain the conservative Halt meaning within this existing repair pass.
	if p.Availability == Paused && p.PauseMode == "" {
		p.PauseMode = PauseImmediate
	}
	counts := repairs.counts
	for i := range p.Roles {
		kept, count := pruneProfileKeys(p.Roles[i].Profiles, repairs.profileKeys)
		counts["roleReferences"] += count
		p.Roles[i].Profiles = kept
	}
	// Repair only consequences of removed entries. Preexisting unrelated bad
	// references must still fail ordinary validation and invoke recovery.
	if counts["profiles"] > 0 || counts["servedModels"] > 0 {
		kept := make(ResidentModels, 0, len(p.ResidentModels))
		for _, ref := range p.ResidentModels {
			_, profile := p.ProfileFor(ref.Engine, ref.Model)
			if repairs.profiles[ref] && !profile || repairs.served[ref] && p.ServedModels != nil && !slices.Contains(p.ServedModels, ref) {
				counts["residentReferences"]++
				continue
			}
			kept = append(kept, ref)
		}
		p.ResidentModels = kept
	}
	if counts["deployments"] > 0 {
		deployments, families := map[string]bool{}, map[string]bool{}
		for _, d := range p.Routing.Deployments {
			deployments[d.ID], families[d.Family] = true, true
		}
		for tier, entries := range p.Tiers {
			kept := make([]TierEntry, 0, len(entries))
			for _, entry := range entries {
				if entry.Kind == EntryAPI && repairs.deployments[entry.Deployment] && !deployments[entry.Deployment] {
					counts["deploymentReferences"]++
					continue
				}
				kept = append(kept, entry)
			}
			p.Tiers[tier] = kept
		}
		for tier, settings := range p.Routing.Tiers {
			kept := make([]string, 0, len(settings.AllowedFamilies))
			for _, family := range settings.AllowedFamilies {
				if repairs.families[family] && !families[family] {
					counts["familyReferences"]++
					continue
				}
				kept = append(kept, family)
			}
			if len(settings.AllowedFamilies) > 0 && len(kept) == 0 {
				settings.RemoteAllowed, settings.PaidAllowed = false, false
				// Empty means unrestricted, so also remove dormant API candidates.
				entries := p.Tiers[tier]
				keptEntries := make([]TierEntry, 0, len(entries))
				for _, entry := range entries {
					if entry.Kind == EntryAPI {
						counts["deploymentReferences"]++
						continue
					}
					keptEntries = append(keptEntries, entry)
				}
				p.Tiers[tier] = keptEntries
			}
			settings.AllowedFamilies = kept
			p.Routing.Tiers[tier] = settings
		}
	}
	// Keep routing rules only for names with a retained API candidate, after
	// deployment and exhausted-family repair have removed their dependents.
	for tier := range p.Routing.Tiers {
		if !slices.ContainsFunc(p.Tiers[tier], func(entry TierEntry) bool { return entry.Kind == EntryAPI }) {
			delete(p.Routing.Tiers, tier)
			counts["routingTiers"]++
		}
	}
}
