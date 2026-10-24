// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
)

// AdvertisedRole is a role as a router sees it: the reference plus the bounds
// of the node's profile for it, resolved by the declaring node.
type AdvertisedRole struct {
	Engine        string       `json:"engine"`
	Model         string       `json:"model"`
	ContextTokens int          `json:"contextTokens,omitempty"`
	Capabilities  []Capability `json:"capabilities,omitempty"`
	// MaxOutputTokens and Qualification come from the profile. A router may
	// offer the role for a hybrid request only when both are present and
	// finite; they are hints, and the destination revalidates every request
	// against its own installed qualification.
	MaxOutputTokens int               `json:"maxOutputTokens,omitempty"`
	Qualification   *QualificationRef `json:"qualification,omitempty"`
}

// UnmarshalJSON keeps a role whose maxOutputTokens or qualification is
// malformed, dropping only that field: they are hints a router validates, and
// a peer's bad hint must not cost the node its advertisement and inventory.
// The other fields keep their R1 decoding, letter-case folding included.
func (r *AdvertisedRole) UnmarshalJSON(data []byte) error {
	type wire AdvertisedRole
	var next wire
	legacy := struct {
		*wire
		MaxOutputTokens json.RawMessage `json:"maxOutputTokens"`
		Qualification   json.RawMessage `json:"qualification"`
	}{wire: &next}
	if err := json.Unmarshal(data, &legacy); err != nil {
		return err
	}
	var output int
	if json.Unmarshal(legacy.MaxOutputTokens, &output) == nil {
		next.MaxOutputTokens = output
	}
	var ref QualificationRef
	if q := bytes.TrimSpace(legacy.Qualification); len(q) > 0 && !bytes.Equal(q, []byte("null")) && json.Unmarshal(q, &ref) == nil {
		next.Qualification = &ref
	}
	*r = AdvertisedRole(next)
	return nil
}

// HybridQualified reports whether the role carries finite qualification
// metadata: a valid reference, a served context and an output ceiling below
// it. It never decodes strictly, so a malformed role from a peer is only
// ineligible for hybrid requests.
func (r AdvertisedRole) HybridQualified() bool {
	return r.Qualification != nil && r.Qualification.Validate() == nil &&
		r.ContextTokens > 0 && r.MaxOutputTokens > 0 && r.MaxOutputTokens < r.ContextTokens
}

// Advertisement is the part of a policy a node publishes to paired peers in
// its inventory document. It carries no tier entries, no routing section, no
// profiles beyond the bounds of the declared roles.
type Advertisement struct {
	ServedModels ServedModels              `json:"servedModels,omitzero"`
	Revision     string                    `json:"revision"`
	Roles        map[string]AdvertisedRole `json:"roles"`
	// PausedBy lets a remote UI show "paused on that node" before it tries.
	PausedBy PauseOrigin `json:"pausedBy,omitempty"`
	// StrictAdmission is absent until destination enforcement and identity
	// reconciliation are ready. Policy.Advertisement never sets it.
	StrictAdmission *StrictAdmissionCapability `json:"strictAdmission,omitempty"`
}

// StrictCapabilityState keeps old peers, future peers and usable contracts
// distinct. Valid describes the schema, not current capacity or qualification.
type StrictCapabilityState string

const (
	StrictCapabilityAbsent      StrictCapabilityState = "absent"
	StrictCapabilityUnsupported StrictCapabilityState = "unsupported"
	StrictCapabilityValid       StrictCapabilityState = "valid"
	StrictCapabilityInvalid     StrictCapabilityState = "invalid"
)

// StrictAdmissionCapability belongs only in the inventory's policy object.
// Unknown versions retain their complete JSON for forward-compatible concrete
// routing; their Models must never be interpreted by hybrid consumers.
type StrictAdmissionCapability struct {
	Version     int                    `json:"version"`
	Models      []StrictAdmissionModel `json:"models"`
	unsupported json.RawMessage
}

// StrictAdmissionModel describes finite destination-owned capacity. Engine and
// model must already use the destination's normalization (A1 owns it).
type StrictAdmissionModel struct {
	Engine        string              `json:"engine"`
	Model         string              `json:"model"`
	MaxConcurrent int                 `json:"maxConcurrent"`
	Identity      StrictModelIdentity `json:"identity"`
}

func (c *StrictAdmissionCapability) State() StrictCapabilityState {
	if c == nil {
		return StrictCapabilityAbsent
	}
	err := c.Validate()
	if errors.Is(err, ErrStrictAdmissionVersion) {
		return StrictCapabilityUnsupported
	}
	if err != nil {
		return StrictCapabilityInvalid
	}
	return StrictCapabilityValid
}

func (c StrictAdmissionCapability) Validate() error {
	if c.Version < 1 || c.Version > 2147483647 {
		return errors.New("version must be between 1 and 2147483647")
	}
	if c.Version != StrictAdmissionVersion {
		return ErrStrictAdmissionVersion
	}
	if len(c.Models) < 1 || len(c.Models) > 1024 {
		return errors.New("models must contain between 1 and 1024 entries")
	}
	seen := map[string]bool{}
	for n, m := range c.Models {
		if err := m.Validate(); err != nil {
			return fmt.Errorf("models[%d]: %w", n, err)
		}
		key := m.Engine + "\x00" + m.Model
		if seen[key] {
			return fmt.Errorf("models[%d]: duplicate engine/model", n)
		}
		seen[key] = true
	}
	return nil
}

func (m StrictAdmissionModel) Validate() error {
	if err := checkModelRef(m.Engine, m.Model, 0, nil); err != nil {
		return err
	}
	if m.MaxConcurrent < 1 || m.MaxConcurrent > 1048576 {
		return errors.New("maxConcurrent must be between 1 and 1048576")
	}
	if err := m.Identity.Validate(); err != nil {
		return fmt.Errorf("identity: %w", err)
	}
	return nil
}

func (m *StrictAdmissionModel) UnmarshalJSON(data []byte) error {
	if _, err := filterModelEntry(data, false); err != nil {
		return err
	}
	type wire StrictAdmissionModel
	var next wire
	if err := decodeStrictContract(data, &next, "engine", "model", "maxConcurrent", "identity"); err != nil {
		return err
	}
	if err := StrictAdmissionModel(next).Validate(); err != nil {
		return err
	}
	*m = StrictAdmissionModel(next)
	return nil
}

func (c *StrictAdmissionCapability) UnmarshalJSON(data []byte) error {
	version, err := strictAdmissionVersion(data)
	if err != nil {
		return err
	}
	if version != StrictAdmissionVersion {
		*c = StrictAdmissionCapability{Version: version, unsupported: append(json.RawMessage(nil), data...)}
		return nil
	}
	// decodeStrictContract normalizes through an interface value. Guard the
	// original model tokens and effective keys before that can repair encoding
	// or collapse repeated members, including at direct readiness intake.
	fields, err := modelObject(data)
	if err != nil {
		return err
	}
	if raw, ok := fields["models"]; ok {
		if _, _, err := filterModelArray(raw, false); err != nil {
			return err
		}
	}
	type wire StrictAdmissionCapability
	var next wire
	if err := decodeStrictContract(data, &next, "version", "models"); err != nil {
		return err
	}
	if err := StrictAdmissionCapability(next).Validate(); err != nil {
		return err
	}
	*c = StrictAdmissionCapability(next)
	return nil
}

func strictAdmissionVersion(data []byte) (int, error) {
	// Envelope parsing permits unknown fields only for unsupported versions.
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return 0, err
	}
	if fields == nil {
		return 0, errors.New("strictAdmission must be an object")
	}
	var version float64
	if err := json.Unmarshal(fields["version"], &version); err != nil {
		return 0, errors.New("version must be a number")
	}
	if version < 1 || version > 2147483647 || version != float64(int(version)) {
		return 0, errors.New("version must be between 1 and 2147483647")
	}
	return int(version), nil
}

func (c StrictAdmissionCapability) MarshalJSON() ([]byte, error) {
	if len(c.unsupported) != 0 {
		return c.unsupported, nil
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	type wire StrictAdmissionCapability
	return json.Marshal(wire(c))
}

// UnmarshalJSON distinguishes an absent capability from explicit null. Existing
// advertisement fields retain their R1 decoding and encoding behavior.
func (a *Advertisement) UnmarshalJSON(data []byte) error {
	clean, err := FilterAdvertisementModels(data)
	if err != nil {
		return err
	}
	data = clean
	if err := checkModelListKeys(data); err != nil {
		return err
	}
	type wire Advertisement
	var next wire
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	for key := range fields {
		if key != "strictAdmission" && strings.EqualFold(key, "strictAdmission") {
			return fmt.Errorf("%q must use the exact field name strictAdmission", key)
		}
	}
	if v, ok := fields["strictAdmission"]; ok && string(v) == "null" {
		return errors.New("strictAdmission must not be null")
	}
	// Shadow only the new field so legacy fields keep encoding/json's existing
	// case folding and document-order behavior. Decode the capability separately
	// from its exact map entry, never through struct-field folding.
	legacy := struct {
		*wire
		StrictAdmission json.RawMessage `json:"strictAdmission"`
	}{wire: &next}
	if err := json.Unmarshal(data, &legacy); err != nil {
		return err
	}
	if v, ok := fields["strictAdmission"]; ok {
		var capability StrictAdmissionCapability
		if err := json.Unmarshal(v, &capability); err != nil {
			return err
		}
		next.StrictAdmission = &capability
	}
	*a = Advertisement(next)
	return nil
}

// FilterAdvertisementModels removes only invalid model entries from recognized
// peer containers before strict nested decoding. Future strict versions remain
// opaque. The returned JSON preserves explicit empty served permission.
func FilterAdvertisementModels(data []byte) ([]byte, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return nil, err
	}
	if fields == nil {
		return nil, errors.New("policy must be a JSON object")
	}
	// Keep exact-field diagnostics ahead of any nested capability validation.
	for key := range fields {
		if key != "strictAdmission" && strings.EqualFold(key, "strictAdmission") {
			return nil, fmt.Errorf("%q must use the exact field name strictAdmission", key)
		}
	}
	if err := checkModelListKeys(data); err != nil {
		return nil, err
	}
	// Transform members in document order. Legacy revision/Revision and other
	// unrelated fields retain encoding/json's last-wins and case-fold behavior.
	d := json.NewDecoder(bytes.NewReader(data))
	if _, err := d.Token(); err != nil {
		return nil, err
	}
	var out bytes.Buffer
	out.WriteByte('{')
	first := true
	for d.More() {
		tok, err := d.Token()
		if err != nil {
			return nil, err
		}
		key, ok := tok.(string)
		if !ok {
			return nil, errors.New("invalid policy key")
		}
		var raw json.RawMessage
		if err := d.Decode(&raw); err != nil {
			return nil, err
		}
		clean, err := filterAdvertisementField(key, raw)
		if err != nil {
			return nil, err
		}
		if clean == nil {
			continue
		}
		if !first {
			out.WriteByte(',')
		}
		first = false
		encodedKey, err := json.Marshal(key)
		if err != nil {
			return nil, err
		}
		out.Write(encodedKey)
		out.WriteByte(':')
		out.Write(clean)
	}
	if _, err := d.Token(); err != nil {
		return nil, err
	}
	out.WriteByte('}')
	return out.Bytes(), nil
}

func filterAdvertisementField(key string, raw json.RawMessage) (json.RawMessage, error) {
	if key == "servedModels" {
		clean, _, err := filterModelArray(raw, true, "engine", "model")
		return clean, err
	}
	if strings.EqualFold(key, "roles") && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		roles, err := modelObject(raw)
		if err != nil {
			return nil, err
		}
		for tier, raw := range roles {
			ok, err := filterModelEntry(raw, true)
			if err != nil {
				return nil, err
			}
			if !ok {
				delete(roles, tier)
			}
		}
		return json.Marshal(roles)
	}
	if key == "strictAdmission" && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		version, err := strictAdmissionVersion(raw)
		if err != nil {
			return nil, err
		}
		if version == StrictAdmissionVersion {
			capability, err := modelObject(raw)
			if err != nil {
				return nil, err
			}
			if len(capability) != 2 || capability["version"] == nil || capability["models"] == nil {
				return nil, errors.New("invalid strict capability shape")
			}
			if models, ok := capability["models"]; ok {
				clean, removed, err := filterModelArray(models, true, "engine", "model", "maxConcurrent", "identity")
				if err != nil {
					return nil, err
				}
				if removed > 0 && bytes.Equal(clean, []byte("[]")) {
					return nil, nil
				}
				capability["models"] = clean
			}
			return json.Marshal(capability)
		}
	}
	return raw, nil
}

// Advertisement returns what p publishes to paired peers. Each declared role
// carries the context bound, capabilities, output ceiling and qualification of
// the profile for the same (engine, model); a role with no profile carries none.
func (p Policy) Advertisement() Advertisement {
	roles := make(map[string]AdvertisedRole, len(p.Roles))
	for tier, role := range p.Roles {
		adv := AdvertisedRole{Engine: role.Engine, Model: role.Model}
		if pr, ok := p.ProfileFor(role.Engine, role.Model); ok {
			adv.ContextTokens = pr.ContextTokens
			adv.Capabilities = nilIfEmpty(slices.Clone(pr.Capabilities))
			adv.MaxOutputTokens = pr.MaxOutputTokens
			if pr.Qualification != nil {
				q := *pr.Qualification
				adv.Qualification = &q
			}
		}
		roles[tier] = adv
	}
	return Advertisement{Revision: p.Revision, Roles: roles, PausedBy: p.PausedBy, ServedModels: slices.Clone(p.ServedModels)}
}
