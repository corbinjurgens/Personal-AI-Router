// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"reflect"
	"slices"
	"strings"
)

// PeerPolicy is what a paired node sees of this node's policy: the device
// fields (roles included), the tier fallback rules and who paused the node.
// The origin-only routing section and every API tier entry
// stay out, so a peer can neither read accounts, credential references or
// deployments nor reconfigure what spends money. Revision is the revision of
// the protected source document.
type PeerPolicy struct {
	Version        int                    `json:"version"`
	Revision       string                 `json:"revision"`
	Availability   Availability           `json:"availability"`
	Pause          PauseConfig            `json:"pause"`
	Admission      Admission              `json:"admission"`
	Idle           IdleConfig             `json:"idle"`
	Profiles       []Profile              `json:"profiles"`
	ServedModels   ServedModels           `json:"servedModels,omitzero"`
	ResidentModels ResidentModels         `json:"residentModels,omitempty"`
	Roles          []Role                 `json:"roles"`
	Tiers          map[string][]TierEntry `json:"tiers"`
	TierPolicy     TierPolicy             `json:"tierPolicy"`
	PausedBy       PauseOrigin            `json:"pausedBy,omitempty"`
	PauseMode      PauseMode              `json:"pauseMode,omitempty"`
}

func (p *PeerPolicy) UnmarshalJSON(data []byte) error {
	if err := CheckModelListKeys(data); err != nil {
		return err
	}
	type wire PeerPolicy
	var next wire
	if err := json.Unmarshal(data, &next); err != nil {
		return err
	}
	*p = PeerPolicy(next)
	return nil
}

// PeerProjection returns the part of p a paired node may see. Tiers is empty
// because API entries are origin-only and local entries are retired.
func (p Policy) PeerProjection() PeerPolicy {
	return PeerPolicy{
		Version:        p.Version,
		Revision:       p.Revision,
		Availability:   p.Availability,
		Pause:          p.Pause,
		Admission:      p.Admission,
		Idle:           p.Idle,
		Profiles:       copyProfiles(p.Profiles),
		ServedModels:   slices.Clone(p.ServedModels),
		ResidentModels: slices.Clone(p.ResidentModels),
		Roles:          copyRoles(p.Roles),
		Tiers:          map[string][]TierEntry{},
		TierPolicy:     p.TierPolicy,
		PausedBy:       p.PausedBy,
		PauseMode:      p.PauseMode,
	}
}

// RemoteSettingLockedError is a definite refusal of a peer policy write. The
// broker maps its type to remote_setting_locked; callers must not parse text.
// Identity is private and is populated only from the stored target policy.
type RemoteSettingLockedError struct {
	engine string
	model  string
	stored bool
}

func (e *RemoteSettingLockedError) Error() string {
	if !e.stored {
		return "inference settings can only be locked on the target device"
	}
	const identityBytes = 96
	return fmt.Sprintf("inference settings are locked on the target device for engine %.96q model %.96q", boundedIdentity(e.engine, identityBytes), boundedIdentity(e.model, identityBytes))
}

func boundedIdentity(value string, limit int) string {
	if len(value) > limit {
		return value[:limit]
	}
	return value
}

func checkPeerInferenceSettings(stored, incoming []Profile) error {
	next := make(map[string]Profile, len(incoming))
	previous := make(map[string]Profile, len(stored))
	for _, profile := range incoming {
		next[profile.Key] = profile
	}
	for _, profile := range stored {
		key := profile.Key
		previous[key] = profile
		if !profile.LocalOnly {
			continue
		}
		candidate, exists := next[key]
		if !exists || !candidate.LocalOnly || candidate.Engine != profile.Engine || candidate.Model != profile.Model || profile.RequestOptionsMode.Effective() != candidate.RequestOptionsMode.Effective() || !EqualRequestOptions(profile.RequestOptions, candidate.RequestOptions) {
			return &RemoteSettingLockedError{engine: profile.Engine, model: profile.Model, stored: true}
		}
	}
	for _, profile := range incoming {
		if !profile.LocalOnly {
			continue
		}
		old, exists := previous[profile.Key]
		if !exists {
			return &RemoteSettingLockedError{}
		}
		if !old.LocalOnly {
			return &RemoteSettingLockedError{engine: old.Engine, model: old.Model, stored: true}
		}
	}
	return nil
}

// EqualRequestOptions compares optional JSON objects without rounding numbers.
// Absent options, null and an empty object are equivalent. Object key order is ignored,
// array order is retained, and equal numeric values compare equal regardless of
// decimal or exponent spelling. Invalid or non-object input compares unequal.
func EqualRequestOptions(a, b json.RawMessage) bool {
	decode := func(raw []byte) (any, error) {
		if len(bytes.TrimSpace(raw)) == 0 {
			return map[string]any{}, nil
		}
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.UseNumber()
		var value map[string]any
		if err := dec.Decode(&value); err != nil {
			return nil, err
		}
		if _, err := dec.Token(); err != io.EOF {
			return nil, errors.New("invalid request options")
		}
		return value, nil
	}
	left, err := decode(a)
	if err != nil {
		return false
	}
	right, err := decode(b)
	return err == nil && equalOptionValue(left, right)
}

func equalOptionValue(a, b any) bool {
	switch left := a.(type) {
	case map[string]any:
		right, ok := b.(map[string]any)
		if !ok || len(left) != len(right) {
			return false
		}
		for key, value := range left {
			other, exists := right[key]
			if !exists || !equalOptionValue(value, other) {
				return false
			}
		}
		return true
	case []any:
		right, ok := b.([]any)
		if !ok || len(left) != len(right) {
			return false
		}
		for i := range left {
			if !equalOptionValue(left[i], right[i]) {
				return false
			}
		}
		return true
	case json.Number:
		right, ok := b.(json.Number)
		return ok && exactOptionNumber(left) == exactOptionNumber(right)
	default:
		return reflect.DeepEqual(a, b)
	}
}

// Canonical decimal coefficient/exponent comparison preserves precision without
// expanding a potentially enormous exponent or converting through float64.
func exactOptionNumber(number json.Number) string {
	value := string(number)
	sign := ""
	if strings.HasPrefix(value, "-") {
		sign, value = "-", value[1:]
	}
	mantissa, exponent, _ := strings.Cut(strings.ToLower(value), "e")
	var power big.Int
	if exponent != "" {
		power.SetString(exponent, 10) // The JSON decoder already validated syntax.
	}
	whole, fraction, _ := strings.Cut(mantissa, ".")
	digits := strings.TrimLeft(whole+fraction, "0")
	if digits == "" {
		return "0"
	}
	trimmed := strings.TrimRight(digits, "0")
	var adjustment big.Int
	adjustment.SetInt64(int64(len(digits) - len(trimmed) - len(fraction)))
	power.Add(&power, &adjustment)
	return sign + trimmed + "e" + power.String()
}

// peerFields returns the top-level fields of a peer document. It refuses two
// keys that are equal under case folding, which Parse would treat as one field
// whose value depends on key order, and a document that is not one JSON object.
func peerFields(raw []byte) (map[string]json.RawMessage, error) {
	notObject := errors.New("policy must be a JSON object")
	dec := json.NewDecoder(bytes.NewReader(raw))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return nil, notObject
	}
	fields := map[string]json.RawMessage{}
	for dec.More() {
		tok, err := dec.Token()
		key, ok := tok.(string)
		if err != nil || !ok {
			return nil, notObject
		}
		for seen := range fields {
			if strings.EqualFold(seen, key) {
				return nil, errors.New("policy has a duplicate field")
			}
		}
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return nil, notObject
		}
		fields[key] = value
	}
	if _, err := dec.Token(); err != nil {
		return nil, notObject
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, notObject
	}
	return fields, nil
}

// MergePeerSet applies a paired node's policy:set document to the stored
// policy. The document must carry no routing section, and its tiers and
// tierPolicy must equal the peer projection of stored: a peer may change device
// fields only, roles included. The result is the document's device fields with
// the stored tiers, tierPolicy, routing, revision, availability and pause metadata
// restored exactly, so the caller must pass the stored policy it is about to
// replace. Errors never echo the document.
func MergePeerSet(stored Policy, raw []byte) (Policy, error) {
	if len(raw) == 0 {
		return Policy{}, errors.New("policy is required")
	}
	fields, err := peerFields(raw)
	if err != nil {
		return Policy{}, err
	}
	// Parse matches keys case-insensitively, so "Routing" would be decoded as
	// the routing section; reject every spelling. Availability and pause metadata
	// are restored from stored and validated together, so every spelling
	// is dropped before the document is parsed on its own.
	for key := range fields {
		if strings.EqualFold(key, "routing") {
			return Policy{}, errors.New("a paired node cannot change the routing configuration")
		}
		if strings.EqualFold(key, "availability") || strings.EqualFold(key, "pausedBy") || strings.EqualFold(key, "pauseMode") {
			delete(fields, key)
		}
	}
	stripped, err := json.Marshal(fields)
	if err != nil {
		return Policy{}, err
	}
	incoming, err := Parse(stripped)
	if err != nil {
		return Policy{}, err
	}
	for _, entries := range incoming.Tiers {
		if len(entries) > 0 {
			return Policy{}, errors.New("a paired node cannot change the tiers")
		}
	}
	if incoming.TierPolicy != stored.TierPolicy {
		return Policy{}, errors.New("a paired node cannot change the tier policy")
	}
	if err := checkPeerInferenceSettings(stored.Profiles, incoming.Profiles); err != nil {
		return Policy{}, err
	}
	merged := incoming
	merged.Profiles = copyProfiles(incoming.Profiles)
	merged.Roles = copyRoles(incoming.Roles)
	merged.Tiers = stored.Tiers
	merged.TierPolicy = stored.TierPolicy
	merged.Routing = stored.Routing
	merged.Revision = stored.Revision
	merged.Availability = stored.Availability
	merged.PausedBy = stored.PausedBy
	merged.PauseMode = stored.PauseMode
	if err := merged.Validate(); err != nil {
		return Policy{}, err
	}
	return merged, nil
}
