// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strings"
)

// PeerPolicy is what a paired node sees of this node's policy: the device
// fields, the local tier entries and the tier fallback rules. The origin-only
// routing section and every API tier entry stay out, so a peer can neither read
// accounts, credential references or deployments nor reconfigure what spends
// money. Revision is the revision of the protected source document.
type PeerPolicy struct {
	Version      int                    `json:"version"`
	Revision     string                 `json:"revision"`
	Availability Availability           `json:"availability"`
	Pause        PauseConfig            `json:"pause"`
	Admission    Admission              `json:"admission"`
	Idle         IdleConfig             `json:"idle"`
	Profiles     []Profile              `json:"profiles"`
	Tiers        map[string][]TierEntry `json:"tiers"`
	TierPolicy   TierPolicy             `json:"tierPolicy"`
}

// PeerProjection returns the part of p a paired node may see. Tiers keep their
// local entries only, and a tier left with none is omitted so the projection
// does not reveal that it holds API entries.
func (p Policy) PeerProjection() PeerPolicy {
	return PeerPolicy{
		Version:      p.Version,
		Revision:     p.Revision,
		Availability: p.Availability,
		Pause:        p.Pause,
		Admission:    p.Admission,
		Idle:         p.Idle,
		Profiles:     p.Profiles,
		Tiers:        localTiers(p.Tiers),
		TierPolicy:   p.TierPolicy,
	}
}

// localTiers copies the local entries of every tier, dropping empty tiers.
func localTiers(tiers map[string][]TierEntry) map[string][]TierEntry {
	out := make(map[string][]TierEntry, len(tiers))
	for tier, entries := range tiers {
		var local []TierEntry
		for _, e := range entries {
			if e.IsLocal() {
				e.Capabilities = nilIfEmpty(e.Capabilities)
				local = append(local, e)
			}
		}
		if len(local) > 0 {
			out[tier] = local
		}
	}
	return out
}

// nonEmptyTiers copies tiers, dropping empty ones, so an explicit empty tier
// and an absent one compare equal.
func nonEmptyTiers(tiers map[string][]TierEntry) map[string][]TierEntry {
	out := make(map[string][]TierEntry, len(tiers))
	for tier, entries := range tiers {
		if len(entries) > 0 {
			out[tier] = entries
		}
	}
	return out
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
// fields only. The result is the document's device fields with the stored
// tiers, tierPolicy, routing, revision and availability restored exactly, so
// the caller must pass the stored policy it is about to replace. Errors never
// echo the document.
func MergePeerSet(stored Policy, raw []byte) (Policy, error) {
	if len(raw) == 0 {
		return Policy{}, errors.New("policy is required")
	}
	fields, err := peerFields(raw)
	if err != nil {
		return Policy{}, err
	}
	// Parse matches keys case-insensitively, so "Routing" would be decoded as
	// the routing section; reject every spelling.
	for key := range fields {
		if strings.EqualFold(key, "routing") {
			return Policy{}, errors.New("a paired node cannot change the routing configuration")
		}
	}
	delete(fields, "availability")
	stripped, err := json.Marshal(fields)
	if err != nil {
		return Policy{}, err
	}
	incoming, err := Parse(stripped)
	if err != nil {
		return Policy{}, err
	}
	if !reflect.DeepEqual(nonEmptyTiers(incoming.Tiers), localTiers(stored.Tiers)) {
		return Policy{}, errors.New("a paired node cannot change the tiers")
	}
	if incoming.TierPolicy != stored.TierPolicy {
		return Policy{}, errors.New("a paired node cannot change the tier policy")
	}
	merged := incoming
	merged.Tiers = stored.Tiers
	merged.TierPolicy = stored.TierPolicy
	merged.Routing = stored.Routing
	merged.Revision = stored.Revision
	merged.Availability = stored.Availability
	if err := merged.Validate(); err != nil {
		return Policy{}, err
	}
	return merged, nil
}
