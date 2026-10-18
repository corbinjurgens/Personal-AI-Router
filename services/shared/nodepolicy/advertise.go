// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import "slices"

// AdvertisedRole is a role as a router sees it: the reference plus the bounds
// of the node's profile for it, resolved by the declaring node.
type AdvertisedRole struct {
	Engine        string       `json:"engine"`
	Model         string       `json:"model"`
	ContextTokens int          `json:"contextTokens,omitempty"`
	Capabilities  []Capability `json:"capabilities,omitempty"`
}

// Advertisement is the part of a policy a node publishes to paired peers in
// its inventory document. It carries no tier entries, no routing section, no
// profiles beyond the bounds of the declared roles.
type Advertisement struct {
	Revision string                    `json:"revision"`
	Roles    map[string]AdvertisedRole `json:"roles"`
	// PausedBy lets a remote UI show "paused on that node" before it tries.
	PausedBy PauseOrigin `json:"pausedBy,omitempty"`
}

// Advertisement returns what p publishes to paired peers. Each declared role
// carries the context bound and capabilities of the profile for the same
// (engine, model); a role with no profile carries neither.
func (p Policy) Advertisement() Advertisement {
	roles := make(map[string]AdvertisedRole, len(p.Roles))
	for tier, role := range p.Roles {
		adv := AdvertisedRole{Engine: role.Engine, Model: role.Model}
		if pr, ok := p.ProfileFor(role.Engine, role.Model); ok {
			adv.ContextTokens = pr.ContextTokens
			adv.Capabilities = nilIfEmpty(slices.Clone(pr.Capabilities))
		}
		roles[tier] = adv
	}
	return Advertisement{Revision: p.Revision, Roles: roles, PausedBy: p.PausedBy}
}
