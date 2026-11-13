// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"
	"strings"
)

// rosterHash fingerprints this node's outbound roster for the peer-state
// stream. A peer that sees the hash change reconciles with this node.
func (m *Manager) rosterHash() string {
	return canonicalRosterHash(m.buildLocalRoster())
}

// canonicalRosterHash is the lowercase hex SHA-256 of the JSON encoding of a
// canonical copy of r: members sorted by NodeUUID with Addr cleared, each
// member's endorsements sorted by their JSON encoding, and tombstones and
// removal proofs in their existing (already sorted) order. Addr is excluded so
// routine address refreshes do not trigger reconciles; addresses converge
// through descriptors and the repair pass. r is never modified.
func canonicalRosterHash(r *Roster) string {
	canonical := Roster{ClusterID: r.ClusterID, Tombstones: r.Tombstones, RemovalProofs: r.RemovalProofs}
	canonical.Members = make([]RosterEntry, len(r.Members))
	for i, member := range r.Members {
		member.Addr = ""
		member.Endorsements = sortedEndorsements(member.Endorsements)
		canonical.Members[i] = member
	}
	slices.SortFunc(canonical.Members, func(a, b RosterEntry) int { return strings.Compare(a.NodeUUID, b.NodeUUID) })
	// Every field is a string, integer or slice of them, so Marshal cannot fail.
	body, _ := json.Marshal(canonical)
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}

func sortedEndorsements(endorsements []Endorsement) []Endorsement {
	if len(endorsements) == 0 {
		return endorsements
	}
	type keyed struct {
		key         string
		endorsement Endorsement
	}
	entries := make([]keyed, len(endorsements))
	for i, endorsement := range endorsements {
		body, _ := json.Marshal(endorsement)
		entries[i] = keyed{string(body), endorsement}
	}
	slices.SortFunc(entries, func(a, b keyed) int { return strings.Compare(a.key, b.key) })
	sorted := make([]Endorsement, len(entries))
	for i, entry := range entries {
		sorted[i] = entry.endorsement
	}
	return sorted
}
