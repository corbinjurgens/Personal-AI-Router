// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"log"

	"nvpair-shared/nodename"
)

// nodeNameView is the local RPC and notification shape of this node's name.
// Name is the effective name; CustomName is the user-set name or nil.
type nodeNameView struct {
	NodeUUID   string  `json:"nodeUuid"`
	Name       string  `json:"name"`
	CustomName *string `json:"customName"`
}

// receivedNodeName validates a name received from a peer or requester. It
// returns the normalized name and true, or nodename.Fallback(nodeUUID) and
// false when the name is invalid.
func receivedNodeName(raw, nodeUUID string) (string, bool) {
	if name, err := nodename.Normalize(raw); err == nil {
		return name, true
	}
	return nodename.Fallback(nodeUUID), false
}

// peerNameToStore is the name to record for a peer from a received name: the
// normalized name when valid; otherwise the name already stored for that peer
// (member, then pin), so an invalid name never replaces a stored one; else the
// UUID fallback.
func (m *Manager) peerNameToStore(raw, nodeUUID string) string {
	name, ok := receivedNodeName(raw, nodeUUID)
	if ok {
		return name
	}
	if n, found := m.memberByUUID(nodeUUID); found && n.Name != "" {
		return n.Name
	}
	if pin, found := m.trust.Get(nodeUUID); found && pin.Name != "" {
		return pin.Name
	}
	return name
}

// initSelfName loads the name state at startup: the generated default for this
// boot and the stored user-set name. A stored name that fails validation is
// treated as unset.
func (m *Manager) initSelfName(stored identityFile) {
	m.identityCreatedAt = stored.CreatedAt
	m.generatedName = defaultNodeName(m.identity.NodeUUID)
	if stored.DisplayName == "" {
		return
	}
	name, err := nodename.Normalize(stored.DisplayName)
	if err != nil {
		log.Printf("identity.json display_name is invalid (%v); using the generated name", err)
		return
	}
	m.customName = name
}

// selfName returns this node's effective display name: the user-set name when
// set, else the generated default.
func (m *Manager) selfName() string {
	m.nameMu.RLock()
	defer m.nameMu.RUnlock()
	return m.effectiveNameLocked()
}

// customSelfName returns the user-set name, or nil when unset.
func (m *Manager) customSelfName() *string {
	m.nameMu.RLock()
	defer m.nameMu.RUnlock()
	return optionalString(m.customName)
}

func (m *Manager) effectiveNameLocked() string {
	if m.customName != "" {
		return m.customName
	}
	return m.generatedName
}

// optionalString returns nil for "" and a pointer to s otherwise.
func optionalString(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

func (m *Manager) selfNameView() nodeNameView {
	m.nameMu.RLock()
	defer m.nameMu.RUnlock()
	return nodeNameView{
		NodeUUID:   m.identity.NodeUUID,
		Name:       m.effectiveNameLocked(),
		CustomName: optionalString(m.customName),
	}
}

// setSelfName sets (non-nil) or clears (nil) the user-set name. It validates
// with nodename.Normalize (an error for which nodename.Reason is non-empty),
// persists display_name in identity.json, then, when the effective name
// changed on a clustered node, renames the self member and kicks an immediate
// roster reconcile so reachable peers learn the name at once. It emits
// cluster:node-name-changed when the effective or custom name changed.
//
// nameWriteMu stays held through the member update and notification so two
// concurrent setters publish in the order they persisted; the self-member
// update takes rosterMu after it. nameMu is held only for the assignment; no
// other lock is taken under it.
func (m *Manager) setSelfName(custom *string) (nodeNameView, error) {
	next := ""
	if custom != nil {
		name, err := nodename.Normalize(*custom)
		if err != nil {
			return nodeNameView{}, err
		}
		next = name
	}

	m.nameWriteMu.Lock()
	defer m.nameWriteMu.Unlock()
	if err := writeIdentityFile(m.clusterDir, identityFile{
		NodeUUID:    m.identity.NodeUUID,
		CreatedAt:   m.identityCreatedAt,
		DisplayName: next,
	}); err != nil {
		return nodeNameView{}, err
	}
	m.nameMu.Lock()
	prevName, prevCustom := m.effectiveNameLocked(), m.customName
	m.customName = next
	m.nameMu.Unlock()

	view := m.selfNameView()
	if view.Name != prevName {
		// The self member is written only under the composition boundary
		// (foundCluster, the pairing commit). Taking rosterMu here means an
		// addSelfMember that read the old name commits before this update.
		// Order: nameWriteMu, then rosterMu, then memMu.
		m.rosterMu.Lock()
		clusterID, _ := m.clusterIdentity()
		renamed := clusterID != "" && m.updateMemberIdentity(m.identity.NodeUUID, m.selfName())
		m.rosterMu.Unlock()
		if renamed {
			m.emitNodesChanged()
		}
		if clusterID != "" {
			go m.reconcileAllMembers()
		}
	}
	if view.Name != prevName || next != prevCustom {
		if err := m.codec.Notify("cluster:node-name-changed", view); err != nil {
			log.Printf("emit cluster:node-name-changed: %v", err)
		}
	}
	return view, nil
}
