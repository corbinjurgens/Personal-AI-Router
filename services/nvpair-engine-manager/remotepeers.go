// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// remotepeers.go maintains this node's view of cluster peers that expose the ec
// remote-control surface. Like nvpair-errors / nvpair-workload-manager, engine-manager
// learns peers from the broker's discovery relay: it subscribes for the ec
// service and folds each discovery:nodes snapshot into a directory keyed by the
// peer's stable per-host UUID (hostUuid), resolving a remote-control target's
// address, port, and cluster UUID (for pin selection). Keying by UUID (not the
// hostname) means a remote target survives a PC rename and two same-named peers
// stay distinct; callers address a target by its hostUuid. The remote
// engine:remote-* methods look targets up here.

import (
	"slices"
	"sync"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
)

// ecPeer is a resolved remote-control target.
type ecPeer struct {
	nodeID             string
	addresses          []string
	port               int
	clusterUUID        string
	selected           string
	generation         uint64
	selectedGeneration uint64
}

// peerDirectory is the relay-fed set of ec-capable peers, replaced wholesale on
// each discovery:nodes snapshot (stateless + self-correcting, matching the relay
// contract).
type peerDirectory struct {
	mu         sync.RWMutex
	peers      map[string]ecPeer
	generation uint64
}

func newPeerDirectory() *peerDirectory {
	return &peerDirectory{peers: make(map[string]ecPeer)}
}

// set replaces the directory from a discovery:nodes snapshot, keeping only nodes
// that advertise the ec service and a dialable address.
func (d *peerDirectory) set(nodes []noderec.DirectoryNode) {
	d.mu.Lock()
	defer d.mu.Unlock()
	next := make(map[string]ecPeer, len(nodes))
	for _, n := range nodes {
		svc, ok := n.Services[noderec.ServiceEngineControl]
		addresses := n.CandidateIPs()
		if !ok || len(addresses) == 0 || svc.Port == 0 || n.HostUUID == "" {
			continue
		}
		// Key by the stable hostUuid so a target survives a rename and same-named
		// peers don't collide. A relay DirectoryNode always carries a hostUuid
		// (the scanner guarantees it), so there is no name fallback.
		p := ecPeer{
			nodeID:      n.HostUUID,
			addresses:   addresses,
			port:        svc.Port,
			clusterUUID: n.ClusterUUID,
		}
		previous, exists := d.peers[n.HostUUID]
		if exists && sameControlEndpoint(previous, p) {
			p.selected, p.generation, p.selectedGeneration = previous.selected, previous.generation, previous.selectedGeneration
		} else {
			d.generation++
			p.generation = d.generation
		}
		next[n.HostUUID] = p
	}
	d.peers = next
}

func sameControlEndpoint(a, b ecPeer) bool {
	return a.nodeID == b.nodeID && a.clusterUUID == b.clusterUUID && a.port == b.port && slices.Equal(a.addresses, b.addresses)
}

func (d *peerDirectory) selectAddress(peer ecPeer, address string) bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	current, ok := d.peers[peer.nodeID]
	if !ok || !sameControlEndpoint(current, peer) || current.generation != peer.generation || address == "" {
		return false
	}
	if current.selected != address {
		d.generation++
		current.selected, current.selectedGeneration = address, d.generation
		d.peers[peer.nodeID] = current
	}
	return true
}

func (m *Manager) resolveControlRoute(uuid string, service noderec.ServiceKey) (clustertrust.PeerServiceRoute, bool) {
	if service != noderec.ServiceEngineControl || m.mesh == nil {
		return clustertrust.PeerServiceRoute{}, false
	}
	m.mesh.Refresh()
	incarnation, ok := m.mesh.PeerServiceIncarnation(uuid)
	if !ok {
		return clustertrust.PeerServiceRoute{}, false
	}
	m.peers.mu.RLock()
	defer m.peers.mu.RUnlock()
	var selected ecPeer
	found := false
	for _, peer := range m.peers.peers {
		if peer.clusterUUID == uuid {
			if found { // Discovery cannot assign one principal to two targets.
				return clustertrust.PeerServiceRoute{}, false
			}
			selected, found = peer, true
		}
	}
	if !found || selected.selected == "" {
		return clustertrust.PeerServiceRoute{}, false
	}
	return clustertrust.PeerServiceRoute{Incarnation: incarnation, Address: selected.selected,
		Protocol: clustertrust.ReverseTLS, Generation: selected.selectedGeneration}, true
}

// lookup resolves a target node id to its ec peer entry.
func (d *peerDirectory) lookup(nodeID string) (ecPeer, bool) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	p, ok := d.peers[nodeID]
	return p, ok
}
