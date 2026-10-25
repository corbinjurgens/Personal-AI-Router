// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"net"
	"slices"
	"strconv"
	"sync"

	"nvpair-shared/clustertrust"
	"nvpair-shared/netpick"
)

// PeerNode is the minimal node shape the broadcast target set needs from
// discovery: an identity, a reachable address, and the port the peer's
// inter-node HTTP server listens on (from its mDNS SRV record).
type PeerNode struct {
	ID        string
	Host      string
	Addresses []string
	Port      int
	TXT       []string // mDNS TXT records (carries cluster-uuid= for cluster mTLS pin lookup)
}

// addresses returns every dialable host for this peer via the shared ranker,
// falling back to the hostname when discovery supplied no address.
func (n PeerNode) addresses() []string {
	if hosts := netpick.Candidates(n.TXT, n.Addresses); len(hosts) > 0 {
		return hosts
	}
	if n.Host != "" {
		return []string{n.Host}
	}
	return nil
}

// PeerSource supplies the current set of peer nodes to broadcast to.
//
// THIS IS THE DISCOVERY SEAM. The implementation is relayPeerSource, fed by the
// broker's discovery relay (discovery:nodes snapshots for the wl service). It
// replaced the former mDNS browser at the discovery-consolidation cutover without
// touching the broadcast, server, or dedup paths — the whole point of the seam.
type PeerSource interface {
	Nodes(ctx context.Context) ([]PeerNode, error)
}

// peerEntry is a discovered peer's broadcast destination: all dialable
// "host:port" candidates plus its cluster principal.
type peerEntry struct {
	candidates []string
	uuid       string
	generation uint64
}

// target is one broadcast destination and the peer uuid used to look up its pin.
type target struct {
	id         string
	candidates []string
	uuid       string
}

// peerSet is the in-memory broadcast target set, replaced wholesale on each
// discovery refresh. The port must come from the peer's advertisement, never
// from this node's listener; the uuid comes from its cluster-uuid= TXT.
type peerSet struct {
	mu         sync.RWMutex
	peers      map[string]peerEntry // node id -> destination
	generation uint64
	onChange   func() // retire idle service connections after a route-set change
}

func newPeerSet() *peerSet {
	return &peerSet{
		peers: make(map[string]peerEntry),
	}
}

// Replace swaps the target set for the destinations derived from nodes.
// Returns the added and removed node ids so the caller can log membership
// churn (spec §13).
func (p *peerSet) Replace(nodes []PeerNode) (added, removed []string) {
	next := make(map[string]peerEntry, len(nodes))
	for _, n := range nodes {
		addresses := n.addresses()
		if len(addresses) == 0 {
			continue
		}
		port := n.Port
		if port < 1 || port > 65535 {
			continue
		}
		id := n.ID
		if id == "" {
			id = addresses[0]
		}
		candidates := make([]string, 0, len(addresses))
		for _, address := range addresses {
			candidates = append(candidates, net.JoinHostPort(address, strconv.Itoa(port)))
		}
		next[id] = peerEntry{candidates: candidates, uuid: clustertrust.ClusterUUIDFromTXT(n.TXT)}
	}

	p.mu.Lock()
	changed := len(p.peers) != len(next)
	for id, e := range next {
		previous, exists := p.peers[id]
		if exists && previous.uuid == e.uuid && slices.Equal(previous.candidates, e.candidates) {
			e.generation = previous.generation
		} else {
			changed = true
			p.generation++
			e.generation = p.generation
		}
		next[id] = e
	}
	for id := range next {
		if _, ok := p.peers[id]; !ok {
			added = append(added, id)
		}
	}
	for id := range p.peers {
		if _, ok := next[id]; !ok {
			removed = append(removed, id)
		}
	}
	p.peers = next
	onChange := p.onChange
	p.mu.Unlock()
	// Cleanup may recheck routes; never call it while holding the peer lock.
	if changed && onChange != nil {
		onChange()
	}
	return added, removed
}

// targets returns the current broadcast destinations, one per peer.
func (p *peerSet) targets() []target {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make([]target, 0, len(p.peers))
	for id, e := range p.peers {
		out = append(out, target{id: id, candidates: e.candidates, uuid: e.uuid})
	}
	return out
}

func (p *peerSet) count() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.peers)
}

// targetUUID resolves the current logical service endpoint. Conflicting
// advertisements for one principal fail closed rather than choosing a map entry.
func (p *peerSet) targetUUID(uuid string) (target, uint64, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	var found target
	var generation uint64
	for id, e := range p.peers {
		if e.uuid != uuid {
			continue
		}
		if found.uuid != "" {
			return target{}, 0, false
		}
		found = target{id: id, candidates: e.candidates, uuid: e.uuid}
		generation = e.generation
	}
	return found, generation, found.uuid != ""
}
