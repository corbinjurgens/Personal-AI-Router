// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"time"

	"nvpair-shared/noderec"
)

const peerServicesMaxAge = 30 * time.Second

type peerServicesOrder struct {
	incarnation string
	sequences   map[string]uint64
}

func peerServicesFresh(claim noderec.PeerServicesClaim, now time.Time) bool {
	observed := time.UnixMilli(claim.ObservedAt)
	return !observed.After(now) && now.Sub(observed) < peerServicesMaxAge
}

func samePeerServicesRoute(a, b noderec.DirectoryNode) bool {
	if a.ClusterUUID != b.ClusterUUID || a.IP != b.IP || !slices.Equal(a.IPs, b.IPs) || len(a.Services) != len(b.Services) {
		return false
	}
	for service, endpoint := range a.Services {
		other, present := b.Services[service]
		if !present || endpoint.Port != other.Port {
			return false
		}
	}
	return true
}

func (d *directory) storeLocked(node noderec.DirectoryNode) {
	if old, present := d.nodes[node.HostUUID]; !present || !samePeerServicesRoute(old, node) {
		d.generations[node.HostUUID]++
	}
	d.nodes[node.HostUUID] = node
}

func (d *directory) routeGeneration(hostUUID string) uint64 {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.generations[hostUUID]
}

func (d *directory) authenticated(hostUUID string) bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	_, ok := d.peerServices[hostUUID]
	return ok
}

func (d *directory) installPeerServices(claim noderec.PeerServicesClaim, node noderec.DirectoryNode) (isNew, routeChanged bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	old, present := d.nodes[node.HostUUID]
	previous, authenticated := d.peerServices[node.HostUUID]
	routeChanged = !present || !samePeerServicesRoute(old, node) || authenticated && previous.Incarnation != claim.Incarnation
	if authenticated && previous.Incarnation != claim.Incarnation {
		d.generations[node.HostUUID]++
	}
	d.peerServices[node.HostUUID] = claim
	d.storeLocked(node)
	return !present, routeChanged
}

// removeBrowse withdraws only the browse claim; authenticated presence wins.
func (d *directory) removeBrowse(hostUUID string) (noderec.DirectoryNode, bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	delete(d.browse, hostUUID)
	if _, authenticated := d.peerServices[hostUUID]; authenticated {
		return noderec.DirectoryNode{}, false
	}
	node, existed := d.nodes[hostUUID]
	delete(d.nodes, hostUUID)
	if existed {
		d.generations[hostUUID]++
	}
	return node, existed
}

func (d *daemon) acceptPeerServices(claim noderec.PeerServicesClaim) bool {
	hostUUID := claim.Descriptor.HostUUID
	if noderec.ValidatePeerServicesDescriptor(claim.Descriptor, hostUUID) != nil || claim.Source == "" || len(claim.Source) > 128 || claim.Sequence == 0 || !peerServicesFresh(claim, time.Now()) {
		return false
	}
	d.mesh.Refresh()
	incarnation, admitted := d.mesh.PeerServiceIncarnation(hostUUID)
	if !admitted || incarnation != claim.Incarnation {
		return false
	}
	d.peerServicesMu.Lock()
	defer d.peerServicesMu.Unlock()
	incarnation, admitted = d.mesh.PeerServiceIncarnation(hostUUID)
	if !admitted || incarnation != claim.Incarnation || !peerServicesFresh(claim, time.Now()) {
		return false
	}
	if d.peerServicesOrder == nil {
		d.peerServicesOrder = make(map[string]peerServicesOrder)
	}
	order := d.peerServicesOrder[hostUUID]
	if order.incarnation != incarnation {
		order = peerServicesOrder{incarnation: incarnation, sequences: make(map[string]uint64)}
	}
	if claim.Sequence <= order.sequences[claim.Source] {
		return false
	}
	// Retain watermarks through expiry for this admission. A new CM process
	// has a fresh source; the broker rejects notifications from replaced children.
	order.sequences[claim.Source] = claim.Sequence
	d.peerServicesOrder[hostUUID] = order
	node := noderec.DirectoryNode{HostUUID: hostUUID, Name: claim.Descriptor.Name, IP: claim.Descriptor.Addresses[0], IPs: slices.Clone(claim.Descriptor.Addresses), ClusterUUID: hostUUID, Trusted: d.mesh.HasPin(hostUUID), Services: make(map[noderec.ServiceKey]noderec.ServiceStatus)}
	for _, service := range claim.Descriptor.Services {
		node.Services[service.Service] = noderec.ServiceStatus{Port: service.Port}
	}
	wasAuthenticated := d.dir.authenticated(hostUUID)
	previousNode, wasPresent := d.dir.get(hostUUID)
	// Descriptor receipt is metadata authority, never evidence of liveness.
	if old, ok := previousNode, wasPresent; ok {
		node.LastSeen = old.LastSeen
		d.dir.mu.RLock()
		previous := d.dir.peerServices[hostUUID]
		d.dir.mu.RUnlock()
		if samePeerServicesRoute(old, node) && previous.Incarnation == claim.Incarnation {
			old.Name = node.Name
			old.Trusted = node.Trusted
			node = old
		}
	}
	isNew, routeChanged := d.dir.installPeerServices(claim, node)
	if routeChanged {
		d.invalidatePeerServicesRoute(hostUUID)
	}
	method := noderec.NotifyNodeUpdated
	if isNew {
		method = noderec.NotifyNodeDiscovered
	}
	if isNew || routeChanged || previousNode.Name != node.Name || previousNode.Trusted != node.Trusted {
		d.emit(method, node)
	}
	// A claim is proof the peer reaches us again, whatever the lapse that put it
	// in the errors view.
	d.clearUnreachable(hostUUID)
	if !wasAuthenticated || routeChanged {
		enrichmentNode := node
		enrichmentNode.Services = maps.Clone(node.Services)
		go d.enrichPeerServices(claim, enrichmentNode, d.dir.routeGeneration(hostUUID))
	}
	return true
}

func (d *daemon) enrichPeerServices(claim noderec.PeerServicesClaim, node noderec.DirectoryNode, generation uint64) {
	// Fetch through the normal service adapters into temporary values. Unlike
	// browse enrichment, an authenticated claim can be replaced during a fetch,
	// so its directory and cache writes share the final authority guard.
	var memory hostMemory
	var info NodeInfoResponse
	var infoFetched, modelsFetched bool
	var inventory modelInventory
	if ni, present := node.Services[noderec.ServiceNodeInfo]; present {
		ask := func(host string) (NodeInfoResponse, bool) {
			candidate, ok := d.fetchNodeInfoForPeer(context.Background(), node.HostUUID, node.ClusterUUID, host, ni.Port, true)
			return candidate, ok && (candidate.HostUUID == "" || candidate.HostUUID == node.HostUUID)
		}
		_, info, infoFetched = askRemembered(&memory, hostKey{hostUUID: node.HostUUID, service: noderec.ServiceNodeInfo}, node.CandidateIPs(), ask)
	}
	if em, present := node.Services[noderec.ServiceEngineManager]; present {
		ask := func(host string) (modelInventory, bool) {
			return d.fetchPeerModels(node.HostUUID, host, em.Port, node.ClusterUUID)
		}
		_, inventory, modelsFetched = askRemembered(&memory, hostKey{hostUUID: node.HostUUID, service: noderec.ServiceEngineManager}, node.CandidateIPs(), ask)
	}
	d.peerServicesMu.Lock()
	defer d.peerServicesMu.Unlock()
	incarnation, admitted := d.mesh.PeerServiceIncarnation(node.HostUUID)
	d.dir.mu.Lock()
	current, exists := d.dir.peerServices[node.HostUUID]
	valid := exists && d.dir.generations[node.HostUUID] == generation && current.Incarnation == claim.Incarnation && admitted && incarnation == claim.Incarnation && peerServicesFresh(current, time.Now())
	if valid {
		node = d.dir.nodes[node.HostUUID]
		if infoFetched {
			node.GPUs, node.CPU, node.Memory = info.GPUs, info.CPU, info.Memory
		}
		if modelsFetched {
			inventory.applyTo(&node)
		}
		d.dir.nodes[node.HostUUID] = node
	}
	d.dir.mu.Unlock()
	if !valid {
		return
	}
	d.infoMu.Lock()
	if infoFetched {
		if d.lastInfo == nil {
			d.lastInfo = make(map[string]NodeInfoResponse)
		}
		if d.lastInfoAt == nil {
			d.lastInfoAt = make(map[string]time.Time)
		}
		d.lastInfo[node.HostUUID], d.lastInfoAt[node.HostUUID] = info, time.Now()
	}
	if modelsFetched {
		d.cacheInventoryLocked(node.HostUUID, inventory)
	}
	d.infoMu.Unlock()
	if infoFetched || modelsFetched {
		d.emit(noderec.NotifyNodeUpdated, node)
	}
}

// reconcilePeerServices removes derived authority on leave, re-pair or expiry.
// Browse claims survive independently and become visible again on retirement.
//
// A claim that expires while its peer is still admitted is the reverse session
// going quiet: cluster-manager stops sending claims when it can no longer reach
// the peer, and the claim's age is the debounce. That is the one point where a
// paired peer that is down is observable without a direct connection, so it is
// reported (reportUnreachable), under the same id enrichment uses, and cleared
// by the next accepted claim, by the peer ceasing to be paired, or by a later
// eviction.
func (d *daemon) reconcilePeerServices() {
	d.peerServicesMu.Lock()
	defer d.peerServicesMu.Unlock()
	d.dir.mu.Lock()
	var changed []noderec.DirectoryNode
	var removed []noderec.DirectoryNode
	var lapsed []noderec.DirectoryNode
	for hostUUID, claim := range d.dir.peerServices {
		incarnation, admitted := d.mesh.PeerServiceIncarnation(hostUUID)
		current := admitted && incarnation == claim.Incarnation
		if current && peerServicesFresh(claim, time.Now()) {
			continue
		}
		old := d.dir.nodes[hostUUID]
		if current {
			lapsed = append(lapsed, old)
		}
		delete(d.dir.peerServices, hostUUID)
		d.dir.generations[hostUUID]++
		if browse, exists := d.dir.browse[hostUUID]; exists {
			browse.Trusted = d.mesh.HasPin(browse.ClusterUUID)
			d.dir.storeLocked(browse)
			changed = append(changed, browse)
		} else {
			delete(d.dir.nodes, hostUUID)
			removed = append(removed, old)
		}
	}
	d.dir.mu.Unlock()
	for _, node := range changed {
		d.invalidatePeerServicesRoute(node.HostUUID)
		d.emit(noderec.NotifyNodeUpdated, node)
	}
	for _, node := range removed {
		d.forget(node.HostUUID)
		d.emit(noderec.NotifyNodeRemoved, node)
	}
	// After the loops above: reverting or forgetting a node clears its report, and
	// this one has to outlive that.
	for _, node := range lapsed {
		d.reportUnreachable(node.HostUUID, fmt.Sprintf("Paired node %q has stopped answering over its reverse session for %s; it may be unreachable from this machine",
			nodeLabel(node), peerServicesMaxAge))
	}
	d.reportSilentPeers(time.Now())
	d.clearUnpairedReports()
}

// reportSilentPeers closes the restart gap in the lapse report above. A claim is
// only ever sent after cluster-manager reads a peer's descriptor, so a scanner
// that starts while a reverse-only peer is down is never handed a claim and so
// never sees one lapse. Once, peerServicesMaxAge after start, it reports every
// admitted peer that has no claim and is not in the directory, with the same id
// and message style as a lapse, and the same events clear it.
//
// The rule is limited to what nothing else covers. A peer with a directory entry
// has either a claim or an mDNS record, so a claim or enrichment owns it
// (enrichment reports when node-info does not answer, and clears when it does);
// only a peer this node knows about from its pin alone is left. cluster-manager
// sends claims to every member whose descriptor it can read, so a peer reached
// directly is not excluded by that: it is skipped only because it has an entry.
// It is a single look rather than a standing rule so that a peer evicted later is
// not reported straight back after its report was cleared by the eviction.
//
// Known limits, accepted: the look can report a peer that is in fact reachable
// when cluster-manager is also new and its first claim takes longer than the
// grace (the claim then clears it), or when it runs between a browse's enrich and
// its upsert, which happens outside peerServicesMu. And a peer missed at the look
// (its pin mid-rewrite, or the mesh not yet clustered) is not looked at again
// until the scanner restarts.
func (d *daemon) reportSilentPeers(now time.Time) {
	if d.startupPeersChecked || d.startedAt.IsZero() || now.Sub(d.startedAt) < peerServicesMaxAge {
		return
	}
	d.startupPeersChecked = true
	for _, hostUUID := range d.mesh.PeerUUIDs() {
		if _, admitted := d.mesh.PeerServiceIncarnation(hostUUID); !admitted {
			continue
		}
		if _, known := d.dir.get(hostUUID); known {
			continue
		}
		d.reportUnreachable(hostUUID, fmt.Sprintf("Paired node %q has not been reachable over its reverse session since this node started; it may be unreachable from this machine",
			nodeLabel(noderec.DirectoryNode{HostUUID: hostUUID})))
	}
}

func (d *daemon) peerServicesLoop(ctx context.Context) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			d.mesh.Refresh()
			d.reconcilePeerServices()
		}
	}
}
