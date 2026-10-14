// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"path/filepath"
	"slices"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
)

type clusterPeerRoute struct {
	address    string
	candidates []string
	generation uint64
}

func (m *Manager) peerServiceOptions() clustertrust.PeerServiceClientOptions {
	return clustertrust.PeerServiceClientOptions{
		PeerClientOptions: clustertrust.PeerClientOptions{
			Timeout: pairingHTTPTimeout, ResolvePin: m.trust.DER,
		},
		BaseConfigDir: filepath.Dir(m.clusterDir),
		ResolveRoute:  m.resolvePeerServiceRoute,
	}
}

// selectPeerServiceAddress retains the caller's existing candidate walk. Its
// selection is valid only while the member/discovery endpoints remain current;
// a change during dial is observed by the adapter before application writes.
func (m *Manager) selectPeerServiceAddress(uuid, address string) bool {
	m.peerRoutesMu.Lock()
	defer m.peerRoutesMu.Unlock()
	node, ok := m.memberByNodeID(uuid)
	if !ok {
		return false
	}
	candidates := m.resolvePeerAddrs(*node)
	if !slices.Contains(candidates, address) {
		return false
	}
	route := m.peerRoutes[uuid]
	if route.address != address || !slices.Equal(route.candidates, candidates) {
		route.generation++
	}
	route.address, route.candidates = address, candidates
	m.peerRoutes[uuid] = route
	return true
}

func (m *Manager) resolvePeerServiceRoute(uuid string, service noderec.ServiceKey) (clustertrust.PeerServiceRoute, bool) {
	if service != noderec.ServiceCluster {
		return clustertrust.PeerServiceRoute{}, false
	}
	m.peerRoutesMu.Lock()
	defer m.peerRoutesMu.Unlock()
	peer, ok := m.reversePeer(uuid)
	if !ok {
		return clustertrust.PeerServiceRoute{}, false
	}
	node, ok := m.memberByNodeID(uuid)
	if !ok {
		return clustertrust.PeerServiceRoute{}, false
	}
	candidates := m.resolvePeerAddrs(*node)
	if len(candidates) == 0 {
		return clustertrust.PeerServiceRoute{}, false
	}
	route := m.peerRoutes[uuid]
	if route.address == "" || !slices.Equal(route.candidates, candidates) {
		if !slices.Contains(candidates, route.address) {
			route.address = candidates[0]
		}
		route.candidates = candidates
		route.generation++
		m.peerRoutes[uuid] = route
	}
	return clustertrust.PeerServiceRoute{
		Incarnation: peer.Incarnation, Address: route.address,
		Protocol: clustertrust.ReverseTLS, Generation: route.generation,
	}, true
}
