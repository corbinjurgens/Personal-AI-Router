// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"net"
	"net/http"
	"slices"
	"strconv"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
)

type scannerPeerEndpoint struct {
	hostUUID, peerUUID, address string
	service                     noderec.ServiceKey
	incarnation                 string
	generation                  uint64
}

func (d *daemon) peerServiceRoute(key scannerPeerEndpoint, host string, port int, directoryBacked bool) (clustertrust.PeerServiceRoute, bool) {
	d.mesh.Refresh()
	incarnation, admitted := d.mesh.PeerServiceIncarnation(key.peerUUID)
	if !admitted {
		return clustertrust.PeerServiceRoute{}, false
	}
	var generation uint64
	if d.dir != nil {
		// Read endpoint authority and its semantic generation in one snapshot.
		d.dir.mu.RLock()
		node, found := d.dir.nodes[key.hostUUID]
		endpoint, registered := node.Services[key.service]
		valid := (!directoryBacked || found) && (!found || registered && endpoint.Port == port && node.ClusterUUID == key.peerUUID && slices.Contains(node.CandidateIPs(), host))
		if claim, authenticated := d.dir.peerServices[key.hostUUID]; authenticated {
			valid = valid && claim.Incarnation == incarnation && peerServicesFresh(claim, time.Now())
		}
		if directoryBacked {
			generation = d.dir.generations[key.hostUUID]
		}
		d.dir.mu.RUnlock()
		if !valid {
			return clustertrust.PeerServiceRoute{}, false
		}
	}
	protocol := clustertrust.ReverseTLS
	if key.service == noderec.ServiceNodeInfo {
		protocol = clustertrust.ReversePlain
	}
	return clustertrust.PeerServiceRoute{Incarnation: incarnation, Address: key.address, Protocol: protocol, Generation: generation}, true
}

// peerServiceClient retains route health per advertised address. Directory reads
// own endpoint authority; initial browse enrichment precedes directory insertion
// and therefore has only the current admission as its authority boundary.
func (d *daemon) peerServiceClient(hostUUID, peerUUID, host string, port int, service noderec.ServiceKey, discovering bool) (*http.Client, func() bool, bool) {
	key := scannerPeerEndpoint{hostUUID: hostUUID, peerUUID: peerUUID, address: net.JoinHostPort(host, strconv.Itoa(port)), service: service}
	directoryBacked := false
	if d.dir != nil {
		_, directoryBacked = d.dir.get(hostUUID)
	}
	if !directoryBacked && !discovering {
		return nil, nil, false
	}
	captured, ok := d.peerServiceRoute(key, host, port, directoryBacked)
	if !ok {
		return nil, nil, false
	}
	key.incarnation, key.generation = captured.Incarnation, captured.Generation
	resolve := func(uuid string, requested noderec.ServiceKey) (clustertrust.PeerServiceRoute, bool) {
		if uuid != peerUUID || requested != service {
			return clustertrust.PeerServiceRoute{}, false
		}
		route, ok := d.peerServiceRoute(key, host, port, directoryBacked)
		return route, ok && route.Incarnation == captured.Incarnation && route.Generation == captured.Generation
	}
	d.peerMu.Lock()
	adapter := d.peerClients[key]
	// Pre-insertion enrichment must not install a permanent route which could
	// outlive a later directory removal.
	if adapter == nil || !directoryBacked {
		for endpoint, old := range d.peerClients {
			if endpoint.hostUUID == hostUUID && (endpoint.incarnation != key.incarnation || endpoint.generation != key.generation) {
				old.CloseIdle()
				delete(d.peerClients, endpoint)
			}
		}
		timeout := modelsFetchTimeout
		if service == noderec.ServiceNodeInfo {
			timeout = nodeInfoFetchTimeout
		}
		adapter = clustertrust.NewPeerServiceClient(d.mesh, clustertrust.PeerServiceClientOptions{
			PeerClientOptions: clustertrust.PeerClientOptions{Timeout: timeout},
			BaseConfigDir:     d.baseDir, ResolveRoute: resolve,
			DirectTimeout: nodeInfoDialTimeout,
		})
		if directoryBacked {
			if d.peerClients == nil {
				d.peerClients = make(map[scannerPeerEndpoint]*clustertrust.PeerServiceClient)
			}
			d.peerClients[key] = adapter
		}
	}
	d.peerMu.Unlock()
	current := func() bool {
		route, ok := resolve(peerUUID, service)
		return ok && route.Incarnation == captured.Incarnation && route.Generation == captured.Generation
	}
	return adapter.Client(peerUUID, service), current, true
}

func (d *daemon) closePeerClients() {
	d.peerMu.Lock()
	defer d.peerMu.Unlock()
	for _, adapter := range d.peerClients {
		adapter.CloseIdle()
	}
}

func (d *daemon) hasPeerClient(peerUUID string) bool {
	d.peerMu.Lock()
	defer d.peerMu.Unlock()
	for endpoint := range d.peerClients {
		if endpoint.peerUUID == peerUUID {
			return true
		}
	}
	return false
}

// nodeInfoOriginGate serializes requests to one node-info origin before the
// http.Client starts its request timeout. The transport carries the same
// one-connection bound, but its internal queue starts Client.Timeout too early:
// a caller waiting behind a healthy slow response can otherwise expire before
// its own request reaches the peer.
//
// Origins are retained for the daemon lifetime. Their count is bounded by the
// discovered node endpoints seen during that process, and retaining the channel
// avoids a release/delete race with a concurrent waiter.
type nodeInfoOriginGate struct {
	mu      sync.Mutex
	origins map[string]chan struct{}
}

func (g *nodeInfoOriginGate) acquire(ctx context.Context, origin string) (func(), bool) {
	g.mu.Lock()
	permit := g.origins[origin]
	if permit == nil {
		if g.origins == nil {
			g.origins = make(map[string]chan struct{})
		}
		permit = make(chan struct{}, 1)
		g.origins[origin] = permit
	}
	g.mu.Unlock()

	select {
	case permit <- struct{}{}:
		return func() { <-permit }, true
	case <-ctx.Done():
		return nil, false
	}
}
