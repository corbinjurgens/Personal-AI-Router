// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net"
	"net/http"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
)

// candidateTransport returns the reverse-proxy / model-list transport for a
// candidate. Plain/self/manual candidates share one long-lived Transport.
// Cluster peers use the process-owned pinned direct/reverse adapter. Callers must
// not CloseIdleConnections on the returned value except via closeIdleTransports.
func (p *Proxy) candidateTransport(c candidate) http.RoundTripper {
	if c.peerUUID == "" {
		return p.plainHTTPTransport()
	}
	return p.peerServiceClient().Client(c.peerUUID, c.fac.profile.DiscoveryService).Transport
}

func newProxyTransport() *http.Transport {
	tr := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   proxyDialTimeout,
			KeepAlive: proxyKeepAlive,
		}).DialContext,
		ResponseHeaderTimeout: proxyResponseTimeout,
		MaxIdleConns:          proxyMaxIdleConns,
		MaxIdleConnsPerHost:   proxyMaxIdleConns,
		IdleConnTimeout:       proxyIdleConnTimeout,
	}
	return tr
}

func (p *Proxy) plainHTTPTransport() *http.Transport {
	p.transportMu.Lock()
	defer p.transportMu.Unlock()
	if p.plainTransport == nil {
		p.plainTransport = newProxyTransport()
	}
	return p.plainTransport
}

func (p *Proxy) peerServiceClient() *clustertrust.PeerServiceClient {
	p.transportMu.Lock()
	defer p.transportMu.Unlock()
	if p.peerServices == nil {
		p.peerServices = clustertrust.NewPeerServiceClient(p.mesh, clustertrust.PeerServiceClientOptions{
			PeerClientOptions: clustertrust.PeerClientOptions{ResponseHeaderTimeout: proxyResponseTimeout},
			BaseConfigDir:     p.baseConfigDir, ResolveRoute: p.resolvePeerServiceRoute,
		})
	}
	return p.peerServices
}

func (p *Proxy) resolvePeerServiceRoute(uuid string, service noderec.ServiceKey) (clustertrust.PeerServiceRoute, bool) {
	if p.mesh == nil {
		return clustertrust.PeerServiceRoute{}, false
	}
	p.mesh.Refresh()
	incarnation, ok := p.mesh.PeerServiceIncarnation(uuid)
	if !ok {
		return clustertrust.PeerServiceRoute{}, false
	}
	for _, f := range p.enabledFacades() {
		if f.profile.DiscoveryService == service {
			endpoint, ok := f.peerEndpoint(uuid)
			if !ok {
				return clustertrust.PeerServiceRoute{}, false
			}
			return clustertrust.PeerServiceRoute{Incarnation: incarnation, Address: endpoint.selected,
				Protocol: clustertrust.ReverseTLS, Generation: endpoint.generation}, true
		}
	}
	return clustertrust.PeerServiceRoute{}, false
}

// Admission is rechecked on every adapter dispatch; the watch also retires idle
// connections promptly. Safe to call from the mesh Watch callback.
func (p *Proxy) dropUnpinnedPeerTransports() {
	p.transportMu.Lock()
	defer p.transportMu.Unlock()
	if p.peerServices != nil {
		p.peerServices.CloseIdle()
	}
}

func (p *Proxy) closeIdleTransports() {
	p.transportMu.Lock()
	defer p.transportMu.Unlock()
	if p.plainTransport != nil {
		p.plainTransport.CloseIdleConnections()
	}
	if p.peerServices != nil {
		p.peerServices.CloseIdle()
	}
}
