// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"slices"
	"sync"

	"nvpair-shared/engines"
	"nvpair-shared/noderec"
)

// registry is the promoted daemon's view of THIS node's advertised services. It
// is fed by relayed discovery:register / unregister / update-txt (each local
// worker registers its service + port; the broker's engine poller registers
// ol/lm at the engine port), and it builds the single _nvpair-node._tcp TXT
// record this node advertises. Transport is never stored — it's policy-derived
// from the service key (see noderec.ServiceKey.Transport).
//
// The node's identity (hostUuid / clusterUuid / addresses) is stamped by the
// daemon, not by the registering workers: hostUuid comes from cluster-manager's
// identity when present (see nodeid.Resolve), clusterUuid from the cluster trust
// dir, addresses from the local address ranker. The display name comes from the
// broker (discovery:set-self-name).
type registry struct {
	mu       sync.Mutex
	hostUUID string
	cluUUID  string
	// ips is this host's addresses in the ranker's order. The first is canonical
	// (published as ip=) and the whole list is published as ips=, because a
	// multi-homed host has no single address every peer can reach: a
	// direct-connect link works only for the machine on its far end.
	ips      []string
	services map[noderec.ServiceKey]int
	// name is this node's display name, set by the broker from the cluster
	// manager (discovery:set-self-name) and advertised as name=. Empty until
	// the first set.
	name string
	// nodeInfoContext identifies the current local ni registration and stamped
	// identity. Retirement cancels reads even if the same port is registered
	// again before their response arrives. It is independent of LAN addresses.
	nodeInfoContext context.Context
	cancelNodeInfo  context.CancelFunc
}

func newRegistry(hostUUID, clusterUUID string, ips []string) *registry {
	return &registry{
		hostUUID: hostUUID,
		cluUUID:  clusterUUID,
		ips:      trimAddresses(ips),
		services: make(map[noderec.ServiceKey]int),
	}
}

// trimAddresses copies and caps a ranked address list to what the record
// publishes, so the registry never holds addresses the wire would drop.
func trimAddresses(ips []string) []string {
	if len(ips) > noderec.MaxAdvertisedIPs {
		ips = ips[:noderec.MaxAdvertisedIPs]
	}
	return slices.Clone(ips)
}

// register adds or updates a service's port. It reports whether the advertised
// record changed (so the caller only re-advertises on a real change). update-txt
// is the same operation (any per-service TXT a future service contributes would
// be merged here); today no service carries TXT — the model list moved to HTTP.
func (r *registry) register(p noderec.RegisterParams) bool {
	if p.Service == "" || p.Port == 0 {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if p.Service == noderec.ServiceNodeInfo {
		// A repeated registration may be a replacement process at the same port.
		// TXT publication remains idempotent, but in-flight authority is retired.
		r.retireNodeInfoLocked()
	}
	if r.services[p.Service] == p.Port {
		return false
	}
	r.services[p.Service] = p.Port
	return true
}

// unregister drops a service. Reports whether the record changed.
func (r *registry) unregister(s noderec.ServiceKey) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.services[s]; !ok {
		return false
	}
	if s == noderec.ServiceNodeInfo {
		r.retireNodeInfoLocked()
	}
	delete(r.services, s)
	return true
}

// setIdentity updates the node's stamped identity (e.g. after cluster-manager
// registers or its identity changes). Reports whether the record changed.
func (r *registry) setIdentity(hostUUID, clusterUUID string, ips []string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	ips = trimAddresses(ips)
	if r.hostUUID == hostUUID && r.cluUUID == clusterUUID && slices.Equal(r.ips, ips) {
		return false
	}
	if r.hostUUID != hostUUID || r.cluUUID != clusterUUID {
		r.retireNodeInfoLocked()
	}
	r.hostUUID, r.cluUUID, r.ips = hostUUID, clusterUUID, ips
	return true
}

type selfNodeInfoAuthority struct {
	hostUUID, clusterUUID string
	port                  int
	incarnation           context.Context
}

// selfNodeInfo snapshots only this scanner's stamped identity and broker-fed
// service registration. A directory, mDNS claim or caller-supplied address
// cannot create local authority. The incarnation is cancelled on retirement.
func (r *registry) selfNodeInfo(hostUUID string) (selfNodeInfoAuthority, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	port := r.services[noderec.ServiceNodeInfo]
	if hostUUID == "" || hostUUID != r.hostUUID || port <= 0 || port > 65535 {
		return selfNodeInfoAuthority{}, false
	}
	if r.nodeInfoContext == nil {
		r.nodeInfoContext, r.cancelNodeInfo = context.WithCancel(context.Background())
	}
	return selfNodeInfoAuthority{r.hostUUID, r.cluUUID, port, r.nodeInfoContext}, true
}

// retireNodeInfoLocked is called with mu held at the registration/identity
// mutation, before another reader can acquire the replacement incarnation.
func (r *registry) retireNodeInfoLocked() {
	if r.cancelNodeInfo != nil {
		r.cancelNodeInfo()
	}
	r.nodeInfoContext, r.cancelNodeInfo = nil, nil
}

// setAddresses updates only the ranked address list, for the periodic re-rank
// that folds in evidence which arrives after startup (a peer appearing on a link,
// an interface that stops sending). It deliberately skips identity so the common
// case — nothing changed — costs no identity file read or trust-store re-read.
//
// Reports whether the record changed. Order is significant: a different order is
// a different canonical address, which is exactly the change worth republishing.
func (r *registry) setAddresses(ips []string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	ips = trimAddresses(ips)
	if slices.Equal(r.ips, ips) {
		return false
	}
	r.ips = ips
	return true
}

// setName updates the advertised display name. Reports whether it changed.
func (r *registry) setName(name string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.name == name {
		return false
	}
	r.name = name
	return true
}

// record snapshots the current node record.
func (r *registry) record() noderec.NodeRecord {
	r.mu.Lock()
	defer r.mu.Unlock()
	svcs := make(map[noderec.ServiceKey]int, len(r.services))
	for k, v := range r.services {
		svcs[k] = v
	}
	var ip string
	if len(r.ips) > 0 {
		ip = r.ips[0]
	}
	return noderec.NodeRecord{
		SchemaVersion: noderec.SchemaVersion,
		HostUUID:      r.hostUUID,
		ClusterUUID:   r.cluUUID,
		Name:          r.name,
		IP:            ip,
		IPs:           slices.Clone(r.ips),
		Services:      svcs,
	}
}

// txt builds the current _nvpair-node TXT strings.
//
// A descriptor engine's service key stays out of TXT: it is in the record, so
// this node's own directory entry routes it locally, but peers learn it only
// over the authenticated peer-services path. The TXT record is unchanged by
// dynamic engines.
func (r *registry) txt() []string {
	rec := r.record()
	for key := range rec.Services {
		if engines.IsDynamicServiceKey(key) {
			delete(rec.Services, key)
		}
	}
	return rec.TXT()
}
