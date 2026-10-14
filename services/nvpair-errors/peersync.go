// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// peersync.go is the outbound half of cross-node error sync: it turns the
// DiscoveryEvents reconciled from the broker's relay snapshots into a live peer
// set and pushes this node's local-origin snapshot to those peers over cluster
// mTLS.
//
// Push (not pull) is the chosen model: a node sends its full
// local-origin list to every peer whenever that list changes, when a
// new peer appears, and on a periodic heartbeat. The receiver treats
// each push as authoritative for the sender's nodeId (see
// Manager.ReconcilePeer), so the full-snapshot payload makes report,
// clear, and initial-sync collapse into one idempotent, self-healing
// operation — a dropped POST is repaired by the next trigger.

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/errors"
	"nvpair-shared/netpick"
	"nvpair-shared/noderec"
	"nvpair-shared/reach"
)

const (
	// heartbeatInterval is the periodic full re-push cadence. It is the
	// backstop that repairs any state divergence from a dropped push or
	// a peer that was briefly unreachable. Kept well above the discovery
	// scanInterval so a freshly-seen peer is normally synced by the
	// discovered-event push long before the first heartbeat.
	heartbeatInterval = 30 * time.Second
	pushTimeout       = 3 * time.Second
	// maxDrainBytes bounds how much of a peer's response body we drain to make
	// its connection reusable. The ingest endpoint answers 204 or a short error
	// string, so every legitimate body drains in full.
	maxDrainBytes = 8 << 10
)

// PeerSync owns the discovered-peer set and the push transport. One
// instance per process; its Run loop is the single writer of the peers
// map, so reads from the (goroutine-spawned) pushers take the mutex.
// peerTarget is a discovered peer's push destination: every host:port it
// published, in its own ranked order, plus its cluster principal (uuid), which
// selects the pinned mTLS client. Only peers this node holds a pin for are ever
// stored, so an entry here is by construction a confirmed cluster member.
//
// Every address rather than the best one, because a multi-homed peer's canonical
// address can be a direct-connect link only its cabled neighbour can reach; the
// address actually used is confirmed at push time (see reach).
type peerTarget struct {
	candidates []string // "host:port", the peer's own order
	uuid       string   // peer cluster uuid
	address    string   // selected original URL authority
	generation uint64
}

type PeerSync struct {
	mgr         *Manager
	localNodeID string

	// mesh is this node's live cluster state. Peer-sync is cluster mTLS only: a
	// per-peer pinned client, or no push at all.
	mesh *clustertrust.Mesh

	// clients reuses direct or reverse connections without replaying a request.
	// Full-snapshot reconciliation and the later heartbeat own convergence.
	clients    *clustertrust.PeerServiceClient
	clusterDir string

	// addrs remembers which of a peer's published addresses answered, so a
	// heartbeat that repeats every 30 seconds does not re-confirm an address that
	// is already working, and a peer whose canonical address is unreachable from
	// here stops costing a push timeout per round.
	addrs *reach.Chooser

	mu         sync.RWMutex
	peers      map[string]peerTarget // nodeId -> push destination
	generation uint64

	// trigger coalesces local-change notifications: the Manager's
	// onLocalChange callback does a non-blocking send, and the Run loop
	// drains it into a single pushAll. A buffered-size-1 channel means
	// a burst of reports collapses to one push.
	trigger chan struct{}
}

func NewPeerSync(mgr *Manager, mesh *clustertrust.Mesh, clusterDir string) *PeerSync {
	ps := &PeerSync{
		mgr:         mgr,
		localNodeID: mgr.LocalNodeID(),
		mesh:        mesh,
		clusterDir:  clusterDir,
		addrs:       reach.NewChooser(),
		peers:       make(map[string]peerTarget),
		trigger:     make(chan struct{}, 1),
	}
	baseDir := ""
	if clusterDir != "" {
		baseDir = filepath.Dir(filepath.Clean(clusterDir))
	}
	ps.clients = clustertrust.NewPeerServiceClient(mesh, clustertrust.PeerServiceClientOptions{
		PeerClientOptions: clustertrust.PeerClientOptions{Timeout: pushTimeout},
		BaseConfigDir:     baseDir,
		ResolveRoute:      ps.resolveRoute,
		ReuseConnections:  true,
	})
	return ps
}

// resolveRoute uses shared admission authority at every adapter authorization
// boundary. Discovery supplies the endpoint only for an unambiguous principal.
func (ps *PeerSync) resolveRoute(uuid string, service noderec.ServiceKey) (clustertrust.PeerServiceRoute, bool) {
	if service != noderec.ServiceErrors {
		return clustertrust.PeerServiceRoute{}, false
	}
	ps.mesh.Refresh()
	incarnation, ok := ps.mesh.PeerServiceIncarnation(uuid)
	if !ok {
		return clustertrust.PeerServiceRoute{}, false
	}
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	var selected peerTarget
	found := false
	for _, target := range ps.peers {
		if target.uuid == uuid {
			if found {
				return clustertrust.PeerServiceRoute{}, false
			}
			selected, found = target, true
		}
	}
	if !found || selected.address == "" {
		return clustertrust.PeerServiceRoute{}, false
	}
	return clustertrust.PeerServiceRoute{
		Incarnation: incarnation, Address: selected.address,
		Protocol: clustertrust.ReverseTLS, Generation: selected.generation,
	}, true
}

// TriggerPush is the Manager.onLocalChange hook. Non-blocking: if a
// push is already pending, this is a no-op (the pending push will carry
// the latest snapshot anyway).
func (ps *PeerSync) TriggerPush() {
	select {
	case ps.trigger <- struct{}{}:
	default:
	}
}

// Run consumes discovery events and drives pushes until ctx is
// cancelled (at which point discovery closes the events channel).
func (ps *PeerSync) Run(ctx context.Context, events <-chan DiscoveryEvent) {
	heartbeat := time.NewTicker(heartbeatInterval)
	defer heartbeat.Stop()
	defer ps.clients.CloseIdle()

	for {
		select {
		case <-ctx.Done():
			return
		case ev, ok := <-events:
			if !ok {
				return
			}
			ps.handleEvent(ctx, ev)
		case <-ps.trigger:
			ps.pushAll(ctx)
		case <-heartbeat.C:
			ps.pushAll(ctx)
		}
	}
}

// handleEvent maps a discovery event onto the peer set. Our own
// advertisement shows up in discovery too, so self (nodeId ==
// localNodeID) is ignored entirely.
func (ps *PeerSync) handleEvent(ctx context.Context, ev DiscoveryEvent) {
	if ev.Node.ID == ps.localNodeID {
		return
	}

	switch ev.Type {
	case "discovered", "updated":
		uuid := clustertrust.ClusterUUIDFromTXT(ev.Node.TXT)
		// Cluster gate: only sync with a peer this node holds a pin for (a paired
		// cluster member). An unpinned or unknown peer is dropped — we could
		// neither verify it nor build a client for it. Refresh first so a cluster
		// joined, or a peer paired, after startup is recognized without a restart.
		//
		// Unconditional, like the ingress gate: HasPin is false for every peer
		// while this node belongs to no cluster, so an unclustered node keeps an
		// empty peer set and pushes nothing rather than falling back to plaintext.
		ps.mesh.Refresh()
		if !ps.mesh.HasPin(uuid) {
			ps.mu.Lock()
			delete(ps.peers, ev.Node.ID)
			ps.mu.Unlock()
			slog.Debug("errors peer is not a pinned cluster member; not syncing",
				"nodeId", ev.Node.ID, "uuid", uuid)
			return
		}
		candidates := peerHostPorts(ev.Node)
		if len(candidates) == 0 {
			ps.mu.Lock()
			delete(ps.peers, ev.Node.ID)
			ps.mu.Unlock()
			slog.Warn("peer has no reachable address", "nodeId", ev.Node.ID)
			return
		}
		target := peerTarget{candidates: candidates, uuid: uuid}
		ps.mu.Lock()
		ps.generation++
		target.generation = ps.generation
		ps.peers[ev.Node.ID] = target
		ps.mu.Unlock()
		slog.Info("errors peer discovered", "nodeId", ev.Node.ID, "addresses", candidates, "event", ev.Type)
		// Cold-start sync: immediately hand the new peer our current
		// state so it doesn't have to wait for our next local change.
		ps.pushOne(ctx, ev.Node.ID, target)

	case "removed":
		ps.mu.Lock()
		delete(ps.peers, ev.Node.ID)
		ps.mu.Unlock()
		slog.Info("errors peer removed", "nodeId", ev.Node.ID)
		// The departed node's errors are no longer current; drop them
		// from the merged view so a node that goes offline stops
		// haunting everyone's list.
		ps.mgr.EvictNode(ev.Node.ID)
	}
}

// pushAll fans out the current snapshot to every known peer. Each push
// runs in its own goroutine so one slow/unreachable peer can't stall
// the others or the Run loop; we wait for the batch so heartbeat/
// trigger cadence reflects real completion.
func (ps *PeerSync) pushAll(ctx context.Context) {
	// Re-derive membership and pins so a cluster joined or left, and a
	// newly-paired (or unpinned) peer, are reflected before we fan out this round.
	// The adapter also refreshes authority at each request boundary.
	ps.mesh.Refresh()

	ps.mu.RLock()
	targets := make(map[string]peerTarget, len(ps.peers))
	for id, t := range ps.peers {
		targets[id] = t
	}
	ps.mu.RUnlock()

	if len(targets) == 0 {
		return
	}

	body, err := ps.snapshotBody()
	if err != nil {
		slog.Warn("push: marshal snapshot failed", "err", err)
		return
	}

	var wg sync.WaitGroup
	for id, t := range targets {
		wg.Add(1)
		go func(id string, t peerTarget) {
			defer wg.Done()
			ps.post(ctx, id, t, body)
		}(id, t)
	}
	wg.Wait()
}

// pushOne sends the current snapshot to a single peer (used for the
// cold-start sync on discovery).
func (ps *PeerSync) pushOne(ctx context.Context, nodeID string, target peerTarget) {
	body, err := ps.snapshotBody()
	if err != nil {
		slog.Warn("push: marshal snapshot failed", "err", err)
		return
	}
	ps.post(ctx, nodeID, target, body)
}

// snapshotBody marshals the local-origin snapshot into a SyncEnvelope
// once per push round so a fan-out reuses the same bytes.
func (ps *PeerSync) snapshotBody() ([]byte, error) {
	env := errors.SyncEnvelope{
		NodeID: ps.localNodeID,
		Errors: ps.mgr.LocalSnapshot(),
	}
	return json.Marshal(env)
}

// post delivers one snapshot to one peer. Failures are logged and
// dropped — the heartbeat (and the next local change) will retry, which
// is the whole point of an idempotent full-snapshot push.
func (ps *PeerSync) post(ctx context.Context, nodeID string, target peerTarget, body []byte) {
	ps.mesh.Refresh()
	if !ps.mesh.HasPin(target.uuid) {
		slog.Debug("push: peer not pinned; skipping", "nodeId", nodeID, "uuid", target.uuid)
		return
	}

	// Error sync is periodic background work, so confirm the address before
	// spending this round's push on it.
	hostport := ps.addrs.ChooseWithin(ctx, nodeID, target.candidates)
	if hostport == "" {
		slog.Warn("push: peer has no reachable address", "nodeId", nodeID)
		return
	}
	// Publish the selected endpoint only if this push still owns the discovery
	// generation it captured. A removed or replaced target cannot regain a route.
	ps.mu.Lock()
	current, ok := ps.peers[nodeID]
	if !ok || current.uuid != target.uuid || current.generation != target.generation {
		ps.mu.Unlock()
		return
	}
	if current.address != hostport {
		ps.generation++
		current.address, current.generation = hostport, ps.generation
		ps.peers[nodeID] = current
	}
	ps.mu.Unlock()
	client := ps.clients.Client(target.uuid, noderec.ServiceErrors)
	url := "https://" + hostport + "/v1/errors"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		slog.Warn("push: build request failed", "nodeId", nodeID, "url", url, "err", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		// The transport failed against this address, which is what retires it: the
		// next push confirms the peer's list again and can move to an address that
		// works. A response — including a non-2xx — means the address is fine and
		// the peer is answering, so it is left alone.
		ps.addrs.Forget(nodeID)
		slog.Warn("push failed", "nodeId", nodeID, "url", url, "err", err)
		return
	}
	// Drain bounded response data and release the connection for later snapshots.
	defer func() {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxDrainBytes))
		_ = resp.Body.Close()
	}()
	if resp.StatusCode >= 300 {
		slog.Warn("push non-2xx", "nodeId", nodeID, "url", url, "status", resp.StatusCode)
		return
	}
	slog.Debug("push ok", "nodeId", nodeID, "url", url, "status", resp.StatusCode)
}

// peerHostPorts builds every "host:port" worth pushing to for a discovered node,
// in the peer's own ranked order — its ip=/ips= list first, then anything else
// discovery resolved for it, then its hostname when it published no address at
// all. Empty when the node has nowhere to be reached.
//
// Every address, because which of them this host can reach is not something the
// peer can know: it ranks from its own vantage point, and a link only its cabled
// neighbour can use looks as good from there as the LAN. Collapsing the list here
// is what left error sync retrying one unreachable address every heartbeat.
func peerHostPorts(node RawNode) []string {
	if node.Port == 0 {
		return nil
	}
	hosts := netpick.Candidates(node.TXT, node.Addresses)
	if len(hosts) == 0 && node.Host != "" {
		hosts = []string{node.Host}
	}
	out := make([]string, 0, len(hosts))
	for _, host := range hosts {
		out = append(out, net.JoinHostPort(host, strconv.Itoa(node.Port)))
	}
	return out
}
