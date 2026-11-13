// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// peersync.go is the subscribing half of cross-node error sync: it turns the
// DiscoveryEvents reconciled from the broker's relay snapshots into a live peer
// set and holds one snapshot stream (GET /v1/errors/events, stream.go) to each
// pinned peer over cluster mTLS.
//
// Every frame is the peer's full local-origin list, and the receiver treats it
// as authoritative for the peer's nodeId (see Manager.ReconcilePeer), so report,
// clear and initial sync collapse into one idempotent operation. A stream starts
// every (re)connect with the full snapshot, so a change missed while a stream was
// down cannot persist and no periodic resend is needed. A peer's entries stay
// when its stream ends; only discovery removal evicts them.

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"net"
	"net/http"
	"path/filepath"
	"slices"
	"strconv"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/errors"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/netpick"
	"nvpair-shared/noderec"
	"nvpair-shared/reach"
)

const (
	errorsStreamPath = "/v1/errors/events"
	// errorsStreamIdle drops a stream that sent no line, keepalives included,
	// for two and a half keepalive intervals.
	errorsStreamIdle = 25 * time.Second
	// errorsStreamConnectTimeout bounds connect and response headers. It is a
	// cancel timer, not a context deadline: a reverse dial copies a deadline
	// onto the yamux stream, which would end the open stream.
	errorsStreamConnectTimeout = 10 * time.Second
	errorsStreamRetryMax       = 30 * time.Second
)

// errorsStreamRetryMin is the first reconnect delay; it doubles up to
// errorsStreamRetryMax and resets after a stream that delivered a frame.
var errorsStreamRetryMin = time.Second

// peerTarget is a discovered peer's stream destination: every host:port it
// published, in its own ranked order, plus its cluster principal (uuid), which
// selects the pinned mTLS client. Only peers this node holds a pin for are ever
// stored, so an entry here is by construction a confirmed cluster member.
//
// Every address rather than the best one, because a multi-homed peer's canonical
// address can be a direct-connect link only its cabled neighbour can reach; the
// address actually used is confirmed at connect time (see reach).
type peerTarget struct {
	candidates []string // "host:port", the peer's own order
	uuid       string   // peer cluster uuid
	address    string   // selected original URL authority
	generation uint64   // changes with the endpoint; part of the route
	stream     uint64   // the stream that owns this target
	cancel     context.CancelFunc
}

// PeerSync owns the discovered-peer set and one stream per peer. Run is the
// only caller of handleEvent and restartStreams; stream goroutines also update
// a target's selected address, so the map is guarded by mu.
type PeerSync struct {
	mgr         *Manager
	localNodeID string

	// mesh is this node's live cluster state. Peer-sync is cluster mTLS only: a
	// per-peer pinned client, or no stream at all.
	mesh *clustertrust.Mesh

	// clients opens streams directly or over a reverse session. It has no total
	// timeout, because a stream stays open.
	clients    *clustertrust.PeerServiceClient
	clusterDir string

	// addrs remembers which of a peer's published addresses answered, so a
	// reconnect does not re-confirm an address that is already working.
	addrs *reach.Chooser

	// discovered is every non-self node discovery last reported, pinned or
	// not, so a membership change can gate them again. Only the goroutine
	// calling handleEvent and restartStreams (Run) uses it.
	discovered map[string]RawNode

	mu         sync.RWMutex
	peers      map[string]peerTarget // nodeId -> stream destination
	generation uint64

	streams sync.WaitGroup
	// resubscribe coalesces membership changes into one restart of every
	// stream (buffer 1, non-blocking send).
	resubscribe chan struct{}
}

func NewPeerSync(mgr *Manager, mesh *clustertrust.Mesh, clusterDir string) *PeerSync {
	ps := &PeerSync{
		mgr:         mgr,
		localNodeID: mgr.LocalNodeID(),
		mesh:        mesh,
		clusterDir:  clusterDir,
		addrs:       reach.NewChooser(),
		discovered:  make(map[string]RawNode),
		peers:       make(map[string]peerTarget),
		resubscribe: make(chan struct{}, 1),
	}
	baseDir := ""
	if clusterDir != "" {
		baseDir = filepath.Dir(filepath.Clean(clusterDir))
	}
	ps.clients = clustertrust.NewPeerServiceClient(mesh, clustertrust.PeerServiceClientOptions{
		BaseConfigDir: baseDir,
		ResolveRoute:  ps.resolveRoute,
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

// Resubscribe gates every discovered peer again and restarts every admitted
// peer's stream. Called when this node joins or leaves a cluster, because the
// authority every stream was opened under changed.
// Non-blocking: a pending restart covers this one.
func (ps *PeerSync) Resubscribe() {
	select {
	case ps.resubscribe <- struct{}{}:
	default:
	}
}

// Run consumes discovery events and membership changes until ctx is cancelled
// (at which point discovery closes the events channel). Every stream ends
// before it returns.
func (ps *PeerSync) Run(ctx context.Context, events <-chan DiscoveryEvent) {
	defer ps.stopStreams()
	for {
		select {
		case <-ctx.Done():
			return
		case ev, ok := <-events:
			if !ok {
				return
			}
			ps.handleEvent(ctx, ev)
		case <-ps.resubscribe:
			ps.restartStreams(ctx)
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
		ps.discovered[ev.Node.ID] = ev.Node
		ps.applyPeer(ctx, ev.Node, false)

	case "removed":
		delete(ps.discovered, ev.Node.ID)
		ps.dropPeer(ev.Node.ID)
		slog.Info("errors peer removed", "nodeId", ev.Node.ID)
		// The departed node's errors are no longer current; drop them
		// from the merged view so a node that goes offline stops
		// haunting everyone's list.
		ps.mgr.EvictNode(ev.Node.ID)
	}
}

// applyPeer gates one discovered node and starts, keeps or drops its stream.
// restart replaces a running stream even when nothing about the node changed.
func (ps *PeerSync) applyPeer(ctx context.Context, node RawNode, restart bool) {
	uuid := clustertrust.ClusterUUIDFromTXT(node.TXT)
	// Cluster gate: only sync with a peer this node holds a pin for (a paired
	// cluster member). An unpinned or unknown peer is dropped — we could
	// neither verify it nor build a client for it. Refresh first so a cluster
	// joined, or a peer paired, after startup is recognized without a restart.
	//
	// Unconditional, like the ingress gate: HasPin is false for every peer
	// while this node belongs to no cluster, so an unclustered node keeps an
	// empty peer set and subscribes to nothing rather than falling back to
	// plaintext.
	ps.mesh.Refresh()
	if !ps.mesh.HasPin(uuid) {
		ps.dropPeer(node.ID)
		slog.Debug("errors peer is not a pinned cluster member; not syncing",
			"nodeId", node.ID, "uuid", uuid)
		return
	}
	candidates := peerHostPorts(node)
	if len(candidates) == 0 {
		ps.dropPeer(node.ID)
		slog.Warn("peer has no reachable address", "nodeId", node.ID)
		return
	}
	ps.mu.Lock()
	defer ps.mu.Unlock()
	if current, ok := ps.peers[node.ID]; ok && !restart && current.uuid == uuid && slices.Equal(current.candidates, candidates) {
		return
	}
	slog.Info("errors peer stream starting", "nodeId", node.ID, "addresses", candidates, "restart", restart)
	ps.startStreamLocked(ctx, node.ID, peerTarget{candidates: candidates, uuid: uuid})
}

// dropPeer cancels nodeID's stream and forgets the peer. Its entries stay.
func (ps *PeerSync) dropPeer(nodeID string) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	if target, ok := ps.peers[nodeID]; ok {
		target.cancel()
		delete(ps.peers, nodeID)
	}
}

// restartStreams gates every discovered node again under the current cluster
// authority and gives each admitted one a new stream. A node discovered before
// this node held its pin (for example, before joining the cluster) is picked
// up here, because discovery sends nothing new for an unchanged record.
func (ps *PeerSync) restartStreams(ctx context.Context) {
	for _, node := range ps.discovered {
		ps.applyPeer(ctx, node, true)
	}
	slog.Info("errors peer streams restarted", "discovered", len(ps.discovered))
}

// startStreamLocked stores target for nodeID under a new generation and starts
// its stream, cancelling the stream it replaces. The caller holds mu.
func (ps *PeerSync) startStreamLocked(ctx context.Context, nodeID string, target peerTarget) {
	if previous, ok := ps.peers[nodeID]; ok {
		previous.cancel()
	}
	ps.generation++
	streamCtx, cancel := context.WithCancel(ctx)
	target.generation, target.stream, target.cancel = ps.generation, ps.generation, cancel
	ps.peers[nodeID] = target
	ps.streams.Add(1)
	go func() {
		defer ps.streams.Done()
		defer cancel()
		ps.runStream(streamCtx, nodeID, target.stream)
	}()
}

// stopStreams ends every stream and waits for them.
func (ps *PeerSync) stopStreams() {
	ps.mu.Lock()
	for _, target := range ps.peers {
		target.cancel()
	}
	ps.mu.Unlock()
	ps.streams.Wait()
	ps.clients.CloseIdle()
}

// runStream keeps one stream to nodeID open until ctx ends, reconnecting after
// a delay that starts at errorsStreamRetryMin, doubles up to
// errorsStreamRetryMax with ±20 % jitter, and resets after a stream that
// delivered a frame.
func (ps *PeerSync) runStream(ctx context.Context, nodeID string, stream uint64) {
	delay := errorsStreamRetryMin
	for {
		if ps.streamOnce(ctx, nodeID, stream) {
			delay = errorsStreamRetryMin
		}
		retry := time.NewTimer(time.Duration(float64(delay) * (0.8 + 0.4*rand.Float64())))
		select {
		case <-ctx.Done():
			retry.Stop()
			return
		case <-retry.C:
		}
		delay = min(2*delay, errorsStreamRetryMax)
	}
}

// streamOnce opens one stream to nodeID and applies its frames until it ends.
// It reports whether the stream delivered a frame.
func (ps *PeerSync) streamOnce(ctx context.Context, nodeID string, stream uint64) bool {
	ps.mu.RLock()
	target, ok := ps.peers[nodeID]
	ps.mu.RUnlock()
	if !ok || target.stream != stream {
		return false
	}
	// Refuse an unadmitted peer before probing its addresses.
	ps.mesh.Refresh()
	if _, admitted := ps.mesh.PeerServiceIncarnation(target.uuid); !admitted {
		slog.Debug("errors stream: peer not admitted", "nodeId", nodeID)
		return false
	}
	// Confirm the address before spending a connect on it.
	hostport := ps.addrs.ChooseWithin(ctx, nodeID, target.candidates)
	if hostport == "" {
		slog.Warn("errors stream: peer has no reachable address", "nodeId", nodeID)
		return false
	}
	// Publish the selected endpoint only while this stream still owns the
	// target. A removed or replaced target cannot regain a route.
	ps.mu.Lock()
	current, ok := ps.peers[nodeID]
	if !ok || current.stream != stream {
		ps.mu.Unlock()
		return false
	}
	if current.address != hostport {
		ps.generation++
		current.address, current.generation = hostport, ps.generation
		ps.peers[nodeID] = current
	}
	ps.mu.Unlock()
	route, ok := ps.resolveRoute(target.uuid, noderec.ServiceErrors)
	if !ok {
		slog.Debug("errors stream: peer route not admitted", "nodeId", nodeID)
		return false
	}

	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	url := "https://" + hostport + errorsStreamPath
	req, err := http.NewRequestWithContext(streamCtx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	watchdog := time.AfterFunc(errorsStreamConnectTimeout, cancel)
	resp, err := ps.clients.Client(target.uuid, noderec.ServiceErrors).Do(req)
	if !watchdog.Stop() && err == nil {
		resp.Body.Close()
		err = fmt.Errorf("connect timed out after %s", errorsStreamConnectTimeout)
	}
	if err != nil {
		if ctx.Err() == nil {
			// A transport failure retires the address, so the next attempt
			// confirms the peer's list again. A response, even a refusal, means
			// the address is fine.
			ps.addrs.Forget(nodeID)
			slog.Warn("errors stream connect failed", "nodeId", nodeID, "url", url, "err", err)
		}
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		slog.Warn("errors stream refused", "nodeId", nodeID, "url", url, "status", resp.StatusCode)
		return false
	}
	slog.Debug("errors stream connected", "nodeId", nodeID, "url", url)
	delivered := false
	err = clustertrust.ReadSnapshotStream(streamCtx, cancel, resp.Body, clustertrust.SnapshotStreamClient[errors.SyncEnvelope]{
		Idle:    errorsStreamIdle,
		MaxLine: jsonrpc.DefaultFrameBytes,
		// The route carries admission, both certificates and the endpoint, so
		// a revoked pin, a readmission or an address change ends the stream.
		Current: func() bool {
			now, ok := ps.resolveRoute(target.uuid, noderec.ServiceErrors)
			return ok && now == route
		},
		Decode: func(line []byte) (errors.SyncEnvelope, error) {
			var env errors.SyncEnvelope
			return env, json.Unmarshal(line, &env)
		},
		OnFrame: func(env errors.SyncEnvelope) error {
			// A peer may only report its own origin: the pinned principal.
			if env.NodeID != target.uuid {
				return fmt.Errorf("frame nodeId %q does not match pinned peer %q", env.NodeID, target.uuid)
			}
			// A superseded stream must not apply its last frame over the
			// newer stream's; holding mu keeps a replacement from starting
			// between the check and the reconcile.
			ps.mu.RLock()
			defer ps.mu.RUnlock()
			if owner, ok := ps.peers[nodeID]; !ok || owner.stream != stream || streamCtx.Err() != nil {
				return context.Canceled
			}
			ps.mgr.ReconcilePeer(env.NodeID, env.Errors)
			delivered = true
			return nil
		},
	})
	slog.Debug("errors stream ended", "nodeId", nodeID, "err", err)
	return delivered
}

// peerHostPorts builds every "host:port" worth subscribing to for a discovered
// node, in the peer's own ranked order — its ip=/ips= list first, then anything
// else discovery resolved for it. The node's Host is a display name and is never
// dialed, so a node with no address has nowhere to be reached and gets an empty slice.
//
// Every address, because which of them this host can reach is not something the
// peer can know: it ranks from its own vantage point, and a link only its cabled
// neighbour can use looks as good from there as the LAN. Collapsing the list here
// is what left error sync retrying one unreachable address on every attempt.
func peerHostPorts(node RawNode) []string {
	if node.Port == 0 {
		return nil
	}
	hosts := netpick.Candidates(node.TXT, node.Addresses)
	out := make([]string, 0, len(hosts))
	for _, host := range hosts {
		out = append(out, net.JoinHostPort(host, strconv.Itoa(node.Port)))
	}
	return out
}
