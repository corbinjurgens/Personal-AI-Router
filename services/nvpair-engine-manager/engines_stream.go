// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// engines_stream.go streams this node's engine status list to pinned peers
// (GET /v1/engines/events) and keeps, per peer, the latest list that peer
// streamed back. engine:remote-get-installed answers from that cache, and each
// change reaches the UI as engine:remote-installed-changed. Peers no longer
// poll GET /v1/engines.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
)

// enginesStreamKeepalive is the idle keepalive on a peer engine-status stream.
// A var so tests can shorten it; production never changes it.
var enginesStreamKeepalive = 10 * time.Second

const (
	// enginesStreamIdle drops a subscription after this long without a line,
	// and bounds how old a cached peer list may be when it is served.
	enginesStreamIdle = 25 * time.Second
	// enginesRebuildEvery catches status changes no engine:state-changed names.
	enginesRebuildEvery = 10 * time.Second
	// enginesRebuildDelay coalesces a burst of local state changes.
	enginesRebuildDelay = 250 * time.Millisecond
	// enginesStreamMaxLine matches the former GET /v1/engines response cap.
	enginesStreamMaxLine = 8 << 20
)

// errRemoteEnginesUnavailable answers engine:remote-get-installed while the
// peer's engine-status stream is not live.
var errRemoteEnginesUnavailable = errors.New("remote engines: engine status stream is not connected")

// enginesPublisher rebuilds the local {"engines":[...]} body and publishes it
// when its bytes change.
type enginesPublisher struct {
	hub    clustertrust.SnapshotHub[json.RawMessage]
	signal chan struct{}
}

func newEnginesPublisher() *enginesPublisher {
	return &enginesPublisher{signal: make(chan struct{}, 1)}
}

// poke asks for a rebuild after enginesRebuildDelay. It never blocks.
func (p *enginesPublisher) poke() {
	if p == nil {
		return
	}
	select {
	case p.signal <- struct{}{}:
	default:
	}
}

// run rebuilds at start, every enginesRebuildEvery and after each poke until
// ctx ends.
func (p *enginesPublisher) run(ctx context.Context, list func() []EngineStatus) {
	var last []byte
	rebuild := func() {
		raw, err := json.Marshal(map[string]any{"engines": list()})
		if err != nil || bytes.Equal(raw, last) {
			return
		}
		last = raw
		p.hub.Publish(json.RawMessage(raw))
	}
	rebuild()
	ticker := time.NewTicker(enginesRebuildEvery)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			rebuild()
		case <-p.signal:
			timer := time.NewTimer(enginesRebuildDelay)
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case <-timer.C:
			}
			rebuild()
		}
	}
}

// handleEngineEvents streams the engine status list to a pinned peer: the
// current list at once, again on change, and a null keepalive otherwise.
func (s *controlServer) handleEngineEvents(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", "GET")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if s.exec.enginesEvents == nil {
		http.Error(w, "engine status unavailable", http.StatusServiceUnavailable)
		return
	}
	ch, closeSub, ok := s.exec.enginesEvents.hub.Subscribe()
	if !ok {
		http.Error(w, "too many engine status subscriptions", http.StatusServiceUnavailable)
		return
	}
	defer closeSub()
	w.Header().Set("Cache-Control", "no-store")
	clustertrust.ServeSnapshotStream(w, r, clustertrust.SnapshotStreamServer[json.RawMessage]{
		Keepalive:    enginesStreamKeepalive,
		WriteTimeout: 5 * time.Second,
		Authorize: func() bool {
			s.mesh.Refresh()
			_, ok := s.mesh.VerifyClientPin(r)
			return ok
		},
		Updates: ch,
	})
}

// remoteEnginesEntry is one peer's live engine-status stream state.
type remoteEnginesEntry struct {
	engines json.RawMessage // the peer's engines array, compacted
	lineAt  time.Time       // last line, keepalives included
	current func() bool     // the stream's route and admission are unchanged
}

// remoteEngineCache holds, per peer node ID, the entry of its live stream.
type remoteEngineCache struct {
	mu    sync.Mutex
	peers map[string]*remoteEnginesEntry
}

// answer returns the peer's cached {"engines":[...]} body when its stream is
// live and still current.
func (c *remoteEngineCache) answer(nodeID string) (json.RawMessage, error) {
	c.mu.Lock()
	entry := c.peers[nodeID]
	var engines json.RawMessage
	var current func() bool
	fresh := false
	if entry != nil && entry.engines != nil {
		engines, current = entry.engines, entry.current
		fresh = time.Since(entry.lineAt) <= enginesStreamIdle
	}
	c.mu.Unlock()
	// current takes the mesh and directory locks; never call it under mu.
	if !fresh || !current() {
		return nil, errRemoteEnginesUnavailable
	}
	return json.Marshal(map[string]json.RawMessage{"engines": engines})
}

// consumeEnginesStream subscribes to one peer's engine-status stream until it
// ends, keeping the peer's cache entry while it is live and notifying each
// change of its list.
func (m *Manager) consumeEnginesStream(parent context.Context, peer ecPeer) {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	response, current, ok := m.openPeerStream(ctx, peer, controlEnginesPath+"/events")
	if !ok {
		return
	}
	defer response.Body.Close()
	entry := &remoteEnginesEntry{current: current}
	cache := &m.remoteEngines
	cache.mu.Lock()
	if cache.peers == nil {
		cache.peers = make(map[string]*remoteEnginesEntry)
	}
	if cache.peers[peer.nodeID] != nil {
		// A replacement stream for this peer is already live; this one was
		// superseded while it was opening.
		cache.mu.Unlock()
		return
	}
	cache.peers[peer.nodeID] = entry
	cache.mu.Unlock()
	defer func() {
		cache.mu.Lock()
		if cache.peers[peer.nodeID] == entry {
			delete(cache.peers, peer.nodeID)
		}
		cache.mu.Unlock()
	}()
	touch := func() {
		cache.mu.Lock()
		entry.lineAt = time.Now()
		cache.mu.Unlock()
	}
	_ = clustertrust.ReadSnapshotStream(ctx, cancel, response.Body, clustertrust.SnapshotStreamClient[json.RawMessage]{
		Idle:        enginesStreamIdle,
		MaxLine:     enginesStreamMaxLine,
		Current:     current,
		OnKeepalive: touch,
		Decode: func(b []byte) (json.RawMessage, error) {
			var frame struct {
				Engines []json.RawMessage `json:"engines"`
			}
			if err := json.Unmarshal(b, &frame); err != nil {
				return nil, err
			}
			if frame.Engines == nil {
				frame.Engines = []json.RawMessage{}
			}
			// Marshal compacts each element, so equal lists compare equal.
			raw, err := json.Marshal(frame.Engines)
			return json.RawMessage(raw), err
		},
		OnFrame: func(engines json.RawMessage) error {
			cache.mu.Lock()
			entry.lineAt = time.Now()
			changed := !bytes.Equal(entry.engines, engines)
			if changed {
				entry.engines = engines
			}
			cache.mu.Unlock()
			if changed {
				m.exec.notify("engine:remote-installed-changed", map[string]any{"nodeId": peer.nodeID, "engines": engines})
			}
			return nil
		},
	})
}
