// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
)

const settingsPath = "/v1/engine-settings/"

// settingsStreamKeepalive is the idle keepalive on a peer settings stream. A
// var so tests can shorten it; production never changes it.
var settingsStreamKeepalive = 10 * time.Second

// settingsStreamIdle drops a subscription after this long without a line.
const settingsStreamIdle = 25 * time.Second

func (s *controlServer) settingsRoutes() []controlRoute {
	return []controlRoute{
		{settingsPath + "set-stated-memory", http.MethodPost, controlMutation, "settings-set-stated-memory", s.settingsProfileHandler("set-stated-memory")},
		{settingsPath + "clear-stated-memory", http.MethodPost, controlMutation, "settings-clear-stated-memory", s.settingsProfileHandler("clear-stated-memory")},
		{settingsPath + "get", http.MethodPost, controlRead, "settings-get", s.settingsHandler("get")},
		{settingsPath + "inspect", http.MethodPost, controlRead, "settings-inspect", s.settingsHandler("inspect")},
		{settingsPath + "preview", http.MethodPost, controlRead, "settings-preview", s.settingsHandler("preview")},
		{settingsPath + "apply", http.MethodPost, controlMutation, "settings-apply", s.settingsHandler("apply")},
		{settingsPath + "events", http.MethodGet, controlRead, "settings-events", ignoreCaller(s.streamSettings)},
	}
}

func (s *controlServer) settingsProfileHandler(method string) pinnedPeerHandler {
	return func(w http.ResponseWriter, r *http.Request, caller string) {
		if s.exec.memoryProfilesParent == nil {
			http.Error(w, "settings coordinator unavailable", http.StatusServiceUnavailable)
			return
		}
		var engine string
		raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 64<<10))
		if err != nil {
			http.Error(w, "invalid memory profile request", http.StatusBadRequest)
			return
		}
		relay := settings.Relay{Method: method, Caller: caller}
		if method == "set-stated-memory" {
			p, err := modelconfig.DecodeSetStatedMemoryRequest(raw)
			if err != nil {
				http.Error(w, "invalid memory profile request", http.StatusBadRequest)
				return
			}
			p.NodeID = ""
			relay.SetStatedMemory = &p
			engine = p.Engine
		} else {
			p, err := modelconfig.DecodeClearStatedMemoryRequest(raw)
			if err != nil {
				http.Error(w, "invalid memory profile request", http.StatusBadRequest)
				return
			}
			p.NodeID = ""
			relay.ClearStatedMemory = &p
			engine = p.Engine
		}
		if err := s.exec.remoteVeto(engine); err != nil {
			var veto *localVetoError
			if errors.As(err, &veto) {
				writeCodedError(w, veto.code, err.Error())
			} else {
				http.Error(w, "target engine unavailable", http.StatusServiceUnavailable)
			}
			return
		}
		result, err := s.exec.memoryProfilesParent(r.Context(), relay)
		if err != nil {
			var coded *codedError
			if errors.As(err, &coded) && wireCodes[coded.code] != "" {
				writeCodedError(w, coded.code, err.Error())
			} else if errors.As(err, &coded) && coded.code == -32602 {
				http.Error(w, err.Error(), http.StatusBadRequest)
			} else {
				http.Error(w, err.Error(), http.StatusConflict)
			}
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(result)
	}
}

func (m *Manager) remoteStatedMemory(ctx context.Context, msg *Message) {
	method := strings.TrimPrefix(msg.Method, "engine:remote-")
	var engine, node, model string
	var payload any
	if method == "set-stated-memory" {
		p, err := modelconfig.DecodeSetStatedMemoryRequest(msg.Params)
		if err != nil {
			m.respondOrErr(msg, nil, modelSettingsParamsError(err))
			return
		}
		engine, node, model = p.Engine, p.NodeID, p.ModelID
		p.NodeID = ""
		payload = p
	} else {
		p, err := modelconfig.DecodeClearStatedMemoryRequest(msg.Params)
		if err != nil {
			m.respondOrErr(msg, nil, modelSettingsParamsError(err))
			return
		}
		engine, node, model = p.Engine, p.NodeID, p.ModelID
		p.NodeID = ""
		payload = p
	}
	peer, ok := m.peers.lookup(node)
	if node == "" || !ok || m.mesh == nil {
		m.respondOrErr(msg, nil, fmt.Errorf("memory profile target unavailable"))
		return
	}
	client, err := m.remoteClient(ctx, peer)
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	raw, err := client.postJSON(ctx, settingsPath+method, engine, payload)
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	result, err := modelconfig.DecodeInspectResult(raw)
	if err == nil && (result.NodeID != node || result.Engine != engine || result.ModelID != model) {
		err = fmt.Errorf("memory profile target reply identity mismatch")
	}
	m.respondOrErr(msg, result, err)
}

// settingsHandler relays one local broker settings method for a pinned peer.
// The registry has already validated the method and, for apply, the caller's
// remote-management grant.
func (s *controlServer) settingsHandler(method string) pinnedPeerHandler {
	return func(w http.ResponseWriter, r *http.Request, caller string) {
		if method == "inspect" && s.exec.inspectParent == nil || method != "inspect" && s.exec.settingsParent == nil {
			http.Error(w, "settings coordinator unavailable", 503)
			return
		}
		raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 64*1024))
		if err != nil {
			http.Error(w, "invalid settings request", http.StatusBadRequest)
			return
		}
		if err := validateServingModelsJSON(raw, "settings", "serving"); err != nil {
			http.Error(w, "invalid settings request", http.StatusBadRequest)
			return
		}
		var p settings.Request
		var inspection modelconfig.InspectRequest
		if method == "inspect" {
			inspect, err := modelconfig.DecodeInspectRequest(raw)
			if err != nil {
				http.Error(w, "invalid model inspection request", http.StatusBadRequest)
				return
			}
			// Preserve the complete typed identity through the existing inspection
			// relay. This peer can address only this target, never another hop.
			inspection = inspect
			inspection.NodeID = ""
		} else {
			p, err = settings.DecodeRequest(raw)
			if err != nil {
				http.Error(w, "invalid settings request", http.StatusBadRequest)
				return
			}
		}
		// A peer can only address this target node, never forward another hop.
		p.NodeID = ""
		if method != "get" && method != "inspect" {
			if err := s.exec.RejectRemoteCORSChange(p); err != nil {
				http.Error(w, err.Error(), 403)
				return
			}
		}
		ctx, cancel := context.WithCancel(r.Context())
		defer cancel()
		go func() {
			ticker := time.NewTicker(time.Second)
			defer ticker.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
					s.mesh.Refresh()
					if _, ok := s.mesh.VerifyClientPin(r); !ok {
						cancel()
						return
					}
				}
			}
		}()
		var result json.RawMessage
		if method == "inspect" {
			result, err = s.exec.inspectParent(ctx, inspection, caller)
		} else {
			result, err = s.exec.settingsParent(ctx, method, p, caller)
		}
		if err != nil {
			// A local veto or a refused management operation is a typed 409
			// the caller maps back to its code; anything else is a plain 409.
			var relayErr *codedError
			if errors.As(err, &relayErr) && relayErr.code == -32602 {
				http.Error(w, err.Error(), http.StatusBadRequest)
			} else if errors.As(err, &relayErr) && wireCodes[relayErr.code] != "" {
				writeCodedError(w, relayErr.code, err.Error())
			} else {
				http.Error(w, err.Error(), 409)
			}
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(result)
	}
}

func (s *controlServer) streamSettings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", 405)
		return
	}
	if s.exec.settingsParent == nil {
		http.Error(w, "settings coordinator unavailable", 503)
		return
	}
	ch, closeSub, ok := s.exec.settingsHub.Subscribe()
	if !ok {
		http.Error(w, "too many settings subscriptions", 503)
		return
	}
	defer closeSub()
	clustertrust.ServeSnapshotStream(w, r, clustertrust.SnapshotStreamServer[[]settings.Snapshot]{
		Keepalive:    settingsStreamKeepalive,
		WriteTimeout: 5 * time.Second,
		Authorize: func() bool {
			s.mesh.Refresh()
			_, ok := s.mesh.VerifyClientPin(r)
			return ok
		},
		Updates: ch,
	})
}

func (m *Manager) remoteSettings(ctx context.Context, msg *Message) {
	if msg.Method == "engine:remote-set-stated-memory" || msg.Method == "engine:remote-clear-stated-memory" {
		m.remoteStatedMemory(ctx, msg)
		return
	}
	if msg.Method == "engine:remote-inspect-model" {
		m.remoteInspectModel(ctx, msg)
		return
	}
	p, err := settings.DecodeRequest(msg.Params)
	if err != nil {
		m.respondOrErr(msg, nil, modelSettingsParamsError(err))
		return
	}
	peer, ok := m.peers.lookup(p.NodeID)
	if !ok || m.mesh == nil {
		m.respondOrErr(msg, nil, fmt.Errorf("settings unavailable: device is offline or does not support settings"))
		return
	}
	client, err := m.remoteClient(ctx, peer)
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	method := strings.TrimSuffix(strings.TrimPrefix(msg.Method, "engine:remote-"), "-settings")
	p.NodeID = ""
	result, err := client.postJSON(ctx, settingsPath+method, p.Engine, p)
	m.respondOrErr(msg, result, err)
}

// Settings and engine status are pushed directly by each authority. Discovery
// polling here only manages connection lifetimes; neither is polled from peers.
func (m *Manager) watchPeerSettings(ctx context.Context) {
	active := map[string]context.CancelFunc{}
	defer func() {
		for _, cancel := range active {
			cancel()
		}
	}()
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if m.mesh == nil {
				continue
			}
			m.mesh.Refresh()
			m.peers.mu.RLock()
			peers := make(map[string]ecPeer, len(m.peers.peers))
			for k, v := range m.peers.peers {
				peers[k] = v
			}
			m.peers.mu.RUnlock()
			for node, cancel := range active {
				p, exists := peers[node]
				_, admitted := m.mesh.PeerServiceIncarnation(p.clusterUUID)
				if !exists || !admitted {
					cancel()
					delete(active, node)
				}
			}
			for node, peer := range peers {
				// HasPin includes self-trust. Settings subscriptions need the
				// same peer-only authority as the transport, keyed by the cluster
				// principal rather than the separately advertised host UUID.
				_, admitted := m.mesh.PeerServiceIncarnation(peer.clusterUUID)
				if active[node] != nil || len(active) >= 64 || !admitted {
					continue
				}
				streamCtx, cancel := context.WithCancel(ctx)
				active[node] = cancel
				go m.keepPeerStream(streamCtx, node, m.consumeSettingsStream, func() {
					m.exec.notify("engine:settings-disconnected", map[string]string{"nodeId": node})
				})
				go m.keepPeerStream(streamCtx, node, m.consumeEnginesStream, func() {})
			}
		}
	}
}

func (m *Manager) notifyPeerSettings(nodeID string, snapshots []settings.Snapshot) {
	for _, snapshot := range snapshots {
		if _, ok := engines.ByName(snapshot.Engine); !ok {
			continue
		}
		snapshot.NodeID = nodeID
		m.exec.notify("engine:settings-changed", snapshot)
	}
}

// keepPeerStream runs consume for node until ctx ends or the node leaves the
// directory, calling ended after each stream and pausing 3 s before the next.
func (m *Manager) keepPeerStream(ctx context.Context, node string, consume func(context.Context, ecPeer), ended func()) {
	for ctx.Err() == nil {
		current, exists := m.peers.lookup(node)
		if !exists {
			return
		}
		consume(ctx, current)
		ended()
		select {
		case <-ctx.Done():
			return
		case <-time.After(3 * time.Second):
		}
	}
}

// openPeerStream opens a read-only snapshot stream on a peer's ec surface. It
// returns the 200 response and the stream's current check, which turns false
// when the peer's route, admission or pinned identity changes.
func (m *Manager) openPeerStream(ctx context.Context, peer ecPeer, path string) (*http.Response, func() bool, bool) {
	client, err := m.remoteClient(ctx, peer)
	if err != nil {
		return nil, nil, false
	}
	route, ok := m.resolveControlRoute(peer.clusterUUID, noderec.ServiceEngineControl)
	config, pinned := m.mesh.ClientTLSConfig(peer.clusterUUID)
	if !ok || !pinned || len(config.Certificates) == 0 {
		return nil, nil, false
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, client.base+path, nil)
	if err != nil {
		return nil, nil, false
	}
	response, err := client.controlClient(false).Do(req)
	if err != nil {
		client.forgetAddress()
		return nil, nil, false
	}
	if response.StatusCode != http.StatusOK {
		response.Body.Close()
		return nil, nil, false
	}
	// The transport checks admission before dispatch. This subscription owner
	// also retires an active body when either identity or the endpoint changes;
	// watchPeerSettings owns the subsequent read-only reconnect.
	current := func() bool {
		now, admitted := m.resolveControlRoute(peer.clusterUUID, noderec.ServiceEngineControl)
		live, pinned := m.mesh.ClientTLSConfig(peer.clusterUUID)
		return admitted && now == route && pinned && len(live.Certificates) != 0 &&
			bytes.Equal(live.Certificates[0].Certificate[0], config.Certificates[0].Certificate[0]) &&
			response.TLS != nil && len(response.TLS.PeerCertificates) != 0 &&
			live.VerifyPeerCertificate([][]byte{response.TLS.PeerCertificates[0].Raw}, nil) == nil
	}
	return response, current, true
}

func (m *Manager) consumeSettingsStream(parent context.Context, peer ecPeer) {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	response, current, ok := m.openPeerStream(ctx, peer, settingsPath+"events")
	if !ok {
		return
	}
	defer response.Body.Close()
	_ = clustertrust.ReadSnapshotStream(ctx, cancel, response.Body, clustertrust.SnapshotStreamClient[[]settings.Snapshot]{
		Idle:    settingsStreamIdle,
		MaxLine: 256 * 1024,
		Current: current,
		Decode: func(b []byte) ([]settings.Snapshot, error) {
			var v []settings.Snapshot
			return v, json.Unmarshal(b, &v)
		},
		OnFrame: func(s []settings.Snapshot) error {
			m.notifyPeerSettings(peer.nodeID, s)
			return nil
		},
	})
}
