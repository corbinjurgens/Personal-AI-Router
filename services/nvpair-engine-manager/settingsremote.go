// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
)

const settingsPath = "/v1/engine-settings/"

func (s *controlServer) settingsRoutes() []controlRoute {
	return []controlRoute{
		{settingsPath + "get", http.MethodPost, controlRead, "settings-get", s.settingsHandler("get")},
		{settingsPath + "inspect", http.MethodPost, controlRead, "settings-inspect", s.settingsHandler("inspect")},
		{settingsPath + "preview", http.MethodPost, controlRead, "settings-preview", s.settingsHandler("preview")},
		{settingsPath + "apply", http.MethodPost, controlMutation, "settings-apply", s.settingsHandler("apply")},
		{settingsPath + "events", http.MethodGet, controlRead, "settings-events", ignoreCaller(s.streamSettings)},
	}
}

// settingsHandler relays one local broker settings method for a pinned peer.
// The registry has already validated the method and, for apply, the caller's
// remote-management grant.
func (s *controlServer) settingsHandler(method string) pinnedPeerHandler {
	return func(w http.ResponseWriter, r *http.Request, caller string) {
		if s.exec.settingsParent == nil {
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
		if method == "inspect" {
			inspect, err := modelconfig.DecodeInspectRequest(raw)
			if err != nil {
				http.Error(w, "invalid model inspection request", http.StatusBadRequest)
				return
			}
			// settingsParent retains its existing executor signature. Resolution
			// is an internal carrier only; settingsRelay.call converts it to the
			// shared Relay.Inspect pointer (and no Relay.Request) before sending.
			// NodeID is cleared here to prohibit forwarding another target hop.
			p = settings.Request{Engine: inspect.Engine, Resolution: inspect.ModelID}
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
		result, err := s.exec.settingsParent(ctx, method, p, caller)
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
	w.Header().Set("Content-Type", "application/x-ndjson")
	controller := http.NewResponseController(w)
	send := func(value any) bool {
		_ = controller.SetWriteDeadline(time.Now().Add(5 * time.Second))
		if json.NewEncoder(w).Encode(value) != nil {
			return false
		}
		return controller.Flush() == nil
	}
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case snapshot := <-ch:
			s.mesh.Refresh()
			if _, ok := s.mesh.VerifyClientPin(r); !ok {
				return
			}
			if !send(snapshot) {
				return
			}
		case <-ticker.C:
			s.mesh.Refresh()
			if _, ok := s.mesh.VerifyClientPin(r); !ok {
				return
			}
			if !send(nil) {
				return
			}
		}
	}
}

func (m *Manager) remoteSettings(ctx context.Context, msg *Message) {
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

// Settings are pushed directly by each authority. Discovery polling here only
// manages connection lifetimes; configuration itself is never polled from peers.
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
				go func(node string) {
					for streamCtx.Err() == nil {
						current, exists := m.peers.lookup(node)
						if !exists {
							return
						}
						m.consumeSettingsStream(streamCtx, current)
						m.exec.notify("engine:settings-disconnected", map[string]string{"nodeId": node})
						select {
						case <-streamCtx.Done():
							return
						case <-time.After(3 * time.Second):
						}
					}
				}(node)
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

func (m *Manager) consumeSettingsStream(parent context.Context, peer ecPeer) {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	client, err := m.remoteClient(ctx, peer)
	if err != nil {
		return
	}
	route, ok := m.resolveControlRoute(peer.clusterUUID, noderec.ServiceEngineControl)
	config, pinned := m.mesh.ClientTLSConfig(peer.clusterUUID)
	if !ok || !pinned || len(config.Certificates) == 0 {
		return
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, client.base+settingsPath+"events", nil)
	if err != nil {
		return
	}
	response, err := client.controlClient(false).Do(req)
	if err != nil {
		client.forgetAddress()
		return
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return
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
	go func() {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if !current() {
					cancel()
					return
				}
			}
		}
	}()
	watchdog := time.AfterFunc(15*time.Second, cancel)
	defer watchdog.Stop()
	scanner := bufio.NewScanner(response.Body)
	scanner.Buffer(make([]byte, 4096), 256*1024)
	for scanner.Scan() {
		if !current() {
			return
		}
		watchdog.Reset(15 * time.Second)
		var snapshots []settings.Snapshot
		if json.Unmarshal(scanner.Bytes(), &snapshots) != nil {
			return
		}
		m.notifyPeerSettings(peer.nodeID, snapshots)
	}
}
