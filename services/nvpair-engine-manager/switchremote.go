// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	settings "nvpair-shared/enginesettings"
)

var switchRemoteMethods = map[string]string{
	settings.RemoteSwitchMethod:       settings.SwitchApplyPath,
	settings.RemoteSwitchStatusMethod: settings.SwitchStatusPath,
}

var switchRouteMethods = map[string]string{
	settings.SwitchApplyPath:  settings.SwitchMethod,
	settings.SwitchStatusPath: settings.SwitchStatusMethod,
}

// switchBrokerCall cancels only the relay wait. The broker owns accepted work
// independently of this context. Correlation is supplied by the integration unit.
type switchBrokerCall func(ctx context.Context, method, caller string, params json.RawMessage) (json.RawMessage, error)

// normalizeSwitchParams strictly validates before replacing the transport alias.
// The original node ID is returned solely for outbound peer resolution.
func normalizeSwitchParams(path string, data []byte, target string) (json.RawMessage, string, error) {
	var body any
	var nodeID string
	switch path {
	case settings.SwitchApplyPath:
		p, err := settings.DecodeSwitchRequest(data)
		if err != nil {
			return nil, "", &codedError{code: settings.SwitchInvalidParamsCode, message: err.Error()}
		}
		nodeID, p.NodeID = p.NodeID, target
		body = p
	case settings.SwitchStatusPath:
		p, err := settings.DecodeSwitchStatusRequest(data)
		if err != nil {
			return nil, "", &codedError{code: settings.SwitchInvalidParamsCode, message: err.Error()}
		}
		nodeID, p.NodeID = p.NodeID, target
		body = p
	default:
		return nil, "", &codedError{code: -32601, message: "unknown engine switch path"}
	}
	params, err := json.Marshal(body)
	return params, nodeID, err
}

// callRemoteSwitch uses the existing discovery directory and pinned ec client
// factory. It never retries through start/stop or forwards to a different peer.
func callRemoteSwitch(ctx context.Context, method string, params json.RawMessage,
	lookup func(string) (ecPeer, bool), clientFor func(context.Context, ecPeer) (*remoteClient, error),
) (json.RawMessage, error) {
	path, ok := switchRemoteMethods[method]
	if !ok {
		return nil, &codedError{code: -32601, message: "unknown remote engine switch method"}
	}
	_, nodeID, err := normalizeSwitchParams(path, params, "")
	if err != nil {
		return nil, err
	}
	if nodeID == "" {
		return nil, &codedError{code: settings.SwitchInvalidParamsCode, message: "nodeId is required"}
	}
	peer, ok := lookup(nodeID)
	if !ok {
		return nil, fmt.Errorf("engine switch unavailable: device is not a discovered ec peer")
	}
	client, err := clientFor(ctx, peer)
	if err != nil {
		return nil, err
	}
	params, _, err = normalizeSwitchParams(path, params, peer.nodeID)
	if err != nil {
		return nil, err
	}
	return client.postSwitch(ctx, path, params)
}

// remoteSwitch dispatches once; subsequent progress uses only status reads.
func (m *Manager) remoteSwitch(ctx context.Context, msg *Message) {
	if m.mesh == nil {
		m.respondOrErr(msg, nil, fmt.Errorf("engine switch unavailable: node is not clustered"))
		return
	}
	var identity switchWatchIdentity
	result, err := callRemoteSwitch(ctx, msg.Method, msg.Params, m.peers.lookup,
		func(ctx context.Context, peer ecPeer) (*remoteClient, error) {
			client, err := m.remoteClient(ctx, peer)
			if err == nil {
				identity, _ = m.switchWatchIdentity(peer.nodeID)
			}
			return client, err
		})
	if err == nil {
		m.switchWatchers.observe(ctx, identity, result, m.switchWatchIdentity, m.remoteClient, m.codec.Notify)
	}
	m.respondOrErr(msg, result, err)
}

func (c *remoteClient) postSwitch(ctx context.Context, path string, params json.RawMessage) (json.RawMessage, error) {
	if _, ok := switchRouteMethods[path]; !ok {
		return nil, &codedError{code: -32601, message: "unknown engine switch path"}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+path, bytes.NewReader(params))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	// Status may wait 25 seconds; the existing readiness client also leaves
	// headroom for relay overhead. The caller bounds its own wait with ctx.
	response, err := c.controlClient(true).Do(req)
	if err != nil {
		c.forgetAddress()
		return nil, err
	}
	defer response.Body.Close()
	data, err := readRemoteResponse(response.Body, settings.MaxSwitchJournalBytes)
	if err != nil {
		return nil, fmt.Errorf("remote engine switch: read response; operation outcome may be uncertain: %w", err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		var body codedBody
		if response.StatusCode == http.StatusConflict && json.Unmarshal(data, &body) == nil && body.Code == settings.SwitchErrorConfigurationConflict {
			return nil, &codedError{code: settings.SwitchConfigurationConflictCode, message: body.Error}
		}
		if response.StatusCode == http.StatusBadRequest {
			return nil, &codedError{code: settings.SwitchInvalidParamsCode, message: string(data)}
		}
		return nil, remoteStatusError("remote engine switch", response.StatusCode, data)
	}
	if !json.Valid(data) {
		return nil, fmt.Errorf("remote engine switch: invalid JSON response; operation outcome may be uncertain")
	}
	return json.RawMessage(data), nil
}

// switchHandler is unregistered. Integration supplies s.requirePinCaller and
// this target's stable node ID in switchRoutes; tests inject the trusted gate.
func switchHandler(target string, requirePinCaller func(pinnedPeerHandler) http.HandlerFunc, relay switchBrokerCall) http.HandlerFunc {
	return requirePinCaller(func(w http.ResponseWriter, r *http.Request, caller string) {
		method, ok := switchRouteMethods[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, settings.MaxSwitchRequestBytes))
		if err != nil {
			http.Error(w, "invalid engine switch request", http.StatusBadRequest)
			return
		}
		params, _, err := normalizeSwitchParams(r.URL.Path, data, target)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if target == "" || relay == nil {
			http.Error(w, "engine switch coordinator unavailable", http.StatusServiceUnavailable)
			return
		}
		// This is always a local broker method, with the target's own node ID.
		// Cancellation abandons waiting, never sends an executor cancellation.
		result, err := relay(r.Context(), method, caller, params)
		if err != nil {
			code := rpcCode(err)
			if code == settings.SwitchInvalidParamsCode {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if code == settings.SwitchConfigurationConflictCode {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusConflict)
				_ = json.NewEncoder(w).Encode(codedBody{Error: err.Error(), Code: settings.SwitchErrorConfigurationConflict})
				return
			}
			if code == settings.SwitchRemoteVetoCode {
				writeCodedError(w, code, err.Error())
				return
			}
			if code == codeRemoteManagementDenied {
				// The target refused a management operation; the origin maps
				// this back to -32012, distinct from the local veto above.
				writeCodedError(w, code, err.Error())
				return
			}
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(result)
	})
}
