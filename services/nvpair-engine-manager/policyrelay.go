// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// policyrelay.go carries a paired node's node-policy and availability calls
// (FORK_DESIGN.md §3.1) over the pinned-mTLS ec surface, the same way
// settingsremote.go carries engine settings:
//
//   - Outbound, engine:remote-policy-get / engine:remote-policy-set /
//     engine:remote-availability-set POST to the target's ec routes.
//   - Inbound, the routes relay to this node's broker as a correlated
//     policy:request notification and wait for its policy:reply. The broker
//     owns the policy; engine-manager only transports it.
//
// A relayed request always has its nodeId cleared, so the target's broker acts
// on itself and can never forward a further hop.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

const (
	controlPolicyGetPath       = "/v1/node-policy/get"
	controlPolicySetPath       = "/v1/node-policy/set"
	controlAvailabilitySetPath = "/v1/node-availability/set"

	policyRequestMethod = "policy:request"
	policyReplyMethod   = "policy:reply"
	policyCancelMethod  = "policy:cancel"

	// policyBodyLimit caps a relayed policy body. A policy carries profiles,
	// tiers and per-model request options, so it is allowed more room than an
	// engine settings request, but it is still a small control message.
	policyBodyLimit = 256 << 10
	// policyRelayBudget bounds how long the ec route waits for the broker. A
	// pause waits for the node to drain, unload and stop engines, so it is
	// generous; the broker completes an accepted pause on its own regardless.
	policyRelayBudget = 15 * time.Minute
	// maxPendingPolicy bounds concurrent relayed requests.
	maxPendingPolicy = 32
)

// policyRoutePaths maps each ec route to the broker relay method.
var policyRoutePaths = map[string]string{
	controlPolicyGetPath:       "get",
	controlPolicySetPath:       "set",
	controlAvailabilitySetPath: "set-availability",
}

// policyRemoteMethods maps each client method to the target's ec route.
var policyRemoteMethods = map[string]string{
	"engine:remote-policy-get":       controlPolicyGetPath,
	"engine:remote-policy-set":       controlPolicySetPath,
	"engine:remote-availability-set": controlAvailabilitySetPath,
}

// policyBody is the one body shape every policy route accepts. Unknown fields
// are refused, so a peer cannot smuggle anything the broker would act on.
type policyBody struct {
	NodeID string          `json:"nodeId,omitempty"`
	Policy json.RawMessage `json:"policy,omitempty"`
	State  string          `json:"state,omitempty"`
}

// policyRelayRequest is the policy:request payload sent to the broker.
type policyRelayRequest struct {
	ID     string          `json:"id"`
	Method string          `json:"method"`
	Caller string          `json:"caller,omitempty"`
	Params json.RawMessage `json:"params"`
}

type policyRelayReply struct {
	ID     string          `json:"id"`
	Result json.RawMessage `json:"result"`
	Error  string          `json:"error,omitempty"`
}

// policyRelay correlates policy:request notifications with policy:reply ones.
type policyRelay struct {
	mu      sync.Mutex
	pending map[string]chan policyRelayReply
	send    func(string, any) error
}

func (r *policyRelay) call(ctx context.Context, method string, params json.RawMessage, caller string) (json.RawMessage, error) {
	if r.send == nil {
		return nil, fmt.Errorf("policy coordinator unavailable")
	}
	ctx, cancel := context.WithTimeout(ctx, policyRelayBudget)
	defer cancel()
	id := newOpID()
	ch := make(chan policyRelayReply, 1)
	r.mu.Lock()
	if len(r.pending) >= maxPendingPolicy {
		r.mu.Unlock()
		return nil, fmt.Errorf("too many policy requests")
	}
	if r.pending == nil {
		r.pending = make(map[string]chan policyRelayReply)
	}
	r.pending[id] = ch
	r.mu.Unlock()
	defer func() { r.mu.Lock(); delete(r.pending, id); r.mu.Unlock() }()
	if err := r.send(policyRequestMethod, policyRelayRequest{ID: id, Method: method, Caller: caller, Params: params}); err != nil {
		return nil, err
	}
	select {
	case reply := <-ch:
		if reply.Error != "" {
			return nil, fmt.Errorf("%s", reply.Error)
		}
		return reply.Result, nil
	case <-ctx.Done():
		_ = r.send(policyCancelMethod, map[string]string{"id": id})
		return nil, ctx.Err()
	}
}

func (r *policyRelay) reply(raw json.RawMessage) {
	var reply policyRelayReply
	if json.Unmarshal(raw, &reply) != nil {
		return
	}
	r.mu.Lock()
	ch := r.pending[reply.ID]
	r.mu.Unlock()
	if ch != nil {
		select {
		case ch <- reply:
		default:
		}
	}
}

// handlePolicyMessage serves the node-policy surface: the broker's
// policy:reply, the remote policy client methods, and the intent-neutral
// lifecycle methods in wakesleep.go. It reports whether it took the message.
func (m *Manager) handlePolicyMessage(ctx context.Context, msg *Message) bool {
	if msg.Method == policyReplyMethod && msg.IsNotification() {
		m.policyRelay.reply(msg.Params)
		return true
	}
	if !msg.IsRequest() {
		return false
	}
	if path, ok := policyRemoteMethods[msg.Method]; ok {
		go m.remotePolicy(ctx, msg, path)
		return true
	}
	return m.handleLifecyclePolicyRequest(ctx, msg)
}

// remotePolicy forwards one client policy call to the named peer's ec route.
func (m *Manager) remotePolicy(ctx context.Context, msg *Message, path string) {
	var p policyBody
	if !m.parse(msg, &p) {
		return
	}
	if p.NodeID == "" {
		m.codec.RespondError(msg.ID, -32602, "nodeId is required")
		return
	}
	peer, ok := m.peers.lookup(p.NodeID)
	if !ok || m.mesh == nil {
		m.respondOrErr(msg, nil, fmt.Errorf("node policy unavailable: device is offline or does not support it"))
		return
	}
	client, err := m.remoteClient(ctx, peer)
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	p.NodeID = ""
	result, err := client.postPolicy(ctx, path, p)
	m.respondOrErr(msg, result, err)
}

// postPolicy POSTs a policy body. Setting availability waits for the target to
// drain, so it uses the long response-header budget the readiness routes use.
func (c *remoteClient) postPolicy(ctx context.Context, path string, body policyBody) (json.RawMessage, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+path, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	client := c.http
	if path == controlAvailabilitySetPath && c.readyHTTP != nil {
		client = c.readyHTTP
	}
	resp, err := client.Do(req)
	if err != nil {
		c.forgetAddress()
		return nil, err
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(io.LimitReader(resp.Body, policyBodyLimit))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("remote node policy: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(out)))
	}
	return json.RawMessage(out), nil
}

// policyRoutes registers the inbound ec policy routes. Each is pin-gated with
// the caller's identity, body-capped, and stripped of nodeId before it reaches
// the broker.
func (s *controlServer) policyRoutes(mux *http.ServeMux) {
	for path, method := range policyRoutePaths {
		mux.HandleFunc(path, s.requirePinCaller(func(w http.ResponseWriter, r *http.Request, caller string) {
			if r.Method != http.MethodPost {
				w.Header().Set("Allow", "POST")
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}
			if s.exec.policyParent == nil {
				http.Error(w, "policy coordinator unavailable", http.StatusServiceUnavailable)
				return
			}
			var p policyBody
			decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, policyBodyLimit))
			decoder.DisallowUnknownFields()
			if decoder.Decode(&p) != nil || decoder.Decode(&struct{}{}) != io.EOF {
				http.Error(w, "invalid policy request", http.StatusBadRequest)
				return
			}
			// A peer can only address this node, never forward another hop.
			p.NodeID = ""
			params, err := json.Marshal(p)
			if err != nil {
				http.Error(w, "invalid policy request", http.StatusBadRequest)
				return
			}
			ctx, cancel := context.WithCancel(r.Context())
			defer cancel()
			// Abandon the wait as soon as the caller is no longer pinned. An
			// accepted availability change still completes on the target.
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
			result, err := s.exec.policyParent(ctx, method, params, caller)
			if err != nil {
				http.Error(w, err.Error(), http.StatusConflict)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(result)
		}))
	}
}
