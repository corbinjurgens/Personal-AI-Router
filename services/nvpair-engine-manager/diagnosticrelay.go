// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"time"

	"nvpair-shared/modelconfig"
)

// Only the parent channel supplies replies. Caller is established by the ec
// pin gate, never by a public request body. Cancellation retires the relay
// waiter; accepted diagnostics and their cleanup belong to the target broker.
type diagnosticRelayRequest struct {
	ID     string          `json:"id"`
	Method string          `json:"method"`
	Caller string          `json:"caller"`
	Params json.RawMessage `json:"params"`
}

type diagnosticRelayReply struct {
	ID        string          `json:"id"`
	Result    json.RawMessage `json:"result"`
	Error     string          `json:"error,omitempty"`
	ErrorCode int             `json:"errorCode,omitempty"`
}

type diagnosticRelay struct {
	mu      sync.Mutex
	pending map[string]chan diagnosticRelayReply
	send    func(context.Context, string, any) error
}

// Cancel nests the broker's 200-second cleanup join inside the caller's
// 210-second envelope. Preview/start/status never wait for the native exercise.
func diagnosticRelayBudget(method string) time.Duration {
	if method == "engine:diagnostic-cancel" {
		return 205 * time.Second
	}
	return 25 * time.Second
}

func (r *diagnosticRelay) call(ctx context.Context, method, caller string, params json.RawMessage) (json.RawMessage, error) {
	if r.send == nil {
		return nil, errors.New("diagnostic coordinator unavailable")
	}
	if _, ok := diagnosticLocalPaths[method]; !ok || caller == "" {
		return nil, &codedError{code: -32602, message: "invalid diagnostic relay authority"}
	}
	ctx, cancel := context.WithTimeout(ctx, diagnosticRelayBudget(method))
	defer cancel()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	id := newOpID()
	request := diagnosticRelayRequest{ID: id, Method: method, Caller: caller, Params: params}
	if err := modelconfig.CheckPayloadSize(request); err != nil {
		return nil, err
	}
	ch := make(chan diagnosticRelayReply, 1)
	r.mu.Lock()
	if len(r.pending) >= 32 {
		r.mu.Unlock()
		return nil, errors.New("too many diagnostic requests")
	}
	if r.pending == nil {
		r.pending = make(map[string]chan diagnosticRelayReply)
	}
	r.pending[id] = ch
	r.mu.Unlock()
	defer func() { r.mu.Lock(); delete(r.pending, id); r.mu.Unlock() }()
	if err := r.send(ctx, "engine:diagnostic-request", request); err != nil {
		return nil, err
	}
	var reply diagnosticRelayReply
	select {
	case reply = <-ch:
	case <-ctx.Done():
	}
	if err := ctx.Err(); err != nil {
		r.mu.Lock()
		delete(r.pending, id)
		r.mu.Unlock()
		cancelRelayWait(r.send, "engine:diagnostic-request-cancel", id)
		return nil, err
	}
	if reply.Error != "" {
		return nil, &codedError{code: reply.ErrorCode, message: reply.Error}
	}
	if !json.Valid(reply.Result) || len(reply.Result) > modelconfig.MaxPayloadBytes {
		return nil, errors.New("invalid diagnostic coordinator response")
	}
	return reply.Result, nil
}

func (r *diagnosticRelay) reply(raw json.RawMessage) {
	if len(raw) > modelconfig.MaxPayloadBytes+4096 {
		return
	}
	var reply diagnosticRelayReply
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

func (m *Manager) handleDiagnosticRelayMessage(ctx context.Context, msg *Message) bool {
	if msg.Method == "engine:diagnostic-reply" && msg.IsNotification() {
		m.diagnosticRelay.reply(msg.Params)
		return true
	}
	if !msg.IsRequest() {
		return false
	}
	switch msg.Method {
	case "engine:remote-diagnostic-preview", "engine:remote-diagnostic-start", "engine:remote-diagnostic-status", "engine:remote-diagnostic-cancel":
		go m.remoteDiagnostic(ctx, msg)
		return true
	}
	return false
}
