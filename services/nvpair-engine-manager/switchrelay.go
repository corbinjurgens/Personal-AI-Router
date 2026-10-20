// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"time"

	settings "nvpair-shared/enginesettings"
)

const maxPendingSwitchCalls = 32

type switchRelayRequest struct {
	ID     string          `json:"id"`
	Method string          `json:"method"`
	Caller string          `json:"caller"`
	Params json.RawMessage `json:"params"`
}

type switchRelayReply struct {
	ID        string          `json:"id"`
	Result    json.RawMessage `json:"result"`
	Error     string          `json:"error,omitempty"`
	ErrorCode int             `json:"errorCode,omitempty"`
}

type switchRelay struct {
	mu      sync.Mutex
	pending map[string]chan switchRelayReply
	send    func(context.Context, string, any) error
}

func (r *switchRelay) call(ctx context.Context, method, caller string, params json.RawMessage) (json.RawMessage, error) {
	if r.send == nil {
		return nil, errors.New("switch coordinator unavailable")
	}
	if method != settings.SwitchMethod && method != settings.SwitchStatusMethod {
		return nil, &codedError{code: -32601, message: "unknown switch relay method"}
	}
	ctx, cancel := context.WithTimeout(ctx, settings.SwitchStatusWaitBudget+10*time.Second)
	defer cancel()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	id := newOpID()
	ch := make(chan switchRelayReply, 1)
	r.mu.Lock()
	if err := ctx.Err(); err != nil {
		r.mu.Unlock()
		return nil, err
	}
	if len(r.pending) >= maxPendingSwitchCalls {
		r.mu.Unlock()
		return nil, &switchBusyError{"switch relay"}
	}
	if r.pending == nil {
		r.pending = make(map[string]chan switchRelayReply)
	}
	r.pending[id] = ch
	r.mu.Unlock()
	defer func() { r.mu.Lock(); delete(r.pending, id); r.mu.Unlock() }()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := r.send(ctx, "engine:switch-request", switchRelayRequest{ID: id, Method: method, Caller: caller, Params: params}); err != nil {
		return nil, err
	}
	var reply switchRelayReply
	select {
	case reply = <-ch:
	case <-ctx.Done():
	}
	if err := ctx.Err(); err != nil {
		// Only retires the waiter. Accepted operations belong to the broker.
		r.mu.Lock()
		delete(r.pending, id)
		r.mu.Unlock()
		cancelRelayWait(r.send, "engine:switch-cancel", id)
		return nil, err
	}
	if reply.Error != "" {
		return nil, &codedError{code: reply.ErrorCode, message: reply.Error}
	}
	if !json.Valid(reply.Result) || len(reply.Result) > settings.MaxSwitchJournalBytes {
		return nil, errors.New("invalid switch coordinator reply")
	}
	return reply.Result, nil
}

func (r *switchRelay) reply(raw json.RawMessage) {
	if len(raw) > settings.MaxSwitchJournalBytes+4096 {
		return
	}
	var reply switchRelayReply
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

func (s *controlServer) switchRoutes(mux *http.ServeMux) {
	// Read the live stable identity per request, including after joining a mesh.
	for path := range switchRouteMethods {
		mux.HandleFunc(path, func(w http.ResponseWriter, r *http.Request) {
			target := ""
			if s.mesh != nil {
				target = s.mesh.NodeUUID()
			}
			switchHandler(target, s.requirePinCaller, s.exec.switchParent)(w, r)
		})
	}
}

func (m *Manager) handleSwitchMessage(ctx context.Context, msg *Message) bool {
	if msg.Method == "engine:switch-reply" && msg.IsNotification() {
		m.switchRelay.reply(msg.Params)
		return true
	}
	if !msg.IsRequest() {
		return false
	}
	switch msg.Method {
	case settings.RemoteSwitchMethod, settings.RemoteSwitchStatusMethod:
		go m.remoteSwitch(ctx, msg)
		return true
	case settings.SwitchPrepareMethod, settings.SwitchStepMethod, settings.SwitchReconcileMethod:
		// This dispatch is only reachable on the parent stdio/IPC channel. No
		// ec route exposes the internal executor methods.
		go func() {
			switch msg.Method {
			case settings.SwitchPrepareMethod:
				var p settings.SwitchPrepareRequest
				if m.parse(msg, &p) {
					m.respondOrErr(msg, m.exec.SwitchPrepare(ctx, p), nil)
				}
			case settings.SwitchStepMethod:
				var p settings.SwitchStepRequest
				if m.parse(msg, &p) {
					m.respondOrErr(msg, m.exec.SwitchStep(ctx, p), nil)
				}
			case settings.SwitchReconcileMethod:
				var p settings.SwitchReconcileRequest
				if m.parse(msg, &p) {
					m.respondOrErr(msg, m.exec.SwitchReconcile(ctx, p), nil)
				}
			}
		}()
		return true
	}
	return false
}
