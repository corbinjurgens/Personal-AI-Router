// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	settings "nvpair-shared/enginesettings"
)

// settingsReply is the broker's engine:settings-reply. Like policy:reply, a
// failure carries its JSON-RPC errorCode, so a local veto (codeLocalVeto)
// reaches a paired caller typed.
type settingsReply struct {
	ID        string          `json:"id"`
	Result    json.RawMessage `json:"result"`
	Error     string          `json:"error,omitempty"`
	ErrorCode int             `json:"errorCode,omitempty"`
}
type settingsRelay struct {
	mu      sync.Mutex
	pending map[string]chan settingsReply
	send    func(context.Context, string, any) error
}

func (r *settingsRelay) call(ctx context.Context, method string, request settings.Request, caller string) (json.RawMessage, error) {
	ctx, cancel := context.WithTimeout(ctx, settings.RelayBudget)
	defer cancel()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	id := newOpID()
	ch := make(chan settingsReply, 1)
	r.mu.Lock()
	if err := ctx.Err(); err != nil {
		r.mu.Unlock()
		return nil, err
	}
	if len(r.pending) >= 64 {
		r.mu.Unlock()
		return nil, fmt.Errorf("too many settings requests")
	}
	if r.pending == nil {
		r.pending = make(map[string]chan settingsReply)
	}
	r.pending[id] = ch
	r.mu.Unlock()
	defer func() { r.mu.Lock(); delete(r.pending, id); r.mu.Unlock() }()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := r.send(ctx, "engine:settings-request", settings.Relay{ID: id, Method: method, Request: request, Caller: caller}); err != nil {
		return nil, err
	}
	var reply settingsReply
	select {
	case reply = <-ch:
	case <-ctx.Done():
	}
	if err := ctx.Err(); err != nil {
		r.mu.Lock()
		delete(r.pending, id)
		r.mu.Unlock()
		cancelRelayWait(r.send, "engine:settings-cancel", id)
		return nil, err
	}
	if reply.Error != "" {
		return nil, &codedError{code: reply.ErrorCode, message: reply.Error}
	}
	return reply.Result, nil
}

func (r *settingsRelay) reply(raw json.RawMessage) {
	var reply settingsReply
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

func (m *Manager) handleSettingsMessage(ctx context.Context, msg *Message) bool {
	if msg.Method == "engine:settings-reply" && msg.IsNotification() {
		m.settingsRelay.reply(msg.Params)
		return true
	}
	if msg.Method == "engine:settings-projection" && msg.IsNotification() {
		var snapshots []settings.Snapshot
		if json.Unmarshal(msg.Params, &snapshots) == nil {
			m.exec.settingsHub.Publish(snapshots)
		}
		return true
	}
	if !msg.IsRequest() {
		return false
	}
	switch msg.Method {
	case "engine:configured-ports":
		go func() {
			type port struct {
				Engine string `json:"engine"`
				Port   int    `json:"port"`
			}
			ports := []port{}
			for _, name := range m.exec.reg.Names() {
				if launch, err := m.exec.LaunchSettings(name, ctx); err == nil {
					ports = append(ports, port{name, launch.ServerPort})
				}
			}
			m.respondOrErr(msg, map[string]any{"engines": ports}, nil)
		}()
		return true
	case "engine:get-launch", "engine:preview-launch", "engine:configure-launch":
		go func() {
			var p settings.Request
			if !m.parse(msg, &p) {
				return
			}
			switch msg.Method {
			case "engine:get-launch":
				result, err := m.exec.LaunchSettings(p.Engine, ctx)
				m.respondOrErr(msg, result, err)
			case "engine:preview-launch":
				result, err := m.exec.PreviewLaunch(p, ctx)
				m.respondOrErr(msg, result, err)
			case "engine:configure-launch":
				configureCtx, cancel := context.WithTimeout(ctx, settings.ConfigureBudget)
				defer cancel()
				var configure settings.Configure
				if !m.parse(msg, &configure) {
					return
				}
				result, err := m.exec.ConfigureLaunch(configureCtx, configure, func() error {
					_, err := m.settingsRelay.call(configureCtx, "rebind", settings.Request{Engine: configure.Engine, RequestID: configure.OperationID, Settings: configure.Settings}, "")
					return err
				})
				m.respondOrErr(msg, result, err)
			}
		}()
		return true
	case "engine:remote-get-settings", "engine:remote-preview-settings", "engine:remote-apply-settings":
		go m.remoteSettings(ctx, msg)
		return true
	}
	return false
}
