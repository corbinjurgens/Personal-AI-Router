// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
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
	if method == "inspect" {
		carrier := settings.Request{Engine: request.Engine, NodeID: request.NodeID, Resolution: request.Resolution}
		if !reflect.DeepEqual(request, carrier) {
			return nil, modelSettingsParamsError(fmt.Errorf("inspect: internal carrier must contain only target identity"))
		}
		return r.callInspect(ctx, modelconfig.InspectRequest{Engine: request.Engine, NodeID: request.NodeID, ModelID: request.Resolution}, caller)
	}
	return r.callRelay(ctx, settings.Relay{Method: method, Request: request, Caller: caller})
}

func (r *settingsRelay) callInspect(ctx context.Context, request modelconfig.InspectRequest, caller string) (json.RawMessage, error) {
	if err := request.Validate(); err != nil {
		return nil, modelSettingsParamsError(err)
	}
	return r.callRelay(ctx, settings.Relay{Method: "inspect", Inspect: &request, Caller: caller})
}

func (r *settingsRelay) callRelay(ctx context.Context, relay settings.Relay) (json.RawMessage, error) {
	ctx, cancel := context.WithTimeout(ctx, settings.RelayBudget)
	defer cancel()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	id := newOpID()
	relay.ID = id
	// The public handler and receiving broker own request validation. This
	// transport owns correlation/cancellation even for an ultimately refused
	// request, and bounds the complete envelope before allocating a waiter.
	if err := modelconfig.CheckPayloadSize(relay); err != nil {
		return nil, err
	}
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
	if err := r.send(ctx, "engine:settings-request", relay); err != nil {
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
	case "engine:set-stated-memory", "engine:clear-stated-memory":
		go m.statedMemory(ctx, msg)
		return true
	case "engine:inspect-model":
		go m.inspectModel(ctx, msg)
		return true
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
			switch msg.Method {
			case "engine:get-launch":
				p, err := decodeModelSettingsLaunch(msg.Params)
				if err != nil {
					m.respondOrErr(msg, nil, err)
					return
				}
				result, err := m.exec.LaunchSettings(p.Engine, ctx)
				m.respondOrErr(msg, result, err)
			case "engine:preview-launch":
				envelope, err := settings.DecodeWorkerRequest[settings.Request](msg.Params)
				if err != nil {
					m.respondOrErr(msg, nil, modelSettingsParamsError(err))
					return
				}
				p := envelope.Request
				if err := p.Validate(); err != nil {
					m.respondOrErr(msg, nil, modelSettingsParamsError(err))
					return
				}
				ctx = m.exec.modelMemoryContext(ctx, p.Engine, envelope.Memory, envelope.NativeInitAllowed && p.ModelChange != nil && p.ModelChange.Load != nil)
				if err := m.exec.validateModelSettingsLaunch(ctx, p, false); err != nil {
					m.respondOrErr(msg, nil, err)
					return
				}
				result, err := m.exec.PreviewLaunch(p, ctx)
				if err == nil && len(result.Errors) == 0 && result.Conflict == nil && p.ModelChange == nil && envelope.ValidateResources {
					err = m.exec.validateUnacceptedAutoMemory(ctx, p, envelope.ResourceModel)
				}
				if err == nil {
					if sizeErr := modelconfig.CheckPayloadSize(result); sizeErr != nil {
						err = modelSettingsParamsError(fmt.Errorf("settings: complete preview exceeds 64 KiB; reduce the settings payload"))
					}
				}
				m.respondOrErr(msg, result, err)
			case "engine:configure-launch":
				configureCtx, cancel := context.WithTimeout(ctx, settings.ConfigureBudget)
				defer cancel()
				var configure settings.Configure
				if err := modelconfig.DecodeStrict(msg.Params, &configure); err != nil {
					m.respondOrErr(msg, nil, modelSettingsParamsError(err))
					return
				}
				canonical, err := configure.Settings.Canonical()
				if err != nil {
					m.respondOrErr(msg, nil, modelSettingsParamsError(err))
					return
				}
				configure.Settings = canonical
				if err := m.exec.validateModelSettingsLaunch(configureCtx, settings.Request{Engine: configure.Engine, RequestID: configure.OperationID, Settings: canonical}, true); err != nil {
					m.respondOrErr(msg, nil, err)
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
	case "engine:remote-get-settings", "engine:remote-preview-settings", "engine:remote-apply-settings", "engine:remote-inspect-model", "engine:remote-set-stated-memory", "engine:remote-clear-stated-memory":
		go m.remoteSettings(ctx, msg)
		return true
	}
	return false
}
