// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
)

func modelSettingsParamsError(err error) error {
	return &codedError{code: -32602, message: err.Error()}
}

// InspectModel is a target-local read. The broker supplies authenticated node
// identity after this reply; a worker never echoes a caller's target as proof.
// Artifact inspection reads bounded local metadata and samples without native,
// network, lifecycle, qualification or settings effects.
func (e *Executor) InspectModel(ctx context.Context, p modelconfig.InspectRequest) (modelconfig.InspectResult, error) {
	if err := p.Validate(); err != nil {
		return modelconfig.InspectResult{}, modelSettingsParamsError(err)
	}
	if p.NodeID != "" {
		return modelconfig.InspectResult{}, modelSettingsParamsError(fmt.Errorf("nodeId: worker inspection must address the local target"))
	}
	if err := ctx.Err(); err != nil {
		return modelconfig.InspectResult{}, err
	}
	result, err := e.inspectModelArtifacts(ctx, p)
	if err != nil {
		return modelconfig.InspectResult{}, err
	}
	if p.Engine == "llamacpp" {
		result.Options, err = modelInspectionOptions(result.Options)
		if err != nil {
			return modelconfig.InspectResult{}, err
		}
	}
	estimate, err := estimateModelSettings(ctx, p, modelconfig.RequestedLoad{})
	if err != nil {
		return modelconfig.InspectResult{}, err
	}
	result.Estimate = estimate
	if err := result.Validate(); err != nil {
		return modelconfig.InspectResult{}, err
	}
	return result, nil
}

func (m *Manager) inspectModel(ctx context.Context, msg *Message) {
	p, err := modelconfig.DecodeInspectRequest(msg.Params)
	if err != nil {
		m.respondOrErr(msg, nil, modelSettingsParamsError(err))
		return
	}
	result, err := m.exec.InspectModel(ctx, p)
	m.respondOrErr(msg, result, err)
}

func (m *Manager) remoteInspectModel(ctx context.Context, msg *Message) {
	p, err := modelconfig.DecodeInspectRequest(msg.Params)
	if err != nil {
		m.respondOrErr(msg, nil, modelSettingsParamsError(err))
		return
	}
	if p.NodeID == "" {
		m.respondOrErr(msg, nil, modelSettingsParamsError(fmt.Errorf("nodeId: remote target required")))
		return
	}
	peer, ok := m.peers.lookup(p.NodeID)
	if !ok || m.mesh == nil {
		m.respondOrErr(msg, nil, fmt.Errorf("model inspection unavailable: device is offline"))
		return
	}
	client, err := m.remoteClient(ctx, peer)
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	target := p.NodeID
	p.NodeID = ""
	raw, err := client.postJSON(ctx, settingsPath+"inspect", p.Engine, p)
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	result, err := modelconfig.DecodeInspectResult(raw)
	if err == nil && (result.Engine != p.Engine || result.ModelID != p.ModelID || result.NodeID != target) {
		err = fmt.Errorf("model inspection: target reply identity mismatch")
	}
	m.respondOrErr(msg, result, err)
}

// Parent launch traffic permits the broker-owned PreserveCORS guard. Public
// settings.DecodeRequest deliberately rejects it, so it is not this decoder.
func decodeModelSettingsLaunch(raw []byte) (settings.Request, error) {
	if err := validateServingModelsJSON(raw, "settings", "serving"); err != nil {
		return settings.Request{}, modelSettingsParamsError(err)
	}
	var p settings.Request
	if err := modelconfig.DecodeStrict(raw, &p); err != nil {
		return settings.Request{}, modelSettingsParamsError(err)
	}
	if err := p.Validate(); err != nil {
		return settings.Request{}, modelSettingsParamsError(err)
	}
	c, err := p.Settings.Canonical()
	if err != nil {
		return settings.Request{}, modelSettingsParamsError(err)
	}
	p.Settings = c
	return p, nil
}

// This guard precedes even Strata preparation, mutation admission and rebind.
// Direct Executor callers need the same guard at settings.go (integration owner).
func (e *Executor) validateModelSettingsLaunch(ctx context.Context, p settings.Request, applying bool) error {
	if err := p.Validate(); err != nil {
		return modelSettingsParamsError(err)
	}
	if p.ModelChange != nil {
		if applying && p.ModelChange.PreviewDigest == "" {
			return modelSettingsParamsError(fmt.Errorf("modelChange.previewDigest: reviewed preview required"))
		}
		if p.ModelChange.Load != nil {
			r := p.ModelChange.Load.Set
			if r.Context != nil && r.Context.Mode == "auto" || r.GPU != nil && r.GPU.Mode == "auto" {
				if _, err := resolveModelSettingsAuto(ctx, modelconfig.InspectRequest{Engine: p.Engine, ModelID: p.ModelChange.ModelID}, r); err != nil {
					return err
				}
			}
		}
		if !applying && p.ModelChange.Load != nil && p.ModelChange.Policy == nil {
			return ctx.Err()
		}
		return modelSettingsParamsError(fmt.Errorf("modelChange: reviewed model settings application is not ready"))
	}
	if len(p.Settings.ModelLoads) != 0 {
		return modelSettingsParamsError(fmt.Errorf("settings.modelLoads: typed load application is not ready"))
	}
	// An ordinary empty payload must not erase stored typed authority while the
	// direct launch/persistence owner still lacks modelLoads preservation.
	st, err := e.state(p.Engine)
	if err != nil {
		return err
	}
	st.mu.Lock()
	hasLoads := len(st.plat.Runtime.ModelLoads) != 0
	st.mu.Unlock()
	if hasLoads {
		return modelSettingsParamsError(fmt.Errorf("settings.modelLoads: stored loads require the preserving launch implementation"))
	}
	return ctx.Err()
}
