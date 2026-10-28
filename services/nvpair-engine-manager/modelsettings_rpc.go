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
	// Artifact inspection and advisory estimation share the operation budget.
	ctx, cancel := context.WithTimeout(ctx, modelInspectionBudget)
	defer cancel()
	if err := p.Validate(); err != nil {
		return modelconfig.InspectResult{}, modelSettingsParamsError(err)
	}
	if p.NodeID != "" {
		return modelconfig.InspectResult{}, modelSettingsParamsError(fmt.Errorf("nodeId: worker inspection must address the local target"))
	}
	if err := ctx.Err(); err != nil {
		return modelconfig.InspectResult{}, err
	}
	var advisory modelEstimateInput
	result, err := e.inspectModelArtifacts(ctx, p, &advisory)
	if err != nil {
		return modelconfig.InspectResult{}, err
	}
	if p.Engine == "llamacpp" {
		result.Options, err = modelInspectionOptions(result.Options)
		if err != nil {
			return modelconfig.InspectResult{}, err
		}
	}
	result.Estimate = e.modelEstimates.estimate(ctx, advisory)
	// The final public reply keeps its existing ceiling. An advisory result
	// cannot turn a successful artifact inspection into a payload-size failure.
	if err := modelconfig.CheckPayloadSize(result); err != nil {
		result.Estimate = modelEstimateUnknown("estimate omitted: complete inspection reply size limit")
	}
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

// The worker accepts frozen native settings from its parent. Coupled policy
// acceptance belongs to the broker. Apply must never invoke an Auto resolver.
func (e *Executor) validateModelSettingsLaunch(ctx context.Context, p settings.Request, applying bool) error {
	if err := p.Validate(); err != nil {
		return modelSettingsParamsError(err)
	}
	if p.ModelChange != nil {
		if applying && p.ModelChange.PreviewDigest == "" {
			return modelSettingsParamsError(fmt.Errorf("modelChange.previewDigest: reviewed preview required"))
		}
		if !applying && p.ModelChange.Load != nil {
			r := p.ModelChange.Load.Set
			// The owner-aware compiler resolves Auto after target baseline and
			// artifact validation. This guard neither resolves nor reads capacity.
			// Keep the exact production refusal when no target owner exists.
			op, _ := ctx.Value(modelAutoOperationKey{}).(modelAutoOperation)
			if modelLoadHasAuto(r) && op.Memory == nil {
				return modelSettingsParamsError(fmt.Errorf("target memory telemetry unavailable"))
			}
		}
		if !applying && p.ModelChange.Load != nil && p.ModelChange.Policy == nil {
			return ctx.Err()
		}
		return modelSettingsParamsError(fmt.Errorf("modelChange: reviewed model settings application is not ready"))
	}
	if applying {
		for _, load := range p.Settings.ModelLoads {
			if load.Resolved == nil {
				return modelSettingsParamsError(fmt.Errorf("settings.modelLoads: reviewed finite resolution required"))
			}
			if modelLoadHasAuto(load.Requested) {
				if err := modelAutoFrozen(load.ModelID, load.Requested, load.Resolved); err != nil {
					return err
				}
			}
		}
	}
	return ctx.Err()
}
