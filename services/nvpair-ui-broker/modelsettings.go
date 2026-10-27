// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
)

type settingsWorkerError struct {
	code    int
	message string
}

func (e *settingsWorkerError) Error() string { return e.message }

func engineSettingsErrorCode(err error) int {
	var workerError *settingsWorkerError
	if errors.As(err, &workerError) && workerError.code != 0 {
		return workerError.code
	}
	return policyErrorCode(err)
}

// validateModelSettingsEntry runs before journal initialization or worker IO.
// The coupled/load implementation replaces this body at its existing owner.
// No maintenance reservation exists yet: apply-build must include this mutation
// boundary in its exclusion when that owner lands.
func validateModelSettingsEntry(p settings.Request) error {
	if err := p.Validate(); err != nil {
		return &invalidParamsError{err: err}
	}
	if p.ModelChange != nil {
		return &invalidParamsError{err: fmt.Errorf("modelChange: reviewed model settings application is not ready")}
	}
	if len(p.Settings.ModelLoads) != 0 {
		return &invalidParamsError{err: fmt.Errorf("settings.modelLoads: typed load application is not ready")}
	}
	return nil
}

// inspectModelSettings is a read, never a settings snapshot initialization or
// apply. Authorization is fresh at the existing target caller boundary.
func (b *Broker) inspectModelSettings(ctx context.Context, p modelconfig.InspectRequest, caller string) (modelconfig.InspectResult, error) {
	if err := p.Validate(); err != nil {
		return modelconfig.InspectResult{}, &invalidParamsError{err: err}
	}
	ctx, cancel := context.WithTimeout(ctx, settings.CallBudget)
	defer cancel()
	b.engineConfigMu.Lock()
	err := b.authorizeSettingsCallerLocked(caller)
	b.engineConfigMu.Unlock()
	if err != nil {
		return modelconfig.InspectResult{}, err
	}
	p.NodeID = ""
	var raw json.RawMessage
	if err := b.settingsWorkerCall(ctx, "engine:inspect-model", p, &raw); err != nil {
		return modelconfig.InspectResult{}, err
	}
	result, err := modelconfig.DecodeInspectResult(raw)
	if err != nil {
		return modelconfig.InspectResult{}, err
	}
	if result.Engine != p.Engine || result.ModelID != p.ModelID {
		return modelconfig.InspectResult{}, fmt.Errorf("model inspection: worker reply identity mismatch")
	}
	result.NodeID = b.nodeID
	if err := result.Validate(); err != nil {
		return modelconfig.InspectResult{}, err
	}
	return result, nil
}

func (b *Broker) handleModelInspection(msg *Message) {
	p, err := modelconfig.DecodeInspectRequest(msg.Params)
	if err != nil {
		_ = b.codec.RespondError(msg.ID, -32602, err.Error())
		return
	}
	if p.NodeID != "" && p.NodeID != b.nodeID {
		worker := b.getEngineMgr()
		if worker == nil {
			_ = b.codec.RespondError(msg.ID, -32000, "engine manager unavailable")
			return
		}
		err = worker.RelayRequest("engine:remote-inspect-model", msg.Params, func(result json.RawMessage, rpcErr *RPCError, err error) {
			if err != nil {
				_ = b.codec.RespondError(msg.ID, -32000, err.Error())
			} else if rpcErr != nil {
				_ = b.codec.RespondError(msg.ID, rpcErr.Code, rpcErr.Message)
			} else {
				_ = b.codec.Respond(msg.ID, result)
			}
		})
		if err != nil {
			_ = b.codec.RespondError(msg.ID, -32000, err.Error())
		}
		return
	}
	result, err := b.inspectModelSettings(context.Background(), p, "")
	if err != nil {
		_ = b.codec.RespondError(msg.ID, engineSettingsErrorCode(err), err.Error())
	} else {
		_ = b.codec.Respond(msg.ID, result)
	}
}
