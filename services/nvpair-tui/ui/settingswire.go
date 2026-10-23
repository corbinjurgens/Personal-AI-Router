// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"encoding/json"
	"errors"
	"fmt"

	"nvpair-shared/engineintent"
	"nvpair-tui/rpc"

	tea "github.com/charmbracelet/bubbletea"
)

// commandFailure preserves the target's refusal and its machine-readable code.
func commandFailure(err error) string {
	var refusal *rpc.RPCError
	if errors.As(err, &refusal) {
		switch refusal.Code {
		case -32013:
			return fmt.Sprintf("engine locked (%d): %s", refusal.Code, refusal.Message)
		case -32012:
			return fmt.Sprintf("remote management denied (%d): %s", refusal.Code, refusal.Message)
		case -32011:
			return fmt.Sprintf("local veto (%d): %s", refusal.Code, refusal.Message)
		case -32010:
			return fmt.Sprintf("revision conflict (%d): %s - reload and retry", refusal.Code, refusal.Message)
		}
	}
	return err.Error()
}

type targetPolicy struct {
	Availability string `json:"availability"`
	Policy       struct {
		PausedBy string `json:"pausedBy"`
	} `json:"policy"`
}

type detailPolicyMsg struct {
	visit  int
	route  int
	policy targetPolicy
	err    error
}

type detailIntentMsg struct {
	visit      int
	generation int
	intent     engineintent.Snapshot
	err        error
}

func (d *nodeDetail) policyCmd() tea.Cmd {
	if !d.enginesQueryable() || d.policyDisconnected || (d.remote() && d.node.presence == presenceOffline) {
		return nil
	}
	route := d.policyRoute
	visit := d.telemetryGen
	params := map[string]any{}
	if d.remote() {
		params["nodeId"] = d.node.key
	}
	return d.own(call(d.client, "policy:get", params, func(msg *rpc.Message, err error) tea.Msg {
		out := detailPolicyMsg{visit: visit, route: route, err: err}
		if err == nil {
			out.err = json.Unmarshal(msg.Result, &out.policy)
		}
		return out
	}))
}

func (d *nodeDetail) intentCmd() tea.Cmd {
	if d.remote() || d.policyDisconnected {
		return nil
	}
	d.intentGeneration++
	generation := d.intentGeneration
	visit := d.telemetryGen
	return d.own(call(d.client, "engine:intent", nil, func(msg *rpc.Message, err error) tea.Msg {
		var result engineintent.Snapshot
		if err == nil {
			err = json.Unmarshal(msg.Result, &result)
		}
		if err == nil {
			err = result.Validate()
		}
		return detailIntentMsg{visit: visit, generation: generation, intent: result, err: err}
	}))
}

func (d *nodeDetail) invalidateIntent() {
	d.intentGeneration++
	d.intent = engineintent.Snapshot{}
	d.refreshEngines()
}

func (d *nodeDetail) policyLabel() string {
	if !d.policyFresh {
		return "Availability unavailable"
	}
	label := "Availability: " + d.policy.Availability
	if d.policy.Policy.PausedBy != "" {
		label += " (paused by " + d.policy.Policy.PausedBy + ")"
	}
	return label
}
