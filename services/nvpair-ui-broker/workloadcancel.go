// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// workloadcancel.go is workloads:cancel (FORK_DESIGN.md §4). A job can only be
// cancelled where it originated, because only the origin's proxy holds the
// request. On the origin the broker asks its proxy; anywhere else it hands the
// cancel to the workload-manager, which carries it to the origin over the
// existing pinned peer channel, and the origin's broker acts on it.

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
)

const (
	methodWorkloadsCancel = "workloads:cancel"
	proxyWorkloadCancel   = "workload/cancel"
)

// workloadCancelParams is the workloads:cancel payload on every hop.
type workloadCancelParams struct {
	OriginatedFrom string `json:"originatedFrom"`
	WorkloadID     string `json:"workloadId"`
	Regenerate     bool   `json:"regenerate,omitempty"`
}

// handleWorkloadsCancel answers a client's workloads:cancel with {ok}. On the
// origin, ok reports whether the proxy found the job still running. Elsewhere
// it reports that the cancel was handed on; the origin applies it
// asynchronously and the outcome arrives on the workloads stream.
func (b *Broker) handleWorkloadsCancel(msg *Message) {
	var p workloadCancelParams
	if json.Unmarshal(msg.Params, &p) != nil || p.OriginatedFrom == "" || p.WorkloadID == "" {
		_ = b.codec.RespondError(msg.ID, -32602, "originatedFrom and workloadId are required")
		return
	}
	if p.OriginatedFrom == b.nodeID {
		found, err := b.cancelLocalWorkload(p)
		if err != nil {
			_ = b.codec.RespondError(msg.ID, -32000, err.Error())
			return
		}
		_ = b.codec.Respond(msg.ID, map[string]bool{"ok": found})
		return
	}
	wm := b.getWorkloadMgr()
	if wm == nil {
		_ = b.codec.RespondError(msg.ID, -32000, "workload-manager not available")
		return
	}
	data, err := json.Marshal(p)
	if err == nil {
		err = wm.Forward(methodWorkloadsCancel, data)
	}
	if err != nil {
		_ = b.codec.RespondError(msg.ID, -32000, fmt.Sprintf("could not relay the cancel: %v", err))
		return
	}
	slog.Info("relayed workload cancel to its origin", "origin", p.OriginatedFrom, "workloadId", p.WorkloadID, "regenerate", p.Regenerate)
	_ = b.codec.Respond(msg.ID, map[string]bool{"ok": true})
}

// cancelLocalWorkload asks this node's proxy to cancel a job it originated.
func (b *Broker) cancelLocalWorkload(p workloadCancelParams) (bool, error) {
	pp := b.policyProxy()
	if pp == nil {
		return false, fmt.Errorf("proxy not available")
	}
	data, err := json.Marshal(map[string]any{"workloadId": p.WorkloadID, "regenerate": p.Regenerate})
	if err != nil {
		return false, err
	}
	result, rpcErr, err := pp.Call(context.Background(), proxyWorkloadCancel, data)
	if err != nil {
		return false, err
	}
	if rpcErr != nil {
		return false, fmt.Errorf("%s", rpcErr.Message)
	}
	var r struct {
		Found bool `json:"found"`
	}
	if err := json.Unmarshal(result, &r); err != nil {
		return false, fmt.Errorf("invalid workload/cancel result: %w", err)
	}
	return r.Found, nil
}

// handlePeerWorkloadCancel acts on a cancel a peer relayed. Only the origin
// acts; every other node drops it, because a peer that does not know the
// origin's address sends it to all of them.
func (b *Broker) handlePeerWorkloadCancel(params json.RawMessage) {
	var p workloadCancelParams
	if json.Unmarshal(params, &p) != nil || p.WorkloadID == "" {
		slog.Warn("workload-manager relayed an invalid workload cancel")
		return
	}
	if p.OriginatedFrom != b.nodeID {
		slog.Debug("ignoring workload cancel for another origin", "origin", p.OriginatedFrom)
		return
	}
	go func() {
		found, err := b.cancelLocalWorkload(p)
		if err != nil {
			slog.Warn("peer-requested workload cancel failed", "workloadId", p.WorkloadID, "err", err)
			return
		}
		slog.Info("peer-requested workload cancel", "workloadId", p.WorkloadID, "regenerate", p.Regenerate, "found", found)
	}()
}
