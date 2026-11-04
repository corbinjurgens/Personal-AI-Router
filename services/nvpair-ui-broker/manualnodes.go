// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"time"

	"nvpair-shared/buildinfo"
	"nvpair-shared/noderec"
)

// manualNodeStatus is the subset of nvpair-manual-nodes' ManualNodeStatus the
// broker needs. Its JSON tags match the producer's so node/discovered|
// updated|removed payloads unmarshal straight into it (the GPU/CPU/memory
// sub-objects reuse the broker's discovery types, whose tags are identical).
// Projection into the discovery store is owned by manualToEnriched.
type manualNodeStatus struct {
	MemoryObservation *noderec.MemoryObservation `json:"memoryObservation,omitempty"`
	MemoryHop         *noderec.MemoryHop         `json:"memoryHop,omitempty"`
	ServiceBuildID    buildinfo.ID               `json:"serviceBuildId,omitempty"`
	ID                string                     `json:"id"`
	Address           string                     `json:"address"`
	NodeInfoPort      int                        `json:"node_info_port"`
	GPUs              []GPUInfo                  `json:"gpus"`
	CPU               *CPUInfo                   `json:"cpu"`
	Memory            *MemoryInfo                `json:"memory"`
	TelemetryValid    bool                       `json:"telemetryValid"`
	MSSince           int64                      `json:"msSince"`
	// HostUUID is the remote's stable per-host identity, learned from its
	// node-info /v1/node-info. It lets a manual node key by the same permanent
	// identity as mDNS-discovered nodes (and dedup with itself when the same
	// machine is also discovered). Empty until the node-info probe succeeds.
	HostUUID string `json:"hostUuid,omitempty"`
}

type manualNodeStatusEntry struct {
	status     manualNodeStatus
	receivedAt time.Time
}

func manualNodeTelemetry(status manualNodeStatus, hostUUID string) noderec.NodeTelemetry {
	var utilization uint32
	for i := range status.GPUs {
		if status.GPUs[i].UtilizationPercent > utilization {
			utilization = status.GPUs[i].UtilizationPercent
		}
	}
	return noderec.NodeTelemetry{
		Memory:            status.MemoryObservation.Clone(),
		MemoryHop:         status.MemoryHop,
		HostUUID:          hostUUID,
		GPUUtilizationPct: utilization,
		TelemetryValid:    status.TelemetryValid,
		MSSince:           status.MSSince,
	}
}

// manualToEnriched projects a manual node's status onto the EnrichedNode
// the discovery store holds, so manual nodes share the single
// discovery:get-nodes / discovery:nodes-changed snapshot with mDNS nodes.
// Port is the node-info HTTP port the prober reached (mirroring the SRV
// port the scanner records for mDNS nodes); the user-entered address
// becomes the node's host/address so the snapshot's ipAddress projection
// can resolve it when the address is an IP literal.
func manualToEnriched(s manualNodeStatus) EnrichedNode {
	// Every node carries a non-empty operational key by the time it reaches the
	// store: the remote's real hostUuid once node-info reports it, else the
	// manual id (the user's name or "manual:<address>") until then. This is the
	// manual-node ingestion boundary — downstream keys off HostUUID with no
	// fallback. Once the real UUID is learned, a manually-added machine
	// that's also discovered over mDNS collapses to the one hostUuid-keyed entry.
	hostUUID := s.HostUUID
	if hostUUID == "" {
		hostUUID = s.ID
	}
	en := EnrichedNode{
		ServiceBuildID: s.ServiceBuildID,
		ID:             s.ID,
		HostUUID:       hostUUID,
		Host:           s.Address,
		Port:           s.NodeInfoPort,
		GPUs:           s.GPUs,
		CPU:            s.CPU,
		Memory:         s.Memory,
	}
	if s.Address != "" {
		en.Addresses = []string{s.Address}
	}
	return en
}
