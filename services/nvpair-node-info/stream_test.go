// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"encoding/json"
	"os"
	"testing"

	"nvpair-shared/noderec"
)

const mib = uint64(1) << 20

// baseNodeInfo is a fully populated sample the threshold cases mutate.
func baseNodeInfo() NodeInfoResponse {
	cluster := "cluster-a"
	return NodeInfoResponse{
		MemoryObservation: &noderec.MemoryObservation{
			RunID: "run-1",
			Domains: []noderec.MemoryDomain{{
				ID:    "system",
				Kind:  "shared",
				Total: noderec.MemoryFact{Valid: true},
				Free:  noderec.MemoryFact{Valid: true},
			}},
		},
		ServiceBuildID: "build-1",
		GPUs: []GPUInfo{{
			Name:               "GPU",
			PhysicalID:         "pci:1",
			VramBytes:          16 << 30,
			VramUsedBytes:      4 << 30,
			UtilizationPercent: 40,
		}},
		CPU:            &CPUInfo{Name: "CPU", Cores: 8, UtilizationPercent: 30},
		Memory:         &MemoryInfo{TotalBytes: 4 << 30, UsedBytes: 2 << 30},
		TelemetryValid: true,
		MSSince:        100,
		HostUUID:       "host-a",
		ClusterUUID:    &cluster,
	}
}

func TestNodeInfoChanged(t *testing.T) {
	empty, other := "", "cluster-b"
	memory32GiB := func(r *NodeInfoResponse) { r.Memory.TotalBytes = 32 << 30 }
	vram4GiB := func(r *NodeInfoResponse) { r.GPUs[0].VramBytes, r.GPUs[0].VramUsedBytes = 4<<30, 1<<30 }
	cases := []struct {
		name   string
		setup  func(*NodeInfoResponse) // applied to both samples
		mutate func(*NodeInfoResponse) // applied to the candidate
		want   bool
	}{
		{"unchanged sample", nil, func(*NodeInfoResponse) {}, false},
		{"age only", nil, func(r *NodeInfoResponse) { r.MSSince = 900 }, false},
		{"GPU utilization +4", nil, func(r *NodeInfoResponse) { r.GPUs[0].UtilizationPercent += 4 }, false},
		{"GPU utilization +5", nil, func(r *NodeInfoResponse) { r.GPUs[0].UtilizationPercent += 5 }, true},
		{"GPU utilization -5", nil, func(r *NodeInfoResponse) { r.GPUs[0].UtilizationPercent -= 5 }, true},
		{"CPU utilization +9", nil, func(r *NodeInfoResponse) { r.CPU.UtilizationPercent += 9 }, false},
		{"CPU utilization +10", nil, func(r *NodeInfoResponse) { r.CPU.UtilizationPercent += 10 }, true},
		{"CPU utilization -10", nil, func(r *NodeInfoResponse) { r.CPU.UtilizationPercent -= 10 }, true},
		{"memory +63 MiB on 4 GiB", nil, func(r *NodeInfoResponse) { r.Memory.UsedBytes += 63 * mib }, false},
		{"memory +64 MiB on 4 GiB", nil, func(r *NodeInfoResponse) { r.Memory.UsedBytes += 64 * mib }, true},
		{"memory -64 MiB on 4 GiB", nil, func(r *NodeInfoResponse) { r.Memory.UsedBytes -= 64 * mib }, true},
		{"memory +300 MiB on 32 GiB", memory32GiB, func(r *NodeInfoResponse) { r.Memory.UsedBytes += 300 * mib }, false},
		{"memory +330 MiB on 32 GiB", memory32GiB, func(r *NodeInfoResponse) { r.Memory.UsedBytes += 330 * mib }, true},
		{"VRAM +63 MiB on 4 GiB", vram4GiB, func(r *NodeInfoResponse) { r.GPUs[0].VramUsedBytes += 63 * mib }, false},
		{"VRAM +64 MiB on 4 GiB", vram4GiB, func(r *NodeInfoResponse) { r.GPUs[0].VramUsedBytes += 64 * mib }, true},
		{"VRAM +160 MiB on 16 GiB", nil, func(r *NodeInfoResponse) { r.GPUs[0].VramUsedBytes += 160 * mib }, false},
		{"VRAM +164 MiB on 16 GiB", nil, func(r *NodeInfoResponse) { r.GPUs[0].VramUsedBytes += 164 * mib }, true},
		{"VRAM -164 MiB on 16 GiB", nil, func(r *NodeInfoResponse) { r.GPUs[0].VramUsedBytes -= 164 * mib }, true},
		{"telemetryValid flip", nil, func(r *NodeInfoResponse) { r.TelemetryValid = false }, true},
		{"clusterUuid value to absent", nil, func(r *NodeInfoResponse) { r.ClusterUUID = nil }, true},
		{"clusterUuid value to empty", nil, func(r *NodeInfoResponse) { r.ClusterUUID = &empty }, true},
		{"clusterUuid value to other", nil, func(r *NodeInfoResponse) { r.ClusterUUID = &other }, true},
		{"serviceBuildId", nil, func(r *NodeInfoResponse) { r.ServiceBuildID = "build-2" }, true},
		{"hostUuid", nil, func(r *NodeInfoResponse) { r.HostUUID = "host-b" }, true},
		{"GPU count", nil, func(r *NodeInfoResponse) { r.GPUs = append(r.GPUs, GPUInfo{Name: "GPU 2"}) }, true},
		{"GPU name", nil, func(r *NodeInfoResponse) { r.GPUs[0].Name = "Other" }, true},
		{"GPU physical ID", nil, func(r *NodeInfoResponse) { r.GPUs[0].PhysicalID = "" }, true},
		{"GPU VRAM total", nil, func(r *NodeInfoResponse) { r.GPUs[0].VramBytes = 8 << 30 }, true},
		{"CPU absent", nil, func(r *NodeInfoResponse) { r.CPU = nil }, true},
		{"CPU name", nil, func(r *NodeInfoResponse) { r.CPU.Name = "Other" }, true},
		{"CPU cores", nil, func(r *NodeInfoResponse) { r.CPU.Cores = 16 }, true},
		{"memory absent", nil, func(r *NodeInfoResponse) { r.Memory = nil }, true},
		{"memory total", nil, func(r *NodeInfoResponse) { r.Memory.TotalBytes = 8 << 30 }, true},
		{"memory observation absent", nil, func(r *NodeInfoResponse) { r.MemoryObservation = nil }, true},
		{"memory observation run ID", nil, func(r *NodeInfoResponse) { r.MemoryObservation.RunID = "run-2" }, true},
		{"memory observation domain ID", nil, func(r *NodeInfoResponse) { r.MemoryObservation.Domains[0].ID = "other" }, true},
		{"memory observation domain added", nil, func(r *NodeInfoResponse) {
			r.MemoryObservation.Domains = append(r.MemoryObservation.Domains, noderec.MemoryDomain{ID: "gpu"})
		}, true},
		{"memory observation total valid", nil, func(r *NodeInfoResponse) { r.MemoryObservation.Domains[0].Total.Valid = false }, true},
		{"memory observation free valid", nil, func(r *NodeInfoResponse) { r.MemoryObservation.Domains[0].Free.Valid = false }, true},
		{"memory observation bytes only", nil, func(r *NodeInfoResponse) { r.MemoryObservation.Domains[0].Free.Bytes = "123" }, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			published, candidate := baseNodeInfo(), baseNodeInfo()
			if tc.setup != nil {
				tc.setup(&published)
				tc.setup(&candidate)
			}
			tc.mutate(&candidate)
			if got := nodeInfoChanged(published, candidate); got != tc.want {
				t.Errorf("nodeInfoChanged = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestNodeInfoChangedReplay publishes a recorded sequence of consecutive
// one-second bodies (captured on a development build) through the threshold
// rule and expects fewer frames than samples.
func TestNodeInfoChangedReplay(t *testing.T) {
	file, err := os.Open("testdata/node-info-sequence.txt")
	if err != nil {
		t.Fatalf("open sequence: %v", err)
	}
	defer file.Close()
	var samples, frames int
	var published NodeInfoResponse
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64<<10), 1<<20)
	for scanner.Scan() {
		var candidate NodeInfoResponse
		if err := json.Unmarshal(scanner.Bytes(), &candidate); err != nil {
			t.Fatalf("sample %d: %v", samples+1, err)
		}
		if samples == 0 || nodeInfoChanged(published, candidate) {
			published = candidate
			frames++
		}
		samples++
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("read sequence: %v", err)
	}
	if samples < 20 {
		t.Fatalf("sequence holds %d samples, want at least 20", samples)
	}
	if frames >= samples {
		t.Fatalf("published %d frames for %d samples, want fewer frames than samples", frames, samples)
	}
	t.Logf("%d frames for %d samples", frames, samples)
}
