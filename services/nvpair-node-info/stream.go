// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"time"

	"nvpair-shared/noderec"
)

// Telemetry stream cadence and change thresholds (see the README's
// "Telemetry stream" section).
var nodeInfoStreamKeepalive = 2 * time.Second // a null line: still within thresholds of the last frame

const (
	nodeInfoStreamWriteTimeout = 5 * time.Second
	nodeInfoPublishEvery       = time.Second
	nodeInfoStreamsPerAddress  = 8 // per non-loopback remote address

	gpuUtilizationStep = 5  // percentage points
	cpuUtilizationStep = 10 // percentage points
	memoryStepFloor    = 64 << 20
	memoryStepFraction = 0.01 // of the GPU's VRAM total or the memory total
)

// nodeInfoChanged reports whether candidate differs from published enough to
// publish a new frame. See the thresholds in the package README.
func nodeInfoChanged(published, candidate NodeInfoResponse) bool {
	if published.ServiceBuildID != candidate.ServiceBuildID ||
		published.HostUUID != candidate.HostUUID ||
		published.TelemetryValid != candidate.TelemetryValid ||
		!sameOptionalString(published.ClusterUUID, candidate.ClusterUUID) ||
		memoryObservationChanged(published.MemoryObservation, candidate.MemoryObservation) {
		return true
	}
	if (published.CPU == nil) != (candidate.CPU == nil) {
		return true
	}
	if published.CPU != nil {
		if published.CPU.Name != candidate.CPU.Name || published.CPU.Cores != candidate.CPU.Cores ||
			distance(uint64(published.CPU.UtilizationPercent), uint64(candidate.CPU.UtilizationPercent)) >= cpuUtilizationStep {
			return true
		}
	}
	if (published.Memory == nil) != (candidate.Memory == nil) {
		return true
	}
	if published.Memory != nil {
		if published.Memory.TotalBytes != candidate.Memory.TotalBytes ||
			memoryMoved(published.Memory.UsedBytes, candidate.Memory.UsedBytes, candidate.Memory.TotalBytes) {
			return true
		}
	}
	if len(published.GPUs) != len(candidate.GPUs) {
		return true
	}
	for i := range published.GPUs {
		a, b := published.GPUs[i], candidate.GPUs[i]
		if a.Name != b.Name || a.PhysicalID != b.PhysicalID || a.VramBytes != b.VramBytes ||
			distance(uint64(a.UtilizationPercent), uint64(b.UtilizationPercent)) >= gpuUtilizationStep ||
			memoryMoved(a.VramUsedBytes, b.VramUsedBytes, b.VramBytes) {
			return true
		}
	}
	return false
}

// sameOptionalString keeps absent, empty and a value as three distinct states.
func sameOptionalString(a, b *string) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return *a == *b
}

// memoryObservationChanged compares the observation's static parts (run ID and
// the ordered domain IDs) and each fact's valid flag. Byte counts move with
// Memory.UsedBytes and are not compared here.
func memoryObservationChanged(a, b *noderec.MemoryObservation) bool {
	if a == nil || b == nil {
		return (a == nil) != (b == nil)
	}
	if a.RunID != b.RunID || len(a.Domains) != len(b.Domains) {
		return true
	}
	for i := range a.Domains {
		x, y := a.Domains[i], b.Domains[i]
		if x.ID != y.ID || x.Total.Valid != y.Total.Valid || x.Free.Valid != y.Free.Valid {
			return true
		}
	}
	return false
}

// memoryMoved reports whether used bytes moved by at least
// max(memoryStepFloor, memoryStepFraction of total).
func memoryMoved(published, candidate, total uint64) bool {
	step := uint64(float64(total) * memoryStepFraction)
	if step < memoryStepFloor {
		step = memoryStepFloor
	}
	return distance(published, candidate) >= step
}

func distance(a, b uint64) uint64 {
	if a > b {
		return a - b
	}
	return b - a
}
