// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"slices"
	"strconv"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

type diagnosticHardware struct {
	CPUModel      string                      `json:"cpuModel"`
	TotalRAMBytes string                      `json:"totalRamBytes"`
	GPUs          []modelconfig.DiagnosticGPU `json:"gpus"`
}

func diagnosticIdentityHardware(identity modelconfig.DiagnosticIdentity) *diagnosticHardware {
	return &diagnosticHardware{CPUModel: identity.CPUModel, TotalRAMBytes: identity.TotalRAMBytes, GPUs: slices.Clone(identity.GPUs)}
}

type diagnosticInspectEnvelope struct {
	settings.WorkerRequest[modelconfig.InspectRequest]
	Hardware *diagnosticHardware `json:"hardware,omitempty"`
}

// This is a read through the existing scanner owner, not another hardware
// cache or native probe. The scanner must serve self via verified loopback.
// A directory GPU array cannot supply this sample's physical association.
func (b *Broker) diagnosticTargetHardware(ctx context.Context, memory *modelconfig.TargetMemory) (*diagnosticHardware, error) {
	if memory == nil || memory.HostID != b.nodeID {
		return nil, diagnosticError("target hardware unavailable")
	}
	observation := memory.At(noderec.MemoryBootClock)
	if observation == nil {
		return nil, diagnosticError("target hardware memory expired")
	}
	scanner := b.getScanner()
	if scanner == nil || scanner.peer == nil {
		return nil, diagnosticError("target node information unavailable")
	}
	ctx, cancel := context.WithTimeout(ctx, nodeInfoRPCBudget)
	defer cancel()
	raw, rpcErr, err := scanner.getNodeInfo(ctx, settingsJSON(nodeInfoRPCParams{HostUUID: b.nodeID}))
	if err != nil || rpcErr != nil {
		return nil, diagnosticError("fresh target node information unavailable")
	}
	b.workersMu.Lock()
	current := b.scanner == scanner && b.scannerSource == scanner
	b.workersMu.Unlock()
	if !current {
		return nil, diagnosticError("node information source changed")
	}
	var info struct {
		HostUUID          string                     `json:"hostUuid"`
		GPUs              []noderec.GPUInfo          `json:"GPUs"`
		CPU               *noderec.CPUInfo           `json:"cpu"`
		Memory            *noderec.MemoryInfo        `json:"memory"`
		MemoryObservation *noderec.MemoryObservation `json:"memoryObservation"`
	}
	// telemetryValid/msSince describe utilization, not independent memory
	// freshness. A current physical pool remains usable without utilization.
	if json.Unmarshal(raw, &info) != nil || info.HostUUID != memory.HostID || info.CPU == nil || info.CPU.Name == "" || info.Memory == nil || info.Memory.TotalBytes == 0 || info.MemoryObservation == nil || info.MemoryObservation.Validate() != nil || info.MemoryObservation.RunID != observation.RunID {
		return nil, diagnosticError("target hardware identity or freshness unavailable")
	}
	hardware := &diagnosticHardware{CPUModel: info.CPU.Name, TotalRAMBytes: strconv.FormatUint(info.Memory.TotalBytes, 10), GPUs: []modelconfig.DiagnosticGPU{}}
	seen := map[string]bool{}
	for _, gpu := range info.GPUs {
		if gpu.PhysicalID == "" || gpu.Name == "" || seen[gpu.PhysicalID] {
			return nil, diagnosticError("physical GPU association unavailable")
		}
		seen[gpu.PhysicalID] = true
		capacity := ""
		for _, pool := range observation.Domains {
			if !slices.Contains(pool.DeviceIDs, gpu.PhysicalID) {
				continue
			}
			if capacity != "" || !pool.Total.Valid {
				return nil, diagnosticError("ambiguous GPU pool association")
			}
			if pool.Kind == "shared" {
				if pool.Total.Bytes != hardware.TotalRAMBytes {
					return nil, diagnosticError("shared GPU capacity differs from RAM")
				}
			} else if pool.Kind != "dedicated" || strconv.FormatUint(gpu.VramBytes, 10) != pool.Total.Bytes || gpu.VramBytes == 0 {
				return nil, diagnosticError("GPU capacity differs from physical pool")
			}
			capacity = pool.Total.Bytes
		}
		if capacity == "" {
			return nil, diagnosticError("GPU missing from current physical pools")
		}
		hardware.GPUs = append(hardware.GPUs, modelconfig.DiagnosticGPU{UUID: gpu.PhysicalID, Name: gpu.Name, TotalMemoryBytes: capacity})
	}
	host := false
	for _, pool := range observation.Domains {
		// Bind the fetched full node-info response to the same producer and
		// topology as the verified local memory cache, without comparing free
		// bytes sampled at different instants.
		matched := false
		for _, fresh := range info.MemoryObservation.Domains {
			ids, other := slices.Clone(pool.DeviceIDs), slices.Clone(fresh.DeviceIDs)
			slices.Sort(ids)
			slices.Sort(other)
			if fresh.ID == pool.ID && fresh.Kind == pool.Kind && fresh.Total.Valid && fresh.Total.Bytes == pool.Total.Bytes && slices.Equal(ids, other) && fresh.Total.AgeMS <= noderec.MemoryFreshMS && !fresh.Total.SampledAt.After(time.Now()) && time.Since(fresh.Total.SampledAt) <= time.Duration(noderec.MemoryFreshMS)*time.Millisecond {
				matched = true
			}
		}
		if !matched {
			return nil, diagnosticError("full hardware sample differs from current memory topology")
		}
		if pool.Kind == "system" || pool.Kind == "shared" {
			host = pool.Total.Valid && pool.Total.Bytes == hardware.TotalRAMBytes
		}
		if pool.Kind == "dedicated" {
			// A physical device may advertise UUID and PCI aliases together.
			// Exactly one exported GPU must bind this pool, not every alias.
			bound := 0
			for _, id := range pool.DeviceIDs {
				if seen[id] {
					bound++
				}
			}
			if bound != 1 {
				return nil, diagnosticError("dedicated GPU pool association is incomplete or ambiguous")
			}
		} else if pool.Kind == "shared" {
			// Shared pools can contain several physical GPUs. Do not let a
			// partial GPU list silently stand for the entire shared pool.
			for _, id := range pool.DeviceIDs {
				if !seen[id] {
					return nil, diagnosticError("usable shared GPU omitted from hardware identity")
				}
			}
		}
	}
	if !host || len(info.MemoryObservation.Domains) != len(observation.Domains) {
		return nil, diagnosticError("host RAM association unavailable")
	}
	slices.SortFunc(hardware.GPUs, func(a, b modelconfig.DiagnosticGPU) int {
		if a.UUID < b.UUID {
			return -1
		}
		if a.UUID > b.UUID {
			return 1
		}
		return 0
	})
	return hardware, nil
}

func (b *Broker) diagnosticResidentReloads() []modelconfig.DiagnosticModelRef {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	policy := r.current
	policy.Profiles = slices.Clone(policy.Profiles)
	policy.ResidentModels = slices.Clone(policy.ResidentModels)
	r.mu.Unlock()
	models := []modelconfig.DiagnosticModelRef{}
	if policy.Availability != nodepolicy.Available || policy.PauseMode != "" {
		return models
	}
	for _, ref := range b.residentRefs(policy) {
		if profile, found := b.residentProfile(policy, ref); found {
			models = append(models, modelconfig.DiagnosticModelRef{Engine: profile.Engine, ModelID: profile.Model, ProfileKey: profile.Key})
		}
	}
	return models
}
