// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"nvpair-shared/noderec"
)

func physicalIdentityDomain(id string, aliases []string, total uint64) *noderec.MemoryDomain {
	return &noderec.MemoryDomain{
		ID: id, Kind: "dedicated", DeviceIDs: aliases,
		Total: memoryFact(total, "fixture", time.Unix(100, 0)),
		Free:  memoryFact(total, "fixture", time.Unix(100, 0)),
	}
}

func physicalIdentityResponse(t *testing.T, static []GPUInfo, snap statsSnapshot) NodeInfoResponse {
	t.Helper()
	var response NodeInfoResponse
	if err := json.Unmarshal(buildResponseAt(static, nil, 1000, snap, "fixture-host", nil, time.Unix(100, 0)), &response); err != nil {
		t.Fatalf("decode node-info response: %v", err)
	}
	return response
}

func TestBuildResponsePhysicalIDsFollowDedicatedAliases(t *testing.T) {
	for _, tc := range []struct {
		name   string
		first  string
		second string
	}{
		{"Windows LUIDs", "luid_0x00000000_0x00000001_phys_0", "luid_0x00000000_0x00000002_phys_0"},
		{"Linux UUIDs", "GPU-12345678-1234-1234-1234-123456789abc", "GPU-22345678-1234-1234-1234-123456789abc"},
		{"Darwin registry IDs", "ioreg:2a", "ioreg:63"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The inventory is reversed relative to the memory-domain order.
			// Neither adapter has utilization or memory-use counters: idle and
			// unused adapters still contribute to the full hardware identity.
			static := []GPUInfo{
				{Name: "GPU B", VramBytes: 200, statsKey: tc.second},
				{Name: "GPU A", VramBytes: 100, statsKey: tc.first},
			}
			before := append([]GPUInfo(nil), static...)
			snap := statsSnapshot{GPU: map[string]gpuStat{
				tc.first:  {Memory: physicalIdentityDomain("pool-a", []string{tc.first, "pci:0000:01:00.0"}, 100)},
				tc.second: {Memory: physicalIdentityDomain("pool-b", []string{tc.second, "pci:0000:02:00.0"}, 200)},
			}}
			response := physicalIdentityResponse(t, static, snap)
			want := []GPUInfo{
				{Name: "GPU B", PhysicalID: tc.second, VramBytes: 200},
				{Name: "GPU A", PhysicalID: tc.first, VramBytes: 100},
			}
			if !reflect.DeepEqual(response.GPUs, want) {
				t.Fatalf("GPU identity = %+v, want %+v", response.GPUs, want)
			}
			if response.MemoryObservation == nil || len(response.MemoryObservation.Domains) != 3 {
				t.Fatalf("physical pools missing: %+v", response.MemoryObservation)
			}
			if got := response.MemoryObservation.Domains[1].DeviceIDs; !reflect.DeepEqual(got, []string{tc.first, "pci:0000:01:00.0"}) {
				t.Fatalf("existing physical aliases changed: %v", got)
			}
			if !reflect.DeepEqual(static, before) {
				t.Fatalf("response mutated caller inventory: %+v", static)
			}
		})
	}
}

func TestBuildResponsePhysicalIDsOmitUnprovenAssociations(t *testing.T) {
	const alias = "ioreg:2a"
	dedicated := physicalIdentityDomain("pool", []string{alias}, 100)
	unknown := &noderec.MemoryDomain{
		ID: "unknown", Kind: "unknown", DeviceIDs: []string{alias},
		Total: noderec.MemoryFact{Reason: "unavailable"}, Free: noderec.MemoryFact{Reason: "unavailable"},
	}
	for _, tc := range []struct {
		name      string
		static    []GPUInfo
		recovered []GPUInfo
		stats     map[string]gpuStat
	}{
		{
			name:   "missing join key clears supplied identity",
			static: []GPUInfo{{Name: "GPU", VramBytes: 100, PhysicalID: alias}},
			stats:  map[string]gpuStat{alias: {Memory: dedicated}},
		},
		{
			name:   "missing sample clears supplied identity",
			static: []GPUInfo{{Name: "GPU", VramBytes: 100, PhysicalID: alias, statsKey: alias}},
		},
		{
			name:   "utilization alone does not establish a pool",
			static: []GPUInfo{{Name: "GPU", VramBytes: 100, statsKey: alias}},
			stats:  map[string]gpuStat{alias: {UtilizationPct: 70}},
		},
		{
			name:   "unknown topology",
			static: []GPUInfo{{Name: "GPU", VramBytes: 100, statsKey: alias}},
			stats:  map[string]gpuStat{alias: {Memory: unknown}},
		},
		{
			name:   "UMA display marker does not prove a shared pool",
			static: []GPUInfo{{Name: "GPU", VramBytes: 100, statsKey: alias, usesSystemMemoryUsage: true}},
			stats:  map[string]gpuStat{alias: {Memory: unknown}},
		},
		{
			name:   "pool ID without device alias",
			static: []GPUInfo{{Name: "GPU", VramBytes: 100, statsKey: alias}},
			stats:  map[string]gpuStat{alias: {Memory: physicalIdentityDomain(alias, nil, 100)}},
		},
		{
			name:   "matching name and capacity cannot join different IDs",
			static: []GPUInfo{{Name: "GPU", VramBytes: 100, statsKey: "other"}},
			stats:  map[string]gpuStat{alias: {Memory: dedicated}},
		},
		{
			name: "duplicate static adapter keys",
			static: []GPUInfo{
				{Name: "GPU A", VramBytes: 100, statsKey: alias},
				{Name: "GPU B", VramBytes: 100, statsKey: alias},
			},
			stats: map[string]gpuStat{alias: {Memory: dedicated}},
		},
		{
			name: "duplicate recovered adapter keys",
			recovered: []GPUInfo{
				{Name: "GPU A", VramBytes: 100, statsKey: alias},
				{Name: "GPU B", VramBytes: 100, statsKey: alias},
			},
			stats: map[string]gpuStat{alias: {Memory: dedicated}},
		},
		{
			name:   "overlapping physical aliases invalidate observation",
			static: []GPUInfo{{Name: "GPU", VramBytes: 100, statsKey: alias}},
			stats: map[string]gpuStat{
				alias:   {Memory: dedicated},
				"other": {Memory: physicalIdentityDomain("other", []string{alias}, 100)},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			response := physicalIdentityResponse(t, tc.static, statsSnapshot{GPU: tc.stats, GPUInventory: tc.recovered})
			if len(response.GPUs) == 0 {
				t.Fatal("unknown association hid GPU inventory")
			}
			for _, gpu := range response.GPUs {
				if gpu.PhysicalID != "" || gpu.VramBytes != 100 {
					t.Fatalf("unproven identity or changed scalar capacity: %+v", gpu)
				}
			}
		})
	}
}

func TestBuildResponsePhysicalIDsSharedPool(t *testing.T) {
	static := []GPUInfo{
		{Name: "Apple GPU B", VramBytes: 1000, statsKey: "ioreg:b"},
		{Name: "Apple GPU A", VramBytes: 1000, statsKey: "ioreg:a"},
	}
	snap := statsSnapshot{
		Memory: systemMemory(1000, 400, "fixture", time.Unix(100, 0)),
		GPU:    map[string]gpuStat{"ioreg:a": {Shared: true}, "ioreg:b": {Shared: true}},
	}
	response := physicalIdentityResponse(t, static, snap)
	want := []GPUInfo{
		{Name: "Apple GPU B", PhysicalID: "ioreg:b", VramBytes: 1000},
		{Name: "Apple GPU A", PhysicalID: "ioreg:a", VramBytes: 1000},
	}
	if !reflect.DeepEqual(response.GPUs, want) {
		t.Fatalf("shared GPU identity = %+v, want %+v", response.GPUs, want)
	}
	memory := response.MemoryObservation
	if memory == nil || len(memory.Domains) != 1 {
		t.Fatalf("UMA must carry one physical pool: %+v", memory)
	}
	pool := memory.Domains[0]
	if pool.Kind != "shared" || pool.Total.Bytes != "1000" || !reflect.DeepEqual(pool.DeviceIDs, []string{"ioreg:a", "ioreg:b"}) {
		t.Fatalf("shared pool or aliases changed: %+v", pool)
	}
	if response.Memory == nil || response.Memory.TotalBytes != 1000 {
		t.Fatalf("RAM inventory changed: %+v", response.Memory)
	}
	if snap.Memory.Domains[0].Kind != "system" || len(snap.Memory.Domains[0].DeviceIDs) != 0 {
		t.Fatalf("response mutated collector RAM pool: %+v", snap.Memory)
	}
}

func TestBuildResponsePhysicalIDsRecoveredInventoryAndCanonicalWire(t *testing.T) {
	const alias = "luid_0x00000000_0x00000001_phys_0"
	static := []GPUInfo{{Name: "GPU", statsKey: alias}}
	recovered := []GPUInfo{{Name: "GPU", VramBytes: 100, statsKey: alias}}
	snap := statsSnapshot{
		GPUInventory: recovered,
		GPU:          map[string]gpuStat{alias: {Memory: physicalIdentityDomain(alias, []string{alias}, 200)}},
	}
	// Both startup recovery and enrichment of an existing adapter use the same
	// canonical wire shape; a memory total never overwrites scalar evidence.
	for _, tc := range []struct {
		name      string
		inventory []GPUInfo
	}{
		{"startup recovery", nil},
		{"existing adapter enrichment", static},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var canonical struct {
				GPUs []noderec.GPUInfo `json:"GPUs"`
			}
			body := buildResponse(tc.inventory, nil, 0, snap, "fixture-host", nil)
			if err := json.Unmarshal(body, &canonical); err != nil {
				t.Fatalf("decode canonical GPU inventory: %v", err)
			}
			want := []noderec.GPUInfo{{Name: "GPU", PhysicalID: alias, VramBytes: 100}}
			if !reflect.DeepEqual(canonical.GPUs, want) {
				t.Fatalf("canonical GPUs = %+v, want %+v", canonical.GPUs, want)
			}
		})
	}
	if static[0].VramBytes != 0 || static[0].PhysicalID != "" || recovered[0].PhysicalID != "" {
		t.Fatalf("response mutated caller inventories: static=%+v recovered=%+v", static, recovered)
	}
}
