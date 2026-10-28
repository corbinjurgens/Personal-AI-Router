// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"math"
	"nvpair-shared/noderec"
	"strings"
	"time"

	"howett.net/plist"
)

type ioRegistryPerformance struct {
	DeviceUtilization   *uint64 `plist:"Device Utilization %"`
	GPUActivity         *uint64 `plist:"GPU Activity(%)"`
	AllocSystemMemory   uint64  `plist:"Alloc system memory"`
	InUseSystemMemory   uint64  `plist:"In use system memory"`
	VRAMUsedBytes       *uint64 `plist:"vramUsedBytes"`
	InUseVidMemoryBytes *uint64 `plist:"inUseVidMemoryBytes"`
	VRAMFreeBytes       *uint64 `plist:"vramFreeBytes"`
}

type ioRegistryEntry struct {
	EntryID               uint64                `plist:"IORegistryEntryID"`
	EntryName             string                `plist:"IORegistryEntryName"`
	ObjectClass           string                `plist:"IOObjectClass"`
	Model                 string                `plist:"model"`
	VRAMTotalMB           uint64                `plist:"VRAM,totalMB"`
	Performance           ioRegistryPerformance `plist:"PerformanceStatistics"`
	RegistryEntryChildren []ioRegistryEntry     `plist:"IORegistryEntryChildren"`
}

func (e *ioRegistryEntry) UnmarshalPlist(unmarshal func(interface{}) error) error {
	// plist struct tags split at commas, so this literal native key needs
	// dictionary access. Decode the remaining fields through their typed owner.
	type entryFields ioRegistryEntry
	var fields entryFields
	if err := unmarshal(&fields); err != nil {
		return err
	}
	var dictionary map[string]interface{}
	if err := unmarshal(&dictionary); err != nil {
		return err
	}
	fields.VRAMTotalMB = 0
	if value, present := dictionary["VRAM,totalMB"]; present {
		total, ok := value.(uint64)
		if !ok {
			return fmt.Errorf("invalid IORegistry VRAM total")
		}
		fields.VRAMTotalMB = total
	}
	*e = ioRegistryEntry(fields)
	return nil
}

type darwinGPURecord struct {
	shared           bool
	memory           *noderec.MemoryDomain
	statsKey         string
	name             string
	vramTotal        uint64
	vramUsed         uint64
	utilizationPct   uint32
	utilizationValid bool
}

func parseIORegistryGPUs(data []byte, systemMemory uint64) ([]darwinGPURecord, error) {
	var roots []ioRegistryEntry
	if _, err := plist.Unmarshal(data, &roots); err != nil {
		return nil, fmt.Errorf("decode ioreg plist: %w", err)
	}

	records := make([]darwinGPURecord, 0, len(roots))
	for _, root := range roots {
		appendIORegistryGPUs(root, systemMemory, &records)
	}
	return records, nil
}

func appendIORegistryGPUs(entry ioRegistryEntry, systemMemory uint64, records *[]darwinGPURecord) {
	if entry.EntryID != 0 {
		name := strings.TrimSpace(entry.Model)
		if name == "" {
			name = strings.TrimSpace(entry.EntryName)
		}
		if name == "" {
			name = strings.TrimSpace(entry.ObjectClass)
		}
		if name != "" {
			*records = append(*records, normalizeIORegistryGPU(entry, name, systemMemory))
		}
	}
	for _, child := range entry.RegistryEntryChildren {
		appendIORegistryGPUs(child, systemMemory, records)
	}
}

func normalizeIORegistryGPU(entry ioRegistryEntry, name string, systemMemory uint64) darwinGPURecord {
	isApple := strings.HasPrefix(entry.ObjectClass, "AGXAccelerator") ||
		strings.HasPrefix(name, "Apple ")
	record := darwinGPURecord{
		statsKey: fmt.Sprintf("ioreg:%x", entry.EntryID),
		name:     name,
	}
	record.shared = strings.HasPrefix(entry.ObjectClass, "AGXAccelerator")
	if !record.shared {
		d := noderec.MemoryDomain{ID: record.statsKey, Kind: "unknown", DeviceIDs: []string{record.statsKey}, Total: noderec.MemoryFact{Reason: "unavailable"}, Free: noderec.MemoryFact{Reason: "unavailable"}}
		if entry.VRAMTotalMB > 0 && entry.VRAMTotalMB <= math.MaxUint64/(1024*1024) {
			d.Kind = "dedicated"
			total := entry.VRAMTotalMB * 1024 * 1024
			d.Total = memoryFact(total, "ioreg-VRAM,totalMB", time.Now().UTC())
			if f := entry.Performance.VRAMFreeBytes; f != nil && *f <= total {
				d.Free = memoryFact(*f, "ioreg-vramFreeBytes", time.Now().UTC())
			} else {
				u := entry.Performance.VRAMUsedBytes
				if u == nil {
					u = entry.Performance.InUseVidMemoryBytes
				}
				if u != nil && *u <= total {
					d.Free = memoryFact(total-*u, "ioreg-total-minus-dedicated-used", time.Now().UTC())
				}
			}
		}
		record.memory = &d
	}
	if isApple {
		record.vramTotal = systemMemory
		record.vramUsed = entry.Performance.AllocSystemMemory
	} else {
		record.vramTotal = entry.VRAMTotalMB * 1024 * 1024
		dedicatedUsed := entry.Performance.VRAMUsedBytes
		if dedicatedUsed == nil {
			dedicatedUsed = entry.Performance.InUseVidMemoryBytes
		}
		if dedicatedUsed != nil {
			record.vramUsed = *dedicatedUsed
		} else {
			record.vramUsed = entry.Performance.AllocSystemMemory
		}
		if record.vramTotal == 0 && dedicatedUsed != nil &&
			entry.Performance.VRAMFreeBytes != nil {
			record.vramTotal = *dedicatedUsed + *entry.Performance.VRAMFreeBytes
		}
	}
	var utilization uint64
	if entry.Performance.DeviceUtilization != nil {
		utilization = *entry.Performance.DeviceUtilization
		record.utilizationValid = true
	} else if entry.Performance.GPUActivity != nil {
		utilization = *entry.Performance.GPUActivity
		record.utilizationValid = true
	}
	if utilization > 100 {
		utilization = 100
	}
	record.utilizationPct = uint32(utilization)
	return record
}
