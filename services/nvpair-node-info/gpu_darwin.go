// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build darwin

package main

import "log"

func detectGPUs() []GPUInfo {
	entries, err := readIOAcceleratorEntries()
	if err != nil {
		log.Printf("GPU detection error: %v", err)
		return nil
	}
	gpus, err := staticGPUsFromEntries(entries, detectMemoryTotal())
	if err != nil {
		log.Printf("GPU detection error: %v", err)
		return nil
	}
	return gpus
}

func staticGPUsFromIORegistry(data []byte, systemMemory uint64) ([]GPUInfo, error) {
	entries, err := decodeIORegistryEntries(data)
	if err != nil {
		return nil, err
	}
	return staticGPUsFromEntries(entries, systemMemory)
}

func staticGPUsFromEntries(entries []ioRegistryEntry, systemMemory uint64) ([]GPUInfo, error) {
	records, err := gpuRecordsFromEntries(entries, systemMemory)
	if err != nil {
		return nil, err
	}
	gpus := make([]GPUInfo, 0, len(records))
	for _, record := range records {
		gpus = append(gpus, GPUInfo{
			Name:      record.name,
			VramBytes: record.vramTotal,
			statsKey:  record.statsKey,
		})
	}
	return gpus, nil
}
