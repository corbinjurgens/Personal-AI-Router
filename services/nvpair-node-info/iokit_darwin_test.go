// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build darwin

package main

import (
	"context"
	"os"
	"os/exec"
	"reflect"
	"strings"
	"testing"
	"time"

	"nvpair-shared/secretenv"
)

func readIoregForTest(tb testing.TB) []byte {
	tb.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "/usr/sbin/ioreg", "-a", "-r", "-d", "1", "-c", "IOAccelerator")
	cmd.Env = secretenv.Remove(os.Environ())
	data, err := cmd.Output()
	if err != nil {
		tb.Fatalf("ioreg: %v", err)
	}
	return data
}

func TestIOAcceleratorEntriesMatchIoreg(t *testing.T) {
	data := readIoregForTest(t)
	wantEntries, err := decodeIORegistryEntries(data)
	if err != nil {
		t.Fatal(err)
	}
	entries, err := readIOAcceleratorEntries()
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) == 0 || len(entries) != len(wantEntries) {
		t.Fatalf("in-process entries = %d, ioreg entries = %d; want matching nonempty inventory", len(entries), len(wantEntries))
	}
	type identity struct {
		name, class, model string
		vramMB             uint64
	}
	identities := func(entries []ioRegistryEntry) map[uint64]identity {
		result := make(map[uint64]identity)
		for _, entry := range entries {
			result[entry.EntryID] = identity{entry.EntryName, entry.ObjectClass, entry.Model, entry.VRAMTotalMB}
		}
		return result
	}
	if got, want := identities(entries), identities(wantEntries); !reflect.DeepEqual(got, want) {
		t.Fatalf("entry identity = %+v, want %+v", got, want)
	}
	systemMemory := detectMemoryTotal()
	want, err := parseIORegistryGPUs(data, systemMemory)
	if err != nil {
		t.Fatal(err)
	}
	got, err := gpuRecordsFromEntries(entries, systemMemory)
	if err != nil {
		t.Fatal(err)
	}
	type staticRecord struct {
		name   string
		total  uint64
		shared bool
	}
	static := func(records []darwinGPURecord) map[string]staticRecord {
		result := make(map[string]staticRecord)
		for _, record := range records {
			result[record.statsKey] = staticRecord{record.name, record.vramTotal, record.shared}
		}
		return result
	}
	if got, want := static(got), static(want); !reflect.DeepEqual(got, want) {
		t.Fatalf("static GPU records = %+v, want %+v", got, want)
	}
}

func TestIOAcceleratorEntriesReadUtilization(t *testing.T) {
	entries, err := readIOAcceleratorEntries()
	if err != nil {
		t.Fatal(err)
	}
	records, err := gpuRecordsFromEntries(entries, detectMemoryTotal())
	if err != nil {
		t.Fatal(err)
	}
	appleEntries := 0
	for _, record := range records {
		if record.shared || strings.HasPrefix(record.name, "Apple ") {
			appleEntries++
			if record.utilizationPct == nil || *record.utilizationPct > 100 {
				t.Errorf("Apple GPU %s utilization = %v, want present value in [0,100]", record.name, record.utilizationPct)
			}
		}
	}
	if appleEntries == 0 {
		t.Fatal("no Apple IOAccelerator entry to validate on this Mac")
	}
}

func BenchmarkGPUReadIoreg(b *testing.B) {
	systemMemory := detectMemoryTotal()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := dynamicGPUReadingFromIORegistry(readIoregForTest(b), systemMemory); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGPUReadInProcess(b *testing.B) {
	systemMemory := detectMemoryTotal()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		entries, err := readIOAcceleratorEntries()
		if err != nil {
			b.Fatal(err)
		}
		if _, err := dynamicGPUReadingFromEntries(entries, systemMemory); err != nil {
			b.Fatal(err)
		}
	}
}
