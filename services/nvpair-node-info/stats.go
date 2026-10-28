// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/rand"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"nvpair-shared/noderec"
)

// gpuStat is the per-adapter dynamic state the service layers on top of the
// static detectGPUs() output. Both fields are zero when the underlying OS
// counter is unavailable; upstream code maps zeros back to "omit from JSON"
// via the omitempty tags on GPUInfo, so clients see a missing field rather
// than a misleading literal zero. The residual ambiguity for UtilizationPct
// (0 could mean "GPU is actually idle" or "we couldn't read") is benign
// because the two cases are visually identical to a user.
type gpuStat struct {
	memoryUsedValid bool
	memoryUsed      uint64
	Memory          *noderec.MemoryDomain
	MemoryBoot      uint64
	Shared          bool
	VRAMUsed        uint64
	UtilizationPct  uint32
}

func responseMemory(snap statsSnapshot) *noderec.MemoryObservation {
	m := memoryAt(snap.Memory, snap.MemoryBoot)
	if m == nil {
		m = &noderec.MemoryObservation{RunID: memoryRunID, Domains: []noderec.MemoryDomain{{ID: "system", Kind: "system", Total: noderec.MemoryFact{Reason: "unavailable"}, Free: noderec.MemoryFact{Reason: "unavailable"}}}}
	}
	keys := make([]string, 0, len(snap.GPU))
	for id := range snap.GPU {
		keys = append(keys, id)
	}
	sort.Strings(keys)
	for _, id := range keys {
		gpu := snap.GPU[id]
		if gpu.Shared {
			m.Domains[0].Kind = "shared"
			m.Domains[0].DeviceIDs = append(m.Domains[0].DeviceIDs, id)
		} else if gpu.Memory != nil {
			one := memoryAt(&noderec.MemoryObservation{RunID: memoryRunID, Domains: []noderec.MemoryDomain{*gpu.Memory}}, gpu.MemoryBoot)
			m.Domains = append(m.Domains, one.Domains...)
		}
	}
	if m.Validate() != nil {
		return nil
	}
	return m
}

// statsSnapshot is the dynamic bundle the HTTP handler reads without locking.
// Darwin combines independently published system and GPU samples so a slow
// ioreg call cannot delay CPU or memory telemetry.
// GPUInventory is normally nil. The Darwin collector populates it from its
// retrying ioreg sample so a transient startup enumeration failure can recover
// without restarting the service.
//
// Unsupported collectors publish a zero-valued snapshot — GPU is a nil map
// (omitempty semantics for downstream lookups come from the per-GPU omitempty
// tags, not from an empty-map check here), CPUUtilPct is 0, MemUsedBytes is 0.
// buildResponse treats each subsystem's zero-value as "unknown" and drops the
// corresponding field per the normal omitempty rules.
type statsSnapshot struct {
	Memory       *noderec.MemoryObservation
	MemoryBoot   uint64
	GPU          map[string]gpuStat
	GPUInventory []GPUInfo
	// GPUSampledAt is the collection time of the latest usable GPU
	// utilization sample. A zero value means no usable sample has ever been
	// collected. Failed collection attempts retain the previous timestamp so
	// consumers can distinguish stale telemetry from freshly sampled data.
	GPUSampledAt time.Time
	CPUUtilPct   uint32
	MemUsedBytes uint64
}

var memoryRunID = func() string {
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		return ""
	}
	return fmt.Sprintf("%x", id)
}()
var memoryClock = noderec.MemoryBootClock

func memoryFact(value uint64, source string, at time.Time) noderec.MemoryFact {
	return noderec.MemoryFact{Bytes: strconv.FormatUint(value, 10), Source: source, SampledAt: at.UTC(), Valid: true}
}

func systemMemory(total, available uint64, source string, at time.Time) *noderec.MemoryObservation {
	if available > total {
		return nil
	}
	return &noderec.MemoryObservation{RunID: memoryRunID, Domains: []noderec.MemoryDomain{{ID: "system", Kind: "system", Total: memoryFact(total, source, at), Free: memoryFact(available, source, at)}}}
}

func memoryAt(m *noderec.MemoryObservation, start uint64) *noderec.MemoryObservation {
	out := m.Clone()
	if out == nil {
		return nil
	}
	end, err := memoryClock()
	if err != nil {
		out.AddAge(-1)
		return out
	}
	ms, err := noderec.ElapsedMemoryMS(start, end)
	if err != nil {
		ms = -1
	}
	out.AddAge(ms)
	return out
}

func sampleSystemMemory(previous statsSnapshot, read func() (uint64, uint64, string, bool)) *statsSnapshot {
	start, err := memoryClock()
	at := time.Now().UTC()
	total, available, source, ok := read()
	snap := &statsSnapshot{Memory: previous.Memory, MemoryBoot: previous.MemoryBoot}
	if ok && available <= total {
		snap.MemUsedBytes = total - available
		if err == nil {
			snap.Memory = systemMemory(total, available, source, at)
			snap.MemoryBoot = start
		}
		if err != nil {
			snap.Memory = previous.Memory.Clone()
			snap.Memory.AddAge(-1)
		}
	}
	if !ok && total > 0 && err == nil {
		snap.Memory = systemMemory(total, 0, source, at)
		snap.Memory.Domains[0].Free = noderec.MemoryFact{Reason: "available-unavailable"}
		snap.MemoryBoot = start
	}
	return snap
}

// applyGPUStats publishes a usable GPU sample or preserves the last usable
// sample after a failed collection. Before the first usable sample, callers may
// still publish partial dynamic fields (for example VRAM usage) with a zero
// sampledAt; those fields remain display-only until utilization becomes valid.
func applyGPUStats(previous statsSnapshot, next *statsSnapshot, sampled map[string]gpuStat, sampledAt time.Time) {
	if sampledAt.IsZero() && !previous.GPUSampledAt.IsZero() {
		next.GPU = make(map[string]gpuStat, len(previous.GPU))
		for key, stat := range previous.GPU {
			next.GPU[key] = stat
		}
		for key, stat := range sampled {
			old := next.GPU[key]
			old.memoryUsedValid = stat.memoryUsedValid
			old.memoryUsed = stat.memoryUsed
			old.Memory = stat.Memory
			old.MemoryBoot = stat.MemoryBoot
			old.Shared = stat.Shared
			next.GPU[key] = old
		}
		next.GPUSampledAt = previous.GPUSampledAt
		return
	}
	next.GPU = sampled
	next.GPUSampledAt = sampledAt
}

// applyDedicatedMemoryFacts joins freshly enumerated DXGI adapters only with
// the current PDH sample. next.GPU may retain legacy utilization/usage display
// values and must never supply a freshly timestamped available-memory fact.
func applyDedicatedMemoryFacts(next *statsSnapshot, adapters []GPUInfo, sampled map[string]gpuStat, at time.Time, boot uint64) {
	counts := map[string]int{}
	for _, adapter := range adapters {
		counts[adapter.statsKey]++
	}
	for _, adapter := range adapters {
		stat := next.GPU[adapter.statsKey]
		d := noderec.MemoryDomain{ID: adapter.statsKey, Kind: "unknown", DeviceIDs: []string{adapter.statsKey}, Total: noderec.MemoryFact{Reason: "unavailable"}, Free: noderec.MemoryFact{Reason: "unavailable"}}
		if adapter.VramBytes > 0 && counts[adapter.statsKey] == 1 {
			d.Kind = "dedicated"
			d.Total = memoryFact(adapter.VramBytes, "dxgi-GetDesc1", at)
			current, present := sampled[adapter.statsKey]
			if present && current.memoryUsedValid && current.memoryUsed <= adapter.VramBytes {
				d.Free = memoryFact(adapter.VramBytes-current.memoryUsed, "dxgi-total-minus-pdh-dedicated-usage", at)
			}
		}
		stat.Memory = &d
		stat.MemoryBoot = boot
		next.GPU[adapter.statsKey] = stat
	}
}

// parseEngineInstance pulls the adapter-LUID key and the engine-type tag
// out of a PDH "GPU Engine" counter instance name, which looks like:
//
//	pid_1234_luid_0x00000000_0x000054f0_phys_0_eng_0_engtype_3D
//
// We need the LUID portion to join against luidKey() (which is how the
// DXGI-enumerated adapters identify themselves) and the engine-type to
// bucket utilization before the across-engine max. The returned luidKey
// is lowercase to match the normalization the VRAM counter reader does
// on its side, so both counter streams key into the same adapter map.
//
// Lives in a platform-neutral file (no build tag) so it's unit-testable
// on any host — the string format is stable across Windows versions and
// doesn't touch the PDH API itself.
func parseEngineInstance(s string) (luidKey, engtype string, ok bool) {
	s = strings.ToLower(s)
	luidIdx := strings.Index(s, "luid_")
	if luidIdx < 0 {
		return "", "", false
	}
	rest := s[luidIdx:]
	engIdx := strings.Index(rest, "_eng_")
	if engIdx < 0 {
		return "", "", false
	}
	luidKey = rest[:engIdx]
	const typeMarker = "_engtype_"
	typeIdx := strings.Index(rest, typeMarker)
	if typeIdx < 0 {
		return "", "", false
	}
	engtype = rest[typeIdx+len(typeMarker):]
	if engtype == "" {
		return "", "", false
	}
	return luidKey, engtype, true
}

// aggregateUtilization implements Task Manager's "overall GPU %" calc:
//
//  1. Sum per-process percentages within each (luid, engine-type) bucket.
//     Two apps each running the 3D engine at 30 % really is 60 % engine
//     load; the PDH counter is per-process-time, so summing is correct.
//  2. Clamp each bucket to 100. Sampling jitter can occasionally push a
//     bucket slightly over, and we don't want to surface >100 % to users.
//  3. Take the max across engine types for each luid. The number users
//     recognize as "the GPU is busy" is the busiest engine on that adapter,
//     not the average — a compute-bound ML job with an idle 3D engine is
//     still a fully-busy GPU.
//
// Result is rounded to whole percent. Sub-percent precision is below the
// visual resolution of any reasonable UI and would only generate noisy
// node/updated events from the change-detection plumbing downstream.
func aggregateUtilization(items map[string]float64) map[string]uint32 {
	byEngine := map[string]map[string]float64{}
	for instance, v := range items {
		luid, engtype, ok := parseEngineInstance(instance)
		if !ok || v < 0 {
			continue
		}
		m := byEngine[luid]
		if m == nil {
			m = map[string]float64{}
			byEngine[luid] = m
		}
		m[engtype] += v
	}
	out := make(map[string]uint32, len(byEngine))
	for luid, m := range byEngine {
		var maxPct float64
		for _, p := range m {
			if p > 100 {
				p = 100
			}
			if p > maxPct {
				maxPct = p
			}
		}
		out[luid] = uint32(math.Round(maxPct))
	}
	return out
}
