// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"log/slog"
	"sort"
	"sync"
	"time"

	"nvpair-shared/noderec"
	"nvpair-shared/schedulerwire"
)

const maxTelemetryMilliseconds = int64(1<<63 - 1)

type telemetryObservation struct {
	memoryBoot       uint64
	memoryClockValid bool
	value            noderec.NodeTelemetry
	receivedAt       time.Time
}

type telemetrySources struct {
	scanner *telemetryObservation
	manual  *telemetryObservation
}

func (s *telemetrySources) setSource(source nodeSource, observation *telemetryObservation) {
	switch source {
	case sourceScanner:
		s.scanner = observation
	case sourceManual:
		s.manual = observation
	}
}

func (s telemetrySources) source(source nodeSource) *telemetryObservation {
	switch source {
	case sourceScanner:
		return s.scanner
	case sourceManual:
		return s.manual
	default:
		return nil
	}
}

func (s telemetrySources) projected() (*telemetryObservation, bool) {
	switch {
	case s.scanner == nil && s.manual == nil:
		return nil, false
	case s.scanner != nil:
		return s.scanner, true
	default:
		return s.manual, true
	}
}

type telemetryCache struct {
	clock func() (uint64, error)
	mu    sync.RWMutex
	nodes map[string]telemetrySources
}

func newTelemetryCache() *telemetryCache {
	return &telemetryCache{nodes: make(map[string]telemetrySources), clock: noderec.MemoryBootClock}
}

func (c *telemetryCache) retireScannerMemory() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, sources := range c.nodes {
		if sources.scanner != nil {
			sources.scanner.value.Memory = nil
			sources.scanner.value.MemoryHop = nil
		}
	}
}

// Upsert records one source's latest observation and returns the source-aware
// projection to forward. Scanner is authoritative while present; manual
// telemetry is the fallback for nodes without a scanner claim.
func (c *telemetryCache) Upsert(source nodeSource, value noderec.NodeTelemetry, receivedAt time.Time) (noderec.NodeTelemetry, bool) {
	if value.HostUUID == "" {
		return noderec.NodeTelemetry{}, false
	}
	observation := telemetryObservation{
		value:      value,
		receivedAt: receivedAt,
	}
	observation.value.Memory = value.Memory.Clone()
	observation.value.MemoryHop = nil
	if value.Memory != nil {
		boot, err := c.clock()
		ms := int64(-1)
		if err == nil && value.MemoryHop != nil {
			if elapsed, hopErr := value.MemoryHop.ElapsedAt(boot); hopErr == nil {
				ms = elapsed
			}
		}
		if observation.value.Memory.Validate() != nil {
			observation.value.Memory = nil
		} else {
			observation.value.Memory.AddAge(ms)
		}
		observation.memoryBoot = boot
		observation.memoryClockValid = err == nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	sources := c.nodes[value.HostUUID]
	sources.setSource(source, &observation)
	c.nodes[value.HostUUID] = sources
	projected, ok := sources.projected()
	if !ok {
		return noderec.NodeTelemetry{}, false
	}
	return c.observedAt(*projected, receivedAt), true
}

// Remove drops one source. If another source survives, its current projection
// is returned; otherwise an invalid observation is returned so the scheduler can
// immediately clear the departed node instead of waiting for freshness expiry.
func (c *telemetryCache) Remove(hostUUID string, source nodeSource, now time.Time) (noderec.NodeTelemetry, bool) {
	if hostUUID == "" {
		return noderec.NodeTelemetry{}, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	sources, ok := c.nodes[hostUUID]
	if !ok || sources.source(source) == nil {
		return noderec.NodeTelemetry{}, false
	}
	sources.setSource(source, nil)
	projected, survives := sources.projected()
	if survives {
		c.nodes[hostUUID] = sources
	} else {
		delete(c.nodes, hostUUID)
	}
	if !survives {
		return noderec.NodeTelemetry{HostUUID: hostUUID}, true
	}
	return c.observedAt(*projected, now), true
}

func (c *telemetryCache) Snapshot(now time.Time) []noderec.NodeTelemetry {
	c.mu.RLock()
	out := make([]noderec.NodeTelemetry, 0, len(c.nodes))
	for _, sources := range c.nodes {
		if projected, ok := sources.projected(); ok {
			out = append(out, c.observedAt(*projected, now))
		}
	}
	c.mu.RUnlock()
	sort.Slice(out, func(i, j int) bool { return out[i].HostUUID < out[j].HostUUID })
	return out
}

func (c *telemetryCache) observedAt(observation telemetryObservation, now time.Time) noderec.NodeTelemetry {
	value := observedTelemetryAt(observation, now)
	if !observation.memoryClockValid {
		value.Memory = value.Memory.Clone()
		value.Memory.AddAge(-1)
	} else {
		value.Memory = value.Memory.Advance(observation.memoryBoot, nil, c.clock)
	}
	value.Memory.Expire()
	value.MemoryHop = nil
	return value
}

func observedTelemetryAt(observation telemetryObservation, now time.Time) noderec.NodeTelemetry {
	value := observation.value
	if !value.TelemetryValid {
		value.MSSince = 0
		return value
	}
	if value.MSSince < 0 {
		value.MSSince = 0
	}
	elapsed := now.Sub(observation.receivedAt).Milliseconds()
	if elapsed <= 0 {
		return value
	}
	if elapsed > maxTelemetryMilliseconds-value.MSSince {
		value.MSSince = maxTelemetryMilliseconds
	} else {
		value.MSSince += elapsed
	}
	return value
}

func (b *Broker) ingestTelemetry(source nodeSource, value noderec.NodeTelemetry) {
	b.ingestTelemetryAt(source, value, time.Now())
}

func (b *Broker) ingestTelemetryAt(source nodeSource, value noderec.NodeTelemetry, receivedAt time.Time) {
	if b.telemetry == nil {
		return
	}
	projected, ok := b.telemetry.Upsert(source, value, receivedAt)
	if ok {
		b.fanTelemetryToScheduler(projected)
	}
}

func (b *Broker) removeTelemetry(source nodeSource, hostUUID string) {
	if b.telemetry == nil {
		return
	}
	projected, ok := b.telemetry.Remove(hostUUID, source, time.Now())
	if ok {
		b.fanTelemetryToScheduler(projected)
	}
}

func (b *Broker) fanTelemetryToScheduler(telemetry noderec.NodeTelemetry) {
	b.schedulerFeedMu.Lock()
	defer b.schedulerFeedMu.Unlock()

	scheduler := b.getScheduler()
	if scheduler == nil {
		return
	}
	if err := scheduler.Notify(schedulerwire.MethodTelemetry, telemetry); err != nil {
		slog.Warn("fan telemetry to scheduler failed", "host_uuid", telemetry.HostUUID, "err", err)
	}
}

// replayTelemetryToScheduler seeds a restarted scheduler from the cache. The
// caller holds schedulerFeedMu, so the replay is an ordered prefix of live
// scanner and manual observations.
func (b *Broker) replayTelemetryToScheduler(scheduler *rpcWorker) int {
	if b.telemetry == nil {
		return 0
	}
	replayed := 0
	for _, telemetry := range b.telemetry.Snapshot(time.Now()) {
		if err := scheduler.Notify(schedulerwire.MethodTelemetry, telemetry); err != nil {
			slog.Warn("replay telemetry to scheduler failed", "host_uuid", telemetry.HostUUID, "err", err)
			break
		}
		replayed++
	}
	return replayed
}
