// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"log/slog"
	"sort"
	"strconv"
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
			sources.scanner.value.MemoryRequest = nil
		}
	}
}

// retireScannerHostMemory leaves scheduler utilization and its receipt intact.
func (c *telemetryCache) retireScannerHostMemory(hostUUID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if observation := c.nodes[hostUUID].scanner; observation != nil {
		observation.value.Memory = nil
		observation.value.MemoryHop = nil
		observation.value.MemoryRequest = nil
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
	observation.value.MemoryRequest = nil
	if source == sourceScanner && value.MemoryRequest != nil && value.MemoryRequest.Validate() == nil {
		context := *value.MemoryRequest
		observation.value.MemoryRequest = &context
	}
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
		if context := observation.value.MemoryRequest; context != nil {
			start, parseErr := strconv.ParseUint(context.RequestBootNS, 10, 64)
			if parseErr != nil || err != nil || start > boot {
				observation.value.MemoryRequest = nil
			}
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	sources := c.nodes[value.HostUUID]
	if previous := sources.source(source); previous != nil && previous.value.MemoryRequest != nil && value.MemoryRequest != nil {
		old, next := previous.value.MemoryRequest, value.MemoryRequest
		oldStart, oldErr := strconv.ParseUint(old.RequestBootNS, 10, 64)
		nextStart, nextErr := strconv.ParseUint(next.RequestBootNS, 10, 64)
		if old.Generation == next.Generation && oldErr == nil && nextErr == nil && nextStart <= oldStart {
			return noderec.NodeTelemetry{}, false
		}
	}
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
	value.MemoryRequest = nil
	return value
}

// memoryView is detached advice, with target authority granted only by the
// broker's local lookup after checking its current scanner and own identity.
type memoryView struct {
	HostUUID      string
	Source        nodeSource
	ProducerRunID string
	Observation   *noderec.MemoryObservation
	Reason        string
	TargetLocal   bool
}

func (c *telemetryCache) LookupMemory(hostUUID string, source nodeSource, producerRunID string, now time.Time) memoryView {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.lookupMemoryLocked(hostUUID, source, producerRunID, now)
}

func (c *telemetryCache) lookupMemoryLocked(hostUUID string, source nodeSource, producerRunID string, now time.Time) memoryView {
	view := memoryView{HostUUID: hostUUID, Source: source, ProducerRunID: producerRunID, Reason: "missing"}
	sources, present := c.nodes[hostUUID]
	observation, ok := sources.projected()
	if !present || !ok {
		return view
	}
	if sources.source(source) != observation {
		view.Reason = "source-mismatch"
		return view
	}
	if observation.value.Memory == nil {
		view.Reason = "invalid"
		return view
	}
	if observation.value.Memory.RunID != producerRunID {
		view.Reason = "producer-mismatch"
		return view
	}
	view.Observation = c.observedAt(*observation, now).Memory
	view.Reason = ""
	if view.Observation == nil || len(view.Observation.Domains) == 0 {
		view.Reason = "invalid"
		return view
	}
	for _, domain := range view.Observation.Domains {
		for _, fact := range []noderec.MemoryFact{domain.Total, domain.Free} {
			if !fact.Valid {
				if fact.Reason != "stale" {
					view.Reason = "invalid"
					return view
				}
				view.Reason = "stale"
			}
		}
	}
	return view
}

func (b *Broker) lookupLocalMemory(hostUUID, producerRunID string, now time.Time) memoryView {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	refused := memoryView{HostUUID: hostUUID, Source: sourceScanner, ProducerRunID: producerRunID, Reason: "unverified-local-source"}
	if hostUUID == "" || hostUUID != b.nodeID || b.scannerSource == nil || b.telemetry == nil {
		return refused
	}
	c := b.telemetry
	c.mu.RLock()
	defer c.mu.RUnlock()
	observation := c.nodes[hostUUID].scanner
	if observation == nil || observation.value.MemoryRequest == nil || !observation.value.MemoryRequest.VerifiedSelfLoopback {
		return refused
	}
	view := c.lookupMemoryLocked(hostUUID, sourceScanner, producerRunID, now)
	// UTC can withdraw self freshness, never grant it or reduce carried age.
	if view.Observation != nil {
		for i := range view.Observation.Domains {
			for _, f := range []*noderec.MemoryFact{&view.Observation.Domains[i].Total, &view.Observation.Domains[i].Free} {
				if !f.Valid {
					continue
				}
				if now.Before(f.SampledAt) || now.Before(observation.receivedAt) {
					*f = noderec.MemoryFact{Reason: "invalid-clock"}
					view.Reason = "invalid"
					continue
				}
				if now.Sub(f.SampledAt) > time.Duration(noderec.MemoryFreshMS)*time.Millisecond {
					*f = noderec.MemoryFact{Reason: "stale"}
					if view.Reason == "" {
						view.Reason = "stale"
					}
				} else {
					wallAge := now.Sub(f.SampledAt)
					ms := int64(wallAge / time.Millisecond)
					if wallAge%time.Millisecond != 0 {
						ms++
					}
					if ms > f.AgeMS {
						f.AgeMS = ms
					}
				}
			}
		}
	}
	view.TargetLocal = view.Reason == ""
	return view
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
