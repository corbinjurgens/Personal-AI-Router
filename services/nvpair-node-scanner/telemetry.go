// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"log/slog"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"nvpair-shared/noderec"
)

// telemetryRefreshInterval keeps utilization observations fresh enough for the
// scheduler while leaving node-info's one-second collector as the faster
// sampling layer. Slow requests remain in flight across ticks without delaying
// dispatch to healthy nodes.
const telemetryRefreshInterval = 2 * time.Second
const telemetryRefreshJitter = 250 * time.Millisecond
const telemetryRetryMax = 30 * time.Second

// telemetryRefreshConcurrency limits simultaneous node-info GETs across every
// telemetry tick.
const telemetryRefreshConcurrency = 16

type telemetryRetryState struct {
	consecutiveFailures int
	retryAt             time.Time
	retryTargetKey      string
	token               uint64
	inFlight            bool
	activeTargetKey     string
	discardActive       bool
}

// telemetryRetryGate is zero-value ready. Active-attempt ownership and retry
// timing are deliberately separate: a changed endpoint can discard another
// target's backoff, but it cannot release a request that has not finished.
// Removal clears retry timing and marks any active result stale while preserving
// the claim until its matching finish.
type telemetryRetryGate struct {
	mu        sync.Mutex
	nextToken uint64
	peers     map[string]telemetryRetryState
}

func (g *telemetryRetryGate) claim(hostUUID, targetKey string, now time.Time) (uint64, bool) {
	g.mu.Lock()
	defer g.mu.Unlock()

	state, exists := g.peers[hostUUID]
	if exists && state.inFlight {
		return 0, false
	}
	if exists && state.retryTargetKey == targetKey && now.Before(state.retryAt) {
		return 0, false
	}
	if state.retryTargetKey != targetKey {
		state.consecutiveFailures = 0
		state.retryAt = time.Time{}
		state.retryTargetKey = ""
	}
	g.nextToken++
	state.token = g.nextToken
	state.inFlight = true
	state.activeTargetKey = targetKey
	state.discardActive = false
	if g.peers == nil {
		g.peers = make(map[string]telemetryRetryState)
	}
	g.peers[hostUUID] = state
	return state.token, true
}

func (g *telemetryRetryGate) finish(hostUUID string, token uint64, ok bool, now time.Time) {
	g.mu.Lock()
	defer g.mu.Unlock()

	state, exists := g.peers[hostUUID]
	if !exists || !state.inFlight || state.token != token {
		return
	}
	state.inFlight = false
	if state.discardActive {
		delete(g.peers, hostUUID)
		return
	}
	if ok {
		delete(g.peers, hostUUID)
		return
	}
	if state.retryTargetKey != state.activeTargetKey {
		state.consecutiveFailures = 0
	}
	state.consecutiveFailures++
	state.retryTargetKey = state.activeTargetKey
	state.retryAt = now.Add(telemetryRetryDelay(state.consecutiveFailures))
	state.activeTargetKey = ""
	g.peers[hostUUID] = state
}

func (g *telemetryRetryGate) forget(hostUUID string) {
	g.mu.Lock()
	defer g.mu.Unlock()

	state, exists := g.peers[hostUUID]
	if !exists {
		return
	}
	if !state.inFlight {
		delete(g.peers, hostUUID)
		return
	}
	state.consecutiveFailures = 0
	state.retryAt = time.Time{}
	state.retryTargetKey = ""
	state.discardActive = true
	g.peers[hostUUID] = state
}

func telemetryTargetKey(hosts []string, port int) string {
	return strconv.Itoa(port) + "\x00" + strings.Join(hosts, "\x00")
}

func telemetryRetryDelay(consecutiveFailures int) time.Duration {
	delay := telemetryRefreshInterval
	for failure := 0; failure < consecutiveFailures; failure++ {
		if delay >= telemetryRetryMax/2 {
			return telemetryRetryMax
		}
		delay *= 2
	}
	return delay
}

func (d *daemon) refreshTelemetryLoop(ctx context.Context) {
	hostUUID := ""
	if d.reg != nil {
		hostUUID = d.reg.record().HostUUID
	}
	d.runTelemetryLoop(ctx, telemetryIntervalForNode(hostUUID))
}

func telemetryIntervalForNode(hostUUID string) time.Duration {
	if hostUUID == "" {
		return telemetryRefreshInterval
	}
	hash := uint32(2166136261)
	for i := 0; i < len(hostUUID); i++ {
		hash ^= uint32(hostUUID[i])
		hash *= 16777619
	}
	windowMillis := int64(2*telemetryRefreshJitter/time.Millisecond) + 1
	offsetMillis := int64(hash%uint32(windowMillis)) - windowMillis/2
	return telemetryRefreshInterval + time.Duration(offsetMillis)*time.Millisecond
}

// runTelemetryLoop dispatches compact scheduler telemetry on a fixed cadence.
// Requests may span ticks, but telemetryRetries excludes another request for
// the same node and the loop-owned semaphore bounds total active work. Shutdown
// cancels and drains every worker before returning.
func (d *daemon) runTelemetryLoop(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	sem := make(chan struct{}, telemetryRefreshConcurrency)
	var workers sync.WaitGroup
	defer workers.Wait()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			d.dispatchTelemetry(ctx, time.Now(), sem, &workers)
		}
	}
}

// refreshTelemetryOnce synchronously dispatches one telemetry pass. Production
// uses runTelemetryLoop's persistent semaphore and workers; this wrapper keeps
// focused callers and tests able to wait for the pass they requested.
func (d *daemon) refreshTelemetryOnce(ctx context.Context) {
	sem := make(chan struct{}, telemetryRefreshConcurrency)
	var workers sync.WaitGroup
	d.dispatchTelemetry(ctx, time.Now(), sem, &workers)
	workers.Wait()
}

// dispatchTelemetry starts every due node without waiting for requests already
// in flight. sem and workers belong to the enclosing loop so work remains
// bounded and drainable across ticks.
func (d *daemon) dispatchTelemetry(
	ctx context.Context,
	now time.Time,
	sem chan struct{},
	workers *sync.WaitGroup,
) {
	self := ""
	if d.reg != nil {
		self = d.reg.record().HostUUID
	}
	for _, n := range d.dir.snapshot(noderec.ServiceNodeInfo) {
		ni, ok := n.Services[noderec.ServiceNodeInfo]
		if !ok {
			continue
		}
		hosts := append([]string(nil), n.CandidateIPs()...)
		if n.HostUUID == self {
			hosts = []string{loopbackHost}
		}
		if len(hosts) == 0 {
			continue
		}
		targetKey := telemetryTargetKey(hosts, ni.Port)
		retryToken, due := d.telemetryRetries.claim(n.HostUUID, targetKey, now)
		if !due {
			continue
		}

		workers.Add(1)
		go func(hostUUID string, hosts []string, port int, token uint64, backoff bool) {
			defer workers.Done()
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				d.telemetryRetries.finish(hostUUID, token, true, time.Now())
				return
			}
			ok := d.refreshNodeTelemetryCandidates(ctx, hostUUID, hosts, port)
			if !backoff || ctx.Err() != nil {
				ok = true
			}
			d.telemetryRetries.finish(hostUUID, token, ok, time.Now())
		}(n.HostUUID, hosts, ni.Port, retryToken, n.HostUUID != self)
	}
}

const maxTelemetryAge = int64(1<<63 - 1)

// refreshNodeTelemetry emits one node's current GPU utilization. Failed fetches
// stay silent so downstream freshness expiry handles temporary unreachability.
// A response naming another host is never attributed to the directory entry
// whose address was queried.
func (d *daemon) refreshNodeTelemetry(ctx context.Context, hostUUID, ip string, niPort int) bool {
	return d.refreshNodeTelemetryCandidates(ctx, hostUUID, []string{ip}, niPort)
}

// telemetrySample is one node-info answer plus when the request that produced it
// began. The reported age has to be advanced by that request's own latency, and a
// failover attempt asks several addresses at once, so the time belongs to the
// answer rather than to the sweep.
type telemetrySample struct {
	boot      uint64
	clockErr  error
	info      NodeInfoResponse
	startedAt time.Time
}

// refreshNodeTelemetryCandidates is refreshNodeTelemetry with address failover.
//
// Healthy nodes are sampled every two seconds. Failed remote nodes back off, and
// each due attempt goes through the same node-info memory the enrichment sweeps
// use: the address that answered is asked by itself, and the rest of the list
// only once it stops answering. Without that, a peer whose canonical address is
// a link this host cannot reach reported no telemetry at all, and the scheduler
// then assigned it neutral pressure and scheduled it blind.
func (d *daemon) refreshNodeTelemetryCandidates(ctx context.Context, hostUUID string, hosts []string, niPort int) bool {
	var generation uint64
	var buildTargetCurrent bool
	if d.dir != nil {
		self := d.reg != nil && hostUUID == d.reg.record().HostUUID && slices.Equal(hosts, []string{loopbackHost})
		d.dir.mu.RLock()
		node, exists := d.dir.nodes[hostUUID]
		generation = d.dir.generations[hostUUID]
		ni, advertised := node.Services[noderec.ServiceNodeInfo]
		buildTargetCurrent = exists && advertised && ni.Port == niPort && (self || slices.Equal(hosts, node.CandidateIPs()))
		d.dir.mu.RUnlock()
	}
	ask := func(host string) (telemetrySample, bool) {
		boot, clockErr := telemetryClock()
		startedAt := time.Now()
		candidate, ok := d.fetchPeerNodeInfoWithin(ctx, hostUUID, "", host, niPort)
		if !ok {
			return telemetrySample{}, false
		}
		if candidate.HostUUID != hostUUID {
			slog.Debug("skipping telemetry reported by a different host at this address; trying next",
				"expected_host_uuid", hostUUID, "reported_host_uuid", candidate.HostUUID, "ip", host)
			return telemetrySample{}, false
		}
		return telemetrySample{info: candidate, startedAt: startedAt, boot: boot, clockErr: clockErr}, true
	}
	key := hostKey{hostUUID: hostUUID, service: noderec.ServiceNodeInfo}
	_, sample, ok := askRemembered(&d.enrichHosts, key, hosts, ask)
	if !ok {
		return false
	}
	info, startedAt := sample.info, sample.startedAt
	memoryCurrent := false
	if buildTargetCurrent {
		d.peerServicesMu.Lock()
		// Claim replacement/retirement shares peerServicesMu with this apply.
		// Recheck admission after the fetch, before any directory/cache/event write.
		d.dir.mu.RLock()
		claim, authenticated := d.dir.peerServices[hostUUID]
		d.dir.mu.RUnlock()
		authorityCurrent := true
		if authenticated {
			incarnation, admitted := d.mesh.PeerServiceIncarnation(hostUUID)
			authorityCurrent = admitted && incarnation == claim.Incarnation && peerServicesFresh(claim, time.Now())
		}
		if authorityCurrent {
			node, changed, current := d.dir.applyServiceBuild(hostUUID, generation, info.ServiceBuildID)
			if current {
				memoryCurrent = true
				d.infoMu.Lock()
				if cached, present := d.lastInfo[hostUUID]; present {
					cached.ServiceBuildID = info.ServiceBuildID
					d.lastInfo[hostUUID] = cached
				}
				d.infoMu.Unlock()
				if changed {
					d.emit(noderec.NotifyNodeUpdated, node)
				}
			}
		}
		d.peerServicesMu.Unlock()
	}

	age := int64(0)
	if info.TelemetryValid {
		age = info.MSSince
		if age < 0 {
			age = 0
		}
		elapsed := time.Since(startedAt).Milliseconds()
		if elapsed > maxTelemetryAge-age {
			age = maxTelemetryAge
		} else {
			age += elapsed
		}
	}
	var utilization uint32
	for i := range info.GPUs {
		if info.GPUs[i].UtilizationPercent > utilization {
			utilization = info.GPUs[i].UtilizationPercent
		}
	}
	hop := noderec.NewMemoryHop(strconv.Itoa(os.Getpid()), telemetryClock)
	if d.dir != nil {
		d.peerServicesMu.Lock()
		defer d.peerServicesMu.Unlock()
		d.dir.mu.RLock()
		claim, authenticated := d.dir.peerServices[hostUUID]
		d.dir.mu.RUnlock()
		if authenticated {
			incarnation, admitted := d.mesh.PeerServiceIncarnation(hostUUID)
			memoryCurrent = memoryCurrent && admitted && incarnation == claim.Incarnation && peerServicesFresh(claim, time.Now())
		}
		d.dir.mu.RLock()
		defer d.dir.mu.RUnlock()
		node, exists := d.dir.nodes[hostUUID]
		ni, advertised := node.Services[noderec.ServiceNodeInfo]
		memoryCurrent = memoryCurrent && exists && advertised && d.dir.generations[hostUUID] == generation && ni.Port == niPort
	}
	d.emitTelemetry(noderec.NodeTelemetry{
		Memory: func() *noderec.MemoryObservation {
			if !memoryCurrent {
				return nil
			}
			return info.MemoryObservation.Advance(sample.boot, sample.clockErr, telemetryClock)
		}(),
		MemoryHop:         hop,
		HostUUID:          hostUUID,
		GPUUtilizationPct: utilization,
		TelemetryValid:    info.TelemetryValid,
		MSSince:           age,
	})
	return true
}

var telemetryClock = noderec.MemoryBootClock
