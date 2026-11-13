// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
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

// nodeInfoPublisher turns the once-per-second local samples into the frames
// of GET /v1/node-info/events: a frame is published only when nodeInfoChanged
// says the sample moved beyond the thresholds of the last published frame.
type nodeInfoPublisher struct {
	hub  clustertrust.SnapshotHub[json.RawMessage]
	body func() []byte
}

// run publishes the first sample at once, then every nodeInfoPublishEvery
// publishes when nodeInfoChanged(last published, candidate) is true. It
// returns when ctx ends.
func (p *nodeInfoPublisher) run(ctx context.Context) {
	ticker := time.NewTicker(nodeInfoPublishEvery)
	defer ticker.Stop()
	var last NodeInfoResponse
	published := false
	for {
		body := p.body()
		var candidate NodeInfoResponse
		if json.Unmarshal(body, &candidate) == nil && (!published || nodeInfoChanged(last, candidate)) {
			p.hub.Publish(json.RawMessage(body))
			last, published = candidate, true
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// nodeInfoAdmitted is the access rule of both node-info routes. While this
// node is a cluster member the caller must present a pinned cluster cert (or
// this node's own leaf); otherwise every caller is served. Refresh picks up a
// membership change or a peer paired after startup.
func nodeInfoAdmitted(mesh *clustertrust.Mesh, r *http.Request) bool {
	mesh.Refresh()
	if !mesh.Clustered() {
		return true
	}
	_, ok := mesh.VerifyClientPin(r)
	return ok
}

// addressStreams counts open streams per non-loopback remote address, so one
// LAN host cannot take every slot of a route that is unauthenticated in
// production. Loopback callers (reverse-session streams relayed by the local
// cluster manager) count only against the hub's total.
type addressStreams struct {
	mu   sync.Mutex
	open map[string]int
}

// acquire reserves a slot for remoteAddr; release must be called when the
// stream ends. ok is false when the address already holds
// nodeInfoStreamsPerAddress streams.
func (a *addressStreams) acquire(remoteAddr string) (release func(), ok bool) {
	host, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		host = remoteAddr
	}
	if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() {
		return func() {}, true
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.open[host] >= nodeInfoStreamsPerAddress {
		return nil, false
	}
	a.open[host]++
	return func() {
		a.mu.Lock()
		defer a.mu.Unlock()
		if a.open[host]--; a.open[host] == 0 {
			delete(a.open, host)
		}
	}, true
}

// nodeInfoStreamHandler serves GET /v1/node-info/events: an NDJSON stream of
// GET /v1/node-info bodies, sent at once and on every threshold change, with a
// null keepalive every nodeInfoStreamKeepalive. The access rule is rechecked
// before every frame.
func nodeInfoStreamHandler(mesh *clustertrust.Mesh, pub *nodeInfoPublisher) http.HandlerFunc {
	perAddress := &addressStreams{open: make(map[string]int)}
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if !nodeInfoAdmitted(mesh, r) {
			http.Error(w, "forbidden: not a pinned cluster peer", http.StatusForbidden)
			return
		}
		release, ok := perAddress.acquire(r.RemoteAddr)
		if !ok {
			http.Error(w, "too many node-info streams", http.StatusServiceUnavailable)
			return
		}
		defer release()
		updates, cancel, ok := pub.hub.Subscribe()
		if !ok {
			http.Error(w, "too many node-info streams", http.StatusServiceUnavailable)
			return
		}
		defer cancel()
		w.Header().Set("Cache-Control", "no-store")
		clustertrust.ServeSnapshotStream(w, r, clustertrust.SnapshotStreamServer[json.RawMessage]{
			Keepalive:    nodeInfoStreamKeepalive,
			WriteTimeout: nodeInfoStreamWriteTimeout,
			Authorize:    func() bool { return nodeInfoAdmitted(mesh, r) },
			Updates:      updates,
		})
	}
}
