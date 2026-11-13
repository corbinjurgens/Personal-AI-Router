// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
)

// Pinned peers serve their node-info as a snapshot stream (node-info's
// GET /v1/node-info/events): the full body on connect and on threshold changes,
// and a null keepalive every 2 s while the current sample stays within
// thresholds of the last frame. The scanner holds one stream per stream peer
// instead of pulling GET /v1/node-info every telemetry tick.
var (
	// nodeInfoStreamIdle drops a stream after three missed keepalives.
	nodeInfoStreamIdle = 6 * time.Second
	// peerStreamConnectTimeout bounds connect and response headers of every
	// peer stream. It is a cancel timer, not a context deadline: a reverse dial
	// copies a deadline onto the yamux stream, which would end the open stream.
	peerStreamConnectTimeout = 10 * time.Second
)

const (
	nodeInfoStreamPath    = "/v1/node-info/events"
	nodeInfoStreamMaxLine = 256 << 10
	// peerStreamLimit caps the streams of one kind; it matches the producers'
	// own total stream caps.
	peerStreamLimit = 64
	// peerStreamHealthy is how long a stream that delivered a frame must stay
	// open for its reconnect backoff to reset.
	peerStreamHealthy = 5 * time.Second
	// nodeInfoStaleSampleAge mirrors node-info's staleGPUSampleAge: once a
	// valid GPU sample is this old, node-info publishes a frame on every msSince
	// change, so a null line then no longer means "fresh".
	nodeInfoStaleSampleAge = 5 * time.Second
)

// nodeInfoStreamState holds the supervised streams of every kind (node-info in
// active, the model inventory in models) and the latest node-info frame of each
// live stream. Zero-value ready.
type nodeInfoStreamState struct {
	mu     sync.Mutex
	nextID uint64
	active map[string]nodeInfoStreamHandle
	models map[string]nodeInfoStreamHandle
	live   map[string]liveNodeInfo
	// modelsLive maps a host to the id of its models stream once that stream
	// has applied a frame, until it ends.
	modelsLive map[string]uint64
}

type nodeInfoStreamHandle struct {
	id     uint64
	cancel context.CancelFunc
}

// liveNodeInfo is the latest frame of one stream. current is the stream's
// route check, so a reader can refuse the frame once admission or the route
// changed, without waiting for the stream to notice.
type liveNodeInfo struct {
	streamID uint64
	info     NodeInfoResponse
	frameAt  time.Time
	lineAt   time.Time
	current  func() bool
}

// age is the cached sample's age at now in milliseconds. A frame whose valid GPU
// sample was already stale keeps ageing from that frame, because node-info's
// interleaved keepalives say nothing about a stale sample. Otherwise the latest
// line, a keepalive included, confirms a sample within thresholds and younger
// than the stale bound, so the age counts from that line.
func (e liveNodeInfo) age(now time.Time) int64 {
	if e.info.TelemetryValid && e.info.MSSince >= nodeInfoStaleSampleAge.Milliseconds() {
		return addTelemetryAge(e.info.MSSince, now.Sub(e.frameAt).Milliseconds())
	}
	return addTelemetryAge(0, now.Sub(e.lineAt).Milliseconds())
}

// liveAt reports whether a line arrived within the idle bound.
func (e liveNodeInfo) liveAt(now time.Time) bool {
	return now.Sub(e.lineAt) <= nodeInfoStreamIdle
}

// isStreamPeer reports whether node's telemetry arrives by stream: a pinned
// cluster peer other than self that advertises node-info. The stream is plain,
// like every paired node-info read.
func (d *daemon) isStreamPeer(node noderec.DirectoryNode, self string) bool {
	_, advertised := node.Services[noderec.ServiceNodeInfo]
	return advertised && node.HostUUID != self && node.ClusterUUID != "" &&
		d.mesh != nil && d.mesh.HasPin(node.ClusterUUID)
}

func (d *daemon) selfHostUUID() string {
	if d.reg == nil {
		return ""
	}
	return d.reg.record().HostUUID
}

// streamPeerNode returns hostUUID's directory entry while it is a stream peer.
func (d *daemon) streamPeerNode(hostUUID string) (noderec.DirectoryNode, bool) {
	if d.dir == nil {
		return noderec.DirectoryNode{}, false
	}
	node, found := d.dir.get(hostUUID)
	return node, found && d.isStreamPeer(node, d.selfHostUUID())
}

// supervisePeerStreams starts the streams every stream peer should have and
// cancels those of nodes that stopped being stream peers: a node-info stream per
// stream peer, and a models stream per stream peer that also advertises
// engine-manager. It runs on every telemetry tick; workers lets the telemetry
// loop drain the streams on shutdown.
func (d *daemon) supervisePeerStreams(ctx context.Context, self string, workers *sync.WaitGroup) {
	nodeInfo := make(map[string]bool)
	models := make(map[string]bool)
	if d.dir != nil {
		for _, node := range d.dir.snapshot(noderec.ServiceNodeInfo) {
			if d.isStreamPeer(node, self) {
				nodeInfo[node.HostUUID] = true
				if _, ok := node.Services[noderec.ServiceEngineManager]; ok {
					models[node.HostUUID] = true
				}
			}
		}
	}
	s := &d.nodeInfoStreams
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reconcileLocked(ctx, &s.active, nodeInfo, workers, "node-info", d.runNodeInfoStream)
	s.reconcileLocked(ctx, &s.models, models, workers, "models", d.runModelsStream)
}

// reconcileLocked makes active hold exactly one stream per wanted host, up to
// peerStreamLimit, each running run until its context is cancelled. Caller
// holds s.mu.
func (s *nodeInfoStreamState) reconcileLocked(ctx context.Context, active *map[string]nodeInfoStreamHandle, wanted map[string]bool,
	workers *sync.WaitGroup, kind string, run func(ctx context.Context, hostUUID string, id uint64)) {
	for hostUUID, stream := range *active {
		if !wanted[hostUUID] {
			stream.cancel()
			delete(*active, hostUUID)
		}
	}
	for hostUUID := range wanted {
		if _, running := (*active)[hostUUID]; running {
			continue
		}
		if len(*active) >= peerStreamLimit {
			slog.Debug("peer stream limit reached", "kind", kind, "limit", peerStreamLimit, "host_uuid", hostUUID)
			continue
		}
		if *active == nil {
			*active = make(map[string]nodeInfoStreamHandle)
		}
		s.nextID++
		id := s.nextID
		streamCtx, cancel := context.WithCancel(ctx)
		(*active)[hostUUID] = nodeInfoStreamHandle{id: id, cancel: cancel}
		workers.Add(1)
		go func() {
			defer workers.Done()
			defer cancel()
			run(streamCtx, hostUUID, id)
		}()
	}
}

// runPeerStream keeps one peer stream open until ctx ends: it calls once for
// each attempt and ended after it, then reconnects after telemetryRetryDelay.
// Failures reset after an attempt once reports healthy.
func runPeerStream(ctx context.Context, once func(context.Context) bool, ended func()) {
	failures := 0
	for {
		healthy := once(ctx)
		ended()
		if ctx.Err() != nil {
			return
		}
		if healthy {
			failures = 0
		} else {
			failures++
		}
		retry := time.NewTimer(telemetryRetryDelay(failures))
		select {
		case <-ctx.Done():
			retry.Stop()
			return
		case <-retry.C:
		}
	}
}

// peerStreamRead reads one stream attempt at one candidate address. tryNext
// reports a connection failure that the next candidate address may not share.
type peerStreamRead func(client *http.Client, current func() bool, node noderec.DirectoryNode, host string, port int) (healthy, tryNext bool)

// streamPeerOnce opens one stream to hostUUID's service through the node's
// candidate addresses and reads it until it ends. It reports whether the
// stream was healthy.
func (d *daemon) streamPeerOnce(hostUUID string, service noderec.ServiceKey, read peerStreamRead) bool {
	node, ok := d.streamPeerNode(hostUUID)
	if !ok {
		return false
	}
	endpoint, ok := node.Services[service]
	if !ok {
		return false
	}
	for _, host := range node.CandidateIPs() {
		client, current, closeClient, admitted := d.peerStreamClient(hostUUID, node.ClusterUUID, host, endpoint.Port, service)
		if !admitted {
			return false
		}
		healthy, tryNext := read(client, current, node, host, endpoint.Port)
		closeClient()
		if !tryNext {
			return healthy
		}
	}
	return false
}

// connectPeerStream sends the stream request on streamCtx, cancelling it when
// the response headers take longer than peerStreamConnectTimeout. It returns
// the response on 200; otherwise tryNext reports whether the next candidate
// address may succeed. ctx is the attempt's parent context.
func connectPeerStream(ctx, streamCtx context.Context, cancel context.CancelFunc, client *http.Client, url, kind, hostUUID, host string) (resp *http.Response, tryNext bool) {
	req, err := http.NewRequestWithContext(streamCtx, http.MethodGet, url, nil)
	if err != nil {
		return nil, false
	}
	watchdog := time.AfterFunc(peerStreamConnectTimeout, cancel)
	resp, err = client.Do(req)
	if !watchdog.Stop() && err == nil {
		resp.Body.Close()
		err = errors.New("peer stream connect timed out")
	}
	if err != nil {
		slog.Debug("peer stream connect failed", "kind", kind, "host_uuid", hostUUID, "ip", host, "err", err)
		return nil, ctx.Err() == nil
	}
	switch resp.StatusCode {
	case http.StatusOK:
		slog.Debug("peer stream connected", "kind", kind, "host_uuid", hostUUID, "ip", host)
		return resp, false
	case http.StatusForbidden, http.StatusServiceUnavailable:
		slog.Debug("peer stream refused", "kind", kind, "host_uuid", hostUUID, "ip", host, "status", resp.StatusCode)
		resp.Body.Close()
		return nil, false
	default:
		slog.Debug("peer stream unexpected status", "kind", kind, "host_uuid", hostUUID, "ip", host, "status", resp.StatusCode)
		resp.Body.Close()
		return nil, true
	}
}

// runNodeInfoStream keeps one node-info stream to hostUUID open until ctx ends.
func (d *daemon) runNodeInfoStream(ctx context.Context, hostUUID string, id uint64) {
	runPeerStream(ctx, func(ctx context.Context) bool {
		return d.streamPeerOnce(hostUUID, noderec.ServiceNodeInfo, func(client *http.Client, current func() bool, _ noderec.DirectoryNode, host string, port int) (bool, bool) {
			return d.readNodeInfoStream(ctx, client, current, hostUUID, host, port, id)
		})
	}, func() { d.dropLiveNodeInfo(hostUUID, id) })
}

// readNodeInfoStream connects to one address and reads its node-info stream.
func (d *daemon) readNodeInfoStream(ctx context.Context, client *http.Client, current func() bool, hostUUID, host string, port int, id uint64) (healthy, tryNext bool) {
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	url := "http://" + net.JoinHostPort(host, strconv.Itoa(port)) + nodeInfoStreamPath
	resp, tryNext := connectPeerStream(ctx, streamCtx, cancel, client, url, "node-info", hostUUID, host)
	if resp == nil {
		return false, tryNext
	}
	defer resp.Body.Close()
	opened := time.Now()
	gotFrame := false
	err := clustertrust.ReadSnapshotStream(streamCtx, cancel, resp.Body, clustertrust.SnapshotStreamClient[NodeInfoResponse]{
		Idle:    nodeInfoStreamIdle,
		MaxLine: nodeInfoStreamMaxLine,
		Current: func() bool {
			_, stillPeer := d.streamPeerNode(hostUUID)
			return stillPeer && current()
		},
		Decode: func(line []byte) (NodeInfoResponse, error) {
			var info NodeInfoResponse
			if err := json.Unmarshal(line, &info); err != nil {
				return NodeInfoResponse{}, err
			}
			if info.HostUUID != hostUUID {
				return NodeInfoResponse{}, errors.New("node-info stream frame names another host")
			}
			return info, nil
		},
		OnFrame: func(info NodeInfoResponse) error {
			d.applyNodeInfoFrame(hostUUID, port, id, info, current)
			gotFrame = true
			return nil
		},
		OnKeepalive: func() { d.reemitNodeInfoKeepalive(hostUUID, id) },
	})
	slog.Debug("node-info stream ended", "host_uuid", hostUUID, "ip", host, "err", err)
	return gotFrame && time.Since(opened) >= peerStreamHealthy, false
}

// applyNodeInfoFrame runs a frame through the pulled sample's apply path, caches
// it for discovery:get-node-info, and applies the peer's reported membership.
func (d *daemon) applyNodeInfoFrame(hostUUID string, port int, id uint64, info NodeInfoResponse, current func() bool) {
	boot, clockErr := telemetryClock()
	now := time.Now()
	d.dir.mu.RLock()
	node, exists := d.dir.nodes[hostUUID]
	generation := d.dir.generations[hostUUID]
	endpoint, advertised := node.Services[noderec.ServiceNodeInfo]
	d.dir.mu.RUnlock()
	buildTargetCurrent := exists && advertised && endpoint.Port == port
	d.applyTelemetrySample(hostUUID, generation, buildTargetCurrent, false, port,
		telemetrySample{info: info, startedAt: now, boot: boot, clockErr: clockErr})

	s := &d.nodeInfoStreams
	s.mu.Lock()
	// A cancelled stream finishing its last frame must not replace the frame
	// of the stream that superseded it.
	if s.active[hostUUID].id == id {
		if s.live == nil {
			s.live = make(map[string]liveNodeInfo)
		}
		s.live[hostUUID] = liveNodeInfo{streamID: id, info: info, frameAt: now, lineAt: now, current: current}
	}
	s.mu.Unlock()

	d.applyReportedClusterIdentity(hostUUID, info.ClusterUUID)
}

// reemitNodeInfoKeepalive re-emits the last frame's utilization as a new
// sample. Memory facts are not re-stamped: no consumer reads a remote node's
// memory, and re-stamping would invent freshness.
func (d *daemon) reemitNodeInfoKeepalive(hostUUID string, id uint64) {
	now := time.Now()
	s := &d.nodeInfoStreams
	s.mu.Lock()
	entry, ok := s.live[hostUUID]
	if ok && entry.streamID == id {
		entry.lineAt = now
		s.live[hostUUID] = entry
	}
	s.mu.Unlock()
	if !ok || entry.streamID != id {
		return
	}
	age := int64(0)
	if entry.info.TelemetryValid {
		age = entry.age(now)
	}
	d.emitTelemetry(noderec.NodeTelemetry{
		HostUUID:          hostUUID,
		GPUUtilizationPct: maxGPUUtilization(entry.info.GPUs),
		TelemetryValid:    entry.info.TelemetryValid,
		MSSince:           age,
	})
}

func (d *daemon) dropLiveNodeInfo(hostUUID string, id uint64) {
	s := &d.nodeInfoStreams
	s.mu.Lock()
	defer s.mu.Unlock()
	if entry, ok := s.live[hostUUID]; ok && entry.streamID == id {
		delete(s.live, hostUUID)
	}
}

// liveNodeInfo returns hostUUID's cached frame while its stream is live.
func (d *daemon) liveNodeInfo(hostUUID string, now time.Time) (liveNodeInfo, bool) {
	s := &d.nodeInfoStreams
	s.mu.Lock()
	entry, ok := s.live[hostUUID]
	s.mu.Unlock()
	return entry, ok && entry.liveAt(now)
}
