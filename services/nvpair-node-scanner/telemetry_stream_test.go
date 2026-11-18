// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
)

// streamPeerFixture is a scanner with one admitted, pinned peer ("peer" at the
// documentation address 192.0.2.40) whose node-info is an httptest server on
// loopback. The scanner's direct dialer maps the peer's advertised address to
// that server, so no reverse session is involved.
type streamPeerFixture struct {
	d         *daemon
	node      noderec.DirectoryNode
	server    *httptest.Server
	port      int
	telemetry chan json.RawMessage
	streams   atomic.Int32
	pulls     atomic.Int32
	// lines carries what the stream handler writes after its first frame.
	lines chan string
	// first is the frame each stream opens with; status, when set, refuses it.
	first  atomic.Pointer[NodeInfoResponse]
	status atomic.Int32
	ended  chan struct{}
}

type eofReader struct{}

func testScannerStreamRetry(t *testing.T, d *daemon, run func(context.Context)) {
	t.Helper()
	previous := newPeerStreamRetry
	newPeerStreamRetry = func() *clustertrust.PeerRetry {
		return &clustertrust.PeerRetry{Initial: 20 * time.Millisecond, Cap: time.Second, StableAfter: time.Second, Jitter: func() float64 { return 1 }}
	}
	t.Cleanup(func() { newPeerStreamRetry = previous })
	dial := d.peerDialContext
	starts := make(chan time.Time, 8)
	d.peerDialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		starts <- time.Now()
		return dial(ctx, network, address)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); run(ctx) }()
	defer func() { cancel(); <-done }()
	var prior time.Time
	for _, delay := range []time.Duration{0, 20 * time.Millisecond, 40 * time.Millisecond, 80 * time.Millisecond} {
		select {
		case at := <-starts:
			if !prior.IsZero() && at.Sub(prior) < delay {
				t.Fatalf("retry after %v, want >= %v", at.Sub(prior), delay)
			}
			prior = at
		case <-time.After(3 * time.Second):
			t.Fatal("missing retry")
		}
	}
}

func TestNodeInfoStreamRetryLadder(t *testing.T) {
	f := newStreamPeerFixture(t)
	f.status.Store(http.StatusServiceUnavailable)
	testScannerStreamRetry(t, f.d, func(ctx context.Context) { f.d.runNodeInfoStream(ctx, "peer", 1) })
}

func (eofReader) Read([]byte) (int, error) { return 0, io.EOF }

func streamFixtureFrame(host string) NodeInfoResponse {
	cluster := host
	return NodeInfoResponse{HostUUID: host, ClusterUUID: &cluster, TelemetryValid: true, MSSince: 100,
		GPUs: []GPUInfo{{Name: "GPU 0", UtilizationPercent: ptr(uint32(40))}, {Name: "GPU 1", UtilizationPercent: ptr(uint32(70))}}}
}

func newStreamPeerFixture(t *testing.T) *streamPeerFixture {
	t.Helper()
	d, _ := newPeerServicesScanner(t)
	f := &streamPeerFixture{d: d, telemetry: make(chan json.RawMessage, 64), lines: make(chan string, 16), ended: make(chan struct{}, 16)}
	first := streamFixtureFrame("peer")
	f.first.Store(&first)
	f.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/node-info":
			f.pulls.Add(1)
			_ = json.NewEncoder(w).Encode(*f.first.Load())
		case nodeInfoStreamPath:
			f.streams.Add(1)
			defer func() { f.ended <- struct{}{} }()
			if status := f.status.Load(); status != 0 {
				w.WriteHeader(int(status))
				return
			}
			w.Header().Set("Content-Type", "application/x-ndjson")
			if err := json.NewEncoder(w).Encode(*f.first.Load()); err != nil {
				return
			}
			w.(http.Flusher).Flush()
			for {
				select {
				case <-r.Context().Done():
					return
				case line := <-f.lines:
					if _, err := io.WriteString(w, line+"\n"); err != nil {
						return
					}
					w.(http.Flusher).Flush()
				}
			}
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(f.server.Close)
	addr := f.server.Listener.Addr().(*net.TCPAddr)
	f.port = addr.Port
	advertised := net.JoinHostPort("192.0.2.40", strconv.Itoa(f.port))
	dialer := &net.Dialer{}
	d.peerDialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		if address != advertised {
			return nil, errors.New("stream fixture refuses unknown direct destination")
		}
		return dialer.DialContext(ctx, network, addr.String())
	}
	d.http = &http.Client{Timeout: nodeInfoFetchTimeout, Transport: nodeInfoTransport(nil)}
	f.node = noderec.DirectoryNode{HostUUID: "peer", ClusterUUID: "peer", IP: "192.0.2.40", Trusted: true,
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: f.port}}}
	d.dir.upsert(f.node)

	reader, writer := io.Pipe()
	d.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{eofReader{}, writer})
	go func() {
		scanner := bufio.NewScanner(reader)
		for scanner.Scan() {
			var message Message
			if json.Unmarshal(scanner.Bytes(), &message) == nil && message.Method == noderec.NotifyNodeTelemetry {
				f.telemetry <- append(json.RawMessage(nil), message.Params...)
			}
		}
	}()
	t.Cleanup(func() { _ = writer.Close() })
	return f
}

// supervise starts the scanner's streams under a context the test cancels and
// drains on cleanup.
func (f *streamPeerFixture) supervise(t *testing.T) (context.Context, *sync.WaitGroup) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	var workers sync.WaitGroup
	t.Cleanup(func() { cancel(); workers.Wait() })
	f.d.supervisePeerStreams(ctx, f.d.selfHostUUID(), &workers)
	return ctx, &workers
}

func (f *streamPeerFixture) nextTelemetry(t *testing.T) (noderec.NodeTelemetry, map[string]json.RawMessage) {
	t.Helper()
	select {
	case raw := <-f.telemetry:
		var got noderec.NodeTelemetry
		var keys map[string]json.RawMessage
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(raw, &keys); err != nil {
			t.Fatal(err)
		}
		return got, keys
	case <-time.After(3 * time.Second):
		t.Fatal("no telemetry emitted")
		return noderec.NodeTelemetry{}, nil
	}
}

func (f *streamPeerFixture) noTelemetry(t *testing.T, within time.Duration) {
	t.Helper()
	select {
	case raw := <-f.telemetry:
		t.Fatalf("unexpected telemetry: %s", raw)
	case <-time.After(within):
	}
}

func (f *streamPeerFixture) waitEnded(t *testing.T) {
	t.Helper()
	select {
	case <-f.ended:
	case <-time.After(3 * time.Second):
		t.Fatal("stream did not end")
	}
}

func shortenNodeInfoStreamIdle(t *testing.T, idle time.Duration) {
	t.Helper()
	previous := nodeInfoStreamIdle
	nodeInfoStreamIdle = idle
	t.Cleanup(func() { nodeInfoStreamIdle = previous })
}

func TestNodeInfoStreamFrameThenKeepalives(t *testing.T) {
	f := newStreamPeerFixture(t)
	f.supervise(t)
	got, keys := f.nextTelemetry(t)
	if got.HostUUID != "peer" || got.GPUUtilizationPct != 70 || !got.TelemetryValid || got.MSSince < 100 || got.MSSince > 1000 {
		t.Fatalf("frame telemetry = %+v", got)
	}
	if _, ok := keys["memoryHop"]; !ok {
		t.Fatalf("frame telemetry did not take the pulled sample's memory path: %v", keys)
	}
	for i := range 3 {
		f.lines <- "null"
		got, keys := f.nextTelemetry(t)
		if got.HostUUID != "peer" || got.GPUUtilizationPct != 70 || !got.TelemetryValid || got.MSSince != 0 {
			t.Fatalf("keepalive %d telemetry = %+v", i, got)
		}
		for _, key := range []string{"memory", "memoryRequest", "memoryHop"} {
			if _, ok := keys[key]; ok {
				t.Fatalf("keepalive %d carried %s: %v", i, key, keys)
			}
		}
	}
	if f.streams.Load() != 1 || f.pulls.Load() != 0 {
		t.Fatalf("streams/pulls = %d/%d, want 1/0", f.streams.Load(), f.pulls.Load())
	}
}

func TestNodeInfoStreamKeepaliveAgesStaleSample(t *testing.T) {
	f := newStreamPeerFixture(t)
	stale := streamFixtureFrame("peer")
	stale.MSSince = 6000
	f.first.Store(&stale)
	f.supervise(t)
	if got, _ := f.nextTelemetry(t); got.MSSince < 6000 {
		t.Fatalf("stale frame age = %d", got.MSSince)
	}
	time.Sleep(200 * time.Millisecond)
	f.lines <- "null"
	if got, _ := f.nextTelemetry(t); got.MSSince < 6200 {
		t.Fatalf("keepalive after a stale frame reported age %d, want the frame's age plus time since it arrived", got.MSSince)
	}
}

func TestNodeInfoStreamFreshFrameEndsStaleAgeing(t *testing.T) {
	f := newStreamPeerFixture(t)
	stale := streamFixtureFrame("peer")
	stale.MSSince = 6000
	f.first.Store(&stale)
	f.supervise(t)
	f.nextTelemetry(t)
	fresh := streamFixtureFrame("peer")
	fresh.MSSince = 200
	line, err := json.Marshal(fresh)
	if err != nil {
		t.Fatal(err)
	}
	f.lines <- string(line)
	if got, _ := f.nextTelemetry(t); got.MSSince < 200 || got.MSSince > 1000 {
		t.Fatalf("recovered frame age = %d", got.MSSince)
	}
	f.lines <- "null"
	if got, _ := f.nextTelemetry(t); got.MSSince != 0 {
		t.Fatalf("keepalive after a recovered frame reported age %d, want 0", got.MSSince)
	}
}

func TestNodeInfoStreamSupersededFrameIsNotCached(t *testing.T) {
	f := newStreamPeerFixture(t)
	f.d.applyNodeInfoFrame("peer", f.port, 7, streamFixtureFrame("peer"), func() bool { return true })
	if _, ok := f.d.nodeInfoStreams.live["peer"]; ok {
		t.Fatal("a frame from a stream that is no longer active was cached")
	}
}

func TestNodeInfoStreamSupersededFrameEmitsNothing(t *testing.T) {
	f := newStreamPeerFixture(t)
	f.d.nodeInfoStreams.active = map[string]nodeInfoStreamHandle{"peer": {id: 8}}
	stale := streamFixtureFrame("peer")
	cluster := "superseded"
	stale.ClusterUUID = &cluster
	stale.ServiceBuildID = scannerDirtyBuild

	f.d.applyNodeInfoFrame("peer", f.port, 7, stale, func() bool { return true })

	f.noTelemetry(t, 100*time.Millisecond)
	node, present := f.d.dir.get("peer")
	if !present || node.ClusterUUID != f.node.ClusterUUID || node.Trusted != f.node.Trusted {
		t.Fatalf("superseded frame changed cluster identity: %+v", node)
	}
	if node.ServiceBuildID != f.node.ServiceBuildID {
		t.Fatalf("superseded frame changed build: %q", node.ServiceBuildID)
	}
}

func TestNodeInfoStreamReplacementFencesConcurrentFrames(t *testing.T) {
	f := newStreamPeerFixture(t)
	fresh := streamFixtureFrame("peer")
	fresh.GPUs = []GPUInfo{{UtilizationPercent: ptr(uint32(90))}}
	fresh.ServiceBuildID = scannerCleanBuild
	f.first.Store(&fresh)
	f.d.nodeInfoStreams.nextID = 7
	f.d.nodeInfoStreams.active = map[string]nodeInfoStreamHandle{"peer": {id: 7, cancel: func() {}}}
	stale := streamFixtureFrame("peer")
	stale.ClusterUUID = nil // Keep the peer admitted while replacement races.
	stale.GPUs = []GPUInfo{{UtilizationPercent: ptr(uint32(10))}}
	stale.ServiceBuildID = scannerDirtyBuild

	ctx, cancel := context.WithCancel(context.Background())
	var streams, actions sync.WaitGroup
	start := make(chan struct{})
	replacementFrame := make(chan struct{})
	oldDone, supervised := make(chan struct{}), make(chan struct{})
	actions.Add(2)
	t.Cleanup(func() { cancel(); actions.Wait(); streams.Wait() })
	go func() {
		defer actions.Done()
		defer close(oldDone)
		<-start
		for range 32 {
			f.d.applyNodeInfoFrame("peer", f.port, 7, stale, func() bool { return true })
		}
		select {
		case <-ctx.Done():
			return
		case <-replacementFrame:
		}
		// Deliver another old frame after the replacement is observable, so
		// both possible winners of the admission/replacement race are covered.
		cluster := "superseded"
		stale.ClusterUUID = &cluster
		f.d.applyNodeInfoFrame("peer", f.port, 7, stale, func() bool { return true })
	}()
	go func() {
		defer actions.Done()
		defer close(supervised)
		<-start
		f.d.dir.remove("peer")
		f.d.supervisePeerStreams(ctx, f.d.selfHostUUID(), &streams)
		f.d.dir.upsert(f.node)
		f.d.supervisePeerStreams(ctx, f.d.selfHostUUID(), &streams)
	}()
	close(start)

	for {
		got, _ := f.nextTelemetry(t)
		if got.GPUUtilizationPct == 90 {
			break
		}
		if got.GPUUtilizationPct != 10 {
			t.Fatalf("unexpected telemetry during replacement: %+v", got)
		}
	}
	close(replacementFrame)
	select {
	case <-oldDone:
	case <-time.After(3 * time.Second):
		t.Fatal("superseded frame apply did not finish")
	}
	<-supervised
	f.noTelemetry(t, 100*time.Millisecond)
	// The telemetry write precedes cache and identity application. Wait for
	// the stream lock before reading all of the replacement's effects.
	entry, live := f.d.liveNodeInfo("peer", time.Now())
	if !live || entry.streamID != 8 || entry.info.ServiceBuildID != fresh.ServiceBuildID {
		t.Fatalf("replacement cache = %+v, live = %v", entry, live)
	}
	node, present := f.d.dir.get("peer")
	if !present || node.ClusterUUID != f.node.ClusterUUID || !node.Trusted || node.ServiceBuildID != fresh.ServiceBuildID {
		t.Fatalf("replacement identity/build overwritten: %+v", node)
	}
}

func TestNodeInfoStreamIdleEndsStream(t *testing.T) {
	shortenNodeInfoStreamIdle(t, 300*time.Millisecond)
	f := newStreamPeerFixture(t)
	f.supervise(t)
	f.nextTelemetry(t)
	f.waitEnded(t)
	f.noTelemetry(t, 500*time.Millisecond)
	if _, live := f.d.liveNodeInfo("peer", time.Now()); live {
		t.Fatal("idle stream left a live frame")
	}
}

func TestNodeInfoStreamRejectsFrameFromAnotherHost(t *testing.T) {
	f := newStreamPeerFixture(t)
	stranger := streamFixtureFrame("stranger")
	f.first.Store(&stranger)
	f.supervise(t)
	f.waitEnded(t)
	f.noTelemetry(t, 300*time.Millisecond)
}

func TestNodeInfoStreamRefusalBacksOff(t *testing.T) {
	f := newStreamPeerFixture(t)
	f.status.Store(http.StatusForbidden)
	f.supervise(t)
	time.Sleep(3 * time.Second)
	if attempts := f.streams.Load(); attempts < 2 || attempts > 3 {
		t.Fatalf("stream attempts in 3 s after 403 = %d, want 2 or 3 (jittered 1s, then 2s)", attempts)
	}
}

func TestNodeInfoStreamEndsWhenPeerLeavesDirectory(t *testing.T) {
	f := newStreamPeerFixture(t)
	ctx, workers := f.supervise(t)
	f.nextTelemetry(t)
	f.d.dir.remove("peer")
	f.d.supervisePeerStreams(ctx, f.d.selfHostUUID(), workers)
	f.waitEnded(t)
}

func TestNodeInfoStreamPeerIsNotPulled(t *testing.T) {
	f := newStreamPeerFixture(t)
	f.d.refreshTelemetryOnce(context.Background())
	if f.pulls.Load() != 0 {
		t.Fatalf("stream peer pulled %d times", f.pulls.Load())
	}
}

func TestNodeInfoStreamLeavesOtherNodesOnPull(t *testing.T) {
	t.Run("unpinned clustered node", func(t *testing.T) {
		f := newStreamPeerFixture(t)
		f.d.dir.remove("peer")
		unpinned := streamFixtureFrame("unpinned")
		f.first.Store(&unpinned)
		f.d.dir.upsert(noderec.DirectoryNode{HostUUID: "unpinned", ClusterUUID: "unpinned", IP: "127.0.0.1",
			Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: f.port}}})
		f.d.refreshTelemetryOnce(context.Background())
		if f.pulls.Load() != 1 {
			t.Fatalf("unpinned clustered node pulls = %d, want 1", f.pulls.Load())
		}
		f.supervise(t)
		if active := len(f.d.nodeInfoStreams.active); active != 0 {
			t.Fatalf("unpinned node was streamed: %d active streams", active)
		}
	})
	t.Run("self over loopback", func(t *testing.T) {
		f := newStreamPeerFixture(t)
		f.d.dir.remove("peer")
		self := streamFixtureFrame("host-uuid")
		f.first.Store(&self)
		f.d.dir.upsert(noderec.DirectoryNode{HostUUID: "host-uuid", ClusterUUID: "host-uuid", IP: "192.0.2.41",
			Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: f.port}}})
		f.d.refreshTelemetryOnce(context.Background())
		if f.pulls.Load() != 1 {
			t.Fatalf("self pulls = %d, want 1", f.pulls.Load())
		}
		f.supervise(t)
		if active := len(f.d.nodeInfoStreams.active); active != 0 {
			t.Fatalf("self was streamed: %d active streams", active)
		}
	})
}
