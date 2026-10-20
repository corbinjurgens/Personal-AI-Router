// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"os"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
)

type workloadReverseFixture struct {
	m             *Manager
	receiver      *Server
	local, remote *clustertrust.ReverseManager
	dir           string
	endpoint      string
	node          PeerNode
	requests      atomic.Int32
	emitted       atomic.Int32
	reverseDials  atomic.Int32
	directDials   atomic.Int32
	connections   atomic.Int32
	closed        atomic.Int32
	expectedHost  atomic.Value
	sessionID     int
}

func newWorkloadReverseFixture(t *testing.T, wrap func(http.Handler) http.Handler) *workloadReverseFixture {
	t.Helper()
	selfDir, peerDir := newPinnedPeerDirs(t)
	f := &workloadReverseFixture{dir: selfDir}
	f.m = NewManager(NewCodec(&bytes.Buffer{}), 0, "self-host", selfDir)
	t.Cleanup(func() { f.m.broadcaster.clients.CloseIdle() })
	var cancel context.CancelFunc
	f.m.ctx, cancel = context.WithCancel(context.Background())
	t.Cleanup(cancel)
	peerMesh := clustertrust.Open(peerDir)
	f.receiver = NewServer(0, newDedupIndex(64), peerMesh,
		func(*Workload) error { f.emitted.Add(1); return nil },
		func(string, string) error { f.emitted.Add(1); return nil })
	f.receiver.emitCancel = func(cancelParams) error { f.emitted.Add(1); return nil }
	handler := http.Handler(http.HandlerFunc(f.receiver.handleEvents))
	if wrap != nil {
		handler = wrap(handler)
	}
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.requests.Add(1)
		if r.URL.Path != eventsPath || r.Header.Get("Content-Type") != "application/json" {
			t.Error("reverse request changed its endpoint or content type")
		}
		if r.Host != f.expectedHost.Load().(string) {
			t.Errorf("Host = %q, want original advertised endpoint", r.Host)
		}
		handler.ServeHTTP(w, r)
	}))
	server.TLS = peerMesh.ServerTLSConfig()
	server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			f.connections.Add(1)
		}
		if state == http.StateClosed {
			f.closed.Add(1)
		}
	}
	server.StartTLS()
	t.Cleanup(server.Close)
	f.endpoint = server.Listener.Addr().String()
	// The advertised listener is unreachable; only the named reverse stream
	// can reach the peer's actual TLS endpoint.
	f.node = PeerNode{ID: "peer-host", Host: "127.0.0.1", Port: 1,
		TXT: []string{clustertrust.ClusterUUIDTXTKey + "=uuid-peer"}}
	f.expectedHost.Store("127.0.0.1:1")
	f.m.peers.Replace([]PeerNode{f.node})
	validate := func(peer clustertrust.ReversePeer) bool { return peer.Incarnation == "v1" }
	var err error
	f.local, err = clustertrust.NewReverseManager(clustertrust.ReverseOptions{
		LocalUUID: "uuid-self", Validate: validate,
		ResolveService: func(context.Context, clustertrust.ReversePeer, noderec.ServiceKey) (clustertrust.ReverseTarget, error) {
			return clustertrust.ReverseTarget{}, errors.New("sender has no inbound test service")
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	f.remote, err = clustertrust.NewReverseManager(clustertrust.ReverseOptions{
		LocalUUID: "uuid-peer", Validate: validate,
		ResolveService: func(_ context.Context, peer clustertrust.ReversePeer, service noderec.ServiceKey) (clustertrust.ReverseTarget, error) {
			if peer.UUID != "uuid-self" || service != noderec.ServiceWorkload {
				return clustertrust.ReverseTarget{}, errors.New("wrong reverse service principal")
			}
			return clustertrust.ReverseTarget{Protocol: clustertrust.ReverseTLS,
				Dial: func(ctx context.Context) (net.Conn, error) {
					return (&net.Dialer{}).DialContext(ctx, "tcp", f.endpoint)
				}}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.local.Close(); err != nil {
			t.Error(err)
		}
		if err := f.remote.Close(); err != nil {
			t.Error(err)
		}
	})
	if err := f.connect(); err != nil {
		t.Fatal(err)
	}
	f.configure(func(context.Context, string, string) (net.Conn, error) {
		return nil, errors.New("inbound listener blocked")
	}, nil)
	return f
}

func (f *workloadReverseFixture) connect() error {
	f.sessionID++
	left, right := net.Pipe()
	id := strconv.FormatInt(int64(f.sessionID), 16)
	for len(id) < 32 {
		id = "0" + id
	}
	if err := f.local.Register(clustertrust.ReversePeer{UUID: "uuid-peer", Incarnation: "v1"}, left, "uuid-self", id); err != nil {
		right.Close()
		return err
	}
	return f.remote.Register(clustertrust.ReversePeer{UUID: "uuid-self", Incarnation: "v1"}, right, "uuid-self", id)
}

func (f *workloadReverseFixture) configure(direct func(context.Context, string, string) (net.Conn, error), reverse func(context.Context) (net.Conn, error)) {
	f.m.broadcaster.clients.CloseIdle()
	f.m.broadcaster.clients = clustertrust.NewPeerServiceClient(f.m.mesh, clustertrust.PeerServiceClientOptions{
		PeerClientOptions: clustertrust.PeerClientOptions{Timeout: perPeerTimeout},
		BaseConfigDir:     filepath.Dir(f.dir), ResolveRoute: f.m.broadcaster.resolveRoute,
		DirectTimeout:    50 * time.Millisecond,
		ReuseConnections: true,
		DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			f.directDials.Add(1)
			return direct(ctx, network, address)
		},
		DialReverse: func(ctx context.Context, base, uuid string, service noderec.ServiceKey) (net.Conn, error) {
			f.reverseDials.Add(1)
			if base != filepath.Dir(f.dir) || uuid != "uuid-peer" || service != noderec.ServiceWorkload {
				return nil, errors.New("incorrect private reverse dial target")
			}
			if reverse != nil {
				return reverse(ctx)
			}
			return f.local.Open(ctx, clustertrust.ReversePeer{UUID: uuid, Incarnation: "v1"}, service)
		},
	})
}

func (f *workloadReverseFixture) discover(t *testing.T, nodes ...noderec.DirectoryNode) {
	t.Helper()
	params, err := json.Marshal(noderec.GetNodesResult{Nodes: nodes})
	if err != nil {
		t.Fatal(err)
	}
	f.m.handleMessage(&Message{JSONRPC: "2.0", Method: noderec.NotifyNodes, Params: params})
	f.m.refreshPeers(context.Background())
}

func (f *workloadReverseFixture) directoryNode() noderec.DirectoryNode {
	return noderec.DirectoryNode{
		HostUUID: f.node.ID, IP: f.node.Host, ClusterUUID: "uuid-peer",
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceWorkload: {Port: f.node.Port}},
	}
}

func (f *workloadReverseFixture) waitClosed(t *testing.T, want int32) {
	t.Helper()
	deadline := time.After(time.Second)
	for f.closed.Load() < want {
		select {
		case <-deadline:
			t.Fatalf("closed connections = %d, want %d promptly after invalidation", f.closed.Load(), want)
		case <-time.After(time.Millisecond):
		}
	}
}

func reverseLifecycleFrame(t *testing.T) []byte {
	t.Helper()
	params, err := json.Marshal(lifecycleParams{WorkloadInfo: &Workload{
		ID: "job-1", Model: "llama", Engine: "ollama", RunID: "run-1", Seq: 1,
		State: StateRunning, OriginatedFrom: "self-host", CreatedAt: 1,
	}})
	if err != nil {
		t.Fatal(err)
	}
	frame, err := json.Marshal(&Message{JSONRPC: "2.0", Method: MethodStarted, Params: params})
	if err != nil {
		t.Fatal(err)
	}
	return frame
}

func TestBroadcastReverseDeliversAndDeduplicates(t *testing.T) {
	f := newWorkloadReverseFixture(t, nil)
	frame := reverseLifecycleFrame(t)
	f.m.broadcaster.Broadcast(context.Background(), frame)
	f.m.broadcaster.Broadcast(context.Background(), frame)
	cancel := cancelFrame(t, cancelParams{OriginatedFrom: "peer-host", WorkloadID: "job-1", Engine: "ollama", RunID: "run-1", CancelID: "cancel-1"})
	f.m.broadcaster.BroadcastTo(context.Background(), cancel, func(t target) bool { return t.id == "peer-host" })
	f.m.broadcaster.BroadcastTo(context.Background(), cancel, func(t target) bool { return t.id == "peer-host" })
	remove := []byte(`{"jsonrpc":"2.0","method":"workloads:remove","params":{"workloadId":"job-1","originatedFrom":"self-host"}}`)
	f.m.broadcaster.Broadcast(context.Background(), remove)
	f.m.broadcaster.Broadcast(context.Background(), remove)
	if got := f.requests.Load(); got != 6 {
		t.Fatalf("requests = %d, want 6 explicit lifecycle, cancel and removal broadcasts", got)
	}
	if got := f.emitted.Load(); got != 3 {
		t.Fatalf("broker mutations = %d, want one each for lifecycle, cancel and removal", got)
	}
	if got := f.directDials.Load(); got != 1 {
		t.Fatalf("direct probes = %d, want 1 before reverse health is cached", got)
	}
	if got := f.reverseDials.Load(); got != 1 {
		t.Fatalf("reverse streams = %d, want 1 reused stream for 6 requests", got)
	}
}

func TestBroadcastReverseRetriesAreBounded(t *testing.T) {
	f := newWorkloadReverseFixture(t, func(http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "broker unavailable", http.StatusInternalServerError)
		})
	})
	f.m.broadcaster.Broadcast(context.Background(), reverseLifecycleFrame(t))
	if got := f.requests.Load(); got != 3 {
		t.Fatalf("HTTP attempts = %d, want exactly 3", got)
	}
	if got := f.reverseDials.Load(); got != 1 {
		t.Fatalf("reverse streams = %d, want 1 reused for exactly 3 application attempts", got)
	}
	if got := f.emitted.Load(); got != 0 {
		t.Fatalf("broker mutations = %d, want 0", got)
	}
}

func TestBroadcastReverseLostReplyReconnectDeduplicates(t *testing.T) {
	test := func(name string, direct bool, frame func(*testing.T) []byte) {
		t.Run(name, func(t *testing.T) {
			var f *workloadReverseFixture
			var lost atomic.Bool
			var armed atomic.Bool
			f = newWorkloadReverseFixture(t, func(next http.Handler) http.Handler {
				return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if !armed.Load() || lost.Swap(true) {
						next.ServeHTTP(w, r)
						return
					}
					// Accept and commit via the real receiver, then discard its
					// reply and close the old reverse session before reconnecting.
					reply := httptest.NewRecorder()
					next.ServeHTTP(reply, r)
					if reply.Code != http.StatusOK {
						t.Errorf("receiver acceptance = %d, want 200", reply.Code)
					}
					conn, _, err := w.(http.Hijacker).Hijack()
					if err != nil {
						t.Error(err)
						return
					}
					f.local.Revoke("uuid-peer")
					f.remote.Revoke("uuid-self")
					conn.Close()
					if err := f.connect(); err != nil {
						t.Error(err)
					}
				})
			})
			if direct {
				f.configure(func(ctx context.Context, _, _ string) (net.Conn, error) {
					return (&net.Dialer{}).DialContext(ctx, "tcp", f.endpoint)
				}, nil)
			}
			b := f.m.broadcaster
			client := b.clients.Client("uuid-peer", noderec.ServiceWorkload)
			url := "https://127.0.0.1:1" + eventsPath
			warm := cancelFrame(t, cancelParams{OriginatedFrom: "peer-host", WorkloadID: "warm", Engine: "ollama", RunID: "run-1", CancelID: "warm"})
			if err := b.post(context.Background(), client, url, warm); err != nil {
				t.Fatal(err)
			}
			armed.Store(true)
			payload := frame(t)
			var reused atomic.Bool
			ctx := httptrace.WithClientTrace(context.Background(), &httptrace.ClientTrace{
				GotConn: func(info httptrace.GotConnInfo) { reused.Store(info.Reused) },
			})
			if err := b.post(ctx, client, url, payload); err == nil {
				t.Fatal("lost reply succeeded, want error exposed without transport replay")
			}
			if !reused.Load() {
				t.Fatal("lost reply did not exercise a reused connection")
			}
			if got := f.requests.Load(); got != 2 {
				t.Fatalf("requests after lost reply = %d, want warmup and one dispatch without replay", got)
			}
			if got := f.emitted.Load(); got != 2 {
				t.Fatalf("mutations after reconnect = %d, want warmup and one accepted event", got)
			}
			b.Broadcast(context.Background(), payload)
			if got := f.requests.Load(); got != 3 {
				t.Fatalf("requests after explicit retry = %d, want 3", got)
			}
			if got := f.emitted.Load(); got != 2 {
				t.Fatalf("completed event duplicated after reconnect: mutations=%d", got)
			}
			b.Broadcast(context.Background(), payload)
			if f.requests.Load() != 4 || f.emitted.Load() != 2 {
				t.Fatalf("completed redelivery: requests=%d mutations=%d, want 4 and 2", f.requests.Load(), f.emitted.Load())
			}
		})
	}
	cancel := func(t *testing.T) []byte {
		return cancelFrame(t, cancelParams{OriginatedFrom: "peer-host", WorkloadID: "job-1", Engine: "ollama", RunID: "run-1", CancelID: "cancel-1"})
	}
	remove := func(t *testing.T) []byte {
		return []byte(`{"jsonrpc":"2.0","method":"workloads:remove","params":{"workloadId":"job-1","originatedFrom":"self-host"}}`)
	}
	test("reverse lifecycle", false, reverseLifecycleFrame)
	test("reverse cancel", false, cancel)
	test("reverse removal", false, remove)
	test("direct lifecycle", true, reverseLifecycleFrame)
	test("direct cancel", true, cancel)
	test("direct removal", true, remove)
}

func TestBroadcastReverseRemovedPinDuringSetupFailsClosed(t *testing.T) {
	f := newWorkloadReverseFixture(t, nil)
	f.configure(func(ctx context.Context, _, _ string) (net.Conn, error) {
		conn, err := (&net.Dialer{}).DialContext(ctx, "tcp", f.endpoint)
		if err != nil {
			return nil, err
		}
		if err := os.Remove(filepath.Join(f.dir, "trusted", "uuid-peer.json")); err != nil {
			conn.Close()
			return nil, err
		}
		return conn, nil
	}, nil)
	f.m.broadcaster.Broadcast(context.Background(), reverseLifecycleFrame(t))
	if got := f.reverseDials.Load(); got != 0 {
		t.Fatalf("reverse dials after pin removal = %d, want 0", got)
	}
	if got := f.requests.Load(); got != 0 {
		t.Fatalf("requests after pin removal = %d, want 0", got)
	}
	if got := f.emitted.Load(); got != 0 {
		t.Fatalf("mutations after pin removal = %d, want 0", got)
	}
}

func TestBroadcastReverseTLSSetupFailureUsesReverseBeforeDispatch(t *testing.T) {
	f := newWorkloadReverseFixture(t, nil)
	f.configure(func(ctx context.Context, _, _ string) (net.Conn, error) {
		left, right := net.Pipe()
		// TCP setup succeeds, but the direct endpoint never completes TLS.
		// Closing at its setup deadline lets the adapter choose reverse while
		// the workload mutation has not yet been dispatched.
		context.AfterFunc(ctx, func() { left.Close(); right.Close() })
		return left, nil
	}, nil)
	f.m.broadcaster.Broadcast(context.Background(), reverseLifecycleFrame(t))
	if got := f.directDials.Load(); got != 1 {
		t.Fatalf("direct TLS attempts = %d, want 1", got)
	}
	if got := f.reverseDials.Load(); got != 1 {
		t.Fatalf("reverse streams = %d, want 1 after the setup failure", got)
	}
	if got := f.requests.Load(); got != 1 {
		t.Fatalf("HTTP dispatches = %d, want only one over reverse", got)
	}
	if got := f.emitted.Load(); got != 1 {
		t.Fatalf("broker mutations = %d, want 1", got)
	}
}

func TestCancelReverseCallerCancellationClosesActiveStream(t *testing.T) {
	accepted, closed := make(chan struct{}), make(chan struct{})
	f := newWorkloadReverseFixture(t, func(http.Handler) http.Handler {
		return http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
			if _, err := io.Copy(io.Discard, r.Body); err != nil {
				t.Error(err)
			}
			close(accepted)
			<-r.Context().Done()
			close(closed)
		})
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	frame := cancelFrame(t, cancelParams{OriginatedFrom: "peer-host", WorkloadID: "job-1", Engine: "ollama", RunID: "run-1", CancelID: "cancel-1"})
	go func() {
		f.m.broadcaster.BroadcastTo(ctx, frame, func(t target) bool { return t.id == "peer-host" })
		close(done)
	}()
	select {
	case <-accepted:
	case <-time.After(time.Second):
		t.Fatal("cancel request did not reach reverse listener")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("cancel delivery did not stop with its caller")
	}
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("peer HTTP request survived caller cancellation")
	}
	if got := f.requests.Load(); got != 1 {
		t.Fatalf("cancel HTTP attempts = %d, want 1 with no retry after cancellation", got)
	}
	if got := f.emitted.Load(); got != 0 {
		t.Fatalf("mutations before blocked receiver acceptance = %d, want 0", got)
	}
}

func TestBroadcastReverseDisconnectedFailsClosed(t *testing.T) {
	f := newWorkloadReverseFixture(t, nil)
	f.local.Revoke("uuid-peer")
	f.remote.Revoke("uuid-self")
	f.m.broadcaster.Broadcast(context.Background(), reverseLifecycleFrame(t))
	if got := f.reverseDials.Load(); got != 3 {
		t.Fatalf("unavailable reverse attempts = %d, want bounded budget of 3", got)
	}
	if got := f.requests.Load(); got != 0 {
		t.Fatalf("requests without a session = %d, want 0", got)
	}
	if got := f.emitted.Load(); got != 0 {
		t.Fatalf("mutations without a session = %d, want 0", got)
	}
}

func TestBroadcastReverseCallerCancellationStopsAttempt(t *testing.T) {
	f := newWorkloadReverseFixture(t, nil)
	started := make(chan struct{})
	f.configure(func(context.Context, string, string) (net.Conn, error) {
		return nil, errors.New("blocked direct listener")
	}, func(ctx context.Context) (net.Conn, error) {
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() {
		f.m.broadcaster.Broadcast(ctx, reverseLifecycleFrame(t))
		close(done)
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("reverse setup did not start")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("broadcast did not stop on caller cancellation")
	}
	if got := f.reverseDials.Load(); got != 1 {
		t.Fatalf("attempts after cancellation = %d, want 1", got)
	}
	if got := f.requests.Load(); got != 0 {
		t.Fatalf("requests after cancelled setup = %d, want 0", got)
	}
}

func TestBroadcastReversePreservesQueuedUpsertRemoveOrder(t *testing.T) {
	f := newWorkloadReverseFixture(t, nil)
	observed := make(chan string, 2)
	f.receiver.emitUpsert = func(*Workload) error { observed <- "upsert"; return nil }
	f.receiver.emitRemove = func(id, origin string) error {
		if id != "job-1" || origin != "self-host" {
			return errors.New("removal changed workload identity")
		}
		observed <- "remove"
		return nil
	}
	f.m.broadcastCh <- reverseLifecycleFrame(t)
	f.m.broadcastFrame(MethodRemove, json.RawMessage(`{"workloadId":"job-1","originatedFrom":"self-host"}`))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() { f.m.broadcastLoop(ctx); close(done) }()
	for _, want := range []string{"upsert", "remove"} {
		select {
		case got := <-observed:
			if got != want {
				t.Fatalf("queued mutation = %q, want %q", got, want)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for %s", want)
		}
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("ordered broadcast loop failed to stop")
	}
}

func TestLocalCancelReverseBypassesLifecycleQueue(t *testing.T) {
	f := newWorkloadReverseFixture(t, nil)
	received := make(chan cancelParams, 1)
	f.receiver.emitCancel = func(p cancelParams) error { received <- p; return nil }
	// No broadcast worker drains the pending lifecycle frame. Cancellation
	// must reach its origin independently of that queue.
	f.m.broadcastCh <- reverseLifecycleFrame(t)
	f.m.handleLocalCancel(&Message{JSONRPC: "2.0", Method: MethodCancel,
		Params: json.RawMessage(`{"originatedFrom":"peer-host","workloadId":"job-1","regenerate":true,"engine":"ollama","runId":"run-1"}`)})
	select {
	case got := <-received:
		if got.OriginatedFrom != "peer-host" || got.WorkloadID != "job-1" || !got.Regenerate ||
			got.Engine != "ollama" || got.RunID != "run-1" || got.CancelID == "" {
			t.Fatalf("reverse cancel = %+v, want original job fields and a new cancel ID", got)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("cancel waited behind undrained lifecycle queue")
	}
	if got := len(f.m.broadcastCh); got != 1 {
		t.Fatalf("pending lifecycle frames = %d, want 1 untouched frame", got)
	}
}

func TestBroadcastReverseEndpointChangeIsResolvedOnRetry(t *testing.T) {
	var f *workloadReverseFixture
	var changed atomic.Bool
	f = newWorkloadReverseFixture(t, func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !changed.Swap(true) {
				updated := f.node
				updated.Port = 2
				f.m.peers.Replace([]PeerNode{updated})
				f.expectedHost.Store("127.0.0.1:2")
				http.Error(w, "retry with current endpoint", http.StatusServiceUnavailable)
				return
			}
			next.ServeHTTP(w, r)
		})
	})
	f.m.broadcaster.Broadcast(context.Background(), reverseLifecycleFrame(t))
	if got := f.requests.Load(); got != 2 {
		t.Fatalf("requests = %d, want original endpoint failure and current endpoint success", got)
	}
	if got := f.emitted.Load(); got != 1 {
		t.Fatalf("mutations = %d, want one event through the new endpoint", got)
	}
	if got := f.reverseDials.Load(); got != 2 {
		t.Fatalf("reverse streams = %d, want new connection for changed endpoint generation", got)
	}
}

type workloadStaleConn struct {
	net.Conn
	fail atomic.Bool
}

func (c *workloadStaleConn) Write(data []byte) (int, error) {
	if c.fail.Load() {
		return 0, errors.New("stale pooled direct connection")
	}
	return c.Conn.Write(data)
}

func TestBroadcastStaleDirectConnectionReturnsWithoutReplay(t *testing.T) {
	f := newWorkloadReverseFixture(t, nil)
	var stale *workloadStaleConn
	f.configure(func(ctx context.Context, _, _ string) (net.Conn, error) {
		conn, err := (&net.Dialer{}).DialContext(ctx, "tcp", f.endpoint)
		if err != nil {
			return nil, err
		}
		stale = &workloadStaleConn{Conn: conn}
		return stale, nil
	}, nil)
	b := f.m.broadcaster
	client := b.clients.Client("uuid-peer", noderec.ServiceWorkload)
	url := "https://127.0.0.1:1" + eventsPath
	frame := reverseLifecycleFrame(t)
	if err := b.post(context.Background(), client, url, frame); err != nil {
		t.Fatal(err)
	}
	stale.fail.Store(true)
	var reused atomic.Bool
	ctx := httptrace.WithClientTrace(context.Background(), &httptrace.ClientTrace{
		GotConn: func(info httptrace.GotConnInfo) { reused.Store(info.Reused) },
	})
	if err := b.post(ctx, client, url, frame); err == nil {
		t.Fatal("stale connection succeeded")
	}
	if !reused.Load() || f.requests.Load() != 1 || f.emitted.Load() != 1 || f.directDials.Load() != 1 || f.reverseDials.Load() != 0 {
		t.Fatalf("stale dispatch: reused=%v requests=%d mutations=%d direct=%d reverse=%d; want true, 1, 1, 1, 0 without replay", reused.Load(), f.requests.Load(), f.emitted.Load(), f.directDials.Load(), f.reverseDials.Load())
	}
	b.Broadcast(context.Background(), frame)
	if f.requests.Load() != 2 || f.emitted.Load() != 1 || f.reverseDials.Load() != 1 {
		t.Fatalf("explicit retry: requests=%d mutations=%d reverse=%d, want 2, 1, 1", f.requests.Load(), f.emitted.Load(), f.reverseDials.Load())
	}
}

func TestBroadcastReusableAuthorityChanges(t *testing.T) {
	test := func(name string, direct bool, change func(*testing.T, *workloadReverseFixture), allowed bool) {
		t.Run(name, func(t *testing.T) {
			f := newWorkloadReverseFixture(t, nil)
			if direct {
				f.configure(func(ctx context.Context, _, _ string) (net.Conn, error) {
					return (&net.Dialer{}).DialContext(ctx, "tcp", f.endpoint)
				}, nil)
			}
			b := f.m.broadcaster
			client := b.clients.Client("uuid-peer", noderec.ServiceWorkload)
			url := "https://127.0.0.1:1" + eventsPath
			frame := reverseLifecycleFrame(t)
			if err := b.post(context.Background(), client, url, frame); err != nil {
				t.Fatal(err)
			}
			change(t, f)
			url = "https://" + f.expectedHost.Load().(string) + eventsPath
			var reused atomic.Bool
			ctx := httptrace.WithClientTrace(context.Background(), &httptrace.ClientTrace{
				GotConn: func(info httptrace.GotConnInfo) { reused.Store(info.Reused) },
			})
			if err := b.post(ctx, client, url, frame); (err == nil) != allowed {
				t.Fatalf("post after authority change: err=%v allowed=%v", err, allowed)
			}
			wantRequests, wantConnections := int32(1), int32(1)
			if allowed {
				wantRequests, wantConnections = 2, 2
			}
			if reused.Load() || f.requests.Load() != wantRequests || f.connections.Load() != wantConnections || f.emitted.Load() != 1 {
				t.Fatalf("authority change: reused=%v requests=%d connections=%d mutations=%d, want false, %d, %d, 1", reused.Load(), f.requests.Load(), f.connections.Load(), f.emitted.Load(), wantRequests, wantConnections)
			}
			deadline := time.After(time.Second)
			for f.closed.Load() == 0 {
				select {
				case <-deadline:
					t.Fatal("old idle connection survived authority change")
				case <-time.After(time.Millisecond):
				}
			}
		})
	}
	writeAdmission := func(record string) func(*testing.T, *workloadReverseFixture) {
		return func(t *testing.T, f *workloadReverseFixture) {
			t.Helper()
			if err := os.WriteFile(filepath.Join(f.dir, "admission.json"), []byte(record), 0o600); err != nil {
				t.Fatal(err)
			}
		}
	}
	peerEpoch := func(t *testing.T, f *workloadReverseFixture) {
		t.Helper()
		path := filepath.Join(f.dir, "trusted", "uuid-peer.json")
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var pin workloadTestPin
		if err := json.Unmarshal(data, &pin); err != nil {
			t.Fatal(err)
		}
		pin.AdmissionEpoch++
		data, err = json.Marshal(pin)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		members, err := json.Marshal([]map[string]any{{"nodeUuid": pin.NodeUUID, "state": "member",
			"clusterId": pin.ClusterID, "admissionEpoch": pin.AdmissionEpoch}})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(f.dir, "members.json"), members, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	revoke := func(t *testing.T, f *workloadReverseFixture) {
		t.Helper()
		if err := os.Remove(filepath.Join(f.dir, "trusted", "uuid-peer.json")); err != nil {
			t.Fatal(err)
		}
	}
	endpoint := func(t *testing.T, f *workloadReverseFixture) {
		t.Helper()
		node := f.node
		node.Port = 2
		f.expectedHost.Store("127.0.0.1:2")
		f.m.peers.Replace([]PeerNode{node})
	}
	generation := func(t *testing.T, f *workloadReverseFixture) {
		t.Helper()
		f.m.peers.Replace(nil)
		f.m.peers.Replace([]PeerNode{f.node})
	}
	missing := func(t *testing.T, f *workloadReverseFixture) {
		t.Helper()
		if err := os.Remove(filepath.Join(f.dir, "admission.json")); err != nil {
			t.Fatal(err)
		}
	}
	readmit := writeAdmission(`{"clusterId":"test-cluster","epoch":2,"counter":2,"activated":2}`)
	malformed := writeAdmission(`{"clusterId":"test-cluster","epoch":1,"counter":0}`)
	cleared := writeAdmission(`{"clusterId":"","epoch":0,"counter":2,"activated":2,"retired":true}`)
	test("direct local readmission", true, readmit, true)
	test("reverse local readmission", false, readmit, true)
	test("direct peer readmission", true, peerEpoch, true)
	test("reverse peer readmission", false, peerEpoch, true)
	test("direct malformed authority", true, malformed, false)
	test("reverse malformed authority", false, malformed, false)
	test("direct cleared admission", true, cleared, false)
	test("reverse cleared admission", false, cleared, false)
	test("direct revoked pin", true, revoke, false)
	test("reverse revoked pin", false, revoke, false)
	test("direct missing admission", true, missing, false)
	test("reverse missing admission", false, missing, false)
	test("direct changed endpoint", true, endpoint, true)
	test("reverse changed endpoint", false, endpoint, true)
	test("direct changed generation", true, generation, true)
	test("reverse changed generation", false, generation, true)
}

func TestBroadcastRevokedPinRetiresIdle(t *testing.T) {
	test := func(name string, direct, targeted bool) {
		t.Run(name, func(t *testing.T) {
			f := newWorkloadReverseFixture(t, nil)
			if direct {
				f.configure(func(ctx context.Context, _, _ string) (net.Conn, error) {
					return (&net.Dialer{}).DialContext(ctx, "tcp", f.endpoint)
				}, nil)
			}
			frame := reverseLifecycleFrame(t)
			send := func() { f.m.broadcaster.Broadcast(context.Background(), frame) }
			if targeted {
				frame = cancelFrame(t, cancelParams{OriginatedFrom: "peer-host", WorkloadID: "job-1", Engine: "ollama", RunID: "run-1", CancelID: "cancel-1"})
				send = func() {
					f.m.broadcaster.BroadcastTo(context.Background(), frame, func(t target) bool { return t.id == "peer-host" })
				}
			}
			send()
			send()
			if f.requests.Load() != 2 || f.connections.Load() != 1 || f.closed.Load() != 0 || f.emitted.Load() != 1 {
				t.Fatalf("warm connection: requests=%d connections=%d closed=%d mutations=%d, want 2, 1, 0, 1", f.requests.Load(), f.connections.Load(), f.closed.Load(), f.emitted.Load())
			}
			if err := os.Remove(filepath.Join(f.dir, "trusted", "uuid-peer.json")); err != nil {
				t.Fatal(err)
			}
			f.m.mesh.Refresh()
			if !f.m.mesh.Clustered() || f.m.mesh.HasPin("uuid-peer") {
				t.Fatal("pin revocation must leave local admission active and the peer unpinned")
			}
			send()
			f.waitClosed(t, 1)
			wantDirect, wantReverse := int32(1), int32(1)
			if direct {
				wantReverse = 0
			}
			if f.requests.Load() != 2 || f.emitted.Load() != 1 || f.connections.Load() != 1 || f.directDials.Load() != wantDirect || f.reverseDials.Load() != wantReverse {
				t.Fatalf("revoked peer: requests=%d mutations=%d connections=%d direct=%d reverse=%d, want 2, 1, 1, %d, %d without dispatch", f.requests.Load(), f.emitted.Load(), f.connections.Load(), f.directDials.Load(), f.reverseDials.Load(), wantDirect, wantReverse)
			}
		})
	}
	test("direct broadcast", true, false)
	test("reverse broadcast", false, false)
	test("direct targeted cancel", true, true)
	test("reverse targeted cancel", false, true)
}

func TestBroadcastPeerUpdateRetiresIdle(t *testing.T) {
	test := func(name string, direct bool, change func(*testing.T, *workloadReverseFixture), allowed bool) {
		t.Run(name, func(t *testing.T) {
			f := newWorkloadReverseFixture(t, nil)
			if direct {
				f.configure(func(ctx context.Context, _, _ string) (net.Conn, error) {
					return (&net.Dialer{}).DialContext(ctx, "tcp", f.endpoint)
				}, nil)
			}
			f.discover(t, f.directoryNode())
			frame := reverseLifecycleFrame(t)
			f.m.broadcaster.Broadcast(context.Background(), frame)
			// A repeated production discovery snapshot must keep the warm route.
			f.discover(t, f.directoryNode())
			f.m.broadcaster.Broadcast(context.Background(), frame)
			if f.requests.Load() != 2 || f.connections.Load() != 1 || f.closed.Load() != 0 || f.emitted.Load() != 1 {
				t.Fatalf("unchanged discovery: requests=%d connections=%d closed=%d mutations=%d, want 2, 1, 0, 1", f.requests.Load(), f.connections.Load(), f.closed.Load(), f.emitted.Load())
			}
			change(t, f)
			// Check retirement before sending again: a zero-target/ambiguous
			// route cannot rely on a subsequent adapter call to do cleanup.
			f.waitClosed(t, 1)
			f.m.broadcaster.Broadcast(context.Background(), frame)
			f.m.broadcaster.BroadcastTo(context.Background(), frame, func(t target) bool { return t.id == "peer-host" })
			wantRequests, wantConnections := int32(2), int32(1)
			if allowed {
				wantRequests, wantConnections = 4, 2
			}
			if f.requests.Load() != wantRequests || f.connections.Load() != wantConnections || f.emitted.Load() != 1 {
				t.Fatalf("updated discovery: requests=%d connections=%d mutations=%d, want %d, %d, 1", f.requests.Load(), f.connections.Load(), f.emitted.Load(), wantRequests, wantConnections)
			}
		})
	}
	remove := func(t *testing.T, f *workloadReverseFixture) { f.discover(t) }
	ambiguous := func(t *testing.T, f *workloadReverseFixture) {
		node := f.directoryNode()
		other := f.directoryNode()
		other.HostUUID = "conflicting-host"
		f.discover(t, node, other)
	}
	endpoint := func(t *testing.T, f *workloadReverseFixture) {
		node := f.directoryNode()
		node.Services[noderec.ServiceWorkload] = noderec.ServiceStatus{Port: 2}
		f.expectedHost.Store("127.0.0.1:2")
		f.discover(t, node)
	}
	generation := func(t *testing.T, f *workloadReverseFixture) {
		f.discover(t)
		f.discover(t, f.directoryNode())
	}
	principal := func(t *testing.T, f *workloadReverseFixture) {
		node := f.directoryNode()
		node.ClusterUUID = "unpinned-peer"
		f.discover(t, node)
	}
	test("direct removed route", true, remove, false)
	test("reverse removed route", false, remove, false)
	test("direct ambiguous route", true, ambiguous, false)
	test("reverse ambiguous route", false, ambiguous, false)
	test("direct changed endpoint", true, endpoint, true)
	test("reverse changed endpoint", false, endpoint, true)
	test("direct changed generation", true, generation, true)
	test("reverse changed generation", false, generation, true)
	test("direct changed principal", true, principal, false)
	test("reverse changed principal", false, principal, false)
}

func TestBroadcastPeerInvalidationPreservesActiveResponse(t *testing.T) {
	test := func(name string, direct bool) {
		t.Run(name, func(t *testing.T) {
			active := make(chan struct{})
			release := make(chan struct{}, 1)
			defer close(release)
			f := newWorkloadReverseFixture(t, func(next http.Handler) http.Handler {
				return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					body, err := io.ReadAll(r.Body)
					if err != nil {
						t.Error(err)
						return
					}
					var msg Message
					if err := json.Unmarshal(body, &msg); err != nil {
						t.Error(err)
						return
					}
					r.Body = io.NopCloser(bytes.NewReader(body))
					if msg.Method != MethodCancel {
						next.ServeHTTP(w, r)
						return
					}
					reply := httptest.NewRecorder()
					next.ServeHTTP(reply, r)
					if reply.Code != http.StatusOK {
						t.Errorf("active cancel acceptance = %d, want 200", reply.Code)
					}
					w.WriteHeader(reply.Code)
					if _, err := w.Write([]byte("pending")); err != nil {
						t.Error(err)
						return
					}
					w.(http.Flusher).Flush()
					close(active)
					select {
					case <-release:
					case <-r.Context().Done():
						t.Error("peer invalidation aborted an active response")
					}
				})
			})
			if direct {
				f.configure(func(ctx context.Context, _, _ string) (net.Conn, error) {
					return (&net.Dialer{}).DialContext(ctx, "tcp", f.endpoint)
				}, nil)
			}
			frame := cancelFrame(t, cancelParams{OriginatedFrom: "peer-host", WorkloadID: "job-1", Engine: "ollama", RunID: "run-1", CancelID: "cancel-1"})
			done := make(chan struct{})
			go func() {
				f.m.broadcaster.BroadcastTo(context.Background(), frame, func(t target) bool { return t.id == "peer-host" })
				close(done)
			}()
			select {
			case <-active:
			case <-time.After(time.Second):
				t.Fatal("cancel response did not become active")
			}
			f.m.broadcaster.Broadcast(context.Background(), reverseLifecycleFrame(t))
			if f.requests.Load() != 2 || f.connections.Load() != 2 || f.closed.Load() != 0 {
				t.Fatalf("active and idle capacity: requests=%d connections=%d closed=%d, want 2, 2, 0", f.requests.Load(), f.connections.Load(), f.closed.Load())
			}
			f.discover(t)
			f.waitClosed(t, 1)
			select {
			case <-done:
				t.Fatal("idle cleanup returned before the active response finished")
			default:
			}
			// Release only this response; a cleanup that closed every socket
			// would have already cancelled the handler and finished the sender.
			release <- struct{}{}
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Fatal("active response could not finish after idle cleanup")
			}
			if f.requests.Load() != 2 || f.emitted.Load() != 2 {
				t.Fatalf("invalidation: requests=%d mutations=%d, want 2, 2 without replay", f.requests.Load(), f.emitted.Load())
			}
		})
	}
	test("direct active response", true)
	test("reverse active response", false)
}

func TestCancelReverseCancellationPreservesOtherActiveRequest(t *testing.T) {
	active, cancelActive := make(chan struct{}), make(chan struct{})
	release := make(chan struct{})
	f := newWorkloadReverseFixture(t, func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Error(err)
				return
			}
			var msg Message
			if err := json.Unmarshal(body, &msg); err != nil {
				t.Error(err)
				return
			}
			if msg.Method == MethodCancel {
				close(cancelActive)
				<-r.Context().Done()
				return
			}
			close(active)
			select {
			case <-release:
				r.Body = io.NopCloser(bytes.NewReader(body))
				next.ServeHTTP(w, r)
			case <-r.Context().Done():
				t.Error("other active lifecycle request was cancelled")
			}
		})
	})
	b := f.m.broadcaster
	client := b.clients.Client("uuid-peer", noderec.ServiceWorkload)
	url := "https://127.0.0.1:1" + eventsPath
	lifecycleDone := make(chan error, 1)
	frame := reverseLifecycleFrame(t)
	go func(payload []byte) { lifecycleDone <- b.post(context.Background(), client, url, payload) }(frame)
	wait := func(ch <-chan struct{}) {
		t.Helper()
		select {
		case <-ch:
		case <-time.After(time.Second):
			t.Fatal("request did not become active")
		}
	}
	wait(active)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cancelDone := make(chan error, 1)
	frame = cancelFrame(t, cancelParams{OriginatedFrom: "peer-host", WorkloadID: "job-1", Engine: "ollama", RunID: "run-1", CancelID: "cancel-1"})
	go func(payload []byte) { cancelDone <- b.post(ctx, client, url, payload) }(frame)
	wait(cancelActive)
	cancel()
	select {
	case err := <-cancelDone:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled request error=%v, want context.Canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("caller cancellation did not stop its request")
	}
	close(release)
	select {
	case err := <-lifecycleDone:
		if err != nil {
			t.Fatalf("independent lifecycle failed: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("independent lifecycle request did not finish")
	}
	if f.requests.Load() != 2 || f.emitted.Load() != 1 || f.reverseDials.Load() != 2 {
		t.Fatalf("concurrent requests=%d mutations=%d streams=%d, want 2, 1, 2", f.requests.Load(), f.emitted.Load(), f.reverseDials.Load())
	}
}
