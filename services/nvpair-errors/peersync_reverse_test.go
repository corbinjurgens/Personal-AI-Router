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
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
	"nvpair-shared/reach"
)

type errorSyncFixture struct {
	sender, receiver *Manager
	ps               *PeerSync
	server           *httptest.Server
	local, remote    *clustertrust.ReverseManager
	event            DiscoveryEvent
	expectedHost     string
	expectedSNI      string
	blocked          atomic.Bool
	dropReply        atomic.Bool
	directCalls      atomic.Int32
	reverseCalls     atomic.Int32
	accepted         atomic.Int32
	failWrites       atomic.Bool
	beforeDial       func()
	beforeIngest     func(*http.Request) bool
	onConnect        func(*errorSyncConn)
}

type errorSyncConn struct {
	net.Conn
	fail      *atomic.Bool
	closed    chan struct{}
	closeOnce sync.Once
}

func (c *errorSyncConn) Close() error {
	err := c.Conn.Close()
	c.closeOnce.Do(func() { close(c.closed) })
	return err
}

func (f *errorSyncFixture) trackConnection(conn net.Conn) *errorSyncConn {
	c := &errorSyncConn{Conn: conn, fail: &f.failWrites, closed: make(chan struct{})}
	if f.onConnect != nil {
		f.onConnect(c)
	}
	return c
}

func (c *errorSyncConn) Write(data []byte) (int, error) {
	if c.fail.Load() {
		return 0, errors.New("injected stale snapshot connection")
	}
	return c.Conn.Write(data)
}

func newErrorSyncFixture(t *testing.T) *errorSyncFixture {
	t.Helper()
	aCert, aKey := genLeaf(t, "uuid-a")
	bCert, bKey := genLeaf(t, "uuid-b")
	dirA := setupNode(t, aCert, aKey, map[string][]byte{"uuid-b": bCert})
	dirB := setupNode(t, bCert, bKey, map[string][]byte{"uuid-a": aCert})
	aMesh, bMesh := clustertrust.Open(dirA), clustertrust.Open(dirB)
	f := &errorSyncFixture{sender: managerForNode("node-a"), receiver: managerForNode("node-b")}
	f.ps = NewPeerSync(f.sender, aMesh, dirA)
	mux := newErrorsMux(f.receiver, bMesh)
	f.server = httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/errors" || r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("snapshot request: method=%s path=%s content-type=%s", r.Method, r.URL.Path, r.Header.Get("Content-Type"))
		}
		if r.Header.Get("Idempotency-Key") != "" || r.Header.Get("X-Idempotency-Key") != "" {
			t.Error("snapshot unexpectedly carries an idempotency key")
		}
		if r.Host != f.expectedHost || r.TLS == nil || r.TLS.ServerName != f.expectedSNI {
			t.Errorf("snapshot authority: Host=%q TLS=%+v", r.Host, r.TLS)
		}
		if f.beforeIngest != nil && !f.beforeIngest(r) {
			return
		}
		// Reconcile through the real ingest/pin gate before optionally losing its
		// reply. A received request and a successful reconciliation are distinct.
		recorder := httptest.NewRecorder()
		mux.ServeHTTP(recorder, r)
		if recorder.Code == http.StatusNoContent {
			f.accepted.Add(1)
		}
		if f.dropReply.Swap(false) {
			hijacker, ok := w.(http.Hijacker)
			if !ok {
				t.Error("HTTP/1 fixture cannot hijack reply")
				return
			}
			conn, _, err := hijacker.Hijack()
			if err != nil {
				t.Errorf("lose reply: %v", err)
				return
			}
			if err := conn.Close(); err != nil {
				t.Errorf("close lost reply: %v", err)
			}
			return
		}
		w.WriteHeader(recorder.Code)
		if _, err := w.Write(recorder.Body.Bytes()); err != nil {
			t.Errorf("write snapshot reply: %v", err)
		}
	}))
	f.server.TLS = bMesh.ServerTLSConfig()
	f.server.StartTLS()
	t.Cleanup(f.server.Close)
	_, port, err := net.SplitHostPort(f.server.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	portNumber, err := strconv.Atoi(port)
	if err != nil {
		t.Fatal(err)
	}
	f.event = DiscoveryEvent{Type: "discovered", Node: RawNode{
		ID: "node-b", Host: "peer.example", Port: portNumber, TXT: []string{"cluster-uuid=uuid-b"},
	}}
	f.expectedHost, f.expectedSNI = net.JoinHostPort("peer.example", port), "peer.example"
	manager := func(uuid string, mesh *clustertrust.Mesh) *clustertrust.ReverseManager {
		m, err := clustertrust.NewReverseManager(clustertrust.ReverseOptions{
			LocalUUID: uuid, SetupTimeout: time.Second,
			Validate: func(peer clustertrust.ReversePeer) bool {
				mesh.Refresh()
				return peer.Incarnation == "fixture-admission" && mesh.HasPin(peer.UUID)
			},
			ResolveService: func(_ context.Context, _ clustertrust.ReversePeer, service noderec.ServiceKey) (clustertrust.ReverseTarget, error) {
				if service != noderec.ServiceErrors {
					return clustertrust.ReverseTarget{}, errors.New("unexpected reverse service")
				}
				return clustertrust.ReverseTarget{Protocol: clustertrust.ReverseTLS, Dial: func(ctx context.Context) (net.Conn, error) {
					return (&net.Dialer{}).DialContext(ctx, "tcp", f.server.Listener.Addr().String())
				}}, nil
			},
		})
		if err != nil {
			t.Fatalf("create reverse manager: %v", err)
		}
		t.Cleanup(func() {
			if err := m.Close(); err != nil {
				t.Errorf("close reverse manager: %v", err)
			}
		})
		return m
	}
	f.local, f.remote = manager("uuid-a", aMesh), manager("uuid-b", bMesh)
	f.ps.clients = clustertrust.NewPeerServiceClient(aMesh, clustertrust.PeerServiceClientOptions{
		PeerClientOptions: clustertrust.PeerClientOptions{Timeout: pushTimeout},
		BaseConfigDir:     filepath.Dir(dirA), ResolveRoute: f.ps.resolveRoute,
		ReuseConnections: true,
		DirectTimeout:    100 * time.Millisecond, ProbeInterval: time.Hour,
		DialContext: func(ctx context.Context, _, address string) (net.Conn, error) {
			f.directCalls.Add(1)
			if address != net.JoinHostPort("peer.example", port) {
				return nil, errors.New("unexpected direct endpoint")
			}
			if f.beforeDial != nil {
				f.beforeDial()
			}
			if f.blocked.Load() {
				return nil, errors.New("inbound blocked")
			}
			conn, err := (&net.Dialer{}).DialContext(ctx, "tcp", f.server.Listener.Addr().String())
			if err != nil {
				return nil, err
			}
			return f.trackConnection(conn), nil
		},
		DialReverse: func(ctx context.Context, base, uuid string, service noderec.ServiceKey) (net.Conn, error) {
			f.reverseCalls.Add(1)
			if base != filepath.Dir(dirA) || uuid != "uuid-b" || service != noderec.ServiceErrors {
				return nil, errors.New("unexpected private dial destination")
			}
			conn, err := f.local.Open(ctx, clustertrust.ReversePeer{UUID: uuid, Incarnation: "fixture-admission"}, service)
			if err != nil {
				return nil, err
			}
			return f.trackConnection(conn), nil
		},
	})
	t.Cleanup(f.ps.clients.CloseIdle)
	return f
}

func (f *errorSyncFixture) connect(t *testing.T, id string) {
	t.Helper()
	left, right := net.Pipe()
	if err := f.local.Register(clustertrust.ReversePeer{UUID: "uuid-b", Incarnation: "fixture-admission"}, left, "uuid-a", id); err != nil {
		right.Close()
		t.Fatalf("register local reverse session: %v", err)
	}
	if err := f.remote.Register(clustertrust.ReversePeer{UUID: "uuid-a", Incarnation: "fixture-admission"}, right, "uuid-a", id); err != nil {
		t.Fatalf("register remote reverse session: %v", err)
	}
}

func assertErrorSnapshot(t *testing.T, mgr *Manager, want []ServiceError) {
	t.Helper()
	if got := mgr.snapshot(); !slices.Equal(got, want) {
		t.Fatalf("snapshot = %+v, want %+v", got, want)
	}
}

func changeErrorSyncRecord(t *testing.T, path, field string, value any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var record map[string]json.RawMessage
	if err := json.Unmarshal(data, &record); err != nil {
		t.Fatal(err)
	}
	record[field] = mustJSON(t, value)
	if err := os.WriteFile(path, mustJSON(t, record), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestErrorsSnapshotDirectAndReverse(t *testing.T) {
	test := func(name string, blocked bool) {
		t.Run(name, func(t *testing.T) {
			f := newErrorSyncFixture(t)
			f.blocked.Store(blocked)
			f.connect(t, "11111111111111111111111111111111")
			f.sender.upsert(localErr("a:one", "node-a", 1000))
			f.sender.upsert(localErr("a:two", "node-a", 1000))
			f.sender.ReconcilePeer("node-c", []ServiceError{localErr("foreign", "node-c", 1000)})
			f.receiver.upsert(localErr("b:keep", "node-b", 1000))
			f.receiver.ReconcilePeer("node-a", []ServiceError{localErr("stale", "node-a", 1000)})
			f.receiver.ReconcilePeer("node-c", []ServiceError{localErr("c:keep", "node-c", 1000)})
			f.ps.handleEvent(context.Background(), f.event)
			assertErrorSnapshot(t, f.receiver, []ServiceError{
				localErr("a:one", "node-a", 1000), localErr("a:two", "node-a", 1000),
				localErr("b:keep", "node-b", 1000), localErr("c:keep", "node-c", 1000),
			})
			if !f.sender.clearByID("a:one") {
				t.Fatal("clear local error did not change sender")
			}
			// Exercise the same full-snapshot owner called by the heartbeat.
			var reused atomic.Int32
			trace := &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) {
				if info.Reused {
					reused.Add(1)
				}
			}}
			f.ps.pushAll(httptrace.WithClientTrace(context.Background(), trace))
			f.ps.pushAll(httptrace.WithClientTrace(context.Background(), trace))
			assertErrorSnapshot(t, f.receiver, []ServiceError{
				localErr("a:two", "node-a", 1000), localErr("b:keep", "node-b", 1000), localErr("c:keep", "node-c", 1000),
			})
			if got := f.accepted.Load(); got != 3 {
				t.Fatalf("accepted snapshots = %d, want 3", got)
			}
			if got := reused.Load(); got != 2 {
				t.Fatalf("reused heartbeat connections = %d, want 2", got)
			}
			wantDirect, wantReverse := int32(1), int32(0)
			if blocked {
				wantDirect, wantReverse = 1, 1
			}
			if direct, reverse := f.directCalls.Load(), f.reverseCalls.Load(); direct != wantDirect || reverse != wantReverse {
				t.Fatalf("dial counts: direct=%d reverse=%d, want %d/%d", direct, reverse, wantDirect, wantReverse)
			}
		})
	}
	test("direct remains preferred", false)
	test("blocked inbound uses reverse", true)
}

func TestErrorsSnapshotLostReplyDoesNotReplay(t *testing.T) {
	test := func(name string, blocked bool) {
		t.Run(name, func(t *testing.T) {
			f := newErrorSyncFixture(t)
			f.blocked.Store(blocked)
			f.connect(t, "11111111111111111111111111111111")
			f.dropReply.Store(true)
			f.sender.upsert(localErr("a:one", "node-a", 1000))
			f.ps.handleEvent(context.Background(), f.event)
			assertErrorSnapshot(t, f.receiver, []ServiceError{localErr("a:one", "node-a", 1000)})
			if got := f.accepted.Load(); got != 1 {
				t.Fatalf("lost reply reconciled %d times, want exactly 1", got)
			}
			wantReverse := int32(0)
			if blocked {
				wantReverse = 1
			}
			if got := f.reverseCalls.Load(); got != wantReverse {
				t.Fatalf("reverse dials after lost reply = %d, want %d (no fallback after dispatch)", got, wantReverse)
			}
			f.local.Revoke("uuid-b")
			f.remote.Revoke("uuid-a")
			f.blocked.Store(true)
			f.ps.pushAll(context.Background())
			if got := f.accepted.Load(); got != 1 {
				t.Fatalf("disconnected session reconciled %d snapshots, want 1", got)
			}
			f.connect(t, "22222222222222222222222222222222")
			// Reconnect itself does not push; a later explicit heartbeat converges.
			if got := f.accepted.Load(); got != 1 {
				t.Fatalf("reconnect replayed snapshot: accepted=%d", got)
			}
			f.ps.pushAll(context.Background())
			assertErrorSnapshot(t, f.receiver, []ServiceError{localErr("a:one", "node-a", 1000)})
			if got := f.accepted.Load(); got != 2 {
				t.Fatalf("reconnect heartbeat accepted=%d, want 2", got)
			}
		})
	}
	test("direct reply lost after reconcile", false)
	test("reverse reply lost after reconcile", true)
}

func TestErrorsSnapshotUnavailableFailsClosed(t *testing.T) {
	test := func(name string, wantDirect, wantReverse int32, prepare func(*testing.T, *errorSyncFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newErrorSyncFixture(t)
			f.blocked.Store(true)
			f.sender.upsert(localErr("a:one", "node-a", 1000))
			prepare(t, f)
			f.ps.handleEvent(context.Background(), f.event)
			assertErrorSnapshot(t, f.receiver, nil)
			if got := f.accepted.Load(); got != 0 {
				t.Fatalf("unavailable peer accepted %d snapshots, want 0", got)
			}
			if direct, reverse := f.directCalls.Load(), f.reverseCalls.Load(); direct != wantDirect || reverse != wantReverse {
				t.Fatalf("unavailable dial counts: direct=%d reverse=%d, want %d/%d", direct, reverse, wantDirect, wantReverse)
			}
		})
	}
	test("missing reverse session", 1, 1, func(t *testing.T, f *errorSyncFixture) {})
	test("missing pin", 0, 0, func(t *testing.T, f *errorSyncFixture) {
		if err := os.Remove(filepath.Join(f.ps.clusterDir, "trusted", "uuid-b.json")); err != nil {
			t.Fatal(err)
		}
	})
	test("local admission cleared", 0, 0, func(t *testing.T, f *errorSyncFixture) {
		changeErrorSyncRecord(t, filepath.Join(f.ps.clusterDir, "admission.json"), "epoch", 0)
	})
	test("retired local admission", 0, 0, func(t *testing.T, f *errorSyncFixture) {
		changeErrorSyncRecord(t, filepath.Join(f.ps.clusterDir, "admission.json"), "retired", true)
	})
	test("local admission exceeds counter", 0, 0, func(t *testing.T, f *errorSyncFixture) {
		changeErrorSyncRecord(t, filepath.Join(f.ps.clusterDir, "admission.json"), "counter", 0)
	})
	test("pin belongs to another cluster", 0, 0, func(t *testing.T, f *errorSyncFixture) {
		changeErrorSyncRecord(t, filepath.Join(f.ps.clusterDir, "trusted", "uuid-b.json"), "clusterId", "other-cluster")
	})
	test("invalid pin", 1, 1, func(t *testing.T, f *errorSyncFixture) {
		f.connect(t, "11111111111111111111111111111111")
		wrongCert, _ := genLeaf(t, "uuid-b")
		changeErrorSyncRecord(t, filepath.Join(f.ps.clusterDir, "trusted", "uuid-b.json"), "certPem", string(wrongCert))
	})
	test("pin revoked during setup", 1, 0, func(t *testing.T, f *errorSyncFixture) {
		f.blocked.Store(false)
		f.connect(t, "11111111111111111111111111111111")
		f.beforeDial = func() {
			if err := os.Remove(filepath.Join(f.ps.clusterDir, "trusted", "uuid-b.json")); err != nil {
				t.Error(err)
			}
		}
	})
}

func TestErrorsSnapshotAuthorityChangesDuringSetup(t *testing.T) {
	test := func(name string, change func(*testing.T, *errorSyncFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newErrorSyncFixture(t)
			f.connect(t, "11111111111111111111111111111111")
			f.beforeDial = func() { change(t, f) }
			f.sender.upsert(localErr("a:one", "node-a", 1000))
			f.ps.handleEvent(context.Background(), f.event)
			assertErrorSnapshot(t, f.receiver, nil)
			if accepted, reverse := f.accepted.Load(), f.reverseCalls.Load(); accepted != 0 || reverse != 0 {
				t.Fatalf("changed authority dispatched snapshot: accepted=%d reverse=%d, want 0/0", accepted, reverse)
			}
		})
	}
	test("local readmission with same certificate", func(t *testing.T, f *errorSyncFixture) {
		path := filepath.Join(f.ps.clusterDir, "admission.json")
		changeErrorSyncRecord(t, path, "counter", 2)
		changeErrorSyncRecord(t, path, "activated", 2)
		changeErrorSyncRecord(t, path, "epoch", 2)
	})
	test("peer readmission with same certificate", func(t *testing.T, f *errorSyncFixture) {
		changeErrorSyncRecord(t, filepath.Join(f.ps.clusterDir, "trusted", "uuid-b.json"), "admissionEpoch", 2)
	})
	test("endpoint withdrawn", func(t *testing.T, f *errorSyncFixture) {
		event := f.event
		event.Type, event.Node.Port = "updated", 0
		f.ps.handleEvent(context.Background(), event)
	})
}

func TestErrorsSnapshotCancellationBeforeDispatch(t *testing.T) {
	f := newErrorSyncFixture(t)
	f.blocked.Store(true)
	f.connect(t, "11111111111111111111111111111111")
	ctx, cancel := context.WithCancel(context.Background())
	f.beforeDial = cancel
	f.sender.upsert(localErr("a:one", "node-a", 1000))
	f.ps.handleEvent(ctx, f.event)
	assertErrorSnapshot(t, f.receiver, nil)
	if got := f.reverseCalls.Load(); got != 0 {
		t.Fatalf("canceled setup attempted %d reverse dials, want 0", got)
	}
}

func TestErrorsSnapshotCancellationAfterDispatch(t *testing.T) {
	f := newErrorSyncFixture(t)
	f.blocked.Store(true)
	f.connect(t, "11111111111111111111111111111111")
	received, canceled := make(chan struct{}), make(chan struct{})
	f.beforeIngest = func(r *http.Request) bool {
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Errorf("read dispatched snapshot: %v", err)
			return false
		}
		close(received)
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
			t.Error("receiver context survived caller cancellation")
		}
		close(canceled)
		return false
	}
	f.sender.upsert(localErr("a:one", "node-a", 1000))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() { defer close(done); f.ps.handleEvent(ctx, f.event) }()
	select {
	case <-received:
	case <-time.After(2 * time.Second):
		t.Fatal("reverse snapshot was not dispatched")
	}
	cancel()
	for _, signal := range []<-chan struct{}{done, canceled} {
		select {
		case <-signal:
		case <-time.After(2 * time.Second):
			t.Fatal("caller cancellation did not end request/receiver")
		}
	}
	assertErrorSnapshot(t, f.receiver, nil)
	if got := f.reverseCalls.Load(); got != 1 {
		t.Fatalf("canceled request used %d reverse streams, want 1", got)
	}
}

func TestErrorsSnapshotAddressRecovery(t *testing.T) {
	f := newErrorSyncFixture(t)
	f.event.Node.Host = ""
	f.event.Node.TXT = []string{"cluster-uuid=uuid-b", "ip=192.0.2.1"}
	f.event.Node.Addresses = []string{"127.0.0.1"}
	f.expectedHost = net.JoinHostPort("127.0.0.1", strconv.Itoa(f.event.Node.Port))
	f.expectedSNI = ""
	// Use the production constructor and direct dialer, proving the selected
	// endpoint is what request authorization accepts.
	f.ps = NewPeerSync(f.sender, f.ps.mesh, f.ps.clusterDir)
	t.Cleanup(f.ps.clients.CloseIdle)
	f.ps.addrs = reach.NewChooser(reach.WithDial(func(_, address string, _ time.Duration) (net.Conn, error) {
		if address != f.expectedHost {
			return nil, errors.New("address blocked")
		}
		return net.Dial("tcp", f.server.Listener.Addr().String())
	}))
	f.sender.upsert(localErr("a:one", "node-a", 1000))
	f.ps.handleEvent(context.Background(), f.event)
	assertErrorSnapshot(t, f.receiver, []ServiceError{localErr("a:one", "node-a", 1000)})
	route, ok := f.ps.resolveRoute("uuid-b", noderec.ServiceErrors)
	if !ok || route.Address != f.expectedHost {
		t.Fatalf("resolved route = %+v, admitted=%t, want chosen loopback endpoint", route, ok)
	}
	var reused atomic.Bool
	trace := &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) { reused.Store(info.Reused) }}
	f.ps.pushAll(httptrace.WithClientTrace(context.Background(), trace))
	if !reused.Load() || f.accepted.Load() != 2 {
		t.Fatalf("production heartbeat: reused=%t accepted=%d, want true/2", reused.Load(), f.accepted.Load())
	}
}

func TestErrorsSnapshotDuplicatePrincipalFailsClosed(t *testing.T) {
	test := func(name string, resolve func(*testing.T, *errorSyncFixture, DiscoveryEvent)) {
		t.Run(name, func(t *testing.T) {
			f := newErrorSyncFixture(t)
			f.sender.upsert(localErr("a:one", "node-a", 1000))
			f.ps.handleEvent(context.Background(), f.event)
			duplicate := f.event
			duplicate.Node.ID = "node-b-alias"
			f.ps.handleEvent(context.Background(), duplicate)
			// Both advertisements are present, even if only the first endpoint has
			// been selected. Repeated resolution must never pick a map entry.
			for range 20 {
				if route, ok := f.ps.resolveRoute("uuid-b", noderec.ServiceErrors); ok || route != (clustertrust.PeerServiceRoute{}) {
					t.Fatalf("ambiguous principal resolved: %+v admitted=%t", route, ok)
				}
			}
			f.ps.pushAll(context.Background())
			if accepted, direct, reverse := f.accepted.Load(), f.directCalls.Load(), f.reverseCalls.Load(); accepted != 1 || direct != 1 || reverse != 0 {
				t.Fatalf("ambiguous snapshot dispatched: accepted=%d direct=%d reverse=%d, want 1/1/0", accepted, direct, reverse)
			}
			resolve(t, f, duplicate)
			f.ps.pushAll(context.Background())
			route, ok := f.ps.resolveRoute("uuid-b", noderec.ServiceErrors)
			if !ok || route.Address != f.expectedHost || f.accepted.Load() != 2 {
				t.Fatalf("resolved duplicate: route=%+v admitted=%t accepted=%d, want current endpoint and 2 snapshots", route, ok, f.accepted.Load())
			}
			assertErrorSnapshot(t, f.receiver, []ServiceError{localErr("a:one", "node-a", 1000)})
		})
	}
	test("remove original advertisement", func(t *testing.T, f *errorSyncFixture, duplicate DiscoveryEvent) {
		event := f.event
		event.Type = "removed"
		f.ps.handleEvent(context.Background(), event)
	})
	test("remove duplicate advertisement", func(t *testing.T, f *errorSyncFixture, duplicate DiscoveryEvent) {
		duplicate.Type = "removed"
		f.ps.handleEvent(context.Background(), duplicate)
	})
	test("replace original with withdrawn endpoint", func(t *testing.T, f *errorSyncFixture, duplicate DiscoveryEvent) {
		event := f.event
		event.Type, event.Node.Port = "updated", 0
		f.ps.handleEvent(context.Background(), event)
	})
	test("replace duplicate with unpinned principal", func(t *testing.T, f *errorSyncFixture, duplicate DiscoveryEvent) {
		duplicate.Type, duplicate.Node.TXT = "updated", []string{"cluster-uuid=unknown"}
		f.ps.handleEvent(context.Background(), duplicate)
	})
}

func TestErrorsSnapshotStalePooledConnectionDoesNotReplay(t *testing.T) {
	test := func(name string, blocked bool) {
		t.Run(name, func(t *testing.T) {
			f := newErrorSyncFixture(t)
			f.blocked.Store(blocked)
			f.connect(t, "11111111111111111111111111111111")
			f.sender.upsert(localErr("a:one", "node-a", 1000))
			f.ps.handleEvent(context.Background(), f.event)
			f.sender.upsert(localErr("a:two", "node-a", 1000))
			f.failWrites.Store(true)
			var reused atomic.Bool
			trace := &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) { reused.Store(info.Reused) }}
			f.ps.pushAll(httptrace.WithClientTrace(context.Background(), trace))
			wantReverse := int32(0)
			if blocked {
				wantReverse = 1
			}
			if !reused.Load() || f.accepted.Load() != 1 || f.directCalls.Load() != 1 || f.reverseCalls.Load() != wantReverse {
				t.Fatalf("stale snapshot: reused=%t accepted=%d direct=%d reverse=%d, want true/1/1/%d", reused.Load(), f.accepted.Load(), f.directCalls.Load(), f.reverseCalls.Load(), wantReverse)
			}
			assertErrorSnapshot(t, f.receiver, []ServiceError{localErr("a:one", "node-a", 1000)})
			f.failWrites.Store(false)
			f.ps.pushAll(context.Background())
			assertErrorSnapshot(t, f.receiver, []ServiceError{localErr("a:one", "node-a", 1000), localErr("a:two", "node-a", 1000)})
			if f.accepted.Load() != 2 || f.reverseCalls.Load() != wantReverse+1 {
				t.Fatalf("later heartbeat: accepted=%d reverse=%d, want 2/%d", f.accepted.Load(), f.reverseCalls.Load(), wantReverse+1)
			}
		})
	}
	test("direct", false)
	test("reverse", true)
}

func TestErrorsSnapshotPooledReplyLossDoesNotReplay(t *testing.T) {
	test := func(name string, blocked bool) {
		t.Run(name, func(t *testing.T) {
			f := newErrorSyncFixture(t)
			f.blocked.Store(blocked)
			f.connect(t, "11111111111111111111111111111111")
			f.sender.upsert(localErr("a:one", "node-a", 1000))
			f.ps.handleEvent(context.Background(), f.event)
			f.sender.upsert(localErr("a:two", "node-a", 1000))
			f.dropReply.Store(true)
			var reused atomic.Bool
			trace := &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) { reused.Store(info.Reused) }}
			f.ps.pushAll(httptrace.WithClientTrace(context.Background(), trace))
			wantReverse := int32(0)
			if blocked {
				wantReverse = 1
			}
			if !reused.Load() || f.accepted.Load() != 2 || f.directCalls.Load() != 1 || f.reverseCalls.Load() != wantReverse {
				t.Fatalf("lost pooled reply: reused=%t accepted=%d direct=%d reverse=%d, want true/2/1/%d", reused.Load(), f.accepted.Load(), f.directCalls.Load(), f.reverseCalls.Load(), wantReverse)
			}
			assertErrorSnapshot(t, f.receiver, []ServiceError{localErr("a:one", "node-a", 1000), localErr("a:two", "node-a", 1000)})
			f.ps.pushAll(context.Background())
			if f.accepted.Load() != 3 || f.reverseCalls.Load() != wantReverse+1 {
				t.Fatalf("later heartbeat after reply loss: accepted=%d reverse=%d, want 3/%d", f.accepted.Load(), f.reverseCalls.Load(), wantReverse+1)
			}
		})
	}
	test("direct", false)
	test("reverse", true)
}

func TestErrorsSnapshotPooledAuthorityChangesBeforeDispatch(t *testing.T) {
	test := func(name string, blocked bool, change func(*testing.T, *errorSyncFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newErrorSyncFixture(t)
			f.blocked.Store(blocked)
			f.connect(t, "11111111111111111111111111111111")
			f.sender.upsert(localErr("a:one", "node-a", 1000))
			f.ps.handleEvent(context.Background(), f.event)
			f.sender.upsert(localErr("a:two", "node-a", 1000))
			var reused atomic.Bool
			trace := &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) {
				reused.Store(info.Reused)
				change(t, f)
			}}
			f.ps.pushAll(httptrace.WithClientTrace(context.Background(), trace))
			wantReverse := int32(0)
			if blocked {
				wantReverse = 1
			}
			if !reused.Load() || f.accepted.Load() != 1 || f.directCalls.Load() != 1 || f.reverseCalls.Load() != wantReverse {
				t.Fatalf("authority changed at pooled dispatch: reused=%t accepted=%d direct=%d reverse=%d, want true/1/1/%d", reused.Load(), f.accepted.Load(), f.directCalls.Load(), f.reverseCalls.Load(), wantReverse)
			}
			assertErrorSnapshot(t, f.receiver, []ServiceError{localErr("a:one", "node-a", 1000)})
		})
	}
	localReadmission := func(t *testing.T, f *errorSyncFixture) {
		path := filepath.Join(f.ps.clusterDir, "admission.json")
		changeErrorSyncRecord(t, path, "counter", 2)
		changeErrorSyncRecord(t, path, "activated", 2)
		changeErrorSyncRecord(t, path, "epoch", 2)
	}
	peerReadmission := func(t *testing.T, f *errorSyncFixture) {
		changeErrorSyncRecord(t, filepath.Join(f.ps.clusterDir, "trusted", "uuid-b.json"), "admissionEpoch", 2)
	}
	revokePin := func(t *testing.T, f *errorSyncFixture) {
		if err := os.Remove(filepath.Join(f.ps.clusterDir, "trusted", "uuid-b.json")); err != nil {
			t.Error(err)
		}
	}
	rotatePin := func(t *testing.T, f *errorSyncFixture) {
		cert, _ := genLeaf(t, "uuid-b")
		changeErrorSyncRecord(t, filepath.Join(f.ps.clusterDir, "trusted", "uuid-b.json"), "certPem", string(cert))
	}
	replaceEndpoint := func(t *testing.T, f *errorSyncFixture) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		event := f.event
		event.Type = "updated"
		f.ps.handleEvent(ctx, event)
	}
	duplicatePrincipal := func(t *testing.T, f *errorSyncFixture) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		event := f.event
		event.Node.ID = "node-b-alias"
		f.ps.handleEvent(ctx, event)
	}
	test("direct local readmission", false, localReadmission)
	test("reverse local readmission", true, localReadmission)
	test("direct peer readmission", false, peerReadmission)
	test("reverse peer readmission", true, peerReadmission)
	test("direct pin revocation", false, revokePin)
	test("reverse pin revocation", true, revokePin)
	test("direct pin rotation", false, rotatePin)
	test("reverse pin rotation", true, rotatePin)
	test("direct endpoint generation", false, replaceEndpoint)
	test("reverse endpoint generation", true, replaceEndpoint)
	test("direct duplicate principal", false, duplicatePrincipal)
	test("reverse duplicate principal", true, duplicatePrincipal)
}

func TestErrorsSnapshotPooledCancellation(t *testing.T) {
	test := func(name string, blocked bool) {
		t.Run(name, func(t *testing.T) {
			f := newErrorSyncFixture(t)
			f.blocked.Store(blocked)
			f.connect(t, "11111111111111111111111111111111")
			var pause atomic.Bool
			received, canceled := make(chan struct{}), make(chan struct{})
			f.beforeIngest = func(r *http.Request) bool {
				if !pause.Load() {
					return true
				}
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					t.Errorf("read canceled snapshot: %v", err)
					return false
				}
				close(received)
				select {
				case <-r.Context().Done():
				case <-time.After(5 * time.Second):
					t.Error("pooled receiver survived cancellation")
				}
				close(canceled)
				return false
			}
			f.sender.upsert(localErr("a:one", "node-a", 1000))
			f.ps.handleEvent(context.Background(), f.event)
			f.sender.upsert(localErr("a:two", "node-a", 1000))
			pause.Store(true)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var reused atomic.Bool
			trace := &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) { reused.Store(info.Reused) }}
			done := make(chan struct{})
			go func() {
				defer close(done)
				f.ps.pushAll(httptrace.WithClientTrace(ctx, trace))
			}()
			select {
			case <-received:
			case <-time.After(2 * time.Second):
				t.Fatal("pooled snapshot was not dispatched")
			}
			cancel()
			for _, signal := range []<-chan struct{}{done, canceled} {
				select {
				case <-signal:
				case <-time.After(2 * time.Second):
					t.Fatal("pooled caller or receiver survived cancellation")
				}
			}
			wantReverse := int32(0)
			if blocked {
				wantReverse = 1
			}
			if !reused.Load() || f.accepted.Load() != 1 || f.directCalls.Load() != 1 || f.reverseCalls.Load() != wantReverse {
				t.Fatalf("canceled pooled snapshot: reused=%t accepted=%d direct=%d reverse=%d, want true/1/1/%d", reused.Load(), f.accepted.Load(), f.directCalls.Load(), f.reverseCalls.Load(), wantReverse)
			}
			assertErrorSnapshot(t, f.receiver, []ServiceError{localErr("a:one", "node-a", 1000)})
			pause.Store(false)
			f.ps.pushAll(context.Background())
			assertErrorSnapshot(t, f.receiver, []ServiceError{localErr("a:one", "node-a", 1000), localErr("a:two", "node-a", 1000)})
			if f.accepted.Load() != 2 {
				t.Fatalf("later heartbeat after cancellation accepted=%d, want 2", f.accepted.Load())
			}
		})
	}
	test("direct", false)
	test("reverse", true)
}

func TestPeerSyncRunExitClosesIdleConnections(t *testing.T) {
	test := func(name string, blocked, cancelRun bool) {
		t.Run(name, func(t *testing.T) {
			f := newErrorSyncFixture(t)
			f.blocked.Store(blocked)
			f.connect(t, "11111111111111111111111111111111")
			connections := make(chan *errorSyncConn, 4)
			f.onConnect = func(conn *errorSyncConn) { connections <- conn }
			wait := func(signal <-chan struct{}, description string) {
				t.Helper()
				select {
				case <-signal:
				case <-time.After(time.Second):
					t.Fatalf("timed out waiting for %s", description)
				}
			}
			reused := make(chan struct{}, 1)
			fresh := make(chan struct{}, 2)
			trace := &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) {
				if info.Reused {
					reused <- struct{}{}
				} else {
					fresh <- struct{}{}
				}
			}}
			ctx, cancel := context.WithCancel(httptrace.WithClientTrace(context.Background(), trace))
			events := make(chan DiscoveryEvent)
			done := make(chan struct{})
			go func() {
				defer close(done)
				f.ps.Run(ctx, events)
			}()
			t.Cleanup(func() {
				cancel()
				wait(done, "Run cleanup")
			})
			// An ignored self event is a barrier: Run receives it only after the
			// preceding synchronous push has drained and closed its response.
			barrier := func() {
				t.Helper()
				select {
				case events <- DiscoveryEvent{Node: RawNode{ID: "node-a"}}:
				case <-time.After(time.Second):
					t.Fatal("Run did not finish its snapshot push")
				}
			}
			f.sender.upsert(localErr("a:one", "node-a", 1000))
			select {
			case events <- f.event:
			case <-time.After(time.Second):
				t.Fatal("Run did not receive discovery")
			}
			wait(fresh, "cold-start snapshot dispatch")
			barrier()
			activeConn := <-connections
			body, err := f.ps.snapshotBody()
			if err != nil {
				t.Fatalf("encode independent snapshot: %v", err)
			}
			req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, "https://"+f.expectedHost+"/v1/errors", bytes.NewReader(body))
			if err != nil {
				t.Fatalf("create independent snapshot: %v", err)
			}
			req.Header.Set("Content-Type", "application/json")
			response, err := f.ps.clients.Client("uuid-b", noderec.ServiceErrors).Do(req)
			if err != nil {
				t.Fatalf("expose independent response: %v", err)
			}
			defer response.Body.Close()
			if response.StatusCode != http.StatusNoContent {
				t.Fatalf("independent response status = %d, want 204", response.StatusCode)
			}
			// Keep that response exposed while Run warms and reuses another
			// connection. The held body belongs to an independent caller context.
			f.ps.TriggerPush()
			wait(fresh, "second snapshot connection dispatch")
			barrier()
			idleConn := <-connections
			f.ps.TriggerPush()
			wait(reused, "idle snapshot connection reuse")
			barrier()
			for _, conn := range []*errorSyncConn{activeConn, idleConn} {
				select {
				case <-conn.closed:
					t.Fatal("snapshot connection closed before Run exit")
				default:
				}
			}
			if cancelRun {
				cancel()
			} else {
				close(events)
			}
			wait(done, "Run exit")
			wait(idleConn.closed, "prompt idle connection closure")
			select {
			case <-activeConn.closed:
				t.Fatal("Run cleanup interrupted the independent exposed response")
			default:
			}
			if data, err := io.ReadAll(response.Body); err != nil || len(data) != 0 {
				t.Fatalf("independent response body = %q, error = %v, want empty successful body", data, err)
			}
			if err := response.Body.Close(); err != nil {
				t.Fatalf("close independent response: %v", err)
			}
			wait(activeConn.closed, "retired active connection closure after body release")
			wantDirect, wantReverse := int32(2), int32(0)
			if blocked {
				wantDirect, wantReverse = 1, 2
			}
			if f.accepted.Load() != 4 || f.directCalls.Load() != wantDirect || f.reverseCalls.Load() != wantReverse {
				t.Fatalf("Run cleanup: accepted=%d direct=%d reverse=%d, want 4/%d/%d", f.accepted.Load(), f.directCalls.Load(), f.reverseCalls.Load(), wantDirect, wantReverse)
			}
			assertErrorSnapshot(t, f.receiver, []ServiceError{localErr("a:one", "node-a", 1000)})
		})
	}
	test("direct context cancellation", false, true)
	test("reverse context cancellation", true, true)
	test("direct event channel closure", false, false)
	test("reverse event channel closure", true, false)
}

func TestErrorsSnapshotReadmissionRetiresPooledConnection(t *testing.T) {
	test := func(name string, blocked bool) {
		t.Run(name, func(t *testing.T) {
			f := newErrorSyncFixture(t)
			f.blocked.Store(blocked)
			f.connect(t, "11111111111111111111111111111111")
			f.sender.upsert(localErr("a:one", "node-a", 1000))
			f.ps.handleEvent(context.Background(), f.event)
			before, ok := f.ps.resolveRoute("uuid-b", noderec.ServiceErrors)
			if !ok {
				t.Fatal("initial route unavailable")
			}
			changeErrorSyncRecord(t, filepath.Join(f.ps.clusterDir, "trusted", "uuid-b.json"), "admissionEpoch", 2)
			after, ok := f.ps.resolveRoute("uuid-b", noderec.ServiceErrors)
			incarnation, admitted := f.ps.mesh.PeerServiceIncarnation("uuid-b")
			if !ok || !admitted || after.Incarnation != incarnation || after.Incarnation == before.Incarnation || after.Generation != before.Generation || after.Address != before.Address {
				t.Fatalf("readmission route: before=%+v after=%+v shared=%q admitted=%t/%t", before, after, incarnation, ok, admitted)
			}
			var reused atomic.Bool
			trace := &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) { reused.Store(info.Reused) }}
			f.ps.pushAll(httptrace.WithClientTrace(context.Background(), trace))
			wantReverse := int32(0)
			if blocked {
				wantReverse = 2
			}
			if reused.Load() || f.accepted.Load() != 2 || f.directCalls.Load() != 2 || f.reverseCalls.Load() != wantReverse {
				t.Fatalf("readmission heartbeat: reused=%t accepted=%d direct=%d reverse=%d, want false/2/2/%d", reused.Load(), f.accepted.Load(), f.directCalls.Load(), f.reverseCalls.Load(), wantReverse)
			}
		})
	}
	test("direct same certificate", false)
	test("reverse same certificate", true)
}
