// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
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

// errorSyncFixture has node-a (subscriber, PeerSync) stream node-b's (peer)
// errors, directly or over a reverse session.
type errorSyncFixture struct {
	subscriber, peer *Manager
	subscriberRW     *captureRW
	peerPub          *errorsPublisher
	ps               *PeerSync
	server           *httptest.Server
	local, remote    *clustertrust.ReverseManager
	event            DiscoveryEvent
	expectedHost     string
	expectedSNI      string
	blocked          atomic.Bool
	directCalls      atomic.Int32
	reverseCalls     atomic.Int32
	streams          atomic.Int32 // stream responses that passed the pin gate
	beforeDial       func()
	onConnect        func(*errorSyncConn)
}

type errorSyncConn struct {
	net.Conn
	closed    chan struct{}
	closeOnce sync.Once
}

func (c *errorSyncConn) Close() error {
	err := c.Conn.Close()
	c.closeOnce.Do(func() { close(c.closed) })
	return err
}

func (f *errorSyncFixture) trackConnection(conn net.Conn) *errorSyncConn {
	c := &errorSyncConn{Conn: conn, closed: make(chan struct{})}
	if f.onConnect != nil {
		f.onConnect(c)
	}
	return c
}

// streamStatusWriter reports the first status a handler writes.
type streamStatusWriter struct {
	http.ResponseWriter
	once   sync.Once
	status func(int)
}

func (w *streamStatusWriter) note(code int) { w.once.Do(func() { w.status(code) }) }

func (w *streamStatusWriter) WriteHeader(code int) {
	w.note(code)
	w.ResponseWriter.WriteHeader(code)
}

func (w *streamStatusWriter) Write(data []byte) (int, error) {
	w.note(http.StatusOK)
	return w.ResponseWriter.Write(data)
}

func (w *streamStatusWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

// setErrorsStreamRetry sets the first reconnect delay for one test.
func setErrorsStreamRetry(t *testing.T, d time.Duration) {
	t.Helper()
	previous := errorsStreamRetryMin
	errorsStreamRetryMin = d
	t.Cleanup(func() { errorsStreamRetryMin = previous })
}

func newErrorSyncFixture(t *testing.T) *errorSyncFixture {
	t.Helper()
	return newErrorSyncFixtureRetry(t, 20*time.Millisecond)
}

// newErrorSyncFixtureRetry sets the first reconnect delay before any stream
// starts; its restore runs after the streams have stopped.
func newErrorSyncFixtureRetry(t *testing.T, retry time.Duration) *errorSyncFixture {
	t.Helper()
	shortenErrorsStreamKeepalive(t, 50*time.Millisecond)
	setErrorsStreamRetry(t, retry)
	aCert, aKey := genLeaf(t, "node-a")
	bCert, bKey := genLeaf(t, "node-b")
	dirA := setupNode(t, aCert, aKey, map[string][]byte{"node-b": bCert})
	dirB := setupNode(t, bCert, bKey, map[string][]byte{"node-a": aCert})
	aMesh, bMesh := clustertrust.Open(dirA), clustertrust.Open(dirB)
	f := &errorSyncFixture{subscriberRW: newCaptureRW(), peer: managerForNode("node-b"), peerPub: newErrorsPublisher()}
	f.subscriber = NewManager(NewCodec(f.subscriberRW))
	f.subscriber.localNodeID = "node-a"
	f.peerPub.publish(f.peer)
	f.peer.SetOnLocalChange(f.publishPeer)
	f.ps = NewPeerSync(f.subscriber, aMesh, dirA)
	mux := newErrorsMux(f.peer, bMesh, f.peerPub)
	f.server = httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != errorsStreamPath {
			t.Errorf("stream request: method=%s path=%s", r.Method, r.URL.Path)
		}
		if r.Host != f.expectedHost || r.TLS == nil || r.TLS.ServerName != f.expectedSNI {
			t.Errorf("stream authority: Host=%q TLS=%+v", r.Host, r.TLS)
		}
		mux.ServeHTTP(&streamStatusWriter{ResponseWriter: w, status: func(code int) {
			if code == http.StatusOK {
				f.streams.Add(1)
			}
		}}, r)
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
		ID: "node-b", Host: "Peer B", Addresses: []string{"192.0.2.10"}, Port: portNumber, TXT: []string{"cluster-uuid=node-b"},
	}}
	f.expectedHost, f.expectedSNI = net.JoinHostPort("192.0.2.10", port), ""
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
	f.local, f.remote = manager("node-a", aMesh), manager("node-b", bMesh)
	f.ps.clients = clustertrust.NewPeerServiceClient(aMesh, clustertrust.PeerServiceClientOptions{
		BaseConfigDir: filepath.Dir(dirA), ResolveRoute: f.ps.resolveRoute,
		DirectTimeout: 100 * time.Millisecond, ProbeInterval: time.Hour,
		DialContext: func(ctx context.Context, _, address string) (net.Conn, error) {
			f.directCalls.Add(1)
			if address != net.JoinHostPort("192.0.2.10", port) {
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
			if base != filepath.Dir(dirA) || uuid != "node-b" || service != noderec.ServiceErrors {
				return nil, errors.New("unexpected private dial destination")
			}
			conn, err := f.local.Open(ctx, clustertrust.ReversePeer{UUID: uuid, Incarnation: "fixture-admission"}, service)
			if err != nil {
				return nil, err
			}
			return f.trackConnection(conn), nil
		},
	})
	// Registered last, so streams end before the reverse sessions and the
	// server close (cleanups run in reverse order).
	t.Cleanup(f.ps.stopStreams)
	return f
}

func (f *errorSyncFixture) connect(t *testing.T, id string) {
	t.Helper()
	left, right := net.Pipe()
	if err := f.local.Register(clustertrust.ReversePeer{UUID: "node-b", Incarnation: "fixture-admission"}, left, "node-a", id); err != nil {
		right.Close()
		t.Fatalf("register local reverse session: %v", err)
	}
	if err := f.remote.Register(clustertrust.ReversePeer{UUID: "node-a", Incarnation: "fixture-admission"}, right, "node-a", id); err != nil {
		t.Fatalf("register remote reverse session: %v", err)
	}
}

// publishPeer publishes the peer's current local-origin snapshot, as its
// onLocalChange hook does.
func (f *errorSyncFixture) publishPeer() { f.peerPub.publish(f.peer) }

func assertErrorSnapshot(t *testing.T, mgr *Manager, want []ServiceError) {
	t.Helper()
	if got := mgr.snapshot(); !slices.Equal(got, want) {
		t.Fatalf("snapshot = %+v, want %+v", got, want)
	}
}

// waitFor polls cond for up to 5 s.
func waitFor(t *testing.T, description string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", description)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func waitErrorSnapshot(t *testing.T, mgr *Manager, want []ServiceError) {
	t.Helper()
	waitFor(t, "converged snapshot", func() bool { return slices.Equal(mgr.snapshot(), want) })
}

// settle gives a stream that should do nothing time to misbehave.
func settle() { time.Sleep(300 * time.Millisecond) }

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
			f.peer.upsert(localErr("b:one", "node-b", 1000))
			f.peer.upsert(localErr("b:two", "node-b", 1000))
			f.peer.ReconcilePeer("node-c", []ServiceError{localErr("foreign", "node-c", 1000)})
			f.publishPeer()
			f.subscriber.upsert(localErr("a:keep", "node-a", 1000))
			f.subscriber.ReconcilePeer("node-b", []ServiceError{localErr("stale", "node-b", 1000)})
			f.subscriber.ReconcilePeer("node-c", []ServiceError{localErr("c:keep", "node-c", 1000)})
			f.ps.handleEvent(context.Background(), f.event)
			waitErrorSnapshot(t, f.subscriber, []ServiceError{
				localErr("a:keep", "node-a", 1000), localErr("b:one", "node-b", 1000),
				localErr("b:two", "node-b", 1000), localErr("c:keep", "node-c", 1000),
			})
			if !f.peer.clearByID("b:one") {
				t.Fatal("clear local error did not change peer")
			}
			f.publishPeer()
			waitErrorSnapshot(t, f.subscriber, []ServiceError{
				localErr("a:keep", "node-a", 1000), localErr("b:two", "node-b", 1000), localErr("c:keep", "node-c", 1000),
			})
			// The change arrived on the open stream, not on a new connection.
			if got := f.streams.Load(); got != 1 {
				t.Fatalf("streams = %d, want 1", got)
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

func TestErrorsSnapshotUnavailableFailsClosed(t *testing.T) {
	test := func(name string, wantDirect, wantReverse int32, prepare func(*testing.T, *errorSyncFixture)) {
		t.Run(name, func(t *testing.T) {
			// One attempt only, so the dial counts are exact.
			f := newErrorSyncFixtureRetry(t, time.Hour)
			f.blocked.Store(true)
			f.peer.upsert(localErr("b:one", "node-b", 1000))
			f.publishPeer()
			prepare(t, f)
			f.ps.handleEvent(context.Background(), f.event)
			settle()
			assertErrorSnapshot(t, f.subscriber, nil)
			if got := f.streams.Load(); got != 0 {
				t.Fatalf("unavailable peer served %d streams, want 0", got)
			}
			if direct, reverse := f.directCalls.Load(), f.reverseCalls.Load(); direct != wantDirect || reverse != wantReverse {
				t.Fatalf("unavailable dial counts: direct=%d reverse=%d, want %d/%d", direct, reverse, wantDirect, wantReverse)
			}
		})
	}
	test("missing reverse session", 1, 1, func(t *testing.T, f *errorSyncFixture) {})
	test("missing pin", 0, 0, func(t *testing.T, f *errorSyncFixture) {
		if err := os.Remove(filepath.Join(f.ps.clusterDir, "trusted", "node-b.json")); err != nil {
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
		changeErrorSyncRecord(t, filepath.Join(f.ps.clusterDir, "trusted", "node-b.json"), "clusterId", "other-cluster")
	})
	test("invalid pin", 1, 1, func(t *testing.T, f *errorSyncFixture) {
		f.connect(t, "11111111111111111111111111111111")
		wrongCert, _ := genLeaf(t, "node-b")
		changeErrorSyncRecord(t, filepath.Join(f.ps.clusterDir, "trusted", "node-b.json"), "certPem", string(wrongCert))
	})
	test("pin revoked during setup", 1, 0, func(t *testing.T, f *errorSyncFixture) {
		f.blocked.Store(false)
		f.connect(t, "11111111111111111111111111111111")
		f.beforeDial = func() {
			if err := os.Remove(filepath.Join(f.ps.clusterDir, "trusted", "node-b.json")); err != nil {
				t.Error(err)
			}
		}
	})
}

func TestErrorsSnapshotAuthorityChangesDuringSetup(t *testing.T) {
	test := func(name string, change func(*testing.T, *errorSyncFixture)) {
		t.Run(name, func(t *testing.T) {
			// One attempt only: a later attempt under the changed authority is a
			// new, correctly authorized stream.
			f := newErrorSyncFixtureRetry(t, time.Hour)
			f.connect(t, "11111111111111111111111111111111")
			f.beforeDial = func() { change(t, f) }
			f.peer.upsert(localErr("b:one", "node-b", 1000))
			f.publishPeer()
			f.ps.handleEvent(context.Background(), f.event)
			settle()
			assertErrorSnapshot(t, f.subscriber, nil)
			if streams, reverse := f.streams.Load(), f.reverseCalls.Load(); streams != 0 || reverse != 0 {
				t.Fatalf("changed authority opened a stream: streams=%d reverse=%d, want 0/0", streams, reverse)
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
		changeErrorSyncRecord(t, filepath.Join(f.ps.clusterDir, "trusted", "node-b.json"), "admissionEpoch", 2)
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
	f.peer.upsert(localErr("b:one", "node-b", 1000))
	f.publishPeer()
	f.ps.handleEvent(ctx, f.event)
	settle()
	assertErrorSnapshot(t, f.subscriber, nil)
	if got := f.reverseCalls.Load(); got != 0 {
		t.Fatalf("canceled setup attempted %d reverse dials, want 0", got)
	}
	if got := f.directCalls.Load(); got != 1 {
		t.Fatalf("canceled stream retried: direct dials = %d, want 1", got)
	}
}

func TestErrorsSnapshotAddressRecovery(t *testing.T) {
	f := newErrorSyncFixture(t)
	f.event.Node.Host = ""
	f.event.Node.TXT = []string{"cluster-uuid=node-b", "ip=192.0.2.1"}
	f.event.Node.Addresses = []string{"127.0.0.1"}
	f.expectedHost = net.JoinHostPort("127.0.0.1", strconv.Itoa(f.event.Node.Port))
	f.expectedSNI = ""
	// Use the production constructor and direct dialer, proving the selected
	// endpoint is what request authorization accepts.
	f.ps = NewPeerSync(f.subscriber, f.ps.mesh, f.ps.clusterDir)
	t.Cleanup(f.ps.stopStreams)
	f.ps.addrs = reach.NewChooser(reach.WithDial(func(_, address string, _ time.Duration) (net.Conn, error) {
		if address != f.expectedHost {
			return nil, errors.New("address blocked")
		}
		return net.Dial("tcp", f.server.Listener.Addr().String())
	}))
	f.peer.upsert(localErr("b:one", "node-b", 1000))
	f.publishPeer()
	f.ps.handleEvent(context.Background(), f.event)
	waitErrorSnapshot(t, f.subscriber, []ServiceError{localErr("b:one", "node-b", 1000)})
	route, ok := f.ps.resolveRoute("node-b", noderec.ServiceErrors)
	if !ok || route.Address != f.expectedHost {
		t.Fatalf("resolved route = %+v, admitted=%t, want chosen loopback endpoint", route, ok)
	}
	settle()
	if got := f.streams.Load(); got != 1 {
		t.Fatalf("streams = %d, want 1 (the selected address kept its stream)", got)
	}
}

func TestErrorsSnapshotDuplicatePrincipalFailsClosed(t *testing.T) {
	test := func(name string, resolve func(*testing.T, *errorSyncFixture, DiscoveryEvent)) {
		t.Run(name, func(t *testing.T) {
			f := newErrorSyncFixture(t)
			f.peer.upsert(localErr("b:one", "node-b", 1000))
			f.publishPeer()
			f.ps.handleEvent(context.Background(), f.event)
			waitErrorSnapshot(t, f.subscriber, []ServiceError{localErr("b:one", "node-b", 1000)})
			duplicate := f.event
			duplicate.Node.ID = "node-b-alias"
			f.ps.handleEvent(context.Background(), duplicate)
			// Both advertisements are present, even if only the first endpoint has
			// been selected. Repeated resolution must never pick a map entry, and
			// the open stream ends at its next line.
			for range 20 {
				if route, ok := f.ps.resolveRoute("node-b", noderec.ServiceErrors); ok || route != (clustertrust.PeerServiceRoute{}) {
					t.Fatalf("ambiguous principal resolved: %+v admitted=%t", route, ok)
				}
			}
			settle()
			if streams, direct, reverse := f.streams.Load(), f.directCalls.Load(), f.reverseCalls.Load(); streams != 1 || direct != 1 || reverse != 0 {
				t.Fatalf("ambiguous principal streamed: streams=%d direct=%d reverse=%d, want 1/1/0", streams, direct, reverse)
			}
			resolve(t, f, duplicate)
			waitFor(t, "a stream for the resolved principal", func() bool { return f.streams.Load() == 2 })
			route, ok := f.ps.resolveRoute("node-b", noderec.ServiceErrors)
			if !ok || route.Address != f.expectedHost {
				t.Fatalf("resolved duplicate: route=%+v admitted=%t, want current endpoint", route, ok)
			}
			waitErrorSnapshot(t, f.subscriber, []ServiceError{localErr("b:one", "node-b", 1000)})
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

func TestPeerSyncRunExitEndsStreams(t *testing.T) {
	test := func(name string, blocked, cancelRun bool) {
		t.Run(name, func(t *testing.T) {
			f := newErrorSyncFixture(t)
			f.blocked.Store(blocked)
			f.connect(t, "11111111111111111111111111111111")
			connections := make(chan *errorSyncConn, 4)
			f.onConnect = func(conn *errorSyncConn) { connections <- conn }
			ctx, cancel := context.WithCancel(context.Background())
			events := make(chan DiscoveryEvent)
			done := make(chan struct{})
			go func() {
				defer close(done)
				f.ps.Run(ctx, events)
			}()
			t.Cleanup(func() {
				cancel()
				<-done
			})
			f.peer.upsert(localErr("b:one", "node-b", 1000))
			f.publishPeer()
			select {
			case events <- f.event:
			case <-time.After(time.Second):
				t.Fatal("Run did not receive discovery")
			}
			waitErrorSnapshot(t, f.subscriber, []ServiceError{localErr("b:one", "node-b", 1000)})
			var conn *errorSyncConn
			select {
			case conn = <-connections:
			case <-time.After(time.Second):
				t.Fatal("no stream connection")
			}
			if cancelRun {
				cancel()
			} else {
				close(events)
			}
			for _, signal := range []<-chan struct{}{done, conn.closed} {
				select {
				case <-signal:
				case <-time.After(2 * time.Second):
					t.Fatal("Run exit left the stream open")
				}
			}
			wantDirect, wantReverse := int32(1), int32(0)
			if blocked {
				wantDirect, wantReverse = 1, 1
			}
			if f.streams.Load() != 1 || f.directCalls.Load() != wantDirect || f.reverseCalls.Load() != wantReverse {
				t.Fatalf("Run exit: streams=%d direct=%d reverse=%d, want 1/%d/%d", f.streams.Load(), f.directCalls.Load(), f.reverseCalls.Load(), wantDirect, wantReverse)
			}
			// Entries stay when a stream ends; only discovery removal evicts.
			assertErrorSnapshot(t, f.subscriber, []ServiceError{localErr("b:one", "node-b", 1000)})
		})
	}
	test("direct context cancellation", false, true)
	test("reverse context cancellation", true, true)
	test("direct event channel closure", false, false)
	test("reverse event channel closure", true, false)
}

func TestErrorsSnapshotReadmissionRestartsStream(t *testing.T) {
	test := func(name string, blocked bool) {
		t.Run(name, func(t *testing.T) {
			f := newErrorSyncFixture(t)
			f.blocked.Store(blocked)
			f.connect(t, "11111111111111111111111111111111")
			f.peer.upsert(localErr("b:one", "node-b", 1000))
			f.publishPeer()
			f.ps.handleEvent(context.Background(), f.event)
			waitErrorSnapshot(t, f.subscriber, []ServiceError{localErr("b:one", "node-b", 1000)})
			before, ok := f.ps.resolveRoute("node-b", noderec.ServiceErrors)
			if !ok {
				t.Fatal("initial route unavailable")
			}
			changeErrorSyncRecord(t, filepath.Join(f.ps.clusterDir, "trusted", "node-b.json"), "admissionEpoch", 2)
			members := mustJSON(t, []map[string]any{{"nodeUuid": "node-b", "state": "member", "clusterId": "test-cluster", "admissionEpoch": 2}})
			if err := os.WriteFile(filepath.Join(f.ps.clusterDir, "members.json"), members, 0o600); err != nil {
				t.Fatal(err)
			}
			after, ok := f.ps.resolveRoute("node-b", noderec.ServiceErrors)
			incarnation, admitted := f.ps.mesh.PeerServiceIncarnation("node-b")
			if !ok || !admitted || after.Incarnation != incarnation || after.Incarnation == before.Incarnation || after.Generation != before.Generation || after.Address != before.Address {
				t.Fatalf("readmission route: before=%+v after=%+v shared=%q admitted=%t/%t", before, after, incarnation, ok, admitted)
			}
			// The old stream ends at its next line and a new one opens on a
			// fresh connection under the new incarnation.
			waitFor(t, "a stream under the new incarnation", func() bool { return f.streams.Load() == 2 })
			wantReverse := int32(0)
			if blocked {
				wantReverse = 2
			}
			if f.directCalls.Load() != 2 || f.reverseCalls.Load() != wantReverse {
				t.Fatalf("readmission: direct=%d reverse=%d, want 2/%d", f.directCalls.Load(), f.reverseCalls.Load(), wantReverse)
			}
		})
	}
	test("direct same certificate", false)
	test("reverse same certificate", true)
}
