// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/ipc"
	"nvpair-shared/noderec"
	"nvpair-shared/splitlisten"
)

type reverseProvenanceTestConn struct {
	net.Conn
	local, remote net.Addr
	closed        chan struct{}
	closeOnce     sync.Once
}

func (c *reverseProvenanceTestConn) LocalAddr() net.Addr  { return c.local }
func (c *reverseProvenanceTestConn) RemoteAddr() net.Addr { return c.remote }
func (c *reverseProvenanceTestConn) Close() error {
	err := c.Conn.Close()
	c.closeOnce.Do(func() { close(c.closed) })
	return err
}

func newReverseProvenanceManager(t *testing.T) *Manager {
	t.Helper()
	m := &Manager{reverse: &reverseState{}}
	t.Cleanup(func() {
		m.reverse.forwardMu.Lock()
		defer m.reverse.forwardMu.Unlock()
		for key, marker := range m.reverse.forwarded {
			marker.timer.Stop()
			delete(m.reverse.forwarded, key)
			marker.conn.Close()
		}
	})
	return m
}

func newReverseProvenancePair(t *testing.T, sourcePort, targetPort int) (*reverseProvenanceTestConn, net.Conn) {
	t.Helper()
	local, accepted := net.Pipe()
	source := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: sourcePort}
	target := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: targetPort}
	forward := &reverseProvenanceTestConn{Conn: local, local: source, remote: target, closed: make(chan struct{})}
	server := &reverseProvenanceTestConn{Conn: accepted, local: target, remote: source, closed: make(chan struct{})}
	t.Cleanup(func() {
		if err := forward.Close(); err != nil {
			t.Errorf("close forwarding fixture: %v", err)
		}
		if err := server.Close(); err != nil {
			t.Errorf("close accepted fixture: %v", err)
		}
	})
	return forward, server
}

func TestReverseForwardCloseBeforeConnContext(t *testing.T) {
	m := newReverseProvenanceManager(t)
	local, accepted := newReverseProvenancePair(t, 41001, 42001)
	m.reverse.forwardMu.Lock()
	forward := m.markReverseForward(local)
	m.reverse.forwardMu.Unlock()
	// Close completes before the listener handoff and HTTP context. Pending
	// I/O stops; Accept then binds provenance to the concrete connection.
	if err := forward.Close(); err != nil {
		t.Fatalf("close before context: %v", err)
	}
	select {
	case <-local.closed:
		t.Fatal("source socket released before the listener consumed provenance")
	default:
	}
	var one [1]byte
	if _, err := forward.Read(one[:]); err == nil {
		t.Fatal("Close left forwarding I/O usable")
	} else if timeout, ok := err.(net.Error); !ok || !timeout.Timeout() {
		t.Fatalf("pending close read = %v, want expired deadline", err)
	}
	accepted = m.acceptReverseForward(accepted)
	ctx := m.reverseConnContext(context.Background(), accepted)
	request := httptest.NewRequest(http.MethodPost, rosterPath, nil).WithContext(ctx)
	if !isReverseForward(request) {
		t.Fatal("close-before-context connection lost its forwarded classification")
	}
	select {
	case <-local.closed:
	default:
		t.Fatal("accepted handoff did not release the closed forwarding socket")
	}
	// Consumption is once-only, and retirement cannot change the captured
	// context later in the handler or classify a reused direct connection.
	_, direct := newReverseProvenancePair(t, 41001, 42001)
	if isReverseForward(httptest.NewRequest(http.MethodPost, rosterPath, nil).WithContext(m.reverseConnContext(context.Background(), m.acceptReverseForward(direct)))) {
		t.Fatal("a later connection inherited consumed provenance")
	}
	if !isReverseForward(request) {
		t.Fatal("handler context lost provenance after the socket closed")
	}
}

func TestReverseForwardUnconsumedMarkerExpires(t *testing.T) {
	m := newReverseProvenanceManager(t)
	local, _ := newReverseProvenancePair(t, 41002, 42002)
	m.reverse.forwardMu.Lock()
	m.markReverseForward(local)
	m.reverse.forwardMu.Unlock()
	// Even a forward that never calls Close must retire its unconsumed marker
	// and close the socket before that source address can be reused.
	select {
	case <-local.closed:
	case <-time.After(6 * time.Second):
		t.Fatal("unconsumed marker and source socket outlived the five-second setup bound")
	}
	m.reverse.forwardMu.Lock()
	remaining := len(m.reverse.forwarded)
	m.reverse.forwardMu.Unlock()
	if remaining != 0 {
		t.Fatalf("retained markers = %d, want 0", remaining)
	}
	// Reuse the exact source and destination tuple after expiry.
	_, direct := newReverseProvenancePair(t, 41002, 42002)
	ctx := m.reverseConnContext(context.Background(), m.acceptReverseForward(direct))
	if isReverseForward(httptest.NewRequest(http.MethodPost, rosterPath, nil).WithContext(ctx)) {
		t.Fatal("expired marker classified a later direct connection")
	}
}

func TestReverseForwardMarkerBindsDestination(t *testing.T) {
	m := newReverseProvenanceManager(t)
	local, forwarded := newReverseProvenancePair(t, 41003, 42003)
	_, direct := newReverseProvenancePair(t, 41003, 42004)
	m.reverse.forwardMu.Lock()
	m.markReverseForward(local)
	m.reverse.forwardMu.Unlock()
	if isReverseForward(httptest.NewRequest(http.MethodPost, rosterPath, nil).WithContext(m.reverseConnContext(context.Background(), m.acceptReverseForward(direct)))) {
		t.Fatal("same source at a different listener inherited provenance")
	}
	if !isReverseForward(httptest.NewRequest(http.MethodPost, rosterPath, nil).WithContext(m.reverseConnContext(context.Background(), m.acceptReverseForward(forwarded)))) {
		t.Fatal("the direct connection consumed the forwarded listener's marker")
	}
}

func TestReverseForwardOldRetirementPreservesReusedAddress(t *testing.T) {
	m := newReverseProvenanceManager(t)
	first, accepted := newReverseProvenancePair(t, 41004, 42004)
	m.reverse.forwardMu.Lock()
	old := m.markReverseForward(first).(*reverseForwardConn)
	m.reverse.forwardMu.Unlock()
	ctx := m.reverseConnContext(context.Background(), m.acceptReverseForward(accepted))
	if !isReverseForward(httptest.NewRequest(http.MethodPost, rosterPath, nil).WithContext(ctx)) {
		t.Fatal("first forwarded connection was not classified")
	}
	if err := old.Close(); err != nil {
		t.Fatal(err)
	}
	second, reused := newReverseProvenancePair(t, 41004, 42004)
	m.reverse.forwardMu.Lock()
	m.markReverseForward(second)
	m.reverse.forwardMu.Unlock()
	// A timer callback already queued before Stop must not retire a new
	// connection that now owns the same address tuple.
	m.reverse.retireReverseForward(old.key, old.marker)
	select {
	case <-second.closed:
		t.Fatal("old retirement closed the replacement connection")
	default:
	}
	if !isReverseForward(httptest.NewRequest(http.MethodPost, rosterPath, nil).WithContext(m.reverseConnContext(context.Background(), m.acceptReverseForward(reused)))) {
		t.Fatal("old retirement deleted replacement provenance")
	}
}

func TestReverseForwardResetTupleReuseBeforeConnContext(t *testing.T) {
	m := newReverseProvenanceManager(t)
	base, err := net.ListenTCP("tcp4", &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer base.Close()
	if err := base.SetDeadline(time.Now().Add(2 * time.Second)); err != nil {
		t.Fatalf("set accept deadline: %v", err)
	}
	var listener net.Listener = reverseForwardListener{Listener: base, manager: m}
	local, err := net.DialTCP("tcp4", nil, base.Addr().(*net.TCPAddr))
	if err != nil {
		t.Fatalf("dial original: %v", err)
	}
	defer local.Close()
	started := time.Now()
	m.reverse.forwardMu.Lock()
	forward := m.markReverseForward(local).(*reverseForwardConn)
	m.reverse.forwardMu.Unlock()
	accepted, err := listener.Accept()
	if err != nil {
		t.Fatalf("accept original: %v", err)
	}
	defer accepted.Close()
	// Reset the accepted socket before HTTP builds its context. Read the RST
	// on the still-open forwarding descriptor so tuple reuse needs no sleep.
	reset := accepted.(*reverseAcceptedConn).Conn.(*net.TCPConn)
	if err := reset.SetLinger(0); err != nil {
		t.Fatalf("set reset linger: %v", err)
	}
	if err := reset.Close(); err != nil {
		t.Fatalf("reset original: %v", err)
	}
	if err := local.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatalf("set reset read deadline: %v", err)
	}
	var one [1]byte
	if _, err := local.Read(one[:]); err == nil || errors.Is(err, io.EOF) {
		t.Fatalf("original read = %v, want TCP reset", err)
	} else if timeout, ok := err.(net.Error); ok && timeout.Timeout() {
		t.Fatalf("original read timed out instead of observing reset: %v", err)
	}
	// Keep the original descriptor open, exactly as the r2 pending Close did.
	replacement, err := net.DialTCP("tcp4", local.LocalAddr().(*net.TCPAddr), base.Addr().(*net.TCPAddr))
	if err != nil {
		t.Fatalf("reuse reset TCP tuple: %v", err)
	}
	defer replacement.Close()
	direct, err := listener.Accept()
	if err != nil {
		t.Fatalf("accept replacement: %v", err)
	}
	defer direct.Close()
	if direct.LocalAddr().String() != accepted.LocalAddr().String() || direct.RemoteAddr().String() != accepted.RemoteAddr().String() {
		t.Fatalf("replacement tuple = %s/%s, want %s/%s", direct.LocalAddr(), direct.RemoteAddr(), accepted.LocalAddr(), accepted.RemoteAddr())
	}
	if time.Since(started) >= reverseSetupTimeout {
		t.Fatal("tuple reuse occurred after the provenance setup bound")
	}
	if isReverseForward(httptest.NewRequest(http.MethodPost, rosterPath, nil).WithContext(m.reverseConnContext(context.Background(), direct))) {
		t.Fatal("same-tuple direct replacement inherited stale forwarded provenance")
	}
	if err := forward.Close(); err != nil {
		t.Fatalf("close original forward: %v", err)
	}
	// Run stale cleanup while the original accepted handoff is unconsumed.
	m.reverse.retireReverseForward(forward.key, forward.marker)
	if err := replacement.SetDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatalf("set replacement deadline: %v", err)
	}
	if _, err := replacement.Write([]byte("G")); err != nil {
		t.Fatalf("write replacement after stale cleanup: %v", err)
	}
	if err := direct.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatalf("set direct read deadline: %v", err)
	}
	if _, err := io.ReadFull(direct, one[:]); err != nil || one[0] != 'G' {
		t.Fatalf("replacement read = %q, %v, want G after stale cleanup", one, err)
	}
	if !isReverseForward(httptest.NewRequest(http.MethodPost, rosterPath, nil).WithContext(m.reverseConnContext(context.Background(), accepted))) {
		t.Fatal("original accepted connection lost provenance after reset and close")
	}
	if isReverseForward(httptest.NewRequest(http.MethodPost, rosterPath, nil).WithContext(m.reverseConnContext(context.Background(), accepted))) {
		t.Fatal("original accepted provenance was consumed twice")
	}
}

func TestReverseForwardContextRequiresAcceptedHandoff(t *testing.T) {
	m := newReverseProvenanceManager(t)
	local, accepted := newReverseProvenancePair(t, 41006, 42006)
	m.reverse.forwardMu.Lock()
	m.markReverseForward(local)
	m.reverse.forwardMu.Unlock()
	if isReverseForward(httptest.NewRequest(http.MethodPost, rosterPath, nil).WithContext(m.reverseConnContext(context.Background(), accepted))) {
		t.Fatal("context inferred provenance from endpoints without an accepted handoff")
	}
	if !isReverseForward(httptest.NewRequest(http.MethodPost, rosterPath, nil).WithContext(m.reverseConnContext(context.Background(), m.acceptReverseForward(accepted)))) {
		t.Fatal("raw context lookup consumed provenance before the accepted handoff")
	}
}

func TestReverseForwardAcceptedHandoffExpires(t *testing.T) {
	m := newReverseProvenanceManager(t)
	local, server := newReverseProvenancePair(t, 41005, 42005)
	m.reverse.forwardMu.Lock()
	m.markReverseForward(local)
	m.reverse.forwardMu.Unlock()
	accepted := m.acceptReverseForward(server)
	if _, ok := accepted.(*reverseAcceptedConn); !ok {
		t.Fatal("listener did not attach accepted provenance")
	}
	select {
	case <-local.closed:
	case <-time.After(6 * time.Second):
		t.Fatal("unconsumed accepted handoff retained its forwarding socket")
	}
	select {
	case <-server.(*reverseProvenanceTestConn).closed:
	case <-time.After(time.Second):
		t.Fatal("unconsumed accepted handoff retained its accepted socket")
	}
	_, direct := newReverseProvenancePair(t, 41005, 42005)
	if isReverseForward(httptest.NewRequest(http.MethodPost, rosterPath, nil).WithContext(m.reverseConnContext(context.Background(), m.acceptReverseForward(direct)))) {
		t.Fatal("expired accepted handoff classified a same-tuple direct connection")
	}
}

func TestReverseForwardProvenanceThroughSplit(t *testing.T) {
	test := func(name string, tlsMode bool) {
		t.Run(name, func(t *testing.T) {
			m := newReverseProvenanceManager(t)
			base, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatalf("listen: %v", err)
			}
			split := splitlisten.New(reverseForwardListener{Listener: base, manager: m})
			defer split.Close()
			m.reverse.forwardMu.Lock()
			local, err := net.DialTimeout("tcp", base.Addr().String(), time.Second)
			if err != nil {
				m.reverse.forwardMu.Unlock()
				t.Fatalf("dial: %v", err)
			}
			forward := m.markReverseForward(local)
			m.reverse.forwardMu.Unlock()
			defer forward.Close()
			first := byte('G')
			listener := split.Plain()
			if tlsMode {
				first = 0x16
				listener = tls.NewListener(split.TLS(), &tls.Config{})
			}
			if _, err := forward.Write([]byte{first}); err != nil {
				t.Fatalf("write split prefix: %v", err)
			}
			accepted, err := listener.Accept()
			if err != nil {
				t.Fatalf("accept split: %v", err)
			}
			defer accepted.Close()
			if err := forward.Close(); err != nil {
				t.Fatalf("close before wrapped context: %v", err)
			}
			ctx := m.reverseConnContext(context.Background(), accepted)
			if !isReverseForward(httptest.NewRequest(http.MethodPost, rosterPath, nil).WithContext(ctx)) {
				t.Fatal("split/TLS wrapping lost original accepted provenance")
			}
			if isReverseForward(httptest.NewRequest(http.MethodPost, rosterPath, nil).WithContext(m.reverseConnContext(context.Background(), accepted))) {
				t.Fatal("split/TLS provenance was consumed twice")
			}
		})
	}
	test("plain prefix", false)
	test("TLS and prefix", true)
}

func newReverseTestManager(t *testing.T) *Manager {
	t.Helper()
	base, err := os.MkdirTemp("", "pair-rv-")
	if err != nil {
		t.Fatalf("temp config: %v", err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(base); err != nil {
			t.Errorf("remove temp config: %v", err)
		}
	})
	manager, err := NewManager(NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), io.Discard}), base, 0)
	if err != nil {
		t.Fatalf("new manager: %v", err)
	}
	if _, err = manager.ensureAdmission("cluster-1"); err != nil {
		t.Fatalf("admission: %v", err)
	}
	manager.setClusterIdentity("cluster-1", "reverse tests")
	manager.addSelfMember()
	t.Cleanup(func() { manager.reverse.manager.Close() })
	return manager
}

func pairReverseManagers(t *testing.T, a, b *Manager) {
	t.Helper()
	pin := func(local, remote *Manager) {
		pinTrusted(t, local, remote.identity.NodeUUID, string(remote.identity.CertPEM), remote.identity.CertFingerprint)
		local.upsertMember(&ClusterNode{NodeUUID: remote.identity.NodeUUID, ID: remote.identity.NodeID, IPAddress: "127.0.0.1", Port: remote.port, ClusterID: "cluster-1", AdmissionEpoch: 1, State: stateMember})
	}
	pin(a, b)
	pin(b, a)
}

func setReverseServices(t *testing.T, m *Manager, generation uint64, services ...noderec.RegisterParams) {
	t.Helper()
	if services == nil {
		services = []noderec.RegisterParams{}
	}
	params, err := json.Marshal(localServicesSnapshot{generation, services})
	if err != nil {
		t.Fatalf("encode services: %v", err)
	}
	if err = m.applyLocalServices(params); err != nil {
		t.Fatalf("apply services: %v", err)
	}
}

func startReverseHTTP(t *testing.T, m *Manager, blocked bool) {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	m.port = listener.Addr().(*net.TCPAddr).Port
	if blocked {
		listener = rejectingPairingListener{Listener: listener}
	}
	ctx, cancel := context.WithCancel(context.Background())
	trusted := http.NewServeMux()
	trusted.HandleFunc(clustertrust.ReverseUpgradePath, m.handleReverseUpgrade)
	done := make(chan error, 1)
	go func() { done <- m.serveSplit(ctx, listener, http.NewServeMux(), trusted) }()
	t.Cleanup(func() {
		cancel()
		select {
		case err := <-done:
			if err != nil {
				t.Errorf("HTTP shutdown: %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Error("HTTP shutdown stalled")
		}
	})
}

func connectReverseManagers(t *testing.T, a, b *Manager) {
	t.Helper()
	node, ok := b.memberByNodeID(a.identity.NodeUUID)
	if !ok {
		t.Fatal("reachable peer missing")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := b.dialReverseSession(ctx, *node); err != nil {
		t.Fatalf("blocked node outbound upgrade: %v", err)
	}
	peer, ok := a.reversePeer(b.identity.NodeUUID)
	if !ok {
		t.Fatal("blocked peer not admitted")
	}
	deadline := time.Now().Add(3 * time.Second)
	for !a.reverse.manager.Has(peer) {
		if time.Now().After(deadline) {
			t.Fatal("reverse control did not become ready")
		}
		time.Sleep(time.Millisecond)
	}
}

func startPrivateReverse(t *testing.T, m *Manager) {
	t.Helper()
	if err := m.startReversePrivate(); err != nil {
		t.Fatalf("private listener: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	m.reverse.wg.Add(1)
	go m.runReversePrivate(ctx)
	t.Cleanup(func() { cancel(); m.reverse.listener.Close(); m.reverse.wg.Wait() })
}

func TestReverseServicesSnapshotsAreAtomic(t *testing.T) {
	m := newReverseTestManager(t)
	setReverseServices(t, m, 0)
	setReverseServices(t, m, 1, noderec.RegisterParams{Service: noderec.ServiceOllama, Port: 19001})
	cases := []struct {
		name   string
		params string
	}{
		{"conflicting generation", `{"generation":1,"services":[{"service":"ol","port":19002}]}`},
		{"bad port after valid entry", `{"generation":2,"services":[{"service":"ol","port":19002},{"service":"ni","port":0}]}`},
		{"unknown service", `{"generation":2,"services":[{"service":"gateway","port":19002}]}`},
		{"duplicate service", `{"generation":2,"services":[{"service":"ol","port":19002},{"service":"ol","port":19003}]}`},
		{"protocol override", `{"generation":2,"services":[{"service":"ol","port":19002,"protocol":"plain"}]}`},
		{"missing generation", `{"services":[]}`},
		{"missing services", `{"generation":2}`},
		{"trailing object", `{"generation":2,"services":[]} {}`},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			if err := m.applyLocalServices(json.RawMessage(test.params)); err == nil {
				t.Fatal("invalid snapshot accepted")
			}
			if m.reverse.generation != 1 || m.reverse.services[noderec.ServiceOllama].Port != 19001 {
				t.Fatal("rejected snapshot mutated registrations")
			}
		})
	}
	setReverseServices(t, m, 0, noderec.RegisterParams{Service: noderec.ServiceOllama, Port: 19999})
	if m.reverse.services[noderec.ServiceOllama].Port != 19001 {
		t.Fatal("old generation replaced live port")
	}
	setReverseServices(t, m, 2, noderec.RegisterParams{Service: noderec.ServiceOllama, Port: 19002})
	if m.reverse.services[noderec.ServiceOllama].Port != 19002 {
		t.Fatal("new generation did not rebind port")
	}
}

func TestReverseServicesRequireBrokerStdio(t *testing.T) {
	m := newReverseTestManager(t)
	m.localServicesAllowed = false
	if err := m.applyLocalServices(json.RawMessage(`{"generation":0,"services":[]}`)); err == nil {
		t.Fatal("non-stdio metadata accepted")
	}
}

func TestReversePeerRequiresCurrentAdmission(t *testing.T) {
	test := func(name string, change func(*Manager, *Manager)) {
		t.Run(name, func(t *testing.T) {
			a, b := newReverseTestManager(t), newReverseTestManager(t)
			pairReverseManagers(t, a, b)
			if _, ok := a.reversePeer(b.identity.NodeUUID); !ok {
				t.Fatal("paired fixture not admitted")
			}
			change(a, b)
			if _, ok := a.reversePeer(b.identity.NodeUUID); ok {
				t.Fatal("obsolete peer admitted")
			}
		})
	}
	test("pending member", func(a, b *Manager) {
		node, _ := a.memberByNodeID(b.identity.NodeUUID)
		node.State = statePendingInbound
		a.upsertMember(node)
	})
	test("missing member", func(a, b *Manager) { a.removeMemberByUUID(b.identity.NodeUUID) })
	test("stale member epoch", func(a, b *Manager) {
		node, _ := a.memberByNodeID(b.identity.NodeUUID)
		node.AdmissionEpoch++
		a.upsertMember(node)
	})
	test("different member cluster", func(a, b *Manager) {
		node, _ := a.memberByNodeID(b.identity.NodeUUID)
		node.ClusterID = "another"
		a.upsertMember(node)
	})
	test("forgotten pin", func(a, b *Manager) { a.trust.Forget(b.identity.NodeUUID) })
	test("teardown journal gate", func(a, b *Manager) { a.teardownPending.Store(true) })
	test("different local identity", func(a, b *Manager) { a.setClusterIdentity("another", "") })
}

func TestReverseBlockedDirectionConcurrentPrivateStreaming(t *testing.T) {
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	startPrivateReverse(t, a) // Before any HTTP goroutines narrow process umask.
	startReverseHTTP(t, a, false)
	startReverseHTTP(t, b, true)
	pairReverseManagers(t, a, b)
	node, _ := a.memberByNodeID(b.identity.NodeUUID)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := a.dialReverseSession(ctx, *node); err == nil {
		t.Fatal("blocked inbound listener accepted upgrade")
	}
	connectReverseManagers(t, a, b)
	payload := bytes.Repeat([]byte("reverse response\n"), 180000)
	service := http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, ok := b.verifyClientPin(r); !ok {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		w.Write(payload)
	})}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	go service.Serve(tls.NewListener(ln, b.buildServerTLSConfig()))
	t.Cleanup(func() { service.Close() })
	setReverseServices(t, b, 0, noderec.RegisterParams{Service: noderec.ServiceOllama, Port: ln.Addr().(*net.TCPAddr).Port})
	var wg sync.WaitGroup
	for range 3 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			conn, err := clustertrust.DialReverse(ctx, filepath.Dir(a.clusterDir), b.identity.NodeUUID, noderec.ServiceOllama)
			if err != nil {
				t.Errorf("private dial: %v", err)
				return
			}
			defer conn.Close()
			der, _ := a.trust.DER(b.identity.NodeUUID)
			inner := tls.Client(conn, clustertrust.ClientTLSConfig(a.identity.Cert, der))
			if err = inner.HandshakeContext(ctx); err != nil {
				t.Errorf("inner mTLS: %v", err)
				return
			}
			if _, err = io.WriteString(inner, "GET /stream HTTP/1.1\r\nHost: peer\r\nConnection: close\r\n\r\n"); err != nil {
				t.Errorf("request: %v", err)
				return
			}
			response, err := http.ReadResponse(bufio.NewReader(inner), &http.Request{Method: http.MethodGet})
			if err != nil {
				t.Errorf("response: %v", err)
				return
			}
			defer response.Body.Close()
			body, err := io.ReadAll(response.Body)
			if err != nil {
				t.Errorf("stream body: %v", err)
				return
			}
			if response.StatusCode != http.StatusOK || !bytes.Equal(body, payload) {
				t.Errorf("bad streamed response: status %d length %d", response.StatusCode, len(body))
			}
		}()
	}
	wg.Wait()
	t.Logf("verified response bytes per concurrent stream: %d", len(payload))
}

func TestReversePeerServiceClientPrivateHTTPSFallback(t *testing.T) {
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	startPrivateReverse(t, a)
	startReverseHTTP(t, a, false)
	startReverseHTTP(t, b, true)
	pairReverseManagers(t, a, b)
	connectReverseManagers(t, a, b)
	var dispatched atomic.Int32
	service := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if uuid, ok := b.verifyClientPin(r); !ok || uuid != a.identity.NodeUUID {
			t.Error("private tunnel did not preserve the pinned inner client certificate")
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		if r.TLS.ServerName != "original.example" || r.URL.Path != "/original" || r.URL.RawQuery != "a=1" {
			t.Errorf("original TLS/URL changed: SNI=%q URL=%s", r.TLS.ServerName, r.URL)
		}
		dispatched.Add(1)
		if _, err := io.WriteString(w, "private pinned response"); err != nil {
			t.Error(err)
		}
	}))
	service.TLS = b.buildServerTLSConfig()
	service.StartTLS()
	defer service.Close()
	port := service.Listener.Addr().(*net.TCPAddr).Port
	setReverseServices(t, b, 1, noderec.RegisterParams{Service: noderec.ServiceOllama, Port: port})
	address := net.JoinHostPort("original.example", strconv.Itoa(port))
	var attempts atomic.Int32
	adapter := clustertrust.NewPeerServiceClient(clustertrust.Open(a.clusterDir), clustertrust.PeerServiceClientOptions{
		PeerClientOptions: clustertrust.PeerClientOptions{Timeout: time.Second, ResolvePin: a.trust.DER},
		BaseConfigDir:     filepath.Dir(a.clusterDir), DirectTimeout: 60 * time.Millisecond,
		ResolveRoute: func(uuid string, service noderec.ServiceKey) (clustertrust.PeerServiceRoute, bool) {
			peer, ok := a.reversePeer(uuid)
			return clustertrust.PeerServiceRoute{Incarnation: peer.Incarnation, Address: address, Protocol: clustertrust.ReverseTLS, Generation: 1}, ok && service == noderec.ServiceOllama
		},
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			attempts.Add(1)
			<-ctx.Done()
			return nil, ctx.Err()
		},
	})
	client := adapter.Client(b.identity.NodeUUID, noderec.ServiceOllama)
	for range 2 {
		response, err := client.Get("https://" + address + "/original?a=1")
		if err != nil {
			t.Fatal(err)
		}
		body, err := io.ReadAll(response.Body)
		closeErr := response.Body.Close()
		if err != nil || closeErr != nil || response.StatusCode != http.StatusOK || string(body) != "private pinned response" {
			t.Fatalf("private HTTPS response: status=%d body=%q read=%v close=%v", response.StatusCode, body, err, closeErr)
		}
	}
	if attempts.Load() != 1 || dispatched.Load() != 2 {
		t.Fatalf("cached route direct=%d HTTP=%d, want 1 and 2", attempts.Load(), dispatched.Load())
	}
}

func TestReversePeerServiceClientCancellationAfterResponse(t *testing.T) {
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	startPrivateReverse(t, a)
	startReverseHTTP(t, a, false)
	pairReverseManagers(t, a, b)
	connectReverseManagers(t, a, b)
	stopped := make(chan struct{})
	service := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, ok := b.verifyClientPin(r); !ok {
			t.Error("missing pinned client")
			w.WriteHeader(http.StatusForbidden)
			return
		}
		if _, err := io.WriteString(w, "start"); err != nil {
			t.Error(err)
			return
		}
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		close(stopped)
	}))
	service.TLS = b.buildServerTLSConfig()
	service.StartTLS()
	defer service.Close()
	port := service.Listener.Addr().(*net.TCPAddr).Port
	setReverseServices(t, b, 1, noderec.RegisterParams{Service: noderec.ServiceOllama, Port: port})
	address := net.JoinHostPort("original.example", strconv.Itoa(port))
	adapter := clustertrust.NewPeerServiceClient(clustertrust.Open(a.clusterDir), clustertrust.PeerServiceClientOptions{
		PeerClientOptions: clustertrust.PeerClientOptions{ResolvePin: a.trust.DER}, BaseConfigDir: filepath.Dir(a.clusterDir),
		ResolveRoute: func(uuid string, service noderec.ServiceKey) (clustertrust.PeerServiceRoute, bool) {
			peer, ok := a.reversePeer(uuid)
			return clustertrust.PeerServiceRoute{Incarnation: peer.Incarnation, Address: address, Protocol: clustertrust.ReverseTLS, Generation: 1}, ok && service == noderec.ServiceOllama
		},
		DialContext: func(context.Context, string, string) (net.Conn, error) { return nil, errors.New("direct blocked") },
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://"+address, nil)
	if err != nil {
		t.Fatal(err)
	}
	response, err := adapter.Client(b.identity.NodeUUID, noderec.ServiceOllama).Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var initial [5]byte
	if _, err := io.ReadFull(response.Body, initial[:]); err != nil || string(initial[:]) != "start" {
		t.Fatalf("initial response=%q err=%v", initial, err)
	}
	cancel()
	if _, err := io.ReadAll(response.Body); !errors.Is(err, context.Canceled) {
		t.Fatalf("response cancellation = %v", err)
	}
	select {
	case <-stopped:
	case <-time.After(time.Second):
		t.Fatal("retained private lifecycle did not cancel remote HTTP")
	}
}

func TestReverseProtocolGateBeforeLoopback(t *testing.T) {
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	startReverseHTTP(t, a, false)
	pairReverseManagers(t, a, b)
	connectReverseManagers(t, a, b)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	var accepts atomic.Int64
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			accepts.Add(1)
			conn.Close()
		}
	}()
	keys := []noderec.ServiceKey{"cl", "ol", "lm", "lc", "em", "ec", "er", "wl"}
	services := make([]noderec.RegisterParams, 0, len(keys))
	for _, key := range keys {
		services = append(services, noderec.RegisterParams{Service: key, Port: ln.Addr().(*net.TCPAddr).Port})
	}
	setReverseServices(t, b, 0, services...)
	peer, _ := a.reversePeer(b.identity.NodeUUID)
	for _, key := range keys {
		t.Run(string(key), func(t *testing.T) {
			for _, input := range [][]byte{[]byte("POST /cluster/reverse/dial HTTP/1.1\r\n\r\n"), {0x16, 3, 3, 0, 4, 1, 0, 0, 0}} {
				ctx, cancel := context.WithTimeout(context.Background(), time.Second)
				stream, err := a.reverse.manager.Open(ctx, peer, key)
				if err != nil {
					cancel()
					t.Fatal(err)
				}
				if _, err = stream.Write(input); err != nil {
					stream.Close()
					cancel()
					t.Fatal(err)
				}
				var reply [1]byte
				_, err = stream.Read(reply[:])
				stream.Close()
				cancel()
				if err == nil {
					t.Fatal("invalid inner protocol accepted")
				}
			}
		})
	}
	if accepts.Load() != 0 {
		t.Fatalf("invalid protocol reached loopback %d times", accepts.Load())
	}
	ln.Close()
	<-done
}

func TestReverseTrustChangesCloseActiveStreams(t *testing.T) {
	test := func(name string, revoke func(*testing.T, *Manager, *Manager)) {
		t.Run(name, func(t *testing.T) {
			a, b := newReverseTestManager(t), newReverseTestManager(t)
			startReverseHTTP(t, a, false)
			pairReverseManagers(t, a, b)
			connectReverseManagers(t, a, b)
			// Register only plain node-info, then use an ordinary loopback listener.
			ln, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			defer ln.Close()
			accepted := make(chan net.Conn, 1)
			go func() {
				conn, err := ln.Accept()
				if err == nil {
					accepted <- conn
				}
			}()
			setReverseServices(t, b, 0, noderec.RegisterParams{Service: "ni", Port: ln.Addr().(*net.TCPAddr).Port})
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			peer, _ := a.reversePeer(b.identity.NodeUUID)
			stream, err := a.reverse.manager.Open(ctx, peer, "ni")
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			var local net.Conn
			select {
			case local = <-accepted:
				defer local.Close()
			case <-ctx.Done():
				t.Fatal("loopback not accepted")
			}
			revoke(t, a, b)
			stream.SetReadDeadline(time.Now().Add(time.Second))
			var data [1]byte
			if _, err = stream.Read(data[:]); err == nil {
				t.Fatal("revoked stream stayed usable")
			}
			local.SetReadDeadline(time.Now().Add(time.Second))
			if _, err = local.Read(data[:]); err == nil {
				t.Fatal("revoked loopback stayed usable")
			} else if timeout, ok := err.(net.Error); ok && timeout.Timeout() {
				t.Fatal("revocation did not close remote forwarder")
			}
		})
	}
	test("in-memory Forget", func(t *testing.T, a, b *Manager) { a.trust.Forget(b.identity.NodeUUID) })
	test("member readmission", func(t *testing.T, a, b *Manager) {
		node, _ := a.memberByNodeID(b.identity.NodeUUID)
		node.AdmissionEpoch++
		a.upsertMember(node)
	})
	test("local identity change", func(t *testing.T, a, b *Manager) { a.setClusterIdentity("another", "") })
	test("leave", func(t *testing.T, a, b *Manager) {
		if err := a.teardownClusterLocal(); err != nil {
			t.Fatal(err)
		}
	})
}

func TestReversePrivateCancellationClosesRemote(t *testing.T) {
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	startPrivateReverse(t, a)
	startReverseHTTP(t, a, false)
	pairReverseManagers(t, a, b)
	connectReverseManagers(t, a, b)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	accepted := make(chan net.Conn, 1)
	go func() {
		conn, err := ln.Accept()
		if err == nil {
			accepted <- conn
		}
	}()
	setReverseServices(t, b, 0, noderec.RegisterParams{Service: "ni", Port: ln.Addr().(*net.TCPAddr).Port})
	ctx, cancel := context.WithCancel(context.Background())
	conn, err := clustertrust.DialReverse(ctx, filepath.Dir(a.clusterDir), b.identity.NodeUUID, "ni")
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	defer conn.Close()
	var local net.Conn
	select {
	case local = <-accepted:
		defer local.Close()
	case <-time.After(2 * time.Second):
		cancel()
		t.Fatal("no remote dial")
	}
	cancel()
	local.SetReadDeadline(time.Now().Add(2 * time.Second))
	var one [1]byte
	if _, err = local.Read(one[:]); err == nil {
		t.Fatal("canceled remote stream still readable")
	} else if timeout, ok := err.(net.Error); ok && timeout.Timeout() {
		t.Fatal("cancellation did not propagate to remote")
	}
}

func TestReverseUnknownServiceUnavailable(t *testing.T) {
	m := newReverseTestManager(t)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, err := clustertrust.DialReverse(ctx, filepath.Dir(m.clusterDir), "stranger", "gateway")
	var unavailable *clustertrust.ReverseUnavailableError
	if !errors.As(err, &unavailable) || unavailable.PeerUUID != "stranger" {
		t.Fatalf("expected typed unavailable, got %v", err)
	}
}

func TestReversePrivateHalfClosePreservesDelayedResponse(t *testing.T) {
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	startPrivateReverse(t, a)
	startReverseHTTP(t, a, false)
	pairReverseManagers(t, a, b)
	connectReverseManagers(t, a, b)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	done := make(chan error, 1)
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			done <- err
			return
		}
		defer conn.Close()
		body, err := io.ReadAll(conn)
		if err != nil {
			done <- err
			return
		}
		if string(body) != "request" {
			done <- errors.New("wrong EOF-framed request")
			return
		}
		time.Sleep(30 * time.Millisecond)
		_, err = io.WriteString(conn, "delayed response")
		done <- err
	}()
	setReverseServices(t, b, 0, noderec.RegisterParams{Service: "ni", Port: ln.Addr().(*net.TCPAddr).Port})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	conn, err := clustertrust.DialReverse(ctx, filepath.Dir(a.clusterDir), b.identity.NodeUUID, "ni")
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if _, err = io.WriteString(conn, "request"); err != nil {
		t.Fatal(err)
	}
	half, ok := conn.(interface{ CloseWrite() error })
	if !ok {
		t.Fatal("private dial lacks half-close")
	}
	if err = half.CloseWrite(); err != nil {
		t.Fatal(err)
	}
	body, err := io.ReadAll(conn)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != "delayed response" {
		t.Fatalf("response %q, want delayed response", body)
	}
	if err = <-done; err != nil {
		t.Fatal(err)
	}
}

func TestReversePrivateListenerPreservesOtherInstance(t *testing.T) {
	m := newReverseTestManager(t)
	if err := m.startReversePrivate(); err != nil {
		t.Fatal(err)
	}
	defer m.reverse.listener.Close()
	second, err := NewManager(NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), io.Discard}), filepath.Dir(m.clusterDir), 0)
	if err != nil {
		t.Fatal(err)
	}
	defer second.reverse.manager.Close()
	if err = second.startReversePrivate(); err == nil {
		second.reverse.listener.Close()
		t.Fatal("second instance stole private endpoint")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m.reverse.wg.Add(1)
	go m.runReversePrivate(ctx)
	defer func() { cancel(); m.reverse.listener.Close(); m.reverse.wg.Wait() }()
	conn, err := ipc.Dial(clustertrust.ReverseDialPath(filepath.Dir(m.clusterDir)))
	if err != nil {
		t.Fatalf("original private endpoint removed: %v", err)
	}
	conn.Close()
}

func TestReverseResolvedTargetRejectsRebind(t *testing.T) {
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	pairReverseManagers(t, a, b)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	port := ln.Addr().(*net.TCPAddr).Port
	setReverseServices(t, b, 1, noderec.RegisterParams{Service: "ni", Port: port})
	peer, ok := b.reversePeer(a.identity.NodeUUID)
	if !ok {
		t.Fatal("paired peer not admitted")
	}
	target, err := b.resolveReverseService(context.Background(), peer, "ni")
	if err != nil {
		t.Fatal(err)
	}
	setReverseServices(t, b, 2, noderec.RegisterParams{Service: "ni", Port: port + 1})
	if conn, err := target.Dial(context.Background()); err == nil {
		conn.Close()
		t.Fatal("resolved stale target dialed after rebind")
	}
	if err = ln.(*net.TCPListener).SetDeadline(time.Now().Add(30 * time.Millisecond)); err != nil {
		t.Fatal(err)
	}
	if conn, err := ln.Accept(); err == nil {
		conn.Close()
		t.Fatal("old listener accessed after metadata rebind")
	}
}

func TestReversePrivateShutdownClosesIncompleteHeader(t *testing.T) {
	m := newReverseTestManager(t)
	if err := m.startReversePrivate(); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m.reverse.wg.Add(1)
	go m.runReversePrivate(ctx)
	conn, err := ipc.Dial(clustertrust.ReverseDialPath(filepath.Dir(m.clusterDir)))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if _, err = io.WriteString(conn, `{"peerUuid":`); err != nil {
		t.Fatal(err)
	}
	cancel()
	joined := make(chan struct{})
	go func() { m.reverse.wg.Wait(); close(joined) }()
	select {
	case <-joined:
	case <-time.After(time.Second):
		t.Fatal("private shutdown did not join incomplete header")
	}
	var reply [1]byte
	if _, err = conn.Read(reply[:]); err == nil {
		t.Fatal("shutdown retained incomplete private request")
	}
}

func TestReversePrivateDeadlineClosesRemote(t *testing.T) {
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	startPrivateReverse(t, a)
	startReverseHTTP(t, a, false)
	pairReverseManagers(t, a, b)
	connectReverseManagers(t, a, b)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	accepted := make(chan net.Conn, 1)
	go func() {
		conn, err := ln.Accept()
		if err == nil {
			accepted <- conn
		}
	}()
	setReverseServices(t, b, 0, noderec.RegisterParams{Service: "ni", Port: ln.Addr().(*net.TCPAddr).Port})
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	conn, err := clustertrust.DialReverse(ctx, filepath.Dir(a.clusterDir), b.identity.NodeUUID, "ni")
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	var local net.Conn
	select {
	case local = <-accepted:
		defer local.Close()
	case <-ctx.Done():
		t.Fatal("no loopback dial before deadline")
	}
	local.SetReadDeadline(time.Now().Add(time.Second))
	var one [1]byte
	if _, err = local.Read(one[:]); err == nil {
		t.Fatal("deadline left remote stream usable")
	} else if timeout, ok := err.(net.Error); ok && timeout.Timeout() {
		t.Fatal("deadline did not propagate to remote")
	}
}

func TestReverseTunnelCannotReachPrivateDialOverHTTP(t *testing.T) {
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	startPrivateReverse(t, b)
	startReverseHTTP(t, a, false)
	startReverseHTTP(t, b, false)
	pairReverseManagers(t, a, b)
	connectReverseManagers(t, a, b)
	setReverseServices(t, b, 0, noderec.RegisterParams{Service: "cl", Port: b.port})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	peer, _ := a.reversePeer(b.identity.NodeUUID)
	conn, err := a.reverse.manager.Open(ctx, peer, "cl")
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	der, _ := a.trust.DER(b.identity.NodeUUID)
	inner := tls.Client(conn, clustertrust.ClientTLSConfig(a.identity.Cert, der))
	if err = inner.HandshakeContext(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err = io.WriteString(inner, "POST /cluster/reverse/dial HTTP/1.1\r\nHost: peer\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"); err != nil {
		t.Fatal(err)
	}
	response, err := http.ReadResponse(bufio.NewReader(inner), &http.Request{Method: http.MethodPost})
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusNotFound {
		t.Fatalf("private operation exposed over tunneled HTTP: status %d", response.StatusCode)
	}
	b.reverse.mu.RLock()
	calls := len(b.reverse.calls)
	b.reverse.mu.RUnlock()
	if calls != 0 {
		t.Fatal("tunneled HTTP created a private dial")
	}
}

func TestReversePrivateCloseSurvivesFullHeaderSlots(t *testing.T) {
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	startPrivateReverse(t, a)
	startReverseHTTP(t, a, false)
	pairReverseManagers(t, a, b)
	connectReverseManagers(t, a, b)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	accepted := make(chan net.Conn, 1)
	go func() {
		conn, err := ln.Accept()
		if err == nil {
			accepted <- conn
		}
	}()
	setReverseServices(t, b, 0, noderec.RegisterParams{Service: "ni", Port: ln.Addr().(*net.TCPAddr).Port})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := clustertrust.DialReverse(ctx, filepath.Dir(a.clusterDir), b.identity.NodeUUID, "ni")
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	select {
	case conn := <-accepted:
		defer conn.Close()
	case <-time.After(time.Second):
		t.Fatal("application stream not forwarded")
	}
	stalled := make([]io.ReadWriteCloser, 0, 254)
	defer func() {
		for _, conn := range stalled {
			conn.Close()
		}
	}()
	for range 254 {
		conn, err := ipc.Dial(clustertrust.ReverseDialPath(filepath.Dir(a.clusterDir)))
		if err != nil {
			t.Fatal(err)
		}
		stalled = append(stalled, conn)
		if _, err = io.WriteString(conn, `{"callId":`); err != nil {
			t.Fatal(err)
		}
	}
	deadline := time.Now().Add(time.Second)
	for len(a.reverse.privateSlots) != 256 {
		if time.Now().After(deadline) {
			t.Fatalf("header slots %d, want exactly 256 before Close", len(a.reverse.privateSlots))
		}
		time.Sleep(time.Millisecond)
	}
	start := time.Now()
	if err = stream.Close(); err != nil {
		t.Fatal(err)
	}
	closeElapsed := time.Since(start)
	if closeElapsed > 200*time.Millisecond {
		t.Fatal("caller Close blocked behind saturated listener")
	}
	deadline = time.Now().Add(time.Second)
	for {
		a.reverse.mu.RLock()
		calls := len(a.reverse.calls)
		a.reverse.mu.RUnlock()
		if calls == 0 && len(a.reverse.privateSlots) == 254 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("Close did not release application and lifecycle resources: calls=%d slots=%d", calls, len(a.reverse.privateSlots))
		}
		time.Sleep(time.Millisecond)
	}
	t.Logf("private header slots saturated at 256; caller Close returned in %s", closeElapsed)
}

func TestReversePrivateRequiresLifecycleBeforeApplicationOpen(t *testing.T) {
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	startPrivateReverse(t, a)
	startReverseHTTP(t, a, false)
	pairReverseManagers(t, a, b)
	connectReverseManagers(t, a, b)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	setReverseServices(t, b, 0, noderec.RegisterParams{Service: "ni", Port: ln.Addr().(*net.TCPAddr).Port})
	conn, err := ipc.Dial(clustertrust.ReverseDialPath(filepath.Dir(a.clusterDir)))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	request := clustertrust.ReverseDialRequest{PeerUUID: b.identity.NodeUUID, Service: "ni", CallID: "0123456789abcdef0123456789abcdef", DeadlineUnixNano: time.Now().Add(100 * time.Millisecond).UnixNano()}
	if err = json.NewEncoder(conn).Encode(request); err != nil {
		t.Fatal(err)
	}
	var reply [1]byte
	if _, err = conn.Read(reply[:]); err == nil {
		t.Fatal("application acknowledged without lifecycle attachment")
	}
	if err = ln.(*net.TCPListener).SetDeadline(time.Now().Add(30 * time.Millisecond)); err != nil {
		t.Fatal(err)
	}
	if conn, err := ln.Accept(); err == nil {
		conn.Close()
		t.Fatal("remote application opened without lifecycle attachment")
	}
	deadline := time.Now().Add(time.Second)
	for {
		a.reverse.mu.RLock()
		calls := len(a.reverse.calls)
		a.reverse.mu.RUnlock()
		if calls == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("failed attachment retained call")
		}
		time.Sleep(time.Millisecond)
	}
}

func TestReversePrivateShutdownJoinsLifecycleWaiter(t *testing.T) {
	m := newReverseTestManager(t)
	if err := m.startReversePrivate(); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m.reverse.wg.Add(1)
	go m.runReversePrivate(ctx)
	conn, err := ipc.Dial(clustertrust.ReverseDialPath(filepath.Dir(m.clusterDir)))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if err = json.NewEncoder(conn).Encode(clustertrust.ReverseDialRequest{CallID: "0123456789abcdef0123456789abcdef", Watch: true}); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for {
		stack := make([]byte, 128*1024)
		n := runtime.Stack(stack, true)
		if strings.Contains(string(stack[:n]), "serveReverseWatch") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("private lifecycle did not reach attachment waiter")
		}
		time.Sleep(time.Millisecond)
	}
	cancel()
	joined := make(chan struct{})
	go func() { m.reverse.wg.Wait(); close(joined) }()
	select {
	case <-joined:
	case <-time.After(time.Second):
		t.Fatal("shutdown did not join lifecycle attachment waiter")
	}
	var reply [1]byte
	if _, err = conn.Read(reply[:]); err == nil {
		t.Fatal("shutdown left unmatched lifecycle connection open")
	}
}
