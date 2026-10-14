// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
)

// Repeated roster heartbeats reuse the same authority-bound mTLS connection.
func TestReconcile_ReusesPeerConnections(t *testing.T) {
	f := newPeerConnectionFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(`{"clusterId":"cluster-1"}`)); err != nil {
			t.Error(err)
		}
	}))
	const rounds = 5
	for i := 0; i < rounds; i++ {
		if outcome, _ := f.m.reconcileWith([]string{f.address}, f.peerUUID); outcome != reconcileAccepted {
			t.Fatalf("round %d: outcome = %v, want accepted", i, outcome)
		}
	}
	if f.newConns.Load() != 1 || f.closedConns.Load() != 0 {
		t.Fatalf("connections new=%d closed=%d, want 1 open reusable connection", f.newConns.Load(), f.closedConns.Load())
	}
}

type peerConnectionFixture struct {
	m           *Manager
	peerUUID    string
	address     string
	newConns    atomic.Int32
	closedConns atomic.Int32
	closed      chan struct{}
}

func newPeerConnectionFixture(t *testing.T, handler http.Handler) *peerConnectionFixture {
	t.Helper()
	m := newReverseTestManager(t)
	t.Cleanup(m.peerServices.CloseIdle)
	t.Cleanup(m.removalPeerServices.CloseIdle)
	t.Cleanup(m.clients.CloseIdle)

	peerUUID, err := newUUIDv4()
	if err != nil {
		t.Fatal(err)
	}
	certPEM, keyPEM, err := generateLeaf(peerUUID, "node-peer")
	if err != nil {
		t.Fatal(err)
	}
	fp, ferr := certFingerprintFromPEM(certPEM)
	if ferr != nil {
		t.Fatal(ferr)
	}
	pinTrusted(t, m, peerUUID, string(certPEM), fp)

	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatal(err)
	}

	f := &peerConnectionFixture{m: m, peerUUID: peerUUID, closed: make(chan struct{}, 1)}
	ts := httptest.NewUnstartedServer(handler)
	ts.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			f.newConns.Add(1)
		}
		if state == http.StateClosed {
			f.closedConns.Add(1)
			select {
			case f.closed <- struct{}{}:
			default:
			}
		}
	}
	ts.TLS = clustertrust.ServerTLSConfig(cert)
	ts.StartTLS()
	t.Cleanup(ts.Close)

	addr := ts.Listener.Addr().String()
	m.upsertMember(&ClusterNode{NodeUUID: peerUUID, ID: "peer", IPAddress: "127.0.0.1", Port: ts.Listener.Addr().(*net.TCPAddr).Port, ClusterID: "cluster-1", AdmissionEpoch: 1, State: stateMember})
	f.address = addr
	return f
}

func TestNotifyPeerRemovalUsesFreshConnections(t *testing.T) {
	var requests atomic.Int32
	f := newPeerConnectionFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != membersRemovePath {
			t.Errorf("request = %s %s, want POST %s", r.Method, r.URL.Path, membersRemovePath)
		}
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Error(err)
		}
		requests.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	for range 2 {
		f.m.notifyPeerRemoval(f.address, f.peerUUID, RemovalProof{})
		select {
		case <-f.closed:
		case <-time.After(time.Second):
			t.Fatal("removal connection was left idle")
		}
	}
	if requests.Load() != 2 || f.newConns.Load() != 2 || f.closedConns.Load() != 2 {
		t.Fatalf("removals=%d connections new=%d closed=%d, want 2 of each", requests.Load(), f.newConns.Load(), f.closedConns.Load())
	}
}

func TestManagerRunClosesIdleRosterConnections(t *testing.T) {
	f := newPeerConnectionFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if _, err := w.Write([]byte(`{"clusterId":"cluster-1"}`)); err != nil {
			t.Error(err)
		}
	}))
	if outcome, _ := f.m.reconcileWith([]string{f.address}, f.peerUUID); outcome != reconcileAccepted {
		t.Fatalf("outcome = %v, want accepted", outcome)
	}
	if f.newConns.Load() != 1 || f.closedConns.Load() != 0 {
		t.Fatalf("before Run: new=%d closed=%d, want 1/0", f.newConns.Load(), f.closedConns.Load())
	}
	if err := f.m.Run(context.Background()); err != nil {
		t.Fatalf("Run on control EOF: %v", err)
	}
	select {
	case <-f.closed:
	case <-time.After(time.Second):
		t.Fatal("Run left roster connection idle")
	}
	if f.newConns.Load() != 1 || f.closedConns.Load() != 1 {
		t.Fatalf("after Run: new=%d closed=%d, want 1/1", f.newConns.Load(), f.closedConns.Load())
	}
}

func TestManagerRunPreservesActiveRosterResponse(t *testing.T) {
	release := make(chan struct{})
	f := newPeerConnectionFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Length", "2")
		if _, err := w.Write([]byte("[")); err != nil {
			t.Error(err)
			return
		}
		w.(http.Flusher).Flush()
		<-release
		if _, err := w.Write([]byte("]")); err != nil {
			t.Error(err)
		}
	}))
	// Release the handler before the fixture's server cleanup, even on failure.
	defer close(release)
	client, err := f.m.peerClient(f.peerUUID)
	if err != nil {
		t.Fatal(err)
	}
	response, err := client.Post("https://"+f.address+rosterPath, "application/json", bytes.NewReader([]byte("{}")))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if err := f.m.Run(context.Background()); err != nil {
		t.Fatalf("Run on control EOF: %v", err)
	}
	if f.newConns.Load() != 1 || f.closedConns.Load() != 0 {
		t.Fatalf("Run closed active response: new=%d closed=%d, want 1/0", f.newConns.Load(), f.closedConns.Load())
	}
	release <- struct{}{}
	body, err := io.ReadAll(response.Body)
	if err != nil || string(body) != "[]" {
		t.Fatalf("response after Run = %q, err=%v, want complete []", body, err)
	}
}

// TestPeerClient_ForgetRevokesWithPinStillOnDisk covers a failed durable pin
// delete during pairing rollback. Forget is the live TrustStore revocation; the
// leftover file must not let Mesh or the connection pool authorize outbound
// traffic after inbound traffic already rejects the peer.
func TestPeerClient_ForgetRevokesWithPinStillOnDisk(t *testing.T) {
	m := newTestManager(t)
	defer m.clients.CloseIdle()

	peerUUID, err := newUUIDv4()
	if err != nil {
		t.Fatal(err)
	}
	certPEM, _, err := generateLeaf(peerUUID, "node-peer")
	if err != nil {
		t.Fatal(err)
	}
	fp, err := certFingerprintFromPEM(certPEM)
	if err != nil {
		t.Fatal(err)
	}
	pinTrusted(t, m, peerUUID, string(certPEM), fp)
	m.upsertMember(&ClusterNode{NodeUUID: peerUUID, ID: "peer", IPAddress: "127.0.0.1", Port: 14321, ClusterID: "cluster-1", AdmissionEpoch: 1, State: stateMember})

	if _, err := m.peerClient(peerUUID); err != nil {
		t.Fatalf("pinned peer must yield a client: %v", err)
	}
	pinPath := m.trust.pinPath(peerUUID)
	if _, err := os.Stat(pinPath); err != nil {
		t.Fatalf("stat pin before Forget: %v", err)
	}

	m.trust.Forget(peerUUID)

	if _, err := os.Stat(pinPath); err != nil {
		t.Fatalf("Forget must leave the durable pin in place: %v", err)
	}
	if _, ok := m.trust.DER(peerUUID); ok {
		t.Fatal("Forget must remove the TrustStore authorization")
	}
	if _, err := m.peerClient(peerUUID); err == nil {
		t.Fatal("forgotten peer yielded a client because Mesh re-read the leftover pin")
	}
}
