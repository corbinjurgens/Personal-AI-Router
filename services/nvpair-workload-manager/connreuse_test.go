// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"

	"nvpair-shared/clustertrust"
)

// TestBroadcast_ReusesPeerConnections guards the field failure where a fresh
// mTLS handshake per lifecycle event exhausted connections and lost terminal
// events, leaving stale running workloads. Several rounds must use one bounded
// connection; the receiver, rather than the transport, owns mutation dedup.
func TestBroadcast_ReusesPeerConnections(t *testing.T) {
	selfDir, peerDir := newPinnedPeerDirs(t)
	self, peer := clustertrust.Open(selfDir), clustertrust.Open(peerDir)

	var mu sync.Mutex
	newConns := 0
	closedConns := 0
	requests := 0

	mux := http.NewServeMux()
	mux.HandleFunc(eventsPath, func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		requests++
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	})
	ts := httptest.NewUnstartedServer(mux)
	ts.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			mu.Lock()
			newConns++
			mu.Unlock()
		}
		if state == http.StateClosed {
			mu.Lock()
			closedConns++
			mu.Unlock()
		}
	}
	// The peer is the server here: we are the pinned caller dialing it.
	ts.TLS = peer.ServerTLSConfig()
	ts.StartTLS()
	defer ts.Close()

	host, portStr, err := net.SplitHostPort(ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatal(err)
	}

	peers := newPeerSet()
	peers.Replace([]PeerNode{{
		ID:        "peer-1",
		Host:      host,
		Addresses: []string{host},
		Port:      port,
		TXT:       []string{clustertrust.ClusterUUIDTXTKey + "=uuid-peer"},
	}})

	b := NewBroadcaster(peers, self, selfDir)
	defer b.clients.CloseIdle()

	const rounds = 5
	for i := 0; i < rounds; i++ {
		b.Broadcast(context.Background(), []byte(`{"jsonrpc":"2.0","method":"workload:started","params":{}}`))
	}

	mu.Lock()
	opened, closed, delivered := newConns, closedConns, requests
	mu.Unlock()
	if opened != 1 || closed != 0 || delivered != rounds {
		t.Fatalf("connections opened=%d closed=%d requests=%d, want 1, 0, %d (handshakes per event lose terminal events)", opened, closed, delivered, rounds)
	}
}

// TestInterNodeServer_ReapsIdleConnections: the listener half of connection
// reuse. Without an IdleTimeout an idle peer keep-alive pins a file descriptor
// for the life of the process. The listener's lifetime must also be strictly
// longer than the calling pool's, so the client is always the side that discards
// a doubtful connection rather than picking one this side is closing.
func TestInterNodeServer_ReapsIdleConnections(t *testing.T) {
	srv := newInterNodeServer(http.NewServeMux())
	if srv.IdleTimeout != clustertrust.PeerListenerIdleTimeout {
		t.Errorf("IdleTimeout = %v, want %v", srv.IdleTimeout, clustertrust.PeerListenerIdleTimeout)
	}
	if clustertrust.PeerListenerIdleTimeout <= clustertrust.PeerIdleTimeout {
		t.Errorf("listener idle (%v) must exceed the client's (%v), or the client can pick a connection the server is closing",
			clustertrust.PeerListenerIdleTimeout, clustertrust.PeerIdleTimeout)
	}
	if srv.ReadHeaderTimeout != interNodeReadHeaderTimeout {
		t.Errorf("ReadHeaderTimeout = %v, want %v", srv.ReadHeaderTimeout, interNodeReadHeaderTimeout)
	}
}
