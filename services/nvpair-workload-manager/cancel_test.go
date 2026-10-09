// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
)

func cancelFrame(t *testing.T, p cancelParams) []byte {
	t.Helper()
	params, _ := json.Marshal(p)
	frame, _ := json.Marshal(&Message{JSONRPC: "2.0", Method: MethodCancel, Params: params})
	return frame
}

func TestServerCancelValidatesAndDedupsByCancelID(t *testing.T) {
	self, peer := newPinnedPeerMeshes(t)
	var got []cancelParams
	fail := true
	srv := NewServer(0, newDedupIndex(64), self, func(*Workload) error { return nil }, func(string, string) error { return nil })
	srv.emitCancel = func(p cancelParams) error {
		if fail {
			return errors.New("broker gone")
		}
		got = append(got, p)
		return nil
	}
	post := serveEventsOverMTLS(t, srv, self, peer)

	if code := post(cancelFrame(t, cancelParams{WorkloadID: "1"})); code != http.StatusBadRequest {
		t.Fatalf("cancel without origin = %d, want 400", code)
	}
	first := cancelParams{OriginatedFrom: "uuid-self", WorkloadID: "1", Regenerate: true, CancelID: "c1"}
	if code := post(cancelFrame(t, first)); code != http.StatusInternalServerError {
		t.Fatalf("failed broker write = %d, want 500 so the sender retries", code)
	}
	fail = false
	if code := post(cancelFrame(t, first)); code != http.StatusOK {
		t.Fatalf("retry = %d", code)
	}
	if code := post(cancelFrame(t, first)); code != http.StatusOK {
		t.Fatalf("redelivery = %d", code)
	}
	second := first
	second.CancelID = "c2"
	if code := post(cancelFrame(t, second)); code != http.StatusOK {
		t.Fatalf("second cancel = %d", code)
	}
	if len(got) != 2 || got[0] != first || got[1] != second {
		t.Fatalf("broker saw %+v; want the first cancel once and the second once", got)
	}
}

// cancelPeer is one peer's events endpoint that records the cancels it hands
// to its broker.
type cancelPeer struct {
	mu   sync.Mutex
	seen []cancelParams
	port int
	host string
}

func (c *cancelPeer) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.seen)
}

func startCancelPeer(t *testing.T, peerMesh *clustertrust.Mesh) *cancelPeer {
	t.Helper()
	c := &cancelPeer{}
	srv := NewServer(0, newDedupIndex(64), peerMesh, func(*Workload) error { return nil }, func(string, string) error { return nil })
	srv.emitCancel = func(p cancelParams) error {
		c.mu.Lock()
		c.seen = append(c.seen, p)
		c.mu.Unlock()
		return nil
	}
	mux := http.NewServeMux()
	mux.HandleFunc(eventsPath, srv.handleEvents)
	ts := httptest.NewUnstartedServer(mux)
	ts.TLS = peerMesh.ServerTLSConfig()
	ts.StartTLS()
	t.Cleanup(ts.Close)
	host, portText, _ := net.SplitHostPort(ts.Listener.Addr().String())
	c.host = host
	c.port, _ = strconv.Atoi(portText)
	return c
}

func TestLocalCancelGoesToTheOriginOrEveryPeer(t *testing.T) {
	selfDir, peerDir := newPinnedPeerDirs(t)
	peerMesh := clustertrust.Open(peerDir)
	origin := startCancelPeer(t, peerMesh)
	bystander := startCancelPeer(t, peerMesh)

	var out bytes.Buffer
	m := NewManager(NewCodec(&out), 0, "uuid-self-host", selfDir)
	t.Cleanup(m.broadcaster.CloseIdle)
	txt := []string{clustertrust.ClusterUUIDTXTKey + "=uuid-peer"}
	m.peers.Replace([]PeerNode{
		{ID: "origin-host", Host: origin.host, Addresses: []string{origin.host}, Port: origin.port, TXT: txt},
		{ID: "bystander-host", Host: bystander.host, Addresses: []string{bystander.host}, Port: bystander.port, TXT: txt},
	})
	wait := func(what string, pred func() bool) {
		t.Helper()
		deadline := time.Now().Add(5 * time.Second)
		for !pred() {
			if time.Now().After(deadline) {
				t.Fatalf("timed out waiting for %s", what)
			}
			time.Sleep(10 * time.Millisecond)
		}
	}

	params, _ := json.Marshal(map[string]any{"originatedFrom": "origin-host", "workloadId": "5", "regenerate": true})
	m.handleMessage(&Message{JSONRPC: "2.0", Method: MethodCancel, Params: params})
	wait("cancel at the origin", func() bool { return origin.count() == 1 })
	time.Sleep(100 * time.Millisecond)
	if bystander.count() != 0 {
		t.Fatal("a cancel for a known origin also went to a bystander")
	}
	origin.mu.Lock()
	got := origin.seen[0]
	origin.mu.Unlock()
	if got.WorkloadID != "5" || !got.Regenerate || got.CancelID == "" {
		t.Fatalf("origin received %+v", got)
	}

	params, _ = json.Marshal(map[string]any{"originatedFrom": "unknown-host", "workloadId": "6"})
	m.handleMessage(&Message{JSONRPC: "2.0", Method: MethodCancel, Params: params})
	wait("cancel at every peer", func() bool { return origin.count() == 2 && bystander.count() == 1 })

	// Malformed local cancels are dropped, never sent.
	m.handleMessage(&Message{JSONRPC: "2.0", Method: MethodCancel, Params: json.RawMessage(`{"workloadId":"7"}`)})
	time.Sleep(100 * time.Millisecond)
	if origin.count() != 2 || bystander.count() != 1 {
		t.Fatal("a malformed cancel was sent")
	}
}
