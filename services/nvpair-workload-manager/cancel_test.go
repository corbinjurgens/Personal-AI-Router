// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
)

func cancelFrame(t *testing.T, p cancelParams) []byte {
	t.Helper()
	params, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	frame, err := json.Marshal(&Message{JSONRPC: "2.0", Method: MethodCancel, Params: params})
	if err != nil {
		t.Fatal(err)
	}
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
	first := cancelParams{OriginatedFrom: "uuid-self", WorkloadID: "1", Engine: "ollama", RunID: "r1", Regenerate: true, CancelID: "c1"}
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

func TestCancelRequiresCompleteTargetAtLocalAndPeerHops(t *testing.T) {
	self, peer := newPinnedPeerMeshes(t)
	emitted := 0
	srv := NewServer(0, newDedupIndex(64), self, func(*Workload) error { return nil }, func(string, string) error { return nil })
	srv.emitCancel = func(cancelParams) error { emitted++; return nil }
	post := serveEventsOverMTLS(t, srv, self, peer)
	test := func(name string, change func(*cancelParams)) {
		t.Run(name, func(t *testing.T) {
			p := cancelParams{OriginatedFrom: "uuid-self", WorkloadID: "7", Engine: "ollama", RunID: "r1", CancelID: "c1"}
			change(&p)
			data, err := json.Marshal(p)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := parseCancel(data); err == nil {
				t.Fatal("local parser accepted incomplete target")
			}
			if code := post(cancelFrame(t, p)); code != http.StatusBadRequest {
				t.Fatalf("peer accepted incomplete target: status=%d", code)
			}
			if emitted != 0 {
				t.Fatal("invalid target emitted to broker")
			}
		})
	}
	test("missing origin", func(p *cancelParams) { p.OriginatedFrom = "" })
	test("malformed origin", func(p *cancelParams) { p.OriginatedFrom = "wrong/origin" })
	test("missing run", func(p *cancelParams) { p.RunID = "" })
	test("malformed run", func(p *cancelParams) { p.RunID = "wrong/run" })
	test("missing engine", func(p *cancelParams) { p.Engine = "" })
	test("malformed engine", func(p *cancelParams) { p.Engine = "wrong/engine" })
}

func TestCancelDedupIncludesCompleteTarget(t *testing.T) {
	self, peer := newPinnedPeerMeshes(t)
	var got []cancelParams
	srv := NewServer(0, newDedupIndex(64), self, func(*Workload) error { return nil }, func(string, string) error { return nil })
	srv.emitCancel = func(p cancelParams) error { got = append(got, p); return nil }
	post := serveEventsOverMTLS(t, srv, self, peer)
	first := cancelParams{OriginatedFrom: "uuid-self", WorkloadID: "7", Engine: "ollama", RunID: "r1", CancelID: "c1"}
	otherEngine, otherRun, otherOrigin, otherWorkload := first, first, first, first
	otherEngine.Engine = "lmstudio"
	otherRun.RunID = "r2"
	otherOrigin.OriginatedFrom = "uuid-peer"
	otherWorkload.WorkloadID = "8"
	for _, p := range []cancelParams{first, otherEngine, otherRun, otherOrigin, otherWorkload, first} {
		if code := post(cancelFrame(t, p)); code != http.StatusOK {
			t.Fatalf("cancel status=%d", code)
		}
	}
	if len(got) != 5 {
		t.Fatalf("distinct full targets delivered=%d, want 5", len(got))
	}
	// The cancellation ID is opaque peer input, so delimiters in it cannot
	// alias any other target component.
	a, b := first, first
	a.WorkloadID, a.CancelID = "x", "y\x00z"
	b.WorkloadID, b.CancelID = "x\x00y", "z"
	if keyCancel(a) == keyCancel(b) {
		t.Fatal("dedup composition aliases distinct tuples")
	}
}

func TestCancelPeerRequiresCancellationID(t *testing.T) {
	self, peer := newPinnedPeerMeshes(t)
	emitted := 0
	srv := NewServer(0, newDedupIndex(64), self, func(*Workload) error { return nil }, func(string, string) error { return nil })
	srv.emitCancel = func(cancelParams) error { emitted++; return nil }
	post := serveEventsOverMTLS(t, srv, self, peer)
	for _, id := range []string{"", string(bytes.Repeat([]byte("c"), 129))} {
		p := cancelParams{OriginatedFrom: "uuid-self", WorkloadID: "7", Engine: "ollama", RunID: "r1", CancelID: id}
		if code := post(cancelFrame(t, p)); code != http.StatusBadRequest {
			t.Fatalf("invalid cancel ID status=%d", code)
		}
	}
	if emitted != 0 {
		t.Fatal("invalid cancellation ID emitted to broker")
	}
}

func TestCancelFanoutSaturationAndRelease(t *testing.T) {
	selfDir, peerDir := newPinnedPeerDirs(t)
	peer := startCancelPeer(t, clustertrust.Open(peerDir))
	var out bytes.Buffer
	m := NewManager(NewCodec(&out), 0, "uuid-self-host", selfDir)
	m.peers.Replace([]PeerNode{{ID: "origin-host", Host: peer.host, Addresses: []string{peer.host}, Port: peer.port, TXT: []string{clustertrust.ClusterUUIDTXTKey + "=uuid-peer"}}})
	params := cancelParams{OriginatedFrom: "origin-host", WorkloadID: "7", Engine: "ollama", RunID: "r1", CancelID: "c1"}
	frame := cancelFrame(t, params)
	data, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	for range 32 {
		cancelBroadcastAdmission <- struct{}{}
	}
	m.handleLocalCancel(&Message{Params: data})
	time.Sleep(50 * time.Millisecond)
	if peer.count() != 0 || len(cancelBroadcastAdmission) != 32 {
		t.Fatal("saturated local fanout launched work")
	}
	for range 32 {
		<-cancelBroadcastAdmission
	}
	for range 32 {
		cancelDeliveryAdmission <- struct{}{}
	}
	m.broadcaster.BroadcastTo(context.Background(), frame, func(target) bool { return true })
	if peer.count() != 0 || len(cancelDeliveryAdmission) != 32 {
		t.Fatal("saturated peer delivery launched work")
	}
	for range 32 {
		<-cancelDeliveryAdmission
	}
	m.handleLocalCancel(&Message{Params: data})
	deadline := time.Now().Add(5 * time.Second)
	for peer.count() != 1 || len(cancelBroadcastAdmission) != 0 || len(cancelDeliveryAdmission) != 0 {
		if time.Now().After(deadline) {
			t.Fatal("admitted fanout did not deliver and release all admission")
		}
		time.Sleep(time.Millisecond)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	m.ctx = ctx
	m.handleLocalCancel(&Message{Params: data})
	deadline = time.Now().Add(time.Second)
	for len(cancelBroadcastAdmission) != 0 || len(cancelDeliveryAdmission) != 0 {
		if time.Now().After(deadline) {
			t.Fatal("cancelled fanout retained admission")
		}
		time.Sleep(time.Millisecond)
	}
	if peer.count() != 1 {
		t.Fatal("cancelled fanout reached a peer")
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
	bystanderCert, bystanderKey := genLeaf(t, "uuid-bystander")
	selfCert, err := os.ReadFile(filepath.Join(selfDir, "node.crt"))
	if err != nil {
		t.Fatal(err)
	}
	bystanderDir := setupNode(t, bystanderCert, bystanderKey, map[string][]byte{"uuid-self": selfCert})
	pin, err := json.Marshal(workloadTestPin{"uuid-bystander", string(bystanderCert), "test-cluster", 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(selfDir, "trusted", "uuid-bystander.json"), pin, 0o600); err != nil {
		t.Fatal(err)
	}
	members, err := json.Marshal([]map[string]any{
		{"nodeUuid": "uuid-peer", "state": "member", "clusterId": "test-cluster", "admissionEpoch": 1},
		{"nodeUuid": "uuid-bystander", "state": "member", "clusterId": "test-cluster", "admissionEpoch": 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(selfDir, "members.json"), members, 0o600); err != nil {
		t.Fatal(err)
	}
	bystander := startCancelPeer(t, clustertrust.Open(bystanderDir))

	var out bytes.Buffer
	m := NewManager(NewCodec(&out), 0, "uuid-self-host", selfDir)
	txt := []string{clustertrust.ClusterUUIDTXTKey + "=uuid-peer"}
	m.peers.Replace([]PeerNode{
		{ID: "origin-host", Host: origin.host, Addresses: []string{origin.host}, Port: origin.port, TXT: txt},
		{ID: "bystander-host", Host: bystander.host, Addresses: []string{bystander.host}, Port: bystander.port, TXT: []string{clustertrust.ClusterUUIDTXTKey + "=uuid-bystander"}},
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

	params, err := json.Marshal(map[string]any{"originatedFrom": "origin-host", "workloadId": "5", "engine": "ollama", "runId": "r1", "regenerate": true})
	if err != nil {
		t.Fatal(err)
	}
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

	params, err = json.Marshal(map[string]any{"originatedFrom": "unknown-host", "workloadId": "6", "engine": "ollama", "runId": "r1"})
	if err != nil {
		t.Fatal(err)
	}
	m.handleMessage(&Message{JSONRPC: "2.0", Method: MethodCancel, Params: params})
	wait("cancel at every peer", func() bool { return origin.count() == 2 && bystander.count() == 1 })

	// Malformed local cancels are dropped, never sent.
	m.handleMessage(&Message{JSONRPC: "2.0", Method: MethodCancel, Params: json.RawMessage(`{"workloadId":"7"}`)})
	time.Sleep(100 * time.Millisecond)
	if origin.count() != 2 || bystander.count() != 1 {
		t.Fatal("a malformed cancel was sent")
	}
}
