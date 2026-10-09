// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/noderec"
)

// syncBuffer is a goroutine-safe writer for a Manager's codec output.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) Read(p []byte) (int, error) { return 0, nil }

func (b *syncBuffer) frames() []Message {
	b.mu.Lock()
	defer b.mu.Unlock()
	var out []Message
	scanner := bufio.NewScanner(bytes.NewReader(b.buf.Bytes()))
	for scanner.Scan() {
		var m Message
		if json.Unmarshal(scanner.Bytes(), &m) == nil {
			out = append(out, m)
		}
	}
	return out
}

type policyCall struct {
	method, caller string
	params         policyBody
}

// policyPeerServer starts a peer's ec surface whose broker relay is recorded.
func policyPeerServer(t *testing.T, peerDir string) (*httptest.Server, func() []policyCall) {
	t.Helper()
	var mu sync.Mutex
	var calls []policyCall
	exec := &Executor{progress: newProgressHub()}
	exec.policyParent = func(_ context.Context, method string, params json.RawMessage, caller string) (json.RawMessage, error) {
		var p policyBody
		if err := json.Unmarshal(params, &p); err != nil {
			t.Errorf("relayed params are not a policy body: %v", err)
		}
		mu.Lock()
		calls = append(calls, policyCall{method: method, caller: caller, params: p})
		mu.Unlock()
		return json.RawMessage(`{"availability":"paused"}`), nil
	}
	mesh := clustertrust.Open(peerDir)
	server := httptest.NewUnstartedServer((&controlServer{exec: exec, mesh: mesh}).mux())
	server.TLS = mesh.ServerTLSConfig()
	server.StartTLS()
	t.Cleanup(server.Close)
	return server, func() []policyCall {
		mu.Lock()
		defer mu.Unlock()
		return append([]policyCall(nil), calls...)
	}
}

func pairedPolicyDirs(t *testing.T) (selfDir, peerDir string) {
	t.Helper()
	selfDir, peerDir = t.TempDir(), t.TempDir()
	clustertrusttest.WriteKeypair(t, selfDir, "uuid-self")
	clustertrusttest.WriteKeypair(t, peerDir, "uuid-peer")
	clustertrusttest.WriteAdmission(t, selfDir, "cluster-1", 1)
	clustertrusttest.WriteAdmission(t, peerDir, "cluster-1", 1)
	writePinFromCert(t, selfDir, "uuid-peer", filepath.Join(peerDir, "node.crt"))
	writePinFromCert(t, peerDir, "uuid-self", filepath.Join(selfDir, "node.crt"))
	return selfDir, peerDir
}

func TestRemotePolicyRelaysToPeerWithoutFurtherHop(t *testing.T) {
	selfDir, peerDir := pairedPolicyDirs(t)
	server, calls := policyPeerServer(t, peerDir)
	host, portText, _ := net.SplitHostPort(server.Listener.Addr().String())
	port, _ := strconv.Atoi(portText)

	out := &syncBuffer{}
	m := NewManager(NewCodec(out), &Executor{progress: newProgressHub()}, clustertrust.Open(selfDir))
	t.Cleanup(m.remoteHTTP.CloseIdle)
	t.Cleanup(m.readyHTTP.CloseIdle)
	m.peers.set([]noderec.DirectoryNode{{
		HostUUID: "host-peer", Name: "peer", IP: host, ClusterUUID: "uuid-peer",
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineControl: {Port: port}},
	}})

	cases := []struct {
		method, params, wantMethod string
	}{
		{"engine:remote-policy-get", `{"nodeId":"host-peer"}`, "get"},
		{"engine:remote-policy-set", `{"nodeId":"host-peer","policy":{"version":1}}`, "set"},
		{"engine:remote-availability-set", `{"nodeId":"host-peer","state":"paused"}`, "set-availability"},
	}
	for i, c := range cases {
		id := json.RawMessage(strconv.Itoa(i + 1))
		m.handleMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: c.method, Params: json.RawMessage(c.params)})
	}
	deadline := time.Now().Add(5 * time.Second)
	for len(calls()) < len(cases) && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	got := calls()
	if len(got) != len(cases) {
		t.Fatalf("peer saw %d relayed calls, want %d", len(got), len(cases))
	}
	seen := map[string]policyCall{}
	for _, c := range got {
		seen[c.method] = c
		if c.caller != "uuid-self" {
			t.Errorf("%s caller = %q, want the authenticated uuid-self", c.method, c.caller)
		}
		if c.params.NodeID != "" {
			t.Errorf("%s reached the peer broker with nodeId %q; a relayed call must not hop again", c.method, c.params.NodeID)
		}
	}
	for _, c := range cases {
		if _, ok := seen[c.wantMethod]; !ok {
			t.Errorf("peer never saw relay method %q", c.wantMethod)
		}
	}
	if seen["set-availability"].params.State != "paused" || len(seen["set"].params.Policy) == 0 {
		t.Errorf("payload lost in transit: %+v", seen)
	}
	for len(out.frames()) < len(cases) && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	for _, f := range out.frames() {
		if f.Error != nil {
			t.Errorf("client response error: %+v", f.Error)
		}
	}
}

func TestPolicyRoutesRejectStrangersAndBadBodies(t *testing.T) {
	selfDir, peerDir := pairedPolicyDirs(t)
	strangerDir := t.TempDir()
	clustertrusttest.WriteKeypair(t, strangerDir, "uuid-stranger")
	clustertrusttest.WriteAdmission(t, strangerDir, "cluster-1", 1)
	writePinFromCert(t, strangerDir, "uuid-peer", filepath.Join(peerDir, "node.crt"))
	server, calls := policyPeerServer(t, peerDir)

	client := func(dir string) *http.Client {
		config, ok := clustertrust.Open(dir).ClientTLSConfig("uuid-peer")
		if !ok {
			t.Fatal("pin unavailable")
		}
		tr := &http.Transport{TLSClientConfig: config}
		t.Cleanup(tr.CloseIdleConnections)
		return &http.Client{Transport: tr}
	}
	post := func(cl *http.Client, path, body string) int {
		t.Helper()
		res, err := cl.Post(server.URL+path, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		_ = res.Body.Close()
		return res.StatusCode
	}
	if code := post(client(strangerDir), controlPolicyGetPath, `{}`); code != http.StatusForbidden {
		t.Fatalf("unpinned caller got %d, want 403", code)
	}
	self := client(selfDir)
	if code := post(self, controlPolicySetPath, `{"policy":{},"extra":1}`); code != http.StatusBadRequest {
		t.Fatalf("unknown field got %d, want 400", code)
	}
	big := `{"policy":{"pad":"` + strings.Repeat("x", policyBodyLimit) + `"}}`
	if code := post(self, controlPolicySetPath, big); code != http.StatusBadRequest {
		t.Fatalf("oversized body got %d, want 400", code)
	}
	res, err := self.Get(server.URL + controlPolicyGetPath)
	if err != nil {
		t.Fatal(err)
	}
	_ = res.Body.Close()
	if res.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("GET got %d, want 405", res.StatusCode)
	}
	if n := len(calls()); n != 0 {
		t.Fatalf("rejected requests reached the broker %d times", n)
	}
	if code := post(self, controlAvailabilitySetPath, `{"nodeId":"elsewhere","state":"available"}`); code != http.StatusOK {
		t.Fatalf("paired availability call got %d, want 200", code)
	}
	if got := calls(); len(got) != 1 || got[0].params.NodeID != "" {
		t.Fatalf("relayed call = %+v, want one with nodeId cleared", got)
	}
}

func TestPolicyRelayCorrelatesReplies(t *testing.T) {
	var relay policyRelay
	sent := make(chan policyRelayRequest, 1)
	relay.send = func(method string, params any) error {
		if method == policyRequestMethod {
			sent <- params.(policyRelayRequest)
		}
		return nil
	}
	go func() {
		req := <-sent
		data, _ := json.Marshal(policyRelayReply{ID: "other", Error: "not yours"})
		relay.reply(data)
		data, _ = json.Marshal(policyRelayReply{ID: req.ID, Result: json.RawMessage(`{"ok":true}`)})
		relay.reply(data)
	}()
	result, err := relay.call(context.Background(), "get", json.RawMessage(`{}`), "peer")
	if err != nil || string(result) != `{"ok":true}` {
		t.Fatalf("call = %s, %v", result, err)
	}

	cancelled := make(chan string, 1)
	relay.send = func(method string, params any) error {
		if method == policyCancelMethod {
			cancelled <- params.(map[string]string)["id"]
		}
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := relay.call(ctx, "set-availability", json.RawMessage(`{}`), "peer"); err == nil {
		t.Fatal("call without a reply succeeded")
	}
	select {
	case id := <-cancelled:
		if id == "" {
			t.Fatal("cancel carried no id")
		}
	case <-time.After(time.Second):
		t.Fatal("abandoned call sent no policy:cancel")
	}
}
