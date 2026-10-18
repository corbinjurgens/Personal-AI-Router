// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
)

// rolesPolicyDocument is a peer projection that declares roles and says who
// paused the node.
const rolesPolicyDocument = `{"version":2,"revision":"9007199254740993123","availability":"paused",` +
	`"roles":{"weak":{"engine":"ollama","model":"qwen3:8b"},"strong":{"engine":"lmstudio","model":"big"}},` +
	`"pausedBy":"local"}`

// TestRemotePolicyRelaysRolesUnchanged checks that the relay carries roles and
// pausedBy byte for byte in both directions: a set reaches the peer's broker
// with the document as sent, and a get returns the peer's document as its
// broker answered.
func TestRemotePolicyRelaysRolesUnchanged(t *testing.T) {
	selfDir, peerDir := pairedPolicyDirs(t)

	var mu sync.Mutex
	var relayed []policyBody
	exec := &Executor{progress: newProgressHub()}
	exec.policyParent = func(_ context.Context, method string, params json.RawMessage, _ string) (json.RawMessage, error) {
		var p policyBody
		if err := json.Unmarshal(params, &p); err != nil {
			t.Errorf("relayed params are not a policy body: %v", err)
		}
		mu.Lock()
		relayed = append(relayed, p)
		mu.Unlock()
		return json.RawMessage(`{"policy":` + rolesPolicyDocument + `,"availability":"paused"}`), nil
	}
	mesh := clustertrust.Open(peerDir)
	server := httptest.NewUnstartedServer((&controlServer{exec: exec, mesh: mesh}).mux())
	server.TLS = mesh.ServerTLSConfig()
	server.StartTLS()
	t.Cleanup(server.Close)
	host, portText, err := net.SplitHostPort(server.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatal(err)
	}

	out := &syncBuffer{}
	m := NewManager(NewCodec(out), &Executor{progress: newProgressHub()}, clustertrust.Open(selfDir), selfDir)
	t.Cleanup(m.controlHTTP.CloseIdle)
	t.Cleanup(m.controlReadyHTTP.CloseIdle)
	m.peers.set([]noderec.DirectoryNode{{
		HostUUID: "host-peer", Name: "peer", IP: host, ClusterUUID: "uuid-peer",
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineControl: {Port: port}},
	}})

	requests := []struct {
		method, params string
	}{
		{"engine:remote-policy-set", `{"nodeId":"host-peer","policy":` + rolesPolicyDocument + `,"expectedPersistedRevision":"9007199254740993123"}`},
		{"engine:remote-policy-get", `{"nodeId":"host-peer"}`},
	}
	for i, r := range requests {
		id := json.RawMessage(strconv.Itoa(i + 1))
		m.handleMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: r.method, Params: json.RawMessage(r.params)})
	}

	responses := map[string]Message{}
	deadline := time.Now().Add(5 * time.Second)
	for len(responses) < len(requests) && time.Now().Before(deadline) {
		for _, f := range out.frames() {
			if f.IsResponse() && f.ID != nil {
				responses[string(*f.ID)] = f
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	if len(responses) != len(requests) {
		t.Fatalf("got %d responses, want %d", len(responses), len(requests))
	}
	for id, resp := range responses {
		if resp.Error != nil {
			t.Fatalf("response %s error: %+v", id, resp.Error)
		}
	}

	mu.Lock()
	var sent json.RawMessage
	for _, p := range relayed {
		if len(p.Policy) > 0 {
			sent = p.Policy
		}
	}
	mu.Unlock()
	if !bytes.Equal(sent, []byte(rolesPolicyDocument)) {
		t.Errorf("set reached the peer broker with policy\n %s\nwant\n %s", sent, rolesPolicyDocument)
	}

	var got struct {
		Policy json.RawMessage `json:"policy"`
	}
	if err := json.Unmarshal(responses["2"].Result, &got); err != nil {
		t.Fatalf("decode remote policy get: %v", err)
	}
	if !bytes.Equal(got.Policy, []byte(rolesPolicyDocument)) {
		t.Errorf("remote policy get returned\n %s\nwant\n %s", got.Policy, rolesPolicyDocument)
	}
}
