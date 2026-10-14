// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"testing"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/noderec"
)

// File requests use fresh pinned connections so an interrupted HTTP request
// cannot be replayed by an idle-connection transport.
func TestRemoteClient_ModelFilesUseFreshConnections(t *testing.T) {
	selfDir := t.TempDir()
	peerDir := t.TempDir()
	clustertrusttest.WriteKeypair(t, selfDir, "uuid-self")
	clustertrusttest.WriteKeypair(t, peerDir, "uuid-peer")
	clustertrusttest.WriteAdmission(t, selfDir, "cluster-1", 1)
	clustertrusttest.WriteAdmission(t, peerDir, "cluster-1", 1)
	writePinFromCert(t, selfDir, "uuid-peer", filepath.Join(peerDir, "node.crt"))
	writePinFromCert(t, peerDir, "uuid-self", filepath.Join(selfDir, "node.crt"))

	selfMesh := clustertrust.Open(selfDir)
	peerMesh := clustertrust.Open(peerDir)
	if !selfMesh.Clustered() || !peerMesh.Clustered() {
		t.Fatal("both nodes must read as clustered")
	}

	var mu sync.Mutex
	newConns := 0
	mux := http.NewServeMux()
	mux.HandleFunc(controlModelFilesPath, func(w http.ResponseWriter, r *http.Request) {
		if !r.Close {
			t.Error("model listing request could retain a replayable idle connection")
		}
		if _, ok := peerMesh.VerifyClientPin(r); !ok {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		if err := json.NewEncoder(w).Encode(streamFrame{Type: "result", Result: json.RawMessage(`{"files":[]}`)}); err != nil {
			t.Errorf("encode file list: %v", err)
		}
	})
	ts := httptest.NewUnstartedServer(mux)
	ts.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			mu.Lock()
			newConns++
			mu.Unlock()
		}
	}
	ts.TLS = peerMesh.ServerTLSConfig()
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

	var out bytes.Buffer
	m := NewManager(NewCodec(&out), &Executor{progress: newProgressHub()}, selfMesh, selfDir)
	defer m.controlHTTP.CloseIdle()
	defer m.controlReadyHTTP.CloseIdle()

	m.peers.set([]noderec.DirectoryNode{{HostUUID: "host-peer", IP: host, ClusterUUID: "uuid-peer",
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineControl: {Port: port}},
	}})
	peer, ok := m.peers.lookup("host-peer")
	if !ok {
		t.Fatal("peer not in directory")
	}
	client, err := m.remoteClient(context.Background(), peer)
	if err != nil {
		t.Fatal(err)
	}

	const rounds = 5
	for i := 0; i < rounds; i++ {
		if _, gerr := client.modelFiles(context.Background(), "ollama", "fixture", nil); gerr != nil {
			t.Fatalf("round %d: %v", i, gerr)
		}
	}

	mu.Lock()
	got := newConns
	mu.Unlock()
	if got != rounds {
		t.Fatalf("peer accepted %d connections for %d modelFiles calls, want %d", got, rounds, rounds)
	}
}

func writePinFromCert(t *testing.T, clusterDir, peerUUID, certPath string) {
	t.Helper()
	certPEM, err := os.ReadFile(certPath)
	if err != nil {
		t.Fatal(err)
	}
	trusted := filepath.Join(clusterDir, "trusted")
	if err := os.MkdirAll(trusted, 0o700); err != nil {
		t.Fatal(err)
	}
	admissionData, err := os.ReadFile(filepath.Join(clusterDir, "admission.json"))
	if err != nil {
		t.Fatal(err)
	}
	var admission struct {
		ClusterID string `json:"clusterId"`
		Epoch     uint64 `json:"epoch"`
	}
	if err := json.Unmarshal(admissionData, &admission); err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(map[string]any{"nodeUuid": peerUUID, "certPem": string(certPEM),
		"clusterId": admission.ClusterID, "admissionEpoch": admission.Epoch})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(trusted, peerUUID+".json"), body, 0o600); err != nil {
		t.Fatal(err)
	}
}
