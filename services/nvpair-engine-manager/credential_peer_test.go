// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"nvpair-shared/clustertrust"
)

// credentialPeerExecutor returns an executor whose running engine holds the
// serving credential, so a route that leaked it would be visible.
func credentialPeerExecutor(t *testing.T) *Executor {
	t.Helper()
	ex := newTestExecutor(t, testEngineManifest(fakeEngineBin))
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.plat.Runtime.Ready = nil
	ownServingCredential(st, 45678)
	st.mu.Lock()
	st.running = true
	st.mu.Unlock()
	return ex
}

func credentialPeerRequests(t *testing.T, client *http.Client, base string, ex *Executor, relay *syncBuffer) {
	t.Helper()
	audit := &syncBuffer{}
	old := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(audit, nil)))
	defer slog.SetDefault(old)
	request := func(t *testing.T, method, path, body string, want int) []byte {
		t.Helper()
		req, err := http.NewRequestWithContext(t.Context(), method, base+path, strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		resp, err := client.Do(req)
		if err != nil {
			t.Fatal("pinned peer request failed")
		}
		data, readErr := io.ReadAll(resp.Body)
		closeErr := resp.Body.Close()
		if readErr != nil || closeErr != nil {
			t.Fatal("read peer reply")
		}
		if resp.StatusCode != want {
			t.Fatalf("peer route %s %s status=%d, want %d", method, path, resp.StatusCode, want)
		}
		if bytes.Contains(data, []byte(credentialTestKey)) {
			t.Fatal("peer reply exposed credential")
		}
		return data
	}
	// An authorized inventory request proves the certificate and real route
	// registry work. Credential refusal cannot be attributed to a broken pin.
	data := request(t, http.MethodGet, controlEnginesPath, "", http.StatusOK)
	var inventory struct {
		Engines []EngineStatus `json:"engines"`
	}
	if json.Unmarshal(data, &inventory) != nil || len(inventory.Engines) != 1 || inventory.Engines[0].Engine != "fake" {
		t.Fatal("inventory positive control failed")
	}
	for _, tc := range []struct{ name, method, path, body string }{
		{"retrieval", http.MethodGet, "/v1/engines/local-credential", ""},
		{"RPC retrieval", http.MethodPost, "/", `{"jsonrpc":"2.0","id":1,"method":"internal:local-engine-credential","params":{"engine":"fake"}}`},
	} {
		t.Run(tc.name, func(t *testing.T) { request(t, tc.method, tc.path, tc.body, http.StatusNotFound) })
	}
	modelData, err := json.Marshal(ex.ModelsResult(t.Context()))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(modelData, []byte(credentialTestKey)) || strings.Contains(credentialOutput(audit), credentialTestKey) || strings.Contains(credentialOutput(relay), credentialTestKey) {
		t.Fatal("credential escaped through model snapshot, audit or relay")
	}
	if len(relay.frames()) != 0 {
		t.Fatal("peer attempts invoked parent relay")
	}
}

func TestCredentialPeerDirectRoutesAbsent(t *testing.T) {
	for _, allowed := range []bool{true, false} {
		name := "management allowed"
		if !allowed {
			name = "management denied"
		}
		t.Run(name, func(t *testing.T) {
			f := rmNewFixture(t, "target", "caller")
			f.writePermissions("1", map[string]bool{"caller": allowed})
			if f.mesh.RemoteManagement("caller").Allowed != allowed {
				t.Fatal("permission fixture mismatch")
			}
			ex := credentialPeerExecutor(t)
			relay := &syncBuffer{}
			NewManager(NewCodec(relay), ex, f.mesh, f.dir)
			s := &controlServer{exec: ex, mesh: f.mesh}
			server := httptest.NewUnstartedServer(s.mux())
			server.TLS = f.mesh.ServerTLSConfig()
			server.StartTLS()
			defer server.Close()
			client := f.client("caller")
			defer client.CloseIdleConnections()
			credentialPeerRequests(t, client, server.URL, ex, relay)
		})
	}
}

func TestCredentialPeerReverseRoutesAbsent(t *testing.T) {
	for _, allowed := range []bool{true, false} {
		name := "management allowed"
		if !allowed {
			name = "management denied"
		}
		t.Run(name, func(t *testing.T) {
			ex := credentialPeerExecutor(t)
			relay := &syncBuffer{}
			f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
				NewManager(NewCodec(relay), ex, mesh, t.TempDir())
				return (&controlServer{exec: ex, mesh: mesh}).mux()
			})
			grantRemoteManagement(t, f.peerDir, "1", map[string]bool{"uuid-self": allowed})
			if clustertrust.Open(f.peerDir).RemoteManagement("uuid-self").Allowed != allowed {
				t.Fatal("permission fixture mismatch")
			}
			client := f.client(t, remoteFixtureContext(t))
			credentialPeerRequests(t, client.controlHTTP, client.base, ex, relay)
			if f.reverseCalls.Load() == 0 {
				t.Fatal("fixture did not use reverse inner mTLS")
			}
		})
	}
}

func TestCredentialPeerRegistryHasNoCredentialRoute(t *testing.T) {
	server := &controlServer{}
	for _, route := range server.routes() {
		if strings.Contains(route.path, "credential") || strings.Contains(route.family, "credential") || route.path == "/" || strings.Contains(strings.ToLower(route.path), "rpc") {
			t.Fatal("credential or generic RPC route registered")
		}
	}
}
