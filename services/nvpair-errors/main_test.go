// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"nvpair-shared/appdir"
	"nvpair-shared/clustertrust"
	"nvpair-shared/errors"
	"nvpair-shared/nodeid"
)

func TestPeerSyncStartupIdentity(t *testing.T) {
	const want = "11111111-2222-4333-8444-555555555555"
	writeIdentity := func(t *testing.T, path, id string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		body := mustJSON(t, struct {
			NodeUUID string `json:"node_uuid"`
		}{id})
		if err := os.WriteFile(path, body, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	test := func(name string, clusterIdentity, defaultRoot bool) {
		t.Run(name, func(t *testing.T) {
			// Isolate the shared default as well as the explicitly configured root.
			privateHome := t.TempDir()
			for _, key := range []string{"HOME", "XDG_CONFIG_HOME", "LOCALAPPDATA", "APPDATA"} {
				t.Setenv(key, privateHome)
			}
			base := t.TempDir()
			clusterDir := filepath.Join(base, "cluster")
			if defaultRoot {
				var err error
				base, err = appdir.Dir()
				if err != nil {
					t.Fatal(err)
				}
				clusterDir = ""
			}
			if clusterIdentity {
				writeIdentity(t, filepath.Join(base, "node-id.json"), "older-node-id")
				writeIdentity(t, filepath.Join(base, "cluster", "identity.json"), want)
			} else {
				writeIdentity(t, filepath.Join(base, "node-id.json"), want)
			}

			mgr := newStartupManager(NewCodec(nopReadWriter{}), true, "", clusterDir)
			if got := mgr.LocalNodeID(); got != want {
				t.Fatalf("startup node ID = %q, want persisted UUID %q", got, want)
			}
			// Construct the push owner before reports, as startup does. It must
			// capture the same UUID used to select local-origin entries.
			peers := NewPeerSync(mgr, clustertrust.Open(clusterDir), clusterDir)
			t.Cleanup(peers.clients.CloseIdle)
			mgr.handleReport(nil, mustJSON(t, localErr("startup", want, 1000)))
			assertErrorSnapshot(t, mgr, []ServiceError{localErr("startup", want, 1000)})
			body, err := peers.snapshotBody()
			if err != nil {
				t.Fatalf("encode outbound snapshot: %v", err)
			}
			get := httptest.NewRecorder()
			handleServeLocal(get, mgr)
			if get.Code != http.StatusOK {
				t.Fatalf("GET status = %d, want 200", get.Code)
			}
			for source, data := range map[string][]byte{"outbound": body, "GET": get.Body.Bytes()} {
				var env errors.SyncEnvelope
				if err := json.Unmarshal(data, &env); err != nil {
					t.Fatalf("decode %s snapshot: %v", source, err)
				}
				if env.NodeID != want || len(env.Errors) != 1 || env.Errors[0] != localErr("startup", want, 1000) {
					t.Errorf("%s snapshot = %+v, want persisted UUID and local error", source, env)
				}
			}
			if got := nodeid.Resolve(base); got != want {
				t.Fatalf("shared resolver = %q, want startup UUID %q", got, want)
			}
		})
	}
	test("configured root prefers cluster identity", true, false)
	test("configured root reuses precluster UUID", false, false)
	test("empty cluster dir uses shared default root", false, true)
}

func TestPeerSyncStartupPersistsFirstUUID(t *testing.T) {
	privateHome := t.TempDir()
	for _, key := range []string{"HOME", "XDG_CONFIG_HOME", "LOCALAPPDATA", "APPDATA"} {
		t.Setenv(key, privateHome)
	}
	base := t.TempDir()
	clusterDir := filepath.Join(base, "cluster")
	mgr := newStartupManager(NewCodec(nopReadWriter{}), true, "", clusterDir)
	body, err := os.ReadFile(filepath.Join(base, "node-id.json"))
	if err != nil {
		t.Fatalf("read startup identity: %v", err)
	}
	var stored struct {
		NodeUUID string `json:"node_uuid"`
	}
	if err := json.Unmarshal(body, &stored); err != nil {
		t.Fatalf("decode persisted identity: %v", err)
	}
	if stored.NodeUUID == "" || stored.NodeUUID != mgr.LocalNodeID() {
		t.Fatalf("persisted UUID = %q, startup UUID = %q", stored.NodeUUID, mgr.LocalNodeID())
	}
	// Cluster-manager's first identity creation calls this same resolver with
	// the cluster dir's parent, so it will adopt the UUID minted at startup.
	if got := nodeid.Resolve(base); got != stored.NodeUUID {
		t.Fatalf("future cluster resolver = %q, want persisted UUID %q", got, stored.NodeUUID)
	}
	restarted := newStartupManager(NewCodec(nopReadWriter{}), true, "", clusterDir)
	if got := restarted.LocalNodeID(); got != stored.NodeUUID {
		t.Fatalf("restarted UUID = %q, want persisted UUID %q", got, stored.NodeUUID)
	}
}

func TestStartupIdentityPreservesExplicitAndLocalOnly(t *testing.T) {
	hostname, err := os.Hostname()
	if err != nil || hostname == "" {
		hostname = "unknown"
	}
	test := func(name string, peerSync bool, override, want string) {
		t.Run(name, func(t *testing.T) {
			base := t.TempDir()
			mgr := newStartupManager(NewCodec(nopReadWriter{}), peerSync, override, filepath.Join(base, "cluster"))
			if got := mgr.LocalNodeID(); got != want {
				t.Fatalf("startup node ID = %q, want %q", got, want)
			}
			entries, err := os.ReadDir(base)
			if err != nil {
				t.Fatal(err)
			}
			if len(entries) != 0 {
				t.Fatalf("startup created identity state despite override/local-only mode: %v", entries)
			}
		})
	}
	test("peer sync preserves broker override", true, "broker-uuid", "broker-uuid")
	test("local only preserves explicit override", false, "explicit-id", "explicit-id")
	test("local only keeps hostname default", false, "", hostname)
}
