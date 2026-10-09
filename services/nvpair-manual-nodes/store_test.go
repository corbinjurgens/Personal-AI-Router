// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func readState(t *testing.T, path string) []ManualEntry {
	t.Helper()
	entries, err := loadEntries(path)
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	return entries
}

func TestNodeAddPersistsAndRemoveDeletes(t *testing.T) {
	m, rw, _ := newTestManager()
	m.statePath = filepath.Join(t.TempDir(), "sub", stateFileName)

	// A hostname is stored as given, never as the address it resolved to.
	m.handleMessage(requestMessage(1, "node/add", ManualEntry{Address: "lab.tailnet.ts.net", Name: "lab", TLSPort: 14319, MTLS: true}))
	readCaptureUntil(t, rw, responseWithID(1))
	m.handleMessage(requestMessage(2, "node/add", ManualEntry{Address: "10.0.0.7"}))
	readCaptureUntil(t, rw, responseWithID(2))

	got := readState(t, m.statePath)
	if len(got) != 2 {
		t.Fatalf("persisted %d entries, want 2: %+v", len(got), got)
	}
	// Sorted by node id: "lab" < "manual:10.0.0.7".
	if got[0] != (ManualEntry{Address: "lab.tailnet.ts.net", Name: "lab", TLSPort: 14319, MTLS: true}) ||
		got[1] != (ManualEntry{Address: "10.0.0.7"}) {
		t.Fatalf("persisted entries = %+v", got)
	}
	if runtime.GOOS != "windows" {
		st, err := os.Stat(m.statePath)
		if err != nil {
			t.Fatal(err)
		}
		if mode := st.Mode().Perm(); mode != 0o600 {
			t.Fatalf("state file mode = %o, want 600", mode)
		}
	}

	m.handleMessage(requestMessage(3, "node/remove", map[string]string{"id": "lab"}))
	readCaptureUntil(t, rw, responseWithID(3))
	got = readState(t, m.statePath)
	if len(got) != 1 || got[0].Address != "10.0.0.7" {
		t.Fatalf("after remove = %+v", got)
	}

	// No temporary files are left beside it.
	files, _ := os.ReadDir(filepath.Dir(m.statePath))
	if len(files) != 1 {
		t.Fatalf("state dir holds %d files, want only the state file", len(files))
	}
}

func TestReAddReplacesTheSavedEntry(t *testing.T) {
	m, rw, _ := newTestManager()
	m.statePath = filepath.Join(t.TempDir(), stateFileName)
	m.handleMessage(requestMessage(1, "node/add", ManualEntry{Address: "old.local", Name: "lab"}))
	readCaptureUntil(t, rw, responseWithID(1))
	m.handleMessage(requestMessage(2, "node/add", ManualEntry{Address: "new.local", Name: "lab"}))
	readCaptureUntil(t, rw, responseWithID(2))
	got := readState(t, m.statePath)
	if len(got) != 1 || got[0].Address != "new.local" {
		t.Fatalf("persisted = %+v, want the one lab entry at its new address", got)
	}
}

func TestReplayRestoresSavedEntries(t *testing.T) {
	m, rw, rt := newTestManager()
	m.statePath = filepath.Join(t.TempDir(), stateFileName)
	if err := saveEntries(m.statePath, []ManualEntry{
		{Address: "node.local", Name: "lab"},
		{Address: "box.tailnet.ts.net"},
	}); err != nil {
		t.Fatal(err)
	}
	configureHealthyNode(rt, "node.local", []string{"llama3"}, sampleInfo())

	m.replay()

	nodes := m.listNodes()
	if len(nodes) != 2 {
		t.Fatalf("restored %d nodes, want 2", len(nodes))
	}
	byID := map[string]ManualNodeStatus{}
	for _, n := range nodes {
		byID[n.ID] = n
	}
	if byID["lab"].Address != "node.local" || byID["manual:box.tailnet.ts.net"].Address != "box.tailnet.ts.net" {
		t.Fatalf("restored nodes = %+v", nodes)
	}
	// Each restored entry is probed and announced like a fresh node/add.
	seen := map[string]bool{}
	for len(seen) < 2 {
		msg := readCaptureUntil(t, rw, methodIs("node/discovered"))
		seen[decodeParams[ManualNodeStatus](t, msg).ID] = true
	}
}

func TestRunReplaysAfterReady(t *testing.T) {
	path := filepath.Join(t.TempDir(), stateFileName)
	if err := saveEntries(path, []ManualEntry{{Address: "127.0.0.1", Name: "loopback"}}); err != nil {
		t.Fatal(err)
	}
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()
	mgr, err := NewManager(NewCodec(server), tlsClientOptions{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	mgr.statePath = path
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { _ = mgr.Run(ctx) }()

	reader := bufio.NewReader(client)
	if first := readPipeFrame(t, client, reader); first.Method != "ready" {
		t.Fatalf("first frame = %+v, want ready before any restored node", first)
	}
	writePipeRequest(t, client, 5, "nodes/list", nil)
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		msg := readPipeFrame(t, client, reader)
		if !responseWithID(5)(msg) {
			continue
		}
		list := decodeResult[map[string][]ManualNodeStatus](t, msg)
		if len(list["nodes"]) != 1 || list["nodes"][0].ID != "loopback" {
			t.Fatalf("nodes/list after restart = %+v", list)
		}
		return
	}
	t.Fatal("no nodes/list response")
}

func TestUnreadableStateFileIsMovedAside(t *testing.T) {
	m, _, _ := newTestManager()
	m.statePath = filepath.Join(t.TempDir(), stateFileName)
	if err := os.WriteFile(m.statePath, []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	m.replay()
	if len(m.listNodes()) != 0 {
		t.Fatal("restored nodes from a corrupt file")
	}
	if _, err := os.Stat(m.statePath + ".invalid"); err != nil {
		t.Fatalf("corrupt file not kept aside: %v", err)
	}
	if _, err := os.Stat(m.statePath); !os.IsNotExist(err) {
		t.Fatalf("corrupt file still in place: %v", err)
	}
}

func TestNoStatePathDisablesPersistence(t *testing.T) {
	m, rw, _ := newTestManager()
	dir := t.TempDir()
	t.Chdir(dir)
	m.handleMessage(requestMessage(1, "node/add", ManualEntry{Address: "node.local"}))
	readCaptureUntil(t, rw, responseWithID(1))
	m.replay()
	if files, _ := os.ReadDir(dir); len(files) != 0 {
		t.Fatalf("wrote %d files with persistence disabled", len(files))
	}
}

// TestProbeTransportsReResolveEveryProbe pins what makes hostname entries
// follow a renumbered machine: no probe transport keeps a pooled connection, so
// every probe dials, and every dial resolves the name afresh (Go's resolver
// keeps no cache of its own).
func TestProbeTransportsReResolveEveryProbe(t *testing.T) {
	m, err := NewManager(NewCodec(newCaptureRW()), tlsClientOptions{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for name, c := range map[string]*http.Client{"plain": m.client, "tls": m.tlsClient} {
		tr, ok := c.Transport.(*http.Transport)
		if !ok || !tr.DisableKeepAlives {
			t.Fatalf("%s probe transport keeps connections alive; a hostname entry would stick to a stale address", name)
		}
	}
}
