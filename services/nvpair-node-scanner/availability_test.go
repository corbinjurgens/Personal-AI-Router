// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"

	"nvpair-shared/clustertrusttest"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

type pauseHintInventory struct {
	Models         []string                  `json:"models"`
	ModelsByEngine map[string][]string       `json:"modelsByEngine"`
	LoadedByEngine map[string][]string       `json:"loadedByEngine"`
	Policy         *nodepolicy.Advertisement `json:"policy"`
}

func pauseHintDocument(revision string, origin nodepolicy.PauseOrigin) pauseHintInventory {
	return pauseHintInventory{
		Models:         []string{"fixture-model"},
		ModelsByEngine: map[string][]string{"fixture-engine": {"fixture-model"}},
		LoadedByEngine: map[string][]string{"fixture-engine": {"fixture-model"}},
		Policy: &nodepolicy.Advertisement{
			Revision: revision, PausedBy: origin,
			Roles: map[string]nodepolicy.AdvertisedRole{
				nodepolicy.TierWeak: {Engine: "fixture-engine", Model: "fixture-model"},
			},
		},
	}
}

// This fixture exercises plaintext loopback propagation, not peer authentication.
// It never starts a scanner browser or responder.
func pauseHintStub(t *testing.T) (int, func(int, pauseHintInventory), *atomic.Int64) {
	t.Helper()
	var listener net.Listener
	for attempt := 0; attempt < 100; attempt++ {
		candidate, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("listen for inventory stub: %v", err)
		}
		port := candidate.Addr().(*net.TCPAddr).Port
		if port != 8080 && port != 11434 && port != 1234 && (port < 14318 || port > 14323) {
			listener = candidate
			break
		}
		if err := candidate.Close(); err != nil {
			t.Fatalf("close protected-port listener: %v", err)
		}
	}
	if listener == nil {
		t.Fatal("could not allocate an unprotected ephemeral port")
	}
	var mu sync.Mutex
	status := http.StatusOK
	var body []byte
	set := func(nextStatus int, inventory pauseHintInventory) {
		t.Helper()
		encoded, err := json.Marshal(inventory)
		if err != nil {
			t.Fatalf("encode inventory: %v", err)
		}
		mu.Lock()
		status, body = nextStatus, encoded
		mu.Unlock()
	}
	requests := &atomic.Int64{}
	server := &httptest.Server{Listener: listener, Config: &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/models" {
			t.Errorf("unexpected inventory request: %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
			return
		}
		mu.Lock()
		code, encoded := status, body
		mu.Unlock()
		requests.Add(1)
		w.WriteHeader(code)
		if _, err := w.Write(encoded); err != nil {
			t.Errorf("write inventory: %v", err)
		}
	})}}
	server.Start()
	t.Cleanup(server.Close)
	return listener.Addr().(*net.TCPAddr).Port, set, requests
}

func assertPauseHintInventory(t *testing.T, d *daemon, want pauseHintInventory) {
	t.Helper()
	node, ok := d.dir.get("pause-peer")
	if !ok || node.Policy == nil {
		t.Fatalf("directory peer present=%v policy=%+v", ok, node.Policy)
	}
	got := pauseHintInventory{node.Models, node.ModelsByEngine, node.LoadedByEngine, node.Policy}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("directory inventory = %+v policy=%+v, want %+v policy=%+v", got, got.Policy, want, want.Policy)
	}
	d.infoMu.Lock()
	cached := pauseHintInventory{d.lastModels["pause-peer"], d.lastModelsByEngine["pause-peer"], d.lastLoadedByEngine["pause-peer"], d.lastPolicy["pause-peer"]}
	d.infoMu.Unlock()
	if cached.Policy == nil || !reflect.DeepEqual(cached, want) {
		t.Fatalf("cached inventory = %+v policy=%+v, want %+v policy=%+v", cached, cached.Policy, want, want.Policy)
	}
}

func TestPauseHintFollowsTheAdvertisement(t *testing.T) {
	d := newRefreshTestDaemon()
	port, set, requests := pauseHintStub(t)
	seedEMNode(d, "pause-peer", "127.0.0.1", port)
	refresh := func(phase string, want pauseHintInventory, changed bool) {
		t.Helper()
		before := requests.Load()
		if got := d.refreshNodeModels("pause-peer", "127.0.0.1", port, ""); got != changed {
			t.Fatalf("%s: changed=%v, want %v", phase, got, changed)
		}
		if got := requests.Load(); got != before+1 {
			t.Fatalf("%s: acknowledged requests=%d, want %d", phase, got, before+1)
		}
		assertPauseHintInventory(t, d, want)
	}
	available := pauseHintDocument("1", "")
	set(http.StatusOK, available)
	refresh("available", available, true)
	paused := pauseHintDocument("2", nodepolicy.PausedLocally)
	set(http.StatusOK, paused)
	refresh("local pause", paused, true)
	refresh("identical pause", paused, false)
	// A valid but different document on the error response must not replace
	// either the directory policy or the cached last-good inventory.
	set(http.StatusInternalServerError, available)
	refresh("acknowledged failure retains pause", paused, false)
	resumed := pauseHintDocument("3", "")
	set(http.StatusOK, resumed)
	refresh("resume", resumed, true)
	peerPaused := pauseHintDocument("4", nodepolicy.PausedByPeer)
	set(http.StatusOK, peerPaused)
	refresh("peer pause", peerPaused, true)
}

type pauseHintOutput struct {
	mu sync.Mutex
	bytes.Buffer
}

func (out *pauseHintOutput) Write(p []byte) (int, error) {
	out.mu.Lock()
	defer out.mu.Unlock()
	return out.Buffer.Write(p)
}

func (out *pauseHintOutput) take(t *testing.T) []Message {
	t.Helper()
	out.mu.Lock()
	defer out.mu.Unlock()
	decoder := json.NewDecoder(bytes.NewReader(out.Buffer.Bytes()))
	var frames []Message
	for {
		var frame Message
		if err := decoder.Decode(&frame); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			t.Fatalf("decode discovery frame: %v", err)
		}
		frames = append(frames, frame)
	}
	out.Buffer.Reset()
	return frames
}

func TestPauseHintEmitsOneNodeUpdated(t *testing.T) {
	d := newRefreshTestDaemon()
	port, set, _ := pauseHintStub(t)
	seedEMNode(d, "pause-peer", "127.0.0.1", port)
	set(http.StatusOK, pauseHintDocument("1", ""))
	if !d.refreshNodeModels("pause-peer", "127.0.0.1", port, "") {
		t.Fatal("initial inventory did not populate")
	}
	out := &pauseHintOutput{}
	d.codec = NewCodec(out)
	check := func(phase string, want pauseHintInventory, count int) {
		t.Helper()
		set(http.StatusOK, want)
		// Direct refresh is synchronous, including codec writes; no writer remains
		// outstanding when take decodes the synchronized output.
		d.refreshNodeModels("pause-peer", "127.0.0.1", port, "")
		frames := out.take(t)
		if len(frames) != count {
			t.Fatalf("%s: frames=%d, want %d", phase, len(frames), count)
		}
		for _, frame := range frames {
			if frame.Method != noderec.NotifyNodeUpdated {
				t.Fatalf("%s: method=%q, want node-updated", phase, frame.Method)
			}
			var event noderec.NodeEvent
			if err := json.Unmarshal(frame.Params, &event); err != nil {
				t.Fatalf("%s: decode updated node: %v", phase, err)
			}
			node := event.Node
			stored, ok := d.dir.get("pause-peer")
			if !ok || node.Policy == nil || node.HostUUID != "pause-peer" || !reflect.DeepEqual(node, stored) || !reflect.DeepEqual(node.Policy, want.Policy) {
				t.Fatalf("%s: emitted node=%+v policy=%+v, want stored node with policy=%+v", phase, node, node.Policy, want.Policy)
			}
		}
	}
	paused := pauseHintDocument("2", nodepolicy.PausedLocally)
	check("pause only", paused, 1)
	check("identical pause", paused, 0)
	check("resume", pauseHintDocument("3", ""), 1)
}

func TestPauseHintIsNeverFetchedWithoutAPin(t *testing.T) {
	clusterDir := filepath.Join(t.TempDir(), "cluster")
	d := newMembershipTestDaemon(clusterDir)
	t.Cleanup(d.closePeerClients)
	var plainCalls atomic.Int64
	d.modelsHTTP = &http.Client{Transport: peerServicesTestTransport(func(r *http.Request) (*http.Response, error) {
		plainCalls.Add(1)
		return nil, errors.New("controlled plaintext refusal")
	})}
	// No socket is opened: the transport refuses synchronously before any dial.
	const port = 19001
	loopback := func(phase string) {
		t.Helper()
		client, scheme, current, ok := d.modelsClient("not-the-self-uuid", "127.0.0.1", port, "principal-stranger", true)
		if !ok || client != d.modelsHTTP || scheme != "http" || current == nil || !current() {
			t.Fatalf("%s: loopback plaintext selection failed: scheme=%q ok=%v", phase, scheme, ok)
		}
	}
	refused := func(phase string, discovering bool) {
		t.Helper()
		if _, _, _, ok := d.modelsClient("untrusted-host", "192.0.2.42", port, "principal-stranger", discovering); ok {
			t.Fatalf("%s: selected an unpinned LAN transport", phase)
		}
		if _, ok := d.fetchModelsForPeer("untrusted-host", "192.0.2.42", port, "principal-stranger", discovering); ok {
			t.Fatalf("%s: unpinned fetch succeeded", phase)
		}
		if got := plainCalls.Load(); got != 0 {
			t.Fatalf("%s: plaintext calls=%d, want zero", phase, got)
		}
		d.peerMu.Lock()
		cached := len(d.peerClients)
		d.peerMu.Unlock()
		if cached != 0 {
			t.Fatalf("%s: unpinned fetch created %d peer clients", phase, cached)
		}
	}
	loopback("standalone")
	refused("standalone discovery", true)
	refused("standalone periodic refresh", false)
	clustertrusttest.Join(t, clusterDir, "pause-cluster", "principal-self", "principal-peer")
	writeScannerAdmissionPin(t, clusterDir, "principal-peer", "pause-cluster", 1)
	d.mesh.Refresh()
	loopback("joined")
	refused("joined discovery", true)
	refused("joined periodic refresh", false)
	if _, ok := d.fetchModelsForPeer("not-the-self-uuid", "127.0.0.1", port, "principal-stranger", true); ok {
		t.Fatal("controlled loopback refusal unexpectedly succeeded")
	}
	if got := plainCalls.Load(); got != 1 {
		t.Fatalf("loopback positive control: plaintext calls=%d, want one", got)
	}
	client, scheme, current, ok := d.modelsClient("pinned-host", "192.0.2.43", port, "principal-peer", true)
	if !ok || client == nil || client == d.modelsHTTP || scheme != "https" || current == nil || !current() {
		t.Fatalf("admitted pinned peer: scheme=%q ok=%v, want HTTPS client", scheme, ok)
	}
}
