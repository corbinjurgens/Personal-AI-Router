// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/modeldownload"
	"nvpair-shared/noderec"
)

func scannerDownloadSnapshot(revision string) *modeldownload.Snapshot {
	percent := 21
	return &modeldownload.Snapshot{Version: modeldownload.Version, SourceID: strings.Repeat("a", 32), Revision: revision,
		Entries: []modeldownload.Entry{{ID: strings.Repeat("b", 32), Engine: "llamacpp", Model: "org/model", Phase: modeldownload.PhaseInProgress,
			BlocksReadiness: true, Percent: &percent}}}
}

func TestModelDownloadRefreshFreshness(t *testing.T) {
	state := scannerDownloadSnapshot("7")
	body := map[string]any{"models": []string{"ready"}, "modelsByEngine": map[string][]string{"llamacpp": {"ready"}},
		"loadedByEngine": map[string][]string{"llamacpp": {"org/model"}}, "downloadState": state}
	stub, port := startModelsStub(t, http.StatusOK, body)
	d := newRefreshTestDaemon()
	seedEMNode(d, "target", "127.0.0.1", port)
	refresh := func(wantChanged, wantCurrent bool) {
		t.Helper()
		if changed := d.refreshNodeModels("target", "127.0.0.1", port, ""); changed != wantChanged {
			t.Fatalf("refresh changed = %v, want %v", changed, wantChanged)
		}
		node, present := d.dir.get("target")
		if !present || node.DownloadState == nil || !node.DownloadState.Equal(*state) || node.DownloadStateCurrent == nil || *node.DownloadStateCurrent != wantCurrent {
			t.Fatalf("wrong download observation: %+v", node)
		}
		if !reflect.DeepEqual(node.Models, []string{"ready"}) || !reflect.DeepEqual(node.LoadedByEngine, map[string][]string{"llamacpp": {"org/model"}}) {
			t.Fatalf("freshness changed ready inventory or loaded observation: %+v", node)
		}
	}
	refresh(true, true)
	refresh(false, true)
	stub.set(http.StatusServiceUnavailable, nil)
	refresh(true, false)
	refresh(false, false)
	stub.set(http.StatusOK, body)
	refresh(true, true) // The same revision restores freshness.
	older := scannerDownloadSnapshot("6")
	stub.set(http.StatusOK, map[string]any{"models": []string{"obsolete"}, "downloadState": older})
	refresh(false, true)
}

func TestModelDownloadInventoryExtensionParsing(t *testing.T) {
	test := func(name, extension string, supported bool, current *bool) {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(fixtureRequestHandler(http.MethodGet, "/v1/models", func(w http.ResponseWriter, r *http.Request) {
				_, err := w.Write([]byte(`{"models":["ready"],"loadedByEngine":{"llamacpp":["resident"]}` + extension + `}`))
				if err != nil {
					t.Errorf("write inventory: %v", err)
				}
			}))
			t.Cleanup(srv.Close)
			d := newRefreshTestDaemon()
			inventory, ok := d.fetchModels("127.0.0.1", portFromURL(t, srv.URL), "")
			wantModels := []string{"ready"}
			if current != nil && !*current {
				wantModels = []string{}
			}
			if !ok || !reflect.DeepEqual(inventory.models, wantModels) || !reflect.DeepEqual(inventory.loadedByEngine, map[string][]string{"llamacpp": {"resident"}}) {
				t.Fatalf("wrong canonical inventory/residency: %+v, %v", inventory, ok)
			}
			if (inventory.snapshot != nil) != supported || !reflect.DeepEqual(inventory.current, current) {
				t.Fatalf("snapshot support/current = %+v/%v, want %v/%v", inventory.snapshot, inventory.current, supported, current)
			}
		})
	}
	current, stale := true, false
	empty := scannerDownloadSnapshot("0")
	empty.Entries = []modeldownload.Entry{}
	raw, err := json.Marshal(empty)
	if err != nil {
		t.Fatalf("encode empty snapshot: %v", err)
	}
	test("explicit empty is current", `,"downloadState":`+string(raw), true, &current)
	test("unsupported is absent", "", false, nil)
	test("null support is absent", `,"downloadState":null`, false, nil)
	test("malformed remains unknown", `,"downloadState":{"version":1,"entries":[]}`, false, &stale)
	test("unsupported version remains unknown", `,"downloadState":{"version":2}`, false, &stale)
	test("oversize lifecycle remains unknown", `,"downloadState":{"padding":"`+strings.Repeat("x", modeldownload.MaxSnapshotBytes)+`"}`, false, &stale)
	test("aliased lifecycle remains unknown", `,"DownloadState":`+string(raw), false, &stale)
	test("duplicate lifecycle remains unknown", `,"downloadState":{},"downloadState":`+string(raw), false, &stale)
}

func TestModelDownloadMalformedInventoryWithdrawsDiscoveryNames(t *testing.T) {
	state := scannerDownloadSnapshot("7")
	loaded := map[string][]string{"llamacpp": {"org/model"}}
	stub, port := startModelsStub(t, http.StatusOK, map[string]any{"models": []string{"ready"}, "modelsByEngine": map[string][]string{"llamacpp": {"ready"}},
		"loadedByEngine": loaded, "downloadState": state})
	d := newRefreshTestDaemon()
	seedEMNode(d, "target", "127.0.0.1", port)
	if !d.refreshNodeModels("target", "127.0.0.1", port, "") {
		t.Fatal("initial current inventory not applied")
	}
	var wire bytes.Buffer
	d.codec = NewCodec(&wire)
	stub.set(http.StatusOK, map[string]any{"models": []string{"unsafe"}, "modelsByEngine": map[string][]string{"llamacpp": {"unsafe"}},
		"loadedByEngine": loaded, "downloadState": map[string]any{"version": 1, "entries": []any{}}})
	if !d.refreshNodeModels("target", "127.0.0.1", port, "") {
		t.Fatal("malformed lifecycle did not emit inventory withdrawal")
	}
	frame, err := d.codec.Read()
	if err != nil {
		t.Fatalf("read discovery notification: %v", err)
	}
	if frame.Method != noderec.NotifyNodeUpdated {
		t.Fatalf("wrong discovery method: %q", frame.Method)
	}
	var event noderec.NodeEvent
	if err := json.Unmarshal(frame.Params, &event); err != nil {
		t.Fatalf("parse discovery event: %v", err)
	}
	node := event.Node
	if len(node.Models) != 0 || node.ModelsByEngine == nil || len(node.EngineModels("llamacpp")) != 0 {
		t.Fatalf("malformed lifecycle promoted ready names through discovery: %+v", node)
	}
	if !reflect.DeepEqual(node.LoadedByEngine, loaded) || node.DownloadState == nil || !node.DownloadState.Equal(*state) || node.DownloadStateCurrent == nil || *node.DownloadStateCurrent {
		t.Fatalf("withdrawal lost residency/stale display details: %+v", node)
	}
	if len(d.lastModels["target"]) != 0 || len(d.lastModelsByEngine["target"]["llamacpp"]) != 0 || !reflect.DeepEqual(d.lastLoadedByEngine["target"], loaded) {
		t.Fatal("last-good cache reintroduced malformed ready inventory or lost loaded evidence")
	}
}

func TestModelDownloadInventoryEnvelopeIsBoundedBeforeDecode(t *testing.T) {
	// Whitespace would decode as harmless if a reader silently truncated it.
	// Reject the full envelope, including its tail, before JSON allocation.
	var reads atomic.Int64
	d := newRefreshTestDaemon()
	d.modelsHTTP = &http.Client{Transport: peerServicesTestTransport(func(r *http.Request) (*http.Response, error) {
		body := &downloadCountingReader{reader: io.MultiReader(strings.NewReader(`{"models":["unsafe"]}`), io.LimitReader(downloadSpaceReader{}, modelsResponseBytes)), reads: &reads}
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(body)}, nil
	})}
	if _, ok := d.fetchModels("127.0.0.1", 14317, ""); ok {
		t.Fatal("oversize inventory envelope accepted")
	}
	if got := reads.Load(); got > modelsResponseBytes+1 {
		t.Fatalf("HTTP body read %d bytes beyond %d-byte bounded sentinel", got, modelsResponseBytes+1)
	}
}

type downloadCountingReader struct {
	reader io.Reader
	reads  *atomic.Int64
}

func (r *downloadCountingReader) Read(p []byte) (int, error) {
	n, err := r.reader.Read(p)
	r.reads.Add(int64(n))
	return n, err
}

type downloadSpaceReader struct{}

func (downloadSpaceReader) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = ' '
	}
	return len(p), nil
}

func TestModelDownloadAuthenticatedEnrichmentAcceptance(t *testing.T) {
	d, claim := newPeerServicesScanner(t)
	state := scannerDownloadSnapshot("8")
	loaded := map[string][]string{"llamacpp": {"org/model"}}
	body := map[string]any{"models": []string{"ready"}, "modelsByEngine": map[string][]string{"llamacpp": {"ready"}}, "loadedByEngine": loaded, "downloadState": state}
	stub, port := startModelsStub(t, http.StatusOK, body)
	d.modelsHTTP = &http.Client{Timeout: modelsFetchTimeout}
	node := noderec.DirectoryNode{HostUUID: "peer", IP: "127.0.0.1", Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineManager: {Port: port}}}
	// Only the private fixture endpoint is loopback. The final authenticated
	// claim/admission fence is the same one used for production peer reads.
	d.dir.installPeerServices(claim, node)
	enrich := func() { d.enrichPeerServices(claim, node, d.dir.routeGeneration("peer")) }
	enrich()
	got := peerServicesNode(t, d)
	if got.DownloadState == nil || !got.DownloadState.Equal(*state) || got.DownloadStateCurrent == nil || !*got.DownloadStateCurrent || !reflect.DeepEqual(got.LoadedByEngine, loaded) {
		t.Fatalf("authenticated enrichment lost current lifecycle or residency: %+v", got)
	}
	older := scannerDownloadSnapshot("7")
	stub.set(http.StatusOK, map[string]any{"models": []string{"obsolete"}, "downloadState": older})
	enrich()
	got = peerServicesNode(t, d)
	if !got.DownloadState.Equal(*state) || !reflect.DeepEqual(got.Models, []string{"ready"}) || !d.lastDownloads["peer"].snapshot.Equal(*state) {
		t.Fatal("lower authenticated revision changed directory or cache")
	}
	stub.set(http.StatusServiceUnavailable, nil)
	enrich()
	got = peerServicesNode(t, d)
	if got.DownloadStateCurrent == nil || *got.DownloadStateCurrent || !got.DownloadState.Equal(*state) || !reflect.DeepEqual(got.LoadedByEngine, loaded) || *d.lastDownloads["peer"].current {
		t.Fatal("failed authenticated fetch lost details/residency or retained freshness")
	}
	stub.set(http.StatusOK, body)
	enrich()
	if got := peerServicesNode(t, d); !*got.DownloadStateCurrent {
		t.Fatal("same-revision authenticated success did not restore freshness")
	}
	replacement := scannerDownloadSnapshot("0")
	replacement.SourceID = strings.Repeat("c", 32)
	stub.set(http.StatusOK, map[string]any{"models": []string{"new-ready"}, "downloadState": replacement, "loadedByEngine": loaded})
	enrich()
	got = peerServicesNode(t, d)
	if !got.DownloadState.Equal(*replacement) || !d.lastDownloads["peer"].snapshot.Equal(*replacement) || !*got.DownloadStateCurrent {
		t.Fatal("current replacement source was not adopted by both owners")
	}
}

func TestModelDownloadAuthenticatedRetirementPreservesLatestDisplay(t *testing.T) {
	d, claim := newPeerServicesScanner(t)
	current := true
	browse := noderec.DirectoryNode{HostUUID: "peer", Name: "browse", IP: "192.0.2.40", ClusterUUID: "peer",
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineManager: {Port: 14317}},
		Models:   []string{"old-ready"}, DownloadState: scannerDownloadSnapshot("1"), DownloadStateCurrent: &current}
	d.dir.upsert(browse)
	generation := d.dir.routeGeneration("peer")
	_, changed := d.dir.installPeerServices(claim, browse)
	node := peerServicesNode(t, d)
	if !changed || d.dir.routeGeneration("peer") == generation || node.DownloadStateCurrent == nil || *node.DownloadStateCurrent {
		t.Fatal("authenticated replacement reused browse freshness or generation")
	}
	latest := scannerDownloadSnapshot("2")
	latest.SourceID = strings.Repeat("c", 32)
	inv := modelInventory{models: []string{}, byEngine: map[string][]string{"llamacpp": {}}, loadedByEngine: map[string][]string{"llamacpp": {"org/model"}},
		downloadObservation: downloadObservation{snapshot: latest, current: &current}, sourceBaseline: browse.DownloadState.SourceID}
	if _, changed, valid := d.dir.applyModels("peer", browse.IP, 14317, inv); !changed || !valid {
		t.Fatal("current replacement download observation refused")
	}
	d.infoMu.Lock()
	if !d.cacheInventoryLocked("peer", inv) {
		d.infoMu.Unlock()
		t.Fatal("replacement download cache refused")
	}
	d.infoMu.Unlock()
	d.dir.mu.Lock()
	expired := d.dir.peerServices["peer"]
	expired.ObservedAt = time.Now().Add(-peerServicesMaxAge - time.Second).UnixMilli()
	d.dir.peerServices["peer"] = expired
	d.dir.mu.Unlock()
	d.reconcilePeerServices()
	node = peerServicesNode(t, d)
	if node.DownloadState == nil || !node.DownloadState.Equal(*latest) || node.DownloadStateCurrent == nil || *node.DownloadStateCurrent ||
		len(node.Models) != 0 || len(node.EngineModels("llamacpp")) != 0 || !reflect.DeepEqual(node.LoadedByEngine, inv.loadedByEngine) {
		t.Fatalf("retirement restored old ready names or lost latest stale display/residency: %+v", node)
	}
	if cached := d.lastDownloads["peer"]; cached.snapshot == nil || !cached.snapshot.Equal(*latest) || cached.current == nil || *cached.current {
		t.Fatal("retirement retained current cache or lost latest snapshot")
	}
	*node.DownloadState.Entries[0].Percent = 90
	if again := peerServicesNode(t, d); *again.DownloadState.Entries[0].Percent != 21 {
		t.Fatal("authenticated retirement exposed directory-owned state")
	}
}

func TestModelDownloadCacheRetiresPriorSourceReads(t *testing.T) {
	d := newRefreshTestDaemon()
	current := true
	a := modelInventory{models: []string{"ready"}, downloadObservation: downloadObservation{snapshot: scannerDownloadSnapshot("9"), current: &current}}
	if !d.cacheInventoryLocked("target", a) {
		t.Fatal("first source refused")
	}
	b := a
	b.snapshot = scannerDownloadSnapshot("0")
	b.snapshot.SourceID = strings.Repeat("c", 32)
	b.sourceBaseline = a.snapshot.SourceID
	if !d.cacheInventoryLocked("target", b) {
		t.Fatal("current replacement source refused")
	}
	if d.cacheInventoryLocked("target", a) {
		t.Fatal("late prior-source read replaced current manager")
	}
	if got := d.lastDownloads["target"]; !got.snapshot.Equal(*b.snapshot) {
		t.Fatalf("replacement baseline changed: %+v", got)
	}
}

func TestModelDownloadCacheRekeyAndEviction(t *testing.T) {
	d := newRefreshTestDaemon()
	current := true
	state := scannerDownloadSnapshot("4")
	d.cacheInventoryLocked("old", modelInventory{downloadObservation: downloadObservation{snapshot: state, current: &current}})
	d.dropSelf("old", "new")
	if _, exists := d.lastDownloads["old"]; exists {
		t.Fatal("old host retained download cache")
	}
	got := d.lastDownloads["new"]
	if got.snapshot == nil || !got.snapshot.Equal(*state) || got.current == nil || *got.current {
		t.Fatalf("rekey lost details or retained currentness: %+v", got)
	}
	d.forget("new")
	if _, exists := d.lastDownloads["new"]; exists {
		t.Fatal("eviction retained download cache")
	}
}

func TestModelDownloadRefreshRetiresInFlightSource(t *testing.T) {
	initial := scannerDownloadSnapshot("10")
	late := scannerDownloadSnapshot("11")
	replacement := scannerDownloadSnapshot("0")
	replacement.SourceID = strings.Repeat("c", 32)
	entered, release := make(chan struct{}), make(chan struct{})
	var calls atomic.Int32
	srv := httptest.NewServer(fixtureRequestHandler(http.MethodGet, "/v1/models", func(w http.ResponseWriter, r *http.Request) {
		state := initial
		switch calls.Add(1) {
		case 2:
			close(entered)
			<-release
			state = late
		case 3:
			state = replacement
		}
		if err := json.NewEncoder(w).Encode(map[string]any{"models": []string{"ready"}, "downloadState": state}); err != nil {
			t.Errorf("write snapshot: %v", err)
		}
	}))
	t.Cleanup(srv.Close)
	// Ensure a failed assertion also unblocks the fixture handler before Close.
	t.Cleanup(func() {
		select {
		case <-release:
		default:
			close(release)
		}
	})
	d := newRefreshTestDaemon()
	port := portFromURL(t, srv.URL)
	seedEMNode(d, "target", "127.0.0.1", port)
	if !d.refreshNodeModels("target", "127.0.0.1", port, "") {
		t.Fatal("initial state not published")
	}
	done := make(chan bool, 1)
	go func() { done <- d.refreshNodeModels("target", "127.0.0.1", port, "") }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("held read never entered")
	}
	assertFixtureRejectsProbe(t, srv.URL)
	if calls.Load() != 2 {
		t.Fatal("unrelated probe advanced the held inventory sequence")
	}
	if !d.refreshNodeModels("target", "127.0.0.1", port, "") {
		t.Fatal("replacement source not published")
	}
	close(release)
	select {
	case changed := <-done:
		if changed {
			t.Fatal("retired in-flight read emitted a change")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("retired read did not finish")
	}
	node, _ := d.dir.get("target")
	if node.DownloadState == nil || !node.DownloadState.Equal(*replacement) || !*node.DownloadStateCurrent || !d.lastDownloads["target"].snapshot.Equal(*replacement) {
		t.Fatalf("retired read overwrote replacement: %+v", node)
	}
}
