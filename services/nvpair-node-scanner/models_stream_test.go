// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/noderec"
)

// modelsStreamFixture is a scanner pinned to one cluster peer ("peer" at the
// documentation address 192.0.2.41) whose engine-manager em surface is an mTLS
// httptest server on loopback, presenting the peer's real cluster leaf. The
// scanner's direct dialer maps the advertised em address to that server; the
// advertised node-info port is refused, so only the models stream connects.
type modelsStreamFixture struct {
	d       *daemon
	node    noderec.DirectoryNode
	emPort  int
	updates chan noderec.DirectoryNode
	fetches atomic.Int32
	streams atomic.Int32
	// lines carries what the stream handler writes after its first frame; a
	// closed stream ends when end receives.
	lines  chan string
	end    chan struct{}
	ended  chan struct{}
	mu     sync.Mutex
	first  map[string]any
	status int
}

func newModelsStreamFixture(t *testing.T, first map[string]any) *modelsStreamFixture {
	t.Helper()
	selfDir := filepath.Join(t.TempDir(), "cluster")
	peerDir := t.TempDir()
	clustertrusttest.WriteKeypair(t, selfDir, "self")
	clustertrusttest.WriteKeypair(t, peerDir, "peer")
	clustertrusttest.WriteAdmission(t, selfDir, "cluster", 1)
	clustertrusttest.WriteAdmission(t, peerDir, "cluster", 1)
	writePinFromCert(t, selfDir, "peer", filepath.Join(peerDir, "node.crt"))
	writePinFromCert(t, peerDir, "self", filepath.Join(selfDir, "node.crt"))
	writeScannerAdmissionPin(t, selfDir, "peer", "cluster", 1)
	peerMesh := clustertrust.Open(peerDir)

	d := newMembershipTestDaemon(selfDir)
	d.modelsHTTP = &http.Client{Timeout: modelsFetchTimeout}
	t.Cleanup(d.closePeerClients)
	f := &modelsStreamFixture{d: d, updates: make(chan noderec.DirectoryNode, 64), lines: make(chan string, 16),
		end: make(chan struct{}, 4), ended: make(chan struct{}, 16), first: first}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		f.fetches.Add(1)
		f.mu.Lock()
		body := f.first
		f.mu.Unlock()
		_ = json.NewEncoder(w).Encode(body)
	})
	mux.HandleFunc(modelsStreamPath, func(w http.ResponseWriter, r *http.Request) {
		if _, ok := peerMesh.VerifyClientPin(r); !ok {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		f.streams.Add(1)
		defer func() { f.ended <- struct{}{} }()
		f.mu.Lock()
		body, status := f.first, f.status
		f.mu.Unlock()
		if status != 0 {
			w.WriteHeader(status)
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		if err := json.NewEncoder(w).Encode(body); err != nil {
			return
		}
		w.(http.Flusher).Flush()
		for {
			select {
			case <-r.Context().Done():
				return
			case <-f.end:
				return
			case line := <-f.lines:
				if _, err := io.WriteString(w, line+"\n"); err != nil {
					return
				}
				w.(http.Flusher).Flush()
			}
		}
	})
	server := httptest.NewUnstartedServer(mux)
	server.TLS = peerMesh.ServerTLSConfig()
	server.StartTLS()
	t.Cleanup(server.Close)
	addr := server.Listener.Addr().(*net.TCPAddr)
	f.emPort = addr.Port
	advertised := net.JoinHostPort("192.0.2.41", strconv.Itoa(f.emPort))
	dialer := &net.Dialer{}
	d.peerDialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		if address != advertised {
			return nil, errors.New("models fixture refuses unknown direct destination")
		}
		return dialer.DialContext(ctx, network, addr.String())
	}
	f.node = noderec.DirectoryNode{HostUUID: "peer", ClusterUUID: "peer", IP: "192.0.2.41", Trusted: true,
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{
			noderec.ServiceNodeInfo:      {Port: 1},
			noderec.ServiceEngineManager: {Port: f.emPort},
		}}
	d.dir.upsert(f.node)

	reader, writer := io.Pipe()
	d.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{eofReader{}, writer})
	go func() {
		scanner := bufio.NewScanner(reader)
		scanner.Buffer(make([]byte, 4096), 1<<20)
		for scanner.Scan() {
			var message Message
			if json.Unmarshal(scanner.Bytes(), &message) != nil || message.Method != noderec.NotifyNodeUpdated {
				continue
			}
			var event noderec.NodeEvent
			if json.Unmarshal(message.Params, &event) == nil && event.Node.HostUUID == "peer" {
				f.updates <- event.Node
			}
		}
	}()
	t.Cleanup(func() { _ = writer.Close() })
	return f
}

func (f *modelsStreamFixture) setStatus(status int) {
	f.mu.Lock()
	f.status = status
	f.mu.Unlock()
}

func (f *modelsStreamFixture) supervise(t *testing.T) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	var workers sync.WaitGroup
	t.Cleanup(func() { cancel(); workers.Wait() })
	f.d.supervisePeerStreams(ctx, f.d.selfHostUUID(), &workers)
}

func (f *modelsStreamFixture) nextUpdate(t *testing.T) noderec.DirectoryNode {
	t.Helper()
	return f.nextUpdateWithin(t, 3*time.Second)
}

func (f *modelsStreamFixture) nextUpdateWithin(t *testing.T, within time.Duration) noderec.DirectoryNode {
	t.Helper()
	select {
	case node := <-f.updates:
		return node
	case <-time.After(within):
		t.Fatal("no discovery:node-updated emitted")
		return noderec.DirectoryNode{}
	}
}

func (f *modelsStreamFixture) noUpdate(t *testing.T, within time.Duration) {
	t.Helper()
	select {
	case node := <-f.updates:
		t.Fatalf("unexpected discovery:node-updated: %+v", node)
	case <-time.After(within):
	}
}

func modelsFrame(t *testing.T, body map[string]any) string {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func TestModelsStreamAppliesFramesOnChange(t *testing.T) {
	first := map[string]any{"models": []string{"a"}, "modelsByEngine": map[string][]string{"llamacpp": {"a"}}}
	f := newModelsStreamFixture(t, first)
	f.supervise(t)
	f.d.nodeInfoStreams.mu.Lock()
	_, supervised := f.d.nodeInfoStreams.models["peer"]
	f.d.nodeInfoStreams.mu.Unlock()
	if !supervised {
		t.Fatal("supervisor started no models stream for a stream peer advertising engine-manager")
	}

	node := f.nextUpdate(t)
	if want := []string{"a"}; !reflect.DeepEqual(node.Models, want) {
		t.Fatalf("first frame Models = %v, want %v", node.Models, want)
	}

	f.lines <- modelsFrame(t, first)
	f.lines <- "null"
	f.noUpdate(t, 300*time.Millisecond)

	f.lines <- modelsFrame(t, map[string]any{"models": []string{"a", "b"}, "modelsByEngine": map[string][]string{"llamacpp": {"a", "b"}}})
	node = f.nextUpdate(t)
	if want := []string{"a", "b"}; !reflect.DeepEqual(node.Models, want) {
		t.Fatalf("changed frame Models = %v, want %v", node.Models, want)
	}
	f.noUpdate(t, 300*time.Millisecond)
	if got := f.fetches.Load(); got != 0 {
		t.Fatalf("stream peer received %d /v1/models fetches", got)
	}
}

func TestModelsStreamEndMarksDownloadsStaleOnce(t *testing.T) {
	first := map[string]any{"models": []string{"a"}, "downloadState": scannerDownloadSnapshot("3")}
	f := newModelsStreamFixture(t, first)
	f.supervise(t)
	node := f.nextUpdate(t)
	if node.DownloadState == nil || node.DownloadStateCurrent == nil || !*node.DownloadStateCurrent {
		t.Fatalf("first frame download state = %+v current=%v, want current", node.DownloadState, node.DownloadStateCurrent)
	}

	// Keep the peer's stream down after it ends, so the reconnect also fails.
	f.setStatus(http.StatusServiceUnavailable)
	f.end <- struct{}{}
	node = f.nextUpdate(t)
	if node.DownloadState == nil || node.DownloadStateCurrent == nil || *node.DownloadStateCurrent {
		t.Fatalf("after stream end download state = %+v current=%v, want retained and stale", node.DownloadState, node.DownloadStateCurrent)
	}
	if want := []string{"a"}; !reflect.DeepEqual(node.Models, want) {
		t.Fatalf("after stream end Models = %v, want the last inventory %v retained", node.Models, want)
	}
	// The refused reconnect (after the first backoff) takes the same path
	// and must stay silent.
	before := f.streams.Load()
	deadline := time.Now().Add(10 * time.Second)
	for f.streams.Load() == before {
		if time.Now().After(deadline) {
			t.Fatal("scanner did not retry the models stream")
		}
		time.Sleep(50 * time.Millisecond)
	}
	f.noUpdate(t, 300*time.Millisecond)
}

func TestRefreshModelsOnceSkipsStreamPeers(t *testing.T) {
	f := newModelsStreamFixture(t, map[string]any{"models": []string{"peer-model"}})
	_, selfPort := startModelsStub(t, http.StatusOK, map[string]any{"models": []string{"self-model"}})
	self := f.d.selfHostUUID()
	f.d.dir.upsert(noderec.DirectoryNode{HostUUID: self, Name: self, IP: "192.0.2.1",
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineManager: {Port: selfPort}}})

	f.d.refreshModelsOnce()

	if got := f.fetches.Load(); got != 0 {
		t.Fatalf("refreshModelsOnce fetched /v1/models from a stream peer %d times", got)
	}
	selfNode, _ := f.d.dir.get(self)
	if want := []string{"self-model"}; !reflect.DeepEqual(selfNode.Models, want) {
		t.Fatalf("self Models = %v, want %v from the loopback fetch", selfNode.Models, want)
	}

	// The same pinned peer without node-info is not a stream peer and keeps
	// the fetch.
	node := f.node
	node.Services = map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineManager: {Port: f.emPort}}
	f.d.dir.upsert(node)
	if f.d.isStreamPeer(node, self) {
		t.Fatal("a peer advertising no node-info counts as a stream peer")
	}
	f.d.refreshModelsOnce()
	if got := f.fetches.Load(); got != 1 {
		t.Fatalf("non-stream peer fetches = %d, want 1", got)
	}
}

// TestModelsStreamOwnsBrowseEnrichment: while a models stream is live, browse
// enrichment applies the stream's last frame instead of fetching, so a browse
// event can neither replace a newer frame nor withdraw freshness that no later
// frame would restore.
func TestModelsStreamOwnsBrowseEnrichment(t *testing.T) {
	first := map[string]any{"models": []string{"a"}, "downloadState": scannerDownloadSnapshot("3")}
	f := newModelsStreamFixture(t, first)
	f.supervise(t)
	f.nextUpdate(t)
	deadline := time.Now().Add(3 * time.Second)
	for !f.d.modelsStreamLive("peer") {
		if time.Now().After(deadline) {
			t.Fatal("models stream never became live after its first frame")
		}
		time.Sleep(10 * time.Millisecond)
	}

	browsed := f.node
	browsed.Models, browsed.DownloadState, browsed.DownloadStateCurrent = nil, nil, nil
	f.d.enrichModelsCandidates(&browsed, browsed.CandidateIPs())
	if got := f.fetches.Load(); got != 0 {
		t.Fatalf("browse enrichment fetched /v1/models %d times from a live stream peer", got)
	}
	if want := []string{"a"}; !reflect.DeepEqual(browsed.Models, want) {
		t.Fatalf("browse enrichment Models = %v, want the stream's %v", browsed.Models, want)
	}
	if browsed.DownloadStateCurrent == nil || !*browsed.DownloadStateCurrent {
		t.Fatalf("browse enrichment withdrew download freshness: current=%v", browsed.DownloadStateCurrent)
	}

	// Once the stream ends, browse enrichment fetches again.
	f.setStatus(http.StatusServiceUnavailable)
	f.end <- struct{}{}
	f.nextUpdate(t)
	if f.d.modelsStreamLive("peer") {
		t.Fatal("an ended models stream still counts as live")
	}
	f.d.enrichModelsCandidates(&browsed, browsed.CandidateIPs())
	if got := f.fetches.Load(); got != 1 {
		t.Fatalf("browse enrichment fetches after the stream ended = %d, want 1", got)
	}
}

// TestModelsStreamEndAfterRouteChangeDefersStale: a stream that ended after the
// node's route generation moved (for example when its own frame carried a new
// download source) does not withdraw freshness; the next attempt, begun on the
// new route, does when it fails.
func TestModelsStreamEndAfterRouteChangeDefersStale(t *testing.T) {
	first := map[string]any{"models": []string{"a"}, "downloadState": scannerDownloadSnapshot("3")}
	f := newModelsStreamFixture(t, first)
	f.supervise(t)
	f.nextUpdate(t)

	f.d.dir.mu.Lock()
	f.d.dir.generations["peer"]++
	f.d.dir.mu.Unlock()
	f.setStatus(http.StatusServiceUnavailable)
	f.end <- struct{}{}
	f.noUpdate(t, 500*time.Millisecond)

	node := f.nextUpdateWithin(t, 10*time.Second) // the refused retry, after the first backoff
	if node.DownloadStateCurrent == nil || *node.DownloadStateCurrent {
		t.Fatalf("failed retry on the new route did not withdraw freshness: current=%v", node.DownloadStateCurrent)
	}
}

func TestModelsStreamDropsSupersededFrames(t *testing.T) {
	d := &daemon{}
	d.nodeInfoStreams.models = map[string]nodeInfoStreamHandle{"peer": {id: 2}}
	if d.activeModelsStream("peer", 1) {
		t.Fatal("a superseded models stream counts as active")
	}
	d.markModelsLive("peer", 1)
	if d.modelsStreamLive("peer") {
		t.Fatal("a superseded models stream marked the host live")
	}
	d.markModelsLive("peer", 2)
	d.dropModelsLive("peer", 1)
	if !d.modelsStreamLive("peer") {
		t.Fatal("a superseded stream's end dropped its replacement's liveness")
	}
}
