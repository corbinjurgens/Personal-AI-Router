// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"nvpair-shared/noderec"
)

const phase04CutAt = 256 << 10

type phase04FileRequest struct{ path, offset string }

type phase04CopyScenario struct {
	root, manifest, weights string
	data                    map[string][]byte
	exposed, release        chan struct{}
}

// This observer forwards authenticated requests to the real engine-manager.
// It never substitutes listing, Range, hash, or publication behavior. Its only
// fault is truncating one exposed file response, after the destination has data.
type phase04FileObserver struct {
	port     int
	mu       sync.Mutex
	requests []phase04FileRequest
	listed   int
	controls int
	scenario *phase04CopyScenario
}

type phase04CutBody struct {
	io.ReadCloser
	remaining int
	ctx       context.Context
	exposed   chan struct{}
	release   chan struct{}
	once      sync.Once
}

func (b *phase04CutBody) Read(p []byte) (int, error) {
	if b.remaining == 0 {
		b.once.Do(func() { close(b.exposed) })
		select {
		case <-b.release:
		case <-b.ctx.Done():
		}
		return 0, io.ErrUnexpectedEOF
	}
	if len(p) > b.remaining {
		p = p[:b.remaining]
	}
	n, err := b.ReadCloser.Read(p)
	b.remaining -= n
	return n, err
}

func newPhase04FileObserver(t *testing.T, target, caller *task1Node) *phase04FileObserver {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	o := &phase04FileObserver{port: ln.Addr().(*net.TCPAddr).Port}
	cfg, ok := caller.mesh.ClientTLSConfig(target.uuid)
	if !ok {
		t.Fatal("missing observer target pin")
	}
	u, err := url.Parse(fmt.Sprintf("https://127.0.0.1:%d", target.ports[noderec.ServiceEngineControl]))
	if err != nil {
		t.Fatal(err)
	}
	transport := &http.Transport{TLSClientConfig: cfg, DisableKeepAlives: true}
	t.Cleanup(transport.CloseIdleConnections)
	proxy := httputil.NewSingleHostReverseProxy(u)
	proxy.Transport = transport
	proxy.FlushInterval = -1
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		http.Error(w, "fixture target unavailable", http.StatusBadGateway)
	}
	proxy.ModifyResponse = func(response *http.Response) error {
		o.mu.Lock()
		scenario := o.scenario
		o.mu.Unlock()
		if scenario != nil && response.Request.URL.Path == "/v1/models/file" && response.Request.URL.Query().Get("path") == scenario.weights && response.Request.Header.Get("Range") == "" && response.StatusCode == http.StatusOK {
			response.Body = &phase04CutBody{ReadCloser: response.Body, remaining: phase04CutAt, ctx: response.Request.Context(), exposed: scenario.exposed, release: scenario.release}
		}
		return nil
	}
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		target.mesh.Refresh()
		peer, pinned := target.mesh.VerifyClientPin(r)
		if !pinned || peer != caller.uuid {
			http.Error(w, "unpinned observer caller", http.StatusForbidden)
			return
		}
		o.mu.Lock()
		scenario := o.scenario
		if r.URL.Path == "/v1/models/files" {
			o.listed++
		}
		if r.URL.Path == "/v1/engines" {
			o.controls++
		}
		if r.URL.Path == "/v1/models/file" {
			o.requests = append(o.requests, phase04FileRequest{r.URL.Query().Get("path"), r.Header.Get("Range")})
		}
		o.mu.Unlock()
		if scenario != nil && r.URL.Path == "/v1/models/file" {
			path := r.URL.Query().Get("path")
			if _, err := os.Stat(filepath.Join(scenario.root, filepath.FromSlash(scenario.manifest))); !errors.Is(err, os.ErrNotExist) {
				t.Errorf("manifest visible before file publication: %v", err)
			}
			if path == scenario.manifest {
				for rel, want := range scenario.data {
					if rel == scenario.manifest {
						continue
					}
					got, err := os.ReadFile(filepath.Join(scenario.root, filepath.FromSlash(rel)))
					if err != nil || phase04SHA(got) != phase04SHA(want) {
						t.Errorf("visibility requested before verified %s: %v", rel, err)
					}
				}
			}
			if path == scenario.weights && r.Header.Get("Range") != "" {
				if r.Header.Get("Range") != "bytes=262144-" {
					t.Errorf("resume Range=%q, want bytes=262144-", r.Header.Get("Range"))
				}
				part, err := os.ReadFile(filepath.Join(scenario.root, filepath.FromSlash(path)) + ".part")
				if err != nil || !bytes.Equal(part, scenario.data[path][:phase04CutAt]) {
					t.Errorf("resume partial bytes=%d err=%v", len(part), err)
				}
			}
		}
		proxy.ServeHTTP(w, r)
	})}
	done := make(chan error, 1)
	go func() { done <- server.Serve(tls.NewListener(ln, target.mesh.ServerTLSConfig())) }()
	t.Cleanup(func() {
		if err := server.Close(); err != nil {
			t.Error(err)
		}
		if err := <-done; err != http.ErrServerClosed {
			t.Errorf("streaming observer cleanup: %v", err)
		}
	})
	return o
}

func phase04SHA(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func phase04SeedModel(t *testing.T, sourceRoot, destinationRoot, name string, seed byte) *phase04CopyScenario {
	t.Helper()
	weights := make([]byte, 8<<20)
	for i := range weights {
		weights[i] = byte(i*17) + seed
	}
	config := []byte(fmt.Sprintf("{\"model_format\":\"gguf\",\"seed\":%d}", seed))
	manifest := "manifests/registry.ollama.ai/library/" + name + "/latest"
	scenario := &phase04CopyScenario{
		root: destinationRoot, manifest: manifest, weights: "blobs/sha256-" + phase04SHA(weights),
		data:    map[string][]byte{"blobs/sha256-" + phase04SHA(weights): weights, "blobs/sha256-" + phase04SHA(config): config},
		exposed: make(chan struct{}), release: make(chan struct{}),
	}
	body := map[string]any{
		"schemaVersion": 2,
		"config":        map[string]any{"digest": "sha256:" + phase04SHA(config), "size": len(config)},
		"layers":        []map[string]any{{"digest": "sha256:" + phase04SHA(weights), "size": len(weights)}},
	}
	manifestData, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	scenario.data[manifest] = manifestData
	for rel, data := range scenario.data {
		path := filepath.Join(sourceRoot, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return scenario
}

func TestReverseModelCopyPhase04(t *testing.T) {
	f := newPhase04Fixture(t)
	test := func(name, model string, source, destination *task1Node, seed byte) {
		t.Run(name, func(t *testing.T) {
			scenario := phase04SeedModel(t, f.stores[source.uuid], f.stores[destination.uuid], model, seed)
			observer := f.files[source.uuid]
			observer.mu.Lock()
			observer.scenario = scenario
			observer.mu.Unlock()
			var releaseOnce sync.Once
			release := func() { releaseOnce.Do(func() { close(scenario.release) }) }
			defer release()
			manager := f.control.engines[destination.uuid]
			id := manager.send(t, "engine:remote-copy-model", map[string]string{"node": source.uuid, "engine": "ollama", "model": model + ":latest"}, true)
			select {
			case <-scenario.exposed:
			case <-time.After(15 * time.Second):
				t.Fatalf("large file did not stream; manager: %s", manager.output.String())
			}
			task1Eventually(t, "resumable partial before visibility", 5*time.Second, func() bool {
				info, err := os.Stat(filepath.Join(scenario.root, filepath.FromSlash(scenario.weights)) + ".part")
				return err == nil && info.Size() == phase04CutAt
			})
			if _, err := os.Stat(filepath.Join(scenario.root, filepath.FromSlash(scenario.weights))); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("unverified weights published: %v", err)
			}
			if _, err := os.Stat(filepath.Join(scenario.root, filepath.FromSlash(scenario.manifest))); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("manifest published with partial: %v", err)
			}
			// Hold the copy response open while inference and control execute on
			// the same peer pair. The stream's first chunk precedes control.
			inference := phase04Request(t, context.Background(), destination, "concurrent-"+model)
			defer inference.Body.Close()
			reader := phase04FirstChunk(t, inference)
			observer.mu.Lock()
			controlsBefore := observer.controls
			observer.mu.Unlock()
			control := manager.call(t, "engine:remote-get-installed", map[string]string{"node": source.uuid})
			var installed struct {
				Engines []struct {
					Engine    string `json:"engine"`
					Installed bool   `json:"installed"`
				} `json:"engines"`
			}
			task2Decode(t, control.Result, &installed)
			found := false
			for _, engine := range installed.Engines {
				if engine.Engine == "ollama" && engine.Installed {
					found = true
				}
			}
			if !found {
				t.Fatalf("concurrent control result=%+v", installed)
			}
			observer.mu.Lock()
			controlsAfter := observer.controls
			observer.mu.Unlock()
			if controlsAfter != controlsBefore+1 {
				t.Fatalf("concurrent target controls=%d, want %d", controlsAfter, controlsBefore+1)
			}
			select {
			case f.engines[source.uuid].release <- struct{}{}:
			case <-time.After(5 * time.Second):
				t.Fatal("concurrent inference stopped before control completed")
			}
			line, err := reader.ReadString('\n')
			if err != nil || line != "{\"model\":\"phase04:latest\",\"response\":\"second\",\"done\":true}\n" {
				t.Fatalf("concurrent inference second chunk=%q err=%v", line, err)
			}
			if tail, err := io.ReadAll(reader); err != nil || len(tail) != 0 {
				t.Fatalf("concurrent inference tail=%q err=%v", tail, err)
			}
			manager.mu.Lock()
			for _, frame := range manager.frames {
				if idEquals(frame.ID, id) {
					t.Error("copy settled before fault release")
				}
			}
			manager.mu.Unlock()
			if source == f.control.b {
				outer := f.control.a.observers[noderec.ServiceCluster]
				before := outer.established(source.uuid)
				outer.interrupt(source.uuid, true)
				release()
				outer.interrupt(source.uuid, false)
				task1Eventually(t, "copy reverse-session recovery", 15*time.Second, func() bool { return outer.established(source.uuid) > before })
			} else {
				release()
			}
			response := task2Response(t, manager, id)
			if response.Error != nil {
				t.Fatalf("resumed production copy: %+v", response.Error)
			}
			var result struct {
				Result struct {
					Files        int   `json:"files"`
					FilesSkipped int   `json:"filesSkipped"`
					BytesTotal   int64 `json:"bytesTotal"`
					BytesCopied  int64 `json:"bytesCopied"`
				} `json:"result"`
			}
			task2Decode(t, response.Result, &result)
			var total int64
			for rel, want := range scenario.data {
				total += int64(len(want))
				got, err := os.ReadFile(filepath.Join(scenario.root, filepath.FromSlash(rel)))
				if err != nil || phase04SHA(got) != phase04SHA(want) {
					t.Fatalf("published %s hash/bytes failed: %v", rel, err)
				}
				if _, err := os.Stat(filepath.Join(scenario.root, filepath.FromSlash(rel)) + ".part"); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("partial remains after publication: %v", err)
				}
			}
			if result.Result.Files != 3 || result.Result.FilesSkipped != 0 || result.Result.BytesTotal != total || result.Result.BytesCopied != total {
				t.Fatalf("copy result=%+v, want three new files totaling %d", result.Result, total)
			}
			observer.mu.Lock()
			requests := append([]phase04FileRequest(nil), observer.requests...)
			observer.scenario = nil
			observer.mu.Unlock()
			var ranges []string
			for _, request := range requests {
				if request.path == scenario.weights {
					ranges = append(ranges, request.offset)
				}
			}
			if len(requests) != 4 || requests[3].path != scenario.manifest || len(ranges) != 2 || ranges[0] != "" || ranges[1] != "bytes=262144-" {
				t.Fatalf("file requests=%+v; want each blob once plus one distinct Range, visibility last", requests)
			}
			if got := f.engines[source.uuid].count("concurrent-" + model); got != 1 {
				t.Fatalf("concurrent inference dispatches=%d, want 1", got)
			}
		})
	}
	test("restricted source resumes over reverse while inference and control run", "from-restricted", f.control.b, f.control.a, 3)
	test("reachable source resumes over direct while inference and control run", "from-reachable", f.control.a, f.control.b, 5)
	testAuthority := func(name, model string, destination, source *task1Node) {
		t.Run(name, func(t *testing.T) {
			pin := filepath.Join(destination.base, "cluster", "trusted", source.uuid+".json")
			original, err := os.ReadFile(pin)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := os.WriteFile(pin, original, 0o600); err != nil {
					t.Error(err)
				}
			}()
			observer := f.files[source.uuid]
			observer.mu.Lock()
			listed, fetched := observer.listed, len(observer.requests)
			observer.mu.Unlock()
			assertRejected := func() {
				id := f.control.engines[destination.uuid].send(t, "engine:remote-copy-model", map[string]string{"node": source.uuid, "engine": "ollama", "model": model}, true)
				response := task2Response(t, f.control.engines[destination.uuid], id)
				if response.Error == nil {
					t.Fatal("invalid pin accepted copy")
				}
				observer.mu.Lock()
				defer observer.mu.Unlock()
				if observer.listed != listed || len(observer.requests) != fetched {
					t.Fatal("invalid pin reached source listing/bytes handler")
				}
			}
			task1WriteJSON(t, pin, map[string]any{"nodeUuid": source.uuid, "nodeId": source.uuid, "certPem": destination.cert, "clusterId": task1ClusterID, "admissionEpoch": 1})
			assertRejected()
			removeAuthorityFile(t, pin)
			assertRejected()
		})
	}
	testAuthority("wrong and removed reverse pins fail before listing or bytes handlers", "from-restricted:latest", f.control.a, f.control.b)
	testAuthority("wrong and removed direct pins fail before listing or bytes handlers", "from-reachable:latest", f.control.b, f.control.a)
}
