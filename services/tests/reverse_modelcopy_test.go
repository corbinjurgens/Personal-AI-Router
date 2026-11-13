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
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
)

const phase04CutAt = 256 << 10

type phase04FileRequest struct{ path, offset string }

// phase04CopyScenario is one llama.cpp HF-cache model on the source: the main
// GGUF (weights) and its multimodal companion are fetched by their path in the
// snapshot, and the destination publishes refs/main only after both are in place.
type phase04CopyScenario struct {
	model            string
	root             string // destination store root
	snapshot         string // destination snapshot directory, relative to root
	refs             string // destination refs/main, relative to root
	revision         string
	weights          string // source request path of the main GGUF (fetched last)
	companion        string // source request path of the multimodal companion (fetched first)
	data             map[string][]byte
	exposed, release chan struct{}
}

// destination is the path of a snapshot file in the destination store.
func (c *phase04CopyScenario) destination(rel string) string {
	return filepath.Join(c.root, filepath.FromSlash(c.snapshot), filepath.FromSlash(rel))
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
			if _, err := os.Stat(filepath.Join(scenario.root, filepath.FromSlash(scenario.refs))); !errors.Is(err, os.ErrNotExist) {
				t.Errorf("refs/main visible before file publication: %v", err)
			}
			if path == scenario.weights && r.Header.Get("Range") == "" {
				got, err := os.ReadFile(scenario.destination(scenario.companion))
				if err != nil || phase04SHA(got) != phase04SHA(scenario.data[scenario.companion]) {
					t.Errorf("weights requested before verified %s: %v", scenario.companion, err)
				}
			}
			if path == scenario.weights && r.Header.Get("Range") != "" {
				if r.Header.Get("Range") != "bytes=262144-" {
					t.Errorf("resume Range=%q, want bytes=262144-", r.Header.Get("Range"))
				}
				part, err := os.ReadFile(scenario.destination(path) + ".part")
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

// phase04SeedModel writes a llama.cpp HF-cache model into the source store and
// lists it in the source engine's native inventory.
func phase04SeedModel(t *testing.T, f *phase04Fixture, source, destination *task1Node, name string, seed byte) *phase04CopyScenario {
	t.Helper()
	sourceRoot, destinationRoot := f.stores[source.uuid], f.stores[destination.uuid]
	const revision = "0123456789abcdef0123456789abcdef01234567"
	repo := "fixture/" + name + "-GGUF"
	payload := make([]byte, 8<<20)
	for i := range payload {
		payload[i] = byte(i*17) + seed
	}
	weights := append(modelSettingsGGUF("blk.0.weight"), payload...)
	companion := []byte(fmt.Sprintf("multimodal projector %d", seed))
	repoDir := "models--" + strings.ReplaceAll(repo, "/", "--")
	scenario := &phase04CopyScenario{
		model: repo + ":Q4_K_M", root: destinationRoot, snapshot: repoDir + "/snapshots/" + revision, refs: repoDir + "/refs/main", revision: revision,
		weights: "model-Q4_K_M.gguf", companion: "mmproj-model-Q8_0.gguf",
		exposed: make(chan struct{}), release: make(chan struct{}),
	}
	scenario.data = map[string][]byte{scenario.weights: weights, scenario.companion: companion}
	for rel, data := range scenario.data {
		path := filepath.Join(sourceRoot, filepath.FromSlash(scenario.snapshot), filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	refs := filepath.Join(sourceRoot, filepath.FromSlash(scenario.refs))
	if err := os.MkdirAll(filepath.Dir(refs), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(refs, []byte(revision), 0o600); err != nil {
		t.Fatal(err)
	}
	f.engines[source.uuid].addModel(scenario.model)
	// The manager records the engine's inventory observation that a copy export
	// is bound to when it is asked for the models.
	f.control.engines[source.uuid].call(t, "engine:models", nil)
	return scenario
}

// The copy client's authority can be revoked without telling the live cluster
// managers that either peer left. Keep the same identity and admissions, and a
// sibling directory so the production reverse adapter still finds the CM's IPC.
func phase04CopyAuthorityDirectory(t *testing.T, node, peer *task1Node) string {
	t.Helper()
	original := filepath.Join(node.base, "cluster")
	isolated := filepath.Join(node.base, "copy-cluster")
	if err := os.MkdirAll(filepath.Join(isolated, "trusted"), 0o700); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"node.crt", "node.key", "identity.json", "admission.json", "members.json", filepath.Join("trusted", peer.uuid+".json")} {
		data := readLiveDurableFile(t, filepath.Join(original, name))
		path := filepath.Join(isolated, name)
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(readLiveDurableFile(t, path), data) {
			t.Fatalf("copy authority changed %s", name)
		}
	}
	want, ok := node.mesh.PeerServiceIncarnation(peer.uuid)
	got, cloned := clustertrust.Open(isolated).PeerServiceIncarnation(peer.uuid)
	if !ok || !cloned || got != want {
		t.Fatal("copy authority did not preserve current peer admission binding")
	}
	return isolated
}

// The seeded CM pin names its peer by UUID and carries no display name. The
// first production roster exchange, about five seconds after CM startup,
// replaces the name with the sender's own display name, which the peer's CM
// reports as its node name. That is authorized display metadata, not copy
// authority; settle it before the byte snapshot so later equality is
// attributable to the copy pin alone.
func phase04AwaitPinIdentity(t *testing.T, node, peer *task1Node) {
	t.Helper()
	path := filepath.Join(node.base, "cluster", "trusted", peer.uuid+".json")
	want := decodeResult[struct {
		Name string `json:"name"`
	}](t, peer.cm.call(t, "cluster:get-node-id", nil)).Name
	if want == "" {
		t.Fatalf("peer %s reports an empty node name", peer.uuid)
	}
	task1Eventually(t, "CM pin name refreshed by first roster exchange", 45*time.Second, func() bool {
		var pin struct {
			Name string `json:"name"`
		}
		return json.Unmarshal(readLiveDurableFile(t, path), &pin) == nil && pin.Name == want
	})
}

// Await a real production CM roster request and its completed reply while the
// copy authority is invalid. No sender or roster is synthesized by the test.
// Loopback fixtures never build a valid peer-state descriptor, so no stream
// frame triggers an exchange here; the next one is the CM's periodic repair
// pass (reconcileEvery, 5 minutes after the initial pass).
func phase04AwaitRosterExchange(t *testing.T, destination, source *task1Node) {
	t.Helper()
	observer := destination.observers[noderec.ServiceCluster]
	before := observer.count(source.uuid, http.MethodPost, task1RosterPath)
	replied := make(chan struct{}, 1)
	observer.mu.Lock()
	observer.afterReply = func(peer, path string) {
		if peer == source.uuid && path == task1RosterPath && observer.count(source.uuid, http.MethodPost, task1RosterPath) > before {
			select {
			case replied <- struct{}{}:
			default:
			}
		}
	}
	observer.mu.Unlock()
	defer func() {
		observer.mu.Lock()
		observer.afterReply = nil
		observer.mu.Unlock()
	}()
	select {
	case <-replied:
	case <-time.After(6 * time.Minute): // CM repairs every 5 minutes.
		t.Fatal("production CM roster exchange did not complete during invalid copy authority")
	}
}

func TestReverseModelCopyPhase04(t *testing.T) {
	f := newPhase04FixtureWithOptions(t, phase04FixtureOptions{isolateCopyAuthority: true})
	test := func(name, model string, source, destination *task1Node, seed byte) {
		t.Run(name, func(t *testing.T) {
			scenario := phase04SeedModel(t, f, source, destination, model, seed)
			observer := f.files[source.uuid]
			observer.mu.Lock()
			observer.scenario = scenario
			observer.mu.Unlock()
			var releaseOnce sync.Once
			release := func() { releaseOnce.Do(func() { close(scenario.release) }) }
			defer release()
			manager := f.control.engines[destination.uuid]
			id := manager.send(t, "engine:remote-copy-model", map[string]string{"node": source.uuid, "engine": "llamacpp", "model": scenario.model}, true)
			select {
			case <-scenario.exposed:
			case <-time.After(15 * time.Second):
				manager.mu.Lock()
				var frames []string
				for _, frame := range manager.frames {
					if idEquals(frame.ID, id) {
						raw, _ := json.Marshal(frame)
						frames = append(frames, string(raw))
					}
				}
				manager.mu.Unlock()
				t.Fatalf("large file did not stream; frames %v; manager: %s", frames, manager.output.String())
			}
			task1Eventually(t, "resumable partial before visibility", 5*time.Second, func() bool {
				info, err := os.Stat(scenario.destination(scenario.weights) + ".part")
				return err == nil && info.Size() == phase04CutAt
			})
			if _, err := os.Stat(scenario.destination(scenario.weights)); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("unverified weights published: %v", err)
			}
			if _, err := os.Stat(filepath.Join(scenario.root, filepath.FromSlash(scenario.refs))); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("refs/main published with partial: %v", err)
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
				if engine.Engine == "llamacpp" && engine.Installed {
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
			if err != nil || line != "data: {\"model\":\"phase04\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"second\"},\"finish_reason\":\"stop\"}]}\n" {
				t.Fatalf("concurrent inference second chunk=%q err=%v", line, err)
			}
			if tail, err := io.ReadAll(reader); err != nil || string(tail) != "\ndata: [DONE]\n\n" {
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
				got, err := os.ReadFile(scenario.destination(rel))
				if err != nil || phase04SHA(got) != phase04SHA(want) {
					t.Fatalf("published %s hash/bytes failed: %v", rel, err)
				}
				if _, err := os.Stat(scenario.destination(rel) + ".part"); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("partial remains after publication: %v", err)
				}
			}
			if ref, err := os.ReadFile(filepath.Join(scenario.root, filepath.FromSlash(scenario.refs))); err != nil || string(ref) != scenario.revision {
				t.Fatalf("refs/main = %q, %v; want %q", ref, err, scenario.revision)
			}
			if result.Result.Files != len(scenario.data) || result.Result.FilesSkipped != 0 || result.Result.BytesTotal != total || result.Result.BytesCopied != total {
				t.Fatalf("copy result=%+v, want %d new files totaling %d", result.Result, len(scenario.data), total)
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
			if len(requests) != 3 || requests[0].path != scenario.companion || len(ranges) != 2 || ranges[0] != "" || ranges[1] != "bytes=262144-" {
				t.Fatalf("file requests=%+v; want the companion once, then the weights plus one distinct Range", requests)
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
			pin := filepath.Join(f.engineClusterDirs[destination.uuid], "trusted", source.uuid+".json")
			original := readLiveDurableFile(t, pin)
			if filepath.Dir(filepath.Dir(pin)) == filepath.Join(destination.base, "cluster") {
				t.Fatal("negative copy authority shares the CM directory")
			}
			for _, node := range []*task1Node{destination, source} {
				phase04AwaitPinIdentity(t, node, f.peer(node))
			}
			// Snapshot only current identity/admission authority, not changing
			// membership metadata such as last-seen timestamps.
			authority := map[string][]byte{}
			incarnations := map[string]string{}
			for _, node := range []*task1Node{destination, source} {
				peer := f.peer(node)
				for _, name := range []string{"node.crt", "node.key", "admission.json", filepath.Join("trusted", peer.uuid+".json")} {
					path := filepath.Join(node.base, "cluster", name)
					authority[path] = readLiveDurableFile(t, path)
				}
				node.mesh.Refresh()
				incarnation, ok := node.mesh.PeerServiceIncarnation(peer.uuid)
				if !ok {
					t.Fatal("CM lacks original current peer admission")
				}
				incarnations[node.uuid] = incarnation
			}
			copyMesh := clustertrust.Open(f.engineClusterDirs[destination.uuid])
			defer func() {
				if err := os.WriteFile(pin, original, 0o600); err != nil {
					t.Error(err)
					return
				}
				restored, err := os.ReadFile(pin)
				if err != nil {
					t.Errorf("read restored copy pin: %v", err)
					return
				}
				if !bytes.Equal(restored, original) {
					t.Error("restored copy pin differs from original")
				}
				copyMesh.Refresh()
				got, ok := copyMesh.PeerServiceIncarnation(source.uuid)
				if !ok || got != incarnations[destination.uuid] {
					t.Error("restored copy authority changed original current peer admission binding")
				}
			}()
			observer := f.files[source.uuid]
			observer.mu.Lock()
			listed, fetched := observer.listed, len(observer.requests)
			observer.mu.Unlock()
			assertRejected := func() {
				id := f.control.engines[destination.uuid].send(t, "engine:remote-copy-model", map[string]string{"node": source.uuid, "engine": "llamacpp", "model": model}, true)
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
			assertIsolated := func() {
				for path, want := range authority {
					if !bytes.Equal(readLiveDurableFile(t, path), want) {
						t.Fatalf("negative copy authority changed CM %s", filepath.Base(path))
					}
				}
				for _, node := range []*task1Node{destination, source} {
					node.mesh.Refresh()
					got, ok := node.mesh.PeerServiceIncarnation(f.peer(node).uuid)
					if !ok || got != incarnations[node.uuid] {
						t.Fatal("negative copy authority changed CM current membership/admission binding")
					}
				}
			}
			task1WriteJSON(t, pin, map[string]any{"nodeUuid": source.uuid, "nodeId": source.uuid, "certPem": destination.cert, "clusterId": task1ClusterID, "admissionEpoch": 1})
			assertRejected()
			wrongPin := readLiveDurableFile(t, pin)
			if source == f.control.b {
				// Exercise the former pruning race once, in the first reverse
				// wrong-pin window. Every state still checks both CM bindings.
				phase04AwaitRosterExchange(t, destination, source)
			}
			assertIsolated()
			if !bytes.Equal(readLiveDurableFile(t, pin), wrongPin) {
				t.Fatal("CM roster exchange repaired the wrong copy pin")
			}
			assertRejected()
			removeAuthorityFile(t, pin)
			assertRejected()
			assertIsolated()
			if _, err := os.Stat(pin); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("CM roster exchange restored removed copy pin: %v", err)
			}
			assertRejected()
		})
	}
	testAuthority("wrong and removed reverse pins fail before listing or bytes handlers", "fixture/from-restricted-GGUF:Q4_K_M", f.control.a, f.control.b)
	testAuthority("wrong and removed direct pins fail before listing or bytes handlers", "fixture/from-reachable-GGUF:Q4_K_M", f.control.b, f.control.a)
}
