// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bufio"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/modeldownload"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-shared/testbin"
)

const downloadProcessEngine = "download-fixture"
const downloadProcessModel = "fixture/catalog:Q4_K_M"

// The registered joint-target-engine image only changes the real manager's ec
// bind port. Stdio and worker lifetime use the existing owned port adapter, so
// the broker's service registration can name the pinned observation proxy.
func init() {
	if bridge := os.Getenv("NVPAIR_TEST_DOWNLOAD_MANAGER_BRIDGE"); bridge != "" && bridge == os.Args[0] {
		port, err := strconv.Atoi(os.Getenv("NVPAIR_TEST_DOWNLOAD_CONTROL_PORT"))
		binary := os.Getenv("NVPAIR_TEST_DOWNLOAD_MANAGER_BINARY")
		if err == nil {
			err = validateFixturePort(port)
		}
		if err == nil && !filepath.IsAbs(binary) {
			err = fmt.Errorf("download manager adapter requires an absolute worker path")
		}
		if err == nil {
			args := append(os.Args[1:], "--control-port", strconv.Itoa(port))
			err = runOwnedPortWorker(binary, args)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
}

func downloadProcessShardName(model string) string {
	return strings.NewReplacer("/", "_", ":", "_").Replace(model) + "-00001-of-00002.gguf"
}

// This fixture supplies only native router responses. Lifecycle, admission,
// persistence, wire identities and relay notifications belong to real workers.
// A router download outlives its SSE subscriber until an explicit unload.
type downloadProcessRouter struct {
	t             *testing.T
	server        *httptest.Server
	root          string
	mu            sync.Mutex
	status        map[string]string
	posts         map[string]int
	unloads       map[string]int
	loads         int
	mode          map[string]string
	streams       map[chan string]bool
	started       chan string
	holdSubscribe <-chan struct{}
	subscribed    chan struct{}
}

func newDownloadProcessRouter(t *testing.T) *downloadProcessRouter {
	t.Helper()
	r := &downloadProcessRouter{t: t, root: t.TempDir(), status: map[string]string{}, posts: map[string]int{}, unloads: map[string]int{}, mode: map[string]string{}, streams: map[chan string]bool{}, started: make(chan string, 64)}
	r.server = newFixtureHTTPServer(t, http.HandlerFunc(r.serve))
	t.Cleanup(r.server.Close)
	return r
}

func (r *downloadProcessRouter) serve(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if req.Method == http.MethodGet && req.URL.Path == "/models/sse" {
		r.stream(w, req)
		return
	}
	if req.Method == http.MethodGet && req.URL.Path == "/models" {
		r.mu.Lock()
		rows := make([]map[string]any, 0, len(r.status))
		for model, status := range r.status {
			rows = append(rows, map[string]any{"id": model, "status": map[string]string{"value": status}})
		}
		r.mu.Unlock()
		r.encode(w, map[string]any{"data": rows})
		return
	}
	if req.Method == http.MethodGet && req.URL.Path == "/ready" {
		r.encode(w, map[string]string{"role": "router"})
		return
	}
	if req.Method != http.MethodPost {
		http.NotFound(w, req)
		return
	}
	var body struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(io.LimitReader(req.Body, 4096)).Decode(&body); err != nil || !strings.HasPrefix(body.Model, "fixture/") {
		r.t.Errorf("unexpected native request %s: model=%q error=%v", req.URL.Path, body.Model, err)
		http.Error(w, "invalid synthetic model", http.StatusBadRequest)
		return
	}
	r.mu.Lock()
	mode := r.mode[body.Model]
	switch req.URL.Path {
	case "/models":
		r.posts[body.Model]++
		r.status[body.Model] = "downloading"
	case "/models/unload":
		r.unloads[body.Model]++
		if mode != "cleanup failure" {
			r.status[body.Model] = "downloaded"
		}
	case "/models/load":
		r.loads++
		r.status[body.Model] = "loaded"
	default:
		r.mu.Unlock()
		http.NotFound(w, req)
		return
	}
	r.mu.Unlock()
	if req.URL.Path == "/models" {
		// The first finalized shard and list membership precede native terminal.
		// Its name is fixture-local; no real cache layout or model is consumed.
		if err := os.WriteFile(filepath.Join(r.root, downloadProcessShardName(body.Model)), []byte("GGUFfixture"), 0o600); err != nil {
			r.t.Errorf("write first synthetic shard: %v", err)
			http.Error(w, "fixture write failed", http.StatusInternalServerError)
			return
		}
		r.started <- body.Model
		if mode == "uncertain acceptance" {
			// Native effect occurred, but its acceptance response is lost.
			conn, _, err := http.NewResponseController(w).Hijack()
			if err != nil {
				r.t.Errorf("disconnect acceptance: %v", err)
				return
			}
			if err := conn.Close(); err != nil {
				r.t.Errorf("close acceptance connection: %v", err)
			}
			return
		}
	}
	if req.URL.Path == "/models/unload" && mode == "cleanup failure" {
		http.Error(w, "fixture refuses cleanup", http.StatusServiceUnavailable)
		return
	}
	r.encode(w, map[string]bool{"success": true})
}

func (r *downloadProcessRouter) encode(w http.ResponseWriter, value any) {
	if err := json.NewEncoder(w).Encode(value); err != nil {
		r.t.Errorf("encode native response: %v", err)
	}
}

func (r *downloadProcessRouter) stream(w http.ResponseWriter, req *http.Request) {
	events := make(chan string, 8)
	r.mu.Lock()
	r.streams[events] = true
	hold, subscribed := r.holdSubscribe, r.subscribed
	r.mu.Unlock()
	defer func() { r.mu.Lock(); delete(r.streams, events); r.mu.Unlock() }()
	if subscribed != nil {
		select {
		case subscribed <- struct{}{}:
		default:
		}
	}
	if hold != nil {
		select {
		case <-hold:
		case <-req.Context().Done():
			return
		}
	}
	w.Header().Set("Content-Type", "text/event-stream")
	if _, err := io.WriteString(w, ": fixture ready\n\n"); err != nil {
		return
	}
	if err := http.NewResponseController(w).Flush(); err != nil {
		return
	}
	for {
		select {
		case <-req.Context().Done():
			return
		case event := <-events:
			if event == "disconnect" {
				return
			}
			if _, err := fmt.Fprintf(w, "data: %s\n\n", event); err != nil {
				return
			}
			if err := http.NewResponseController(w).Flush(); err != nil {
				return
			}
		}
	}
}

func (r *downloadProcessRouter) event(t *testing.T, model, event string) {
	t.Helper()
	data, err := json.Marshal(map[string]any{"model": model, "event": event, "data": map[string]any{"progress": map[string]any{"first.gguf": map[string]int{"done": 21, "total": 100}}}})
	if err != nil {
		t.Fatal(err)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.streams) == 0 {
		t.Fatal("no native SSE subscription at event barrier")
	}
	for stream := range r.streams {
		select {
		case stream <- string(data):
		default:
			t.Fatal("native SSE fixture event budget exceeded")
		}
	}
}

func (r *downloadProcessRouter) disconnect(t *testing.T) {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	for stream := range r.streams {
		select {
		case stream <- "disconnect":
		default:
			t.Fatal("native disconnect event budget exceeded")
		}
	}
}

func (r *downloadProcessRouter) setStatus(model, status string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.status[model] = status
}

func (r *downloadProcessRouter) counts(model string) (posts, unloads, loads int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.posts[model], r.unloads[model], r.loads
}

func (r *downloadProcessRouter) waitStarted(t *testing.T, model string) {
	t.Helper()
	select {
	case got := <-r.started:
		if got != model {
			t.Fatalf("native POST model=%q, want %q", got, model)
		}
	case <-time.After(serviceCallTimeout):
		t.Fatalf("native POST did not reach first-shard barrier for %q", model)
	}
}

type downloadProcessModels struct {
	Models        []string                `json:"models"`
	ByEngine      map[string][]string     `json:"modelsByEngine"`
	Loaded        map[string][]string     `json:"loadedByEngine"`
	DownloadState *modeldownload.Snapshot `json:"downloadState"`
}

type downloadProcessPullFrame struct {
	Type          string                  `json:"type"`
	OpID          string                  `json:"opId"`
	Engine        string                  `json:"engine"`
	Op            string                  `json:"op"`
	Stage         string                  `json:"stage"`
	Percent       *int                    `json:"percent"`
	ID            string                  `json:"id"`
	Model         string                  `json:"model"`
	DownloadState *modeldownload.Snapshot `json:"downloadState"`
}

// Retain the actual ec response independently of target snapshots. Both the
// scanner and retained channel are bounded, and the request context owns reads.
func captureDownloadProcessPull(ctx context.Context, body io.Reader) (<-chan downloadProcessPullFrame, <-chan error) {
	const maxFrames = 64
	const maxBytes = 4 * modeldownload.MaxSnapshotBytes
	frames := make(chan downloadProcessPullFrame, maxFrames)
	done := make(chan error, 1)
	go func() {
		defer close(frames)
		defer close(done)
		done <- func() error {
			scanner := bufio.NewScanner(io.LimitReader(body, maxBytes+1))
			scanner.Buffer(make([]byte, 4096), modeldownload.MaxSnapshotBytes+4096)
			count, bytes := 0, 0
			for scanner.Scan() {
				count++
				bytes += len(scanner.Bytes()) + 1
				if count > maxFrames || bytes > maxBytes {
					return fmt.Errorf("queued pull capture exceeded frame/byte budget")
				}
				var frame downloadProcessPullFrame
				if err := json.Unmarshal(scanner.Bytes(), &frame); err != nil {
					return fmt.Errorf("decode queued pull frame: %w", err)
				}
				select {
				case frames <- frame:
				case <-ctx.Done():
					return ctx.Err()
				}
			}
			return scanner.Err()
		}()
	}()
	return frames, done
}

func nextDownloadProcessPullFrame(t *testing.T, frames <-chan downloadProcessPullFrame) downloadProcessPullFrame {
	t.Helper()
	select {
	case frame, ok := <-frames:
		if !ok {
			t.Fatal("queued pull stream ended before expected frame")
		}
		return frame
	case <-time.After(serviceCallTimeout):
		t.Fatal("queued pull stream did not deliver a frame")
		return downloadProcessPullFrame{}
	}
}

func assertDownloadProcessPullIdentity(t *testing.T, frame downloadProcessPullFrame, request, model, id, source string) modeldownload.Entry {
	t.Helper()
	if frame.OpID != request || frame.Engine != downloadProcessEngine || frame.Op != "pull" || frame.Model != model || frame.ID != id {
		t.Fatalf("queued stream crossed request/model/download identity: %+v", frame)
	}
	if frame.DownloadState == nil {
		t.Fatal("queued stream omitted target lifecycle")
	}
	if err := frame.DownloadState.Validate(); err != nil {
		t.Fatal(err)
	}
	if frame.DownloadState.SourceID != source {
		t.Fatal("queued stream changed target source")
	}
	entry := downloadProcessEntry(t, downloadProcessModels{DownloadState: frame.DownloadState}, model)
	if entry.ID != id {
		t.Fatalf("queued stream snapshot selected another download: %+v", entry)
	}
	return entry
}

func (f *downloadProcessFixture) queuedPull(t *testing.T, model, opID string) (context.CancelFunc, <-chan downloadProcessPullFrame, <-chan error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("https://127.0.0.1:%d/v1/models/pull", f.service.ports.engineControl), strings.NewReader(string(mustMarshal(t, map[string]string{"opId": opID, "engine": downloadProcessEngine, "model": model}))))
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	// This request has its own lifetime bound instead of the ordinary read
	// client's total timeout, because it deliberately waits behind another pull.
	client := *f.direct
	client.Timeout = 0
	response, err := client.Do(request)
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK || response.Header.Get("Content-Type") != "application/x-ndjson" {
		cancel()
		if err := response.Body.Close(); err != nil {
			t.Error(err)
		}
		t.Fatalf("queued pull HTTP=%d type=%q", response.StatusCode, response.Header.Get("Content-Type"))
	}
	frames, done := captureDownloadProcessPull(ctx, response.Body)
	t.Cleanup(func() {
		cancel()
		if err := response.Body.Close(); err != nil {
			t.Errorf("close queued response: %v", err)
		}
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Error("queued response capture did not stop")
		}
	})
	return cancel, frames, done
}

type downloadProcessFixture struct {
	service        *serviceFixture
	router         *downloadProcessRouter
	local, other   *serviceRPCClient
	controller     *task1Process
	controllerNode *task1Node
	targetNode     *task1Node
	observerNode   *task1Node
	observer       *http.Client
	direct         *http.Client
	scanner        *task1Process
	outer          *task1Observer
	serviceArgs    []string
	journal        string
	listeners      []int
	controlMu      sync.Mutex
	controlCalls   map[task1RequestKey]int
}

func newDownloadProcessFixture(t *testing.T, reverse bool) *downloadProcessFixture {
	t.Helper()
	// Preserve the wrapper's fixed-bin root, but never inherit another fixture's
	// child selectors, pairing faults or engine endpoint overrides.
	for _, item := range os.Environ() {
		key, _, _ := strings.Cut(item, "=")
		if strings.HasPrefix(strings.ToUpper(key), "NVPAIR_TEST_") && !strings.EqualFold(key, "NVPAIR_TEST_BIN_DIR") {
			t.Setenv(key, "")
		}
	}
	t.Setenv("OLLAMA_HOST", "")
	f := &downloadProcessFixture{router: newDownloadProcessRouter(t), controlCalls: make(map[task1RequestKey]int)}
	f.service = newServiceFixture(t)
	s := f.service
	// Cleanup ownership comes from the existing bundle's executable/PID/creation
	// inventory. Register closure checks before starting any owned process.
	t.Cleanup(func() {
		for _, port := range f.listeners {
			assertInteropListenerClosed(t, port)
		}
	})
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	// The registered bundle scanner is the existing empty-directory fixture.
	// A separate production scanner below runs --mdns=false, without claims.
	jointCopyExecutable(t, self, s.scannerPath)
	for _, binary := range []string{engineMgrBin, clusterMgrBin} {
		s.inventory.paths = append(s.inventory.paths, s.copyBinary(binary))
	}
	scannerPort := temporaryClusterPort(t)
	f.listeners = append(f.listeners, scannerPort, s.ports.engineControl, s.ports.engineManagerHTTP, s.ports.clusterManager)
	s.env = append(s.env, "NVPAIR_TEST_SUBSCRIPTION_SCANNER="+s.scannerPath, "NVPAIR_TEST_SUBSCRIPTION_PORT="+strconv.Itoa(scannerPort))
	inert := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusServiceUnavailable) }))
	t.Cleanup(inert.Close)
	jointIsolateEngines(t, s.configDir, portOfURL(t, inert.URL))
	writePrivateBrokerEngineSettings(t, s.configDir, s.clusterDir, map[string]int{"llamacpp": temporaryClusterPort(t), "ollama": temporaryClusterPort(t)})
	policy := nodepolicy.Default()
	policy.Revision = "12"
	jointWriteJSON(t, s.policyPath(), policy)
	f.targetNode = &task1Node{uuid: "download-target", base: s.dir, ports: map[noderec.ServiceKey]int{noderec.ServiceCluster: s.ports.clusterManager, noderec.ServiceEngineControl: s.ports.engineControl, noderec.ServiceEngineManager: s.ports.engineManagerHTTP}}
	f.controllerNode = &task1Node{uuid: "download-controller", base: shortClusterTempDir(t), ports: map[noderec.ServiceKey]int{noderec.ServiceCluster: temporaryClusterPort(t)}}
	f.observerNode = &task1Node{uuid: "download-observer", base: shortClusterTempDir(t), ports: map[noderec.ServiceKey]int{noderec.ServiceCluster: portOfURL(t, inert.URL)}}
	nodes := []*task1Node{f.targetNode, f.controllerNode, f.observerNode}
	for _, n := range nodes {
		dir := filepath.Join(n.base, "cluster")
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
		n.cert = mintClusterIdentity(t, dir, n.uuid)
		jointWriteJSON(t, filepath.Join(dir, "identity.json"), map[string]any{"node_uuid": n.uuid, "created_at": 1})
		jointWriteJSON(t, filepath.Join(dir, "admission.json"), map[string]any{"counter": 1, "activated": 1, "clusterId": task1ClusterID, "epoch": 1})
	}
	for _, n := range nodes {
		for _, peer := range nodes {
			if n != peer {
				jointWriteJSON(t, filepath.Join(n.base, "cluster", "trusted", peer.uuid+".json"), map[string]any{"nodeUuid": peer.uuid, "nodeId": peer.uuid, "certPem": peer.cert, "clusterId": task1ClusterID, "admissionEpoch": 1})
			}
		}
		n.mesh = clustertrust.Open(filepath.Join(n.base, "cluster"))
	}
	// Existing pinned reverse-session observation proxy; it never supplies a
	// lifecycle result. Only the reachable controller needs an outer listener.
	outer := newTask1Observer(t, f.controllerNode, noderec.ServiceCluster, nodes)
	f.outer = outer
	f.listeners = append(f.listeners, outer.port, f.controllerNode.ports[noderec.ServiceCluster])
	blockedPort := 0
	if reverse {
		blocked := listenFixtureLoopback(t)
		blockedPort = blocked.Addr().(*net.TCPAddr).Port
		f.listeners = append(f.listeners, blockedPort)
		done := make(chan struct{})
		go func() {
			defer close(done)
			for {
				conn, err := blocked.Accept()
				if err != nil {
					return
				}
				_ = conn.Close()
			}
		}()
		t.Cleanup(func() {
			if err := blocked.Close(); err != nil {
				t.Error(err)
			}
			<-done
		})
	}
	for _, n := range nodes {
		members := make([]task1Member, 0, len(nodes))
		for _, peer := range nodes {
			port := peer.ports[noderec.ServiceCluster]
			if peer == f.controllerNode {
				port = outer.port
			}
			if reverse && peer == f.targetNode {
				port = blockedPort
			}
			members = append(members, task1Member{ID: peer.uuid, NodeUUID: peer.uuid, Name: peer.uuid, IPAddress: "127.0.0.1", Port: port, ClusterID: task1ClusterID, AdmissionEpoch: 1, State: "member"})
		}
		jointWriteJSON(t, filepath.Join(n.base, "cluster", "members.json"), members)
	}
	managementGrantFixture(t, s.clusterDir, f.targetNode.uuid, f.controllerNode.uuid)
	f.controllerNode.cm = startTask1Process(t, f.controllerNode.base, clusterMgrBin, "--config-dir", f.controllerNode.base, "--port", strconv.Itoa(f.controllerNode.ports[noderec.ServiceCluster]))
	f.controllerNode.cm.call(t, "cluster:get-node-id", nil)
	f.controllerNode.cm.call(t, "cluster:set-local-services", map[string]any{"generation": 1, "services": []noderec.RegisterParams{{Service: noderec.ServiceCluster, Port: outer.port}}})
	f.writeManifest(t, s.configDir, portOfURL(t, f.router.server.URL), true)
	f.journal = filepath.Join(jointAppDir(s.configDir), "engine-bin", "model-download-state.json")
	manager := filepath.Join(s.binDir, filepath.Base(engineMgrBin))
	bridge := filepath.Join(testbin.Dir(t, "tests", "policy-joint", "joint-target-engine"), testbin.Exe("joint-target-engine"))
	jointCopyExecutable(t, self, bridge)
	if runtime.GOOS == "windows" {
		canonical, err := canonicalServicePath(bridge)
		if err != nil {
			t.Fatal(err)
		}
		s.inventory.paths = append(s.inventory.paths, canonical)
	}
	backendPort := s.ports.engineControl
	s.env = append(s.env, "NVPAIR_TEST_DOWNLOAD_MANAGER_BRIDGE="+bridge, "NVPAIR_TEST_DOWNLOAD_MANAGER_BINARY="+manager, "NVPAIR_TEST_DOWNLOAD_CONTROL_PORT="+strconv.Itoa(backendPort))
	s.ports.engineControl = f.observeControl(t, backendPort)
	f.serviceArgs = append(s.args(), "--engine-manager-control-host", "127.0.0.1", "--engine-manager-path", bridge)
	f.start(t)
	if reverse {
		task1Eventually(t, "pinned reverse session", 20*time.Second, func() bool { return outer.established(f.targetNode.uuid) > 0 })
	}
	base := f.controllerNode.base
	jointIsolateEngines(t, base, portOfURL(t, inert.URL))
	f.writeManifest(t, base, 0, false)
	f.controller = startTask1Process(t, base, engineMgrBin, "--cluster-dir", filepath.Join(base, "cluster"), "--control-host", "127.0.0.1", "--loaded-poll-interval", "0")
	completeEngineManagerRecovery(t, func(method string, params any) jsonrpc.Message { return f.controller.call(t, method, params) })
	ecPort, emPort := s.ports.engineControl, s.ports.engineManagerHTTP
	if reverse {
		ecPort, emPort = blockedPort, blockedPort
	}
	f.controller.send(t, noderec.NotifyNodes, noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{{HostUUID: f.targetNode.uuid, ClusterUUID: f.targetNode.uuid, IP: "127.0.0.1", Name: f.targetNode.uuid, Trusted: true, Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineControl: {Port: ecPort}, noderec.ServiceEngineManager: {Port: emPort}}}}}, false)
	f.direct = f.pinnedClient(t, f.controllerNode)
	f.observer = f.pinnedClient(t, f.observerNode)
	// A production target-local scanner enriches its registered em over its
	// existing loopback self path. Independent peer reads use the observer pin;
	// loopback peer descriptors are deliberately not forged for the scanner.
	f.scanner = startTask1Process(t, s.configDir, scannerBin, "--cluster-dir", s.clusterDir, "--mdns=false")
	f.scanner.call(t, noderec.MethodGetNodes, nil)
	t.Cleanup(func() {
		if err := filepath.WalkDir(base, func(path string, entry os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if strings.HasSuffix(entry.Name(), ".gguf") {
				return fmt.Errorf("controller stored model bytes at %s", path)
			}
			return nil
		}); err != nil {
			t.Error(err)
		}
	})
	return f
}

// Count authenticated HTTP entry before forwarding to the real control owner,
// including requests that the owner refuses before a native engine effect.
// Reverse traffic reaches this same endpoint through the broker's ec registry.
func (f *downloadProcessFixture) observeControl(t *testing.T, backendPort int) int {
	t.Helper()
	address, err := url.Parse(fmt.Sprintf("https://127.0.0.1:%d", backendPort))
	if err != nil {
		t.Fatal(err)
	}
	proxies := make(map[string]*httputil.ReverseProxy)
	for _, node := range []*task1Node{f.controllerNode, f.observerNode} {
		cfg, ok := node.mesh.ClientTLSConfig(f.targetNode.uuid)
		if !ok {
			t.Fatal("control observer lacks target pin")
		}
		transport := &http.Transport{TLSClientConfig: cfg, DisableKeepAlives: true}
		t.Cleanup(transport.CloseIdleConnections)
		proxy := httputil.NewSingleHostReverseProxy(address)
		proxy.Transport, proxy.FlushInterval = transport, -1
		proxies[node.uuid] = proxy
	}
	listener := listenFixtureLoopback(t)
	port := listener.Addr().(*net.TCPAddr).Port
	plan := f.service.ports
	plan.engineControl = port
	if err := plan.validate(); err != nil || port == backendPort {
		closeErr := listener.Close()
		t.Fatalf("control observer must have a distinct private port: frontend=%d manager=%d error=%v", port, backendPort, errors.Join(err, closeErr))
	}
	f.listeners = append(f.listeners, port)
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		f.targetNode.mesh.Refresh()
		peer, pinned := f.targetNode.mesh.VerifyClientPin(request)
		proxy := proxies[peer]
		if !pinned || proxy == nil {
			http.Error(w, "unexpected fixture control caller", http.StatusForbidden)
			return
		}
		f.controlMu.Lock()
		f.controlCalls[task1RequestKey{peer, request.Method, request.URL.Path}]++
		f.controlMu.Unlock()
		proxy.ServeHTTP(w, request)
	})}
	done := make(chan error, 1)
	go func() { done <- server.Serve(tls.NewListener(listener, f.targetNode.mesh.ServerTLSConfig())) }()
	t.Cleanup(func() {
		if err := server.Close(); err != nil {
			t.Error(err)
		}
		if err := <-done; err != http.ErrServerClosed {
			t.Errorf("control observer exit: %v", err)
		}
	})
	return port
}

func (f *downloadProcessFixture) controlCount(method, path string) int {
	f.controlMu.Lock()
	defer f.controlMu.Unlock()
	return f.controlCalls[task1RequestKey{f.controllerNode.uuid, method, path}]
}

func (f *downloadProcessFixture) writeManifest(t *testing.T, base string, port int, target bool) {
	t.Helper()
	probe := any(nil)
	if target {
		probe = map[string]any{"http": "http://127.0.0.1:{port}/ready", "status": http.StatusOK, "timeout_s": 1}
	}
	manifest := map[string]any{
		"engine": downloadProcessEngine, "display_name": "Synthetic catalog router", "manifest_version": 1, "family": "llamacpp", "model_naming": "exact",
		"runtime":   map[string]any{"mode": "process", "bin": filepath.Join(base, "never-start"), "port": port, "bind": "127.0.0.1", "ready": probe, "health": nil},
		"platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{"detect": []string{}}},
		"actions": map[string]any{
			"list_models":   map[string]any{"http": map[string]string{"method": http.MethodGet, "path": "/models"}, "result": map[string]string{"array": "data", "field": "id"}},
			"loaded_models": map[string]any{"http": map[string]string{"method": http.MethodGet, "path": "/models"}, "result": map[string]any{"array": "data", "field": "id", "match": map[string]any{"field": "status.value", "in": []string{"loaded"}}}},
			"pull_model":    map[string]any{"http": map[string]any{"method": http.MethodPost, "path": "/models", "body_schema": map[string]string{"model": "string"}}, "progress_protocol": "llamacpp-models-sse"},
			"load_model":    map[string]any{"http": map[string]any{"method": http.MethodPost, "path": "/models/load", "body_schema": map[string]string{"model": "string"}}},
		},
	}
	if target {
		manifest["models_dir"] = f.router.root
	}
	jointWriteJSON(t, filepath.Join(jointAppDir(base), "engines", downloadProcessEngine+".json"), manifest)
	if err := os.MkdirAll(filepath.Join(jointAppDir(base), "engine-bin"), 0o700); err != nil {
		t.Fatal(err)
	}
}

func (f *downloadProcessFixture) start(t *testing.T) {
	t.Helper()
	s := f.service
	s.service = s.startChild(f.serviceArgs...)
	f.local = s.readyClient()
	f.other = s.readyClient()
	for _, client := range []*serviceRPCClient{f.local, f.other} {
		client.mustCall(t, "401", "engine:subscribe", nil)
	}
	var status struct {
		Running bool `json:"running"`
		Port    int  `json:"port"`
	}
	task2Decode(t, f.local.mustCall(t, "402", "engine:status", mustMarshal(t, map[string]string{"engine": downloadProcessEngine})).Result, &status)
	if !status.Running || status.Port != portOfURL(t, f.router.server.URL) {
		t.Fatalf("synthetic router adoption=%+v", status)
	}
	for _, port := range []int{s.ports.engineControl, f.targetNode.ports[noderec.ServiceEngineControl], s.ports.engineManagerHTTP} {
		waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), serviceCallTimeout)
	}
}

func (f *downloadProcessFixture) pinnedClient(t *testing.T, node *task1Node) *http.Client {
	t.Helper()
	cfg, ok := node.mesh.ClientTLSConfig(f.targetNode.uuid)
	if !ok {
		t.Fatal("synthetic peer lacks target pin")
	}
	client := &http.Client{Transport: &http.Transport{TLSClientConfig: cfg}, Timeout: 15 * time.Second}
	t.Cleanup(client.CloseIdleConnections)
	return client
}

func downloadProcessDecode(t *testing.T, raw json.RawMessage) downloadProcessModels {
	t.Helper()
	var result downloadProcessModels
	task2Decode(t, raw, &result)
	if result.DownloadState == nil {
		t.Fatal("current producer omitted downloadState")
	}
	if err := result.DownloadState.Validate(); err != nil {
		t.Fatalf("invalid target snapshot: %v", err)
	}
	return result
}

func (f *downloadProcessFixture) models(t *testing.T, client *serviceRPCClient) downloadProcessModels {
	t.Helper()
	return downloadProcessDecode(t, client.mustCall(t, "410", "engine:models", nil).Result)
}

func (f *downloadProcessFixture) scannerModels(t *testing.T, expected modeldownload.Snapshot) downloadProcessModels {
	t.Helper()
	// Exercise registry-driven self enrichment without waiting for the periodic
	// refresh clock or introducing an alternate inventory producer.
	f.scanner.call(t, noderec.MethodUnregister, noderec.UnregisterParams{Service: noderec.ServiceEngineManager})
	f.scanner.call(t, noderec.MethodRegister, noderec.RegisterParams{Service: noderec.ServiceEngineManager, Port: f.service.ports.engineManagerHTTP})
	var result downloadProcessModels
	task1Eventually(t, "real scanner target-local convergence", serviceCallTimeout, func() bool {
		var nodes noderec.GetNodesResult
		task2Decode(t, f.scanner.call(t, noderec.MethodGetNodes, noderec.GetNodesParams{Service: noderec.ServiceEngineManager}).Result, &nodes)
		for _, node := range nodes.Nodes {
			if node.ClusterUUID == f.targetNode.uuid && node.DownloadStateCurrent != nil && *node.DownloadStateCurrent && node.DownloadState != nil && node.DownloadState.Equal(expected) {
				result = downloadProcessModels{Models: node.Models, ByEngine: node.ModelsByEngine, Loaded: node.LoadedByEngine, DownloadState: node.DownloadState}
				return true
			}
		}
		return false
	})
	return result
}

func (f *downloadProcessFixture) remoteModels(t *testing.T) downloadProcessModels {
	t.Helper()
	response, err := f.observer.Get(fmt.Sprintf("https://127.0.0.1:%d/v1/models", f.service.ports.engineManagerHTTP))
	if err != nil {
		t.Fatalf("independent pinned observer read: %v", err)
	}
	defer func() {
		if err := response.Body.Close(); err != nil {
			t.Errorf("close pinned inventory response: %v", err)
		}
	}()
	data, err := io.ReadAll(io.LimitReader(response.Body, modeldownload.MaxSnapshotBytes+65536))
	if err != nil || response.StatusCode != http.StatusOK {
		t.Fatalf("observer inventory HTTP=%d error=%v", response.StatusCode, err)
	}
	return downloadProcessDecode(t, data)
}

func downloadProcessEntry(t *testing.T, models downloadProcessModels, model string) modeldownload.Entry {
	t.Helper()
	for _, entry := range models.DownloadState.Entries {
		if entry.Engine == downloadProcessEngine && entry.Model == model {
			return entry
		}
	}
	t.Fatalf("target snapshot missing exact model %q: %+v", model, models.DownloadState)
	return modeldownload.Entry{}
}

func (f *downloadProcessFixture) waitPhase(t *testing.T, model, phase string) (downloadProcessModels, modeldownload.Entry) {
	t.Helper()
	var models downloadProcessModels
	var entry modeldownload.Entry
	task1Eventually(t, "target phase "+phase, serviceCallTimeout, func() bool {
		models = f.models(t, f.local)
		for _, candidate := range models.DownloadState.Entries {
			if candidate.Model == model && candidate.Engine == downloadProcessEngine {
				entry = candidate
				return entry.Phase == phase
			}
		}
		return false
	})
	return models, entry
}

func (f *downloadProcessFixture) pull(t *testing.T, model string) int {
	t.Helper()
	return f.controller.send(t, "engine:remote-pull-model", map[string]string{"node": f.targetNode.uuid, "engine": downloadProcessEngine, "model": model}, true)
}

func (f *downloadProcessFixture) reply(t *testing.T, id int, success bool) jsonrpc.Message {
	t.Helper()
	reply := task2Response(t, f.controller, id)
	if (reply.Error == nil) != success {
		t.Fatalf("pull response error=%+v, want success=%t", reply.Error, success)
	}
	return reply
}

func (f *downloadProcessFixture) progress(t *testing.T, client *serviceRPCClient, model, phase string) modeldownload.Snapshot {
	t.Helper()
	var decodeErr error
	message, err := client.next(serviceCallTimeout, func(message *jsonrpc.Message) bool {
		if message.Method != "engine:pull-progress" {
			return false
		}
		var event struct {
			Model         string                  `json:"model"`
			Engine        string                  `json:"engine"`
			Op            string                  `json:"op"`
			ID            string                  `json:"id"`
			DownloadState *modeldownload.Snapshot `json:"downloadState"`
		}
		if err := json.Unmarshal(message.Params, &event); err != nil {
			decodeErr = err
			return true // Fail only after next releases its client mutex.
		}
		if event.Model != model || event.DownloadState == nil {
			return false
		}
		if event.Engine != downloadProcessEngine || event.Op != "pull" {
			decodeErr = fmt.Errorf("relayed progress changed engine or operation")
			return true
		}
		for _, entry := range event.DownloadState.Entries {
			if entry.ID == event.ID && entry.Model == model && entry.Phase == phase {
				return phase != modeldownload.PhaseInProgress || entry.Percent != nil && *entry.Percent == 21
			}
		}
		return false
	})
	if decodeErr != nil {
		t.Fatalf("decode relayed progress: %v", decodeErr)
	}
	if err != nil {
		t.Fatalf("service relayed %s progress: %v", phase, err)
	}
	var event struct {
		DownloadState modeldownload.Snapshot `json:"downloadState"`
	}
	task2Decode(t, message.Params, &event)
	if err := event.DownloadState.Validate(); err != nil {
		t.Fatal(err)
	}
	return event.DownloadState
}

func (f *downloadProcessFixture) originProgress(t *testing.T, expected modeldownload.Snapshot, id string) {
	t.Helper()
	task1Eventually(t, "controller progress with exact target source/id/revision", serviceCallTimeout, func() bool {
		f.controller.mu.Lock()
		frames := slices.Clone(f.controller.frames)
		f.controller.mu.Unlock()
		found, requestID := false, ""
		for _, message := range frames {
			if message.Method != "engine:remote-progress" {
				continue
			}
			var event struct {
				Node          string                  `json:"node"`
				OpID          string                  `json:"opId"`
				ID            string                  `json:"id"`
				Model         string                  `json:"model"`
				Engine        string                  `json:"engine"`
				Op            string                  `json:"op"`
				DownloadState *modeldownload.Snapshot `json:"downloadState"`
			}
			task2Decode(t, message.Params, &event)
			if event.ID == id {
				if event.Node != f.targetNode.uuid || event.Model != downloadProcessModel || event.Engine != downloadProcessEngine || event.Op != "pull" || event.OpID == "" || event.OpID == id || requestID != "" && requestID != event.OpID {
					t.Fatalf("origin progress changed request or target identity: %+v", event)
				}
				requestID = event.OpID
				if event.DownloadState != nil && event.DownloadState.Equal(expected) {
					found = true
				}
			}
		}
		return found
	})
}

func (f *downloadProcessFixture) terminalConvergence(t *testing.T, models downloadProcessModels, entry modeldownload.Entry, clients ...*serviceRPCClient) {
	t.Helper()
	for _, client := range clients {
		state := f.progress(t, client, entry.Model, entry.Phase)
		if !state.Equal(*models.DownloadState) {
			t.Fatalf("terminal relay changed target source/id/revision: got=%+v want=%+v", state, models.DownloadState)
		}
	}
	f.originProgress(t, *models.DownloadState, entry.ID)
}

func (f *downloadProcessFixture) authenticatedControlRead(t *testing.T) {
	t.Helper()
	before := f.controlCount(http.MethodGet, "/v1/engines")
	reply := f.controller.call(t, "engine:remote-get-installed", map[string]string{"node": f.targetNode.uuid})
	var inventory struct {
		Engines []struct {
			Engine  string `json:"engine"`
			Running bool   `json:"running"`
			Port    int    `json:"port"`
		} `json:"engines"`
	}
	task2Decode(t, reply.Result, &inventory)
	found := false
	for _, engine := range inventory.Engines {
		if engine.Engine == downloadProcessEngine && engine.Running && engine.Port == portOfURL(t, f.router.server.URL) {
			found = true
		}
	}
	if !found || f.controlCount(http.MethodGet, "/v1/engines") != before+1 {
		t.Fatalf("replacement route did not deliver an authenticated target read: %+v", inventory)
	}
}

func (f *downloadProcessFixture) journalModels(t *testing.T) []string {
	t.Helper()
	data, err := os.ReadFile(f.journal)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		t.Fatalf("read private download journal: %v", err)
	}
	if len(data) > modeldownload.MaxJournalBytes {
		t.Fatal("journal exceeded fixed byte bound")
	}
	var journal struct {
		Version int `json:"version"`
		Entries []struct {
			Model string `json:"model"`
		} `json:"entries"`
	}
	task2Decode(t, data, &journal)
	if journal.Version != 1 || journal.Entries == nil || len(journal.Entries) > modeldownload.MaxUnsafeEntries {
		t.Fatalf("invalid private journal shape: %+v", journal)
	}
	models := make([]string, 0, len(journal.Entries))
	for _, entry := range journal.Entries {
		models = append(models, entry.Model)
	}
	return models
}

func (f *downloadProcessFixture) assertFences(t *testing.T, model string) {
	t.Helper()
	beforePosts, beforeUnloads, beforeLoads := f.router.counts(model)
	for _, call := range []struct {
		method string
		params any
	}{
		{"engine:load-model", map[string]string{"engine": downloadProcessEngine, "model": model}},
		{"engine:action", map[string]any{"engine": downloadProcessEngine, "action": "load_model", "params": map[string]string{"model": model}}},
	} {
		response, err := f.local.call("420", call.method, mustMarshal(t, call.params), serviceCallTimeout)
		if err != nil || response.Error == nil {
			t.Fatalf("unsafe %s refusal: response=%+v error=%v", call.method, response, err)
		}
		if !strings.Contains(response.Error.Message, "model content is busy") {
			t.Fatalf("%s refused at unrelated boundary: %+v", call.method, response.Error)
		}
	}
	// Existing source-read routes only; no Task 6 transfer/receipt assertions.
	for _, path := range []string{"/v1/models/files", "/v1/models/file"} {
		query := url.Values{"engine": {downloadProcessEngine}, "model": {model}, "path": {"first.gguf"}}
		response, err := f.direct.Get(fmt.Sprintf("https://127.0.0.1:%d%s?%s", f.service.ports.engineControl, path, query.Encode()))
		if err != nil {
			t.Fatal(err)
		}
		data, readErr := io.ReadAll(io.LimitReader(response.Body, 4096))
		closeErr := response.Body.Close()
		if readErr != nil || closeErr != nil || response.StatusCode != http.StatusConflict || strings.TrimSpace(string(data)) != "model content unavailable" {
			t.Fatalf("unsafe source read %s HTTP=%d read=%v close=%v body=%q", path, response.StatusCode, readErr, closeErr, data)
		}
	}
	posts, unloads, loads := f.router.counts(model)
	if posts != beforePosts || unloads != beforeUnloads || loads != beforeLoads || loads != 0 {
		t.Fatalf("blocked calls caused native effects: posts=%d/%d unloads=%d/%d loads=%d/%d", posts, beforePosts, unloads, beforeUnloads, loads, beforeLoads)
	}
}

func (f *downloadProcessFixture) restart(t *testing.T) {
	t.Helper()
	// Graceful owner restart preserves journals but cannot preserve an active
	// request. It also exercises the service's complete broker-descendant reap.
	f.local.mustCall(t, "430", "service/stop", nil)
	f.service.waitChild(f.service.service, serviceStopTimeout)
	f.service.cleanup()
	f.service.assertStopped()
	f.start(t)
}

// Alter only an authentic target terminal on its way to the controller. This
// reader has no producer lifecycle and no background goroutine to outlive its
// response. The real target can complete even when its caller rejects the wire.
type downloadProcessWrongOperation struct {
	io.ReadCloser
	reader  *bufio.Reader
	pending []byte
}

func (r *downloadProcessWrongOperation) Read(data []byte) (int, error) {
	if len(r.pending) == 0 {
		line, err := r.reader.ReadBytes('\n')
		if err != nil {
			return 0, err
		}
		var frame map[string]json.RawMessage
		if err := json.Unmarshal(line, &frame); err != nil {
			return 0, err
		}
		var kind string
		if err := json.Unmarshal(frame["type"], &kind); err != nil {
			return 0, err
		}
		if kind == "result" {
			frame["opId"] = json.RawMessage(`"wrong-operation"`)
			line, err = json.Marshal(frame)
			if err != nil {
				return 0, err
			}
			line = append(line, '\n')
		}
		r.pending = line
	}
	count := copy(data, r.pending)
	r.pending = r.pending[count:]
	return count, nil
}

func (f *downloadProcessFixture) wrongOperationProxy(t *testing.T) {
	t.Helper()
	address, err := url.Parse(fmt.Sprintf("https://127.0.0.1:%d", f.service.ports.engineControl))
	if err != nil {
		t.Fatal(err)
	}
	cfg, ok := f.controllerNode.mesh.ClientTLSConfig(f.targetNode.uuid)
	if !ok {
		t.Fatal("proxy lacks real target pin")
	}
	transport := &http.Transport{TLSClientConfig: cfg}
	t.Cleanup(transport.CloseIdleConnections)
	proxy := httputil.NewSingleHostReverseProxy(address)
	proxy.Transport = transport
	proxy.FlushInterval = -1
	proxy.ModifyResponse = func(response *http.Response) error {
		if response.Request.URL.Path == "/v1/models/pull" && response.StatusCode == http.StatusOK {
			response.Body = &downloadProcessWrongOperation{ReadCloser: response.Body, reader: bufio.NewReader(response.Body)}
		}
		return nil
	}
	listener := listenFixtureLoopback(t)
	port := listener.Addr().(*net.TCPAddr).Port
	f.listeners = append(f.listeners, port)
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		f.targetNode.mesh.Refresh()
		if peer, ok := f.targetNode.mesh.VerifyClientPin(request); !ok || peer != f.controllerNode.uuid {
			http.Error(w, "unexpected fixture peer", http.StatusForbidden)
			return
		}
		proxy.ServeHTTP(w, request)
	})}
	done := make(chan error, 1)
	go func() { done <- server.Serve(tls.NewListener(listener, f.targetNode.mesh.ServerTLSConfig())) }()
	t.Cleanup(func() {
		if err := server.Close(); err != nil {
			t.Error(err)
		}
		if err := <-done; err != http.ErrServerClosed {
			t.Errorf("terminal proxy exit: %v", err)
		}
	})
	f.controller.send(t, noderec.NotifyNodes, noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{{HostUUID: f.targetNode.uuid, ClusterUUID: f.targetNode.uuid, IP: "127.0.0.1", Name: f.targetNode.uuid, Trusted: true, Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineControl: {Port: port}}}}}, false)
}

func TestModelDownloadStateProcess(t *testing.T) {
	t.Run("direct observers hydrate target state and separate completion inventory residency", func(t *testing.T) {
		f := newDownloadProcessFixture(t, false)
		id := f.pull(t, downloadProcessModel)
		f.router.waitStarted(t, downloadProcessModel)
		f.router.event(t, downloadProcessModel, "download_progress")
		var models downloadProcessModels
		var entry modeldownload.Entry
		task1Eventually(t, "target measured progress", serviceCallTimeout, func() bool {
			models, entry = f.waitPhase(t, downloadProcessModel, modeldownload.PhaseInProgress)
			return entry.Percent != nil && *entry.Percent == 21
		})
		shard, err := os.Stat(filepath.Join(f.router.root, downloadProcessShardName(downloadProcessModel)))
		if err != nil || shard.Size() != int64(len("GGUFfixture")) {
			t.Fatalf("first shard barrier: info=%v error=%v", shard, err)
		}
		if !entry.BlocksReadiness || slices.Contains(models.ByEngine[downloadProcessEngine], downloadProcessModel) {
			t.Fatalf("first shard published ready: %+v", models)
		}
		for _, client := range []*serviceRPCClient{f.local, f.other} {
			state := f.progress(t, client, downloadProcessModel, modeldownload.PhaseInProgress)
			if !state.Equal(*models.DownloadState) {
				t.Fatal("service relay changed target source/revision/snapshot")
			}
			if got := downloadProcessEntry(t, downloadProcessModels{DownloadState: &state}, downloadProcessModel); got.ID != entry.ID {
				t.Fatal("service relay changed target operation identity")
			}
		}
		f.originProgress(t, *models.DownloadState, entry.ID)
		fresh := f.service.readyClient()
		fresh.mustCall(t, "403", "engine:subscribe", nil)
		freshModels := f.models(t, fresh)
		remote := f.remoteModels(t)
		for _, observed := range []downloadProcessModels{freshModels, remote, f.scannerModels(t, *models.DownloadState)} {
			got := downloadProcessEntry(t, observed, downloadProcessModel)
			if got.ID != entry.ID || !observed.DownloadState.Equal(*models.DownloadState) || got.Phase != modeldownload.PhaseInProgress || slices.Contains(observed.ByEngine[downloadProcessEngine], downloadProcessModel) {
				t.Fatalf("observer did not converge without initiation map: %+v", observed)
			}
		}
		f.assertFences(t, downloadProcessModel)
		f.router.setStatus(downloadProcessModel, "downloaded")
		f.router.event(t, downloadProcessModel, "download_finished")
		reply := f.reply(t, id, true)
		var result struct {
			ID            string                 `json:"id"`
			Model         string                 `json:"model"`
			DownloadState modeldownload.Snapshot `json:"downloadState"`
		}
		task2Decode(t, reply.Result, &result)
		if result.ID != entry.ID || result.Model != downloadProcessModel || result.DownloadState.SourceID != models.DownloadState.SourceID {
			t.Fatalf("origin terminal lost target identity: %+v", result)
		}
		complete, outcome := f.waitPhase(t, downloadProcessModel, modeldownload.PhaseComplete)
		f.terminalConvergence(t, complete, outcome, f.local, f.other, fresh)
		if outcome.BlocksReadiness || slices.Contains(complete.ByEngine[downloadProcessEngine], downloadProcessModel) || slices.Contains(complete.Loaded[downloadProcessEngine], downloadProcessModel) {
			t.Fatalf("terminal/transient inventory manufactured readiness or residency: %+v", complete)
		}
		f.router.setStatus(downloadProcessModel, "unloaded")
		ready := f.models(t, fresh)
		if !slices.Contains(ready.ByEngine[downloadProcessEngine], downloadProcessModel) || slices.Contains(ready.Loaded[downloadProcessEngine], downloadProcessModel) {
			t.Fatalf("stable inventory did not publish independently of loaded: %+v", ready)
		}
		f.router.setStatus(downloadProcessModel, "loaded")
		loaded := f.remoteModels(t)
		if !slices.Contains(loaded.Loaded[downloadProcessEngine], downloadProcessModel) {
			t.Fatal("independent real loaded observation was lost")
		}
		if len(f.journalModels(t)) != 0 {
			t.Fatal("successful pull retained unsafe marker")
		}
		if posts, _, loads := f.router.counts(downloadProcessModel); posts != 1 || loads != 0 {
			t.Fatalf("native posts=%d loads=%d, want 1 and 0", posts, loads)
		}
	})

	t.Run("queued second stream keeps exact identities through first terminal", func(t *testing.T) {
		f := newDownloadProcessFixture(t, false)
		firstID := f.pull(t, downloadProcessModel)
		f.router.waitStarted(t, downloadProcessModel)
		const second, requestID = "fixture/queued:Q4_K_M", "queued-stream-request"
		_, frames, done := f.queuedPull(t, second, requestID)
		models, queued := f.waitPhase(t, second, modeldownload.PhaseQueued)
		initial := nextDownloadProcessPullFrame(t, frames)
		entry := assertDownloadProcessPullIdentity(t, initial, requestID, second, queued.ID, models.DownloadState.SourceID)
		if initial.Type != "progress" || initial.Stage != "queued" || initial.Percent != nil || entry.Phase != modeldownload.PhaseQueued || entry.Percent != nil {
			t.Fatalf("second response did not begin with its own indeterminate queue: %+v", initial)
		}
		f.router.event(t, downloadProcessModel, "download_progress")
		task1Eventually(t, "first measured progress while second stream is open", serviceCallTimeout, func() bool {
			_, e := f.waitPhase(t, downloadProcessModel, modeldownload.PhaseInProgress)
			return e.Percent != nil && *e.Percent == 21
		})
		f.router.setStatus(downloadProcessModel, "unloaded")
		f.router.event(t, downloadProcessModel, "download_finished")
		f.reply(t, firstID, true)
		f.router.waitStarted(t, second)
		both, _ := f.waitPhase(t, second, modeldownload.PhaseInProgress)
		if first := downloadProcessEntry(t, both, downloadProcessModel); first.Phase != modeldownload.PhaseComplete {
			t.Fatalf("first pull did not terminate while second stream stayed open: %+v", first)
		}
		// This is the response's own post-queue barrier. No native progress or
		// terminal has been sent for the second model, even after first success.
		captured := []downloadProcessPullFrame{initial}
		for {
			frame := nextDownloadProcessPullFrame(t, frames)
			captured = append(captured, frame)
			entry := assertDownloadProcessPullIdentity(t, frame, requestID, second, queued.ID, models.DownloadState.SourceID)
			if frame.Type != "progress" || frame.Percent != nil || entry.Percent != nil || (entry.Phase != modeldownload.PhaseQueued && entry.Phase != modeldownload.PhaseInProgress) {
				t.Fatalf("first percent/terminal contaminated the second response before its own terminal: %+v", frame)
			}
			if entry.Phase == modeldownload.PhaseQueued && frame.Stage != "queued" || entry.Phase == modeldownload.PhaseInProgress && frame.Stage != "pulling" {
				t.Fatalf("second stream received another operation's stage: %+v", frame)
			}
			if entry.Phase == modeldownload.PhaseInProgress {
				break
			}
		}
		f.router.setStatus(second, "unloaded")
		f.router.event(t, second, "download_finished")
		finished := false
		timer := time.NewTimer(serviceCallTimeout)
		defer timer.Stop()
		for !finished {
			select {
			case frame, ok := <-frames:
				if !ok {
					finished = true
					break
				}
				captured = append(captured, frame)
			case <-timer.C:
				t.Fatal("second response did not finish after its own native terminal")
			}
		}
		if err := <-done; err != nil {
			t.Fatalf("bounded queued response capture: %v", err)
		}
		var previous *modeldownload.Snapshot
		results := 0
		for _, frame := range captured {
			entry := assertDownloadProcessPullIdentity(t, frame, requestID, second, queued.ID, models.DownloadState.SourceID)
			if previous != nil {
				before, err := strconv.ParseUint(previous.Revision, 10, 64)
				if err != nil {
					t.Fatal(err)
				}
				after, err := strconv.ParseUint(frame.DownloadState.Revision, 10, 64)
				if err != nil {
					t.Fatal(err)
				}
				if after < before || after == before && !previous.Equal(*frame.DownloadState) {
					t.Fatal("second response regressed or changed an established lifecycle revision")
				}
			}
			previous = frame.DownloadState
			if results != 0 {
				t.Fatal("second response emitted a frame after its terminal result")
			}
			if frame.Type != "progress" && frame.Type != "result" {
				t.Fatalf("unexpected second response frame: %+v", frame)
			}
			if entry.Percent != nil || frame.Percent != nil && (entry.Phase != modeldownload.PhaseComplete || *frame.Percent != 100) {
				t.Fatalf("first measured percent leaked into second response: %+v", frame)
			}
			if entry.Phase == modeldownload.PhaseQueued && frame.Stage != "queued" || entry.Phase == modeldownload.PhaseInProgress && frame.Stage != "pulling" || entry.Phase == modeldownload.PhaseComplete && frame.Stage != "success" {
				t.Fatalf("second stream stage disagrees with its own lifecycle: %+v", frame)
			}
			if frame.Type == "result" {
				results++
				if entry.Phase != modeldownload.PhaseComplete || entry.BlocksReadiness {
					t.Fatalf("second response result was not its own successful terminal: %+v", frame)
				}
			}
		}
		if results != 1 {
			t.Fatalf("second response terminal results=%d, want 1", results)
		}
		if posts, unloads, _ := f.router.counts(second); posts != 1 || unloads != 0 {
			t.Fatalf("second stream native effects: posts=%d unloads=%d", posts, unloads)
		}
	})

	t.Run("queued cancellation has no native effects or unsafe marker", func(t *testing.T) {
		f := newDownloadProcessFixture(t, false)
		id := f.pull(t, downloadProcessModel)
		f.router.waitStarted(t, downloadProcessModel)
		const second = "fixture/queued:Q4_K_M"
		cancel, frames, done := f.queuedPull(t, second, "queue-cancel-request")
		models, queued := f.waitPhase(t, second, modeldownload.PhaseQueued)
		initial := nextDownloadProcessPullFrame(t, frames)
		entry := assertDownloadProcessPullIdentity(t, initial, "queue-cancel-request", second, queued.ID, models.DownloadState.SourceID)
		if initial.Type != "progress" || initial.Percent != nil || entry.Phase != modeldownload.PhaseQueued || entry.Percent != nil {
			t.Fatalf("cancelled request did not have its own queue frame: %+v", initial)
		}
		f.router.event(t, downloadProcessModel, "download_progress")
		task1Eventually(t, "first measured progress", serviceCallTimeout, func() bool {
			_, e := f.waitPhase(t, downloadProcessModel, modeldownload.PhaseInProgress)
			return e.Percent != nil && *e.Percent == 21
		})
		stillQueued := downloadProcessEntry(t, f.models(t, f.local), second)
		if stillQueued.ID != queued.ID || stillQueued.Phase != modeldownload.PhaseQueued || stillQueued.Percent != nil {
			t.Fatalf("first progress leaked into queued model: %+v", stillQueued)
		}
		cancel()
		select {
		case err := <-done:
			if err != nil && !errors.Is(err, context.Canceled) {
				t.Fatalf("cancel queued capture: %v", err)
			}
		case <-time.After(serviceCallTimeout):
			t.Fatal("cancelled queue response capture did not stop")
		}
		_, cancelled := f.waitPhase(t, second, modeldownload.PhaseFailed)
		if cancelled.BlocksReadiness || cancelled.Reason != modeldownload.ReasonInterrupted || slices.Contains(f.journalModels(t), second) {
			t.Fatalf("cancelled queue created unsafe content: %+v", cancelled)
		}
		if posts, unloads, _ := f.router.counts(second); posts != 0 || unloads != 0 {
			t.Fatalf("queued cancellation effected target: posts=%d unloads=%d", posts, unloads)
		}
		f.router.setStatus(downloadProcessModel, "unloaded")
		f.router.event(t, downloadProcessModel, "download_finished")
		f.reply(t, id, true)
		if got := downloadProcessEntry(t, f.models(t, f.local), second); !reflect.DeepEqual(got, cancelled) {
			t.Fatalf("first terminal changed second model: %+v -> %+v", cancelled, got)
		}
	})

	for _, scenario := range []struct {
		name, mode, phase, reason string
		disconnect                bool
		unloads                   int
	}{
		{name: "download failed", phase: modeldownload.PhaseFailed, reason: modeldownload.ReasonDownloadFailed},
		{name: "missing terminal", phase: modeldownload.PhaseFailed, reason: modeldownload.ReasonInterrupted, disconnect: true, unloads: 1},
		{name: "cleanup failure", mode: "cleanup failure", phase: modeldownload.PhaseUnknown, reason: modeldownload.ReasonCleanupUnknown, disconnect: true, unloads: 1},
		{name: "uncertain acceptance", mode: "uncertain acceptance", phase: modeldownload.PhaseUnknown, reason: modeldownload.ReasonAcceptanceUnknown},
	} {
		t.Run(scenario.name+" remains unsafe across observers and restart", func(t *testing.T) {
			f := newDownloadProcessFixture(t, false)
			fresh := f.service.readyClient()
			fresh.mustCall(t, "403", "engine:subscribe", nil)
			f.router.mu.Lock()
			f.router.mode[downloadProcessModel] = scenario.mode
			f.router.mu.Unlock()
			id := f.pull(t, downloadProcessModel)
			f.router.waitStarted(t, downloadProcessModel)
			if scenario.disconnect {
				f.router.disconnect(t)
			} else if scenario.mode != "uncertain acceptance" {
				f.router.setStatus(downloadProcessModel, "downloaded")
				f.router.event(t, downloadProcessModel, "download_failed")
			}
			f.reply(t, id, false)
			before, entry := f.waitPhase(t, downloadProcessModel, scenario.phase)
			f.terminalConvergence(t, before, entry, f.local, f.other, fresh)
			if !entry.BlocksReadiness || entry.Reason != scenario.reason {
				t.Fatalf("failure outcome=%+v, want %s", entry, scenario.reason)
			}
			// Even a stable-looking raw cache entry cannot erase PAIR's marker.
			f.router.setStatus(downloadProcessModel, "unloaded")
			for _, models := range []downloadProcessModels{f.models(t, f.service.readyClient()), f.remoteModels(t), f.scannerModels(t, *before.DownloadState)} {
				if got := downloadProcessEntry(t, models, downloadProcessModel); !reflect.DeepEqual(got, entry) || slices.Contains(models.ByEngine[downloadProcessEngine], downloadProcessModel) {
					t.Fatalf("fresh observer accepted partial inventory: %+v", models)
				}
			}
			f.assertFences(t, downloadProcessModel)
			if !slices.Contains(f.journalModels(t), downloadProcessModel) {
				t.Fatal("unsafe outcome missing durable marker")
			}
			f.restart(t)
			after, recovered := f.waitPhase(t, downloadProcessModel, modeldownload.PhaseUnknown)
			if after.DownloadState.SourceID == before.DownloadState.SourceID || recovered.ID != entry.ID || recovered.Reason != modeldownload.ReasonInterrupted || !recovered.BlocksReadiness || slices.Contains(after.ByEngine[downloadProcessEngine], downloadProcessModel) {
				t.Fatalf("restart recovery lost generation/marker: before=%+v after=%+v", before, after)
			}
			for _, observed := range []downloadProcessModels{f.remoteModels(t), f.scannerModels(t, *after.DownloadState)} {
				if !observed.DownloadState.Equal(*after.DownloadState) || slices.Contains(observed.ByEngine[downloadProcessEngine], downloadProcessModel) {
					t.Fatalf("observer retained retired source/readiness after restart: %+v", observed)
				}
			}
			f.assertFences(t, downloadProcessModel)
			if posts, unloads, loads := f.router.counts(downloadProcessModel); posts != 1 || unloads != scenario.unloads || loads != 0 {
				t.Fatalf("restart replay/effects: posts=%d unloads=%d loads=%d", posts, unloads, loads)
			}
		})
	}

	t.Run("direct controller disconnect does not replay a dispatched POST", func(t *testing.T) {
		f := newDownloadProcessFixture(t, false)
		f.pull(t, downloadProcessModel)
		f.router.waitStarted(t, downloadProcessModel)
		_, active := f.waitPhase(t, downloadProcessModel, modeldownload.PhaseInProgress)
		if count := f.controlCount(http.MethodPost, "/v1/models/pull"); count != 1 {
			t.Fatalf("direct dispatched control POSTs=%d, want 1", count)
		}
		// EOF cancels the real controller manager and its pinned HTTP request.
		f.controller.stop()
		_, failed := f.waitPhase(t, downloadProcessModel, modeldownload.PhaseFailed)
		if failed.ID != active.ID || failed.Reason != modeldownload.ReasonInterrupted || !failed.BlocksReadiness {
			t.Fatalf("direct disconnect outcome=%+v", failed)
		}
		f.restart(t)
		f.waitPhase(t, downloadProcessModel, modeldownload.PhaseUnknown)
		if count := f.controlCount(http.MethodPost, "/v1/models/pull"); count != 1 {
			t.Fatalf("direct disconnect replayed control POSTs=%d", count)
		}
		if posts, unloads, loads := f.router.counts(downloadProcessModel); posts != 1 || unloads != 1 || loads != 0 {
			t.Fatalf("direct disconnect replay/effects: posts=%d unloads=%d loads=%d", posts, unloads, loads)
		}
	})

	t.Run("reverse pull disconnect does not replay a dispatched POST", func(t *testing.T) {
		f := newDownloadProcessFixture(t, true)
		id := f.pull(t, downloadProcessModel)
		f.router.waitStarted(t, downloadProcessModel)
		f.router.event(t, downloadProcessModel, "download_progress")
		_, entry := f.waitPhase(t, downloadProcessModel, modeldownload.PhaseInProgress)
		if entry.ID == "" {
			t.Fatal("reverse dispatched without target identity")
		}
		controlPosts := f.controlCount(http.MethodPost, "/v1/models/pull")
		if controlPosts != 1 {
			t.Fatalf("reverse dispatched control POSTs=%d, want 1", controlPosts)
		}
		established := f.outer.established(f.targetNode.uuid)
		// Sever the actual upgraded session after dispatch. The product must not
		// turn this ambiguous POST into another POST on a replacement route.
		f.outer.interrupt(f.targetNode.uuid, true)
		f.reply(t, id, false)
		_, failed := f.waitPhase(t, downloadProcessModel, modeldownload.PhaseFailed)
		if failed.ID != entry.ID || failed.Reason != modeldownload.ReasonInterrupted {
			t.Fatalf("reverse disconnect outcome=%+v", failed)
		}
		f.outer.interrupt(f.targetNode.uuid, false)
		task1Eventually(t, "replacement authenticated reverse session", 25*time.Second, func() bool { return f.outer.established(f.targetNode.uuid) > established })
		f.authenticatedControlRead(t)
		if count := f.controlCount(http.MethodPost, "/v1/models/pull"); count != controlPosts {
			t.Fatalf("replacement route replayed control POSTs: %d -> %d", controlPosts, count)
		}
		established = f.outer.established(f.targetNode.uuid)
		f.restart(t)
		f.waitPhase(t, downloadProcessModel, modeldownload.PhaseUnknown)
		task1Eventually(t, "authenticated reverse session after target restart", 25*time.Second, func() bool { return f.outer.established(f.targetNode.uuid) > established })
		f.authenticatedControlRead(t)
		if count := f.controlCount(http.MethodPost, "/v1/models/pull"); count != controlPosts {
			t.Fatalf("restart replayed control POSTs: %d -> %d", controlPosts, count)
		}
		if posts, unloads, loads := f.router.counts(downloadProcessModel); posts != 1 || unloads != 1 || loads != 0 {
			t.Fatalf("reverse POST replay/effects: posts=%d unloads=%d loads=%d", posts, unloads, loads)
		}
	})

	t.Run("wrong operation terminal is refused without replaying target success", func(t *testing.T) {
		f := newDownloadProcessFixture(t, false)
		f.wrongOperationProxy(t)
		id := f.pull(t, downloadProcessModel)
		f.router.waitStarted(t, downloadProcessModel)
		f.router.setStatus(downloadProcessModel, "downloaded")
		f.router.event(t, downloadProcessModel, "download_finished")
		reply := f.reply(t, id, false)
		if !strings.Contains(reply.Error.Message, "mismatched request identity") {
			t.Fatalf("wrong-op refusal=%+v", reply.Error)
		}
		_, terminal := f.waitPhase(t, downloadProcessModel, modeldownload.PhaseComplete)
		fresh := f.remoteModels(t)
		if got := downloadProcessEntry(t, fresh, downloadProcessModel); got.ID != terminal.ID || got.Phase != modeldownload.PhaseComplete || slices.Contains(fresh.ByEngine[downloadProcessEngine], downloadProcessModel) {
			t.Fatalf("wire fault overwrote authentic target/inventory state: %+v", fresh)
		}
		if posts, unloads, loads := f.router.counts(downloadProcessModel); posts != 1 || unloads != 0 || loads != 0 {
			t.Fatalf("wrong-op replay/effects: posts=%d unloads=%d loads=%d", posts, unloads, loads)
		}
	})

	t.Run("successful marker removal survives restart without replay", func(t *testing.T) {
		f := newDownloadProcessFixture(t, false)
		id := f.pull(t, downloadProcessModel)
		f.router.waitStarted(t, downloadProcessModel)
		f.router.setStatus(downloadProcessModel, "unloaded")
		f.router.event(t, downloadProcessModel, "download_finished")
		f.reply(t, id, true)
		before, _ := f.waitPhase(t, downloadProcessModel, modeldownload.PhaseComplete)
		if len(f.journalModels(t)) != 0 {
			t.Fatal("marker was not removed before restart barrier")
		}
		f.restart(t)
		after := f.models(t, f.local)
		if after.DownloadState.SourceID == before.DownloadState.SourceID || len(after.DownloadState.Entries) != 0 || !slices.Contains(after.ByEngine[downloadProcessEngine], downloadProcessModel) || slices.Contains(after.Loaded[downloadProcessEngine], downloadProcessModel) {
			t.Fatalf("resolved restart invented terminal history/readiness/residency: %+v", after)
		}
		if posts, unloads, loads := f.router.counts(downloadProcessModel); posts != 1 || unloads != 0 || loads != 0 {
			t.Fatalf("resolved restart effects: posts=%d unloads=%d loads=%d", posts, unloads, loads)
		}
	})

	t.Run("native success with failed marker removal remains unknown and recovery fenced", func(t *testing.T) {
		f := newDownloadProcessFixture(t, false)
		id := f.pull(t, downloadProcessModel)
		f.router.waitStarted(t, downloadProcessModel)
		before, active := f.waitPhase(t, downloadProcessModel, modeldownload.PhaseInProgress)
		if !slices.Contains(f.journalModels(t), downloadProcessModel) {
			t.Fatal("missing unsafe journal at removal-failure barrier")
		}
		// Only this disposable profile's journal is obstructed. Keep its exact
		// unsafe bytes beside it; never forge successful persistence evidence.
		if err := os.Rename(f.journal, f.journal+".retained"); err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(f.journal, 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(f.journal, "obstruction"), []byte("fixture"), 0o600); err != nil {
			t.Fatal(err)
		}
		f.router.setStatus(downloadProcessModel, "unloaded")
		f.router.event(t, downloadProcessModel, "download_finished")
		f.reply(t, id, false)
		_, uncertain := f.waitPhase(t, downloadProcessModel, modeldownload.PhaseUnknown)
		if uncertain.ID != active.ID || uncertain.Reason != modeldownload.ReasonPersistenceError || !uncertain.BlocksReadiness {
			t.Fatalf("failed marker removal outcome=%+v", uncertain)
		}
		f.assertFences(t, downloadProcessModel)
		f.restart(t)
		after := f.models(t, f.local)
		if after.DownloadState.SourceID == before.DownloadState.SourceID || !after.DownloadState.RecoveryRequired || slices.Contains(after.ByEngine[downloadProcessEngine], downloadProcessModel) {
			t.Fatalf("unreadable recovery manufactured ready content: %+v", after)
		}
		f.assertFences(t, downloadProcessModel)
		if posts, unloads, loads := f.router.counts(downloadProcessModel); posts != 1 || unloads != 0 || loads != 0 {
			t.Fatalf("persistence-failure recovery effects: posts=%d unloads=%d loads=%d", posts, unloads, loads)
		}
	})

	t.Run("restart before native effect leaves no unsafe marker or replay", func(t *testing.T) {
		f := newDownloadProcessFixture(t, false)
		hold := make(chan struct{})
		subscribed := make(chan struct{}, 1)
		f.router.mu.Lock()
		f.router.holdSubscribe = hold
		f.router.subscribed = subscribed
		f.router.mu.Unlock()
		id := f.pull(t, downloadProcessModel)
		select {
		case <-subscribed:
		case <-time.After(serviceCallTimeout):
			t.Fatal("pull never reached pre-effect SSE header barrier")
		}
		before, queued := f.waitPhase(t, downloadProcessModel, modeldownload.PhaseQueued)
		if slices.Contains(f.journalModels(t), downloadProcessModel) {
			t.Fatal("pre-effect request created unsafe marker")
		}
		f.restart(t)
		f.reply(t, id, false)
		close(hold)
		after := f.models(t, f.local)
		if after.DownloadState.SourceID == before.DownloadState.SourceID || len(after.DownloadState.Entries) != 0 || queued.ID == "" {
			t.Fatalf("pre-effect restart retained admission history: %+v", after)
		}
		if posts, unloads, loads := f.router.counts(downloadProcessModel); posts != 0 || unloads != 0 || loads != 0 {
			t.Fatalf("pre-effect restart effects: posts=%d unloads=%d loads=%d", posts, unloads, loads)
		}
	})

	t.Run("restart after first shard recovers unresolved identity without replay", func(t *testing.T) {
		f := newDownloadProcessFixture(t, false)
		id := f.pull(t, downloadProcessModel)
		f.router.waitStarted(t, downloadProcessModel)
		before, active := f.waitPhase(t, downloadProcessModel, modeldownload.PhaseInProgress)
		if !slices.Contains(f.journalModels(t), downloadProcessModel) {
			t.Fatal("first effect lacked durable unsafe marker")
		}
		f.restart(t)
		f.reply(t, id, false)
		after, recovered := f.waitPhase(t, downloadProcessModel, modeldownload.PhaseUnknown)
		if after.DownloadState.SourceID == before.DownloadState.SourceID || recovered.ID != active.ID || !recovered.BlocksReadiness || recovered.Reason != modeldownload.ReasonInterrupted {
			t.Fatalf("first-shard restart recovery=%+v", after)
		}
		f.router.setStatus(downloadProcessModel, "unloaded")
		f.assertFences(t, downloadProcessModel)
		if posts, unloads, loads := f.router.counts(downloadProcessModel); posts != 1 || unloads > 1 || loads != 0 {
			t.Fatalf("first-shard restart effects: posts=%d unloads=%d loads=%d", posts, unloads, loads)
		}
	})

	t.Run("recent outcomes are bounded and fresh clients do not require evicted terminal", func(t *testing.T) {
		f := newDownloadProcessFixture(t, false)
		for index := 0; index <= modeldownload.MaxRecentEntries; index++ {
			model := fmt.Sprintf("fixture/recent-%02d:Q4_K_M", index)
			id := f.pull(t, model)
			f.router.waitStarted(t, model)
			f.router.setStatus(model, "unloaded")
			f.router.event(t, model, "download_finished")
			f.reply(t, id, true)
		}
		models := f.models(t, f.service.readyClient())
		if len(models.DownloadState.Entries) != modeldownload.MaxRecentEntries {
			t.Fatalf("recent entries=%d, want fixed limit %d", len(models.DownloadState.Entries), modeldownload.MaxRecentEntries)
		}
		for _, entry := range models.DownloadState.Entries {
			if entry.Model == "fixture/recent-00:Q4_K_M" || entry.Phase != modeldownload.PhaseComplete || entry.BlocksReadiness {
				t.Fatalf("invalid recent retention: %+v", entry)
			}
		}
		if !slices.Contains(models.ByEngine[downloadProcessEngine], "fixture/recent-00:Q4_K_M") {
			t.Fatal("evicted terminal removed independently ready inventory")
		}
		if len(f.journalModels(t)) != 0 {
			t.Fatal("recent replay grew durable unresolved journal")
		}
	})

	t.Run("unresolved capacity refuses a new key before native dispatch", func(t *testing.T) {
		f := newDownloadProcessFixture(t, false)
		for index := 0; index < modeldownload.MaxUnsafeEntries; index++ {
			model := fmt.Sprintf("fixture/unsafe-%02d:Q4_K_M", index)
			id := f.pull(t, model)
			f.router.waitStarted(t, model)
			f.router.setStatus(model, "downloaded")
			f.router.event(t, model, "download_failed")
			f.reply(t, id, false)
		}
		before := f.models(t, f.local)
		const refused = "fixture/over-capacity:Q4_K_M"
		id := f.pull(t, refused)
		f.reply(t, id, false)
		after := f.models(t, f.service.readyClient())
		if len(after.DownloadState.Entries) != modeldownload.MaxUnsafeEntries || !before.DownloadState.Equal(*after.DownloadState) || len(f.journalModels(t)) != modeldownload.MaxUnsafeEntries {
			t.Fatalf("capacity refusal changed retained unresolved state: %+v", after)
		}
		if posts, unloads, loads := f.router.counts(refused); posts != 0 || unloads != 0 || loads != 0 {
			t.Fatalf("capacity refusal effects: posts=%d unloads=%d loads=%d", posts, unloads, loads)
		}
	})
}
