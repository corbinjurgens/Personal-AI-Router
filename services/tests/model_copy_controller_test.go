// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"context"
	"crypto/tls"
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
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	psnet "github.com/shirou/gopsutil/v4/net"
	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-shared/testbin"
)

const controllerCopyAlias = "tiel-coder-35b-a3b-msi-ctx32768"

// Wire objects are deliberately decoded independently of the implementation's
// unexported types. Hash expectations come from the synthetic input, not from
// echoing the source's response or trusting the success acknowledgement.
type controllerCopyArtifact struct {
	Role   string `json:"role"`
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

type controllerCopyManifest struct {
	Engine   string                   `json:"engine"`
	Model    string                   `json:"model"`
	Layout   string                   `json:"layout"`
	ExportID string                   `json:"exportId"`
	Files    []controllerCopyArtifact `json:"files"`
}

type controllerCopyProgress struct {
	OpID              string `json:"opId"`
	SourceNodeID      string `json:"sourceNodeId"`
	DestinationNodeID string `json:"destinationNodeId"`
	Engine            string `json:"engine"`
	Model             string `json:"model"`
	Stage             string `json:"stage"`
	ExportID          string `json:"exportId"`
	BytesDone         int64  `json:"bytesDone"`
}

type controllerCopyModel struct {
	manifest controllerCopyManifest
	data     map[string][]byte
	paths    map[string]string
}

type controllerCopyRequest struct {
	OpID              string `json:"opId"`
	SourceNodeID      string `json:"sourceNodeId"`
	DestinationNodeID string `json:"destinationNodeId"`
	Engine            string `json:"engine"`
	Model             string `json:"model"`
	ExpectedExportID  string `json:"expectedExportId,omitempty"`
}

type controllerCopySeen struct{ peer, method, path, model, offset string }

// A fault belongs to one synthetic model/operation. There is no product flag,
// replacement transport, public URL, or alternate implementation of file IO.
type controllerCopyFault struct {
	model       string
	hold        bool
	loseAck     bool
	interrupt   bool
	beforeGET   func() error
	hit, closed chan struct{}
	release     chan struct{}
	once        sync.Once
	closeOnce   sync.Once
	used        bool
}

func (f *controllerCopyFault) unblock() { f.once.Do(func() { close(f.release) }) }

type controllerCopyObserver struct {
	port                   int
	mu                     sync.Mutex
	seen                   []controllerCopySeen
	fault                  *controllerCopyFault
	forbidden              []byte
	forbiddenResponseBytes int64
}

type controllerCopyObservedBody struct {
	io.ReadCloser
	owner *controllerCopyObserver
	tail  []byte
}

func (b *controllerCopyObservedBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	b.owner.mu.Lock()
	needle := b.owner.forbidden
	if len(needle) != 0 && n > 0 {
		data := append(b.tail, p[:n]...)
		b.owner.forbiddenResponseBytes += int64(bytes.Count(data, needle) * len(needle))
		keep := min(len(data), len(needle)-1)
		b.tail = bytes.Clone(data[len(data)-keep:])
	}
	b.owner.mu.Unlock()
	return n, err
}

// The body forwards positive bytes before holding an incomplete response.
// Delay one chunk so the real copy owner's progress interval elapses; the test
// still requires a received positive-progress frame and an on-disk partial.
type controllerCopyBody struct {
	io.ReadCloser
	ctx       context.Context
	fault     *controllerCopyFault
	remaining int
	delayed   bool
}

func (b *controllerCopyBody) Read(p []byte) (int, error) {
	if b.remaining == 0 {
		close(b.fault.hit)
		select {
		case <-b.fault.release:
		case <-b.ctx.Done():
		}
		return 0, io.ErrUnexpectedEOF
	}
	if !b.delayed && b.remaining <= phase04CutAt/2 {
		b.delayed = true
		timer := time.NewTimer(600 * time.Millisecond)
		defer timer.Stop()
		select {
		case <-timer.C:
		case <-b.ctx.Done():
			return 0, b.ctx.Err()
		}
	}
	if len(p) > b.remaining {
		p = p[:b.remaining]
	}
	n, err := b.ReadCloser.Read(p)
	b.remaining -= n
	return n, err
}

func (b *controllerCopyBody) Close() error {
	err := b.ReadCloser.Close()
	b.fault.closeOnce.Do(func() { close(b.fault.closed) })
	return err
}

func newControllerCopyObserver(t *testing.T, target *task1Node, nodes []*task1Node) *controllerCopyObserver {
	t.Helper()
	ln := listenFixtureLoopback(t)
	o := &controllerCopyObserver{port: ln.Addr().(*net.TCPAddr).Port}
	proxies := map[string]*httputil.ReverseProxy{}
	for _, caller := range nodes {
		if caller == target {
			continue
		}
		cfg, ok := caller.mesh.ClientTLSConfig(target.uuid)
		if !ok {
			t.Fatal("copy observer requires the caller's exact target pin")
		}
		upstream, err := url.Parse("https://127.0.0.1:" + strconv.Itoa(target.ports[noderec.ServiceEngineControl]))
		if err != nil {
			t.Fatal(err)
		}
		transport := &http.Transport{TLSClientConfig: cfg, DisableKeepAlives: true, Proxy: nil}
		t.Cleanup(transport.CloseIdleConnections)
		proxy := httputil.NewSingleHostReverseProxy(upstream)
		proxy.Transport, proxy.FlushInterval = transport, -1
		proxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, _ error) {
			http.Error(w, "private copy upstream unavailable", http.StatusBadGateway)
		}
		proxy.ModifyResponse = func(response *http.Response) error {
			if response.Request.URL.Path == "/v1/models/file" && response.StatusCode >= http.StatusOK && response.StatusCode < http.StatusMultipleChoices {
				response.Body = &controllerCopyObservedBody{ReadCloser: response.Body, owner: o}
			}
			o.mu.Lock()
			fault := o.fault
			file := fault != nil && fault.hold && !fault.used && response.Request.URL.Path == "/v1/models/file" && response.Request.URL.Query().Get("model") == fault.model && response.Request.Header.Get("Range") == "" && response.StatusCode == http.StatusOK
			lost := fault != nil && fault.loseAck && !fault.used && response.Request.URL.Path == "/v1/models/copy"
			if file || lost {
				fault.used = true
			}
			o.mu.Unlock()
			if file {
				response.Body = &controllerCopyBody{ReadCloser: response.Body, ctx: response.Request.Context(), fault: fault, remaining: phase04CutAt}
			}
			if lost {
				defer fault.closeOnce.Do(func() { close(fault.closed) })
				// Join the real accepted operation and close its upstream body,
				// then drop the downstream connection before ReverseProxy can
				// expose any response headers. RoundTrip must fail after dispatch,
				// with the authenticated alternate route still available.
				read, readErr := io.Copy(io.Discard, io.LimitReader(response.Body, (1<<20)+1))
				closeErr := response.Body.Close()
				response.Body = http.NoBody
				if readErr != nil || closeErr != nil {
					return errors.Join(readErr, closeErr)
				}
				if read > 1<<20 {
					return errors.New("copy acknowledgement exceeds fixture drain bound")
				}
				conn, ok := response.Request.Context().Value(task1ConnKey{}).(net.Conn)
				if !ok {
					return errors.New("copy observer lacks downstream connection")
				}
				if err := conn.Close(); err != nil {
					return fmt.Errorf("drop copy acknowledgement connection: %w", err)
				}
				close(fault.hit)
				panic(http.ErrAbortHandler)
			}
			return nil
		}
		proxies[caller.uuid] = proxy
	}
	server := &http.Server{ConnContext: func(ctx context.Context, conn net.Conn) context.Context {
		return context.WithValue(ctx, task1ConnKey{}, conn)
	}, Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		target.mesh.Refresh()
		peer, ok := target.mesh.VerifyClientPin(r)
		if !ok || proxies[peer] == nil {
			http.Error(w, "unpinned copy observer caller", http.StatusForbidden)
			return
		}
		o.mu.Lock()
		o.seen = append(o.seen, controllerCopySeen{peer, r.Method, r.URL.Path, r.URL.Query().Get("model"), r.Header.Get("Range")})
		fault := o.fault
		var swap func() error
		interrupted := fault != nil && fault.interrupt && fault.used && r.URL.Path == "/v1/models/file" && r.URL.Query().Get("model") == fault.model
		if fault != nil && fault.beforeGET != nil && r.URL.Path == "/v1/models/file" && r.URL.Query().Get("model") == fault.model {
			swap, fault.beforeGET = fault.beforeGET, nil
		}
		o.mu.Unlock()
		if interrupted {
			http.Error(w, "private source stream remains interrupted until explicit retry", http.StatusServiceUnavailable)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/v1/models/file") && (target.uuid == "copy-controller-mac" || peer == "copy-controller-mac") {
			t.Errorf("controller artifact transport attempted: target=%s caller=%s path=%s", target.uuid, peer, r.URL.Path)
			http.Error(w, "controller cannot read or serve model artifacts", http.StatusForbidden)
			return
		}
		if swap != nil {
			if err := swap(); err != nil {
				t.Errorf("swap private source fixture: %v", err)
				http.Error(w, "fixture swap failed", http.StatusInternalServerError)
				return
			}
		}
		proxies[peer].ServeHTTP(w, r)
	})}
	done := make(chan error, 1)
	go func() { done <- server.Serve(tls.NewListener(ln, target.mesh.ServerTLSConfig())) }()
	t.Cleanup(func() {
		o.mu.Lock()
		if o.fault != nil && o.fault.release != nil {
			o.fault.unblock()
		}
		o.mu.Unlock()
		if err := server.Close(); err != nil {
			t.Error(err)
		}
		if err := <-done; err != http.ErrServerClosed {
			t.Errorf("copy observer cleanup: %v", err)
		}
	})
	return o
}

func (o *controllerCopyObserver) requests() []controllerCopySeen {
	o.mu.Lock()
	defer o.mu.Unlock()
	return append([]controllerCopySeen(nil), o.seen...)
}

type controllerCopyFixture struct {
	controller, source, target    *task1Node
	broker, sourceEM, targetEM    *task1Process
	files                         map[string]*controllerCopyObserver
	stores                        map[string]string
	copyCluster, preset, external string
	aliases                       map[string]string
	ports                         []int
	blocked                       int
	op                            int
	egress                        atomic.Int32
}

func newControllerCopyFixture(t *testing.T) *controllerCopyFixture {
	t.Helper()
	if runtime.GOOS == "windows" && (!filepath.IsAbs(os.Getenv("NVPAIR_TEST_BIN_DIR")) || !strings.Contains(os.Getenv("GOFLAGS"), "nvpair-test-exec")) {
		t.Fatal("controller copy requires the registered fixed-path Windows test wrapper")
	}
	f := &controllerCopyFixture{files: map[string]*controllerCopyObserver{}, stores: map[string]string{}, aliases: map[string]string{}}
	// The caller owns one serialized native lane. TestMain's existing registered
	// products are reused; no new listener image or executable role is created.
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	scanner := filepath.Join(testbin.Dir(t, "tests", "subscription-scanner", "subscription-scanner"), testbin.Exe("subscription-scanner"))
	jointCopyExecutable(t, self, scanner)
	paths := []string{brokerBin, engineMgrBin, clusterMgrBin, scanner}
	t.Cleanup(func() {
		inventory := &serviceProcessInventory{known: map[int32]serviceProcessIdentity{}}
		for _, path := range paths {
			canonical, err := canonicalServicePath(path)
			if err != nil {
				t.Error(err)
				continue
			}
			inventory.paths = append(inventory.paths, canonical)
		}
		owned, err := inventory.scan()
		if err != nil || len(owned) != 0 {
			t.Errorf("copy fixture survivors=%+v err=%v", owned, err)
		}
		for _, port := range f.ports {
			assertInteropListenerClosed(t, port)
		}
	})
	// Remove inherited engine/model selectors before any constructor. The
	// fixture owns only its child environment; no installed setting is changed.
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		upper := strings.ToUpper(key)
		if (strings.HasPrefix(upper, "LLAMA_") && upper != "LLAMA_CPP_REPO") || strings.HasPrefix(upper, "OLLAMA_") || strings.HasPrefix(upper, "HF_") || (strings.HasPrefix(upper, "NVPAIR_TEST_") && upper != "NVPAIR_TEST_BIN_DIR") {
			t.Setenv(key, "")
		}
	}
	deny := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		f.egress.Add(1)
		http.Error(w, "fixture forbids catalog/download egress", http.StatusForbidden)
	}))
	t.Cleanup(deny.Close)
	for _, key := range []string{"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"} {
		t.Setenv(key, deny.URL)
	}
	t.Setenv("NO_PROXY", "127.0.0.1,localhost,::1")
	// Hold service allocations together; no two nodes can receive the same port.
	var reservations []net.Listener
	defer func() {
		for _, listener := range reservations {
			if err := listener.Close(); err != nil {
				t.Error(err)
			}
		}
	}()
	allocate := func() int {
		ln := listenFixtureLoopback(t)
		reservations = append(reservations, ln)
		port := ln.Addr().(*net.TCPAddr).Port
		f.ports = append(f.ports, port)
		return port
	}
	plan := fixtureServicePorts{nodeInfo: allocate(), errors: allocate(), workload: allocate(), clusterManager: allocate(), engineManagerHTTP: allocate(), engineControl: allocate()}
	if err := plan.validate(); err != nil {
		t.Fatal(err)
	}
	scannerPort := allocate()
	for _, id := range []string{"copy-controller-mac", "copy-source-msi", "copy-target-joes"} {
		base := shortClusterTempDir(t)
		n := &task1Node{uuid: id, base: base, ports: map[noderec.ServiceKey]int{}, observers: map[noderec.ServiceKey]*task1Observer{}}
		dir := filepath.Join(base, "cluster")
		if err := os.MkdirAll(filepath.Join(dir, "trusted"), 0o700); err != nil {
			t.Fatal(err)
		}
		n.cert = mintClusterIdentity(t, dir, id)
		task1WriteJSON(t, filepath.Join(dir, "identity.json"), map[string]any{"node_uuid": id, "created_at": 1})
		task1WriteJSON(t, filepath.Join(dir, "admission.json"), map[string]any{"counter": 1, "activated": 1, "clusterId": task1ClusterID, "epoch": 1})
		n.ports[noderec.ServiceCluster], n.ports[noderec.ServiceEngineControl] = allocate(), allocate()
		f.stores[id] = filepath.Join(base, "model-store") // Deliberately absent on the controller.
		switch id {
		case "copy-controller-mac":
			f.controller = n
			n.ports[noderec.ServiceEngineControl] = plan.engineControl
		case "copy-source-msi":
			f.source = n
		case "copy-target-joes":
			f.target = n
		}
	}
	nodes := []*task1Node{f.controller, f.source, f.target}
	for _, n := range nodes {
		for _, peer := range nodes {
			if peer != n {
				task1WriteJSON(t, filepath.Join(n.base, "cluster", "trusted", peer.uuid+".json"), map[string]any{"nodeUuid": peer.uuid, "nodeId": peer.uuid, "certPem": peer.cert, "clusterId": task1ClusterID, "admissionEpoch": 1})
			}
		}
		n.mesh = clustertrust.Open(filepath.Join(n.base, "cluster"))
	}
	blocked := listenFixtureLoopback(t)
	f.blocked = blocked.Addr().(*net.TCPAddr).Port
	f.ports = append(f.ports, f.blocked)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			conn, err := blocked.Accept()
			if err != nil {
				return
			}
			if err := conn.Close(); err != nil {
				t.Errorf("close rejected fixture connection: %v", err)
			}
		}
	}()
	t.Cleanup(func() {
		if err := blocked.Close(); err != nil {
			t.Error(err)
		}
		<-done
	})
	for _, n := range nodes {
		n.observers[noderec.ServiceCluster] = newTask1Observer(t, n, noderec.ServiceCluster, nodes)
		f.files[n.uuid] = newControllerCopyObserver(t, n, nodes)
		f.ports = append(f.ports, n.observers[noderec.ServiceCluster].port, f.files[n.uuid].port)
	}
	for _, ln := range reservations {
		if err := ln.Close(); err != nil {
			t.Fatal(err)
		}
	}
	reservations = nil
	for _, n := range nodes {
		var members []task1Member
		for _, peer := range nodes {
			port := peer.observers[noderec.ServiceCluster].port
			if peer == f.source {
				port = f.blocked
			}
			members = append(members, task1Member{ID: peer.uuid, NodeUUID: peer.uuid, Name: peer.uuid, IPAddress: "127.0.0.1", Port: port, ClusterID: task1ClusterID, AdmissionEpoch: 1, State: "member"})
		}
		task1WriteJSON(t, filepath.Join(n.base, "cluster", "members.json"), members)
		writeDisabledBuiltinEngineOverrides(t, n.base, jointAppDir(n.base))
		// Known Off and locked also establish that COPY does not activate an engine.
		if err := os.MkdirAll(filepath.Join(jointAppDir(n.base), "engine-bin"), 0o700); err != nil {
			t.Fatal(err)
		}
		task1WriteJSON(t, filepath.Join(jointAppDir(n.base), "engine-bin", "engine-state.json"), map[string]any{"engines": map[string]bool{"llamacpp": false}, "locked": map[string]bool{"llamacpp": true}})
		facades := map[string]int{}
		for _, engine := range engines.All() {
			facades[engine.Name] = allocatePrivateCopyPort(t)
		}
		writePrivateBrokerEngineSettings(t, n.base, filepath.Join(n.base, "cluster"), facades)
		n.cm = startTask1Process(t, n.base, clusterMgrBin, "--config-dir", n.base, "--port", strconv.Itoa(n.ports[noderec.ServiceCluster]))
		n.cm.call(t, "cluster:get-node-id", nil)
		n.cm.call(t, "cluster:set-local-services", map[string]any{"generation": 1, "services": []noderec.RegisterParams{{Service: noderec.ServiceCluster, Port: n.observers[noderec.ServiceCluster].port}, {Service: noderec.ServiceEngineControl, Port: f.files[n.uuid].port}}})
	}
	managementSet(t, f.target.cm, f.controller.uuid, true)
	f.copyCluster = f.cloneTargetAuthority(t)
	f.external = filepath.Join(f.source.base, "external", "huggingface", "hub")
	f.preset = filepath.Join(f.source.base, "external", "models.ini")
	if err := os.MkdirAll(f.external, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(f.preset, []byte("# synthetic controller COPY fixture\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, n := range nodes {
		if n == f.controller {
			continue
		}
		f.writeManifest(t, n)
		if err := os.MkdirAll(f.stores[n.uuid], 0o700); err != nil {
			t.Fatal(err)
		}
		dir := filepath.Join(n.base, "cluster")
		if n == f.target {
			dir = f.copyCluster
		}
		manager := startTask1Process(t, n.base, engineMgrBin, "--cluster-dir", dir, "--control-host", "127.0.0.1", "--control-port", strconv.Itoa(n.ports[noderec.ServiceEngineControl]), "--http-port", "0", "--loaded-poll-interval", "0")
		completeEngineManagerRecovery(t, func(method string, params any) jsonrpc.Message { return manager.call(t, method, params) })
		policy := nodepolicy.Default()
		manager.send(t, "engine:node-projection", map[string]any{"policy": policy.Advertisement(), "pausedLocally": false, "managementBlocked": false}, false)
		manager.call(t, "engine:models", nil) // Ordered projection/recovery barrier.
		if n == f.source {
			f.sourceEM = manager
		} else {
			f.targetEM = manager
		}
	}
	args := append(plan.args(), "--scanner-path", scanner, "--engine-manager-path", engineMgrBin, "--engine-manager-control-host", "127.0.0.1", "--proxy-engines=", "--cluster-dir", filepath.Join(f.controller.base, "cluster"))
	f.broker = startTask1ProcessWithEnv(t, f.controller.base, brokerBin, []string{"NVPAIR_TEST_SUBSCRIPTION_SCANNER=" + scanner, "NVPAIR_TEST_SUBSCRIPTION_PORT=" + strconv.Itoa(scannerPort)}, args...)
	task1Eventually(t, "controller broker readiness", 20*time.Second, func() bool {
		f.broker.mu.Lock()
		defer f.broker.mu.Unlock()
		for _, frame := range f.broker.frames {
			if frame.Method == "app:ready" {
				return true
			}
		}
		return false
	})
	f.broker.call(t, "engine:get-installed", nil)
	f.broker.call(t, "engine:subscribe", map[string]any{})
	waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(scannerPort)), 15*time.Second)
	client := &http.Client{Timeout: 5 * time.Second}
	t.Cleanup(client.CloseIdleConnections)
	for _, n := range nodes {
		event := noderec.NodeEvent{Node: f.directory(n, false)}
		frame := jsonrpc.Message{JSONRPC: "2.0", Method: noderec.NotifyNodeDiscovered, Params: mustMarshal(t, event)}
		egressProcessHTTP(t, client, http.MethodPost, scannerPort, "/", mustMarshal(t, frame), http.StatusOK)
	}
	f.directoryToManagers(t, false)
	for _, n := range nodes {
		f.assertControlBind(t, n.ports[noderec.ServiceEngineControl])
		t.Logf("copy_identity host=%s cluster=%s private_root=%q ec=127.0.0.1:%d", n.uuid, n.mesh.NodeUUID(), n.base, n.ports[noderec.ServiceEngineControl])
	}
	task1Eventually(t, "source-to-target authenticated reverse session", 20*time.Second, func() bool { return f.target.observers[noderec.ServiceCluster].established(f.source.uuid) > 0 })
	task1Eventually(t, "target-to-controller alternate authenticated reverse session", 20*time.Second, func() bool { return f.controller.observers[noderec.ServiceCluster].established(f.target.uuid) > 0 })
	return f
}

func allocatePrivateCopyPort(t *testing.T) int {
	t.Helper()
	ln := listenFixtureLoopback(t)
	port := ln.Addr().(*net.TCPAddr).Port
	if err := ln.Close(); err != nil {
		t.Fatal(err)
	}
	return port
}

func (f *controllerCopyFixture) cloneTargetAuthority(t *testing.T) string {
	t.Helper()
	dir := filepath.Join(f.target.base, "copy-cluster")
	names := []string{"node.crt", "node.key", "identity.json", "admission.json", "members.json", clustertrust.RemoteManagementFile, clustertrust.RemoteManagementMigrationFile}
	for _, peer := range []*task1Node{f.source, f.controller} {
		names = append(names, filepath.Join("trusted", peer.uuid+".json"))
	}
	for _, name := range names {
		data := readLiveDurableFile(t, filepath.Join(f.target.base, "cluster", name))
		path := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func (f *controllerCopyFixture) writeManifest(t *testing.T, n *task1Node) {
	t.Helper()
	f.writeManifestWithRouter(t, n, 0)
}

// Task 6 adopts only an already-listening synthetic router. The installed
// marker remains non-executable, so this never authorizes an engine launch.
func (f *controllerCopyFixture) writeManifestWithRouter(t *testing.T, n *task1Node, port int) {
	t.Helper()
	args := []string{}
	if n == f.source {
		args = []string{"--models-preset", f.preset}
	}
	// Detect uses existence and also remembers the detected path for launch.
	// A non-executable private marker proves installation without making any
	// real engine or service image available to an accidental activation.
	marker := filepath.Join(n.base, "synthetic-engine-installed")
	if err := os.WriteFile(marker, []byte("fixture installation marker; not executable"), 0o600); err != nil {
		t.Fatal(err)
	}
	rt := map[string]any{"bin": marker, "cli": filepath.Join(n.base, "never-start-cli"), "port": 0, "args": args, "env": map[string]string{"LLAMA_CACHE": f.stores[n.uuid]}, "ready": nil, "health": nil, "auth_statuses": []int{}}
	rt["editable_launch"] = map[string]any{"fixed_args": []string{}}
	if port != 0 {
		if err := validateFixturePort(port); err != nil {
			t.Fatal(err)
		}
		rt["port"] = port
		rt["ready"] = map[string]any{"http": "http://127.0.0.1:{port}/ready", "status": http.StatusOK, "timeout_s": 1}
	}
	task1WriteJSON(t, filepath.Join(jointAppDir(n.base), "engines", "llamacpp.json"), map[string]any{"engine": "llamacpp", "models_dir": f.stores[n.uuid], "runtime": rt, "platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{"detect": []string{marker}, "models_dir": f.stores[n.uuid], "runtime": rt}}})
}

func (f *controllerCopyFixture) directory(n *task1Node, reverse bool) noderec.DirectoryNode {
	port := f.files[n.uuid].port
	if reverse && n == f.source {
		port = f.blocked
	}
	return noderec.DirectoryNode{HostUUID: n.uuid, ClusterUUID: n.uuid, Name: n.uuid, IP: "127.0.0.1", Trusted: true, Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineControl: {Port: port}}}
}

func (f *controllerCopyFixture) directoryToManagers(t *testing.T, reverse bool) {
	t.Helper()
	nodes := []noderec.DirectoryNode{f.directory(f.controller, reverse), f.directory(f.source, reverse), f.directory(f.target, reverse)}
	for _, manager := range []*task1Process{f.sourceEM, f.targetEM} {
		manager.send(t, noderec.NotifyNodes, noderec.GetNodesResult{Nodes: nodes}, false)
		manager.call(t, "engine:configured-ports", nil)
	}
}

func (f *controllerCopyFixture) assertControlBind(t *testing.T, port int) {
	t.Helper()
	// A successful loopback dial alone also accepts a wildcard listener. Inspect
	// the actual OS listener instead of inferring isolation from its arguments.
	task1Eventually(t, "actual loopback ec listener", 10*time.Second, func() bool {
		connections, err := psnet.Connections("tcp")
		if err != nil {
			t.Fatalf("inspect actual ec bind: %v", err)
		}
		found := false
		for _, connection := range connections {
			if connection.Laddr.Port == uint32(port) && connection.Status == "LISTEN" {
				if connection.Laddr.IP != "127.0.0.1" {
					t.Fatalf("ec port %d bound %s", port, connection.Laddr.IP)
				}
				found = true
			}
		}
		return found
	})
}

func (f *controllerCopyFixture) seed(t *testing.T, alias string, seed byte) controllerCopyModel {
	t.Helper()
	return f.seedSized(t, alias, seed, 768<<10)
}

func (f *controllerCopyFixture) seedSized(t *testing.T, alias string, seed byte, modelBytes int) controllerCopyModel {
	t.Helper()
	m := controllerCopyModel{manifest: controllerCopyManifest{Engine: "llamacpp", Model: alias, Layout: "llama-artifacts-v1"}, data: map[string][]byte{}, paths: map[string]string{}}
	for _, item := range []struct {
		role string
		size int
	}{{"model", modelBytes}, {"draft", 4096}} {
		name := alias + "-" + item.role + ".gguf"
		rel := item.role + "/" + name
		data := make([]byte, item.size)
		for i := range data {
			data[i] = byte(i*17) + seed
		}
		copy(data, []byte("GGUF\x03\x00\x00\x00"))
		path := filepath.Join(f.external, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		m.data[rel], m.paths[rel] = data, path
		m.manifest.Files = append(m.manifest.Files, controllerCopyArtifact{Role: item.role, Path: rel, Size: int64(len(data)), SHA256: phase04SHA(data)})
	}
	m.manifest.ExportID = phase04SHA(mustMarshal(t, struct {
		Version   int                      `json:"version"`
		Model     string                   `json:"model"`
		Artifacts []controllerCopyArtifact `json:"artifacts"`
	}{1, alias, m.manifest.Files}))
	f.aliases[alias] = fmt.Sprintf("[%s]\nmodel = huggingface/hub/%s-model.gguf\nmodel-draft = huggingface/hub/%s-draft.gguf\n", alias, alias, alias)
	var preset strings.Builder
	var aliases []string
	for alias := range f.aliases {
		aliases = append(aliases, alias)
	}
	sort.Strings(aliases)
	for _, alias := range aliases {
		preset.WriteString(f.aliases[alias])
	}
	if err := os.WriteFile(f.preset, []byte(preset.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	return m
}

func (f *controllerCopyFixture) request(model, expected string) controllerCopyRequest {
	f.op++
	return controllerCopyRequest{OpID: fmt.Sprintf("%032x", f.op), SourceNodeID: f.source.uuid, DestinationNodeID: f.target.uuid, Engine: "llamacpp", Model: model, ExpectedExportID: expected}
}

func controllerCopyAliasDir(root, alias string) string {
	return filepath.Join(root, ".pair-copies", phase04SHA([]byte(alias)))
}

func (f *controllerCopyFixture) assertNoReceipt(t *testing.T, m controllerCopyModel) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(controllerCopyAliasDir(f.stores[f.target.uuid], m.manifest.Model), "copy.json")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("incomplete/refused custom receipt exists: %v", err)
	}
}

func (f *controllerCopyFixture) assertCopied(t *testing.T, m controllerCopyModel, response jsonrpc.Message) {
	t.Helper()
	f.assertCopiedArtifacts(t, m, response)
	f.assertController(t)
}

// Artifact/receipt verification is shared with the download join, whose source
// or destination can report an adopted fixture router independently of COPY.
func (f *controllerCopyFixture) assertCopiedArtifacts(t *testing.T, m controllerCopyModel, response jsonrpc.Message) {
	t.Helper()
	if response.Error != nil {
		t.Fatalf("controller COPY: %+v", response.Error)
	}
	var result struct {
		OpID   string `json:"opId"`
		Result struct {
			Layout                string `json:"layout"`
			ExportID              string `json:"exportId"`
			Files                 int    `json:"files"`
			BytesTotal            int64  `json:"bytesTotal"`
			RequiresConfiguration bool   `json:"requiresConfiguration"`
			Durability            string `json:"durability"`
		} `json:"result"`
	}
	task2Decode(t, response.Result, &result)
	var total int64
	root := controllerCopyAliasDir(f.stores[f.target.uuid], m.manifest.Model)
	for rel, want := range m.data {
		total += int64(len(want))
		path := filepath.Join(root, m.manifest.ExportID, filepath.FromSlash(rel))
		got, err := os.ReadFile(path)
		if err != nil || phase04SHA(got) != phase04SHA(want) {
			t.Fatalf("destination full hash %s: %v", rel, err)
		}
		if _, err := os.Stat(path + ".part"); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("completed partial remains: %v", err)
		}
		source, err := os.ReadFile(m.paths[rel])
		if err != nil || !bytes.Equal(source, want) {
			t.Fatalf("source changed %s: %v", rel, err)
		}
		t.Logf("copy_file_verified path=%s sha256=%s source_unchanged=true", rel, phase04SHA(want))
	}
	var receipt struct {
		Version  int                    `json:"version"`
		Manifest controllerCopyManifest `json:"manifest"`
	}
	task2Decode(t, readLiveDurableFile(t, filepath.Join(root, "copy.json")), &receipt)
	if receipt.Version != 1 || !bytes.Equal(mustMarshal(t, receipt.Manifest), mustMarshal(t, m.manifest)) {
		t.Fatalf("destination receipt differs from independently hashed manifest: %+v", receipt)
	}
	if result.OpID == "" || result.Result.Layout != m.manifest.Layout || result.Result.ExportID != m.manifest.ExportID || result.Result.Files != len(m.data) || result.Result.BytesTotal != total || !result.Result.RequiresConfiguration || (result.Result.Durability != "confirmed" && result.Result.Durability != "uncertain") {
		t.Fatalf("copy result=%+v", result)
	}
	t.Logf("copy_verified model=%s export=%s bytes=%d durability=%s", m.manifest.Model, m.manifest.ExportID, total, result.Result.Durability)
	for _, seen := range f.files[f.source.uuid].requests() {
		if seen.model == m.manifest.Model && seen.path == "/v1/models/file" {
			t.Logf("copy_source_reader certificate_identity=%s range=%q", seen.peer, seen.offset)
		}
	}
}

func (f *controllerCopyFixture) assertController(t *testing.T) {
	t.Helper()
	// These are observable transport/store assertions. Production has no
	// filesystem-attempt observer; they cannot establish zero attempted local
	// artifact reads/writes. That remaining acceptance seam is documented.
	for _, name := range []string{f.stores[f.controller.uuid], filepath.Join(f.controller.base, "absent-models")} {
		if _, err := os.Stat(name); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("controller artifact store created: %q %v", name, err)
		}
	}
	for _, request := range f.files[f.source.uuid].requests() {
		if request.path == "/v1/models/file" || request.path == "/v1/models/files" {
			if request.peer != f.target.uuid {
				t.Fatalf("source artifact reader=%s, want destination certificate", request.peer)
			}
		}
	}
	for _, request := range f.files[f.controller.uuid].requests() {
		if strings.HasPrefix(request.path, "/v1/models/file") {
			t.Fatalf("controller artifact serve attempted: %+v", request)
		}
	}
	if calls := f.egress.Load(); calls != 0 {
		t.Fatalf("catalog/downloader egress=%d", calls)
	}
	for _, manager := range []*task1Process{f.broker, f.sourceEM, f.targetEM} {
		var installed struct {
			Engines []struct {
				Engine  string `json:"engine"`
				Running bool   `json:"running"`
				Locked  *bool  `json:"locked"`
			} `json:"engines"`
		}
		task2Decode(t, manager.call(t, "engine:get-installed", nil).Result, &installed)
		if len(installed.Engines) != len(engines.All()) {
			t.Fatal("incomplete isolated engine inventory")
		}
		for _, engine := range installed.Engines {
			if engine.Running {
				t.Fatalf("COPY activated %s", engine.Engine)
			}
			if engine.Engine == "llamacpp" && (engine.Locked == nil || !*engine.Locked) {
				t.Fatal("COPY lost the durable local engine lock")
			}
		}
	}
}

func (f *controllerCopyFixture) idle(t *testing.T) {
	t.Helper()
	for _, manager := range []*task1Process{f.broker, f.targetEM} {
		task1Eventually(t, "empty active COPY reservations", 10*time.Second, func() bool {
			var state struct {
				Operations []json.RawMessage `json:"operations"`
			}
			task2Decode(t, manager.call(t, "engine:active-model-copies", map[string]any{}).Result, &state)
			return len(state.Operations) == 0
		})
	}
}

func controllerCopyProgressFor(t *testing.T, manager *task1Process, request controllerCopyRequest, positive bool) controllerCopyProgress {
	t.Helper()
	var progress controllerCopyProgress
	task1Eventually(t, "controller tuple progress", 15*time.Second, func() bool {
		manager.mu.Lock()
		defer manager.mu.Unlock()
		for _, frame := range manager.frames {
			if frame.Method != "engine:remote-progress" {
				continue
			}
			var candidate controllerCopyProgress
			if err := json.Unmarshal(frame.Params, &candidate); err != nil {
				t.Fatal(err)
			}
			if candidate.OpID != request.OpID {
				continue
			}
			if candidate.SourceNodeID != request.SourceNodeID || candidate.DestinationNodeID != request.DestinationNodeID || candidate.Model != request.Model || candidate.Engine != request.Engine {
				t.Fatalf("copy tuple changed: %+v", candidate)
			}
			if candidate.ExportID != "" && (!positive || candidate.Stage == "downloading" && candidate.BytesDone > 0) {
				progress = candidate
				return true
			}
		}
		return false
	})
	return progress
}

func controllerCopyResponse(t *testing.T, manager *task1Process, id int) jsonrpc.Message {
	t.Helper()
	// Exhausting the production file retry policy includes bounded backoff.
	// Keep that policy unchanged instead of adding a fast-retry product flag.
	task1Eventually(t, "copy response after bounded file retries", time.Minute, func() bool {
		manager.mu.Lock()
		defer manager.mu.Unlock()
		for _, frame := range manager.frames {
			if idEquals(frame.ID, id) {
				return true
			}
		}
		return false
	})
	return task2Response(t, manager, id)
}

func controllerCopyTerminal(t *testing.T, manager *task1Process, request controllerCopyRequest, stage string) {
	t.Helper()
	manager.mu.Lock()
	defer manager.mu.Unlock()
	count := 0
	for _, frame := range manager.frames {
		if frame.Method != "engine:remote-progress" {
			continue
		}
		var progress controllerCopyProgress
		if err := json.Unmarshal(frame.Params, &progress); err != nil {
			t.Fatal(err)
		}
		if progress.OpID != request.OpID {
			continue
		}
		switch progress.Stage {
		case "done", "error", "cancelled", "unknown":
			count++
			if progress.Stage != stage || progress.SourceNodeID != request.SourceNodeID || progress.DestinationNodeID != request.DestinationNodeID || progress.Model != request.Model {
				t.Fatalf("terminal tuple/outcome=%+v, want %s", progress, stage)
			}
		}
	}
	if count != 1 {
		t.Fatalf("terminal outcome frames=%d, want exactly one %s", count, stage)
	}
}

func TestModelCopyControllerProcess(t *testing.T) {
	f := newControllerCopyFixture(t)
	custom := func(name string, reverse bool) {
		t.Run(name, func(t *testing.T) {
			f.directoryToManagers(t, reverse)
			m := f.seed(t, controllerCopyAlias, 7)
			request := f.request(m.manifest.Model, "")
			id := f.broker.send(t, "engine:copy-model", request, true)
			f.assertCopied(t, m, task2Response(t, f.broker, id))
			controllerCopyTerminal(t, f.broker, request, "done")
			progress := controllerCopyProgressFor(t, f.broker, request, false)
			if progress.ExportID != m.manifest.ExportID {
				t.Fatal("controller learned another export identity")
			}
			f.idle(t)
		})
	}
	custom("reported alias direct with external model and draft", false)
	// A new alias forces a byte transfer; a matching receipt must not turn the
	// reverse scenario into a destination-only idempotency check.
	t.Run("authenticated reverse source", func(t *testing.T) {
		f.directoryToManagers(t, true)
		m := f.seed(t, "reverse-custom", 9)
		before := len(f.files[f.source.uuid].requests())
		id := f.broker.send(t, "engine:copy-model", f.request(m.manifest.Model, ""), true)
		f.assertCopied(t, m, task2Response(t, f.broker, id))
		if len(f.files[f.source.uuid].requests()) <= before {
			t.Fatal("reverse copy never reached source")
		}
		f.idle(t)
	})
	t.Run("destination local Copy to this PC uses reported alias", func(t *testing.T) {
		f.directoryToManagers(t, false)
		m := f.seed(t, controllerCopyAlias, 7)
		// Require the original destination-local workflow to fetch bytes again;
		// a previous controller receipt must not conceal source alias lookup.
		if err := os.RemoveAll(controllerCopyAliasDir(f.stores[f.target.uuid], controllerCopyAlias)); err != nil {
			t.Fatal(err)
		}
		before := len(f.files[f.source.uuid].requests())
		id := f.targetEM.send(t, "engine:remote-copy-model", map[string]string{"node": f.source.uuid, "engine": "llamacpp", "model": controllerCopyAlias}, true)
		f.assertCopied(t, m, task2Response(t, f.targetEM, id))
		fetched := false
		for _, request := range f.files[f.source.uuid].requests()[before:] {
			if request.path == "/v1/models/file" {
				fetched = true
			}
		}
		if !fetched {
			t.Fatal("destination-local alias copy did not fetch source bytes")
		}
		f.idle(t)
	})
	t.Run("canonical Hugging Face cache preserves baseline", func(t *testing.T) {
		const repo, commit = "owner/tiny-GGUF", "0123456789abcdef0123456789abcdef01234567"
		repoDir := "models--owner--tiny-GGUF"
		data := []byte("GGUF\x03\x00\x00\x00 synthetic canonical fixture")
		for rel, content := range map[string][]byte{filepath.Join(repoDir, "refs", "main"): []byte(commit), filepath.Join(repoDir, "snapshots", commit, "tiny-Q4_K_M.gguf"): data} {
			path := filepath.Join(f.stores[f.source.uuid], rel)
			if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, content, 0o600); err != nil {
				t.Fatal(err)
			}
		}
		id := f.broker.send(t, "engine:copy-model", f.request(repo+":Q4_K_M", ""), true)
		if response := task2Response(t, f.broker, id); response.Error != nil {
			t.Fatalf("canonical COPY: %+v", response.Error)
		}
		copied := readLiveDurableFile(t, filepath.Join(f.stores[f.target.uuid], repoDir, "snapshots", commit, "tiny-Q4_K_M.gguf"))
		if phase04SHA(copied) != phase04SHA(data) {
			t.Fatal("canonical full hash differs")
		}
		if source := readLiveDurableFile(t, filepath.Join(f.stores[f.source.uuid], repoDir, "snapshots", commit, "tiny-Q4_K_M.gguf")); !bytes.Equal(source, data) {
			t.Fatal("canonical source changed")
		}
		if ref := readLiveDurableFile(t, filepath.Join(f.stores[f.target.uuid], repoDir, "refs", "main")); string(ref) != commit {
			t.Fatal("canonical snapshot not published")
		}
		f.assertController(t)
		f.idle(t)
	})
	t.Run("ambiguous source alias is explicitly refused", func(t *testing.T) {
		m := f.seed(t, "ambiguous-alias", 11)
		original := readLiveDurableFile(t, f.preset)
		defer func() {
			if err := os.WriteFile(f.preset, original, 0o600); err != nil {
				t.Error(err)
			}
		}()
		duplicate := append(append([]byte{}, original...), []byte("\n[ambiguous-alias]\nmodel = other.gguf\n")...)
		if err := os.WriteFile(f.preset, duplicate, 0o600); err != nil {
			t.Fatal(err)
		}
		id := f.broker.send(t, "engine:copy-model", f.request(m.manifest.Model, ""), true)
		response := task2Response(t, f.broker, id)
		if response.Error == nil || !strings.Contains(response.Error.Message, "unique exact model IDs") {
			t.Fatalf("ambiguous alias refusal=%+v", response.Error)
		}
		f.assertNoReceipt(t, m)
		f.idle(t)
	})
	interrupt := func(name, alias string, reverse, cancel bool) {
		t.Run(name, func(t *testing.T) {
			f.directoryToManagers(t, reverse)
			m := f.seed(t, alias, 13)
			request := f.request(alias, "")
			fault := &controllerCopyFault{model: alias, hold: true, interrupt: !cancel, hit: make(chan struct{}), closed: make(chan struct{}), release: make(chan struct{})}
			defer fault.unblock()
			observer := f.files[f.source.uuid]
			observer.mu.Lock()
			observer.fault = fault
			observer.mu.Unlock()
			defer func() { observer.mu.Lock(); observer.fault = nil; observer.mu.Unlock() }()
			id := f.broker.send(t, "engine:copy-model", request, true)
			select {
			case <-fault.hit:
			case <-time.After(15 * time.Second):
				t.Fatal("source did not expose incomplete positive bytes")
			}
			progress := controllerCopyProgressFor(t, f.broker, request, true)
			if progress.ExportID != m.manifest.ExportID {
				t.Fatal("interrupted export identity differs")
			}
			part := filepath.Join(controllerCopyAliasDir(f.stores[f.target.uuid], alias), m.manifest.ExportID, filepath.FromSlash(m.manifest.Files[0].Path)) + ".part"
			task1Eventually(t, "positive destination partial", 5*time.Second, func() bool { info, err := os.Stat(part); return err == nil && info.Size() == phase04CutAt })
			f.assertNoReceipt(t, m)
			// Both interruption and cancellation are user-visible and require a
			// new explicit, identity-bearing request. No automatic POST retry.
			if cancel {
				var result struct {
					Cancelled bool `json:"cancelled"`
				}
				task2Decode(t, f.broker.call(t, "engine:cancel-model-copy", map[string]string{"opId": request.OpID}).Result, &result)
				if !result.Cancelled {
					t.Fatal("controller did not own cancellable operation")
				}
			} else {
				// Drop this exposed file response. Later automatic file attempts
				// stay unavailable until the user makes the explicit COPY retry.
				fault.unblock()
			}
			if response := controllerCopyResponse(t, f.broker, id); response.Error == nil {
				t.Fatal("interrupted copy reported success")
			}
			stage := "error"
			if cancel {
				stage = "cancelled"
			}
			controllerCopyTerminal(t, f.broker, request, stage)
			select {
			case <-fault.closed:
			case <-time.After(10 * time.Second):
				t.Fatal("source response body was not closed")
			}
			f.idle(t)
			f.assertNoReceipt(t, m)
			partial := readLiveDurableFile(t, part)
			if !bytes.Equal(partial, m.data[m.manifest.Files[0].Path][:phase04CutAt]) {
				t.Fatal("cancelled partial did not preserve exact prefix")
			}
			fault.unblock()
			observer.mu.Lock()
			observer.fault = nil
			observer.mu.Unlock()
			beforeRetry := len(observer.requests())
			retry := f.request(alias, progress.ExportID)
			id = f.broker.send(t, "engine:copy-model", retry, true)
			f.assertCopied(t, m, task2Response(t, f.broker, id))
			resumed := false
			for _, seen := range observer.requests()[beforeRetry:] {
				if seen.model == alias && seen.path == "/v1/models/file" && seen.offset == "bytes=262144-" && seen.peer == f.target.uuid {
					resumed = true
				}
			}
			if !resumed {
				t.Fatal("explicit retry did not send observed nonzero Range")
			}
			f.idle(t)
		})
	}
	interrupt("direct interruption retains partial for explicit retry", "direct-resume", false, false)
	interrupt("controller cancel closes authenticated reverse body", "reverse-cancel", true, true)
	// Each refusal uses fresh content so an earlier completed receipt cannot
	// short-circuit the source leg or conceal an unauthorized publication.
	t.Run("destination permission denial", func(t *testing.T) {
		m := f.seed(t, "denied-target", 17)
		managementSet(t, f.target.cm, f.controller.uuid, false)
		f.cloneTargetAuthority(t)
		defer func() { managementSet(t, f.target.cm, f.controller.uuid, true); f.cloneTargetAuthority(t) }()
		before := len(f.files[f.source.uuid].requests())
		id := f.broker.send(t, "engine:copy-model", f.request(m.manifest.Model, ""), true)
		response := task2Response(t, f.broker, id)
		if response.Error == nil || response.Error.Code != -32012 {
			t.Fatalf("target grant refusal=%+v", response.Error)
		}
		if len(f.files[f.source.uuid].requests()) != before {
			t.Fatal("denied destination read source")
		}
		f.assertNoReceipt(t, m)
		f.idle(t)
	})
	t.Run("destination needs its own source pin", func(t *testing.T) {
		m := f.seed(t, "missing-source-pin", 19)
		pin := filepath.Join(f.copyCluster, "trusted", f.source.uuid+".json")
		original := readLiveDurableFile(t, pin)
		defer func() {
			if err := os.WriteFile(pin, original, 0o600); err != nil {
				t.Error(err)
			}
		}()
		removeAuthorityFile(t, pin)
		before := len(f.files[f.source.uuid].requests())
		id := f.broker.send(t, "engine:copy-model", f.request(m.manifest.Model, ""), true)
		if response := task2Response(t, f.broker, id); response.Error == nil || !strings.Contains(response.Error.Message, "not a pinned cluster peer") {
			t.Fatalf("missing source authority=%+v", response.Error)
		}
		if len(f.files[f.source.uuid].requests()) != before {
			t.Fatal("controller trust substituted for missing target/source pin")
		}
		f.assertNoReceipt(t, m)
		f.idle(t)
	})
	t.Run("source swap between listing and GET is refused", func(t *testing.T) {
		f.directoryToManagers(t, false)
		m := f.seed(t, "source-swap", 23)
		path := m.paths[m.manifest.Files[0].Path]
		observer := f.files[f.source.uuid]
		observer.mu.Lock()
		observer.fault = &controllerCopyFault{model: m.manifest.Model, beforeGET: func() error {
			// Windows retains the issued export's open handle. Mutate that
			// file in place, so the size fence rather than rename permission
			// determines whether the stale export can be read.
			return os.WriteFile(path, []byte("GGUF changed after issued listing"), 0o600)
		}}
		observer.mu.Unlock()
		defer func() { observer.mu.Lock(); observer.fault = nil; observer.mu.Unlock() }()
		id := f.broker.send(t, "engine:copy-model", f.request(m.manifest.Model, ""), true)
		if response := task2Response(t, f.broker, id); response.Error == nil || !strings.Contains(response.Error.Message, "changed") {
			t.Fatalf("source swap refusal=%+v", response.Error)
		}
		f.assertNoReceipt(t, m)
		f.idle(t)
	})
	t.Run("destination conflict preserves existing bytes", func(t *testing.T) {
		m := f.seed(t, "destination-conflict", 29)
		path := filepath.Join(controllerCopyAliasDir(f.stores[f.target.uuid], m.manifest.Model), m.manifest.ExportID, filepath.FromSlash(m.manifest.Files[0].Path))
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		original := []byte("different existing destination bytes")
		if err := os.WriteFile(path, original, 0o600); err != nil {
			t.Fatal(err)
		}
		id := f.broker.send(t, "engine:copy-model", f.request(m.manifest.Model, ""), true)
		if response := task2Response(t, f.broker, id); response.Error == nil || response.Error.Code != -32010 {
			t.Fatalf("destination conflict=%+v", response.Error)
		}
		if !bytes.Equal(readLiveDurableFile(t, path), original) {
			t.Fatal("conflict overwrote destination")
		}
		f.assertNoReceipt(t, m)
		f.idle(t)
	})
	t.Run("failed mutation acknowledgement prevents file effects", func(t *testing.T) {
		m := f.seed(t, "withdrawal-denied", 31)
		f.targetEM.mu.Lock()
		f.targetEM.mutationError = "fixture withdrawal denied"
		f.targetEM.mu.Unlock()
		defer func() { f.targetEM.mu.Lock(); f.targetEM.mutationError = ""; f.targetEM.mu.Unlock() }()
		before := len(f.files[f.source.uuid].requests())
		id := f.broker.send(t, "engine:copy-model", f.request(m.manifest.Model, ""), true)
		if response := task2Response(t, f.broker, id); response.Error == nil || !strings.Contains(response.Error.Message, "withdrawal") {
			t.Fatalf("mutation acknowledgement refusal=%+v", response.Error)
		}
		for _, seen := range f.files[f.source.uuid].requests()[before:] {
			if seen.path == "/v1/models/file" {
				t.Fatal("failed withdrawal fetched file bytes")
			}
		}
		if _, err := os.Stat(controllerCopyAliasDir(f.stores[f.target.uuid], m.manifest.Model)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("failed withdrawal wrote destination: %v", err)
		}
		f.idle(t)
	})
	t.Run("dispatched POST is not replayed on alternate reverse route", func(t *testing.T) {
		m := f.seed(t, "lost-ack", 37)
		observer := f.files[f.target.uuid]
		before := len(observer.requests())
		fault := &controllerCopyFault{loseAck: true, hit: make(chan struct{}), closed: make(chan struct{})}
		observer.mu.Lock()
		observer.fault = fault
		observer.mu.Unlock()
		defer func() {
			observer.mu.Lock()
			used := fault.used
			observer.fault = nil
			observer.mu.Unlock()
			if used {
				select {
				case <-fault.closed:
				case <-time.After(10 * time.Second):
					t.Error("lost-acknowledgement fault did not join and close upstream body")
				}
			}
		}()
		request := f.request(m.manifest.Model, "")
		id := f.broker.send(t, "engine:copy-model", request, true)
		if response := task2Response(t, f.broker, id); response.Error == nil || !strings.Contains(response.Error.Message, "outcome is unknown") {
			t.Fatalf("lost acknowledgement=%+v", response.Error)
		}
		controllerCopyTerminal(t, f.broker, request, "unknown")
		select {
		case <-fault.hit:
		case <-time.After(10 * time.Second):
			t.Fatal("copy response failed without the deliberate downstream connection drop")
		}
		select {
		case <-fault.closed:
		case <-time.After(10 * time.Second):
			t.Fatal("lost-acknowledgement fault did not close upstream body")
		}
		f.idle(t)
		posts := 0
		for _, seen := range observer.requests()[before:] {
			if seen.method == http.MethodPost && seen.path == "/v1/models/copy" {
				posts++
			}
		}
		if posts != 1 {
			t.Fatalf("dispatched copy POST count=%d, want 1 with alternate route available", posts)
		}
		observer.mu.Lock()
		observer.fault = nil
		observer.mu.Unlock()
		// Explicit retry rehashes the committed receipt, without fetching bytes.
		sourceRequests := len(f.files[f.source.uuid].requests())
		id = f.broker.send(t, "engine:copy-model", f.request(m.manifest.Model, m.manifest.ExportID), true)
		f.assertCopied(t, m, task2Response(t, f.broker, id))
		if len(f.files[f.source.uuid].requests()) != sourceRequests {
			t.Fatal("verified identity-bound retry contacted source")
		}
		f.idle(t)
	})
	f.assertController(t)
}
