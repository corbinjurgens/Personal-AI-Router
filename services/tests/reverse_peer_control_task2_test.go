// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/mdns"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

// All workers run their production mains and constructors. The public stdio
// boundary supplies broker discovery and settings/policy replies; HTTP proxies
// only observe real target handlers. No adapter or private dial is substituted.
type task2Fixture struct {
	a, b        *task1Node
	engines     map[string]*task1Process
	scanner     *task1Process
	info        *task1Process
	blockedPort int
	mu          sync.Mutex
	infoCalls   int
	unloads     int
	loaded      bool
	listeners   []int
	skipScanner bool
}

type task2FixtureOption func(*task2Fixture)

// task2WithoutScanner keeps the owned mesh and its explicit directory snapshots
// without browsing or advertising on the host's multicast network.
func task2WithoutScanner(f *task2Fixture) {
	f.skipScanner = true
}

func (f *task2Fixture) stopScanner() {
	if f.scanner != nil {
		f.scanner.stop()
	}
}

func task2Response(t *testing.T, p *task1Process, id int) jsonrpc.Message {
	t.Helper()
	var response jsonrpc.Message
	task1Eventually(t, "control response", 12*time.Second, func() bool {
		p.mu.Lock()
		defer p.mu.Unlock()
		for _, frame := range p.frames {
			if idEquals(frame.ID, id) {
				response = frame
				return true
			}
		}
		if p.readErr != nil {
			t.Fatalf("control stdout: %v; stderr: %s", p.readErr, p.output.String())
		}
		return false
	})
	return response
}

func task2Decode(t *testing.T, raw json.RawMessage, result any) {
	t.Helper()
	if err := json.Unmarshal(raw, result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
}

func (f *task2Fixture) directory(n *task1Node) noderec.DirectoryNode {
	services := map[noderec.ServiceKey]noderec.ServiceStatus{}
	for _, service := range []noderec.ServiceKey{noderec.ServiceCluster, noderec.ServiceEngineControl, noderec.ServiceEngineManager, noderec.ServiceNodeInfo} {
		port := n.ports[service]
		if observer := n.observers[service]; observer != nil {
			port = observer.port
		}
		if n == f.b {
			port = f.blockedPort
		}
		services[service] = noderec.ServiceStatus{Port: port}
	}
	return noderec.DirectoryNode{HostUUID: n.uuid, ClusterUUID: n.uuid, IP: "127.0.0.1", Name: n.uuid, Trusted: true, Services: services}
}

func newTask2Fixture(t *testing.T, options ...task2FixtureOption) *task2Fixture {
	t.Helper()
	f := &task2Fixture{engines: map[string]*task1Process{}, loaded: true}
	for _, option := range options {
		option(f)
	}
	t.Cleanup(func() {
		for _, port := range f.listeners {
			conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), time.Second)
			if conn != nil {
				if err := conn.Close(); err != nil {
					t.Error(err)
				}
			}
			if err == nil {
				t.Errorf("fixture listener remains on %d", port)
			}
		}
	})
	blocked := listenFixtureLoopback(t)
	f.blockedPort = blocked.Addr().(*net.TCPAddr).Port
	f.listeners = append(f.listeners, f.blockedPort)
	blockedDone := make(chan struct{})
	go func() {
		defer close(blockedDone)
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
		<-blockedDone
	})
	for _, uuid := range []string{"task2-reachable", "task2-restricted"} {
		base := shortClusterTempDir(t)
		dir := filepath.Join(base, "cluster")
		if err := os.MkdirAll(filepath.Join(dir, "trusted"), 0o700); err != nil {
			t.Fatal(err)
		}
		n := &task1Node{uuid: uuid, base: base, ports: map[noderec.ServiceKey]int{}, observers: map[noderec.ServiceKey]*task1Observer{}}
		n.cert = mintClusterIdentity(t, dir, uuid)
		task1WriteJSON(t, filepath.Join(dir, "identity.json"), map[string]any{"node_uuid": uuid, "created_at": 1})
		task1WriteJSON(t, filepath.Join(dir, "admission.json"), map[string]any{"counter": 1, "activated": 1, "clusterId": task1ClusterID, "epoch": 1})
		for _, service := range []noderec.ServiceKey{noderec.ServiceCluster, noderec.ServiceEngineControl, noderec.ServiceEngineManager, noderec.ServiceNodeInfo} {
			n.ports[service] = temporaryClusterPort(t)
			f.listeners = append(f.listeners, n.ports[service])
		}
		if f.a == nil {
			f.a = n
		} else {
			f.b = n
		}
	}
	nodes := []*task1Node{f.a, f.b}
	for _, n := range nodes {
		peer := f.a
		if n == f.a {
			peer = f.b
		}
		task1WriteJSON(t, filepath.Join(n.base, "cluster", "trusted", peer.uuid+".json"), map[string]any{
			"nodeUuid": peer.uuid, "nodeId": peer.uuid, "certPem": peer.cert, "clusterId": task1ClusterID, "admissionEpoch": 1,
		})
		n.mesh = clustertrust.Open(filepath.Join(n.base, "cluster"))
	}
	for _, n := range nodes {
		for _, service := range []noderec.ServiceKey{noderec.ServiceCluster, noderec.ServiceEngineControl, noderec.ServiceEngineManager} {
			n.observers[service] = newTask1Observer(t, n, service, nodes)
			f.listeners = append(f.listeners, n.observers[service].port)
		}
	}
	// Only a fixture-owned loopback API can mutate. Every bundled engine's port
	// and executable detection are overridden before either manager starts.
	fake := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		defer f.mu.Unlock()
		var value any
		switch r.URL.Path {
		case "/api/tags":
			value = map[string]any{"models": []map[string]string{{"name": "task2-model"}}}
		case "/api/ps":
			models := []map[string]string{}
			if f.loaded {
				models = append(models, map[string]string{"name": "task2-model"})
			}
			value = map[string]any{"models": models}
		case "/api/generate":
			var body struct {
				Model     string `json:"model"`
				KeepAlive *int   `json:"keep_alive"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || r.Method != http.MethodPost || body.Model != "task2-model" || body.KeepAlive == nil || *body.KeepAlive != 0 {
				t.Errorf("invalid safe unload: method=%s body=%+v err=%v", r.Method, body, err)
				http.Error(w, "invalid unload", http.StatusBadRequest)
				return
			}
			f.unloads++
			f.loaded = false
			value = map[string]any{"model": "task2-model", "done": true}
		default:
			value = map[string]string{"version": "task2-fake"}
		}
		if err := json.NewEncoder(w).Encode(value); err != nil {
			t.Errorf("fake engine response: %v", err)
		}
	}))
	t.Cleanup(fake.Close)
	fakeURL, err := url.Parse(fake.URL)
	if err != nil {
		t.Fatal(err)
	}
	fakePort, err := strconv.Atoi(fakeURL.Port())
	if err != nil {
		t.Fatal(err)
	}
	f.listeners = append(f.listeners, fakePort)
	for _, n := range nodes {
		config := n.base
		if runtime.GOOS == "darwin" {
			config = filepath.Join(config, "Library", "Application Support")
		}
		manifestDir := filepath.Join(config, "Nvidia Corporation", "Personal AI Router", "engines")
		if err := os.MkdirAll(manifestDir, 0o700); err != nil {
			t.Fatal(err)
		}
		for _, engine := range []string{"ollama", "lmstudio", "llamacpp"} {
			port := temporaryClusterPort(t)
			if engine == "ollama" {
				port = fakePort
			}
			task1WriteJSON(t, filepath.Join(manifestDir, engine+".json"), map[string]any{
				"engine":  engine,
				"runtime": map[string]any{"port": port, "ready": map[string]any{"http": "http://127.0.0.1:{port}/api/version", "status": 200}},
				"platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{
					"detect": []string{nodeInfoBin}, "runtime": map[string]any{"bin": nodeInfoBin, "port": port},
				}},
			})
		}
		var members []task1Member
		for _, peer := range nodes {
			members = append(members, task1Member{ID: peer.uuid, NodeUUID: peer.uuid, Name: peer.uuid, IPAddress: "127.0.0.1", Port: f.directory(peer).Services[noderec.ServiceCluster].Port, ClusterID: task1ClusterID, AdmissionEpoch: 1, State: "member"})
		}
		task1WriteJSON(t, filepath.Join(n.base, "cluster", "members.json"), members)
		f.engines[n.uuid] = startTask1Process(t, n.base, engineMgrBin, "--cluster-dir", filepath.Join(n.base, "cluster"), "--control-port", strconv.Itoa(n.ports[noderec.ServiceEngineControl]), "--http-port", strconv.Itoa(n.ports[noderec.ServiceEngineManager]), "--loaded-poll-interval", "0")
		completeEngineManagerRecovery(t, func(method string, params any) jsonrpc.Message {
			return f.engines[n.uuid].call(t, method, params)
		})
		// Standalone managers have no broker to publish policy. Supply its
		// actual notification shape after recovery so granted mutations reach
		// their handlers rather than the unknown-projection gate.
		policy := nodepolicy.Default()
		policy.Revision = "12"
		advertisement := policy.Advertisement()
		f.engines[n.uuid].send(t, "engine:node-projection", struct {
			Policy            nodepolicy.Advertisement `json:"policy"`
			PausedLocally     bool                     `json:"pausedLocally"`
			ManagementBlocked bool                     `json:"managementBlocked"`
		}{Policy: advertisement, PausedLocally: false, ManagementBlocked: false}, false)
		var models struct {
			Policy *nodepolicy.Advertisement `json:"policy"`
		}
		task2Decode(t, f.engines[n.uuid].call(t, "engine:models", nil).Result, &models)
		if models.Policy == nil || !reflect.DeepEqual(*models.Policy, advertisement) {
			t.Fatalf("standalone manager %s did not publish its available projection: %+v", n.uuid, models.Policy)
		}
		f.engines[n.uuid].call(t, "engine:configured-ports", nil)
		n.cm = startTask1Process(t, n.base, clusterMgrBin, "--config-dir", n.base, "--port", strconv.Itoa(n.ports[noderec.ServiceCluster]))
		n.cm.call(t, "cluster:get-node-id", nil)
		services := []noderec.RegisterParams{}
		for _, service := range []noderec.ServiceKey{noderec.ServiceCluster, noderec.ServiceEngineControl, noderec.ServiceEngineManager} {
			services = append(services, noderec.RegisterParams{Service: service, Port: n.observers[service].port})
		}
		services = append(services, noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: n.ports[noderec.ServiceNodeInfo]})
		n.cm.call(t, "cluster:set-local-services", map[string]any{"generation": 1, "services": services})
	}
	f.info = startTask1Process(t, f.b.base, nodeInfoBin, "--port", strconv.Itoa(f.b.ports[noderec.ServiceNodeInfo]), "--node-id", f.b.uuid)
	f.info.send(t, noderec.MethodSetClusterIdentity, noderec.ClusterIdentityParams{ClusterUUID: f.b.uuid}, false)
	waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(f.b.ports[noderec.ServiceNodeInfo])), 10*time.Second)
	// ni stays the production plain protocol inside the pinned outer session.
	u, err := url.Parse("http://127.0.0.1:" + strconv.Itoa(f.b.ports[noderec.ServiceNodeInfo]))
	if err != nil {
		t.Fatal(err)
	}
	proxy := httputil.NewSingleHostReverseProxy(u)
	ni := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.infoCalls++
		f.mu.Unlock()
		proxy.ServeHTTP(w, r)
	}))
	t.Cleanup(ni.Close)
	niURL, err := url.Parse(ni.URL)
	if err != nil {
		t.Fatal(err)
	}
	niPort, err := strconv.Atoi(niURL.Port())
	if err != nil {
		t.Fatal(err)
	}
	f.listeners = append(f.listeners, niPort)
	f.b.cm.call(t, "cluster:set-local-services", map[string]any{"generation": 2, "services": []noderec.RegisterParams{
		{Service: noderec.ServiceCluster, Port: f.b.observers[noderec.ServiceCluster].port},
		{Service: noderec.ServiceEngineControl, Port: f.b.observers[noderec.ServiceEngineControl].port},
		{Service: noderec.ServiceEngineManager, Port: f.b.observers[noderec.ServiceEngineManager].port},
		{Service: noderec.ServiceNodeInfo, Port: niPort},
	}})
	task1Eventually(t, "pinned outer reverse session", 15*time.Second, func() bool {
		return f.a.observers[noderec.ServiceCluster].established(f.b.uuid) > 0
	})
	if !f.skipScanner {
		f.scanner = startTask1Process(t, f.a.base, scannerBin, "--cluster-dir", filepath.Join(f.a.base, "cluster"))
		f.scanner.call(t, "discovery:get-nodes", nil)
		responder, err := mdns.NewResponder("task2-"+filepath.Base(f.b.base), noderec.ServiceType, noderec.Domain, noderec.SRVPort, []string{
			"v=1", "uuid=" + f.b.uuid, "cluster-uuid=" + f.b.uuid, "ip=127.0.0.1",
			fmt.Sprintf("ni=%d", f.blockedPort), fmt.Sprintf("em=%d", f.blockedPort), fmt.Sprintf("ec=%d", f.blockedPort),
		})
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		advertiseDone := make(chan error, 1)
		go func() { advertiseDone <- responder.Run(ctx) }()
		t.Cleanup(func() {
			cancel()
			select {
			case err := <-advertiseDone:
				if err != nil {
					t.Error(err)
				}
			case <-time.After(5 * time.Second):
				t.Error("fixture advertiser did not stop")
			}
		})
	}
	for _, n := range nodes {
		peer := f.a
		if n == f.a {
			peer = f.b
		}
		f.engines[n.uuid].send(t, noderec.NotifyNodes, noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{f.directory(peer)}}, false)
	}
	return f
}

func (f *task2Fixture) restricted(t *testing.T) noderec.DirectoryNode {
	t.Helper()
	response := f.scanner.call(t, "discovery:get-nodes", noderec.GetNodesParams{Service: noderec.ServiceEngineControl})
	var result noderec.GetNodesResult
	task2Decode(t, response.Result, &result)
	for _, node := range result.Nodes {
		if node.HostUUID == f.b.uuid {
			return node
		}
	}
	return noderec.DirectoryNode{}
}

func (f *task2Fixture) assertMembership(t *testing.T) {
	t.Helper()
	for _, n := range []*task1Node{f.a, f.b} {
		peer := f.a
		if n == f.a {
			peer = f.b
		}
		var result struct {
			Nodes []task1Member `json:"nodes"`
		}
		task2Decode(t, n.cm.call(t, "nodes:get-initial", nil).Result, &result)
		found := false
		for _, row := range result.Nodes {
			if row.NodeUUID == peer.uuid && row.ClusterID == task1ClusterID && row.AdmissionEpoch == 1 && row.State == "member" {
				found = true
			}
		}
		if !found {
			t.Fatalf("no-route changed authoritative %s membership: %+v", n.uuid, result.Nodes)
		}
	}
	if f.restricted(t).ClusterUUID != f.b.uuid {
		t.Fatal("no-route state was reported as membership removal")
	}
}

func task2Relay(t *testing.T, source, target *task1Process, method string, params any, notification, replyMethod, caller string, value any) jsonrpc.Message {
	t.Helper()
	target.mu.Lock()
	start := len(target.frames)
	target.mu.Unlock()
	id := source.send(t, method, params, true)
	var request struct {
		ID      string           `json:"id"`
		Method  string           `json:"method"`
		Caller  string           `json:"caller"`
		Request settings.Request `json:"request"`
	}
	task1Eventually(t, notification, 10*time.Second, func() bool {
		target.mu.Lock()
		defer target.mu.Unlock()
		for _, frame := range target.frames[start:] {
			if frame.Method == notification {
				task2Decode(t, frame.Params, &request)
				return true
			}
		}
		return false
	})
	if request.ID == "" || request.Caller != caller {
		t.Fatalf("authenticated relay id=%q caller=%q want=%q", request.ID, request.Caller, caller)
	}
	if notification == "engine:settings-request" && (request.Method != "get" || request.Request.Engine != "ollama" || request.Request.ExpectedRevision != 7 || request.Request.NodeID != "") {
		t.Fatalf("settings relay lost target/revision: %+v", request)
	}
	if notification == "policy:request" && request.Method != "get" {
		t.Fatalf("policy method=%q want=get", request.Method)
	}
	target.send(t, replyMethod, map[string]any{"id": request.ID, "result": value}, false)
	response := task2Response(t, source, id)
	if response.Error != nil {
		t.Fatalf("%s: %+v", method, response.Error)
	}
	return response
}

func task2Event(t *testing.T, p *task1Process, node string, sequence uint64) bool {
	t.Helper()
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, frame := range p.frames {
		if frame.Method == "engine:settings-changed" {
			var snapshot settings.Snapshot
			task2Decode(t, frame.Params, &snapshot)
			if snapshot.NodeID == node && snapshot.Engine == "ollama" && snapshot.Sequence == sequence && snapshot.Revision == 7 && snapshot.Settings.ServerPort == 45678 {
				return true
			}
		}
	}
	return false
}

func TestReversePeerControlTask2WithoutScannerReadsAuthenticatedTarget(t *testing.T) {
	f := newTask2Fixture(t, task2WithoutScanner)
	t.Cleanup(f.stopScanner)
	if f.scanner != nil {
		t.Fatal("scanner-free fixture launched a scanner")
	}
	var local, remote struct {
		Engines []struct {
			Engine  string `json:"engine"`
			Running bool   `json:"running"`
			Port    int    `json:"port"`
		} `json:"engines"`
	}
	task2Decode(t, f.engines[f.b.uuid].call(t, "engine:get-installed", nil).Result, &local)
	wantPort := 0
	for _, engine := range local.Engines {
		if engine.Engine == "ollama" && engine.Running {
			wantPort = engine.Port
		}
	}
	if wantPort <= 0 {
		t.Fatalf("target-local fixture engine missing/running=false: %+v", local)
	}
	task2Decode(t, f.engines[f.a.uuid].call(t, "engine:remote-get-installed", map[string]string{"node": f.b.uuid}).Result, &remote)
	if !reflect.DeepEqual(remote, local) {
		t.Fatalf("authenticated target inventory = %+v, want target-local %+v", remote, local)
	}
	if count := f.b.observers[noderec.ServiceEngineControl].count(f.a.uuid, http.MethodGet, "/v1/engines"); count != 1 {
		t.Fatalf("authenticated target inventory reads = %d, want 1", count)
	}
}

func TestReversePeerControlTask2(t *testing.T) {
	tRoot := t
	f := newTask2Fixture(t)
	t.Cleanup(func() {
		if t.Failed() {
			t.Logf("cluster reachable: %s; restricted: %s", f.a.cm.output.String(), f.b.cm.output.String())
		}
	})
	a, b := f.engines[f.a.uuid], f.engines[f.b.uuid]
	ec := f.b.observers[noderec.ServiceEngineControl]
	outer := f.a.observers[noderec.ServiceCluster]
	snapshot := settings.Snapshot{NodeID: f.b.uuid, Engine: "ollama", Revision: 7, AppliedRevision: 7, Sequence: 1, Epoch: "task2-epoch", Settings: settings.Config{ServerPort: 45678}, Editable: true, Phase: "applied"}
	assertEngines := func(t *testing.T, source *task1Process, target string) {
		t.Helper()
		response := source.call(t, "engine:remote-get-installed", map[string]string{"node": target})
		var result struct {
			Engines []struct {
				Engine  string `json:"engine"`
				Running bool   `json:"running"`
				Port    int    `json:"port"`
			} `json:"engines"`
		}
		task2Decode(t, response.Result, &result)
		for _, engine := range result.Engines {
			if engine.Engine == "ollama" && engine.Running && engine.Port > 0 {
				return
			}
		}
		t.Fatalf("fixture engine missing/running=false: %+v", result)
	}
	t.Run("constructor routes control and scanner through private gateway", func(t *testing.T) {
		assertEngines(t, a, f.b.uuid)
		if ec.count(f.a.uuid, http.MethodGet, "/v1/engines") != 1 {
			t.Fatal("control did not reach the authenticated target once")
		}
		task1Eventually(t, "restricted ec entry, real node info and model inventory", 35*time.Second, func() bool {
			node := f.restricted(t)
			return node.ClusterUUID == f.b.uuid && node.Trusted && node.Services[noderec.ServiceEngineControl].Port == f.blockedPort && node.CPU != nil && node.CPU.Cores > 0 && reflect.DeepEqual(node.Models, []string{"task2-model"}) && reflect.DeepEqual(node.ModelsByEngine["ollama"], []string{"task2-model"})
		})
		f.mu.Lock()
		infoCalls := f.infoCalls
		f.mu.Unlock()
		if infoCalls == 0 || f.b.observers[noderec.ServiceEngineManager].count(f.a.uuid, http.MethodGet, "/v1/models") == 0 {
			t.Fatal("scanner did not reach both production inventory handlers")
		}
		// Use the scanner's actual discovered ec entry for subsequent controls.
		a.send(t, noderec.NotifyNodes, noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{f.restricted(t)}}, false)
		assertEngines(t, a, f.b.uuid)
		// Closing the gateway owner makes its private IPC unavailable. Only
		// restoring that owner at the custom base lets the constructors recover.
		f.a.cm.stop()
		response := task2Response(t, a, a.send(t, "engine:remote-get-installed", map[string]string{"node": f.b.uuid}, true))
		if response.Error == nil {
			t.Fatal("control succeeded without constructor-derived private IPC")
		}
		f.a.cm = startTask1Process(tRoot, f.a.base, clusterMgrBin, "--config-dir", f.a.base, "--port", strconv.Itoa(f.a.ports[noderec.ServiceCluster]))
		f.a.cm.call(t, "cluster:get-node-id", nil)
		f.a.cm.call(t, "cluster:set-local-services", map[string]any{"generation": 1, "services": []noderec.RegisterParams{{Service: noderec.ServiceCluster, Port: outer.port}, {Service: noderec.ServiceEngineControl, Port: f.a.observers[noderec.ServiceEngineControl].port}}})
		task1Eventually(t, "outer reconnect after gateway restart", 15*time.Second, func() bool { return outer.established(f.b.uuid) >= 2 })
		assertEngines(t, a, f.b.uuid)
	})
	t.Run("direct opposite direction remains usable without reverse", func(t *testing.T) {
		outer.interrupt(f.b.uuid, true)
		assertEngines(t, b, f.a.uuid)
		if f.a.observers[noderec.ServiceEngineControl].count(f.b.uuid, http.MethodGet, "/v1/engines") != 1 {
			t.Fatal("direct opposite-direction control did not dispatch once")
		}
		outer.interrupt(f.b.uuid, false)
		task1Eventually(t, "outer session restored", 15*time.Second, func() bool { return outer.established(f.b.uuid) >= 3 })
	})
	t.Run("settings policy and live events retain authenticated caller", func(t *testing.T) {
		response := task2Relay(t, a, b, "engine:remote-get-settings", settings.Request{NodeID: f.b.uuid, Engine: "ollama", ExpectedRevision: 7}, "engine:settings-request", "engine:settings-reply", f.a.uuid, snapshot)
		var actual settings.Snapshot
		task2Decode(t, response.Result, &actual)
		if !reflect.DeepEqual(actual, snapshot) {
			t.Fatalf("settings=%+v want=%+v", actual, snapshot)
		}
		policy := nodepolicy.Default()
		policy.Admission.MaxResidentModels = 3
		response = task2Relay(t, a, b, "engine:remote-policy-get", map[string]string{"nodeId": f.b.uuid}, "policy:request", "policy:reply", f.a.uuid, policy)
		var actualPolicy nodepolicy.Policy
		task2Decode(t, response.Result, &actualPolicy)
		if !reflect.DeepEqual(actualPolicy, policy) {
			t.Fatalf("policy=%+v want=%+v", actualPolicy, policy)
		}
		b.send(t, "engine:settings-projection", []settings.Snapshot{snapshot}, false)
		task1Eventually(t, "settings event", 12*time.Second, func() bool { return task2Event(t, a, f.b.uuid, 1) })
	})
	t.Run("invalid inner client pin fails closed without dropping membership", func(t *testing.T) {
		// A different leaf with the same principal reaches the real inner
		// handler directly and is refused even while the outer session is live.
		wrongDir := t.TempDir()
		mintClusterIdentity(t, wrongDir, f.a.uuid)
		wrongLeaf, err := tls.LoadX509KeyPair(filepath.Join(wrongDir, "node.crt"), filepath.Join(wrongDir, "node.key"))
		if err != nil {
			t.Fatal(err)
		}
		cfg, ok := f.a.mesh.ClientTLSConfig(f.b.uuid)
		if !ok {
			t.Fatal("missing target leaf pin")
		}
		cfg.Certificates = []tls.Certificate{wrongLeaf}
		client := &http.Client{Timeout: time.Second, Transport: &http.Transport{TLSClientConfig: cfg, DisableKeepAlives: true}}
		defer client.CloseIdleConnections()
		inner, err := client.Get("https://127.0.0.1:" + strconv.Itoa(f.b.ports[noderec.ServiceEngineControl]) + "/v1/engines")
		if err != nil {
			t.Fatalf("inner pin probe: %v", err)
		}
		if err := inner.Body.Close(); err != nil {
			t.Fatal(err)
		}
		if inner.StatusCode != http.StatusForbidden {
			t.Fatalf("wrong inner leaf status=%d want=403", inner.StatusCode)
		}
		pin := filepath.Join(f.b.base, "cluster", "trusted", f.a.uuid+".json")
		original := readLiveDurableFile(t, pin)
		removeAuthorityFile(t, pin)
		t.Cleanup(func() {
			if err := os.WriteFile(pin, original, 0o600); err != nil {
				t.Error(err)
			}
		})
		before := ec.count(f.a.uuid, http.MethodGet, "/v1/engines")
		response := task2Response(t, a, a.send(t, "engine:remote-get-installed", map[string]string{"node": f.b.uuid}, true))
		if response.Error == nil || ec.count(f.a.uuid, http.MethodGet, "/v1/engines") != before {
			t.Fatal("invalid target pin dispatched/succeeded")
		}
		snapshot.Sequence = 2
		b.send(t, "engine:settings-projection", []settings.Snapshot{snapshot}, false)
		time.Sleep(1500 * time.Millisecond)
		if task2Event(t, a, f.b.uuid, 2) {
			t.Fatal("settings event crossed a revoked target pin")
		}
		f.assertMembership(t)
		if err := os.WriteFile(pin, original, 0o600); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("lost mutation reply dispatches once and never replays", func(t *testing.T) {
		task1Eventually(t, "outer ready for mutation", 15*time.Second, func() bool {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			conn, err := clustertrust.DialReverse(ctx, f.a.base, f.b.uuid, noderec.ServiceEngineControl)
			if err != nil {
				return false
			}
			if err := conn.Close(); err != nil {
				t.Fatal(err)
			}
			return true
		})
		ec.mu.Lock()
		ec.afterReply = func(peer, path string) {
			if peer == f.a.uuid && path == "/v1/models/unload" {
				outer.interrupt(f.b.uuid, true)
			}
		}
		ec.mu.Unlock()
		response := task2Response(t, a, a.send(t, "engine:remote-unload-model", map[string]string{"node": f.b.uuid, "engine": "ollama", "model": "task2-model"}, true))
		if response.Error == nil {
			t.Fatal("interrupted mutation falsely reported success")
		}
		ec.mu.Lock()
		ec.afterReply = nil
		ec.mu.Unlock()
		f.mu.Lock()
		count, loaded := f.unloads, f.loaded
		f.mu.Unlock()
		if count != 1 || loaded || ec.count(f.a.uuid, http.MethodPost, "/v1/models/unload") != 1 {
			t.Fatalf("mutation count=%d loaded=%v", count, loaded)
		}
		response = task2Response(t, a, a.send(t, "engine:remote-get-installed", map[string]string{"node": f.b.uuid}, true))
		if response.Error == nil {
			t.Fatal("read succeeded with direct and reverse unavailable")
		}
		f.assertMembership(t)
		snapshot.Sequence = 3
		b.send(t, "engine:settings-projection", []settings.Snapshot{snapshot}, false)
		time.Sleep(1500 * time.Millisecond)
		if task2Event(t, a, f.b.uuid, 3) {
			t.Fatal("event crossed disconnected session")
		}
	})
	t.Run("new session recovers reads events and scanner without mutation replay", func(t *testing.T) {
		generation := outer.established(f.b.uuid)
		outer.interrupt(f.b.uuid, false)
		task1Eventually(t, "fresh outer session", 15*time.Second, func() bool { return outer.established(f.b.uuid) > generation })
		assertEngines(t, a, f.b.uuid)
		snapshot.Sequence = 4
		b.send(t, "engine:settings-projection", []settings.Snapshot{snapshot}, false)
		task1Eventually(t, "recovered settings event", 15*time.Second, func() bool { return task2Event(t, a, f.b.uuid, 4) })
		task1Eventually(t, "scanner reports unloaded fixture inventory", 35*time.Second, func() bool {
			node := f.restricted(t)
			loaded, known := node.LoadedByEngine["ollama"]
			return node.ClusterUUID == f.b.uuid && node.Trusted && reflect.DeepEqual(node.Models, []string{"task2-model"}) && known && len(loaded) == 0
		})
		f.mu.Lock()
		count := f.unloads
		f.mu.Unlock()
		if count != 1 || ec.count(f.a.uuid, http.MethodPost, "/v1/models/unload") != 1 {
			t.Fatalf("mutation replayed after recovery: %d", count)
		}
	})
	t.Run("production processes and private listener close", func(t *testing.T) {
		f.scanner.stop()
		f.info.stop()
		a.stop()
		b.stop()
		f.b.cm.stop()
		f.a.cm.stop()
		for _, n := range []*task1Node{f.a, f.b} {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			conn, err := clustertrust.DialReverse(ctx, n.base, f.b.uuid, noderec.ServiceEngineControl)
			cancel()
			if conn != nil {
				if err := conn.Close(); err != nil {
					t.Error(err)
				}
			}
			if err == nil {
				t.Fatal("private gateway remains after owner exit")
			}
			for _, service := range []noderec.ServiceKey{noderec.ServiceCluster, noderec.ServiceEngineControl, noderec.ServiceEngineManager} {
				conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(n.ports[service])), time.Second)
				if conn != nil {
					if err := conn.Close(); err != nil {
						t.Error(err)
					}
				}
				if err == nil {
					t.Fatalf("production listener remains: %s/%s", n.uuid, service)
				}
			}
		}
	})
}
