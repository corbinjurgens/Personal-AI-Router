// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

// The serving scanner runs only before the receiving service exists. Its exact
// public self snapshot initializes the serving CM, never the receiving directory.
// Thus no remote mDNS responder exists during any bootstrap or recovery below.
func pairedBootstrapSelf(t *testing.T, remote *task1Node) noderec.DirectoryNode {
	t.Helper()
	scanner := startTask1Process(t, remote.base, scannerBin, "--cluster-dir", filepath.Join(remote.base, "cluster"))
	var self noderec.DirectoryNode
	task1Eventually(t, "serving scanner self snapshot", 10*time.Second, func() bool {
		var result noderec.GetNodesResult
		task2Decode(t, scanner.call(t, noderec.MethodGetNodes, nil).Result, &result)
		for _, node := range result.Nodes {
			if node.HostUUID == remote.uuid {
				self = node
				address := net.ParseIP(node.IP)
				return address != nil && address.IsGlobalUnicast() && !address.IsLoopback()
			}
		}
		return false
	})
	scanner.stop()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	stopped, err := serviceProcessStopped(ctx, int32(scanner.cmd.Process.Pid))
	if err != nil || !stopped {
		t.Fatalf("serving scanner remains before receiving scanner starts: stopped=%t err=%v", stopped, err)
	}
	t.Logf("serving scanner stopped before receiving service startup: pid=%d", scanner.cmd.Process.Pid)
	return self
}

// Reverse forwarding dials loopback; an ordinary peer dials the scanner's LAN
// address. Rejecting only the latter models a blocked inbound application route
// without touching the host firewall or changing service registrations.
type pairedBootstrapListener struct {
	net.Listener
	blocked  atomic.Bool
	rejected atomic.Int64
}

func (l *pairedBootstrapListener) Accept() (net.Conn, error) {
	for {
		conn, err := l.Listener.Accept()
		if err != nil {
			return nil, err
		}
		addr, ok := conn.LocalAddr().(*net.TCPAddr)
		if l.blocked.Load() && ok && !addr.IP.IsLoopback() {
			l.rejected.Add(1)
			if tcp, ok := conn.(*net.TCPConn); ok {
				_ = tcp.SetLinger(0)
			}
			_ = conn.Close()
			continue
		}
		return conn, nil
	}
}

type pairedBootstrapFixture struct {
	local              *serviceFixture
	client             *serviceRPCClient
	a, b               *task1Node
	self               noderec.DirectoryNode
	listeners          []*pairedBootstrapListener
	mu                 sync.Mutex
	counts             map[string]int
	sample             uint32
	id                 int
	wantPolicy         nodepolicy.Policy
	wantSettings       settings.Snapshot
	clusterRejects     atomic.Int64
	clusterLocatorPort int
}

func newPairedBootstrapFixture(t *testing.T, blocked bool) *pairedBootstrapFixture {
	t.Helper()
	f := &pairedBootstrapFixture{counts: map[string]int{}, sample: 61, id: 1100}
	f.local = newServiceFixture(t)
	for _, uuid := range []string{"bootstrap-receiver", "bootstrap-serving"} {
		base := shortClusterTempDir(t)
		if f.a == nil {
			base = f.local.dir
		}
		dir := filepath.Join(base, "cluster")
		if err := os.MkdirAll(filepath.Join(dir, "trusted"), 0o700); err != nil {
			t.Fatal(err)
		}
		node := &task1Node{uuid: uuid, base: base, ports: map[noderec.ServiceKey]int{}, observers: map[noderec.ServiceKey]*task1Observer{}}
		node.cert = mintClusterIdentity(t, dir, uuid)
		task1WriteJSON(t, filepath.Join(dir, "identity.json"), map[string]any{"node_uuid": uuid, "created_at": 1})
		task1WriteJSON(t, filepath.Join(dir, "admission.json"), map[string]any{"counter": 1, "activated": 1, "clusterId": task1ClusterID, "epoch": 1})
		node.ports[noderec.ServiceCluster] = temporaryClusterPort(t)
		if f.a == nil {
			f.a = node
		} else {
			f.b = node
		}
	}
	nodes := []*task1Node{f.a, f.b}
	t.Cleanup(func() {
		for _, node := range nodes {
			for _, port := range node.ports {
				conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), time.Second)
				if conn != nil {
					_ = conn.Close()
				}
				if err == nil {
					t.Errorf("owned production listener remains after teardown: node=%s port=%d", node.uuid, port)
				}
			}
		}
	})
	for _, node := range nodes {
		peer := f.a
		if node == f.a {
			peer = f.b
		}
		task1WriteJSON(t, filepath.Join(node.base, "cluster", "trusted", peer.uuid+".json"), map[string]any{"nodeUuid": peer.uuid, "nodeId": peer.uuid, "certPem": peer.cert, "clusterId": task1ClusterID, "admissionEpoch": 1})
		node.mesh = clustertrust.Open(filepath.Join(node.base, "cluster"))
	}
	f.self = pairedBootstrapSelf(t, f.b)
	for _, node := range nodes {
		node.observers[noderec.ServiceCluster] = newTask1Observer(t, node, noderec.ServiceCluster, nodes)
		observer := node.observers[noderec.ServiceCluster]
		forward := observer.server.Handler
		// Keep this fixture's explicit membership locator stable. A roster from
		// another same-machine process would otherwise advertise its real,
		// reachable loopback listener and undo the blocked-network condition.
		// Pair authority was already established durably on both sides above.
		observer.server.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == task1RosterPath {
				http.Error(w, "fixture roster exchange unavailable", http.StatusServiceUnavailable)
				return
			}
			forward.ServeHTTP(w, r)
		})
		if !blocked {
			observer.interrupt(f.peer(node).uuid, true)
		}
	}
	f.wantPolicy = nodepolicy.Default()
	f.wantPolicy.Admission.MaxConcurrentPerModel = 17
	f.wantSettings = settings.Snapshot{NodeID: f.b.uuid, Engine: "ollama", Revision: 43, AppliedRevision: 43, Sequence: 9, Epoch: "bootstrap-settings", Settings: settings.Config{ServerPort: 45679}, Editable: true, Phase: "applied"}
	f.serve(t, noderec.ServiceNodeInfo, blocked, false)
	f.serve(t, noderec.ServiceEngineControl, blocked, true)
	blockedClusterPort := 0
	if blocked {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		blockedClusterPort = listener.Addr().(*net.TCPAddr).Port
		done := make(chan struct{})
		go func() {
			defer close(done)
			for {
				conn, err := listener.Accept()
				if err != nil {
					return
				}
				f.clusterRejects.Add(1)
				_ = conn.Close()
			}
		}()
		t.Cleanup(func() {
			if err := listener.Close(); err != nil {
				t.Error(err)
			}
			<-done
		})
	}
	f.clusterLocatorPort = f.b.observers[noderec.ServiceCluster].port
	if blocked {
		f.clusterLocatorPort = blockedClusterPort
	}
	for _, node := range nodes {
		var members []task1Member
		for _, peer := range nodes {
			port := peer.observers[noderec.ServiceCluster].port
			if blocked && peer == f.b {
				port = blockedClusterPort
			}
			members = append(members, task1Member{ID: peer.uuid, NodeUUID: peer.uuid, Name: peer.uuid, IPAddress: "127.0.0.1", Port: port, ClusterID: task1ClusterID, AdmissionEpoch: 1, State: "member"})
		}
		task1WriteJSON(t, filepath.Join(node.base, "cluster", "members.json"), members)
	}
	f.b.cm = startTask1Process(t, f.b.base, clusterMgrBin, "--config-dir", f.b.base, "--port", strconv.Itoa(f.b.ports[noderec.ServiceCluster]))
	f.b.cm.call(t, "cluster:get-node-id", nil)
	f.b.cm.call(t, "cluster:set-local-services", map[string]any{"generation": 1, "services": []noderec.RegisterParams{
		{Service: noderec.ServiceCluster, Port: f.b.observers[noderec.ServiceCluster].port},
		{Service: noderec.ServiceNodeInfo, Port: f.b.ports[noderec.ServiceNodeInfo]},
		{Service: noderec.ServiceEngineControl, Port: f.b.ports[noderec.ServiceEngineControl]},
	}})
	localCM, localEngine := f.local.copyBinary(clusterMgrBin), f.local.copyBinary(engineMgrBin)
	bridge, env := newPortInteropBridge(t, localCM, f.a.ports[noderec.ServiceCluster])
	f.local.env = append(f.local.env, env...)
	for _, path := range []string{bridge, localCM, localEngine} {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		f.local.inventory.paths = append(f.local.inventory.paths, canonical)
	}
	// No local installed engine or engine API participates in this test.
	config := f.local.configDir
	if runtime.GOOS == "darwin" {
		config = filepath.Join(config, "Library", "Application Support")
	}
	for _, engine := range []string{"ollama", "lmstudio", "llamacpp"} {
		manifestDir := filepath.Join(config, "Nvidia Corporation", "Personal AI Router", "engines")
		if err := os.MkdirAll(manifestDir, 0o700); err != nil {
			t.Fatal(err)
		}
		task1WriteJSON(t, filepath.Join(manifestDir, engine+".json"), map[string]any{
			"engine":    engine,
			"runtime":   map[string]any{"port": temporaryClusterPort(t)},
			"platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{"detect": []string{filepath.Join(f.local.dir, "absent-engine")}}},
		})
	}
	f.a.ports[noderec.ServiceEngineManager] = temporaryClusterPort(t)
	f.a.ports[noderec.ServiceEngineControl] = temporaryClusterPort(t)
	args := append(f.local.args(), "--cluster-manager-path", bridge, "--engine-manager-path", localEngine,
		"--engine-manager-http-port", strconv.Itoa(f.a.ports[noderec.ServiceEngineManager]),
		"--engine-manager-control-port", strconv.Itoa(f.a.ports[noderec.ServiceEngineControl]))
	f.local.service = f.local.startChild(args...)
	waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(f.a.ports[noderec.ServiceCluster])), 10*time.Second)
	f.client = f.local.dial()
	f.client.notification(t, "app:ready")
	servicePolicy(t, f.client)
	owned := f.local.scan()
	for _, path := range []string{f.local.servicePath, f.local.brokerPath, f.local.scannerPath, localCM, localEngine} {
		found := false
		for _, process := range owned {
			if process.Path == path {
				found = true
			}
		}
		if !found {
			t.Fatalf("production bootstrap fixture missing %s: %+v", filepath.Base(path), owned)
		}
	}
	t.Cleanup(func() {
		if t.Failed() {
			t.Logf("receiving service: %s; serving CM: %s", f.local.service.output.String(), f.b.cm.output.String())
		}
	})
	if f.hasPeer(t) {
		t.Fatal("remote peer appeared before authenticated self-description was initialized")
	}
	if blocked {
		task1Eventually(t, "real reverse session before bootstrap", 15*time.Second, func() bool {
			return f.a.observers[noderec.ServiceCluster].established(f.b.uuid) > 0
		})
	}
	return f
}

func (f *pairedBootstrapFixture) peer(node *task1Node) *task1Node {
	if node == f.a {
		return f.b
	}
	return f.a
}

func (f *pairedBootstrapFixture) serve(t *testing.T, service noderec.ServiceKey, blocked, protected bool) {
	t.Helper()
	ln, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatal(err)
	}
	listener := &pairedBootstrapListener{Listener: ln}
	listener.blocked.Store(blocked)
	f.listeners = append(f.listeners, listener)
	f.b.ports[service] = ln.Addr().(*net.TCPAddr).Port
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if protected {
			f.b.mesh.Refresh()
			if peer, ok := f.b.mesh.VerifyClientPin(r); !ok || peer != f.a.uuid {
				http.Error(w, "not the admitted caller", http.StatusForbidden)
				return
			}
		}
		f.mu.Lock()
		defer f.mu.Unlock()
		var value any
		switch r.Method + " " + r.URL.Path {
		case "GET /v1/node-info":
			principal := f.b.uuid
			value = phase04NodeInfoSample{HostUUID: f.b.uuid, ClusterUUID: &principal, TelemetryValid: true, MSSince: 100, CPU: &noderec.CPUInfo{Name: "bootstrap CPU", Cores: 7}, GPUs: []noderec.GPUInfo{{Name: "bootstrap GPU", UtilizationPercent: f.sample}}}
		case "GET /v1/engines":
			value = map[string]any{"engines": []map[string]any{{"engine": "ollama", "installed": true, "running": true, "port": 45679}}}
		case "POST /v1/node-policy/get":
			value = map[string]any{"policy": f.wantPolicy, "availability": nodepolicy.Available}
		case "POST /v1/engine-settings/get":
			var request settings.Request
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil || request.Engine != "ollama" || request.NodeID != "" {
				t.Errorf("settings target request: %+v err=%v", request, err)
				http.Error(w, "invalid settings target", http.StatusBadRequest)
				return
			}
			value = f.wantSettings
		case "GET /v1/engine-settings/events":
			// The production settings subscriber is independent of explicit reads.
			return
		default:
			t.Errorf("unexpected target handler: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected endpoint", http.StatusNotFound)
			return
		}
		f.counts[r.Method+" "+r.URL.Path]++
		if err := json.NewEncoder(w).Encode(value); err != nil {
			t.Errorf("target response: %v", err)
		}
	})}
	var serving net.Listener = listener
	if protected {
		serving = tls.NewListener(listener, f.b.mesh.ServerTLSConfig())
	}
	done := make(chan error, 1)
	go func() { done <- server.Serve(serving) }()
	t.Cleanup(func() {
		if err := server.Close(); err != nil {
			t.Error(err)
		}
		if err := <-done; err != http.ErrServerClosed {
			t.Errorf("target listener exit: %v", err)
		}
		conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(f.b.ports[service])), time.Second)
		if conn != nil {
			_ = conn.Close()
		}
		if err == nil {
			t.Error("target listener remains after cleanup")
		}
	})
}

func (f *pairedBootstrapFixture) call(t *testing.T, method string, params any) *jsonrpc.Message {
	t.Helper()
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	f.id++
	response, err := f.client.call(strconv.Itoa(f.id), method, raw, 8*time.Second)
	if err != nil {
		t.Fatalf("%s through production service: %v", method, err)
	}
	return response
}

func (f *pairedBootstrapFixture) hasPeer(t *testing.T) bool {
	t.Helper()
	response := f.call(t, "discovery:get-nodes", nil)
	if response.Error != nil {
		t.Fatalf("public discovery: %+v", response.Error)
	}
	var result struct {
		Nodes []struct {
			HostUUID string `json:"hostUuid"`
			Trusted  bool   `json:"trusted"`
		} `json:"nodes"`
	}
	task2Decode(t, response.Result, &result)
	found := 0
	for _, node := range result.Nodes {
		if node.HostUUID == f.b.uuid {
			if !node.Trusted {
				t.Fatal("authenticated bootstrap produced an untrusted peer")
			}
			found++
		}
	}
	if found > 1 {
		t.Fatalf("public discovery duplicated the peer: %d", found)
	}
	return found == 1
}

func (f *pairedBootstrapFixture) bootstrap(t *testing.T) {
	t.Helper()
	// This is solely the serving process's own scanner-produced row. No test
	// input installs a remote row into the receiving broker/scanner/subscribers.
	f.b.cm.send(t, noderec.NotifyNodes, noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{f.self}}, false)
	task1Eventually(t, "authenticated peer appears in public discovery without mDNS", 20*time.Second, func() bool { return f.hasPeer(t) })
	// Independently inspect the authenticated serving endpoint's complete map;
	// the public broker projection omits service keys. The public reads below
	// then prove its ni/ec ports reached the ordinary scanner subscribers.
	address := net.JoinHostPort("127.0.0.1", strconv.Itoa(f.clusterLocatorPort))
	adapter := clustertrust.NewPeerServiceClient(f.a.mesh, clustertrust.PeerServiceClientOptions{
		BaseConfigDir:     f.a.base,
		PeerClientOptions: clustertrust.PeerClientOptions{Timeout: 5 * time.Second},
		ResolveRoute: func(uuid string, service noderec.ServiceKey) (clustertrust.PeerServiceRoute, bool) {
			f.a.mesh.Refresh()
			incarnation, ok := f.a.mesh.PeerServiceIncarnation(uuid)
			return clustertrust.PeerServiceRoute{Incarnation: incarnation, Address: address, Protocol: clustertrust.ReverseTLS, Generation: 1}, ok && uuid == f.b.uuid && service == noderec.ServiceCluster
		},
	})
	response, err := adapter.Client(f.b.uuid, noderec.ServiceCluster).Get("https://" + address + "/v1/cluster/services")
	if err != nil {
		t.Fatalf("authenticated complete service descriptor: %v", err)
	}
	descriptor, decodeErr := noderec.DecodePeerServices(response.Body, f.b.uuid)
	closeErr := response.Body.Close()
	if decodeErr != nil || closeErr != nil || response.StatusCode != http.StatusOK {
		t.Fatalf("authenticated descriptor status=%d decode=%v close=%v", response.StatusCode, decodeErr, closeErr)
	}
	wantServices := []noderec.PeerService{
		{Service: noderec.ServiceCluster, Port: f.b.observers[noderec.ServiceCluster].port},
		{Service: noderec.ServiceEngineControl, Port: f.b.ports[noderec.ServiceEngineControl]},
		{Service: noderec.ServiceNodeInfo, Port: f.b.ports[noderec.ServiceNodeInfo]},
	}
	if descriptor.HostUUID != f.b.uuid || descriptor.Name != f.self.Name || !reflect.DeepEqual(descriptor.Addresses, f.self.CandidateIPs()) || !reflect.DeepEqual(descriptor.Services, wantServices) {
		t.Fatalf("complete authenticated advertised identity/ports differ: %+v", descriptor)
	}
}

func (f *pairedBootstrapFixture) count(path string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.counts[path]
}

func (f *pairedBootstrapFixture) assertReads(t *testing.T) {
	t.Helper()
	test := func(name, method, path string, params any, check func(*testing.T, json.RawMessage)) {
		t.Run(name, func(t *testing.T) {
			before := f.count(path)
			response := f.call(t, method, params)
			if response.Error != nil {
				t.Fatalf("%s failed: %+v", method, response.Error)
			}
			check(t, response.Result)
			if after := f.count(path); after <= before {
				t.Fatalf("%s returned without fresh intended target dispatch: before=%d after=%d", method, before, after)
			}
		})
	}
	test("fresh node information", "discovery:get-node-info", "GET /v1/node-info", map[string]string{"hostUuid": f.b.uuid}, func(t *testing.T, raw json.RawMessage) {
		var got phase04NodeInfoSample
		task2Decode(t, raw, &got)
		f.mu.Lock()
		wantSample := f.sample
		f.mu.Unlock()
		if got.HostUUID != f.b.uuid || got.ClusterUUID == nil || *got.ClusterUUID != f.b.uuid || got.CPU == nil || got.CPU.Cores != 7 || len(got.GPUs) != 1 || got.GPUs[0].UtilizationPercent != wantSample {
			t.Fatalf("fresh node sample differs: %+v", got)
		}
	})
	test("engine inventory", "engine:remote-get-installed", "GET /v1/engines", map[string]string{"node": f.b.uuid}, func(t *testing.T, raw json.RawMessage) {
		var got struct {
			Engines []struct {
				Engine    string `json:"engine"`
				Installed bool   `json:"installed"`
				Running   bool   `json:"running"`
				Port      int    `json:"port"`
			} `json:"engines"`
		}
		task2Decode(t, raw, &got)
		if len(got.Engines) != 1 || got.Engines[0].Engine != "ollama" || !got.Engines[0].Installed || !got.Engines[0].Running || got.Engines[0].Port != 45679 {
			t.Fatalf("fresh intended engine inventory differs: %+v", got)
		}
	})
	test("node policy", "policy:get", "POST /v1/node-policy/get", map[string]string{"nodeId": f.b.uuid}, func(t *testing.T, raw json.RawMessage) {
		var got struct {
			Policy       nodepolicy.Policy       `json:"policy"`
			Availability nodepolicy.Availability `json:"availability"`
		}
		task2Decode(t, raw, &got)
		if !reflect.DeepEqual(got.Policy, f.wantPolicy) || got.Availability != nodepolicy.Available {
			t.Fatalf("fresh intended policy differs: %+v", got)
		}
	})
	test("engine settings", "engine:get-settings", "POST /v1/engine-settings/get", map[string]string{"nodeId": f.b.uuid, "engine": "ollama"}, func(t *testing.T, raw json.RawMessage) {
		var got settings.Snapshot
		task2Decode(t, raw, &got)
		if !reflect.DeepEqual(got, f.wantSettings) {
			t.Fatalf("fresh intended settings differ: %+v", got)
		}
	})
}

func TestPairedDirectoryBootstrap(t *testing.T) {
	t.Run("authenticated direct bootstrap without a reverse session", func(t *testing.T) {
		f := newPairedBootstrapFixture(t, false)
		f.bootstrap(t)
		f.assertReads(t)
		if f.a.observers[noderec.ServiceCluster].established(f.b.uuid) != 0 || f.b.observers[noderec.ServiceCluster].established(f.a.uuid) != 0 {
			t.Fatal("direct bootstrap unexpectedly had an available reverse session")
		}
		if f.b.observers[noderec.ServiceCluster].count(f.a.uuid, http.MethodGet, "/v1/cluster/services") == 0 {
			t.Fatal("direct bootstrap never reached authenticated descriptor handler")
		}
	})
	t.Run("blocked direct bootstrap and recovery use a real reverse session", func(t *testing.T) {
		f := newPairedBootstrapFixture(t, true)
		f.bootstrap(t)
		f.assertReads(t)
		if f.clusterRejects.Load() == 0 {
			t.Fatal("bootstrap never attempted the blocked direct cluster locator")
		}
		for _, listener := range f.listeners {
			if listener.rejected.Load() == 0 {
				t.Fatal("protected reads did not independently attempt the blocked direct service route")
			}
		}
		t.Run("both routes unavailable preserve authority and restoration recovers", func(t *testing.T) {
			type memberAuthority struct {
				Cluster string
				Epoch   uint64
				State   string
			}
			membership := func(node *task1Node) map[string]memberAuthority {
				t.Helper()
				data := readLiveDurableFile(t, filepath.Join(node.base, "cluster", "members.json"))
				var members []task1Member
				task2Decode(t, data, &members)
				result := map[string]memberAuthority{}
				for _, member := range members {
					result[member.NodeUUID] = memberAuthority{member.ClusterID, member.AdmissionEpoch, member.State}
				}
				return result
			}
			wantMembership := map[string]memberAuthority{
				f.a.uuid: {task1ClusterID, 1, "member"},
				f.b.uuid: {task1ClusterID, 1, "member"},
			}
			for _, node := range []*task1Node{f.a, f.b} {
				if got := membership(node); !reflect.DeepEqual(got, wantMembership) {
					t.Fatalf("%s outage fixture lacks exact current mutual admission: %+v", node.uuid, got)
				}
			}
			paths := []string{
				filepath.Join(f.a.base, "cluster", "admission.json"), filepath.Join(f.a.base, "cluster", "trusted", f.b.uuid+".json"),
				filepath.Join(f.b.base, "cluster", "admission.json"), filepath.Join(f.b.base, "cluster", "trusted", f.a.uuid+".json"),
			}
			before := map[string][]byte{}
			for _, path := range paths {
				data := readLiveDurableFile(t, path)
				before[path] = data
			}
			outer := f.a.observers[noderec.ServiceCluster]
			opposite := f.b.observers[noderec.ServiceCluster]
			previous := outer.established(f.b.uuid)
			outer.interrupt(f.b.uuid, true)
			opposite.interrupt(f.a.uuid, true)
			t.Cleanup(func() {
				outer.interrupt(f.b.uuid, false)
				opposite.interrupt(f.a.uuid, false)
			})
			start := time.Now()
			dispatches := f.count("GET /v1/engines")
			response := f.call(t, "engine:remote-get-installed", map[string]string{"node": f.b.uuid})
			if response.Error == nil || len(response.Result) != 0 {
				t.Fatalf("unavailable routes returned success: %+v", response)
			}
			if elapsed := time.Since(start); elapsed > 7*time.Second {
				t.Fatalf("both-route failure exceeded bound: %s", elapsed)
			}
			if after := f.count("GET /v1/engines"); after != dispatches {
				t.Fatalf("both-route outage reached target handler: before=%d after=%d", dispatches, after)
			}
			task1Eventually(t, "unrefreshed authenticated claim expires during outage", 40*time.Second, func() bool { return !f.hasPeer(t) })
			for _, node := range []*task1Node{f.a, f.b} {
				if got := membership(node); !reflect.DeepEqual(got, wantMembership) {
					t.Fatalf("transport outage changed %s membership authority: %+v", node.uuid, got)
				}
			}
			for _, path := range paths {
				data := readLiveDurableFile(t, path)
				if !reflect.DeepEqual(data, before[path]) {
					t.Fatalf("transport outage altered durable authority: %s", filepath.Base(path))
				}
			}
			f.mu.Lock()
			f.sample = 84
			f.wantSettings.Revision = 44
			f.wantSettings.AppliedRevision = 44
			f.mu.Unlock()
			outer.interrupt(f.b.uuid, false)
			opposite.interrupt(f.a.uuid, false)
			task1Eventually(t, "restored reverse route without process restart", 15*time.Second, func() bool { return outer.established(f.b.uuid) > previous })
			task1Eventually(t, "authenticated bootstrap recovers after claim expiry", 20*time.Second, func() bool { return f.hasPeer(t) })
			f.assertReads(t)
		})
		assertAuthority := func(name string, replace func(*testing.T, string)) {
			t.Run(name, func(t *testing.T) {
				path := filepath.Join(f.a.base, "cluster", "trusted", f.b.uuid+".json")
				original := readLiveDurableFile(t, path)
				t.Cleanup(func() {
					if err := os.WriteFile(path, original, 0o600); err != nil {
						t.Error(err)
					}
				})
				replace(t, path)
				for _, call := range []struct {
					method, path string
					params       any
				}{
					{"engine:remote-get-installed", "GET /v1/engines", map[string]string{"node": f.b.uuid}},
					{"policy:get", "POST /v1/node-policy/get", map[string]string{"nodeId": f.b.uuid}},
					{"engine:get-settings", "POST /v1/engine-settings/get", map[string]string{"nodeId": f.b.uuid, "engine": "ollama"}},
				} {
					before := f.count(call.path)
					response := f.call(t, call.method, call.params)
					if response.Error == nil || len(response.Result) != 0 {
						t.Fatalf("%s accepted replaced authority: %+v", call.method, response)
					}
					if after := f.count(call.path); after != before {
						t.Fatalf("%s dispatched after authority loss: before=%d after=%d", call.method, before, after)
					}
				}
				if err := os.WriteFile(path, original, 0o600); err != nil {
					t.Fatal(err)
				}
				task1Eventually(t, "restored authenticated peer authority", 20*time.Second, func() bool {
					return f.call(t, "engine:remote-get-installed", map[string]string{"node": f.b.uuid}).Error == nil
				})
			})
		}
		assertAuthority("removed pin rejects before protected dispatch", func(t *testing.T, path string) { removeAuthorityFile(t, path) })
		assertAuthority("wrong pinned certificate rejects before protected dispatch", func(t *testing.T, path string) {
			task1WriteJSON(t, path, map[string]any{"nodeUuid": f.b.uuid, "nodeId": f.b.uuid, "certPem": f.a.cert, "clusterId": task1ClusterID, "admissionEpoch": 1})
		})
		assertAuthority("replaced admission rejects old authority before protected dispatch", func(t *testing.T, path string) {
			task1WriteJSON(t, path, map[string]any{"nodeUuid": f.b.uuid, "nodeId": f.b.uuid, "certPem": f.b.cert, "clusterId": task1ClusterID, "admissionEpoch": 2})
		})
	})
}
