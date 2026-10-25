// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/mdns"
	"nvpair-shared/noderec"
)

// Reuse the Phase 03 node/process and authenticated observation boundaries,
// without starting extra scanners, engines or hardware samplers.
func newPhase04NodeInfoPeers(t *testing.T) (*task2Fixture, *atomic.Int64) {
	t.Helper()
	f := &task2Fixture{}
	resets := &atomic.Int64{}
	blocked, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	f.blockedPort = blocked.Addr().(*net.TCPAddr).Port
	var connections sync.WaitGroup
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			conn, err := blocked.Accept()
			if err != nil {
				return
			}
			connections.Add(1)
			go func() {
				defer connections.Done()
				defer conn.Close()
				if err := conn.SetReadDeadline(time.Now().Add(2 * time.Second)); err != nil {
					t.Error(err)
					return
				}
				reader := bufio.NewReader(io.LimitReader(conn, 4096))
				first, err := reader.Peek(1)
				if err != nil {
					return // TCP-only liveness probes have no HTTP first data.
				}
				if tcp, ok := conn.(*net.TCPConn); ok {
					if err := tcp.SetLinger(0); err != nil {
						t.Error(err)
					}
				}
				if first[0] != 'G' {
					return // Cluster-manager TLS/pairing traffic is not a ni GET.
				}
				request, err := http.ReadRequest(reader)
				if err != nil {
					t.Errorf("restricted HTTP request: %v", err)
					return
				}
				if err := request.Body.Close(); err != nil {
					t.Error(err)
				}
				// On this advertised loopback address, scanner background reads
				// use its ordinary keep-alive HTTP client. The admitted getter's
				// PeerServiceClient uses a fresh Connection: close request. Count
				// only that exact bodyless ni request, never unrelated traffic.
				if request.Method == http.MethodGet && request.URL.Path == "/v1/node-info" && request.URL.RawQuery == "" && request.ContentLength == 0 && request.Close {
					resets.Add(1)
				}
			}()
		}
	}()
	t.Cleanup(func() {
		if err := blocked.Close(); err != nil {
			t.Error(err)
		}
		<-done
		connections.Wait()
	})
	for _, uuid := range []string{"phase04-ni-reachable", "phase04-ni-restricted"} {
		base := shortClusterTempDir(t)
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
	for _, node := range nodes {
		peer := f.a
		if node == f.a {
			peer = f.b
		}
		task1WriteJSON(t, filepath.Join(node.base, "cluster", "trusted", peer.uuid+".json"), map[string]any{"nodeUuid": peer.uuid, "nodeId": peer.uuid, "certPem": peer.cert, "clusterId": task1ClusterID, "admissionEpoch": 1})
		node.mesh = clustertrust.Open(filepath.Join(node.base, "cluster"))
	}
	for _, node := range nodes {
		node.observers[noderec.ServiceCluster] = newTask1Observer(t, node, noderec.ServiceCluster, nodes)
	}
	for _, node := range nodes {
		members := []task1Member{}
		for _, peer := range nodes {
			port := peer.observers[noderec.ServiceCluster].port
			if peer == f.b {
				port = f.blockedPort
			}
			members = append(members, task1Member{ID: peer.uuid, NodeUUID: peer.uuid, Name: peer.uuid, IPAddress: "127.0.0.1", Port: port, ClusterID: task1ClusterID, AdmissionEpoch: 1, State: "member"})
		}
		task1WriteJSON(t, filepath.Join(node.base, "cluster", "members.json"), members)
		node.cm = startTask1Process(t, node.base, clusterMgrBin, "--config-dir", node.base, "--port", strconv.Itoa(node.ports[noderec.ServiceCluster]))
		node.cm.call(t, "cluster:get-node-id", nil)
		node.cm.call(t, "cluster:set-local-services", map[string]any{"generation": 1, "services": []noderec.RegisterParams{{Service: noderec.ServiceCluster, Port: node.observers[noderec.ServiceCluster].port}}})
	}
	task1Eventually(t, "pinned reverse session", 15*time.Second, func() bool { return f.a.observers[noderec.ServiceCluster].established(f.b.uuid) > 0 })
	f.namespace = newFixtureMDNSNamespace(t)
	responder, err := mdns.NewResponder("phase04-ni-"+filepath.Base(f.b.base), f.namespace.Service, f.namespace.Domain, noderec.SRVPort, []string{"v=1", "uuid=" + phase04NodeInfoHost, "cluster-uuid=" + f.b.uuid, "ip=127.0.0.1", fmt.Sprintf("ni=%d", f.blockedPort)})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	advertising := make(chan error, 1)
	go func() { advertising <- responder.Run(ctx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case err := <-advertising:
			if err != nil {
				t.Error(err)
			}
		case <-time.After(5 * time.Second):
			t.Error("node-info advertiser did not stop")
		}
	})
	return f, resets
}

type phase04NodeInfoSample struct {
	GPUs           []noderec.GPUInfo   `json:"GPUs"`
	CPU            *noderec.CPUInfo    `json:"cpu"`
	Memory         *noderec.MemoryInfo `json:"memory"`
	TelemetryValid bool                `json:"telemetryValid"`
	MSSince        int64               `json:"msSince"`
	HostUUID       string              `json:"hostUuid"`
	ClusterUUID    *string             `json:"clusterUuid"`
}

const phase04NodeInfoHost = "phase04-ni-physical-host"

// This acceptance uses the production service socket, broker and scanner. The
// Phase 03 fixture supplies real admitted cluster managers and a pinned reverse
// session; only the remote hardware sample is deterministic test data.
func TestReverseNodeInfoPhase04(t *testing.T) {
	f, resets := newPhase04NodeInfoPeers(t)
	var mu sync.Mutex
	host, principal := phase04NodeInfoHost, f.b.uuid
	utilization := uint32(73)
	stall := false
	requests := 0
	entered := make(chan struct{}, 32)
	info := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/node-info" {
			t.Errorf("node-info request: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected node-info request", http.StatusBadRequest)
			return
		}
		mu.Lock()
		requests++
		wait := stall
		sample := phase04NodeInfoSample{
			GPUs:           []noderec.GPUInfo{{Name: "restricted GPU", VramBytes: 8192, VramUsedBytes: 2048, UtilizationPercent: utilization}},
			CPU:            &noderec.CPUInfo{Name: "restricted CPU", Cores: 8, UtilizationPercent: 42},
			Memory:         &noderec.MemoryInfo{TotalBytes: 16384, UsedBytes: 4096},
			TelemetryValid: true, MSSince: 100, HostUUID: host,
		}
		p := principal
		sample.ClusterUUID = &p
		mu.Unlock()
		if wait {
			select {
			case entered <- struct{}{}:
			default:
			}
			<-r.Context().Done()
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(sample); err != nil {
			t.Errorf("node-info response: %v", err)
		}
	}))
	t.Cleanup(info.Close)
	u, err := url.Parse(info.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	f.b.cm.call(t, "cluster:set-local-services", map[string]any{"generation": 3, "services": []noderec.RegisterParams{
		{Service: noderec.ServiceCluster, Port: f.b.observers[noderec.ServiceCluster].port},
		{Service: noderec.ServiceNodeInfo, Port: port},
	}})

	local := newServiceFixture(t)
	local.clusterDir = filepath.Join(f.a.base, "cluster")
	local.service = local.startChild(append(local.args(), fixtureMDNSBrokerArgs(f.namespace)...)...)
	client := local.readyClient()
	second := local.dial()
	nextID := 200
	call := func(t *testing.T, c *serviceRPCClient, params json.RawMessage) *jsonrpc.Message {
		t.Helper()
		nextID++
		response, err := c.call(strconv.Itoa(nextID), "discovery:get-node-info", params, 4*time.Second)
		if err != nil {
			t.Fatalf("local node-info RPC: %v", err)
		}
		return response
	}
	params, err := json.Marshal(map[string]string{"hostUuid": phase04NodeInfoHost})
	if err != nil {
		t.Fatal(err)
	}
	assertSample := func(t *testing.T, response *jsonrpc.Message, wantUtilization uint32) {
		t.Helper()
		if response.Error != nil {
			t.Fatalf("node-info RPC: %+v", response.Error)
		}
		var got phase04NodeInfoSample
		task2Decode(t, response.Result, &got)
		wantPrincipal := f.b.uuid
		want := phase04NodeInfoSample{
			GPUs:           []noderec.GPUInfo{{Name: "restricted GPU", VramBytes: 8192, VramUsedBytes: 2048, UtilizationPercent: wantUtilization}},
			CPU:            &noderec.CPUInfo{Name: "restricted CPU", Cores: 8, UtilizationPercent: 42},
			Memory:         &noderec.MemoryInfo{TotalBytes: 16384, UsedBytes: 4096},
			TelemetryValid: true, MSSince: 100, HostUUID: phase04NodeInfoHost, ClusterUUID: &wantPrincipal,
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("fresh full sample = %+v, want %+v", got, want)
		}
	}
	assertUnavailable := func(t *testing.T, response *jsonrpc.Message) {
		t.Helper()
		if response.Error == nil || response.Error.Code != -32000 || response.Error.Message != "node information unavailable" || len(response.Result) != 0 {
			t.Fatalf("unavailable must reject without a stale result: %+v", response)
		}
	}
	task1Eventually(t, "broker scanner discovers restricted identity", 35*time.Second, func() bool {
		nextID++
		response := client.mustCall(t, strconv.Itoa(nextID), "discovery:get-nodes", nil)
		var snapshot struct {
			Nodes []struct {
				HostUUID string `json:"hostUuid"`
			} `json:"nodes"`
		}
		task2Decode(t, response.Result, &snapshot)
		for _, node := range snapshot.Nodes {
			if node.HostUUID == phase04NodeInfoHost {
				return true
			}
		}
		return false
	})

	t.Run("direct first-data reset falls back to fresh pinned reverse sample", func(t *testing.T) {
		beforeGetter := resets.Load()
		if beforeGetter != 0 {
			t.Fatalf("unexpected admitted getter before first service RPC: %d", beforeGetter)
		}
		assertSample(t, call(t, client, params), 73)
		afterGetter := resets.Load()
		if afterGetter-beforeGetter != 1 {
			t.Fatalf("first service getter direct GET reset delta = %d, want 1 (before=%d after=%d)", afterGetter-beforeGetter, beforeGetter, afterGetter)
		}
		t.Logf("first service getter direct GET reset delta=%d before any standalone probe", afterGetter-beforeGetter)
		direct := &http.Client{Timeout: time.Second}
		response, err := direct.Get("http://127.0.0.1:" + strconv.Itoa(f.blockedPort) + "/v1/node-info")
		if response != nil {
			if closeErr := response.Body.Close(); closeErr != nil {
				t.Error(closeErr)
			}
		}
		if err == nil {
			t.Fatal("restricted direct endpoint unexpectedly answered")
		}
		if resets.Load() != afterGetter {
			t.Fatal("standalone ordinary HTTP probe contaminated the admitted getter reset counter")
		}
		mu.Lock()
		utilization = 81
		before := requests
		mu.Unlock()
		assertSample(t, call(t, client, params), 81)
		mu.Lock()
		after := requests
		mu.Unlock()
		if after <= before {
			t.Fatal("getter reused enrichment instead of fetching a new sample")
		}
	})

	t.Run("concurrent clients with the same request ID keep success and error correlated", func(t *testing.T) {
		if err := client.request("900", "discovery:get-node-info", params); err != nil {
			t.Fatal(err)
		}
		if err := second.request("900", "discovery:get-node-info", json.RawMessage(`{"hostUuid":"missing-host"}`)); err != nil {
			t.Fatal(err)
		}
		if err := client.request("901", "discovery:get-node-info", params); err != nil {
			t.Fatal(err)
		}
		missing, err := second.response("900", 4*time.Second)
		if err != nil {
			t.Fatal(err)
		}
		assertUnavailable(t, missing)
		for _, id := range []string{"901", "900"} {
			response, err := client.response(id, 4*time.Second)
			if err != nil {
				t.Fatal(err)
			}
			assertSample(t, response, 81)
		}
	})

	t.Run("caller cannot inject an endpoint or cluster principal", func(t *testing.T) {
		for _, extra := range []string{`"url":"http://127.0.0.1"`, `"port":14318`, `"clusterUuid":"other"`} {
			response := call(t, client, json.RawMessage(`{"hostUuid":"`+phase04NodeInfoHost+`",`+extra+`}`))
			if response.Error == nil || response.Error.Code != -32602 || len(response.Result) != 0 {
				t.Fatalf("injected authority accepted: %+v", response)
			}
		}
	})

	t.Run("host and principal mismatches reject without a cached result", func(t *testing.T) {
		mu.Lock()
		host = "other-host"
		mu.Unlock()
		assertUnavailable(t, call(t, client, params))
		mu.Lock()
		host = phase04NodeInfoHost
		principal = "other-principal"
		mu.Unlock()
		assertUnavailable(t, call(t, client, params))
		mu.Lock()
		principal = f.b.uuid
		mu.Unlock()
		assertSample(t, call(t, client, params), 81)
	})

	t.Run("backend timeout bounds a stalled sample and preserves subsequent calls", func(t *testing.T) {
		mu.Lock()
		stall = true
		mu.Unlock()
		start := time.Now()
		if err := client.request("950", "discovery:get-node-info", params); err != nil {
			t.Fatal(err)
		}
		select {
		case <-entered:
		case <-time.After(time.Second):
			t.Fatal("stalled RPC did not reach reverse handler")
		}
		if err := second.request("950", "discovery:get-node-info", json.RawMessage(`{"hostUuid":"missing-host"}`)); err != nil {
			t.Fatal(err)
		}
		missing, err := second.response("950", 500*time.Millisecond)
		if err != nil {
			t.Fatalf("concurrent RPC blocked behind slow node-info: %v", err)
		}
		assertUnavailable(t, missing)
		response, err := client.response("950", 3*time.Second)
		if err != nil {
			t.Fatal(err)
		}
		assertUnavailable(t, response)
		elapsed := time.Since(start)
		if elapsed < time.Second || elapsed > 2500*time.Millisecond {
			t.Fatalf("backend deadline elapsed %s, want 1s..2.5s", elapsed)
		}
		t.Logf("stalled reverse getter returned unavailable after %s", elapsed)
		mu.Lock()
		stall = false
		mu.Unlock()
		assertSample(t, call(t, client, params), 81)
	})

	t.Run("lost session rejects while membership stays and a fresh session repairs the cached client", func(t *testing.T) {
		observer := f.a.observers[noderec.ServiceCluster]
		previous := observer.established(f.b.uuid)
		observer.interrupt(f.b.uuid, true)
		t.Cleanup(func() { observer.interrupt(f.b.uuid, false) })
		assertUnavailable(t, call(t, client, params))
		for _, node := range []*task1Node{f.a, f.b} {
			peer := f.a
			if node == f.a {
				peer = f.b
			}
			var roster struct {
				Nodes []task1Member `json:"nodes"`
			}
			task2Decode(t, node.cm.call(t, "nodes:get-initial", nil).Result, &roster)
			found := false
			for _, member := range roster.Nodes {
				if member.NodeUUID == peer.uuid && member.ClusterID == task1ClusterID && member.AdmissionEpoch == 1 && member.State == "member" {
					found = true
				}
			}
			if !found {
				t.Fatalf("session outage changed %s membership: %+v", node.uuid, roster.Nodes)
			}
		}
		mu.Lock()
		utilization = 92
		mu.Unlock()
		observer.interrupt(f.b.uuid, false)
		task1Eventually(t, "new authenticated reverse session", 15*time.Second, func() bool { return observer.established(f.b.uuid) > previous })
		assertSample(t, call(t, client, params), 92)
	})

	testAuthority := func(name, path string) {
		t.Run(name, func(t *testing.T) {
			original := readLiveDurableFile(t, path)
			t.Cleanup(func() {
				if err := os.WriteFile(path, original, 0o600); err != nil {
					t.Error(err)
				}
			})
			removeAuthorityFile(t, path)
			mu.Lock()
			before := requests
			mu.Unlock()
			assertUnavailable(t, call(t, client, params))
			mu.Lock()
			after := requests
			mu.Unlock()
			if after != before {
				t.Fatalf("revoked authority dispatched node-info: before=%d after=%d", before, after)
			}
			if err := os.WriteFile(path, original, 0o600); err != nil {
				t.Fatal(err)
			}
			task1Eventually(t, "fresh sample after restored authority", 15*time.Second, func() bool {
				response := call(t, client, params)
				if response.Error != nil {
					assertUnavailable(t, response)
					return false
				}
				assertSample(t, response, 92)
				return true
			})
		})
	}
	testAuthority("removed peer pin rejects before target dispatch", filepath.Join(f.a.base, "cluster", "trusted", f.b.uuid+".json"))
	testAuthority("removed local admission rejects before target dispatch", filepath.Join(f.a.base, "cluster", "admission.json"))
}
