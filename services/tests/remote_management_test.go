// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-shared/testbin"
)

type managementSnapshot struct {
	Version  int    `json:"version"`
	Revision string `json:"revision"`
	Grants   []struct {
		PeerUUID string `json:"peerUuid"`
		Allowed  bool   `json:"allowed"`
		State    string `json:"state"`
	} `json:"grants"`
}

func managementSet(t *testing.T, cm *task1Process, peer string, allowed bool) managementSnapshot {
	t.Helper()
	before := decodeResult[managementSnapshot](t, cm.call(t, "cluster:get-remote-management", map[string]any{}))
	if before.Version != 1 || before.Revision == "" {
		t.Fatal("local permission snapshot is unavailable")
	}
	after := decodeResult[managementSnapshot](t, cm.call(t, "cluster:set-remote-management", map[string]any{
		"peerUuid": peer, "allowed": allowed, "expectedRevision": before.Revision,
	}))
	if after.Revision == before.Revision {
		t.Fatal("local permission update did not advance revision")
	}
	for _, grant := range after.Grants {
		if grant.PeerUUID == peer {
			wantState := "denied"
			if allowed {
				wantState = "allowed"
			}
			if grant.Allowed != allowed || grant.State != wantState {
				t.Fatalf("local permission allowed=%t state=%q, want %t/%s", grant.Allowed, grant.State, allowed, wantState)
			}
			return after
		}
	}
	t.Fatal("updated peer absent from local permission snapshot")
	return after
}

// managementGrantFixture gives an existing synthetic caller an explicit local
// grant through the real writer before the fixture starts its control clients.
func managementGrantFixture(t *testing.T, clusterDir, targetUUID, peerUUID string) {
	t.Helper()
	mesh := clustertrust.Open(clusterDir)
	before, ok := mesh.PeerServiceIncarnation(peerUUID)
	if !ok || mesh.NodeUUID() != targetUUID {
		t.Fatal("permission fixture requires a coherent target and peer admission")
	}
	// Certificate-only fixtures need matching metadata so cluster-manager
	// adopts their existing identity instead of minting a replacement leaf.
	identityPath := filepath.Join(clusterDir, "identity.json")
	identityData, err := os.ReadFile(identityPath)
	if os.IsNotExist(err) {
		jointWriteJSON(t, identityPath, map[string]string{"node_uuid": targetUUID})
	} else if err != nil {
		t.Fatalf("read fixture identity metadata: %v", err)
	} else {
		var identity struct {
			NodeUUID string `json:"node_uuid"`
		}
		if err := json.Unmarshal(identityData, &identity); err != nil || identity.NodeUUID != targetUUID {
			t.Fatal("fixture identity metadata disagrees with its certificate")
		}
	}
	listener := jointListener(t)
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	base := filepath.Dir(clusterDir)
	cm := startTask1Process(t, base, clusterMgrBin, "--config-dir", base, "--port", strconv.Itoa(port))
	defer cm.stop()
	identity := decodeResult[cmNodeID](t, cm.call(t, "cluster:get-node-id", nil))
	if identity.NodeUUID != targetUUID {
		t.Fatal("permission writer replaced the fixture identity")
	}
	managementSet(t, cm, peerUUID, true)
	mesh.Refresh()
	after, ok := mesh.PeerServiceIncarnation(peerUUID)
	permission := mesh.RemoteManagement(peerUUID)
	if !ok || after != before || !permission.Allowed || permission.Incarnation != before {
		t.Fatal("fixture grant did not preserve and authorize the original admission")
	}
}

// This fixture keeps the real broker/engine-manager policy writer and the
// existing absent-engine overrides. A second independently pinned certificate
// supplies the opposite grant; target-local changes use a real cluster-manager.
type managementFixture struct {
	joint  *policyJointFixture
	cm     *task1Process
	cmPort int
	denied *http.Client
}

func newManagementFixture(t *testing.T) *managementFixture {
	t.Helper()
	f := &managementFixture{joint: newPolicyJointFixture(t), cmPort: temporaryClusterPort(t)}
	base := t.TempDir()
	cert := mintClusterIdentity(t, base, "management-denied")
	writeActiveAdmission(t, base)
	targetCert, err := os.ReadFile(filepath.Join(f.joint.cluster, "node.crt"))
	if err != nil {
		t.Fatal(err)
	}
	writePin(t, base, "policy-joint-target", string(targetCert))
	writeMembers(t, base, "management-denied", "policy-joint-target")
	writePin(t, f.joint.cluster, "management-denied", cert)
	writeMembers(t, f.joint.cluster, "policy-joint-target", "policy-joint-origin", "management-denied")
	cfg, ok := clustertrust.Open(base).ClientTLSConfig("policy-joint-target")
	if !ok {
		t.Fatal("denied fixture peer has no target TLS authority")
	}
	f.denied = &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{TLSClientConfig: cfg}}
	t.Cleanup(f.denied.CloseIdleConnections)
	f.startCM(t)
	managementSet(t, f.cm, "policy-joint-origin", true)
	managementSet(t, f.cm, "management-denied", false)
	return f
}

func (f *managementFixture) startCM(t *testing.T) {
	t.Helper()
	f.cm = startTask1Process(t, f.joint.base, clusterMgrBin, "--config-dir", f.joint.base, "--port", strconv.Itoa(f.cmPort))
	f.cm.call(t, "cluster:get-node-id", nil)
}

func managementHTTPResponse(t *testing.T, client *http.Client, port int, method, path string, params any) (int, string, []byte) {
	t.Helper()
	var body io.Reader
	if params != nil {
		body = bytes.NewReader(mustMarshal(t, params))
	}
	req, err := http.NewRequest(method, "https://127.0.0.1:"+strconv.Itoa(port)+path, body)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("fixture ec request: %v", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, (256<<10)+1))
	if err != nil || len(data) > 256<<10 {
		t.Fatal("ec response exceeds read bound or is unreadable")
	}
	return resp.StatusCode, resp.Header.Get("Content-Type"), data
}

func managementAssertHTTP(t *testing.T, method, path string, actual int, contentType string, data []byte, status int) {
	t.Helper()
	if actual != status {
		t.Fatalf("%s %s status=%d want=%d", method, path, actual, status)
	}
	if status == http.StatusConflict {
		var refusal struct {
			Code  string `json:"code"`
			Error string `json:"error"`
		}
		if err := json.Unmarshal(data, &refusal); err != nil {
			t.Fatalf("decode permission refusal: %v", err)
		}
		if refusal.Code != "remote_management_denied" || refusal.Error != "Remote management is blocked on this device. Change permission locally on that device." {
			t.Fatalf("permission refusal code=%q", refusal.Code)
		}
		if contentType == "application/x-ndjson" {
			t.Fatal("denial opened a progress stream")
		}
	}
}

func managementHTTP(t *testing.T, client *http.Client, port int, method, path string, params any, status int) []byte {
	t.Helper()
	actual, contentType, data := managementHTTPResponse(t, client, port, method, path, params)
	managementAssertHTTP(t, method, path, actual, contentType, data, status)
	return data
}

func (f *managementFixture) permission(t *testing.T, client *http.Client, allowed bool) {
	t.Helper()
	data := managementHTTP(t, client, f.joint.ports[1], http.MethodGet, "/v1/remote-management?peerUuid=policy-joint-origin&local=true", nil, http.StatusOK)
	var view struct {
		Version  int    `json:"version"`
		Allowed  bool   `json:"allowed"`
		Revision string `json:"revision"`
		State    string `json:"state"`
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&view); err != nil {
		t.Fatalf("caller-only permission view: %v", err)
	}
	wantState := "denied"
	if allowed {
		wantState = "allowed"
	}
	if view.Version != 1 || view.Allowed != allowed || view.State != wantState || view.Revision == "" {
		t.Fatalf("caller permission allowed=%t state=%q, want %t/%s", view.Allowed, view.State, allowed, wantState)
	}
}

func TestRemoteManagementDirectMutationBoundary(t *testing.T) {
	f := newManagementFixture(t)
	f.permission(t, f.joint.client, true)
	f.permission(t, f.denied, false)
	beforePolicy := readPolicyBytes(t, f.joint.path)
	permissionPath := filepath.Join(f.joint.cluster, clustertrust.RemoteManagementFile)
	beforePermission := readLiveDurableFile(t, permissionPath)
	// These are every mutation registered on ec. Empty/forged fields must be
	// refused by permission before any handler parses them or starts a stream.
	for _, path := range []string{
		"/v1/engines/install", "/v1/models/pull", "/v1/models/load",
		"/v1/models/unload", "/v1/models/delete", "/v1/engines/start", "/v1/engines/stop",
		"/v1/node-policy/set", "/v1/node-availability/set", "/v1/engine-settings/apply", settings.SwitchApplyPath,
	} {
		t.Run(path, func(t *testing.T) {
			managementHTTP(t, f.denied, f.joint.ports[1], http.MethodPost, path,
				map[string]any{"engine": "ollama", "model": "fixture-model", "caller": "policy-joint-origin", "nodeId": "policy-joint-origin", "local": true}, http.StatusConflict)
			f.joint.assertUnchanged(t, beforePolicy)
			if !bytes.Equal(readLiveDurableFile(t, permissionPath), beforePermission) {
				t.Fatal("remote request changed local grants")
			}
		})
	}
	// A granted peer reaches the actual broker writer, providing a positive
	// control independent of malformed-body rejection and absent engines.
	peer := jointExpectedPeer(f.joint.want)
	peer.Idle.UnloadAfterMinutes = 19
	data := managementHTTP(t, f.joint.client, f.joint.ports[1], http.MethodPost, "/v1/node-policy/set", map[string]any{
		"policy": peer, "expectedPersistedRevision": f.joint.want.Revision,
	}, http.StatusOK)
	var result struct {
		Policy struct {
			Revision string `json:"revision"`
		} `json:"policy"`
	}
	if err := json.Unmarshal(data, &result); err != nil || result.Policy.Revision == "" || result.Policy.Revision == f.joint.want.Revision {
		t.Fatal("allowed writer did not advance policy revision")
	}
	f.joint.want.Idle = peer.Idle
	f.joint.want.Revision = result.Policy.Revision
	f.joint.assertPolicy(t)
}

func TestRemoteManagementDeniedReadsAndLocalWriterBoundary(t *testing.T) {
	f := newManagementFixture(t)
	port := f.joint.ports[1]
	data := managementHTTP(t, f.denied, port, http.MethodPost, "/v1/node-policy/get", map[string]any{}, http.StatusOK)
	assertJointPeer(t, data, f.joint.want)
	managementHTTP(t, f.denied, port, http.MethodGet, "/v1/engines", nil, http.StatusOK)
	data = managementHTTP(t, f.denied, port, http.MethodPost, "/v1/engine-settings/get", settings.Request{Engine: "ollama"}, http.StatusOK)
	var snapshot settings.Snapshot
	if err := json.Unmarshal(data, &snapshot); err != nil {
		t.Fatalf("decode readable settings: %v", err)
	}
	managementHTTP(t, f.denied, port, http.MethodPost, "/v1/engine-settings/preview", settings.Request{Engine: "ollama", ExpectedRevision: snapshot.Revision, Settings: snapshot.Settings}, http.StatusOK)
	managementHTTP(t, f.denied, port, http.MethodPost, settings.SwitchStatusPath, settings.SwitchStatusRequest{}, http.StatusOK)
	f.permission(t, f.denied, false)
	before := readLiveDurableFile(t, filepath.Join(f.joint.cluster, clustertrust.RemoteManagementFile))
	for _, client := range []*http.Client{f.denied, f.joint.client} {
		for _, allowed := range []bool{false, true} {
			managementHTTP(t, client, port, http.MethodPost, "/v1/remote-management", map[string]any{
				"peerUuid": "management-denied", "allowed": allowed, "caller": "", "local": true,
			}, http.StatusMethodNotAllowed)
		}
	}
	if !bytes.Equal(readLiveDurableFile(t, filepath.Join(f.joint.cluster, clustertrust.RemoteManagementFile)), before) {
		t.Fatal("network setter attempt changed grants")
	}
}

func TestRemoteManagementRevokePersistsAcrossWorkerRestart(t *testing.T) {
	f := newManagementFixture(t)
	f.permission(t, f.joint.client, true)
	managementSet(t, f.cm, "policy-joint-origin", false)
	f.permission(t, f.joint.client, false)
	assertDenied := func() {
		response := f.joint.originCall(t, "engine:remote-policy-set", map[string]any{
			"policy": jointExpectedPeer(f.joint.want), "expectedPersistedRevision": f.joint.want.Revision,
		})
		if response.Error == nil || response.Error.Code != -32012 || len(response.Result) != 0 {
			t.Fatal("origin RPC did not preserve revoked permission code -32012")
		}
		f.joint.assertPolicy(t)
	}
	assertDenied()
	f.cm.stop()
	f.startCM(t)
	f.permission(t, f.joint.client, false)
	assertDenied()
	// The broker owns its child manager; restarting this pair tests a cold
	// independent permission reader and broker writer against the same profile.
	f.joint.origin.cleanup()
	f.joint.target.cleanup()
	f.joint.client.CloseIdleConnections()
	f.denied.CloseIdleConnections()
	f.joint.startTarget(t)
	f.joint.startOrigin(t)
	f.permission(t, f.joint.client, false)
	assertDenied()
}

// managementPolicyAttempt reads the writer's current revision independently for
// every attempt, so denial cannot be explained by stale CAS or invalid policy.
func managementPolicyAttempt(t *testing.T, writer *interopProcess, path string, client *http.Client, port, status int) {
	t.Helper()
	response := callRPC(t, writer.in, writer.msgs, "policy:get", map[string]any{}, 10*time.Second)
	var before struct {
		Policy nodepolicy.Policy `json:"policy"`
	}
	if response.Error != nil {
		t.Fatal("read actual policy writer")
	}
	task2Decode(t, response.Result, &before)
	assertStoredPolicy(t, path, before.Policy)
	beforeBytes := readPolicyBytes(t, path)
	draft := before.Policy.PeerProjection()
	draft.Idle.UnloadAfterMinutes++
	data := managementHTTP(t, client, port, http.MethodPost, "/v1/node-policy/set", map[string]any{
		"policy": draft, "expectedPersistedRevision": before.Policy.Revision,
	}, status)
	want := before.Policy
	if status == http.StatusOK {
		var result struct {
			Policy nodepolicy.PeerPolicy `json:"policy"`
		}
		if err := json.Unmarshal(data, &result); err != nil {
			t.Fatal(err)
		}
		if result.Policy.Revision == "" || result.Policy.Revision == before.Policy.Revision {
			t.Fatal("allowed policy mutation did not advance revision")
		}
		want.Idle = draft.Idle
		want.Revision = result.Policy.Revision
		draft.Revision = result.Policy.Revision
		if !reflect.DeepEqual(result.Policy, draft) {
			t.Fatal("allowed policy result differs from the submitted device edit")
		}
	} else if !bytes.Equal(readPolicyBytes(t, path), beforeBytes) {
		t.Fatal("denied mutation changed actual policy bytes")
	}
	response = callRPC(t, writer.in, writer.msgs, "policy:get", map[string]any{}, 10*time.Second)
	if response.Error != nil {
		t.Fatal("read policy after mutation")
	}
	var after struct {
		Policy nodepolicy.Policy `json:"policy"`
	}
	task2Decode(t, response.Result, &after)
	if !reflect.DeepEqual(after.Policy, want) {
		t.Fatal("actual policy writer differs from the expected accepted or denied edit")
	}
	assertStoredPolicy(t, path, want)
}

// Reuse the registered joint images and empty scanner, but bind the real
// broker/manager to an existing cluster-manager's actual profile.
func managementPolicyWriterAt(t *testing.T, base string) *policyJointFixture {
	t.Helper()
	f := &policyJointFixture{base: base, cluster: filepath.Join(base, "cluster")}
	inert := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(inert.Close)
	f.inertPort = inert.Listener.Addr().(*net.TCPAddr).Port
	jointIsolateEngines(t, base, f.inertPort)
	f.path = filepath.Join(jointAppDir(base), nodepolicy.FileName)
	f.want = nodepolicy.Default()
	f.want.Revision = "12"
	jointWriteJSON(t, f.path, f.want)
	binDir := testbin.Dir(t, "tests", "policy-joint", "joint-target-engine", "joint-origin-engine", "joint-empty-scanner")
	f.engine = filepath.Join(binDir, testbin.Exe("joint-target-engine"))
	f.scanner = filepath.Join(binDir, testbin.Exe("joint-empty-scanner"))
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	jointCopyExecutable(t, engineMgrBin, f.engine)
	jointCopyExecutable(t, self, f.scanner)
	for range 3 {
		f.ports = append(f.ports, temporaryClusterPort(t))
	}
	f.startTarget(t)
	return f
}

// The switch fixture's synthetic engine and real proxy are retained, with the
// registered empty scanner replacing discovery of installed peers.
func managementSwitchWriterAt(t *testing.T, base string) *switchProcessFixture {
	t.Helper()
	f := &switchProcessFixture{base: base, ports: reserveFixtureServicePorts(t), sourcePort: temporaryClusterPort(t), targetPort: temporaryClusterPort(t), facadePort: temporaryClusterPort(t), release: make(chan struct{}), starts: make(map[string][]int32)}
	f.control = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		pid, err := strconv.ParseInt(r.URL.Query().Get("pid"), 10, 32)
		if err != nil || r.URL.Path != "/started" {
			http.Error(w, "invalid synthetic engine event", http.StatusBadRequest)
			return
		}
		f.mu.Lock()
		f.starts[r.URL.Query().Get("engine")] = append(f.starts[r.URL.Query().Get("engine")], int32(pid))
		f.mu.Unlock()
	}))
	t.Cleanup(f.control.Close)
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	jointIsolateEngines(t, base, temporaryClusterPort(t))
	journal := make(map[string]any)
	for _, engine := range []struct {
		name           string
		server, facade int
	}{{"ollama", f.sourcePort, f.facadePort}, {"llamacpp", f.targetPort, temporaryClusterPort(t)}} {
		probe := map[string]any{"http": "http://127.0.0.1:{port}/", "status": http.StatusOK, "timeout_s": 10, "interval_s": 60, "json_match": nil}
		rt := map[string]any{"mode": "process", "bin": self, "port": engine.server, "bind": "127.0.0.1", "ready": probe, "health": probe,
			"env":  map[string]string{"NVPAIR_TEST_SWITCH_ENGINE": self, "NVPAIR_TEST_SWITCH_NAME": engine.name, "NVPAIR_TEST_SWITCH_PORT": "{port}", "NVPAIR_TEST_SWITCH_CONTROL": f.control.URL},
			"stop": map[string]any{"signal": "term", "grace_s": 1}}
		jointWriteJSON(t, filepath.Join(jointAppDir(base), "engines", engine.name+".json"), map[string]any{
			"engine": engine.name, "runtime": rt, "platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{"detect": []string{self}, "runtime": rt}},
		})
		journal[engine.name] = map[string]any{"explicit": true, "snapshot": map[string]any{
			"engine": engine.name, "revision": 1, "appliedRevision": 1, "phase": "idle", "format": "pair-arguments-v1",
			"settings": settings.Config{ServerPort: engine.server, ProxyPort: engine.facade},
		}}
	}
	jointWriteJSON(t, filepath.Join(base, "engine-settings-operations.json"), journal)
	jointWriteJSON(t, filepath.Join(base, "engine-switch-operations.json"), settings.SwitchJournal{Version: settings.SwitchVersion, Receipts: make(map[string]settings.SwitchReceipt)})
	policy := nodepolicy.Default()
	policy.Revision = "12"
	jointWriteJSON(t, filepath.Join(jointAppDir(base), nodepolicy.FileName), policy)
	binDir := testbin.Dir(t, "tests", "policy-joint", "joint-target-engine", "joint-origin-engine", "joint-empty-scanner")
	enginePath := filepath.Join(binDir, testbin.Exe("joint-target-engine"))
	scannerPath := filepath.Join(binDir, testbin.Exe("joint-empty-scanner"))
	jointCopyExecutable(t, engineMgrBin, enginePath)
	jointCopyExecutable(t, self, scannerPath)
	scannerPort := temporaryClusterPort(t)
	args := append([]string{"--scanner-path", scannerPath, "--engine-manager-path", enginePath, "--proxy-path", proxyBin, "--proxy-engines", "ollama,llamacpp", "--cluster-dir", filepath.Join(base, "cluster")}, f.ports.args()...)
	f.p = startInteropProcess(t, base, brokerBin, f.ports.engineManagerHTTP, []string{"OLLAMA_HOST=", "NVPAIR_TEST_SUBSCRIPTION_SCANNER=" + scannerPath, "NVPAIR_TEST_SUBSCRIPTION_PORT=" + strconv.Itoa(scannerPort)}, args...)
	for _, path := range []string{enginePath, scannerPath, proxyBin} {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		f.p.paths = append(f.p.paths, canonical)
	}
	f.p.ports = append(f.p.ports, f.ports.engineControl, scannerPort, f.sourcePort, f.targetPort)
	waitForMethod(t, f.p.msgs, "app:ready", 20*time.Second)
	if port := waitProxyReady(t, f.p.in, f.p.msgs, 20*time.Second); port != f.facadePort {
		t.Fatal("synthetic source proxy did not use its explicit fixture port")
	}
	waitEngineProxyReady(t, "llamacpp-proxy", f.p.in, f.p.msgs, 20*time.Second)
	f.rpcOK(t, "engine:start", map[string]string{"engine": "ollama"})
	f.pid(t, "ollama")
	// Initialize every settings baseline before a switch revision is captured.
	// Otherwise the periodic refresh can add an unrelated engine's revision.
	for _, engine := range engines.Names() {
		f.rpcOK(t, "engine:get-settings", map[string]string{"engine": engine})
	}
	return f
}

func TestRemoteManagementReverseMutationBoundary(t *testing.T) {
	nodes, blockedPort := newTask1Nodes(t)
	allowed, target, denied := nodes[0], nodes[1], nodes[2]
	f := managementSwitchWriterAt(t, target.base)
	target.ports[noderec.ServiceEngineControl] = f.ports.engineControl
	ec := newTask1Observer(t, target, noderec.ServiceEngineControl, nodes)
	target.cm.call(t, "cluster:set-local-services", map[string]any{"generation": 2, "services": []noderec.RegisterParams{
		{Service: noderec.ServiceCluster, Port: target.observers[noderec.ServiceCluster].port},
		{Service: noderec.ServiceEngineControl, Port: ec.port},
	}})
	managementSet(t, target.cm, allowed.uuid, true)
	managementSet(t, target.cm, denied.uuid, false)
	reverseClient := func(node *task1Node) *http.Client {
		t.Helper()
		task1Eventually(t, "authenticated reverse ec route for "+node.uuid, 15*time.Second, func() bool {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			conn, err := clustertrust.DialReverse(ctx, node.base, target.uuid, noderec.ServiceEngineControl)
			if err != nil {
				return false
			}
			if err := conn.Close(); err != nil {
				t.Fatal(err)
			}
			return true
		})
		adapter := clustertrust.NewPeerServiceClient(node.mesh, clustertrust.PeerServiceClientOptions{
			BaseConfigDir: node.base, PeerClientOptions: clustertrust.PeerClientOptions{Timeout: 10 * time.Second},
			ResolveRoute: func(uuid string, service noderec.ServiceKey) (clustertrust.PeerServiceRoute, bool) {
				node.mesh.Refresh()
				incarnation, ok := node.mesh.PeerServiceIncarnation(uuid)
				return clustertrust.PeerServiceRoute{Incarnation: incarnation, Address: net.JoinHostPort("127.0.0.1", strconv.Itoa(blockedPort)), Protocol: clustertrust.ReverseTLS, Generation: 1}, ok && uuid == target.uuid && service == noderec.ServiceEngineControl
			},
		})
		t.Cleanup(adapter.CloseIdle)
		return adapter.Client(target.uuid, noderec.ServiceEngineControl)
	}
	allowedClient, deniedClient := reverseClient(allowed), reverseClient(denied)
	jointIsolateEngines(t, denied.base, blockedPort)
	deniedOrigin := startInteropProcess(t, denied.base, engineMgrBin, 0, []string{"OLLAMA_HOST="}, "--cluster-dir", filepath.Join(denied.base, "cluster"), "--loaded-poll-interval", "0")
	waitForMethod(t, deniedOrigin.msgs, "engine:ready", 10*time.Second)
	completeEngineManagerRecovery(t, func(method string, params any) jsonrpc.Message {
		return callRPC(t, deniedOrigin.in, deniedOrigin.msgs, method, params, 10*time.Second)
	})
	sendLine(t, deniedOrigin.in, jsonrpc.Message{JSONRPC: "2.0", Method: noderec.NotifyNodes, Params: mustMarshal(t, noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{{HostUUID: target.uuid, ClusterUUID: target.uuid, IP: "127.0.0.1", Trusted: true,
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineControl: {Port: blockedPort}}}}})})
	for _, caller := range []struct {
		node    *task1Node
		client  *http.Client
		allowed bool
	}{{allowed, allowedClient, true}, {denied, deniedClient, false}} {
		data := managementHTTP(t, caller.client, blockedPort, http.MethodGet, "/v1/remote-management", nil, http.StatusOK)
		var view struct {
			Allowed bool   `json:"allowed"`
			State   string `json:"state"`
		}
		task2Decode(t, data, &view)
		wantState := "denied"
		if caller.allowed {
			wantState = "allowed"
		}
		if view.Allowed != caller.allowed || view.State != wantState || ec.count(caller.node.uuid, http.MethodGet, "/v1/remote-management") != 1 {
			t.Fatal("reverse read did not authenticate each distinct caller with its opposite grant")
		}
	}
	permissionPath := filepath.Join(target.base, "cluster", clustertrust.RemoteManagementFile)
	grantBytes := readLiveDurableFile(t, permissionPath)
	policyPath := filepath.Join(jointAppDir(target.base), nodepolicy.FileName)
	assertForgedDenied := func(t *testing.T, path string, canonical any) {
		t.Helper()
		policyBytes := readPolicyBytes(t, policyPath)
		settingsBytes := readLiveDurableFile(t, filepath.Join(target.base, "engine-settings-operations.json"))
		switchBytes := readLiveDurableFile(t, filepath.Join(target.base, "engine-switch-operations.json"))
		status := f.status(t, settings.SwitchStatusRequest{})
		f.mu.Lock()
		starts := make(map[string][]int32)
		for engine, pids := range f.starts {
			starts[engine] = append([]int32(nil), pids...)
		}
		f.mu.Unlock()
		assertUnchanged := func() {
			t.Helper()
			afterStatus := f.status(t, settings.SwitchStatusRequest{})
			if !bytes.Equal(readPolicyBytes(t, policyPath), policyBytes) || !bytes.Equal(readLiveDurableFile(t, filepath.Join(target.base, "engine-settings-operations.json")), settingsBytes) || !bytes.Equal(readLiveDurableFile(t, filepath.Join(target.base, "engine-switch-operations.json")), switchBytes) || afterStatus.ConfigurationRevision != status.ConfigurationRevision || afterStatus.ActiveRequestID != status.ActiveRequestID || !reflect.DeepEqual(afterStatus.Receipt, status.Receipt) {
				t.Fatal("forged identity changed actual policy/settings/switch writer state")
			}
			beforePolicy, err := nodepolicy.Parse(policyBytes)
			if err != nil {
				t.Fatal(err)
			}
			var actualPolicy struct {
				Policy nodepolicy.Policy `json:"policy"`
			}
			task2Decode(t, f.rpcOK(t, "policy:get", map[string]any{}), &actualPolicy)
			if !reflect.DeepEqual(actualPolicy.Policy, beforePolicy) {
				t.Fatal("forged identity changed in-memory policy writer state")
			}
			f.mu.Lock()
			unchanged := reflect.DeepEqual(f.starts, starts)
			f.mu.Unlock()
			if !unchanged || !bytes.Equal(readLiveDurableFile(t, permissionPath), grantBytes) {
				t.Fatal("forged identity changed engine start counters or local grants")
			}
		}
		// nodeId is recognized by each real decoder. Unlike the noisy body
		// below, this request can reach the writer if authenticated identity
		// is incorrectly replaced by the submitted allowed principal.
		var validForged map[string]json.RawMessage
		if err := json.Unmarshal(mustMarshal(t, canonical), &validForged); err != nil {
			t.Fatal(err)
		}
		validForged["nodeId"] = mustMarshal(t, allowed.uuid)
		count := ec.count(denied.uuid, http.MethodPost, path)
		actual, contentType, data := managementHTTPResponse(t, deniedClient, blockedPort, http.MethodPost, path, validForged)
		if ec.count(denied.uuid, http.MethodPost, path) != count+1 {
			t.Fatal("valid forged request did not traverse denied caller's real reverse transport")
		}
		// Inspect effects before the refusal status so an identity-substitution
		// control witnesses an actual writer change, not just parser ordering.
		assertUnchanged()
		managementAssertHTTP(t, http.MethodPost, path, actual, contentType, data, http.StatusConflict)
		var forged map[string]json.RawMessage
		if err := json.Unmarshal(mustMarshal(t, canonical), &forged); err != nil {
			t.Fatal(err)
		}
		for _, key := range []string{"caller", "peerUuid", "nodeUuid", "nodeId"} {
			forged[key] = mustMarshal(t, allowed.uuid)
		}
		count = ec.count(denied.uuid, http.MethodPost, path)
		// The exact typed wire representation of RPC -32012 must precede body
		// parsing, including strict settings/switch unknown-field validation.
		managementHTTP(t, deniedClient, blockedPort, http.MethodPost, path, forged, http.StatusConflict)
		if ec.count(denied.uuid, http.MethodPost, path) != count+1 {
			t.Fatal("forged request did not traverse denied caller's real reverse transport")
		}
		method := map[string]string{"/v1/node-policy/set": "engine:remote-policy-set", "/v1/engine-settings/apply": "engine:remote-apply-settings", settings.SwitchApplyPath: settings.RemoteSwitchMethod}[path]
		var rpcParams map[string]json.RawMessage
		if err := json.Unmarshal(mustMarshal(t, canonical), &rpcParams); err != nil {
			t.Fatal(err)
		}
		rpcParams["nodeId"] = mustMarshal(t, target.uuid)
		response := callRPC(t, deniedOrigin.in, deniedOrigin.msgs, method, rpcParams, 15*time.Second)
		if response.Error == nil || response.Error.Code != -32012 || len(response.Result) != 0 {
			t.Fatal("reverse origin RPC did not retain typed -32012")
		}
		assertUnchanged()
	}
	assertAllowedRoute := func(t *testing.T, path string) {
		t.Helper()
		if ec.count(allowed.uuid, http.MethodPost, path) != 1 {
			t.Fatal("canonical allowed control did not traverse its own authenticated reverse transport once")
		}
		if !bytes.Equal(readLiveDurableFile(t, permissionPath), grantBytes) {
			t.Fatal("allowed control changed target-local grants")
		}
	}
	t.Run("policy forged identity", func(t *testing.T) {
		var view struct {
			Policy nodepolicy.Policy `json:"policy"`
		}
		task2Decode(t, f.rpcOK(t, "policy:get", map[string]any{}), &view)
		draft := view.Policy.PeerProjection()
		draft.Idle.UnloadAfterMinutes++
		assertForgedDenied(t, "/v1/node-policy/set", map[string]any{"policy": draft, "expectedPersistedRevision": view.Policy.Revision})
		managementPolicyAttempt(t, f.p, policyPath, allowedClient, blockedPort, http.StatusOK)
		assertAllowedRoute(t, "/v1/node-policy/set")
	})
	t.Run("settings forged identity", func(t *testing.T) {
		var snapshot settings.Snapshot
		task2Decode(t, managementHTTP(t, allowedClient, blockedPort, http.MethodPost, "/v1/engine-settings/get", settings.Request{Engine: "ollama"}, http.StatusOK), &snapshot)
		request := settings.Request{Engine: "ollama", ExpectedRevision: snapshot.Revision, RequestID: "reverse-settings-control", Settings: snapshot.Settings}
		assertForgedDenied(t, "/v1/engine-settings/apply", request)
		var receipt struct {
			Phase    string `json:"phase"`
			Revision uint64 `json:"revision"`
		}
		task2Decode(t, managementHTTP(t, allowedClient, blockedPort, http.MethodPost, "/v1/engine-settings/apply", request, http.StatusOK), &receipt)
		if receipt.Phase != "succeeded" || receipt.Revision != snapshot.Revision {
			t.Fatal("canonical allowed settings control did not persist its successful receipt")
		}
		stored := readOutboundJSON[map[string]struct {
			Receipts map[string]struct {
				Phase    string `json:"phase"`
				Revision uint64 `json:"revision"`
			} `json:"receipts"`
		}](t, filepath.Join(target.base, "engine-settings-operations.json"))
		persisted := stored["ollama"].Receipts[request.RequestID]
		if persisted.Phase != "succeeded" || persisted.Revision != snapshot.Revision {
			t.Fatal("allowed settings receipt was not saved by the real writer")
		}
		assertAllowedRoute(t, "/v1/engine-settings/apply")
	})
	t.Run("switch forged identity", func(t *testing.T) {
		request := f.request(t, "reverse-switch-control")
		assertForgedDenied(t, settings.SwitchApplyPath, request)
		var result settings.SwitchResult
		task2Decode(t, managementHTTP(t, allowedClient, blockedPort, http.MethodPost, settings.SwitchApplyPath, request, http.StatusOK), &result)
		if result.Receipt.Caller != allowed.uuid || result.Receipt.NodeID != target.uuid || result.Receipt.RequestID != request.RequestID {
			t.Fatal("allowed switch control lost authenticated caller")
		}
		var terminal settings.SwitchStatus
		task1Eventually(t, "allowed reverse switch completes", 45*time.Second, func() bool {
			task2Decode(t, managementHTTP(t, allowedClient, blockedPort, http.MethodPost, settings.SwitchStatusPath, settings.SwitchStatusRequest{RequestID: request.RequestID}, http.StatusOK), &terminal)
			return terminal.Receipt != nil && terminal.Receipt.Terminal
		})
		if terminal.Receipt.Phase != settings.SwitchCompleted || terminal.Receipt.Error != nil || terminal.Receipt.Caller != allowed.uuid || terminal.Receipt.NodeID != target.uuid {
			t.Fatal("canonical allowed switch did not complete against the synthetic engines")
		}
		journal, err := settings.DecodeSwitchJournal(readLiveDurableFile(t, filepath.Join(target.base, "engine-switch-operations.json")))
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(journal.Receipts[request.RequestID], *terminal.Receipt) {
			t.Fatal("allowed reverse switch did not persist its terminal receipt")
		}
		assertSwitchProcessStopped(t, f.pid(t, "ollama"))
		f.pid(t, "llamacpp")
		assertAllowedRoute(t, settings.SwitchApplyPath)
	})
}

func TestRemoteManagementReverseRevocationBoundary(t *testing.T) {
	f := newTask2Fixture(t)
	a := f.engines[f.a.uuid]
	// Inventory reconciles the fixture-owned HTTP engine's presence before an
	// action can use it; no real executable or installed engine is adopted.
	a.call(t, "engine:remote-get-installed", map[string]string{"node": f.b.uuid})
	managementSet(t, f.b.cm, f.a.uuid, false)
	response := task2Response(t, a, a.send(t, "engine:remote-management-get", map[string]string{"nodeId": f.b.uuid}, true))
	if response.Error != nil {
		t.Fatalf("reverse permission read: %+v", response.Error)
	}
	var capability struct {
		Allowed bool   `json:"allowed"`
		State   string `json:"state"`
	}
	task2Decode(t, response.Result, &capability)
	if capability.Allowed || capability.State != "denied" {
		t.Fatal("reverse permission lost inner mTLS caller")
	}
	request := map[string]string{"node": f.b.uuid, "engine": "ollama", "model": "task2-model"}
	response = task2Response(t, a, a.send(t, "engine:remote-unload-model", request, true))
	if response.Error == nil || response.Error.Code != -32012 {
		t.Fatal("reverse mutation did not preserve -32012")
	}
	f.mu.Lock()
	unloads := f.unloads
	f.mu.Unlock()
	if unloads != 0 {
		t.Fatal("denied reverse mutation reached fake engine")
	}
	managementSet(t, f.b.cm, f.a.uuid, true)
	// The observer has consumed the completed target response, but withholds
	// it from the caller. Revoke at this barrier: accepted work remains a
	// success, while the next independent operation must authorize freshly.
	reached, release := make(chan struct{}), make(chan struct{})
	var reachedOnce, releaseOnce sync.Once
	t.Cleanup(func() { releaseOnce.Do(func() { close(release) }) })
	ec := f.b.observers[noderec.ServiceEngineControl]
	ec.mu.Lock()
	ec.afterReply = func(peer, path string) {
		if peer == f.a.uuid && path == "/v1/models/unload" {
			reachedOnce.Do(func() { close(reached) })
			<-release
		}
	}
	ec.mu.Unlock()
	id := a.send(t, "engine:remote-unload-model", request, true)
	select {
	case <-reached:
	case <-time.After(10 * time.Second):
		t.Fatal("allowed reverse mutation never reached completed-response barrier")
	}
	managementSet(t, f.b.cm, f.a.uuid, false)
	releaseOnce.Do(func() { close(release) })
	response = task2Response(t, a, id)
	if response.Error != nil {
		t.Fatalf("allowed reverse unload: %+v", response.Error)
	}
	f.mu.Lock()
	unloads = f.unloads
	f.mu.Unlock()
	if unloads != 1 {
		t.Fatalf("allowed reverse mutation calls=%d, want 1", unloads)
	}
	response = task2Response(t, a, a.send(t, "engine:remote-unload-model", request, true))
	if response.Error == nil || response.Error.Code != -32012 {
		t.Fatal("reverse reader reused allow after revoke")
	}
	f.mu.Lock()
	unloads = f.unloads
	f.mu.Unlock()
	if unloads != 1 {
		t.Fatal("revoked reverse mutation reached fake engine")
	}
	for _, method := range []string{"engine:remote-install", "engine:remote-pull-model"} {
		response := task2Response(t, a, a.send(t, method, request, true))
		if response.Error == nil || response.Error.Code != -32012 {
			t.Fatalf("reverse streaming %s lost -32012", method)
		}
	}
}

func TestRemoteManagementPairingChoicesRemainTargetLocal(t *testing.T) {
	f := newBlockedPairingFixture(t, nil)
	response := decodeResult[outboundInviteResult](t, f.b.call("cluster:respond-to-invite", map[string]any{
		"inviteId": f.invite.InviteID, "accept": true, "pin": f.invite.Pin, "allowRemoteManagement": true,
	}))
	if response.State != "paired" {
		t.Fatalf("pairing state=%q, want paired", response.State)
	}
	waitInviteState(t, f.a, f.invite.InviteID, "paired")
	assertOutboundDurablePair(t, f)
	assertChoice := func(cm *cmProc, peer string, allowed bool) {
		view := decodeResult[managementSnapshot](t, cm.call("cluster:get-remote-management", map[string]any{}))
		for _, grant := range view.Grants {
			if grant.PeerUUID == peer {
				if grant.Allowed != allowed {
					t.Fatalf("target-local grant=%t want=%t", grant.Allowed, allowed)
				}
				return
			}
		}
		t.Fatal("paired peer has no local permission choice")
	}
	assertChoice(f.a, f.bID.NodeUUID, false)
	assertChoice(f.b, f.aID.NodeUUID, true)
}

func TestRemoteManagementUnfinishedPairingNeverGrants(t *testing.T) {
	newFixture := func(t *testing.T, extraEnv []string) blockedPairingFixture {
		t.Helper()
		f := blockedPairingFixture{aDir: shortClusterTempDir(t), bDir: shortClusterTempDir(t), aPort: temporaryClusterPort(t), bPort: temporaryClusterPort(t)}
		f.a = startCMEnv(t, f.aDir, f.aPort, append([]string{"NVPAIR_TEST_PAIR_REJECT_INBOUND=1"}, extraEnv...))
		f.b = startCMEnv(t, f.bDir, f.bPort, extraEnv)
		t.Cleanup(f.a.stop)
		t.Cleanup(f.b.stop)
		f.aID = decodeResult[cmNodeID](t, f.a.call("cluster:get-node-id", nil))
		f.bID = decodeResult[cmNodeID](t, f.b.call("cluster:get-node-id", nil))
		waitTCP(t, "127.0.0.1:"+strconv.Itoa(f.aPort), 10*time.Second)
		waitTCP(t, "127.0.0.1:"+strconv.Itoa(f.bPort), 10*time.Second)
		f.a.call("cluster:create", map[string]any{"clusterFriendlyName": "unfinished permission fixture"})
		f.invite = decodeResult[outboundInviteResult](t, f.a.call("cluster:invite-node", map[string]any{"address": "127.0.0.1", "port": f.bPort, "allowRemoteManagement": true}))
		if f.invite.State != "pending" || len(f.invite.Pin) != 6 {
			t.Fatal("invitation did not enter pending state")
		}
		f.b.waitNotify("cluster:invite-received")
		return f
	}
	assertDenied := func(t *testing.T, f blockedPairingFixture) {
		t.Helper()
		for _, target := range []struct {
			cm        *cmProc
			dir, peer string
		}{{f.a, f.aDir, f.bID.NodeUUID}, {f.b, f.bDir, f.aID.NodeUUID}} {
			mesh := clustertrust.Open(filepath.Join(target.dir, "cluster"))
			if mesh.HasPin(target.peer) || mesh.RemoteManagement(target.peer).Allowed {
				t.Fatal("unfinished pairing retained peer authority")
			}
			view := decodeResult[managementSnapshot](t, target.cm.call("cluster:get-remote-management", map[string]any{}))
			for _, grant := range view.Grants {
				if grant.PeerUUID == target.peer && grant.Allowed {
					t.Fatal("unfinished pairing published a local allow")
				}
			}
		}
	}
	t.Run("required choice and wrong PIN", func(t *testing.T) {
		f := newFixture(t, nil)
		missing := f.b.callExpectError("cluster:respond-to-invite", map[string]any{"inviteId": f.invite.InviteID, "accept": true, "pin": f.invite.Pin})
		if missing.Error == nil || missing.Error.Code != -32602 {
			t.Fatal("missing local pairing choice did not fail with -32602")
		}
		wrong := "000000"
		if f.invite.Pin == wrong {
			wrong = "000001"
		}
		result := decodeResult[outboundInviteResult](t, f.b.call("cluster:respond-to-invite", map[string]any{"inviteId": f.invite.InviteID, "accept": true, "pin": wrong, "allowRemoteManagement": true}))
		if result.State != "failed" || result.Reason != "incorrect-pin" {
			t.Fatal("wrong PIN did not return failed/incorrect-pin")
		}
		waitInviteState(t, f.a, f.invite.InviteID, "failed")
		waitInviteState(t, f.b, f.invite.InviteID, "failed")
		assertDenied(t, f)
	})
	t.Run("decline", func(t *testing.T) {
		f := newFixture(t, nil)
		f.b.call("cluster:respond-to-invite", map[string]any{"inviteId": f.invite.InviteID, "accept": false})
		waitInviteState(t, f.a, f.invite.InviteID, "declined")
		waitInviteState(t, f.b, f.invite.InviteID, "declined")
		assertDenied(t, f)
	})
	t.Run("expiry", func(t *testing.T) {
		f := newFixture(t, []string{"NVPAIR_TEST_INVITE_TTL_MS=500", "NVPAIR_TEST_INVITE_EXPIRY_INTERVAL_MS=25"})
		waitInviteState(t, f.a, f.invite.InviteID, "expired")
		waitInviteState(t, f.b, f.invite.InviteID, "expired")
		assertDenied(t, f)
	})
	t.Run("superseded allow choice cannot be accepted", func(t *testing.T) {
		f := newFixture(t, nil)
		second := decodeResult[outboundInviteResult](t, f.a.call("cluster:invite-node", map[string]any{"address": "127.0.0.1", "port": f.bPort, "allowRemoteManagement": false}))
		if second.State != "pending" || second.InviteID == f.invite.InviteID {
			t.Fatal("replacement invitation did not create a fresh pending session")
		}
		waitInviteState(t, f.a, f.invite.InviteID, "declined")
		waitInviteState(t, f.b, f.invite.InviteID, "canceled")
		stale := f.b.callExpectError("cluster:respond-to-invite", map[string]any{"inviteId": f.invite.InviteID, "accept": true, "pin": f.invite.Pin, "allowRemoteManagement": true})
		if stale.Error == nil {
			t.Fatal("superseded invitation accepted its obsolete allow choice")
		}
		assertDenied(t, f)
		f.b.call("cluster:respond-to-invite", map[string]any{"inviteId": second.InviteID, "accept": false})
		waitInviteState(t, f.a, second.InviteID, "declined")
		assertDenied(t, f)
	})
}

func TestRemoteManagementDeniedPeerCanInfer(t *testing.T) {
	f := newPhase04Fixture(t)
	a, b := f.control.a, f.control.b
	managementSet(t, b.cm, a.uuid, false)
	managementSet(t, a.cm, b.uuid, false)
	for _, node := range []*task1Node{a, b} {
		if clustertrust.Open(filepath.Join(node.base, "cluster")).RemoteManagement(f.peer(node).uuid).Allowed {
			t.Fatal("inference control requires effective denied management")
		}
	}
	t.Run("denied peer uses reverse inference", func(t *testing.T) { phase04AssertOrdered(t, f, a, "management-denied-reverse") })
	t.Run("denied peer uses direct inference", func(t *testing.T) { phase04AssertOrdered(t, f, b, "management-denied-direct") })
}

func TestRemoteManagementDeniedPeerCanReadModelExports(t *testing.T) {
	f := newPhase04Fixture(t)
	a, b := f.control.a, f.control.b
	managementSet(t, b.cm, a.uuid, false)
	managementSet(t, a.cm, b.uuid, false)
	test := func(name, model string, source, destination *task1Node) {
		t.Run(name, func(t *testing.T) {
			scenario := phase04SeedModel(t, f.stores[source.uuid], f.stores[destination.uuid], model, 37)
			// Copy is a local destination action. The denied peer only reads
			// the source's real ec file list and file export handlers.
			response := f.control.engines[destination.uuid].call(t, "engine:remote-copy-model", map[string]string{"node": source.uuid, "engine": "ollama", "model": model + ":latest"})
			var result struct {
				Result struct {
					Files       int   `json:"files"`
					BytesCopied int64 `json:"bytesCopied"`
				} `json:"result"`
			}
			task2Decode(t, response.Result, &result)
			if result.Result.Files != len(scenario.data) || result.Result.BytesCopied <= 0 {
				t.Fatalf("exported files=%d bytes=%d want=%d files with copied bytes", result.Result.Files, result.Result.BytesCopied, len(scenario.data))
			}
			for path, want := range scenario.data {
				for _, root := range []string{f.stores[source.uuid], f.stores[destination.uuid]} {
					got, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(got, want) {
						t.Fatal("source export or destination copy differs from synthetic content")
					}
				}
			}
		})
	}
	test("denied peer reads exports over reverse", "management-export-reverse", b, a)
	test("denied peer reads exports directly", "management-export-direct", a, b)
}

func TestRemoteManagementGrantPreservesLocalPause(t *testing.T) {
	f := newManagementFixture(t)
	f.permission(t, f.joint.client, true)
	response, err := f.joint.client.Post("https://127.0.0.1:"+strconv.Itoa(f.joint.ports[1])+"/v1/engines/start", "application/json", bytes.NewReader(mustMarshal(t, map[string]string{"engine": "ollama"})))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var refusal struct {
		Code string `json:"code"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 4096)).Decode(&refusal); err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusConflict || refusal.Code != "local_veto" {
		t.Fatalf("granted peer under local pause: HTTP %d code=%s, want 409/local_veto", response.StatusCode, refusal.Code)
	}
	f.joint.assertPolicy(t)
}

func TestRemoteManagementEndorsementAndReadmission(t *testing.T) {
	aDir, bDir, cDir := shortClusterTempDir(t), shortClusterTempDir(t), shortClusterTempDir(t)
	aPort, bPort, cPort := temporaryClusterPort(t), temporaryClusterPort(t), temporaryClusterPort(t)
	a, b, c := startCM(t, aDir, aPort), startCM(t, bDir, bPort), startCM(t, cDir, cPort)
	t.Cleanup(a.stop)
	t.Cleanup(b.stop)
	t.Cleanup(c.stop)
	a.call("cluster:get-node-id", nil)
	bID := decodeResult[cmNodeID](t, b.call("cluster:get-node-id", nil))
	cID := decodeResult[cmNodeID](t, c.call("cluster:get-node-id", nil))
	waitTCP(t, "127.0.0.1:"+strconv.Itoa(aPort), 10*time.Second)
	waitTCP(t, "127.0.0.1:"+strconv.Itoa(bPort), 10*time.Second)
	waitTCP(t, "127.0.0.1:"+strconv.Itoa(cPort), 10*time.Second)
	a.call("cluster:create", map[string]any{"clusterFriendlyName": "permission endorsement fixture"})
	pairNodes(t, a, b, bPort, "management-b")
	pairNodes(t, a, c, cPort, "management-c")
	waitMembers(t, b, 3)
	waitMembers(t, c, 3)
	mesh := clustertrust.Open(filepath.Join(bDir, "cluster"))
	task1Eventually(t, "endorsed admission durable and unreviewed", 10*time.Second, func() bool {
		mesh.Refresh()
		view := mesh.RemoteManagement(cID.NodeUUID)
		if view.Allowed {
			t.Fatal("endorsement granted management")
		}
		return mesh.HasPin(cID.NodeUUID) && memberUUIDs(t, b)[cID.NodeUUID] && view.State == "unreviewed"
	})
	writer := managementPolicyWriterAt(t, bDir)
	newCaller := func() *http.Client {
		t.Helper()
		callerMesh := clustertrust.Open(filepath.Join(cDir, "cluster"))
		cfg, ok := callerMesh.ClientTLSConfig(bID.NodeUUID)
		if !ok || !callerMesh.Clustered() {
			t.Fatal("endorsed caller lacks current target admission and TLS authority")
		}
		client := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{TLSClientConfig: cfg}}
		t.Cleanup(client.CloseIdleConnections)
		return client
	}
	caller := newCaller()
	readTarget := func(t *testing.T, client *http.Client) {
		t.Helper()
		data := managementHTTP(t, client, writer.ports[1], http.MethodPost, "/v1/node-policy/get", map[string]any{}, http.StatusOK)
		var remote struct {
			Policy nodepolicy.PeerPolicy `json:"policy"`
		}
		task2Decode(t, data, &remote)
		var actual struct {
			Policy nodepolicy.Policy `json:"policy"`
		}
		response := callRPC(t, writer.target.in, writer.target.msgs, "policy:get", map[string]any{}, 10*time.Second)
		if response.Error != nil {
			t.Fatal("read B's actual broker policy")
		}
		task2Decode(t, response.Result, &actual)
		if remote.Policy.Revision == "" || !reflect.DeepEqual(remote.Policy, actual.Policy.PeerProjection()) {
			t.Fatal("C's ec read is not B's actual broker policy")
		}
	}
	t.Run("endorsed peer reads but cannot mutate before local grant", func(t *testing.T) {
		readTarget(t, caller)
		managementPolicyAttempt(t, writer.target, writer.path, caller, writer.ports[1], http.StatusConflict)
	})
	beforePin := readOutboundJSON[struct {
		AdmissionEpoch uint64 `json:"admissionEpoch"`
	}](t, filepath.Join(bDir, "cluster", "trusted", cID.NodeUUID+".json"))
	snapshot := decodeResult[managementSnapshot](t, b.call("cluster:get-remote-management", map[string]any{}))
	b.call("cluster:set-remote-management", map[string]any{"peerUuid": cID.NodeUUID, "allowed": true, "expectedRevision": snapshot.Revision})
	if !mesh.RemoteManagement(cID.NodeUUID).Allowed {
		t.Fatal("local grant positive control did not allow endorsed peer")
	}
	t.Run("local grant permits endorsed peer mutation", func(t *testing.T) {
		readTarget(t, caller)
		managementPolicyAttempt(t, writer.target, writer.path, caller, writer.ports[1], http.StatusOK)
	})
	a.call("nodes:remove", map[string]any{"nodeUuid": cID.NodeUUID})
	waitMembers(t, b, 2)
	waitMembers(t, c, 0)
	if mesh.RemoteManagement(cID.NodeUUID).Allowed {
		t.Fatal("removed peer retained effective management")
	}
	t.Run("removed peer cannot read or mutate target", func(t *testing.T) {
		caller.CloseIdleConnections()
		managementHTTP(t, caller, writer.ports[1], http.MethodPost, "/v1/node-policy/get", map[string]any{}, http.StatusForbidden)
		managementPolicyAttempt(t, writer.target, writer.path, caller, writer.ports[1], http.StatusForbidden)
	})
	pairNodes(t, a, c, cPort, "management-c-readmitted")
	waitMembers(t, b, 3)
	waitMembers(t, c, 3)
	task1Eventually(t, "new admission endorsement durably replaces the removed pin", 20*time.Second, func() bool {
		data, err := os.ReadFile(filepath.Join(bDir, "cluster", "trusted", cID.NodeUUID+".json"))
		if os.IsNotExist(err) || isWindowsSharingError(err) {
			return false
		}
		if err != nil {
			t.Fatal(err)
		}
		var pin struct {
			AdmissionEpoch uint64 `json:"admissionEpoch"`
		}
		if err := json.Unmarshal(data, &pin); err != nil {
			t.Fatal(err)
		}
		mesh.Refresh()
		if mesh.RemoteManagement(cID.NodeUUID).Allowed {
			t.Fatal("readmission resurrected management")
		}
		return pin.AdmissionEpoch > beforePin.AdmissionEpoch && mesh.RemoteManagement(cID.NodeUUID).State == "unreviewed"
	})
	if !mesh.HasPin(cID.NodeUUID) || !memberUUIDs(t, b)[cID.NodeUUID] {
		t.Fatal("readmitted peer is not trusted")
	}
	if mesh.RemoteManagement(cID.NodeUUID).Allowed {
		t.Fatal("readmission resurrected the previous admission's grant")
	}
	if !memberUUIDs(t, c)[bID.NodeUUID] {
		t.Fatal("readmitted peer did not receive the existing member endorsement")
	}
	caller.CloseIdleConnections()
	caller = newCaller()
	t.Run("fresh admission reads but cannot reuse old grant", func(t *testing.T) {
		readTarget(t, caller)
		managementPolicyAttempt(t, writer.target, writer.path, caller, writer.ports[1], http.StatusConflict)
	})
	snapshot = decodeResult[managementSnapshot](t, b.call("cluster:get-remote-management", map[string]any{}))
	b.call("cluster:set-remote-management", map[string]any{"peerUuid": cID.NodeUUID, "allowed": true, "expectedRevision": snapshot.Revision})
	if !mesh.RemoteManagement(cID.NodeUUID).Allowed {
		t.Fatal("fresh local grant did not authorize new admission")
	}
	t.Run("fresh local grant permits readmitted peer mutation", func(t *testing.T) {
		readTarget(t, caller)
		managementPolicyAttempt(t, writer.target, writer.path, caller, writer.ports[1], http.StatusOK)
	})
}

func TestRemoteManagementPersistsAcrossSeparateProcessRestarts(t *testing.T) {
	rf := newRemoteServiceFixture(t)
	f, client, allowedClient, enginePath, clusterPath := rf.serviceFixture, rf.peer, rf.allowed, rf.enginePath, rf.clusterPath
	f.start()
	c := f.readyClient()
	view := decodeResult[managementSnapshot](t, *c.mustCall(t, "701", "cluster:get-remote-management", mustMarshal(t, map[string]any{})))
	c.mustCall(t, "702", "cluster:set-remote-management", mustMarshal(t, map[string]any{
		"peerUuid": "management-service-peer", "allowed": false, "expectedRevision": view.Revision,
	}))
	view = decodeResult[managementSnapshot](t, *c.mustCall(t, "704", "cluster:get-remote-management", mustMarshal(t, map[string]any{})))
	c.mustCall(t, "705", "cluster:set-remote-management", mustMarshal(t, map[string]any{
		"peerUuid": "management-service-allowed", "allowed": true, "expectedRevision": view.Revision,
	}))
	permissionPath := rf.permissionPath
	want := readLiveDurableFile(t, permissionPath)
	assertOppositeGrants := func(t *testing.T) {
		t.Helper()
		client.CloseIdleConnections()
		allowedClient.CloseIdleConnections()
		managementHTTP(t, client, f.ports.engineControl, http.MethodPost, "/v1/engines/stop", map[string]any{"engine": "ollama"}, http.StatusConflict)
		managementHTTP(t, client, f.ports.engineControl, http.MethodGet, "/v1/engines", nil, http.StatusOK)
		for _, caller := range []struct {
			client  *http.Client
			allowed bool
		}{{client, false}, {allowedClient, true}} {
			var capability struct {
				Allowed bool `json:"allowed"`
			}
			task2Decode(t, managementHTTP(t, caller.client, f.ports.engineControl, http.MethodGet, "/v1/remote-management", nil, http.StatusOK), &capability)
			if capability.Allowed != caller.allowed {
				t.Fatal("restart changed an authenticated caller's effective grant")
			}
		}
		before := decodeServicePolicy(t, c.mustCall(t, "706", "policy:get", nil))
		assertStoredPolicy(t, f.policyPath(), before)
		beforeBytes := readPolicyBytes(t, f.policyPath())
		draft := before.PeerProjection()
		draft.Idle.UnloadAfterMinutes++
		params := map[string]any{"policy": draft, "expectedPersistedRevision": before.Revision}
		managementHTTP(t, client, f.ports.engineControl, http.MethodPost, "/v1/node-policy/set", params, http.StatusConflict)
		if !bytes.Equal(readPolicyBytes(t, f.policyPath()), beforeBytes) {
			t.Fatal("denied otherwise-valid post-restart mutation changed policy")
		}
		var accepted struct {
			Policy nodepolicy.PeerPolicy `json:"policy"`
		}
		task2Decode(t, managementHTTP(t, allowedClient, f.ports.engineControl, http.MethodPost, "/v1/node-policy/set", params, http.StatusOK), &accepted)
		if accepted.Policy.Revision == "" || accepted.Policy.Revision == before.Revision {
			t.Fatal("allowed post-restart mutation did not advance policy revision")
		}
		wantPolicy := before
		wantPolicy.Idle = draft.Idle
		wantPolicy.Revision = accepted.Policy.Revision
		draft.Revision = accepted.Policy.Revision
		if !reflect.DeepEqual(accepted.Policy, draft) {
			t.Fatal("allowed post-restart mutation returned the wrong policy")
		}
		actual := decodeServicePolicy(t, c.mustCall(t, "707", "policy:get", nil))
		if !reflect.DeepEqual(actual, wantPolicy) {
			t.Fatal("allowed mutation did not reach the actual restarted broker writer")
		}
		assertStoredPolicy(t, f.policyPath(), wantPolicy)
		// Try a fresh valid revision after the allowed writer, not its stale CAS.
		draft.Idle.UnloadAfterMinutes++
		managementHTTP(t, client, f.ports.engineControl, http.MethodPost, "/v1/node-policy/set", map[string]any{"policy": draft, "expectedPersistedRevision": actual.Revision}, http.StatusConflict)
		actual = decodeServicePolicy(t, c.mustCall(t, "708", "policy:get", nil))
		if !reflect.DeepEqual(actual, wantPolicy) {
			t.Fatal("denied peer changed the restarted writer after the allowed control")
		}
		assertStoredPolicy(t, f.policyPath(), wantPolicy)
		if !bytes.Equal(readLiveDurableFile(t, permissionPath), want) {
			t.Fatal("process restart or remote mutation changed the explicit opposite local grants")
		}
	}
	t.Run("before restart", func(t *testing.T) { assertOppositeGrants(t) })
	for _, target := range []struct{ name, path string }{{"cluster-manager", clusterPath}, {"engine-manager", enginePath}, {"broker", f.brokerPath}} {
		t.Run(target.name, func(t *testing.T) {
			var old serviceProcessIdentity
			for _, identity := range f.scan() {
				if identity.Path == target.path {
					old = identity
				}
			}
			if old.PID == 0 {
				t.Fatal("owned restart target is absent")
			}
			if err := f.inventory.kill(old); err != nil {
				t.Fatalf("terminate owned restart target: %v", err)
			}
			task1Eventually(t, "replacement "+target.name, 20*time.Second, func() bool {
				for _, identity := range f.scan() {
					if identity.Path == target.path && identity != old {
						return true
					}
				}
				return false
			})
			waitTCP(t, "127.0.0.1:"+strconv.Itoa(f.ports.engineControl), 10*time.Second)
			assertOppositeGrants(t)
		})
	}
	t.Run("service", func(t *testing.T) {
		c.mustClose(t)
		c = f.readyClient()
		var oldService serviceProcessIdentity
		for _, identity := range f.scan() {
			if identity.Path == f.servicePath {
				oldService = identity
			}
		}
		if oldService.PID == 0 {
			t.Fatal("owned service restart target is absent")
		}
		assertServiceStoppedReply(t, c.mustCall(t, "703", "service/stop", nil))
		f.waitChild(f.service, serviceStopTimeout)
		if f.service.err != nil {
			t.Fatalf("owned service stop: %v", f.service.err)
		}
		f.assertStopped()
		f.start()
		c = f.readyClient()
		task1Eventually(t, "replacement service", 20*time.Second, func() bool {
			for _, identity := range f.scan() {
				if identity.Path == f.servicePath && identity != oldService {
					return true
				}
			}
			return false
		})
		assertOppositeGrants(t)
	})
}
