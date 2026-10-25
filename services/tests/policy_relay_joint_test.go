// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-shared/testbin"
)

// The origin is a real engine-manager with a fixture discovery directory. The
// target is a real engine-manager supervised by the real broker: its inbound
// mTLS policy routes send policy:request over child stdio, and only the broker
// loads, merges and persists the policy before returning policy:reply. The sole
// unrelated worker is the empty-directory subscription scanner fixture. No
// policy response is synthesized by a fixture.
func TestPolicyRelayJoint(t *testing.T) {
	f := newPolicyJointFixture(t, jointAvailable)
	if !t.Run("peer get excludes routing and every API entry", func(t *testing.T) {
		f.assertPolicy(t)
		body := f.http(t, "/v1/node-policy/get", map[string]any{}, http.StatusOK)
		assertJointPeer(t, body, f.want)
		assertJointPeer(t, f.rpc(t, "engine:remote-policy-get", map[string]any{}).Result, f.want)
	}) {
		return
	}
	for _, edit := range []bool{false, true} {
		name := "unchanged projection preserves complete protected policy"
		if edit {
			name = "device edit preserves complete protected policy"
		}
		if !t.Run(name, func(t *testing.T) {
			body := f.http(t, "/v1/node-policy/get", map[string]any{}, http.StatusOK)
			assertJointPeer(t, body, f.want)
			var projection struct {
				Policy nodepolicy.PeerPolicy `json:"policy"`
			}
			if err := json.Unmarshal(body, &projection); err != nil {
				t.Fatal("decode fetched peer projection for round-trip")
			}
			peer := projection.Policy
			want := f.want
			if edit {
				peer.Admission.MaxConcurrentPerModel = 3
				peer.Idle.UnloadAfterMinutes = 7
				// An older peer's false pause switch is accepted and saved as
				// the canonical true; the drain timeout stays editable.
				peer.Pause.UnloadModels = false
				peer.Pause.DrainTimeoutSeconds = 45
				want.Admission = peer.Admission
				want.Idle = peer.Idle
				want.Pause = peer.Pause
				want.Pause.UnloadModels = true
			}
			params := map[string]any{"policy": peer, "expectedPersistedRevision": f.want.Revision}
			started := time.Now().UnixNano()
			var result []byte
			if edit {
				result = f.rpc(t, "engine:remote-policy-set", params).Result
			} else {
				result = f.http(t, "/v1/node-policy/set", params, http.StatusOK)
			}
			var envelope struct {
				Policy nodepolicy.PeerPolicy `json:"policy"`
			}
			if err := json.Unmarshal(result, &envelope); err != nil {
				t.Fatal("successful peer set did not return a policy")
			}
			assertNextRevision(t, f.want.Revision, envelope.Policy.Revision, started, time.Now().UnixNano())
			want.Revision = envelope.Policy.Revision
			assertJointPeer(t, result, want)
			f.want = want
			f.assertPolicy(t)
		}) {
			return
		}
	}

	const required = "expectedPersistedRevision is required: send the revision from the preceding policy:get"
	const malformed = "expectedPersistedRevision must be a canonical decimal string: 0 or a nonzero leading digit, within the signed 64-bit range"
	const routing = "a paired node cannot change the routing configuration"
	conflict := "policy conflict: the persisted revision is " + f.want.Revision + ", not the expected 0; reload the policy and review it before saving"
	cases := []struct {
		name   string
		change func(*testing.T, map[string]any)
		status int
		code   int
		text   string
	}{
		{"changed local tier", func(t *testing.T, p map[string]any) {
			peer := jointExpectedPeer(f.want)
			peer.Tiers["weak"][0].Model = "different-local"
			p["policy"] = peer
		}, http.StatusBadRequest, -32602, "a paired node cannot change the tiers"},
		{"changed tier fallback", func(t *testing.T, p map[string]any) {
			peer := jointExpectedPeer(f.want)
			peer.TierPolicy.AllowWeaker = !peer.TierPolicy.AllowWeaker
			p["policy"] = peer
		}, http.StatusBadRequest, -32602, "a paired node cannot change the tier policy"},
		{"introduced empty routing", func(t *testing.T, p map[string]any) {
			f.cloudDraft(t, p, map[string]any{})
		}, http.StatusBadRequest, -32602, routing},
		{"introduced unchanged cloud configuration", func(t *testing.T, p map[string]any) {
			f.cloudDraft(t, p, f.want.Routing)
		}, http.StatusBadRequest, -32602, routing},
		{"changed account deployment limits and listener", func(t *testing.T, p map[string]any) {
			var cloud nodepolicy.Routing
			if err := json.Unmarshal(mustMarshal(t, f.want.Routing), &cloud); err != nil {
				t.Fatal("copy cloud fixture")
			}
			cloud.Accounts[0].Credential = "NVPAIR_PROVIDER_SECRET_JOINT_OTHER_REFERENCE"
			cloud.Deployments[0].Model = "different-cloud"
			cloud.Limits.MaxAttempts++
			cloud.Listener.AllowedOrigins = []string{"http://localhost:4000"}
			f.cloudDraft(t, p, cloud)
		}, http.StatusBadRequest, -32602, routing},
		{"missing revision", func(t *testing.T, p map[string]any) { delete(p, "expectedPersistedRevision") }, http.StatusBadRequest, -32602, required},
		{"null revision", func(t *testing.T, p map[string]any) { p["expectedPersistedRevision"] = nil }, http.StatusBadRequest, -32602, required},
		{"empty revision", func(t *testing.T, p map[string]any) { p["expectedPersistedRevision"] = "" }, http.StatusBadRequest, -32602, required},
		{"noncanonical revision", func(t *testing.T, p map[string]any) { p["expectedPersistedRevision"] = "01" }, http.StatusBadRequest, -32602, malformed},
		{"nonnumeric revision", func(t *testing.T, p map[string]any) { p["expectedPersistedRevision"] = "invalid" }, http.StatusBadRequest, -32602, malformed},
		{"overflow revision", func(t *testing.T, p map[string]any) { p["expectedPersistedRevision"] = "9223372036854775808" }, http.StatusBadRequest, -32602, malformed},
		{"stale revision", func(t *testing.T, p map[string]any) { p["expectedPersistedRevision"] = "0" }, http.StatusConflict, -32010, conflict},
		{"stale revision precedes protected draft rejection", func(t *testing.T, p map[string]any) {
			p["expectedPersistedRevision"] = "0"
			f.cloudDraft(t, p, map[string]any{})
		}, http.StatusConflict, -32010, conflict},
	}
	for _, tc := range cases {
		before := assertStoredPolicy(t, f.path, f.want)
		if !t.Run(tc.name, func(t *testing.T) {
			params := map[string]any{"policy": jointExpectedPeer(f.want), "expectedPersistedRevision": f.want.Revision}
			tc.change(t, params)
			body := f.http(t, "/v1/node-policy/set", params, tc.status)
			wantBody := tc.text + "\n"
			if tc.code == -32010 {
				// A revision conflict is a typed refusal the origin maps back
				// to its code.
				wantBody = string(mustMarshal(t, struct {
					Error string `json:"error"`
					Code  string `json:"code"`
				}{tc.text, "policy_conflict"})) + "\n"
			}
			if string(body) != wantBody {
				t.Fatal("HTTP rejection text differs from the independently specified error")
			}
			f.assertUnchanged(t, before)
			response := f.originCall(t, "engine:remote-policy-set", params)
			wantMessage := fmt.Sprintf("remote node policy: HTTP %d: %s", tc.status, tc.text)
			if response.Error == nil || response.Error.Code != tc.code || response.Error.Message != wantMessage || len(response.Result) != 0 {
				t.Fatalf("origin RPC classification mismatch: want_code=%d error_present=%t result_bytes=%d", tc.code, response.Error != nil, len(response.Result))
			}
			f.assertUnchanged(t, before)
		}) {
			return
		}
		// A fresh broker must read the same bytes after both rejected requests;
		// graceful teardown also flushes any mistakenly queued persistence.
		// Restart the caller too, so this assertion reads a cold target rather
		// than exercising recovery of a pooled connection to an exited server.
		f.origin.cleanup()
		f.target.cleanup()
		f.client.CloseIdleConnections()
		f.startTarget(t)
		f.startOrigin(t)
		if !t.Run(tc.name+" cold reload", func(t *testing.T) {
			f.assertUnchanged(t, before)
			assertJointPeer(t, f.rpc(t, "engine:remote-policy-get", map[string]any{}).Result, f.want)
		}) {
			return
		}
	}
	t.Run("numeric revision is rejected at HTTP and origin RPC decoders", func(t *testing.T) {
		before := assertStoredPolicy(t, f.path, f.want)
		params := map[string]any{"policy": jointExpectedPeer(f.want), "expectedPersistedRevision": 12}
		if body := f.http(t, "/v1/node-policy/set", params, http.StatusBadRequest); string(body) != "invalid policy request\n" {
			t.Fatal("numeric revision HTTP decoder error mismatch")
		}
		response := f.originCall(t, "engine:remote-policy-set", params)
		const message = "invalid params: json: cannot unmarshal number into Go struct field policyBody.expectedPersistedRevision of type string"
		if response.Error == nil || response.Error.Code != -32602 || response.Error.Message != message || len(response.Result) != 0 {
			t.Fatal("numeric revision origin RPC decoder classification mismatch")
		}
		f.assertUnchanged(t, before)
	})
	// Pause locally after the ordinary policy writes: a paired node cannot
	// resume it, and the node's own resume still can.
	t.Run("local pause vetoes a relayed resume until the node resumes itself", func(t *testing.T) {
		local := callRPC(t, f.target.in, f.target.msgs, "node:set-availability", map[string]any{"state": "paused"}, 15*time.Second)
		if local.Error != nil {
			t.Fatalf("local pause before relayed resume: %+v", local.Error)
		}
		var paused nodepolicy.Policy
		if err := json.Unmarshal(readPolicyBytes(t, f.path), &paused); err != nil {
			t.Fatalf("decode locally paused policy: %v", err)
		}
		want := f.want
		want.Availability, want.PausedBy, want.Revision = nodepolicy.Paused, nodepolicy.PausedLocally, paused.Revision
		if paused.Revision == f.want.Revision || !reflect.DeepEqual(paused, want) {
			t.Fatal("local pause changed protected policy or failed to advance its revision")
		}
		f.want = want
		before := assertStoredPolicy(t, f.path, f.want)
		const veto = "This node is paused. Resume it on this node before managing it remotely."
		body := f.http(t, "/v1/node-availability/set", map[string]any{"state": "available"}, http.StatusConflict)
		var refusal struct {
			Error string `json:"error"`
			Code  string `json:"code"`
		}
		if json.Unmarshal(body, &refusal) != nil || refusal.Code != "local_veto" || refusal.Error != veto {
			t.Fatalf("relayed resume refusal = %s, want a local_veto body", body)
		}
		f.assertUnchanged(t, before)
		response := f.originCall(t, "engine:remote-availability-set", map[string]any{"state": "available"})
		if response.Error == nil || response.Error.Code != -32011 || response.Error.Message != "remote node policy: HTTP 409: "+veto {
			t.Fatalf("origin resume of a local pause = %+v, want code -32011", response.Error)
		}
		f.assertUnchanged(t, before)
		local = callRPC(t, f.target.in, f.target.msgs, "node:set-availability", map[string]any{"state": "available"}, 15*time.Second)
		if local.Error != nil {
			t.Fatalf("local resume after a vetoed relayed resume: %+v", local.Error)
		}
		var stored nodepolicy.Policy
		if err := json.Unmarshal(readPolicyBytes(t, f.path), &stored); err != nil || stored.Availability != nodepolicy.Available || stored.PausedBy != "" {
			t.Fatalf("local resume stored availability %q pausedBy %q (%v), want available with no provenance", stored.Availability, stored.PausedBy, err)
		}
	})
}

func TestPolicyRelayJointModelConfiguration(t *testing.T) {
	f := newPolicyJointFixture(t, jointAvailable)
	peer := jointExpectedPeer(f.want)
	peer.Profiles = append(peer.Profiles, fixtureResidentProfile())
	peer.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: "sr-status-fixture"}}
	peer.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "sr-status-fixture"}}
	result := f.rpc(t, "engine:remote-policy-set", map[string]any{
		"policy": peer, "expectedPersistedRevision": f.want.Revision,
	}).Result
	var accepted struct {
		Policy nodepolicy.PeerPolicy `json:"policy"`
	}
	if err := json.Unmarshal(result, &accepted); err != nil {
		t.Fatalf("decode paired resident declaration: %v", err)
	}
	if accepted.Policy.Revision == "" || accepted.Policy.Revision == f.want.Revision {
		t.Fatal("paired resident declaration did not advance the policy revision")
	}
	f.want.Revision = accepted.Policy.Revision
	f.want.Profiles = peer.Profiles
	f.want.ServedModels = peer.ServedModels
	f.want.ResidentModels = peer.ResidentModels
	assertJointPeer(t, result, f.want)
	assertFixtureResidentConfiguration(t, result, f.want.Revision)
	f.assertPolicy(t)

	local := callRPC(t, f.target.in, f.target.msgs, "policy:get", map[string]any{}, 10*time.Second)
	if local.Error != nil {
		t.Fatalf("local resident readback: %+v", local.Error)
	}
	localConfiguration := decodeFixtureModelConfiguration(t, local.Result, f.want.Revision)
	for _, readback := range []struct {
		name string
		raw  []byte
	}{
		{"paired HTTP", f.http(t, "/v1/node-policy/get", map[string]any{}, http.StatusOK)},
		{"paired manager RPC", f.rpc(t, "engine:remote-policy-get", map[string]any{}).Result},
	} {
		t.Run(readback.name, func(t *testing.T) {
			assertJointPeer(t, readback.raw, f.want)
			assertFixtureResidentConfiguration(t, readback.raw, f.want.Revision)
			configuration := decodeFixtureModelConfiguration(t, readback.raw, f.want.Revision)
			if !reflect.DeepEqual(configuration, localConfiguration) {
				t.Fatal("paired model configuration differs from target local readback")
			}
		})
	}

	peer = jointExpectedPeer(f.want)
	peer.ServedModels = nodepolicy.ServedModels{}
	peer.ResidentModels = nil
	result = f.http(t, "/v1/node-policy/set", map[string]any{
		"policy": peer, "expectedPersistedRevision": f.want.Revision,
	}, http.StatusOK)
	if err := json.Unmarshal(result, &accepted); err != nil {
		t.Fatalf("decode paired empty served declaration: %v", err)
	}
	f.want.Revision = accepted.Policy.Revision
	f.want.ServedModels = peer.ServedModels
	f.want.ResidentModels = nil
	assertJointPeer(t, result, f.want)
	configuration := decodeFixtureModelConfiguration(t, result, f.want.Revision)
	if len(configuration.Models) != 0 {
		t.Fatal("paired explicit empty served set retained removed model configuration")
	}
	assertJointPeer(t, f.rpc(t, "engine:remote-policy-get", map[string]any{}).Result, f.want)
	f.assertPolicy(t)
}

type policyJointFixture struct {
	inertPort                                           int
	inert                                               *httptest.Server
	base, cluster, engine, scanner, originCluster, path string
	originBase, originEngine                            string
	ports                                               []int
	target, origin                                      *interopProcess
	client                                              *http.Client
	want                                                nodepolicy.Policy
}

type policyJointFixtureOption func(*nodepolicy.Policy)

// jointAvailable opts mutation fixtures into an available node. Status and
// local-pause consumers retain the seeded pause when no option is supplied.
func jointAvailable(policy *nodepolicy.Policy) {
	policy.Availability = nodepolicy.Available
	policy.PausedBy = ""
}

func newPolicyJointFixture(t *testing.T, options ...policyJointFixtureOption) *policyJointFixture {
	t.Helper()
	f := &policyJointFixture{base: shortClusterTempDir(t)}
	originBase := shortClusterTempDir(t)
	f.originBase = originBase
	f.cluster = filepath.Join(f.base, "cluster")
	f.originCluster = filepath.Join(originBase, "cluster")
	for _, dir := range []string{f.cluster, f.originCluster} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	const targetID, originID = "policy-joint-target", "policy-joint-origin"
	targetCert := mintClusterIdentity(t, f.cluster, targetID)
	originCert := mintClusterIdentity(t, f.originCluster, originID)
	for _, member := range []struct{ dir, id, peerID, cert string }{
		{f.cluster, targetID, originID, originCert}, {f.originCluster, originID, targetID, targetCert},
	} {
		writeActiveAdmission(t, member.dir)
		writePin(t, member.dir, member.peerID, member.cert)
		writeMembers(t, member.dir, targetID, originID)
		jointWriteJSON(t, filepath.Join(member.dir, "identity.json"), map[string]any{"node_uuid": member.id})
	}
	managementGrantFixture(t, f.cluster, targetID, originID)
	mesh := clustertrust.Open(f.originCluster)
	cfg, ok := mesh.ClientTLSConfig(targetID)
	if !ok || !mesh.Clustered() {
		t.Fatal("fixture cluster did not provide pinned target TLS")
	}
	f.client = &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{TLSClientConfig: cfg}}
	t.Cleanup(f.client.CloseIdleConnections)

	// Every bundled engine is made absent and its probes point at an inert
	// loopback fixture. Even a background inventory poll cannot contact or
	// adopt an installed engine, its default port, or the owner's model store.
	inert := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	if err := inert.Listener.Close(); err != nil {
		t.Fatal(err)
	}
	inert.Listener = jointListener(t)
	inert.Start()
	t.Cleanup(inert.Close)
	f.inert = inert
	inertPort := inert.Listener.Addr().(*net.TCPAddr).Port
	f.inertPort = inertPort
	for _, base := range []string{f.base, originBase} {
		jointIsolateEngines(t, base, inertPort)
	}
	f.path = filepath.Join(jointAppDir(f.base), "node-policy.json")
	var err error
	f.want, err = nodepolicy.Parse([]byte(`{
		"version":2,"revision":"12","availability":"paused",
		"tiers":{
			"weak":[{"kind":"api","deployment":"free-small"},{"kind":"local","engine":"ollama","model":"joint-local","contextTokens":8192,"capabilities":["tools"]},{"kind":"api","deployment":"paid-big"}],
			"strong":[{"kind":"api","deployment":"paid-big"}]
		},
		"routing":{
			"accounts":[
				{"id":"free-acct","credential":"NVPAIR_PROVIDER_SECRET_JOINT_UNUSED_FREE","billing":"free","perRequestCeiling":"0","dailyCeiling":"0","monthlyCeiling":"0"},
				{"id":"paid-acct","credential":"NVPAIR_PROVIDER_SECRET_JOINT_UNUSED_PAID","billing":"paid","perRequestCeiling":"1000000","dailyCeiling":"9007199254740993","monthlyCeiling":"9223372036854775807"}
			],
			"deployments":[
				{"id":"free-small","account":"free-acct","model":"small-1","family":"alpha","capabilities":["tools"],"contextTokens":32768,"maxOutputTokens":4096,"prices":{"currency":"USD","unit":"nanoUSDPerMillionTokens","inputPerMillion":"0","outputPerMillion":"0"},"enabled":false},
				{"id":"paid-big","account":"paid-acct","model":"big-1","family":"beta","contextTokens":200000,"maxOutputTokens":8192,"enabled":false}
			],
			"tiers":{"weak":{"bareMode":"local","remoteAllowed":false,"paidAllowed":false,"allowedFamilies":["alpha"]}},
			"limits":{"queueSeconds":5,"coldStartSeconds":10,"maxAttempts":2,"preOutputBudgetSeconds":20,"maxOutputTokens":1024},
			"listener":{"enabled":false,"port":0,"allowedOrigins":["http://localhost:3000"]}
		}
	}`))
	if err != nil {
		t.Fatalf("invalid joint policy fixture: %v", err)
	}
	for _, option := range options {
		option(&f.want)
	}
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
	// Hold all reservations together so the three normal listener ports are
	// distinct; release them immediately before launching the owned processes.
	var reservations []net.Listener
	for range 3 {
		listener := jointListener(t)
		reservations = append(reservations, listener)
		f.ports = append(f.ports, listener.Addr().(*net.TCPAddr).Port)
	}
	for _, listener := range reservations {
		if err := listener.Close(); err != nil {
			t.Fatal(err)
		}
	}
	f.startTarget(t)
	f.originEngine = filepath.Join(binDir, testbin.Exe("joint-origin-engine"))
	jointCopyExecutable(t, engineMgrBin, f.originEngine)
	f.startOrigin(t)
	return f
}

func (f *policyJointFixture) startOrigin(t *testing.T) {
	t.Helper()
	f.origin = startInteropProcess(t, f.originBase, f.originEngine, 0, []string{"OLLAMA_HOST="},
		"--cluster-dir", f.originCluster, "--loaded-poll-interval", "0", "--log-level", "error")
	waitForMethod(t, f.origin.msgs, "engine:ready", 10*time.Second)
	completeEngineManagerRecovery(t, func(method string, params any) jsonrpc.Message {
		return callRPC(t, f.origin.in, f.origin.msgs, method, params, 10*time.Second)
	})
	assertJointEngineIsolation(t, f.origin, f.inertPort)
	sendLine(t, f.origin.in, jsonrpc.Message{JSONRPC: "2.0", Method: noderec.NotifyNodes, Params: mustMarshal(t, noderec.GetNodesResult{
		Nodes: []noderec.DirectoryNode{{HostUUID: "policy-joint-target", ClusterUUID: "policy-joint-target", IP: "127.0.0.1", Trusted: true,
			Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineControl: {Port: f.ports[1]}}}},
	})})
}

func (f *policyJointFixture) startTarget(t *testing.T) {
	t.Helper()
	ports := reserveFixtureServicePorts(t)
	ports.engineManagerHTTP, ports.engineControl = f.ports[0], f.ports[1]
	args := append([]string{"--scanner-path", f.scanner, "--engine-manager-path", f.engine, "--proxy-engines=", "--cluster-dir", f.cluster, "--log-level", "error"}, ports.args()...)
	f.target = startInteropProcess(t, f.base, brokerBin, f.ports[0],
		[]string{"OLLAMA_HOST=", "NVPAIR_TEST_SUBSCRIPTION_SCANNER=" + f.scanner, "NVPAIR_TEST_SUBSCRIPTION_PORT=" + strconv.Itoa(f.ports[2])},
		args...)
	f.target.paths = append(f.target.paths, f.engine, f.scanner)
	f.target.ports = append(f.target.ports, f.ports[1:]...)
	waitForMethod(t, f.target.msgs, "app:ready", 15*time.Second)
	for _, port := range f.ports {
		waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), 10*time.Second)
	}
	assertJointEngineIsolation(t, f.target, f.inertPort)
	t.Logf("policy_route broker_pid=%d supervised_engine=%q control_port=%d scanner=empty_directory_fixture", f.target.cmd.Process.Pid, f.engine, f.ports[1])
}

func (f *policyJointFixture) originCall(t *testing.T, method string, params map[string]any) jsonrpc.Message {
	t.Helper()
	input := make(map[string]any, len(params)+1)
	for key, value := range params {
		input[key] = value
	}
	input["nodeId"] = "policy-joint-target"
	return callRPC(t, f.origin.in, f.origin.msgs, method, input, 15*time.Second)
}

func (f *policyJointFixture) rpc(t *testing.T, method string, params map[string]any) jsonrpc.Message {
	t.Helper()
	response := f.originCall(t, method, params)
	if response.Error != nil || len(response.Result) == 0 {
		t.Fatalf("origin policy RPC failed: method=%s error_present=%t result_bytes=%d", method, response.Error != nil, len(response.Result))
	}
	return response
}

func (f *policyJointFixture) http(t *testing.T, path string, params map[string]any, status int) []byte {
	t.Helper()
	url := "https://127.0.0.1:" + strconv.Itoa(f.ports[1]) + path
	response, err := f.client.Post(url, "application/json", bytes.NewReader(mustMarshal(t, params)))
	if err != nil {
		t.Fatalf("target policy HTTP transport: %v", err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, (256<<10)+1))
	if err != nil || len(body) > 256<<10 {
		t.Fatal("target policy HTTP body could not be read within the bound")
	}
	if response.StatusCode != status {
		t.Fatalf("target policy HTTP status=%d want=%d", response.StatusCode, status)
	}
	return body
}

func (f *policyJointFixture) assertPolicy(t *testing.T) {
	t.Helper()
	response := callRPC(t, f.target.in, f.target.msgs, "policy:get", map[string]any{}, 10*time.Second)
	var envelope struct {
		Policy nodepolicy.Policy `json:"policy"`
	}
	if response.Error != nil || json.Unmarshal(response.Result, &envelope) != nil || !reflect.DeepEqual(envelope.Policy, f.want) {
		t.Fatal("broker's complete in-memory policy differs from the seeded policy plus accepted device edits")
	}
	assertStoredPolicy(t, f.path, f.want)
}

func (f *policyJointFixture) assertUnchanged(t *testing.T, before []byte) {
	t.Helper()
	f.assertPolicy(t)
	if !bytes.Equal(readPolicyBytes(t, f.path), before) {
		t.Fatal("rejected peer set changed persisted bytes")
	}
}

func (f *policyJointFixture) cloudDraft(t *testing.T, params map[string]any, cloud any) {
	t.Helper()
	var draft map[string]json.RawMessage
	if err := json.Unmarshal(mustMarshal(t, jointExpectedPeer(f.want)), &draft); err != nil {
		t.Fatal("copy peer fixture")
	}
	draft["routing"] = mustMarshal(t, cloud)
	params["policy"] = draft
}

func jointExpectedPeer(p nodepolicy.Policy) nodepolicy.PeerPolicy {
	return nodepolicy.PeerPolicy{
		Version: 2, Revision: p.Revision, Availability: p.Availability,
		Pause: p.Pause, Admission: p.Admission, Idle: p.Idle, Profiles: p.Profiles, Roles: map[string]nodepolicy.Role{}, TierPolicy: p.TierPolicy,
		ServedModels: p.ServedModels, ResidentModels: p.ResidentModels,
		Tiers: map[string][]nodepolicy.TierEntry{"weak": {{Kind: "local", Engine: "ollama", Model: "joint-local", ContextTokens: 8192, Capabilities: []nodepolicy.Capability{"tools"}}}},
	}
}

func assertJointPeer(t *testing.T, raw []byte, want nodepolicy.Policy) {
	t.Helper()
	var envelope struct {
		Policy             nodepolicy.PeerPolicy          `json:"policy"`
		Availability       nodepolicy.Availability        `json:"availability,omitempty"`
		ModelConfiguration *nodepolicy.ModelConfiguration `json:"modelConfiguration"`
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&envelope) != nil || decoder.Decode(&struct{}{}) != io.EOF {
		t.Fatal("peer result contains unknown fields, origin-only routing, or invalid JSON")
	}
	if !reflect.DeepEqual(envelope.Policy, jointExpectedPeer(want)) {
		t.Fatal("peer policy differs from the independently spelled device/local-only projection")
	}
	decodeFixtureModelConfiguration(t, raw, want.Revision)
}

func jointAppDir(base string) string {
	if runtime.GOOS == "darwin" {
		base = filepath.Join(base, "Library", "Application Support")
	}
	return filepath.Join(base, "Nvidia Corporation", "Personal AI Router")
}

func jointIsolateEngines(t *testing.T, base string, port int) {
	t.Helper()
	for _, engine := range engines.Names() {
		absent := filepath.Join(base, "absent-"+engine)
		probe := map[string]any{"http": "http://127.0.0.1:{port}/joint-no-engine", "status": 200, "timeout_s": 1, "interval_s": 60}
		rt := map[string]any{"port": port, "bin": absent, "cli": absent, "ready": probe, "health": probe}
		jointWriteJSON(t, filepath.Join(jointAppDir(base), "engines", engine+".json"), map[string]any{
			"engine": engine, "runtime": rt, "models_dir": filepath.Join(base, "absent-models", engine),
			"platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{
				"detect": []string{absent}, "runtime": rt, "models_dir": filepath.Join(base, "absent-models", engine),
			}},
		})
	}
}

func assertJointEngineIsolation(t *testing.T, process *interopProcess, port int) {
	t.Helper()
	response := callRPC(t, process.in, process.msgs, "engine:get-installed", map[string]any{}, 10*time.Second)
	var result struct {
		Engines []struct {
			Engine    string `json:"engine"`
			Installed bool   `json:"installed"`
			Running   bool   `json:"running"`
			Port      int    `json:"port"`
		} `json:"engines"`
	}
	if response.Error != nil || json.Unmarshal(response.Result, &result) != nil || len(result.Engines) != len(engines.Names()) {
		t.Fatal("isolated engine inventory was not returned")
	}
	seen := make(map[string]bool)
	for _, engine := range result.Engines {
		if _, known := engines.ByName(engine.Engine); !known || seen[engine.Engine] || engine.Installed || engine.Running || engine.Port != port {
			t.Fatal("engine isolation override was not applied to every bundled engine")
		}
		seen[engine.Engine] = true
	}
}

func jointWriteJSON(t *testing.T, path string, value any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, mustMarshal(t, value), 0o600); err != nil {
		t.Fatal(err)
	}
}

func jointCopyExecutable(t *testing.T, source, target string) {
	t.Helper()
	data, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, data, 0o700); err != nil {
		t.Fatalf("copy joint executable to %q (a previous process may still be running): %v", target, err)
	}
}

func jointListener(t *testing.T) net.Listener {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if port == 8080 || port == 11434 || port == 1234 || port >= 14318 && port <= 14323 {
		if err := listener.Close(); err != nil {
			t.Error(err)
		}
		t.Fatal("OS selected a protected listener port")
	}
	return listener
}
