// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Cross-process tests for the two identity/anti-entropy guarantees the
// workload store + workload-manager must uphold:
//
//   - Cross-engine identity: llama.cpp and its descriptor engine each begin their request
//     counter at 1, so two concurrent cross-engine jobs both mint workload id
//     "1". They must remain DISTINCT records all the way to the client stream
//     ((origin,engine,runId,id) key), not collapse into one.
//   - Rehydration on restart: when the supervised workload-manager crashes and
//     the supervisor respawns it, the broker replays its still-active local
//     workloads so the fresh manager can keep re-asserting them to peers. Kill
//     the manager mid-flight and assert a discovered peer hears the active
//     workload again from the restarted process.
//
// Both drive the real binaries TestMain builds. A synthetic scanner supplies
// profile advertisements and pinned peers through the broker's discovery relay.
package tests

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-shared/testbin"
)

// TestWorkloadCrossEngineIdentityDistinct fires exactly one inference through
// each supervised proxy — the first request each serves, so both mint workload
// id "1". They differ only by engine (llamacpp vs llamacpp-custom) and the proxy's
// per-process runId, which must keep them as two separate records on the
// broker's workloads:* stream. Before the identity fix the (origin,id) store
// key collapsed them into one.
func TestWorkloadCrossEngineIdentityDistinct(t *testing.T) {
	// Fake engines: 200 on any request so each inference completes promptly.
	fakeEngine := func(body string) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			io.Copy(io.Discard, r.Body)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(body))
		}))
	}
	llamacpp := fakeEngine(`{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	defer llamacpp.Close()
	custom := fakeEngine(`{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	defer custom.Close()
	llamacppPort := portOfURL(t, llamacpp.URL)
	customPort := portOfURL(t, custom.URL)

	configDir, clusterDir := t.TempDir(), filepath.Join(t.TempDir(), "cluster")
	facades := map[string]int{"llamacpp": freePort(t), "llamacpp-custom": freePort(t)}
	writePrivateBrokerEngineSettings(t, configDir, clusterDir, facades)
	manifest := privateFixtureManifest(t, "llamacpp-custom", filepath.Join(configDir, "absent-models"), filepath.Join(configDir, "absent-engine"), map[string]any{"port": 0, "ready": nil, "health": nil}, "openai-v1")
	jointWriteJSON(t, filepath.Join(jointAppDir(configDir), "engines", "llamacpp-custom.json"), manifest)
	control := newFixtureProxyControl(t)
	scanner, scannerPort, scannerEnv := newWorkloadScanner(t)
	stdin, msgs, _, cleanup := startBrokerWithDirsAndEnv(t, configDir, clusterDir, append(control.env(), scannerEnv...),
		"--scanner-path", scanner,
		"--proxy-path", control.image, "--proxy-engines", "llamacpp",
		"--engine-manager-path", engineMgrBin,
		"--workload-manager-path", workloadMgrBin,
	)
	t.Cleanup(cleanup)

	waitForMethod(t, msgs, "app:ready", 10*time.Second)
	t.Log("broker ready")

	writeRawFrame(t, stdin, `{"jsonrpc":"2.0","id":1,"method":"workloads:subscribe"}`)
	ack := waitForResponse(t, msgs, 5*time.Second)
	var sr subscriptionResult
	if err := json.Unmarshal(ack.Result, &sr); err != nil || !sr.Subscribed {
		t.Fatalf("workloads:subscribe ack = %s (err %v), want subscribed:true", ack.Result, err)
	}

	control.ready(t, "llamacpp", facades["llamacpp"])
	control.ready(t, "llamacpp-custom", facades["llamacpp-custom"])
	// Setup through the transparent proxy relay preserves the first HTTP
	// inference request as workload id "1" for each facade.
	call := func(t *testing.T, method string, params any) jsonrpc.Message {
		t.Helper()
		reply, ok := control.componentCall(t, method, params)
		if !ok {
			t.Fatalf("unexpected workload fixture method %q", method)
		}
		return reply
	}
	for engine, port := range map[string]int{"llamacpp": llamacppPort, "llamacpp-custom": customPort} {
		addWorkloadInferenceNode(t, call, scannerPort, engine, "fake-"+engine, port, "crossengine-model")
	}
	t.Log("both fake engines pinned as proxy targets")

	client := &http.Client{Timeout: 5 * time.Second}
	fire := func(port int, path string) {
		egressProcessHTTP(t, client, http.MethodPost, port, path, []byte(`{"model":"crossengine-model","messages":[]}`), http.StatusOK)
	}
	fire(facades["llamacpp"], "/v1/chat/completions")
	fire(facades["llamacpp-custom"], "/v1/chat/completions")

	// Collect the first workload id observed per engine. Both cross-engine
	// workloads must surface; the fix is proven when they share one id yet
	// arrive as two distinct engine records.
	engineIDs := map[string]string{}
	deadline := time.After(20 * time.Second)
	for len(engineIDs) < 2 {
		select {
		case msg, ok := <-msgs:
			if !ok {
				t.Fatal("broker stream closed before both engine workloads seen")
			}
			if msg.Method != "workloads:upsert" {
				continue
			}
			var p wlParams
			if err := json.Unmarshal(msg.Params, &p); err != nil {
				t.Fatalf("decode engine workload: %v", err)
			}
			eng := p.WorkloadInfo.Engine
			if (eng == "llamacpp" || eng == "llamacpp-custom") && p.WorkloadInfo.ID != "" {
				if p.WorkloadInfo.Model != "crossengine-model" {
					t.Fatalf("%s workload model=%q, want crossengine-model", eng, p.WorkloadInfo.Model)
				}
				if _, seen := engineIDs[eng]; !seen {
					engineIDs[eng] = p.WorkloadInfo.ID
					t.Logf("tracked %s workload id=%q state=%q", eng, p.WorkloadInfo.ID, p.WorkloadInfo.State)
				}
			}
		case <-deadline:
			t.Fatalf("timed out: only saw engine workloads %v, want both llamacpp and llamacpp-custom (cross-engine records collapsed?)", engineIDs)
		}
	}

	if engineIDs["llamacpp"] != engineIDs["llamacpp-custom"] {
		t.Fatalf("expected the two proxies to collide on one workload id (both counters start at 1); got llamacpp=%q llamacpp-custom=%q — determinism assumption broke", engineIDs["llamacpp"], engineIDs["llamacpp-custom"])
	}
	t.Logf("cross-engine OK: llamacpp and llamacpp-custom both tracked distinct workloads sharing id %q", engineIDs["llamacpp"])
}

// TestWorkloadManagerRehydratesActiveWorkloadOnRestart drives the rehydration
// path end-to-end. A proxied inference is held open by a blocking fake engine,
// so its workload reaches "running" once the engine emits its first byte. The
// supervised workload-manager is then killed; the broker's supervisor respawns
// it and replays the still-active local workload. A discovered stub peer must
// hear the running workload AGAIN from the restarted manager — which only
// happens if the broker rehydrated the fresh process's active set.
func TestWorkloadManagerRehydratesActiveWorkloadOnRestart(t *testing.T) {
	// Fake llama.cpp that starts a streaming response, flushes one chunk so the
	// proxy commits and the workload becomes "running", then blocks — keeping
	// the proxied inference in-flight so it never reaches a terminal state. The
	// handler also unblocks if the request is cancelled, so a client-side abort
	// can't wedge it.
	//
	// The chunk is required: the proxy commits on the first byte of response
	// body, not on the headers, because headers only prove an engine accepted
	// the request. An engine that blocks before producing content leaves the
	// workload "queued", which is the truth but not what this test is about.
	release := make(chan struct{})
	var releaseOnce sync.Once
	stopEngine := func() { releaseOnce.Do(func() { close(release) }) }
	llamacpp := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `data: {"id":"fixture","model":"rehydrate-model","choices":[{"index":0,"delta":{"role":"assistant","content":"ready"},"finish_reason":null}]}`+"\n\n")
		if fl, ok := w.(http.Flusher); ok {
			fl.Flush()
		}
		select {
		case <-release:
		case <-r.Context().Done():
		}
		_, _ = io.WriteString(w, `data: [DONE]`+"\n\n")
	}))
	// Defer order matters: stopEngine() must run BEFORE llamacpp.Close(), because
	// Close() blocks until the deliberately-held request finishes, which only
	// happens once release is closed. Registering llamacpp.Close() first makes it
	// run last (LIFO).
	defer llamacpp.Close()
	defer stopEngine()
	llamacppPort := portOfURL(t, llamacpp.URL)

	// Stub peer: a cross-pinned cluster peer that records every workload frame the
	// manager fans out to it. The inter-node interface is cluster mTLS only, so the
	// stub must hold a cluster identity the node under test pins.
	fx := newInterNodeCluster(t)
	received, peerPort := fx.startDirectClusterPeer(t)
	configDir := t.TempDir()
	privateProxyPort := freePort(t)
	writePrivateBrokerEngineSettings(t, configDir, fx.nodeDir, map[string]int{"llamacpp": privateProxyPort})

	scanner, scannerPort, scannerEnv := newWorkloadScanner(t)
	stdin, msgs, stderr, cleanup := startBrokerWithDirsAndEnv(t, configDir, fx.nodeDir, scannerEnv,
		"--scanner-path", scanner,
		"--proxy-path", proxyBin, "--proxy-engines", "llamacpp",
		"--workload-manager-path", workloadMgrBin,
	)
	t.Cleanup(cleanup)

	// Drain broker stderr continuously (avoids pipe backpressure over the long
	// waits) and surface each workload-manager start pid — the initial spawn and
	// every supervisor respawn.
	wmPids := make(chan int, 8)
	go func() {
		for line := range stderr {
			if m := wmStartedPidRe.FindStringSubmatch(line); m != nil {
				if pid, err := strconv.Atoi(m[1]); err == nil {
					select {
					case wmPids <- pid:
					default:
					}
				}
			}
		}
	}()

	waitForMethod(t, msgs, "app:ready", 10*time.Second)
	proxyPort := waitEngineProxyReady(t, "llamacpp-proxy", stdin, msgs, 15*time.Second)
	if proxyPort != privateProxyPort {
		t.Fatalf("llama.cpp proxy port=%d want private port=%d", proxyPort, privateProxyPort)
	}
	t.Logf("broker ready; proxy on %d", proxyPort)

	pid1 := awaitInt(t, wmPids, 10*time.Second, "initial workload-manager start")
	t.Logf("original workload-manager pid=%d", pid1)

	// Pin the fake llamacpp, then fire one inference in the background — the
	// backend blocks, so the workload stays running until teardown.
	call := workloadBrokerCaller(stdin, msgs)
	addWorkloadInferenceNode(t, call, scannerPort, "llamacpp", "fake-llamacpp", llamacppPort, "rehydrate-model")
	publishWorkloadNode(t, scannerPort, workloadPeerNode(fx, "rehydrate-peer-uuid", peerPort))
	requestStarted := make(chan error, 1)
	go func() {
		c := &http.Client{Timeout: 120 * time.Second}
		resp, err := c.Post(
			fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", proxyPort),
			"application/json",
			strings.NewReader(`{"model":"rehydrate-model","messages":[],"stream":true}`),
		)
		if err != nil {
			requestStarted <- err
			return
		}
		if resp.StatusCode != http.StatusOK {
			requestStarted <- fmt.Errorf("inference status=%d, want %d", resp.StatusCode, http.StatusOK)
		} else {
			requestStarted <- nil
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}()
	select {
	case err := <-requestStarted:
		if err != nil {
			t.Fatalf("start held workload: %v", err)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("held workload did not commit HTTP headers")
	}

	// The original manager backfills its active set to the discovered peer.
	waitStubPeerWorkload(t, received, "rehydrate-model", "running", 45*time.Second)
	t.Log("original manager asserted the running workload to the peer")

	// Kill the manager; drain what the peer has already seen so the assertion
	// below can only pass on a post-restart re-assertion.
	drainWorkloadMessages(received)
	t.Logf("killing supervised workload-manager pid=%d", pid1)
	if proc, err := os.FindProcess(pid1); err == nil {
		_ = proc.Kill()
	}

	pid2 := awaitInt(t, wmPids, 30*time.Second, "workload-manager respawn")
	if pid2 == pid1 {
		t.Fatalf("respawned workload-manager reused pid %d; expected a fresh process", pid1)
	}
	t.Logf("workload-manager respawned pid=%d", pid2)

	// The rehydrated manager must re-assert the still-active workload to the
	// (re-discovered) peer. Without rehydration its activeLocal is empty after
	// the restart and the peer never hears about the job again.
	waitStubPeerWorkload(t, received, "rehydrate-model", "running", 75*time.Second)
	t.Log("rehydrate OK: restarted manager re-asserted the active workload to the peer")
}

// TestWorkloadManagerRehydratesRecentTerminalOnRestart is the restart-during-
// terminal path the running-workload test doesn't cover. A proxied inference
// COMPLETES (terminal), which the manager keeps in its re-sync set for a couple
// of heartbeats so a peer that missed the terminal still converges. When the
// manager is killed mid-window, the broker must replay the recent terminal (not
// just active workloads) so the restarted manager re-asserts the completed
// state — otherwise a peer's wrongly-inferred "failed" (or a missed terminal)
// can never be repaired after a restart.
func TestWorkloadManagerRehydratesRecentTerminalOnRestart(t *testing.T) {
	// Fake llama.cpp that completes immediately, so the workload reaches a terminal
	// (completed) state right away.
	llamacpp := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"finished"},"finish_reason":"stop"}]}`))
	}))
	defer llamacpp.Close()
	llamacppPort := portOfURL(t, llamacpp.URL)

	fx := newInterNodeCluster(t)
	received, peerPort := fx.startDirectClusterPeer(t)
	configDir := t.TempDir()
	privateProxyPort := freePort(t)
	writePrivateBrokerEngineSettings(t, configDir, fx.nodeDir, map[string]int{"llamacpp": privateProxyPort})

	scanner, scannerPort, scannerEnv := newWorkloadScanner(t)
	stdin, msgs, stderr, cleanup := startBrokerWithDirsAndEnv(t, configDir, fx.nodeDir, scannerEnv,
		"--scanner-path", scanner,
		"--proxy-path", proxyBin, "--proxy-engines", "llamacpp",
		"--workload-manager-path", workloadMgrBin,
	)
	t.Cleanup(cleanup)

	wmPids := make(chan int, 8)
	go func() {
		for line := range stderr {
			if m := wmStartedPidRe.FindStringSubmatch(line); m != nil {
				if pid, err := strconv.Atoi(m[1]); err == nil {
					select {
					case wmPids <- pid:
					default:
					}
				}
			}
		}
	}()

	waitForMethod(t, msgs, "app:ready", 10*time.Second)
	proxyPort := waitEngineProxyReady(t, "llamacpp-proxy", stdin, msgs, 15*time.Second)
	if proxyPort != privateProxyPort {
		t.Fatalf("llama.cpp proxy port=%d want private port=%d", proxyPort, privateProxyPort)
	}
	pid1 := awaitInt(t, wmPids, 10*time.Second, "initial workload-manager start")

	call := workloadBrokerCaller(stdin, msgs)
	addWorkloadInferenceNode(t, call, scannerPort, "llamacpp", "fake-llamacpp", llamacppPort, "rehydrate-term-model")
	publishWorkloadNode(t, scannerPort, workloadPeerNode(fx, "rehydrate-term-peer-uuid", peerPort))

	// Fire one inference; it completes promptly → terminal workload.
	client := &http.Client{Timeout: 10 * time.Second}
	egressProcessHTTP(t, client, http.MethodPost, proxyPort, "/v1/chat/completions", []byte(`{"model":"rehydrate-term-model","messages":[]}`), http.StatusOK)

	// The original manager backfills the recent terminal to the discovered peer.
	waitStubPeerWorkload(t, received, "rehydrate-term-model", "completed", 45*time.Second)
	t.Log("original manager asserted the completed workload to the peer")

	drainWorkloadMessages(received)
	t.Logf("killing supervised workload-manager pid=%d", pid1)
	if proc, err := os.FindProcess(pid1); err == nil {
		_ = proc.Kill()
	}

	pid2 := awaitInt(t, wmPids, 30*time.Second, "workload-manager respawn")
	if pid2 == pid1 {
		t.Fatalf("respawned workload-manager reused pid %d; expected a fresh process", pid1)
	}
	t.Logf("workload-manager respawned pid=%d", pid2)

	// The rehydrated manager must re-assert the recent terminal. If the broker
	// only replayed active workloads, the completed state is gone after the
	// restart and the peer never hears it again.
	waitStubPeerWorkload(t, received, "rehydrate-term-model", "completed", 75*time.Second)
	t.Log("rehydrate OK: restarted manager re-asserted the recent terminal to the peer")
}

// --- helpers ---

// Reuse the registered scanner role so inference fixtures can advertise
// profiles without browsing installed nodes or relying on multicast delivery.
func newWorkloadScanner(t *testing.T) (string, int, []string) {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatalf("locate workload scanner fixture: %v", err)
	}
	scanner := filepath.Join(testbin.Dir(t, "tests", "subscription-scanner", "subscription-scanner"), testbin.Exe("subscription-scanner"))
	jointCopyExecutable(t, self, scanner)
	port := freePort(t)
	return scanner, port, []string{"NVPAIR_TEST_SUBSCRIPTION_SCANNER=" + scanner, "NVPAIR_TEST_SUBSCRIPTION_PORT=" + strconv.Itoa(port)}
}

func workloadBrokerCaller(stdin io.Writer, messages <-chan jsonrpc.Message) egressProcessCall {
	nextID := 50
	return func(t *testing.T, method string, params any) jsonrpc.Message {
		t.Helper()
		nextID++
		writeRawFrame(t, stdin, fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"method":%q,"params":%s}`, nextID, method, mustMarshal(t, params)))
		return waitForResponseID(t, messages, nextID, 5*time.Second)
	}
}

func publishWorkloadNode(t *testing.T, scannerPort int, node noderec.DirectoryNode) {
	t.Helper()
	if err := validateFixturePort(scannerPort); err != nil {
		t.Fatal(err)
	}
	waitTCP(t, "127.0.0.1:"+strconv.Itoa(scannerPort), 10*time.Second)
	frame := jsonrpc.Message{JSONRPC: "2.0", Method: noderec.NotifyNodeDiscovered,
		Params: mustMarshal(t, noderec.NodeEvent{Node: node})}
	egressProcessHTTP(t, egressProcessHTTPClient(t), http.MethodPost, scannerPort, "/", mustMarshal(t, frame), http.StatusOK)
}

func workloadPeerNode(cluster *interNodeCluster, id string, port int) noderec.DirectoryNode {
	return noderec.DirectoryNode{HostUUID: id, ClusterUUID: cluster.peerUUID, IP: "127.0.0.1",
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceWorkload: {Port: port}}}
}

// The existing concrete workload names are explicitly declared profile keys.
// A same-ID manual entry grants only the private plaintext fixture transport;
// the scanner advertisement supplies the profile eligibility facts.
func addWorkloadInferenceNode(t *testing.T, call egressProcessCall, scannerPort int, engine, id string, port int, models ...string) {
	t.Helper()
	advertisement := &nodepolicy.Advertisement{Revision: "1"}
	for _, model := range models {
		advertisement.Profiles = append(advertisement.Profiles, nodepolicy.AdvertisedProfile{
			Key: model, DisplayName: model, Engine: engine, Model: model,
		})
	}
	service := privateFixtureEngine(t, engine).DiscoveryService
	publishWorkloadNode(t, scannerPort, noderec.DirectoryNode{
		HostUUID: id, Name: id, IP: "127.0.0.1",
		Services:       map[noderec.ServiceKey]noderec.ServiceStatus{service: {Port: port}},
		ModelsByEngine: map[string][]string{engine: models}, Policy: advertisement,
	})
	prefix := engine + "-proxy:"
	neEventually(t, "workload profile advertisement reaches "+engine+" proxy", 10*time.Second, func() bool {
		reply := call(t, prefix+"nodes/list", map[string]any{})
		egressProcessOK(t, reply)
		var result struct {
			Nodes []struct {
				ID     string   `json:"id"`
				Models []string `json:"models"`
			} `json:"nodes"`
		}
		egressProcessDecode(t, reply.Result, &result)
		for _, node := range result.Nodes {
			if node.ID == id && slices.Equal(node.Models, models) {
				return true
			}
		}
		return false
	})
	egressProcessOK(t, call(t, prefix+"node/add-manual", map[string]any{
		"id": id, "host": "127.0.0.1", "port": port, "addresses": []string{"127.0.0.1"}, "models": models,
	}))
	egressProcessOK(t, call(t, prefix+"node/select", map[string]string{"id": id}))
}

var wmStartedPidRe = regexp.MustCompile(`workload-manager started.*\bpid=(\d+)`)

// waitStubPeerWorkload blocks until the stub peer records a workload:* frame
// for the given model in the given state.
func waitStubPeerWorkload(t *testing.T, received <-chan jsonrpc.Message, model, state string, timeout time.Duration) {
	t.Helper()
	deadline := time.After(timeout)
	for {
		select {
		case msg, ok := <-received:
			if !ok {
				t.Fatalf("stub peer channel closed before %q workload %s", model, state)
			}
			if !strings.HasPrefix(msg.Method, "workload:") {
				continue
			}
			var p wlParams
			if json.Unmarshal(msg.Params, &p) != nil {
				continue
			}
			if p.WorkloadInfo.Model == model && p.WorkloadInfo.State == state {
				return
			}
		case <-deadline:
			t.Fatalf("timed out (%s) waiting for stub peer to receive %q workload in state %q", timeout, model, state)
		}
	}
}

// drainWorkloadMessages empties the channel of anything buffered so far.
func drainWorkloadMessages(ch <-chan jsonrpc.Message) {
	for {
		select {
		case <-ch:
		default:
			return
		}
	}
}

// awaitInt receives one int from ch or fails after timeout.
func awaitInt(t *testing.T, ch <-chan int, timeout time.Duration, what string) int {
	t.Helper()
	select {
	case v := <-ch:
		return v
	case <-time.After(timeout):
		t.Fatalf("timed out (%s) waiting for %s", timeout, what)
		return 0
	}
}

// portOfURL extracts the TCP port from an httptest server URL.
func portOfURL(t *testing.T, raw string) int {
	t.Helper()
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("parse url %q: %v", raw, err)
	}
	_, portStr, err := net.SplitHostPort(u.Host)
	if err != nil {
		t.Fatalf("split host:port %q: %v", u.Host, err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatalf("atoi port %q: %v", portStr, err)
	}
	return port
}
