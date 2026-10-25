// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/testbin"
)

func TestRoutingIdentityProcess(t *testing.T) {
	t.Run("zero facade binding and reused facade ID reject an old run cancel", func(t *testing.T) {
		base := routingIdentityAppDir(t)
		first := startEgressProcessProxy(t, base)
		firstAck := routingIdentityBind(t, first)
		firstPort := first.enable(t)
		engine := newRoutingIdentityEngine(t, egressProcessModel, "text/event-stream",
			"data: {\"choices\":[{\"delta\":{\"content\":\"ready\"}}]}\n\n")
		egressProcessAddConcrete(t, first.call, "llamacpp:", engine.port)
		request := routingIdentityStream(t, firstPort, "/v1/chat/completions", egressProcessModel)
		old := routingIdentityProxyRunning(t, first, firstAck)
		routingIdentityCancel(t, first, firstAck.OriginNodeID, old, true)
		engine.awaitCancelled(t, 1)
		request()
		first.stop(t)

		// Reuse the same disposable configuration, engine, model and facade
		// counter. Only the producing process incarnation changes.
		second := startEgressProcessProxy(t, base)
		secondAck := routingIdentityBind(t, second)
		if secondAck.RunID == firstAck.RunID {
			t.Fatal("replacement proxy reused the previous process run")
		}
		secondPort := second.enable(t)
		egressProcessAddConcrete(t, second.call, "llamacpp:", engine.port)
		request = routingIdentityStream(t, secondPort, "/v1/chat/completions", egressProcessModel)
		current := routingIdentityProxyRunning(t, second, secondAck)
		if old.ID != "1" || current.ID != old.ID || current.Engine != old.Engine || secondAck.OriginNodeID != firstAck.OriginNodeID {
			t.Fatalf("facade ID reuse prerequisite: old=%+v current=%+v", old.key(), current.key())
		}
		before := routingIdentityAdmission(t, second, secondAck.RunID)
		routingIdentityCancel(t, second, firstAck.OriginNodeID, old, false)
		after := routingIdentityAdmission(t, second, secondAck.RunID)
		if !reflect.DeepEqual(before, after) {
			t.Fatalf("old-run cancel changed admission: before=%+v after=%+v", before, after)
		}
		engine.assertEffects(t, 2, 1, 1)
		// The qualified positive control joins the actual engine cancellation;
		// no quiet time window is used as evidence of old-run isolation.
		routingIdentityCancel(t, second, secondAck.OriginNodeID, current, true)
		engine.awaitCancelled(t, 2)
		request()
		engine.assertEffects(t, 2, 2, 0)
	})

	t.Run("late peer old-run frames have no effect and peer cancel reaches owner", func(t *testing.T) {
		cluster := newInterNodeCluster(t)
		ports := reserveFixtureServicePorts(t)
		engine := newRoutingIdentityEngine(t, "departed-run-model", "application/x-ndjson",
			`{"message":{"role":"assistant","content":"ready"},"done":false}`+"\n")
		f := routingIdentityBroker(t, cluster.nodeDir, ports, engine.port)
		pid := mustPid(t, waitForStderr(t, f.proxyLines, proxyPidRe, 10*time.Second))
		firstRequest := f.startRequest(t)
		old := waitDepartedRunFrame(t, f.events, "workloads:upsert", 15*time.Second, func(p departedRunParams) bool {
			return p.WorkloadInfo.State == "running"
		}).WorkloadInfo
		assertDepartedRunRunning(t, old)
		assertDepartedRunCancelResult(t, f.cancel(t, old), true)
		terminal := waitDepartedRunFrame(t, f.events, "workloads:upsert", 15*time.Second, func(p departedRunParams) bool {
			return p.WorkloadInfo.key() == old.key() && p.WorkloadInfo.State == "cancelled"
		}).WorkloadInfo
		engine.awaitCancelled(t, 1)
		joinDepartedRunRequest(t, firstRequest)
		// Finish before restart: retirement of an in-flight departed run is
		// already covered by TestWorkloadDepartedRunRetired.
		killDepartedRunProxy(t, pid)
		replacementPID := mustPid(t, waitForStderr(t, f.proxyLines, proxyPidRe, 15*time.Second))
		if replacementPID == pid {
			t.Fatal("supervisor did not replace the proxy process")
		}
		f.port = waitProxyReady(t, f.stdin, f.control, 15*time.Second)
		f.selectEngine(t)
		secondRequest := f.startRequest(t)
		current := waitDepartedRunFrame(t, f.events, "workloads:upsert", 15*time.Second, func(p departedRunParams) bool {
			return p.WorkloadInfo.State == "running" && p.WorkloadInfo.RunID != old.RunID
		}).WorkloadInfo
		assertDepartedRunRunning(t, current)
		if current.ID != old.ID || current.OriginatedFrom != old.OriginatedFrom || current.Engine != old.Engine {
			t.Fatalf("replacement did not reuse facade identity: old=%+v current=%+v", old.key(), current.key())
		}

		peer := cluster.clientAsPeer(t)
		endpoint := fmt.Sprintf("https://127.0.0.1:%d/v1/workloads/events", ports.workload)
		post := func(method string, params any) {
			t.Helper()
			frame := jsonrpc.Message{JSONRPC: "2.0", Method: method, Params: egressProcessJSON(t, params)}
			postPeerEvent(t, peer, endpoint, string(egressProcessJSON(t, frame)), 15*time.Second)
		}
		// resync bypasses the manager's dedup cache, so rejection must happen
		// in the broker store rather than being hidden by transport dedup.
		// This exercises late peer replays. Injecting a callback from a retired
		// proxy handle requires an in-process seam in routeProxyWorkloadFrom;
		// the real-process harness does not expose that source handle.
		// Retired local callbacks have component-level evidence in TestDepartedRun:
		// "old callback admitted before replacement is rechecked"
		// (nvpair-ui-broker/departedrun_test.go:425) and
		// "replacement run stays live and synthetic frames leave fence alone"
		// (nvpair-ui-broker/departedrun_test.go:281). Process-side retirement is
		// covered by TestWorkloadDepartedRunRetired; late peer replay here is the
		// process evidence accepted by Mac, not retired-local-callback injection.
		post("workload:started", departedRunParams{WorkloadInfo: old, Resync: true})
		post("workload:errored", departedRunParams{WorkloadInfo: terminal, Resync: true})
		// A distinct lifecycle frame on the same manager stdout is an explicit
		// processing barrier for the preceding late frames.
		barrier := current
		barrier.ID, barrier.OriginatedFrom, barrier.RunID = "identity-barrier", "identity-peer", "identity-peer-run"
		post("workload:started", departedRunParams{WorkloadInfo: barrier})
		deadline := time.NewTimer(15 * time.Second)
		defer deadline.Stop()
	barrierLoop:
		for {
			select {
			case msg, ok := <-f.events:
				if !ok {
					t.Fatal("workload stream closed before late-frame barrier")
				}
				if msg.Method != "workloads:upsert" {
					continue
				}
				w := decodeDepartedRunFrame(t, msg).WorkloadInfo
				if w.key() == old.key() || w.key() == current.key() {
					t.Fatalf("late old-run frames emitted an identity update: %+v", w)
				}
				if w.key() == barrier.key() {
					break barrierLoop
				}
			case <-deadline.C:
				t.Fatal("manager late-frame processing barrier did not reach broker")
			}
		}
		if got := f.initialWorkload(t, old.key()); !reflect.DeepEqual(got, terminal) {
			t.Fatalf("late old-run frames changed terminal: got=%+v want=%+v", got, terminal)
		}
		if got := f.initialWorkload(t, current.key()); !reflect.DeepEqual(got, current) {
			t.Fatalf("late old-run frames changed replacement: got=%+v want=%+v", got, current)
		}
		assertDepartedRunCancelResult(t, f.cancel(t, old), false)
		engine.assertEffects(t, 2, 1, 1)

		// Exercise the real receiving workload-manager's pinned peer HTTP
		// interface, its stdout relay, the broker origin check and proxy owner.
		post("workloads:cancel", map[string]string{"originatedFrom": current.OriginatedFrom,
			"workloadId": current.ID, "engine": current.Engine, "runId": current.RunID, "cancelId": "identity-peer-cancel"})
		cancelled := waitDepartedRunFrame(t, f.events, "workloads:upsert", 15*time.Second, func(p departedRunParams) bool {
			return p.WorkloadInfo.key() == current.key() && p.WorkloadInfo.State == "cancelled"
		}).WorkloadInfo
		assertDepartedRunTerminal(t, cancelled, current, "cancelled", "cancelled by request")
		engine.awaitCancelled(t, 2)
		joinDepartedRunRequest(t, secondRequest)
		engine.assertEffects(t, 2, 2, 0)
	})
}

func routingIdentityAppDir(t *testing.T) string {
	t.Helper()
	base, err := filepath.Abs(t.TempDir())
	if err != nil {
		t.Fatalf("resolve disposable identity appdir: %v", err)
	}
	return base
}

func routingIdentityBind(t *testing.T, p *egressProcessProxy) nodepolicy.PolicyAck {
	t.Helper()
	// startEgressProcessProxy starts with no facade. Bind before enable; all
	// observations here therefore come from the process, not a facade.
	ack := p.activate(t, nodepolicy.Default())
	nonce, err := hex.DecodeString(ack.RunID)
	if err != nil || len(nonce) != 16 || ack.RunID != strings.ToLower(ack.RunID) {
		t.Fatalf("process run is not a 128-bit lowercase hex nonce: %q", ack.RunID)
	}
	snapshot := routingIdentityAdmission(t, p, ack.RunID)
	if snapshot.Active != 0 || snapshot.Queued != 0 {
		t.Fatalf("zero-facade admission counts = %+v", snapshot)
	}
	msg := p.client.notification(t, nodepolicy.NotifyStrictReadiness)
	var readiness nodepolicy.StrictAdmissionReadiness
	egressProcessDecode(t, msg.Params, &readiness)
	if err := readiness.Validate(); err != nil {
		t.Fatalf("zero-facade readiness: %v", err)
	}
	if readiness.ProxyRunID != ack.RunID || readiness.PolicyRevision != ack.Revision || readiness.Capability != nil {
		t.Fatalf("zero-facade readiness = %+v, acknowledgement = %+v", readiness, ack)
	}
	return ack
}

func routingIdentityAdmission(t *testing.T, p *egressProcessProxy, run string) nodepolicy.AdmissionSnapshotResult {
	t.Helper()
	reply := p.call(t, nodepolicy.MethodAdmissionSnapshot, nodepolicy.AdmissionSnapshotParams{})
	egressProcessOK(t, reply)
	var snapshot nodepolicy.AdmissionSnapshotResult
	egressProcessDecode(t, reply.Result, &snapshot)
	if snapshot.RunID != run {
		t.Fatalf("admission snapshot run = %q, want acknowledged run %q", snapshot.RunID, run)
	}
	return snapshot
}

func routingIdentityProxyRunning(t *testing.T, p *egressProcessProxy, ack nodepolicy.PolicyAck) departedRunInfo {
	t.Helper()
	msg := p.client.notification(t, "workload:started")
	w := decodeDepartedRunFrame(t, *msg).WorkloadInfo
	// The proxy leaves origin empty for the broker to stamp (proxy.go:145,
	// 169, 525); the policy acknowledgement binds cancellation origin separately.
	if w.ID != "1" || w.Engine != "llamacpp" || w.RunID != ack.RunID || w.OriginatedFrom != "" || w.State != "running" || w.Model != egressProcessModel {
		t.Fatalf("proxy workload wire identity = %+v, acknowledgement = %+v", w, ack)
	}
	return w
}

func routingIdentityCancel(t *testing.T, p *egressProcessProxy, origin string, w departedRunInfo, want bool) {
	t.Helper()
	reply := p.call(t, nodepolicy.MethodWorkloadCancel, nodepolicy.WorkloadCancelParams{
		OriginatedFrom: origin, WorkloadID: w.ID, Engine: w.Engine, RunID: w.RunID,
	})
	egressProcessOK(t, reply)
	var result struct {
		Found *bool `json:"found"`
	}
	egressProcessDecode(t, reply.Result, &result)
	if result.Found == nil || *result.Found != want {
		t.Fatalf("qualified workload/cancel found = %v, want %t", result.Found, want)
	}
}

type routingIdentityEngine struct {
	port      int
	calls     atomic.Int32
	cancelled atomic.Int32
	active    atomic.Int32
	exits     chan struct{}
}

func newRoutingIdentityEngine(t *testing.T, model, contentType, chunk string) *routingIdentityEngine {
	t.Helper()
	e := &routingIdentityEngine{exits: make(chan struct{}, 4)}
	release := make(chan struct{})
	server := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The llama.cpp facade rewrites client /v1/models to engine /models.
		if r.Method == http.MethodGet && r.URL.Path == "/models" {
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"id": model}}}); err != nil {
				t.Errorf("write identity engine inventory: %v", err)
			}
			return
		}
		var body struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode identity inference: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if r.Method != http.MethodPost || (r.URL.Path != "/api/chat" && r.URL.Path != "/v1/chat/completions") || body.Model != model {
			t.Errorf("unexpected identity engine route: method=%s path=%s model=%q", r.Method, r.URL.Path, body.Model)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		e.calls.Add(1)
		e.active.Add(1)
		cancelled := false
		defer func() {
			e.active.Add(-1)
			if cancelled {
				e.cancelled.Add(1)
				e.exits <- struct{}{}
			}
		}()
		w.Header().Set("Content-Type", contentType)
		if _, err := io.WriteString(w, chunk); err != nil {
			t.Errorf("write identity engine first chunk: %v", err)
			return
		}
		if err := http.NewResponseController(w).Flush(); err != nil {
			t.Errorf("flush identity engine first chunk: %v", err)
			return
		}
		select {
		case <-r.Context().Done():
			cancelled = true
		case <-release:
		}
	}))
	e.port = portOfURL(t, server.URL)
	t.Cleanup(func() { close(release); server.Close() })
	return e
}

func (e *routingIdentityEngine) awaitCancelled(t *testing.T, want int32) {
	t.Helper()
	select {
	case <-e.exits:
		if got := e.cancelled.Load(); got != want {
			t.Fatalf("engine cancellations = %d, want %d", got, want)
		}
	case <-time.After(10 * time.Second):
		t.Fatalf("engine did not observe cancellation %d", want)
	}
}

func (e *routingIdentityEngine) assertEffects(t *testing.T, calls, cancelled, active int32) {
	t.Helper()
	if gotCalls, gotCancelled, gotActive := e.calls.Load(), e.cancelled.Load(), e.active.Load(); gotCalls != calls || gotCancelled != cancelled || gotActive != active {
		t.Fatalf("engine effects: calls=%d cancelled=%d active=%d; want %d/%d/%d", gotCalls, gotCancelled, gotActive, calls, cancelled, active)
	}
}

func routingIdentityStream(t *testing.T, port int, path, model string) func() {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	t.Cleanup(cancel)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d%s", port, path),
		strings.NewReader(string(egressProcessJSON(t, map[string]any{"model": model, "messages": []string{}, "stream": true}))))
	if err != nil {
		t.Fatalf("create identity inference: %v", err)
	}
	request.Header.Set("Content-Type", "application/json")
	client := egressProcessHTTPClient(t)
	client.Timeout = 0 // The request context owns this deliberately held stream.
	response, err := client.Do(request)
	if err != nil {
		t.Fatalf("start identity inference: %v", err)
	}
	t.Cleanup(func() {
		if err := response.Body.Close(); err != nil {
			t.Errorf("close identity inference: %v", err)
		}
	})
	if response.StatusCode != http.StatusOK {
		t.Fatalf("identity inference status = %d, want 200", response.StatusCode)
	}
	return func() {
		// Cancellation can truncate the stream; only the structured workload
		// and engine context above establish its outcome.
		_, _ = io.Copy(io.Discard, response.Body)
		if err := response.Body.Close(); err != nil {
			t.Fatalf("close cancelled identity inference: %v", err)
		}
	}
}

// Keep the existing departed-run request/control helpers, but supply an
// explicit port plan: startDepartedRunFixture does not expose the randomly
// allocated workload-manager port required by a pinned peer POST.
func routingIdentityBroker(t *testing.T, clusterDir string, ports fixtureServicePorts, enginePort int) *departedRunFixture {
	t.Helper()
	f := &departedRunFixture{enginePort: enginePort, nextID: 1000,
		client: &http.Client{Transport: &http.Transport{}, Timeout: 60 * time.Second}}
	var cancel context.CancelFunc
	f.ctx, cancel = context.WithCancel(context.Background())
	configDir := routingIdentityAppDir(t)
	writePrivateBrokerEngineSettings(t, configDir, clusterDir, map[string]int{"ollama": freePort(t)})
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	scanner := filepath.Join(testbin.Dir(t, "tests", "subscription-scanner", "subscription-scanner"), testbin.Exe("subscription-scanner"))
	jointCopyExecutable(t, self, scanner)
	stdin, messages, stderr, cleanup := startBrokerWithPorts(t, configDir, clusterDir,
		[]string{"OLLAMA_HOST=", "NVPAIR_TEST_SUBSCRIPTION_SCANNER=" + scanner,
			"NVPAIR_TEST_SUBSCRIPTION_PORT=" + strconv.Itoa(freePort(t))}, ports,
		"--scanner-path", scanner, "--engine-manager-path", engineMgrBin,
		"--proxy-path", proxyBin, "--proxy-engines", "ollama", "--workload-manager-path", workloadMgrBin)
	f.stdin = stdin
	control, events, lines := make(chan jsonrpc.Message, 64), make(chan jsonrpc.Message, 64), make(chan string, 8)
	f.control, f.events, f.proxyLines = control, events, lines
	discard := make(chan struct{})
	messageDone, stderrDone := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(messageDone)
		defer close(control)
		defer close(events)
		for msg := range messages {
			target := control
			if strings.HasPrefix(msg.Method, "workloads:") {
				target = events
			} else if msg.ID == nil && msg.Method != "app:ready" {
				continue
			}
			select {
			case target <- msg:
			case <-discard:
			}
		}
	}()
	go func() {
		defer close(stderrDone)
		defer close(lines)
		for line := range stderr {
			if proxyPidRe.MatchString(line) {
				select {
				case lines <- line:
				case <-discard:
				}
			}
		}
	}()
	t.Cleanup(func() {
		cancel()
		f.client.CloseIdleConnections()
		joined := make(chan struct{})
		go func() { f.requests.Wait(); close(joined) }()
		select {
		case <-joined:
		case <-time.After(8 * time.Second):
			t.Error("identity HTTP clients did not exit within cleanup deadline")
		}
		close(discard)
		cleanup()
		for _, done := range []<-chan struct{}{messageDone, stderrDone} {
			select {
			case <-done:
			case <-time.After(8 * time.Second):
				t.Error("identity broker pipe reader did not join within cleanup deadline")
			}
		}
	})
	waitForMethod(t, f.control, "app:ready", 10*time.Second)
	reply := f.call(t, "workloads:subscribe", `{}`)
	egressProcessOK(t, reply)
	var subscription subscriptionResult
	egressProcessDecode(t, reply.Result, &subscription)
	if !subscription.Subscribed {
		t.Fatal("identity broker did not acknowledge workload subscription")
	}
	f.port = waitProxyReady(t, f.stdin, f.control, 15*time.Second)
	f.selectEngine(t)
	return f
}
