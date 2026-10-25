// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
)

func TestWorkloadDepartedRunRetired(t *testing.T) {
	if portBusy(11435) {
		t.Skip("ollama-proxy default port 11435 already in use; skipping")
	}

	t.Run("OriginRetiresAndReplacementCancels", func(t *testing.T) {
		f := startDepartedRunFixture(t, t.TempDir())
		pid1 := mustPid(t, waitForStderr(t, f.proxyLines, proxyPidRe, 10*time.Second))
		firstRequest := f.startRequest(t)
		// Consume the submitted frame with the existing helper; the running
		// barrier below also retains the run, sequence and terminal metadata.
		submitted := waitForWorkloadEvent(t, f.events, "workloads:upsert", "1", 15*time.Second)
		if submitted.WorkloadInfo.State != "queued" {
			t.Fatalf("first workload state = %q, want queued before running", submitted.WorkloadInfo.State)
		}
		running := waitDepartedRunFrame(t, f.events, "workloads:upsert", 15*time.Second, func(p departedRunParams) bool {
			return p.WorkloadInfo.State == "running"
		}).WorkloadInfo
		assertDepartedRunRunning(t, running)

		killDepartedRunProxy(t, pid1)
		failed := waitDepartedRunFrame(t, f.events, "workloads:upsert", 15*time.Second, func(p departedRunParams) bool {
			return p.WorkloadInfo.key() == running.key() && p.WorkloadInfo.State == "failed"
		}).WorkloadInfo
		assertDepartedRunTerminal(t, failed, running, "failed", "proxy exited before the request finished")
		// During the respawn gap an RPC error is also an honest refusal.
		refused := f.cancel(t, running)
		if refused.Error == nil {
			assertDepartedRunCancelResult(t, refused, false)
		}
		assertDepartedRunTerminal(t, f.initialWorkload(t, running.key()), running, "failed", "proxy exited before the request finished")
		joinDepartedRunRequest(t, firstRequest)

		pid2 := mustPid(t, waitForStderr(t, f.proxyLines, proxyPidRe, 15*time.Second))
		if pid2 == pid1 {
			t.Fatalf("replacement proxy PID = %d, want a different PID", pid2)
		}
		f.port = waitProxyReady(t, f.stdin, f.control, 15*time.Second)
		f.selectEngine(t)
		secondRequest := f.startRequest(t)
		replacement := waitDepartedRunFrame(t, f.events, "workloads:upsert", 15*time.Second, func(p departedRunParams) bool {
			return p.WorkloadInfo.RunID != running.RunID && p.WorkloadInfo.State == "running"
		}).WorkloadInfo
		assertDepartedRunRunning(t, replacement)
		if replacement.OriginatedFrom != running.OriginatedFrom || replacement.Engine != running.Engine || replacement.ID != running.ID {
			t.Fatalf("replacement identity = %+v, want the same origin, engine and reused workload ID as %+v", replacement.key(), running.key())
		}

		assertDepartedRunCancelResult(t, f.cancel(t, running), false)
		// Observe the entire window, including notifications that arrived before
		// the cancel reply. An early HTTP exit is also a failure of isolation.
		window := time.NewTimer(2 * time.Second)
		defer window.Stop()
	observing:
		for {
			select {
			case msg, ok := <-f.events:
				if !ok {
					t.Fatal("broker workload stream closed during old-run cancellation observation")
				}
				p := decodeDepartedRunFrame(t, msg)
				if p.WorkloadInfo.key() == replacement.key() && p.WorkloadInfo.State != "running" {
					t.Fatalf("old-run cancel changed replacement state to %q", p.WorkloadInfo.State)
				}
			case result := <-secondRequest:
				t.Fatalf("replacement HTTP request ended after old-run cancel: status=%d read=%v close=%v", result.status, result.readErr, result.closeErr)
			case <-window.C:
				break observing
			}
		}
		if current := f.initialWorkload(t, replacement.key()); current.State != "running" {
			t.Fatalf("replacement snapshot state after old-run cancel = %q, want running", current.State)
		}
		assertDepartedRunCancelResult(t, f.cancel(t, replacement), true)
		cancelled := waitDepartedRunFrame(t, f.events, "workloads:upsert", 15*time.Second, func(p departedRunParams) bool {
			return p.WorkloadInfo.key() == replacement.key() && p.WorkloadInfo.State == "cancelled"
		}).WorkloadInfo
		assertDepartedRunTerminal(t, cancelled, replacement, "cancelled", "cancelled by request")
		joinDepartedRunRequest(t, secondRequest)
	})

	t.Run("PeerHearsFailed", func(t *testing.T) {
		// Like the workload-manager restart neighbours, this needs working mDNS
		// and a cross-pinned joined cluster, outside the network sandbox.
		cluster := newInterNodeCluster(t)
		received := cluster.startStubClusterPeer(t, "departed-run-peer", "departed-run-peer-uuid")
		f := startDepartedRunFixture(t, cluster.nodeDir, fixtureMDNSBrokerArgs(cluster.namespace)...)
		pid := mustPid(t, waitForStderr(t, f.proxyLines, proxyPidRe, 10*time.Second))
		request := f.startRequest(t)
		running := waitDepartedRunFrame(t, f.events, "workloads:upsert", 15*time.Second, func(p departedRunParams) bool {
			return p.WorkloadInfo.State == "running"
		}).WorkloadInfo
		assertDepartedRunRunning(t, running)
		peerRunning := waitDepartedRunFrame(t, received, "workload:started", 45*time.Second, func(p departedRunParams) bool {
			return p.WorkloadInfo.key() == running.key() && p.WorkloadInfo.State == "running"
		}).WorkloadInfo
		if !reflect.DeepEqual(peerRunning, running) {
			t.Fatalf("peer running workload differs from the origin: peer=%+v origin=%+v", peerRunning, running)
		}
		drainWorkloadMessages(received)
		killDepartedRunProxy(t, pid)

		direct := waitDepartedRunFrame(t, received, "workload:errored", 15*time.Second, func(p departedRunParams) bool {
			return p.WorkloadInfo.key() == running.key() && p.WorkloadInfo.State == "failed" && !p.Resync
		}).WorkloadInfo
		assertDepartedRunTerminal(t, direct, running, "failed", "proxy exited before the request finished")
		// A second direct delivery cannot satisfy anti-entropy: the manager
		// must reassert the same terminal with its existing resync marker.
		heartbeat := waitDepartedRunFrame(t, received, "workload:errored", 45*time.Second, func(p departedRunParams) bool {
			return p.WorkloadInfo.key() == running.key() && p.WorkloadInfo.State == "failed" && p.Resync
		}).WorkloadInfo
		assertDepartedRunTerminal(t, heartbeat, running, "failed", "proxy exited before the request finished")
		if !reflect.DeepEqual(heartbeat, direct) {
			t.Fatalf("terminal reassertion differs from direct broadcast: reassertion=%+v direct=%+v", heartbeat, direct)
		}
		joinDepartedRunRequest(t, request)
	})
}

// Keep the extra lifecycle fields here: the shared wlInfo intentionally only
// covers the smaller assertions made by its existing callers.
type departedRunInfo struct {
	wlInfo
	RunID          string  `json:"runId"`
	Seq            int64   `json:"seq"`
	CreatedAt      int64   `json:"createdAt"`
	StartedAt      *int64  `json:"startedAt"`
	CompletedAt    *int64  `json:"completedAt"`
	Error          *string `json:"error"`
	ScheduledOn    string  `json:"scheduledOn"`
	RequesterID    *string `json:"requesterId"`
	RequestedModel string  `json:"requestedModel"`
}

type departedRunKey struct {
	origin, engine, run, id string
}

func (w departedRunInfo) key() departedRunKey {
	return departedRunKey{w.OriginatedFrom, w.Engine, w.RunID, w.ID}
}

type departedRunParams struct {
	WorkloadInfo departedRunInfo `json:"workloadInfo"`
	Resync       bool            `json:"resync"`
}

func decodeDepartedRunFrame(t *testing.T, msg jsonrpc.Message) departedRunParams {
	t.Helper()
	var p departedRunParams
	if err := json.Unmarshal(msg.Params, &p); err != nil {
		t.Fatalf("decode %s: %v", msg.Method, err)
	}
	return p
}

func waitDepartedRunFrame(t *testing.T, messages <-chan jsonrpc.Message, method string, timeout time.Duration, matches func(departedRunParams) bool) departedRunParams {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			t.Fatalf("timed out (%s) waiting for matching %s workload", timeout, method)
		}
		p := decodeDepartedRunFrame(t, waitForMethod(t, messages, method, remaining))
		if matches(p) {
			return p
		}
	}
}

func assertDepartedRunRunning(t *testing.T, w departedRunInfo) {
	t.Helper()
	if w.ID != "1" || w.Engine != "ollama" || w.RunID == "" || w.OriginatedFrom == "" || w.Model != "departed-run-model" {
		t.Fatalf("running workload identity/model = %+v, want complete ollama identity, reused ID 1 and departed-run-model", w)
	}
	if w.State != "running" || w.Seq <= 0 || w.StartedAt == nil || w.CompletedAt != nil || w.Error != nil {
		t.Fatalf("workload did not commit a nonterminal running frame: %+v", w)
	}
}

func assertDepartedRunTerminal(t *testing.T, got, running departedRunInfo, state, message string) {
	t.Helper()
	if got.key() != running.key() || got.State != state || got.Error == nil || *got.Error != message || got.CompletedAt == nil || got.Seq <= running.Seq {
		t.Fatalf("terminal workload = %+v, want key=%+v state=%q error=%q completedAt present and seq > %d", got, running.key(), state, message, running.Seq)
	}
	if got.Model != running.Model || got.ScheduledOn != running.ScheduledOn || got.CreatedAt != running.CreatedAt || got.RequestedModel != running.RequestedModel || !reflect.DeepEqual(got.StartedAt, running.StartedAt) || !reflect.DeepEqual(got.RequesterID, running.RequesterID) {
		t.Fatalf("terminal changed request metadata: terminal=%+v running=%+v", got, running)
	}
}

func assertDepartedRunCancelResult(t *testing.T, reply jsonrpc.Message, want bool) {
	t.Helper()
	if reply.Error != nil {
		t.Fatalf("workloads:cancel RPC error: %+v", reply.Error)
	}
	var result struct {
		OK *bool `json:"ok"`
	}
	if err := json.Unmarshal(reply.Result, &result); err != nil {
		t.Fatalf("decode workloads:cancel result: %v", err)
	}
	if result.OK == nil || *result.OK != want {
		t.Fatalf("workloads:cancel result = %s, want ok:%t", reply.Result, want)
	}
}

func killDepartedRunProxy(t *testing.T, pid int) {
	t.Helper()
	process, err := os.FindProcess(pid)
	if err != nil {
		t.Fatalf("find proxy PID %d: %v", pid, err)
	}
	defer func() {
		if err := process.Release(); err != nil {
			t.Errorf("release proxy PID %d handle: %v", pid, err)
		}
	}()
	if err := process.Kill(); err != nil {
		t.Fatalf("kill proxy PID %d: %v", pid, err)
	}
}

type departedRunHTTPResult struct {
	status   int
	readErr  error
	closeErr error
}

type departedRunFixture struct {
	stdin      io.WriteCloser
	control    <-chan jsonrpc.Message
	events     <-chan jsonrpc.Message
	proxyLines <-chan string
	port       int
	enginePort int
	nextID     int
	client     *http.Client
	ctx        context.Context
	requests   sync.WaitGroup
}

func startDepartedRunFixture(t *testing.T, clusterDir string, extraArgs ...string) *departedRunFixture {
	t.Helper()
	configDir := t.TempDir()
	release := make(chan struct{})
	var releaseOnce sync.Once
	stopEngine := func() { releaseOnce.Do(func() { close(release) }) }
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Errorf("read fake engine request: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)
		if _, err := io.WriteString(w, `{"message":{"role":"assistant","content":"ready"},"done":false}`+"\n"); err != nil {
			t.Errorf("write fake engine first chunk: %v", err)
			return
		}
		if err := http.NewResponseController(w).Flush(); err != nil {
			t.Errorf("flush fake engine first chunk: %v", err)
			return
		}
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	// This also covers a fatal setup error before broker cleanup is registered.
	t.Cleanup(func() {
		stopEngine()
		engine.Close()
	})
	f := &departedRunFixture{enginePort: portOfURL(t, engine.URL), nextID: 1000,
		client: &http.Client{Transport: &http.Transport{}, Timeout: 120 * time.Second}}
	var cancel context.CancelFunc
	f.ctx, cancel = context.WithCancel(context.Background())
	args := append([]string{
		"--proxy-path", proxyBin, "--proxy-engines", "ollama",
		"--workload-manager-path", workloadMgrBin,
	}, extraArgs...)
	stdin, messages, stderr, cleanup := startBrokerWithDirs(t, configDir, clusterDir, args...)
	f.stdin = stdin
	control, events := make(chan jsonrpc.Message, 64), make(chan jsonrpc.Message, 64)
	proxyLines := make(chan string, 8)
	f.control, f.events, f.proxyLines = control, events, proxyLines
	discard := make(chan struct{})
	messagesDone, stderrDone := make(chan struct{}), make(chan struct{})
	// Separate control replies from workload events so an RPC waiter cannot
	// discard a terminal notification racing its reply. Keep draining both pipes
	// while the peer test waits for the manager's heartbeat.
	go func() {
		defer close(messagesDone)
		defer close(control)
		defer close(events)
		for msg := range messages {
			var target chan jsonrpc.Message
			switch {
			case strings.HasPrefix(msg.Method, "workloads:"):
				target = events
			case msg.ID != nil || msg.Method == "app:ready":
				target = control
			default:
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
		defer close(proxyLines)
		for line := range stderr {
			if proxyPidRe.MatchString(line) {
				select {
				case proxyLines <- line:
				case <-discard:
				}
			}
		}
	}()
	t.Cleanup(func() {
		stopEngine()
		cancel()
		f.client.CloseIdleConnections()
		joined := make(chan struct{})
		go func() { f.requests.Wait(); close(joined) }()
		select {
		case <-joined:
		case <-time.After(5 * time.Second):
			t.Error("HTTP clients did not join within cleanup deadline")
		}
		close(discard)
		cleanup()
		for _, done := range []<-chan struct{}{messagesDone, stderrDone} {
			select {
			case <-done:
			case <-time.After(3 * time.Second):
				t.Error("broker pipe reader did not join within cleanup deadline")
			}
		}
	})

	waitForMethod(t, f.control, "app:ready", 10*time.Second)
	reply := f.call(t, "workloads:subscribe", `{}`)
	var subscription subscriptionResult
	if reply.Error != nil {
		t.Fatalf("workloads:subscribe RPC error: %+v", reply.Error)
	}
	if err := json.Unmarshal(reply.Result, &subscription); err != nil {
		t.Fatalf("decode workloads:subscribe result: %v", err)
	}
	if !subscription.Subscribed {
		t.Fatal("workloads:subscribe did not acknowledge subscribed:true")
	}
	f.port = waitProxyReady(t, f.stdin, f.control, 15*time.Second)
	f.selectEngine(t)
	return f
}

func (f *departedRunFixture) call(t *testing.T, method, params string) jsonrpc.Message {
	t.Helper()
	f.nextID++
	writeRawFrame(t, f.stdin, fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"method":%q,"params":%s}`, f.nextID, method, params))
	deadline := time.Now().Add(5 * time.Second)
	for {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			t.Fatalf("timed out waiting for %s reply %d", method, f.nextID)
		}
		reply := waitForResponse(t, f.control, remaining)
		var id int
		if err := json.Unmarshal(*reply.ID, &id); err != nil {
			t.Fatalf("decode %s reply ID: %v", method, err)
		}
		if id == f.nextID {
			return reply
		}
	}
}

func (f *departedRunFixture) selectEngine(t *testing.T) {
	t.Helper()
	params := fmt.Sprintf(`{"id":"fake-departed-ollama","host":"127.0.0.1","port":%d,"addresses":["127.0.0.1"],"models":["departed-run-model"]}`, f.enginePort)
	if reply := f.call(t, "ollama-proxy:node/add-manual", params); reply.Error != nil {
		t.Fatalf("add fake engine: %+v", reply.Error)
	}
	if reply := f.call(t, "ollama-proxy:node/select", `{"id":"fake-departed-ollama"}`); reply.Error != nil {
		t.Fatalf("select fake engine: %+v", reply.Error)
	}
}

func (f *departedRunFixture) cancel(t *testing.T, w departedRunInfo) jsonrpc.Message {
	t.Helper()
	return f.call(t, "workloads:cancel", fmt.Sprintf(`{"originatedFrom":%q,"workloadId":%q,"engine":%q,"runId":%q}`, w.OriginatedFrom, w.ID, w.Engine, w.RunID))
}

func (f *departedRunFixture) initialWorkload(t *testing.T, key departedRunKey) departedRunInfo {
	t.Helper()
	reply := f.call(t, "workloads:get-initial", `{}`)
	if reply.Error != nil {
		t.Fatalf("workloads:get-initial RPC error: %+v", reply.Error)
	}
	var initial struct {
		Workloads []departedRunInfo `json:"workloads"`
	}
	if err := json.Unmarshal(reply.Result, &initial); err != nil {
		t.Fatalf("decode workloads:get-initial: %v", err)
	}
	for _, w := range initial.Workloads {
		if w.key() == key {
			return w
		}
	}
	t.Fatalf("workloads:get-initial omitted key %+v", key)
	return departedRunInfo{}
}

func (f *departedRunFixture) startRequest(t *testing.T) <-chan departedRunHTTPResult {
	t.Helper()
	req, err := http.NewRequestWithContext(f.ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/api/chat", f.port), strings.NewReader(`{"model":"departed-run-model","messages":[],"stream":true}`))
	if err != nil {
		t.Fatalf("create inference request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	headers := make(chan error, 1)
	done := make(chan departedRunHTTPResult, 1)
	f.requests.Add(1)
	go func() {
		defer f.requests.Done()
		resp, err := f.client.Do(req)
		if err != nil {
			headers <- err
			return
		}
		if resp.StatusCode != http.StatusOK {
			headers <- fmt.Errorf("inference HTTP status = %d, want %d", resp.StatusCode, http.StatusOK)
		} else {
			headers <- nil
		}
		_, readErr := io.Copy(io.Discard, resp.Body)
		closeErr := resp.Body.Close()
		done <- departedRunHTTPResult{status: resp.StatusCode, readErr: readErr, closeErr: closeErr}
	}()
	select {
	case err := <-headers:
		if err != nil {
			t.Fatalf("inference request did not commit HTTP 200: %v", err)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("inference HTTP headers did not arrive within deadline")
	}
	return done
}

func joinDepartedRunRequest(t *testing.T, done <-chan departedRunHTTPResult) {
	t.Helper()
	select {
	case result := <-done:
		if result.status != http.StatusOK || result.closeErr != nil {
			t.Fatalf("inference cleanup: status=%d close=%v", result.status, result.closeErr)
		}
		// A killed proxy can tear the committed HTTP stream; the authoritative
		// workload outcome, rather than EOF versus a read error, is asserted above.
	case <-time.After(5 * time.Second):
		t.Fatal("inference HTTP client did not exit within deadline")
	}
}
