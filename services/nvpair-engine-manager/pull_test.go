// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/modeldownload"
)

func TestPullProgressFromLine(t *testing.T) {
	ev := pullProgressFromLine("ollama", []byte(`{"status":"pulling manifest","total":200,"completed":50}`))
	if ev.Op != "pull" || ev.Engine != "ollama" {
		t.Fatalf("unexpected event meta: %+v", ev)
	}
	if ev.Percent != 25 {
		t.Fatalf("expected 25%%, got %d", ev.Percent)
	}
	if ev.Stage != "pulling manifest" {
		t.Fatalf("expected stage from status, got %q", ev.Stage)
	}

	// No total -> 0% (avoids divide-by-zero), still carries the status.
	ev = pullProgressFromLine("ollama", []byte(`{"status":"verifying"}`))
	if ev.Percent != 0 || ev.Stage != "verifying" {
		t.Fatalf("unexpected zero-total event: %+v", ev)
	}
}

func TestHandlePullRejectsMissingTarget(t *testing.T) {
	s := &controlServer{exec: &Executor{progress: newProgressHub()}}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", controlPullPath, strings.NewReader(`{"opId":"x","engine":"ollama"}`))
	s.handlePull(rec, req)
	if rec.Code != 400 {
		t.Fatalf("expected 400 when neither model nor params set, got %d", rec.Code)
	}
}

func TestModelFromParams(t *testing.T) {
	cases := []struct {
		params, want string
	}{
		{`{"name":"llama3.2"}`, "llama3.2"},      // Ollama body key
		{`{"model":"owner/repo"}`, "owner/repo"}, // generic placeholder
		{`{"name":"a","model":"b"}`, "a"},        // name wins
		{`{}`, ""},                               // neither present
		{``, ""},                                 // empty params
	}
	for _, c := range cases {
		if got := modelFromParams([]byte(c.params)); got != c.want {
			t.Fatalf("modelFromParams(%q) = %q, want %q", c.params, got, c.want)
		}
	}
}

func TestLlamaCPPModelsEventPercentAggregatesFiles(t *testing.T) {
	var event llamaCPPModelsEvent
	err := json.Unmarshal([]byte(`{
		"model":"owner/repo:Q4_K_M",
		"event":"download_progress",
		"data":{"progress":{
			"model.gguf":{"done":75,"total":100},
			"mmproj.gguf":{"done":25,"total":100}
		}}
	}`), &event)
	if err != nil {
		t.Fatalf("decode event: %v", err)
	}
	if got := event.percent(); got != 50 {
		t.Fatalf("aggregate percent = %d, want 50", got)
	}
}

func newHTTPPullTestExecutor(t *testing.T, server *httptest.Server, action Action) *Executor {
	t.Helper()
	_, portText, err := net.SplitHostPort(server.Listener.Addr().String())
	if err != nil {
		t.Fatalf("split test server address: %v", err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatalf("parse test server port: %v", err)
	}
	manifest := testEngineManifest(fakeEngineBin)
	manifest.Actions[pullModelAction] = action
	registry := NewRegistry()
	registry.engines[manifest.Engine] = manifest
	executor := NewExecutor(registry, NewReporter(nil), nil, t.TempDir())
	acceptModelMutations(executor)
	state, err := executor.state(manifest.Engine)
	if err != nil {
		t.Fatalf("resolve engine state: %v", err)
	}
	state.running = true
	state.port = port
	return executor
}

func TestPullModelLlamaCPPSSESubscribesBeforeStarting(t *testing.T) {
	const model = "owner/repo:Q4_K_M"
	subscribed := make(chan struct{})
	started := make(chan struct{})
	var postBeforeSubscribe atomic.Bool
	mux := http.NewServeMux()
	mux.HandleFunc("/models/sse", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		close(subscribed)
		_, _ = fmt.Fprint(w, ": ready\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-started:
		case <-r.Context().Done():
			return
		}
		write := func(event map[string]any) {
			data, err := json.Marshal(event)
			if err != nil {
				t.Errorf("encode SSE event: %v", err)
				return
			}
			_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
			w.(http.Flusher).Flush()
		}
		write(map[string]any{"model": "other/model", "event": "download_finished", "data": map[string]any{}})
		write(map[string]any{
			"model": model,
			"event": "download_progress",
			"data": map[string]any{"progress": map[string]any{
				"model.gguf":  map[string]int64{"done": 75, "total": 100},
				"mmproj.gguf": map[string]int64{"done": 25, "total": 100},
			}},
		})
		write(map[string]any{"model": model, "event": "download_finished", "data": map[string]any{}})
	})
	mux.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-subscribed:
		default:
			postBeforeSubscribe.Store(true)
		}
		var body struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Model != model {
			http.Error(w, "invalid model", http.StatusBadRequest)
			return
		}
		close(started)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"success":true}`)
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	ex := newHTTPPullTestExecutor(t, server, Action{
		HTTP:             &ActionHTTP{Method: http.MethodPost, Path: "/models"},
		ProgressProtocol: pullProgressProtocolLlamaCPPModelsSSE,
	})
	progress, cancel := ex.progress.subscribe("fake")
	defer cancel()

	result, err := ex.PullModelStream(context.Background(), "fake", model, json.RawMessage(`{"model":"`+model+`"}`))
	if err != nil {
		t.Fatalf("pull model: %v", err)
	}
	var response struct {
		Success bool `json:"success"`
	}
	if err := json.Unmarshal(result, &response); err != nil || !response.Success {
		t.Fatalf("pull result = %s, error %v", result, err)
	}
	if postBeforeSubscribe.Load() {
		t.Fatal("download POST arrived before the SSE subscription was open")
	}
	var events []ProgressEvent
	for len(progress) > 0 {
		events = append(events, <-progress)
	}
	if len(events) != 4 {
		t.Fatalf("progress events = %+v, want queued, pulling, downloading and success", events)
	}
	if events[0].Stage != "queued" || events[1].Stage != "pulling" {
		t.Fatalf("initial events = %+v", events[:2])
	}
	if events[2].Stage != "downloading" || events[2].Percent != 50 {
		t.Fatalf("download event = %+v, want 50%%", events[2])
	}
	if events[3].Stage != "success" || events[3].Percent != 100 {
		t.Fatalf("terminal event = %+v, want success at 100%%", events[3])
	}
	for _, event := range events {
		if event.ID == "" || event.Model != model || event.DownloadState == nil {
			t.Fatalf("missing target identity: %+v", event)
		}
	}
}

func TestPullModelOllamaAdvancingProgressRefreshesTimeout(t *testing.T) {
	const (
		idleTimeout     = 400 * time.Millisecond
		progressDelay   = 90 * time.Millisecond
		progressUpdates = 6
	)
	mux := http.NewServeMux()
	mux.HandleFunc("/api/pull", func(w http.ResponseWriter, _ *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Error("test response does not support flushing")
			return
		}
		for completed := 1; completed <= progressUpdates; completed++ {
			if _, err := fmt.Fprintf(
				w,
				"{\"status\":\"pulling\",\"digest\":\"sha256:model\",\"total\":%d,\"completed\":%d}\n",
				progressUpdates,
				completed,
			); err != nil {
				return
			}
			flusher.Flush()
			time.Sleep(progressDelay)
		}
		_, _ = fmt.Fprintln(w, `{"status":"success"}`)
		flusher.Flush()
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	ex := newHTTPPullTestExecutor(t, server, Action{
		HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/api/pull"},
	})
	ex.pullProgressTimeout = idleTimeout

	started := time.Now()
	result, err := ex.PullModelStream(
		context.Background(),
		"fake",
		"demo:1b",
		json.RawMessage(`{"name":"demo:1b"}`),
	)
	if err != nil {
		t.Fatalf("pull model: %v", err)
	}
	if elapsed := time.Since(started); elapsed <= idleTimeout {
		t.Fatalf("pull completed in %s, want longer than one %s idle interval", elapsed, idleTimeout)
	}
	if !strings.Contains(string(result), `"status":"success"`) {
		t.Fatalf("pull result = %s, want terminal success", result)
	}
}

func TestPullModelLlamaCPPAdvancingProgressRefreshesTimeout(t *testing.T) {
	const (
		model           = "owner/repo:Q4_K_M"
		idleTimeout     = 400 * time.Millisecond
		progressDelay   = 90 * time.Millisecond
		progressUpdates = 6
	)
	started := make(chan struct{})
	mux := http.NewServeMux()
	mux.HandleFunc("/models/sse", func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Error("test response does not support flushing")
			return
		}
		_, _ = fmt.Fprint(w, ": ready\n\n")
		flusher.Flush()
		select {
		case <-started:
		case <-r.Context().Done():
			return
		}
		for completed := 1; completed <= progressUpdates; completed++ {
			if _, err := fmt.Fprintf(
				w,
				"data: {\"model\":%q,\"event\":\"download_progress\",\"data\":{\"progress\":{\"model.gguf\":{\"done\":%d,\"total\":%d}}}}\n\n",
				model,
				completed,
				progressUpdates,
			); err != nil {
				return
			}
			flusher.Flush()
			time.Sleep(progressDelay)
		}
		_, _ = fmt.Fprintf(w, "data: {\"model\":%q,\"event\":\"download_finished\",\"data\":{}}\n\n", model)
		flusher.Flush()
	})
	mux.HandleFunc("/models", func(w http.ResponseWriter, _ *http.Request) {
		close(started)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"success":true}`)
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	ex := newHTTPPullTestExecutor(t, server, Action{
		HTTP:             &ActionHTTP{Method: http.MethodPost, Path: "/models"},
		ProgressProtocol: pullProgressProtocolLlamaCPPModelsSSE,
	})
	ex.pullProgressTimeout = idleTimeout

	startedAt := time.Now()
	result, err := ex.PullModelStream(
		context.Background(),
		"fake",
		model,
		json.RawMessage(`{"model":"`+model+`"}`),
	)
	if err != nil {
		t.Fatalf("pull model: %v", err)
	}
	if elapsed := time.Since(startedAt); elapsed <= idleTimeout {
		t.Fatalf("pull completed in %s, want longer than one %s idle interval", elapsed, idleTimeout)
	}
	if !strings.Contains(string(result), `"success":true`) {
		t.Fatalf("pull result = %s, want accepted download response", result)
	}
}

func TestPullModelDuplicateProgressDoesNotRefreshTimeout(t *testing.T) {
	const idleTimeout = 150 * time.Millisecond
	mux := http.NewServeMux()
	mux.HandleFunc("/api/pull", func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Error("test response does not support flushing")
			return
		}
		ticker := time.NewTicker(25 * time.Millisecond)
		defer ticker.Stop()
		for {
			_, err := fmt.Fprintln(w, `{"status":"pulling","digest":"sha256:model","total":100,"completed":1}`)
			if err != nil {
				return
			}
			flusher.Flush()
			select {
			case <-r.Context().Done():
				return
			case <-ticker.C:
			}
		}
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	ex := newHTTPPullTestExecutor(t, server, Action{
		HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/api/pull"},
	})
	ex.pullProgressTimeout = idleTimeout

	_, err := ex.PullModelStream(
		context.Background(),
		"fake",
		"demo:1b",
		json.RawMessage(`{"name":"demo:1b"}`),
	)
	if err == nil {
		t.Fatal("pull model succeeded despite duplicate-only progress")
	}
	if want := "model download made no progress for 150ms"; !strings.Contains(err.Error(), want) {
		t.Fatalf("pull error = %q, want %q", err, want)
	}
}

func TestPullProgressWatchdogPreservesParentCancellation(t *testing.T) {
	parent, cancel := context.WithCancel(context.Background())
	ctx, watchdog := newPullProgressWatchdog(parent, time.Hour)
	defer watchdog.stop()

	cancel()
	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("watchdog context did not observe parent cancellation")
	}
	if cause := context.Cause(ctx); cause != context.Canceled {
		t.Fatalf("watchdog cause = %v, want context canceled", cause)
	}
}

// TestActionPullModelStreamsProgress verifies that engine:action with action
// "pull_model" is routed through the streaming pull path, so a local pull
// emits live engine:pull-progress notifications (with computed percentages) and
// still returns the pull's terminal result — matching what remote pulls already
// surface via engine:remote-progress.
func TestActionPullModelStreamsProgress(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	m.Actions["pull_model"] = Action{HTTP: &ActionHTTP{Method: "POST", Path: "/api/pull"}}

	var mu sync.Mutex
	var pulls []map[string]any
	reg := NewRegistry()
	reg.engines[m.Engine] = m
	ex := NewExecutor(reg, NewReporter(nil), func(method string, params any) {
		if method != "engine:pull-progress" {
			return
		}
		mp, _ := params.(map[string]any)
		mu.Lock()
		pulls = append(pulls, mp)
		mu.Unlock()
	}, t.TempDir())

	ctx := context.Background()
	t.Cleanup(func() { _ = ex.Stop("fake") })
	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}

	var out bytes.Buffer
	mgr := NewManager(NewCodec(&out), ex, nil, "")
	acceptModelMutations(ex)
	id := json.RawMessage("7")
	mgr.runAction(ctx, &Message{JSONRPC: "2.0", ID: &id, Method: "engine:action",
		Params: json.RawMessage(`{"engine":"fake","action":"pull_model","params":{"name":"demo:1b"}}`)})

	if !strings.Contains(out.String(), "success") {
		t.Fatalf("expected the pull's terminal result in the response, got %s", out.String())
	}

	mu.Lock()
	defer mu.Unlock()
	if len(pulls) == 0 {
		t.Fatal("engine:action{action:pull_model} emitted no engine:pull-progress notifications")
	}
	sawPercent := false
	for _, p := range pulls {
		if p["op"] != "pull" {
			t.Fatalf("expected op=pull on every frame, got %+v", p)
		}
		if pct, ok := p["percent"].(int); ok && pct > 0 {
			sawPercent = true
		}
	}
	if !sawPercent {
		t.Fatalf("expected at least one frame with a computed percent > 0, got %+v", pulls)
	}
}

// TestActionPullModelCmdMarkerAndResult covers the CLI (LM Studio `lms get`)
// pull path through the same engine:action routing: a Cmd-based pull_model can't
// expose structured line progress, so it emits a single "pulling" marker and
// returns the command's terminal result — the counterpart to the HTTP streaming
// path in TestActionPullModelStreamsProgress.
func TestActionPullModelCmdMarkerAndResult(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	m.Actions["pull_model"] = Action{Cmd: []string{fakeEngineBin, "echo", "pulled:{model}"}}

	var mu sync.Mutex
	var pulls []map[string]any
	reg := NewRegistry()
	reg.engines[m.Engine] = m
	ex := NewExecutor(reg, NewReporter(nil), func(method string, params any) {
		if method != "engine:pull-progress" {
			return
		}
		mp, _ := params.(map[string]any)
		mu.Lock()
		pulls = append(pulls, mp)
		mu.Unlock()
	}, t.TempDir())

	// A Cmd action just runs a binary; the engine need not be started.
	var out bytes.Buffer
	mgr := NewManager(NewCodec(&out), ex, nil, "")
	acceptModelMutations(ex)
	id := json.RawMessage("8")
	mgr.runAction(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:action",
		Params: json.RawMessage(`{"engine":"fake","action":"pull_model","params":{"model":"demo:1b"}}`)})

	if !strings.Contains(out.String(), "pulled:demo:1b") {
		t.Fatalf("expected the cmd pull's terminal result in the response, got %s", out.String())
	}

	mu.Lock()
	defer mu.Unlock()
	if len(pulls) != 3 {
		t.Fatalf("expected queued, pulling and complete for a CLI pull, got %d: %+v", len(pulls), pulls)
	}
	if pulls[0]["stage"] != "queued" || pulls[1]["op"] != "pull" || pulls[1]["stage"] != "pulling" || pulls[1]["model"] != "demo:1b" || pulls[2]["stage"] != "success" {
		t.Fatalf("unexpected CLI lifecycle: %+v", pulls)
	}
	if _, hasPercent := pulls[1]["percent"]; hasPercent {
		t.Fatalf("CLI pull marker must omit indeterminate percent, got %+v", pulls[1])
	}
	state, err := ex.state("fake")
	if err != nil {
		t.Fatalf("read CLI engine state: %v", err)
	}
	state.mu.Lock()
	running, run := state.running, state.runID
	state.mu.Unlock()
	if running || run != "" {
		t.Fatalf("CLI pull required or started an engine: running=%t run=%q", running, run)
	}
	snapshot := ex.downloads.snapshot()
	if len(snapshot.Entries) != 1 || snapshot.Entries[0].Phase != modeldownload.PhaseComplete || snapshot.Entries[0].BlocksReadiness {
		t.Fatalf("stopped CLI pull did not complete: %+v", snapshot)
	}
	markers, err := ex.downloads.store.load()
	if err != nil || len(markers) != 0 {
		t.Fatalf("stopped CLI pull retained unsafe markers: %+v, %v", markers, err)
	}
}

func TestModelDownloadNativeCompletionBelongsToDispatchRun(t *testing.T) {
	test := func(name string, retire func(*engineState), phase string, blocked bool) {
		t.Run(name, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			acceptModelMutations(ex)
			st := ex.engines["ollama"]
			params := json.RawMessage(`{"target":"demo","TARGET":"opaque\nvariant"}`)
			started, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var once sync.Once
			var effects atomic.Int32
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
				effects.Add(1)
				body, err := io.ReadAll(r.Body)
				if err != nil {
					return nil, err
				}
				if r.Method != http.MethodPost || r.URL.Path != "/pull" || !bytes.Equal(body, params) {
					return nil, errors.New("pull changed its native dispatch or caller params")
				}
				close(started)
				select {
				case <-release:
					return mutationJSON(http.StatusOK, `{"status":"success"}`)
				case <-r.Context().Done():
					return nil, r.Context().Err()
				}
			})
			var pullErr error
			go func() {
				_, pullErr = ex.PullModelStream(ctx, "ollama", "demo", params)
				close(finished)
			}()
			t.Cleanup(func() {
				once.Do(func() { close(release) })
				cancel()
				select {
				case <-finished:
				case <-time.After(5 * time.Second):
					t.Error("pull did not join after releasing native completion")
				}
			})
			select {
			case <-started:
			case <-ctx.Done():
				t.Fatalf("pull did not reach native completion barrier: %v", ctx.Err())
			}
			before := ex.downloads.snapshot()
			if len(before.Entries) != 1 || before.Entries[0].Phase != modeldownload.PhaseInProgress || !before.Entries[0].BlocksReadiness {
				t.Fatalf("pull was not unsafe before terminal: %+v", before)
			}
			st.mu.Lock()
			retire(st)
			st.mu.Unlock()
			once.Do(func() { close(release) })
			select {
			case <-finished:
			case <-ctx.Done():
				t.Fatalf("pull did not settle: %v", ctx.Err())
			}
			if (pullErr != nil) != blocked || effects.Load() != 1 {
				t.Fatalf("pull error=%v native effects=%d, blocked=%t", pullErr, effects.Load(), blocked)
			}
			snapshot := ex.downloads.snapshot()
			if err := snapshot.Validate(); err != nil {
				t.Fatalf("invalid terminal snapshot: %v", err)
			}
			if len(snapshot.Entries) != 1 {
				t.Fatalf("terminal snapshot=%+v", snapshot)
			}
			entry := snapshot.Entries[0]
			if entry.ID != before.Entries[0].ID || entry.Engine != "ollama" || entry.Model != "demo" || entry.Phase != phase || entry.BlocksReadiness != blocked {
				t.Fatalf("terminal lost dispatch identity or readiness: %+v", entry)
			}
			if blocked && entry.Reason != modeldownload.ReasonInterrupted {
				t.Fatalf("retired run terminal reason=%q", entry.Reason)
			}
			markers, err := ex.downloads.store.load()
			if err != nil {
				t.Fatalf("read retained journal: %v", err)
			}
			marker, retained := markers[downloadContentKey{"ollama", "demo"}]
			if retained != blocked || retained && marker.ID != entry.ID {
				t.Fatalf("unsafe journal=%+v, blocked=%t", markers, blocked)
			}
		})
	}
	test("same run success clears marker", func(*engineState) {}, modeldownload.PhaseComplete, false)
	test("same process health generation preserves completion", func(st *engineState) {
		st.gen++
	}, modeldownload.PhaseComplete, false)
	test("retired run success retains marker", func(st *engineState) {
		st.runID = "replacement-run"
		st.gen++
	}, modeldownload.PhaseUnknown, true)
	test("stopped run success retains marker", func(st *engineState) {
		st.running, st.runID = false, ""
	}, modeldownload.PhaseUnknown, true)
}

// TestActionPullModelFailureEmitsTerminalError proves a failed LOCAL pull emits
// a terminal engine:pull-progress error frame (in addition to the JSON-RPC
// error), so a UI that already stopped waiting on the synchronous call still
// converges off "pulling" instead of appearing stuck.
func TestActionPullModelFailureEmitsTerminalError(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	m.Actions["pull_model"] = Action{HTTP: &ActionHTTP{Method: "POST", Path: "/api/pull"}}

	var mu sync.Mutex
	var pulls []map[string]any
	reg := NewRegistry()
	reg.engines[m.Engine] = m
	var reportBuf bytes.Buffer
	reporter := NewReporter(NewCodec(&reportBuf))
	ex := NewExecutor(reg, reporter, func(method string, params any) {
		if method != "engine:pull-progress" {
			return
		}
		mp, _ := params.(map[string]any)
		mu.Lock()
		pulls = append(pulls, mp)
		mu.Unlock()
	}, t.TempDir())

	// The engine is never started, so the HTTP pull path fails with "not running".
	var out bytes.Buffer
	mgr := NewManager(NewCodec(&out), ex, nil, "")
	acceptModelMutations(ex)
	id := json.RawMessage("9")
	mgr.runAction(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:action",
		Params: json.RawMessage(`{"engine":"fake","action":"pull_model","params":{"name":"demo:1b"}}`)})

	if !strings.Contains(out.String(), "Fake Engine experienced an error while downloading a model") {
		t.Fatalf("expected a formatted JSON-RPC error for the failed pull, got %s", out.String())
	}

	mu.Lock()
	defer mu.Unlock()
	if len(pulls) != 2 || pulls[0]["stage"] != "queued" {
		t.Fatalf("expected queued and one terminal error frame, got %d: %+v", len(pulls), pulls)
	}
	terminal := pulls[1]
	if terminal["op"] != "pull" || terminal["stage"] != "error" {
		t.Fatalf("expected a terminal error frame, got %+v", terminal)
	}
	if pct, _ := terminal["percent"].(int); pct != -1 {
		t.Fatalf("expected percent -1 on the error frame, got %+v", terminal)
	}
	state, ok := terminal["downloadState"].(*modeldownload.Snapshot)
	if !ok || len(state.Entries) != 1 || state.Entries[0].Phase != "failed" || state.Entries[0].BlocksReadiness {
		t.Fatalf("pre-effect outcome = %+v", terminal)
	}

	snap := reporter.snapshot()
	if len(snap) != 1 {
		t.Fatalf("expected one errors:report entry, got %+v", snap)
	}
	if snap[0].Operation != "pull" || snap[0].ModelName != "demo:1b" || snap[0].Action != "retry" {
		t.Fatalf("unexpected errors:report entry: %+v", snap[0])
	}
	if !strings.Contains(reportBuf.String(), `"errors:report"`) {
		t.Fatalf("expected errors:report on the wire, got %s", reportBuf.String())
	}
}
