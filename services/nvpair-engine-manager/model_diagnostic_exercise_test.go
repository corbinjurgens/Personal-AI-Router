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
	"log/slog"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/modelconfig"
)

type modelDiagnosticExerciseFake struct {
	t           *testing.T
	server      *httptest.Server
	child       modelDiagnosticExerciseChild
	executor    *Executor
	requests    atomic.Int32
	active      atomic.Int32
	onCount     func(http.ResponseWriter, *http.Request, modelDiagnosticChatRequest) bool
	onComplete  func(http.ResponseWriter, *http.Request, modelDiagnosticChatRequest)
	mu          sync.Mutex
	completions []modelDiagnosticChatRequest
}

func newModelDiagnosticExerciseFake(t *testing.T) *modelDiagnosticExerciseFake {
	t.Helper()
	f := &modelDiagnosticExerciseFake{t: t}
	f.child = modelDiagnosticExerciseChild{ModelID: "fixture-model", RunID: "fixture-run", Done: make(chan struct{}), Check: func(ctx context.Context) error { return ctx.Err() }}
	f.server = httptest.NewUnstartedServer(http.HandlerFunc(f.serve))
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen for fake llama: %v", err)
	}
	f.server.Listener = listener
	f.server.Start()
	t.Cleanup(f.server.Close)
	_, port, err := net.SplitHostPort(listener.Addr().String())
	if err != nil {
		t.Fatalf("read fake llama address: %v", err)
	}
	f.child.Port, err = strconv.Atoi(port)
	if err != nil {
		t.Fatalf("read fake llama port: %v", err)
	}
	f.executor = &Executor{client: f.server.Client(), budgetClients: map[time.Duration]*http.Client{modelDiagnosticExerciseLimit: f.server.Client()}}
	return f
}

func modelDiagnosticFakeTokens(body modelDiagnosticChatRequest) int32 {
	return int32(len(body.Messages[0].Content)/4 + 8)
}

func (f *modelDiagnosticExerciseFake) serve(w http.ResponseWriter, r *http.Request) {
	f.requests.Add(1)
	f.active.Add(1)
	defer f.active.Add(-1)
	if r.Method != http.MethodPost || r.Header.Get(engineIdentityProbeHeader) != "1" || r.URL.Query().Get("autoload") != "false" || r.URL.Query().Get("model") != f.child.ModelID {
		f.t.Error("exercise did not use an identity probe and non-loading POST")
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	var body modelDiagnosticChatRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		f.t.Errorf("decode exercise request: %v", err)
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	if body.Model != f.child.ModelID || len(body.Messages) != 1 || body.Messages[0].Role != "user" || body.MaxTokens != 128 || body.Temperature != 0 || body.CachePrompt {
		f.t.Error("exercise request did not preserve the pinned chat parameters")
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	switch r.URL.Path {
	case "/v1/chat/completions/input_tokens":
		if f.onCount != nil && f.onCount(w, r, body) {
			return
		}
		if err := json.NewEncoder(w).Encode(struct {
			Object string `json:"object"`
			Tokens int32  `json:"input_tokens"`
		}{"response.input_tokens", modelDiagnosticFakeTokens(body)}); err != nil {
			f.t.Errorf("write token count: %v", err)
		}
	case "/v1/chat/completions":
		f.mu.Lock()
		f.completions = append(f.completions, body)
		f.mu.Unlock()
		if f.onComplete != nil {
			f.onComplete(w, r, body)
			return
		}
		f.finish(w, body, modelDiagnosticFakeTokens(body), 0)
	default:
		f.t.Errorf("unexpected exercise path %s", r.URL.Path)
		http.NotFound(w, r)
	}
}

func (f *modelDiagnosticExerciseFake) finish(w http.ResponseWriter, body modelDiagnosticChatRequest, prompt, cached int32) {
	f.t.Helper()
	timings := fmt.Sprintf(`"timings":{"prompt_n":%d,"cache_n":%d,"prompt_ms":200,"prompt_per_second":%g,"predicted_n":128,"predicted_ms":1000,"predicted_per_second":127}`, prompt, cached, float64(prompt)*5)
	var response string
	if body.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		response = "data: {\"choices\":[{\"delta\":{\"content\":\"fixture token\"},\"finish_reason\":null}]}\n\n" +
			"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]," + timings + "}\n\ndata: [DONE]\n\n"
	} else {
		response = "{\"choices\":[{\"message\":{\"content\":\"fixture output\"},\"finish_reason\":\"length\"}]," + timings + "}"
	}
	if _, err := io.WriteString(w, response); err != nil {
		f.t.Errorf("write completion: %v", err)
	}
}

func modelDiagnosticWait(t *testing.T, ch <-chan struct{}, description string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", description)
	}
}

func TestModelDiagnosticExerciseTargets(t *testing.T) {
	tests := []struct {
		name           string
		context, slots int32
		low, high      int32
		errorReason    string
	}{
		{name: "minimum supported context", context: 208, slots: 1, low: 32, high: 64},
		{name: "short low target", context: 1024, slots: 2, low: 440, high: 880},
		{name: "low target cap", context: 8192, slots: 8, low: 512, high: 8048},
		{name: "context too small", context: 207, slots: 1, errorReason: "context-too-small"},
		{name: "too many slots", context: 8192, slots: 9, errorReason: "unsupported-slot-count"},
		{name: "no slots", context: 8192, errorReason: "unsupported-slot-count"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			low, high, err := modelDiagnosticTargets(tc.context, tc.slots)
			if tc.errorReason != "" {
				if err == nil || err.Error() != tc.errorReason {
					t.Fatalf("targets error = %v, want %s", err, tc.errorReason)
				}
				return
			}
			if err != nil || low != tc.low || high != tc.high {
				t.Fatalf("targets = %d/%d, %v; want %d/%d", low, high, err, tc.low, tc.high)
			}
		})
	}
}

func TestModelDiagnosticExerciseLowValues(t *testing.T) {
	f := newModelDiagnosticExerciseFake(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	phase, err := f.executor.modelDiagnosticExercise(ctx, f.child, modelDiagnosticExercisePhase{Phase: "low", TargetPromptTokens: 512, Requests: 1})
	if err != nil {
		t.Fatalf("exercise low phase: %v", err)
	}
	if err := phase.Validate(); err != nil {
		t.Fatalf("invalid low phase: %v", err)
	}
	if phase.Coverage != "complete" || len(phase.Requests) != 1 || phase.Overlapped != nil || phase.OverlapWindowSampled != nil || phase.CombinedGeneratedPerSecond != nil {
		t.Fatalf("unexpected low phase: %+v", phase)
	}
	r := phase.Requests[0]
	if r.PromptTokens == nil || *r.PromptTokens != 512 || r.CachedTokens == nil || *r.CachedTokens != 0 || r.GeneratedTokens == nil || *r.GeneratedTokens != 128 || r.PromptMS == nil || *r.PromptMS != 200 || r.PromptPerSecond == nil || *r.PromptPerSecond != 2560 || r.DecodeMS == nil || *r.DecodeMS != 1000 || r.DecodePerSecond == nil || *r.DecodePerSecond != 127 || r.FirstTokenMS != nil || r.EndMS == nil {
		t.Fatalf("decoded low values: %+v", r)
	}
	if len(f.completions) != 1 || f.completions[0].Stream {
		t.Fatal("low phase did not send one non-streaming request")
	}
}

type modelDiagnosticObserveTransport struct {
	base http.RoundTripper
	read chan<- struct{}
}

func (tr modelDiagnosticObserveTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := tr.base.RoundTrip(req)
	if err == nil && req.URL.Path == "/v1/chat/completions" {
		resp.Body = &modelDiagnosticObserveBody{ReadCloser: resp.Body, read: tr.read}
	}
	return resp, err
}

type modelDiagnosticObserveBody struct {
	io.ReadCloser
	read chan<- struct{}
	seen bool
	once sync.Once
}

func (b *modelDiagnosticObserveBody) Read(data []byte) (int, error) {
	if b.seen {
		// The scanner requests its next read after consuming the first frame.
		b.once.Do(func() { b.read <- struct{}{} })
	}
	n, err := b.ReadCloser.Read(data)
	if n > 0 {
		b.seen = true
	}
	return n, err
}

func TestModelDiagnosticExerciseConcurrentHighOverlap(t *testing.T) {
	f := newModelDiagnosticExerciseFake(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	const slots = 3
	read := make(chan struct{}, slots)
	client := *f.server.Client()
	client.Transport = modelDiagnosticObserveTransport{base: http.DefaultTransport, read: read}
	f.executor.budgetClients[modelDiagnosticExerciseLimit] = &client
	allRead := make(chan struct{})
	collectorDone := make(chan struct{})
	go func() {
		defer close(collectorDone)
		for i := 0; i < slots; i++ {
			select {
			case <-read:
			case <-ctx.Done():
				return
			}
		}
		// Reading every first token proves ordering, but Windows can report
		// both that read and all finishes in one monotonic clock tick. Cross
		// an observed clock boundary before ending the shared decode window.
		observed := time.Now()
		tick := time.NewTicker(time.Millisecond)
		defer tick.Stop()
		for !time.Now().After(observed) {
			select {
			case <-tick.C:
			case <-ctx.Done():
				return
			}
		}
		close(allRead)
	}()
	defer func() { cancel(); <-collectorDone }()
	f.onComplete = func(w http.ResponseWriter, r *http.Request, body modelDiagnosticChatRequest) {
		w.Header().Set("Content-Type", "text/event-stream")
		// An initial role-only frame must not count as a generated token.
		if _, err := io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"reasoning_content\":\"a token\"}}]}\n\n"); err != nil {
			t.Errorf("write first token: %v", err)
			return
		}
		w.(http.Flusher).Flush()
		select {
		case <-allRead:
			f.finish(w, body, modelDiagnosticFakeTokens(body), 0)
		case <-r.Context().Done():
		}
	}
	marker := time.Now().Add(-100 * time.Millisecond)
	phase, err := f.executor.modelDiagnosticExercise(ctx, f.child, modelDiagnosticExercisePhase{Phase: "high", TargetPromptTokens: 1024, Requests: slots, StartedAt: marker})
	if err != nil {
		t.Fatalf("exercise high phase: %v", err)
	}
	if err := phase.Validate(); err != nil {
		t.Fatalf("invalid high phase: %v", err)
	}
	if phase.Coverage != "complete" || phase.Overlapped == nil || !*phase.Overlapped || phase.OverlapWindowSampled != nil || phase.CombinedGeneratedPerSecond == nil {
		t.Fatalf("unexpected provisional high coverage: %+v", phase)
	}
	latest, earliest := 0.0, math.Inf(1)
	for _, request := range phase.Requests {
		if request.FirstTokenMS == nil || request.EndMS == nil || *request.FirstTokenMS < 100 || request.GeneratedTokens == nil || *request.GeneratedTokens != 128 || request.DecodePerSecond == nil || *request.DecodePerSecond != 127 {
			t.Fatalf("missing streamed timing: %+v", request)
		}
		latest, earliest = max(latest, *request.FirstTokenMS), min(earliest, *request.EndMS)
	}
	if latest >= earliest {
		t.Fatalf("overlap window is empty: %g through %g", latest, earliest)
	}
	// WallMS is truncated for the wire, so allow its sub-millisecond rounding.
	lowRate := float64(slots*128) * 1000 / float64(phase.WallMS+1)
	highRate := float64(slots*128) * 1000 / float64(phase.WallMS)
	if rate := *phase.CombinedGeneratedPerSecond; rate < lowRate || rate > highRate {
		t.Fatalf("combined throughput %g outside wall interval [%g,%g]", rate, lowRate, highRate)
	}
	seen := map[string]bool{}
	for _, body := range f.completions {
		prompt := body.Messages[0].Content
		prefix := strings.SplitN(prompt, "\n", 2)[0]
		if seen[prefix] || !body.Stream || modelDiagnosticFakeTokens(body) != 1024 {
			t.Fatal("high requests reused a prefix or missed their counted target")
		}
		seen[prefix] = true
	}
}

func TestModelDiagnosticExercisePartialPromptCoverage(t *testing.T) {
	tests := []struct {
		name           string
		prompt, cached int32
	}{
		{name: "cached prompt", prompt: 512, cached: 1},
		{name: "below 95 percent", prompt: 486},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := newModelDiagnosticExerciseFake(t)
			f.onComplete = func(w http.ResponseWriter, _ *http.Request, body modelDiagnosticChatRequest) {
				f.finish(w, body, tc.prompt, tc.cached)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			phase, err := f.executor.modelDiagnosticExercise(ctx, f.child, modelDiagnosticExercisePhase{Phase: "low", TargetPromptTokens: 512, Requests: 1})
			if err != nil || phase.Coverage != "partial" {
				t.Fatalf("coverage = %s, %v; want partial", phase.Coverage, err)
			}
			if phase.Requests[0].PromptTokens == nil || *phase.Requests[0].PromptTokens != tc.prompt || phase.Requests[0].CachedTokens == nil || *phase.Requests[0].CachedTokens != tc.cached {
				t.Fatalf("achieved counts were lost: %+v", phase.Requests[0])
			}
		})
	}
}

func TestModelDiagnosticExerciseNoGenerationOverlap(t *testing.T) {
	f := newModelDiagnosticExerciseFake(t)
	read := make(chan struct{}, 2)
	client := *f.server.Client()
	client.Transport = modelDiagnosticObserveTransport{base: http.DefaultTransport, read: read}
	f.executor.budgetClients[modelDiagnosticExerciseLimit] = &client
	firstEnded := make(chan struct{})
	f.onComplete = func(w http.ResponseWriter, r *http.Request, body modelDiagnosticChatRequest) {
		if strings.HasPrefix(body.Messages[0].Content, "high/0/") {
			// FinishReason is deliberately in the first frame. The next client
			// read establishes its observed end before request 1 can emit a token.
			w.Header().Set("Content-Type", "text/event-stream")
			if _, err := fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"first\"},\"finish_reason\":\"length\"}],\"timings\":{\"prompt_n\":%d,\"cache_n\":0,\"predicted_n\":128}}\n\n", modelDiagnosticFakeTokens(body)); err != nil {
				t.Errorf("write completed first request: %v", err)
				return
			}
			w.(http.Flusher).Flush()
			select {
			case <-read:
				close(firstEnded)
				if _, err := io.WriteString(w, "data: [DONE]\n\n"); err != nil {
					t.Errorf("write first request terminator: %v", err)
				}
			case <-r.Context().Done():
			}
		} else {
			select {
			case <-firstEnded:
				f.finish(w, body, modelDiagnosticFakeTokens(body), 0)
			case <-r.Context().Done():
			}
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	phase, err := f.executor.modelDiagnosticExercise(ctx, f.child, modelDiagnosticExercisePhase{Phase: "high", TargetPromptTokens: 1024, Requests: 2})
	if err != nil {
		t.Fatalf("exercise disjoint streams: %v", err)
	}
	if phase.Coverage != "partial" || phase.Overlapped == nil || *phase.Overlapped {
		t.Fatalf("disjoint generation reported full overlap: %+v", phase)
	}
}

func TestModelDiagnosticExerciseCancellationJoinsRequests(t *testing.T) {
	for _, stage := range []string{"token counting", "prefill", "generation"} {
		t.Run(stage, func(t *testing.T) {
			f := newModelDiagnosticExerciseFake(t)
			entered := make(chan struct{}, 2)
			exited := make(chan struct{}, 2)
			block := func(w http.ResponseWriter, r *http.Request) {
				if stage == "generation" {
					w.Header().Set("Content-Type", "text/event-stream")
					if _, err := io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\n"); err != nil {
						t.Errorf("write generated token: %v", err)
					}
					w.(http.Flusher).Flush()
				}
				entered <- struct{}{}
				<-r.Context().Done()
				exited <- struct{}{}
			}
			if stage == "token counting" {
				f.onCount = func(w http.ResponseWriter, r *http.Request, _ modelDiagnosticChatRequest) bool {
					block(w, r)
					return true
				}
			} else {
				f.onComplete = func(w http.ResponseWriter, r *http.Request, _ modelDiagnosticChatRequest) { block(w, r) }
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			returned := make(chan error, 1)
			go func() {
				_, err := f.executor.modelDiagnosticExercise(ctx, f.child, modelDiagnosticExercisePhase{Phase: "high", TargetPromptTokens: 512, Requests: 2})
				returned <- err
			}()
			inFlight := 2
			if stage == "token counting" {
				inFlight = 1
			}
			for i := 0; i < inFlight; i++ {
				modelDiagnosticWait(t, entered, stage)
			}
			cancel()
			select {
			case err := <-returned:
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("cancel error = %v", err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("cancel did not join exercise requests")
			}
			// Observe remote cancellation too. The exercise joins its client
			// work; the operation still owns native unload/stop confirmation.
			for i := 0; i < inFlight; i++ {
				modelDiagnosticWait(t, exited, "fake request cancellation")
			}
			before := f.requests.Load()
			_, err := f.executor.modelDiagnosticExercise(ctx, f.child, modelDiagnosticExercisePhase{Phase: "low", TargetPromptTokens: 512, Requests: 1})
			if !errors.Is(err, context.Canceled) || f.requests.Load() != before {
				t.Fatalf("cancelled exercise sent another request: %v, before=%d after=%d", err, before, f.requests.Load())
			}
		})
	}
}

func TestModelDiagnosticExerciseChildExit(t *testing.T) {
	f := newModelDiagnosticExerciseFake(t)
	exit := make(chan struct{})
	f.child.Done = exit
	f.onCount = func(w http.ResponseWriter, _ *http.Request, _ modelDiagnosticChatRequest) bool {
		close(exit)
		conn, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Errorf("hijack fake child connection: %v", err)
			return true
		}
		if err := conn.Close(); err != nil {
			t.Errorf("close fake child connection: %v", err)
		}
		return true
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := f.executor.modelDiagnosticExercise(ctx, f.child, modelDiagnosticExercisePhase{Phase: "low", TargetPromptTokens: 512, Requests: 1})
	if !errors.Is(err, errModelDiagnosticEngineExited) {
		t.Fatalf("child exit error = %v", err)
	}
	if len(f.completions) != 0 {
		t.Fatal("generation dispatched after child exit")
	}
}

func TestModelDiagnosticExerciseIdentityChange(t *testing.T) {
	f := newModelDiagnosticExerciseFake(t)
	var changed atomic.Bool
	f.child.Check = func(context.Context) error {
		if changed.Load() {
			return errors.New("private identity detail")
		}
		return nil
	}
	f.onCount = func(w http.ResponseWriter, _ *http.Request, _ modelDiagnosticChatRequest) bool {
		changed.Store(true)
		if _, err := io.WriteString(w, `{"input_tokens":32}`); err != nil {
			t.Errorf("write replacement token count: %v", err)
		}
		return true
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := f.executor.modelDiagnosticExercise(ctx, f.child, modelDiagnosticExercisePhase{Phase: "low", TargetPromptTokens: 512, Requests: 1})
	if !errors.Is(err, errModelDiagnosticIdentity) || len(f.completions) != 0 {
		t.Fatalf("identity replacement = %v, completions=%d", err, len(f.completions))
	}
}

func TestModelDiagnosticExerciseRefusesRedirectAndRedactsBodies(t *testing.T) {
	tests := []struct {
		name   string
		status int
	}{
		{name: "redirect", status: http.StatusTemporaryRedirect},
		{name: "server error", status: http.StatusInternalServerError},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := newModelDiagnosticExerciseFake(t)
			var logs bytes.Buffer
			previous := slog.Default()
			slog.SetDefault(slog.New(slog.NewTextHandler(&logs, nil)))
			t.Cleanup(func() { slog.SetDefault(previous) })
			f.onCount = func(w http.ResponseWriter, _ *http.Request, body modelDiagnosticChatRequest) bool {
				w.Header().Set("Location", "/models/load")
				w.WriteHeader(tc.status)
				if _, err := io.WriteString(w, body.Messages[0].Content+" private-response-body"); err != nil {
					t.Errorf("write private body: %v", err)
				}
				return true
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_, err := f.executor.modelDiagnosticExercise(ctx, f.child, modelDiagnosticExercisePhase{Phase: "low", TargetPromptTokens: 512, Requests: 1})
			if err == nil || f.requests.Load() != 1 || len(f.completions) != 0 {
				t.Fatalf("unsafe response handling: err=%v requests=%d", err, f.requests.Load())
			}
			if logs.Len() != 0 || strings.Contains(err.Error(), f.child.RunID) || strings.Contains(err.Error(), "private-response-body") {
				t.Fatalf("exercise exposed private text: %v", err)
			}
		})
	}
}

func TestModelDiagnosticExerciseTimingSemantics(t *testing.T) {
	tests := []struct {
		name, raw  string
		wantRate   *float64
		wantTokens *int32
	}{
		{name: "llama decode steps exclude first token", raw: `{"predicted_n":128,"predicted_ms":1000,"predicted_per_second":127}`, wantRate: modelDiagnosticTestPtr(127.0), wantTokens: modelDiagnosticTestPtr(int32(128))},
		{name: "output count rate is contradictory", raw: `{"predicted_n":128,"predicted_ms":1000,"predicted_per_second":128}`, wantTokens: modelDiagnosticTestPtr(int32(128))},
		{name: "one generated token has no decode work", raw: `{"predicted_n":1,"predicted_ms":0,"predicted_per_second":0}`, wantTokens: modelDiagnosticTestPtr(int32(1))},
		{name: "missing remains unknown", raw: `{}`},
		{name: "negative count remains unknown", raw: `{"predicted_n":-1,"predicted_ms":10,"predicted_per_second":4}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var timing modelDiagnosticTimings
			if err := json.Unmarshal([]byte(tc.raw), &timing); err != nil {
				t.Fatalf("decode timing fixture: %v", err)
			}
			var request modelconfig.MeasuredRequest
			modelDiagnosticApplyTimings(&request, &timing)
			if (request.DecodePerSecond == nil) != (tc.wantRate == nil) || (request.GeneratedTokens == nil) != (tc.wantTokens == nil) {
				t.Fatalf("unexpected optional timing: %+v", request)
			}
			if tc.wantRate != nil && *request.DecodePerSecond != *tc.wantRate {
				t.Fatalf("reported rate = %g, want %g", *request.DecodePerSecond, *tc.wantRate)
			}
			if tc.wantTokens != nil && *request.GeneratedTokens != *tc.wantTokens {
				t.Fatalf("generated = %d, want %d", *request.GeneratedTokens, *tc.wantTokens)
			}
		})
	}
}

func TestModelDiagnosticExerciseRejectsIncompleteOrOversizedStreams(t *testing.T) {
	tests := []struct {
		name, stream string
	}{
		{name: "missing finish", stream: "data: {\"choices\":[{\"delta\":{\"content\":\"token\"}}]}\n\ndata: [DONE]\n\n"},
		{name: "missing done", stream: "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"},
		{name: "native error", stream: "data: {\"error\":{\"message\":\"private response\"}}\n\n"},
		{name: "oversized event", stream: "data: " + strings.Repeat("x", modelDiagnosticEventBytes) + "\n\n"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var request modelconfig.MeasuredRequest
			err := modelDiagnosticReadStream(strings.NewReader(tc.stream), &request, time.Now())
			if err == nil || strings.Contains(err.Error(), "private response") {
				t.Fatalf("unsafe stream result: %v", err)
			}
		})
	}
}

func TestModelDiagnosticExerciseBoundsJSONResponses(t *testing.T) {
	var result struct {
		Tokens int32 `json:"input_tokens"`
	}
	input := `{"input_tokens":1,"padding":"` + strings.Repeat("x", modelDiagnosticResponseBytes) + `"}`
	if err := modelDiagnosticReadJSON(strings.NewReader(input), &result); err == nil {
		t.Fatal("oversized JSON response was accepted")
	}
}

func modelDiagnosticTestPtr[T int32 | float64](value T) *T { return &value }
