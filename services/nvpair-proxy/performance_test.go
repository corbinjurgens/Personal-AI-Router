// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
	"nvpair-shared/performancewire"
)

const performanceStrataFinal = `{"choices":[{"index":0,"finish_reason":"stop"}],"usage":{"completion_tokens":20},"timings":{"predicted_n":20,"predicted_ms":400,"predicted_per_second":55}}`
const performanceLlamaFinal = `{"choices":[{"index":0,"finish_reason":"stop"}],"timings":{"predicted_n":11,"predicted_ms":250,"predicted_per_second":40}}`

func performanceIdentity() nodepolicy.EngineIdentity {
	return nodepolicy.EngineIdentity{EngineRunID: "runtime-1", SettingsFingerprint: strings.Repeat("a", 64), LaunchDigest: strings.Repeat("b", 64), ModelDigests: map[string]string{"m": strings.Repeat("c", 64), "fixture-model": strings.Repeat("c", 64)}}
}

func performanceIdentify(p *Proxy, host string) {
	f := p.soleFacade()
	p.noteSelf(host, f.profile.Name, []string{"m", "fixture-model"})
	p.admission.setModelIdentity(map[string]nodepolicy.EngineIdentity{f.profile.Name: performanceIdentity()})
}

func performanceProxy(t *testing.T, engine string) *Proxy {
	t.Helper()
	profile, ok := profileFor(engine)
	if engine == "strata" {
		var err error
		profile, err = dynamicProfile(strataDescriptor(t))
		if err != nil {
			t.Fatal(err)
		}
		ok = true
	}
	if !ok {
		t.Fatalf("missing profile %s", engine)
	}
	disc := NewDiscovery()
	disc.AddManual(selfNode("destination", profile.FacadePort, "m"))
	p := testProxy(profile, disc, profile.FacadePort)
	performanceIdentify(p, "destination")
	t.Cleanup(p.closeIdleTransports)
	return p
}

func performanceSamples(p *Proxy) []performancewire.Sample {
	q := &p.performance
	q.mu.Lock()
	defer q.mu.Unlock()
	samples := make([]performancewire.Sample, q.size)
	for i := range samples {
		samples[i] = q.items[(q.head+i)%len(q.items)]
	}
	return samples
}

func performanceOne(t *testing.T, p *Proxy) performancewire.Sample {
	t.Helper()
	s := performanceSamples(p)
	if len(s) != 1 {
		t.Fatalf("samples=%d, want one; drops=%v", len(s), p.performance.drops)
	}
	if err := s[0].Validate(); err != nil {
		t.Fatal(err)
	}
	return s[0]
}

func performanceUpstream(t *testing.T, payload string) *httptest.Server {
	t.Helper()
	u := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, payload); err != nil {
			t.Error(err)
		}
	}))
	t.Cleanup(u.Close)
	return u
}

func TestPerformanceDestinationSelf(t *testing.T) {
	test := func(name, engine, final, body string, rate float64) {
		t.Run(name, func(t *testing.T) {
			p := performanceProxy(t, engine)
			u := performanceUpstream(t, "data: "+final+"\n\ndata: [DONE]\n\n")
			setBackend(t, p.soleFacade(), u.URL, true)
			w := httptest.NewRecorder()
			p.soleFacade().handleHTTP(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(body)))
			s := performanceOne(t, p)
			if w.Code != http.StatusOK || s.Key.Host != "destination" || s.Key.Engine != p.soleFacade().profile.Name || s.Key.Model != p.soleFacade().profile.normalizeModel("m") {
				t.Fatalf("status/key=%d %+v", w.Code, s.Key)
			}
			if s.Outcome != performancewire.OutcomeCompleted || s.ProtocolComplete == nil || !*s.ProtocolComplete || s.EngineDecodeTokensPerSecond == nil || *s.EngineDecodeTokensPerSecond != rate || s.DecodeNs != nil || s.FirstTokenMs != nil {
				t.Fatalf("measurement=%+v", s.Measurement)
			}
			if s.HeaderMs == nil || s.FirstBodyMs == nil || s.TotalMs == nil || s.AdmissionWaitMs == nil {
				t.Fatalf("missing local milestone: %+v", s.Measurement)
			}
			if *s.HeaderMs > *s.FirstBodyMs || *s.FirstBodyMs > *s.TotalMs {
				t.Fatalf("unordered local milestones: %+v", s.Measurement)
			}
		})
	}
	test("llama finish without usage", "llamacpp", performanceLlamaFinal, `{"model":"m","stream":true}`, 40)
	test("llama empty choices usage", "llamacpp", `{"choices":[],"usage":{"completion_tokens":11},"timings":{"predicted_n":11,"predicted_ms":250,"predicted_per_second":40}}`, `{"model":"m","stream":true,"stream_options":{"include_usage":true}}`, 40)
	test("Strata native count differs", "strata", performanceStrataFinal, `{"model":"m","stream":true}`, 55)
}

func TestPerformancePeerOriginNeverCounts(t *testing.T) {
	for _, transport := range []string{"direct", "reverse"} {
		t.Run(transport, func(t *testing.T) {
			fx := newProxyReverseFixture(t, "llamacpp", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				if _, err := io.WriteString(w, "data: "+performanceLlamaFinal+"\n\ndata: [DONE]\n\n"); err != nil {
					t.Error(err)
				}
			}))
			performanceIdentify(fx.p, "origin")
			performanceIdentify(fx.peer, "host-peer")
			fx.directHealthy.Store(transport == "direct")
			w := httptest.NewRecorder()
			fx.p.soleFacade().handleHTTP(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"fixture-model","stream":true}`)))
			if w.Code != http.StatusOK {
				t.Fatalf("status=%d", w.Code)
			}
			if s := performanceSamples(fx.p); len(s) != 0 {
				t.Fatalf("origin double-counted: %+v", s)
			}
			s := performanceOne(t, fx.peer)
			if s.Key.Host != "host-peer" || s.Key.Model != "fixture-model" || s.EngineDecodeTokensPerSecond == nil || *s.EngineDecodeTokensPerSecond != 40 {
				t.Fatalf("destination=%+v", s)
			}
			if transport == "reverse" && fx.reverseCalls.Load() == 0 {
				t.Fatal("reverse path was not exercised")
			}
		})
	}
}

func TestPerformanceStrictDestinations(t *testing.T) {
	for _, path := range []string{"self", "ingress"} {
		t.Run(path, func(t *testing.T) {
			p := testProxy(ollamaOnlyProfile(t), NewDiscovery(), 0)
			configureStrict(t, p.admission, filepath.Join(t.TempDir(), "capacity.json"))
			p.noteSelf("destination", "ollama", []string{"m"})
			u := performanceUpstream(t, "data: {\"choices\":[{\"index\":0,\"finish_reason\":\"tool_calls\"}],\"usage\":{\"completion_tokens\":7}}\n\ndata: [DONE]\n\n")
			setBackend(t, p.soleFacade(), u.URL, true)
			w := httptest.NewRecorder()
			serveStrictHTTP(p, path, w, strictHTTPRequestWith(t, `{"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":16,"stream":true}`))
			s := performanceOne(t, p)
			if s.OutputTokens == nil || *s.OutputTokens != 7 || s.Outcome != performancewire.OutcomeCompleted {
				t.Fatalf("sample=%+v", s)
			}
			if p.admission.currentActive() != 1 {
				t.Fatal("telemetry released strict uncertainty")
			}
		})
	}
}

func TestPerformanceMissingIdentitySuppresses(t *testing.T) {
	for _, missing := range []string{"engine", "model", "run", "settings", "launch", "host"} {
		t.Run(missing, func(t *testing.T) {
			p := performanceProxy(t, "llamacpp")
			id := performanceIdentity()
			switch missing {
			case "model":
				id.ModelDigests = map[string]string{}
			case "run":
				id.EngineRunID = ""
			case "settings":
				id.SettingsFingerprint = ""
			case "launch":
				id.LaunchDigest = ""
			case "host":
				p.selfNodeID = ""
			}
			ids := map[string]nodepolicy.EngineIdentity{p.soleFacade().profile.Name: id}
			if missing == "engine" {
				ids = nil
			}
			p.admission.setModelIdentity(ids)
			u := performanceUpstream(t, "data: "+performanceLlamaFinal+"\n\ndata: [DONE]\n\n")
			setBackend(t, p.soleFacade(), u.URL, true)
			w := httptest.NewRecorder()
			p.soleFacade().serveIngress(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)), "peer")
			if w.Code != http.StatusOK || len(performanceSamples(p)) != 0 || p.performance.drops[1] != 1 {
				t.Fatalf("status=%d samples=%d drops=%v", w.Code, len(performanceSamples(p)), p.performance.drops)
			}
		})
	}
}

func TestPerformanceFingerprint(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	f := p.soleFacade()
	base := []byte(`{"model":"m","messages":[],"stream":true,"temperature":1}`)
	key, _, ok := f.performanceFingerprint("m", routeChatCompletions, base)
	if !ok {
		t.Fatal("valid projection refused")
	}
	same, _, ok := f.performanceFingerprint("m", routeChatCompletions, []byte(`{"temperature":1e0,"stream":true,"messages":[{"role":"user","content":"different"}],"model":"alias"}`))
	if !ok || same != key {
		t.Fatalf("equivalent effective options changed key: %+v %+v", key, same)
	}
	for _, change := range []string{"options", "runtime", "settings", "launch", "model", "api", "schema"} {
		t.Run(change, func(t *testing.T) {
			p := performanceProxy(t, "llamacpp")
			f := p.soleFacade()
			body := base
			api := routeChatCompletions
			id := performanceIdentity()
			switch change {
			case "options":
				body = []byte(`{"model":"m","stream":true,"temperature":0}`)
			case "runtime":
				id.EngineRunID = "runtime-2"
			case "settings":
				id.SettingsFingerprint = strings.Repeat("d", 64)
			case "launch":
				id.LaunchDigest = strings.Repeat("d", 64)
			case "model":
				id.ModelDigests["m"] = strings.Repeat("d", 64)
			case "api":
				api = "/v1/completions"
			case "schema":
				f.profile.Family = "strata"
			}
			p.admission.setModelIdentity(map[string]nodepolicy.EngineIdentity{f.profile.Name: id})
			changed, _, ok := f.performanceFingerprint("m", api, body)
			if !ok || changed.Fingerprint == key.Fingerprint {
				t.Fatalf("%s did not separate profile", change)
			}
		})
	}
	if _, _, ok := f.performanceFingerprint("m", routeChatCompletions, []byte(`{"model":"m","mystery_override":true}`)); ok {
		t.Fatal("unknown override qualified")
	}
}

func TestPerformanceOptionNumbers(t *testing.T) {
	for _, pair := range [][2]string{{"1", "1e0"}, {"-0.25", "-25e-2"}, {"0", "-0"}, {"9007199254740993", "9007199254740993.0"}, {"1e-20", "10e-21"}} {
		t.Run(pair[0], func(t *testing.T) {
			a := performanceCanonicalNumber(json.Number(pair[0]))
			b := performanceCanonicalNumber(json.Number(pair[1]))
			if a != b {
				t.Fatalf("equivalent options differ: %v / %v", a, b)
			}
		})
	}
	// Outside the lexical bound, retain the exact JSON number without expansion.
	if performanceCanonicalNumber(json.Number("0e999999999")) != json.Number("0e999999999") {
		t.Fatal("unbounded exponent expanded")
	}
}

func TestPerformanceStaleIdentityDrops(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	a := p.soleFacade().newPerformanceAttempt(context.Background(), "m", routeChatCompletions, []byte(`{"model":"m","stream":true}`), 0)
	if a == nil {
		t.Fatal("no attempt")
	}
	a.start = time.Now()
	id := performanceIdentity()
	id.EngineRunID = "runtime-2"
	p.admission.setModelIdentity(map[string]nodepolicy.EngineIdentity{"llamacpp": id})
	a.finish(&statusCapture{}, true)
	if len(performanceSamples(p)) != 0 || p.performance.drops[2] != 1 {
		t.Fatal("stale identity relabeled")
	}
}

func TestPerformanceRetryAttemptsOnce(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	var hits atomic.Int32
	u := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, "data: "+performanceLlamaFinal+"\n\ndata: [DONE]\n\n"); err != nil {
			t.Error(err)
		}
	}))
	defer u.Close()
	setBackend(t, p.soleFacade(), u.URL, true)
	w := httptest.NewRecorder()
	p.soleFacade().handleHTTP(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("retry status=%d", w.Code)
	}
	s := performanceSamples(p)
	if hits.Load() != 2 || len(s) != 2 || s[0].Seq != 1 || s[1].Seq != 2 || s[0].Outcome != performancewire.OutcomeFailed || s[1].Outcome != performancewire.OutcomeCompleted {
		t.Fatalf("retry samples=%+v hits=%d", s, hits.Load())
	}
}

func TestPerformanceFailedDialIsAttempt(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := l.Addr().String()
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	setBackend(t, p.soleFacade(), "http://"+address, true)
	w := httptest.NewRecorder()
	p.soleFacade().serveIngress(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)), "peer")
	s := performanceOne(t, p)
	if s.Outcome != performancewire.OutcomeFailed || s.HeaderMs != nil || s.FirstBodyMs != nil || s.ProtocolComplete != nil {
		t.Fatalf("failed dial=%+v", s)
	}
}

func TestPerformanceRefusalHasNoAttempt(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	pol := nodepolicy.Default()
	pol.Availability = nodepolicy.Paused
	p.admission.setPolicy(pol)
	w := httptest.NewRecorder()
	p.soleFacade().serveIngress(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)), "peer")
	if w.Code != http.StatusServiceUnavailable || len(performanceSamples(p)) != 0 {
		t.Fatalf("refusal=%d samples=%d", w.Code, len(performanceSamples(p)))
	}
}

func TestPerformanceNonstreamUnobserved(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	const body = `{"choices":[{"finish_reason":"stop"}],"timings":{"predicted_n":11,"predicted_ms":250,"predicted_per_second":40}}`
	u := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if _, err := io.WriteString(w, body); err != nil {
			t.Error(err)
		}
	}))
	defer u.Close()
	setBackend(t, p.soleFacade(), u.URL, true)
	w := httptest.NewRecorder()
	p.soleFacade().serveIngress(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":false}`)), "peer")
	s := performanceOne(t, p)
	if w.Body.String() != body || s.Outcome != performancewire.OutcomeCompleted || s.ProtocolComplete != nil || s.EngineDecodeTokensPerSecond != nil || s.OutputTokens != nil {
		t.Fatalf("nonstream observed/rewritten: %+v", s)
	}
}

type performanceBlockedWriter struct {
	entered, release chan struct{}
	once             sync.Once
}

func (w *performanceBlockedWriter) Read([]byte) (int, error) { return 0, io.EOF }
func (w *performanceBlockedWriter) Write(b []byte) (int, error) {
	w.once.Do(func() { close(w.entered) })
	<-w.release
	return len(b), nil
}

func TestPerformanceSenderNeverBlocksInference(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	// Existing admission notifications share the codec's synchronous write gate.
	// Isolate the telemetry producer from that independent notification path.
	p.admission.notify = func(string, any) {}
	writer := &performanceBlockedWriter{entered: make(chan struct{}), release: make(chan struct{})}
	p.codec = NewCodec(writer)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() { defer close(done); p.sendPerformance(ctx) }()
	defer func() {
		cancel()
		close(writer.release)
		waitChan(t, done, "sender returns after write completes")
	}()
	u := performanceUpstream(t, "data: "+performanceLlamaFinal+"\n\ndata: [DONE]\n\n")
	setBackend(t, p.soleFacade(), u.URL, true)
	request := func() {
		p.soleFacade().serveIngress(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)), "peer")
	}
	firstReturned := make(chan struct{})
	go func() { defer close(firstReturned); request() }()
	waitChan(t, writer.entered, "sender blocked in codec")
	waitChan(t, firstReturned, "first inference returns")
	returned := make(chan struct{})
	go func() {
		defer close(returned)
		for i := 0; i < performanceQueueCapacity+3; i++ {
			request()
		}
	}()
	waitChan(t, returned, "inference and full FIFO return")
	p.performance.mu.Lock()
	size, drops := p.performance.size, p.performance.drops[0]
	p.performance.mu.Unlock()
	if size != performanceQueueCapacity || drops != 3 {
		t.Fatalf("size=%d drops=%d", size, drops)
	}
	cancel()
	shutdown := make(chan struct{})
	go func() { p.shutdown(context.Background()); close(shutdown) }()
	waitChan(t, shutdown, "proxy shutdown while telemetry write waits")
}

type performanceGateContext struct {
	context.Context
	waiting chan struct{}
	checks  atomic.Int32
}

func (c *performanceGateContext) Err() error {
	err := c.Context.Err()
	if c.checks.Add(1) == 2 {
		close(c.waiting)
	}
	return err
}

func TestPerformanceSenderCancelsAtWriteGate(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	writer := &performanceBlockedWriter{entered: make(chan struct{}), release: make(chan struct{})}
	p.codec = NewCodec(writer)
	holder := make(chan struct{})
	go func() {
		defer close(holder)
		if err := p.codec.Notify("other", nil); err != nil {
			t.Error(err)
		}
	}()
	waitChan(t, writer.entered, "other notification holds write gate")
	parent, cancel := context.WithCancel(context.Background())
	ctx := &performanceGateContext{Context: parent, waiting: make(chan struct{})}
	done := make(chan struct{})
	defer func() {
		cancel()
		close(writer.release)
		waitChan(t, holder, "other notification released")
		waitChan(t, done, "sender cleaned up")
	}()
	// A queued sample avoids the idle select. Wait for the second Err check
	// to capture nil, so cancellation cannot pass via an early loop return.
	p.performance.items[0] = performancewire.Sample{}
	p.performance.size = 1
	go func() { defer close(done); p.sendPerformance(ctx) }()
	select {
	case <-ctx.waiting:
	case <-time.After(time.Second):
		t.Fatal("sender did not reach notification dispatch")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("sender did not exit while another notification held the write gate")
	}
}

func TestPerformanceUnobservedCopyOutcome(t *testing.T) {
	for _, scenario := range []string{"headers then stall", "body read error", "normal body"} {
		t.Run(scenario, func(t *testing.T) {
			setForTest(t, &firstBodyTimeout, 20*time.Millisecond)
			p := performanceProxy(t, "llamacpp")
			var hits atomic.Int32
			u := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				w.Header().Set("Content-Type", "application/json")
				if scenario == "body read error" {
					w.Header().Set("Content-Length", "100")
				}
				w.WriteHeader(http.StatusOK)
				if scenario == "headers then stall" {
					w.(http.Flusher).Flush()
					<-r.Context().Done()
					return
				}
				if _, err := io.WriteString(w, `{}`); err != nil {
					t.Error(err)
				}
			}))
			defer u.Close()
			setBackend(t, p.soleFacade(), u.URL, true)
			w := httptest.NewRecorder()
			p.soleFacade().handleHTTP(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":false}`)))
			samples := performanceSamples(p)
			wantAttempts, wantOutcome := 1, performancewire.OutcomeFailed
			if scenario == "headers then stall" {
				wantAttempts = maxDispatchAttempts
				if w.Code != http.StatusGatewayTimeout {
					t.Fatalf("status=%d, want 504", w.Code)
				}
			}
			if scenario == "normal body" {
				wantOutcome = performancewire.OutcomeCompleted
				if w.Code != http.StatusOK || w.Body.String() != `{}` {
					t.Fatalf("status/body=%d %q", w.Code, w.Body.String())
				}
			}
			if len(samples) != wantAttempts || int(hits.Load()) != wantAttempts {
				t.Fatalf("samples=%d dispatches=%d, want %d", len(samples), hits.Load(), wantAttempts)
			}
			for i, sample := range samples {
				if sample.Outcome != wantOutcome || sample.ProtocolComplete != nil {
					t.Errorf("attempt %d measurement=%+v, want outcome %s and unknown protocol", i, sample.Measurement, wantOutcome)
				}
			}
		})
	}
}

func TestPerformanceSenderStopsOutsideWrite(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); p.sendPerformance(ctx) }()
	cancel()
	waitChan(t, done, "idle sender stops")
}

func TestPerformanceFIFOCompletionOrderAndIncarnation(t *testing.T) {
	w := &recordingWriter{}
	p := performanceProxy(t, "llamacpp")
	p.codec = NewCodec(w)
	f := p.soleFacade()
	body := []byte(`{"model":"m","stream":true}`)
	first := f.newPerformanceAttempt(context.Background(), "m", routeChatCompletions, body, time.Millisecond)
	second := f.newPerformanceAttempt(context.Background(), "m", routeChatCompletions, body, 2*time.Millisecond)
	first.start = time.Now()
	second.start = time.Now()
	second.finish(&statusCapture{}, true)
	first.finish(&statusCapture{}, true)
	s := performanceSamples(p)
	if len(s) != 2 || s[0].Seq != 1 || s[1].Seq != 2 || *s[0].AdmissionWaitMs != 2 || *s[1].AdmissionWaitMs != 1 {
		t.Fatalf("enqueue order=%+v", s)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); p.sendPerformance(ctx) }()
	deadline := time.After(2 * time.Second)
	for len(frames(t, w)) < 2 {
		select {
		case <-deadline:
			t.Fatal("sender did not drain")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	cancel()
	waitChan(t, done, "sender drained and stopped")
	for i, frame := range frames(t, w) {
		var sample performancewire.Sample
		if err := json.Unmarshal(frame.Params, &sample); err != nil {
			t.Fatal(err)
		}
		if frame.Method != "performance:sample" || sample.Seq != uint64(i+1) || sample.Producer != p.runID {
			t.Fatalf("sent=%+v", sample)
		}
	}
	p2 := performanceProxy(t, "llamacpp")
	if p2.runID == p.runID {
		t.Fatal("process incarnation reused")
	}
}

func TestPerformanceDeliveryFailureExcludesRate(t *testing.T) {
	for _, cause := range []string{"write failure", "panic", "client cancellation"} {
		t.Run(cause, func(t *testing.T) {
			p := performanceProxy(t, "strata")
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			a := p.soleFacade().newPerformanceAttempt(ctx, "m", routeChatCompletions, []byte(`{"model":"m","stream":true}`), 0)
			a.start = time.Now()
			r, err := newStreamFrameReader(streamFrameConfig{Protocol: streamProtocolSSE, Upstream: io.NopCloser(strings.NewReader("data: " + performanceStrataFinal + "\n\ndata: [DONE]\n\n")), Extract: func(b []byte) performanceExtractMetadata { return extractPerformanceObserved(b, a.qualification) }})
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			if _, err := io.Copy(io.Discard, r); err != nil {
				t.Fatal(err)
			}
			a.reader = r
			sc := &statusCapture{}
			returned := true
			switch cause {
			case "write failure":
				sc.wroteErr = errors.New("client write")
			case "panic":
				returned = false
			case "client cancellation":
				cancel()
			}
			a.finish(sc, returned)
			s := performanceOne(t, p)
			if s.Outcome == performancewire.OutcomeCompleted || s.ProtocolComplete == nil || *s.ProtocolComplete {
				t.Fatalf("delivery failure qualified: %+v", s)
			}
		})
	}
}

func TestPerformanceOllamaNativeDestination(t *testing.T) {
	p := performanceProxy(t, "ollama")
	u := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		if _, err := io.WriteString(w, "{\"done\":true,\"eval_count\":3,\"eval_duration\":1000000000}\n"); err != nil {
			t.Error(err)
		}
	}))
	defer u.Close()
	setBackend(t, p.soleFacade(), u.URL, true)
	w := httptest.NewRecorder()
	p.soleFacade().serveIngress(w, httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"m","stream":true,"prompt":"x"}`)), "peer")
	s := performanceOne(t, p)
	if s.Key.Model != "m:latest" || s.Outcome != performancewire.OutcomeCompleted || s.DecodeNs == nil || *s.DecodeNs != 1000000000 || s.OutputTokens == nil || *s.OutputTokens != 3 || s.EngineDecodeTokensPerSecond != nil {
		t.Fatalf("native measurement=%+v", s)
	}
}

func TestPerformanceStreamNumericSeam(t *testing.T) {
	test := func(name, payload string, check func(*testing.T, streamFrameResult)) {
		t.Run(name, func(t *testing.T) {
			read := func(observe bool) ([]byte, streamFrameResult) {
				config := streamFrameConfig{Protocol: streamProtocolSSE, Upstream: io.NopCloser(strings.NewReader(payload))}
				if observe {
					config.Extract = func(b []byte) performanceExtractMetadata {
						return extractPerformanceObserved(b, performanceExtractQualification{EngineFamily: "strata", APIPath: routeChatCompletions, Streaming: true, SchemaID: performanceExtractStrataSchema})
					}
				}
				r, err := newStreamFrameReader(config)
				if err != nil {
					t.Fatal(err)
				}
				defer r.Close()
				b, err := io.ReadAll(r)
				if err != nil {
					t.Fatal(err)
				}
				return b, r.Result()
			}
			plain, baseline := read(false)
			observed, result := read(true)
			if !bytes.Equal(plain, observed) || baseline.Outcome != result.Outcome || baseline.Reason != result.Reason {
				t.Fatal("numeric callback changed SC bytes/result")
			}
			check(t, result)
		})
	}
	completed := func(t *testing.T, r streamFrameResult) {
		t.Helper()
		if r.Outcome != streamOutcomeCompleted || r.Numeric.EngineDecodeTokensPerSecond == nil || *r.Numeric.EngineDecodeTokensPerSecond != 55 {
			t.Fatalf("result=%+v", r)
		}
	}
	test("native rate", "data: "+performanceStrataFinal+"\n\ndata: [DONE]\n\n", completed)
	test("tool finish", "data: "+strings.Replace(performanceStrataFinal, "stop", "tool_calls", 1)+"\n\ndata: [DONE]\n\n", completed)
	test("reasoning finish", "data: "+strings.Replace(performanceStrataFinal, "stop", "length", 1)+"\n\ndata: [DONE]\n\n", completed)
	test("usage timings split", "data: {\"choices\":[],\"usage\":{\"completion_tokens\":20}}\n\ndata: "+performanceStrataFinal+"\n\ndata: [DONE]\n\n", completed)
	test("repeated never adds", "data: "+performanceStrataFinal+"\n\ndata: "+performanceStrataFinal+"\n\ndata: [DONE]\n\n", completed)
	test("conflicting counters", "data: "+performanceStrataFinal+"\n\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":21}}\n\ndata: "+performanceStrataFinal+"\n\ndata: [DONE]\n\n", func(t *testing.T, r streamFrameResult) {
		if r.Numeric.OutputTokens != nil || r.Numeric.EngineDecodeTokensPerSecond != nil {
			t.Fatalf("conflict restored: %+v", r.Numeric)
		}
	})
	test("sticky MCP", "data: {\"choices\":[],\"strata_mcp\":{}}\n\ndata: "+performanceStrataFinal+"\n\ndata: [DONE]\n\n", func(t *testing.T, r streamFrameResult) {
		if !r.Numeric.PhaseDisqualified || r.Numeric.EngineDecodeTokensPerSecond != nil {
			t.Fatal("MCP requalified")
		}
	})
	test("invalid later timing", "data: "+performanceStrataFinal+"\n\ndata: {\"choices\":[],\"timings\":{\"predicted_ms\":-1}}\n\ndata: [DONE]\n\n", func(t *testing.T, r streamFrameResult) {
		if r.Numeric.EngineDecodeTokensPerSecond != nil {
			t.Fatal("invalid later timing retained rate")
		}
	})
	test("invalid later cache", "data: {\"choices\":[],\"timings\":{\"cache_n\":4}}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":-1}}}\n\ndata: [DONE]\n\n", func(t *testing.T, r streamFrameResult) {
		if r.Numeric.CachedPromptTokens != nil {
			t.Fatal("invalid later cache retained count")
		}
	})
	test("usage only", "data: {\"choices\":[],\"usage\":{\"completion_tokens\":20}}\n\ndata: [DONE]\n\n", func(t *testing.T, r streamFrameResult) {
		if r.Numeric.OutputTokens == nil || r.Numeric.EngineDecodeTokensPerSecond != nil {
			t.Fatal("usage invented speed")
		}
	})
	test("interrupted", "data: "+performanceStrataFinal+"\n\ndata: {", func(t *testing.T, r streamFrameResult) {
		if r.Outcome != streamOutcomeInterrupted {
			t.Fatalf("outcome=%v", r.Outcome)
		}
	})
	test("error", "data: {\"error\":{\"message\":\"upstream\"}}\n\n", func(t *testing.T, r streamFrameResult) {
		if r.Outcome != streamOutcomeUpstreamError {
			t.Fatalf("outcome=%v", r.Outcome)
		}
	})
}

func TestPerformanceResultOwnsNumericCopy(t *testing.T) {
	r, err := newStreamFrameReader(streamFrameConfig{Protocol: streamProtocolNDJSON, Upstream: io.NopCloser(strings.NewReader("{\"done\":true,\"eval_count\":3,\"eval_duration\":1000000000}\n")), Extract: func(b []byte) performanceExtractMetadata {
		return extractPerformancePayload(b, performanceExtractQualification{EngineFamily: "ollama", APIPath: "/api/generate", Streaming: true})
	}})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if _, err := io.Copy(io.Discard, r); err != nil {
		t.Fatal(err)
	}
	a := r.Result()
	b := r.Result()
	if !reflect.DeepEqual(a, b) || a.Numeric.OutputTokens == nil {
		t.Fatal("missing stable numeric copy")
	}
	*a.Numeric.OutputTokens = 99
	if *r.Result().Numeric.OutputTokens != 3 {
		t.Fatal("result aliases reader storage")
	}
}

func TestPerformanceTimingReadFieldsInvalidateStickyEvidence(t *testing.T) {
	families := []struct {
		name, schema, final string
	}{
		{"llamacpp", performanceExtractLlamaSchema, `{"choices":[{"index":0,"finish_reason":"stop"}],"timings":{"prompt_n":10,"prompt_ms":200,"prompt_per_second":50,"prompt_per_token_ms":20,"cache_n":4,"predicted_n":11,"predicted_ms":250,"predicted_per_second":40,"predicted_per_token_ms":25},"usage":{"completion_tokens":11}}`},
		{"strata", performanceExtractStrataSchema, `{"choices":[{"index":0,"finish_reason":"stop"}],"timings":{"prompt_n":10,"prompt_ms":200,"prompt_per_second":50,"prompt_per_token_ms":20,"cache_n":4,"predicted_n":20,"predicted_ms":400,"predicted_per_second":55,"predicted_per_token_ms":18.182},"usage":{"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":4}}}`},
	}
	fields := []struct {
		name, container, key, affected string
		strataOnly                     bool
	}{
		{"prompt count", "timings", "prompt_n", "prompt", false},
		{"prompt interval", "timings", "prompt_ms", "prefill", false},
		{"prompt rate", "timings", "prompt_per_second", "prompt", false},
		{"prompt reciprocal", "timings", "prompt_per_token_ms", "prompt", false},
		{"cache count", "timings", "cache_n", "cache", false},
		{"output count", "timings", "predicted_n", "output", false},
		{"decode interval", "timings", "predicted_ms", "decode", false},
		{"decode rate", "timings", "predicted_per_second", "rate", false},
		{"decode reciprocal", "timings", "predicted_per_token_ms", "decode", false},
		{"usage output count", "usage", "completion_tokens", "output", false},
		{"usage cached count", "prompt_tokens_details", "cached_tokens", "cache", true},
	}
	for _, family := range families {
		t.Run(family.name, func(t *testing.T) {
			q := performanceExtractQualification{EngineFamily: family.name, APIPath: routeChatCompletions, Streaming: true, SchemaID: family.schema}
			for _, field := range fields {
				if field.strataOnly && family.name != "strata" {
					continue
				}
				t.Run(field.name, func(t *testing.T) {
					for _, invalid := range []string{`"bad"`, `null`, `-1`, `1e999`} {
						t.Run(invalid, func(t *testing.T) {
							patch := `{"` + field.container + `":{"` + field.key + `":` + invalid + `}}`
							if field.container == "prompt_tokens_details" {
								patch = `{"usage":` + patch + `}`
							}
							body := "data: " + family.final + "\n\ndata: " + patch + "\n\ndata: " + family.final + "\n\ndata: [DONE]\n\n"
							r, err := newStreamFrameReader(streamFrameConfig{Protocol: streamProtocolSSE, Upstream: io.NopCloser(strings.NewReader(body)), Extract: func(b []byte) performanceExtractMetadata {
								return extractPerformanceObserved(b, q)
							}})
							if err != nil {
								t.Fatal(err)
							}
							defer r.Close()
							if _, err := io.Copy(io.Discard, r); err != nil {
								t.Fatal(err)
							}
							result := r.Result()
							if result.Outcome != streamOutcomeCompleted {
								t.Fatalf("outcome=%v, want completed", result.Outcome)
							}
							n := result.Numeric
							switch field.affected {
							case "prompt":
								if n.PromptTokens != nil || n.PrefillNs != nil {
									t.Fatalf("invalid prompt evidence restored: %+v", n)
								}
							case "prefill":
								if n.PrefillNs != nil {
									t.Fatalf("invalid prefill interval restored: %+v", n)
								}
							case "cache":
								if n.CachedPromptTokens != nil {
									t.Fatalf("invalid cache count restored: %+v", n)
								}
							case "output":
								if n.OutputTokens != nil || n.EngineDecodeNs != nil || n.EngineDecodeTokensPerSecond != nil {
									t.Fatalf("invalid output evidence restored: %+v", n)
								}
							case "decode":
								if n.EngineDecodeNs != nil || n.EngineDecodeTokensPerSecond != nil {
									t.Fatalf("invalid decode phase restored: %+v", n)
								}
							case "rate":
								if n.EngineDecodeTokensPerSecond != nil {
									t.Fatalf("invalid decode rate restored: %+v", n)
								}
							}
							if field.affected == "prompt" || field.affected == "prefill" || field.affected == "cache" {
								if n.EngineDecodeTokensPerSecond == nil {
									t.Fatalf("unrelated decode rate removed: %+v", n)
								}
							} else if n.PromptTokens == nil || n.PrefillNs == nil || n.CachedPromptTokens == nil {
								t.Fatalf("unrelated prompt/cache evidence removed: %+v", n)
							}
						})
					}
				})
			}
		})
	}
}
