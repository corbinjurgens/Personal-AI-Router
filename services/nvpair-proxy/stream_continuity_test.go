// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

const continuitySSEEvent = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"intact\"}}]}\n\n"
const continuityNDJSONEvent = "{\"response\":\"intact\",\"done\":false}\n"

// Every case goes through a real origin server. Waiting for returned joins the
// handler's terminal and release defers before inspecting its notifications.
type continuityFixture struct {
	p        *Proxy
	w        *recordingWriter
	origin   *httptest.Server
	returned chan struct{}
	nodeID   string
}

func newContinuityFixture(t *testing.T, upstream string, idle time.Duration) *continuityFixture {
	t.Helper()
	w := &recordingWriter{}
	disc := NewDiscovery()
	disc.AddManual(nodeForModel(t, "node-a", upstream, "m"))
	p := newTestProxy(ollamaOnlyProfile(t), NewCodec(w), disc, 0)
	p.SetPriority([]string{"node-a"})
	fx := &continuityFixture{p: p, w: w, returned: make(chan struct{}, 8)}
	fx.origin = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() { fx.returned <- struct{}{} }()
		p.soleFacade().handleHTTPWithStreamIdle(w, r, idle)
	}))
	t.Cleanup(fx.origin.Close)
	return fx
}

func (fx *continuityFixture) request(t *testing.T, path, body string) (*http.Response, []byte, error) {
	t.Helper()
	resp, err := fx.origin.Client().Post(fx.origin.URL+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("request origin: %v", err)
	}
	t.Cleanup(func() { _ = resp.Body.Close() })
	data, readErr := io.ReadAll(resp.Body)
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("close response: %v", err)
	}
	waitChan(t, fx.returned, "origin handler return")
	return resp, data, readErr
}

func (fx *continuityFixture) terminal(t *testing.T, state, reason string) Workload {
	t.Helper()
	nodeID := fx.nodeID
	if nodeID == "" {
		nodeID = "node-a"
	}
	var terminal Workload
	count := 0
	lastSeq := int64(0)
	for _, f := range frames(t, fx.w) {
		switch f.Method {
		case workloadSubmittedMethod, workloadStartedMethod, workloadCompletedMethod, workloadErroredMethod:
			var params workloadParams
			if err := json.Unmarshal(f.Params, &params); err != nil {
				t.Fatalf("decode workload: %v", err)
			}
			wl := params.WorkloadInfo
			if count != 0 || wl.ID != "1" || wl.Seq <= lastSeq {
				t.Fatalf("late/resurrected or inconsistent workload: %+v", wl)
			}
			lastSeq = wl.Seq
			if f.Method == workloadCompletedMethod || f.Method == workloadErroredMethod {
				terminal = wl
				count++
				if (f.Method == workloadCompletedMethod) != (state == "completed") {
					t.Fatalf("terminal method = %s for wanted state %s", f.Method, state)
				}
			}
		}
	}
	if count != 1 || terminal.State != state || terminal.ScheduledOn != nodeID || terminal.CompletedAt == nil {
		t.Fatalf("terminal count=%d, workload=%+v; want one %s on %s", count, terminal, state, nodeID)
	}
	if reason != "" {
		want := "stream interrupted after output began on " + nodeID + ": " + reason
		if terminal.Error == nil || *terminal.Error != want {
			t.Fatalf("workload failure = %v, want %q", terminal.Error, want)
		}
		requests := notificationsOf(t, fx.w, "ollama:proxy/request")
		if len(requests) != 1 {
			t.Fatalf("proxy/request count=%d", len(requests))
		}
		var event RequestEvent
		if err := json.Unmarshal(requests[0], &event); err != nil {
			t.Fatalf("decode proxy/request: %v", err)
		}
		if event.Error != want || event.Status != http.StatusOK || event.NodeID != nodeID {
			t.Fatalf("request result = %+v, want committed 200 with sanitized failure", event)
		}
	}
	fx.p.priorityMu.Lock()
	defer fx.p.priorityMu.Unlock()
	if len(fx.p.priorityReservations) != 0 {
		t.Fatalf("request leaked reservations: %v", fx.p.priorityReservations)
	}
	return terminal
}

// Decode each event independently; a torn upstream fragment must never be
// rescued accidentally by concatenating it with the synthetic error.
func assertContinuityError(t *testing.T, protocol streamProtocol, data []byte, reason string) {
	t.Helper()
	wantMessage := "stream interrupted: " + reason
	errors, done := 0, 0
	if protocol == streamProtocolSSE {
		if !bytes.HasSuffix(data, []byte("\n\n")) {
			t.Fatal("client SSE has an unterminated event")
		}
		for _, event := range bytes.Split(bytes.TrimSuffix(data, []byte("\n\n")), []byte("\n\n")) {
			payload, ok := bytes.CutPrefix(event, []byte("data: "))
			if !ok {
				t.Fatalf("unexpected SSE event shape: %q", event)
			}
			if bytes.Equal(payload, []byte("[DONE]")) {
				done++
				continue
			}
			var object struct {
				Error *struct{ Message, Type, Code string }
			}
			if err := json.Unmarshal(payload, &object); err != nil {
				t.Fatalf("client SSE contains malformed JSON: %v", err)
			}
			if object.Error != nil {
				errors++
				if object.Error.Message != wantMessage || object.Error.Type != "server_error" || object.Error.Code != "stream_interrupted" {
					t.Fatalf("wrong client error: %+v", object.Error)
				}
			}
		}
		if done != 1 {
			t.Fatalf("client DONE count = %d, want 1", done)
		}
	} else {
		dec := json.NewDecoder(bytes.NewReader(data))
		for {
			var object struct {
				Error string
				Done  bool
			}
			if err := dec.Decode(&object); err == io.EOF {
				break
			} else if err != nil {
				t.Fatalf("client NDJSON contains malformed JSON: %v", err)
			}
			if object.Done {
				t.Fatal("interrupted stream fabricated done:true")
			}
			if object.Error != "" {
				errors++
				if object.Error != wantMessage {
					t.Fatalf("client error = %q, want %q", object.Error, wantMessage)
				}
			}
		}
	}
	if errors != 1 {
		t.Fatalf("client error count = %d, want 1", errors)
	}
}

// Hijacking gives precise control over HTTP chunk framing. The first frame is
// flushed intact, then the peer dies partway through the next HTTP chunk.
func TestStreamContinuityRealWireCut(t *testing.T) {
	test := func(name, path, media, intact, torn string, protocol streamProtocol) {
		t.Run(name, func(t *testing.T) {
			var hits atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					t.Errorf("read upstream request: %v", err)
					return
				}
				conn, rw, err := w.(http.Hijacker).Hijack()
				if err != nil {
					t.Errorf("hijack: %v", err)
					return
				}
				defer conn.Close()
				if _, err := fmt.Fprintf(rw, "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nTransfer-Encoding: chunked\r\n\r\n%x\r\n%s\r\n", media, len(intact), intact); err != nil {
					t.Errorf("write intact frame: %v", err)
					return
				}
				if err := rw.Flush(); err != nil {
					t.Errorf("flush intact frame: %v", err)
					return
				}
				if _, err := fmt.Fprintf(rw, "%x\r\n%s", len(torn)+100, torn); err != nil {
					t.Errorf("write torn chunk: %v", err)
					return
				}
				if err := rw.Flush(); err != nil {
					t.Errorf("flush torn chunk: %v", err)
				}
			}))
			defer upstream.Close()
			fx := newContinuityFixture(t, upstream.URL, time.Second)
			resp, data, err := fx.request(t, path, `{"model":"m","stream":true}`)
			if err != nil {
				t.Fatalf("client HTTP body must end cleanly: %v", err)
			}
			if resp.StatusCode != http.StatusOK || !bytes.HasPrefix(data, []byte(intact)) || bytes.Contains(data, []byte("TORN_SENTINEL")) {
				t.Fatalf("status=%d; intact prefix=%v; torn tail exposed=%v", resp.StatusCode, bytes.HasPrefix(data, []byte(intact)), bytes.Contains(data, []byte("TORN_SENTINEL")))
			}
			assertContinuityError(t, protocol, data, streamInterruptReasonReadFailed)
			fx.terminal(t, "failed", streamInterruptReasonReadFailed)
			if hits.Load() != 1 {
				t.Fatalf("upstream dispatches=%d, want one", hits.Load())
			}
		})
	}
	test("SSE", "/v1/chat/completions", "text/event-stream", continuitySSEEvent, `data: {"TORN_SENTINEL":"`, streamProtocolSSE)
	test("NDJSON", "/api/chat", "application/x-ndjson", continuityNDJSONEvent, `{"TORN_SENTINEL":"`, streamProtocolNDJSON)
}

func TestStreamContinuityProtocolSelection(t *testing.T) {
	test := func(name, path, flag, media, payload string, protocol streamProtocol, reason string) {
		t.Run(name, func(t *testing.T) {
			var seen string
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Errorf("read request: %v", err)
					return
				}
				seen = string(body)
				w.Header().Set("Content-Type", media)
				w.Header().Set("Content-MD5", "obsolete")
				w.Header().Set("Digest", "obsolete")
				w.Header().Set("Content-Digest", "obsolete")
				w.Header().Set("Repr-Digest", "obsolete")
				w.Header().Set("ETag", "obsolete")
				w.Header().Set("Content-Length", fmt.Sprint(len(payload)))
				w.Header().Set("Access-Control-Allow-Origin", "https://app.example")
				if _, err := io.WriteString(w, payload); err != nil {
					t.Errorf("write upstream: %v", err)
				}
			}))
			defer upstream.Close()
			fx := newContinuityFixture(t, upstream.URL, time.Second)
			body := `{ "model":"m", "messages":[], "keep":{"order":1}` + flag + ` }`
			resp, data, err := fx.request(t, path, body)
			if err != nil || resp.StatusCode != http.StatusOK {
				t.Fatalf("status=%d read=%v", resp.StatusCode, err)
			}
			if seen != body {
				t.Fatal("stream validation changed unrelated request bytes")
			}
			if resp.Header.Get("Content-Type") != media || resp.Header.Get("Access-Control-Allow-Origin") != "https://app.example" {
				t.Fatalf("response provenance/CORS/media changed: %v", resp.Header)
			}
			if isInferenceRequest(ollamaOnlyProfile(t), http.MethodPost, path) && (resp.Header.Get(nodepolicy.NodeHeader) != "node-a" || resp.Header.Get(nodepolicy.ModelHeader) != "m" || resp.Header.Get(nodepolicy.EngineHeader) != "ollama") {
				t.Fatalf("inference provenance changed: %v", resp.Header)
			}
			if protocol != streamProtocolUnknown {
				for _, name := range []string{"Content-Length", "Content-MD5", "Digest", "Content-Digest", "Repr-Digest", "ETag"} {
					if resp.Header.Get(name) != "" {
						t.Fatalf("managed response retained %s", name)
					}
				}
			}
			if reason != "" {
				assertContinuityError(t, protocol, data, reason)
				fx.terminal(t, "failed", reason)
			} else {
				if string(data) != payload {
					t.Fatal("complete/raw response bytes changed")
				}
				if isInferenceRequest(ollamaOnlyProfile(t), http.MethodPost, path) {
					fx.terminal(t, "completed", "")
				} else if len(notificationsOf(t, fx.w, workloadCompletedMethod)) != 0 {
					t.Fatal("unclassified route emitted an inference workload")
				}
			}
		})
	}
	const explicitTrue = `,"stream":true`
	const explicitFalse = `,"stream":false`
	test("OpenAI declared", "/v1/chat/completions", explicitTrue, "text/event-stream", continuitySSEEvent+"data: [DONE]\n\n", streamProtocolSSE, "")
	test("legacy completions response selected", "/v1/completions", "", "text/event-stream; charset=utf-8", "data: [DONE]\n\n", streamProtocolSSE, "")
	test("OpenAI finish reason clean EOF", "/v1/chat/completions", explicitTrue, "text/event-stream", "data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"}]}\n\n", streamProtocolSSE, "")
	test("OpenAI all choices required", "/v1/chat/completions", explicitTrue+`,"n":2`, "text/event-stream", "data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"}]}\n\n", streamProtocolSSE, streamInterruptReasonPrematureEOF)
	test("OpenAI default raw", "/v1/chat/completions", "", "application/json", `{"choices":[]}`, streamProtocolUnknown, "")
	test("OpenAI null raw", "/v1/chat/completions", `,"stream":null`, "application/json", `{"choices":[]}`, streamProtocolUnknown, "")
	test("OpenAI null SSE", "/v1/completions", `,"stream":null`, "text/event-stream", continuitySSEEvent, streamProtocolSSE, streamInterruptReasonPrematureEOF)
	test("OpenAI explicit false overrides media", "/v1/chat/completions", explicitFalse, "text/event-stream", "raw response", streamProtocolUnknown, "")
	test("Ollama default json", "/api/chat", "", "application/json", continuityNDJSONEvent+`{"done":true}`, streamProtocolNDJSON, "")
	test("Ollama chat null", "/api/chat", `,"stream":null`, "application/json", continuityNDJSONEvent, streamProtocolNDJSON, streamInterruptReasonPrematureEOF)
	test("Ollama generate null", "/api/generate", `,"stream":null`, "application/x-ndjson", "{\"done\":true}\n", streamProtocolNDJSON, "")
	test("Ollama generate declared", "/api/generate", explicitTrue, "application/x-ndjson", "{\"done\":true}\n", streamProtocolNDJSON, "")
	test("Ollama explicit false", "/api/generate", explicitFalse, "application/json", `{"response":"single"}`, streamProtocolUnknown, "")
	test("OpenAI wrong media", "/v1/chat/completions", explicitTrue, "application/json", `{"choices":[]}`, streamProtocolSSE, streamInterruptReasonMalformed)
	test("Ollama wrong media", "/api/chat", "", "text/plain", `{"done":true}`, streamProtocolNDJSON, streamInterruptReasonMalformed)
	test("OpenAI empty stream", "/v1/completions", explicitTrue, "text/event-stream", "", streamProtocolSSE, streamInterruptReasonPrematureEOF)
	test("Ollama empty stream", "/api/chat", "", "application/json", "", streamProtocolNDJSON, streamInterruptReasonPrematureEOF)
	test("malformed SSE", "/v1/chat/completions", explicitTrue, "text/event-stream", "data: bad-json\n\n", streamProtocolSSE, streamInterruptReasonMalformed)
	test("malformed NDJSON", "/api/chat", "", "application/x-ndjson", "bad-json\n", streamProtocolNDJSON, streamInterruptReasonMalformed)
	test("premature clean EOF", "/api/chat", "", "application/json", continuityNDJSONEvent, streamProtocolNDJSON, streamInterruptReasonPrematureEOF)
	test("Anthropic stays raw", "/v1/messages", explicitTrue, "text/event-stream", "event: message_stop\ndata: {}\n\n", streamProtocolUnknown, "")
	test("Responses stays raw", "/v1/responses", explicitTrue, "text/event-stream", "event: response.completed\ndata: {}\n\n", streamProtocolUnknown, "")
}

func TestStreamContinuityRejectsInvalidStreamFlag(t *testing.T) {
	for _, flag := range []string{`"true"`, `1`, `[]`, `true,"stream":false`, `null,"stream":null`} {
		t.Run(flag, func(t *testing.T) {
			var hits atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { hits.Add(1) }))
			defer upstream.Close()
			fx := newContinuityFixture(t, upstream.URL, time.Second)
			resp, _, err := fx.request(t, "/api/chat", `{"model":"m","stream":`+flag+`}`)
			if err != nil || resp.StatusCode != http.StatusBadRequest || hits.Load() != 0 {
				t.Fatalf("status=%d read=%v dispatches=%d", resp.StatusCode, err, hits.Load())
			}
		})
	}
}

func TestNonInferenceRequestOnInferencePathPreservesRequest(t *testing.T) {
	const requestBody = `{"stream":"engine-extension"}`
	const responseBody = "engine control response"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read forwarded request: %v", err)
			return
		}
		if r.Method != http.MethodGet || !bytes.Equal(body, []byte(requestBody)) {
			t.Errorf("forwarded request = %s %q, want GET with original body %q", r.Method, body, requestBody)
		}
		if got := r.Header.Get(peerStreamFramingHeader); got != "" {
			t.Errorf("forwarded internal framing header = %q, want absent", got)
		}
		w.WriteHeader(http.StatusTeapot)
		_, _ = io.WriteString(w, responseBody)
	}))
	defer upstream.Close()

	fx := newContinuityFixture(t, upstream.URL, time.Second)
	fac := fx.p.soleFacade()
	fac.port = fac.profile.FacadePort
	fac.discovery.AddManual(selfNode("node-a", fac.port, "m"))
	setBackend(t, fac, upstream.URL, true)

	r, err := http.NewRequest(http.MethodGet, fx.origin.URL+"/api/chat", strings.NewReader(requestBody))
	if err != nil {
		t.Fatal(err)
	}
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set(peerStreamFramingHeader, "client-spoof")
	resp, err := fx.origin.Client().Do(r)
	if err != nil {
		t.Fatalf("request origin: %v", err)
	}
	data, readErr := io.ReadAll(resp.Body)
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("close response: %v", err)
	}
	waitChan(t, fx.returned, "origin handler return")
	if readErr != nil || resp.StatusCode != http.StatusTeapot || string(data) != responseBody {
		t.Fatalf("response status=%d body=%q read=%v, want upstream 418 and %q", resp.StatusCode, data, readErr, responseBody)
	}
	for _, frame := range frames(t, fx.w) {
		if frame.Method == workloadErroredMethod {
			t.Fatalf("non-inference request emitted workload failure: %s", frame.Params)
		}
	}
}

func TestStreamContinuityEffectiveLocalDefaults(t *testing.T) {
	test := func(name, path, request, defaults, media, payload, reason string) {
		t.Run(name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Errorf("read effective request: %v", err)
					return
				}
				if !bytes.Equal(body, mergeDefaults([]byte(request), []byte(defaults))) {
					t.Error("engine did not receive effective defaults")
				}
				w.Header().Set("Content-Type", media)
				w.Header().Set(peerStreamFramingHeader, "engine-spoof")
				w.Header().Set("Trailer", peerStreamFramingHeader)
				if _, err := io.WriteString(w, payload); err != nil {
					t.Errorf("write response: %v", err)
				}
				w.Header().Set(peerStreamFramingHeader, "late-engine-spoof")
			}))
			defer upstream.Close()
			fx := newContinuityFixture(t, upstream.URL, time.Second)
			fac := fx.p.soleFacade()
			fac.port = fac.profile.FacadePort
			fac.discovery.AddManual(selfNode("node-a", fac.port, "m"))
			setBackend(t, fac, upstream.URL, true)
			pol := nodepolicy.Default()
			pol.Profiles = []nodepolicy.Profile{{Name: "m", Engine: "ollama", Model: "m", RequestOptions: json.RawMessage(defaults)}}
			fx.p.admission.setPolicy(pol)
			resp, data, err := fx.request(t, path, request)
			if err != nil || resp.StatusCode != http.StatusOK {
				t.Fatalf("status=%d read=%v", resp.StatusCode, err)
			}
			if resp.Header.Get(peerStreamFramingHeader) != "" || resp.Trailer.Get(peerStreamFramingHeader) != "" {
				t.Fatal("internal metadata leaked from non-peer engine")
			}
			if reason != "" {
				assertContinuityError(t, streamProtocolSSE, data, reason)
				fx.terminal(t, "failed", reason)
			} else {
				if string(data) != payload {
					t.Fatalf("raw response changed: length=%d want=%d", len(data), len(payload))
				}
				fx.terminal(t, "completed", "")
			}
			if fx.p.admission.currentActive() != 0 {
				t.Fatal("local admission ticket leaked")
			}
		})
	}
	test("all default choices required", "/v1/chat/completions", `{"model":"m","stream":true}`, `{"n":2}`, "text/event-stream", "data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"}]}\n\n", streamInterruptReasonPrematureEOF)
	test("default disables native streaming", "/api/chat", `{"model":"m"}`, `{"stream":false}`, "application/json", `{"response":"`+strings.Repeat("x", 1<<20)+`","done":true}`, "")
	test("default declares OpenAI streaming", "/v1/completions", `{"model":"m"}`, `{"stream":true}`, "application/json", `{"choices":[]}`, streamInterruptReasonMalformed)
}

func TestStreamContinuityRequestsIdentityEncoding(t *testing.T) {
	const payload = continuitySSEEvent + "data: [DONE]\n\n"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Accept-Encoding"); got != "identity" {
			t.Errorf("engine Accept-Encoding=%q, want identity", got)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, payload); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	defer upstream.Close()
	fx := newContinuityFixture(t, upstream.URL, time.Second)
	req, err := http.NewRequest(http.MethodPost, fx.origin.URL+"/v1/chat/completions", strings.NewReader(`{"model":"m","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Accept-Encoding", "gzip, deflate")
	resp, err := fx.origin.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil || resp.StatusCode != http.StatusOK || string(data) != payload {
		t.Fatalf("status=%d length=%d read=%v", resp.StatusCode, len(data), err)
	}
	waitChan(t, fx.returned, "origin return")
	fx.terminal(t, "completed", "")
}

func TestStreamContinuityRejectsCompressedStream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Content-Encoding", "gzip")
		compressed := gzip.NewWriter(w)
		if _, err := io.WriteString(compressed, continuitySSEEvent+"data: [DONE]\n\n"); err != nil {
			t.Errorf("compress response: %v", err)
		}
		if err := compressed.Close(); err != nil {
			t.Errorf("finish compressed response: %v", err)
		}
	}))
	defer upstream.Close()
	fx := newContinuityFixture(t, upstream.URL, time.Second)
	resp, data, err := fx.request(t, "/v1/chat/completions", `{"model":"m","stream":true}`)
	if err != nil || resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Encoding") != "" {
		t.Fatalf("status=%d read=%v encoding=%q", resp.StatusCode, err, resp.Header.Get("Content-Encoding"))
	}
	assertContinuityError(t, streamProtocolSSE, data, streamInterruptReasonMalformed)
	fx.terminal(t, "failed", streamInterruptReasonMalformed)
}

func TestStreamContinuityStrictLocalHidesPeerMetadata(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(peerStreamFramingHeader, "engine-spoof")
		w.Header().Set("Trailer", peerStreamFramingHeader)
		if _, err := io.WriteString(w, "data: [DONE]\n\n"); err != nil {
			t.Error(err)
		}
		w.Header().Set(peerStreamFramingHeader, "late-engine-spoof")
	}))
	defer upstream.Close()
	p := testProxy(ollamaOnlyProfile(t), NewDiscovery(), 0)
	configureStrict(t, p.admission, filepath.Join(t.TempDir(), "capacity.json"))
	setBackend(t, p.soleFacade(), upstream.URL, true)
	rec := httptest.NewRecorder()
	p.soleFacade().handleHTTP(rec, strictHTTPRequest(t))
	resp := rec.Result()
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK || resp.Header.Get(peerStreamFramingHeader) != "" || resp.Trailer.Get(peerStreamFramingHeader) != "" {
		t.Fatalf("status=%d internal header=%q trailer=%q", resp.StatusCode, resp.Header.Get(peerStreamFramingHeader), resp.Trailer.Get(peerStreamFramingHeader))
	}
	if p.admission.currentActive() != 1 {
		t.Fatal("HTTP completion released uncertain strict capacity")
	}
}

func TestStreamContinuityStripsIntegrityTrailers(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Header().Set("Trailer", "Digest, Content-Digest, X-Fixture")
		if _, err := io.WriteString(w, continuityNDJSONEvent); err != nil {
			t.Errorf("write upstream frame: %v", err)
			return
		}
		w.Header().Set("Digest", "obsolete")
		w.Header().Set("Content-Digest", "obsolete")
		w.Header().Set("X-Fixture", "preserved")
	}))
	defer upstream.Close()
	fx := newContinuityFixture(t, upstream.URL, time.Second)
	resp, data, err := fx.request(t, "/api/chat", `{"model":"m"}`)
	if err != nil {
		t.Fatalf("read managed response: %v", err)
	}
	assertContinuityError(t, streamProtocolNDJSON, data, streamInterruptReasonPrematureEOF)
	if resp.Trailer.Get("Digest") != "" || resp.Trailer.Get("Content-Digest") != "" || resp.Trailer.Get("X-Fixture") != "preserved" {
		t.Fatalf("managed trailers = %v; want only unrelated fixture metadata", resp.Trailer)
	}
	fx.terminal(t, "failed", streamInterruptReasonPrematureEOF)
}

func TestStreamContinuityIdleDeadline(t *testing.T) {
	test := func(name string, drip, refresh bool) {
		t.Run(name, func(t *testing.T) {
			stopped := make(chan struct{})
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(stopped)
				w.Header().Set("Content-Type", "text/event-stream")
				if _, err := io.WriteString(w, continuitySSEEvent); err != nil {
					t.Errorf("initial frame: %v", err)
					return
				}
				w.(http.Flusher).Flush()
				ticker := time.NewTicker(20 * time.Millisecond)
				defer ticker.Stop()
				for i := 0; ; i++ {
					select {
					case <-r.Context().Done():
						return
					case <-ticker.C:
						if refresh && i == 15 {
							_, _ = io.WriteString(w, "data: [DONE]\n\n")
							return
						}
						if refresh {
							_, _ = io.WriteString(w, ": heartbeat\n\n")
						} else if drip {
							_, _ = io.WriteString(w, "x")
						}
						w.(http.Flusher).Flush()
					}
				}
			}))
			defer upstream.Close()
			fx := newContinuityFixture(t, upstream.URL, 120*time.Millisecond)
			fx.origin.Client().Timeout = 2 * time.Second
			start := time.Now()
			_, data, err := fx.request(t, "/v1/chat/completions", `{"model":"m","stream":true}`)
			if err != nil {
				t.Fatalf("idle termination must provide clean bounded EOF: %v", err)
			}
			waitChan(t, stopped, "upstream body closure")
			if refresh {
				if time.Since(start) < 240*time.Millisecond || !bytes.HasSuffix(data, []byte("data: [DONE]\n\n")) {
					t.Fatal("complete events failed to refresh the idle deadline")
				}
				fx.terminal(t, "completed", "")
			} else {
				assertContinuityError(t, streamProtocolSSE, data, streamInterruptReasonIdle)
				fx.terminal(t, "failed", streamInterruptReasonIdle)
			}
		})
	}
	test("silent after event", false, false)
	test("partial byte drip does not refresh", true, false)
	test("complete heartbeat refreshes", false, true)
}

func TestStreamContinuityUpstreamErrorAndLogs(t *testing.T) {
	const sentinel = "SYNTHETIC_PRIVATE_CONTENT_KEY_73816"
	var logs bytes.Buffer
	old := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	defer slog.SetDefault(old)
	test := func(name, path, media, payload string) {
		t.Run(name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", media)
				if _, err := io.WriteString(w, payload); err != nil {
					t.Errorf("write error frame: %v", err)
				}
			}))
			defer upstream.Close()
			fx := newContinuityFixture(t, upstream.URL, time.Second)
			_, data, err := fx.request(t, path, `{"model":"m","stream":true,"messages":[{"role":"user","content":"`+sentinel+`"}]}`)
			if err != nil || string(data) != payload {
				t.Fatalf("known upstream error must be preserved once: read=%v", err)
			}
			fx.terminal(t, "failed", streamInterruptReasonUpstreamError)
		})
	}
	test("SSE", "/v1/chat/completions", "text/event-stream", "data: {\"error\":{\"message\":\""+sentinel+"\"}}\n\n")
	test("NDJSON", "/api/chat", "application/json", "{\"error\":\""+sentinel+"\"}\n")
	if strings.Contains(logs.String(), sentinel) {
		t.Fatal("request or response content leaked into operational logs")
	}
}

func TestStreamContinuityUnmanagedAbort(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, rw, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Errorf("hijack: %v", err)
			return
		}
		defer conn.Close()
		if _, err := io.WriteString(rw, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 500\r\n\r\n{\"partial\":"); err != nil {
			t.Errorf("write unmanaged response: %v", err)
			return
		}
		if err := rw.Flush(); err != nil {
			t.Errorf("flush unmanaged response: %v", err)
		}
	}))
	defer upstream.Close()
	fx := newContinuityFixture(t, upstream.URL, time.Second)
	_, data, err := fx.request(t, "/v1/chat/completions", `{"model":"m","stream":false}`)
	if err == nil || string(data) != `{"partial":` {
		t.Fatalf("unmanaged abort must retain HTTP copy failure and raw bytes: read=%v", err)
	}
	fx.terminal(t, "failed", "upstream response copy aborted")
}

func TestStreamContinuityClientDisconnect(t *testing.T) {
	stopped := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(stopped)
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, continuitySSEEvent); err != nil {
			t.Errorf("write first event: %v", err)
			return
		}
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer upstream.Close()
	fx := newContinuityFixture(t, upstream.URL, time.Second)
	resp, err := fx.origin.Client().Post(fx.origin.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"m","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	first := make([]byte, len(continuitySSEEvent))
	if _, err := io.ReadFull(resp.Body, first); err != nil {
		t.Fatalf("read first event: %v", err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatal(err)
	}
	waitChan(t, stopped, "disconnected upstream")
	waitChan(t, fx.returned, "cancelled handler")
	fx.terminal(t, "cancelled", "")
}
