// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"unsafe"
)

type responseSinkWriter struct {
	header   http.Header
	body     bytes.Buffer
	statuses []int
	short    bool
	flushErr error
}

func (w *responseSinkWriter) Header() http.Header    { return w.header }
func (w *responseSinkWriter) WriteHeader(status int) { w.statuses = append(w.statuses, status) }
func (w *responseSinkWriter) Write(p []byte) (int, error) {
	if w.short && len(p) > 0 {
		return w.body.Write(p[:len(p)-1])
	}
	return w.body.Write(p)
}
func (w *responseSinkWriter) FlushError() error { return w.flushErr }

func TestResponseSink(t *testing.T) {
	t.Run("fixed metadata borrows retained parsing envelope", func(t *testing.T) {
		ctx, root, owner := chatTestInferenceContext(t, 256<<10, 0)
		parser, err := owner.scope.tryAcquire(ctx, 96<<10)
		if err != nil {
			t.Fatal(err)
		}
		owner.retainParser(parser)
		sink := newResponseSink(httptest.NewRecorder(), nil)
		_ = withResponseSink(ctx, sink)
		if unsafe.Sizeof(*sink) > 256 || sink.bootstrap == nil || sink.bootstrap.charge != parser.charge || root.used != 96<<10 {
			t.Fatalf("unaccounted sink: size=%d borrower=%v used=%d", unsafe.Sizeof(*sink), sink.bootstrap != nil, root.used)
		}
		parser.close()
		if root.used != 96<<10 {
			t.Fatalf("sink borrower lost envelope: %d", root.used)
		}
		owner.close()
		if root.used != 0 || root.live != 0 || sink.bootstrap != nil {
			t.Fatalf("sink cleanup retained storage: used=%d live=%d borrower=%v", root.used, root.live, sink.bootstrap != nil)
		}
	})

	t.Run("full delivery requires successful write and flush", func(t *testing.T) {
		w := &responseSinkWriter{header: make(http.Header)}
		sink := newResponseSink(w, nil)
		sink.WriteHeader(http.StatusAccepted)
		sink.WriteHeader(http.StatusBadGateway)
		sink.output(5, true)
		if _, err := sink.Write([]byte("frame")); err != nil {
			t.Fatal(err)
		}
		sink.streamResult(streamFrameResult{Outcome: streamOutcomeCompleted})
		before := sink.result()
		if !before.ProtocolComplete || before.DeliveryComplete || before.Written != 5 || before.Flushed != 0 || before.Genuine != 5 || before.Synthetic != 0 {
			t.Fatalf("pre-flush delivery: %+v", before)
		}
		if err := sink.FlushError(); err != nil {
			t.Fatal(err)
		}
		after := sink.result()
		if !after.DeliveryComplete || after.Flushed != 5 || len(w.statuses) != 1 || w.statuses[0] != http.StatusAccepted {
			t.Fatalf("delivery/status: %+v statuses=%v", after, w.statuses)
		}
		if err := sink.interrupt(streamProtocolSSE, streamInterruptReasonReadFailed); err != nil {
			t.Fatal(err)
		}
		if w.body.String() != "frame" {
			t.Fatalf("late interruption appended after successful terminal: %q", w.body.String())
		}
		if responseSinkFrom(withResponseSink(context.Background(), sink)) != sink || sink.Unwrap() != w {
			t.Fatal("sink identity lost")
		}
	})

	t.Run("partial write cancels without claiming delivery", func(t *testing.T) {
		cancelled := 0
		w := &responseSinkWriter{header: make(http.Header), short: true}
		sink := newResponseSink(w, func() { cancelled++ })
		sink.output(5, true)
		n, err := sink.Write([]byte("frame"))
		if n != 4 || !errors.Is(err, io.ErrShortWrite) || cancelled != 1 {
			t.Fatalf("partial write: n=%d err=%v cancelled=%d", n, err, cancelled)
		}
		if err := sink.FlushError(); err != nil {
			t.Fatal(err)
		}
		sink.streamResult(streamFrameResult{Outcome: streamOutcomeCompleted})
		if got := sink.result(); got.DeliveryComplete || !errors.Is(got.WriteError, io.ErrShortWrite) || got.Written != 4 {
			t.Fatalf("partial delivery: %+v", got)
		}
		if err := sink.interrupt(streamProtocolSSE, streamInterruptReasonReadFailed); !errors.Is(err, io.ErrShortWrite) {
			t.Fatalf("late interruption lost client failure: %v", err)
		}
		if w.body.String() != "fram" || cancelled != 1 {
			t.Fatalf("wrote after client failure: %q cancellations=%d", w.body.String(), cancelled)
		}
	})

	t.Run("flush failure cancels without successful delivery", func(t *testing.T) {
		cancelled := 0
		w := &responseSinkWriter{header: make(http.Header), flushErr: io.ErrClosedPipe}
		sink := newResponseSink(w, func() { cancelled++ })
		sink.output(5, true)
		if _, err := sink.Write([]byte("frame")); err != nil {
			t.Fatal(err)
		}
		if err := sink.FlushError(); !errors.Is(err, io.ErrClosedPipe) {
			t.Fatalf("flush error: %v", err)
		}
		sink.streamResult(streamFrameResult{Outcome: streamOutcomeCompleted})
		if got := sink.result(); got.DeliveryComplete || got.Flushed != 0 || !errors.Is(got.WriteError, io.ErrClosedPipe) || cancelled != 1 {
			t.Fatalf("failed flush: %+v cancelled=%d", got, cancelled)
		}
		if err := sink.interrupt(streamProtocolSSE, streamInterruptReasonReadFailed); !errors.Is(err, io.ErrClosedPipe) {
			t.Fatalf("late interruption lost flush failure: %v", err)
		}
		if w.body.String() != "frame" || cancelled != 1 {
			t.Fatalf("wrote after flush failure: %q cancellations=%d", w.body.String(), cancelled)
		}
	})

	t.Run("one interruption retains status and records unavailable continuation", func(t *testing.T) {
		w := httptest.NewRecorder()
		sink := newResponseSink(w, nil)
		sink.WriteHeader(http.StatusAccepted)
		if err := sink.interrupt(streamProtocolSSE, streamInterruptReasonBudget); err != nil {
			t.Fatal(err)
		}
		if err := sink.interrupt(streamProtocolSSE, streamInterruptReasonIdle); err != nil {
			t.Fatal(err)
		}
		want := "data: {\"error\":{\"message\":\"stream interrupted: stream frame buffer exhausted\",\"type\":\"server_error\",\"code\":\"stream_interrupted\"}}\n\ndata: [DONE]\n\n"
		if w.Code != http.StatusAccepted || w.Body.String() != want {
			t.Fatalf("interruption: status=%d body=%q", w.Code, w.Body.String())
		}
		if got := sink.result(); got.Reason != streamInterruptReasonBudget || !got.ContinuationUnavailable || got.ProtocolComplete || !got.DeliveryComplete || got.Genuine != 0 || got.Synthetic != int64(len(want)) {
			t.Fatalf("interruption disposition: %+v", got)
		}
	})

	t.Run("bootstrap refusal body uses one clean ending", func(t *testing.T) {
		w := httptest.NewRecorder()
		sink := newResponseSink(w, nil)
		body := sink.interruptedBody(streamProtocolNDJSON, streamInterruptReasonBudget)
		if _, err := io.Copy(sink, body); err != nil {
			t.Fatal(err)
		}
		if err := body.Close(); err != nil {
			t.Fatal(err)
		}
		if err := sink.FlushError(); err != nil {
			t.Fatal(err)
		}
		if err := sink.interrupt(streamProtocolNDJSON, streamInterruptReasonBudget); err != nil {
			t.Fatal(err)
		}
		want := "{\"error\":\"stream interrupted: stream frame buffer exhausted\"}\n"
		if w.Body.String() != want || !sink.result().DeliveryComplete {
			t.Fatalf("bootstrap ending: body=%q result=%+v", w.Body.String(), sink.result())
		}
	})

	t.Run("public DONE remains withheld on trailing bytes and read failure", func(t *testing.T) {
		for _, scenario := range []string{"trailing data", "failed EOF", "clean EOF"} {
			t.Run(scenario, func(t *testing.T) {
				q, _ := chatFixture()
				budget := &chatTestBudget{}
				body := chatCompleteStream()
				var reader io.Reader = strings.NewReader(body)
				if scenario == "trailing data" {
					reader = strings.NewReader(body + "data: {}\n\n")
				}
				if scenario == "failed EOF" {
					reader = io.MultiReader(strings.NewReader(body), responseSinkFailedEOF{})
				}
				var exposed bytes.Buffer
				result, err := q.parseResponse(context.Background(), reader, true, 100, 100, q.Now(), budget.acquire, func(frame []byte) error { _, e := exposed.Write(frame); return e })
				wantComplete := scenario == "clean EOF"
				if result.Complete != wantComplete || (err == nil) != wantComplete || budget.held != 0 {
					t.Fatalf("public result: %+v err=%v held=%d", result, err, budget.held)
				}
				done := bytes.Count(exposed.Bytes(), []byte("[DONE]"))
				if (wantComplete && done != 1) || (!wantComplete && done != 0) {
					t.Fatalf("premature public success: complete=%v done=%d body=%q", result.Complete, done, exposed.String())
				}
			})
		}
	})

	t.Run("peer trailer validation is strict and authenticated", func(t *testing.T) {
		request := originStreamRequest{peerNormalized: true}
		for _, values := range [][]string{nil, {"v1;completed", "v1;completed"}, {"v2;completed"}, {"v1;interrupted;unknown"}, {"v1;completed;extra"}, {strings.Repeat("x", 129)}} {
			if _, err := request.withPeerTrailer(values); err == nil {
				t.Fatalf("accepted trailer %q", values)
			}
		}
		if _, err := (originStreamRequest{}).withPeerTrailer([]string{"v1;completed"}); err == nil {
			t.Fatal("engine supplied metadata trusted")
		}
		if reason, err := request.withPeerTrailer([]string{"v1;interrupted;upstream read failed"}); err != nil || reason != "upstream read failed" {
			t.Fatalf("valid trailer: %q %v", reason, err)
		}
	})

	t.Run("authenticated peer ending survives exactly once", func(t *testing.T) {
		for _, protocol := range []streamProtocol{streamProtocolSSE, streamProtocolNDJSON} {
			payload := wantSSEInterruption(streamInterruptReasonIdle)
			if protocol == streamProtocolNDJSON {
				payload = wantNDJSONInterruption(streamInterruptReasonIdle)
			}
			w := httptest.NewRecorder()
			sink := newResponseSink(w, nil)
			budget := newFrameBudget(1 << 20)
			reader, err := newStreamFrameReader(streamFrameConfig{Protocol: protocol, Upstream: newChunkedStreamBody(payload, 1), Budget: budget, Sink: sink, PeerNormalized: true, PeerTrailer: func() []string { return []string{"v1;interrupted;upstream stopped sending events"} }})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := io.Copy(sink, reader); err != nil {
				t.Fatal(err)
			}
			if err := reader.Close(); err != nil {
				t.Fatal(err)
			}
			if err := sink.FlushError(); err != nil {
				t.Fatal(err)
			}
			if err := sink.interrupt(protocol, streamInterruptReasonReadFailed); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(w.Body.Bytes(), payload) || sink.result().Reason != streamInterruptReasonIdle || !sink.result().DeliveryComplete || budget.inUse() != 0 {
				t.Fatalf("peer ending: body=%q result=%+v budget=%d", w.Body.Bytes(), sink.result(), budget.inUse())
			}
		}
	})

	t.Run("missing malformed or torn peer ending cannot advertise success", func(t *testing.T) {
		cases := []struct {
			name, body string
			trailer    []string
			wantReason string
		}{
			{"missing success trailer", "data: [DONE]\n\n", nil, streamInterruptReasonPeerLink},
			{"malformed trailer", "data: [DONE]\n\n", []string{"v1;interrupted;unknown"}, streamInterruptReasonMalformed},
			{"duplicate trailer", "data: [DONE]\n\n", []string{"v1;completed", "v1;completed"}, streamInterruptReasonMalformed},
			{"torn advertised ending", string(wantSSEInterruption(streamInterruptReasonIdle)[:40]), []string{"v1;interrupted;upstream stopped sending events"}, streamInterruptReasonPeerLink},
			{"missing terminal after error", strings.TrimSuffix(string(wantSSEInterruption(streamInterruptReasonIdle)), "data: [DONE]\n\n"), []string{"v1;interrupted;upstream stopped sending events"}, streamInterruptReasonPeerLink},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				w := httptest.NewRecorder()
				sink := newResponseSink(w, nil)
				budget := newFrameBudget(1 << 20)
				reader, err := newStreamFrameReader(streamFrameConfig{Protocol: streamProtocolSSE, Upstream: io.NopCloser(strings.NewReader(tc.body)), Budget: budget, Sink: sink, PeerNormalized: true, PeerTrailer: func() []string { return tc.trailer }})
				if err != nil {
					t.Fatal(err)
				}
				if _, err := io.Copy(sink, reader); err != nil {
					t.Fatal(err)
				}
				if err := reader.Close(); err != nil {
					t.Fatal(err)
				}
				if err := sink.FlushError(); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(w.Body.Bytes(), wantSSEInterruption(tc.wantReason)) || reader.Result().Reason != tc.wantReason || sink.result().ProtocolComplete || budget.inUse() != 0 {
					t.Fatalf("peer repaired incorrectly: body=%q result=%+v sink=%+v retained=%d", w.Body.Bytes(), reader.Result(), sink.result(), budget.inUse())
				}
			})
		}
	})
}

type responseSinkFailedEOF struct{}

func (responseSinkFailedEOF) Read([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }
