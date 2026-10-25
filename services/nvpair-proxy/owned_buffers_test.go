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
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"
	"unsafe"

	"nvpair-shared/nodepolicy"
)

type ownedBodySource struct {
	remaining int64
	budget    *bufferBudget
	peak      int64
	reads     int
	probe     bool
	closed    bool
	failure   error
}

func (s *ownedBodySource) Read(p []byte) (int, error) {
	s.budget.mu.Lock()
	used, capacity := s.budget.used, s.budget.capacity
	s.budget.mu.Unlock()
	if used > capacity {
		panic("allocation exceeded configured inference root")
	}
	if used > s.peak {
		s.peak = used
	}
	s.reads++
	if s.remaining == 0 {
		s.probe = len(p) == 1
		if s.failure != nil {
			return 0, s.failure
		}
		return 0, io.EOF
	}
	n := int64(len(p))
	if n > s.remaining {
		n = s.remaining
	}
	for i := range p[:int(n)] {
		p[i] = ' '
	}
	s.remaining -= n
	return int(n), nil
}

func (s *ownedBodySource) Close() error { s.closed = true; return nil }

func TestOwnedInferenceBuffers(t *testing.T) {
	t.Run("known length EOF probe avoids replacement capacity", func(t *testing.T) {
		budget := bufferTestBudget(t, 192<<20, 0)
		p := &Proxy{buffers: budget}
		source := &ownedBodySource{remaining: 17, budget: budget}
		r := httptest.NewRequest(http.MethodPost, "/api/generate", nil)
		r.Body, r.ContentLength = source, 17
		r, owner := p.ownInferenceBuffers(r)
		defer owner.close()
		body, _, err := bufferBodyAndModel(httptest.NewRecorder(), r, 100)
		if err != nil || len(body) != 17 || cap(body) != 17 || !source.probe || source.reads != 2 {
			t.Fatalf("known EOF: length=%d capacity=%d probe=%v reads=%d err=%v", len(body), cap(body), source.probe, source.reads, err)
		}
		bufferAssertTotals(t, budget, 17+(96<<10), 2, 0)
		owner.close()
		bufferAssertTotals(t, budget, 0, 0, 0)
	})
	t.Run("known length extra byte stays charged during replacement", func(t *testing.T) {
		budget := bufferTestBudget(t, 117, time.Hour)
		p := &Proxy{buffers: budget}
		source := &ownedBodySource{remaining: 18, budget: budget}
		r := httptest.NewRequest(http.MethodPost, "/api/generate", nil)
		r.Body, r.ContentLength = source, 17
		r, owner := p.ownInferenceBuffers(r)
		defer owner.close()
		_, _, err := bufferBodyAndModel(httptest.NewRecorder(), r, 100)
		bufferAssertLimit(t, err) // 17 old + 1 pending + 100 replacement exceeds 117.
		if source.reads != 2 || source.remaining != 0 || !source.closed {
			t.Fatalf("replacement refusal: reads=%d remaining=%d closed=%v", source.reads, source.remaining, source.closed)
		}
		bufferAssertTotals(t, budget, 0, 0, 0)
	})
	t.Run("known length probe read failure cannot return body", func(t *testing.T) {
		budget := bufferTestBudget(t, 192<<20, 0)
		p := &Proxy{buffers: budget}
		failure := errors.New("known-length probe failed")
		source := &ownedBodySource{remaining: 17, budget: budget, failure: failure}
		r := httptest.NewRequest(http.MethodPost, "/api/generate", nil)
		r.Body, r.ContentLength = source, 17
		r, owner := p.ownInferenceBuffers(r)
		defer owner.close()
		body, _, err := bufferBodyAndModel(httptest.NewRecorder(), r, 100)
		if body != nil || !errors.Is(err, failure) || source.reads != 2 || !source.closed {
			t.Fatalf("probe failure: body=%d reads=%d closed=%v err=%v", len(body), source.reads, source.closed, err)
		}
		bufferAssertTotals(t, budget, 0, 0, 0)
	})
	t.Run("model decoding uses one charged retained representation", func(t *testing.T) {
		for _, tc := range []struct{ name, raw, want string }{
			{"escapes", `{"model":"m\\\"\u754c"}`, "m\\\"界"},
			{"surrogate pair", `{"model":"\uD83D\uDE00"}`, "😀"},
			{"unpaired surrogate", `{"model":"\uD800x\uDC00"}`, "�x�"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				budget := bufferTestBudget(t, 192<<20, 0)
				p := &Proxy{buffers: budget}
				r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(tc.raw)))
				r.ContentLength = -1 // This case exercises unknown-length decoding storage.
				defer owner.close()
				body, model, err := bufferBodyAndModel(httptest.NewRecorder(), r, 64<<20)
				if err != nil || model != tc.want || !bytes.Equal(body, []byte(tc.raw)) || cap(body) != 32<<10 {
					t.Fatalf("model=%q err=%v, want %q", model, err, tc.want)
				}
				// Unknown-length body backing + parser scratch + exactly one
				// retained decoded string, including its builder allowance.
				decodedCharge := int64(len(tc.want)) + 8192
				bufferAssertTotals(t, budget, (32<<10)+(96<<10)+decodedCharge, 3, 0)
				owner.close()
				bufferAssertTotals(t, budget, 0, 0, 0)
			})
		}
	})
	t.Run("invalid model decoding retains only body and parser charges", func(t *testing.T) {
		for _, tc := range []struct{ name, raw string }{
			{"raw invalid UTF-8", "{\"model\":\"\xff\xfe\"}"},
			{"escaped C0", `{"model":"m\n\t\\\""}`},
		} {
			t.Run(tc.name, func(t *testing.T) {
				budget := bufferTestBudget(t, 192<<20, 0)
				p := &Proxy{buffers: budget}
				r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(tc.raw)))
				r.ContentLength = -1
				defer owner.close()
				body, model, err := bufferBodyAndModel(httptest.NewRecorder(), r, 64<<20)
				var identifier *requestIdentifierError
				if !errors.As(err, &identifier) || bufferRequestStatus(err) != http.StatusBadRequest || body != nil || model != "" {
					t.Fatalf("invalid model: body=%d model=%q err=%v", len(body), model, err)
				}
				bufferAssertTotals(t, budget, (32<<10)+(96<<10), 2, 0)
				owner.close()
				bufferAssertTotals(t, budget, 0, 0, 0)
			})
		}
	})
	t.Run("maximal oversized model is a typed identifier rejection", func(t *testing.T) {
		budget := bufferTestBudget(t, 192<<20, 0)
		p := &Proxy{buffers: budget}
		r := httptest.NewRequest(http.MethodPost, "/api/generate", nil)
		source := &ownedBodySource{remaining: (64 << 20) - 12, budget: budget}
		r.Body = io.NopCloser(io.MultiReader(strings.NewReader(`{"model":"`), source, strings.NewReader(`"}`)))
		r.ContentLength = -1
		r, owner := p.ownInferenceBuffers(r)
		defer owner.close()
		body, model, err := bufferBodyAndModel(httptest.NewRecorder(), r, 64<<20)
		var identifier *requestIdentifierError
		if !errors.As(err, &identifier) || bufferRequestStatus(err) != http.StatusBadRequest || body != nil || model != "" {
			t.Fatalf("oversized identifier: body=%d model=%d err=%v", len(body), len(model), err)
		}
		bufferAssertTotals(t, budget, (64<<20)+(96<<10), 2, 0)
	})
	maximalJSON := func(name string, known bool) {
		t.Run(name, func(t *testing.T) {
			budget := bufferTestBudget(t, 192<<20, 0)
			p := &Proxy{buffers: budget}
			const prefix, suffix = `{"model":"m","input":"`, `"}`
			source := &ownedBodySource{remaining: (64 << 20) - int64(len(prefix)+len(suffix)), budget: budget}
			r := httptest.NewRequest(http.MethodPost, "/api/generate", nil)
			r.Body = io.NopCloser(io.MultiReader(strings.NewReader(prefix), source, strings.NewReader(suffix)))
			r.ContentLength = -1
			if known {
				r.ContentLength = 64 << 20
			}
			r, owner := p.ownInferenceBuffers(r)
			defer owner.close()
			body, model, err := bufferBodyAndModel(httptest.NewRecorder(), r, 64<<20)
			if err != nil || len(body) != 64<<20 || cap(body) != 64<<20 || model != "m" || source.remaining != 0 {
				t.Fatalf("maximal legal JSON: bytes=%d capacity=%d model=%q remaining=%d err=%v", len(body), cap(body), model, source.remaining, err)
			}
			bufferAssertTotals(t, budget, (64<<20)+(96<<10)+1, 3, 0)
			owner.close()
			bufferAssertTotals(t, budget, 0, 0, 0)
		})
	}
	maximalJSON("known maximal legal JSON preserves body acceptance", true)
	maximalJSON("chunked maximal legal JSON preserves body acceptance", false)
	t.Run("bounded scanner preserves JSON syntax", func(t *testing.T) {
		for _, tc := range []struct {
			name, raw string
			valid     bool
		}{
			{"empty", "", false}, {"number", "-1.25e+9", true}, {"leading zero", "01", false}, {"missing fraction", "1.", false}, {"missing exponent", "1e+", false},
			{"object", ` {"model":"m","a":[true,false,null,{"x":"\uD83D\uDE00"}]} `, true},
			{"object comma", `{"a":1,}`, false}, {"array comma", `[1,]`, false}, {"missing colon", `{"a" 1}`, false}, {"extra root", "{}[]", false}, {"bad escape", `"\x"`, false}, {"bad unicode", `"\u000g"`, false},
			{"exact nesting", strings.Repeat("[", 10000) + "0" + strings.Repeat("]", 10000), true},
			{"excess nesting", strings.Repeat("[", 10001) + "0" + strings.Repeat("]", 10001), false},
		} {
			t.Run(tc.name, func(t *testing.T) {
				if got := ownedJSONValid([]byte(tc.raw)); got != tc.valid {
					t.Fatalf("valid=%v, want %v", got, tc.valid)
				}
			})
		}
	})
	readMaximal := func(name string, known bool) {
		t.Run(name, func(t *testing.T) {
			budget := bufferTestBudget(t, 192<<20, 0)
			p := &Proxy{buffers: budget}
			r := httptest.NewRequest(http.MethodPost, "/api/generate", nil)
			source := &ownedBodySource{remaining: 64 << 20, budget: budget}
			r.Body = source
			r.ContentLength = -1
			if known {
				r.ContentLength = 64 << 20
			}
			r, owner := p.ownInferenceBuffers(r)
			defer owner.close()
			body, model, err := bufferBodyAndModel(httptest.NewRecorder(), r, 64<<20)
			if err != nil || len(body) != 64<<20 || cap(body) != 64<<20 || model != "" {
				t.Fatalf("maximal body: length=%d capacity=%d model=%q err=%v", len(body), cap(body), model, err)
			}
			if !source.probe || !source.closed || source.peak < 64<<20 || source.peak > 192<<20 {
				t.Fatalf("boundary read: probe=%v closed=%v peak=%d", source.probe, source.closed, source.peak)
			}
			bufferAssertTotals(t, budget, (64<<20)+(96<<10), 2, 0)
		})
	}
	readMaximal("known exact 64 MiB at 192 MiB root", true)
	readMaximal("chunked exact 64 MiB at 192 MiB root", false)
	t.Run("chunked growth charges old and replacement concurrently", func(t *testing.T) {
		budget := bufferTestBudget(t, (96<<20)-1, time.Hour)
		p := &Proxy{buffers: budget}
		r := httptest.NewRequest(http.MethodPost, "/api/generate", nil)
		source := &ownedBodySource{remaining: 64 << 20, budget: budget}
		r.Body, r.ContentLength = source, -1
		r, owner := p.ownInferenceBuffers(r)
		defer owner.close()
		_, _, err := bufferBodyAndModel(httptest.NewRecorder(), r, 64<<20)
		bufferAssertLimit(t, err)
		if source.remaining != 32<<20 || !source.closed {
			t.Fatalf("replacement read before admission: remaining=%d closed=%v", source.remaining, source.closed)
		}
		bufferAssertTotals(t, budget, 0, 0, 0)
	})
	t.Run("custom logical limit charges separate overflow probe", func(t *testing.T) {
		budget := bufferTestBudget(t, 192<<20, 0)
		p := &Proxy{buffers: budget}
		r := httptest.NewRequest(http.MethodPost, "/api/generate", nil)
		source := &ownedBodySource{remaining: 41, budget: budget}
		r.Body, r.ContentLength = source, -1
		r, owner := p.ownInferenceBuffers(r)
		defer owner.close()
		_, _, err := bufferBodyAndModel(httptest.NewRecorder(), r, 40)
		bufferAssertLimit(t, err)
		if source.peak != 41 || !source.closed {
			t.Fatalf("overflow storage: peak=%d closed=%v", source.peak, source.closed)
		}
		bufferAssertTotals(t, budget, 0, 0, 0)
	})
	t.Run("known logical overflow refuses without reading", func(t *testing.T) {
		budget := bufferTestBudget(t, 192<<20, time.Hour)
		p := &Proxy{buffers: budget}
		source := &ownedBodySource{remaining: 41, budget: budget}
		r := httptest.NewRequest(http.MethodPost, "/api/generate", nil)
		r.Body, r.ContentLength = source, 41
		r, owner := p.ownInferenceBuffers(r)
		defer owner.close()
		_, _, err := bufferBodyAndModel(httptest.NewRecorder(), r, 40)
		bufferAssertLimit(t, err)
		if source.reads != 0 || !source.closed {
			t.Fatalf("refusal effects: reads=%d closed=%v", source.reads, source.closed)
		}
		bufferAssertTotals(t, budget, 0, 0, 0)
	})
	t.Run("read failure cannot return a truncated body", func(t *testing.T) {
		budget := bufferTestBudget(t, 192<<20, 0)
		p := &Proxy{buffers: budget}
		r := httptest.NewRequest(http.MethodPost, "/api/generate", nil)
		failure := errors.New("read failed")
		r.Body = &ownedBodySource{remaining: 17, budget: budget, failure: failure}
		r.ContentLength = -1
		r, owner := p.ownInferenceBuffers(r)
		defer owner.close()
		body, _, err := bufferBodyAndModel(httptest.NewRecorder(), r, 100)
		if !errors.Is(err, failure) || body != nil || bufferRequestStatus(err) != http.StatusBadRequest {
			t.Fatalf("read failure: body=%v err=%v", body, err)
		}
		bufferAssertTotals(t, budget, 0, 0, 0)
	})
	t.Run("own replacement impossibility refuses before waiting", func(t *testing.T) {
		budget := bufferTestBudget(t, 100, time.Hour)
		scope := bufferMustScope(t, budget, 1000)
		lease, err := scope.acquire(context.Background(), 60)
		if err != nil {
			t.Fatal(err)
		}
		defer lease.close()
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		_, err = lease.grow(ctx, 50)
		bufferAssertLimit(t, err)
		bufferAssertTotals(t, budget, 60, 1, 0)
	})
	t.Run("contention waits and cancellation removes proposal", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			budget := bufferTestBudget(t, 128<<10, time.Hour)
			hold := bufferMustAcquire(t, budget, 128<<10)
			p := &Proxy{buffers: budget}
			r := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"m"}`))
			ctx, cancel := context.WithCancel(r.Context())
			defer cancel()
			r, owner := p.ownInferenceBuffers(r.WithContext(ctx))
			defer owner.close()
			done := make(chan error, 1)
			go func() { _, _, err := bufferBodyAndModel(httptest.NewRecorder(), r, 100); done <- err }()
			synctest.Wait()
			bufferAssertTotals(t, budget, 128<<10, 1, 1)
			cancel()
			synctest.Wait()
			bufferAssertBusy(t, <-done)
			bufferAssertTotals(t, budget, 128<<10, 1, 0)
			hold.close()
		})
	})
	t.Run("contention waits then succeeds", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			budget := bufferTestBudget(t, 128<<10, time.Hour)
			hold := bufferMustAcquire(t, budget, 128<<10)
			p := &Proxy{buffers: budget}
			r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"m"}`)))
			r.ContentLength = -1 // Preserve this contention case's unknown-length growth.
			defer owner.close()
			done := make(chan error, 1)
			go func() {
				body, model, err := bufferBodyAndModel(httptest.NewRecorder(), r, 100)
				if err == nil && (string(body) != `{"model":"m"}` || model != "m") {
					err = errors.New("body or model changed")
				}
				done <- err
			}()
			synctest.Wait()
			bufferAssertTotals(t, budget, 128<<10, 1, 1)
			hold.close()
			synctest.Wait()
			if err := <-done; err != nil {
				t.Fatal(err)
			}
			bufferAssertTotals(t, budget, 101+(96<<10), 3, 0)
		})
	})
	t.Run("borrower retains replaced storage", func(t *testing.T) {
		budget := bufferTestBudget(t, 100, 0)
		scope := bufferMustScope(t, budget, 1000)
		lease, err := scope.acquire(context.Background(), 30)
		if err != nil {
			t.Fatal(err)
		}
		borrower, err := lease.borrow()
		if err != nil {
			t.Fatal(err)
		}
		defer borrower.close()
		replacement, err := lease.grow(context.Background(), 50)
		if err != nil {
			t.Fatal(err)
		}
		defer replacement.close()
		lease.close()
		bufferAssertTotals(t, budget, 80, 2, 0)
		borrower.close()
		bufferAssertTotals(t, budget, 50, 1, 0)
	})
	t.Run("stream frame storage cannot bypass inference root", func(t *testing.T) {
		budget := bufferTestBudget(t, 128<<10, 0)
		p := &Proxy{buffers: budget}
		r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"m"}`)))
		defer owner.close()
		if _, _, err := bufferBodyAndModel(httptest.NewRecorder(), r, 64<<20); err != nil {
			t.Fatal(err)
		}
		hold := bufferMustAcquire(t, budget, (32<<10)-14)
		upstream := io.NopCloser(bytes.NewReader([]byte("data: [DONE]\n\n")))
		reader, err := newStreamFrameReader(streamFrameConfig{Protocol: streamProtocolSSE, Upstream: upstream, Parent: r.Context()})
		if err != nil {
			t.Fatal(err)
		}
		defer reader.Close()
		out, err := io.ReadAll(reader)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(out, wantSSEInterruption(streamInterruptReasonBudget)) {
			t.Fatalf("unexpected exhaustion output: %q", out)
		}
		assertOutcome(t, reader.Result(), streamOutcomeInterrupted, streamInterruptReasonBudget)
		bufferAssertTotals(t, budget, 128<<10, 4, 0)
		hold.close()
		owner.close()
		// The interruption reader still borrows the actual parser allowance.
		bufferAssertTotals(t, budget, 96<<10, 1, 0)
		if err := reader.Close(); err != nil {
			t.Fatal(err)
		}
		bufferAssertTotals(t, budget, 0, 0, 0)
	})
	t.Run("stream bootstrap account alone grants no allowance", func(t *testing.T) {
		budget := bufferTestBudget(t, 16<<10, 0)
		p := &Proxy{buffers: budget}
		r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, "/api/generate", nil))
		defer owner.close()
		hold := bufferMustAcquire(t, budget, 16<<10)
		defer hold.close()
		reader, err := newStreamFrameReader(streamFrameConfig{Protocol: streamProtocolSSE, Upstream: io.NopCloser(strings.NewReader("data: [DONE]\n\n")), Parent: r.Context()})
		if reader != nil || !errors.Is(err, errStreamFrameBudget) {
			t.Fatalf("unadmitted bootstrap: reader=%p err=%v", reader, err)
		}
		bufferAssertTotals(t, budget, 16<<10, 1, 0)
	})
	t.Run("request frame scopes share the root and retain individual limits", func(t *testing.T) {
		budget := bufferTestBudget(t, 192<<20, 0)
		p := &Proxy{buffers: budget}
		newAccount := func(p *Proxy) (*ownedInferenceBuffers, *frameBudget) {
			t.Helper()
			r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, "/api/generate", nil))
			r.Body = nil
			if _, _, err := bufferBodyAndModel(httptest.NewRecorder(), r, 64<<20); err != nil {
				t.Fatal(err)
			}
			return owner, ownedFrameBudget(r.Context())
		}
		first, a := newAccount(p)
		defer first.close()
		second, b := newAccount(p)
		defer second.close()
		if !a.reserve(40 << 20) {
			t.Fatal("first stream reservation refused")
		}
		defer a.release(40 << 20)
		if !b.reserve(30 << 20) {
			t.Fatal("independent request refused despite spare root capacity")
		}
		defer b.release(30 << 20)
		if !b.reserve(34 << 20) {
			t.Fatal("individual frame scope exact boundary refused")
		}
		defer b.release(34 << 20)
		if b.reserve(1) {
			b.release(1)
			t.Fatal("request frame scope exceeded 64 MiB despite spare root capacity")
		}
		bufferAssertTotals(t, budget, (104<<20)+(192<<10), 5, 0)
		hold := bufferMustAcquire(t, budget, (88<<20)-(192<<10))
		defer hold.close()
		bufferAssertTotals(t, budget, 192<<20, 6, 0)
		if a.reserve(1) {
			a.release(1)
			t.Fatal("stream grant bypassed saturated root")
		}
		otherRoot := bufferTestBudget(t, 192<<20, 0)
		third, c := newAccount(&Proxy{buffers: otherRoot})
		defer third.close()
		if !c.reserve(64 << 20) {
			t.Fatal("separate configured root inherited another root's saturation")
		}
		defer c.release(64 << 20)
		bufferAssertTotals(t, otherRoot, (64<<20)+(96<<10), 2, 0)
	})
}

func TestOwnedInferenceBuffersBodyEdits(t *testing.T) {
	t.Run("admission and alias errors retain canonical charged fields", func(t *testing.T) {
		budget := bufferTestBudget(t, 512, time.Hour)
		p := &Proxy{buffers: budget}
		r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, "/api/generate", nil))
		defer owner.close()
		recorder := httptest.NewRecorder()
		w := &inferenceErrorWriter{ResponseWriter: recorder, ctx: r.Context()}
		writeAdmissionRejection(w, nodepolicy.RejectPaused)
		want := `{"code":"admission","error":"this node is not accepting the request (paused)","reason":"paused"}`
		if recorder.Code != http.StatusServiceUnavailable || recorder.Body.String() != want {
			t.Fatalf("admission status=%d body=%q, want %q", recorder.Code, recorder.Body.String(), want)
		}
		bufferAssertTotals(t, budget, 0, 0, 0)
		recorder = httptest.NewRecorder()
		w = &inferenceErrorWriter{ResponseWriter: recorder, ctx: r.Context()}
		writeAliasRejection(w, " \tweak-api\n", "<&\xff")
		want = `{"error":"hybrid routing alias is not available on this request","code":"feature_unavailable","reason":"\u003c\u0026\ufffd","model":"weak-api"}`
		if recorder.Code != http.StatusBadRequest || recorder.Body.String() != want {
			t.Fatalf("alias status=%d body=%q, want %q", recorder.Code, recorder.Body.String(), want)
		}
		bufferAssertTotals(t, budget, 0, 0, 0)
		hold := bufferMustAcquire(t, budget, 512)
		defer hold.close()
		recorder = httptest.NewRecorder()
		w = &inferenceErrorWriter{ResponseWriter: recorder, ctx: r.Context()}
		writeAliasRejection(w, "weak-api", aliasReasonNotInferenceReady)
		var response struct{ Code string }
		if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
			t.Fatal(err)
		}
		if recorder.Code != http.StatusServiceUnavailable || response.Code != "inference-buffer" {
			t.Fatalf("contended alias: status=%d code=%q", recorder.Code, response.Code)
		}
		bufferAssertTotals(t, budget, 512, 1, 0)
	})
	t.Run("nonpooled proxy errors preserve canonical strings and needs", func(t *testing.T) {
		budget := bufferTestBudget(t, 1024, time.Hour)
		p := &Proxy{buffers: budget}
		r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, "/api/generate", nil))
		defer owner.close()
		recorder := httptest.NewRecorder()
		w := &inferenceErrorWriter{ResponseWriter: recorder, ctx: r.Context()}
		writeProxyError(w, http.StatusNotFound, []string{"tier ", "<&\xff\u2028"}, []string{"vision", "tools"}, true)
		want := `{"error":"tier \u003c\u0026\ufffd\u2028","needs":["vision","tools"]}`
		if recorder.Code != http.StatusNotFound || recorder.Body.String() != want {
			t.Fatalf("status=%d body=%q, want %q", recorder.Code, recorder.Body.String(), want)
		}
		bufferAssertTotals(t, budget, 0, 0, 0)
		recorder = httptest.NewRecorder()
		w = &inferenceErrorWriter{ResponseWriter: recorder, ctx: r.Context()}
		writeProxyError(w, http.StatusNotFound, []string{"none"}, nil, true)
		if recorder.Body.String() != `{"error":"none","needs":null}` {
			t.Fatalf("nil needs body=%q", recorder.Body.String())
		}
	})
	t.Run("local errors use charged nonpooled serialization", func(t *testing.T) {
		budget := bufferTestBudget(t, 256, time.Hour)
		p := &Proxy{buffers: budget}
		r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, "/api/generate", nil))
		defer owner.close()
		recorder := httptest.NewRecorder()
		w := &inferenceErrorWriter{ResponseWriter: recorder, ctx: r.Context()}
		writeIngressError(w, http.StatusBadRequest, "bad", "<>&\u2028")
		if recorder.Code != http.StatusBadRequest || recorder.Body.String() != `{"code":"bad","error":"\u003c\u003e\u0026\u2028"}` {
			t.Fatalf("error response: status=%d body=%q", recorder.Code, recorder.Body.String())
		}
		bufferAssertTotals(t, budget, 0, 0, 0)
		hold := bufferMustAcquire(t, budget, 256)
		recorder = httptest.NewRecorder()
		w = &inferenceErrorWriter{ResponseWriter: recorder, ctx: r.Context()}
		writeIngressError(w, http.StatusBadRequest, "bad", "read failed")
		var result struct{ Code, Error string }
		if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if recorder.Code != http.StatusServiceUnavailable || result.Code != "inference-buffer" {
			t.Fatalf("contended error response: status=%d value=%+v", recorder.Code, result)
		}
		bufferAssertTotals(t, budget, 256, 1, 0)
		hold.close()
		recorder = httptest.NewRecorder()
		w = &inferenceErrorWriter{ResponseWriter: recorder, ctx: r.Context()}
		writeIngressError(w, http.StatusBadRequest, "bad", strings.Repeat("x", 300))
		if recorder.Code != http.StatusRequestEntityTooLarge {
			t.Fatalf("impossible error status=%d", recorder.Code)
		}
		bufferAssertTotals(t, budget, 0, 0, 0)
	})
	t.Run("rewrite and recursive defaults allocate one final preserved body", func(t *testing.T) {
		budget := bufferTestBudget(t, 192<<20, 0)
		p := &Proxy{buffers: budget}
		input := " { \"model\" : \"weak\", \"n\":1.50e2,\"options\":{\"temperature\":0} }\n"
		defaults := []byte(`{"options":{"temperature":1,"num_ctx":4096},"keep_alive":"10m"}`)
		r := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(input))
		r.ContentLength = int64(len(input))
		r, owner := p.ownInferenceBuffers(r)
		defer owner.close()
		body, _, err := bufferBodyAndModel(httptest.NewRecorder(), r, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		base := int64(len(input) + 4 + (96 << 10))
		bufferAssertTotals(t, budget, base, 3, 0)
		out, release, err := transformBody(r.Context(), body, "m<&", true, defaults)
		if err != nil {
			t.Fatal(err)
		}
		defer release()
		want := " { \"model\" : \"m\\u003c\\u0026\", \"n\":1.50e2,\"options\":{\"temperature\":0,\"num_ctx\":4096} ,\"keep_alive\":\"10m\"}\n"
		if string(out) != want || cap(out) != len(want) {
			t.Fatalf("effective bytes=%q capacity=%d, want %q", out, cap(out), want)
		}
		bufferAssertTotals(t, budget, base+int64(len(want))+256, 4, 0)
		release()
		bufferAssertTotals(t, budget, base, 3, 0)
		owner.close()
		bufferAssertTotals(t, budget, 0, 0, 0)
	})
	t.Run("deep recursive merge has no intermediate serialized bodies", func(t *testing.T) {
		budget := bufferTestBudget(t, 192<<20, 0)
		p := &Proxy{buffers: budget}
		prefix, suffix := strings.Repeat(`{"x":`, 512), strings.Repeat("}", 512)
		input := prefix + `{}` + suffix
		defaults := []byte(prefix + `{"n":1}` + suffix)
		want := prefix + `{"n":1}` + suffix
		r := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(input))
		r.ContentLength = int64(len(input))
		r, owner := p.ownInferenceBuffers(r)
		defer owner.close()
		body, _, err := bufferBodyAndModel(httptest.NewRecorder(), r, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		out, release, err := transformBody(r.Context(), body, "", false, defaults)
		if err != nil {
			t.Fatal(err)
		}
		defer release()
		if string(out) != want || cap(out) != len(want) {
			t.Fatalf("deep merge: length=%d capacity=%d, want %d", len(out), cap(out), len(want))
		}
		bufferAssertTotals(t, budget, int64(len(input)+len(want)+(96<<10))+256, 3, 0)
		release()
		owner.close()
		bufferAssertTotals(t, budget, 0, 0, 0)
	})
	t.Run("many tiny members pay decoded metadata before allocation", func(t *testing.T) {
		var input strings.Builder
		input.WriteByte('{')
		for i := 0; i < 30000; i++ {
			if i > 0 {
				input.WriteByte(',')
			}
			input.WriteByte('"')
			input.WriteString(strconv.Itoa(i))
			input.WriteString(`":0`)
		}
		input.WriteByte('}')
		budget := bufferTestBudget(t, 1<<20, time.Hour)
		p := &Proxy{buffers: budget}
		r := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(input.String()))
		r.ContentLength = int64(input.Len())
		r, owner := p.ownInferenceBuffers(r)
		defer owner.close()
		body, _, err := bufferBodyAndModel(httptest.NewRecorder(), r, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		out, release, err := transformBody(r.Context(), body, "", false, []byte(`{"missing":1}`))
		if release != nil {
			defer release()
		}
		bufferAssertLimit(t, err)
		if out != nil {
			t.Fatal("impossible member metadata produced a send body")
		}
		bufferAssertTotals(t, budget, int64(input.Len()+(96<<10)), 2, 0)
	})
	t.Run("capability views preserve repeated typed-field behavior", func(t *testing.T) {
		for _, tc := range []struct {
			raw                string
			structured, vision bool
			extra              int
		}{
			{`{"response_format":{"type":"json_schema"},"response_format":{},"options":{"num_predict":7},"options":{}}`, true, false, 7},
			{`{"response_format":{"type":"json_schema"},"response_format":null,"max_tokens":5,"max_tokens":"bad","max_completion_tokens":9}`, false, false, 5},
			{`{"messages":[{"content":[{"type":"image_url"}]}],"messages":false}`, false, true, 0},
			{`{"messages":[{"content":[{"type":"image_url"},4]}]}`, false, false, 0},
		} {
			needs := parseRequestNeeds([]byte(tc.raw), "/v1/chat/completions")
			if slices.Contains(needs.caps, nodepolicy.CapStructured) != tc.structured || slices.Contains(needs.caps, nodepolicy.CapVision) != tc.vision || needs.contextTokens != len(tc.raw)/4+tc.extra {
				t.Fatalf("capability result=%+v for %s", needs, tc.raw)
			}
		}
	})
}

func TestOwnedInferenceBuffersAliasBorrowedScan(t *testing.T) {
	for _, tc := range []struct{ name, target, body, want string }{
		{"folded escaped key and value", "/api/chat", `{"Na\u004de":"  weak\u002dapi\u2003"}`, "weak-api"},
		{"completed prefix before malformed suffix", "/api/chat", `{"name":"weak-api","broken":`, "weak-api"},
		{"completed string before invalid separator", "/api/chat", `{"name":"weak-api"invalid`, "weak-api"},
		{"malformed value before later alias", "/api/chat", `{"ignored":[,],"name":"weak-api"}`, ""},
		{"unfinished alias string", "/api/chat", `{"name":"weak-api`, ""},
		{"null and bare aliases remain eligible", "/api/chat", `{"model":null,"name":"weak"}`, ""},
		{"query percent and Unicode whitespace", "/props?other=%xx&%6dodel=%E2%80%83weak%2dapi+", `{}`, "weak-api"},
		{"malformed query alias skipped", "/props?model=weak-api%xx&model=medium-local", `{}`, "medium-local"},
		{"semicolon query field skipped", "/props?model=weak-api;ignored&model=strong-api", `{}`, "strong-api"},
		{"query model key remains case sensitive", "/props?MODEL=weak-api", `{}`, ""},
		{"model detail path", "/v1/models/weak-local", `{}`, "weak-local"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, tc.target, nil)
			alias, reason, rejected := requestAliasRejection(r, []byte(tc.body), "", true)
			if alias != tc.want || rejected != (tc.want != "") || rejected && reason != aliasReasonNotInferenceReady {
				t.Fatalf("alias=%q reason=%q rejected=%v, want %q", alias, reason, rejected, tc.want)
			}
		})
	}
	t.Run("large ignored members allocate no retained alias representation", func(t *testing.T) {
		test := func(name, encoded, want string) {
			t.Run(name, func(t *testing.T) {
				body := []byte(`{"ignored":"` + strings.Repeat("x", 1<<20) + `","Na\u004de":"` + strings.Repeat(" ", 1<<20) + encoded + `\u2003"}`)
				alias := jsonModelAlias(body)
				assertCanonical := func() {
					t.Helper()
					// Canonical literal identity excludes both a heap copy and a
					// view retaining any request storage through the result.
					if alias != want || unsafe.StringData(alias) != unsafe.StringData(want) {
						t.Fatalf("borrowed alias=%q storage=%p, want canonical %q storage=%p", alias, unsafe.StringData(alias), want, unsafe.StringData(want))
					}
				}
				assertCanonical()
				clear(body)
				body = nil
				assertCanonical()
			})
		}
		test("plain weak api", "weak-api", "weak-api")
		test("escaped weak api", `weak\u002dapi`, "weak-api")
		test("plain weak local", "weak-local", "weak-local")
		test("escaped weak local", `weak\u002dlocal`, "weak-local")
		test("plain medium api", "medium-api", "medium-api")
		test("escaped medium api", `medium\u002dapi`, "medium-api")
		test("plain medium local", "medium-local", "medium-local")
		test("escaped medium local", `medium\u002dlocal`, "medium-local")
		test("plain strong api", "strong-api", "strong-api")
		test("escaped strong api", `strong\u002dapi`, "strong-api")
		test("plain strong local", "strong-local", "strong-local")
		test("escaped strong local", `strong\u002dlocal`, "strong-local")
	})
}

func TestOwnedInferenceBuffersIdentifierLimit(t *testing.T) {
	for _, tc := range []struct {
		name, value string
		accept      bool
	}{
		{"exact ASCII", strings.Repeat("m", 256), true},
		{"ASCII overflow", strings.Repeat("m", 257), false},
		{"escaped exact", strings.Repeat(`\u006d`, 256), true},
		{"escaped overflow", strings.Repeat(`\u006d`, 257), false},
		{"multibyte exact", strings.Repeat(`\u00e9`, 128), true},
		{"multibyte overflow", strings.Repeat(`\u00e9`, 128) + "m", false},
		{"padding counts before trim", strings.Repeat(" ", 256) + "m", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			budget := bufferTestBudget(t, 192<<20, 0)
			p := &Proxy{buffers: budget}
			input := `{"model":"` + tc.value + `"}`
			r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(input)))
			defer owner.close()
			body, model, err := bufferBodyAndModel(httptest.NewRecorder(), r, 64<<20)
			if tc.accept {
				var expected struct{ Model string }
				if err := json.Unmarshal([]byte(input), &expected); err != nil {
					t.Fatal(err)
				}
				if err != nil || string(body) != input || model != expected.Model || len(model) != 256 {
					t.Fatalf("accepted boundary: bytes=%d model bytes=%d err=%v", len(body), len(model), err)
				}
			} else {
				var identifier *requestIdentifierError
				if !errors.As(err, &identifier) || bufferRequestStatus(err) != http.StatusBadRequest || body != nil || model != "" {
					t.Fatalf("identifier refusal: body=%d model=%d err=%v", len(body), len(model), err)
				}
				// No decoded oversized identifier was allocated or retained.
				bufferAssertTotals(t, budget, int64(len(input))+(96<<10), 2, 0)
			}
			owner.close()
			bufferAssertTotals(t, budget, 0, 0, 0)
		})
	}
	t.Run("handler and ingress refuse before effects", func(t *testing.T) {
		pol := ollamaTierPolicy(nil)
		fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"a": {"m"}}, nil)
		backend := newRecordingEngine(t, http.StatusOK)
		f := fx.p.soleFacade()
		setBackend(t, f, backend.URL, true)
		for _, ingress := range []bool{false, true} {
			rec := httptest.NewRecorder()
			r := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"`+strings.Repeat(" ", 256)+`m"}`))
			if ingress {
				f.serveIngress(rec, r, "peer")
			} else {
				f.handleHTTP(rec, r)
			}
			if rec.Code != http.StatusBadRequest || backend.hits() != 0 || fx.eng["a"].hits() != 0 || len(notificationsOf(t, fx.w, workloadSubmittedMethod)) != 0 {
				t.Fatalf("identifier effects: ingress=%v status=%d local=%d remote=%d", ingress, rec.Code, backend.hits(), fx.eng["a"].hits())
			}
			ownedAssertHandlerDrained(t, fx.p, nil)
		}
	})
}

func ownedAssertHandlerDrained(t *testing.T, p *Proxy, control *routingSpend) {
	t.Helper()
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
	p.identityMu.Lock()
	requests, associations := len(p.requests), len(p.workloadRequests)
	p.identityMu.Unlock()
	p.priorityMu.Lock()
	claims := len(p.priorityReservations)
	p.priorityMu.Unlock()
	p.admission.mu.Lock()
	active, queued, tickets, inflight := p.admission.active, p.admission.queued, len(p.admission.tickets), len(p.admission.inflight)
	p.admission.mu.Unlock()
	p.performance.mu.Lock()
	telemetry := p.performance.size
	p.performance.mu.Unlock()
	if requests != 0 || associations != 0 || claims != 0 || active != 0 || queued != 0 || tickets != 0 || inflight != 0 || telemetry != 0 {
		t.Fatalf("remaining owners: requests=%d associations=%d claims=%d active=%d queued=%d tickets=%d inflight=%d telemetry=%d", requests, associations, claims, active, queued, tickets, inflight, telemetry)
	}
	if control != nil {
		control.mu.Lock()
		done, attempt := control.done, control.active
		control.mu.Unlock()
		if !done || attempt != nil {
			t.Fatalf("controller retained an attempt: done=%v attempt=%p", done, attempt)
		}
	}
}

func TestOwnedInferenceBuffersFinishedModelStorage(t *testing.T) {
	t.Run("known resolved key owns small decoded model storage", func(t *testing.T) {
		budget := bufferTestBudget(t, 192<<20, 0)
		p := &Proxy{buffers: budget}
		input := strings.Repeat(" ", 255) + "m"
		r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"`+input+`"}`)))
		defer owner.close()
		_, model, err := bufferBodyAndModel(httptest.NewRecorder(), r, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		h := newAdmHarness(t)
		ticket := mustAdmit(t, h.admit("llamacpp", model, 0))
		ticket.release()
		owner.close()
		bufferAssertTotals(t, budget, 0, 0, 0)
		h.a.mu.Lock()
		defer h.a.mu.Unlock()
		if len(h.a.finishedAt) != 1 {
			t.Fatalf("known model timestamps=%d, want one", len(h.a.finishedAt))
		}
		for key := range h.a.finishedAt {
			if key.model != "m" || unsafe.StringData(key.model) == unsafe.StringData(strings.TrimSpace(model)) {
				t.Fatalf("finished key did not own the normalized identifier: %q", key.model)
			}
		}
	})
	t.Run("terminal ingress unknown names do not grow history", func(t *testing.T) {
		fx := newTierFixture(t, ollamaCase(t), ollamaTierPolicy(nil), map[string][]string{"a": {"known:latest"}}, nil)
		f := fx.p.soleFacade()
		backend := newRecordingEngine(t, http.StatusNotFound)
		setBackend(t, f, backend.URL, true)
		for i := range 4 {
			rec := httptest.NewRecorder()
			f.serveIngress(rec, httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(fmt.Sprintf(`{"model":"unknown-%d"}`, i))), "peer")
			if rec.Code != http.StatusNotFound || backend.hits() != i+1 {
				t.Fatalf("terminal forwarding: status=%d hits=%d", rec.Code, backend.hits())
			}
			fx.p.admission.mu.Lock()
			timestamps := len(fx.p.admission.finishedAt)
			fx.p.admission.mu.Unlock()
			if timestamps != 0 {
				t.Fatalf("unresolved model minted %d timestamps", timestamps)
			}
			ownedAssertHandlerDrained(t, fx.p, nil)
		}
		fx.p.admission.setResidency(map[string][]string{"ollama": {"known"}})
		rec := httptest.NewRecorder()
		f.serveIngress(rec, httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"known"}`)), "peer")
		fx.p.admission.mu.Lock()
		_, known := fx.p.admission.finishedAt[admKey{engine: "ollama", model: "known:latest"}]
		fx.p.admission.mu.Unlock()
		if rec.Code != http.StatusNotFound || !known {
			t.Fatalf("resident identity lost timestamp: status=%d known=%v", rec.Code, known)
		}
		fx.p.admission.setResidency(map[string][]string{})
		fx.p.admission.mu.Lock()
		remaining := len(fx.p.admission.finishedAt)
		fx.p.admission.mu.Unlock()
		if remaining != 0 {
			t.Fatalf("residency reconciliation retained %d timestamps", remaining)
		}
	})
	t.Run("unknown origin resolution creates no timestamp", func(t *testing.T) {
		fx := newTierFixture(t, ollamaCase(t), ollamaTierPolicy(nil), map[string][]string{"a": {"known:latest"}}, nil)
		rec := fx.do(t, "/api/generate", `{"model":"unknown"}`)
		fx.p.admission.mu.Lock()
		timestamps := len(fx.p.admission.finishedAt)
		fx.p.admission.mu.Unlock()
		if rec.Code != http.StatusBadGateway || timestamps != 0 || fx.eng["a"].hits() != 0 {
			t.Fatalf("unknown resolution: status=%d timestamps=%d hits=%d", rec.Code, timestamps, fx.eng["a"].hits())
		}
		ownedAssertHandlerDrained(t, fx.p, nil)
	})
	t.Run("resolved cold self model records completion", func(t *testing.T) {
		w := &recordingWriter{}
		p, f, _ := twoEngineProxy(t, w)
		backend := newRecordingEngine(t, http.StatusOK)
		f.discovery.AddManual(selfNode("self", f.profile.FacadePort, "known:latest"))
		setBackend(t, f, backend.URL, true)
		rec := httptest.NewRecorder()
		f.handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"known"}`)))
		p.admission.mu.Lock()
		_, known := p.admission.finishedAt[admKey{engine: "ollama", model: "known:latest"}]
		p.admission.mu.Unlock()
		if rec.Code != http.StatusOK || !known || backend.hits() != 1 {
			t.Fatalf("resolved cold model: status=%d timestamp=%v hits=%d", rec.Code, known, backend.hits())
		}
	})
}

func TestOwnedInferenceBuffersStrictIdentifierStorage(t *testing.T) {
	for _, tc := range []struct {
		name       string
		dispatched bool
		evidence   strictTerminalEvidence
		retained   bool
	}{
		{"not dispatched completion", false, strictNotDispatched, false},
		{"qualified completion", true, strictQualifiedComplete, false},
		{"uncertain execution", true, strictUncertain, true},
		{"unknown terminal evidence", true, strictTerminalEvidence("unknown"), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			budget := bufferTestBudget(t, 192<<20, 0)
			p := &Proxy{buffers: budget}
			input := strings.Repeat(" ", 248) + "m:latest"
			r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"`+input+`"}`)))
			defer owner.close()
			_, model, err := bufferBodyAndModel(httptest.NewRecorder(), r, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			if len(model) != 256 {
				t.Fatalf("decoded model bytes=%d, want boundary 256", len(model))
			}
			h := newStrictHarness(t)
			ticket, err := h.a.admitStrictNow(context.Background(), "ollama", model, strictRequest())
			if err != nil {
				t.Fatal(err)
			}
			owner.close()
			bufferAssertTotals(t, budget, 0, 0, 0)
			assertOwned := func(name, value string) {
				t.Helper()
				if value != "m:latest" || unsafe.StringData(value) == unsafe.StringData(strings.TrimSpace(model)) {
					t.Fatalf("strict %s retains decoded request backing or changed model: %q", name, value)
				}
			}
			h.a.mu.Lock()
			var admitted string
			for _, live := range h.a.tickets {
				admitted = live.key.model
			}
			h.a.mu.Unlock()
			assertOwned("ticket", admitted)
			if tc.dispatched {
				if err := ticket.BeforeStrictDispatch(); err != nil {
					t.Fatal(err)
				}
				h.a.mu.Lock()
				var marked string
				for _, marker := range h.a.strict.markers {
					marked = marker.Model
				}
				h.a.mu.Unlock()
				assertOwned("dispatch marker", marked)
			}
			ticket.Finish(tc.evidence)
			h.a.mu.Lock()
			active, tickets, markers, timestamps := h.a.active, len(h.a.tickets), len(h.a.strict.markers), len(h.a.finishedAt)
			var retained string
			for key := range h.a.finishedAt {
				retained = key.model
			}
			for _, marker := range h.a.strict.markers {
				retained = marker.Model
			}
			h.a.mu.Unlock()
			assertOwned("terminal identifier", retained)
			if tc.retained {
				if active != 1 || tickets != 0 || markers != 1 || timestamps != 0 {
					t.Fatalf("uncertain capacity changed: active=%d tickets=%d markers=%d timestamps=%d", active, tickets, markers, timestamps)
				}
			} else {
				if active != 0 || tickets != 0 || markers != 0 || timestamps != 1 {
					t.Fatalf("completion cleanup: active=%d tickets=%d markers=%d timestamps=%d", active, tickets, markers, timestamps)
				}
				h.a.setResidency(map[string][]string{})
				h.a.mu.Lock()
				timestamps = len(h.a.finishedAt)
				h.a.mu.Unlock()
				if timestamps != 0 {
					t.Fatalf("residency cleanup retained %d strict timestamps", timestamps)
				}
			}
		})
	}
}

// Each fixture requests one choice. Keep response grants distinct from the
// fingerprint scanner/storage and the request's retained parser allowance.
const bufferFixtureResponseBytes int64 = streamFrameMetadataBytes + 1 + streamFrameReadChunk + streamFrameNumericBytes + 3*streamFrameMaxBytes

func bufferFixtureRemotePreparationBytes(t *testing.T, body string) int64 {
	t.Helper()
	request, err := parseOriginStreamRequest("/api/generate", []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if request.disabled || request.protocol == streamProtocolUnknown {
		return 0
	}
	// The origin observes peer framing with syntax storage only; it does not
	// fingerprint remote execution. Native NDJSON has no SSE payload grant.
	return streamFrameAllowance(request.protocol, int64(request.choices), streamFrameSyntaxBytes)
}

func TestOwnedInferenceBuffersHandlerPreparationCancellation(t *testing.T) {
	run := func(name string, later, disconnect bool) {
		t.Run(name, func(t *testing.T) {
			pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "small:1b"}}})
			fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"a": {"small:1b"}, "b": {"small:1b"}}, nil)
			capacity := int64(256<<10) + registryNextTestBytes(fx.p)
			if later {
				capacity += bufferFixtureRemotePreparationBytes(t, `{"model":"small:1b"}`)
			}
			fx.p.buffers = bufferTestBudget(t, capacity, time.Hour)
			type heldResult struct {
				lease *bufferLease
				err   error
			}
			held := make(chan heldResult, 1)
			var hold *bufferLease
			var once sync.Once
			if later {
				first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					once.Do(func() {
						lease, err := fx.p.buffers.acquire(r.Context(), 128<<10)
						// Remove startup-only headroom after the first dispatch. Its
						// prepaid grants drain; the retry's transform must still wait.
						if err == nil {
							err = fx.p.buffers.reconfigure((256<<10)+registryTestBytes(fx.p), time.Hour)
						}
						held <- heldResult{lease, err}
					})
					w.WriteHeader(http.StatusServiceUnavailable)
				}))
				defer first.Close()
				fx.p.soleFacade().discovery.AddManual(nodeForModel(t, "a", first.URL, "small:1b"))
			} else {
				hold = bufferMustAcquire(t, fx.p.buffers, 128<<10)
			}
			defer func() {
				if hold != nil {
					hold.close()
				}
			}()
			handlerDone := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(handlerDone)
				fx.p.soleFacade().handleHTTP(w, r)
			}))
			defer server.Close()
			ctx, cancelClient := context.WithCancel(context.Background())
			defer cancelClient()
			request, err := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+"/api/generate", strings.NewReader(`{"model":"weak"}`))
			if err != nil {
				t.Fatal(err)
			}
			clientDone := make(chan error, 1)
			go func() {
				resp, err := server.Client().Do(request)
				if resp != nil {
					_, readErr := io.Copy(io.Discard, resp.Body)
					closeErr := resp.Body.Close()
					err = errors.Join(err, readErr, closeErr)
				}
				clientDone <- err
			}()
			if later {
				select {
				case result := <-held:
					if result.err != nil {
						t.Fatal(result.err)
					}
					hold = result.lease
				case <-time.After(2 * time.Second):
					t.Fatal("initial dispatch did not establish retry contention")
				}
			}
			waitForCond(t, 2*time.Second, "handler preparation FIFO barrier", func() bool {
				fx.p.buffers.mu.Lock()
				defer fx.p.buffers.mu.Unlock()
				return fx.p.buffers.waiters.Len() == 1
			})
			fx.p.identityMu.Lock()
			var control *routingSpend
			for _, request := range fx.p.requests {
				control = request.owner
			}
			fx.p.identityMu.Unlock()
			if control == nil {
				t.Fatal("preparation lost its request cancellation controller")
			}
			fx.p.priorityMu.Lock()
			claims := len(fx.p.priorityReservations)
			fx.p.priorityMu.Unlock()
			publications := notificationsOf(t, fx.w, workloadSubmittedMethod)
			if claims != 0 || fx.eng["b"].hits() != 0 {
				t.Fatalf("queued preparation has claims=%d later-node hits=%d", claims, fx.eng["b"].hits())
			}
			if !later && (len(publications) != 0 || fx.eng["a"].hits() != 0) {
				t.Fatalf("initial preparation published/forwarded: jobs=%d hits=%d", len(publications), fx.eng["a"].hits())
			}
			if later {
				if len(publications) < 3 {
					t.Fatal("later preparation lacks original and cleared placement history")
				}
				var last workloadParams
				if err := json.Unmarshal(publications[len(publications)-1], &last); err != nil {
					t.Fatal(err)
				}
				if last.WorkloadInfo.ScheduledOn != "" {
					t.Fatalf("later preparation retained placement: %+v", last.WorkloadInfo)
				}
			}
			if disconnect {
				cancelClient() // Transport closes the live HTTP request connection.
			} else if !control.cancel(false) {
				t.Fatal("initial/later handler cancellation was not found")
			}
			select {
			case <-handlerDone:
			case <-time.After(2 * time.Second):
				t.Fatal("cancelled handler did not leave preparation")
			}
			select {
			case err := <-clientDone:
				if disconnect && !errors.Is(err, context.Canceled) {
					t.Fatalf("disconnect result=%v, want context cancellation", err)
				}
			case <-time.After(2 * time.Second):
				t.Fatal("client remained attached after handler cancellation")
			}
			hold.close()
			ownedAssertHandlerDrained(t, fx.p, control)
			if len(notificationsOf(t, fx.w, workloadSubmittedMethod)) != len(publications) || fx.eng["b"].hits() != 0 {
				t.Fatal("cancellation produced a new placement or forwarding attempt")
			}
		})
	}
	run("initial preparation request cancellation", false, false)
	run("initial preparation client disconnect", false, true)
	run("later preparation request cancellation", true, false)
	run("later preparation client disconnect", true, true)
}
