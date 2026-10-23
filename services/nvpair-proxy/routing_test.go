// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

func TestHandleHTTP_SoleOwnerWaitsForFirstContent(t *testing.T) {
	setForTest(t, &firstBodyTimeout, 100*time.Millisecond)
	setForTest(t, &retryBackoff, []time.Duration{time.Millisecond})
	tc := anyCase(t)
	var hits atomic.Int32
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		timer := time.NewTimer(300 * time.Millisecond)
		defer timer.Stop()
		select {
		case <-r.Context().Done():
			return
		case <-timer.C:
		}
		if _, err := io.WriteString(w, `{"done":true}`); err != nil {
			t.Errorf("write delayed engine response: %v", err)
		}
	}))
	defer engine.Close()
	disc := NewDiscovery()
	disc.AddManual(nodeForModel(t, "sole-owner", engine.URL, tc.advertisedModel))
	p := testProxy(tc.profile, disc, tc.profile.FacadePort)
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	rec := httptest.NewRecorder()
	p.soleFacade().handleHTTP(rec, tc.inferenceRequest().WithContext(ctx))
	if rec.Code != http.StatusOK || hits.Load() != 1 {
		t.Fatalf("sole slow owner: status %d, engine hits %d; want 200 and exactly one inference dispatch", rec.Code, hits.Load())
	}
	if got := decodeJSONBody(t, rec.Body.String())["done"]; got != true {
		t.Fatalf("response done = %v, want true", got)
	}
}

func TestUnifiedDispatch(t *testing.T) {
	t.Run("local peer and public share one attempt slot and allowance", func(t *testing.T) {
		owner := newRoutingRequest(nil)
		spendGrant(t, owner, "100", "60")
		for _, destination := range []string{"local", "peer"} {
			ctx, cancel := context.WithCancel(context.Background())
			a, err := owner.newDispatchAttempt(ctx, cancel, destination)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := owner.newAttempt(); !errors.Is(err, errRoutingSpend) {
				t.Fatalf("parallel public attempt admitted while %s is active: %v", destination, err)
			}
			a.finished(publicAttemptTerminal{})
			cancel()
		}
		public := spendAttempt(t, owner)
		remaining, err := public.tryAccount(nodepolicy.Account{Billing: nodepolicy.BillingPaid, PerRequestCeiling: "200"})
		if err != nil || remaining != 40 || owner.ceiling != 100 || owner.granted != 60 || public.ordinal != 4 {
			t.Fatalf("destination transition changed request allowance: remaining=%d ceiling=%d granted=%d ordinal=%d err=%v", remaining, owner.ceiling, owner.granted, public.ordinal, err)
		}
	})

	t.Run("stale availability callback cannot cancel replacement public attempt", func(t *testing.T) {
		var cancelled atomic.Int32
		owner := newRoutingRequest(func(string) { cancelled.Add(1) })
		old, err := owner.newDispatchAttempt(context.Background(), func() { cancelled.Add(1) }, "local")
		if err != nil {
			t.Fatal(err)
		}
		old.finished(publicAttemptTerminal{})
		current := spendAttempt(t, owner)
		if err := current.bindCancel(func() { cancelled.Add(1) }); err != nil {
			t.Fatal(err)
		}
		owner.abortLocalIf(old)
		old.firstSuccessfulByte()
		if cancelled.Load() != 0 || owner.cancelled || owner.committed || owner.active != current || old.abandoned() || owner.isExcluded("local") {
			t.Fatalf("stale callback changed replacement: cancellations=%d cancelled=%t committed=%t active=%p", cancelled.Load(), owner.cancelled, owner.committed, owner.active)
		}
		if !owner.cancel(false) || cancelled.Load() != 2 {
			t.Fatalf("request cancellation missed active public context: callbacks=%d", cancelled.Load())
		}
		if _, err := owner.newAttempt(); !errors.Is(err, errRoutingSpend) {
			t.Fatalf("cancelled owner admitted replacement: %v", err)
		}
	})

	t.Run("only one concurrent attempt is admitted", func(t *testing.T) {
		owner := newRoutingRequest(nil)
		start := make(chan struct{})
		var admitted atomic.Int32
		var done sync.WaitGroup
		for range 16 {
			done.Add(1)
			go func() {
				defer done.Done()
				<-start
				if _, err := owner.newAttempt(); err == nil {
					admitted.Add(1)
				}
			}()
		}
		close(start)
		done.Wait()
		if admitted.Load() != 1 || owner.ordinal != 1 {
			t.Fatalf("admitted=%d ordinal=%d, want one", admitted.Load(), owner.ordinal)
		}
	})

	t.Run("accepted headers do not commit but partial invalid raw byte does", func(t *testing.T) {
		owner := newRoutingRequest(nil)
		a, err := owner.newDispatchAttempt(context.Background(), func() {}, "local")
		if err != nil || !a.acceptResponse() {
			t.Fatalf("response setup: %v", err)
		}
		if owner.committed || a.committed || owner.state != routingAwaitBody {
			t.Fatal("headers committed the request")
		}
		body := routingResponseBody{ReadCloser: io.NopCloser(strings.NewReader("!")), attempt: a}
		var scratch [1]byte
		if n, err := body.Read(scratch[:]); n != 1 || err != nil || scratch[0] != '!' || !owner.committed || !a.committed || owner.state != routingCommitted {
			t.Fatalf("raw read boundary n=%d err=%v byte=%q request=%t attempt=%t state=%s", n, err, scratch, owner.committed, a.committed, owner.state)
		}
		a.finished(publicAttemptTerminal{})
		if _, err := owner.newAttempt(); !errors.Is(err, errRoutingSpend) {
			t.Fatalf("invalid partial frame permitted fallback: %v", err)
		}
	})

	t.Run("cancellation winning before raw read prevents exposure", func(t *testing.T) {
		owner := newRoutingRequest(nil)
		a, err := owner.newDispatchAttempt(context.Background(), func() {}, "peer")
		if err != nil || !a.acceptResponse() {
			t.Fatalf("response setup: %v", err)
		}
		cancelled := make(chan struct{})
		go func() { owner.cancel(false); close(cancelled) }()
		<-cancelled
		body := routingResponseBody{ReadCloser: io.NopCloser(strings.NewReader("a")), attempt: a}
		var scratch [1]byte
		if n, err := body.Read(scratch[:]); n != 0 || !errors.Is(err, context.Canceled) || owner.committed || a.committed {
			t.Fatalf("cancelled raw read exposed bytes: n=%d err=%v request=%t attempt=%t", n, err, owner.committed, a.committed)
		}
	})

	t.Run("first raw byte winning prevents regeneration", func(t *testing.T) {
		owner := newRoutingRequest(nil)
		a, err := owner.newDispatchAttempt(context.Background(), func() {}, "peer")
		if err != nil || !a.acceptResponse() {
			t.Fatalf("response setup: %v", err)
		}
		committed := make(chan struct{})
		go func() { a.firstSuccessfulByte(); close(committed) }()
		<-committed
		if !owner.cancel(true) || !owner.cancelled || !owner.committed || a.redispatched() || owner.isExcluded("peer") {
			t.Fatal("committed regeneration redispatched or lost commitment")
		}
	})

	t.Run("callbacks execute outside request mutex", func(t *testing.T) {
		var owner *routingSpend
		var callbacks int
		owner = newRoutingRequest(func(string) {
			owner.requestCancelled()
			callbacks++
		})
		a, err := owner.newDispatchAttempt(context.Background(), func() {
			owner.requestCancelled()
			callbacks++
		}, "local")
		if err != nil {
			t.Fatal(err)
		}
		if !owner.cancel(false) || callbacks != 2 || a.committed {
			t.Fatalf("cancellation callback facts: callbacks=%d committed=%t", callbacks, a.committed)
		}
	})

	t.Run("nil and stale lifecycle hooks deny", func(t *testing.T) {
		var absent *publicAttemptLifecycle
		if absent.commit() || absent.admitInterruption(streamInterruptReasonIdle) || absent.acceptedByGuard(context.Background()) == nil || absent.bindCancel(func() {}) == nil || absent.associated(accountReservation{}) == nil {
			t.Fatal("nil attempt granted authority")
		}
		owner := newRoutingRequest(nil)
		old := spendAttempt(t, owner)
		old.finished(publicAttemptTerminal{})
		current := spendAttempt(t, owner)
		if old.commit() || old.admitInterruption(streamInterruptReasonIdle) || old.acceptedByGuard(context.Background()) == nil || old.associated(accountReservation{Durability: accountReserveDurable}) == nil || owner.active != current {
			t.Fatal("stale attempt granted authority or replaced active attempt")
		}
	})

	t.Run("live stale snapshot tightens and deleted account denies", func(t *testing.T) {
		p, _, _, snapshot := egressFixture(t)
		owner := newRoutingRequest(nil)
		spendGrant(t, owner, "100", "20")
		policy := p.admission.currentPolicy()
		policy.Routing.Accounts[0].PerRequestCeiling = "30"
		if err := p.admission.activatePolicy(policy); err != nil {
			t.Fatal(err)
		}
		a := spendAttempt(t, owner)
		if err := p.observePublicAccount(snapshot, a); err != nil || owner.ceiling != 30 {
			t.Fatalf("stale D snapshot hid lower live policy: ceiling=%d err=%v", owner.ceiling, err)
		}
		a.finished(publicAttemptTerminal{})
		policy.Routing.Accounts[0].PerRequestCeiling = "200"
		if err := p.admission.activatePolicy(policy); err != nil {
			t.Fatal(err)
		}
		a = spendAttempt(t, owner)
		if err := p.observePublicAccount(snapshot, a); err != nil || owner.ceiling != 30 || owner.granted != 20 {
			t.Fatalf("higher live policy refilled allowance: ceiling=%d granted=%d err=%v", owner.ceiling, owner.granted, err)
		}
		a.finished(publicAttemptTerminal{})
		policy.Routing.Accounts = nil
		if err := p.admission.activatePolicy(policy); err != nil {
			t.Fatal(err)
		}
		a = spendAttempt(t, owner)
		if err := p.observePublicAccount(snapshot, a); !errors.Is(err, errPublicPermission) || owner.ceiling != 30 || owner.granted != 20 {
			t.Fatalf("missing account invented policy: ceiling=%d granted=%d err=%v", owner.ceiling, owner.granted, err)
		}
	})
}

func TestUnifiedDispatchPassthroughBeforeOwner(t *testing.T) {
	test := func(name string, firstStatus, wantStatus, wantSecond int32) {
		t.Run(name, func(t *testing.T) {
			var firstCalls, secondCalls atomic.Int32
			first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				firstCalls.Add(1)
				body, err := io.ReadAll(r.Body)
				if err != nil || string(body) != `{"model":"ignored","value":1}` {
					t.Errorf("control body changed: %q %v", body, err)
				}
				w.WriteHeader(int(firstStatus))
				_, _ = io.WriteString(w, "first")
			}))
			defer first.Close()
			second := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				secondCalls.Add(1)
				w.WriteHeader(http.StatusOK)
				_, _ = io.WriteString(w, "second")
			}))
			defer second.Close()
			tc := anyCase(t)
			discovery := NewDiscovery()
			discovery.AddManual(nodeFor(t, "a", first.URL))
			discovery.AddManual(nodeFor(t, "b", second.URL))
			writer := &recordingWriter{}
			p := newTestProxy(tc.profile, NewCodec(writer), discovery, 0)
			p.SetPriority([]string{"a", "b"})
			p.requestSequence = math.MaxInt64
			p.buffers = bufferTestBudget(t, 0, 0)
			recorder := httptest.NewRecorder()
			p.soleFacade().handleHTTP(recorder, httptest.NewRequest(http.MethodPost, "/control/test", strings.NewReader(`{"model":"ignored","value":1}`)))
			if recorder.Code != int(wantStatus) || firstCalls.Load() != 1 || secondCalls.Load() != wantSecond {
				t.Fatalf("single pass status=%d calls=%d/%d, want %d and 1/%d", recorder.Code, firstCalls.Load(), secondCalls.Load(), wantStatus, wantSecond)
			}
			if p.requestSequence != math.MaxInt64 || len(p.activeRoutingRequests()) != 0 || len(notificationsOf(t, writer, workloadSubmittedMethod)) != 0 || p.admission.currentActive() != 0 {
				t.Fatal("control entered inference identity/admission/workload machinery")
			}
			bufferAssertTotals(t, p.buffers, 0, 0, 0)
			p.priorityMu.Lock()
			claims := len(p.priorityReservations)
			p.priorityMu.Unlock()
			if claims != 0 || len(notificationsOf(t, writer, engines.AddressMethod(tc.profile.Name, "proxy/request-started"))) != 1 || len(notificationsOf(t, writer, engines.AddressMethod(tc.profile.Name, "proxy/request"))) != 1 {
				t.Fatal("control changed placement or request notifications")
			}
		})
	}
	test("successful control bypasses unavailable inference owners", http.StatusOK, http.StatusOK, 0)
	test("control 404 does not use inference model failover", http.StatusNotFound, http.StatusNotFound, 0)
	test("control 503 makes one pass over concrete candidates", http.StatusServiceUnavailable, http.StatusOK, 1)
}

func TestUnifiedDispatchPostcommitAllocationRefusal(t *testing.T) {
	received, resume := make(chan struct{}), make(chan struct{})
	var resumeOnce sync.Once
	release := func() { resumeOnce.Do(func() { close(resume) }) }
	t.Cleanup(release)
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Error(err)
			return
		}
		close(received)
		<-resume
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusAccepted)
		// The initial raw array is prepaid. Make the first frame require a
		// replacement grant after commitment, before any frame can be delivered.
		event := strings.Replace(continuitySSEEvent, "intact", strings.Repeat("x", streamFrameReadChunk), 1)
		if _, err := io.WriteString(w, event+"data: [DONE]\n\n"); err != nil {
			t.Error(err)
		}
	}))
	defer engine.Close()
	other := newRecordingEngine(t, http.StatusOK)
	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(t, "a", engine.URL, "m"))
	discovery.AddManual(nodeForModel(t, "b", other.URL, "m"))
	writer := &recordingWriter{}
	p := newTestProxy(ollamaOnlyProfile(t), NewCodec(writer), discovery, 0)
	p.SetPriority([]string{"a", "b"})
	p.buffers = bufferTestBudget(t, (256<<10)+bufferFixtureRemotePreparationBytes(t, `{"model":"m","stream":true}`), 0)
	recorder, done := httptest.NewRecorder(), make(chan struct{})
	go func() {
		defer close(done)
		p.soleFacade().handleHTTP(recorder, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)))
	}()
	waitChan(t, received, "prepared request at upstream")
	p.buffers.mu.Lock()
	remaining := p.buffers.capacity - p.buffers.used
	p.buffers.mu.Unlock()
	blocker := bufferMustAcquire(t, p.buffers, remaining)
	release()
	waitChan(t, done, "postcommit refusal cleanup")
	if recorder.Code != http.StatusAccepted || recorder.Body.String() != string(wantSSEInterruption(streamInterruptReasonBudget)) || other.hits() != 0 {
		t.Fatalf("committed refusal rewrote status or fell back: status=%d body=%q fallback=%d", recorder.Code, recorder.Body.String(), other.hits())
	}
	var event RequestEvent
	completed := notificationsOf(t, writer, engines.AddressMethod(p.soleFacade().profile.Name, "proxy/request"))
	if len(completed) != 1 {
		t.Fatalf("completion notifications=%d, want one", len(completed))
	}
	if err := json.Unmarshal(completed[0], &event); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(event.Error, streamInterruptReasonBudget) {
		t.Fatalf("allocation reason lost: %+v", event)
	}
	blocker.close()
	ownedAssertHandlerDrained(t, p, nil)
}

func TestUnifiedDispatchUncertainCapacityOutlivesTransport(t *testing.T) {
	h := newStrictHarness(t)
	ticket := mustStrictTicket(t, h.a)
	if err := ticket.BeforeStrictDispatch(); err != nil {
		t.Fatal(err)
	}
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	p.admission = h.a
	p.buffers = bufferTestBudget(t, 256<<10, 0)
	r, buffers := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)))
	if _, _, err := bufferBodyAndModel(httptest.NewRecorder(), r, 1<<20); err != nil {
		t.Fatal(err)
	}
	owner := newRoutingRequest(nil)
	registered, err := p.registerRoutingRequest(owner)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.registerWorkload(registered, workloadKey{engine: "ollama", id: "1"}); err != nil {
		t.Fatal(err)
	}
	p.priorityMu.Lock()
	placement := p.claimReservationLocked("local")
	p.priorityMu.Unlock()
	a, err := owner.newDispatchAttempt(r.Context(), func() {}, "local")
	if err != nil || !a.acceptResponse() {
		t.Fatalf("prepare accepted attempt: %v", err)
	}
	sink := newResponseSink(httptest.NewRecorder(), nil)
	ctx := withResponseSink(r.Context(), sink)
	response := &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: &routingResponseBody{ReadCloser: io.NopCloser(strings.NewReader("data: [DONE]\n\n")), attempt: a}}
	reader, err := installResponseStream(response, originStreamRequest{protocol: streamProtocolSSE, declared: true, choices: 1}, ctx, time.Second, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.Copy(sink, response.Body); err != nil {
		t.Fatal(err)
	}
	if err := reader.Close(); err != nil {
		t.Fatal(err)
	}
	if err := sink.FlushError(); err != nil || !sink.result().DeliveryComplete || !sink.result().ProtocolComplete || !owner.committed {
		t.Fatalf("delivered terminal: err=%v sink=%+v committed=%v", err, sink.result(), owner.committed)
	}
	if h.a.currentActive() != 1 {
		t.Fatal("delivered terminal marker became engine-idle evidence")
	}
	ticket.Finish(strictUncertain)
	a.finished(publicAttemptTerminal{ProtocolComplete: true, Delivered: true})
	p.releaseReservation(placement)
	p.unregisterRoutingRequest(registered)
	buffers.close()
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
	h.a.mu.Lock()
	active, tickets, markers, queue := h.a.active, len(h.a.tickets), len(h.a.strict.markers), h.a.queued
	h.a.mu.Unlock()
	p.priorityMu.Lock()
	placements := len(p.priorityReservations)
	p.priorityMu.Unlock()
	if active != 1 || tickets != 0 || markers != 1 || queue != 0 || placements != 0 || len(p.activeRoutingRequests()) != 0 || len(p.workloadControls("ollama", "1")) != 0 {
		t.Fatalf("liability/transport cleanup conflated: active=%d tickets=%d markers=%d queue=%d placements=%d requests=%d", active, tickets, markers, queue, placements, len(p.activeRoutingRequests()))
	}
}

func TestUnifiedDispatchCancelDuringFirstRawRead(t *testing.T) {
	owner := newRoutingRequest(nil)
	a, err := owner.newDispatchAttempt(context.Background(), func() {}, "peer")
	if err != nil || !a.acceptResponse() {
		t.Fatalf("prepare response: %v", err)
	}
	upstream := &routingReadBarrier{entered: make(chan struct{}), release: make(chan struct{})}
	body := &routingResponseBody{ReadCloser: upstream, attempt: a}
	type readResult struct {
		n   int
		err error
	}
	read := make(chan readResult, 1)
	go func() {
		var scratch [1]byte
		n, err := body.Read(scratch[:])
		read <- readResult{n, err}
	}()
	waitChan(t, upstream.entered, "upstream first-byte read barrier")
	if !owner.cancel(false) {
		t.Fatal("cancel did not win while the raw read was blocked")
	}
	close(upstream.release)
	select {
	case result := <-read:
		if result.n != 0 || !errors.Is(result.err, context.Canceled) || owner.committed || a.committed {
			t.Fatalf("cancelled barrier exposed raw data: n=%d err=%v committed=%v/%v", result.n, result.err, owner.committed, a.committed)
		}
	case <-time.After(time.Second):
		t.Fatal("raw read failed to leave barrier")
	}
	if err := body.Close(); err != nil || !upstream.closed {
		t.Fatalf("raw borrower did not close: err=%v closed=%v", err, upstream.closed)
	}
	if _, err := owner.newAttempt(); !errors.Is(err, errRoutingSpend) {
		t.Fatalf("cancelled raw-read request retried: %v", err)
	}
}

type routingReadBarrier struct {
	entered, release chan struct{}
	closed           bool
}

func (b *routingReadBarrier) Read(p []byte) (int, error) {
	close(b.entered)
	<-b.release
	p[0] = '!'
	return 1, nil
}

func (b *routingReadBarrier) Close() error { b.closed = true; return nil }

func TestUnifiedDispatchBodyPreparationDeadline(t *testing.T) {
	original := jobDeadline
	jobDeadline = 250 * time.Millisecond
	defer func() { jobDeadline = original }()
	for _, stage := range []string{"initial allocation", "growth", "EOF probe", "parser allowance"} {
		t.Run(stage, func(t *testing.T) {
			writer := &recordingWriter{}
			engine := newRecordingEngine(t, http.StatusOK)
			discovery := NewDiscovery()
			discovery.AddManual(nodeForModel(t, "a", engine.URL, "m"))
			p := newTestProxy(ollamaOnlyProfile(t), NewCodec(writer), discovery, 0)
			p.buffers = bufferTestBudget(t, 256<<10, time.Hour)
			var hold *bufferLease
			defer func() {
				if hold != nil {
					hold.close()
				}
			}()
			body := `{"model":"m"}`
			if stage == "growth" {
				body = strings.Repeat(" ", 64<<10)
			}
			r := httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(body))
			if stage == "initial allocation" {
				hold = bufferMustAcquire(t, p.buffers, 256<<10)
			} else {
				r.Body = &routingPressureBody{Reader: strings.NewReader(body), eofWithData: stage == "parser allowance", pressure: func() {
					p.buffers.mu.Lock()
					remaining := p.buffers.capacity - p.buffers.used
					p.buffers.mu.Unlock()
					hold = bufferMustAcquire(t, p.buffers, remaining)
				}}
				if stage == "growth" {
					r.ContentLength = -1
				}
			}
			ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
			defer cancel()
			rec, done := httptest.NewRecorder(), make(chan struct{})
			go func() { defer close(done); p.soleFacade().handleHTTP(rec, r.WithContext(ctx)) }()
			select {
			case <-done:
			case <-time.After(time.Second):
				cancel()
				<-done
				t.Fatal("owned body wait exceeded the request deadline")
			}
			bodyError := strings.Contains(rec.Body.String(), `"code":"request-body"`) || strings.Contains(rec.Body.String(), `"code":"inference-buffer"`)
			if rec.Code != http.StatusServiceUnavailable || !bodyError || engine.hits() != 0 || len(notificationsOf(t, writer, workloadSubmittedMethod)) != 0 {
				t.Fatalf("body deadline status=%d body=%q upstream=%d", rec.Code, rec.Body.String(), engine.hits())
			}
			hold.close()
			ownedAssertHandlerDrained(t, p, nil)
		})
	}
}

type routingPressureBody struct {
	*strings.Reader
	pressure    func()
	eofWithData bool
}

func (b *routingPressureBody) Read(p []byte) (int, error) {
	if b.pressure != nil {
		pressure := b.pressure
		b.pressure = nil
		pressure()
	}
	n, err := b.Reader.Read(p)
	if b.eofWithData && b.Len() == 0 {
		err = io.EOF
	}
	return n, err
}
func (*routingPressureBody) Close() error { return nil }

func TestUnifiedDispatchCandidatePreparationDeadline(t *testing.T) {
	original := jobDeadline
	jobDeadline = 600 * time.Millisecond
	defer func() { jobDeadline = original }()
	for _, later := range []bool{false, true} {
		name := "initial preparation"
		if later {
			name = "retry consumes original deadline"
		}
		t.Run(name, func(t *testing.T) {
			pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "small:1b"}}})
			fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"a": {"small:1b"}, "b": {"small:1b"}}, nil)
			capacity := int64(256 << 10)
			if later {
				capacity += bufferFixtureRemotePreparationBytes(t, `{"model":"small:1b"}`)
			}
			fx.p.buffers = bufferTestBudget(t, capacity, time.Hour)
			held := make(chan *bufferLease, 1)
			var firstCalls atomic.Int32
			if later {
				first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					firstCalls.Add(1)
					select {
					case <-time.After(250 * time.Millisecond):
					case <-r.Context().Done():
						return
					}
					lease, err := fx.p.buffers.acquire(r.Context(), 128<<10)
					if err != nil {
						t.Error(err)
						return
					}
					// Restore the original pressure before the next candidate's
					// transform, without denying the first candidate preparation.
					if err := fx.p.buffers.reconfigure(256<<10, time.Hour); err != nil {
						lease.close()
						t.Error(err)
						return
					}
					held <- lease
					w.WriteHeader(http.StatusServiceUnavailable)
				}))
				defer first.Close()
				fx.p.soleFacade().discovery.AddManual(nodeForModel(t, "a", first.URL, "small:1b"))
			} else {
				held <- bufferMustAcquire(t, fx.p.buffers, 128<<10)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			rec, done := httptest.NewRecorder(), make(chan struct{})
			r := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"weak"}`)).WithContext(ctx)
			go func() { defer close(done); fx.p.soleFacade().handleHTTP(rec, r) }()
			var hold *bufferLease
			select {
			case hold = <-held:
			case <-ctx.Done():
				t.Fatal("failed to establish pressure")
			}
			defer hold.close()
			waitForCond(t, time.Second, "preparation waiter", func() bool {
				fx.p.buffers.mu.Lock()
				defer fx.p.buffers.mu.Unlock()
				return fx.p.buffers.waiters.Len() == 1
			})
			owners := fx.p.activeRoutingRequests()
			if len(owners) != 1 {
				t.Fatalf("owners=%d, want one", len(owners))
			}
			owner := fx.p.routingRequestControl(owners[0])
			deadline := owner.deadline
			fx.p.buffers.mu.Lock()
			waiter := fx.p.buffers.waiters.Front().Value.(*bufferWaiter)
			waitDeadline := waiter.deadline
			fx.p.buffers.mu.Unlock()
			if !waitDeadline.Equal(deadline) {
				cancel()
				<-done
				t.Fatalf("wait deadline=%v, want original %v", waitDeadline, deadline)
			}
			fx.p.priorityMu.Lock()
			claims := len(fx.p.priorityReservations)
			fx.p.priorityMu.Unlock()
			if claims != 0 || fx.eng["b"].hits() != 0 || fx.p.admission.currentActive() != 0 {
				t.Fatal("queued preparation caused placement/admission/forwarding")
			}
			select {
			case <-done:
			case <-time.After(time.Until(deadline) + 400*time.Millisecond):
				cancel()
				<-done
				t.Fatal("preparation replenished or exceeded original deadline")
			}
			if rec.Code != http.StatusServiceUnavailable || !strings.Contains(rec.Body.String(), `"code":"inference-buffer"`) || fx.eng["b"].hits() != 0 || !later && (firstCalls.Load() != 0 || fx.eng["a"].hits() != 0 || len(notificationsOf(t, fx.w, workloadSubmittedMethod)) != 0) {
				t.Fatalf("preparation deadline status=%d body=%q first=%d later=%d", rec.Code, rec.Body.String(), firstCalls.Load(), fx.eng["b"].hits())
			}
			hold.close()
			ownedAssertHandlerDrained(t, fx.p, owner)
		})
	}
}

func TestUnifiedDispatchDestinationPreparationDeadline(t *testing.T) {
	original := jobDeadline
	jobDeadline = 250 * time.Millisecond
	defer func() { jobDeadline = original }()
	for _, path := range []string{"self strict", "ingress strict", "ingress concrete", "self strict qualification", "ingress strict qualification"} {
		t.Run(path, func(t *testing.T) {
			p, engine := strictHTTPFixture(t)
			p.buffers = bufferTestBudget(t, 256<<10, time.Hour)
			r := strictHTTPRequest(t)
			pressure := int64(128 << 10)
			if strings.Contains(path, "qualification") {
				body := strings.TrimSuffix(strictBody("m"), "}") + `,"store":false}`
				r = strictHTTPRequestWith(t, body)
				// Body, fixed scanner and decoded model fit. The admitted B7
				// store:false removal must then wait for its own representation.
				pressure = (256 << 10) - (96 << 10) - int64(len(body)) - 1 - 128
			}
			hold := bufferMustAcquire(t, p.buffers, pressure)
			defer hold.close()
			if path == "ingress concrete" {
				r.Header.Del(nodepolicy.StrictAdmissionHeader)
			}
			ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
			defer cancel()
			rec, done := httptest.NewRecorder(), make(chan struct{})
			go func() {
				defer close(done)
				if strings.HasPrefix(path, "self") {
					p.soleFacade().handleHTTP(rec, r.WithContext(ctx))
				} else {
					p.soleFacade().serveIngress(rec, r.WithContext(ctx), "peer")
				}
			}()
			select {
			case <-done:
			case <-time.After(time.Second):
				cancel()
				<-done
				t.Fatal("destination preparation exceeded request deadline")
			}
			if rec.Code != http.StatusServiceUnavailable || !strings.Contains(rec.Body.String(), `"code":"inference-buffer"`) || engine.hits() != 0 {
				t.Fatalf("destination status=%d body=%q upstream=%d", rec.Code, rec.Body.String(), engine.hits())
			}
			p.admission.mu.Lock()
			markers := len(p.admission.strict.markers)
			p.admission.mu.Unlock()
			if markers != 0 {
				t.Fatalf("preparation created strict capacity markers=%d", markers)
			}
			hold.close()
			ownedAssertHandlerDrained(t, p, nil)
		})
	}
}

func TestUnifiedDispatchPreparedStreamOutlivesDeadline(t *testing.T) {
	original := jobDeadline
	jobDeadline = 250 * time.Millisecond
	defer func() { jobDeadline = original }()
	for _, path := range []string{"self", "ingress", "self strict", "ingress strict"} {
		t.Run(path, func(t *testing.T) {
			p, _ := strictHTTPFixture(t)
			p.buffers = bufferTestBudget(t, 2<<20, 0)
			release := make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			defer unblock()
			engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, continuitySSEEvent)
				w.(http.Flusher).Flush()
				select {
				case <-release:
					_, _ = io.WriteString(w, "data: [DONE]\n\n")
				case <-r.Context().Done():
				}
			}))
			defer engine.Close()
			defer unblock()
			setBackend(t, p.soleFacade(), engine.URL, true)
			p.soleFacade().discovery.AddManual(selfNode("self", p.soleFacade().port, "m"))
			metadata := make(chan bool, 1)
			transport := p.plainHTTPTransport()
			dial := transport.DialContext
			transport.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
				allowance, ok := ctx.Value(performancePreparationKey{}).(performancePreparation)
				metadata <- ok && allowance.bytes > 0 && inferenceBuffers(ctx) != nil
				return dial(ctx, network, address)
			}
			body := strings.TrimSuffix(strictBody("m"), "}") + `,"stream":true}`
			r := strictHTTPRequestWith(t, body)
			if !strings.Contains(path, "strict") {
				r.Header.Del(nodepolicy.StrictAdmissionHeader)
			}
			ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
			defer cancel()
			writer := &strictFirstByteWriter{ResponseRecorder: httptest.NewRecorder(), first: make(chan struct{})}
			done := make(chan struct{})
			started := time.Now()
			go func() {
				defer close(done)
				if strings.HasPrefix(path, "self") {
					p.soleFacade().handleHTTP(writer, r.WithContext(ctx))
				} else {
					p.soleFacade().serveIngress(writer, r.WithContext(ctx), "peer")
				}
			}()
			waitChan(t, writer.first, "accepted stream first frame")
			select {
			case admitted := <-metadata:
				if !admitted {
					t.Error("live transport lost admitted performance allowance")
				}
			case <-ctx.Done():
				t.Fatal("no transport metadata observation")
			}
			<-time.After(time.Until(started.Add(jobDeadline + 100*time.Millisecond)))
			select {
			case <-done:
				t.Error("temporary preparation cancellation/deadline truncated accepted stream")
			default:
			}
			unblock()
			waitChan(t, done, "stream completion after preparation deadline")
			if writer.Code != http.StatusOK || writer.Body.String() != continuitySSEEvent+"data: [DONE]\n\n" {
				t.Fatalf("accepted stream status=%d body=%q", writer.Code, writer.Body.String())
			}
			bufferAssertTotals(t, p.buffers, 0, 0, 0)
			if len(p.activeRoutingRequests()) != 0 {
				t.Fatal("completed stream retained request registry")
			}
		})
	}
}

func TestUnifiedDispatchStrictPreOutputDeadline(t *testing.T) {
	original := jobDeadline
	jobDeadline = 250 * time.Millisecond
	defer func() { jobDeadline = original }()
	for _, stage := range []string{"headers wait", "first body wait"} {
		t.Run(stage, func(t *testing.T) {
			p, _ := strictHTTPFixture(t)
			p.buffers = bufferTestBudget(t, 2<<20, 0)
			entered, cancelled := make(chan struct{}), make(chan struct{})
			stopEngine := make(chan struct{})
			engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					t.Error(err)
					return
				}
				close(entered)
				if stage == "first body wait" {
					w.WriteHeader(http.StatusOK)
					w.(http.Flusher).Flush()
				}
				select {
				case <-r.Context().Done():
					close(cancelled)
				case <-stopEngine:
				}
			}))
			defer engine.Close()
			defer close(stopEngine)
			setBackend(t, p.soleFacade(), engine.URL, true)
			r := strictHTTPRequest(t) // Existing strict pre-output envelope is 1s.
			ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
			defer cancel()
			rec, done := httptest.NewRecorder(), make(chan struct{})
			go func() { defer close(done); p.soleFacade().handleHTTP(rec, r.WithContext(ctx)) }()
			waitChan(t, entered, "strict transport before first output")
			select {
			case <-done:
			case <-time.After(time.Second):
				cancel()
				<-done
				t.Fatal("strict pre-output transport outlived the request deadline")
			}
			waitChan(t, cancelled, "strict transport cancellation at request deadline")
			if rec.Code != http.StatusGatewayTimeout {
				t.Fatalf("strict request-deadline status=%d body=%q, want504", rec.Code, rec.Body.String())
			}
			bufferAssertTotals(t, p.buffers, 0, 0, 0)
			p.admission.mu.Lock()
			active, queued, tickets, markers := p.admission.active, p.admission.queued, len(p.admission.tickets), len(p.admission.strict.markers)
			p.admission.mu.Unlock()
			if active != 1 || markers != 1 || tickets != 0 || queued != 0 || len(p.activeRoutingRequests()) != 0 {
				t.Fatalf("transport cleanup erased uncertain capacity or retained slots: active=%d markers=%d tickets=%d queued=%d", active, markers, tickets, queued)
			}
		})
	}
}

func TestUnifiedDispatchSocketBodyInterruption(t *testing.T) {
	original := jobDeadline
	defer func() { jobDeadline = original }()
	for _, scenario := range []string{"origin deadline", "origin owner cancellation", "origin disconnect", "origin shutdown", "ingress deadline", "ingress parent cancellation"} {
		t.Run(scenario, func(t *testing.T) {
			jobDeadline = 10 * time.Second
			if strings.HasSuffix(scenario, "deadline") {
				jobDeadline = 150 * time.Millisecond
			}
			p := newTestProxy(ollamaOnlyProfile(t), NewCodec(&recordingWriter{}), NewDiscovery(), 0)
			p.buffers = bufferTestBudget(t, 1<<20, time.Hour)
			p.serveCtx, p.cancel = context.WithCancel(context.Background())
			defer p.cancel()
			done, entered := make(chan struct{}), make(chan context.CancelFunc, 1)
			reading := make(chan struct{})
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				ctx, cancel := context.WithCancel(r.Context())
				defer cancel()
				entered <- cancel
				r.Body = &routingUploadReadWitness{ReadCloser: r.Body, entered: reading}
				if strings.HasPrefix(scenario, "ingress") {
					p.soleFacade().serveIngress(w, r.WithContext(ctx), "peer")
				} else {
					p.soleFacade().handleHTTP(w, r.WithContext(ctx))
				}
				close(done)
			}))
			server.Config.BaseContext = func(net.Listener) context.Context { return p.serveCtx }
			server.Start()
			defer server.Close()
			conn, err := net.Dial("tcp", strings.TrimPrefix(server.URL, "http://"))
			if err != nil {
				t.Fatal(err)
			}
			defer conn.Close()
			if err := conn.SetDeadline(time.Now().Add(3 * time.Second)); err != nil {
				t.Fatal(err)
			}
			if _, err := io.WriteString(conn, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 13\r\nConnection: close\r\n\r\n"); err != nil {
				t.Fatal(err)
			}
			cancel := <-entered
			var owner *routingSpend
			if strings.HasPrefix(scenario, "origin") {
				waitForCond(t, time.Second, "registered socket upload", func() bool { return len(p.activeRoutingRequests()) == 1 })
				owner = p.routingRequestControl(p.activeRoutingRequests()[0])
			}
			waitChan(t, reading, "actual initial socket body read")
			switch scenario {
			case "origin owner cancellation":
				if owner == nil || !owner.cancel(false) {
					t.Fatal("owner cancellation did not win")
				}
			case "origin disconnect":
				if err := conn.Close(); err != nil {
					t.Fatal(err)
				}
			case "origin shutdown":
				p.cancel()
			case "ingress parent cancellation":
				cancel()
			}
			bounded := false
			select {
			case <-done:
				bounded = true
			case <-time.After(750 * time.Millisecond):
			}
			live := len(p.activeRoutingRequests())
			_ = conn.Close()
			cancel()
			waitChan(t, done, "socket upload cleanup")
			if !bounded || live != 0 {
				t.Fatalf("body read retained handler/registry before caller cleanup: returned=%v live=%d", bounded, live)
			}
			ownedAssertHandlerDrained(t, p, owner)
		})
	}
}

type routingUploadReadWitness struct {
	io.ReadCloser
	entered chan struct{}
	once    sync.Once
}

func (b *routingUploadReadWitness) Read(p []byte) (int, error) {
	b.once.Do(func() { close(b.entered) })
	return b.ReadCloser.Read(p)
}

func TestUnifiedDispatchConcretePreOutputDeadline(t *testing.T) {
	original := jobDeadline
	jobDeadline = 150 * time.Millisecond
	defer func() { jobDeadline = original }()
	for _, stage := range []string{"headers wait", "first body wait", "first body wait with delayed deadline callback"} {
		t.Run(stage, func(t *testing.T) {
			entered, cancelled, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					t.Error(err)
					return
				}
				close(entered)
				if strings.HasPrefix(stage, "first body wait") {
					w.WriteHeader(http.StatusOK)
					w.(http.Flusher).Flush()
				}
				select {
				case <-r.Context().Done():
					close(cancelled)
				case <-release:
				}
			}))
			defer upstream.Close()
			defer unblock()
			fallback := newRecordingEngine(t, http.StatusOK)
			discovery := NewDiscovery()
			discovery.AddManual(nodeForModel(t, "a", upstream.URL, "m"))
			discovery.AddManual(nodeForModel(t, "b", fallback.URL, "m"))
			p := newTestProxy(ollamaOnlyProfile(t), NewCodec(&recordingWriter{}), discovery, 0)
			p.buffers = bufferTestBudget(t, 2<<20, time.Hour)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			rec, done := httptest.NewRecorder(), make(chan struct{})
			r := httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m"}`)).WithContext(ctx)
			go func() { defer close(done); p.soleFacade().handleHTTP(rec, r) }()
			waitChan(t, entered, "concrete pre-output transport")
			keys := p.activeRoutingRequests()
			if len(keys) != 1 {
				t.Fatalf("registered requests=%d", len(keys))
			}
			owner := p.routingRequestControl(keys[0])
			if strings.Contains(stage, "delayed deadline callback") {
				// Suppress delivery of the timer collaborator. The synchronous
				// body/retry boundary must still record expiry at the sole owner.
				owner.mu.Lock()
				stop := owner.stopPreOutput
				owner.mu.Unlock()
				if stop == nil || !stop() {
					t.Fatal("deadline callback already ran before its scheduling barrier")
				}
			}
			bounded := false
			select {
			case <-done:
				bounded = true
			case <-time.After(750 * time.Millisecond):
			}
			cancel()
			waitChan(t, done, "concrete pre-output cleanup")
			waitChan(t, cancelled, "concrete transport cancelled")
			if !bounded || rec.Code != http.StatusServiceUnavailable || fallback.hits() != 0 || owner.committed || !owner.cancelled {
				t.Fatalf("pre-output deadline returned=%v status=%d fallback=%d committed=%v cancelled=%v", bounded, rec.Code, fallback.hits(), owner.committed, owner.cancelled)
			}
			if _, err := owner.newAttempt(); !errors.Is(err, errRoutingSpend) {
				t.Fatalf("deadline allowed another attempt: %v", err)
			}
			bufferAssertTotals(t, p.buffers, 0, 0, 0)
			if len(p.activeRoutingRequests()) != 0 {
				t.Fatal("deadline retained request registry")
			}
		})
	}
}

func TestUnifiedDispatchDeadlineCommitArbitration(t *testing.T) {
	t.Run("late raw byte cannot outrun deadline callback", func(t *testing.T) {
		owner := newRoutingRequest(nil)
		owner.deadline = time.Now().Add(30 * time.Millisecond)
		var cancelled atomic.Int32
		a, err := owner.newDispatchAttempt(context.Background(), func() { cancelled.Add(1) }, "local")
		if err != nil || !a.acceptResponse() {
			t.Fatalf("attempt setup: %v", err)
		}
		<-time.After(time.Until(owner.deadline) + time.Millisecond)
		if a.commit() || owner.committed || !owner.cancelled || cancelled.Load() != 1 || owner.reason != routingDeadlineReason {
			t.Fatalf("late byte won: committed=%v cancelled=%v callbacks=%d reason=%q", owner.committed, owner.cancelled, cancelled.Load(), owner.reason)
		}
	})
	t.Run("committed raw byte defeats delayed expiry", func(t *testing.T) {
		owner := newRoutingRequest(nil)
		owner.deadline = time.Now().Add(time.Second)
		var cancelled atomic.Int32
		a, err := owner.newDispatchAttempt(context.Background(), func() { cancelled.Add(1) }, "local")
		if err != nil || !a.acceptResponse() || !a.commit() {
			t.Fatalf("commit setup: %v", err)
		}
		if owner.expireBeforeOutput() || owner.cancelled || cancelled.Load() != 0 || !owner.committed {
			t.Fatalf("expiry rewrote raw winner: cancelled=%v callbacks=%d committed=%v", owner.cancelled, cancelled.Load(), owner.committed)
		}
	})
}

type routingReadDeadlineWriter struct {
	http.ResponseWriter
	mu        sync.Mutex
	deadlines []time.Time
}

func (w *routingReadDeadlineWriter) SetReadDeadline(deadline time.Time) error {
	w.mu.Lock()
	w.deadlines = append(w.deadlines, deadline)
	w.mu.Unlock()
	return http.NewResponseController(w.ResponseWriter).SetReadDeadline(deadline)
}

func TestUnifiedDispatchClearsBodyReadDeadline(t *testing.T) {
	original := jobDeadline
	jobDeadline = 200 * time.Millisecond
	defer func() { jobDeadline = original }()
	engine := newRecordingEngine(t, http.StatusOK)
	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(t, "a", engine.URL, "m"))
	p := newTestProxy(ollamaOnlyProfile(t), NewCodec(&recordingWriter{}), discovery, 0)
	observations := make(chan []time.Time, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		wrapped := &routingReadDeadlineWriter{ResponseWriter: w}
		p.soleFacade().handleHTTP(wrapped, r)
		wrapped.mu.Lock()
		observations <- append([]time.Time(nil), wrapped.deadlines...)
		wrapped.mu.Unlock()
	}))
	defer server.Close()
	conn, err := net.Dial("tcp", strings.TrimPrefix(server.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if err := conn.SetDeadline(time.Now().Add(3 * time.Second)); err != nil {
		t.Fatal(err)
	}
	reader := bufio.NewReader(conn)
	for i := 0; i < 2; i++ {
		if _, err := io.WriteString(conn, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 13\r\n\r\n{\"model\":\"m\"}"); err != nil {
			t.Fatal(err)
		}
		response, err := http.ReadResponse(reader, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, readErr := io.Copy(io.Discard, response.Body)
		closeErr := response.Body.Close()
		if readErr != nil || closeErr != nil || response.StatusCode != http.StatusOK {
			t.Fatalf("reused response status=%d read=%v close=%v", response.StatusCode, readErr, closeErr)
		}
		deadlines := <-observations
		if len(deadlines) < 2 || deadlines[0].IsZero() || !deadlines[len(deadlines)-1].IsZero() {
			t.Fatalf("body deadline was not installed then cleared: %v", deadlines)
		}
		if i == 0 {
			<-time.After(jobDeadline + 50*time.Millisecond)
		}
	}
	if engine.hits() != 2 {
		t.Fatalf("reused connection forwarded=%d", engine.hits())
	}
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
}

type routingReadDeadlineBarrier struct {
	*routingReadDeadlineWriter
	interrupted, release, cleared chan struct{}
}

func (w *routingReadDeadlineBarrier) SetReadDeadline(deadline time.Time) error {
	err := w.routingReadDeadlineWriter.SetReadDeadline(deadline)
	if deadline.IsZero() {
		close(w.cleared)
	} else if time.Until(deadline) < time.Second {
		close(w.interrupted)
		<-w.release
	}
	return err
}

func TestUnifiedDispatchJoinsBodyInterrupt(t *testing.T) {
	original := jobDeadline
	jobDeadline = 10 * time.Second
	defer func() { jobDeadline = original }()
	p := newTestProxy(ollamaOnlyProfile(t), NewCodec(&recordingWriter{}), NewDiscovery(), 0)
	interrupted, release, cleared, done := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	defer unblock()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p.soleFacade().handleHTTP(&routingReadDeadlineBarrier{routingReadDeadlineWriter: &routingReadDeadlineWriter{ResponseWriter: w}, interrupted: interrupted, release: release, cleared: cleared}, r)
		close(done)
	}))
	defer server.Close()
	defer unblock()
	conn, err := net.Dial("tcp", strings.TrimPrefix(server.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if _, err := io.WriteString(conn, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 13\r\nConnection: close\r\n\r\n"); err != nil {
		t.Fatal(err)
	}
	waitForCond(t, time.Second, "registered interrupt barrier", func() bool { return len(p.activeRoutingRequests()) == 1 })
	owner := p.routingRequestControl(p.activeRoutingRequests()[0])
	if owner == nil || !owner.cancel(false) {
		t.Fatal("cancel did not win")
	}
	waitChan(t, interrupted, "real socket interrupted before callback return")
	early := false
	select {
	case <-cleared:
		early = true
	case <-done:
		early = true
	case <-time.After(100 * time.Millisecond):
	}
	live := len(p.activeRoutingRequests())
	unblock()
	waitChan(t, done, "joined interrupt cleanup")
	if early || live != 1 {
		t.Fatalf("handler cleared/released ownership before interrupt joined: early=%v live=%d", early, live)
	}
	waitChan(t, cleared, "temporary read deadline clear")
	ownedAssertHandlerDrained(t, p, owner)
}

func TestUnifiedDispatchIngressPreOutputDeadline(t *testing.T) {
	original := jobDeadline
	defer func() { jobDeadline = original }()
	for _, stage := range []string{"headers deadline", "first body deadline", "headers cancellation", "first body cancellation"} {
		t.Run(stage, func(t *testing.T) {
			jobDeadline = 150 * time.Millisecond
			if strings.HasSuffix(stage, "cancellation") {
				jobDeadline = 10 * time.Second
			}
			p, _ := strictHTTPFixture(t)
			p.buffers = bufferTestBudget(t, 2<<20, 0)
			entered, engineDone, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(engineDone)
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					return
				}
				close(entered)
				if strings.HasPrefix(stage, "first body") {
					w.WriteHeader(http.StatusOK)
					w.(http.Flusher).Flush()
				}
				select {
				case <-r.Context().Done():
				case <-release:
				}
			}))
			defer engine.Close()
			defer unblock()
			setBackend(t, p.soleFacade(), engine.URL, true)
			r := strictHTTPRequest(t)
			r.Header.Del(nodepolicy.StrictAdmissionHeader)
			ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
			defer cancel()
			rec, done := httptest.NewRecorder(), make(chan struct{})
			go func() { defer close(done); p.soleFacade().serveIngress(rec, r.WithContext(ctx), "peer") }()
			waitChan(t, entered, "ordinary terminal ingress transport")
			if len(p.activeRoutingRequests()) != 0 || p.requestSequence != 0 {
				t.Error("destination published an invented origin request")
			}
			if strings.HasSuffix(stage, "cancellation") {
				cancel()
			}
			bounded := false
			select {
			case <-done:
				bounded = true
			case <-time.After(750 * time.Millisecond):
			}
			cancel()
			waitChan(t, done, "ordinary ingress handler cleanup")
			waitChan(t, engineDone, "ordinary ingress upstream cancelled")
			if !bounded {
				t.Fatal("ordinary ingress transport ignored the same request deadline/cancellation")
			}
			if stage == "headers deadline" && rec.Code != http.StatusServiceUnavailable {
				t.Fatalf("silent ingress status=%d body=%q", rec.Code, rec.Body.String())
			}
			ownedAssertHandlerDrained(t, p, nil)
		})
	}
}

func TestUnifiedDispatchIngressPanicCleanup(t *testing.T) {
	for _, strict := range []bool{false, true} {
		name := "ordinary"
		if strict {
			name = "strict"
		}
		t.Run(name, func(t *testing.T) {
			p, _ := strictHTTPFixture(t)
			p.buffers = bufferTestBudget(t, 2<<20, 0)
			engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, continuitySSEEvent+"data: [DONE]\n\n")
			}))
			defer engine.Close()
			setBackend(t, p.soleFacade(), engine.URL, true)
			body := strings.TrimSuffix(strictBody("m"), "}") + `,"stream":true}`
			r := strictHTTPRequestWith(t, body)
			if !strict {
				r.Header.Del(nodepolicy.StrictAdmissionHeader)
			}
			var panicked interface{}
			func() {
				defer func() { panicked = recover() }()
				p.soleFacade().serveIngress(&registryPanicWriter{httptest.NewRecorder()}, r, "peer")
			}()
			if panicked != "registry cleanup fixture" {
				t.Fatalf("panic unwind=%v", panicked)
			}
			bufferAssertTotals(t, p.buffers, 0, 0, 0)
			p.admission.mu.Lock()
			active, queued, tickets, markers := p.admission.active, p.admission.queued, len(p.admission.tickets), len(p.admission.strict.markers)
			p.admission.mu.Unlock()
			want := 0
			if strict {
				want = 1
			}
			if active != want || markers != want || queued != 0 || tickets != 0 || len(p.activeRoutingRequests()) != 0 {
				t.Fatalf("panic cleanup/capacity facts: active=%d markers=%d queued=%d tickets=%d", active, markers, queued, tickets)
			}
		})
	}
}

type routingParserRefusalConn struct {
	net.Conn
	once   sync.Once
	refuse func()
}

func (c *routingParserRefusalConn) Read(p []byte) (int, error) {
	n, err := c.Conn.Read(p)
	if n > 0 {
		c.once.Do(c.refuse)
	}
	return n, err
}

func TestUnifiedDispatchPostcommitModifyResponseRefusal(t *testing.T) {
	test := func(name string, cancelBeforeSink bool) {
		t.Run(name, func(t *testing.T) {
			var upstream, refusals atomic.Int32
			observed := make(chan *routingSpend, 1)
			engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					return
				}
				upstream.Add(1)
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusAccepted)
				_, _ = io.WriteString(w, continuitySSEEvent+"data: [DONE]\n\n")
			}))
			defer engine.Close()
			fallback := newRecordingEngine(t, http.StatusOK)
			discovery := NewDiscovery()
			discovery.AddManual(nodeForModel(t, "a", engine.URL, "m"))
			discovery.AddManual(nodeForModel(t, "b", fallback.URL, "m"))
			p := newTestProxy(ollamaOnlyProfile(t), NewCodec(&recordingWriter{}), discovery, 0)
			p.SetPriority([]string{"a", "b"})
			p.buffers = bufferTestBudget(t, (256<<10)+bufferFixtureRemotePreparationBytes(t, `{"model":"m","stream":true}`), 0)
			transport := p.plainHTTPTransport()
			dial := transport.DialContext
			transport.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
				conn, err := dial(ctx, network, address)
				if err != nil {
					return nil, err
				}
				buffers := inferenceBuffers(ctx)
				if buffers == nil {
					return conn, nil
				}
				return &routingParserRefusalConn{Conn: conn, refuse: func() {
					// The sink retains its real parser borrower. Make only the owner's
					// handle unborrowable, forcing ModifyResponse's explicit error path.
					buffers.mu.Lock()
					parser := buffers.parser
					buffers.mu.Unlock()
					if parser == nil {
						t.Error("missing retained parser handle")
						return
					}
					parser.close()
					if probe := buffers.borrowParserAllowance(); probe != nil {
						probe.close()
						t.Error("parser fixture did not reach the unborrowable-handle branch")
					}
					keys := p.activeRoutingRequests()
					if len(keys) != 1 {
						t.Errorf("parser refusal has %d request owners", len(keys))
					} else {
						owner := p.routingRequestControl(keys[0])
						if cancelBeforeSink {
							owner.sink.mu.Lock()
						}
						observed <- owner
					}
					refusals.Add(1)
				}}, nil
			}
			rec, done := httptest.NewRecorder(), make(chan struct{})
			go func() {
				defer close(done)
				p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)))
			}()
			if cancelBeforeSink {
				var owner *routingSpend
				select {
				case owner = <-observed:
				case <-done:
					t.Fatal("handler returned before the parser refusal barrier")
				case <-time.After(3 * time.Second):
					t.Fatal("timed out waiting for the parser refusal barrier")
				}
				locked := true
				defer func() {
					if locked {
						owner.sink.mu.Unlock()
					}
				}()
				stacks := make([]byte, 1<<20)
				waitForCond(t, time.Second, "budget error waiting for sink admission", func() bool {
					n := runtime.Stack(stacks, true)
					return bytes.Contains(stacks[:n], []byte("(*responseSink).planInterruption"))
				})
				if owner.sink.endingPlanned || !owner.cancel(false) {
					t.Fatal("budget fixture did not establish cancellation before new sink planning")
				}
				owner.mu.Lock()
				committed, cancelled, reason := owner.committed, owner.cancelled, owner.reason
				owner.mu.Unlock()
				if !committed || !cancelled || reason != "cancelled by request" {
					t.Fatalf("budget cancellation winner: committed=%v cancelled=%v reason=%q", committed, cancelled, reason)
				}
				owner.sink.mu.Unlock()
				locked = false
				waitChan(t, done, "cancelled budget error cleanup")
				result := owner.sink.result()
				if result.Reason != "" || result.Synthetic != 0 || rec.Body.Len() != 0 || refusals.Load() != 1 || upstream.Load() != 1 || fallback.hits() != 0 {
					t.Fatalf("cancelled budget error planned output: sink=%+v body=%q refusals=%d upstream=%d fallback=%d", result, rec.Body.String(), refusals.Load(), upstream.Load(), fallback.hits())
				}
				ownedAssertHandlerDrained(t, p, owner)
				return
			}
			waitChan(t, done, "postcommit ModifyResponse refusal cleanup")
			select {
			case owner := <-observed:
				if owner == nil || !owner.committed {
					t.Fatal("ModifyResponse refusal did not follow genuine raw commitment")
				}
			default:
				t.Fatal("no real request owner observed at the parser refusal branch")
			}
			if refusals.Load() != 1 || upstream.Load() != 1 || fallback.hits() != 0 || rec.Code != http.StatusAccepted || rec.Body.String() != string(wantSSEInterruption(streamInterruptReasonBudget)) {
				t.Fatalf("postcommit ModifyResponse refusal: handles=%d upstream=%d fallback=%d status=%d body=%q", refusals.Load(), upstream.Load(), fallback.hits(), rec.Code, rec.Body.String())
			}
			ownedAssertHandlerDrained(t, p, nil)
		})
	}
	test("committed allocation error", false)
	test("cancellation before budget sink admission", true)
}

func TestUnifiedDispatchCommittedPeekTimer(t *testing.T) {
	setForTest(t, &firstBodyTimeout, 50*time.Millisecond)
	setForTest(t, &retryBackoff, []time.Duration(nil))
	test := func(name string, acceptedAttempt int32, cancelAfterCommit, sinkAdmissionBarrier bool) {
		t.Run(name, func(t *testing.T) {
			if acceptedAttempt == maxDispatchAttempts {
				// A sole candidate now waits through the remaining job
				// deadline, including this blocked publication of its peek.
				setForTest(t, &jobDeadline, 200*time.Millisecond)
			}
			entered, send := make(chan struct{}), make(chan struct{})
			var sends atomic.Int32
			engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					return
				}
				if sends.Add(1) < acceptedAttempt {
					w.WriteHeader(http.StatusServiceUnavailable)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusAccepted)
				w.(http.Flusher).Flush()
				close(entered)
				<-send
				_, _ = io.WriteString(w, continuitySSEEvent+"data: [DONE]\n\n")
			}))
			defer engine.Close()
			var sendOnce sync.Once
			unblockEngine := func() { sendOnce.Do(func() { close(send) }) }
			defer unblockEngine()
			fallback := newRecordingEngine(t, http.StatusOK)
			discovery := NewDiscovery()
			discovery.AddManual(nodeForModel(t, "a", engine.URL, "m"))
			if acceptedAttempt == 1 {
				discovery.AddManual(nodeForModel(t, "b", fallback.URL, "m"))
			}
			writer := &recordingWriter{}
			p := newTestProxy(ollamaOnlyProfile(t), NewCodec(writer), discovery, 0)
			p.SetPriority([]string{"a", "b"})
			p.buffers = bufferTestBudget(t, 256<<10+bufferFixtureRemotePreparationBytes(t, `{"model":"m","stream":true}`), 0)
			rec, done := httptest.NewRecorder(), make(chan struct{})
			go func() {
				defer close(done)
				p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)))
			}()
			waitChan(t, entered, "accepted response headers")
			keys := p.activeRoutingRequests()
			if len(keys) != 1 {
				t.Fatalf("registered requests=%d", len(keys))
			}
			owner := p.routingRequestControl(keys[0])
			committed, resume, callbackDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var resumeOnce sync.Once
			unblockCallback := func() { resumeOnce.Do(func() { close(resume) }) }
			defer unblockCallback()
			owner.mu.Lock()
			stop := owner.stopPreOutput
			var blocked atomic.Bool
			owner.stopPreOutput = func() bool {
				if blocked.CompareAndSwap(false, true) {
					defer close(callbackDone)
					close(committed)
					<-resume
				}
				return stop()
			}
			owner.mu.Unlock()
			sinkLocked := false
			unlockSink := func() {
				if sinkLocked {
					owner.sink.mu.Unlock()
					sinkLocked = false
				}
			}
			defer unlockSink()
			if sinkAdmissionBarrier {
				owner.sink.mu.Lock()
				sinkLocked = true
			}
			unblockEngine()
			waitChan(t, committed, "recorded raw commitment before peek publication")
			if sinkAdmissionBarrier {
				stacks := make([]byte, 1<<20)
				waitForCond(t, time.Second, "handler waiting to admit its sink plan", func() bool {
					n := runtime.Stack(stacks, true)
					return bytes.Contains(stacks[:n], []byte("(*responseSink).planInterruptionForAttempt"))
				})
				if owner.sink.endingPlanned {
					t.Fatal("sink barrier did not precede new planning")
				}
			}
			if cancelAfterCommit && !owner.cancel(false) {
				t.Fatal("request cancellation did not win")
			}
			if sinkAdmissionBarrier {
				owner.mu.Lock()
				cancelled, reason := owner.cancelled, owner.reason
				owner.mu.Unlock()
				if !cancelled || reason != "cancelled by request" {
					t.Fatalf("cancellation lost before sink admission: cancelled=%v reason=%q", cancelled, reason)
				}
				unlockSink()
			}
			waitChan(t, done, "handler return before callback release")
			owner.mu.Lock()
			pending, terminal, rawCommitted := owner.firstBodyDone, owner.done, owner.committed
			owner.mu.Unlock()
			if !terminal || !rawCommitted || sends.Load() != acceptedAttempt || fallback.hits() != 0 {
				t.Fatalf("timer changed request winner: terminal=%v committed=%v sends=%d fallback=%d", terminal, rawCommitted, sends.Load(), fallback.hits())
			}
			if pending == nil || p.routingRequestControl(keys[0]) != owner || owner.cancel(false) {
				t.Fatal("pending byte borrower lost terminal owner discovery")
			}
			if _, err := owner.newAttempt(); !errors.Is(err, errRoutingSpend) {
				t.Fatalf("terminal committed request allowed dispatch: %v", err)
			}
			bufferAssertTotals(t, p.buffers, 96<<10, 1, 0)
			p.identityMu.Lock()
			associations := len(p.workloadRequests)
			p.identityMu.Unlock()
			if associations != 0 {
				t.Fatalf("terminal workload associations=%d", associations)
			}
			result := owner.sink.result()
			if cancelAfterCommit {
				if rec.Body.Len() != 0 || result.Reason != "" || result.DeliveryComplete {
					t.Fatalf("cancelled response delivered a terminal body: body=%q sink=%+v", rec.Body.String(), result)
				}
			} else if rec.Code != http.StatusAccepted || rec.Body.String() != string(wantSSEInterruption(streamInterruptReasonIdle)) || result.Reason != streamInterruptReasonIdle || result.ProtocolComplete || !result.DeliveryComplete || !result.ContinuationUnavailable || result.Genuine != 0 {
				t.Fatalf("committed timer bypassed sink: status=%d body=%q sink=%+v", rec.Code, rec.Body.String(), result)
			}
			completed := notificationsOf(t, writer, engines.AddressMethod(p.soleFacade().profile.Name, "proxy/request"))
			if len(completed) != 1 {
				t.Fatalf("completion notifications=%d", len(completed))
			}
			var event RequestEvent
			if err := json.Unmarshal(completed[0], &event); err != nil {
				t.Fatal(err)
			}
			wantReason := streamInterruptReasonIdle
			if cancelAfterCommit {
				wantReason = "cancelled by request"
			}
			if event.Error != wantReason || (!cancelAfterCommit && (event.Status != http.StatusAccepted || event.NodeID != "a")) {
				t.Fatalf("false terminal reason/status: %+v", event)
			}
			unblockCallback()
			waitChan(t, callbackDone, "pure stop callback completion")
			waitChan(t, pending, "peek discard and charged borrower release")
			waitForCond(t, time.Second, "registry cleanup after peek release", func() bool { return len(p.activeRoutingRequests()) == 0 })
			ownedAssertHandlerDrained(t, p, owner)
		})
	}
	test("multiple candidates", 1, false, false)
	test("last permitted candidate", maxDispatchAttempts, false, false)
	test("cancellation after recorded commitment", 1, true, false)
	test("cancellation while sink admission is blocked", 1, true, true)
}

type routingTrackedPeekBody struct {
	entered, release, closed chan struct{}
	err                      error
	reading, overlap         atomic.Bool
	reads, closes            atomic.Int32
	closeOnce                sync.Once
}

func (b *routingTrackedPeekBody) Read(p []byte) (int, error) {
	if b.reads.Add(1) != 1 {
		return 0, io.EOF
	}
	if b.reading.Swap(true) {
		b.overlap.Store(true)
	}
	defer b.reading.Store(false)
	close(b.entered)
	<-b.release
	p[0] = '!'
	return 1, b.err
}

func (b *routingTrackedPeekBody) Close() error {
	if b.reading.Load() {
		b.overlap.Store(true)
	}
	b.closes.Add(1)
	b.closeOnce.Do(func() { close(b.closed) })
	return nil
}

func TestUnifiedDispatchPeekReadErrorAndTimeout(t *testing.T) {
	t.Run("committed timeout closes finished raw IO without another reader", func(t *testing.T) {
		owner := newRoutingRequest(nil)
		a, err := owner.newDispatchAttempt(context.Background(), func() {}, "local")
		if err != nil || !a.acceptResponse() {
			t.Fatalf("attempt setup: %v", err)
		}
		committed, resume := make(chan struct{}), make(chan struct{})
		var once sync.Once
		unblock := func() { once.Do(func() { close(resume) }) }
		defer unblock()
		owner.stopPreOutput = func() bool { close(committed); <-resume; return true }
		upstream := &routingTrackedPeekBody{entered: make(chan struct{}), release: make(chan struct{}), closed: make(chan struct{})}
		close(upstream.release)
		body := &routingResponseBody{ReadCloser: upstream, attempt: a}
		result := make(chan error, 1)
		go func() { _, err := awaitFirstBody(body, 50*time.Millisecond); result <- err }()
		waitChan(t, committed, "raw IO finished before blocked callback")
		if err := <-result; !errors.Is(err, errFirstBodyTimeout) {
			t.Fatalf("committed peek timer=%v", err)
		}
		if upstream.reads.Load() != 1 || upstream.closes.Load() != 1 || upstream.overlap.Load() {
			t.Fatalf("committed timeout raw IO: reads=%d closes=%d overlap=%v", upstream.reads.Load(), upstream.closes.Load(), upstream.overlap.Load())
		}
		owner.mu.Lock()
		pending := owner.firstBodyDone
		owner.mu.Unlock()
		select {
		case <-pending:
			t.Fatal("peek discarded its scratch before callback completion")
		default:
		}
		unblock()
		waitChan(t, pending, "finished callback and peek worker")
	})
	t.Run("raw error survives first byte publication without second read", func(t *testing.T) {
		owner := newRoutingRequest(nil)
		rec := httptest.NewRecorder()
		sink := newResponseSink(rec, nil)
		owner.sink = sink
		a, err := owner.newDispatchAttempt(context.Background(), func() {}, "local")
		if err != nil || !a.acceptResponse() {
			t.Fatalf("attempt setup: %v", err)
		}
		committed, resume := make(chan struct{}), make(chan struct{})
		var once sync.Once
		unblock := func() { once.Do(func() { close(resume) }) }
		defer unblock()
		owner.stopPreOutput = func() bool { close(committed); <-resume; return true }
		upstream := &routingTrackedPeekBody{entered: make(chan struct{}), release: make(chan struct{}), closed: make(chan struct{}), err: io.ErrUnexpectedEOF}
		close(upstream.release)
		body := &routingResponseBody{ReadCloser: upstream, attempt: a}
		type peekResult struct {
			body io.ReadCloser
			err  error
		}
		result := make(chan peekResult, 1)
		go func() { b, err := awaitFirstBody(body, time.Second); result <- peekResult{b, err} }()
		waitChan(t, committed, "raw error commitment before peek publication")
		if !a.requestCommitted() {
			t.Fatal("positive raw read with error did not commit")
		}
		unblock()
		peek := <-result
		if peek.err != nil {
			t.Fatalf("committed raw error rejected before delivery: %v", peek.err)
		}
		budget := newFrameBudget(64 << 20)
		reader, err := newStreamFrameReader(streamFrameConfig{Upstream: peek.body, Protocol: streamProtocolSSE, Budget: budget, Sink: sink})
		if err != nil {
			t.Fatal(err)
		}
		sink.WriteHeader(http.StatusAccepted)
		if _, err := io.Copy(sink, reader); err != nil {
			t.Fatal(err)
		}
		if err := sink.FlushError(); err != nil {
			t.Fatal(err)
		}
		if err := reader.Close(); err != nil {
			t.Fatal(err)
		}
		resultSink := sink.result()
		if rec.Code != http.StatusAccepted || rec.Body.String() != string(wantSSEInterruption(streamInterruptReasonReadFailed)) || resultSink.Reason != streamInterruptReasonReadFailed || resultSink.ProtocolComplete || !resultSink.DeliveryComplete || budget.inUse() != 0 {
			t.Fatalf("raw error did not end through sink: status=%d body=%q sink=%+v bytes=%d", rec.Code, rec.Body.String(), resultSink, budget.inUse())
		}
		if upstream.reads.Load() != 1 || upstream.closes.Load() != 1 || upstream.overlap.Load() {
			t.Fatalf("raw IO lifetime: reads=%d closes=%d overlap=%v", upstream.reads.Load(), upstream.closes.Load(), upstream.overlap.Load())
		}
	})
	t.Run("timeout wins before late raw byte and joins the reader", func(t *testing.T) {
		owner := newRoutingRequest(nil)
		var cancellations atomic.Int32
		a, err := owner.newDispatchAttempt(context.Background(), func() { cancellations.Add(1) }, "local")
		if err != nil || !a.acceptResponse() {
			t.Fatalf("attempt setup: %v", err)
		}
		upstream := &routingTrackedPeekBody{entered: make(chan struct{}), release: make(chan struct{}), closed: make(chan struct{})}
		var once sync.Once
		unblock := func() { once.Do(func() { close(upstream.release) }) }
		defer unblock()
		body := &routingResponseBody{ReadCloser: upstream, attempt: a}
		result := make(chan error, 1)
		go func() { _, err := awaitFirstBody(body, 20*time.Millisecond); result <- err }()
		waitChan(t, upstream.entered, "pending raw first read")
		waitChan(t, upstream.closed, "timeout cancels and closes pending body")
		select {
		case err := <-result:
			t.Fatalf("timeout returned before raw reader joined: %v", err)
		default:
		}
		owner.mu.Lock()
		ending, reason := a.state, a.reason
		owner.mu.Unlock()
		if ending != routingEnding || reason != errFirstBodyTimeout.Error() {
			t.Fatalf("timeout did not record its owner transition: state=%s reason=%q", ending, reason)
		}
		if cancellations.Load() != 0 {
			t.Fatal("failed upstream read was misclassified as requester cancellation")
		}
		unblock()
		if err := <-result; !errors.Is(err, errFirstBodyTimeout) || owner.committed || a.committed {
			t.Fatalf("late raw byte defeated timeout: err=%v committed=%v/%v", err, owner.committed, a.committed)
		}
		a.finished(publicAttemptTerminal{})
		if _, err := owner.newAttempt(); err != nil {
			t.Fatalf("genuine precommit timeout prohibited retry: %v", err)
		}
	})
}

// serveHybridDispatch runs one inference request on the common driver under a
// hybrid policy, prepared as handleHTTP prepares a concrete one. No production
// caller sends a hybrid policy yet.
func serveHybridDispatch(t *testing.T, f *facade, body string, policy dispatchPolicy) (*httptest.ResponseRecorder, *routingSpend) {
	t.Helper()
	p := f.host
	var request struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal([]byte(body), &request); err != nil {
		t.Fatal(err)
	}
	recorder := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	ctl := newRoutingRequest(nil)
	r, buffers := p.ownInferenceBuffers(r)
	owner, err := p.registerRoutingRequest(ctl)
	if err != nil {
		buffers.close()
		t.Fatalf("register routing request: %v", err)
	}
	defer func() {
		buffers.close()
		p.unregisterRoutingRequest(owner)
	}()
	ctl.sink = newResponseSink(recorder, func() { ctl.cancelWithReason("client connection lost") })
	r = r.WithContext(withResponseSink(r.Context(), ctl.sink))
	ctl.dispatch(f, ctl.sink, r, time.Now(), "hybrid-request", []byte(body), request.Model, 0,
		func(reason string) string { return reason }, owner, policy)
	return recorder, ctl
}

func hybridTestPolicy(attempts int, budget time.Duration) dispatchPolicy {
	return dispatchPolicy{maxAttempts: attempts, deadline: time.Now().Add(budget), queueWait: budget, coldWait: budget, hybrid: true}
}

const hybridTestBody = `{"model":"weak","messages":[{"role":"user","content":"hi"}]}`

// hybridPeerFixture serves the weak tier from qualified declared peers. Each
// listed peer answers with its status.
func hybridPeerFixture(t *testing.T, peers map[string]int) *tierFixture {
	t.Helper()
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "q:1b"}}})
	pol.TierPolicy.AllowStronger = false
	nodes := map[string][]string{}
	for id := range peers {
		nodes[id] = []string{"q:1b"}
	}
	fx := newTierFixture(t, ollamaCase(t), pol, nodes, nil)
	f := fx.p.soleFacade()
	for id, status := range peers {
		if status != http.StatusOK {
			e := newRecordingEngine(t, status)
			n := nodeFor(t, id, e.URL)
			n.Models = []string{"q:1b"}
			f.discovery.AddManual(n)
			fx.eng[id] = e
		}
		fx.advertise(t, id, map[string]nodepolicy.AdvertisedRole{"weak": qualifiedRole("q:1b")}, strictCapabilityFor("ollama", "q:1b"))
	}
	return fx
}

func decodeStrictHeader(t *testing.T, h http.Header) nodepolicy.StrictAdmissionRequest {
	t.Helper()
	req, err := parseStrictAdmission(h)
	if err != nil || req == nil {
		t.Fatalf("strict admission header = %v (err %v), want one", h.Values(nodepolicy.StrictAdmissionHeader), err)
	}
	if err := req.Validate(); err != nil {
		t.Fatalf("strict admission header is invalid: %v", err)
	}
	return *req
}

func TestUnifiedDispatchHybridPeerCarriesStrictRequest(t *testing.T) {
	fx := hybridPeerFixture(t, map[string]int{"peer": http.StatusOK})
	recorder, ctl := serveHybridDispatch(t, fx.p.soleFacade(), hybridTestBody, hybridTestPolicy(1, 5*time.Second))
	if recorder.Code != http.StatusOK || fx.eng["peer"].hits() != 1 {
		t.Fatalf("hybrid status = %d, peer hits = %d; want 200 from the peer", recorder.Code, fx.eng["peer"].hits())
	}
	req := decodeStrictHeader(t, fx.eng["peer"].lastHeader())
	if req.Request != ctl.key || ctl.key.OriginNodeID != fx.p.originNodeID || ctl.key.ProxyRunID != fx.p.runID {
		t.Fatalf("strict request key = %+v, want this request's registry key %+v", req.Request, ctl.key)
	}
	if req.Identity != strictRequest().Identity || req.PreOutputBudgetMs > 5000 || req.QueueBudgetMs > req.PreOutputBudgetMs {
		t.Fatalf("strict request = %+v, want the advertised identity within the 5 s budget", req)
	}
	hybridBody := fx.eng["peer"].lastBody()

	// The same candidate under the concrete policy: same destination and body
	// model, and no strict envelope.
	legacy := fx.do(t, "/v1/chat/completions", hybridTestBody)
	if legacy.Code != http.StatusOK || fx.eng["peer"].hits() != 2 {
		t.Fatalf("concrete status = %d, peer hits = %d; want the same peer", legacy.Code, fx.eng["peer"].hits())
	}
	if h := fx.eng["peer"].lastHeader().Values(nodepolicy.StrictAdmissionHeader); len(h) != 0 {
		t.Fatalf("concrete attempt carried a strict envelope: %v", h)
	}
	var hybridModel, concreteModel struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal([]byte(hybridBody), &hybridModel); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(fx.eng["peer"].lastBody()), &concreteModel); err != nil {
		t.Fatal(err)
	}
	if hybridModel.Model != "q:1b" || concreteModel.Model != hybridModel.Model {
		t.Fatalf("forwarded models = %q (hybrid) and %q (concrete), want q:1b for both", hybridModel.Model, concreteModel.Model)
	}
}

func TestUnifiedDispatchHybridSpendsOneAttemptBudget(t *testing.T) {
	test := func(name string, attempts, wantStatus, wantFailedHits, wantServedHits int) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			fx := hybridPeerFixture(t, map[string]int{"a-busy": http.StatusServiceUnavailable, "b-ready": http.StatusOK})
			recorder, ctl := serveHybridDispatch(t, fx.p.soleFacade(), hybridTestBody, hybridTestPolicy(attempts, 5*time.Second))
			busy, ready := fx.eng["a-busy"], fx.eng["b-ready"]
			if recorder.Code != wantStatus || busy.hits() != wantFailedHits || ready.hits() != wantServedHits || ctl.ordinal != uint64(attempts) {
				t.Fatalf("status = %d, hits busy/ready = %d/%d, attempts = %d; want %d, %d/%d, %d",
					recorder.Code, busy.hits(), ready.hits(), ctl.ordinal, wantStatus, wantFailedHits, wantServedHits, attempts)
			}
			if ready.hits() == 0 {
				return
			}
			// The second attempt sees what the first spent of the shared
			// pre-output deadline and allowances, never a fresh budget.
			first, second := decodeStrictHeader(t, busy.lastHeader()), decodeStrictHeader(t, ready.lastHeader())
			if second.PreOutputBudgetMs > first.PreOutputBudgetMs || second.QueueBudgetMs > first.QueueBudgetMs || second.ColdStartBudgetMs > first.ColdStartBudgetMs {
				t.Fatalf("later attempt budgets %+v exceed the earlier %+v", second, first)
			}
		})
	}
	test("one attempt answers with the first failure", 1, http.StatusServiceUnavailable, 1, 0)
	test("two attempts fall back to the next candidate", 2, http.StatusOK, 1, 1)
}

func TestUnifiedDispatchHybridPreOutputDeadline(t *testing.T) {
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "q:1b"}}})
	pol.TierPolicy.AllowStronger = false
	fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"slow": {"q:1b"}}, nil)
	stalled := newStallingEngine(t)
	defer stalled.doRelease()
	n := nodeFor(t, "slow", stalled.URL)
	n.Models = []string{"q:1b"}
	fx.p.soleFacade().discovery.AddManual(n)
	fx.advertise(t, "slow", map[string]nodepolicy.AdvertisedRole{"weak": qualifiedRole("q:1b")}, strictCapabilityFor("ollama", "q:1b"))

	started := time.Now()
	recorder, ctl := serveHybridDispatch(t, fx.p.soleFacade(), hybridTestBody, hybridTestPolicy(3, 300*time.Millisecond))
	if recorder.Code != http.StatusServiceUnavailable || !strings.Contains(recorder.Body.String(), routingDeadlineReason) {
		t.Fatalf("status = %d body = %q, want the pre-output deadline", recorder.Code, recorder.Body.String())
	}
	if elapsed := time.Since(started); elapsed > 3*time.Second {
		t.Fatalf("hybrid request took %v, want the 300 ms budget rather than the concrete job deadline", elapsed)
	}
	if ctl.ordinal != 1 {
		t.Fatalf("attempts = %d, want the stalled one only", ctl.ordinal)
	}
}

func TestUnifiedDispatchHybridPublicRefusalSpendsNothing(t *testing.T) {
	p, accounts, _, _ := egressFixture(t)
	egressGrant(t, p, 1, 5*time.Second)
	profile, ok := profileFor("ollama")
	if !ok {
		t.Fatal("missing ollama profile")
	}
	f := newFacade(p, profile, NewDiscovery(), 0)
	if p.facades == nil {
		p.facades = map[string]*facade{}
	}
	p.facades["ollama"] = f
	var credentialReads atomic.Int32
	publicChatInputs = func(chatAccountSnapshot) (*chatQualification, *publicClient, func(nodepolicy.CredentialRef) (string, bool), chatByteBudget, bool) {
		return accounts.profile, accounts.transport, func(ref nodepolicy.CredentialRef) (string, bool) {
			credentialReads.Add(1)
			return accounts.resolve(ref)
		}, accounts.budget.acquire, true
	}
	t.Cleanup(func() { publicChatInputs = nil })

	policy := hybridTestPolicy(3, 2*time.Second)
	policy.publicAllowed = true
	recorder, ctl := serveHybridDispatch(t, f, hybridTestBody, policy)
	if recorder.Code != http.StatusNotFound {
		t.Fatalf("API-only hybrid status = %d body = %q, want no eligible model", recorder.Code, recorder.Body.String())
	}
	store, err := readAccountStore(accounts.ledger.storePath)
	if err != nil {
		t.Fatal(err)
	}
	if credentialReads.Load() != 0 || len(store.Holds) != 0 || accounts.calls.Load() != 0 || ctl.ordinal != 0 {
		t.Fatalf("refused public selection did work: credential reads = %d, holds = %d, public Do = %d, attempts = %d; want all zero",
			credentialReads.Load(), len(store.Holds), accounts.calls.Load(), ctl.ordinal)
	}
}

func TestUnifiedDispatchHybridPublicUnavailableBeforePreparation(t *testing.T) {
	policy := hybridDispatchPolicy(time.Now(), nodepolicy.Limits{MaxAttempts: 1, PreOutputBudgetSeconds: 2},
		nodepolicy.TierRouting{RemoteAllowed: true})
	if policy.publicAllowed {
		t.Fatal("unavailable public dispatch was selected before account preparation")
	}
}

// hybridSelfProxy is a router whose own Ollama engine serves the weak tier as
// a qualified declared role, with its strict storage in a private directory.
func hybridSelfProxy(t *testing.T, resident bool) (*Proxy, *facade, *recordingEngine) {
	t.Helper()
	p, oll, _ := twoEngineProxy(t, &recordingWriter{})
	e := newRecordingEngine(t, http.StatusOK)
	oll.discovery.AddManual(selfNode("self", oll.profile.FacadePort, "llama3:latest"))
	setBackend(t, oll, e.URL, true)
	p.admission.strict.path = filepath.Join(t.TempDir(), "strict-capacity.json")
	p.admission.strict.storageErr = nil
	installQualified(t, p.admission, admKey{engine: "ollama", model: "llama3:latest"})
	pol := p.admission.currentPolicy()
	pol.Roles = map[string]nodepolicy.Role{"weak": {Engine: "ollama", Model: "llama3:latest"}}
	pol.Admission.MaxConcurrentPerModel = 1
	p.admission.setPolicy(pol)
	if resident {
		p.admission.setResidency(map[string][]string{"ollama": {"llama3:latest"}})
	}
	return p, oll, e
}

const hybridSelfBody = `{"model":"weak","messages":[{"role":"user","content":"hi"}],"max_tokens":16}`

func TestUnifiedDispatchHybridSelfAdmitsStrictly(t *testing.T) {
	p, oll, e := hybridSelfProxy(t, true)
	recorder, ctl := serveHybridDispatch(t, oll, hybridSelfBody, hybridTestPolicy(1, 5*time.Second))
	if recorder.Code != http.StatusOK || e.hits() != 1 || ctl.ordinal != 1 {
		t.Fatalf("status = %d, engine hits = %d, attempts = %d; want one strict local attempt", recorder.Code, e.hits(), ctl.ordinal)
	}
	if h := e.lastHeader().Values(nodepolicy.StrictAdmissionHeader); len(h) != 0 {
		t.Fatalf("local engine received the internal strict envelope: %v", h)
	}
	// Dispatched strict work is uncertain until reconciliation, so its slot
	// stays quarantined rather than released by the attempt's end.
	p.admission.mu.Lock()
	markers := len(p.admission.strict.markers)
	p.admission.mu.Unlock()
	if markers != 1 {
		t.Fatalf("strict capacity markers = %d, want the dispatched attempt quarantined", markers)
	}
}

func TestUnifiedDispatchHybridSelfRefusalFallsBackToPeer(t *testing.T) {
	p, oll, local := hybridSelfProxy(t, false) // Not resident and no cold-start allowance.
	peer := newRecordingEngine(t, http.StatusOK)
	n := nodeFor(t, "peer", peer.URL)
	n.Models = []string{"llama3:latest"}
	n.Roles = map[string]nodepolicy.AdvertisedRole{"weak": qualifiedRole("llama3:latest")}
	n.PolicyRevision = "1"
	n.StrictAdmission = strictCapabilityFor("ollama", "llama3:latest")
	oll.discovery.AddManual(n)
	p.SetPriority([]string{"self", "peer"}) // The tier order leads with this node.

	policy := hybridTestPolicy(2, 5*time.Second)
	policy.coldWait = 0
	recorder, ctl := serveHybridDispatch(t, oll, hybridSelfBody, policy)
	if recorder.Code != http.StatusOK || local.hits() != 0 || peer.hits() != 1 || ctl.ordinal != 2 {
		t.Fatalf("status = %d, local/peer hits = %d/%d, attempts = %d; want the refused self attempt then the peer",
			recorder.Code, local.hits(), peer.hits(), ctl.ordinal)
	}
	decodeStrictHeader(t, peer.lastHeader())
	p.admission.mu.Lock()
	markers := len(p.admission.strict.markers)
	p.admission.mu.Unlock()
	if markers != 0 {
		t.Fatalf("strict capacity markers = %d, want none for an undispatched refusal", markers)
	}
}
