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
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
	"nvpair-shared/performancewire"
)

// The upstream and client barriers distinguish admission, raw commitment and
// actual delivery. Timeouts only bound failed positive waits, never prove that
// an unwanted send did not happen.
type stopPauseEngine struct {
	*httptest.Server
	entered, cancelled chan struct{}
	release            chan struct{}
	hits               atomic.Int32
	once               sync.Once
}

func newStopPauseEngine(t *testing.T, path, media, first string) *stopPauseEngine {
	t.Helper()
	e := &stopPauseEngine{entered: make(chan struct{}), cancelled: make(chan struct{}), release: make(chan struct{})}
	e.Server = httptest.NewServer(fixtureRequestHandler(http.MethodPost, path, func(w http.ResponseWriter, r *http.Request) {
		e.hits.Add(1)
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Errorf("read engine request: %v", err)
			return
		}
		e.once.Do(func() { close(e.entered) })
		if first != "" {
			w.Header().Set("Content-Type", media)
			if _, err := io.WriteString(w, first); err != nil {
				t.Errorf("write first frame: %v", err)
				return
			}
			if err := http.NewResponseController(w).Flush(); err != nil {
				t.Errorf("flush first frame: %v", err)
				return
			}
		}
		select {
		case <-r.Context().Done():
			close(e.cancelled)
		case <-e.release:
			if _, err := io.WriteString(w, "data: [DONE]\n\n"); err != nil {
				t.Errorf("complete stream: %v", err)
			}
		}
	}))
	t.Cleanup(func() { close(e.release); e.Close() })
	return e
}

func TestStopPauseRouting(t *testing.T) {
	committed := func(name, entry, path, media, first string, protocol streamProtocol, requesterCancel bool) {
		t.Run(name, func(t *testing.T) {
			w := &recordingWriter{}
			p := performanceProxyWithCodec(t, "ollama", NewCodec(w))
			e := newStopPauseEngine(t, path, media, first)
			f := p.soleFacade()
			setBackend(t, f, e.URL, true)
			alternate := newRecordingEngine(t, http.StatusOK)
			f.discovery.AddManual(nodeForModel(t, "alternate", alternate.URL, "m"))
			p.SetPriority([]string{"destination", "alternate"})
			strict := entry == "strict local" || entry == "strict peer" || entry == "hybrid"
			if strict {
				configureStrict(t, p.admission, filepath.Join(t.TempDir(), "capacity.json"))
				// Use the qualified identity for performance as well as admission.
			}
			body := `{"model":"m","stream":true}`
			if strict {
				body = strings.TrimSuffix(strictBody("m"), "}") + `,"stream":true}`
			}
			r := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
			ctx, cancel := context.WithCancel(r.Context())
			defer cancel()
			r = r.WithContext(ctx)
			if entry == "strict local" || entry == "strict peer" {
				r.Header = strictHTTPRequest(t).Header
			}
			rec := &strictFirstByteWriter{ResponseRecorder: httptest.NewRecorder(), first: make(chan struct{})}
			done := make(chan struct{})
			if entry == "hybrid" {
				installQualified(t, p.admission, admKey{engine: "ollama", model: "m:latest"})
				pol := p.admission.currentPolicy()
				pol.Roles = map[string]nodepolicy.Role{"weak": {Engine: "ollama", Model: "m:latest"}}
				p.admission.setPolicy(pol)
			}
			go func() {
				defer close(done)
				switch entry {
				case "peer", "strict peer":
					f.serveIngress(rec, r, "peer")
				case "hybrid":
					stopPauseHybrid(t, f, rec, r, strings.Replace(body, `"m"`, `"weak"`, 1), 1)
				default:
					f.handleHTTP(rec, r)
				}
			}()
			waitChan(t, rec.first, "first frame delivered to client")
			if e.hits.Load() != 1 {
				waitChan(t, done, "refused handler result")
				t.Fatalf("first client bytes were a refusal: status=%d body=%s", rec.Code, rec.Body)
			}
			if requesterCancel {
				cancel()
			}
			p.admission.setAvailability(nodepolicy.Draining, true)
			waitChan(t, e.cancelled, "engine execution cancellation")
			waitChan(t, done, "interrupted handler cleanup")
			if !requesterCancel && ctx.Err() != nil {
				t.Fatalf("Pause cancelled requester delivery: %v", ctx.Err())
			}
			if rec.Code != http.StatusOK || e.hits.Load() != 1 || alternate.hits() != 0 {
				t.Fatalf("status=%d local/alternate sends=%d/%d", rec.Code, e.hits.Load(), alternate.hits())
			}
			if !requesterCancel {
				assertContinuityError(t, protocol, rec.Body.Bytes(), streamInterruptReasonNodePaused)
			}
			want := performancewire.OutcomeFailed
			if requesterCancel {
				want = performancewire.OutcomeCancelled
			}
			if sample := performanceOne(t, p); sample.Outcome != want || sample.ProtocolComplete == nil || *sample.ProtocolComplete {
				t.Fatalf("performance=%+v, want %s and incomplete protocol", sample.Measurement, want)
			}
			if entry == "local" || entry == "strict local" || entry == "hybrid" {
				facts := nativeOnlySource(t, p)
				state, cause := factFailed, factNodePaused
				if requesterCancel {
					state, cause = factCancelled, factClientGone
				}
				if routingFactState(facts.disposition&255) != state || routingFactReason(facts.disposition>>8&255) != cause || facts.served != 0 {
					t.Fatalf("request facts=%+v, want state=%d cause=%d", facts, state, cause)
				}
				if routingFactReason(facts.attempts[0].disposition>>8&255) != cause {
					t.Fatalf("attempt cause=%d, want %d", facts.attempts[0].disposition>>8&255, cause)
				}
			}
			if entry == "local" {
				fx := &continuityFixture{p: p, w: w, nodeID: "destination"}
				if requesterCancel {
					fx.terminal(t, "cancelled", "")
				} else {
					fx.terminal(t, "failed", streamInterruptReasonNodePaused)
				}
			}
			if strict {
				if active := p.admission.currentActive(); active != 1 {
					t.Fatalf("SC-a ending released uncertain strict capacity: %d", active)
				}
				if slots := readCapacity(t, p.admission.strict.path).Slots; len(slots) != 1 {
					t.Fatalf("uncertain strict markers=%v", slots)
				}
				if err := p.admission.reconcileStrictCapacity("ollama", "m", strictQualifiedComplete); err == nil {
					t.Fatal("synthetic completion accepted as idle proof")
				}
				if err := p.admission.reconcileStrictCapacity("ollama", "m", strictAuthoritativeIdle); err != nil {
					t.Fatal(err)
				}
			}
			if active := p.admission.currentActive(); active != 0 {
				t.Fatalf("capacity retained after cleanup/proof: %d", active)
			}
			bufferAssertTotals(t, p.buffers, 0, 0, 0)
		})
	}
	committed("local concrete committed SSE", "local", routeChatCompletions, "text/event-stream", continuitySSEEvent, streamProtocolSSE, false)
	committed("local concrete committed NDJSON", "local", "/api/chat", "application/x-ndjson", continuityNDJSONEvent, streamProtocolNDJSON, false)
	committed("terminal peer ordinary committed SSE", "peer", routeChatCompletions, "text/event-stream", continuitySSEEvent, streamProtocolSSE, false)
	committed("terminal peer ordinary committed NDJSON", "peer", "/api/chat", "application/x-ndjson", continuityNDJSONEvent, streamProtocolNDJSON, false)
	committed("local strict committed SSE", "strict local", routeChatCompletions, "text/event-stream", continuitySSEEvent, streamProtocolSSE, false)
	committed("terminal peer strict committed SSE", "strict peer", routeChatCompletions, "text/event-stream", continuitySSEEvent, streamProtocolSSE, false)
	committed("local hybrid committed SSE", "hybrid", routeChatCompletions, "text/event-stream", continuitySSEEvent, streamProtocolSSE, false)
	committed("requester cancellation remains cancelled", "local", routeChatCompletions, "text/event-stream", continuitySSEEvent, streamProtocolSSE, true)

	precommit := func(name string, hybrid, alternate, peer bool) {
		t.Run(name, func(t *testing.T) {
			e := newStopPauseEngine(t, routeChatCompletions, "", "")
			w := &recordingWriter{}
			p := performanceProxyWithCodec(t, "ollama", NewCodec(w))
			f := p.soleFacade()
			setBackend(t, f, e.URL, true)
			other := newRecordingEngine(t, http.StatusOK)
			if hybrid {
				configureStrict(t, p.admission, filepath.Join(t.TempDir(), "capacity.json"))
				installQualified(t, p.admission, admKey{engine: "ollama", model: "m:latest"})
				pol := p.admission.currentPolicy()
				pol.Roles = map[string]nodepolicy.Role{"weak": {Engine: "ollama", Model: "m:latest"}}
				p.admission.setPolicy(pol)
			}
			if alternate {
				n := nodeForModel(t, "alternate", other.URL, "m")
				if hybrid {
					n.Roles = map[string]nodepolicy.AdvertisedRole{"weak": qualifiedRole("m:latest")}
					n.PolicyRevision = "1"
					n.StrictAdmission = strictCapabilityFor("ollama", "m:latest")
				}
				f.discovery.AddManual(n)
				if hybrid {
					pinRecordingPairPeer(t, f, "alternate", other)
				}
			}
			p.SetPriority([]string{"destination", "alternate"})
			rec := httptest.NewRecorder()
			done := make(chan struct{})
			go func() {
				defer close(done)
				r := httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(strictBody("m")))
				if peer {
					if hybrid {
						r.Header = strictHTTPRequest(t).Header
					}
					f.serveIngress(rec, r, "peer")
				} else if hybrid {
					stopPauseHybrid(t, f, rec, r, strictBody("weak"), 2)
				} else {
					f.handleHTTP(rec, r)
				}
			}()
			waitChan(t, e.entered, "local precommit execution")
			p.admission.setAvailability(nodepolicy.Draining, true)
			waitChan(t, e.cancelled, "withdrawn local execution")
			waitChan(t, done, "precommit request result")
			wantOther := 0
			if alternate && !peer {
				wantOther = 1
			}
			if other.hits() != wantOther || e.hits.Load() != 1 {
				t.Fatalf("local/alternate sends=%d/%d, want 1/%d; status=%d body=%s", e.hits.Load(), other.hits(), wantOther, rec.Code, rec.Body)
			}
			if (rec.Code == http.StatusOK) != (wantOther == 1) {
				t.Fatalf("status=%d, want success=%t; body=%s", rec.Code, wantOther == 1, rec.Body)
			}
			if !peer {
				facts := nativeOnlySource(t, p)
				want := factFailed
				if wantOther == 1 {
					want = factCompleted
				}
				if routingFactState(facts.disposition&255) != want {
					t.Fatalf("request disposition=%d, want %d", facts.disposition, want)
				}
			}
			if hybrid {
				if p.admission.currentActive() != 1 {
					t.Fatal("precommit cancelled strict dispatch lost uncertainty hold")
				}
				if err := p.admission.reconcileStrictCapacity("ollama", "m", strictAuthoritativeIdle); err != nil {
					t.Fatal(err)
				}
			}
			bufferAssertTotals(t, p.buffers, 0, 0, 0)
		})
	}
	precommit("local concrete precommit alternate", false, true, false)
	precommit("local concrete precommit no alternate", false, false, false)
	precommit("local hybrid precommit alternate", true, true, false)
	precommit("local hybrid precommit no alternate", true, false, false)
	precommit("terminal peer ordinary refuses with alternate", false, true, true)
	precommit("terminal peer strict refuses with alternate", true, true, true)

	t.Run("preparation rechecks fence before wake or forward", stopPausePreparation)
	t.Run("strict queued request rechecks fence", stopPauseStrictQueue)
	t.Run("strict cancellation registration cannot miss withdrawal", stopPauseRegistration)
	t.Run("origin pause preserves remote execution", stopPauseRemote)
	t.Run("destination failure remains failed at origin", stopPausePeerFailure)
	t.Run("terminal peer headers do not commit before raw content", stopPausePeerHeaders)
	t.Run("raw and nonstream get no synthetic framing", stopPauseRaw)
	t.Run("queued and waking ingress obeys fence", stopPauseWaitingIngress)
	t.Run("strict cleanup precedes reconciliation release", stopPauseStrictCleanup)
	t.Run("late fact attachment preserves typed withdrawal", stopPauseLateFacts)
	t.Run("raw commitment before stream setup keeps failure delivery", stopPauseCommitBeforeSetup)
}

func stopPausePeerHeaders(t *testing.T) {
	test := func(name, path, media, complete string) {
		t.Run(name, func(t *testing.T) {
			engineCancelled := make(chan struct{})
			var engineHits atomic.Int32
			peer := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				engineHits.Add(1)
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					t.Errorf("read engine request: %v", err)
					return
				}
				w.Header().Set("Content-Type", media)
				if err := http.NewResponseController(w).Flush(); err != nil {
					t.Errorf("flush engine headers: %v", err)
					return
				}
				// No engine body byte can exist before withdrawal.
				<-r.Context().Done()
				close(engineCancelled)
			}))
			peer.directHealthy.Store(true)
			performanceIdentify(peer.p, "origin")
			performanceIdentify(peer.peer, "host-peer")
			w := &recordingWriter{}
			peer.p.codec = NewCodec(w)

			readStarted, releaseRead := make(chan struct{}), make(chan struct{})
			var releaseReadOnce sync.Once
			unblockRead := func() { releaseReadOnce.Do(func() { close(releaseRead) }) }
			t.Cleanup(unblockRead)
			transport := newProxyTransport()
			t.Cleanup(transport.CloseIdleConnections)
			// Keep real HTTP to the fake engine, but stop the raw body read
			// after flushed headers have crossed the destination transport.
			peer.peer.plainHTTPTransport().RegisterProtocol("http", publicTestRoundTrip(func(r *http.Request) (*http.Response, error) {
				resp, err := transport.RoundTrip(r)
				if err == nil {
					resp.Body = &bindingBodyReadBarrier{ReadCloser: resp.Body, entered: readStarted, release: releaseRead}
				}
				return resp, err
			}))
			destinationSink := make(chan *responseSink, 1)
			destinationDone := make(chan struct{})
			ingress := peer.server.Config.Handler
			peer.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(destinationDone)
				sink := newResponseSink(w, nil)
				destinationSink <- sink
				ingress.ServeHTTP(sink, r)
			})

			alternateEntered, releaseAlternate := make(chan struct{}), make(chan struct{})
			var alternateHits atomic.Int32
			var alternateOnce sync.Once
			unblockAlternate := func() { alternateOnce.Do(func() { close(releaseAlternate) }) }
			alternate := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				alternateHits.Add(1)
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					t.Errorf("read alternate request: %v", err)
					return
				}
				close(alternateEntered)
				select {
				case <-releaseAlternate:
					w.Header().Set("Content-Type", media)
					if _, err := io.WriteString(w, complete); err != nil {
						t.Errorf("complete alternate stream: %v", err)
					}
				case <-r.Context().Done():
				}
			}))
			t.Cleanup(alternate.Close)
			t.Cleanup(unblockAlternate)
			peer.p.soleFacade().discovery.AddManual(nodeForModel(t, "alternate", alternate.URL, "fixture-model"))
			peer.p.SetPriority([]string{"host-peer", "alternate"})
			rec := &strictFirstByteWriter{ResponseRecorder: httptest.NewRecorder(), first: make(chan struct{})}
			done := make(chan struct{})
			go func() {
				defer close(done)
				peer.p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"model":"fixture-model","stream":true}`)))
			}()
			defer func() {
				unblockRead()
				unblockAlternate()
				waitChan(t, done, "origin cleanup")
			}()
			waitChan(t, readStarted, "destination raw read after engine headers")
			sink := <-destinationSink
			if status := sink.result().Status; status != 0 {
				t.Errorf("destination published status %d before any raw content", status)
			}
			peer.peer.admission.setAvailability(nodepolicy.Paused, true)
			waitChan(t, engineCancelled, "engine cancellation before raw read release")
			unblockRead()
			select {
			case <-alternateEntered:
				// The alternate is dispatched but also has no body yet. Neither
				// the refused destination nor its SC-a control bytes may commit.
				peer.p.identityMu.Lock()
				var owner *routingSpend
				for _, request := range peer.p.requests {
					owner = request.owner
				}
				peer.p.identityMu.Unlock()
				if owner == nil {
					t.Fatal("missing active origin request")
				}
				owner.mu.Lock()
				committed := owner.committed
				owner.mu.Unlock()
				if committed {
					t.Error("origin committed before alternate raw content")
				}
				select {
				case <-rec.first:
					t.Error("origin delivered body before alternate raw content")
				default:
				}
			case <-done:
				t.Errorf("origin ended without dispatching eligible alternate: status=%d body=%q", rec.Code, rec.Body.String())
			case <-time.After(5 * time.Second):
				t.Fatal("neither alternate dispatch nor origin termination")
			}
			unblockAlternate()
			waitChan(t, done, "completed alternate result")
			waitChan(t, destinationDone, "terminal destination cleanup")
			if rec.Code != http.StatusOK || rec.Body.String() != complete || engineHits.Load() != 1 || alternateHits.Load() != 1 {
				t.Fatalf("status=%d engine/alternate sends=%d/%d body=%q", rec.Code, engineHits.Load(), alternateHits.Load(), rec.Body.String())
			}
			if status := sink.result().Status; status != http.StatusServiceUnavailable {
				t.Fatalf("destination status=%d, want retryable refusal", status)
			}
			fx := &continuityFixture{p: peer.p, w: w, nodeID: "alternate"}
			fx.terminal(t, "completed", "")
			facts := nativeOnlySource(t, peer.p)
			if routingFactState(facts.disposition&255) != factCompleted || facts.total != 2 || facts.served != 2 || facts.attempts[0].flags&factFirstByteKnown != 0 || facts.attempts[1].flags&factFirstByteKnown == 0 {
				t.Fatalf("origin state=%d attempts=%d served=%d attempt flags=%d/%d, want completed alternate only", facts.disposition, facts.total, facts.served, facts.attempts[0].flags, facts.attempts[1].flags)
			}
			if sample := performanceOne(t, peer.peer); sample.Outcome != performancewire.OutcomeFailed {
				t.Fatalf("withdrawn destination outcome=%s", sample.Outcome)
			}
			if active := peer.peer.admission.currentActive(); active != 0 {
				t.Fatalf("destination retained admission after raw read cleanup: %d", active)
			}
			bufferAssertTotals(t, peer.peer.buffers, 0, 0, 0)
		})
	}
	test("SSE", routeChatCompletions, "text/event-stream", continuitySSEEvent+"data: [DONE]\n\n")
	test("NDJSON", "/api/chat", "application/x-ndjson", continuityNDJSONEvent+"{\"done\":true}\n")
}

func stopPauseHybrid(t *testing.T, f *facade, w http.ResponseWriter, r *http.Request, body string, attempts int) {
	t.Helper()
	p := f.host
	setReplayBody(r, []byte(body))
	r, buffers := p.ownInferenceBuffers(r)
	defer buffers.close()
	bodyBytes, _, err := bufferInferenceBody(w, r, p.requestBodyLimit())
	if err != nil {
		t.Error(err)
		return
	}
	ctl := newRoutingRequest(nil)
	owner, err := p.registerRoutingRequest(ctl)
	if err != nil {
		t.Error(err)
		return
	}
	defer p.unregisterRoutingRequest(owner)
	p.attachRoutingFacts(owner, time.Now())
	ctl.sink = newResponseSink(w, nil)
	r = r.WithContext(withResponseSink(r.Context(), ctl.sink))
	ctl.dispatch(f, ctl.sink, r, time.Now(), owner.key.RequestID, bodyBytes, "weak", 0, func(reason string) string { return reason }, owner, hybridTestPolicy(attempts, 3*time.Second))
}

func stopPausePreparation(t *testing.T) {
	h := newAdmHarness(t)
	entered, release := make(chan struct{}), make(chan struct{})
	done := make(chan admissionDecision, 1)
	go func() {
		done <- h.a.admit(t.Context(), admissionRequest{engine: "ollama", model: "m", waitCap: -1,
			prepare: func(context.Context, nodepolicy.Profile, bool, uint64) error { close(entered); <-release; return nil },
		})
	}()
	waitChan(t, entered, "preparation outside admission lock")
	h.a.setAvailability(nodepolicy.Paused, true)
	close(release)
	select {
	case decision := <-done:
		mustReject(t, decision, nodepolicy.RejectPaused)
	case <-time.After(3 * time.Second):
		t.Fatal("preparation did not finish after fence")
	}
	if len(h.notes(nodepolicy.NotifyAdmissionWake)) != 0 || len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 0 || h.a.currentActive() != 0 {
		t.Fatal("fenced preparation caused execution effects")
	}
}

func stopPauseStrictQueue(t *testing.T) {
	h := newStrictHarness(t)
	first := mustStrictTicket(t, h.a)
	defer first.Finish(strictNotDispatched)
	c := useStrictClock(h.a)
	done := asyncStrict(h.a, t.Context(), strictRequest())
	awaitStrictTimer(t, c)
	h.a.setAvailability(nodepolicy.Paused, true)
	result := awaitStrict(t, done)
	if result.ticket != nil || !errors.Is(result.err, errStrictUnavailable) {
		t.Fatalf("queued admission=%v, error=%v", result.ticket, result.err)
	}
	if len(h.notes(nodepolicy.NotifyAdmissionWake)) != 0 || len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 0 {
		t.Fatal("fenced strict queue caused engine effects")
	}
}

func stopPauseRegistration(t *testing.T) {
	h := newStrictHarness(t)
	ticket := mustStrictTicket(t, h.a).(*strictTicket)
	h.a.setAvailability(nodepolicy.Draining, true)
	h.a.setAvailability(nodepolicy.Available, false)
	ctx, cancel := context.WithCancelCause(t.Context())
	ticket.base.setCancel(func() { cancel(executionInterruption{}) })
	if !executionWasWithdrawn(ctx) {
		t.Fatal("late cancellation registration lost accepted withdrawal")
	}
	if err := ticket.BeforeStrictDispatch(); !errors.Is(err, errStrictUnavailable) {
		t.Fatalf("withdrawn ticket became executable after Resume: %v", err)
	}
	ticket.Finish(strictNotDispatched)
	if h.a.currentActive() != 0 {
		t.Fatal("never-dispatched ticket retained capacity")
	}
}

func stopPauseRemote(t *testing.T) {
	e := newStopPauseEngine(t, routeChatCompletions, "text/event-stream", continuitySSEEvent)
	fx := newContinuityFixture(t, e.URL, time.Second)
	rec := &strictFirstByteWriter{ResponseRecorder: httptest.NewRecorder(), first: make(chan struct{})}
	done := make(chan struct{})
	go func() {
		defer close(done)
		fx.p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)))
	}()
	waitChan(t, rec.first, "remote response delivery")
	fx.p.admission.setAvailability(nodepolicy.Paused, true)
	// Positive remote completion, rather than silence after cancellation.
	e.release <- struct{}{}
	waitChan(t, done, "remote execution completion through paused origin")
	if !bytes.HasSuffix(rec.Body.Bytes(), []byte("data: [DONE]\n\n")) || e.hits.Load() != 1 {
		t.Fatalf("remote response=%s sends=%d", rec.Body, e.hits.Load())
	}
	fx.terminal(t, "completed", "")
}

func stopPausePeerFailure(t *testing.T) {
	e := newStopPauseEngine(t, routeChatCompletions, "text/event-stream", continuitySSEEvent)
	peer := newProxyReverseFixture(t, "ollama", e.Config.Handler)
	peer.directHealthy.Store(true)
	w := &recordingWriter{}
	peer.p.codec = NewCodec(w)
	doneDestination := make(chan struct{})
	ingress := peer.server.Config.Handler
	peer.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(doneDestination)
		ingress.ServeHTTP(w, r)
	})
	fx := &continuityFixture{p: peer.p, w: w, nodeID: "host-peer"}
	rec := &strictFirstByteWriter{ResponseRecorder: httptest.NewRecorder(), first: make(chan struct{})}
	done := make(chan struct{})
	go func() {
		defer close(done)
		fx.p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"fixture-model","stream":true}`)))
	}()
	waitChan(t, rec.first, "peer stream at origin client")
	peer.peer.admission.setAvailability(nodepolicy.Paused, true)
	waitChan(t, done, "origin failed stream result")
	waitChan(t, doneDestination, "destination cleanup")
	if !bytes.HasSuffix(rec.Body.Bytes(), []byte("data: [DONE]\n\n")) {
		t.Fatalf("peer ending=%q", rec.Body.String())
	}
	assertContinuityError(t, streamProtocolSSE, rec.Body.Bytes(), streamInterruptReasonNodePaused)
	fx.terminal(t, "failed", streamInterruptReasonNodePaused)
	facts := nativeOnlySource(t, fx.p)
	if routingFactState(facts.disposition&255) != factFailed || facts.served != 0 {
		t.Fatalf("destination's synthetic terminal looked successful: %+v", facts)
	}
}

func stopPauseRaw(t *testing.T) {
	test := func(name, body, media, first string) {
		t.Run(name, func(t *testing.T) {
			e := newStopPauseEngine(t, routeChatCompletions, media, first)
			w := &recordingWriter{}
			p := performanceProxyWithCodec(t, "ollama", NewCodec(w))
			setBackend(t, p.soleFacade(), e.URL, true)
			rec := &strictFirstByteWriter{ResponseRecorder: httptest.NewRecorder(), first: make(chan struct{})}
			done := make(chan struct{})
			go func() {
				defer close(done)
				p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(body)))
			}()
			waitChan(t, rec.first, "raw client bytes")
			p.admission.setAvailability(nodepolicy.Paused, true)
			waitChan(t, e.cancelled, "raw execution cancellation")
			waitChan(t, done, "raw handler cleanup")
			if rec.Body.Len() == 0 || !bytes.HasPrefix([]byte(first), rec.Body.Bytes()) {
				t.Fatalf("raw response gained synthetic bytes: %q", rec.Body.String())
			}
			if sample := performanceOne(t, p); sample.Outcome != performancewire.OutcomeFailed {
				t.Fatalf("raw performance=%+v", sample.Measurement)
			}
			fx := &continuityFixture{p: p, w: w, nodeID: "destination"}
			fx.terminal(t, "failed", streamInterruptReasonNodePaused)
			if facts := nativeOnlySource(t, p); routingFactState(facts.disposition&255) != factFailed || facts.served != 0 {
				t.Fatal("raw withdrawal completed the origin facts")
			}
		})
	}
	test("nonstream JSON", `{"model":"m","stream":false}`, "application/json", `{"partial":`)
	test("unsupported protocol", `{"model":"m"}`, "application/octet-stream", "raw-partial")
}

func stopPauseWaitingIngress(t *testing.T) {
	test := func(name string, peer, waking bool) {
		t.Run(name, func(t *testing.T) {
			p := performanceProxy(t, "ollama")
			e := newRecordingEngine(t, http.StatusOK)
			setBackend(t, p.soleFacade(), e.URL, !waking)
			entered := make(chan struct{})
			var once sync.Once
			var wakes atomic.Int32
			p.admission.notify = func(method string, params any) {
				if method == nodepolicy.NotifyAdmissionWake {
					wakes.Add(1)
					if waking {
						once.Do(func() { close(entered) })
					}
				}
				if state, ok := params.(nodepolicy.AdmissionState); ok && !waking && state.Queued == 1 {
					once.Do(func() { close(entered) })
				}
			}
			pol := p.admission.currentPolicy()
			pol.Admission.MaxConcurrentPerModel = 1
			p.admission.setPolicy(pol)
			if waking {
				p.admission.setIntent(map[string]bool{"ollama": true})
			} else {
				blocker := mustAdmit(t, p.admission.admit(t.Context(), admissionRequest{engine: "ollama", model: "m", waitCap: -1}))
				defer blocker.release()
				p.admission.mu.Lock()
				p.admission.stateSentAt = time.Time{}
				p.admission.mu.Unlock()
			}
			rec, done := httptest.NewRecorder(), make(chan struct{})
			serve := func(w http.ResponseWriter) {
				r := httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(strictBody("m")))
				if peer {
					p.soleFacade().serveIngress(w, r, "peer")
				} else {
					p.soleFacade().handleHTTP(w, r)
				}
			}
			go func() { defer close(done); serve(rec) }()
			waitChan(t, entered, "queued or waking admission")
			p.admission.setAvailability(nodepolicy.Paused, true)
			waitChan(t, done, "fenced ingress result")
			if rec.Code != http.StatusServiceUnavailable || e.hits() != 0 {
				t.Fatalf("fenced result status=%d engine sends=%d", rec.Code, e.hits())
			}
			// A completed request after the fence proves rejection, rather than
			// relying on an interval with no observed engine traffic.
			serve(httptest.NewRecorder())
			wantWake := int32(0)
			if waking {
				wantWake = 1
			}
			if e.hits() != 0 || wakes.Load() != wantWake {
				t.Fatalf("post-fence sends=%d wakes=%d, want 0/%d", e.hits(), wakes.Load(), wantWake)
			}
		})
	}
	test("local queued", false, false)
	test("terminal peer queued", true, false)
	test("local waking", false, true)
	test("terminal peer waking", true, true)
}

func stopPauseStrictCleanup(t *testing.T) {
	p, _ := strictHTTPFixture(t)
	upstream := installBlockingFirstBody(t, p, http.StatusOK, false)
	rec, done := httptest.NewRecorder(), make(chan struct{})
	go func() {
		defer close(done)
		p.soleFacade().serveIngress(rec, strictHTTPRequest(t), "peer")
	}()
	defer func() { upstream.unblockClose(); upstream.unblockRead(); waitChan(t, done, "released strict handler") }()
	waitChan(t, upstream.readStarted, "strict upstream read")
	account := <-upstream.account
	p.admission.setAvailability(nodepolicy.Paused, true)
	waitChan(t, upstream.closeStarted, "strict body Close")
	if err := p.admission.reconcileStrictCapacity("ollama", "m", strictAuthoritativeIdle); err != nil {
		t.Fatal(err)
	}
	if p.admission.currentActive() != 1 {
		t.Fatal("reconciliation released a live reader's ticket")
	}
	upstream.unblockClose()
	select {
	case <-done:
	case <-time.After(routingShutdownGrace + time.Second):
		t.Fatal("strict HTTP return exceeded the reader cleanup bound")
	}
	if rec.Code != http.StatusServiceUnavailable || p.admission.currentActive() != 1 || upstream.requests.Load() != 1 {
		t.Fatalf("strict pending result status=%d active=%d sends=%d", rec.Code, p.admission.currentActive(), upstream.requests.Load())
	}
	if account.pendingBody == nil {
		t.Fatal("strict return lost pending reader ownership")
	}
	p.buffers.mu.Lock()
	retained := account.scope.used
	p.buffers.mu.Unlock()
	if retained == 0 {
		t.Fatal("strict return released the pending reader's buffers")
	}
	upstream.unblockRead()
	waitChan(t, account.pendingBody, "actual reader and Close join")
	waitForCond(t, time.Second, "strict ticket cleanup after actual join", func() bool {
		p.admission.mu.Lock()
		defer p.admission.mu.Unlock()
		return len(p.admission.tickets) == 0
	})
	if p.admission.currentActive() != 1 {
		t.Fatal("reader cleanup alone released uncertain execution capacity")
	}
	if err := p.admission.reconcileStrictCapacity("ollama", "m", strictAuthoritativeIdle); err != nil {
		t.Fatal(err)
	}
	if p.admission.currentActive() != 0 {
		t.Fatal("positive idle proof failed to release cleaned capacity")
	}
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
}

func stopPauseLateFacts(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	r := newRoutingRequest(nil)
	r.mu.Lock()
	r.nativeCancelLocked(factNodePaused)
	r.mu.Unlock()
	owner, err := p.registerRoutingRequest(r)
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(owner)
	p.attachRoutingFacts(owner, time.Now())
	if facts := nativeCopy(t, r); routingFactState(facts.disposition&255) != factFailed || routingFactReason(facts.disposition>>8&255) != factNodePaused {
		t.Fatalf("late attachment lost failed withdrawal: disposition=%d", facts.disposition)
	}
	r.observeIngress(routingIngressResult{factCompleted, factReasonNone, http.StatusOK})
	if facts := nativeCopy(t, r); routingFactState(facts.disposition&255) != factFailed {
		t.Fatal("generic completion overwrote authoritative withdrawal")
	}
}

func stopPauseCommitBeforeSetup(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	row := nativeRegistered(t, p, time.Now())
	r := row.owner
	ctx, cancel := r.attemptContext(t.Context())
	defer cancel()
	a, err := r.newDispatchAttempt(ctx, cancel, "self", destinationDescriptor("m", "self", "ollama", false))
	if err != nil {
		t.Fatal(err)
	}
	if !a.acceptResponse() || !a.commit() {
		t.Fatal("raw commitment refused")
	}
	rec := httptest.NewRecorder()
	sink := newResponseSink(rec, func() { t.Error("Pause cancelled client delivery") })
	sink.bindExecution(ctx)
	r.abortLocalIf(a)
	body := sink.interruptedBodyForAttempt(streamProtocolSSE, streamInterruptReasonBudget, a)
	if body == nil {
		t.Fatal("availability withdrawal suppressed the committed failure ending")
	}
	data, err := io.ReadAll(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := body.Close(); err != nil {
		t.Fatal(err)
	}
	assertContinuityError(t, streamProtocolSSE, data, streamInterruptReasonNodePaused)
	rawSink := newResponseSink(httptest.NewRecorder(), nil)
	rawSink.bindExecution(ctx)
	raw := rawSink.interruptedBodyForAttempt(streamProtocolUnknown, streamInterruptReasonBudget, a)
	if raw == nil {
		t.Fatal("raw interruption was not admitted")
	}
	if data, err := io.ReadAll(raw); len(data) != 0 || !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("raw commit/setup ending emitted bytes=%q or clean EOF: %v", data, err)
	}
	if err := raw.Close(); err != nil {
		t.Fatal(err)
	}
	if facts := nativeCopy(t, r); routingFactState(facts.disposition&255) != factFailed || routingFactReason(facts.attempts[0].disposition>>8&255) != factNodePaused {
		t.Fatalf("commit/setup interruption lost typed failure: %d/%d", facts.disposition, facts.attempts[0].disposition)
	}
}
