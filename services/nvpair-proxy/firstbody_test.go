// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
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
)

// Commit is the first byte of response body, not the response headers, because
// headers can mean nothing more than "accepted" (spec §5.2 for the engine
// measurements behind that). These bodies pin the boundary: what fails over,
// what commits, what terminates, and what is left alone.
//
// firstBodyTimeout stands in at 200ms for its production 120s throughout, so a
// body that drives a node to its five-dispatch budget costs a second of wall
// clock rather than ten minutes.
const testFirstBodyTimeout = 200 * time.Millisecond

// Read and Close have independent completion barriers. Returning HTTP must not
// mistake either one for the other, or call the raw Close again on error.
type blockingFirstBody struct {
	readStarted, closeStarted chan struct{}
	readDone, closeDone       chan struct{}
	releaseRead, releaseClose chan struct{}
	readOnce, closeOnce       sync.Once
	reads, closes, requests   atomic.Int32
	account                   chan *ownedInferenceBuffers
}

func (b *blockingFirstBody) Read(p []byte) (int, error) {
	if b.reads.Add(1) != 1 {
		return 0, io.EOF
	}
	close(b.readStarted)
	<-b.releaseRead
	p[0] = '!'
	close(b.readDone)
	return 1, nil
}

func (b *blockingFirstBody) Close() error {
	if b.closes.Add(1) != 1 {
		return nil
	}
	close(b.closeStarted)
	<-b.releaseClose
	close(b.closeDone)
	return nil
}

func (b *blockingFirstBody) unblockRead()  { b.readOnce.Do(func() { close(b.releaseRead) }) }
func (b *blockingFirstBody) unblockClose() { b.closeOnce.Do(func() { close(b.releaseClose) }) }

func installBlockingFirstBody(t *testing.T, p *Proxy, status int, retireParser bool) *blockingFirstBody {
	t.Helper()
	b := &blockingFirstBody{
		readStarted: make(chan struct{}), closeStarted: make(chan struct{}),
		readDone: make(chan struct{}), closeDone: make(chan struct{}),
		releaseRead: make(chan struct{}), releaseClose: make(chan struct{}),
		account: make(chan *ownedInferenceBuffers, 1),
	}
	// Exercise the complete production ReverseProxy path with an arbitrary
	// ReadCloser, using Transport's existing in-memory protocol seam. No socket
	// or additional executable is needed for this regression.
	p.plainHTTPTransport().RegisterProtocol("http", publicTestRoundTrip(func(r *http.Request) (*http.Response, error) {
		if b.requests.Add(1) != 1 {
			return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: http.NoBody, Request: r}, nil
		}
		account := inferenceBuffers(r.Context())
		if retireParser {
			// Match the existing postcommit-refusal seam: retire the original
			// handle before awaitFirstBody, while the sink's borrower stays live.
			account.mu.Lock()
			parser := account.parser
			account.mu.Unlock()
			if parser == nil {
				t.Error("missing original parser handle")
			} else {
				parser.close()
				if probe := account.borrowParserAllowance(); probe != nil {
					probe.close()
					t.Error("original parser handle remained borrowable")
				}
			}
		}
		b.account <- account
		return &http.Response{StatusCode: status, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: b, Request: r}, nil
	}))
	t.Cleanup(b.unblockRead)
	t.Cleanup(b.unblockClose)
	return b
}

func TestHandleHTTP_PrecommitBlockingCloseRetainsOwnership(t *testing.T) {
	setForTest(t, &firstBodyTimeout, 20*time.Millisecond)
	test := func(name string, status int, finishesFirst string, retireParser bool) {
		t.Run(name, func(t *testing.T) {
			discovery := NewDiscovery()
			discovery.AddManual(nodeForModel(t, "self", "http://127.0.0.1:11435", "m"))
			discovery.AddManual(nodeForModel(t, "other", "http://127.0.0.1:11436", "m"))
			p := testProxy(ollamaOnlyProfile(t), discovery, 11435)
			p.SetPriority([]string{"self", "other"})
			p.admission.setAvailability(nodepolicy.Available, false)
			if err := p.soleFacade().setLocalBackend(localBackend{Host: "127.0.0.1", Port: 11434, Healthy: true}); err != nil {
				t.Fatal(err)
			}
			upstream := installBlockingFirstBody(t, p, status, retireParser)
			rec, done := httptest.NewRecorder(), make(chan struct{})
			go func() {
				defer close(done)
				p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)))
			}()
			defer func() {
				upstream.unblockRead()
				upstream.unblockClose()
				waitChan(t, done, "released handler")
			}()
			waitChan(t, upstream.readStarted, "first raw read")
			account := <-upstream.account
			p.buffers.mu.Lock()
			wantScope := account.scope.used
			p.buffers.mu.Unlock()
			waitChan(t, upstream.closeStarted, "timeout Close")
			keys := p.activeRoutingRequests()
			if len(keys) != 1 {
				t.Fatalf("registered requests=%d", len(keys))
			}
			owner := p.routingRequestControl(keys[0])
			switch finishesFirst {
			case "read":
				upstream.unblockRead()
				waitChan(t, upstream.readDone, "read completed while Close blocks")
			case "close":
				upstream.unblockClose()
				waitChan(t, upstream.closeDone, "Close completed while read blocks")
			}
			select {
			case <-done:
			case <-time.After(routingShutdownGrace + time.Second):
				t.Fatal("HTTP return exceeded the precommit cleanup bound")
			}
			if rec.Code != http.StatusGatewayTimeout || upstream.requests.Load() != 1 || upstream.closes.Load() != 1 || upstream.reads.Load() != 1 {
				t.Fatalf("precommit result: status=%d requests=%d closes=%d reads=%d", rec.Code, upstream.requests.Load(), upstream.closes.Load(), upstream.reads.Load())
			}
			var result struct {
				Error string `json:"error"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil || result.Error != "upstream error: "+errFirstBodyJoinTimeout.Error() {
				t.Fatalf("timeout response=%+v decode=%v", result, err)
			}
			owner.mu.Lock()
			pending, facts, terminal, committed := owner.firstBodyDone, owner.facts, owner.done, owner.committed
			owner.mu.Unlock()
			if pending == nil || facts == nil || !terminal || committed || p.routingRequestControl(keys[0]) != owner || owner.cancel(false) {
				t.Fatal("HTTP return lost the terminal canonical owner or accepted a late byte")
			}
			if _, err := owner.newAttempt(); err == nil {
				t.Fatal("pending cleanup allowed a new attempt")
			}
			var before routingNativeCopy
			if !facts.copyInto(&before) || before.flags&factFrozen != 0 || before.flags&factEndKnown == 0 || before.binding&factAttached != 0 {
				t.Fatal("pending body froze native facts or retained a workload association")
			}
			p.identityMu.Lock()
			associations := len(p.workloadRequests)
			p.identityMu.Unlock()
			if associations != 0 {
				t.Fatalf("terminal workload index entries=%d", associations)
			}
			assertPending := func() {
				t.Helper()
				select {
				case <-pending:
					t.Fatal("body join completed before both collaborators returned")
				default:
				}
				registryBytes := registryTestBytes(p)
				p.buffers.mu.Lock()
				used, scopeUsed, waiters := p.buffers.used, account.scope.used, p.buffers.waiters.Len()
				p.buffers.mu.Unlock()
				if scopeUsed != wantScope || used != scopeUsed+registryBytes || waiters != 0 || p.admission.currentActive() != 1 {
					t.Fatalf("pending ownership: scope=%d want=%d root=%d registry=%d waiters=%d active=%d", scopeUsed, wantScope, used, registryBytes, waiters, p.admission.currentActive())
				}
			}
			assertPending()
			if finishesFirst == "neither" {
				upstream.unblockClose()
				waitChan(t, upstream.closeDone, "late Close completion")
				assertPending()
			}
			upstream.unblockRead()
			upstream.unblockClose()
			waitChan(t, pending, "actual Close and read join")
			waitForCond(t, time.Second, "canonical owner and charge cleanup", func() bool {
				p.buffers.mu.Lock()
				empty := p.buffers.used == 0
				p.buffers.mu.Unlock()
				return empty && len(p.activeRoutingRequests()) == 0
			})
			var after routingNativeCopy
			if !facts.copyInto(&after) || after.flags&factFrozen == 0 || after.ended != before.ended || upstream.closes.Load() != 1 || upstream.requests.Load() != 1 {
				t.Fatal("late completion changed terminal facts, retried or closed the raw body twice")
			}
			ownedAssertHandlerDrained(t, p, owner)
		})
	}
	test("Close blocks after raw read completes", http.StatusOK, "read", false)
	test("raw read blocks after Close completes", http.StatusOK, "close", false)
	test("Close and raw read complete separately after HTTP return", http.StatusOK, "neither", false)
	test("rejected response retains its canonical cleanup owner", http.StatusBadRequest, "read", false)
	test("retired parser keeps its sink grant through late Close", http.StatusOK, "read", true)
}

func TestStrictHTTP_PrecommitBlockingCloseRetainsOwnership(t *testing.T) {
	for _, path := range []string{"self", "ingress"} {
		t.Run(path, func(t *testing.T) {
			p := testProxy(ollamaOnlyProfile(t), NewDiscovery(), 11435)
			configureStrict(t, p.admission, filepath.Join(t.TempDir(), "capacity.json"))
			if err := p.soleFacade().setLocalBackend(localBackend{Host: "127.0.0.1", Port: 11434, Healthy: true}); err != nil {
				t.Fatal(err)
			}
			upstream := installBlockingFirstBody(t, p, http.StatusOK, false)
			r := strictHTTPRequest(t)
			rec, done := httptest.NewRecorder(), make(chan struct{})
			go func() { defer close(done); serveStrictHTTP(p, path, rec, r) }()
			defer func() {
				upstream.unblockRead()
				upstream.unblockClose()
				waitChan(t, done, "released strict handler")
			}()
			waitChan(t, upstream.readStarted, "strict first raw read")
			account := <-upstream.account
			p.buffers.mu.Lock()
			wantScope := account.scope.used
			p.buffers.mu.Unlock()
			waitChan(t, upstream.closeStarted, "strict timeout Close")
			upstream.unblockRead()
			waitChan(t, upstream.readDone, "strict raw read completion")
			select {
			case <-done:
			case <-time.After(routingShutdownGrace + time.Second):
				t.Fatal("strict HTTP return exceeded the cleanup bound")
			}
			if rec.Code != http.StatusGatewayTimeout {
				t.Fatalf("strict deadline status=%d, want 504", rec.Code)
			}
			p.buffers.mu.Lock()
			retained := account.scope.used
			p.buffers.mu.Unlock()
			if retained != wantScope || retained == 0 || upstream.closes.Load() != 1 || upstream.requests.Load() != 1 || p.admission.currentActive() != 1 {
				t.Fatalf("strict pending ownership: scope=%d want=%d closes=%d requests=%d active=%d", retained, wantScope, upstream.closes.Load(), upstream.requests.Load(), p.admission.currentActive())
			}
			upstream.unblockClose()
			waitChan(t, upstream.closeDone, "strict late Close")
			waitForCond(t, time.Second, "strict storage cleanup", func() bool {
				p.buffers.mu.Lock()
				defer p.buffers.mu.Unlock()
				return p.buffers.used == 0
			})
			// Strict dispatch remains uncertain after join: neither EOF nor a
			// completed Close is evidence that the engine stopped execution.
			if p.admission.currentActive() != 1 || len(readCapacity(t, p.admission.strict.path).Slots) != 1 || len(p.activeRoutingRequests()) != 0 {
				t.Fatal("late strict cleanup released quarantined capacity or retained the origin registry")
			}
		})
	}
}

func TestHandleHTTP_CommittedBlockingCloseUsesSink(t *testing.T) {
	setForTest(t, &firstBodyTimeout, testFirstBodyTimeout)
	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(t, "a", "http://127.0.0.1:11434", "m"))
	discovery.AddManual(nodeForModel(t, "b", "http://127.0.0.1:11436", "m"))
	p := testProxy(ollamaOnlyProfile(t), discovery, 11435)
	p.SetPriority([]string{"a", "b"})
	upstream := installBlockingFirstBody(t, p, http.StatusAccepted, false)
	rec, done := httptest.NewRecorder(), make(chan struct{})
	committed, resume := make(chan struct{}), make(chan struct{})
	unblockCallback := sync.OnceFunc(func() { close(resume) })
	go func() {
		defer close(done)
		p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)))
	}()
	defer func() {
		unblockCallback()
		upstream.unblockRead()
		upstream.unblockClose()
		waitChan(t, done, "released committed handler")
	}()
	waitChan(t, upstream.readStarted, "pending committed peek")
	keys := p.activeRoutingRequests()
	if len(keys) != 1 {
		t.Fatalf("active requests=%d", len(keys))
	}
	owner := p.routingRequestControl(keys[0])
	owner.mu.Lock()
	stop := owner.stopPreOutput
	owner.stopPreOutput = func() bool { close(committed); <-resume; return stop() }
	owner.mu.Unlock()
	upstream.unblockRead()
	waitChan(t, committed, "raw commitment before blocked publication")
	waitChan(t, upstream.closeStarted, "committed timeout Close")
	unblockCallback()
	select {
	case <-done:
	case <-time.After(routingShutdownGrace + time.Second):
		t.Fatal("committed HTTP return exceeded the cleanup bound")
	}
	owner.mu.Lock()
	pending, terminal, rawCommitted := owner.firstBodyDone, owner.done, owner.committed
	owner.mu.Unlock()
	result := owner.sink.result()
	if !terminal || !rawCommitted || rec.Code != http.StatusAccepted || rec.Body.String() != string(wantSSEInterruption(streamInterruptReasonIdle)) || result.Reason != streamInterruptReasonIdle || result.ProtocolComplete || !result.DeliveryComplete || result.Genuine != 0 || upstream.requests.Load() != 1 {
		t.Fatalf("committed cleanup bypassed sink: terminal=%v committed=%v status=%d body=%q sink=%+v requests=%d", terminal, rawCommitted, rec.Code, rec.Body.String(), result, upstream.requests.Load())
	}
	if pending == nil || p.routingRequestControl(keys[0]) != owner {
		t.Fatal("blocked Close lost canonical discovery")
	}
	select {
	case <-pending:
		t.Fatal("committed Close falsely joined before release")
	default:
	}
	upstream.unblockClose()
	waitChan(t, pending, "committed late Close and peek join")
	waitForCond(t, time.Second, "committed canonical and charge cleanup", func() bool {
		p.buffers.mu.Lock()
		empty := p.buffers.used == 0
		p.buffers.mu.Unlock()
		return empty && len(p.activeRoutingRequests()) == 0
	})
	if upstream.closes.Load() != 1 {
		t.Fatalf("raw Close calls=%d", upstream.closes.Load())
	}
	ownedAssertHandlerDrained(t, p, owner)
}

// stalledEngine is an upstream that sends 200 and headers immediately and then
// produces no body until released, which is the shape an engine presents for a
// streaming request queued behind other work.
type stalledEngine struct {
	url string
	// hits counts dispatches, so a test can show the retry budget was spent
	// rather than inferring it from timing.
	hits    func() int
	release func()
}

func newStalledEngine(t *testing.T, method, path string) stalledEngine {
	t.Helper()
	var n atomic.Int32
	releaseCh := make(chan struct{})
	var once sync.Once
	release := func() { once.Do(func() { close(releaseCh) }) }

	srv := httptest.NewServer(fixtureRequestHandler(method, path, func(w http.ResponseWriter, r *http.Request) {
		n.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush() // headers on the wire, body still empty
		}
		select {
		case <-releaseCh:
		case <-r.Context().Done():
		}
	}))
	// Cleanup is LIFO, so release runs before Close. Close waits for the
	// handler to return and the handler is parked on release, so the other
	// order deadlocks the end of the test.
	t.Cleanup(srv.Close)
	t.Cleanup(release)

	return stalledEngine{
		url:     srv.URL,
		hits:    func() int { return int(n.Load()) },
		release: release,
	}
}

// TestHandleHTTP_HeadersWithoutContentFailsOver is the regression for the
// engine that accepts and stalls. The first owner sends headers and no content;
// the request must move to the next owner rather than binding itself to a node
// that has not started generating.
func TestHandleHTTP_HeadersWithoutContentFailsOver(t *testing.T) {
	setForTest(t, &firstBodyTimeout, testFirstBodyTimeout)

	forEachEngine(t, func(t *testing.T, tc engineCase) {
		stalled := newStalledEngine(t, http.MethodPost, tc.inferencePath)

		var servedBody string
		good := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			servedBody = string(b)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			io.WriteString(w, `{"done":true}`)
		}))
		defer good.Close()

		disc := NewDiscovery()
		disc.AddManual(nodeForModel(t, "stalled", stalled.url, tc.advertisedModel))
		disc.AddManual(nodeForModel(t, "good", good.URL, tc.advertisedModel))
		p := testProxy(tc.profile, disc, tc.profile.FacadePort)
		p.soleFacade().SetSelected("stalled") // deterministic: the staller is tried first

		rec := httptest.NewRecorder()
		p.soleFacade().handleHTTP(rec, tc.inferenceRequest())

		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (should have failed over past the stalled engine)", rec.Code)
		}
		if got := decodeJSONBody(t, rec.Body.String()); got["done"] != true {
			t.Fatalf(`body has done = %v, want true: the response came from the wrong node`, got["done"])
		}
		if servedBody != tc.inferenceBody() {
			t.Errorf("failover node got body %q, want the original request body", servedBody)
		}
	})
}

// TestHandleHTTP_NoContentOnFinalAttemptTerminates covers the exhausted case,
// which is the one that could hang: a first-content timeout must not commit
// even with no attempt left, because the peek reader is still blocked on the
// upstream body (spec §5.2).
//
// 504 rather than 502: the node answered, it just never produced anything.
func TestHandleHTTP_NoContentOnFinalAttemptTerminates(t *testing.T) {
	setForTest(t, &firstBodyTimeout, testFirstBodyTimeout)

	forEachEngine(t, func(t *testing.T, tc engineCase) {
		// Alternatives retain the per-attempt first-content cap, even when
		// the final dispatch has no remaining attempt to fail over to.
		stalled := newStalledEngine(t, http.MethodPost, tc.inferencePath)
		other := newStalledEngine(t, http.MethodPost, tc.inferencePath)

		disc := NewDiscovery()
		disc.AddManual(nodeForModel(t, "silent", stalled.url, tc.advertisedModel))
		disc.AddManual(nodeForModel(t, "other-silent", other.url, tc.advertisedModel))
		p := testProxy(tc.profile, disc, tc.profile.FacadePort)

		rec := httptest.NewRecorder()
		done := make(chan struct{})
		go func() {
			defer close(done)
			p.soleFacade().handleHTTP(rec, tc.inferenceRequest())
		}()
		waitForCond(t, time.Second, "first stalled dispatch", func() bool { return stalled.hits()+other.hits() > 0 })
		assertFixtureRejectsProbe(t, stalled.url)
		assertFixtureRejectsProbe(t, other.url)

		select {
		case <-done:
		// Five attempts at the shortened budget is about a second. Any finite
		// guard catches the regression this test exists for, since a committed
		// silent upstream blocks forever.
		case <-time.After(5 * time.Second):
			t.Fatal("handleHTTP never returned: a silent upstream committed and hung the request")
		}

		if got := stalled.hits() + other.hits(); got != maxDispatchAttempts {
			t.Errorf("upstreams saw %d dispatches, want the full budget of %d", got, maxDispatchAttempts)
		}
		if rec.Code != http.StatusGatewayTimeout {
			t.Fatalf("status = %d, want 504 for an upstream that answered but produced no content", rec.Code)
		}
		want := "upstream error: " + errFirstBodyTimeout.Error()
		if got := decodeJSONBody(t, rec.Body.String()); got["error"] != want {
			t.Fatalf("body error = %v, want %q", got["error"], want)
		}
	})
}

// TestHandleHTTP_FirstBytePreservedOnCommit guards the splice: the byte
// awaitFirstBody consumes to detect the commit has to be handed back, or every
// committed response would be served a byte short. Compared bytewise rather
// than as JSON, since losing the first byte is exactly what would still parse
// as something.
func TestHandleHTTP_FirstBytePreservedOnCommit(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		const payload = `{"response":"whole body intact","done":true}`
		good := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			io.WriteString(w, payload)
		}))
		defer good.Close()

		disc := NewDiscovery()
		disc.AddManual(nodeForModel(t, "good", good.URL, tc.advertisedModel))
		p := testProxy(tc.profile, disc, tc.profile.FacadePort)

		rec := httptest.NewRecorder()
		p.soleFacade().handleHTTP(rec, tc.inferenceRequest())

		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", rec.Code)
		}
		if got := rec.Body.String(); got != payload {
			t.Fatalf("body = %q, want %q (the peeked first byte was dropped)", got, payload)
		}
	})
}

// An empty 200 commits promptly, but only a non-streaming response can complete
// without a protocol end marker.
func TestHandleHTTP_EmptyBodyCommits(t *testing.T) {
	// Long, so a wait would be unmistakable: if the empty body were treated as
	// a stall, the guard below would trip well before this expires.
	setForTest(t, &firstBodyTimeout, 30*time.Second)

	forEachEngine(t, func(t *testing.T, tc engineCase) {
		test := func(name, streamFlag string, streaming bool) {
			t.Run(name, func(t *testing.T) {
				empty := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					mediaType := "application/json"
					if streaming && tc.profile.Name != "ollama" {
						mediaType = "text/event-stream"
					}
					w.Header().Set("Content-Type", mediaType)
					w.WriteHeader(http.StatusOK)
				}))
				defer empty.Close()

				disc := NewDiscovery()
				disc.AddManual(nodeForModel(t, "empty", empty.URL, tc.advertisedModel))
				writer := &recordingWriter{}
				p := newTestProxy(tc.profile, NewCodec(writer), disc, tc.profile.FacadePort)
				rec := httptest.NewRecorder()
				done := make(chan struct{})
				go func() {
					defer close(done)
					body := strings.TrimSuffix(tc.inferenceBody(), "}") + `,"stream":` + streamFlag + `}`
					p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, tc.inferencePath, strings.NewReader(body)))
				}()

				select {
				case <-done:
				case <-time.After(5 * time.Second):
					t.Fatal("an empty 200 body blocked on the first-byte wait instead of committing")
				}
				if rec.Code != http.StatusOK {
					t.Fatalf("status = %d, want committed 200", rec.Code)
				}
				terminals := append(notificationsOf(t, writer, workloadCompletedMethod), notificationsOf(t, writer, workloadErroredMethod)...)
				if len(terminals) != 1 {
					t.Fatalf("terminal notifications = %d, want exactly one", len(terminals))
				}
				var terminal workloadParams
				if err := json.Unmarshal(terminals[0], &terminal); err != nil {
					t.Fatalf("decode workload terminal: %v", err)
				}
				if !streaming {
					if terminal.WorkloadInfo.State != "completed" || rec.Body.Len() != 0 {
						t.Fatalf("empty non-streaming response: state = %q, bytes = %d", terminal.WorkloadInfo.State, rec.Body.Len())
					}
					return
				}
				if terminal.WorkloadInfo.State != "failed" || terminal.WorkloadInfo.Error == nil || *terminal.WorkloadInfo.Error == "" {
					t.Fatalf("empty declared stream terminal = %+v, want failed with reason", terminal.WorkloadInfo)
				}
				errorFrame := strings.TrimSpace(rec.Body.String())
				if tc.profile.Name != "ollama" {
					events := strings.Split(errorFrame, "\n\n")
					if len(events) != 2 || events[1] != "data: [DONE]" || !strings.HasPrefix(events[0], "data: ") {
						t.Fatalf("empty stream SSE terminator = %q", errorFrame)
					}
					errorFrame = strings.TrimPrefix(events[0], "data: ")
				}
				if frame := decodeJSONBody(t, errorFrame); frame["error"] == nil {
					t.Fatal("empty declared stream has no protocol error")
				}
			})
		}
		test("declared stream fails after commit", "true", true)
		test("non-streaming empty body completes", "false", false)
	})
}

// TestHandleHTTP_NonInferenceCommitsOnHeaders: the gate is scoped to inference.
// A control endpoint's headers mean what they say, and some of them (a pull's
// progress stream) legitimately produce their first byte late, so waiting there
// would add latency for no benefit.
func TestHandleHTTP_NonInferenceCommitsOnHeaders(t *testing.T) {
	setForTest(t, &firstBodyTimeout, testFirstBodyTimeout)

	forEachEngine(t, func(t *testing.T, tc engineCase) {
		stalled := newStalledEngine(t, http.MethodGet, tc.nonInferencePath)

		disc := NewDiscovery()
		disc.AddManual(nodeForModel(t, "stalled", stalled.url, tc.advertisedModel))
		p := testProxy(tc.profile, disc, tc.profile.FacadePort)

		rec := httptest.NewRecorder()
		done := make(chan struct{})
		go func() {
			req := httptest.NewRequest(http.MethodGet, tc.nonInferencePath, nil)
			p.soleFacade().handleHTTP(rec, req)
			close(done)
		}()

		// Still blocked on the upstream body well past the first-content
		// budget, which is the whole assertion: if the gate applied here the
		// handler would have given up at firstBodyTimeout and returned 504,
		// and simply waiting for it to finish would pass either way.
		blocked := 2 * testFirstBodyTimeout
		time.Sleep(blocked)
		select {
		case <-done:
			t.Fatalf("handler returned after %v with status %d: the first-content gate was applied to a non-inference route",
				blocked, rec.Code)
		default:
		}

		stalled.release()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("a non-inference request did not complete after the upstream released")
		}
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200: the request committed on headers and streamed normally", rec.Code)
		}
	})
}
