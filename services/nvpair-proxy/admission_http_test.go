// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Admission at both of its entry points: the cluster ingress, for work a peer
// sends here, and handleHTTP's self candidate, for work this node runs itself.

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/nodepolicy"
)

func strictHTTPFixture(t *testing.T) (*Proxy, *recordingEngine) {
	t.Helper()
	profile, ok := profileFor("ollama")
	if !ok {
		t.Fatal("missing ollama profile")
	}
	p := testProxy(profile, NewDiscovery(), profile.FacadePort)
	configureStrict(t, p.admission, filepath.Join(t.TempDir(), "capacity.json"))
	e := newRecordingEngine(t, http.StatusOK)
	setBackend(t, p.soleFacade(), e.URL, true)
	return p, e
}
func strictHTTPRequest(t *testing.T) *http.Request {
	t.Helper()
	return strictHTTPRequestWith(t, strictBody("m"))
}

// strictHTTPRequestWith is a strict chat request carrying body.
func strictHTTPRequestWith(t *testing.T, body string) *http.Request {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(body))
	b, err := json.Marshal(strictRequest())
	if err != nil {
		t.Fatal(err)
	}
	r.Header.Set(nodepolicy.StrictAdmissionHeader, string(b))
	return r
}
func serveStrictHTTP(p *Proxy, path string, w http.ResponseWriter, r *http.Request) {
	if path == "ingress" {
		p.soleFacade().serveIngress(w, r, "peer")
	} else {
		p.soleFacade().handleHTTP(w, r)
	}
}

func TestStrictHTTPStripsInternalHeaders(t *testing.T) {
	for _, path := range []string{"ingress", "self"} {
		t.Run(path, func(t *testing.T) {
			p, e := strictHTTPFixture(t)
			r := strictHTTPRequest(t)
			for _, h := range []string{nodepolicy.AdmissionWaitHeader, "X-PAIR-Admission", "X-PAIR-Model", "X-PAIR-Engine", "X-PAIR-Node", "X-PAIR-Tier", engineIdentityProbeHeader} {
				r.Header.Set(h, "internal")
			}
			r.Header.Set("X-Custom-Engine-Option", "keep")
			w := httptest.NewRecorder()
			serveStrictHTTP(p, path, w, r)
			if w.Code != http.StatusOK || e.hits() != 1 {
				t.Fatalf("status=%d hits=%d body=%s", w.Code, e.hits(), w.Body)
			}
			for h := range e.lastHeader() {
				if strings.HasPrefix(strings.ToLower(h), "x-pair-") || strings.EqualFold(h, engineIdentityProbeHeader) {
					t.Fatalf("leaked header %s", h)
				}
			}
			if e.lastHeader().Get("X-Custom-Engine-Option") != "keep" {
				t.Fatal("engine header lost")
			}
			if p.admission.currentActive() != 1 {
				t.Fatal("HTTP EOF incorrectly released uncertain capacity")
			}
			if len(readCapacity(t, p.admission.strict.path).Slots) != 1 {
				t.Fatal("missing dispatch marker")
			}
		})
	}
}

func TestStrictHTTPRejectsMalformedAndDuplicateHeaders(t *testing.T) {
	for _, path := range []string{"ingress", "self"} {
		t.Run(path, func(t *testing.T) {
			for _, scenario := range []string{"malformed", "duplicate", "empty"} {
				t.Run(scenario, func(t *testing.T) {
					p, e := strictHTTPFixture(t)
					r := strictHTTPRequest(t)
					switch scenario {
					case "malformed":
						r.Header.Set(nodepolicy.StrictAdmissionHeader, "{")
					case "duplicate":
						r.Header.Add(nodepolicy.StrictAdmissionHeader, r.Header.Get(nodepolicy.StrictAdmissionHeader))
					case "empty":
						r.Header.Set(nodepolicy.StrictAdmissionHeader, "")
					}
					w := httptest.NewRecorder()
					serveStrictHTTP(p, path, w, r)
					if w.Code != http.StatusBadRequest || e.hits() != 0 || p.admission.currentActive() != 0 {
						t.Fatalf("status=%d hits=%d active=%d", w.Code, e.hits(), p.admission.currentActive())
					}
				})
			}
		})
	}
}

func TestStrictHTTPPersistenceFailureSendsNoInference(t *testing.T) {
	for _, path := range []string{"ingress", "self"} {
		t.Run(path, func(t *testing.T) {
			p, e := strictHTTPFixture(t)
			p.admission.strict.write = func(string, []byte, os.FileMode) error { return errors.New("injected disk failure") }
			w := httptest.NewRecorder()
			serveStrictHTTP(p, path, w, strictHTTPRequest(t))
			if w.Code != http.StatusServiceUnavailable || e.hits() != 0 {
				t.Fatalf("status=%d hits=%d", w.Code, e.hits())
			}
		})
	}
}

func TestStrictHTTPCancellationIgnoringEngineQuarantines(t *testing.T) {
	for _, path := range []string{"ingress", "self"} {
		t.Run(path, func(t *testing.T) {
			p, _ := strictHTTPFixture(t)
			received := make(chan struct{})
			release := make(chan struct{})
			e := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { close(received); <-release; w.WriteHeader(http.StatusOK) }))
			defer e.Close()
			defer close(release)
			setBackend(t, p.soleFacade(), e.URL, true)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			r := strictHTTPRequest(t).WithContext(ctx)
			done := make(chan struct{})
			go func() { defer close(done); serveStrictHTTP(p, path, httptest.NewRecorder(), r) }()
			waitChan(t, received, "engine request")
			cancel()
			waitChan(t, done, "cancelled handler")
			if p.admission.currentActive() != 1 || len(readCapacity(t, p.admission.strict.path).Slots) != 1 {
				t.Fatal("cancel released unproved engine capacity")
			}
		})
	}
}

func TestStrictHTTPPreOutputDeadlineQuarantines(t *testing.T) {
	for _, path := range []string{"ingress", "self"} {
		t.Run(path, func(t *testing.T) {
			p, _ := strictHTTPFixture(t)
			c := useStrictClock(p.admission)
			received := make(chan struct{})
			release := make(chan struct{})
			e := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				w.(http.Flusher).Flush()
				close(received)
				<-release
			}))
			defer e.Close()
			defer close(release)
			setBackend(t, p.soleFacade(), e.URL, true)
			done := make(chan struct{})
			w := httptest.NewRecorder()
			go func() { defer close(done); serveStrictHTTP(p, path, w, strictHTTPRequest(t)) }()
			awaitStrictTimer(t, c)
			waitChan(t, received, "headers before body")
			c.Advance(time.Second)
			waitChan(t, done, "pre-output deadline")
			if w.Code != http.StatusGatewayTimeout {
				t.Fatalf("status=%d", w.Code)
			}
			if p.admission.currentActive() != 1 {
				t.Fatal("timeout released uncertain slot")
			}
		})
	}
}

func TestStrictHTTPStatusMapping(t *testing.T) {
	for _, tc := range []struct {
		err    error
		status int
	}{{errStrictUnavailable, http.StatusServiceUnavailable}, {errStrictOccupied, http.StatusTooManyRequests}, {errStrictCold, http.StatusServiceUnavailable}, {context.DeadlineExceeded, http.StatusGatewayTimeout}} {
		t.Run(tc.err.Error(), func(t *testing.T) {
			w := httptest.NewRecorder()
			writeStrictAdmissionError(w, tc.err)
			if w.Code != tc.status {
				t.Fatalf("status=%d want=%d", w.Code, tc.status)
			}
			var typed *strictAdmissionError
			if !errors.Is(tc.err, context.DeadlineExceeded) && !errors.As(tc.err, &typed) {
				t.Fatal("missing typed outcome")
			}
		})
	}
}

type strictFirstByteWriter struct {
	*httptest.ResponseRecorder
	first chan struct{}
	once  sync.Once
}

func (w *strictFirstByteWriter) Write(b []byte) (int, error) {
	n, err := w.ResponseRecorder.Write(b)
	if n > 0 {
		w.once.Do(func() { close(w.first) })
	}
	return n, err
}

func TestStrictHTTPHealthyGenerationOutlivesSelectionBudget(t *testing.T) {
	p, _ := strictHTTPFixture(t)
	c := useStrictClock(p.admission)
	release := make(chan struct{})
	e := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "first")
		w.(http.Flusher).Flush()
		<-release
		_, _ = io.WriteString(w, "last")
	}))
	defer e.Close()
	setBackend(t, p.soleFacade(), e.URL, true)
	w := &strictFirstByteWriter{ResponseRecorder: httptest.NewRecorder(), first: make(chan struct{})}
	done := make(chan struct{})
	r := strictHTTPRequest(t)
	go func() { defer close(done); serveStrictHTTP(p, "self", w, r) }()
	awaitStrictTimer(t, c)
	waitChan(t, w.first, "committed first body")
	c.Advance(2 * time.Second)
	select {
	case <-done:
		t.Fatal("selection budget interrupted active generation")
	default:
	}
	close(release)
	waitChan(t, done, "healthy generation completion")
	if w.Body.String() != "firstlast" {
		t.Fatalf("body=%q", w.Body.String())
	}
}

// selfNode is a discovery node that resolves to the facade's own listener, so
// resolveCandidates maps it to the local backend.
func selfNode(id string, port int, models ...string) Node {
	return Node{ID: id, Addresses: []string{"127.0.0.1"}, Port: port, Models: models}
}

func setBackend(t *testing.T, f *facade, serverURL string, healthy bool) {
	t.Helper()
	n := nodeFor(t, "backend", serverURL)
	if err := f.setLocalBackend(localBackend{Host: n.Addresses[0], Port: n.Port, Healthy: healthy}); err != nil {
		t.Fatal(err)
	}
}

// recordingEngine is an upstream that records what it was sent.
type recordingEngine struct {
	*httptest.Server
	mu      sync.Mutex
	bodies  []string
	headers []http.Header
}

func newRecordingEngine(t *testing.T, status int) *recordingEngine {
	t.Helper()
	e := &recordingEngine{}
	e.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		e.mu.Lock()
		e.bodies = append(e.bodies, string(b))
		e.headers = append(e.headers, r.Header.Clone())
		e.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, `{"done":true}`)
	}))
	t.Cleanup(e.Close)
	return e
}

func (e *recordingEngine) hits() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.bodies)
}

func (e *recordingEngine) lastBody() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.bodies) == 0 {
		return ""
	}
	return e.bodies[len(e.bodies)-1]
}

func (e *recordingEngine) lastHeader() http.Header {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.headers) == 0 {
		return nil
	}
	return e.headers[len(e.headers)-1]
}

// stallingEngine accepts a request and holds it, sending nothing, until the
// client goes away or the test releases it.
type stallingEngine struct {
	*httptest.Server
	received  chan struct{}
	cancelled chan struct{}
	release   chan struct{}
	once      sync.Once
}

func newStallingEngine(t *testing.T) *stallingEngine {
	t.Helper()
	e := &stallingEngine{
		received:  make(chan struct{}, 8),
		cancelled: make(chan struct{}, 8),
		release:   make(chan struct{}),
	}
	e.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		e.received <- struct{}{}
		select {
		case <-r.Context().Done():
			e.cancelled <- struct{}{}
		case <-e.release:
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, `{"done":true}`)
		}
	}))
	t.Cleanup(func() {
		e.doRelease()
		e.Close()
	})
	return e
}

func (e *stallingEngine) doRelease() { e.once.Do(func() { close(e.release) }) }

func waitChan(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(3 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
	}
}

func assertAdmissionRejection(t *testing.T, rec *httptest.ResponseRecorder, reason nodepolicy.RejectReason) {
	t.Helper()
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503: %s", rec.Code, rec.Body)
	}
	if got := rec.Header().Get(nodepolicy.AdmissionHeader); got != string(reason) {
		t.Fatalf("%s = %q, want %q", nodepolicy.AdmissionHeader, got, reason)
	}
	if rec.Header().Get("Retry-After") == "" {
		t.Fatal("admission rejection carries no Retry-After")
	}
}

func ingressRequest(tc engineCase, body string, wait string) *http.Request {
	r := httptest.NewRequest(http.MethodPost, tc.inferencePath, strings.NewReader(body))
	if wait != "" {
		r.Header.Set(nodepolicy.AdmissionWaitHeader, wait)
	}
	return r
}

func TestIngress_AppliesAdmission(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		engine := newRecordingEngine(t, http.StatusOK)
		p := testProxy(tc.profile, NewDiscovery(), tc.profile.FacadePort)
		f := p.soleFacade()
		setBackend(t, f, engine.URL, true)

		p.admission.setAvailability(nodepolicy.Paused, false)
		rec := httptest.NewRecorder()
		f.serveIngress(rec, ingressRequest(tc, tc.inferenceBody(), ""), "peer")
		assertAdmissionRejection(t, rec, nodepolicy.RejectPaused)
		if engine.hits() != 0 {
			t.Fatal("a paused node forwarded a peer's request to its engine")
		}

		// Available: admitted, the profile's requestOptions merged in with the
		// request's own keys winning, and the router's wait header not passed
		// to the engine.
		p.admission.setAvailability(nodepolicy.Available, false)
		pol := nodepolicy.Default()
		pol.Profiles = []nodepolicy.Profile{{
			Name: "p", Engine: tc.profile.Name, Model: tc.advertisedModel,
			RequestOptions: json.RawMessage(`{"keep_alive":"10m","stream":true}`),
		}}
		p.admission.setPolicy(pol)
		rec = httptest.NewRecorder()
		body := `{"model":"` + tc.requestedModel + `","stream":false}`
		f.serveIngress(rec, ingressRequest(tc, body, "0"), "peer")
		if rec.Code != http.StatusOK {
			t.Fatalf("admitted ingress status = %d: %s", rec.Code, rec.Body)
		}
		want := `{"model":"` + tc.requestedModel + `","stream":false,"keep_alive":"10m"}`
		if got := engine.lastBody(); got != want {
			t.Fatalf("engine got %s, want %s", got, want)
		}
		if engine.lastHeader().Get(nodepolicy.AdmissionWaitHeader) != "" {
			t.Fatal("the router's admission wait header reached the engine")
		}
		if p.admission.active != 0 {
			t.Fatalf("active = %d after the ingress request finished", p.admission.active)
		}
	})
}

// A non-inference route is not work and is not admitted.
func TestIngress_NonInferenceBypassesAdmission(t *testing.T) {
	tc := anyCase(t)
	engine := newRecordingEngine(t, http.StatusOK)
	p := testProxy(tc.profile, NewDiscovery(), tc.profile.FacadePort)
	f := p.soleFacade()
	setBackend(t, f, engine.URL, true)
	p.admission.setAvailability(nodepolicy.Paused, false)
	rec := httptest.NewRecorder()
	f.serveIngress(rec, httptest.NewRequest(http.MethodGet, tc.modelListPath, nil), "peer")
	if rec.Code != http.StatusOK || engine.hits() != 1 {
		t.Fatalf("model list through a paused ingress: status %d, hits %d", rec.Code, engine.hits())
	}
}

func TestIngress_WaitHeaderZeroRejectsBusyImmediately(t *testing.T) {
	tc := anyCase(t)
	engine := newRecordingEngine(t, http.StatusOK)
	p := testProxy(tc.profile, NewDiscovery(), tc.profile.FacadePort)
	f := p.soleFacade()
	setBackend(t, f, engine.URL, true)
	pol := nodepolicy.Default()
	pol.Admission.MaxConcurrentPerModel = 1
	p.admission.setPolicy(pol)
	held := mustAdmit(t, p.admission.admit(t.Context(), admissionRequest{engine: tc.profile.Name, model: tc.requestedModel, waitCap: -1}))
	defer held.release()

	started := time.Now()
	rec := httptest.NewRecorder()
	f.serveIngress(rec, ingressRequest(tc, tc.inferenceBody(), "0"), "peer")
	assertAdmissionRejection(t, rec, nodepolicy.RejectBusy)
	if waited := time.Since(started); waited > 500*time.Millisecond {
		t.Fatalf("X-PAIR-Admission-Wait: 0 still queued for %v", waited)
	}
}

func TestIngress_DrainingFinishesActiveAndCancelActiveCancels(t *testing.T) {
	tc := anyCase(t)
	engine := newStallingEngine(t)
	p := testProxy(tc.profile, NewDiscovery(), tc.profile.FacadePort)
	f := p.soleFacade()
	setBackend(t, f, engine.URL, true)

	// Draining: the running request finishes, a new one is refused.
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		rec := httptest.NewRecorder()
		f.serveIngress(rec, ingressRequest(tc, tc.inferenceBody(), ""), "peer")
		done <- rec
	}()
	waitChan(t, engine.received, "the first request at the engine")
	p.admission.setAvailability(nodepolicy.Draining, false)
	rec := httptest.NewRecorder()
	f.serveIngress(rec, ingressRequest(tc, tc.inferenceBody(), ""), "peer")
	assertAdmissionRejection(t, rec, nodepolicy.RejectDraining)
	engine.doRelease()
	select {
	case first := <-done:
		if first.Code != http.StatusOK {
			t.Fatalf("the active request did not finish under draining: %d", first.Code)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the active request never finished")
	}

	// cancelActive: the running request is cancelled at the engine.
	stall := newStallingEngine(t)
	setBackend(t, f, stall.URL, true)
	p.admission.setAvailability(nodepolicy.Available, false)
	go func() {
		rec := httptest.NewRecorder()
		f.serveIngress(rec, ingressRequest(tc, tc.inferenceBody(), ""), "peer")
		done <- rec
	}()
	waitChan(t, stall.received, "the request at the engine")
	p.admission.setAvailability(nodepolicy.Draining, true)
	waitChan(t, stall.cancelled, "cancelActive to reach the engine")
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the cancelled ingress request never returned")
	}
	waitForCond(t, time.Second, "active to reach zero", func() bool {
		return p.admission.currentActive() == 0
	})
}

// The real TLS ingress, end to end: a pinned peer's paused node answers the
// router with the admission 503, which the router passes back as its answer
// when it has nowhere else to go.
func TestIngress_ClusterPathAppliesAdmission(t *testing.T) {
	tc := anyCase(t)
	aDir, bDir := t.TempDir(), t.TempDir()
	clustertrusttest.Join(t, aDir, "cluster", "a")
	clustertrusttest.Join(t, bDir, "cluster", "b")
	pin := func(dst, src, id string) {
		t.Helper()
		pem, err := os.ReadFile(filepath.Join(src, "node.crt"))
		if err != nil {
			t.Fatal(err)
		}
		body, err := json.Marshal(map[string]any{"nodeUuid": id, "certPem": string(pem),
			"clusterId": "cluster", "admissionEpoch": 1})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Join(dst, "trusted"), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dst, "trusted", id+".json"), body, 0o600); err != nil {
			t.Fatal(err)
		}
		members, err := json.Marshal([]map[string]any{{"nodeUuid": id, "state": "member", "clusterId": "cluster", "admissionEpoch": 1}})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dst, "members.json"), members, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	pin(aDir, bDir, "b")
	pin(bDir, aDir, "a")

	engine := newRecordingEngine(t, http.StatusOK)
	peer := testProxy(tc.profile, NewDiscovery(), tc.profile.StandalonePort)
	peer.mesh = clustertrust.Open(bDir)
	setBackend(t, peer.soleFacade(), engine.URL, true)
	peer.admission.setAvailability(nodepolicy.Paused, false)
	ingress := httptest.NewUnstartedServer(http.HandlerFunc(peer.soleFacade().handleClusterIngress))
	ingress.TLS = peer.mesh.ServerTLSConfig()
	ingress.StartTLS()
	defer ingress.Close()

	router := testProxy(tc.profile, NewDiscovery(), tc.profile.StandalonePort)
	router.mesh = clustertrust.Open(aDir)
	n := nodeForModel(t, "peer", ingress.URL, tc.advertisedModel)
	n.ClusterUUID = "b"
	router.soleFacade().discovery.SetSubscribed([]Node{n})

	rec := httptest.NewRecorder()
	router.soleFacade().handleHTTP(rec, tc.inferenceRequest())
	assertAdmissionRejection(t, rec, nodepolicy.RejectPaused)
	if engine.hits() != 0 {
		t.Fatal("the paused peer's engine was reached")
	}

	peer.admission.setAvailability(nodepolicy.Available, false)
	rec = httptest.NewRecorder()
	router.soleFacade().handleHTTP(rec, tc.inferenceRequest())
	if rec.Code != http.StatusOK || engine.hits() != 1 {
		t.Fatalf("available peer: status %d, engine hits %d", rec.Code, engine.hits())
	}
}

func TestHandleHTTP_SelfCandidateRejectedFailsOver(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		local := newRecordingEngine(t, http.StatusOK)
		peer := newRecordingEngine(t, http.StatusOK)
		port := tc.profile.FacadePort
		disc := NewDiscovery()
		disc.AddManual(selfNode("self", port, tc.advertisedModel))
		disc.AddManual(nodeForModel(t, "peer", peer.URL, tc.advertisedModel))
		p := testProxy(tc.profile, disc, port)
		f := p.soleFacade()
		setBackend(t, f, local.URL, true)
		p.SetPriority([]string{"self", "peer"})

		// Available: this node serves its own request.
		rec := httptest.NewRecorder()
		f.handleHTTP(rec, tc.inferenceRequest())
		if rec.Code != http.StatusOK || local.hits() != 1 {
			t.Fatalf("available self: status %d, local hits %d", rec.Code, local.hits())
		}
		if local.lastHeader().Get(nodepolicy.AdmissionWaitHeader) != "" {
			t.Fatal("the admission wait header reached this node's own engine")
		}

		// Paused: the self candidate is skipped like a failed one, and the
		// client never sees the rejection.
		p.admission.setAvailability(nodepolicy.Paused, false)
		rec = httptest.NewRecorder()
		f.handleHTTP(rec, tc.inferenceRequest())
		if rec.Code != http.StatusOK || peer.hits() != 1 || local.hits() != 1 {
			t.Fatalf("paused self: status %d, peer hits %d, local hits %d", rec.Code, peer.hits(), local.hits())
		}
		if got := rec.Header().Get(nodepolicy.NodeHeader); got != "peer" {
			t.Fatalf("%s = %q, want peer", nodepolicy.NodeHeader, got)
		}
	})
}

// With nowhere else to go, the self rejection is the answer.
func TestHandleHTTP_SelfOnlyRejectionAnswersWithAdmission503(t *testing.T) {
	tc := anyCase(t)
	local := newRecordingEngine(t, http.StatusOK)
	port := tc.profile.FacadePort
	disc := NewDiscovery()
	disc.AddManual(selfNode("self", port, tc.advertisedModel))
	p := testProxy(tc.profile, disc, port)
	setBackend(t, p.soleFacade(), local.URL, true)
	p.admission.setAvailability(nodepolicy.Paused, false)

	// With the production backoff: a pause will not lift between rounds, so
	// the answer must not wait for them.
	retryBackoff = defaultRetryBackoff
	defer func() { retryBackoff = []time.Duration{time.Millisecond} }()
	started := time.Now()
	rec := httptest.NewRecorder()
	p.soleFacade().handleHTTP(rec, tc.inferenceRequest())
	assertAdmissionRejection(t, rec, nodepolicy.RejectPaused)
	if local.hits() != 0 {
		t.Fatal("a paused node ran its own request")
	}
	if waited := time.Since(started); waited > 500*time.Millisecond {
		t.Fatalf("a paused single node took %v to answer", waited)
	}
}

// The router tells each destination how long it may queue: not at all while
// the round has another candidate, the policy's queue timeout for the last.
func TestHandleHTTP_SendsAdmissionWaitHeader(t *testing.T) {
	tc := anyCase(t)
	busy := newRecordingEngine(t, http.StatusServiceUnavailable)
	ok := newRecordingEngine(t, http.StatusOK)
	disc := NewDiscovery()
	disc.AddManual(nodeForModel(t, "a", busy.URL, tc.advertisedModel))
	disc.AddManual(nodeForModel(t, "b", ok.URL, tc.advertisedModel))
	p := testProxy(tc.profile, disc, tc.profile.FacadePort)
	p.SetPriority([]string{"a", "b"})
	pol := nodepolicy.Default()
	pol.Admission.QueueTimeoutSeconds = 17
	p.admission.setPolicy(pol)

	rec := httptest.NewRecorder()
	p.soleFacade().handleHTTP(rec, tc.inferenceRequest())
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	if got := busy.lastHeader().Get(nodepolicy.AdmissionWaitHeader); got != "0" {
		t.Fatalf("first candidate got wait %q, want 0", got)
	}
	if got := ok.lastHeader().Get(nodepolicy.AdmissionWaitHeader); got != "17" {
		t.Fatalf("last candidate got wait %q, want the queue timeout 17", got)
	}
}

// A busy self candidate with another candidate left hands the request on at
// once instead of queueing it.
func TestHandleHTTP_BusySelfHandsOnImmediately(t *testing.T) {
	tc := anyCase(t)
	local := newRecordingEngine(t, http.StatusOK)
	peer := newRecordingEngine(t, http.StatusOK)
	port := tc.profile.FacadePort
	disc := NewDiscovery()
	disc.AddManual(selfNode("self", port, tc.advertisedModel))
	disc.AddManual(nodeForModel(t, "peer", peer.URL, tc.advertisedModel))
	p := testProxy(tc.profile, disc, port)
	setBackend(t, p.soleFacade(), local.URL, true)
	p.SetPriority([]string{"self", "peer"})
	pol := nodepolicy.Default()
	pol.Admission.MaxConcurrentPerModel = 1
	p.admission.setPolicy(pol)
	held := mustAdmit(t, p.admission.admit(t.Context(), admissionRequest{engine: tc.profile.Name, model: tc.requestedModel, waitCap: -1}))
	defer held.release()

	started := time.Now()
	rec := httptest.NewRecorder()
	p.soleFacade().handleHTTP(rec, tc.inferenceRequest())
	if rec.Code != http.StatusOK || peer.hits() != 1 || local.hits() != 0 {
		t.Fatalf("status %d, peer hits %d, local hits %d", rec.Code, peer.hits(), local.hits())
	}
	if waited := time.Since(started); waited > time.Second {
		t.Fatalf("a busy self candidate held the request for %v", waited)
	}
}

func TestHandleHTTP_SelfDispatchMergesRequestOptions(t *testing.T) {
	tc := anyCase(t)
	local := newRecordingEngine(t, http.StatusOK)
	port := tc.profile.FacadePort
	disc := NewDiscovery()
	disc.AddManual(selfNode("self", port, tc.advertisedModel))
	p := testProxy(tc.profile, disc, port)
	setBackend(t, p.soleFacade(), local.URL, true)
	pol := nodepolicy.Default()
	pol.Profiles = []nodepolicy.Profile{{
		Name: "q", Engine: tc.profile.Name, Model: tc.requestedModel,
		RequestOptions: json.RawMessage(`{"options":{"num_ctx":8192,"temperature":1},"keep_alive":"5m"}`),
	}}
	p.admission.setPolicy(pol)

	body := `{"model":"` + tc.requestedModel + `", "options": {"temperature": 0.1}}`
	rec := httptest.NewRecorder()
	p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, tc.inferencePath, strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	want := `{"model":"` + tc.requestedModel + `", "options": {"temperature": 0.1,"num_ctx":8192},"keep_alive":"5m"}`
	if got := local.lastBody(); got != want {
		t.Fatalf("engine got\n%s\nwant\n%s", got, want)
	}
}

// A stopped engine saved On is woken by a local request: the dormant self
// candidate asks the broker to start it and is served once it reports healthy.
func TestHandleHTTP_WakesStoppedEngineAndWaitsForHealthy(t *testing.T) {
	tc := anyCase(t)
	local := newRecordingEngine(t, http.StatusOK)
	w := &recordingWriter{}
	p := newTestProxy(tc.profile, NewCodec(w), NewDiscovery(), tc.profile.FacadePort)
	f := p.soleFacade()
	setBackend(t, f, local.URL, false)
	p.noteSelf("self-uuid", tc.profile.Name, []string{tc.advertisedModel})
	p.admission.setIntent(map[string]bool{tc.profile.Name: true})

	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		rec := httptest.NewRecorder()
		f.handleHTTP(rec, tc.inferenceRequest())
		done <- rec
	}()
	waitForCond(t, 3*time.Second, "admission/wake", func() bool {
		return len(notificationsOf(t, w, nodepolicy.NotifyAdmissionWake)) > 0
	})
	if local.hits() != 0 {
		t.Fatal("the stopped engine was dialed before it reported healthy")
	}
	setBackend(t, f, local.URL, true)
	select {
	case rec := <-done:
		if rec.Code != http.StatusOK || local.hits() != 1 {
			t.Fatalf("woken engine: status %d, hits %d", rec.Code, local.hits())
		}
		if got := rec.Header().Get(nodepolicy.NodeHeader); got != "self-uuid" {
			t.Fatalf("%s = %q, want self-uuid", nodepolicy.NodeHeader, got)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the request never completed after the engine came up")
	}
}

// An engine saved Off is neither offered nor woken.
func TestHandleHTTP_EngineOffIsNotWoken(t *testing.T) {
	tc := anyCase(t)
	local := newRecordingEngine(t, http.StatusOK)
	w := &recordingWriter{}
	p := newTestProxy(tc.profile, NewCodec(w), NewDiscovery(), tc.profile.FacadePort)
	f := p.soleFacade()
	setBackend(t, f, local.URL, false)
	p.noteSelf("self-uuid", tc.profile.Name, []string{tc.advertisedModel})
	p.admission.setIntent(map[string]bool{tc.profile.Name: false})

	rec := httptest.NewRecorder()
	f.handleHTTP(rec, tc.inferenceRequest())
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want the no-owner 502", rec.Code)
	}
	if n := len(notificationsOf(t, w, nodepolicy.NotifyAdmissionWake)); n != 0 {
		t.Fatalf("sent %d admission/wake for an engine saved Off", n)
	}
}

// cancelActive on a request this node is running itself, before it commits,
// moves the request to another candidate rather than failing it.
func TestHandleHTTP_CancelActiveMovesSelfDispatchOn(t *testing.T) {
	tc := anyCase(t)
	local := newStallingEngine(t)
	peer := newRecordingEngine(t, http.StatusOK)
	port := tc.profile.FacadePort
	disc := NewDiscovery()
	disc.AddManual(selfNode("self", port, tc.advertisedModel))
	disc.AddManual(nodeForModel(t, "peer", peer.URL, tc.advertisedModel))
	w := &recordingWriter{}
	p := newTestProxy(tc.profile, NewCodec(w), disc, port)
	setBackend(t, p.soleFacade(), local.URL, true)
	p.SetPriority([]string{"self", "peer"})

	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		rec := httptest.NewRecorder()
		p.soleFacade().handleHTTP(rec, tc.inferenceRequest())
		done <- rec
	}()
	waitChan(t, local.received, "the request at this node's engine")
	rpcOK(t, p, w, nodepolicy.MethodSetAvailability, nodepolicy.SetAvailabilityParams{State: nodepolicy.Draining, CancelActive: true})
	waitChan(t, local.cancelled, "the local execution to be cancelled")
	select {
	case rec := <-done:
		if rec.Code != http.StatusOK || peer.hits() != 1 {
			t.Fatalf("status %d, peer hits %d", rec.Code, peer.hits())
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the request never completed elsewhere")
	}
	waitForCond(t, time.Second, "active to reach zero", func() bool {
		return p.admission.currentActive() == 0
	})
}

func TestHandleHTTP_SelfAdmissionCancelStaysWithAttempt(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		local := newStallingEngine(t)
		peer := newStallingEngine(t)
		port := tc.profile.FacadePort
		disc := NewDiscovery()
		disc.AddManual(selfNode("self", port, tc.advertisedModel))
		disc.AddManual(nodeForModel(t, "peer", peer.URL, tc.advertisedModel))
		p := newTestProxy(tc.profile, NewCodec(&recordingWriter{}), disc, port)
		setBackend(t, p.soleFacade(), local.URL, true)
		p.SetPriority([]string{"self", "peer"})

		done := make(chan *httptest.ResponseRecorder, 1)
		go func() {
			rec := httptest.NewRecorder()
			p.soleFacade().handleHTTP(rec, tc.inferenceRequest())
			done <- rec
		}()
		waitChan(t, local.received, "the self attempt")

		// Collect the actual ticket callback as setAvailability does, then
		// invoke it outside the admission lock at controlled attempt boundaries.
		p.admission.mu.Lock()
		var cancel func()
		for _, ticket := range p.admission.tickets {
			cancel = ticket.cancel
		}
		tickets := len(p.admission.tickets)
		p.admission.mu.Unlock()
		if tickets != 1 || cancel == nil {
			t.Fatalf("self admission has %d tickets, callback present = %t", tickets, cancel != nil)
		}

		// While its attempt is current, the callback must still abort it.
		cancel()
		waitChan(t, local.cancelled, "the self attempt to be cancelled")
		waitChan(t, peer.received, "the replacement attempt")

		p.workloadsMu.Lock()
		ctl := p.workloads[workloadKey{engine: tc.profile.Name, id: "1"}]
		p.workloadsMu.Unlock()
		if ctl == nil {
			t.Fatal("replacement workload is missing")
		}

		// The same callback can outlive its released ticket. It must not
		// abandon or exclude the replacement, even before that attempt commits.
		cancel()
		ctl.mu.Lock()
		att := ctl.attempt
		peerAbandoned := att != nil && att.claim.abandoned()
		peerExcluded := ctl.excluded["peer"]
		ctl.mu.Unlock()
		if att == nil || peerAbandoned || peerExcluded {
			t.Fatalf("stale callback changed replacement: missing=%t abandoned=%t excluded=%t", att == nil, peerAbandoned, peerExcluded)
		}

		peer.doRelease()
		select {
		case rec := <-done:
			if rec.Code != http.StatusOK || rec.Header().Get(nodepolicy.NodeHeader) != "peer" {
				t.Fatalf("replacement response: status=%d node=%q", rec.Code, rec.Header().Get(nodepolicy.NodeHeader))
			}
		case <-time.After(3 * time.Second):
			t.Fatal("replacement attempt did not complete")
		}
	})
}
