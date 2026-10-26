// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/schedulerwire"
)

// These fixtures run only proxy components and loopback HTTP engines. The
// synthetic parent drives correlated replies and publication explicitly; no
// installed engine, lifecycle worker, discovery network or process is used.
type bindingRequestFixture struct {
	p        *Proxy
	f        *facade
	requests chan nodepolicy.PrepareBoundEngineRequest
	cancels  chan nodepolicy.CancelBoundEngineRequest
	wakes    chan nodepolicy.WakeRequest
}

func newBindingRequestFixture(t *testing.T) *bindingRequestFixture {
	t.Helper()
	fx := &bindingRequestFixture{requests: make(chan nodepolicy.PrepareBoundEngineRequest, 8), cancels: make(chan nodepolicy.CancelBoundEngineRequest, 8), wakes: make(chan nodepolicy.WakeRequest, 8)}
	w := bufferAckWriter(func(raw []byte) (int, error) {
		var msg Message
		if err := json.Unmarshal(raw, &msg); err != nil {
			return 0, err
		}
		switch msg.Method {
		case nodepolicy.MethodPrepareBoundEngine:
			var in nodepolicy.PrepareBoundEngineRequest
			if err := json.Unmarshal(msg.Params, &in); err != nil {
				return 0, err
			}
			fx.requests <- in
		case nodepolicy.MethodCancelBoundEngine:
			var in nodepolicy.CancelBoundEngineRequest
			if err := json.Unmarshal(msg.Params, &in); err != nil {
				return 0, err
			}
			fx.cancels <- in
		case nodepolicy.NotifyAdmissionWake:
			var in nodepolicy.WakeRequest
			if err := json.Unmarshal(msg.Params, &in); err != nil {
				return 0, err
			}
			fx.wakes <- in
		}
		return len(raw), nil
	})
	profile := ollamaCase(t).profile
	fx.p = newTestProxy(profile, NewCodec(w), NewDiscovery(), profile.FacadePort)
	fx.p.originNodeID = "self"
	fx.f = fx.p.soleFacade()
	pol := nodepolicy.Default()
	pol.Revision = "1"
	pol.EngineBindings = nodepolicy.EngineBindings{{Engine: profile.Name, Model: "fixture"}}
	pol.Tiers = map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: profile.Name, Model: "fixture"}}}
	fx.p.admission.setPolicy(pol)
	fx.p.admission.setIntent(map[string]bool{profile.Name: false})
	fx.offer(t, fx.f, "fixture")
	return fx
}

func (fx *bindingRequestFixture) offer(t *testing.T, f *facade, model string) {
	t.Helper()
	n := selfNode("self", f.profile.FacadePort)
	n.PolicyRevision = fx.p.admission.currentPolicy().Revision
	n.EngineBindingOffers = bindingOffer(f.profile.Name, model)
	f.discovery.AddManual(n)
	fx.p.noteSelf("self", f.profile.Name, nil)
}

func (fx *bindingRequestFixture) next(t *testing.T) nodepolicy.PrepareBoundEngineRequest {
	t.Helper()
	select {
	case in := <-fx.requests:
		return in
	case <-time.After(3 * time.Second):
		t.Fatal("no bound preparation notification")
		return nodepolicy.PrepareBoundEngineRequest{}
	}
}

func (fx *bindingRequestFixture) reply(t *testing.T, in nodepolicy.PrepareBoundEngineRequest, result nodepolicy.BoundEnginePrepared) {
	t.Helper()
	result.RequestID, result.ProxyRunID = in.RequestID, in.ProxyRunID
	if !result.Success {
		result.Code = -32000
		result.Message = "Synthetic preparation refusal."
	}
	fx.p.receiveBoundEnginePrepared(egressJSON(t, result))
}

type bindingResponseWriter struct {
	*httptest.ResponseRecorder
	wrote chan struct{}
	once  sync.Once
}

func (w *bindingResponseWriter) WriteHeader(code int) {
	w.once.Do(func() { close(w.wrote) })
	w.ResponseRecorder.WriteHeader(code)
}
func (w *bindingResponseWriter) Write(b []byte) (int, error) {
	w.once.Do(func() { close(w.wrote) })
	return w.ResponseRecorder.Write(b)
}
func (w *bindingResponseWriter) Flush() {
	w.once.Do(func() { close(w.wrote) })
	w.ResponseRecorder.Flush()
}

func runBindingHTTP(t *testing.T, f *facade, req *http.Request) (*bindingResponseWriter, <-chan struct{}) {
	t.Helper()
	w := &bindingResponseWriter{ResponseRecorder: httptest.NewRecorder(), wrote: make(chan struct{})}
	done := make(chan struct{})
	ctx, cancel := context.WithCancel(req.Context())
	t.Cleanup(cancel)
	go func() { defer close(done); f.handleHTTP(w, req.WithContext(ctx)) }()
	return w, done
}

func bindingRequest(model, path string) *http.Request {
	return httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"model":"`+model+`","stream":false}`))
}

func assertBindingRefusal(t *testing.T, rec *httptest.ResponseRecorder, reason string) {
	t.Helper()
	var result struct{ Code, Error, Reason string }
	if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if rec.Code != http.StatusServiceUnavailable || result.Code != "admission" || result.Reason != reason || result.Error == "" || rec.Header().Get(nodepolicy.AdmissionHeader) != reason {
		t.Fatalf("status=%d result=%+v headers=%v", rec.Code, result, rec.Header())
	}
}

type bindingEngine struct {
	*httptest.Server
	hits             atomic.Int32
	mu               sync.Mutex
	header           http.Header
	body             []byte
	inventoryEntered chan struct{}
	inventoryRelease <-chan struct{}
	listOnce         sync.Once
}

func newBindingEngine(t *testing.T, model string, inventoryRelease <-chan struct{}) *bindingEngine {
	t.Helper()
	e := &bindingEngine{inventoryEntered: make(chan struct{}), inventoryRelease: inventoryRelease}
	e.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			e.mu.Lock()
			e.header = r.Header.Clone()
			e.mu.Unlock()
			e.listOnce.Do(func() { close(e.inventoryEntered) })
			if e.inventoryRelease != nil {
				select {
				case <-e.inventoryRelease:
				case <-r.Context().Done():
					return
				}
			}
			if r.URL.Path == "/api/tags" {
				_ = json.NewEncoder(w).Encode(map[string]any{"models": []map[string]string{{"model": model}}})
			} else {
				_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"id": model}}})
			}
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read synthetic inference: %v", err)
			return
		}
		e.mu.Lock()
		e.header, e.body = r.Header.Clone(), body
		e.mu.Unlock()
		e.hits.Add(1)
		if openAICompatibleInference[r.URL.Path] {
			_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"fixture"},"finish_reason":"stop"}]}`)
		} else {
			_, _ = io.WriteString(w, `{"done":true}`)
		}
	}))
	t.Cleanup(e.Close)
	return e
}

func TestEngineBindingRequestsSelfPreparationBeforeAdmission(t *testing.T) {
	test := func(name, model string) {
		t.Run(name, func(t *testing.T) {
			fx := newBindingRequestFixture(t)
			e := newBindingEngine(t, "fixture", nil)
			w, done := runBindingHTTP(t, fx.f, bindingRequest(model, "/api/generate"))
			in := fx.next(t)
			if in.Model != "fixture" || in.Engine != "ollama" || in.PolicyRevision != "1" || in.PeerID != "" || in.PeerIncarnation != "" || in.BudgetMs < 1 || in.BudgetMs > 90000 {
				t.Fatalf("request=%+v", in)
			}
			select {
			case <-w.wrote:
				t.Fatal("output before preparation")
			default:
			}
			if fx.p.admission.currentActive() != 0 {
				t.Fatal("preparation holds execution/resident/strict admission")
			}
			fx.p.priorityMu.Lock()
			held := len(fx.p.priorityReservations)
			fx.p.priorityMu.Unlock()
			if held != 0 {
				t.Fatal("preparation holds placement reservation")
			}
			setBackend(t, fx.f, e.URL, true)
			fx.reply(t, in, nodepolicy.BoundEnginePrepared{Success: true, SwitchRequestID: "binding:run:1"})
			waitChan(t, done, "prepared inference")
			if w.Code != http.StatusOK || e.hits.Load() != 1 {
				t.Fatalf("status=%d hits=%d body=%s", w.Code, e.hits.Load(), w.Body)
			}
			if w.Header().Get(nodepolicy.ModelHeader) != "fixture" || w.Header().Get(nodepolicy.EngineHeader) != "ollama" || w.Header().Get(nodepolicy.NodeHeader) != "self" {
				t.Fatalf("final headers=%v", w.Header())
			}
			if model == "weak" && w.Header().Get(nodepolicy.TierHeader) != "weak" {
				t.Fatal("tier identity lost")
			}
			if fx.p.admission.currentActive() != 0 {
				t.Fatal("completed request retained admission")
			}
		})
	}
	test("ordinary concrete", "fixture")
	test("existing concrete tier", "weak")
}

func TestEngineBindingRequestsRefusalReasons(t *testing.T) {
	test := func(reason string, transient bool) {
		t.Run(reason, func(t *testing.T) {
			fx := newBindingRequestFixture(t)
			w, done := runBindingHTTP(t, fx.f, bindingRequest("fixture", "/api/generate"))
			in := fx.next(t)
			fx.reply(t, in, nodepolicy.BoundEnginePrepared{Reason: reason, SwitchRequestID: "binding:run:1"})
			waitChan(t, done, "refused preparation")
			assertBindingRefusal(t, w.ResponseRecorder, reason)
			if (w.Header().Get("Retry-After") != "") != transient || w.Header().Get(nodepolicy.SwitchRequestIDHeader) != "binding:run:1" {
				t.Fatalf("refusal headers=%v", w.Header())
			}
			select {
			case next := <-fx.requests:
				t.Fatalf("resubmitted refused/uncertain acceptance: %+v", next)
			default:
			}
		})
	}
	test("engine-switch-timeout", true)
	test("engine-switch-busy", true)
	test("engine-switch-cooldown", true)
	test("engine-locked", false)
	test("management-denied", false)
	test("model-unavailable", false)
	test("paused", true)
	test("draining", true)
	test("no-fit", true)
}

func TestEngineBindingRequestsReceiptDoesNotProveModelPresence(t *testing.T) {
	fx := newBindingRequestFixture(t)
	e := newBindingEngine(t, "different-model", nil)
	w, done := runBindingHTTP(t, fx.f, bindingRequest("fixture", "/api/generate"))
	in := fx.next(t)
	setBackend(t, fx.f, e.URL, true)
	// Even stale positive discovery cannot override fresh canonical inventory.
	n := selfNode("self", fx.f.profile.FacadePort, "fixture")
	n.PolicyRevision, n.EngineBindingOffers = "1", bindingOffer("ollama", "fixture")
	fx.f.discovery.AddManual(n)
	fx.reply(t, in, nodepolicy.BoundEnginePrepared{Success: true, SwitchRequestID: "binding:run:1"})
	waitChan(t, done, "model absence refusal")
	assertBindingRefusal(t, w.ResponseRecorder, "model-unavailable")
	if e.hits.Load() != 0 || w.Header().Get(nodepolicy.SwitchRequestIDHeader) != "binding:run:1" {
		t.Fatal("receipt authorized absent model or lost accepted ID")
	}
}

func TestEngineBindingRequestsRevalidateAfterInventoryWait(t *testing.T) {
	test := func(name string, change func(*bindingRequestFixture)) {
		t.Run(name, func(t *testing.T) {
			fx := newBindingRequestFixture(t)
			release := make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			e := newBindingEngine(t, "fixture", release)
			t.Cleanup(unblock)
			fx.p.admission.setModelIdentity(map[string]nodepolicy.EngineIdentity{"ollama": {EngineRunID: "initial", ModelDigests: map[string]string{"fixture:latest": "content"}}})
			w, done := runBindingHTTP(t, fx.f, bindingRequest("weak", "/api/generate"))
			in := fx.next(t)
			setBackend(t, fx.f, e.URL, true)
			fx.reply(t, in, nodepolicy.BoundEnginePrepared{Success: true})
			waitChan(t, e.inventoryEntered, "canonical inventory barrier")
			change(fx)
			unblock()
			waitChan(t, done, "freshness refusal")
			assertBindingRefusal(t, w.ResponseRecorder, "model-unavailable")
			if e.hits.Load() != 0 || fx.p.admission.currentActive() != 0 {
				t.Fatal("stale evidence dispatched or held ticket")
			}
		})
	}
	test("changed engine identity", func(fx *bindingRequestFixture) {
		fx.p.admission.setModelIdentity(map[string]nodepolicy.EngineIdentity{"ollama": {EngineRunID: "replacement"}})
	})
	test("removed binding", func(fx *bindingRequestFixture) {
		pol := fx.p.admission.currentPolicy()
		pol.Revision = "2"
		pol.EngineBindings = nil
		fx.p.admission.setPolicy(pol)
	})
	test("serving permission withdrawn", func(fx *bindingRequestFixture) {
		pol := fx.p.admission.currentPolicy()
		pol.Revision = "2"
		pol.ServedModels = nodepolicy.ServedModels{}
		fx.p.admission.setPolicy(pol)
	})
	test("tier context changed", func(fx *bindingRequestFixture) {
		pol := fx.p.admission.currentPolicy()
		pol.Revision = "2"
		pol.Tiers = map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "other", ContextTokens: 1}}}
		fx.p.admission.setPolicy(pol)
	})
	test("backend withdrawn", func(fx *bindingRequestFixture) {
		if err := fx.f.setLocalBackend(localBackend{Engine: "ollama"}); err != nil {
			t.Fatal(err)
		}
	})
}

func TestEngineBindingRequestsCancellationHasNoEarlyOutput(t *testing.T) {
	fx := newBindingRequestFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	w, done := runBindingHTTP(t, fx.f, bindingRequest("fixture", "/api/generate").WithContext(ctx))
	in := fx.next(t)
	cancel()
	waitChan(t, done, "caller cancellation")
	select {
	case got := <-fx.cancels:
		if got.RequestID != in.RequestID || got.ProxyRunID != in.ProxyRunID {
			t.Fatalf("cancel=%+v request=%+v", got, in)
		}
	case <-time.After(time.Second):
		t.Fatal("no correlated cancellation")
	}
	fx.reply(t, in, nodepolicy.BoundEnginePrepared{Success: true})
	select {
	case <-w.wrote:
		t.Fatal("cancelled waiter emitted output")
	default:
	}
	if fx.p.admission.currentActive() != 0 {
		t.Fatal("cancelled preparation retained execution")
	}
	select {
	case got := <-fx.requests:
		t.Fatalf("unknown acceptance submitted new request: %+v", got)
	default:
	}
}

func TestEngineBindingRequestsExpiredBudgetHasNoAcceptanceID(t *testing.T) {
	fx := newBindingRequestFixture(t)
	req := bindingRequest("fixture", "/api/generate")
	req.Header.Set(nodepolicy.EnginePrepareBudgetHeader, "40")
	w, done := runBindingHTTP(t, fx.f, req)
	in := fx.next(t)
	if in.BudgetMs < 1 || in.BudgetMs > 40 {
		t.Fatalf("remaining budget=%d", in.BudgetMs)
	}
	waitChan(t, done, "preparation deadline")
	assertBindingRefusal(t, w.ResponseRecorder, "engine-switch-timeout")
	if w.Header().Get(nodepolicy.SwitchRequestIDHeader) != "" {
		t.Fatal("invented acceptance ID")
	}
	fx.reply(t, in, nodepolicy.BoundEnginePrepared{Success: true, SwitchRequestID: "binding:late:1"})
	select {
	case got := <-fx.requests:
		t.Fatalf("unknown acceptance replayed as new request: %+v", got)
	default:
	}
}

func TestEngineBindingRequestsCrossFacadeUsesFinalEngineAndDefaults(t *testing.T) {
	fx := newBindingRequestFixture(t)
	lm := newFacade(fx.p, lmstudioCase(t).profile, NewDiscovery(), lmstudioCase(t).profile.FacadePort)
	fx.p.facades[lm.profile.Name] = lm
	pol := fx.p.admission.currentPolicy()
	pol.EngineBindings = nodepolicy.EngineBindings{{Engine: "lmstudio", Model: "fixture"}}
	pol.Profiles = []nodepolicy.Profile{{Engine: "ollama", Model: "fixture", RequestOptions: json.RawMessage(`{"temperature":0.1}`)}, {Engine: "lmstudio", Model: "fixture", RequestOptions: json.RawMessage(`{"temperature":0.7}`)}}
	fx.p.admission.setPolicy(pol)
	fx.offer(t, lm, "fixture")
	e := newBindingEngine(t, "fixture", nil)
	req := bindingRequest("fixture", routeChatCompletions)
	req.Header.Set("Authorization", "Bearer caller-secret")
	req.Header.Set("X-Api-Key", "caller-api-key")
	req.Header.Set("X-PAIR-Peer-Id", "forged-peer")
	req.Header.Set("X-PAIR-Activation-Identity", "forged-authority")
	req.Header.Set(nodepolicy.EnginePrepareBudgetHeader, "1000")
	req.Header.Set("X-Ordinary-Option", "keep")
	w, done := runBindingHTTP(t, fx.f, req)
	in := fx.next(t)
	if in.Engine != "lmstudio" || in.PeerID != "" || in.PeerIncarnation != "" {
		t.Fatalf("cross-facade demand=%+v", in)
	}
	setBackend(t, lm, e.URL, true)
	backend := lm.currentLocalBackend()
	backend.Credential = "native-engine-secret"
	if err := lm.setLocalBackend(backend); err != nil {
		t.Fatal(err)
	}
	fx.reply(t, in, nodepolicy.BoundEnginePrepared{Success: true})
	waitChan(t, done, "cross-facade request")
	if w.Code != http.StatusOK || e.hits.Load() != 1 || w.Header().Get(nodepolicy.EngineHeader) != "lmstudio" || w.Header().Get(nodepolicy.ModelHeader) != "fixture" || w.Header().Get(nodepolicy.NodeHeader) != "self" {
		t.Fatalf("status=%d headers=%v body=%s", w.Code, w.Header(), w.Body)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	var body struct {
		Model       string
		Temperature float64
	}
	if err := json.Unmarshal(e.body, &body); err != nil {
		t.Fatal(err)
	}
	if body.Model != "fixture" || body.Temperature != 0.7 {
		t.Fatalf("final body=%+v", body)
	}
	if e.header.Get("Authorization") != "Bearer native-engine-secret" || e.header.Get("X-Api-Key") != "" || e.header.Get("X-Ordinary-Option") != "keep" {
		t.Fatalf("engine headers=%v", e.header)
	}
	for key := range e.header {
		if strings.HasPrefix(strings.ToLower(key), "x-pair-") {
			t.Fatalf("private header reached engine: %s", key)
		}
	}
}

func TestEngineBindingRequestsUnboundAndNativeStayEngineLocal(t *testing.T) {
	test := func(name, path string, bound bool) {
		t.Run(name, func(t *testing.T) {
			fx := newBindingRequestFixture(t)
			lm := newFacade(fx.p, lmstudioCase(t).profile, NewDiscovery(), lmstudioCase(t).profile.FacadePort)
			fx.p.facades[lm.profile.Name] = lm
			pol := fx.p.admission.currentPolicy()
			pol.EngineBindings = nil
			if bound {
				pol.EngineBindings = nodepolicy.EngineBindings{{Engine: "lmstudio", Model: "fixture"}}
			}
			fx.p.admission.setPolicy(pol)
			fx.f.discovery.RemoveManual("self")
			e := newBindingEngine(t, "fixture", nil)
			setBackend(t, lm, e.URL, true)
			lm.discovery.AddManual(selfNode("self", lm.profile.FacadePort, "fixture"))
			w := httptest.NewRecorder()
			fx.f.handleHTTP(w, bindingRequest("fixture", path))
			if w.Code != http.StatusBadGateway || e.hits.Load() != 0 {
				t.Fatalf("crossed unbound/incompatible protocol: status=%d hits=%d", w.Code, e.hits.Load())
			}
		})
	}
	test("unbound OpenAI name", routeChatCompletions, false)
	test("bound name on native Ollama protocol", "/api/generate", true)
}

func TestEngineBindingRequestsDirectConflictingIngressRefuses(t *testing.T) {
	test := func(name string, strict bool) {
		t.Run(name, func(t *testing.T) {
			fx := newBindingRequestFixture(t)
			pol := fx.p.admission.currentPolicy()
			pol.EngineBindings = nodepolicy.EngineBindings{{Engine: "lmstudio", Model: "fixture"}}
			fx.p.admission.setPolicy(pol)
			e := newBindingEngine(t, "fixture", nil)
			setBackend(t, fx.f, e.URL, true)
			req := bindingRequest("fixture", routeChatCompletions)
			if strict {
				envelope := strictRequest()
				envelope.Request.OriginNodeID = "peer"
				req.Header.Set(nodepolicy.StrictAdmissionHeader, string(egressJSON(t, envelope)))
			}
			w := httptest.NewRecorder()
			fx.f.serveIngress(w, req, "peer")
			assertBindingRefusal(t, w, "model-unavailable")
			if e.hits.Load() != 0 || fx.p.admission.currentActive() != 0 {
				t.Fatal("old ingress ignored binding")
			}
			select {
			case in := <-fx.requests:
				t.Fatalf("conflicting ingress attempted onward preparation: %+v", in)
			default:
			}
		})
	}
	test("ordinary", false)
	test("stale strict envelope", true)
}

func TestEngineBindingRequestsReadyAlternativeBeatsSchedulerCold(t *testing.T) {
	fx := newBindingRequestFixture(t)
	ready := newBindingEngine(t, "fixture", nil)
	fx.f.discovery.AddManual(nodeForModel(t, "ready", ready.URL, "fixture"))
	applySnapshot(fx.p, schedulerwire.Priority{Nodes: []string{"self", "ready"}, Ranks: []schedulerwire.NodeRank{{ID: "self"}, {ID: "ready", Pending: 100}}})
	w := httptest.NewRecorder()
	fx.f.handleHTTP(w, bindingRequest("fixture", "/api/generate"))
	if w.Code != http.StatusOK || ready.hits.Load() != 1 || w.Header().Get(nodepolicy.NodeHeader) != "ready" {
		t.Fatalf("ready preference: status=%d headers=%v", w.Code, w.Header())
	}
	select {
	case in := <-fx.requests:
		t.Fatalf("scheduler triggered cold replacement: %+v", in)
	default:
	}
}

func pinnedBindingClient(t *testing.T, fx *bindingRequestFixture, decorate ...func(*http.Request)) (*http.Client, string, string) {
	t.Helper()
	selfDir, peerDir := t.TempDir(), t.TempDir()
	clustertrusttest.Join(t, selfDir, "fixture-cluster", "destination")
	clustertrusttest.Join(t, peerDir, "fixture-cluster", "authenticated-origin")
	proxyReversePin(t, selfDir, peerDir, "authenticated-origin")
	proxyReversePin(t, peerDir, selfDir, "destination")
	fx.p.mesh = clustertrust.Open(selfDir)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for _, apply := range decorate {
			apply(r)
		}
		fx.f.handleClusterIngress(w, r)
	}))
	server.TLS = fx.p.mesh.ServerTLSConfig()
	server.StartTLS()
	t.Cleanup(server.Close)
	tlsConfig, ok := clustertrust.Open(peerDir).ClientTLSConfig("destination")
	if !ok {
		t.Fatal("missing pinned TLS configuration")
	}
	transport := &http.Transport{TLSClientConfig: tlsConfig}
	t.Cleanup(transport.CloseIdleConnections)
	return &http.Client{Transport: transport, Timeout: 3 * time.Second}, server.URL, selfDir
}

// Wire-protocol tests record at a pinned PAIR receiving boundary. A directly
// configured manual engine is no longer allowed to receive these headers.
func pinRecordingPairPeer(t *testing.T, f *facade, id string, receiver *recordingEngine) {
	t.Helper()
	p := f.host
	if p.mesh == nil {
		p.baseConfigDir = t.TempDir()
		selfDir := filepath.Join(p.baseConfigDir, "cluster")
		clustertrusttest.Join(t, selfDir, "fixture-cluster", p.originNodeID)
		p.mesh = clustertrust.Open(selfDir)
		t.Cleanup(p.closeIdleTransports)
	}
	selfDir := filepath.Join(p.baseConfigDir, "cluster")
	peerDir, uuid := t.TempDir(), "recording-"+id
	clustertrusttest.Join(t, peerDir, "fixture-cluster", uuid)
	var members []map[string]any
	data, err := os.ReadFile(filepath.Join(selfDir, "members.json"))
	if err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}
	if err == nil {
		if err := json.Unmarshal(data, &members); err != nil {
			t.Fatal(err)
		}
	}
	proxyReversePin(t, selfDir, peerDir, uuid)
	proxyReversePin(t, peerDir, selfDir, p.originNodeID)
	members = append(members, map[string]any{"nodeUuid": uuid, "state": "member", "clusterId": "fixture-cluster", "admissionEpoch": 1})
	data, err = json.Marshal(members)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(selfDir, "members.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	p.mesh.Refresh()
	mesh := clustertrust.Open(peerDir)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		principal, ok := mesh.VerifyClientPin(r)
		if !ok || principal != p.originNodeID {
			t.Error("recording PAIR receiver lacks authenticated origin")
			w.WriteHeader(http.StatusForbidden)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
			return
		}
		stream, err := parseOriginStreamRequest(r.URL.Path, body)
		if err != nil {
			t.Error(err)
			return
		}
		w.Header().Set(peerStreamFramingHeader, stream.peerMetadata())
		w.Header().Add("Trailer", peerStreamFramingHeader)
		setReplayBody(r, body)
		receiver.Config.Handler.ServeHTTP(w, r)
		// This recorder writes its complete response synchronously. A pinned
		// receiver must attest that ending as well as its initial framing;
		// otherwise a real HTTP origin correctly aborts on the missing trailer.
		w.Header().Set(peerStreamFramingHeader, peerDisposition(streamFrameResult{Outcome: streamOutcomeCompleted}))
	}))
	server.TLS = mesh.ServerTLSConfig()
	server.StartTLS()
	t.Cleanup(server.Close)
	n, ok := discoveredNode(f, id)
	if !ok {
		t.Fatalf("missing recording node %s", id)
	}
	endpoint := nodeFor(t, id, server.URL)
	n.Addresses, n.Port, n.ClusterUUID = endpoint.Addresses, endpoint.Port, uuid
	f.discovery.AddManual(n)
}

func TestEngineBindingRequestsPinnedPrincipalAndReadyPermission(t *testing.T) {
	fx := newBindingRequestFixture(t)
	client, ingressURL, _ := pinnedBindingClient(t, fx)
	request := func() *http.Request {
		t.Helper()
		req, err := http.NewRequest(http.MethodPost, ingressURL+routeChatCompletions, strings.NewReader(`{"model":"fixture","stream":false,"peerId":"body-forgery"}`))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("X-PAIR-Peer-Id", "header-forgery")
		req.Header.Set("X-PAIR-Peer-Incarnation", "forged-incarnation")
		req.Header.Set(nodepolicy.EnginePrepareBudgetHeader, "1000")
		return req
	}
	type response struct {
		status int
		body   []byte
		err    error
		header http.Header
	}
	responses := make(chan response, 1)
	firstRequest := request()
	go func() {
		resp, err := client.Do(firstRequest)
		if err != nil {
			responses <- response{err: err}
			return
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		responses <- response{status: resp.StatusCode, body: body, err: err, header: resp.Header}
	}()
	in := fx.next(t)
	incarnation, ok := fx.p.mesh.PeerServiceIncarnation("authenticated-origin")
	if !ok || in.PeerID != "authenticated-origin" || in.PeerIncarnation != incarnation {
		t.Fatalf("pinned authority=%+v expected=%s", in, incarnation)
	}
	fx.reply(t, in, nodepolicy.BoundEnginePrepared{Reason: "management-denied"})
	first := <-responses
	if first.err != nil {
		t.Fatal(first.err)
	}
	var refusal struct{ Reason string }
	if err := json.Unmarshal(first.body, &refusal); err != nil {
		t.Fatal(err)
	}
	if first.status != http.StatusServiceUnavailable || refusal.Reason != "management-denied" || first.header.Get(nodepolicy.SwitchRequestIDHeader) != "" {
		t.Fatalf("denied activation=%+v result=%+v", first, refusal)
	}
	// Inference permission is independent of management. Ready traffic neither
	// submits another activation nor inherits caller-supplied identities.
	e := newBindingEngine(t, "fixture", nil)
	setBackend(t, fx.f, e.URL, true)
	fx.p.noteSelf("self", "ollama", []string{"fixture"})
	resp, err := client.Do(request())
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if _, err := io.Copy(io.Discard, resp.Body); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK || e.hits.Load() != 1 || resp.Header.Get(nodepolicy.EngineHeader) != "ollama" || resp.Header.Get(nodepolicy.NodeHeader) != "self" {
		t.Fatalf("ready inference: status=%d headers=%v", resp.StatusCode, resp.Header)
	}
	select {
	case in := <-fx.requests:
		t.Fatalf("ready peer required management: %+v", in)
	default:
	}
}

func TestEngineBindingRequestsTotalBudgetAcrossFailover(t *testing.T) {
	fx := newProxyReverseFixture(t, "ollama", http.NotFoundHandler())
	fx.directHealthy.Store(true)
	// Preparation metadata belongs to pinned PAIR endpoints. These facade
	// fixtures verify the certificate before recording the received allowance.
	peer := func(handler http.Handler) *httptest.Server {
		t.Helper()
		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if _, ok := fx.peer.mesh.VerifyClientPin(r); !ok {
				t.Error("budget reached unauthenticated peer")
				w.WriteHeader(http.StatusForbidden)
				return
			}
			handler.ServeHTTP(w, r)
		}))
		server.TLS = fx.peer.mesh.ServerTLSConfig()
		server.StartTLS()
		t.Cleanup(server.Close)
		return server
	}
	var budgets []time.Duration
	var mu sync.Mutex
	first := peer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		budgets = append(budgets, nodepolicy.EnginePrepareBudget(r.Header.Values(nodepolicy.EnginePrepareBudgetHeader)))
		mu.Unlock()
		if r.Header.Get(nodepolicy.AdmissionWaitHeader) != "0" {
			t.Errorf("ready alternative queued: %s", r.Header.Get(nodepolicy.AdmissionWaitHeader))
		}
		timer := time.NewTimer(40 * time.Millisecond)
		defer timer.Stop()
		select {
		case <-timer.C:
		case <-r.Context().Done():
			return
		}
		writeBoundEngineRefusal(w, boundEngineRefusal("engine-switch-cooldown"))
	}))
	second := peer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		budgets = append(budgets, nodepolicy.EnginePrepareBudget(r.Header.Values(nodepolicy.EnginePrepareBudgetHeader)))
		mu.Unlock()
		if parseAdmissionWait(r.Header) <= 0 {
			t.Error("cold wait not allowed after ready choices")
		}
		timer := time.NewTimer(nodepolicy.EnginePrepareBudget(r.Header.Values(nodepolicy.EnginePrepareBudgetHeader)))
		defer timer.Stop()
		select {
		case <-timer.C:
		case <-r.Context().Done():
			return
		}
		writeBoundEngineRefusal(w, boundEngineRefusal("engine-switch-timeout"))
	}))
	oll := fx.p.soleFacade()
	ready := nodeForModel(t, "a-ready", first.URL, "fixture")
	ready.ClusterUUID = "uuid-peer"
	oll.discovery.SetSubscribed([]Node{ready})
	lm := newFacade(fx.p, lmstudioCase(t).profile, NewDiscovery(), lmstudioCase(t).profile.FacadePort)
	fx.p.facades[lm.profile.Name] = lm
	cold := nodeFor(t, "b-cold", second.URL)
	cold.PolicyRevision, cold.ClusterUUID = "1", "uuid-peer"
	cold.EngineBindingOffers = bindingOffer("lmstudio", "fixture")
	lm.discovery.SetSubscribed([]Node{cold})
	pol := nodepolicy.Default()
	pol.Tiers = map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "fixture"}, {Kind: nodepolicy.EntryLocal, Engine: "lmstudio", Model: "fixture"}}}
	fx.p.admission.setPolicy(pol)
	req := bindingRequest("weak", routeChatCompletions)
	req.Header.Set(nodepolicy.EnginePrepareBudgetHeader, "250")
	w := httptest.NewRecorder()
	started := time.Now()
	oll.handleHTTP(w, req)
	assertBindingRefusal(t, w, "engine-switch-timeout")
	if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
		t.Fatalf("total preparation wait reset: %v", elapsed)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(budgets) != 2 || budgets[0] > 250*time.Millisecond || budgets[1] >= budgets[0]-25*time.Millisecond || budgets[1] <= 0 {
		t.Fatalf("reset/enlarged budgets=%v", budgets)
	}
}

func TestEngineBindingRequestsOldStreamDrainsWithoutWaitingTicket(t *testing.T) {
	fx := newBindingRequestFixture(t)
	lm := newFacade(fx.p, lmstudioCase(t).profile, NewDiscovery(), lmstudioCase(t).profile.FacadePort)
	fx.p.facades[lm.profile.Name] = lm
	pol := fx.p.admission.currentPolicy()
	pol.EngineBindings = nodepolicy.EngineBindings{{Engine: "lmstudio", Model: "fixture"}}
	fx.p.admission.setPolicy(pol)
	fx.offer(t, lm, "fixture")
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	defer unblock()
	old := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"fixture\"},\"finish_reason\":null}]}\n\n")
		w.(http.Flusher).Flush()
		close(entered)
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer func() { unblock(); old.Close() }()
	setBackend(t, fx.f, old.URL, true)
	fx.f.discovery.AddManual(selfNode("self", fx.f.profile.FacadePort, "old"))
	fx.p.noteSelf("self", "ollama", []string{"old"})
	_, oldDone := runBindingHTTP(t, fx.f, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"old","stream":true}`)))
	waitChan(t, entered, "old streaming execution")
	w, newDone := runBindingHTTP(t, fx.f, bindingRequest("fixture", routeChatCompletions))
	in := fx.next(t)
	fx.p.admission.setDrain("ollama", true)
	if fx.p.admission.currentActive() != 1 {
		t.Fatalf("waiting replacement changed drain count: %d", fx.p.admission.currentActive())
	}
	refused := httptest.NewRecorder()
	fx.f.serveIngress(refused, bindingRequest("old", routeChatCompletions), "peer")
	assertAdmissionRejection(t, refused, nodepolicy.RejectDraining)
	select {
	case <-w.wrote:
		t.Fatal("replacement emitted output before drain")
	default:
	}
	unblock()
	waitChan(t, oldDone, "old stream drained")
	if fx.p.admission.currentActive() != 0 {
		t.Fatal("old stream did not release drain count")
	}
	if err := fx.f.setLocalBackend(localBackend{Engine: "ollama"}); err != nil {
		t.Fatal(err)
	}
	e := newBindingEngine(t, "fixture", nil)
	setBackend(t, lm, e.URL, true)
	fx.reply(t, in, nodepolicy.BoundEnginePrepared{Success: true})
	waitChan(t, newDone, "replacement after drain")
	if w.Code != http.StatusOK || e.hits.Load() != 1 {
		t.Fatalf("replacement status=%d hits=%d body=%s", w.Code, e.hits.Load(), w.Body)
	}
}

func TestEngineBindingRequestsPinnedPreparationStaysTerminalAndFresh(t *testing.T) {
	test := func(name string, removePeer bool) {
		t.Run(name, func(t *testing.T) {
			fx := newBindingRequestFixture(t)
			client, ingressURL, selfDir := pinnedBindingClient(t, fx)
			release := make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			// Register after engine creation so cleanup unblocks its handler first.
			e := newBindingEngine(t, "fixture", release)
			t.Cleanup(unblock)
			onward := newBindingEngine(t, "fixture", nil)
			fx.f.discovery.AddManual(nodeForModel(t, "onward", onward.URL, "fixture"))
			req, err := http.NewRequest(http.MethodPost, ingressURL+routeChatCompletions, strings.NewReader(`{"model":"fixture","stream":false}`))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set(nodepolicy.EnginePrepareBudgetHeader, "1000")
			type result struct {
				status int
				header http.Header
				body   []byte
				err    error
			}
			responses := make(chan result, 1)
			go func() {
				resp, err := client.Do(req)
				if err != nil {
					responses <- result{err: err}
					return
				}
				defer resp.Body.Close()
				body, err := io.ReadAll(resp.Body)
				responses <- result{status: resp.StatusCode, header: resp.Header, body: body, err: err}
			}()
			in := fx.next(t)
			if in.PeerID != "authenticated-origin" || in.PeerIncarnation == "" {
				t.Fatalf("peer evidence=%+v", in)
			}
			select {
			case r := <-responses:
				t.Fatalf("early peer output: %+v", r)
			default:
			}
			if fx.p.admission.currentActive() != 0 {
				t.Fatal("peer preparation owns ticket")
			}
			setBackend(t, fx.f, e.URL, true)
			fx.reply(t, in, nodepolicy.BoundEnginePrepared{Success: true})
			waitChan(t, e.inventoryEntered, "peer inventory barrier")
			if removePeer {
				if err := os.Remove(filepath.Join(selfDir, "trusted", "authenticated-origin.json")); err != nil {
					t.Fatal(err)
				}
			}
			unblock()
			var got result
			select {
			case got = <-responses:
			case <-time.After(3 * time.Second):
				t.Fatal("peer preparation did not finish")
			}
			if got.err != nil {
				t.Fatal(got.err)
			}
			if removePeer {
				var refusal struct{ Reason string }
				if err := json.Unmarshal(got.body, &refusal); err != nil {
					t.Fatal(err)
				}
				if got.status != http.StatusServiceUnavailable || refusal.Reason != "model-unavailable" || e.hits.Load() != 0 {
					t.Fatalf("stale peer dispatched: status=%d refusal=%+v hits=%d", got.status, refusal, e.hits.Load())
				}
			} else if got.status != http.StatusOK || e.hits.Load() != 1 || got.header.Get(nodepolicy.ModelHeader) != "fixture" || got.header.Get(nodepolicy.EngineHeader) != "ollama" || got.header.Get(nodepolicy.NodeHeader) != "self" {
				t.Fatalf("prepared peer status=%d hits=%d headers=%v", got.status, e.hits.Load(), got.header)
			}
			if onward.hits.Load() != 0 {
				t.Fatal("terminal ingress routed to another destination")
			}
		})
	}
	test("current pinned peer prepares locally", false)
	test("removed peer cannot use accepted preparation", true)
}

func TestEngineBindingRequestsCompetingTargetsAndCooldown(t *testing.T) {
	fx := newBindingRequestFixture(t)
	lm := newFacade(fx.p, lmstudioCase(t).profile, NewDiscovery(), lmstudioCase(t).profile.FacadePort)
	fx.p.facades[lm.profile.Name] = lm
	pol := fx.p.admission.currentPolicy()
	pol.EngineBindings = nodepolicy.EngineBindings{{Engine: "ollama", Model: "fixture"}, {Engine: "lmstudio", Model: "other"}}
	fx.p.admission.setPolicy(pol)
	fx.offer(t, lm, "other")
	first, firstDone := runBindingHTTP(t, fx.f, bindingRequest("fixture", "/api/generate"))
	second, secondDone := runBindingHTTP(t, fx.f, bindingRequest("other", routeChatCompletions))
	a, b := fx.next(t), fx.next(t)
	if a.Engine != "ollama" {
		a, b = b, a
	}
	if a.Engine != "ollama" || b.Engine != "lmstudio" {
		t.Fatalf("competing demands=%+v %+v", a, b)
	}
	fx.reply(t, b, nodepolicy.BoundEnginePrepared{Reason: "engine-switch-busy"})
	waitChan(t, secondDone, "competing target refusal")
	assertBindingRefusal(t, second.ResponseRecorder, "engine-switch-busy")
	e := newBindingEngine(t, "fixture", nil)
	setBackend(t, fx.f, e.URL, true)
	fx.reply(t, a, nodepolicy.BoundEnginePrepared{Success: true})
	waitChan(t, firstDone, "first target preparation")
	if first.Code != http.StatusOK || e.hits.Load() != 1 {
		t.Fatalf("first target status=%d hits=%d", first.Code, e.hits.Load())
	}
	third, thirdDone := runBindingHTTP(t, fx.f, bindingRequest("other", routeChatCompletions))
	next := fx.next(t)
	fx.reply(t, next, nodepolicy.BoundEnginePrepared{Reason: "engine-switch-cooldown"})
	waitChan(t, thirdDone, "alternating target cooldown")
	assertBindingRefusal(t, third.ResponseRecorder, "engine-switch-cooldown")
	if e.hits.Load() != 1 || fx.p.admission.currentActive() != 0 {
		t.Fatal("refused alternation replayed or retained execution")
	}
	select {
	case in := <-fx.requests:
		t.Fatalf("competing/cooldown demand retried acceptance: %+v", in)
	default:
	}
}

func TestEngineBindingRequestsMalformedBudgetCannotActivate(t *testing.T) {
	test := func(name string, values []string) {
		t.Run(name, func(t *testing.T) {
			fx := newBindingRequestFixture(t)
			req := bindingRequest("fixture", "/api/generate")
			for _, value := range values {
				req.Header.Add(nodepolicy.EnginePrepareBudgetHeader, value)
			}
			w := httptest.NewRecorder()
			fx.f.handleHTTP(w, req)
			assertBindingRefusal(t, w, "engine-switch-timeout")
			select {
			case in := <-fx.requests:
				t.Fatalf("malformed budget activated: %+v", in)
			default:
			}
		})
	}
	test("malformed", []string{"invalid"})
	test("repeated", []string{"1", "90000"})
	test("overflow", []string{"18446744073709551616"})
}

func TestEngineBindingRequestsQueuedIdentityChangeDoesNotResubmitAcceptance(t *testing.T) {
	fx := newBindingRequestFixture(t)
	pol := fx.p.admission.currentPolicy()
	pol.Admission.MaxConcurrentPerModel = 1
	fx.p.admission.setPolicy(pol)
	release := make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	e := newBindingEngine(t, "fixture", release)
	t.Cleanup(unblock)
	w, done := runBindingHTTP(t, fx.f, bindingRequest("fixture", "/api/generate"))
	in := fx.next(t)
	setBackend(t, fx.f, e.URL, true)
	fx.reply(t, in, nodepolicy.BoundEnginePrepared{Success: true, SwitchRequestID: "binding:run:1"})
	waitChan(t, e.inventoryEntered, "queued request inventory")
	first := fx.p.admission.admit(context.Background(), admissionRequest{engine: "ollama", model: "fixture", waitCap: 0})
	if first.ticket == nil {
		t.Fatalf("competing ordinary admission=%+v", first)
	}
	defer first.ticket.release()
	unblock()
	waitForCond(t, time.Second, "prepared request waiting on ordinary admission", func() bool {
		fx.p.admission.mu.Lock()
		defer fx.p.admission.mu.Unlock()
		return fx.p.admission.queued == 1
	})
	select {
	case <-w.wrote:
		t.Fatal("queued request emitted early output")
	default:
	}
	fx.p.admission.setModelIdentity(map[string]nodepolicy.EngineIdentity{"ollama": {EngineRunID: "replacement"}})
	waitChan(t, done, "queued freshness refusal")
	assertBindingRefusal(t, w.ResponseRecorder, "model-unavailable")
	if w.Header().Get(nodepolicy.SwitchRequestIDHeader) != "binding:run:1" || e.hits.Load() != 0 || fx.p.admission.currentActive() != 1 {
		t.Fatalf("stale queue response=%v hits=%d active=%d", w.Header(), e.hits.Load(), fx.p.admission.currentActive())
	}
	select {
	case in := <-fx.requests:
		t.Fatalf("accepted operation replaced by new submission: %+v", in)
	default:
	}
}

func TestEngineBindingRequestsRemovedBindingCannotBorrowStaleOffer(t *testing.T) {
	fx := newBindingRequestFixture(t)
	e := newBindingEngine(t, "fixture", nil)
	setBackend(t, fx.f, e.URL, true)
	pol := fx.p.admission.currentPolicy()
	pol.Revision, pol.EngineBindings = "2", nil
	fx.p.admission.setPolicy(pol)
	// The old offer is still in discovery, but no canonical inventory has
	// established this model on the now-running engine.
	w := httptest.NewRecorder()
	fx.f.handleHTTP(w, bindingRequest("fixture", "/api/generate"))
	assertBindingRefusal(t, w, "model-unavailable")
	if e.hits.Load() != 0 {
		t.Fatal("removed binding borrowed stale offer as dispatch authority")
	}
	select {
	case in := <-fx.requests:
		t.Fatalf("removed binding submitted activation: %+v", in)
	default:
	}
}

func coldBindingPeer(fx *proxyReverseFixture) {
	fx.node.Models, fx.node.PolicyRevision = nil, "1"
	fx.node.EngineBindingOffers = bindingOffer("ollama", "fixture-model")
	fx.p.soleFacade().discovery.SetSubscribed([]Node{fx.node})
	fx.directHealthy.Store(true)
}

func TestEngineBindingRequestsReviewColdPinnedSlotBudget(t *testing.T) {
	release := make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	var held, inference atomic.Int32
	fx := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			held.Add(1)
			select {
			case <-release:
			case <-r.Context().Done():
				return
			}
		} else {
			inference.Add(1)
		}
		_, _ = io.WriteString(w, `{"done":true}`)
	}))
	t.Cleanup(unblock)
	coldBindingPeer(fx)
	cand := fx.candidate(t)
	transport := fx.p.candidateTransport(cand)
	var occupying sync.WaitGroup
	for range 8 {
		req := proxyReverseRequest(t, http.MethodGet, cand.url.String()+"/api/tags", nil)
		occupying.Add(1)
		go func() {
			defer occupying.Done()
			resp, err := transport.RoundTrip(req)
			if err != nil {
				return
			}
			defer resp.Body.Close()
			_, _ = io.Copy(io.Discard, resp.Body)
		}()
	}
	t.Cleanup(occupying.Wait)
	// Unblock before joining requests and before the fixture closes servers.
	t.Cleanup(unblock)
	waitForCond(t, 3*time.Second, "all pinned peer slots occupied", func() bool { return held.Load() == 8 })
	ctx, cancel := context.WithTimeout(context.Background(), 600*time.Millisecond)
	defer cancel()
	req := bindingRequest("fixture-model", "/api/generate").WithContext(ctx)
	req.Header.Set(nodepolicy.EnginePrepareBudgetHeader, "80")
	started := time.Now()
	w, done := runBindingHTTP(t, fx.p.soleFacade(), req)
	waitChan(t, done, "cold pinned slot refusal")
	if elapsed := time.Since(started); elapsed < 60*time.Millisecond || elapsed > 350*time.Millisecond {
		t.Errorf("cold slot refusal outside preparation bound: %v", elapsed)
	} else {
		t.Logf("cold slot preparation refusal after %v", elapsed)
	}
	assertBindingRefusal(t, w.ResponseRecorder, "engine-switch-timeout")
	if inference.Load() != 0 {
		t.Fatal("expired cold request reached peer engine")
	}
}

func TestEngineBindingRequestsReviewColdPinnedHeaderBudgetAndStream(t *testing.T) {
	streamEntered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	var calls atomic.Int32
	var streamCancelled atomic.Bool
	fx := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Error(err)
			return
		}
		if calls.Add(1) == 1 {
			select {
			case <-r.Context().Done():
			case <-release:
			}
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = io.WriteString(w, "{\"response\":\"first\",\"done\":false}\n")
		if err := http.NewResponseController(w).Flush(); err != nil {
			t.Error(err)
			return
		}
		close(streamEntered)
		select {
		case <-release:
		case <-r.Context().Done():
			streamCancelled.Store(true)
			return
		}
		_, _ = io.WriteString(w, "{\"response\":\"tail\",\"done\":true}\n")
	}))
	t.Cleanup(unblock)
	coldBindingPeer(fx)
	ctx, cancel := context.WithTimeout(context.Background(), 600*time.Millisecond)
	defer cancel()
	req := bindingRequest("fixture-model", "/api/generate").WithContext(ctx)
	req.Header.Set(nodepolicy.EnginePrepareBudgetHeader, "80")
	started := time.Now()
	w, done := runBindingHTTP(t, fx.p.soleFacade(), req)
	waitChan(t, done, "cold peer header refusal")
	if elapsed := time.Since(started); elapsed < 60*time.Millisecond || elapsed > 350*time.Millisecond {
		t.Errorf("cold header refusal outside preparation bound: %v", elapsed)
	} else {
		t.Logf("cold header preparation refusal after %v", elapsed)
	}
	assertBindingRefusal(t, w.ResponseRecorder, "engine-switch-timeout")
	// A later cold request that produces its headers in time owns its healthy
	// stream beyond that preparation bound. The header timer must be retired.
	streamReq := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"fixture-model","stream":true}`))
	streamReq.Header.Set(nodepolicy.EnginePrepareBudgetHeader, "150")
	stream, streamDone := runBindingHTTP(t, fx.p.soleFacade(), streamReq)
	waitChan(t, streamEntered, "accepted stream")
	waitChan(t, stream.wrote, "origin stream output")
	timer := time.NewTimer(250 * time.Millisecond)
	defer timer.Stop()
	<-timer.C
	select {
	case <-streamDone:
		t.Fatal("preparation deadline killed accepted stream")
	default:
	}
	unblock()
	waitChan(t, streamDone, "stream after preparation deadline")
	if stream.Code != http.StatusOK || streamCancelled.Load() {
		t.Fatalf("accepted stream status=%d cancelled=%v", stream.Code, streamCancelled.Load())
	}
	var last struct {
		Response string
		Done     bool
	}
	decoder := json.NewDecoder(strings.NewReader(stream.Body.String()))
	for decoder.More() {
		if err := decoder.Decode(&last); err != nil {
			t.Fatal(err)
		}
	}
	if !last.Done || last.Response != "tail" {
		t.Fatalf("stream tail=%+v body=%s", last, stream.Body)
	}
}

type bindingBodyReadBarrier struct {
	io.ReadCloser
	entered chan struct{}
	release <-chan struct{}
	once    sync.Once
}

func (b *bindingBodyReadBarrier) Read(p []byte) (int, error) {
	b.once.Do(func() { close(b.entered) })
	<-b.release
	return b.ReadCloser.Read(p)
}

func TestEngineBindingRequestsReviewPinnedRelationshipBeforeBodyWait(t *testing.T) {
	fx := newBindingRequestFixture(t)
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	client, ingressURL, selfDir := pinnedBindingClient(t, fx, func(r *http.Request) {
		r.Body = &bindingBodyReadBarrier{ReadCloser: r.Body, entered: entered, release: release}
	})
	t.Cleanup(unblock)
	old, ok := fx.p.mesh.PeerServiceIncarnation("authenticated-origin")
	if !ok {
		t.Fatal("missing initial relationship")
	}
	req, err := http.NewRequest(http.MethodPost, ingressURL+"/api/generate", strings.NewReader(`{"model":"fixture","stream":false}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("X-PAIR-Peer-Id", "authenticated-origin")
	req.Header.Set("X-PAIR-Peer-Incarnation", old)
	type result struct {
		status int
		body   []byte
		err    error
	}
	done := make(chan result, 1)
	go func() {
		resp, err := client.Do(req)
		if err != nil {
			done <- result{err: err}
			return
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		done <- result{status: resp.StatusCode, body: body, err: err}
	}()
	waitChan(t, entered, "authenticated body read")
	replacement := t.TempDir()
	clustertrusttest.Join(t, replacement, "fixture-cluster", "authenticated-origin")
	proxyReversePin(t, selfDir, replacement, "authenticated-origin")
	fx.p.mesh.Refresh()
	fresh, ok := fx.p.mesh.PeerServiceIncarnation("authenticated-origin")
	if !ok || fresh == old {
		t.Fatal("replacement did not change relationship incarnation")
	}
	unblock()
	var got result
	select {
	case in := <-fx.requests:
		fx.reply(t, in, nodepolicy.BoundEnginePrepared{Reason: "management-denied"})
		got = <-done
		t.Errorf("old authenticated body rebound to new relationship: request=%+v", in)
	case got = <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("relationship refusal did not finish")
	}
	if got.err != nil {
		t.Fatal(got.err)
	}
	var refusal struct{ Reason string }
	if err := json.Unmarshal(got.body, &refusal); err != nil {
		t.Fatal(err)
	}
	if got.status != http.StatusServiceUnavailable || refusal.Reason != "management-denied" {
		t.Fatalf("old relationship response=%d %+v", got.status, refusal)
	}
}

func TestEngineBindingRequestsReviewBoundReadyPeerCannotLegacyWake(t *testing.T) {
	fx := newBindingRequestFixture(t)
	e := newBindingEngine(t, "fixture", nil)
	setBackend(t, fx.f, e.URL, true)
	fx.p.noteSelf("self", "ollama", []string{"fixture"})
	pol := fx.p.admission.currentPolicy()
	pol.Admission.MaxConcurrentPerModel = 1
	pol.Idle.StartOnDemand = true
	pol.Idle.WakeTimeoutSeconds = 1
	fx.p.admission.setPolicy(pol)
	fx.p.admission.setIntent(map[string]bool{"ollama": true})
	first := fx.p.admission.admit(context.Background(), admissionRequest{engine: "ollama", model: "fixture", waitCap: 0})
	if first.ticket == nil {
		t.Fatalf("first admission=%+v", first)
	}
	defer first.ticket.release()
	client, ingressURL, _ := pinnedBindingClient(t, fx)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ingressURL+"/api/generate", strings.NewReader(`{"model":"fixture","stream":false}`))
	if err != nil {
		t.Fatal(err)
	}
	type result struct {
		status int
		body   []byte
		err    error
	}
	done := make(chan result, 1)
	go func() {
		resp, err := client.Do(req)
		if err != nil {
			done <- result{err: err}
			return
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		done <- result{status: resp.StatusCode, body: body, err: err}
	}()
	waitForCond(t, time.Second, "ready bound peer queued", func() bool {
		fx.p.admission.mu.Lock()
		defer fx.p.admission.mu.Unlock()
		return fx.p.admission.queued == 1
	})
	b := fx.f.currentLocalBackend()
	b.Healthy = false
	if err := fx.f.setLocalBackend(b); err != nil {
		t.Fatal(err)
	}
	select {
	case wake := <-fx.wakes:
		cancel()
		<-done
		t.Fatalf("inference-only bound peer issued legacy wake: %+v", wake)
	case in := <-fx.requests:
		fx.reply(t, in, nodepolicy.BoundEnginePrepared{Reason: "management-denied"})
		<-done
		t.Fatalf("ready-check race issued unexpected activation: %+v", in)
	case got := <-done:
		if got.err != nil {
			t.Fatal(got.err)
		}
		var refusal struct{ Reason string }
		if err := json.Unmarshal(got.body, &refusal); err != nil {
			t.Fatal(err)
		}
		if got.status != http.StatusServiceUnavailable || refusal.Reason != "model-unavailable" {
			t.Fatalf("health withdrawal response=%d %+v", got.status, refusal)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("health withdrawal left admission waiting")
	}
	if e.hits.Load() != 0 {
		t.Fatal("withdrawn engine received inference")
	}
}

func TestEngineBindingRequestsReviewManualEngineStripsPrivateHeaders(t *testing.T) {
	test := func(name, method, path string) {
		t.Run(name, func(t *testing.T) {
			fx := newBindingRequestFixture(t)
			fx.f.discovery.RemoveManual("self")
			e := newBindingEngine(t, "fixture", nil)
			fx.f.discovery.AddManual(nodeForModel(t, "manual", e.URL, "fixture"))
			req := bindingRequest("fixture", path)
			if method == http.MethodGet {
				req = httptest.NewRequest(method, path, nil)
			}
			for _, name := range []string{nodepolicy.EnginePrepareBudgetHeader, nodepolicy.AdmissionWaitHeader, peerStreamFramingHeader, "X-PAIR-Activation-Identity", "X-PAIR-Peer-Id", nodepolicy.SwitchRequestIDHeader} {
				req.Header.Set(name, "spoofed-private")
			}
			req.Header.Set("X-Ordinary-Option", "keep")
			w := httptest.NewRecorder()
			fx.f.handleHTTP(w, req)
			e.mu.Lock()
			defer e.mu.Unlock()
			if w.Code != http.StatusOK || e.header == nil {
				t.Fatalf("manual status=%d header=%v", w.Code, e.header)
			}
			for key := range e.header {
				if strings.HasPrefix(strings.ToLower(key), "x-pair-") {
					t.Errorf("private header reached direct manual engine: %s", key)
				}
			}
			if e.header.Get("X-Ordinary-Option") != "keep" {
				t.Fatal("ordinary option stripped")
			}
		})
	}
	test("inference", http.MethodPost, "/api/generate")
	test("control", http.MethodGet, "/api/version")
	test("model inventory", http.MethodGet, "/api/tags")
}
