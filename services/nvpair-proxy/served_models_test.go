// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func TestServedModelsConcrete(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		test := func(name string, served nodepolicy.ServedModels, allowed bool) {
			t.Run(name, func(t *testing.T) {
				for _, path := range []string{"self", "ingress"} {
					t.Run(path, func(t *testing.T) {
						e := newRecordingEngine(t, http.StatusOK)
						d := NewDiscovery()
						d.AddManual(selfNode("self", tc.profile.FacadePort, tc.advertisedModel))
						p := testProxy(tc.profile, d, tc.profile.FacadePort)
						setBackend(t, p.soleFacade(), e.URL, true)
						pol := p.admission.currentPolicy()
						pol.ServedModels = served
						p.admission.setPolicy(pol)
						w := httptest.NewRecorder()
						serveStrictHTTP(p, path, w, tc.inferenceRequest())
						if allowed {
							if w.Code != http.StatusOK || e.hits() != 1 {
								t.Fatalf("status=%d hits=%d", w.Code, e.hits())
							}
						} else {
							assertAdmissionRejection(t, w, rejectNotServed)
							if e.hits() != 0 {
								t.Fatal("denied request reached engine")
							}
						}
					})
				}
			})
		}
		test("absent", nil, true)
		test("empty", nodepolicy.ServedModels{}, false)
		test("subset included", nodepolicy.ServedModels{{Engine: tc.profile.Name, Model: tc.advertisedModel}}, true)
		test("subset excluded", nodepolicy.ServedModels{{Engine: tc.profile.Name, Model: "other"}}, false)
		test("wrong engine", nodepolicy.ServedModels{{Engine: "other", Model: tc.advertisedModel}}, false)
		test("case-variant model", nodepolicy.ServedModels{{Engine: tc.profile.Name, Model: strings.ToUpper(tc.requestedModel)}}, false)
	})
}

func TestServedModelsStrict(t *testing.T) {
	test := func(name string, served nodepolicy.ServedModels, allowed bool) {
		t.Run(name, func(t *testing.T) {
			h := newStrictHarness(t)
			p := h.a.currentPolicy()
			p.ServedModels = served
			h.a.setPolicy(p)
			ticket, err := h.a.admitStrictNow(context.Background(), "ollama", "m", strictRequest())
			if !allowed {
				if ticket != nil || !errors.Is(err, errStrictUnavailable) {
					t.Fatalf("ticket=%v err=%v", ticket, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			defer ticket.Finish(strictQualifiedComplete)
			if err := ticket.BeforeStrictDispatch(); err != nil {
				t.Fatal(err)
			}
		})
	}
	test("absent", nil, true)
	test("empty", nodepolicy.ServedModels{}, false)
	test("normalized subset", nodepolicy.ServedModels{{Engine: "ollama", Model: "m:latest"}}, true)
	test("excluded subset", nodepolicy.ServedModels{{Engine: "ollama", Model: "other"}}, false)
	test("case-variant model", nodepolicy.ServedModels{{Engine: "ollama", Model: "M"}}, false)
	test("case-variant tag", nodepolicy.ServedModels{{Engine: "ollama", Model: "m:LATEST"}}, false)
}

// TestServedModelsCaseSensitivity pins the permission comparison itself:
// after each side is normalized by its engine's naming rule, the engine's
// spelling decides permission, so a case-folded comparison would serve "M"
// for a policy that serves only "m".
func TestServedModelsCaseSensitivity(t *testing.T) {
	test := func(name, engine, servedModel, requestModel string, want bool) {
		t.Run(name, func(t *testing.T) {
			served := nodepolicy.ServedModels{{Engine: engine, Model: servedModel}}
			if got := modelServed(served, engine, requestModel, normalizeEngineModel); got != want {
				t.Fatalf("modelServed(%q, served %q, request %q) = %v, want %v",
					engine, servedModel, requestModel, got, want)
			}
		})
	}
	test("ollama model case", "ollama", "m", "M", false)
	test("ollama tag case", "ollama", "m:latest", "m:LATEST", false)
	test("lmstudio model case", "lmstudio", "qwen3-8b", "QWEN3-8B", false)
	test("ollama exact", "ollama", "m:latest", "m:latest", true)
	test("ollama untagged request", "ollama", "m", "m:latest", true)
	test("lmstudio exact", "lmstudio", "qwen3-8b", "qwen3-8b", true)
}

// assertAmbiguousModelRejection checks the structured client error a body
// with two top-level "model" members gets.
func assertAmbiguousModelRejection(t *testing.T, rec *httptest.ResponseRecorder) {
	t.Helper()
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", rec.Code, rec.Body)
	}
	var body struct {
		Error string `json:"error"`
		Code  string `json:"code"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode rejection body %q: %v", rec.Body, err)
	}
	if body.Code != "ambiguous-model" {
		t.Fatalf("code = %q, want ambiguous-model", body.Code)
	}
}

// A body naming "model" twice is ambiguous: the proxy decodes the last member
// for admission while the tier rewrite replaces the first, so the model
// admitted ("m") and the model an engine decodes from the forwarded body
// ("weak") could disagree and a model outside the served set could run. The
// ambiguous body is refused before admission; the ordinary body still works.
func TestServedModelsTierDuplicateModel(t *testing.T) {
	tc := ollamaCase(t)
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{
		"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "m"}},
	})
	pol.ServedModels = nodepolicy.ServedModels{{Engine: "ollama", Model: "m"}}
	if err := pol.Validate(); err != nil {
		t.Fatal(err)
	}
	e := newRecordingEngine(t, http.StatusOK)
	d := NewDiscovery()
	d.AddManual(selfNode("self", tc.profile.FacadePort, "m", "weak"))
	p := testProxy(tc.profile, d, tc.profile.FacadePort)
	setBackend(t, p.soleFacade(), e.URL, true)
	p.admission.setPolicy(pol)
	f := p.soleFacade()
	post := func(body string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		f.handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body)))
		return rec
	}

	// Control: the ordinary tier body rewrites to the served concrete model
	// and reaches the engine.
	rec := post(`{"model":"weak"}`)
	if rec.Code != http.StatusOK || e.hits() != 1 || e.lastBody() != `{"model":"m"}` {
		t.Fatalf("control: status=%d hits=%d body=%s", rec.Code, e.hits(), e.lastBody())
	}

	// The ambiguous body is refused on both local inference entry points,
	// before admission, and nothing from it reaches the engine.
	for _, body := range []string{`{"model":"unused","model":"weak"}`, `{"model":"unused","Model":"weak"}`} {
		rec = post(body)
		assertAmbiguousModelRejection(t, rec)
		if e.hits() != 1 {
			t.Fatalf("duplicate-model body %s reached the engine: hits=%d", body, e.hits())
		}
	}
	ingressRec := httptest.NewRecorder()
	f.serveIngress(ingressRec, httptest.NewRequest(http.MethodPost, "/api/chat",
		strings.NewReader(`{"model":"unused","model":"weak"}`)), "peer")
	assertAmbiguousModelRejection(t, ingressRec)
	if e.hits() != 1 {
		t.Fatalf("duplicate-model ingress body reached the engine: hits=%d", e.hits())
	}
}

func TestServedModelsStrictRoutes(t *testing.T) {
	for _, path := range []string{"self", "ingress"} {
		t.Run(path, func(t *testing.T) {
			p, e := strictHTTPFixture(t)
			pol := p.admission.currentPolicy()
			pol.ServedModels = nodepolicy.ServedModels{}
			p.admission.setPolicy(pol)
			w := httptest.NewRecorder()
			serveStrictHTTP(p, path, w, strictHTTPRequest(t))
			if w.Code != http.StatusServiceUnavailable || e.hits() != 0 || p.admission.currentActive() != 0 {
				t.Fatalf("status=%d hits=%d active=%d", w.Code, e.hits(), p.admission.currentActive())
			}
		})
	}
}

func TestServedModelsRejectBeforeWake(t *testing.T) {
	h := newAdmHarness(t)
	h.setHealthy("ollama", false)
	h.a.setIntent(map[string]bool{"ollama": true})
	pol := h.a.currentPolicy()
	pol.Idle.StartOnDemand = true
	pol.ServedModels = nodepolicy.ServedModels{{Engine: "ollama", Model: "m"}}
	h.a.setPolicy(pol)
	for _, model := range []string{"other", ""} {
		d := h.a.admit(context.Background(), admissionRequest{engine: "ollama", model: model})
		if d.reject != rejectNotServed || d.ticket != nil {
			t.Fatalf("model=%q decision=%+v", model, d)
		}
	}
	if len(h.notes("admission/wake")) != 0 || len(h.notes("admission/unload")) != 0 {
		t.Fatal("denied request initiated engine work")
	}
}

func TestServedModelsPeerOverlay(t *testing.T) {
	tc := ollamaCase(t)
	test := func(name string, served nodepolicy.ServedModels, allowed bool) {
		t.Run(name, func(t *testing.T) {
			n, ok := subscribedToNode(tc.profile, noderec.DirectoryNode{
				HostUUID: "peer", IP: "192.0.2.1", Models: []string{"m"},
				Services: map[noderec.ServiceKey]noderec.ServiceStatus{tc.profile.DiscoveryService: {Port: tc.profile.FacadePort}},
				Policy:   &nodepolicy.Advertisement{ServedModels: served},
			})
			if !ok {
				t.Fatal("overlay projection failed")
			}
			d := NewDiscovery()
			d.AddManual(n)
			p := testProxy(tc.profile, d, tc.profile.FacadePort)
			got := p.soleFacade().resolveCandidates("m")
			if (len(got) == 1) != allowed {
				t.Fatalf("candidates=%+v allowed=%v", got, allowed)
			}
			if len(n.Models) != 1 {
				t.Fatal("inventory was filtered")
			}
		})
	}
	test("absent", nil, true)
	test("empty", nodepolicy.ServedModels{}, false)
	test("normalized subset", nodepolicy.ServedModels{{Engine: "ollama", Model: "m:latest"}}, true)
	test("excluded", nodepolicy.ServedModels{{Engine: "ollama", Model: "other"}}, false)
}

func TestServedModelsDormantSelf(t *testing.T) {
	test := func(name string, served nodepolicy.ServedModels, allowed bool) {
		t.Run(name, func(t *testing.T) {
			tc := ollamaCase(t)
			p := testProxy(tc.profile, NewDiscovery(), tc.profile.FacadePort)
			p.noteSelf("self", "ollama", []string{"m"})
			p.admission.setIntent(map[string]bool{"ollama": true})
			pol := p.admission.currentPolicy()
			pol.Idle.StartOnDemand = true
			pol.ServedModels = served
			p.admission.setPolicy(pol)
			_, ok := p.soleFacade().dormantSelfCandidate("m")
			if ok != allowed {
				t.Fatalf("dormant=%v want=%v", ok, allowed)
			}
		})
	}
	test("absent", nil, true)
	test("empty", nodepolicy.ServedModels{}, false)
	test("normalized subset", nodepolicy.ServedModels{{Engine: "ollama", Model: "m:latest"}}, true)
	test("excluded", nodepolicy.ServedModels{{Engine: "ollama", Model: "other"}}, false)
}

func TestServedModelsRolesRetainedButIneligible(t *testing.T) {
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{
		"weak":   {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "router"}},
		"medium": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "router"}},
	})
	pol.TierPolicy.AllowStronger = false
	fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"a": {"own", "router"}}, nil)
	fx.declare(t, "a", map[string]nodepolicy.AdvertisedRole{"weak": ollamaRole("own")})
	n, ok := discoveredNode(fx.p.soleFacade(), "a")
	if !ok {
		t.Fatal("missing peer")
	}
	if got := resolvedTier(fx.p.soleFacade(), "weak", requestNeeds{}, false); len(got) != 1 {
		t.Fatalf("unrestricted declared role unavailable: %+v", got)
	}
	n.ServedModels = nodepolicy.ServedModels{{Engine: "ollama", Model: "router"}}
	fx.p.soleFacade().discovery.AddManual(n)
	if got := resolvedTier(fx.p.soleFacade(), "weak", requestNeeds{}, false); len(got) != 0 {
		t.Fatalf("unserved role fell back: %+v", got)
	}
	if got := resolvedTier(fx.p.soleFacade(), "medium", requestNeeds{}, false); len(got) != 0 {
		t.Fatalf("omitted role fell back: %+v", got)
	}
	if len(fx.p.soleFacade().declaredRoles(pol).peers["a"]) != 1 {
		t.Fatal("role declaration removed")
	}
	w := fx.do(t, routeChatCompletions, `{"model":"weak"}`)
	if w.Code == http.StatusOK || fx.eng["a"].hits() != 0 {
		t.Fatalf("status=%d hits=%d", w.Code, fx.eng["a"].hits())
	}
}

func TestServedModelsActivationBeforeForwarding(t *testing.T) {
	for _, path := range []string{"self", "ingress"} {
		t.Run(path, func(t *testing.T) {
			tc := ollamaCase(t)
			e := newRecordingEngine(t, http.StatusOK)
			d := NewDiscovery()
			d.AddManual(selfNode("self", tc.profile.FacadePort, tc.advertisedModel))
			p := testProxy(tc.profile, d, tc.profile.FacadePort)
			setBackend(t, p.soleFacade(), e.URL, true)
			activated := make(chan error, 1)
			p.plainHTTPTransport().DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
				pol := p.admission.currentPolicy()
				pol.Revision = "1"
				pol.ServedModels = nodepolicy.ServedModels{}
				activated <- p.admission.activatePolicy(pol)
				return (&net.Dialer{}).DialContext(ctx, network, address)
			}
			w := httptest.NewRecorder()
			serveStrictHTTP(p, path, w, tc.inferenceRequest())
			select {
			case err := <-activated:
				if err != nil {
					t.Fatal(err)
				}
			default:
				t.Fatal("activation hook not reached")
			}
			if w.Code != http.StatusOK || e.hits() != 1 {
				t.Fatalf("admitted work status=%d hits=%d", w.Code, e.hits())
			}
			w = httptest.NewRecorder()
			serveStrictHTTP(p, path, w, tc.inferenceRequest())
			assertAdmissionRejection(t, w, rejectNotServed)
			if e.hits() != 1 {
				t.Fatal("new work forwarded after restriction")
			}
		})
	}
}

func TestServedModelsStrictActivationAfterAdmission(t *testing.T) {
	h := newStrictHarness(t)
	ticket, err := h.a.admitStrictNow(context.Background(), "ollama", "m", strictRequest())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket.Finish(strictQualifiedComplete)
	pol := h.a.currentPolicy()
	pol.Revision = "1"
	pol.ServedModels = nodepolicy.ServedModels{}
	if err := h.a.activatePolicy(pol); err != nil {
		t.Fatal(err)
	}
	if err := ticket.BeforeStrictDispatch(); err != nil {
		t.Fatalf("admitted work revoked: %v", err)
	}
	if next, err := h.a.admitStrictNow(context.Background(), "ollama", "m", strictRequest()); next != nil || !errors.Is(err, errStrictUnavailable) {
		t.Fatalf("new ticket=%v err=%v", next, err)
	}
}

// A restrictive activation recompiles what this node advertises: a served-excluded
// model leaves the outward strict capability, while the internal installation
// and the already-admitted ticket stay intact (admission is the linearization
// point, and later restriction does not revoke admitted work).
func TestServedModelsStrictReadinessAfterDenyAll(t *testing.T) {
	h := newStrictHarness(t)
	before := h.a.StrictReadiness()
	if before.Capability == nil || len(before.Capability.Models) != 1 {
		t.Fatalf("installed capability missing: %+v", before)
	}
	ticket, err := h.a.admitStrictNow(context.Background(), "ollama", "m", strictRequest())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket.Finish(strictQualifiedComplete)
	pol := h.a.currentPolicy()
	pol.Revision = "1"
	pol.ServedModels = nodepolicy.ServedModels{}
	if err := h.a.activatePolicy(pol); err != nil {
		t.Fatal(err)
	}
	if report := h.a.StrictReadiness(); report.Capability != nil {
		t.Fatalf("deny-all still advertises strict model capability: %+v", report)
	}
	if err := ticket.BeforeStrictDispatch(); err != nil {
		t.Fatalf("admitted ticket revoked by serving-only change: %v", err)
	}
	if next, err := h.a.admitStrictNow(context.Background(), "ollama", "m", strictRequest()); next != nil || !errors.Is(err, errStrictUnavailable) {
		t.Fatalf("new ticket=%v err=%v", next, err)
	}
}

// Both strict-ticket constructors snapshot the admission-time served set, so a
// cold ticket fences the same permission a warm one does.
func TestServedModelsStrictColdTicketCarriesServedSnapshot(t *testing.T) {
	h := newStrictHarness(t)
	pol := h.a.currentPolicy()
	pol.ServedModels = nodepolicy.ServedModels{{Engine: "ollama", Model: "m"}}
	h.a.setPolicy(pol)
	// Installed but not resident: admission takes the cold constructor.
	h.a.setResidency(map[string][]string{})
	ticket, err := h.a.admitStrictNow(context.Background(), "ollama", "m", strictRequest())
	if err != nil {
		t.Fatalf("cold admission: %v", err)
	}
	defer ticket.Finish(strictNotDispatched)
	st, ok := ticket.(*strictTicket)
	if !ok {
		t.Fatalf("ticket type %T, want *strictTicket", ticket)
	}
	if st.warmOnly {
		t.Fatal("expected a cold ticket, got a warm one")
	}
	if len(st.servedModels) != 1 || st.servedModels[0].Engine != "ollama" || st.servedModels[0].Model != "m" {
		t.Fatalf("cold ticket servedModels = %+v, want the admission snapshot", st.servedModels)
	}
}
