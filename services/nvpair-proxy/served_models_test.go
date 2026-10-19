// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
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
