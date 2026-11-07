// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"nvpair-shared/nodepolicy"
)

func TestProfileRoutingAuditServedNativeModel(t *testing.T) {
	t.Run("dormant self", func(t *testing.T) {
		p := testProxy(llamacppCase(t).profile, NewDiscovery(), 11435)
		p.noteSelf("self", "llamacpp", []string{"native-file"})
		p.admission.setIntent(map[string]bool{"llamacpp": true})
		pol := nodepolicy.Default()
		pol.Idle.StartOnDemand = true
		pol.Profiles = []nodepolicy.Profile{{Key: "public-profile", DisplayName: "Public", Engine: "llamacpp", Model: "native-file"}}
		pol.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: "native-file"}}
		p.admission.setPolicy(pol)
		if _, ok := p.soleFacade().dormantSelfCandidate("public-profile"); !ok {
			t.Fatal("native serving permission excluded dormant profile")
		}
	})
	for _, destination := range []string{"self", "peer", "peer ingress"} {
		t.Run(destination, func(t *testing.T) {
			pol := nodepolicy.Default()
			pol.Profiles = []nodepolicy.Profile{{Key: "public-profile", DisplayName: "Public", Engine: "llamacpp", Model: "native-file"}}
			pol.Roles = []nodepolicy.Role{{Name: "coder", Profiles: []string{"public-profile"}}}
			pol.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: "native-file"}}
			fx := newTierFixture(t, llamacppCase(t), pol, map[string][]string{"peer": {"native-file"}}, nil)
			f := fx.p.soleFacade()
			e := fx.eng["peer"]
			if destination == "peer" {
				n := f.discovery.Nodes()[0]
				n.Profiles = []nodepolicy.AdvertisedProfile{{Key: "public-profile", Engine: "llamacpp", Model: "native-file"}}
				n.Roles = pol.Roles
				n.ServedModels = pol.ServedModels
				f.discovery.AddManual(n)
				pol.Profiles = nil
				pol.Roles = nil
				fx.p.admission.setPolicy(pol)
			} else {
				f.discovery.RemoveManual("peer")
				f.discovery.AddManual(selfNode("self", f.selfPort(), "native-file"))
				setBackend(t, f, e.URL, true)
			}
			var w *httptest.ResponseRecorder
			if destination == "peer ingress" {
				w = httptest.NewRecorder()
				f.serveIngress(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"public-profile"}`)), "peer")
			} else {
				w = fx.do(t, routeChatCompletions, `{"model":"coder"}`)
			}
			if w.Code != http.StatusOK || e.hits() != 1 {
				t.Fatalf("native serving permission lost: status=%d hits=%d body=%s", w.Code, e.hits(), w.Body)
			}
		})
	}
}

func TestProfileRoutingAuditCrossEngineCatalogPermissions(t *testing.T) {
	for _, scenario := range []struct {
		name    string
		status  int
		want    int
		ingress bool
	}{
		{name: "both engines approve", status: http.StatusOK, want: http.StatusOK},
		{name: "other engine denies", status: http.StatusForbidden, want: http.StatusForbidden},
		{name: "peer ingress both engines approve", status: http.StatusOK, want: http.StatusOK, ingress: true},
		{name: "peer ingress other engine denies", status: http.StatusForbidden, want: http.StatusForbidden, ingress: true},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			var otherChecks atomic.Int32
			allowed := corsEngine(t, llamacppCase(t), "http://app.test", http.StatusOK)
			defer allowed.Close()
			other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				otherChecks.Add(1)
				if scenario.status == http.StatusOK {
					w.Header().Set("Access-Control-Allow-Origin", "http://app.test")
				}
				w.WriteHeader(scenario.status)
				if _, err := w.Write([]byte(`{"data":[{"id":"private-native"}]}`)); err != nil {
					t.Error(err)
				}
			}))
			defer other.Close()
			p := testProxy(llamacppCase(t).profile, NewDiscovery(), 11435)
			f := p.soleFacade()
			g := newFacade(p, customProfile(t), NewDiscovery(), 11436)
			p.facades[g.profile.Name] = g
			setBackend(t, f, allowed.URL, true)
			setBackend(t, g, other.URL, true)
			f.discovery.AddManual(selfNode("self", f.selfPort(), "native-file"))
			g.discovery.AddManual(selfNode("self", g.selfPort(), "private-native"))
			pol := nodepolicy.Default()
			pol.Profiles = []nodepolicy.Profile{
				{Key: "entry-profile", DisplayName: "Entry", Engine: f.profile.Name, Model: "native-file"},
				{Key: "other-profile", DisplayName: "Other", Engine: g.profile.Name, Model: "private-native"},
			}
			p.admission.setPolicy(pol)
			w := httptest.NewRecorder()
			r := corsRequest(http.MethodGet, "/v1/models", "http://app.test")
			if scenario.ingress {
				f.serveIngress(w, r, "peer")
			} else {
				f.handleHTTP(w, r)
			}
			if w.Code != scenario.want || otherChecks.Load() != 1 {
				t.Fatalf("cross-engine permission status=%d want=%d other checks=%d body=%s", w.Code, scenario.want, otherChecks.Load(), w.Body)
			}
			if scenario.want == http.StatusOK {
				assertProfileCatalog(t, w, "entry-profile", "other-profile")
			} else if w.Header().Get("Access-Control-Allow-Origin") != "" {
				t.Fatal("denied engine's catalog granted browser access")
			}
		})
	}
}

func TestProfileRoutingPeerCatalogPermissionStaysLocal(t *testing.T) {
	var engineReads, peerReads atomic.Int32
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		engineReads.Add(1)
		requireLocalKey(t, r.Header)
		w.Header().Set("Access-Control-Allow-Origin", "http://app.test")
		if _, err := w.Write([]byte(`{"data":[{"id":"native-file"}]}`)); err != nil {
			t.Error(err)
		}
	}))
	defer engine.Close()
	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		peerReads.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer peer.Close()
	p := testProxy(llamacppCase(t).profile, NewDiscovery(), 11435)
	f := p.soleFacade()
	setKeyedBackend(t, f, engine.URL, proxyTestKey)
	f.discovery.AddManual(nodeFor(t, "other-peer", peer.URL))
	pol := nodepolicy.Default()
	pol.Profiles = []nodepolicy.Profile{{Key: "public-profile", DisplayName: "Public", Engine: "llamacpp", Model: "native-file"}}
	p.admission.setPolicy(pol)
	w := httptest.NewRecorder()
	f.serveIngress(w, withCallerCredentials(corsRequest(http.MethodGet, "/v1/models", "http://app.test")), "peer")
	assertProfileCatalog(t, w, "public-profile")
	if engineReads.Load() != 1 || peerReads.Load() != 0 || w.Header().Get("Access-Control-Allow-Origin") != "http://app.test" {
		t.Fatalf("terminal catalog permission: engine=%d peer=%d headers=%v", engineReads.Load(), peerReads.Load(), w.Header())
	}
}

func TestProfileRoutingAuditNativeCompletion(t *testing.T) {
	for _, scenario := range []struct {
		name    string
		model   string
		ingress bool
		want    int
	}{
		{name: "local native refused", model: "native-file", want: http.StatusNotFound},
		{name: "peer ingress native refused", model: "native-file", ingress: true, want: http.StatusNotFound},
		{name: "local profile", model: "public-profile", want: http.StatusOK},
		{name: "local role", model: "coder", want: http.StatusOK},
		{name: "peer ingress profile", model: "public-profile", ingress: true, want: http.StatusOK},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			pol := nodepolicy.Default()
			pol.Profiles = []nodepolicy.Profile{{Key: "public-profile", DisplayName: "Public", Engine: "llamacpp", Model: "native-file"}}
			pol.Roles = []nodepolicy.Role{{Name: "coder", Profiles: []string{"public-profile"}}}
			fx := newTierFixture(t, llamacppCase(t), pol, nil, nil)
			f := fx.p.soleFacade()
			e := newRecordingEngineForRequest(t, http.StatusOK, http.MethodPost, "/completion")
			setBackend(t, f, e.URL, true)
			f.discovery.AddManual(selfNode("self", f.selfPort(), "native-file"))
			r := httptest.NewRequest(http.MethodPost, "/completion", strings.NewReader(`{"model":"`+scenario.model+`","prompt":"test"}`))
			w := httptest.NewRecorder()
			if scenario.ingress {
				f.serveIngress(w, r, "peer")
			} else {
				f.handleHTTP(w, r)
			}
			if w.Code != scenario.want {
				t.Fatalf("completion status=%d want=%d body=%s", w.Code, scenario.want, w.Body)
			}
			if scenario.want == http.StatusNotFound {
				var refusal struct {
					Error string `json:"error"`
				}
				if err := json.Unmarshal(w.Body.Bytes(), &refusal); err != nil {
					t.Fatal(err)
				}
				if refusal.Error != "no profile or role named native-file" || e.hits() != 0 {
					t.Fatalf("unprofiled completion: refusal=%+v hits=%d", refusal, e.hits())
				}
			} else if e.hits() != 1 {
				t.Fatalf("profiled completion hits=%d", e.hits())
			}
		})
	}
}

func TestProfileRoutingAuditPeerCatalog(t *testing.T) {
	pol := nodepolicy.Default()
	pol.Profiles = []nodepolicy.Profile{{Key: "public-profile", DisplayName: "Public", Engine: "llamacpp", Model: "native-file"}}
	pol.Roles = []nodepolicy.Role{{Name: "coder", Profiles: []string{"public-profile"}}}
	fx := newTierFixture(t, llamacppCase(t), pol, nil, nil)
	f := fx.p.soleFacade()
	var nativeReads atomic.Int32
	e := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		nativeReads.Add(1)
		if _, err := w.Write([]byte(`{"object":"list","data":[{"id":"native-file"}]}`)); err != nil {
			t.Error(err)
		}
	}))
	defer e.Close()
	setBackend(t, f, e.URL, true)
	f.discovery.AddManual(selfNode("self", f.selfPort(), "native-file"))
	for _, path := range []string{"/models", "/v1/models"} {
		t.Run(path, func(t *testing.T) {
			w := httptest.NewRecorder()
			f.serveIngress(w, httptest.NewRequest(http.MethodGet, path, nil), "peer")
			assertProfileCatalog(t, w, "coder", "coder:api", "coder:local", "public-profile")
			if nativeReads.Load() != 0 {
				t.Fatalf("peer catalog fetched native inventory %d times", nativeReads.Load())
			}
		})
	}
}
