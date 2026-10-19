// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"nvpair-shared/nodepolicy"
)

func reviewR2TierFixture(t *testing.T, failover bool) (*Proxy, *recordingEngine, *recordingEngine) {
	t.Helper()
	tc := ollamaCase(t)
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{
		"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "m"}},
	})
	pol.ServedModels = nodepolicy.ServedModels{{Engine: "ollama", Model: "m"}}
	if err := pol.Validate(); err != nil {
		t.Fatal(err)
	}
	busy := newRecordingEngine(t, http.StatusServiceUnavailable)
	local := newRecordingEngine(t, http.StatusOK)
	d := NewDiscovery()
	if failover {
		d.AddManual(nodeForModel(t, "busy", busy.URL, "m"))
	}
	d.AddManual(selfNode("self", tc.profile.FacadePort, "m", "weak"))
	p := testProxy(tc.profile, d, tc.profile.FacadePort)
	p.SetPriority([]string{"busy", "self"})
	setBackend(t, p.soleFacade(), local.URL, true)
	p.admission.setPolicy(pol)
	return p, busy, local
}

func assertReviewR2EngineModel(t *testing.T, e *recordingEngine, want string) {
	t.Helper()
	var selected struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal([]byte(e.lastBody()), &selected); err != nil {
		t.Fatalf("decode engine body: %v", err)
	}
	if selected.Model != want {
		t.Fatalf("admitted %q but forwarded model=%q", want, selected.Model)
	}
}

func testReviewR2TierKeys(t *testing.T, failover bool) {
	t.Helper()
	for _, c := range []struct {
		name, body string
		accepted   bool
	}{
		{"canonical", `{"model":"weak"}`, true},
		{"escaped canonical", `{"mo\u0064el":"weak"}`, true},
		{"title case", `{"Model":"weak"}`, false},
		{"uppercase", `{"MODEL":"weak"}`, false},
		{"escaped title case", `{"Mo\u0064el":"weak"}`, false},
		{"duplicate case variant", `{"model":"unused","MODEL":"weak"}`, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			p, busy, local := reviewR2TierFixture(t, failover)
			w := httptest.NewRecorder()
			p.soleFacade().handleHTTP(w, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(c.body)))
			if !c.accepted {
				assertAmbiguousModelRejection(t, w)
				if busy.hits() != 0 || local.hits() != 0 {
					t.Fatal("unsupported model member dispatched before rejection")
				}
				return
			}
			if w.Code != http.StatusOK || local.hits() != 1 || w.Header().Get(nodepolicy.ModelHeader) != "m" {
				t.Fatalf("status=%d local hits=%d response model=%q", w.Code, local.hits(), w.Header().Get(nodepolicy.ModelHeader))
			}
			assertReviewR2EngineModel(t, local, "m")
			if failover {
				if busy.hits() != 1 {
					t.Fatalf("busy hits=%d, want one before self failover", busy.hits())
				}
				assertReviewR2EngineModel(t, busy, "m")
			}
		})
	}
}

func TestReviewR2TierSingleModelKey(t *testing.T) { testReviewR2TierKeys(t, false) }
func TestReviewR2TierFailover(t *testing.T)       { testReviewR2TierKeys(t, true) }

func TestReviewR2ConcreteModelKeys(t *testing.T) {
	for _, path := range []string{"self", "ingress", "strict self", "strict ingress"} {
		t.Run(path, func(t *testing.T) {
			for _, key := range []string{`model`, `mo\u0064el`, `Model`, `MODEL`, `Mo\u0064el`} {
				t.Run(key, func(t *testing.T) {
					p, e := strictHTTPFixture(t)
					body := strings.Replace(strictBody("m"), `"model"`, `"`+key+`"`, 1)
					r := httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(body))
					if strings.HasPrefix(path, "strict ") {
						r = strictHTTPRequestWith(t, body)
					} else {
						p.soleFacade().discovery.AddManual(selfNode("self", p.soleFacade().profile.FacadePort, "m"))
					}
					w := httptest.NewRecorder()
					serveStrictHTTP(p, strings.TrimPrefix(path, "strict "), w, r)
					if key != `model` && key != `mo\u0064el` {
						assertAmbiguousModelRejection(t, w)
						if e.hits() != 0 {
							t.Fatal("unsupported model member reached engine")
						}
						return
					}
					if w.Code != http.StatusOK || e.hits() != 1 {
						t.Fatalf("status=%d hits=%d body=%s", w.Code, e.hits(), w.Body)
					}
					assertReviewR2EngineModel(t, e, "m")
				})
			}
		})
	}
}

func TestReviewR2ModelDefaults(t *testing.T) {
	for _, path := range []string{"self", "ingress", "strict self", "strict ingress", "failover"} {
		t.Run(path, func(t *testing.T) {
			for _, c := range []struct {
				name, defaults string
				accepted       bool
			}{
				{"ordinary defaults", `{"temperature":0.5}`, true},
				{"title case model", `{"Model":"forbidden"}`, false},
				{"uppercase model", `{"MODEL":"forbidden"}`, false},
				{"escaped title case model", `{"Mo\u0064el":"forbidden"}`, false},
				{"duplicate escaped model", `{"mo\u0064el":"m","model":"forbidden","Model":"forbidden"}`, false},
			} {
				t.Run(c.name, func(t *testing.T) {
					var p *Proxy
					var e, busy *recordingEngine
					strict := strings.HasPrefix(path, "strict ")
					if strict {
						p, e = strictHTTPFixture(t)
					} else {
						p, busy, e = reviewR2TierFixture(t, path == "failover")
					}
					pol := p.admission.currentPolicy()
					pol.Revision = "0"
					pol.ServedModels = nodepolicy.ServedModels{{Engine: "ollama", Model: "m"}}
					if !strict {
						pol.Profiles = []nodepolicy.Profile{{Name: "m", Engine: "ollama", Model: "m"}}
					}
					if err := pol.Validate(); err != nil {
						t.Fatal(err)
					}
					// Bypass configuration validation deliberately: the effective-body
					// assertion must independently contain invalid installed defaults.
					pol.Profiles[0].RequestOptions = json.RawMessage(c.defaults)
					if strict {
						q := p.admission.strict.registry[*pol.Profiles[0].Qualification]
						q.RequestOptions = pol.Profiles[0].RequestOptions
					}
					p.admission.setPolicy(pol)
					r := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"m"}`))
					if path == "failover" {
						r = httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"weak"}`))
					} else if strict {
						r = strictHTTPRequest(t)
					}
					w := httptest.NewRecorder()
					serveStrictHTTP(p, strings.TrimPrefix(path, "strict "), w, r)
					if path == "failover" && busy.hits() != 1 {
						t.Fatalf("busy hits=%d, want one before self failover", busy.hits())
					}
					if !c.accepted {
						assertAmbiguousModelRejection(t, w)
						if e.hits() != 0 {
							t.Fatal("defaults changed admitted model and reached engine")
						}
						return
					}
					if w.Code != http.StatusOK || e.hits() != 1 {
						t.Fatalf("status=%d hits=%d body=%s", w.Code, e.hits(), w.Body)
					}
					assertReviewR2EngineModel(t, e, "m")
				})
			}
		})
	}
}

func TestReviewR2QualificationModelDefaults(t *testing.T) {
	for _, key := range []string{`model`, `Model`, `MODEL`, `mo\u0064el`, `Mo\u0064el`} {
		t.Run(key, func(t *testing.T) {
			q := testQualification("q", "ollama", "m")
			if err := q.validate(); err != nil {
				t.Fatal(err)
			}
			q.RequestOptions = json.RawMessage(`{"` + key + `":"forbidden"}`)
			if err := q.validate(); err == nil || err.Error() != "requestOptions may not set model" {
				t.Fatalf("model-equivalent defaults validation: %v", err)
			}
		})
	}
}

func TestReviewR2ModelMemberSpelling(t *testing.T) {
	for _, c := range []struct {
		name, body string
		rejected   bool
	}{
		{"canonical", `{"model":"m"}`, false},
		{"escaped canonical", `{"mo\u0064el":"m"}`, false},
		{"title case", `{"Model":"m"}`, true},
		{"uppercase", `{"MODEL":"m"}`, true},
		{"escaped title case", `{"Mo\u0064el":"m"}`, true},
		{"exact duplicate", `{"model":"m","model":"m"}`, true},
		{"escaped duplicate", `{"model":"m","mo\u0064el":"m"}`, true},
		{"nested case variant", `{"model":"m","options":{"Model":"other"}}`, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := ambiguousModelMember([]byte(c.body)); got != c.rejected {
				t.Fatalf("unsupported model member rejected=%v, want %v", got, c.rejected)
			}
		})
	}
}

func TestReviewR2EffectiveModel(t *testing.T) {
	for _, c := range []struct {
		name, body, admitted string
		matches              bool
	}{
		{"canonical", `{"model":"m"}`, "m", true},
		{"escaped canonical", `{"mo\u0064el":"m"}`, "m", true},
		{"nested model", `{"model":"m","options":{"Model":"other"}}`, "m", true},
		{"different model", `{"model":"other"}`, "m", false},
		{"duplicate model", `{"model":"m","model":"m"}`, "m", false},
		{"case variant", `{"Model":"m"}`, "m", false},
		{"overriding model", `{"model":"m","Model":"forbidden"}`, "m", false},
		{"missing model", `{}`, "", false},
		{"null model", `{"model":null}`, "", false},
		{"object model", `{"model":{}}`, "", false},
		{"array body", `[]`, "", false},
		{"trailing body", `{"model":"m"} {}`, "m", false},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := bodyModelMatches([]byte(c.body), c.admitted); got != c.matches {
				t.Fatalf("effective model matches=%v, want %v", got, c.matches)
			}
		})
	}
}
