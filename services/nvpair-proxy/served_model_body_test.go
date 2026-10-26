// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

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

func TestRequestOptionsProductionRoutes(t *testing.T) {
	for _, path := range []string{"self", "ingress", "strict self", "strict ingress", "failover"} {
		t.Run(path, func(t *testing.T) {
			for _, mode := range []nodepolicy.RequestOptionsMode{nodepolicy.RequestOptionsDefaults, nodepolicy.RequestOptionsEnforced} {
				t.Run(string(mode), func(t *testing.T) {
					strict := strings.HasPrefix(path, "strict ")
					var p *Proxy
					var local, busy *recordingEngine
					if strict {
						p, local = strictHTTPFixture(t)
					} else {
						p, busy, local = reviewR2TierFixture(t, path == "failover")
					}
					pol := p.admission.currentPolicy()
					if !strict {
						pol.Profiles = []nodepolicy.Profile{{Name: "m", Engine: "ollama", Model: "m"}}
					}
					pol.Profiles[0].RequestOptions = json.RawMessage(`{"temperature":0.1,"top_k":4,"min_p":0.02,"repeat_penalty":1.1}`)
					pol.Profiles[0].RequestOptionsMode = mode
					pol.Revision = "0"
					if err := pol.Validate(); err != nil {
						t.Fatalf("policy: %v", err)
					}
					if strict {
						q := p.admission.strict.registry[*pol.Profiles[0].Qualification]
						q.RequestOptions, q.RequestOptionsMode = pol.Profiles[0].RequestOptions, mode
					}
					p.admission.setPolicy(pol)
					model := "m"
					if path == "failover" {
						model = "weak"
					}
					body := strings.TrimSuffix(strictBody(model), "}") + `,"temperature":0.8}`
					r := httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(body))
					if strict {
						r = strictHTTPRequestWith(t, body)
					}
					w := httptest.NewRecorder()
					serveStrictHTTP(p, strings.TrimPrefix(path, "strict "), w, r)
					if w.Code != http.StatusOK || local.hits() != 1 {
						t.Fatalf("status=%d hits=%d error=%s", w.Code, local.hits(), w.Body)
					}
					var got struct {
						Model         string  `json:"model"`
						Temperature   float64 `json:"temperature"`
						TopK          int     `json:"top_k"`
						MinP          float64 `json:"min_p"`
						RepeatPenalty float64 `json:"repeat_penalty"`
						MaxTokens     int     `json:"max_tokens"`
					}
					if err := json.Unmarshal([]byte(local.lastBody()), &got); err != nil {
						t.Fatalf("engine body: %v", err)
					}
					wantTemperature := 0.8
					if mode == nodepolicy.RequestOptionsEnforced {
						wantTemperature = 0.1
					}
					if got.Model != "m" || got.Temperature != wantTemperature || got.TopK != 4 || got.MinP != 0.02 || got.RepeatPenalty != 1.1 || got.MaxTokens != 16 {
						t.Fatalf("engine settings=%+v", got)
					}
					if path == "failover" {
						var peer map[string]json.RawMessage
						if err := json.Unmarshal([]byte(busy.lastBody()), &peer); err != nil {
							t.Fatalf("peer body: %v", err)
						}
						if busy.hits() != 1 || string(peer["temperature"]) != "0.8" || peer["top_k"] != nil {
							t.Fatalf("origin applied local profile to peer: %s", busy.lastBody())
						}
					}
					bufferAssertTotals(t, p.buffers, 0, 0, 0)
				})
			}
		})
	}
}

func TestRequestOptionsAmbiguityHTTP(t *testing.T) {
	for _, path := range []string{"self", "ingress", "strict self", "strict ingress"} {
		t.Run(path, func(t *testing.T) {
			for _, tc := range []struct{ name, fields string }{
				{"exact duplicate", `"temperature":0.8,"temperature":0.9`},
				{"escaped duplicate", `"temperature":0.8,"tempera\u0074ure":0.9`},
				{"folded duplicate", `"temperature":0.8,"Temperature":0.9`},
				{"single folded variant", `"Temperature":0.9`},
			} {
				t.Run(tc.name, func(t *testing.T) {
					p, e := strictHTTPFixture(t)
					pol := p.admission.currentPolicy()
					pol.Profiles[0].RequestOptions = json.RawMessage(`{"temperature":0.1}`)
					pol.Profiles[0].RequestOptionsMode = nodepolicy.RequestOptionsEnforced
					q := p.admission.strict.registry[*pol.Profiles[0].Qualification]
					q.RequestOptions, q.RequestOptionsMode = pol.Profiles[0].RequestOptions, nodepolicy.RequestOptionsEnforced
					p.admission.setPolicy(pol)
					body := strings.TrimSuffix(strictBody("m"), "}") + "," + tc.fields + "}"
					r := httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(body))
					if strings.HasPrefix(path, "strict ") {
						r = strictHTTPRequestWith(t, body)
					} else {
						p.soleFacade().discovery.AddManual(selfNode("self", p.soleFacade().profile.FacadePort, "m"))
					}
					w := httptest.NewRecorder()
					serveStrictHTTP(p, strings.TrimPrefix(path, "strict "), w, r)
					var refusal struct {
						Code  string `json:"code"`
						Error string `json:"error"`
					}
					if err := json.Unmarshal(w.Body.Bytes(), &refusal); err != nil {
						t.Fatalf("decode refusal: %v", err)
					}
					if w.Code != http.StatusBadRequest || refusal.Code != "ambiguous-options" || refusal.Error != ambiguousOptionsMessage || e.hits() != 0 || p.admission.currentActive() != 0 {
						t.Fatalf("status=%d refusal=%+v hits=%d", w.Code, refusal, e.hits())
					}
					bufferAssertTotals(t, p.buffers, 0, 0, 0)
				})
			}
		})
	}
}

func TestRequestOptionsServingOwner(t *testing.T) {
	destination, _, local := reviewR2TierFixture(t, false)
	pol := destination.admission.currentPolicy()
	pol.Profiles = []nodepolicy.Profile{{Name: "destination", Engine: "ollama", Model: "m", RequestOptions: json.RawMessage(`{"temperature":0.1}`), RequestOptionsMode: nodepolicy.RequestOptionsEnforced}}
	destination.admission.setPolicy(pol)
	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { destination.soleFacade().serveIngress(w, r, "origin") }))
	defer peer.Close()
	originPolicy := nodepolicy.Default()
	originPolicy.Profiles = []nodepolicy.Profile{{Name: "origin", Engine: "ollama", Model: "m", RequestOptions: json.RawMessage(`{"temperature":0.9,"seed":7}`), RequestOptionsMode: nodepolicy.RequestOptionsEnforced}}
	fx := newTierFixture(t, ollamaCase(t), originPolicy, map[string][]string{"a": {"m"}}, nil)
	fx.p.soleFacade().discovery.AddManual(nodeForModel(t, "a", peer.URL, "m"))
	w := fx.do(t, routeChatCompletions, strings.TrimSuffix(strictBody("m"), "}")+`,"temperature":0.8}`)
	if w.Code != http.StatusOK || local.hits() != 1 || fx.eng["a"].hits() != 0 {
		t.Fatalf("status=%d destination=%d origin=%d response=%s", w.Code, local.hits(), fx.eng["a"].hits(), w.Body)
	}
	var got map[string]json.RawMessage
	if err := json.Unmarshal([]byte(local.lastBody()), &got); err != nil {
		t.Fatalf("destination body: %v", err)
	}
	if string(got["temperature"]) != "0.1" || got["seed"] != nil {
		t.Fatalf("serving profile lost or origin profile applied: %s", local.lastBody())
	}
	bufferAssertTotals(t, fx.p.buffers, 0, 0, 0)
	bufferAssertTotals(t, destination.buffers, 0, 0, 0)
}

func TestRequestOptionsHybridSelf(t *testing.T) {
	for _, tc := range []struct {
		name, fields string
		mode         nodepolicy.RequestOptionsMode
		invalid      bool
		want         float64
	}{
		{"defaults keeps caller", `"temperature":0.8`, nodepolicy.RequestOptionsDefaults, false, 0.8},
		{"enforced replaces caller", `"temperature":0.8`, nodepolicy.RequestOptionsEnforced, false, 0.1},
		{"enforced ambiguity refuses self", `"temperature":0.8,"Temperature":0.9`, nodepolicy.RequestOptionsEnforced, true, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, f, e := hybridSelfProxy(t, true)
			pol := p.admission.currentPolicy()
			pol.Profiles = append([]nodepolicy.Profile(nil), pol.Profiles...)
			pol.Profiles[0].RequestOptions = json.RawMessage(`{"temperature":0.1}`)
			pol.Profiles[0].RequestOptionsMode = tc.mode
			q := p.admission.strict.registry[*pol.Profiles[0].Qualification]
			q.RequestOptions, q.RequestOptionsMode = pol.Profiles[0].RequestOptions, tc.mode
			p.admission.setPolicy(pol)
			body := strings.TrimSuffix(hybridSelfBody, "}") + "," + tc.fields + "}"
			w, _ := serveHybridDispatch(t, f, body, hybridTestPolicy(1, 5*time.Second))
			if tc.invalid {
				var refusal struct {
					Code  string `json:"code"`
					Error string `json:"error"`
				}
				if err := json.Unmarshal(w.Body.Bytes(), &refusal); err != nil {
					t.Fatalf("decode refusal: %v", err)
				}
				if w.Code != http.StatusBadRequest || refusal.Code != "ambiguous-options" || refusal.Error != ambiguousOptionsMessage || e.hits() != 0 {
					t.Fatalf("status=%d refusal=%+v hits=%d", w.Code, refusal, e.hits())
				}
				return
			}
			if w.Code != http.StatusOK || e.hits() != 1 {
				t.Fatalf("status=%d hits=%d response=%s", w.Code, e.hits(), w.Body)
			}
			var got struct {
				Model       string  `json:"model"`
				Temperature float64 `json:"temperature"`
			}
			if err := json.Unmarshal([]byte(e.lastBody()), &got); err != nil {
				t.Fatalf("engine body: %v", err)
			}
			if got.Model != "llama3:latest" || got.Temperature != tc.want {
				t.Fatalf("rewritten effective body=%+v", got)
			}
		})
	}
}
