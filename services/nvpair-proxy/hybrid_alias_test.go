// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"nvpair-shared/nodepolicy"
)

// The nine reserved hybrid alias names, spelled independently of the code
// under test.
var (
	bareAliases   = []string{"weak", "medium", "strong"}
	suffixAliases = []string{"weak-api", "weak-local", "medium-api", "medium-local", "strong-api", "strong-local"}
	allAliases    = append(slices.Clone(bareAliases), suffixAliases...)
)

// stagedHybridPolicy has weak as API-only, medium as API then local, and strong
// as local only, with the disabled provider objects the API entries refer to.
func stagedHybridPolicy(t *testing.T) nodepolicy.Policy {
	t.Helper()
	pol, err := nodepolicy.Parse([]byte(apiTierPolicy))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	pol.Tiers["strong"] = []nodepolicy.TierEntry{{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "big:70b"}}
	return pol
}

// aliasFixture serves every alias name as a model on its one node, so a request
// that wrongly reached routing as a concrete model would find an owner and
// reach the engine.
func aliasFixture(t *testing.T, tc engineCase) *tierFixture {
	t.Helper()
	models := append([]string{"mid:7b", "big:70b"}, allAliases...)
	for _, a := range allAliases {
		models = append(models, a+":latest")
	}
	return newTierFixture(t, tc, stagedHybridPolicy(t), map[string][]string{"a": models}, nil)
}

type aliasRefusal struct {
	Error  string `json:"error"`
	Code   string `json:"code"`
	Reason string `json:"reason"`
	Model  string `json:"model"`
}

// assertRefusedWithoutSideEffects checks the 400 answer and that nothing past
// request parsing happened: no engine request, no workload or admission
// notification, no admission slot, and only the one proxy/request report.
func assertRefusedWithoutSideEffects(t *testing.T, tc engineCase, fx *tierFixture, rec *httptest.ResponseRecorder, model, wantReason string) {
	t.Helper()
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body %s", rec.Code, rec.Body)
	}
	var got aliasRefusal
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode refusal %q: %v", rec.Body, err)
	}
	if got.Code != "feature_unavailable" || got.Reason != wantReason || got.Model != strings.TrimSpace(model) {
		t.Errorf("refusal = %+v, want code feature_unavailable, reason %q, model %q", got, wantReason, strings.TrimSpace(model))
	}
	for id, e := range fx.eng {
		if n := e.hits(); n != 0 {
			t.Errorf("engine %s received %d request(s), want none", id, n)
		}
	}
	if n := fx.p.admission.currentActive(); n != 0 {
		t.Errorf("admission active = %d, want 0", n)
	}
	var methods []string
	for _, f := range frames(t, fx.w) {
		methods = append(methods, f.Method)
	}
	requestMethod := tc.profile.Name + ":proxy/request"
	if want := []string{requestMethod}; !slices.Equal(methods, want) {
		t.Fatalf("notifications = %v, want only %v (no workload, wake or admission)", methods, want)
	}
	var ev RequestEvent
	if err := json.Unmarshal(notificationsOf(t, fx.w, requestMethod)[0], &ev); err != nil {
		t.Fatalf("decode request event: %v", err)
	}
	if ev.Status != http.StatusBadRequest || ev.Error != wantReason {
		t.Errorf("request event status %d error %q, want 400 %q", ev.Status, ev.Error, wantReason)
	}
}

// A -api or -local alias is refused on every route, including the routes that
// serve a bare tier, and a bare tier is refused where no legacy tier path
// exists.
func TestAliasRequestsAreRefusedBeforeAnySideEffect(t *testing.T) {
	test := func(t *testing.T, tc engineCase, path, model, wantReason string) {
		t.Helper()
		fx := aliasFixture(t, tc)
		rec := fx.do(t, path, `{"model":"`+model+`","messages":[{"role":"user","content":"x"}]}`)
		assertRefusedWithoutSideEffects(t, tc, fx, rec, model, wantReason)
	}

	t.Run("ollama", func(t *testing.T) {
		tc := ollamaCase(t)
		for _, path := range []string{"/api/chat", "/api/generate", "/api/embed", "/v1/chat/completions", "/v1/completions", "/v1/messages"} {
			for _, alias := range suffixAliases {
				t.Run(path+" "+alias, func(t *testing.T) { test(t, tc, path, alias, aliasReasonNotInferenceReady) })
			}
		}
		// Not classified as inference, so the proxy would forward the alias
		// to the engine as if it were a model.
		for _, path := range []string{"/v1/responses", "/api/show", "/api/pull", "/api/unknown"} {
			for _, alias := range allAliases {
				reason := aliasReasonUnsupportedUse
				if slices.Contains(suffixAliases, alias) {
					reason = aliasReasonNotInferenceReady
				}
				t.Run(path+" "+alias, func(t *testing.T) { test(t, tc, path, alias, reason) })
			}
		}
	})

	t.Run("lmstudio", func(t *testing.T) {
		tc := lmstudioCase(t)
		for _, path := range []string{"/v1/chat/completions", "/v1/messages"} {
			for _, alias := range suffixAliases {
				t.Run(path+" "+alias, func(t *testing.T) { test(t, tc, path, alias, aliasReasonNotInferenceReady) })
			}
		}
		for _, path := range []string{"/v1/responses", "/api/v0/chat/completions"} {
			for _, alias := range bareAliases {
				t.Run(path+" "+alias, func(t *testing.T) { test(t, tc, path, alias, aliasReasonUnsupportedUse) })
			}
		}
	})

	t.Run("padded name is still an alias", func(t *testing.T) {
		tc := ollamaCase(t)
		test(t, tc, "/api/chat", " weak-api ", aliasReasonNotInferenceReady)
		test(t, tc, "/api/chat", " weak", aliasReasonUnsupportedUse)
		test(t, tc, "/api/chat", "weak ", aliasReasonUnsupportedUse)
	})
}

// Names that merely resemble an alias are ordinary models and are forwarded.
func TestAliasLookalikesAreOrdinaryModels(t *testing.T) {
	tc := ollamaCase(t)
	models := []string{"Weak", "weak-api-x", "weaker", "weak:latest", "weak-api:latest"}
	fx := newTierFixture(t, tc, stagedHybridPolicy(t), map[string][]string{"a": models}, nil)
	for _, model := range models {
		if rec := fx.do(t, "/api/chat", `{"model":"`+model+`"}`); rec.Code != http.StatusOK {
			t.Errorf("model %q: status %d, want 200", model, rec.Code)
		}
	}
	if got := fx.eng["a"].hits(); got != len(models) {
		t.Errorf("engine received %d requests, want %d", got, len(models))
	}
}

// A bare tier on a classified inference route keeps the legacy local-only path
// byte for byte, whether or not disabled hybrid settings are staged.
func TestBareTierIsUnchangedByStagedHybridSettings(t *testing.T) {
	legacy := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{
		"medium": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "mid:7b"}},
		"strong": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "big:70b"}},
	})
	staged := stagedHybridPolicy(t)
	if staged.Routing.Listener.Enabled {
		t.Fatal("fixture error: the staged listener must be disabled")
	}

	type outcome struct {
		status int
		served string
		body   string
		tier   string
		events []string
	}
	run := func(t *testing.T, pol nodepolicy.Policy, path, model string) outcome {
		t.Helper()
		fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"a": {"mid:7b", "big:70b"}}, nil)
		rec := fx.do(t, path, `{"model":"`+model+`","prompt":"x"}`)
		var methods []string
		for _, f := range frames(t, fx.w) {
			methods = append(methods, f.Method)
		}
		out := outcome{status: rec.Code, served: fx.servedBy(t), tier: rec.Header().Get("X-PAIR-Tier"), events: methods}
		if e := fx.eng["a"]; e != nil {
			out.body = e.lastBody()
		}
		return out
	}

	test := func(name, path, model string, wantStatus int, wantUpstreamModel string) {
		t.Run(name, func(t *testing.T) {
			want := run(t, legacy, path, model)
			got := run(t, staged, path, model)
			if got.status != wantStatus {
				t.Fatalf("staged status = %d, want %d", got.status, wantStatus)
			}
			if wantUpstreamModel != "" {
				var sent struct {
					Model string `json:"model"`
				}
				if err := json.Unmarshal([]byte(got.body), &sent); err != nil {
					t.Fatalf("decode upstream body %q: %v", got.body, err)
				}
				if sent.Model != wantUpstreamModel {
					t.Errorf("upstream model = %q, want %q", sent.Model, wantUpstreamModel)
				}
			}
			if got.status != want.status || got.served != want.served || got.body != want.body ||
				got.tier != want.tier || !slices.Equal(got.events, want.events) {
				t.Errorf("staged settings changed the bare tier path:\n legacy %+v\n staged %+v", want, got)
			}
		})
	}
	test("openai chat route", "/v1/chat/completions", "strong", http.StatusOK, "big:70b")
	test("native chat route", "/api/chat", "strong", http.StatusOK, "big:70b")
	test("medium skips the API entry", "/v1/chat/completions", "medium", http.StatusOK, "mid:7b")
	// weak has only an API entry, so the request falls through to the
	// stronger local entry exactly as it does when weak has no entry at all.
	test("api-only tier falls through to a stronger local entry", "/v1/chat/completions", "weak", http.StatusOK, "mid:7b")
}

func modelIDs(t *testing.T, rec *httptest.ResponseRecorder) []string {
	t.Helper()
	var out struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode model list %q: %v", rec.Body, err)
	}
	if out.Object != "list" {
		t.Errorf("object = %q, want list", out.Object)
	}
	ids := []string{}
	for _, d := range out.Data {
		ids = append(ids, d.ID)
	}
	return ids
}

func aliasesOf(tiers ...string) []string {
	var out []string
	for _, t := range tiers {
		out = append(out, t, t+"-api", t+"-local")
	}
	return out
}

func TestConfiguredAliases(t *testing.T) {
	local := nodepolicy.TierEntry{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "m"}
	api := nodepolicy.TierEntry{Kind: nodepolicy.EntryAPI, Deployment: "d"}
	test := func(name string, tiers map[string][]nodepolicy.TierEntry, want []string) {
		t.Run(name, func(t *testing.T) {
			pol := nodepolicy.Default()
			pol.Tiers = tiers
			if got := configuredAliases(pol, nil); !slices.Equal(got, want) {
				t.Errorf("configuredAliases = %v, want %v", got, want)
			}
		})
	}
	test("no tiers", nil, nil)
	test("empty tier", map[string][]nodepolicy.TierEntry{"weak": {}}, nil)
	test("local only", map[string][]nodepolicy.TierEntry{"weak": {local}}, aliasesOf("weak"))
	test("api only", map[string][]nodepolicy.TierEntry{"strong": {api}}, aliasesOf("strong"))
	test("weakest first", map[string][]nodepolicy.TierEntry{"strong": {local}, "weak": {api}, "medium": {local, api}},
		aliasesOf("weak", "medium", "strong"))

	// A declared tier is listed without any entry: by this node's own roles,
	// or by a peer's.
	t.Run("declared tiers", func(t *testing.T) {
		pol := nodepolicy.Default()
		pol.Tiers = map[string][]nodepolicy.TierEntry{"weak": {local}}
		pol.Roles = map[string]nodepolicy.Role{"strong": {Engine: "ollama", Model: "m"}}
		if got, want := configuredAliases(pol, nil), aliasesOf("weak", "strong"); !slices.Equal(got, want) {
			t.Errorf("self roles: configuredAliases = %v, want %v", got, want)
		}
		if got, want := configuredAliases(pol, map[string]bool{"medium": true}), aliasesOf("weak", "medium", "strong"); !slices.Equal(got, want) {
			t.Errorf("peer tiers: configuredAliases = %v, want %v", got, want)
		}
	})
}

// A facade's model list adds the tiers peers in its overlays declare; the
// routing listener, which has no overlay, lists this node's own roles only.
func TestModelListOffersDeclaredTiers(t *testing.T) {
	pol := nodepolicy.Default()
	pol.Roles = map[string]nodepolicy.Role{"strong": {Engine: "lmstudio", Model: "qwen3-8b"}}
	fx := listFixture(t, pol, true)
	f := fx.p.soleFacade()
	for _, n := range f.discovery.Nodes() {
		n.Roles = map[string]nodepolicy.AdvertisedRole{"medium": {Engine: "lmstudio", Model: "not-installed"}}
		f.discovery.AddManual(n)
	}
	if got, want := modelIDs(t, fx.listModels(t, "/v1/models")), append([]string{"qwen3-8b"}, aliasesOf("medium", "strong")...); !slices.Equal(got, want) {
		t.Errorf("facade model list = %v, want %v", got, want)
	}
	var ids []string
	for _, raw := range hybridAliasRecords(fx.p.admission.currentPolicy(), nil, nil) {
		var item struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(raw, &item); err != nil {
			t.Fatalf("decode %s: %v", raw, err)
		}
		ids = append(ids, item.ID)
	}
	if want := aliasesOf("strong"); !slices.Equal(ids, want) {
		t.Errorf("policy-only aliases = %v, want %v", ids, want)
	}
}

// Declared roles are local candidates: a -api alias never reaches them, and
// the suffixed aliases stay refused until hybrid dispatch exists.
func TestSuffixedAliasesNeverReachDeclaredRoles(t *testing.T) {
	tc := ollamaCase(t)
	for _, alias := range []string{"weak-api", "weak-local"} {
		t.Run(alias, func(t *testing.T) {
			fx := newTierFixture(t, tc, nodepolicy.Default(), map[string][]string{"a": {"small:1b"}}, nil)
			fx.declare(t, "a", map[string]nodepolicy.AdvertisedRole{"weak": {Engine: "ollama", Model: "small:1b"}})
			rec := fx.do(t, "/v1/chat/completions", `{"model":"`+alias+`"}`)
			assertRefusedWithoutSideEffects(t, tc, fx, rec, alias, aliasReasonNotInferenceReady)
		})
	}
}

// The records are synthetic and carry nothing of the policy beyond the name.
func TestHybridAliasRecordsNameOnlyTheAlias(t *testing.T) {
	pol := stagedHybridPolicy(t)
	records := hybridAliasRecords(pol, nil, nil)
	if len(records) != len(aliasesOf("weak", "medium", "strong")) {
		t.Fatalf("got %d records, want one per alias", len(records))
	}
	for _, raw := range records {
		var rec map[string]any
		if err := json.Unmarshal(raw, &rec); err != nil {
			t.Fatalf("decode %s: %v", raw, err)
		}
		keys := make([]string, 0, len(rec))
		for k := range rec {
			keys = append(keys, k)
		}
		slices.Sort(keys)
		if want := []string{"created", "id", "object", "owned_by"}; !slices.Equal(keys, want) {
			t.Errorf("record %s has fields %v, want %v", raw, keys, want)
		}
	}
	skipped := hybridAliasRecords(pol, nil, map[string]modelListItem{"weak-api": {}})
	if len(skipped) != len(records)-1 {
		t.Errorf("skipping a seen id left %d records, want %d", len(skipped), len(records)-1)
	}
}

func (fx *tierFixture) listModels(t *testing.T, path string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	fx.p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
	return rec
}

// listFixture is an LM Studio facade whose one node answers /v1/models.
func listFixture(t *testing.T, pol nodepolicy.Policy, up bool) *tierFixture {
	t.Helper()
	fx := &tierFixture{w: &recordingWriter{}, eng: map[string]*recordingEngine{}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"qwen3-8b"}]}`))
	}))
	url := srv.URL
	if !up {
		srv.Close()
	} else {
		t.Cleanup(srv.Close)
	}
	tc := lmstudioCase(t)
	disc := NewDiscovery()
	disc.AddManual(nodeFor(t, "n1", url))
	fx.p = newTestProxy(tc.profile, NewCodec(fx.w), disc, tc.profile.FacadePort)
	if err := pol.Validate(); err != nil {
		t.Fatal(err)
	}
	fx.p.admission.setPolicy(pol)
	return fx
}

func TestModelList_AliasesMergeWithUpstreamModels(t *testing.T) {
	fx := listFixture(t, stagedHybridPolicy(t), true)
	rec := fx.listModels(t, "/v1/models")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	want := append([]string{"qwen3-8b"}, aliasesOf("weak", "medium", "strong")...)
	if got := modelIDs(t, rec); !slices.Equal(got, want) {
		t.Errorf("model list = %v, want %v", got, want)
	}
}

// With no upstream answering, configured aliases are still listed.
func TestModelList_AliasesSurviveEveryUpstreamBeingDown(t *testing.T) {
	fx := listFixture(t, stagedHybridPolicy(t), false)
	rec := fx.listModels(t, "/v1/models")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	if got, want := modelIDs(t, rec), aliasesOf("weak", "medium", "strong"); !slices.Equal(got, want) {
		t.Errorf("model list = %v, want %v", got, want)
	}
}

// A tier whose entries are all API deployments is listed with both variants
// even though it has no local entry.
func TestModelList_ListsATierWithZeroLocalEntries(t *testing.T) {
	pol := stagedHybridPolicy(t)
	delete(pol.Tiers, "medium")
	delete(pol.Tiers, "strong")
	for _, up := range []bool{true, false} {
		fx := listFixture(t, pol, up)
		rec := fx.listModels(t, "/v1/models")
		if rec.Code != http.StatusOK {
			t.Fatalf("upstream up=%v: status = %d, want 200", up, rec.Code)
		}
		got := modelIDs(t, rec)
		for _, id := range aliasesOf("weak") {
			if !slices.Contains(got, id) {
				t.Errorf("upstream up=%v: %q missing from %v", up, id, got)
			}
		}
		if slices.Contains(got, "medium") || slices.Contains(got, "strong-api") {
			t.Errorf("upstream up=%v: unconfigured tier listed in %v", up, got)
		}
	}
}

// Without configured tiers the answer is what it was: 503 with every upstream
// down, and the upstream list otherwise.
func TestModelList_WithoutTiersKeepsItsStatuses(t *testing.T) {
	down := listFixture(t, nodepolicy.Default(), false)
	if rec := down.listModels(t, "/v1/models"); rec.Code != http.StatusServiceUnavailable {
		t.Errorf("upstream down: status = %d, want 503", rec.Code)
	}
	up := listFixture(t, nodepolicy.Default(), true)
	rec := up.listModels(t, "/v1/models")
	if rec.Code != http.StatusOK {
		t.Fatalf("upstream up: status = %d, want 200", rec.Code)
	}
	if got, want := modelIDs(t, rec), []string{"qwen3-8b"}; !slices.Equal(got, want) {
		t.Errorf("upstream up: model list = %v, want %v", got, want)
	}
}

// The native Ollama list has no alias records, so it stays 503 with every
// upstream down even when tiers are configured.
func TestModelList_NativeListStays503WhenDown(t *testing.T) {
	tc := ollamaCase(t)
	e := httptest.NewServer(http.NotFoundHandler())
	url := e.URL
	e.Close()
	fx := &tierFixture{w: &recordingWriter{}, eng: map[string]*recordingEngine{}}
	disc := NewDiscovery()
	disc.AddManual(nodeFor(t, "n1", url))
	fx.p = newTestProxy(tc.profile, NewCodec(fx.w), disc, tc.profile.FacadePort)
	fx.p.admission.setPolicy(stagedHybridPolicy(t))
	if rec := fx.listModels(t, "/api/tags"); rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", rec.Code)
	}
}

// The Ollama native list merges models by their normalized name, so a reserved
// alias that an engine reports bare stays distinct from the same word tagged
// ":latest", while an ordinary untagged name still merges with its tagged form.
// No alias record is invented on the native list.
func TestModelList_NativeListKeepsAReservedNameUntagged(t *testing.T) {
	tc := ollamaCase(t)
	e := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"models":[{"model":"weak"},{"model":"weak:latest"},{"model":"llama3"},{"model":"llama3:latest"}]}`))
	}))
	t.Cleanup(e.Close)
	fx := &tierFixture{w: &recordingWriter{}, eng: map[string]*recordingEngine{}}
	disc := NewDiscovery()
	disc.AddManual(nodeFor(t, "n1", e.URL))
	fx.p = newTestProxy(tc.profile, NewCodec(fx.w), disc, tc.profile.FacadePort)
	fx.p.admission.setPolicy(stagedHybridPolicy(t))

	rec := fx.listModels(t, "/api/tags")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	var list struct {
		Models []struct {
			Model string `json:"model"`
		} `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
		t.Fatalf("decode %s: %v", rec.Body, err)
	}
	var got []string
	for _, m := range list.Models {
		got = append(got, m.Model)
	}
	if want := []string{"weak", "weak:latest", "llama3"}; !slices.Equal(got, want) {
		t.Errorf("native model list = %v, want %v", got, want)
	}
}

func (fx *tierFixture) request(t *testing.T, method, target, body string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	fx.p.soleFacade().handleHTTP(rec, httptest.NewRequest(method, target, strings.NewReader(body)))
	return rec
}

// A -api or -local alias is refused wherever an engine could read a model name
// from, not only in the "model" key the router routes on.
func TestAliasInOtherModelPlacesIsRefused(t *testing.T) {
	test := func(t *testing.T, tc engineCase, method, target, body, alias string) {
		t.Helper()
		fx := aliasFixture(t, tc)
		rec := fx.request(t, method, target, body)
		assertRefusedWithoutSideEffects(t, tc, fx, rec, alias, aliasReasonNotInferenceReady)
	}

	t.Run("ollama", func(t *testing.T) {
		tc := ollamaCase(t)
		for _, path := range []string{"/api/pull", "/api/show", "/api/delete", "/api/create"} {
			t.Run(path+" name", func(t *testing.T) {
				test(t, tc, http.MethodPost, path, `{"name":"strong-local"}`, "strong-local")
			})
		}
		t.Run("padded name", func(t *testing.T) {
			test(t, tc, http.MethodPost, "/api/show", `{"name":" weak-api "}`, "weak-api")
		})
		t.Run("case-variant Model beside a plain model", func(t *testing.T) {
			test(t, tc, http.MethodPost, "/api/chat", `{"model":"mid:7b","Model":"weak-api"}`, "weak-api")
		})
		t.Run("case-variant MODEL ahead of a plain model", func(t *testing.T) {
			test(t, tc, http.MethodPost, "/api/chat", `{"MODEL":"medium-local","model":"mid:7b"}`, "medium-local")
		})
		t.Run("case-variant NAME", func(t *testing.T) {
			test(t, tc, http.MethodPost, "/api/pull", `{"NAME":"weak-local"}`, "weak-local")
		})
		t.Run("repeated model key", func(t *testing.T) {
			test(t, tc, http.MethodPost, "/api/chat", `{"model":"weak-api","model":"mid:7b"}`, "weak-api")
		})
		t.Run("model query parameter", func(t *testing.T) {
			test(t, tc, http.MethodGet, "/props?model=weak-api", ``, "weak-api")
		})
	})

	t.Run("lmstudio", func(t *testing.T) {
		tc := lmstudioCase(t)
		t.Run("v1 model path", func(t *testing.T) {
			test(t, tc, http.MethodGet, "/v1/models/weak-api", ``, "weak-api")
		})
		t.Run("v0 model path", func(t *testing.T) {
			test(t, tc, http.MethodGet, "/api/v0/models/strong-local", ``, "strong-local")
		})
		t.Run("escaped model path", func(t *testing.T) {
			test(t, tc, http.MethodGet, "/v1/models/medium%2Dapi", ``, "medium-api")
		})
	})
}

// Bare tiers in those places, and names that only resemble an alias, keep
// today's behavior: they are forwarded.
func TestBareTierAndLookalikesInOtherModelPlacesAreForwarded(t *testing.T) {
	test := func(name string, tc engineCase, method, target, body string) {
		t.Run(name, func(t *testing.T) {
			fx := aliasFixture(t, tc)
			if rec := fx.request(t, method, target, body); rec.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
			}
			if got := fx.eng["a"].hits(); got != 1 {
				t.Errorf("engine received %d requests, want 1", got)
			}
		})
	}
	ollama, lmstudio := ollamaCase(t), lmstudioCase(t)
	test("bare tier as name", ollama, http.MethodPost, "/api/pull", `{"name":"weak"}`)
	test("bare tier as query model", ollama, http.MethodGet, "/props?model=medium", ``)
	test("bare tier as model path", lmstudio, http.MethodGet, "/v1/models/strong", ``)
	test("tagged alias as name", ollama, http.MethodPost, "/api/pull", `{"name":"weak-api:latest"}`)
	test("case variant alias as name", ollama, http.MethodPost, "/api/pull", `{"name":"Weak-API"}`)
	test("organization-qualified model path", lmstudio, http.MethodGet, "/v1/models/org/weak-api", ``)
	test("alias in an unrelated key", ollama, http.MethodPost, "/api/chat", `{"model":"mid:7b","prompt":"weak-api","title":"weak-local"}`)
	test("alias as a nested model", ollama, http.MethodPost, "/api/chat", `{"model":"mid:7b","options":{"model":"weak-api"}}`)
}

// spyRecorder snapshots proxy state at the moment a status is written, which
// is while handleHTTP still holds whatever it has taken for the request.
type spyRecorder struct {
	*httptest.ResponseRecorder
	snapshot func()
}

func (s *spyRecorder) WriteHeader(code int) {
	s.snapshot()
	s.ResponseRecorder.WriteHeader(code)
}

// held is what a request has taken from the proxy by the time it answers.
type held struct{ reservations, active, activity, inflight, wakes int }

func (h held) none() bool { return h == held{} }

// A refused alias answers before the CORS preflight, the reservation and
// admission. A self candidate that advertises every alias would be reserved,
// admitted or woken if the refusal came any later, and each would be visible
// at the moment the 400 is written. The control request proves the snapshot
// does see those effects.
func TestAliasRefusalPrecedesPreflightReservationAndAdmission(t *testing.T) {
	tc := ollamaCase(t)
	models := append([]string{"mid:7b"}, allAliases...)
	for _, a := range allAliases {
		models = append(models, a+":latest")
	}
	const selfID = "self-uuid"

	type fixture struct {
		p     *Proxy
		w     *recordingWriter
		local *recordingEngine
	}
	newFixture := func(t *testing.T, healthy bool) fixture {
		t.Helper()
		fx := fixture{w: &recordingWriter{}, local: newRecordingEngine(t, http.StatusOK)}
		disc := NewDiscovery()
		disc.AddManual(selfNode(selfID, tc.profile.FacadePort, models...))
		fx.p = newTestProxy(tc.profile, NewCodec(fx.w), disc, tc.profile.FacadePort)
		setBackend(t, fx.p.soleFacade(), fx.local.URL, healthy)
		fx.p.noteSelf(selfID, tc.profile.Name, models)
		fx.p.SetPriority([]string{selfID})
		fx.p.admission.setIntent(map[string]bool{tc.profile.Name: true})
		pol := stagedHybridPolicy(t)
		if err := pol.Validate(); err != nil {
			t.Fatal(err)
		}
		fx.p.admission.setPolicy(pol)
		return fx
	}
	// serve runs one request and returns its answer and what was held when the
	// status was written.
	serve := func(fx fixture, r *http.Request) (*httptest.ResponseRecorder, held) {
		var at held
		rec := &spyRecorder{ResponseRecorder: httptest.NewRecorder()}
		rec.snapshot = func() {
			fx.p.priorityMu.Lock()
			at.reservations = len(fx.p.priorityReservations)
			fx.p.priorityMu.Unlock()
			a := fx.p.admission
			a.mu.Lock()
			at.active = a.active
			at.activity = len(a.lastActivity)
			at.inflight = len(a.inflight)
			at.wakes = len(a.wakeSentAt)
			a.mu.Unlock()
		}
		fx.p.soleFacade().handleHTTP(rec, r)
		return rec.ResponseRecorder, at
	}
	chat := func(model string) *http.Request {
		return httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"`+model+`"}`))
	}

	t.Run("control: a concrete model is reserved and admitted", func(t *testing.T) {
		fx := newFixture(t, true)
		rec, at := serve(fx, chat("mid:7b"))
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", rec.Code)
		}
		if at.reservations == 0 || at.active == 0 || at.activity == 0 {
			t.Fatalf("snapshot saw %+v for an admitted request, want a reservation, an active slot and activity", at)
		}
	})

	refused := func(name string, healthy bool, request func() *http.Request) {
		t.Run(name, func(t *testing.T) {
			fx := newFixture(t, healthy)
			rec, at := serve(fx, request())
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body %s", rec.Code, rec.Body)
			}
			if !at.none() {
				t.Errorf("held %+v when the refusal was written, want nothing", at)
			}
			if n := fx.local.hits(); n != 0 {
				t.Errorf("local engine received %d request(s), want none", n)
			}
			var methods []string
			for _, f := range frames(t, fx.w) {
				methods = append(methods, f.Method)
			}
			if want := []string{tc.profile.Name + ":proxy/request"}; !slices.Equal(methods, want) {
				t.Errorf("notifications = %v, want only %v (no workload or wake)", methods, want)
			}
		})
	}
	refused("healthy self", true, func() *http.Request { return chat("weak-api") })
	refused("stopped self that a request would wake", false, func() *http.Request { return chat("strong-local") })
	refused("name key", true, func() *http.Request {
		return httptest.NewRequest(http.MethodPost, "/api/pull", strings.NewReader(`{"name":"medium-api"}`))
	})
	refused("preflight carrying an alias body", true, func() *http.Request {
		r := httptest.NewRequest(http.MethodOptions, "/api/chat", strings.NewReader(`{"model":"weak-local"}`))
		r.Header.Set("Origin", "http://example.test")
		r.Header.Set("Access-Control-Request-Method", "POST")
		return r
	})
}
