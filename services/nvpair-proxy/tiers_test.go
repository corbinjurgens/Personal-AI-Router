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

func TestParseRequestNeeds(t *testing.T) {
	caps := func(body, path string) []string {
		return parseRequestNeeds([]byte(body), path).names()
	}
	cases := []struct {
		name, body, path string
		want             []string
	}{
		{"plain chat", `{"model":"weak","messages":[{"role":"user","content":"hi"}]}`, "/v1/chat/completions", nil},
		{"openai tools", `{"model":"weak","tools":[{"type":"function"}]}`, "/v1/chat/completions", []string{"tools"}},
		{"empty tools", `{"model":"weak","tools":[]}`, "/v1/chat/completions", nil},
		{"legacy functions", `{"model":"weak","functions":[{"name":"f"}]}`, "/v1/chat/completions", []string{"tools"}},
		{"openai image part", `{"model":"weak","messages":[{"role":"user","content":[{"type":"text","text":"x"},{"type":"image_url","image_url":{"url":"data:"}}]}]}`, "/v1/chat/completions", []string{"vision"}},
		{"ollama message images", `{"model":"weak","messages":[{"role":"user","content":"x","images":["aGk="]}]}`, "/api/chat", []string{"vision"}},
		{"ollama generate images", `{"model":"weak","prompt":"x","images":["aGk="]}`, "/api/generate", []string{"vision"}},
		{"json schema", `{"model":"weak","response_format":{"type":"json_schema","json_schema":{}}}`, "/v1/chat/completions", []string{"structured"}},
		{"json object", `{"model":"weak","response_format":{"type":"json_object"}}`, "/v1/chat/completions", []string{"structured"}},
		{"text format", `{"model":"weak","response_format":{"type":"text"}}`, "/v1/chat/completions", nil},
		{"ollama format", `{"model":"weak","format":"json"}`, "/api/chat", []string{"structured"}},
		{"ollama schema", `{"model":"weak","format":{"type":"object"}}`, "/api/chat", []string{"structured"}},
		{"embeddings route", `{"model":"weak","input":"x"}`, "/v1/embeddings", []string{"embeddings"}},
		{"ollama embed", `{"model":"weak","input":"x"}`, "/api/embed", []string{"embeddings"}},
		{"everything", `{"model":"weak","tools":[{}],"format":"json","messages":[{"images":["x"]}]}`, "/api/chat", []string{"tools", "vision", "structured"}},
	}
	for _, c := range cases {
		if got := caps(c.body, c.path); !slices.Equal(got, c.want) {
			t.Errorf("%s: needs = %v, want %v", c.name, got, c.want)
		}
	}

	body := `{"model":"weak","max_tokens":1000}`
	if got := parseRequestNeeds([]byte(body), "/v1/completions").contextTokens; got != len(body)/4+1000 {
		t.Errorf("context with max_tokens = %d, want %d", got, len(body)/4+1000)
	}
	body = `{"model":"weak","max_completion_tokens":50}`
	if got := parseRequestNeeds([]byte(body), "/v1/chat/completions").contextTokens; got != len(body)/4+50 {
		t.Errorf("context with max_completion_tokens = %d", got)
	}
	body = `{"model":"weak","options":{"num_predict":300}}`
	if got := parseRequestNeeds([]byte(body), "/api/generate").contextTokens; got != len(body)/4+300 {
		t.Errorf("context with num_predict = %d", got)
	}
	body = `{"model":"weak","options":{"num_predict":-1}}`
	if got := parseRequestNeeds([]byte(body), "/api/generate").contextTokens; got != len(body)/4 {
		t.Errorf("an unbounded num_predict must not inflate the estimate: %d", got)
	}
}

func TestTierSearchOrder(t *testing.T) {
	cases := []struct {
		tier             string
		stronger, weaker bool
		want             []string
	}{
		{"weak", false, false, []string{"weak"}},
		{"weak", true, false, []string{"weak", "medium", "strong"}},
		{"medium", true, true, []string{"medium", "strong", "weak"}},
		{"strong", false, true, []string{"strong", "medium", "weak"}},
		{"strong", true, false, []string{"strong"}},
	}
	for _, c := range cases {
		got := tierSearchOrder(c.tier, nodepolicy.TierPolicy{AllowStronger: c.stronger, AllowWeaker: c.weaker})
		if !slices.Equal(got, c.want) {
			t.Errorf("%s stronger=%v weaker=%v: %v, want %v", c.tier, c.stronger, c.weaker, got, c.want)
		}
	}
}

func TestReplaceModelPreservesEveryOtherByte(t *testing.T) {
	body := `{ "stream" :true,"model" : "weak" ,"messages":[{"content":"model: \"x\" é"}],"n":1.50e2 }` + "\n"
	got, ok := replaceModel([]byte(body), "qwen3:8b")
	if !ok {
		t.Fatal("replaceModel refused a valid body")
	}
	want := `{ "stream" :true,"model" : "qwen3:8b" ,"messages":[{"content":"model: \"x\" é"}],"n":1.50e2 }` + "\n"
	if string(got) != want {
		t.Fatalf("got\n%s\nwant\n%s", got, want)
	}
	// Only the top-level model is replaced.
	nested := `{"meta":{"model":"weak"},"model":"weak"}`
	got, _ = replaceModel([]byte(nested), "m")
	if string(got) != `{"meta":{"model":"weak"},"model":"m"}` {
		t.Fatalf("nested model touched: %s", got)
	}
	for _, bad := range []string{``, `[]`, `{"model":"weak"} trailing`, `{"stream":true}`, `{"model":`} {
		if _, ok := replaceModel([]byte(bad), "m"); ok {
			t.Errorf("replaceModel accepted %q", bad)
		}
	}
}

func TestMergeDefaults(t *testing.T) {
	cases := []struct{ body, defaults, want string }{
		{`{"model":"m"}`, `{"keep_alive":"10m"}`, `{"model":"m","keep_alive":"10m"}`},
		{`{"model":"m","keep_alive":"1m"}`, `{"keep_alive":"10m"}`, `{"model":"m","keep_alive":"1m"}`},
		{`{"model":"m","options":{"temperature":0}}`, `{"options":{"num_ctx":4096,"temperature":1}}`, `{"model":"m","options":{"temperature":0,"num_ctx":4096}}`},
		{`{"model":"m","options":null}`, `{"options":{"num_ctx":4096}}`, `{"model":"m","options":null}`},
		{`{}`, `{"a":1,"b":[1,2]}`, `{"a":1,"b":[1,2]}`},
		{`{"model":"m"}`, ``, `{"model":"m"}`},
		{`not json`, `{"a":1}`, `not json`},
	}
	for _, c := range cases {
		if got := string(mergeDefaults([]byte(c.body), []byte(c.defaults))); got != c.want {
			t.Errorf("merge(%s, %s) = %s, want %s", c.body, c.defaults, got, c.want)
		}
	}
}

// tierFixture is one facade with manual peers and a policy.
type tierFixture struct {
	p   *Proxy
	w   *recordingWriter
	eng map[string]*recordingEngine
}

func newTierFixture(t *testing.T, tc engineCase, pol nodepolicy.Policy, nodes map[string][]string, loaded map[string][]string) *tierFixture {
	t.Helper()
	fx := &tierFixture{w: &recordingWriter{}, eng: map[string]*recordingEngine{}}
	disc := NewDiscovery()
	ids := make([]string, 0, len(nodes))
	for id, models := range nodes {
		e := newRecordingEngine(t, http.StatusOK)
		fx.eng[id] = e
		n := nodeFor(t, id, e.URL)
		n.Models = models
		n.Loaded = loaded[id]
		disc.AddManual(n)
		ids = append(ids, id)
	}
	slices.Sort(ids)
	fx.p = newTestProxy(tc.profile, NewCodec(fx.w), disc, tc.profile.FacadePort)
	fx.p.SetPriority(ids)
	if err := pol.Validate(); err != nil {
		t.Fatal(err)
	}
	fx.p.admission.setPolicy(pol)
	return fx
}

func (fx *tierFixture) do(t *testing.T, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	fx.p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, path, strings.NewReader(body)))
	return rec
}

func (fx *tierFixture) servedBy(t *testing.T) string {
	t.Helper()
	for id, e := range fx.eng {
		if e.hits() > 0 {
			return id
		}
	}
	return ""
}

func ollamaTierPolicy(tiers map[string][]nodepolicy.TierEntry) nodepolicy.Policy {
	pol := nodepolicy.Default()
	pol.Tiers = tiers
	return pol
}

func TestTier_CapabilityFilterAndBodyRewrite(t *testing.T) {
	tc := ollamaCase(t)
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{
		"weak": {
			{Engine: "ollama", Model: "small:1b"},
			{Engine: "ollama", Model: "small-tools:1b", Capabilities: []nodepolicy.Capability{nodepolicy.CapTools}},
		},
	})
	fx := newTierFixture(t, tc, pol, map[string][]string{
		"a": {"small:1b"}, "b": {"small-tools:1b"},
	}, nil)

	body := `{"model":"weak","tools":[{"type":"function"}],"temperature":0.2}`
	rec := fx.do(t, "/v1/chat/completions", body)
	if rec.Code != http.StatusOK || fx.servedBy(t) != "b" {
		t.Fatalf("status %d, served by %q, want b (the only tools model)", rec.Code, fx.servedBy(t))
	}
	if got := fx.eng["b"].lastBody(); got != `{"model":"small-tools:1b","tools":[{"type":"function"}],"temperature":0.2}` {
		t.Fatalf("engine got %s", got)
	}
	// Headers name what answered.
	for h, want := range map[string]string{
		nodepolicy.ModelHeader: "small-tools:1b", nodepolicy.EngineHeader: "ollama",
		nodepolicy.NodeHeader: "b", nodepolicy.TierHeader: "weak",
	} {
		if got := rec.Header().Get(h); got != want {
			t.Errorf("%s = %q, want %q", h, got, want)
		}
	}
}

func TestTier_ContextFilterAndStrongerFallback(t *testing.T) {
	tc := ollamaCase(t)
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{
		"weak":   {{Engine: "ollama", Model: "small:1b", ContextTokens: 100}},
		"medium": {{Engine: "ollama", Model: "mid:7b", ContextTokens: 32000}},
	})
	fx := newTierFixture(t, tc, pol, map[string][]string{"a": {"small:1b"}, "b": {"mid:7b"}}, nil)

	// Small enough for weak.
	if rec := fx.do(t, "/v1/chat/completions", `{"model":"weak"}`); rec.Code != http.StatusOK || fx.servedBy(t) != "a" {
		t.Fatalf("short request: status %d served by %q, want a", rec.Code, fx.servedBy(t))
	}
	// Too long for weak's context: falls to the stronger tier.
	rec := fx.do(t, "/v1/chat/completions", `{"model":"weak","max_tokens":4000}`)
	if rec.Code != http.StatusOK || fx.eng["b"].hits() != 1 {
		t.Fatalf("long request: status %d, b hits %d", rec.Code, fx.eng["b"].hits())
	}
	if rec.Header().Get(nodepolicy.TierHeader) != "weak" || rec.Header().Get(nodepolicy.ModelHeader) != "mid:7b" {
		t.Fatalf("headers %v", rec.Header())
	}

	// Without allowStronger nothing is eligible: 404 naming the tier.
	pol.TierPolicy.AllowStronger = false
	fx.p.admission.setPolicy(pol)
	rec = fx.do(t, "/v1/chat/completions", `{"model":"weak","max_tokens":4000,"tools":[{}]}`)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", rec.Code)
	}
	var out struct {
		Error string   `json:"error"`
		Needs []string `json:"needs"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Error != "no eligible model for tier weak" || !slices.Equal(out.Needs, []string{"tools"}) {
		t.Fatalf("404 body = %s", rec.Body)
	}
}

func TestTier_NoEntriesIs404WithEmptyNeeds(t *testing.T) {
	tc := ollamaCase(t)
	fx := newTierFixture(t, tc, nodepolicy.Default(), map[string][]string{"a": {"small:1b"}}, nil)
	rec := fx.do(t, "/api/chat", `{"model":"strong"}`)
	if rec.Code != http.StatusNotFound || !strings.Contains(rec.Body.String(), `"needs":[]`) {
		t.Fatalf("status %d body %s", rec.Code, rec.Body)
	}
	if fx.servedBy(t) != "" {
		t.Fatal("an ineligible tier request reached an engine")
	}
}

func TestTier_WarmPreference(t *testing.T) {
	tc := ollamaCase(t)
	tiers := map[string][]nodepolicy.TierEntry{
		"weak":   {{Engine: "ollama", Model: "cold:1b"}, {Engine: "ollama", Model: "warm:1b"}},
		"medium": {{Engine: "ollama", Model: "mid:7b"}},
	}
	nodes := map[string][]string{"a": {"cold:1b"}, "b": {"warm:1b"}, "c": {"mid:7b"}}

	// Within the requested tier a loaded model leads, ahead of entry order.
	fx := newTierFixture(t, tc, ollamaTierPolicy(tiers), nodes, map[string][]string{"b": {"warm:1b"}, "c": {"mid:7b"}})
	if rec := fx.do(t, "/v1/chat/completions", `{"model":"weak"}`); rec.Code != http.StatusOK || fx.servedBy(t) != "b" {
		t.Fatalf("tier mode: served by %q, want warm b", fx.servedBy(t))
	}

	// "tier" keeps tier order first: a warm stronger model never beats a cold
	// model of the requested tier.
	fx = newTierFixture(t, tc, ollamaTierPolicy(tiers), nodes, map[string][]string{"c": {"mid:7b"}})
	if rec := fx.do(t, "/v1/chat/completions", `{"model":"weak"}`); rec.Code != http.StatusOK || fx.servedBy(t) != "a" {
		t.Fatalf("tier mode, only medium warm: served by %q, want a", fx.servedBy(t))
	}

	// "any" prefers any loaded eligible model.
	pol := ollamaTierPolicy(tiers)
	pol.TierPolicy.PreferLoaded = "any"
	fx = newTierFixture(t, tc, pol, nodes, map[string][]string{"c": {"mid:7b"}})
	rec := fx.do(t, "/v1/chat/completions", `{"model":"weak"}`)
	if rec.Code != http.StatusOK || fx.servedBy(t) != "c" {
		t.Fatalf("any mode: served by %q, want warm c", fx.servedBy(t))
	}
}

// Workload events carry the concrete model and the tier asked for.
func TestTier_WorkloadCarriesRequestedModel(t *testing.T) {
	tc := ollamaCase(t)
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"medium": {{Engine: "ollama", Model: "mid:7b"}}})
	fx := newTierFixture(t, tc, pol, map[string][]string{"a": {"mid:7b"}}, nil)
	if rec := fx.do(t, "/api/generate", `{"model":"medium","prompt":"x"}`); rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
	events := notificationsOf(t, fx.w, workloadStartedMethod)
	events = append(events, notificationsOf(t, fx.w, workloadSubmittedMethod)...)
	events = append(events, notificationsOf(t, fx.w, workloadCompletedMethod)...)
	if len(events) < 3 {
		t.Fatalf("got %d workload events", len(events))
	}
	for _, raw := range events {
		var params workloadParams
		if err := json.Unmarshal(raw, &params); err != nil {
			t.Fatal(err)
		}
		if params.WorkloadInfo.Model != "mid:7b" || params.WorkloadInfo.RequestedModel != "medium" {
			t.Fatalf("workload model=%q requestedModel=%q, want mid:7b / medium", params.WorkloadInfo.Model, params.WorkloadInfo.RequestedModel)
		}
	}

	// A request that names its model carries no requestedModel at all.
	fx2 := newTierFixture(t, tc, pol, map[string][]string{"a": {"mid:7b"}}, nil)
	fx2.do(t, "/api/generate", `{"model":"mid:7b"}`)
	for _, line := range fx2.w.lines() {
		if strings.Contains(string(line), "requestedModel") {
			t.Fatalf("requestedModel on a non-tier workload: %s", line)
		}
	}
}

// Routing headers are present on an ordinary routed response too.
func TestHandleHTTP_RoutingHeadersOnPlainRequest(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		e := newRecordingEngine(t, http.StatusOK)
		disc := NewDiscovery()
		disc.AddManual(nodeForModel(t, "n1", e.URL, tc.advertisedModel))
		p := testProxy(tc.profile, disc, tc.profile.FacadePort)
		rec := httptest.NewRecorder()
		p.soleFacade().handleHTTP(rec, tc.inferenceRequest())
		if rec.Header().Get(nodepolicy.ModelHeader) != tc.requestedModel ||
			rec.Header().Get(nodepolicy.EngineHeader) != tc.profile.Name ||
			rec.Header().Get(nodepolicy.NodeHeader) != "n1" ||
			rec.Header().Get(nodepolicy.TierHeader) != "" {
			t.Fatalf("headers = %v", rec.Header())
		}
	})
}

// twoEngineProxy hosts an Ollama and an LM Studio facade in one process, the
// way production does, without binding either.
func twoEngineProxy(t *testing.T, w *recordingWriter) (*Proxy, *facade, *facade) {
	t.Helper()
	oll, lms := ollamaCase(t).profile, lmstudioCase(t).profile
	p := newTestProxy(oll, NewCodec(w), NewDiscovery(), oll.FacadePort)
	lm := newFacade(p, lms, NewDiscovery(), lms.FacadePort)
	p.facades[lms.Name] = lm
	return p, p.facades[oll.Name], lm
}

// An OpenAI-compatible tier request entering through one engine's facade may
// be served by another engine on this node: the self candidate targets that
// engine's local backend, and admission is keyed to that engine.
func TestTier_CrossEngineSelfRouting(t *testing.T) {
	w := &recordingWriter{}
	p, oll, lm := twoEngineProxy(t, w)
	lmEngine := newRecordingEngine(t, http.StatusOK)
	ollEngine := newRecordingEngine(t, http.StatusOK)
	lm.discovery.AddManual(selfNode("self", lm.profile.FacadePort, "qwen3-8b"))
	setBackend(t, lm, lmEngine.URL, true)
	oll.discovery.AddManual(selfNode("self", oll.profile.FacadePort, "llama3:latest"))
	setBackend(t, oll, ollEngine.URL, true)

	pol := nodepolicy.Default()
	pol.Tiers = map[string][]nodepolicy.TierEntry{"strong": {{Engine: "lmstudio", Model: "qwen3-8b"}}}
	p.admission.setPolicy(pol)

	rec := httptest.NewRecorder()
	oll.handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"strong","messages":[]}`)))
	if rec.Code != http.StatusOK || lmEngine.hits() != 1 || ollEngine.hits() != 0 {
		t.Fatalf("status %d, lmstudio hits %d, ollama hits %d", rec.Code, lmEngine.hits(), ollEngine.hits())
	}
	if got := lmEngine.lastBody(); got != `{"model":"qwen3-8b","messages":[]}` {
		t.Fatalf("lmstudio engine got %s", got)
	}
	if rec.Header().Get(nodepolicy.EngineHeader) != "lmstudio" {
		t.Fatalf("%s = %q, want lmstudio", nodepolicy.EngineHeader, rec.Header().Get(nodepolicy.EngineHeader))
	}

	// Admission is the other engine's: draining LM Studio refuses it.
	p.admission.setDrain("lmstudio", true)
	rec = httptest.NewRecorder()
	oll.handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"strong"}`)))
	assertAdmissionRejection(t, rec, nodepolicy.RejectDraining)

	// A native route never crosses engines.
	p.admission.setDrain("lmstudio", false)
	rec = httptest.NewRecorder()
	oll.handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"strong"}`)))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("native route crossed engines: status %d", rec.Code)
	}
}

// A peer candidate on another engine targets that peer's facade for that
// engine: the port the other facade's discovery overlay carries.
func TestTier_CrossEnginePeerUsesThatEnginesPort(t *testing.T) {
	w := &recordingWriter{}
	p, oll, lm := twoEngineProxy(t, w)
	peerLM := newRecordingEngine(t, http.StatusOK)
	lm.discovery.AddManual(nodeForModel(t, "peer", peerLM.URL, "qwen3-8b"))
	pol := nodepolicy.Default()
	pol.Tiers = map[string][]nodepolicy.TierEntry{"weak": {{Engine: "lmstudio", Model: "qwen3-8b"}}}
	p.admission.setPolicy(pol)

	cands := oll.resolveTierCandidates("weak", requestNeeds{}, true)
	if len(cands) != 1 || cands[0].engine != "lmstudio" || cands[0].url.Host != strings.TrimPrefix(peerLM.URL, "http://") {
		t.Fatalf("candidates = %+v", cands)
	}
	if cands[0].fac != lm {
		t.Fatal("a cross-engine candidate must be owned by its engine's facade")
	}
}

func TestModelList_OffersTiers(t *testing.T) {
	tc := lmstudioCase(t)
	e := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"qwen3-8b"}]}`))
	}))
	defer e.Close()
	disc := NewDiscovery()
	disc.AddManual(nodeFor(t, "n1", e.URL))
	p := testProxy(tc.profile, disc, tc.profile.FacadePort)
	pol := nodepolicy.Default()
	pol.Tiers = map[string][]nodepolicy.TierEntry{
		"weak":   {{Engine: "lmstudio", Model: "qwen3-8b"}},
		"strong": {{Engine: "lmstudio", Model: "qwen3-8b"}},
	}
	p.admission.setPolicy(pol)
	rec := httptest.NewRecorder()
	p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	var out struct {
		Data []struct {
			ID      string `json:"id"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, d := range out.Data {
		ids = append(ids, d.ID+"/"+d.OwnedBy)
	}
	if !slices.Equal(ids, []string{"qwen3-8b/", "weak/pair", "strong/pair"}) {
		t.Fatalf("model list = %v", ids)
	}
}

func TestModelList_NativeShapeGetsNoTiers(t *testing.T) {
	tc := ollamaCase(t)
	e := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"models":[{"name":"qwen3:latest","model":"qwen3:latest"}]}`))
	}))
	defer e.Close()
	disc := NewDiscovery()
	disc.AddManual(nodeFor(t, "n1", e.URL))
	p := testProxy(tc.profile, disc, tc.profile.FacadePort)
	pol := nodepolicy.Default()
	pol.Tiers = map[string][]nodepolicy.TierEntry{"weak": {{Engine: "ollama", Model: "qwen3"}}}
	p.admission.setPolicy(pol)
	rec := httptest.NewRecorder()
	p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/tags", nil))
	if strings.Contains(rec.Body.String(), "weak") {
		t.Fatalf("native model list gained a tier: %s", rec.Body)
	}
}
