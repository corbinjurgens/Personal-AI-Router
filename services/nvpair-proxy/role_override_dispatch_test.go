// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

// Exercise the listener's HTTP owner without binding a socket or using an
// installed port. The context is the one its bound-listener adapter supplies.
func roleOverrideHTTP(p *Proxy, method, path, body string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, "http://127.0.0.1:43127"+path, strings.NewReader(body))
	r = r.WithContext(context.WithValue(r.Context(), routingPortContextKey{}, 43127))
	rec := httptest.NewRecorder()
	(&routingHandler{host: p}).ServeHTTP(rec, r)
	return rec
}

func roleOverrideError(t *testing.T, rec *httptest.ResponseRecorder, status int, want string) {
	t.Helper()
	var got struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode status %d body %q: %v", rec.Code, rec.Body.String(), err)
	}
	if rec.Code != status || got.Error != want {
		t.Fatalf("response = %d %q, want %d %q", rec.Code, got.Error, status, want)
	}
}

// Reuse the existing disposable account ledger and in-memory provider
// transport. This does not bypass PublicDispatchAvailable or install a source
// usable by production. The counter proves the disabled gate precedes lookup.
func roleOverridePublicFixture(t *testing.T) (*Proxy, *chatAccountFixture, *facade, *atomic.Int32) {
	t.Helper()
	p, accounts, _, _ := egressFixture(t)
	profile, ok := profileFor("llamacpp")
	if !ok {
		t.Fatal("missing llamacpp profile")
	}
	f := newFacade(p, profile, NewDiscovery(), 0)
	p.facades = map[string]*facade{"llamacpp": f}
	lookups := &atomic.Int32{}
	publicChatInputs = func(chatAccountSnapshot) (*chatQualification, *publicClient, func(nodepolicy.CredentialRef) (string, bool), chatByteBudget, bool) {
		lookups.Add(1)
		return accounts.profile, accounts.transport, accounts.resolve, accounts.budget.acquire, true
	}
	t.Cleanup(func() { publicChatInputs = nil })
	return p, accounts, f, lookups
}

func roleOverrideNoPublicSpend(t *testing.T, accounts *chatAccountFixture, lookups *atomic.Int32) {
	t.Helper()
	store, err := readAccountStore(accounts.ledger.storePath)
	if err != nil {
		t.Fatal(err)
	}
	if lookups.Load() != 0 || accounts.calls.Load() != 0 || len(store.Holds) != 0 {
		t.Fatalf("public source lookups/calls/holds = %d/%d/%d, want zero", lookups.Load(), accounts.calls.Load(), len(store.Holds))
	}
}

func TestRoleOverrideAPIEntryRequiresEmptyDeclaration(t *testing.T) {
	p, accounts, f, lookups := roleOverridePublicFixture(t)
	pol := p.admission.currentPolicy()
	pol.Roles = nil
	pol.Tiers = map[string][]nodepolicy.TierEntry{"medium": {{Kind: nodepolicy.EntryAPI, Deployment: "chat-b"}}}
	pol.Routing.Tiers = map[string]nodepolicy.TierRouting{"medium": {RemoteAllowed: true, PaidAllowed: true}}
	p.admission.setPolicy(pol)
	egressGrant(t, p, 1, 5*time.Second)

	for _, model := range []string{"medium", "medium:api"} {
		rec := roleOverrideHTTP(p, http.MethodPost, "/v1/chat/completions", `{"model":"`+model+`","messages":[{"role":"user","content":"hi"}]}`)
		roleOverrideError(t, rec, http.StatusNotFound, "no profile or role named "+model)
	}
	catalog := roleOverrideHTTP(p, http.MethodGet, "/v1/models", "")
	if catalog.Code != http.StatusOK {
		t.Fatalf("undeclared catalog status = %d: %s", catalog.Code, catalog.Body.String())
	}
	if got := modelIDs(t, catalog); len(got) != 0 {
		t.Fatalf("API entry declared catalog names: %v", got)
	}
	roleOverrideNoPublicSpend(t, accounts, lookups)

	pol.Roles = []nodepolicy.Role{{Name: "medium", Profiles: []string{}}}
	p.admission.setPolicy(pol)
	if !f.roleRequested("medium") {
		t.Fatal("empty declaration did not become a role request")
	}
	want := []string{"medium", "medium:api", "medium:local"}
	catalog = roleOverrideHTTP(p, http.MethodGet, "/v1/models", "")
	if catalog.Code != http.StatusOK {
		t.Fatalf("declared catalog status = %d: %s", catalog.Code, catalog.Body.String())
	}
	if got := modelIDs(t, catalog); !slices.Equal(got, want) {
		t.Fatalf("declared catalog = %v, want %v", got, want)
	}
	selected, err := accounts.ledger.prepareChatAccount("chat-b")
	if err != nil {
		t.Fatal(err)
	}
	if err := p.publicPermission("medium", selected); err != nil {
		t.Fatalf("configured medium deployment lacks synthetic permission: %v", err)
	}
	other, err := accounts.ledger.prepareChatAccount("chat-a")
	if err != nil {
		t.Fatal(err)
	}
	if p.publicPermission("medium", other) == nil {
		t.Fatal("medium permission accepted the other deployment")
	}
	// The positive permission/deployment control above is not a dispatch
	// claim: the shipped gate still prevents the HTTP owner from sending.
	if nodepolicy.PublicDispatchAvailable() {
		t.Fatal("public dispatch unexpectedly available in this build")
	}
	rec := roleOverrideHTTP(p, http.MethodPost, "/v1/chat/completions", `{"model":"medium:api","messages":[{"role":"user","content":"hi"}]}`)
	roleOverrideError(t, rec, http.StatusNotFound, "no eligible model for tier medium")
	roleOverrideNoPublicSpend(t, accounts, lookups)
}

func TestRoleOverrideCoderLocalSelectsOnlyQualifiedCandidate(t *testing.T) {
	p, accounts, f, lookups := roleOverridePublicFixture(t)
	pol := p.admission.currentPolicy()
	pol.Roles = []nodepolicy.Role{{Name: "coder"}}
	pol.Tiers = map[string][]nodepolicy.TierEntry{"coder": {{Kind: nodepolicy.EntryAPI, Deployment: "chat-a"}}}
	pol.Routing.Tiers = map[string]nodepolicy.TierRouting{"coder": {RemoteAllowed: true, PaidAllowed: true}}
	p.admission.setPolicy(pol)
	egressGrant(t, p, 1, 5*time.Second)
	selected, err := accounts.ledger.prepareChatAccount("chat-a")
	if err != nil {
		t.Fatal(err)
	}
	if err := p.publicPermission("coder", selected); err != nil {
		t.Fatalf("configured coder deployment lacks synthetic permission: %v", err)
	}
	other, err := accounts.ledger.prepareChatAccount("chat-b")
	if err != nil {
		t.Fatal(err)
	}
	if p.publicPermission("coder", other) == nil {
		t.Fatal("coder permission accepted the other deployment")
	}
	qualified := newRecordingEngine(t, http.StatusOK)
	unqualified := newRecordingEngine(t, http.StatusOK)
	fx := &tierFixture{p: p, eng: map[string]*recordingEngine{"qualified": qualified, "unqualified": unqualified}}
	f.discovery.AddManual(nodeForModel(t, "qualified", qualified.URL, "qualified-profile"))
	f.discovery.AddManual(nodeForModel(t, "unqualified", unqualified.URL, "unqualified-profile"))
	qualifiedRole := qualifiedProfile("qualified-profile")
	qualifiedRole.Key = "qualified-profile"
	unqualifiedRole := qualifiedProfile("unqualified-profile")
	unqualifiedRole.Key = "unqualified-profile"
	fx.advertise(t, "qualified", map[string]nodepolicy.AdvertisedProfile{"coder": qualifiedRole}, strictCapabilityFor("llamacpp", "qualified-profile"))
	fx.advertise(t, "unqualified", map[string]nodepolicy.AdvertisedProfile{"coder": unqualifiedRole}, nil)
	p.SetPriority([]string{"unqualified", "qualified"})
	pinRecordingPairPeer(t, f, "qualified", qualified)
	pinRecordingPairPeer(t, f, "unqualified", unqualified)

	if nodepolicy.PublicDispatchAvailable() {
		t.Fatal("public dispatch unexpectedly available in this build")
	}
	rec := roleOverrideHTTP(p, http.MethodPost, "/v1/chat/completions", `{"model":"coder:api","messages":[{"role":"user","content":"hi"}],"max_tokens":16}`)
	roleOverrideError(t, rec, http.StatusNotFound, "no eligible model for tier coder")
	if qualified.hits() != 0 || unqualified.hits() != 0 {
		t.Fatalf("API override reached local candidates: qualified/unqualified = %d/%d", qualified.hits(), unqualified.hits())
	}
	roleOverrideNoPublicSpend(t, accounts, lookups)

	rec = roleOverrideHTTP(p, http.MethodPost, "/v1/chat/completions", `{"model":"coder:local","messages":[{"role":"user","content":"hi"}],"max_tokens":16}`)
	if rec.Code != http.StatusOK || qualified.hits() != 1 || unqualified.hits() != 0 {
		t.Fatalf("local override = %d %q, qualified/unqualified hits = %d/%d, want 200 and 1/0", rec.Code, rec.Body.String(), qualified.hits(), unqualified.hits())
	}
	var forwarded struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal([]byte(qualified.lastBody()), &forwarded); err != nil {
		t.Fatal(err)
	}
	if forwarded.Model != "qualified-profile" {
		t.Fatalf("forwarded model = %q, want qualified-profile", forwarded.Model)
	}
	decodeStrictHeader(t, qualified.lastHeader())
	roleOverrideNoPublicSpend(t, accounts, lookups)
}

func TestRoleOverrideProfileWithAPISuffixUsesConcretePath(t *testing.T) {
	fx := newTierFixture(t, customCase(t), nodepolicy.Default(), map[string][]string{"peer": {"x-api"}}, nil)
	rec := roleOverrideHTTP(fx.p, http.MethodPost, "/v1/chat/completions", `{"model":"x-api","messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusOK || fx.eng["peer"].hits() != 1 {
		t.Fatalf("concrete alias-shaped request = %d %q, hits = %d, want 200 and one hit", rec.Code, rec.Body.String(), fx.eng["peer"].hits())
	}
	var forwarded struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal([]byte(fx.eng["peer"].lastBody()), &forwarded); err != nil {
		t.Fatal(err)
	}
	if forwarded.Model != "x-api" || len(fx.eng["peer"].lastHeader().Values(nodepolicy.StrictAdmissionHeader)) != 0 {
		t.Fatalf("concrete request changed model or acquired strict routing: model=%q headers=%v", forwarded.Model, fx.eng["peer"].lastHeader())
	}
	test := func(name, path, body string) {
		t.Run(name, func(t *testing.T) {
			fx := newTierFixture(t, customCase(t), nodepolicy.Default(), map[string][]string{"peer": {"x:api"}}, nil)
			rec := httptest.NewRecorder()
			fx.p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, path, strings.NewReader(body)))
			var refusal aliasRefusal
			if err := json.Unmarshal(rec.Body.Bytes(), &refusal); err != nil {
				t.Fatal(err)
			}
			if rec.Code != http.StatusBadRequest || refusal.Code != "feature_unavailable" || refusal.Reason != aliasReasonNotInferenceReady || refusal.Model != "x:api" || fx.eng["peer"].hits() != 0 {
				t.Fatalf("alternate selector = %d %+v, engine hits = %d, want refusal with no hit", rec.Code, refusal, fx.eng["peer"].hits())
			}
		})
	}
	test("name member remains refused", "/v1/chat/completions", `{"model":"x:api","name":"x:api"}`)
	test("escaped name remains refused", "/v1/chat/completions", `{"model":"x:api","name":"x\u003aapi"}`)
	test("query model remains refused", "/v1/chat/completions?model=x%3Aapi", `{"model":"x:api"}`)
}

func TestRoleOverrideMixedCatalogHTTPExcludesNativeInventory(t *testing.T) {
	pol := stagedHybridPolicy(t)
	pol.Roles = []nodepolicy.Role{{Name: "deep"}, {Name: "coder"}}
	pol.Tiers = map[string][]nodepolicy.TierEntry{"unlisted": {{Kind: nodepolicy.EntryAPI, Deployment: "cloud-small"}}}
	fx := listFixture(t, pol, true)
	var nativeFetches atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		nativeFetches.Add(1)
		if r.Method != http.MethodGet || r.URL.Path != "/v1/models" {
			t.Errorf("catalog upstream received %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"z-concrete","owned_by":"engine","context_length":2048},{"id":"coder","owned_by":"engine","context_length":999},{"id":"a-concrete","owned_by":"engine","context_length":1024},{"id":"deep","owned_by":"engine"},{"id":"batch:api","owned_by":"engine"}]}`))
	}))
	t.Cleanup(upstream.Close)
	n := nodeForModel(t, "n1", upstream.URL, "peer-coder")
	n.Roles = []nodepolicy.Role{{Name: "ops"}, {Name: "batch"}, {Name: "coder", Profiles: []string{"peer-key"}}}
	n.Profiles = []nodepolicy.AdvertisedProfile{{Key: "peer-key", Engine: fx.p.soleFacade().profile.Name, Model: "peer-key", ContextTokens: 8192}}
	fx.p.soleFacade().discovery.AddManual(n)

	wantRoles := []string{"deep", "deep:api", "deep:local", "coder", "coder:api", "coder:local", "batch", "batch:api", "batch:local", "ops", "ops:api", "ops:local"}
	wantCatalog := append(wantRoles, "peer-key")
	assert := func(t *testing.T, rec *httptest.ResponseRecorder, want []string) {
		t.Helper()
		if rec.Code != http.StatusOK {
			t.Fatalf("catalog status = %d, body = %q", rec.Code, rec.Body.String())
		}
		var out struct {
			Object string `json:"object"`
			Data   []struct {
				ID      string `json:"id"`
				Owner   string `json:"owned_by"`
				Context *int   `json:"context_length"`
			} `json:"data"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatal(err)
		}
		var ids []string
		for _, item := range out.Data {
			ids = append(ids, item.ID)
			wantContext := 0
			wantOwner := "pair"
			switch item.ID {
			case "coder", "coder:api", "coder:local", "peer-key":
				wantContext = 8192
			}
			if item.Owner != wantOwner || (wantContext == 0 && item.Context != nil) || (wantContext != 0 && (item.Context == nil || *item.Context != wantContext)) {
				t.Fatalf("catalog record = %+v, want owner %q and context %d (zero means omitted)", item, wantOwner, wantContext)
			}
		}
		if out.Object != "list" || !slices.Equal(ids, want) {
			t.Fatalf("catalog = %q %v, want list %v", out.Object, ids, want)
		}
	}
	t.Run("facade lists profiles and roles", func(t *testing.T) {
		assert(t, fx.listModels(t, "/v1/models"), wantCatalog)
	})
	t.Run("listener lists own order then sorted empty peer roles", func(t *testing.T) {
		assert(t, roleOverrideHTTP(fx.p, http.MethodGet, "/v1/models", ""), wantCatalog)
	})
	if nativeFetches.Load() != 0 {
		t.Fatalf("catalog fetched native inventory %d times", nativeFetches.Load())
	}
}
