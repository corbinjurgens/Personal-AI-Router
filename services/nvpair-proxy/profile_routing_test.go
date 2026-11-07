// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

func TestProfileRoutingSelfKey(t *testing.T) {
	pol := nodepolicy.Default()
	pol.Profiles = []nodepolicy.Profile{{Key: "my-profile", DisplayName: "My profile", Engine: "llamacpp", Model: "native-file"}}
	fx := newTierFixture(t, llamacppCase(t), pol, nil, nil)
	f := fx.p.soleFacade()
	e := newRecordingEngineForRequest(t, http.StatusOK, http.MethodPost, routeChatCompletions)
	setBackend(t, f, e.URL, true)
	f.discovery.AddManual(nodeFor(t, "self", "http://127.0.0.1:"+strconv.Itoa(f.selfPort())))
	w := fx.do(t, routeChatCompletions, `{"model":"my-profile"}`)
	if w.Code != http.StatusOK || e.hits() != 1 || w.Header().Get(nodepolicy.ModelHeader) != "my-profile" {
		t.Fatalf("self profile status=%d hits=%d served=%q body=%s", w.Code, e.hits(), w.Header().Get(nodepolicy.ModelHeader), w.Body)
	}
	var sent struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal([]byte(e.lastBody()), &sent); err != nil {
		t.Fatal(err)
	}
	if sent.Model != "my-profile" {
		t.Fatalf("served id = %q", sent.Model)
	}
}

func TestProfileRoutingPeerOutsideRoles(t *testing.T) {
	fx := newTierFixture(t, llamacppCase(t), nodepolicy.Default(), map[string][]string{"peer": {"native-file"}}, nil)
	f := fx.p.soleFacade()
	n := f.discovery.Nodes()[0]
	n.Profiles = []nodepolicy.AdvertisedProfile{{Key: "peer-profile", Engine: "llamacpp", Model: "native-file"}}
	f.discovery.AddManual(n)
	w := fx.do(t, routeChatCompletions, `{"model":"peer-profile"}`)
	if w.Code != http.StatusOK || fx.eng["peer"].hits() != 1 || w.Header().Get(nodepolicy.ModelHeader) != "peer-profile" {
		t.Fatalf("unlisted peer profile status=%d hits=%d served=%q body=%s", w.Code, fx.eng["peer"].hits(), w.Header().Get(nodepolicy.ModelHeader), w.Body)
	}
}

func TestProfileRoutingRefusesUnprofiledNames(t *testing.T) {
	for _, name := range []string{"native-file", "ollama/example@latest"} {
		t.Run(name, func(t *testing.T) {
			fx := newTierFixture(t, llamacppCase(t), nodepolicy.Default(), map[string][]string{"peer": {name}}, nil)
			n := fx.p.soleFacade().discovery.Nodes()[0]
			n.Profiles = nil
			fx.p.soleFacade().discovery.AddManual(n)
			body, err := json.Marshal(map[string]string{"model": name})
			if err != nil {
				t.Fatal(err)
			}
			w := fx.do(t, routeChatCompletions, string(body))
			var refusal struct {
				Error string `json:"error"`
			}
			if err := json.Unmarshal(w.Body.Bytes(), &refusal); err != nil {
				t.Fatal(err)
			}
			if w.Code != http.StatusNotFound || refusal.Error != "no profile or role named "+name || fx.eng["peer"].hits() != 0 {
				t.Fatalf("unprofiled refusal status=%d body=%+v hits=%d", w.Code, refusal, fx.eng["peer"].hits())
			}
		})
	}
}

func TestProfileRoutingCatalogs(t *testing.T) {
	pol := nodepolicy.Default()
	pol.Roles = []nodepolicy.Role{{Name: "coder"}}
	pol.Profiles = []nodepolicy.Profile{{Key: "local-profile", DisplayName: "Local", Engine: "llamacpp", Model: "local-native", LocalOnly: true}}
	fx := newTierFixture(t, llamacppCase(t), pol, map[string][]string{"peer": {"peer-native"}}, nil)
	f := fx.p.soleFacade()
	n := f.discovery.Nodes()[0]
	n.Profiles = []nodepolicy.AdvertisedProfile{{Key: "peer-profile", Engine: "llamacpp", Model: "peer-native"}}
	f.discovery.AddManual(n)
	check := func(t *testing.T, w *httptest.ResponseRecorder) {
		t.Helper()
		var catalog struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &catalog); err != nil {
			t.Fatal(err)
		}
		var ids []string
		for _, row := range catalog.Data {
			ids = append(ids, row.ID)
		}
		want := []string{"coder", "coder:api", "coder:local", "local-profile", "peer-profile"}
		if w.Code != http.StatusOK || !slices.Equal(ids, want) {
			t.Fatalf("catalog status=%d ids=%v, want %v", w.Code, ids, want)
		}
	}
	t.Run("facade", func(t *testing.T) {
		w := httptest.NewRecorder()
		f.handleHTTP(w, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
		check(t, w)
	})
	t.Run("routing listener", func(t *testing.T) {
		w := httptest.NewRecorder()
		h := routingHandler{host: fx.p}
		h.listModels(w)
		check(t, w)
	})
}

func TestProfileRoutingAdmissionRequiresProfileBeforeEffects(t *testing.T) {
	h := newAdmHarness(t)
	prepared := false
	d := h.a.admit(context.Background(), admissionRequest{engine: "llamacpp", model: "native-file", prepare: func(context.Context, nodepolicy.Profile, bool, uint64) error { prepared = true; return nil }})
	if d.ticket != nil {
		d.ticket.release()
		t.Fatal("unprofiled request admitted")
	}
	if d.reject != rejectNotServed || prepared || h.a.currentActive() != 0 {
		t.Fatalf("decision=%+v prepared=%v active=%d", d, prepared, h.a.currentActive())
	}
}

func TestProfileRoutingSharesStrictModelLimits(t *testing.T) {
	h := newStrictHarness(t)
	pol := h.a.currentPolicy()
	pol.Profiles[0].Key = "public-profile"
	pol.Profiles[0].MaxConcurrent = 1
	pol.Admission.MaxConcurrentPerModel = 4
	h.a.setPolicy(pol)
	capability := h.a.StrictReadiness().Capability
	if capability == nil || len(capability.Models) != 1 || capability.Models[0].Model != "m" || capability.Models[0].MaxConcurrent != 1 {
		t.Fatalf("profile key changed strict model bounds: %+v", capability)
	}
	ticket, err := h.a.admitStrictNow(context.Background(), "llamacpp", "m", strictRequest())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket.Finish(strictNotDispatched)
	mustReject(t, h.admit("llamacpp", "public-profile", 0), nodepolicy.RejectBusy)
}

func TestProfileRoutingRetainsModelMemoryBound(t *testing.T) {
	h := newAdmHarness(t)
	pol := nodepolicy.Default()
	pol.Profiles = []nodepolicy.Profile{{Key: "public-profile", DisplayName: "Public profile", Engine: "llamacpp", Model: "native-model", MemoryBytes: 1024}}
	pol.Admission.MemoryBudgetBytes = 512
	h.a.setPolicy(pol)
	mustReject(t, h.admit("llamacpp", "public-profile", 0), nodepolicy.RejectNoFit)
}

func TestProfileRoutingQualifiedRoleUsesStrictModelIdentity(t *testing.T) {
	fx := newReadinessEnvelopeFixture(t, "llamacpp")
	fx.policy.Profiles[0].Key = "public-profile"
	fx.policy.Roles = []nodepolicy.Role{{Name: "coder", Profiles: []string{"public-profile"}}}
	fx.install(t)
	p := testProxy(llamacppCase(t).profile, NewDiscovery(), 11435)
	p.admission = fx.a
	f := p.soleFacade()
	e := newRecordingEngine(t, http.StatusOK)
	setBackend(t, f, e.URL, true)
	f.discovery.AddManual(selfNode("self", f.selfPort(), "public-profile"))
	candidates := f.hybridLocalCandidates("coder", requestNeeds{}, false)
	if len(candidates) != 1 || candidates[0].model != "m" || candidates[0].strict.Model != "m" || candidates[0].role.Key != "public-profile" {
		t.Fatalf("qualified profile key lost strict model identity: %+v", candidates)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	ticket, err := fx.a.admitStrictNow(ctx, "llamacpp", candidates[0].model, fx.request())
	if err != nil {
		t.Fatal(err)
	}
	ticket.Finish(strictNotDispatched)
}
