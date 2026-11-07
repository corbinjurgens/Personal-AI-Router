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

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

func TestModelPathAdmissionIncludesProtocolRegistry(t *testing.T) {
	original := protocolRoutes[engines.ProtocolOpenAIV1]
	protocolRoutes[engines.ProtocolOpenAIV1] = append(slices.Clone(original), route{Path: "/future/registered-api", Role: roleInferencePOST})
	t.Cleanup(func() { protocolRoutes[engines.ProtocolOpenAIV1] = original })
	profile := strataProfile(t) // Declares only chat, not the extended protocol.
	if !isInferenceRequest(profile, http.MethodPost, "/future/registered-api") {
		t.Fatal("new protocol route escaped the shared inference classifier")
	}
	if isInferenceRequest(profile, http.MethodGet, "/future/registered-api") {
		t.Fatal("inference classification ignored the route's method")
	}
}

// Explicit aliases pin the known surface; registry-derived cases also cover
// routes introduced by another protocol or descriptor without editing this test.
func TestModelPathAdmission(t *testing.T) {
	paths := []string{
		"/completion", "/completions", "/chat/completions", "/infill",
		"/embedding", "/embeddings", "/rerank", "/reranking",
		"/v1/rerank", "/v1/reranking", "/responses", "/v1/responses", "/v1/messages",
		"/tokenize", "/detokenize", "/apply-template", "/tts",
		"/chat/completions/input_tokens", "/v1/chat/completions/input_tokens",
		"/responses/input_tokens", "/v1/responses/input_tokens", "/v1/messages/count_tokens",
		"/v1/chat/completions/control", "/v1/systemone",
		"/props", "/v1/load", "/v1/unload", "/slots/0", "/lora-adapters",
		"/future/model-api", // No route declaration: the body still names a model.
	}
	for _, routes := range protocolRoutes {
		for _, route := range routes {
			if route.Role == roleInferencePOST && !slices.Contains(paths, route.Path) {
				paths = append(paths, route.Path)
			}
		}
	}
	slices.Sort(paths)
	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			for _, scenario := range []struct {
				name    string
				model   string
				ingress bool
				want    int
			}{
				{name: "local unprofiled", model: "native-file", want: http.StatusNotFound},
				{name: "peer unprofiled", model: "native-file", ingress: true, want: http.StatusNotFound},
				{name: "local profile", model: "public-profile", want: http.StatusOK},
				{name: "local role", model: "coder", want: http.StatusOK},
				{name: "peer profile", model: "public-profile", ingress: true, want: http.StatusOK},
			} {
				t.Run(scenario.name, func(t *testing.T) {
					pol := nodepolicy.Default()
					pol.Profiles = []nodepolicy.Profile{{Key: "public-profile", DisplayName: "Public", Engine: "llamacpp", Model: "native-file", Capabilities: []nodepolicy.Capability{nodepolicy.CapEmbeddings}}}
					pol.Roles = []nodepolicy.Role{{Name: "coder", Profiles: []string{"public-profile"}}}
					fx := newTierFixture(t, llamacppCase(t), pol, nil, nil)
					f := fx.p.soleFacade()
					e := newRecordingEngineForRequest(t, http.StatusOK, http.MethodPost, path)
					setBackend(t, f, e.URL, true)
					f.discovery.AddManual(selfNode("self", f.selfPort(), "native-file"))
					r := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"model":"`+scenario.model+`","prompt":"test"}`))
					w := httptest.NewRecorder()
					if scenario.ingress {
						f.serveIngress(w, r, "peer")
					} else {
						f.handleHTTP(w, r)
					}
					if w.Code != scenario.want {
						t.Fatalf("status=%d want=%d hits=%d body=%s", w.Code, scenario.want, e.hits(), w.Body)
					}
					if scenario.want == http.StatusNotFound {
						var refusal struct {
							Error string `json:"error"`
						}
						if err := json.Unmarshal(w.Body.Bytes(), &refusal); err != nil {
							t.Fatal(err)
						}
						if refusal.Error != "no profile or role named native-file" || e.hits() != 0 {
							t.Fatalf("refusal=%+v hits=%d", refusal, e.hits())
						}
					} else if e.hits() != 1 {
						t.Fatalf("admitted hits=%d", e.hits())
					}
					if fx.p.buffers != nil && preparationUsed(fx.p.buffers) != 0 {
						t.Fatal("request retained inference buffer capacity")
					}
				})
			}
		})
	}
}
