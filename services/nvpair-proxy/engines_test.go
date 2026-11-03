// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net/http"
	"slices"
	"testing"

	"nvpair-shared/engines"
)

func TestProfilesMatchSharedEngines(t *testing.T) {
	names := make([]string, len(profiles))
	for i, profile := range profiles {
		names[i] = profile.Name
	}
	if !slices.Equal(names, engines.Names()) {
		t.Fatalf("proxy profiles = %v, want canonical engines %v", names, engines.Names())
	}
}

func TestNormalizeModelPreservesNamingAndAliases(t *testing.T) {
	test := func(name, input, want string, naming modelNaming) {
		t.Run(name, func(t *testing.T) {
			profile := engineProfile{ModelNaming: naming}
			if got := profile.normalizeModel(input); got != want {
				t.Fatalf("normalizeModel(%q) = %q, want %q", input, got, want)
			}
		})
	}
	test("reserved weak alias", " weak ", "weak", exactID)
	test("reserved strong alias", "strong", "strong", exactID)
	test("registry name", " registry:5000/org/model ", "registry:5000/org/model", exactID)
	test("explicit tag", "org/model:v1", "org/model:v1", exactID)
	test("explicit digest", "org/model@sha256:abc", "org/model@sha256:abc", exactID)
	test("exact identifier", " org/model ", "org/model", exactID)
}

func TestLlamaCPPProfile(t *testing.T) {
	profile, ok := profileFor("llamacpp")
	if !ok {
		t.Fatal("llamacpp profile missing")
	}
	if role, ok := profile.roleFor("POST", "/v1/chat/completions"); !ok || role != roleInferencePOST {
		t.Fatalf("chat route = %v, %v", role, ok)
	}
	if got := profile.normalizeModel("org/model:Q4_K_M"); got != "org/model:Q4_K_M" {
		t.Fatalf("exact model id normalized to %q", got)
	}
	if profile.StandalonePort != 8080 || profile.ReservedPersistedPort != 8081 {
		t.Fatalf("ports = facade %d, reserved %d", profile.StandalonePort, profile.ReservedPersistedPort)
	}
}

func TestLlamaCPPModelListRoutes(t *testing.T) {
	profile, ok := profileFor("llamacpp")
	if !ok {
		t.Fatal("llamacpp profile missing")
	}
	for _, path := range []string{"/models", "/v1/models"} {
		t.Run(path, func(t *testing.T) {
			route, ok := profile.routeFor(http.MethodGet, path)
			if !ok {
				t.Fatalf("GET %s is not classified", path)
			}
			if route.Role != roleModelListOpenAIGET || route.upstreamPath() != "/models" {
				t.Fatalf("GET %s route = %+v, want OpenAI model list at upstream /models", path, route)
			}
		})
	}
}

// Routes is a classifier. Unlisted engine controls and OPTIONS pass through.
func TestRoleForClassifiesOnlyDeclaredRoutes(t *testing.T) {
	llamacpp, ok := profileFor("llamacpp")
	if !ok {
		t.Fatal("llamacpp profile missing")
	}

	for _, tc := range []struct {
		name     string
		profile  engineProfile
		method   string
		path     string
		wantRole routeRole
		wantOK   bool
	}{
		{"llamacpp chat", llamacpp, "POST", "/v1/chat/completions", roleInferencePOST, true},
		{"llamacpp list", llamacpp, "GET", "/v1/models", roleModelListOpenAIGET, true},
		{"llamacpp messages passthrough", llamacpp, http.MethodPost, "/v1/messages", 0, false},

		// The method is part of the classification. Without it a POST to the
		// model-list path would be served as a list, and a GET to an
		// inference path would emit a workload.
		{"wrong method on list", llamacpp, "POST", "/v1/models", 0, false},
		{"wrong method on inference", llamacpp, "GET", "/v1/chat/completions", 0, false},
		{"llamacpp models post passthrough", llamacpp, http.MethodPost, "/models", 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			role, ok := tc.profile.roleFor(tc.method, tc.path)
			if ok != tc.wantOK {
				t.Fatalf("roleFor(%s %s) ok = %v, want %v", tc.method, tc.path, ok, tc.wantOK)
			}
			if ok && role != tc.wantRole {
				t.Fatalf("roleFor(%s %s) role = %v, want %v", tc.method, tc.path, role, tc.wantRole)
			}
		})
	}
}

// A path may legitimately be declared twice under different methods. roleFor
// used to return on the first path match, which made the second entry
// unreachable — it would be forwarded verbatim instead of classified, and
// nothing would say so.
func TestRoleForFindsAPathDeclaredUnderTwoMethods(t *testing.T) {
	p := engineProfile{Routes: []route{
		{Path: "/v1/models", Role: roleInferencePOST},
		{Path: "/v1/models", Role: roleModelListOpenAIGET},
	}}

	if role, ok := p.roleFor("GET", "/v1/models"); !ok || role != roleModelListOpenAIGET {
		t.Errorf("GET /v1/models = (%v, %v), want the model-list role", role, ok)
	}
	if role, ok := p.roleFor("POST", "/v1/models"); !ok || role != roleInferencePOST {
		t.Errorf("POST /v1/models = (%v, %v), want the inference role", role, ok)
	}
	if _, ok := p.roleFor("DELETE", "/v1/models"); ok {
		t.Error("DELETE /v1/models classified; an undeclared method must forward verbatim")
	}
}

// No shipped engine declares the same (path, method) twice; a duplicate would
// make whichever entry came second dead.
func TestNoDuplicateRoutePerMethod(t *testing.T) {
	for _, p := range profiles {
		seen := map[string]bool{}
		for _, r := range p.Routes {
			key := r.Role.method() + " " + r.Path
			if seen[key] {
				t.Errorf("%s declares %q twice; the second entry is unreachable", p.Name, key)
			}
			seen[key] = true
		}
	}
}

// The error-ID prefix derives from Name so it cannot disagree with the
// broker's expectations. The persisted-port filename comes from the shared table.
func TestDerivedIdentifiers(t *testing.T) {
	profile, ok := profileFor("llamacpp")
	if !ok {
		t.Fatal("llamacpp profile missing")
	}
	if got := profile.PortFile; got != "llamacpp-proxy-port.json" {
		t.Errorf("llamacpp PortFile = %q", got)
	}
	if got := upstreamUnreachableID(profile, "peer-A"); got != "llamacpp-proxy:upstream-unreachable:peer-A" {
		t.Errorf("llamacpp upstreamUnreachableID = %q", got)
	}
}

// chooseStartupPort restores a previously chosen port, except when the broker
// forces one or when the stored value belongs to the engine itself.
func TestChooseStartupPort(t *testing.T) {
	profile, ok := profileFor("llamacpp")
	if !ok {
		t.Fatal("llamacpp profile missing")
	}

	for _, tc := range []struct {
		name            string
		profile         engineProfile
		flagPort        int
		ignorePersisted bool
		persisted       int
		hasPersisted    bool
		want            int
	}{
		{"restores persisted", profile, 8080, false, 8100, true, 8100},
		{"broker override wins", profile, 8080, true, 8100, true, 8080},
		{"no persisted value", profile, 8080, false, 0, false, 8080},
		{"engine port is reserved", profile, 8080, false, 8081, true, 8080},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := chooseStartupPort(tc.profile, tc.flagPort, tc.ignorePersisted, tc.persisted, tc.hasPersisted)
			if got != tc.want {
				t.Fatalf("chooseStartupPort = %d, want %d", got, tc.want)
			}
		})
	}
}
