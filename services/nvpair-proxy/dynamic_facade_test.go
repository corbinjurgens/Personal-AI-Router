// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Coverage for descriptor-backed (dynamic) engine facades: the built-in
// profiles stay exactly as they were, a descriptor engine gets a facade from
// its protocols alone, routes locally and to a pinned peer, and anything the
// proxy cannot front fails closed.

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func customDescriptor() engines.Descriptor {
	return engines.Descriptor{
		Version:     engines.DescriptorVersion,
		Engine:      "llamacpp-custom",
		Family:      engines.NameLlamaCPP,
		Label:       "llama.cpp (custom)",
		Platforms:   []string{"darwin/arm64"},
		Available:   true,
		Protocols:   []string{engines.ProtocolOpenAIV1, engines.ProtocolLlamaCPPRouter},
		ModelNaming: engines.ModelNamingExact,
		Operations:  map[string]engines.OperationCapability{},
	}
}

func customProfile(t *testing.T) engineProfile {
	t.Helper()
	profile, err := dynamicProfile(customDescriptor())
	if err != nil {
		t.Fatalf("dynamic profile: %v", err)
	}
	return profile
}

// The built-in facades are fixed values, pinned here entry by entry. A dynamic
// engine is additive: nothing about these may move.
func TestBuiltInProfilesAreUnchanged(t *testing.T) {
	post := func(path string) route { return route{Path: path, Role: roleInferencePOST} }
	openAI := []route{post("/v1/chat/completions"), post("/v1/completions"), post("/v1/embeddings")}
	want := []engineProfile{
		{
			Engine: engines.Engine{Name: "ollama", DisplayName: "Ollama", DiscoveryService: "ol", FacadePort: 11434, EnginePortBase: 11435, PortFile: "proxy-port.json"},
			Routes: slices.Concat([]route{
				post("/api/generate"), post("/api/chat"), post("/api/embeddings"), post("/api/embed"),
				{Path: "/api/tags", Role: roleModelListNativeGET},
				{Path: "/v1/models", Role: roleModelListOpenAIGET},
			}, openAI, []route{post("/v1/messages")}),
			StandalonePort: 11435, ModelNaming: impliedLatestTag, SupportsHostAlias: true,
		},
		{
			Engine:         engines.Engine{Name: "lmstudio", DisplayName: "LM Studio", DiscoveryService: "lm", FacadePort: 1234, EnginePortBase: 1235, PortFile: "lmstudio-proxy-port.json"},
			Routes:         slices.Concat([]route{{Path: "/v1/models", Role: roleModelListOpenAIGET}}, openAI, []route{post("/v1/messages")}),
			StandalonePort: 1234, ModelNaming: exactID, ReservedPersistedPort: 1235,
		},
		{
			Engine: engines.Engine{Name: "llamacpp", DisplayName: "llama.cpp", DiscoveryService: "lc", FacadePort: 8080, EnginePortBase: 8081, PortFile: "llamacpp-proxy-port.json"},
			Routes: slices.Concat([]route{
				{Path: "/models", Role: roleModelListOpenAIGET},
				{Path: "/v1/models", UpstreamPath: "/models", Role: roleModelListOpenAIGET},
			}, openAI),
			StandalonePort: 8080, ModelNaming: exactID, ReservedPersistedPort: 8081,
		},
	}
	if !reflect.DeepEqual(profiles, want) {
		t.Fatalf("built-in profiles changed:\n got %+v\nwant %+v", profiles, want)
	}
	for _, p := range profiles {
		if p.dynamic() {
			t.Errorf("built-in %s reports dynamic", p.Name)
		}
		if got := engines.AddressMethod(p.Name, "ready"); got != p.Name+":ready" {
			t.Errorf("%s facade address = %q", p.Name, got)
		}
	}
}

func TestProtocolRoutesCoverFacadeProtocols(t *testing.T) {
	for _, protocol := range engines.FacadeProtocols() {
		if len(protocolRoutes[protocol]) == 0 {
			t.Errorf("facade protocol %q has no routes", protocol)
		}
	}
	if len(protocolRoutes) != len(engines.FacadeProtocols()) {
		t.Errorf("protocolRoutes has %d entries, engines.FacadeProtocols %d", len(protocolRoutes), len(engines.FacadeProtocols()))
	}
}

func TestDynamicProfileFromDescriptor(t *testing.T) {
	t.Run("llama.cpp fork", func(t *testing.T) {
		p := customProfile(t)
		if p.Name != "llamacpp-custom" || p.DiscoveryService != "e-llamacpp-custom" || p.PortFile != "llamacpp-custom-proxy-port.json" || !p.dynamic() {
			t.Fatalf("identity = %+v", p.Engine)
		}
		if p.StandalonePort != 0 || p.ReservedPersistedPort != 0 || p.SupportsHostAlias || p.FacadePort != 0 {
			t.Fatalf("dynamic profile borrowed fixed ports or alias support: %+v", p)
		}
		// The router list wins /v1/models even though openai-v1 is declared first.
		if r, ok := p.routeFor(http.MethodGet, "/v1/models"); !ok || r.upstreamPath() != "/models" {
			t.Fatalf("GET /v1/models = %+v, %v; want the router list at /models", r, ok)
		}
		for _, path := range []string{"/v1/chat/completions", "/v1/completions", "/v1/embeddings"} {
			if role, ok := p.roleFor(http.MethodPost, path); !ok || role != roleInferencePOST {
				t.Errorf("POST %s not inference", path)
			}
		}
		if _, ok := p.roleFor(http.MethodPost, "/v1/messages"); ok {
			t.Error("undeclared Anthropic route classified")
		}
		if got := p.normalizeModel("org/model:Q4_K_M"); got != "org/model:Q4_K_M" {
			t.Errorf("exact naming normalized to %q", got)
		}
	})
	t.Run("chat only", func(t *testing.T) {
		d := customDescriptor()
		d.Engine, d.Family, d.Protocols = "strata", "strata", []string{engines.ProtocolOpenAIChat}
		p, err := dynamicProfile(d)
		if err != nil {
			t.Fatal(err)
		}
		want := []route{{Path: "/v1/models", Role: roleModelListOpenAIGET}, {Path: "/v1/chat/completions", Role: roleInferencePOST}}
		if !reflect.DeepEqual(p.Routes, want) {
			t.Fatalf("routes = %+v, want %+v", p.Routes, want)
		}
	})
	t.Run("ollama api fork", func(t *testing.T) {
		d := customDescriptor()
		d.Engine, d.Family, d.Protocols, d.ModelNaming = "ollama-fork", engines.NameOllama, []string{engines.ProtocolOllamaAPI, engines.ProtocolOpenAIV1}, engines.ModelNamingImpliedLatestTag
		p, err := dynamicProfile(d)
		if err != nil {
			t.Fatal(err)
		}
		if role, ok := p.roleFor(http.MethodGet, "/api/tags"); !ok || role != roleModelListNativeGET {
			t.Fatal("native model list missing")
		}
		if role, ok := p.roleFor(http.MethodGet, "/v1/models"); !ok || role != roleModelListOpenAIGET {
			t.Fatal("OpenAI model list missing")
		}
		if got := p.normalizeModel("llama3"); got != "llama3:latest" {
			t.Fatalf("implied-tag naming gave %q", got)
		}
	})
}

// Every way of asking for a facade the proxy cannot honour is refused before a
// listener exists.
func TestEnableFacadeFailsClosed(t *testing.T) {
	redirectConfigDir(t)
	p := newBoundProxy(NewCodec(rwNop{}))
	p.serveCtx = t.Context()
	t.Cleanup(func() { p.shutdown(t.Context()) })
	custom := customDescriptor()
	builtin := customDescriptor()
	builtin.Engine = engines.NameLlamaCPP
	unknownProtocol := customDescriptor()
	unknownProtocol.Protocols = []string{"grpc"}
	listOnly := customDescriptor()
	listOnly.Protocols = []string{engines.ProtocolLlamaCPPRouter}
	newer := customDescriptor()
	newer.Version = engines.DescriptorVersion + 1
	long := customDescriptor()
	long.Engine = strings.Repeat("a", engines.MaxDynamicIDLen+1)
	for _, tc := range []struct {
		name   string
		params enableFacadeParams
		want   string
	}{
		{"unknown engine without descriptor", enableFacadeParams{Engine: "llamacpp-custom", Port: 1}, "unknown engine"},
		{"descriptor for a built-in", enableFacadeParams{Engine: engines.NameLlamaCPP, Port: 1, Descriptor: &builtin}, "built in"},
		{"descriptor names another engine", enableFacadeParams{Engine: "strata", Port: 1, Descriptor: &custom}, "not \"strata\""},
		{"unknown protocol", enableFacadeParams{Engine: "llamacpp-custom", Port: 1, Descriptor: &unknownProtocol}, "unknown protocol"},
		{"model list only", enableFacadeParams{Engine: "llamacpp-custom", Port: 1, Descriptor: &listOnly}, "no protocol"},
		{"unknown descriptor version", enableFacadeParams{Engine: "llamacpp-custom", Port: 1, Descriptor: &newer}, "version"},
		{"overlong id", enableFacadeParams{Engine: long.Engine, Port: 1, Descriptor: &long}, "at most"},
		{"no port", enableFacadeParams{Engine: "llamacpp-custom", Descriptor: &custom}, "explicit facade port"},
		{"alias on a dynamic engine", enableFacadeParams{Engine: "llamacpp-custom", Port: 1, Descriptor: &custom, AliasAddresses: []string{"127.0.0.2"}}, "alias"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := p.enableFacade(tc.params); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("enableFacade error = %v, want %q", err, tc.want)
			}
			if got := len(p.enabledFacades()); got != 0 {
				t.Fatalf("a refused enable left %d facades", got)
			}
		})
	}
}

// A llama.cpp fork registered only by manifest gets a facade over the wire,
// learns its local backend and its own discovery entry through addressed
// messages, and serves inference from a fake engine on 127.0.0.1.
func TestDynamicFacadeRoutesLocally(t *testing.T) {
	redirectConfigDir(t)
	w := &recordingWriter{}
	p := newBoundProxy(NewCodec(w))
	p.serveCtx = t.Context()
	t.Cleanup(func() { p.shutdown(t.Context()) })

	var engineCalls atomic.Int32
	engine := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		if r.URL.Path != "/v1/chat/completions" || json.NewDecoder(r.Body).Decode(&body) != nil || body.Model != "org/model:Q4_K_M" {
			t.Errorf("engine got %s %s model %q", r.Method, r.URL.Path, body.Model)
		}
		engineCalls.Add(1)
		_, _ = io.WriteString(rw, `{"choices":[]}`)
	}))
	t.Cleanup(engine.Close)

	descriptor := customDescriptor()
	var port int
	for attempt := 0; attempt < 8 && port == 0; attempt++ {
		candidate := freeTCPPort(t)
		f := rpc(t, p, w, "facade/enable", enableFacadeParams{Engine: descriptor.Engine, Port: candidate, IgnorePersistedPort: true, Descriptor: &descriptor})
		if f.Error == nil {
			port = candidate
		} else if f.Error.Code != codeFacadeBindFailed {
			t.Fatalf("facade/enable: %s", f.Error.Message)
		}
	}
	if port == 0 {
		t.Fatal("no free port for the dynamic facade")
	}
	if persisted, ok := loadPersistedPort(customProfile(t)); !ok || persisted != port {
		t.Fatalf("persisted facade port = %d, %v; want %d", persisted, ok, port)
	}
	readyAddressed := false
	for _, f := range frames(t, w) {
		readyAddressed = readyAddressed || f.Method == "e-llamacpp-custom:ready"
	}
	if !readyAddressed {
		t.Fatal("the dynamic facade's ready notification is not addressed e-llamacpp-custom:")
	}

	backend := nodeFor(t, "engine", engine.URL)
	rpcOK(t, p, w, engines.AddressMethod(descriptor.Engine, "node/set-local-backend"),
		localBackend{Engine: descriptor.Engine, Host: backend.Addresses[0], Port: backend.Port, Healthy: true})
	snapshot, err := json.Marshal(noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{{
		HostUUID: "uuid-self", Name: "self", IP: "127.0.0.1",
		Services:       map[noderec.ServiceKey]noderec.ServiceStatus{"e-llamacpp-custom": {Port: port}},
		ModelsByEngine: map[string][]string{descriptor.Engine: {"org/model:Q4_K_M"}},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	p.handleMessage(&Message{JSONRPC: "2.0", Method: engines.AddressMethod(descriptor.Engine, noderec.NotifyNodes), Params: snapshot})
	if nodes := p.facadeFor(descriptor.Engine).discovery.Nodes(); len(nodes) != 1 {
		t.Fatalf("dynamic facade routing set = %+v, want the self entry", nodes)
	}

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Post(fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port), "application/json",
		strings.NewReader(`{"model":"org/model:Q4_K_M","messages":[]}`))
	if err != nil {
		t.Fatalf("request through the dynamic facade: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK || engineCalls.Load() != 1 {
		t.Fatalf("status %d body %q engine calls %d", resp.StatusCode, body, engineCalls.Load())
	}
	if got := resp.Header.Get(nodepolicy.EngineHeader); got != descriptor.Engine {
		t.Fatalf("engine header = %q, want %q", got, descriptor.Engine)
	}
}

// A peer advertising the same descriptor engine is reached over the pinned
// direct path, and over the reverse session under the dynamic service key when
// direct fails.
func TestDynamicFacadeRoutesToPinnedPeer(t *testing.T) {
	for _, tc := range []struct {
		name        string
		direct      bool
		wantReverse int32
	}{
		{"direct", true, 0},
		{"reverse", false, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			f := newProxyReverseFixtureFor(t, customProfile(t), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				_, _ = io.WriteString(w, "peer-result")
			}))
			f.directHealthy.Store(tc.direct)
			recorder := httptest.NewRecorder()
			f.p.soleFacade().handleHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"fixture-model"}`)))
			if recorder.Code != http.StatusOK || recorder.Body.String() != "peer-result" || calls.Load() != 1 {
				t.Fatalf("status/body/calls = %d/%q/%d", recorder.Code, recorder.Body.String(), calls.Load())
			}
			if got := f.reverseCalls.Load(); got != tc.wantReverse {
				t.Fatalf("reverse dials = %d, want %d", got, tc.wantReverse)
			}
			if f.directCalls.Load() == 0 {
				t.Fatal("direct path was not tried first")
			}
		})
	}
}
