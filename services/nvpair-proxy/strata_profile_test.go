// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Fixtures for a prepared Strata runtime behind the proxy.
//
// Strata is not a built-in engine. It is declared by the engine-manager
// user-manifest template, which publishes a descriptor, and the proxy fronts it
// from that descriptor like any other dynamic engine. These tests read the
// template, build the descriptor it publishes and take the path production
// does: engines.ProxyCanFront, dynamicProfile and facade/enable. The upstream
// is a loopback fake of the prepared Strata server.

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/noderec"
)

// strataManifestPath is the engine-manager template. The descriptor is derived
// from it so the proxy fixture cannot drift from what the template publishes.
var strataManifestPath = filepath.Join("..", "nvpair-engine-manager", "testdata", "strata", "strata.json")

// strataDescriptor is the descriptor engine-manager publishes for the template:
// identity, protocols, naming, the platforms the manifest defines and the
// operations it binds.
func strataDescriptor(t *testing.T) engines.Descriptor {
	t.Helper()
	data, err := os.ReadFile(strataManifestPath)
	if err != nil {
		t.Fatalf("read Strata template: %v", err)
	}
	type operation struct {
		ResponseTimeoutS int `json:"response_timeout_s"`
	}
	var manifest struct {
		Engine      string                     `json:"engine"`
		DisplayName string                     `json:"display_name"`
		Family      string                     `json:"family"`
		Protocols   []string                   `json:"protocols"`
		ModelNaming string                     `json:"model_naming"`
		Platforms   map[string]json.RawMessage `json:"platforms"`
		Operations  map[string]operation       `json:"operations"`
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode Strata template: %v", err)
	}
	d := engines.Descriptor{
		Version:     engines.DescriptorVersion,
		Engine:      manifest.Engine,
		Family:      manifest.Family,
		Label:       manifest.DisplayName,
		Available:   true,
		Protocols:   manifest.Protocols,
		ModelNaming: manifest.ModelNaming,
		Operations:  map[string]engines.OperationCapability{},
	}
	for platform := range manifest.Platforms {
		d.Platforms = append(d.Platforms, platform)
	}
	slices.Sort(d.Platforms)
	for name, op := range manifest.Operations {
		d.Operations[name] = engines.OperationCapability{ResponseTimeoutS: op.ResponseTimeoutS}
	}
	if err := engines.ProxyCanFront(d); err != nil {
		t.Fatalf("the proxy cannot front the Strata template: %v", err)
	}
	return d
}

// strataProfile is the facade profile the proxy builds for Strata.
func strataProfile(t *testing.T) engineProfile {
	t.Helper()
	profile, err := dynamicProfile(strataDescriptor(t))
	if err != nil {
		t.Fatalf("dynamic profile for Strata: %v", err)
	}
	return profile
}

// strataUpstream is a loopback fake of the prepared Strata server. It records
// every request and answers by path.
type strataUpstream struct {
	*httptest.Server
	mu   sync.Mutex
	seen []strataRequest
}

type strataRequest struct {
	method, path, contentType, body string
}

func (u *strataUpstream) requests() []strataRequest {
	u.mu.Lock()
	defer u.mu.Unlock()
	return slices.Clone(u.seen)
}

func newStrataUpstream(t *testing.T, handler http.HandlerFunc) *strataUpstream {
	t.Helper()
	u := &strataUpstream{}
	u.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read upstream request body: %v", err)
		}
		u.mu.Lock()
		u.seen = append(u.seen, strataRequest{
			method:      r.Method,
			path:        r.URL.Path,
			contentType: r.Header.Get("Content-Type"),
			body:        string(body),
		})
		u.mu.Unlock()
		handler(w, r)
	}))
	t.Cleanup(u.Close)
	return u
}

// strataProxy attaches one Strata facade, not started, to discovery nodes that
// advertise the given models.
func strataProxy(t *testing.T, rec *recRW, nodes ...Node) *Proxy {
	t.Helper()
	disc := NewDiscovery()
	for _, n := range nodes {
		disc.AddManual(n)
	}
	profile := strataProfile(t)
	return newTestProxy(profile, NewCodec(rec), disc, 18081)
}

func TestStrataIsFrontedFromItsDescriptor(t *testing.T) {
	d := strataDescriptor(t)
	if !slices.Equal(d.Protocols, []string{engines.ProtocolOpenAIChat}) || d.ModelNaming != engines.ModelNamingExact {
		t.Fatalf("Strata descriptor protocols %v naming %q, want openai-chat with exact ids", d.Protocols, d.ModelNaming)
	}
	if _, ok := profileFor("strata"); ok {
		t.Fatal("profileFor(strata) resolved: Strata has no built-in profile")
	}
	profile := strataProfile(t)
	if !profile.dynamic() || profile.DisplayName != "Strata" {
		t.Fatalf("profile = %+v, want a dynamic Strata profile", profile.Engine)
	}
	if profile.DiscoveryService != "e-strata" || profile.PortFile != "strata-proxy-port.json" {
		t.Fatalf("service key %q port file %q, want e-strata and strata-proxy-port.json", profile.DiscoveryService, profile.PortFile)
	}
	if profile.SupportsHostAlias {
		t.Fatal("a dynamic engine has no inherited host variable for an alias")
	}
}

func TestStrataEnableIsRefusedWithoutAFrontableDescriptor(t *testing.T) {
	d := strataDescriptor(t)
	withoutBase := d
	withoutBase.Protocols = []string{engines.ProtocolAnthropicMessages}
	p := newBoundProxy(NewCodec(rwNop{}))
	for _, tc := range []struct {
		name   string
		params enableFacadeParams
		want   string
	}{
		{"no descriptor", enableFacadeParams{Engine: "strata", Port: 18090}, `unknown engine "strata"`},
		{"no port", enableFacadeParams{Engine: "strata", Descriptor: &d}, "explicit facade port"},
		{"no chat and model list protocol", enableFacadeParams{Engine: "strata", Port: 18090, Descriptor: &withoutBase}, "no protocol a facade can serve"},
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

// facade/enable with the template's descriptor brings up a real facade on
// loopback, announces it under the e-strata address, persists its port and
// serves chat completions and the model list for a node that advertises the
// model. The fake Strata server stands in for the prepared runtime.
func TestStrataFacadeServesThroughEnable(t *testing.T) {
	redirectConfigDir(t)
	w := &recordingWriter{}
	p := newBoundProxy(NewCodec(w))
	p.serveCtx = t.Context()
	t.Cleanup(func() { p.shutdown(t.Context()) })

	const model = "strata-qwen"
	engine := newStrataUpstream(t, func(rw http.ResponseWriter, r *http.Request) {
		rw.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/models" {
			io.WriteString(rw, `{"object":"list","data":[{"id":"`+model+`","object":"model","status":{"value":"unloaded"}}]}`)
			return
		}
		io.WriteString(rw, `{"choices":[]}`)
	})

	descriptor := strataDescriptor(t)
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
		t.Fatal("no free port for the Strata facade")
	}
	if persisted, ok := loadPersistedPort(strataProfile(t)); !ok || persisted != port {
		t.Fatalf("persisted facade port = %d, %v; want %d", persisted, ok, port)
	}
	if !slices.ContainsFunc(frames(t, w), func(f frame) bool { return f.Method == "e-strata:ready" }) {
		t.Fatal("the Strata facade's ready notification is not addressed e-strata:")
	}

	backend := nodeFor(t, "engine", engine.URL)
	rpcOK(t, p, w, engines.AddressMethod(descriptor.Engine, "node/set-local-backend"),
		localBackend{Engine: descriptor.Engine, Host: backend.Addresses[0], Port: backend.Port, Healthy: true})
	snapshot, err := json.Marshal(noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{{
		HostUUID: "uuid-self", Name: "self", IP: "127.0.0.1",
		Services:       map[noderec.ServiceKey]noderec.ServiceStatus{"e-strata": {Port: port}},
		ModelsByEngine: map[string][]string{descriptor.Engine: {model}},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	p.handleMessage(&Message{JSONRPC: "2.0", Method: engines.AddressMethod(descriptor.Engine, noderec.NotifyNodes), Params: snapshot})

	client := &http.Client{Timeout: 5 * time.Second}
	base := fmt.Sprintf("http://127.0.0.1:%d", port)
	resp, err := client.Post(base+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"`+model+`","messages":[]}`))
	if err != nil {
		t.Fatalf("chat completion through the Strata facade: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || string(body) != `{"choices":[]}` {
		t.Fatalf("chat completion = %d %q", resp.StatusCode, body)
	}
	resp, err = client.Get(base + "/v1/models")
	if err != nil {
		t.Fatalf("model list through the Strata facade: %v", err)
	}
	body, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(body), `"id":"`+model+`"`) || !strings.Contains(string(body), `"object":"list"`) {
		t.Fatalf("model list = %d %q", resp.StatusCode, body)
	}
	var paths []string
	for _, r := range engine.requests() {
		paths = append(paths, r.method+" "+r.path)
	}
	if want := []string{"POST /v1/chat/completions", "GET /v1/models"}; !slices.Equal(paths, want) {
		t.Fatalf("engine saw %v, want %v", paths, want)
	}
}

func TestStrataRouteClassification(t *testing.T) {
	profile := strataProfile(t)
	for _, tc := range []struct {
		name          string
		method, path  string
		wantRole      routeRole
		wantClassed   bool
		wantInference bool
	}{
		{name: "chat completions", method: http.MethodPost, path: "/v1/chat/completions", wantRole: roleInferencePOST, wantClassed: true, wantInference: true},
		{name: "model list", method: http.MethodGet, path: "/v1/models", wantRole: roleModelListOpenAIGET, wantClassed: true},
		{name: "chat completions by GET", method: http.MethodGet, path: "/v1/chat/completions"},
		{name: "model list by POST", method: http.MethodPost, path: "/v1/models"},
		{name: "legacy completions", method: http.MethodPost, path: "/v1/completions"},
		{name: "embeddings", method: http.MethodPost, path: "/v1/embeddings"},
		{name: "anthropic messages", method: http.MethodPost, path: "/v1/messages"},
		{name: "responses", method: http.MethodPost, path: "/v1/responses"},
		{name: "llama.cpp router list", method: http.MethodGet, path: "/models"},
		{name: "load control", method: http.MethodPost, path: "/v1/load"},
		{name: "unload control", method: http.MethodPost, path: "/v1/unload"},
		{name: "health", method: http.MethodGet, path: "/health"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			role, classed := profile.roleFor(tc.method, tc.path)
			if classed != tc.wantClassed || (classed && role != tc.wantRole) {
				t.Fatalf("roleFor(%s %s) = %v, %v; want %v, %v", tc.method, tc.path, role, classed, tc.wantRole, tc.wantClassed)
			}
			if got := isInferenceRequest(profile, tc.method, tc.path); got != tc.wantInference {
				t.Fatalf("isInferenceRequest(%s %s) = %v, want %v", tc.method, tc.path, got, tc.wantInference)
			}
		})
	}
}

// Strata model ids are exact: the proxy must not treat "m" and "m:latest" as one
// model, as it does for Ollama.
func TestStrataModelIDsAreExact(t *testing.T) {
	profile := strataProfile(t)
	for _, id := range []string{"qwen3-30b-a3b", "org/qwen3:Q4_K_M", "qwen3-30b-a3b:latest"} {
		if got := profile.normalizeModel(id); got != id {
			t.Errorf("normalizeModel(%q) = %q, want it unchanged", id, got)
		}
	}
	node := Node{ID: "strata-node", Models: []string{"qwen3-30b-a3b:latest"}}
	if nodeAdvertisesModel(profile, node, "qwen3-30b-a3b") {
		t.Error("an untagged request matched a :latest inventory entry")
	}
	if !nodeAdvertisesModel(profile, node, "qwen3-30b-a3b:latest") {
		t.Error("the exact inventory id was not matched")
	}
}

const strataChatBody = `{"model":"strata-qwen","messages":[{"role":"user","content":"hi"}],"stream":false}`

func TestStrataChatCompletionsPassThrough(t *testing.T) {
	const reply = `{"id":"chatcmpl-1","object":"chat.completion","model":"strata-qwen","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]}`
	upstream := newStrataUpstream(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, reply)
	})
	rec := &recRW{}
	p := strataProxy(t, rec, nodeForModel(t, "strata-node", upstream.URL, "strata-qwen"))

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(strataChatBody))
	req.Header.Set("Content-Type", "application/json")
	p.soleFacade().handleHTTP(w, req)

	if w.Code != http.StatusOK || w.Body.String() != reply {
		t.Fatalf("client saw %d %q, want 200 with the upstream body", w.Code, w.Body.String())
	}
	got := upstream.requests()
	want := strataRequest{method: http.MethodPost, path: "/v1/chat/completions", contentType: "application/json", body: strataChatBody}
	if len(got) != 1 || got[0] != want {
		t.Fatalf("upstream requests = %+v, want exactly %+v", got, want)
	}
	if !waitFor(t, rec, workloadSubmittedMethod) || !rec.has(`"engine":"strata"`) {
		t.Fatal("chat completion was not tracked as a strata workload")
	}
}

func TestStrataStreamedChatCompletionsPassThrough(t *testing.T) {
	chunks := []string{
		`data: {"choices":[{"delta":{"content":"he"}}]}` + "\n\n",
		`data: {"choices":[{"delta":{"content":"llo"}}]}` + "\n\n",
		"data: [DONE]\n\n",
	}
	upstream := newStrataUpstream(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, chunk := range chunks {
			io.WriteString(w, chunk)
			w.(http.Flusher).Flush()
		}
	})
	p := strataProxy(t, &recRW{}, nodeForModel(t, "strata-node", upstream.URL, "strata-qwen"))

	w := httptest.NewRecorder()
	body := `{"model":"strata-qwen","stream":true}`
	p.soleFacade().handleHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))

	if got := w.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", got)
	}
	if got, want := w.Body.String(), strings.Join(chunks, ""); got != want {
		t.Fatalf("streamed body = %q, want %q", got, want)
	}
}

func TestStrataModelListIsFederatedInTheOpenAIDialect(t *testing.T) {
	listing := func(rows ...string) http.HandlerFunc {
		return func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"object":"list","data":[%s]}`, strings.Join(rows, ","))
		}
	}
	row := func(id, status string) string {
		return fmt.Sprintf(`{"id":%q,"object":"model","status":{"value":%q}}`, id, status)
	}
	first := newStrataUpstream(t, listing(row("strata-a", "loaded"), row("strata-b", "unloaded")))
	second := newStrataUpstream(t, listing(row("strata-a", "loaded"), row("strata-c", "unloaded")))
	p := strataProxy(t, &recRW{},
		nodeForModel(t, "node-1", first.URL, "strata-a"),
		nodeForModel(t, "node-2", second.URL, "strata-c"))

	w := httptest.NewRecorder()
	p.soleFacade().handleHTTP(w, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("model list status = %d: %s", w.Code, w.Body)
	}
	var list struct {
		Object string `json:"object"`
		Data   []struct {
			ID     string `json:"id"`
			Status struct {
				Value string `json:"value"`
			} `json:"status"`
		} `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &list); err != nil {
		t.Fatalf("model list is not JSON: %v: %s", err, w.Body)
	}
	if list.Object != "list" {
		t.Fatalf("object = %q, want list", list.Object)
	}
	var ids []string
	for _, m := range list.Data {
		ids = append(ids, m.ID+"="+m.Status.Value)
	}
	if want := []string{"strata-a=loaded", "strata-b=unloaded", "strata-c=unloaded"}; !slices.Equal(ids, want) {
		t.Fatalf("model list = %v, want %v (one row per id, upstream status preserved)", ids, want)
	}
	for _, u := range []*strataUpstream{first, second} {
		if got := u.requests(); len(got) != 1 || got[0].method != http.MethodGet || got[0].path != "/v1/models" {
			t.Fatalf("upstream requests = %+v, want one GET /v1/models", got)
		}
	}
}

// Chat completions for a model no node advertises is refused at the proxy and
// never reaches the engine.
func TestStrataChatCompletionsRefuseAnUnadvertisedModel(t *testing.T) {
	upstream := newStrataUpstream(t, func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, `{}`)
	})
	rec := &recRW{}
	p := strataProxy(t, rec, nodeForModel(t, "strata-node", upstream.URL, "strata-qwen"))

	w := httptest.NewRecorder()
	body := `{"model":"strata-other","messages":[]}`
	p.soleFacade().handleHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusBadGateway || !strings.Contains(w.Body.String(), "no available node advertises the requested model") {
		t.Fatalf("client saw %d %q, want 502 naming the missing model", w.Code, w.Body.String())
	}
	if got := upstream.requests(); len(got) != 0 {
		t.Fatalf("engine received %+v, want no request", got)
	}
	if rec.has(workloadSubmittedMethod) {
		t.Fatal("a refused request was tracked as a workload")
	}
}

// openai-chat claims two routes and the proxy classifies only those. Any other
// path is forwarded verbatim as ordinary traffic: the proxy does not refuse it,
// does not gate it on the advertised model, and does not track it as a
// workload. The engine's own answer reaches the client.
func TestStrataUnclassifiedPathsAreForwardedUntracked(t *testing.T) {
	for _, tc := range []struct {
		method, path, body string
	}{
		{http.MethodPost, "/v1/completions", `{"model":"strata-unadvertised","prompt":"hi"}`},
		{http.MethodPost, "/v1/embeddings", `{"model":"strata-unadvertised","input":"hi"}`},
		{http.MethodPost, "/v1/messages", `{"model":"strata-unadvertised","messages":[]}`},
		{http.MethodPost, "/v1/responses", `{"model":"strata-unadvertised","input":"hi"}`},
		{http.MethodPost, "/v1/unload", `{"model":"strata-qwen"}`},
	} {
		t.Run(tc.method+" "+tc.path, func(t *testing.T) {
			const notFound = `{"error":{"message":"not found"}}`
			upstream := newStrataUpstream(t, func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusNotFound)
				io.WriteString(w, notFound)
			})
			rec := &recRW{}
			p := strataProxy(t, rec, nodeForModel(t, "strata-node", upstream.URL, "strata-qwen"))

			w := httptest.NewRecorder()
			p.soleFacade().handleHTTP(w, httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body)))

			if w.Code != http.StatusNotFound || w.Body.String() != notFound {
				t.Fatalf("client saw %d %q, want the engine's own 404", w.Code, w.Body.String())
			}
			got := upstream.requests()
			if len(got) != 1 || got[0].path != tc.path || got[0].body != tc.body {
				t.Fatalf("upstream requests = %+v, want one verbatim %s", got, tc.path)
			}
			if rec.has(workloadSubmittedMethod) {
				t.Fatal("an unclassified path was tracked as a workload")
			}
		})
	}
}
