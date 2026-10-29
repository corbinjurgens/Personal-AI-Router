// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// bundledTestExecutor is an executor over the bundled manifests alone.
func bundledTestExecutor(t *testing.T) *Executor {
	t.Helper()
	reg := NewRegistry()
	if err := reg.LoadFS(bundledManifests, "manifests"); err != nil {
		t.Fatalf("LoadFS bundled: %v", err)
	}
	ex := NewExecutor(reg, NewReporter(nil), func(string, any) {}, t.TempDir())
	acceptModelMutations(ex)
	return ex
}

// The bundled bindings must produce exactly the action and params the model
// operations sent before bindings moved into the manifests. Load, unload and
// delete reproduce the former per-engine Go mapping byte for byte. A pull that
// names only a model formerly sent both "name" and "model"; it now sends the
// one key each engine reads.
func TestBundledModelOperationsKeepTheirWire(t *testing.T) {
	ex := bundledTestExecutor(t)
	const model = "m:1"
	cases := []struct {
		engine, op, action, params string
	}{
		{"ollama", "load", "run_model", `{"model":"m:1","stream":false}`},
		{"ollama", "unload", "unload_model", `{"keep_alive":0,"model":"m:1"}`},
		{"ollama", "delete", "delete_model", `{"name":"m:1"}`},
		{"ollama", "pull", "pull_model", `{"name":"m:1"}`},
		{"lmstudio", "load", "load_model", `{"model":"m:1"}`},
		{"lmstudio", "unload", "unload_model", `{"model":"m:1"}`},
		{"lmstudio", "delete", "delete_model", `{"model":"m:1"}`},
		{"lmstudio", "pull", "pull_model", `{"model":"m:1"}`},
		{"llamacpp", "load", "load_model", `{"model":"m:1"}`},
		{"llamacpp", "unload", "unload_model", `{"model":"m:1"}`},
		{"llamacpp", "delete", "delete_model", `{"model":"m:1"}`},
		{"llamacpp", "pull", "pull_model", `{"model":"m:1"}`},
	}
	for _, tc := range cases {
		t.Run(tc.engine+" "+tc.op, func(t *testing.T) {
			action, params, err := ex.modelOperationCall(tc.engine, tc.op, model)
			if err != nil {
				t.Fatal(err)
			}
			if action != tc.action || string(params) != tc.params {
				t.Fatalf("got %s %s, want %s %s", action, params, tc.action, tc.params)
			}
		})
	}
}

// Only Ollama's load waits longer than the ordinary budget, as before, and the
// budget it binds is the former 10-minute load allowance.
func TestBundledLoadBudgets(t *testing.T) {
	ex := bundledTestExecutor(t)
	want := map[string]time.Duration{"ollama": 10 * time.Minute, "lmstudio": 0, "llamacpp": 0}
	for engine, budget := range want {
		if got := ex.loadResponseBudget(engine); got != budget {
			t.Errorf("%s load budget = %s, want %s", engine, got, budget)
		}
	}
	ollama, _ := ex.reg.Get("ollama")
	if got := ollama.actionResponseBudget("run_model"); got != 10*time.Minute {
		t.Errorf("ollama run_model budget = %s, want 10m, so engine:action keeps it too", got)
	}
	for _, action := range []string{"unload_model", "delete_model", "pull_model", "list_models"} {
		if got := ollama.actionResponseBudget(action); got != 0 {
			t.Errorf("ollama %s budget = %s, want the ordinary budget", action, got)
		}
	}
}

func TestModelOperationErrors(t *testing.T) {
	ex := bundledTestExecutor(t)
	if _, _, err := ex.modelOperationCall("ollama", "load", ""); err == nil || !strings.Contains(err.Error(), "model is required") {
		t.Errorf("empty model error = %v", err)
	}
	if _, _, err := ex.modelOperationCall("nope", "load", "m"); err == nil || !strings.Contains(err.Error(), "unknown engine") {
		t.Errorf("unknown engine error = %v", err)
	}
	m := testEngineManifest(fakeEngineBin)
	other := newTestExecutor(t, m)
	if _, _, err := other.modelOperationCall("fake", "load", "m"); err == nil || !strings.Contains(err.Error(), "does not support the model load operation") {
		t.Errorf("unsupported operation error = %v", err)
	}
}

type recordedRequest struct {
	method, path, query, body string
}

// modelOperationEngine serves an HTTP engine whose bindings use renamed model
// params, defaults and a query-string delete, and records what it receives.
func modelOperationEngine(t *testing.T) (*Executor, func() []recordedRequest) {
	t.Helper()
	var mu sync.Mutex
	var seen []recordedRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !((r.Method == http.MethodPost && (r.URL.Path == "/v1/load" || r.URL.Path == "/v1/unload")) || (r.Method == http.MethodDelete && r.URL.Path == "/models")) {
			http.NotFound(w, r)
			return
		}
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		seen = append(seen, recordedRequest{r.Method, r.URL.Path, r.URL.RawQuery, string(body)})
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	t.Cleanup(srv.Close)

	m := testEngineManifest(fakeEngineBin)
	m.Engine = "variant"
	m.Actions = map[string]Action{
		"warm":  {HTTP: &ActionHTTP{Method: "POST", Path: "/v1/load"}},
		"cool":  {HTTP: &ActionHTTP{Method: "POST", Path: "/v1/unload"}},
		"erase": {HTTP: &ActionHTTP{Method: "DELETE", Path: "/models", ParamsIn: "query"}},
	}
	m.Operations = map[string]OperationBinding{
		"load":   {Action: "warm", ModelParam: "target", Params: map[string]json.RawMessage{"wait": json.RawMessage(`true`)}, ResponseTimeoutS: 5},
		"unload": {Action: "cool", ModelParam: "target"},
		"delete": {Action: "erase", Params: map[string]json.RawMessage{"scope": json.RawMessage(`"cache"`)}},
	}
	if err := m.Validate(); err != nil {
		t.Fatal(err)
	}
	ex := newTestExecutor(t, m)
	st, err := ex.state("variant")
	if err != nil {
		t.Fatal(err)
	}
	st.running = true
	st.port = srv.Listener.Addr().(*net.TCPAddr).Port
	return ex, func() []recordedRequest {
		mu.Lock()
		defer mu.Unlock()
		return append([]recordedRequest(nil), seen...)
	}
}

// engine:load-model, engine:unload-model and engine:delete-model take only an
// engine and a model; the manifest's bindings decide the engine's spelling.
func TestTypedModelOperationRPCs(t *testing.T) {
	ex, seen := modelOperationEngine(t)
	var out bytes.Buffer
	m := NewManager(NewCodec(&out), ex, nil, "")
	acceptModelMutations(ex)
	id := json.RawMessage("1")
	for _, method := range []string{loadModelMethod, unloadModelMethod, deleteModelMethod} {
		out.Reset()
		m.runModelOperation(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: method,
			Params: json.RawMessage(`{"engine":"variant","model":"org/m"}`)})
		if !strings.Contains(out.String(), `"ok":true`) {
			t.Fatalf("%s response = %s, want the engine's result", method, out.String())
		}
	}
	want := []recordedRequest{
		{"POST", "/v1/load", "", `{"target":"org/m","wait":true}`},
		{"POST", "/v1/unload", "", `{"target":"org/m"}`},
		{"DELETE", "/models", "model=org%2Fm&scope=cache", ""},
	}
	got := seen()
	if len(got) != len(want) {
		t.Fatalf("engine saw %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("request %d = %+v, want %+v", i, got[i], want[i])
		}
	}

	for _, tc := range []struct{ params, want string }{
		{`{"engine":"variant"}`, "engine and model are required"},
		{`{"model":"m"}`, "engine and model are required"},
		{`{"engine":"variant","model":"m"`, "invalid params"},
	} {
		out.Reset()
		m.runModelOperation(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: loadModelMethod, Params: json.RawMessage(tc.params)})
		mustContain(t, out.String(), tc.want)
	}
	out.Reset()
	m.runModelOperation(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: unloadModelMethod,
		Params: json.RawMessage(`{"engine":"nope","model":"m"}`)})
	mustContain(t, out.String(), "unknown engine")
}

// The typed methods are dispatched by the read loop, not only reachable from a
// direct call.
func TestTypedModelOperationsAreDispatched(t *testing.T) {
	ex, seen := modelOperationEngine(t)
	var out lockedBuffer
	m := NewManager(NewCodec(&out), ex, nil, "")
	acceptModelMutations(ex)
	id := json.RawMessage("7")
	m.handleMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: deleteModelMethod,
		Params: json.RawMessage(`{"engine":"variant","model":"m"}`)})
	deadline := time.Now().Add(5 * time.Second)
	for !strings.Contains(out.String(), `"id":7`) {
		if time.Now().After(deadline) {
			t.Fatalf("no response to %s: %s", deleteModelMethod, out.String())
		}
		time.Sleep(10 * time.Millisecond)
	}
	if got := seen(); len(got) != 1 || got[0].method != "DELETE" {
		t.Fatalf("engine saw %+v, want one DELETE", got)
	}
}

// A remote pull that names only a model gets its params from the pull binding.
func TestPullWithoutParamsUsesThePullBinding(t *testing.T) {
	bodies := make(chan string, 1)
	srv := httptest.NewServer(fixtureRequestHandler(http.MethodPost, "/api/pull", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		bodies <- string(body)
		_, _ = w.Write([]byte(`{"status":"success"}` + "\n"))
	}))
	defer srv.Close()
	ex := newHTTPPullTestExecutor(t, srv, Action{HTTP: &ActionHTTP{Method: "POST", Path: "/api/pull"}})
	m, _ := ex.reg.Get("fake")
	m.Operations = map[string]OperationBinding{"pull": {Action: pullModelAction, ModelParam: "name"}}
	if _, err := ex.PullModelStream(context.Background(), "fake", "tiny", nil); err != nil {
		t.Fatal(err)
	}
	if got := <-bodies; got != `{"name":"tiny"}` {
		t.Fatalf("pull body = %s, want the bound name param", got)
	}
}
