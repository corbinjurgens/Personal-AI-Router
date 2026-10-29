// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"sort"
	"strings"
	"testing"

	"nvpair-shared/engines"
)

func descriptorFor(t *testing.T, reg *Registry, engine string) engines.Descriptor {
	t.Helper()
	for _, d := range reg.Descriptors() {
		if d.Engine == engine {
			return d
		}
	}
	t.Fatalf("no descriptor for %q", engine)
	return engines.Descriptor{}
}

func TestBundledDescriptors(t *testing.T) {
	reg := bundledTestExecutor(t).reg
	all := []engines.Descriptor{
		{
			Engine: "llamacpp", Family: "llamacpp", Label: "llama.cpp",
			Protocols:   []string{engines.ProtocolLlamaCPPRouter, engines.ProtocolOpenAIV1},
			ModelNaming: engines.ModelNamingExact, ModelStore: "llamacpp", Catalog: "llamacpp",
			Operations: map[string]engines.OperationCapability{"load": {}, "unload": {}, "pull": {}, "delete": {}},
		},
		{
			Engine: "lmstudio", Family: "lmstudio", Label: "LM Studio",
			Protocols:   []string{engines.ProtocolOpenAIV1, engines.ProtocolAnthropicMessages},
			ModelNaming: engines.ModelNamingExact, ModelStore: "lmstudio", Catalog: "lmstudio",
			Operations: map[string]engines.OperationCapability{"load": {}, "unload": {}, "pull": {}, "delete": {}},
		},
		{
			Engine: "ollama", Family: "ollama", Label: "Ollama",
			Protocols:   []string{engines.ProtocolOllamaAPI, engines.ProtocolOpenAIV1, engines.ProtocolAnthropicMessages},
			ModelNaming: engines.ModelNamingImpliedLatestTag, ModelStore: "ollama", Catalog: "ollama",
			Operations: map[string]engines.OperationCapability{"load": {ResponseTimeoutS: 600}, "unload": {}, "pull": {}, "delete": {}},
		},
	}
	got := reg.Descriptors()
	if len(got) != len(all) {
		t.Fatalf("got %d descriptors, want %d", len(got), len(all))
	}
	host := runtime.GOOS + "/" + runtime.GOARCH
	for i, want := range all {
		d := got[i]
		if err := d.Validate(); err != nil {
			t.Errorf("%s: %v", d.Engine, err)
		}
		if d.Version != engines.DescriptorVersion || !slices.IsSorted(d.Platforms) || len(d.Platforms) == 0 {
			t.Errorf("%s: version %d platforms %v", d.Engine, d.Version, d.Platforms)
		}
		if d.Available != slices.Contains(d.Platforms, host) {
			t.Errorf("%s: available = %v for host %s and platforms %v", d.Engine, d.Available, host, d.Platforms)
		}
		want.Version, want.Platforms, want.Available = d.Version, d.Platforms, d.Available
		if !reflect.DeepEqual(d, want) {
			t.Errorf("descriptor\n got %+v\nwant %+v", d, want)
		}
	}
}

// writeForkManifest writes, as a user would, a complete manifest for a second
// llama.cpp build: its own ID, port and model store, family llamacpp, and the
// conventional action names without an operations block.
func writeForkManifest(t *testing.T, dir, modelsDir string) {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "llama-server-custom")
	fork := map[string]any{
		"engine":           "llamacpp-custom",
		"display_name":     "llama.cpp (custom build)",
		"manifest_version": 1,
		"family":           "llamacpp",
		"protocols":        []string{"llamacpp-router", "openai-v1"},
		"models_dir":       modelsDir,
		"runtime": map[string]any{
			"bin":   bin,
			"args":  []string{"--host", "{host}", "--port", "{port}"},
			"env":   map[string]string{"LLAMA_CACHE": "{models_dir}"},
			"port":  18181,
			"ready": map[string]any{"http": "http://127.0.0.1:{port}/props", "status": 200},
		},
		"platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{"detect": []string{bin}}},
		"actions": map[string]any{
			"list_models":  map[string]any{"http": map[string]string{"method": "GET", "path": "/models"}, "result": map[string]string{"array": "data", "field": "id"}},
			"pull_model":   map[string]any{"http": map[string]string{"method": "POST", "path": "/models"}, "progress_protocol": "llamacpp-models-sse"},
			"load_model":   map[string]any{"http": map[string]string{"method": "POST", "path": "/models/load"}},
			"unload_model": map[string]any{"http": map[string]string{"method": "POST", "path": "/models/unload"}},
			"delete_model": map[string]any{"http": map[string]string{"method": "DELETE", "path": "/models", "params_in": "query"}},
		},
	}
	data, err := json.MarshalIndent(fork, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "llamacpp-custom.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

// forkRegistry loads the bundled manifests and the fork through the startup
// path: bundled first, then the per-user engines directory.
func forkRegistry(t *testing.T, modelsDir string) *Registry {
	t.Helper()
	reg := NewRegistry()
	if err := reg.LoadFS(bundledManifests, "manifests"); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(t.TempDir(), "engines")
	writeForkManifest(t, dir, modelsDir)
	if err := reg.LoadOverrideDir(dir); err != nil {
		t.Fatal(err)
	}
	if _, ok := reg.Get("llamacpp-custom"); !ok {
		t.Fatal("the fork manifest was not loaded")
	}
	return reg
}

// A fork declared only as manifest data gets its family's model-store layout,
// download catalogue and response budgets, while keeping its own identity,
// port and store.
func TestForkManifestGetsItsFamilyBehavior(t *testing.T) {
	t.Setenv("LLAMA_CACHE", "")
	modelsDir := t.TempDir()
	reg := forkRegistry(t, modelsDir)
	ex := NewExecutor(reg, NewReporter(nil), func(string, any) {}, t.TempDir())

	t.Run("identity", func(t *testing.T) {
		fork, _ := reg.Get("llamacpp-custom")
		stock, _ := reg.Get("llamacpp")
		forkPlatform, _ := fork.HostPlatform()
		stockPlatform, _ := stock.HostPlatform()
		if forkPlatform.Runtime.Port == stockPlatform.Runtime.Port || forkPlatform.ModelsDir == stockPlatform.ModelsDir {
			t.Fatalf("fork shares the stock port %d or store %q", stockPlatform.Runtime.Port, stockPlatform.ModelsDir)
		}
		d := descriptorFor(t, reg, "llamacpp-custom")
		if err := d.Validate(); err != nil {
			t.Fatal(err)
		}
		stockDescriptor := descriptorFor(t, reg, "llamacpp")
		if d.Family != "llamacpp" || d.ModelStore != "llamacpp" || d.Catalog != "llamacpp" || !d.Available {
			t.Fatalf("fork descriptor = %+v", d)
		}
		if !reflect.DeepEqual(d.Operations, stockDescriptor.Operations) || !reflect.DeepEqual(d.Protocols, stockDescriptor.Protocols) {
			t.Fatalf("fork operations %v protocols %v, want llama.cpp's %v %v", d.Operations, d.Protocols, stockDescriptor.Operations, stockDescriptor.Protocols)
		}
	})

	t.Run("model store", func(t *testing.T) {
		_, store, err := ex.modelStoreFor("llamacpp-custom")
		if err != nil {
			t.Fatal(err)
		}
		if store.adapter != "llamacpp" || store.engine != "llamacpp-custom" || store.root != filepath.Clean(modelsDir) {
			t.Fatalf("store = %+v, want the llama.cpp layout rooted at the fork's own store %s", store, modelsDir)
		}
		const repo, commit = "owner/tiny-GGUF", "0123456789abcdef0123456789abcdef01234567"
		writeLlamaCache(t, modelsDir, repo, commit, map[string][]byte{
			"tiny-Q4_K_M.gguf": testBytes(4_000, 1),
			"tiny-Q8_0.gguf":   testBytes(6_000, 2),
		})
		copyTestNativeModels(t, ex, "llamacpp-custom", repo+":Q4_K_M")
		list, err := ex.modelSourceFiles(context.Background(), "llamacpp-custom", repo+":Q4_K_M")
		if err != nil {
			t.Fatal(err)
		}
		if list.Engine != "llamacpp-custom" || list.Revision != commit || len(list.Files) != 1 || list.Files[0].Path != "tiny-Q4_K_M.gguf" {
			t.Fatalf("listing = %+v, want llama.cpp's snapshot selection", list)
		}
	})

	t.Run("catalog", func(t *testing.T) {
		srv, _ := llamaCPPServer(t, "")
		c := newCatalogService()
		c.baseURL = srv.URL + "/catalog"
		if _, err := c.Catalog(context.Background(), "llamacpp-custom", "", "", ""); err == nil {
			t.Fatal("an unresolved fork ID found a catalogue by name")
		}
		c.familyOf = reg.familyOf
		res, err := c.Catalog(context.Background(), "llamacpp-custom", "", "", "")
		if err != nil {
			t.Fatal(err)
		}
		if !res.Searchable || len(res.Models) != len(llamaCPPPublishers) {
			t.Fatalf("fork catalogue = %+v, want llama.cpp's searchable browse list", res)
		}
		// The client spellings of the built-ins still resolve.
		if got := c.adapterFor("llama.cpp"); got != "llamacpp" {
			t.Fatalf("llama.cpp spelling selects %q", got)
		}
	})

	t.Run("timeouts and operations", func(t *testing.T) {
		if got, want := ex.loadResponseBudget("llamacpp-custom"), ex.loadResponseBudget("llamacpp"); got != want {
			t.Fatalf("fork load budget %s, llama.cpp %s", got, want)
		}
		if waitsForEngineReadiness(controlLoadPath, ex.loadResponseBudget("llamacpp-custom")) {
			t.Fatal("a remote fork load waits on the readiness budget; llama.cpp's does not")
		}
		fork, _ := reg.Get("llamacpp-custom")
		if got := fork.actionResponseBudget("load_model"); got != 0 {
			t.Fatalf("fork load_model budget = %s, want the ordinary budget", got)
		}
		for _, op := range engines.Operations() {
			forkAction, forkParams, err := ex.modelOperationCall("llamacpp-custom", op, "owner/m:Q4_K_M")
			if err != nil {
				t.Fatal(err)
			}
			stockAction, stockParams, _ := ex.modelOperationCall("llamacpp", op, "owner/m:Q4_K_M")
			if forkAction != stockAction || !bytes.Equal(forkParams, stockParams) {
				t.Errorf("%s: fork %s %s, llama.cpp %s %s", op, forkAction, forkParams, stockAction, stockParams)
			}
		}
	})
}

// A descriptor is published beyond engine-manager, so nothing a manifest says
// about how to run the engine may reach it: binaries, commands, environment,
// paths, URLs, action definitions or operation parameters.
func TestDescriptorsExcludeSecretsAndPaths(t *testing.T) {
	host := runtime.GOOS + "/" + runtime.GOARCH
	secret := map[string]any{
		"engine":           "leaky",
		"display_name":     "Leaky",
		"manifest_version": 1,
		"family":           "llamacpp",
		"protocols":        []string{"openai-v1"},
		"models_dir":       "/home/user/SECRET-MODELS",
		"detect":           []string{"/opt/SECRET-DETECT/server"},
		"install": map[string]any{
			"fetch": map[string]string{"url": "https://example.invalid/SECRET-URL.zip", "sha256": strings.Repeat("ab", 32)},
			"run":   []string{"SECRET-INSTALL", "{download}"},
		},
		"uninstall": map[string]any{"run": []string{"SECRET-UNINSTALL"}, "remove": []string{"/opt/SECRET-REMOVE"}},
		"runtime": map[string]any{
			"bin":    "/opt/SECRET-BIN/server",
			"args":   []string{"--api-key", "SECRET-ARG"},
			"env":    map[string]string{"API_TOKEN": "SECRET-ENV-VALUE"},
			"cli":    "/opt/SECRET-CLI",
			"port":   18200,
			"ready":  map[string]any{"http": "http://127.0.0.1:{port}/SECRET-READY"},
			"health": map[string]any{"tcp": "127.0.0.1:{port}"},
			"stop":   map[string]any{"cmd": []string{"SECRET-STOP"}},
		},
		"platforms": map[string]any{host: map[string]any{}},
		"operations": map[string]any{
			"load": map[string]any{"action": "load_model", "params": map[string]string{"token": "SECRET-PARAM"}, "response_timeout_s": 120},
		},
		"actions": map[string]any{
			"load_model":   map[string]any{"description": "SECRET-DESCRIPTION", "http": map[string]any{"method": "POST", "path": "/SECRET-PATH", "body_schema": map[string]string{"SECRET-SCHEMA": "string"}}},
			"delete_model": map[string]any{"remove_path": map[string]string{"root": "/opt/SECRET-ROOT", "path": "/opt/SECRET-ROOT/{model}"}},
			"pull_model":   map[string]any{"cmd": []string{"SECRET-CMD", "{model}"}},
		},
	}
	data, err := json.Marshal(secret)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "leaky.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	reg := NewRegistry()
	if err := reg.LoadFS(bundledManifests, "manifests"); err != nil {
		t.Fatal(err)
	}
	if err := reg.LoadDir(dir); err != nil {
		t.Fatalf("the secret-laden manifest must be valid for this test to mean anything: %v", err)
	}
	d := descriptorFor(t, reg, "leaky")
	if d.Operations["load"].ResponseTimeoutS != 120 || d.Operations["delete"] != (engines.OperationCapability{}) {
		t.Fatalf("leaky operations = %+v", d.Operations)
	}

	published, err := json.Marshal(map[string]any{"engines": reg.Descriptors()})
	if err != nil {
		t.Fatal(err)
	}
	for _, leaked := range []string{"SECRET", "/opt/", "/home/", "~", "{port}", "{model}", "{host}", "http:", "https:", "example.invalid", ".exe", "install_dir", "models_dir", "LLAMA_CACHE", "OLLAMA_", "llama-server", "/bin/", "serve"} {
		if bytes.Contains(published, []byte(leaked)) {
			t.Errorf("published descriptors contain %q: %s", leaked, published)
		}
	}

	// The descriptor's fields are an allowlist. A new one has to be added here
	// deliberately, after checking it cannot carry anything local.
	var shapes []map[string]json.RawMessage
	var envelope struct {
		Engines json.RawMessage `json:"engines"`
	}
	if err := json.Unmarshal(published, &envelope); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(envelope.Engines, &shapes); err != nil {
		t.Fatal(err)
	}
	allowed := []string{"available", "catalog", "engine", "family", "label", "modelNaming", "modelStore", "operations", "platforms", "protocols", "version"}
	for _, shape := range shapes {
		keys := make([]string, 0, len(shape))
		for k := range shape {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			if !slices.Contains(allowed, k) {
				t.Errorf("descriptor field %q is not in the reviewed allowlist", k)
			}
		}
	}
}

// engine:descriptors answers with every registered engine's descriptor.
func TestDescriptorsRPC(t *testing.T) {
	var out bytes.Buffer
	m := NewManager(NewCodec(&out), bundledTestExecutor(t), nil, "")
	id := json.RawMessage("3")
	m.handleMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: descriptorsMethod})
	var resp struct {
		Result struct {
			Engines []engines.Descriptor `json:"engines"`
		} `json:"result"`
	}
	if err := json.Unmarshal(out.Bytes(), &resp); err != nil {
		t.Fatalf("response %s: %v", out.String(), err)
	}
	var names []string
	for _, d := range resp.Result.Engines {
		names = append(names, d.Engine)
	}
	if strings.Join(names, ",") != "llamacpp,lmstudio,ollama" {
		t.Fatalf("descriptors for %v", names)
	}
}

// A family the manifest does not name is the engine's own ID, which selects
// no shared adapter unless the ID is itself an adapter family.
func TestFamilyDefaultsToEngineID(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	m.Engine = "strata"
	d := m.descriptor(runtime.GOOS, runtime.GOARCH)
	if d.Family != "strata" || d.ModelStore != "" || d.Catalog != "" || d.ModelNaming != engines.ModelNamingExact {
		t.Fatalf("descriptor = %+v", d)
	}
	if len(d.Protocols) != 0 || d.Protocols == nil {
		t.Fatalf("protocols = %#v, want an empty list rather than null", d.Protocols)
	}
	if err := d.Validate(); err != nil {
		t.Fatal(err)
	}
	ex := newTestExecutor(t, m)
	if _, _, err := ex.modelStoreFor("strata"); err == nil {
		t.Fatal("an engine with no store adapter resolved a model store")
	}
}

func TestManifestFieldValidation(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Manifest)
		want   string
	}{
		{"bad family", func(m *Manifest) { m.Family = "a/b" }, "family"},
		{"dot family", func(m *Manifest) { m.Family = ".." }, "family"},
		{"unknown protocol", func(m *Manifest) { m.Protocols = []string{"grpc"} }, "unknown protocol"},
		{"duplicate protocol", func(m *Manifest) { m.Protocols = []string{"openai-v1", "openai-v1"} }, "listed twice"},
		{"unknown naming", func(m *Manifest) { m.ModelNaming = "fuzzy" }, "model_naming"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := testEngineManifest(fakeEngineBin)
			tc.mutate(m)
			if err := m.Validate(); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Validate error = %v, want %q", err, tc.want)
			}
		})
	}
}
