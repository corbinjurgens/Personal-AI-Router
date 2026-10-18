// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"nvpair-shared/engines"
)

func strataManifest(t *testing.T) *Manifest {
	t.Helper()
	reg := NewRegistry()
	if err := reg.LoadOverrideDir("testdata/strata"); err != nil {
		t.Fatalf("load user-manifest fixture: %v", err)
	}
	m, ok := reg.Get("strata")
	if !ok {
		t.Fatal("Strata fixture was not registered")
	}
	if err := m.Validate(); err != nil {
		t.Fatalf("validate Strata fixture: %v", err)
	}
	return m
}

// Only an in-process HTTP fake is started. No lifecycle launch method is used.
func strataHTTPExecutor(t *testing.T, handler http.HandlerFunc) (*Executor, int) {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewUnstartedServer(handler)
	if err := server.Listener.Close(); err != nil {
		listener.Close()
		t.Fatalf("close unused httptest listener: %v", err)
	}
	server.Listener = listener
	server.Start()
	t.Cleanup(server.Close)
	address, ok := listener.Addr().(*net.TCPAddr)
	if !ok {
		t.Fatalf("listener address type = %T", listener.Addr())
	}
	ex := newTestExecutor(t, strataManifest(t))
	st, err := ex.state("strata")
	if err != nil {
		t.Fatal(err)
	}
	st.running, st.port = true, address.Port
	return ex, address.Port
}

func TestStrataManifestDescriptor(t *testing.T) {
	m := strataManifest(t)
	want := engines.Descriptor{
		Version: 1, Engine: "strata", Family: "strata", Label: "Strata",
		Platforms: []string{"darwin/arm64", "linux/amd64", "windows/amd64"},
		Available: true, Protocols: []string{engines.ProtocolOpenAIChat}, ModelNaming: engines.ModelNamingExact,
		Operations: map[string]engines.OperationCapability{
			"load": {ResponseTimeoutS: 600}, "unload": {ResponseTimeoutS: 30},
		},
	}
	got := m.descriptor("windows", "amd64")
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("descriptor = %+v, want %+v", got, want)
	}
	if err := got.Validate(); err != nil {
		t.Fatal(err)
	}
	// Check the complete serialized shape as well as its typed value.
	data, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	var shape map[string]json.RawMessage
	if err := json.Unmarshal(data, &shape); err != nil {
		t.Fatal(err)
	}
	if keys := sortedKeys(shape); !reflect.DeepEqual(keys, []string{"available", "engine", "family", "label", "modelNaming", "operations", "platforms", "protocols", "version"}) {
		t.Fatalf("descriptor fields = %v", keys)
	}
	for _, forbidden := range []string{"REPLACE_WITH", "server.py", "config.json", "python", "--", "http://", "runtime", "detect", "env", "args", "cmd"} {
		if strings.Contains(string(data), forbidden) {
			t.Errorf("descriptor leaks %q: %s", forbidden, data)
		}
	}
	platform, ok := m.PlatformFor("windows", "amd64")
	if !ok {
		t.Fatal("missing Windows platform")
	}
	rt := platform.Runtime
	wantArgs := []string{"/REPLACE_WITH_ABSOLUTE_PATH/server.py", "--engine", "strata", "--config", "/REPLACE_WITH_ABSOLUTE_PATH/config.json", "--host", "127.0.0.1", "--port", "{port}", "--lazy"}
	if rt.Mode != "process" || rt.Bin != "/REPLACE_WITH_ABSOLUTE_PATH/python" || rt.Bind != "127.0.0.1" || rt.Port != 18082 || !reflect.DeepEqual(rt.Args, wantArgs) {
		t.Fatalf("runtime = %+v", rt)
	}
}

func TestStrataReadinessIdentity(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
		want bool
	}{
		{"Strata", `{"service":"strata"}`, true},
		{"foreign service", `{"service":"ollama"}`, false},
		{"missing identity", `{"status":"ok"}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ex, port := strataHTTPExecutor(t, func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodGet || r.URL.RequestURI() != "/health" {
					t.Errorf("probe request = %s %s", r.Method, r.URL.RequestURI())
				}
				if _, err := io.WriteString(w, tc.body); err != nil {
					t.Errorf("write health: %v", err)
				}
			})
			m, _ := ex.reg.Get("strata")
			p, ok := m.HostPlatform()
			if !ok {
				t.Fatal("missing host platform")
			}
			if got := ex.probe(context.Background(), p.Runtime.Ready, port); got != tc.want {
				t.Fatalf("readiness = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestStrataModelsAndResidency(t *testing.T) {
	ex, _ := strataHTTPExecutor(t, func(w http.ResponseWriter, r *http.Request) {
		var body string
		switch r.URL.RequestURI() {
		case "/health":
			body = `{"service":"strata"}`
		case "/v1/models":
			body = `{"data":[{"id":"resident","status":{"value":"loaded"}},{"id":"cold","status":{"value":"unloaded"}},{"id":"unknown"},{"id":"loading","status":{"value":"loading"}}]}`
		default:
			t.Errorf("unexpected request: %s", r.URL.RequestURI())
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if r.Method != http.MethodGet {
			t.Errorf("request method = %s", r.Method)
		}
		if _, err := io.WriteString(w, body); err != nil {
			t.Errorf("write response: %v", err)
		}
	})
	got := ex.ModelsResult(context.Background())
	want := ModelsResult{
		Models:         []string{"resident", "cold", "unknown", "loading"},
		ByEngine:       map[string][]string{"strata": {"resident", "cold", "unknown", "loading"}},
		LoadedByEngine: map[string][]string{"strata": {"resident"}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("models = %+v, want %+v", got, want)
	}
}

func TestStrataModelOperationWireAndTypedErrors(t *testing.T) {
	for _, tc := range []struct {
		name, op, body string
		status         int
	}{
		{"load succeeds", "load", `{"ok":true}`, http.StatusOK},
		{"unload succeeds", "unload", `{"ok":true}`, http.StatusOK},
		{"load unknown model", "load", `{"error":"unknown model"}`, http.StatusNotFound},
		{"unload busy model", "unload", `{"error":"busy"}`, http.StatusConflict},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var requests atomic.Int32
			ex, _ := strataHTTPExecutor(t, func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Errorf("read request: %v", err)
				}
				if r.Method != http.MethodPost || r.URL.RequestURI() != "/v1/"+tc.op || r.Header.Get("Content-Type") != "application/json" || string(body) != `{"model":"configured-model"}` {
					t.Errorf("request = %s %s content-type %q body %q", r.Method, r.URL.RequestURI(), r.Header.Get("Content-Type"), body)
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.status)
				if _, err := io.WriteString(w, tc.body); err != nil {
					t.Errorf("write response: %v", err)
				}
			})
			result, err := ex.modelOperation(context.Background(), "strata", tc.op, "configured-model")
			if requests.Load() != 1 {
				t.Fatalf("requests = %d, want 1", requests.Load())
			}
			if tc.status == http.StatusOK {
				if err != nil || string(result) != tc.body {
					t.Fatalf("result = %s, error = %v", result, err)
				}
				return
			}
			var failure *ActionHTTPError
			if !errors.As(err, &failure) {
				t.Fatalf("error = %T %v, want ActionHTTPError", err, err)
			}
			want := ActionHTTPError{Action: tc.op + "_model", StatusCode: tc.status, Body: tc.body}
			if *failure != want {
				t.Fatalf("HTTP error = %+v, want %+v", failure, want)
			}
		})
	}
}

func TestStrataUnsupportedOperationsMakeNoRequests(t *testing.T) {
	var requests atomic.Int32
	ex, _ := strataHTTPExecutor(t, func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	})
	ctx := context.Background()
	t.Run("pull", func(t *testing.T) {
		if _, err := ex.PullModelStream(ctx, "strata", "model", nil); err == nil || err.Error() != `engine "strata" has no action "pull_model"` {
			t.Fatalf("pull error = %v", err)
		}
	})
	t.Run("delete", func(t *testing.T) {
		if _, err := ex.ModelDelete(ctx, "strata", "model"); err == nil || !strings.Contains(err.Error(), "does not support") {
			t.Fatalf("delete error = %v", err)
		}
	})
	t.Run("copy", func(t *testing.T) {
		if _, err := ex.CopyModelFrom(ctx, nil, "strata", "model", nil); !errors.Is(err, errModelCopyUnsupported) {
			t.Fatalf("copy error = %v", err)
		}
	})
	if requests.Load() != 0 {
		t.Fatalf("unsupported operations sent %d requests", requests.Load())
	}
}
