// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/performancewire"
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

func TestStrataManifestDetectionSelectsInterpreter(t *testing.T) {
	data, err := os.ReadFile("testdata/strata/strata.json")
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.ToSlash(t.TempDir())
	interpreter := root + "/synthetic-interpreter"
	data = bytes.ReplaceAll(data, []byte("/REPLACE_WITH_ABSOLUTE_PATH/python"), []byte(interpreter))
	data = bytes.ReplaceAll(data, []byte("/REPLACE_WITH_ABSOLUTE_PATH"), []byte(root))
	reg := NewRegistry()
	if _, err := reg.addManifest("synthetic Strata template", data); err != nil {
		t.Fatal(err)
	}
	m, ok := reg.Get("strata")
	if !ok {
		t.Fatal("Strata template missing")
	}
	// These are inert regular files. No interpreter, Python import or Strata
	// process is executed, and the launch is only resolved into argv.
	for _, path := range []string{interpreter, root + "/server.py", root + "/config.json"} {
		writeDetectionFile(t, path)
	}
	ex := newTestExecutor(t, m)
	if installed, err := ex.Detect("strata"); err != nil || !installed {
		t.Fatalf("complete template detection = %v, %v", installed, err)
	}
	st, err := ex.state("strata")
	if err != nil {
		t.Fatal(err)
	}
	launch, err := resolveProcessLaunch(st.plat.Runtime, st.binPath, map[string]string{"host": "127.0.0.1", "port": "18082"})
	if err != nil || launch.Bin != interpreter || launch.Args[0] != root+"/server.py" {
		t.Fatalf("resolved launch = %+v, %v", launch, err)
	}
	removeDetectionFile(t, root+"/config.json")
	if installed, err := ex.Detect("strata"); err != nil || installed || st.binPath != interpreter {
		t.Fatalf("missing config detection = %v, %v, binary %q", installed, err, st.binPath)
	}
	if !m.descriptor(runtime.GOOS, runtime.GOARCH).Available {
		t.Fatal("missing config changed descriptor availability")
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
					http.NotFound(w, r)
					return
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
			got, err := ex.probe(context.Background(), p.Runtime.Ready, port, candidateCredential{port: port})
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("readiness = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestStrataModelsAndResidency(t *testing.T) {
	ex, _ := strataHTTPExecutor(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || (r.URL.RequestURI() != "/health" && r.URL.RequestURI() != "/v1/models") {
			http.NotFound(w, r)
			return
		}
		var body string
		switch r.URL.RequestURI() {
		case "/health":
			body = `{"service":"strata"}`
		case "/v1/models":
			body = `{"data":[{"id":"resident","status":{"value":"loaded"}},{"id":"cold","status":{"value":"unloaded"}},{"id":"unknown"},{"id":"loading","status":{"value":"loading"}}]}`
		default:
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
	managerIncarnation := ex.switchIncarnation
	downloadState := ex.downloads.snapshot()
	want := ModelsResult{
		Models:         []string{"resident", "cold", "unknown", "loading"},
		ByEngine:       map[string][]string{"strata": {"resident", "cold", "unknown", "loading"}},
		LoadedByEngine: map[string][]string{"strata": {"resident"}},
		DownloadState:  &downloadState,
		MemoryEvidence: &performancewire.MemoryEvidence{
			Version: 1, OmittedEntries: 0,
			Entries: []performancewire.MemoryEvidenceEntry{{
				Engine: "strata", Model: "resident", State: performancewire.MemoryStateUnknown,
				Uncertainty: []performancewire.MemoryUncertainty{
					performancewire.MemoryUncertaintyUnsupported,
					performancewire.MemoryUncertaintyQualificationMissing,
				},
				Source:          performancewire.MemorySource{ManagerIncarnation: &managerIncarnation},
				AppliedSettings: performancewire.MemoryAppliedSettings{},
				Envelope:        performancewire.MemoryEnvelope{},
				Observed:        performancewire.MemoryObservation{},
				Qualification:   performancewire.MemoryQualification{},
				Provenance:      performancewire.MemoryProvenance{},
				Freshness:       performancewire.MemoryFreshness{ReadingValidForMs: 15000},
			}},
		},
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
				if r.Method != http.MethodPost || r.URL.RequestURI() != "/v1/"+tc.op {
					http.NotFound(w, r)
					return
				}
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
