// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

const credentialTestKey = "test-key-123"

// ownServingCredential makes st's process the one serving port with the
// manager-generated credentialTestKey.
func ownServingCredential(st *engineState, port int) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.port = port
	st.proc = &managedProc{}
	st.plat.Runtime.servingLaunch = &llamaServingLaunch{Credential: credentialTestKey}
}

// servingCredentialExecutor returns a diagnosis executor that owns the serving
// credential for port.
func servingCredentialExecutor(t *testing.T, endpoint string, port int) (*Executor, *engineState, string, string) {
	t.Helper()
	ex, st, started, stopped := newDiagnosisExecutor(t, endpoint, port)
	ownServingCredential(st, port)
	return ex, st, started, stopped
}

func credentialResponse(t *testing.T, w http.ResponseWriter, value any) {
	t.Helper()
	if err := json.NewEncoder(w).Encode(value); err != nil {
		t.Error("fake engine response failed")
	}
}

func credentialPublicOutput(t *testing.T, ex *Executor) (*syncBuffer, *syncBuffer) {
	t.Helper()
	out, logs := &syncBuffer{}, &syncBuffer{}
	codec := NewCodec(out)
	ex.reporter = NewReporter(codec)
	ex.emit = func(method string, params any) {
		if err := codec.Notify(method, params); err != nil {
			t.Error("capture notification failed")
		}
	}
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(logs, nil)))
	t.Cleanup(func() { slog.SetDefault(previous) })
	return out, logs
}

func assertCredentialNotPublic(t *testing.T, ex *Executor, out, logs *syncBuffer) {
	t.Helper()
	errors, err := json.Marshal(ex.Errors())
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(errors, []byte(credentialTestKey)) || strings.Contains(credentialOutput(out), credentialTestKey) || strings.Contains(credentialOutput(logs), credentialTestKey) {
		t.Fatal("credential appeared in reports, notifications or logs")
	}
}

func TestCredentialUseProbeDiagnosisHealthAndAction(t *testing.T) {
	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !(r.Method == http.MethodGet && (r.URL.Path == "/" || r.URL.Path == "/api/tags")) {
			http.NotFound(w, r)
			return
		}
		requests.Add(1)
		if r.Header.Get("Authorization") != "Bearer "+credentialTestKey {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		credentialResponse(t, w, map[string]any{"models": []string{}, "role": "router"})
	}))
	defer srv.Close()
	port := diagnosisServerPort(t, srv)
	ex, st, _, _ := servingCredentialExecutor(t, srv.URL, port)
	c, err := ex.credentialForCandidate(t.Context(), st, port)
	if err != nil {
		t.Fatal(err)
	}
	st.plat.Runtime.Ready.JSONMatch = &ProbeJSONMatch{Field: "role", Value: "router"}
	if ok, status, err := ex.probeStatus(t.Context(), st.plat.Runtime.Ready, port, c); err != nil || !ok || status != http.StatusOK {
		t.Fatal("keyed identity probe failed")
	}
	if d, err := ex.diagnoseConnection(t.Context(), st, port); err != nil || d != diagnosisAppRunning {
		t.Fatal("diagnosis did not use the probe key")
	}
	st.mu.Lock()
	st.running, st.healthy, st.gen = true, false, 1
	st.mu.Unlock()
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan struct{})
	go func() { defer close(done); ex.runHealth(ctx, st, "fake", port, 1) }()
	defer func() { cancel(); <-done }()
	deadline := time.After(4 * time.Second)
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	for {
		st.mu.Lock()
		healthy := st.healthy
		st.mu.Unlock()
		if healthy {
			break
		}
		select {
		case <-deadline:
			t.Fatal("health never became ready")
		case <-ticker.C:
		}
	}
	result, err := ex.Action(t.Context(), "fake", "list_models", nil)
	if err != nil {
		t.Fatal(err)
	}
	var response struct {
		Models []string `json:"models"`
	}
	if json.Unmarshal(result, &response) != nil || response.Models == nil || requests.Load() < 4 {
		t.Fatal("keyed list_models did not return structured inventory")
	}
}

func TestCredentialUseAbsentAndRejectedDiagnosis(t *testing.T) {
	for _, configured := range []bool{false, true} {
		name := "absent"
		if configured {
			name = "serving key rejected"
		}
		t.Run(name, func(t *testing.T) {
			var attached atomic.Bool
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodGet || r.URL.Path != "/" {
					http.NotFound(w, r)
					return
				}
				attached.Store(r.Header.Get("Authorization") != "")
				w.WriteHeader(http.StatusForbidden)
			}))
			defer srv.Close()
			want := diagnosisAuthRequired
			if configured {
				want = diagnosisAuthRejected
			}
			port := diagnosisServerPort(t, srv)
			ex, st, started, stopped := newDiagnosisExecutor(t, srv.URL, port)
			if configured {
				ownServingCredential(st, port)
			}
			out, logs := credentialPublicOutput(t, ex)
			d, err := ex.diagnoseConnection(t.Context(), st, port)
			if err != nil || d != want || attached.Load() != configured {
				t.Fatal("authentication diagnosis lacked request evidence")
			}
			if err := ex.Start(t.Context(), "fake"); err == nil || err.Error() != diagnosisMessage(st.manifest.DisplayName, want) {
				t.Fatal("pre-existing authentication refusal lost its diagnosis")
			}
			if fileExists(started) || fileExists(stopped) {
				t.Fatal("authentication refusal ran start/stop")
			}
			found := false
			for _, frame := range out.frames() {
				if frame.Method != "errors:report" {
					continue
				}
				var report serviceError
				if json.Unmarshal(frame.Params, &report) != nil || report.Message != diagnosisMessage(st.manifest.DisplayName, want) {
					t.Fatal("failure notification lost structured diagnosis")
				}
				found = true
			}
			if !found {
				t.Fatal("failure fixture emitted no error report")
			}
			assertCredentialNotPublic(t, ex, out, logs)
			c, err := ex.credentialForCandidate(t.Context(), st, port)
			if err != nil {
				t.Fatal(err)
			}
			wantErr := errAuthRequired
			if configured {
				wantErr = errAuthRejected
			}
			if err := ex.waitReady(t.Context(), st.plat.Runtime.Ready, port, st.plat.Runtime.AuthStatuses, c); err != wantErr {
				t.Fatal("readiness authentication evidence changed")
			}
		})
	}
}

func TestCredentialUseOwnedServingKeyAndStoppedRefusal(t *testing.T) {
	ex := newTestExecutor(t, testEngineManifest(fakeEngineBin))
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	ownServingCredential(st, 45678)
	c, err := ex.credentialForCandidate(t.Context(), st, 45678)
	if err != nil || c.key != credentialTestKey || c.port != 45678 {
		t.Fatal("owned prepared key unavailable before readiness")
	}
	if got, err := ex.LocalCredential("fake"); err != nil || got.Credential != "" {
		t.Fatal("stopped engine exported a candidate")
	}
	if _, err := ex.Action(t.Context(), "fake", "list_models", nil); err == nil {
		t.Fatal("stopped HTTP action accepted")
	}
	st.running = true
	if got, err := ex.LocalCredential("fake"); err != nil || got.Credential != credentialTestKey || got.Port != 45678 {
		t.Fatal("running owned key unavailable")
	}
	if c, err := ex.credentialForCandidate(t.Context(), st, 45679); err != nil || c.key != "" {
		t.Fatal("key was selected for a port the process does not serve")
	}
	st.proc = nil
	if got, err := ex.LocalCredential("fake"); err != nil || got.Credential != "" {
		t.Fatal("adopted Strata exported prepared key")
	}
	if c, err := ex.credentialForCandidate(t.Context(), st, 45678); err != nil || c.key != "" {
		t.Fatal("adopted engine selected a key")
	}
}

func TestCredentialUseInitialURLAndTransportRefusal(t *testing.T) {
	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		requests.Add(1)
	}))
	defer srv.Close()
	port := diagnosisServerPort(t, srv)
	c := candidateCredential{key: credentialTestKey, port: port}
	for _, endpoint := range []string{
		"http://localhost:" + strconv.Itoa(port) + "/", "http://192.0.2.1:" + strconv.Itoa(port) + "/",
		"http://127.0.0.1:" + strconv.Itoa(port+1) + "/", "http://127.0.0.1/", "https://127.0.0.1:" + strconv.Itoa(port) + "/",
		"http://user@127.0.0.1:" + strconv.Itoa(port) + "/", srv.URL + "/#fragment", "http:opaque",
	} {
		t.Run(strings.ReplaceAll(endpoint, "/", "_"), func(t *testing.T) {
			ex := &Executor{client: newEngineHTTPClient(time.Second)}
			if ok, status, err := ex.probeStatus(t.Context(), &Probe{HTTP: endpoint}, port, c); ok || status != 0 || err != errCredentialRequest {
				t.Fatal("invalid keyed endpoint was not refused")
			}
		})
	}
	ex := &Executor{client: &http.Client{Transport: healthProbeTransport(func(*http.Request) (*http.Response, error) {
		t.Error("unsupported transport was called")
		return nil, errors.New("unsafe")
	})}}
	if _, _, err := ex.probeStatus(t.Context(), &Probe{HTTP: srv.URL}, port, c); err != errCredentialRequest {
		t.Fatal("unsupported transport accepted")
	}
	if requests.Load() != 0 {
		t.Fatal("refused URL made network requests")
	}
	keyed, st, _, _ := servingCredentialExecutor(t, srv.URL, port)
	ownServingCredential(st, 0)
	if _, err := keyed.credentialForCandidate(t.Context(), st, 0); err != errCredentialRequest {
		t.Fatal("invalid candidate port accepted")
	}
}

func TestCredentialUseUnkeyedActionPreservesPortAndTransport(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	m.Actions["read"] = Action{HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/read"}}
	ex := newTestExecutor(t, m)
	st, err := ex.state(m.Engine)
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.port, st.running = 0, true
	st.mu.Unlock()
	calls := 0
	ex.client.Transport = healthProbeTransport(func(req *http.Request) (*http.Response, error) {
		calls++
		if req.URL.Port() != "0" || req.Header.Get("Authorization") != "" {
			t.Error("unkeyed request changed its endpoint or attached a credential")
		}
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`{"ok":true}`))}, nil
	})
	result, err := ex.Action(t.Context(), m.Engine, "read", nil)
	if err != nil || string(result) != `{"ok":true}` || calls != 1 {
		t.Fatalf("unkeyed action did not retain its transport: calls=%d err=%v", calls, err)
	}
}

func TestCredentialUseRedirectsNeverFollow(t *testing.T) {
	var targetCalls, anonymous atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/target" {
			http.NotFound(w, r)
			return
		}
		targetCalls.Add(1)
	}))
	defer target.Close()
	var destination atomic.Value
	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !(r.Method == http.MethodGet && (r.URL.Path == "/" || r.URL.Path == "/target" || r.URL.Path == "/api/tags")) {
			http.NotFound(w, r)
			return
		}
		if r.URL.Path == "/target" {
			targetCalls.Add(1)
			return
		}
		if r.Header.Get("Authorization") != "Bearer "+credentialTestKey {
			anonymous.Add(1)
		}
		http.Redirect(w, r, destination.Load().(string), http.StatusTemporaryRedirect)
	}))
	defer source.Close()
	port := diagnosisServerPort(t, source)
	ex, st, _, _ := servingCredentialExecutor(t, source.URL, port)
	st.running = true
	for _, tc := range []struct{ name, destination string }{
		{"same origin", source.URL + "/target"}, {"cross origin", strings.Replace(target.URL, "127.0.0.1", "localhost", 1) + "/target"}, {"different port", target.URL + "/target"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			destination.Store(tc.destination)
			if ok, status, err := ex.probeStatus(t.Context(), &Probe{HTTP: source.URL}, port, candidateCredential{key: credentialTestKey, port: port}); err != nil || ok || status != http.StatusTemporaryRedirect {
				t.Fatal("keyed redirect was not preserved")
			}
			var actionErr *ActionHTTPError
			if _, err := ex.Action(t.Context(), "fake", "list_models", nil); !errors.As(err, &actionErr) || actionErr.StatusCode != http.StatusTemporaryRedirect {
				t.Fatal("keyed action followed redirect")
			}
		})
	}
	if targetCalls.Load() != 0 || anonymous.Load() != 0 {
		t.Fatal("redirect leaked or stripped credential")
	}
}

func TestCredentialUseProxyNeverReceivesKey(t *testing.T) {
	var proxyCalls atomic.Int32
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !(r.Method == http.MethodGet && (r.URL.Path == "/" || r.URL.Path == "/api/tags")) {
			http.NotFound(w, r)
			return
		}
		proxyCalls.Add(1)
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer proxy.Close()
	t.Setenv("HTTP_PROXY", proxy.URL)
	t.Setenv("HTTPS_PROXY", proxy.URL)
	t.Setenv("NO_PROXY", "")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !(r.Method == http.MethodGet && (r.URL.Path == "/" || r.URL.Path == "/api/tags")) {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer "+credentialTestKey {
			w.WriteHeader(http.StatusUnauthorized)
		}
	}))
	defer srv.Close()
	port := diagnosisServerPort(t, srv)
	ex, st, _, _ := servingCredentialExecutor(t, srv.URL, port)
	tr, ok := ex.client.Transport.(*http.Transport)
	if !ok {
		t.Fatal("missing transport")
	}
	// Explicit proxy injection also proves the clone clears Proxy, since Go's
	// environment proxy excludes loopback even with NO_PROXY unset.
	proxyURL, err := http.NewRequest(http.MethodGet, proxy.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	tr.Proxy = http.ProxyURL(proxyURL.URL)
	c, err := ex.credentialForCandidate(t.Context(), st, port)
	if err != nil {
		t.Fatal(err)
	}
	if ok, _, err := ex.probeStatus(t.Context(), st.plat.Runtime.Ready, port, c); err != nil || !ok {
		t.Fatal("keyed probe used proxy")
	}
	st.running = true
	if _, err := ex.Action(t.Context(), "fake", "list_models", nil); err != nil {
		t.Fatal("keyed action used proxy")
	}
	if proxyCalls.Load() != 0 || tr.Proxy == nil {
		t.Fatal("proxy received key or shared transport mutated")
	}
}

func TestCredentialUseActionRedactsSuccessAndFailure(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		status     int
	}{
		{"JSON field names and escaped values", `{"test-key-123":{"nested":["test-key-\u003123",1234567890123456789]}}`, http.StatusOK},
		{"text", "echo " + credentialTestKey, http.StatusOK},
		{"failure", "echo " + credentialTestKey, http.StatusUnauthorized},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodGet || r.URL.Path != "/api/tags" {
					http.NotFound(w, r)
					return
				}
				w.WriteHeader(tc.status)
				if _, err := io.WriteString(w, tc.body); err != nil {
					t.Error("write failed")
				}
			}))
			defer srv.Close()
			ex, st, _, _ := servingCredentialExecutor(t, srv.URL, diagnosisServerPort(t, srv))
			st.running = true
			result, err := ex.Action(t.Context(), "fake", "list_models", nil)
			if tc.status == http.StatusOK {
				if err != nil || bytes.Contains(result, []byte(credentialTestKey)) {
					t.Fatal("successful action exposed credential or failed")
				}
				if tc.name == "JSON field names and escaped values" {
					var decoded map[string]struct {
						Nested []json.RawMessage `json:"nested"`
					}
					if json.Unmarshal(result, &decoded) != nil || len(decoded["[redacted]"].Nested) != 2 {
						t.Fatal("redaction lost JSON structure")
					}
					var value string
					if json.Unmarshal(decoded["[redacted]"].Nested[0], &value) != nil || value != "[redacted]" || string(decoded["[redacted]"].Nested[1]) != "1234567890123456789" {
						t.Fatal("redaction lost strings or numeric precision")
					}
				} else {
					var value string
					if json.Unmarshal(result, &value) != nil || value != "echo [redacted]" {
						t.Fatal("text was not redacted")
					}
				}
			} else {
				var httpErr *ActionHTTPError
				if !errors.As(err, &httpErr) || httpErr.StatusCode != tc.status || strings.Contains(err.Error(), credentialTestKey) || strings.Contains(err.Error(), "echo") {
					t.Fatal("failure exposed response body or lost HTTP status")
				}
			}
		})
	}
}

func credentialOutput(out *syncBuffer) string {
	out.mu.Lock()
	defer out.mu.Unlock()
	return out.buf.String()
}

// Keep the fake CLI environment entirely separate from the serving API key.
func TestCredentialUseCLIActionReceivesNoServingKey(t *testing.T) {
	ex, st, _, _ := servingCredentialExecutor(t, "http://127.0.0.1:{port}/", diagnosisRefusedPort(t))
	helper, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	st.manifest.Actions["echo"] = Action{Cmd: []string{helper, "credential-cli-helper"}}
	result, err := ex.Action(t.Context(), "fake", "echo", nil)
	if err != nil || bytes.Contains(result, []byte(credentialTestKey)) {
		t.Fatal("CLI action received serving credential")
	}
	var response struct {
		Present *bool `json:"credentialPresent"`
	}
	if json.Unmarshal(result, &response) != nil || response.Present == nil || *response.Present {
		t.Fatal("serving credential appeared in CLI argv/environment")
	}
}

func TestCredentialUseTransportFailuresAreSanitized(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()
	port := diagnosisServerPort(t, srv)
	ex, st, _, _ := servingCredentialExecutor(t, srv.URL, port)
	transport, ok := ex.client.Transport.(*http.Transport)
	if !ok {
		t.Fatal("missing transport")
	}
	transport.DialContext = func(context.Context, string, string) (net.Conn, error) {
		return nil, errors.New("reflected transport " + credentialTestKey)
	}
	c, err := ex.credentialForCandidate(t.Context(), st, port)
	if err != nil {
		t.Fatal(err)
	}
	if ok, _, err := ex.probeStatus(t.Context(), st.plat.Runtime.Ready, port, c); ok || err != errCredentialHTTP {
		t.Fatal("probe exposed transport error")
	}
	if d, err := ex.diagnoseConnection(t.Context(), st, port); d != diagnosisNone || err != errCredentialHTTP {
		t.Fatal("diagnosis replaced credential transport error")
	}
	st.running = true
	if _, err := ex.Action(t.Context(), "fake", "list_models", nil); err != errCredentialHTTP {
		t.Fatal("action exposed transport error")
	}
}
