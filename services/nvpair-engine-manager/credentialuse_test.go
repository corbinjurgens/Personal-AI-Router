// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

	"nvpair-shared/engines"
)

const credentialTestKey = "test-key-123"

func credentialExecutor(t *testing.T, endpoint string, port int, store credentialBackend) (*Executor, *engineState, string, string) {
	t.Helper()
	ex, st, started, stopped := newDiagnosisExecutor(t, endpoint, port)
	st.plat.Runtime.UserCredential = true
	ex.credentials = credentialStore{backend: store}
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

func TestCredentialUseColdCommandStartLoadsSavedKey(t *testing.T) {
	port := diagnosisRefusedPort(t)
	fake := &credentialFake{value: credentialTestKey}
	ex, st, started, stopped := credentialExecutor(t, fmt.Sprintf("http://127.0.0.1:%d/", port), port, fake)
	st.plat.Runtime.Health = nil
	st.plat.Runtime.Ready.TimeoutS = 5
	var keyed atomic.Int32
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer "+credentialTestKey {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		keyed.Add(1)
		credentialResponse(t, w, map[string]bool{"ready": true})
	}))
	if err := srv.Listener.Close(); err != nil {
		t.Fatal(err)
	}
	serverDone := make(chan error, 1)
	ctx, cancel := context.WithTimeout(t.Context(), 8*time.Second)
	defer cancel()
	go func() {
		ticker := time.NewTicker(5 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				serverDone <- ctx.Err()
				return
			case <-ticker.C:
				if !fileExists(started) {
					continue
				}
				listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
				if err != nil {
					serverDone <- err
					return
				}
				srv.Listener = listener
				srv.Start()
				serverDone <- nil
				return
			}
		}
	}()
	startErr := ex.Start(ctx, "fake")
	if startErr != nil {
		cancel()
	}
	if err := <-serverDone; err != nil {
		t.Fatal("stand-in server did not start")
	}
	defer srv.Close()
	if startErr != nil {
		t.Fatalf("cold start: %v", startErr)
	}
	if !fileExists(started) || fileExists(stopped) || keyed.Load() == 0 {
		t.Fatal("cold start did not use the saved key without premature cleanup")
	}
	st.mu.Lock()
	running, healthy, adopted, proc := st.running, st.healthy, st.adopted, st.proc
	st.mu.Unlock()
	if !running || !healthy || adopted || proc != nil || fake.writes != 0 || fake.gets != 1 {
		t.Fatal("cold command start did not use a lazy durable read")
	}
	srv.Close()
	if err := ex.Stop("fake"); err != nil {
		t.Fatal(err)
	}
}

func TestCredentialUseFirstAdoptionLoadsSavedKey(t *testing.T) {
	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		requests.Add(1)
		if r.Header.Get("Authorization") != "Bearer "+credentialTestKey {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		credentialResponse(t, w, map[string]bool{"ready": true})
	}))
	defer srv.Close()
	fake := &credentialFake{value: credentialTestKey}
	ex, st, started, stopped := credentialExecutor(t, srv.URL, diagnosisServerPort(t, srv), fake)
	out, logs := credentialPublicOutput(t, ex)
	if st.running || st.proc != nil {
		t.Fatal("fixture was already running")
	}
	if err := ex.Start(t.Context(), "fake"); err != nil {
		t.Fatal(err)
	}
	if !st.running || !st.healthy || !st.adopted || fileExists(started) || fileExists(stopped) || fake.writes != 0 || fake.gets != 1 || requests.Load() == 0 {
		t.Fatal("first keyed adoption changed lifecycle or native state")
	}
	got, err := ex.LocalCredential("fake")
	if err != nil || got.Credential != credentialTestKey || got.Port != st.port {
		t.Fatal("adopted command key was not exported with its port")
	}
	found := false
	for _, frame := range out.frames() {
		if frame.Method == "engine:state-changed" {
			found = true
		}
	}
	if !found {
		t.Fatal("success fixture emitted no state notification")
	}
	assertCredentialNotPublic(t, ex, out, logs)
}

func TestCredentialUseStatusAtPortUsesCandidatePort(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer "+credentialTestKey {
			w.WriteHeader(http.StatusUnauthorized)
		}
	}))
	defer srv.Close()
	port := diagnosisServerPort(t, srv)
	fake := &credentialFake{value: credentialTestKey}
	ex, st, started, stopped := credentialExecutor(t, "http://127.0.0.1:{port}/", diagnosisRefusedPort(t), fake)
	if st.port == port {
		t.Fatal("fixture lacks stale port")
	}
	status, err := ex.StatusAtPort("fake", port)
	if err != nil || !status.Running || !status.Healthy || status.Port != port {
		t.Fatal("candidate port was not adopted")
	}
	got, err := ex.LocalCredential("fake")
	if err != nil || got.Port != port || got.Credential != credentialTestKey || fake.writes != 0 || fileExists(started) || fileExists(stopped) {
		t.Fatal("candidate export or lifecycle changed")
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
	ex, st, _, _ := credentialExecutor(t, srv.URL, port, &credentialFake{value: credentialTestKey})
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

func TestCredentialUseKnownAbsentAndRejectedDiagnosis(t *testing.T) {
	for _, configured := range []bool{false, true} {
		name := "absent"
		if configured {
			name = "saved key rejected"
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
			fake := &credentialFake{}
			want := diagnosisAuthRequired
			if configured {
				fake.value = credentialTestKey
				want = diagnosisAuthRejected
			}
			port := diagnosisServerPort(t, srv)
			ex, st, started, stopped := credentialExecutor(t, srv.URL, port, fake)
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

func TestCredentialUseUnavailableRetryWithoutRestart(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer "+credentialTestKey {
			w.WriteHeader(http.StatusUnauthorized)
		}
	}))
	defer srv.Close()
	port := diagnosisServerPort(t, srv)
	fake := &credentialFake{value: credentialTestKey, getErr: errors.New("locked native diagnostic")}
	ex, st, started, stopped := credentialExecutor(t, srv.URL, port, fake)
	if _, err := ex.credentialState(t.Context(), "fake"); err != errCredentialUnavailable {
		t.Fatal("unavailable claimed a state")
	}
	if _, err := ex.Status("fake"); err != errCredentialUnavailable {
		t.Fatal("presence swallowed unavailable")
	}
	if d, err := ex.diagnoseConnection(t.Context(), st, port); err != errCredentialUnavailable || d != diagnosisNone {
		t.Fatal("unavailable replaced by process diagnosis")
	}
	if err := ex.Start(t.Context(), "fake"); err != errCredentialUnavailable {
		t.Fatal("unavailable replaced by start diagnosis")
	}
	if st.running || st.healthy || fileExists(started) || fileExists(stopped) {
		t.Fatal("unavailable promoted state or ran commands")
	}
	fake.getErr = nil
	status, err := ex.Status("fake")
	if err != nil || !status.Healthy || !status.Running {
		t.Fatal("presence sweep did not retry availability")
	}
	if fake.gets < 5 || fake.writes != 0 {
		t.Fatal("availability retry repeated mutation or latched failure")
	}
}

func TestCredentialUseOwnedStrataPriorityAndStoppedRefusal(t *testing.T) {
	ex := newTestExecutor(t, testEngineManifest(fakeEngineBin))
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	fake := &credentialFake{value: "user-key"}
	ex.credentials = credentialStore{backend: fake}
	st.port = 45678
	st.plat.Runtime.UserCredential = true
	st.plat.Runtime.servingLaunch = &llamaServingLaunch{Credential: "owned-key"}
	st.proc = &managedProc{}
	c, err := ex.credentialForCandidate(t.Context(), st, st.port)
	if err != nil || c.key != "owned-key" || fake.gets != 0 {
		t.Fatal("owned prepared key lost priority before readiness")
	}
	if got, err := ex.LocalCredential("fake"); err != nil || got.Credential != "" {
		t.Fatal("stopped engine exported a candidate")
	}
	if _, err := ex.Action(t.Context(), "fake", "list_models", nil); err == nil {
		t.Fatal("stopped HTTP action accepted")
	}
	st.running = true
	if got, err := ex.LocalCredential("fake"); err != nil || got.Credential != "owned-key" {
		t.Fatal("running owned key unavailable")
	}
	st.proc = nil
	st.plat.Runtime.UserCredential = false
	if got, err := ex.LocalCredential("fake"); err != nil || got.Credential != "" {
		t.Fatal("adopted Strata exported prepared key")
	}
	if c, err := ex.credentialForCandidate(t.Context(), st, st.port); err != nil || c.key != "" {
		t.Fatal("unsupported engine selected a saved key")
	}
	if _, err := ex.setCredential(t.Context(), "fake", "key"); err != errInvalidCredential {
		t.Fatal("unsupported engine accepted a key")
	}
	if state, err := ex.credentialState(t.Context(), "fake"); err != nil || state != (engines.CredentialState{}) || fake.gets != 0 || fake.writes != 0 {
		t.Fatal("unsupported state touched native store")
	}
}

func TestCredentialUseExecutorRestartReloadsSharedStore(t *testing.T) {
	fake := &credentialFake{value: credentialTestKey}
	port := diagnosisRefusedPort(t)
	first, _, _, _ := credentialExecutor(t, "http://127.0.0.1:{port}/", port, fake)
	state, err := first.credentialState(t.Context(), "fake")
	if err != nil || !state.Configured {
		t.Fatal("first executor failed to load")
	}
	first.credentials.close()
	second, _, _, _ := credentialExecutor(t, "http://127.0.0.1:{port}/", port, fake)
	state, err = second.credentialState(t.Context(), "fake")
	if err != nil || !state.Configured || fake.gets != 2 || fake.writes != 0 {
		t.Fatal("new executor did not reload durable state")
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
	keyed, st, _, _ := credentialExecutor(t, srv.URL, port, &credentialFake{value: credentialTestKey})
	if _, err := keyed.credentialForCandidate(t.Context(), st, 0); err != errCredentialRequest {
		t.Fatal("invalid candidate port accepted")
	}
}

func TestCredentialUseUnkeyedActionPreservesPortAndTransport(t *testing.T) {
	for _, supported := range []bool{false, true} {
		t.Run(fmt.Sprintf("credential_supported_%t", supported), func(t *testing.T) {
			m := testEngineManifest(fakeEngineBin)
			m.Actions["read"] = Action{HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/read"}}
			ex := newTestExecutor(t, m)
			st, err := ex.state(m.Engine)
			if err != nil {
				t.Fatal(err)
			}
			st.mu.Lock()
			st.port, st.running = 0, true
			st.plat.Runtime.UserCredential = supported
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
		})
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
	ex, st, _, _ := credentialExecutor(t, source.URL, port, &credentialFake{value: credentialTestKey})
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
	ex, st, _, _ := credentialExecutor(t, srv.URL, port, &credentialFake{value: credentialTestKey})
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
			ex, st, _, _ := credentialExecutor(t, srv.URL, diagnosisServerPort(t, srv), &credentialFake{value: credentialTestKey})
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

func TestCredentialUsePrivateRPCSanitizedStateAndValidation(t *testing.T) {
	fake := &credentialFake{}
	ex, _, _, _ := credentialExecutor(t, "http://127.0.0.1:{port}/", diagnosisRefusedPort(t), fake)
	unsupported := testEngineManifest(fakeEngineBin)
	unsupported.Engine = "unsupported"
	ex.reg.engines[unsupported.Engine] = unsupported
	logs := &syncBuffer{}
	oldLogger := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(logs, nil)))
	defer slog.SetDefault(oldLogger)
	for _, tc := range []struct {
		name, method, params string
		code                 int
		configured           bool
	}{
		{"set", engines.SetCredentialMethod, `{"engine":"fake","credential":"test-key-123"}`, 0, true},
		{"state", engines.CredentialStateMethod, `{"engine":"fake"}`, 0, true},
		{"node target", engines.SetCredentialMethod, `{"engine":"fake","credential":"test-key-123","node":null}`, -32602, false},
		{"nodeId target", engines.CredentialStateMethod, `{"engine":"fake","nodeId":"remote"}`, -32602, false},
		{"invalid name", engines.SetCredentialMethod, `{"engine":"../fake","credential":"test-key-123"}`, -32602, false},
		{"invalid credential", engines.SetCredentialMethod, `{"engine":"fake","credential":"bad key"}`, -32602, false},
		{"missing credential", engines.SetCredentialMethod, `{"engine":"fake"}`, -32602, false},
		{"null credential", engines.SetCredentialMethod, `{"engine":"fake","credential":null}`, -32602, false},
		{"null engine", engines.CredentialStateMethod, `{"engine":null}`, -32602, false},
		{"unknown field", engines.CredentialStateMethod, `{"engine":"fake","credential":"test-key-123"}`, -32602, false},
		{"unsupported set", engines.SetCredentialMethod, `{"engine":"unsupported","credential":"test-key-123"}`, -32602, false},
		{"unsupported state", engines.CredentialStateMethod, `{"engine":"unsupported"}`, 0, false},
		{"unknown engine", engines.CredentialStateMethod, `{"engine":"unknown"}`, -32602, false},
		{"clear", engines.SetCredentialMethod, `{"engine":"fake","credential":""}`, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := &syncBuffer{}
			manager := NewManager(NewCodec(out), ex, nil, t.TempDir())
			id := json.RawMessage("1")
			manager.handleMessage(t.Context(), &Message{JSONRPC: "2.0", ID: &id, Method: tc.method, Params: json.RawMessage(tc.params)})
			response := credentialWaitReply(t, out)
			if tc.code != 0 {
				if response.Error == nil || response.Error.Code != tc.code || response.Error.Message != errInvalidCredential.Error() {
					t.Fatal("invalid RPC lacked fixed error")
				}
			} else {
				var state engines.CredentialState
				if response.Error != nil || json.Unmarshal(response.Result, &state) != nil || state.Supported != (tc.name != "unsupported state") || state.Configured != tc.configured {
					t.Fatal("private RPC state is incorrect")
				}
			}
			if strings.Contains(credentialOutput(out), credentialTestKey) {
				t.Fatal("RPC reply exposed credential")
			}
		})
	}
	if fake.writes != 2 {
		t.Fatal("invalid params touched durable store")
	}
	encoded, err := json.Marshal(ex.GetInstalled())
	if err != nil {
		t.Fatal(err)
	}
	lines, err := ex.Logs("fake")
	if err != nil {
		t.Fatal(err)
	}
	public, err := json.Marshal(struct {
		Status json.RawMessage
		Errors []serviceError
		Logs   []LogLine
	}{encoded, ex.Errors(), lines})
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(public, []byte(credentialTestKey)) || strings.Contains(credentialOutput(logs), credentialTestKey) {
		t.Fatal("public state or logs exposed credential")
	}
}

func credentialWaitReply(t *testing.T, out *syncBuffer) Message {
	t.Helper()
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	deadline := time.After(3 * time.Second)
	for {
		var response Message
		if json.Unmarshal([]byte(credentialOutput(out)), &response) == nil && response.ID != nil {
			return response
		}
		select {
		case <-deadline:
			t.Fatal("private RPC response did not arrive")
		case <-ticker.C:
		}
	}
}

func credentialOutput(out *syncBuffer) string {
	out.mu.Lock()
	defer out.mu.Unlock()
	return out.buf.String()
}

func TestCredentialUseUnavailableRPCNeverClaimsAbsent(t *testing.T) {
	fake := &credentialFake{getErr: errors.New("native diagnostic " + credentialTestKey), writeErr: errors.New("native diagnostic " + credentialTestKey)}
	ex, _, _, _ := credentialExecutor(t, "http://127.0.0.1:{port}/", diagnosisRefusedPort(t), fake)
	for _, method := range []string{engines.SetCredentialMethod, engines.CredentialStateMethod} {
		out := &syncBuffer{}
		manager := NewManager(NewCodec(out), ex, nil, t.TempDir())
		id := json.RawMessage("1")
		params := json.RawMessage(`{"engine":"fake"}`)
		if method == engines.SetCredentialMethod {
			params = json.RawMessage(`{"engine":"fake","credential":"test-key-123"}`)
		}
		manager.handleMessage(t.Context(), &Message{ID: &id, Method: method, Params: params})
		response := credentialWaitReply(t, out)
		if response.Error == nil || response.Error.Code != -32000 || response.Error.Message != errCredentialUnavailable.Error() || len(response.Result) != 0 || strings.Contains(credentialOutput(out), credentialTestKey) {
			t.Fatal("unavailable RPC exposed native diagnostics or claimed absence")
		}
	}
}

// Keep the fake CLI environment entirely separate from the saved API key.
func TestCredentialUseCLIActionReceivesNoSavedKey(t *testing.T) {
	ex, st, _, _ := credentialExecutor(t, "http://127.0.0.1:{port}/", diagnosisRefusedPort(t), &credentialFake{value: credentialTestKey})
	if _, err := ex.credentialState(t.Context(), "fake"); err != nil {
		t.Fatal(err)
	}
	helper, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	st.manifest.Actions["echo"] = Action{Cmd: []string{helper, "credential-cli-helper"}}
	result, err := ex.Action(t.Context(), "fake", "echo", nil)
	if err != nil || bytes.Contains(result, []byte(credentialTestKey)) {
		t.Fatal("CLI action received saved credential")
	}
	var response struct {
		Present *bool `json:"credentialPresent"`
	}
	if json.Unmarshal(result, &response) != nil || response.Present == nil || *response.Present {
		t.Fatal("saved credential appeared in CLI argv/environment")
	}
}

func TestCredentialUseAmbiguousMutationRequiresFreshEligibility(t *testing.T) {
	fake := &credentialFake{value: credentialTestKey}
	ex, st, _, _ := credentialExecutor(t, "http://127.0.0.1:{port}/", diagnosisRefusedPort(t), fake)
	st.running = true
	old, err := ex.credentialForCandidate(t.Context(), st, st.port)
	if err != nil {
		t.Fatal(err)
	}
	fake.writeErr = errors.New("lost reply")
	if _, err := ex.setCredential(t.Context(), "fake", "replacement"); err != errCredentialUnavailable {
		t.Fatal("ambiguous mutation reported success")
	}
	fake.getErr = errors.New("locked")
	if c, err := ex.credentialForCandidate(t.Context(), st, st.port); err != errCredentialUnavailable || c.key != "" {
		t.Fatal("failed reconciliation exposed stale candidate")
	}
	if got, err := ex.LocalCredential("fake"); err != errCredentialUnavailable || got.Credential != "" {
		t.Fatal("failed reconciliation exported stale key")
	}
	fake.getErr = nil
	got, err := ex.LocalCredential("fake")
	if err != nil || got.Credential != "replacement" || fake.writes != 1 || old.key != credentialTestKey {
		t.Fatal("fresh eligibility did not reconcile durable write or modified in-flight snapshot")
	}
}

func TestCredentialUseHealthRetriesUnavailableStore(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer "+credentialTestKey {
			w.WriteHeader(http.StatusUnauthorized)
		}
	}))
	defer srv.Close()
	port := diagnosisServerPort(t, srv)
	fake := &credentialFake{value: credentialTestKey, getErr: errors.New("locked")}
	ex, st, _, _ := credentialExecutor(t, srv.URL, port, fake)
	st.running, st.healthy, st.gen = true, true, 1
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan struct{})
	go func() { defer close(done); ex.runHealth(ctx, st, "fake", port, 1) }()
	defer func() { cancel(); <-done }()
	report := waitDiagnosisReport(t, ex, unhealthyID("fake"))
	if report.Message != errCredentialUnavailable.Error() {
		t.Fatal("health replaced store unavailable with process diagnosis")
	}
	fake.lock.Lock()
	fake.getErr = nil
	fake.lock.Unlock()
	deadline := time.After(4 * time.Second)
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	for {
		st.mu.Lock()
		healthy := st.healthy
		st.mu.Unlock()
		if healthy && len(ex.reporter.snapshot()) == 0 {
			break
		}
		select {
		case <-deadline:
			t.Fatal("health did not retry and clear unavailable")
		case <-ticker.C:
		}
	}
}

func TestCredentialUseTransportFailuresAreSanitized(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()
	port := diagnosisServerPort(t, srv)
	ex, st, _, _ := credentialExecutor(t, srv.URL, port, &credentialFake{value: credentialTestKey})
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

type credentialReadBarrier struct {
	credentialFake
	state            *engineState
	entered, release chan struct{}
	stateUnlocked    bool
}

func (f *credentialReadBarrier) Get(ctx context.Context, purpose, id string) ([]byte, error) {
	if f.state.mu.TryLock() {
		f.stateUnlocked = true
		f.state.mu.Unlock()
	}
	close(f.entered)
	<-f.release
	return f.credentialFake.Get(ctx, purpose, id)
}

func TestCredentialUseRunningEligibilitySurvivesNativeWait(t *testing.T) {
	for _, scenario := range []string{"stopped export", "replacement run action"} {
		t.Run(scenario, func(t *testing.T) {
			var requests atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodGet || r.URL.Path != "/" {
					http.NotFound(w, r)
					return
				}
				requests.Add(1)
			}))
			defer srv.Close()
			ex, st, _, _ := credentialExecutor(t, srv.URL, diagnosisServerPort(t, srv), nil)
			st.running, st.runID = true, "before"
			store := &credentialReadBarrier{credentialFake: credentialFake{value: credentialTestKey}, state: st, entered: make(chan struct{}), release: make(chan struct{})}
			ex.credentials = credentialStore{backend: store}
			type result struct {
				credential engines.LocalCredential
				err        error
			}
			done := make(chan result, 1)
			go func() {
				if scenario == "stopped export" {
					credential, err := ex.LocalCredential("fake")
					done <- result{credential: credential, err: err}
				} else {
					_, err := ex.Action(t.Context(), "fake", "list_models", nil)
					done <- result{err: err}
				}
			}()
			<-store.entered
			if !store.stateUnlocked {
				close(store.release)
				<-done
				t.Fatal("native I/O held engine state mutex")
			}
			st.mu.Lock()
			if scenario == "stopped export" {
				st.running = false
			} else {
				st.runID = "after"
			}
			st.mu.Unlock()
			close(store.release)
			got := <-done
			if got.credential.Credential != "" || requests.Load() != 0 {
				t.Fatal("stale run credential escaped after native wait")
			}
			if scenario == "stopped export" && got.err != nil {
				t.Fatal("stopped export did not return empty credential")
			}
			if scenario == "replacement run action" && got.err == nil {
				t.Fatal("replacement run accepted a stale action")
			}
		})
	}
}
