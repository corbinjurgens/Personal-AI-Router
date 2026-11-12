// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
)

func mustRemoteSettingJSON(t *testing.T, value any) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestRemoteSettingLockedExactWireClassification(t *testing.T) {
	if codeRemoteSettingLocked != -32014 || wireCodes[-32014] != "remote_setting_locked" {
		t.Fatal("remote setting lock numeric/wire assignment changed")
	}
	test := func(name string, status int, body string, wantCode int) {
		t.Run(name, func(t *testing.T) {
			err := remoteStatusError("remote node policy", status, []byte(body))
			if got := rpcCode(fmt.Errorf("wrapped: %w", err)); got != wantCode {
				t.Fatalf("RPC code = %d, want %d", got, wantCode)
			}
		})
	}
	test("typed lock with unrelated text", http.StatusConflict, `{"code":"remote_setting_locked","error":"invalid params: unrelated diagnostic"}`, -32014)
	test("diagnostic alone", http.StatusConflict, `inference settings are locked on the target device`, -32000)
	test("text in unknown envelope", http.StatusConflict, `{"code":"unknown","error":"remote_setting_locked"}`, -32000)
	test("case differs", http.StatusConflict, `{"code":"REMOTE_SETTING_LOCKED","error":"locked"}`, -32000)
	test("numeric wire value", http.StatusConflict, `{"code":-32014,"error":"locked"}`, -32000)
	test("wrong HTTP status", http.StatusForbidden, `{"code":"remote_setting_locked","error":"locked"}`, -32000)
	for _, c := range []struct {
		wire string
		code int
	}{
		{"policy_conflict", -32010},
		{"local_veto", -32011},
		{"remote_management_denied", -32012},
	} {
		test(c.wire, http.StatusConflict, string(mustRemoteSettingJSON(t, clustertrust.CodedRefusal{Code: c.wire, Error: "inference settings are locked on the target device"})), c.code)
	}
}

func TestRemoteClientModelFilesUseOrdinaryHeaderBudget(t *testing.T) {
	test := func(name string, call func(context.Context, *remoteClient) error) {
		t.Run(name, func(t *testing.T) {
			release := make(chan struct{})
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if !(r.Method == http.MethodGet && (r.URL.Path == controlModelFilesPath || r.URL.Path == controlModelFilePath)) {
					http.NotFound(w, r)
					return
				}
				select {
				case <-release:
				case <-r.Context().Done():
				}
			}))
			defer func() {
				close(release)
				srv.Close()
			}()
			base := http.DefaultTransport.(*http.Transport)
			client := &remoteClient{
				controlHTTP:      newRemoteHTTPClient(base, 20*time.Millisecond),
				controlReadyHTTP: newRemoteHTTPClient(base, time.Second),
				base:             srv.URL,
			}
			ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
			defer cancel()
			err := call(ctx, client)
			if err == nil || !strings.Contains(err.Error(), "timeout awaiting response headers") {
				t.Fatalf("model request error = %v, want ordinary header timeout", err)
			}
		})
	}
	test("listing", func(ctx context.Context, client *remoteClient) error {
		_, err := client.modelFiles(ctx, "llamacpp", "tiny", nil)
		return err
	})
	test("download", func(ctx context.Context, client *remoteClient) error {
		body, _, err := client.modelFile(ctx, "llamacpp", "tiny", "fixture.gguf", "", 0)
		if body != nil {
			err = errors.Join(err, body.Close())
		}
		return err
	})
}

// A manifest-bound cold load and a remote delete use the longer response-header
// budget; other model calls remain bounded.
func TestRemoteSlowModelOperationsUseReadinessHeaderBudget(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !(r.Method == http.MethodPost && (r.URL.Path == controlLoadPath || r.URL.Path == controlDeletePath || r.URL.Path == controlUnloadPath)) {
			http.NotFound(w, r)
			return
		}
		time.Sleep(100 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	manifest := llamaTestEngineManifest(fakeEngineBin)
	manifest.Engine = "custom-load"
	manifest.Actions["load_model"] = Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/models/load"}}
	manifest.Operations = map[string]OperationBinding{
		"load": {Action: "load_model", ResponseTimeoutS: 60},
	}
	if err := manifest.Validate(); err != nil {
		t.Fatalf("invalid load-budget fixture: %v", err)
	}
	ex := newTestExecutor(t, manifest)
	base := http.DefaultTransport.(*http.Transport)
	c := &remoteClient{
		controlHTTP:      newRemoteHTTPClient(base, 20*time.Millisecond),
		controlReadyHTTP: newRemoteHTTPClient(base, time.Second),
		base:             srv.URL,
		loadBudget:       ex.loadResponseBudget,
	}
	slow := modelActionRequest{Engine: "custom-load", Model: "fixture-model"}
	if _, err := c.postJSON(context.Background(), controlLoadPath, slow.Engine, slow); err != nil {
		t.Fatalf("remote load was cut off by the ordinary response-header budget: %v", err)
	}
	ordinary := modelActionRequest{Engine: "llamacpp", Model: "fixture-model"}
	if _, err := c.postJSON(context.Background(), controlDeletePath, ordinary.Engine, ordinary); err != nil {
		t.Fatalf("remote delete was cut off by the ordinary response-header budget: %v", err)
	}
	if _, err := c.postJSON(context.Background(), controlLoadPath, ordinary.Engine, ordinary); err == nil ||
		!strings.Contains(err.Error(), "timeout awaiting response headers") {
		t.Fatalf("remote load error = %v, want the ordinary bounded response-header timeout", err)
	}
	if _, err := c.postJSON(context.Background(), controlUnloadPath, slow.Engine, slow); err == nil ||
		!strings.Contains(err.Error(), "timeout awaiting response headers") {
		t.Fatalf("remote unload error = %v, want the ordinary bounded response-header timeout", err)
	}
}

func TestRemoteReadinessBudgetCoversEngineStartupAllowance(t *testing.T) {
	if remoteReadyResponseHeaderTimeout <= remoteResponseHeaderTimeout {
		t.Fatalf("readiness budget %s must exceed the ordinary budget %s",
			remoteReadyResponseHeaderTimeout, remoteResponseHeaderTimeout)
	}
	ex := bundledTestExecutor(t)
	cases := []struct {
		path   string
		engine string
		want   bool
	}{
		{controlDeletePath, "llamacpp", true},
		{controlLoadPath, "llamacpp", false},
		{controlLoadPath, "peer-only", false},
		{controlUnloadPath, "llamacpp", false},
		{controlEnginesPath, "", false},
	}
	for _, tc := range cases {
		if got := waitsForEngineReadiness(tc.path, ex.loadResponseBudget(tc.engine)); got != tc.want {
			t.Errorf("waitsForEngineReadiness(%q, budget of %q) = %v, want %v", tc.path, tc.engine, got, tc.want)
		}
	}
}

func TestRemoteDeleteHeaderWaitRemainsBounded(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !(r.Method == http.MethodPost && r.URL.Path == controlDeletePath) {
			http.NotFound(w, r)
			return
		}
		<-release
		_, _ = w.Write([]byte(`{"engine":"llamacpp"}`))
	}))
	defer func() {
		close(release)
		srv.Close()
	}()

	base := http.DefaultTransport.(*http.Transport)
	c := &remoteClient{
		controlHTTP:      newRemoteHTTPClient(base, time.Second),
		controlReadyHTTP: newRemoteHTTPClient(base, 50*time.Millisecond),
		base:             srv.URL,
	}
	started := time.Now()
	_, err := c.postJSON(context.Background(), controlDeletePath, "llamacpp", modelActionRequest{Engine: "llamacpp", Model: "tiny"})
	if err == nil || !strings.Contains(err.Error(), "timeout awaiting response headers") {
		t.Fatalf("remote delete error = %v, want bounded response-header timeout", err)
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("remote delete took %s, want a prompt bounded failure", elapsed)
	}
}

// rmTestClient is a remoteClient pointed at a plain httptest server: these tests
// exercise the wire mapping, not the pin gate.
func rmTestClient(srv *httptest.Server) *remoteClient {
	base := http.DefaultTransport.(*http.Transport)
	return &remoteClient{
		controlHTTP:      newRemoteHTTPClient(base, time.Second),
		controlReadyHTTP: newRemoteHTTPClient(base, time.Second),
		base:             srv.URL,
	}
}

func TestRemoteManagementGetReportsOldTargetUnsupported(t *testing.T) {
	for _, status := range []int{http.StatusNotFound, http.StatusMethodNotAllowed} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				http.Error(w, "no such route", status)
			}))
			defer srv.Close()

			capability, err := rmTestClient(srv).getRemoteManagement(context.Background())
			if err != nil {
				t.Fatalf("old target read error = %v, want unsupported without an error", err)
			}
			if capability.State != remoteManagementStateUnsupported || capability.Allowed {
				t.Fatalf("old target capability = %+v, want unsupported and not allowed", capability)
			}
		})
	}
}

func TestRemoteManagementGetPreservesTypedDenial(t *testing.T) {
	cases := []struct {
		name string
		body string
		code int
	}{
		{name: "remote management denied", body: `{"error":"` + clustertrust.RemoteManagementDeniedMessage + `","code":"remote_management_denied"}`, code: codeRemoteManagementDenied},
		{name: "local veto stays distinct", body: `{"error":"this node was paused on the node itself; resume it there","code":"local_veto"}`, code: codeLocalVeto},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusConflict)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer srv.Close()

			_, err := rmTestClient(srv).getRemoteManagement(context.Background())
			var coded *codedError
			if !errors.As(err, &coded) || coded.code != tc.code {
				t.Fatalf("permission read error = %v (code %d), want coded %d", err, rpcCode(err), tc.code)
			}
			if coded.message == "" {
				t.Fatal("typed refusal carried no message")
			}
		})
	}
}

func TestRemoteManagementGetRejectsInconsistentOrUnknownBody(t *testing.T) {
	cases := []struct {
		name    string
		body    string
		wantErr bool
		allowed bool
		state   string
	}{
		{
			// An inconsistent target answer never becomes an allow.
			name: "allowed flag with a denied state", body: `{"version":1,"allowed":true,"revision":"3","state":"denied"}`,
			allowed: false, state: remoteManagementStateDenied,
		},
		{
			name: "unrecognized version", body: `{"version":9,"allowed":true,"revision":"3","state":"allowed"}`,
			wantErr: true,
		},
		{
			name: "unrecognized state", body: `{"version":1,"allowed":true,"revision":"3","state":"maybe"}`,
			wantErr: true,
		},
		{
			name: "not an object", body: `[]`,
			wantErr: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(tc.body))
			}))
			defer srv.Close()

			capability, err := rmTestClient(srv).getRemoteManagement(context.Background())
			if tc.wantErr {
				if err == nil {
					t.Fatalf("unrecognized body %s was accepted: %+v", tc.body, capability)
				}
				if capability.Allowed {
					t.Fatalf("unrecognized body %s produced an allow", tc.body)
				}
				return
			}
			if err != nil {
				t.Fatalf("body %s error = %v", tc.body, err)
			}
			if capability.Allowed != tc.allowed || capability.State != tc.state {
				t.Fatalf("capability = %+v, want allowed=%v state=%q", capability, tc.allowed, tc.state)
			}
		})
	}
}

func TestRemoteManagementTypedDenialSurvivesRegularAndStreamingClients(t *testing.T) {
	deniedBody := `{"error":"` + clustertrust.RemoteManagementDeniedMessage + `","code":"remote_management_denied"}`

	// Regular (non-streaming) POST: remoteStatusError keeps the typed code.
	t.Run("regular", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusConflict)
			_, _ = w.Write([]byte(deniedBody))
		}))
		defer srv.Close()

		_, err := rmTestClient(srv).postJSON(context.Background(), controlLoadPath, "fixture", modelActionRequest{Engine: "fixture", Model: "tiny"})
		if code := rpcCode(err); code != codeRemoteManagementDenied {
			t.Fatalf("regular typed error code = %d, want %d (%v)", code, codeRemoteManagementDenied, err)
		}
	})

	// Streaming preflight: a typed 409 before any NDJSON header keeps the code.
	t.Run("stream preflight", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusConflict)
			_, _ = w.Write([]byte(deniedBody))
		}))
		defer srv.Close()

		_, err := rmTestClient(srv).stream(context.Background(), controlInstallPath, map[string]string{"engine": "fixture"}, nil)
		if code := rpcCode(err); code != codeRemoteManagementDenied {
			t.Fatalf("stream preflight code = %d, want %d (%v)", code, codeRemoteManagementDenied, err)
		}
	})

	// A stream error frame carrying the wire name keeps the code too.
	t.Run("stream error frame", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = w.Write([]byte(`{"type":"error","message":"` + clustertrust.RemoteManagementDeniedMessage + `","code":"remote_management_denied"}` + "\n"))
		}))
		defer srv.Close()

		_, err := rmTestClient(srv).stream(context.Background(), controlInstallPath, map[string]string{"engine": "fixture"}, nil)
		if code := rpcCode(err); code != codeRemoteManagementDenied {
			t.Fatalf("stream error frame code = %d, want %d (%v)", code, codeRemoteManagementDenied, err)
		}
	})

	// The code is never confusable with the local veto or with generic failure.
	if rpcCode(remoteStatusError("relay", http.StatusConflict, []byte(deniedBody))) != codeRemoteManagementDenied {
		t.Fatal("remoteStatusError did not preserve the remote-management code")
	}
	if code, _ := codeOfWire(clustertrust.RemoteManagementDeniedCode); code != codeRemoteManagementDenied {
		t.Fatalf("wire name %q does not resolve to %d", clustertrust.RemoteManagementDeniedCode, codeRemoteManagementDenied)
	}
	if code, _ := codeOfWire("local_veto"); code == codeRemoteManagementDenied {
		t.Fatal("local_veto resolves to the remote-management code")
	}
}
