// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestActionBodyReadFailureIsTransportError(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	m.Actions["read"] = Action{HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/read"}}
	ex := newTestExecutor(t, m)
	st, err := ex.state(m.Engine)
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.running = true
	st.mu.Unlock()
	ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: readbackBrokenBody{strings.NewReader(`{"ok":true}`)}}, nil
	})
	result, err := ex.Action(context.Background(), m.Engine, "read", nil)
	if !errors.Is(err, io.ErrUnexpectedEOF) || result != nil {
		t.Fatalf("result=%s err=%v, want no result and wrapped body read error", result, err)
	}
	var status *ActionHTTPError
	if errors.As(err, &status) {
		t.Fatal("body read error was classified as HTTP status failure")
	}
}

func TestActionObservationRequiresHTTP200(t *testing.T) {
	for _, action := range []string{"list_models", "loaded_models", "load_model"} {
		t.Run(action, func(t *testing.T) {
			for _, status := range []int{http.StatusOK, http.StatusCreated, http.StatusNoContent} {
				t.Run(http.StatusText(status), func(t *testing.T) {
					m := testEngineManifest(fakeEngineBin)
					m.Actions[action] = Action{HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/read"}}
					ex := newTestExecutor(t, m)
					st, err := ex.state(m.Engine)
					if err != nil {
						t.Fatal(err)
					}
					st.mu.Lock()
					st.running = true
					st.mu.Unlock()
					ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
						return &http.Response{StatusCode: status, Body: http.NoBody}, nil
					})
					result, err := ex.Action(context.Background(), m.Engine, action, nil)
					if action == "load_model" || status == http.StatusOK {
						if err != nil || string(result) != "null" {
							t.Fatalf("result=%s err=%v, want successful empty action", result, err)
						}
						return
					}
					var failure *ActionHTTPError
					if !errors.As(err, &failure) || failure.StatusCode != status || result != nil {
						t.Fatalf("result=%s err=%v, want HTTP %d observation failure", result, err, status)
					}
				})
			}
		})
	}
}

func TestActionHTTPErrorPreservesDiagnostic(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		status     int
	}{
		{"trimmed JSON", " \t\r\n{\"error\":\"busy\"}\n ", http.StatusConflict},
		{"empty body", "", http.StatusNotFound},
		{"Unicode whitespace", "\u2003failure\u2003", http.StatusBadRequest},
		{"bounded excerpt", strings.Repeat("x", 8*1024*1024) + "omitted", http.StatusBadGateway},
	} {
		t.Run(tc.name, func(t *testing.T) {
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
				// The runner intentionally closes after its excerpt limit.
				if _, err := io.WriteString(w, tc.body); err != nil && len(tc.body) <= 8*1024*1024 {
					t.Errorf("write error body: %v", err)
				}
			}))
			if err := server.Listener.Close(); err != nil {
				listener.Close()
				t.Fatal(err)
			}
			server.Listener = listener
			server.Start()
			defer server.Close()
			addr, ok := listener.Addr().(*net.TCPAddr)
			if !ok {
				t.Fatalf("listener address type = %T", listener.Addr())
			}
			m := testEngineManifest(fakeEngineBin)
			const action = "quoted\"action"
			m.Actions[action] = Action{HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/failure"}}
			ex := newTestExecutor(t, m)
			st, err := ex.state(m.Engine)
			if err != nil {
				t.Fatal(err)
			}
			st.running, st.port = true, addr.Port
			_, err = ex.Action(context.Background(), m.Engine, action, nil)
			var failure *ActionHTTPError
			if !errors.As(err, &failure) {
				t.Fatalf("error = %T %v, want ActionHTTPError", err, err)
			}
			bounded := tc.body[:min(len(tc.body), 8*1024*1024)]
			wantBody := strings.TrimSpace(bounded)
			want := fmt.Sprintf("action %q: engine returned HTTP %d: %s", action, tc.status, wantBody)
			if err.Error() != want {
				t.Fatalf("diagnostic differs from the original format (got %d bytes, want %d)", len(err.Error()), len(want))
			}
			if failure.Action != action || failure.StatusCode != tc.status || failure.Body != wantBody {
				t.Fatal("typed error did not preserve the action, HTTP status and bounded excerpt")
			}
			var wrapped *ActionHTTPError
			if !errors.As(fmt.Errorf("operation: %w", err), &wrapped) || wrapped != failure {
				t.Fatal("errors.As did not preserve the HTTP failure through wrapping")
			}
		})
	}
}
