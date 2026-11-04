// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
)

func probeRefusedPort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("allocate refused port: %v", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	if err := l.Close(); err != nil {
		t.Fatalf("close refused listener: %v", err)
	}
	return port
}

func probeServerPort(t *testing.T, srv *httptest.Server) int {
	t.Helper()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("parse fake server URL: %v", err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatalf("parse fake server port: %v", err)
	}
	return port
}

func TestProbeStatusPreservesIdentity(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		body   string
		want   bool
	}{
		{"matching JSON", http.StatusOK, `{"role":"router"}`, true},
		{"wrong identity", http.StatusOK, `{"role":"model"}`, false},
		{"invalid JSON", http.StatusOK, `{`, false},
		{"auth with invalid JSON", http.StatusUnauthorized, `{`, false},
		{"oversized JSON", http.StatusOK, strings.Repeat(" ", maxProbeJSONBytes+1), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var identityHeader atomic.Bool
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodGet || r.URL.Path != "/" {
					http.NotFound(w, r)
					return
				}
				identityHeader.Store(r.Header.Get(engineIdentityProbeHeader) == "1")
				w.WriteHeader(tc.status)
				if _, err := w.Write([]byte(tc.body)); err != nil {
					t.Errorf("write fake probe response: %v", err)
				}
			}))
			defer srv.Close()
			port := probeServerPort(t, srv)
			ex := newTestExecutor(t, testEngineManifest(fakeEngineBin))
			p := &Probe{HTTP: srv.URL, Status: http.StatusOK, TimeoutS: 1, JSONMatch: &ProbeJSONMatch{Field: "role", Value: "router"}}
			ok, status, err := ex.probeStatus(context.Background(), p, port, candidateCredential{port: port})
			if err != nil {
				t.Fatal(err)
			}
			if ok != tc.want || status != tc.status || !identityHeader.Load() {
				t.Fatalf("probe = (%v, %d), identity header = %v; want (%v, %d), header true", ok, status, identityHeader.Load(), tc.want, tc.status)
			}
			got, err := ex.probe(context.Background(), p, port, candidateCredential{port: port})
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("boolean probe = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestProbeStatusWithoutHTTPResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	port := probeServerPort(t, srv)
	ex := newTestExecutor(t, testEngineManifest(fakeEngineBin))
	refused := probeRefusedPort(t)
	for _, tc := range []struct {
		name  string
		probe *Probe
		want  bool
	}{
		{"no probe", nil, true},
		{"TCP reachable", &Probe{TCP: fmt.Sprintf("127.0.0.1:%d", port)}, true},
		{"TCP refused", &Probe{TCP: fmt.Sprintf("127.0.0.1:%d", refused)}, false},
		{"HTTP refused", &Probe{HTTP: fmt.Sprintf("http://127.0.0.1:%d/", refused)}, false},
		{"unresolved placeholder", &Probe{HTTP: "http://127.0.0.1:{missing}/"}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ok, status, err := ex.probeStatus(context.Background(), tc.probe, port, candidateCredential{port: port})
			if err != nil {
				t.Fatal(err)
			}
			if ok != tc.want || status != 0 {
				t.Fatalf("probe = (%v, %d), want (%v, 0)", ok, status, tc.want)
			}
		})
	}
}
