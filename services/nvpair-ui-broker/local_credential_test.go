// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"nvpair-shared/engines"
)

const brokerTestKey = "FAKE-BROKER-KEY-0123456789"

func TestLocalCredentialMethodIsRefusedToClients(t *testing.T) {
	b, out := newPolicyBroker(t)
	engine := attachFakeEngine(t, b)
	b.handleMessage(clientRequest(1, engines.LocalCredentialMethod, map[string]string{"engine": "strata"}))
	response := out.waitFrame(t, engines.LocalCredentialMethod, isResponse(1))
	if response.Error == nil || response.Error.Code != -32601 || len(engine.snapshot()) != 0 {
		t.Fatal("a public client reached the native-key method")
	}
}

// The key is used only when engine-manager pairs it with the port being
// configured; any other answer leaves the backend without one.
func TestLocalEngineCredentialPairsWithPort(t *testing.T) {
	for _, tc := range []struct {
		name   string
		answer any
		errMsg string
		want   string
	}{
		{"same port", engines.LocalCredential{Credential: brokerTestKey, Port: 4321}, "", brokerTestKey},
		{"other port", engines.LocalCredential{Credential: brokerTestKey, Port: 4322}, "", ""},
		{"no key", engines.LocalCredential{}, "", ""},
		{"engine-manager error", nil, "unknown engine", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, _ := newPolicyBroker(t)
			engine := attachFakeEngine(t, b)
			engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
				if method != engines.LocalCredentialMethod {
					return map[string]bool{"ok": true}, ""
				}
				return tc.answer, tc.errMsg
			})
			if got := b.localEngineCredential("strata", 4321); got != tc.want {
				t.Fatalf("credential = %q, want %q", got, tc.want)
			}
		})
	}
	t.Run("no engine-manager", func(t *testing.T) {
		b, _ := newPolicyBroker(t)
		if got := b.localEngineCredential("strata", 4321); got != "" {
			t.Fatalf("credential = %q without engine-manager", got)
		}
	})
}

// An authenticated engine is healthy only with the key, and the probe does
// not follow a redirect with it.
func TestEngineHealthProbeCarriesTheKey(t *testing.T) {
	var landed atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			if r.Header.Get("Authorization") != "Bearer "+brokerTestKey {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			_, _ = w.Write([]byte(`{"object":"list","data":[]}`))
		case "/redirect":
			http.Redirect(w, r, "/landing", http.StatusFound)
		case "/landing":
			landed.Add(1)
		}
	}))
	t.Cleanup(server.Close)
	port := server.Listener.Addr().(*net.TCPAddr).Port
	profile := engineProxyProfile{HealthProbePath: "/v1/models"}
	if checkEngineHealth(profile, server.Client(), port, "") {
		t.Fatal("authenticated engine healthy without the key")
	}
	if !checkEngineHealth(profile, server.Client(), port, brokerTestKey) {
		t.Fatal("authenticated engine unhealthy with the key")
	}
	if checkEngineHealth(engineProxyProfile{HealthProbePath: "/redirect"}, server.Client(), port, brokerTestKey) || landed.Load() != 0 {
		t.Fatalf("redirect followed with the key: landing hits %d", landed.Load())
	}
}
