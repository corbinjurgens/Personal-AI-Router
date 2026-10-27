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

const brokerTestKey = "test-key-123"

func TestLocalCredentialMethodIsRefusedToClients(t *testing.T) {
	if _, refused := internalClientMethods[engines.LocalCredentialMethod]; !refused {
		t.Fatal("retrieval is missing from the pre-relay internal-method refusal")
	}
	test := func(name string, params any) {
		t.Run(name, func(t *testing.T) {
			b, out := newPolicyBroker(t)
			engine := attachFakeEngine(t, b)
			b.handleMessage(clientRequest(1, engines.LocalCredentialMethod, params))
			response := out.waitFrame(t, "private method refusal", isResponse(1))
			if response.Error == nil || response.Error.Code != -32601 || len(response.Result) != 0 {
				t.Fatal("public retrieval request was not refused")
			}
			notification := clientRequest(2, engines.LocalCredentialMethod, params)
			notification.ID = nil
			b.handleMessage(notification)
			if len(engine.snapshot()) != 0 || len(out.frames()) != 1 {
				t.Fatal("public retrieval request or notification reached the manager")
			}
		})
	}
	test("omitted engine", nil)
	test("owned Strata", engines.CredentialParams{Engine: "strata"})
	test("local LM Studio without node selector", engines.CredentialParams{Engine: "lmstudio"})
	test("self node selector", map[string]string{"engine": "lmstudio", "nodeId": "self-node"})
	test("remote node selector", map[string]string{"engine": "lmstudio", "nodeId": "remote"})
	test("malformed params", json.RawMessage(`[]`))
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
			engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method != engines.LocalCredentialMethod || string(raw) != `{"engine":"strata"}` {
					t.Error("unexpected owned-engine credential request")
				}
				return tc.answer, tc.errMsg
			})
			if got := b.localEngineCredential("strata", 4321); got != tc.want {
				t.Fatal("credential did not match the expected port-bound answer")
			}
		})
	}
	t.Run("no engine-manager", func(t *testing.T) {
		b, _ := newPolicyBroker(t)
		if got := b.localEngineCredential("strata", 4321); got != "" {
			t.Fatal("credential returned without engine-manager")
		}
	})
	// The broker consumes the manager's running, port-bound snapshot for
	// either command-started or adopted LM Studio; it owns no process test.
	for _, tc := range []struct {
		name   string
		answer engines.LocalCredential
		want   string
	}{
		{"running command-mode LM Studio", engines.LocalCredential{Credential: brokerTestKey, Port: 4321}, brokerTestKey},
		{"running adopted LM Studio", engines.LocalCredential{Credential: brokerTestKey, Port: 4321}, brokerTestKey},
		{"LM Studio wrong returned port", engines.LocalCredential{Credential: brokerTestKey, Port: 4322}, ""},
		{"LM Studio has no running key", engines.LocalCredential{}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, _ := newPolicyBroker(t)
			engine := attachFakeEngine(t, b)
			engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method != engines.LocalCredentialMethod || string(raw) != `{"engine":"lmstudio"}` {
					t.Error("unexpected LM Studio credential request")
				}
				return tc.answer, ""
			})
			if b.localEngineCredential("lmstudio", 4321) != tc.want || len(engine.snapshot()) != 1 {
				t.Fatal("LM Studio credential was not bound to its running port")
			}
		})
	}
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
