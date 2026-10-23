// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"nvpair-shared/nodepolicy"
)

const proxyTestKey = "FAKE-PROXY-KEY-0123456789"

func setKeyedBackend(t *testing.T, f *facade, serverURL, key string) {
	t.Helper()
	n := nodeFor(t, "backend", serverURL)
	if err := f.setLocalBackend(localBackend{Host: n.Addresses[0], Port: n.Port, Healthy: true, Credential: key}); err != nil {
		t.Fatal(err)
	}
}

// withCallerCredentials sets the credentials a client might send, which must
// never replace the local key.
func withCallerCredentials(r *http.Request) *http.Request {
	r.Header.Set("Authorization", "Bearer caller-key")
	r.Header.Set("X-Api-Key", "caller-key")
	return r
}

func requireLocalKey(t *testing.T, h http.Header) {
	t.Helper()
	if h.Get("Authorization") != "Bearer "+proxyTestKey || h.Get("X-Api-Key") != "" {
		t.Fatalf("engine saw Authorization %q, X-Api-Key %q; want only the local key", h.Get("Authorization"), h.Get("X-Api-Key"))
	}
}

func TestLocalCredentialIsPairedWithTheBackendPort(t *testing.T) {
	p := testProxy(anyCase(t).profile, NewDiscovery(), anyCase(t).profile.FacadePort)
	f := p.soleFacade()
	if err := f.setLocalBackend(localBackend{Host: "127.0.0.1", Port: 4321, Healthy: true, Credential: proxyTestKey}); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		host string
		want string
	}{
		{"127.0.0.1:4321", proxyTestKey},
		{"127.0.0.1:4322", ""},
		{"10.0.0.2:4321", ""},
	} {
		if got := f.localCredentialFor(&url.URL{Scheme: "http", Host: tc.host}); got != tc.want {
			t.Fatalf("credential for %s = %q, want %q", tc.host, got, tc.want)
		}
	}
	if err := f.setLocalBackend(localBackend{Host: "127.0.0.1", Port: 4321, Healthy: false, Credential: proxyTestKey}); err != nil {
		t.Fatal(err)
	}
	if got := f.localCredentialFor(&url.URL{Scheme: "http", Host: "127.0.0.1:4321"}); got != "" {
		t.Fatal("an unhealthy backend still offers its key")
	}
}

// Local self dispatch: inference and the model list carry the key in place of
// the caller's; another forwarded path carries what the caller sent.
func TestLocalCredentialOnSelfRequests(t *testing.T) {
	tc := lmstudioCase(t)
	engine := newRecordingEngine(t, http.StatusOK)
	port := tc.profile.FacadePort
	disc := NewDiscovery()
	disc.AddManual(selfNode("self", port, tc.advertisedModel))
	p := testProxy(tc.profile, disc, port)
	f := p.soleFacade()
	setKeyedBackend(t, f, engine.URL, proxyTestKey)

	f.handleHTTP(httptest.NewRecorder(), withCallerCredentials(tc.inferenceRequest()))
	if engine.hits() != 1 {
		t.Fatalf("inference hits = %d", engine.hits())
	}
	requireLocalKey(t, engine.lastHeader())

	f.handleHTTP(httptest.NewRecorder(), withCallerCredentials(httptest.NewRequest(http.MethodGet, tc.modelListPath, nil)))
	if engine.hits() != 2 {
		t.Fatalf("model list hits = %d", engine.hits())
	}
	requireLocalKey(t, engine.lastHeader())
}

// Peer ingress: admitted inference and the model list carry the key; any
// other path is forwarded with only what the peer sent.
func TestLocalCredentialOnPeerIngress(t *testing.T) {
	tc := lmstudioCase(t)
	engine := newRecordingEngine(t, http.StatusOK)
	p := testProxy(tc.profile, NewDiscovery(), tc.profile.FacadePort)
	f := p.soleFacade()
	setKeyedBackend(t, f, engine.URL, proxyTestKey)
	p.admission.setAvailability(nodepolicy.Available, false)

	rec := httptest.NewRecorder()
	f.serveIngress(rec, withCallerCredentials(ingressRequest(tc, tc.inferenceBody(), "0")), "peer")
	if rec.Code != http.StatusOK || engine.hits() != 1 {
		t.Fatalf("ingress inference: status %d, hits %d", rec.Code, engine.hits())
	}
	requireLocalKey(t, engine.lastHeader())

	f.serveIngress(httptest.NewRecorder(), withCallerCredentials(httptest.NewRequest(http.MethodGet, tc.modelListPath, nil)), "peer")
	if engine.hits() != 2 {
		t.Fatalf("ingress model list hits = %d", engine.hits())
	}
	requireLocalKey(t, engine.lastHeader())

	other := httptest.NewRequest(http.MethodPost, tc.nonInferencePath, strings.NewReader(`{}`))
	f.serveIngress(httptest.NewRecorder(), other, "peer")
	if engine.hits() != 3 {
		t.Fatalf("ingress passthrough hits = %d", engine.hits())
	}
	if got := engine.lastHeader().Get("Authorization"); got != "" {
		t.Fatalf("a passthrough path carried Authorization %q", got)
	}
	peerOwn := withCallerCredentials(httptest.NewRequest(http.MethodPost, tc.nonInferencePath, strings.NewReader(`{}`)))
	f.serveIngress(httptest.NewRecorder(), peerOwn, "peer")
	if got := engine.lastHeader().Get("Authorization"); got != "Bearer caller-key" {
		t.Fatalf("a passthrough path replaced the peer's own credential: %q", got)
	}
}

// Strict admission serves both ingress and local self dispatch.
func TestLocalCredentialOnStrictRequests(t *testing.T) {
	for _, path := range []string{"ingress", "self"} {
		t.Run(path, func(t *testing.T) {
			p, e := strictHTTPFixture(t)
			setKeyedBackend(t, p.soleFacade(), e.URL, proxyTestKey)
			w := httptest.NewRecorder()
			serveStrictHTTP(p, path, w, withCallerCredentials(strictHTTPRequest(t)))
			if w.Code != http.StatusOK || e.hits() != 1 {
				t.Fatalf("status=%d hits=%d body=%s", w.Code, e.hits(), w.Body)
			}
			requireLocalKey(t, e.lastHeader())
		})
	}
}

// Without a key nothing changes: the caller's own credentials pass through.
func TestUnkeyedBackendKeepsCallerCredentials(t *testing.T) {
	tc := anyCase(t)
	engine := newRecordingEngine(t, http.StatusOK)
	p := testProxy(tc.profile, NewDiscovery(), tc.profile.FacadePort)
	f := p.soleFacade()
	setBackend(t, f, engine.URL, true)
	p.admission.setAvailability(nodepolicy.Available, false)
	f.serveIngress(httptest.NewRecorder(), withCallerCredentials(ingressRequest(tc, tc.inferenceBody(), "0")), "peer")
	if got := engine.lastHeader().Get("Authorization"); got != "Bearer caller-key" {
		t.Fatalf("Authorization = %q, want the caller's", got)
	}
}

// Only chat completions and the model list carry the key. A cross-engine tier
// request on /v1/completions can still reach the keyed local
// engine, and there the caller's own credentials pass through unchanged.
func TestLocalCredentialStaysOffOtherInferencePaths(t *testing.T) {
	w := &recordingWriter{}
	p, oll, lm := twoEngineProxy(t, w)
	engine := newRecordingEngine(t, http.StatusOK)
	lm.discovery.AddManual(selfNode("self", lm.profile.FacadePort, "qwen3-8b"))
	setKeyedBackend(t, lm, engine.URL, proxyTestKey)
	oll.discovery.AddManual(selfNode("self", oll.profile.FacadePort, "llama3:latest"))
	setBackend(t, oll, newRecordingEngine(t, http.StatusOK).URL, true)
	pol := nodepolicy.Default()
	pol.Tiers = map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "lmstudio", Model: "qwen3-8b"}}}
	p.admission.setPolicy(pol)

	send := func(path string) http.Header {
		t.Helper()
		before := engine.hits()
		r := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"model":"weak","messages":[]}`))
		rec := httptest.NewRecorder()
		oll.handleHTTP(rec, withCallerCredentials(r))
		if rec.Code != http.StatusOK || engine.hits() != before+1 {
			t.Fatalf("%s: status %d, hits %d -> %d", path, rec.Code, before, engine.hits())
		}
		return engine.lastHeader()
	}
	requireLocalKey(t, send("/v1/chat/completions"))
	for _, path := range []string{"/v1/completions"} {
		h := send(path)
		if h.Get("Authorization") != "Bearer caller-key" || h.Get("X-Api-Key") != "caller-key" {
			t.Fatalf("%s carried Authorization %q, X-Api-Key %q; want only the caller's", path, h.Get("Authorization"), h.Get("X-Api-Key"))
		}
	}
}
