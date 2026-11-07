// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

func corsModels(tc engineCase) string {
	return `{"object":"list","data":[{"id":"private-model"}]}`
}

func corsRequest(method, path, origin string) *http.Request {
	r := httptest.NewRequest(method, path, nil)
	r.RemoteAddr = "127.0.0.1:40000"
	if origin != "" {
		r.Header.Set("Origin", origin)
	}
	if method == "OPTIONS" {
		r.Header.Set("Access-Control-Request-Method", "POST")
		r.Header.Set("Access-Control-Request-Headers", "Content-Type")
	}
	return r
}
func corsEngine(t *testing.T, tc engineCase, allowed string, status int) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if allowed != "" && (r.Header.Get("Origin") == allowed || allowed == "*") {
			w.Header().Set("Access-Control-Allow-Origin", allowed)
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
			w.Header().Set("Access-Control-Allow-Credentials", "true")
			w.Header().Add("Vary", "Origin")
		}
		w.Header().Set("Content-Type", "application/json")
		if status != 200 {
			w.WriteHeader(status)
			if _, err := io.WriteString(w, "engine refused"); err != nil {
				t.Error(err)
			}
			return
		}
		if r.Method == "OPTIONS" {
			w.WriteHeader(204)
			return
		}
		if _, err := io.WriteString(w, corsModels(tc)); err != nil {
			t.Error(err)
		}
	}))
}
func proxyForCORSTargets(t *testing.T, tc engineCase, servers ...*httptest.Server) *facade {
	d := NewDiscovery()
	for i, s := range servers {
		d.AddManual(nodeFor(t, string(rune('a'+i)), s.URL))
	}
	return testProxy(tc.profile, d, tc.profile.StandalonePort).soleFacade()
}
func TestCORSOriginDenialIsNotRewritten(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		for _, status := range []int{200, 401, 403} {
			t.Run(http.StatusText(status), func(t *testing.T) {
				engine := corsEngine(t, tc, "", status)
				defer engine.Close()
				p := proxyForCORSTargets(t, tc, engine)
				for _, method := range []string{"OPTIONS", "GET"} {
					rec := httptest.NewRecorder()
					p.handlePlain(rec, corsRequest(method, "/policy", "http://example.com"))
					want := status
					if method == "OPTIONS" && status == 200 {
						want = 204
					}
					if rec.Code != want || rec.Header().Get("Access-Control-Allow-Origin") != "" {
						t.Fatalf("%s: %d %v", method, rec.Code, rec.Header())
					}
					if status != 200 && rec.Body.String() != "engine refused" {
						t.Fatal("upstream error body changed")
					}
				}
			})
		}
	})
}

// A preflight is shared only where every engine that answered agrees. The
// request it precedes would route around an engine that is down, so that
// engine's silence must not deny the browser the ones that are up.
func TestCORSClusterRequiresEveryRespondingTarget(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		for _, policy := range []struct {
			name, origin string
			status, want int
		}{
			{"agree", "http://app.test", 200, 204}, {"different origin", "http://other.test", 200, 403},
			{"missing policy", "", 200, 403}, {"denied", "", 403, 403}, {"unavailable", "", 503, 204},
		} {
			t.Run(policy.name, func(t *testing.T) {
				a := corsEngine(t, tc, "http://app.test", 200)
				defer a.Close()
				b := corsEngine(t, tc, policy.origin, policy.status)
				defer b.Close()
				p := proxyForCORSTargets(t, tc, a, b)
				rec := httptest.NewRecorder()
				p.handlePlain(rec, corsRequest("OPTIONS", tc.inferencePath, "http://app.test"))
				if rec.Code != policy.want {
					t.Fatalf("status=%d want=%d", rec.Code, policy.want)
				}
				if policy.want != 204 && rec.Header().Get("Access-Control-Allow-Origin") != "" {
					t.Fatal("granted denied origin")
				}
				if policy.want == 204 && (rec.Header().Get("Access-Control-Max-Age") == "" || rec.Header().Get("Access-Control-Allow-Credentials") != "true") {
					t.Fatal("invalid agreement")
				}
			})
		}
	})
}

func TestCORSNoRetryOnPermissionDenial(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		for _, status := range []int{401, 403} {
			a := corsEngine(t, tc, "", status)
			var calls atomic.Int32
			b := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls.Add(1); w.WriteHeader(200) }))
			p := proxyForCORSTargets(t, tc, a, b)
			rec := httptest.NewRecorder()
			p.handlePlain(rec, corsRequest("GET", "/policy", "http://app.test"))
			if rec.Code != status || calls.Load() != 0 {
				t.Fatal("retried a permission denial")
			}
			a.Close()
			b.Close()
		}
	})
}
func TestCORSOrdinaryOptionsAndPolicyChanges(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		var origin atomic.Value
		origin.Store("http://app.test")
		var calls atomic.Int32
		a := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", origin.Load().(string))
			w.Header().Set("Access-Control-Allow-Methods", "POST")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
			w.WriteHeader(200)
		}))
		defer a.Close()
		b := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			w.Header().Set("Access-Control-Allow-Origin", "http://app.test")
			w.Header().Set("Access-Control-Allow-Methods", "POST")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
			w.WriteHeader(200)
		}))
		defer b.Close()
		p := proxyForCORSTargets(t, tc, a, b)
		req := httptest.NewRequest("OPTIONS", "/policy", nil)
		rec := httptest.NewRecorder()
		p.handleHTTP(rec, req)
		if rec.Code != 200 || calls.Load() != 0 {
			t.Fatal("ordinary OPTIONS was synthesized or fanned out")
		}
		rec = httptest.NewRecorder()
		p.handleHTTP(rec, corsRequest("OPTIONS", "/policy", "http://app.test"))
		if rec.Code != 204 {
			t.Fatal("initial agreement failed")
		}
		origin.Store("http://other.test")
		rec = httptest.NewRecorder()
		p.handleHTTP(rec, corsRequest("OPTIONS", "/policy", "http://app.test"))
		if rec.Code != 403 {
			t.Fatal("cached obsolete permission")
		}
	})
}
func TestCORSPairedIngressPreservesPolicyAndIsTerminal(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		aDir, bDir := t.TempDir(), t.TempDir()
		clustertrusttest.Join(t, aDir, "cluster", "a")
		clustertrusttest.Join(t, bDir, "cluster", "b")
		pin := func(dst, src, id string) {
			t.Helper()
			pem, err := os.ReadFile(filepath.Join(src, "node.crt"))
			if err != nil {
				t.Fatal(err)
			}
			body, err := json.Marshal(map[string]any{"nodeUuid": id, "certPem": string(pem),
				"clusterId": "cluster", "admissionEpoch": 1})
			if err != nil {
				t.Fatal(err)
			}
			if err = os.MkdirAll(filepath.Join(dst, "trusted"), 0700); err != nil {
				t.Fatal(err)
			}
			if err = os.WriteFile(filepath.Join(dst, "trusted", id+".json"), body, 0600); err != nil {
				t.Fatal(err)
			}
			members, err := json.Marshal([]map[string]any{{"nodeUuid": id, "state": "member", "clusterId": "cluster", "admissionEpoch": 1}})
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dst, "members.json"), members, 0o600); err != nil {
				t.Fatal(err)
			}
		}
		pin(aDir, bDir, "b")
		pin(bDir, aDir, "a")
		local := corsEngine(t, tc, "http://app.test", 200)
		defer local.Close()
		var unexpected atomic.Int32
		other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { unexpected.Add(1) }))
		defer other.Close()
		peer := proxyForCORSTargets(t, tc, other)
		peer.host.mesh = clustertrust.Open(bDir)
		backend := nodeFor(t, "local", local.URL)
		if err := peer.setLocalBackend(localBackend{Host: backend.Addresses[0], Port: backend.Port, Healthy: true}); err != nil {
			t.Fatal(err)
		}
		ingress := httptest.NewUnstartedServer(http.HandlerFunc(peer.handleClusterIngress))
		ingress.TLS = peer.host.mesh.ServerTLSConfig()
		ingress.StartTLS()
		defer ingress.Close()
		caller := testProxy(tc.profile, NewDiscovery(), tc.profile.StandalonePort).soleFacade()
		caller.host.mesh = clustertrust.Open(aDir)
		n := nodeFor(t, "peer", ingress.URL)
		n.ClusterUUID = "b"
		caller.discovery.SetSubscribed([]Node{n})
		for _, origin := range []string{"http://app.test", "http://denied.test"} {
			for _, method := range []string{"OPTIONS", "GET"} {
				rec := httptest.NewRecorder()
				caller.handlePlain(rec, corsRequest(method, "/policy", origin))
				expected := ""
				if origin == "http://app.test" {
					expected = origin
				}
				if rec.Header().Get("Access-Control-Allow-Origin") != expected {
					t.Fatalf("paired policy lost: %d %v", rec.Code, rec.Header())
				}
				if rec.Code != 200 && rec.Code != 204 {
					t.Fatalf("paired status %d: %s", rec.Code, rec.Body)
				}
			}
		}
		if unexpected.Load() != 0 {
			t.Fatal("paired ingress routed onward")
		}
	})
}

// Opt-in fixture for real browser validation; regular suites do not keep a server running.
func TestCORSBrowserFixture(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		file := os.Getenv("PAIR_CORS_BROWSER_FIXTURE")
		if file == "" {
			t.Skip("browser fixture is opt-in")
		}
		page := func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/html")
			if _, err := io.WriteString(w, "<!doctype html><title>CORS test</title>"); err != nil {
				t.Error(err)
			}
		}
		allowed := httptest.NewServer(http.HandlerFunc(page))
		defer allowed.Close()
		denied := httptest.NewServer(http.HandlerFunc(page))
		defer denied.Close()
		engine := corsEngine(t, tc, allowed.URL, 200)
		defer engine.Close()
		second := corsEngine(t, tc, allowed.URL, 200)
		defer second.Close()
		p := proxyForCORSTargets(t, tc, engine, second)
		proxy := httptest.NewServer(http.HandlerFunc(p.handlePlain))
		defer proxy.Close()
		none := proxyForCORSTargets(t, tc)
		unavailable := httptest.NewServer(http.HandlerFunc(none.handlePlain))
		defer unavailable.Close()
		data, _ := json.Marshal(map[string]string{"allowed": allowed.URL, "denied": denied.URL, "proxy": proxy.URL, "engine": engine.URL, "unavailable": unavailable.URL, "models": tc.modelListPath})
		if err := os.WriteFile(file, data, 0600); err != nil {
			t.Fatal(err)
		}
		deadline := time.After(60 * time.Second)
		tick := time.NewTicker(100 * time.Millisecond)
		defer tick.Stop()
		for {
			select {
			case <-deadline:
				t.Fatal("browser fixture timed out")
			case <-tick.C:
				if _, err := os.Stat(file + ".done"); err == nil {
					if err := os.Remove(file + ".done"); err != nil {
						t.Fatal(err)
					}
					return
				}
			}
		}
	})
}

func TestCORSStreamingResponseIsPreserved(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		release := make(chan struct{})
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("Origin") != "http://app.test" {
				t.Error("lost stream origin")
			}
			w.Header().Set("Access-Control-Allow-Origin", "http://app.test")
			w.Header().Set("Access-Control-Expose-Headers", "X-Engine")
			w.Header().Set("X-Engine", "metadata")
			if _, err := io.WriteString(w, "first"); err != nil {
				t.Error(err)
			}
			w.(http.Flusher).Flush()
			select {
			case <-release:
				if _, err := io.WriteString(w, "last"); err != nil {
					t.Error(err)
				}
			case <-r.Context().Done():
			}
		}))
		defer upstream.Close()
		p := proxyForCORSTargets(t, tc, upstream)
		proxy := httptest.NewServer(http.HandlerFunc(p.handlePlain))
		defer proxy.Close()
		defer func() {
			select {
			case <-release:
			default:
				close(release)
			}
		}()
		req, err := http.NewRequest("GET", proxy.URL+"/stream", nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Origin", "http://app.test")
		client := &http.Client{Timeout: 3 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := resp.Body.Close(); err != nil {
				t.Error(err)
			}
		}()
		first := make([]byte, 5)
		if _, err = io.ReadFull(resp.Body, first); err != nil {
			t.Fatal(err)
		}
		if string(first) != "first" || resp.Header.Get("Access-Control-Allow-Origin") != "http://app.test" || resp.Header.Get("Access-Control-Expose-Headers") != "X-Engine" {
			t.Fatal("stream changed")
		}
		close(release)
		rest, err := io.ReadAll(resp.Body)
		if err != nil || string(rest) != "last" {
			t.Fatal("stream incomplete", err)
		}
	})
}
