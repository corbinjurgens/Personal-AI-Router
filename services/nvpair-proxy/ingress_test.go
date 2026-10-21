// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/nodepolicy"
)

func TestIngressPinnedOriginRefusesBeforeEffects(t *testing.T) {
	test := func(name, origin string, omitOrigin bool, wantStatus int) {
		t.Run(name, func(t *testing.T) {
			p, engine := strictHTTPFixture(t)
			strict := strictRequest()
			strict.Request.OriginNodeID = origin
			encoded, err := json.Marshal(strict)
			if err != nil {
				t.Fatal(err)
			}
			if omitOrigin {
				const member = `"originNodeId":"",`
				if strings.Count(string(encoded), member) != 1 {
					t.Fatal("missing-origin fixture requires exactly one empty origin member")
				}
				encoded = []byte(strings.Replace(string(encoded), member, "", 1))
			}
			body := &routingUploadReadWitness{ReadCloser: io.NopCloser(strings.NewReader(strictBody("m"))), entered: make(chan struct{})}
			r := httptest.NewRequest(http.MethodPost, routeChatCompletions, body)
			r.Header.Set(nodepolicy.StrictAdmissionHeader, string(encoded))
			// Neither a claimed HTTP identity nor the destination's own identity
			// may replace the peer supplied by pin authentication.
			r.Header.Set("X-PAIR-Origin-Node-ID", origin)
			notifications := 0
			p.admission.notify = func(string, any) { notifications++ }
			w := httptest.NewRecorder()
			p.soleFacade().serveIngress(w, r, "peer")
			var refusal struct {
				Code string `json:"code"`
			}
			if err := json.Unmarshal(w.Body.Bytes(), &refusal); err != nil {
				t.Fatal(err)
			}
			if w.Code != wantStatus || refusal.Code != "strict-admission" {
				t.Fatalf("status/code = %d/%q, want %d/strict-admission", w.Code, refusal.Code, wantStatus)
			}
			select {
			case <-body.entered:
				t.Error("refused strict origin read the inference body")
			default:
			}
			p.admission.mu.Lock()
			active, queued, tickets, nextTicket, markers := p.admission.active, p.admission.queued, len(p.admission.tickets), p.admission.nextTicket, len(p.admission.strict.markers)
			p.admission.mu.Unlock()
			if engine.hits() != 0 || active != 0 || queued != 0 || tickets != 0 || nextTicket != 0 || markers != 0 || notifications != 0 {
				t.Fatalf("refusal effects: engine=%d active=%d queued=%d tickets=%d nextTicket=%d markers=%d notifications=%d", engine.hits(), active, queued, tickets, nextTicket, markers, notifications)
			}
			if len(p.activeRoutingRequests()) != 0 || len(p.workloadRequests) != 0 || p.requestSequence != 0 {
				t.Fatal("refused strict origin registered a routing request or workload")
			}
		})
	}
	test("different origin", "other-origin", false, http.StatusForbidden)
	test("destination origin", "test-origin", false, http.StatusForbidden)
	test("case differs", "Peer", false, http.StatusForbidden)
	test("missing origin", "", true, http.StatusBadRequest)
	test("empty origin", "", false, http.StatusBadRequest)
}

func TestIngressPinnedOriginAuthenticatedHandoff(t *testing.T) {
	test := func(name, origin string, strict bool, wantStatus, wantHits, wantActive int) {
		t.Run(name, func(t *testing.T) {
			p, engine := strictHTTPFixture(t)
			selfDir, peerDir := t.TempDir(), t.TempDir()
			clustertrusttest.Join(t, selfDir, "fixture-cluster", "destination-node")
			clustertrusttest.Join(t, peerDir, "fixture-cluster", "authenticated-origin")
			proxyReversePin(t, selfDir, peerDir, "authenticated-origin")
			proxyReversePin(t, peerDir, selfDir, "destination-node")
			p.mesh = clustertrust.Open(selfDir)
			handlerDone := make(chan struct{})
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(handlerDone)
				p.soleFacade().handleClusterIngress(w, r)
			}))
			server.TLS = p.mesh.ServerTLSConfig()
			server.StartTLS()
			defer server.Close()
			clientTLS, ok := clustertrust.Open(peerDir).ClientTLSConfig("destination-node")
			if !ok {
				t.Fatal("pinned client TLS configuration unavailable")
			}
			transport := &http.Transport{TLSClientConfig: clientTLS}
			defer transport.CloseIdleConnections()
			client := &http.Client{Transport: transport, Timeout: 5 * time.Second}
			r, err := http.NewRequest(http.MethodPost, server.URL+routeChatCompletions, strings.NewReader(strictBody("m")))
			if err != nil {
				t.Fatal(err)
			}
			if strict {
				envelope := strictRequest()
				envelope.Request.OriginNodeID = origin
				encoded, err := json.Marshal(envelope)
				if err != nil {
					t.Fatal(err)
				}
				r.Header.Set(nodepolicy.StrictAdmissionHeader, string(encoded))
			}
			r.Header.Set("X-PAIR-Origin-Node-ID", origin)
			response, err := client.Do(r)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			var result struct {
				Code string `json:"code"`
				Done bool   `json:"done"`
			}
			if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
				t.Fatal(err)
			}
			waitChan(t, handlerDone, "authenticated ingress handler cleanup")
			if response.StatusCode != wantStatus || engine.hits() != wantHits || p.admission.currentActive() != wantActive {
				t.Fatalf("authenticated ingress status/hits/active = %d/%d/%d, want %d/%d/%d", response.StatusCode, engine.hits(), p.admission.currentActive(), wantStatus, wantHits, wantActive)
			}
			if wantStatus == http.StatusForbidden {
				if result.Code != "strict-admission" {
					t.Fatalf("refusal code = %q, want strict-admission", result.Code)
				}
			} else if !result.Done {
				t.Fatal("authenticated ingress did not deliver the synthetic engine response")
			}
			if len(p.activeRoutingRequests()) != 0 || len(p.workloadRequests) != 0 || p.requestSequence != 0 {
				t.Fatal("terminal ingress registered an origin request or workload")
			}
		})
	}
	test("mismatch refused", "claimed-origin", true, http.StatusForbidden, 0, 0)
	test("exact authenticated origin served", "authenticated-origin", true, http.StatusOK, 1, 1)
	test("non-strict inference unchanged", "claimed-origin", false, http.StatusOK, 1, 0)
}

// TestResolveCandidatesUnclusteredDropsRelayPeers is the core isolation
// assertion: an unclustered node (nil mesh) must not route inference to
// relay-discovered peers — even one advertising a cluster principal — because it
// holds no pin to reach them over mTLS. Only an explicit, user-added manual node
// survives, dialed plaintext.
func TestResolveCandidatesUnclusteredDropsRelayPeers(t *testing.T) {
	disc := NewDiscovery()
	disc.SetSubscribed([]Node{{
		ID: "peer-a", Host: "peer-a", Port: 11434,
		Addresses:   []string{"192.0.2.10"},
		IP:          "192.0.2.10",
		ClusterUUID: "cluster-uuid-a",
	}})
	disc.AddManual(Node{
		ID: "manual-x", Host: "manual-x", Port: 11434,
		Addresses: []string{"192.0.2.20"}, IP: "192.0.2.20",
	})
	p := testProxy(anyProfile(t), disc, 11435) // mesh nil => unclustered

	cands := p.soleFacade().resolveCandidates("")
	if len(cands) != 1 {
		t.Fatalf("unclustered candidate set = %+v, want exactly the manual node", cands)
	}
	if cands[0].id != "manual-x" || cands[0].peerUUID != "" || cands[0].url.Scheme != "http" {
		t.Fatalf("unclustered candidate = %+v, want plaintext manual-x with no peerUUID", cands[0])
	}
}

// TestHandlePlainRejectsNonLoopback proves the plaintext personality is
// loopback-only: a LAN caller is refused (closing the former open-relay), so
// peers cannot use the plaintext path at all.
func TestHandlePlainRejectsNonLoopback(t *testing.T) {
	p := testProxy(anyProfile(t), NewDiscovery(), 11435)
	req := httptest.NewRequest(http.MethodPost, "/api/generate", nil)
	req.RemoteAddr = "192.0.2.50:40000"
	rec := httptest.NewRecorder()

	p.soleFacade().handlePlain(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("non-loopback plaintext status = %d, want %d", rec.Code, http.StatusForbidden)
	}
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Errorf("Access-Control-Allow-Origin = %q, want no CORS header on the refusal", got)
	}
}

// Preflight is subject to the same ingress gate as ordinary requests.
func TestHandlePlainRejectsPreflightAtLoopbackGate(t *testing.T) {
	p := testProxy(anyProfile(t), NewDiscovery(), 11435)
	req := httptest.NewRequest(http.MethodOptions, "/api/generate", nil)
	req.RemoteAddr = "192.0.2.50:40000"
	rec := httptest.NewRecorder()

	p.soleFacade().handlePlain(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("preflight status = %d, want %d", rec.Code, http.StatusForbidden)
	}
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Errorf("Access-Control-Allow-Origin = %q, want no CORS header", got)
	}
}

// engine-manager marks its own identity probes so the compatibility facade can
// never be adopted as the engine itself. The rejection has to name the engine
// this proxy fronts, or the operator reading it is sent to the wrong process —
// which is why this runs per engine rather than asserting only the status.
func TestHandlePlainRejectsEngineIdentityProbe(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		p := testProxy(tc.profile, NewDiscovery(), tc.profile.StandalonePort)
		req := httptest.NewRequest(http.MethodGet, "/api/version", nil)
		req.RemoteAddr = "127.0.0.1:40000"
		req.Header.Set(engineIdentityProbeHeader, "1")
		rec := httptest.NewRecorder()

		p.soleFacade().handlePlain(rec, req)
		if rec.Code != http.StatusConflict {
			t.Fatalf("identity probe status = %d, want %d", rec.Code, http.StatusConflict)
		}
		if body := rec.Body.String(); !strings.Contains(body, tc.profile.DisplayName) {
			t.Fatalf("rejection does not name %s: %s", tc.profile.DisplayName, body)
		}
	})
}

// TestHandleClusterIngressUnclusteredForbids proves the mTLS ingress fails
// closed on an unclustered node (nil mesh): with no cluster identity there are
// no pins, so every caller is rejected.
func TestHandleClusterIngressUnclusteredForbids(t *testing.T) {
	p := testProxy(anyProfile(t), NewDiscovery(), 11435)
	req := httptest.NewRequest(http.MethodPost, "/api/generate", nil) // no client cert
	rec := httptest.NewRecorder()

	p.soleFacade().handleClusterIngress(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("unclustered ingress status = %d, want %d", rec.Code, http.StatusForbidden)
	}
}

func TestIsLoopbackRemote(t *testing.T) {
	for _, c := range []struct {
		addr string
		want bool
	}{
		{"127.0.0.1:5000", true},
		{"[::1]:5000", true},
		{"192.168.1.10:5000", false},
		{"10.0.0.5:80", false},
		{"", false},
		{"garbage", false},
	} {
		if got := isLoopbackRemote(c.addr); got != c.want {
			t.Errorf("isLoopbackRemote(%q) = %v, want %v", c.addr, got, c.want)
		}
	}
}

func TestLocalReverseProxyUsesSharedPlainTransport(t *testing.T) {
	p := testProxy(anyProfile(t), NewDiscovery(), 11435)
	shared := p.plainHTTPTransport()
	target := &url.URL{Scheme: "http", Host: "127.0.0.1:1"}
	rp := p.soleFacade().newLocalReverseProxy(target)
	tr, ok := rp.Transport.(proxyResponseTransport)
	if !ok {
		t.Fatalf("Transport type = %T, want trailer-isolating transport", rp.Transport)
	}
	if tr.RoundTripper != shared {
		t.Fatal("ingress reverse proxy did not use the shared plain Transport")
	}
}
