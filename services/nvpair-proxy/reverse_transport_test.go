// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-shared/reach"
)

// Outer admission is supplied at the ReverseManager.Register seam; real yamux
// streams carry pinned inner TLS into the production terminal ingress. The
// integration lane separately covers outer authentication and private dial IPC.
type proxyReverseFixture struct {
	p, peer                   *Proxy
	selfDir, peerDir          string
	server                    *httptest.Server
	node                      Node
	local, remote             *clustertrust.ReverseManager
	directHealthy             atomic.Bool
	directConnectFails        atomic.Bool
	directCalls, reverseCalls atomic.Int32
	closedRequests            atomic.Int32
	beforeReverse             func()
}

func newProxyReverseFixture(t *testing.T, engine string, handler http.Handler) *proxyReverseFixture {
	t.Helper()
	profile, ok := profileFor(engine)
	if !ok {
		t.Fatalf("unknown fixture engine %q", engine)
	}
	return newProxyReverseFixtureFor(t, profile, handler)
}

// newProxyReverseFixtureFor builds the fixture for any profile, including a
// descriptor engine's dynamic one.
func newProxyReverseFixtureFor(t *testing.T, profile engineProfile, handler http.Handler) *proxyReverseFixture {
	t.Helper()
	engine := profile.Name
	f := &proxyReverseFixture{selfDir: filepath.Join(t.TempDir(), "cluster"), peerDir: filepath.Join(t.TempDir(), "cluster")}
	clustertrusttest.WriteKeypair(t, f.selfDir, "uuid-self")
	clustertrusttest.WriteKeypair(t, f.peerDir, "uuid-peer")
	clustertrusttest.WriteAdmission(t, f.selfDir, "fixture-cluster", 1)
	clustertrusttest.WriteAdmission(t, f.peerDir, "fixture-cluster", 1)
	proxyReversePin(t, f.selfDir, f.peerDir, "uuid-peer")
	proxyReversePin(t, f.peerDir, f.selfDir, "uuid-self")
	mesh, peerMesh := clustertrust.Open(f.selfDir), clustertrust.Open(f.peerDir)
	f.p = testProxy(profile, NewDiscovery(), 0)
	f.p.mesh, f.p.baseConfigDir = mesh, filepath.Dir(f.selfDir)
	f.peer = testProxy(profile, NewDiscovery(), 0)
	f.peer.mesh = peerMesh
	backend := httptest.NewServer(handler)
	t.Cleanup(backend.Close)
	n := nodeFor(t, "local-engine", backend.URL)
	if err := f.peer.soleFacade().setLocalBackend(localBackend{Engine: engine, Host: n.Addresses[0], Port: n.Port, Healthy: true}); err != nil {
		t.Fatal(err)
	}
	f.server = httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.TLS == nil || len(r.TLS.PeerCertificates) != 1 {
			t.Error("ingress did not receive the inner client certificate")
		}
		if r.Close {
			f.closedRequests.Add(1)
		}
		f.peer.soleFacade().handleClusterIngress(w, r)
	}))
	f.server.TLS = peerMesh.ServerTLSConfig()
	f.server.StartTLS()
	t.Cleanup(f.server.Close)
	f.node = nodeFor(t, "host-peer", f.server.URL)
	f.node.ClusterUUID, f.node.Models = "uuid-peer", []string{"fixture-model"}
	f.p.soleFacade().discovery.SetSubscribed([]Node{f.node})
	resolve := func(ctx context.Context, _ clustertrust.ReversePeer, service noderec.ServiceKey) (clustertrust.ReverseTarget, error) {
		if service != profile.DiscoveryService {
			return clustertrust.ReverseTarget{}, errors.New("wrong reverse service")
		}
		return clustertrust.ReverseTarget{Protocol: clustertrust.ReverseTLS, Dial: func(ctx context.Context) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "tcp", f.server.Listener.Addr().String())
		}}, nil
	}
	owner := func(uuid string) *clustertrust.ReverseManager {
		manager, err := clustertrust.NewReverseManager(clustertrust.ReverseOptions{LocalUUID: uuid,
			Validate:       func(peer clustertrust.ReversePeer) bool { return peer.Incarnation == "fixture-v1" },
			ResolveService: resolve, SetupTimeout: time.Second})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			done := make(chan error, 1)
			go func() { done <- manager.Close() }()
			select {
			case err := <-done:
				if err != nil {
					t.Errorf("close reverse owner: %v", err)
				}
			case <-time.After(3 * time.Second):
				t.Error("reverse owner cleanup did not finish")
			}
		})
		return manager
	}
	f.local, f.remote = owner("uuid-self"), owner("uuid-peer")
	left, right := net.Pipe()
	if err := f.local.Register(clustertrust.ReversePeer{UUID: "uuid-peer", Incarnation: "fixture-v1"}, left,
		"uuid-self", "11111111111111111111111111111111"); err != nil {
		right.Close()
		t.Fatal(err)
	}
	if err := f.remote.Register(clustertrust.ReversePeer{UUID: "uuid-self", Incarnation: "fixture-v1"}, right,
		"uuid-self", "11111111111111111111111111111111"); err != nil {
		t.Fatal(err)
	}
	f.p.peerServices = clustertrust.NewPeerServiceClient(mesh, clustertrust.PeerServiceClientOptions{
		PeerClientOptions: clustertrust.PeerClientOptions{ResponseHeaderTimeout: proxyResponseTimeout},
		BaseConfigDir:     filepath.Dir(f.selfDir), ResolveRoute: f.p.resolvePeerServiceRoute,
		DirectTimeout: 100 * time.Millisecond,
		DialContext: func(ctx context.Context, _, address string) (net.Conn, error) {
			f.directCalls.Add(1)
			if f.directConnectFails.Load() {
				return nil, net.ErrClosed
			}
			if f.directHealthy.Load() {
				return (&net.Dialer{}).DialContext(ctx, "tcp", address)
			}
			// TCP connects, then first TLS data resets before HTTP dispatch.
			left, right := net.Pipe()
			if err := right.Close(); err != nil {
				t.Errorf("close TLS rejector: %v", err)
			}
			return left, nil
		},
		DialReverse: func(ctx context.Context, base, uuid string, service noderec.ServiceKey) (net.Conn, error) {
			f.reverseCalls.Add(1)
			if f.beforeReverse != nil {
				f.beforeReverse()
			}
			if base != filepath.Dir(f.selfDir) || uuid != "uuid-peer" || service != profile.DiscoveryService {
				return nil, errors.New("wrong reverse destination or config directory")
			}
			return f.local.Open(ctx, clustertrust.ReversePeer{UUID: uuid, Incarnation: "fixture-v1"}, service)
		},
	})
	t.Cleanup(f.p.closeIdleTransports)
	t.Cleanup(f.peer.closeIdleTransports)
	return f
}

func TestProxyReverseNativeNonstreamPeer(t *testing.T) {
	const payload = `{"model":"m:latest","response":"complete","done":true}`
	test := func(name string, direct bool) {
		t.Run(name, func(t *testing.T) {
			f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body struct {
					Model  string `json:"model"`
					Stream *bool  `json:"stream"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Errorf("decode engine request: %v", err)
					return
				}
				if r.Method != http.MethodPost || r.URL.Path != "/api/generate" || body.Model != "m:latest" || body.Stream == nil || *body.Stream {
					t.Error("engine did not receive the native nonstreaming request")
					return
				}
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
				if _, err := io.WriteString(w, payload); err != nil {
					t.Errorf("write engine response: %v", err)
				}
			}))
			f.directHealthy.Store(direct)
			f.node.Models = []string{"m:latest"}
			f.p.soleFacade().discovery.SetSubscribed([]Node{f.node})
			proxy := httptest.NewServer(http.HandlerFunc(f.p.soleFacade().handleHTTP))
			t.Cleanup(proxy.Close)
			request := proxyReverseRequest(t, http.MethodPost, proxy.URL+"/api/generate",
				strings.NewReader(`{"model":"m:latest","stream":false}`))
			response, err := proxy.Client().Do(request)
			if err != nil {
				t.Fatalf("request native nonstreaming response: %v", err)
			}
			data, readErr := io.ReadAll(response.Body)
			closeErr := response.Body.Close()
			if readErr != nil {
				t.Fatalf("read native nonstreaming response: %v", readErr)
			}
			if closeErr != nil {
				t.Fatalf("close native nonstreaming response: %v", closeErr)
			}
			if response.StatusCode != http.StatusOK {
				t.Fatalf("status = %d, want %d", response.StatusCode, http.StatusOK)
			}
			if string(data) != payload {
				t.Fatalf("response body changed: length = %d, want %d", len(data), len(payload))
			}
			if f.directCalls.Load() == 0 {
				t.Fatal("peer direct transport was not attempted")
			}
			if direct && f.reverseCalls.Load() != 0 {
				t.Fatalf("direct response used %d reverse calls, want 0", f.reverseCalls.Load())
			}
			if !direct && f.reverseCalls.Load() == 0 {
				t.Fatal("peer reverse transport was not used")
			}
		})
	}
	test("direct", true)
	test("reverse", false)
}

func TestProxyReverseCrossEngineTierUsesPeerService(t *testing.T) {
	var calls atomic.Int32
	f := newProxyReverseFixture(t, "lmstudio", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var body struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if body.Model != "fixture-model" {
			t.Errorf("tier forwarded model = %q, want fixture-model", body.Model)
		}
		if _, err := io.WriteString(w, "tier-result"); err != nil {
			t.Error(err)
		}
	}))
	profile, ok := profileFor("ollama")
	if !ok {
		t.Fatal("missing ollama profile")
	}
	ollama := newFacade(f.p, profile, NewDiscovery(), 0)
	f.p.facades["ollama"] = ollama
	policy := nodepolicy.Default()
	policy.Tiers = map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "lmstudio", Model: "fixture-model"}}}
	if err := policy.Validate(); err != nil {
		t.Fatal(err)
	}
	f.p.admission.setPolicy(policy)
	recorder := httptest.NewRecorder()
	ollama.handleHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"weak"}`)))
	if recorder.Code != http.StatusOK || recorder.Body.String() != "tier-result" || recorder.Header().Get(nodepolicy.EngineHeader) != "lmstudio" {
		t.Fatalf("tier status/body/engine = %d/%q/%q", recorder.Code, recorder.Body.String(), recorder.Header().Get(nodepolicy.EngineHeader))
	}
	if calls.Load() != 1 || f.reverseCalls.Load() != 1 {
		t.Fatalf("handler/reverse = %d/%d, want 1/1", calls.Load(), f.reverseCalls.Load())
	}
}

func TestProxyReverseRejectsAuthorityChangedDuringSetup(t *testing.T) {
	test := func(name string, change func(*testing.T, *proxyReverseFixture)) {
		t.Run(name, func(t *testing.T) {
			var calls atomic.Int32
			f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				w.WriteHeader(http.StatusNoContent)
			}))
			c := f.candidate(t)
			f.beforeReverse = func() { change(t, f) }
			response, err := f.p.candidateTransport(c).RoundTrip(proxyReverseRequest(t, http.MethodPost, c.url.String(), strings.NewReader(`{}`)))
			if err == nil {
				response.Body.Close()
				t.Fatal("changed setup authority reached HTTP")
			}
			if calls.Load() != 0 || f.reverseCalls.Load() != 1 {
				t.Fatalf("handler/reverse = %d/%d, want 0/1", calls.Load(), f.reverseCalls.Load())
			}
		})
	}
	test("removed pin", func(t *testing.T, f *proxyReverseFixture) {
		clustertrusttest.RemovePeerPin(t, f.selfDir, "uuid-peer")
	})
	test("endpoint removed and republished", func(t *testing.T, f *proxyReverseFixture) {
		facade := f.p.soleFacade()
		facade.discovery.SetSubscribed(nil)
		facade.refreshPeerEndpoints()
		facade.discovery.SetSubscribed([]Node{f.node})
		facade.refreshPeerEndpoints()
	})
}

func TestProxyReverseRejectsInvalidPinBeforeDial(t *testing.T) {
	var calls atomic.Int32
	f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusNoContent)
	}))
	c := f.candidate(t)
	// A different certificate for the same principal is not the paired authority.
	clustertrusttest.WritePeerPin(t, f.selfDir, "uuid-peer")
	response, err := f.p.candidateTransport(c).RoundTrip(proxyReverseRequest(t, http.MethodPost, c.url.String(), nil))
	if err == nil {
		response.Body.Close()
		t.Fatal("invalid admitted pin reached HTTP")
	}
	if calls.Load() != 0 || f.directCalls.Load() != 0 || f.reverseCalls.Load() != 0 {
		t.Fatalf("handler/direct/reverse = %d/%d/%d, want 0/0/0", calls.Load(), f.directCalls.Load(), f.reverseCalls.Load())
	}
}

func TestProxyReverseTransportDoesNotReplayDispatchedHTTP(t *testing.T) {
	test := func(name string, direct bool, wantReverse int32) {
		t.Run(name, func(t *testing.T) {
			var calls atomic.Int32
			f := newProxyReverseFixture(t, "ollama", http.NotFoundHandler())
			f.directHealthy.Store(direct)
			// Drop the pinned ingress after it has received HTTP but before headers.
			f.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if _, ok := f.peer.mesh.VerifyClientPin(r); !ok {
					t.Error("dispatched request was not pinned")
					return
				}
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					t.Error(err)
					return
				}
				calls.Add(1)
				conn, _, err := http.NewResponseController(w).Hijack()
				if err != nil {
					t.Error(err)
					return
				}
				if err := conn.Close(); err != nil {
					t.Error(err)
				}
			})
			c := f.candidate(t)
			request := proxyReverseRequest(t, http.MethodPost, c.url.String()+"/v1/chat/completions", strings.NewReader(`{"model":"fixture-model"}`))
			request.Header.Set("Idempotency-Key", "inference-is-not-deduplicated")
			if response, err := f.p.candidateTransport(c).RoundTrip(request); err == nil {
				response.Body.Close()
				t.Fatal("dispatched connection loss returned a response")
			}
			if calls.Load() != 1 || f.directCalls.Load() != 1 || f.reverseCalls.Load() != wantReverse {
				t.Fatalf("handler/direct/reverse = %d/%d/%d, want 1/1/%d", calls.Load(), f.directCalls.Load(), f.reverseCalls.Load(), wantReverse)
			}
		})
	}
	test("direct dispatch", true, 0)
	test("reverse dispatch", false, 1)
}

func TestProxyReverseOwnerRetriesBeforeFirstBody(t *testing.T) {
	setForTest(t, &retryBackoff, []time.Duration{time.Millisecond})
	var calls atomic.Int32
	f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			// Fail before producing content, rather than timing out a sole
			// owner's healthy prefill. The owner must still retry read errors.
			w.Header().Set("Content-Length", "1")
			w.WriteHeader(http.StatusOK)
			if err := http.NewResponseController(w).Flush(); err != nil {
				t.Error(err)
				return
			}
			return
		}
		if _, err := io.WriteString(w, "recovered"); err != nil {
			t.Error(err)
		}
	}))
	recorder := httptest.NewRecorder()
	request := proxyReverseRequest(t, http.MethodPost, "http://localhost/v1/chat/completions", strings.NewReader(`{"model":"fixture-model"}`))
	f.p.soleFacade().handleHTTP(recorder, request)
	if recorder.Code != http.StatusOK || recorder.Body.String() != "recovered" {
		t.Fatalf("owner retry status/body = %d/%q", recorder.Code, recorder.Body.String())
	}
	if calls.Load() != 2 || f.reverseCalls.Load() != 2 {
		t.Fatalf("owner handler/reverse attempts = %d/%d, want 2/2", calls.Load(), f.reverseCalls.Load())
	}
}

func TestProxyReverseInterruptedExposedStreamIsNotReplayed(t *testing.T) {
	var calls atomic.Int32
	f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "text/plain")
		if _, err := io.WriteString(w, "first\n"); err != nil {
			t.Error(err)
			return
		}
		if err := http.NewResponseController(w).Flush(); err != nil {
			t.Error(err)
			return
		}
		conn, _, err := http.NewResponseController(w).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		if err := conn.Close(); err != nil {
			t.Error(err)
		}
	}))
	proxy := httptest.NewServer(http.HandlerFunc(f.p.soleFacade().handleHTTP))
	t.Cleanup(proxy.Close)
	request := proxyReverseRequest(t, http.MethodPost, proxy.URL+"/v1/chat/completions", strings.NewReader(`{"model":"fixture-model"}`))
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err == nil || response.StatusCode != http.StatusOK || string(body) != "first\n" {
		t.Fatalf("interrupted status/body/error = %d/%q/%v, want exposed first chunk then failure", response.StatusCode, body, err)
	}
	if calls.Load() != 1 || f.reverseCalls.Load() != 1 {
		t.Fatalf("interrupted handler/reverse = %d/%d, want 1/1", calls.Load(), f.reverseCalls.Load())
	}
}

func TestProxyReverseCancellationEndsExposedStream(t *testing.T) {
	var calls atomic.Int32
	cancelled := make(chan struct{})
	f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "text/plain")
		if _, err := io.WriteString(w, "first\n"); err != nil {
			t.Error(err)
			return
		}
		if err := http.NewResponseController(w).Flush(); err != nil {
			t.Error(err)
			return
		}
		<-r.Context().Done()
		close(cancelled)
	}))
	proxy := httptest.NewServer(http.HandlerFunc(f.p.soleFacade().handleHTTP))
	t.Cleanup(proxy.Close)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, proxy.URL+"/v1/chat/completions", strings.NewReader(`{"model":"fixture-model"}`))
	if err != nil {
		t.Fatal(err)
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body := make([]byte, len("first\n"))
	if _, err := io.ReadFull(response.Body, body); err != nil {
		t.Fatal(err)
	}
	if string(body) != "first\n" {
		t.Fatalf("first body = %q", body)
	}
	cancel()
	if _, err := io.ReadAll(response.Body); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled body error = %v", err)
	}
	select {
	case <-cancelled:
	case <-time.After(2 * time.Second):
		t.Fatal("reverse cancellation did not reach local engine")
	}
	if calls.Load() != 1 || f.reverseCalls.Load() != 1 {
		t.Fatalf("cancelled handler/reverse = %d/%d, want 1/1", calls.Load(), f.reverseCalls.Load())
	}
}

func reverseContinuityOrigin(t *testing.T, f *proxyReverseFixture) *continuityFixture {
	t.Helper()
	fx := &continuityFixture{p: f.p, w: &recordingWriter{}, returned: make(chan struct{}, 8), nodeID: "host-peer"}
	f.p.codec = NewCodec(fx.w)
	fx.origin = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() { fx.returned <- struct{}{} }()
		f.p.soleFacade().handleHTTP(w, r)
	}))
	t.Cleanup(fx.origin.Close)
	return fx
}

func TestProxyReverseManagedInterruption(t *testing.T) {
	var calls atomic.Int32
	f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, continuitySSEEvent+`data: {"TORN_SENTINEL":"`); err != nil {
			t.Error(err)
			return
		}
		if err := http.NewResponseController(w).Flush(); err != nil {
			t.Error(err)
			return
		}
		conn, _, err := http.NewResponseController(w).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		if err := conn.Close(); err != nil {
			t.Error(err)
		}
	}))
	fx := reverseContinuityOrigin(t, f)
	resp, data, err := fx.request(t, "/v1/chat/completions", `{"model":"fixture-model","stream":true}`)
	if err != nil || resp.StatusCode != http.StatusOK || !bytes.HasPrefix(data, []byte(continuitySSEEvent)) || bytes.Contains(data, []byte("TORN_SENTINEL")) {
		t.Fatalf("status=%d length=%d read=%v", resp.StatusCode, len(data), err)
	}
	assertContinuityError(t, streamProtocolSSE, data, streamInterruptReasonReadFailed)
	fx.terminal(t, "failed", streamInterruptReasonReadFailed)
	if calls.Load() != 1 || f.reverseCalls.Load() != 1 {
		t.Fatalf("engine/reverse calls=%d/%d, want 1/1", calls.Load(), f.reverseCalls.Load())
	}
}

func TestProxyReverseManagedCancellation(t *testing.T) {
	var calls atomic.Int32
	cancelled := make(chan struct{})
	f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, continuitySSEEvent); err != nil {
			t.Error(err)
			return
		}
		if err := http.NewResponseController(w).Flush(); err != nil {
			t.Error(err)
			return
		}
		<-r.Context().Done()
		close(cancelled)
	}))
	fx := reverseContinuityOrigin(t, f)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fx.origin.URL+"/v1/chat/completions", strings.NewReader(`{"model":"fixture-model","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := fx.origin.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	data := make([]byte, len(continuitySSEEvent))
	if _, err := io.ReadFull(resp.Body, data); err != nil || string(data) != continuitySSEEvent {
		t.Fatalf("first frame length=%d read=%v", len(data), err)
	}
	cancel()
	if _, err := io.ReadAll(resp.Body); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled read=%v", err)
	}
	waitChan(t, cancelled, "reverse cancellation at engine")
	waitChan(t, fx.returned, "origin return")
	fx.terminal(t, "cancelled", "")
	if calls.Load() != 1 || f.reverseCalls.Load() != 1 {
		t.Fatalf("engine/reverse calls=%d/%d, want 1/1", calls.Load(), f.reverseCalls.Load())
	}
}

func TestProxyReverseEffectiveFramingDefaults(t *testing.T) {
	test := func(name, path, request, defaults, media, payload, reason string) {
		t.Run(name, func(t *testing.T) {
			f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
					return
				}
				if !bytes.Equal(body, mergeDefaults([]byte(request), []byte(defaults))) {
					t.Error("peer engine did not receive effective defaults")
				}
				if r.Header.Get("Accept-Encoding") != "identity" {
					t.Error("peer engine must receive identity encoding")
				}
				w.Header().Set("Content-Type", media)
				w.Header().Set(peerStreamFramingHeader, "engine-spoof")
				if _, err := io.WriteString(w, payload); err != nil {
					t.Error(err)
				}
			}))
			pol := nodepolicy.Default()
			pol.Profiles = []nodepolicy.Profile{{Name: "m", Engine: "ollama", Model: "fixture-model", RequestOptions: json.RawMessage(defaults)}}
			f.peer.admission.setPolicy(pol)
			fx := reverseContinuityOrigin(t, f)
			resp, data, err := fx.request(t, path, request)
			if err != nil || resp.StatusCode != http.StatusOK || resp.Header.Get(peerStreamFramingHeader) != "" {
				t.Fatalf("status=%d read=%v internal metadata=%q", resp.StatusCode, err, resp.Header.Get(peerStreamFramingHeader))
			}
			if reason != "" {
				assertContinuityError(t, streamProtocolSSE, data, reason)
				fx.terminal(t, "failed", reason)
			} else {
				if string(data) != payload {
					t.Fatalf("raw response changed: length=%d want=%d", len(data), len(payload))
				}
				fx.terminal(t, "completed", "")
			}
			if f.reverseCalls.Load() != 1 {
				t.Fatalf("reverse calls=%d, want 1", f.reverseCalls.Load())
			}
		})
	}
	test("destination n two truncated", "/v1/chat/completions", `{"model":"fixture-model","stream":true}`, `{"n":2}`, "text/event-stream", "data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"}]}\n\n", streamInterruptReasonPrematureEOF)
	test("destination n two complete", "/v1/chat/completions", `{"model":"fixture-model","stream":true}`, `{"n":2}`, "text/event-stream", "data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"},{\"index\":1,\"finish_reason\":\"stop\"}]}\n\n", "")
	test("destination disables streaming", "/api/generate", `{"model":"fixture-model"}`, `{"stream":false}`, "application/json", `{"response":"`+strings.Repeat("x", 1<<20)+`","done":true}`, "")
}

func TestProxyReverseRejectsIncompatibleFraming(t *testing.T) {
	test := func(name string, metadata []string, fallback bool) {
		t.Run(name, func(t *testing.T) {
			setForTest(t, &retryBackoff, []time.Duration{time.Millisecond})
			f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				t.Error("incompatible peer must bypass updated ingress")
			}))
			var peerCalls atomic.Int32
			// Simulate an older or incompatible, still pin-authenticated peer.
			f.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				peerCalls.Add(1)
				for _, value := range metadata {
					w.Header().Add(peerStreamFramingHeader, value)
				}
				w.Header().Set("Content-Type", "text/event-stream")
				if _, err := io.WriteString(w, "data: {\"LEAK_SENTINEL\":true}\n\n"); err != nil {
					t.Error(err)
				}
			})
			fx := reverseContinuityOrigin(t, f)
			if fallback {
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "text/event-stream")
					if _, err := io.WriteString(w, "data: [DONE]\n\n"); err != nil {
						t.Error(err)
					}
				}))
				defer upstream.Close()
				f.p.soleFacade().discovery.AddManual(nodeForModel(t, "fallback", upstream.URL, "fixture-model"))
				f.p.SetPriority([]string{"host-peer", "fallback"})
				fx.nodeID = "fallback"
			}
			resp, data, err := fx.request(t, "/v1/chat/completions", `{"model":"fixture-model","stream":true}`)
			if err != nil || bytes.Contains(data, []byte("LEAK_SENTINEL")) || resp.Header.Get(peerStreamFramingHeader) != "" {
				t.Fatalf("read=%v leaked body=%v metadata=%q", err, bytes.Contains(data, []byte("LEAK_SENTINEL")), resp.Header.Get(peerStreamFramingHeader))
			}
			started := notificationsOf(t, fx.w, "ollama:proxy/request-started")
			if fallback {
				if resp.StatusCode != http.StatusOK || string(data) != "data: [DONE]\n\n" || peerCalls.Load() != 1 || len(started) != 1 {
					t.Fatalf("failover status=%d peer calls=%d starts=%d", resp.StatusCode, peerCalls.Load(), len(started))
				}
				fx.terminal(t, "completed", "")
			} else {
				want := (&peerFramingMetadataError{invalid: len(metadata) != 0}).Error()
				if resp.StatusCode != http.StatusBadGateway || len(started) != 0 || peerCalls.Load() != maxDispatchAttempts {
					t.Fatalf("exhausted status=%d peer calls=%d starts=%d", resp.StatusCode, peerCalls.Load(), len(started))
				}
				var object struct{ Error string }
				if err := json.Unmarshal(data, &object); err != nil || object.Error != "upstream error: "+want {
					t.Fatalf("explicit failure=%+v decode=%v", object, err)
				}
				wl := fx.terminal(t, "failed", "")
				if wl.Error == nil || *wl.Error != want {
					t.Fatalf("terminal error=%v want=%s", wl.Error, want)
				}
				requests := notificationsOf(t, fx.w, "ollama:proxy/request")
				var event RequestEvent
				if len(requests) != 1 {
					t.Fatalf("proxy/request count=%d", len(requests))
				}
				if err := json.Unmarshal(requests[0], &event); err != nil || event.Error != want || event.Status != http.StatusBadGateway {
					t.Fatalf("proxy/request=%+v decode=%v", event, err)
				}
			}
		})
	}
	test("missing fails over", nil, true)
	test("missing fails explicitly", nil, false)
	test("invalid fails over", []string{"v2;declared;1"}, true)
	test("invalid fails explicitly", []string{"v1;declared;0"}, false)
	test("duplicate fails explicitly", []string{"v1;declared;1", "v1;declared;1"}, false)
}

func TestProxyReverseFramingMetadataValidation(t *testing.T) {
	request := originStreamRequest{protocol: streamProtocolSSE, choices: 1}
	for _, value := range []string{"", "v2;declared;1", "v1;unknown;1", "v1;declared;-1", "v1;declared;+1", "v1;declared;9999999999999999999999", "v1;declared;1;extra"} {
		t.Run(value, func(t *testing.T) {
			_, err := request.withPeerMetadata([]string{value})
			var typed *peerFramingMetadataError
			if !errors.As(err, &typed) || !typed.invalid {
				t.Fatalf("invalid metadata returned %v; want typed incompatibility", err)
			}
		})
	}
}

func TestProxyReverseFramingExcludesUnsupportedRoutes(t *testing.T) {
	f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(peerStreamFramingHeader, "engine-spoof")
		if _, err := io.WriteString(w, "raw"); err != nil {
			t.Error(err)
		}
	}))
	proxy := httptest.NewServer(http.HandlerFunc(f.p.soleFacade().handleHTTP))
	defer proxy.Close()
	for _, path := range []string{"/v1/responses", "/api/version"} {
		t.Run(path, func(t *testing.T) {
			resp, err := proxy.Client().Post(proxy.URL+path, "application/json", strings.NewReader(`{"model":"fixture-model","stream":true}`))
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			data, err := io.ReadAll(resp.Body)
			if err != nil || resp.StatusCode != http.StatusOK || string(data) != "raw" || resp.Header.Get(peerStreamFramingHeader) != "" {
				t.Fatalf("status=%d length=%d read=%v metadata=%q", resp.StatusCode, len(data), err, resp.Header.Get(peerStreamFramingHeader))
			}
		})
	}
}

func TestStreamContinuityPreservesUpgrade(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, rw, err := http.NewResponseController(w).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()
		if _, err := io.WriteString(rw, "HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: pair-fixture\r\nX-PAIR-Stream-Framing: engine-spoof\r\n\r\n"); err != nil {
			t.Error(err)
			return
		}
		if err := rw.Flush(); err != nil {
			t.Error(err)
			return
		}
		if _, err := io.CopyN(conn, rw, 1); err != nil {
			t.Error(err)
		}
	}))
	defer upstream.Close()
	fx := newContinuityFixture(t, upstream.URL, time.Second)
	proxy := fx.origin
	req := proxyReverseRequest(t, http.MethodGet, proxy.URL+"/api/version", nil)
	req.Header.Set("Connection", "Upgrade")
	req.Header.Set("Upgrade", "pair-fixture")
	resp, err := proxy.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusSwitchingProtocols || resp.Header.Get(peerStreamFramingHeader) != "" {
		t.Fatalf("upgrade status=%d internal metadata=%q", resp.StatusCode, resp.Header.Get(peerStreamFramingHeader))
	}
	duplex, ok := resp.Body.(io.ReadWriteCloser)
	if !ok {
		t.Fatal("upgrade lost its bidirectional body")
	}
	if _, err := duplex.Write([]byte{'x'}); err != nil {
		t.Fatal(err)
	}
	var echo [1]byte
	if _, err := io.ReadFull(duplex, echo[:]); err != nil || echo[0] != 'x' {
		t.Fatalf("upgrade echo read=%v", err)
	}
}

func TestProxyReverseConnectFailureRetainsUnreachableInventoryOwner(t *testing.T) {
	f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	f.directConnectFails.Store(true)
	service := f.p.soleFacade().profile.DiscoveryService
	snapshot, err := json.Marshal(noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{{
		HostUUID: "host-peer", Name: "restricted", IP: f.node.Addresses[0], ClusterUUID: "uuid-peer",
		Services:       map[noderec.ServiceKey]noderec.ServiceStatus{service: {Port: f.node.Port, Probe: noderec.ProbeUnreachable}},
		ModelsByEngine: map[string][]string{"ollama": {"fixture-model"}},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	f.p.soleFacade().replaceSubscribed(snapshot)
	c := f.candidate(t)
	if c.url.Host != f.server.Listener.Addr().String() {
		t.Fatalf("runtime port target = %q, want %q", c.url.Host, f.server.Listener.Addr().String())
	}
	proxyReverseRoundTrip(t, f.p.candidateTransport(c), c.url.String())
	if f.directCalls.Load() != 1 || f.reverseCalls.Load() != 1 {
		t.Fatalf("connect failure direct/reverse = %d/%d, want 1/1", f.directCalls.Load(), f.reverseCalls.Load())
	}
}

func TestProxyReverseDirectPrefersConfirmedPublishedAddress(t *testing.T) {
	f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	f.directHealthy.Store(true)
	node := f.node
	node.Addresses = []string{"192.0.2.10", f.node.Addresses[0]}
	f.p.soleFacade().discovery.SetSubscribed([]Node{node})
	f.p.soleFacade().targets = reach.NewChooser(reach.WithDial(func(_, address string, budget time.Duration) (net.Conn, error) {
		if address != f.server.Listener.Addr().String() {
			return nil, net.ErrClosed
		}
		return net.DialTimeout("tcp", address, budget)
	}))
	waitForTarget(t, f.p, node, f.server.Listener.Addr().String())
	c := f.candidate(t)
	if c.url.Host != f.server.Listener.Addr().String() {
		t.Fatalf("direct address = %q, want confirmed second published address", c.url.Host)
	}
	proxyReverseRoundTrip(t, f.p.candidateTransport(c), c.url.String())
	if f.directCalls.Load() != 1 || f.reverseCalls.Load() != 0 {
		t.Fatalf("multi-address direct/reverse = %d/%d, want 1/0", f.directCalls.Load(), f.reverseCalls.Load())
	}
}

func TestProxyReversePinsInnerServerCertificate(t *testing.T) {
	var calls atomic.Int32
	f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusNoContent)
	}))
	c := f.candidate(t)
	wrongIdentity := filepath.Join(t.TempDir(), "cluster")
	clustertrusttest.WriteKeypair(t, wrongIdentity, "uuid-peer")
	proxyReversePin(t, f.selfDir, wrongIdentity, "uuid-peer")
	response, err := f.p.candidateTransport(c).RoundTrip(proxyReverseRequest(t, http.MethodPost, c.url.String(), nil))
	if err == nil {
		response.Body.Close()
		t.Fatal("reverse inner TLS accepted a certificate outside the current pin")
	}
	if calls.Load() != 0 || f.directCalls.Load() != 1 || f.reverseCalls.Load() != 1 {
		t.Fatalf("wrong pin handler/direct/reverse = %d/%d/%d, want 0/1/1", calls.Load(), f.directCalls.Load(), f.reverseCalls.Load())
	}
}

func proxyReversePin(t *testing.T, destination, source, uuid string) {
	t.Helper()
	cert, err := os.ReadFile(filepath.Join(source, "node.crt"))
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(map[string]any{"nodeUuid": uuid, "certPem": string(cert),
		"clusterId": "fixture-cluster", "admissionEpoch": 1})
	if err != nil {
		t.Fatal(err)
	}
	trusted := filepath.Join(destination, "trusted")
	if err := os.MkdirAll(trusted, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(trusted, uuid+".json"), body, 0o600); err != nil {
		t.Fatal(err)
	}
	members, err := json.Marshal([]map[string]any{{"nodeUuid": uuid, "state": "member",
		"clusterId": "fixture-cluster", "admissionEpoch": 1}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(destination, "members.json"), members, 0o600); err != nil {
		t.Fatal(err)
	}
}

func (f *proxyReverseFixture) candidate(t *testing.T) candidate {
	t.Helper()
	candidates := f.p.soleFacade().resolveCandidates("fixture-model")
	if len(candidates) != 1 || candidates[0].peerUUID != "uuid-peer" {
		t.Fatalf("candidates = %+v, want the pinned fixture peer", candidates)
	}
	return candidates[0]
}

func proxyReverseRequest(t *testing.T, method, target string, body io.Reader) *http.Request {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	request, err := http.NewRequestWithContext(ctx, method, target, body)
	if err != nil {
		t.Fatal(err)
	}
	return request
}

func proxyReverseRoundTrip(t *testing.T, transport http.RoundTripper, target string) {
	t.Helper()
	response, err := transport.RoundTrip(proxyReverseRequest(t, http.MethodGet, target, nil))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusNoContent {
		t.Fatalf("status = %d, want 204", response.StatusCode)
	}
	if _, err := io.Copy(io.Discard, response.Body); err != nil {
		t.Fatal(err)
	}
}

func TestProxyReverseStreamsAllEngineServicesAndRuntimePorts(t *testing.T) {
	const firstEvent = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"first\"}}]}\n\n"
	const remainingEvents = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"second\"}}]}\n\ndata: [DONE]\n\n"
	for _, engine := range []string{"ollama", "lmstudio", "llamacpp"} {
		t.Run(engine, func(t *testing.T) {
			release := make(chan struct{})
			f := newProxyReverseFixture(t, engine, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body struct {
					Model string `json:"model"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					return
				}
				if body.Model != "fixture-model" || r.URL.Path != "/v1/chat/completions" {
					t.Errorf("request model/path = %q/%q", body.Model, r.URL.Path)
				}
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("X-Fixture", "preserved")
				if _, err := io.WriteString(w, firstEvent); err != nil {
					t.Error(err)
					return
				}
				if err := http.NewResponseController(w).Flush(); err != nil {
					t.Error(err)
					return
				}
				select {
				case <-release:
				case <-r.Context().Done():
					return
				}
				if _, err := io.WriteString(w, remainingEvents); err != nil {
					t.Error(err)
				}
			}))
			proxy := httptest.NewServer(http.HandlerFunc(f.p.soleFacade().handleHTTP))
			t.Cleanup(proxy.Close)
			request := proxyReverseRequest(t, http.MethodPost, proxy.URL+"/v1/chat/completions",
				strings.NewReader(`{"model":"fixture-model","stream":true}`))
			response, err := http.DefaultClient.Do(request)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			if response.StatusCode != http.StatusOK || response.Header.Get("X-Fixture") != "preserved" {
				t.Fatalf("status/header = %d/%q", response.StatusCode, response.Header.Get("X-Fixture"))
			}
			first := make([]byte, len(firstEvent))
			if _, err := io.ReadFull(response.Body, first); err != nil {
				t.Fatal(err)
			}
			if string(first) != firstEvent {
				t.Fatalf("first chunk = %q", first)
			}
			close(release)
			remaining, err := io.ReadAll(response.Body)
			if err != nil {
				t.Fatal(err)
			}
			if string(remaining) != remainingEvents {
				t.Fatalf("remaining body = %q", remaining)
			}
			if f.directCalls.Load() != 1 || f.reverseCalls.Load() != 1 {
				t.Fatalf("direct/reverse = %d/%d, want 1/1", f.directCalls.Load(), f.reverseCalls.Load())
			}
		})
	}
}

func TestProxyReverseInventoryUsesSelectedFacade(t *testing.T) {
	f := newProxyReverseFixture(t, "llamacpp", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" || r.Method != http.MethodGet {
			t.Errorf("inventory method/path = %s %s", r.Method, r.URL.Path)
		}
		if err := json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []map[string]string{{"id": "fixture-model"}}}); err != nil {
			t.Error(err)
		}
	}))
	recorder := httptest.NewRecorder()
	f.p.soleFacade().handleHTTP(recorder, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	var result struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if recorder.Code != http.StatusOK || len(result.Data) != 1 || result.Data[0].ID != "fixture-model" {
		t.Fatalf("inventory status/result = %d/%+v", recorder.Code, result)
	}
	if f.reverseCalls.Load() != 1 {
		t.Fatalf("reverse inventory dials = %d, want 1", f.reverseCalls.Load())
	}
}

func TestProxyReverseRejectsRemovedOrAmbiguousEndpoint(t *testing.T) {
	test := func(name string, change func(*proxyReverseFixture)) {
		t.Run(name, func(t *testing.T) {
			var calls atomic.Int32
			f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				w.WriteHeader(http.StatusNoContent)
			}))
			c := f.candidate(t)
			transport := f.p.candidateTransport(c)
			change(f)
			if response, err := transport.RoundTrip(proxyReverseRequest(t, http.MethodPost, c.url.String(), bytes.NewBufferString(`{}`))); err == nil {
				response.Body.Close()
				t.Fatal("stale endpoint retained dispatch authority")
			}
			if calls.Load() != 0 || f.directCalls.Load() != 0 || f.reverseCalls.Load() != 0 {
				t.Fatalf("handler/direct/reverse = %d/%d/%d, want 0/0/0", calls.Load(), f.directCalls.Load(), f.reverseCalls.Load())
			}
		})
	}
	test("removed endpoint", func(f *proxyReverseFixture) { f.p.soleFacade().discovery.SetSubscribed(nil) })
	test("runtime port changed", func(f *proxyReverseFixture) {
		node := f.node
		node.Port++
		f.p.soleFacade().discovery.SetSubscribed([]Node{node})
	})
	test("duplicate principal", func(f *proxyReverseFixture) {
		node := f.node
		node.ID = "other-host"
		f.p.soleFacade().discovery.SetSubscribed([]Node{f.node, node})
	})
}

// A hybrid peer attempt waits for a peer service slot no longer than the
// request's queue allowance, and an occupied peer costs it one attempt. A
// concrete attempt keeps blocking acquisition.
func TestProxyReverseHybridPeerSlotWaitIsBounded(t *testing.T) {
	release := make(chan struct{})
	var releaseOnce sync.Once
	stopHolding := func() { releaseOnce.Do(func() { close(release) }) }
	var held atomic.Int32
	fx := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			held.Add(1)
			select {
			case <-release:
			case <-r.Context().Done():
			}
		}
		if _, err := io.WriteString(w, "ok"); err != nil {
			t.Error(err)
		}
	}))
	defer stopHolding()
	fx.node.Roles = map[string]nodepolicy.AdvertisedRole{"weak": qualifiedRole("fixture-model")}
	fx.node.PolicyRevision = "1"
	// Ollama names the untagged model by its normalized tag.
	fx.node.StrictAdmission = strictCapabilityFor("ollama", "fixture-model:latest")
	fx.p.soleFacade().discovery.SetSubscribed([]Node{fx.node})
	policy := nodepolicy.Default()
	policy.Tiers = map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "fixture-model"}}}
	if err := policy.Validate(); err != nil {
		t.Fatal(err)
	}
	fx.p.admission.setPolicy(policy)

	candidate := fx.candidate(t)
	transport := fx.p.candidateTransport(candidate)
	var occupying sync.WaitGroup
	for range 8 {
		occupying.Add(1)
		go func() {
			defer occupying.Done()
			response, err := transport.RoundTrip(proxyReverseRequest(t, http.MethodGet, candidate.url.String()+"/api/tags", nil))
			if err != nil {
				return
			}
			_, _ = io.Copy(io.Discard, response.Body)
			_ = response.Body.Close()
		}()
	}
	deadline := time.Now().Add(5 * time.Second)
	for held.Load() < 8 {
		if time.Now().After(deadline) {
			t.Fatalf("held peer requests = %d, want every slot occupied", held.Load())
		}
		time.Sleep(5 * time.Millisecond)
	}

	hybrid := hybridTestPolicy(1, 3*time.Second)
	hybrid.queueWait = 50 * time.Millisecond
	started := time.Now()
	recorder, ctl := serveHybridDispatch(t, fx.p.soleFacade(), `{"model":"weak","messages":[{"role":"user","content":"hi"}]}`, hybrid)
	if recorder.Code != http.StatusBadGateway || !strings.Contains(recorder.Body.String(), clustertrust.ErrPeerSlotOccupied.Error()) {
		t.Fatalf("hybrid status = %d body = %q, want the occupied peer as the answer", recorder.Code, recorder.Body.String())
	}
	if ctl.ordinal != 1 {
		t.Fatalf("attempts = %d, want the occupied peer to cost exactly one", ctl.ordinal)
	}
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Fatalf("hybrid attempt waited %v for a slot, want about its 50 ms queue allowance", elapsed)
	}

	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		concrete := httptest.NewRecorder()
		fx.p.soleFacade().handleHTTP(concrete, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"fixture-model","messages":[]}`)))
		done <- concrete
	}()
	select {
	case concrete := <-done:
		t.Fatalf("concrete attempt did not wait for a slot: status %d body %q", concrete.Code, concrete.Body.String())
	case <-time.After(300 * time.Millisecond):
	}
	stopHolding()
	occupying.Wait()
	select {
	case concrete := <-done:
		if concrete.Code != http.StatusOK || concrete.Body.String() != "ok" {
			t.Fatalf("concrete status = %d body = %q, want the peer's answer once a slot freed", concrete.Code, concrete.Body.String())
		}
	case <-time.After(5 * time.Second):
		t.Fatal("concrete attempt never acquired a released slot")
	}
}

func TestProxyReverseHybridPeerSlotUsesRemainingQueueAndDeadline(t *testing.T) {
	base := context.Background()
	peer := candidate{peerUUID: "peer"}
	for _, tc := range []struct {
		name             string
		queue, preOutput time.Duration
		want             time.Duration
	}{
		{"queue fits before destination", 2 * time.Second, 6 * time.Second, 2 * time.Second},
		{"equal queue and deadline", 5 * time.Second, 5 * time.Second, 5 * time.Second},
		{"deadline bounds queue", 8 * time.Second, 5 * time.Second, 5 * time.Second},
		{"zero queue", 0, 5 * time.Second, 0},
		{"expired deadline", 2 * time.Second, -time.Second, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, wait := peerSlotContext(base, dispatchPolicy{hybrid: true}, peer, tc.queue, tc.preOutput)
			if ctx == base || wait != tc.want {
				t.Fatalf("hybrid peer slot context = (%v, %v), want bound %v", ctx, wait, tc.want)
			}
		})
	}
	for _, tc := range []struct {
		name   string
		policy dispatchPolicy
		cand   candidate
	}{
		{name: "concrete peer", cand: peer},
		{name: "hybrid self", policy: dispatchPolicy{hybrid: true}, cand: candidate{self: true, peerUUID: "peer"}},
		{name: "hybrid direct", policy: dispatchPolicy{hybrid: true}, cand: candidate{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, wait := peerSlotContext(base, tc.policy, tc.cand, 5*time.Second, 5*time.Second)
			if ctx != base || wait != 0 {
				t.Fatalf("slot context = (%v, %v), want unbounded original context", ctx, wait)
			}
		})
	}
}

func TestUnifiedDispatchHybridStrictBudgetsAfterPeerSlot(t *testing.T) {
	test := func(name string, wait, queue, cold, pre time.Duration, wantQueue, wantCold, wantPre int, wantOK bool) {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				policy := hybridTestPolicy(1, pre)
				key := nodepolicy.RoutingRequestKey{OriginNodeID: "origin", ProxyRunID: "run", RequestID: "1"}
				model := nodepolicy.StrictAdmissionModel{Identity: strictRequest().Identity}
				time.Sleep(wait)
				req, ok := policy.strictRequest(key, model, queue, cold, wait)
				if ok != wantOK {
					t.Fatalf("available = %v, want %v", ok, wantOK)
				}
				if !ok {
					return
				}
				if req.QueueBudgetMs != wantQueue || req.ColdStartBudgetMs != wantCold || req.PreOutputBudgetMs != wantPre {
					t.Fatalf("budgets = queue %d cold %d pre-output %d, want %d/%d/%d", req.QueueBudgetMs, req.ColdStartBudgetMs, req.PreOutputBudgetMs, wantQueue, wantCold, wantPre)
				}
				if req.Request != key || req.Identity != model.Identity {
					t.Fatalf("request identity changed: %+v", req)
				}
				if err := req.Validate(); err != nil {
					t.Fatalf("invalid recomputed request: %v", err)
				}
			})
		})
	}
	test("free slot preserves allowances", 0, 6*time.Second, 8*time.Second, 10*time.Second, 6000, 8000, 10000, true)
	test("partial wait charged once", 2*time.Second, 6*time.Second, 8*time.Second, 10*time.Second, 4000, 6000, 8000, true)
	test("spent cold allowance clamps to zero", 3*time.Second, 6*time.Second, time.Second, 10*time.Second, 3000, 0, 7000, true)
	test("remaining deadline caps allowances", 2*time.Second, 9*time.Second, 10*time.Second, 5*time.Second, 3000, 3000, 3000, true)
	test("submillisecond remainder refuses dispatch", 9999500*time.Nanosecond, time.Second, time.Second, 10*time.Millisecond, 0, 0, 0, false)
	test("expired deadline refuses dispatch", 5*time.Second, 6*time.Second, 8*time.Second, 5*time.Second, 0, 0, 0, false)
}

func TestProxyReverseHybridPeerSlotRecomputesSentBudgets(t *testing.T) {
	test := func(name string, direct bool) {
		t.Run(name, func(t *testing.T) {
			fx := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				if _, err := io.WriteString(w, "{}"); err != nil {
					t.Errorf("write engine response: %v", err)
				}
			}))
			fx.directHealthy.Store(direct)
			fx.node.Roles = map[string]nodepolicy.AdvertisedRole{"weak": qualifiedRole("fixture-model")}
			fx.node.PolicyRevision = "1"
			fx.node.StrictAdmission = strictCapabilityFor("ollama", "fixture-model:latest")
			fx.p.soleFacade().discovery.SetSubscribed([]Node{fx.node})
			policy := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "fixture-model"}}})
			fx.p.admission.setPolicy(policy)
			headers := make(chan http.Header, 1)
			// Capture the envelope at authenticated ingress before strict admission.
			ingress := fx.server.Config.Handler
			fx.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodPost {
					headers <- r.Header.Clone()
					// This fixture measures the origin's envelope; use ordinary
					// destination admission after capturing it.
					r.Header.Del(nodepolicy.StrictAdmissionHeader)
				}
				ingress.ServeHTTP(w, r)
			})
			cand := fx.candidate(t)
			transport := fx.p.candidateTransport(cand)
			var held []*http.Response
			defer func() {
				for _, response := range held {
					if err := response.Body.Close(); err != nil {
						t.Errorf("close occupying response: %v", err)
					}
				}
			}()
			for range 8 {
				response, err := transport.RoundTrip(proxyReverseRequest(t, http.MethodGet, cand.url.String()+"/api/tags", nil))
				if err != nil {
					t.Fatal(err)
				}
				held = append(held, response)
			}
			released := make(chan struct{})
			go func() {
				defer close(released)
				time.Sleep(200 * time.Millisecond)
				if err := held[0].Body.Close(); err != nil {
					t.Errorf("release peer slot: %v", err)
				}
			}()
			hybrid := hybridTestPolicy(1, 8*time.Second)
			hybrid.queueWait, hybrid.coldWait = 3*time.Second, 4*time.Second
			recorder, ctl := serveHybridDispatch(t, fx.p.soleFacade(), hybridTestBody, hybrid)
			<-released
			if recorder.Code != http.StatusOK || ctl.ordinal != 1 {
				t.Fatalf("status=%d attempts=%d body=%q, want one successful peer attempt", recorder.Code, ctl.ordinal, recorder.Body.String())
			}
			req := decodeStrictHeader(t, <-headers)
			if req.QueueBudgetMs < 2000 || req.QueueBudgetMs > 2850 || req.ColdStartBudgetMs-req.QueueBudgetMs != 1000 {
				t.Fatalf("destination budgets=%+v, want only the actual slot wait deducted from queue/cold", req)
			}
			if req.PreOutputBudgetMs < 6500 || req.PreOutputBudgetMs > 7850 || req.Request != ctl.key {
				t.Fatalf("destination request=%+v, want elapsed pre-output time and the registry key", req)
			}
		})
	}
	test("direct", true)
	test("reverse", false)
}
