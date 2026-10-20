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
	"strings"
	"sync/atomic"
	"testing"
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
	setForTest(t, &firstBodyTimeout, 50*time.Millisecond)
	setForTest(t, &retryBackoff, []time.Duration{time.Millisecond})
	var calls atomic.Int32
	f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.WriteHeader(http.StatusOK)
			if err := http.NewResponseController(w).Flush(); err != nil {
				t.Error(err)
				return
			}
			<-r.Context().Done()
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
