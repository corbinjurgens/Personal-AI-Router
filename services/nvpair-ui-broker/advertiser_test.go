// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
)

func TestLocalEnginePortRequiresCurrentManagerStatus(t *testing.T) {
	b := &Broker{} // no engine-manager worker
	for _, engine := range []string{"ollama", "lmstudio", "custom"} {
		if got, ok := b.localEnginePort(engine); ok || got != 0 {
			t.Fatalf("no engine-manager: %s port = (%d, %v), want (0, false)", engine, got, ok)
		}
	}
}

type advertiserTestTransport func(*http.Request) (*http.Response, error)

func (f advertiserTestTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestAdvertisersRequireCurrentManagerBeforeHealthAttempt(t *testing.T) {
	for _, engine := range []string{"ollama", "lmstudio", "custom"} {
		t.Run(engine, func(t *testing.T) {
			for _, scenario := range []string{"missing manager", "closed manager", "RPC error", "stopped", "malformed", "running"} {
				t.Run(scenario, func(t *testing.T) {
					profile := ollamaProxyProfile
					if engine == "lmstudio" {
						profile = lmstudioProxyProfile
					} else if engine == "custom" {
						profile = testDefaultEngineProxyProfile()
						profile.DiscoveryService = "e-fixedtest"
					}
					b := brokerWithEngineProxyProfile(profile)
					b.regCache = relay.NewRegistrationCache()
					// A previously confirmed port cannot substitute for current status.
					b.engineProxy(profile).backendPort.Store(25002)
					updates := attachAdvertiserProxy(t, b, profile, 25001)
					if scenario != "missing manager" {
						worker, codec := newTestRPCWorkerPipe(t)
						b.setEngineMgr(worker)
						if scenario == "closed manager" {
							worker.peer.Close()
						} else {
							go func() {
								msg, err := codec.Read()
								if err != nil {
									t.Errorf("read status: %v", err)
									return
								}
								var params struct {
									Engine string `json:"engine"`
								}
								if err := json.Unmarshal(msg.Params, &params); err != nil || msg.Method != "engine:status" || params.Engine != profile.Name {
									t.Errorf("status request = %+v, params=%+v, error=%v", msg, params, err)
								}
								switch scenario {
								case "RPC error":
									err = codec.RespondError(msg.ID, -32000, "unavailable")
								case "malformed":
									err = codec.Respond(msg.ID, "not a status")
								default:
									err = codec.Respond(msg.ID, map[string]any{"running": scenario == "running", "port": 25002})
								}
								if err != nil {
									t.Errorf("respond status: %v", err)
									return
								}
								if scenario == "running" {
									msg, err := codec.Read()
									if err != nil {
										t.Errorf("read local credential: %v", err)
										return
									}
									if msg.Method != engines.LocalCredentialMethod {
										t.Errorf("unexpected request %s", msg.Method)
									}
									if err := codec.Respond(msg.ID, engines.LocalCredential{Port: 25002}); err != nil {
										t.Errorf("respond local credential: %v", err)
									}
								}
							}()
						}
					}
					var attempts atomic.Int32
					client := &http.Client{Transport: advertiserTestTransport(func(r *http.Request) (*http.Response, error) {
						attempts.Add(1)
						if r.URL.Port() != "25002" {
							return nil, fmt.Errorf("unexpected health target %s", r.URL)
						}
						return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("{}"))}, nil
					})}
					switch engine {
					case "ollama":
						b.reconcileAdvertise(client)
					case "lmstudio":
						b.reconcileAdvertiseLMStudio(client)
					default:
						b.reconcileAdvertiseDynamicEngine(profile, client)
					}
					wantAttempts := int32(0)
					if scenario == "running" {
						wantAttempts = 1
					}
					if got := attempts.Load(); got != wantAttempts {
						t.Fatalf("health attempts = %d, want %d", got, wantAttempts)
					}
					select {
					case update := <-updates:
						if update.Healthy != (scenario == "running") {
							t.Fatalf("local backend = %+v", update)
						}
					case <-time.After(2 * time.Second):
						t.Fatal("proxy did not receive health result")
					}
					if got := b.regCache.Snapshot(); len(got) != int(wantAttempts) {
						t.Fatalf("registrations = %+v, want %d", got, wantAttempts)
					}
				})
			}
		})
	}
}

func TestRunningEnginePort(t *testing.T) {
	for _, tc := range []struct {
		name string
		raw  string
		port int
		ok   bool
	}{
		{name: "running", raw: `{"running":true,"port":1235}`, port: 1235, ok: true},
		{name: "stopped is authoritative", raw: `{"running":false,"port":1235}`},
		{name: "running without a port", raw: `{"running":true,"port":0}`},
		{name: "running with an invalid port", raw: `{"running":true,"port":65536}`},
		{name: "malformed response", raw: `{`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			port, ok := runningEnginePort([]byte(tc.raw))
			if port != tc.port || ok != tc.ok {
				t.Fatalf("runningEnginePort = (%d, %v), want (%d, %v)", port, ok, tc.port, tc.ok)
			}
		})
	}
}

func TestLMStudioMissingManagerClearsProxyDestination(t *testing.T) {
	proxyClient, proxyServer := net.Pipe()
	defer proxyClient.Close()
	defer proxyServer.Close()
	proxy := &proxyProcess{
		peer:        NewPeer(NewCodec(proxyClient)),
		facadeState: readyFacade(lmstudioProxyProfile.Name, defaultLMStudioPort),
	}
	go proxy.peer.Serve(nil, nil)

	localBackend := make(chan proxyLocalBackend, 1)
	go func() {
		codec := NewCodec(proxyServer)
		msg, err := codec.Read()
		if err != nil {
			return
		}
		var got proxyLocalBackend
		if json.Unmarshal(msg.Params, &got) == nil {
			localBackend <- got
		}
		_ = codec.Respond(msg.ID, map[string]bool{"ok": true})
	}()

	b := &Broker{regCache: relay.NewRegistrationCache()}
	b.setLMStudioProxy(proxy)

	// A nil client proves missing manager status refuses health HTTP.
	b.reconcileAdvertiseLMStudio(nil)
	if got := b.regCache.Snapshot(); len(got) != 0 {
		t.Fatalf("LM Studio proxy was advertised as an engine: %+v", got)
	}
	select {
	case got := <-localBackend:
		if got.Port != 0 || got.Healthy {
			t.Fatalf("proxy listener was retained as the local backend: %+v", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("LM Studio proxy did not receive a cleared local backend")
	}
}

func TestLMStudioMissingManagerPreservesKnownBackendCache(t *testing.T) {
	b := &Broker{regCache: relay.NewRegistrationCache()}
	b.lmstudioState().backendPort.Store(managedLMStudioBackendStart)

	b.reconcileAdvertiseLMStudio(nil)

	if got := int(b.lmstudioState().backendPort.Load()); got != managedLMStudioBackendStart {
		t.Fatalf("backend cache = %d, want %d", got, managedLMStudioBackendStart)
	}
}

func TestEngineAdvertiserTracksEngineHealth(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer backend.Close()
	_, portText, err := net.SplitHostPort(backend.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	backendPort, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatal(err)
	}
	proxyPort := 44000
	if backendPort == proxyPort {
		proxyPort++
	}

	profile := testDefaultEngineProxyProfile()
	profile.DiscoveryService = noderec.ServiceLMStudio
	b := brokerWithEngineProxyProfile(profile)
	b.regCache = relay.NewRegistrationCache()
	b.engineProxy(profile).backendPort.Store(int32(backendPort))
	updates := attachAdvertiserProxy(t, b, profile, proxyPort)

	b.reconcileAdvertiseEngine(profile, backend.Client())
	registrations := b.regCache.Snapshot()
	if len(registrations) != 1 || registrations[0].Service != profile.DiscoveryService ||
		registrations[0].Port != proxyPort {
		t.Fatalf("healthy registration = %+v, want %s on %d", registrations, profile.DiscoveryService, proxyPort)
	}
	if got := <-updates; !got.Healthy || got.Port != backendPort || got.Engine != profile.Name {
		t.Fatalf("healthy local backend = %+v", got)
	}

	backend.Close()
	b.reconcileAdvertiseEngine(profile, backend.Client())
	if got := b.regCache.Snapshot(); len(got) != 0 {
		t.Fatalf("unhealthy engine remained advertised: %+v", got)
	}
	if got := <-updates; got.Healthy || got.Port != backendPort {
		t.Fatalf("unhealthy local backend = %+v", got)
	}
}

func TestEngineAdvertiserRejectsSelfForwardLoop(t *testing.T) {
	profile := testDefaultEngineProxyProfile()
	profile.DiscoveryService = noderec.ServiceLMStudio
	b := brokerWithEngineProxyProfile(profile)
	b.regCache = relay.NewRegistrationCache()
	b.regCache.Register(noderec.RegisterParams{Service: profile.DiscoveryService, Port: 44000})
	b.engineProxy(profile).backendPort.Store(44000)
	updates := attachAdvertiserProxy(t, b, profile, 44000)

	// A nil client proves the collision check short-circuits before probing the
	// facade as though it were the backend.
	b.reconcileAdvertiseEngine(profile, nil)
	if got := b.regCache.Snapshot(); len(got) != 0 {
		t.Fatalf("self-forwarding facade remained advertised: %+v", got)
	}
	if got := <-updates; got.Healthy {
		t.Fatalf("self-forwarding backend remained healthy: %+v", got)
	}
}

func attachAdvertiserProxy(
	t *testing.T,
	b *Broker,
	profile engineProxyProfile,
	port int,
) <-chan proxyLocalBackend {
	t.Helper()
	proxyClient, proxyServer := net.Pipe()
	t.Cleanup(func() {
		_ = proxyClient.Close()
		_ = proxyServer.Close()
	})
	proxy := &proxyProcess{
		peer:        NewPeer(NewCodec(proxyClient)),
		facadeState: readyFacade(profile.Name, port),
	}
	go proxy.peer.Serve(nil, nil)
	b.setEngineProxyHandle(profile, proxy)

	updates := make(chan proxyLocalBackend, 4)
	go func() {
		codec := NewCodec(proxyServer)
		for {
			msg, err := codec.Read()
			if err != nil {
				return
			}
			var update proxyLocalBackend
			if json.Unmarshal(msg.Params, &update) == nil {
				updates <- update
			}
			_ = codec.Respond(msg.ID, map[string]bool{"ok": true})
		}
	}()
	return updates
}

// TestProxyListenPortNoProxy: with no proxy supervised, proxyListenPort is 0,
// so the self-forward collision check (port == proxy port) never falsely trips.
func TestProxyListenPortNoProxy(t *testing.T) {
	b := &Broker{} // no proxy worker
	if got := b.proxyListenPort(); got != 0 {
		t.Errorf("no proxy: proxyListenPort = %d, want 0", got)
	}
}

func TestOllamaFacadeIsPendingBackend(t *testing.T) {
	b := &Broker{}
	b.ollamaState().managedFacade.Store(true)
	b.ollamaState().backendPort.Store(managedOllamaFacadePort)
	if !b.ollamaFacadeIsPendingBackend() {
		t.Fatal("managed facade must block liveness probes while the backend still points at 11434")
	}
	b.ollamaState().backendPort.Store(11435)
	if b.ollamaFacadeIsPendingBackend() {
		t.Fatal("liveness probes should resume after the backend moves off 11434")
	}
	b.managedOllamaBackend.Store(11436)
	if !b.ollamaFacadeIsPendingBackend() {
		t.Fatal("a pending 11435 to 11436 move must keep liveness probes gated")
	}
	b.managedOllamaBackend.Store(0)
	b.ollamaMoveInFlight.Store(true)
	if !b.ollamaFacadeIsPendingBackend() {
		t.Fatal("an in-flight backend move must keep liveness probes gated")
	}
	b.ollamaMoveInFlight.Store(false)

	b.ollamaState().managedFacade.Store(false)
	b.ollamaState().backendPort.Store(managedOllamaFacadePort)
	b.setProxy(&proxyProcess{
		facadeState: readyFacade(ollamaProxyProfile.Name, managedOllamaFacadePort),
	})
	if !b.ollamaFacadeIsPendingBackend() {
		t.Fatal("recovery must keep probes blocked until the proxy vacates 11434")
	}
}
