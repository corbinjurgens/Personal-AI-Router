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
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
)

func awaitLMStudioCredentialUpdate(t *testing.T, updates <-chan proxyLocalBackend, port int, ready bool) {
	t.Helper()
	select {
	case update := <-updates:
		wantKey := ""
		if ready {
			wantKey = brokerTestKey
		}
		if update.Engine != "lmstudio" || update.Host != "127.0.0.1" || update.Port != port || update.Healthy != ready || update.Credential != wantKey {
			t.Fatal("proxy did not receive the exact port-bound credential and readiness")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("missing LM Studio proxy update")
	}
}

func assertLMStudioCredentialRegistration(t *testing.T, b *Broker, proxyPort int, ready bool) {
	t.Helper()
	registrations := b.regCache.Snapshot()
	if ready {
		if len(registrations) != 1 || registrations[0].Service != noderec.ServiceLMStudio || registrations[0].Port != proxyPort || len(registrations[0].TXT) != 0 {
			t.Fatal("discovery did not register exactly the LM Studio proxy port")
		}
	} else if len(registrations) != 0 {
		t.Fatal("ordinary engine readiness remained advertised")
	}
	assertCredentialAbsent(t, "advertised registration", registrations)
}

func TestLMStudioAdvertisementUsesLocalCredential(t *testing.T) {
	for _, scenario := range []string{"no key", "rejected key", "Clear", "wrong returned port", "engine down", "status unavailable", "credential unavailable", "manager unavailable"} {
		t.Run(scenario, func(t *testing.T) {
			logs := captureCredentialLogs(t)
			var requests, authenticated atomic.Int32
			server := httptest.NewServer(fixtureRequestHandler(http.MethodGet, "/v1/models", func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				if r.URL.Path != "/v1/models" || r.Header.Get("Authorization") != "Bearer "+brokerTestKey {
					w.WriteHeader(http.StatusUnauthorized)
					return
				}
				authenticated.Add(1)
				w.WriteHeader(http.StatusOK)
			}))
			t.Cleanup(server.Close)
			port := server.Listener.Addr().(*net.TCPAddr).Port
			const proxyPort = 25001
			b, out := newPolicyBroker(t)
			b.engineSettingsLoaded = true
			b.regCache = relay.NewRegistrationCache()
			updates := attachAdvertiserProxy(t, b, lmstudioProxyProfile, proxyPort)
			engine := attachFakeEngine(t, b)
			answer := func(failed bool) func(string, json.RawMessage) (any, string) {
				return func(method string, params json.RawMessage) (any, string) {
					if string(params) != `{"engine":"lmstudio"}` {
						t.Error("manager request did not select exactly LM Studio")
					}
					switch method {
					case "engine:status":
						if failed && scenario == "status unavailable" {
							return nil, "status unavailable"
						}
						return map[string]any{"running": !(failed && scenario == "engine down"), "port": port}, ""
					case engines.LocalCredentialMethod:
						credential := engines.LocalCredential{Credential: brokerTestKey, Port: port}
						if failed {
							switch scenario {
							case "no key", "Clear":
								credential.Credential = ""
							case "rejected key":
								credential.Credential = "rejected-test-key"
							case "wrong returned port":
								credential.Port++
							case "credential unavailable":
								return nil, "credential unavailable"
							}
						}
						return credential, ""
					default:
						t.Error("unexpected manager operation during advertisement")
						return nil, "unexpected operation"
					}
				}
			}
			engine.setAnswer(answer(false))
			b.reconcileAdvertiseLMStudio(server.Client())
			awaitLMStudioCredentialUpdate(t, updates, port, true)
			assertLMStudioCredentialRegistration(t, b, proxyPort, true)
			if requests.Load() != 1 || authenticated.Load() != 1 {
				t.Fatal("saved key did not authenticate exactly one local health request")
			}
			engine.setAnswer(answer(true))
			if scenario == "manager unavailable" {
				b.setEngineMgr(nil)
			}
			b.reconcileAdvertiseLMStudio(server.Client())
			wantPort := port
			if scenario == "engine down" || scenario == "status unavailable" || scenario == "manager unavailable" {
				wantPort = 0
				if requests.Load() != 1 {
					t.Fatal("unavailable engine was health-probed")
				}
			} else if requests.Load() != 2 {
				t.Fatal("credential refusal did not exercise the health listener")
			}
			awaitLMStudioCredentialUpdate(t, updates, wantPort, false)
			assertLMStudioCredentialRegistration(t, b, proxyPort, false)
			if authenticated.Load() != 1 {
				t.Fatal("failed or wrong-port credential authenticated the listener")
			}
			assertCredentialAbsent(t, "client output", out.frames())
			assertCredentialLogsClean(t, logs)
		})
	}
}

func TestLMStudioAdvertisementUsesLocalCredentialExcludesProxyPort(t *testing.T) {
	for _, cached := range []bool{false, true} {
		name := "no cached engine port"
		if cached {
			name = "cached effective engine port"
		}
		t.Run(name, func(t *testing.T) {
			var hits atomic.Int32
			server := httptest.NewServer(fixtureRequestHandler(http.MethodGet, "/v1/models", func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				if r.Header.Get("Authorization") != "Bearer "+brokerTestKey {
					w.WriteHeader(http.StatusUnauthorized)
				}
			}))
			t.Cleanup(server.Close)
			port := server.Listener.Addr().(*net.TCPAddr).Port
			const proxyPort = 25001
			b, _ := newPolicyBroker(t)
			b.engineSettingsLoaded = true
			b.regCache = relay.NewRegistrationCache()
			updates := attachAdvertiserProxy(t, b, lmstudioProxyProfile, proxyPort)
			if cached {
				b.lmstudioState().backendPort.Store(int32(port))
			}
			engine := attachFakeEngine(t, b)
			engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == "engine:status" {
					return map[string]any{"running": true, "port": proxyPort}, ""
				}
				if method != engines.LocalCredentialMethod || string(raw) != `{"engine":"lmstudio"}` {
					t.Error("unexpected credential lookup")
				}
				return engines.LocalCredential{Credential: brokerTestKey, Port: port}, ""
			})
			b.reconcileAdvertiseLMStudio(server.Client())
			wantPort, wantCalls, wantHits := 0, 1, int32(0)
			if cached {
				wantPort, wantCalls, wantHits = port, 2, 1
			}
			awaitLMStudioCredentialUpdate(t, updates, wantPort, cached)
			assertLMStudioCredentialRegistration(t, b, proxyPort, cached)
			if len(engine.snapshot()) != wantCalls || hits.Load() != wantHits {
				t.Fatal("proxy collision did not gate credential lookup and health probing")
			}
		})
	}
}

func TestLMStudioAdvertisementUsesLocalCredentialPreservesManagedFacade(t *testing.T) {
	h := newBindingOfferHarness(t)
	server := httptest.NewServer(fixtureRequestHandler(http.MethodGet, "/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	t.Cleanup(server.Close)
	port := server.Listener.Addr().(*net.TCPAddr).Port
	base := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:status":
			return map[string]any{"running": true, "port": port}, ""
		case engines.LocalCredentialMethod:
			return engines.LocalCredential{Credential: brokerTestKey, Port: port}, ""
		default:
			return base(method, raw)
		}
	})
	h.b.reconcileAdvertiseLMStudio(server.Client())
	_, proxyPort := h.pp.Status("lmstudio")
	assertLMStudioCredentialRegistration(t, h.b, proxyPort, true)
	if !h.b.lmstudioState().bindingRegistered.Load() {
		t.Fatal("authentication refusal removed the managed cold facade")
	}
	var found bool
	for _, call := range h.proxy.snapshot() {
		if call.method == "lmstudio:node/set-local-backend" {
			var update proxyLocalBackend
			if json.Unmarshal(call.params, &update) != nil || update.Healthy || update.Credential != "" || update.Port != port {
				t.Fatal("cold facade retained a ready or keyed local engine")
			}
			found = true
		}
	}
	if !found {
		t.Fatal("cold facade did not receive the cleared local engine state")
	}
}

func TestLMStudioAdvertisementUsesLocalCredentialRefusesRedirect(t *testing.T) {
	var targetHits, targetHeaders, originAuthenticated atomic.Int32
	target := httptest.NewServer(fixtureRequestHandler(http.MethodGet, "/", func(w http.ResponseWriter, r *http.Request) {
		targetHits.Add(1)
		if r.Header.Get("Authorization") != "" {
			targetHeaders.Add(1)
		}
	}))
	t.Cleanup(target.Close)
	server := httptest.NewServer(fixtureRequestHandler(http.MethodGet, "/v1/models", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") == "Bearer "+brokerTestKey {
			originAuthenticated.Add(1)
		}
		http.Redirect(w, r, target.URL, http.StatusFound)
	}))
	t.Cleanup(server.Close)
	port := server.Listener.Addr().(*net.TCPAddr).Port
	b, _ := newPolicyBroker(t)
	assertFixtureRejectsProbe(t, server.URL)
	b.engineSettingsLoaded = true
	b.regCache = relay.NewRegistrationCache()
	updates := attachAdvertiserProxy(t, b, lmstudioProxyProfile, 25001)
	engine := attachFakeEngine(t, b)
	engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == "engine:status" {
			return map[string]any{"running": true, "port": port}, ""
		}
		return engines.LocalCredential{Credential: brokerTestKey, Port: port}, ""
	})
	b.reconcileAdvertiseLMStudio(server.Client())
	awaitLMStudioCredentialUpdate(t, updates, port, false)
	assertLMStudioCredentialRegistration(t, b, 25001, false)
	if originAuthenticated.Load() != 1 || targetHits.Load() != 0 || targetHeaders.Load() != 0 {
		t.Fatal("health probe failed to authenticate only the original loopback endpoint")
	}
}
