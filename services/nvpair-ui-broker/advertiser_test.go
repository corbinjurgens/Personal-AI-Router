// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
)

func TestLocalEnginePortRequiresCurrentManagerStatus(t *testing.T) {
	b := &Broker{} // no engine-manager worker
	for _, engine := range []string{engines.NameLlamaCPP, "custom"} {
		if got, ok := b.localEnginePort(engine); ok || got != 0 {
			t.Fatalf("no engine-manager: %s port = (%d, %v), want (0, false)", engine, got, ok)
		}
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
	profile.DiscoveryService = noderec.ServiceLlamaCPP
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
	profile.DiscoveryService = noderec.ServiceLlamaCPP
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

// With no proxy supervised, engineProxyListenPort is 0,
// so the self-forward collision check (port == proxy port) never falsely trips.
func TestProxyListenPortNoProxy(t *testing.T) {
	b := &Broker{} // no proxy worker
	if got := b.engineProxyListenPort(mustEngineProxyProfile("llamacpp")); got != 0 {
		t.Errorf("no proxy: engineProxyListenPort = %d, want 0", got)
	}
}
