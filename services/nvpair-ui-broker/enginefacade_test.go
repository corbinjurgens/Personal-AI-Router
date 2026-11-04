// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
)

func testDefaultEngineProxyProfile() engineProxyProfile {
	return engineProxyProfile{
		Engine: engines.Engine{
			Name:           "fixedtest",
			DisplayName:    "Fixed Test",
			FacadePort:     1233,
			EnginePortBase: 1234,
			PortFile:       "fixedtest-proxy-port.json",
		},
		Ownership:       managedEngine,
		HealthProbePath: "/health",
	}
}

func brokerWithEngineProxyProfile(profile engineProxyProfile) *Broker {
	b := &Broker{}
	b.engineProxiesOnce.Do(func() {
		b.engineProxies = map[string]*engineProxyRuntime{
			profile.Name: {profile: profile},
		}
		for _, sibling := range engineProxyProfiles {
			if sibling.Name != profile.Name {
				b.engineProxies[sibling.Name] = &engineProxyRuntime{profile: sibling}
			}
		}
	})
	return b
}

func TestDefaultEngineFacadeRetriesAwayFromReservedPorts(t *testing.T) {
	isolateBrokerTestConfig(t)
	profile := testDefaultEngineProxyProfile()
	profile.FacadePort = 8078
	profile.EnginePortBase = 8079
	b := brokerWithEngineProxyProfile(profile)

	proxyClient, proxyServer := net.Pipe()
	t.Cleanup(func() {
		_ = proxyClient.Close()
		_ = proxyServer.Close()
	})
	proxy := &proxyProcess{peer: NewPeer(NewCodec(proxyClient))}
	go proxy.peer.Serve(nil, nil)
	attempts := make(chan enableFacadeRequest, 2)
	serveFacadeEnable(t, proxyServer, map[int]bool{profile.FacadePort: true}, attempts)

	err := b.enableEngineFacadeWithPortCheck(
		context.Background(),
		proxy,
		profile,
		func(int) bool { return true },
	)
	if err != nil {
		t.Fatalf("enable engine facade: %v", err)
	}
	first, second := <-attempts, <-attempts
	if first.Port != profile.FacadePort {
		t.Fatalf("first port = %d, want stock facade %d", first.Port, profile.FacadePort)
	}
	// Skip this engine's 8079, llama.cpp's facade 8080 and its engine port 8081.
	if second.Port != 8082 {
		t.Fatalf("fallback port = %d, want 8082 after engine and sibling exclusions", second.Port)
	}
	if !second.IgnorePersistedPort {
		t.Fatal("fallback retry could restore the port that just failed")
	}

	restart := b.defaultEngineFacadeSpec(profile)
	if restart.Port != second.Port || !restart.IgnorePersistedPort {
		t.Fatalf("restart spec = %+v, want explicit fallback port %d", restart, second.Port)
	}
}

func TestLlamaCPPFacadePreparationPreservesConfiguredPorts(t *testing.T) {
	profile := mustEngineProxyProfile("llamacpp")
	for _, tc := range []struct {
		name       string
		serverPort int
		proxyPort  int
		explicit   bool
	}{
		{name: "manifest default", serverPort: profile.EnginePortBase},
		{name: "explicit settings", serverPort: 18081, proxyPort: 18080, explicit: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := &Broker{
				proxyPath:            "test-proxy",
				proxyEngines:         []string{profile.Name},
				engineSettingsLoaded: true,
			}
			if tc.explicit {
				b.engineSettings = map[string]*engineSettingsRecord{
					profile.Name: {Explicit: true, Snapshot: settings.Snapshot{
						Settings: settings.Config{ServerPort: tc.serverPort, ProxyPort: tc.proxyPort},
					}},
				}
			}
			worker, codec := newTestRPCWorkerPipe(t)
			b.setEngineMgr(worker)
			var calls atomic.Int32
			go func() {
				for {
					msg, err := codec.Read()
					if err != nil {
						return
					}
					calls.Add(1)
					if err := codec.Respond(msg.ID, enginePortStatus{Running: true, Port: tc.serverPort}); err != nil {
						t.Errorf("respond to unexpected engine request %s: %v", msg.Method, err)
						return
					}
				}
			}()

			b.prepareEnabledFacades()

			if got := calls.Load(); got != 0 {
				t.Fatalf("preparation issued %d engine requests, want no probing or relocation", got)
			}
			state := b.engineProxy(profile)
			if got := int(state.backendPort.Load()); got != tc.serverPort {
				t.Fatalf("engine port = %d, want %d", got, tc.serverPort)
			}
			if got := int(state.startupPort.Load()); got != tc.proxyPort {
				t.Fatalf("startup proxy port = %d, want %d", got, tc.proxyPort)
			}
			if got := state.explicitSettings.Load(); got != tc.explicit {
				t.Fatalf("explicit settings = %v, want %v", got, tc.explicit)
			}
		})
	}
}

func TestFacadePreparationUsesTheGenericPathForEveryTableEngine(t *testing.T) {
	for _, profile := range engineProxyProfiles {
		t.Run(profile.Name, func(t *testing.T) {
			b := &Broker{
				proxyPath: "test-proxy", proxyEngines: []string{profile.Name},
				engineSettingsLoaded: true,
			}
			b.prepareEnabledFacades()
			for _, candidate := range engineProxyProfiles {
				want := 0
				if candidate.Name == profile.Name {
					want = candidate.EnginePortBase
				}
				if got := int(b.engineProxy(candidate).backendPort.Load()); got != want {
					t.Fatalf("%s engine port = %d, want %d", candidate.Name, got, want)
				}
			}
			spec := b.defaultEngineFacadeSpec(profile)
			if spec.Engine != profile.Name || spec.Port != profile.FacadePort || spec.IgnorePersistedPort {
				t.Fatalf("facade spec = %+v, want %s on %d with ordinary port restore", spec, profile.Name, profile.FacadePort)
			}
		})
	}
}

func TestLlamaCPPProxyNotificationDispatchPreservesFacadeAddress(t *testing.T) {
	profile := mustEngineProxyProfile("llamacpp")
	client, server := net.Pipe()
	t.Cleanup(func() {
		_ = client.Close()
		_ = server.Close()
	})
	b := &Broker{codec: NewCodec(server)}
	b.proxyMu.Lock()
	b.setEngineProxySubscribed(profile, true)
	b.proxyMu.Unlock()

	payload := json.RawMessage(`{"port":8080}`)
	done := make(chan struct{})
	go func() {
		b.forwardProxyProcessNotification(profile.addressed("ready"), payload)
		close(done)
	}()
	if err := client.SetReadDeadline(time.Now().Add(2 * time.Second)); err != nil {
		t.Fatalf("set read deadline: %v", err)
	}
	msg, err := NewCodec(client).Read()
	if err != nil {
		t.Fatalf("read forwarded notification: %v", err)
	}
	if msg.Method != profile.ComponentName()+":ready" {
		t.Fatalf("notification method = %s, want %s:ready", msg.Method, profile.ComponentName())
	}
	var ready proxyReadyParams
	if err := json.Unmarshal(msg.Params, &ready); err != nil {
		t.Fatalf("decode ready notification: %v", err)
	}
	if ready.Port != 8080 {
		t.Fatalf("ready port = %d, want 8080", ready.Port)
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("notification forwarding did not finish")
	}
}

func TestProxyNotificationsRejectReplacedProcessForEveryTableEngine(t *testing.T) {
	for _, profile := range engineProxyProfiles {
		t.Run(profile.Name, func(t *testing.T) {
			var output bytes.Buffer
			b := &Broker{codec: NewCodec(&output)}
			previous, current := &proxyProcess{}, &proxyProcess{}
			b.setProxyProcessHandle(current)
			b.setEngineProxySubscribed(profile, true)
			params := json.RawMessage(`{"port":25080}`)
			b.forwardProxyNotificationFrom(previous, profile.addressed("ready"), params)
			if output.Len() != 0 {
				t.Fatal("replaced proxy forwarded a ready notification")
			}
			b.forwardProxyNotificationFrom(current, profile.addressed("ready"), params)
			msg, err := NewCodec(&output).Read()
			if err != nil {
				t.Fatalf("read current proxy notification: %v", err)
			}
			var ready proxyReadyParams
			if err := json.Unmarshal(msg.Params, &ready); err != nil {
				t.Fatalf("decode current proxy notification: %v", err)
			}
			if msg.Method != profile.ComponentName()+":ready" || ready.Port != 25080 {
				t.Fatalf("forwarded notification = %s %+v, want %s:ready on 25080", msg.Method, ready, profile.ComponentName())
			}
		})
	}
}
