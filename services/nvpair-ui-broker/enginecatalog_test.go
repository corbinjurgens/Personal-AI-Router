// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/appdir"
	"nvpair-shared/engines"
	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
)

func testDescriptor(id string, protocols ...string) engines.Descriptor {
	return engines.Descriptor{
		Version: engines.DescriptorVersion, Engine: id, Family: engines.NameLlamaCPP, Label: id,
		Platforms: []string{"darwin/arm64"}, Available: true, Protocols: protocols,
		ModelNaming: engines.ModelNamingExact, Operations: map[string]engines.OperationCapability{},
	}
}

func customEngineDescriptor() engines.Descriptor {
	return testDescriptor("llamacpp-custom", engines.ProtocolLlamaCPPRouter, engines.ProtocolOpenAIV1)
}

func TestSelectDynamicEnginesFailsClosed(t *testing.T) {
	unavailable := testDescriptor("offhost", engines.ProtocolOpenAIChat)
	unavailable.Available = false
	newer := testDescriptor("future", engines.ProtocolOpenAIChat)
	newer.Version++
	duplicate := testDescriptor("strata", engines.ProtocolOpenAIV1)
	duplicate.Label = "second"
	got := selectDynamicEngines([]engines.Descriptor{
		testDescriptor("strata", engines.ProtocolOpenAIChat),
		testDescriptor(engines.NameLlamaCPP, engines.ProtocolOpenAIV1),
		customEngineDescriptor(),
		unavailable,
		newer,
		testDescriptor("grpc-engine", "grpc"),
		testDescriptor("list-only", engines.ProtocolLlamaCPPRouter),
		duplicate,
	})
	var ids []string
	for _, d := range got {
		ids = append(ids, d.Engine)
	}
	if !reflect.DeepEqual(ids, []string{"llamacpp-custom", "strata"}) {
		t.Fatalf("fronted = %v, want the two frontable descriptor engines in id order", ids)
	}
	if got[1].Label != "strata" {
		t.Fatal("a repeated id replaced the first descriptor")
	}

	var many []engines.Descriptor
	for i := range engines.MaxDynamicEngines + 3 {
		many = append(many, testDescriptor(fmt.Sprintf("engine%02d", i), engines.ProtocolOpenAIChat))
	}
	if got := selectDynamicEngines(many); len(got) != engines.MaxDynamicEngines || got[0].Engine != "engine00" {
		t.Fatalf("bounded selection = %d entries starting %q", len(got), got[0].Engine)
	}
}

func TestDynamicEnginesPlanFacadesClearOfEveryOtherPort(t *testing.T) {
	isolateOllamaHostTestConfig(t)
	// A persisted port for strata is honoured by the proxy and kept off the plan
	// for every other engine.
	path, err := appdir.Path("strata-proxy-port.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(`{"port":18180}`), 0o600); err != nil {
		t.Fatal(err)
	}
	b := &Broker{proxyPath: "test-proxy", proxyEngines: engines.Names()}
	busy := map[int]bool{18181: true}
	b.setDynamicEngines([]engines.Descriptor{customEngineDescriptor(), testDescriptor("strata", engines.ProtocolOpenAIChat)},
		func(port int) bool { return !busy[port] })

	profiles := b.dynamicEngineProfiles()
	if len(profiles) != 2 || profiles[0].Name != "llamacpp-custom" || profiles[0].DiscoveryService != "e-llamacpp-custom" {
		t.Fatalf("profiles = %+v", profiles)
	}
	if profiles[0].Ownership != managedEngine || profiles[0].HealthProbePath != "/v1/models" {
		t.Fatalf("broker metadata = %+v", profiles[0])
	}
	custom, strata := b.engineProxy(profiles[0]), b.engineProxy(profiles[1])
	if got := int(custom.startupPort.Load()); got != 18182 {
		t.Fatalf("llamacpp-custom planned port = %d, want 18182 past strata's persisted 18180 and busy 18181", got)
	}
	if got := int(strata.startupPort.Load()); got != 18183 {
		t.Fatalf("strata planned port = %d, want 18183", got)
	}
	for _, p := range profiles {
		if !b.proxyEnabled(p) {
			t.Errorf("%s not enabled with a resolved proxy binary", p.Name)
		}
	}
	if fronted := b.frontedEngineProfiles(); !reflect.DeepEqual(fronted[:len(engineProxyProfiles)], engineProxyProfiles) {
		t.Fatal("descriptor engines displaced the built-in preparation order")
	}
	if (&Broker{}).proxyEnabled(profiles[0]) {
		t.Fatal("descriptor engine enabled without a proxy binary")
	}
}

// The descriptor read happens once at startup through engine-manager; a
// malformed answer fronts nothing.
func TestLoadDynamicEnginesFromEngineManager(t *testing.T) {
	for _, tc := range []struct {
		name   string
		answer any
		want   int
	}{
		{"descriptors", map[string]any{"engines": []engines.Descriptor{customEngineDescriptor()}}, 1},
		{"malformed", map[string]any{"engines": "nope"}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			isolateOllamaHostTestConfig(t)
			b := &Broker{proxyPath: "test-proxy"}
			worker, codec := newTestRPCWorkerPipe(t)
			b.setEngineMgr(worker)
			go func() {
				msg, err := codec.Read()
				if err != nil {
					return
				}
				if msg.Method != "engine:descriptors" {
					_ = codec.RespondError(msg.ID, -32601, "unexpected "+msg.Method)
					return
				}
				_ = codec.Respond(msg.ID, tc.answer)
			}()
			b.loadDynamicEngines()
			if got := len(b.dynamicEngineProfiles()); got != tc.want {
				t.Fatalf("fronted %d descriptor engines, want %d", got, tc.want)
			}
		})
	}
}

// The facade request carries the descriptor and the planned port, and a lost
// bind race is retried in process on another planned port.
func TestDynamicEngineFacadeEnableCarriesTheDescriptor(t *testing.T) {
	isolateOllamaHostTestConfig(t)
	b := &Broker{proxyPath: "test-proxy"}
	b.setDynamicEngines([]engines.Descriptor{customEngineDescriptor()}, func(int) bool { return true })
	profile := b.dynamicEngineProfiles()[0]

	proxyClient, proxyServer := net.Pipe()
	t.Cleanup(func() {
		_ = proxyClient.Close()
		_ = proxyServer.Close()
	})
	proxy := &proxyProcess{peer: NewPeer(NewCodec(proxyClient))}
	go proxy.peer.Serve(nil, nil)
	attempts := make(chan enableFacadeRequest, 2)
	serveFacadeEnable(t, proxyServer, map[int]bool{engines.DynamicFacadePortBase: true}, attempts)

	if err := b.enableEngineFacadeWithPortCheck(context.Background(), proxy, profile, ollamaHostAlias{}, func(int) bool { return true }); err != nil {
		t.Fatalf("enable descriptor engine facade: %v", err)
	}
	first, second := <-attempts, <-attempts
	if first.Engine != "llamacpp-custom" || first.Port != engines.DynamicFacadePortBase || first.Descriptor == nil ||
		!reflect.DeepEqual(*first.Descriptor, customEngineDescriptor()) || first.IgnorePersistedPort {
		t.Fatalf("first request = %+v", first)
	}
	if second.Port == first.Port || !second.IgnorePersistedPort || second.Descriptor == nil {
		t.Fatalf("fallback request = %+v", second)
	}
	if got := int(b.engineProxy(profile).startupPort.Load()); got != second.Port {
		t.Fatalf("fallback port not kept for the next spawn: %d, want %d", got, second.Port)
	}
}

// While engine-manager reports the descriptor engine running and it answers
// its health probe, its e-<id> service is registered for the authenticated and
// reverse paths and the proxy gets its loopback backend, addressed e-<id>:.
// When it stops, the registration is withdrawn.
func TestDynamicEngineAdvertisementRegistersAndWithdraws(t *testing.T) {
	isolateOllamaHostTestConfig(t)
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("health probe path = %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(engine.Close)
	_, portText, err := net.SplitHostPort(engine.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	enginePort, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatal(err)
	}

	b := &Broker{proxyPath: "test-proxy", regCache: relay.NewRegistrationCache()}
	b.setDynamicEngines([]engines.Descriptor{customEngineDescriptor()}, func(int) bool { return true })
	profile := b.dynamicEngineProfiles()[0]

	var running atomic.Bool
	running.Store(true)
	worker, emCodec := newTestRPCWorkerPipe(t)
	b.setEngineMgr(worker)
	go func() {
		for {
			msg, err := emCodec.Read()
			if err != nil {
				return
			}
			_ = emCodec.Respond(msg.ID, map[string]any{"running": running.Load(), "port": enginePort})
		}
	}()

	proxyClient, proxyServer := net.Pipe()
	t.Cleanup(func() {
		_ = proxyClient.Close()
		_ = proxyServer.Close()
	})
	const facadePort = 18190
	proxy := &proxyProcess{peer: NewPeer(NewCodec(proxyClient)), facadeState: readyFacade(profile.Name, facadePort)}
	go proxy.peer.Serve(nil, nil)
	b.setEngineProxyHandle(profile, proxy)
	type backendCall struct {
		method string
		params proxyLocalBackend
	}
	calls := make(chan backendCall, 4)
	go func() {
		codec := NewCodec(proxyServer)
		for {
			msg, err := codec.Read()
			if err != nil {
				return
			}
			var params proxyLocalBackend
			_ = json.Unmarshal(msg.Params, &params)
			calls <- backendCall{msg.Method, params}
			_ = codec.Respond(msg.ID, map[string]bool{"ok": true})
		}
	}()

	client := &http.Client{Timeout: 2 * time.Second}
	b.reconcileAdvertiseDynamicEngine(profile, client)
	if got := b.regCache.Snapshot(); len(got) != 1 || !reflect.DeepEqual(got[0], noderec.RegisterParams{Service: "e-llamacpp-custom", Port: facadePort}) {
		t.Fatalf("registrations = %+v, want e-llamacpp-custom on the facade port", got)
	}
	call := <-calls
	if call.method != "e-llamacpp-custom:node/set-local-backend" || !call.params.Healthy || call.params.Port != enginePort || call.params.Engine != "llamacpp-custom" {
		t.Fatalf("local backend call = %+v", call)
	}

	running.Store(false)
	b.reconcileAdvertiseDynamicEngine(profile, client)
	if got := b.regCache.Snapshot(); len(got) != 0 {
		t.Fatalf("stopped descriptor engine still registered: %+v", got)
	}
	if call := <-calls; call.params.Healthy {
		t.Fatalf("stopped descriptor engine backend = %+v", call)
	}
}

// A notification addressed to a fronted descriptor engine reaches its handler;
// one addressed to an engine the broker does not front is dropped.
func TestDynamicEngineNotificationsRouteToTheirProfile(t *testing.T) {
	b := &Broker{proxyPath: "test-proxy"}
	b.setDynamicEngines([]engines.Descriptor{customEngineDescriptor()}, func(int) bool { return true })
	profile := b.dynamicEngineProfiles()[0]
	b.proxyMu.Lock()
	b.setEngineProxySubscribed(profile, true)
	b.proxyMu.Unlock()
	if got, ok := b.dynamicEngineProfile("strata"); ok {
		t.Fatalf("unfronted engine resolved to %+v", got)
	}
	engine, bare := engines.SplitAddressedMethod("e-llamacpp-custom:node/discovered")
	if engine != profile.Name || bare != "node/discovered" {
		t.Fatalf("split = %q, %q", engine, bare)
	}
	if method, ok := facadeMethodFor(profile, "e-llamacpp-custom:node/discovered"); !ok || method != "node/discovered" {
		t.Fatalf("facade method = %q, %v", method, ok)
	}
}
