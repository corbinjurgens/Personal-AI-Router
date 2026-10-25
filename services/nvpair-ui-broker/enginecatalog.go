// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// enginecatalog.go fronts the descriptor-backed engines this node has
// registered beyond the built-in table: a llama.cpp fork under its own ID, or a
// user manifest such as Strata. Nothing here is per engine.
//
// The broker reads engine-manager's engine:descriptors once at startup, before
// it prepares facades, and fronts every descriptor that is available on this
// host and that engines.ProxyCanFront accepts. Each gets the ordinary generic
// machinery: a facade in the one proxy process (enabled with the descriptor,
// which is how the proxy learns its protocols), an engineProxyRuntime, and the
// default advertisement loop. Its backend port is whatever engine-manager
// reports for it; its facade port is planned from DynamicFacadePortBase and
// then persisted by the proxy.
//
// Registering its "e-<id>" service is what admits it on the authenticated
// peer-services path and the reverse transport; unregistering when the engine
// stops withdraws it from both. The set is fixed for the broker's lifetime,
// because engine-manager has no manifest reload: a new manifest takes effect on
// the next PAIR start. Unknown, malformed, unavailable or unfrontable
// descriptors are not fronted.

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"slices"
	"sort"
	"sync"
	"time"

	"nvpair-shared/engines"
)

// engineDescriptorsTimeout bounds the startup descriptor read. engine-manager
// answers it from memory, so a miss means it is wedged; the broker then fronts
// only the built-in engines rather than wait.
const engineDescriptorsTimeout = 5 * time.Second

// dynamicEngineCatalog is the broker's set of fronted descriptor engines. It is
// written once at startup and only read afterwards; mu is a leaf lock.
type dynamicEngineCatalog struct {
	mu          sync.RWMutex
	profiles    []engineProxyProfile
	descriptors map[string]engines.Descriptor
	runtimes    map[string]*engineProxyRuntime
}

// engineModelNaming resolves the built-in profiles or the installed catalog.
// The catalog lock is a leaf: this may be called while policy state is locked.
func (b *Broker) engineModelNaming(engine string) (string, bool) {
	if profile, ok := engineProxyProfileFor(engine); ok {
		if profile.Name == engines.NameOllama {
			return engines.ModelNamingImpliedLatestTag, true
		}
		return engines.ModelNamingExact, true
	}
	c := &b.dynamicEngines
	c.mu.RLock()
	defer c.mu.RUnlock()
	d, ok := c.descriptors[engine]
	return d.ModelNaming, ok && (d.ModelNaming == engines.ModelNamingExact || d.ModelNaming == engines.ModelNamingImpliedLatestTag)
}

// loadDynamicEngines reads engine-manager's descriptors and fronts the ones
// this node can. With no proxy binary there is nothing to front them with.
func (b *Broker) loadDynamicEngines() {
	em := b.getEngineMgr()
	if em == nil || b.proxyPath == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), engineDescriptorsTimeout)
	defer cancel()
	raw, rpcErr, err := em.Call(ctx, "engine:descriptors", nil)
	if err != nil || rpcErr != nil {
		slog.Warn("engine descriptors unavailable; fronting only the built-in engines", "err", err, "rpcErr", rpcErr)
		return
	}
	var result struct {
		Engines []engines.Descriptor `json:"engines"`
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		slog.Warn("engine descriptors malformed; fronting only the built-in engines", "err", err)
		return
	}
	b.setDynamicEngines(selectDynamicEngines(result.Engines), b.portAvailable)
}

// selectDynamicEngines keeps the descriptors this node fronts: not built in,
// available here, frontable, first of any repeated ID, sorted by ID and at
// most engines.MaxDynamicEngines of them.
func selectDynamicEngines(list []engines.Descriptor) []engines.Descriptor {
	var out []engines.Descriptor
	for _, d := range list {
		if _, builtin := engines.ByName(d.Engine); builtin {
			continue
		}
		if err := engines.ProxyCanFront(d); err != nil {
			slog.Warn("not fronting descriptor engine", "engine", d.Engine, "err", err)
			continue
		}
		if !d.Available {
			continue
		}
		if slices.ContainsFunc(out, func(have engines.Descriptor) bool { return have.Engine == d.Engine }) {
			continue
		}
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Engine < out[j].Engine })
	if len(out) > engines.MaxDynamicEngines {
		slog.Warn("too many descriptor engines; fronting the first by id",
			"count", len(out), "max", engines.MaxDynamicEngines)
		out = out[:engines.MaxDynamicEngines]
	}
	return out
}

// setDynamicEngines records the fronted descriptor engines and plans each
// facade's port: clear of every built-in engine's ports, every descriptor
// engine's persisted port and each port planned before it. The proxy restores
// an engine's persisted port in preference to this plan.
//
// Planning runs before the set is published: siblingEngineProxyPorts reads
// proxyEnabled, which reads this catalog.
func (b *Broker) setDynamicEngines(list []engines.Descriptor, available func(int) bool) {
	var profiles []engineProxyProfile
	descriptors := make(map[string]engines.Descriptor, len(list))
	runtimes := make(map[string]*engineProxyRuntime, len(list))
	excluded := []int{}
	for _, d := range list {
		identity, ok := engines.DynamicEngine(d.Engine, d.Label)
		if !ok {
			continue
		}
		profile := engineProxyProfile{Engine: identity, Ownership: managedEngine, HealthProbePath: dynamicHealthProbePath(d)}
		profiles = append(profiles, profile)
		descriptors[d.Engine] = d
		runtimes[d.Engine] = &engineProxyRuntime{profile: profile}
		if persisted := configuredEngineProxyPort(profile); persisted > 0 {
			excluded = append(excluded, persisted)
		}
	}
	for _, profile := range profiles {
		for port := range b.siblingEngineProxyPorts(profile) {
			excluded = append(excluded, port)
		}
		port := nextAvailablePortExcluding(engines.DynamicFacadePortBase, excluded, available)
		runtimes[profile.Name].startupPort.Store(int32(port))
		excluded = append(excluded, port)
		slog.Info("fronting descriptor engine", "engine", profile.Name, "service", string(profile.DiscoveryService), "plannedPort", port)
	}
	c := &b.dynamicEngines
	c.mu.Lock()
	c.profiles, c.descriptors, c.runtimes = profiles, descriptors, runtimes
	c.mu.Unlock()
}

// dynamicHealthProbePath is where a descriptor engine answers 200 when it is
// up: its OpenAI model list when it serves one, otherwise Ollama's.
func dynamicHealthProbePath(d engines.Descriptor) string {
	for _, p := range d.Protocols {
		if p == engines.ProtocolOpenAIV1 || p == engines.ProtocolOpenAIChat {
			return "/v1/models"
		}
	}
	return "/api/tags"
}

// dynamicEngineProfiles returns the fronted descriptor engines in ID order.
func (b *Broker) dynamicEngineProfiles() []engineProxyProfile {
	c := &b.dynamicEngines
	c.mu.RLock()
	defer c.mu.RUnlock()
	return slices.Clone(c.profiles)
}

// frontedEngineProfiles is every engine the broker may front: the built-in
// table in preparation order, then the descriptor engines.
func (b *Broker) frontedEngineProfiles() []engineProxyProfile {
	return slices.Concat(engineProxyProfiles, b.dynamicEngineProfiles())
}

// frontedEngineProfile resolves either a built-in or a fronted descriptor engine.
func (b *Broker) frontedEngineProfile(engine string) (engineProxyProfile, bool) {
	if p, ok := engineProxyProfileFor(engine); ok {
		return p, true
	}
	return b.dynamicEngineProfile(engine)
}

// dynamicEngineProfile resolves a fronted descriptor engine by ID.
func (b *Broker) dynamicEngineProfile(name string) (engineProxyProfile, bool) {
	c := &b.dynamicEngines
	c.mu.RLock()
	defer c.mu.RUnlock()
	if rt := c.runtimes[name]; rt != nil {
		return rt.profile, true
	}
	return engineProxyProfile{}, false
}

// dynamicEngineRuntime returns a fronted descriptor engine's runtime, or nil.
func (b *Broker) dynamicEngineRuntime(name string) *engineProxyRuntime {
	c := &b.dynamicEngines
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.runtimes[name]
}

// dynamicEngineFacadeSpec is a descriptor engine's facade/enable request. The
// descriptor travels with it, because the proxy has no table entry for the
// engine and builds its profile from the descriptor's protocols.
func (b *Broker) dynamicEngineFacadeSpec(profile engineProxyProfile) (enableFacadeRequest, error) {
	c := &b.dynamicEngines
	c.mu.RLock()
	d, ok := c.descriptors[profile.Name]
	c.mu.RUnlock()
	if !ok {
		return enableFacadeRequest{}, fmt.Errorf("engine %q is not a fronted descriptor engine", profile.Name)
	}
	return enableFacadeRequest{
		Engine:     profile.Name,
		Port:       int(b.engineProxy(profile).startupPort.Load()),
		Descriptor: &d,
	}, nil
}

// dynamicEngineFallbackPort picks the port to retry a descriptor engine's
// facade on after a lost bind race, and keeps it for this broker's lifetime.
func (b *Broker) dynamicEngineFallbackPort(profile engineProxyProfile, failed int, available func(int) bool) int {
	excluded := []int{failed}
	for port := range b.siblingEngineProxyPorts(profile) {
		excluded = append(excluded, port)
	}
	for _, other := range b.dynamicEngineProfiles() {
		if other.Name == profile.Name {
			continue
		}
		excluded = append(excluded, int(b.engineProxy(other).startupPort.Load()), b.engineProxyListenPort(other))
	}
	if backend := int(b.engineProxy(profile).backendPort.Load()); backend > 0 {
		excluded = append(excluded, backend)
	}
	fallback := nextAvailablePortExcluding(engines.DynamicFacadePortBase, excluded, available)
	b.engineProxy(profile).startupPort.Store(int32(fallback))
	return fallback
}

// dynamicEngineAvailabilityRunners advertises each descriptor engine while it
// is healthy, the same way the default engine path does.
func (b *Broker) dynamicEngineAvailabilityRunners() []func(context.Context) {
	var runners []func(context.Context)
	for _, profile := range b.dynamicEngineProfiles() {
		runners = append(runners, func(ctx context.Context) { b.runAutoAdvertiseDynamicEngine(ctx, profile) })
	}
	return runners
}

func (b *Broker) runAutoAdvertiseDynamicEngine(ctx context.Context, profile engineProxyProfile) {
	client := &http.Client{Timeout: 2 * time.Second}
	ticker := time.NewTicker(b.advertisementInterval())
	defer ticker.Stop()
	b.reconcileAdvertiseDynamicEngine(profile, client)
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			b.reconcileAdvertiseDynamicEngine(profile, client)
		}
	}
}

// reconcileAdvertiseDynamicEngine refreshes a descriptor engine's backend port
// from engine-manager, then advertises or withdraws it. A descriptor engine has
// no stock port to fall back on: when engine-manager does not report it
// running, its backend is unknown and it is withdrawn.
func (b *Broker) reconcileAdvertiseDynamicEngine(profile engineProxyProfile, client *http.Client) {
	port, running := b.localEnginePort(profile.Name, 0)
	if !running {
		port = 0
	}
	b.engineProxy(profile).backendPort.Store(int32(port))
	b.reconcileAdvertiseEngine(profile, client)
}
