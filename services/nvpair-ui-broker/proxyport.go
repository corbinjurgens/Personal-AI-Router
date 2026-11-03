// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"log/slog"

	"nvpair-shared/engines"
)

// proxyport.go owns the broker's shared proxy/engine port checks.
type enginePortStatus struct {
	Running bool `json:"running"`
	Port    int  `json:"port"`
}

type managedPortPlan struct {
	Enabled     bool
	BackendPort int
	Blocked     string
}

func nextAvailablePort(start int, available func(int) bool) int {
	for port := start; port <= 65535; port++ {
		if available(port) {
			return port
		}
	}
	return 0
}

func nextAvailablePortExcluding(start int, excludedPorts []int, available func(int) bool) int {
	excluded := map[int]bool{}
	for _, port := range excludedPorts {
		excluded[port] = true
	}
	return nextAvailablePort(start, func(port int) bool {
		return !excluded[port] && available(port)
	})
}

// portAvailable permits a fixture-owned check without changing the production
// port preflight or relying on a process-wide override.
func (b *Broker) portAvailable(port int) bool {
	if b.portAvailableCheck != nil {
		return b.portAvailableCheck(port)
	}
	return tcpPortAvailable(port)
}

// configureProxySupervisorCallbacks wires the one proxy supervisor.
//
// Every engine's facade lives in the supervised process, so a crash is reported
// once for the process and clears every engine's handle — shared fate is the
// accepted cost of putting all the facades' reservations in one place. Each
// facade shares the same process lifetime.
func (b *Broker) configureProxySupervisorCallbacks(sup *supervisor) {
	sup.onCrash, sup.onRecovered = b.supervisedWorkerCallbacks(engines.ProxyComponent, func() {
		// clearProxyHandles fences the captured process at the same workersMu
		// boundary as history acceptance, before monitor replacement.
		b.clearProxyHandles()
		// Capture before this monitor can respawn. Emission belongs to the
		// coalesced supervision owner, outside Stop's proxySupMu/join boundary.
		b.captureDepartedRunWorkloads("proxy crashed")
	})
	sup.onExhausted = func(attempt int) {
		slog.Warn("the proxy is terminally unavailable", "attempt", attempt)
	}
	sup.onSpawned = b.replayProxyStateAfterSpawn
}

func enginePortAssignmentRequest(method string, params json.RawMessage) (string, int, bool) {
	var request struct {
		Engine string `json:"engine"`
		Port   int    `json:"port"`
		Start  bool   `json:"start"`
	}
	if json.Unmarshal(params, &request) != nil || request.Engine == "" || request.Port <= 0 {
		return "", 0, false
	}
	switch method {
	case "engine:set-port", "engine:start":
		return request.Engine, request.Port, true
	case "engine:install":
		return request.Engine, request.Port, request.Start
	default:
		return "", 0, false
	}
}

// rejectProxyHeldPort refuses to assign the routing listener's port to an
// engine or facade, which would lose the bind or take it after a proxy restart.
func (b *Broker) rejectProxyHeldPort(msg *Message, port int, owner string) bool {
	reason := ""
	if b.routingListenerPort() == port {
		reason = fmt.Sprintf("port %d is the routing listener's port; change routing.listener.port in the node policy before assigning that port to %s",
			port, owner)
	}
	if reason == "" {
		return false
	}
	if err := b.codec.RespondError(msg.ID, -32000, reason); err != nil {
		log.Printf("failed to reject port assignment conflicting with a proxy-held port: %v", err)
	}
	return true
}

// routingListenerPort is the port the persisted policy has the routing listener
// serve, or 0 while it is disabled.
func (b *Broker) routingListenerPort() int {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	if l := r.current.Routing.Listener; l.Enabled {
		return l.Port
	}
	return 0
}

// facadePortOwner names the configured facade a port belongs to, or "". It
// covers what the broker knows of each: the engine's compatibility port or the
// one persisted in its place, the port planned for the next spawn, the port
// the running facade bound.
func (b *Broker) facadePortOwner(port int) string {
	for _, profile := range engineProxyProfiles {
		if !b.proxyEnabled(profile) {
			continue
		}
		planned := int(b.engineProxy(profile).startupPort.Load())
		bound := 0
		if pp := b.engineProxyHandle(profile); pp != nil {
			_, bound = pp.Status(profile.Name)
		}
		if port == configuredEngineProxyPort(profile) || port == planned || port == bound {
			return fmt.Sprintf("the %s proxy", profile.DisplayName)
		}
	}
	return ""
}

// runningEnginePorts asks nvpair-engine-manager which ports its running engines
// hold, so the proxy can be steered clear of them. A missing/slow
// engine-manager yields an empty set (no bump) — safe, because a port with no
// live listener can't actually collide at bind time; only a running engine is
// a real conflict.
func (b *Broker) runningEnginePorts() map[int]bool {
	ports := map[int]bool{}
	em := b.getEngineMgr()
	if em == nil {
		return ports
	}
	ctx, cancel := context.WithTimeout(context.Background(), rpcWorkerCallTimeout)
	defer cancel()
	result, rpcErr, err := em.Call(ctx, "engine:get-installed", nil)
	if err != nil || rpcErr != nil {
		slog.Warn("engine port lookup failed; skipping proxy port conflict check", "err", err, "rpcErr", rpcErr)
		return ports
	}
	var resp struct {
		Engines []struct {
			Running bool `json:"running"`
			Port    int  `json:"port"`
		} `json:"engines"`
	}
	if err := json.Unmarshal(result, &resp); err != nil {
		slog.Warn("engine port lookup returned unparseable result", "err", err)
		return ports
	}
	for _, e := range resp.Engines {
		if e.Running && e.Port > 0 {
			ports[e.Port] = true
		}
	}
	return ports
}

// nextFreeProxyPort returns requested when it's free, otherwise the lowest
// port above it not in taken (capped at 65535).
func nextFreeProxyPort(requested int, taken map[int]bool) int {
	p := requested
	for p < 65535 && taken[p] {
		p++
	}
	return p
}
