// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/httpcon"
	"nvpair-shared/noderec"
)

const (
	// engineManagerHTTPPort is the default model-list HTTP port (em).
	// Broker configuration supplies the port used for both spawn and discovery.
	engineManagerHTTPPort = 14322

	// engineControlPort is the default cluster-scoped mTLS control port (ec).
	engineControlPort = 14323

	// autoAdvertiseInterval is how often the broker polls a local engine to
	// decide whether to register it with (or unregister it from) the discovery
	// daemon (a 5s cadence).
	autoAdvertiseInterval = 5 * time.Second
)

// engineManagerPorts resolves the immutable startup configuration. Bare Broker
// fixtures and zero-valued workerPaths retain the product defaults.
func (b *Broker) engineManagerPorts() (httpPort, controlPort int) {
	httpPort, controlPort = b.engineMgrHTTPPort, b.engineMgrControlPort
	if httpPort == 0 {
		httpPort = engineManagerHTTPPort
	}
	if controlPort == 0 {
		controlPort = engineControlPort
	}
	return httpPort, controlPort
}

// runAutoAdvertiseEngine reconciles an engine using the configured port
// recorded in its runtime profile, without compatibility-port reconciliation.
func (b *Broker) runAutoAdvertiseEngine(ctx context.Context, profile engineProxyProfile) {
	client := &http.Client{Timeout: 2 * time.Second}
	ticker := time.NewTicker(b.advertisementInterval())
	defer ticker.Stop()

	b.reconcileAdvertiseEngine(profile, client)
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			b.reconcileAdvertiseEngine(profile, client)
		}
	}
}

func (b *Broker) reconcileAdvertiseEngine(profile engineProxyProfile, client *http.Client) {
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if b.loadEngineSettingsLocked() != nil && b.knownModelSettingsFenceLocked() {
		return
	}
	if b.modelSettingsPendingLocked(profile.Name) || b.advertisementWithdrawnLocked(profile.Name) {
		return
	}

	enginePort := int(b.engineProxy(profile).backendPort.Load())
	proxyPort := b.engineProxyListenPort(profile)
	credential := ""
	if enginePort > 0 {
		credential = b.localEngineCredential(profile.Name, enginePort)
	}
	up := enginePort > 0 &&
		proxyPort > 0 &&
		enginePort != proxyPort &&
		checkEngineHealth(profile, client, enginePort, credential)
	if up {
		b.registerService(noderec.RegisterParams{Service: profile.DiscoveryService, Port: proxyPort})
		b.setProxyLocalBackendKeyed(b.engineProxyHandle(profile), profile.Name, enginePort, true, credential)
		return
	}
	b.unregisterService(profile.DiscoveryService)
	b.setProxyLocalBackendKeyed(b.engineProxyHandle(profile), profile.Name, enginePort, false, "")
}

// proxyLocalBackend is the node/set-local-backend payload: the loopback engine
// the proxy's cluster mTLS ingress forwards to, and the proxy's own self
// candidate on the local routing path.
//
// Credential is the engine's native API key, when it has one. It is sent only
// over the broker's private stdio channel to its own proxy, never logged, and
// the proxy uses it only for that loopback engine.
type proxyLocalBackend struct {
	Engine     string `json:"engine"`
	Host       string `json:"host"`
	Port       int    `json:"port"`
	Healthy    bool   `json:"healthy"`
	Credential string `json:"credential,omitempty"`
}

// setProxyLocalBackend hands the proxy its local (loopback) engine endpoint, or
// clears it (healthy=false) when the engine is down / unresolved. Best-effort
// and idempotent — re-sent on every reconcile so a freshly (re)spawned proxy
// re-learns its backend within one poll interval. A nil proxy is a no-op.
func (b *Broker) setProxyLocalBackend(p *proxyProcess, engine string, port int, healthy bool) {
	if p == nil {
		return
	}
	credential := ""
	if healthy && port > 0 {
		credential = b.localEngineCredential(engine, port)
	}
	b.setProxyLocalBackendKeyed(p, engine, port, healthy, credential)
}

// setProxyLocalBackendKeyed is setProxyLocalBackend with the credential the
// caller already read for this port.
func (b *Broker) setProxyLocalBackendKeyed(p *proxyProcess, engine string, port int, healthy bool, credential string) {
	if p == nil {
		return
	}
	if healthy {
		r := b.nodePolicyRuntime()
		r.mu.Lock()
		healthy = !r.drains[engine]
		r.mu.Unlock()
		if !healthy {
			credential = ""
		}
	}
	raw, err := json.Marshal(proxyLocalBackend{
		Engine:     engine,
		Host:       "127.0.0.1",
		Port:       port,
		Healthy:    healthy,
		Credential: credential,
	})
	if err != nil {
		slog.Warn("encode local engine destination failed", "engine", engine, "err", err)
		return
	}
	if _, rpcErr, err := p.Call(context.Background(), engines.AddressMethod(engine, "node/set-local-backend"), raw); err != nil {
		slog.Warn("set local engine destination failed", "engine", engine, "err", err)
	} else if rpcErr != nil {
		slog.Warn("set local engine destination rejected", "engine", engine, "code", rpcErr.Code)
	}
}

func (b *Broker) advertisementWithdrawnLocked(engine string) bool {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.drains[engine]
}

// localEngineCredential asks engine-manager for the native API key of the
// engine's running instance (an owned launch or adopted command-mode server),
// and keeps it only when that instance serves the port about to be configured.
// Any failure gives no key. A protected engine then fails its health probe.
func (b *Broker) localEngineCredential(engine string, port int) string {
	em := b.getEngineMgr()
	if em == nil {
		return ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	params, _ := json.Marshal(map[string]string{"engine": engine})
	result, rpcErr, err := em.Call(ctx, engines.LocalCredentialMethod, params)
	if err != nil || rpcErr != nil {
		return ""
	}
	var answer engines.LocalCredential
	if json.Unmarshal(result, &answer) != nil || answer.Port != port {
		return ""
	}
	return answer.Credential
}

// localEnginePort asks engine-manager for the port the named engine is actually
// serving on. The bool says whether there is a port worth probing. A valid
// running:false response, missing manager or failed RPC all refuse health probes.
func (b *Broker) localEnginePort(engine string) (int, bool) {
	em := b.getEngineMgr()
	if em == nil {
		return 0, false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	params, _ := json.Marshal(map[string]string{"engine": engine})
	result, rpcErr, err := em.Call(ctx, "engine:status", params)
	if err != nil || rpcErr != nil {
		return 0, false
	}
	return runningEnginePort(result)
}

func runningEnginePort(result json.RawMessage) (int, bool) {
	var st struct {
		Running bool `json:"running"`
		Port    int  `json:"port"`
	}
	if json.Unmarshal(result, &st) != nil || !st.Running || st.Port <= 0 || st.Port > 65535 {
		return 0, false
	}
	return st.Port, true
}

// engineProxyListenPort returns an engine proxy's current listen port, or 0 if
// it is not supervised or has not reported ready. Used to refuse advertising an
// engine at its own proxy's port, which would be a self-forward loop.
func (b *Broker) engineProxyListenPort(profile engineProxyProfile) int {
	if p := b.engineProxyHandle(profile); p != nil {
		if ready, port := p.Status(profile.Name); ready {
			return port
		}
	}
	return 0
}

// checkEngineHealth reports whether a local engine is answering on the given
// port, by probing the path its own liveness convention uses. The port is
// resolved per poll (see localEnginePort) rather than hardcoded, so the proxy
// is never mistaken for the engine it fronts.
//
// With a native API key the probe carries it, and a redirect is answered as is
// rather than followed with the key.
func checkEngineHealth(profile engineProxyProfile, client *http.Client, port int, credential string) bool {
	req, err := http.NewRequest(http.MethodGet, fmt.Sprintf("http://localhost:%d%s", port, profile.HealthProbePath), nil)
	if err != nil {
		return false
	}
	if credential != "" {
		req.Header.Set("Authorization", "Bearer "+credential)
		keyed := *client
		keyed.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
		client = &keyed
	}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	httpcon.DrainAndClose(resp.Body)
	return resp.StatusCode == http.StatusOK
}
