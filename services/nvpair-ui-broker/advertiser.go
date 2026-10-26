// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/httpcon"
	"nvpair-shared/noderec"
)

// An engine's stock port is its FacadePort in the engine table, used ONLY as a
// fallback when engine-manager cannot report the real one. Never hardcode the
// advertise or health port: the product-default proxy takes Ollama's 11434, so
// a fixed 11434 would advertise the proxy as Ollama and make the proxy — and
// peers — self-forward into a loop. The real port is resolved per poll via
// localEnginePort.
var (
	defaultOllamaPort   = ollamaProxyProfile.FacadePort
	defaultLMStudioPort = lmstudioProxyProfile.FacadePort
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

// runAutoAdvertise is the broker's ollama engine-registration loop. It polls
// the local ollama server on a fixed cadence and reconciles this node's ol
// service registration in the discovery daemon against it: register (with the
// served model list) when ollama is up, unregister when it goes away. The
// daemon folds the registration into this node's single _nvpair-node record, so
// peers discover the engine through the shared channel. Runs until ctx is
// cancelled (broker shutdown).
//
// (Pre-cutover this loop also spawned a nvpair-advertiser subprocess to publish an
// _nvpair-ollama record; that per-service advertisement was retired when the
// discovery consolidation landed and the binary was deleted.)
func (b *Broker) runAutoAdvertise(ctx context.Context) {
	client := &http.Client{Timeout: 2 * time.Second}
	ticker := time.NewTicker(b.advertisementInterval())
	defer ticker.Stop()

	b.reconcileAdvertise(client)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			b.reconcileAdvertise(client)
		}
	}
}

// reconcileAdvertise checks local ollama and brings this node's ol registration
// into line with it. Post-secure-inference the advertised ol endpoint is the
// promoted PROXY port, never the engine port: peers dial the proxy over cluster
// mTLS and it forwards to the loopback engine. The engine's real (loopback) port
// is a private detail handed only to the local proxy via node/set-local-backend.
//
//   - engine healthy + proxy up -> register {ol, PROXY port} + set-local-backend{enginePort, healthy}
//   - otherwise                  -> unregister ol + clear the proxy's local backend
//
// The model list is not carried here — it lives on engine-manager's em
// /v1/models endpoint (registered separately), which peers fetch during
// enrichment.
func (b *Broker) reconcileAdvertise(client *http.Client) {
	defer b.refreshBindingAdvertisement()
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if b.advertisementWithdrawnLocked("ollama") {
		return
	}
	// During the managed bind -> backend-move transition, engine:status would
	// probe :11434 and could mistake the proxy (or a remote response forwarded
	// through it) for an externally started Ollama. Do not query liveness until
	// the backend has moved or managed setup has safely fallen back.
	if b.ollamaFacadeIsPendingBackend() {
		b.unregisterService(noderec.ServiceOllama)
		b.setProxyLocalBackend(b.getProxy(), "ollama", 0, false)
		return
	}
	enginePort, probe := b.localEnginePort("ollama")
	proxyPort := b.proxyListenPort()
	// Advertise only when the engine is healthy AND the proxy is up AND the two
	// ports differ. Equal ports mean we can't tell the engine from the proxy
	// (or there is no separate engine), and setting the local backend to the
	// proxy's own port would make the ingress forward to itself.
	up := probe && proxyPort != 0 && enginePort != proxyPort && checkEngineHealth(ollamaProxyProfile, client, enginePort, "")
	if up {
		b.registerService(noderec.RegisterParams{Service: noderec.ServiceOllama, Port: proxyPort})
		b.setProxyLocalBackend(b.getProxy(), "ollama", enginePort, true)
	} else {
		if !b.registerBindingFacadeLocked(ollamaProxyProfile) {
			b.unregisterService(noderec.ServiceOllama)
		}
		b.setProxyLocalBackend(b.getProxy(), "ollama", enginePort, false)
	}
}

func (b *Broker) ollamaFacadeIsPendingBackend() bool {
	if b.managedOllamaBackend.Load() != 0 || b.ollamaMoveInFlight.Load() {
		return true
	}
	if int(b.ollamaState().backendPort.Load()) != managedOllamaFacadePort {
		return false
	}
	// Recovery flips managed mode off before it live-rebinds the proxy away
	// from :11434. Keep probes gated through that interval (and indefinitely if
	// the rebind fails) so the proxy can never be adopted as Ollama.
	return b.ollamaState().managedFacade.Load() || b.proxyListenPort() == managedOllamaFacadePort
}

// runAutoAdvertiseLMStudio is the LM Studio sibling of runAutoAdvertise: it
// polls the local LM Studio server and reconciles this node's lm service
// registration against it, so an LM Studio host appears on the cluster the same
// way an Ollama host does. Kept parallel to the Ollama path rather than folded
// into it: the two are a deliberate temporary pair, to be unified when the
// proxies are.
func (b *Broker) runAutoAdvertiseLMStudio(ctx context.Context) {
	client := &http.Client{Timeout: 2 * time.Second}
	ticker := time.NewTicker(b.advertisementInterval())
	defer ticker.Stop()

	b.reconcileAdvertiseLMStudio(client)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			b.reconcileAdvertiseLMStudio(client)
		}
	}
}

// reconcileAdvertiseLMStudio brings this node's lm registration into line with
// the local LM Studio server, mirroring reconcileAdvertise: it advertises the
// promoted proxy port (never the engine) and hands the engine's loopback port to
// the LM Studio proxy via node/set-local-backend.
func (b *Broker) reconcileAdvertiseLMStudio(client *http.Client) {
	defer b.refreshBindingAdvertisement()
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if b.advertisementWithdrawnLocked("lmstudio") {
		return
	}
	enginePort, probe := b.localEnginePort("lmstudio")
	proxyPort := b.lmstudioProxyListenPort()
	if proxyPort != 0 && enginePort == proxyPort {
		// Never hand the proxy its own listener as a local destination.
		if cached := int(b.lmstudioState().backendPort.Load()); cached > 0 && cached != proxyPort {
			enginePort = cached
		} else {
			enginePort = 0
			probe = false
		}
	}
	up := probe && proxyPort != 0 && enginePort != proxyPort && checkEngineHealth(lmstudioProxyProfile, client, enginePort, "")
	if up {
		b.registerService(noderec.RegisterParams{Service: noderec.ServiceLMStudio, Port: proxyPort})
		b.setProxyLocalBackend(b.getLMStudioProxy(), "lmstudio", enginePort, true)
	} else {
		if !b.registerBindingFacadeLocked(lmstudioProxyProfile) {
			b.unregisterService(noderec.ServiceLMStudio)
		}
		b.setProxyLocalBackend(b.getLMStudioProxy(), "lmstudio", enginePort, false)
	}
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
	defer b.refreshBindingAdvertisement()
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if b.advertisementWithdrawnLocked(profile.Name) {
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
	if !b.registerBindingFacadeLocked(profile) {
		b.unregisterService(profile.DiscoveryService)
	}
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
		if profile, ok := b.frontedEngineProfile(engine); ok {
			b.engineProxy(profile).bindingRegistered.Store(false)
		}
	}
	b.callProxyManual(p, engine, "node/set-local-backend", proxyLocalBackend{
		Engine:     engine,
		Host:       "127.0.0.1",
		Port:       port,
		Healthy:    healthy,
		Credential: credential,
	}, "local-backend")
}

// localEngineCredential asks engine-manager for the native API key of the
// engine's owned launch, and keeps it only when that launch serves the port
// about to be configured. Any failure gives no key; the engine then rejects
// PAIR's requests and is not advertised, with no unauthenticated fallback.
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

func (b *Broker) proxyListenPort() int {
	return b.engineProxyListenPort(ollamaProxyProfile)
}

func (b *Broker) lmstudioProxyListenPort() int {
	return b.engineProxyListenPort(lmstudioProxyProfile)
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
