// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"nvpair-shared/applog"
	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
)

// proxyCallTimeout bounds how long the broker waits for the proxy to answer a
// relayed control-plane request before giving up.
const proxyCallTimeout = 5 * time.Second

// proxyProcess is the broker's handle on its child nvpair-proxy: a full
// bidirectional JSON-RPC peer hosting a facade per enabled engine, each of
// which emits an engine-addressed "ready" notification carrying its bound port
// and accepts id-bearing control-plane requests. The id-correlation + read pump
// live in the shared jsonrpc.Peer; this handle owns the OS process and the
// per-engine readiness/port state.
type proxyProcess struct {
	// name identifies the process in shutdown diagnostics. It is
	// engines.ProxyComponent: one process covers every engine, so there is no
	// per-engine handle to name.
	name  string
	cmd   *exec.Cmd
	stdin io.WriteCloser
	peer  *Peer
	done  chan struct{}

	// credentialRefs is the provider credential reference set this process was
	// launched with, sorted. Its environment holds these and no other name in
	// the reserved namespace, so a policy that configures another set needs a
	// new process.
	credentialRefs []nodepolicy.CredentialRef

	// bringUpDone publishes listenerOnly after facade bring-up finishes. Until
	// then a listener change must leave the in-flight spawn alone.
	listenerOnly atomic.Bool
	bringUpDone  atomic.Bool

	// readyMu guards facadeState, written by the read pump when a facade's
	// addressed "ready" notification arrives and read by the broker to answer
	// <engine>-proxy:get-status / replay the baseline on a fresh subscribe.
	//
	// Keyed by engine because one process hosts several facades on different
	// ports. A single port for the process would be whichever facade readied
	// last, so get-status would hand clients another engine's port.
	readyMu     sync.Mutex
	facadeState map[string]proxyFacadeState

	// onNotify, if non-nil, is invoked for every notification the proxy emits
	// (including "ready") so the broker can forward its event stream.
	onNotify func(method string, params json.RawMessage)

	// relayDir is the broker's LAN directory. When a facade sends a
	// discovery:subscribe, the broker subscribes it here and pushes
	// discovery:nodes snapshots back down for that facade's routing set. nil
	// when the relay isn't wired (older call paths / tests).
	//
	// subIDs is keyed by engine because each facade subscribes for its own
	// engine's discovery service: one id per process would let a second
	// facade's subscribe replace the first's. Nil until the first subscribe.
	relayDir *relay.Directory
	subMu    sync.Mutex
	subIDs   map[string]int
}

// proxyReadyParams mirrors the proxy's "ready" notification payload: a facade
// binds its port synchronously and echoes it here, the authoritative source of
// that facade's listen port.
type proxyReadyParams struct {
	Version string `json:"version"`
	Port    int    `json:"port"`
}

// proxyFacadeState is what one facade has told the broker about itself.
type proxyFacadeState struct {
	ready bool
	port  int
	// params is the verbatim ready payload, replayed to a client that
	// subscribes after the fact so it does not have to wait for the next one.
	params json.RawMessage
}

// SetLogLevel forwards an already-validated log level as a log/set-level
// notification (through the peer's codec).
func (p *proxyProcess) SetLogLevel(level string) error {
	return p.peer.Notify(applog.SetLevelMethod, applog.SetLevelParams{Level: level})
}

// Done implements supervisedHandle: the returned channel closes once the proxy
// process has exited (cmd.Wait returned).
func (p *proxyProcess) Done() <-chan struct{} { return p.done }

// Status reports whether one engine's facade has announced itself ready and, if
// so, the HTTP port it bound (0 until its "ready" arrives).
func (p *proxyProcess) Status(engine string) (bool, int) {
	p.readyMu.Lock()
	defer p.readyMu.Unlock()
	state := p.facadeState[engine]
	return state.ready, state.port
}

// ReadyParams returns the verbatim payload of one facade's "ready"
// notification, or nil if that engine hasn't announced itself yet.
func (p *proxyProcess) ReadyParams(engine string) json.RawMessage {
	p.readyMu.Lock()
	defer p.readyMu.Unlock()
	return p.facadeState[engine].params
}

// startProxy spawns nvpair-proxy with the broker's current log level and an
// environment that keeps only the credential references refs from the reserved
// provider-secret namespace, hides the console window on Windows, and runs the
// peer read pump to capture each facade's "ready" notification (and thus its
// bound port). An environment that cannot be built fails the launch before any
// process starts. The process starts with no facade; the caller enables them.
// onNotify (may be nil) is invoked for every notification the proxy emits so
// the broker can forward its event stream. Proxy stderr goes to the broker's
// non-blocking sink (see stderrsink.go), so a stalled reader cannot block the
// proxy's exit.
func startProxy(name, binaryPath, logLevel string, refs []nodepolicy.CredentialRef, relayDir *relay.Directory, onNotify func(*proxyProcess, string, json.RawMessage), extraArgs ...string) (*proxyProcess, error) {
	return startProxyWithHistory(nil, name, binaryPath, logLevel, refs, relayDir, onNotify, extraArgs...)
}

// A nil history owner is the explicit ordinary constructor. Main always passes
// its owner; failure to attach that guard never selects an ordinary codec.
func startProxyWithHistory(history *brokerRoutingHistory, name, binaryPath, logLevel string, refs []nodepolicy.CredentialRef, relayDir *relay.Directory, onNotify func(*proxyProcess, string, json.RawMessage), extraArgs ...string) (*proxyProcess, error) {
	env, err := proxyEnv(os.Environ(), refs)
	if err != nil {
		return nil, err
	}
	args := append([]string{"--log-level", logLevel}, extraArgs...)
	cmd := exec.Command(binaryPath, args...)
	configureSubprocess(cmd)
	cmd.Env = env

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return nil, fmt.Errorf("stdout pipe: %w", err)
	}
	cmd.Stderr = stderrOut

	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		_ = stdout.Close()
		return nil, fmt.Errorf("start %s: %w", binaryPath, err)
	}

	pp := &proxyProcess{
		name:     name,
		cmd:      cmd,
		stdin:    stdin,
		done:     make(chan struct{}),
		relayDir: relayDir,
		// Compared with the policy by proxyCredentialsStale.
		credentialRefs: refs,
	}
	if onNotify != nil {
		pp.onNotify = func(method string, params json.RawMessage) { onNotify(pp, method, params) }
	}
	if history == nil {
		pp.attachPeer(readWriter{stdout, stdin})
	} else if err := pp.attachRoutingHistoryProxyPipe(history.b, stdout); err != nil {
		// No read pump or Wait goroutine owns this child yet. Close both pipes,
		// stop and reap this exact process before returning the launch failure.
		_ = stdin.Close()
		_ = stdout.Close()
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		pp.retireRoutingHistoryRead()
		return nil, fmt.Errorf("attach proxy routing history: %w", err)
	}

	go pp.servePeer()
	go func() {
		_ = cmd.Wait()
		pp.retireRoutingHistoryRead()
		pp.peer.Close()
		// Every facade's subscription goes, not just one: the process is gone,
		// so a surviving registration would push snapshots at a closed peer.
		pp.subMu.Lock()
		for engine, id := range pp.subIDs {
			if id != 0 && pp.relayDir != nil {
				pp.relayDir.Unsubscribe(id)
			}
			delete(pp.subIDs, engine)
		}
		pp.subMu.Unlock()
		close(pp.done)
	}()

	return pp, nil
}

func (b *Broker) forwardProxyNotificationFrom(source *proxyProcess, ollamaGeneration, lmstudioGeneration uint64, method string, params json.RawMessage) {
	if b.handleBoundEngineNotification(source, method, params) {
		return
	}
	if routingHistoryMethod(method) {
		b.routeRoutingHistoryFrom(source, method, params)
		return
	}
	if source == nil || b.proxyProcessHandle() != source {
		return
	}
	if method == nodepolicy.NotifyStrictReadiness {
		b.observeStrictReadinessFrom(source, params)
		return
	}
	engine, bare := engines.SplitAddressedMethod(method)
	if proxyWorkloadMethods[bare] {
		if engine != "" {
			_, known := engineProxyProfileFor(engine)
			if !known {
				_, known = b.dynamicEngineProfile(engine)
			}
			if !known {
				return
			}
		}
		// Keep the actual process source through the serialized apply/fence
		// boundary. Done does not join a notification already past this guard.
		b.routeProxyWorkloadFrom(source, bare, params)
		return
	}
	b.forwardProxyProcessNotification(ollamaGeneration, lmstudioGeneration, method, params)
}

// attachPeer builds the JSON-RPC peer over the proxy's pipes. A Call whose
// write blocks on a proxy that stopped reading is cancellable: see
// interruptWrite.
func (p *proxyProcess) attachPeer(rw io.ReadWriter) {
	p.peer = jsonrpc.NewPeerWithWriteInterrupt(NewCodec(rw), p.interruptWrite)
}

// interruptWrite releases a Call blocked writing to a proxy that stopped
// reading: an interrupted frame makes the proxy's input unusable, so close its
// pipe and end the process, and supervision replaces it. This hook must not
// join the process or wait for the interrupted Call.
func (p *proxyProcess) interruptWrite() {
	if p.stdin != nil {
		_ = p.stdin.Close()
	}
	if p.cmd != nil && p.cmd.Process != nil {
		_ = p.cmd.Process.Kill()
	}
}

// handleNotify records the proxy's "ready" port and forwards every notification
// to the broker. A malformed "ready" is logged and not forwarded (matching the
// pre-refactor behavior).
//
// Facade-scoped notifications arrive addressed to their engine, so the address
// is split off before dispatching on the method. The addressed form is what
// goes to onNotify: each consumer re-splits and checks the engine itself,
// rather than trusting a bare method to have come from the engine it expected.
func (p *proxyProcess) handleNotify(method string, params json.RawMessage) {
	engine, bare := engines.SplitAddressedMethod(method)
	if bare == "ready" {
		var rp proxyReadyParams
		if err := json.Unmarshal(params, &rp); err != nil {
			slog.Warn("proxy emitted invalid ready payload", "engine", engine, "err", err)
			return
		}
		p.readyMu.Lock()
		if p.facadeState == nil {
			p.facadeState = make(map[string]proxyFacadeState, len(engines.Names()))
		}
		p.facadeState[engine] = proxyFacadeState{
			ready:  true,
			port:   rp.Port,
			params: append(json.RawMessage(nil), params...),
		}
		p.readyMu.Unlock()
		slog.Info("proxy reported ready", "engine", engine, "version", rp.Version, "port", rp.Port)
	}
	// The proxy subscribes upward for its routing targets: wire it to
	// the relay directory and push events down, rather than forwarding this as a
	// client-facing event.
	if bare == noderec.MethodSubscribe {
		p.handleSubscribe(engine, params)
		return
	}
	if p.onNotify != nil {
		p.onNotify(method, params)
	}
}

// handleSubscribe wires one facade's discovery:subscribe into the relay
// directory: it registers a subscriber whose Send pushes a discovery:nodes
// snapshot down the proxy's peer, then sends the initial snapshot so the
// facade's routing set is populated immediately. A re-subscribe (e.g. the
// facade resends) drops that facade's prior registration so it isn't
// double-fed.
//
// Subscriptions are tracked per engine, not per process. Each facade filters on
// its own engine's discovery service, so a single id per process would make a
// second facade's subscribe silently replace the first's — leaving that engine
// with a routing set that never updates again.
//
// Snapshots are pushed back addressed to the requesting engine, so the child
// hands each one to the facade that asked for it.
func (p *proxyProcess) handleSubscribe(engine string, params json.RawMessage) {
	if p.relayDir == nil {
		return
	}
	send := func(nodes []noderec.DirectoryNode) {
		method := engines.AddressMethod(engine, noderec.NotifyNodes)
		if err := p.peer.Notify(method, noderec.GetNodesResult{Nodes: nodes}); err != nil {
			slog.Debug("failed to push node snapshot to proxy", "engine", engine, "err", err)
		}
	}
	p.subMu.Lock()
	if p.subIDs == nil {
		p.subIDs = make(map[string]int, len(engines.Names()))
	}
	if prior := p.subIDs[engine]; prior != 0 {
		p.relayDir.Unsubscribe(prior)
	}
	id, sub, err := subscribeRelay(p.relayDir, params, send)
	p.subIDs[engine] = id
	p.subMu.Unlock()
	if err != nil {
		slog.Warn("proxy sent invalid discovery:subscribe", "engine", engine, "err", err)
		return
	}
	p.relayDir.Deliver(sub)
}

// Call issues an id-bearing control-plane request to the proxy and blocks until
// the response arrives, ctx is cancelled, the proxy exits, or proxyCallTimeout
// elapses.
func (p *proxyProcess) Call(ctx context.Context, method string, params json.RawMessage) (json.RawMessage, *RPCError, error) {
	if p == nil || p.peer == nil {
		return nil, nil, fmt.Errorf("proxy peer not available")
	}
	ctx, cancel := context.WithTimeout(ctx, proxyCallTimeout)
	defer cancel()
	return p.peer.Call(ctx, method, params)
}

// Stop signals the proxy to exit by closing its stdin (observed as EOF; the
// proxy drains its HTTP server) and waits for it to exit, escalating if it does
// not — see waitForStdinClose. Safe to call multiple times.
func (p *proxyProcess) Stop() {
	waitForStdinClose(p.name, p.cmd, p.stdin, p.done)
}

// proxyProcessHandle returns the proxy process while it runs, whether or not
// any facade came up in it, or nil. A process that has exited is nil: a crash
// clears the per-engine handles but would leave this one pointing at a dead
// peer, and the policy would then report a failed push where it should report
// that no proxy runs.
func (b *Broker) proxyProcessHandle() *proxyProcess {
	b.workersMu.Lock()
	pp := b.proxyProc
	b.workersMu.Unlock()
	if pp == nil {
		return nil
	}
	select {
	case <-pp.Done():
		return nil
	default:
		return pp
	}
}

func (b *Broker) setProxyProcessHandle(pp *proxyProcess) {
	b.workersMu.Lock()
	old := b.proxyProc
	if b.proxyProc != pp {
		b.fenceRoutingHistoryLocked(b.proxyProc)
		b.historyOrigin, b.historyRun = "", ""
		if pp != nil {
			// Exhaustion disables history for this process; never wrap identity.
			if b.historyIncarnation != ^uint64(0) {
				b.historyIncarnation++
			}
		}
	}
	b.proxyProc = pp
	b.workersMu.Unlock()
	if old != pp {
		old.retireRoutingHistoryRead()
	}
	b.wakeRoutingHistory()
}

// clearProxyHandles withdraws the process handle and every engine's handle.
func (b *Broker) clearProxyHandles() {
	b.setProxyProcessHandle(nil)
	for _, profile := range b.frontedEngineProfiles() {
		b.setEngineProxyHandle(profile, nil)
	}
}

// getProxySupervisor reads proxySup, which a policy change can now replace
// while a respawn reads it.
func (b *Broker) getProxySupervisor() *supervisor {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	return b.proxySup
}

func (b *Broker) setProxySupervisor(sup *supervisor) {
	b.workersMu.Lock()
	b.proxySup = sup
	b.workersMu.Unlock()
}

// proxySupPhase says whether the proxy supervisor may be started or stopped by
// a policy change.
type proxySupPhase int

const (
	// proxySupStarting is before Serve has decided whether the proxy runs, so a
	// policy change arriving from a paired node cannot start it ahead of the
	// facades' preparation.
	proxySupStarting proxySupPhase = iota
	proxySupLive
	// proxySupClosed is after shutdown, so a late policy change cannot start a
	// proxy that nothing would stop.
	proxySupClosed
)

// routingListenerEnabled reports whether the persisted policy enables the
// routing listener, which only the proxy process can serve.
func (b *Broker) routingListenerEnabled() bool {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.current.Routing.Listener.Enabled
}

// listenerNeedsProxy is routingListenerEnabled for a node that has a proxy
// binary to run it.
func (b *Broker) listenerNeedsProxy() bool {
	return b.proxyPath != "" && b.routingListenerEnabled()
}

// startProxySupervision is Serve's one decision to run the proxy: for the
// facades the operator asked for, or for the routing listener the policy
// enables. It opens the supervisor to later policy changes.
func (b *Broker) startProxySupervision(facadesConfigured bool) {
	b.proxySupMu.Lock()
	defer b.proxySupMu.Unlock()
	if facadesConfigured || b.listenerNeedsProxy() {
		b.startProxySupervisorLocked()
	}
	b.proxySupPhase = proxySupLive
	b.resumeProxySupervision()
}

// startProxySupervisorLocked creates and starts the one proxy supervisor.
// Caller holds proxySupMu.
func (b *Broker) startProxySupervisorLocked() {
	sup := newSupervisor(engines.ProxyComponent, defaultRestartPolicy(), b.spawnProxy)
	b.configureProxySupervisorCallbacks(sup)
	b.proxySupExhausted.Store(false)
	onExhausted := sup.onExhausted
	sup.onExhausted = func(attempt int) {
		onExhausted(attempt)
		if b.beforeProxyExhaustionPublished != nil {
			b.beforeProxyExhaustionPublished()
		}
		b.proxySupExhausted.Store(true)
		b.resumeProxySupervision()
	}
	// Published before Start, because the first spawn's bring-up context reads it.
	b.setProxySupervisor(sup)
	if err := sup.Start(); err != nil {
		slog.Warn("proxy failed to start; continuing without local engine proxies",
			"path", b.proxyPath, "err", err)
		b.setProxySupervisor(nil)
		// No facade will come up, so both engines take the terminal path:
		// report the blocked claim where one was made, then finish, which
		// is what returns the OLLAMA_HOST alias reservation.
		if b.ollamaState().managedFacade.Load() {
			b.blockManagedOllamaFacade("the proxy could not be started")
		}
		b.finishOllamaProxyTerminal()
		if b.lmstudioState().managedFacade.Load() {
			_, _ = b.blockManagedLMStudioFacade("the proxy could not be started", nil)
		}
		b.finishLMStudioProxyTerminal()
	}
}

// stopProxySupervisorLocked stops the proxy and withdraws its handles. Caller
// holds proxySupMu.
func (b *Broker) stopProxySupervisorLocked() {
	// Fence before waiting for the monitor/pipe. Done cannot join a callback
	// already admitted at the initial source check.
	b.fenceRoutingHistory()
	b.withdrawEgressLease()
	// The process is stopped on purpose, so a push its closing pipe fails is
	// not a failed activation: retire it first and such an answer is discarded.
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.incarnation = nil
	r.egress.anchored = false
	r.mu.Unlock()
	if sup := b.getProxySupervisor(); sup != nil {
		sup.Stop()
		b.setProxySupervisor(nil)
	}
	b.clearProxyHandles()
	b.captureDepartedRunWorkloads("proxy stopped")
}

// credentialRefs is the sorted set of provider credential references the
// policy configures, one per routing account.
func credentialRefs(p nodepolicy.Policy) []nodepolicy.CredentialRef {
	refs := make([]nodepolicy.CredentialRef, 0, len(p.Routing.Accounts))
	for _, a := range p.Routing.Accounts {
		refs = append(refs, a.Credential)
	}
	slices.Sort(refs)
	return slices.Compact(refs)
}

// credentialRefsDiffer reports whether two policies configure different
// reference sets.
func credentialRefsDiffer(a, b nodepolicy.Policy) bool {
	return !slices.Equal(credentialRefs(a), credentialRefs(b))
}

// proxyCredentialRefs snapshots the persisted policy's references for a proxy
// launch.
func (b *Broker) proxyCredentialRefs() []nodepolicy.CredentialRef {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	return credentialRefs(r.current)
}

// proxyCredentialsStale reports whether pp was launched with another reference
// set than the persisted policy now configures.
func (b *Broker) proxyCredentialsStale(pp *proxyProcess) bool {
	return !slices.Equal(pp.credentialRefs, b.proxyCredentialRefs())
}

// restartProxyForCredentialsLocked replaces the running proxy, whose
// environment holds a stale reference set, through the same stop and start a
// listener change uses, so it spends none of the crash restart budget. Caller
// holds proxySupMu.
func (b *Broker) restartProxyForCredentialsLocked() {
	slog.Info("restarting the proxy for a changed provider credential reference set")
	b.stopProxySupervisorLocked()
	b.startProxySupervisorLocked()
}

// shutdownProxySupervision stops the proxy for good. Idempotent.
func (b *Broker) shutdownProxySupervision() {
	b.proxySupClosing.Store(true)
	if b.beforeProxyShutdownLocked != nil {
		b.beforeProxyShutdownLocked()
	}
	b.proxySupMu.Lock()
	b.proxySupPhase = proxySupClosed
	b.stopProxySupervisorLocked()
	b.proxySupMu.Unlock()
	// Cap retirement's drain allowance so later shutdown work keeps its budget.
	// Accepted terminals stay in history even if unread output blocks fanout.
	b.waitProxyRetirements()
}

// reconcileProxySupervision asks for the proxy to be brought in line with the
// persisted policy, and returns at once. Starting a proxy replays the policy to
// it and stopping one joins its exit, neither of which a policy:set may wait
// for beyond its own activation bound, so the work runs on one background
// goroutine and policy:set reports the revision pending until the new process
// acknowledges the replay. Requests made while it runs are folded into one more
// pass, which reads the policy afresh, so the last change wins.
func (b *Broker) reconcileProxySupervision() {
	b.proxySupWorkMu.Lock()
	defer b.proxySupWorkMu.Unlock()
	b.proxySupDirty = true
	b.resumeProxySupervisionLocked()
}

// resumeProxySupervision wakes only a pending listener change. Lifecycle
// publication alone must not renew an exhausted supervisor's restart budget.
func (b *Broker) resumeProxySupervision() {
	b.proxySupWorkMu.Lock()
	defer b.proxySupWorkMu.Unlock()
	b.resumeProxySupervisionLocked()
}

func (b *Broker) resumeProxySupervisionLocked() {
	b.resumeProxyRetirementsLocked()
	if !b.proxySupDirty || b.proxySupWorking {
		return
	}
	b.proxySupWorking = true
	go func() {
		for b.reconcileProxySupervisionNow() {
		}
	}()
}

// reconcileProxySupervisionNow is one pass of reconcileProxySupervision,
// through the same supervisor Serve starts rather than a second path.
//
// An unfinished spawn or a supervisor still recovering defers the request until
// publication wakes the worker. A proxy that stays, because a facade is serving
// or the listener is enabled, keeps its lifecycle unless it was launched with a
// credential reference set the policy no longer configures, in which case it
// is replaced once. Otherwise an enabled listener starts a missing proxy, and a
// disabled listener stops a process whose bring-up yielded no facades.
func (b *Broker) reconcileProxySupervisionNow() bool {
	b.proxySupMu.Lock()
	defer b.proxySupMu.Unlock()
	// Decide whether the request can be consumed under the same lock as a
	// publication wakeup. A blocked pass leaves it dirty and retires the worker
	// before publication can wake it, so neither ordering loses the request.
	b.proxySupWorkMu.Lock()
	sup, exhausted := b.getProxySupervisor(), b.proxySupExhausted.Load()
	pp := b.proxyProcessHandle()
	blocked := b.proxySupPhase == proxySupStarting ||
		(sup != nil && !exhausted && (pp == nil || !pp.bringUpDone.Load()))
	if !b.proxySupDirty || (b.proxySupPhase != proxySupClosed && blocked) {
		b.proxySupWorking = false
		b.proxySupWorkMu.Unlock()
		return false
	}
	b.proxySupDirty = false
	b.proxySupWorkMu.Unlock()
	if b.proxySupPhase != proxySupLive {
		return true
	}
	running := sup != nil && !exhausted
	switch listener := b.listenerNeedsProxy(); {
	case running && (listener || !pp.listenerOnly.Load()):
		if b.proxyCredentialsStale(pp) {
			b.restartProxyForCredentialsLocked()
		}
	case listener:
		if sup != nil {
			b.stopProxySupervisorLocked()
		}
		b.startProxySupervisorLocked()
	case !listener && sup != nil:
		b.stopProxySupervisorLocked()
	}
	return true
}
