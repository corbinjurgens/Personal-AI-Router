// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"nvpair-shared/applog"
	"nvpair-shared/engines"
	"nvpair-shared/noderec"
)

// spawnScanner / spawnNodeInfo / spawnProxy / spawnWorkloadManager are the
// per-worker spawn closures the supervisor calls for the initial start and
// every restart. Each starts a fresh process (which wires its own reader
// goroutines), stores the concrete handle on the Broker under workersMu,
// and returns it as a supervisedHandle for the supervisor to watch. A
// spawn failure returns the error without storing a handle, leaving the
// previous (dead) handle in place until the next successful (re)start.
func (b *Broker) spawnScanner() (supervisedHandle, error) {
	if err := b.scannerWaits.validate(); err != nil {
		return nil, err
	}
	args, err := b.scannerArgs()
	if err != nil {
		return nil, err
	}
	args = append(args, b.scannerWaits.args()...)
	sp, err := startScannerBound(
		b.scannerPath,
		applog.LevelString(),
		b.store,
		b.relayDir,
		b.failWorkloadsForNode,
		nil,
		nil,
		b.dispatchScannerErrorsNotif,
		func(source *scannerProcess) {
			b.bindScannerSource(source)
		},
		args...,
	)
	if err != nil {
		return nil, err
	}
	b.setScanner(sp)
	// Replay cached registrations to the (re)started daemon so it re-advertises
	// this node's services without any worker reconnect logic (epoch replay).
	for _, p := range b.regCache.Snapshot() {
		go sp.pushRegister(p)
	}
	slog.Info("scanner started", "path", b.scannerPath, "pid", sp.cmd.Process.Pid)
	return sp, nil
}

func (b *Broker) bindScannerSource(source *scannerProcess) {
	b.workersMu.Lock()
	b.scannerSource = source
	if b.telemetry != nil {
		b.telemetry.retireScannerMemory()
	}
	b.workersMu.Unlock()
	source.current = func() bool { b.workersMu.Lock(); defer b.workersMu.Unlock(); return b.scannerSource == source }
	source.onMemoryRetired = func(hostUUID string) {
		b.workersMu.Lock()
		defer b.workersMu.Unlock()
		if b.scannerSource == source && b.telemetry != nil {
			b.telemetry.retireScannerHostMemory(hostUUID)
		}
	}
	source.onTelemetry = func(value noderec.NodeTelemetry) {
		b.workersMu.Lock()
		if b.scannerSource != source || b.telemetry == nil {
			b.workersMu.Unlock()
			return
		}
		projected, ok := b.telemetry.Upsert(sourceScanner, value, time.Now())
		b.workersMu.Unlock()
		if ok {
			b.fanTelemetryToScheduler(projected)
		}
	}
	source.onTelemetryRemoved = func(hostUUID string) {
		b.workersMu.Lock()
		if b.scannerSource != source || b.telemetry == nil {
			b.workersMu.Unlock()
			return
		}
		projected, ok := b.telemetry.Remove(hostUUID, sourceScanner, time.Now())
		b.workersMu.Unlock()
		if ok {
			b.fanTelemetryToScheduler(projected)
		}
	}
	source.onExit = func() {
		b.workersMu.Lock()
		current := b.scannerSource == source
		if current {
			b.scannerSource = nil
		}
		if current && b.telemetry != nil {
			b.telemetry.retireScannerMemory()
		}
		b.workersMu.Unlock()
	}
}

func (b *Broker) spawnNodeInfo() (supervisedHandle, error) {
	// Deliberately NOT passed b.clusterDirArgs(): node-info stays plain HTTP even
	// when clustered, and is the ONE documented exception to "the inter-node
	// cluster data plane is always mTLS" (errors and workload-manager admit only
	// pinned members, in either direction). Its GPU/CPU/memory inventory is the
	// lowest-sensitivity inter-node surface and is read directly by consumers that
	// hold no cluster identity — the desktop app's two-second telemetry poll and
	// baseline LAN discovery — so gating it would remove node telemetry from the UI.
	// The accepted risk and its compensating controls are recorded under
	// "Documented Risk Acceptances" in desktop/SECURITY.md.
	//
	// node-info still implements the cluster-gated mode (pass it --cluster-dir and
	// it admits only pinned peers). It is kept, unenabled, on purpose: it is the
	// swap point if the desktop telemetry read is ever solved another way. Enabling
	// it here without solving that first WILL blank the UI's node cards.
	//
	// Do pass --node-id: node-info reports the broker's already-resolved UUID so
	// /v1/node-info agrees with the identity the fleet keys on, even on a custom
	// --cluster-dir data root (node-info gets no cluster-dir, so it would
	// otherwise resolve the default root and could report a different UUID).
	port := b.servicePorts().nodeInfo
	extra := append([]string{"--node-id", b.nodeID}, servicePortArgs(port, nodeInfoHTTPPort)...)
	np, err := startNodeInfo(b.nodeInfoPath, applog.LevelString(), b.forwardNodeInfoNotification, extra...)
	if err != nil {
		return nil, err
	}
	b.setNodeInfo(np)
	// Tell it our membership: node-info reports the cluster principal on
	// /v1/node-info but holds no cluster dir to read it from, so this push is the
	// only source. It runs on every spawn, which also covers a supervised restart.
	b.pushClusterIdentityToNodeInfo()
	// Register node-info's service so the daemon advertises ni= on _nvpair-node.
	// node-info binds the planned port it was given (force_ports is inert), so
	// the broker knows its port. Idempotent across restarts.
	b.registerService(noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: port})
	slog.Info("node-info started", "path", b.nodeInfoPath, "pid", np.cmd.Process.Pid)
	return np, nil
}

// proxyArgs builds the child's argv. Only process-scoped flags remain: which
// engines it fronts and their ports travel
// in facade/enable, because the broker plans a different port per engine and a
// single-valued flag cannot carry two plans.
func (b *Broker) proxyArgs() []string {
	// Thread the cluster dir so the proxy can bring up its pin-gated LAN mTLS
	// ingress (and dial peers over mTLS) once this node is clustered; empty/
	// absent certs leave it loopback-plaintext only.
	return b.clusterDirArgs()
}

// setServeCtx records the broker's own lifetime so work started on a spawn
// goroutine can be abandoned when the broker is shutting down.
func (b *Broker) setServeCtx(ctx context.Context) {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	b.serveCtx = ctx
}

func (b *Broker) getServeCtx() context.Context {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	return b.serveCtx
}

// proxyBringUpContext bounds facade bring-up and abandons it the moment either
// the broker or the proxy supervisor starts shutting down.
//
// This exists because spawn stopped being instantaneous. It used to fork a
// process and return; it now issues facade/enable RPCs — up to two per engine,
// each bounded by proxyCallTimeout — and supervisor.Stop blocks on the spawn
// returning with no timeout of its own. A crash-looping proxy whose fresh child
// never answers could therefore spend the whole teardownBudget inside the
// proxy's join, and graceFor would clip every worker behind it, including
// engine-manager's stop grace. stopwait.go records that cutting that grace
// short is what orphaned engines.
//
// Both signals are needed, not either one. Stopping() covers a respawn, but the
// *first* spawn runs synchronously inside Start on the Serve goroutine — Serve
// cannot reach shutdownInferenceStack to call Stop until that spawn returns, so
// Stopping() can never close during it. The Serve context is what interrupts
// that case.
//
// The caller must cancel, as with any context: shutdown is one way out, a
// completed bring-up is the other.
func (b *Broker) proxyBringUpContext() (context.Context, context.CancelFunc) {
	parent := b.getServeCtx()
	if parent == nil {
		parent = context.Background()
	}
	ctx, cancel := context.WithCancel(parent)

	sup := b.getProxySupervisor()
	if sup == nil {
		return ctx, cancel
	}
	go func() {
		select {
		case <-sup.Stopping():
			cancel()
		case <-ctx.Done():
		}
	}()
	return ctx, cancel
}

// enableEngineFacade brings up one engine's facade with that engine's port plan
// and fallback policy.
func (b *Broker) enableEngineFacade(
	ctx context.Context, pp *proxyProcess, profile engineProxyProfile,
) error {
	return b.enableEngineFacadeWithPortCheck(ctx, pp, profile, b.portAvailable)
}

func (b *Broker) enableEngineFacadeWithPortCheck(
	ctx context.Context,
	pp *proxyProcess,
	profile engineProxyProfile,
	available func(int) bool,
) error {
	if _, dynamic := b.dynamicEngineProfile(profile.Name); dynamic {
		spec, err := b.dynamicEngineFacadeSpec(profile)
		if err != nil {
			return err
		}
		return b.enableProxyFacadeWithFallback(ctx, pp, spec, func(failed int) int {
			return b.dynamicEngineFallbackPort(profile, failed, available)
		})
	}
	return b.enableProxyFacadeWithFallback(
		ctx,
		pp,
		b.defaultEngineFacadeSpec(profile),
		func(failed int) int {
			return b.defaultEngineFallbackPortWithCheck(profile, failed, available)
		},
	)
}

// facadeCameUpAnyway reports whether a facade that failed to answer its enable
// nevertheless announced itself ready.
//
// Only for **transport-level ambiguity** — a timeout or a lost response, where
// the child does the bind, two notification writes and the serve inside that
// call and may genuinely be serving. Disowning it then would strand a listener
// on a port the broker reports as unavailable, with no facade/disable to
// withdraw it, so the facade's own ready notification gets the deciding vote.
//
// Deliberately not applied to an explicit rejection. A rejection means the
// child decided against the facade and withdrew it, and a recorded ready from
// earlier in the same attempt is then stale: honouring it would publish a dead
// port and handle, and the broker would never retry because it believes the
// facade is up. errFacadeBindFailed is the clearest case — the child deleted
// the facade before answering — but the rule holds for any answered error.
func facadeCameUpAnyway(pp *proxyProcess, engine string, enableErr error) bool {
	if !isTransportAmbiguous(enableErr) {
		return false
	}
	ready, port := pp.Status(engine)
	if !ready || port <= 0 {
		return false
	}
	slog.Warn("facade enable did not answer, but the facade reported ready; keeping it",
		"engine", engine, "port", port, "err", enableErr)
	return true
}

// isTransportAmbiguous reports whether an enable failure leaves the child's
// actual state unknown, as opposed to the child having answered.
//
// errRPCAnswered wraps every case where a response arrived, so the absence of
// it is what identifies a timeout, a closed pipe, or a cancelled call.
func isTransportAmbiguous(err error) bool {
	return err != nil && !stderrors.Is(err, errRPCAnswered)
}

// spawnProxy starts the one nvpair-proxy process and brings up a facade in it
// for every enabled engine.
//
// One process for all of them is what makes the scheduler's picture whole: the
// estimated-work reservations a proxy takes between scheduler snapshots live in
// the process, so two processes each kept half the story and could send
// simultaneous bursts to the same node believing it idle.
//
// A facade that fails to come up is logged and skipped, not fatal. The spawn
// only fails if no engine came up at all and the policy does not enable the
// routing listener, because a process with no listener serves nobody and should
// be retried by the supervisor rather than left running.
func (b *Broker) spawnProxy() (supervisedHandle, error) {
	if b.proxySupClosing.Load() {
		return nil, fmt.Errorf("proxy supervision is shutting down")
	}
	if ctx := b.getServeCtx(); ctx != nil && ctx.Err() != nil {
		return nil, ctx.Err()
	}
	pp, err := startProxyWithHistory(b.routingHistory, engines.ProxyComponent, b.proxyPath, applog.LevelString(), b.proxyCredentialRefs(), b.relayDir,
		func(source *proxyProcess, method string, params json.RawMessage) {
			b.forwardProxyNotificationFrom(source, method, params)
		}, b.proxyArgs()...)
	if err != nil {
		return nil, err
	}
	// Published for every engine before any facade is enabled, so a facade's
	// own ready notification finds the handle it belongs to. The process handle
	// is published even for a node with no facade: the policy and the routing
	// listener reach the proxy through it.
	b.setProxyProcessHandle(pp)
	for _, profile := range b.frontedEngineProfiles() {
		if b.proxyEnabled(profile) {
			b.setEngineProxyHandle(profile, pp)
		}
	}
	slog.Info("proxy started", "path", b.proxyPath, "pid", pp.cmd.Process.Pid)

	// The child is serving its JSON-RPC channel but has no listeners yet, so
	// ask for each facade before reporting the worker up. Synchronous, on the
	// spawn goroutine, so a caller that treats spawn success as "the engines
	// are being brought up" stays correct.
	bringUp, cancelBringUp := b.proxyBringUpContext()
	defer cancelBringUp()

	// Node policy first, before any facade listens, so a paused node is never
	// briefly admitting work after a proxy restart.
	if !b.replayPolicyToProxy(bringUp, pp) {
		pp.Stop()
		b.clearProxyHandles()
		return nil, fmt.Errorf("engine switch invalidations could not be replayed before facade enable")
	}
	// A new proxy run has lost its in-memory resident holds. Refresh inventory
	// after activation rather than reuse observations from the retired run.
	b.workersMu.Lock()
	modelWorker, modelSource := b.engineMgr, b.engineMgrSource
	b.workersMu.Unlock()
	go b.refreshEnginePolicySnapshotFrom(modelWorker, modelSource)

	enabled := 0
	var failed []engineProxyProfile
	for _, profile := range b.frontedEngineProfiles() {
		if !b.proxyEnabled(profile) {
			continue
		}
		err := b.enableEngineFacade(bringUp, pp, profile)
		switch {
		case err == nil:
			enabled++
		case facadeCameUpAnyway(pp, profile.Name, err):
			enabled++
		default:
			slog.Warn("facade enable failed; continuing without a local proxy for it",
				"engine", profile.Name, "err", err)
			failed = append(failed, profile)
		}
	}
	// A process with no facade is kept only for the routing listener, which
	// belongs to the process and is served from the policy replayed above.
	if enabled == 0 && !b.routingListenerEnabled() {
		// A policy change can withdraw the listener after replay but before
		// bring-up finishes. Retire this deliberately stopped incarnation
		// before closing its pipe, just as stopProxySupervisorLocked does.
		r := b.nodePolicyRuntime()
		r.mu.Lock()
		if r.incarnation == pp {
			r.incarnation = nil
		}
		r.mu.Unlock()
		// The supervisor retries with the existing facade port plans.
		pp.Stop()
		b.clearProxyHandles()
		return nil, fmt.Errorf("no engine facade could be brought up in %s", b.proxyPath)
	}

	// This process incarnation is staying and will not be retried, because a
	// facade is serving or the listener is. Only now is a failed engine terminal
	// for it.
	for _, profile := range failed {
		b.setEngineProxyHandle(profile, nil)
	}
	if b.beforeProxySpawnPublished != nil {
		b.beforeProxySpawnPublished()
	}
	pp.listenerOnly.Store(enabled == 0)
	pp.bringUpDone.Store(true)
	b.resumeProxySupervision()
	return pp, nil
}

func (b *Broker) spawnWorkloadManager() (supervisedHandle, error) {
	port := b.servicePorts().workload
	extra := append(b.clusterDirArgs(), servicePortArgs(port, workloadHTTPPort)...)
	wm, err := startWorkloadManager(b.workloadMgrPath, applog.LevelString(), b.relayDir, b.forwardWorkloadManagerNotification, extra...)
	if err != nil {
		return nil, err
	}
	b.setWorkloadMgr(wm)
	b.registerService(noderec.RegisterParams{Service: noderec.ServiceWorkload, Port: port})
	// A (re)started manager's anti-entropy set starts empty. Replay this node's
	// active local-origin workloads so a supervised restart mid-job doesn't
	// leave a later-joining peer unable to learn those jobs (and the heartbeat
	// non-empty). No-op on the initial spawn (the store is empty).
	b.rehydrateWorkloadManager(wm)
	slog.Info("workload-manager started", "path", b.workloadMgrPath, "pid", wm.cmd.Process.Pid)
	return wm, nil
}

func (b *Broker) spawnErrors() (supervisedHandle, error) {
	// Pass the resolved node UUID so nvpair-errors attributes this node's errors
	// to the same identity the broker stamps (and the cluster keys on), not its
	// hostname — keeping local-origin attribution and clear-by-id in lockstep.
	port := b.servicePorts().errors
	extra := append(b.clusterDirArgs(), "--node-id", b.nodeID)
	extra = append(extra, servicePortArgs(port, errorsHTTPPort)...)
	ep, err := startErrors(b.errorsPath, applog.LevelString(), b.relayDir, b.onErrorsUpdate, extra...)
	if err != nil {
		return nil, err
	}
	b.setErrors(ep)
	b.replayStickyErrors()
	b.registerService(noderec.RegisterParams{Service: noderec.ServiceErrors, Port: port})
	slog.Info("nvpair-errors started", "path", b.errorsPath, "pid", ep.cmd.Process.Pid)
	return ep, nil
}

func (b *Broker) spawnEngineMgr() (supervisedHandle, error) {
	ports := b.servicePorts()
	if err := validateServicePorts(ports); err != nil {
		return nil, fmt.Errorf("service ports: %w", err)
	}
	if err := validateEngineManagerControlHost(b.engineMgrControlHost); err != nil {
		return nil, err
	}
	httpPort, controlPort := ports.engineManagerHTTP, ports.engineControl
	// Tell engine-manager to serve its LAN HTTP surface on the configured em port so
	// peers can fetch this node's model list at /v1/models, and its cluster-scoped
	// mTLS remote-control surface (ec) on the configured control port. --cluster-dir
	// gates the ec surface: it is bound whenever a cluster dir is configured, but
	// admits a caller only while this node is a live cluster member (its TLS
	// identity is resolved per handshake from the live cluster dir), so a join or
	// leave needs no restart here.
	args := append(b.logLevelArgs(),
		"--http-port", fmt.Sprintf("%d", httpPort),
		"--control-port", fmt.Sprintf("%d", controlPort),
		"--control-host", b.engineMgrControlHost)
	args = append(args, b.clusterDirArgs()...)
	// Serialize source replacement and pending settings invalidation with readers.
	// Keep the established engineConfigMu -> policy runtime mutex order.
	b.engineConfigMu.Lock()
	source := b.activateEngineManagerSource()
	b.invalidatePendingModelSettingsRunningLocked()
	b.engineConfigMu.Unlock()
	w, err := startRPCWorker("engine-manager", b.engineMgrPath, args, func(method string, params json.RawMessage) {
		b.forwardEngineNotificationFrom(source, method, params)
	}, func(exited *rpcWorker) {
		// Retire the source first so a waiting confirmation already sees it
		// gone when the binding wake below lets it run.
		b.retireEngineManagerSource(source)
		b.clearEngineMgrBinding(source)
		b.engineManagerExited(exited)
	})
	if err != nil {
		b.retireEngineManagerSource(source)
		b.clearEngineMgrBinding(source)
		return nil, err
	}
	b.workersMu.Lock()
	b.engineMgrRecovering = true
	b.workersMu.Unlock()
	b.setEngineMgr(w)
	b.bindEngineMgr(w, source)
	if !b.recoverReplacementEngineMgr(w) {
		// Keep supervising the live replacement, but expose no forward surface
		// while its pause shutdown proof cannot be verified after bounded retries.
		return w, nil
	}
	// The recovery loop owns all retries; this continuation runs once, only
	// after successful recovery of this still-current replacement.
	go b.refreshEnginePolicySnapshotFrom(w, source)
	// Initial restore waits for facade startup. A later engine-manager respawn
	// restores here after that shared startup boundary has completed.
	if b.facadeStartupReady() {
		b.restoreEnabledEngines(w)
	}
	// Register engine-manager's HTTP surface so the daemon advertises em= on this
	// node's _nvpair-node record; peers enrich models from it. The same configured
	// port is replayed across scanner restarts by regCache.
	b.registerService(noderec.RegisterParams{Service: noderec.ServiceEngineManager, Port: httpPort})
	// Advertise the ec remote-control surface too, whenever a cluster dir is
	// configured — which is exactly when engine-manager binds it. The port is
	// bound for the life of the process and admits callers by live membership
	// (an unclustered node presents no leaf, so every handshake is refused), so
	// this advertises a port that is genuinely listening in both states. That is
	// also why no re-registration is needed on a membership change: nothing about
	// the advertised address depends on whether this node is currently a member.
	if b.clusterDir != "" {
		b.registerService(noderec.RegisterParams{Service: noderec.ServiceEngineControl, Port: controlPort})
	}
	slog.Info("engine-manager started", "path", b.engineMgrPath, "pid", w.cmd.Process.Pid, "httpPort", httpPort, "controlPort", controlPort)
	return w, nil
}

func (b *Broker) spawnManualNodes() (supervisedHandle, error) {
	// Bind the callback's source before the read pump can deliver its first
	// notification. The eventual process handle is published separately.
	source := &rpcWorker{name: "manual-nodes"}
	callback, retire := b.bindManualNodesSource(source)
	w, err := startRPCWorker("manual-nodes", b.manualNodesPath, b.manualNodesArgs(), callback, func(*rpcWorker) { retire() })
	if err != nil {
		retire()
		return nil, err
	}
	b.manualMu.Lock()
	if b.manualNodesSource == source {
		b.setManualNodes(w)
	}
	b.manualMu.Unlock()
	slog.Info("manual-nodes started", "path", b.manualNodesPath, "pid", w.cmd.Process.Pid)
	return w, nil
}

func (b *Broker) bindManualNodesSource(source *rpcWorker) (func(string, json.RawMessage), func()) {
	b.manualMu.Lock()
	b.manualNodesSource = source
	b.manualMu.Unlock()
	callback := func(method string, params json.RawMessage) {
		b.manualMu.Lock()
		defer b.manualMu.Unlock()
		if b.manualNodesSource != source {
			return
		}
		b.forwardManualNodesNotificationLocked(method, params)
	}
	retire := func() {
		b.manualMu.Lock()
		defer b.manualMu.Unlock()
		if b.manualNodesSource == source {
			b.clearManualNodesStateLocked()
		}
	}
	return callback, retire
}

func (b *Broker) spawnSettings() (supervisedHandle, error) {
	w, err := startRPCWorker("settings", b.settingsPath, b.logLevelArgs(), b.forwardSettingsNotification)
	if err != nil {
		return nil, err
	}
	b.setSettings(w)
	slog.Info("settings started", "path", b.settingsPath, "pid", w.cmd.Process.Pid)
	return w, nil
}

func (b *Broker) spawnJobScheduler() (supervisedHandle, error) {
	w, err := startRPCWorker("scheduler", b.schedulerPath, b.logLevelArgs(), b.forwardSchedulerNotification)
	if err != nil {
		return nil, err
	}

	replayed, telemetryReplayed := b.initializeScheduler(w)

	if replayed > 0 {
		slog.Info("replayed active workloads to scheduler", "count", replayed)
	}
	if telemetryReplayed > 0 {
		slog.Info("replayed telemetry to scheduler", "count", telemetryReplayed)
	}
	slog.Info("scheduler started", "path", b.schedulerPath, "pid", w.cmd.Process.Pid)
	return w, nil
}

func (b *Broker) initializeScheduler(w *rpcWorker) (int, int) {
	// workloadEmitMu prevents an accepted workload transition from interleaving
	// with its baseline. schedulerFeedMu also holds discovery fanout until the
	// three ordered baselines are queued to the new child.
	b.workloadEmitMu.Lock()
	b.schedulerFeedMu.Lock()
	b.setScheduler(w)
	replayed := b.replayActiveWorkloadsToScheduler(w)
	telemetryReplayed := b.replayTelemetryToScheduler(w)
	if err := w.Notify("discovery:nodes-changed", discoveryNotificationNodes(b.store.Snapshot())); err != nil {
		slog.Warn("push discovery baseline to scheduler failed", "err", err)
	}
	b.schedulerFeedMu.Unlock()
	b.workloadEmitMu.Unlock()

	return replayed, telemetryReplayed
}

func (b *Broker) spawnClusterManager() (supervisedHandle, error) {
	// The manager owns <base>/cluster; the broker's clusterDir IS that subtree, so
	// its parent is the base — the same derivation resolveLocalNodeID performs.
	// Passing it keeps the only writer of the cluster dir and the workers reading it
	// on one directory.
	port := b.servicePorts().clusterManager
	var callbackMu sync.Mutex
	var cm *clusterManagerProcess
	callbackMu.Lock()
	started, err := startClusterManager(b.clusterMgrPath, applog.LevelString(), b.clusterManagerConfigDir(), b.relayDir, servicePortArgs(port, clusterManagerHTTPPort), func(method string, params json.RawMessage) {
		if method == peerServicesEvent {
			callbackMu.Lock()
			child := cm
			callbackMu.Unlock()
			b.relayPeerServices(child, params)
			return
		}
		b.forwardClusterManagerNotification(method, params)
	})
	if err != nil {
		callbackMu.Unlock()
		return nil, err
	}
	cm = started
	b.setClusterMgr(cm)
	callbackMu.Unlock()
	b.registerService(noderec.RegisterParams{Service: noderec.ServiceCluster, Port: port})
	// Reflect the persisted clusterId (owned by nvpair-node-settings) back into the
	// cluster-manager, which doesn't persist it itself. Synchronous so the first
	// spawn restores before app:ready/readLoop (no client-visible race); also
	// covers a crash respawn.
	b.restoreClusterIdentity(cm)
	// Converge the scanner's advertised uuid= on the cluster principal. The
	// scanner spawns first and mints its own node-id before cluster-manager
	// writes identity.json, so on a fresh host its uuid= can start out diverged.
	go b.convergeScannerIdentity()
	slog.Info("cluster-manager started", "path", b.clusterMgrPath, "pid", cm.cmd.Process.Pid)
	return cm, nil
}

// startOptionalWorker creates a supervisor for an optional (non-fatal)
// worker, wires its crash/recovery callbacks (report into the pipeline,
// drop the handle while down), and starts it. It returns the supervisor,
// or nil when the binary wasn't resolved or the first spawn failed — in
// both cases the broker simply runs without that worker. The caller defers
// Stop on a non-nil result.
func (b *Broker) startOptionalWorker(name, path string, spawn func() (supervisedHandle, error), clearHandle func()) *supervisor {
	if path == "" {
		slog.Info("worker path not resolved; running without it", "worker", name)
		return nil
	}
	sup := newSupervisor(name, defaultRestartPolicy(), spawn)
	sup.onCrash, sup.onRecovered = b.supervisedWorkerCallbacks(name, clearHandle)
	if err := sup.Start(); err != nil {
		slog.Warn("worker failed to start; continuing without it", "worker", name, "path", path, "err", err)
		return nil
	}
	return sup
}

func (b *Broker) shutdownInferenceStack() {
	// Arm the shared budget here, at the first thing shutdown does, rather than
	// letting the first join arm it: everything below draws on the same window the
	// parent is waiting through, including the proxy joins and the engine StopAll
	// wait that both run before any deferred join.
	beginTeardown()

	// Stop ingress first so no new inference can arrive while engine-manager is
	// draining engines. supervisor.Stop uses the proxy's stdin-close/join path;
	// it never adds a parent-side kill timeout. One process holds every
	// engine's facade, so this closes all of them at once — the child drains
	// each facade before releasing the shared transport pool.
	b.shutdownProxySupervision()
	b.prepareEngineManagerShutdown()
}

// prepareEngineManagerShutdown requests a synchronous engine shutdown from
// engine-manager and waits for it to finish. engine-manager bounds its own
// StopAll internally (see nvpair-engine-manager/lifecycle.go) and the peer.Call
// returns promptly with an error if it exits instead of responding, but neither
// of those is a bound the broker controls, and this step runs before any worker
// has been joined. Without a deadline of its own it could spend the parent's
// entire shutdown window here and leave nothing for the joins that follow — so it
// draws on the same teardown budget they do. A missing, failed, or slow
// engine-manager falls through to the stdin-close backstop.
func (b *Broker) prepareEngineManagerShutdown() {
	w := b.getEngineMgr()
	if w == nil {
		return
	}
	deadline := engineShutdownDeadline()
	slog.Info("requesting engine shutdown before teardown", "deadlineMs", deadline.Milliseconds())

	ctx, cancel := context.WithTimeout(context.Background(), deadline)
	defer cancel()
	if _, rpcErr, err := w.CallNoTimeout(ctx, prepareEngineShutdownMethod, nil); err != nil || rpcErr != nil {
		slog.Warn("engine prepare-shutdown did not complete cleanly; relying on stdin-close backstop", "err", err, "rpcErr", rpcErr)
	}
}
