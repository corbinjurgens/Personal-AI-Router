// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"nvpair-shared/appdir"
	"nvpair-shared/applog"
	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/noderec"
)

func main() {
	ipcPath := flag.String("ipc", "", "IPC endpoint: Unix domain socket path or Windows named pipe (default: stdin/stdout)")
	scannerPath := flag.String("scanner-path", "", "path to nvpair-node-scanner binary (default: ./nvpair-node-scanner in the current working directory)")
	scannerMDNS := flag.Bool("scanner-mdns", true, "enable scanner multicast discovery and advertisement; false passes --mdns=false before scanner startup")
	scannerMDNSService := flag.String("scanner-mdns-service", noderec.ServiceType, "scanner mDNS DNS-SD service (_name._tcp)")
	scannerMDNSDomain := flag.String("scanner-mdns-domain", noderec.Domain, "scanner mDNS domain, including its advertised hostname")
	manualOllamaPort := flag.Int("manual-ollama-port", 11434, "manual-node Ollama probe port (0 disables the probe)")
	manualLMStudioPort := flag.Int("manual-lmstudio-port", 1234, "manual-node LM Studio probe port (0 disables the probe)")
	manualNodeInfoPort := flag.Int("manual-node-info-port", nodeInfoHTTPPort, "manual-node plain node-info probe port (0 disables it; explicit entry TLS ports take precedence)")
	nodeInfoPath := flag.String("node-info-path", "", "path to nvpair-node-info binary (default: ./nvpair-node-info in the current working directory)")
	proxyPath := flag.String("proxy-path", "", "path to nvpair-proxy binary (default: ./nvpair-proxy in the current working directory)")
	proxyEngines := flag.String("proxy-engines", strings.Join(engines.Names(), ","), "comma-separated engines to front with a proxy; one nvpair-proxy process hosts a facade for each entry")
	workloadMgrPath := flag.String("workload-manager-path", "", "path to nvpair-workload-manager binary (default: ./nvpair-workload-manager in the current working directory)")
	errorsPath := flag.String("errors-path", "", "path to nvpair-errors binary (default: ./nvpair-errors in the current working directory)")
	engineMgrPath := flag.String("engine-manager-path", "", "path to nvpair-engine-manager binary (default: ./nvpair-engine-manager in the current working directory)")
	engineMgrHTTPPort := flag.Int("engine-manager-http-port", engineManagerHTTPPort, "engine-manager model HTTP service port (1..65535; must differ from the control port)")
	engineMgrControlPort := flag.Int("engine-manager-control-port", engineControlPort, "engine-manager remote-control service port (1..65535; must differ from the HTTP port)")
	engineMgrControlHost := flag.String("engine-manager-control-host", defaultEngineManagerControlHost, "literal IP address passed to engine-manager --control-host for the ec listener only")
	nodeInfoPort := flag.Int("node-info-port", nodeInfoHTTPPort, "node-info HTTP service port (ni) passed to nvpair-node-info and registered with discovery (1..65535; unique among the local service ports)")
	errorsPort := flag.Int("errors-port", errorsHTTPPort, "errors peer-sync service port (er) passed to nvpair-errors and registered with discovery (1..65535; unique among the local service ports)")
	workloadMgrPort := flag.Int("workload-manager-port", workloadHTTPPort, "workload-manager service port (wl) passed to nvpair-workload-manager and registered with discovery (1..65535; unique among the local service ports)")
	clusterMgrPort := flag.Int("cluster-manager-port", clusterManagerHTTPPort, "cluster-manager service port (cl) passed to nvpair-cluster-manager and registered with discovery (1..65535; unique among the local service ports)")
	manualNodesPath := flag.String("manual-nodes-path", "", "path to nvpair-manual-nodes binary (default: ./nvpair-manual-nodes in the current working directory)")
	settingsPath := flag.String("settings-path", "", "path to nvpair-node-settings binary (default: ./nvpair-node-settings in the current working directory)")
	clusterMgrPath := flag.String("cluster-manager-path", "", "path to nvpair-cluster-manager binary (default: ./nvpair-cluster-manager in the current working directory)")
	schedulerPath := flag.String("scheduler-path", "", "path to nvpair-job-scheduler binary (default: ./nvpair-job-scheduler in the current working directory)")
	clusterDirFlag := flag.String("cluster-dir", "", "cluster config dir (node.crt/node.key + trusted/) the broker passes to its mDNS workers (nvpair-errors, nvpair-workload-manager, nvpair-node-info, nvpair-node-scanner, nvpair-manual-nodes) to enable cluster-scoped inter-node mTLS; defaults to the per-user Nvidia Corporation/Personal AI Router cluster/ dir, where nvpair-cluster-manager mints them")
	showVersion := flag.Bool("version", false, "print version and exit")
	resolveLevel := applog.RegisterFlag(nil, slog.LevelInfo)
	flag.Parse()
	scannerNamespace, err := (noderec.MDNSNamespace{Service: *scannerMDNSService, Domain: *scannerMDNSDomain}).Normalize()
	if err != nil {
		log.Fatal(err)
	}
	manualPorts := &manualProbePorts{ollama: *manualOllamaPort, lmstudio: *manualLMStudioPort, nodeInfo: *manualNodeInfoPort}
	if err := manualPorts.validate(); err != nil {
		log.Fatal(err)
	}
	if err := validateEngineManagerControlHost(*engineMgrControlHost); err != nil {
		log.Fatal(err)
	}

	if *showVersion {
		fmt.Println(Version)
		os.Exit(0)
	}

	// Route this process's logs, and every worker's, through one non-blocking
	// sink so the parent's read rate can never hold a write open. Installed
	// before applog.Init because Init captures the writer.
	sink := newStderrSink(os.Stderr, defaultStderrSpillPath())
	stderrOut = sink
	applog.SetOutput(sink)
	defer sink.Close()

	applog.Init("nvpair-ui-broker", resolveLevel())

	// Fatal paths must not return through the async queue. applog bridges the
	// stdlib log package into slog, which now writes into the sink, and os.Exit
	// skips the deferred Close — so a stdlib fatal would enqueue its message, race
	// the drain goroutine, and normally lose, leaving the process to exit 1 with no
	// explanation at all. Flush before exiting instead. Close is once-guarded, so
	// the defer above stays correct as a no-op.
	fatalf := func(format string, a ...any) {
		slog.Error(fmt.Sprintf(format, a...))
		sink.Close()
		os.Exit(1)
	}
	if err := validateServicePorts(servicePortPlan{
		nodeInfo:          *nodeInfoPort,
		errors:            *errorsPort,
		workload:          *workloadMgrPort,
		clusterManager:    *clusterMgrPort,
		engineManagerHTTP: *engineMgrHTTPPort,
		engineControl:     *engineMgrControlPort,
	}); err != nil {
		fatalf("service ports: %v", err)
	}

	// The inter-node workers do cluster-scoped mTLS off the cert + pins
	// nvpair-cluster-manager mints under the per-user data dir's cluster/ subtree.
	// Default the broker's view to that same path so mTLS turns on automatically
	// once this node is clustered, with nothing for the spawning UI to pass; an
	// explicit --cluster-dir overrides it.
	//
	// This directory is load-bearing in a way it did not used to be: it is the only
	// channel by which a cluster-scoped worker learns its membership, and the
	// inter-node data plane is mTLS only. A node that resolves no cluster dir
	// therefore exchanges no cluster traffic at all — it does not quietly fall back
	// to plain HTTP. Resolve it with the same chain nvpair-cluster-manager uses for
	// its own base (appdir, then next to the executable) so the writer and the
	// readers cannot pick different directories, and log the result once: this line
	// is the single diagnostic that tells a healthy-looking but isolated node apart
	// from a working one.
	clusterDir := *clusterDirFlag
	if clusterDir == "" {
		clusterDir = defaultClusterDir()
	}
	if clusterDir == "" {
		slog.Warn("no cluster directory could be resolved; this node cannot join or serve a cluster " +
			"(no per-user data dir and no executable path)")
	} else {
		slog.Info("resolved cluster directory", "clusterDir", clusterDir)
	}

	resolvedScanner, err := resolveScannerPath(*scannerPath)
	if err != nil {
		fatalf("scanner binary: %v", err)
	}

	// node-info is auxiliary: unlike the scanner, a missing binary is not
	// fatal. If the user passed an explicit --node-info-path we still
	// surface the failure loudly (it's a clear operator mistake) by
	// exiting; otherwise an absent default sibling just means the broker
	// runs without spawning a local node advertiser.
	resolvedNodeInfo, err := resolveNodeInfoPath(*nodeInfoPath)
	if err != nil {
		if *nodeInfoPath != "" {
			fatalf("node-info binary: %v", err)
		}
		slog.Warn("node-info binary not found; broker will run without local node advertisement", "err", err)
		resolvedNodeInfo = ""
	}

	// nvpair-proxy is auxiliary too, resolved with the same rules: an
	// explicit --proxy-path that doesn't exist is a loud operator mistake
	// (fatal), but an absent default sibling just means the broker runs
	// without local reverse proxies.
	resolvedProxy, err := resolveProxyPath(*proxyPath)
	if err != nil {
		if *proxyPath != "" {
			fatalf("proxy binary: %v", err)
		}
		slog.Warn("proxy binary not found; broker will run without local engine proxies", "err", err)
		resolvedProxy = ""
	}

	// One process, a facade per selected engine. An unknown name is an operator
	// mistake worth failing on rather than silently fronting fewer engines
	// than asked for.
	selectedEngines, err := parseProxyEngines(*proxyEngines)
	if err != nil {
		fatalf("proxy engines: %v", err)
	}

	// nvpair-workload-manager is auxiliary too, resolved with the same rules:
	// an explicit --workload-manager-path that doesn't exist is a loud
	// operator mistake (fatal), but an absent default sibling just means
	// the broker runs without cluster workload relay.
	resolvedWorkloadMgr, err := resolveWorkloadManagerPath(*workloadMgrPath)
	if err != nil {
		if *workloadMgrPath != "" {
			fatalf("workload-manager binary: %v", err)
		}
		slog.Warn("workload-manager binary not found; broker will run without cluster workload relay", "err", err)
		resolvedWorkloadMgr = ""
	}

	// nvpair-errors is auxiliary too, resolved with the same rules: an
	// explicit --errors-path that doesn't exist is a loud operator mistake
	// (fatal), but an absent default sibling just means the broker runs
	// with the service-error pipeline disabled.
	resolvedErrors, err := resolveErrorsPath(*errorsPath)
	if err != nil {
		if *errorsPath != "" {
			fatalf("errors binary: %v", err)
		}
		slog.Warn("errors binary not found; broker will run with the error pipeline disabled", "err", err)
		resolvedErrors = ""
	}

	// nvpair-engine-manager, nvpair-manual-nodes, and nvpair-node-settings are
	// auxiliary too, resolved with the same rules: an explicit path that
	// doesn't exist is a loud operator mistake (fatal), but an absent
	// default sibling just means the broker runs without that worker.
	resolvedEngineMgr, err := resolveEngineManagerPath(*engineMgrPath)
	if err != nil {
		if *engineMgrPath != "" {
			fatalf("engine-manager binary: %v", err)
		}
		slog.Warn("engine-manager binary not found; broker will run without engine management", "err", err)
		resolvedEngineMgr = ""
	}

	resolvedManualNodes, err := resolveManualNodesPath(*manualNodesPath)
	if err != nil {
		if *manualNodesPath != "" {
			fatalf("manual-nodes binary: %v", err)
		}
		slog.Warn("manual-nodes binary not found; broker will run without manual nodes", "err", err)
		resolvedManualNodes = ""
	}

	resolvedSettings, err := resolveSettingsPath(*settingsPath)
	if err != nil {
		if *settingsPath != "" {
			fatalf("settings binary: %v", err)
		}
		slog.Warn("settings binary not found; broker will run without node settings", "err", err)
		resolvedSettings = ""
	}

	// nvpair-cluster-manager is auxiliary too, resolved with the same rules: an
	// explicit --cluster-manager-path that doesn't exist is a loud operator
	// mistake (fatal), but an absent default sibling just means the broker
	// runs without cluster pairing / membership (cluster:* and nodes:*).
	resolvedClusterMgr, err := resolveClusterManagerPath(*clusterMgrPath)
	if err != nil {
		if *clusterMgrPath != "" {
			fatalf("cluster-manager binary: %v", err)
		}
		slog.Warn("cluster-manager binary not found; broker will run without cluster pairing", "err", err)
		resolvedClusterMgr = ""
	}

	// nvpair-job-scheduler is auxiliary too, resolved with the same rules: an
	// explicit --scheduler-path that doesn't exist is a loud operator mistake
	// (fatal), but an absent default sibling just means the broker runs without
	// scheduler-driven proxy priority (proxies keep their own default ordering).
	resolvedScheduler, err := resolveSchedulerPath(*schedulerPath)
	if err != nil {
		if *schedulerPath != "" {
			fatalf("scheduler binary: %v", err)
		}
		slog.Warn("scheduler binary not found; broker will run without job scheduling", "err", err)
		resolvedScheduler = ""
	}

	var transport io.ReadWriteCloser
	if *ipcPath != "" {
		conn, err := dialIPC(*ipcPath)
		if err != nil {
			fatalf("failed to connect to IPC endpoint %q: %v", *ipcPath, err)
		}
		transport = conn
		log.Printf("using IPC transport: %s", *ipcPath)
	} else {
		transport, err = newStdioTransport()
		if err != nil {
			fatalf("stdio transport: %v", err)
		}
		log.Print("using stdio transport")
	}
	defer transport.Close()

	lifetime := newBrokerMainLifetime()
	ctx := lifetime.root
	defer lifetime.cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		select {
		case sig := <-sigCh:
			slog.Info("received signal, shutting down", "signal", sig.String())
			lifetime.cancel()
		case <-ctx.Done():
		}
	}()

	paths := workerPaths{
		scanner:              resolvedScanner,
		nodeInfo:             resolvedNodeInfo,
		proxy:                resolvedProxy,
		proxyEngines:         selectedEngines,
		workloadMgr:          resolvedWorkloadMgr,
		errors:               resolvedErrors,
		engineMgr:            resolvedEngineMgr,
		engineMgrHTTPPort:    *engineMgrHTTPPort,
		engineMgrControlPort: *engineMgrControlPort,
		engineMgrControlHost: engineMgrControlHost,
		nodeInfoPort:         *nodeInfoPort,
		errorsPort:           *errorsPort,
		workloadMgrPort:      *workloadMgrPort,
		clusterMgrPort:       *clusterMgrPort,
		manualNodes:          resolvedManualNodes,
		disableMDNS:          !*scannerMDNS,
		scannerNamespace:     &scannerNamespace,
		manualProbePorts:     manualPorts,
		settings:             resolvedSettings,
		clusterMgr:           resolvedClusterMgr,
		scheduler:            resolvedScheduler,
		clusterDir:           clusterDir,
	}
	b, err := newMainBroker(lifetime, paths, transport, *ipcPath != "")
	if err != nil {
		fatalf("broker construction: %v", err)
	}
	if err := b.serveMainLifetime(lifetime); err != nil && ctx.Err() == nil {
		fatalf("broker error: %v", err)
	}
	slog.Info("shutdown complete")
}

// Main constructs authority before either guarded reader can receive traffic.
// Explicit NewBroker callers may still construct an ordinary session.
func newMainBroker(lifetime *brokerMainLifetime, paths workerPaths, transport io.ReadWriteCloser, ipc bool) (*Broker, error) {
	b := NewBroker(nil, paths)
	var err error
	b.routingHistory, err = newBrokerRoutingHistory(b, lifetime)
	if err == nil {
		b.codec, err = newRoutingHistoryLocalCodec(b, transport, ipc)
	}
	if err != nil {
		if lifetime != nil {
			lifetime.cancel()
		}
		if stdio, ok := transport.(*stdioTransport); ok {
			if input, ok := stdio.Reader.(io.Closer); ok {
				_ = input.Close()
			}
			if output, ok := stdio.Writer.(io.Closer); ok {
				_ = output.Close()
			}
		}
		_ = transport.Close()
		return nil, err
	}
	return b, nil
}

// IPC owns a real connection; stdioTransport's
// Close is a no-op, so its captured stdin reader owns read interruption.
// Tests with other transports use the explicit owner hook at acquire instead.
func newRoutingHistoryLocalCodec(b *Broker, transport io.ReadWriteCloser, ipc bool) (*Codec, error) {
	if b.routingHistory == nil {
		return nil, jsonrpc.ErrHistoryInvalid
	}
	var input io.Closer = transport
	var output io.Closer = transport
	if !ipc {
		stdio, ok := transport.(*stdioTransport)
		if !ok {
			return nil, jsonrpc.ErrHistoryInvalid
		}
		input, ok = stdio.Reader.(io.Closer)
		if !ok {
			return nil, jsonrpc.ErrHistoryInvalid
		}
		output, ok = stdio.Writer.(io.Closer)
		if !ok {
			return nil, jsonrpc.ErrHistoryInvalid
		}
	}
	interrupt := func() { _ = output.Close() }
	codec, err := b.routingHistory.readers.acquire(historyLocalControl, transport, func() { _ = input.Close() }, interrupt)
	if err == nil {
		b.routingHistory.interruptLocalWrite = interrupt
	}
	return codec, err
}

// defaultClusterDir resolves the cluster subtree this node's workers read and
// nvpair-cluster-manager writes, mirroring that manager's own base-dir chain
// (defaultBaseDir): the per-user data dir first, then next to the executable for
// dev builds. Mirroring it is the point — the two processes must never resolve
// different directories, because the manager would then pair into a tree no worker
// reads, leaving a node that looks clustered but exchanges nothing.
//
// Returns "" only when neither is available, which the caller reports; unlike the
// manager there is deliberately no "." tier, since a cwd-relative cluster identity
// would move with however the app was launched.
func defaultClusterDir() string {
	if dir, err := appdir.Path("cluster"); err == nil {
		return dir
	}
	if exe, err := os.Executable(); err == nil {
		return filepath.Join(filepath.Dir(exe), "cluster")
	}
	return ""
}

// resolveScannerPath honours the explicit --scanner-path override if
// provided; otherwise it expects ./nvpair-node-scanner (with the platform
// executable extension) sitting in the broker's current working
// directory. See resolveSiblingBinary for the shared resolution rules.
func resolveScannerPath(override string) (string, error) {
	return resolveSiblingBinary(override, "nvpair-node-scanner", "--scanner-path")
}

// resolveNodeInfoPath mirrors resolveScannerPath for the nvpair-node-info
// binary the broker supervises. Its result is optional at the call site:
// a not-found default sibling degrades to "no local node advertisement"
// rather than aborting the broker.
func resolveNodeInfoPath(override string) (string, error) {
	return resolveSiblingBinary(override, "nvpair-node-info", "--node-info-path")
}

// resolveProxyPath mirrors resolveNodeInfoPath for the nvpair-proxy binary the
// broker supervises. One process fronts every engine: it is spawned once and
// then asked for a facade per enabled engine over facade/enable. Like node-info
// its result is optional at the call site: a not-found default sibling degrades
// to "no local engine proxies" rather than aborting the broker.
func resolveProxyPath(override string) (string, error) {
	return resolveSiblingBinary(override, "nvpair-proxy", "--proxy-path")
}

// parseProxyEngines narrows the --proxy-engines list against the shared engine
// table. An unknown name fails rather than being skipped: the operator asked
// for an engine that does not exist, and quietly fronting the others would
// look like the flag worked.
func parseProxyEngines(csv string) ([]string, error) {
	seen := map[string]bool{}
	var out []string
	for _, raw := range strings.Split(csv, ",") {
		name := strings.TrimSpace(raw)
		if name == "" {
			continue
		}
		if _, ok := engines.ByName(name); !ok {
			return nil, fmt.Errorf("unknown engine %q; known engines are %s", name, strings.Join(engines.Names(), ", "))
		}
		if !seen[name] {
			seen[name] = true
			out = append(out, name)
		}
	}
	return out, nil
}

// resolveWorkloadManagerPath mirrors resolveProxyPath for the
// nvpair-workload-manager binary the broker supervises. Like node-info and the
// proxy its result is optional at the call site: a not-found default sibling
// degrades to "no cluster workload relay" rather than aborting the broker.
func resolveWorkloadManagerPath(override string) (string, error) {
	return resolveSiblingBinary(override, "nvpair-workload-manager", "--workload-manager-path")
}

// resolveErrorsPath mirrors resolveProxyPath for the nvpair-errors binary the
// broker supervises. Like the other auxiliaries its result is optional at
// the call site: a not-found default sibling degrades to "no service-error
// pipeline" rather than aborting the broker.
func resolveErrorsPath(override string) (string, error) {
	return resolveSiblingBinary(override, "nvpair-errors", "--errors-path")
}

// resolveEngineManagerPath / resolveManualNodesPath / resolveSettingsPath /
// resolveClusterManagerPath mirror resolveErrorsPath for the remaining
// adopted workers. Each result is optional at the call site: a not-found
// default sibling degrades to "without that worker" rather than aborting
// the broker.
func resolveEngineManagerPath(override string) (string, error) {
	return resolveSiblingBinary(override, "nvpair-engine-manager", "--engine-manager-path")
}

func resolveManualNodesPath(override string) (string, error) {
	return resolveSiblingBinary(override, "nvpair-manual-nodes", "--manual-nodes-path")
}

func resolveSettingsPath(override string) (string, error) {
	return resolveSiblingBinary(override, "nvpair-node-settings", "--settings-path")
}

func resolveClusterManagerPath(override string) (string, error) {
	return resolveSiblingBinary(override, "nvpair-cluster-manager", "--cluster-manager-path")
}

func resolveSchedulerPath(override string) (string, error) {
	return resolveSiblingBinary(override, "nvpair-job-scheduler", "--scheduler-path")
}

// resolveSiblingBinary locates a worker binary the broker spawns. It
// honours an explicit override if provided; otherwise it expects the
// binary (with the platform executable extension) sitting in the broker's
// current working directory. We deliberately do NOT fall back to PATH
// lookup or sibling directories — the broker should run alongside the
// other NVPAIR binaries in a known layout, and a confusing PATH match for a
// stale dev build is worse than a clear "not found" error. flagName is
// the override flag surfaced in the not-found error so the message points
// at the right knob.
func resolveSiblingBinary(override, name, flagName string) (string, error) {
	if override != "" {
		if _, err := os.Stat(override); err != nil {
			return "", fmt.Errorf("%s binary %q not accessible: %w", name, override, err)
		}
		return override, nil
	}
	bin := name
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("get cwd: %w", err)
	}
	candidate := filepath.Join(cwd, bin)
	if _, err := os.Stat(candidate); err != nil {
		return "", fmt.Errorf("%s not found in %s (use %s to override): %w", bin, cwd, flagName, err)
	}
	return candidate, nil
}
