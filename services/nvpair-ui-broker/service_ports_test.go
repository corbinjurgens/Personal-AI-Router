// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/applog"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/noderec"
)

// servicePortWorker is one port-bearing worker the broker spawns, with the
// accessors a test needs to observe its current child.
type servicePortWorker struct {
	name    string
	service noderec.ServiceKey
	spawn   func() (supervisedHandle, error)
	command func() *exec.Cmd
}

func servicePortWorkers(b *Broker) []servicePortWorker {
	return []servicePortWorker{
		{name: "node-info", service: noderec.ServiceNodeInfo, spawn: b.spawnNodeInfo, command: func() *exec.Cmd {
			if p := b.getNodeInfo(); p != nil {
				return p.cmd
			}
			return nil
		}},
		{name: "errors", service: noderec.ServiceErrors, spawn: b.spawnErrors, command: func() *exec.Cmd {
			if p := b.getErrors(); p != nil {
				return p.cmd
			}
			return nil
		}},
		{name: "workload-manager", service: noderec.ServiceWorkload, spawn: b.spawnWorkloadManager, command: func() *exec.Cmd {
			if p := b.getWorkloadMgr(); p != nil {
				return p.cmd
			}
			return nil
		}},
		{name: "cluster-manager", service: noderec.ServiceCluster, spawn: b.spawnClusterManager, command: func() *exec.Cmd {
			if p := b.getClusterMgr(); p != nil {
				return p.cmd
			}
			return nil
		}},
	}
}

// buildServicePortFixture builds the JSON-RPC scanner fixture. It also stands in
// for every port-bearing worker: it answers requests, ignores notifications and
// exits on stdin EOF, which is all these spawn paths need from a child.
func buildServicePortFixture(t *testing.T) string {
	t.Helper()
	portScannerOnce.Do(func() {
		if err := os.WriteFile(portScannerSource, []byte(engineManagerPortScannerFixture), 0600); err != nil {
			portScannerErr = err
			return
		}
		portScannerErr = buildPortTestBinary(portScannerBinary, ".", portScannerSource)
	})
	if portScannerErr != nil {
		t.Fatalf("build scanner fixture: %v", portScannerErr)
	}
	return portScannerBinary
}

var (
	portScannerOnce   sync.Once
	portScannerBinary string
	portScannerSource string
	portScannerErr    error
)

func cachedRegistrations(b *Broker) map[noderec.ServiceKey]int {
	got := map[noderec.ServiceKey]int{}
	for _, registration := range b.regCache.Snapshot() {
		got[registration.Service] = registration.Port
	}
	return got
}

func TestServicePortsDefaultPlanKeepsWorkerArgvAndRegistrations(t *testing.T) {
	isolateOllamaHostTestConfig(t)
	fixture := buildServicePortFixture(t)
	base := t.TempDir()
	clusterDir := filepath.Join(base, "cluster")
	b := NewBroker(nil, workerPaths{
		nodeInfo: fixture, errors: fixture, workloadMgr: fixture, clusterMgr: fixture,
		clusterDir: clusterDir,
	})
	wantPlan := servicePortPlan{
		nodeInfo: 14318, errors: 14319, workload: 14320, clusterManager: 14321,
		engineManagerHTTP: 14322, engineControl: 14323,
	}
	if got := b.servicePorts(); got != wantPlan {
		t.Fatalf("default plan = %+v, want %+v", got, wantPlan)
	}
	if err := validateServicePorts(wantPlan); err != nil {
		t.Fatalf("default plan rejected: %v", err)
	}

	level := applog.LevelString()
	// The argv each worker received before ports became configurable.
	wantArgs := map[string][]string{
		"node-info":        {fixture, "--log-level", level, "--node-id", b.nodeID},
		"errors":           {fixture, "--peer-sync", "--log-level", level, "--cluster-dir", clusterDir, "--node-id", b.nodeID},
		"workload-manager": {fixture, "--log-level", level, "--cluster-dir", clusterDir},
		"cluster-manager":  {fixture, "--log-level", level, "--config-dir", base},
	}
	for _, worker := range servicePortWorkers(b) {
		handle, err := worker.spawn()
		if err != nil {
			t.Fatalf("spawn %s: %v", worker.name, err)
		}
		t.Cleanup(handle.Stop)
		if got := worker.command().Args; !reflect.DeepEqual(got, wantArgs[worker.name]) {
			t.Errorf("%s argv = %q, want %q", worker.name, got, wantArgs[worker.name])
		}
	}
	want := map[noderec.ServiceKey]int{"ni": 14318, "er": 14319, "wl": 14320, "cl": 14321}
	if got := cachedRegistrations(b); !reflect.DeepEqual(got, want) {
		t.Fatalf("registrations = %v, want %v", got, want)
	}
}

func TestServicePortsAlternatePlanFlowsThroughSpawnRestartAndReplay(t *testing.T) {
	isolateOllamaHostTestConfig(t)
	fixture := buildServicePortFixture(t)
	b := NewBroker(nil, workerPaths{
		scanner: fixture, nodeInfo: fixture, errors: fixture, workloadMgr: fixture, clusterMgr: fixture,
		nodeInfoPort: 25318, errorsPort: 25319, workloadMgrPort: 25320, clusterMgrPort: 25321,
		clusterDir: filepath.Join(t.TempDir(), "cluster"),
	})
	if err := validateServicePorts(b.servicePorts()); err != nil {
		t.Fatalf("alternate plan rejected: %v", err)
	}
	want := map[noderec.ServiceKey]int{"ni": 25318, "er": 25319, "wl": 25320, "cl": 25321}
	fast := restartPolicy{baseDelay: 10 * time.Millisecond, maxDelay: 10 * time.Millisecond, maxAttempts: 3}

	assertPortArg := func(t *testing.T, worker servicePortWorker, command *exec.Cmd) {
		t.Helper()
		port := strconv.Itoa(want[worker.service])
		args := command.Args
		for i := range args[:len(args)-1] {
			if args[i] == "--port" && args[i+1] == port {
				return
			}
		}
		t.Fatalf("%s argv = %q, want --port %s", worker.name, args, port)
	}

	for _, worker := range servicePortWorkers(b) {
		t.Run(worker.name, func(t *testing.T) {
			sup := newSupervisor(worker.name, fast, worker.spawn)
			if err := sup.Start(); err != nil {
				t.Fatalf("start: %v", err)
			}
			t.Cleanup(sup.Stop)
			first := worker.command()
			assertPortArg(t, worker, first)
			if got := cachedRegistrations(b)[worker.service]; got != want[worker.service] {
				t.Fatalf("registered %s port = %d, want %d", worker.service, got, want[worker.service])
			}

			// An unexpected exit is respawned by the supervisor with the same plan.
			if err := first.Process.Kill(); err != nil {
				t.Fatalf("kill child: %v", err)
			}
			deadline := time.Now().Add(10 * time.Second)
			for {
				if current := worker.command(); current != nil && current != first {
					assertPortArg(t, worker, current)
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("supervisor did not respawn the worker")
				}
				time.Sleep(10 * time.Millisecond)
			}
			if got := cachedRegistrations(b)[worker.service]; got != want[worker.service] {
				t.Fatalf("registered %s port after restart = %d, want %d", worker.service, got, want[worker.service])
			}
		})
	}

	// A fresh scanner receives the configured ports from the registration cache.
	handle, err := b.spawnScanner()
	if err != nil {
		t.Fatalf("spawn scanner: %v", err)
	}
	defer handle.Stop()
	defer b.setScanner(nil)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for {
		result, rpcErr, err := b.getScanner().peer.Call(ctx, "test:get-registrations", nil)
		if err != nil || rpcErr != nil {
			t.Fatalf("read scanner registrations: %v, %v", err, rpcErr)
		}
		var got map[noderec.ServiceKey]int
		if err := json.Unmarshal(result, &got); err != nil {
			t.Fatalf("decode scanner registrations: %v", err)
		}
		if reflect.DeepEqual(got, want) {
			break
		}
		if ctx.Err() != nil {
			t.Fatalf("scanner replay = %v, want %v", got, want)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestServicePortsInvalidPlansRejectedBeforeAnyChild(t *testing.T) {
	isolateOllamaHostTestConfig(t)
	fixture := buildServicePortFixture(t)
	for _, tc := range []struct {
		name  string
		paths workerPaths
		want  string
	}{
		{name: "node-info below range", paths: workerPaths{nodeInfoPort: -1}, want: "--node-info-port must be between 1 and 65535 (got -1)"},
		{name: "cluster-manager above range", paths: workerPaths{clusterMgrPort: 65536}, want: "--cluster-manager-port must be between 1 and 65535 (got 65536)"},
		{name: "duplicate service ports", paths: workerPaths{nodeInfoPort: 25318, errorsPort: 25318}, want: "--node-info-port and --errors-port must differ (both are 25318)"},
		{name: "override collides with a default", paths: workerPaths{workloadMgrPort: 14321}, want: "--workload-manager-port and --cluster-manager-port must differ (both are 14321)"},
		{name: "service collides with engine-manager", paths: workerPaths{errorsPort: 25122, engineMgrHTTPPort: 25122}, want: "--errors-port and --engine-manager-http-port must differ (both are 25122)"},
		{name: "compatibility facade reserved", paths: workerPaths{errorsPort: 1234}, want: "--errors-port 1234 is reserved: the LM Studio compatibility proxy uses port 1234"},
		{name: "managed backend base reserved", paths: workerPaths{clusterMgrPort: 11435}, want: "--cluster-manager-port 11435 is reserved: the managed Ollama backend uses port 11435"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			paths := tc.paths
			paths.scanner, paths.nodeInfo, paths.errors, paths.workloadMgr, paths.clusterMgr = fixture, fixture, fixture, fixture, fixture
			b := NewBroker(nil, paths)
			err := b.Serve(context.Background())
			if err == nil || !strings.Contains(err.Error(), "service ports: "+tc.want) {
				t.Fatalf("Serve error = %v, want %q", err, tc.want)
			}
			if b.getScanner() != nil || b.getNodeInfo() != nil || b.getErrors() != nil ||
				b.getWorkloadMgr() != nil || b.getClusterMgr() != nil || len(b.regCache.Snapshot()) != 0 {
				t.Fatal("invalid plan spawned or registered a worker")
			}
		})
	}
}

func TestServicePortsReservationsFollowTheConfiguredPlan(t *testing.T) {
	b := NewBroker(nil, workerPaths{nodeInfoPort: 25318, errorsPort: 25319, workloadMgrPort: 25320, clusterMgrPort: 25321})
	for _, tc := range []struct {
		port int
		want string
	}{
		{port: 25318, want: "the node-info service uses port 25318"},
		{port: 25319, want: "the errors service uses port 25319"},
		{port: 25320, want: "the workload service uses port 25320"},
		{port: 25321, want: "the cluster manager uses port 25321"},
		{port: 14318},
		{port: 14319},
		{port: 14320},
		{port: 14321},
	} {
		t.Run(strconv.Itoa(tc.port), func(t *testing.T) {
			if got := b.reservedOllamaHostAliasPort(tc.port, nil, nil); got != tc.want {
				t.Fatalf("alias reservation = %q, want %q", got, tc.want)
			}
		})
	}

	for _, port := range []int{25318, 25319, 25320, 25321} {
		err := b.validateSettingsPortsLocked(context.Background(), "ollama", settings.Config{ServerPort: port, ProxyPort: 25400}, settings.Snapshot{})
		if err == nil || err.Error() != "a selected port is reserved by a PAIR service or proxy alias" {
			t.Fatalf("settings validation for %d = %v, want PAIR reservation", port, err)
		}
	}

	worker, codec := newTestRPCWorkerPipe(t)
	b.setEngineMgr(worker)
	go func() {
		request, err := codec.Read()
		if err != nil {
			t.Errorf("read configured ports request: %v", err)
			return
		}
		if err := codec.Respond(request.ID, map[string]any{"engines": []any{}}); err != nil {
			t.Errorf("respond with configured ports: %v", err)
		}
	}()
	current := settings.Snapshot{Running: true, EffectiveServerPort: 14318, EffectiveProxyPort: 14319}
	if err := b.validateSettingsPortsLocked(context.Background(), "ollama", settings.Config{ServerPort: 14318, ProxyPort: 14319}, current); err != nil {
		t.Fatalf("unused default service ports remain reserved: %v", err)
	}
}
