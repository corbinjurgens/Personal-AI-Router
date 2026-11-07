// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"net"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"nvpair-shared/engines"
)

// TestClusterManagerConfigDirTracksBrokerClusterDir pins the invariant that keeps
// the cluster dir single-sourced: the broker hands its workers <base>/cluster and
// must hand nvpair-cluster-manager the same <base>. The manager is the only writer
// of that tree and the workers' only source of membership, so if the two resolved
// different bases the node would pair into a directory nothing reads — a healthy
// roster with no cluster traffic, and no restart left to mask it.
func TestClusterManagerConfigDirTracksBrokerClusterDir(t *testing.T) {
	base := filepath.Join(t.TempDir(), "Personal AI Router")
	b := &Broker{clusterDir: filepath.Join(base, "cluster")}
	if got := b.clusterManagerConfigDir(); got != base {
		t.Fatalf("clusterManagerConfigDir = %q, want %q", got, base)
	}

	// With no cluster dir there is nothing to pass, and the manager falls back to
	// its own default — the one case where the two can still diverge, which the
	// broker reports at startup.
	if got := (&Broker{}).clusterManagerConfigDir(); got != "" {
		t.Fatalf("clusterManagerConfigDir with no cluster dir = %q, want empty", got)
	}
}

func TestEngineAvailabilityWaitsForFacadeStartup(t *testing.T) {
	engine, codec := newTestRPCWorkerPipe(t)

	b := &Broker{
		clusterDir:   filepath.Join(t.TempDir(), "cluster"),
		facadesReady: make(chan struct{}),
	}
	b.setEngineMgr(engine)
	restore := recordEnabledEngineRestoration(t, codec)
	advertised := make(chan string, 3)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan bool, 1)
	go func() {
		done <- b.runEngineAvailability(
			ctx,
			func(context.Context) { advertised <- "first" },
			func(context.Context) { advertised <- "second" },
			func(context.Context) { advertised <- "llamacpp" },
		)
	}()

	select {
	case got := <-restore:
		t.Fatalf("restore %q ran before facade startup", got)
	case got := <-advertised:
		t.Fatalf("%s advertising ran before facade startup", got)
	case <-time.After(100 * time.Millisecond):
	}
	close(b.facadesReady)

	for _, want := range []string{engineNodeProjectionMethod, "engine:settings-projection", restoreEnabledEnginesMethod} {
		select {
		case got := <-restore:
			if got != want {
				t.Fatalf("restoration method = %q, want %q", got, want)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("%s did not run after facade startup", want)
		}
	}
	seen := map[string]bool{}
	for len(seen) < 3 {
		select {
		case got := <-advertised:
			seen[got] = true
		case <-time.After(2 * time.Second):
			t.Fatalf("advertising did not start for every engine: %v", seen)
		}
	}
	select {
	case completed := <-done:
		if !completed {
			t.Fatal("availability orchestration reported cancellation")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("availability orchestration did not return")
	}
}

func TestEngineAvailabilityCanceledBeforeFacadeStartup(t *testing.T) {
	b := &Broker{facadesReady: make(chan struct{})}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if b.runEngineAvailability(ctx, func(context.Context) { t.Error("advertising ran after cancellation") }) {
		t.Fatal("availability reported success after cancellation")
	}
	if b.facadeStartupReady() {
		t.Fatal("unfinished facade startup reported ready")
	}
}

type orderedLifecycleHandle struct {
	name  string
	order chan<- string
	done  chan struct{}
	once  sync.Once
}

func newOrderedLifecycleHandle(name string, order chan<- string) *orderedLifecycleHandle {
	return &orderedLifecycleHandle{name: name, order: order, done: make(chan struct{})}
}

func (h *orderedLifecycleHandle) Done() <-chan struct{} { return h.done }
func (h *orderedLifecycleHandle) Stop() {
	h.once.Do(func() {
		h.order <- h.name
		close(h.done)
	})
}

// Ingress stops before engines drain, so no new inference can arrive while
// engine-manager is stopping engines. One proxy process holds every engine's
// facade now, so there is one proxy stop rather than two — but it must still
// precede engine-manager.
func TestInferenceShutdownStopsTheProxyBeforeEngines(t *testing.T) {
	order := make(chan string, 2)
	proxy := newOrderedLifecycleHandle(engines.ProxyComponent, order)
	proxySup := newSupervisor(engines.ProxyComponent, noRestartPolicy(),
		func() (supervisedHandle, error) { return proxy, nil })
	if err := proxySup.Start(); err != nil {
		t.Fatal(err)
	}

	engineClient, engineServer := net.Pipe()
	defer engineClient.Close()
	defer engineServer.Close()
	engine := &rpcWorker{peer: NewPeer(NewCodec(engineClient))}
	go engine.peer.Serve(nil, nil)
	go func() {
		codec := NewCodec(engineServer)
		msg, err := codec.Read()
		if err != nil {
			return
		}
		order <- "engine-manager"
		_ = codec.Respond(msg.ID, nil)
	}()

	b := &Broker{proxySup: proxySup}
	b.setEngineMgr(engine)
	b.shutdownInferenceStack()

	got := []string{<-order, <-order}
	if got[0] != engines.ProxyComponent || got[1] != "engine-manager" {
		t.Fatalf("shutdown order = %v, want [%s engine-manager]", got, engines.ProxyComponent)
	}
}
