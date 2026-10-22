// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/testbin"
)

// Persist the endpoint, then discard all process state as on manager restart.
// The source-stop case has no local-Off flag, so it reaches the pending-stop
// recovery branch rather than the earlier remote local-Off cleanup loop.
func switchRecoveryCleanupFixture(t *testing.T, route string, unhealthy bool) (*Executor, settings.SwitchPrepareRequest, *engineState, string, int) {
	t.Helper()
	m, dir, port := restoreOrphanManifest(t)
	ex, p := switchFixture(t)
	engine := "target"
	if route != "target local Off" {
		engine = "source"
	}
	m.Engine = engine
	platform := m.Platforms[hostKey()]
	platform.Runtime.Port = 0
	if unhealthy {
		platform.Runtime.Ready.Status = http.StatusCreated
	}
	m.Platforms[hostKey()] = platform
	ex.reg.engines[engine] = m
	p.Origin = settings.SwitchOriginRemote
	p = prepareSwitch(t, ex, p)
	if engine == "target" {
		switchStep(t, ex, p, settings.SwitchStepSourceStop)
	}
	st, err := ex.state(engine)
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.port = port
	st.mu.Unlock()
	saved, err := ex.desired.load()
	if err != nil {
		t.Fatal(err)
	}
	cp := saved.PendingSwitch
	if cp == nil {
		t.Fatal("prepared switch has no checkpoint")
	}
	if engine == "target" {
		cp.Phase = settings.SwitchCheckpointStartPending
	} else {
		cp.Phase = settings.SwitchCheckpointStopPending
	}
	if err := ex.saveSwitch(cp, engine, false); err != nil {
		t.Fatalf("persist interrupted cleanup endpoint: %v", err)
	}
	if route != "pending source stop" {
		admitRecoveryVeto(t, ex, p, engine+" Off")
	}
	restarted := NewExecutor(ex.reg, NewReporter(nil), nil, ex.baseDir)
	t.Cleanup(restarted.StopAll)
	restarted.nodeProjection.set(nodeProjection{})
	st, err = restarted.state(engine)
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.installDir = dir
	st.mu.Unlock()
	return restarted, p, st, platform.Runtime.Bin, port
}

func assertSwitchRecoveryCleanupIntents(t *testing.T, ex *Executor, p settings.SwitchPrepareRequest, route string) {
	t.Helper()
	saved, err := ex.desired.load()
	if err != nil {
		t.Fatal(err)
	}
	cp := saved.PendingSwitch
	if cp == nil || cp.SwitchPrepareRequest != p || cp.TargetCommitted || cp.Phase == settings.SwitchCheckpointReleased {
		t.Fatalf("cleanup lost reservation or committed target: %+v", saved)
	}
	if saved.Engines["source"] || saved.Engines["target"] {
		t.Fatalf("cleanup changed durable Off intent: %+v", saved)
	}
	if (route == "target local Off" && !cp.TargetLocalOff) || (route == "source local Off" && !cp.SourceLocalOff) {
		t.Fatalf("cleanup lost persisted local Off: %+v", cp)
	}
	for _, engine := range []string{"source", "target"} {
		st, err := ex.state(engine)
		if err != nil {
			t.Fatal(err)
		}
		st.mu.Lock()
		owned, generation := st.proc != nil, st.gen
		st.mu.Unlock()
		if owned || generation != 0 {
			t.Errorf("cleanup launched %s: owned=%v generation=%d", engine, owned, generation)
		}
	}
}

func TestSwitchRecoveryCleanupCancelledBeforeSignal(t *testing.T) {
	for _, route := range []string{"target local Off", "source local Off", "pending source stop"} {
		t.Run(route, func(t *testing.T) {
			ex, p, st, bin, port := switchRecoveryCleanupFixture(t, route, false)
			orphan := spawnFakeListener(t, bin, port)
			engine := "target"
			if route != "target local Off" {
				engine = "source"
			}
			// Recover the durable endpoint before Status, as SwitchReconcile does.
			st.mu.Lock()
			st.port = port
			st.mu.Unlock()
			status, err := ex.Status(engine)
			if err != nil || !status.Running || !status.Healthy {
				t.Fatalf("healthy passive adoption: %+v err=%v", status, err)
			}
			st.mu.Lock()
			adopted := st.adopted && st.proc == nil
			st.mu.Unlock()
			if !adopted {
				t.Fatal("fixture did not passively adopt the managed orphan")
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			result := ex.SwitchReconcile(ctx, settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
			if result.Error == nil || !strings.Contains(result.Error.Message, "canceled") {
				t.Fatalf("cancelled cleanup = %+v", result)
			}
			observed := result.Observed.Target
			if engine == "source" {
				observed = result.Observed.Source
			}
			if observed.TerminationVerified || !pidAlive(orphan.Process.Pid) || !portServing(port) {
				t.Fatalf("cancelled cleanup signalled orphan or verified termination: observed=%+v alive=%v serving=%v", observed, pidAlive(orphan.Process.Pid), portServing(port))
			}
			assertSwitchRecoveryCleanupIntents(t, ex, p, route)
		})
	}
}

func TestSwitchRecoveryCleanupReclaimsUnhealthyOrphan(t *testing.T) {
	for _, route := range []string{"target local Off", "source local Off", "pending source stop"} {
		t.Run(route, func(t *testing.T) {
			ex, p, st, bin, port := switchRecoveryCleanupFixture(t, route, true)
			orphan := spawnFakeListener(t, bin, port)
			st.mu.Lock()
			cold := !st.running && !st.adopted && st.proc == nil && st.port == 0
			st.mu.Unlock()
			if !cold {
				t.Fatal("replacement manager must have no endpoint or adopted process state")
			}
			result := ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
			observed := result.Observed.Target
			if route != "target local Off" {
				observed = result.Observed.Source
			}
			if !observed.TerminationVerified || !orphanPIDExited(orphan.Process.Pid) || portServing(port) {
				t.Fatalf("unhealthy orphan cleanup: observed=%+v exited=%v serving=%v error=%+v", observed, orphanPIDExited(orphan.Process.Pid), portServing(port), result.Error)
			}
			if route == "pending source stop" && (result.Error != nil || result.Checkpoint.Phase != settings.SwitchCheckpointSourceOff) {
				t.Fatalf("pending source cleanup did not finish source Off: %+v", result)
			}
			st.mu.Lock()
			fenced, pid, recoveredPort := st.reclaimFailed, st.reclaimPID, st.port
			st.mu.Unlock()
			if fenced || pid != 0 || recoveredPort != port {
				t.Fatalf("verified cleanup retained fence or lost endpoint: fenced=%v pid=%d port=%d want=%d", fenced, pid, recoveredPort, port)
			}
			assertSwitchRecoveryCleanupIntents(t, ex, p, route)
		})
	}
}

func TestSwitchRecoveryLocalOffPreservesUnownedListeners(t *testing.T) {
	for _, scenario := range []string{"foreign image", "external install", "different port"} {
		t.Run(scenario, func(t *testing.T) {
			ex, p, st, bin, port := switchRecoveryCleanupFixture(t, "target local Off", true)
			listenerPort := port
			switch scenario {
			case "foreign image":
				bin = filepath.Join(testbin.Dir(t, "nvpair-engine-manager", "stop-foreign", "foreignengine"), testbin.Exe("foreignengine"))
				copyFile(t, fakeEngineBin, bin)
			case "external install":
				st.mu.Lock()
				st.installDir = ""
				st.mu.Unlock()
			case "different port":
				var err error
				listenerPort, err = freePort()
				if err != nil {
					t.Fatal(err)
				}
			}
			listener := spawnFakeListener(t, bin, listenerPort)
			result := ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
			if !pidAlive(listener.Process.Pid) || !portServing(listenerPort) {
				t.Fatal("local Off cleanup touched an unowned listener")
			}
			if result.Observed.Target.TerminationVerified != (scenario == "different port") {
				t.Fatalf("unowned listener termination observation: %+v", result)
			}
			assertSwitchRecoveryCleanupIntents(t, ex, p, "target local Off")
		})
	}
}
