// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
)

func automaticSwitchFixture(t *testing.T) (*Executor, settings.SwitchPrepareRequest) {
	t.Helper()
	ex, p := switchFixture(t)
	if err := ex.Start(context.Background(), "source"); err != nil {
		t.Fatal(err)
	}
	p.Automatic = true
	return ex, p
}

func TestEngineBindingsSwitchSourceEvidence(t *testing.T) {
	test := func(name string, change func(*testing.T, *Executor)) {
		t.Run(name, func(t *testing.T) {
			ex, p := automaticSwitchFixture(t)
			change(t, ex)
			snapshot := ex.SwitchPrepare(context.Background(), settings.SwitchPrepareRequest{})
			if snapshot.Error != nil {
				t.Fatal(snapshot.Error)
			}
			p.ConfigurationFence = snapshot.ConfigurationFence
			result := ex.SwitchPrepare(context.Background(), p)
			if result.Error == nil || result.Error.Kind != settings.SwitchErrorValidation {
				t.Fatalf("prepare=%+v", result)
			}
			if result.Checkpoint != nil {
				t.Fatal("invalid source reserved")
			}
		})
	}
	test("adopted source", func(t *testing.T, ex *Executor) {
		st, err := ex.state("source")
		if err != nil {
			t.Fatal(err)
		}
		st.mu.Lock()
		st.adopted = true
		st.mu.Unlock()
	})
	test("uncertain reclaim", func(t *testing.T, ex *Executor) {
		st, err := ex.state("source")
		if err != nil {
			t.Fatal(err)
		}
		st.mu.Lock()
		st.reclaimFailed = true
		st.mu.Unlock()
		t.Cleanup(func() { st.mu.Lock(); st.reclaimFailed = false; st.mu.Unlock() })
	})
	test("another running source", func(t *testing.T, ex *Executor) {
		if err := ex.Start(context.Background(), "other"); err != nil {
			t.Fatal(err)
		}
	})
}

func TestEngineBindingsSwitchRejectsSourceAdoptionBeforeEffects(t *testing.T) {
	ex, p := automaticSwitchFixture(t)
	p = prepareSwitch(t, ex, p)
	st, err := ex.state("source")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.adopted = true
	st.mu.Unlock()
	result := ex.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: settings.SwitchStepSourceStop})
	if result.Error == nil {
		t.Fatal("adoption race stopped source")
	}
	ex.desired.mu.Lock()
	state, err := ex.desired.load()
	ex.desired.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if !state.Engines["source"] || state.Engines["target"] || state.PendingSwitch.Phase != settings.SwitchCheckpointReserved {
		t.Fatalf("source effect before refusal: %+v", state)
	}
	if err := ex.Start(context.Background(), "other"); err == nil {
		t.Fatal("automatic sole-source reservation allowed another engine start")
	}
}

func TestEngineBindingsSwitchLocalOffAndStaleScope(t *testing.T) {
	ex, p := automaticSwitchFixture(t)
	p = prepareSwitch(t, ex, p)
	done, err := ex.admitStop("source", intentLocal, p.OperationID)
	if err != nil || done == nil {
		t.Fatalf("local automatic Off=%v", err)
	}
	done()
	result := ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p, LocalVeto: &settings.SwitchLocalVeto{Engine: "source"}})
	if result.Checkpoint == nil || !result.Observed.Source.TerminationVerified {
		t.Fatalf("Off cleanup=%+v", result)
	}
	switchStep(t, ex, p, settings.SwitchStepRelease)
	if err := ex.Start(context.Background(), "source"); err != nil {
		t.Fatal(err)
	}
	if done, err := ex.admitStop("source", intentLocal, p.OperationID); err != nil || done != nil {
		t.Fatalf("stale scoped Off admitted=%t err=%v", done != nil, err)
	}
	status, err := ex.Status("source")
	if err != nil || !status.Running {
		t.Fatalf("stale Off affected newer On: %+v, %v", status, err)
	}
}

func TestEngineBindingsSwitchAutomaticDeadline(t *testing.T) {
	ex, p := automaticSwitchFixture(t)
	p.Deadline = time.Now().Add(settings.SwitchOperationBudget)
	snapshot := ex.SwitchPrepare(context.Background(), settings.SwitchPrepareRequest{})
	if snapshot.Error != nil {
		t.Fatal(snapshot.Error)
	}
	p.ConfigurationFence = snapshot.ConfigurationFence
	result := ex.SwitchPrepare(context.Background(), p)
	if result.Error == nil || result.Error.Kind != settings.SwitchErrorInvalidRequest {
		t.Fatalf("unbounded automatic prepare=%+v", result)
	}
}

func TestEngineBindingsSwitchAlternatingTargets(t *testing.T) {
	ex, p := automaticSwitchFixture(t)
	// A target explicitly stopped earlier is still eligible after verified stop.
	if err := ex.Start(context.Background(), "target"); err != nil {
		t.Fatal(err)
	}
	if err := ex.Stop("target"); err != nil {
		t.Fatal(err)
	}
	p = prepareSwitch(t, ex, p)
	switchStep(t, ex, p, settings.SwitchStepSourceStop)
	switchStep(t, ex, p, settings.SwitchStepTargetStart)
	switchStep(t, ex, p, settings.SwitchStepRelease)
	p.OperationID = "automatic-return"
	p.SourceEngine, p.TargetEngine = p.TargetEngine, p.SourceEngine
	p = prepareSwitch(t, ex, p)
	switchStep(t, ex, p, settings.SwitchStepSourceStop)
	result := switchStep(t, ex, p, settings.SwitchStepTargetStart)
	if !result.Checkpoint.TargetCommitted {
		t.Fatal("return switch did not commit")
	}
	switchStep(t, ex, p, settings.SwitchStepRelease)
}

func TestEngineBindingsSwitchAdoptionAfterDurableOff(t *testing.T) {
	ex, p := automaticSwitchFixture(t)
	p = prepareSwitch(t, ex, p)
	st, err := ex.state("source")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	original := st.proc
	st.mu.Unlock()
	if original == nil {
		t.Fatal("fixture has no owned process")
	}
	// Intent emission is after the durable Off write and before the stop owner.
	// Simulate the watcher losing its handle and reconciliation adopting a new
	// incarnation at precisely that boundary, after switch preflight succeeded.
	ex.emit = func(method string, _ any) {
		if method == intentChangedMethod {
			st.mu.Lock()
			st.proc, st.adopted, st.runID = nil, true, "replacement"
			st.mu.Unlock()
		}
	}
	t.Cleanup(func() {
		ex.emit = nil
		st.mu.Lock()
		st.proc, st.adopted = original, false
		st.mu.Unlock()
	})
	result := ex.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: settings.SwitchStepSourceStop})
	if result.Error == nil || result.Observed.Source.TerminationVerified {
		t.Fatalf("late adoption stop=%+v", result)
	}
	if result.Checkpoint.Phase != settings.SwitchCheckpointReserved {
		t.Fatal("ownership refusal was left eligible for source reclaim")
	}
	reconciled := ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
	if reconciled.Checkpoint == nil || reconciled.Checkpoint.Phase != settings.SwitchCheckpointReserved {
		t.Fatalf("refused source cleanup=%+v", reconciled)
	}
	select {
	case <-original.done:
		t.Fatal("late adoption caused process termination")
	default:
	}
	target, err := ex.Status("target")
	if err != nil || target.Running {
		t.Fatalf("target after adoption refusal=%+v %v", target, err)
	}
}

type automaticSourceRecovery struct {
	ex       *Executor
	request  settings.SwitchPrepareRequest
	original *managedProc
	bin      string
	port     int
}

func automaticSourceRecoveryFixture(t *testing.T) automaticSourceRecovery {
	t.Helper()
	m, installDir, port := restoreOrphanManifest(t)
	m.Engine = "source"
	reg := NewRegistry()
	reg.engines["source"] = m
	target := testEngineManifest(fakeEngineBin)
	target.Engine = "target"
	reg.engines["target"] = target
	base := t.TempDir()
	ex := NewExecutor(reg, NewReporter(nil), nil, base)
	t.Cleanup(ex.StopAll)
	ex.nodeProjection.set(nodeProjection{ManagementBlocked: false})
	source, err := ex.state("source")
	if err != nil {
		t.Fatal(err)
	}
	source.installDir = installDir
	if err := ex.Start(context.Background(), "source"); err != nil {
		t.Fatal(err)
	}
	p := settings.SwitchPrepareRequest{
		Automatic: true, OperationID: "automatic-source-recovery", Caller: settings.SwitchOriginLocal, Origin: settings.SwitchOriginLocal,
		SourceEngine: "source", TargetEngine: "target", PayloadHash: "sha256:" + strings.Repeat("a", 64),
		ManagerIncarnation: ex.switchIncarnation, Deadline: time.Now().Add(time.Minute).UTC(),
	}
	p = prepareSwitch(t, ex, p)
	source.mu.Lock()
	original := source.proc
	runID := source.runID
	source.mu.Unlock()
	if original == nil {
		t.Fatal("source fixture has no child")
	}
	ex.desired.mu.Lock()
	state, err := ex.desired.load()
	if err == nil {
		if state.PendingSwitchPorts == nil || state.PendingSwitchPorts.SourcePID != original.cmd.Process.Pid || state.PendingSwitchPorts.SourceRunID != runID {
			ex.desired.mu.Unlock()
			t.Fatal("source execution evidence was not reserved")
		}
		state.PendingSwitch.Phase = settings.SwitchCheckpointStopPending
		state.setIntent("source", false, intentLocal)
		err = ex.desired.save(state)
	}
	ex.desired.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	restarted := NewExecutor(reg, NewReporter(nil), nil, base)
	t.Cleanup(restarted.StopAll)
	restarted.nodeProjection.set(nodeProjection{ManagementBlocked: false})
	restoredSource, err := restarted.state("source")
	if err != nil {
		t.Fatal(err)
	}
	restoredSource.installDir = installDir
	return automaticSourceRecovery{restarted, p, original, m.Platforms[hostKey()].Runtime.Bin, port}
}

func TestEngineBindingsSwitchRecoveryRetainsOriginalChildFence(t *testing.T) {
	t.Run("reclaims the original controlled orphan", func(t *testing.T) {
		f := automaticSourceRecoveryFixture(t)
		result := f.ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: f.request})
		if result.Error != nil || result.Checkpoint == nil || result.Checkpoint.Phase != settings.SwitchCheckpointSourceOff || !result.Observed.Source.TerminationVerified {
			t.Fatalf("original orphan recovery=%+v", result)
		}
		select {
		case <-f.original.done:
		default:
			t.Fatal("original source was not reaped")
		}
	})
	t.Run("refuses a replacement from the same managed image", func(t *testing.T) {
		f := automaticSourceRecoveryFixture(t)
		f.original.stop(0)
		replacement := spawnFakeListener(t, f.bin, f.port)
		result := f.ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: f.request})
		if result.Error == nil || result.Checkpoint == nil || result.Checkpoint.Phase != settings.SwitchCheckpointStopPending || result.Observed.Source.TerminationVerified {
			t.Fatalf("replacement recovery=%+v", result)
		}
		if !pidAlive(replacement.Process.Pid) {
			t.Fatal("automatic cleanup terminated the replacement")
		}
	})
}

func TestEngineBindingsSwitchOwnedStopBudget(t *testing.T) {
	test := func(name string, grace time.Duration) {
		t.Run(name, func(t *testing.T) {
			proc, err := startManagedProc(fakeEngineBin, []string{"noserve"}, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { proc.stop(0) })
			// Keep reaping evidence withheld after signalling. A forced signal
			// alone must not count as termination or lead to an unbounded wait.
			pending := &managedProc{cmd: proc.cmd, done: make(chan struct{})}
			ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
			defer cancel()
			started := time.Now()
			if err := pending.stopContext(ctx, grace); !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("stop=%v", err)
			}
			if time.Since(started) > time.Second {
				t.Fatal("stop exceeded preparation deadline")
			}
		})
	}
	test("grace does not extend remaining budget", time.Hour)
	test("forced signal still requires reaping evidence", 0)
}

func TestEngineBindingsSwitchCommandStopBudget(t *testing.T) {
	ex, _ := switchFixture(t)
	st, err := ex.state("source")
	if err != nil {
		t.Fatal(err)
	}
	rt := Runtime{Stop: &StopSpec{Cmd: []string{fakeEngineBin, "noserve"}, GraceS: 60}}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	started := time.Now()
	if err := ex.runCommandStopContext(ctx, st, "source", rt, 0); err == nil {
		t.Fatal("hung stop CLI succeeded")
	}
	if ctx.Err() == nil || time.Since(started) > time.Second {
		t.Fatal("command stop reset its preparation budget")
	}
}

func TestEngineBindingsSwitchReconcileStopBudget(t *testing.T) {
	ex, p := automaticSwitchFixture(t)
	p = prepareSwitch(t, ex, p)
	st, err := ex.state("source")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	original := st.proc
	if original == nil {
		st.mu.Unlock()
		t.Fatal("fixture has no owned process")
	}
	// Withhold reaping evidence after signalling, as a blocked output reader
	// can do. Cleanup must retain its reservation when its context expires.
	st.proc = &managedProc{cmd: original.cmd, done: make(chan struct{})}
	st.mu.Unlock()
	t.Cleanup(func() { st.mu.Lock(); st.proc = original; st.mu.Unlock() })
	ex.desired.mu.Lock()
	state, err := ex.desired.load()
	if err == nil {
		state.PendingSwitch.Phase = settings.SwitchCheckpointStopPending
		state.setIntent("source", false, intentLocal)
		err = ex.desired.save(state)
	}
	ex.desired.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(100 * time.Millisecond)
	started := time.Now()
	result := ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p, CleanupDeadline: &deadline})
	if result.Error == nil || result.Checkpoint == nil || result.Checkpoint.Phase != settings.SwitchCheckpointStopPending || result.Observed.Source.TerminationVerified {
		t.Fatalf("expired cleanup released ownership: %+v", result)
	}
	if time.Since(started) > time.Second {
		t.Fatal("cleanup stop exceeded its wire budget")
	}
}

func TestEngineBindingsSwitchFailedTargetCleanupBudget(t *testing.T) {
	test := func(name string, configure func(*Runtime)) {
		t.Run(name, func(t *testing.T) {
			ex, p := automaticSwitchFixture(t)
			st, err := ex.state("target")
			if err != nil {
				t.Fatal(err)
			}
			st.mu.Lock()
			configure(&st.plat.Runtime)
			st.mu.Unlock()
			p = prepareSwitch(t, ex, p)
			switchStep(t, ex, p, settings.SwitchStepSourceStop)
			ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
			defer cancel()
			started := time.Now()
			result := ex.SwitchStep(ctx, settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: settings.SwitchStepTargetStart})
			if result.Error == nil || result.Checkpoint == nil || result.Checkpoint.TargetCommitted || result.Observed.Target.TerminationVerified {
				t.Fatalf("failed target cleanup=%+v", result)
			}
			if time.Since(started) > time.Second {
				t.Fatal("failed target cleanup extended the preparation budget")
			}
			if result.Checkpoint.Phase != settings.SwitchCheckpointStartPending {
				t.Fatal("uncertain target cleanup lost its reservation")
			}
		})
	}
	test("owned process retains cleanup evidence", func(rt *Runtime) {
		rt.Args = []string{"noserve"}
	})
	test("stop CLI does not get another budget", func(rt *Runtime) {
		rt.Mode = "command"
		rt.Start = [][]string{{fakeEngineBin, "noserve"}}
		rt.Stop = &StopSpec{Cmd: []string{fakeEngineBin, "noserve"}, GraceS: 60}
	})
}
