// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
)

func committedRecoveryVetoFixture(t *testing.T) (*Executor, settings.SwitchPrepareRequest, *engineState, string, int) {
	t.Helper()
	m, managedDir, port := restoreOrphanManifest(t)
	ex, p := switchFixture(t)
	m.Engine = "target"
	ex.reg.engines["target"] = m
	p.Origin = settings.SwitchOriginRemote
	p = prepareSwitch(t, ex, p)
	cp := switchStep(t, ex, p, settings.SwitchStepSourceStop).Checkpoint
	cp.Phase = settings.SwitchCheckpointTargetCommitted
	cp.TargetCommitted = true
	cp.Target.Intent = settings.SwitchIntentOn
	if err := ex.saveSwitch(cp, "target", true); err != nil {
		t.Fatalf("persist committed target: %v", err)
	}
	restarted := NewExecutor(ex.reg, NewReporter(nil), nil, ex.baseDir)
	t.Cleanup(restarted.StopAll)
	restarted.nodeProjection.set(nodeProjection{ManagementBlocked: false})
	st, err := restarted.state("target")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.installDir = managedDir
	st.mu.Unlock()
	return restarted, p, st, m.Platforms[hostKey()].Runtime.Bin, port
}

// Admit just the veto, leaving physical stop delivery held as the broker may
// do. This exercises the existing durable admission and cancellation owners.
func admitRecoveryVeto(t *testing.T, ex *Executor, p settings.SwitchPrepareRequest, scenario string) {
	t.Helper()
	if scenario == "local pause" {
		(&Manager{exec: ex}).applyNodeProjection(&Message{Params: json.RawMessage(`{"pausedLocally":true,"managementBlocked":true}`)})
		return
	}
	engine := "target"
	if scenario == "source Off" {
		engine = "source"
	}
	done, err := ex.admitStop(engine, intentLocal, p.OperationID)
	if err != nil {
		t.Fatalf("admit %s: %v", scenario, err)
	}
	if done == nil {
		t.Fatalf("%s was not admitted against the checkpoint", scenario)
	}
	done()
}

func assertRecoveryVeto(t *testing.T, ex *Executor, result settings.SwitchStepResult) {
	t.Helper()
	if result.Error == nil || result.Error.Code != codeLocalVeto || result.Error.Kind != settings.SwitchErrorRemoteVeto {
		t.Fatalf("recovery veto = %+v", result)
	}
	if result.Checkpoint == nil || result.Checkpoint.TargetCommitted || result.Observed.Target.Intent != settings.SwitchIntentOff {
		t.Fatalf("veto accepted committed On: %+v", result)
	}
	ex.desired.mu.Lock()
	saved, err := ex.desired.load()
	ex.desired.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if saved.Engines["source"] || saved.Engines["target"] || saved.PendingSwitch == nil || saved.PendingSwitch.TargetCommitted {
		t.Fatalf("veto changed durable Off/commit: %+v", saved)
	}
	if source, err := ex.Status("source"); err != nil || source.Running {
		t.Fatalf("veto restored source: %+v, %v", source, err)
	}
}

func waitForCommittedRecoveryRegistration(t *testing.T, ex *Executor) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		ex.remoteWork.mu.Lock()
		registered := false
		for _, op := range ex.remoteWork.ops {
			registered = registered || op.engine == "target"
		}
		ex.remoteWork.mu.Unlock()
		if registered {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("committed recovery did not register remote work before opMu")
		}
		time.Sleep(time.Millisecond)
	}
}

func TestSwitchCommittedRecoveryVetoBeforeLaunchPreservesOrphan(t *testing.T) {
	for _, scenario := range []string{"local pause", "target Off", "source Off"} {
		t.Run(scenario, func(t *testing.T) {
			ex, p, st, bin, port := committedRecoveryVetoFixture(t)
			orphan := spawnFakeListener(t, bin, port)
			st.opMu.Lock()
			release := sync.OnceFunc(st.opMu.Unlock)
			defer release()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			completed := make(chan settings.SwitchStepResult, 1)
			go func() {
				completed <- ex.SwitchReconcile(ctx, settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
			}()
			// Registration proves recovery passed its initial pause check. The
			// held target opMu prevents reclaim or launch before veto admission.
			waitForCommittedRecoveryRegistration(t, ex)
			admitRecoveryVeto(t, ex, p, scenario)
			ex.remoteWork.mu.Lock()
			remaining := len(ex.remoteWork.ops)
			ex.remoteWork.mu.Unlock()
			if remaining != 0 {
				t.Fatalf("veto retained %d remote operations", remaining)
			}
			release()
			select {
			case result := <-completed:
				assertRecoveryVeto(t, ex, result)
			case <-time.After(10 * time.Second):
				t.Fatal("vetoed recovery did not finish")
			}
			st.mu.Lock()
			owned, running, generation := st.proc != nil, st.running, st.gen
			st.mu.Unlock()
			if owned || running || generation != 0 || !pidAlive(orphan.Process.Pid) || !portServing(port) {
				t.Fatalf("veto before signal caused lifecycle effects: owned=%v running=%v generation=%d orphanAlive=%v serving=%v", owned, running, generation, pidAlive(orphan.Process.Pid), portServing(port))
			}
		})
	}
}

func TestSwitchCommittedRecoveryVetoBeforeReadinessCommitStopsLaunch(t *testing.T) {
	for _, scenario := range []string{"local pause", "target Off", "source Off"} {
		t.Run(scenario, func(t *testing.T) {
			ex, p, st, _, _ := committedRecoveryVetoFixture(t)
			var veto sync.Once
			var launched atomic.Bool
			ex.emit = func(method string, _ any) {
				if method != "engine:state-changed" {
					return
				}
				st.mu.Lock()
				owned := st.proc != nil
				st.mu.Unlock()
				if owned {
					veto.Do(func() {
						launched.Store(true)
						admitRecoveryVeto(t, ex, p, scenario)
					})
				}
			}
			result := ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
			if !launched.Load() {
				t.Fatal("recovery did not reach owned readiness before veto")
			}
			assertRecoveryVeto(t, ex, result)
			st.mu.Lock()
			owned, running := st.proc != nil, st.running
			st.mu.Unlock()
			if owned || running || !result.Observed.Target.TerminationVerified {
				t.Fatalf("veto after launch retained target: owned=%v running=%v observation=%+v", owned, running, result.Observed.Target)
			}
		})
	}
}

func TestSwitchCommittedRecoveryRevalidatesDurableStateBeforeLaunch(t *testing.T) {
	for _, scenario := range []string{"target durable Off", "checkpoint released"} {
		t.Run(scenario, func(t *testing.T) {
			ex, p, st, bin, port := committedRecoveryVetoFixture(t)
			orphan := spawnFakeListener(t, bin, port)
			st.opMu.Lock()
			release := sync.OnceFunc(st.opMu.Unlock)
			defer release()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			completed := make(chan settings.SwitchStepResult, 1)
			go func() {
				completed <- ex.SwitchReconcile(ctx, settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
			}()
			waitForCommittedRecoveryRegistration(t, ex)
			// Deliberately leave the registered context uncanceled, requiring
			// the launch owner to read current durability rather than its copy.
			ex.desired.mu.Lock()
			current, err := ex.desired.load()
			if err == nil {
				if scenario == "target durable Off" {
					current.setIntent("target", false, intentPeer)
				} else {
					current.PendingSwitch.Phase = settings.SwitchCheckpointReleased
				}
				err = ex.desired.save(current)
			}
			ex.desired.mu.Unlock()
			if err != nil {
				t.Fatalf("change durable state: %v", err)
			}
			release()
			select {
			case result := <-completed:
				if result.Error == nil {
					t.Fatalf("accepted stale committed On: %+v", result)
				}
			case <-time.After(10 * time.Second):
				t.Fatal("recovery did not reject changed durable state")
			}
			st.mu.Lock()
			owned, running, generation := st.proc != nil, st.running, st.gen
			st.mu.Unlock()
			if owned || running || generation != 0 || !pidAlive(orphan.Process.Pid) || !portServing(port) {
				t.Fatalf("stale recovery caused lifecycle effects: owned=%v running=%v generation=%d orphanAlive=%v", owned, running, generation, pidAlive(orphan.Process.Pid))
			}
			ex.desired.mu.Lock()
			saved, err := ex.desired.load()
			ex.desired.mu.Unlock()
			if err != nil {
				t.Fatal(err)
			}
			if saved.Engines["source"] || saved.Engines["target"] != current.Engines["target"] || saved.PendingSwitch == nil || saved.PendingSwitch.Phase != current.PendingSwitch.Phase {
				t.Fatalf("stale recovery replaced current durable state: %+v", saved)
			}
			if source, err := ex.Status("source"); err != nil || source.Running {
				t.Fatalf("stale recovery restored source: %+v, %v", source, err)
			}
		})
	}
}

func TestSwitchSavePreservesUnacknowledgedLocalOffEvidence(t *testing.T) {
	for _, scenario := range []string{"target Off", "source Off"} {
		t.Run(scenario, func(t *testing.T) {
			ex, p := switchFixture(t)
			p.Origin = settings.SwitchOriginRemote
			p = prepareSwitch(t, ex, p)
			stale, err := ex.desired.load()
			if err != nil {
				t.Fatal(err)
			}
			admitRecoveryVeto(t, ex, p, scenario)
			admitted, err := ex.desired.load()
			if err != nil {
				t.Fatal(err)
			}
			// A private checkpoint has not acknowledged the admitted Off. Its
			// claimed observations cannot replace the admission's evidence.
			cp := stale.PendingSwitch
			cp.Source = settings.SwitchEngineState{Intent: settings.SwitchIntentOn, TerminationVerified: true}
			cp.Target = settings.SwitchEngineState{Intent: settings.SwitchIntentOff, TerminationVerified: true}
			if err := ex.saveSwitch(cp, "", false); err != nil {
				t.Fatal(err)
			}
			saved, err := ex.desired.load()
			if err != nil {
				t.Fatal(err)
			}
			if saved.PendingSwitch == nil || !reflect.DeepEqual(saved.PendingSwitch.Error, admitted.PendingSwitch.Error) {
				t.Fatalf("stale checkpoint replaced durable veto: %+v", saved.PendingSwitch)
			}
			got, want := saved.PendingSwitch.Target, admitted.PendingSwitch.Target
			flag := saved.PendingSwitch.TargetLocalOff
			engine := "target"
			if scenario == "source Off" {
				got, want = saved.PendingSwitch.Source, admitted.PendingSwitch.Source
				flag = saved.PendingSwitch.SourceLocalOff
				engine = "source"
			}
			if !flag || !reflect.DeepEqual(got, want) || saved.Engines[engine] || saved.OffBy[engine] != intentLocal {
				t.Fatalf("stale checkpoint replaced local Off: flag=%v got=%+v want=%+v intent=%v provenance=%q", flag, got, want, saved.Engines[engine], saved.OffBy[engine])
			}
		})
	}
}
