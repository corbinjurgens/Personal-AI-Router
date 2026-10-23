// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	settings "nvpair-shared/enginesettings"
)

func switchFixture(t *testing.T) (*Executor, settings.SwitchPrepareRequest) {
	t.Helper()
	reg := NewRegistry()
	for _, name := range []string{"source", "target", "other"} {
		manifest := testEngineManifest(fakeEngineBin)
		manifest.Engine = name
		reg.engines[name] = manifest
	}
	ex := NewExecutor(reg, NewReporter(nil), nil, t.TempDir())
	t.Cleanup(ex.StopAll)
	ex.nodeProjection.set(nodeProjection{ManagementBlocked: false})
	p := settings.SwitchPrepareRequest{OperationID: "switch-test", Caller: "local", Origin: settings.SwitchOriginLocal, SourceEngine: "source", TargetEngine: "target", PayloadHash: "sha256:0123456789012345678901234567890123456789012345678901234567890123", ManagerIncarnation: ex.switchIncarnation, Deadline: time.Now().Add(time.Minute).UTC()}
	return ex, p
}

func prepareSwitch(t *testing.T, ex *Executor, p settings.SwitchPrepareRequest) settings.SwitchPrepareRequest {
	t.Helper()
	snapshot := ex.SwitchPrepare(context.Background(), settings.SwitchPrepareRequest{})
	if snapshot.Error != nil {
		t.Fatalf("snapshot: %+v", snapshot.Error)
	}
	p.ConfigurationFence = snapshot.ConfigurationFence
	result := ex.SwitchPrepare(context.Background(), p)
	if result.Error != nil || result.Checkpoint == nil {
		t.Fatalf("prepare: %+v", result)
	}
	return p
}

func switchStep(t *testing.T, ex *Executor, p settings.SwitchPrepareRequest, step string) settings.SwitchStepResult {
	t.Helper()
	result := ex.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: step})
	if result.Error != nil {
		t.Fatalf("%s: %+v", step, result.Error)
	}
	return result
}

func TestSwitchPrepareValidations(t *testing.T) {
	for _, tc := range []struct {
		name   string
		modify func(*Executor, *settings.SwitchPrepareRequest)
		kind   string
	}{
		{"equal IDs", func(_ *Executor, p *settings.SwitchPrepareRequest) { p.TargetEngine = p.SourceEngine }, settings.SwitchErrorInvalidRequest},
		{"unregistered", func(_ *Executor, p *settings.SwitchPrepareRequest) { p.TargetEngine = "absent" }, settings.SwitchErrorValidation},
		{"unprepared", func(ex *Executor, _ *settings.SwitchPrepareRequest) {
			ex.reg.engines["target"].Platforms = map[string]Platform{hostKey(): {Detect: []string{filepath.Join(t.TempDir(), "missing")}}}
		}, settings.SwitchErrorUnprepared},
		{"platform", func(ex *Executor, _ *settings.SwitchPrepareRequest) {
			ex.reg.engines["target"].Platforms = map[string]Platform{}
		}, settings.SwitchErrorValidation},
		{"running target", func(ex *Executor, _ *settings.SwitchPrepareRequest) {
			if err := ex.Start(context.Background(), "target"); err != nil {
				t.Fatal(err)
			}
		}, settings.SwitchErrorTargetRunning},
		{"invalid port", func(ex *Executor, _ *settings.SwitchPrepareRequest) {
			st, err := ex.state("target")
			if err != nil {
				t.Fatal(err)
			}
			st.port = -1
		}, settings.SwitchErrorValidation},
		{"no readiness", func(ex *Executor, _ *settings.SwitchPrepareRequest) {
			st, err := ex.state("target")
			if err != nil {
				t.Fatal(err)
			}
			st.plat.Runtime.Ready = nil
		}, settings.SwitchErrorValidation},
		{"remote locked target", func(ex *Executor, p *settings.SwitchPrepareRequest) {
			p.Origin = settings.SwitchOriginRemote
			if _, err := ex.desired.setLocked("target", true); err != nil {
				t.Fatal(err)
			}
		}, settings.SwitchErrorEngineLocked},
		{"remote local pause", func(ex *Executor, p *settings.SwitchPrepareRequest) {
			p.Origin = settings.SwitchOriginRemote
			ex.nodeProjection.set(nodeProjection{PausedLocally: true, ManagementBlocked: true})
		}, settings.SwitchErrorRemoteVeto},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ex, p := switchFixture(t)
			tc.modify(ex, &p)
			snapshot := ex.SwitchPrepare(context.Background(), settings.SwitchPrepareRequest{})
			if snapshot.Error != nil {
				t.Fatal(snapshot.Error)
			}
			p.ConfigurationFence = snapshot.ConfigurationFence
			result := ex.SwitchPrepare(context.Background(), p)
			if result.Error == nil || result.Error.Kind != tc.kind {
				t.Fatalf("error = %+v, want %s", result.Error, tc.kind)
			}
			if tc.kind == settings.SwitchErrorRemoteVeto && result.Error.Code != -32011 {
				t.Fatalf("veto code = %d", result.Error.Code)
			}
		})
	}
}

func TestSwitchReservationBlocksMutationEntries(t *testing.T) {
	ex, p := switchFixture(t)
	if err := ex.desired.set("source", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	p = prepareSwitch(t, ex, p)
	st, err := ex.state("source")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		call func() error
	}{
		{"start", func() error { return ex.Start(context.Background(), "source") }},
		{"stop", func() error { return ex.Stop("source") }},
		{"restart", func() error { return ex.Restart(context.Background(), "source") }},
		{"wake", func() error { return ex.Wake(context.Background(), "source") }},
		{"sleep", func() error { return ex.Sleep("source") }},
		{"install", func() error { return ex.Install(context.Background(), "source") }},
		{"install with start", func() error {
			return ex.InstallWithStart(context.Background(), context.Background(), "source", startOpts{})
		}},
		{"uninstall", func() error { return ex.Uninstall(context.Background(), "source") }},
		{"set port", func() error { _, err := ex.SetPort(context.Background(), "source", 12345); return err }},
		{"configure", func() error {
			_, err := ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "source"}, func() error { t.Fatal("rebind called"); return nil })
			return err
		}},
		{"restart after", func() error { return ex.restartAfterAction(context.Background(), st, "source", "delete") }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var busy *switchBusyError
			if err := tc.call(); !errors.As(err, &busy) {
				t.Fatalf("error = %v, want reservation busy", err)
			}
		})
	}
	if err := ex.RestoreEnabled(context.Background()); err != nil {
		t.Fatal(err)
	}
	if status, err := ex.Status("source"); err != nil || status.Running {
		t.Fatalf("restore started source: %+v, %v", status, err)
	}
	if err := ex.Start(context.Background(), "other"); err != nil {
		t.Fatalf("unrelated start: %v", err)
	}
	switchStep(t, ex, p, settings.SwitchStepRelease)
	if err := ex.Start(context.Background(), "source"); err != nil {
		t.Fatalf("start after release: %v", err)
	}
}

func TestSwitchSuccessCommitsIntentAndCheckpointTogether(t *testing.T) {
	ex, p := switchFixture(t)
	if err := ex.Start(context.Background(), "source"); err != nil {
		t.Fatal(err)
	}
	if err := ex.desired.set("other", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	p = prepareSwitch(t, ex, p)
	stopped := switchStep(t, ex, p, settings.SwitchStepSourceStop)
	if !stopped.Observed.Source.TerminationVerified || stopped.Observed.Source.Intent != settings.SwitchIntentOff {
		t.Fatalf("source = %+v", stopped.Observed.Source)
	}
	committed := switchStep(t, ex, p, settings.SwitchStepTargetStart)
	if !committed.Checkpoint.TargetCommitted || committed.Checkpoint.Phase != settings.SwitchCheckpointTargetCommitted {
		t.Fatalf("checkpoint = %+v", committed.Checkpoint)
	}
	ex.desired.mu.Lock()
	state, err := ex.desired.load()
	ex.desired.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(state.Engines, map[string]bool{"source": false, "target": true, "other": true}) || !state.PendingSwitch.TargetCommitted {
		t.Fatalf("durable state = %+v", state)
	}
	before, err := os.ReadFile(ex.desired.path)
	if err != nil {
		t.Fatal(err)
	}
	switchStep(t, ex, p, settings.SwitchStepSourceStop)
	switchStep(t, ex, p, settings.SwitchStepTargetStart)
	after, err := os.ReadFile(ex.desired.path)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Fatal("idempotent step rewrote checkpoint")
	}
	reloaded := newDesiredStateStore(ex.baseDir)
	reloaded.mu.Lock()
	loaded, err := reloaded.load()
	reloaded.mu.Unlock()
	if err != nil || loaded.PendingSwitch == nil || !loaded.PendingSwitch.TargetCommitted {
		t.Fatalf("reload = %+v, %v", loaded, err)
	}
}

func TestSwitchTargetFailureNeverRestoresSource(t *testing.T) {
	ex, p := switchFixture(t)
	st, err := ex.state("target")
	if err != nil {
		t.Fatal(err)
	}
	st.plat.Runtime.Args = []string{"--fail-launch"}
	p = prepareSwitch(t, ex, p)
	switchStep(t, ex, p, settings.SwitchStepSourceStop)
	result := ex.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: settings.SwitchStepTargetStart})
	if result.Error == nil || result.Error.Kind != settings.SwitchErrorTargetStart {
		t.Fatalf("result = %+v", result)
	}
	for _, name := range []string{"source", "target"} {
		on, known, err := ex.desired.get(name)
		if err != nil || !known || on {
			t.Fatalf("%s intent = %v/%v, %v", name, on, known, err)
		}
	}
}

func TestSwitchForeignSourceNeverStartsTarget(t *testing.T) {
	ex, p := switchFixture(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusForbidden) }))
	defer server.Close()
	st, err := ex.state("source")
	if err != nil {
		t.Fatal(err)
	}
	st.port = server.Listener.Addr().(*net.TCPAddr).Port
	p = prepareSwitch(t, ex, p)
	result := ex.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: settings.SwitchStepSourceStop})
	if result.Error == nil || result.Error.Kind != settings.SwitchErrorSourceStop || result.Observed.Source.TerminationVerified || result.Observed.Source.Intent != settings.SwitchIntentOff {
		t.Fatalf("source stop = %+v", result)
	}
	result = ex.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: settings.SwitchStepTargetStart})
	if result.Error == nil || result.Checkpoint.TargetCommitted {
		t.Fatalf("target start = %+v", result)
	}
}

func TestSwitchLocalAllowedDespiteSavedOff(t *testing.T) {
	ex, p := switchFixture(t)
	if err := ex.desired.set("target", false, intentLocal); err != nil {
		t.Fatal(err)
	}
	p = prepareSwitch(t, ex, p)
	switchStep(t, ex, p, settings.SwitchStepSourceStop)
	switchStep(t, ex, p, settings.SwitchStepTargetStart)
}

func TestSwitchLocalOffCancelsRemoteTargetStart(t *testing.T) {
	ex, p := switchFixture(t)
	p.Origin = settings.SwitchOriginRemote
	st, err := ex.state("target")
	if err != nil {
		t.Fatal(err)
	}
	st.plat.Runtime.Args = []string{"noserve"}
	st.plat.Runtime.Ready.TimeoutS = 60
	p = prepareSwitch(t, ex, p)
	switchStep(t, ex, p, settings.SwitchStepSourceStop)
	resultCh := make(chan settings.SwitchStepResult, 1)
	go func() {
		resultCh <- ex.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: settings.SwitchStepTargetStart})
	}()
	deadline := time.Now().Add(5 * time.Second)
	for {
		ex.remoteWork.mu.Lock()
		accepted := len(ex.remoteWork.ops) == 1
		ex.remoteWork.mu.Unlock()
		if accepted {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("target remote work was not admitted")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if err := ex.Stop("target"); err != nil {
		t.Fatalf("local stop = %v", err)
	}
	select {
	case result := <-resultCh:
		if result.Error == nil || result.Error.Code != -32011 || result.Checkpoint.TargetCommitted {
			t.Fatalf("result = %+v", result)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("local veto did not cancel start")
	}
	requireOffOrigin(t, ex.desired, "target", "local")
}

func TestSwitchCheckpointJSONPreservesLegacyIntent(t *testing.T) {
	ex, _ := switchFixture(t)
	if err := os.WriteFile(ex.desired.path, []byte(`{"engines":{"target":false,"other":true}}`), 0600); err != nil {
		t.Fatal(err)
	}
	requireOffOrigin(t, ex.desired, "target", "local")
	var state desiredStateFile
	data, err := os.ReadFile(ex.desired.path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &state); err != nil || state.PendingSwitch != nil {
		t.Fatalf("legacy decode: %+v, %v", state, err)
	}
}

func TestSwitchSourceOffIsDurableBeforeStop(t *testing.T) {
	ex, p := switchFixture(t)
	if err := ex.Start(context.Background(), "source"); err != nil {
		t.Fatal(err)
	}
	p = prepareSwitch(t, ex, p)
	observedBoundary := false
	ex.emit = func(method string, _ any) {
		if method != intentChangedMethod || observedBoundary {
			return
		}
		ex.desired.mu.Lock()
		state, err := ex.desired.load()
		ex.desired.mu.Unlock()
		if err != nil {
			t.Fatal(err)
		}
		st, err := ex.state("source")
		if err != nil {
			t.Fatal(err)
		}
		st.mu.Lock()
		running := st.running
		st.mu.Unlock()
		if state.Engines["source"] || state.PendingSwitch.Phase != settings.SwitchCheckpointStopPending || !running {
			t.Fatalf("stop boundary: state=%+v running=%v", state, running)
		}
		observedBoundary = true
	}
	switchStep(t, ex, p, settings.SwitchStepSourceStop)
	if !observedBoundary {
		t.Fatal("stop boundary was not observed")
	}
}

func TestSwitchPrepareIdentityAndConfigurationFence(t *testing.T) {
	ex, p := switchFixture(t)
	initial := ex.SwitchPrepare(context.Background(), settings.SwitchPrepareRequest{})
	if initial.Error != nil {
		t.Fatal(initial.Error)
	}
	done, err := ex.admitMutation("other")
	if err != nil {
		t.Fatal(err)
	}
	busy := ex.SwitchPrepare(context.Background(), settings.SwitchPrepareRequest{})
	done()
	if busy.Error == nil || busy.Error.Kind != settings.SwitchErrorBusy {
		t.Fatalf("snapshot during mutation = %+v", busy)
	}
	p.ConfigurationFence = initial.ConfigurationFence
	if err := ex.desired.set("other", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	stale := ex.SwitchPrepare(context.Background(), p)
	if stale.Error == nil || stale.Error.Code != -32010 {
		t.Fatalf("stale fence = %+v", stale)
	}
	p = prepareSwitch(t, ex, p)
	first := ex.SwitchPrepare(context.Background(), p)
	if first.Error != nil || first.Checkpoint == nil {
		t.Fatalf("repeat prepare = %+v", first)
	}
	p.Caller = "different"
	reused := ex.SwitchPrepare(context.Background(), p)
	if reused.Error == nil || reused.Error.Code != -32602 {
		t.Fatalf("reused ID = %+v", reused)
	}
}

func TestSwitchRecoveryDoesNotResumeTargetStart(t *testing.T) {
	ex, p := switchFixture(t)
	p = prepareSwitch(t, ex, p)
	stopped := switchStep(t, ex, p, settings.SwitchStepSourceStop)
	cp := stopped.Checkpoint
	cp.Phase = settings.SwitchCheckpointStartPending
	target, err := ex.state("target")
	if err != nil {
		t.Fatal(err)
	}
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	target.port = port
	if err := ex.saveSwitch(cp, "", false); err != nil {
		t.Fatal(err)
	}
	restarted := NewExecutor(ex.reg, NewReporter(nil), nil, ex.baseDir)
	t.Cleanup(restarted.StopAll)
	if err := restarted.RestoreEnabled(context.Background()); err != nil {
		t.Fatal(err)
	}
	result := restarted.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
	if result.Error == nil || result.Checkpoint.TargetCommitted || result.Observed.Target.Intent != settings.SwitchIntentOff {
		t.Fatalf("reconcile = %+v", result)
	}
	if *result.Observed.Target.Running {
		t.Fatal("recovery started target")
	}
	switchStep(t, restarted, p, settings.SwitchStepRelease)
	switchStep(t, restarted, p, settings.SwitchStepRelease)
}

func TestSwitchRestartRejectsForwardStepsBeforeReconciliation(t *testing.T) {
	for _, step := range []string{settings.SwitchStepSourceStop, settings.SwitchStepTargetStart} {
		t.Run(step, func(t *testing.T) {
			ex, p := switchFixture(t)
			p = prepareSwitch(t, ex, p)
			if step == settings.SwitchStepTargetStart {
				switchStep(t, ex, p, settings.SwitchStepSourceStop)
			}
			restarted := NewExecutor(ex.reg, NewReporter(nil), nil, ex.baseDir)
			t.Cleanup(restarted.StopAll)
			result := restarted.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: step})
			if result.Error == nil || result.Error.Code != -32000 || result.Error.Kind != settings.SwitchErrorInterrupted {
				t.Fatalf("inherited forward step = %+v, want coded interrupted error", result)
			}
			if result.Checkpoint.TargetCommitted {
				t.Fatal("replacement committed target")
			}
			repeated := restarted.SwitchPrepare(context.Background(), p)
			if repeated.Error == nil || repeated.Error.Kind != settings.SwitchErrorInterrupted {
				t.Fatalf("inherited prepare = %+v, want interrupted", repeated)
			}
			reconciled := restarted.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
			if reconciled.Error != nil || reconciled.Checkpoint.TargetCommitted {
				t.Fatalf("reconcile inherited envelope = %+v", reconciled)
			}
			switchStep(t, restarted, p, settings.SwitchStepRelease)
		})
	}
}

func TestSwitchPrepareRejectsDifferentManagerIncarnation(t *testing.T) {
	ex, p := switchFixture(t)
	snapshot := ex.SwitchPrepare(context.Background(), settings.SwitchPrepareRequest{})
	if snapshot.Error != nil {
		t.Fatal(snapshot.Error)
	}
	p.ConfigurationFence = snapshot.ConfigurationFence
	p.ManagerIncarnation = "previous-manager"
	result := ex.SwitchPrepare(context.Background(), p)
	if result.Error == nil || result.Error.Code != -32000 || result.Error.Kind != settings.SwitchErrorInterrupted || result.Checkpoint != nil {
		t.Fatalf("stale incarnation prepare = %+v", result)
	}
}

func TestSwitchRemoteOffVetoPrecedesHeldLifecycleLocks(t *testing.T) {
	for _, engine := range []string{"source", "target"} {
		t.Run(engine, func(t *testing.T) {
			ex, p := switchFixture(t)
			p.Origin = settings.SwitchOriginRemote
			p = prepareSwitch(t, ex, p)
			startCtx, done, err := ex.beginRemote(context.Background(), p.TargetEngine)
			if err != nil {
				t.Fatal(err)
			}
			defer done()
			st, err := ex.state(engine)
			if err != nil {
				t.Fatal(err)
			}
			ex.switchMu.Lock()
			defer ex.switchMu.Unlock()
			st.opMu.Lock()
			locked := true
			defer func() {
				if locked {
					st.opMu.Unlock()
				}
			}()
			stopped := make(chan error, 1)
			go func() { stopped <- ex.Stop(engine) }()
			select {
			case <-startCtx.Done():
			case <-time.After(2 * time.Second):
				t.Fatal("held lifecycle locks prevented startup veto")
			}
			data, err := os.ReadFile(ex.desired.path)
			if err != nil {
				t.Fatal(err)
			}
			var state desiredStateFile
			if err := json.Unmarshal(data, &state); err != nil {
				t.Fatal(err)
			}
			if on, known := state.Engines[engine]; !known || on || state.OffBy[engine] != intentLocal || state.PendingSwitch.Error == nil || state.PendingSwitch.Error.Code != codeLocalVeto {
				t.Fatalf("veto not durable before cleanup: %+v", state)
			}
			select {
			case err := <-stopped:
				t.Fatalf("stop returned before lifecycle lock release: %v", err)
			default:
			}
			st.opMu.Unlock()
			locked = false
			select {
			case err := <-stopped:
				if err != nil {
					t.Fatal(err)
				}
			case <-time.After(2 * time.Second):
				t.Fatal("stop did not finish after lifecycle lock release")
			}
		})
	}
}

func TestSwitchLocalOffDuringLocalReservationStaysBusy(t *testing.T) {
	for _, engine := range []string{"source", "target"} {
		t.Run(engine, func(t *testing.T) {
			ex, p := switchFixture(t)
			prepareSwitch(t, ex, p)
			var busy *switchBusyError
			if err := ex.Stop(engine); !errors.As(err, &busy) {
				t.Fatalf("local switch stop = %v, want busy", err)
			}
			requireOffOrigin(t, ex.desired, engine, "")
		})
	}
}

func TestSwitchStopStorageFailureDoesNotStopSource(t *testing.T) {
	ex, p := switchFixture(t)
	if err := ex.Start(context.Background(), "source"); err != nil {
		t.Fatal(err)
	}
	p = prepareSwitch(t, ex, p)
	useEngineFileReplacer(t, durablefile.Replacer{CreateTemp: func(string, string, os.FileMode) (durablefile.File, error) {
		return nil, errors.New("injected storage failure")
	}})
	result := ex.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: settings.SwitchStepSourceStop})
	if result.Error == nil || result.Error.Kind != settings.SwitchErrorStorage || result.Checkpoint.Phase != settings.SwitchCheckpointReserved {
		t.Fatalf("result = %+v", result)
	}
	status, err := ex.Status("source")
	if err != nil || !status.Running {
		t.Fatalf("source = %+v, %v", status, err)
	}
	on, known, err := ex.desired.get("source")
	if err != nil || !known || !on {
		t.Fatalf("source intent = %v/%v, %v", on, known, err)
	}
}

func TestSwitchPrepareRejectsForeignTargetPort(t *testing.T) {
	ex, p := switchFixture(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusForbidden) }))
	defer server.Close()
	st, err := ex.state("target")
	if err != nil {
		t.Fatal(err)
	}
	st.port = server.Listener.Addr().(*net.TCPAddr).Port
	snapshot := ex.SwitchPrepare(context.Background(), settings.SwitchPrepareRequest{})
	if snapshot.Error != nil {
		t.Fatal(snapshot.Error)
	}
	p.ConfigurationFence = snapshot.ConfigurationFence
	result := ex.SwitchPrepare(context.Background(), p)
	if result.Error == nil || result.Error.Kind != settings.SwitchErrorValidation {
		t.Fatalf("prepare = %+v", result)
	}
	if result.Checkpoint.Phase != settings.SwitchCheckpointReleased {
		t.Fatal("failed prepare retained reservation")
	}
}

func TestSwitchLocalOffBeforeTargetStartCancelsRemoteOperation(t *testing.T) {
	ex, p := switchFixture(t)
	p.Origin = settings.SwitchOriginRemote
	if err := ex.Start(context.Background(), "source"); err != nil {
		t.Fatal(err)
	}
	p = prepareSwitch(t, ex, p)
	if err := ex.Stop("source"); err != nil {
		t.Fatalf("local veto stop: %v", err)
	}
	result := ex.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: settings.SwitchStepSourceStop})
	if result.Error == nil || result.Error.Code != codeLocalVeto {
		t.Fatalf("step after local veto = %+v", result)
	}
	target, err := ex.Status("target")
	if err != nil || target.Running {
		t.Fatalf("target = %+v, %v", target, err)
	}
}

func TestSwitchReservationBusyKindSurvivesLifecycleRPC(t *testing.T) {
	ex, p := switchFixture(t)
	prepareSwitch(t, ex, p)
	var out bytes.Buffer
	manager := &Manager{exec: ex, codec: NewCodec(&out)}
	id := json.RawMessage(`1`)
	manager.runOp(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:start", Params: json.RawMessage(`{"engine":"source"}`)})
	var response struct {
		Error struct {
			Code int `json:"code"`
			Data struct {
				Kind string `json:"kind"`
			} `json:"data"`
		} `json:"error"`
	}
	if err := json.Unmarshal(out.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Error.Code != -32000 || response.Error.Data.Kind != "busy" {
		t.Fatalf("response = %+v", response)
	}
}

func TestSwitchStopCheckpointFailureReturnsCurrentObservation(t *testing.T) {
	ex, p := switchFixture(t)
	if err := ex.Start(context.Background(), "source"); err != nil {
		t.Fatal(err)
	}
	p = prepareSwitch(t, ex, p)
	writes := 0
	useEngineFileReplacer(t, durablefile.Replacer{CreateTemp: func(dir, base string, _ os.FileMode) (durablefile.File, error) {
		writes++
		if writes == 2 {
			return nil, errors.New("injected post-stop checkpoint failure")
		}
		return os.CreateTemp(dir, base+".*.tmp")
	}})
	result := ex.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: settings.SwitchStepSourceStop})
	if result.Error == nil || result.Error.Kind != settings.SwitchErrorStorage || result.Checkpoint.Phase != settings.SwitchCheckpointStopPending {
		t.Fatalf("result = %+v", result)
	}
	if result.Observed.Source.Running == nil || *result.Observed.Source.Running || !result.Observed.Source.TerminationVerified {
		t.Fatalf("source = %+v", result.Observed.Source)
	}
}
