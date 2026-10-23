// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
)

func TestSwitchReservedSourceVetoStopsBeforeDelayedScopedStop(t *testing.T) {
	ex, p := switchFixture(t)
	if err := ex.Start(context.Background(), "source"); err != nil {
		t.Fatalf("start source: %v", err)
	}
	p.Origin = settings.SwitchOriginRemote
	p = prepareSwitch(t, ex, p)
	// Broker admission holds the ordinary stop delivery. Drain fails before
	// any source-stop step; terminal cleanup delivers only the admitted veto.
	cleaned := ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{
		SwitchPrepareRequest: p, LocalVeto: &settings.SwitchLocalVeto{Engine: "source"},
	})
	if cleaned.Error == nil || cleaned.Error.Kind != settings.SwitchErrorRemoteVeto || cleaned.Checkpoint == nil || cleaned.Checkpoint.Phase != settings.SwitchCheckpointReserved || !cleaned.Observed.Source.TerminationVerified {
		t.Fatalf("pre-source-stop veto cleanup = %+v", cleaned)
	}
	switchStep(t, ex, p, settings.SwitchStepRelease)
	if err := ex.StopForSwitch("source", p.OperationID); err != nil {
		t.Fatalf("deliver held scoped stop: %v", err)
	}
	status, err := ex.Status("source")
	if err != nil || status.Running {
		t.Fatalf("source after release and delayed delivery = %+v, %v", status, err)
	}
	requireOffOrigin(t, ex.desired, "source", "local")
}

func TestSwitchRejectedPrepareOlderPeerStartLosesToBrokerOff(t *testing.T) {
	ex, p := switchFixture(t)
	if err := ex.desired.set("source", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	p.Origin = settings.SwitchOriginRemote
	snapshot := ex.SwitchPrepare(context.Background(), settings.SwitchPrepareRequest{})
	if snapshot.Error != nil {
		t.Fatal(snapshot.Error)
	}
	p.ConfigurationFence = snapshot.ConfigurationFence
	st, err := ex.state("source")
	if err != nil {
		t.Fatal(err)
	}
	st.opMu.Lock()
	locked := true
	defer func() {
		if locked {
			st.opMu.Unlock()
		}
	}()
	peerDone := make(chan struct{})
	response := httptest.NewRecorder()
	go func() {
		(&controlServer{exec: ex}).handleStart(response, httptest.NewRequest(http.MethodPost, "/v1/engines/start", strings.NewReader(`{"engine":"source"}`)))
		close(peerDone)
	}()
	deadline := time.Now().Add(5 * time.Second)
	for {
		ex.switchAdmission.Lock()
		admitted := ex.switchMutations["source"] > 0
		ex.switchAdmission.Unlock()
		if admitted {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("older peer start was not admitted")
		}
		time.Sleep(time.Millisecond)
	}
	// The broker has admitted Off now, but holds its scoped delivery. Prepare
	// rejects while the older authenticated peer path still owns a mutation.
	rejected := ex.SwitchPrepare(context.Background(), p)
	if rejected.Error == nil || rejected.Error.Kind != settings.SwitchErrorBusy || rejected.Checkpoint != nil {
		t.Fatalf("busy prepare: %+v", rejected)
	}
	st.opMu.Unlock()
	locked = false
	select {
	case <-peerDone:
	case <-time.After(10 * time.Second):
		t.Fatal("older peer start did not finish")
	}
	if response.Code != http.StatusOK {
		t.Fatalf("older peer start: HTTP %d", response.Code)
	}
	if err := ex.Stop("source"); err != nil {
		t.Fatalf("broker discharge: %v", err)
	}
	if observed, err := ex.verifyStopped("source"); err != nil || !observed.TerminationVerified || observed.Intent != settings.SwitchIntentOff {
		t.Fatalf("discharge: %+v, %v", observed, err)
	}
	if err := ex.StopForSwitch("source", p.OperationID); err != nil {
		t.Fatal(err)
	}
	if err := ex.RestoreEnabled(context.Background()); err != nil {
		t.Fatal(err)
	}
	if status, err := ex.Status("source"); err != nil || status.Running {
		t.Fatalf("source restored after Off: %+v %v", status, err)
	}
}

func TestSwitchHistoricalLocalOffAdoptedSourceIsObservationOnly(t *testing.T) {
	ex, p := switchFixture(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))
	t.Cleanup(server.Close)
	st, err := ex.state("source")
	if err != nil {
		t.Fatal(err)
	}
	stopMarker := filepath.Join(t.TempDir(), "unexpected-stop")
	st.port = server.Listener.Addr().(*net.TCPAddr).Port
	st.plat.Runtime.Mode = "command"
	st.plat.Runtime.Stop = &StopSpec{Cmd: []string{fakeEngineBin, "touch", stopMarker}}
	if err := ex.desired.set("source", false, intentLocal); err != nil {
		t.Fatal(err)
	}
	p.Origin = settings.SwitchOriginRemote
	p = prepareSwitch(t, ex, p)
	status, err := ex.Status("source")
	if err != nil || !status.Running {
		t.Fatalf("adopted source: %+v, %v", status, err)
	}
	// Failed drain sends no forward source-stop and admitted no new local veto.
	cleaned := ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
	if cleaned.Error != nil || cleaned.Observed.Source.Running == nil || !*cleaned.Observed.Source.Running || cleaned.Observed.Source.Intent != settings.SwitchIntentOff {
		t.Fatalf("historical Off cleanup: %+v", cleaned)
	}
	if _, err := os.Stat(stopMarker); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("stop CLI ran without operation veto: %v", err)
	}
	switchStep(t, ex, p, settings.SwitchStepRelease)
	// Avoid the fixture's shutdown running the intentionally forbidden CLI.
	server.Close()
	ex.markStopped(st, "source")
}

func TestSwitchRejectedPrepareDelayedScopedStop(t *testing.T) {
	for _, scenario := range []string{"broker not yet discharged", "later local On", "later reservation"} {
		t.Run(scenario, func(t *testing.T) {
			ex, p := switchFixture(t)
			snapshot := ex.SwitchPrepare(context.Background(), settings.SwitchPrepareRequest{})
			if snapshot.Error != nil {
				t.Fatalf("configuration snapshot: %+v", snapshot.Error)
			}
			p.ConfigurationFence = snapshot.ConfigurationFence
			if err := ex.Start(context.Background(), "source"); err != nil {
				t.Fatalf("start source: %v", err)
			}
			p.Origin = settings.SwitchOriginRemote
			rejected := ex.SwitchPrepare(context.Background(), p)
			if rejected.Error == nil || rejected.Error.Kind != settings.SwitchErrorConfigurationConflict || rejected.Checkpoint != nil {
				t.Fatalf("prepare must reject before checkpoint = %+v", rejected)
			}
			if scenario == "later local On" {
				// Reasserting On is newer work even though the saved boolean
				// and its provenance do not change.
				if err := ex.Start(context.Background(), "source"); err != nil {
					t.Fatalf("later local On: %v", err)
				}
			}
			var b settings.SwitchPrepareRequest
			if scenario == "later reservation" {
				b = p
				b.OperationID = "later-switch"
				b = prepareSwitch(t, ex, b)
			}
			if err := ex.StopForSwitch("source", p.OperationID); err != nil {
				t.Fatalf("delayed scoped stop: %v", err)
			}
			wantRunning := true
			status, err := ex.Status("source")
			if err != nil || status.Running != wantRunning {
				t.Fatalf("source status = %+v, %v; want running %v", status, err, wantRunning)
			}
			on, known, err := ex.desired.get("source")
			if err != nil || !known || on != wantRunning {
				t.Fatalf("source intent = %v, %v, %v; want On %v", on, known, err, wantRunning)
			}
			if scenario == "later reservation" {
				switchStep(t, ex, b, settings.SwitchStepSourceStop)
				switchStep(t, ex, b, settings.SwitchStepTargetStart)
				switchStep(t, ex, b, settings.SwitchStepRelease)
			}
		})
	}
}

func TestSwitchRejectedPrepareRequiresBrokerDischarge(t *testing.T) {
	ex, p := switchFixture(t)
	snapshot := ex.SwitchPrepare(context.Background(), settings.SwitchPrepareRequest{})
	if snapshot.Error != nil {
		t.Fatalf("configuration snapshot: %+v", snapshot.Error)
	}
	p.ConfigurationFence = snapshot.ConfigurationFence
	if err := ex.Start(context.Background(), "source"); err != nil {
		t.Fatalf("start source: %v", err)
	}
	p.Origin = settings.SwitchOriginRemote
	rejected := ex.SwitchPrepare(context.Background(), p)
	if rejected.Error == nil || rejected.Error.Kind != settings.SwitchErrorConfigurationConflict || rejected.Checkpoint != nil {
		t.Fatalf("prepare rejection = %+v", rejected)
	}
	cleaned := ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{
		SwitchPrepareRequest: p, LocalVeto: &settings.SwitchLocalVeto{Engine: "source"},
	})
	if cleaned.Error == nil || cleaned.Error.Kind != settings.SwitchErrorInvalidRequest || cleaned.Checkpoint != nil {
		t.Fatalf("rejected prepare Off cleanup = %+v", cleaned)
	}
	if err := ex.Stop("source"); err != nil {
		t.Fatalf("broker ordinary discharge: %v", err)
	}
	if observed, err := ex.verifyStopped("source"); err != nil || !observed.TerminationVerified || observed.Intent != settings.SwitchIntentOff {
		t.Fatalf("verified discharge = %+v, %v", observed, err)
	}
	if err := ex.StopForSwitch("source", p.OperationID); err != nil {
		t.Fatalf("held stop after cleanup: %v", err)
	}
}

func TestSwitchRecoveryReplacementWithoutCheckpointDischargesBeforeRestore(t *testing.T) {
	ex, _ := switchFixture(t)
	if err := ex.desired.set("source", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	// Rejection allocated no checkpoint; the broker's journal is the only owner.
	replacement := NewExecutor(ex.reg, NewReporter(nil), nil, filepath.Dir(ex.desired.path))
	t.Cleanup(replacement.StopAll)
	replacement.nodeProjection.set(nodeProjection{ManagementBlocked: false})
	snapshot := replacement.SwitchPrepare(context.Background(), settings.SwitchPrepareRequest{})
	if snapshot.Error != nil || snapshot.Checkpoint != nil {
		t.Fatalf("replacement inspection: %+v", snapshot)
	}
	if err := replacement.Stop("source"); err != nil {
		t.Fatal(err)
	}
	if result, err := replacement.verifyStopped("source"); err != nil || !result.TerminationVerified || result.Intent != settings.SwitchIntentOff {
		t.Fatalf("replacement discharge: %+v %v", result, err)
	}
	if err := replacement.RestoreEnabled(context.Background()); err != nil {
		t.Fatal(err)
	}
	if status, err := replacement.Status("source"); err != nil || status.Running {
		t.Fatalf("replacement restored source: %+v %v", status, err)
	}
}

func TestSwitchReservedSourceVetoRetainsUnverifiedCleanup(t *testing.T) {
	ex, p := switchFixture(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
	}))
	t.Cleanup(server.Close)
	st, err := ex.state("source")
	if err != nil {
		t.Fatal(err)
	}
	st.port = server.Listener.Addr().(*net.TCPAddr).Port
	p.Origin = settings.SwitchOriginRemote
	p = prepareSwitch(t, ex, p)
	cleaned := ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{
		SwitchPrepareRequest: p, LocalVeto: &settings.SwitchLocalVeto{Engine: "source"},
	})
	if cleaned.Error == nil || cleaned.Error.Kind != settings.SwitchErrorInterrupted || cleaned.Checkpoint == nil || cleaned.Checkpoint.Phase == settings.SwitchCheckpointReleased || cleaned.Observed.Source.TerminationVerified {
		t.Fatalf("unverified source cleanup = %+v", cleaned)
	}
	released := ex.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: settings.SwitchStepRelease})
	if released.Error == nil || released.Error.Kind != settings.SwitchErrorBusy || released.Checkpoint == nil || released.Checkpoint.Phase == settings.SwitchCheckpointReleased {
		t.Fatalf("release without verified Off = %+v", released)
	}
}

func TestSwitchProvisionalPrepareRefusalRetainsSourceVetoCleanup(t *testing.T) {
	ex, p := switchFixture(t)
	if err := ex.Start(context.Background(), "source"); err != nil {
		t.Fatalf("start source: %v", err)
	}
	target, err := ex.state("target")
	if err != nil {
		t.Fatal(err)
	}
	target.plat.Runtime.Ready = nil
	p.Origin = settings.SwitchOriginRemote
	snapshot := ex.SwitchPrepare(context.Background(), settings.SwitchPrepareRequest{})
	if snapshot.Error != nil {
		t.Fatalf("configuration snapshot: %+v", snapshot.Error)
	}
	p.ConfigurationFence = snapshot.ConfigurationFence
	rejected := ex.SwitchPrepare(context.Background(), p)
	if rejected.Error == nil || rejected.Error.Kind != settings.SwitchErrorValidation || rejected.Checkpoint == nil || rejected.Checkpoint.Phase != settings.SwitchCheckpointReserved {
		t.Fatalf("provisional prepare rejection = %+v", rejected)
	}
	cleaned := ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{
		SwitchPrepareRequest: p, LocalVeto: &settings.SwitchLocalVeto{Engine: "source"},
	})
	if cleaned.Error == nil || cleaned.Error.Kind != settings.SwitchErrorRemoteVeto || !cleaned.Observed.Source.TerminationVerified {
		t.Fatalf("provisional rejection source cleanup = %+v", cleaned)
	}
	switchStep(t, ex, p, settings.SwitchStepRelease)
}

func TestSwitchRemoteVetoAfterTargetCommitCleansTarget(t *testing.T) {
	for _, scenario := range []string{"source Off", "target Off", "local pause"} {
		t.Run(scenario, func(t *testing.T) {
			ex, p := switchFixture(t)
			p.Origin = settings.SwitchOriginRemote
			p = prepareSwitch(t, ex, p)
			switchStep(t, ex, p, settings.SwitchStepSourceStop)
			switchStep(t, ex, p, settings.SwitchStepTargetStart)
			switch scenario {
			case "source Off":
				if err := ex.StopForSwitch("source", p.OperationID); err != nil {
					t.Fatal(err)
				}
			case "target Off":
				if err := ex.StopForSwitch("target", p.OperationID); err != nil {
					t.Fatal(err)
				}
			case "local pause":
				ex.nodeProjection.set(nodeProjection{PausedLocally: true, ManagementBlocked: true})
			}
			result := ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
			if result.Error == nil || result.Error.Kind != settings.SwitchErrorRemoteVeto || result.Checkpoint == nil || result.Checkpoint.TargetCommitted || !result.Observed.Target.TerminationVerified || result.Observed.Target.Intent != settings.SwitchIntentOff {
				t.Fatalf("committed target veto cleanup = %+v", result)
			}
			for _, engine := range []string{"source", "target"} {
				on, known, err := ex.desired.get(engine)
				if err != nil || !known || on {
					t.Fatalf("%s durable intent: on %v, known %v, error %v", engine, on, known, err)
				}
				status, err := ex.Status(engine)
				if err != nil || status.Running {
					t.Fatalf("%s after cleanup: %+v, %v", engine, status, err)
				}
			}
			switchStep(t, ex, p, settings.SwitchStepRelease)
		})
	}
}

func TestSwitchDelayedScopedStopPreservesLaterReservation(t *testing.T) {
	for _, engine := range []string{"source", "target"} {
		t.Run(engine, func(t *testing.T) {
			ex, p := switchFixture(t)
			p.Origin = settings.SwitchOriginRemote
			p = prepareSwitch(t, ex, p)
			switchStep(t, ex, p, settings.SwitchStepSourceStop)
			switchStep(t, ex, p, settings.SwitchStepTargetStart)

			// The broker admitted Off against A, but its stop request is held
			// before manager admission while reconcile delivers the veto itself.
			ctx, cancel := context.WithCancel(context.Background())
			t.Cleanup(cancel)
			deliver := make(chan struct{})
			stopped := make(chan error, 1)
			go func() {
				select {
				case <-deliver:
					stopped <- ex.StopForSwitch(engine, p.OperationID)
				case <-ctx.Done():
				}
			}()
			cleaned := ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p, LocalVeto: &settings.SwitchLocalVeto{Engine: engine}})
			if cleaned.Error == nil || cleaned.Error.Kind != settings.SwitchErrorRemoteVeto || !cleaned.Observed.Target.TerminationVerified {
				t.Fatalf("A veto cleanup = %+v", cleaned)
			}
			switchStep(t, ex, p, settings.SwitchStepRelease)
			if err := ex.Start(context.Background(), engine); err != nil {
				t.Fatalf("later local On: %v", err)
			}
			b := p
			b.OperationID = "switch-later"
			b.SourceEngine = engine
			b.TargetEngine = "other"
			b = prepareSwitch(t, ex, b)
			close(deliver)
			select {
			case err := <-stopped:
				if err != nil {
					t.Fatalf("delayed A stop: %v", err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("delayed A stop did not return")
			}
			ex.desired.mu.Lock()
			state, err := ex.desired.load()
			ex.desired.mu.Unlock()
			if err != nil {
				t.Fatalf("load B reservation: %v", err)
			}
			if cp := state.PendingSwitch; cp == nil || cp.OperationID != b.OperationID || cp.Error != nil {
				t.Fatalf("B reservation after delayed A stop = %+v", cp)
			}
			status, err := ex.Status(engine)
			if err != nil || !status.Running {
				t.Fatalf("later source after delayed A stop = %+v, %v", status, err)
			}
			switchStep(t, ex, b, settings.SwitchStepSourceStop)
			switchStep(t, ex, b, settings.SwitchStepTargetStart)
			switchStep(t, ex, b, settings.SwitchStepRelease)
		})
	}
}

func TestSwitchDelayedScopedStopPreservesLaterLocalOn(t *testing.T) {
	for _, engine := range []string{"source", "target"} {
		t.Run(engine, func(t *testing.T) {
			ex, p := switchFixture(t)
			p.Origin = settings.SwitchOriginRemote
			p = prepareSwitch(t, ex, p)
			switchStep(t, ex, p, settings.SwitchStepRelease)
			if err := ex.Start(context.Background(), engine); err != nil {
				t.Fatalf("later local On: %v", err)
			}
			if err := ex.StopForSwitch(engine, p.OperationID); err != nil {
				t.Fatalf("delayed scoped stop: %v", err)
			}
			on, known, err := ex.desired.get(engine)
			if err != nil || !known || !on {
				t.Fatalf("later local intent: on %v, known %v, error %v", on, known, err)
			}
			status, err := ex.Status(engine)
			if err != nil || !status.Running {
				t.Fatalf("later local engine = %+v, %v", status, err)
			}
		})
	}
}

func TestSwitchReconcileDeliversLatchedVetoBeforeLocalStop(t *testing.T) {
	for _, scenario := range []struct {
		name string
		veto settings.SwitchLocalVeto
	}{
		{"source Off", settings.SwitchLocalVeto{Engine: "source"}},
		{"target Off", settings.SwitchLocalVeto{Engine: "target"}},
		{"local pause", settings.SwitchLocalVeto{Pause: true}},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			ex, p := switchFixture(t)
			p.Origin = settings.SwitchOriginRemote
			p = prepareSwitch(t, ex, p)
			switchStep(t, ex, p, settings.SwitchStepSourceStop)
			switchStep(t, ex, p, settings.SwitchStepTargetStart)

			// No Stop or pause projection has reached the manager yet.
			result := ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p, LocalVeto: &scenario.veto})
			if result.Error == nil || result.Error.Kind != settings.SwitchErrorRemoteVeto || result.Checkpoint == nil || result.Checkpoint.TargetCommitted || !result.Observed.Target.TerminationVerified || result.Observed.Target.Intent != settings.SwitchIntentOff {
				t.Fatalf("delivered veto cleanup = %+v", result)
			}
			if scenario.veto.Engine != "" {
				requireOffOrigin(t, ex.desired, scenario.veto.Engine, "local")
			}
			switchStep(t, ex, p, settings.SwitchStepRelease)
			if scenario.veto.Engine != "" {
				if err := ex.Stop(scenario.veto.Engine); err != nil {
					t.Fatalf("late ordinary local stop: %v", err)
				}
			}
			for _, engine := range []string{"source", "target"} {
				on, known, err := ex.desired.get(engine)
				if err != nil || !known || on {
					t.Fatalf("%s durable intent: on %v, known %v, error %v", engine, on, known, err)
				}
				status, err := ex.Status(engine)
				if err != nil || status.Running {
					t.Fatalf("%s after cleanup: %+v, %v", engine, status, err)
				}
			}
		})
	}
}

func TestSwitchReleaseRetainsVetoRecordedAfterReconcileRead(t *testing.T) {
	for _, engine := range []string{"source", "target"} {
		t.Run(engine, func(t *testing.T) {
			ex, p := switchFixture(t)
			p.Origin = settings.SwitchOriginRemote
			p = prepareSwitch(t, ex, p)
			switchStep(t, ex, p, settings.SwitchStepSourceStop)
			switchStep(t, ex, p, settings.SwitchStepTargetStart)
			beforeVeto := ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
			if beforeVeto.Error != nil || beforeVeto.Checkpoint == nil || !beforeVeto.Checkpoint.TargetCommitted {
				t.Fatalf("pre-veto reconcile = %+v", beforeVeto)
			}

			// Deliver only the durable admission, leaving the physical local stop
			// outstanding. Release must not trust the earlier reconcile's copy.
			done, err := ex.admitStop(engine, intentLocal, "")
			if err != nil {
				t.Fatal(err)
			}
			defer done()
			release := ex.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: settings.SwitchStepRelease})
			if release.Error == nil || release.Error.Kind != settings.SwitchErrorBusy || release.Checkpoint == nil || release.Checkpoint.Phase == settings.SwitchCheckpointReleased {
				t.Fatalf("release with uncleaned veto = %+v", release)
			}
			var busy *switchBusyError
			if _, err := ex.admitMutation("target"); !errors.As(err, &busy) {
				t.Fatalf("target recovery reservation = %v, want busy", err)
			}
			cleaned := ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
			if cleaned.Error == nil || cleaned.Error.Kind != settings.SwitchErrorRemoteVeto || !cleaned.Observed.Target.TerminationVerified || cleaned.Observed.Target.Intent != settings.SwitchIntentOff {
				t.Fatalf("retained veto cleanup = %+v", cleaned)
			}
			switchStep(t, ex, p, settings.SwitchStepRelease)
		})
	}
}
