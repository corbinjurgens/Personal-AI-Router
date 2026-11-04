// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"testing"

	settings "nvpair-shared/enginesettings"
)

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
				if err := ex.Stop("source"); err != nil {
					t.Fatal(err)
				}
			case "target Off":
				if err := ex.Stop("target"); err != nil {
					t.Fatal(err)
				}
			case "local pause":
				ex.nodeProjection.set(nodeProjection{PausedLocally: true})
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
				off, err := ex.desired.offLocally(scenario.veto.Engine)
				if err != nil || !off {
					t.Fatalf("veto engine local Off = %v, %v", off, err)
				}
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
			done, err := ex.admitStop(engine, intentLocal)
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
