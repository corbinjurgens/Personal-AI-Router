// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
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
