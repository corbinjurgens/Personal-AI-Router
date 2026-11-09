// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"strings"
	"testing"

	settings "nvpair-shared/enginesettings"
)

// V44: an apply starts a stopped engine only when its restore refused the stale
// approval that apply replaced, and only while the node is available.
func TestReapprovalStartsOnlyARefusedRestore(t *testing.T) {
	apply := func(t *testing.T, e *Executor) {
		t.Helper()
		request := settingsRequest(t, e)
		request.Settings.LaunchText = strings.TrimSpace(request.Settings.LaunchText + " --synthetic-edit")
		preview := previewSettings(t, e, request)
		if len(preview.Errors) != 0 {
			t.Fatalf("preview errors: %v", preview.Errors)
		}
		if _, err := e.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { return nil }); err != nil {
			t.Fatalf("ConfigureLaunch: %v", err)
		}
	}
	running := func(st *engineState) (bool, bool) {
		st.mu.Lock()
		defer st.mu.Unlock()
		return st.running, st.restoreApprovalRefused
	}
	for _, tc := range []struct {
		name        string
		setup       func(*testing.T, *Executor, *engineState)
		wantRunning bool
		wantRefused bool
	}{
		{
			name:  "ordinary apply on a stopped engine",
			setup: func(*testing.T, *Executor, *engineState) {},
		},
		{
			name:        "refused restore",
			setup:       func(_ *testing.T, _ *Executor, st *engineState) { st.restoreApprovalRefused = true },
			wantRunning: true,
		},
		{
			name: "refused restore on a paused node",
			setup: func(_ *testing.T, e *Executor, st *engineState) {
				st.restoreApprovalRefused = true
				e.nodeProjection.set(nodeProjection{PausedLocally: true, ManagementBlocked: true})
			},
			wantRefused: true,
		},
		{
			name: "stopped on purpose after a refused restore",
			setup: func(t *testing.T, e *Executor, st *engineState) {
				st.restoreApprovalRefused = true
				if err := e.Sleep("fake"); err != nil {
					t.Fatalf("Sleep: %v", err)
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := settingsExecutor(t, false)
			st := settingsState(t, e)
			tc.setup(t, e, st)
			apply(t, e)
			gotRunning, gotRefused := running(st)
			if gotRunning != tc.wantRunning || gotRefused != tc.wantRefused {
				t.Fatalf("running=%t refused=%t, want running=%t refused=%t", gotRunning, gotRefused, tc.wantRunning, tc.wantRefused)
			}
		})
	}
}

func TestStaleApprovalSurvivesWrapping(t *testing.T) {
	stale := staleApproval(servingError("Native source conflict; obtain a fresh preview before applying."))
	for _, tc := range []struct {
		name string
		err  error
		want bool
	}{
		{"marked", stale, true},
		{"wrapped as a start refuses", fmt.Errorf("NE preset validation: %w", stale), true},
		{"another refusal", servingError("Native launch configuration failed."), false},
		{"no refusal", nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := isStaleApproval(tc.err); got != tc.want {
				t.Fatalf("isStaleApproval = %t, want %t", got, tc.want)
			}
		})
	}
}

// The refusal a frozen Auto load gets at start counts as a stale approval.
func TestAutoBuildChangeRefusesRestoreAsStaleApproval(t *testing.T) {
	f, frozen, _ := autoTestFrozenLaunch(t)
	if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: frozen.Settings, OperationID: "auto"}, func() error { return nil }); err != nil {
		t.Fatalf("ConfigureLaunch: %v", err)
	}
	writeTestFile(t, f.st.binPath, []byte("second synthetic Auto native build"))
	if err := f.ex.RestoreInstalled(context.Background()); err == nil || !strings.Contains(err.Error(), "explicitly Recalculate") {
		t.Fatalf("restore: %v", err)
	}
	if !f.st.restoreApprovalRefused {
		t.Fatal("the frozen Auto refusal did not count as a stale approval")
	}
}
