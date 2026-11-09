// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
)

// reapprovalFixture holds an approved, stopped engine with its approval's source.
type reapprovalFixture struct {
	llamaServingFixture
	st       *engineState
	original []byte
	mtime    time.Time
}

func newReapprovalFixture(t *testing.T) reapprovalFixture {
	t.Helper()
	f := newLlamaServingFixture(t)
	applyLlamaServing(t, f, approveLlamaServing(t, f))
	original, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(f.source)
	if err != nil {
		t.Fatal(err)
	}
	return reapprovalFixture{llamaServingFixture: f, st: settingsState(t, f.ex), original: original, mtime: info.ModTime()}
}

// recoverInputs puts the source back as approved: its time is part of the digest.
func (f reapprovalFixture) recoverInputs(t *testing.T) {
	t.Helper()
	writeServingFixture(t, f.source, f.original)
	if err := os.Chtimes(f.source, f.mtime, f.mtime); err != nil {
		t.Fatal(err)
	}
	f.wantReadback(t, "unverified")
}

// refuseRestore moves the approved source and runs a real restore, which refuses.
func (f reapprovalFixture) refuseRestore(t *testing.T) {
	t.Helper()
	writeServingFixture(t, f.source, append(bytes.Clone(f.original), []byte("# changed after approval\n")...))
	if err := f.ex.restoreInstalled(context.Background(), "fake"); !isStaleApproval(err) {
		t.Fatalf("restore: %v", err)
	}
	f.wantReadback(t, "mismatch")
}

func (f reapprovalFixture) wantReadback(t *testing.T, want string) {
	t.Helper()
	state, err := f.ex.LaunchSettings("fake")
	if err != nil {
		t.Fatal(err)
	}
	if state.ServingReadback == nil || state.ServingReadback.Status != want {
		t.Fatalf("readback = %+v, want %s", state.ServingReadback, want)
	}
}

// apply reviews the same settings again, with no saved digest to conflict, and
// applies the result, as both clients do.
func (f reapprovalFixture) apply(t *testing.T) error {
	t.Helper()
	preview := previewSettings(t, f.ex, f.request)
	if len(preview.Errors) != 0 {
		t.Fatalf("preview errors: %v", preview.Errors)
	}
	_, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { return nil })
	return err
}

func (f reapprovalFixture) want(t *testing.T, wantRunning, wantRefused bool) {
	t.Helper()
	f.st.mu.Lock()
	defer f.st.mu.Unlock()
	if f.st.running != wantRunning || f.st.restoreApprovalRefused != wantRefused {
		t.Fatalf("running=%t refused=%t, want running=%t refused=%t", f.st.running, f.st.restoreApprovalRefused, wantRunning, wantRefused)
	}
}

// V44: an apply starts a stopped engine only when it replaces a stale approval
// that a restore or crash relaunch refused, and only while the node is available.
func TestReapprovalStartsOnlyARefusedRestore(t *testing.T) {
	for _, tc := range []struct {
		name        string
		setup       func(*testing.T, reapprovalFixture)
		wantRunning bool
		wantRefused bool
	}{
		{
			name:  "ordinary apply on a stopped engine",
			setup: func(*testing.T, reapprovalFixture) {},
		},
		{
			name:        "refused restore",
			setup:       func(t *testing.T, f reapprovalFixture) { f.refuseRestore(t) },
			wantRunning: true,
		},
		{
			name: "refused restore on a paused node",
			setup: func(t *testing.T, f reapprovalFixture) {
				f.refuseRestore(t)
				f.ex.nodeProjection.set(nodeProjection{PausedLocally: true, ManagementBlocked: true})
			},
			wantRefused: true,
		},
		{
			name: "stopped on purpose after a refused restore",
			setup: func(t *testing.T, f reapprovalFixture) {
				f.refuseRestore(t)
				if err := f.ex.Sleep("fake"); err != nil {
					t.Fatalf("Sleep: %v", err)
				}
			},
		},
		{
			name: "inputs recovered after a refused restore",
			setup: func(t *testing.T, f reapprovalFixture) {
				f.refuseRestore(t)
				f.recoverInputs(t)
			},
			wantRefused: true,
		},
		{
			name: "stale again after the inputs recovered",
			setup: func(t *testing.T, f reapprovalFixture) {
				f.refuseRestore(t)
				f.recoverInputs(t)
				if err := f.apply(t); err != nil {
					t.Fatalf("apply: %v", err)
				}
				f.want(t, false, true)
				writeServingFixture(t, f.source, append(bytes.Clone(f.original), []byte("# changed again\n")...))
				f.wantReadback(t, "mismatch")
			},
			wantRunning: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newReapprovalFixture(t)
			tc.setup(t, f)
			if err := f.apply(t); err != nil {
				t.Fatalf("apply: %v", err)
			}
			f.want(t, tc.wantRunning, tc.wantRefused)
		})
	}
}

// A failed start keeps the record, before spawn as here too. That apply saved its
// approval, so nothing is left for a later apply to replace.
func TestReapprovalFailedStartIsNotRetriedByALaterApply(t *testing.T) {
	f := newReapprovalFixture(t)
	f.refuseRestore(t)
	f.st.mu.Lock()
	port := f.st.plat.Runtime.Port
	f.st.mu.Unlock()
	listener, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	if err := f.apply(t); err == nil {
		t.Fatal("the start succeeded on an occupied port")
	}
	f.want(t, false, true)
	f.wantReadback(t, "unverified")
	listener.Close()
	if err := f.apply(t); err != nil {
		t.Fatalf("later apply: %v", err)
	}
	f.want(t, false, true)
}

// Strata reads its conflict through the same readback.
func TestReapprovalStartsARefusedStrataRestore(t *testing.T) {
	f := newStrataServingFixture(t)
	applyLlamaServing(t, f.llamaServingFixture, approveLlamaServing(t, f.llamaServingFixture))
	st := settingsState(t, f.ex)
	st.mu.Lock()
	st.plat.Runtime.Serving.Models[0].ExpectedNativeSourceDigest = strings.Repeat("b", 64)
	st.mu.Unlock()
	if err := f.ex.restoreInstalled(context.Background(), "fake"); !isStaleApproval(err) {
		t.Fatalf("restore: %v", err)
	}
	preview := previewSettings(t, f.ex, f.request)
	if len(preview.Errors) != 0 {
		t.Fatalf("preview errors: %v", preview.Errors)
	}
	if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { return nil }); err != nil {
		t.Fatalf("ConfigureLaunch: %v", err)
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if !st.running {
		t.Fatal("the re-approval did not start the engine")
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
	state, err := f.ex.LaunchSettings("fake")
	if err != nil {
		t.Fatal(err)
	}
	if state.ServingReadback == nil || state.ServingReadback.Status != "mismatch" {
		t.Fatalf("readback = %+v: a re-approval would not count", state.ServingReadback)
	}
}
