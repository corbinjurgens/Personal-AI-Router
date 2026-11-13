// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"os"
	"sync"
	"testing"
)

// stageAndSwap runs a refresh up to the caller's start attempt, holding the
// same locks restore and install hold.
func stageAndSwap(t *testing.T, ex *Executor, st *engineState, engine string) refreshFinish {
	t.Helper()
	st.refreshMu.Lock()
	defer st.refreshMu.Unlock()
	if !ex.stageManagedRefresh(context.Background(), st, engine) {
		t.Fatal("staging the new build failed")
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	finish := ex.swapManagedRefresh(context.Background(), st, engine)
	if finish == nil {
		t.Fatal("swapping the new build failed")
	}
	return finish
}

// Audit finding 1: approvals are bound to the native build, so after a real
// pin move the first start refuses the stale approval. That must keep the new
// build for the reviewed apply to re-approve, not roll back on every start.
func TestRefreshKeepsNewBuildWhenStartRefusesStaleApproval(t *testing.T) {
	server := newBuildServer(t)
	const engine = "refresh-stale-approval"
	ex := newTestExecutor(t, managedBuildManifest(t, engine, server.pin("b1")))
	t.Cleanup(ex.StopAll)
	if err := ex.Install(context.Background(), engine); err != nil {
		t.Fatalf("first install: %v", err)
	}
	st := setPin(t, ex, engine, server.pin("b2"))
	finish := stageAndSwap(t, ex, st, engine)

	refusal := &ServingFieldError{Field: "modelLoads", Reason: "approval is stale", StaleApproval: true}
	restarts := 0
	st.opMu.Lock()
	err := finish(refusal, func() error { restarts++; return nil })
	st.opMu.Unlock()
	if !isStaleApproval(err) || restarts != 0 {
		t.Fatalf("finish = %v with %d restarts, want the stale-approval refusal and no rollback start", err, restarts)
	}
	if got := installedBuild(t, st); got != "b2" {
		t.Fatalf("installed build = %q, want the new build b2", got)
	}
	if marker, _ := readInstallMarker(st.installDir); marker.Recipe != installRecipeID(server.pin("b2")) {
		t.Fatalf("marker recipe = %q, want the new recipe", marker.Recipe)
	}
	assertNoRefreshLeftovers(t, ex, engine)
	if ex.managedInstallStale(st) {
		t.Fatal("the install is still stale after a stale-approval start; the next start would refresh again")
	}
}

// Audit finding 2: concurrent refreshes of one engine must not share or wipe
// one staging directory, and only one download of the new build happens.
func TestConcurrentRestoreAndInstallRefreshOnce(t *testing.T) {
	server := newBuildServer(t)
	const engine = "refresh-concurrent"
	ex := newTestExecutor(t, managedBuildManifest(t, engine, server.pin("b1")))
	t.Cleanup(ex.StopAll)
	if err := ex.Install(context.Background(), engine); err != nil {
		t.Fatalf("first install: %v", err)
	}
	st := setPin(t, ex, engine, server.pin("b2"))
	before := server.downloads.Load()

	var wg sync.WaitGroup
	errs := make(chan error, 4)
	for i := 0; i < 2; i++ {
		wg.Add(2)
		go func() { defer wg.Done(); errs <- ex.RestoreInstalled(context.Background()) }()
		go func() { defer wg.Done(); errs <- ex.Install(context.Background(), engine) }()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent refresh: %v", err)
		}
	}
	if got := installedBuild(t, st); got != "b2" {
		t.Fatalf("installed build = %q, want b2", got)
	}
	if downloads := server.downloads.Load() - before; downloads != 1 {
		t.Fatalf("new build downloaded %d times, want once", downloads)
	}
	assertRunningFrom(t, ex, st, engine)
	assertNoRefreshLeftovers(t, ex, engine)
}

// Audit finding 3: a rollback whose previous build is missing keeps the new
// build in place instead of leaving no install.
func TestRollbackWithoutPreviousBuildKeepsTheNewBuild(t *testing.T) {
	server := newBuildServer(t)
	const engine = "refresh-no-previous"
	ex := newTestExecutor(t, managedBuildManifest(t, engine, server.pin("b1")))
	t.Cleanup(ex.StopAll)
	if err := ex.Install(context.Background(), engine); err != nil {
		t.Fatalf("first install: %v", err)
	}
	st := setPin(t, ex, engine, server.pin("b2"))
	finish := stageAndSwap(t, ex, st, engine)
	_, previous := ex.refreshDirs(engine)
	if err := os.RemoveAll(previous); err != nil {
		t.Fatal(err)
	}
	startErr := errors.New("injected start failure")
	st.opMu.Lock()
	err := finish(startErr, func() error { t.Fatal("rollback restarted without an old build"); return nil })
	st.opMu.Unlock()
	if !errors.Is(err, startErr) {
		t.Fatalf("finish = %v, want the start failure", err)
	}
	if got := installedBuild(t, st); got != "b2" {
		t.Fatalf("installed build = %q, want the new build kept", got)
	}
	assertNoRefreshLeftovers(t, ex, engine)
}

// Audit finding 4: leftovers from a crash beside an existing install are
// removed before the next refresh check.
func TestRefreshLeftoversBesideAnInstallAreRemoved(t *testing.T) {
	server := newBuildServer(t)
	const engine = "refresh-leftovers"
	ex := newTestExecutor(t, managedBuildManifest(t, engine, server.pin("b1")))
	t.Cleanup(ex.StopAll)
	if err := ex.Install(context.Background(), engine); err != nil {
		t.Fatalf("first install: %v", err)
	}
	staging, previous := ex.refreshDirs(engine)
	for _, dir := range []string{staging, previous} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := ex.RestoreInstalled(context.Background()); err != nil {
		t.Fatal(err)
	}
	st, _ := ex.state(engine)
	if got := installedBuild(t, st); got != "b1" {
		t.Fatalf("installed build = %q, want b1", got)
	}
	assertNoRefreshLeftovers(t, ex, engine)
}
