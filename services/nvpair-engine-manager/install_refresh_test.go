// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/testbin"
)

// buildServer serves one payload per build label at /<label>.zip.
type buildServer struct {
	*httptest.Server
	downloads atomic.Int32
}

func newBuildServer(t *testing.T) *buildServer {
	t.Helper()
	s := &buildServer{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		label := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/"), ".zip")
		if r.Method != http.MethodGet || label == "" || strings.Contains(label, "/") {
			http.NotFound(w, r)
			return
		}
		s.downloads.Add(1)
		_, _ = w.Write([]byte(label))
	}))
	t.Cleanup(s.Close)
	return s
}

func (s *buildServer) pin(label string) *Install {
	return &Install{
		Artifacts: []InstallArtifact{pinnedArtifact("server", s.URL+"/"+label+".zip", []byte(label))},
		Run:       []string{fakeEngineBin, "installbuild", "{install_dir}", "{download_server}"},
	}
}

// managedBuildManifest installs the fake engine itself into its managed
// directory, so the started process runs from the directory a refresh swaps.
func managedBuildManifest(t *testing.T, engine string, install *Install) *Manifest {
	t.Helper()
	manifest := testEngineManifest(fakeEngineBin)
	manifest.Engine = engine
	platform := manifest.Platforms[hostKey()]
	bin := filepath.Join("{install_dir}", testbin.Exe("fake-engine"))
	platform.Detect = []string{bin}
	platform.Runtime.Bin = bin
	platform.Install = install
	manifest.Platforms[hostKey()] = platform
	if err := manifest.Validate(); err != nil {
		t.Fatalf("validate fixture: %v", err)
	}
	return manifest
}

func installedBuild(t *testing.T, st *engineState) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(st.installDir, "build.txt"))
	if err != nil {
		t.Fatalf("read installed build label: %v", err)
	}
	return string(data)
}

func setPin(t *testing.T, ex *Executor, engine string, install *Install) *engineState {
	t.Helper()
	st, err := ex.state(engine)
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.plat.Install = install
	st.mu.Unlock()
	return st
}

func assertRunningFrom(t *testing.T, ex *Executor, st *engineState, engine string) {
	t.Helper()
	status, err := ex.Status(engine)
	if err != nil || !status.Running {
		t.Fatalf("engine status = %+v, %v; want running", status, err)
	}
	st.mu.Lock()
	binPath := st.binPath
	st.mu.Unlock()
	if !isManagedInstallPath(binPath, st.installDir) {
		t.Fatalf("engine runs %q, outside %q", binPath, st.installDir)
	}
}

func assertNoRefreshLeftovers(t *testing.T, ex *Executor, engine string) {
	t.Helper()
	staging, previous := ex.refreshDirs(engine)
	for _, dir := range []string{staging, previous} {
		if _, err := os.Stat(dir); !os.IsNotExist(err) {
			t.Fatalf("refresh left %s behind (%v)", dir, err)
		}
	}
}

// Before this fix a pin bump returned already-installed and kept the old build.
func TestInstallRefreshesManagedBuildWhenThePinMoves(t *testing.T) {
	server := newBuildServer(t)
	const engine = "refresh-pin"
	ex := newTestExecutor(t, managedBuildManifest(t, engine, server.pin("b1")))
	t.Cleanup(ex.StopAll)
	if err := ex.Install(context.Background(), engine); err != nil {
		t.Fatalf("first install: %v", err)
	}
	st := setPin(t, ex, engine, server.pin("b2"))
	if got := installedBuild(t, st); got != "b1" {
		t.Fatalf("first install build = %q", got)
	}
	assertRunningFrom(t, ex, st, engine)

	if err := ex.Install(context.Background(), engine); err != nil {
		t.Fatalf("install after pin bump: %v", err)
	}
	if got := installedBuild(t, st); got != "b2" {
		t.Fatalf("installed build after pin bump = %q, want b2", got)
	}
	marker, ok := readInstallMarker(st.installDir)
	if !ok || marker.Recipe != installRecipeID(server.pin("b2")) {
		t.Fatalf("marker = %+v, want the new recipe", marker)
	}
	assertRunningFrom(t, ex, st, engine)
	assertNoRefreshLeftovers(t, ex, engine)

	// The same pin is not downloaded again.
	before := server.downloads.Load()
	if err := ex.Install(context.Background(), engine); err != nil {
		t.Fatal(err)
	}
	if server.downloads.Load() != before {
		t.Fatal("an up-to-date managed install was downloaded again")
	}
}

// A PAIR upgrade restarts the manager; startup restoration refreshes then.
func TestRestoreInstalledRefreshesStaleManagedBuild(t *testing.T) {
	server := newBuildServer(t)
	const engine = "refresh-restore"
	base := t.TempDir()
	first := newExecutorAt(t, managedBuildManifest(t, engine, server.pin("b1")), base)
	if err := first.Install(context.Background(), engine); err != nil {
		t.Fatalf("first install: %v", err)
	}
	first.StopAll()

	ex := newExecutorAt(t, managedBuildManifest(t, engine, server.pin("b2")), base)
	t.Cleanup(ex.StopAll)
	if err := ex.RestoreInstalled(context.Background()); err != nil {
		t.Fatalf("restore: %v", err)
	}
	st, _ := ex.state(engine)
	if got := installedBuild(t, st); got != "b2" {
		t.Fatalf("restored build = %q, want b2", got)
	}
	assertRunningFrom(t, ex, st, engine)
	assertNoRefreshLeftovers(t, ex, engine)
}

func TestInstallRefreshFailureKeepsTheInstalledBuild(t *testing.T) {
	server := newBuildServer(t)
	const engine = "refresh-fail"
	ex := newTestExecutor(t, managedBuildManifest(t, engine, server.pin("b1")))
	t.Cleanup(ex.StopAll)
	if err := ex.Install(context.Background(), engine); err != nil {
		t.Fatalf("first install: %v", err)
	}
	broken := server.pin("b2")
	broken.Artifacts[0].SHA256 = strings.Repeat("0", 64)
	st := setPin(t, ex, engine, broken)

	if err := ex.Install(context.Background(), engine); err != nil {
		t.Fatalf("install with a failing refresh: %v", err)
	}
	if got := installedBuild(t, st); got != "b1" {
		t.Fatalf("installed build after failed refresh = %q, want b1", got)
	}
	marker, _ := readInstallMarker(st.installDir)
	if marker.Recipe != installRecipeID(server.pin("b1")) {
		t.Fatal("a failed refresh rewrote the marker")
	}
	assertRunningFrom(t, ex, st, engine)
	assertNoRefreshLeftovers(t, ex, engine)
	reported := false
	for _, problem := range ex.reporter.snapshot() {
		reported = reported || problem.ID == refreshFailedID(engine) && strings.Contains(problem.Message, "checksum mismatch")
	}
	if !reported {
		t.Fatal("a failed refresh was not reported")
	}
}

// A build outside the managed directory is the user's choice, and is kept.
func TestInstallLeavesUserSelectedBuildOnPinMove(t *testing.T) {
	server := newBuildServer(t)
	const engine = "refresh-user"
	manifest := managedBuildManifest(t, engine, server.pin("b1"))
	platform := manifest.Platforms[hostKey()]
	platform.Detect = []string{fakeEngineBin}
	platform.Runtime.Bin = fakeEngineBin
	manifest.Platforms[hostKey()] = platform
	ex := newTestExecutor(t, manifest)
	t.Cleanup(ex.StopAll)
	st, err := ex.state(engine)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeInstallMarker(st.installDir, installMarker{Engine: engine, Recipe: installRecipeID(server.pin("b1"))}); err != nil {
		t.Fatal(err)
	}
	setPin(t, ex, engine, server.pin("b2"))
	if err := ex.Install(context.Background(), engine); err != nil {
		t.Fatal(err)
	}
	if server.downloads.Load() != 0 {
		t.Fatal("a user-selected build was refreshed")
	}
}

func TestInterruptedRefreshRestoresThePreviousBuild(t *testing.T) {
	server := newBuildServer(t)
	const engine = "refresh-interrupted"
	ex := newTestExecutor(t, managedBuildManifest(t, engine, server.pin("b1")))
	t.Cleanup(ex.StopAll)
	if err := ex.Install(context.Background(), engine); err != nil {
		t.Fatalf("first install: %v", err)
	}
	st, _ := ex.state(engine)
	st.opMu.Lock()
	err := ex.doStop(st, engine)
	st.opMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	_, previous := ex.refreshDirs(engine)
	if err := os.Rename(st.installDir, previous); err != nil {
		t.Fatal(err)
	}
	before := server.downloads.Load()
	if err := ex.Install(context.Background(), engine); err != nil {
		t.Fatal(err)
	}
	if got := installedBuild(t, st); got != "b1" || server.downloads.Load() != before {
		t.Fatalf("interrupted refresh recovery = %q with %d downloads", got, server.downloads.Load()-before)
	}
	assertNoRefreshLeftovers(t, ex, engine)
	deadline := time.Now().Add(10 * time.Second)
	for !func() bool { s, _ := ex.Status(engine); return s.Running }() {
		if time.Now().After(deadline) {
			t.Fatal("recovered build did not start")
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func newExecutorAt(t *testing.T, m *Manifest, base string) *Executor {
	t.Helper()
	reg := NewRegistry()
	reg.engines[m.Engine] = m
	ex := NewExecutor(reg, NewReporter(nil), func(string, any) {}, base)
	acceptModelMutations(ex)
	return ex
}

// A new build that is detected but cannot start is rolled back to the old one.
func TestInstallRefreshRollsBackABuildThatDoesNotStart(t *testing.T) {
	server := newBuildServer(t)
	const engine = "refresh-nostart"
	ex := newTestExecutor(t, managedBuildManifest(t, engine, server.pin("b1")))
	t.Cleanup(ex.StopAll)
	if err := ex.Install(context.Background(), engine); err != nil {
		t.Fatalf("first install: %v", err)
	}
	unstartable := server.pin("b2")
	unstartable.Run = []string{fakeEngineBin, "touch", filepath.Join("{install_dir}", testbin.Exe("fake-engine"))}
	st := setPin(t, ex, engine, unstartable)

	if err := ex.Install(context.Background(), engine); err != nil {
		t.Fatalf("install with an unstartable new build: %v", err)
	}
	if got := installedBuild(t, st); got != "b1" {
		t.Fatalf("installed build after rollback = %q, want b1", got)
	}
	assertRunningFrom(t, ex, st, engine)
	assertNoRefreshLeftovers(t, ex, engine)
	reported := false
	for _, problem := range ex.reporter.snapshot() {
		reported = reported || problem.ID == refreshFailedID(engine) && strings.Contains(problem.Message, "did not start")
	}
	if !reported {
		t.Fatal("the rollback was not reported")
	}
}

func TestUninstallRemovesRefreshLeftovers(t *testing.T) {
	server := newBuildServer(t)
	const engine = "refresh-uninstall"
	manifest := managedBuildManifest(t, engine, server.pin("b1"))
	platform := manifest.Platforms[hostKey()]
	platform.Uninstall = &Uninstall{Remove: []string{"{install_dir}"}}
	manifest.Platforms[hostKey()] = platform
	ex := newTestExecutor(t, manifest)
	t.Cleanup(ex.StopAll)
	if err := ex.Install(context.Background(), engine); err != nil {
		t.Fatalf("install: %v", err)
	}
	_, previous := ex.refreshDirs(engine)
	if err := os.MkdirAll(previous, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := ex.Uninstall(context.Background(), engine); err != nil {
		t.Fatalf("uninstall: %v", err)
	}
	assertNoRefreshLeftovers(t, ex, engine)
}
