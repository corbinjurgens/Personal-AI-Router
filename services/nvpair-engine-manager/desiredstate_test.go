// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"nvpair-shared/durablefile"
)

func desiredTestExecutor(t *testing.T, manifest *Manifest, baseDir string) *Executor {
	t.Helper()
	reg := NewRegistry()
	reg.engines[manifest.Engine] = manifest
	ex := NewExecutor(reg, NewReporter(nil), func(string, any) {}, baseDir)
	t.Cleanup(ex.StopAll)
	return ex
}

func assertDesiredFileBytes(t *testing.T, path string, want []byte) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("desired-state bytes = %q, want %q", got, want)
	}
}

func TestInstalledEngineRestoresAfterShutdownAndInternalStop(t *testing.T) {
	baseDir := t.TempDir()
	ctx := context.Background()

	first := desiredTestExecutor(t, testEngineManifest(fakeEngineBin), baseDir)
	if err := first.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	first.StopAll()

	second := desiredTestExecutor(t, testEngineManifest(fakeEngineBin), baseDir)
	t.Cleanup(second.StopAll)
	if err := second.RestoreInstalled(ctx); err != nil {
		t.Fatalf("restore ON: %v", err)
	}
	if st, err := second.Status("fake"); err != nil || !st.Running {
		t.Fatalf("restored status = %+v, err=%v", st, err)
	}
	if err := second.Stop("fake"); err != nil {
		t.Fatalf("explicit stop: %v", err)
	}

	third := desiredTestExecutor(t, testEngineManifest(fakeEngineBin), baseDir)
	t.Cleanup(third.StopAll)
	if err := third.RestoreInstalled(ctx); err != nil {
		t.Fatalf("restore after OFF: %v", err)
	}
	if st, err := third.Status("fake"); err != nil || !st.Running {
		t.Fatalf("installed status = %+v, err=%v", st, err)
	}
}

func TestRestoreInstalledPreservesForeignListenerOnManagedPort(t *testing.T) {
	testRestoreInstalledPreservesForeignListener(t, false)
}

func TestRestoreInstalledPreservesForeignListenerOnRecordedPort(t *testing.T) {
	testRestoreInstalledPreservesForeignListener(t, true)
}

func testRestoreInstalledPreservesForeignListener(t *testing.T, portZero bool) {
	t.Helper()
	baseDir := t.TempDir()
	managedDir := filepath.Join(baseDir, "fake")
	if err := os.MkdirAll(managedDir, 0o700); err != nil {
		t.Fatalf("create managed install: %v", err)
	}
	managedBin := filepath.Join(managedDir, filepath.Base(fakeEngineBin))
	foreignBin := filepath.Join(t.TempDir(), filepath.Base(fakeEngineBin))
	copyFile(t, fakeEngineBin, managedBin)
	copyFile(t, fakeEngineBin, foreignBin)
	port, err := freePort()
	if err != nil {
		t.Fatalf("reserve foreign listener port: %v", err)
	}
	foreign := spawnFakeListener(t, foreignBin, port)
	manifest := testEngineManifest(managedBin)
	platform := manifest.Platforms[hostKey()]
	platform.Runtime.Port = port
	if portZero {
		platform.Runtime.Port = 0
	}
	platform.Runtime.Health = nil
	manifest.Platforms[hostKey()] = platform
	ex := desiredTestExecutor(t, manifest, baseDir)
	if portZero {
		if err := ex.desired.setAllocatedPort("fake", port); err != nil {
			t.Fatalf("save allocated port: %v", err)
		}
	}
	if err := ex.RestoreInstalled(context.Background()); err != nil {
		t.Fatalf("restore with foreign listener: %v", err)
	}
	state, err := ex.state("fake")
	if err != nil {
		t.Fatalf("read restored state: %v", err)
	}
	state.mu.Lock()
	running, adopted, proc := state.running, state.adopted, state.proc
	state.mu.Unlock()
	if !running || !adopted || proc != nil || !pidAlive(foreign.Process.Pid) {
		t.Fatalf("foreign listener changed: running=%v adopted=%v owned=%v foreignAlive=%v", running, adopted, proc != nil, pidAlive(foreign.Process.Pid))
	}
	if portZero {
		if err := ex.Stop("fake"); err == nil {
			t.Fatal("stop must refuse the foreign image")
		}
		if saved, err := ex.desired.allocatedPort("fake"); err != nil || saved != port {
			t.Fatalf("refused stop lost endpoint evidence: port=%d err=%v", saved, err)
		}
		if !pidAlive(foreign.Process.Pid) || !portServing(port) {
			t.Fatal("foreign listener was touched by restore or stop")
		}
	}
}

func TestAllocatedPortLegacyStateAndFixedPort(t *testing.T) {
	base := t.TempDir()
	store := newDesiredStateStore(base)
	if err := os.WriteFile(store.path, []byte(`{"engines":{"fake":true}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if port, err := store.allocatedPort("fake"); err != nil || port != 0 {
		t.Fatalf("legacy allocated port = %d, %v; want zero", port, err)
	}
	if err := store.setAllocatedPort("fake", 12345); err != nil {
		t.Fatal(err)
	}
	m := testEngineManifest(fakeEngineBin)
	p := m.Platforms[hostKey()]
	p.Runtime.Port = privateTestPorts(t, 1)[0]
	m.Platforms[hostKey()] = p
	ex := desiredTestExecutor(t, m, base)
	if err := ex.RestoreInstalled(context.Background()); err != nil {
		t.Fatal(err)
	}
	if st, err := ex.Status("fake"); err != nil || st.Port != p.Runtime.Port || !st.Running {
		t.Fatalf("fixed port overridden by allocation record: %+v, %v", st, err)
	}
}

func TestAllocatedPortWriteFailurePreventsSpawn(t *testing.T) {
	ex := desiredTestExecutor(t, testEngineManifest(fakeEngineBin), t.TempDir())
	pidFile := filepath.Join(t.TempDir(), "fake.pid")
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.plat.Runtime.Env["FAKE_PID_FILE"] = pidFile
	failure := errors.New("allocated endpoint write failed")
	useEngineFileReplacer(t, durablefile.Replacer{CreateTemp: func(string, string, os.FileMode) (durablefile.File, error) {
		return nil, failure
	}})
	if err := ex.Start(context.Background(), "fake"); !errors.Is(err, failure) {
		t.Fatalf("start error = %v, want allocation persistence failure", err)
	}
	if _, err := os.Stat(pidFile); !os.IsNotExist(err) {
		t.Fatalf("child spawned before durable endpoint: PID file stat error = %v", err)
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.proc != nil || st.running {
		t.Fatalf("failed durable endpoint left proc=%p running=%v", st.proc, st.running)
	}
}

func TestAllocatedPortSleepRetriesFailedCleanup(t *testing.T) {
	base := t.TempDir()
	ex := desiredTestExecutor(t, testEngineManifest(fakeEngineBin), base)
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	started, err := ex.Status("fake")
	if err != nil {
		t.Fatal(err)
	}
	failure := errors.New("allocated endpoint cleanup failed")
	failed := false
	useEngineFileReplacer(t, durablefile.Replacer{CreateTemp: func(dir, base string, _ os.FileMode) (durablefile.File, error) {
		if !failed {
			failed = true
			return nil, failure
		}
		return os.CreateTemp(dir, base+".*.tmp")
	}})
	if err := ex.Sleep("fake"); !errors.Is(err, failure) {
		t.Fatalf("first sleep = %v, want endpoint cleanup failure", err)
	}
	if portServing(started.Port) {
		t.Fatal("sleep did not stop the owned process before the persistence failure")
	}
	if port, err := newDesiredStateStore(base).allocatedPort("fake"); err != nil || port != started.Port {
		t.Fatalf("failed cleanup endpoint = %d, %v; want %d", port, err, started.Port)
	}
	if err := ex.Sleep("fake"); err != nil {
		t.Fatalf("retry sleep: %v", err)
	}
	saved, err := newDesiredStateStore(base).load()
	if err != nil {
		t.Fatal(err)
	}
	if _, present := saved.AllocatedPorts["fake"]; present {
		t.Fatalf("retry retained endpoint or lost saved On: %+v", saved)
	}
}

func TestRestoreInstalledPreservesDetectedExternalImage(t *testing.T) {
	externalBin := filepath.Join(t.TempDir(), filepath.Base(fakeEngineBin))
	copyFile(t, fakeEngineBin, externalBin)
	port, err := freePort()
	if err != nil {
		t.Fatalf("reserve external listener port: %v", err)
	}
	external := spawnFakeListener(t, externalBin, port)
	manifest := testEngineManifest(externalBin)
	platform := manifest.Platforms[hostKey()]
	platform.Runtime.Port = port
	platform.Runtime.Health = nil
	manifest.Platforms[hostKey()] = platform
	ex := desiredTestExecutor(t, manifest, t.TempDir())
	if installed, err := ex.Detect("fake"); err != nil || !installed {
		t.Fatalf("detect external install = %v, err=%v", installed, err)
	}
	state, err := ex.state("fake")
	if err != nil {
		t.Fatalf("read detected state: %v", err)
	}
	state.mu.Lock()
	binPath, installDir := state.binPath, state.installDir
	state.mu.Unlock()
	pid, image, found := pidOnPort(port)
	if !found || pid != external.Process.Pid || !isOurEngineImage(image, binPath) || isManagedInstallPath(binPath, installDir) {
		t.Fatalf("external fixture must match detected image outside managed install: listener=(%d,%q,%v) detected=%q installDir=%q", pid, image, found, binPath, installDir)
	}
	if err := ex.RestoreInstalled(context.Background()); err != nil {
		t.Fatalf("restore with detected external listener: %v", err)
	}
	state.mu.Lock()
	running, adopted, proc := state.running, state.adopted, state.proc
	state.mu.Unlock()
	pid, image, found = pidOnPort(port)
	if !running || !adopted || proc != nil || !pidAlive(external.Process.Pid) || !found || pid != external.Process.Pid || !isOurEngineImage(image, externalBin) {
		t.Fatalf("external listener changed: running=%v adopted=%v owned=%v originalAlive=%v listener=(%d,%q,%v)", running, adopted, proc != nil, pidAlive(external.Process.Pid), pid, image, found)
	}
}

func TestOrphanReclaimFailureReportsStartFailed(t *testing.T) {
	test := func(name string, listenerProbe bool, wantMessage string) {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusOK)
			}))
			t.Cleanup(srv.Close)
			u, err := url.Parse(srv.URL)
			if err != nil {
				t.Fatalf("parse listener URL: %v", err)
			}
			port, err := strconv.Atoi(u.Port())
			if err != nil {
				t.Fatalf("parse listener port: %v", err)
			}
			ex := desiredTestExecutor(t, testEngineManifest(fakeEngineBin), t.TempDir())
			// Exercise the post-signal owner helper with this live PID and
			// listener, without sending any signal to the test process.
			rt := Runtime{Stop: &StopSpec{Signal: "kill"}}
			if listenerProbe {
				rt.Ready = &Probe{HTTP: srv.URL, Status: http.StatusOK}
			}
			err = ex.confirmOrphanStopped("fake", os.Getpid(), port, rt)
			if err == nil || !strings.Contains(err.Error(), wantMessage) {
				t.Fatalf("confirm orphan stopped = %v, want %q", err, wantMessage)
			}
			reported := ex.Errors()
			if len(reported) != 1 || reported[0].ID != startFailedID("fake") || reported[0].Message != err.Error() || reported[0].Operation != "start" || reported[0].Action != "retry" {
				t.Fatalf("reclaim failure reports = %+v, want retryable start failure", reported)
			}
		})
	}
	test("listener still available", true, "listener on port")
	test("PID still alive without readiness probe", false, "pid ")
}

func TestRestoreInstalledReclaimsUnhealthyManagedOrphan(t *testing.T) {
	baseDir := t.TempDir()
	managedDir := filepath.Join(baseDir, "fake")
	if err := os.MkdirAll(managedDir, 0o700); err != nil {
		t.Fatalf("create managed install: %v", err)
	}
	managedBin := filepath.Join(managedDir, filepath.Base(fakeEngineBin))
	copyFile(t, fakeEngineBin, managedBin)
	port, err := freePort()
	if err != nil {
		t.Fatalf("reserve orphan port: %v", err)
	}
	orphan := spawnFakeListener(t, managedBin, port)
	manifest := testEngineManifest(managedBin)
	platform := manifest.Platforms[hostKey()]
	platform.Runtime.Port = port
	platform.Runtime.Ready.Status = http.StatusCreated // listener is alive but fails readiness
	platform.Runtime.Ready.TimeoutS = 1
	platform.Runtime.Health = nil
	manifest.Platforms[hostKey()] = platform
	ex := desiredTestExecutor(t, manifest, baseDir)
	if err := ex.RestoreInstalled(context.Background()); err == nil || !strings.Contains(err.Error(), "did not become ready") {
		t.Fatalf("restore with deliberately unmatched readiness = %v, want cold-start readiness failure", err)
	}
	if portServing(port) {
		t.Fatalf("unhealthy managed orphan PID %d still holds port %d", orphan.Process.Pid, port)
	}
}

func TestFailedRestoreRetainsDesiredOnForRetry(t *testing.T) {
	baseDir := t.TempDir()
	seed := desiredTestExecutor(t, testEngineManifest(fakeEngineBin), baseDir)
	if err := seed.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("seed start: %v", err)
	}
	seed.StopAll()

	failing := testEngineManifest(fakeEngineBin)
	key := runtime.GOOS + "/" + runtime.GOARCH
	platform := failing.Platforms[key]
	platform.Runtime.Args = []string{"noserve"}
	platform.Runtime.Ready.TimeoutS = 1
	failing.Platforms[key] = platform
	if err := desiredTestExecutor(t, failing, baseDir).RestoreInstalled(context.Background()); err == nil {
		t.Fatal("expected restore failure")
	}

	retry := desiredTestExecutor(t, testEngineManifest(fakeEngineBin), baseDir)
	t.Cleanup(retry.StopAll)
	if err := retry.RestoreInstalled(context.Background()); err != nil {
		t.Fatalf("retry restore: %v", err)
	}
	if st, err := retry.Status("fake"); err != nil || !st.Running {
		t.Fatalf("retry status = %+v, err=%v (state file %s)", st, err, filepath.Join(baseDir, desiredStateFileName))
	}
}

// Retired transaction and engine intent fields do not change the retained
// process cleanup evidence, and a read never rewrites the document.
func TestDesiredStateDecodeDropsRetiredTransactionKeysWithoutWriting(t *testing.T) {
	store := newDesiredStateStore(t.TempDir())
	document := []byte(`{"engines":{"on":true,"off":false},"offBy":{"off":"peer"},"allocatedPorts":{"on":23456},"pendingSwitch":{"automatic":"retired-invalid-value"},"pendingSwitchPorts":{"target":-1}}`)
	if err := os.WriteFile(store.path, document, 0o600); err != nil {
		t.Fatal(err)
	}
	state, err := store.load()
	if err != nil {
		t.Fatalf("decode retained state: %v", err)
	}
	want := desiredStateFile{AllocatedPorts: map[string]int{"on": 23456}}
	if !reflect.DeepEqual(state, want) {
		t.Fatalf("retained state = %+v, want %+v", state, want)
	}
	assertDesiredFileBytes(t, store.path, document)
	data, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"pendingSwitch", "pendingSwitchPorts"} {
		if _, present := fields[key]; present {
			t.Fatalf("decoded state retained %s", key)
		}
	}
}
