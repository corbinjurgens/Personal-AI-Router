// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
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
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/engineintent"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

func TestDesiredStateSnapshotReadOnly(t *testing.T) {
	test := func(name, content string) {
		t.Run(name, func(t *testing.T) {
			store := newDesiredStateStore(t.TempDir())
			if content != "" {
				if err := os.WriteFile(store.path, []byte(content), 0o400); err != nil {
					t.Fatal(err)
				}
			}
			before, err := os.Stat(store.path)
			if err != nil && !os.IsNotExist(err) {
				t.Fatal(err)
			}
			writes := 0
			useEngineFileReplacer(t, durablefile.Replacer{CreateTemp: func(string, string, os.FileMode) (durablefile.File, error) {
				writes++
				return nil, errors.New("read must not write")
			}})
			if _, err := store.snapshot([]string{"fake"}); err != nil {
				t.Fatal(err)
			}
			if writes != 0 {
				t.Fatalf("snapshot attempted %d writes", writes)
			}
			after, err := os.Stat(store.path)
			if content == "" {
				if !os.IsNotExist(err) {
					t.Fatalf("snapshot created missing file: stat error %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			data, err := os.ReadFile(store.path)
			if err != nil {
				t.Fatal(err)
			}
			if string(data) != content || !before.ModTime().Equal(after.ModTime()) || before.Mode() != after.Mode() {
				t.Fatalf("snapshot changed desired file: content=%s before=%v after=%v", data, before, after)
			}
		})
	}
	test("missing file stays missing", "")
	test("legacy Off stays byte-identical", `{"engines":{"fake":false}}`)
	test("peer Off stays byte-identical", `{"engines":{"fake":false},"offBy":{"fake":"peer"}}`)
}

func TestDesiredStateSnapshotConcurrentConsistency(t *testing.T) {
	store := newDesiredStateStore(t.TempDir())
	states := []desiredStateFile{
		{Engines: map[string]bool{"first": true, "second": false}, OffBy: map[string]intentOrigin{"second": intentLocal}},
		{Engines: map[string]bool{"first": false, "second": true}, OffBy: map[string]intentOrigin{"first": intentPeer}},
	}
	want := []engineintent.Snapshot{
		{EnabledByEngine: map[string]bool{"first": true, "second": false}, OffByEngine: map[string]engineintent.Origin{"second": "local"}},
		{EnabledByEngine: map[string]bool{"first": false, "second": true}, OffByEngine: map[string]engineintent.Origin{"first": "peer"}},
	}
	if err := store.save(states[0]); err != nil {
		t.Fatal(err)
	}
	start := make(chan struct{})
	writer := make(chan error, 1)
	go func() {
		<-start
		for i := 0; i < 100; i++ {
			store.mu.Lock()
			err := store.save(states[i%2])
			store.mu.Unlock()
			if err != nil {
				writer <- err
				return
			}
		}
		writer <- nil
	}()
	close(start)
	// Always join the writer before the temporary store is removed.
	defer func() {
		if err := <-writer; err != nil {
			t.Error(err)
		}
	}()
	for i := 0; i < 1000; i++ {
		got, err := store.snapshot([]string{"first", "second"})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want[0]) && !reflect.DeepEqual(got, want[1]) {
			t.Fatalf("snapshot combined different store revisions: %+v", got)
		}
	}
}

func TestDesiredStateOnClearsOffBy(t *testing.T) {
	for _, origin := range []intentOrigin{intentLocal, intentPeer} {
		t.Run(string(origin), func(t *testing.T) {
			store := newDesiredStateStore(t.TempDir())
			if err := store.set("fake", false, origin); err != nil {
				t.Fatal(err)
			}
			if err := store.set("fake", true, intentLocal); err != nil {
				t.Fatal(err)
			}
			state, err := store.load()
			if err != nil {
				t.Fatal(err)
			}
			if !state.Engines["fake"] || len(state.OffBy) != 0 {
				t.Fatalf("saved On retained provenance: %+v", state)
			}
		})
	}
}

func desiredTestExecutor(t *testing.T, manifest *Manifest, baseDir string) *Executor {
	t.Helper()
	reg := NewRegistry()
	reg.engines[manifest.Engine] = manifest
	return NewExecutor(reg, NewReporter(nil), func(string, any) {}, baseDir)
}

func TestResidentPreservesDesiredStateAndFailedWrites(t *testing.T) {
	f := newResidentFixture(t)
	_, p := switchFixture(t)
	p.SourceEngine, p.TargetEngine = "source", "target"
	p.ConfigurationFence = "e3v1:" + nameDigest("configuration")
	state := desiredStateFile{
		PendingSwitch:      &settings.SwitchCheckpoint{SwitchPrepareRequest: p, Phase: settings.SwitchCheckpointSourceOff},
		PendingSwitchPorts: &switchCheckpointPorts{Source: 17001, Target: 17002},
		Engines:            map[string]bool{f.p.Engine: true, "source": false, "target": true, "legacy": false},
		OffBy:              map[string]intentOrigin{"source": intentLocal, "legacy": intentPeer},
	}
	if err := f.ex.desired.save(state); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(f.ex.desired.path)
	if err != nil {
		t.Fatal(err)
	}
	initial, err := f.ex.desired.load()
	if err != nil {
		t.Fatal(err)
	}
	// Inject failure at the existing atomic writer, not a residency-only seam.
	writes := 0
	useEngineFileReplacer(t, durablefile.Replacer{CreateTemp: func(string, string, os.FileMode) (durablefile.File, error) {
		writes++
		return nil, errors.New("injected desired write failure")
	}})
	if result := f.run(t); result.Outcome != nodepolicy.ResidentLoaded {
		t.Fatalf("result = %+v", result)
	}
	if writes != 0 {
		t.Fatalf("residency attempted %d desired-state writes", writes)
	}
	assertDesiredFileBytes(t, f.ex.desired.path, before)
	if err := f.ex.desired.set("unrelated", false, intentLocal); err == nil {
		t.Fatal("failed write reported success")
	}
	if writes != 1 {
		t.Fatalf("injected writer calls = %d, want one failed intent write", writes)
	}
	assertDesiredFileBytes(t, f.ex.desired.path, before)
	// Reopen the store to prove checkpoint, port, intent and provenance survive
	// the disk round trip rather than relying on the original store instance.
	after, err := newDesiredStateStore(filepath.Dir(f.ex.desired.path)).load()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(initial, after) {
		t.Fatalf("desired state changed: before=%+v after=%+v", initial, after)
	}
}

func TestResidentFailedDesiredReadDoesNotRepairState(t *testing.T) {
	f := newResidentFixture(t)
	before, err := os.ReadFile(f.ex.desired.path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.WriteFile(f.ex.desired.path, before, 0o600); err != nil {
			t.Errorf("restore desired state for fixture cleanup: %v", err)
		}
	})
	const malformed = `{"engines":`
	if err := os.WriteFile(f.ex.desired.path, []byte(malformed), 0o600); err != nil {
		t.Fatal(err)
	}
	writes := 0
	useEngineFileReplacer(t, durablefile.Replacer{CreateTemp: func(string, string, os.FileMode) (durablefile.File, error) {
		writes++
		return nil, errors.New("residency must not repair desired state")
	}})
	if result := f.run(t); result.Outcome != nodepolicy.ResidentNotDispatched {
		t.Fatalf("failed desired read result = %+v", result)
	}
	f.noCalls(t)
	if f.dispatches != 0 || writes != 0 {
		t.Fatalf("failed desired read dispatched=%d writes=%d", f.dispatches, writes)
	}
	assertDesiredFileBytes(t, f.ex.desired.path, []byte(malformed))
	if _, err := newDesiredStateStore(filepath.Dir(f.ex.desired.path)).load(); err == nil {
		t.Fatal("malformed desired state became readable")
	}
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

func TestDesiredStateStoresExplicitOffAndLeavesLegacyUnknown(t *testing.T) {
	store := newDesiredStateStore(t.TempDir())
	if enabled, known, err := store.get("fake"); err != nil || known || enabled {
		t.Fatalf("legacy state = (%v, %v, %v), want unknown", enabled, known, err)
	}
	if err := store.set("fake", false, intentLocal); err != nil {
		t.Fatal(err)
	}
	if enabled, known, err := store.get("fake"); err != nil || !known || enabled {
		t.Fatalf("saved OFF = (%v, %v, %v), want known false", enabled, known, err)
	}
	if err := store.set("fake", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	if enabled, known, err := store.get("fake"); err != nil || !known || !enabled {
		t.Fatalf("saved ON = (%v, %v, %v), want known true", enabled, known, err)
	}
}

func TestDesiredStateRecordsOffProvenance(t *testing.T) {
	requireOffLocally := func(t *testing.T, store *desiredStateStore, want bool) {
		t.Helper()
		if got, err := store.offLocally("fake"); err != nil || got != want {
			t.Fatalf("offLocally = (%v, %v), want %v", got, err, want)
		}
	}
	t.Run("unknown intent is not a local Off", func(t *testing.T) {
		requireOffLocally(t, newDesiredStateStore(t.TempDir()), false)
	})
	t.Run("local Off vetoes until a start clears it", func(t *testing.T) {
		store := newDesiredStateStore(t.TempDir())
		if err := store.set("fake", false, intentLocal); err != nil {
			t.Fatal(err)
		}
		requireOffLocally(t, store, true)
		if err := store.set("fake", true, intentPeer); err != nil {
			t.Fatal(err)
		}
		requireOffLocally(t, store, false)
		if state, err := store.load(); err != nil || len(state.OffBy) != 0 {
			t.Fatalf("saving On left Off provenance %v (%v)", state.OffBy, err)
		}
	})
	t.Run("peer Off does not veto", func(t *testing.T) {
		store := newDesiredStateStore(t.TempDir())
		if err := store.set("fake", false, intentPeer); err != nil {
			t.Fatal(err)
		}
		requireOffLocally(t, store, false)
	})
	t.Run("peer Off never downgrades a local Off", func(t *testing.T) {
		store := newDesiredStateStore(t.TempDir())
		if err := store.set("fake", false, intentLocal); err != nil {
			t.Fatal(err)
		}
		if err := store.set("fake", false, intentPeer); err != nil {
			t.Fatal(err)
		}
		requireOffLocally(t, store, true)
	})
	t.Run("local Off replaces a peer Off", func(t *testing.T) {
		store := newDesiredStateStore(t.TempDir())
		if err := store.set("fake", false, intentPeer); err != nil {
			t.Fatal(err)
		}
		if err := store.set("fake", false, intentLocal); err != nil {
			t.Fatal(err)
		}
		requireOffLocally(t, store, true)
	})
	t.Run("legacy Off without provenance counts as local", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, desiredStateFileName), []byte(`{"engines":{"fake":false}}`), 0o600); err != nil {
			t.Fatal(err)
		}
		store := newDesiredStateStore(dir)
		requireOffLocally(t, store, true)
		if err := store.set("fake", false, intentPeer); err != nil {
			t.Fatal(err)
		}
		requireOffLocally(t, store, true)
	})
}

func TestDesiredOnSurvivesShutdownAndExplicitOffDoesNot(t *testing.T) {
	baseDir := t.TempDir()
	ctx := context.Background()

	first := desiredTestExecutor(t, testEngineManifest(fakeEngineBin), baseDir)
	if err := first.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	first.StopAll()

	second := desiredTestExecutor(t, testEngineManifest(fakeEngineBin), baseDir)
	t.Cleanup(second.StopAll)
	if err := second.RestoreEnabled(ctx); err != nil {
		t.Fatalf("restore ON: %v", err)
	}
	if st, err := second.Status("fake"); err != nil || !st.Running {
		t.Fatalf("restored status = %+v, err=%v", st, err)
	}
	if err := second.Stop("fake"); err != nil {
		t.Fatalf("explicit stop: %v", err)
	}

	third := desiredTestExecutor(t, testEngineManifest(fakeEngineBin), baseDir)
	if err := third.RestoreEnabled(ctx); err != nil {
		t.Fatalf("restore after OFF: %v", err)
	}
	if st, err := third.Status("fake"); err != nil || st.Running {
		t.Fatalf("OFF status = %+v, err=%v", st, err)
	}
}

func TestRestoreEnabledPreservesForeignListenerOnManagedPort(t *testing.T) {
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
	platform.Runtime.Health = nil
	manifest.Platforms[hostKey()] = platform
	ex := desiredTestExecutor(t, manifest, baseDir)
	if err := ex.desired.set("fake", true, intentLocal); err != nil {
		t.Fatalf("save ON intent: %v", err)
	}
	if err := ex.RestoreEnabled(context.Background()); err != nil {
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
}

func TestRestoreEnabledReclaimsUnhealthyManagedOrphan(t *testing.T) {
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
	if err := ex.desired.set("fake", true, intentLocal); err != nil {
		t.Fatalf("save ON intent: %v", err)
	}
	if err := ex.RestoreEnabled(context.Background()); err == nil || !strings.Contains(err.Error(), "did not become ready") {
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
	if err := desiredTestExecutor(t, failing, baseDir).RestoreEnabled(context.Background()); err == nil {
		t.Fatal("expected restore failure")
	}

	retry := desiredTestExecutor(t, testEngineManifest(fakeEngineBin), baseDir)
	t.Cleanup(retry.StopAll)
	if err := retry.RestoreEnabled(context.Background()); err != nil {
		t.Fatalf("retry restore: %v", err)
	}
	if st, err := retry.Status("fake"); err != nil || !st.Running {
		t.Fatalf("retry status = %+v, err=%v (state file %s)", st, err, filepath.Join(baseDir, desiredStateFileName))
	}
}

// TestDeclinedAdoptedStopStillPersistsOffIntent proves a user Stop that is
// declined because the adopted listener is a genuinely foreign process (its
// image is not our managed binary) still records the OFF intent, so a later
// RestoreEnabled honors it rather than flipping the engine back on.
func TestDeclinedAdoptedStopStillPersistsOffIntent(t *testing.T) {
	baseDir := t.TempDir()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}

	// The httptest listener runs in this test process, whose image is not the
	// fake engine binary, so the stop is declined (never terminated).
	m := testEngineManifest(fakeEngineBin)
	key := runtime.GOOS + "/" + runtime.GOARCH
	p := m.Platforms[key]
	p.Runtime.Port = port
	p.Runtime.Ready = &Probe{HTTP: srv.URL, Status: http.StatusOK}
	p.Runtime.Health = nil
	m.Platforms[key] = p

	first := desiredTestExecutor(t, m, baseDir)
	if st, err := first.Status("fake"); err != nil || !st.Running {
		t.Fatalf("adopt foreign listener: status=%+v err=%v", st, err)
	}
	if err := first.Stop("fake"); err == nil || !strings.Contains(err.Error(), "external management") {
		t.Fatalf("stop error = %v, want external-management decline", err)
	}
	if enabled, known, err := first.desired.get("fake"); err != nil || !known || enabled {
		t.Fatalf("desired after declined stop = (%v, %v, %v), want known OFF", enabled, known, err)
	}

	srv.Close()
	fresh := desiredTestExecutor(t, m, baseDir)
	if err := fresh.RestoreEnabled(context.Background()); err != nil {
		t.Fatalf("restore after declined OFF: %v", err)
	}
	if st, err := fresh.Status("fake"); err != nil || st.Running {
		t.Fatalf("declined-OFF engine was restarted: status=%+v err=%v", st, err)
	}
}

func TestConfirmedStopPersistsOffDespiteCommandExitError(t *testing.T) {
	baseDir := t.TempDir()
	marker := filepath.Join(t.TempDir(), "stop-attempted")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}

	manifest := testEngineManifest(fakeEngineBin)
	key := runtime.GOOS + "/" + runtime.GOARCH
	platform := manifest.Platforms[key]
	platform.Runtime = Runtime{
		Mode:  "command",
		Port:  port,
		Ready: &Probe{HTTP: srv.URL, Status: http.StatusOK},
		Stop:  &StopSpec{Cmd: []string{fakeEngineBin, "failmark", marker}, GraceS: 2},
	}
	manifest.Platforms[key] = platform

	first := desiredTestExecutor(t, manifest, baseDir)
	if err := first.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("adopt and save ON: %v", err)
	}
	closed := make(chan struct{})
	go func() {
		for !fileExists(marker) {
			time.Sleep(10 * time.Millisecond)
		}
		srv.Close()
		close(closed)
	}()
	if err := first.Stop("fake"); err != nil {
		t.Fatalf("endpoint-down stop should succeed despite command exit: %v", err)
	}
	<-closed
	if enabled, known, err := first.desired.get("fake"); err != nil || !known || enabled {
		t.Fatalf("saved state after confirmed stop = (%v, %v, %v), want known OFF", enabled, known, err)
	}

	fresh := desiredTestExecutor(t, manifest, baseDir)
	if err := fresh.RestoreEnabled(context.Background()); err != nil {
		t.Fatalf("restore after confirmed OFF: %v", err)
	}
	if st, err := fresh.Status("fake"); err != nil || st.Running {
		t.Fatalf("confirmed OFF was restored unexpectedly: status=%+v err=%v", st, err)
	}
}
