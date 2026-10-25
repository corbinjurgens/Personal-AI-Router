// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"nvpair-shared/engineintent"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"sync"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/testbin"
)

func unsupportedStopFixture(t *testing.T) (*Executor, string, int) {
	t.Helper()
	baseDir := t.TempDir()
	installDir := filepath.Join(baseDir, "fake")
	if err := os.MkdirAll(installDir, 0o700); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(installDir, testbin.Exe("fakeengine"))
	copyFile(t, fakeEngineBin, bin)
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	m := testEngineManifest(bin)
	delete(m.Platforms, hostKey())
	return desiredTestExecutor(t, m, baseDir), bin, port
}

func TestSleepUnsupportedReclaimsManagedOrphan(t *testing.T) {
	ex, bin, port := unsupportedStopFixture(t)
	orphan := spawnFakeListener(t, bin, port)
	if err := ex.desired.setProcessStop("fake", processStopEvidence{Bin: bin, Port: port, Address: "127.0.0.1:" + strconv.Itoa(port), PID: orphan.Process.Pid}); err != nil {
		t.Fatal(err)
	}
	before, err := ex.Status("fake")
	if err != nil || before.Running || before.TerminationVerified {
		t.Fatalf("unprobed orphan status=%+v err=%v", before, err)
	}
	if err := ex.Sleep("fake"); err != nil {
		t.Fatalf("sleep unsupported orphan: %v", err)
	}
	if !orphanPIDExited(orphan.Process.Pid) || portServing(port) {
		t.Fatal("Sleep returned before physical termination")
	}
	status, err := ex.Status("fake")
	if err != nil || status.Running || !status.TerminationVerified {
		t.Fatalf("stopped status=%+v err=%v", status, err)
	}
	installed := ex.GetInstalled()
	if len(installed) != 1 || !installed[0].TerminationVerified {
		t.Fatalf("installed status=%+v", installed)
	}
	raw, err := json.Marshal(installed[0])
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		TerminationVerified *bool `json:"terminationVerified"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.TerminationVerified == nil || !*wire.TerminationVerified {
		t.Fatalf("wire lost positive evidence: %s", raw)
	}
	if err := ex.Start(context.Background(), "fake"); err == nil {
		t.Fatal("cleanup record authorized activation")
	}
	if len(ex.engines) != 0 {
		t.Fatal("unsupported engine was adopted into activatable state")
	}
	// A further replacement can prove the saved PID exited without any adoption.
	restarted := NewExecutor(ex.reg, NewReporter(nil), nil, ex.baseDir)
	status, err = restarted.Status("fake")
	if err != nil || !status.TerminationVerified {
		t.Fatalf("second replacement status=%+v err=%v", status, err)
	}
}

func TestSleepUnsupportedRefusesUncertainEvidence(t *testing.T) {
	for _, scenario := range []string{"missing record", "spawn window", "foreign image", "external install", "closed endpoint surviving PID"} {
		t.Run(scenario, func(t *testing.T) {
			ex, bin, port := unsupportedStopFixture(t)
			evidence := processStopEvidence{Bin: bin, Port: port, Address: "127.0.0.1:" + strconv.Itoa(port)}
			var survivingPID int
			switch scenario {
			case "foreign image", "external install":
				foreign := filepath.Join(t.TempDir(), testbin.Exe("foreignengine"))
				copyFile(t, fakeEngineBin, foreign)
				child := spawnFakeListener(t, foreign, port)
				survivingPID = child.Process.Pid
				evidence.PID = survivingPID
				if scenario == "external install" {
					evidence.Bin = foreign
				}
			case "closed endpoint surviving PID":
				evidence.PID = os.Getpid()
			}
			if scenario != "missing record" {
				if err := ex.desired.setProcessStop("fake", evidence); err != nil {
					t.Fatal(err)
				}
			}
			if scenario == "closed endpoint surviving PID" {
				if _, err := ex.unsupportedStopState("fake"); err != nil {
					t.Fatal(err)
				}
				if err := ex.confirmOrphanStopped("fake", evidence.PID, port, Runtime{}); err == nil {
					t.Fatal("surviving PID did not establish failed reclaim fence")
				}
			}
			if err := ex.Sleep("fake"); err == nil {
				t.Fatal("uncertain Sleep succeeded")
			}
			for attempt := 0; attempt < 2; attempt++ {
				status, err := ex.Status("fake")
				if err != nil || status.TerminationVerified || status.Running {
					t.Fatalf("passive status=%+v err=%v", status, err)
				}
				if got := ex.GetInstalled(); len(got) != 1 || got[0].TerminationVerified {
					t.Fatalf("passive inventory=%+v", got)
				}
			}
			if survivingPID > 0 && !pidAlive(survivingPID) {
				t.Fatal("cleanup killed an external process")
			}
			if len(ex.engines) != 0 {
				t.Fatal("passive read adopted unsupported engine")
			}
		})
	}
}

func TestProcessLaunchPersistsStopIdentity(t *testing.T) {
	for _, host := range []string{"127.0.0.1", ""} {
		name := "explicit loopback host"
		if host == "" {
			name = "empty TCP host"
		}
		t.Run(name, func(t *testing.T) {
			m := testEngineManifest(fakeEngineBin)
			platform := m.Platforms[hostKey()]
			platform.Runtime.Ready = &Probe{TCP: host + ":{port}"}
			m.Platforms[hostKey()] = platform
			ex := desiredTestExecutor(t, m, t.TempDir())
			other := testEngineManifest(fakeEngineBin)
			other.Engine = "other"
			ex.reg.engines["other"] = other
			if err := ex.Start(context.Background(), "fake"); err != nil {
				t.Fatal(err)
			}
			st, err := ex.state("fake")
			if err != nil {
				t.Fatal(err)
			}
			st.mu.Lock()
			port, pid := st.port, st.proc.cmd.Process.Pid
			st.mu.Unlock()
			evidence, known, err := ex.desired.processStop("fake")
			if err != nil || !known || evidence.Bin != fakeEngineBin || evidence.Port != port || evidence.Address != host+":"+strconv.Itoa(port) || evidence.PID != pid {
				t.Fatalf("saved stop identity=%+v known=%v err=%v; want port=%d pid=%d", evidence, known, err, port, pid)
			}
			done, err := ex.admitMutation("other")
			if err != nil {
				t.Fatalf("another engine's mutation was poisoned: %v", err)
			}
			done()
		})
	}
}

func TestProcessStopEvidenceRejectsBeforeWrite(t *testing.T) {
	store := newDesiredStateStore(t.TempDir())
	if err := store.set("other", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	if err := store.setProcessStop("fake", processStopEvidence{Bin: fakeEngineBin, Port: 12345, Address: "malformed"}); err == nil {
		t.Fatal("invalid stop record was saved")
	}
	if enabled, known, err := store.get("other"); err != nil || !known || !enabled {
		t.Fatalf("invalid record poisoned existing state: enabled=%v known=%v err=%v", enabled, known, err)
	}
}

func TestProcessStopEvidenceWriteFailureLeavesNoOwnedProcess(t *testing.T) {
	for _, tc := range []struct {
		name      string
		failWrite int
	}{{"before spawn", 1}, {"after spawn", 2}} {
		t.Run(tc.name, func(t *testing.T) {
			port, err := freePort()
			if err != nil {
				t.Fatal(err)
			}
			m := testEngineManifest(fakeEngineBin)
			platform := m.Platforms[hostKey()]
			platform.Runtime.Port = port
			m.Platforms[hostKey()] = platform
			ex := desiredTestExecutor(t, m, t.TempDir())
			failure := errors.New("stop evidence write failed")
			writes := 0
			useEngineFileReplacer(t, durablefile.Replacer{CreateTemp: func(dir, base string, _ os.FileMode) (durablefile.File, error) {
				writes++
				if writes == tc.failWrite {
					return nil, failure
				}
				return os.CreateTemp(dir, base+".*.tmp")
			}})
			if err := ex.Start(context.Background(), "fake"); !errors.Is(err, failure) {
				t.Fatalf("start error=%v", err)
			}
			st, err := ex.state("fake")
			if err != nil {
				t.Fatal(err)
			}
			st.mu.Lock()
			owned, running := st.proc != nil, st.running
			st.mu.Unlock()
			if owned || running || portServing(port) {
				t.Fatalf("failed evidence write left owned=%v running=%v listener=%v", owned, running, portServing(port))
			}
		})
	}
}

// intentRecorder captures engine:intent-changed payloads.
type intentRecorder struct {
	mu   sync.Mutex
	seen []engineintent.Snapshot
}

func (r *intentRecorder) emit(method string, params any) {
	if method != intentChangedMethod {
		return
	}
	p, ok := params.(engineintent.Snapshot)
	if !ok {
		panic("intent notification must use engineintent.Snapshot")
	}
	r.mu.Lock()
	r.seen = append(r.seen, p)
	r.mu.Unlock()
}

func (r *intentRecorder) last() (engineintent.Snapshot, int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.seen) == 0 {
		return engineintent.Snapshot{}, 0
	}
	return r.seen[len(r.seen)-1], len(r.seen)
}

func TestIntentAndNotificationOffOrigins(t *testing.T) {
	test := func(name, seed string, changes []struct {
		enabled bool
		origin  intentOrigin
	}, wantEnabled bool, wantOrigin map[string]engineintent.Origin) {
		t.Run(name, func(t *testing.T) {
			rec := &intentRecorder{}
			e := wakeTestExecutor(t, t.TempDir(), rec)
			if seed != "" {
				if err := os.WriteFile(e.desired.path, []byte(seed), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			for _, change := range changes {
				if err := e.setDesiredEnabled("fake", change.enabled, change.origin); err != nil {
					t.Fatal(err)
				}
			}
			e.emitIntent()
			got, err := e.Intent()
			if err != nil {
				t.Fatal(err)
			}
			want := engineintent.Snapshot{EnabledByEngine: map[string]bool{"fake": wantEnabled}, OffByEngine: wantOrigin}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Intent = %+v, want %+v", got, want)
			}
			pushed, count := rec.last()
			if count != len(changes)+1 || !reflect.DeepEqual(pushed, want) {
				t.Fatalf("push = %+v (%d), want %+v (%d)", pushed, count, want, len(changes)+1)
			}
			data, err := json.Marshal(pushed)
			if err != nil {
				t.Fatal(err)
			}
			var wire engineintent.Snapshot
			if err := json.Unmarshal(data, &wire); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(wire, want) {
				t.Fatalf("wire = %+v, want %+v", wire, want)
			}
		})
	}
	type change = struct {
		enabled bool
		origin  intentOrigin
	}
	test("explicit local Off", "", []change{{false, intentLocal}}, false, map[string]engineintent.Origin{"fake": "local"})
	test("explicit peer Off", "", []change{{false, intentPeer}}, false, map[string]engineintent.Origin{"fake": "peer"})
	test("peer cannot downgrade local Off", "", []change{{false, intentLocal}, {false, intentPeer}}, false, map[string]engineintent.Origin{"fake": "local"})
	test("peer Off preserves peer origin", "", []change{{false, intentPeer}, {false, intentPeer}}, false, map[string]engineintent.Origin{"fake": "peer"})
	test("On clears Off origin", "", []change{{false, intentPeer}, {true, intentLocal}}, true, map[string]engineintent.Origin{})
	test("unknown has no origin", "", nil, false, map[string]engineintent.Origin{})
	test("legacy Off projects local", `{"engines":{"fake":false}}`, nil, false, map[string]engineintent.Origin{"fake": "local"})
	test("legacy Off survives peer Off", `{"engines":{"fake":false}}`, []change{{false, intentPeer}}, false, map[string]engineintent.Origin{"fake": "local"})
	test("stale origins excluded", `{"engines":{"gone":false},"offBy":{"gone":"peer","fake":"local"}}`, nil, false, map[string]engineintent.Origin{})
	test("On ignores stale origin", `{"engines":{"fake":true},"offBy":{"fake":"peer"}}`, nil, true, map[string]engineintent.Origin{})
}

func TestIntentReadErrorsSuppressNotification(t *testing.T) {
	test := func(name, content string, directory bool) {
		t.Run(name, func(t *testing.T) {
			rec := &intentRecorder{}
			e := wakeTestExecutor(t, t.TempDir(), rec)
			if directory {
				if err := os.Mkdir(e.desired.path, 0o700); err != nil {
					t.Fatal(err)
				}
			} else if err := os.WriteFile(e.desired.path, []byte(content), 0o600); err != nil {
				t.Fatal(err)
			}
			if got, err := e.Intent(); err == nil {
				t.Fatalf("Intent accepted unreadable/invalid state: %+v", got)
			}
			e.emitIntent()
			if _, count := rec.last(); count != 0 {
				t.Fatalf("invalid intent emitted %d notifications", count)
			}
		})
	}
	test("read failure", "", true)
	test("decode failure", "{", false)
	test("invalid saved origin", `{"engines":{"fake":false},"offBy":{"fake":"remote"}}`, false)
	test("empty saved origin", `{"engines":{"fake":false},"offBy":{"fake":""}}`, false)
}

func TestIntentDoesNotAcquireLifecycleLock(t *testing.T) {
	e := wakeTestExecutor(t, t.TempDir(), nil)
	st, err := e.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.opMu.Lock()
	done := make(chan error, 1)
	go func() {
		_, err := e.Intent()
		done <- err
	}()
	select {
	case err := <-done:
		st.opMu.Unlock()
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		st.opMu.Unlock()
		<-done
		t.Fatal("Intent waited for the engine lifecycle lock")
	}
}

func wakeTestExecutor(t *testing.T, baseDir string, rec *intentRecorder) *Executor {
	t.Helper()
	reg := NewRegistry()
	m := testEngineManifest(fakeEngineBin)
	reg.engines[m.Engine] = m
	emit := func(string, any) {}
	if rec != nil {
		emit = rec.emit
	}
	e := NewExecutor(reg, NewReporter(nil), emit, baseDir)
	t.Cleanup(e.StopAll)
	return e
}

func TestWakeRefusesUnknownAndOffIntent(t *testing.T) {
	e := wakeTestExecutor(t, t.TempDir(), nil)
	if err := e.Wake(context.Background(), "fake"); !errors.Is(err, errIntentOff) {
		t.Fatalf("wake with no saved intent = %v, want errIntentOff", err)
	}
	if err := e.setDesiredEnabled("fake", false, intentLocal); err != nil {
		t.Fatal(err)
	}
	if err := e.Wake(context.Background(), "fake"); !errors.Is(err, errIntentOff) {
		t.Fatalf("wake with intent Off = %v, want errIntentOff", err)
	}
	if st, _ := e.Status("fake"); st.Running {
		t.Fatal("wake started an engine saved Off")
	}
}

func TestSleepAndWakeKeepSavedIntent(t *testing.T) {
	baseDir := t.TempDir()
	rec := &intentRecorder{}
	e := wakeTestExecutor(t, baseDir, rec)
	ctx := context.Background()
	if err := e.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	if intent, n := rec.last(); n == 0 || !intent.EnabledByEngine["fake"] {
		t.Fatalf("start did not push intent On: %+v (%d pushes)", intent, n)
	}
	_, pushesBefore := rec.last()

	if err := e.Sleep("fake"); err != nil {
		t.Fatalf("sleep: %v", err)
	}
	if st, _ := e.Status("fake"); st.Running {
		t.Fatal("sleep left the engine running")
	}
	if enabled, known, err := e.desired.get("fake"); err != nil || !known || !enabled {
		t.Fatalf("intent after sleep = (%v, %v, %v), want known On", enabled, known, err)
	}

	if err := e.Wake(ctx, "fake"); err != nil {
		t.Fatalf("wake: %v", err)
	}
	if st, _ := e.Status("fake"); !st.Running {
		t.Fatal("wake did not start an engine saved On")
	}
	if _, n := rec.last(); n != pushesBefore {
		t.Fatalf("sleep/wake pushed intent changes (%d → %d); they must be intent-neutral", pushesBefore, n)
	}
	intent, err := e.Intent()
	if err != nil || !intent.EnabledByEngine["fake"] {
		t.Fatalf("Intent() = %+v, %v", intent, err)
	}

	if err := e.Stop("fake"); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if intent, _ := rec.last(); intent.EnabledByEngine["fake"] {
		t.Fatal("explicit stop did not push intent Off")
	}
}

func TestWakeRefusedAfterShutdown(t *testing.T) {
	e := wakeTestExecutor(t, t.TempDir(), nil)
	if err := e.setDesiredEnabled("fake", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	e.StopAll()
	if err := e.Wake(context.Background(), "fake"); err == nil {
		t.Fatal("wake succeeded after StopAll")
	}
	if st, _ := e.Status("fake"); st.Running {
		t.Fatal("wake started an engine during shutdown")
	}
}
