// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"testing"

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
	if err := store.setAllocatedPort("other", 23456); err != nil {
		t.Fatal(err)
	}
	if err := store.setProcessStop("fake", processStopEvidence{Bin: fakeEngineBin, Port: 12345, Address: "malformed"}); err == nil {
		t.Fatal("invalid stop record was saved")
	}
	if port, err := store.allocatedPort("other"); err != nil || port != 23456 {
		t.Fatalf("existing port=%d err=%v", port, err)
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

func wakeTestExecutor(t *testing.T, baseDir string, _ any) *Executor {
	t.Helper()
	reg := NewRegistry()
	m := testEngineManifest(fakeEngineBin)
	reg.engines[m.Engine] = m
	emit := func(string, any) {}
	e := NewExecutor(reg, NewReporter(nil), emit, baseDir)
	t.Cleanup(e.StopAll)
	return e
}

func TestWakeRefusedAfterShutdown(t *testing.T) {
	e := wakeTestExecutor(t, t.TempDir(), nil)
	e.StopAll()
	if err := e.Wake(context.Background(), "fake"); err == nil {
		t.Fatal("wake succeeded after StopAll")
	}
	if st, _ := e.Status("fake"); st.Running {
		t.Fatal("wake started an engine during shutdown")
	}
}
