// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/testbin"
)

// Each case owns its executable copy so a previous process's mapped image
// cannot block the next copy on Windows. All fixture listeners use loopback.
func restoreOrphanManifest(t *testing.T) (*Manifest, string, int) {
	t.Helper()
	dir := t.TempDir()
	bin := filepath.Join(dir, testbin.Exe("fakeorphan"))
	copyFile(t, fakeEngineBin, bin)
	port, err := freePort()
	if err != nil {
		t.Fatalf("reserve orphan endpoint: %v", err)
	}
	m := testEngineManifest(bin)
	p := m.Platforms[hostKey()]
	p.Runtime.Port = port
	p.Runtime.Health = nil
	p.Runtime.Stop = &StopSpec{Signal: "kill"}
	m.Platforms[hostKey()] = p
	return m, dir, port
}

func restoreOrphanExecutor(t *testing.T, m *Manifest, installDir string) *Executor {
	t.Helper()
	ex := desiredTestExecutor(t, m, t.TempDir())
	st, err := ex.state(m.Engine)
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.installDir = installDir
	st.mu.Unlock()
	return ex
}

func spawnUnboundOrphan(t *testing.T, bin string, port int) *managedProc {
	t.Helper()
	started := make(chan struct{})
	orphan, err := startManagedProc(bin, nil, map[string]string{
		"FAKE_ADDR":        "127.0.0.1:" + strconv.Itoa(port),
		"FAKE_START_DELAY": "1h",
		"FAKE_OUTPUT":      "unbound orphan started",
	}, func(stream, line string) {
		if stream == "stdout" && line == "unbound orphan started" {
			close(started)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { orphan.stop(0) })
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("delayed orphan did not start")
	}
	return orphan
}

func TestRestoreReclaimsSavedPIDBeforeOrphanBinds(t *testing.T) {
	m, dir, port := restoreOrphanManifest(t)
	bin := m.Platforms[hostKey()].Runtime.Bin
	orphan := spawnUnboundOrphan(t, bin, port)
	if got := probeListener(context.Background(), m.Platforms[hostKey()].Runtime.Ready, port); got != listenerProbeRefused {
		t.Fatalf("delayed orphan endpoint = %v, want refused", got)
	}
	ex := restoreOrphanExecutor(t, m, dir)
	if err := ex.desired.setProcessStop("fake", processStopEvidence{
		Bin: bin, PID: orphan.cmd.Process.Pid, Port: port, Address: "127.0.0.1:" + strconv.Itoa(port),
	}); err != nil {
		t.Fatal(err)
	}
	if err := ex.desired.set("fake", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	if err := ex.RestoreEnabled(context.Background()); err != nil {
		t.Fatalf("restore delayed orphan: %v", err)
	}
	if !orphanPIDExited(orphan.cmd.Process.Pid) {
		t.Fatal("cold start raced the still-live unbound orphan")
	}
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	proc, adopted, running := st.proc, st.adopted, st.running
	st.mu.Unlock()
	if proc == nil || proc == orphan || adopted || !running {
		t.Fatalf("restore must own a cold child: proc=%p orphan=%p adopted=%v running=%v", proc, orphan, adopted, running)
	}
}

func TestReclaimSavedPIDRequiresCurrentManagedImageAndEndpoint(t *testing.T) {
	for _, scenario := range []string{"foreign current image", "different recorded image", "different saved port", "external install", "unknown PID"} {
		t.Run(scenario, func(t *testing.T) {
			m, dir, port := restoreOrphanManifest(t)
			bin := m.Platforms[hostKey()].Runtime.Bin
			orphanBin := bin
			if scenario == "foreign current image" {
				orphanBin = fakeEngineBin
			}
			orphan := spawnUnboundOrphan(t, orphanBin, port)
			evidence := processStopEvidence{
				Bin: bin, PID: orphan.cmd.Process.Pid, Port: port, Address: "127.0.0.1:" + strconv.Itoa(port),
			}
			switch scenario {
			case "different recorded image":
				evidence.Bin = fakeEngineBin
			case "different saved port":
				evidence.Port = port%65535 + 1
			case "external install":
				dir = t.TempDir()
			case "unknown PID":
				evidence.PID = 0
			}
			ex := restoreOrphanExecutor(t, m, dir)
			if installed, err := ex.Detect("fake"); err != nil || !installed {
				t.Fatalf("detect fixture: installed=%v err=%v", installed, err)
			}
			if err := ex.desired.setProcessStop("fake", evidence); err != nil {
				t.Fatal(err)
			}
			st, err := ex.state("fake")
			if err != nil {
				t.Fatal(err)
			}
			st.opMu.Lock()
			reclaimed, err := ex.reclaimOrphan(context.Background(), st, "fake", port)
			st.opMu.Unlock()
			if err != nil || reclaimed || !pidAlive(orphan.cmd.Process.Pid) {
				t.Fatalf("nonmatching saved PID was reclaimed: reclaimed=%v err=%v alive=%v", reclaimed, err, pidAlive(orphan.cmd.Process.Pid))
			}
		})
	}
}

func TestOrphanReclaimFailureRetainsFenceWhilePIDSurvivesClosedListener(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(server.Close)
	_, rawPort, err := net.SplitHostPort(server.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(rawPort)
	if err != nil {
		t.Fatal(err)
	}
	m := testEngineManifest(fakeEngineBin)
	p := m.Platforms[hostKey()]
	p.Runtime.Port = port
	p.Runtime.Health = nil
	m.Platforms[hostKey()] = p
	ex := desiredTestExecutor(t, m, t.TempDir())
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	// No signal is sent: the confirmation helper safely simulates a survivor
	// using the test PID, while the loopback server supplies positive readiness.
	failure := ex.confirmOrphanStopped("fake", os.Getpid(), port, Runtime{})
	if failure == nil || !strings.Contains(failure.Error(), "did not stop") {
		t.Fatalf("confirm surviving orphan = %v", failure)
	}
	for attempt := 0; attempt < 3; attempt++ {
		status, err := ex.Status("fake")
		if err != nil {
			t.Fatal(err)
		}
		if status.Running || status.Healthy {
			t.Errorf("status poll %d adopted failed reclaim: %+v", attempt, status)
		}
		for _, installed := range ex.GetInstalled() {
			if installed.Engine == "fake" && (installed.Running || installed.Healthy) {
				t.Errorf("installed poll %d adopted failed reclaim: %+v", attempt, installed)
			}
		}
		st.opMu.Lock()
		ex.reconcilePresence(context.Background(), "fake", st, true, port, false)
		st.opMu.Unlock()
		st.mu.Lock()
		running, adopted := st.running, st.adopted
		st.mu.Unlock()
		if running || adopted {
			t.Errorf("presence poll %d adopted failed reclaim: running=%v adopted=%v", attempt, running, adopted)
		}
	}
	reported := ex.Errors()
	if len(reported) != 1 || reported[0].ID != startFailedID("fake") || reported[0].Message != failure.Error() {
		t.Errorf("failed reclaim report changed during passive polls: %+v", reported)
	}
	if err := ex.Start(context.Background(), "fake"); err == nil {
		t.Fatal("explicit Start adopted the surviving failed reclaim")
	}
	if err := ex.Stop("fake"); err == nil {
		t.Fatal("Stop reported success while the unowned listener survived")
	}
	if err := ex.Start(context.Background(), "fake"); err == nil {
		t.Fatal("failed Stop cleared the reclaim adoption fence")
	}
	status, err := ex.Status("fake")
	if err != nil || status.Running || status.Healthy {
		t.Fatalf("failed Stop changed fenced status: %+v err=%v", status, err)
	}
	server.Close()
	if err := ex.Stop("fake"); err == nil {
		t.Fatal("listener refusal cleared fence while recorded PID is still alive")
	}
	if ex.terminationVerified(st) {
		t.Fatal("closed listener reported verified termination while failed-reclaim PID survives")
	}
	if err := ex.Start(context.Background(), "fake"); err == nil {
		t.Fatal("closed listener permitted a cold start while failed-reclaim PID survives")
	}
	st.mu.Lock()
	fenced, recordedPID, proc := st.reclaimFailed, st.reclaimPID, st.proc
	st.mu.Unlock()
	if !fenced || recordedPID != os.Getpid() || proc != nil || !pidAlive(recordedPID) {
		t.Fatalf("closed listener lost survivor evidence: fenced=%v pid=%d owned=%v", fenced, recordedPID, proc != nil)
	}
}

func TestOrphanReclaimFenceClearsAfterVerifiedChildExit(t *testing.T) {
	m, dir, port := restoreOrphanManifest(t)
	child := spawnFakeListener(t, m.Platforms[hostKey()].Runtime.Bin, port)
	ex := restoreOrphanExecutor(t, m, dir)
	if installed, err := ex.Detect("fake"); err != nil || !installed {
		t.Fatalf("detect managed child: installed=%v err=%v", installed, err)
	}
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	if err := ex.confirmOrphanStopped("fake", child.Process.Pid, port, Runtime{}); err == nil {
		t.Fatal("live child did not establish failed reclaim")
	}
	if err := ex.Stop("fake"); err != nil {
		t.Fatalf("precisely matched child stop: %v", err)
	}
	if !orphanPIDExited(child.Process.Pid) {
		t.Fatal("stop succeeded without positive child exit evidence")
	}
	st.mu.Lock()
	fenced, recordedPID := st.reclaimFailed, st.reclaimPID
	st.mu.Unlock()
	if fenced || recordedPID != 0 {
		t.Fatalf("verified stop retained fence: fenced=%v pid=%d", fenced, recordedPID)
	}
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("explicit retry after verified child exit: %v", err)
	}
	status, err := ex.Status("fake")
	if err != nil || !status.Running || !status.Healthy {
		t.Fatalf("successful retry remains blocked: %+v err=%v", status, err)
	}
	st.mu.Lock()
	owned := st.proc != nil && !st.adopted
	st.mu.Unlock()
	if !owned {
		t.Fatal("retry must cold-launch an owned child")
	}
	for _, report := range ex.Errors() {
		if report.ID == startFailedID("fake") {
			t.Errorf("successful retry retained reclaim failure: %+v", report)
		}
	}
}

func TestOrphanReclaimFenceClearsAfterColdStartWithoutStop(t *testing.T) {
	m, dir, port := restoreOrphanManifest(t)
	child := spawnFakeListener(t, m.Platforms[hostKey()].Runtime.Bin, port)
	ex := restoreOrphanExecutor(t, m, dir)
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	if err := ex.confirmOrphanStopped("fake", child.Process.Pid, port, Runtime{}); err == nil {
		t.Fatal("live child did not establish failed reclaim")
	}
	// The orphan exits independently of the manager. No Stop or restore reclaim
	// may clear the fence before the explicit cold Start commits readiness.
	if err := child.Process.Kill(); err != nil {
		t.Fatalf("kill fixture orphan: %v", err)
	}
	waitFor(t, 3*time.Second, func() bool { return orphanPIDExited(child.Process.Pid) })
	status, err := ex.Status("fake")
	if err != nil || status.Running || status.Healthy {
		t.Fatalf("status after independent orphan exit: %+v err=%v", status, err)
	}
	st.mu.Lock()
	fenced, recordedPID := st.reclaimFailed, st.reclaimPID
	st.mu.Unlock()
	if !fenced || recordedPID != child.Process.Pid {
		t.Fatalf("passive status cleared fence before start: fenced=%v pid=%d", fenced, recordedPID)
	}
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("cold start after independent orphan exit: %v", err)
	}
	st.mu.Lock()
	fenced, recordedPID, proc, adopted := st.reclaimFailed, st.reclaimPID, st.proc, st.adopted
	st.mu.Unlock()
	if fenced || recordedPID != 0 || proc == nil || adopted {
		t.Fatalf("cold start retained fence or adopted: fenced=%v pid=%d owned=%v adopted=%v", fenced, recordedPID, proc != nil, adopted)
	}
	if proc.cmd.Process.Pid == child.Process.Pid {
		t.Fatal("cold start reused the orphan PID")
	}
	status, err = ex.Status("fake")
	if err != nil || !status.Running || !status.Healthy {
		t.Fatalf("cold start remains blocked: %+v err=%v", status, err)
	}
	for _, report := range ex.Errors() {
		if report.ID == startFailedID("fake") {
			t.Errorf("cold start retained reclaim failure: %+v", report)
		}
	}
}

func TestRestoreCancellationAfterVerifiedReclaimAllowsColdRetry(t *testing.T) {
	m, dir, port := restoreOrphanManifest(t)
	orphan := spawnFakeListener(t, m.Platforms[hostKey()].Runtime.Bin, port)
	ex := restoreOrphanExecutor(t, m, dir)
	if err := ex.desired.set("fake", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cancelledAfterStop := false
	ex.emit = func(method string, _ any) {
		if method != "engine:state-changed" || cancelledAfterStop {
			return
		}
		// markStopped notifies after both PID exit and listener refusal have
		// been verified, before doStart's cancellation check and cold launch.
		if !orphanPIDExited(orphan.Process.Pid) || portServing(port) {
			t.Fatal("stopped notification preceded verified orphan termination")
		}
		cancelledAfterStop = true
		cancel()
	}
	if err := ex.restoreEnabled(ctx, "fake"); !errors.Is(err, context.Canceled) {
		t.Fatalf("restore cancelled after verified reclaim = %v", err)
	}
	if !cancelledAfterStop || !orphanPIDExited(orphan.Process.Pid) || portServing(port) {
		t.Fatal("cancellation did not follow verified orphan termination")
	}
	st.mu.Lock()
	fenced, pid, proc, running, adopted, generation := st.reclaimFailed, st.reclaimPID, st.proc, st.running, st.adopted, st.gen
	st.mu.Unlock()
	if fenced || pid != 0 || proc != nil || running || adopted || generation != 0 {
		t.Fatalf("cancelled restore lost verified stop or launched: fenced=%v pid=%d owned=%v running=%v adopted=%v generation=%d", fenced, pid, proc != nil, running, adopted, generation)
	}
	if enabled, known, err := ex.desired.get("fake"); err != nil || !known || !enabled {
		t.Fatalf("cancellation changed desired On: enabled=%v known=%v err=%v", enabled, known, err)
	}
	if err := ex.restoreEnabled(context.Background(), "fake"); err != nil {
		t.Fatalf("cold restore retry after verified reclaim: %v", err)
	}
	st.mu.Lock()
	proc, running, adopted = st.proc, st.running, st.adopted
	fenced, pid = st.reclaimFailed, st.reclaimPID
	st.mu.Unlock()
	if proc == nil || !running || adopted || fenced || pid != 0 {
		t.Fatalf("retry did not recover cold: owned=%v running=%v adopted=%v fenced=%v pid=%d", proc != nil, running, adopted, fenced, pid)
	}
	if proc.cmd.Process.Pid == orphan.Process.Pid {
		t.Fatal("cold restore retry reused the reclaimed PID")
	}
}

func TestRestoreRejectsIdentifiedReplacementAfterSuccessfulReclaim(t *testing.T) {
	m, dir, port := restoreOrphanManifest(t)
	orphan := spawnFakeListener(t, m.Platforms[hostKey()].Runtime.Bin, port)
	ex := restoreOrphanExecutor(t, m, dir)
	if err := ex.desired.set("fake", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	foreignBin := filepath.Join(testbin.Dir(t, "nvpair-engine-manager", "stop-foreign", "foreignengine"), testbin.Exe("foreignengine"))
	copyFile(t, fakeEngineBin, foreignBin)
	foreignPID := 0
	ex.emit = func(method string, _ any) {
		if method != "engine:state-changed" || foreignPID != 0 {
			return
		}
		// markStopped emits after confirmed reclaim and before the next presence
		// check. The callback supplies a deterministic identified replacement.
		if !orphanPIDExited(orphan.Process.Pid) {
			t.Fatal("replacement callback ran before orphan exit")
		}
		foreign := spawnFakeListener(t, foreignBin, port)
		foreignPID = foreign.Process.Pid
	}
	err := ex.restoreEnabled(context.Background(), "fake")
	if err == nil || err.Error() != "cannot start owned engine: another process already owns the server port" {
		t.Fatalf("restore with post-reclaim replacement = %v", err)
	}
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	owned := st.proc != nil
	st.mu.Unlock()
	if foreignPID == 0 || !pidAlive(foreignPID) || !portServing(port) || owned {
		t.Fatalf("replacement changed: pid=%d alive=%v serving=%v owned=%v", foreignPID, pidAlive(foreignPID), portServing(port), owned)
	}
}
