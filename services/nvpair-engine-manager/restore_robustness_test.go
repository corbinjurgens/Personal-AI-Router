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

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/testbin"
)

// Reuse the registered, fixed-path orphan fixture; these cases must remain
// serial because the executable paths are shared with the stop tests.
func restoreOrphanManifest(t *testing.T) (*Manifest, string, int) {
	t.Helper()
	dir := testbin.Dir(t, "nvpair-engine-manager", "stop-orphan", "fakeorphan")
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

func TestSwitchRecoveryReclaimsOnlyManagedDurableTarget(t *testing.T) {
	for _, scenario := range []string{"healthy managed orphan", "unhealthy managed orphan", "foreign image", "external install", "different port"} {
		t.Run(scenario, func(t *testing.T) {
			m, managedDir, port := restoreOrphanManifest(t)
			ex, p := switchFixture(t)
			m.Engine = "target"
			platform := m.Platforms[hostKey()]
			// The replacement manager must recover the endpoint from the durable
			// checkpoint, not from a configured port or predecessor's memory.
			platform.Runtime.Port = 0
			if scenario == "unhealthy managed orphan" {
				platform.Runtime.Ready.Status = http.StatusCreated
			}
			m.Platforms[hostKey()] = platform
			ex.reg.engines["target"] = m
			p = prepareSwitch(t, ex, p)
			stopped := switchStep(t, ex, p, settings.SwitchStepSourceStop)
			target, err := ex.state("target")
			if err != nil {
				t.Fatal(err)
			}
			target.mu.Lock()
			target.port = port
			target.mu.Unlock()
			cp := stopped.Checkpoint
			cp.Phase = settings.SwitchCheckpointStartPending
			if err := ex.saveSwitch(cp, "", false); err != nil {
				t.Fatalf("checkpoint interrupted target start: %v", err)
			}
			bin := platform.Runtime.Bin
			listenerPort := port
			if scenario == "foreign image" {
				bin = filepath.Join(testbin.Dir(t, "nvpair-engine-manager", "stop-foreign", "foreignengine"), testbin.Exe("foreignengine"))
				copyFile(t, fakeEngineBin, bin)
			}
			if scenario == "different port" {
				listenerPort, err = freePort()
				if err != nil {
					t.Fatal(err)
				}
			}
			orphan := spawnFakeListener(t, bin, listenerPort)
			restarted := NewExecutor(ex.reg, NewReporter(nil), nil, ex.baseDir)
			t.Cleanup(restarted.StopAll)
			state, err := restarted.state("target")
			if err != nil {
				t.Fatal(err)
			}
			state.mu.Lock()
			if scenario != "external install" {
				state.installDir = managedDir
			}
			state.mu.Unlock()
			if err := restarted.RestoreEnabled(context.Background()); err != nil {
				t.Fatalf("restore while switch reserved: %v", err)
			}
			if !pidAlive(orphan.Process.Pid) {
				t.Fatal("automatic restore touched reserved target before reconcile")
			}
			result := restarted.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
			if result.Error == nil || result.Checkpoint == nil || result.Checkpoint.TargetCommitted {
				t.Fatalf("interrupted target recovery = %+v", result)
			}
			owned := scenario == "healthy managed orphan" || scenario == "unhealthy managed orphan"
			if alive := pidAlive(orphan.Process.Pid); alive == owned {
				t.Errorf("orphan alive=%v, want %v after durable endpoint cleanup", alive, !owned)
			}
			if result.Observed.Target.TerminationVerified != (owned || scenario == "different port") {
				t.Errorf("target termination verified=%v in %s", result.Observed.Target.TerminationVerified, scenario)
			}
			saved, err := restarted.desired.load()
			if err != nil {
				t.Fatal(err)
			}
			if saved.Engines["source"] || saved.Engines["target"] || result.Observed.Target.Intent != settings.SwitchIntentOff {
				t.Fatalf("recovery changed source/target Off intent: saved=%+v observed=%+v", saved, result.Observed)
			}
			if source, err := restarted.Status("source"); err != nil || source.Running {
				t.Fatalf("source automatically rolled back: %+v err=%v", source, err)
			}
			state.mu.Lock()
			proc, recoveredPort := state.proc, state.port
			state.mu.Unlock()
			if proc != nil || recoveredPort != port {
				t.Fatalf("reconcile launched a target or lost durable endpoint: owned=%v port=%d want=%d", proc != nil, recoveredPort, port)
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
	if ex.switchTerminated(st) {
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

func TestCancelledSwitchRecoveryPreservesStatusAdoptedPendingOrphan(t *testing.T) {
	m, dir, port := restoreOrphanManifest(t)
	ex, p := switchFixture(t)
	m.Engine = "target"
	ex.reg.engines["target"] = m
	p = prepareSwitch(t, ex, p)
	cp := switchStep(t, ex, p, settings.SwitchStepSourceStop).Checkpoint
	cp.Phase = settings.SwitchCheckpointStartPending
	if err := ex.saveSwitch(cp, "", false); err != nil {
		t.Fatal(err)
	}
	orphan := spawnFakeListener(t, m.Platforms[hostKey()].Runtime.Bin, port)
	restarted := NewExecutor(ex.reg, NewReporter(nil), nil, ex.baseDir)
	t.Cleanup(restarted.StopAll)
	st, err := restarted.state("target")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.installDir = dir
	st.mu.Unlock()
	status, err := restarted.Status("target")
	if err != nil || !status.Running || !status.Healthy {
		t.Fatalf("passive pre-adoption: %+v err=%v", status, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result := restarted.SwitchReconcile(ctx, settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
	if result.Error == nil || result.Checkpoint == nil || result.Checkpoint.TargetCommitted || result.Observed.Target.TerminationVerified {
		t.Fatalf("cancelled pending recovery = %+v", result)
	}
	st.mu.Lock()
	proc := st.proc
	st.mu.Unlock()
	if !pidAlive(orphan.Process.Pid) || !portServing(port) || proc != nil {
		t.Fatalf("cancelled recovery killed or replaced passive orphan: alive=%v serving=%v owned=%v", pidAlive(orphan.Process.Pid), portServing(port), proc != nil)
	}
	saved, err := restarted.desired.load()
	if err != nil {
		t.Fatal(err)
	}
	if saved.Engines["source"] || saved.Engines["target"] {
		t.Fatalf("cancelled cleanup changed Off intent: %+v", saved)
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
	if err == nil || !strings.Contains(err.Error(), "another process already owns the server port") {
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

func TestCommittedStrataRecoveryPreparesColdRuntimeBeforeOperationLock(t *testing.T) {
	f := newStrataServingFixture(t)
	preview := approveLlamaServing(t, f.llamaServingFixture)
	applyLlamaServing(t, f.llamaServingFixture, preview)
	st := settingsState(t, f.ex)
	source := testEngineManifest(fakeEngineBin)
	source.Engine = "source"
	f.ex.reg.engines[source.Engine] = source
	p := settings.SwitchPrepareRequest{
		OperationID: "switch-test", Caller: "local", Origin: settings.SwitchOriginLocal,
		SourceEngine: "source", TargetEngine: "fake",
		PayloadHash:        "sha256:0123456789012345678901234567890123456789012345678901234567890123",
		ManagerIncarnation: f.ex.switchIncarnation, Deadline: time.Now().Add(time.Minute).UTC(),
	}
	p = prepareSwitch(t, f.ex, p)
	cp := switchStep(t, f.ex, p, settings.SwitchStepSourceStop).Checkpoint
	if cp == nil {
		t.Fatal("source stop did not return a checkpoint")
	}
	cp.Phase = settings.SwitchCheckpointTargetCommitted
	cp.TargetCommitted = true
	cp.Target.Intent = settings.SwitchIntentOn
	if err := f.ex.saveSwitch(cp, "fake", true); err != nil {
		t.Fatal(err)
	}
	// Use the approved binding's exact native path: it is also the production
	// cache key, including Windows separators and drive-letter casing.
	path, wantHash := "", ""
	for _, artifact := range st.plat.Runtime.Serving.Models[0].Artifacts {
		if artifact.Role == "runtime:lib:attested" {
			path, wantHash = artifact.Binding.SourcePath, artifact.SHA256
		}
	}
	if path == "" || !filepath.IsAbs(path) || filepath.Clean(path) != path {
		t.Fatalf("fixture runtime binding has no canonical native path: %q", path)
	}
	if sum, err := strataRuntimeDigest(path); err != nil || sum != wantHash {
		t.Fatalf("fixture runtime cache key %q: digest=%q want=%q err=%v", path, sum, wantHash, err)
	}
	strataRuntimeHashes.Lock()
	delete(strataRuntimeHashes.files, path)
	_, cached := strataRuntimeHashes.files[path]
	strataRuntimeHashes.Unlock()
	if cached {
		t.Fatal("fixture runtime hash cache entry survived eviction")
	}
	if _, err := strataRuntimeDigest(path); err == nil {
		t.Fatal("fixture runtime hash is not cold")
	}
	// Holding opMu forbids both reclamation and launch. Preparation must still
	// finish, proving this cold-manager path hashes before entering that lock.
	st.opMu.Lock()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan settings.SwitchReconcileResult, 1)
	go func() {
		done <- f.ex.SwitchReconcile(ctx, settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
	}()
	deadline := time.Now().Add(3 * time.Second)
	var preparationErr error
	for {
		_, preparationErr = strataRuntimeDigest(path)
		if preparationErr == nil || time.Now().After(deadline) {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	strataRuntimeHashes.Lock()
	prepared, cached := strataRuntimeHashes.files[path]
	strataRuntimeHashes.Unlock()
	cancel()
	st.opMu.Unlock()
	select {
	case result := <-done:
		if result.Error == nil || !strings.Contains(result.Error.Message, "canceled") {
			t.Fatalf("cancelled prepared recovery: error=%+v checkpoint=%+v observed=%+v", result.Error, result.Checkpoint, result.Observed)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("cancelled recovery did not return")
	}
	if preparationErr != nil {
		t.Fatalf("committed recovery did not prepare cold hash outside opMu: %v", preparationErr)
	}
	if !cached || prepared.sum != wantHash {
		t.Fatalf("preparation barrier missed runtime cache entry %q: cached=%v digest=%q want=%q", path, cached, prepared.sum, wantHash)
	}
	command, err := launchForState(st, st.port)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := compileStrataServing(command, st.plat.Runtime, false); err != nil {
		t.Fatalf("prepared recovery runtime cannot compile: %v", err)
	}
	st.mu.Lock()
	owned := st.proc != nil
	st.mu.Unlock()
	if owned {
		t.Fatal("cancelled synthetic recovery launched a process")
	}
}

func TestSwitchRecoveryColdLaunchesCommittedOnTarget(t *testing.T) {
	m, managedDir, port := restoreOrphanManifest(t)
	ex, p := switchFixture(t)
	m.Engine = "target"
	platform := m.Platforms[hostKey()]
	platform.Runtime.Port = 0
	m.Platforms[hostKey()] = platform
	ex.reg.engines["target"] = m
	p = prepareSwitch(t, ex, p)
	cp := switchStep(t, ex, p, settings.SwitchStepSourceStop).Checkpoint
	target, err := ex.state("target")
	if err != nil {
		t.Fatal(err)
	}
	target.mu.Lock()
	target.port = port
	target.mu.Unlock()
	cp.Phase = settings.SwitchCheckpointTargetCommitted
	cp.TargetCommitted = true
	cp.Target.Intent = settings.SwitchIntentOn
	if err := ex.saveSwitch(cp, "target", true); err != nil {
		t.Fatalf("persist committed target On: %v", err)
	}
	orphan := spawnFakeListener(t, platform.Runtime.Bin, port)
	restarted := NewExecutor(ex.reg, NewReporter(nil), nil, ex.baseDir)
	t.Cleanup(restarted.StopAll)
	st, err := restarted.state("target")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.installDir = managedDir
	st.mu.Unlock()
	if err := restarted.RestoreEnabled(context.Background()); err != nil {
		t.Fatalf("restore reserved committed target: %v", err)
	}
	result := restarted.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
	if result.Error != nil || result.Checkpoint == nil || !result.Checkpoint.TargetCommitted || result.Observed.Target.Intent != settings.SwitchIntentOn {
		t.Fatalf("committed recovery = %+v", result)
	}
	st.mu.Lock()
	proc, adopted, running, recoveredPort := st.proc, st.adopted, st.running, st.port
	st.mu.Unlock()
	if proc == nil || adopted || !running || recoveredPort != port || pidAlive(orphan.Process.Pid) {
		t.Fatalf("committed recovery must cold-launch: owned=%v adopted=%v running=%v port=%d want=%d orphanAlive=%v", proc != nil, adopted, running, recoveredPort, port, pidAlive(orphan.Process.Pid))
	}
	if proc.cmd.Process.Pid == orphan.Process.Pid {
		t.Fatal("committed recovery reused orphan PID")
	}
	saved, err := restarted.desired.load()
	if err != nil {
		t.Fatal(err)
	}
	if saved.Engines["source"] || !saved.Engines["target"] || saved.PendingSwitch == nil || !saved.PendingSwitch.TargetCommitted {
		t.Fatalf("committed recovery changed durable intent/checkpoint: %+v", saved)
	}
	if source, err := restarted.Status("source"); err != nil || source.Running {
		t.Fatalf("committed recovery restored source: %+v err=%v", source, err)
	}
}

func TestSwitchTargetStartColdLaunchesMatchingOrphanAfterPrepare(t *testing.T) {
	m, managedDir, port := restoreOrphanManifest(t)
	ex, p := switchFixture(t)
	m.Engine = "target"
	ex.reg.engines["target"] = m
	st, err := ex.state("target")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.installDir = managedDir
	st.mu.Unlock()
	p = prepareSwitch(t, ex, p)
	switchStep(t, ex, p, settings.SwitchStepSourceStop)
	// Prepare saw a vacant endpoint. A managed orphan appears before the
	// forward start, so the start owner must reclaim instead of adopting it.
	orphan := spawnFakeListener(t, m.Platforms[hostKey()].Runtime.Bin, port)
	result := ex.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: settings.SwitchStepTargetStart})
	if result.Error != nil || result.Checkpoint == nil || !result.Checkpoint.TargetCommitted {
		t.Fatalf("fresh target start with matching orphan = %+v", result)
	}
	st.mu.Lock()
	proc, adopted, running := st.proc, st.adopted, st.running
	st.mu.Unlock()
	if proc == nil || adopted || !running || pidAlive(orphan.Process.Pid) {
		t.Fatalf("fresh target must cold-launch: owned=%v adopted=%v running=%v orphanAlive=%v", proc != nil, adopted, running, pidAlive(orphan.Process.Pid))
	}
}

func TestSwitchRecoveryCommittedOnPreservesForeignListener(t *testing.T) {
	m, managedDir, port := restoreOrphanManifest(t)
	ex, p := switchFixture(t)
	m.Engine = "target"
	ex.reg.engines["target"] = m
	p = prepareSwitch(t, ex, p)
	cp := switchStep(t, ex, p, settings.SwitchStepSourceStop).Checkpoint
	cp.Phase = settings.SwitchCheckpointTargetCommitted
	cp.TargetCommitted = true
	cp.Target.Intent = settings.SwitchIntentOn
	if err := ex.saveSwitch(cp, "target", true); err != nil {
		t.Fatal(err)
	}
	foreignBin := filepath.Join(testbin.Dir(t, "nvpair-engine-manager", "stop-foreign", "foreignengine"), testbin.Exe("foreignengine"))
	copyFile(t, fakeEngineBin, foreignBin)
	foreign := spawnFakeListener(t, foreignBin, port)
	restarted := NewExecutor(ex.reg, NewReporter(nil), nil, ex.baseDir)
	t.Cleanup(restarted.StopAll)
	st, err := restarted.state("target")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.installDir = managedDir
	st.mu.Unlock()
	result := restarted.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
	if result.Error == nil || !strings.Contains(result.Error.Message, "another process already owns the server port") {
		t.Fatalf("committed target with foreign listener = %+v, want existing port-ownership error", result)
	}
	st.mu.Lock()
	owned := st.proc != nil
	st.mu.Unlock()
	if owned || !pidAlive(foreign.Process.Pid) || !portServing(port) {
		t.Fatalf("foreign listener changed: owned=%v alive=%v serving=%v", owned, pidAlive(foreign.Process.Pid), portServing(port))
	}
	saved, err := restarted.desired.load()
	if err != nil {
		t.Fatal(err)
	}
	if saved.Engines["source"] || !saved.Engines["target"] || saved.PendingSwitch == nil || !saved.PendingSwitch.TargetCommitted {
		t.Fatalf("foreign listener error changed committed intent: %+v", saved)
	}
}
