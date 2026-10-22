// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"path/filepath"
	"sync"
	"testing"
	"testing/synctest"
	"time"
)

func TestCrashRelaunchSavedOnReturnsToReady(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	p := m.Platforms[hostKey()]
	p.Runtime.Health = nil
	m.Platforms[hostKey()] = p
	ex := desiredTestExecutor(t, m, t.TempDir())
	running := make(chan EngineStatus, 8)
	ex.emit = func(method string, value any) {
		if status, ok := value.(EngineStatus); ok && method == "engine:state-changed" && status.Running {
			running <- status
		}
	}
	if err := ex.Start(context.Background(), m.Engine); err != nil {
		t.Fatalf("initial owned start: %v", err)
	}
	<-running
	st, err := ex.state(m.Engine)
	if err != nil {
		t.Fatalf("engine state: %v", err)
	}
	st.mu.Lock()
	first := st.proc
	st.mu.Unlock()
	if first == nil {
		t.Fatal("initial ready engine has no owned process")
	}
	// Crash the test-owned child one second after readiness, without Stop.
	timer := time.NewTimer(time.Second)
	defer timer.Stop()
	<-timer.C
	if err := first.cmd.Process.Kill(); err != nil {
		t.Fatalf("crash owned fake engine: %v", err)
	}
	select {
	case <-first.done:
	case <-time.After(5 * time.Second):
		t.Fatal("crashed fake engine was not reaped")
	}
	select {
	case status := <-running:
		if !status.Healthy {
			t.Fatalf("relaunch state = %+v, want ready", status)
		}
	case <-time.After(12 * time.Second):
		t.Fatal("saved-On engine did not publish running after unexpected exit")
	}
	st.mu.Lock()
	second := st.proc
	st.mu.Unlock()
	if second == nil || second == first || second.cmd.Process.Pid == first.cmd.Process.Pid {
		t.Fatalf("relaunch did not replace owned process: first=%p second=%p", first, second)
	}
	// Readiness resets the per-engine failure episode: another crash gets the
	// first delay again, rather than the second (30-second) backoff.
	if err := second.cmd.Process.Kill(); err != nil {
		t.Fatalf("crash recovered fake engine: %v", err)
	}
	select {
	case <-running:
	case <-time.After(12 * time.Second):
		t.Fatal("successful readiness did not reset the relaunch backoff")
	}
}

type crashReadyTransport struct {
	base  http.RoundTripper
	once  sync.Once
	ready func()
}

func (tr *crashReadyTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	response, err := tr.base.RoundTrip(request)
	if err == nil && response.StatusCode == http.StatusOK && request.URL.Path == "/" {
		response.Body = &crashReadyBody{ReadCloser: response.Body, ready: func() { tr.once.Do(tr.ready) }}
	}
	return response, err
}

type crashReadyBody struct {
	io.ReadCloser
	ready func()
}

func (body *crashReadyBody) Close() error {
	err := body.ReadCloser.Close()
	// probe has already chosen its successful return value; its deferred
	// Close fences the last readiness result before doStart commits running.
	body.ready()
	return err
}

func TestCrashRelaunchPauseAtReadinessStopsUncommittedChild(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	p := m.Platforms[hostKey()]
	p.Runtime.Health = nil
	m.Platforms[hostKey()] = p
	ex := desiredTestExecutor(t, m, t.TempDir())
	running := make(chan EngineStatus, 8)
	ex.emit = func(method string, value any) {
		if status, ok := value.(EngineStatus); ok && method == "engine:state-changed" && status.Running {
			running <- status
		}
	}
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("initial start: %v", err)
	}
	<-running
	st, err := ex.state("fake")
	if err != nil {
		t.Fatalf("engine state: %v", err)
	}
	params, err := json.Marshal(nodeProjection{PausedLocally: true})
	if err != nil {
		t.Fatalf("encode pause: %v", err)
	}
	readyPID := make(chan int, 1)
	manager := &Manager{exec: ex}
	ex.client.Transport = &crashReadyTransport{base: ex.client.Transport, ready: func() {
		st.mu.Lock()
		proc := st.proc
		st.mu.Unlock()
		if proc != nil {
			readyPID <- proc.cmd.Process.Pid
		}
		manager.applyNodeProjection(&Message{Params: params})
	}}
	st.mu.Lock()
	first := st.proc
	st.mu.Unlock()
	if err := first.cmd.Process.Kill(); err != nil {
		t.Fatalf("crash initial engine: %v", err)
	}
	<-first.done
	recovery := waitCrashEpisode(t, st)
	select {
	case <-recovery.done:
	case <-time.After(12 * time.Second):
		t.Fatal("readiness cancellation did not complete recovery cleanup")
	}
	var pid int
	select {
	case pid = <-readyPID:
	default:
		t.Fatal("test did not reach successful owned readiness")
	}
	select {
	case status := <-running:
		t.Fatalf("paused uncommitted child published running: %+v", status)
	default:
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.running || st.proc != nil || pidAlive(pid) {
		t.Fatalf("paused child survived: running=%v proc=%p pid=%d alive=%v", st.running, st.proc, pid, pidAlive(pid))
	}
}

func TestCrashRelaunchPauseDuringIdentityProbeNeverAdopts(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	p := m.Platforms[hostKey()]
	p.Runtime.Health = nil
	m.Platforms[hostKey()] = p
	ex := desiredTestExecutor(t, m, t.TempDir())
	running := make(chan EngineStatus, 8)
	ex.emit = func(method string, value any) {
		if status, ok := value.(EngineStatus); ok && method == "engine:state-changed" && status.Running {
			running <- status
		}
	}
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("initial start: %v", err)
	}
	<-running
	st, err := ex.state("fake")
	if err != nil {
		t.Fatalf("engine state: %v", err)
	}
	st.opMu.Lock()
	unlock := sync.OnceFunc(st.opMu.Unlock)
	defer unlock()
	st.mu.Lock()
	first, port := st.proc, st.port
	st.mu.Unlock()
	if err := first.cmd.Process.Kill(); err != nil {
		t.Fatalf("crash owned engine: %v", err)
	}
	<-first.done
	recovery := waitCrashEpisode(t, st)
	// This listener is test-owned, not manager-owned and outside its managed
	// install root. Cancellation must neither adopt nor signal it.
	listener := spawnFakeListener(t, fakeEngineBin, port)
	params, err := json.Marshal(nodeProjection{PausedLocally: true})
	if err != nil {
		t.Fatalf("encode pause: %v", err)
	}
	probed := make(chan struct{})
	manager := &Manager{exec: ex}
	ex.client.Transport = &crashReadyTransport{base: ex.client.Transport, ready: func() {
		manager.applyNodeProjection(&Message{Params: params})
		close(probed)
	}}
	unlock() // Transport and listener are installed before restore can probe.
	select {
	case <-recovery.done:
	case <-time.After(12 * time.Second):
		t.Fatal("cancelled identity probe did not complete recovery")
	}
	select {
	case <-probed:
	default:
		t.Fatal("test did not reach successful identity probe")
	}
	select {
	case status := <-running:
		t.Fatalf("cancelled identity probe published adoption: %+v", status)
	default:
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.running || st.adopted || st.proc != nil || !pidAlive(listener.Process.Pid) {
		t.Fatalf("cancelled adoption: running=%v adopted=%v proc=%p listenerPID=%d alive=%v", st.running, st.adopted, st.proc, listener.Process.Pid, pidAlive(listener.Process.Pid))
	}
}

func waitCrashEpisode(t *testing.T, st *engineState) *engineRelaunch {
	t.Helper()
	deadline := time.NewTimer(3 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		st.mu.Lock()
		recovery := st.relaunch
		st.mu.Unlock()
		if recovery != nil {
			return recovery
		}
		select {
		case <-tick.C:
		case <-deadline.C:
			t.Fatal("unexpected exit did not register recovery")
		}
	}
}

func TestCrashRelaunchExplicitStartResetsEpisode(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	p := m.Platforms[hostKey()]
	p.Runtime.Health = nil
	m.Platforms[hostKey()] = p
	ex := desiredTestExecutor(t, m, t.TempDir())
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("initial start: %v", err)
	}
	st, err := ex.state("fake")
	if err != nil {
		t.Fatalf("engine state: %v", err)
	}
	st.mu.Lock()
	proc := st.proc
	st.mu.Unlock()
	if err := proc.cmd.Process.Kill(); err != nil {
		t.Fatalf("crash owned engine: %v", err)
	}
	<-proc.done
	old := waitCrashEpisode(t, st)
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("explicit start during backoff: %v", err)
	}
	select {
	case <-old.done:
	case <-time.After(2 * time.Second):
		t.Fatal("successful explicit readiness left old failure episode attached")
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.relaunch != nil || !st.running {
		t.Fatalf("readiness reset: recovery=%p running=%v", st.relaunch, st.running)
	}
}

func TestCrashRelaunchExplicitLocalStartWhilePausedRemainsAuthorized(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	p := m.Platforms[hostKey()]
	p.Runtime.Health = nil
	m.Platforms[hostKey()] = p
	ex := desiredTestExecutor(t, m, t.TempDir())
	st, err := ex.state("fake")
	if err != nil {
		t.Fatalf("engine state: %v", err)
	}
	// Retain the cancelled episode while an explicit local start takes over.
	// Its presence must not give that independent start automatic-start rules.
	_, cancel := context.WithCancel(context.Background())
	recovery := &engineRelaunch{cancel: cancel, done: make(chan struct{})}
	t.Cleanup(func() { cancel(); close(recovery.done) })
	st.mu.Lock()
	st.relaunch = recovery
	st.mu.Unlock()
	params, err := json.Marshal(nodeProjection{PausedLocally: true})
	if err != nil {
		t.Fatalf("encode pause projection: %v", err)
	}
	manager := &Manager{exec: ex}
	manager.applyNodeProjection(&Message{Params: params})
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("explicit local start while paused: %v", err)
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if !st.running || st.adopted || st.proc == nil || st.relaunch != nil {
		t.Fatalf("explicit readiness: running=%v adopted=%v proc=%p recovery=%p", st.running, st.adopted, st.proc, st.relaunch)
	}
}

func TestCrashRelaunchPauseDuringOperationWaitCancels(t *testing.T) {
	m := testEngineManifest(filepath.Join(t.TempDir(), "absent-engine"))
	p := m.Platforms[hostKey()]
	p.Runtime.Ready, p.Runtime.Health = nil, nil
	m.Platforms[hostKey()] = p
	ex := desiredTestExecutor(t, m, t.TempDir())
	if err := ex.setDesiredEnabled("fake", true, intentLocal); err != nil {
		t.Fatalf("save On intent: %v", err)
	}
	st, err := ex.state("fake")
	if err != nil {
		t.Fatalf("engine state: %v", err)
	}
	st.opMu.Lock()
	locked := true
	defer func() {
		if locked {
			st.opMu.Unlock()
		}
	}()
	ex.scheduleCrashRelaunch(st, "fake")
	recovery := waitCrashEpisode(t, st)
	deadline := time.NewTimer(10 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		ex.switchAdmission.Lock()
		admitted := ex.switchMutations["fake"] > 0
		ex.switchAdmission.Unlock()
		if admitted {
			break // Restore passed the initial pause check and awaits opMu.
		}
		select {
		case <-tick.C:
		case <-deadline.C:
			t.Fatal("recovery did not enter mutation admission")
		}
	}
	params, err := json.Marshal(nodeProjection{PausedLocally: true})
	if err != nil {
		t.Fatalf("encode pause projection: %v", err)
	}
	manager := &Manager{exec: ex}
	manager.applyNodeProjection(&Message{Params: params})
	st.opMu.Unlock()
	locked = false
	select {
	case <-recovery.done:
	case <-time.After(2 * time.Second):
		t.Fatal("late pause did not cancel admitted automatic recovery")
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.proc != nil || st.running || st.gen != 0 {
		t.Fatalf("paused recovery launched: proc=%p running=%v generation=%d", st.proc, st.running, st.gen)
	}
}

// Virtual-time controls use a completed owned-process handle and an absent
// executable. They cannot launch a child; the real-binary proof above owns
// readiness, replacement identity and the running notification assertions.
func crashRelaunchStoppedFixture(t *testing.T, enabled bool) (*Executor, *engineState, *engineRelaunch) {
	t.Helper()
	m := testEngineManifest(filepath.Join(t.TempDir(), "absent-engine"))
	p := m.Platforms[hostKey()]
	p.Runtime.Ready = nil
	p.Runtime.Health = nil
	m.Platforms[hostKey()] = p
	ex := desiredTestExecutor(t, m, t.TempDir())
	if err := ex.setDesiredEnabled(m.Engine, enabled, intentLocal); err != nil {
		t.Fatalf("save fixture intent: %v", err)
	}
	st, err := ex.state(m.Engine)
	if err != nil {
		t.Fatalf("fixture state: %v", err)
	}
	proc := &managedProc{done: make(chan struct{})}
	st.mu.Lock()
	st.proc, st.running = proc, true
	st.mu.Unlock()
	ex.watch(st, m.Engine, proc)
	close(proc.done)
	synctest.Wait()
	st.mu.Lock()
	recovery := st.relaunch
	st.mu.Unlock()
	if recovery == nil {
		t.Fatal("unexpected exit did not register a bounded recovery episode")
	}
	return ex, st, recovery
}

func TestCrashRelaunchSavedOffNeverStarts(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		_, st, recovery := crashRelaunchStoppedFixture(t, false)
		<-recovery.done
		st.mu.Lock()
		defer st.mu.Unlock()
		if st.proc != nil || st.running || st.gen != 0 {
			t.Fatalf("saved Off started: running=%v proc=%p generation=%d", st.running, st.proc, st.gen)
		}
	})
}

func TestCrashRelaunchStopsAtRetryCap(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		start := time.Now()
		ex, st, recovery := crashRelaunchStoppedFixture(t, true)
		<-recovery.done
		if elapsed := time.Since(start); elapsed != 155*time.Second {
			t.Fatalf("failed-start backoff = %s, want 5s + 30s + 2m", elapsed)
		}
		// Making the executable available after exhaustion must not schedule a
		// fourth start. No positive timer remains to advance this episode.
		st.mu.Lock()
		st.plat.Runtime.Bin = fakeEngineBin
		st.mu.Unlock()
		time.Sleep(time.Hour)
		st.mu.Lock()
		if st.relaunch != nil || st.running || st.proc != nil || st.gen != 0 {
			t.Fatalf("retry cap bypassed: episode=%p running=%v proc=%p generation=%d", st.relaunch, st.running, st.proc, st.gen)
		}
		st.mu.Unlock()
		found := false
		for _, problem := range ex.reporter.snapshot() {
			if problem.ID == exitedID("fake") && problem.Action == "none" {
				found = true
			}
		}
		if !found {
			t.Fatal("retry exhaustion cleared the unexpected-exit error")
		}
	})
}

func TestCrashRelaunchCancelledByIntentionalStop(t *testing.T) {
	for _, action := range []string{"stop", "pause sleep", "shutdown", "episode context"} {
		t.Run(action, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ex, st, recovery := crashRelaunchStoppedFixture(t, true)
				switch action {
				case "stop":
					if err := ex.Stop("fake"); err != nil {
						t.Fatalf("explicit stop: %v", err)
					}
				case "pause sleep":
					ex.nodeProjection.set(nodeProjection{PausedLocally: true})
					if err := ex.Sleep("fake"); err != nil {
						t.Fatalf("pause sleep: %v", err)
					}
				case "shutdown":
					ex.StopAll()
				case "episode context":
					recovery.cancel()
				}
				<-recovery.done
				time.Sleep(time.Hour)
				st.mu.Lock()
				defer st.mu.Unlock()
				if st.relaunch != nil || st.proc != nil || st.running || st.gen != 0 {
					t.Fatalf("cancelled recovery started: episode=%p proc=%p running=%v generation=%d", st.relaunch, st.proc, st.running, st.gen)
				}
			})
		})
	}
}

func TestCrashRelaunchIntentionalStopBeforeWatcherNeverSchedules(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ex, st, recovery := crashRelaunchStoppedFixture(t, true)
		if err := ex.Sleep("fake"); err != nil {
			t.Fatalf("sleep completed child: %v", err)
		}
		<-recovery.done
		// Model a late watcher reaching scheduling after the intentional sleep.
		ex.scheduleCrashRelaunch(st, "fake")
		time.Sleep(time.Hour)
		st.mu.Lock()
		defer st.mu.Unlock()
		if st.relaunch != nil || !st.stopping {
			t.Fatalf("late watcher escaped stop fence: episode=%p stopping=%v", st.relaunch, st.stopping)
		}
	})
}

func TestCrashRelaunchDefersToPendingSwitch(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ex, request := switchFixture(t)
		request = prepareSwitch(t, ex, request)
		if err := ex.setDesiredEnabled("target", true, intentLocal); err != nil {
			t.Fatalf("save target On: %v", err)
		}
		st, err := ex.state("target")
		if err != nil {
			t.Fatalf("target state: %v", err)
		}
		ex.scheduleCrashRelaunch(st, "target")
		st.mu.Lock()
		recovery := st.relaunch
		st.mu.Unlock()
		if recovery == nil {
			t.Fatal("missing recovery episode")
		}
		<-recovery.done
		st.mu.Lock()
		defer st.mu.Unlock()
		if st.proc != nil || st.running || st.gen != 0 {
			t.Fatalf("reserved target launched: proc=%p running=%v generation=%d", st.proc, st.running, st.gen)
		}
		ex.desired.mu.Lock()
		state, err := ex.desired.load()
		ex.desired.mu.Unlock()
		checkpoint := state.PendingSwitch
		if err != nil || checkpoint == nil || checkpoint.OperationID != request.OperationID {
			t.Fatalf("pending switch replaced: checkpoint=%+v err=%v", checkpoint, err)
		}
	})
}

func TestCrashRelaunchPreservesFailedReclaimFence(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ex, st, recovery := crashRelaunchStoppedFixture(t, true)
		st.mu.Lock()
		st.reclaimFailed = true
		st.mu.Unlock()
		<-recovery.done
		st.mu.Lock()
		defer st.mu.Unlock()
		if !st.reclaimFailed || st.adopted || st.proc != nil || st.running {
			t.Fatalf("failed reclaim bypassed: fenced=%v adopted=%v proc=%p running=%v", st.reclaimFailed, st.adopted, st.proc, st.running)
		}
		if len(ex.reporter.snapshot()) == 0 {
			t.Fatal("failed recovery lost its error evidence")
		}
	})
}
