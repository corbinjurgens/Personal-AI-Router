// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/testbin"
)

func TestStopPauseOwnership(t *testing.T) {
	t.Run("adopted engine refuses unscoped stop", func(t *testing.T) {
		e := wakeTestExecutor(t, t.TempDir(), nil)
		st, err := e.state("fake")
		if err != nil {
			t.Fatal(err)
		}
		st.mu.Lock()
		st.plat.Runtime.Ready = nil
		st.running = true
		st.adopted = true
		st.mu.Unlock()
		err = e.Sleep("fake")
		if err == nil || !strings.Contains(err.Error(), "external management") {
			t.Fatalf("sleep of external engine = %v", err)
		}
		st.mu.Lock()
		running, adopted := st.running, st.adopted
		st.mu.Unlock()
		if !running || !adopted {
			t.Fatalf("external engine state changed: running=%v adopted=%v", running, adopted)
		}
	})
}

func TestStopPauseOwnershipExternalEnginesHaveNoCleanupEffects(t *testing.T) {
	test := func(name string) {
		t.Run(name, func(t *testing.T) {
			var unloads atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if !((r.Method == http.MethodGet && (r.URL.Path == "/" || r.URL.Path == "/loaded")) || (r.Method == http.MethodPost && r.URL.Path == "/unload")) {
					http.NotFound(w, r)
					return
				}
				if r.URL.Path == "/unload" {
					unloads.Add(1)
				}
				w.WriteHeader(http.StatusOK)
			}))
			t.Cleanup(server.Close)
			parsed, err := url.Parse(server.URL)
			if err != nil {
				t.Fatal(err)
			}
			port, err := strconv.Atoi(parsed.Port())
			if err != nil {
				t.Fatal(err)
			}
			m := llamaTestEngineManifest(fakeEngineBin)
			platform := m.Platforms[hostKey()]
			platform.Runtime.Port = port
			platform.Runtime.Ready = &Probe{HTTP: server.URL, Status: http.StatusOK}
			platform.Runtime.Stop = &StopSpec{GraceS: 1}
			m.Platforms[hostKey()] = platform
			m.Actions["loaded_models"] = Action{HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/loaded"}, Result: &ActionResult{Array: "models", Field: "name"}}
			m.Actions["unload_model"] = Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/unload"}}
			ex := newTestExecutor(t, m)
			status, err := ex.Status("fake")
			if err != nil || !status.Running {
				t.Fatalf("external adoption: %+v, %v", status, err)
			}
			if err := ex.setDesiredEnabled("fake", true, intentLocal); err != nil {
				t.Fatal(err)
			}
			ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
			if err := ex.Sleep("fake"); err == nil || !strings.Contains(err.Error(), "external management") {
				t.Fatalf("external Sleep: %v", err)
			}
			if unloads.Load() != 0 {
				t.Fatal("Pause unloaded an external model")
			}
			if !portServing(port) {
				t.Fatal("Pause terminated an external listener")
			}
			enabled, known, err := ex.desired.get("fake")
			if err != nil || !known || !enabled {
				t.Fatalf("Pause rewrote On intent: enabled=%v known=%v err=%v", enabled, known, err)
			}
			server.Close() // The fixture's external owner stops its own listener.
			err = ex.Sleep("fake")
			if err == nil {
				t.Fatal("listener closure concealed a surviving external owner or unknown model residency")
			}
		})
	}
	test("process owner")
}

func TestStopPauseOwnershipUnknownListenerRefuses(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(server.Close)
	parsed, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(parsed.Port())
	if err != nil {
		t.Fatal(err)
	}
	m := llamaTestEngineManifest(fakeEngineBin)
	p := m.Platforms[hostKey()]
	p.Runtime.Port = port
	p.Runtime.Ready = &Probe{HTTP: server.URL, Status: http.StatusOK}
	m.Platforms[hostKey()] = p
	ex := newTestExecutor(t, m)
	if err := ex.Sleep("fake"); err == nil || !strings.Contains(err.Error(), "ownership is unverified") {
		t.Fatalf("unknown listener: %v", err)
	}
	if !portServing(port) {
		t.Fatal("unknown listener was stopped")
	}
}

func TestStopPauseOwnershipControlledProcessStopsAndResumesSavedOn(t *testing.T) {
	ex := wakeTestExecutor(t, t.TempDir(), nil)
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	proc, port := st.proc, st.port
	st.mu.Unlock()
	if proc == nil {
		t.Fatal("fixture did not own a process")
	}
	ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	ex.cancelBlockedActivations()
	if err := ex.Sleep("fake"); err != nil {
		t.Fatal(err)
	}
	select {
	case <-proc.done:
	default:
		t.Fatal("Sleep returned before owned process reaping")
	}
	if portServing(port) || !orphanPIDExited(proc.cmd.Process.Pid) {
		t.Fatal("owned process or listener survived Sleep")
	}
	enabled, known, err := ex.desired.get("fake")
	if err != nil || !known || !enabled {
		t.Fatalf("Pause changed saved On: %v %v %v", enabled, known, err)
	}
	ex.nodeProjection.set(nodeProjection{})
	if err := ex.Wake(context.Background(), "fake"); err != nil {
		t.Fatalf("saved-On Resume: %v", err)
	}
	st.mu.Lock()
	next, running := st.proc, st.running
	st.mu.Unlock()
	if next == nil || next == proc || !running {
		t.Fatal("Resume did not use a fresh owned process")
	}
}

func TestStopPauseOwnershipUnresolvedSpawnRemainsIncomplete(t *testing.T) {
	ex := wakeTestExecutor(t, t.TempDir(), nil)
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.port = port
	if err := ex.desired.setProcessStop("fake", processStopEvidence{Bin: fakeEngineBin, Port: port, Address: "127.0.0.1:" + strconv.Itoa(port)}); err != nil {
		t.Fatal(err)
	}
	ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	if err := ex.Sleep("fake"); err == nil {
		t.Fatal("running:false and a refused listener established unknown spawn termination")
	}
}

func TestStopPauseOwnershipPauseCancelsStartingProcess(t *testing.T) {
	ex := newTestExecutor(t, slowStartManifest())
	t.Cleanup(ex.StopAll)
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- ex.Start(context.Background(), "fake") }()
	awaitLockCondition(t, "owned process awaiting readiness", func() bool { st.mu.Lock(); defer st.mu.Unlock(); return st.proc != nil })
	st.mu.Lock()
	proc := st.proc
	st.mu.Unlock()
	ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	ex.cancelBlockedActivations()
	if err := awaitLockCommand(t, done); rpcCode(err) != codeLocalVeto {
		t.Fatalf("cancelled start: %v", err)
	}
	if err := ex.Sleep("fake"); err != nil {
		t.Fatalf("Sleep after start cancellation: %v", err)
	}
	select {
	case <-proc.done:
	default:
		t.Fatal("cancelled startup child was not reaped")
	}
	if !orphanPIDExited(proc.cmd.Process.Pid) {
		t.Fatal("cancelled startup child survived")
	}
	if err := ex.Start(context.Background(), "fake"); rpcCode(err) != codeLocalVeto {
		t.Fatalf("post-pause restart: %v", err)
	}
}

func TestStopPauseOwnershipReclaimsExactManagedOrphan(t *testing.T) {
	dir := testbin.Dir(t, "nvpair-engine-manager", "stop-orphan", "fakeorphan")
	bin := filepath.Join(dir, testbin.Exe("fakeorphan"))
	copyFile(t, fakeEngineBin, bin)
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	orphan := spawnFakeListener(t, bin, port)
	m := llamaTestEngineManifest(bin)
	ex := managedFixtureExecutor(t, m, dir)
	if err := ex.desired.setProcessStop(m.Engine, processStopEvidence{Bin: bin, Port: port, Address: "127.0.0.1:" + strconv.Itoa(port), PID: orphan.Process.Pid}); err != nil {
		t.Fatal(err)
	}
	if err := ex.desired.setAllocatedPort(m.Engine, port); err != nil {
		t.Fatal(err)
	}
	if err := ex.setDesiredEnabled(m.Engine, true, intentLocal); err != nil {
		t.Fatal(err)
	}
	ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	if err := ex.Sleep(m.Engine); err != nil {
		t.Fatalf("managed orphan Sleep: %v", err)
	}
	if !orphanPIDExited(orphan.Process.Pid) || portServing(port) {
		t.Fatal("Sleep completed before exact orphan termination")
	}
	enabled, known, err := ex.desired.get(m.Engine)
	if err != nil || !known || !enabled {
		t.Fatalf("orphan cleanup changed intent: %v %v %v", enabled, known, err)
	}
}

func TestStopPauseOwnershipCancelsAcceptedStartAndLoad(t *testing.T) {
	m, ex, _ := lockTestManager(t)
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	ex.nodeProjection.set(nodeProjection{})
	startCtx, finishStart := ex.startContext(context.Background(), st)
	defer finishStart()
	loadCtx, finishLoad := registerServingLoad(context.Background(), st)
	defer finishLoad()
	remoteCtx, finishRemote, err := ex.beginRemote(context.Background(), "fake")
	if err != nil {
		t.Fatal(err)
	}
	defer finishRemote()
	st.opMu.Lock()
	unlock := sync.OnceFunc(st.opMu.Unlock)
	defer unlock()
	m.applyNodeProjection(&Message{Params: json.RawMessage(`{"managementBlocked":true}`)})
	for name, ctx := range map[string]context.Context{"start": startCtx, "load": loadCtx, "remote": remoteCtx} {
		if rpcCode(context.Cause(ctx)) != codeLocalVeto {
			t.Fatalf("accepted %s cause=%v", name, context.Cause(ctx))
		}
	}
	done := make(chan error, 1)
	go func() { done <- ex.Sleep("fake") }()
	select {
	case err := <-done:
		t.Fatalf("Sleep crossed held lifecycle lock: %v", err)
	default:
	}
	unlock()
	if err := awaitLockCommand(t, done); err != nil {
		t.Fatalf("Sleep after cancellation: %v", err)
	}
	if err := ex.Start(context.Background(), "fake"); rpcCode(err) != codeLocalVeto {
		t.Fatalf("post-fence start: %v", err)
	}
	// Exercise admission on an engine that actually supports model loading;
	// unsupported-operation validation is not execution admission.
	loadExecutor := newTestExecutor(t, lockLoadManifest())
	loadExecutor.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	if _, err := loadExecutor.ModelLoad(context.Background(), "fake", "tiny"); rpcCode(err) != codeLocalVeto {
		t.Fatalf("post-fence load: %v", err)
	}
}

func TestStopPauseOwnershipCancelsDispatchedLoad(t *testing.T) {
	ex := newTestExecutor(t, lockLoadManifest())
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.running, st.port = true, 12345
	entered := make(chan struct{})
	ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
		close(entered)
		<-r.Context().Done()
		return nil, r.Context().Err()
	})
	done := make(chan error, 1)
	go func() { _, err := ex.ModelLoad(context.Background(), "fake", "tiny"); done <- err }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("load did not reach native effect barrier")
	}
	ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	ex.cancelBlockedActivations()
	if err := awaitLockCommand(t, done); rpcCode(err) != codeLocalVeto {
		t.Fatalf("cancelled native load: %v", err)
	}
	st.mu.Lock()
	st.running = false
	st.mu.Unlock()
}

func TestStopPauseOwnershipResumePreservesOffAndLock(t *testing.T) {
	test := func(name string, locked bool) {
		t.Run(name, func(t *testing.T) {
			ex := wakeTestExecutor(t, t.TempDir(), nil)
			if err := ex.setDesiredEnabled("fake", false, intentLocal); err != nil {
				t.Fatal(err)
			}
			if locked {
				if _, err := ex.desired.setLocked("fake", true); err != nil {
					t.Fatal(err)
				}
			}
			ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
			if err := ex.Sleep("fake"); err != nil {
				t.Fatal(err)
			}
			ex.nodeProjection.set(nodeProjection{})
			if err := ex.Wake(context.Background(), "fake"); !errors.Is(err, errIntentOff) {
				t.Fatalf("Resume activated saved Off: %v", err)
			}
			state, err := ex.Status("fake")
			if err != nil || state.Running || state.Locked == nil || *state.Locked != locked {
				t.Fatalf("Resume state=%+v err=%v", state, err)
			}
		})
	}
	test("saved Off", false)
	test("persistent lock", true)
}

func TestStopPauseOwnershipSleepReplyCarriesPhysicalProof(t *testing.T) {
	m, ex, out := lockTestManager(t)
	ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	id := json.RawMessage(`91`)
	if !m.handleLifecyclePolicyRequest(context.Background(), &Message{ID: &id, Method: sleepMethod, Params: json.RawMessage(`{"engine":"fake"}`)}) {
		t.Fatal("Sleep request was not handled")
	}
	var reply Message
	awaitLockCondition(t, "Sleep proof response", func() bool {
		for _, frame := range out.frames() {
			if frame.ID != nil && string(*frame.ID) == string(id) {
				reply = frame
				return true
			}
		}
		return false
	})
	if reply.Error != nil {
		t.Fatalf("Sleep response: %+v", reply.Error)
	}
	var status EngineStatus
	if err := json.Unmarshal(reply.Result, &status); err != nil {
		t.Fatal(err)
	}
	if status.Engine != "fake" || status.Running || !status.TerminationVerified {
		t.Fatalf("Sleep lacked positive proof: %+v", status)
	}
}

func TestStopPauseOwnershipSpawnChecksFenceAfterPreparation(t *testing.T) {
	f := newLlamaServingFixture(t)
	applyLlamaServing(t, f, approveLlamaServing(t, f))
	st := settingsState(t, f.ex)
	st.opMu.Lock()
	defer st.opMu.Unlock()
	if st.plat.Runtime.servingSnapshot == nil {
		t.Fatal("cold apply did not prepare a snapshot")
	}
	st.plat.Runtime.servingSnapshot.Release()
	st.plat.Runtime.servingSnapshot = nil
	vars := st.pathVars()
	vars["host"], vars["port"] = "127.0.0.1", strconv.Itoa(st.port)
	prepared := false
	prepare := func(root string, data []byte) (string, func(), error) {
		path, release, err := prepareServingSnapshot(root, data)
		if err != nil {
			return path, release, err
		}
		prepared = true
		f.ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
		return path, release, nil
	}
	err := f.ex.bringUpProcessWithSnapshot(context.Background(), st, "fake", st.plat.Runtime, st.port, vars, prepare)
	if !prepared {
		t.Fatal("start never reached the snapshot preparation barrier")
	}
	if rpcCode(err) != codeLocalVeto {
		t.Fatalf("pre-spawn pause: %v", err)
	}
	assertNoSpawn(t, f)
}

func stopPauseOwnershipExternalExecutor(t *testing.T) (*Executor, *engineState) {
	t.Helper()
	m := testEngineManifest(fakeEngineBin)
	m.Actions["load_model"] = Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/load"}}
	m.Actions["unload_model"] = Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/unload"}}
	ex := newTestExecutor(t, m)
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	// This is an external native listener in memory. Every HTTP operation below
	// is intercepted before dial; this fixture starts no process or listener.
	st.running, st.adopted, st.port = true, true, 12345
	return ex, st
}

func TestStopPauseOwnershipExternalListenerClosureNeedsPIDExit(t *testing.T) {
	test := func(name string, known, exited bool) {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodGet || r.URL.Path != "/" {
					http.NotFound(w, r)
					return
				}
				w.WriteHeader(http.StatusOK)
			}))
			t.Cleanup(server.Close)
			parsed, err := url.Parse(server.URL)
			if err != nil {
				t.Fatal(err)
			}
			port, err := strconv.Atoi(parsed.Port())
			if err != nil {
				t.Fatal(err)
			}
			m := llamaTestEngineManifest(fakeEngineBin)
			p := m.Platforms[hostKey()]
			p.Runtime.Port, p.Runtime.Ready = port, &Probe{HTTP: server.URL, Status: http.StatusOK}
			m.Platforms[hostKey()] = p
			ex := newTestExecutor(t, m)
			st, err := ex.state("fake")
			if err != nil {
				t.Fatal(err)
			}
			owner := func(int) (int, string, bool) { return os.Getpid(), "private-fixture-owner", known }
			pidExited := func(pid int) bool { return pid > 0 && exited }
			presence, _ := ex.reconcilePresenceStateWithProcessProof(context.Background(), "fake", st, true, port, true, owner, pidExited)
			if !presence.Identified {
				t.Fatal("private fixture was not adopted")
			}
			server.Close() // Listener closes while the fixture owner stays alive.
			presence, _ = ex.reconcilePresenceStateWithProcessProof(context.Background(), "fake", st, true, port, true, owner, pidExited)
			status := ex.snapshot("fake", st)
			wantStopped := known && exited
			if status.Running == wantStopped || presence.Occupied == wantStopped || status.Healthy || status.TerminationVerified {
				t.Fatalf("closure proof: status=%+v presence=%+v wantStopped=%v", status, presence, wantStopped)
			}
			ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
			err = ex.Sleep("fake")
			if (err == nil) != wantStopped {
				t.Fatalf("Sleep after external closure: %v wantStopped=%v", err, wantStopped)
			}
			if err != nil && !strings.Contains(err.Error(), `engine "fake"`) {
				t.Fatalf("incomplete shutdown omitted engine: %v", err)
			}
		})
	}
	test("unknown external PID stays fenced", false, false)
	test("observed external PID remains alive", true, false)
	test("positive observed PID exit permits completion", true, true)
}

func TestStopPauseOwnershipNativeDispatchRechecksFinalFence(t *testing.T) {
	test := func(name, action string, cli bool) {
		t.Run(name, func(t *testing.T) {
			ex, st := stopPauseOwnershipExternalExecutor(t)
			marker := filepath.Join(t.TempDir(), "late-native-load")
			if action == "load_model" {
				if cli {
					st.manifest.Actions[action] = Action{Cmd: []string{fakeEngineBin, "touch", marker}}
				} else {
					st.manifest.Actions[action] = Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/load"}}
				}
			}
			var effects atomic.Int32
			ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
				effects.Add(1)
				return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"ok":true}`))}, nil
			})
			params := json.RawMessage(`{"model":"tiny"}`)
			ctx, release, err := ex.guardServingLoad(context.Background(), st, action, params)
			if err != nil {
				t.Fatal(err)
			}
			defer release()
			// Admission and lifecycle authorization have completed. Hold the
			// final runner's saved-lock read, after its first projection check.
			ex.desired.mu.Lock()
			releaseDesired := sync.OnceFunc(ex.desired.mu.Unlock)
			defer releaseDesired()
			done := make(chan error, 1)
			go func() {
				_, err := ex.dispatchAction(ctx, st, "fake", action, st.manifest.Actions[action], params)
				done <- err
			}()
			if err := servingWaitForPhase(context.Background(), "beginActivationEffect("); err != nil {
				t.Fatal(err)
			}
			ex.remoteWork.mu.Lock()
			releaseRemote := sync.OnceFunc(ex.remoteWork.mu.Unlock)
			defer releaseRemote()
			applied := make(chan struct{})
			go func() {
				(&Manager{exec: ex}).applyNodeProjection(&Message{Params: json.RawMessage(`{"managementBlocked":true}`)})
				close(applied)
			}()
			if err := servingWaitForPhase(context.Background(), "(*Executor).cancelVetoedRemoteWork("); err != nil {
				t.Fatal(err)
			}
			if rpcCode(context.Cause(ctx)) != codeLocalVeto {
				t.Fatalf("execution cancellation waited for remote bookkeeping: %v", context.Cause(ctx))
			}
			releaseDesired()
			if err := awaitLockCommand(t, done); rpcCode(err) != codeLocalVeto {
				t.Fatalf("native dispatch after final-check barrier: %v", err)
			}
			if effects.Load() != 0 {
				t.Fatal("native HTTP execution began after the fence")
			}
			if _, err := os.Stat(marker); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("native CLI execution began after the fence: %v", err)
			}
			release()
			st.mu.Lock()
			active := len(st.loads.active)
			st.mu.Unlock()
			if active != 0 {
				t.Fatalf("final dispatch refusal leaked registered execution: %d", active)
			}
			releaseRemote()
			select {
			case <-applied:
			case <-time.After(5 * time.Second):
				t.Fatal("projection did not finish after bookkeeping release")
			}
		})
	}
	test("native CLI load", "load_model", true)
	test("native HTTP load", "load_model", false)
}

func TestStopPauseOwnershipNativeHTTPResponseDoesNotHoldProjection(t *testing.T) {
	ex, _ := stopPauseOwnershipExternalExecutor(t)
	entered := make(chan struct{})
	ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
		close(entered)
		<-r.Context().Done()
		return nil, r.Context().Err()
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := ex.Action(ctx, "fake", "load_model", json.RawMessage(`{"model":"tiny"}`))
		done <- err
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("native HTTP dispatch did not reach response barrier")
	}
	ex.remoteWork.mu.Lock()
	releaseRemote := sync.OnceFunc(ex.remoteWork.mu.Unlock)
	defer releaseRemote()
	applied := make(chan struct{})
	go func() {
		(&Manager{exec: ex}).applyNodeProjection(&Message{Params: json.RawMessage(`{"managementBlocked":true}`)})
		close(applied)
	}()
	if err := servingWaitForPhase(context.Background(), "(*Executor).cancelVetoedRemoteWork("); err != nil {
		t.Fatal(err)
	}
	if err := awaitLockCommand(t, done); rpcCode(err) != codeLocalVeto {
		t.Fatalf("in-flight HTTP execution was not cancelled before remote bookkeeping: %v", err)
	}
	releaseRemote()
	select {
	case <-applied:
	case <-time.After(5 * time.Second):
		t.Fatal("projection remained blocked after HTTP cancellation")
	}
}
