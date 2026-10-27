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
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/testbin"
)

func TestStopPauseOwnership(t *testing.T) {
	t.Run("adopted command engine refuses unscoped stop", func(t *testing.T) {
		e := wakeTestExecutor(t, t.TempDir(), nil)
		st, err := e.state("fake")
		if err != nil {
			t.Fatal(err)
		}
		st.mu.Lock()
		st.plat.Runtime.Mode = "command"
		st.plat.Runtime.Ready = nil
		st.running = true
		st.adopted = true
		st.mu.Unlock()
		err = e.Sleep("fake")
		if err == nil || !strings.Contains(err.Error(), "external management") {
			t.Fatalf("sleep of external command engine = %v", err)
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
	test := func(name, mode string) {
		t.Run(name, func(t *testing.T) {
			var unloads atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
			marker := filepath.Join(t.TempDir(), "unscoped-stop")
			m := testEngineManifest(fakeEngineBin)
			platform := m.Platforms[hostKey()]
			platform.Runtime.Mode = mode
			platform.Runtime.Port = port
			platform.Runtime.Ready = &Probe{HTTP: server.URL, Status: http.StatusOK}
			platform.Runtime.Stop = &StopSpec{Cmd: []string{fakeEngineBin, "touch", marker}, GraceS: 1}
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
			if _, err := os.Stat(marker); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("unscoped stop marker: %v", err)
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
	test("process owner", "process")
	test("command owner", "command")
}

func TestStopPauseOwnershipCommandResidencyProof(t *testing.T) {
	test := func(name string, retain bool) {
		t.Run(name, func(t *testing.T) {
			var resident atomic.Bool
			resident.Store(true)
			var unloads, stops atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/loaded":
					rows := []map[string]string{}
					if resident.Load() {
						rows = append(rows, map[string]string{"name": "tiny"})
					}
					if err := json.NewEncoder(w).Encode(map[string]any{"models": rows}); err != nil {
						t.Errorf("loaded reply: %v", err)
					}
				case "/unload":
					var p struct {
						Model string `json:"model"`
					}
					if err := json.NewDecoder(r.Body).Decode(&p); err != nil || p.Model != "tiny" {
						t.Errorf("unload params: %+v, %v", p, err)
						w.WriteHeader(http.StatusBadRequest)
						return
					}
					unloads.Add(1)
					if !retain {
						resident.Store(false)
					}
					w.WriteHeader(http.StatusOK)
				default:
					w.WriteHeader(http.StatusOK)
				}
			}))
			t.Cleanup(server.Close)
			control := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				stops.Add(1)
				if resident.Load() {
					t.Error("server stop ran before empty residency proof")
					w.WriteHeader(http.StatusConflict)
					return
				}
				if err := server.Listener.Close(); err != nil {
					t.Errorf("close owned fixture listener: %v", err)
				}
				w.WriteHeader(http.StatusOK)
			}))
			t.Cleanup(control.Close)
			helper, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			ex := commandStopTestExecutor(t, server.URL, []string{helper, "command-test-helper", control.URL}, 1, false)
			st, err := ex.state("command")
			if err != nil {
				t.Fatal(err)
			}
			st.manifest.Actions = map[string]Action{
				"loaded_models": {HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/loaded"}, Result: &ActionResult{Array: "models", Field: "name"}},
				"unload_model":  {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/unload"}},
			}
			ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
			err = ex.Sleep("command")
			if retain {
				if err == nil || stops.Load() != 0 || !resident.Load() {
					t.Fatalf("surviving residency: err=%v stops=%d resident=%v", err, stops.Load(), resident.Load())
				}
			} else if err != nil || stops.Load() != 1 || resident.Load() || !ex.switchTerminated(st) {
				t.Fatalf("physical stop: err=%v stops=%d resident=%v", err, stops.Load(), resident.Load())
			}
			if unloads.Load() != 1 {
				t.Fatalf("unloads=%d, want one owned model unload", unloads.Load())
			}
		})
	}
	test("empty readback precedes server stop", false)
	test("successful unload command with residency remaining is incomplete", true)
}

func TestStopPauseOwnershipUnknownListenerRefuses(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusServiceUnavailable) }))
	t.Cleanup(server.Close)
	parsed, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(parsed.Port())
	if err != nil {
		t.Fatal(err)
	}
	m := testEngineManifest(fakeEngineBin)
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

func TestStopPauseOwnershipResidencyRequiresClassifiableRows(t *testing.T) {
	spec := &ActionResult{Array: "models", Field: "key", Match: &ResultMatch{Field: "loaded_instances", Nonempty: true}}
	for _, tc := range []struct {
		name, body string
		valid      bool
		models     []string
	}{
		{name: "empty native inventory", body: `{"models":[]}`, valid: true, models: []string{}},
		{name: "explicit empty residency", body: `{"models":[{"key":"tiny","loaded_instances":[]}]}`, valid: true, models: []string{}},
		{name: "resident instance", body: `{"models":[{"key":"tiny","loaded_instances":[{}]}]}`, valid: true, models: []string{"tiny"}},
		{name: "missing residency", body: `{"models":[{"key":"tiny"}]}`},
		{name: "null residency", body: `{"models":[{"key":"tiny","loaded_instances":null}]}`},
		{name: "invalid residency", body: `{"models":[{"key":"tiny","loaded_instances":false}]}`},
		{name: "unidentified loaded instance", body: `{"models":[{"loaded_instances":[{}]}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			models, valid := commandResidentModels(json.RawMessage(tc.body), spec)
			if valid != tc.valid || (valid && !reflect.DeepEqual(models, tc.models)) {
				t.Fatalf("residency models=%v valid=%v; want %v/%v", models, valid, tc.models, tc.valid)
			}
		})
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
	m := testEngineManifest(bin)
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
	assertNoLlamaSpawn(t, f)
}

func stopPauseOwnershipChatExecutor(t *testing.T) (*Executor, *engineState) {
	t.Helper()
	m := testEngineManifest(fakeEngineBin)
	m.Family = "lmstudio"
	m.Actions["chat"] = Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/v1/chat/completions"}}
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

func TestStopPauseOwnershipChatRefusesExecutionAfterFence(t *testing.T) {
	ex, _ := stopPauseOwnershipChatExecutor(t)
	var effects atomic.Int32
	ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
		effects.Add(1)
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"ok":true}`))}, nil
	})
	ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	_, err := ex.Action(context.Background(), "fake", "chat", json.RawMessage(`{"model":"tiny","messages":[]}`))
	if rpcCode(err) != codeLocalVeto || effects.Load() != 0 {
		t.Fatalf("post-fence native chat: err=%v effects=%d", err, effects.Load())
	}
	// The same external listener remains available for metadata reads.
	raw, err := ex.Action(context.Background(), "fake", "list_models", nil)
	if err != nil {
		t.Fatalf("paused metadata read: %v", err)
	}
	var reply struct {
		OK bool `json:"ok"`
	}
	if err := json.Unmarshal(raw, &reply); err != nil {
		t.Fatal(err)
	}
	if !reply.OK || effects.Load() != 1 {
		t.Fatalf("metadata reply=%+v calls=%d", reply, effects.Load())
	}
	raw, err = ex.ModelUnload(context.Background(), "fake", "tiny")
	if err != nil {
		t.Fatalf("explicit local unload while paused: %v", err)
	}
	if err := json.Unmarshal(raw, &reply); err != nil {
		t.Fatal(err)
	}
	if !reply.OK || effects.Load() != 2 {
		t.Fatalf("local unload reply=%+v calls=%d", reply, effects.Load())
	}
}

func TestStopPauseOwnershipChatCancellationUsesExistingLoadRegistry(t *testing.T) {
	ex, st := stopPauseOwnershipChatExecutor(t)
	entered := make(chan context.Context, 1)
	ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
		entered <- r.Context()
		<-r.Context().Done()
		return nil, r.Context().Err()
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := ex.Action(ctx, "fake", "chat", json.RawMessage(`{"model":"tiny","messages":[]}`))
		done <- err
	}()
	var native context.Context
	select {
	case native = <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("native chat did not reach the execution barrier")
	}
	ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	ex.cancelBlockedActivations()
	if rpcCode(context.Cause(native)) != codeLocalVeto {
		cancel()
		t.Errorf("native chat cancellation cause=%v", context.Cause(native))
	}
	if err := awaitLockCommand(t, done); rpcCode(err) != codeLocalVeto {
		t.Fatalf("cancelled native chat=%v", err)
	}
	st.mu.Lock()
	active := len(st.loads.active)
	st.mu.Unlock()
	if active != 0 {
		t.Fatalf("native chat leaked load registrations: %d", active)
	}
}

func TestStopPauseOwnershipQueuedChatRechecksFence(t *testing.T) {
	ex, st := stopPauseOwnershipChatExecutor(t)
	var effects atomic.Int32
	ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
		effects.Add(1)
		return nil, errors.New("queued chat must not execute")
	})
	st.opMu.Lock()
	unlock := sync.OnceFunc(st.opMu.Unlock)
	defer unlock()
	done := make(chan error, 1)
	go func() {
		_, err := ex.Action(context.Background(), "fake", "chat", json.RawMessage(`{"model":"tiny","messages":[]}`))
		done <- err
	}()
	awaitLockCondition(t, "queued native chat registration", func() bool { st.mu.Lock(); defer st.mu.Unlock(); return len(st.loads.active) == 1 })
	ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	ex.cancelBlockedActivations()
	if err := awaitLockCommand(t, done); rpcCode(err) != codeLocalVeto {
		t.Fatalf("queued chat cancellation: %v", err)
	}
	if effects.Load() != 0 {
		t.Fatal("queued chat executed after the pause fence")
	}
}

func TestStopPauseOwnershipStoppedCommandRequiresPositiveResidency(t *testing.T) {
	test := func(name, readback string, wantStopped bool) {
		t.Run(name, func(t *testing.T) {
			port, err := freePort()
			if err != nil {
				t.Fatal(err)
			}
			m := testEngineManifest(fakeEngineBin)
			p := m.Platforms[hostKey()]
			p.Runtime.Mode, p.Runtime.Port = "command", port
			p.Runtime.Ready = &Probe{HTTP: "http://127.0.0.1:" + strconv.Itoa(port), Status: http.StatusOK}
			marker := filepath.Join(t.TempDir(), "unscoped-stop")
			p.Runtime.Stop = &StopSpec{Cmd: []string{fakeEngineBin, "touch", marker}}
			m.Platforms[hostKey()] = p
			m.Actions["loaded_models"] = Action{Cmd: []string{fakeEngineBin, "echo", readback}, Result: &ActionResult{Array: "models", Field: "name"}}
			ex := newTestExecutor(t, m)
			if err := ex.setDesiredEnabled("fake", true, intentLocal); err != nil {
				t.Fatal(err)
			}
			ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
			check := func(owner *Executor) {
				t.Helper()
				err := owner.Sleep("fake")
				if (err == nil) != wantStopped {
					t.Fatalf("stopped command with readback %s: %v", readback, err)
				}
				if err != nil && !strings.Contains(err.Error(), `engine "fake"`) {
					t.Fatalf("incomplete Pause omitted engine: %v", err)
				}
			}
			check(ex)
			check(ex) // A repeated Pause must obtain fresh native proof.
			restarted := NewExecutor(ex.reg, NewReporter(nil), func(string, any) {}, ex.baseDir)
			restarted.nodeProjection.set(nodeProjection{ManagementBlocked: true})
			check(restarted) // Paused startup has no running HTTP server.
			if _, err := os.Stat(marker); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("stopped engine ran an unscoped stop: %v", err)
			}
			enabled, known, err := restarted.desired.get("fake")
			if err != nil || !known || !enabled {
				t.Fatalf("cleanup changed saved On: enabled=%v known=%v err=%v", enabled, known, err)
			}
		})
	}
	test("positive empty native inventory", `{"models":[]}`, true)
	test("stopped server with resident model", `{"models":[{"name":"tiny"}]}`, false)
	test("unknown residency", `{"models":null}`, false)
}

func TestStopPauseOwnershipStoppedLMStudioNativeReadback(t *testing.T) {
	test := func(name, body string, readErr error, wantStopped bool) {
		t.Run(name, func(t *testing.T) {
			_, st := stopPauseOwnershipChatExecutor(t)
			st.plat.Runtime.CLI = filepath.Join(t.TempDir(), "private-lms")
			st.running, st.adopted = false, false
			var calls int
			err := stoppedLMStudioResidency(context.Background(), st, func(_ context.Context, argv []string) (string, error) {
				calls++
				want := []string{st.plat.Runtime.CLI, "ps", "--json"}
				if !reflect.DeepEqual(argv, want) {
					t.Fatalf("native residency argv=%v want=%v", argv, want)
				}
				return body, readErr
			})
			if calls != 1 || (err == nil) != wantStopped {
				t.Fatalf("native readback: calls=%d err=%v", calls, err)
			}
		})
	}
	test("empty model processes", `[]`, nil, true)
	test("surviving model process", `[{"identifier":"tiny"}]`, nil, false)
	test("unknown native schema", `{}`, nil, false)
	test("null is unknown", `null`, nil, false)
	test("lost native reply", `[]`, errors.New("lost readback"), false)
}

func TestStopPauseOwnershipExternalListenerClosureNeedsPIDExit(t *testing.T) {
	test := func(name string, known, exited bool) {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))
			t.Cleanup(server.Close)
			parsed, err := url.Parse(server.URL)
			if err != nil {
				t.Fatal(err)
			}
			port, err := strconv.Atoi(parsed.Port())
			if err != nil {
				t.Fatal(err)
			}
			m := testEngineManifest(fakeEngineBin)
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

func TestStopPauseOwnershipRawOllamaUnloadUsesCanonicalBinding(t *testing.T) {
	ex, st := stopPauseOwnershipChatExecutor(t)
	st.manifest.Family = "ollama"
	st.manifest.Actions["unload_model"] = Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/api/generate"}}
	st.manifest.Operations = map[string]OperationBinding{engines.OperationUnload: {Action: "unload_model", Params: map[string]json.RawMessage{"keep_alive": json.RawMessage(`0`)}}}
	var calls int
	ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
		calls++
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		var model string
		var keepAlive int
		if err := json.Unmarshal(body["model"], &model); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(body["keep_alive"], &keepAlive); err != nil {
			t.Fatal(err)
		}
		if len(body) != 2 || model != "tiny" || keepAlive != 0 || r.URL.Path != "/api/generate" {
			t.Fatalf("raw unload could execute generation: %s %v", r.URL.Path, body)
		}
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"done":true}`))}, nil
	})
	ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	for _, params := range []string{`{"model":"tiny"}`, `{"model":"tiny","keep_alive":-1,"prompt":"must not execute"}`} {
		if _, err := ex.Action(context.Background(), "fake", "unload_model", json.RawMessage(params)); err != nil {
			t.Fatalf("safe paused unload: %v", err)
		}
	}
	if _, err := ex.ModelUnload(context.Background(), "fake", "tiny"); err != nil {
		t.Fatalf("explicit local unload: %v", err)
	}
	st.manifest.Operations[engines.OperationUnload] = OperationBinding{Action: "unload_model"}
	if _, err := ex.Action(context.Background(), "fake", "unload_model", json.RawMessage(`{"model":"tiny"}`)); err == nil {
		t.Fatal("unproven unload binding reached generate while paused")
	}
	if calls != 3 || !st.running || !st.adopted {
		t.Fatalf("unload effects=%d external running=%v adopted=%v", calls, st.running, st.adopted)
	}
}

func TestStopPauseOwnershipCommandSpawnOrdersAgainstProjection(t *testing.T) {
	ex := wakeTestExecutor(t, t.TempDir(), nil)
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	ctx, finish := ex.startContext(context.Background(), st)
	defer finish()
	marker := filepath.Join(t.TempDir(), "late-command")
	rt := st.plat.Runtime
	rt.Mode, rt.Ready = "command", nil
	rt.Start = [][]string{{fakeEngineBin, "touch", marker}}
	st.plat.Runtime = rt
	// activationVeto reads the available projection first, then waits on the
	// desired-state mutex. This holds the actual start between check and spawn.
	ex.desired.mu.Lock()
	releaseDesired := sync.OnceFunc(ex.desired.mu.Unlock)
	defer releaseDesired()
	done := make(chan error, 1)
	go func() { done <- ex.bringUpCommand(ctx, context.Background(), st, "fake", rt, 0, st.pathVars()) }()
	if err := servingWaitForPhase(context.Background(), "(*Executor).activationVeto("); err != nil {
		t.Fatal(err)
	}
	// The command has passed its early check. Pause stores the projection but
	// cannot reach the cancellation sweep while remote work holds its mutex.
	ex.remoteWork.mu.Lock()
	releaseRemote := sync.OnceFunc(ex.remoteWork.mu.Unlock)
	defer releaseRemote()
	manager := &Manager{exec: ex}
	applied := make(chan struct{})
	go func() {
		manager.applyNodeProjection(&Message{Params: json.RawMessage(`{"managementBlocked":true}`)})
		close(applied)
	}()
	if err := servingWaitForPhase(context.Background(), "(*Executor).cancelVetoedRemoteWork("); err != nil {
		t.Fatal(err)
	}
	if rpcCode(context.Cause(ctx)) != codeLocalVeto {
		t.Fatal("stored projection did not cancel execution independently of remote bookkeeping")
	}
	releaseDesired()
	if err := awaitLockCommand(t, done); rpcCode(err) != codeLocalVeto {
		t.Fatalf("command spawned after accepted Pause: %v", err)
	}
	if _, err := os.Stat(marker); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("post-fence process effect: %v", err)
	}
	releaseRemote()
	select {
	case <-applied:
	case <-time.After(5 * time.Second):
		t.Fatal("projection cancellation did not finish")
	}
}

func TestStopPauseOwnershipCommandWaitDoesNotHoldProjectionGate(t *testing.T) {
	ex := wakeTestExecutor(t, t.TempDir(), nil)
	entered, release := make(chan struct{}), make(chan struct{})
	unblock := sync.OnceFunc(func() { close(release) })
	defer unblock()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		close(entered)
		<-release
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(server.Close)
	helper, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		done <- runPrivateLaunchCommandWithStart(context.Background(), []string{helper, "command-test-helper", server.URL}, nil, nil, ex.startActivationCommand)
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("private CLI did not reach wait barrier")
	}
	applied := make(chan struct{})
	go func() { ex.nodeProjection.set(nodeProjection{ManagementBlocked: true}); close(applied) }()
	select {
	case <-applied:
	case <-time.After(5 * time.Second):
		t.Fatal("blocking CLI prevented Pause publication")
	}
	unblock()
	if err := awaitLockCommand(t, done); err != nil {
		t.Fatalf("private CLI completion: %v", err)
	}
}

func TestStopPauseOwnershipNativeDispatchRechecksFinalFence(t *testing.T) {
	test := func(name, action string, cli bool) {
		t.Run(name, func(t *testing.T) {
			ex, st := stopPauseOwnershipChatExecutor(t)
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
			params := json.RawMessage(`{"model":"tiny","messages":[]}`)
			ctx, release, err := ex.guardActionExecution(context.Background(), st, action, params)
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
	test("native chat", "chat", false)
}

func TestStopPauseOwnershipExternalLMStudioOwnerShutdownAllowsRetry(t *testing.T) {
	test := func(name, readback string, readErr error, wantStopped bool) {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))
			t.Cleanup(server.Close)
			unloadMarker, stopMarker := filepath.Join(t.TempDir(), "external-unload"), filepath.Join(t.TempDir(), "external-stop")
			ex := commandStopTestExecutor(t, server.URL, []string{fakeEngineBin, "touch", stopMarker}, 1, true)
			st, err := ex.state("command")
			if err != nil {
				t.Fatal(err)
			}
			st.manifest.Family = "lmstudio"
			st.plat.Runtime.CLI = filepath.Join(t.TempDir(), "private-native-owner")
			st.manifest.Actions = map[string]Action{"unload_model": {Cmd: []string{fakeEngineBin, "touch", unloadMarker}}}
			if err := ex.setDesiredEnabled("command", true, intentLocal); err != nil {
				t.Fatal(err)
			}
			ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
			var reads int
			read := func(_ context.Context, argv []string) (string, error) {
				reads++
				if !reflect.DeepEqual(argv, []string{st.plat.Runtime.CLI, "ps", "--json"}) {
					t.Fatalf("external owner readback argv=%v", argv)
				}
				return readback, readErr
			}
			if err := ex.sleepWithNativeReadback("command", read); err == nil || reads != 0 {
				t.Fatalf("live external owner was not protected: err=%v nativeReads=%d", err, reads)
			}
			server.Close()
			status, err := ex.Status("command")
			if err != nil || !status.Running || status.Healthy {
				t.Fatalf("listener refusal alone claimed termination: %+v, %v", status, err)
			}
			err = ex.sleepWithNativeReadback("command", read)
			if (err == nil) != wantStopped || reads != 1 {
				t.Fatalf("owner shutdown retry: err=%v nativeReads=%d wantStopped=%v", err, reads, wantStopped)
			}
			st.mu.Lock()
			running, adopted := st.running, st.adopted
			st.mu.Unlock()
			if running == wantStopped || adopted == wantStopped {
				t.Fatalf("retry ownership: running=%v adopted=%v wantStopped=%v", running, adopted, wantStopped)
			}
			for _, marker := range []string{unloadMarker, stopMarker} {
				if _, err := os.Stat(marker); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("Pause modified external ownership via %s: %v", filepath.Base(marker), err)
				}
			}
			enabled, known, err := ex.desired.get("command")
			if err != nil || !known || !enabled {
				t.Fatalf("retry changed saved On: enabled=%v known=%v err=%v", enabled, known, err)
			}
			if blocked, _ := ex.nodeProjection.managementBlocked(); !blocked {
				t.Fatal("physical proof implicitly resumed the node")
			}
		})
	}
	test("owner stopped server and unloaded models", `[]`, nil, true)
	test("model residency survives server shutdown", `[{"identifier":"tiny"}]`, nil, false)
	test("native readback unavailable", `[]`, errors.New("private owner readback failed"), false)
}

func TestStopPauseOwnershipNativeHTTPResponseDoesNotHoldProjection(t *testing.T) {
	ex, _ := stopPauseOwnershipChatExecutor(t)
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
		_, err := ex.Action(ctx, "fake", "chat", json.RawMessage(`{"model":"tiny","messages":[]}`))
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
