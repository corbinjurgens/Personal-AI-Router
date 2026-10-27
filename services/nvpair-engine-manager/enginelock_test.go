// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/durablefile"
	"nvpair-shared/engineintent"
	settings "nvpair-shared/enginesettings"
)

// lockTestManager is a Manager over a registered, never-started fake engine.
func lockTestManager(t *testing.T) (*Manager, *Executor, *syncBuffer) {
	t.Helper()
	exec := newTestExecutor(t, testEngineManifest(fakeEngineBin))
	out := &syncBuffer{}
	dir := t.TempDir()
	m := NewManager(NewCodec(out), exec, clustertrust.Open(dir), dir)
	t.Cleanup(m.controlHTTP.CloseIdle)
	t.Cleanup(m.controlReadyHTTP.CloseIdle)
	return m, exec, out
}

// setLockReply sends one engine:set-lock request and returns its reply.
func setLockReply(t *testing.T, m *Manager, out *syncBuffer, id int, params string) Message {
	t.Helper()
	before := len(out.frames())
	rawID := json.RawMessage(mustJSONText(t, id))
	m.handleMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &rawID, Method: setLockMethod, Params: json.RawMessage(params)})
	var reply Message
	awaitLockCondition(t, "engine:set-lock reply", func() bool {
		for _, frame := range out.frames()[before:] {
			if frame.ID != nil && bytes.Equal(*frame.ID, rawID) {
				reply = frame
				return true
			}
		}
		return false
	})
	return reply
}

func mustJSONText(t *testing.T, v any) string {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("encode %v: %v", v, err)
	}
	return string(data)
}

func TestEngineLockRequestContract(t *testing.T) {
	m, _, out := lockTestManager(t)
	id := 0
	for _, params := range []string{`{"engine":"fake","locked":true}`, `{"engine":"fake","locked":false}`} {
		id++
		reply := setLockReply(t, m, out, id, params)
		if reply.Error != nil {
			t.Fatalf("engine:set-lock %s = %+v, want success", params, reply)
		}
	}
	test := func(name, params string, wantCode int) {
		t.Run(name, func(t *testing.T) {
			id++
			reply := setLockReply(t, m, out, id, params)
			if reply.Error == nil || reply.Error.Code != wantCode {
				t.Fatalf("engine:set-lock %s = %+v, want error code %d", params, reply, wantCode)
			}
		})
	}
	test("rejects missing engine", `{"locked":true}`, -32602)
	test("rejects empty engine", `{"engine":"","locked":true}`, -32602)
	test("rejects unregistered engine", `{"engine":"other","locked":true}`, -32602)
	test("rejects missing locked", `{"engine":"fake"}`, -32602)
	test("rejects null locked", `{"engine":"fake","locked":null}`, -32602)
	test("rejects string locked", `{"engine":"fake","locked":"true"}`, -32602)
	test("rejects node", `{"engine":"fake","locked":true,"node":"peer"}`, -32602)
	test("rejects nodeId", `{"engine":"fake","locked":true,"nodeId":"peer"}`, -32602)
	test("rejects caller", `{"engine":"fake","locked":true,"caller":"uuid-peer"}`, -32602)
	test("rejects origin", `{"engine":"fake","locked":true,"origin":"local"}`, -32602)
	test("rejects unknown field", `{"engine":"fake","locked":true,"extra":1}`, -32602)
	test("rejects array params", `[{"engine":"fake","locked":true}]`, -32602)
	test("rejects trailing data", `{"engine":"fake","locked":true}{}`, -32602)
	test("rejects duplicate locked", `{"engine":"fake","locked":true,"locked":false}`, -32602)
	test("rejects duplicate engine", `{"engine":"other","engine":"fake","locked":true}`, -32602)
	test("rejects case variant of locked", `{"engine":"fake","Locked":true}`, -32602)
	test("rejects case variant beside locked", `{"engine":"fake","locked":true,"LOCKED":false}`, -32602)
	test("rejects case variant of engine", `{"Engine":"fake","locked":true}`, -32602)
	test("rejects missing params", ``, -32602)

	t.Run("no ec route", func(t *testing.T) {
		mux := (&controlServer{exec: m.exec, mesh: m.mesh}).mux()
		for _, path := range []string{"/v1/engine-lock", "/v1/engines/lock", "/v1/set-lock"} {
			rec := httptest.NewRecorder()
			mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, path, bytes.NewReader([]byte(`{"engine":"fake","locked":true}`))))
			if rec.Code != http.StatusNotFound {
				t.Fatalf("POST %s = %d, want 404", path, rec.Code)
			}
		}
	})
}

func TestEngineLockPersistsOffAndUnlockDoesNotStart(t *testing.T) {
	m, exec, out := lockTestManager(t)
	t.Cleanup(exec.StopAll)
	if err := exec.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("start engine: %v", err)
	}
	statePath := filepath.Join(exec.baseDir, desiredStateFileName)
	reply := setLockReply(t, m, out, 1, `{"engine":"fake","locked":true}`)
	if reply.Error != nil {
		t.Fatalf("lock = %+v, want success", reply)
	}
	after, err := os.ReadFile(statePath)
	if err != nil {
		t.Fatalf("read desired state after: %v", err)
	}
	var state desiredStateFile
	if err := json.Unmarshal(after, &state); err != nil {
		t.Fatalf("decode desired state: %v", err)
	}
	if state.Engines["fake"] || !state.Locked["fake"] || state.OffBy["fake"] != intentLocal {
		t.Fatalf("locked state = %+v, want locked and locally Off", state)
	}
	statusAfter, err := exec.Status("fake")
	if err != nil {
		t.Fatalf("status after: %v", err)
	}
	if statusAfter.Locked == nil || !*statusAfter.Locked {
		t.Fatalf("status = %+v, want locked", statusAfter)
	}
	reply = setLockReply(t, m, out, 2, `{"engine":"fake","locked":false}`)
	if reply.Error != nil {
		t.Fatalf("unlock = %+v, want success", reply)
	}
	enabled, known, err := exec.desired.get("fake")
	if err != nil || !known || enabled {
		t.Fatalf("intent after unlock = %v, %v, %v; want known Off", enabled, known, err)
	}
	locked, err := exec.desired.locked("fake")
	if err != nil || locked {
		t.Fatalf("lock after unlock = %v, %v; want false", locked, err)
	}
	requireOffOrigin(t, exec.desired, "fake", "local")
	status, err := exec.Status("fake")
	if err != nil || status.Running || status.Locked == nil || *status.Locked {
		t.Fatalf("unlock status = %+v, %v; want unlocked and stopped", status, err)
	}
}

func requireOffOrigin(t *testing.T, store *desiredStateStore, engine string, want engineintent.Origin) {
	t.Helper()
	got, err := store.snapshot([]string{engine})
	if err != nil || got.OffByEngine[engine] != want || (want != "" && got.EnabledByEngine[engine]) {
		t.Fatalf("intent for %s = %+v, %v; want Off origin %q", engine, got, err, want)
	}
}

func lockLoadManifest() *Manifest {
	m := testEngineManifest(fakeEngineBin)
	m.Actions["load_model"] = Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/api/generate"}}
	return m
}

func awaitLockCondition(t *testing.T, what string, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !condition() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

func awaitLockCommand(t *testing.T, done <-chan error) error {
	t.Helper()
	select {
	case err := <-done:
		return err
	case <-time.After(10 * time.Second):
		t.Fatal("lock command did not finish")
		return nil
	}
}

func TestLockReadLoopProcessesSettingsReplyWhileLifecycleBusy(t *testing.T) {
	ex := settingsExecutor(t, false)
	out := &syncBuffer{}
	input, writer := io.Pipe()
	t.Cleanup(func() { _ = writer.Close(); _ = input.Close() })
	dir := t.TempDir()
	m := NewManager(NewCodec(struct {
		io.Reader
		io.Writer
	}{input, out}), ex, clustertrust.Open(dir), dir)
	t.Cleanup(m.controlHTTP.CloseIdle)
	t.Cleanup(m.controlReadyHTTP.CloseIdle)
	sent := make(chan settings.Relay, 1)
	m.settingsRelay.send = func(_ context.Context, _ string, value any) error {
		request, ok := value.(settings.Relay)
		if !ok {
			return errors.New("unexpected settings relay payload")
		}
		sent <- request
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	request := settingsRequest(t, ex)
	configured := make(chan error, 1)
	go func() {
		_, err := ex.ConfigureLaunch(ctx, settings.Configure{Engine: "fake", Settings: request.Settings}, func() error {
			_, err := m.settingsRelay.call(ctx, "rebind", request, "")
			return err
		})
		configured <- err
	}()
	var rebind settings.Relay
	select {
	case rebind = <-sent:
	case <-ctx.Done():
		t.Fatal("settings never reached rebind with lifecycle ownership")
	}
	readDone := make(chan error, 1)
	go func() { readDone <- m.readLoop(ctx) }()
	encoder := json.NewEncoder(writer)
	if err := encoder.Encode(map[string]any{"jsonrpc": "2.0", "id": 1, "method": setLockMethod, "params": map[string]any{"engine": "fake", "locked": true}}); err != nil {
		t.Fatal(err)
	}
	awaitLockCondition(t, "lock persisted while settings owns opMu", func() bool { locked, err := ex.desired.locked("fake"); return err == nil && locked })
	if err := encoder.Encode(map[string]any{"jsonrpc": "2.0", "method": "engine:settings-reply", "params": settingsReply{ID: rebind.ID, Result: json.RawMessage(`{}`)}}); err != nil {
		t.Fatal(err)
	}
	if err := awaitLockCommand(t, configured); err != nil {
		t.Fatalf("settings reply could not pass set-lock: %v", err)
	}
	awaitLockCondition(t, "lock acknowledgement after rebind", func() bool {
		for _, frame := range out.frames() {
			if frame.ID != nil && string(*frame.ID) == "1" {
				if frame.Error != nil {
					t.Fatalf("lock after rebind: %+v", frame.Error)
				}
				return true
			}
		}
		return false
	})
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	if err := awaitLockCommand(t, readDone); err != nil {
		t.Fatal(err)
	}
}

func TestLockReclaimsUntrackedManagedOrphan(t *testing.T) {
	for _, scenario := range []string{"healthy listener", "unhealthy listener", "termination unverified"} {
		t.Run(scenario, func(t *testing.T) {
			manifest, managedDir, port := restoreOrphanManifest(t)
			if scenario == "unhealthy listener" {
				platform := manifest.Platforms[hostKey()]
				platform.Runtime.Ready.Status = http.StatusCreated
				manifest.Platforms[hostKey()] = platform
			}
			ex := restoreOrphanExecutor(t, manifest, managedDir)
			orphan := spawnFakeListener(t, manifest.Platforms[hostKey()].Runtime.Bin, port)
			st, err := ex.state("fake")
			if err != nil {
				t.Fatal(err)
			}
			if st.running || st.proc != nil {
				t.Fatal("fixture already represents the orphan as running")
			}
			if scenario == "termination unverified" {
				// A live previous-PID fence refuses signalling the new listener.
				// The test process is evidence only, never signal authority.
				st.reclaimPID = os.Getpid()
			}
			err = ex.setEngineLocked("fake", true)
			status, statusErr := ex.Status("fake")
			if statusErr != nil || status.Locked == nil || !*status.Locked {
				t.Fatalf("lock readback = %+v, %v", status, statusErr)
			}
			if scenario == "termination unverified" {
				if err == nil || rpcCode(err) != -32000 || !status.Running || status.Healthy || !pidAlive(orphan.Process.Pid) {
					t.Fatalf("failed termination facts = %+v, %v; want live, unhealthy, locked", status, err)
				}
				if err := ex.setEngineLocked("fake", false); err != nil {
					t.Fatal(err)
				}
				if err := ex.Start(context.Background(), "fake"); err == nil {
					t.Fatal("unlock allowed an unverified orphan to be treated as an owned running engine")
				}
			} else if err != nil || status.Running || portServing(port) || pidAlive(orphan.Process.Pid) {
				t.Fatalf("orphan lock = %+v, %v; want verified termination", status, err)
			}
			requireOffOrigin(t, ex.desired, "fake", "local")
		})
	}
}

func TestLockReadbackDocumentationIsCurrent(t *testing.T) {
	data, err := os.ReadFile("README.md")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "This service does not populate it until lock enforcement lands.") {
		t.Fatal("EngineStatus documentation still describes lock readback as a stub")
	}
}

func TestLocalLoadRPCCancellationRetainsVetoCode(t *testing.T) {
	for _, scenario := range []struct {
		name       string
		projection string
		code       int
	}{
		{"lock", "", -32013},
		{"local pause", `{"pausedLocally":true,"managementBlocked":true}`, -32011},
		{"peer pause", `{"policy":{"pausedBy":"peer"},"pausedLocally":false,"managementBlocked":true}`, -32011},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			ex := newTestExecutor(t, lockLoadManifest())
			t.Cleanup(ex.StopAll)
			if err := ex.Start(context.Background(), "fake"); err != nil {
				t.Fatal(err)
			}
			out := &syncBuffer{}
			m := &Manager{exec: ex, codec: NewCodec(out)}
			entered := make(chan struct{})
			ex.client.Transport = memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
				if request.URL.Path != "/api/generate" {
					return nil, errors.New("unexpected action request")
				}
				close(entered)
				<-request.Context().Done()
				return nil, request.Context().Err()
			})
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			id := json.RawMessage(`42`)
			done := make(chan error, 1)
			go func() {
				m.runModelOperation(ctx, &Message{ID: &id, Method: loadModelMethod, Params: json.RawMessage(`{"engine":"fake","model":"tiny"}`)})
				done <- nil
			}()
			select {
			case <-entered:
			case <-ctx.Done():
				t.Fatal("local load never dispatched")
			}
			locked := make(chan error, 1)
			if scenario.projection == "" {
				go func() { locked <- ex.setEngineLocked("fake", true) }()
			} else {
				m.applyNodeProjection(&Message{Params: json.RawMessage(scenario.projection)})
			}
			if err := awaitLockCommand(t, done); err != nil {
				t.Fatal(err)
			}
			frames := out.frames()
			if len(frames) != 1 || frames[0].Error == nil || frames[0].Error.Code != scenario.code {
				t.Fatalf("local load RPC = %+v, want code %d", frames, scenario.code)
			}
			if scenario.projection == "" {
				if err := awaitLockCommand(t, locked); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

func TestServingLoadStopFenceRetainsVetoCause(t *testing.T) {
	for _, code := range []int{-32013, -32011} {
		t.Run(mustJSONText(t, code), func(t *testing.T) {
			st := &engineState{}
			finish := stopServingLoadsWithCause(st, &localVetoError{code: code, reason: "activation cancelled"})
			ctx, cleanup := registerServingLoad(context.Background(), st)
			if rpcCode(context.Cause(ctx)) != code {
				t.Fatalf("load registered inside stop fence = %v, want %d", context.Cause(ctx), code)
			}
			cleanup()
			finish()
			ctx, cleanup = registerServingLoad(context.Background(), st)
			defer cleanup()
			if ctx.Err() != nil {
				t.Fatalf("completed stop left a cancellation fence: %v", ctx.Err())
			}
		})
	}
}

func TestDeleteRestartRefusalRetainsHTTPCode(t *testing.T) {
	for _, scenario := range []struct {
		name string
		code string
	}{
		{"pause", "local_veto"},
		{"lock", "engine_locked"},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			manifest := testEngineManifest(fakeEngineBin)
			manifest.Actions["delete_model"] = Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/delete"}, RestartAfter: true}
			ex := newTestExecutor(t, manifest)
			t.Cleanup(ex.StopAll)
			ex.modelMutationParent = func(context.Context, string) error { return nil }
			if err := ex.Start(context.Background(), "fake"); err != nil {
				t.Fatal(err)
			}
			ex.client.Transport = memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
				if request.URL.Path != "/delete" {
					return nil, errors.New("unexpected deletion request")
				}
				if scenario.name == "pause" {
					(&Manager{exec: ex}).applyNodeProjection(&Message{Params: json.RawMessage(`{"managementBlocked":true}`)})
				} else if _, err := ex.desired.setLocked("fake", true); err != nil {
					return nil, err
				}
				return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{}`))}, nil
			})
			reply := postControl(t, (&controlServer{exec: ex}).handleDelete, controlDeletePath, `{"engine":"fake","model":"tiny"}`)
			var body codedBody
			if err := json.Unmarshal(reply.Body.Bytes(), &body); err != nil || reply.Code != http.StatusConflict || body.Code != scenario.code {
				t.Fatalf("delete restart refusal = %d %s, %v; want 409 %s", reply.Code, reply.Body.String(), err, scenario.code)
			}
		})
	}
}

func TestLockBusyDuringSwitchDoesNotChangeState(t *testing.T) {
	for _, locked := range []bool{true, false} {
		t.Run(mustJSONText(t, locked), func(t *testing.T) {
			ex, p := switchFixture(t)
			prepareSwitch(t, ex, p)
			before, err := os.ReadFile(ex.desired.path)
			if err != nil {
				t.Fatal(err)
			}
			var busy *switchBusyError
			if err := ex.setEngineLocked("target", locked); !errors.As(err, &busy) || err.Error() != `engine "target" is reserved by an engine switch (busy)` {
				t.Fatalf("lock during switch = %v, want existing busy refusal", err)
			}
			after, err := os.ReadFile(ex.desired.path)
			if err != nil || !bytes.Equal(before, after) {
				t.Fatalf("busy lock changed desired state: %s, %v", after, err)
			}
		})
	}
}

func TestLockCancelsStartAndLoadBeforeLifecycleWait(t *testing.T) {
	ex := newTestExecutor(t, lockLoadManifest())
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	ex.nodeProjection.set(nodeProjection{ManagementBlocked: false})
	remoteCtx, finish, err := ex.beginRemote(context.Background(), "fake")
	if err != nil {
		t.Fatal(err)
	}
	defer finish()
	startCtx, finishStart := ex.startContext(context.Background(), st)
	defer finishStart()
	st.opMu.Lock()
	unlock := sync.OnceFunc(st.opMu.Unlock)
	defer unlock()
	loadDone := make(chan error, 1)
	go func() {
		_, err := ex.ModelLoad(context.Background(), "fake", "tiny")
		loadDone <- err
	}()
	awaitLockCondition(t, "load registration", func() bool {
		st.mu.Lock()
		defer st.mu.Unlock()
		return len(st.loads.active) == 1
	})
	locked := make(chan error, 1)
	go func() { locked <- ex.setEngineLocked("fake", true) }()
	awaitLockCondition(t, "start cancellation before lifecycle release", func() bool { return startCtx.Err() != nil && remoteCtx.Err() != nil })
	if rpcCode(context.Cause(remoteCtx)) != codeEngineLocked {
		t.Fatalf("remote cancellation = %v, want -32013", context.Cause(remoteCtx))
	}
	if rpcCode(context.Cause(startCtx)) != -32013 {
		t.Fatalf("local start cancellation = %v, want -32013", context.Cause(startCtx))
	}
	if err := awaitLockCommand(t, loadDone); rpcCode(err) != -32013 {
		t.Fatalf("waiting local load = %v, want -32013", err)
	}
	requireOffOrigin(t, ex.desired, "fake", "local")
	select {
	case err := <-locked:
		t.Fatalf("lock acknowledged before lifecycle ownership: %v", err)
	default:
	}
	unlock()
	if err := awaitLockCommand(t, locked); err != nil {
		t.Fatal(err)
	}
	t.Run("local start awaiting readiness", func(t *testing.T) {
		ex := newTestExecutor(t, slowStartManifest())
		t.Cleanup(ex.StopAll)
		st, err := ex.state("fake")
		if err != nil {
			t.Fatal(err)
		}
		started := make(chan error, 1)
		go func() { started <- ex.Start(context.Background(), "fake") }()
		awaitLockCondition(t, "local readiness wait", func() bool { st.mu.Lock(); defer st.mu.Unlock(); return st.proc != nil })
		locked := make(chan error, 1)
		go func() { locked <- ex.setEngineLocked("fake", true) }()
		if err := awaitLockCommand(t, started); rpcCode(err) != -32013 {
			t.Fatalf("cancelled local start = %v, want -32013", err)
		}
		if err := awaitLockCommand(t, locked); err != nil {
			t.Fatal(err)
		}
	})
}

func TestConcurrentLockUnlockWaitsForStopAcknowledgement(t *testing.T) {
	ex := newTestExecutor(t, lockLoadManifest())
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.opMu.Lock()
	unlock := sync.OnceFunc(st.opMu.Unlock)
	defer unlock()
	lockDone := make(chan error, 1)
	go func() { lockDone <- ex.setEngineLocked("fake", true) }()
	awaitLockCondition(t, "durable lock", func() bool { locked, err := ex.desired.locked("fake"); return err == nil && locked })
	unlockEntered := make(chan struct{})
	unlockDone := make(chan error, 1)
	go func() { close(unlockEntered); unlockDone <- ex.setEngineLocked("fake", false) }()
	<-unlockEntered
	// Let the second client attempt admission while the first waits for stop.
	select {
	case err := <-unlockDone:
		t.Fatalf("unlock overtook pending lock: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	if locked, err := ex.desired.locked("fake"); err != nil || !locked {
		t.Fatalf("pending lock lost durable fence: %v, %v", locked, err)
	}
	unlock()
	if err := awaitLockCommand(t, lockDone); err != nil {
		t.Fatal(err)
	}
	if err := awaitLockCommand(t, unlockDone); err != nil {
		t.Fatal(err)
	}
	if locked, err := ex.desired.locked("fake"); err != nil || locked {
		t.Fatalf("ordered unlock = %v, %v", locked, err)
	}
	requireOffOrigin(t, ex.desired, "fake", "local")
}

func TestLockCannotBeOverwrittenByLateStart(t *testing.T) {
	s, ex := vetoTestServer(t, lockLoadManifest())
	ex.nodeProjection.set(nodeProjection{ManagementBlocked: false})
	lockDone := make(chan error, 1)
	persisted := make(chan struct{})
	var announce sync.Once
	ex.emit = func(method string, value any) {
		if status, ok := value.(EngineStatus); ok && method == "engine:state-changed" && status.Locked != nil && *status.Locked {
			announce.Do(func() { close(persisted) })
		}
	}
	var lockOnce sync.Once
	ex.client.Transport = &crashReadyTransport{base: ex.client.Transport, ready: func() {
		lockOnce.Do(func() {
			go func() { lockDone <- ex.setEngineLocked("fake", true) }()
			select {
			case <-persisted:
			case <-time.After(5 * time.Second):
				t.Error("lock did not persist while readiness was held")
			}
		})
	}}
	rec := postControl(t, s.handleStart, controlStartPath, `{"engine":"fake"}`)
	var body codedBody
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil || rec.Code != http.StatusConflict || body.Code != "engine_locked" {
		t.Fatalf("late start = %d %s, %v", rec.Code, rec.Body.String(), err)
	}
	if err := awaitLockCommand(t, lockDone); err != nil {
		t.Fatal(err)
	}
	if err := ex.setDesiredEnabled("fake", true, intentPeer); rpcCode(err) != codeEngineLocked {
		t.Fatalf("late On write = %v, want -32013", err)
	}
	requireOffOrigin(t, ex.desired, "fake", "local")
	status, err := ex.Status("fake")
	if err != nil || status.Running || status.Locked == nil || !*status.Locked {
		t.Fatalf("late readiness defeated lock: %+v, %v", status, err)
	}
}

func TestLockedActivationPaths(t *testing.T) {
	for _, path := range []struct {
		name string
		call func(*Executor, *engineState) error
	}{
		{"local start", func(ex *Executor, _ *engineState) error { return ex.Start(context.Background(), "fake") }},
		{"peer start", func(ex *Executor, _ *engineState) error {
			return ex.StartWith(context.Background(), "fake", startOpts{Origin: intentPeer})
		}},
		{"restart", func(ex *Executor, _ *engineState) error { return ex.Restart(context.Background(), "fake") }},
		{"install with start", func(ex *Executor, _ *engineState) error {
			return ex.InstallWithStart(context.Background(), context.Background(), "fake", startOpts{})
		}},
		{"typed load", func(ex *Executor, _ *engineState) error {
			_, err := ex.ModelLoad(context.Background(), "fake", "tiny")
			return err
		}},
		{"manifest action load", func(ex *Executor, _ *engineState) error {
			_, err := ex.Action(context.Background(), "fake", "load_model", json.RawMessage(`{"model":"tiny"}`))
			return err
		}},
		{"restart after action", func(ex *Executor, st *engineState) error {
			return ex.restartAfterAction(context.Background(), st, "fake", "delete_model")
		}},
	} {
		t.Run(path.name, func(t *testing.T) {
			ex := newTestExecutor(t, lockLoadManifest())
			t.Cleanup(ex.StopAll)
			ex.nodeProjection.set(nodeProjection{ManagementBlocked: false})
			// Restart-after is dispatched only for an observed running engine.
			if path.name == "restart after action" {
				if err := ex.Start(context.Background(), "fake"); err != nil {
					t.Fatal(err)
				}
				if _, err := ex.desired.setLocked("fake", true); err != nil {
					t.Fatal(err)
				}
			} else if err := ex.setEngineLocked("fake", true); err != nil {
				t.Fatal(err)
			}
			st, err := ex.state("fake")
			if err != nil {
				t.Fatal(err)
			}
			if err := path.call(ex, st); rpcCode(err) != codeEngineLocked {
				t.Fatalf("%s = %v, want -32013", path.name, err)
			}
			if st.proc != nil {
				t.Fatal("locked path spawned a process")
			}
			requireOffOrigin(t, ex.desired, "fake", "local")
		})
	}
	t.Run("restore and wake preserve Off", func(t *testing.T) {
		ex := newTestExecutor(t, lockLoadManifest())
		if err := ex.setEngineLocked("fake", true); err != nil {
			t.Fatal(err)
		}
		if err := ex.RestoreEnabled(context.Background()); err != nil {
			t.Fatal(err)
		}
		if err := ex.Wake(context.Background(), "fake"); !errors.Is(err, errIntentOff) {
			t.Fatalf("wake = %v, want saved-Off refusal", err)
		}
		status, err := ex.Status("fake")
		if err != nil || status.Running {
			t.Fatalf("automatic locked activation: %+v, %v", status, err)
		}
	})
}

func TestLockedSwitchRefusesBeforeSourceStop(t *testing.T) {
	for _, origin := range []string{settings.SwitchOriginLocal, settings.SwitchOriginRemote} {
		t.Run(origin, func(t *testing.T) {
			ex, p := switchFixture(t)
			if err := ex.Start(context.Background(), "source"); err != nil {
				t.Fatal(err)
			}
			if err := ex.setEngineLocked("target", true); err != nil {
				t.Fatal(err)
			}
			p.Origin = origin
			p.ConfigurationFence = ex.SwitchPrepare(context.Background(), settings.SwitchPrepareRequest{}).ConfigurationFence
			result := ex.SwitchPrepare(context.Background(), p)
			if result.Error == nil || result.Error.Code != -32013 || result.Error.Kind != "engine_locked" {
				t.Fatalf("locked switch = %+v", result)
			}
			status, err := ex.Status("source")
			if err != nil || !status.Running {
				t.Fatalf("locked switch stopped source: %+v, %v", status, err)
			}
		})
	}
	t.Run("pause after prepare precedes source stop", func(t *testing.T) {
		ex, p := switchFixture(t)
		if err := ex.Start(context.Background(), "source"); err != nil {
			t.Fatal(err)
		}
		p = prepareSwitch(t, ex, p)
		ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
		result := ex.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: settings.SwitchStepSourceStop})
		if result.Error == nil || result.Error.Code != -32011 {
			t.Fatalf("paused step = %+v", result)
		}
		status, err := ex.Status("source")
		if err != nil || !status.Running {
			t.Fatalf("paused step stopped source: %+v, %v", status, err)
		}
	})
}

func TestLockStoreFailuresFailClosed(t *testing.T) {
	t.Run("failure before replacement preserves runtime and bytes", func(t *testing.T) {
		ex := newTestExecutor(t, lockLoadManifest())
		t.Cleanup(ex.StopAll)
		if err := ex.Start(context.Background(), "fake"); err != nil {
			t.Fatal(err)
		}
		before, err := os.ReadFile(ex.desired.path)
		if err != nil {
			t.Fatal(err)
		}
		st, err := ex.state("fake")
		if err != nil {
			t.Fatal(err)
		}
		proc := st.proc
		useEngineFileReplacer(t, durablefile.Replacer{CreateTemp: func(string, string, os.FileMode) (durablefile.File, error) {
			return nil, errors.New("injected replacement failure")
		}})
		if err := ex.setEngineLocked("fake", true); err == nil {
			t.Fatal("lock acknowledged a failed replacement")
		}
		after, err := os.ReadFile(ex.desired.path)
		if err != nil || !bytes.Equal(before, after) {
			t.Fatalf("failed lock changed bytes: %s, %v", after, err)
		}
		status, err := ex.Status("fake")
		if err != nil || !status.Running || st.proc != proc || status.Locked == nil || *status.Locked {
			t.Fatalf("failed save changed runtime: %+v, %v", status, err)
		}
	})
	for _, document := range []string{`{"locked":null}`, `{"locked":{"fake":null}}`, `{"locked":{"fake":"true"}}`, `{"locked":[]}`, `{"locked":{"fake":true},"engines":{"fake":true}}`, `{`} {
		t.Run(document, func(t *testing.T) {
			ex := newTestExecutor(t, lockLoadManifest())
			var loads atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				loads.Add(1)
				_, _ = w.Write([]byte(`{"ok":true}`))
			}))
			t.Cleanup(server.Close)
			st, err := ex.state("fake")
			if err != nil {
				t.Fatal(err)
			}
			st.port = server.Listener.Addr().(*net.TCPAddr).Port
			st.running = true
			if _, err := ex.ModelLoad(context.Background(), "fake", "tiny"); err != nil || loads.Load() != 1 {
				t.Fatalf("valid-store fixture load = %v, dispatches=%d; want success", err, loads.Load())
			}
			if err := os.WriteFile(ex.desired.path, []byte(document), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := ex.Start(context.Background(), "fake"); err == nil {
				t.Fatal("unreadable store allowed activation")
			}
			_, storeErr := ex.desired.locked("fake")
			if storeErr == nil {
				t.Fatal("invalid document did not fail store read")
			}
			if _, err := ex.ModelLoad(context.Background(), "fake", "tiny"); err == nil || err.Error() != storeErr.Error() || loads.Load() != 1 {
				t.Fatalf("invalid-store load = %v, dispatches=%d; want store error %v without dispatch", err, loads.Load(), storeErr)
			}
			if err := ex.setEngineLocked("fake", false); err == nil {
				t.Fatal("unlock rewrote invalid store")
			}
			status, err := ex.Status("fake")
			if err != nil || status.Locked != nil || !status.Running {
				t.Fatalf("invalid lock published as unlocked: %+v, %v", status, err)
			}
		})
	}
}

func TestLockStopFailureAfterReplacementRetainsFenceAndRetries(t *testing.T) {
	ex := newTestExecutor(t, lockLoadManifest())
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusForbidden) }))
	t.Cleanup(server.Close)
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.port = server.Listener.Addr().(*net.TCPAddr).Port
	st.running, st.adopted = true, true
	for attempt := 0; attempt < 2; attempt++ {
		if err := ex.setEngineLocked("fake", true); err == nil || rpcCode(err) != -32000 {
			t.Fatalf("stop attempt %d = %v, want operational failure", attempt, err)
		}
		status, err := ex.Status("fake")
		if err != nil || !status.Running || status.Locked == nil || !*status.Locked {
			t.Fatalf("failed stop facts = %+v, %v", status, err)
		}
		requireOffOrigin(t, ex.desired, "fake", "local")
	}
	server.Close()
	if err := ex.setEngineLocked("fake", true); err == nil || rpcCode(err) != -32000 {
		t.Fatalf("closed listener without process-exit proof = %v, want refusal", err)
	}
	status, err := ex.Status("fake")
	if err != nil || !status.Running || status.Healthy || status.Locked == nil || !*status.Locked {
		t.Fatalf("closed-listener fence = %+v, %v", status, err)
	}
	requireOffOrigin(t, ex.desired, "fake", "local")
	st.mu.Lock()
	st.reclaimPID = os.Getpid()
	st.mu.Unlock()
	ex.reconcilePresenceStateWithProcessProof(t.Context(), "fake", st, true, st.port, true,
		func(int) (int, string, bool) { return os.Getpid(), "fixture-owner", true },
		func(pid int) bool { return pid == os.Getpid() })
	if err := ex.setEngineLocked("fake", true); err != nil {
		t.Fatalf("retry after positive external process exit: %v", err)
	}
	status, err = ex.Status("fake")
	if err != nil || status.Running || status.Healthy || status.Locked == nil || !*status.Locked {
		t.Fatalf("completed lock after exit = %+v, %v", status, err)
	}
	requireOffOrigin(t, ex.desired, "fake", "local")
}

func TestLockPreservesCheckpointAndUnknownEngines(t *testing.T) {
	ex, p := switchFixture(t)
	p = prepareSwitch(t, ex, p)
	switchStep(t, ex, p, settings.SwitchStepRelease)
	before, err := ex.desired.load()
	if err != nil {
		t.Fatal(err)
	}
	if err := ex.desired.set("future-engine", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	if err := ex.setEngineLocked("target", true); err != nil {
		t.Fatal(err)
	}
	after, err := newDesiredStateStore(ex.baseDir).load()
	if err != nil || !after.Engines["future-engine"] || !reflect.DeepEqual(before.PendingSwitch, after.PendingSwitch) {
		t.Fatalf("lock lost checkpoint or unknown engine: %+v, %v", after, err)
	}
}

func TestLockCommandMutexCoversRPCReply(t *testing.T) {
	m, ex, _ := lockTestManager(t)
	server, client := net.Pipe()
	t.Cleanup(func() { _ = server.Close(); _ = client.Close() })
	m.codec = NewCodec(server)
	id := json.RawMessage(`1`)
	replied := make(chan struct{})
	go func() {
		m.runSetLock(&Message{ID: &id, Params: json.RawMessage(`{"engine":"fake","locked":true}`)})
		close(replied)
	}()
	awaitLockCondition(t, "lock awaiting its RPC acknowledgement", func() bool { locked, err := ex.desired.locked("fake"); return err == nil && locked })
	unlockDone := make(chan error, 1)
	go func() { unlockDone <- ex.setEngineLocked("fake", false) }()
	select {
	case err := <-unlockDone:
		t.Fatalf("unlock passed a blocked lock reply: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	if err := client.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	line, err := bufio.NewReader(client).ReadBytes('\n')
	if err != nil {
		t.Fatal(err)
	}
	var reply Message
	if err := json.Unmarshal(line, &reply); err != nil || reply.Error != nil {
		t.Fatalf("lock reply = %s, %v", line, err)
	}
	<-replied
	if err := awaitLockCommand(t, unlockDone); err != nil {
		t.Fatal(err)
	}
}

func TestLockedSettingsAndPauseRecoveryWithholdActivation(t *testing.T) {
	for _, scenario := range []string{"locked", "paused", "paused running"} {
		t.Run(scenario, func(t *testing.T) {
			ex := settingsExecutor(t, false)
			if scenario == "paused running" {
				if err := ex.Start(context.Background(), "fake"); err != nil {
					t.Fatal(err)
				}
			}
			if scenario == "locked" {
				if err := ex.setEngineLocked("fake", true); err != nil {
					t.Fatal(err)
				}
			} else {
				ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
			}
			request := settingsRequest(t, ex)
			port, err := freePort()
			if err != nil {
				t.Fatal(err)
			}
			request.Settings.ServerPort = port
			request.Resolution = "server"
			preview := previewSettings(t, ex, request)
			if len(preview.Errors) != 0 || preview.Conflict != nil {
				t.Fatalf("settings preview: %+v", preview)
			}
			state, err := ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings, Resume: true}, func() error { return nil })
			if err != nil || state.Running || state.ServerPort != port {
				t.Fatalf("configuration activated fenced engine: %+v, %v", state, err)
			}
		})
	}
}

func TestRemoteSwitchOverridesPlainOff(t *testing.T) {
	for _, origin := range []intentOrigin{intentLocal, intentPeer, "legacy", "missing"} {
		t.Run(string(origin), func(t *testing.T) {
			ex, p := switchFixture(t)
			if err := ex.Start(context.Background(), "source"); err != nil {
				t.Fatal(err)
			}
			if origin != "missing" {
				state, err := ex.desired.load()
				if err != nil {
					t.Fatal(err)
				}
				state.Engines["target"] = false
				if origin != "legacy" {
					state.OffBy["target"] = origin
				}
				if err := ex.desired.save(state); err != nil {
					t.Fatal(err)
				}
			}
			p.Origin = settings.SwitchOriginRemote
			p = prepareSwitch(t, ex, p)
			switchStep(t, ex, p, settings.SwitchStepSourceStop)
			switchStep(t, ex, p, settings.SwitchStepTargetStart)
			status, err := ex.Status("target")
			if err != nil || !status.Running {
				t.Fatalf("peer switch over Off = %+v, %v", status, err)
			}
			requireOffOrigin(t, ex.desired, "target", "")
		})
	}
}

// TestEngineLockedCodeKeepsNumericIdentity carries -32013 through every remote
// path that already carries -32010, -32011 and -32012, without merging codes.
func TestEngineLockedCodeKeepsNumericIdentity(t *testing.T) {
	codes := map[int]string{
		codePolicyConflict:         "policy_conflict",
		codeLocalVeto:              "local_veto",
		codeRemoteManagementDenied: "remote_management_denied",
		codeEngineLocked:           "engine_locked",
	}
	if settings.SwitchEngineLockedCode != -32013 || codeEngineLocked != settings.SwitchEngineLockedCode {
		t.Fatalf("engine locked code = %d, switch code %d, want both -32013", codeEngineLocked, settings.SwitchEngineLockedCode)
	}
	for code, wire := range codes {
		t.Run(wire, func(t *testing.T) {
			t.Run("wire name", func(t *testing.T) {
				got, ok := codeOfWire(wire)
				if !ok || got != code || wireCodes[code] != wire {
					t.Fatalf("codeOfWire(%q) = %d, %v; want %d", wire, got, ok, code)
				}
			})
			t.Run("ec conflict body", func(t *testing.T) {
				rec := httptest.NewRecorder()
				writeCodedError(rec, code, "target text")
				var body codedBody
				if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
					t.Fatalf("decode body: %v", err)
				}
				if rec.Code != http.StatusConflict || body.Code != wire || body.Error != "target text" {
					t.Fatalf("coded body = %d %+v, want 409 %q", rec.Code, body, wire)
				}
				err := remoteStatusError("remote", rec.Code, rec.Body.Bytes())
				var coded *codedError
				if !errors.As(err, &coded) || rpcCode(err) != code {
					t.Fatalf("decoded error = %v (code %d), want %d", err, rpcCode(err), code)
				}
			})
			t.Run("stream error frame", func(t *testing.T) {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "application/x-ndjson")
					_ = json.NewEncoder(w).Encode(streamFrame{Type: "error", Message: "target text", Code: wire})
				}))
				defer srv.Close()
				_, err := rmTestClient(srv).stream(context.Background(), controlInstallPath, map[string]string{"engine": "fixture"}, nil)
				if rpcCode(err) != code || err.Error() != "target text" {
					t.Fatalf("stream error = %v (code %d), want %d with the target text", err, rpcCode(err), code)
				}
			})
			t.Run("JSON-RPC reply", func(t *testing.T) {
				out := &syncBuffer{}
				m := &Manager{codec: NewCodec(out)}
				rawID := json.RawMessage(`1`)
				m.respondOrErr(&Message{ID: &rawID}, nil, &codedError{code: code, message: "target text"})
				frames := out.frames()
				if len(frames) != 1 || frames[0].Error == nil || frames[0].Error.Code != code || frames[0].Error.Data != nil {
					t.Fatalf("reply = %+v, want code %d without data", frames, code)
				}
			})
		})
	}
	t.Run("relay reply", func(t *testing.T) {
		r := &switchRelay{}
		r.send = func(_ context.Context, _ string, value any) error {
			request, ok := value.(switchRelayRequest)
			if !ok {
				t.Fatalf("relay request = %T", value)
			}
			raw, err := json.Marshal(switchRelayReply{ID: request.ID, Error: "locked", ErrorCode: codeEngineLocked})
			if err != nil {
				return err
			}
			r.reply(raw)
			return nil
		}
		_, err := r.call(context.Background(), settings.SwitchMethod, "pinned-peer", json.RawMessage(`{}`))
		if rpcCode(err) != codeEngineLocked {
			t.Fatalf("relay code = %d, want %d (%v)", rpcCode(err), codeEngineLocked, err)
		}
	})
}

func TestNodeProjectionManagementBlocked(t *testing.T) {
	test := func(name, params string, wantBlocked, wantPausedLocally bool) {
		t.Run(name, func(t *testing.T) {
			m, exec, _ := lockTestManager(t)
			if _, known := exec.nodeProjection.managementBlocked(); known {
				t.Fatal("management state known before the first projection")
			}
			m.applyNodeProjection(&Message{Params: json.RawMessage(params)})
			blocked, known := exec.nodeProjection.managementBlocked()
			if !known || blocked != wantBlocked {
				t.Fatalf("managementBlocked = %v, %v; want %v, known", blocked, known, wantBlocked)
			}
			if paused, _ := exec.nodeProjection.pausedLocally(); paused != wantPausedLocally {
				t.Fatalf("pausedLocally = %v, want %v", paused, wantPausedLocally)
			}
		})
	}
	test("available", `{"policy":{"revision":"1","roles":{}},"pausedLocally":false,"managementBlocked":false}`, false, false)
	test("peer pause", `{"policy":{"revision":"1","roles":{},"pausedBy":"peer"},"pausedLocally":false,"managementBlocked":true}`, true, false)
	test("local pause", `{"policy":{"revision":"1","roles":{}},"pausedLocally":true,"managementBlocked":true}`, true, true)
	// An older broker's projection is not upgraded from pausedBy.
	test("older broker", `{"policy":{"revision":"1","roles":{},"pausedBy":"peer"},"pausedLocally":false}`, false, false)
}

func TestLockVetoIdentitySurvivesEveryErrorBoundary(t *testing.T) {
	for _, tc := range []struct {
		code                int
		wire, kind, message string
	}{
		{-32011, "local_veto", "remote_veto", "This node is paused. Resume it on this node before managing it remotely."},
		{-32013, "engine_locked", "engine_locked", "This engine is locked on this node. Unlock it on this node before starting or loading a model."},
	} {
		t.Run(tc.wire, func(t *testing.T) {
			veto := &localVetoError{code: tc.code, reason: tc.message}
			if rpcCode(veto) != tc.code {
				t.Fatalf("RPC code = %d", rpcCode(veto))
			}
			rec := httptest.NewRecorder()
			writeRemoteError(rec, veto)
			var body codedBody
			if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if rec.Code != http.StatusConflict || body.Code != tc.wire || body.Error != tc.message {
				t.Fatalf("HTTP refusal = %d %+v", rec.Code, body)
			}
			s := &controlServer{exec: newTestExecutor(t, lockLoadManifest())}
			rec = httptest.NewRecorder()
			s.streamOp(rec, httptest.NewRequest(http.MethodPost, controlInstallPath, nil), "operation", "fake", "install", func(context.Context) (streamFrame, error) { return streamFrame{}, veto })
			var frame streamFrame
			if err := json.Unmarshal(bytes.TrimSpace(rec.Body.Bytes()), &frame); err != nil {
				t.Fatal(err)
			}
			if frame.Type != "error" || frame.Code != tc.wire || frame.Message != tc.message {
				t.Fatalf("terminal frame = %+v", frame)
			}
			failure := switchFailure(switchCode(veto), settings.SwitchErrorTargetStart, veto)
			if failure.Code != tc.code || failure.Kind != tc.kind || failure.Message != tc.message {
				t.Fatalf("switch refusal = %+v", failure)
			}
		})
	}
}

func TestLockPublishesIntentAndLockOnlyState(t *testing.T) {
	for _, seed := range []struct {
		name    string
		enabled bool
		origin  intentOrigin
		intents int
	}{
		{"On becomes Off", true, intentLocal, 1},
		{"peer Off becomes local Off", false, intentPeer, 1},
		{"local Off changes only lock", false, intentLocal, 0},
	} {
		t.Run(seed.name, func(t *testing.T) {
			ex := newTestExecutor(t, lockLoadManifest())
			if err := ex.desired.set("fake", seed.enabled, seed.origin); err != nil {
				t.Fatal(err)
			}
			var intents []engineintent.Snapshot
			var states []bool
			ex.emit = func(method string, value any) {
				if method == intentChangedMethod {
					snapshot, ok := value.(engineintent.Snapshot)
					if !ok {
						t.Errorf("intent push type = %T", value)
						return
					}
					intents = append(intents, snapshot)
				}
				if method == "engine:state-changed" {
					status, ok := value.(EngineStatus)
					if !ok || status.Locked == nil {
						t.Errorf("lock state push = %+v", value)
						return
					}
					states = append(states, *status.Locked)
				}
			}
			if err := ex.setEngineLocked("fake", true); err != nil {
				t.Fatal(err)
			}
			if err := ex.setEngineLocked("fake", false); err != nil {
				t.Fatal(err)
			}
			if len(intents) != seed.intents || !reflect.DeepEqual(states, []bool{true, false}) {
				t.Fatalf("pushes = intents %+v, locks %v", intents, states)
			}
			for _, snapshot := range intents {
				if snapshot.EnabledByEngine["fake"] || snapshot.OffByEngine["fake"] != "local" {
					t.Fatalf("lock intent push = %+v", snapshot)
				}
			}
		})
	}
}

func TestLockedStatusIncludesUninstalledShells(t *testing.T) {
	m := lockLoadManifest()
	m.Platforms = map[string]Platform{}
	ex := newTestExecutor(t, m)
	for _, locked := range []bool{true, false} {
		if _, err := ex.desired.setLocked("fake", locked); err != nil {
			t.Fatal(err)
		}
		status, err := ex.Status("fake")
		if err != nil || status.Locked == nil || *status.Locked != locked {
			t.Fatalf("status shell = %+v, %v", status, err)
		}
		installed := ex.GetInstalled()
		if len(installed) != 1 || installed[0].Locked == nil || *installed[0].Locked != locked {
			t.Fatalf("installed shell = %+v", installed)
		}
	}
}

func TestPauseCleansCommittedLocalSwitch(t *testing.T) {
	ex, p := switchFixture(t)
	if err := ex.Start(context.Background(), "source"); err != nil {
		t.Fatal(err)
	}
	p = prepareSwitch(t, ex, p)
	switchStep(t, ex, p, settings.SwitchStepSourceStop)
	switchStep(t, ex, p, settings.SwitchStepTargetStart)
	ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	result := ex.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: settings.SwitchStepRelease})
	if result.Error == nil || result.Error.Kind != settings.SwitchErrorBusy {
		t.Fatalf("paused committed target released: %+v", result)
	}
	result = ex.SwitchReconcile(context.Background(), settings.SwitchReconcileRequest{SwitchPrepareRequest: p})
	if result.Error == nil || result.Error.Code != -32011 || result.Checkpoint.TargetCommitted || !result.Observed.Target.TerminationVerified || result.Observed.Target.Intent != settings.SwitchIntentOff {
		t.Fatalf("paused local cleanup = %+v", result)
	}
	switchStep(t, ex, p, settings.SwitchStepRelease)
	for _, engine := range []string{"source", "target"} {
		status, err := ex.Status(engine)
		if err != nil || status.Running {
			t.Fatalf("pause left %s running: %+v, %v", engine, status, err)
		}
	}
}

func TestLockAllowsSwitchAwayFromLockedSource(t *testing.T) {
	ex, p := switchFixture(t)
	if err := ex.Start(context.Background(), "source"); err != nil {
		t.Fatal(err)
	}
	if err := ex.setEngineLocked("source", true); err != nil {
		t.Fatal(err)
	}
	p = prepareSwitch(t, ex, p)
	switchStep(t, ex, p, settings.SwitchStepSourceStop)
	switchStep(t, ex, p, settings.SwitchStepTargetStart)
	status, err := ex.Status("target")
	if err != nil || !status.Running {
		t.Fatalf("switch away = %+v, %v", status, err)
	}
	locked, err := ex.desired.locked("source")
	if err != nil || !locked {
		t.Fatalf("switch cleared source lock: %v, %v", locked, err)
	}
}
