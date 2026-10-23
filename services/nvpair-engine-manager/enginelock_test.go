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
	"os"
	"path/filepath"
	"testing"

	"nvpair-shared/clustertrust"
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
	frames := out.frames()
	if len(frames) != before+1 {
		t.Fatalf("engine:set-lock wrote %d frames, want one reply", len(frames)-before)
	}
	return frames[len(frames)-1]
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
	test := func(name, params string, wantCode int) {
		t.Run(name, func(t *testing.T) {
			id++
			reply := setLockReply(t, m, out, id, params)
			if reply.Error == nil || reply.Error.Code != wantCode {
				t.Fatalf("engine:set-lock %s = %+v, want error code %d", params, reply, wantCode)
			}
		})
	}
	// A well-formed request reaches the unavailable stub.
	test("lock", `{"engine":"fake","locked":true}`, -32000)
	test("unlock", `{"engine":"fake","locked":false}`, -32000)

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

func TestEngineLockStubDoesNotMutate(t *testing.T) {
	m, exec, out := lockTestManager(t)
	if err := exec.desired.set("fake", true, intentLocal); err != nil {
		t.Fatalf("save engine On: %v", err)
	}
	statePath := filepath.Join(exec.baseDir, desiredStateFileName)
	before, err := os.ReadFile(statePath)
	if err != nil {
		t.Fatalf("read desired state: %v", err)
	}
	statusBefore, err := exec.Status("fake")
	if err != nil {
		t.Fatalf("status before: %v", err)
	}

	reply := setLockReply(t, m, out, 1, `{"engine":"fake","locked":true}`)
	if reply.Error == nil || reply.Error.Code != -32000 || reply.Error.Message != engineLockUnavailableMessage {
		t.Fatalf("engine:set-lock = %+v, want -32000 %q", reply, engineLockUnavailableMessage)
	}
	if reply.Error.Data != nil {
		t.Fatalf("engine:set-lock error data = %s, want none", reply.Error.Data)
	}

	after, err := os.ReadFile(statePath)
	if err != nil {
		t.Fatalf("read desired state after: %v", err)
	}
	if !bytes.Equal(before, after) {
		t.Fatalf("desired state changed:\nbefore %s\nafter  %s", before, after)
	}
	statusAfter, err := exec.Status("fake")
	if err != nil {
		t.Fatalf("status after: %v", err)
	}
	if statusAfter != statusBefore || statusAfter.Locked != nil {
		t.Fatalf("status = %+v, want unchanged %+v with lock unavailable", statusAfter, statusBefore)
	}
	for _, frame := range out.frames() {
		if frame.Method != "" {
			t.Fatalf("stub emitted %s", frame.Method)
		}
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
