// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net"
	"strings"
	"testing"
	"time"
)

func TestManagerRejectsRemovedLifecycleMethods(t *testing.T) {
	for _, method := range []string{
		"engine:start", "engine:stop", "engine:restart",
		"engine:remote-start", "engine:remote-stop",
	} {
		t.Run(method, func(t *testing.T) {
			var out bytes.Buffer
			// No executor or peer directory: refusal cannot perform an operation
			// or resolve a remote target, even with invalid parameters.
			m := &Manager{codec: NewCodec(&out)}
			id := json.RawMessage(`194`)
			m.handleMessage(context.Background(), &Message{
				JSONRPC: "2.0", ID: &id, Method: method, Params: json.RawMessage(`{}`),
			})
			var reply Message
			if err := json.Unmarshal(out.Bytes(), &reply); err != nil {
				t.Fatal(err)
			}
			if reply.ID == nil || string(*reply.ID) != string(id) || reply.Error == nil || reply.Error.Code != -32601 {
				t.Fatalf("removed method reply: %+v", reply)
			}
		})
	}
}

func TestGlobalMutationBusyRetainsRPCIdentity(t *testing.T) {
	ex := newLifecycleFixture(t)
	ex.mutationAdmission.Lock()
	done, err := ex.trackMutationLocked("")
	ex.mutationAdmission.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	defer done()
	var out bytes.Buffer
	m := NewManager(NewCodec(&out), ex, nil, "")
	id := json.RawMessage(`195`)
	m.runOp(context.Background(), &Message{ID: &id, Method: "engine:uninstall", Params: json.RawMessage(`{"engine":"source"}`)})
	var reply Message
	if err := json.Unmarshal(out.Bytes(), &reply); err != nil {
		t.Fatal(err)
	}
	if reply.Error == nil || reply.Error.Code != -32000 {
		t.Fatalf("global mutation reply: %+v", reply)
	}
	var data struct {
		Kind string `json:"kind"`
	}
	if err := json.Unmarshal(reply.Error.Data, &data); err != nil {
		t.Fatal(err)
	}
	if data.Kind != "busy" || strings.Contains(reply.Error.Message, "switch") {
		t.Fatalf("global mutation error: %+v, data=%+v", reply.Error, data)
	}
}

// TestManagerErrorResponses drives handleMessage / runOp / runAction over
// an in-memory codec and asserts the JSON-RPC error responses for the
// failure paths the e2e harness can't easily reach.
func TestManagerErrorResponses(t *testing.T) {
	newM := func() (*Manager, *bytes.Buffer) {
		var out bytes.Buffer
		ex := NewExecutor(NewRegistry(), NewReporter(nil), func(string, any) {}, t.TempDir())
		return NewManager(NewCodec(&out), ex, nil, ""), &out
	}
	id := json.RawMessage("1")
	ctx := context.Background()

	m, out := newM()
	m.handleMessage(ctx, &Message{JSONRPC: "2.0", ID: &id, Method: "bogus"})
	mustContain(t, out.String(), "-32601")

	m, out = newM()
	m.handleMessage(ctx, &Message{JSONRPC: "2.0", ID: &id, Method: "engine:describe", Params: json.RawMessage(`{"engine":"nope"}`)})
	mustContain(t, out.String(), "unknown engine")

	m, out = newM()
	m.runOp(ctx, &Message{JSONRPC: "2.0", ID: &id, Method: "engine:install", Params: json.RawMessage(`{}`)})
	mustContain(t, out.String(), "engine is required")

	m, out = newM()
	m.runOp(ctx, &Message{JSONRPC: "2.0", ID: &id, Method: "engine:install", Params: json.RawMessage(`{bad}`)})
	mustContain(t, out.String(), "invalid params")

	m, out = newM()
	m.runAction(ctx, &Message{JSONRPC: "2.0", ID: &id, Method: "engine:action", Params: json.RawMessage(`{"engine":"x"}`)})
	mustContain(t, out.String(), "engine and action are required")
}

func TestRunOpRejectsRetiredPortOverride(t *testing.T) {
	var out bytes.Buffer
	ex := NewExecutor(NewRegistry(), NewReporter(nil), func(string, any) {}, t.TempDir())
	m := NewManager(NewCodec(&out), ex, nil, "")
	id := json.RawMessage("1")
	m.runOp(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:install",
		Params: json.RawMessage(`{"engine":"fake","port":70000}`)})
	mustContain(t, out.String(), "unknown field")
}

func TestRunOpRejectsRetiredBindOverride(t *testing.T) {
	var out bytes.Buffer
	ex := NewExecutor(NewRegistry(), NewReporter(nil), func(string, any) {}, t.TempDir())
	m := NewManager(NewCodec(&out), ex, nil, "")
	id := json.RawMessage("1")
	m.runOp(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:install",
		Params: json.RawMessage(`{"engine":"fake","bind":"not-an-ip"}`)})
	mustContain(t, out.String(), "unknown field")
}

func TestRunOpInstallAutostart(t *testing.T) {
	var out bytes.Buffer
	ex := newTestExecutor(t, testEngineManifest(fakeEngineBin))
	t.Cleanup(func() { _ = ex.Stop("fake") })
	m := NewManager(NewCodec(&out), ex, nil, "")
	id := json.RawMessage("1")
	m.runOp(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:install",
		Params: json.RawMessage(`{"engine":"fake"}`)})
	st, _ := ex.Status("fake")
	if !st.Running {
		t.Fatalf("expected running after install+autostart, got %+v", st)
	}
}

func TestStatusQueriesDoNotBlockMessageDispatchDuringEngineOperation(t *testing.T) {
	tests := []struct {
		name   string
		method string
		params json.RawMessage
	}{
		{name: "status", method: "engine:status", params: json.RawMessage(`{"engine":"fake"}`)},
		{name: "get installed", method: "engine:get-installed"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ex := newTestExecutor(t, testEngineManifest(fakeEngineBin))
			st, err := ex.state("fake")
			if err != nil {
				t.Fatal(err)
			}
			managerConn, clientConn := net.Pipe()
			defer managerConn.Close()
			defer clientConn.Close()
			m := NewManager(NewCodec(managerConn), ex, nil, "")
			id := json.RawMessage("1")
			response := make(chan string, 1)
			go func() {
				line, _ := bufio.NewReader(clientConn).ReadString('\n')
				response <- line
			}()

			// A lifecycle operation holds opMu for its full duration. Status
			// reconciliation may wait for that operation, but dispatch must return
			// immediately so the manager can keep reading shutdown and other RPCs.
			st.opMu.Lock()
			returned := make(chan struct{})
			go func() {
				m.handleMessage(context.Background(), &Message{
					JSONRPC: "2.0",
					ID:      &id,
					Method:  tc.method,
					Params:  tc.params,
				})
				close(returned)
			}()

			select {
			case <-returned:
				// Expected: the potentially blocking status work was dispatched.
			case <-time.After(time.Second):
				st.opMu.Unlock()
				t.Fatalf("%s blocked message dispatch while an engine operation held opMu", tc.method)
			}

			st.opMu.Unlock()
			select {
			case line := <-response:
				if !strings.Contains(line, `"id":1`) {
					t.Fatalf("%s returned an unexpected response: %s", tc.method, line)
				}
			case <-time.After(2 * time.Second):
				t.Fatalf("%s did not respond after the engine operation completed", tc.method)
			}
		})
	}
}

func mustContain(t *testing.T, s, sub string) {
	t.Helper()
	if !strings.Contains(s, sub) {
		t.Fatalf("expected %q to contain %q", s, sub)
	}
}
