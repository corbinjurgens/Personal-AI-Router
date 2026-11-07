// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"encoding/json"
	"net/http"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
	"time"

	"nvpair-shared/engineintent"
	"nvpair-shared/jsonrpc"
)

// PC3 executes these process cases. On the Mac, only vet and -list are allowed.
func TestOBOffOrigin(t *testing.T) {
	t.Run("local Off", func(t *testing.T) {
		f := newPolicyJointFixture(t)
		c := subscribeOBIntent(t, f.target)
		c.call(t, "engine:stop", map[string]string{"engine": "llamacpp"})
		c.assertIntent(t, obIntent(false, map[string]engineintent.Origin{"llamacpp": engineintent.Local}))
	})

	t.Run("peer Off", func(t *testing.T) {
		f := newPolicyJointFixture(t, jointAvailable)
		c := subscribeOBIntent(t, f.target)
		f.http(t, "/v1/engines/stop", map[string]any{"engine": "llamacpp"}, http.StatusOK)
		want := obIntent(false, map[string]engineintent.Origin{"llamacpp": engineintent.Peer})
		c.assertIntent(t, want)
		// A second peer Off must preserve its origin too.
		f.http(t, "/v1/engines/stop", map[string]any{"engine": "llamacpp"}, http.StatusOK)
		c.assertIntent(t, want)
	})

	t.Run("On clears", func(t *testing.T) {
		// The switch fixture uses this test executable as a fake engine image,
		// disposable manifests/settings, and loopback engine/controller ports.
		f := newEngineProcessFixture(t)
		c := subscribeOBIntent(t, f.p)
		c.call(t, "engine:stop", map[string]string{"engine": "llamacpp-custom"})
		c.assertIntent(t, obIntent(false, map[string]engineintent.Origin{"llamacpp-custom": engineintent.Local}))
		c.call(t, "engine:start", map[string]string{"engine": "llamacpp-custom"})
		c.assertIntent(t, obIntent(true, map[string]engineintent.Origin{}))
		assertOBStoredIntent(t, filepath.Join(jointAppDir(f.base), "engine-bin", "engine-state.json"), "llamacpp-custom", true, "")
	})

	t.Run("legacy known Off projects local", func(t *testing.T) {
		f := newPolicyJointFixture(t)
		f.target.cleanup()
		path := filepath.Join(jointAppDir(f.base), "engine-bin", "engine-state.json")
		// Seed disk state only while the manager is stopped. Include an
		// unregistered stale origin to prove it cannot leak onto the wire.
		jointWriteJSON(t, path, map[string]any{
			"engines": map[string]bool{"llamacpp-custom": false},
			"offBy":   map[string]string{"unregistered-ob-engine": "peer"},
		})
		f.startTarget(t)
		c := subscribeOBIntent(t, f.target)
		want := obIntent(false, map[string]engineintent.Origin{"llamacpp-custom": engineintent.Local})
		assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), want)
		// A separate mutation produces a push without replacing the legacy
		// engine's missing persisted provenance.
		c.call(t, "engine:stop", map[string]string{"engine": "llamacpp"})
		want.OffByEngine["llamacpp"] = engineintent.Local
		c.assertIntent(t, want)
		assertOBStoredIntent(t, path, "llamacpp-custom", false, "")
	})

	t.Run("unknown has no origin", func(t *testing.T) {
		f := newPolicyJointFixture(t)
		c := subscribeOBIntent(t, f.target)
		assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), obIntent(false, map[string]engineintent.Origin{}))
		// Publishing another engine's Off must leave never-saved custom engine
		// false without attributing that default to a local or peer action.
		c.call(t, "engine:stop", map[string]string{"engine": "llamacpp"})
		c.assertIntent(t, obIntent(false, map[string]engineintent.Origin{"llamacpp": engineintent.Local}))
	})

	t.Run("restart preserves origin", func(t *testing.T) {
		f := newPolicyJointFixture(t, jointAvailable)
		c := subscribeOBIntent(t, f.target)
		f.http(t, "/v1/engines/stop", map[string]any{"engine": "llamacpp-custom"}, http.StatusOK)
		c.assertIntent(t, obIntent(false, map[string]engineintent.Origin{"llamacpp-custom": engineintent.Peer}))
		c.call(t, "engine:stop", map[string]string{"engine": "llamacpp"})
		want := obIntent(false, map[string]engineintent.Origin{"llamacpp-custom": engineintent.Peer, "llamacpp": engineintent.Local})
		c.assertIntent(t, want)
		f.target.cleanup()
		f.startTarget(t)
		c = subscribeOBIntent(t, f.target)
		assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), want)
		// Reassert peer Off to obtain a subscribed post-restart push while
		// retaining both the peer and the untouched local origin.
		f.http(t, "/v1/engines/stop", map[string]any{"engine": "llamacpp-custom"}, http.StatusOK)
		c.assertIntent(t, want)
	})

	t.Run("peer Off preserves local origin", func(t *testing.T) {
		f := newPolicyJointFixture(t, jointAvailable)
		c := subscribeOBIntent(t, f.target)
		c.call(t, "engine:stop", map[string]string{"engine": "llamacpp"})
		want := obIntent(false, map[string]engineintent.Origin{"llamacpp": engineintent.Local})
		c.assertIntent(t, want)
		f.http(t, "/v1/engines/stop", map[string]any{"engine": "llamacpp"}, http.StatusOK)
		c.assertIntent(t, want)
		assertOBStoredIntent(t, filepath.Join(jointAppDir(f.base), "engine-bin", "engine-state.json"), "llamacpp", false, engineintent.Local)
	})

	t.Run("saved local Off permits peer start", func(t *testing.T) {
		f, _, newClient := newSwitchClientFixture(t)
		c := subscribeOBIntent(t, f.p)
		pid := f.pid(t, "llamacpp-custom")
		c.call(t, "engine:stop", map[string]string{"engine": "llamacpp-custom"})
		c.assertIntent(t, obIntent(false, map[string]engineintent.Origin{"llamacpp-custom": engineintent.Local}))
		assertFixtureProcessStopped(t, pid)
		path := filepath.Join(jointAppDir(f.base), "engine-bin", "engine-state.json")
		assertOBStoredIntent(t, path, "llamacpp-custom", false, engineintent.Local)
		// Only the owned fake engine image and loopback controller can start.
		managementHTTP(t, newClient(), f.ports.engineControl, http.MethodPost, "/v1/engines/start", map[string]string{"engine": "llamacpp-custom"}, http.StatusOK)
		c.assertIntent(t, obIntent(true, map[string]engineintent.Origin{}))
		assertOBStoredIntent(t, path, "llamacpp-custom", true, "")
		f.mu.Lock()
		defer f.mu.Unlock()
		if len(f.starts["llamacpp-custom"]) != 2 || len(f.starts["llamacpp"]) != 0 {
			t.Fatalf("peer start changed unexpected engine execution: %v", f.starts)
		}
		if f.starts["llamacpp-custom"][1] == pid {
			t.Fatal("peer start did not replace the stopped fixture engine")
		}
	})
}

// Retain pushes encountered before an RPC reply; callRPC intentionally drops
// them, which would erase the very broker subscription boundary tested here.
type obIntentClient struct {
	p       *interopProcess
	pending []json.RawMessage
}

func subscribeOBIntent(t *testing.T, p *interopProcess) *obIntentClient {
	t.Helper()
	c := &obIntentClient{p: p}
	var result struct {
		Subscribed bool `json:"subscribed"`
	}
	if err := json.Unmarshal(c.call(t, "engine:subscribe", nil), &result); err != nil {
		t.Fatalf("decode engine subscription: %v", err)
	}
	if !result.Subscribed {
		t.Fatal("engine subscription was not accepted")
	}
	return c
}

func (c *obIntentClient) call(t *testing.T, method string, params any) json.RawMessage {
	t.Helper()
	id := json.RawMessage(strconv.FormatInt(newRPCID(), 10))
	sendLine(t, c.p.in, jsonrpc.Message{JSONRPC: "2.0", ID: &id, Method: method, Params: mustMarshal(t, params)})
	timer := time.NewTimer(35 * time.Second)
	defer timer.Stop()
	for {
		select {
		case msg, ok := <-c.p.msgs:
			if !ok {
				t.Fatalf("broker closed before %s reply: %s", method, c.p.output.String())
			}
			if msg.IsNotification() && msg.Method == "engine:intent-changed" {
				c.pending = append(c.pending, msg.Params)
			}
			if msg.ID != nil && string(*msg.ID) == string(id) && msg.Method == "" {
				if msg.Error != nil {
					t.Fatalf("%s: %+v", method, msg.Error)
				}
				return msg.Result
			}
		case <-timer.C:
			t.Fatalf("timeout waiting for %s reply", method)
		}
	}
}

func (c *obIntentClient) assertIntent(t *testing.T, want engineintent.Snapshot) {
	t.Helper()
	got := decodeOBIntent(t, c.call(t, "engine:intent", nil))
	assertOBIntent(t, got, want)
	var raw json.RawMessage
	if len(c.pending) > 0 {
		raw = c.pending[0]
		c.pending = c.pending[1:]
	} else {
		raw = waitForMethod(t, c.p.msgs, "engine:intent-changed", 10*time.Second).Params
	}
	push := decodeOBIntent(t, raw)
	assertOBIntent(t, push, want)
	if !reflect.DeepEqual(push, got) {
		t.Fatalf("subscribed push %+v differs from RPC result %+v", push, got)
	}
}

func decodeOBIntent(t *testing.T, raw json.RawMessage) engineintent.Snapshot {
	t.Helper()
	var snapshot engineintent.Snapshot
	if err := json.Unmarshal(raw, &snapshot); err != nil {
		t.Fatalf("decode intent snapshot: %v", err)
	}
	if err := snapshot.Validate(); err != nil {
		t.Fatalf("invalid intent snapshot: %v", err)
	}
	return snapshot
}

func obIntent(sourceOn bool, origins map[string]engineintent.Origin) engineintent.Snapshot {
	enabled := make(map[string]bool)
	for _, name := range privateFixtureEngineNames() {
		enabled[name] = name == "llamacpp-custom" && sourceOn
	}
	return engineintent.Snapshot{EnabledByEngine: enabled, OffByEngine: origins}
}

func assertOBIntent(t *testing.T, got, want engineintent.Snapshot) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("intent snapshot = %+v, want %+v", got, want)
	}
}

func assertOBStoredIntent(t *testing.T, path, engine string, enabled bool, origin engineintent.Origin) {
	t.Helper()
	var stored struct {
		Engines map[string]bool                `json:"engines"`
		OffBy   map[string]engineintent.Origin `json:"offBy"`
	}
	if err := json.Unmarshal(readPolicyBytes(t, path), &stored); err != nil {
		t.Fatalf("decode desired state: %v", err)
	}
	if on, known := stored.Engines[engine]; !known || on != enabled {
		t.Fatalf("stored %s intent = %t (known %t), want %t", engine, on, known, enabled)
	}
	got, present := stored.OffBy[engine]
	if got != origin || present != (origin != "") {
		t.Fatalf("stored %s origin = %q (present %t), want %q", engine, got, present, origin)
	}
}
