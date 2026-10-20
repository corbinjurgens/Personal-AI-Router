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
	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
)

// PC3 executes these process cases. On the Mac, only vet and -list are allowed.
func TestOBOffOrigin(t *testing.T) {
	t.Run("local Off", func(t *testing.T) {
		f := newPolicyJointFixture(t)
		c := subscribeOBIntent(t, f.target)
		c.call(t, "engine:stop", map[string]string{"engine": "ollama"})
		c.assertIntent(t, obIntent(false, map[string]engineintent.Origin{"ollama": engineintent.Local}))
	})

	t.Run("peer Off", func(t *testing.T) {
		f := newPolicyJointFixture(t)
		c := subscribeOBIntent(t, f.target)
		f.http(t, "/v1/engines/stop", map[string]any{"engine": "ollama"}, http.StatusOK)
		want := obIntent(false, map[string]engineintent.Origin{"ollama": engineintent.Peer})
		c.assertIntent(t, want)
		// A second peer Off must preserve its origin too.
		f.http(t, "/v1/engines/stop", map[string]any{"engine": "ollama"}, http.StatusOK)
		c.assertIntent(t, want)
	})

	t.Run("On clears", func(t *testing.T) {
		// The switch fixture uses this test executable as a fake engine image,
		// disposable manifests/settings, and loopback engine/controller ports.
		f := newSwitchProcessFixture(t)
		c := subscribeOBIntent(t, f.p)
		c.call(t, "engine:stop", map[string]string{"engine": "ollama"})
		c.assertIntent(t, obIntent(false, map[string]engineintent.Origin{"ollama": engineintent.Local}))
		c.call(t, "engine:start", map[string]string{"engine": "ollama"})
		c.assertIntent(t, obIntent(true, map[string]engineintent.Origin{}))
		assertOBStoredIntent(t, filepath.Join(jointAppDir(f.base), "engine-bin", "engine-state.json"), true, "")
	})

	t.Run("legacy known Off projects local", func(t *testing.T) {
		f := newPolicyJointFixture(t)
		f.target.cleanup()
		path := filepath.Join(jointAppDir(f.base), "engine-bin", "engine-state.json")
		// Seed disk state only while the manager is stopped. Include an
		// unregistered stale origin to prove it cannot leak onto the wire.
		jointWriteJSON(t, path, map[string]any{
			"engines": map[string]bool{"ollama": false},
			"offBy":   map[string]string{"unregistered-ob-engine": "peer"},
		})
		f.startTarget(t)
		c := subscribeOBIntent(t, f.target)
		want := obIntent(false, map[string]engineintent.Origin{"ollama": engineintent.Local})
		assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), want)
		// A separate mutation produces a push without replacing the legacy
		// engine's missing persisted provenance.
		c.call(t, "engine:stop", map[string]string{"engine": "llamacpp"})
		want.OffByEngine["llamacpp"] = engineintent.Local
		c.assertIntent(t, want)
		assertOBStoredIntent(t, path, false, "")
	})

	t.Run("unknown has no origin", func(t *testing.T) {
		f := newPolicyJointFixture(t)
		c := subscribeOBIntent(t, f.target)
		assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), obIntent(false, map[string]engineintent.Origin{}))
		// Publishing another engine's Off must leave never-saved ollama
		// false without attributing that default to a local or peer action.
		c.call(t, "engine:stop", map[string]string{"engine": "llamacpp"})
		c.assertIntent(t, obIntent(false, map[string]engineintent.Origin{"llamacpp": engineintent.Local}))
	})

	t.Run("restart preserves origin", func(t *testing.T) {
		f := newPolicyJointFixture(t)
		c := subscribeOBIntent(t, f.target)
		f.http(t, "/v1/engines/stop", map[string]any{"engine": "ollama"}, http.StatusOK)
		c.assertIntent(t, obIntent(false, map[string]engineintent.Origin{"ollama": engineintent.Peer}))
		c.call(t, "engine:stop", map[string]string{"engine": "llamacpp"})
		want := obIntent(false, map[string]engineintent.Origin{"ollama": engineintent.Peer, "llamacpp": engineintent.Local})
		c.assertIntent(t, want)
		f.target.cleanup()
		f.startTarget(t)
		c = subscribeOBIntent(t, f.target)
		assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), want)
		// Reassert peer Off to obtain a subscribed post-restart push while
		// retaining both the peer and the untouched local origin.
		f.http(t, "/v1/engines/stop", map[string]any{"engine": "ollama"}, http.StatusOK)
		c.assertIntent(t, want)
	})

	t.Run("peer Off cannot downgrade local veto", func(t *testing.T) {
		f := newPolicyJointFixture(t)
		c := subscribeOBIntent(t, f.target)
		c.call(t, "engine:stop", map[string]string{"engine": "ollama"})
		want := obIntent(false, map[string]engineintent.Origin{"ollama": engineintent.Local})
		c.assertIntent(t, want)
		f.http(t, "/v1/engines/stop", map[string]any{"engine": "ollama"}, http.StatusOK)
		c.assertIntent(t, want)
		assertOBStoredIntent(t, filepath.Join(jointAppDir(f.base), "engine-bin", "engine-state.json"), false, engineintent.Local)
		// Remove the fixture's node pause so it cannot mask a lost engine veto.
		c.call(t, "node:set-availability", map[string]string{"state": "available"})
		task1Eventually(t, "engine-specific local veto after peer Off", 10*time.Second, func() bool {
			body := f.http(t, "/v1/engines/start", map[string]any{"engine": "ollama"}, http.StatusConflict)
			var refusal struct {
				Code  string `json:"code"`
				Error string `json:"error"`
			}
			if err := json.Unmarshal(body, &refusal); err != nil {
				t.Fatalf("decode peer start refusal: %v", err)
			}
			return refusal.Code == "local_veto" && refusal.Error == "ollama was turned off on the node itself; start it there"
		})
		assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), want)
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

func obIntent(ollamaOn bool, origins map[string]engineintent.Origin) engineintent.Snapshot {
	enabled := make(map[string]bool)
	for _, name := range engines.Names() {
		enabled[name] = name == "ollama" && ollamaOn
	}
	return engineintent.Snapshot{EnabledByEngine: enabled, OffByEngine: origins}
}

func assertOBIntent(t *testing.T, got, want engineintent.Snapshot) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("intent snapshot = %+v, want %+v", got, want)
	}
}

func assertOBStoredIntent(t *testing.T, path string, enabled bool, origin engineintent.Origin) {
	t.Helper()
	var stored struct {
		Engines map[string]bool                `json:"engines"`
		OffBy   map[string]engineintent.Origin `json:"offBy"`
	}
	if err := json.Unmarshal(readPolicyBytes(t, path), &stored); err != nil {
		t.Fatalf("decode desired state: %v", err)
	}
	if on, known := stored.Engines["ollama"]; !known || on != enabled {
		t.Fatalf("stored ollama intent = %t (known %t), want %t", on, known, enabled)
	}
	got, present := stored.OffBy["ollama"]
	if got != origin || present != (origin != "") {
		t.Fatalf("stored ollama origin = %q (present %t), want %q", got, present, origin)
	}
}
