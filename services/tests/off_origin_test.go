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
	testSavedOff := func(name string, origins map[string]engineintent.Origin) {
		t.Run(name, func(t *testing.T) {
			f := newPolicyJointFixture(t)
			f.target.cleanup()
			path := filepath.Join(jointAppDir(f.base), "engine-bin", "engine-state.json")
			// Seed pre-existing intent while its owner is stopped. Phase 15 owns
			// removing this state; phase 11 removes the controls that wrote it.
			jointWriteJSON(t, path, map[string]any{
				"engines": map[string]bool{"llamacpp-custom": false}, "offBy": origins,
			})
			f.startTarget(t)
			c := subscribeOBIntent(t, f.target)
			origin := origins["llamacpp-custom"]
			projected := origin
			if projected == "" {
				projected = engineintent.Local
			}
			want := obIntent(false, map[string]engineintent.Origin{"llamacpp-custom": projected})
			assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), want)
			for _, state := range []string{"available", "paused", "available"} {
				c.call(t, "node:set-availability", map[string]string{"state": state})
				assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), want)
				assertOBStoredIntent(t, path, "llamacpp-custom", false, origin)
			}
			f.target.cleanup()
			f.startTarget(t)
			c = subscribeOBIntent(t, f.target)
			assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), want)
		})
	}
	testSavedOff("saved local Off survives pause resume and restart", map[string]engineintent.Origin{"llamacpp-custom": engineintent.Local})
	testSavedOff("saved peer Off survives pause resume and restart", map[string]engineintent.Origin{"llamacpp-custom": engineintent.Peer})
	testSavedOff("legacy known Off projects local and omits unknown origin", map[string]engineintent.Origin{"unregistered-ob-engine": engineintent.Peer})
	t.Run("unknown has no origin", func(t *testing.T) {
		f := newPolicyJointFixture(t)
		c := subscribeOBIntent(t, f.target)
		assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), obIntent(false, map[string]engineintent.Origin{}))
	})
	t.Run("installation start clears saved Off and publishes intent", func(t *testing.T) {
		f := prepareEngineProcessFixture(t, t.TempDir(), reserveFixtureServicePorts(t), "")
		path := filepath.Join(jointAppDir(f.base), "engine-bin", "engine-state.json")
		jointWriteJSON(t, path, map[string]any{
			"engines": map[string]bool{"llamacpp-custom": false},
			"offBy":   map[string]engineintent.Origin{"llamacpp-custom": engineintent.Local},
		})
		f.start(t)
		c := subscribeOBIntent(t, f.p)
		assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), obIntent(false, map[string]engineintent.Origin{"llamacpp-custom": engineintent.Local}))
		c.pending = nil // Only the installation's subsequent intent push is asserted.
		c.call(t, "engine:install", map[string]any{"engine": "llamacpp-custom", "start": true})
		c.assertIntent(t, obIntent(true, map[string]engineintent.Origin{}))
		assertOBStoredIntent(t, path, "llamacpp-custom", true, "")
	})
	t.Run("peer installation start clears saved local Off", func(t *testing.T) {
		f, _, newClient := newEngineClientFixture(t)
		pid := f.pid(t, "llamacpp-custom")
		f.p.cleanup()
		assertFixtureProcessStopped(t, pid)
		path := filepath.Join(jointAppDir(f.base), "engine-bin", "engine-state.json")
		jointWriteJSON(t, path, map[string]any{
			"engines": map[string]bool{"llamacpp-custom": false},
			"offBy":   map[string]engineintent.Origin{"llamacpp-custom": engineintent.Local},
		})
		f.start(t)
		c := subscribeOBIntent(t, f.p)
		assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), obIntent(false, map[string]engineintent.Origin{"llamacpp-custom": engineintent.Local}))
		c.pending = nil
		managementHTTP(t, newClient(), f.ports.engineControl, http.MethodPost, "/v1/engines/install", map[string]any{"engine": "llamacpp-custom", "start": true}, http.StatusOK)
		c.assertIntent(t, obIntent(true, map[string]engineintent.Origin{}))
		assertOBStoredIntent(t, path, "llamacpp-custom", true, "")
		f.mu.Lock()
		defer f.mu.Unlock()
		if len(f.starts["llamacpp-custom"]) != 2 || len(f.starts["llamacpp"]) != 0 || f.starts["llamacpp-custom"][1] == pid {
			t.Fatalf("peer installation start changed unexpected execution: %v", f.starts)
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
