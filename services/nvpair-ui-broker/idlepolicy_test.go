// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

func TestResidentProtectionAutomaticUnload(t *testing.T) {
	for _, reason := range []string{"idle", "admission"} {
		t.Run(reason, func(t *testing.T) {
			b, _ := newPolicyBroker(t)
			worker := attachFakeEngine(t, b)
			r := b.nodePolicyRuntime()
			r.mu.Lock()
			r.current.ResidentModels = nodepolicy.ResidentModels{{Engine: "ollama", Model: "resident"}}
			r.mu.Unlock()
			b.unloadModels("ollama", []string{"resident:latest", "neighbor"}, reason)
			calls := worker.snapshot()
			if len(calls) != 1 || calls[0].method != engineUnloadModelMethod {
				t.Fatalf("automatic unload calls = %+v, want only neighbor unload", calls)
			}
			var params map[string]string
			if err := json.Unmarshal(calls[0].params, &params); err != nil {
				t.Fatalf("decode unload: %v", err)
			}
			if params["model"] != "neighbor" {
				t.Fatalf("unloaded %q, want neighbor", params["model"])
			}
		})
	}
}

func TestResidentProtectionRechecksQueuedUnload(t *testing.T) {
	b, _ := newPolicyBroker(t)
	worker := attachFakeEngine(t, b)
	r := b.nodePolicyRuntime()
	worker.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == engineUnloadModelMethod {
			r.mu.Lock()
			r.current.ResidentModels = nodepolicy.ResidentModels{{Engine: "ollama", Model: "later"}}
			r.mu.Unlock()
		}
		return map[string]bool{"ok": true}, ""
	})
	b.unloadModels("ollama", []string{"first", "later:latest"}, "admission")
	if calls := worker.snapshot(); len(calls) != 1 {
		t.Fatalf("queued unload ignored new resident preference: %+v", calls)
	}
}

func TestResidentProtectionPauseMayUnload(t *testing.T) {
	b, _ := newPolicyBroker(t)
	worker := attachFakeEngine(t, b)
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.ResidentModels = nodepolicy.ResidentModels{{Engine: "ollama", Model: "resident"}}
	r.mu.Unlock()
	b.unloadModels("ollama", []string{"resident"}, "pause")
	if got := worker.methods(""); !reflect.DeepEqual(got, []string{engineUnloadModelMethod}) {
		t.Fatalf("pause calls = %v, want resident unload", got)
	}
}

func TestResidentProtectionIdleSleepPlanning(t *testing.T) {
	b, _ := newPolicyBroker(t)
	worker := attachFakeEngine(t, b)
	r := b.nodePolicyRuntime()
	now := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	r.mu.Lock()
	r.now = func() time.Time { return now }
	r.current.Idle.StopEngineAfterMinutes = 1
	r.current.ResidentModels = nodepolicy.ResidentModels{{Engine: "ollama", Model: "resident"}}
	r.running["ollama"] = true
	r.intent["ollama"] = true
	r.runningSince["ollama"] = now.Add(-2 * time.Minute)
	r.mu.Unlock()
	b.idleTick(context.Background())
	if calls := worker.snapshot(); len(calls) != 0 {
		t.Fatalf("resident engine slept: %+v", calls)
	}
	r.mu.Lock()
	r.current.ResidentModels = nil
	r.mu.Unlock()
	b.idleTick(context.Background())
	if got := worker.methods(""); !reflect.DeepEqual(got, []string{engineSleepMethod}) {
		t.Fatalf("calls after clearing resident preference = %v, want sleep", got)
	}
}

func TestResidentProtectionRechecksQueuedIdleSleep(t *testing.T) {
	b, _ := newPolicyBroker(t)
	worker := attachFakeEngine(t, b)
	r := b.nodePolicyRuntime()
	now := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	r.mu.Lock()
	r.now = func() time.Time { return now }
	r.current.Idle.UnloadAfterMinutes = 1
	r.current.Idle.StopEngineAfterMinutes = 1
	r.running["ollama"] = true
	r.intent["ollama"] = true
	r.runningSince["ollama"] = now.Add(-2 * time.Minute)
	r.loaded["ollama"] = []string{"neighbor"}
	r.mu.Unlock()
	worker.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == engineUnloadModelMethod {
			r.mu.Lock()
			r.current.ResidentModels = nodepolicy.ResidentModels{{Engine: "ollama", Model: "resident"}}
			r.mu.Unlock()
		}
		return map[string]bool{"ok": true}, ""
	})
	b.idleTick(context.Background())
	if got := worker.methods(""); !reflect.DeepEqual(got, []string{engineUnloadModelMethod}) {
		t.Fatalf("queued idle sleep ignored new resident preference: %v", got)
	}
}
