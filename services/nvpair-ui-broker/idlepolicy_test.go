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

func TestIdlePlanningWithdrawsEngineManagerSourceState(t *testing.T) {
	test := func(name string, transition func(*Broker, *engineManagerSource) *engineManagerSource) {
		t.Run(name, func(t *testing.T) {
			b, _ := newPolicyBroker(t)
			r := b.nodePolicyRuntime()
			now := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
			r.mu.Lock()
			r.now = func() time.Time { return now }
			r.current.Idle.StopEngineAfterMinutes = 1
			r.mu.Unlock()
			old := b.activateEngineManagerSource()
			b.observeEngineRunningFrom(old, "ollama", true)
			b.observeLoadedFrom(old, map[string][]string{"ollama": {"model"}})
			b.observeIntentFrom(old, map[string]bool{"ollama": true})
			r.mu.Lock()
			r.runningSince["ollama"] = now.Add(-2 * time.Minute)
			r.residencyAt["ollama"] = now.Add(-2 * time.Minute)
			r.idleUnloadedAt["ollama"] = now.Add(-2 * time.Minute)
			r.idleSleptAt["ollama"] = now.Add(-3 * time.Minute)
			r.mu.Unlock()

			current := transition(b, old)
			// A late callback from the retired source must not restore its state.
			b.observeEngineRunningFrom(old, "ollama", true)
			b.observeLoadedFrom(old, map[string][]string{"ollama": {"model"}})
			b.observeIntentFrom(old, map[string]bool{"ollama": true})
			ctx, cancel := context.WithCancel(context.Background())
			cancel() // Allow planning but prevent any worker RPC.
			b.idleTick(ctx)
			r.mu.Lock()
			planned := r.idleSleptAt["ollama"]
			_, running := r.running["ollama"]
			_, loaded := r.loaded["ollama"]
			_, intent := r.intent["ollama"]
			_, observed := r.runningObserved["ollama"]
			_, runningSince := r.runningSince["ollama"]
			_, residencyAt := r.residencyAt["ollama"]
			_, unloadedAt := r.idleUnloadedAt["ollama"]
			r.mu.Unlock()
			if !planned.IsZero() {
				t.Errorf("idle policy planned sleep for retired source's engine at %v", planned)
			}
			if running || loaded || intent {
				t.Errorf("retired source state remains: running=%v loaded=%v intent=%v", running, loaded, intent)
			}
			if observed || runningSince || residencyAt || unloadedAt {
				t.Errorf("retired source bookkeeping remains: observed=%v runningSince=%v residencyAt=%v unloadedAt=%v", observed, runningSince, residencyAt, unloadedAt)
			}
			if got := b.runningSnapshot(); len(got) != 0 {
				t.Errorf("pause running snapshot retains retired engines: %v", got)
			}
			if got := b.loadedSnapshot(); len(got) != 0 {
				t.Errorf("pause loaded snapshot retains retired engines: %v", got)
			}

			if current == nil {
				current = b.activateEngineManagerSource()
			}
			b.observeEngineRunningFrom(current, "llamacpp", true)
			b.observeLoadedFrom(current, map[string][]string{"llamacpp": {"new-model"}})
			b.observeIntentFrom(current, map[string]bool{"llamacpp": true})
			b.retireEngineManagerSource(old)
			r.mu.Lock()
			newRunning := r.running["llamacpp"]
			newLoaded := append([]string(nil), r.loaded["llamacpp"]...)
			newIntent := r.intent["llamacpp"]
			r.mu.Unlock()
			if !newRunning || !newIntent || !reflect.DeepEqual(newLoaded, []string{"new-model"}) {
				t.Errorf("replacement state lost after old retirement: running=%v loaded=%v intent=%v", newRunning, newLoaded, newIntent)
			}
		})
	}
	test("retirement before replacement", func(b *Broker, old *engineManagerSource) *engineManagerSource {
		b.retireEngineManagerSource(old)
		return nil
	})
	test("activation replaces a live source", func(b *Broker, _ *engineManagerSource) *engineManagerSource {
		return b.activateEngineManagerSource()
	})
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
