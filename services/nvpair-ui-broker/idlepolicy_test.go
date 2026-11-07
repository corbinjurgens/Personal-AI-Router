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

func TestIdlePolicyUnloadsOnlyIdleNonresidentModels(t *testing.T) {
	b, _ := newPolicyBroker(t)
	worker := attachFakeEngine(t, b)
	now := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.now = func() time.Time { return now }
	r.current.Idle.UnloadAfterMinutes = 5
	r.current.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "resident"}}
	r.mu.Unlock()
	b.observeLoaded(map[string][]string{"llamacpp": {"idle", "active", "resident"}})
	now = now.Add(6 * time.Minute)
	b.observeAdmission(nodepolicy.AdmissionState{
		Active: 1, ActiveByEngine: map[string]int{"llamacpp": 1},
		ActiveByModel:  map[string]map[string]int{"llamacpp": {"active": 1}},
		LastActivityMs: map[string]map[string]int64{"llamacpp": {"active": now.UnixMilli()}},
	})
	b.idleTick(context.Background())
	calls := worker.snapshot()
	if len(calls) != 1 || calls[0].method != engineUnloadModelMethod {
		t.Fatalf("idle actions = %+v, want one model unload", calls)
	}
	var params struct {
		Engine string `json:"engine"`
		Model  string `json:"model"`
	}
	if err := json.Unmarshal(calls[0].params, &params); err != nil {
		t.Fatal(err)
	}
	if params.Engine != "llamacpp" || params.Model != "idle" {
		t.Fatalf("unload = %+v, want idle model", params)
	}
	b.idleTick(context.Background())
	if len(worker.snapshot()) != 1 {
		t.Fatal("idle unload repeated for unchanged baseline")
	}
	if got := b.runningSnapshot(); !reflect.DeepEqual(got, []string{"llamacpp"}) {
		t.Fatalf("idle changed engine state: %v", got)
	}
}

func TestIdleResidencyArrivalIsPerModel(t *testing.T) {
	b, _ := newPolicyBroker(t)
	worker := attachFakeEngine(t, b)
	now := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.now = func() time.Time { return now }
	r.current.Idle.UnloadAfterMinutes = 5
	r.mu.Unlock()
	b.observeLoaded(map[string][]string{"llamacpp": {"old"}})
	now = now.Add(4 * time.Minute)
	b.observeLoaded(map[string][]string{"llamacpp": {"old", "new"}})
	now = now.Add(2 * time.Minute)
	b.idleTick(context.Background())
	calls := worker.snapshot()
	if len(calls) != 1 || calls[0].method != engineUnloadModelMethod {
		t.Fatalf("idle actions = %+v", calls)
	}
	var params map[string]string
	if err := json.Unmarshal(calls[0].params, &params); err != nil {
		t.Fatal(err)
	}
	if params["model"] != "old" {
		t.Fatalf("unloaded %q, want old model", params["model"])
	}
}

func TestResidentProtectionAutomaticUnload(t *testing.T) {
	for _, reason := range []string{"idle", "admission"} {
		t.Run(reason, func(t *testing.T) {
			b, _ := newPolicyBroker(t)
			worker := attachFakeEngine(t, b)
			r := b.nodePolicyRuntime()
			r.mu.Lock()
			r.current.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "resident"}}
			r.runningSince["llamacpp"] = r.now().Add(-10 * time.Minute)
			r.mu.Unlock()
			b.unloadModels("llamacpp", []string{"resident", "neighbor"}, reason)
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
			r.current.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "later"}}
			r.mu.Unlock()
		}
		return map[string]bool{"ok": true}, ""
	})
	b.unloadModels("llamacpp", []string{"first", "later"}, "admission")
	if calls := worker.snapshot(); len(calls) != 1 {
		t.Fatalf("queued unload ignored new resident preference: %+v", calls)
	}
}

func TestResidentProtectionPauseMayUnload(t *testing.T) {
	b, _ := newPolicyBroker(t)
	worker := attachFakeEngine(t, b)
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "resident"}}
	r.mu.Unlock()
	b.unloadModels("llamacpp", []string{"resident"}, "pause")
	if got := worker.methods(""); !reflect.DeepEqual(got, []string{engineUnloadModelMethod}) {
		t.Fatalf("pause calls = %v, want resident unload", got)
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
			r.current.Idle.UnloadAfterMinutes = 1
			r.mu.Unlock()
			old := b.activateEngineManagerSource()
			b.observeEngineRunningFrom(old, "llamacpp", true)
			b.observeLoadedFrom(old, map[string][]string{"llamacpp": {"model"}})
			r.mu.Lock()
			r.runningSince["llamacpp"] = now.Add(-2 * time.Minute)
			r.residencyAt[nodepolicy.ModelRef{Engine: "llamacpp", Model: "model"}] = now.Add(-2 * time.Minute)
			r.idleUnloadedAt[nodepolicy.ModelRef{Engine: "llamacpp", Model: "model"}] = now.Add(-2 * time.Minute)
			r.mu.Unlock()

			current := transition(b, old)
			// A late callback from the retired source must not restore its state.
			b.observeEngineRunningFrom(old, "llamacpp", true)
			b.observeLoadedFrom(old, map[string][]string{"llamacpp": {"model"}})
			ctx, cancel := context.WithCancel(context.Background())
			cancel() // Allow planning but prevent any worker RPC.
			b.idleTick(ctx)
			r.mu.Lock()
			_, running := r.running["llamacpp"]
			_, loaded := r.loaded["llamacpp"]
			_, observed := r.runningObserved["llamacpp"]
			_, runningSince := r.runningSince["llamacpp"]
			_, residencyAt := r.residencyAt[nodepolicy.ModelRef{Engine: "llamacpp", Model: "model"}]
			_, unloadedAt := r.idleUnloadedAt[nodepolicy.ModelRef{Engine: "llamacpp", Model: "model"}]
			r.mu.Unlock()
			if running || loaded {
				t.Errorf("retired source state remains: running=%v loaded=%v", running, loaded)
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
			b.retireEngineManagerSource(old)
			r.mu.Lock()
			newRunning := r.running["llamacpp"]
			newLoaded := append([]string(nil), r.loaded["llamacpp"]...)
			r.mu.Unlock()
			if !newRunning || !reflect.DeepEqual(newLoaded, []string{"new-model"}) {
				t.Errorf("replacement state lost after old retirement: running=%v loaded=%v", newRunning, newLoaded)
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
