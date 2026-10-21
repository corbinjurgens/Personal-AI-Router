// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"reflect"
	"sync"
	"testing"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

type modelConfigFixture struct {
	*strictFixture
	source *engineManagerSource
}

func newModelConfigFixture(t *testing.T) *modelConfigFixture {
	t.Helper()
	f := &modelConfigFixture{strictFixture: newStrictFixture(t)}
	b := f.b
	f.source = b.activateEngineManagerSource()
	b.workersMu.Lock()
	worker := b.engineMgr
	b.workersMu.Unlock()
	b.bindEngineMgr(worker, f.source)
	seedStrictIdentity(b, "llamacpp", "resident")
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Profiles = []nodepolicy.Profile{{Name: "resident", Engine: "llamacpp", Model: "resident", ContextTokens: 8192, MaxOutputTokens: 1024, MaxConcurrent: 1, MemoryBytes: 1024,
		Qualification: &nodepolicy.QualificationRef{ID: "fixture-only", Revision: "1"}}}
	r.current.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: "resident"}}
	r.current.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "resident"}}
	r.intent = map[string]bool{"llamacpp": true}
	r.running = map[string]bool{"llamacpp": true}
	r.loaded = map[string][]string{"llamacpp": {}}
	r.models.loadedSource = f.source
	r.models.loadedProxy = f.pp
	id := r.engineIdentity["llamacpp"]
	ram, vram := int64(1024), int64(0)
	id.Serving = map[string]engines.ServingEnvelope{"resident": {
		ModelID: "resident", ContentDigest: id.ModelDigests["resident"], NativeSourceDigest: id.LaunchDigest,
		PerSlotContextTokens: 8192, TotalContextTokens: 8192, Slots: 1, MaxOutputTokens: 1024,
		Memory: engines.ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram}, Retention: "-1", RouterCapacity: 2,
	}}
	r.engineIdentity["llamacpp"] = id
	r.mu.Unlock()
	if !b.replayPolicyToProxy(context.Background(), f.pp) {
		t.Fatal("activate resident fixture policy")
	}
	reportStrict(t, b, "0", f.acks.runID, strictCapability(t, "llamacpp", "resident"))
	f.proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
		switch method {
		case nodepolicy.MethodPrepareResident:
			return nodepolicy.PrepareResidentResult{ReservationID: "fixture-reservation", EvidenceGeneration: 1}, ""
		case nodepolicy.MethodDispatchResident:
			return nodepolicy.DispatchResidentResult{Dispatched: true}, ""
		case nodepolicy.MethodFinishResident:
			var p nodepolicy.FinishResidentParams
			if err := json.Unmarshal(params, &p); err != nil {
				return nil, err.Error()
			}
			return nodepolicy.FinishResidentResult{ReservationID: p.ReservationID, Outcome: p.Outcome}, ""
		}
		return f.acks.answer(method, params)
	})
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			return emptySwitchManagerSnapshot(), ""
		}
		if method == nodepolicy.MethodReconcileResident {
			return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
		}
		return map[string]bool{"ok": true}, ""
	})
	e := b.captureModelConfig()
	entry, _, _, _ := b.modelConfigEntry(e, nodepolicy.Role{Engine: "llamacpp", Model: "resident"})
	if entry.State != nodepolicy.ModelStatePending {
		t.Fatalf("resident fixture is not eligible: %+v", entry)
	}
	return f
}

func modelConfigCalls(f *fakeWorker, method string) []policyCall {
	var calls []policyCall
	for _, call := range f.snapshot() {
		if call.method == method {
			calls = append(calls, call)
		}
	}
	return calls
}

func decodeModelConfigCall[T any](t *testing.T, call policyCall) T {
	t.Helper()
	var params T
	if err := json.Unmarshal(call.params, &params); err != nil {
		t.Fatalf("decode %s: %v", call.method, err)
	}
	return params
}

func TestModelConfigLoadUsesRevisionProfile(t *testing.T) {
	test := func(t *testing.T, contextTokens int) {
		f := newModelConfigFixture(t)
		// A valid nonallocation multiple detects rounding, clamping and defaults.
		r := f.b.nodePolicyRuntime()
		r.mu.Lock()
		r.current.Profiles[0].ContextTokens = contextTokens
		id := r.engineIdentity["llamacpp"]
		envelope := id.Serving["resident"]
		envelope.PerSlotContextTokens, envelope.TotalContextTokens = int64(contextTokens), int64(contextTokens)
		id.Serving["resident"] = envelope
		r.engineIdentity["llamacpp"] = id
		r.mu.Unlock()
		f.b.reconcileModelConfig(context.Background())
		calls := modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)
		if len(calls) != 1 {
			t.Fatalf("manager reconcile calls=%d, want one", len(calls))
		}
		p := decodeModelConfigCall[nodepolicy.ReconcileResidentParams](t, calls[0])
		if p.Load.ContextLength != contextTokens || p.Load.KeepAliveSeconds != 60 || p.DeclaredResidentCount != 1 || p.PolicyRevision != "0" || p.Reservation.ManagerIncarnation != "fixture-manager" || p.Identity.EngineRunID != "engine-run-1" {
			t.Fatalf("reconcile lost exact revision/profile/count contract: %+v", p)
		}
		prepared := modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)
		finished := modelConfigCalls(f.proxy, nodepolicy.MethodFinishResident)
		if len(prepared) != 1 || len(finished) != 1 {
			t.Fatalf("prepare=%d finish=%d, want one each", len(prepared), len(finished))
		}
		prepare := decodeModelConfigCall[nodepolicy.PrepareResidentParams](t, prepared[0])
		finish := decodeModelConfigCall[nodepolicy.FinishResidentParams](t, finished[0])
		if prepare.PolicyRevision != p.PolicyRevision || prepare.Identity != p.Identity || prepare.OperationID != p.OperationID || finish.OperationID != p.OperationID || finish.ReservationID != p.Reservation.ReservationID {
			t.Fatal("prepare, reconcile and finish do not share the captured operation")
		}
	}
	t.Run("small nonaligned context", func(t *testing.T) { test(t, 4097) })
	t.Run("review nonaligned context", func(t *testing.T) { test(t, 8193) })
	t.Run("large nonaligned context", func(t *testing.T) { test(t, 32769) })
}

func TestModelConfigConfirmingInventoryDoesNotReload(t *testing.T) {
	f := newModelConfigFixture(t)
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			return emptySwitchManagerSnapshot(), ""
		}
		return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
	})
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	for range 3 {
		f.b.reconcileModelConfig(context.Background())
	}
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 1 {
		t.Fatalf("confirming inventory issued %d loads, want one completed load", got)
	}
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].State != nodepolicy.ModelStateSatisfied {
		t.Fatalf("confirming inventory lost satisfaction: %+v", status)
	}
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {}})
	f.b.reconcileModelConfig(context.Background())
	f.b.reconcileModelConfig(context.Background())
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 2 {
		t.Fatalf("actual unload issued %d loads, want original and one retry", got)
	}
}

func TestModelConfigInflightConfirmingInventoryCoalesces(t *testing.T) {
	f := newModelConfigFixture(t)
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	var wg sync.WaitGroup
	for range 12 {
		wg.Add(1)
		go func() { defer wg.Done(); f.b.reconcileModelConfig(context.Background()) }()
	}
	wg.Wait()
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			return emptySwitchManagerSnapshot(), ""
		}
		return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
	})
	f.b.reconcileModelConfig(context.Background())
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 1 {
		t.Fatalf("inflight confirmation issued %d loads, want one", got)
	}
}

func residentLaunchReadback(e modelConfigEvidence) settings.LaunchState {
	envelope := e.Identity.Engines["llamacpp"].Serving["resident"]
	return settings.LaunchState{Engine: "llamacpp", Running: true, ServingReadback: &engines.ServingReadback{
		Status: "observed", LaunchValidated: []engines.ServingEnvelope{envelope},
		Observed: []engines.ServingObservation{{ModelID: "resident", PerSlotContextTokens: envelope.PerSlotContextTokens, Slots: envelope.Slots}},
	}}
}

func TestModelConfigWorkerRestartConfirmsWithoutReload(t *testing.T) {
	for _, worker := range []string{"manager", "proxy"} {
		t.Run(worker, func(t *testing.T) {
			f := newModelConfigFixture(t)
			done := f.startLoadedAttempt(t)
			f.awaitGrant(t)
			f.engine.releaseHeld()
			awaitModelConfigDone(t, done)
			before := f.b.captureModelConfig()
			manager := f.engine
			if worker == "manager" {
				manager = attachFakeEngine(t, f.b)
				f.source = f.b.activateEngineManagerSource()
				f.b.bindEngineMgr(f.b.getEngineMgr(), f.source)
				r := f.b.nodePolicyRuntime()
				r.mu.Lock()
				id := before.Identity.Engines["llamacpp"]
				r.engineIdentity = map[string]engineManagerIdentity{"llamacpp": {EngineRunID: id.EngineRunID, LaunchDigest: id.LaunchDigest, ModelDigests: id.ModelDigests, Serving: id.Serving}}
				r.intent["llamacpp"], r.running["llamacpp"] = true, true
				r.mu.Unlock()
			} else {
				pp, _, acks := attachAckingProxy(t, f.b)
				if !f.b.replayPolicyToProxy(context.Background(), pp) {
					t.Fatal("activate replacement proxy")
				}
				reportStrict(t, f.b, "0", acks.runID, strictCapability(t, "llamacpp", "resident"))
			}
			readback := residentLaunchReadback(before)
			manager.setAnswer(func(method string, _ json.RawMessage) (any, string) {
				if method == "engine:get-launch" {
					return readback, ""
				}
				if method == settings.SwitchPrepareMethod {
					return emptySwitchManagerSnapshot(), ""
				}
				return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
			})
			f.b.reconcileModelConfig(context.Background())
			status := f.b.modelConfiguration(f.b.captureModelConfig())
			if status.Models[0].State == nodepolicy.ModelStateSatisfied {
				t.Fatal("replacement inherited completion before fresh inventory")
			}
			f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
			for range 3 {
				f.b.reconcileModelConfig(context.Background())
			}
			status = f.b.modelConfiguration(f.b.captureModelConfig())
			if status.Models[0].State != nodepolicy.ModelStateSatisfied || len(modelConfigCalls(manager, "engine:get-launch")) != 1 {
				t.Fatalf("fresh observed envelope was not confirmed once: %+v", status)
			}
			wantLoads := 1
			if worker == "manager" {
				wantLoads = 0
			}
			if got := len(modelConfigCalls(manager, nodepolicy.MethodReconcileResident)); got != wantLoads {
				t.Fatalf("replacement issued duplicate resident load: got %d want %d", got, wantLoads)
			}
		})
	}
}

func TestModelConfigLoadedConfirmationRequiresObservedEnvelope(t *testing.T) {
	for _, invalid := range []string{"cold only", "wrong context", "wrong slots", "wrong envelope", "local Off", "local pause"} {
		t.Run(invalid, func(t *testing.T) {
			f := newModelConfigFixture(t)
			readback := residentLaunchReadback(f.b.captureModelConfig())
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			switch invalid {
			case "cold only":
				readback.ServingReadback.Observed = nil
			case "wrong context":
				readback.ServingReadback.Observed[0].PerSlotContextTokens++
			case "wrong slots":
				readback.ServingReadback.Observed[0].Slots++
			case "wrong envelope":
				readback.ServingReadback.LaunchValidated[0].RouterCapacity++
			case "local Off":
				r.intent["llamacpp"] = false
			case "local pause":
				r.localPauseRequested = true
			}
			r.mu.Unlock()
			f.engine.setAnswer(func(string, json.RawMessage) (any, string) { return readback, "" })
			f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
			f.b.reconcileModelConfig(context.Background())
			status := f.b.modelConfiguration(f.b.captureModelConfig())
			if status.Models[0].State == nodepolicy.ModelStateSatisfied || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 {
				t.Fatalf("invalid observation authorized satisfaction or load: %+v", status)
			}
		})
	}
}

func TestModelConfigConfirmingInventoryCannotSettleUncertainHold(t *testing.T) {
	f := newModelConfigFixture(t)
	f.proxy.mu.Lock()
	answer := f.proxy.answer
	f.proxy.mu.Unlock()
	f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodFinishResident {
			return nodepolicy.FinishResidentResult{ReservationID: "fixture-reservation", Outcome: nodepolicy.ResidentUnknown, RecoveryID: "fixture-recovery"}, ""
		}
		return answer(method, raw)
	})
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	readback := residentLaunchReadback(f.b.captureModelConfig())
	f.engine.setAnswer(func(string, json.RawMessage) (any, string) { return readback, "" })
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.b.reconcileModelConfig(context.Background())
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].State != nodepolicy.ModelStateUnknown || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 || len(modelConfigCalls(f.engine, "engine:get-launch")) != 0 {
		t.Fatalf("ordinary inventory retried or settled uncertain hold: %+v", status)
	}
}

func TestModelConfigReplacedRevisionReconcilesOnce(t *testing.T) {
	f := newModelConfigFixture(t)
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Revision = "1"
	r.current.Profiles[0].ContextTokens = 8193
	id := r.engineIdentity["llamacpp"]
	envelope := id.Serving["resident"]
	envelope.PerSlotContextTokens, envelope.TotalContextTokens = 8193, 8193
	id.Serving["resident"] = envelope
	r.engineIdentity["llamacpp"] = id
	r.mu.Unlock()
	if !f.b.replayPolicyToProxy(context.Background(), f.pp) {
		t.Fatal("activate replacement revision")
	}
	reportStrict(t, f.b, "1", f.acks.runID, strictCapability(t, "llamacpp", "resident"))
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].State == nodepolicy.ModelStateSatisfied {
		t.Fatal("replacement revision inherited old completion")
	}
	readback := residentLaunchReadback(f.b.captureModelConfig())
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			return emptySwitchManagerSnapshot(), ""
		}
		if method == "engine:get-launch" {
			return readback, ""
		}
		return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
	})
	f.b.reconcileModelConfig(context.Background())
	f.b.reconcileModelConfig(context.Background())
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.b.reconcileModelConfig(context.Background())
	status = f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].State != nodepolicy.ModelStateSatisfied || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 2 {
		t.Fatalf("replacement revision did not reconcile exactly once: %+v", status)
	}
	call := modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)[1]
	p := decodeModelConfigCall[nodepolicy.ReconcileResidentParams](t, call)
	if p.PolicyRevision != "1" || p.Load.ContextLength != 8193 {
		t.Fatalf("replacement load lost new revision/envelope: %+v", p)
	}
}

func TestModelConfigResidentSetNormalizedPerEngine(t *testing.T) {
	f := newModelConfigFixture(t)
	p := f.b.captureModelConfig().Policy
	p.ResidentModels = nodepolicy.ResidentModels{
		{Engine: "ollama", Model: "m"}, {Engine: "llamacpp", Model: "resident"},
		{Engine: "ollama", Model: "m:latest"}, {Engine: "ollama", Model: "other"},
	}
	e := f.b.captureModelConfig()
	e.Policy = p
	_, _, _, count := f.b.modelConfigEntry(e, nodepolicy.Role{Engine: "ollama", Model: "m:latest"})
	if count != 2 {
		t.Fatalf("normalized ollama resident count=%d, want 2", count)
	}
	_, _, _, count = f.b.modelConfigEntry(e, nodepolicy.Role{Engine: "llamacpp", Model: "resident"})
	if count != 1 {
		t.Fatalf("llamacpp count=%d, want 1", count)
	}
	want := []nodepolicy.Role{{Engine: "llamacpp", Model: "resident"}, {Engine: "ollama", Model: "m:latest"}, {Engine: "ollama", Model: "other:latest"}}
	if got := f.b.residentRefs(p); !reflect.DeepEqual(got, want) {
		t.Fatalf("normalized deterministic resident set=%v, want %v", got, want)
	}
}

func TestModelConfigFailedAttemptWaitsForChangedEvidence(t *testing.T) {
	f := newModelConfigFixture(t)
	for range 5 {
		f.b.reconcileModelConfig(context.Background())
		f.b.signalModelConfig()
	}
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 1 {
		t.Fatalf("unchanged evidence retried failed attempt %d times", got)
	}
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.loaded["llamacpp"] = []string{"new-observation"}
	r.mu.Unlock()
	f.b.reconcileModelConfig(context.Background())
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 2 {
		t.Fatalf("changed evidence did not allow one retry: calls=%d", got)
	}
}

func TestModelConfigAdmissionRefusalsHaveNoLoad(t *testing.T) {
	cases := []struct {
		name   string
		change func(*policyRuntime)
	}{
		{"missing activation ack", func(r *policyRuntime) { r.activeRevision = "old" }},
		{"local Off", func(r *policyRuntime) { r.intent["llamacpp"] = false }},
		{"local pause", func(r *policyRuntime) { r.localPauseRequested = true }},
		{"unknown loaded inventory", func(r *policyRuntime) { delete(r.loaded, "llamacpp") }},
		{"missing qualification", func(r *policyRuntime) { r.strictReadiness.Capability = nil }},
		{"missing unique profile", func(r *policyRuntime) { r.current.Profiles = nil }},
		{"engine drain", func(r *policyRuntime) { r.drains = map[string]bool{"llamacpp": true} }},
		{"context exceeds native envelope", func(r *policyRuntime) { r.current.Profiles[0].ContextTokens++ }},
		{"unsupported native retention", func(r *policyRuntime) {
			id := r.engineIdentity["llamacpp"]
			envelope := id.Serving["resident"]
			envelope.Retention = "300"
			id.Serving["resident"] = envelope
			r.engineIdentity["llamacpp"] = id
		}},
		{"native router capacity absent", func(r *policyRuntime) {
			id := r.engineIdentity["llamacpp"]
			envelope := id.Serving["resident"]
			envelope.RouterCapacity = 0
			id.Serving["resident"] = envelope
			r.engineIdentity["llamacpp"] = id
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newModelConfigFixture(t)
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			tc.change(r)
			r.mu.Unlock()
			f.b.reconcileModelConfig(context.Background())
			if len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 || len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)) != 0 {
				t.Fatal("refused evidence reached resident preparation or manager load")
			}
		})
	}
}

func TestModelConfigConcurrentTriggersCoalesce(t *testing.T) {
	f := newModelConfigFixture(t)
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			return emptySwitchManagerSnapshot(), ""
		}
		if method == nodepolicy.MethodReconcileResident {
			return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, holdSuccessReply
		}
		return map[string]bool{"ok": true}, ""
	})
	done := make(chan struct{})
	go func() { defer close(done); f.b.reconcileModelConfig(context.Background()) }()
	f.engine.waitFor("held resident request", func(calls []policyCall) bool {
		return len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) == 1
	})
	var wg sync.WaitGroup
	for range 12 {
		wg.Add(1)
		go func() { defer wg.Done(); f.b.reconcileModelConfig(context.Background()) }()
	}
	wg.Wait()
	f.engine.releaseHeld()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("resident operation did not finish")
	}
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 1 {
		t.Fatalf("concurrent triggers started %d manager operations", got)
	}
}

func TestModelConfigStatusUnknownEmptyAndRemoval(t *testing.T) {
	f := newModelConfigFixture(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	delete(r.loaded, "llamacpp")
	r.mu.Unlock()
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if len(status.Models) != 1 || status.Models[0].ObservedLoaded != nil || status.Models[0].State != nodepolicy.ModelStateUnknown {
		t.Fatalf("unknown inventory status=%+v", status)
	}
	r.mu.Lock()
	r.loaded["llamacpp"] = []string{}
	r.mu.Unlock()
	status = f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].ObservedLoaded == nil || *status.Models[0].ObservedLoaded || status.Models[0].State == nodepolicy.ModelStateSatisfied {
		t.Fatalf("known empty inventory status=%+v", status)
	}
	r.mu.Lock()
	r.current.ResidentModels = nil
	r.mu.Unlock()
	f.b.reconcileModelConfig(context.Background())
	status = f.b.modelConfiguration(f.b.captureModelConfig())
	if len(status.Models) != 0 || status.Models == nil {
		t.Fatalf("removed resident status=%+v", status)
	}
	for _, call := range f.engine.snapshot() {
		if call.method == "engine:unload-model" || call.method == "engine:stop" || call.method == nodepolicy.MethodReconcileResident {
			t.Fatalf("removing preference sent %s", call.method)
		}
	}
}

func TestModelConfigInternalClientMethodsDenied(t *testing.T) {
	for _, method := range []string{nodepolicy.MethodReconcileResident, residentDispatchRequestMethod, residentDispatchReplyMethod} {
		t.Run(method, func(t *testing.T) {
			b, out := newPolicyBroker(t)
			engine := attachFakeEngine(t, b)
			b.handleMessage(clientRequest(1, method, map[string]string{"id": "forged"}))
			response := out.waitFrame(t, method, isResponse(1))
			if response.Error == nil || response.Error.Code != -32601 || len(engine.snapshot()) != 0 {
				t.Fatal("internal resident method reached a worker from public client")
			}
		})
	}
}

func (f *modelConfigFixture) startLoadedAttempt(t *testing.T) <-chan struct{} {
	t.Helper()
	f.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			return emptySwitchManagerSnapshot(), ""
		}
		if method != nodepolicy.MethodReconcileResident {
			return map[string]bool{"ok": true}, ""
		}
		var p nodepolicy.ReconcileResidentParams
		if err := json.Unmarshal(raw, &p); err != nil {
			return nil, err.Error()
		}
		request, err := json.Marshal(struct {
			ID     string                            `json:"id"`
			Params nodepolicy.DispatchResidentParams `json:"params"`
		}{"fixture-dispatch", nodepolicy.DispatchResidentParams{
			ReservationID: p.Reservation.ReservationID, ProxyRunID: p.Reservation.ProxyRunID,
			ManagerIncarnation: p.Reservation.ManagerIncarnation, PolicyRevision: p.PolicyRevision,
			OperationID: p.OperationID, Engine: p.Engine, Model: p.Model, Identity: p.Identity,
		}})
		if err != nil {
			return nil, err.Error()
		}
		f.b.dispatchResidentFrom(f.source, request)
		return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentLoaded,
			Observation: &nodepolicy.ResidentLoadedObservation{
				ManagerIncarnation: p.Reservation.ManagerIncarnation, Generation: 1,
				Engine: p.Engine, Model: p.Model, Identity: p.Identity, Loaded: true,
			}}, holdSuccessReply
	})
	done := make(chan struct{})
	go func() { defer close(done); f.b.reconcileModelConfig(context.Background()) }()
	return done
}

func awaitModelConfigDone(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("resident reconciliation did not complete")
	}
}

func (f *modelConfigFixture) awaitGrant(t *testing.T) {
	t.Helper()
	f.engine.waitFor("correlated resident grant reply", func(calls []policyCall) bool {
		for _, call := range calls {
			if call.method == residentDispatchReplyMethod {
				return true
			}
		}
		return false
	})
	calls := modelConfigCalls(f.engine, residentDispatchReplyMethod)
	var reply struct {
		ID     string                            `json:"id"`
		Result nodepolicy.DispatchResidentResult `json:"result"`
	}
	if err := json.Unmarshal(calls[0].params, &reply); err != nil {
		t.Fatal(err)
	}
	if reply.ID != "fixture-dispatch" || !reply.Result.Dispatched {
		t.Fatalf("correlated grant reply=%+v", reply)
	}
}

func TestModelConfigGrantCompletesOnlyAfterFinish(t *testing.T) {
	f := newModelConfigFixture(t)
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].State == nodepolicy.ModelStateSatisfied {
		t.Fatal("grant alone published satisfied before terminal load and finish")
	}
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	status = f.b.modelConfiguration(f.b.captureModelConfig())
	if len(status.Models) != 1 || status.Models[0].State != nodepolicy.ModelStateSatisfied || status.Models[0].ObservedLoaded == nil || !*status.Models[0].ObservedLoaded {
		t.Fatalf("finished resident load status=%+v", status)
	}
	if got := len(modelConfigCalls(f.proxy, nodepolicy.MethodDispatchResident)); got != 1 {
		t.Fatalf("proxy final grants=%d, want one", got)
	}
}

func TestModelConfigManagerReplacementInvalidatesPreparedWork(t *testing.T) {
	f := newModelConfigFixture(t)
	f.proxy.mu.Lock()
	answer := f.proxy.answer
	f.proxy.mu.Unlock()
	f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, failure := answer(method, raw)
		if method == nodepolicy.MethodPrepareResident {
			return result, holdSuccessReply
		}
		return result, failure
	})
	done := make(chan struct{})
	go func() { defer close(done); f.b.reconcileModelConfig(context.Background()) }()
	f.proxy.waitFor("held resident preparation", func(calls []policyCall) bool {
		return len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)) == 1
	})
	replacement := attachFakeEngine(t, f.b)
	f.proxy.releaseHeld()
	awaitModelConfigDone(t, done)
	if len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 || len(modelConfigCalls(replacement, nodepolicy.MethodReconcileResident)) != 0 {
		t.Fatal("stale prepared work reached original or replacement manager")
	}
	if len(modelConfigCalls(f.proxy, nodepolicy.MethodFinishResident)) != 1 {
		t.Fatal("stale reservation was not finished on its original proxy")
	}
}

func TestModelConfigManagerReplacementDuringLoadOrFinish(t *testing.T) {
	for _, phase := range []string{"load", "finish"} {
		t.Run(phase, func(t *testing.T) {
			f := newModelConfigFixture(t)
			if phase == "finish" {
				f.proxy.mu.Lock()
				answer := f.proxy.answer
				f.proxy.mu.Unlock()
				f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
					result, failure := answer(method, raw)
					if method == nodepolicy.MethodFinishResident {
						return result, holdSuccessReply
					}
					return result, failure
				})
			}
			done := f.startLoadedAttempt(t)
			f.awaitGrant(t)
			if phase == "finish" {
				f.engine.releaseHeld()
				f.proxy.waitFor("held resident finish", func(calls []policyCall) bool {
					return len(modelConfigCalls(f.proxy, nodepolicy.MethodFinishResident)) == 1
				})
			}
			replacement := attachFakeEngine(t, f.b)
			f.engine.releaseHeld()
			f.proxy.releaseHeld()
			awaitModelConfigDone(t, done)
			status := f.b.modelConfiguration(f.b.captureModelConfig())
			if status.Models[0].State == nodepolicy.ModelStateSatisfied || len(modelConfigCalls(replacement, nodepolicy.MethodReconcileResident)) != 0 {
				t.Fatal("replacement inherited old load completion or resident work")
			}
		})
	}
}

func TestModelConfigManagerReplacementDuringGrant(t *testing.T) {
	f := newModelConfigFixture(t)
	f.proxy.mu.Lock()
	answer := f.proxy.answer
	f.proxy.mu.Unlock()
	f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, failure := answer(method, raw)
		if method == nodepolicy.MethodDispatchResident {
			return result, holdSuccessReply
		}
		return result, failure
	})
	done := f.startLoadedAttempt(t)
	f.proxy.waitFor("held final grant", func(calls []policyCall) bool {
		return len(modelConfigCalls(f.proxy, nodepolicy.MethodDispatchResident)) == 1
	})
	replacement := attachFakeEngine(t, f.b)
	f.proxy.releaseHeld()
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	if len(modelConfigCalls(replacement, residentDispatchReplyMethod)) != 0 || len(modelConfigCalls(replacement, nodepolicy.MethodReconcileResident)) != 0 {
		t.Fatal("old grant or load was delivered to replacement manager")
	}
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].State == nodepolicy.ModelStateSatisfied {
		t.Fatal("late grant satisfied replacement source")
	}
}

func TestModelConfigDispatchRejectsForeignAndDuplicateRequests(t *testing.T) {
	f := newModelConfigFixture(t)
	e := f.b.captureModelConfig()
	_, _, id, _ := f.b.modelConfigEntry(e, nodepolicy.Role{Engine: "llamacpp", Model: "resident"})
	op := &modelConfigAttempt{ctx: context.Background(), evidence: e, reservation: "fixture-reservation", deadline: time.Now().Add(time.Second),
		prepare: nodepolicy.PrepareResidentParams{PolicyRevision: e.Policy.Revision, ProxyRunID: e.Run,
			ManagerIncarnation: "fixture-manager", OperationID: "resident-1", Engine: "llamacpp", Model: "resident", Identity: id}}
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.models.active = op
	r.mu.Unlock()
	p := nodepolicy.DispatchResidentParams{PolicyRevision: e.Policy.Revision, ProxyRunID: e.Run, ManagerIncarnation: "fixture-manager",
		OperationID: "resident-1", Engine: "llamacpp", Model: "resident", Identity: id, ReservationID: "fixture-reservation"}
	encode := func(p nodepolicy.DispatchResidentParams) json.RawMessage {
		t.Helper()
		raw, err := json.Marshal(struct {
			ID     string                            `json:"id"`
			Params nodepolicy.DispatchResidentParams `json:"params"`
		}{"fixture-dispatch", p})
		if err != nil {
			t.Fatal(err)
		}
		return raw
	}
	foreign := p
	foreign.PolicyRevision = "999"
	f.b.dispatchResidentFrom(f.source, encode(foreign))
	foreign = p
	foreign.Identity.EngineRunID = "foreign-run"
	f.b.dispatchResidentFrom(f.source, encode(foreign))
	f.b.dispatchResidentFrom(&engineManagerSource{}, encode(p))
	f.b.dispatchResidentFrom(f.source, json.RawMessage(`{"id":"bad","params":null}`))
	r.mu.Lock()
	foreignStarted := op.grantStarted
	r.mu.Unlock()
	if foreignStarted {
		t.Fatal("foreign or malformed dispatch request claimed the active operation")
	}
	if len(modelConfigCalls(f.proxy, nodepolicy.MethodDispatchResident)) != 0 {
		t.Fatal("foreign or malformed grant request reached proxy")
	}
	f.b.dispatchResidentFrom(f.source, encode(p))
	f.b.dispatchResidentFrom(f.source, encode(p))
	f.awaitGrant(t)
	if got := len(modelConfigCalls(f.proxy, nodepolicy.MethodDispatchResident)); got != 1 {
		t.Fatalf("duplicate request produced %d final grants", got)
	}
}

func TestModelConfigProxyRefusalDoesNotReachManager(t *testing.T) {
	f := newModelConfigFixture(t)
	f.proxy.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == nodepolicy.MethodPrepareResident {
			return nodepolicy.PrepareResidentResult{EvidenceGeneration: 7, Refusals: []nodepolicy.ResidentRefusal{{Engine: "llamacpp", Model: "resident", Reason: nodepolicy.ModelReasonNoFit}}}, ""
		}
		return map[string]bool{"ok": true}, ""
	})
	f.b.reconcileModelConfig(context.Background())
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 || status.Models[0].State != nodepolicy.ModelStateBlocked || status.Models[0].Reason != nodepolicy.ModelReasonNoFit {
		t.Fatalf("proxy capacity refusal lost bounded status or reached manager: %+v", status)
	}
}

func TestModelConfigRecoveryRequiresTerminalManagerBarrier(t *testing.T) {
	for _, terminal := range []bool{false, true} {
		name := "transport failure retains uncertain hold"
		if terminal {
			name = "terminal response permits fresh empty observation"
		}
		t.Run(name, func(t *testing.T) {
			f := newModelConfigFixture(t)
			f.proxy.mu.Lock()
			answer := f.proxy.answer
			f.proxy.mu.Unlock()
			f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == nodepolicy.MethodFinishResident {
					return nodepolicy.FinishResidentResult{ReservationID: "fixture-reservation", Outcome: nodepolicy.ResidentUnknown, RecoveryID: "fixture-recovery"}, ""
				}
				if method == nodepolicy.MethodObserveResident {
					return nodepolicy.ObserveResidentResult{ReservationID: "fixture-reservation", Settled: true}, ""
				}
				return answer(method, raw)
			})
			f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
				switch method {
				case settings.SwitchPrepareMethod:
					return emptySwitchManagerSnapshot(), ""
				case nodepolicy.MethodReconcileResident:
					if !terminal {
						return nil, "fixture lost terminal response"
					}
					return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentUnknown}, ""
				case "engine:models":
					return map[string]any{"loadedByEngine": map[string][]string{"llamacpp": {}}}, ""
				}
				return map[string]bool{"ok": true}, ""
			})
			f.b.reconcileModelConfig(context.Background())
			observations := modelConfigCalls(f.proxy, nodepolicy.MethodObserveResident)
			if !terminal {
				if len(observations) != 0 || len(modelConfigCalls(f.engine, "engine:models")) != 0 {
					t.Fatal("missing terminal barrier performed recovery observation")
				}
				return
			}
			if len(observations) != 1 || len(modelConfigCalls(f.engine, "engine:models")) != 1 {
				t.Fatal("terminal manager result did not require one fresh observation")
			}
			p := decodeModelConfigCall[nodepolicy.ObserveResidentParams](t, observations[0])
			if !p.OperationComplete || p.Observation.Loaded || p.Sequence != 1 || p.Observation.ManagerIncarnation != "fixture-manager" || p.RecoveryID != "fixture-recovery" {
				t.Fatalf("recovery did not bind completion and fresh empty evidence: %+v", p)
			}
		})
	}
}

func TestModelConfigAuthorityChangesDuringPreparePreventLoad(t *testing.T) {
	cases := []struct {
		name   string
		change func(*modelConfigFixture)
	}{
		{"proxy replacement", func(f *modelConfigFixture) {
			attachAckingProxy(t, f.b)
		}},
		{"local pause", func(f *modelConfigFixture) {
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			r.localPauseRequested = true
			r.mu.Unlock()
		}},
		{"local Off", func(f *modelConfigFixture) {
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			r.intent["llamacpp"] = false
			r.mu.Unlock()
		}},
		{"profile revision changes", func(f *modelConfigFixture) {
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			r.current.Revision = "1"
			r.current.Profiles[0].ContextTokens = 4096
			r.mu.Unlock()
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newModelConfigFixture(t)
			f.proxy.mu.Lock()
			answer := f.proxy.answer
			f.proxy.mu.Unlock()
			f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				result, failure := answer(method, raw)
				if method == nodepolicy.MethodPrepareResident {
					return result, holdSuccessReply
				}
				return result, failure
			})
			done := make(chan struct{})
			go func() { defer close(done); f.b.reconcileModelConfig(context.Background()) }()
			f.proxy.waitFor("held preparation", func(calls []policyCall) bool {
				return len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)) == 1
			})
			tc.change(f)
			f.proxy.releaseHeld()
			awaitModelConfigDone(t, done)
			if len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 {
				t.Fatal("authority changed during prepare but manager load was still requested")
			}
			if len(modelConfigCalls(f.proxy, nodepolicy.MethodFinishResident)) != 1 {
				t.Fatal("revoked preparation was not finished on original proxy")
			}
		})
	}
}

func TestModelConfigAuthorityChangesDuringGrantLoadAndFinish(t *testing.T) {
	for _, phase := range []string{"grant", "load", "finish"} {
		for _, change := range []string{"proxy replacement", "local pause", "local Off"} {
			t.Run(phase+"/"+change, func(t *testing.T) {
				f := newModelConfigFixture(t)
				f.proxy.mu.Lock()
				answer := f.proxy.answer
				f.proxy.mu.Unlock()
				heldMethod := ""
				if phase == "grant" {
					heldMethod = nodepolicy.MethodDispatchResident
				}
				if phase == "finish" {
					heldMethod = nodepolicy.MethodFinishResident
				}
				f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
					result, failure := answer(method, raw)
					if method == heldMethod {
						return result, holdSuccessReply
					}
					return result, failure
				})
				done := f.startLoadedAttempt(t)
				if phase != "grant" {
					f.awaitGrant(t)
				}
				if phase == "finish" {
					f.engine.releaseHeld()
				}
				if heldMethod != "" {
					f.proxy.waitFor("held "+phase, func(calls []policyCall) bool {
						return len(modelConfigCalls(f.proxy, heldMethod)) == 1
					})
				}
				var replacement *fakeWorker
				switch change {
				case "proxy replacement":
					_, replacement, _ = attachAckingProxy(t, f.b)
				case "local pause":
					r := f.b.nodePolicyRuntime()
					r.mu.Lock()
					r.localPauseRequested = true
					r.mu.Unlock()
				case "local Off":
					r := f.b.nodePolicyRuntime()
					r.mu.Lock()
					r.intent["llamacpp"] = false
					r.mu.Unlock()
				}
				f.proxy.releaseHeld()
				f.engine.releaseHeld()
				awaitModelConfigDone(t, done)
				status := f.b.modelConfiguration(f.b.captureModelConfig())
				if status.Models[0].State == nodepolicy.ModelStateSatisfied {
					t.Fatalf("%s during %s retained satisfied state", change, phase)
				}
				if replacement != nil && (len(modelConfigCalls(replacement, nodepolicy.MethodDispatchResident)) != 0 || len(modelConfigCalls(replacement, nodepolicy.MethodFinishResident)) != 0) {
					t.Fatal("replacement proxy inherited grant or finish for original reservation")
				}
			})
		}
	}
}

func TestModelConfigAbsolutePolicyDeadlineBoundsLoad(t *testing.T) {
	f := newModelConfigFixture(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Admission.SwitchTimeoutSeconds = 1
	r.mu.Unlock()
	preparedDeadline := make(chan time.Time, 1)
	f.proxy.mu.Lock()
	answer := f.proxy.answer
	f.proxy.mu.Unlock()
	f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodPrepareResident {
			r.mu.Lock()
			preparedDeadline <- r.models.active.deadline
			r.mu.Unlock()
		}
		return answer(method, raw)
	})
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	r.mu.Lock()
	op := r.models.active
	deadline := op.deadline
	r.mu.Unlock()
	if deadline.After(time.Now().Add(time.Second)) || !deadline.Equal(<-preparedDeadline) {
		t.Fatal("load extended preparation's absolute policy deadline")
	}
	params := decodeModelConfigCall[nodepolicy.ReconcileResidentParams](t, modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)[0])
	if params.TimeoutMs <= 0 || params.TimeoutMs > 1000 {
		t.Fatalf("manager timeout=%d, want remaining policy budget", params.TimeoutMs)
	}
	// The terminal load reply stays held. The original deadline must finish the
	// attempt without acquiring a new load budget or inferring loaded evidence.
	awaitModelConfigDone(t, done)
	if time.Now().Before(deadline) {
		t.Fatal("held load completed before its absolute deadline")
	}
	finish := decodeModelConfigCall[nodepolicy.FinishResidentParams](t, modelConfigCalls(f.proxy, nodepolicy.MethodFinishResident)[0])
	if finish.Outcome != nodepolicy.ResidentUnknown {
		t.Fatalf("expired load finished as %q, want unknown", finish.Outcome)
	}
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].State == nodepolicy.ModelStateSatisfied {
		t.Fatal("expired load published satisfaction")
	}
	f.engine.releaseHeld()
}

func TestModelConfigInflightStatusRetainsEveryDesiredReference(t *testing.T) {
	f := newModelConfigFixture(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	second := r.current.Profiles[0]
	second.Name, second.Model = "second", "second"
	r.current.Profiles = append(r.current.Profiles, second)
	r.current.ResidentModels = append(r.current.ResidentModels, nodepolicy.Role{Engine: "llamacpp", Model: "second"})
	r.current.ServedModels = append(r.current.ServedModels, nodepolicy.Role{Engine: "llamacpp", Model: "second"})
	r.mu.Unlock()
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if len(status.Models) != 2 || status.Models[0].Model != "resident" || status.Models[1].Model != "second" || !status.Models[0].DesiredResident || !status.Models[1].DesiredResident {
		t.Fatalf("in-flight status dropped or reordered desired references: %+v", status)
	}
	p := decodeModelConfigCall[nodepolicy.ReconcileResidentParams](t, modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)[0])
	if p.DeclaredResidentCount != 2 {
		t.Fatalf("revision's two-model resident set sent count=%d", p.DeclaredResidentCount)
	}
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
}

func TestModelConfigCapacityEvidenceRetriesOneRefusedAttempt(t *testing.T) {
	f := newModelConfigFixture(t)
	f.proxy.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == nodepolicy.MethodPrepareResident {
			return nodepolicy.PrepareResidentResult{EvidenceGeneration: 3, Refusals: []nodepolicy.ResidentRefusal{{Engine: "llamacpp", Model: "resident", Reason: nodepolicy.ModelReasonNoFit}}}, ""
		}
		return map[string]bool{"ok": true}, ""
	})
	f.b.reconcileModelConfig(context.Background())
	f.b.reconcileModelConfig(context.Background())
	if len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)) != 1 {
		t.Fatal("unchanged capacity retried refusal")
	}
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.admission.ActiveByEngine = map[string]int{"llamacpp": 1}
	r.mu.Unlock()
	f.b.reconcileModelConfig(context.Background())
	f.b.reconcileModelConfig(context.Background())
	if len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)) != 2 {
		t.Fatal("changed capacity did not allow exactly one retry")
	}
	if len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 {
		t.Fatal("capacity refusals reached manager load")
	}
}

func TestModelConfigProxyReplacementRequiresFreshInventory(t *testing.T) {
	f := newModelConfigFixture(t)
	f.proxy.mu.Lock()
	answer := f.proxy.answer
	f.proxy.mu.Unlock()
	pp, proxy, acks := attachAckingProxy(t, f.b)
	if !f.b.replayPolicyToProxy(context.Background(), pp) {
		t.Fatal("replacement proxy did not activate current policy")
	}
	proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetPolicy {
			return acks.answer(method, raw)
		}
		return answer(method, raw)
	})
	reportStrict(t, f.b, "0", acks.runID, strictCapability(t, "llamacpp", "resident"))
	e := f.b.captureModelConfig()
	if e.Activation.Activation != activationActive || e.Readiness.Capability == nil {
		t.Fatal("replacement fixture lacks fresh activation and strict readiness")
	}
	if e.Loaded != nil {
		t.Fatal("replacement proxy inherited old proxy's loaded inventory")
	}
	f.b.reconcileModelConfig(context.Background())
	if len(modelConfigCalls(proxy, nodepolicy.MethodPrepareResident)) != 0 || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 {
		t.Fatal("replacement activation/readiness authorized a load before fresh inventory")
	}
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].ObservedLoaded != nil || status.Models[0].State != nodepolicy.ModelStateUnknown {
		t.Fatalf("replacement without fresh inventory status=%+v", status)
	}

	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {}})
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	loadedProxy := r.models.loadedProxy
	r.mu.Unlock()
	if loadedProxy != pp {
		t.Fatal("fresh source-paired observation was not bound to replacement proxy")
	}
	f.b.reconcileModelConfig(context.Background())
	if len(modelConfigCalls(proxy, nodepolicy.MethodPrepareResident)) != 1 || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
		t.Fatal("fresh replacement-bound inventory did not permit one resident attempt")
	}
}

func TestModelConfigConcurrentEvidenceChangeIsReevaluated(t *testing.T) {
	cases := []struct {
		name   string
		change func(*modelConfigFixture)
	}{
		{"inventory", func(f *modelConfigFixture) {
			f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {}})
		}},
		{"capacity", func(f *modelConfigFixture) {
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			r.admission.ActiveByEngine = map[string]int{"llamacpp": 1}
			r.mu.Unlock()
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newModelConfigFixture(t)
			f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"neighbor"}})
			done := f.startLoadedAttempt(t)
			f.awaitGrant(t)
			tc.change(f)
			f.engine.releaseHeld()
			awaitModelConfigDone(t, done)
			status := f.b.modelConfiguration(f.b.captureModelConfig())
			if status.Models[0].State == nodepolicy.ModelStateSatisfied {
				t.Fatalf("completion retained satisfaction after concurrent %s change: %+v", tc.name, status)
			}
			if tc.name == "inventory" && (status.Models[0].ObservedLoaded == nil || *status.Models[0].ObservedLoaded) {
				t.Fatalf("concurrent empty inventory lost its observation: %+v", status)
			}
			f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
				if method == settings.SwitchPrepareMethod {
					return emptySwitchManagerSnapshot(), ""
				}
				return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
			})
			f.b.reconcileModelConfig(context.Background())
			f.b.reconcileModelConfig(context.Background())
			if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 2 {
				t.Fatalf("concurrent %s change permitted %d attempts, want initial and one retry", tc.name, got)
			}
		})
	}
}

func TestModelConfigNotificationWithoutLoadedInventoryClearsKnowledge(t *testing.T) {
	f := newModelConfigFixture(t)
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.b.handlePolicyEngineNotification(f.source, engineModelsChangedMethod, json.RawMessage(`{"ok":true}`))
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].ObservedLoaded == nil || !*status.Models[0].ObservedLoaded {
		t.Fatalf("malformed notification changed known inventory: %+v", status)
	}
	f.b.handlePolicyEngineNotification(f.source, engineModelsChangedMethod, json.RawMessage(`{"models":{"models":[]}}`))
	status = f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].ObservedLoaded != nil || status.Models[0].State != nodepolicy.ModelStateUnknown {
		t.Fatalf("valid snapshot without loaded inventory retained knowledge: %+v", status)
	}
	f.b.reconcileModelConfig(context.Background())
	if len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)) != 0 || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 {
		t.Fatal("unknown inventory notification permitted resident load")
	}
	f.b.handlePolicyEngineNotification(f.source, engineModelsChangedMethod, json.RawMessage(`{"models":{"models":[],"loadedByEngine":{"llamacpp":[]}}}`))
	status = f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].ObservedLoaded == nil || *status.Models[0].ObservedLoaded {
		t.Fatalf("explicit empty inventory was not preserved: %+v", status)
	}
	f.b.reconcileModelConfig(context.Background())
	if len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
		t.Fatal("explicit empty inventory did not permit resident attempt")
	}
}
