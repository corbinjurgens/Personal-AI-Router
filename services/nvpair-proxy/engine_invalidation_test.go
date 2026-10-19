// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"testing"

	"nvpair-shared/nodepolicy"
)

func engineInvalidation(engine, operation string) nodepolicy.InvalidateEngineParams {
	return nodepolicy.InvalidateEngineParams{Engine: engine, OperationID: operation, Reason: nodepolicy.InvalidationReasonEngineSwitch}
}

func TestInvalidateEngineFencesHeldTicketsAndPreservesQuarantine(t *testing.T) {
	h := newStrictHarness(t)
	p := &Proxy{admission: h.a, runID: "proxy-run"}
	quarantined := mustStrictTicket(t, h.a)
	if err := quarantined.BeforeStrictDispatch(); err != nil {
		t.Fatal(err)
	}
	quarantined.Finish(strictUncertain)
	// A second model permits a held ticket alongside uncertain capacity.
	installQualified(t, h.a, admKey{engine: "ollama", model: "m"}, admKey{engine: "ollama", model: "n"})
	h.a.setResidency(map[string][]string{"ollama": {"m", "n"}})
	held, err := h.a.admitStrictNow(context.Background(), "ollama", "n", strictRequest())
	if err != nil {
		t.Fatal(err)
	}
	defer held.Finish(strictNotDispatched)
	before, err := os.ReadFile(h.a.strict.path)
	if err != nil {
		t.Fatal(err)
	}
	result, err := p.applyInvalidateEngine(json.RawMessage(`{"engine":"ollama","operationId":"switch-1","reason":"engine-switch"}`))
	if err != nil || !result.OK || result.RunID != p.runID {
		t.Fatalf("result=%+v error=%v", result, err)
	}
	if len(h.a.strict.models) != 0 {
		t.Fatalf("installed models remain: %+v", h.a.strict.models)
	}
	if err := held.BeforeStrictDispatch(); !errors.Is(err, errStrictUnavailable) {
		t.Fatalf("held dispatch error=%v", err)
	}
	if h.a.currentActive() != 1 || len(h.a.strict.markers) != 1 {
		t.Fatalf("quarantine lost: active=%d markers=%+v", h.a.currentActive(), h.a.strict.markers)
	}
	after, err := os.ReadFile(h.a.strict.path)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Fatal("invalidation changed durable uncertain markers")
	}
	if got := h.a.StrictReadiness(); got.Capability != nil {
		t.Fatalf("readiness advertised after invalidation: %+v", got)
	}
}

func TestInvalidateEnginePreservesOtherEngineEligibility(t *testing.T) {
	h := newStrictHarness(t)
	installQualified(t, h.a, admKey{engine: "ollama", model: "m"}, admKey{engine: "lmstudio", model: "m"})
	h.a.setResidency(map[string][]string{"ollama": {"m"}, "lmstudio": {"m"}})
	held, err := h.a.admitStrictNow(context.Background(), "lmstudio", "m", strictRequest())
	if err != nil {
		t.Fatal(err)
	}
	defer held.Finish(strictQualifiedComplete)
	key := admKey{engine: "lmstudio", model: "m"}
	generation := h.a.strict.generation[key]
	h.a.invalidateEngine(engineInvalidation("ollama", "switch-1"))
	if h.a.strict.generation[key] != generation {
		t.Fatal("other engine generation changed")
	}
	if err := held.BeforeStrictDispatch(); err != nil {
		t.Fatalf("other engine's held ticket rejected: %v", err)
	}
	held.Finish(strictQualifiedComplete)
	next, err := h.a.admitStrictNow(context.Background(), "lmstudio", "m", strictRequest())
	if err != nil {
		t.Fatalf("other engine lost eligibility: %v", err)
	}
	next.Finish(strictNotDispatched)
	// The same switch operation can invalidate the target independently.
	h.a.invalidateEngine(engineInvalidation("lmstudio", "switch-1"))
	if _, err := h.a.admitStrictNow(context.Background(), "lmstudio", "m", strictRequest()); !errors.Is(err, errStrictUnavailable) {
		t.Fatalf("target engine not invalidated: %v", err)
	}
}

func TestInvalidateEngineRepeatDoesNotFenceFreshCapability(t *testing.T) {
	h := newStrictHarness(t)
	key := admKey{engine: "ollama", model: "m:latest"}
	generation := h.a.strict.generation[key]
	in := engineInvalidation("ollama", "switch-1")
	h.a.invalidateEngine(in)
	if h.a.strict.generation[key] != generation+1 {
		t.Fatal("invalidation did not advance generation")
	}
	configureStrict(t, h.a, h.a.strict.path)
	held := mustStrictTicket(t, h.a)
	defer held.Finish(strictQualifiedComplete)
	generation = h.a.strict.generation[key]
	h.a.invalidateEngine(in)
	if h.a.strict.generation[key] != generation {
		t.Fatal("repeat advanced generation")
	}
	if err := held.BeforeStrictDispatch(); err != nil {
		t.Fatalf("repeat invalidated newly installed capability: %v", err)
	}
	h.a.invalidateEngine(engineInvalidation("ollama", "switch-2"))
	if h.a.strict.generation[key] != generation+1 {
		t.Fatal("fresh operation did not invalidate")
	}
}

func TestIdentityWithdrawalAndSwitchInvalidationShareCapacityFence(t *testing.T) {
	h := newStrictHarness(t)
	held := mustStrictTicket(t, h.a)
	defer held.Finish(strictNotDispatched)
	key := admKey{engine: "ollama", model: "m:latest"}
	generation := h.a.strict.generation[key]
	h.a.setModelIdentity(nil)
	if h.a.strict.generation[key] != generation+1 {
		t.Fatal("identity withdrawal did not fence the old installation")
	}
	h.a.invalidateEngine(engineInvalidation("ollama", "switch-1"))
	if h.a.strict.generation[key] != generation+1 {
		t.Fatal("switch invalidation counted the withdrawn installation again")
	}
	h.a.setPolicy(h.a.currentPolicy())
	if got := h.a.StrictReadiness(); got.Capability != nil {
		t.Fatalf("policy refresh restored withdrawn identity: %+v", got)
	}
	h.a.setModelIdentity(map[string]nodepolicy.EngineIdentity{"ollama": testIdentity("m")})
	if err := held.BeforeStrictDispatch(); !errors.Is(err, errStrictUnavailable) {
		t.Fatalf("old ticket survived identity re-observation: %v", err)
	}
	if got := h.a.StrictReadiness(); got.Capability == nil || len(got.Capability.Models) != 1 {
		t.Fatalf("fresh qualified identity was not restored: %+v", got)
	}
}

func TestInvalidateEngineWakesQueuedWaiters(t *testing.T) {
	h := newStrictHarness(t)
	c := useStrictClock(h.a)
	first := mustStrictTicket(t, h.a)
	defer first.Finish(strictNotDispatched)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	waiting := asyncStrict(h.a, ctx, strictRequest())
	awaitStrictTimer(t, c)
	p := &Proxy{admission: h.a, runID: "proxy-run"}
	snapshot, err := p.applyAdmissionSnapshot(json.RawMessage(`{}`))
	if err != nil || snapshot.Active != 1 || snapshot.Queued != 1 {
		t.Fatalf("queued snapshot=%+v error=%v", snapshot, err)
	}
	h.a.invalidateEngine(engineInvalidation("ollama", "switch-1"))
	if result := awaitStrict(t, waiting); result.ticket != nil || !errors.Is(result.err, errStrictUnavailable) {
		t.Fatalf("queued result=%+v", result)
	}
	h.a.mu.Lock()
	queued := h.a.queued
	h.a.mu.Unlock()
	if queued != 0 {
		t.Fatalf("queued=%d", queued)
	}
}

func TestInvalidateEngineLeavesDispatchedWorkRunning(t *testing.T) {
	h := newStrictHarness(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ticket, err := h.a.admitStrictNow(ctx, "ollama", "m", strictRequest())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket.Finish(strictQualifiedComplete)
	if err := ticket.BeforeStrictDispatch(); err != nil {
		t.Fatal(err)
	}
	h.a.invalidateEngine(engineInvalidation("ollama", "switch-1"))
	if ctx.Err() != nil || h.a.currentActive() != 1 || len(h.a.strict.markers) != 1 {
		t.Fatal("invalidation cancelled or released dispatched work")
	}
	if err := ticket.BeforeStrictDispatch(); err != nil {
		t.Fatalf("dispatched ticket changed: %v", err)
	}
	ticket.Finish(strictQualifiedComplete)
	if h.a.currentActive() != 0 || len(readCapacity(t, h.a.strict.path).Slots) != 0 {
		t.Fatal("qualified completion did not release dispatched work")
	}
}

func TestAdmissionSnapshotRPCReflectsCurrentInflight(t *testing.T) {
	h := newStrictHarness(t)
	p, w := policyProxy(t)
	p.admission = h.a
	snapshot := func() nodepolicy.AdmissionSnapshotResult {
		t.Helper()
		f := rpc(t, p, w, nodepolicy.MethodAdmissionSnapshot, nodepolicy.AdmissionSnapshotParams{})
		if f.Error != nil {
			t.Fatalf("snapshot error=%+v", f.Error)
		}
		var result nodepolicy.AdmissionSnapshotResult
		if err := json.Unmarshal(f.Result, &result); err != nil {
			t.Fatal(err)
		}
		if result.RunID == "" || result.RunID != p.runID {
			t.Fatalf("snapshot run ID=%q, want=%q", result.RunID, p.runID)
		}
		return result
	}
	before := snapshot()
	if before.Active != 0 {
		t.Fatalf("initial snapshot=%+v", before)
	}
	held := mustStrictTicket(t, h.a)
	defer held.Finish(strictNotDispatched)
	rpcOK(t, p, w, nodepolicy.MethodSetEngineDrain, nodepolicy.SetEngineDrainParams{Engine: "ollama", Drain: true})
	drained := snapshot()
	if drained.Active != 1 || drained.ActiveByEngine["ollama"] != 1 || len(before.ActiveByEngine) != 0 {
		t.Fatalf("snapshot after drain=%+v, initial=%+v", drained, before)
	}
	held.Finish(strictNotDispatched)
	if after := snapshot(); after.Active != 0 || after.ActiveByEngine["ollama"] != 0 {
		t.Fatalf("released snapshot=%+v", after)
	}
	if drained.ActiveByEngine["ollama"] != 1 {
		t.Fatal("snapshot map aliases live counts")
	}
	rpcOK(t, p, w, nodepolicy.MethodSetEngineDrain, nodepolicy.SetEngineDrainParams{Engine: "ollama", Drain: false})
	ticket := mustStrictTicket(t, h.a)
	if err := ticket.BeforeStrictDispatch(); err != nil {
		t.Fatal(err)
	}
	ticket.Finish(strictUncertain)
	if got := snapshot(); got.Active != 1 || got.ActiveByEngine["ollama"] != 1 {
		t.Fatalf("quarantined snapshot=%+v", got)
	}
}

func TestAdmissionSnapshotCopiesMaps(t *testing.T) {
	h := newStrictHarness(t)
	p := &Proxy{admission: h.a, runID: "proxy-run"}
	ticket := mustStrictTicket(t, h.a)
	defer ticket.Finish(strictNotDispatched)
	first, err := p.applyAdmissionSnapshot(json.RawMessage(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	activity := first.LastActivityMs["ollama"]
	first.ActiveByEngine["ollama"] = 99
	first.LastActivityMs["ollama"] = 0
	next, err := p.applyAdmissionSnapshot(json.RawMessage(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	if next.ActiveByEngine["ollama"] != 1 || activity == 0 || next.LastActivityMs["ollama"] != activity {
		t.Fatalf("snapshot maps alias controller state: %+v", next)
	}
}

func TestInvalidateEngineRPCValidationAndAcknowledgement(t *testing.T) {
	h := newStrictHarness(t)
	p, w := policyProxy(t)
	p.admission = h.a
	for name, params := range map[string]json.RawMessage{
		"missing params":    nil,
		"null":              json.RawMessage(`null`),
		"invalid engine":    json.RawMessage(`{"engine":"../ollama","operationId":"switch-1","reason":"engine-switch"}`),
		"invalid operation": json.RawMessage(`{"engine":"ollama","operationId":"-switch","reason":"engine-switch"}`),
		"unknown reason":    json.RawMessage(`{"engine":"ollama","operationId":"switch-1","reason":"restart"}`),
		"extra field":       json.RawMessage(`{"engine":"ollama","operationId":"switch-1","reason":"engine-switch","force":true}`),
	} {
		t.Run(name, func(t *testing.T) {
			generation := h.a.strict.generation[admKey{engine: "ollama", model: "m:latest"}]
			f := rpc(t, p, w, nodepolicy.MethodInvalidateEngine, params)
			if f.Error == nil || f.Error.Code != -32602 {
				t.Fatalf("invalid request response=%+v", f)
			}
			if len(h.a.strict.models) != 1 || h.a.strict.generation[admKey{engine: "ollama", model: "m:latest"}] != generation {
				t.Fatal("invalid request changed eligibility")
			}
		})
	}
	f := rpc(t, p, w, nodepolicy.MethodInvalidateEngine, engineInvalidation("ollama", "switch-1"))
	if f.Error != nil {
		t.Fatalf("invalidation error=%+v", f.Error)
	}
	var result nodepolicy.InvalidateEngineResult
	if err := json.Unmarshal(f.Result, &result); err != nil {
		t.Fatal(err)
	}
	if !result.OK || result.RunID == "" || result.RunID != p.runID {
		t.Fatalf("acknowledgement=%+v", result)
	}
	for _, params := range []json.RawMessage{nil, json.RawMessage(`null`), json.RawMessage(`{"engine":"ollama"}`)} {
		h.a.mu.Lock()
		before := h.a.snapshotLocked().state
		h.a.mu.Unlock()
		f := rpc(t, p, w, nodepolicy.MethodAdmissionSnapshot, params)
		if f.Error == nil || f.Error.Code != -32602 {
			t.Fatalf("invalid snapshot response=%+v", f)
		}
		h.a.mu.Lock()
		after := h.a.snapshotLocked().state
		h.a.mu.Unlock()
		if !reflect.DeepEqual(before, after) {
			t.Fatal("invalid snapshot changed admission state")
		}
	}
}
