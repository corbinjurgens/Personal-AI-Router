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
	rpcOK(t, p, w, nodepolicy.MethodSetEngineDrain, nodepolicy.SetEngineDrainParams{Engine: "llamacpp", Drain: true})
	drained := snapshot()
	if drained.Active != 1 || drained.ActiveByEngine["llamacpp"] != 1 || len(before.ActiveByEngine) != 0 {
		t.Fatalf("snapshot after drain=%+v, initial=%+v", drained, before)
	}
	held.Finish(strictNotDispatched)
	if after := snapshot(); after.Active != 0 || after.ActiveByEngine["llamacpp"] != 0 {
		t.Fatalf("released snapshot=%+v", after)
	}
	if drained.ActiveByEngine["llamacpp"] != 1 {
		t.Fatal("snapshot map aliases live counts")
	}
	rpcOK(t, p, w, nodepolicy.MethodSetEngineDrain, nodepolicy.SetEngineDrainParams{Engine: "llamacpp", Drain: false})
	ticket := mustStrictTicket(t, h.a)
	if err := ticket.BeforeStrictDispatch(); err != nil {
		t.Fatal(err)
	}
	ticket.Finish(strictUncertain)
	if got := snapshot(); got.Active != 1 || got.ActiveByEngine["llamacpp"] != 1 {
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
	activity := first.LastActivityMs["llamacpp"]
	first.ActiveByEngine["llamacpp"] = 99
	first.LastActivityMs["llamacpp"] = 0
	next, err := p.applyAdmissionSnapshot(json.RawMessage(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	if next.ActiveByEngine["llamacpp"] != 1 || activity == 0 || next.LastActivityMs["llamacpp"] != activity {
		t.Fatalf("snapshot maps alias controller state: %+v", next)
	}
}

func TestAdmissionSnapshotRPCRejectsInvalidParamsWithoutMutation(t *testing.T) {
	h := newStrictHarness(t)
	p, w := policyProxy(t)
	p.admission = h.a
	for name, params := range map[string]json.RawMessage{
		"missing params": nil,
		"null":           json.RawMessage(`null`),
		"unknown field":  json.RawMessage(`{"engine":"llamacpp"}`),
	} {
		t.Run(name, func(t *testing.T) {
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
		})
	}
}

func TestAdmissionSnapshotPreservesQuarantineOnIdentityWithdrawal(t *testing.T) {
	h := newStrictHarness(t)
	p := &Proxy{admission: h.a, runID: "proxy-run"}
	quarantined := mustStrictTicket(t, h.a)
	if err := quarantined.BeforeStrictDispatch(); err != nil {
		t.Fatal(err)
	}
	quarantined.Finish(strictUncertain)
	// A second model permits a held ticket alongside uncertain capacity.
	installQualified(t, h.a, admKey{engine: "llamacpp", model: "m"}, admKey{engine: "llamacpp", model: "n"})
	h.a.setResidency(map[string][]string{"llamacpp": {"m", "n"}})
	held, err := h.a.admitStrictNow(context.Background(), "llamacpp", "n", strictRequest())
	if err != nil {
		t.Fatal(err)
	}
	defer held.Finish(strictNotDispatched)
	before, err := os.ReadFile(h.a.strict.path)
	if err != nil {
		t.Fatal(err)
	}
	h.a.setModelIdentity(nil)
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
		t.Fatal("identity withdrawal changed durable uncertain markers")
	}
	if got := h.a.StrictReadiness(); got.Capability != nil {
		t.Fatalf("readiness advertised after identity withdrawal: %+v", got)
	}
	if got, err := p.applyAdmissionSnapshot(json.RawMessage(`{}`)); err != nil || got.RunID != "proxy-run" || got.Active != 1 || got.ActiveByEngine["llamacpp"] != 1 {
		t.Fatalf("quarantined snapshot=%+v error=%v", got, err)
	}
}

func TestAdmissionSnapshotCountsQueuedUntilIdentityWithdrawal(t *testing.T) {
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
	if err != nil || snapshot.RunID != "proxy-run" || snapshot.Active != 1 || snapshot.Queued != 1 {
		t.Fatalf("queued snapshot=%+v error=%v", snapshot, err)
	}
	h.a.setModelIdentity(nil)
	if result := awaitStrict(t, waiting); result.ticket != nil || !errors.Is(result.err, errStrictUnavailable) {
		t.Fatalf("queued result=%+v", result)
	}
	if got, err := p.applyAdmissionSnapshot(json.RawMessage(`{}`)); err != nil || got.Queued != 0 {
		t.Fatalf("released queue snapshot=%+v error=%v", got, err)
	}
}

func TestAdmissionSnapshotPreservesDispatchedWorkOnIdentityWithdrawal(t *testing.T) {
	h := newStrictHarness(t)
	p := &Proxy{admission: h.a, runID: "proxy-run"}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ticket, err := h.a.admitStrictNow(ctx, "llamacpp", "m", strictRequest())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket.Finish(strictQualifiedComplete)
	if err := ticket.BeforeStrictDispatch(); err != nil {
		t.Fatal(err)
	}
	h.a.setModelIdentity(nil)
	if ctx.Err() != nil || h.a.currentActive() != 1 || len(h.a.strict.markers) != 1 {
		t.Fatal("identity withdrawal cancelled or released dispatched work")
	}
	if err := ticket.BeforeStrictDispatch(); err != nil {
		t.Fatalf("dispatched ticket changed: %v", err)
	}
	if got, err := p.applyAdmissionSnapshot(json.RawMessage(`{}`)); err != nil || got.Active != 1 || got.ActiveByEngine["llamacpp"] != 1 {
		t.Fatalf("dispatched snapshot=%+v error=%v", got, err)
	}
	ticket.Finish(strictQualifiedComplete)
	if h.a.currentActive() != 0 || len(readCapacity(t, h.a.strict.path).Slots) != 0 {
		t.Fatal("qualified completion did not release dispatched work")
	}
}
