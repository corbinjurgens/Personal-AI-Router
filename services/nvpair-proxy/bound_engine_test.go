// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

func TestBoundEngineCorrelatedNotification(t *testing.T) {
	requests := make(chan nodepolicy.PrepareBoundEngineRequest, 1)
	w := bufferAckWriter(func(raw []byte) (int, error) {
		var frame Message
		if err := json.Unmarshal(raw, &frame); err != nil {
			return 0, err
		}
		if frame.IsRequest() {
			return 0, errors.New("preparation used upward Call")
		}
		if frame.Method == nodepolicy.MethodPrepareBoundEngine {
			var in nodepolicy.PrepareBoundEngineRequest
			if err := json.Unmarshal(frame.Params, &in); err != nil {
				return 0, err
			}
			requests <- in
		}
		return len(raw), nil
	})
	p := NewProxy(NewCodec(w))
	done := make(chan nodepolicy.BoundEnginePrepared, 1)
	go func() {
		result, err := p.prepareBoundEngine(context.Background(), nodepolicy.PrepareBoundEngineRequest{PolicyRevision: "1", Engine: "ollama", Model: "model", BudgetMs: 1000})
		if err != nil {
			result.Message = err.Error()
		}
		done <- result
	}()
	in := <-requests
	if in.RequestID == "" || in.ProxyRunID != p.runID {
		t.Fatalf("correlation=%+v", in)
	}
	stale := nodepolicy.BoundEnginePrepared{RequestID: in.RequestID, ProxyRunID: "old-run", Success: true}
	p.receiveBoundEnginePrepared(egressJSON(t, stale))
	select {
	case result := <-done:
		t.Fatalf("stale reply completed waiter: %+v", result)
	default:
	}
	stale.ProxyRunID = p.runID
	p.handleMessage(&Message{Method: nodepolicy.NotifyBoundEnginePrepared, Params: egressJSON(t, stale)})
	select {
	case result := <-done:
		if !result.Success {
			t.Fatalf("result=%+v", result)
		}
	case <-time.After(time.Second):
		t.Fatal("reply reader did not complete waiter")
	}
	p.boundEngines.mu.Lock()
	retained := len(p.boundEngines.pending)
	p.boundEngines.mu.Unlock()
	if retained != 0 || p.admission.currentActive() != 0 {
		t.Fatal("waiter retained or admission ticket acquired")
	}
	p.receiveBoundEnginePrepared(egressJSON(t, stale)) // retired duplicate is inert
}

func TestBoundEngineCancellationRetiresWaiter(t *testing.T) {
	requests := make(chan nodepolicy.PrepareBoundEngineRequest, 1)
	cancels := make(chan nodepolicy.CancelBoundEngineRequest, 1)
	w := bufferAckWriter(func(raw []byte) (int, error) {
		var frame Message
		if err := json.Unmarshal(raw, &frame); err != nil {
			return 0, err
		}
		switch frame.Method {
		case nodepolicy.MethodPrepareBoundEngine:
			var in nodepolicy.PrepareBoundEngineRequest
			if err := json.Unmarshal(frame.Params, &in); err != nil {
				return 0, err
			}
			requests <- in
		case nodepolicy.MethodCancelBoundEngine:
			var in nodepolicy.CancelBoundEngineRequest
			if err := json.Unmarshal(frame.Params, &in); err != nil {
				return 0, err
			}
			cancels <- in
		}
		return len(raw), nil
	})
	p := NewProxy(NewCodec(w))
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := p.prepareBoundEngine(ctx, nodepolicy.PrepareBoundEngineRequest{PolicyRevision: "1", Engine: "ollama", Model: "model", BudgetMs: 1000})
		done <- err
	}()
	in := <-requests
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error=%v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("cancellation blocked")
	}
	got := <-cancels
	if got.RequestID != in.RequestID || got.ProxyRunID != in.ProxyRunID {
		t.Fatalf("cancel=%+v request=%+v", got, in)
	}
	p.boundEngines.mu.Lock()
	retained := len(p.boundEngines.pending)
	p.boundEngines.mu.Unlock()
	if retained != 0 {
		t.Fatal("cancelled waiter retained")
	}
}

func TestBoundEngineWaiterCapacity(t *testing.T) {
	p := NewProxy(NewCodec(&recordingWriter{}))
	p.boundEngines.pending = make(map[string]chan nodepolicy.BoundEnginePrepared)
	for i := 0; i < bufferBudgetMaxWaiters; i++ {
		p.boundEngines.pending[string(rune(i))] = make(chan nodepolicy.BoundEnginePrepared, 1)
	}
	if _, err := p.prepareBoundEngine(context.Background(), nodepolicy.PrepareBoundEngineRequest{PolicyRevision: "1", Engine: "ollama", Model: "model", BudgetMs: 1000}); err == nil {
		t.Fatal("full queue accepted waiter")
	}
}

func TestBoundEngineSendsOnlyRemainingCallerBudget(t *testing.T) {
	requests := make(chan nodepolicy.PrepareBoundEngineRequest, 1)
	w := bufferAckWriter(func(raw []byte) (int, error) {
		var frame Message
		if err := json.Unmarshal(raw, &frame); err != nil {
			return 0, err
		}
		if frame.Method == nodepolicy.MethodPrepareBoundEngine {
			var in nodepolicy.PrepareBoundEngineRequest
			if err := json.Unmarshal(frame.Params, &in); err != nil {
				return 0, err
			}
			requests <- in
		}
		return len(raw), nil
	})
	p := NewProxy(NewCodec(w))
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := p.prepareBoundEngine(ctx, nodepolicy.PrepareBoundEngineRequest{PolicyRevision: "1", Engine: "ollama", Model: "model", BudgetMs: 90000})
		done <- err
	}()
	in := <-requests
	if in.BudgetMs < 1 || in.BudgetMs > 100 {
		t.Fatalf("transmitted budget=%d", in.BudgetMs)
	}
	select {
	case err := <-done:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("wait error=%v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("caller budget extended")
	}
}

func TestBoundEngineCancellationDuringNotificationWrite(t *testing.T) {
	w := newNotificationWriter(1)
	_, codec := ownedNotificationCodec(t, w, time.Second)
	p := NewProxy(codec)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := p.prepareBoundEngine(ctx, nodepolicy.PrepareBoundEngineRequest{PolicyRevision: "1", Engine: "ollama", Model: "model", BudgetMs: 1000})
		done <- err
	}()
	lifetimeReceive(t, w.entered)
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("write cancellation=%v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("blocked preparation write survived caller cancellation")
	}
	p.boundEngines.mu.Lock()
	pending := len(p.boundEngines.pending)
	p.boundEngines.mu.Unlock()
	if pending != 0 {
		t.Fatal("cancelled write retained waiter")
	}
}

func TestBoundEngineDeadlineIsAbsoluteAndClamped(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	req := httptest.NewRequest("POST", "/api/generate", nil).WithContext(ctx)
	req.Header.Set(nodepolicy.EnginePrepareBudgetHeader, "900000")
	start := time.Now().Add(-50 * time.Millisecond)
	req = withBoundEngineDeadline(req, start)
	deadline, ok := req.Context().Value(boundEngineDeadlineKey{}).(time.Time)
	callerDeadline, _ := ctx.Deadline()
	if !ok || !deadline.Equal(callerDeadline) {
		t.Fatalf("deadline=%v caller=%v", deadline, callerDeadline)
	}
	first := boundEngineRemaining(req.Context())
	// Refresh contexts inherit the same immutable absolute deadline, even
	// with a new, longer local preparation allowance.
	refreshed, stop := context.WithTimeout(req.Context(), time.Minute)
	defer stop()
	if got := boundEngineRemaining(refreshed); got > first || got > 200*time.Millisecond {
		t.Fatalf("refresh reset budget: first=%v refreshed=%v", first, got)
	}
	old := httptest.NewRequest("POST", "/api/generate", nil)
	old = withBoundEngineDeadline(old, time.Now().Add(-91*time.Second))
	if got := boundEngineRemaining(old.Context()); got != 0 {
		t.Fatalf("entry time was reset: %v", got)
	}
}

func TestBoundEngineAdmissionRevalidatesAfterQueueWait(t *testing.T) {
	fx := newBindingRequestFixture(t)
	e := newBindingEngine(t, "fixture", nil)
	setBackend(t, fx.f, e.URL, true)
	pol := fx.p.admission.currentPolicy()
	pol.Admission.MaxConcurrentPerModel = 1
	fx.p.admission.setPolicy(pol)
	a := fx.p.admission
	first := a.admit(context.Background(), admissionRequest{engine: "ollama", model: "fixture", waitCap: 0})
	if first.ticket == nil {
		t.Fatalf("initial admission=%+v", first)
	}
	defer first.ticket.release()
	a.mu.Lock()
	evidence := &boundEngineEvidence{backend: fx.f.currentLocalBackend(), policy: a.policyGeneration, identity: a.strict.identity["ollama"]}
	a.mu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	done := make(chan admissionDecision, 1)
	go func() {
		done <- a.admit(ctx, admissionRequest{engine: "ollama", model: "fixture", waitCap: time.Second, revalidate: func() bool { return fx.f.boundEvidenceCurrentLocked(evidence) }})
	}()
	waitForCond(t, time.Second, "bound request queued without a ticket", func() bool { a.mu.Lock(); defer a.mu.Unlock(); return a.queued == 1 })
	a.setModelIdentity(map[string]nodepolicy.EngineIdentity{"ollama": {EngineRunID: "replacement"}})
	select {
	case result := <-done:
		if result.ticket != nil || result.err != nil || result.reject != "model-unavailable" {
			t.Fatalf("stale queued admission=%+v", result)
		}
	case <-time.After(time.Second):
		t.Fatal("stale queued request did not wake")
	}
	if a.currentActive() != 1 || fx.f.beforeBoundDispatch(first.ticket, evidence) {
		t.Fatal("changed identity granted/retained stale dispatch authority")
	}
}

func TestBoundEngineConcreteFacadeOrderIsStable(t *testing.T) {
	fx := newBindingRequestFixture(t)
	fx.f.discovery.RemoveManual("self")
	for _, name := range []string{"lmstudio", "llamacpp"} {
		profile, ok := profileFor(name)
		if !ok {
			t.Fatalf("missing fixture profile %s", name)
		}
		f := newFacade(fx.p, profile, NewDiscovery(), profile.FacadePort)
		fx.p.facades[name] = f
		n := prNode(name)
		n.Models, n.PolicyRevision, n.EngineBindingOffers = []string{"fixture"}, "1", bindingOffer(name, "fixture")
		f.discovery.AddManual(n)
	}
	for range 32 {
		cands := fx.f.resolveConcreteCandidates("fixture", routeChatCompletions)
		if len(cands) != 2 || cands[0].engine != "llamacpp" || cands[1].engine != "lmstudio" {
			t.Fatalf("unstable facade fallback: %+v", cands)
		}
	}
}
