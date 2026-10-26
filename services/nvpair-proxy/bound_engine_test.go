// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
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
