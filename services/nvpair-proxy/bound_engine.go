// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"sync"
	"time"

	"nvpair-shared/nodepolicy"
)

type boundEngineWaiters struct {
	mu      sync.Mutex
	pending map[string]chan nodepolicy.BoundEnginePrepared
}

// Callers hold their request buffer, but no placement, resident or execution
// ticket. Routing must refresh all dispatch evidence after success.
func (p *Proxy) prepareBoundEngine(ctx context.Context, in nodepolicy.PrepareBoundEngineRequest) (nodepolicy.BoundEnginePrepared, error) {
	in.RequestID, in.ProxyRunID = rand.Text(), p.runID
	if err := in.Validate(); err != nil {
		return nodepolicy.BoundEnginePrepared{}, err
	}
	ctx, cancel := context.WithTimeout(ctx, time.Duration(in.BudgetMs)*time.Millisecond)
	defer cancel()
	deadline, _ := ctx.Deadline()
	in.BudgetMs = min(in.BudgetMs, int(time.Until(deadline).Milliseconds()))
	if in.BudgetMs < 1 {
		return nodepolicy.BoundEnginePrepared{}, context.DeadlineExceeded
	}
	w := &p.boundEngines
	ch := make(chan nodepolicy.BoundEnginePrepared, 1)
	w.mu.Lock()
	if len(w.pending) >= bufferBudgetMaxWaiters {
		w.mu.Unlock()
		return nodepolicy.BoundEnginePrepared{}, errors.New("engine preparation queue is full")
	}
	if w.pending == nil {
		w.pending = make(map[string]chan nodepolicy.BoundEnginePrepared)
	}
	w.pending[in.RequestID] = ch
	w.mu.Unlock()
	defer func() { w.mu.Lock(); delete(w.pending, in.RequestID); w.mu.Unlock() }()
	if err := ctx.Err(); err != nil {
		return nodepolicy.BoundEnginePrepared{}, err
	}
	if err := p.codec.NotifyContext(ctx, nodepolicy.MethodPrepareBoundEngine, in); err != nil {
		if ctx.Err() != nil {
			w.mu.Lock()
			delete(w.pending, in.RequestID)
			w.mu.Unlock()
			p.cancelBoundEngineWaiter(in.RequestID)
		}
		return nodepolicy.BoundEnginePrepared{}, err
	}
	select {
	case reply := <-ch:
		return reply, nil
	case <-ctx.Done():
		w.mu.Lock()
		delete(w.pending, in.RequestID)
		w.mu.Unlock()
		p.cancelBoundEngineWaiter(in.RequestID)
		return nodepolicy.BoundEnginePrepared{}, ctx.Err()
	}
}

func (p *Proxy) cancelBoundEngineWaiter(id string) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	_ = p.codec.NotifyContext(ctx, nodepolicy.MethodCancelBoundEngine, nodepolicy.CancelBoundEngineRequest{RequestID: id, ProxyRunID: p.runID})
}

func (p *Proxy) receiveBoundEnginePrepared(raw json.RawMessage) {
	var reply nodepolicy.BoundEnginePrepared
	if json.Unmarshal(raw, &reply) != nil || reply.ProxyRunID != p.runID {
		return
	}
	w := &p.boundEngines
	w.mu.Lock()
	defer w.mu.Unlock()
	if ch := w.pending[reply.RequestID]; ch != nil {
		delete(w.pending, reply.RequestID)
		ch <- reply
	}
}
