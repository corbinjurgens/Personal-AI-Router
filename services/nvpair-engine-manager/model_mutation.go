// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// model_mutation.go is the engine-manager side of the internal model-mutation
// handshake (nodepolicy.MethodModelMutationRequest): before this service
// installs, deletes, pulls or copies models into an engine, it asks the broker
// to withdraw that engine's model identity from the running proxy and waits for
// the correlated reply. Only an accepted reply authorizes the engine-visible
// change, so strict admission cannot keep admitting under a tuple the mutation
// is about to invalidate.
//
// The request is a notification on the same parent codec every other relay uses
// (manager.go, policyrelay.go); the reply arrives on the read loop and is handed
// to the pending entry it names. The send is synchronous and the wait is
// bounded, so a parent that stops reading cannot authorize anything.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

const (
	// modelMutationBudget sets a deadline before the synchronous send, enforced
	// on the reply wait once that send returns. As with the other relays, a
	// blocked send cannot authorize a mutation. The broker handles the same
	// request within its own activation bound (proxyCallTimeout,
	// 5s; broker policy.go activationBoundLocked), which covers waiting for
	// the bound worker and the confirmed proxy push. Three times that leaves
	// the broker room to consume its whole budget and still answer here, so
	// this side reports the broker's own refusal rather than racing its own
	// deadline.
	modelMutationBudget = 15 * time.Second
	// maxPendingModelMutations bounds requests waiting for a reply at once.
	maxPendingModelMutations = 32
)

// modelMutationRelay correlates engine:model-mutation-request notifications
// with the broker's engine:model-mutation-reply ones, exactly as policyRelay
// correlates policy:request with policy:reply. send is the parent codec's
// synchronous Notify; a nil send means no broker is attached.
type modelMutationRelay struct {
	mu      sync.Mutex
	pending map[string]chan nodepolicy.ModelMutationReply
	send    func(string, any) error
}

// call asks the broker to withdraw engine's model identity and waits for the
// correlated reply. A missing relay, a send error, a refusal, a mismatched
// reply, the deadline and cancellation are all failures; only ok=true
// succeeds. The pending entry is removed on every path that returns.
func (r *modelMutationRelay) call(ctx context.Context, engine string) error {
	if r == nil || r.send == nil {
		return fmt.Errorf("engine %q cannot change its models: this engine manager is not attached to a broker", engine)
	}
	ctx, cancel := context.WithTimeout(ctx, modelMutationBudget)
	defer cancel()
	id := newOpID()
	ch := make(chan nodepolicy.ModelMutationReply, 1)
	r.mu.Lock()
	if len(r.pending) >= maxPendingModelMutations {
		r.mu.Unlock()
		return fmt.Errorf("engine %q cannot change its models: too many model changes are already waiting for confirmation", engine)
	}
	if r.pending == nil {
		r.pending = make(map[string]chan nodepolicy.ModelMutationReply)
	}
	r.pending[id] = ch
	r.mu.Unlock()
	defer func() { r.mu.Lock(); delete(r.pending, id); r.mu.Unlock() }()

	if err := r.send(nodepolicy.MethodModelMutationRequest, nodepolicy.ModelMutationRequest{RequestID: id, Engine: engine}); err != nil {
		return fmt.Errorf("engine %q cannot change its models: the withdrawal request could not be sent: %w", engine, err)
	}
	select {
	case reply := <-ch:
		if reply.Engine != engine {
			return fmt.Errorf("engine %q cannot change its models: the broker confirmed a withdrawal for engine %q instead", engine, reply.Engine)
		}
		if !reply.OK {
			return fmt.Errorf("engine %q cannot change its models: the destination did not confirm the withdrawal: %s", engine, reply.Error)
		}
		if err := ctx.Err(); err != nil {
			return modelMutationContextError(engine, err)
		}
		return nil
	case <-ctx.Done():
		return modelMutationContextError(engine, ctx.Err())
	}
}

func modelMutationContextError(engine string, err error) error {
	if errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("engine %q cannot change its models: the withdrawal was not confirmed before the deadline: %w", engine, err)
	}
	return fmt.Errorf("engine %q cannot change its models: the withdrawal confirmation was cancelled: %w", engine, err)
}

// reply hands one decoded reply to the request waiting for it. A reply nobody
// waits for (an unknown request id) or an unreadable one is dropped: neither
// can authorize a mutation.
func (r *modelMutationRelay) reply(raw json.RawMessage) {
	var reply nodepolicy.ModelMutationReply
	if err := json.Unmarshal(raw, &reply); err != nil {
		slog.Warn("ignoring an unreadable model mutation reply", "err", err)
		return
	}
	r.mu.Lock()
	ch := r.pending[reply.RequestID]
	r.mu.Unlock()
	if ch == nil {
		return
	}
	select {
	case ch <- reply:
	default:
	}
}

// beginModelMutation sends the withdrawal request for engine and waits for the
// broker's confirmation, after suppressing identity and joining earlier
// publications. No publication or inventory lock is held across the wait.
func (e *Executor) beginModelMutation(ctx context.Context, engine string) error {
	e.inventoryMu.Lock()
	if e.inventoryDigests == nil {
		e.inventoryDigests = make(map[string]observedDigests)
	}
	current := e.inventoryDigests[engine]
	current.epoch++
	current.suppressed = true
	e.inventoryDigests[engine] = current
	e.inventoryMu.Unlock()
	// A previous synchronous publication can remain blocked in Notify. Hand
	// ownership over only while this caller is waiting; otherwise release the
	// mutex when the earlier publication eventually finishes.
	joined := make(chan struct{})
	go func() {
		e.publishMu.Lock()
		select {
		case joined <- struct{}{}:
		case <-ctx.Done():
			e.publishMu.Unlock()
		}
	}()
	select {
	case <-joined:
	case <-ctx.Done():
		return modelMutationContextError(engine, ctx.Err())
	}
	e.publishMu.Unlock()
	if err := ctx.Err(); err != nil {
		return modelMutationContextError(engine, err)
	}
	parent := e.modelMutationParent
	if parent == nil {
		return fmt.Errorf("engine %q cannot change its models: this engine manager is not attached to a broker", engine)
	}
	return parent(ctx, engine)
}

// endModelMutation runs on every exit of a guarded mutation, successful or not.
// Advance the epoch again so observations begun during the mutation cannot
// restore identity, even if they complete after the fresh sweep was requested.
func (e *Executor) endModelMutation(engine string, _ error) {
	e.inventoryMu.Lock()
	current := e.inventoryDigests[engine]
	current.epoch++
	current.suppressed = false
	current.models = nil
	current.runID = ""
	e.inventoryDigests[engine] = current
	e.inventoryMu.Unlock()
	e.pokeLoaded()
}

// withModelMutation runs fn as the only content mutation of engine in flight:
// it holds that engine's content semaphore (separate from the lifecycle opMu), asks
// the broker to withdraw the engine's model identity, and runs fn only once
// that withdrawal is confirmed. Every exit releases the mutex and ends the
// mutation. No engine-visible effect of fn happens before the confirmation.
func (e *Executor) withModelMutation(ctx context.Context, engine string, fn func() error) (err error) {
	st, err := e.state(engine)
	if err != nil {
		return err
	}
	select {
	case st.contentMu <- struct{}{}:
	case <-ctx.Done():
		return modelMutationContextError(engine, ctx.Err())
	}
	defer func() { <-st.contentMu }()
	if err := ctx.Err(); err != nil {
		return modelMutationContextError(engine, err)
	}
	defer func() { e.endModelMutation(engine, err) }()
	if err = e.beginModelMutation(ctx, engine); err != nil {
		return err
	}
	return fn()
}

// mutatesModelContent reports whether running action changes the engine's
// installed models. It resolves the manifest's semantic operation bindings
// (modelops.go) rather than the action's spelling, so an action bound to delete
// is guarded however the manifest names it. A pull is always the pull_model
// action (validateOperations), whose streaming owner takes the guard itself;
// this covers the action-dispatch path. Load, unload and every unbound action
// are not content mutations.
func (m *Manifest) mutatesModelContent(action string) bool {
	for _, op := range []string{engines.OperationDelete, engines.OperationPull} {
		if b, ok := m.operationBinding(op); ok && b.Action == action {
			return true
		}
	}
	return false
}
