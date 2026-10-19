// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// model_mutation.go answers engine-manager's internal model-mutation request
// (nodepolicy.MethodModelMutationRequest): engine-manager asks before it
// installs, deletes, pulls or copies models into an engine, and the broker
// withdraws that engine's model identity from the proxy before answering.
// The request arrives on engine-manager's read pump; the withdrawal is done
// synchronously there, and the confirmed push runs on its own goroutine so the
// same pump stays free to deliver the proxy's acknowledgement and the reply.
//
// The reply goes through the worker bound to the source that sent the request
// (broker.go), never through getEngineMgr: a request from an exited worker must
// not be able to name, or answer through, its replacement.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"

	"nvpair-shared/nodepolicy"
)

// errModelMutationUnavailable is the failure a request whose worker pairing is
// gone reports. It never authorizes a mutation and engine-manager's own
// deadline covers the case where there is nobody left to answer.
var errModelMutationUnavailable = errors.New("the engine manager that requested the model mutation is not the worker this broker bound")

// observeModelMutationRequest withdraws engine's model identity synchronously
// and confirms it off the read pump. On a stale source it does nothing: the
// replacement's identity is not this request's to withdraw.
func (b *Broker) observeModelMutationRequest(source *engineManagerSource, params json.RawMessage) {
	var req nodepolicy.ModelMutationRequest
	if err := json.Unmarshal(params, &req); err != nil {
		slog.Warn("ignoring an unreadable model mutation request", "err", err)
		return
	}
	if !b.withdrawModelIdentity(source, req.Engine) {
		return
	}
	go func() {
		if err := b.confirmModelMutation(source, req); err != nil {
			slog.Warn("model mutation confirmation failed",
				"engine", req.Engine, "requestId", req.RequestID, "err", err)
		}
	}()
}

// withdrawModelIdentity removes the engine's engine-manager identity under
// r.mu. The settings fingerprint stays: a content mutation does not change the
// applied settings, and modelIdentityLocked omits an engine without a reported
// identity, so every join is withdrawn until engine-manager publishes a fresh
// snapshot. Reports whether source is still current.
func (b *Broker) withdrawModelIdentity(source *engineManagerSource, engine string) bool {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if source == nil || r.engineManagerSource != source {
		r.mu.Unlock()
		return false
	}
	_, known := r.engineIdentity[engine]
	delete(r.engineIdentity, engine)
	r.mu.Unlock()
	if known {
		// Stop advertising strict admission for an engine the proxy no longer
		// has an identity for. The confirmed push below follows.
		go b.refreshStrictAdvertisement()
	}
	return true
}

// confirmModelMutation waits for the pair bound to source (only a request read
// before its bind waits), confirms the proxy accepted an identity without the
// engine, then replies through that same worker. The whole wait, binding
// included, draws on the confirmation budget.
func (b *Broker) confirmModelMutation(source *engineManagerSource, req nodepolicy.ModelMutationRequest) error {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	bound := r.activationBoundLocked()
	r.mu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), bound)
	defer cancel()

	worker, err := b.awaitEngineMgrPair(ctx, source)
	if err != nil {
		b.replyModelMutation(source, nil, req, nodepolicy.ModelMutationUnavailable, err)
		return err
	}
	if err := b.pushModelIdentityConfirmed(ctx); err != nil {
		b.replyModelMutation(source, worker, req, nodepolicy.ModelMutationWithdrawalFailed, err)
		return err
	}
	r.mu.Lock()
	current := source != nil && r.engineManagerSource == source
	r.mu.Unlock()
	if !current {
		err := errors.New("the engine manager was replaced before the model identity was confirmed")
		b.replyModelMutation(source, worker, req, nodepolicy.ModelMutationUnavailable, err)
		return err
	}
	b.replyModelMutation(source, worker, req, 0, nil)
	return nil
}

// awaitEngineMgrPair returns the worker bound to source, waiting only while
// source is current and nothing else owns the slot. Another source, a
// retirement, or the budget running out is a failure.
func (b *Broker) awaitEngineMgrPair(ctx context.Context, source *engineManagerSource) (*rpcWorker, error) {
	for {
		b.workersMu.Lock()
		worker, bound := b.engineMgr, b.engineMgrSource
		changed := b.engineMgrBindingSignalLocked()
		b.workersMu.Unlock()
		if worker != nil && bound == source {
			return worker, nil
		}
		if bound != nil && bound != source {
			return nil, errModelMutationUnavailable
		}
		if !b.engineManagerSourceCurrent(source) {
			return nil, errModelMutationUnavailable
		}
		select {
		case <-changed:
		case <-ctx.Done():
			return nil, fmt.Errorf("no engine manager worker was bound before the withdrawal deadline: %w", ctx.Err())
		}
	}
}

func (b *Broker) engineManagerSourceCurrent(source *engineManagerSource) bool {
	if source == nil {
		return false
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.engineManagerSource == source
}

// engineMgrPairCurrent reports whether (worker, source) is still the bound
// pair. A reply is sent only through that pair; there is no path through the
// current handle alone.
func (b *Broker) engineMgrPairCurrent(worker *rpcWorker, source *engineManagerSource) bool {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	return worker != nil && b.engineMgr == worker && b.engineMgrSource == source
}

// replyModelMutation notifies the bound worker, or does nothing when this
// request's worker is no longer bound. code is zero on success.
func (b *Broker) replyModelMutation(source *engineManagerSource, worker *rpcWorker, req nodepolicy.ModelMutationRequest, code int, failure error) {
	if !b.engineMgrPairCurrent(worker, source) {
		return
	}
	reply := nodepolicy.ModelMutationReply{RequestID: req.RequestID, Engine: req.Engine, OK: failure == nil, Code: code}
	if failure != nil {
		reply.Error = failure.Error()
	}
	if err := worker.Notify(nodepolicy.MethodModelMutationReply, reply); err != nil {
		slog.Warn("failed to deliver a model mutation reply",
			"engine", req.Engine, "requestId", req.RequestID, "err", err)
	}
}
