// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

// mutationFixture is an activated broker with a seeded strict identity, a
// current engine-manager source, and the fake worker that source is bound to.
type mutationFixture struct {
	*activatedBroker
	source *engineManagerSource
	worker *rpcWorker
	engine *fakeWorker
}

func newMutationFixture(t *testing.T) *mutationFixture {
	t.Helper()
	ab := newActivatedBroker(t)
	source := ab.b.activateEngineManagerSource()
	seedStrictIdentity(ab.b, "ollama")
	ab.b.identityChanged()
	// Let the seeded identity reach the proxy, so the next identity push it
	// receives is the one the confirmation sends.
	waitIdentity(t, ab.proxy, "the seeded identity", func(p nodepolicy.SetModelIdentityParams) bool {
		_, ok := p.Engines["ollama"]
		return ok
	})
	worker, engine := attachBoundEngineManager(t, ab.b, source)
	return &mutationFixture{activatedBroker: ab, source: source, worker: worker, engine: engine}
}

// attachBoundEngineManager installs a fake engine-manager worker and binds it
// the way spawnEngineMgr does.
func attachBoundEngineManager(t *testing.T, b *Broker, source *engineManagerSource) (*rpcWorker, *fakeWorker) {
	t.Helper()
	peer, engine := newFakeWorkerPipe(t)
	worker := &rpcWorker{peer: peer}
	b.setEngineMgr(worker)
	b.bindEngineMgr(worker, source)
	return worker, engine
}

func mutationParams(t *testing.T, requestID, engine string) json.RawMessage {
	t.Helper()
	data, err := json.Marshal(nodepolicy.ModelMutationRequest{RequestID: requestID, Engine: engine})
	if err != nil {
		t.Fatalf("encode model mutation request: %v", err)
	}
	return data
}

// joinedEngineIdentity is the identity the broker would send the proxy for
// engine now.
func joinedEngineIdentity(t *testing.T, b *Broker, engine string) (nodepolicy.EngineIdentity, bool) {
	t.Helper()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	id, ok := r.modelIdentityLocked().Engines[engine]
	return id, ok
}

func mutationReplies(t *testing.T, engine *fakeWorker) []nodepolicy.ModelMutationReply {
	t.Helper()
	var out []nodepolicy.ModelMutationReply
	for _, call := range engine.snapshot() {
		if call.method != nodepolicy.MethodModelMutationReply {
			continue
		}
		if !call.notify {
			t.Fatalf("%s arrived as a request, want the internal notification", call.method)
		}
		var reply nodepolicy.ModelMutationReply
		if err := json.Unmarshal(call.params, &reply); err != nil {
			t.Fatalf("decode %s %s: %v", call.method, call.params, err)
		}
		out = append(out, reply)
	}
	return out
}

func waitMutationReply(t *testing.T, engine *fakeWorker, requestID string) nodepolicy.ModelMutationReply {
	t.Helper()
	var found nodepolicy.ModelMutationReply
	engine.waitFor("the model mutation reply for "+requestID, func([]policyCall) bool {
		for _, reply := range mutationReplies(t, engine) {
			if reply.RequestID == requestID {
				found = reply
				return true
			}
		}
		return false
	})
	return found
}

func TestModelMutationRequestWithdrawsBeforeTheProxyAccepts(t *testing.T) {
	f := newMutationFixture(t)
	before := len(identityCalls(t, f.proxy.snapshot()))
	f.proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetModelIdentity {
			return map[string]bool{"ok": true}, holdSuccessReply
		}
		return f.acks.answer(method, params)
	})

	f.b.forwardEngineNotificationFrom(f.source, nodepolicy.MethodModelMutationRequest, mutationParams(t, "mutation-1", "ollama"))
	f.proxy.waitFor("the withdrawal push", func(calls []policyCall) bool {
		return len(identityCalls(t, calls)) > before
	})

	// The confirmation request reached the proxy and has not been answered;
	// the broker's join must already omit the engine.
	if _, ok := joinedEngineIdentity(t, f.b, "ollama"); ok {
		t.Fatal("the broker still joins the engine's identity while the proxy has not acknowledged the withdrawal")
	}
	if replies := mutationReplies(t, f.engine); len(replies) != 0 {
		t.Fatalf("the broker replied before the proxy accepted: %+v", replies)
	}

	f.proxy.releaseHeld()
	reply := waitMutationReply(t, f.engine, "mutation-1")
	if !reply.OK || reply.Code != 0 || reply.Error != "" {
		t.Fatalf("reply = %+v, want success", reply)
	}
	if _, ok := joinedEngineIdentity(t, f.b, "ollama"); ok {
		t.Fatal("the engine's identity returned before engine-manager published a fresh snapshot")
	}
}

func TestModelMutationConfirmationFailureKeepsIdentityWithdrawn(t *testing.T) {
	t.Run("the proxy is busy", func(t *testing.T) {
		f := newMutationFixture(t)
		setActivationWait(f.b, 100*time.Millisecond)
		r := f.b.nodePolicyRuntime()
		if !r.acquirePush(context.Background()) {
			t.Fatal("the push slot was not free")
		}
		defer r.releasePush()

		f.b.forwardEngineNotificationFrom(f.source, nodepolicy.MethodModelMutationRequest, mutationParams(t, "busy-1", "ollama"))
		reply := waitMutationReply(t, f.engine, "busy-1")
		if reply.OK || reply.Code != nodepolicy.ModelMutationWithdrawalFailed {
			t.Fatalf("reply = %+v, want a withdrawal failure", reply)
		}
		if !strings.Contains(reply.Error, "the proxy is busy") {
			t.Fatalf("reply error = %q, want the busy proxy text", reply.Error)
		}
		if _, ok := joinedEngineIdentity(t, f.b, "ollama"); ok {
			t.Fatal("a failed confirmation restored the engine's identity")
		}
	})

	t.Run("the proxy refuses", func(t *testing.T) {
		f := newMutationFixture(t)
		setActivationWait(f.b, 100*time.Millisecond)
		f.proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
			if method == nodepolicy.MethodSetModelIdentity {
				return nil, "identity rejected"
			}
			return f.acks.answer(method, params)
		})

		f.b.forwardEngineNotificationFrom(f.source, nodepolicy.MethodModelMutationRequest, mutationParams(t, "refused-1", "ollama"))
		reply := waitMutationReply(t, f.engine, "refused-1")
		if reply.OK || reply.Code != nodepolicy.ModelMutationWithdrawalFailed {
			t.Fatalf("reply = %+v, want a withdrawal failure", reply)
		}
		if !strings.Contains(reply.Error, "refused to withdraw") {
			t.Fatalf("reply error = %q, want the refusal text", reply.Error)
		}
		if _, ok := joinedEngineIdentity(t, f.b, "ollama"); ok {
			t.Fatal("a refused confirmation restored the engine's identity")
		}
	})

	t.Run("the proxy does not answer", func(t *testing.T) {
		f := newMutationFixture(t)
		setActivationWait(f.b, 100*time.Millisecond)
		f.proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
			if method == nodepolicy.MethodSetModelIdentity {
				return nil, holdReply
			}
			return f.acks.answer(method, params)
		})

		f.b.forwardEngineNotificationFrom(f.source, nodepolicy.MethodModelMutationRequest, mutationParams(t, "silent-1", "ollama"))
		reply := waitMutationReply(t, f.engine, "silent-1")
		if reply.OK || reply.Code != nodepolicy.ModelMutationWithdrawalFailed {
			t.Fatalf("reply = %+v, want a transport failure", reply)
		}
		if !strings.Contains(reply.Error, "strict admission") {
			t.Fatalf("reply error = %q, want the transport failure text", reply.Error)
		}
		if _, ok := joinedEngineIdentity(t, f.b, "ollama"); ok {
			t.Fatal("an unanswered confirmation restored the engine's identity")
		}
	})
}

func TestModelMutationConfirmsWithNoRunningProxy(t *testing.T) {
	b, _ := newPolicyBroker(t)
	source := b.activateEngineManagerSource()
	seedStrictIdentity(b, "ollama")
	_, engine := attachBoundEngineManager(t, b, source)

	b.forwardEngineNotificationFrom(source, nodepolicy.MethodModelMutationRequest, mutationParams(t, "no-proxy-1", "ollama"))
	reply := waitMutationReply(t, engine, "no-proxy-1")
	if !reply.OK {
		t.Fatalf("reply = %+v, want success with no proxy to withdraw from", reply)
	}
	if _, ok := joinedEngineIdentity(t, b, "ollama"); ok {
		t.Fatal("the engine's identity is still joined")
	}
}

func TestModelMutationFromAReplacedSourceLeavesTheReplacement(t *testing.T) {
	ab := newActivatedBroker(t)
	old := ab.b.activateEngineManagerSource()
	attachBoundEngineManager(t, ab.b, old)
	seedStrictIdentity(ab.b, "ollama")

	replacement := ab.b.activateEngineManagerSource()
	ab.b.observeModelIdentity(replacement, engineManagerSnapshot("ollama"))
	_, replacementEngine := attachBoundEngineManager(t, ab.b, replacement)
	if _, ok := joinedEngineIdentity(t, ab.b, "ollama"); !ok {
		t.Fatal("the replacement's identity is not joined before the stale request")
	}

	ab.b.forwardEngineNotificationFrom(old, nodepolicy.MethodModelMutationRequest, mutationParams(t, "stale-1", "ollama"))

	if _, ok := joinedEngineIdentity(t, ab.b, "ollama"); !ok {
		t.Fatal("a request from the replaced source withdrew the replacement's identity")
	}
	if replies := mutationReplies(t, replacementEngine); len(replies) != 0 {
		t.Fatalf("the broker replied through the replacement: %+v", replies)
	}
}

func TestModelMutationWaitsForTheStartWindowBinding(t *testing.T) {
	ab := newActivatedBroker(t)
	source := ab.b.activateEngineManagerSource()
	seedStrictIdentity(ab.b, "ollama")
	peer, engine := newFakeWorkerPipe(t)
	worker := &rpcWorker{peer: peer}
	// Installed but not bound: the read pump can already deliver a request.
	ab.b.setEngineMgr(worker)

	ab.b.forwardEngineNotificationFrom(source, nodepolicy.MethodModelMutationRequest, mutationParams(t, "start-1", "ollama"))
	if replies := mutationReplies(t, engine); len(replies) != 0 {
		t.Fatalf("the broker replied before the worker was bound: %+v", replies)
	}
	ab.b.bindEngineMgr(worker, source)

	reply := waitMutationReply(t, engine, "start-1")
	if !reply.OK {
		t.Fatalf("reply = %+v, want success after the bind", reply)
	}
}

func TestModelMutationRequestThatOutlivesItsSourceFails(t *testing.T) {
	ab := newActivatedBroker(t)
	source := ab.b.activateEngineManagerSource()
	seedStrictIdentity(ab.b, "ollama")
	peer, engine := newFakeWorkerPipe(t)
	ab.b.setEngineMgr(&rpcWorker{peer: peer})
	// A long budget proves the failure below is prompt, not a timeout.
	setActivationWait(ab.b, 30*time.Second)

	done := make(chan error, 1)
	go func() {
		done <- ab.b.confirmModelMutation(source, nodepolicy.ModelMutationRequest{RequestID: "retire-1", Engine: "ollama"})
	}()
	ab.b.retireEngineManagerSource(source)
	ab.b.clearEngineMgrBinding(source)

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("a confirmation whose source retired succeeded")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the confirmation did not fail promptly after its source retired")
	}
	if replies := mutationReplies(t, engine); len(replies) != 0 {
		t.Fatalf("the broker replied with no bound pair: %+v", replies)
	}
}

func TestModelMutationPairMismatchFails(t *testing.T) {
	ab := newActivatedBroker(t)
	source := ab.b.activateEngineManagerSource()
	seedStrictIdentity(ab.b, "ollama")
	firstPeer, firstEngine := newFakeWorkerPipe(t)
	ab.b.setEngineMgr(&rpcWorker{peer: firstPeer})
	// A different worker is installed, which clears the bound source.
	secondPeer, secondEngine := newFakeWorkerPipe(t)
	ab.b.setEngineMgr(&rpcWorker{peer: secondPeer})
	setActivationWait(ab.b, 100*time.Millisecond)

	err := ab.b.confirmModelMutation(source, nodepolicy.ModelMutationRequest{RequestID: "mismatch-1", Engine: "ollama"})
	if err == nil {
		t.Fatal("a confirmation without a bound pair succeeded")
	}
	for name, engine := range map[string]*fakeWorker{"the replaced worker": firstEngine, "the replacement": secondEngine} {
		if replies := mutationReplies(t, engine); len(replies) != 0 {
			t.Fatalf("the broker replied through %s: %+v", name, replies)
		}
	}
}

func TestModelMutationClientRequestCannotConfirm(t *testing.T) {
	f := newMutationFixture(t)
	f.b.handleMessage(clientRequest(700, nodepolicy.MethodModelMutationRequest, map[string]any{
		"requestId": "client-1",
		"engine":    "ollama",
	}))

	if _, ok := joinedEngineIdentity(t, f.b, "ollama"); !ok {
		t.Fatal("a client request withdrew the engine's identity")
	}
	if replies := mutationReplies(t, f.engine); len(replies) != 0 {
		t.Fatalf("a client request produced a model mutation reply: %+v", replies)
	}
}

func TestModelMutationRequestIsNotForwardedToSubscribedClients(t *testing.T) {
	f := newMutationFixture(t)
	f.b.engineMu.Lock()
	f.b.engineSubscribed = true
	f.b.engineMu.Unlock()

	f.b.forwardEngineNotificationFrom(f.source, nodepolicy.MethodModelMutationRequest, mutationParams(t, "forward-1", "ollama"))

	for _, frame := range f.out.frames() {
		if frame.Method == nodepolicy.MethodModelMutationRequest {
			t.Fatal("the internal model mutation request was forwarded to a subscribed client")
		}
	}
	// The request was still consumed: the confirmation answers it.
	reply := waitMutationReply(t, f.engine, "forward-1")
	if !reply.OK {
		t.Fatalf("reply = %+v, want success", reply)
	}
}
