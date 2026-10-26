// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

type boundHarness struct {
	*switchHarness
	sourceRunning, targetRunning atomic.Bool
	modelAvailable               atomic.Bool
	activations                  atomic.Int32
	request                      nodepolicy.PrepareBoundEngineRequest
}

func newBoundHarness(t *testing.T, sourceRunning, targetRunning bool) *boundHarness {
	t.Helper()
	h := &boundHarness{switchHarness: newSwitchHarness(t)}
	bindRemoteManagementTestSource(h.b)
	h.b.startedAt = time.Now().Add(-boundEngineDwell)
	h.sourceRunning.Store(sourceRunning)
	h.targetRunning.Store(targetRunning)
	h.modelAvailable.Store(true)
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current = nodepolicy.Default()
	r.current.Revision = "1"
	r.current.Availability = nodepolicy.Available
	r.current.EngineBindings = nodepolicy.EngineBindings{{Engine: "lmstudio", Model: "model"}, {Engine: "llamacpp", Model: "other-model"}}
	r.live = nodepolicy.Available
	r.activeRevision = "1"
	r.incarnation = h.pp
	r.runID = "proxy-run"
	r.mu.Unlock()
	h.request = nodepolicy.PrepareBoundEngineRequest{RequestID: "wait-1", ProxyRunID: "proxy-run", PolicyRevision: "1", Engine: "lmstudio", Model: "model", BudgetMs: 2000}
	base := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:get-installed":
			source, target := h.sourceRunning.Load(), h.targetRunning.Load()
			h.mu.Lock()
			if h.cp != nil {
				if h.cp.Source.Running != nil {
					source = *h.cp.Source.Running
				}
				if h.cp.Target.Running != nil {
					target = *h.cp.Target.Running
				}
			}
			h.mu.Unlock()
			return map[string]any{"engines": []map[string]any{
				{"engine": "llamacpp", "installed": true, "running": source, "healthy": source, "locked": false},
				{"engine": "lmstudio", "installed": true, "running": target, "healthy": target, "locked": false},
			}}, ""
		case "engine:models":
			models := []string{}
			if h.modelAvailable.Load() {
				models = []string{"model"}
			}
			return map[string]any{"modelsByEngine": map[string][]string{"lmstudio": models}}, ""
		case nodepolicy.MethodActivateBoundEngine:
			var in nodepolicy.ActivateBoundEngineRequest
			if err := json.Unmarshal(raw, &in); err != nil {
				return nil, err.Error()
			}
			if in.Engine != "lmstudio" || in.PolicyRevision != "1" || in.ManagerIncarnation != "manager-run" {
				return nil, "activation envelope mismatch"
			}
			h.activations.Add(1)
			h.targetRunning.Store(true)
			return map[string]bool{"success": true}, ""
		}
		return base(method, raw)
	})
	return h
}

func requireNoBoundEffects(t *testing.T, h *boundHarness) {
	t.Helper()
	if h.activations.Load() != 0 || len(switchSteps(t, h.engine)) != 0 {
		t.Fatal("refusal produced lifecycle effects")
	}
}

func waitBoundOperation(t *testing.T, b *Broker, waiters int) *boundEngineOperation {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		b.engineConfigMu.Lock()
		op := b.boundCoordinator().active
		enough := op != nil && op.waiters == waiters
		b.engineConfigMu.Unlock()
		if enough {
			return op
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("preparation waiters did not register")
	return nil
}

func TestBoundEngineReadyReadbackAndColdStart(t *testing.T) {
	test := func(name string, source, target bool, wantStarts int32) {
		t.Run(name, func(t *testing.T) {
			h := newBoundHarness(t, source, target)
			result := h.b.prepareBoundEngine(context.Background(), h.pp, h.request)
			if !result.Success || h.activations.Load() != wantStarts || len(switchSteps(t, h.engine)) != 0 {
				t.Fatalf("result=%+v starts=%d", result, h.activations.Load())
			}
		})
	}
	test("ready target bypasses startup dwell", false, true, 0)
	test("zero source starts bound target", false, false, 1)
}

func TestBoundEngineReadyPeerNeedsNoManagementGrant(t *testing.T) {
	h := newBoundHarness(t, false, true)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
	mesh := clustertrust.Open(h.b.clusterDir)
	incarnation, ok := mesh.PeerServiceIncarnation("peer")
	if !ok {
		t.Fatal("peer incarnation missing")
	}
	h.request.PeerID, h.request.PeerIncarnation = "peer", incarnation
	result := h.b.prepareBoundEngine(context.Background(), h.pp, h.request)
	if !result.Success {
		t.Fatalf("ready inference=%+v", result)
	}
	requireNoBoundEffects(t, h)
	h.targetRunning.Store(false)
	result = h.b.prepareBoundEngine(context.Background(), h.pp, h.request)
	if result.Code != -32012 || result.Reason != "management-denied" {
		t.Fatalf("cold inference=%+v", result)
	}
	requireNoBoundEffects(t, h)
}

func TestBoundEngineModelRefusalDoesNotRestart(t *testing.T) {
	h := newBoundHarness(t, false, true)
	h.modelAvailable.Store(false)
	result := h.b.prepareBoundEngine(context.Background(), h.pp, h.request)
	if result.Success || result.Reason != "model-unavailable" {
		t.Fatalf("missing model=%+v", result)
	}
	requireNoBoundEffects(t, h)
}

func TestBoundEngineAutomaticReplacement(t *testing.T) {
	h := newBoundHarness(t, true, false)
	result := h.b.prepareBoundEngine(context.Background(), h.pp, h.request)
	if !result.Success || result.SwitchRequestID == "" || h.activations.Load() != 0 {
		t.Fatalf("replacement=%+v", result)
	}
	receipt := h.receipt(t, result.SwitchRequestID)
	if !receipt.Automatic || receipt.Phase != settings.SwitchCompleted || !receipt.Source.TerminationVerified {
		t.Fatalf("receipt=%+v", receipt)
	}
	if got := switchSteps(t, h.engine); !reflect.DeepEqual(got, []string{settings.SwitchStepSourceStop, settings.SwitchStepTargetStart, settings.SwitchStepRelease}) {
		t.Fatalf("steps=%v", got)
	}
	// Completed manual and automatic replacements use the same dwell owner.
	h.mu.Lock()
	h.cp = nil
	h.mu.Unlock()
	result = h.b.prepareBoundEngine(context.Background(), h.pp, h.request)
	if result.Reason != "engine-switch-cooldown" {
		t.Fatalf("replacement dwell=%+v", result)
	}
}

func TestBoundEnginePreparationRefusals(t *testing.T) {
	test := func(name string, change func(*testing.T, *boundHarness), reason string) {
		t.Run(name, func(t *testing.T) {
			h := newBoundHarness(t, true, false)
			change(t, h)
			result := h.b.prepareBoundEngine(context.Background(), h.pp, h.request)
			if result.Success || result.Reason != reason {
				t.Fatalf("refusal=%+v want %s", result, reason)
			}
			requireNoBoundEffects(t, h)
		})
	}
	test("stale policy", func(_ *testing.T, h *boundHarness) { h.request.PolicyRevision = "0" }, "stale-revision")
	test("wrong engine", func(_ *testing.T, h *boundHarness) { h.request.Engine = "llamacpp" }, "model-unavailable")
	test("startup dwell", func(_ *testing.T, h *boundHarness) { h.b.startedAt = time.Now() }, "engine-switch-cooldown")
	test("protected source", func(_ *testing.T, h *boundHarness) {
		r := h.b.nodePolicyRuntime()
		r.mu.Lock()
		r.current.ResidentModels = []nodepolicy.Role{{Engine: "llamacpp", Model: "protected"}}
		r.mu.Unlock()
	}, "no-fit")
	test("paused", func(_ *testing.T, h *boundHarness) {
		r := h.b.nodePolicyRuntime()
		r.mu.Lock()
		r.live = nodepolicy.Paused
		r.mu.Unlock()
	}, "paused")
	test("unknown run", func(_ *testing.T, h *boundHarness) { h.request.ProxyRunID = "previous" }, "unavailable")
	test("manager recovering", func(_ *testing.T, h *boundHarness) {
		h.b.workersMu.Lock()
		h.b.engineMgrRecovering = true
		h.b.workersMu.Unlock()
	}, "unavailable")
}

func TestBoundEngineCoalescesWithoutExtendingDeadline(t *testing.T) {
	h := newBoundHarness(t, true, false)
	h.active.Store(1)
	first := make(chan nodepolicy.BoundEnginePrepared, 1)
	go func() { first <- h.b.prepareBoundEngine(context.Background(), h.pp, h.request) }()
	op := waitBoundOperation(t, h.b, 1)
	secondRequest := h.request
	secondRequest.RequestID = "wait-2"
	secondRequest.BudgetMs = 90000
	second := make(chan nodepolicy.BoundEnginePrepared, 1)
	go func() { second <- h.b.prepareBoundEngine(context.Background(), h.pp, secondRequest) }()
	if got := waitBoundOperation(t, h.b, 2); got != op || !got.deadline.Equal(op.deadline) {
		t.Fatal("same target did not coalesce")
	}
	competing := h.request
	competing.Engine, competing.Model, competing.RequestID = "llamacpp", "other-model", "competing"
	if result := h.b.prepareBoundEngine(context.Background(), h.pp, competing); result.Reason != "engine-switch-busy" {
		t.Fatalf("competing=%+v", result)
	}
	h.active.Store(0)
	for _, done := range []chan nodepolicy.BoundEnginePrepared{first, second} {
		select {
		case result := <-done:
			if !result.Success {
				t.Fatalf("coalesced=%+v", result)
			}
		case <-time.After(4 * time.Second):
			t.Fatal("preparation did not finish")
		}
	}
	if len(switchSteps(t, h.engine)) != 3 {
		t.Fatal("coalesced demand duplicated replacement")
	}
}

func TestBoundEngineLastWaiterCancelledBeforeAcceptance(t *testing.T) {
	h := newBoundHarness(t, true, false)
	h.b.startedAt = time.Now().Add(-boundEngineDwell + 500*time.Millisecond)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan nodepolicy.BoundEnginePrepared, 1)
	go func() { done <- h.b.prepareBoundEngine(ctx, h.pp, h.request) }()
	op := waitBoundOperation(t, h.b, 1)
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("waiter cancellation blocked")
	}
	select {
	case <-op.done:
	case <-time.After(time.Second):
		t.Fatal("unaccepted preparation survived last waiter")
	}
	requireNoBoundEffects(t, h)
}

func TestBoundEnginePolicyBusyBeforePersistence(t *testing.T) {
	h := newBoundHarness(t, true, false)
	h.active.Store(1)
	done := make(chan nodepolicy.BoundEnginePrepared, 1)
	go func() { done <- h.b.prepareBoundEngine(context.Background(), h.pp, h.request) }()
	waitBoundOperation(t, h.b, 1)
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	before := r.current
	path := filepath.Join(t.TempDir(), "node-policy.json")
	r.path = path
	r.mu.Unlock()
	changed := before
	changed.EngineBindings = nil
	raw, err := json.Marshal(changed)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.b.setLocalPolicy(raw, json.RawMessage(`"1"`)); err == nil {
		t.Fatal("reserved binding edit saved")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("busy policy touched disk: %v", err)
	}
	r.mu.Lock()
	after := r.current
	r.mu.Unlock()
	if !reflect.DeepEqual(before, after) {
		t.Fatal("busy edit changed policy")
	}
	h.b.engineConfigMu.Lock()
	unrelated := before
	unrelated.Admission.QueueTimeoutSeconds++
	err = h.b.boundPolicyChangeLocked(before, unrelated)
	h.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatalf("unrelated policy edit blocked: %v", err)
	}
	h.active.Store(0)
	select {
	case result := <-done:
		if !result.Success {
			t.Fatalf("preparation=%+v", result)
		}
	case <-time.After(4 * time.Second):
		t.Fatal("preparation did not finish")
	}
}

func TestBoundEngineSourceReplacementBeforeAcceptance(t *testing.T) {
	h := newBoundHarness(t, true, false)
	base := h.engine.answer
	reached, resume := make(chan struct{}), make(chan struct{})
	var once sync.Once
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:get-installed" {
			once.Do(func() { close(reached); <-resume })
		}
		return base(method, raw)
	})
	done := make(chan nodepolicy.BoundEnginePrepared, 1)
	go func() { done <- h.b.prepareBoundEngine(context.Background(), h.pp, h.request) }()
	select {
	case <-reached:
	case <-time.After(time.Second):
		t.Fatal("readback barrier not reached")
	}
	replacement := &proxyProcess{done: make(chan struct{})}
	h.b.workersMu.Lock()
	h.b.proxyProc = replacement
	h.b.workersMu.Unlock()
	close(resume)
	select {
	case result := <-done:
		if result.Success {
			t.Fatal("replaced source accepted")
		}
	case <-time.After(time.Second):
		t.Fatal("replaced preparation blocked")
	}
	requireNoBoundEffects(t, h)
	if h.b.boundProxyCurrent(h.pp, "proxy-run") {
		t.Fatal("reply eligible for retired proxy")
	}
	if h.b.boundProxyCurrent(replacement, "proxy-run") {
		t.Fatal("old reply eligible for replacement")
	}
}

func TestBoundEnginePeerAuthorityChangesBeforeAcceptance(t *testing.T) {
	test := func(name string, change func(*testing.T, *boundHarness)) {
		t.Run(name, func(t *testing.T) {
			h := newBoundHarness(t, false, false)
			clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
			grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", true)
			mesh := clustertrust.Open(h.b.clusterDir)
			incarnation, ok := mesh.PeerServiceIncarnation("peer")
			if !ok {
				t.Fatal("missing incarnation")
			}
			h.request.PeerID, h.request.PeerIncarnation = "peer", incarnation
			base := h.engine.answer
			reached, resume := make(chan struct{}), make(chan struct{})
			var once sync.Once
			h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == "engine:get-installed" {
					once.Do(func() { close(reached); <-resume })
				}
				return base(method, raw)
			})
			done := make(chan nodepolicy.BoundEnginePrepared, 1)
			go func() { done <- h.b.prepareBoundEngine(context.Background(), h.pp, h.request) }()
			select {
			case <-reached:
			case <-time.After(time.Second):
				t.Fatal("authority barrier not reached")
			}
			change(t, h)
			close(resume)
			select {
			case result := <-done:
				if result.Code != -32012 || result.Success {
					t.Fatalf("authority refusal=%+v", result)
				}
			case <-time.After(time.Second):
				t.Fatal("authority refusal blocked")
			}
			requireNoBoundEffects(t, h)
		})
	}
	test("management revoked", func(t *testing.T, h *boundHarness) {
		grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
	})
	test("peer removed", func(t *testing.T, h *boundHarness) { clustertrusttest.RemovePeerPin(t, h.b.clusterDir, "peer") })
	test("peer readmitted with new grant", func(t *testing.T, h *boundHarness) {
		pinPath := filepath.Join(h.b.clusterDir, "trusted", "peer.json")
		var pin map[string]json.RawMessage
		rmReadJSON(t, pinPath, &pin)
		clustertrusttest.RemovePeerPin(t, h.b.clusterDir, "peer")
		pin["admissionEpoch"] = json.RawMessage(`2`)
		rmWriteJSON(t, pinPath, pin)
		rmWriteJSON(t, filepath.Join(h.b.clusterDir, "members.json"), []map[string]any{{"nodeUuid": "peer", "state": "member", "clusterId": "cluster", "admissionEpoch": 2}})
		grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", true)
		incarnation, ok := clustertrust.Open(h.b.clusterDir).PeerServiceIncarnation("peer")
		if !ok || incarnation == h.request.PeerIncarnation {
			t.Fatal("fixture did not replace relationship")
		}
	})
}

func TestBoundEngineRefusesLockedAndAuthUnknown(t *testing.T) {
	test := func(name string, locked, authFailure bool, wantReason string) {
		t.Run(name, func(t *testing.T) {
			h := newBoundHarness(t, false, true)
			base := h.engine.answer
			h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if locked && method == "engine:get-installed" {
					return map[string]any{"engines": []map[string]any{{"engine": "lmstudio", "installed": true, "running": true, "healthy": true, "locked": true}}}, ""
				}
				if authFailure && method == "engine:models" {
					return nil, "native authentication secret must not escape"
				}
				return base(method, raw)
			})
			result := h.b.prepareBoundEngine(context.Background(), h.pp, h.request)
			if result.Success || result.Reason != wantReason {
				t.Fatalf("refusal=%+v", result)
			}
			if result.Message != "Bound engine preparation refused: "+wantReason+"." {
				t.Fatalf("unbounded native detail escaped: %+v", result)
			}
			requireNoBoundEffects(t, h)
		})
	}
	test("locked target", true, false, "engine-locked")
	test("model authentication unavailable", false, true, "model-unavailable")
}

func TestBoundEngineWorkerMethodsAreNotPublic(t *testing.T) {
	h := newBoundHarness(t, false, false)
	for _, method := range []string{nodepolicy.MethodPrepareBoundEngine, nodepolicy.MethodCancelBoundEngine, nodepolicy.NotifyBoundEnginePrepared, nodepolicy.MethodActivateBoundEngine} {
		t.Run(method, func(t *testing.T) {
			before := len(h.engine.snapshot())
			id := json.RawMessage(`"private"`)
			h.b.handleMessage(&Message{Method: method, ID: &id, Params: settingsJSON(h.request)})
			if len(h.engine.snapshot()) != before {
				t.Fatal("public request reached manager")
			}
			frames := h.output.frames()
			last := frames[len(frames)-1]
			if last.Error == nil || last.Error.Code != -32601 {
				t.Fatalf("public response=%+v", last)
			}
		})
	}
}

func TestBoundEngineBrokerStopPreemptsColdReadiness(t *testing.T) {
	h := newBoundHarness(t, false, false)
	base := h.engine.answer
	started := make(chan nodepolicy.ActivateBoundEngineRequest, 1)
	var activationID string
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case nodepolicy.MethodActivateBoundEngine:
			var in nodepolicy.ActivateBoundEngineRequest
			if err := json.Unmarshal(raw, &in); err != nil {
				return nil, err.Error()
			}
			activationID = in.OperationID
			started <- in
			return map[string]bool{"success": true}, holdSuccessReply
		case "engine:stop":
			var in struct {
				Engine      string `json:"engine"`
				OperationID string `json:"boundOperationId"`
			}
			if err := json.Unmarshal(raw, &in); err != nil {
				return nil, err.Error()
			}
			if in.OperationID != activationID || in.Engine != "lmstudio" {
				return nil, "cold Off was not scoped"
			}
			h.engine.releaseHeld()
			return map[string]bool{"applied": true}, ""
		case "engine:status":
			return map[string]any{"engine": "lmstudio", "running": false, "installed": true, "healthy": false}, ""
		}
		return base(method, raw)
	})
	preparation := make(chan nodepolicy.BoundEnginePrepared, 1)
	go func() { preparation <- h.b.prepareBoundEngine(context.Background(), h.pp, h.request) }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("cold readiness wait not reached")
	}
	id := json.RawMessage(`"cold-Off"`)
	h.b.handleMessage(&Message{Method: "engine:stop", ID: &id, Params: settingsJSON(map[string]string{"engine": "lmstudio"})})
	select {
	case result := <-preparation:
		if result.Success {
			t.Fatal("Off preparation succeeded")
		}
	case <-time.After(time.Second):
		t.Fatal("broker Stop waited behind cold startup")
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		for _, frame := range h.output.frames() {
			if frame.ID != nil && string(*frame.ID) == `"cold-Off"` {
				if frame.Error != nil {
					t.Fatalf("Off response=%+v", frame.Error)
				}
				return
			}
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("broker did not return actual Off result")
}

func TestBoundEngineUnsettledReleaseProtectsPolicy(t *testing.T) {
	h := newBoundHarness(t, true, false)
	base := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == settings.SwitchStepMethod {
			var in settings.SwitchStepRequest
			if err := json.Unmarshal(raw, &in); err != nil {
				return nil, err.Error()
			}
			if in.Step == settings.SwitchStepRelease {
				return nil, "release failed"
			}
		}
		return base(method, raw)
	})
	result := h.b.prepareBoundEngine(context.Background(), h.pp, h.request)
	if result.Success || result.SwitchRequestID == "" {
		t.Fatalf("unsettled preparation=%+v", result)
	}
	h.waitIdle(t)
	h.b.engineConfigMu.Lock()
	defer h.b.engineConfigMu.Unlock()
	c := h.b.switchCoordinator()
	if c.store.journal.AutomaticSettled[result.SwitchRequestID] != 0 {
		t.Fatal("failed release marked settled")
	}
	before := h.b.nodePolicy.current
	after := before
	after.EngineBindings = nil
	if err := h.b.boundPolicyChangeLocked(before, after); err == nil {
		t.Fatal("unsettled terminal automatic reservation allowed binding edit")
	}
}

func TestBoundEngineUnknownColdAcceptanceRetainsOriginalID(t *testing.T) {
	h := newBoundHarness(t, false, false)
	h.request.BudgetMs = 100
	base := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodActivateBoundEngine {
			h.activations.Add(1)
			return nil, holdReply
		}
		if method == "engine:status" {
			return map[string]any{"engine": "lmstudio", "running": false, "installed": true, "healthy": false}, ""
		}
		return base(method, raw)
	})
	result := h.b.prepareBoundEngine(context.Background(), h.pp, h.request)
	if result.Success {
		t.Fatal("unknown activation succeeded")
	}
	h.b.engineConfigMu.Lock()
	op := h.b.boundCoordinator().active
	h.b.engineConfigMu.Unlock()
	if op == nil {
		t.Fatal("unknown accepted operation forgotten")
	}
	select {
	case <-op.done:
	case <-time.After(time.Second):
		t.Fatal("unknown outcome not bounded")
	}
	h.b.engineConfigMu.Lock()
	uncertain, id := op.uncertain, op.operationID
	h.b.engineConfigMu.Unlock()
	if !uncertain || id == "" {
		t.Fatal("unknown acceptance lost original ID")
	}
	h.request.BudgetMs = 1000
	h.request.RequestID = "retry"
	result = h.b.prepareBoundEngine(context.Background(), h.pp, h.request)
	if result.Success || h.activations.Load() != 1 {
		t.Fatalf("unknown outcome resubmitted: %+v activations=%d", result, h.activations.Load())
	}
	idRaw := json.RawMessage(`"resolve-Off"`)
	h.b.handleMessage(&Message{Method: "engine:stop", ID: &idRaw, Params: settingsJSON(map[string]string{"engine": "lmstudio"})})
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		for _, frame := range h.output.frames() {
			if frame.ID != nil && string(*frame.ID) == `"resolve-Off"` {
				if frame.Error != nil {
					t.Fatalf("Off recovery=%+v", frame.Error)
				}
				h.b.engineConfigMu.Lock()
				active, fence := h.b.boundCoordinator().active, h.b.switchCoordinator().recoveryErr
				h.b.engineConfigMu.Unlock()
				if active != nil || fence != nil {
					t.Fatal("verified Off retained cold uncertainty")
				}
				return
			}
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("verified Off did not resolve unknown activation")
}

func TestBoundEngineRetiredManagerPairCannotReply(t *testing.T) {
	h := newBoundHarness(t, false, true)
	base := h.engine.answer
	reached, resume := make(chan struct{}), make(chan struct{})
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:models" {
			close(reached)
			<-resume
		}
		return base(method, raw)
	})
	if !h.b.handleBoundEngineNotification(h.pp, nodepolicy.MethodPrepareBoundEngine, settingsJSON(h.request)) {
		t.Fatal("preparation notification was not owned")
	}
	select {
	case <-reached:
	case <-time.After(time.Second):
		t.Fatal("reply barrier not reached")
	}
	// Even the same worker pointer cannot lend a replacement source pairing
	// to a callback that retained the original manager source.
	bindRemoteManagementTestSource(h.b)
	close(resume)
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		c := h.b.boundCoordinator()
		c.mu.Lock()
		pending := len(c.waiters)
		c.mu.Unlock()
		if pending == 0 {
			for _, call := range h.proxy.snapshot() {
				if call.method == nodepolicy.NotifyBoundEnginePrepared {
					t.Fatal("retired manager pair sent preparation reply")
				}
			}
			requireNoBoundEffects(t, h)
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("retired callback did not finish")
}

func TestBoundEngineSameEngineCoalescesWithPerModelReadback(t *testing.T) {
	h := newBoundHarness(t, true, false)
	h.active.Store(1)
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.EngineBindings = append(r.current.EngineBindings, nodepolicy.Role{Engine: "lmstudio", Model: "missing-model"})
	r.mu.Unlock()
	first := make(chan nodepolicy.BoundEnginePrepared, 1)
	go func() { first <- h.b.prepareBoundEngine(context.Background(), h.pp, h.request) }()
	op := waitBoundOperation(t, h.b, 1)
	secondRequest := h.request
	secondRequest.RequestID = "second-model"
	secondRequest.Model = "missing-model"
	second := make(chan nodepolicy.BoundEnginePrepared, 1)
	go func() { second <- h.b.prepareBoundEngine(context.Background(), h.pp, secondRequest) }()
	if got := waitBoundOperation(t, h.b, 2); got != op {
		t.Fatal("same engine did not share preparation")
	}
	h.active.Store(0)
	select {
	case result := <-first:
		if !result.Success {
			t.Fatalf("available model=%+v", result)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("first waiter blocked")
	}
	select {
	case result := <-second:
		if result.Success || result.Reason != "model-unavailable" {
			t.Fatalf("missing model=%+v", result)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("second waiter blocked")
	}
	if len(switchSteps(t, h.engine)) != 3 {
		t.Fatal("same engine demand duplicated lifecycle effects")
	}
}

func TestBoundEngineSourceOffCancelsDwellBeforeAcceptance(t *testing.T) {
	h := newBoundHarness(t, true, false)
	h.request.BudgetMs = 10000
	h.b.startedAt = time.Now().Add(-boundEngineDwell + 5*time.Second)
	done := make(chan nodepolicy.BoundEnginePrepared, 1)
	go func() { done <- h.b.prepareBoundEngine(context.Background(), h.pp, h.request) }()
	op := waitBoundOperation(t, h.b, 1)
	deadline := time.Now().Add(time.Second)
	for {
		h.b.engineConfigMu.Lock()
		source := op.sourceEngine
		h.b.engineConfigMu.Unlock()
		if source == "llamacpp" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("source was not observed before dwell")
		}
		time.Sleep(time.Millisecond)
	}
	id := json.RawMessage(`"source-Off"`)
	h.b.handleMessage(&Message{Method: "engine:stop", ID: &id, Params: settingsJSON(map[string]string{"engine": "llamacpp"})})
	select {
	case result := <-done:
		if result.Success {
			t.Fatal("source Off did not veto pending preparation")
		}
	case <-time.After(time.Second):
		t.Fatal("source Off did not cancel dwell")
	}
	select {
	case <-op.done:
	case <-time.After(time.Second):
		t.Fatal("cancelled source preparation retained operation")
	}
	requireNoBoundEffects(t, h)
}

func TestBoundEngineFailedColdOffRetainsPolicyReservation(t *testing.T) {
	h := newBoundHarness(t, false, false)
	h.request.BudgetMs = 100
	base := h.engine.answer
	started := make(chan struct{}, 1)
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodActivateBoundEngine {
			started <- struct{}{}
			return map[string]bool{"success": true}, holdSuccessReply
		}
		if method == "engine:stop" {
			return nil, "stop failed"
		}
		return base(method, raw)
	})
	done := make(chan nodepolicy.BoundEnginePrepared, 1)
	go func() { done <- h.b.prepareBoundEngine(context.Background(), h.pp, h.request) }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("cold activation not submitted")
	}
	id := json.RawMessage(`"failed-Off"`)
	h.b.handleMessage(&Message{Method: "engine:stop", ID: &id, Params: settingsJSON(map[string]string{"engine": "lmstudio"})})
	select {
	case result := <-done:
		if result.Success {
			t.Fatal("failed Off preparation succeeded")
		}
	case <-time.After(time.Second):
		t.Fatal("failed Off did not settle its waiter")
	}
	h.b.engineConfigMu.Lock()
	defer h.b.engineConfigMu.Unlock()
	op := h.b.boundCoordinator().active
	if op == nil || !op.uncertain || op.offError == nil {
		t.Fatal("failed cold cleanup lost its reservation")
	}
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	before := r.current
	r.mu.Unlock()
	after := before
	after.EngineBindings = nil
	if err := h.b.boundPolicyChangeLocked(before, after); err == nil {
		t.Fatal("failed cold cleanup allowed binding edit")
	}
}
