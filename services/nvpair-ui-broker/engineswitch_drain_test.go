// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

func TestSwitchDrainActiveSourceMustReachFreshZero(t *testing.T) {
	h := newSwitchHarness(t)
	h.active.Store(2)
	freshSnapshot := make(chan struct{})
	var snapshotOnce sync.Once
	base := h.proxy.answer
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, errText := base(method, raw)
		if method == nodepolicy.MethodAdmissionSnapshot {
			// Signal after reading the active count, so the first fresh reply
			// contains the active source even if the test then clears it.
			snapshotOnce.Do(func() { close(freshSnapshot) })
		}
		return result, errText
	})
	p := h.request(t, "active-drain")
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	snapshotCtx, cancelSnapshot := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancelSnapshot()
	select {
	case <-freshSnapshot:
	case <-snapshotCtx.Done():
		t.Fatal("wait for fresh admission snapshot")
	}
	if steps := switchSteps(t, h.engine); len(steps) != 0 {
		t.Fatalf("stopped active source: %v", steps)
	}
	// A stale asynchronous zero never satisfies the drain.
	h.b.observeAdmission(nodepolicy.AdmissionState{ActiveByEngine: map[string]int{}})
	if steps := switchSteps(t, h.engine); len(steps) != 0 {
		t.Fatalf("stale zero stopped source: %v", steps)
	}
	h.active.Store(0)
	if r := h.receipt(t, p.RequestID); r.Phase != settings.SwitchCompleted {
		t.Fatalf("receipt = %+v", r)
	}
	var order []string
	for _, call := range h.proxy.snapshot() {
		order = append(order, call.method)
	}
	// The acknowledged manager startup identity precedes the switch's two
	// identity withdrawals and the original drain sequence.
	wantPrefix := []string{nodepolicy.MethodSetModelIdentity, nodepolicy.MethodSetModelIdentity, nodepolicy.MethodSetModelIdentity, "llamacpp:node/set-local-backend", nodepolicy.MethodSetEngineDrain, nodepolicy.MethodInvalidateEngine, nodepolicy.MethodInvalidateEngine, nodepolicy.MethodAdmissionSnapshot}
	if len(order) < len(wantPrefix) || !reflect.DeepEqual(order[:len(wantPrefix)], wantPrefix) {
		t.Fatalf("drain order = %v", order)
	}
}

func TestSwitchDrainRefusalsNeverStopSource(t *testing.T) {
	for _, scenario := range []string{"no proxy", "missing admission", "wrong snapshot incarnation", "wrong invalidation incarnation", "invalidation ack loss", "drain ack loss", "snapshot RPC failure"} {
		t.Run(scenario, func(t *testing.T) {
			h := newSwitchHarness(t)
			p := h.request(t, "failed-drain")
			base := h.proxy.answer
			h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				switch {
				case scenario == "missing admission" && method == nodepolicy.MethodAdmissionSnapshot:
					return map[string]string{"runId": "proxy-run"}, ""
				case scenario == "wrong snapshot incarnation" && method == nodepolicy.MethodAdmissionSnapshot:
					return nodepolicy.AdmissionSnapshotResult{RunID: "other-run", AdmissionState: nodepolicy.AdmissionState{ActiveByEngine: map[string]int{}}}, ""
				case scenario == "wrong invalidation incarnation" && method == nodepolicy.MethodInvalidateEngine:
					return nodepolicy.InvalidateEngineResult{OK: true, RunID: "other-run"}, ""
				case scenario == "invalidation ack loss" && method == nodepolicy.MethodInvalidateEngine:
					return map[string]bool{"ok": true}, ""
				case scenario == "drain ack loss" && method == nodepolicy.MethodSetEngineDrain:
					return map[string]bool{"ok": false}, ""
				case scenario == "snapshot RPC failure" && method == nodepolicy.MethodAdmissionSnapshot:
					return nil, "snapshot unavailable"
				}
				return base(method, raw)
			})
			if scenario == "no proxy" {
				// Keep facade status for validation, remove only the process handle.
				h.b.setProxyProcessHandle(nil)
			}
			if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err != nil {
				t.Fatal(err)
			}
			r := h.receipt(t, p.RequestID)
			if r.Phase != settings.SwitchFailedDrain || r.Error.Kind != settings.SwitchErrorDrain {
				t.Fatalf("receipt = %+v", r)
			}
			for _, step := range switchSteps(t, h.engine) {
				if step != settings.SwitchStepRelease {
					t.Fatalf("failed drain issued %s", step)
				}
			}
		})
	}
}

func TestSwitchDrainPushSlotAndDeadlineAreBounded(t *testing.T) {
	for _, scenario := range []string{"push slot contention", "active source deadline", "missing acknowledgement deadline"} {
		t.Run(scenario, func(t *testing.T) {
			h := newSwitchHarness(t)
			r := h.b.nodePolicyRuntime()
			if scenario == "push slot contention" {
				if !r.acquirePush(context.Background()) {
					t.Fatal("acquire push slot")
				}
				defer r.releasePush()
			}
			if scenario == "active source deadline" {
				h.active.Store(1)
			}
			if scenario == "missing acknowledgement deadline" {
				base := h.proxy.answer
				h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
					if method == nodepolicy.MethodInvalidateEngine {
						return nil, holdReply
					}
					return base(method, raw)
				})
				defer h.proxy.releaseHeld()
			}
			ctx, cancel := context.WithTimeout(context.Background(), 35*time.Millisecond)
			defer cancel()
			started := time.Now()
			err := h.b.drainEngineForSwitch(ctx, settings.SwitchPrepareRequest{OperationID: "bounded-drain", SourceEngine: "llamacpp", TargetEngine: "lmstudio"})
			if err == nil || time.Since(started) > time.Second {
				t.Fatalf("drain bound: error %v, elapsed %v", err, time.Since(started))
			}
			if len(switchSteps(t, h.engine)) > 0 {
				t.Fatal("bounded drain issued lifecycle step")
			}
		})
	}
}

func TestSwitchReplayInvalidatesBeforeFacadeAndFailsClosed(t *testing.T) {
	h := newSwitchHarness(t)
	r := switchStoreReceipt("replay-invalidations")
	r.Request.TargetEngine = "lmstudio"
	r.PayloadHash = settings.SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
	switchStoreAccept(t, h.b.engineSwitch.store, r)
	for _, phase := range []string{settings.SwitchValidating, settings.SwitchDraining} {
		r = switchStoreNext(r, phase)
		if err := h.b.engineSwitch.store.Advance(r.RequestID, r); err != nil {
			t.Fatal(err)
		}
	}
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case nodepolicy.MethodSetPolicy:
			var policy nodepolicy.SetPolicyParams
			if err := json.Unmarshal(raw, &policy); err != nil {
				return nil, err.Error()
			}
			return nodepolicy.PolicyAck{OriginNodeID: policy.OriginNodeID, RunID: "proxy-run", Revision: policy.Policy.Revision}, ""
		case nodepolicy.MethodInvalidateEngine:
			return nodepolicy.InvalidateEngineResult{OK: true, RunID: "proxy-run"}, ""
		default:
			return map[string]bool{"ok": true}, ""
		}
	})
	if !h.b.replayPolicyToProxy(context.Background(), h.pp) {
		t.Fatal("pending invalidations did not replay")
	}
	var invalidations []nodepolicy.InvalidateEngineParams
	for _, call := range h.proxy.snapshot() {
		if call.method == nodepolicy.MethodInvalidateEngine {
			var p nodepolicy.InvalidateEngineParams
			if err := json.Unmarshal(call.params, &p); err != nil {
				t.Fatal(err)
			}
			invalidations = append(invalidations, p)
		}
	}
	if len(invalidations) != 2 || invalidations[0].Engine != "llamacpp" || invalidations[1].Engine != "lmstudio" || invalidations[0].OperationID != r.RequestID {
		t.Fatalf("replay = %+v", invalidations)
	}
	base := h.proxy.answer
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodInvalidateEngine {
			return nil, "invalidation refused"
		}
		return base(method, raw)
	})
	if h.b.replayPolicyToProxy(context.Background(), h.pp) {
		t.Fatal("facade bring-up was allowed after failed replay")
	}
}

func TestSwitchDrainWrongRunBetweenInvalidationAcknowledgements(t *testing.T) {
	h := newSwitchHarness(t)
	var count atomic.Int32
	base := h.proxy.answer
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodInvalidateEngine {
			run := "proxy-run"
			if count.Add(1) == 2 {
				run = "restarted"
			}
			return nodepolicy.InvalidateEngineResult{OK: true, RunID: run}, ""
		}
		return base(method, raw)
	})
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := h.b.drainEngineForSwitch(ctx, settings.SwitchPrepareRequest{OperationID: "restart", SourceEngine: "llamacpp", TargetEngine: "lmstudio"}); err == nil {
		t.Fatal("accepted evidence from two proxy runs")
	}
}

type nodeDrainHarness struct {
	b       *Broker
	pp      *proxyProcess
	proxy   *fakeWorker
	engines []string
	ctx     context.Context
}

func newNodeDrainHarness(t *testing.T) *nodeDrainHarness {
	t.Helper()
	isolateOllamaHostTestConfig(t)
	b, _ := newPolicyBroker(t)
	// Use the existing catalog owner, including descriptor engines outside the
	// built-in table. Port planning is synthetic and never opens a listener.
	b.setDynamicEngines([]engines.Descriptor{
		customEngineDescriptor(), testDescriptor("strata", engines.ProtocolOpenAIChat),
	}, func(int) bool { return true })
	pp, proxy := attachFakeProxy(t, b)
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	t.Cleanup(cancel)
	h := &nodeDrainHarness{b: b, pp: pp, proxy: proxy, ctx: ctx}
	for _, profile := range b.frontedEngineProfiles() {
		h.engines = append(h.engines, profile.Name)
		b.setEngineProxyHandle(profile, pp)
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.incarnation, r.runID = pp, "node-drain-run"
	r.mu.Unlock()
	proxy.setAnswer(h.answer)
	return h
}

func (h *nodeDrainHarness) idle() nodepolicy.AdmissionSnapshotResult {
	return nodepolicy.AdmissionSnapshotResult{
		RunID:          "node-drain-run",
		AdmissionState: nodepolicy.AdmissionState{ActiveByEngine: map[string]int{}},
	}
}

func (h *nodeDrainHarness) answer(method string, raw json.RawMessage) (any, string) {
	switch method {
	case nodepolicy.MethodSetEngineDrain:
		var p nodepolicy.SetEngineDrainParams
		if err := json.Unmarshal(raw, &p); err != nil {
			return nil, err.Error()
		}
		for _, engine := range h.engines {
			if p.Engine == engine {
				return map[string]bool{"ok": true}, ""
			}
		}
		return nil, "unexpected drain engine: " + p.Engine
	case nodepolicy.MethodAdmissionSnapshot:
		var p nodepolicy.AdmissionSnapshotParams
		if err := json.Unmarshal(raw, &p); err != nil {
			return nil, err.Error()
		}
		return h.idle(), ""
	default:
		return nil, "unexpected node drain method: " + method
	}
}

func (h *nodeDrainHarness) setPreexisting(engine string) {
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.drains[engine] = true
	r.mu.Unlock()
}

func (h *nodeDrainHarness) requireDrains(t *testing.T, engines []string) {
	t.Helper()
	want := make(map[string]bool, len(engines))
	for _, engine := range engines {
		want[engine] = true
	}
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	got := make(map[string]bool, len(r.drains))
	for engine, draining := range r.drains {
		got[engine] = draining
	}
	r.mu.Unlock()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("registry drains = %v, want %v", got, want)
	}
}

func (h *nodeDrainHarness) drainCalls(t *testing.T, drain bool) map[string]int {
	t.Helper()
	got := make(map[string]int)
	for _, call := range h.proxy.snapshot() {
		if call.method != nodepolicy.MethodSetEngineDrain {
			continue
		}
		if call.notify {
			t.Fatal("drain sent without an acknowledgement request")
		}
		var p nodepolicy.SetEngineDrainParams
		if err := json.Unmarshal(call.params, &p); err != nil {
			t.Fatal(err)
		}
		if p.Drain == drain {
			got[p.Engine]++
		}
	}
	return got
}

func (h *nodeDrainHarness) release(t *testing.T, lease *nodeDrainLease) {
	t.Helper()
	if lease == nil {
		t.Fatal("successful node drain returned no release lease")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := lease.release(ctx); err != nil {
		t.Fatalf("release node drain: %v", err)
	}
}

type nodeDrainTestResult struct {
	lease *nodeDrainLease
	err   error
}

func (h *nodeDrainHarness) start(ctx context.Context) <-chan nodeDrainTestResult {
	done := make(chan nodeDrainTestResult, 1)
	go func() {
		lease, err := h.b.drainNode(ctx)
		done <- nodeDrainTestResult{lease: lease, err: err}
	}()
	return done
}

func awaitNodeDrain(t *testing.T, ctx context.Context, done <-chan nodeDrainTestResult) nodeDrainTestResult {
	t.Helper()
	select {
	case result := <-done:
		return result
	case <-ctx.Done():
		t.Fatal("node drain did not return within the test context")
		return nodeDrainTestResult{}
	}
}

func awaitNodeDrainBarrier(t *testing.T, h *nodeDrainHarness, reached <-chan struct{}, done <-chan nodeDrainTestResult) {
	t.Helper()
	select {
	case <-reached:
	case result := <-done:
		t.Fatalf("node drain returned before the controlled proxy barrier: lease %v, error %v", result.lease, result.err)
	case <-h.ctx.Done():
		t.Fatal("node drain never reached the controlled proxy barrier")
	}
}

func TestNodeDrainAcknowledgesWholeDynamicCatalogBeforeFreshIdle(t *testing.T) {
	h := newNodeDrainHarness(t)
	// A cached zero cannot substitute for a new proxy request.
	h.b.observeAdmission(h.idle().AdmissionState)
	lease, err := h.b.drainNode(h.ctx)
	if err != nil {
		t.Fatalf("idle node acquisition: %v", err)
	}
	h.requireDrains(t, h.engines)
	want := make(map[string]int, len(h.engines))
	for _, engine := range h.engines {
		want[engine] = 1
	}
	if got := h.drainCalls(t, true); !reflect.DeepEqual(got, want) {
		t.Fatalf("acknowledged engines = %v, want complete current catalog %v", got, want)
	}
	seen := make(map[string]bool)
	snapshots := 0
	for _, call := range h.proxy.snapshot() {
		switch call.method {
		case nodepolicy.MethodSetEngineDrain:
			var p nodepolicy.SetEngineDrainParams
			if err := json.Unmarshal(call.params, &p); err != nil {
				t.Fatal(err)
			}
			seen[p.Engine] = p.Drain
		case nodepolicy.MethodAdmissionSnapshot:
			snapshots++
			if len(seen) != len(want) {
				t.Fatalf("snapshot preceded every engine acknowledgement: %v", seen)
			}
		default:
			t.Fatalf("node drain issued an unrelated effect: %s", call.method)
		}
	}
	if snapshots == 0 {
		t.Fatal("cached admission notification substituted for fresh idle evidence")
	}
	h.release(t, lease)
	h.requireDrains(t, nil)
	if got := h.drainCalls(t, false); !reflect.DeepEqual(got, want) {
		t.Fatalf("released engines = %v, want %v", got, want)
	}
}

func TestNodeDrainRetainsSettingsExclusionUntilCleanupSucceeds(t *testing.T) {
	h := newNodeDrainHarness(t)
	lease, err := h.b.drainNode(h.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer h.release(t, lease)
	// Settings holds this mutex before registering/clearing its drain; switch
	// admission uses TryLock on the same owner. Neither may overlap this lease.
	if h.b.settingsApplyMu.TryLock() {
		h.b.settingsApplyMu.Unlock()
		t.Fatal("node drain admitted a competing settings/switch owner")
	}
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetEngineDrain {
			var p nodepolicy.SetEngineDrainParams
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if !p.Drain {
				return nil, "controlled cleanup refusal"
			}
		}
		return h.answer(method, raw)
	})
	if err := lease.release(h.ctx); err == nil {
		t.Fatal("cleanup refusal succeeded")
	}
	if h.b.settingsApplyMu.TryLock() {
		h.b.settingsApplyMu.Unlock()
		t.Fatal("incomplete cleanup released settings/switch exclusion")
	}
	h.proxy.setAnswer(h.answer)
	h.release(t, lease)
	if !h.b.settingsApplyMu.TryLock() {
		t.Fatal("completed cleanup retained settings/switch exclusion")
	}
	h.b.settingsApplyMu.Unlock()
}

func TestNodeDrainRefusesCompetingOwnerWithoutEffects(t *testing.T) {
	h := newNodeDrainHarness(t)
	h.b.settingsApplyMu.Lock()
	lease, err := h.b.drainNode(h.ctx)
	h.b.settingsApplyMu.Unlock()
	if err == nil || lease != nil {
		t.Fatalf("competing settings owner: lease %v, error %v", lease, err)
	}
	h.requireDrains(t, nil)
	if calls := h.proxy.snapshot(); len(calls) != 0 {
		t.Fatalf("competing owner sent %d proxy calls", len(calls))
	}
	lease, err = h.b.drainNode(h.ctx)
	if err != nil {
		t.Fatal(err)
	}
	h.release(t, lease)
}

func TestNodeDrainEarlyRefusalReleasesSettingsExclusion(t *testing.T) {
	h := newNodeDrainHarness(t)
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.runID = ""
	r.mu.Unlock()
	lease, err := h.b.drainNode(h.ctx)
	if err == nil || lease != nil {
		t.Fatalf("unverified proxy: lease %v, error %v", lease, err)
	}
	if !h.b.settingsApplyMu.TryLock() {
		t.Fatal("early refusal retained settings/switch exclusion")
	}
	h.b.settingsApplyMu.Unlock()
}

func TestNodeDrainReleasePreservesPreexistingSwitchDrain(t *testing.T) {
	h := newNodeDrainHarness(t)
	preexisting := h.engines[0]
	h.setPreexisting(preexisting)
	lease, err := h.b.drainNode(h.ctx)
	if err != nil {
		t.Fatalf("acquire beside switch drain: %v", err)
	}
	h.requireDrains(t, h.engines)
	if h.drainCalls(t, true)[preexisting] == 0 {
		t.Fatal("preexisting switch drain was not acknowledged for this proxy run")
	}
	h.release(t, lease)
	h.requireDrains(t, []string{preexisting})
	want := make(map[string]int)
	for _, engine := range h.engines[1:] {
		want[engine] = 1
	}
	if got := h.drainCalls(t, false); !reflect.DeepEqual(got, want) {
		t.Fatalf("release changed drains outside its lease: %v, want %v", got, want)
	}
}

func TestNodeDrainRequiresEveryEngineAcknowledgement(t *testing.T) {
	// A failure for any current engine, including a descriptor engine and a
	// preexisting switch drain, must prevent the first admission read.
	catalog := newNodeDrainHarness(t).engines
	for _, refused := range catalog {
		t.Run(refused, func(t *testing.T) {
			h := newNodeDrainHarness(t)
			preexisting := h.engines[0]
			h.setPreexisting(preexisting)
			var refusedCalls, snapshots atomic.Int32
			h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == nodepolicy.MethodAdmissionSnapshot {
					snapshots.Add(1)
				}
				if method == nodepolicy.MethodSetEngineDrain {
					var p nodepolicy.SetEngineDrainParams
					if err := json.Unmarshal(raw, &p); err != nil {
						return nil, err.Error()
					}
					if p.Engine == refused && p.Drain {
						refusedCalls.Add(1)
						return map[string]bool{"ok": false}, ""
					}
				}
				return h.answer(method, raw)
			})
			lease, err := h.b.drainNode(h.ctx)
			if err == nil || lease != nil || refusedCalls.Load() == 0 || snapshots.Load() != 0 {
				t.Fatalf("refused engine %s: lease %v, error %v, refusals %d, snapshots %d", refused, lease, err, refusedCalls.Load(), snapshots.Load())
			}
			h.requireDrains(t, []string{preexisting})
			if h.drainCalls(t, false)[preexisting] != 0 {
				t.Fatal("partial failure released the switch's preexisting drain")
			}
			failed := false
			for _, call := range h.proxy.snapshot() {
				if call.method != nodepolicy.MethodSetEngineDrain {
					if failed {
						t.Fatalf("effect after drain refusal: %s", call.method)
					}
					continue
				}
				var p nodepolicy.SetEngineDrainParams
				if err := json.Unmarshal(call.params, &p); err != nil {
					t.Fatal(err)
				}
				if failed && p.Drain {
					t.Fatalf("added drain after refusal: %s", p.Engine)
				}
				failed = failed || (p.Drain && p.Engine == refused)
			}
		})
	}
}

func TestNodeDrainWaitsForAggregateAndEveryEngineZero(t *testing.T) {
	test := func(name string, busy func(*nodeDrainHarness) nodepolicy.AdmissionSnapshotResult) {
		t.Run(name, func(t *testing.T) {
			h := newNodeDrainHarness(t)
			first, second, allowSecond := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var reads atomic.Int32
			h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method != nodepolicy.MethodAdmissionSnapshot {
					return h.answer(method, raw)
				}
				switch reads.Add(1) {
				case 1:
					close(first)
					return busy(h), ""
				case 2:
					close(second)
					select {
					case <-allowSecond:
					case <-h.ctx.Done():
						return nil, h.ctx.Err().Error()
					}
				}
				return h.idle(), ""
			})
			done := h.start(h.ctx)
			awaitNodeDrainBarrier(t, h, first, done)
			awaitNodeDrainBarrier(t, h, second, done)
			h.requireDrains(t, h.engines)
			close(allowSecond)
			result := awaitNodeDrain(t, h.ctx, done)
			if result.err != nil {
				t.Fatalf("fresh all-zero read: %v", result.err)
			}
			h.release(t, result.lease)
			h.requireDrains(t, nil)
		})
	}
	test("aggregate active with zero engine map", func(h *nodeDrainHarness) nodepolicy.AdmissionSnapshotResult {
		s := h.idle()
		s.Active = 1
		return s
	})
	test("aggregate queued without active requests", func(h *nodeDrainHarness) nodepolicy.AdmissionSnapshotResult {
		s := h.idle()
		s.Queued = 1
		return s
	})
	for _, engine := range newNodeDrainHarness(t).engines {
		test("active engine "+engine, func(h *nodeDrainHarness) nodepolicy.AdmissionSnapshotResult {
			s := h.idle()
			// A zero aggregate cannot conceal a nonzero per-engine count.
			s.ActiveByEngine[engine] = 1
			return s
		})
	}
	test("active engine outside catalog", func(h *nodeDrainHarness) nodepolicy.AdmissionSnapshotResult {
		s := h.idle()
		s.ActiveByEngine["other-fronted-engine"] = 1
		return s
	})
}

func TestNodeDrainFreshSnapshotsSeeWorkArrivingAfterFirstRead(t *testing.T) {
	h := newNodeDrainHarness(t)
	h.b.observeAdmission(h.idle().AdmissionState)
	second, third := make(chan struct{}), make(chan struct{})
	allowSecond, allowThird := make(chan struct{}), make(chan struct{})
	var reads atomic.Int32
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method != nodepolicy.MethodAdmissionSnapshot {
			return h.answer(method, raw)
		}
		s := h.idle()
		switch reads.Add(1) {
		case 1:
			s.Active, s.ActiveByEngine[h.engines[0]] = 1, 1
		case 2:
			close(second)
			select {
			case <-allowSecond:
			case <-h.ctx.Done():
				return nil, h.ctx.Err().Error()
			}
			// The original work ended, but new work arrived on a descriptor
			// engine and in the queue before this fresh read.
			s.Active, s.Queued, s.ActiveByEngine[h.engines[len(h.engines)-1]] = 1, 1, 1
		case 3:
			close(third)
			select {
			case <-allowThird:
			case <-h.ctx.Done():
				return nil, h.ctx.Err().Error()
			}
		}
		return s, ""
	})
	done := h.start(h.ctx)
	awaitNodeDrainBarrier(t, h, second, done)
	// Stale asynchronous zero after the first busy read cannot finish it.
	h.b.observeAdmission(h.idle().AdmissionState)
	close(allowSecond)
	awaitNodeDrainBarrier(t, h, third, done)
	h.requireDrains(t, h.engines)
	// Conversely, cached busy evidence cannot override the fresh final zero.
	h.b.observeAdmission(nodepolicy.AdmissionState{Active: 9, Queued: 9, ActiveByEngine: map[string]int{h.engines[0]: 9}})
	close(allowThird)
	result := awaitNodeDrain(t, h.ctx, done)
	if result.err != nil {
		t.Fatalf("idle after newly arrived work ended: %v", result.err)
	}
	h.release(t, result.lease)
	h.requireDrains(t, nil)
}

func TestNodeDrainRejectsMissingMalformedOrRefusedSnapshots(t *testing.T) {
	test := func(name string, reply any, refusal string) {
		t.Run(name, func(t *testing.T) {
			h := newNodeDrainHarness(t)
			preexisting := h.engines[0]
			h.setPreexisting(preexisting)
			var reads atomic.Int32
			h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == nodepolicy.MethodAdmissionSnapshot {
					reads.Add(1)
					return reply, refusal
				}
				return h.answer(method, raw)
			})
			lease, err := h.b.drainNode(h.ctx)
			if err == nil || lease != nil || reads.Load() != 1 {
				t.Fatalf("invalid snapshot: lease %v, error %v, reads %d", lease, err, reads.Load())
			}
			h.requireDrains(t, []string{preexisting})
			if h.drainCalls(t, false)[preexisting] != 0 {
				t.Fatal("snapshot failure released a preexisting switch drain")
			}
			failed := false
			for _, call := range h.proxy.snapshot() {
				if failed {
					if call.method != nodepolicy.MethodSetEngineDrain {
						t.Fatalf("effect after invalid idle evidence: %s", call.method)
					}
					var p nodepolicy.SetEngineDrainParams
					if err := json.Unmarshal(call.params, &p); err != nil {
						t.Fatal(err)
					}
					if p.Drain {
						t.Fatalf("added drain after invalid idle evidence: %s", p.Engine)
					}
				}
				failed = failed || call.method == nodepolicy.MethodAdmissionSnapshot
			}
		})
	}
	test("RPC refusal", nil, "snapshot unavailable")
	test("null result", nil, "")
	test("non-object result", []string{}, "")
	test("missing all evidence", map[string]string{}, "")
	valid := func() map[string]any {
		return map[string]any{"runId": "node-drain-run", "active": 0, "queued": 0, "activeByEngine": map[string]int{}}
	}
	for _, field := range []string{"runId", "active", "queued", "activeByEngine"} {
		reply := valid()
		delete(reply, field)
		test("missing "+field, reply, "")
		reply = valid()
		reply[field] = nil
		test("null "+field, reply, "")
	}
	for _, field := range []string{"active", "queued", "activeByEngine"} {
		reply := valid()
		reply[field] = "not a count"
		test("malformed "+field, reply, "")
	}
	for _, field := range []string{"active", "queued"} {
		reply := valid()
		reply[field] = -1
		test("negative "+field, reply, "")
	}
	reply := valid()
	reply["runId"] = "replacement-run"
	test("wrong proxy run", reply, "")
	reply = valid()
	reply["runId"] = ""
	test("empty proxy run", reply, "")
	reply = valid()
	reply["activeByEngine"] = map[string]int{"other-fronted-engine": -1}
	test("negative engine count outside catalog", reply, "")
	reply = valid()
	reply["activeByEngine"] = map[string]any{"llamacpp": nil}
	test("null engine count", reply, "")
}

func TestNodeDrainRejectsMissingMalformedOrRefusedDrainAcknowledgements(t *testing.T) {
	for _, tc := range []struct {
		name    string
		reply   any
		refusal string
	}{
		{name: "missing acknowledgement", reply: map[string]bool{}},
		{name: "null acknowledgement"},
		{name: "malformed acknowledgement", reply: map[string]string{"ok": "yes"}},
		{name: "RPC refusal", refusal: "drain unavailable"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newNodeDrainHarness(t)
			preexisting := h.engines[0]
			h.setPreexisting(preexisting)
			var attempts, snapshots atomic.Int32
			h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == nodepolicy.MethodAdmissionSnapshot {
					snapshots.Add(1)
				}
				if method == nodepolicy.MethodSetEngineDrain {
					var p nodepolicy.SetEngineDrainParams
					if err := json.Unmarshal(raw, &p); err != nil {
						return nil, err.Error()
					}
					if p.Drain {
						attempts.Add(1)
						return tc.reply, tc.refusal
					}
				}
				return h.answer(method, raw)
			})
			lease, err := h.b.drainNode(h.ctx)
			if err == nil || lease != nil || attempts.Load() == 0 || snapshots.Load() != 0 {
				t.Fatalf("unacknowledged drain: lease %v, error %v, attempts %d, snapshots %d", lease, err, attempts.Load(), snapshots.Load())
			}
			h.requireDrains(t, []string{preexisting})
			if h.drainCalls(t, false)[preexisting] != 0 {
				t.Fatal("acknowledgement failure cleared the switch drain")
			}
		})
	}
}

func TestNodeDrainProxyReplacementBeforeIdleReplyAbortsAndCleansOnlyNewDrains(t *testing.T) {
	for _, scenario := range []string{"process replacement", "run replacement on same handle", "proxy disappears"} {
		t.Run(scenario, func(t *testing.T) {
			h := newNodeDrainHarness(t)
			preexisting := h.engines[0]
			h.setPreexisting(preexisting)
			peer, replacement := newFakeWorkerPipe(t)
			newProxy := &proxyProcess{peer: peer}
			replacement.setAnswer(h.answer)
			reached, reply := make(chan struct{}), make(chan struct{})
			var reads atomic.Int32
			h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == nodepolicy.MethodAdmissionSnapshot {
					if reads.Add(1) == 1 {
						s := h.idle()
						s.Active, s.ActiveByEngine[h.engines[0]] = 1, 1
						return s, ""
					}
					close(reached)
					select {
					case <-reply:
					case <-h.ctx.Done():
						return nil, h.ctx.Err().Error()
					}
					return h.idle(), ""
				}
				return h.answer(method, raw)
			})
			done := h.start(h.ctx)
			awaitNodeDrainBarrier(t, h, reached, done)
			r := h.b.nodePolicyRuntime()
			switch scenario {
			case "process replacement":
				h.b.setProxyProcessHandle(newProxy)
				r.mu.Lock()
				r.incarnation, r.runID = newProxy, "replacement-run"
				r.mu.Unlock()
			case "run replacement on same handle":
				r.mu.Lock()
				r.runID = "replacement-run"
				r.mu.Unlock()
			case "proxy disappears":
				h.b.setProxyProcessHandle(nil)
			}
			close(reply)
			result := awaitNodeDrain(t, h.ctx, done)
			if result.err == nil || result.lease != nil {
				t.Fatalf("accepted retired proxy idle reply: %+v", result)
			}
			h.requireDrains(t, []string{preexisting})
			for _, worker := range []*fakeWorker{h.proxy, replacement} {
				for _, call := range worker.snapshot() {
					if call.method != nodepolicy.MethodSetEngineDrain {
						continue
					}
					var p nodepolicy.SetEngineDrainParams
					if err := json.Unmarshal(call.params, &p); err != nil {
						t.Fatal(err)
					}
					if !p.Drain && p.Engine == preexisting {
						t.Fatal("proxy replacement released the switch's drain")
					}
				}
			}
		})
	}
}

func TestNodeDrainCancellationDuringAcknowledgementOrSnapshotReleasesOnlyNewDrains(t *testing.T) {
	for _, blockedMethod := range []string{nodepolicy.MethodSetEngineDrain, nodepolicy.MethodAdmissionSnapshot} {
		t.Run(blockedMethod, func(t *testing.T) {
			h := newNodeDrainHarness(t)
			preexisting := h.engines[0]
			h.setPreexisting(preexisting)
			reached := make(chan struct{})
			var once sync.Once
			h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == blockedMethod {
					block := method == nodepolicy.MethodAdmissionSnapshot
					if method == nodepolicy.MethodSetEngineDrain {
						var p nodepolicy.SetEngineDrainParams
						if err := json.Unmarshal(raw, &p); err != nil {
							return nil, err.Error()
						}
						block = p.Drain && p.Engine == h.engines[len(h.engines)-1]
					}
					if block {
						once.Do(func() { close(reached) })
						return nil, holdReply
					}
				}
				return h.answer(method, raw)
			})
			ctx, cancel := context.WithCancel(h.ctx)
			defer cancel()
			done := h.start(ctx)
			awaitNodeDrainBarrier(t, h, reached, done)
			if blockedMethod == nodepolicy.MethodAdmissionSnapshot {
				h.requireDrains(t, h.engines)
			}
			cancel()
			result := awaitNodeDrain(t, h.ctx, done)
			if !errors.Is(result.err, context.Canceled) || result.lease != nil {
				t.Fatalf("cancelled acquisition = %+v", result)
			}
			h.requireDrains(t, []string{preexisting})
			if h.drainCalls(t, false)[preexisting] != 0 {
				t.Fatal("cancel released the switch drain")
			}
		})
	}
}

func TestNodeDrainProxyChangesDuringAcknowledgementPreventSnapshot(t *testing.T) {
	h := newNodeDrainHarness(t)
	preexisting := h.engines[0]
	h.setPreexisting(preexisting)
	peer, replacement := newFakeWorkerPipe(t)
	newProxy := &proxyProcess{peer: peer}
	replacement.setAnswer(h.answer)
	var changed atomic.Bool
	var snapshots atomic.Int32
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodAdmissionSnapshot {
			snapshots.Add(1)
		}
		if method == nodepolicy.MethodSetEngineDrain {
			var p nodepolicy.SetEngineDrainParams
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if p.Drain && !changed.Swap(true) {
				h.b.setProxyProcessHandle(newProxy)
				r := h.b.nodePolicyRuntime()
				r.mu.Lock()
				r.incarnation, r.runID = newProxy, "replacement-run"
				r.mu.Unlock()
			}
		}
		return h.answer(method, raw)
	})
	lease, err := h.b.drainNode(h.ctx)
	if err == nil || lease != nil || !changed.Load() || snapshots.Load() != 0 {
		t.Fatalf("replacement during acknowledgement: lease %v, error %v, changed %v, snapshots %d", lease, err, changed.Load(), snapshots.Load())
	}
	h.requireDrains(t, []string{preexisting})
	for _, call := range replacement.snapshot() {
		if call.method != nodepolicy.MethodSetEngineDrain {
			t.Fatalf("continued acquisition on replacement proxy: %s", call.method)
		}
		var p nodepolicy.SetEngineDrainParams
		if err := json.Unmarshal(call.params, &p); err != nil {
			t.Fatal(err)
		}
		if p.Drain || p.Engine == preexisting {
			t.Fatalf("replacement received an effect beyond new-drain cleanup: %+v", p)
		}
	}
}

func TestNodeDrainDeadlineIncludesPushSlotAcknowledgementAndActiveWait(t *testing.T) {
	for _, scenario := range []string{"push slot", "acknowledgement", "active work", "snapshot reply"} {
		t.Run(scenario, func(t *testing.T) {
			h := newNodeDrainHarness(t)
			preexisting := h.engines[0]
			h.setPreexisting(preexisting)
			r := h.b.nodePolicyRuntime()
			if scenario == "push slot" {
				if !r.acquirePush(h.ctx) {
					t.Fatal("acquire policy push slot")
				}
				defer r.releasePush()
			}
			h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if scenario == "acknowledgement" && method == nodepolicy.MethodSetEngineDrain {
					var p nodepolicy.SetEngineDrainParams
					if err := json.Unmarshal(raw, &p); err != nil {
						return nil, err.Error()
					}
					if p.Drain {
						return nil, holdReply
					}
				}
				if method == nodepolicy.MethodAdmissionSnapshot {
					if scenario == "snapshot reply" {
						return nil, holdReply
					}
					if scenario == "active work" {
						s := h.idle()
						s.Active, s.ActiveByEngine[h.engines[len(h.engines)-1]] = 1, 1
						return s, ""
					}
				}
				return h.answer(method, raw)
			})
			ctx, cancel := context.WithTimeout(h.ctx, 150*time.Millisecond)
			defer cancel()
			result := awaitNodeDrain(t, h.ctx, h.start(ctx))
			if !errors.Is(result.err, context.DeadlineExceeded) || result.lease != nil {
				t.Fatalf("%s was not covered by acquisition deadline: %+v", scenario, result)
			}
			h.requireDrains(t, []string{preexisting})
			if h.drainCalls(t, false)[preexisting] != 0 {
				t.Fatal("deadline released the switch drain")
			}
			if scenario == "push slot" && len(h.proxy.snapshot()) != 0 {
				t.Fatal("node drain bypassed the policy push serialization slot")
			}
		})
	}
}

func TestNodeDrainRefusesUnavailableOrUnverifiedProxyWithoutClearingSwitchDrain(t *testing.T) {
	for _, scenario := range []string{"missing proxy", "missing run", "wrong incarnation", "cancelled context"} {
		t.Run(scenario, func(t *testing.T) {
			h := newNodeDrainHarness(t)
			preexisting := h.engines[0]
			h.setPreexisting(preexisting)
			ctx, cancel := context.WithCancel(h.ctx)
			defer cancel()
			r := h.b.nodePolicyRuntime()
			switch scenario {
			case "missing proxy":
				h.b.setProxyProcessHandle(nil)
			case "missing run":
				r.mu.Lock()
				r.runID = ""
				r.mu.Unlock()
			case "wrong incarnation":
				r.mu.Lock()
				r.incarnation = &proxyProcess{}
				r.mu.Unlock()
			case "cancelled context":
				cancel()
			}
			lease, err := h.b.drainNode(ctx)
			if err == nil || lease != nil {
				t.Fatalf("unverified node drain: lease %v, error %v", lease, err)
			}
			h.requireDrains(t, []string{preexisting})
			if calls := h.proxy.snapshot(); len(calls) != 0 {
				t.Fatalf("unverified proxy received effects: %v", calls)
			}
		})
	}
}

func TestNodeDrainReleaseRetainsFailedUndrainForRetry(t *testing.T) {
	for _, tc := range []struct {
		name    string
		reply   any
		refusal string
	}{
		{name: "refused acknowledgement", reply: map[string]bool{"ok": false}},
		{name: "missing acknowledgement", reply: map[string]bool{}},
		{name: "null result"},
		{name: "null acknowledgement", reply: map[string]any{"ok": nil}},
		{name: "malformed acknowledgement", reply: map[string]string{"ok": "yes"}},
		{name: "RPC refusal", refusal: "undrain unavailable"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newNodeDrainHarness(t)
			preexisting, failed := h.engines[0], h.engines[len(h.engines)-1]
			h.setPreexisting(preexisting)
			lease, err := h.b.drainNode(h.ctx)
			if err != nil {
				t.Fatalf("acquire node drain: %v", err)
			}
			h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == nodepolicy.MethodSetEngineDrain {
					var p nodepolicy.SetEngineDrainParams
					if err := json.Unmarshal(raw, &p); err != nil {
						return nil, err.Error()
					}
					if !p.Drain && p.Engine == failed {
						return tc.reply, tc.refusal
					}
				}
				return h.answer(method, raw)
			})
			if err := lease.release(h.ctx); err == nil {
				t.Fatal("failed undrain was reported released")
			}
			h.requireDrains(t, []string{preexisting, failed})
			h.proxy.setAnswer(h.answer)
			h.release(t, lease)
			h.requireDrains(t, []string{preexisting})
			h.release(t, lease)
			want := make(map[string]int)
			for _, engine := range h.engines[1:] {
				want[engine] = 1
			}
			want[failed]++
			if got := h.drainCalls(t, false); !reflect.DeepEqual(got, want) {
				t.Fatalf("retry/idempotent release calls = %v, want %v", got, want)
			}
		})
	}
}

func TestNodeDrainConcurrentReleaseAcknowledgesEachOwnedEntryOnce(t *testing.T) {
	h := newNodeDrainHarness(t)
	preexisting := h.engines[0]
	h.setPreexisting(preexisting)
	lease, err := h.b.drainNode(h.ctx)
	if err != nil {
		t.Fatalf("acquire node drain: %v", err)
	}
	reached := make(chan struct{})
	var once sync.Once
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetEngineDrain {
			var p nodepolicy.SetEngineDrainParams
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if !p.Drain && p.Engine == h.engines[1] {
				once.Do(func() { close(reached) })
				return map[string]bool{"ok": true}, holdSuccessReply
			}
		}
		return h.answer(method, raw)
	})
	done := make(chan error, 2)
	go func() { done <- lease.release(h.ctx) }()
	select {
	case <-reached:
	case err := <-done:
		t.Fatalf("release returned before acknowledgement barrier: %v", err)
	case <-h.ctx.Done():
		t.Fatal("release never requested undrain acknowledgement")
	}
	h.requireDrains(t, h.engines)
	started := make(chan struct{})
	go func() {
		close(started)
		done <- lease.release(h.ctx)
	}()
	<-started
	h.proxy.releaseHeld()
	for range cap(done) {
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("concurrent release: %v", err)
			}
		case <-h.ctx.Done():
			t.Fatal("concurrent release did not finish")
		}
	}
	h.release(t, lease)
	h.requireDrains(t, []string{preexisting})
	want := make(map[string]int)
	for _, engine := range h.engines[1:] {
		want[engine] = 1
	}
	if got := h.drainCalls(t, false); !reflect.DeepEqual(got, want) {
		t.Fatalf("concurrent/idempotent release calls = %v, want %v", got, want)
	}
}

func TestNodeDrainCancelledReleaseDoesNotAcceptLateAcknowledgement(t *testing.T) {
	h := newNodeDrainHarness(t)
	preexisting, firstOwned := h.engines[0], h.engines[1]
	h.setPreexisting(preexisting)
	lease, err := h.b.drainNode(h.ctx)
	if err != nil {
		t.Fatalf("acquire node drain: %v", err)
	}
	reached := make(chan struct{})
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetEngineDrain {
			var p nodepolicy.SetEngineDrainParams
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if !p.Drain && p.Engine == firstOwned {
				close(reached)
				return map[string]bool{"ok": true}, holdSuccessReply
			}
		}
		return h.answer(method, raw)
	})
	ctx, cancel := context.WithCancel(h.ctx)
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- lease.release(ctx) }()
	select {
	case <-reached:
	case err := <-done:
		t.Fatalf("release returned before held acknowledgement: %v", err)
	case <-h.ctx.Done():
		t.Fatal("release never reached acknowledgement barrier")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled release error = %v", err)
		}
	case <-h.ctx.Done():
		t.Fatal("cancelled release did not return")
	}
	h.proxy.releaseHeld()
	h.requireDrains(t, h.engines)
	h.proxy.setAnswer(h.answer)
	h.release(t, lease)
	h.requireDrains(t, []string{preexisting})
	want := make(map[string]int)
	for _, engine := range h.engines[1:] {
		want[engine] = 1
	}
	want[firstOwned]++
	if got := h.drainCalls(t, false); !reflect.DeepEqual(got, want) {
		t.Fatalf("late acknowledgement lost retry authority: calls %v, want %v", got, want)
	}
}

func TestNodeDrainReleaseDeadlineIncludesPushSlot(t *testing.T) {
	h := newNodeDrainHarness(t)
	lease, err := h.b.drainNode(h.ctx)
	if err != nil {
		t.Fatalf("acquire node drain: %v", err)
	}
	r := h.b.nodePolicyRuntime()
	if !r.acquirePush(h.ctx) {
		t.Fatal("hold policy push slot")
	}
	ctx, cancel := context.WithTimeout(h.ctx, 35*time.Millisecond)
	err = lease.release(ctx)
	cancel()
	r.releasePush()
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("release push-slot deadline error = %v", err)
	}
	h.requireDrains(t, h.engines)
	if calls := h.drainCalls(t, false); len(calls) != 0 {
		t.Fatalf("release bypassed held push slot: %v", calls)
	}
	h.release(t, lease)
	h.requireDrains(t, nil)
}

func TestNodeDrainRollbackFailureReturnsLeaseWithRemainingCleanupAuthority(t *testing.T) {
	h := newNodeDrainHarness(t)
	preexisting, failed := h.engines[0], h.engines[len(h.engines)-1]
	h.setPreexisting(preexisting)
	reached := make(chan struct{})
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetEngineDrain {
			var p nodepolicy.SetEngineDrainParams
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if p.Engine == failed {
				if p.Drain {
					close(reached)
					return nil, holdReply
				}
				return nil, "rollback undrain unavailable"
			}
		}
		return h.answer(method, raw)
	})
	ctx, cancel := context.WithCancel(h.ctx)
	defer cancel()
	done := h.start(ctx)
	awaitNodeDrainBarrier(t, h, reached, done)
	cancel()
	result := awaitNodeDrain(t, h.ctx, done)
	if !errors.Is(result.err, context.Canceled) || result.lease == nil {
		t.Fatalf("failed rollback lost acquisition error or cleanup lease: %+v", result)
	}
	h.requireDrains(t, []string{preexisting, failed})
	h.proxy.setAnswer(h.answer)
	h.release(t, result.lease)
	h.requireDrains(t, []string{preexisting})
	want := make(map[string]int)
	for _, engine := range h.engines[1:] {
		want[engine] = 1
	}
	want[failed]++
	if got := h.drainCalls(t, false); !reflect.DeepEqual(got, want) {
		t.Fatalf("rollback retry released unowned/already released entries: %v, want %v", got, want)
	}
}

func TestNodeDrainReleaseRejectsRetiredProxyAcknowledgement(t *testing.T) {
	for _, scenario := range []string{"process replacement", "run replacement"} {
		t.Run(scenario, func(t *testing.T) {
			h := newNodeDrainHarness(t)
			preexisting := h.engines[0]
			h.setPreexisting(preexisting)
			lease, err := h.b.drainNode(h.ctx)
			if err != nil {
				t.Fatalf("acquire node drain: %v", err)
			}
			peer, replacement := newFakeWorkerPipe(t)
			newProxy := &proxyProcess{peer: peer}
			replacement.setAnswer(h.answer)
			var changed atomic.Bool
			h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == nodepolicy.MethodSetEngineDrain {
					var p nodepolicy.SetEngineDrainParams
					if err := json.Unmarshal(raw, &p); err != nil {
						return nil, err.Error()
					}
					if !p.Drain && !changed.Swap(true) {
						r := h.b.nodePolicyRuntime()
						if scenario == "process replacement" {
							h.b.setProxyProcessHandle(newProxy)
						}
						r.mu.Lock()
						if scenario == "process replacement" {
							r.incarnation = newProxy
						}
						r.runID = "replacement-run"
						r.mu.Unlock()
					}
				}
				return h.answer(method, raw)
			})
			if err := lease.release(h.ctx); err == nil || !changed.Load() {
				t.Fatalf("release accepted retired proxy acknowledgement: error %v, changed %v", err, changed.Load())
			}
			h.requireDrains(t, h.engines)
			h.release(t, lease)
			h.requireDrains(t, []string{preexisting})
			worker := h.proxy
			if scenario == "process replacement" {
				worker = replacement
			}
			seen := make(map[string]bool)
			for _, call := range worker.snapshot() {
				if call.method != nodepolicy.MethodSetEngineDrain {
					continue
				}
				var p nodepolicy.SetEngineDrainParams
				if err := json.Unmarshal(call.params, &p); err != nil {
					t.Fatal(err)
				}
				if !p.Drain {
					if p.Engine == preexisting {
						t.Fatal("release removed the switch drain on replacement run")
					}
					seen[p.Engine] = true
				}
			}
			for _, engine := range h.engines[1:] {
				if !seen[engine] {
					t.Fatalf("replacement run did not acknowledge release of %s", engine)
				}
			}
		})
	}
}

func TestNodeDrainReleaseRetainsOwnershipUntilReplacementRunIsVerified(t *testing.T) {
	h := newNodeDrainHarness(t)
	preexisting := h.engines[0]
	h.setPreexisting(preexisting)
	lease, err := h.b.drainNode(h.ctx)
	if err != nil {
		t.Fatalf("acquire node drain: %v", err)
	}
	peer, replacement := newFakeWorkerPipe(t)
	newProxy := &proxyProcess{peer: peer}
	replacement.setAnswer(h.answer)
	h.b.setProxyProcessHandle(newProxy)
	if err := lease.release(h.ctx); err == nil {
		t.Fatal("release discarded ownership on an unverified live replacement")
	}
	h.requireDrains(t, h.engines)
	if calls := replacement.snapshot(); len(calls) != 0 {
		t.Fatalf("release used an unverified replacement: %v", calls)
	}
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.incarnation, r.runID = newProxy, ""
	r.mu.Unlock()
	if err := lease.release(h.ctx); err == nil {
		t.Fatal("release accepted a missing replacement run ID")
	}
	h.requireDrains(t, h.engines)
	if calls := replacement.snapshot(); len(calls) != 0 {
		t.Fatalf("release used a replacement without a run ID: %v", calls)
	}
	r.mu.Lock()
	r.runID = "replacement-run"
	r.mu.Unlock()
	h.release(t, lease)
	h.requireDrains(t, []string{preexisting})
	want := make(map[string]int)
	for _, engine := range h.engines[1:] {
		want[engine] = 1
	}
	got := make(map[string]int)
	for _, call := range replacement.snapshot() {
		if call.method != nodepolicy.MethodSetEngineDrain {
			t.Fatalf("release issued an unrelated replacement effect: %s", call.method)
		}
		var p nodepolicy.SetEngineDrainParams
		if err := json.Unmarshal(call.params, &p); err != nil {
			t.Fatal(err)
		}
		if p.Drain {
			t.Fatalf("release added a drain: %+v", p)
		}
		got[p.Engine]++
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("verified replacement release calls = %v, want %v", got, want)
	}
}

func TestNodeDrainRollbackRemovesOwnedEntriesBeforeWaitingReplacementReplay(t *testing.T) {
	h := newNodeDrainHarness(t)
	preexisting := h.engines[0]
	h.setPreexisting(preexisting)
	peer, replacement := newFakeWorkerPipe(t)
	newProxy := &proxyProcess{peer: peer}
	replacement.setAnswer(h.answer)
	reached, reply := make(chan struct{}), make(chan struct{})
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodAdmissionSnapshot {
			close(reached)
			select {
			case <-reply:
			case <-h.ctx.Done():
				return nil, h.ctx.Err().Error()
			}
			return h.idle(), ""
		}
		return h.answer(method, raw)
	})
	done := h.start(h.ctx)
	awaitNodeDrainBarrier(t, h, reached, done)
	h.b.setProxyProcessHandle(newProxy)
	r := h.b.nodePolicyRuntime()
	// Replay cannot establish the replacement's incarnation or read drains
	// until it gets the same push slot still held by this acquisition.
	started := make(chan struct{})
	replayed := make(chan map[string]bool, 1)
	go func() {
		close(started)
		if !r.acquirePush(h.ctx) {
			replayed <- nil
			return
		}
		defer r.releasePush()
		r.mu.Lock()
		drains := make(map[string]bool, len(r.drains))
		for engine, draining := range r.drains {
			drains[engine] = draining
		}
		r.incarnation, r.runID = newProxy, "replacement-run"
		r.mu.Unlock()
		replayed <- drains
	}()
	<-started
	close(reply)
	result := awaitNodeDrain(t, h.ctx, done)
	if result.err == nil || result.lease != nil {
		t.Fatalf("replacement rollback = %+v", result)
	}
	select {
	case drains := <-replayed:
		if !reflect.DeepEqual(drains, map[string]bool{preexisting: true}) {
			t.Fatalf("replacement replay saw abandoned node drains: %v", drains)
		}
	case <-h.ctx.Done():
		t.Fatal("rollback deadlocked on replacement replay's push slot")
	}
	h.requireDrains(t, []string{preexisting})
	if calls := replacement.snapshot(); len(calls) != 0 {
		t.Fatalf("rollback contacted replacement before its policy replay: %v", calls)
	}
}
