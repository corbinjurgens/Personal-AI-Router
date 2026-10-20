// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

func TestSwitchDrainActiveSourceMustReachFreshZero(t *testing.T) {
	h := newSwitchHarness(t)
	h.active.Store(2)
	p := h.request(t, "active-drain")
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	h.proxy.waitFor("fresh admission request", func(calls []policyCall) bool {
		for _, call := range calls {
			if call.method == nodepolicy.MethodAdmissionSnapshot {
				return true
			}
		}
		return false
	})
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
	wantPrefix := []string{nodepolicy.MethodSetModelIdentity, nodepolicy.MethodSetModelIdentity, "llamacpp:node/set-local-backend", nodepolicy.MethodSetEngineDrain, nodepolicy.MethodInvalidateEngine, nodepolicy.MethodInvalidateEngine, nodepolicy.MethodAdmissionSnapshot}
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
