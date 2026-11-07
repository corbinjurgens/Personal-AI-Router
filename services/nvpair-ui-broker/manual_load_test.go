// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"nvpair-shared/buildapply"
	"nvpair-shared/nodepolicy"
)

func TestManualLoadRequiresAuthoritativeEnvelope(t *testing.T) {
	f := newModelConfigFixture(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	id := r.engineIdentity["llamacpp"]
	id.Serving = nil
	r.engineIdentity["llamacpp"] = id
	r.mu.Unlock()
	err := f.b.manualLoad(t.Context(), manualLoadParams{Engine: "llamacpp", Model: "resident"}, nil)
	if err == nil || !strings.Contains(err.Message, "unsupported") {
		t.Fatalf("missing envelope refusal = %+v", err)
	}
	if len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)) != 0 || len(modelConfigCalls(f.engine, "engine:load-model")) != 0 || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 {
		t.Fatal("missing envelope reserved or reached a native load")
	}
}

func TestManualLoadUsesResidentReservationAndObservedFinish(t *testing.T) {
	test := func(name string, outcome nodepolicy.ResidentOutcome, loaded bool, wantSuccess bool) {
		t.Run(name, func(t *testing.T) {
			f := newModelConfigFixture(t)
			if outcome == nodepolicy.ResidentUnknown {
				f.proxy.mu.Lock()
				answer := f.proxy.answer
				f.proxy.mu.Unlock()
				f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
					if method == nodepolicy.MethodFinishResident {
						var p nodepolicy.FinishResidentParams
						if err := json.Unmarshal(raw, &p); err != nil {
							return nil, err.Error()
						}
						return nodepolicy.FinishResidentResult{ReservationID: p.ReservationID, Outcome: p.Outcome, RecoveryID: "fixture-recovery"}, ""
					}
					return answer(method, raw)
				})
			}
			f.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == buildapply.EngineOwnerEvidenceMethod {
					return modelConfigOwnerEvidence(t), ""
				}
				if method != nodepolicy.MethodReconcileResident {
					return map[string]bool{"ok": true}, ""
				}
				var p nodepolicy.ReconcileResidentParams
				if err := json.Unmarshal(raw, &p); err != nil {
					return nil, err.Error()
				}
				grant, err := json.Marshal(struct {
					ID     string                            `json:"id"`
					Params nodepolicy.DispatchResidentParams `json:"params"`
				}{"fixture-dispatch", nodepolicy.DispatchResidentParams{
					ReservationID: p.Reservation.ReservationID, ProxyRunID: p.Reservation.ProxyRunID,
					ManagerIncarnation: p.Reservation.ManagerIncarnation, OperationID: p.OperationID,
					PolicyRevision: p.PolicyRevision, Engine: p.Engine, Model: p.Model, Identity: p.Identity,
				}})
				if err != nil {
					return nil, err.Error()
				}
				f.b.dispatchResidentFrom(f.source, grant)
				result := nodepolicy.ReconcileResidentResult{Outcome: outcome, Observation: &nodepolicy.ResidentLoadedObservation{
					ManagerIncarnation: p.Reservation.ManagerIncarnation, Generation: 1,
					Engine: p.Engine, Model: p.Model, Identity: p.Identity, Loaded: loaded,
				}}
				if outcome == nodepolicy.ResidentUnknown {
					result.Observation = nil
				}
				return result, holdSuccessReply
			})
			result := make(chan *RPCError, 1)
			go func() {
				result <- f.b.manualLoad(context.Background(), manualLoadParams{Engine: "llamacpp", Model: "resident"}, nil)
			}()
			f.awaitGrant(t)
			f.engine.releaseHeld()
			select {
			case err := <-result:
				if (err == nil) != wantSuccess {
					t.Fatalf("manual outcome = %+v, want success %v", err, wantSuccess)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("manual load did not finish")
			}
			prepares := modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)
			finishes := modelConfigCalls(f.proxy, nodepolicy.MethodFinishResident)
			if len(prepares) != 1 || len(finishes) != 1 {
				t.Fatalf("prepare/finish calls = %d/%d", len(prepares), len(finishes))
			}
			prepared := decodeModelConfigCall[nodepolicy.PrepareResidentParams](t, prepares[0])
			finished := decodeModelConfigCall[nodepolicy.FinishResidentParams](t, finishes[0])
			if prepared.Purpose != nodepolicy.ResidentPurposeManual || finished.OperationID != prepared.OperationID || finished.Outcome != outcome {
				t.Fatalf("manual reservation = %+v, finish = %+v", prepared, finished)
			}
			if outcome != nodepolicy.ResidentUnknown && (finished.Observation == nil || finished.Observation.Generation != 1 || finished.Observation.Loaded != loaded) {
				t.Fatalf("finish lost authoritative observation: %+v", finished)
			}
			if outcome == nodepolicy.ResidentUnknown {
				r := f.b.nodePolicyRuntime()
				ref := nodepolicy.ModelRef{Engine: "llamacpp", Model: "resident"}
				f.b.reconcileModelConfig(t.Context())
				r.mu.Lock()
				_, held := r.models.unsettled[ref]
				r.mu.Unlock()
				if !held || len(modelConfigCalls(f.proxy, nodepolicy.MethodObserveResident)) != 0 || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
					t.Fatal("uncertain load lost its hold or dispatched again without fresh inventory")
				}
			}
			if len(modelConfigCalls(f.engine, "engine:load-model")) != 0 {
				t.Fatal("manual load bypassed the reservation executor")
			}
		})
	}
	test("loaded observation", nodepolicy.ResidentLoaded, true, true)
	test("definite failure observed absent", nodepolicy.ResidentFailed, false, false)
	test("unknown dispatch retains hold", nodepolicy.ResidentUnknown, false, false)
}

func TestManualLoadPrivateRelayRejectsClients(t *testing.T) {
	for _, method := range []string{admittedLoadRequestMethod, admittedLoadReplyMethod, "llamacpp-proxy:" + admittedLoadRequestMethod} {
		t.Run(method, func(t *testing.T) {
			b, out := newPolicyBroker(t)
			engine := attachFakeEngine(t, b)
			b.handleMessage(clientRequest(1, method, map[string]any{"id": "private", "params": manualLoadParams{Engine: "llamacpp", Model: "m"}}))
			reply := out.waitFrame(t, "private load refusal", isResponse(1))
			if reply.Error == nil || reply.Error.Code != -32601 || len(engine.snapshot()) != 0 {
				t.Fatalf("private relay escaped: %+v", reply)
			}
		})
	}
}

func TestManualLoadRetiredProxyRequiresTerminalAndFreshInventory(t *testing.T) {
	f := newModelConfigFixture(t)
	e := f.b.captureModelConfig()
	dead := make(chan struct{})
	close(dead)
	old := e
	old.Proxy = &proxyProcess{done: dead}
	hold := residentUnsettled{op: &modelConfigAttempt{evidence: old, prepare: nodepolicy.PrepareResidentParams{
		Purpose: nodepolicy.ResidentPurposeManual, Engine: "llamacpp", Model: "resident", ManagerIncarnation: "fixture-manager",
	}}}
	inventoryAvailable := false
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		switch method {
		case buildapply.EngineOwnerEvidenceMethod:
			return modelConfigOwnerEvidence(t), ""
		case "engine:models":
			if inventoryAvailable {
				return map[string]any{"loadedByEngine": map[string][]string{"llamacpp": {"resident"}}}, ""
			}
			return map[string]any{"loadedByEngine": map[string][]string{}}, ""
		}
		return nil, "unexpected manager call"
	})
	if f.b.recoverRetiredManualLoad(t.Context(), hold) {
		t.Fatal("nonterminal uncertain dispatch lost its hold")
	}
	hold.terminal = true
	if f.b.recoverRetiredManualLoad(t.Context(), hold) {
		t.Fatal("missing fresh inventory discharged a terminal hold")
	}
	inventoryAvailable = true
	if !f.b.recoverRetiredManualLoad(t.Context(), hold) {
		t.Fatal("terminal same-manager completion and fresh inventory stayed blocked")
	}
	calls := modelConfigCalls(f.proxy, nodepolicy.MethodSetResidency)
	if len(calls) == 0 {
		t.Fatal("replacement proxy never received settled residency")
	}
	last := decodeModelConfigCall[nodepolicy.SetResidencyParams](t, calls[len(calls)-1])
	if len(last.LoadedByEngine["llamacpp"]) != 1 || last.LoadedByEngine["llamacpp"][0] != "resident" {
		t.Fatalf("replacement capacity observation = %+v", last)
	}
}

func replacementManualHold(t *testing.T) (*modelConfigFixture, *proxyProcess, *fakeWorker, *fakeWorker, nodepolicy.ModelRef) {
	t.Helper()
	f := newModelConfigFixture(t)
	e := f.b.captureModelConfig()
	oldProxy, oldWorker := f.pp, f.proxy
	ref := nodepolicy.ModelRef{Engine: "llamacpp", Model: "resident"}
	op := &modelConfigAttempt{evidence: e, granted: true, prepare: nodepolicy.PrepareResidentParams{
		Purpose: nodepolicy.ResidentPurposeManual, Engine: ref.Engine, Model: ref.Model, ManagerIncarnation: "fixture-manager",
	}}
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.models.unsettled = map[nodepolicy.ModelRef]residentUnsettled{ref: {
		op: op, finish: nodepolicy.FinishResidentParams{ReservationID: "held-manual", Outcome: nodepolicy.ResidentUnknown},
	}}
	r.mu.Unlock()
	oldWorker.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == nodepolicy.MethodFinishResident {
			return nodepolicy.FinishResidentResult{ReservationID: "held-manual", Outcome: nodepolicy.ResidentUnknown, RecoveryID: "still-uncertain"}, ""
		}
		return nil, "unexpected old proxy call"
	})
	pp, worker, _ := attachAckingProxy(t, f.b)
	if !f.b.replayPolicyToProxy(t.Context(), pp) {
		t.Fatal("replacement proxy replay failed")
	}
	return f, oldProxy, oldWorker, worker, ref
}

func assertManualDrain(t *testing.T, b *Broker, worker *fakeWorker, ref nodepolicy.ModelRef, want bool) {
	t.Helper()
	b.pushPolicyNow(t.Context(), pushDrainPrefix+ref.Engine)
	drains := drainPushes(worker.snapshot())
	if len(drains) == 0 || drains[len(drains)-1] != (nodepolicy.SetEngineDrainParams{Engine: ref.Engine, Drain: want}) {
		t.Fatalf("engine drain pushes=%+v, want final drain %v", drains, want)
	}
	if got := b.captureModelConfig().Drains[ref.Engine]; got != want {
		t.Fatalf("reported engine drain=%v, want %v", got, want)
	}
}

func TestManualLoadReplacementDrainsOnlyTargetBeforeActivation(t *testing.T) {
	f, _, _, worker, ref := replacementManualHold(t)
	calls := worker.snapshot()
	if len(calls) == 0 || calls[0].method != nodepolicy.MethodSetEngineDrain {
		t.Fatalf("replacement activated before its drain: %+v", calls)
	}
	for _, drain := range drainPushes(calls) {
		if drain.Engine != ref.Engine || !drain.Drain {
			t.Fatalf("manual uncertainty drained another engine or cleared early: %+v", drain)
		}
	}
	e := f.b.captureModelConfig()
	if !e.Running[ref.Engine] || e.Live != nodepolicy.Available || e.Drains["llamacpp-custom"] {
		t.Fatalf("manual drain changed engine health, node availability or another engine: %+v", e)
	}
	entry, _, _, _ := f.b.modelConfigEntry(e, ref)
	if entry.Reason != nodepolicy.ModelReasonOperationBusy {
		t.Fatalf("draining operation reported as %s", entry.Reason)
	}
	if err := f.b.manualLoad(t.Context(), manualLoadParams{Engine: ref.Engine, Model: "other-model"}, nil); err == nil || !strings.Contains(err.Message, "draining") {
		t.Fatalf("another model bypassed the replacement drain: %+v", err)
	}
}

func TestManualLoadReplacementTerminalInventoryReleasesDrain(t *testing.T) {
	f, old, _, worker, ref := replacementManualHold(t)
	old.done = make(chan struct{})
	close(old.done)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	hold := r.models.unsettled[ref]
	hold.terminal = true
	r.models.unsettled[ref] = hold
	r.mu.Unlock()
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == buildapply.EngineOwnerEvidenceMethod {
			return modelConfigOwnerEvidence(t), ""
		}
		return map[string]any{"loadedByEngine": map[string][]string{ref.Engine: {}}}, ""
	})
	f.b.recoverManualLoads(t.Context())
	assertManualDrain(t, f.b, worker, ref, false)
}

func TestManualLoadExistingFinishResolutionReleasesDrain(t *testing.T) {
	f, _, oldWorker, worker, ref := replacementManualHold(t)
	oldWorker.setAnswer(func(string, json.RawMessage) (any, string) {
		return nodepolicy.FinishResidentResult{ReservationID: "held-manual", Outcome: nodepolicy.ResidentFailed}, ""
	})
	f.b.recoverManualLoads(t.Context())
	assertManualDrain(t, f.b, worker, ref, false)
}

func TestManualLoadDrainDoesNotOutliveForgottenEntry(t *testing.T) {
	f, _, _, worker, ref := replacementManualHold(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.models.unsettled = nil
	r.mu.Unlock()
	assertManualDrain(t, f.b, worker, ref, false)
}

func TestManualAndSettingsDrainsReleaseIndependently(t *testing.T) {
	for _, first := range []string{"manual", "settings"} {
		t.Run(first+" clears first", func(t *testing.T) {
			f, _, oldWorker, worker, ref := replacementManualHold(t)
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			r.drains[ref.Engine] = true
			r.mu.Unlock()
			if first == "manual" {
				oldWorker.setAnswer(func(string, json.RawMessage) (any, string) {
					return nodepolicy.FinishResidentResult{ReservationID: "held-manual", Outcome: nodepolicy.ResidentFailed}, ""
				})
				f.b.recoverManualLoads(t.Context())
			} else {
				if _, err := f.b.pushEngineDrain(t.Context(), ref.Engine, false); err != nil {
					t.Fatal(err)
				}
				r.mu.Lock()
				delete(r.drains, ref.Engine)
				r.mu.Unlock()
			}
			assertManualDrain(t, f.b, worker, ref, true)
		})
	}
}

func TestManualLoadActiveGrantDrainsReplacement(t *testing.T) {
	f, _, _, worker, ref := replacementManualHold(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.models.active = r.models.unsettled[ref].op
	r.models.unsettled = nil
	r.mu.Unlock()
	assertManualDrain(t, f.b, worker, ref, true)
	r.mu.Lock()
	r.models.active = nil
	r.mu.Unlock()
	assertManualDrain(t, f.b, worker, ref, false)
}

func TestManualLoadDrainMustBeAcknowledgedBeforeReplacementActivation(t *testing.T) {
	f, _, _, _, _ := replacementManualHold(t)
	pp, worker, acks := attachAckingProxy(t, f.b)
	worker.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetEngineDrain {
			return nil, "drain refused"
		}
		return acks.answer(method, raw)
	})
	f.b.replayPolicyToProxy(t.Context(), pp)
	for _, call := range worker.snapshot() {
		if call.method == nodepolicy.MethodSetAvailability || call.method == nodepolicy.MethodSetPolicy {
			t.Fatalf("replacement activated without the uncertainty drain: %s", call.method)
		}
	}
}
