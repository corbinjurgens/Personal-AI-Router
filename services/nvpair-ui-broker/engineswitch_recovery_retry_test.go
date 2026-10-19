// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrusttest"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

func TestSwitchRecoveryLateOffReconcilesBeforeRelease(t *testing.T) {
	test := func(name, engine string, duringDrain bool) {
		t.Run(name, func(t *testing.T) {
			h := newSwitchHarness(t)
			r := switchStoreReceipt("recovery-late-off")
			r.Caller = "peer"
			r.PayloadHash = settings.SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
			switchStoreAccept(t, h.b.engineSwitch.store, r)
			h.mu.Lock()
			h.cp = &settings.SwitchCheckpoint{
				SwitchPrepareRequest: settings.SwitchPrepareRequest{OperationID: r.RequestID, SourceEngine: r.Request.SourceEngine, TargetEngine: r.Request.TargetEngine, Origin: settings.SwitchOriginRemote},
				Phase:                settings.SwitchCheckpointTargetCommitted, TargetCommitted: true,
				Source: switchTestState(false, false, settings.SwitchIntentOff),
				Target: switchTestState(true, true, settings.SwitchIntentOn),
			}
			h.mu.Unlock()
			w := hideSwitchRecoveryManager(t, h.b)
			reached, resume := make(chan struct{}), make(chan struct{})
			var once sync.Once
			open := func() { once.Do(func() { close(resume) }) }
			t.Cleanup(open)
			base := h.engine.answer
			var reconciles atomic.Int32
			h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == settings.SwitchReconcileMethod {
					n := reconciles.Add(1)
					result, errText := base(method, raw)
					if n == 1 && !duringDrain {
						// The manager already observed committed On, before Off.
						close(reached)
						<-resume
					}
					return result, errText
				}
				if method == settings.SwitchStepMethod {
					var p settings.SwitchStepRequest
					if err := json.Unmarshal(raw, &p); err != nil {
						return nil, err.Error()
					}
					if p.Step == settings.SwitchStepRelease {
						h.mu.Lock()
						cp := *h.cp
						h.mu.Unlock()
						if cp.Target.Intent != settings.SwitchIntentOff || !cp.Target.TerminationVerified {
							t.Errorf("release before verified target Off: %+v", cp)
						}
					}
				}
				return base(method, raw)
			})
			if duringDrain {
				runtime := h.b.nodePolicyRuntime()
				runtime.mu.Lock()
				runtime.drains[r.Request.SourceEngine] = true
				runtime.mu.Unlock()
				proxyBase := h.proxy.answer
				var drainOnce sync.Once
				h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
					if method == nodepolicy.MethodSetEngineDrain {
						drainOnce.Do(func() { close(reached); <-resume })
					}
					return proxyBase(method, raw)
				})
			}
			done := make(chan bool, 1)
			go func() { done <- h.b.recoverReplacementEngineMgr(w) }()
			requireGateOpens(t, reached, "recovery reconcile/drain barrier")
			// No ordinary stop request is delivered. Recovery owns the veto.
			remote, reserved, id, err := h.b.admitLocalEngineOff(engine)
			if err != nil {
				t.Fatal(err)
			}
			if !remote || !reserved || id != r.RequestID {
				t.Fatalf("recovery Off admission = %v, %v, %q", remote, reserved, id)
			}
			open()
			select {
			case ok := <-done:
				if !ok {
					t.Fatal("recovery did not finish")
				}
			case <-time.After(5 * time.Second):
				t.Fatal("recovery did not return")
			}
			h.mu.Lock()
			cp := *h.cp
			h.mu.Unlock()
			if reconciles.Load() != 2 || cp.Phase != settings.SwitchCheckpointReleased || cp.Target.Intent != settings.SwitchIntentOff || !cp.Target.TerminationVerified || cp.Source.Intent != settings.SwitchIntentOff {
				t.Fatalf("recovered checkpoint = %+v, reconciles = %d", cp, reconciles.Load())
			}
			if got := h.receipt(t, r.RequestID); got.Phase != settings.SwitchFailedInterrupted {
				t.Fatalf("recovery receipt = %+v", got)
			}
		})
	}
	test("source Off during reconcile", "llamacpp", false)
	test("target Off during reconcile", "vllm", false)
	test("source Off during drain release", "llamacpp", true)
}

func hideSwitchRecoveryManager(t *testing.T, b *Broker) *rpcWorker {
	t.Helper()
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	if b.engineMgr == nil {
		t.Fatal("missing test engine manager")
	}
	b.engineMgrRecovering = true
	return b.engineMgr
}

func TestSwitchRecoveryReplaysJournalOffWithoutCheckpoint(t *testing.T) {
	test := func(name string, restart, verify bool) {
		t.Run(name, func(t *testing.T) {
			h := newSwitchHarness(t)
			r := switchStoreReceipt("durable-off")
			r.Caller = "peer"
			r.PayloadHash = settings.SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
			switchStoreAccept(t, h.b.engineSwitch.store, r)
			remote, reserved, id, err := h.b.admitLocalEngineOff(r.Request.SourceEngine)
			if err != nil || !remote || !reserved || id != r.RequestID {
				t.Fatalf("admit Off: %v %v %q %v", remote, reserved, id, err)
			}
			if disk := switchStoreDisk(t, h.b.engineSwitch.store); !disk.LocalObligations[id].SourceOff || disk.LocalObligations[id].Discharged {
				t.Fatalf("undischarged disk obligation: %+v", disk.LocalObligations)
			}
			if restart {
				// A fresh coordinator and store lose every latch/resolved bit.
				path := h.b.engineSwitch.store.path
				h.b.engineSwitch = &engineSwitchCoordinator{store: newEngineSwitchStore(path), changed: make(chan struct{}), resolved: make(map[string]bool)}
			}
			w := hideSwitchRecoveryManager(t, h.b)
			base := h.engine.answer
			var stopped, ready bool
			h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == "engine:stop" {
					var p struct {
						Engine      string `json:"engine"`
						VerifyStop  bool   `json:"verifyStop"`
						OperationID string `json:"switchOperationId"`
					}
					if err := json.Unmarshal(raw, &p); err != nil {
						return nil, err.Error()
					}
					if p.Engine != r.Request.SourceEngine || !p.VerifyStop || p.OperationID != "" {
						return nil, "discharge must be ordinary verified source Off"
					}
					stopped = true
					if !verify {
						return settings.SwitchEngineState{Intent: settings.SwitchIntentOff}, ""
					}
				}
				if method == "engine:switch-recovery-ready" {
					ready = true
					if !stopped || !switchStoreDisk(t, h.b.engineSwitch.store).LocalObligations[id].Discharged {
						t.Error("readiness preceded durable Off discharge")
					}
				}
				return base(method, raw)
			})
			ok := h.b.recoverReplacementEngineMgr(w)
			if ok != verify || ready != verify || !stopped {
				t.Fatalf("recover=%v ready=%v stopped=%v", ok, ready, stopped)
			}
			o := switchStoreDisk(t, h.b.engineSwitch.store).LocalObligations[id]
			if o.Discharged != verify {
				t.Fatalf("discharge after recovery: %+v", o)
			}
			if !verify && (h.b.getEngineMgr() != nil || h.b.engineSwitch.resolved[id]) {
				t.Fatal("unverified Off released recovery ownership")
			}
		})
	}
	test("manager replacement before cleanup", false, true)
	test("broker restart before cleanup", true, true)
	test("unverified stop retains recovery", true, false)
}

func TestSwitchVetoJournalWriteFailureRefusesLocalOff(t *testing.T) {
	h := newSwitchHarness(t)
	r := switchStoreReceipt("off-write-failure")
	r.Caller = "peer"
	r.PayloadHash = settings.SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
	switchStoreAccept(t, h.b.engineSwitch.store, r)
	h.b.engineSwitch.store.write = func(string, []byte, os.FileMode) error { return errors.New("journal write refused") }
	id := json.RawMessage(`190`)
	h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:stop", Params: settingsJSON(map[string]string{"engine": r.Request.SourceEngine})})
	var refused bool
	for _, frame := range h.output.frames() {
		if frame.ID != nil && string(*frame.ID) == "190" && frame.Error != nil {
			refused = true
		}
	}
	if !refused || h.b.engineSwitch.localVeto != nil || len(h.engine.methods("engine:stop")) != 0 {
		t.Fatal("failed durable admission latched or dispatched Off, or hid the refusal")
	}
	if len(switchStoreDisk(t, h.b.engineSwitch.store).LocalObligations) != 0 {
		t.Fatal("failed write changed durable obligations")
	}
}

func TestSwitchRecoveryRetriesInspectionBeforePublishingManager(t *testing.T) {
	h := newSwitchHarness(t)
	w := hideSwitchRecoveryManager(t, h.b)
	base := h.engine.answer
	var inspections atomic.Int32
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			if h.b.getEngineMgr() != nil {
				t.Error("manager published before recovery inspection succeeded")
			}
			if inspections.Add(1) == 1 {
				return nil, "temporary inspection failure"
			}
		}
		return base(method, raw)
	})
	if !h.b.recoverReplacementEngineMgr(w) {
		t.Fatal("transient inspection failure did not recover")
	}
	if inspections.Load() != 2 || h.b.getEngineMgr() != w {
		t.Fatalf("inspections = %d, manager = %p", inspections.Load(), h.b.getEngineMgr())
	}
	if h.b.engineSwitch.err != nil || h.b.engineSwitch.recoveryErr != nil {
		t.Fatalf("recovery retained errors: permanent %v, retryable %v", h.b.engineSwitch.err, h.b.engineSwitch.recoveryErr)
	}
	if steps := switchSteps(t, h.engine); len(steps) != 0 {
		t.Fatalf("inspection recovery executed forward steps: %v", steps)
	}
}

func TestSwitchRecoveryRetriesDrainReleaseWithoutRestart(t *testing.T) {
	h := newSwitchHarness(t)
	w := hideSwitchRecoveryManager(t, h.b)
	r := switchStoreReceipt("retry-drain-release")
	switchStoreAccept(t, h.b.engineSwitch.store, r)
	runtime := h.b.nodePolicyRuntime()
	runtime.mu.Lock()
	runtime.drains[r.Request.SourceEngine] = true
	runtime.mu.Unlock()
	base := h.proxy.answer
	var releases atomic.Int32
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetEngineDrain {
			var p nodepolicy.SetEngineDrainParams
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if !p.Drain {
				if h.b.getEngineMgr() != nil {
					t.Error("manager published before drain release acknowledgement")
				}
				if releases.Add(1) == 1 {
					return nil, "proxy restarting"
				}
			}
		}
		return base(method, raw)
	})
	if !h.b.recoverReplacementEngineMgr(w) {
		t.Fatal("transient drain release failure did not recover")
	}
	if releases.Load() != 2 || h.b.getEngineMgr() != w {
		t.Fatalf("releases = %d, manager = %p", releases.Load(), h.b.getEngineMgr())
	}
	if got := h.receipt(t, r.RequestID); got.Phase != settings.SwitchFailedInterrupted || !got.RecoveryRequired {
		t.Fatalf("interrupted receipt = %+v", got)
	}
	if !h.b.engineSwitch.resolved[r.RequestID] || h.b.engineSwitch.recoveryErr != nil {
		t.Fatal("successful retry did not resolve conservative cleanup")
	}
	if steps := switchSteps(t, h.engine); len(steps) != 0 {
		t.Fatalf("drain recovery executed forward steps: %v", steps)
	}
}

func TestSwitchRecoveryExhaustionKeepsManagerHiddenAndReportsBlocked(t *testing.T) {
	h := newSwitchHarness(t)
	w := hideSwitchRecoveryManager(t, h.b)
	var inspections atomic.Int32
	h.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			inspections.Add(1)
		}
		return nil, "inspection unavailable"
	})
	if h.b.recoverReplacementEngineMgr(w) || h.b.getEngineMgr() != nil {
		t.Fatal("exhausted recovery exposed the manager")
	}
	if inspections.Load() != 3 || h.b.engineSwitch.err != nil || h.b.engineSwitch.recoveryErr == nil {
		t.Fatalf("inspections = %d, permanent = %v, retryable = %v", inspections.Load(), h.b.engineSwitch.err, h.b.engineSwitch.recoveryErr)
	}
	h.b.stickyErrorsMu.Lock()
	report, ok := h.b.stickyErrors[engineSwitchRecoveryBlockedID]
	h.b.stickyErrorsMu.Unlock()
	if !ok || report.Severity != "error" || report.Message == "" {
		t.Fatalf("blocked recovery report = %+v, found = %v", report, ok)
	}
}

func TestSwitchRecoveryNeverClearsJournalCorruption(t *testing.T) {
	h := newSwitchHarness(t)
	w := hideSwitchRecoveryManager(t, h.b)
	corruption := &engineSwitchRecoveryError{Err: errors.New("invalid journal")}
	h.b.engineSwitch.err = corruption
	if h.b.recoverReplacementEngineMgr(w) || h.b.getEngineMgr() != nil {
		t.Fatal("journal corruption exposed the manager")
	}
	if !errors.Is(h.b.engineSwitch.err, corruption) {
		t.Fatalf("journal corruption was replaced: %v", h.b.engineSwitch.err)
	}
	if calls := h.engine.snapshot(); len(calls) != 0 {
		t.Fatalf("corrupt journal allowed recovery RPCs: %v", calls)
	}
}

func TestSwitchRecoveryNeverRetriesManagerStorageUncertainty(t *testing.T) {
	h := newSwitchHarness(t)
	w := hideSwitchRecoveryManager(t, h.b)
	var inspections atomic.Int32
	h.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			inspections.Add(1)
		}
		return settings.SwitchPrepareResult{Error: &settings.SwitchError{
			Code: settings.SwitchOperationErrorCode, Kind: settings.SwitchErrorStorage, Message: "checkpoint durability uncertain",
		}}, ""
	})
	if h.b.recoverReplacementEngineMgr(w) {
		t.Fatal("manager storage uncertainty was accepted")
	}
	if !switchRecoveryPermanent(h.b.engineSwitch.err) || inspections.Load() != 1 {
		t.Fatalf("permanent = %v, inspections = %d", h.b.engineSwitch.err, inspections.Load())
	}
	if h.b.recoverReplacementEngineMgr(w) || inspections.Load() != 1 || h.b.getEngineMgr() != nil {
		t.Fatal("subsequent recovery retried or cleared storage uncertainty")
	}
}

func TestSwitchRecoveryRetryDoesNotFollowReplacedManager(t *testing.T) {
	h := newSwitchHarness(t)
	w := hideSwitchRecoveryManager(t, h.b)
	peer, replacement := newFakeWorkerPipe(t)
	newWorker := &rpcWorker{peer: peer}
	var inspections atomic.Int32
	h.engine.setAnswer(func(_ string, _ json.RawMessage) (any, string) {
		inspections.Add(1)
		h.b.setEngineMgr(newWorker)
		return nil, "old manager disconnected"
	})
	if h.b.recoverReplacementEngineMgr(w) {
		t.Fatal("obsolete recovery published a replacement")
	}
	if inspections.Load() != 1 || len(replacement.snapshot()) != 0 || h.b.getEngineMgr() != nil {
		t.Fatalf("old inspections = %d, replacement calls = %v, published = %p", inspections.Load(), replacement.snapshot(), h.b.getEngineMgr())
	}
}

func TestSwitchRecoveryStorageFailureSurvivesDrainReleaseFailure(t *testing.T) {
	h := newSwitchHarness(t)
	h.publicationFailure = true
	p := h.request(t, "storage-and-drain")
	baseEngine := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == settings.SwitchReconcileMethod {
			return settings.SwitchReconcileResult{Error: &settings.SwitchError{Code: settings.SwitchOperationErrorCode, Kind: settings.SwitchErrorStorage, Message: "checkpoint durability uncertain"}}, ""
		}
		return baseEngine(method, raw)
	})
	baseProxy := h.proxy.answer
	var drainReleases atomic.Int32
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetEngineDrain {
			var p nodepolicy.SetEngineDrainParams
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if !p.Drain {
				drainReleases.Add(1)
				return nil, "proxy temporarily unavailable"
			}
		}
		return baseProxy(method, raw)
	})
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	r := h.receipt(t, p.RequestID)
	if !r.RecoveryRequired || !switchRecoveryPermanent(h.b.engineSwitch.err) || drainReleases.Load() != 1 {
		t.Fatalf("lost storage fence: receipt %+v, error %v, drain failures %d", r, h.b.engineSwitch.err, drainReleases.Load())
	}
	// Even after the proxy/manager would answer successfully, the observed
	// storage uncertainty must prevent any later recovery RPC or publication.
	h.engine.setAnswer(baseEngine)
	h.proxy.setAnswer(baseProxy)
	before := len(h.engine.snapshot())
	w := hideSwitchRecoveryManager(t, h.b)
	if h.b.recoverReplacementEngineMgr(w) || len(h.engine.snapshot()) != before || h.b.getEngineMgr() != nil {
		t.Fatal("later recovery retried past permanent storage uncertainty")
	}
}

func TestSwitchRecoveryClassifiesEveryJoinedError(t *testing.T) {
	storage := switchError(settings.SwitchErrorStorage, "checkpoint storage failure")
	transient := switchError(settings.SwitchErrorDrain, "proxy drain failure")
	for _, scenario := range []struct {
		name string
		err  error
	}{
		{"storage first", errors.Join(storage, transient)},
		{"storage after another RPC error", errors.Join(transient, storage)},
		{"wrapped nested storage", fmt.Errorf("cleanup: %w", errors.Join(transient, errors.Join(errors.New("transport"), storage)))},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			if !switchRecoveryPermanent(scenario.err) {
				t.Fatalf("storage uncertainty classified retryable: %v", scenario.err)
			}
		})
	}
	if switchRecoveryPermanent(errors.Join(transient, errors.New("transport"))) {
		t.Fatal("operational errors classified permanent")
	}
}

func TestSwitchRecoveryRetainsVetoUntilManagerAdmission(t *testing.T) {
	test := func(name string, forgetLatch bool) {
		t.Run(name, func(t *testing.T) {
			h := newSwitchHarness(t)
			clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
			p := h.request(t, "retry-latched-veto")
			reached, release := holdSwitchCompletion(t, h)
			base := h.engine.answer
			var reconciles atomic.Int32
			h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == settings.SwitchReconcileMethod {
					if reconciles.Add(1) == 1 {
						return nil, "veto delivery temporarily unavailable"
					}
					var cleanup settings.SwitchReconcileRequest
					if err := json.Unmarshal(raw, &cleanup); err != nil {
						return nil, err.Error()
					}
					want := settings.SwitchLocalVeto{Engine: "llamacpp"}
					if cleanup.OperationID != p.RequestID || cleanup.LocalVeto == nil || *cleanup.LocalVeto != want {
						t.Errorf("recovery lost operation veto: %+v", cleanup)
					}
				}
				return base(method, raw)
			})
			if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
				t.Fatal(err)
			}
			requireGateOpens(t, reached, "post-publication barrier")
			if remote, reserved, _, err := h.b.admitLocalEngineOff("llamacpp"); err != nil || !remote || !reserved {
				t.Fatalf("Off admission: remote %v, reserved %v", remote, reserved)
			}
			release()
			r := h.receipt(t, p.RequestID)
			if !r.RecoveryRequired || r.Error == nil || r.Error.Kind != settings.SwitchErrorRemoteVeto {
				t.Fatalf("unacknowledged veto receipt = %+v", r)
			}
			h.mu.Lock()
			committed := h.cp.TargetCommitted && h.cp.Phase != settings.SwitchCheckpointReleased
			h.mu.Unlock()
			if !committed {
				t.Fatal("failed delivery did not retain committed target reservation")
			}
			if forgetLatch {
				// Model loss of the in-memory latch on broker restart. Recovery
				// must still honor the durable veto failure receipt.
				h.b.workersMu.Lock()
				h.b.engineSwitch.localVeto = nil
				h.b.engineSwitch.vetoID = ""
				h.b.workersMu.Unlock()
			}
			w := hideSwitchRecoveryManager(t, h.b)
			if !h.b.recoverReplacementEngineMgr(w) {
				t.Fatal("veto recovery failed")
			}
			h.mu.Lock()
			cp := *h.cp
			h.mu.Unlock()
			if reconciles.Load() != 2 || cp.Phase != settings.SwitchCheckpointReleased || cp.Target.Intent != settings.SwitchIntentOff || !cp.Target.TerminationVerified || cp.Source.Intent != settings.SwitchIntentOff {
				t.Fatalf("recovered veto checkpoint = %+v; reconcile calls = %d", cp, reconciles.Load())
			}
		})
	}
	test("same broker retains exact veto", false)
	test("durable receipt retains veto after latch loss", true)
}
