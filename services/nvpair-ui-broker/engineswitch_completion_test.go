// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"

	"nvpair-shared/clustertrusttest"
	"nvpair-shared/durablefile"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

// Hold the source drain-release acknowledgement after target facade publication.
// The fake still reports successful acknowledgement when the barrier opens.
func holdSwitchCompletion(t *testing.T, h *switchHarness) (<-chan struct{}, func()) {
	t.Helper()
	reached, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	open := func() { once.Do(func() { close(release) }) }
	t.Cleanup(open)
	base := h.proxy.answer
	var reachedOnce sync.Once
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetEngineDrain {
			var p nodepolicy.SetEngineDrainParams
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if !p.Drain {
				reachedOnce.Do(func() { close(reached) })
				<-release
			}
		}
		return base(method, raw)
	})
	return reached, open
}

func TestSwitchLocalOffAfterPublicationFailsBeforeStopReply(t *testing.T) {
	h := newSwitchHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	p := h.request(t, "published-off")
	reached, release := holdSwitchCompletion(t, h)
	vetoDurable := make(chan struct{})
	vetoPath := filepath.Join(t.TempDir(), "manager-checkpoint.json")
	base := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:stop" {
			var stop struct {
				Engine            string `json:"engine"`
				SwitchOperationID string `json:"switchOperationId"`
			}
			if err := json.Unmarshal(raw, &stop); err != nil {
				return nil, err.Error()
			}
			if stop.Engine != "lmstudio" || stop.SwitchOperationID != p.RequestID {
				t.Errorf("stop lost admitted switch identity: %+v", stop)
			}
			// Model admission's durable checkpoint/intent update separately
			// from the lifecycle stop response, which remains held.
			h.mu.Lock()
			h.cp.Target = switchTestState(false, false, settings.SwitchIntentOff)
			h.cp.Error = &settings.SwitchError{Code: settings.SwitchRemoteVetoCode, Kind: settings.SwitchErrorRemoteVeto, Message: "Local Off veto."}
			data, err := json.Marshal(h.cp)
			if err == nil {
				err = durablefile.WriteFile(vetoPath, data, 0600)
			}
			h.mu.Unlock()
			if err != nil {
				return nil, err.Error()
			}
			close(vetoDurable)
			return nil, holdReply
		}
		return base(method, raw)
	})
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
		t.Fatal(err)
	}
	requireGateOpens(t, reached, "published target waiting for drain release")
	stopped := make(chan struct{})
	go func() {
		id := json.RawMessage(`101`)
		h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:stop", Params: settingsJSON(map[string]string{"engine": "lmstudio"})})
		close(stopped)
	}()
	requireGateOpens(t, vetoDurable, "durable Off before stop reply")
	data, err := os.ReadFile(vetoPath)
	if err != nil {
		t.Fatal(err)
	}
	var checkpoint settings.SwitchCheckpoint
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		t.Fatal(err)
	}
	if checkpoint.Target.Intent != settings.SwitchIntentOff || checkpoint.Error == nil || checkpoint.Error.Kind != settings.SwitchErrorRemoteVeto {
		t.Fatalf("persisted manager veto = %+v", checkpoint)
	}
	release()
	r := h.receipt(t, p.RequestID)
	if !r.Terminal || r.Phase != settings.SwitchFailedPublication || r.Error == nil || r.Error.Kind != settings.SwitchErrorRemoteVeto || r.Target.Intent != settings.SwitchIntentOff || r.Source.Intent != settings.SwitchIntentOff {
		t.Fatalf("post-publication veto receipt = %+v", r)
	}
	select {
	case <-stopped:
		t.Fatal("stop response was not held across terminal persistence")
	default:
	}
	h.engine.releaseHeld()
	requireGateOpens(t, stopped, "stop waiter cleanup")
}

func TestSwitchManagerDoneBeforeSupervisorCallbackFailsInterrupted(t *testing.T) {
	h := newSwitchHarness(t)
	p := h.request(t, "manager-done")
	w := h.b.getEngineMgr()
	w.done = make(chan struct{})
	reached, release := holdSwitchCompletion(t, h)
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	requireGateOpens(t, reached, "completion barrier")
	// Deliberately omit setEngineMgr(nil) and the supervisor callback.
	close(w.done)
	release()
	r := h.receipt(t, p.RequestID)
	if r.Phase != settings.SwitchFailedInterrupted || !r.RecoveryRequired || r.Error == nil || r.Error.Kind != settings.SwitchErrorInterrupted {
		t.Fatalf("already-dead manager receipt = %+v", r)
	}
	if got := switchSteps(t, h.engine); len(got) != 2 {
		t.Fatalf("interrupted operation released manager reservation: %v", got)
	}
	peer, replacement := newFakeWorkerPipe(t)
	replacement.setAnswer(h.engine.answer)
	next := &rpcWorker{peer: peer}
	h.b.workersMu.Lock()
	h.b.engineMgrRecovering = true
	h.b.workersMu.Unlock()
	h.b.setEngineMgr(next)
	if h.b.getEngineMgr() != nil {
		t.Fatal("replacement exposed before conservative recovery")
	}
	if !h.b.recoverReplacementEngineMgr(next) {
		t.Fatal("replacement recovery failed")
	}
	steps := switchSteps(t, replacement)
	if len(steps) != 1 || steps[0] != settings.SwitchStepRelease {
		t.Fatalf("replacement resumed forward work: %v", steps)
	}
}

func TestSwitchExitOwnerFencesBeforeSupervisorClearsHandle(t *testing.T) {
	h := newSwitchHarness(t)
	p := h.request(t, "exit-owner")
	reached, release := holdSwitchCompletion(t, h)
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	requireGateOpens(t, reached, "completion barrier")
	w := h.b.getEngineMgr()
	h.b.engineManagerExited(w)
	if h.b.getEngineMgr() != nil {
		t.Fatal("exit owner left the dead manager visible")
	}
	release()
	if r := h.receipt(t, p.RequestID); r.Phase != settings.SwitchFailedInterrupted || !r.RecoveryRequired {
		t.Fatalf("exit-owner fence receipt = %+v", r)
	}
}

func TestSwitchOffAdmissionRechecksAfterNegativeReservationLookup(t *testing.T) {
	h := newSwitchHarness(t)
	// Ordinary settings work owns the operation lock when Off first sees no
	// reservation. The journal reader is a barrier in that real lookup.
	h.b.settingsApplyMu.Lock()
	var unlockOnce sync.Once
	unlock := func() { unlockOnce.Do(h.b.settingsApplyMu.Unlock) }
	defer unlock()
	negative := make(chan struct{})
	c := h.b.switchCoordinator()
	c.store.loaded = false
	read := c.store.read
	c.store.read = func(path string) ([]byte, error) {
		data, err := read(path)
		close(negative)
		return data, err
	}
	stopDelivered := make(chan struct{})
	base := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:stop" {
			close(stopDelivered)
		}
		return base(method, raw)
	})
	done := make(chan struct{})
	go func() {
		id := json.RawMessage(`102`)
		h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:stop", Params: settingsJSON(map[string]string{"engine": "vllm"})})
		close(done)
	}()
	requireGateOpens(t, negative, "negative reservation lookup")
	// Install accepted remote work before Off may obtain the operation lock.
	// This exercises the waiting iteration without a production timing hook.
	h.b.engineConfigMu.Lock()
	r := switchStoreReceipt("off-admission")
	r.Caller = "peer"
	r.PayloadHash = settings.SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
	stored := switchStoreAccept(t, c.store, r)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	h.b.workersMu.Lock()
	c.activeID = r.RequestID
	h.b.workersMu.Unlock()
	c.waitMu.Lock()
	c.remoteSource, c.remoteTarget, c.remoteCancel = r.Request.SourceEngine, r.Request.TargetEngine, cancel
	c.waitMu.Unlock()
	h.b.switchChangedLocked(stored)
	h.b.engineConfigMu.Unlock()
	requireGateOpens(t, stopDelivered, "Off delivered while switch owns operation lock")
	requireGateOpens(t, done, "local Off response")
	if ctx.Err() == nil {
		t.Fatal("Off did not cancel the newly accepted remote switch")
	}
	unlock()
}

func TestSwitchOffNegativeLookupAndOperationAdmissionAreOrdered(t *testing.T) {
	h := newSwitchHarness(t)
	p := h.request(t, "off-ordered")
	remote, reserved, _ := h.b.admitLocalEngineOff("lmstudio")
	defer h.b.settingsApplyMu.Unlock()
	if remote || reserved {
		t.Fatalf("unexpected Off reservation: remote %v, reserved %v", remote, reserved)
	}
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err == nil || switchErrorDetail(err, "").Kind != settings.SwitchErrorBusy {
		t.Fatalf("switch passed Off's operation admission: %v", err)
	}
}

func TestSwitchLatchedOffCleanupPrecedesLocalStopAdmission(t *testing.T) {
	for _, engine := range []string{"llamacpp", "lmstudio"} {
		t.Run(engine, func(t *testing.T) {
			h := newSwitchHarness(t)
			clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
			p := h.request(t, "latched-off")
			reached, release := holdSwitchCompletion(t, h)
			base := h.engine.answer
			h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == settings.SwitchReconcileMethod {
					var cleanup settings.SwitchReconcileRequest
					if err := json.Unmarshal(raw, &cleanup); err != nil {
						return nil, err.Error()
					}
					if cleanup.OperationID != p.RequestID || cleanup.LocalVeto == nil || cleanup.LocalVeto.Engine != engine || cleanup.LocalVeto.Pause {
						t.Errorf("cleanup did not carry the admitted operation/Off: %+v", cleanup)
					}
				}
				return base(method, raw)
			})
			if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
				t.Fatal(err)
			}
			requireGateOpens(t, reached, "post-publication barrier")
			// Pause the local handler at its actual admission boundary, before
			// dispatching any engine:stop to the manager.
			remote, reserved, operationID := h.b.admitLocalEngineOff(engine)
			if !remote || !reserved || operationID != p.RequestID {
				t.Fatalf("Off admission = %v, %v, %q", remote, reserved, operationID)
			}
			release()
			r := h.receipt(t, p.RequestID)
			if r.Phase != settings.SwitchFailedPublication || r.Error == nil || r.Error.Kind != settings.SwitchErrorRemoteVeto || r.RecoveryRequired || !r.Target.TerminationVerified || r.Target.Intent != settings.SwitchIntentOff || r.Source.Intent != settings.SwitchIntentOff {
				t.Fatalf("latched-veto terminal = %+v", r)
			}
			if stops := h.engine.methods("engine:stop"); len(stops) != 0 {
				t.Fatalf("cleanup depended on local stop delivery: %v", stops)
			}
			h.mu.Lock()
			cp := *h.cp
			h.mu.Unlock()
			if cp.Phase != settings.SwitchCheckpointReleased || cp.Target.Intent != settings.SwitchIntentOff || !cp.Target.TerminationVerified {
				t.Fatalf("manager released without verified target Off: %+v", cp)
			}
			// Deliver the held lifecycle request only after terminal cleanup and
			// release. Its original scope makes it harmless to newer intent.
			w := h.b.getEngineMgr()
			if _, rpcErr, err := w.CallNoTimeout(context.Background(), "engine:stop", settingsJSON(map[string]string{"engine": engine, "switchOperationId": operationID})); err != nil || rpcErr != nil {
				t.Fatalf("late scoped stop: %v, %+v", err, rpcErr)
			}
			h.b.cancelRemoteEngineSwitchOperation(engine, operationID)
		})
	}
}

func TestSwitchDrainFailureReconcilesReservedSourceOffBeforeDelayedStop(t *testing.T) {
	h := newSwitchHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	p := h.request(t, "reserved-source-off")
	reached, resume := make(chan struct{}), make(chan struct{})
	var once sync.Once
	open := func() { once.Do(func() { close(resume) }) }
	t.Cleanup(open)
	proxyAnswer := h.proxy.answer
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodAdmissionSnapshot {
			close(reached)
			<-resume
			return nil, "drain admission evidence rejected"
		}
		return proxyAnswer(method, raw)
	})
	engineAnswer := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == settings.SwitchReconcileMethod {
			var cleanup settings.SwitchReconcileRequest
			if err := json.Unmarshal(raw, &cleanup); err != nil {
				return nil, err.Error()
			}
			if cleanup.OperationID != p.RequestID || cleanup.LocalVeto == nil || cleanup.LocalVeto.Engine != "llamacpp" {
				t.Errorf("reserved cleanup lost admitted Off: %+v", cleanup)
			}
			h.mu.Lock()
			if h.cp == nil || h.cp.Phase != settings.SwitchCheckpointReserved || h.cp.Source.Running == nil || !*h.cp.Source.Running {
				t.Errorf("cleanup must begin with running reserved source: %+v", h.cp)
			} else {
				// The manager's physical-stop contract is exercised separately
				// by TestSwitchReservedSourceVetoStopsBeforeDelayedScopedStop.
				h.cp.Source = switchTestState(false, false, settings.SwitchIntentOff)
			}
			h.mu.Unlock()
		}
		return engineAnswer(method, raw)
	})
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
		t.Fatal(err)
	}
	requireGateOpens(t, reached, "drain before source-stop")
	remote, reserved, operationID := h.b.admitLocalEngineOff("llamacpp")
	if !remote || !reserved || operationID != p.RequestID {
		t.Fatalf("source Off admission = %v, %v, %q", remote, reserved, operationID)
	}
	open()
	r := h.receipt(t, p.RequestID)
	if r.Phase != settings.SwitchFailedDrain || r.RecoveryRequired || !r.Source.TerminationVerified || r.Source.Intent != settings.SwitchIntentOff {
		t.Fatalf("drain failure source Off terminal = %+v", r)
	}
	for _, step := range switchSteps(t, h.engine) {
		if step != settings.SwitchStepRelease {
			t.Fatalf("failed drain advanced forward step: %s", step)
		}
	}
	w := h.b.getEngineMgr()
	if _, rpcErr, err := w.CallNoTimeout(context.Background(), "engine:stop", settingsJSON(map[string]string{"engine": "llamacpp", "switchOperationId": operationID})); err != nil || rpcErr != nil {
		t.Fatalf("held stop after terminal/release: %v, %+v", err, rpcErr)
	}
}

func TestSwitchVetoDuringReconcileRepeatsCleanupBeforeTerminal(t *testing.T) {
	h := newSwitchHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	h.publicationFailure = true
	p := h.request(t, "veto-during-cleanup")
	read, resume := make(chan struct{}), make(chan struct{})
	var once sync.Once
	open := func() { once.Do(func() { close(resume) }) }
	t.Cleanup(open)
	base := h.engine.answer
	var reconciles atomic.Int32
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, errText := base(method, raw)
		if method == settings.SwitchReconcileMethod && reconciles.Add(1) == 1 {
			close(read)
			<-resume
		}
		return result, errText
	})
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
		t.Fatal(err)
	}
	requireGateOpens(t, read, "reconcile read before veto")
	if remote, reserved, _ := h.b.admitLocalEngineOff("llamacpp"); !remote || !reserved {
		t.Fatalf("Off admission: remote %v, reserved %v", remote, reserved)
	}
	open()
	r := h.receipt(t, p.RequestID)
	if r.Phase != settings.SwitchFailedPublication || r.RecoveryRequired || r.Target.Intent != settings.SwitchIntentOff || !r.Target.TerminationVerified || reconciles.Load() != 2 {
		t.Fatalf("late-latch terminal = %+v; reconcile calls = %d", r, reconciles.Load())
	}
}

func TestSwitchLatchedPauseReconcileCleansCommittedTarget(t *testing.T) {
	h := newSwitchHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	p := h.request(t, "latched-pause")
	reached, release := holdSwitchCompletion(t, h)
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
		t.Fatal(err)
	}
	requireGateOpens(t, reached, "post-publication barrier")
	// Pause's latch precedes projection delivery in the real availability path.
	h.b.vetoRemoteEngineSwitch("", false)
	release()
	r := h.receipt(t, p.RequestID)
	if r.Phase != settings.SwitchFailedPublication || r.RecoveryRequired || r.Target.Intent != settings.SwitchIntentOff || !r.Target.TerminationVerified {
		t.Fatalf("latched pause terminal = %+v", r)
	}
}

func TestSwitchLateOffResponseDoesNotCancelLaterOperation(t *testing.T) {
	h := newSwitchHarness(t)
	c := h.b.switchCoordinator()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c.activeID = "later-switch"
	c.remoteCancel, c.remoteSource, c.remoteTarget = cancel, "llamacpp", "lmstudio"
	h.b.cancelRemoteEngineSwitchOperation("lmstudio", "released-switch")
	if ctx.Err() != nil || c.localVeto != nil {
		t.Fatal("late Off response vetoed a later switch")
	}
	h.b.cancelRemoteEngineSwitchOperation("lmstudio", "later-switch")
	if ctx.Err() == nil {
		t.Fatal("matching Off response did not cancel its operation")
	}
}
