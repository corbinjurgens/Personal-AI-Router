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
				VerifyStop        bool   `json:"verifyStop"`
			}
			if err := json.Unmarshal(raw, &stop); err != nil {
				return nil, err.Error()
			}
			if stop.VerifyStop {
				return base(method, raw)
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
	remote, reserved, _, err := h.b.admitLocalEngineOff("lmstudio")
	if err != nil {
		t.Fatal(err)
	}
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
			remote, reserved, operationID, err := h.b.admitLocalEngineOff(engine)
			if err != nil {
				t.Fatal(err)
			}
			if !remote || !reserved || operationID != p.RequestID {
				t.Fatalf("Off admission = %v, %v, %q", remote, reserved, operationID)
			}
			release()
			r := h.receipt(t, p.RequestID)
			if r.Phase != settings.SwitchFailedPublication || r.Error == nil || r.Error.Kind != settings.SwitchErrorRemoteVeto || r.RecoveryRequired || !r.Target.TerminationVerified || r.Target.Intent != settings.SwitchIntentOff || r.Source.Intent != settings.SwitchIntentOff {
				t.Fatalf("latched-veto terminal = %+v", r)
			}
			if stops := h.engine.methods("engine:stop"); len(stops) != 1 {
				t.Fatalf("broker must discharge the Off before held delivery: %v", stops)
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
	remote, reserved, operationID, err := h.b.admitLocalEngineOff("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
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

func TestSwitchPrepareRejectedByOlderPeerStartDischargesAdmittedOff(t *testing.T) {
	h := newSwitchHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	p := h.request(t, "older-peer-start-off")
	prepare, reject, stop, peerSettled := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{})
	var rejectOnce, peerOnce sync.Once
	openPrepare := func() { rejectOnce.Do(func() { close(reject) }) }
	settlePeer := func() { peerOnce.Do(func() { close(peerSettled) }) }
	t.Cleanup(openPrepare)
	t.Cleanup(settlePeer)
	base := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			var request settings.SwitchPrepareRequest
			if err := json.Unmarshal(raw, &request); err != nil {
				return nil, err.Error()
			}
			if request.OperationID != "" {
				close(prepare)
				<-reject
				return settings.SwitchPrepareResult{Error: &settings.SwitchError{Code: settings.SwitchOperationErrorCode, Kind: settings.SwitchErrorBusy, Message: "older peer start still active"}}, ""
			}
		}
		if method == settings.SwitchReconcileMethod {
			return settings.SwitchReconcileResult{Observed: settings.SwitchObserved{Source: switchTestState(true, true, settings.SwitchIntentOn), Target: switchTestState(false, false, settings.SwitchIntentOff)}, Error: &settings.SwitchError{Code: settings.SwitchInvalidParamsCode, Kind: settings.SwitchErrorInvalidRequest, Message: "no checkpoint"}}, ""
		}
		if method == "engine:stop" {
			var request struct {
				OperationID string `json:"switchOperationId"`
				VerifyStop  bool   `json:"verifyStop"`
			}
			if err := json.Unmarshal(raw, &request); err != nil {
				return nil, err.Error()
			}
			if request.OperationID != "" || !request.VerifyStop {
				return nil, "expected broker-owned ordinary discharge"
			}
			close(stop)
			<-peerSettled
		}
		return base(method, raw)
	})
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
		t.Fatal(err)
	}
	requireGateOpens(t, prepare, "prepare overlapping older peer start")
	remote, reserved, id, err := h.b.admitLocalEngineOff("llamacpp")
	if err != nil || !remote || !reserved || id != p.RequestID {
		t.Fatalf("Off admission: %v %v %s %v", remote, reserved, id, err)
	}
	openPrepare()
	requireGateOpens(t, stop, "ordinary discharge waiting for older start")
	if disk := switchStoreDisk(t, h.b.engineSwitch.store); disk.Receipts[id].Terminal || disk.LocalObligations[id].Discharged {
		t.Fatal("operation released before older start settled and Off was verified")
	}
	settlePeer()
	r := h.receipt(t, id)
	if r.Phase != settings.SwitchFailedValidation || r.RecoveryRequired || r.Source.Intent != settings.SwitchIntentOff || !r.Source.TerminationVerified {
		t.Fatalf("rejected prepare discharge: %+v", r)
	}
	if disk := switchStoreDisk(t, h.b.engineSwitch.store); !disk.LocalObligations[id].Discharged {
		t.Fatal("terminal preceded durable discharge")
	}
}

func TestSwitchLocalOnRelayOwnsOperationLock(t *testing.T) {
	for _, scenario := range []struct {
		name, method, params string
		wantLocked           bool
	}{
		{"start without port", "engine:start", `{"engine":"llamacpp"}`, true},
		{"restart", "engine:restart", `{"engine":"llamacpp"}`, true},
		{"install and start without port", "engine:install", `{"engine":"llamacpp","start":true}`, true},
		{"install and start with port", "engine:install", `{"engine":"llamacpp","start":true,"port":12345}`, true},
		{"install only", "engine:install", `{"engine":"llamacpp"}`, false},
		{"install without start with port", "engine:install", `{"engine":"llamacpp","start":false,"port":12345}`, false},
		{"action preserves intent", "engine:action", `{"engine":"llamacpp","action":"delete-model"}`, false},
		{"remote start targets peer", "engine:remote-start", `{"node":"peer","engine":"llamacpp"}`, false},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			h := newSwitchHarness(t)
			h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				locked := !h.b.settingsApplyMu.TryLock()
				if !locked {
					h.b.settingsApplyMu.Unlock()
				}
				wantLocked := scenario.wantLocked && method != "engine:install"
				if locked != wantLocked {
					t.Errorf("%s operation lock held = %v, want %v", method, locked, wantLocked)
				}
				if !h.b.engineConfigMu.TryLock() {
					t.Error("worker wait holds journal lock")
				} else {
					h.b.engineConfigMu.Unlock()
				}
				return map[string]bool{"ok": true}, ""
			})
			id := json.RawMessage(`110`)
			h.b.relayToEngineNow(&Message{ID: &id, Method: scenario.method, Params: json.RawMessage(scenario.params)})
			if response := h.output.frames(); len(response) != 1 || response[0].Error != nil {
				t.Fatalf("local relay response = %+v", response)
			}
		})
	}
}

func TestSwitchInstallStartBoundaryPreservesNewerOff(t *testing.T) {
	h := newSwitchHarness(t)
	installed, installDone, offDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
	open := h.engine.releaseHeld
	t.Cleanup(open)
	var running atomic.Bool
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:install":
			var request struct {
				Engine string `json:"engine"`
				Start  bool   `json:"start"`
				Port   int    `json:"port"`
			}
			if err := json.Unmarshal(raw, &request); err != nil {
				return nil, err.Error()
			}
			if request.Engine != "llamacpp" || request.Start || request.Port != 0 {
				return nil, "expected preparation-only install"
			}
			close(installed)
			return map[string]bool{"ok": true}, holdSuccessReply
		case "engine:start":
			running.Store(true)
		case "engine:stop":
			running.Store(false)
		}
		return map[string]bool{"ok": true}, ""
	})
	go func() {
		id := json.RawMessage(`111`)
		h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:install", Params: json.RawMessage(`{"engine":"llamacpp","start":true}`)})
		close(installDone)
	}()
	requireGateOpens(t, installed, "install/start boundary")
	if !h.b.settingsApplyMu.TryLock() {
		t.Error("installer holds broker operation lock")
	} else {
		h.b.settingsApplyMu.Unlock()
	}
	go func() {
		id := json.RawMessage(`112`)
		h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:stop", Params: json.RawMessage(`{"engine":"llamacpp"}`)})
		close(offDone)
	}()
	requireGateOpens(t, offDone, "newer Off completion")
	open()
	requireGateOpens(t, installDone, "older install/start completion")
	if running.Load() {
		t.Fatal("older install continuation overrode newer Off")
	}
	for _, response := range h.output.frames() {
		if response.ID != nil && string(*response.ID) == "111" {
			if response.Error == nil || response.Error.Code != -32000 {
				t.Fatalf("superseded install must return busy: %+v", response)
			}
		} else if response.Error != nil {
			t.Fatalf("Off failed: %+v", response)
		}
	}
}

func TestSwitchHeldInstallAllowsOffStatusAndAcceptanceAndPreservesDischarge(t *testing.T) {
	h := newSwitchHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	p := h.request(t, "held-install-discharge")
	installed, installDone, offDone, prepare, resume := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	openPrepare := func() { once.Do(func() { close(resume) }) }
	t.Cleanup(h.engine.releaseHeld)
	t.Cleanup(openPrepare)
	var running atomic.Bool
	running.Store(true)
	base := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:install":
			close(installed)
			return map[string]bool{"ok": true}, holdSuccessReply
		case "engine:start":
			running.Store(true)
		case "engine:stop":
			var request struct {
				Engine string `json:"engine"`
			}
			if err := json.Unmarshal(raw, &request); err != nil {
				return nil, err.Error()
			}
			if request.Engine == "llamacpp" {
				running.Store(false)
			}
		case settings.SwitchPrepareMethod:
			var request settings.SwitchPrepareRequest
			if err := json.Unmarshal(raw, &request); err != nil {
				return nil, err.Error()
			}
			if request.OperationID != "" {
				close(prepare)
				<-resume
				return settings.SwitchPrepareResult{Error: &settings.SwitchError{Code: settings.SwitchOperationErrorCode, Kind: settings.SwitchErrorBusy, Message: "installer mutation active"}}, ""
			}
		case settings.SwitchReconcileMethod:
			return settings.SwitchReconcileResult{Observed: settings.SwitchObserved{Source: switchTestState(true, true, settings.SwitchIntentOn), Target: switchTestState(false, false, settings.SwitchIntentOff)}, Error: &settings.SwitchError{Code: settings.SwitchInvalidParamsCode, Kind: settings.SwitchErrorInvalidRequest, Message: "no checkpoint"}}, ""
		}
		return base(method, raw)
	})
	go func() {
		id := json.RawMessage(`116`)
		h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:install", Params: json.RawMessage(`{"engine":"llamacpp","start":true}`)})
		close(installDone)
	}()
	requireGateOpens(t, installed, "held installer")
	go func() {
		id := json.RawMessage(`117`)
		h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:stop", Params: json.RawMessage(`{"engine":"lmstudio"}`)})
		close(offDone)
	}()
	requireGateOpens(t, offDone, "unrelated Off while installer held")
	accepted := make(chan struct{})
	go func() {
		if _, err := h.b.engineSwitchStatus(context.Background(), settings.SwitchStatusRequest{}, "peer"); err != nil {
			t.Error(err)
		}
		if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
			t.Error(err)
		}
		close(accepted)
	}()
	requireGateOpens(t, accepted, "status and acceptance while installer held")
	requireGateOpens(t, prepare, "prepare during held install")
	if remote, reserved, id, err := h.b.admitLocalEngineOff("llamacpp"); err != nil || !remote || !reserved || id != p.RequestID {
		t.Fatalf("Off admission: %v %v %s %v", remote, reserved, id, err)
	}
	openPrepare()
	r := h.receipt(t, p.RequestID)
	if r.RecoveryRequired || r.Source.Intent != settings.SwitchIntentOff || !r.Source.TerminationVerified {
		t.Fatalf("discharged receipt = %+v", r)
	}
	if disk := switchStoreDisk(t, h.b.engineSwitch.store); !disk.LocalObligations[p.RequestID].Discharged {
		t.Fatal("Off was not durably discharged")
	}
	h.engine.releaseHeld()
	requireGateOpens(t, installDone, "superseded start after discharge")
	if running.Load() {
		t.Fatal("older install start overrode discharged Off")
	}
	for _, response := range h.output.frames() {
		if response.ID != nil && string(*response.ID) == "116" && (response.Error == nil || response.Error.Code != -32000) {
			t.Fatalf("older install response = %+v", response)
		}
	}
}

func TestSwitchInstallStartPreservesOptionsAndStatusResponse(t *testing.T) {
	for _, params := range []string{`{"engine":"llamacpp","start":true}`, `{"engine":"llamacpp","start":true,"port":12345,"bind":"127.0.0.1"}`} {
		t.Run(params, func(t *testing.T) {
			h := newSwitchHarness(t)
			want := json.RawMessage(`{"engine":"llamacpp","running":true,"port":12345,"version":"test"}`)
			calls := 0
			h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				calls++
				var got, original map[string]json.RawMessage
				if err := json.Unmarshal(raw, &got); err != nil {
					return nil, err.Error()
				}
				if err := json.Unmarshal([]byte(params), &original); err != nil {
					return nil, err.Error()
				}
				if method == "engine:install" {
					original["start"] = json.RawMessage(`false`)
				}
				if !reflect.DeepEqual(got, original) {
					t.Errorf("%s params = %s, original = %s", method, raw, params)
				}
				if calls == 1 && method != "engine:install" || calls == 2 && method != "engine:start" {
					t.Errorf("call %d = %s", calls, method)
				}
				if method == "engine:install" {
					return map[string]bool{"preparation": true}, ""
				}
				return want, ""
			})
			id := json.RawMessage(`118`)
			h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:install", Params: json.RawMessage(params)})
			responses := h.output.frames()
			if calls != 2 || len(responses) != 1 || responses[0].Error != nil {
				t.Fatalf("calls=%d responses=%+v", calls, responses)
			}
			var got, expected map[string]any
			if err := json.Unmarshal(responses[0].Result, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(want, &expected); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, expected) {
				t.Fatalf("response=%v want=%v", got, expected)
			}
		})
	}
}

func TestSwitchPrivateExecutorMethodsRejectClientRequests(t *testing.T) {
	for _, method := range []string{settings.SwitchPrepareMethod, settings.SwitchStepMethod, settings.SwitchReconcileMethod} {
		t.Run(method, func(t *testing.T) {
			h := newSwitchHarness(t)
			id := json.RawMessage(`115`)
			h.b.handleMessage(&Message{ID: &id, Method: method, Params: settingsJSON(settings.SwitchStepRequest{Step: settings.SwitchStepTargetStart})})
			response := h.output.frames()
			if len(response) != 1 || response[0].Error == nil || response[0].Error.Code != -32601 {
				t.Fatalf("private executor response = %+v", response)
			}
			if calls := h.engine.snapshot(); len(calls) != 0 {
				t.Fatalf("client reached private switch executor: %+v", calls)
			}
		})
	}
}

func TestSwitchInstallWithStartRefusesRetainedReservation(t *testing.T) {
	h := newSwitchHarness(t)
	// An accepted operation remains fenced after its running coordinator is
	// lost. Install-with-start must apply the same reservation check as start.
	r := switchStoreReceipt("install-reserved")
	switchStoreAccept(t, h.b.engineSwitch.store, r)
	var installed, started atomic.Bool
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:install" {
			installed.Store(true)
		}
		if method == "engine:start" {
			started.Store(true)
		}
		return map[string]bool{"ok": true}, ""
	})
	id := json.RawMessage(`114`)
	h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:install", Params: settingsJSON(map[string]any{"engine": r.Request.SourceEngine, "start": true})})
	if !installed.Load() || started.Load() {
		t.Fatal("expected completed preparation and refused start")
	}
	response := h.output.frames()
	if len(response) != 1 || response[0].Error == nil || response[0].Error.Code != -32000 {
		t.Fatalf("reserved install response = %+v", response)
	}
}

func TestSwitchInstallWithStartAfterAdmittedOffDischargeStaysOn(t *testing.T) {
	h := newSwitchHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	p := h.request(t, "install-after-discharge")
	prepare, resume, installDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	open := func() { once.Do(func() { close(resume) }) }
	t.Cleanup(open)
	var running, discharged atomic.Bool
	running.Store(true)
	base := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			var request settings.SwitchPrepareRequest
			if err := json.Unmarshal(raw, &request); err != nil {
				return nil, err.Error()
			}
			if request.OperationID != "" {
				close(prepare)
				<-resume
				return settings.SwitchPrepareResult{Error: &settings.SwitchError{Code: settings.SwitchOperationErrorCode, Kind: settings.SwitchErrorBusy, Message: "mutation active"}}, ""
			}
		}
		if method == settings.SwitchReconcileMethod {
			return settings.SwitchReconcileResult{Observed: settings.SwitchObserved{Source: switchTestState(true, true, settings.SwitchIntentOn), Target: switchTestState(false, false, settings.SwitchIntentOff)}, Error: &settings.SwitchError{Code: settings.SwitchInvalidParamsCode, Kind: settings.SwitchErrorInvalidRequest, Message: "no checkpoint"}}, ""
		}
		if method == "engine:stop" {
			running.Store(false)
			discharged.Store(true)
		}
		if method == "engine:start" {
			if !discharged.Load() {
				t.Error("install reached manager before admitted Off discharge")
			}
			disk := switchStoreDisk(t, h.b.engineSwitch.store)
			if !disk.LocalObligations[p.RequestID].Discharged || !disk.Receipts[p.RequestID].Terminal {
				t.Error("install overtook durable discharge/terminal")
			}
			running.Store(true)
		}
		return base(method, raw)
	})
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
		t.Fatal(err)
	}
	requireGateOpens(t, prepare, "accepted switch before rejected prepare")
	if remote, reserved, id, err := h.b.admitLocalEngineOff("llamacpp"); err != nil || !remote || !reserved || id != p.RequestID {
		t.Fatalf("Off admission: remote=%v reserved=%v id=%s err=%v", remote, reserved, id, err)
	}
	go func() {
		id := json.RawMessage(`113`)
		h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:install", Params: json.RawMessage(`{"engine":"llamacpp","start":true}`)})
		close(installDone)
	}()
	open()
	requireGateOpens(t, installDone, "install after switch discharge")
	r := h.receipt(t, p.RequestID)
	if r.RecoveryRequired || r.Source.Intent != settings.SwitchIntentOff || !r.Source.TerminationVerified {
		t.Fatalf("discharge receipt = %+v", r)
	}
	if !running.Load() {
		t.Fatal("discharge overrode newer install-with-start")
	}
	for _, response := range h.output.frames() {
		if response.ID != nil && string(*response.ID) == "113" && response.Error != nil {
			t.Fatalf("newer install refused: %+v", response)
		}
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
	if remote, reserved, _, err := h.b.admitLocalEngineOff("llamacpp"); err != nil || !remote || !reserved {
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
	if err := h.b.vetoRemoteEngineSwitch("", false); err != nil {
		t.Fatal(err)
	}
	runtime := h.b.nodePolicyRuntime()
	runtime.mu.Lock()
	runtime.current.Availability, runtime.current.PausedBy = nodepolicy.Paused, nodepolicy.PausedLocally
	runtime.mu.Unlock()
	release()
	r := h.receipt(t, p.RequestID)
	if r.Phase != settings.SwitchFailedPublication || r.RecoveryRequired || r.Target.Intent != settings.SwitchIntentOff || !r.Target.TerminationVerified {
		t.Fatalf("latched pause terminal = %+v", r)
	}
}

func TestSwitchUnappliedPauseRetainsDurableObligation(t *testing.T) {
	h := newSwitchHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	p := h.request(t, "unapplied-pause")
	reached, release := holdSwitchCompletion(t, h)
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
		t.Fatal(err)
	}
	requireGateOpens(t, reached, "post-publication barrier")
	if err := h.b.vetoRemoteEngineSwitch("", false); err != nil {
		t.Fatal(err)
	}
	release()
	r := h.receipt(t, p.RequestID)
	o := switchStoreDisk(t, h.b.engineSwitch.store).LocalObligations[p.RequestID]
	if !r.RecoveryRequired || !o.Pause || o.Discharged {
		t.Fatalf("unapplied pause: receipt=%+v obligation=%+v", r, o)
	}
	for _, step := range switchSteps(t, h.engine) {
		if step == settings.SwitchStepRelease {
			t.Fatal("released before pause applied")
		}
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

func TestSwitchInstallStartSupersededByLocalPause(t *testing.T) {
	h := newSwitchHarness(t)
	installed, installDone := make(chan struct{}), make(chan struct{})
	open := h.engine.releaseHeld
	t.Cleanup(open)
	var started atomic.Bool
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:install":
			close(installed)
			return map[string]bool{"ok": true}, holdSuccessReply
		case "engine:start":
			started.Store(true)
		}
		return map[string]bool{"ok": true}, ""
	})
	go func() {
		id := json.RawMessage(`121`)
		h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:install", Params: json.RawMessage(`{"engine":"llamacpp","start":true}`)})
		close(installDone)
	}()
	requireGateOpens(t, installed, "install/start boundary")

	// Hold the transition so only pause admission runs; the projection it
	// publishes follows the install supersession.
	r := h.b.nodePolicyRuntime()
	r.transitionMu.Lock()
	waitCtx, stopWaiting := context.WithCancel(context.Background())
	pauseDone := make(chan struct{})
	go func() {
		_, _ = h.b.setAvailability(waitCtx, nodepolicy.Paused, nodepolicy.PausedLocally)
		close(pauseDone)
	}()
	deadline := time.Now().Add(time.Second)
	for {
		h.engine.mu.Lock()
		published := len(h.engine.projections) > 0
		h.engine.mu.Unlock()
		if published {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("local pause admission did not publish its projection")
		}
		time.Sleep(5 * time.Millisecond)
	}
	stopWaiting()
	r.transitionMu.Unlock()
	requireGateOpens(t, pauseDone, "pause caller return")

	open()
	requireGateOpens(t, installDone, "older install/start completion")
	if started.Load() {
		t.Fatal("install admitted before a local pause started its engine")
	}
	for _, response := range h.output.frames() {
		if response.ID != nil && string(*response.ID) == "121" {
			if response.Error == nil || response.Error.Code != -32000 {
				t.Fatalf("superseded install must return busy: %+v", response)
			}
		}
	}
}
