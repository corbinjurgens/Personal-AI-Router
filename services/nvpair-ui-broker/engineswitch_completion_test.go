// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
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
	remote, reserved := h.b.admitLocalEngineOff("lmstudio")
	defer h.b.settingsApplyMu.Unlock()
	if remote || reserved {
		t.Fatalf("unexpected Off reservation: remote %v, reserved %v", remote, reserved)
	}
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err == nil || switchErrorDetail(err, "").Kind != settings.SwitchErrorBusy {
		t.Fatalf("switch passed Off's operation admission: %v", err)
	}
}
