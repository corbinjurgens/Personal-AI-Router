// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	settings "nvpair-shared/enginesettings"
)

func TestEngineBindingsSwitchAutomaticRingAndReplay(t *testing.T) {
	h := newSwitchHarness(t)
	manual := h.request(t, "manual")
	if _, err := h.b.acceptEngineSwitch(context.Background(), manual, ""); err != nil {
		t.Fatal(err)
	}
	h.waitIdle(t)
	var first settings.SwitchReceipt
	var original *automaticSwitchSubmission
	for i := 0; i < settings.MaxAutomaticSwitchReceipts+2; i++ {
		p := h.request(t, "")
		deadline := time.Now().Add(settings.AutomaticSwitchBudget)
		result, err := h.b.acceptAutomaticEngineSwitch(context.Background(), p, "", h.pp, deadline)
		if err != nil {
			t.Fatal(err)
		}
		r := h.receipt(t, result.Receipt.RequestID)
		if !r.Automatic || r.Phase != settings.SwitchCompleted || !r.Deadline.Equal(deadline) || !strings.HasPrefix(r.RequestID, "binding:") {
			t.Fatalf("automatic receipt=%+v", r)
		}
		if i == 0 {
			first = r
			c := h.b.switchCoordinator()
			original = &automaticSwitchSubmission{c.automaticRun, c.automaticSequence, h.pp, deadline}
		}
	}
	h.b.engineConfigMu.Lock()
	c := h.b.switchCoordinator()
	j := switchStoreDisk(t, c.store)
	h.b.engineConfigMu.Unlock()
	if _, ok := j.Receipts[first.RequestID]; ok {
		t.Fatal("first automatic ID was not pruned")
	}
	if len(j.AutomaticSettled) != settings.MaxAutomaticSwitchReceipts {
		t.Fatalf("automatic ring=%d", len(j.AutomaticSettled))
	}
	if _, ok := j.Receipts[manual.RequestID]; !ok {
		t.Fatal("manual ID was pruned")
	}
	before := h.engine.snapshot()
	checkReplay := func(t *testing.T) {
		t.Helper()
		if _, err := h.b.acceptEngineSwitchSubmission(context.Background(), first.Request, "", original); err == nil {
			t.Fatal("pruned internal ID replay accepted")
		}
		if _, err := h.b.acceptEngineSwitch(context.Background(), first.Request, ""); err == nil {
			t.Fatal("pruned public ID replay accepted")
		}
		if _, err := h.b.fenceEngineSwitch(context.Background(), settings.SwitchFenceRequest{RequestID: first.RequestID}, ""); err == nil {
			t.Fatal("pruned automatic ID publicly fenced")
		}
		if !reflect.DeepEqual(before, h.engine.snapshot()) {
			t.Fatal("pruned replay reached executor")
		}
	}
	t.Run("same run", checkReplay)
	h.b.engineConfigMu.Lock()
	c.store = newEngineSwitchStore(c.store.path)
	c.automaticRun, c.automaticSequence, c.automaticConsumed = "replacement-run", 0, 0
	h.b.engineConfigMu.Unlock()
	t.Run("after restart", checkReplay)
	got, err := h.b.acceptEngineSwitch(context.Background(), manual, "")
	if err != nil || got.Receipt.RequestID != manual.RequestID {
		t.Fatalf("manual retry=%+v %v", got, err)
	}
}

func TestEngineBindingsSwitchLegacyNamespace(t *testing.T) {
	h := newSwitchHarness(t)
	p := h.request(t, "binding:legacy:1")
	p.NodeID = h.b.nodeID
	r := switchStoreReceipt(p.RequestID)
	r.NodeID, r.Request, r.ExpectedRevision, r.AcceptedRevision = h.b.nodeID, p, p.ExpectedRevision, p.ExpectedRevision
	r.PayloadHash = settings.SwitchPayloadHash(r.Caller, r.NodeID, p)
	r = switchStoreNext(r, settings.SwitchRejected)
	h.b.engineConfigMu.Lock()
	_, err := h.b.engineSwitch.store.Reject(r)
	h.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	got, err := h.b.acceptEngineSwitch(context.Background(), p, "")
	if err != nil || !reflect.DeepEqual(got.Receipt, r) {
		t.Fatalf("legacy retry=%+v, %v", got, err)
	}
	fence, err := h.b.fenceEngineSwitch(context.Background(), settings.SwitchFenceRequest{RequestID: p.RequestID}, "")
	if err != nil || fence.Receipt == nil || !reflect.DeepEqual(*fence.Receipt, r) {
		t.Fatalf("legacy fence=%+v, %v", fence, err)
	}
	p.TargetEngine = "ollama"
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err == nil || switchErrorDetail(err, "").Kind != settings.SwitchErrorReusedID {
		t.Fatalf("legacy changed payload=%v", err)
	}
	h.b.engineConfigMu.Lock()
	_, err = h.b.engineSwitch.store.Fence(h.b.nodeID, "local", "binding:legacy:fenced")
	h.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	fence, err = h.b.fenceEngineSwitch(context.Background(), settings.SwitchFenceRequest{RequestID: "binding:legacy:fenced"}, "")
	if err != nil || !fence.Fenced {
		t.Fatalf("legacy tombstone=%+v, %v", fence, err)
	}
}

func TestEngineBindingsSwitchDrainDeadlineNeverStopsSource(t *testing.T) {
	h := newSwitchHarness(t)
	h.active.Store(1)
	p := h.request(t, "")
	deadline := time.Now().Add(200 * time.Millisecond)
	result, err := h.b.acceptAutomaticEngineSwitch(context.Background(), p, "", h.pp, deadline)
	if err != nil {
		t.Fatal(err)
	}
	r := h.receipt(t, result.Receipt.RequestID)
	if r.Phase != settings.SwitchFailedDrain || !r.RecoveryRequired || !r.Deadline.Equal(deadline) {
		t.Fatalf("drain refusal=%+v", r)
	}
	if steps := switchSteps(t, h.engine); len(steps) != 0 {
		t.Fatalf("effects after held admission: %v", steps)
	}
}

func TestEngineBindingsSwitchLocalAutomaticOffObligation(t *testing.T) {
	h := newSwitchHarness(t)
	h.active.Store(1)
	cleanup := make(chan settings.SwitchReconcileRequest, 1)
	base := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == settings.SwitchReconcileMethod {
			var p settings.SwitchReconcileRequest
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			select {
			case cleanup <- p:
			default:
			}
		}
		return base(method, raw)
	})
	p := h.request(t, "")
	result, err := h.b.acceptAutomaticEngineSwitch(context.Background(), p, "", h.pp, time.Now().Add(45*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	h.b.engineConfigMu.Lock()
	yields, reserved, id, err := h.b.switchLocalOffReservationLocked("lmstudio")
	h.b.engineConfigMu.Unlock()
	if err != nil || !yields || !reserved || id != result.Receipt.RequestID {
		t.Fatalf("automatic local Off yields=%t reserved=%t id=%s err=%v", yields, reserved, id, err)
	}
	h.b.cancelRemoteEngineSwitchOperation("lmstudio", id)
	r := h.receipt(t, id)
	if !r.Automatic || r.Caller != settings.SwitchOriginLocal || r.Error == nil || r.Error.Kind != settings.SwitchErrorRemoteVeto {
		t.Fatalf("automatic local veto=%+v", r)
	}
	select {
	case request := <-cleanup:
		if request.CleanupDeadline == nil || !request.CleanupDeadline.Equal(r.Deadline) || !request.Deadline.Equal(r.Deadline) {
			t.Fatalf("cleanup lost remaining deadline or changed identity: %+v", request)
		}
	default:
		t.Fatal("automatic veto did not send bounded cleanup")
	}
}

func TestEngineBindingsSwitchAcceptanceRechecksProxySession(t *testing.T) {
	h := newSwitchHarness(t)
	p := h.request(t, "")
	base := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			var envelope settings.SwitchPrepareRequest
			if err := json.Unmarshal(raw, &envelope); err != nil {
				return nil, err.Error()
			}
			if envelope.OperationID == "" {
				h.b.workersMu.Lock()
				h.b.proxyProc = nil
				h.b.workersMu.Unlock()
			}
		}
		return base(method, raw)
	})
	_, err := h.b.acceptAutomaticEngineSwitch(context.Background(), p, "", h.pp, time.Now().Add(time.Minute))
	h.b.workersMu.Lock()
	h.b.proxyProc = h.pp
	h.b.workersMu.Unlock()
	if err == nil {
		t.Fatal("replaced source accepted")
	}
	h.b.engineConfigMu.Lock()
	count := len(h.b.engineSwitch.store.journal.Receipts)
	h.b.engineConfigMu.Unlock()
	if count != 0 || len(switchSteps(t, h.engine)) != 0 {
		t.Fatalf("replaced source left %d receipts or effects", count)
	}
}

func TestEngineBindingsSwitchProxyExitDuringAcceptance(t *testing.T) {
	h := newSwitchHarness(t)
	p := h.request(t, "")
	h.pp.done = make(chan struct{})
	h.b.engineConfigMu.Lock()
	store := h.b.switchCoordinator().store
	write := store.write
	store.write = func(path string, data []byte, mode os.FileMode) error {
		var j settings.SwitchJournal
		if err := json.Unmarshal(data, &j); err != nil {
			return err
		}
		if j.ActiveRequestID != "" && j.Receipts[j.ActiveRequestID].Phase == settings.SwitchAccepted {
			close(h.pp.done) // retirement while acceptance is durably written
		}
		return write(path, data, mode)
	}
	h.b.engineConfigMu.Unlock()
	result, err := h.b.acceptAutomaticEngineSwitch(context.Background(), p, "", h.pp, time.Now().Add(time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	r := h.receipt(t, result.Receipt.RequestID)
	if !r.Terminal || r.Phase == settings.SwitchCompleted {
		t.Fatalf("retired source receipt=%+v", r)
	}
	if steps := switchSteps(t, h.engine); len(steps) != 0 {
		t.Fatalf("retired source effects=%v", steps)
	}
}

func TestEngineBindingsSwitchUncertainAcceptanceRetainsOriginalID(t *testing.T) {
	h := newSwitchHarness(t)
	p := h.request(t, "")
	h.b.engineConfigMu.Lock()
	store := h.b.switchCoordinator().store
	store.write = func(path string, data []byte, mode os.FileMode) error {
		if err := durablefile.WriteFile(path, data, mode); err != nil {
			return err
		}
		return durablefile.ErrReplaced
	}
	h.b.engineConfigMu.Unlock()
	result, err := h.b.acceptAutomaticEngineSwitch(context.Background(), p, "", h.pp, time.Now().Add(time.Minute))
	if !errors.Is(err, durablefile.ErrReplaced) || !result.Receipt.Automatic || result.Receipt.RequestID == "" {
		t.Fatalf("unknown acceptance lost original ID: %+v, %v", result, err)
	}
	if err := result.Receipt.Validate(); err != nil {
		t.Fatal(err)
	}
	status, err := h.b.engineSwitchStatus(context.Background(), settings.SwitchStatusRequest{RequestID: result.Receipt.RequestID}, "")
	if err != nil || status.Receipt == nil || status.Receipt.RequestID != result.Receipt.RequestID {
		t.Fatalf("original readback=%+v %v", status, err)
	}
	if steps := switchSteps(t, h.engine); len(steps) != 0 {
		t.Fatalf("effects after uncertain acceptance=%v", steps)
	}
}
