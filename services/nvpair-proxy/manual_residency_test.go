// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"testing/synctest"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

func manualResidentHarness(t *testing.T, models ...string) (*admHarness, nodepolicy.PrepareResidentParams) {
	t.Helper()
	f := newReadinessEnvelopeFixture(t, "llamacpp")
	profile, envelope := f.policy.Profiles[0], f.id.Serving["m"]
	f.policy.Profiles = nil
	f.policy.ResidentModels = nil
	f.policy.Admission.MaxResidentModels = 2
	f.policy.Admission.MemoryBudgetBytes = 100
	f.policy.Admission.SwitchModels = true
	f.policy.Admission.SwitchTimeoutSeconds = 1
	f.id.ModelDigests = map[string]string{}
	f.id.Serving = map[string]engines.ServingEnvelope{}
	for _, model := range models {
		pr := profile
		pr.Key, pr.Model, pr.Qualification, pr.MemoryBytes = model, model, nil, 40
		f.policy.Profiles = append(f.policy.Profiles, pr)
		bound := envelope.Clone()
		bound.ModelID = model
		memory := int64(40)
		bound.Memory.VRAMBytes = &memory
		f.id.ModelDigests[model] = envelope.ContentDigest
		f.id.Serving[model] = bound
	}
	f.install(t)
	f.a.setResidency(map[string][]string{f.engine: {}})
	h := &admHarness{a: f.a}
	f.a.notify = func(method string, params any) {
		h.mu.Lock()
		defer h.mu.Unlock()
		h.sent = append(h.sent, notification{method, params})
	}
	return h, nodepolicy.PrepareResidentParams{Purpose: nodepolicy.ResidentPurposeManual, PolicyRevision: "1", ProxyRunID: "proxy", ManagerIncarnation: "manager",
		Engine: f.engine, Model: models[0], Identity: f.request().Identity, OperationID: "resident:1", TimeoutMs: 1000}
}

type manualPreparationResult struct {
	result nodepolicy.PrepareResidentResult
	err    error
}

func asyncManualPreparation(a *admissionController, in nodepolicy.PrepareResidentParams) <-chan manualPreparationResult {
	ch := make(chan manualPreparationResult, 1)
	go func() {
		result, err := a.prepareResident("proxy", in)
		ch <- manualPreparationResult{result, err}
	}()
	return ch
}

func awaitManualPreparation(t *testing.T, ch <-chan manualPreparationResult) nodepolicy.PrepareResidentResult {
	t.Helper()
	select {
	case got := <-ch:
		if got.err != nil || got.result.ReservationID == "" || len(got.result.Refusals) != 0 {
			t.Fatalf("manual preparation=%+v error=%v", got.result, got.err)
		}
		return got.result
	case <-time.After(3 * time.Second):
		t.Fatal("manual preparation did not finish")
		return nodepolicy.PrepareResidentResult{}
	}
}

func TestManualResidentUnqualifiedProfile(t *testing.T) {
	h, in := manualResidentHarness(t, "m")
	prepared := prepareResidentOK(t, h.a, in)
	dispatchResidentOK(t, h.a, in, prepared)
	finish := residentFinish(in, prepared, nodepolicy.ResidentLoaded)
	finish.Observation = residentObservationFor(in, 1, true)
	settled := finishResidentOK(t, h.a, finish)
	if settled.RecoveryID != "" || !h.a.isLoaded(in.Engine, in.Model) || h.a.currentActive() != 0 {
		t.Fatalf("manual load did not settle into ordinary inventory: %+v", settled)
	}
	h.a.mu.Lock()
	state := h.a.snapshotLocked().state
	h.a.mu.Unlock()
	if state.LastActivityMs[in.Engine][in.Model] == 0 {
		t.Fatal("manual load did not record per-model activity")
	}
}

func TestAutomaticResidentStillRequiresQualification(t *testing.T) {
	h, in := manualResidentHarness(t, "m")
	p := h.a.currentPolicy()
	p.ResidentModels = nodepolicy.ResidentModels{{Engine: in.Engine, Model: in.Model}}
	h.a.setPolicy(p)
	in.Purpose = ""
	result, err := h.a.prepareResident("proxy", in)
	if err != nil || result.ReservationID != "" || len(result.Refusals) != 1 || result.Refusals[0].Reason != nodepolicy.ModelReasonUnqualified {
		t.Fatalf("automatic residency accepted an unqualified profile: %+v error=%v", result, err)
	}
}

func TestManualResidentEvictsLRUAtCap(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h, in := manualResidentHarness(t, "m", "n", "z")
		h.a.mu.Lock()
		h.a.lastActivity[admKey{in.Engine, "m"}] = time.Now().Add(-2 * time.Minute).UnixMilli()
		h.a.lastActivity[admKey{in.Engine, "n"}] = time.Now().Add(-time.Minute).UnixMilli()
		h.a.mu.Unlock()
		h.a.setResidency(map[string][]string{in.Engine: {"m", "n"}})
		in.Model = "z"
		pending := asyncManualPreparation(h.a, in)
		synctest.Wait()
		notes := h.notes(nodepolicy.NotifyAdmissionUnload)
		if len(notes) != 1 {
			t.Fatalf("unload requests=%+v", notes)
		}
		unload, ok := notes[0].params.(nodepolicy.UnloadRequest)
		if !ok || unload.Engine != in.Engine || !reflect.DeepEqual(unload.Models, []string{"m"}) {
			t.Fatalf("LRU unload=%+v", notes[0].params)
		}
		if h.a.currentActive() != 0 {
			t.Fatal("manual preparation exceeded cap before unload confirmation")
		}
		h.a.setResidency(map[string][]string{in.Engine: {"n"}})
		prepared := awaitManualPreparation(t, pending)
		dispatchResidentOK(t, h.a, in, prepared)
		finish := residentFinish(in, prepared, nodepolicy.ResidentLoaded)
		finish.Observation = residentObservationFor(in, 1, true)
		finishResidentOK(t, h.a, finish)
		h.a.mu.Lock()
		resident := h.a.residentLocked()
		h.a.mu.Unlock()
		if len(resident) != 2 || !resident[admKey{in.Engine, "n"}] || !resident[admKey{in.Engine, "z"}] {
			t.Fatalf("settled models=%+v", resident)
		}
	})
}

func TestOrdinaryAdmissionEvictsLRUAtTwoModelCap(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h, in := manualResidentHarness(t, "m", "n", "z")
		h.a.mu.Lock()
		h.a.lastActivity[admKey{in.Engine, "m"}] = time.Now().Add(-time.Minute).UnixMilli()
		h.a.lastActivity[admKey{in.Engine, "n"}] = time.Now().Add(-2 * time.Minute).UnixMilli()
		h.a.mu.Unlock()
		h.a.setResidency(map[string][]string{in.Engine: {"m", "n"}})
		pending := h.admitAsync(in.Engine, "z", -1)
		synctest.Wait()
		notes := h.notes(nodepolicy.NotifyAdmissionUnload)
		if len(notes) != 1 {
			t.Fatalf("ordinary unload requests=%+v", notes)
		}
		unload, ok := notes[0].params.(nodepolicy.UnloadRequest)
		if !ok || !reflect.DeepEqual(unload.Models, []string{"n"}) {
			t.Fatalf("ordinary LRU unload=%+v", notes[0].params)
		}
		h.a.setResidency(map[string][]string{in.Engine: {"m"}})
		mustAdmit(t, awaitDecision(t, pending, time.Second)).release()
	})
}

func TestManualResidentWaitsForInflightBeforeEviction(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h, in := manualResidentHarness(t, "m", "n")
		p := h.a.currentPolicy()
		p.Admission.MaxResidentModels = 1
		h.a.setPolicy(p)
		h.a.setResidency(map[string][]string{in.Engine: {"m"}})
		ticket := mustAdmit(t, h.admit(in.Engine, "m", 0))
		in.Model = "n"
		pending := asyncManualPreparation(h.a, in)
		synctest.Wait()
		if len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 0 || h.a.currentActive() != 1 {
			t.Fatal("manual load evicted an inflight model or reserved above cap")
		}
		ticket.release()
		synctest.Wait()
		if len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 1 {
			t.Fatal("manual load did not use the shared eviction planner after drain")
		}
		h.a.setResidency(map[string][]string{in.Engine: {}})
		prepared := awaitManualPreparation(t, pending)
		finishResidentOK(t, h.a, residentFinish(in, prepared, nodepolicy.ResidentNotDispatched))
	})
}

func TestManualResidentRefusals(t *testing.T) {
	test := func(name string, edit func(*admissionController, *nodepolicy.PrepareResidentParams), want nodepolicy.ModelConfigurationReason) {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				h, in := manualResidentHarness(t, "m", "n")
				edit(h.a, &in)
				result, err := h.a.prepareResident("proxy", in)
				if err != nil || result.ReservationID != "" || len(result.Refusals) != 1 || result.Refusals[0].Reason != want || h.a.currentActive() != 0 {
					t.Fatalf("manual refusal=%+v error=%v active=%d", result, err, h.a.currentActive())
				}
			})
		})
	}
	test("configured pause", func(a *admissionController, _ *nodepolicy.PrepareResidentParams) {
		a.setAvailability(nodepolicy.Paused, false)
	}, nodepolicy.ModelReasonPausedLocal)
	test("halt cancels active", func(a *admissionController, _ *nodepolicy.PrepareResidentParams) {
		a.setAvailability(nodepolicy.Paused, true)
	}, nodepolicy.ModelReasonPausedLocal)
	test("settings drain", func(a *admissionController, in *nodepolicy.PrepareResidentParams) { a.setDrain(in.Engine, true) }, nodepolicy.ModelReasonOperationBusy)
	test("identity mismatch", func(_ *admissionController, in *nodepolicy.PrepareResidentParams) {
		in.Identity.EngineRunID = "replacement"
	}, nodepolicy.ModelReasonUnqualified)
	test("missing envelope", func(a *admissionController, in *nodepolicy.PrepareResidentParams) {
		id := a.strict.identity[in.Engine]
		id.Serving = nil
		a.setModelIdentity(map[string]nodepolicy.EngineIdentity{in.Engine: id})
	}, nodepolicy.ModelReasonUnsupportedLoad)
	test("memory budget", func(a *admissionController, _ *nodepolicy.PrepareResidentParams) {
		p := a.currentPolicy()
		p.Admission.MemoryBudgetBytes = 39
		a.setPolicy(p)
	}, nodepolicy.ModelReasonNoFit)
	test("unreadable capacity ownership", func(a *admissionController, _ *nodepolicy.PrepareResidentParams) {
		a.strict.storageErr = errors.New("capacity read failed")
	}, nodepolicy.ModelReasonNoFit)
	test("unconfirmed eviction times out", func(a *admissionController, in *nodepolicy.PrepareResidentParams) {
		p := a.currentPolicy()
		p.Admission.MaxResidentModels = 1
		a.setPolicy(p)
		a.setResidency(map[string][]string{in.Engine: {"n"}})
	}, nodepolicy.ModelReasonNoFit)
}

func TestManualResidentDefiniteFailureReleasesReservation(t *testing.T) {
	h, in := manualResidentHarness(t, "m")
	prepared := prepareResidentOK(t, h.a, in)
	dispatchResidentOK(t, h.a, in, prepared)
	finish := residentFinish(in, prepared, nodepolicy.ResidentFailed)
	finish.Observation = residentObservationFor(in, 1, false)
	result := finishResidentOK(t, h.a, finish)
	if result.RecoveryID != "" || h.a.currentActive() != 0 || h.a.isLoaded(in.Engine, in.Model) {
		t.Fatalf("definite failure retained reservation: %+v", result)
	}
}

func TestManualResidentUncertainDispatchKeepsReservation(t *testing.T) {
	h, in := manualResidentHarness(t, "m", "n")
	prepared := prepareResidentOK(t, h.a, in)
	dispatchResidentOK(t, h.a, in, prepared)
	finish := finishResidentOK(t, h.a, residentFinish(in, prepared, nodepolicy.ResidentUnknown))
	if finish.RecoveryID == "" || h.a.currentActive() != 1 {
		t.Fatalf("uncertain dispatch lost reservation: %+v", finish)
	}
	h.a.setResidency(map[string][]string{in.Engine: {}})
	if h.a.currentActive() != 1 {
		t.Fatal("ordinary inventory released uncertain dispatch")
	}
	recovery := residentRecovery(in, finish, 1, false)
	observed, err := h.a.observeResident("proxy", recovery)
	if err != nil || !observed.Settled || h.a.currentActive() != 0 {
		t.Fatalf("authoritative recovery=%+v error=%v", observed, err)
	}
}

func TestManualResidentDispatchRechecksIdentity(t *testing.T) {
	h, in := manualResidentHarness(t, "m")
	prepared := prepareResidentOK(t, h.a, in)
	id := h.a.strict.identity[in.Engine]
	id.EngineRunID = "replacement"
	h.a.setModelIdentity(map[string]nodepolicy.EngineIdentity{in.Engine: id})
	result, err := h.a.dispatchResident("proxy", residentDispatch(in, prepared))
	if err != nil || result.Dispatched || result.Reason != nodepolicy.DispatchReasonReservationRevoked || h.a.currentActive() != 0 {
		t.Fatalf("stale manual grant=%+v error=%v", result, err)
	}
}

func TestManualResidentProtectsAutomaticResident(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h, in := manualResidentHarness(t, "m", "n")
		p := h.a.currentPolicy()
		p.Admission.MaxResidentModels = 1
		for ref := range h.a.strict.registry {
			p.Profiles[0].Qualification = &ref
		}
		p.ResidentModels = nodepolicy.ResidentModels{{Engine: in.Engine, Model: "m"}}
		h.a.setPolicy(p)
		h.a.setResidency(map[string][]string{in.Engine: {"m"}})
		in.Model = "n"
		result, err := h.a.prepareResident("proxy", in)
		if err != nil || result.ReservationID != "" || len(result.Refusals) != 1 || result.Refusals[0].Reason != nodepolicy.ModelReasonNoFit {
			t.Fatalf("protected resident result=%+v error=%v", result, err)
		}
		if len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 0 {
			t.Fatal("manual load evicted a qualified automatic resident")
		}
	})
}

func TestManualResidentUncertainHoldEnforcesCap(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h, in := manualResidentHarness(t, "m", "n")
		p := h.a.currentPolicy()
		p.Admission.MaxResidentModels = 1
		h.a.setPolicy(p)
		prepared := prepareResidentOK(t, h.a, in)
		dispatchResidentOK(t, h.a, in, prepared)
		finish := finishResidentOK(t, h.a, residentFinish(in, prepared, nodepolicy.ResidentUnknown))
		other := in
		other.Model, other.OperationID, other.TimeoutMs = "n", "resident:2", 10
		result, err := h.a.prepareResident("proxy", other)
		if err != nil || result.ReservationID != "" || len(result.Refusals) != 1 || result.Refusals[0].Reason != nodepolicy.ModelReasonNoFit || h.a.currentActive() != 1 {
			t.Fatalf("uncertain hold allowed another load: %+v error=%v", result, err)
		}
		if len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 0 {
			t.Fatal("uncertain load was selected for eviction")
		}
		if _, err := h.a.observeResident("proxy", residentRecovery(in, finish, 1, false)); err != nil {
			t.Fatal(err)
		}
	})
}

func TestManualResidentPreparationKeepsControlReaderAvailable(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h, in := manualResidentHarness(t, "m", "n")
		policy := h.a.currentPolicy()
		policy.Admission.MaxResidentModels = 1
		h.a.setPolicy(policy)
		h.a.setResidency(map[string][]string{in.Engine: {"n"}})
		writer := &recordingWriter{}
		proxy := &Proxy{admission: h.a, codec: NewCodec(writer), runID: in.ProxyRunID}
		id := json.RawMessage(`-1`)
		params, err := json.Marshal(in)
		if err != nil {
			t.Fatal(err)
		}
		if !proxy.handlePolicyMethod(&Message{ID: &id, Params: params}, nodepolicy.MethodPrepareResident) {
			t.Fatal("manual preparation was not handled")
		}
		synctest.Wait()
		if len(frames(t, writer)) != 0 || len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 1 {
			t.Fatal("preparation did not wait for capacity independently of the reader")
		}
		// The same reader must be able to process the confirmation the waiting
		// preparation needs. No facade or network listener is involved.
		rpcOK(t, proxy, writer, nodepolicy.MethodSetResidency, nodepolicy.SetResidencyParams{LoadedByEngine: map[string][]string{in.Engine: {}}})
		synctest.Wait()
		var prepared nodepolicy.PrepareResidentResult
		found := false
		for _, frame := range frames(t, writer) {
			if frame.ID != nil && string(*frame.ID) == string(id) {
				if frame.Error != nil {
					t.Fatalf("manual preparation RPC error=%+v", frame.Error)
				}
				if err := json.Unmarshal(frame.Result, &prepared); err != nil {
					t.Fatal(err)
				}
				found = true
			}
		}
		if !found || prepared.ReservationID == "" {
			t.Fatalf("manual preparation RPC=%+v found=%v", prepared, found)
		}
		finishResidentOK(t, h.a, residentFinish(in, prepared, nodepolicy.ResidentNotDispatched))
	})
}

func TestManualResidentPendingRetriesShareReservation(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h, in := manualResidentHarness(t, "m", "n")
		p := h.a.currentPolicy()
		p.Admission.MaxResidentModels = 1
		h.a.setPolicy(p)
		h.a.setResidency(map[string][]string{in.Engine: {"n"}})
		first := asyncManualPreparation(h.a, in)
		synctest.Wait()
		second := asyncManualPreparation(h.a, in)
		synctest.Wait()
		h.a.setResidency(map[string][]string{in.Engine: {}})
		prepared, duplicate := awaitManualPreparation(t, first), awaitManualPreparation(t, second)
		if !reflect.DeepEqual(prepared, duplicate) || h.a.currentActive() != 1 {
			t.Fatalf("duplicate preparation counted twice: first=%+v second=%+v", prepared, duplicate)
		}
		changed := in
		changed.Purpose = ""
		if _, err := h.a.prepareResident("proxy", changed); err == nil {
			t.Fatal("changed purpose reused a manual operation")
		}
		finishResidentOK(t, h.a, residentFinish(in, prepared, nodepolicy.ResidentNotDispatched))
	})
}

func TestManualResidentPauseWhileWaitingRevokesPreparation(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h, in := manualResidentHarness(t, "m", "n")
		p := h.a.currentPolicy()
		p.Admission.MaxResidentModels = 1
		h.a.setPolicy(p)
		h.a.setResidency(map[string][]string{in.Engine: {"n"}})
		pending := asyncManualPreparation(h.a, in)
		synctest.Wait()
		h.a.setAvailability(nodepolicy.Paused, true)
		got := <-pending
		if got.err != nil || got.result.ReservationID != "" || len(got.result.Refusals) != 1 || got.result.Refusals[0].Reason != nodepolicy.ModelReasonPausedLocal || h.a.currentActive() != 0 {
			t.Fatalf("paused pending manual load=%+v error=%v", got.result, got.err)
		}
	})
}
