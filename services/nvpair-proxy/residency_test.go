// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

func TestResidentNativeAllocation(t *testing.T) {
	test := func(name string, edit func(*readinessEnvelopeFixture), wantGrant bool) {
		t.Run(name, func(t *testing.T) {
			f := newReadinessEnvelopeFixture(t, "llamacpp")
			f.policy.ResidentModels = nodepolicy.ResidentModels{{Engine: f.engine, Model: "m"}}
			edit(&f)
			f.install(t)
			f.a.setIntent(map[string]bool{f.engine: true})
			f.a.setResidency(map[string][]string{f.engine: {}})
			in := nodepolicy.PrepareResidentParams{PolicyRevision: "1", ProxyRunID: "proxy", ManagerIncarnation: "manager", Engine: f.engine, Model: "m", Identity: f.request().Identity, OperationID: "resident:1", TimeoutMs: 1000}
			result, err := f.a.prepareResident("proxy", in)
			if err != nil {
				t.Fatal(err)
			}
			if wantGrant {
				if result.ReservationID == "" || len(result.Refusals) != 0 {
					t.Fatalf("exact native allocation refused: %+v", result)
				}
				dispatchResidentOK(t, f.a, in, result)
			} else if result.ReservationID != "" || len(result.Refusals) != 1 || result.Refusals[0].Reason != nodepolicy.ModelReasonUnqualified || f.a.currentActive() != 0 || len(f.a.inflight) != 0 {
				t.Fatalf("unqualified native allocation reserved capacity: %+v active=%d", result, f.a.currentActive())
			}
			if f.a.isLoaded(f.engine, "m") {
				t.Fatal("cold declaration became loaded inventory")
			}
		})
	}
	test("exact allocation", func(*readinessEnvelopeFixture) {}, true)
	test("profile cannot shrink native context", func(f *readinessEnvelopeFixture) { f.policy.Profiles[0].ContextTokens = 2048 }, false)
	test("qualification must cover native context", func(f *readinessEnvelopeFixture) {
		f.policy.Profiles[0].ContextTokens, f.q.ContextTokens = 2048, 2048
	}, false)
	test("qualification must cover native output", func(f *readinessEnvelopeFixture) {
		f.policy.Profiles[0].MaxOutputTokens, f.q.MaxOutputTokens = 512, 512
	}, false)
	test("effective concurrency must cover native slots", func(f *readinessEnvelopeFixture) { f.policy.Admission.MaxConcurrentPerModel = 1 }, false)
}

func TestResidentNativeAllocationRevalidation(t *testing.T) {
	test := func(name string, replace func(*readinessEnvelopeFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newReadinessEnvelopeFixture(t, "llamacpp")
			f.policy.ResidentModels = nodepolicy.ResidentModels{{Engine: f.engine, Model: "m"}}
			f.install(t)
			f.a.setIntent(map[string]bool{f.engine: true})
			f.a.setResidency(map[string][]string{f.engine: {}})
			in := nodepolicy.PrepareResidentParams{PolicyRevision: "1", ProxyRunID: "proxy", ManagerIncarnation: "manager", Engine: f.engine, Model: "m", Identity: f.request().Identity, OperationID: "resident:1", TimeoutMs: 1000}
			prepared := prepareResidentOK(t, f.a, in)
			replace(&f)
			result, err := f.a.dispatchResident("proxy", residentDispatch(in, prepared))
			if err != nil || result.Dispatched || f.a.currentActive() != 0 || len(f.a.inflight) != 0 {
				t.Fatalf("stale native allocation granted: %+v err=%v active=%d", result, err, f.a.currentActive())
			}
		})
	}
	test("qualification narrows without generation refresh", func(f *readinessEnvelopeFixture) {
		f.a.mu.Lock()
		defer f.a.mu.Unlock()
		f.q.MaxOutputTokens = 512
		f.a.policy.Profiles[0].MaxOutputTokens = 512
	})
	test("envelope replacement", func(f *readinessEnvelopeFixture) {
		id := f.id
		envelope := id.Serving["m"].Clone()
		envelope.NativeSourceDigest = strings.Repeat("e", 64)
		id.Serving = map[string]engines.ServingEnvelope{"m": envelope}
		f.a.setModelIdentity(map[string]nodepolicy.EngineIdentity{f.engine: id})
	})
	test("run replacement", func(f *readinessEnvelopeFixture) {
		id := f.id
		id.EngineRunID = "replacement-run"
		f.a.setModelIdentity(map[string]nodepolicy.EngineIdentity{f.engine: id})
	})
}

func residentHarness(t *testing.T, models ...string) (*admHarness, nodepolicy.PrepareResidentParams) {
	t.Helper()
	h := newAdmHarness(t)
	h.a.strict.storageErr = nil
	keys := make([]admKey, 0, len(models))
	for _, model := range models {
		keys = append(keys, admKey{engine: "ollama", model: model})
	}
	installQualified(t, h.a, keys...)
	p := h.a.currentPolicy()
	p.Revision = "1"
	p.Admission.MaxResidentModels = 2
	p.Admission.MaxConcurrentPerModel = 1
	p.Admission.MemoryBudgetBytes = 100
	p.Admission.SwitchTimeoutSeconds = 5
	p.Admission.QueueTimeoutSeconds = 0
	for i := range p.Profiles {
		p.Profiles[i].MemoryBytes = 40
	}
	p.ResidentModels = nodepolicy.ResidentModels{{Engine: "ollama", Model: models[0]}}
	h.a.setPolicy(p)
	h.a.setIntent(map[string]bool{"ollama": true})
	h.a.setResidency(map[string][]string{"ollama": {}})
	return h, nodepolicy.PrepareResidentParams{PolicyRevision: "1", ProxyRunID: "proxy", ManagerIncarnation: "manager", Engine: "ollama", Model: models[0], Identity: strictRequest().Identity, OperationID: "resident:1", TimeoutMs: 10_000}
}

func TestResidentContextBoundPrepare(t *testing.T) {
	test := func(name string, context, bound int, collision bool, wantGrant bool) {
		t.Run(name, func(t *testing.T) {
			h, in := residentHarness(t, "m:latest")
			p := h.a.currentPolicy()
			p.Profiles[0].ContextTokens = context
			if collision {
				alias := p.Profiles[0]
				alias.Name, alias.Model = "alias", "m"
				p.Profiles = append(p.Profiles, alias)
			}
			h.a.mu.Lock()
			q := h.a.strict.registry[*p.Profiles[0].Qualification]
			copy := *q
			copy.ContextTokens = bound
			h.a.strict.registry[*p.Profiles[0].Qualification] = &copy
			h.a.mu.Unlock()
			h.a.setPolicy(p)
			result, err := h.a.prepareResident("proxy", in)
			if err != nil {
				t.Fatal(err)
			}
			if wantGrant {
				if result.ReservationID == "" || len(result.Refusals) != 0 || h.a.currentActive() != 1 {
					t.Fatalf("expected counted grant: %+v, active=%d", result, h.a.currentActive())
				}
				if got := h.a.resident.ops[in.OperationID].context; got != context {
					t.Fatalf("captured context=%d, want %d", got, context)
				}
				dispatchResidentOK(t, h.a, in, result)
			} else {
				if result.ReservationID != "" || len(result.Refusals) != 1 || result.Refusals[0].Reason != nodepolicy.ModelReasonUnqualified || h.a.currentActive() != 0 || len(h.a.inflight) != 0 {
					t.Fatalf("expected uncounted refusal: %+v, active=%d inflight=%v", result, h.a.currentActive(), h.a.inflight)
				}
			}
			if h.a.isLoaded(in.Engine, in.Model) || len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 0 || len(h.notes(nodepolicy.NotifyAdmissionWake)) != 0 {
				t.Fatal("context evaluation caused an engine effect")
			}
		})
	}
	test("exact bound", 4096, 4096, false, true)
	test("bound plus one", 4097, 4096, false, false)
	test("absent bound", 4096, 0, false, false)
	test("negative bound", 4096, -1, false, false)
	test("absent profile context", 0, 4096, false, false)
	test("negative profile context", -1, 4096, false, false)
	test("maximum integer", int(^uint(0)>>1), 4096, false, false)
	test("below protocol ceiling above model limit", 8192, 4096, false, false)
	test("normalized collision", 4096, 4096, true, false)
}

func TestResidentContextBoundPerModel(t *testing.T) {
	h, in := residentHarness(t, "m:latest", "n:latest")
	p := h.a.currentPolicy()
	p.ResidentModels = append(p.ResidentModels, nodepolicy.Role{Engine: "ollama", Model: "n:latest"})
	p.Profiles[1].ContextTokens = 2048
	h.a.mu.Lock()
	ref := *p.Profiles[1].Qualification
	q := *h.a.strict.registry[ref]
	q.ContextTokens = 2048
	h.a.strict.registry[ref] = &q
	h.a.mu.Unlock()
	h.a.setPolicy(p)
	first := prepareResidentOK(t, h.a, in)
	dispatchResidentOK(t, h.a, in, first)
	in.Model, in.OperationID = "n:latest", "resident:2"
	second := prepareResidentOK(t, h.a, in)
	dispatchResidentOK(t, h.a, in, second)
	if h.a.resident.ops[in.OperationID].context != 2048 || h.a.currentActive() != 2 {
		t.Fatal("model-specific bound was not captured independently")
	}
}

func TestResidentContextBoundRevalidation(t *testing.T) {
	test := func(name string, mutate func(*admissionController, admKey)) {
		t.Run(name, func(t *testing.T) {
			h, in := residentHarness(t, "m:latest")
			prepared := prepareResidentOK(t, h.a, in)
			h.a.mu.Lock()
			// Deliberately replace evidence without a refresh signal: final grant
			// must itself revalidate, rather than rely only on generation updates.
			mutate(h.a, admKey{in.Engine, in.Model})
			h.a.mu.Unlock()
			result, err := h.a.dispatchResident("proxy", residentDispatch(in, prepared))
			if err != nil || result.Dispatched || result.Reason != nodepolicy.DispatchReasonReservationRevoked || h.a.currentActive() != 0 || len(h.a.inflight) != 0 {
				t.Fatalf("stale context grant=%+v error=%v active=%d", result, err, h.a.currentActive())
			}
			if len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 0 || len(h.notes(nodepolicy.NotifyAdmissionWake)) != 0 || h.a.isLoaded(in.Engine, in.Model) {
				t.Fatal("revoked context caused an engine effect")
			}
		})
	}
	test("profile replacement still below qualification", func(a *admissionController, _ admKey) {
		a.policy.Profiles[0].ContextTokens = 2048
	})
	test("qualification replacement below profile", func(a *admissionController, k admKey) {
		ref := *a.policy.Profiles[0].Qualification
		q := *a.strict.registry[ref]
		q.ContextTokens = 2048
		a.strict.registry[ref] = &q
	})
}

func prepareResidentOK(t *testing.T, a *admissionController, in nodepolicy.PrepareResidentParams) nodepolicy.PrepareResidentResult {
	t.Helper()
	result, err := a.prepareResident("proxy", in)
	if err != nil || result.ReservationID == "" || len(result.Refusals) != 0 {
		t.Fatalf("prepare result=%+v error=%v", result, err)
	}
	return result
}

func residentDispatch(in nodepolicy.PrepareResidentParams, result nodepolicy.PrepareResidentResult) nodepolicy.DispatchResidentParams {
	return nodepolicy.DispatchResidentParams{ReservationID: result.ReservationID, ProxyRunID: in.ProxyRunID, ManagerIncarnation: in.ManagerIncarnation,
		OperationID: in.OperationID, PolicyRevision: in.PolicyRevision, Engine: in.Engine, Model: in.Model, Identity: in.Identity}
}

func dispatchResidentOK(t *testing.T, a *admissionController, in nodepolicy.PrepareResidentParams, prepared nodepolicy.PrepareResidentResult) {
	t.Helper()
	result, err := a.dispatchResident("proxy", residentDispatch(in, prepared))
	if err != nil || !result.Dispatched || result.Reason != "" {
		t.Fatalf("dispatch result=%+v error=%v", result, err)
	}
}

func residentFinish(in nodepolicy.PrepareResidentParams, result nodepolicy.PrepareResidentResult, outcome nodepolicy.ResidentOutcome) nodepolicy.FinishResidentParams {
	return nodepolicy.FinishResidentParams{ReservationID: result.ReservationID, ProxyRunID: in.ProxyRunID, OperationID: in.OperationID, Outcome: outcome}
}

func finishResidentOK(t *testing.T, a *admissionController, in nodepolicy.FinishResidentParams) nodepolicy.FinishResidentResult {
	t.Helper()
	result, err := a.finishResident("proxy", in)
	if err != nil || result.ReservationID != in.ReservationID || result.Outcome != in.Outcome {
		t.Fatalf("finish result=%+v error=%v", result, err)
	}
	return result
}

func residentObservationFor(in nodepolicy.PrepareResidentParams, generation uint64, loaded bool) *nodepolicy.ResidentLoadedObservation {
	return &nodepolicy.ResidentLoadedObservation{ManagerIncarnation: in.ManagerIncarnation, Generation: generation, Engine: in.Engine, Model: in.Model, Identity: in.Identity, Loaded: loaded}
}

func residentOperation(t *testing.T, sequence uint64) string {
	t.Helper()
	id, err := nodepolicy.ResidentOperationID(sequence)
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func residentRecovery(in nodepolicy.PrepareResidentParams, finish nodepolicy.FinishResidentResult, sequence uint64, loaded bool) nodepolicy.ObserveResidentParams {
	return nodepolicy.ObserveResidentParams{ReservationID: finish.ReservationID, ProxyRunID: in.ProxyRunID, OperationID: in.OperationID, RecoveryID: finish.RecoveryID, Sequence: sequence, OperationComplete: true,
		Observation: nodepolicy.ResidentCapacityObservation{ManagerIncarnation: in.ManagerIncarnation, Engine: in.Engine, Model: in.Model, Identity: in.Identity, Loaded: loaded}}
}

func observeResidentOK(t *testing.T, a *admissionController, in nodepolicy.ObserveResidentParams) nodepolicy.ObserveResidentResult {
	t.Helper()
	result, err := a.observeResident("proxy", in)
	if err != nil || !result.Settled || result.ReservationID != in.ReservationID {
		t.Fatalf("recovery=%+v error=%v", result, err)
	}
	return result
}

func fillReleasedResidentHistory(t *testing.T, a *admissionController, in nodepolicy.PrepareResidentParams) {
	t.Helper()
	for sequence := a.resident.highestSequence + 1; sequence <= maxResidentOperations; sequence++ {
		request := in
		request.OperationID = residentOperation(t, sequence)
		request.ManagerIncarnation = "completed-other-source"
		a.resident.ops[request.OperationID] = &residentHold{request: request, sequence: sequence,
			result: nodepolicy.PrepareResidentResult{EvidenceGeneration: sequence, Refusals: []nodepolicy.ResidentRefusal{{Engine: in.Engine, Model: in.Model, Reason: nodepolicy.ModelReasonNoFit}}}}
	}
	a.resident.highestSequence, a.resident.generation = maxResidentOperations, maxResidentOperations
}

func TestResidentCapacityInstalledIsNotLoaded(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	prepareResidentOK(t, h.a, in)
	if h.a.isLoaded("ollama", "m:latest") {
		t.Fatal("prepare fabricated loaded inventory")
	}
	if active := h.a.currentActive(); active != 1 {
		t.Fatalf("active=%d, want one held capacity slot", active)
	}
	if len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 0 || len(h.notes(nodepolicy.NotifyAdmissionWake)) != 0 {
		t.Fatal("prepare requested wake or eviction")
	}
}

func TestResidentCapacityLastSlotAcrossCallers(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	start := make(chan struct{})
	type answer struct {
		result nodepolicy.PrepareResidentResult
		err    error
	}
	answers := make(chan answer, 2)
	firstEvaluated := make(chan struct{})
	for _, operation := range []string{"resident:1", "resident:2"} {
		go func(operation string) {
			<-start
			if operation == "resident:2" {
				<-firstEvaluated
			}
			request := in
			request.OperationID = operation
			result, err := h.a.prepareResident("proxy", request)
			if operation == "resident:1" {
				close(firstEvaluated)
			}
			answers <- answer{result, err}
		}(operation)
	}
	close(start)
	granted := 0
	for range 2 {
		answer := <-answers
		if answer.err != nil {
			t.Fatal(answer.err)
		}
		if answer.result.ReservationID != "" {
			granted++
		} else if len(answer.result.Refusals) != 1 || answer.result.Refusals[0].Reason != nodepolicy.ModelReasonNoFit {
			t.Fatalf("unexpected refusal=%+v", answer.result)
		}
	}
	if granted != 1 || h.a.currentActive() != 1 {
		t.Fatalf("granted=%d active=%d, want one", granted, h.a.currentActive())
	}
	mustReject(t, h.admit("ollama", "m:latest", 0), nodepolicy.RejectBusy)
	req := strictRequest()
	req.QueueBudgetMs = 0
	if ticket, err := h.a.admitStrictNow(context.Background(), "ollama", "m:latest", req); ticket != nil || err == nil {
		t.Fatalf("strict bypassed held capacity: ticket=%v error=%v", ticket, err)
	}
}

func TestResidentCapacityLastSlotCompetesWithConcrete(t *testing.T) {
	h, in := residentHarness(t, "m:latest", "n:latest")
	p := h.a.currentPolicy()
	p.Admission.MaxResidentModels = 1
	h.a.setPolicy(p)
	start := make(chan struct{})
	type residentAnswer struct {
		result nodepolicy.PrepareResidentResult
		err    error
	}
	resident := make(chan residentAnswer, 1)
	concrete := make(chan admissionDecision, 1)
	go func() {
		<-start
		result, err := h.a.prepareResident("proxy", in)
		resident <- residentAnswer{result, err}
	}()
	go func() { <-start; concrete <- h.admit("ollama", "n:latest", 0) }()
	close(start)
	residentResult, concreteResult := <-resident, <-concrete
	if residentResult.err != nil {
		t.Fatal(residentResult.err)
	}
	grants := 0
	if residentResult.result.ReservationID != "" {
		grants++
	}
	if concreteResult.ticket != nil {
		grants++
	}
	if grants != 1 || h.a.currentActive() != 1 {
		t.Fatalf("resident=%+v concrete=%+v active=%d", residentResult.result, concreteResult, h.a.currentActive())
	}
	if concreteResult.ticket != nil {
		concreteResult.ticket.release()
	}
}

func TestResidentCapacityCountsStrictQuarantine(t *testing.T) {
	h, in := residentHarness(t, "m:latest", "n:latest")
	p := h.a.currentPolicy()
	p.Admission.MaxResidentModels = 1
	h.a.setPolicy(p)
	// Recovered strict markers enter these shared counts before any requests.
	k := admKey{"ollama", "n:latest"}
	h.a.mu.Lock()
	h.a.strict.markers[1] = capacityMarker{ID: 1, Engine: k.engine, Model: k.model}
	h.a.inflight[k], h.a.active, h.a.activeByEngine[k.engine] = 1, 1, 1
	h.a.mu.Unlock()
	result, err := h.a.prepareResident("proxy", in)
	if err != nil || result.ReservationID != "" || len(result.Refusals) != 1 || result.Refusals[0].Reason != nodepolicy.ModelReasonNoFit || h.a.currentActive() != 1 {
		t.Fatalf("strict quarantine bypassed: result=%+v error=%v", result, err)
	}
}

func TestResidentCapacityAggregateRefusal(t *testing.T) {
	test := func(name string, mutate func(*admissionController, *nodepolicy.Policy), reason nodepolicy.ModelConfigurationReason) {
		t.Run(name, func(t *testing.T) {
			h, in := residentHarness(t, "m:latest", "n:latest")
			p := h.a.currentPolicy()
			p.ResidentModels = append(p.ResidentModels, nodepolicy.Role{Engine: "ollama", Model: "n:latest"})
			mutate(h.a, &p)
			h.a.setPolicy(p)
			result, err := h.a.prepareResident("proxy", in)
			if err != nil || result.ReservationID != "" || len(result.Refusals) != 1 || result.Refusals[0].Reason != reason {
				t.Fatalf("prepare=%+v error=%v, want %s", result, err, reason)
			}
			if h.a.currentActive() != 0 {
				t.Fatal("refused batch consumed capacity")
			}
		})
	}
	test("desired count overcommit", func(_ *admissionController, p *nodepolicy.Policy) { p.Admission.MaxResidentModels = 1 }, nodepolicy.ModelReasonNoFit)
	test("desired memory overcommit", func(_ *admissionController, p *nodepolicy.Policy) { p.Admission.MemoryBudgetBytes = 70 }, nodepolicy.ModelReasonNoFit)
	test("unknown desired memory", func(_ *admissionController, p *nodepolicy.Policy) { p.Profiles[1].MemoryBytes = 0 }, nodepolicy.ModelReasonNoFit)
	test("unqualified desired member", func(_ *admissionController, p *nodepolicy.Policy) { p.Profiles[1].Qualification = nil }, nodepolicy.ModelReasonUnqualified)
	test("desired alias collision", func(_ *admissionController, p *nodepolicy.Policy) { p.ResidentModels[1].Model = "m" }, nodepolicy.ModelReasonUnqualified)
	test("profile alias collision", func(_ *admissionController, p *nodepolicy.Policy) {
		alias := p.Profiles[0]
		alias.Name, alias.Model = "alias", "m"
		p.Profiles = append(p.Profiles, alias)
	}, nodepolicy.ModelReasonUnqualified)
	test("unknown existing memory", func(a *admissionController, p *nodepolicy.Policy) {
		p.Admission.MaxResidentModels = 3
		a.setResidency(map[string][]string{"ollama": {"other:latest"}})
	}, nodepolicy.ModelReasonNoFit)
	test("pending unload", func(a *admissionController, _ *nodepolicy.Policy) {
		a.unloadSentAt[admKey{"ollama", "m:latest"}] = a.strict.now()
	}, nodepolicy.ModelReasonSwitchBusy)
	test("installed digest missing", func(a *admissionController, _ *nodepolicy.Policy) {
		a.setModelIdentity(map[string]nodepolicy.EngineIdentity{"ollama": testIdentity("n:latest")})
	}, nodepolicy.ModelReasonNotInstalled)
	test("installed aliases even with equal digest", func(a *admissionController, _ *nodepolicy.Policy) {
		a.setModelIdentity(map[string]nodepolicy.EngineIdentity{"ollama": testIdentity("m", "m:latest", "n:latest")})
	}, nodepolicy.ModelReasonUnqualified)
}

func TestResidentCapacityDuplicatePayloadAndDeadline(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	clock := useStrictClock(h.a)
	first := prepareResidentOK(t, h.a, in)
	deadline := h.a.resident.ops[in.OperationID].deadline
	if want := clock.Now().Add(5 * time.Second); !deadline.Equal(want) {
		t.Fatalf("deadline=%v want=%v", deadline, want)
	}
	clock.Advance(time.Second)
	again := prepareResidentOK(t, h.a, in)
	if !reflect.DeepEqual(first, again) || !h.a.resident.ops[in.OperationID].deadline.Equal(deadline) || h.a.currentActive() != 1 {
		t.Fatal("duplicate extended deadline or acquired another slot")
	}
	changed := in
	changed.TimeoutMs--
	if _, err := h.a.prepareResident("proxy", changed); err == nil || h.a.currentActive() != 1 {
		t.Fatal("changed operation payload accepted")
	}
	clock.Advance(5 * time.Second)
	result, err := h.a.dispatchResident("proxy", residentDispatch(in, first))
	if err != nil || result.Dispatched || result.Reason != nodepolicy.DispatchReasonReservationRevoked || h.a.currentActive() != 0 {
		t.Fatalf("expired final grant=%+v error=%v active=%d", result, err, h.a.currentActive())
	}
}

func TestResidentCapacityIncomingDeadlineWins(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	clock := useStrictClock(h.a)
	in.TimeoutMs = 250
	prepareResidentOK(t, h.a, in)
	if got, want := h.a.resident.ops[in.OperationID].deadline, clock.Now().Add(250*time.Millisecond); !got.Equal(want) {
		t.Fatalf("deadline=%v want=%v", got, want)
	}
}

func TestResidentCapacityMemorySumCannotOverflow(t *testing.T) {
	h, in := residentHarness(t, "m:latest", "n:latest")
	p := h.a.currentPolicy()
	p.Admission.MemoryBudgetBytes = 1<<63 - 1
	p.Profiles[0].MemoryBytes, p.Profiles[1].MemoryBytes = 1<<62, 1<<62
	p.ResidentModels = append(p.ResidentModels, nodepolicy.Role{Engine: "ollama", Model: "n:latest"})
	h.a.setPolicy(p)
	result, err := h.a.prepareResident("proxy", in)
	if err != nil || result.ReservationID != "" || len(result.Refusals) != 1 || result.Refusals[0].Reason != nodepolicy.ModelReasonNoFit {
		t.Fatalf("overflowed desired memory fit: %+v error=%v", result, err)
	}
	h.a.setResidency(map[string][]string{"ollama": {"m:latest"}})
	mustReject(t, h.admit("ollama", "n:latest", 0), nodepolicy.RejectNoFit)
}

func TestServedResidentPrepareBindingRefusals(t *testing.T) {
	for _, field := range []string{"proxy", "revision", "manager", "identity", "normalization"} {
		t.Run(field, func(t *testing.T) {
			h, in := residentHarness(t, "m:latest")
			switch field {
			case "proxy":
				in.ProxyRunID = "old-proxy"
			case "revision":
				in.PolicyRevision = "0"
			case "manager":
				in.ManagerIncarnation = ""
			case "identity":
				in.Identity.EngineRunID = "old-engine"
			case "normalization":
				in.Model = "m"
			}
			result, err := h.a.prepareResident("proxy", in)
			if field == "proxy" {
				if err == nil || h.a.resident.highestSequence != 0 || h.a.currentActive() != 0 {
					t.Fatal("wrong proxy run advanced preparation")
				}
				return
			}
			if err != nil || result.ReservationID != "" || len(result.Refusals) != 1 || result.Refusals[0].Reason != nodepolicy.ModelReasonUnqualified || h.a.currentActive() != 0 {
				t.Fatalf("binding accepted: result=%+v error=%v", result, err)
			}
		})
	}
}

func TestServedResidentFinalGrantRevocation(t *testing.T) {
	test := func(name string, revoke func(*admissionController)) {
		t.Run(name, func(t *testing.T) {
			h, in := residentHarness(t, "m:latest")
			prepared := prepareResidentOK(t, h.a, in)
			revoke(h.a)
			result, err := h.a.dispatchResident("proxy", residentDispatch(in, prepared))
			if err != nil || result.Dispatched || result.Reason != nodepolicy.DispatchReasonReservationRevoked || h.a.currentActive() != 0 {
				t.Fatalf("revoked dispatch=%+v error=%v active=%d", result, err, h.a.currentActive())
			}
		})
	}
	test("served withdrawal", func(a *admissionController) {
		p := a.currentPolicy()
		p.ServedModels = nodepolicy.ServedModels{}
		a.setPolicy(p)
	})
	test("revision replaced", func(a *admissionController) { p := a.currentPolicy(); p.Revision = "2"; a.setPolicy(p) })
	test("local pause", func(a *admissionController) { a.setAvailability(nodepolicy.Paused, false) })
	test("saved Off", func(a *admissionController) { a.setIntent(map[string]bool{"ollama": false}) })
	test("drain", func(a *admissionController) { a.setDrain("ollama", true) })
	test("identity withdrawal", func(a *admissionController) { a.setModelIdentity(map[string]nodepolicy.EngineIdentity{}) })
	test("switch invalidation", func(a *admissionController) {
		a.invalidateEngine(nodepolicy.InvalidateEngineParams{Engine: "ollama", OperationID: "switch", Reason: nodepolicy.InvalidationReasonEngineSwitch})
	})
	test("accepted launch digest changed", func(a *admissionController) {
		a.mu.Lock()
		for _, q := range a.strict.registry {
			q.LaunchDigests = append(q.LaunchDigests, strings.Repeat("d", 64))
		}
		a.mu.Unlock()
		id := testIdentity("m:latest")
		id.LaunchDigest = strings.Repeat("d", 64)
		a.setModelIdentity(map[string]nodepolicy.EngineIdentity{"ollama": id})
	})
}

func TestServedResidentDispatchBindings(t *testing.T) {
	for _, field := range []string{"proxy", "reservation", "manager", "operation", "revision", "engine", "model", "identity"} {
		t.Run(field, func(t *testing.T) {
			h, in := residentHarness(t, "m:latest")
			prepared := prepareResidentOK(t, h.a, in)
			request := residentDispatch(in, prepared)
			switch field {
			case "proxy":
				request.ProxyRunID = "old-proxy"
			case "reservation":
				request.ReservationID = "old-reservation"
			case "manager":
				request.ManagerIncarnation = "old-manager"
			case "operation":
				request.OperationID = "old-op"
			case "revision":
				request.PolicyRevision = "0"
			case "engine":
				request.Engine = "lmstudio"
			case "model":
				request.Model = "n:latest"
			case "identity":
				request.Identity.EngineRunID = "old-engine"
			}
			result, err := h.a.dispatchResident("proxy", request)
			if err != nil || result.Dispatched || result.Reason == "" || h.a.currentActive() != 1 {
				t.Fatalf("mismatched dispatch=%+v error=%v active=%d", result, err, h.a.currentActive())
			}
			dispatchResidentOK(t, h.a, in, prepared)
		})
	}
}

func TestResidentCapacityDispatchedUnknownRetained(t *testing.T) {
	for _, outcome := range []nodepolicy.ResidentOutcome{nodepolicy.ResidentUnknown, nodepolicy.ResidentTimeout, nodepolicy.ResidentCancelled, nodepolicy.ResidentFailed, nodepolicy.ResidentNotDispatched} {
		t.Run(string(outcome), func(t *testing.T) {
			h, in := residentHarness(t, "m:latest")
			clock := useStrictClock(h.a)
			prepared := prepareResidentOK(t, h.a, in)
			dispatchResidentOK(t, h.a, in, prepared)
			h.a.setAvailability(nodepolicy.Paused, true)
			clock.Advance(time.Minute)
			finish := residentFinish(in, prepared, outcome)
			first := finishResidentOK(t, h.a, finish)
			if again := finishResidentOK(t, h.a, finish); again != first {
				t.Fatal("duplicate finish changed result")
			}
			h.a.setResidency(map[string][]string{"ollama": {}})
			if h.a.currentActive() != 1 || h.a.inflight[admKey{"ollama", "m:latest"}] != 1 {
				t.Fatal("timeout, cancellation, revocation or unversioned inventory cleared dispatched uncertainty")
			}
		})
	}
}

func TestResidentCapacityUndispatchedFinishReleasesOnce(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	prepared := prepareResidentOK(t, h.a, in)
	finish := residentFinish(in, prepared, nodepolicy.ResidentNotDispatched)
	finishResidentOK(t, h.a, finish)
	finishResidentOK(t, h.a, finish)
	if h.a.currentActive() != 0 || len(h.a.inflight) != 0 {
		t.Fatal("undispatched finish did not release exactly once")
	}
	finish.Outcome = nodepolicy.ResidentCancelled
	if _, err := h.a.finishResident("proxy", finish); err == nil {
		t.Fatal("changed terminal finish accepted")
	}
	result, err := h.a.dispatchResident("proxy", residentDispatch(in, prepared))
	if err != nil || result.Dispatched {
		t.Fatalf("finished reservation dispatched: %+v %v", result, err)
	}
}

func TestResidentCapacityFinishObservations(t *testing.T) {
	for _, loaded := range []bool{true, false} {
		t.Run(fmt.Sprintf("loaded=%t", loaded), func(t *testing.T) {
			h, in := residentHarness(t, "m:latest")
			prepared := prepareResidentOK(t, h.a, in)
			dispatchResidentOK(t, h.a, in, prepared)
			finish := residentFinish(in, prepared, nodepolicy.ResidentFailed)
			if loaded {
				finish.Outcome = nodepolicy.ResidentLoaded
			}
			finish.Observation = residentObservationFor(in, 1, loaded)
			finishResidentOK(t, h.a, finish)
			finishResidentOK(t, h.a, finish)
			if h.a.currentActive() != 0 || h.a.isLoaded("ollama", in.Model) != loaded {
				t.Fatalf("observation did not convert capacity: active=%d loaded=%t", h.a.currentActive(), h.a.isLoaded("ollama", in.Model))
			}
			h.a.setResidency(map[string][]string{"ollama": {}})
			h.a.mu.Lock()
			resident := h.a.residentLocked()[admKey{"ollama", in.Model}]
			h.a.mu.Unlock()
			if resident {
				t.Fatal("completed load did not rejoin ordinary residency accounting")
			}
		})
	}
}

func TestResidentCapacityStaleFinishCannotReleaseReplacement(t *testing.T) {
	for _, field := range []string{"proxy", "operation", "reservation", "manager", "model", "engine", "identity", "generation"} {
		t.Run(field, func(t *testing.T) {
			h, in := residentHarness(t, "m:latest")
			prepared := prepareResidentOK(t, h.a, in)
			dispatchResidentOK(t, h.a, in, prepared)
			finish := residentFinish(in, prepared, nodepolicy.ResidentLoaded)
			finish.Observation = residentObservationFor(in, 1, true)
			switch field {
			case "proxy":
				finish.ProxyRunID = "old-proxy"
			case "operation":
				finish.OperationID = "old-op"
			case "reservation":
				finish.ReservationID = "old-reservation"
			case "manager":
				finish.Observation.ManagerIncarnation = "old-manager"
			case "model":
				finish.Observation.Model = "n:latest"
			case "engine":
				finish.Observation.Engine = "lmstudio"
			case "identity":
				finish.Observation.Identity.EngineRunID = "old-engine"
			case "generation":
				finish.Observation.Generation = 0
			}
			if _, err := h.a.finishResident("proxy", finish); err == nil || h.a.currentActive() != 1 {
				t.Fatal("stale completion released matching live reservation")
			}
		})
	}
}

func TestResidentCapacityRetainsPreparedMemory(t *testing.T) {
	for _, change := range []string{"profile removed", "bound reduced", "qualification withdrawn"} {
		t.Run(change, func(t *testing.T) {
			h, in := residentHarness(t, "m:latest", "n:latest")
			p := h.a.currentPolicy()
			p.Profiles[0].MemoryBytes = 80
			h.a.setPolicy(p)
			prepared := prepareResidentOK(t, h.a, in)
			dispatchResidentOK(t, h.a, in, prepared)
			p = h.a.currentPolicy()
			p.Revision = "2"
			p.ResidentModels = nodepolicy.ResidentModels{{Engine: "ollama", Model: "n:latest"}}
			switch change {
			case "profile removed":
				p.Profiles = p.Profiles[1:]
			case "bound reduced":
				p.Profiles[0].MemoryBytes = 1
			case "qualification withdrawn":
				p.Profiles[0].Qualification = nil
			}
			h.a.setPolicy(p)
			h.a.setResidency(map[string][]string{"ollama": {}})
			in.OperationID, in.PolicyRevision, in.Model = "resident:2", "2", "n:latest"
			result, err := h.a.prepareResident("proxy", in)
			if err != nil || len(result.Refusals) != 1 || result.Refusals[0].Reason != nodepolicy.ModelReasonNoFit || h.a.currentActive() != 1 {
				t.Fatalf("prepared bound lost: result=%+v error=%v active=%d", result, err, h.a.currentActive())
			}
			mustReject(t, h.admit("ollama", "n:latest", 0), nodepolicy.RejectNoFit)
		})
	}
}

func TestServedResidentHardProtectionAndConcreteUnknownMemory(t *testing.T) {
	h, _ := residentHarness(t, "m:latest", "n:latest")
	p := h.a.currentPolicy()
	p.Admission.MaxResidentModels = 1
	p.Admission.SwitchModels = true
	h.a.setPolicy(p)
	h.a.setResidency(map[string][]string{"ollama": {"m:latest"}})
	mustReject(t, h.admit("ollama", "n:latest", 0), nodepolicy.RejectNoFit)
	if len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 0 {
		t.Fatal("ordinary demand evicted protected resident")
	}
	p = h.a.currentPolicy()
	p.Admission.MaxResidentModels = 2
	p.Profiles[1].MemoryBytes = 0
	h.a.setPolicy(p)
	mustReject(t, h.admit("ollama", "n:latest", 0), nodepolicy.RejectNoFit)
	// Removing the preference restores existing concrete unknown-memory behavior.
	p.ResidentModels = nil
	h.a.setPolicy(p)
	decision := h.admit("ollama", "n:latest", 0)
	if decision.ticket == nil {
		t.Fatalf("legacy concrete behavior changed: %+v", decision)
	}
	decision.ticket.release()
}

func TestServedResidentConcreteSwitchTimeoutDoesNotBypassProtection(t *testing.T) {
	h, _ := residentHarness(t, "m:latest", "n:latest", "idle:latest")
	p := h.a.currentPolicy()
	p.Admission.MaxResidentModels = 2
	p.Admission.SwitchModels = true
	p.Admission.SwitchTimeoutSeconds = 0
	h.a.setPolicy(p)
	h.a.setResidency(map[string][]string{"ollama": {"m:latest", "idle:latest"}})
	h.a.mu.Lock()
	plan, evict := h.a.fitLocked(admKey{"ollama", "n:latest"}, p)
	h.a.mu.Unlock()
	if plan != fitEvict || !reflect.DeepEqual(evict, []admKey{{"ollama", "idle:latest"}}) {
		t.Fatalf("fit=%v evict=%+v, want only unprotected idle model", plan, evict)
	}
	mustReject(t, h.admit("ollama", "n:latest", 0), nodepolicy.RejectNoFit)
	if h.a.currentActive() != 0 {
		t.Fatal("switch timeout granted over protected capacity")
	}
	// On an engine with no protected residents the existing timeout applies.
	p.ResidentModels = nil
	h.a.setPolicy(p)
	decision := h.admit("ollama", "n:latest", 0)
	if decision.ticket == nil {
		t.Fatalf("legacy switch-timeout behavior changed: %+v", decision)
	}
	decision.ticket.release()
}

func TestServedResidentUnknownMemorySameModelFailsBeforeWarmShortcut(t *testing.T) {
	h, _ := residentHarness(t, "m:latest")
	p := h.a.currentPolicy()
	p.Profiles[0].MemoryBytes = 0
	h.a.setPolicy(p)
	h.a.setResidency(map[string][]string{"ollama": {"m:latest"}})
	mustReject(t, h.admit("ollama", "m:latest", 0), nodepolicy.RejectNoFit)
}

func TestResidentCapacityFinalDispatchRechecksActualCapacity(t *testing.T) {
	h, in := residentHarness(t, "m:latest", "n:latest")
	p := h.a.currentPolicy()
	p.Admission.MaxResidentModels = 1
	h.a.setPolicy(p)
	prepared := prepareResidentOK(t, h.a, in)
	// This report is evidence of additional use, never permission to overbook.
	h.a.setResidency(map[string][]string{"ollama": {"n:latest"}})
	result, err := h.a.dispatchResident("proxy", residentDispatch(in, prepared))
	if err != nil || result.Dispatched || result.Reason != nodepolicy.DispatchResidentReason(nodepolicy.ModelReasonNoFit) || h.a.currentActive() != 0 {
		t.Fatalf("actual capacity changed after prepare: result=%+v error=%v active=%d", result, err, h.a.currentActive())
	}
}

func TestServedResidentRevokeRestoreDoesNotReviveGrant(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	prepared := prepareResidentOK(t, h.a, in)
	h.a.setDrain("ollama", true)
	h.a.setDrain("ollama", false)
	result, err := h.a.dispatchResident("proxy", residentDispatch(in, prepared))
	if err != nil || result.Dispatched || h.a.currentActive() != 0 {
		t.Fatalf("revived grant=%+v error=%v", result, err)
	}
}

func TestServedResidentDuplicateDispatchDoesNotGrantAgain(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	prepared := prepareResidentOK(t, h.a, in)
	dispatchResidentOK(t, h.a, in, prepared)
	result, err := h.a.dispatchResident("proxy", residentDispatch(in, prepared))
	if err != nil || result.Dispatched || result.Reason == "" || h.a.currentActive() != 1 {
		t.Fatalf("duplicate dispatch=%+v error=%v", result, err)
	}
}

func TestServedResidentDispatchAndRevocationLinearize(t *testing.T) {
	for _, order := range []string{"revocation first", "dispatch first"} {
		t.Run(order, func(t *testing.T) {
			h, in := residentHarness(t, "m:latest")
			prepared := prepareResidentOK(t, h.a, in)
			firstDone := make(chan struct{})
			type dispatchAnswer struct {
				result nodepolicy.DispatchResidentResult
				err    error
			}
			answer := make(chan dispatchAnswer, 1)
			if order == "revocation first" {
				go func() { h.a.setAvailability(nodepolicy.Paused, false); close(firstDone) }()
				go func() {
					<-firstDone
					result, err := h.a.dispatchResident("proxy", residentDispatch(in, prepared))
					answer <- dispatchAnswer{result, err}
				}()
			} else {
				go func() {
					result, err := h.a.dispatchResident("proxy", residentDispatch(in, prepared))
					answer <- dispatchAnswer{result, err}
					close(firstDone)
				}()
				<-firstDone
				h.a.setAvailability(nodepolicy.Paused, false)
			}
			got := <-answer
			if got.err != nil {
				t.Fatal(got.err)
			}
			wantCount := 0
			if order == "dispatch first" {
				wantCount = 1
			}
			if got.result.Dispatched != (order == "dispatch first") || h.a.currentActive() != wantCount {
				t.Fatalf("dispatch=%+v active=%d want active=%d", got.result, h.a.currentActive(), wantCount)
			}
		})
	}
}

func TestResidentCapacityProxyReplacementRejectsOldCompletion(t *testing.T) {
	old, in := residentHarness(t, "m:latest")
	oldPrepared := prepareResidentOK(t, old.a, in)
	dispatchResidentOK(t, old.a, in, oldPrepared)
	next, nextIn := residentHarness(t, "m:latest")
	nextIn.ProxyRunID = "new-proxy"
	prepared, err := next.a.prepareResident("new-proxy", nextIn)
	if err != nil || prepared.ReservationID == "" {
		t.Fatalf("new proxy prepare=%+v error=%v", prepared, err)
	}
	finish := residentFinish(in, oldPrepared, nodepolicy.ResidentNotDispatched)
	if _, err := next.a.finishResident("new-proxy", finish); err == nil || next.a.currentActive() != 1 {
		t.Fatal("old proxy completion cleared replacement hold with reused operation/reservation IDs")
	}
}

func TestResidentCapacityOldGenerationCannotReleaseNextHold(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	prepared := prepareResidentOK(t, h.a, in)
	dispatchResidentOK(t, h.a, in, prepared)
	finish := residentFinish(in, prepared, nodepolicy.ResidentFailed)
	finish.Observation = residentObservationFor(in, 7, false)
	finishResidentOK(t, h.a, finish)
	in.OperationID = "resident:2"
	prepared = prepareResidentOK(t, h.a, in)
	dispatchResidentOK(t, h.a, in, prepared)
	finish = residentFinish(in, prepared, nodepolicy.ResidentFailed)
	finish.Observation = residentObservationFor(in, 7, false)
	if _, err := h.a.finishResident("proxy", finish); err == nil || h.a.currentActive() != 1 {
		t.Fatal("old observation generation cleared next hold")
	}
	finish.Observation.Generation = 8
	finishResidentOK(t, h.a, finish)
	if h.a.currentActive() != 0 {
		t.Fatal("new matching post-dispatch observation did not release")
	}
}

func TestResidentCapacityFinishedLoadConvergesAfterPreferenceRemoval(t *testing.T) {
	h, in := residentHarness(t, "m:latest", "n:latest")
	p := h.a.currentPolicy()
	p.Admission.MaxResidentModels = 1
	h.a.setPolicy(p)
	prepared := prepareResidentOK(t, h.a, in)
	dispatchResidentOK(t, h.a, in, prepared)
	finish := residentFinish(in, prepared, nodepolicy.ResidentLoaded)
	finish.Observation = residentObservationFor(in, 1, true)
	finishResidentOK(t, h.a, finish)
	p = h.a.currentPolicy()
	p.Revision = "2"
	p.ResidentModels = nodepolicy.ResidentModels{{Engine: "ollama", Model: "n:latest"}}
	h.a.setPolicy(p)
	// SR5 performs the normal unload; its ordinary loaded watcher reports empty.
	h.a.setResidency(map[string][]string{"ollama": {}})
	in.OperationID, in.PolicyRevision, in.Model = "resident:2", "2", "n:latest"
	prepareResidentOK(t, h.a, in)
	if h.a.currentActive() != 1 {
		t.Fatal("settled load retained a phantom capacity hold")
	}
}

func TestResidentCapacityUnrelatedObservationGenerationDoesNotStrandFinish(t *testing.T) {
	h, first := residentHarness(t, "m:latest", "n:latest")
	p := h.a.currentPolicy()
	p.ResidentModels = append(p.ResidentModels, nodepolicy.Role{Engine: "ollama", Model: "n:latest"})
	h.a.setPolicy(p)
	firstPrepared := prepareResidentOK(t, h.a, first)
	second := first
	second.OperationID, second.Model = "resident:2", "n:latest"
	secondPrepared := prepareResidentOK(t, h.a, second)
	dispatchResidentOK(t, h.a, first, firstPrepared)
	dispatchResidentOK(t, h.a, second, secondPrepared)
	secondFinish := residentFinish(second, secondPrepared, nodepolicy.ResidentLoaded)
	secondFinish.Observation = residentObservationFor(second, 2, true)
	finishResidentOK(t, h.a, secondFinish)
	firstFinish := residentFinish(first, firstPrepared, nodepolicy.ResidentLoaded)
	firstFinish.Observation = residentObservationFor(first, 1, true)
	finishResidentOK(t, h.a, firstFinish)
	if h.a.currentActive() != 0 || !h.a.isLoaded("ollama", first.Model) || !h.a.isLoaded("ollama", second.Model) {
		t.Fatal("unrelated model generation stranded a matching authoritative completion")
	}
}

func TestResidentCapacityDuplicateFinishRace(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	prepared := prepareResidentOK(t, h.a, in)
	finish := residentFinish(in, prepared, nodepolicy.ResidentNotDispatched)
	var wg sync.WaitGroup
	errors := make(chan error, 2)
	start := make(chan struct{})
	for range 2 {
		wg.Add(1)
		go func() { defer wg.Done(); <-start; _, err := h.a.finishResident("proxy", finish); errors <- err }()
	}
	close(start)
	wg.Wait()
	for range 2 {
		if err := <-errors; err != nil {
			t.Fatal(err)
		}
	}
	if h.a.currentActive() != 0 || len(h.a.inflight) != 0 {
		t.Fatal("concurrent finishes did not release exactly once")
	}
}

func TestResidentCapacityBoundNeverEvictsUnresolvedDispatch(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	p := h.a.currentPolicy()
	p.Admission.MaxResidentModels, p.Admission.MemoryBudgetBytes = 0, 0
	h.a.setPolicy(p)
	prepared := prepareResidentOK(t, h.a, in)
	dispatchResidentOK(t, h.a, in, prepared)
	unknown := finishResidentOK(t, h.a, residentFinish(in, prepared, nodepolicy.ResidentUnknown))
	for len(h.a.resident.ops) < maxResidentOperations {
		sequence := uint64(len(h.a.resident.ops) + 1)
		request := in
		request.OperationID = fmt.Sprintf("resident:%d", sequence)
		request.Model = fmt.Sprintf("held-%d:latest", sequence)
		key := admKey{request.Engine, request.Model}
		h.a.resident.ops[request.OperationID] = &residentHold{request: request, sequence: sequence, key: key, counted: true, dispatched: true, revoked: true, memory: 40}
		h.a.inflight[key]++
		h.a.active++
		h.a.activeByEngine[key.engine]++
	}
	h.a.resident.highestSequence = maxResidentOperations
	next := in
	next.OperationID = fmt.Sprintf("resident:%d", maxResidentOperations+1)
	result, err := h.a.prepareResident("proxy", next)
	if err != nil || len(result.Refusals) != 1 || result.Refusals[0].Reason != nodepolicy.ModelReasonNoFit || h.a.currentActive() != maxResidentOperations || h.a.resident.ops[in.OperationID] == nil {
		t.Fatalf("bounded retention dropped dispatch: %+v error=%v", result, err)
	}
	if _, err := h.a.prepareResident("proxy", next); err == nil || !strings.Contains(err.Error(), "retired") {
		t.Fatal("unrecorded saturation refusal became reusable")
	}
	observeResidentOK(t, h.a, residentRecovery(in, unknown, 1, false))
	next.OperationID = residentOperation(t, maxResidentOperations+2)
	prepareResidentOK(t, h.a, next)
	if h.a.currentActive() != maxResidentOperations || len(h.a.resident.ops) > maxResidentOperations {
		t.Fatal("settled saturation did not resume within metadata bound")
	}
}

func TestServedResidentRPCs(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	p, writer := policyProxy(t)
	p.admission = h.a
	in.ProxyRunID = p.runID
	response := rpc(t, p, writer, nodepolicy.MethodPrepareResident, in)
	var prepared nodepolicy.PrepareResidentResult
	if response.Error != nil {
		t.Fatalf("prepare RPC: %+v", response.Error)
	}
	if err := json.Unmarshal(response.Result, &prepared); err != nil || prepared.ReservationID == "" {
		t.Fatalf("prepare RPC result=%s error=%v", response.Result, err)
	}
	response = rpc(t, p, writer, nodepolicy.MethodDispatchResident, residentDispatch(in, prepared))
	var dispatched nodepolicy.DispatchResidentResult
	if response.Error != nil {
		t.Fatalf("dispatch RPC: %+v", response.Error)
	}
	if err := json.Unmarshal(response.Result, &dispatched); err != nil || !dispatched.Dispatched {
		t.Fatalf("dispatch RPC result=%s error=%v", response.Result, err)
	}
	finish := residentFinish(in, prepared, nodepolicy.ResidentLoaded)
	finish.Observation = residentObservationFor(in, 1, true)
	response = rpc(t, p, writer, nodepolicy.MethodFinishResident, finish)
	var finished nodepolicy.FinishResidentResult
	if response.Error != nil {
		t.Fatalf("finish RPC: %+v", response.Error)
	}
	if err := json.Unmarshal(response.Result, &finished); err != nil || finished.Outcome != nodepolicy.ResidentLoaded {
		t.Fatalf("finish RPC result=%s error=%v", response.Result, err)
	}
	for _, method := range []string{nodepolicy.MethodPrepareResident, nodepolicy.MethodDispatchResident, nodepolicy.MethodFinishResident, nodepolicy.MethodObserveResident} {
		response := rpc(t, p, writer, method, json.RawMessage(`{"operationId":"op","OperationId":"other"}`))
		if response.Error == nil || response.Error.Code != -32602 {
			t.Fatalf("malformed %s accepted: %+v", method, response)
		}
	}
}

func TestResidentCapacityRecoveryPreservesImmutableFinish(t *testing.T) {
	for _, loaded := range []bool{false, true} {
		t.Run(fmt.Sprintf("loaded=%t", loaded), func(t *testing.T) {
			h, in := residentHarness(t, "m:latest")
			prepared := prepareResidentOK(t, h.a, in)
			dispatchResidentOK(t, h.a, in, prepared)
			finish := residentFinish(in, prepared, nodepolicy.ResidentUnknown)
			original := finishResidentOK(t, h.a, finish)
			if original.RecoveryID == "" {
				t.Fatal("unknown dispatch has no recovery binding")
			}
			h.a.setResidency(map[string][]string{"ollama": {}})
			if h.a.currentActive() != 1 {
				t.Fatal("legacy empty recovered uncertainty")
			}
			recovery := residentRecovery(in, original, 2, loaded)
			receipt := observeResidentOK(t, h.a, recovery)
			if duplicate := observeResidentOK(t, h.a, recovery); duplicate != receipt {
				t.Fatal("recovery duplicate changed receipt")
			}
			if h.a.currentActive() != 0 || h.a.isLoaded(in.Engine, in.Model) != loaded {
				t.Fatal("authoritative recovery did not settle capacity truthfully")
			}
			if duplicate := finishResidentOK(t, h.a, finish); duplicate != original {
				t.Fatal("recovery rewrote terminal unknown outcome or binding")
			}
			changed := finish
			changed.Observation = residentObservationFor(in, 99, loaded)
			if _, err := h.a.finishResident("proxy", changed); err == nil {
				t.Fatal("recovery weakened immutable finish contract")
			}
			recovery.Sequence = 1
			if _, err := h.a.observeResident("proxy", recovery); err == nil {
				t.Fatal("stale recovery sequence accepted after settlement")
			}
		})
	}
}

func TestResidentCapacityRecoveryBindingRefusals(t *testing.T) {
	for _, field := range []string{"proxy", "reservation", "operation", "recovery", "manager", "engine", "model", "identity", "sequence", "incomplete", "current identity"} {
		t.Run(field, func(t *testing.T) {
			h, in := residentHarness(t, "m:latest")
			prepared := prepareResidentOK(t, h.a, in)
			dispatchResidentOK(t, h.a, in, prepared)
			finish := finishResidentOK(t, h.a, residentFinish(in, prepared, nodepolicy.ResidentTimeout))
			recovery := residentRecovery(in, finish, 1, false)
			switch field {
			case "proxy":
				recovery.ProxyRunID = "old-proxy"
			case "reservation":
				recovery.ReservationID = "old-reservation"
			case "operation":
				recovery.OperationID = "resident:2"
			case "recovery":
				recovery.RecoveryID = "old-recovery"
			case "manager":
				recovery.Observation.ManagerIncarnation = "old-manager"
			case "engine":
				recovery.Observation.Engine = "lmstudio"
			case "model":
				recovery.Observation.Model = "other:latest"
			case "identity":
				recovery.Observation.Identity.EngineRunID = "old-engine"
			case "sequence":
				recovery.Sequence = 0
			case "incomplete":
				recovery.OperationComplete = false
			case "current identity":
				h.a.setModelIdentity(map[string]nodepolicy.EngineIdentity{})
			}
			if _, err := h.a.observeResident("proxy", recovery); err == nil || h.a.currentActive() != 1 {
				t.Fatal("stale or incomplete recovery released capacity")
			}
		})
	}
}

func TestResidentCapacityRecoveryRequiresTerminalFinish(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	prepared := prepareResidentOK(t, h.a, in)
	dispatchResidentOK(t, h.a, in, prepared)
	fake := nodepolicy.FinishResidentResult{ReservationID: prepared.ReservationID, Outcome: nodepolicy.ResidentUnknown, RecoveryID: "recovery-1"}
	if _, err := h.a.observeResident("proxy", residentRecovery(in, fake, 1, false)); err == nil || h.a.currentActive() != 1 {
		t.Fatal("observation settled an operation with no terminal finish")
	}
}

func TestResidentCapacityLateRecoveryCannotReleaseReplacement(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	prepared := prepareResidentOK(t, h.a, in)
	dispatchResidentOK(t, h.a, in, prepared)
	finish := finishResidentOK(t, h.a, residentFinish(in, prepared, nodepolicy.ResidentCancelled))
	recovery := residentRecovery(in, finish, 1, false)
	observeResidentOK(t, h.a, recovery)
	in.OperationID = "resident:2"
	prepareResidentOK(t, h.a, in)
	observeResidentOK(t, h.a, recovery)
	if h.a.currentActive() != 1 {
		t.Fatal("duplicate old recovery cleared replacement hold")
	}
	recovery.Sequence++
	if _, err := h.a.observeResident("proxy", recovery); err == nil || h.a.currentActive() != 1 {
		t.Fatal("changed old recovery cleared replacement hold")
	}
}

func TestResidentCapacitySuccessfulLoadsRetireWithoutReplay(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	clock := useStrictClock(h.a)
	first := in
	var firstPrepared nodepolicy.PrepareResidentResult
	for sequence := uint64(1); sequence <= maxResidentOperations+4; sequence++ {
		in.OperationID = residentOperation(t, sequence)
		prepared := prepareResidentOK(t, h.a, in)
		if sequence == 1 {
			firstPrepared = prepared
		}
		dispatchResidentOK(t, h.a, in, prepared)
		finish := residentFinish(in, prepared, nodepolicy.ResidentLoaded)
		finish.Observation = residentObservationFor(in, sequence, true)
		finishResidentOK(t, h.a, finish)
		if len(h.a.resident.ops) > maxResidentOperations {
			t.Fatal("completed history is unbounded")
		}
	}
	clock.Advance(365 * 24 * time.Hour)
	for _, changed := range []bool{false, true} {
		request := first
		if changed {
			request.TimeoutMs--
		}
		if _, err := h.a.prepareResident("proxy", request); err == nil || !strings.Contains(err.Error(), "retired") {
			t.Fatal("retired request received fresh evaluation")
		}
	}
	if _, err := h.a.finishResident("proxy", residentFinish(first, firstPrepared, nodepolicy.ResidentNotDispatched)); err == nil || !strings.Contains(err.Error(), "retired") {
		t.Fatal("retired finish was not explicitly rejected")
	}
	if h.a.currentActive() != 0 || len(h.a.resident.generations) != 1 || h.a.resident.generations["manager"] != maxResidentOperations+4 {
		t.Fatal("retirement lost current generation floor or retained capacity")
	}
}

func TestResidentCapacityUncertainHoldSurvivesCompletedChurn(t *testing.T) {
	h, first := residentHarness(t, "m:latest", "n:latest")
	p := h.a.currentPolicy()
	p.ResidentModels = append(p.ResidentModels, nodepolicy.Role{Engine: "ollama", Model: "n:latest"})
	h.a.setPolicy(p)
	prepared := prepareResidentOK(t, h.a, first)
	dispatchResidentOK(t, h.a, first, prepared)
	finish := finishResidentOK(t, h.a, residentFinish(first, prepared, nodepolicy.ResidentUnknown))
	for sequence := uint64(2); sequence <= maxResidentOperations+3; sequence++ {
		in := first
		in.OperationID, in.Model = residentOperation(t, sequence), "n:latest"
		next := prepareResidentOK(t, h.a, in)
		finishResidentOK(t, h.a, residentFinish(in, next, nodepolicy.ResidentNotDispatched))
	}
	if h.a.currentActive() != 1 || h.a.resident.ops[first.OperationID] == nil || !h.a.resident.ops[first.OperationID].counted || h.a.memoryLocked(admKey{"ollama", first.Model}) != 40 {
		t.Fatal("retirement evicted unresolved capacity or its prepared bound")
	}
	observeResidentOK(t, h.a, residentRecovery(first, finish, 1, false))
	if h.a.currentActive() != 0 {
		t.Fatal("old unresolved operation lost its recovery route during churn")
	}
}

func TestResidentCapacitySourceGenerationMetadataIsBounded(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	for sequence := uint64(1); sequence <= 64; sequence++ {
		in.OperationID, in.ManagerIncarnation = residentOperation(t, sequence), fmt.Sprintf("source-%d", sequence)
		prepared := prepareResidentOK(t, h.a, in)
		dispatchResidentOK(t, h.a, in, prepared)
		finish := residentFinish(in, prepared, nodepolicy.ResidentFailed)
		finish.Observation = residentObservationFor(in, 1, false)
		finishResidentOK(t, h.a, finish)
		if len(h.a.resident.generations) != 1 || h.a.resident.generations[in.ManagerIncarnation] != 1 {
			t.Fatal("completed source history grew or lost current source floor")
		}
	}
}

func TestResidentCapacityCurrentSourceFloorSurvivesLastRecordRetirement(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	prepared := prepareResidentOK(t, h.a, in)
	dispatchResidentOK(t, h.a, in, prepared)
	finish := residentFinish(in, prepared, nodepolicy.ResidentFailed)
	finish.Observation = residentObservationFor(in, 7, false)
	finishResidentOK(t, h.a, finish)
	fillReleasedResidentHistory(t, h.a, in)
	h.a.setAvailability(nodepolicy.Paused, false)
	refused := in
	refused.OperationID = residentOperation(t, maxResidentOperations+1)
	if result, err := h.a.prepareResident("proxy", refused); err != nil || len(result.Refusals) != 1 {
		t.Fatalf("paused prepare=%+v error=%v", result, err)
	}
	if h.a.resident.ops[in.OperationID] != nil || h.a.resident.generations["manager"] != 7 {
		t.Fatal("retiring last completed source record lost its current floor")
	}
	h.a.setAvailability(nodepolicy.Available, false)
	in.OperationID = residentOperation(t, maxResidentOperations+2)
	prepared = prepareResidentOK(t, h.a, in)
	dispatchResidentOK(t, h.a, in, prepared)
	finish = residentFinish(in, prepared, nodepolicy.ResidentFailed)
	finish.Observation = residentObservationFor(in, 7, false)
	if _, err := h.a.finishResident("proxy", finish); err == nil || h.a.currentActive() != 1 {
		t.Fatal("retirement allowed old generation to settle a new dispatch")
	}
}

func TestResidentCapacitySourceChurnKeepsCountedSourceFloor(t *testing.T) {
	h, in := residentHarness(t, "m:latest", "n:latest")
	p := h.a.currentPolicy()
	p.ResidentModels = append(p.ResidentModels, nodepolicy.Role{Engine: "ollama", Model: "n:latest"})
	h.a.setPolicy(p)
	in.Model = "n:latest"
	prepared := prepareResidentOK(t, h.a, in)
	dispatchResidentOK(t, h.a, in, prepared)
	finish := residentFinish(in, prepared, nodepolicy.ResidentFailed)
	finish.Observation = residentObservationFor(in, 7, false)
	finishResidentOK(t, h.a, finish)
	in.OperationID, in.Model = "resident:2", "m:latest"
	prepared = prepareResidentOK(t, h.a, in)
	dispatchResidentOK(t, h.a, in, prepared)
	finishResidentOK(t, h.a, residentFinish(in, prepared, nodepolicy.ResidentUnknown))
	for sequence := uint64(3); sequence <= 66; sequence++ {
		next := in
		next.OperationID, next.Model, next.ManagerIncarnation = residentOperation(t, sequence), "n:latest", fmt.Sprintf("bookkeeping-source-%d", sequence)
		prepared := prepareResidentOK(t, h.a, next)
		dispatchResidentOK(t, h.a, next, prepared)
		finish := residentFinish(next, prepared, nodepolicy.ResidentFailed)
		finish.Observation = residentObservationFor(next, 1, false)
		finishResidentOK(t, h.a, finish)
		if len(h.a.resident.generations) != 2 || h.a.resident.generations["manager"] != 7 || h.a.currentActive() != 1 {
			t.Fatal("source churn dropped live source floor or retained completed source history")
		}
	}
}

func TestResidentCapacityDuplicateAndRetirementLinearize(t *testing.T) {
	for _, order := range []string{"duplicate first", "retirement first"} {
		t.Run(order, func(t *testing.T) {
			h, in := residentHarness(t, "m:latest")
			prepared := prepareResidentOK(t, h.a, in)
			finishResidentOK(t, h.a, residentFinish(in, prepared, nodepolicy.ResidentNotDispatched))
			fillReleasedResidentHistory(t, h.a, in)
			next := in
			next.OperationID = residentOperation(t, maxResidentOperations+1)
			firstDone := make(chan struct{})
			type answer struct {
				result nodepolicy.PrepareResidentResult
				err    error
			}
			duplicate, retirement := make(chan answer, 1), make(chan answer, 1)
			if order == "duplicate first" {
				go func() {
					result, err := h.a.prepareResident("proxy", in)
					duplicate <- answer{result, err}
					close(firstDone)
				}()
				go func() {
					<-firstDone
					result, err := h.a.prepareResident("proxy", next)
					retirement <- answer{result, err}
				}()
			} else {
				go func() {
					result, err := h.a.prepareResident("proxy", next)
					retirement <- answer{result, err}
					close(firstDone)
				}()
				go func() { <-firstDone; result, err := h.a.prepareResident("proxy", in); duplicate <- answer{result, err} }()
			}
			old, fresh := <-duplicate, <-retirement
			if fresh.err != nil || fresh.result.ReservationID == "" || h.a.currentActive() != 1 {
				t.Fatalf("new operation=%+v error=%v active=%d", fresh.result, fresh.err, h.a.currentActive())
			}
			if order == "duplicate first" {
				if old.err != nil || !reflect.DeepEqual(old.result, prepared) {
					t.Fatal("retained duplicate lost original result")
				}
			} else if old.err == nil || !strings.Contains(old.err.Error(), "retired") {
				t.Fatal("retired duplicate received new grant")
			}
		})
	}
}

func TestResidentCapacityRecoveryAndRetirementLinearize(t *testing.T) {
	for _, order := range []string{"recovery first", "retirement first"} {
		t.Run(order, func(t *testing.T) {
			h, in := residentHarness(t, "m:latest", "n:latest")
			p := h.a.currentPolicy()
			p.ResidentModels = append(p.ResidentModels, nodepolicy.Role{Engine: "ollama", Model: "n:latest"})
			h.a.setPolicy(p)
			prepared := prepareResidentOK(t, h.a, in)
			recoveryNotGranted := nodepolicy.FinishResidentResult{ReservationID: prepared.ReservationID, Outcome: nodepolicy.ResidentUnknown, RecoveryID: "recovery-1"}
			if _, err := h.a.observeResident("proxy", residentRecovery(in, recoveryNotGranted, 1, false)); err == nil {
				t.Fatal("undispatched observation was accepted")
			}
			dispatchResidentOK(t, h.a, in, prepared)
			unknown := finishResidentOK(t, h.a, residentFinish(in, prepared, nodepolicy.ResidentUnknown))
			fillReleasedResidentHistory(t, h.a, in)
			recovery := residentRecovery(in, unknown, 1, false)
			next := in
			next.OperationID, next.Model = residentOperation(t, maxResidentOperations+1), "n:latest"
			firstDone := make(chan struct{})
			errors := make(chan error, 2)
			if order == "recovery first" {
				go func() { _, err := h.a.observeResident("proxy", recovery); errors <- err; close(firstDone) }()
				go func() { <-firstDone; _, err := h.a.prepareResident("proxy", next); errors <- err }()
			} else {
				go func() { _, err := h.a.prepareResident("proxy", next); errors <- err; close(firstDone) }()
				go func() { <-firstDone; _, err := h.a.observeResident("proxy", recovery); errors <- err }()
			}
			for range 2 {
				if err := <-errors; err != nil {
					t.Fatal(err)
				}
			}
			if h.a.currentActive() != 1 || !h.a.resident.ops[next.OperationID].counted {
				t.Fatal("recovery/retirement cleared replacement capacity")
			}
			_, err := h.a.observeResident("proxy", recovery)
			if order == "recovery first" && (err == nil || !strings.Contains(err.Error(), "retired")) {
				t.Fatal("retired recovery was not explicitly rejected")
			}
			if order == "retirement first" && err != nil {
				t.Fatalf("retained recovery receipt lost: %v", err)
			}
			if h.a.currentActive() != 1 {
				t.Fatal("late recovery changed replacement count")
			}
		})
	}
}

func TestResidentCapacitySequenceConsumption(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	for _, operation := range []string{"arbitrary-op", "resident:0", "resident:2"} {
		request := in
		request.OperationID = operation
		if _, err := h.a.prepareResident("proxy", request); err == nil || h.a.resident.highestSequence != 0 {
			t.Fatal("malformed ID or gap advanced sequence")
		}
	}
	wrongRun := in
	wrongRun.ProxyRunID = "old-proxy"
	if _, err := h.a.prepareResident("proxy", wrongRun); err == nil || h.a.resident.highestSequence != 0 {
		t.Fatal("wrong run advanced sequence")
	}
	h.a.setAvailability(nodepolicy.Paused, false)
	refused, err := h.a.prepareResident("proxy", in)
	if err != nil || len(refused.Refusals) != 1 || h.a.resident.highestSequence != 1 {
		t.Fatalf("refusal=%+v error=%v", refused, err)
	}
	h.a.setAvailability(nodepolicy.Available, false)
	again, err := h.a.prepareResident("proxy", in)
	if err != nil || !reflect.DeepEqual(again, refused) || h.a.currentActive() != 0 {
		t.Fatal("consumed refusal turned into a grant after evidence changed")
	}
	in.OperationID = "resident:2"
	prepareResidentOK(t, h.a, in)
}

func TestResidentCapacitySequenceOverflowFailsClosed(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	h.a.resident.highestSequence = ^uint64(0) - 1
	in.OperationID = residentOperation(t, ^uint64(0))
	prepared := prepareResidentOK(t, h.a, in)
	finishResidentOK(t, h.a, residentFinish(in, prepared, nodepolicy.ResidentNotDispatched))
	in.OperationID = "resident:18446744073709551616"
	if _, err := h.a.prepareResident("proxy", in); err == nil || h.a.resident.highestSequence != ^uint64(0) || h.a.currentActive() != 0 {
		t.Fatal("sequence overflow reopened operation stream")
	}
}

func TestServedResidentRecoveryRPC(t *testing.T) {
	h, in := residentHarness(t, "m:latest")
	p, writer := policyProxy(t)
	p.admission, in.ProxyRunID = h.a, p.runID
	response := rpc(t, p, writer, nodepolicy.MethodPrepareResident, in)
	var prepared nodepolicy.PrepareResidentResult
	if response.Error != nil {
		t.Fatalf("prepare: %+v", response.Error)
	}
	if err := json.Unmarshal(response.Result, &prepared); err != nil {
		t.Fatal(err)
	}
	response = rpc(t, p, writer, nodepolicy.MethodDispatchResident, residentDispatch(in, prepared))
	if response.Error != nil {
		t.Fatalf("dispatch: %+v", response.Error)
	}
	response = rpc(t, p, writer, nodepolicy.MethodFinishResident, residentFinish(in, prepared, nodepolicy.ResidentUnknown))
	var finish nodepolicy.FinishResidentResult
	if response.Error != nil {
		t.Fatalf("finish: %+v", response.Error)
	}
	if err := json.Unmarshal(response.Result, &finish); err != nil || finish.RecoveryID == "" {
		t.Fatalf("finish=%s error=%v", response.Result, err)
	}
	response = rpc(t, p, writer, nodepolicy.MethodObserveResident, residentRecovery(in, finish, 1, false))
	var observed nodepolicy.ObserveResidentResult
	if response.Error != nil {
		t.Fatalf("observe: %+v", response.Error)
	}
	if err := json.Unmarshal(response.Result, &observed); err != nil || !observed.Settled || h.a.currentActive() != 0 {
		t.Fatalf("observe=%s error=%v active=%d", response.Result, err, h.a.currentActive())
	}
}
