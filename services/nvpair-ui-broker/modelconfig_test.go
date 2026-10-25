// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"reflect"
	"sync"
	"testing"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

type modelConfigFixture struct {
	*strictFixture
	source *engineManagerSource
}

func newModelConfigFixture(t *testing.T) *modelConfigFixture {
	t.Helper()
	f := &modelConfigFixture{strictFixture: newStrictFixture(t)}
	b := f.b
	f.source = b.activateEngineManagerSource()
	b.workersMu.Lock()
	worker := b.engineMgr
	b.workersMu.Unlock()
	b.bindEngineMgr(worker, f.source)
	seedStrictIdentity(b, "llamacpp", "resident")
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Profiles = []nodepolicy.Profile{{Name: "resident", Engine: "llamacpp", Model: "resident", ContextTokens: 8192, MaxOutputTokens: 1024, MaxConcurrent: 1, MemoryBytes: 1024,
		Qualification: &nodepolicy.QualificationRef{ID: "fixture-only", Revision: "1"}}}
	r.current.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: "resident"}}
	r.current.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "resident"}}
	r.intent = map[string]bool{"llamacpp": true}
	r.running = map[string]bool{"llamacpp": true}
	r.loaded = map[string][]string{"llamacpp": {}}
	r.models.loadedSource = f.source
	r.models.loadedProxy = f.pp
	id := r.engineIdentity["llamacpp"]
	ram, vram := int64(1024), int64(0)
	id.Serving = map[string]engines.ServingEnvelope{"resident": {
		ModelID: "resident", ContentDigest: id.ModelDigests["resident"], NativeSourceDigest: id.LaunchDigest,
		PerSlotContextTokens: 8192, TotalContextTokens: 8192, Slots: 1, MaxOutputTokens: 1024,
		Memory: engines.ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram}, Retention: "-1", RouterCapacity: 2,
	}}
	r.engineIdentity["llamacpp"] = id
	r.mu.Unlock()
	if !b.replayPolicyToProxy(context.Background(), f.pp) {
		t.Fatal("activate resident fixture policy")
	}
	reportStrict(t, b, "0", f.acks.runID, strictCapability(t, "llamacpp", "resident"))
	f.proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
		switch method {
		case nodepolicy.MethodPrepareResident:
			return nodepolicy.PrepareResidentResult{ReservationID: "fixture-reservation", EvidenceGeneration: 1}, ""
		case nodepolicy.MethodDispatchResident:
			return nodepolicy.DispatchResidentResult{Dispatched: true}, ""
		case nodepolicy.MethodFinishResident:
			var p nodepolicy.FinishResidentParams
			if err := json.Unmarshal(params, &p); err != nil {
				return nil, err.Error()
			}
			return nodepolicy.FinishResidentResult{ReservationID: p.ReservationID, Outcome: p.Outcome}, ""
		}
		return f.acks.answer(method, params)
	})
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			return emptySwitchManagerSnapshot(), ""
		}
		if method == nodepolicy.MethodReconcileResident {
			return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
		}
		return map[string]bool{"ok": true}, ""
	})
	e := b.captureModelConfig()
	entry, _, _, _ := b.modelConfigEntry(e, nodepolicy.Role{Engine: "llamacpp", Model: "resident"})
	if entry.State != nodepolicy.ModelStatePending {
		t.Fatalf("resident fixture is not eligible: %+v", entry)
	}
	return f
}

func modelConfigCalls(f *fakeWorker, method string) []policyCall {
	var calls []policyCall
	for _, call := range f.snapshot() {
		if call.method == method {
			calls = append(calls, call)
		}
	}
	return calls
}

func decodeModelConfigCall[T any](t *testing.T, call policyCall) T {
	t.Helper()
	var params T
	if err := json.Unmarshal(call.params, &params); err != nil {
		t.Fatalf("decode %s: %v", call.method, err)
	}
	return params
}

// TestModelConfigEventStateMachine explores ordered event pairs at each owned
// reconciliation phase. Each row uses actual RPC workers and reconciliation;
// the only clock advanced is policyRuntime.now. The small triples exercise
// capacity/watcher and veto/resume schedules with an intervening restart.
func TestModelConfigEventStateMachine(t *testing.T) {
	events := []string{"load ok", "load error", "finish ok", "finish error", "publish matching", "publish mismatching", "capacity", "revision", "pause", "resume", "Off", "On", "drain", "undrain", "proxy restart", "worker restart", "timeout", "shutdown"}
	states := []string{"prepare in flight", "grant in flight", "load in flight", "finish in flight", "pending observation", "terminal finish failure", "nonterminal uncertainty", "retry"}
	for _, state := range states {
		// Independent fixtures partition the first event, keeping long live-RPC
		// phases from serializing the matrix behind a single fixture owner.
		for partition, name := range []string{"first", "middle", "last"} {
			t.Run(state+"/"+name, func(t *testing.T) {
				t.Parallel()
				f := newModelConfigFixture(t)
				defer close(f.done)
				base := f.b.captureModelConfig()
				f.proxy.mu.Lock()
				proxyAnswer := f.proxy.answer
				f.proxy.mu.Unlock()
				r := f.b.nodePolicyRuntime()
				r.mu.Lock()
				identities := r.engineIdentity
				baseModels := r.models
				r.mu.Unlock()
				seed := &modelConfigMatrixSeed{manager: emptySwitchManagerSnapshot()}
				for firstIndex, first := range events {
					if firstIndex%3 != partition {
						continue
					}
					for _, second := range events {
						runModelConfigEventTrace(t, state, []string{first, second}, f, base, baseModels, identities, proxyAnswer, seed)
					}
				}
				for _, trace := range [][]string{
					{"capacity", "publish matching", "capacity"},
					{"pause", "capacity", "resume"},
					{"Off", "proxy restart", "On"},
					{"drain", "capacity", "undrain"},
					{"proxy restart", "publish matching", "timeout"},
					{"finish error", "proxy restart", "publish matching"},
					{"publish matching", "revision", "capacity"},
				} {
					if partition == 0 {
						runModelConfigEventTrace(t, state, trace, f, base, baseModels, identities, proxyAnswer, seed)
					}
				}
			})
		}
	}
}

func runModelConfigEventTrace(t *testing.T, state string, events []string, f *modelConfigFixture, base modelConfigEvidence, baseModels modelConfigRuntime, identities map[string]engineManagerIdentity, proxyAnswer func(string, json.RawMessage) (any, string), seed *modelConfigMatrixSeed) {
	t.Helper()
	// Restore the same quiescent fixture between traces. No held RPC/evaluator
	// crosses this boundary; the deferred join below is part of every row.
	publishTestProxy(f.b, base.Proxy)
	f.source = base.Source
	f.b.bindEngineMgr(f.b.getEngineMgr(), base.Source)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current, r.live, r.localPauseRequested = base.Policy, base.Live, false
	r.incarnation, r.runID, r.activeRevision = base.Proxy, base.Run, base.Policy.Revision
	r.activationErr, r.availabilityErr = "", ""
	r.strictReadiness = base.Readiness
	r.engineManagerSource, r.engineIdentity = base.Source, identities
	r.intent, r.running, r.loaded = map[string]bool{"llamacpp": true}, map[string]bool{"llamacpp": true}, map[string][]string{"llamacpp": {}}
	r.drains, r.admission.ActiveByEngine = nil, nil
	r.models = baseModels
	r.mu.Unlock()
	f.engine.mu.Lock()
	f.engine.calls = nil
	f.engine.held = make(chan struct{})
	f.engine.holdOnce = sync.Once{}
	f.engine.mu.Unlock()
	f.proxy.mu.Lock()
	f.proxy.calls = nil
	f.proxy.held = make(chan struct{})
	f.proxy.holdOnce = sync.Once{}
	f.proxy.mu.Unlock()
	f.acks.mu.Lock()
	f.acks.held = "0"
	f.acks.mu.Unlock()
	ref := nodepolicy.Role{Engine: "llamacpp", Model: "resident"}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	clock := time.Unix(5000, 0)
	r.mu.Lock()
	r.now = func() time.Time { return clock }
	r.mu.Unlock()

	// Seed phases are independent of future events. The held reply points to
	// wireResult, whose final value is published by closing the reply barrier.
	outcome := nodepolicy.ResidentLoaded
	if state == "nonterminal uncertainty" {
		outcome = nodepolicy.ResidentUnknown
	}
	if state == "retry" {
		outcome = nodepolicy.ResidentFailed
	}
	finishError := state == "terminal finish failure" || state == "nonterminal uncertainty"
	wireFinish := &nodepolicy.FinishResidentResult{ReservationID: "fixture-reservation", Outcome: outcome}
	finishEncoded := make(chan struct{})
	finishReply := &modelConfigMatrixReply{value: wireFinish, encoded: finishEncoded}
	phaseEncoded := make(chan struct{})
	// Shared answer flags are read by fake-worker goroutines.
	var answerMu sync.Mutex
	f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if (state == "prepare in flight" && method == nodepolicy.MethodPrepareResident) || (state == "grant in flight" && method == nodepolicy.MethodDispatchResident) {
			result, _ := proxyAnswer(method, raw)
			return &modelConfigMatrixReply{value: result, encoded: phaseEncoded}, holdSuccessReply
		}
		if method == nodepolicy.MethodFinishResident {
			answerMu.Lock()
			fail := finishError
			answerMu.Unlock()
			if fail {
				return nil, "injected finish failure"
			}
			result, failure := proxyAnswer(method, raw)
			if state == "finish in flight" {
				return finishReply, holdSuccessReply
			}
			return result, failure
		}
		return proxyAnswer(method, raw)
	})
	var done <-chan struct{}
	var wireResult *nodepolicy.ReconcileResidentResult
	var loadEncoded <-chan struct{}
	seeded := seed.models != nil
	loadLive, finishLive := false, false
	if seeded {
		r.mu.Lock()
		r.models = cloneModelConfigMatrixState(*seed.models)
		r.mu.Unlock()
		closed := make(chan struct{})
		close(closed)
		done = closed
	} else {
		done, wireResult, loadEncoded = startModelConfigMatrixAttempt(t, ctx, f, seed.manager, outcome, state == "nonterminal uncertainty")
		if state == "prepare in flight" || state == "grant in flight" {
			heldMethod := nodepolicy.MethodPrepareResident
			if state == "grant in flight" {
				heldMethod = nodepolicy.MethodDispatchResident
			}
			awaitModelConfigMatrixCall(t, f.proxy, heldMethod)
		} else {
			awaitModelConfigMatrixCall(t, f.engine, residentDispatchReplyMethod)
			f.awaitGrant(t)
			loadLive = true
		}
		if state != "prepare in flight" && state != "grant in flight" && state != "load in flight" {
			f.engine.releaseHeld()
			loadLive = false
			if state == "finish in flight" && !finishError {
				awaitModelConfigMatrixCall(t, f.proxy, nodepolicy.MethodFinishResident)
				finishLive = true
			} else {
				awaitModelConfigDone(t, done)
			}
		}
		if state != "prepare in flight" && state != "grant in flight" && state != "load in flight" && state != "finish in flight" {
			r.mu.Lock()
			copy := cloneModelConfigMatrixState(r.models)
			seed.models = &copy
			r.mu.Unlock()
		}
	}
	defer func() {
		cancel()
		f.engine.releaseHeld()
		f.proxy.releaseHeld()
		awaitModelConfigDone(t, done)
		if !seeded && state != "nonterminal uncertainty" && state != "prepare in flight" {
			<-loadEncoded
		}
		if !seeded && state == "finish in flight" {
			<-finishEncoded
		}
		if state == "prepare in flight" || state == "grant in flight" {
			<-phaseEncoded
		}
	}()
	// After the seed operation, any accidental second dispatch gets an immediate
	// negative reply. This makes duplicate detection finite, never a timeout.
	readback := residentLaunchReadback(base)
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		switch method {
		case settings.SwitchPrepareMethod:
			return seed.manager, ""
		case "engine:get-launch":
			return readback, ""
		case nodepolicy.MethodReconcileResident:
			return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
		}
		return map[string]bool{"ok": true}, ""
	})
	loads := func() int {
		n := 0
		f.engine.mu.Lock()
		defer f.engine.mu.Unlock()
		for _, call := range f.engine.calls {
			if call.method == nodepolicy.MethodReconcileResident {
				n++
			}
		}
		return n
	}
	shutdown := false
	// The ledger persists after a record disappears. A buggy clearing path
	// cannot evade duplicate detection on the next event by deleting ownership.
	issued := map[string]int{"0": 1}
	if state == "retry" || state == "prepare in flight" || state == "grant in flight" {
		issued["0"] = 0
	}
	for _, event := range events {
		beforeLoads := loads()
		if shutdown {
			f.b.reconcileModelConfig(ctx)
			if loads() != beforeLoads {
				t.Fatalf("trace %v: event after shutdown issued a load", events)
			}
			continue
		}
		r.mu.Lock()
		beforeActive, beforeBusy := r.models.active, r.models.busy
		beforeGranted := beforeActive != nil && beforeActive.granted
		beforeProof, beforePending := r.models.pending[ref]
		beforeHold, beforeUnsettled := r.models.unsettled[ref]
		revision := r.current.Revision
		paused, intent, draining := r.localPauseRequested, r.intent["llamacpp"], r.drains["llamacpp"]
		capacity := r.admission.ActiveByEngine["llamacpp"]
		r.mu.Unlock()
		allowedExit := (beforeActive != nil && beforeActive.evidence.Policy.Revision != revision) ||
			(beforePending && beforeProof.evidence.Policy.Revision != revision) ||
			(beforeUnsettled && beforeHold.op.evidence.Policy.Revision != revision)
		trigger := !shutdown
		switch event {
		case "load ok", "load error":
			trigger = loadLive && !shutdown
			if loadLive {
				if event == "load error" {
					wireResult.Outcome, wireResult.Observation = nodepolicy.ResidentFailed, nil
				}
				f.engine.releaseHeld()
				loadLive = false
				if state == "finish in flight" && !finishError {
					finishLive = true
				} else {
					awaitModelConfigDone(t, done)
				}
				allowedExit = allowedExit || event == "load error" || (beforeActive != nil && !beforeGranted)
			}
		case "finish ok", "finish error":
			trigger = (finishLive || beforeUnsettled) && !shutdown
			answerMu.Lock()
			finishError = event == "finish error"
			answerMu.Unlock()
			if finishLive {
				if event == "finish error" {
					wireFinish.ReservationID = ""
				}
				f.proxy.releaseHeld()
				finishLive = false
				awaitModelConfigDone(t, done)
			}
		case "publish matching":
			f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
			// Name presence cannot discharge a live native operation or uncertain
			// completion; only the reconciler's confirming transfer below can.
		case "publish mismatching":
			f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {}})
		case "capacity":
			trigger = capacity != 1
			r.mu.Lock()
			r.admission.ActiveByEngine = map[string]int{"llamacpp": 1}
			r.mu.Unlock()
		case "revision":
			r.mu.Lock()
			if r.current.Revision == "0" {
				r.current.Revision = "1"
			} else {
				r.current.Revision = "2"
			}
			revision := r.current.Revision
			r.mu.Unlock()
			if !f.b.replayPolicyToProxy(ctx, f.b.policyProxy()) && !shutdown {
				t.Fatal("revision activation failed")
			}
			reportStrict(t, f.b, revision, f.b.captureModelConfig().Run, strictCapability(t, "llamacpp", "resident"))
			allowedExit = true
		case "pause", "resume":
			trigger = paused != (event == "pause")
			r.mu.Lock()
			r.localPauseRequested = event == "pause"
			r.mu.Unlock()
		case "Off", "On":
			trigger = intent != (event == "On")
			r.mu.Lock()
			r.intent["llamacpp"] = event == "On"
			r.mu.Unlock()
		case "drain", "undrain":
			trigger = draining != (event == "drain")
			r.mu.Lock()
			r.drains = map[string]bool{"llamacpp": event == "drain"}
			r.mu.Unlock()
		case "proxy restart":
			if !shutdown {
				pp, proxy, acks := attachAckingProxy(t, f.b)
				defer proxy.conn.Close()
				proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
					if method == nodepolicy.MethodSetPolicy {
						return acks.answer(method, raw)
					}
					return proxyAnswer(method, raw)
				})
				if !f.b.replayPolicyToProxy(ctx, pp) {
					t.Fatal("replacement activation failed")
				}
				reportStrict(t, f.b, f.b.captureModelConfig().Policy.Revision, acks.runID, strictCapability(t, "llamacpp", "resident"))
			}
		case "worker restart":
			// The original native worker is retained as the completion source;
			// changing its source handle invalidates current inventory provenance.
			f.source = f.b.activateEngineManagerSource()
			f.b.bindEngineMgr(f.b.getEngineMgr(), f.source)
			r.mu.Lock()
			r.engineIdentity = identities
			r.mu.Unlock()
		case "timeout":
			r.mu.Lock()
			clock = clock.Add(residentObservationTimeout + time.Second)
			r.mu.Unlock()
			allowedExit = allowedExit || !beforeUnsettled || state != "nonterminal uncertainty"
		case "shutdown":
			// An unbuffered wake rendezvous proves the owner reached its select
			// after creating the ticker. Busy admission coalesces this wake so
			// starting the owner cannot introduce a new load before cancellation.
			wake := make(chan struct{})
			r.mu.Lock()
			oldBusy, oldWake := r.models.busy, r.models.wake
			r.models.busy, r.models.wake = true, wake
			r.mu.Unlock()
			idleDone := make(chan struct{})
			go func() { defer close(idleDone); f.b.runIdlePolicy(ctx) }()
			wake <- struct{}{}
			cancel()
			shutdown = true
			awaitModelConfigDone(t, idleDone)
			r.mu.Lock()
			r.models.wake = oldWake
			if r.models.active != nil {
				r.models.busy = oldBusy
			} else {
				r.models.busy = false
			}
			r.mu.Unlock()
			f.engine.releaseHeld()
			f.proxy.releaseHeld()
			loadLive, finishLive = false, false
			awaitModelConfigDone(t, done)
		}
		if trigger {
			f.b.reconcileModelConfig(ctx)
		}
		r.mu.Lock()
		active, busy := r.models.active, r.models.busy
		pending, hasPending := r.models.pending[ref]
		hold, hasHold := r.models.unsettled[ref]
		veto := r.localPauseRequested || !r.intent["llamacpp"] || r.drains["llamacpp"]
		_, confirmed := r.models.confirmed[ref]
		currentRevision := r.current.Revision
		r.mu.Unlock()
		if event == "publish matching" && confirmed && !hasPending && !hasHold {
			allowedExit = true
		}
		if allowedExit {
			issued[revision] = 0
		}
		if delta := loads() - beforeLoads; delta > 0 {
			issued[currentRevision] += delta
			if issued[currentRevision] > 1 {
				t.Fatalf("trace %v: duplicate dispatch for revision %s", events, currentRevision)
			}
			// Every post-seed fake attempt immediately replies NotDispatched,
			// which is itself the allowed load-failure exit for that new attempt.
			issued[currentRevision] = 0
		}
		if (veto || shutdown) && loads() != beforeLoads {
			t.Fatalf("%s issued load under veto/shutdown", event)
		}
		if !allowedExit && (beforePending || beforeUnsettled || beforeActive != nil) && loads() != beforeLoads {
			t.Fatalf("%s issued another load before an allowed fact exit", event)
		}
		if !allowedExit && beforePending && !hasPending {
			t.Fatalf("%s discarded pending observation without an allowed exit", event)
		}
		if event == "publish matching" && beforeUnsettled && beforeHold.terminal && beforeHold.finish.Outcome == nodepolicy.ResidentLoaded && beforeHold.op.evidence.Proxy != f.b.policyProxy() && !veto && !shutdown && hasHold {
			t.Fatalf("trace %v: fresh replacement publication left terminal load stuck", events)
		}
		operationPending := false
		select {
		case <-done:
		default:
			operationPending = true
		}
		if beforeBusy && operationPending && (active != beforeActive || !busy) {
			t.Fatalf("%s discarded live ownership", event)
		}
		if !shutdown && hasPending && !pending.expires.After(clock) {
			t.Fatalf("%s left pending without live observation deadline", event)
		}
		if !shutdown && hasHold && hold.terminal && !hold.expires.After(clock) {
			t.Fatalf("%s left terminal hold without live observation deadline", event)
		}
		if hasHold && !hold.terminal && (hold.op == nil || hold.finish.ReservationID == "") {
			t.Fatalf("%s lost original completion/revision exit", event)
		}
		if busy && active == nil {
			t.Fatalf("%s left busy without an operation", event)
		}
		if shutdown && (busy || active != nil) {
			t.Fatalf("shutdown retained operation goroutine ownership")
		}
	}
	// Joining every held evaluation is the scoped goroutine leak assertion. No
	// per-key timer is created: expiration is driven by the injected idle clock.
	cancel()
	f.engine.releaseHeld()
	f.proxy.releaseHeld()
	awaitModelConfigDone(t, done)
	r.mu.Lock()
	busy, active := r.models.busy, r.models.active
	r.mu.Unlock()
	if busy || active != nil {
		t.Fatal("trace cleanup retained evaluator")
	}
	r.mu.Lock()
	remainingHold, held := r.models.unsettled[ref]
	r.mu.Unlock()
	if !held || remainingHold.terminal {
		return
	}
	// Pending and terminal waits have a live injected-clock deadline, checked
	// above and exercised by every timeout event. A genuinely uncertain native
	// operation instead needs explicit declaration retirement; execute that exit
	// rather than treating a dead operation pointer as a pending completion.
	r.mu.Lock()
	r.localPauseRequested = true
	r.current.Revision = "9"
	r.models.last = nil
	r.mu.Unlock()
	f.b.reconcileModelConfig(context.Background())
	r.mu.Lock()
	_, pendingAfterRetire := r.models.pending[ref]
	_, heldAfterRetire := r.models.unsettled[ref]
	r.mu.Unlock()
	if pendingAfterRetire || heldAfterRetire {
		t.Fatalf("trace %v: declaration exit unreachable", events)
	}
}

type modelConfigMatrixSeed struct {
	models  *modelConfigRuntime
	manager switchManagerSnapshot
}

func awaitModelConfigMatrixCall(t *testing.T, worker *fakeWorker, method string) {
	t.Helper()
	deadline := time.NewTimer(3 * time.Second)
	defer deadline.Stop()
	for {
		for _, call := range worker.snapshot() {
			if call.method == method {
				return
			}
		}
		select {
		case <-worker.signal:
		case <-deadline.C:
			t.Fatalf("missing matrix RPC %s", method)
		}
	}
}

func cloneModelConfigMatrixState(from modelConfigRuntime) modelConfigRuntime {
	copy := from
	copy.pending = make(map[nodepolicy.Role]residentPendingObservation)
	for ref, pending := range from.pending {
		copy.pending[ref] = pending
	}
	copy.unsettled = make(map[nodepolicy.Role]residentUnsettled)
	for ref, hold := range from.unsettled {
		op := *hold.op
		if op.recovery != nil {
			recovery := *op.recovery
			op.recovery = &recovery
		}
		hold.op = &op
		copy.unsettled[ref] = hold
	}
	copy.confirmed = make(map[nodepolicy.Role]modelConfigEvidence)
	for ref, evidence := range from.confirmed {
		copy.confirmed[ref] = evidence
	}
	copy.status.Models = append([]nodepolicy.ModelConfigurationEntry(nil), from.status.Models...)
	return copy
}

func startModelConfigMatrixAttempt(t *testing.T, ctx context.Context, f *modelConfigFixture, manager switchManagerSnapshot, outcome nodepolicy.ResidentOutcome, nonterminal bool) (<-chan struct{}, *nodepolicy.ReconcileResidentResult, <-chan struct{}) {
	t.Helper()
	result := &nodepolicy.ReconcileResidentResult{Outcome: outcome}
	encoded := make(chan struct{})
	reply := &modelConfigMatrixReply{value: result, encoded: encoded}
	f.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			return manager, ""
		}
		if method != nodepolicy.MethodReconcileResident {
			return map[string]bool{"ok": true}, ""
		}
		var p nodepolicy.ReconcileResidentParams
		if err := json.Unmarshal(raw, &p); err != nil {
			return nil, err.Error()
		}
		request, err := json.Marshal(struct {
			ID     string                            `json:"id"`
			Params nodepolicy.DispatchResidentParams `json:"params"`
		}{"fixture-dispatch", nodepolicy.DispatchResidentParams{
			ReservationID: p.Reservation.ReservationID, ProxyRunID: p.Reservation.ProxyRunID,
			ManagerIncarnation: p.Reservation.ManagerIncarnation, PolicyRevision: p.PolicyRevision,
			OperationID: p.OperationID, Engine: p.Engine, Model: p.Model, Identity: p.Identity,
		}})
		if err != nil {
			return nil, err.Error()
		}
		if outcome == nodepolicy.ResidentLoaded {
			result.Observation = &nodepolicy.ResidentLoadedObservation{
				ManagerIncarnation: p.Reservation.ManagerIncarnation, Generation: 1,
				Engine: p.Engine, Model: p.Model, Identity: p.Identity, Loaded: true,
			}
		}
		f.b.dispatchResidentFrom(f.source, request)
		if nonterminal {
			return nil, holdReply
		}
		return reply, holdSuccessReply
	})
	done := make(chan struct{})
	go func() { defer close(done); f.b.reconcileModelConfig(ctx) }()
	return done, result, encoded
}

// Encoding joins the fake held-response goroutine even when cancellation wins
// the RPC response race. That barrier makes resetting fakeWorker.held safe.
type modelConfigMatrixReply struct {
	value   any
	encoded chan struct{}
	once    sync.Once
}

func (r *modelConfigMatrixReply) MarshalJSON() ([]byte, error) {
	defer r.once.Do(func() { close(r.encoded) })
	return json.Marshal(r.value)
}

func TestModelConfigLoadUsesRevisionProfile(t *testing.T) {
	test := func(t *testing.T, contextTokens int) {
		f := newModelConfigFixture(t)
		// A valid nonallocation multiple detects rounding, clamping and defaults.
		r := f.b.nodePolicyRuntime()
		r.mu.Lock()
		r.current.Profiles[0].ContextTokens = contextTokens
		id := r.engineIdentity["llamacpp"]
		envelope := id.Serving["resident"]
		envelope.PerSlotContextTokens, envelope.TotalContextTokens = int64(contextTokens), int64(contextTokens)
		id.Serving["resident"] = envelope
		r.engineIdentity["llamacpp"] = id
		r.mu.Unlock()
		f.b.reconcileModelConfig(context.Background())
		calls := modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)
		if len(calls) != 1 {
			t.Fatalf("manager reconcile calls=%d, want one", len(calls))
		}
		p := decodeModelConfigCall[nodepolicy.ReconcileResidentParams](t, calls[0])
		if p.Load.ContextLength != contextTokens || p.Load.KeepAliveSeconds != 60 || p.DeclaredResidentCount != 1 || p.PolicyRevision != "0" || p.Reservation.ManagerIncarnation != "fixture-manager" || p.Identity.EngineRunID != "engine-run-1" {
			t.Fatalf("reconcile lost exact revision/profile/count contract: %+v", p)
		}
		prepared := modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)
		finished := modelConfigCalls(f.proxy, nodepolicy.MethodFinishResident)
		if len(prepared) != 1 || len(finished) != 1 {
			t.Fatalf("prepare=%d finish=%d, want one each", len(prepared), len(finished))
		}
		prepare := decodeModelConfigCall[nodepolicy.PrepareResidentParams](t, prepared[0])
		finish := decodeModelConfigCall[nodepolicy.FinishResidentParams](t, finished[0])
		if prepare.PolicyRevision != p.PolicyRevision || prepare.Identity != p.Identity || prepare.OperationID != p.OperationID || finish.OperationID != p.OperationID || finish.ReservationID != p.Reservation.ReservationID {
			t.Fatal("prepare, reconcile and finish do not share the captured operation")
		}
	}
	t.Run("small nonaligned context", func(t *testing.T) { test(t, 4097) })
	t.Run("review nonaligned context", func(t *testing.T) { test(t, 8193) })
	t.Run("large nonaligned context", func(t *testing.T) { test(t, 32769) })
}

func TestModelConfigConfirmingInventoryDoesNotReload(t *testing.T) {
	f := newModelConfigFixture(t)
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			return emptySwitchManagerSnapshot(), ""
		}
		return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
	})
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	for range 3 {
		f.b.reconcileModelConfig(context.Background())
	}
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 1 {
		t.Fatalf("confirming inventory issued %d loads, want one completed load", got)
	}
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].State != nodepolicy.ModelStateSatisfied {
		t.Fatalf("confirming inventory lost satisfaction: %+v", status)
	}
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {}})
	f.b.reconcileModelConfig(context.Background())
	f.b.reconcileModelConfig(context.Background())
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 2 {
		t.Fatalf("actual unload issued %d loads, want original and one retry", got)
	}
}

func TestModelConfigInflightConfirmingInventoryCoalesces(t *testing.T) {
	f := newModelConfigFixture(t)
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	var wg sync.WaitGroup
	for range 12 {
		wg.Add(1)
		go func() { defer wg.Done(); f.b.reconcileModelConfig(context.Background()) }()
	}
	wg.Wait()
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			return emptySwitchManagerSnapshot(), ""
		}
		return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
	})
	f.b.reconcileModelConfig(context.Background())
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 1 {
		t.Fatalf("inflight confirmation issued %d loads, want one", got)
	}
}

func TestModelConfigBusyRetainsOriginalOwnership(t *testing.T) {
	f := newModelConfigFixture(t)
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	original := r.models.active
	r.mu.Unlock()
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.b.reconcileModelConfig(context.Background())
	r.mu.Lock()
	active, busy := r.models.active, r.models.busy
	r.mu.Unlock()
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	if original == nil || active != original || !busy || len(modelConfigCalls(f.engine, "engine:get-launch")) != 0 {
		t.Fatalf("changed evidence lost original owner: original=%p active=%p busy=%v launch-reads=%d", original, active, busy, len(modelConfigCalls(f.engine, "engine:get-launch")))
	}
	if status := f.b.modelConfiguration(f.b.captureModelConfig()); status.Models[0].State != nodepolicy.ModelStateSatisfied {
		t.Fatalf("original operation did not finish satisfied: %+v", status)
	}
}

func TestModelConfigDrainRetainsInflightOwnership(t *testing.T) {
	f := newModelConfigFixture(t)
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	original := r.models.active
	r.drains = map[string]bool{"llamacpp": true}
	r.mu.Unlock()
	f.b.reconcileModelConfig(context.Background())
	r.mu.Lock()
	retained := original != nil && r.models.active == original && r.models.busy
	r.drains["llamacpp"] = false
	r.mu.Unlock()
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	if !retained || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
		t.Fatal("drain lost original in-flight ownership or dispatched overlapping work")
	}
	f.b.reconcileModelConfig(context.Background())
	if status := f.b.modelConfiguration(f.b.captureModelConfig()); status.Models[0].State != nodepolicy.ModelStateSatisfied {
		t.Fatalf("original operation did not complete: %+v", status)
	}
}

func TestModelConfigNonterminalReplacementTimeoutPreservesBarrier(t *testing.T) {
	for _, replacement := range []string{"manager", "proxy"} {
		t.Run(replacement, func(t *testing.T) {
			f := newModelConfigFixture(t)
			f.proxy.mu.Lock()
			answer := f.proxy.answer
			f.proxy.mu.Unlock()
			f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == nodepolicy.MethodFinishResident {
					return nil, "finish unavailable"
				}
				return answer(method, raw)
			})
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done, _, _ := startModelConfigMatrixAttempt(t, ctx, f, emptySwitchManagerSnapshot(), nodepolicy.ResidentUnknown, true)
			f.awaitGrant(t)
			cancel()
			awaitModelConfigDone(t, done)
			f.engine.releaseHeld()
			r := f.b.nodePolicyRuntime()
			ref := nodepolicy.Role{Engine: "llamacpp", Model: "resident"}
			r.mu.Lock()
			original, present := r.models.unsettled[ref]
			identity := r.engineIdentity
			r.mu.Unlock()
			if !present || original.terminal || original.op == nil {
				t.Fatal("missing original nonterminal granted uncertainty")
			}
			assertRetained := func(stage string) {
				t.Helper()
				r.mu.Lock()
				retained, held := r.models.unsettled[ref]
				r.mu.Unlock()
				if !held || retained.op != original.op || !reflect.DeepEqual(retained, original) {
					t.Fatalf("%s changed original nonterminal record: held=%v original=%p retained=%p", stage, held, original.op, retained.op)
				}
				if loads := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); loads != 1 {
					t.Fatalf("%s issued %d loads, want exactly one original load", stage, loads)
				}
			}
			assertRetained("before replacement")
			if replacement == "manager" {
				f.source = f.b.activateEngineManagerSource()
				f.b.bindEngineMgr(f.b.getEngineMgr(), f.source)
				r.mu.Lock()
				r.engineIdentity = identity
				r.mu.Unlock()
			} else {
				pp, _, acks := attachAckingProxy(t, f.b)
				if !f.b.replayPolicyToProxy(context.Background(), pp) {
					t.Fatal("activate replacement proxy")
				}
				reportStrict(t, f.b, "0", acks.runID, strictCapability(t, "llamacpp", "resident"))
			}
			assertRetained("after replacement")
			f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
				if method == settings.SwitchPrepareMethod {
					return emptySwitchManagerSnapshot(), ""
				}
				return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
			})
			r.mu.Lock()
			now := original.expires.Add(time.Hour)
			r.now = func() time.Time { return now }
			r.mu.Unlock()
			f.b.reconcileModelConfig(context.Background())
			assertRetained("past observation deadline")
			f.b.observeLoadedFrom(f.source, nil)
			f.b.reconcileModelConfig(context.Background())
			assertRetained("unknown publication")
			f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {}})
			f.b.reconcileModelConfig(context.Background())
			assertRetained("known-empty publication")
			r.mu.Lock()
			r.localPauseRequested = true
			r.mu.Unlock()
			f.b.reconcileModelConfig(context.Background())
			assertRetained("paused declaration")
			// Declaration removal is an allowed exit even while paused.
			r.mu.Lock()
			r.current.ResidentModels = nil
			r.mu.Unlock()
			f.b.reconcileModelConfig(context.Background())
			r.mu.Lock()
			_, held := r.models.unsettled[ref]
			r.mu.Unlock()
			if held || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
				t.Fatal("paused declaration removal retained uncertainty or dispatched another load")
			}
		})
	}
}

func TestModelConfigTerminalLoadedObservationBound(t *testing.T) {
	test := func(t *testing.T, paused bool) {
		t.Helper()
		f := newModelConfigFixture(t)
		r := f.b.nodePolicyRuntime()
		r.mu.Lock()
		initial := r.now()
		now := initial
		r.now = func() time.Time { return now }
		r.mu.Unlock()
		f.proxy.mu.Lock()
		answer := f.proxy.answer
		f.proxy.mu.Unlock()
		f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
			if method == nodepolicy.MethodFinishResident {
				return nil, "finish unavailable"
			}
			return answer(method, raw)
		})
		done := f.startLoadedAttempt(t)
		f.awaitGrant(t)
		f.engine.releaseHeld()
		awaitModelConfigDone(t, done)
		ref := nodepolicy.Role{Engine: "llamacpp", Model: "resident"}
		r.mu.Lock()
		hold, held := r.models.unsettled[ref]
		r.mu.Unlock()
		if !held || !hold.terminal || hold.finish.Outcome != nodepolicy.ResidentLoaded {
			t.Fatal("missing terminal loaded / failed-finish record")
		}
		if residentObservationTimeout != time.Minute || hold.expires.Sub(initial) != residentObservationTimeout {
			t.Fatalf("terminal observation duration=%v, domain constant=%v, want 60s", hold.expires.Sub(initial), residentObservationTimeout)
		}
		f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
			if method == settings.SwitchPrepareMethod {
				return emptySwitchManagerSnapshot(), ""
			}
			return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
		})
		f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {}})
		r.mu.Lock()
		now = initial.Add(residentObservationTimeout - time.Nanosecond)
		r.localPauseRequested = paused
		r.mu.Unlock()
		f.b.reconcileModelConfig(context.Background())
		r.mu.Lock()
		before, retained := r.models.unsettled[ref]
		r.mu.Unlock()
		if !retained || before.op != hold.op || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
			t.Fatal("terminal record expired before the fixed observation bound")
		}
		r.mu.Lock()
		// The just-before evaluation installs the unchanged-evidence retry tick.
		// Advance independently of hold.expires to its next eligible evaluation.
		now = initial.Add(residentObservationTimeout + idleCheckInterval)
		r.mu.Unlock()
		f.b.reconcileModelConfig(context.Background())
		r.mu.Lock()
		_, retained = r.models.unsettled[ref]
		r.mu.Unlock()
		if retained {
			t.Fatal("terminal record survived the next eligible evaluation after the fixed bound")
		}
		wantLoads := 2
		if paused {
			wantLoads = 1
		}
		if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != wantLoads {
			t.Fatalf("post-bound loads=%d, want %d", got, wantLoads)
		}
		if paused {
			f.b.reconcileModelConfig(context.Background())
			if len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
				t.Fatal("paused evaluation dispatched after terminal retirement")
			}
			r.mu.Lock()
			r.localPauseRequested = false
			r.mu.Unlock()
			f.b.reconcileModelConfig(context.Background())
			if len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 2 {
				t.Fatal("resume lost known-empty retry after terminal retirement")
			}
		}
	}
	t.Run("eligible known-empty retries", func(t *testing.T) { test(t, false) })
	t.Run("paused retries only after resume", func(t *testing.T) { test(t, true) })
}

func TestModelConfigTerminalFinishFailureHasBoundedExit(t *testing.T) {
	for _, exit := range []string{"fresh confirmation", "observation timeout"} {
		t.Run(exit, func(t *testing.T) {
			f := newModelConfigFixture(t)
			f.proxy.mu.Lock()
			answer := f.proxy.answer
			f.proxy.mu.Unlock()
			f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == nodepolicy.MethodFinishResident {
					return nil, "retired proxy"
				}
				return answer(method, raw)
			})
			done := f.startLoadedAttempt(t)
			f.awaitGrant(t)
			f.engine.releaseHeld()
			awaitModelConfigDone(t, done)
			before := f.b.captureModelConfig()
			r := f.b.nodePolicyRuntime()
			ref := nodepolicy.Role{Engine: "llamacpp", Model: "resident"}
			r.mu.Lock()
			hold, ok := r.models.unsettled[ref]
			r.mu.Unlock()
			if !ok || !hold.terminal || hold.expires.IsZero() {
				t.Fatal("terminal finish failure has no bounded observation exit")
			}
			pp, _, acks := attachAckingProxy(t, f.b)
			if !f.b.replayPolicyToProxy(context.Background(), pp) {
				t.Fatal("activate replacement proxy")
			}
			reportStrict(t, f.b, "0", acks.runID, strictCapability(t, "llamacpp", "resident"))
			readback := residentLaunchReadback(before)
			f.engine.setAnswer(func(string, json.RawMessage) (any, string) { return readback, "" })
			if exit == "fresh confirmation" {
				f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
			} else {
				r.mu.Lock()
				now := hold.expires.Add(time.Second)
				r.now = func() time.Time { return now }
				r.mu.Unlock()
			}
			f.b.reconcileModelConfig(context.Background())
			r.mu.Lock()
			_, retained := r.models.unsettled[ref]
			r.mu.Unlock()
			if retained || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
				t.Fatal("terminal observation remained stuck or duplicated load")
			}
			if exit == "fresh confirmation" {
				status := f.b.modelConfiguration(f.b.captureModelConfig())
				if status.Models[0].State != nodepolicy.ModelStateSatisfied || len(modelConfigCalls(f.engine, "engine:get-launch")) != 1 {
					t.Fatalf("replacement bypassed fresh envelope confirmation: %+v", status)
				}
			}
		})
	}
}

func TestModelConfigCompletedLoadWaitsForWatcher(t *testing.T) {
	f := newModelConfigFixture(t)
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			return emptySwitchManagerSnapshot(), ""
		}
		return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
	})
	r := f.b.nodePolicyRuntime()
	for capacity := range 3 {
		r.mu.Lock()
		r.admission.ActiveByEngine = map[string]int{"llamacpp": capacity}
		r.mu.Unlock()
		f.b.reconcileModelConfig(context.Background())
	}
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 1 {
		t.Fatalf("capacity change before watcher issued %d loads, want one", got)
	}
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {}})
	f.b.reconcileModelConfig(context.Background())
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 1 {
		t.Fatalf("empty publication during pending observation issued %d loads", got)
	}
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.b.reconcileModelConfig(context.Background())
	if status := f.b.modelConfiguration(f.b.captureModelConfig()); status.Models[0].State != nodepolicy.ModelStateSatisfied || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
		t.Fatalf("fresh observation did not confirm one load: %+v", status)
	}
}

func TestModelConfigTemporaryVetoPreservesCompletedLoad(t *testing.T) {
	for _, scenario := range []string{"pause", "Off", "drain", "pause in flight"} {
		t.Run(scenario, func(t *testing.T) {
			f := newModelConfigFixture(t)
			r := f.b.nodePolicyRuntime()
			setVeto := func(enabled bool) {
				r.mu.Lock()
				defer r.mu.Unlock()
				switch scenario {
				case "Off":
					r.intent["llamacpp"] = !enabled
				case "drain":
					r.drains = map[string]bool{"llamacpp": enabled}
				default:
					r.localPauseRequested = enabled
				}
			}
			done := f.startLoadedAttempt(t)
			f.awaitGrant(t)
			if scenario == "pause in flight" {
				setVeto(true)
				f.b.reconcileModelConfig(context.Background())
			}
			f.engine.releaseHeld()
			awaitModelConfigDone(t, done)
			f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
				if method == settings.SwitchPrepareMethod {
					return emptySwitchManagerSnapshot(), ""
				}
				return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
			})
			setVeto(true)
			f.b.reconcileModelConfig(context.Background())
			if status := f.b.modelConfiguration(f.b.captureModelConfig()); status.Models[0].State != nodepolicy.ModelStateBlocked {
				t.Fatalf("local veto lost precedence: %+v", status)
			}
			setVeto(false)
			f.b.reconcileModelConfig(context.Background())
			if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 1 {
				t.Fatalf("resume before watcher issued %d loads, want exactly one", got)
			}
			if status := f.b.modelConfiguration(f.b.captureModelConfig()); status.Models[0].State != nodepolicy.ModelStateSatisfied {
				t.Fatalf("resume lost completed load proof: %+v", status)
			}
		})
	}
}

func TestModelConfigPendingObservationAllowedExits(t *testing.T) {
	for _, tc := range []struct {
		name          string
		outcome       nodepolicy.ResidentOutcome
		wantLoads     int
		wantConfirmed bool
	}{
		{"confirming observation", nodepolicy.ResidentLoaded, 1, true},
		{"load failure", nodepolicy.ResidentFailed, 2, false},
		{"revision replacement", nodepolicy.ResidentLoaded, 2, false},
		{"resident removal", nodepolicy.ResidentLoaded, 1, false},
		{"observation timeout", nodepolicy.ResidentLoaded, 2, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newModelConfigFixture(t)
			done := f.startResidentAttempt(t, tc.outcome)
			f.awaitGrant(t)
			f.engine.releaseHeld()
			awaitModelConfigDone(t, done)
			r := f.b.nodePolicyRuntime()
			ref := nodepolicy.Role{Engine: "llamacpp", Model: "resident"}
			r.mu.Lock()
			_, pending := r.models.pending[ref]
			r.mu.Unlock()
			if pending != (tc.outcome == nodepolicy.ResidentLoaded) {
				t.Fatalf("completion pending=%v outcome=%s", pending, tc.outcome)
			}
			f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
				if method == settings.SwitchPrepareMethod {
					return emptySwitchManagerSnapshot(), ""
				}
				return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
			})
			if tc.name == "confirming observation" {
				f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
			} else {
				r.mu.Lock()
				switch tc.name {
				case "revision replacement":
					r.current.Revision = "1"
				case "resident removal":
					r.current.ResidentModels = nil
				default:
					now := r.now().Add(residentObservationTimeout + time.Second)
					r.now = func() time.Time { return now }
				}
				r.mu.Unlock()
			}
			if tc.name == "revision replacement" {
				if !f.b.replayPolicyToProxy(context.Background(), f.pp) {
					t.Fatal("activate replacement revision")
				}
				reportStrict(t, f.b, "1", f.acks.runID, strictCapability(t, "llamacpp", "resident"))
			}
			f.b.reconcileModelConfig(context.Background())
			r.mu.Lock()
			_, pending = r.models.pending[ref]
			_, confirmed := r.models.confirmed[ref]
			r.mu.Unlock()
			if pending || confirmed != tc.wantConfirmed {
				t.Fatalf("exit pending=%v confirmed=%v, want false/%v", pending, confirmed, tc.wantConfirmed)
			}
			if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != tc.wantLoads {
				t.Fatalf("exit loads=%d, want %d", got, tc.wantLoads)
			}
		})
	}
}

func TestModelConfigObservationTimeoutReevaluates(t *testing.T) {
	f := newModelConfigFixture(t)
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			return emptySwitchManagerSnapshot(), ""
		}
		return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
	})
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	now := r.now().Add(2*idleCheckInterval + time.Second)
	r.now = func() time.Time { return now }
	r.mu.Unlock()
	f.b.reconcileModelConfig(context.Background())
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 2 {
		t.Fatalf("observation timeout issued %d loads, want original and re-evaluation", got)
	}
}

func TestModelConfigTransientRecoveryRetries(t *testing.T) {
	for _, scenario := range []string{"inventory failure", "observe failure", "finish transport failure", "revision replacement", "cancelled recovery"} {
		t.Run(scenario, func(t *testing.T) {
			f := newModelConfigFixture(t)
			var mu sync.Mutex
			healthy := false
			f.proxy.mu.Lock()
			answer := f.proxy.answer
			f.proxy.mu.Unlock()
			f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				mu.Lock()
				ok := healthy
				mu.Unlock()
				if method == nodepolicy.MethodFinishResident {
					if !ok && scenario == "finish transport failure" {
						return nil, "transient finish failure"
					}
					return nodepolicy.FinishResidentResult{ReservationID: "fixture-reservation", Outcome: nodepolicy.ResidentUnknown, RecoveryID: "fixture-recovery"}, ""
				}
				if method == nodepolicy.MethodObserveResident {
					if !ok && scenario == "observe failure" {
						return nil, "transient observe failure"
					}
					return nodepolicy.ObserveResidentResult{ReservationID: "fixture-reservation", Settled: true}, ""
				}
				return answer(method, raw)
			})
			done := f.startResidentAttempt(t, nodepolicy.ResidentUnknown)
			f.awaitGrant(t)
			f.engine.mu.Lock()
			engineAnswer := f.engine.answer
			f.engine.mu.Unlock()
			readback := residentLaunchReadback(f.b.captureModelConfig())
			f.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				mu.Lock()
				ok := healthy
				mu.Unlock()
				if method == "engine:models" {
					if !ok && scenario != "observe failure" && scenario != "finish transport failure" {
						return nil, "transient inventory failure"
					}
					return map[string]any{"loadedByEngine": map[string][]string{"llamacpp": {"resident"}}}, ""
				}
				if method == "engine:get-launch" {
					return readback, ""
				}
				if method == nodepolicy.MethodReconcileResident {
					_, mode := engineAnswer(method, raw)
					return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentUnknown}, mode
				}
				return engineAnswer(method, raw)
			})
			f.engine.releaseHeld()
			awaitModelConfigDone(t, done)
			mu.Lock()
			healthy = true
			mu.Unlock()
			r := f.b.nodePolicyRuntime()
			if scenario == "revision replacement" {
				r.mu.Lock()
				r.current.Revision = "1"
				r.mu.Unlock()
				if !f.b.replayPolicyToProxy(context.Background(), f.pp) {
					t.Fatal("activate replacement revision")
				}
				reportStrict(t, f.b, "1", f.acks.runID, strictCapability(t, "llamacpp", "resident"))
				f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
				// Replacement must evaluate immediately, without a retry tick.
				f.b.reconcileModelConfig(context.Background())
			}
			if scenario == "cancelled recovery" {
				r.mu.Lock()
				now := r.now().Add(idleCheckInterval + time.Second)
				r.now = func() time.Time { return now }
				r.mu.Unlock()
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				f.b.reconcileModelConfig(ctx)
			}
			// No changed evidence is needed: advance the broker tick clock.
			r.mu.Lock()
			now := r.now().Add(idleCheckInterval + time.Second)
			r.now = func() time.Time { return now }
			r.mu.Unlock()
			f.b.reconcileModelConfig(context.Background())
			f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
			f.b.reconcileModelConfig(context.Background())
			status := f.b.modelConfiguration(f.b.captureModelConfig())
			wantLoads := 1
			if scenario == "cancelled recovery" {
				// Both tick advances cross the terminal observation deadline.
				// Its bounded exit permits one new attempt after cancellation.
				wantLoads = 2
			}
			if status.Models[0].State != nodepolicy.ModelStateSatisfied || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != wantLoads {
				t.Fatalf("transient failure suppressed settlement or duplicated load: %+v loads=%d", status, len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)))
			}
			if scenario != "revision replacement" && len(modelConfigCalls(f.proxy, nodepolicy.MethodObserveResident)) == 0 {
				t.Fatal("original operation was not recovered before confirmation")
			}
			if scenario == "observe failure" {
				calls := modelConfigCalls(f.proxy, nodepolicy.MethodObserveResident)
				if len(calls) != 2 || !reflect.DeepEqual(decodeModelConfigCall[nodepolicy.ObserveResidentParams](t, calls[0]), decodeModelConfigCall[nodepolicy.ObserveResidentParams](t, calls[1])) || len(modelConfigCalls(f.engine, "engine:models")) != 1 {
					t.Fatal("observe retry did not preserve the exact original recovery receipt")
				}
			}
		})
	}
}

func TestModelConfigRecoveryCannotUseReplacementManager(t *testing.T) {
	f := newModelConfigFixture(t)
	f.proxy.mu.Lock()
	answer := f.proxy.answer
	f.proxy.mu.Unlock()
	f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodFinishResident {
			return nodepolicy.FinishResidentResult{ReservationID: "fixture-reservation", Outcome: nodepolicy.ResidentUnknown, RecoveryID: "fixture-recovery"}, ""
		}
		return answer(method, raw)
	})
	done := f.startResidentAttempt(t, nodepolicy.ResidentUnknown)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	before := f.b.captureModelConfig()
	manager := attachFakeEngine(t, f.b)
	f.source = f.b.activateEngineManagerSource()
	f.b.bindEngineMgr(f.b.getEngineMgr(), f.source)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	id := before.Identity.Engines["llamacpp"]
	r.engineIdentity = map[string]engineManagerIdentity{"llamacpp": {EngineRunID: id.EngineRunID, LaunchDigest: id.LaunchDigest, ModelDigests: id.ModelDigests, Serving: id.Serving}}
	r.intent["llamacpp"], r.running["llamacpp"] = true, true
	r.mu.Unlock()
	manager.setAnswer(func(string, json.RawMessage) (any, string) { return residentLaunchReadback(before), "" })
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.b.reconcileModelConfig(context.Background())
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].State != nodepolicy.ModelStateUnknown || len(modelConfigCalls(manager, "engine:get-launch")) != 0 || len(modelConfigCalls(manager, nodepolicy.MethodReconcileResident)) != 0 {
		t.Fatalf("replacement manager settled or retried the original hold: %+v", status)
	}
}

func residentLaunchReadback(e modelConfigEvidence) settings.LaunchState {
	envelope := e.Identity.Engines["llamacpp"].Serving["resident"]
	return settings.LaunchState{Engine: "llamacpp", Running: true, ServingReadback: &engines.ServingReadback{
		Status: "observed", LaunchValidated: []engines.ServingEnvelope{envelope},
		Observed: []engines.ServingObservation{{ModelID: "resident", PerSlotContextTokens: envelope.PerSlotContextTokens, Slots: envelope.Slots}},
	}}
}

func TestModelConfigWorkerRestartConfirmsWithoutReload(t *testing.T) {
	for _, worker := range []string{"manager", "proxy"} {
		t.Run(worker, func(t *testing.T) {
			f := newModelConfigFixture(t)
			done := f.startLoadedAttempt(t)
			f.awaitGrant(t)
			f.engine.releaseHeld()
			awaitModelConfigDone(t, done)
			before := f.b.captureModelConfig()
			manager := f.engine
			if worker == "manager" {
				manager = attachFakeEngine(t, f.b)
				f.source = f.b.activateEngineManagerSource()
				f.b.bindEngineMgr(f.b.getEngineMgr(), f.source)
				r := f.b.nodePolicyRuntime()
				r.mu.Lock()
				id := before.Identity.Engines["llamacpp"]
				r.engineIdentity = map[string]engineManagerIdentity{"llamacpp": {EngineRunID: id.EngineRunID, LaunchDigest: id.LaunchDigest, ModelDigests: id.ModelDigests, Serving: id.Serving}}
				r.intent["llamacpp"], r.running["llamacpp"] = true, true
				r.mu.Unlock()
			} else {
				pp, _, acks := attachAckingProxy(t, f.b)
				if !f.b.replayPolicyToProxy(context.Background(), pp) {
					t.Fatal("activate replacement proxy")
				}
				reportStrict(t, f.b, "0", acks.runID, strictCapability(t, "llamacpp", "resident"))
			}
			readback := residentLaunchReadback(before)
			manager.setAnswer(func(method string, _ json.RawMessage) (any, string) {
				if method == "engine:get-launch" {
					return readback, ""
				}
				if method == settings.SwitchPrepareMethod {
					return emptySwitchManagerSnapshot(), ""
				}
				return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
			})
			f.b.reconcileModelConfig(context.Background())
			status := f.b.modelConfiguration(f.b.captureModelConfig())
			if status.Models[0].State == nodepolicy.ModelStateSatisfied {
				t.Fatal("replacement inherited completion before fresh inventory")
			}
			f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
			for range 3 {
				f.b.reconcileModelConfig(context.Background())
			}
			status = f.b.modelConfiguration(f.b.captureModelConfig())
			if status.Models[0].State != nodepolicy.ModelStateSatisfied || len(modelConfigCalls(manager, "engine:get-launch")) != 1 {
				t.Fatalf("fresh observed envelope was not confirmed once: %+v", status)
			}
			wantLoads := 1
			if worker == "manager" {
				wantLoads = 0
			}
			if got := len(modelConfigCalls(manager, nodepolicy.MethodReconcileResident)); got != wantLoads {
				t.Fatalf("replacement issued duplicate resident load: got %d want %d", got, wantLoads)
			}
		})
	}
}

func TestModelConfigLoadedConfirmationRequiresObservedEnvelope(t *testing.T) {
	for _, invalid := range []string{"cold only", "wrong context", "wrong slots", "wrong envelope", "local Off", "local pause"} {
		t.Run(invalid, func(t *testing.T) {
			f := newModelConfigFixture(t)
			readback := residentLaunchReadback(f.b.captureModelConfig())
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			switch invalid {
			case "cold only":
				readback.ServingReadback.Observed = nil
			case "wrong context":
				readback.ServingReadback.Observed[0].PerSlotContextTokens++
			case "wrong slots":
				readback.ServingReadback.Observed[0].Slots++
			case "wrong envelope":
				readback.ServingReadback.LaunchValidated[0].RouterCapacity++
			case "local Off":
				r.intent["llamacpp"] = false
			case "local pause":
				r.localPauseRequested = true
			}
			r.mu.Unlock()
			f.engine.setAnswer(func(string, json.RawMessage) (any, string) { return readback, "" })
			f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
			f.b.reconcileModelConfig(context.Background())
			status := f.b.modelConfiguration(f.b.captureModelConfig())
			if status.Models[0].State == nodepolicy.ModelStateSatisfied || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 {
				t.Fatalf("invalid observation authorized satisfaction or load: %+v", status)
			}
		})
	}
}

func TestModelConfigConfirmingInventoryCannotSettleUncertainHold(t *testing.T) {
	f := newModelConfigFixture(t)
	f.proxy.mu.Lock()
	answer := f.proxy.answer
	f.proxy.mu.Unlock()
	f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodFinishResident {
			return nodepolicy.FinishResidentResult{ReservationID: "fixture-reservation", Outcome: nodepolicy.ResidentUnknown, RecoveryID: "fixture-recovery"}, ""
		}
		return answer(method, raw)
	})
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	readback := residentLaunchReadback(f.b.captureModelConfig())
	f.engine.setAnswer(func(string, json.RawMessage) (any, string) { return readback, "" })
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.b.reconcileModelConfig(context.Background())
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].State != nodepolicy.ModelStateUnknown || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 || len(modelConfigCalls(f.engine, "engine:get-launch")) != 0 {
		t.Fatalf("ordinary inventory retried or settled uncertain hold: %+v", status)
	}
}

func TestModelConfigUnchangedRevisionLoadRequestDoesNotReload(t *testing.T) {
	f := newModelConfigFixture(t)
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.b.reconcileModelConfig(context.Background())
	readback := residentLaunchReadback(f.b.captureModelConfig())
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		switch method {
		case "engine:get-launch":
			return readback, ""
		case settings.SwitchPrepareMethod:
			return emptySwitchManagerSnapshot(), ""
		default:
			return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
		}
	})
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Revision = "1"
	r.mu.Unlock()
	if !f.b.replayPolicyToProxy(context.Background(), f.pp) {
		t.Fatal("activate unchanged load revision")
	}
	reportStrict(t, f.b, "1", f.acks.runID, strictCapability(t, "llamacpp", "resident"))
	for range 3 {
		f.b.reconcileModelConfig(context.Background())
	}
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.PolicyRevision != "1" || status.Models[0].State != nodepolicy.ModelStateSatisfied || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 || len(modelConfigCalls(f.engine, "engine:get-launch")) != 1 {
		t.Fatalf("unchanged load revision reloaded or lost confirmation: status=%+v loads=%d readbacks=%d", status, len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)), len(modelConfigCalls(f.engine, "engine:get-launch")))
	}
}

func TestModelConfigReplacedRevisionReconcilesOnce(t *testing.T) {
	f := newModelConfigFixture(t)
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Revision = "1"
	// Replace the slice rather than mutating the old captured policy evidence.
	r.current.Profiles = append([]nodepolicy.Profile(nil), r.current.Profiles...)
	r.current.Profiles[0].ContextTokens = 8193
	r.mu.Unlock()
	// A later inventory publication must not relabel the old allocation as
	// loaded under the new revision's request.
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	stale := f.b.captureModelConfig()
	if stale.LoadedCurrent["llamacpp"]["resident"] {
		t.Fatal("changed load request accepted old inventory with unchanged identity")
	}
	oldReadback := residentLaunchReadback(stale)
	f.engine.setAnswer(func(string, json.RawMessage) (any, string) { return oldReadback, "" })
	if !f.b.replayPolicyToProxy(context.Background(), f.pp) {
		t.Fatal("activate changed load before identity publication")
	}
	reportStrict(t, f.b, "1", f.acks.runID, strictCapability(t, "llamacpp", "resident"))
	f.b.reconcileModelConfig(context.Background())
	if got := f.b.modelConfiguration(f.b.captureModelConfig()); got.Models[0].State == nodepolicy.ModelStateSatisfied || len(modelConfigCalls(f.engine, "engine:get-launch")) != 0 {
		t.Fatalf("old identity/readback confirmed changed request: %+v", got)
	}
	r.mu.Lock()
	id := r.engineIdentity["llamacpp"]
	envelope := id.Serving["resident"]
	envelope.PerSlotContextTokens, envelope.TotalContextTokens = 8193, 8193
	id.Serving["resident"] = envelope
	r.engineIdentity["llamacpp"] = id
	r.mu.Unlock()
	if !f.b.replayPolicyToProxy(context.Background(), f.pp) {
		t.Fatal("activate replacement revision")
	}
	reportStrict(t, f.b, "1", f.acks.runID, strictCapability(t, "llamacpp", "resident"))
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].State == nodepolicy.ModelStateSatisfied {
		t.Fatal("replacement revision inherited old completion")
	}
	f.engine.mu.Lock()
	f.engine.held = make(chan struct{})
	f.engine.holdOnce = sync.Once{}
	f.engine.mu.Unlock()
	done = f.startLoadedAttempt(t)
	f.engine.waitFor("replacement resident grant", func(calls []policyCall) bool {
		grants := 0
		for _, call := range calls {
			if call.method == residentDispatchReplyMethod {
				grants++
			}
		}
		return grants == 2
	})
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.b.reconcileModelConfig(context.Background())
	status = f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].State != nodepolicy.ModelStateSatisfied || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 2 {
		t.Fatalf("replacement revision did not reconcile exactly once: status=%+v loads=%d grants=%d", status, len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)), len(modelConfigCalls(f.engine, residentDispatchReplyMethod)))
	}
	call := modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)[1]
	p := decodeModelConfigCall[nodepolicy.ReconcileResidentParams](t, call)
	if p.PolicyRevision != "1" || p.Load.ContextLength != 8193 {
		t.Fatalf("replacement load lost new revision/envelope: %+v", p)
	}
}

func TestModelConfigResidentSetNormalizedPerEngine(t *testing.T) {
	f := newModelConfigFixture(t)
	p := f.b.captureModelConfig().Policy
	p.ResidentModels = nodepolicy.ResidentModels{
		{Engine: "ollama", Model: "m"}, {Engine: "llamacpp", Model: "resident"},
		{Engine: "ollama", Model: "m:latest"}, {Engine: "ollama", Model: "other"},
	}
	e := f.b.captureModelConfig()
	e.Policy = p
	_, _, _, count := f.b.modelConfigEntry(e, nodepolicy.Role{Engine: "ollama", Model: "m:latest"})
	if count != 2 {
		t.Fatalf("normalized ollama resident count=%d, want 2", count)
	}
	_, _, _, count = f.b.modelConfigEntry(e, nodepolicy.Role{Engine: "llamacpp", Model: "resident"})
	if count != 1 {
		t.Fatalf("llamacpp count=%d, want 1", count)
	}
	want := []nodepolicy.Role{{Engine: "llamacpp", Model: "resident"}, {Engine: "ollama", Model: "m:latest"}, {Engine: "ollama", Model: "other:latest"}}
	if got := f.b.residentRefs(p); !reflect.DeepEqual(got, want) {
		t.Fatalf("normalized deterministic resident set=%v, want %v", got, want)
	}
}

func assertModelConfigLoadFailed(t *testing.T, f *modelConfigFixture) {
	t.Helper()
	e := f.b.captureModelConfig()
	for _, read := range []struct {
		name   string
		status nodepolicy.ModelConfiguration
	}{
		{"current status", f.b.modelConfiguration(e)},
		{"local policy readback", f.b.localPolicy().ModelConfiguration},
		{"peer policy readback", f.b.peerPolicy().ModelConfiguration},
		{"revision readback", f.b.modelConfigurationFor(e.Policy)},
	} {
		if len(read.status.Models) != 1 {
			t.Errorf("%s models=%+v, want one failed resident", read.name, read.status.Models)
			continue
		}
		entry := read.status.Models[0]
		if entry.Engine != "llamacpp" || entry.Model != "resident" || !entry.DesiredResident || entry.State != "unknown" || entry.Reason != "load-failed" {
			t.Errorf("%s entry=%+v, want desired resident unknown/load-failed", read.name, entry)
		}
		if entry.ObservedLoaded == nil || *entry.ObservedLoaded {
			t.Errorf("%s lost the fixture's known empty inventory: %+v", read.name, entry)
		}
	}
}

func TestModelConfigDefiniteLoadFailureReason(t *testing.T) {
	test := func(name string, recovery bool) {
		t.Run(name, func(t *testing.T) {
			f := newModelConfigFixture(t)
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			initial := r.now()
			r.now = func() time.Time { return initial }
			r.mu.Unlock()
			if recovery {
				f.proxy.mu.Lock()
				answer := f.proxy.answer
				f.proxy.mu.Unlock()
				f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
					switch method {
					case nodepolicy.MethodFinishResident:
						return nodepolicy.FinishResidentResult{ReservationID: "fixture-reservation", Outcome: nodepolicy.ResidentFailed, RecoveryID: "fixture-recovery"}, ""
					case nodepolicy.MethodObserveResident:
						return nodepolicy.ObserveResidentResult{ReservationID: "fixture-reservation", Settled: true}, ""
					}
					return answer(method, raw)
				})
			}
			done := f.startResidentAttempt(t, nodepolicy.ResidentFailed)
			f.awaitGrant(t)
			if recovery {
				f.engine.mu.Lock()
				answer := f.engine.answer
				f.engine.mu.Unlock()
				f.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
					if method == "engine:models" {
						return map[string]any{"loadedByEngine": map[string][]string{"llamacpp": {}}}, ""
					}
					return answer(method, raw)
				})
			}
			f.engine.releaseHeld()
			awaitModelConfigDone(t, done)
			finishes := modelConfigCalls(f.proxy, nodepolicy.MethodFinishResident)
			if len(finishes) != 1 {
				t.Fatalf("finish calls=%d, want one", len(finishes))
			}
			finish := decodeModelConfigCall[nodepolicy.FinishResidentParams](t, finishes[0])
			if finish.Outcome != "failed" || finish.Observation != nil {
				t.Fatalf("definite manager failure was not finished intact: %+v", finish)
			}
			observations := modelConfigCalls(f.proxy, nodepolicy.MethodObserveResident)
			wantObservations := 0
			if recovery {
				wantObservations = 1
			}
			if len(observations) != wantObservations || len(modelConfigCalls(f.engine, "engine:models")) != wantObservations {
				t.Fatalf("recovery observations=%d inventory reads=%d, want %d each", len(observations), len(modelConfigCalls(f.engine, "engine:models")), wantObservations)
			}
			if recovery {
				observation := decodeModelConfigCall[nodepolicy.ObserveResidentParams](t, observations[0])
				if !observation.OperationComplete || observation.Observation.Loaded || observation.RecoveryID != "fixture-recovery" || observation.OperationID != finish.OperationID {
					t.Fatalf("failed load recovery lost its terminal/empty observation: %+v", observation)
				}
			}
			r.mu.Lock()
			holds, pending, retryAt := len(r.models.unsettled), len(r.models.pending), r.models.retryAt
			r.mu.Unlock()
			if holds != 0 || pending != 0 || !retryAt.Equal(initial.Add(30*time.Second)) || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
				t.Fatalf("failed load changed cleanup or retry: holds=%d pending=%d retryAt=%s", holds, pending, retryAt)
			}
			assertModelConfigLoadFailed(t, f)
		})
	}
	test("normal finish", false)
	test("recovered finish", true)
}

func TestModelConfigLoadFailureReasonIsStatusOnly(t *testing.T) {
	f := newModelConfigFixture(t)
	done := f.startResidentAttempt(t, nodepolicy.ResidentFailed)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	assertModelConfigLoadFailed(t, f)
	status := f.b.localPolicy().ModelConfiguration
	raw, err := json.Marshal(status)
	if err != nil {
		t.Fatal(err)
	}
	var readback nodepolicy.ModelConfiguration
	if err := json.Unmarshal(raw, &readback); err != nil {
		t.Fatalf("produced failed-load status rejected: %v", err)
	}
	if len(readback.Models) != 1 {
		t.Fatalf("produced status has no unique resident: %+v", readback)
	}
	entry := readback.Models[0]
	if entry.Reason != nodepolicy.ModelReasonLoadFailed {
		t.Fatalf("produced failure lost its status reason: %+v", entry)
	}
	refusal := nodepolicy.ResidentRefusal{Engine: entry.Engine, Model: entry.Model, Reason: entry.Reason}
	if err := refusal.Validate(); err == nil {
		t.Error("completed load failure accepted as a prepare refusal")
	}
	if err := (nodepolicy.DispatchResidentResult{Reason: nodepolicy.DispatchResidentReason(entry.Reason)}).Validate(); err == nil {
		t.Error("completed load failure accepted as a dispatch refusal")
	}
}

// The manager's ordinary definite failure carries its readable post-load
// inventory, which omits the model; the proxy settles it without recovery.
func TestModelConfigObservedLoadFailureReason(t *testing.T) {
	f := newModelConfigFixture(t)
	f.setResidentAttemptAnswer(t, nodepolicy.ResidentFailed)
	f.engine.mu.Lock()
	answer := f.engine.answer
	f.engine.mu.Unlock()
	f.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		// The fixture answer still grants the dispatch; only its result changes.
		value, reply := answer(method, raw)
		if method != nodepolicy.MethodReconcileResident {
			return value, reply
		}
		var p nodepolicy.ReconcileResidentParams
		if err := json.Unmarshal(raw, &p); err != nil {
			return nil, err.Error()
		}
		return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentFailed, Observation: &nodepolicy.ResidentLoadedObservation{
			ManagerIncarnation: p.Reservation.ManagerIncarnation, Generation: 1,
			Engine: p.Engine, Model: p.Model, Identity: p.Identity, Loaded: false,
		}}, reply
	})
	done := make(chan struct{})
	go func() { defer close(done); f.b.reconcileModelConfig(context.Background()) }()
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	finishes := modelConfigCalls(f.proxy, nodepolicy.MethodFinishResident)
	if len(finishes) != 1 {
		t.Fatalf("finish calls=%d, want one", len(finishes))
	}
	finish := decodeModelConfigCall[nodepolicy.FinishResidentParams](t, finishes[0])
	if finish.Outcome != nodepolicy.ResidentFailed || finish.Observation == nil || finish.Observation.Loaded {
		t.Fatalf("observed manager failure was not finished intact: %+v", finish)
	}
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	holds, pending := len(r.models.unsettled), len(r.models.pending)
	r.mu.Unlock()
	if holds != 0 || pending != 0 || len(modelConfigCalls(f.proxy, nodepolicy.MethodObserveResident)) != 0 || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
		t.Fatalf("settled observed failure left work behind: holds=%d pending=%d", holds, pending)
	}
	assertModelConfigLoadFailed(t, f)
}

func TestModelConfigDefiniteLoadFailureRetainedReason(t *testing.T) {
	for _, failure := range []string{"finish transport failure", "recovery inventory failure", "recovery observe failure"} {
		t.Run(failure, func(t *testing.T) {
			f := newModelConfigFixture(t)
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			initial := r.now()
			now := initial
			r.now = func() time.Time { return now }
			r.mu.Unlock()
			f.proxy.mu.Lock()
			answer := f.proxy.answer
			f.proxy.mu.Unlock()
			f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				switch method {
				case nodepolicy.MethodFinishResident:
					if failure == "finish transport failure" {
						return nil, "private fixture finish failure"
					}
					return nodepolicy.FinishResidentResult{ReservationID: "fixture-reservation", Outcome: nodepolicy.ResidentFailed, RecoveryID: "fixture-recovery"}, ""
				case nodepolicy.MethodObserveResident:
					return nil, "private fixture observation failure"
				}
				return answer(method, raw)
			})
			done := f.startResidentAttempt(t, nodepolicy.ResidentFailed)
			f.awaitGrant(t)
			f.engine.mu.Lock()
			engineAnswer := f.engine.answer
			f.engine.mu.Unlock()
			f.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == "engine:models" {
					if failure == "recovery inventory failure" {
						return nil, "private fixture inventory failure"
					}
					return map[string]any{"loadedByEngine": map[string][]string{"llamacpp": {}}}, ""
				}
				return engineAnswer(method, raw)
			})
			f.engine.releaseHeld()
			awaitModelConfigDone(t, done)
			ref := nodepolicy.Role{Engine: "llamacpp", Model: "resident"}
			r.mu.Lock()
			hold, held := r.models.unsettled[ref]
			retryAt := r.models.retryAt
			r.mu.Unlock()
			if !held || !hold.terminal || hold.op == nil || !hold.op.granted || hold.finish.Outcome != "failed" || !hold.expires.Equal(initial.Add(time.Minute)) {
				t.Fatalf("failed load lost its terminal hold: %+v held=%v", hold, held)
			}
			if (hold.op.recovery != nil) != (failure == "recovery observe failure") {
				t.Fatalf("unexpected retained recovery receipt: %+v", hold.op.recovery)
			}
			assertModelConfigLoadFailed(t, f)
			r.mu.Lock()
			now = initial.Add(30*time.Second - time.Nanosecond)
			r.mu.Unlock()
			f.b.reconcileModelConfig(context.Background())
			r.mu.Lock()
			unchangedRetry := r.models.retryAt
			r.mu.Unlock()
			if !retryAt.Equal(initial.Add(30*time.Second)) || !unchangedRetry.Equal(retryAt) || len(modelConfigCalls(f.proxy, nodepolicy.MethodFinishResident)) != 1 {
				t.Fatal("failure readback moved the retry deadline or retried early")
			}
			f.proxy.mu.Lock()
			retryAnswer := f.proxy.answer
			f.proxy.mu.Unlock()
			f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				value, errText := retryAnswer(method, raw)
				if method == nodepolicy.MethodFinishResident {
					if errText != "" {
						return nil, holdReply
					}
					return value, holdSuccessReply
				}
				return value, errText
			})
			r.mu.Lock()
			now = initial.Add(30 * time.Second)
			r.mu.Unlock()
			retried := make(chan struct{})
			go func() { defer close(retried); f.b.reconcileModelConfig(context.Background()) }()
			defer func() { f.proxy.releaseHeld(); awaitModelConfigDone(t, retried) }()
			f.proxy.waitFor("retained failed-load finish retry", func(calls []policyCall) bool {
				finishes := 0
				for _, call := range calls {
					if call.method == nodepolicy.MethodFinishResident {
						finishes++
					}
				}
				return finishes == 2
			})
			assertModelConfigLoadFailed(t, f)
			f.proxy.releaseHeld()
			awaitModelConfigDone(t, retried)
			r.mu.Lock()
			after, retained := r.models.unsettled[ref]
			retryAt = r.models.retryAt
			r.mu.Unlock()
			if !retained || after.op != hold.op || !after.expires.Equal(hold.expires) || !reflect.DeepEqual(after.finish, hold.finish) || !retryAt.Equal(initial.Add(time.Minute)) || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
				t.Fatal("failed-load retry changed hold ownership, expiry, deadline or load count")
			}
			assertModelConfigLoadFailed(t, f)
		})
	}
}

func TestModelConfigUncertainReadbackSurvivesInventoryWithdrawal(t *testing.T) {
	f := newModelConfigFixture(t)
	f.proxy.mu.Lock()
	answer := f.proxy.answer
	f.proxy.mu.Unlock()
	f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodFinishResident {
			return nodepolicy.FinishResidentResult{ReservationID: "fixture-reservation", Outcome: nodepolicy.ResidentUnknown, RecoveryID: "fixture-recovery"}, ""
		}
		return answer(method, raw)
	})
	done := f.startResidentAttempt(t, nodepolicy.ResidentUnknown)
	f.awaitGrant(t)
	f.engine.mu.Lock()
	engineAnswer := f.engine.answer
	f.engine.mu.Unlock()
	f.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:models" {
			return nil, "private fixture inventory unavailable"
		}
		return engineAnswer(method, raw)
	})
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	r := f.b.nodePolicyRuntime()
	ref := nodepolicy.Role{Engine: "llamacpp", Model: "resident"}
	r.mu.Lock()
	hold, held := r.models.unsettled[ref]
	r.mu.Unlock()
	if !held || !hold.terminal || !hold.op.granted || hold.finish.Outcome != nodepolicy.ResidentUnknown {
		t.Fatal("fixture did not retain its dispatched uncertain operation")
	}
	// The watcher now withdraws failed inventory instead of retaining its old
	// loaded set. That must not erase the dispatched operation's uncertainty.
	f.b.handlePolicyEngineNotification(f.source, engineModelsChangedMethod, json.RawMessage(`{"models":{"models":[]}}`))
	assertReadback := func() {
		t.Helper()
		e := f.b.captureModelConfig()
		for _, status := range []nodepolicy.ModelConfiguration{
			f.b.modelConfiguration(e), f.b.localPolicy().ModelConfiguration,
			f.b.peerPolicy().ModelConfiguration, f.b.modelConfigurationFor(e.Policy),
		} {
			if len(status.Models) != 1 || status.Models[0].State != nodepolicy.ModelStateUnknown || status.Models[0].Reason != nodepolicy.ModelReasonOperationTimeout || status.Models[0].ObservedLoaded != nil {
				t.Errorf("withdrawn inventory lost uncertain operation: %+v", status)
			}
		}
	}
	assertReadback()
	f.b.reconcileModelConfig(context.Background())
	assertReadback()
	r.mu.Lock()
	after, retained := r.models.unsettled[ref]
	r.mu.Unlock()
	if !retained || after.op != hold.op || !after.expires.Equal(hold.expires) || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
		t.Fatal("inventory withdrawal changed the retained hold or dispatched another load")
	}
	checkVeto := func(name string, set func(bool), reason nodepolicy.ModelConfigurationReason) {
		t.Helper()
		r.mu.Lock()
		set(true)
		r.mu.Unlock()
		status := f.b.modelConfiguration(f.b.captureModelConfig())
		if len(status.Models) != 1 || status.Models[0].State != nodepolicy.ModelStateBlocked || status.Models[0].Reason != reason {
			t.Errorf("%s was hidden by uncertain operation: %+v", name, status)
		}
		r.mu.Lock()
		set(false)
		r.mu.Unlock()
		assertReadback()
	}
	checkVeto("local pause", func(on bool) { r.localPauseRequested = on }, nodepolicy.ModelReasonPaused)
	checkVeto("engine Off", func(on bool) { r.intent["llamacpp"] = !on }, nodepolicy.ModelReasonEngineOff)
}

func TestModelConfigRetainedFailureReadbackTracksInventory(t *testing.T) {
	for _, inventory := range []string{"changed inventory", "unknown inventory"} {
		t.Run(inventory, func(t *testing.T) {
			f := newModelConfigFixture(t)
			f.proxy.mu.Lock()
			answer := f.proxy.answer
			f.proxy.mu.Unlock()
			f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == nodepolicy.MethodFinishResident {
					return nil, "private fixture finish failure"
				}
				return answer(method, raw)
			})
			done := f.startResidentAttempt(t, nodepolicy.ResidentFailed)
			f.awaitGrant(t)
			f.engine.releaseHeld()
			awaitModelConfigDone(t, done)
			r := f.b.nodePolicyRuntime()
			ref := nodepolicy.Role{Engine: "llamacpp", Model: "resident"}
			r.mu.Lock()
			hold, held := r.models.unsettled[ref]
			if inventory == "unknown inventory" {
				delete(r.loaded, "llamacpp")
			} else {
				r.loaded["llamacpp"] = []string{"other-model"}
			}
			r.mu.Unlock()
			if !held || !hold.terminal || hold.finish.Outcome != "failed" {
				t.Fatal("fixture did not retain the definite manager failure")
			}
			assertReadback := func() {
				t.Helper()
				e := f.b.captureModelConfig()
				for _, status := range []nodepolicy.ModelConfiguration{
					f.b.modelConfiguration(e), f.b.localPolicy().ModelConfiguration,
					f.b.peerPolicy().ModelConfiguration, f.b.modelConfigurationFor(e.Policy),
				} {
					if len(status.Models) != 1 || status.Models[0].State != "unknown" || status.Models[0].Reason != "load-failed" {
						t.Errorf("retained failed-load readback=%+v, want unknown/load-failed", status)
						continue
					}
					observed := status.Models[0].ObservedLoaded
					if inventory == "unknown inventory" {
						if observed != nil {
							t.Errorf("failure invented loaded evidence for unknown inventory: %v", *observed)
						}
					} else if observed == nil || *observed {
						t.Error("failure lost current inventory's known absent model")
					}
				}
			}
			assertReadback()
			f.b.reconcileModelConfig(context.Background())
			assertReadback()
			r.mu.Lock()
			after, retained := r.models.unsettled[ref]
			r.mu.Unlock()
			if !retained || after.op != hold.op || !after.expires.Equal(hold.expires) || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
				t.Fatal("inventory readback changed the retained hold or dispatched another load")
			}
		})
	}
}

// A retained failure explains an otherwise eligible resident. A current local
// veto still decides the published state without discharging that failure.
func TestModelConfigRetainedFailureKeepsCurrentVeto(t *testing.T) {
	test := func(name string, setVeto func(r *policyRuntime, enabled bool), reason nodepolicy.ModelConfigurationReason) {
		t.Run(name, func(t *testing.T) {
			f := newModelConfigFixture(t)
			f.proxy.mu.Lock()
			answer := f.proxy.answer
			f.proxy.mu.Unlock()
			f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == nodepolicy.MethodFinishResident {
					return nil, "private fixture finish failure"
				}
				return answer(method, raw)
			})
			done := f.startResidentAttempt(t, nodepolicy.ResidentFailed)
			f.awaitGrant(t)
			f.engine.releaseHeld()
			awaitModelConfigDone(t, done)
			r := f.b.nodePolicyRuntime()
			ref := nodepolicy.Role{Engine: "llamacpp", Model: "resident"}
			r.mu.Lock()
			hold, held := r.models.unsettled[ref]
			setVeto(r, true)
			r.mu.Unlock()
			if !held || !hold.terminal || hold.finish.Outcome != nodepolicy.ResidentFailed {
				t.Fatal("fixture did not retain the definite manager failure")
			}
			assertVeto := func(phase string) {
				t.Helper()
				e := f.b.captureModelConfig()
				for _, status := range []nodepolicy.ModelConfiguration{
					f.b.modelConfiguration(e), f.b.localPolicy().ModelConfiguration,
					f.b.peerPolicy().ModelConfiguration, f.b.modelConfigurationFor(e.Policy),
				} {
					if len(status.Models) != 1 || status.Models[0].State != nodepolicy.ModelStateBlocked || status.Models[0].Reason != reason {
						t.Errorf("%s with retained failure=%+v, want blocked/%s", phase, status.Models, reason)
					}
				}
			}
			assertVeto("readback")
			f.b.reconcileModelConfig(context.Background())
			assertVeto("reconcile")
			r.mu.Lock()
			after, retained := r.models.unsettled[ref]
			setVeto(r, false)
			r.mu.Unlock()
			if !retained || after.op != hold.op || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
				t.Fatal("veto discharged the retained failure or dispatched another load")
			}
			assertModelConfigLoadFailed(t, f)
		})
	}
	test("local pause", func(r *policyRuntime, enabled bool) { r.localPauseRequested = enabled }, nodepolicy.ModelReasonPaused)
	test("pending peer pause", func(r *policyRuntime, enabled bool) { r.pendingPauseRequested = enabled }, nodepolicy.ModelReasonPaused)
	test("saved peer pause before live transition", func(r *policyRuntime, enabled bool) {
		r.current.PausedBy = nodepolicy.PausedByPeer
		r.current.Availability = nodepolicy.Available
		if enabled {
			r.current.Availability = nodepolicy.Paused
		}
	}, nodepolicy.ModelReasonPaused)
	test("ordinary saved Off", func(r *policyRuntime, enabled bool) { r.intent["llamacpp"] = !enabled }, nodepolicy.ModelReasonEngineOff)
}

func TestModelConfigNonfailureOutcomeReasons(t *testing.T) {
	test := func(name string, outcome nodepolicy.ResidentOutcome, transportFailure bool, state nodepolicy.ModelConfigurationState, reason nodepolicy.ModelConfigurationReason) {
		t.Run(name, func(t *testing.T) {
			f := newModelConfigFixture(t)
			if outcome == nodepolicy.ResidentNotDispatched {
				f.b.reconcileModelConfig(context.Background())
			} else {
				done, _, _ := startModelConfigMatrixAttempt(t, context.Background(), f, emptySwitchManagerSnapshot(), outcome, transportFailure)
				f.awaitGrant(t)
				f.engine.releaseHeld()
				awaitModelConfigDone(t, done)
			}
			status := f.b.localPolicy().ModelConfiguration
			if len(status.Models) != 1 || status.Models[0].State != state || status.Models[0].Reason != reason {
				t.Fatalf("%s status=%+v, want %s/%s", name, status, state, reason)
			}
			if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 1 {
				t.Fatalf("manager reconcile calls=%d, want one", got)
			}
		})
	}
	test("loaded", nodepolicy.ResidentLoaded, false, "satisfied", "")
	test("terminal unknown", nodepolicy.ResidentUnknown, false, "unknown", "operation-timeout")
	test("terminal timeout", nodepolicy.ResidentTimeout, false, "unknown", "operation-timeout")
	test("terminal cancelled", nodepolicy.ResidentCancelled, false, "unknown", "operation-timeout")
	test("transport failure", nodepolicy.ResidentUnknown, true, "unknown", "operation-timeout")
	test("not dispatched", nodepolicy.ResidentNotDispatched, false, "blocked", "unsupported-load")
}

func TestModelConfigTransientEngineOffKeepsPendingAndRetryDeadline(t *testing.T) {
	f := newModelConfigFixture(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	initial := r.now()
	now := initial
	r.now = func() time.Time { return now }
	r.mu.Unlock()
	f.proxy.mu.Lock()
	answer := f.proxy.answer
	f.proxy.mu.Unlock()
	f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodPrepareResident {
			return nodepolicy.PrepareResidentResult{EvidenceGeneration: 1, Refusals: []nodepolicy.ResidentRefusal{{Engine: "llamacpp", Model: "resident", Reason: nodepolicy.ModelReasonEngineOff}}}, ""
		}
		return answer(method, raw)
	})
	f.b.reconcileModelConfig(context.Background())
	e := f.b.captureModelConfig()
	if e.Source != f.source || !e.Running["llamacpp"] {
		t.Fatal("transient refusal fixture lost its current running source")
	}
	assertPending := func() {
		t.Helper()
		status := f.b.localPolicy().ModelConfiguration
		if len(status.Models) != 1 || status.Models[0].State != "pending" || status.Models[0].Reason != "" {
			t.Fatalf("current running source with stale engine-off refusal=%+v, want pending without reason", status)
		}
	}
	assertPending()
	r.mu.Lock()
	retryAt := r.models.retryAt
	now = initial.Add(30*time.Second - time.Nanosecond)
	r.mu.Unlock()
	f.b.reconcileModelConfig(context.Background())
	assertPending()
	r.mu.Lock()
	unchangedRetry := r.models.retryAt
	r.mu.Unlock()
	if !retryAt.Equal(initial.Add(30*time.Second)) || !unchangedRetry.Equal(retryAt) || len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)) != 1 {
		t.Fatal("transient pending status moved its retry deadline or retried early")
	}
	r.mu.Lock()
	now = retryAt
	r.mu.Unlock()
	f.b.reconcileModelConfig(context.Background())
	assertPending()
	r.mu.Lock()
	nextRetry, holds := r.models.retryAt, len(r.models.unsettled)
	r.mu.Unlock()
	if !nextRetry.Equal(initial.Add(time.Minute)) || len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)) != 2 || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 || len(modelConfigCalls(f.proxy, nodepolicy.MethodFinishResident)) != 0 || holds != 0 {
		t.Fatal("transient engine-off refusal changed retry cadence or dispatched a load")
	}
}

func TestModelConfigFailedAttemptWaitsForChangedEvidence(t *testing.T) {
	f := newModelConfigFixture(t)
	for range 5 {
		f.b.reconcileModelConfig(context.Background())
		f.b.signalModelConfig()
	}
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 1 {
		t.Fatalf("unchanged evidence retried failed attempt %d times", got)
	}
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.loaded["llamacpp"] = []string{"new-observation"}
	r.mu.Unlock()
	f.b.reconcileModelConfig(context.Background())
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 2 {
		t.Fatalf("changed evidence did not allow one retry: calls=%d", got)
	}
}

func TestModelConfigFailedAttemptRetriesOnTick(t *testing.T) {
	f := newModelConfigFixture(t)
	f.b.reconcileModelConfig(context.Background())
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	now := r.now().Add(idleCheckInterval + time.Second)
	r.now = func() time.Time { return now }
	r.mu.Unlock()
	f.b.reconcileModelConfig(context.Background())
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 2 {
		t.Fatalf("unchanged failed attempt did not retry on tick: loads=%d", got)
	}
}

func TestModelConfigAdmissionRefusalsHaveNoLoad(t *testing.T) {
	cases := []struct {
		name   string
		change func(*policyRuntime)
	}{
		{"missing activation ack", func(r *policyRuntime) { r.activeRevision = "old" }},
		{"local Off", func(r *policyRuntime) { r.intent["llamacpp"] = false }},
		{"local pause", func(r *policyRuntime) { r.localPauseRequested = true }},
		{"unknown loaded inventory", func(r *policyRuntime) { delete(r.loaded, "llamacpp") }},
		{"missing qualification", func(r *policyRuntime) { r.strictReadiness.Capability = nil }},
		{"missing unique profile", func(r *policyRuntime) { r.current.Profiles = nil }},
		{"engine drain", func(r *policyRuntime) { r.drains = map[string]bool{"llamacpp": true} }},
		{"context exceeds native envelope", func(r *policyRuntime) { r.current.Profiles[0].ContextTokens++ }},
		{"unsupported native retention", func(r *policyRuntime) {
			id := r.engineIdentity["llamacpp"]
			envelope := id.Serving["resident"]
			envelope.Retention = "300"
			id.Serving["resident"] = envelope
			r.engineIdentity["llamacpp"] = id
		}},
		{"native router capacity absent", func(r *policyRuntime) {
			id := r.engineIdentity["llamacpp"]
			envelope := id.Serving["resident"]
			envelope.RouterCapacity = 0
			id.Serving["resident"] = envelope
			r.engineIdentity["llamacpp"] = id
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newModelConfigFixture(t)
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			tc.change(r)
			r.mu.Unlock()
			f.b.reconcileModelConfig(context.Background())
			if len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 || len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)) != 0 {
				t.Fatal("refused evidence reached resident preparation or manager load")
			}
		})
	}
}

func TestModelConfigConcurrentTriggersCoalesce(t *testing.T) {
	f := newModelConfigFixture(t)
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			return emptySwitchManagerSnapshot(), ""
		}
		if method == nodepolicy.MethodReconcileResident {
			return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, holdSuccessReply
		}
		return map[string]bool{"ok": true}, ""
	})
	done := make(chan struct{})
	go func() { defer close(done); f.b.reconcileModelConfig(context.Background()) }()
	f.engine.waitFor("held resident request", func(calls []policyCall) bool {
		return len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) == 1
	})
	var wg sync.WaitGroup
	for range 12 {
		wg.Add(1)
		go func() { defer wg.Done(); f.b.reconcileModelConfig(context.Background()) }()
	}
	wg.Wait()
	f.engine.releaseHeld()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("resident operation did not finish")
	}
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 1 {
		t.Fatalf("concurrent triggers started %d manager operations", got)
	}
}

func TestModelConfigStatusUnknownEmptyAndRemoval(t *testing.T) {
	f := newModelConfigFixture(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	delete(r.loaded, "llamacpp")
	r.mu.Unlock()
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if len(status.Models) != 1 || status.Models[0].ObservedLoaded != nil || status.Models[0].State != nodepolicy.ModelStateUnknown {
		t.Fatalf("unknown inventory status=%+v", status)
	}
	r.mu.Lock()
	r.loaded["llamacpp"] = []string{}
	r.mu.Unlock()
	status = f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].ObservedLoaded == nil || *status.Models[0].ObservedLoaded || status.Models[0].State == nodepolicy.ModelStateSatisfied {
		t.Fatalf("known empty inventory status=%+v", status)
	}
	r.mu.Lock()
	r.current.ResidentModels = nil
	r.mu.Unlock()
	f.b.reconcileModelConfig(context.Background())
	status = f.b.modelConfiguration(f.b.captureModelConfig())
	if len(status.Models) != 0 || status.Models == nil {
		t.Fatalf("removed resident status=%+v", status)
	}
	for _, call := range f.engine.snapshot() {
		if call.method == "engine:unload-model" || call.method == "engine:stop" || call.method == nodepolicy.MethodReconcileResident {
			t.Fatalf("removing preference sent %s", call.method)
		}
	}
}

func TestModelConfigSnapshotEmptyInventoryLoadsNewResident(t *testing.T) {
	f := newModelConfigFixture(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.ResidentModels = nil
	r.loaded = nil
	r.models.loadedSource = nil
	r.models.loadedProxy = nil
	r.mu.Unlock()
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		switch method {
		case "engine:get-installed":
			return map[string]any{"engines": []any{map[string]any{"engine": "llamacpp", "running": true}}}, ""
		case "engine:models":
			return map[string]any{"models": []string{"resident"}, "loadedByEngine": map[string][]string{"llamacpp": {}}}, ""
		default:
			return map[string]bool{"ok": true}, ""
		}
	})
	f.b.refreshEnginePolicySnapshotFrom(f.b.getEngineMgr(), f.source)
	r.mu.Lock()
	r.current.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "resident"}}
	r.mu.Unlock()
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if len(status.Models) != 1 || status.Models[0].ObservedLoaded == nil || *status.Models[0].ObservedLoaded || status.Models[0].State != nodepolicy.ModelStatePending {
		t.Fatalf("empty manager snapshot did not authorize resident load: %+v", status)
	}
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 1 {
		t.Fatalf("manager resident calls = %d, want one", got)
	}
	status = f.b.modelConfiguration(f.b.captureModelConfig())
	if len(status.Models) != 1 || status.Models[0].State != nodepolicy.ModelStateSatisfied || status.Models[0].ObservedLoaded == nil || !*status.Models[0].ObservedLoaded {
		t.Fatalf("correlated load did not satisfy resident: %+v", status)
	}
}

func TestModelConfigUnknownInventoryDoesNotLoadNewResident(t *testing.T) {
	f := newModelConfigFixture(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.loaded = nil
	r.models.loadedSource = nil
	r.models.loadedProxy = nil
	r.mu.Unlock()
	f.b.reconcileModelConfig(context.Background())
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if len(status.Models) != 1 || status.Models[0].ObservedLoaded != nil || status.Models[0].State != nodepolicy.ModelStateUnknown {
		t.Fatalf("missing snapshot status = %+v", status)
	}
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 0 {
		t.Fatalf("unknown inventory dispatched %d loads", got)
	}
}

func TestModelConfigRestartStartResponseRecoversResident(t *testing.T) {
	f := newModelConfigFixture(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	delete(r.running, "llamacpp") // Attach snapshot preceded the engine start.
	r.mu.Unlock()
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == "engine:start" {
			return map[string]any{"engine": "llamacpp", "running": true}, ""
		}
		return map[string]bool{"ok": true}, ""
	})
	f.b.relayToEngineNow(clientRequest(1, "engine:start", map[string]string{"engine": "llamacpp"}))
	response := f.out.waitFrame(t, "engine:start response", isResponse(1))
	if response.Error != nil {
		t.Fatalf("engine:start response: %+v", response.Error)
	}
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if len(status.Models) != 1 || status.Models[0].State != nodepolicy.ModelStatePending {
		t.Fatalf("running start result left resident blocked: %+v", status)
	}
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	status = f.b.modelConfiguration(f.b.captureModelConfig())
	if len(status.Models) != 1 || status.Models[0].State != nodepolicy.ModelStateSatisfied {
		t.Fatalf("resident did not settle: %+v", status)
	}
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 1 {
		t.Fatalf("resident loads = %d, want one", got)
	}
}

func TestModelConfigRestartProxyEngineOffRefusalRetries(t *testing.T) {
	f := newModelConfigFixture(t)
	f.proxy.mu.Lock()
	answer := f.proxy.answer
	f.proxy.mu.Unlock()
	refused := false
	f.proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
		if method == nodepolicy.MethodPrepareResident && !refused {
			refused = true
			return nodepolicy.PrepareResidentResult{EvidenceGeneration: 1, Refusals: []nodepolicy.ResidentRefusal{{Engine: "llamacpp", Model: "resident", Reason: nodepolicy.ModelReasonEngineOff}}}, ""
		}
		return answer(method, params)
	})
	f.b.reconcileModelConfig(context.Background())
	status := f.b.localPolicy().ModelConfiguration
	if len(status.Models) != 1 || status.Models[0].State != nodepolicy.ModelStatePending || status.Models[0].Reason != "" {
		t.Fatalf("running engine with stale proxy view = %+v, want pending", status)
	}
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 0 {
		t.Fatalf("refused prepare dispatched %d loads", got)
	}
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.now = time.Now
	r.models.retryAt = time.Now().Add(60 * time.Millisecond)
	r.mu.Unlock()
	f.setResidentAttemptAnswer(t, nodepolicy.ResidentLoaded)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); f.b.runIdlePolicy(ctx) }()
	defer func() { cancel(); awaitModelConfigDone(t, done) }()
	f.awaitGrant(t)
	f.engine.releaseHeld()
	settled := time.After(2 * time.Second)
	for {
		r.mu.Lock()
		busy := r.models.busy
		r.mu.Unlock()
		if !busy {
			break
		}
		select {
		case <-settled:
			t.Fatal("retried resident reconciliation did not finish")
		case <-time.After(10 * time.Millisecond):
		}
	}
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.b.reconcileModelConfig(context.Background())
	status = f.b.localPolicy().ModelConfiguration
	if len(status.Models) != 1 || status.Models[0].State != nodepolicy.ModelStateSatisfied {
		t.Fatalf("resident did not recover after proxy caught up: %+v", status)
	}
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 1 {
		t.Fatalf("resident loads = %d, want one", got)
	}
}

func TestModelConfigRetryWakesAtDeadlineBetweenIdleTicks(t *testing.T) {
	f := newModelConfigFixture(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.now = time.Now
	r.mu.Unlock()
	f.b.reconcileModelConfig(context.Background())
	if got := len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)); got != 1 {
		t.Fatalf("initial prepares = %d, want one", got)
	}
	r.mu.Lock()
	r.models.retryAt = time.Now().Add(60 * time.Millisecond)
	r.mu.Unlock()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); f.b.runIdlePolicy(ctx) }()
	defer func() { cancel(); awaitModelConfigDone(t, done) }()
	deadline := time.After(500 * time.Millisecond)
	for len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)) < 2 {
		select {
		case <-deadline:
			r.mu.Lock()
			retryAt, now, busy := r.models.retryAt, r.now(), r.models.busy
			r.mu.Unlock()
			t.Fatalf("retry missed deadline: retryAt=%s now=%s busy=%v status=%+v calls=%v", retryAt, now, busy, f.b.localPolicy().ModelConfiguration, f.proxy.methods(""))
		case <-time.After(10 * time.Millisecond):
		}
	}
}

func TestModelConfigRestartSnapshotDoesNotUndoNewerRunningStatus(t *testing.T) {
	f := newModelConfigFixture(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	delete(r.running, "llamacpp")
	r.mu.Unlock()
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		switch method {
		case "engine:get-installed":
			return map[string]any{"engines": []any{map[string]any{"engine": "llamacpp", "running": false}}}, holdSuccessReply
		case "engine:start", "engine:status":
			return map[string]any{"engine": "llamacpp", "running": true}, ""
		case "engine:models":
			return map[string]any{"models": []string{}, "loadedByEngine": map[string][]string{}}, ""
		default:
			return map[string]bool{"ok": true}, ""
		}
	})
	done := make(chan struct{})
	go func() {
		defer close(done)
		f.b.refreshEnginePolicySnapshotFrom(f.b.getEngineMgr(), f.source)
	}()
	f.engine.waitFor("initial installed snapshot", func(calls []policyCall) bool {
		return len(modelConfigCalls(f.engine, "engine:get-installed")) > 0
	})
	for id, method := range []string{"engine:start", "engine:status"} {
		requestID := id + 1
		f.b.relayToEngineNow(clientRequest(requestID, method, map[string]string{"engine": "llamacpp"}))
		response := f.out.waitFrame(t, method+" response", isResponse(requestID))
		if response.Error != nil {
			t.Fatalf("%s response: %+v", method, response.Error)
		}
	}
	f.engine.releaseHeld()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("initial installed snapshot did not finish")
	}
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if len(status.Models) != 1 || status.Models[0].State != nodepolicy.ModelStateUnknown || status.Models[0].Reason != nodepolicy.ModelReasonObservationUnknown {
		t.Fatalf("stale installed snapshot overrode running status: %+v", status)
	}
}

func TestModelConfigStoppedStatusKeepsEngineOff(t *testing.T) {
	f := newModelConfigFixture(t)
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == "engine:status" {
			return map[string]any{"engine": "llamacpp", "running": false}, ""
		}
		return map[string]bool{"ok": true}, ""
	})
	f.b.relayToEngineNow(clientRequest(1, "engine:status", map[string]string{"engine": "llamacpp"}))
	response := f.out.waitFrame(t, "engine:status response", isResponse(1))
	if response.Error != nil {
		t.Fatalf("engine:status response: %+v", response.Error)
	}
	f.b.reconcileModelConfig(context.Background())
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if len(status.Models) != 1 || status.Models[0].State != nodepolicy.ModelStateBlocked || status.Models[0].Reason != nodepolicy.ModelReasonEngineOff {
		t.Fatalf("stopped engine status = %+v, want engine-off", status)
	}
	if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 0 {
		t.Fatalf("stopped engine dispatched %d loads", got)
	}
}

func TestModelConfigInternalClientMethodsDenied(t *testing.T) {
	for _, method := range []string{nodepolicy.MethodReconcileResident, residentDispatchRequestMethod, residentDispatchReplyMethod} {
		t.Run(method, func(t *testing.T) {
			b, out := newPolicyBroker(t)
			engine := attachFakeEngine(t, b)
			b.handleMessage(clientRequest(1, method, map[string]string{"id": "forged"}))
			response := out.waitFrame(t, method, isResponse(1))
			if response.Error == nil || response.Error.Code != -32601 || len(engine.snapshot()) != 0 {
				t.Fatal("internal resident method reached a worker from public client")
			}
		})
	}
}

func (f *modelConfigFixture) startLoadedAttempt(t *testing.T) <-chan struct{} {
	return f.startResidentAttempt(t, nodepolicy.ResidentLoaded)
}

func (f *modelConfigFixture) startResidentAttempt(t *testing.T, outcome nodepolicy.ResidentOutcome) <-chan struct{} {
	t.Helper()
	f.setResidentAttemptAnswer(t, outcome)
	done := make(chan struct{})
	go func() { defer close(done); f.b.reconcileModelConfig(context.Background()) }()
	return done
}

func (f *modelConfigFixture) setResidentAttemptAnswer(t *testing.T, outcome nodepolicy.ResidentOutcome) {
	t.Helper()
	f.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			return emptySwitchManagerSnapshot(), ""
		}
		if method != nodepolicy.MethodReconcileResident {
			return map[string]bool{"ok": true}, ""
		}
		var p nodepolicy.ReconcileResidentParams
		if err := json.Unmarshal(raw, &p); err != nil {
			return nil, err.Error()
		}
		request, err := json.Marshal(struct {
			ID     string                            `json:"id"`
			Params nodepolicy.DispatchResidentParams `json:"params"`
		}{"fixture-dispatch", nodepolicy.DispatchResidentParams{
			ReservationID: p.Reservation.ReservationID, ProxyRunID: p.Reservation.ProxyRunID,
			ManagerIncarnation: p.Reservation.ManagerIncarnation, PolicyRevision: p.PolicyRevision,
			OperationID: p.OperationID, Engine: p.Engine, Model: p.Model, Identity: p.Identity,
		}})
		if err != nil {
			return nil, err.Error()
		}
		f.b.dispatchResidentFrom(f.source, request)
		result := nodepolicy.ReconcileResidentResult{Outcome: outcome}
		if outcome == nodepolicy.ResidentLoaded {
			result.Observation = &nodepolicy.ResidentLoadedObservation{
				ManagerIncarnation: p.Reservation.ManagerIncarnation, Generation: 1,
				Engine: p.Engine, Model: p.Model, Identity: p.Identity, Loaded: true,
			}
		}
		return result, holdSuccessReply
	})
}

func awaitModelConfigDone(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("resident reconciliation did not complete")
	}
}

func (f *modelConfigFixture) awaitGrant(t *testing.T) {
	t.Helper()
	f.engine.waitFor("correlated resident grant reply", func(calls []policyCall) bool {
		for _, call := range calls {
			if call.method == residentDispatchReplyMethod {
				return true
			}
		}
		return false
	})
	calls := modelConfigCalls(f.engine, residentDispatchReplyMethod)
	var reply struct {
		ID     string                            `json:"id"`
		Result nodepolicy.DispatchResidentResult `json:"result"`
	}
	if err := json.Unmarshal(calls[0].params, &reply); err != nil {
		t.Fatal(err)
	}
	if reply.ID != "fixture-dispatch" || !reply.Result.Dispatched {
		t.Fatalf("correlated grant reply=%+v", reply)
	}
}

func TestModelConfigGrantCompletesOnlyAfterFinish(t *testing.T) {
	f := newModelConfigFixture(t)
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].State == nodepolicy.ModelStateSatisfied {
		t.Fatal("grant alone published satisfied before terminal load and finish")
	}
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	status = f.b.modelConfiguration(f.b.captureModelConfig())
	if len(status.Models) != 1 || status.Models[0].State != nodepolicy.ModelStateSatisfied || status.Models[0].ObservedLoaded == nil || !*status.Models[0].ObservedLoaded {
		t.Fatalf("finished resident load status=%+v", status)
	}
	if got := len(modelConfigCalls(f.proxy, nodepolicy.MethodDispatchResident)); got != 1 {
		t.Fatalf("proxy final grants=%d, want one", got)
	}
}

func TestModelConfigManagerReplacementInvalidatesPreparedWork(t *testing.T) {
	f := newModelConfigFixture(t)
	f.proxy.mu.Lock()
	answer := f.proxy.answer
	f.proxy.mu.Unlock()
	f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, failure := answer(method, raw)
		if method == nodepolicy.MethodPrepareResident {
			return result, holdSuccessReply
		}
		return result, failure
	})
	done := make(chan struct{})
	go func() { defer close(done); f.b.reconcileModelConfig(context.Background()) }()
	f.proxy.waitFor("held resident preparation", func(calls []policyCall) bool {
		return len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)) == 1
	})
	replacement := attachFakeEngine(t, f.b)
	f.proxy.releaseHeld()
	awaitModelConfigDone(t, done)
	if len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 || len(modelConfigCalls(replacement, nodepolicy.MethodReconcileResident)) != 0 {
		t.Fatal("stale prepared work reached original or replacement manager")
	}
	if len(modelConfigCalls(f.proxy, nodepolicy.MethodFinishResident)) != 1 {
		t.Fatal("stale reservation was not finished on its original proxy")
	}
}

func TestModelConfigManagerReplacementDuringLoadOrFinish(t *testing.T) {
	for _, phase := range []string{"load", "finish"} {
		t.Run(phase, func(t *testing.T) {
			f := newModelConfigFixture(t)
			if phase == "finish" {
				f.proxy.mu.Lock()
				answer := f.proxy.answer
				f.proxy.mu.Unlock()
				f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
					result, failure := answer(method, raw)
					if method == nodepolicy.MethodFinishResident {
						return result, holdSuccessReply
					}
					return result, failure
				})
			}
			done := f.startLoadedAttempt(t)
			f.awaitGrant(t)
			if phase == "finish" {
				f.engine.releaseHeld()
				f.proxy.waitFor("held resident finish", func(calls []policyCall) bool {
					return len(modelConfigCalls(f.proxy, nodepolicy.MethodFinishResident)) == 1
				})
			}
			replacement := attachFakeEngine(t, f.b)
			f.engine.releaseHeld()
			f.proxy.releaseHeld()
			awaitModelConfigDone(t, done)
			status := f.b.modelConfiguration(f.b.captureModelConfig())
			if status.Models[0].State == nodepolicy.ModelStateSatisfied || len(modelConfigCalls(replacement, nodepolicy.MethodReconcileResident)) != 0 {
				t.Fatal("replacement inherited old load completion or resident work")
			}
		})
	}
}

func TestModelConfigManagerReplacementDuringGrant(t *testing.T) {
	f := newModelConfigFixture(t)
	f.proxy.mu.Lock()
	answer := f.proxy.answer
	f.proxy.mu.Unlock()
	f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, failure := answer(method, raw)
		if method == nodepolicy.MethodDispatchResident {
			return result, holdSuccessReply
		}
		return result, failure
	})
	done := f.startLoadedAttempt(t)
	f.proxy.waitFor("held final grant", func(calls []policyCall) bool {
		return len(modelConfigCalls(f.proxy, nodepolicy.MethodDispatchResident)) == 1
	})
	replacement := attachFakeEngine(t, f.b)
	f.proxy.releaseHeld()
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	if len(modelConfigCalls(replacement, residentDispatchReplyMethod)) != 0 || len(modelConfigCalls(replacement, nodepolicy.MethodReconcileResident)) != 0 {
		t.Fatal("old grant or load was delivered to replacement manager")
	}
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].State == nodepolicy.ModelStateSatisfied {
		t.Fatal("late grant satisfied replacement source")
	}
}

func TestModelConfigDispatchRejectsForeignAndDuplicateRequests(t *testing.T) {
	f := newModelConfigFixture(t)
	e := f.b.captureModelConfig()
	_, _, id, _ := f.b.modelConfigEntry(e, nodepolicy.Role{Engine: "llamacpp", Model: "resident"})
	op := &modelConfigAttempt{ctx: context.Background(), evidence: e, reservation: "fixture-reservation", deadline: time.Now().Add(time.Second),
		prepare: nodepolicy.PrepareResidentParams{PolicyRevision: e.Policy.Revision, ProxyRunID: e.Run,
			ManagerIncarnation: "fixture-manager", OperationID: "resident-1", Engine: "llamacpp", Model: "resident", Identity: id}}
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.models.active = op
	r.mu.Unlock()
	p := nodepolicy.DispatchResidentParams{PolicyRevision: e.Policy.Revision, ProxyRunID: e.Run, ManagerIncarnation: "fixture-manager",
		OperationID: "resident-1", Engine: "llamacpp", Model: "resident", Identity: id, ReservationID: "fixture-reservation"}
	encode := func(p nodepolicy.DispatchResidentParams) json.RawMessage {
		t.Helper()
		raw, err := json.Marshal(struct {
			ID     string                            `json:"id"`
			Params nodepolicy.DispatchResidentParams `json:"params"`
		}{"fixture-dispatch", p})
		if err != nil {
			t.Fatal(err)
		}
		return raw
	}
	foreign := p
	foreign.PolicyRevision = "999"
	f.b.dispatchResidentFrom(f.source, encode(foreign))
	foreign = p
	foreign.Identity.EngineRunID = "foreign-run"
	f.b.dispatchResidentFrom(f.source, encode(foreign))
	f.b.dispatchResidentFrom(&engineManagerSource{}, encode(p))
	f.b.dispatchResidentFrom(f.source, json.RawMessage(`{"id":"bad","params":null}`))
	r.mu.Lock()
	foreignStarted := op.grantStarted
	r.mu.Unlock()
	if foreignStarted {
		t.Fatal("foreign or malformed dispatch request claimed the active operation")
	}
	if len(modelConfigCalls(f.proxy, nodepolicy.MethodDispatchResident)) != 0 {
		t.Fatal("foreign or malformed grant request reached proxy")
	}
	f.b.dispatchResidentFrom(f.source, encode(p))
	f.b.dispatchResidentFrom(f.source, encode(p))
	f.awaitGrant(t)
	if got := len(modelConfigCalls(f.proxy, nodepolicy.MethodDispatchResident)); got != 1 {
		t.Fatalf("duplicate request produced %d final grants", got)
	}
}

func TestModelConfigProxyRefusalDoesNotReachManager(t *testing.T) {
	f := newModelConfigFixture(t)
	f.proxy.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == nodepolicy.MethodPrepareResident {
			return nodepolicy.PrepareResidentResult{EvidenceGeneration: 7, Refusals: []nodepolicy.ResidentRefusal{{Engine: "llamacpp", Model: "resident", Reason: nodepolicy.ModelReasonNoFit}}}, ""
		}
		return map[string]bool{"ok": true}, ""
	})
	f.b.reconcileModelConfig(context.Background())
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 || status.Models[0].State != nodepolicy.ModelStateBlocked || status.Models[0].Reason != nodepolicy.ModelReasonNoFit {
		t.Fatalf("proxy capacity refusal lost bounded status or reached manager: %+v", status)
	}
}

func TestModelConfigRecoveryRequiresTerminalManagerBarrier(t *testing.T) {
	for _, terminal := range []bool{false, true} {
		name := "transport failure retains uncertain hold"
		if terminal {
			name = "terminal response permits fresh empty observation"
		}
		t.Run(name, func(t *testing.T) {
			f := newModelConfigFixture(t)
			f.proxy.mu.Lock()
			answer := f.proxy.answer
			f.proxy.mu.Unlock()
			f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == nodepolicy.MethodFinishResident {
					return nodepolicy.FinishResidentResult{ReservationID: "fixture-reservation", Outcome: nodepolicy.ResidentUnknown, RecoveryID: "fixture-recovery"}, ""
				}
				if method == nodepolicy.MethodObserveResident {
					return nodepolicy.ObserveResidentResult{ReservationID: "fixture-reservation", Settled: true}, ""
				}
				return answer(method, raw)
			})
			f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
				switch method {
				case settings.SwitchPrepareMethod:
					return emptySwitchManagerSnapshot(), ""
				case nodepolicy.MethodReconcileResident:
					if !terminal {
						return nil, "fixture lost terminal response"
					}
					return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentUnknown}, ""
				case "engine:models":
					return map[string]any{"loadedByEngine": map[string][]string{"llamacpp": {}}}, ""
				}
				return map[string]bool{"ok": true}, ""
			})
			f.b.reconcileModelConfig(context.Background())
			observations := modelConfigCalls(f.proxy, nodepolicy.MethodObserveResident)
			if !terminal {
				if len(observations) != 0 || len(modelConfigCalls(f.engine, "engine:models")) != 0 {
					t.Fatal("missing terminal barrier performed recovery observation")
				}
				return
			}
			if len(observations) != 1 || len(modelConfigCalls(f.engine, "engine:models")) != 1 {
				t.Fatal("terminal manager result did not require one fresh observation")
			}
			p := decodeModelConfigCall[nodepolicy.ObserveResidentParams](t, observations[0])
			if !p.OperationComplete || p.Observation.Loaded || p.Sequence != 1 || p.Observation.ManagerIncarnation != "fixture-manager" || p.RecoveryID != "fixture-recovery" {
				t.Fatalf("recovery did not bind completion and fresh empty evidence: %+v", p)
			}
		})
	}
}

func TestModelConfigAuthorityChangesDuringPreparePreventLoad(t *testing.T) {
	cases := []struct {
		name   string
		change func(*modelConfigFixture)
	}{
		{"proxy replacement", func(f *modelConfigFixture) {
			attachAckingProxy(t, f.b)
		}},
		{"local pause", func(f *modelConfigFixture) {
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			r.localPauseRequested = true
			r.mu.Unlock()
		}},
		{"local Off", func(f *modelConfigFixture) {
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			r.intent["llamacpp"] = false
			r.mu.Unlock()
		}},
		{"profile revision changes", func(f *modelConfigFixture) {
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			r.current.Revision = "1"
			r.current.Profiles[0].ContextTokens = 4096
			r.mu.Unlock()
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newModelConfigFixture(t)
			f.proxy.mu.Lock()
			answer := f.proxy.answer
			f.proxy.mu.Unlock()
			f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				result, failure := answer(method, raw)
				if method == nodepolicy.MethodPrepareResident {
					return result, holdSuccessReply
				}
				return result, failure
			})
			done := make(chan struct{})
			go func() { defer close(done); f.b.reconcileModelConfig(context.Background()) }()
			f.proxy.waitFor("held preparation", func(calls []policyCall) bool {
				return len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)) == 1
			})
			tc.change(f)
			f.proxy.releaseHeld()
			awaitModelConfigDone(t, done)
			if len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 {
				t.Fatal("authority changed during prepare but manager load was still requested")
			}
			if len(modelConfigCalls(f.proxy, nodepolicy.MethodFinishResident)) != 1 {
				t.Fatal("revoked preparation was not finished on original proxy")
			}
		})
	}
}

func TestModelConfigAuthorityChangesDuringGrantLoadAndFinish(t *testing.T) {
	for _, phase := range []string{"grant", "load", "finish"} {
		for _, change := range []string{"proxy replacement", "local pause", "local Off"} {
			t.Run(phase+"/"+change, func(t *testing.T) {
				f := newModelConfigFixture(t)
				f.proxy.mu.Lock()
				answer := f.proxy.answer
				f.proxy.mu.Unlock()
				heldMethod := ""
				if phase == "grant" {
					heldMethod = nodepolicy.MethodDispatchResident
				}
				if phase == "finish" {
					heldMethod = nodepolicy.MethodFinishResident
				}
				f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
					result, failure := answer(method, raw)
					if method == heldMethod {
						return result, holdSuccessReply
					}
					return result, failure
				})
				done := f.startLoadedAttempt(t)
				if phase != "grant" {
					f.awaitGrant(t)
				}
				if phase == "finish" {
					f.engine.releaseHeld()
				}
				if heldMethod != "" {
					f.proxy.waitFor("held "+phase, func(calls []policyCall) bool {
						return len(modelConfigCalls(f.proxy, heldMethod)) == 1
					})
				}
				var replacement *fakeWorker
				switch change {
				case "proxy replacement":
					_, replacement, _ = attachAckingProxy(t, f.b)
				case "local pause":
					r := f.b.nodePolicyRuntime()
					r.mu.Lock()
					r.localPauseRequested = true
					r.mu.Unlock()
				case "local Off":
					r := f.b.nodePolicyRuntime()
					r.mu.Lock()
					r.intent["llamacpp"] = false
					r.mu.Unlock()
				}
				f.proxy.releaseHeld()
				f.engine.releaseHeld()
				awaitModelConfigDone(t, done)
				status := f.b.modelConfiguration(f.b.captureModelConfig())
				if status.Models[0].State == nodepolicy.ModelStateSatisfied {
					t.Fatalf("%s during %s retained satisfied state", change, phase)
				}
				if replacement != nil && (len(modelConfigCalls(replacement, nodepolicy.MethodDispatchResident)) != 0 || len(modelConfigCalls(replacement, nodepolicy.MethodFinishResident)) != 0) {
					t.Fatal("replacement proxy inherited grant or finish for original reservation")
				}
			})
		}
	}
}

func TestModelConfigAbsolutePolicyDeadlineBoundsLoad(t *testing.T) {
	f := newModelConfigFixture(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Admission.SwitchTimeoutSeconds = 1
	r.mu.Unlock()
	preparedDeadline := make(chan time.Time, 1)
	f.proxy.mu.Lock()
	answer := f.proxy.answer
	f.proxy.mu.Unlock()
	f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodPrepareResident {
			r.mu.Lock()
			preparedDeadline <- r.models.active.deadline
			r.mu.Unlock()
		}
		return answer(method, raw)
	})
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	r.mu.Lock()
	op := r.models.active
	deadline := op.deadline
	r.mu.Unlock()
	if deadline.After(time.Now().Add(time.Second)) || !deadline.Equal(<-preparedDeadline) {
		t.Fatal("load extended preparation's absolute policy deadline")
	}
	params := decodeModelConfigCall[nodepolicy.ReconcileResidentParams](t, modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)[0])
	if params.TimeoutMs <= 0 || params.TimeoutMs > 1000 {
		t.Fatalf("manager timeout=%d, want remaining policy budget", params.TimeoutMs)
	}
	// The terminal load reply stays held. The original deadline must finish the
	// attempt without acquiring a new load budget or inferring loaded evidence.
	awaitModelConfigDone(t, done)
	if time.Now().Before(deadline) {
		t.Fatal("held load completed before its absolute deadline")
	}
	finish := decodeModelConfigCall[nodepolicy.FinishResidentParams](t, modelConfigCalls(f.proxy, nodepolicy.MethodFinishResident)[0])
	if finish.Outcome != nodepolicy.ResidentUnknown {
		t.Fatalf("expired load finished as %q, want unknown", finish.Outcome)
	}
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].State == nodepolicy.ModelStateSatisfied {
		t.Fatal("expired load published satisfaction")
	}
	f.engine.releaseHeld()
}

func TestModelConfigInflightStatusRetainsEveryDesiredReference(t *testing.T) {
	f := newModelConfigFixture(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	second := r.current.Profiles[0]
	second.Name, second.Model = "second", "second"
	r.current.Profiles = append(r.current.Profiles, second)
	r.current.ResidentModels = append(r.current.ResidentModels, nodepolicy.Role{Engine: "llamacpp", Model: "second"})
	r.current.ServedModels = append(r.current.ServedModels, nodepolicy.Role{Engine: "llamacpp", Model: "second"})
	r.mu.Unlock()
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if len(status.Models) != 2 || status.Models[0].Model != "resident" || status.Models[1].Model != "second" || !status.Models[0].DesiredResident || !status.Models[1].DesiredResident {
		t.Fatalf("in-flight status dropped or reordered desired references: %+v", status)
	}
	p := decodeModelConfigCall[nodepolicy.ReconcileResidentParams](t, modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)[0])
	if p.DeclaredResidentCount != 2 {
		t.Fatalf("revision's two-model resident set sent count=%d", p.DeclaredResidentCount)
	}
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
}

func TestModelConfigCapacityEvidenceRetriesOneRefusedAttempt(t *testing.T) {
	f := newModelConfigFixture(t)
	f.proxy.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == nodepolicy.MethodPrepareResident {
			return nodepolicy.PrepareResidentResult{EvidenceGeneration: 3, Refusals: []nodepolicy.ResidentRefusal{{Engine: "llamacpp", Model: "resident", Reason: nodepolicy.ModelReasonNoFit}}}, ""
		}
		return map[string]bool{"ok": true}, ""
	})
	f.b.reconcileModelConfig(context.Background())
	f.b.reconcileModelConfig(context.Background())
	if len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)) != 1 {
		t.Fatal("unchanged capacity retried refusal")
	}
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.admission.ActiveByEngine = map[string]int{"llamacpp": 1}
	r.mu.Unlock()
	f.b.reconcileModelConfig(context.Background())
	f.b.reconcileModelConfig(context.Background())
	if len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)) != 2 {
		t.Fatal("changed capacity did not allow exactly one retry")
	}
	if len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 {
		t.Fatal("capacity refusals reached manager load")
	}
}

func TestModelConfigProxyReplacementRequiresFreshInventory(t *testing.T) {
	f := newModelConfigFixture(t)
	f.proxy.mu.Lock()
	answer := f.proxy.answer
	f.proxy.mu.Unlock()
	pp, proxy, acks := attachAckingProxy(t, f.b)
	if !f.b.replayPolicyToProxy(context.Background(), pp) {
		t.Fatal("replacement proxy did not activate current policy")
	}
	proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetPolicy {
			return acks.answer(method, raw)
		}
		return answer(method, raw)
	})
	reportStrict(t, f.b, "0", acks.runID, strictCapability(t, "llamacpp", "resident"))
	e := f.b.captureModelConfig()
	if e.Activation.Activation != activationActive || e.Readiness.Capability == nil {
		t.Fatal("replacement fixture lacks fresh activation and strict readiness")
	}
	if e.Loaded != nil {
		t.Fatal("replacement proxy inherited old proxy's loaded inventory")
	}
	f.b.reconcileModelConfig(context.Background())
	if len(modelConfigCalls(proxy, nodepolicy.MethodPrepareResident)) != 0 || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 {
		t.Fatal("replacement activation/readiness authorized a load before fresh inventory")
	}
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].ObservedLoaded != nil || status.Models[0].State != nodepolicy.ModelStateUnknown {
		t.Fatalf("replacement without fresh inventory status=%+v", status)
	}

	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {}})
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	loadedProxy := r.models.loadedProxy
	r.mu.Unlock()
	if loadedProxy != pp {
		t.Fatal("fresh source-paired observation was not bound to replacement proxy")
	}
	f.b.reconcileModelConfig(context.Background())
	if len(modelConfigCalls(proxy, nodepolicy.MethodPrepareResident)) != 1 || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
		t.Fatal("fresh replacement-bound inventory did not permit one resident attempt")
	}
}

func TestModelConfigConcurrentEvidenceChangeIsReevaluated(t *testing.T) {
	cases := []struct {
		name   string
		change func(*modelConfigFixture)
	}{
		{"inventory", func(f *modelConfigFixture) {
			f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {}})
		}},
		{"capacity", func(f *modelConfigFixture) {
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			r.admission.ActiveByEngine = map[string]int{"llamacpp": 1}
			r.mu.Unlock()
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newModelConfigFixture(t)
			f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"neighbor"}})
			// Failed attempts must reevaluate changed evidence; successful loads
			// instead wait for observation without another dispatch.
			done := f.startResidentAttempt(t, nodepolicy.ResidentUnknown)
			f.awaitGrant(t)
			tc.change(f)
			f.engine.releaseHeld()
			awaitModelConfigDone(t, done)
			status := f.b.modelConfiguration(f.b.captureModelConfig())
			if status.Models[0].State == nodepolicy.ModelStateSatisfied {
				t.Fatalf("completion retained satisfaction after concurrent %s change: %+v", tc.name, status)
			}
			if tc.name == "inventory" && (status.Models[0].ObservedLoaded == nil || *status.Models[0].ObservedLoaded) {
				t.Fatalf("concurrent empty inventory lost its observation: %+v", status)
			}
			f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
				if method == settings.SwitchPrepareMethod {
					return emptySwitchManagerSnapshot(), ""
				}
				return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
			})
			f.b.reconcileModelConfig(context.Background())
			f.b.reconcileModelConfig(context.Background())
			if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); got != 2 {
				t.Fatalf("concurrent %s change permitted %d attempts, want initial and one retry", tc.name, got)
			}
		})
	}
}

func TestModelConfigNotificationWithoutLoadedInventoryClearsKnowledge(t *testing.T) {
	f := newModelConfigFixture(t)
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.b.handlePolicyEngineNotification(f.source, engineModelsChangedMethod, json.RawMessage(`{"ok":true}`))
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].ObservedLoaded == nil || !*status.Models[0].ObservedLoaded {
		t.Fatalf("malformed notification changed known inventory: %+v", status)
	}
	f.b.handlePolicyEngineNotification(f.source, engineModelsChangedMethod, json.RawMessage(`{"models":{"models":[]}}`))
	status = f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].ObservedLoaded != nil || status.Models[0].State != nodepolicy.ModelStateUnknown {
		t.Fatalf("valid snapshot without loaded inventory retained knowledge: %+v", status)
	}
	f.b.reconcileModelConfig(context.Background())
	if len(modelConfigCalls(f.proxy, nodepolicy.MethodPrepareResident)) != 0 || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 0 {
		t.Fatal("unknown inventory notification permitted resident load")
	}
	f.b.handlePolicyEngineNotification(f.source, engineModelsChangedMethod, json.RawMessage(`{"models":{"models":[],"loadedByEngine":{"llamacpp":[]}}}`))
	status = f.b.modelConfiguration(f.b.captureModelConfig())
	if status.Models[0].ObservedLoaded == nil || *status.Models[0].ObservedLoaded {
		t.Fatalf("explicit empty inventory was not preserved: %+v", status)
	}
	f.b.reconcileModelConfig(context.Background())
	if len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
		t.Fatal("explicit empty inventory did not permit resident attempt")
	}
}

func modelConfigNotDispatched(readback *settings.LaunchState) func(string, json.RawMessage) (any, string) {
	return func(method string, _ json.RawMessage) (any, string) {
		switch method {
		case settings.SwitchPrepareMethod:
			return emptySwitchManagerSnapshot(), ""
		case "engine:get-launch":
			if readback != nil {
				return *readback, ""
			}
		}
		return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, ""
	}
}

func TestModelConfigEnvelopeChangeThenInventoryReloads(t *testing.T) {
	f := newModelConfigFixture(t)
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.b.reconcileModelConfig(context.Background())
	old := residentLaunchReadback(f.b.captureModelConfig())
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Revision = "1"
	r.current.Profiles = append([]nodepolicy.Profile(nil), r.current.Profiles...)
	r.current.Profiles[0].ContextTokens = 8193
	id := r.engineIdentity["llamacpp"]
	env := id.Serving["resident"]
	env.PerSlotContextTokens, env.TotalContextTokens = 8193, 8193
	id.Serving = map[string]engines.ServingEnvelope{"resident": env}
	r.engineIdentity["llamacpp"] = id
	r.mu.Unlock()
	if !f.b.replayPolicyToProxy(context.Background(), f.pp) {
		t.Fatal("activate changed resident policy")
	}
	reportStrict(t, f.b, "1", f.acks.runID, strictCapability(t, "llamacpp", "resident"))
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.engine.setAnswer(modelConfigNotDispatched(&old))
	f.b.reconcileModelConfig(context.Background())
	if loads := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); loads != 2 {
		t.Fatalf("changed request after inventory produced %d loads, want 2; readbacks=%d", loads, len(modelConfigCalls(f.engine, "engine:get-launch")))
	}
}

func TestModelConfigTerminalLoadedReplyPreventsTickReload(t *testing.T) {
	f := newModelConfigFixture(t)
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	declared := r.current.ResidentModels
	r.current.ResidentModels = nil
	r.mu.Unlock()
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	r.mu.Lock()
	r.current.ResidentModels = declared
	r.mu.Unlock()
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	f.engine.setAnswer(modelConfigNotDispatched(nil))
	for range 2 {
		r.mu.Lock()
		now := r.now().Add(idleCheckInterval + time.Second)
		r.now = func() time.Time { return now }
		r.mu.Unlock()
		f.b.reconcileModelConfig(context.Background())
	}
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if loads := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); loads != 1 || status.Models[0].State != nodepolicy.ModelStateSatisfied {
		t.Fatalf("terminal loaded reply did not prevent tick reload: loads=%d status=%+v", loads, status)
	}
}

func TestModelConfigObservedUnloadDropsRecordedRequest(t *testing.T) {
	f := newModelConfigFixture(t)
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.b.reconcileModelConfig(context.Background())
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {}})
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Revision = "1"
	r.current.Profiles = append([]nodepolicy.Profile(nil), r.current.Profiles...)
	r.current.Profiles[0].ContextTokens = 8193
	id := r.engineIdentity["llamacpp"]
	env := id.Serving["resident"]
	env.PerSlotContextTokens, env.TotalContextTokens = 8193, 8193
	id.Serving = map[string]engines.ServingEnvelope{"resident": env}
	r.engineIdentity["llamacpp"] = id
	r.mu.Unlock()
	if !f.b.replayPolicyToProxy(context.Background(), f.pp) {
		t.Fatal("activate changed resident policy")
	}
	reportStrict(t, f.b, "1", f.acks.runID, strictCapability(t, "llamacpp", "resident"))
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	readback := residentLaunchReadback(f.b.captureModelConfig())
	f.engine.setAnswer(modelConfigNotDispatched(&readback))
	f.b.reconcileModelConfig(context.Background())
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if loads := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); loads != 1 || status.Models[0].State != nodepolicy.ModelStateSatisfied {
		t.Fatalf("observed unload caused extra load: loads=%d status=%+v", loads, status)
	}
}

// completedResidentLoad finishes one correlated load and leaves its pending
// observation open; later manager calls answer not-dispatched.
func (f *modelConfigFixture) completedResidentLoad(t *testing.T) {
	t.Helper()
	done := f.startLoadedAttempt(t)
	f.awaitGrant(t)
	f.engine.releaseHeld()
	awaitModelConfigDone(t, done)
	f.engine.setAnswer(modelConfigNotDispatched(nil))
}

func (f *modelConfigFixture) residentEntry(t *testing.T) nodepolicy.ModelConfigurationEntry {
	t.Helper()
	status := f.b.modelConfiguration(f.b.captureModelConfig())
	if len(status.Models) != 1 {
		t.Fatalf("status models=%+v, want one resident", status.Models)
	}
	return status.Models[0]
}

func TestModelConfigAbsentEngineKeepsConfirmedProof(t *testing.T) {
	f := newModelConfigFixture(t)
	f.completedResidentLoad(t)
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.b.reconcileModelConfig(context.Background())
	if entry := f.residentEntry(t); entry.State != nodepolicy.ModelStateSatisfied {
		t.Fatalf("confirmed resident: %+v", entry)
	}
	// An engine absent from a publication has unknown, not empty, inventory.
	f.b.observeLoadedFrom(f.source, map[string][]string{})
	f.b.reconcileModelConfig(context.Background())
	if entry := f.residentEntry(t); entry.State != nodepolicy.ModelStateUnknown || entry.ObservedLoaded != nil {
		t.Fatalf("absent engine inventory: %+v", entry)
	}
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	_, confirmed := r.models.confirmed[nodepolicy.Role{Engine: "llamacpp", Model: "resident"}]
	r.mu.Unlock()
	if !confirmed {
		t.Fatal("unknown inventory deleted the confirmed proof")
	}
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.b.reconcileModelConfig(context.Background())
	entry := f.residentEntry(t)
	if readbacks, loads := len(modelConfigCalls(f.engine, "engine:get-launch")), len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); entry.State != nodepolicy.ModelStateSatisfied || readbacks != 0 || loads != 1 {
		t.Fatalf("returning inventory: entry=%+v readbacks=%d loads=%d, want satisfied by retained proof", entry, readbacks, loads)
	}
}

func TestModelConfigContradictingPendingObservationIsUnknown(t *testing.T) {
	ref := nodepolicy.Role{Engine: "llamacpp", Model: "resident"}
	test := func(name string, inventory []string) {
		t.Run(name, func(t *testing.T) {
			f := newModelConfigFixture(t)
			f.completedResidentLoad(t)
			if entry := f.residentEntry(t); entry.State != nodepolicy.ModelStateSatisfied {
				t.Fatalf("correlated load before any later inventory: %+v", entry)
			}
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			window := r.models.pending[ref]
			r.mu.Unlock()
			if before, known := window.evidence.Loaded["llamacpp"]; !known || len(before) != 0 {
				t.Fatalf("inventory before the load=%v known=%v, want known empty", before, known)
			}
			for range 2 {
				f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": inventory})
				f.b.reconcileModelConfig(context.Background())
				entry := f.residentEntry(t)
				if entry.State != nodepolicy.ModelStateUnknown || entry.Reason != nodepolicy.ModelReasonObservationUnknown || entry.ObservedLoaded == nil || *entry.ObservedLoaded {
					t.Fatalf("contradicting inventory during pending observation published %+v, want unknown observation with loaded=false", entry)
				}
			}
			r.mu.Lock()
			pending, ok := r.models.pending[ref]
			r.mu.Unlock()
			if loads := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); loads != 1 || !ok {
				t.Fatalf("contradiction loads=%d pending=%v, want the original load still pending", loads, ok)
			}
			if !pending.expires.Equal(window.expires) {
				t.Fatalf("contradiction moved the observation deadline to %s, want %s", pending.expires, window.expires)
			}
			f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
			f.b.reconcileModelConfig(context.Background())
			if entry := f.residentEntry(t); entry.State != nodepolicy.ModelStateSatisfied || len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) != 1 {
				t.Fatalf("confirming inventory after contradiction: %+v", entry)
			}
		})
	}
	test("other model", []string{"other"})
	test("repeated empty", []string{})
}

func TestModelConfigPendingObservationIgnoresUnprovenInventory(t *testing.T) {
	ref := nodepolicy.Role{Engine: "llamacpp", Model: "resident"}
	t.Run("retired source", func(t *testing.T) {
		f := newModelConfigFixture(t)
		f.completedResidentLoad(t)
		f.b.observeLoadedFrom(&engineManagerSource{}, map[string][]string{"llamacpp": {}})
		f.b.reconcileModelConfig(context.Background())
		if entry := f.residentEntry(t); entry.State != nodepolicy.ModelStateSatisfied {
			t.Fatalf("retired-source inventory changed the correlated load: %+v", entry)
		}
		f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {}})
		f.b.reconcileModelConfig(context.Background())
		if entry := f.residentEntry(t); entry.State != nodepolicy.ModelStateUnknown || entry.Reason != nodepolicy.ModelReasonObservationUnknown {
			t.Fatalf("current-source inventory after a retired one published %+v, want unknown observation", entry)
		}
	})
	t.Run("unknown inventory", func(t *testing.T) {
		f := newModelConfigFixture(t)
		f.completedResidentLoad(t)
		f.b.observeLoadedFrom(f.source, map[string][]string{})
		f.b.reconcileModelConfig(context.Background())
		entry := f.residentEntry(t)
		r := f.b.nodePolicyRuntime()
		r.mu.Lock()
		_, pending := r.models.pending[ref]
		r.mu.Unlock()
		if loads := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); entry.State != nodepolicy.ModelStateUnknown || entry.ObservedLoaded != nil || !pending || loads != 1 {
			t.Fatalf("unknown inventory: entry=%+v pending=%v loads=%d, want unknown without observed absence, still pending, one load", entry, pending, loads)
		}
	})
}

func TestModelConfigRetirementDropsRecordedLoadRequest(t *testing.T) {
	for _, scenario := range []string{"removal", "Off", "pause"} {
		t.Run(scenario, func(t *testing.T) {
			f := newModelConfigFixture(t)
			f.completedResidentLoad(t)
			r := f.b.nodePolicyRuntime()
			ref := nodepolicy.Role{Engine: "llamacpp", Model: "resident"}
			r.mu.Lock()
			_, recorded := r.models.loadedRequests[ref]
			switch scenario {
			case "removal":
				r.current.ResidentModels = nil
			case "Off":
				r.intent["llamacpp"] = false
			default:
				r.localPauseRequested = true
			}
			r.mu.Unlock()
			if !recorded {
				t.Fatal("completed load did not record its request")
			}
			f.b.reconcileModelConfig(context.Background())
			r.mu.Lock()
			_, recorded = r.models.loadedRequests[ref]
			r.mu.Unlock()
			if recorded {
				t.Fatalf("%s retained the recorded load request", scenario)
			}
		})
	}
}

// nonterminalHold leaves a granted load whose completion never decoded and
// whose finish cannot reach the proxy: the original nonterminal barrier.
func (f *modelConfigFixture) nonterminalHold(t *testing.T) residentUnsettled {
	t.Helper()
	f.proxy.mu.Lock()
	answer := f.proxy.answer
	f.proxy.mu.Unlock()
	f.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodFinishResident {
			return nil, "finish unavailable"
		}
		return answer(method, raw)
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done, _, _ := startModelConfigMatrixAttempt(t, ctx, f, emptySwitchManagerSnapshot(), nodepolicy.ResidentUnknown, true)
	f.awaitGrant(t)
	cancel()
	awaitModelConfigDone(t, done)
	f.engine.releaseHeld()
	f.engine.setAnswer(modelConfigNotDispatched(nil))
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	hold, held := r.models.unsettled[nodepolicy.Role{Engine: "llamacpp", Model: "resident"}]
	r.mu.Unlock()
	if !held || hold.terminal {
		t.Fatal("missing original nonterminal granted uncertainty")
	}
	return hold
}

func TestModelConfigNonterminalHoldConfirmsReplacementReadback(t *testing.T) {
	f := newModelConfigFixture(t)
	original := f.nonterminalHold(t)
	r := f.b.nodePolicyRuntime()
	ref := nodepolicy.Role{Engine: "llamacpp", Model: "resident"}
	// The original allocation may still be loading: no readback replaces it.
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	f.b.reconcileModelConfig(context.Background())
	if entry := f.residentEntry(t); entry.State != nodepolicy.ModelStateUnknown || entry.Reason != nodepolicy.ModelReasonOperationTimeout || len(modelConfigCalls(f.engine, "engine:get-launch")) != 0 {
		t.Fatalf("original-allocation inventory confirmed a nonterminal hold: %+v", entry)
	}
	r.mu.Lock()
	identity := r.engineIdentity
	r.mu.Unlock()
	f.source = f.b.activateEngineManagerSource()
	f.b.bindEngineMgr(f.b.getEngineMgr(), f.source)
	r.mu.Lock()
	r.engineIdentity = identity
	r.mu.Unlock()
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
	readback := residentLaunchReadback(f.b.captureModelConfig())
	f.engine.setAnswer(modelConfigNotDispatched(&readback))
	assertBarrier := func(stage string, want nodepolicy.ModelConfigurationState) {
		t.Helper()
		f.b.reconcileModelConfig(context.Background())
		r.mu.Lock()
		retained, held := r.models.unsettled[ref]
		r.mu.Unlock()
		entry := f.residentEntry(t)
		if loads := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)); entry.State != want || !held || retained.op != original.op || loads != 1 {
			t.Fatalf("%s: entry=%+v held=%v loads=%d, want %s with the original barrier and one load", stage, entry, held, loads, want)
		}
	}
	assertBarrier("replacement readback", nodepolicy.ModelStateSatisfied)
	if len(modelConfigCalls(f.engine, "engine:get-launch")) == 0 {
		t.Fatal("replacement confirmation did not read the observed envelope")
	}
	f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {}})
	assertBarrier("later known-empty inventory", nodepolicy.ModelStateUnknown)
}

func TestModelConfigUnchangedStatusDoesNotNotify(t *testing.T) {
	for _, scenario := range []string{"satisfied", "blocked", "uncertain hold", "failed attempt"} {
		t.Run(scenario, func(t *testing.T) {
			f := newModelConfigFixture(t)
			r := f.b.nodePolicyRuntime()
			want := nodepolicy.ModelStateBlocked
			switch scenario {
			case "satisfied":
				want = nodepolicy.ModelStateSatisfied
				f.completedResidentLoad(t)
				f.b.observeLoadedFrom(f.source, map[string][]string{"llamacpp": {"resident"}})
			case "blocked":
				r.mu.Lock()
				r.intent["llamacpp"] = false
				r.mu.Unlock()
			case "failed attempt":
				// Each pass publishes pending, then the manager's not-dispatched reply.
			default:
				want = nodepolicy.ModelStateUnknown
				f.nonterminalHold(t)
			}
			tick := func() {
				r.mu.Lock()
				now := r.now().Add(idleCheckInterval + time.Second)
				r.now = func() time.Time { return now }
				r.mu.Unlock()
				f.b.reconcileModelConfig(context.Background())
			}
			tick()
			if entry := f.residentEntry(t); entry.State != want {
				t.Fatalf("steady state %+v, want %s", entry, want)
			}
			changed := func() int {
				n := 0
				for _, frame := range f.out.frames() {
					if frame.Method == notifyPolicyChanged {
						n++
					}
				}
				return n
			}
			before := changed()
			generation := f.b.modelConfiguration(f.b.captureModelConfig()).ObservationGeneration
			loads := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident))
			for range 3 {
				tick()
			}
			if after := changed(); after != before {
				t.Fatalf("unchanged status announced policy:changed %d times", after-before)
			}
			if got := f.b.modelConfiguration(f.b.captureModelConfig()).ObservationGeneration; got != generation+3 {
				t.Fatalf("observationGeneration=%d after three passes from %d", got, generation)
			}
			if got := len(modelConfigCalls(f.engine, nodepolicy.MethodReconcileResident)) - loads; scenario == "failed attempt" && got != 3 {
				t.Fatalf("failed attempt retried %d times in three passes, want 3", got)
			}
			r.mu.Lock()
			r.localPauseRequested = true
			r.mu.Unlock()
			tick()
			if changed() == before {
				t.Fatal("changed status was not announced")
			}
		})
	}
}
