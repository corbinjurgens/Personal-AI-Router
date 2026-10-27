// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

type stopPauseExecutionManager struct {
	mu        sync.Mutex
	states    map[string]pauseEngineStatus
	hold      string
	holdFirst bool
	failure   string
}

func stopPauseExecutionFixture(t *testing.T, names ...string) (*routedBroker, *stopPauseExecutionManager) {
	t.Helper()
	rb := newRoutedBroker(t)
	m := &stopPauseExecutionManager{states: make(map[string]pauseEngineStatus)}
	for _, name := range names {
		m.states[name] = pauseEngineStatus{Running: true}
	}
	rb.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		m.mu.Lock()
		defer m.mu.Unlock()
		switch method {
		case "engine:switch-prepare":
			return emptySwitchManagerSnapshot(), ""
		case "engine:get-installed":
			rows := make([]map[string]any, 0, len(m.states))
			for name, status := range m.states {
				rows = append(rows, map[string]any{"engine": name, "running": status.Running, "terminationVerified": status.TerminationVerified})
			}
			return map[string]any{"engines": rows}, ""
		case engineSleepMethod:
			var p struct {
				Engine string `json:"engine"`
			}
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if m.failure != "" {
				return nil, m.failure
			}
			m.states[p.Engine] = pauseEngineStatus{TerminationVerified: true}
			reply := map[string]any{"engine": p.Engine, "running": false, "terminationVerified": true}
			if m.holdFirst && m.hold == "" {
				m.hold = p.Engine
			}
			if p.Engine == m.hold {
				return reply, holdSuccessReply
			}
			return reply, ""
		case engineUnloadModelMethod:
			return nil, "broker must not unload before ownership admission"
		}
		return map[string]bool{"ok": true}, ""
	})
	t.Cleanup(func() {
		r := rb.b.nodePolicyRuntime()
		r.mu.Lock()
		if r.transitionCancel != nil {
			r.transitionCancel()
		}
		r.mu.Unlock()
	})
	return rb, m
}

type stopPauseExecutionOutcome struct {
	state nodepolicy.Availability
	err   error
}

func stopPauseExecutionImmediate(b *Broker) <-chan stopPauseExecutionOutcome {
	done := make(chan stopPauseExecutionOutcome, 1)
	go func() {
		r := b.nodePolicyRuntime()
		r.mu.Lock()
		state, err := b.setAvailabilityModeLocked(context.Background(), r, nodepolicy.Paused, nodepolicy.PausedLocally, nodepolicy.PauseImmediate)
		done <- stopPauseExecutionOutcome{state, err}
	}()
	return done
}

func stopPauseExecutionWait(t *testing.T, done <-chan stopPauseExecutionOutcome) stopPauseExecutionOutcome {
	t.Helper()
	select {
	case result := <-done:
		return result
	case <-time.After(5 * time.Second):
		t.Fatal("availability transition did not finish")
		return stopPauseExecutionOutcome{}
	}
}

func stopPauseExecutionRequireFence(t *testing.T, b *Broker, want nodepolicy.Availability) {
	t.Helper()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	live, stored, blocked := r.live, r.current.Availability, r.managementBlockedLocked()
	r.mu.Unlock()
	if live != want || stored != nodepolicy.Paused || !blocked {
		t.Fatalf("pause state: live=%s stored=%s blocked=%v; want %s/paused/true", live, stored, blocked, want)
	}
}

func TestStopPauseExecutionAcceptancePrecedesBarriers(t *testing.T) {
	test := func(name string, lock func(*Broker) func()) {
		t.Run(name, func(t *testing.T) {
			rb, _ := stopPauseExecutionFixture(t, "owned")
			unlock := sync.OnceFunc(lock(rb.b))
			defer unlock()
			done := stopPauseExecutionImmediate(rb.b)
			rb.proxy.waitFor("immediate cancellation before barrier release", func(calls []policyCall) bool {
				for _, p := range availabilityPushes(calls) {
					if p.State == nodepolicy.Draining && p.CancelActive {
						return true
					}
				}
				return false
			})
			stopPauseExecutionRequireFence(t, rb.b, nodepolicy.Draining)
			data, err := os.ReadFile(rb.path)
			if err != nil {
				t.Fatalf("read acceptance policy: %v", err)
			}
			var persisted nodepolicy.Policy
			if err := json.Unmarshal(data, &persisted); err != nil {
				t.Fatal(err)
			}
			if persisted.Availability != nodepolicy.Paused || persisted.PausedBy != nodepolicy.PausedLocally {
				t.Fatalf("acceptance was not durable: %+v", persisted)
			}
			select {
			case got := <-done:
				t.Fatalf("transition crossed held barrier: %+v", got)
			default:
			}
			unlock()
			if got := stopPauseExecutionWait(t, done); got.err != nil || got.state != nodepolicy.Paused {
				t.Fatalf("immediate pause: %+v", got)
			}
		})
	}
	test("configuration", func(b *Broker) func() { b.engineConfigMu.Lock(); return b.engineConfigMu.Unlock })
	test("earlier transition", func(b *Broker) func() {
		r := b.nodePolicyRuntime()
		r.transitionMu.Lock()
		return r.transitionMu.Unlock
	})
}

func TestStopPauseExecutionSkipsBothAdmissionWaits(t *testing.T) {
	test := func(name string, active nodepolicy.OnActive) {
		t.Run(name, func(t *testing.T) {
			rb, _ := stopPauseExecutionFixture(t, "owned")
			r := rb.b.nodePolicyRuntime()
			r.mu.Lock()
			r.current.Pause.OnActive = active
			r.current.Pause.DrainTimeoutSeconds = 3600
			r.haveAdmission = true
			r.admission.Active = 1
			r.mu.Unlock()
			if got := stopPauseExecutionWait(t, stopPauseExecutionImmediate(rb.b)); got.err != nil || got.state != nodepolicy.Paused {
				t.Fatalf("immediate with permanently active admission: %+v", got)
			}
			r.mu.Lock()
			stillActive := r.admission.Active
			r.mu.Unlock()
			if stillActive != 1 {
				t.Fatal("fixture admission was unexpectedly released")
			}
		})
	}
	test("configured finish", nodepolicy.FinishActive)
	test("configured cancellation", nodepolicy.CancelActive)
}

func TestStopPauseExecutionCallerDetachDoesNotAbandonStop(t *testing.T) {
	rb, manager := stopPauseExecutionFixture(t, "owned")
	manager.mu.Lock()
	manager.hold = "owned"
	manager.mu.Unlock()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan stopPauseExecutionOutcome, 1)
	go func() {
		r := rb.b.nodePolicyRuntime()
		r.mu.Lock()
		state, err := rb.b.setAvailabilityModeLocked(ctx, r, nodepolicy.Paused, nodepolicy.PausedLocally, nodepolicy.PauseImmediate)
		done <- stopPauseExecutionOutcome{state, err}
	}()
	rb.engine.waitFor("accepted stop before caller detach", func(calls []policyCall) bool {
		for _, call := range calls {
			if call.method == engineSleepMethod {
				return true
			}
		}
		return false
	})
	cancel()
	if got := stopPauseExecutionWait(t, done); !errors.Is(got.err, context.Canceled) {
		t.Fatalf("detached caller: %+v", got)
	}
	stopPauseExecutionRequireFence(t, rb.b, nodepolicy.Draining)
	rb.engine.releaseHeld()
	rb.proxy.waitFor("physical completion after caller detach", func(calls []policyCall) bool {
		for _, p := range availabilityPushes(calls) {
			if p.State == nodepolicy.Paused {
				return true
			}
		}
		return false
	})
	stopPauseExecutionRequireFence(t, rb.b, nodepolicy.Paused)
}

func TestStopPauseExecutionIndependentStops(t *testing.T) {
	rb, manager := stopPauseExecutionFixture(t, "a-slow", "z-other")
	manager.mu.Lock()
	manager.holdFirst = true
	manager.mu.Unlock()
	done := stopPauseExecutionImmediate(rb.b)
	rb.engine.waitFor("both engine stop requests while one reply is held", func(calls []policyCall) bool {
		seen := map[string]bool{}
		for _, call := range calls {
			if call.method == engineUnloadModelMethod {
				t.Fatal("broker unloaded a model before ownership admission")
			}
			if call.method == engineSleepMethod {
				var p struct {
					Engine string `json:"engine"`
				}
				if err := json.Unmarshal(call.params, &p); err != nil {
					t.Fatal(err)
				}
				seen[p.Engine] = true
			}
		}
		return seen["a-slow"] && seen["z-other"]
	})
	stopPauseExecutionRequireFence(t, rb.b, nodepolicy.Draining)
	select {
	case got := <-done:
		t.Fatalf("unacknowledged stop completed pause: %+v", got)
	default:
	}
	rb.engine.releaseHeld()
	if got := stopPauseExecutionWait(t, done); got.err != nil || got.state != nodepolicy.Paused {
		t.Fatalf("joined pause: %+v", got)
	}
}

func TestStopPauseExecutionConfiguredEscalation(t *testing.T) {
	rb, _ := stopPauseExecutionFixture(t, "owned")
	r := rb.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Pause.OnActive = nodepolicy.FinishActive
	r.current.Pause.DrainTimeoutSeconds = 3600
	r.haveAdmission = true
	r.admission.Active = 1
	r.mu.Unlock()
	configured := make(chan stopPauseExecutionOutcome, 1)
	go func() {
		state, err := rb.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally)
		configured <- stopPauseExecutionOutcome{state, err}
	}()
	rb.proxy.waitFor("configured finish drain", func(calls []policyCall) bool {
		for _, p := range availabilityPushes(calls) {
			if p.State == nodepolicy.Draining && !p.CancelActive {
				return true
			}
		}
		return false
	})
	if calls := rb.engine.methods(engineSleepMethod); len(calls) != 0 {
		t.Fatalf("configured finish skipped drain: %v", calls)
	}
	immediate := stopPauseExecutionImmediate(rb.b)
	if got := stopPauseExecutionWait(t, configured); !errors.Is(got.err, errAvailabilitySuperseded) {
		t.Fatalf("configured transition was not superseded: %+v", got)
	}
	if got := stopPauseExecutionWait(t, immediate); got.err != nil || got.state != nodepolicy.Paused {
		t.Fatalf("escalation: %+v", got)
	}
}

func TestStopPauseExecutionKnownStopsPrecedeInventoryBarrier(t *testing.T) {
	rb, manager := stopPauseExecutionFixture(t, "owned")
	r := rb.b.nodePolicyRuntime()
	r.mu.Lock()
	r.running["owned"] = true
	r.mu.Unlock()
	rb.engine.mu.Lock()
	base := rb.engine.answer
	rb.engine.mu.Unlock()
	var reads int
	stoppedReply := make(chan struct{})
	rb.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, errText := base(method, raw)
		if method == engineSleepMethod {
			close(stoppedReply)
		}
		if method == "engine:get-installed" {
			reads++
			if reads == 1 {
				return result, holdSuccessReply
			}
		}
		return result, errText
	})
	done := stopPauseExecutionImmediate(rb.b)
	rb.engine.waitFor("known engine stop while status reply is held", func(calls []policyCall) bool {
		for _, call := range calls {
			if call.method == engineSleepMethod {
				return true
			}
		}
		return false
	})
	select {
	case <-stoppedReply:
	case <-time.After(5 * time.Second):
		t.Fatal("known engine stop did not reach its reply")
	}
	manager.mu.Lock()
	stopped := !manager.states["owned"].Running
	manager.mu.Unlock()
	if !stopped {
		t.Fatal("known engine did not progress independently of status readback")
	}
	rb.engine.releaseHeld()
	if got := stopPauseExecutionWait(t, done); got.err != nil {
		t.Fatalf("pause after inventory barrier: %+v", got)
	}
}

func TestStopPauseExecutionRetryAndLocalResume(t *testing.T) {
	rb, manager := stopPauseExecutionFixture(t, "ollama")
	r := rb.b.nodePolicyRuntime()
	r.mu.Lock()
	role := nodepolicy.Role{Engine: "ollama", Model: "tiny"}
	r.current.Roles = map[string]nodepolicy.Role{"weak": role}
	r.current.ResidentModels = nodepolicy.ResidentModels{role}
	r.current.EngineBindings = nodepolicy.EngineBindings{role}
	r.intent["ollama"] = true
	r.loaded["ollama"] = []string{"tiny"}
	r.mu.Unlock()
	before := rb.b.localPolicy().Policy
	manager.mu.Lock()
	manager.failure = "external ownership is unresolved"
	manager.mu.Unlock()
	if got := stopPauseExecutionWait(t, stopPauseExecutionImmediate(rb.b)); got.err == nil || got.state != nodepolicy.Draining {
		t.Fatalf("incomplete pause: %+v", got)
	}
	stopPauseExecutionRequireFence(t, rb.b, nodepolicy.Draining)
	for _, state := range []nodepolicy.Availability{nodepolicy.Paused, nodepolicy.Available} {
		_, err := rb.b.setAvailabilityAs(context.Background(), state, nodepolicy.PausedByPeer, "pinned-peer")
		var veto *localVetoError
		if !errors.As(err, &veto) {
			t.Fatalf("peer %s defeated incomplete pause: %v", state, err)
		}
	}
	manager.mu.Lock()
	manager.failure = ""
	manager.mu.Unlock()
	if got := stopPauseExecutionWait(t, stopPauseExecutionImmediate(rb.b)); got.err != nil || got.state != nodepolicy.Paused {
		t.Fatalf("local retry: %+v", got)
	}
	after := rb.b.localPolicy().Policy
	if !reflect.DeepEqual(before.Roles, after.Roles) || !reflect.DeepEqual(before.ResidentModels, after.ResidentModels) || !reflect.DeepEqual(before.EngineBindings, after.EngineBindings) {
		t.Fatal("pause changed execution declarations")
	}
	state, err := rb.b.setAvailability(context.Background(), nodepolicy.Available, nodepolicy.PausedLocally)
	if err != nil || state != nodepolicy.Available {
		t.Fatalf("local resume: %s, %v", state, err)
	}
	r.mu.Lock()
	blocked := r.managementBlockedLocked()
	intent := r.intent["ollama"]
	r.mu.Unlock()
	if blocked || !intent {
		t.Fatalf("local Resume: blocked=%v preserved On intent=%v", blocked, intent)
	}
	rb.engine.waitFor("intent-neutral wake after Resume", func(calls []policyCall) bool {
		for _, call := range calls {
			if call.method == engineWakeMethod {
				return true
			}
		}
		return false
	})
}

func TestStopPauseExecutionLostReplyNeedsPositiveEvidence(t *testing.T) {
	test := func(name string, verified bool) {
		t.Run(name, func(t *testing.T) {
			rb, _ := stopPauseExecutionFixture(t, "owned")
			rb.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
				switch method {
				case "engine:get-installed":
					return map[string]any{"engines": []any{map[string]any{"engine": "owned", "running": false, "terminationVerified": verified}}}, ""
				case engineSleepMethod:
					return nil, "stop reply unavailable"
				}
				return map[string]bool{"ok": true}, ""
			})
			err := rb.b.pauseEnginesOn(context.Background(), rb.b.getEngineMgr())
			if verified && err != nil {
				t.Fatalf("positive current-manager readback refused: %v", err)
			}
			if !verified && err == nil {
				t.Fatal("running:false alone forgave an uncertain stop")
			}
		})
	}
	test("unverified false", false)
	test("positive termination readback", true)
}

func TestStopPauseExecutionLostTransportRetainsFence(t *testing.T) {
	rb, _ := stopPauseExecutionFixture(t, "owned")
	rb.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		switch method {
		case "engine:get-installed":
			return map[string]any{"engines": []any{map[string]any{"engine": "owned", "running": true}}}, ""
		case engineSleepMethod:
			if err := rb.engine.conn.Close(); err != nil {
				t.Errorf("close fixture transport: %v", err)
			}
			return nil, "reply lost after acceptance"
		}
		return map[string]bool{"ok": true}, ""
	})
	if got := stopPauseExecutionWait(t, stopPauseExecutionImmediate(rb.b)); got.err == nil || got.state != nodepolicy.Draining {
		t.Fatalf("lost manager transport: %+v", got)
	}
	stopPauseExecutionRequireFence(t, rb.b, nodepolicy.Draining)
}

func TestStopPauseExecutionPublicationRejectsRetiredProof(t *testing.T) {
	test := func(name string, replace bool) {
		t.Run(name, func(t *testing.T) {
			rb, _ := stopPauseExecutionFixture(t, "owned")
			r := rb.b.nodePolicyRuntime()
			r.mu.Lock()
			r.current.Availability = nodepolicy.Paused
			r.live = nodepolicy.Draining
			r.mu.Unlock()
			worker, source := rb.b.pauseManager()
			proved := make(chan struct{})
			publish := make(chan struct{})
			openPublication := sync.OnceFunc(func() { close(publish) })
			t.Cleanup(openPublication)
			done := make(chan error, 1)
			go func() {
				if err := rb.b.pauseEnginesOn(context.Background(), worker); err != nil {
					done <- err
					return
				}
				close(proved)
				<-publish
				done <- rb.b.publishPhysicalPause(context.Background(), worker, source)
			}()
			select {
			case <-proved:
			case err := <-done:
				t.Fatalf("shutdown proof: %v", err)
			case <-time.After(5 * time.Second):
				t.Fatal("shutdown proof did not complete")
			}
			rb.b.retireEngineManagerSource(source)
			if replace {
				next := rb.b.activateEngineManagerSource()
				attachFakeEngine(t, rb.b)
				rb.b.bindEngineMgr(rb.b.getEngineMgr(), next)
			}
			openPublication()
			select {
			case err := <-done:
				if err == nil {
					t.Fatal("retired manager proof published Paused")
				}
			case <-time.After(5 * time.Second):
				t.Fatal("publication did not finish")
			}
			stopPauseExecutionRequireFence(t, rb.b, nodepolicy.Draining)
			for _, p := range availabilityPushes(rb.proxy.snapshot()) {
				if p.State == nodepolicy.Paused {
					t.Fatal("stale physical proof reached proxy")
				}
			}
		})
	}
	test("retirement", false)
	test("replacement", true)
}

func TestStopPauseExecutionRestartAndRetirementFence(t *testing.T) {
	b, _ := newPolicyBroker(t)
	policy := nodepolicy.Default()
	policy.Availability, policy.PausedBy = nodepolicy.Paused, nodepolicy.PausedByPeer
	path := filepath.Join(t.TempDir(), nodepolicy.FileName)
	writePolicyFile(t, path, policy)
	b.loadNodePolicy(path)
	stopPauseExecutionRequireFence(t, b, nodepolicy.Draining)
	if !b.skipRestoreWhilePaused() {
		t.Fatal("persisted Pause permitted startup restoration")
	}
	rb, _ := stopPauseExecutionFixture(t, "owned")
	if got := stopPauseExecutionWait(t, stopPauseExecutionImmediate(rb.b)); got.err != nil {
		t.Fatalf("initial pause: %+v", got)
	}
	rb.b.retireEngineManagerSource(rb.source)
	stopPauseExecutionRequireFence(t, rb.b, nodepolicy.Draining)
}

func TestStopPauseExecutionRecoveryRejectsRetirementBeforePublication(t *testing.T) {
	rb, _ := stopPauseExecutionFixture(t, "owned")
	r := rb.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Availability = nodepolicy.Paused
	r.live = nodepolicy.Draining
	r.mu.Unlock()
	rb.engine.mu.Lock()
	base := rb.engine.answer
	rb.engine.mu.Unlock()
	var reads int
	rb.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, errText := base(method, raw)
		if method == "engine:get-installed" {
			reads++
			if reads == 2 {
				return result, holdSuccessReply
			}
		}
		return result, errText
	})
	worker, source := rb.b.pauseManager()
	done := make(chan bool, 1)
	go func() { done <- rb.b.recoverEngineSwitchOn(worker) }()
	rb.engine.waitFor("recovery shutdown readback", func(calls []policyCall) bool {
		count := 0
		for _, call := range calls {
			if call.method == "engine:get-installed" {
				count++
			}
		}
		return count == 2
	})
	// Recovery has released this lock for physical shutdown and must reacquire
	// it before publication. Keep that boundary closed while its source retires.
	rb.b.engineConfigMu.Lock()
	unlock := sync.OnceFunc(rb.b.engineConfigMu.Unlock)
	defer unlock()
	rb.engine.releaseHeld()
	rb.b.retireEngineManagerSource(source)
	next := rb.b.activateEngineManagerSource()
	attachFakeEngine(t, rb.b)
	rb.b.bindEngineMgr(rb.b.getEngineMgr(), next)
	unlock()
	select {
	case ok := <-done:
		if ok {
			t.Fatal("recovery accepted retired physical proof")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("recovery did not finish")
	}
	stopPauseExecutionRequireFence(t, rb.b, nodepolicy.Draining)
	for _, p := range availabilityPushes(rb.proxy.snapshot()) {
		if p.State == nodepolicy.Paused {
			t.Fatal("recovery published stale Paused")
		}
	}
}
