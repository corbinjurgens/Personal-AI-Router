// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

func boundActivationFixture(t *testing.T) (*Manager, nodepolicy.ActivateBoundEngineRequest) {
	t.Helper()
	ex, _ := switchFixture(t)
	ex.nodeProjection.set(nodeProjection{Policy: nodepolicy.Advertisement{Revision: "1"}})
	m := &Manager{exec: ex}
	p := nodepolicy.ActivateBoundEngineRequest{OperationID: "cold-start", Engine: "target", Model: "model", PolicyRevision: "1", ManagerIncarnation: ex.switchIncarnation, Deadline: time.Now().Add(time.Second * 5).UTC()}
	return m, p
}

func requireBoundNoStart(t *testing.T, m *Manager) {
	t.Helper()
	st, err := m.exec.state("target")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	running, proc := st.running, st.proc
	st.mu.Unlock()
	if running || proc != nil {
		t.Fatal("refusal started target")
	}
	on, known, err := m.exec.desired.get("target")
	if err != nil {
		t.Fatal(err)
	}
	if known && on {
		t.Fatal("refusal persisted target On")
	}
}

func TestBoundEngineColdActivation(t *testing.T) {
	m, p := boundActivationFixture(t)
	if err := m.activateBoundEngine(context.Background(), p); err != nil {
		t.Fatal(err)
	}
	st, err := m.exec.state("target")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	running, adopted := st.running, st.adopted
	st.mu.Unlock()
	if !running || adopted {
		t.Fatalf("cold activation running=%t adopted=%t", running, adopted)
	}
	on, known, err := m.exec.desired.get("target")
	if err != nil || !known || !on {
		t.Fatalf("intent=(%t,%t) %v", on, known, err)
	}
}

func TestBoundEngineActivationRefusals(t *testing.T) {
	test := func(name string, change func(*testing.T, *Manager, *nodepolicy.ActivateBoundEngineRequest)) {
		t.Run(name, func(t *testing.T) {
			m, p := boundActivationFixture(t)
			change(t, m, &p)
			if err := m.activateBoundEngine(context.Background(), p); err == nil {
				t.Fatal("activation accepted")
			}
			requireBoundNoStart(t, m)
		})
	}
	test("locked", func(t *testing.T, m *Manager, _ *nodepolicy.ActivateBoundEngineRequest) {
		if _, err := m.exec.desired.setLocked("target", true); err != nil {
			t.Fatal(err)
		}
	})
	test("paused", func(_ *testing.T, m *Manager, _ *nodepolicy.ActivateBoundEngineRequest) {
		m.exec.nodeProjection.set(nodeProjection{Policy: nodepolicy.Advertisement{Revision: "1"}, ManagementBlocked: true})
	})
	test("stale revision", func(_ *testing.T, _ *Manager, p *nodepolicy.ActivateBoundEngineRequest) { p.PolicyRevision = "2" })
	test("replaced manager", func(_ *testing.T, _ *Manager, p *nodepolicy.ActivateBoundEngineRequest) {
		p.ManagerIncarnation = "replacement"
	})
	test("expired deadline", func(_ *testing.T, _ *Manager, p *nodepolicy.ActivateBoundEngineRequest) {
		p.Deadline = time.Now().Add(-time.Second)
	})
	test("unverified peer", func(_ *testing.T, _ *Manager, p *nodepolicy.ActivateBoundEngineRequest) {
		p.PeerID, p.PeerIncarnation = "peer", "unknown"
	})
	test("running source", func(t *testing.T, m *Manager, _ *nodepolicy.ActivateBoundEngineRequest) {
		if err := m.exec.Start(context.Background(), "source"); err != nil {
			t.Fatal(err)
		}
	})
}

func waitBoundLifecycleAdmission(t *testing.T, ex *Executor) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		ex.switchAdmission.Lock()
		admitted := ex.switchMutations[""] > 0
		ex.switchAdmission.Unlock()
		if admitted {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("cold activation did not enter lifecycle admission")
}

func TestBoundEnginePeerRevocationDuringLifecycleWait(t *testing.T) {
	m, p := boundActivationFixture(t)
	f := rmNewFixture(t, "self", "peer")
	f.writePermissions("1", map[string]bool{"peer": true})
	m.mesh = f.mesh
	incarnation, ok := f.mesh.PeerServiceIncarnation("peer")
	if !ok {
		t.Fatal("peer incarnation unavailable")
	}
	p.PeerID, p.PeerIncarnation = "peer", incarnation
	st, err := m.exec.state("target")
	if err != nil {
		t.Fatal(err)
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	done := make(chan error, 1)
	go func() { done <- m.activateBoundEngine(context.Background(), p) }()
	waitBoundLifecycleAdmission(t, m.exec)
	f.writePermissions("2", map[string]bool{"peer": false})
	select {
	case err := <-done:
		var veto *localVetoError
		if !errors.As(err, &veto) || veto.code != codeRemoteManagementDenied {
			t.Fatalf("revocation=%v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("revocation could not cancel lifecycle waiting")
	}
	requireBoundNoStart(t, m)
}

func TestBoundEngineCallerCancellationDuringLifecycleWait(t *testing.T) {
	m, p := boundActivationFixture(t)
	st, err := m.exec.state("target")
	if err != nil {
		t.Fatal(err)
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- m.activateBoundEngine(ctx, p) }()
	waitBoundLifecycleAdmission(t, m.exec)
	if done, err := m.exec.admitMutation("other"); err == nil {
		done()
		t.Fatal("competing engine mutation admitted")
	}
	cancel()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("cancelled activation succeeded")
		}
	case <-time.After(time.Second):
		t.Fatal("cancelled lifecycle wait blocked")
	}
	requireBoundNoStart(t, m)
}

func TestBoundEngineActualStopPreemptsLifecycleWait(t *testing.T) {
	m, p := boundActivationFixture(t)
	st, err := m.exec.state("target")
	if err != nil {
		t.Fatal(err)
	}
	st.opMu.Lock()
	locked := true
	defer func() {
		if locked {
			st.opMu.Unlock()
		}
	}()
	activation := make(chan error, 1)
	go func() { activation <- m.activateBoundEngine(context.Background(), p) }()
	waitBoundLifecycleAdmission(t, m.exec)
	stop := make(chan error, 1)
	go func() { stop <- m.exec.Stop("target") }()
	select {
	case err := <-activation:
		if err == nil {
			t.Fatal("Off did not preempt activation")
		}
	case <-time.After(time.Second):
		t.Fatal("actual Stop could not cancel activation behind lifecycle lock")
	}
	requireBoundNoStart(t, m)
	st.opMu.Unlock()
	locked = false
	select {
	case err := <-stop:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("actual Stop did not finish")
	}
	on, known, err := m.exec.desired.get("target")
	if err != nil || !known || on {
		t.Fatalf("Off intent=(%t,%t) %v", on, known, err)
	}
	if err := m.exec.Start(context.Background(), "target"); err != nil {
		t.Fatal(err)
	}
	applied := false
	if err := m.exec.stopFromApplied("target", intentLocal, p.OperationID, &applied); err != nil || applied {
		t.Fatalf("late scoped Off=(%t,%v)", applied, err)
	}
	st.mu.Lock()
	running := st.running
	st.mu.Unlock()
	if !running {
		t.Fatal("late Off stopped newer manual On")
	}
}

func TestBoundEnginePausePreemptsLifecycleWait(t *testing.T) {
	m, p := boundActivationFixture(t)
	st, err := m.exec.state("target")
	if err != nil {
		t.Fatal(err)
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	done := make(chan error, 1)
	go func() { done <- m.activateBoundEngine(context.Background(), p) }()
	waitBoundLifecycleAdmission(t, m.exec)
	raw, err := json.Marshal(nodeProjection{Policy: nodepolicy.Advertisement{Revision: "1"}, ManagementBlocked: true})
	if err != nil {
		t.Fatal(err)
	}
	m.applyNodeProjection(&Message{Method: nodeProjectionMethod, Params: raw})
	select {
	case err := <-done:
		var veto *localVetoError
		if !errors.As(err, &veto) || veto.code != codeLocalVeto {
			t.Fatalf("pause=%v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("pause could not cancel lifecycle waiting")
	}
	requireBoundNoStart(t, m)
}

func TestBoundEngineOldE4ScopeCannotVetoColdActivation(t *testing.T) {
	m, p := boundActivationFixture(t)
	st, err := m.exec.state("target")
	if err != nil {
		t.Fatal(err)
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	activation := make(chan error, 1)
	go func() { activation <- m.activateBoundEngine(ctx, p) }()
	waitBoundLifecycleAdmission(t, m.exec)
	stop := make(chan error, 1)
	go func() { stop <- m.exec.StopForSwitch("target", p.OperationID) }()
	select {
	case err := <-stop:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("E4 scope entered a cold operation with the same text ID")
	}
	select {
	case err := <-activation:
		t.Fatalf("E4 scope cancelled cold activation: %v", err)
	default:
	}
	cancel()
	select {
	case <-activation:
	case <-time.After(time.Second):
		t.Fatal("cold activation did not cancel")
	}
	requireBoundNoStart(t, m)
}
