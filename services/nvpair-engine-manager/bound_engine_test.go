// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
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
	logs := captureBoundAuthorityLogs(t)
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
	records := readBoundAuthorityLogs(t, logs)
	if len(records) == 0 {
		t.Fatalf("accepted watcher refusal diagnostics = %+v", records)
	}
	// This fixture rewrites its marker and document while the watcher reads.
	// A transition read must fail closed too; synchronous denial is asserted
	// exactly by TestBoundEngineAuthorityRefusalDiagnostic below.
	for _, record := range records {
		transition := strings.HasPrefix(record.Reason, "permission_marker_") || strings.HasPrefix(record.Reason, "permission_document_")
		if record.Site != "engine_manager.remote_management" ||
			!(record.Reason == "permission_denied" && record.Permission == "denied" || transition && record.Permission == "unavailable") {
			t.Fatalf("unexpected watcher refusal diagnostic: %+v", record)
		}
	}
}

type boundAuthorityLogRecord struct {
	Message             string `json:"msg"`
	Site                string `json:"site"`
	Reason              string `json:"reason"`
	NodeID              string `json:"node_id"`
	PeerID              string `json:"peer_id"`
	ClusterID           string `json:"cluster_id"`
	Epoch               uint64 `json:"epoch"`
	Counter             uint64 `json:"counter"`
	ExpectedIncarnation string `json:"expected_incarnation"`
	ObservedIncarnation string `json:"observed_incarnation"`
	Permission          string `json:"permission"`
}

func captureBoundAuthorityLogs(t *testing.T) *lockedBuffer {
	t.Helper()
	logs := &lockedBuffer{}
	previous := slog.Default()
	previousWriter, previousFlags := log.Writer(), log.Flags()
	slog.SetDefault(slog.New(slog.NewJSONHandler(logs, nil)))
	t.Cleanup(func() {
		slog.SetDefault(previous)
		log.SetOutput(previousWriter)
		log.SetFlags(previousFlags)
	})
	return logs
}

func readBoundAuthorityLogs(t *testing.T, logs *lockedBuffer) []boundAuthorityLogRecord {
	t.Helper()
	decoder := json.NewDecoder(strings.NewReader(logs.String()))
	var records []boundAuthorityLogRecord
	for {
		var record boundAuthorityLogRecord
		if err := decoder.Decode(&record); errors.Is(err, io.EOF) {
			return records
		} else if err != nil {
			t.Fatalf("decode bound authority diagnostic: %v", err)
		}
		if record.Message == "bound engine management refused" {
			records = append(records, record)
		}
	}
}

func TestBoundEngineAuthorityRefusalDiagnostic(t *testing.T) {
	test := func(name, site, reason, permission string, change func(*testing.T, *Manager, *nodepolicy.ActivateBoundEngineRequest, *rmFixture)) {
		t.Run(name, func(t *testing.T) {
			m, p := boundActivationFixture(t)
			f := rmNewFixture(t, "self", "peer")
			f.writePermissions("1", map[string]bool{"peer": true})
			m.mesh = f.mesh
			incarnation, ok := f.mesh.PeerServiceIncarnation("peer")
			if !ok {
				t.Fatal("fixture lacks current peer authority")
			}
			p.PeerID, p.PeerIncarnation = "peer", incarnation
			change(t, m, &p, f)
			logs := captureBoundAuthorityLogs(t)
			var veto *localVetoError
			if err := m.boundEngineAuthority(p); !errors.As(err, &veto) || veto.code != codeRemoteManagementDenied {
				t.Fatalf("bound authority refusal = %v", err)
			}
			records := readBoundAuthorityLogs(t, logs)
			if len(records) != 1 || records[0].Site != site || records[0].Reason != reason ||
				records[0].Permission != permission || records[0].PeerID != "peer" || records[0].ExpectedIncarnation != p.PeerIncarnation {
				t.Fatalf("bound authority log = %+v", records)
			}
			if reason != "mesh_unavailable" && records[0].NodeID != "self" {
				t.Fatalf("refusal lost its deciding identity: %+v", records[0])
			}
			requireBoundNoStart(t, m)
		})
	}
	test("missing mesh", "engine_manager.peer_incarnation", "mesh_unavailable", "not_checked", func(_ *testing.T, m *Manager, _ *nodepolicy.ActivateBoundEngineRequest, _ *rmFixture) {
		m.mesh = nil
	})
	test("authority read", "engine_manager.peer_incarnation", "members_read", "not_checked", func(t *testing.T, _ *Manager, _ *nodepolicy.ActivateBoundEngineRequest, f *rmFixture) {
		if err := os.Remove(filepath.Join(f.dir, "members.json")); err != nil {
			t.Fatal(err)
		}
	})
	test("incarnation mismatch", "engine_manager.peer_incarnation", "incarnation_mismatch", "not_checked", func(_ *testing.T, _ *Manager, p *nodepolicy.ActivateBoundEngineRequest, _ *rmFixture) {
		p.PeerIncarnation = strings.Repeat("0", 64)
	})
	test("permission denied", "engine_manager.remote_management", "permission_denied", "denied", func(_ *testing.T, _ *Manager, _ *nodepolicy.ActivateBoundEngineRequest, f *rmFixture) {
		f.writePermissions("2", map[string]bool{"peer": false})
	})
}

func TestBoundEngineAuthoritySuccessDoesNotLog(t *testing.T) {
	m, p := boundActivationFixture(t)
	f := rmNewFixture(t, "self", "peer")
	f.writePermissions("1", map[string]bool{"peer": true})
	m.mesh = f.mesh
	incarnation, ok := f.mesh.PeerServiceIncarnation("peer")
	if !ok {
		t.Fatal("fixture lacks current peer authority")
	}
	p.PeerID, p.PeerIncarnation = "peer", incarnation
	logs := captureBoundAuthorityLogs(t)
	if err := m.boundEngineAuthority(p); err != nil {
		t.Fatalf("valid authority refused: %v", err)
	}
	if records := readBoundAuthorityLogs(t, logs); len(records) != 0 {
		t.Fatalf("successful bound authority emitted refusal: %+v", records)
	}
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
