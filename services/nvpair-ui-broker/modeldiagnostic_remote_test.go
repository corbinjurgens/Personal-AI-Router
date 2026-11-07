// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrusttest"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

func diagnosticPeerFixture(t *testing.T) (*Broker, *fakeWorker, modelconfig.DiagnosticPreviewRequest, diagnosticObservation, *engineManagerSource, context.Context) {
	t.Helper()
	b, manager, p, observation := diagnosticPreviewFixture(t)
	clustertrusttest.Join(t, b.clusterDir, "cluster", "target", "peer")
	grantRemoteManagement(t, b.clusterDir, "cluster", "target", "peer", true)
	source := bindRemoteManagementTestSource(b)
	ctx := context.WithValue(context.Background(), diagnosticPeerContextKey{}, &diagnosticPeerAuthority{source, b.getEngineMgr(), "peer"})
	return b, manager, p, observation, source, ctx
}

func TestModelDiagnosticPeerConfiguredPauseAcceptsWithoutChangingPause(t *testing.T) {
	b, manager, p, observation, _, ctx := diagnosticPeerFixture(t)
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.PauseMode, r.current.Availability = nodepolicy.PauseConfigured, nodepolicy.Paused
	r.mu.Unlock()
	preview, err := b.previewDiagnostic(ctx, p)
	if err != nil {
		t.Fatal(err)
	}
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case diagnosticObserveMethod:
			return observation, ""
		case "engine:diagnostic-reserve":
			close(entered)
			<-release
			return nil, "synthetic reservation refused"
		case "engine:diagnostic-run-status":
			return map[string]string{"state": "none"}, ""
		}
		return nil, "unexpected diagnostic effect"
	})
	run, err := b.startDiagnostic(ctx, modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: p, PreviewID: preview.PreviewID, Consent: true})
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("accepted operation did not reach reservation owner")
	}
	b.diagnostics.mu.Lock()
	op := b.diagnostics.active
	b.diagnostics.mu.Unlock()
	if op == nil || op.runID != run.RunID {
		t.Fatal("accepted run lost target ownership")
	}
	status, err := b.dispatchDiagnostic(ctx, "engine:diagnostic-status", rmJSON(t, modelconfig.DiagnosticRunRef{}), false)
	if err != nil {
		t.Fatal(err)
	}
	active, ok := status.(*modelconfig.DiagnosticRun)
	if !ok || active == nil || active.RunID != run.RunID {
		t.Fatalf("reconnect status=%+v", status)
	}
	unblock()
	select {
	case <-op.done:
	case <-time.After(3 * time.Second):
		t.Fatal("synthetic refusal did not join cleanup")
	}
	requireDiagnosticReleased(t, b)
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.current.PauseMode != nodepolicy.PauseConfigured || r.current.Availability != nodepolicy.Paused {
		t.Fatal("diagnostic changed saved pause")
	}
}

func TestModelDiagnosticPeerPreviewStartVetoesHaveNoEffects(t *testing.T) {
	for _, scenario := range []string{"permission", "Halt", "maintenance", "pending pause", "retired source", "different worker", "cancelled waiter"} {
		t.Run(scenario, func(t *testing.T) {
			b, manager, p, _, source, ctx := diagnosticPeerFixture(t)
			preview, err := b.previewDiagnostic(ctx, p)
			if err != nil {
				t.Fatal(err)
			}
			switch scenario {
			case "permission":
				grantRemoteManagement(t, b.clusterDir, "cluster", "target", "peer", false)
			case "Halt":
				r := b.nodePolicyRuntime()
				r.mu.Lock()
				r.current.PauseMode = nodepolicy.PauseImmediate
				r.mu.Unlock()
			case "maintenance":
				b.engineConfigMu.Lock()
				b.buildApplyMaintenance.active = &buildApplyReservation{}
				b.engineConfigMu.Unlock()
			case "pending pause":
				r := b.nodePolicyRuntime()
				r.mu.Lock()
				r.pendingPauseRequested = true
				r.mu.Unlock()
			case "retired source":
				b.retireEngineManagerSource(source)
			case "different worker":
				peer, _ := newFakeWorkerPipe(t)
				b.setEngineMgr(&rpcWorker{peer: peer})
			case "cancelled waiter":
				cancelled, cancel := context.WithCancel(ctx)
				cancel()
				ctx = cancelled
			}
			if _, err := b.previewDiagnostic(ctx, p); err == nil {
				t.Fatal("vetoed preview accepted")
			}
			if _, err := b.startDiagnostic(ctx, modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: p, PreviewID: preview.PreviewID, Consent: true}); err == nil {
				t.Fatal("vetoed start accepted")
			}
			requireNoDiagnosticEffects(t, b, manager)
		})
	}
}

func TestModelDiagnosticPeerRetirementDuringObservationRefusesAcceptance(t *testing.T) {
	b, manager, p, observation, source, ctx := diagnosticPeerFixture(t)
	preview, err := b.previewDiagnostic(ctx, p)
	if err != nil {
		t.Fatal(err)
	}
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == diagnosticObserveMethod {
			close(entered)
			<-release
			return observation, ""
		}
		return nil, "unexpected effect"
	})
	done := make(chan error, 1)
	go func() {
		_, err := b.startDiagnostic(ctx, modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: p, PreviewID: preview.PreviewID, Consent: true})
		done <- err
	}()
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("start observation did not begin")
	}
	b.retireEngineManagerSource(source)
	unblock()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("retired source accepted after observation")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("stale acceptance did not join")
	}
	requireNoDiagnosticEffects(t, b, manager)
}

func TestModelDiagnosticPeerPreviewCannotBeConsumedByLocalCaller(t *testing.T) {
	b, manager, p, _, _, ctx := diagnosticPeerFixture(t)
	preview, err := b.previewDiagnostic(ctx, p)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.startDiagnostic(context.Background(), modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: p, PreviewID: preview.PreviewID, Consent: true}); err == nil {
		t.Fatal("different caller consumed peer consent")
	}
	requireNoDiagnosticEffects(t, b, manager)
}

func TestModelDiagnosticPeerRevocationDuringObservationRefusesAcceptance(t *testing.T) {
	b, manager, p, observation, _, ctx := diagnosticPeerFixture(t)
	preview, err := b.previewDiagnostic(ctx, p)
	if err != nil {
		t.Fatal(err)
	}
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method != diagnosticObserveMethod {
			return nil, "unexpected effect"
		}
		grantRemoteManagement(t, b.clusterDir, "cluster", "target", "peer", false)
		return observation, ""
	})
	_, err = b.startDiagnostic(ctx, modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: p, PreviewID: preview.PreviewID, Consent: true})
	rmDeny(t, err)
	requireNoDiagnosticEffects(t, b, manager)
}

func TestModelDiagnosticPeerPreviewBindsExactSourceEvenWhenWorkerIsRebound(t *testing.T) {
	b, manager, p, _, _, ctx := diagnosticPeerFixture(t)
	preview, err := b.previewDiagnostic(ctx, p)
	if err != nil {
		t.Fatal(err)
	}
	worker := b.getEngineMgr()
	source := b.activateEngineManagerSource()
	b.bindEngineMgr(worker, source)
	ctx = context.WithValue(context.Background(), diagnosticPeerContextKey{}, &diagnosticPeerAuthority{source, worker, "peer"})
	if _, err := b.startDiagnostic(ctx, modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: p, PreviewID: preview.PreviewID, Consent: true}); err == nil {
		t.Fatal("new source consumed retired source's preview")
	}
	requireNoDiagnosticEffects(t, b, manager)
}

func TestModelDiagnosticPeerRelaySourceAndCallerFence(t *testing.T) {
	b, manager, _, _, source, _ := diagnosticPeerFixture(t)
	run := modelconfig.DiagnosticRun{RunID: "peer-run", Engine: "llamacpp", ModelID: "fixture.gguf", ProfileKey: "fixture", ConfigurationKey: strings.Repeat("a", 64), Phase: "high", Outcome: "running", StartedAt: time.Now().Format(time.RFC3339Nano)}
	b.diagnostics.last = &run
	request := diagnosticRelayRequest{ID: "lookup", Method: "engine:diagnostic-status", Caller: "peer", Params: rmJSON(t, modelconfig.DiagnosticRunRef{})}
	b.forwardEngineNotificationFrom(source, "engine:diagnostic-request", rmJSON(t, request))
	reply := rmWaitReply(t, manager, "engine:diagnostic-reply", request.ID)
	var got modelconfig.DiagnosticRun
	if err := modelconfig.DecodeStrict(reply.Result, &got); err != nil {
		t.Fatal(err)
	}
	if got.RunID != run.RunID || reply.Error != "" {
		t.Fatalf("relay lookup=%+v %+v", got, reply)
	}
	request.ID, request.Caller = "blank-caller", ""
	b.forwardEngineNotificationFrom(source, "engine:diagnostic-request", rmJSON(t, request))
	if refusal := rmWaitReply(t, manager, "engine:diagnostic-reply", request.ID); refusal.Error == "" {
		t.Fatal("blank caller took local path")
	}
	b.retireEngineManagerSource(source)
	request.ID, request.Caller = "stale", "peer"
	b.forwardEngineNotificationFrom(source, "engine:diagnostic-request", rmJSON(t, request))
	b.diagnostics.mu.Lock()
	_, queued := b.diagnostics.relayCancels[diagnosticRelayKey{source, request.ID}]
	b.diagnostics.mu.Unlock()
	if queued {
		t.Fatal("retired source registered relay work")
	}
}

func TestModelDiagnosticRelayWaitCancellationIsSourceScoped(t *testing.T) {
	b, _, _, _, old, _ := diagnosticPeerFixture(t)
	current := b.activateEngineManagerSource()
	b.bindEngineMgr(b.getEngineMgr(), current)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	b.diagnostics.mu.Lock()
	b.diagnostics.relayCancels = map[diagnosticRelayKey]context.CancelFunc{{current, "request"}: cancel}
	b.diagnostics.mu.Unlock()
	params := rmJSON(t, map[string]string{"id": "request"})
	b.forwardEngineNotificationFrom(old, "engine:diagnostic-request-cancel", params)
	if ctx.Err() != nil {
		t.Fatal("retired source cancelled replacement waiter")
	}
	b.forwardEngineNotificationFrom(current, "engine:diagnostic-request-cancel", params)
	if !errors.Is(ctx.Err(), context.Canceled) {
		t.Fatal("current source failed to retire exact waiter")
	}
}

func TestModelDiagnosticPeerCancelRemainsAvailableDuringHalt(t *testing.T) {
	b, manager, op := diagnosticBrokerFixture(t)
	clustertrusttest.Join(t, b.clusterDir, "cluster", "target", "peer")
	grantRemoteManagement(t, b.clusterDir, "cluster", "target", "peer", true)
	source := bindRemoteManagementTestSource(b)
	ctx := context.WithValue(context.Background(), diagnosticPeerContextKey{}, &diagnosticPeerAuthority{source, op.worker, "peer"})
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.PauseMode = nodepolicy.PauseImmediate
	r.mu.Unlock()
	run := op.run.Clone()
	run.Phase, run.Outcome = "done", "cancelled"
	token := op.token
	setDiagnosticContractAnswer(t, manager, op, func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:diagnostic-run-status" {
			return diagnosticStatusReply{Run: run, State: "cleanup-complete", Token: token, WorkStopped: true}, ""
		}
		return nil, "unexpected effect"
	})
	result, err := b.cancelDiagnostic(ctx, op.runID)
	if err != nil || result == nil || result.RunID != op.runID {
		t.Fatalf("cancel=%+v error=%v", result, err)
	}
	requireDiagnosticReleased(t, b)
}

func TestModelDiagnosticOriginRoutesPeerAndRejectsPrivateOrMismatchedRun(t *testing.T) {
	for _, scenario := range []string{"public run", "different run", "private authority"} {
		t.Run(scenario, func(t *testing.T) {
			b, _ := newPolicyBroker(t)
			peer, manager := newFakeWorkerPipe(t)
			b.setEngineMgr(&rpcWorker{peer: peer})
			run := diagnosticPublicRun{DiagnosticRun: modelconfig.DiagnosticRun{RunID: "exact", Engine: "llamacpp", ModelID: "fixture.gguf", ProfileKey: "fixture", ConfigurationKey: strings.Repeat("a", 64), Phase: "high", Outcome: "running", StartedAt: time.Now().Format(time.RFC3339Nano)}, NodeID: "host-peer"}
			manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method != "engine:remote-diagnostic-status" {
					return nil, "wrong remote route"
				}
				var p modelconfig.DiagnosticRunRef
				if err := modelconfig.DecodeStrict(raw, &p); err != nil {
					return nil, err.Error()
				}
				if p.NodeID != "host-peer" || p.RunID != "exact" {
					return nil, "request identity changed"
				}
				switch scenario {
				case "different run":
					run.RunID = "later"
				case "private authority":
					return map[string]any{"run": run, "token": "private"}, ""
				}
				return run, ""
			})
			result, err := b.dispatchDiagnostic(context.Background(), "engine:diagnostic-status", rmJSON(t, modelconfig.DiagnosticRunRef{NodeID: "host-peer", RunID: "exact"}), true)
			if scenario != "public run" {
				if err == nil {
					t.Fatal("unsafe response exposed publicly")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			raw, ok := result.(json.RawMessage)
			if !ok {
				t.Fatal("remote response not projected")
			}
			var got diagnosticPublicRun
			if err := modelconfig.DecodeStrict(raw, &got); err != nil {
				t.Fatal(err)
			}
			if got.RunID != "exact" || got.NodeID != "host-peer" {
				t.Fatalf("origin response=%+v", got)
			}
			if b.diagnosticHoldActive() {
				t.Fatal("origin acquired target resources")
			}
		})
	}
}

func TestModelDiagnosticOriginEmptyPeerStatusIsNull(t *testing.T) {
	b, _ := newPolicyBroker(t)
	peer, manager := newFakeWorkerPipe(t)
	b.setEngineMgr(&rpcWorker{peer: peer})
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method != "engine:remote-diagnostic-status" {
			return nil, "unexpected method"
		}
		return nil, ""
	})
	result, err := b.dispatchDiagnostic(context.Background(), "engine:diagnostic-status", rmJSON(t, modelconfig.DiagnosticRunRef{NodeID: "host-peer"}), true)
	if err != nil {
		t.Fatal(err)
	}
	raw, ok := result.(json.RawMessage)
	if !ok || string(raw) != "null" {
		t.Fatalf("empty status=%+v", result)
	}
}
