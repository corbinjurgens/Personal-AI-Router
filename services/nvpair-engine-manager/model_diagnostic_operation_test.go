// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"sync"
	"testing"
	"time"

	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

func modelDiagnosticOperationFixture(t *testing.T, configure func(inspectionFixture)) (inspectionFixture, *Manager, *modelDiagnosticOperation) {
	t.Helper()
	f := newAppliedKeyFixture(t)
	f.ex.emit = nil
	f.ex.nodeProjection.set(nodeProjection{})
	if configure != nil {
		configure(f)
	}
	ctx := f.ex.modelMemoryContext(context.Background(), "llamacpp", modelBufferTestTargetMemory(), false)
	identity, err := f.ex.inspectModelArtifacts(ctx, f.request)
	if err != nil {
		t.Fatalf("inspect operation fixture: %v", err)
	}
	fence, err := f.ex.diagnosticEffectsFence(ctx)
	if err != nil {
		t.Fatalf("observe operation effects: %v", err)
	}
	p := diagnosticReserveParams{RunID: "operation-test", Engine: "llamacpp", ModelID: f.request.ModelID,
		ProfileKey: f.request.ProfileKey, ConfigurationKey: identity.ConfigurationKey, ExecutionKey: identity.ExecutionKey,
		EffectsFence: fence, TargetMemory: modelBufferTestTargetMemory(), Hardware: diagnosticTestHardware()}
	reserved, err := f.ex.reserveDiagnostic(ctx, p)
	if err != nil {
		t.Fatalf("reserve operation: %v", err)
	}
	before := f.ex.diagnosticRunStatus(p.RunID)
	if before.State != "reserved" || before.Token != reserved.Token || !before.WorkStopped {
		t.Fatalf("reserved status: %+v", before)
	}
	op, start, err := f.ex.acceptDiagnostic(diagnosticRunParams{diagnosticReleaseParams: diagnosticReleaseParams{p.RunID, reserved.Token}})
	if err != nil || !start {
		t.Fatalf("accept operation: start=%v err=%v", start, err)
	}
	t.Cleanup(op.cancel)
	return f, &Manager{exec: f.ex}, op
}

func TestModelDiagnosticAcceptBindsReservationWithoutGuessedTargets(t *testing.T) {
	f, _, op := modelDiagnosticOperationFixture(t, nil)
	if op.params.Low != 0 || op.params.High != 0 || op.params.Generate != 128 || op.status.ProfileKey != f.request.ProfileKey {
		t.Fatalf("accepted inputs: %+v", op.params)
	}
	replay, start, err := f.ex.acceptDiagnostic(diagnosticRunParams{diagnosticReleaseParams: diagnosticReleaseParams{op.params.RunID, op.params.Token}})
	if err != nil || start || replay != op {
		t.Fatalf("repeated acceptance: op=%p start=%v err=%v", replay, start, err)
	}
	changed := op.params
	changed.ModelID = "different-model"
	if _, _, err := f.ex.acceptDiagnostic(changed); err == nil {
		t.Fatal("different model reused run authority")
	}
	if err := f.ex.releaseDiagnostic(diagnosticReleaseParams{op.params.RunID, op.params.Token}); err == nil {
		t.Fatal("accepted work released without cleanup")
	}
}

func TestModelDiagnosticBaselineAcknowledgementIsPhaseAndTokenBound(t *testing.T) {
	f, _, op := modelDiagnosticOperationFixture(t, nil)
	valid := diagnosticBaselineParams{diagnosticReleaseParams{op.params.RunID, op.params.Token}, "baseline"}
	if err := f.ex.diagnosticBaselineDone(valid); err == nil {
		t.Fatal("acknowledged before cold baseline")
	}
	f.ex.diagnosticPhase(op, "baseline", "baseline-ready")
	for _, tc := range []struct {
		name   string
		change func(*diagnosticBaselineParams)
	}{
		{"wrong run", func(p *diagnosticBaselineParams) { p.RunID = "other-run" }},
		{"wrong token", func(p *diagnosticBaselineParams) { p.Token = "other-token" }},
		{"wrong phase", func(p *diagnosticBaselineParams) { p.Phase = "loading" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := valid
			tc.change(&p)
			if err := f.ex.diagnosticBaselineDone(p); err == nil {
				t.Fatal("accepted a mismatched barrier")
			}
		})
	}
	if err := f.ex.diagnosticBaselineDone(valid); err != nil {
		t.Fatalf("acknowledge baseline: %v", err)
	}
	if err := f.ex.diagnosticBaselineDone(valid); err != nil {
		t.Fatalf("repeat acknowledgement: %v", err)
	}
	modelDiagnosticWait(t, op.baseline, "baseline acknowledgement")
	f.ex.diagnosticPhase(op, "loading", "")
	if err := f.ex.diagnosticBaselineDone(valid); err == nil {
		t.Fatal("late acknowledgement accepted after loading began")
	}
	valid.Phase = "after-load"
	if err := f.ex.diagnosticBaselineDone(valid); err == nil {
		t.Fatal("acknowledged before load identity was confirmed")
	}
	f.ex.diagnosticPhase(op, "loading", "loaded")
	if err := f.ex.diagnosticBaselineDone(valid); err != nil {
		t.Fatalf("acknowledge after-load samples: %v", err)
	}
	modelDiagnosticWait(t, op.afterLoad, "after-load acknowledgement")
	f.ex.diagnosticPhase(op, "low", "low-start")
	if err := f.ex.diagnosticBaselineDone(valid); err == nil {
		t.Fatal("late after-load acknowledgement accepted during exercise")
	}
}

func TestModelDiagnosticAbortJoinsBaselineAndRetainsCompletedBinding(t *testing.T) {
	f, manager, op := modelDiagnosticOperationFixture(t, nil)
	baseline := make(chan struct{}, 1)
	f.ex.emit = func(method string, value any) {
		if method == diagnosticChangedMethod {
			if status, ok := value.(diagnosticOperationStatus); ok && status.Marker == "baseline-ready" {
				baseline <- struct{}{}
			}
		}
	}
	go manager.runDiagnosticOperation(op, false)
	modelDiagnosticWait(t, baseline, "cold baseline")
	status, err := manager.abortDiagnostic(diagnosticReleaseParams{op.params.RunID, op.params.Token})
	if err != nil {
		t.Fatalf("abort operation: %v", err)
	}
	if status.Outcome != "cancelled" || status.Phase != "done" || !status.WorkStopped || status.State != "cleanup-complete" {
		t.Fatalf("joined status: %+v", status)
	}
	modelDiagnosticWait(t, op.done, "operation join")
	retained, ok := f.ex.completedDiagnosticRun(op.params.RunID)
	if !ok || retained.Reservation.ConfigurationKey != op.params.ConfigurationKey || retained.Status.Markers["baseline-ready"].IsZero() {
		t.Fatalf("completed binding: %+v, found=%v", retained, ok)
	}
	delete(retained.Status.Markers, "baseline-ready")
	if f.ex.diagnosticRunStatus(op.params.RunID).Markers["baseline-ready"].IsZero() {
		t.Fatal("caller mutated retained markers")
	}
	if _, err := f.ex.admitMutation("llamacpp"); err == nil {
		t.Fatal("physical cleanup released execution before broker recording")
	}
	checkpoint, err := f.ex.desired.pendingDiagnostic()
	if err != nil || checkpoint == nil || checkpoint.State != "cleanup-complete" {
		t.Fatalf("cleanup proof was not retained for broker recording: checkpoint=%+v error=%v", checkpoint, err)
	}
	if err := f.ex.releaseDiagnostic(diagnosticReleaseParams{op.params.RunID, op.params.Token}); err != nil {
		t.Fatalf("explicit release after conclusive cleanup: %v", err)
	}
	if done, err := f.ex.admitMutation("llamacpp"); err != nil {
		t.Fatalf("explicit release retained execution: %v", err)
	} else {
		done()
	}
	if f.ex.diagnosticRunStatus("unknown").State != "none" {
		t.Fatal("unknown run was reported as active")
	}
}

func TestModelDiagnosticNewPauseCancelsBeforeTransitionJoin(t *testing.T) {
	for _, mode := range []nodepolicy.PauseMode{nodepolicy.PauseConfigured, nodepolicy.PauseImmediate} {
		t.Run(string(mode), func(t *testing.T) {
			f, manager, op := modelDiagnosticOperationFixture(t, nil)
			baseline := make(chan struct{}, 1)
			f.ex.emit = func(method string, value any) {
				if status, ok := value.(diagnosticOperationStatus); ok && method == diagnosticChangedMethod && status.Marker == "baseline-ready" {
					baseline <- struct{}{}
				}
			}
			go manager.runDiagnosticOperation(op, false)
			modelDiagnosticWait(t, baseline, "baseline before pause")
			projection := nodeProjection{ManagementBlocked: true, PausedLocally: true, Policy: nodepolicy.Advertisement{PauseMode: mode}}
			params, err := json.Marshal(projection)
			if err != nil {
				t.Fatalf("encode projection: %v", err)
			}
			manager.applyNodeProjection(&Message{Params: params})
			f.ex.abortDiagnosticAndJoin()
			status := f.ex.diagnosticRunStatus(op.params.RunID)
			if status.Outcome != "cancelled" || !status.WorkStopped {
				t.Fatalf("pause returned without cancellation and cleanup: %+v", status)
			}
			if adv := f.ex.nodeProjection.advertisement(); adv == nil || adv.PauseMode != mode {
				t.Fatalf("cleanup changed saved pause projection: %+v", adv)
			}
		})
	}
}

func TestModelDiagnosticCleanupFailureRetainsHoldUntilRetry(t *testing.T) {
	f, manager, op := modelDiagnosticOperationFixture(t, nil)
	op.outcome = "cancelled"
	op.engines = []diagnosticEngineBefore{{engine: "unavailable-engine", touched: true}}
	manager.runDiagnosticOperation(op, true)
	status := f.ex.diagnosticRunStatus(op.params.RunID)
	if status.Outcome != "cleanup-failed" || status.WorkStopped {
		t.Fatalf("unconfirmed cleanup status: %+v", status)
	}
	if done, err := f.ex.admitMutation("llamacpp"); err == nil {
		done()
		t.Fatal("cleanup failure reopened admission")
	}
	if err := f.ex.releaseDiagnostic(diagnosticReleaseParams{op.params.RunID, op.params.Token}); err == nil {
		t.Fatal("cleanup failure released reservation")
	}
	// The failed observation is resolved. Retry performs only cleanup and
	// preserves the original outcome; it never repeats generation or load.
	op.engines = nil
	status, err := manager.abortDiagnostic(diagnosticReleaseParams{op.params.RunID, op.params.Token})
	if err != nil {
		t.Fatalf("retry cleanup: %v", err)
	}
	if !status.WorkStopped || status.Outcome != "cancelled" {
		t.Fatalf("retry status: %+v", status)
	}
	if err := f.ex.releaseDiagnostic(diagnosticReleaseParams{op.params.RunID, op.params.Token}); err != nil {
		t.Fatalf("release after cleanup and recording interval: %v", err)
	}
	if done, err := f.ex.admitMutation("llamacpp"); err != nil {
		t.Fatalf("admission after cleanup: %v", err)
	} else {
		done()
	}
}

type modelDiagnosticOperationHTTP struct {
	mu          sync.Mutex
	loaded      bool
	markers     []string
	requests    []string
	beforeLoad  func(*http.Request) bool
	beforeCount func(*http.Request) bool
	fixture     *modelDiagnosticExerciseFake
}

func (f *modelDiagnosticOperationHTTP) serve(t *testing.T, w http.ResponseWriter, r *http.Request) {
	t.Helper()
	switch r.URL.Path {
	case "/models":
		f.mu.Lock()
		loaded := f.loaded
		f.mu.Unlock()
		status := "unloaded"
		if loaded {
			status = "loaded"
		}
		if err := json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "exact-alias", "status": map[string]string{"value": status}}}}); err != nil {
			t.Errorf("write residency: %v", err)
		}
	case "/models/load", "/models/unload":
		if r.URL.Path == "/models/load" && f.beforeLoad != nil && f.beforeLoad(r) {
			return
		}
		f.mu.Lock()
		f.loaded = r.URL.Path == "/models/load"
		f.requests = append(f.requests, r.URL.Path)
		f.mu.Unlock()
		if _, err := w.Write([]byte(`{}`)); err != nil {
			t.Errorf("write mutation result: %v", err)
		}
	case "/props":
		if r.URL.Query().Get("autoload") != "false" {
			t.Error("props read could load a model")
		}
		if _, err := w.Write([]byte(`{"build_info":"synthetic-native-build","total_slots":1,"default_generation_settings":{"n_ctx":4096},"devices":[{"id":"GPU-observed"}],"n_gpu_layers":0}`)); err != nil {
			t.Errorf("write props: %v", err)
		}
	default:
		if f.beforeCount != nil && r.URL.Path == "/v1/chat/completions/input_tokens" && f.beforeCount(r) {
			return
		}
		f.fixture.serve(w, r)
	}
}

func modelDiagnosticRunningOperationFixture(t *testing.T, configure ...func(inspectionFixture)) (inspectionFixture, *Manager, *modelDiagnosticOperation, *modelDiagnosticOperationHTTP) {
	t.Helper()
	fixture := &modelDiagnosticOperationHTTP{loaded: true, fixture: &modelDiagnosticExerciseFake{t: t, child: modelDiagnosticExerciseChild{ModelID: "exact-alias"}}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { fixture.serve(t, w, r) }))
	t.Cleanup(server.Close)
	_, portString, err := net.SplitHostPort(server.Listener.Addr().String())
	if err != nil {
		t.Fatalf("read fake server address: %v", err)
	}
	port, err := strconv.Atoi(portString)
	if err != nil {
		t.Fatalf("read fake server port: %v", err)
	}
	f, manager, op := modelDiagnosticOperationFixture(t, func(f inspectionFixture) {
		f.ex.client = server.Client()
		f.st.plat.Detect = []string{f.st.binPath}
		f.st.manifest.Actions = map[string]Action{
			"loaded_models": {HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/models"}, Result: &ActionResult{Array: "data", Field: "id", Match: &ResultMatch{Field: "status.value", In: []string{"loaded"}}}},
			"load_model":    {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/models/load"}},
			"unload_model":  {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/models/unload"}},
		}
		f.st.running, f.st.runID, f.st.port = true, "owned-fixture-run", port
		f.st.launchDigest = "owned-fixture-launch"
		f.st.proc = &managedProc{done: make(chan struct{})}
		for _, edit := range configure {
			edit(f)
		}
	})
	f.ex.emit = func(method string, value any) {
		if method != diagnosticChangedMethod {
			return
		}
		status, ok := value.(diagnosticOperationStatus)
		if !ok {
			return
		}
		if status.Marker != "" {
			fixture.mu.Lock()
			fixture.markers = append(fixture.markers, status.Marker)
			fixture.mu.Unlock()
		}
		phase := ""
		if status.Marker == "baseline-ready" {
			phase = "baseline"
		} else if status.Marker == "loaded" {
			phase = "after-load"
		}
		if phase != "" {
			if err := f.ex.diagnosticBaselineDone(diagnosticBaselineParams{diagnosticReleaseParams{op.params.RunID, op.params.Token}, phase}); err != nil {
				t.Errorf("acknowledge baseline: %v", err)
			}
		}
	}
	return f, manager, op, fixture
}

func TestModelDiagnosticMeasuresObservedAllocationAndOrdersMarkers(t *testing.T) {
	f, manager, op, fixture := modelDiagnosticRunningOperationFixture(t)
	manager.runDiagnosticOperation(op, false)
	status := f.ex.diagnosticRunStatus(op.params.RunID)
	if status.Outcome != "completed" || !status.WorkStopped || status.Observation == nil || status.Observation.PerSlotContext != 4096 || status.Observation.BuildInfo != "synthetic-native-build" || status.Observation.NativeBuildID == "" {
		t.Fatalf("completed measurement: %+v", status)
	}
	if status.Low == nil || status.High == nil || status.Low.Requests[0].TargetPromptTokens != 512 || status.High.Requests[0].TargetPromptTokens != 3952 {
		t.Fatalf("observed exercise targets: low=%+v high=%+v", status.Low, status.High)
	}
	fixture.mu.Lock()
	defer fixture.mu.Unlock()
	if !slices.Equal(fixture.markers, []string{"baseline-ready", "loaded", "low-start", "low-end", "high-start", "high-end"}) {
		t.Fatalf("markers: %v", fixture.markers)
	}
	if fixture.loaded || !slices.Equal(fixture.requests, []string{"/models/unload", "/models/load", "/models/unload"}) {
		t.Fatalf("cold baseline/load/cleanup: loaded=%v requests=%v", fixture.loaded, fixture.requests)
	}
}

func TestModelDiagnosticPostLoadKeyMismatchRestoresWithoutExercise(t *testing.T) {
	f, manager, op, fixture := modelDiagnosticRunningOperationFixture(t)
	fixture.beforeLoad = func(*http.Request) bool {
		f.st.mu.Lock()
		f.st.plat.Runtime.ModelLoads[0].Requested.Context = &modelconfig.Context{Mode: "fixed", Tokens: 8192}
		f.st.mu.Unlock()
		return false
	}
	manager.runDiagnosticOperation(op, false)
	status := f.ex.diagnosticRunStatus(op.params.RunID)
	if status.Outcome != "failed" || status.FailedPhase != "loading" || !status.WorkStopped || status.Low != nil || status.High != nil {
		t.Fatalf("mismatched load status: %+v", status)
	}
	fixture.mu.Lock()
	loaded := fixture.loaded
	fixture.mu.Unlock()
	if loaded {
		t.Fatal("key mismatch left the measured model loaded")
	}
}

func TestModelDiagnosticPauseJoinsNativeRequests(t *testing.T) {
	for _, phase := range []string{"load", "token counting", "generation"} {
		t.Run(phase, func(t *testing.T) {
			_, manager, op, fixture := modelDiagnosticRunningOperationFixture(t)
			entered, ended := make(chan struct{}), make(chan struct{})
			release := make(chan struct{})
			t.Cleanup(func() { close(release) })
			block := func(request *http.Request) bool {
				// Consume the request so the HTTP server can observe a closed
				// connection while the synthetic native operation is blocked.
				if _, err := io.Copy(io.Discard, request.Body); err != nil {
					t.Errorf("read blocked request body: %v", err)
					return true
				}
				close(entered)
				select {
				case <-request.Context().Done():
					close(ended)
				case <-release:
				}
				return true
			}
			if phase == "load" {
				fixture.beforeLoad = block
			} else if phase == "token counting" {
				fixture.beforeCount = block
			} else {
				fixture.fixture.onComplete = func(_ http.ResponseWriter, request *http.Request, _ modelDiagnosticChatRequest) { block(request) }
			}
			go manager.runDiagnosticOperation(op, false)
			modelDiagnosticWait(t, entered, "in-flight "+phase)
			projection := nodeProjection{ManagementBlocked: true, PausedLocally: true, Policy: nodepolicy.Advertisement{PauseMode: nodepolicy.PauseImmediate}}
			raw, err := json.Marshal(projection)
			if err != nil {
				t.Fatalf("encode Halt projection: %v", err)
			}
			manager.applyNodeProjection(&Message{Params: raw})
			status, err := manager.abortDiagnostic(diagnosticReleaseParams{op.params.RunID, op.params.Token})
			if err != nil {
				t.Fatalf("join %s: %v", phase, err)
			}
			modelDiagnosticWait(t, ended, "cancelled HTTP request")
			if !status.WorkStopped || status.Outcome != "cancelled" {
				t.Fatalf("Halt join during %s: %+v", phase, status)
			}
			fixture.mu.Lock()
			loaded := fixture.loaded
			fixture.mu.Unlock()
			if loaded {
				t.Fatal("cancelled diagnostic retained a loaded model")
			}
			wantRequests := 0
			if phase == "generation" {
				wantRequests = 1
			}
			if len(fixture.fixture.completions) != wantRequests {
				t.Fatal("generation dispatched after cancellation")
			}
		})
	}
}

func TestModelDiagnosticChildExitKeepsTypedReasonAndCleansUp(t *testing.T) {
	f, manager, op, fixture := modelDiagnosticRunningOperationFixture(t)
	var once sync.Once
	fixture.beforeCount = func(*http.Request) bool { once.Do(func() { close(f.st.proc.done) }); return false }
	manager.runDiagnosticOperation(op, false)
	status := f.ex.diagnosticRunStatus(op.params.RunID)
	if !status.WorkStopped || status.Outcome != "failed" || status.Reason != "engine-exited" {
		t.Fatalf("child exit status: %+v", status)
	}
	fixture.mu.Lock()
	loaded := fixture.loaded
	fixture.mu.Unlock()
	if loaded {
		t.Fatal("child exit left loaded work unconfirmed")
	}
}

func TestModelDiagnosticConfiguredPauseRefreshDoesNotCancelExistingConsent(t *testing.T) {
	f, manager, op := modelDiagnosticOperationFixture(t, func(f inspectionFixture) {
		f.ex.nodeProjection.set(nodeProjection{ManagementBlocked: true, PausedLocally: true, Policy: nodepolicy.Advertisement{PauseMode: nodepolicy.PauseConfigured}})
	})
	projection := *f.ex.nodeProjection.projection
	params, err := json.Marshal(projection)
	if err != nil {
		t.Fatalf("encode configured pause: %v", err)
	}
	manager.applyNodeProjection(&Message{Params: params})
	if op.ctx.Err() != nil {
		t.Fatalf("unchanged configured pause cancelled consent: %v", op.ctx.Err())
	}
	ctx, err := f.ex.diagnosticContext(op.ctx, op.params.Engine, op.params.RunID, op.params.Token)
	if err != nil {
		t.Fatalf("scope diagnostic: %v", err)
	}
	ctx = f.ex.activationDispatchContext(ctx, op.params.Engine)
	finish, err := beginActivationEffect(ctx)
	if err != nil {
		t.Fatalf("final dispatch vetoed configured pause authority: %v", err)
	}
	finish()
	if mode := f.ex.nodeProjection.advertisement().PauseMode; mode != nodepolicy.PauseConfigured {
		t.Fatalf("dispatch changed pause to %q", mode)
	}
}

func TestModelDiagnosticExistingConfiguredPauseAllowsWakeLoadAndExercise(t *testing.T) {
	f, manager, op, _ := modelDiagnosticRunningOperationFixture(t, func(f inspectionFixture) {
		f.ex.nodeProjection.set(nodeProjection{ManagementBlocked: true, PausedLocally: true, Policy: nodepolicy.Advertisement{PauseMode: nodepolicy.PauseConfigured}})
	})
	manager.runDiagnosticOperation(op, false)
	status := f.ex.diagnosticRunStatus(op.params.RunID)
	if status.Outcome != "completed" || !status.WorkStopped || status.Low == nil || status.High == nil {
		t.Fatalf("configured-pause measurement: %+v", status)
	}
	if mode := f.ex.nodeProjection.advertisement().PauseMode; mode != nodepolicy.PauseConfigured {
		t.Fatalf("measurement changed configured pause to %q", mode)
	}
}

func TestModelDiagnosticStatusCarriesBrokerJoinContract(t *testing.T) {
	f, _, op := modelDiagnosticOperationFixture(t, nil)
	raw, err := json.Marshal(f.ex.diagnosticRunStatus(op.params.RunID))
	if err != nil {
		t.Fatalf("encode status: %v", err)
	}
	var reply struct {
		State       string                     `json:"state"`
		Token       string                     `json:"token"`
		Run         *modelconfig.DiagnosticRun `json:"run"`
		WorkStopped bool                       `json:"workStopped"`
	}
	if err := json.Unmarshal(raw, &reply); err != nil {
		t.Fatalf("decode broker status: %v", err)
	}
	if reply.State != "accepted" || reply.Token != op.params.Token || reply.Run == nil || reply.Run.RunID != op.params.RunID || reply.WorkStopped {
		t.Fatalf("broker join contract: %+v", reply)
	}
	if limit, ok := op.ctx.Deadline(); !ok || time.Until(limit) > 20*time.Minute {
		t.Fatalf("operation lacks manager-owned bound: %v, known=%v", limit, ok)
	}
}
