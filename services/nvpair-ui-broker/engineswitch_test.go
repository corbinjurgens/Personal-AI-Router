// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrusttest"
	"nvpair-shared/durablefile"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

type switchHarness struct {
	b                  *Broker
	engine, proxy      *fakeWorker
	pp                 *proxyProcess
	output             *syncBuffer
	mu                 sync.Mutex
	cp                 *settings.SwitchCheckpoint
	config             settings.SwitchConfigurationSnapshot
	active             atomic.Int32
	runningTarget      bool
	sleeping           map[string]bool
	failStep           string
	publicationFailure bool
}

func newSwitchHarness(t *testing.T) *switchHarness {
	t.Helper()
	output := &syncBuffer{}
	b := &Broker{nodeID: "target", codec: NewCodec(output), clusterDir: t.TempDir()}
	c := b.switchCoordinator()
	c.store = newEngineSwitchStore(filepath.Join(t.TempDir(), "engine-switch-operations.json"))
	if err := c.store.Load(); err != nil {
		t.Fatal(err)
	}
	h := &switchHarness{b: b, output: output, sleeping: make(map[string]bool), config: settings.SwitchConfigurationSnapshot{ManagerIncarnation: "manager-run", RegistryRuntimeFingerprint: "runtime", Engines: []settings.SwitchConfigurationEngine{{Engine: "llamacpp", Intent: settings.SwitchIntentOn}, {Engine: "lmstudio", Intent: settings.SwitchIntentOff}}}}
	h.engine = attachFakeEngine(t, b)
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		h.mu.Lock()
		defer h.mu.Unlock()
		result := settings.SwitchPrepareResult{}
		if h.cp != nil {
			copy := *h.cp
			result.Checkpoint = &copy
			result.Observed = settings.SwitchObserved{Source: copy.Source, Target: copy.Target}
		}
		switch method {
		case "engine:get-installed":
			engines := make([]map[string]any, 0, len(h.config.Engines))
			for _, config := range h.config.Engines {
				running := config.Intent == settings.SwitchIntentOn || (config.Engine == "lmstudio" && h.runningTarget)
				if h.cp != nil {
					state := h.cp.Source
					if config.Engine == h.cp.TargetEngine {
						state = h.cp.Target
					}
					if state.Running != nil {
						running = *state.Running
					}
				}
				engines = append(engines, map[string]any{"engine": config.Engine, "running": running && !h.sleeping[config.Engine]})
			}
			return map[string]any{"engines": engines}, ""
		case "engine:models":
			return map[string]any{"models": []string{}, "loadedByEngine": map[string][]string{}}, ""
		case engineSleepMethod:
			var p struct {
				Engine string `json:"engine"`
			}
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if h.cp != nil && h.cp.Phase != settings.SwitchCheckpointReleased && (p.Engine == h.cp.SourceEngine || p.Engine == h.cp.TargetEngine) {
				return nil, "engine is reserved by an engine switch (busy)"
			}
			h.sleeping[p.Engine] = true
			if h.cp != nil {
				if p.Engine == h.cp.SourceEngine {
					h.cp.Source = switchTestState(false, false, h.cp.Source.Intent)
				} else if p.Engine == h.cp.TargetEngine {
					h.cp.Target = switchTestState(false, false, h.cp.Target.Intent)
				}
			}
			return map[string]any{"engine": p.Engine, "running": false}, ""
		case "engine:stop":
			var p struct {
				Engine     string `json:"engine"`
				VerifyStop bool   `json:"verifyStop"`
			}
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if !p.VerifyStop {
				return map[string]bool{"ok": true}, ""
			}
			state := switchTestState(false, false, settings.SwitchIntentOff)
			if h.cp != nil {
				if p.Engine == h.cp.SourceEngine {
					h.cp.Source = state
				} else if p.Engine == h.cp.TargetEngine {
					h.cp.Target = state
				}
			}
			return state, ""
		case settings.SwitchPrepareMethod:
			var p settings.SwitchPrepareRequest
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if p.OperationID == "" {
				return switchManagerSnapshot{SwitchPrepareResult: result, Configuration: h.config, ConfigurationFence: settings.SwitchConfigurationToken(h.config)}, ""
			}
			if p.ConfigurationFence != settings.SwitchConfigurationToken(h.config) {
				return nil, "public token passed instead of manager fence"
			}
			data, err := os.ReadFile(c.store.path)
			if err != nil {
				return nil, err.Error()
			}
			journal, err := settings.DecodeSwitchJournal(data)
			if err != nil {
				return nil, err.Error()
			}
			if journal.Receipts[p.OperationID].Phase != settings.SwitchValidating {
				return nil, "prepare before durable validating phase"
			}
			if h.runningTarget {
				result.Error = &settings.SwitchError{Code: -32000, Kind: settings.SwitchErrorTargetRunning, Message: "Target is already running."}
				return result, ""
			}
			source := switchTestState(true, true, settings.SwitchIntentOn)
			target := switchTestState(false, false, settings.SwitchIntentOff)
			h.cp = &settings.SwitchCheckpoint{SwitchPrepareRequest: p, Phase: settings.SwitchCheckpointReserved, Source: source, Target: target, PreSource: source, PreTarget: target}
			result.Checkpoint = h.cp
			result.Observed = settings.SwitchObserved{Source: source, Target: target}
			return result, ""
		case settings.SwitchStepMethod:
			var p settings.SwitchStepRequest
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if h.cp == nil || p.SwitchPrepareRequest != h.cp.SwitchPrepareRequest {
				return nil, "executor envelope changed"
			}
			data, err := os.ReadFile(c.store.path)
			if err != nil {
				return nil, err.Error()
			}
			journal, err := settings.DecodeSwitchJournal(data)
			if err != nil {
				return nil, err.Error()
			}
			wantPhase := settings.SwitchSourceStopping
			if p.Step == settings.SwitchStepTargetStart {
				wantPhase = settings.SwitchTargetStarting
			}
			if p.Step == settings.SwitchStepRelease {
				if !journal.Receipts[p.OperationID].Terminal {
					return nil, "release before durable terminal"
				}
				h.cp.Phase = settings.SwitchCheckpointReleased
			} else {
				if journal.Receipts[p.OperationID].Phase != wantPhase {
					return nil, "step before durable phase"
				}
				if p.Step == h.failStep {
					result.Error = &settings.SwitchError{Code: -32011, Kind: settings.SwitchErrorRemoteVeto, Message: "Local Off veto."}
					h.cp.Error = result.Error
					result.Checkpoint = h.cp
					return result, ""
				}
				if p.Step == settings.SwitchStepSourceStop {
					h.cp.Phase = settings.SwitchCheckpointSourceOff
					h.cp.Source = switchTestState(false, false, settings.SwitchIntentOff)
				}
				if p.Step == settings.SwitchStepTargetStart {
					h.cp.Phase = settings.SwitchCheckpointTargetCommitted
					h.cp.TargetCommitted = true
					h.cp.Target = switchTestState(true, true, settings.SwitchIntentOn)
				}
			}
			return settings.SwitchPrepareResult{Checkpoint: h.cp, Observed: settings.SwitchObserved{Source: h.cp.Source, Target: h.cp.Target}}, ""
		case settings.SwitchReconcileMethod:
			var p settings.SwitchReconcileRequest
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if p.LocalVeto != nil && h.cp != nil && h.cp.Phase != settings.SwitchCheckpointReleased {
				h.cp.Error = &settings.SwitchError{Code: settings.SwitchRemoteVetoCode, Kind: settings.SwitchErrorRemoteVeto, Message: "Local Off or pause veto."}
				h.cp.TargetCommitted = false
				h.cp.Phase = settings.SwitchCheckpointSourceOff
				h.cp.Target = switchTestState(false, false, settings.SwitchIntentOff)
				copy := *h.cp
				return settings.SwitchReconcileResult{Checkpoint: &copy, Observed: settings.SwitchObserved{Source: copy.Source, Target: copy.Target}, Error: copy.Error}, ""
			}
			return result, ""
		case "engine:get-launch":
			return settings.LaunchState{Engine: "lmstudio", Running: true, EffectivePort: 12345}, ""
		case "engine:switch-recovery-ready":
			var p struct {
				ManagerIncarnation string `json:"managerIncarnation"`
			}
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			return p, ""
		default:
			return map[string]bool{"ok": true}, ""
		}
	})
	h.pp, h.proxy = attachFakeProxy(t, b)
	h.pp.facadeState = map[string]proxyFacadeState{"llamacpp": {ready: true, port: 12346}, "lmstudio": {ready: true, port: 12347}}
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case nodepolicy.MethodInvalidateEngine:
			return nodepolicy.InvalidateEngineResult{OK: true, RunID: "proxy-run"}, ""
		case nodepolicy.MethodAdmissionSnapshot:
			return nodepolicy.AdmissionSnapshotResult{RunID: "proxy-run", AdmissionState: nodepolicy.AdmissionState{Active: int(h.active.Load()), ActiveByEngine: map[string]int{"llamacpp": int(h.active.Load())}}}, ""
		case "lmstudio:node/set-local-backend":
			if h.publicationFailure {
				return nil, "publication refused"
			}
		}
		return map[string]bool{"ok": true}, ""
	})
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.incarnation, r.runID = h.pp, "proxy-run"
	r.mu.Unlock()
	t.Cleanup(func() { h.waitIdle(t) })
	return h
}

func TestSwitchRecoveryReadyAcknowledgementPrecedesManagerPublication(t *testing.T) {
	h := newSwitchHarness(t)
	w := hideSwitchRecoveryManager(t, h.b)
	base := h.engine.answer
	reached, resume := make(chan struct{}), make(chan struct{})
	var once sync.Once
	open := func() { once.Do(func() { close(resume) }) }
	t.Cleanup(open)
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:switch-recovery-ready" {
			close(reached)
			<-resume
		}
		return base(method, raw)
	})
	done := make(chan bool, 1)
	go func() { done <- h.b.recoverReplacementEngineMgr(w) }()
	requireGateOpens(t, reached, "manager readiness acknowledgement")
	if h.b.getEngineMgr() != nil {
		t.Fatal("broker published the manager before readiness was acknowledged")
	}
	open()
	select {
	case ok := <-done:
		if !ok || h.b.getEngineMgr() != w {
			t.Fatal("successful readiness acknowledgement did not publish the recovered manager")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("recovery did not return after readiness acknowledgement")
	}
}

func TestSwitchRecoveryBlockedManagerDoesNotReceiveReadiness(t *testing.T) {
	h := newSwitchHarness(t)
	w := hideSwitchRecoveryManager(t, h.b)
	base := h.engine.answer
	var ready atomic.Int32
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			return settings.SwitchPrepareResult{Error: &settings.SwitchError{
				Code: settings.SwitchOperationErrorCode, Kind: settings.SwitchErrorStorage, Message: "checkpoint unreadable",
			}}, ""
		}
		if method == "engine:switch-recovery-ready" {
			ready.Add(1)
		}
		return base(method, raw)
	})
	if h.b.recoverReplacementEngineMgr(w) || h.b.getEngineMgr() != nil {
		t.Fatal("blocked recovery published the manager")
	}
	if got := ready.Load(); got != 0 {
		t.Fatalf("blocked recovery sent %d readiness acknowledgements", got)
	}
}

func TestSwitchRecoveryReadyRejectsDifferentAcknowledgementIncarnation(t *testing.T) {
	h := newSwitchHarness(t)
	w := hideSwitchRecoveryManager(t, h.b)
	base := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:switch-recovery-ready" {
			return map[string]string{"managerIncarnation": "another-manager"}, ""
		}
		return base(method, raw)
	})
	if h.b.recoverEngineSwitchOn(w) || h.b.getEngineMgr() != nil {
		t.Fatal("a different acknowledgement incarnation published the manager")
	}
	h.b.engineConfigMu.Lock()
	err := h.b.switchCoordinator().recoveryErr
	h.b.engineConfigMu.Unlock()
	if err == nil {
		t.Fatal("missing recovery failure after mismatched readiness acknowledgement")
	}
}

func switchTestState(running, healthy bool, intent string) settings.SwitchEngineState {
	now := time.Now().UTC()
	return settings.SwitchEngineState{Intent: intent, Running: &running, Healthy: &healthy, ObservedAt: &now, TerminationVerified: !running}
}

func (h *switchHarness) request(t *testing.T, id string) settings.SwitchRequest {
	t.Helper()
	status, err := h.b.engineSwitchStatus(context.Background(), settings.SwitchStatusRequest{}, "")
	if err != nil {
		t.Fatal(err)
	}
	return settings.SwitchRequest{RequestID: id, SourceEngine: "llamacpp", TargetEngine: "lmstudio", ExpectedRevision: status.ConfigurationRevision}
}

func (h *switchHarness) waitIdle(t *testing.T) {
	t.Helper()
	done := make(chan struct{})
	go func() { h.b.settingsApplyMu.Lock(); h.b.settingsApplyMu.Unlock(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("switch did not finish")
	}
}

func (h *switchHarness) receipt(t *testing.T, id string) settings.SwitchReceipt {
	t.Helper()
	h.waitIdle(t)
	h.b.engineConfigMu.Lock()
	defer h.b.engineConfigMu.Unlock()
	r, ok := h.b.engineSwitch.store.Lookup(id)
	if !ok {
		t.Fatalf("missing receipt %s", id)
	}
	return r
}

func switchSteps(t *testing.T, worker *fakeWorker) []string {
	t.Helper()
	var out []string
	for _, call := range worker.snapshot() {
		if call.method != settings.SwitchStepMethod {
			continue
		}
		var p settings.SwitchStepRequest
		if err := json.Unmarshal(call.params, &p); err != nil {
			t.Fatal(err)
		}
		out = append(out, p.Step)
	}
	return out
}

func TestSwitchCoordinatorCompletesAfterDurablePhasesAndPublication(t *testing.T) {
	h := newSwitchHarness(t)
	p := h.request(t, "complete")
	accepted, err := h.b.acceptEngineSwitch(context.Background(), p, "")
	if err != nil {
		t.Fatal(err)
	}
	if accepted.Receipt.Phase != settings.SwitchAccepted {
		t.Fatalf("acceptance = %+v", accepted)
	}
	r := h.receipt(t, p.RequestID)
	if r.Phase != settings.SwitchCompleted || r.RecoveryRequired || !r.Source.TerminationVerified || r.Target.Intent != settings.SwitchIntentOn {
		t.Fatalf("completion = %+v", r)
	}
	want := []string{settings.SwitchStepSourceStop, settings.SwitchStepTargetStart, settings.SwitchStepRelease}
	if got := switchSteps(t, h.engine); !reflect.DeepEqual(got, want) {
		t.Fatalf("steps = %v, want %v", got, want)
	}
	var sequences []uint64
	for _, frame := range h.output.frames() {
		if frame.Method != settings.SwitchChangedNotification {
			continue
		}
		var changed settings.SwitchChanged
		if err := json.Unmarshal(frame.Params, &changed); err != nil {
			t.Fatal(err)
		}
		if changed.NodeID != "target" || changed.RequestID != p.RequestID {
			t.Fatalf("notification = %+v", changed)
		}
		sequences = append(sequences, changed.Sequence)
	}
	if !reflect.DeepEqual(sequences, []uint64{1, 2, 3, 4, 5, 6, 7, 8}) {
		t.Fatalf("sequences = %v", sequences)
	}
	// Configuration generation changes, but identical replay wins before CAS.
	replayed, err := h.b.acceptEngineSwitch(context.Background(), p, "")
	if err != nil || !reflect.DeepEqual(replayed.Receipt, r) {
		t.Fatalf("replay = %+v, %v", replayed, err)
	}
	if got := switchSteps(t, h.engine); !reflect.DeepEqual(got, want) {
		t.Fatalf("replay repeated steps: %v", got)
	}
}

func TestSettingsServingFullSwitchPreservesDeclarations(t *testing.T) {
	test := func(name, failStep string) {
		t.Run(name, func(t *testing.T) {
			h := newSwitchHarness(t)
			h.failStep = failStep
			serving := brokerServingFixture()
			base := h.engine.answer
			h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method != "engine:get-launch" {
					return base(method, raw)
				}
				var p struct {
					Engine string `json:"engine"`
				}
				if err := json.Unmarshal(raw, &p); err != nil {
					return nil, err.Error()
				}
				return settings.LaunchState{Engine: p.Engine, ServerPort: 12345, EffectivePort: 12345, Running: true, Editable: true, Format: "pair-arguments-v1", Serving: serving.Clone()}, ""
			})
			for _, engine := range []string{"llamacpp", "lmstudio"} {
				snapshot, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: engine}, "")
				if err != nil || !snapshot.Settings.Serving.Equal(serving) {
					t.Fatal("switch baseline lost serving")
				}
			}
			request := h.request(t, "serving-switch")
			if _, err := h.b.acceptEngineSwitch(context.Background(), request, ""); err != nil {
				t.Fatal(err)
			}
			receipt := h.receipt(t, request.RequestID)
			if !receipt.Terminal || (failStep == "" && receipt.Phase != settings.SwitchCompleted) || (failStep != "" && receipt.Phase == settings.SwitchCompleted) {
				t.Fatalf("unexpected switch phase %s", receipt.Phase)
			}
			for _, engine := range []string{"llamacpp", "lmstudio"} {
				snapshot, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: engine}, "")
				if err != nil || !snapshot.Settings.Serving.Equal(serving) {
					t.Fatal("switch completion/compensation lost declaration")
				}
			}
		})
	}
	test("completed switch", "")
	test("failed target start compensation", settings.SwitchStepTargetStart)
}

func TestSwitchCoordinatorRejectsUnroutableTargetBeforeDrain(t *testing.T) {
	h := newSwitchHarness(t)
	p := h.request(t, "unroutable")
	p.TargetEngine = "custom-llama"
	result, err := h.b.acceptEngineSwitch(context.Background(), p, "")
	if err == nil || result.Receipt.Phase != settings.SwitchRejected || result.Receipt.Error.Kind != settings.SwitchErrorUnroutable {
		t.Fatalf("rejection = %+v, %v", result, err)
	}
	if len(h.proxy.snapshot()) != 0 || len(switchSteps(t, h.engine)) != 0 {
		t.Fatal("unroutable request caused effects")
	}
}

func TestSwitchCoordinatorRejectsRunningTargetBeforeDrain(t *testing.T) {
	h := newSwitchHarness(t)
	h.runningTarget = true
	p := h.request(t, "running-target")
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	r := h.receipt(t, p.RequestID)
	if r.Phase != settings.SwitchFailedValidation || r.Error.Kind != settings.SwitchErrorTargetRunning {
		t.Fatalf("receipt = %+v", r)
	}
	for _, call := range h.proxy.snapshot() {
		if call.method != nodepolicy.MethodSetModelIdentity {
			t.Fatalf("running target caused proxy effect %s", call.method)
		}
	}
	if len(switchSteps(t, h.engine)) != 0 {
		t.Fatal("running target caused lifecycle effects")
	}
}

func TestSwitchCoordinatorInspectsSuccessfulRPCErrorEnvelope(t *testing.T) {
	h := newSwitchHarness(t)
	h.failStep = settings.SwitchStepTargetStart
	p := h.request(t, "veto-envelope")
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	r := h.receipt(t, p.RequestID)
	if r.Phase != settings.SwitchFailedTargetStart || r.Error.Code != -32011 || r.Error.Kind != settings.SwitchErrorRemoteVeto {
		t.Fatalf("receipt = %+v", r)
	}
	if r.Target.Intent == settings.SwitchIntentOn {
		t.Fatal("failed start committed target intent")
	}
}

func TestSwitchCoordinatorPersistenceBoundariesHaltEffects(t *testing.T) {
	for _, phase := range []string{settings.SwitchAccepted, settings.SwitchValidating, settings.SwitchDraining, settings.SwitchSourceStopping, settings.SwitchSourceOff, settings.SwitchTargetStarting, settings.SwitchTargetReady, settings.SwitchCompleted} {
		for _, replaced := range []bool{false, true} {
			name := phase + " before replacement"
			if replaced {
				name = phase + " after replacement"
			}
			t.Run(name, func(t *testing.T) {
				h := newSwitchHarness(t)
				p := h.request(t, "write-failure")
				s := h.b.engineSwitch.store
				s.write = func(path string, data []byte, mode os.FileMode) error {
					journal, err := settings.DecodeSwitchJournal(data)
					if err != nil {
						return err
					}
					if journal.Receipts[p.RequestID].Phase == phase {
						if replaced {
							if err := durablefile.WriteFile(path, data, mode); err != nil {
								return err
							}
							return fmt.Errorf("test durability: %w", durablefile.ErrReplaced)
						}
						return errors.New("test write failure")
					}
					return durablefile.WriteFile(path, data, mode)
				}
				_, err := h.b.acceptEngineSwitch(context.Background(), p, "")
				if phase == settings.SwitchAccepted && err == nil {
					t.Fatal("acceptance ignored persistence failure")
				}
				h.waitIdle(t)
				h.b.engineConfigMu.Lock()
				blocked := h.b.engineSwitch.err
				r, exists := s.Lookup(p.RequestID)
				h.b.engineConfigMu.Unlock()
				if blocked == nil {
					t.Fatal("persistence failure did not fence operations")
				}
				if replaced && (!exists || r.Phase != phase) {
					t.Fatalf("landed state not adopted: %+v", r)
				}
				steps := switchSteps(t, h.engine)
				for _, step := range steps {
					if step == settings.SwitchStepRelease {
						t.Fatal("released manager after uncertain terminal write")
					}
					if step == settings.SwitchStepSourceStop && (phase == settings.SwitchAccepted || phase == settings.SwitchValidating || phase == settings.SwitchDraining || phase == settings.SwitchSourceStopping) {
						t.Fatal("source stopped after failed pre-stop write")
					}
					if step == settings.SwitchStepTargetStart && phase != settings.SwitchTargetReady && phase != settings.SwitchCompleted {
						t.Fatal("target started after earlier write failure")
					}
				}
			})
		}
	}
}

func TestSwitchStatusWaitDoesNotHoldOperationOrConfigurationLock(t *testing.T) {
	h := newSwitchHarness(t)
	h.active.Store(1)
	p := h.request(t, "status-wait")
	ctx, cancel := context.WithCancel(context.Background())
	accepted, err := h.b.acceptEngineSwitch(ctx, p, "")
	if err != nil {
		t.Fatal(err)
	}
	cancel() // caller cancellation cannot cancel accepted work
	h.proxy.waitFor("fresh admission snapshot", func(calls []policyCall) bool {
		for _, call := range calls {
			if call.method == nodepolicy.MethodAdmissionSnapshot {
				return true
			}
		}
		return false
	})
	status, err := h.b.engineSwitchStatus(context.Background(), settings.SwitchStatusRequest{RequestID: p.RequestID}, "")
	if err != nil || status.Receipt == nil || status.Receipt.Terminal {
		t.Fatalf("running status = %+v, %v", status, err)
	}
	after := status.Receipt.Sequence
	done := make(chan settings.SwitchStatus, 1)
	go func() {
		got, err := h.b.engineSwitchStatus(context.Background(), settings.SwitchStatusRequest{RequestID: p.RequestID, AfterSequence: &after}, "")
		if err != nil {
			t.Error(err)
		}
		done <- got
	}()
	if accepted.Receipt.Terminal {
		t.Fatal("accepted receipt terminal")
	}
	h.active.Store(0)
	select {
	case next := <-done:
		if next.Receipt == nil || next.Receipt.Sequence <= after {
			t.Fatalf("wait result = %+v", next)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("status waiter blocked advancement")
	}
	if r := h.receipt(t, p.RequestID); r.Phase != settings.SwitchCompleted {
		t.Fatalf("caller cancellation aborted operation: %+v", r)
	}
}

func TestSwitchRecoveryInterruptsBeforeSettingsReplayAndRestore(t *testing.T) {
	h := newSwitchHarness(t)
	p := h.request(t, "interrupted")
	r := switchStoreReceipt(p.RequestID)
	r.NodeID, p.NodeID = h.b.nodeID, h.b.nodeID
	r.Request, r.ExpectedRevision, r.AcceptedRevision = p, p.ExpectedRevision, p.ExpectedRevision
	r.PayloadHash = settings.SwitchPayloadHash("local", h.b.nodeID, p)
	switchStoreAccept(t, h.b.engineSwitch.store, r)
	envelope := settings.SwitchPrepareRequest{OperationID: p.RequestID, Caller: "local", Origin: "local", SourceEngine: p.SourceEngine, TargetEngine: p.TargetEngine, PayloadHash: r.PayloadHash, ConfigurationFence: settings.SwitchConfigurationToken(h.config), Deadline: r.Deadline}
	h.mu.Lock()
	h.cp = &settings.SwitchCheckpoint{SwitchPrepareRequest: envelope, Phase: settings.SwitchCheckpointReserved, Source: r.Source, Target: r.Target}
	h.mu.Unlock()
	h.b.restoreEnabledEngines(h.b.getEngineMgr())
	receipt := h.receipt(t, p.RequestID)
	if receipt.Phase != settings.SwitchFailedInterrupted || !receipt.RecoveryRequired {
		t.Fatalf("recovery receipt = %+v", receipt)
	}
	steps := switchSteps(t, h.engine)
	if !reflect.DeepEqual(steps, []string{settings.SwitchStepRelease}) {
		t.Fatalf("recovery resumed lifecycle: %v", steps)
	}
	calls := h.engine.snapshot()
	var reconciled, released bool
	for _, call := range calls {
		if call.method == settings.SwitchReconcileMethod {
			reconciled = true
		}
		if call.method == settings.SwitchStepMethod {
			released = true
		}
		if call.method == restoreEnabledEnginesMethod && (!reconciled || !released) {
			t.Fatal("restore preceded switch reconciliation")
		}
	}
}

func TestSwitchRecoveryConflictingSettingsJournalFailsClosed(t *testing.T) {
	h := newSwitchHarness(t)
	r := switchStoreReceipt("conflict")
	r.Request.SourceEngine = "llamacpp"
	r.PayloadHash = settings.SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
	switchStoreAccept(t, h.b.engineSwitch.store, r)
	h.b.engineSettingsLoaded = true
	h.b.engineSettings = map[string]*engineSettingsRecord{"llamacpp": {Snapshot: settings.Snapshot{Phase: "applying"}, Resume: true}}
	h.b.restoreEnabledEngines(h.b.getEngineMgr())
	if h.b.engineSwitch.err == nil {
		t.Fatal("conflicting journals were not fenced")
	}
	for _, call := range h.engine.snapshot() {
		if call.method == "engine:configure-launch" || call.method == restoreEnabledEnginesMethod || call.method == settings.SwitchStepMethod {
			t.Fatalf("conflict caused %s", call.method)
		}
	}
}

func TestSwitchPauseRecoveryReleasesBeforePinnedShutdown(t *testing.T) {
	test := func(name string, origin nodepolicy.PauseOrigin) {
		t.Run(name, func(t *testing.T) {
			h := newSwitchHarness(t)
			p := h.request(t, "paused-recovery")
			p.NodeID = h.b.nodeID
			r := switchStoreReceipt(p.RequestID)
			r.NodeID, r.Caller = h.b.nodeID, "peer"
			r.Request, r.ExpectedRevision, r.AcceptedRevision = p, p.ExpectedRevision, p.ExpectedRevision
			r.PayloadHash = settings.SwitchPayloadHash(r.Caller, r.NodeID, p)
			r.Source = switchTestState(true, true, settings.SwitchIntentOn)
			r.Target = switchTestState(false, false, settings.SwitchIntentOff)
			switchStoreAccept(t, h.b.engineSwitch.store, r)
			if err := h.b.engineSwitch.store.SaveLocalObligation(p.RequestID, settings.SwitchLocalObligation{Pause: true}); err != nil {
				t.Fatalf("save pause obligation: %v", err)
			}
			h.mu.Lock()
			h.cp = &settings.SwitchCheckpoint{SwitchPrepareRequest: settings.SwitchPrepareRequest{
				OperationID: p.RequestID, Caller: r.Caller, Origin: settings.SwitchOriginRemote,
				SourceEngine: p.SourceEngine, TargetEngine: p.TargetEngine, PayloadHash: r.PayloadHash,
				ConfigurationFence: settings.SwitchConfigurationToken(h.config), Deadline: r.Deadline,
			}, Phase: settings.SwitchCheckpointReserved, Source: r.Source, Target: r.Target}
			h.mu.Unlock()
			runtime := h.b.nodePolicyRuntime()
			runtime.mu.Lock()
			runtime.current.Availability, runtime.current.PausedBy = nodepolicy.Paused, origin
			runtime.live = nodepolicy.Paused
			runtime.mu.Unlock()
			worker := h.b.getEngineMgr()
			h.b.workersMu.Lock()
			h.b.engineMgrRecovering = true
			h.b.workersMu.Unlock()
			if h.b.getEngineMgr() != nil {
				t.Fatal("replacement was exposed before conservative recovery")
			}
			if !h.b.recoverEngineSwitchOn(worker) {
				t.Fatal("paused replacement could not release its reservation and verify shutdown")
			}
			o := switchStoreDisk(t, h.b.engineSwitch.store).LocalObligations[p.RequestID]
			if !o.PauseApplied || !o.Discharged || !h.b.engineSwitch.resolved[p.RequestID] {
				t.Fatalf("physical pause obligation not settled: %+v", o)
			}
			h.mu.Lock()
			cp := *h.cp
			sourceSlept, targetSlept := h.sleeping[p.SourceEngine], h.sleeping[p.TargetEngine]
			h.mu.Unlock()
			if cp.Phase != settings.SwitchCheckpointReleased || !sourceSlept || !targetSlept || cp.Source.Intent != settings.SwitchIntentOn {
				t.Fatalf("release or intent-neutral shutdown incomplete: checkpoint=%+v sourceSlept=%v targetSlept=%v", cp, sourceSlept, targetSlept)
			}
			var released, slept bool
			for _, call := range h.engine.snapshot() {
				if call.method == settings.SwitchStepMethod {
					var step settings.SwitchStepRequest
					if err := json.Unmarshal(call.params, &step); err != nil {
						t.Fatalf("decode switch step: %v", err)
					}
					released = released || step.Step == settings.SwitchStepRelease
				}
				if call.method == engineSleepMethod {
					if !released {
						t.Fatal("physical shutdown entered before switch reservation release")
					}
					slept = true
				}
				if call.method == "engine:switch-recovery-ready" && !slept {
					t.Fatal("replacement admitted peer writes before physical pause")
				}
			}
		})
	}
	test("local pause", nodepolicy.PausedLocally)
	test("peer pause", nodepolicy.PausedByPeer)
}

func TestSwitchRelayCancellationRetiresOnlyWaiter(t *testing.T) {
	h := newSwitchHarness(t)
	ctx, cancel := context.WithCancel(context.Background())
	c := h.b.switchCoordinator()
	c.waitMu.Lock()
	c.waiters["waiter"] = cancel
	c.waitMu.Unlock()
	operation, operationCancel := context.WithCancel(context.Background())
	defer operationCancel()
	c.waitMu.Lock()
	c.remoteCancel = operationCancel
	c.waitMu.Unlock()
	if !h.b.handleSwitchEngineNotification(bindRemoteManagementTestSource(h.b), "engine:switch-cancel", settingsJSON(map[string]string{"id": "waiter"})) {
		t.Fatal("cancel not handled")
	}
	if ctx.Err() == nil || operation.Err() != nil {
		t.Fatal("relay cancel affected accepted operation")
	}
}

func TestFailedStartupPauseAllowsLocalRecovery(t *testing.T) {
	test := func(name string, state nodepolicy.Availability) {
		t.Run(name, func(t *testing.T) {
			h := newSwitchHarness(t)
			runtime := h.b.nodePolicyRuntime()
			runtime.mu.Lock()
			runtime.current.Availability, runtime.current.PausedBy = nodepolicy.Paused, nodepolicy.PausedByPeer
			runtime.live = nodepolicy.Paused
			runtime.mu.Unlock()
			h.b.observeAdmission(nodepolicy.AdmissionState{ActiveByEngine: map[string]int{}})
			worker := h.b.getEngineMgr()
			h.b.workersMu.Lock()
			h.b.engineMgrRecovering = true
			h.b.workersMu.Unlock()
			var fail atomic.Bool
			fail.Store(true)
			base := h.engine.answer
			h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == engineSleepMethod && fail.Load() {
					return nil, "fixture engine could not be stopped"
				}
				return base(method, raw)
			})
			if h.b.recoverEngineSwitchOn(worker) {
				t.Fatal("failed startup shutdown was reported as recovered")
			}
			runtime.mu.Lock()
			live, blocked := runtime.live, runtime.managementBlockedLocked()
			runtime.mu.Unlock()
			if h.b.getEngineMgr() != nil || live != nodepolicy.Draining || !blocked {
				t.Fatalf("failed startup lost its fence: live=%s blocked=%v", live, blocked)
			}
			beforeSleep := len(h.engine.methods(engineSleepMethod))
			fail.Store(false)
			if got, err := h.b.setAvailability(context.Background(), state, nodepolicy.PausedLocally); err != nil || got != state {
				t.Fatalf("local recovery = %s, %v; want %s", got, err, state)
			}
			if h.b.getEngineMgr() != worker {
				t.Fatal("local recovery left the acknowledged current manager hidden")
			}
			afterSleep := len(h.engine.methods(engineSleepMethod))
			if state == nodepolicy.Paused {
				if afterSleep <= beforeSleep {
					t.Fatal("local Retry Pause did not retry physical shutdown")
				}
				if len(h.engine.methods(restoreEnabledEnginesMethod)) != 0 {
					t.Fatal("Retry Pause restored engines before Resume")
				}
				if got, err := h.b.setAvailability(context.Background(), nodepolicy.Available, nodepolicy.PausedLocally); err != nil || got != nodepolicy.Available {
					t.Fatalf("resume after retry = %s, %v", got, err)
				}
			} else if afterSleep != beforeSleep {
				t.Fatal("local Resume executed the superseded physical pause")
			}
			h.engine.waitFor("deferred saved-On restoration after local Resume", func(calls []policyCall) bool {
				for _, call := range calls {
					if call.method == restoreEnabledEnginesMethod {
						return true
					}
				}
				return false
			})
		})
	}
	test("local Retry Pause", nodepolicy.Paused)
	test("local Resume", nodepolicy.Available)
}

func TestSwitchLocalPauseVetoDoesNotWaitForTransitionLocks(t *testing.T) {
	h := newSwitchHarness(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c := h.b.switchCoordinator()
	c.waitMu.Lock()
	c.remoteCancel = cancel
	c.waitMu.Unlock()
	r := h.b.nodePolicyRuntime()
	r.transitionMu.Lock()
	waitCtx, stopWaiting := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { _, _ = h.b.setAvailability(waitCtx, nodepolicy.Paused, nodepolicy.PausedLocally); close(done) }()
	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("local veto waited behind transition lock")
	}
	if _, local := h.b.settingsPause(); !local {
		t.Error("pending local pause was not authoritative")
	}
	var projections []policyCall
	projectionDeadline := time.NewTimer(time.Second)
	defer projectionDeadline.Stop()
projectionWait:
	for {
		h.engine.mu.Lock()
		projections = append([]policyCall(nil), h.engine.projections...)
		h.engine.mu.Unlock()
		if len(projections) > 0 {
			break
		}
		select {
		case <-h.engine.signal:
		case <-projectionDeadline.C:
			break projectionWait
		}
	}
	if len(projections) == 0 {
		t.Error("pending local pause did not reach the manager before transition-lock waiting")
	} else {
		var projection nodeProjection
		if err := json.Unmarshal(projections[len(projections)-1].params, &projection); err != nil {
			t.Fatal(err)
		}
		if !projection.PausedLocally {
			t.Error("manager projection omitted the pending local pause veto")
		}
	}
	stopWaiting()
	<-done
	r.mu.Lock()
	if r.transitionCancel != nil {
		r.transitionCancel()
	}
	r.mu.Unlock()
	r.transitionMu.Unlock()
}

func TestSwitchManagerProgressNotificationKeepsRemoteIdentity(t *testing.T) {
	h := newSwitchHarness(t)
	want := settings.SwitchChanged{NodeID: "remote-target", RequestID: "remote-operation", Sequence: 9}
	h.b.forwardEngineNotificationFrom(h.b.activateEngineManagerSource(), settings.SwitchChangedNotification, settingsJSON(want))
	frames := h.output.frames()
	if len(frames) != 1 || frames[0].Method != settings.SwitchChangedNotification {
		t.Fatalf("forwarded frames = %+v", frames)
	}
	var got settings.SwitchChanged
	if err := json.Unmarshal(frames[0].Params, &got); err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("forwarded notification = %+v, want %+v", got, want)
	}
}

func TestSwitchValidationCleanupFailureRetainsRecoveryFence(t *testing.T) {
	for _, scenario := range []string{"RPC failure", "storage envelope"} {
		t.Run(scenario, func(t *testing.T) {
			h := newSwitchHarness(t)
			p := h.request(t, "uncertain-prepare")
			base := h.engine.answer
			h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == settings.SwitchPrepareMethod {
					var prepare settings.SwitchPrepareRequest
					if err := json.Unmarshal(raw, &prepare); err != nil {
						return nil, err.Error()
					}
					if prepare.OperationID != "" {
						// A prepare can reserve durably and lose its reply.
						return nil, "prepare response lost"
					}
				}
				if method == settings.SwitchReconcileMethod {
					if scenario == "RPC failure" {
						return nil, "manager unavailable"
					}
					return settings.SwitchReconcileResult{Error: &settings.SwitchError{Code: -32000, Kind: settings.SwitchErrorStorage, Message: "Cannot read checkpoint."}}, ""
				}
				return base(method, raw)
			})
			if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err != nil {
				t.Fatal(err)
			}
			r := h.receipt(t, p.RequestID)
			if r.Phase != settings.SwitchFailedValidation || !r.RecoveryRequired {
				t.Fatalf("uncertain cleanup = %+v", r)
			}
			h.b.engineConfigMu.Lock()
			err := h.b.pendingSwitchLocked(p.SourceEngine)
			h.b.engineConfigMu.Unlock()
			if err == nil || len(switchSteps(t, h.engine)) != 0 {
				t.Fatal("uncertain cleanup released mutation reservation")
			}
		})
	}
}

func TestSwitchUncertainAcceptanceStillAllowsIdenticalRetryAndStatus(t *testing.T) {
	h := newSwitchHarness(t)
	p := h.request(t, "landed-acceptance")
	h.b.engineSwitch.store.write = func(path string, data []byte, mode os.FileMode) error {
		if err := durablefile.WriteFile(path, data, mode); err != nil {
			return err
		}
		return durablefile.ErrReplaced
	}
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); !errors.Is(err, durablefile.ErrReplaced) {
		t.Fatalf("acceptance error = %v", err)
	}
	retry, err := h.b.acceptEngineSwitch(context.Background(), p, "")
	if err != nil || retry.Receipt.Phase != settings.SwitchAccepted {
		t.Fatalf("identical retry = %+v, %v", retry, err)
	}
	status, err := h.b.engineSwitchStatus(context.Background(), settings.SwitchStatusRequest{RequestID: p.RequestID}, "")
	if err != nil || status.Receipt == nil || status.Receipt.Sequence != retry.Receipt.Sequence {
		t.Fatalf("status = %+v, %v", status, err)
	}
	p.RequestID = "fresh-after-fault"
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err == nil {
		t.Fatal("fresh mutation ignored storage fence")
	}
	if len(switchSteps(t, h.engine)) != 0 || len(h.proxy.snapshot()) != 0 {
		t.Fatal("uncertain acceptance caused effects")
	}
}

func TestSwitchAcceptedRemoteOperationIsCancelledByLocalPause(t *testing.T) {
	h := newSwitchHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", true)
	h.active.Store(1)
	p := h.request(t, "remote-pause")
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
		t.Fatal(err)
	}
	h.proxy.waitFor("remote drain", func(calls []policyCall) bool {
		for _, call := range calls {
			if call.method == nodepolicy.MethodAdmissionSnapshot {
				return true
			}
		}
		return false
	})
	runtime := h.b.nodePolicyRuntime()
	runtime.transitionMu.Lock()
	waitCtx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		_, _ = h.b.setAvailability(waitCtx, nodepolicy.Paused, nodepolicy.PausedLocally)
		close(done)
	}()
	r := h.receipt(t, p.RequestID)
	cancel()
	<-done
	runtime.mu.Lock()
	if runtime.transitionCancel != nil {
		runtime.transitionCancel()
	}
	runtime.mu.Unlock()
	runtime.transitionMu.Unlock()
	if r.Phase != settings.SwitchFailedDrain {
		t.Fatalf("pause outcome = %+v", r)
	}
	for _, step := range switchSteps(t, h.engine) {
		if step != settings.SwitchStepRelease {
			t.Fatalf("local pause allowed lifecycle step %s", step)
		}
	}
}

func TestSwitchLocalOffVetoOnlyMatchesReservedEngines(t *testing.T) {
	h := newSwitchHarness(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c := h.b.switchCoordinator()
	c.waitMu.Lock()
	c.remoteCancel, c.remoteSource, c.remoteTarget = cancel, "llamacpp", "lmstudio"
	c.waitMu.Unlock()
	h.b.cancelRemoteEngineSwitch("ollama")
	if ctx.Err() != nil {
		t.Fatal("unrelated local Off cancelled remote switch")
	}
	h.b.cancelRemoteEngineSwitch("lmstudio")
	if ctx.Err() == nil {
		t.Fatal("target local Off did not cancel remote switch")
	}
}

func TestSwitchLocalOffReachesManagerWhileTargetReadinessIsHeld(t *testing.T) {
	h := newSwitchHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", true)
	p := h.request(t, "held-readiness-off")
	base := h.engine.answer
	started := make(chan struct{})
	vetoPath := filepath.Join(t.TempDir(), "local-off.json")
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == settings.SwitchStepMethod {
			var step settings.SwitchStepRequest
			if err := json.Unmarshal(raw, &step); err != nil {
				return nil, err.Error()
			}
			if step.Step == settings.SwitchStepTargetStart {
				// No clock or readiness completion can release this request. It
				// outlives any cleanup budget unless the local veto reaches us.
				close(started)
				return nil, holdReply
			}
		}
		if method == "engine:stop" {
			var p struct {
				VerifyStop bool `json:"verifyStop"`
			}
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if p.VerifyStop {
				return base(method, raw)
			}
			if err := os.WriteFile(vetoPath, settingsJSON(map[string]string{"lmstudio": "off"}), 0600); err != nil {
				return nil, err.Error()
			}
			h.mu.Lock()
			h.cp.Target = switchTestState(false, false, settings.SwitchIntentOff)
			h.cp.Error = &settings.SwitchError{Code: settings.SwitchRemoteVetoCode, Kind: settings.SwitchErrorRemoteVeto, Message: "Local Off veto."}
			h.mu.Unlock()
			h.engine.releaseHeld()
			return map[string]bool{"ok": true}, ""
		}
		return base(method, raw)
	})
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
		t.Fatal(err)
	}
	requireGateOpens(t, started, "target startup")
	done := make(chan struct{})
	go func() {
		id := json.RawMessage(`91`)
		h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:stop", Params: settingsJSON(map[string]string{"engine": "lmstudio"})})
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		h.engine.releaseHeld()
		t.Fatal("local Off waited behind the switch operation lock")
	}
	data, err := os.ReadFile(vetoPath)
	if err != nil {
		t.Fatal(err)
	}
	var intents map[string]string
	if err := json.Unmarshal(data, &intents); err != nil {
		t.Fatal(err)
	}
	if intents["lmstudio"] != settings.SwitchIntentOff {
		t.Fatalf("persisted intent = %v", intents)
	}
	r := h.receipt(t, p.RequestID)
	if !r.Terminal || r.Phase != settings.SwitchFailedTargetStart || r.Target.Intent != settings.SwitchIntentOff || r.Source.Intent != settings.SwitchIntentOff {
		t.Fatalf("veto receipt = %+v", r)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.cp.TargetCommitted {
		t.Fatal("target On committed after local Off")
	}
}

func TestSwitchLocalOffDuringLocalSwitchIsBusy(t *testing.T) {
	h := newSwitchHarness(t)
	h.active.Store(1)
	p := h.request(t, "local-switch-off")
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() {
		id := json.RawMessage(`92`)
		h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:stop", Params: settingsJSON(map[string]string{"engine": "lmstudio"})})
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		h.active.Store(0)
		t.Fatal("reserved local Off waited instead of refusing busy")
	}
	var response *Message
	for _, frame := range h.output.frames() {
		if frame.ID != nil && string(*frame.ID) == "92" {
			copy := frame
			response = &copy
		}
	}
	if response == nil || response.Error == nil || response.Error.Code != settings.SwitchOperationErrorCode {
		t.Fatalf("local Off response = %+v", response)
	}
	for _, call := range h.engine.snapshot() {
		if call.method == "engine:stop" {
			t.Fatal("local Off reached manager during local switch")
		}
	}
	h.active.Store(0)
	if r := h.receipt(t, p.RequestID); r.Phase != settings.SwitchCompleted {
		t.Fatalf("local Off cancelled local switch: %+v", r)
	}
}

func TestSwitchLocalOffReachesManagerDuringRecoveryReservation(t *testing.T) {
	for _, scenario := range []string{"receipt", "orphan checkpoint"} {
		t.Run(scenario, func(t *testing.T) {
			h := newSwitchHarness(t)
			r := switchStoreReceipt("remote-recovery-off")
			r.Caller = "peer"
			r.PayloadHash = settings.SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
			if scenario == "orphan checkpoint" {
				h.b.engineSwitch.recoveryCheckpoint = &settings.SwitchCheckpoint{SwitchPrepareRequest: settings.SwitchPrepareRequest{Origin: settings.SwitchOriginRemote, SourceEngine: r.Request.SourceEngine, TargetEngine: r.Request.TargetEngine}, Phase: settings.SwitchCheckpointStartPending}
			} else {
				switchStoreAccept(t, h.b.engineSwitch.store, r)
				if err := h.b.engineSwitch.store.InterruptNonterminal(time.Now().UTC()); err != nil {
					t.Fatal(err)
				}
			}
			h.b.settingsApplyMu.Lock()
			h.b.workersMu.Lock()
			h.b.engineMgrRecovering = true
			h.b.workersMu.Unlock()
			done := make(chan struct{})
			go func() {
				id := json.RawMessage(`93`)
				h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:stop", Params: settingsJSON(map[string]string{"engine": r.Request.TargetEngine})})
				close(done)
			}()
			select {
			case <-done:
			case <-time.After(2 * time.Second):
				h.b.settingsApplyMu.Unlock()
				t.Fatal("recovery reservation blocked local Off")
			}
			h.b.settingsApplyMu.Unlock()
			var found bool
			for _, call := range h.engine.snapshot() {
				found = found || call.method == "engine:stop"
			}
			if scenario == "orphan checkpoint" {
				var refused bool
				for _, frame := range h.output.frames() {
					if frame.ID != nil && string(*frame.ID) == "93" && frame.Error != nil {
						refused = true
					}
				}
				if found || !refused {
					t.Fatal("orphan without a journal identity must visibly refuse durable Off admission")
				}
			} else if !found {
				t.Fatal("local Off did not reach recovery manager")
			}
		})
	}
}

func TestSwitchManagerReplacementAfterSourceOffNeverStartsTarget(t *testing.T) {
	h := newSwitchHarness(t)
	p := h.request(t, "manager-replacement")
	base := h.engine.answer
	peer, replacement := newFakeWorkerPipe(t)
	w := &rpcWorker{peer: peer}
	replacement.setAnswer(base)
	// The durable source-off write is an exact barrier after the stop reply
	// and before target-start dispatch, without a scheduler-dependent sleep.
	h.b.engineSwitch.store.write = func(path string, data []byte, mode os.FileMode) error {
		if err := durablefile.WriteFile(path, data, mode); err != nil {
			return err
		}
		journal, err := settings.DecodeSwitchJournal(data)
		if err != nil {
			return err
		}
		if journal.Receipts[p.RequestID].Phase == settings.SwitchSourceOff {
			h.b.setEngineMgr(w)
		}
		return nil
	}
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	r := h.receipt(t, p.RequestID)
	if r.Phase != settings.SwitchFailedInterrupted || !r.RecoveryRequired || r.Error == nil || r.Error.Kind != settings.SwitchErrorInterrupted {
		t.Fatalf("restart receipt = %+v", r)
	}
	if h.b.getEngineMgr() != nil {
		t.Fatal("replacement exposed before recovery")
	}
	if len(replacement.snapshot()) != 0 {
		t.Fatal("replacement received work before recovery")
	}
	if !h.b.recoverReplacementEngineMgr(w) {
		t.Fatal("replacement recovery failed")
	}
	if h.b.getEngineMgr() != w {
		t.Fatal("replacement remained fenced after successful recovery")
	}
	if got := switchSteps(t, replacement); !reflect.DeepEqual(got, []string{settings.SwitchStepRelease}) {
		t.Fatalf("replacement steps = %v", got)
	}
	var reconciled bool
	for _, call := range replacement.snapshot() {
		reconciled = reconciled || call.method == settings.SwitchReconcileMethod
	}
	if !reconciled {
		t.Fatal("replacement did not reconcile inherited reservation")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.cp.TargetCommitted {
		t.Fatal("replacement committed target On")
	}
}

func TestSwitchPublicationRechecksRoutability(t *testing.T) {
	h := newSwitchHarness(t)
	p := h.request(t, "facade-disappeared")
	base := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, errText := base(method, raw)
		if method == settings.SwitchStepMethod {
			var step settings.SwitchStepRequest
			if err := json.Unmarshal(raw, &step); err != nil {
				return nil, err.Error()
			}
			if step.Step == settings.SwitchStepTargetStart {
				h.pp.readyMu.Lock()
				h.pp.facadeState["lmstudio"] = proxyFacadeState{}
				h.pp.readyMu.Unlock()
			}
		}
		return result, errText
	})
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	r := h.receipt(t, p.RequestID)
	if r.Phase != settings.SwitchFailedPublication || r.Error.Kind != settings.SwitchErrorUnroutable {
		t.Fatalf("publication outcome = %+v", r)
	}
	for _, call := range h.proxy.snapshot() {
		if call.method == "lmstudio:node/set-local-backend" {
			t.Fatal("published a target whose facade disappeared")
		}
	}
}

func TestSwitchRelayUsesAuthenticatedCallerAndCodedReply(t *testing.T) {
	h := newSwitchHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", true)
	p := h.request(t, "relay-revision")
	// A valid but stale token exercises the coded refusal path.
	p.ExpectedRevision = settings.SwitchConfigurationToken(settings.SwitchConfigurationSnapshot{NodeID: "stale"})
	h.b.forwardEngineNotificationFrom(bindRemoteManagementTestSource(h.b), "engine:switch-request", settingsJSON(struct {
		ID     string                 `json:"id"`
		Method string                 `json:"method"`
		Caller string                 `json:"caller"`
		Params settings.SwitchRequest `json:"params"`
	}{"relay-1", settings.SwitchMethod, "peer", p}))
	calls := h.engine.waitFor("switch relay reply", func(calls []policyCall) bool {
		for _, call := range calls {
			if call.method == "engine:switch-reply" {
				return true
			}
		}
		return false
	})
	for _, call := range calls {
		if call.method != "engine:switch-reply" {
			continue
		}
		var reply struct {
			ID        string                `json:"id"`
			Result    settings.SwitchResult `json:"result"`
			ErrorCode int                   `json:"errorCode"`
		}
		if err := json.Unmarshal(call.params, &reply); err != nil {
			t.Fatal(err)
		}
		if reply.ID != "relay-1" || reply.ErrorCode != settings.SwitchConfigurationConflictCode || reply.Result.Receipt.Caller != "peer" || reply.Result.Receipt.Phase != settings.SwitchRejected {
			t.Fatalf("relay reply = %+v", reply)
		}
	}
}

func TestSwitchFailedTerminalWritesNeverReleaseManager(t *testing.T) {
	cases := []struct {
		phase string
		path  []string
	}{
		{settings.SwitchFailedValidation, []string{settings.SwitchValidating}},
		{settings.SwitchFailedDrain, []string{settings.SwitchValidating, settings.SwitchDraining}},
		{settings.SwitchFailedSourceStop, []string{settings.SwitchValidating, settings.SwitchDraining, settings.SwitchSourceStopping}},
		{settings.SwitchFailedTargetStart, []string{settings.SwitchValidating, settings.SwitchDraining, settings.SwitchSourceStopping, settings.SwitchSourceOff, settings.SwitchTargetStarting}},
		{settings.SwitchFailedPublication, []string{settings.SwitchValidating, settings.SwitchDraining, settings.SwitchSourceStopping, settings.SwitchSourceOff, settings.SwitchTargetStarting, settings.SwitchTargetReady}},
		{settings.SwitchFailedInterrupted, nil},
	}
	for _, tc := range cases {
		for _, replaced := range []bool{false, true} {
			name := tc.phase + " before replacement"
			if replaced {
				name = tc.phase + " after replacement"
			}
			t.Run(name, func(t *testing.T) {
				h := newSwitchHarness(t)
				initial := switchStoreReceipt("terminal-fault")
				now := time.Now().UTC()
				initial.AcceptedAt, initial.UpdatedAt, initial.Deadline = now, now, now.Add(settings.SwitchOperationBudget)
				r := switchStoreAccept(t, h.b.engineSwitch.store, initial)
				for _, phase := range tc.path {
					r = switchStoreNext(r, phase)
					if err := h.b.engineSwitch.store.Advance(r.RequestID, r); err != nil {
						t.Fatal(err)
					}
				}
				p := settings.SwitchPrepareRequest{OperationID: r.RequestID, Caller: r.Caller, Origin: settings.SwitchOriginLocal, SourceEngine: r.Request.SourceEngine, TargetEngine: r.Request.TargetEngine, PayloadHash: r.PayloadHash, ConfigurationFence: r.ExpectedRevision, Deadline: r.Deadline}
				h.mu.Lock()
				h.cp = &settings.SwitchCheckpoint{SwitchPrepareRequest: p, Phase: settings.SwitchCheckpointReserved, Source: r.Source, Target: r.Target}
				h.mu.Unlock()
				h.b.engineSwitch.store.write = func(path string, data []byte, mode os.FileMode) error {
					if replaced {
						if err := durablefile.WriteFile(path, data, mode); err != nil {
							return err
						}
						return durablefile.ErrReplaced
					}
					return errors.New("terminal write failed")
				}
				h.b.finishEngineSwitch(p, tc.phase, nil, errors.New("operation failed"))
				if h.b.engineSwitch.err == nil || len(switchSteps(t, h.engine)) != 0 {
					t.Fatal("uncertain terminal receipt released manager")
				}
				stored, ok := h.b.engineSwitch.store.Lookup(r.RequestID)
				if !ok || stored.Terminal != replaced {
					t.Fatalf("landed terminal state = %+v", stored)
				}
			})
		}
	}
}

func TestSwitchRejectedReceiptWriteFailureHasNoEffects(t *testing.T) {
	for _, replaced := range []bool{false, true} {
		name := "before replacement"
		if replaced {
			name = "after replacement"
		}
		t.Run(name, func(t *testing.T) {
			h := newSwitchHarness(t)
			p := h.request(t, "reject-fault")
			p.TargetEngine = "unroutable-variant"
			h.b.engineSwitch.store.write = func(path string, data []byte, mode os.FileMode) error {
				if replaced {
					if err := durablefile.WriteFile(path, data, mode); err != nil {
						return err
					}
					return durablefile.ErrReplaced
				}
				return errors.New("rejection write failed")
			}
			if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err == nil {
				t.Fatal("ignored rejection persistence failure")
			}
			stored, exists := h.b.engineSwitch.store.Lookup(p.RequestID)
			if exists != replaced || (exists && stored.Phase != settings.SwitchRejected) {
				t.Fatalf("rejected receipt = %+v, exists %v", stored, exists)
			}
			if h.b.engineSwitch.err == nil || len(switchSteps(t, h.engine)) != 0 || len(h.proxy.snapshot()) != 0 {
				t.Fatal("rejection write failure did not halt effects")
			}
		})
	}
}

func TestSwitchStatusWaitTimeoutReturnsUnchangedReceipt(t *testing.T) {
	h := newSwitchHarness(t)
	h.active.Store(1)
	p := h.request(t, "unchanged-status")
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	h.proxy.waitFor("draining operation", func(calls []policyCall) bool {
		for _, call := range calls {
			if call.method == nodepolicy.MethodAdmissionSnapshot {
				return true
			}
		}
		return false
	})
	status, err := h.b.engineSwitchStatus(context.Background(), settings.SwitchStatusRequest{RequestID: p.RequestID}, "")
	if err != nil || status.Receipt == nil {
		t.Fatalf("initial status = %+v, %v", status, err)
	}
	sequence := status.Receipt.Sequence
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	next, err := h.b.engineSwitchStatus(ctx, settings.SwitchStatusRequest{RequestID: p.RequestID, AfterSequence: &sequence}, "")
	h.active.Store(0)
	if err != nil || next.Receipt == nil || next.Receipt.Sequence != sequence {
		t.Fatalf("unchanged status = %+v, %v", next, err)
	}
}

func TestSwitchManagerRPCErrorsKeepCodeAndKind(t *testing.T) {
	for _, tc := range []struct {
		kind string
		code int
	}{
		{settings.SwitchErrorInvalidRequest, settings.SwitchInvalidParamsCode},
		{settings.SwitchErrorConfigurationConflict, settings.SwitchConfigurationConflictCode},
		{settings.SwitchErrorRemoteVeto, settings.SwitchRemoteVetoCode},
		{settings.SwitchErrorEngineLocked, settings.SwitchEngineLockedCode},
	} {
		t.Run(tc.kind, func(t *testing.T) {
			if built := switchErrorDetail(switchError(tc.kind, "Built refusal."), settings.SwitchErrorStorage); built.Code != tc.code || built.Validate() != nil {
				t.Fatalf("switchError(%q) = %+v, want code %d", tc.kind, built, tc.code)
			}
			worker, server := newTestRPCWorkerPipe(t)
			b := &Broker{}
			b.setEngineMgr(worker)
			go func() {
				request, err := server.Read()
				if err != nil {
					t.Error(err)
					return
				}
				if err := server.RespondError(request.ID, tc.code, "Coded manager refusal."); err != nil {
					t.Error(err)
				}
			}()
			var result settings.SwitchPrepareResult
			err := b.switchManagerCall(context.Background(), settings.SwitchPrepareMethod, settings.SwitchPrepareRequest{}, &result)
			if err == nil {
				t.Fatal("manager refusal was ignored")
			}
			detail := switchErrorDetail(err, settings.SwitchErrorValidation)
			if detail.Code != tc.code || detail.Kind != tc.kind {
				t.Fatalf("manager error = %+v", detail)
			}
			if err := detail.Validate(); err != nil {
				t.Fatalf("manager error cannot be persisted: %v", err)
			}
		})
	}
}

func TestSwitchBrokerPauseErrorKeepsRemoteVetoCodeAndKind(t *testing.T) {
	detail := switchErrorDetail(&localVetoError{}, settings.SwitchErrorValidation)
	if detail.Code != settings.SwitchRemoteVetoCode || detail.Kind != settings.SwitchErrorRemoteVeto ||
		detail.Message != "This node is paused. Resume it on this node before managing it remotely." {
		t.Fatalf("broker pause error = %+v", detail)
	}
	if err := detail.Validate(); err != nil {
		t.Fatalf("broker pause error cannot cross the switch relay: %v", err)
	}
}

func TestSwitchDelayedSnapshotCannotWriteReceiptAfterPauseAcceptance(t *testing.T) {
	h := newSwitchHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", true)
	p := h.request(t, "snapshot-before-pause")
	base := h.engine.answer
	reached := make(chan struct{})
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, errText := base(method, raw)
		if method == settings.SwitchPrepareMethod && errText == "" {
			close(reached)
			return result, holdSuccessReply
		}
		return result, errText
	})
	var writes atomic.Int32
	write := h.b.engineSwitch.store.write
	h.b.engineSwitch.store.write = func(path string, data []byte, mode os.FileMode) error {
		writes.Add(1)
		return write(path, data, mode)
	}
	generation := h.b.engineSwitch.store.Generation()
	done := make(chan error, 1)
	go func() { _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); done <- err }()
	receive(t, "switch snapshot readback to wait", reached)
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.pendingPauseRequested = true
	r.mu.Unlock()
	h.b.publishNodeProjection()
	h.engine.releaseHeld()
	err := receive(t, "delayed switch refusal", done)
	var paused *localVetoError
	if !errors.As(err, &paused) || policyErrorCode(err) != codeLocalVeto {
		t.Fatalf("delayed switch = %v, want broker pause refusal", err)
	}
	if _, exists := h.b.engineSwitch.store.Lookup(p.RequestID); exists || writes.Load() != 0 || h.b.engineSwitch.store.Generation() != generation {
		t.Fatalf("refused switch wrote a receipt: exists=%t writes=%d generation=%d", exists, writes.Load(), h.b.engineSwitch.store.Generation())
	}
	if steps := switchSteps(t, h.engine); len(steps) != 0 {
		t.Fatalf("refused switch dispatched lifecycle steps: %v", steps)
	}
}

func TestPausedSwitchDuplicateReceiptIsRefusedWithoutJournalChange(t *testing.T) {
	h := newSwitchHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", true)
	p := h.request(t, "duplicate-while-paused")
	p.NodeID = h.b.nodeID
	r := switchStoreReceipt(p.RequestID)
	r.NodeID, r.Caller, r.Request = h.b.nodeID, "peer", p
	r.ExpectedRevision, r.AcceptedRevision = p.ExpectedRevision, p.ExpectedRevision
	r.PayloadHash = settings.SwitchPayloadHash("peer", h.b.nodeID, p)
	before := switchStoreAccept(t, h.b.engineSwitch.store, r)
	data, err := os.ReadFile(h.b.engineSwitch.store.path)
	if err != nil {
		t.Fatalf("read accepted switch receipt: %v", err)
	}
	runtime := h.b.nodePolicyRuntime()
	runtime.mu.Lock()
	runtime.current.Availability, runtime.current.PausedBy = nodepolicy.Paused, nodepolicy.PausedByPeer
	runtime.mu.Unlock()
	_, err = h.b.acceptEngineSwitch(context.Background(), p, "peer")
	var paused *localVetoError
	if !errors.As(err, &paused) {
		t.Fatalf("paused duplicate = %v, want broker pause refusal", err)
	}
	after, exists := h.b.engineSwitch.store.Lookup(p.RequestID)
	if !exists || !reflect.DeepEqual(after, before) {
		t.Fatalf("paused duplicate changed receipt: %+v -> %+v", before, after)
	}
	afterData, err := os.ReadFile(h.b.engineSwitch.store.path)
	if err != nil {
		t.Fatalf("read refused switch receipt: %v", err)
	}
	if string(afterData) != string(data) {
		t.Fatal("paused duplicate rewrote the switch journal")
	}
}
