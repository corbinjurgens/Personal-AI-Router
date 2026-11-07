// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

type diagnosticRecordParams struct {
	RunID   string                      `json:"runId"`
	Token   string                      `json:"token"`
	Profile modelconfig.MeasuredProfile `json:"profile"`
}

// Lifecycle cases still exercise inspection, authenticated persistence and
// explicit release on the wire. No test marks the broker ledger as recorded.
func setDiagnosticContractAnswer(t *testing.T, manager *fakeWorker, op *brokerDiagnostic, answer func(string, json.RawMessage) (any, string)) {
	t.Helper()
	run := op.run.Clone()
	identity := op.review.preview.DiagnosticIdentity.Clone()
	expectedToken := op.token
	if expectedToken == "" {
		expectedToken = "token"
	}
	stored := false
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:inspect-model":
			var request diagnosticInspectEnvelope
			if err := modelconfig.DecodeStrict(raw, &request); err != nil {
				return nil, err.Error()
			}
			if request.NativeInitAllowed || request.Request.Engine != run.Engine || request.Request.ModelID != run.ModelID || request.Request.ProfileKey != run.ProfileKey {
				return nil, "inspection selection changed"
			}
			if !reflect.DeepEqual(request.Hardware, diagnosticIdentityHardware(identity)) {
				return nil, "inspection omitted producing hardware"
			}
			inspection := diagnosticInspectionFixture(run, identity)
			if err := inspection.Validate(); err != nil {
				return nil, err.Error()
			}
			return inspection, ""
		case "engine:record-measured-memory":
			var p diagnosticRecordParams
			if err := modelconfig.DecodeStrict(raw, &p); err != nil {
				return nil, err.Error()
			}
			if err := p.Profile.Validate(); err != nil {
				return nil, err.Error()
			}
			if p.RunID != run.RunID || p.Token != expectedToken || p.Profile.ConfigurationKey != run.ConfigurationKey || p.Profile.ExecutionKey != run.ExecutionKey || p.Profile.RunID != run.RunID || !diagnosticRequestedIdentityMatches(p.Profile.DiagnosticIdentity, identity) {
				return nil, "record authority or identity changed"
			}
			stored = true
			return map[string]bool{"ok": true}, ""
		case "engine:diagnostic-release":
			var p diagnosticAuthority
			if err := modelconfig.DecodeStrict(raw, &p); err != nil {
				return nil, err.Error()
			}
			if p.RunID != run.RunID || p.Token != expectedToken || !stored {
				return nil, "release preceded authenticated record"
			}
			return map[string]bool{"ok": true}, ""
		}
		return answer(method, raw)
	})
}

func diagnosticInspectionFixture(run modelconfig.DiagnosticRun, identity modelconfig.DiagnosticIdentity) modelconfig.InspectResult {
	result := modelconfig.InspectResult{Engine: run.Engine, ModelID: run.ModelID, Status: "resolved", Estimate: modelconfig.Estimate{Status: "unknown"}, ConfigurationKey: identity.ConfigurationKey, ExecutionKey: identity.ExecutionKey, ConfigurationKeyReason: identity.ConfigurationKeyReason, NativeBuildID: identity.NativeBuildID, Artifacts: slices.Clone(identity.Artifacts)}
	if identity.ConfigurationKey != "" {
		result.ConfigurationPools = []string{"host", "device"}
	}
	return result
}

func TestModelDiagnosticFailedIdentityDriftReleasesAfterRecording(t *testing.T) {
	for _, scenario := range []struct {
		name        string
		unavailable bool
		outcome     string
	}{
		{"failed identity changed", false, "failed"},
		{"failed inspection unavailable", true, "failed"},
		{"completed identity changed after stop", false, "completed"},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			b, manager, op := diagnosticBrokerFixture(t)
			run := op.run.Clone()
			run.Phase, run.Outcome = "done", scenario.outcome
			if run.Outcome == "failed" {
				run.FailedPhase, run.Reason = "high", "identity-changed"
			}
			identity := op.review.preview.DiagnosticIdentity.Clone()
			stored, released := false, false
			manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				switch method {
				case "engine:diagnostic-run-status":
					return diagnosticStatusReply{Run: run, State: "cleanup-complete", Token: op.token, WorkStopped: true, Identity: &identity}, ""
				case "engine:inspect-model":
					if scenario.unavailable {
						return nil, "artifact unavailable"
					}
					changed := diagnosticInspectionFixture(run, identity)
					changed.ConfigurationKey = strings.Repeat("c", 64)
					return changed, ""
				case "engine:record-measured-memory":
					var p diagnosticRecordParams
					if err := modelconfig.DecodeStrict(raw, &p); err != nil {
						return nil, err.Error()
					}
					if p.Profile.Outcome != run.Outcome || !reflect.DeepEqual(p.Profile.DiagnosticIdentity, identity) || p.Profile.BufferSampleID != "" {
						return nil, "failure did not retain its producing identity"
					}
					stored = true
					return nil, ""
				case "engine:diagnostic-release":
					if !stored {
						return nil, "release preceded failure evidence"
					}
					released = true
					return nil, ""
				}
				return nil, "unexpected method"
			})
			if err := b.cleanupDiagnostic(context.Background(), op); err != nil {
				t.Fatalf("physically stopped failure retained resources: %v", err)
			}
			if !stored || !released {
				t.Fatal("failure evidence and explicit release did not complete")
			}
			requireDiagnosticReleased(t, b)
		})
	}
}

func TestModelDiagnosticPreviewFiniteTargets(t *testing.T) {
	b, manager, request, observation := diagnosticPreviewFixture(t)
	observation.Identity.PerSlotContext, observation.Identity.Slots = 0, 0
	observation.Identity.BuildInfo, observation.Identity.Placement = "", nil
	observation.PlannedPerSlotContext, observation.PlannedSlots = 4096, 2
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method != diagnosticObserveMethod {
			return nil, "preview attempted an effect"
		}
		return observation, ""
	})
	preview, err := b.previewDiagnostic(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if preview.LowPromptTokens != 512 || preview.HighPromptTokens != 3952 || preview.PlannedSlots != 2 || preview.PlannedPerSlotContext != 4096 || preview.Slots != 0 || preview.PerSlotContext != 0 || len(preview.Placement) != 0 {
		t.Fatalf("finite consent did not distinguish planned targets: %+v", preview)
	}
}

func seedDiagnosticTelemetry(t *testing.T, b *Broker, at time.Time, memory *noderec.MemoryObservation) {
	t.Helper()
	if err := ingestDiagnosticTelemetry(b, at, memory); err != nil {
		t.Fatal(err)
	}
}

func ingestDiagnosticTelemetry(b *Broker, at time.Time, memory *noderec.MemoryObservation) error {
	if b.telemetry == nil {
		b.telemetry = newTelemetryCache()
		b.scannerSource = &scannerProcess{}
	}
	boot, err := noderec.MemoryBootClock()
	if err != nil {
		return err
	}
	input := noderec.NodeTelemetry{HostUUID: b.nodeID, Memory: memory, MemoryHop: noderec.NewMemoryHop("scanner", noderec.MemoryBootClock), MemoryRequest: &noderec.MemoryRequestContext{Generation: "1", RequestBootNS: strconv.FormatUint(boot, 10), VerifiedSelfLoopback: true}}
	if _, ok := b.telemetry.Upsert(sourceScanner, input, at); !ok {
		return fmt.Errorf("fresh local telemetry refused")
	}
	return nil
}

func seedDiagnosticHardware(t *testing.T, b *Broker, identity modelconfig.DiagnosticIdentity, at time.Time) *fakeWorker {
	t.Helper()
	memory := diagnosticMemoryFixture(at, 9000)
	memory.Domains[0].Total.Bytes = identity.TotalRAMBytes
	gpus := []noderec.GPUInfo{}
	for _, gpu := range identity.GPUs {
		total, err := strconv.ParseUint(gpu.TotalMemoryBytes, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		domain := memory.Domains[0]
		domain.ID, domain.Kind, domain.DeviceIDs = gpu.UUID, "dedicated", []string{gpu.UUID}
		domain.Total.Bytes = gpu.TotalMemoryBytes
		memory.Domains = append(memory.Domains, domain)
		gpus = append(gpus, noderec.GPUInfo{PhysicalID: gpu.UUID, Name: gpu.Name, VramBytes: total})
	}
	seedDiagnosticTelemetry(t, b, at, memory)
	peer, scanner := newFakeWorkerPipe(t)
	owner := &scannerProcess{peer: peer}
	b.workersMu.Lock()
	b.scanner, b.scannerSource = owner, owner
	b.workersMu.Unlock()
	ram, err := strconv.ParseUint(identity.TotalRAMBytes, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	scanner.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		var p nodeInfoRPCParams
		if method != "discovery:get-node-info" || json.Unmarshal(raw, &p) != nil || p.HostUUID != b.nodeID {
			return nil, "unexpected hardware read"
		}
		return map[string]any{"hostUuid": b.nodeID, "GPUs": gpus, "cpu": noderec.CPUInfo{Name: identity.CPUModel}, "memory": noderec.MemoryInfo{TotalBytes: ram}, "memoryObservation": memory, "telemetryValid": true, "msSince": 0}, ""
	})
	return scanner
}

func diagnosticPreviewFixture(t *testing.T) (*Broker, *fakeWorker, modelconfig.DiagnosticPreviewRequest, diagnosticObservation) {
	t.Helper()
	b, _ := newPolicyBroker(t)
	seedDiagnosticHardware(t, b, diagnosticIdentityFixture(), time.Now())
	peer, manager := newFakeWorkerPipe(t)
	b.setEngineMgr(&rpcWorker{peer: peer, done: make(chan struct{})})
	p := modelconfig.DiagnosticPreviewRequest{Engine: "llamacpp", ModelID: "fixture.gguf", ProfileKey: "fixture"}
	observation := diagnosticObservation{Identity: diagnosticIdentityFixture(), EffectsFence: "effects", DrainEngines: []string{"llamacpp"}, Supported: true}
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method != diagnosticObserveMethod {
			return nil, "unexpected effect"
		}
		return observation, ""
	})
	return b, manager, p, observation
}

func requireNoDiagnosticEffects(t *testing.T, b *Broker, manager *fakeWorker) {
	t.Helper()
	for _, call := range manager.snapshot() {
		if call.method != diagnosticObserveMethod {
			t.Fatalf("unexpected effect: %s", call.method)
		}
	}
	b.engineConfigMu.Lock()
	leases := len(b.buildApplyMaintenance.settings)
	b.engineConfigMu.Unlock()
	if leases != 0 || b.diagnosticHoldActive() {
		t.Fatal("refused start acquired a ledger resource")
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	drains := len(r.drains)
	r.mu.Unlock()
	if drains != 0 {
		t.Fatal("refused start changed drains")
	}
}

func TestModelDiagnosticPreviewAndStartPreconditions(t *testing.T) {
	for _, scenario := range []string{"Halt", "build apply", "settings apply", "missing telemetry", "remote target", "unsupported engine"} {
		t.Run(scenario, func(t *testing.T) {
			b, manager, p, _ := diagnosticPreviewFixture(t)
			preview, err := b.previewDiagnostic(context.Background(), p)
			if err != nil {
				t.Fatal(err)
			}
			switch scenario {
			case "Halt":
				b.nodePolicy.current.PauseMode = nodepolicy.PauseImmediate
			case "build apply":
				b.buildApplyMaintenance.active = &buildApplyReservation{}
			case "settings apply":
				b.activeSettings = map[string]activeSettingsOperation{"llamacpp": {}}
			case "missing telemetry":
				b.telemetry = nil
			case "remote target":
				p.NodeID = "peer"
			case "unsupported engine":
				p.Engine = "other"
			}
			if _, err := b.previewDiagnostic(context.Background(), p); err == nil {
				t.Fatal("preview accepted failed precondition")
			}
			if _, err := b.startDiagnostic(context.Background(), modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: p, PreviewID: preview.PreviewID, Consent: true}); err == nil {
				t.Fatal("start accepted failed precondition")
			}
			requireNoDiagnosticEffects(t, b, manager)
		})
	}
}

func TestModelDiagnosticChangedConsentEffectsRefuseStart(t *testing.T) {
	for _, scenario := range []string{"fence", "loaded models", "restore models", "drained engines", "engine start", "second GPU", "artifacts", "requested settings", "build"} {
		t.Run(scenario, func(t *testing.T) {
			b, manager, p, observation := diagnosticPreviewFixture(t)
			preview, err := b.previewDiagnostic(context.Background(), p)
			if err != nil {
				t.Fatal(err)
			}
			switch scenario {
			case "fence":
				observation.EffectsFence = "changed"
			case "loaded models":
				observation.UnloadModels = []modelconfig.DiagnosticModelRef{{Engine: "llamacpp", ModelID: "another", ProfileKey: "another"}}
			case "restore models":
				r := b.nodePolicyRuntime()
				r.mu.Lock()
				r.current.Profiles = append(r.current.Profiles, nodepolicy.Profile{Key: "resident", Engine: "llamacpp", Model: "resident-model"})
				r.current.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "resident-model"}}
				r.mu.Unlock()
			case "drained engines":
				observation.DrainEngines = append(observation.DrainEngines, "other")
			case "engine start":
				observation.StartsEngine = true
			case "second GPU":
				observation.Identity.GPUs[1].Name = "replacement"
			case "artifacts":
				observation.Identity.Artifacts[0].Identity = "replacement"
			case "requested settings":
				value := int32(2048)
				observation.Identity.Requested.Threads = &value
			case "build":
				observation.Identity.NativeBuildID = "sha256:" + strings.Repeat("d", 64)
			}
			manager.setAnswer(func(method string, raw json.RawMessage) (any, string) { return observation, "" })
			if _, err := b.startDiagnostic(context.Background(), modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: p, PreviewID: preview.PreviewID, Consent: true}); err == nil {
				t.Fatal("changed effects accepted")
			}
			requireNoDiagnosticEffects(t, b, manager)
		})
	}
}

func TestModelDiagnosticReserveRefusalReleasesOnlySettingsLease(t *testing.T) {
	for _, reason := range []string{"busy", "effects conflict"} {
		t.Run(reason, func(t *testing.T) {
			b, manager, _, op := diagnosticAcquisitionFixture(t)
			manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				switch method {
				case "engine:diagnostic-reserve":
					return nil, reason
				case "engine:diagnostic-run-status":
					return map[string]string{"state": "none"}, ""
				}
				return nil, "unexpected effect"
			})
			if err := b.acquireDiagnostic(context.Background(), op); err == nil {
				t.Fatal("refused reserve accepted")
			}
			if err := b.cleanupDiagnostic(context.Background(), op); err != nil {
				t.Fatal(err)
			}
			requireDiagnosticReleased(t, b)
			for _, method := range manager.methods("engine:") {
				if !slices.Contains([]string{"engine:diagnostic-reserve", "engine:diagnostic-run-status"}, method) {
					t.Fatalf("refusal caused %s", method)
				}
			}
		})
	}
}

func TestModelDiagnosticCancelDuringDrainReleasesReservation(t *testing.T) {
	b, manager, proxy, op := diagnosticAcquisitionFixture(t)
	var reserved atomic.Bool
	run := op.run.Clone()
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:diagnostic-reserve":
			reserved.Store(true)
			return map[string]string{"runId": run.RunID, "token": "token", "state": "reserved"}, ""
		case "engine:diagnostic-run-status":
			return diagnosticStatusReply{Run: run, State: "reserved", Token: "token", WorkStopped: true}, ""
		case "engine:diagnostic-release":
			reserved.Store(false)
			return map[string]bool{"ok": true}, ""
		}
		return nil, "unexpected effect"
	})
	entered := make(chan struct{})
	var once sync.Once
	proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodAdmissionSnapshot {
			once.Do(func() { close(entered) })
			return map[string]any{"runId": "proxy-run", "active": 1, "queued": 0, "activeByEngine": map[string]int{"llamacpp": 1}}, holdSuccessReply
		}
		return map[string]bool{"ok": true}, ""
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- b.acquireDiagnostic(ctx, op) }()
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("drain did not start")
	}
	if !reserved.Load() {
		t.Fatal("drain preceded reservation")
	}
	cancel()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("cancelled drain accepted")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("drain did not join")
	}
	if err := b.cleanupDiagnostic(context.Background(), op); err != nil {
		t.Fatal(err)
	}
	requireDiagnosticReleased(t, b)
	if reserved.Load() || slices.Contains(manager.methods("engine:"), "engine:diagnostic-run") {
		t.Fatal("cancelled drain left authority or dispatched a run")
	}
}

func TestModelDiagnosticLostReserveReplyThenCancelSettlesAuthority(t *testing.T) {
	b, manager, _, op := diagnosticAcquisitionFixture(t)
	var reserved atomic.Bool
	var reserves atomic.Int32
	run := op.run.Clone()
	entered := make(chan struct{})
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:diagnostic-reserve":
			reserved.Store(true)
			result := map[string]string{"runId": run.RunID, "token": "token", "state": "reserved"}
			if reserves.Add(1) == 1 {
				close(entered)
				return result, holdSuccessReply
			}
			return result, ""
		case "engine:diagnostic-run-status":
			return diagnosticStatusReply{Run: run, State: "reserved", Token: "token", WorkStopped: true}, ""
		case "engine:diagnostic-release":
			reserved.Store(false)
			return map[string]bool{"ok": true}, ""
		}
		return nil, "unexpected effect"
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- b.acquireDiagnostic(ctx, op) }()
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("reserve did not start")
	}
	cancel()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("cancelled acquisition accepted")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("lost reserve was not reconciled")
	}
	if err := b.cleanupDiagnostic(context.Background(), op); err != nil {
		t.Fatal(err)
	}
	requireDiagnosticReleased(t, b)
	if reserved.Load() || reserves.Load() != 2 {
		t.Fatalf("reservation remains=%v attempts=%d", reserved.Load(), reserves.Load())
	}
}

func TestModelDiagnosticLostReserveReplyReconcilesBrokerLedger(t *testing.T) {
	b, manager, _, op := diagnosticAcquisitionFixture(t)
	close(op.done)
	lease, err := b.reserveBuildApplySettings()
	if err != nil {
		t.Fatal(err)
	}
	op.lease, op.reserveAttempted, op.reserveUncertain = lease, true, true
	op.reserveRequest = diagnosticReserveRequest{RunID: op.runID, Engine: op.run.Engine, ModelID: op.run.ModelID, ProfileKey: op.run.ProfileKey, ConfigurationKey: op.run.ConfigurationKey, EffectsFence: "effects"}
	var attempts atomic.Int32
	var reserved atomic.Bool
	run := op.run.Clone()
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:diagnostic-reserve":
			var p diagnosticReserveRequest
			if err := modelconfig.DecodeStrict(raw, &p); err != nil {
				return nil, err.Error()
			}
			if p.RunID != run.RunID || p.EffectsFence != "effects" {
				return nil, "not an idempotent retry"
			}
			reserved.Store(true)
			result := map[string]string{"runId": run.RunID, "token": "token", "state": "reserved"}
			if attempts.Add(1) == 1 {
				return result, holdSuccessReply
			}
			return result, ""
		case "engine:diagnostic-run-status":
			return diagnosticStatusReply{Run: run, State: "reserved", Token: "token", WorkStopped: true}, ""
		case "engine:diagnostic-release":
			reserved.Store(false)
			return map[string]bool{"ok": true}, ""
		}
		return nil, "unexpected effect"
	})
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	var result any
	if err := b.diagnosticCall(ctx, op, "engine:diagnostic-reserve", op.reserveRequest, &result); err == nil {
		t.Fatal("lost reply returned success")
	}
	if err := b.reconcileDiagnostics(context.Background(), op.worker); err != nil {
		t.Fatal(err)
	}
	requireDiagnosticReleased(t, b)
	if reserved.Load() || attempts.Load() != 2 {
		t.Fatal("broker reconciliation did not settle the idempotent reservation")
	}
}

func TestModelDiagnosticMissingPhaseEndDoesNotInventMinimum(t *testing.T) {
	at := time.Now()
	minimum, count := diagnosticMinimum([]diagnosticPoolSample{{at: at.Add(time.Second), available: 1}}, at, time.Time{})
	if minimum != "" || count != 0 {
		t.Fatalf("unfinished phase minimum=%q count=%d", minimum, count)
	}
}

func TestModelDiagnosticBuildAndRealizedFiniteIdentityAreBound(t *testing.T) {
	for _, scenario := range []string{"BuildInfo", "context", "slots", "CPU", "RAM", "companion"} {
		t.Run(scenario, func(t *testing.T) {
			expected := diagnosticIdentityFixture()
			actual := expected.Clone()
			switch scenario {
			case "BuildInfo":
				actual.BuildInfo = "different"
			case "context":
				actual.PerSlotContext++
			case "slots":
				actual.Slots++
			case "CPU":
				actual.CPUModel = "different"
			case "RAM":
				actual.TotalRAMBytes = "1"
			case "companion":
				actual.Artifacts = append(actual.Artifacts, modelconfig.ArtifactSummary{Role: "companion", Identity: "mmproj", Size: "1", Assurance: "sampled"})
			}
			if diagnosticRequestedIdentityMatches(actual, expected) {
				t.Fatal("changed identity matched consent")
			}
		})
	}
}

func TestModelDiagnosticRunSamplesAndRecordsThroughWorkerContracts(t *testing.T) {
	b, manager, _, op := diagnosticAcquisitionFixture(t)
	at := time.Now().UTC()
	var clock atomic.Int64
	clock.Store(at.UnixNano())
	b.diagnostics.now = func() time.Time { return time.Unix(0, clock.Load()).UTC() }
	identity := diagnosticIdentityFixture()
	identity.TotalRAMBytes = "10000"
	for i := range identity.GPUs {
		identity.GPUs[i].TotalMemoryBytes = "10000"
	}
	op.review.preview.DiagnosticIdentity = identity.Clone()
	initialMemory := diagnosticMemoryFixture(at.Add(-time.Second), 9000)
	for _, gpu := range identity.GPUs {
		domain := initialMemory.Domains[0]
		domain.ID, domain.Kind, domain.DeviceIDs = gpu.UUID, "dedicated", []string{gpu.UUID}
		initialMemory.Domains = append(initialMemory.Domains, domain)
	}
	seedDiagnosticTelemetry(t, b, time.Now(), initialMemory)
	op.review.preview.LowPromptTokens, op.review.preview.HighPromptTokens = 440, 880
	phase := func(target int32) *modelconfig.MeasuredPhase {
		cached, generated := int32(0), int32(128)
		first, end := float64(500), float64(3000)
		overlapped := true
		return &modelconfig.MeasuredPhase{Coverage: "complete", WallMS: 4000, Overlapped: &overlapped, Requests: []modelconfig.MeasuredRequest{{TargetPromptTokens: target, PromptTokens: &target, CachedTokens: &cached, GeneratedTokens: &generated, FirstTokenMS: &first, EndMS: &end}}}
	}
	values := []uint64{9000, 7000, 8000, 6000, 5500, 5000, 4500, 4000, 3500, 3600}
	markers := map[string]time.Time{"baseline-ready": at, "loaded": at.Add(6 * time.Second), "low-start": at.Add(10 * time.Second), "low-end": at.Add(12 * time.Second), "high-start": at.Add(14 * time.Second), "high-end": at.Add(18 * time.Second)}
	var accepted, stopped, reserved, stored, baselineAccepted, afterLoadAccepted atomic.Bool
	var steps atomic.Int32
	var record modelconfig.MeasuredProfile
	run := op.run.Clone()
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:diagnostic-reserve":
			var p diagnosticReserveRequest
			if err := modelconfig.DecodeStrict(raw, &p); err != nil {
				return nil, err.Error()
			}
			if p.RunID != run.RunID || p.ConfigurationKey != identity.ConfigurationKey || p.EffectsFence != op.review.effects {
				return nil, "reserve did not bind consent"
			}
			if p.TargetMemory == nil || p.TargetMemory.Validate() != nil || len(p.TargetMemory.Observation.Domains) != len(identity.Placement) {
				return nil, "reserve omitted hardware pool evidence"
			}
			if !reflect.DeepEqual(p.Hardware, diagnosticIdentityHardware(identity)) {
				return nil, "reserve omitted consented hardware"
			}
			reserved.Store(true)
			return map[string]string{"runId": run.RunID, "token": "token", "state": "reserved"}, ""
		case "engine:diagnostic-run":
			var p diagnosticRunRequest
			if err := modelconfig.DecodeStrict(raw, &p); err != nil {
				return nil, err.Error()
			}
			if !reserved.Load() || p.Token != "token" || p.RunID != run.RunID || p.ProfileKey != run.ProfileKey || p.ConfigurationKey != identity.ConfigurationKey || p.Low != 440 || p.High != 880 || p.Generate != 128 {
				return nil, "run contract did not match reservation"
			}
			accepted.Store(true)
			return diagnosticStatusReply{Run: run, State: "accepted", Token: "token"}, ""
		case "engine:diagnostic-baseline-done":
			var p struct {
				RunID string `json:"runId"`
				Token string `json:"token"`
				Phase string `json:"phase"`
			}
			if err := modelconfig.DecodeStrict(raw, &p); err != nil {
				return nil, err.Error()
			}
			if p.RunID != run.RunID || p.Token != "token" {
				return nil, "sample acknowledgement authority mismatch"
			}
			switch {
			case p.Phase == "baseline" && steps.Load() == 3:
				baselineAccepted.Store(true)
			case p.Phase == "after-load" && steps.Load() == 5:
				afterLoadAccepted.Store(true)
			default:
				return nil, "sample acknowledgement preceded required samples"
			}
			return map[string]bool{"ok": true}, ""
		case "engine:diagnostic-run-status":
			if !accepted.Load() {
				return diagnosticStatusReply{Run: run, State: "reserved", Token: "token", WorkStopped: true}, ""
			}
			step := int(steps.Add(1)) - 1
			if step > 2 && !baselineAccepted.Load() {
				return nil, "baseline barrier bypassed"
			}
			if step > 4 && !afterLoadAccepted.Load() {
				return nil, "after-load barrier bypassed"
			}
			now := at.Add(time.Duration(step) * 2 * time.Second)
			clock.Store(now.UnixNano())
			seen := make(map[string]time.Time)
			for name, marker := range markers {
				if !marker.After(now) {
					seen[name] = marker
				}
			}
			current := run.Clone()
			current.Phase = "baseline"
			if step >= 3 {
				current.Phase = "loading"
			}
			if step >= 5 {
				current.Phase = "low"
			}
			if step >= 7 {
				current.Phase = "high"
			}
			if step >= len(values) {
				stopped.Store(true)
				current.Phase, current.Outcome = "done", "completed"
				return diagnosticStatusReply{Run: current, State: "cleanup-complete", Token: "token", WorkStopped: true, Identity: &identity, Markers: seen, Low: phase(440), High: phase(880)}, ""
			}
			memory := diagnosticMemoryFixture(now, values[step])
			for _, gpu := range identity.GPUs {
				domain := memory.Domains[0]
				domain.ID, domain.Kind, domain.DeviceIDs = gpu.UUID, "dedicated", []string{gpu.UUID}
				memory.Domains = append(memory.Domains, domain)
			}
			if err := ingestDiagnosticTelemetry(b, time.Now(), memory); err != nil {
				return nil, err.Error()
			}
			return diagnosticStatusReply{Run: current, State: "running", Token: "token", Markers: seen}, ""
		case "engine:inspect-model":
			inspection := diagnosticInspectionFixture(run, identity)
			if err := inspection.Validate(); err != nil {
				return nil, err.Error()
			}
			return inspection, ""
		case "engine:record-measured-memory":
			var p diagnosticRecordParams
			if err := modelconfig.DecodeStrict(raw, &p); err != nil {
				return nil, err.Error()
			}
			if !stopped.Load() || p.RunID != run.RunID || p.Token != "token" || p.Profile.ConfigurationKey != identity.ConfigurationKey || p.Profile.ExecutionKey != "" || !reflect.DeepEqual(p.Profile.DiagnosticIdentity, identity) {
				return nil, "record authority or identity mismatch"
			}
			if err := p.Profile.Validate(); err != nil {
				return nil, err.Error()
			}
			record = p.Profile.Clone()
			stored.Store(true)
			return map[string]bool{"ok": true}, ""
		case "engine:diagnostic-release":
			if !stopped.Load() || !stored.Load() {
				return nil, "release preceded stopped record"
			}
			reserved.Store(false)
			return map[string]bool{"ok": true}, ""
		}
		return nil, "unexpected method: " + method
	})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	op.cancel = cancel
	t.Cleanup(cancel)
	go b.executeDiagnostic(ctx, op)
	select {
	case <-op.done:
	case <-time.After(35 * time.Second):
		t.Fatal("run and owned ticker did not join")
	}
	if reserved.Load() || !stored.Load() || record.Outcome != "completed" {
		t.Fatalf("run did not persist/release: reserved=%v stored=%v outcome=%s phase=%s reason=%s methods=%v", reserved.Load(), stored.Load(), record.Outcome, op.run.Phase, op.run.Reason, manager.methods("engine:"))
	}
	requireDiagnosticReleased(t, b)
	if len(record.Pools) != len(identity.Placement) {
		t.Fatalf("recorded pool set=%+v", record.Pools)
	}
	for _, pool := range record.Pools {
		if pool.BaselineAvailableBytes != "8000" || pool.AfterLoadAvailableBytes != "5000" || pool.LowMinAvailableBytes != "4500" || pool.HighMinAvailableBytes != "3500" || pool.PeakUsedBytes != "4500" {
			t.Fatalf("pool=%+v", pool)
		}
	}
	if record.High == nil || record.High.OverlapWindowSampled == nil || !*record.High.OverlapWindowSampled || record.High.Coverage != "complete" {
		t.Fatalf("sampled overlap=%+v", record.High)
	}
	status, err := b.diagnosticStatus("")
	if err != nil || status == nil || status.Result == nil || status.Result.RunID != record.RunID {
		t.Fatalf("public terminal result=%+v error=%v", status, err)
	}
}

func TestModelDiagnosticPublicRunClockDoesNotRegressOnManagerAdoption(t *testing.T) {
	b, _, op := diagnosticBrokerFixture(t)
	at := time.Now().UTC()
	op.run.StartedAt = at.Format(time.RFC3339Nano)
	op.started = at
	b.diagnostics.now = func() time.Time { return at.Add(12 * time.Second) }
	before, err := b.diagnosticStatus("")
	if err != nil {
		t.Fatal(err)
	}
	managerRun := op.run.Clone()
	managerRun.StartedAt, managerRun.ElapsedMS, managerRun.Phase = at.Add(10*time.Second).Format(time.RFC3339Nano), 500, "low"
	b.setDiagnosticRun(op, managerRun)
	after, err := b.diagnosticStatus("")
	if err != nil {
		t.Fatal(err)
	}
	if after.StartedAt != before.StartedAt || after.ElapsedMS < before.ElapsedMS || after.ElapsedMS != 12000 {
		t.Fatalf("clock regressed: before=%+v after=%+v", before, after)
	}
}

func TestModelDiagnosticPublicPushStripsManagerAuthority(t *testing.T) {
	b, out := newPolicyBroker(t)
	stop := b.startEngineNotificationRelay(context.Background())
	t.Cleanup(stop)
	source := b.activateEngineManagerSource()
	peer, _ := newFakeWorkerPipe(t)
	worker := &rpcWorker{peer: peer, done: make(chan struct{})}
	b.setEngineMgr(worker)
	b.bindEngineMgr(worker, source)
	b.resumeEngineNotifications()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	run := modelconfig.DiagnosticRun{RunID: "public", Engine: "llamacpp", ModelID: "fixture.gguf", ProfileKey: "fixture", ConfigurationKey: strings.Repeat("a", 64), Phase: "baseline", Outcome: "running", StartedAt: time.Now().Format(time.RFC3339Nano)}
	op := &brokerDiagnostic{runID: run.RunID, run: run.Clone(), worker: worker, cancel: cancel, done: make(chan struct{}), wake: make(chan struct{}, 1)}
	b.diagnostics.active = op
	b.forwardEngineNotificationFrom(source, "engine:diagnostic-changed", settingsJSON(map[string]any{"run": run, "state": "running", "token": "private-authority", "workStopped": false, "markers": map[string]time.Time{"baseline-ready": time.Now()}, "observation": map[string]int{"slots": 1}}))
	frame := out.waitFrame(t, "public diagnostic progress", func(message Message) bool { return message.Method == "engine:diagnostic-changed" })
	var public modelconfig.DiagnosticRun
	if err := modelconfig.DecodeStrict(frame.Params, &public); err != nil {
		t.Fatalf("internal fields leaked: %v", err)
	}
	if err := public.Validate(); err != nil {
		t.Fatal(err)
	}
	if public.RunID != run.RunID {
		t.Fatalf("public run=%+v", public)
	}
	select {
	case <-ctx.Done():
		t.Fatal("notification cancelled broker work")
	default:
	}
}

func TestModelDiagnosticDrainingProgressIsPublishedBeforeIdleWait(t *testing.T) {
	b, manager, proxy, op := diagnosticAcquisitionFixture(t)
	started, err := time.Parse(time.RFC3339Nano, op.run.StartedAt)
	if err != nil {
		t.Fatal(err)
	}
	b.diagnostics.now = func() time.Time { return started.Add(3 * time.Second) }
	out := &syncBuffer{}
	b.codec = NewCodec(out)
	source := b.activateEngineManagerSource()
	b.bindEngineMgr(op.worker, source)
	stop := b.startEngineNotificationRelay(context.Background())
	t.Cleanup(stop)
	b.resumeEngineNotifications()
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:diagnostic-reserve" {
			return map[string]string{"runId": op.runID, "token": "token", "state": "reserved"}, ""
		}
		return nil, "unexpected effect"
	})
	entered := make(chan struct{})
	proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodAdmissionSnapshot {
			close(entered)
			return nil, holdSuccessReply
		}
		return map[string]bool{"ok": true}, ""
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- b.acquireDiagnostic(ctx, op) }()
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("idle wait did not begin")
	}
	frame := out.waitFrame(t, "draining push before idle reply", func(message Message) bool { return message.Method == "engine:diagnostic-changed" })
	var progress modelconfig.DiagnosticRun
	if err := modelconfig.DecodeStrict(frame.Params, &progress); err != nil {
		t.Fatal(err)
	}
	if progress.Phase != "draining" || progress.ElapsedMS <= 0 {
		t.Fatalf("public draining progress=%+v", progress)
	}
	status, err := b.diagnosticStatus("")
	if err != nil || status == nil || status.Phase != "draining" {
		t.Fatalf("draining progress=%+v error=%v", status, err)
	}
	cancel()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("draining owner did not join")
	}
}

func TestModelDiagnosticReleaseFailureReestablishesDrainsUntilRetry(t *testing.T) {
	b, manager, op := diagnosticBrokerFixture(t)
	run := op.run.Clone()
	run.Outcome, run.Phase = "cancelled", "done"
	setDiagnosticContractAnswer(t, manager, op, func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:diagnostic-run-status" {
			return diagnosticStatusReply{Run: run, State: "cleanup-complete", Token: "reservation", WorkStopped: true}, ""
		}
		return nil, "unexpected method"
	})
	manager.mu.Lock()
	base := manager.answer
	manager.mu.Unlock()
	var releases atomic.Int32
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:diagnostic-release" && releases.Add(1) == 1 {
			return nil, "release uncertain"
		}
		return base(method, raw)
	})
	if err := b.cleanupDiagnostic(context.Background(), op); err == nil {
		t.Fatal("release failure was ignored")
	}
	requireDiagnosticHeld(t, b, op)
	if err := b.cleanupDiagnostic(context.Background(), op); err != nil {
		t.Fatal(err)
	}
	requireDiagnosticReleased(t, b)
	if releases.Load() != 2 {
		t.Fatalf("explicit release calls=%d", releases.Load())
	}
	var stores int
	for _, method := range manager.methods("engine:") {
		if method == "engine:record-measured-memory" {
			stores++
		}
	}
	if stores != 1 {
		t.Fatalf("release retry repeated persistence: %d", stores)
	}
}

func TestModelDiagnosticLostReleaseReplyReconcilesNoneAfterStopAndRecord(t *testing.T) {
	b, manager, op := diagnosticBrokerFixture(t)
	run := op.run.Clone()
	run.Outcome, run.Phase = "cancelled", "done"
	var released atomic.Bool
	setDiagnosticContractAnswer(t, manager, op, func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:diagnostic-run-status" {
			if released.Load() {
				return map[string]string{"state": "none"}, ""
			}
			return diagnosticStatusReply{Run: run, State: "cleanup-complete", Token: "reservation", WorkStopped: true}, ""
		}
		return nil, "unexpected method"
	})
	manager.mu.Lock()
	base := manager.answer
	manager.mu.Unlock()
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, refusal := base(method, raw)
		if method == "engine:diagnostic-release" && refusal == "" {
			released.Store(true)
			return result, holdSuccessReply
		}
		return result, refusal
	})
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if err := b.cleanupDiagnostic(ctx, op); err == nil {
		t.Fatal("lost release reply returned success")
	}
	requireDiagnosticHeld(t, b, op)
	if err := b.reconcileDiagnostics(context.Background(), op.worker); err != nil {
		t.Fatal(err)
	}
	requireDiagnosticReleased(t, b)
}

func TestModelDiagnosticManagerReplacementKeepsHoldAndRetryAuthority(t *testing.T) {
	b, _, op := diagnosticBrokerFixture(t)
	peer, manager := newFakeWorkerPipe(t)
	replacement := &rpcWorker{peer: peer, done: make(chan struct{})}
	b.setEngineMgr(replacement)
	run := op.run.Clone()
	run.Phase, run.Outcome = "restoring", "cleanup-failed"
	var complete atomic.Bool
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:diagnostic-run-status":
			if complete.Load() {
				terminal := run.Clone()
				terminal.Phase, terminal.Outcome = "done", "failed"
				return diagnosticStatusReply{Run: terminal, State: "cleanup-complete", Token: "recovered", WorkStopped: true}, ""
			}
			return diagnosticStatusReply{Run: run, State: "cleanup-failed", Token: "recovered"}, ""
		case "engine:diagnostic-abort":
			var p diagnosticAuthority
			if err := modelconfig.DecodeStrict(raw, &p); err != nil {
				return nil, err.Error()
			}
			if p.Token != "recovered" || p.RunID != run.RunID {
				return nil, "retired authority"
			}
			complete.Store(true)
			terminal := run.Clone()
			terminal.Phase, terminal.Outcome = "done", "failed"
			return diagnosticStatusReply{Run: terminal, State: "cleanup-complete", Token: "recovered", WorkStopped: true}, ""
		case "engine:diagnostic-release":
			if !complete.Load() {
				return nil, "release preceded recovered join"
			}
			return map[string]bool{"ok": true}, ""
		}
		return nil, "unexpected recovery method: " + method
	})
	// Reconstructed cleanup owns the existing settings lock. Recovery must
	// expose its Retry surface instead of waiting for its own held mutex.
	b.settingsApplyMu.Lock()
	op.drain.settingsHeld = true
	ready := make(chan bool, 1)
	go func() { ready <- b.recoverReplacementEngineMgr(replacement) }()
	select {
	case ok := <-ready:
		if !ok {
			t.Fatal("replacement recovery refused checkpoint")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("replacement recovery deadlocked on retained settings lease")
	}
	requireDiagnosticHeld(t, b, op)
	if op.worker != replacement || op.token != "recovered" {
		t.Fatal("replacement did not adopt checkpoint authority")
	}
	// Task12 owns the interrupted record; the recovered broker must not
	// manufacture another measurement from pre-restart samples.
	if _, err := b.cancelDiagnostic(context.Background(), run.RunID); err != nil {
		t.Fatal(err)
	}
	requireDiagnosticReleased(t, b)
	if !complete.Load() {
		t.Fatal("Retry cleanup did not join replacement manager")
	}
}

func TestModelDiagnosticCleanupFailureSurvivesBrokerRestart(t *testing.T) {
	b, _ := newPolicyBroker(t)
	peer, manager := newFakeWorkerPipe(t)
	worker := &rpcWorker{peer: peer, done: make(chan struct{})}
	b.setEngineMgr(worker)
	run := modelconfig.DiagnosticRun{RunID: "restart", Engine: "llamacpp", ModelID: "fixture.gguf", ProfileKey: "fixture", ConfigurationKey: strings.Repeat("a", 64), StartedAt: time.Now().Format(time.RFC3339Nano), Phase: "restoring", Outcome: "cleanup-failed"}
	var retry atomic.Bool
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:diagnostic-abort" {
			retry.Store(true)
		}
		return diagnosticStatusReply{Run: run, State: "cleanup-failed", Token: "recovered"}, ""
	})
	if err := b.reconcileDiagnostics(context.Background(), worker); err != nil {
		t.Fatal(err)
	}
	b.diagnostics.mu.Lock()
	op := b.diagnostics.active
	b.diagnostics.mu.Unlock()
	if op == nil {
		t.Fatal("checkpoint did not recreate public operation")
	}
	if _, err := b.cancelDiagnostic(context.Background(), run.RunID); err == nil {
		t.Fatal("failed recovered cleanup returned success")
	}
	requireDiagnosticHeld(t, b, op)
	if !retry.Load() {
		t.Fatal("recovered Retry did not call abort")
	}
	if err := b.reconcileDiagnostics(context.Background(), worker); err != nil {
		t.Fatal(err)
	}
	requireDiagnosticHeld(t, b, op)
	// Finish the synthetic checkpoint owner before the fixture exits.
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		terminal := run.Clone()
		terminal.Phase, terminal.Outcome = "done", "failed"
		return diagnosticStatusReply{Run: terminal, State: "cleanup-complete", Token: "recovered", WorkStopped: true}, ""
	})
	if _, err := b.cancelDiagnostic(context.Background(), run.RunID); err != nil {
		t.Fatal(err)
	}
	requireDiagnosticReleased(t, b)
}

func TestModelDiagnosticSampleOutsideOverlapMakesHighPartial(t *testing.T) {
	at := time.Now().UTC()
	identity := diagnosticIdentityFixture()
	identity.Placement = identity.Placement[:1]
	target, cached := int32(880), int32(0)
	first, end := float64(100), float64(200)
	overlapped := true
	high := &modelconfig.MeasuredPhase{Coverage: "complete", Overlapped: &overlapped, Requests: []modelconfig.MeasuredRequest{{TargetPromptTokens: target, PromptTokens: &target, CachedTokens: &cached, FirstTokenMS: &first, EndMS: &end}}}
	samples := diagnosticSamples{pools: map[string][]diagnosticPoolSample{"ram": {{at: at.Add(time.Second), available: 100}, {at: at.Add(2 * time.Second), available: 90}}}}
	status := diagnosticManagerStatus{Run: &modelconfig.DiagnosticRun{RunID: "overlap", Outcome: "completed"}, High: high, Markers: map[string]time.Time{"high-start": at, "high-end": at.Add(3 * time.Second)}}
	profile, err := samples.measured(identity, status, at.Add(4*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if profile.High == nil || profile.High.OverlapWindowSampled == nil || *profile.High.OverlapWindowSampled || profile.High.Coverage != "partial" {
		t.Fatalf("uncovered overlap=%+v", profile.High)
	}
}

func TestModelDiagnosticUMASamplingUsesOneVerifiedSharedPool(t *testing.T) {
	b, _ := newPolicyBroker(t)
	at := time.Now().UTC().Add(-3 * time.Second)
	identity := diagnosticIdentityFixture()
	identity.Placement = []modelconfig.DiagnosticPlacement{{DomainID: "unified", Kind: "shared"}}
	var samples diagnosticSamples
	for i, value := range []uint64{9000, 7000, 8000} {
		now := at.Add(time.Duration(i) * time.Second)
		memory := diagnosticMemoryFixture(now, value)
		memory.Domains[0].ID, memory.Domains[0].Kind, memory.Domains[0].DeviceIDs = "unified", "shared", []string{"gpu-a"}
		seedDiagnosticTelemetry(t, b, time.Now(), memory)
		observation, err := b.diagnosticMemory()
		if err != nil {
			t.Fatal(err)
		}
		if err := samples.observe(observation, now); err != nil {
			t.Fatal(err)
		}
	}
	status := diagnosticManagerStatus{Run: &modelconfig.DiagnosticRun{RunID: "uma", Outcome: "failed"}, Markers: map[string]time.Time{"baseline-ready": at, "loaded": at.Add(3 * time.Second)}}
	profile, err := samples.measured(identity, status, at.Add(4*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if len(profile.Pools) != 1 || profile.Pools[0].DomainID != "unified" || profile.Pools[0].BaselineAvailableBytes != "8000" {
		t.Fatalf("UMA double-counted or lost baseline: %+v", profile.Pools)
	}
}

func TestModelDiagnosticExpiredOrGenerationChangedPreviewHasNoEffects(t *testing.T) {
	for _, scenario := range []string{"expired", "maintenance generation"} {
		t.Run(scenario, func(t *testing.T) {
			b, manager, p, _ := diagnosticPreviewFixture(t)
			preview, err := b.previewDiagnostic(context.Background(), p)
			if err != nil {
				t.Fatal(err)
			}
			if scenario == "expired" {
				b.diagnostics.now = func() time.Time { return time.Now().Add(diagnosticPreviewTTL) }
			} else {
				lease, err := b.reserveBuildApplySettings()
				if err != nil {
					t.Fatal(err)
				}
				lease.Release()
			}
			if _, err := b.startDiagnostic(context.Background(), modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: p, PreviewID: preview.PreviewID, Consent: true}); err == nil {
				t.Fatal("stale consent accepted")
			}
			requireNoDiagnosticEffects(t, b, manager)
		})
	}
}

func TestNodeDrainPreservesPreexistingOwnerAndWaitsForEveryEngine(t *testing.T) {
	b, _, proxy, _ := diagnosticAcquisitionFixture(t)
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.drains["llamacpp"] = true
	r.mu.Unlock()
	var snapshots atomic.Int32
	proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodAdmissionSnapshot {
			count := snapshots.Add(1)
			active := 0
			if count < 3 {
				active = int(count)
			}
			return map[string]any{"runId": "proxy-run", "active": 0, "queued": 0, "activeByEngine": map[string]int{"custom": active}}, ""
		}
		return map[string]bool{"ok": true}, ""
	})
	lease, err := b.drainNode(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	added := slices.Clone(lease.added)
	if err := lease.release(context.Background()); err != nil {
		t.Fatal(err)
	}
	if snapshots.Load() != 3 {
		t.Fatalf("idle accepted before all engine activity ended: %d", snapshots.Load())
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.drains["llamacpp"] {
		t.Fatal("released another owner's drain")
	}
	for _, engine := range added {
		if r.drains[engine] {
			t.Fatalf("owned drain survived release: %s", engine)
		}
	}
}

func TestNodeDrainProxyReplacementAbortsAndReleasesOnlyItsDrains(t *testing.T) {
	b, _, proxy, _ := diagnosticAcquisitionFixture(t)
	peer, _ := newFakeWorkerPipe(t)
	replacement := &proxyProcess{peer: peer}
	proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodAdmissionSnapshot {
			b.setProxyProcessHandle(replacement)
			return map[string]any{"runId": "proxy-run", "active": 0, "queued": 0, "activeByEngine": map[string]int{}}, ""
		}
		return map[string]bool{"ok": true}, ""
	})
	lease, err := b.drainNode(context.Background())
	if err == nil {
		t.Fatal("replacement proxy accepted retired idle proof")
	}
	if lease != nil {
		t.Fatal("unactivated replacement left an unreleased drain ledger")
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.drains) != 0 {
		t.Fatalf("aborted node drain leaked registry: %+v", r.drains)
	}
}

func TestModelDiagnosticPublicStatusWithoutRunIDUsesBrokerOperation(t *testing.T) {
	b, manager, op := diagnosticBrokerFixture(t)
	out := &syncBuffer{}
	b.codec = NewCodec(out)
	b.handleMessage(clientRequest(1, "engine:diagnostic-status", map[string]string{}))
	response := out.waitFrame(t, "active public status", isResponse(1))
	if response.Error != nil {
		t.Fatalf("status error=%+v", response.Error)
	}
	var run modelconfig.DiagnosticRun
	if err := modelconfig.DecodeStrict(response.Result, &run); err != nil {
		t.Fatal(err)
	}
	if run.RunID != op.runID {
		t.Fatalf("public active run=%+v", run)
	}
	if calls := manager.snapshot(); len(calls) != 0 {
		t.Fatalf("public status relayed internally: %+v", calls)
	}
}

func TestModelDiagnosticTerminalCheckpointClearsOnlyAfterProxyReplay(t *testing.T) {
	b, _ := newPolicyBroker(t)
	peer, manager := newFakeWorkerPipe(t)
	worker := &rpcWorker{peer: peer, done: make(chan struct{})}
	b.setEngineMgr(worker)
	run := modelconfig.DiagnosticRun{RunID: "stopped-checkpoint", Engine: "llamacpp", ModelID: "fixture.gguf", ProfileKey: "fixture", ConfigurationKey: strings.Repeat("a", 64), StartedAt: time.Now().Format(time.RFC3339Nano), Phase: "done", Outcome: "failed"}
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:diagnostic-release" {
			return map[string]bool{"ok": true}, ""
		}
		if method == "engine:diagnostic-run-status" {
			return diagnosticStatusReply{Run: run, State: "cleanup-complete", Token: "recovered", WorkStopped: true}, ""
		}
		return nil, "unexpected checkpoint method"
	})
	pp, proxy, _ := attachAckingProxy(t, b)
	if !b.replayPolicyToProxy(context.Background(), pp) {
		t.Fatal("stopped checkpoint prevented proxy replay")
	}
	manager.waitFor("post-replay reservation release", func(calls []policyCall) bool {
		for _, call := range calls {
			if call.method == "engine:diagnostic-release" {
				return true
			}
		}
		return false
	})
	b.diagnostics.mu.Lock()
	op := b.diagnostics.active
	b.diagnostics.mu.Unlock()
	if op != nil {
		op.cleanupMu.Lock()
		op.cleanupMu.Unlock()
	}
	requireDiagnosticReleased(t, b)
	drains := map[string]bool{}
	policySeen := false
	for _, call := range proxy.snapshot() {
		if call.method == nodepolicy.MethodSetEngineDrain {
			var p nodepolicy.SetEngineDrainParams
			if err := json.Unmarshal(call.params, &p); err != nil {
				t.Fatal(err)
			}
			drains[p.Engine] = p.Drain
		}
		if call.method == nodepolicy.MethodSetPolicy {
			policySeen = true
			for _, profile := range b.frontedEngineProfiles() {
				if !drains[profile.Name] {
					t.Fatalf("cleanup cleared %s before policy activation", profile.Name)
				}
			}
		}
	}
	if !policySeen {
		t.Fatal("no policy replay observed")
	}
}
