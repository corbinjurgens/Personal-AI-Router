// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-shared/performancewire"
)

func measurementBrokerFixture(t *testing.T) (*Broker, *fakeWorker) {
	t.Helper()
	b, _ := newPolicyBroker(t)
	manager := attachFakeEngine(t, b)
	source := b.activateEngineManagerSource()
	b.bindEngineMgr(b.getEngineMgr(), source)
	b.observeModelIdentity(source, engineManagerSnapshot("llamacpp"))
	b.observeSettingsIdentity([]settings.Snapshot{appliedSnapshot("llamacpp", 3)})
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Profiles = []nodepolicy.Profile{{Key: "exact-profile", Engine: "llamacpp", Model: "model.gguf"}}
	r.mu.Unlock()
	seedDiagnosticHardware(t, b, diagnosticIdentityFixture(), time.Now())
	measurement := modelconfig.MeasurementIdentity{Digest: strings.Repeat("a", 64), Complete: true}
	manager.setEngineAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method != "engine:inspect-model" {
			return nil, "unexpected method: " + method
		}
		var p diagnosticInspectEnvelope
		if err := modelconfig.DecodeStrict(raw, &p); err != nil {
			return nil, err.Error()
		}
		if p.NativeInitAllowed || p.Hardware == nil || p.Memory == nil || p.Request.ProfileKey != "exact-profile" {
			return nil, "background inspection requested effects or lost identity"
		}
		return modelconfig.InspectResult{Engine: p.Request.Engine, ModelID: p.Request.ModelID, Status: "resolved", Artifacts: diagnosticIdentityFixture().Artifacts, Estimate: modelconfig.Estimate{Status: "unknown"}, MeasurementIdentity: &measurement}, ""
	})
	return b, manager
}

func requireMeasurement(t *testing.T, b *Broker, complete bool) map[string]map[string]nodepolicy.MeasurementState {
	t.Helper()
	states := b.currentMeasurementIdentities()
	got := states["llamacpp"]["exact-profile"].Identity
	if got.Complete != complete || complete && got.Digest != strings.Repeat("a", 64) {
		t.Fatalf("measurement: %+v, want complete=%t", got, complete)
	}
	return states
}

func TestMeasurementBackgroundInspectionAndReplayPreserveObservationLifetime(t *testing.T) {
	b, _ := measurementBrokerFixture(t)
	b.refreshPerformanceConfigurations(context.Background())
	before := requireMeasurement(t, b, true)
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	replay := r.modelIdentityLocked()
	r.mu.Unlock()
	if !reflect.DeepEqual(replay.Measurements, before) {
		t.Fatalf("replay changed lifetime or identity: %+v != %+v", replay.Measurements, before)
	}
	if replay.Measurements["llamacpp"]["exact-profile"].ValidityMS > noderec.MemoryFreshMS {
		t.Fatal("extended underlying memory lifetime")
	}
	var wire map[string]json.RawMessage
	raw, err := json.Marshal(replay)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire) != 2 || wire["engines"] == nil || wire["measurements"] == nil {
		t.Fatalf("control envelope changed strict identity: %v", wire)
	}
}

func TestPerformanceConfigurationOwnerWithdrawsImmediatelyAndReobservesUnchanged(t *testing.T) {
	b, _ := measurementBrokerFixture(t)
	f := newBrokerPerformanceFixture(t)
	p := f.b.performance
	b.performance = p
	p.configuration = b.performanceConfigurationSnapshot
	b.refreshPerformanceConfigurations(context.Background())
	sample := performanceConfiguredSample(1, strings.Repeat("a", 64), "retained", 40)
	sample.Key.Model = "exact-profile"
	if err := p.store.Record(sample); err != nil {
		t.Fatal(err)
	}
	p.flush()
	requirePerformanceApplicability(t, p, map[string]performancewire.Applicability{"retained": performancewire.ApplicabilityCurrent})
	prior := p.store.Stats().PublishedRevision
	b.forgetSettingsIdentity("llamacpp")
	p.mu.Lock()
	withdrawn := p.checkpoint
	p.mu.Unlock()
	if !withdrawn {
		t.Fatal("settings invalidation did not immediately schedule checkpoint")
	}
	if _, raw := p.publication(); raw != nil {
		t.Fatal("settings invalidation left old current publication available")
	}
	requirePerformanceApplicability(t, p, map[string]performancewire.Applicability{"retained": performancewire.ApplicabilityUnknown})
	// A new settings journal revision alone changes aggregation provenance,
	// while the inspection owner can re-attest the same semantic identity.
	b.observeSettingsIdentity([]settings.Snapshot{appliedSnapshot("llamacpp", 4)})
	b.refreshPerformanceConfigurations(context.Background())
	requirePerformanceApplicability(t, p, map[string]performancewire.Applicability{"retained": performancewire.ApplicabilityUnknown})
	p.flush()
	requirePerformanceApplicability(t, p, map[string]performancewire.Applicability{"retained": performancewire.ApplicabilityCurrent})
	if p.store.Stats().PublishedRevision <= prior || p.store.Stats().Samples != 1 {
		t.Fatal("reobservation did not reuse retained sample with a new durable revision")
	}
	_, frozen := p.publication()
	revision := p.store.Stats().PublishedRevision
	observed := p.store.Local().Entries[0].NewestObservedAtMs
	b.refreshPerformanceConfigurations(context.Background())
	p.flush()
	_, replayed := p.publication()
	if !bytes.Equal(frozen, replayed) || p.store.Stats().PublishedRevision != revision || p.store.Local().Entries[0].NewestObservedAtMs != observed {
		t.Fatal("unchanged reobservation changed frozen bytes, revision or sample time")
	}
}

func TestPerformanceApplicabilityRejectsChangedIdentityInputs(t *testing.T) {
	for _, tc := range []struct {
		name     string
		edit     func(*modelconfig.DiagnosticIdentity)
		semantic string
		want     performancewire.Applicability
	}{
		{name: "semantic settings and artifacts", semantic: strings.Repeat("d", 64), want: performancewire.ApplicabilityHistorical},
		{name: "native build", edit: func(i *modelconfig.DiagnosticIdentity) { i.NativeBuildID = "sha256:" + strings.Repeat("d", 64) }, want: performancewire.ApplicabilityHistorical},
		{name: "CPU", edit: func(i *modelconfig.DiagnosticIdentity) { i.CPUModel = "replacement CPU" }, want: performancewire.ApplicabilityHistorical},
		{name: "total RAM", edit: func(i *modelconfig.DiagnosticIdentity) { i.TotalRAMBytes = "137438953472" }, want: performancewire.ApplicabilityHistorical},
		{name: "GPU identity", edit: func(i *modelconfig.DiagnosticIdentity) {
			i.GPUs[1].UUID = "replacement"
			i.Placement[2].DeviceID = "replacement"
		}, want: performancewire.ApplicabilityHistorical},
		{name: "GPU total memory", edit: func(i *modelconfig.DiagnosticIdentity) { i.GPUs[1].TotalMemoryBytes = "34359738368" }, want: performancewire.ApplicabilityHistorical},
		{name: "realized context", edit: func(i *modelconfig.DiagnosticIdentity) { i.PerSlotContext = 2048 }, want: performancewire.ApplicabilityHistorical},
		{name: "realized slots", edit: func(i *modelconfig.DiagnosticIdentity) { i.Slots = 2 }, want: performancewire.ApplicabilityHistorical},
		{name: "realized placement", edit: func(i *modelconfig.DiagnosticIdentity) { i.Placement = i.Placement[:1] }, want: performancewire.ApplicabilityHistorical},
		{name: "cold native fit", edit: func(i *modelconfig.DiagnosticIdentity) { i.PerSlotContext = 0; i.Placement = nil }, want: performancewire.ApplicabilityUnknown},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, manager := measurementBrokerFixture(t)
			f := newBrokerPerformanceFixture(t)
			p := f.b.performance
			b.performance = p
			p.configuration = b.performanceConfigurationSnapshot
			base := diagnosticIdentityFixture()
			semantic := strings.Repeat("c", 64)
			original := modelconfig.MeasurementFromDiagnostic(semantic, base)
			if !original.Complete {
				t.Fatal("missing complete positive control")
			}
			answer := func(identity modelconfig.MeasurementIdentity) {
				manager.setEngineAnswer(func(method string, raw json.RawMessage) (any, string) {
					if method != "engine:inspect-model" {
						return nil, "unexpected inspection method"
					}
					return modelconfig.InspectResult{Engine: "llamacpp", ModelID: "model.gguf", Status: "resolved", Artifacts: base.Artifacts, Estimate: modelconfig.Estimate{Status: "unknown"}, MeasurementIdentity: &identity}, ""
				})
			}
			answer(original)
			b.refreshPerformanceConfigurations(context.Background())
			sample := performanceConfiguredSample(1, original.Digest, "retained-fast", 200)
			sample.Key.Model = "exact-profile"
			if err := p.store.Record(sample); err != nil {
				t.Fatal(err)
			}
			p.flush()
			requirePerformanceApplicability(t, p, map[string]performancewire.Applicability{"retained-fast": performancewire.ApplicabilityCurrent})
			if tc.edit != nil {
				tc.edit(&base)
			}
			if tc.semantic != "" {
				semantic = tc.semantic
			}
			changed := modelconfig.MeasurementFromDiagnostic(semantic, base)
			if changed == original {
				t.Fatal("identity owner did not detect changed input")
			}
			answer(changed)
			b.refreshPerformanceConfigurations(context.Background())
			requirePerformanceApplicability(t, p, map[string]performancewire.Applicability{"retained-fast": tc.want})
			if _, raw := p.publication(); raw != nil {
				t.Fatal("configuration change published old current frame")
			}
			p.flush()
			_, raw := p.publication()
			summary, err := performancewire.DecodeSummary(raw)
			if err != nil || len(summary.Entries) != 1 || summary.Entries[0].Applicability != tc.want || summary.Entries[0].EngineDecode.TokensPerSecond != 200 {
				t.Fatalf("changed-input publication = %+v, %v", summary, err)
			}
		})
	}
}

func TestPerformancePolicySaveImmediatelyCheckpointsConfiguration(t *testing.T) {
	b, _ := measurementBrokerFixture(t)
	f := newBrokerPerformanceFixture(t)
	p := f.b.performance
	b.performance = p
	p.configuration = b.performanceConfigurationSnapshot
	b.refreshPerformanceConfigurations(context.Background())
	sample := performanceConfiguredSample(1, strings.Repeat("a", 64), "retained", 40)
	sample.Key.Model = "exact-profile"
	if err := p.store.Record(sample); err != nil {
		t.Fatal(err)
	}
	p.flush()
	p.configurationWake = make(chan struct{}, 1)
	p.mu.Lock()
	pending := p.checkpoint
	p.mu.Unlock()
	if pending {
		t.Fatal("positive control checkpoint did not finish")
	}
	_, err := b.persistPolicy(func(nodepolicy.Policy) error { return nil }, func(nodepolicy.Policy) error { return nil }, func(stored nodepolicy.Policy) (nodepolicy.Policy, error) { return stored, nil })
	if err != nil {
		t.Fatal(err)
	}
	p.mu.Lock()
	pending = p.checkpoint
	p.mu.Unlock()
	if !pending || len(p.configurationWake) != 1 {
		t.Fatal("direct durable policy save did not immediately withdraw and wake configuration")
	}
	requirePerformanceApplicability(t, p, map[string]performancewire.Applicability{"retained": performancewire.ApplicabilityUnknown})
	if _, raw := p.publication(); raw != nil {
		t.Fatal("direct policy save left old current publication available")
	}
}

func TestMeasurementRefreshFailureWithdrawsPriorCompleteness(t *testing.T) {
	b, manager := measurementBrokerFixture(t)
	b.refreshPerformanceConfigurations(context.Background())
	requireMeasurement(t, b, true)
	manager.setEngineAnswer(func(string, json.RawMessage) (any, string) { return nil, "inspection failed" })
	b.refreshPerformanceConfigurations(context.Background())
	requireMeasurement(t, b, false)
}

func TestMeasurementCurrentnessFences(t *testing.T) {
	for _, scenario := range []string{"settings", "policy", "profile", "manager source", "worker", "memory source", "memory expiry", "content mutation"} {
		t.Run(scenario, func(t *testing.T) {
			b, _ := measurementBrokerFixture(t)
			b.refreshPerformanceConfigurations(context.Background())
			requireMeasurement(t, b, true)
			r := b.nodePolicyRuntime()
			switch scenario {
			case "settings":
				b.observeSettingsIdentity([]settings.Snapshot{appliedSnapshot("llamacpp", 4)})
			case "policy":
				r.mu.Lock()
				r.current.Revision = "99"
				r.mu.Unlock()
			case "profile":
				r.mu.Lock()
				r.current.Profiles[0].Model = "replacement.gguf"
				r.mu.Unlock()
			case "manager source":
				b.activateEngineManagerSource()
			case "worker":
				b.setEngineMgr(nil)
			case "content mutation":
				if !b.withdrawModelIdentity(r.engineManagerSource, "llamacpp") {
					t.Fatal("mutation withdrawal lost its source")
				}
			case "memory source":
				b.setScanner(nil)
			case "memory expiry":
				r.mu.Lock()
				c := r.measurementConfigurations["llamacpp"]["exact-profile"]
				c.memory.Hop.BootNS = "0"
				r.measurementConfigurations["llamacpp"]["exact-profile"] = c
				r.mu.Unlock()
			}
			requireMeasurement(t, b, false)
		})
	}
}

func TestMeasurementStaleRefreshCannotInstallCompleteness(t *testing.T) {
	for _, scenario := range []string{"settings", "policy", "manager source", "memory source", "content mutation"} {
		t.Run(scenario, func(t *testing.T) {
			b, manager := measurementBrokerFixture(t)
			manager.mu.Lock()
			answer := manager.answer
			manager.mu.Unlock()
			manager.setEngineAnswer(func(method string, raw json.RawMessage) (any, string) {
				result, failure := answer(method, raw)
				r := b.nodePolicyRuntime()
				switch scenario {
				case "settings":
					b.observeSettingsIdentity([]settings.Snapshot{appliedSnapshot("llamacpp", 4)})
				case "policy":
					r.mu.Lock()
					r.current.Revision = "99"
					r.mu.Unlock()
				case "manager source":
					b.activateEngineManagerSource()
				case "memory source":
					b.setScanner(nil)
				case "content mutation":
					if !b.withdrawModelIdentity(r.engineManagerSource, "llamacpp") {
						t.Error("mutation withdrawal lost its source")
					}
				}
				return result, failure
			})
			b.refreshPerformanceConfigurations(context.Background())
			requireMeasurement(t, b, false)
		})
	}
}

func TestMeasurementSettingsRevisionRefreshReestablishesSameSemanticIdentity(t *testing.T) {
	b, _ := measurementBrokerFixture(t)
	b.refreshPerformanceConfigurations(context.Background())
	before := requireMeasurement(t, b, true)["llamacpp"]["exact-profile"].Identity
	b.observeSettingsIdentity([]settings.Snapshot{appliedSnapshot("llamacpp", 4)})
	requireMeasurement(t, b, false)
	b.refreshPerformanceConfigurations(context.Background())
	if after := requireMeasurement(t, b, true)["llamacpp"]["exact-profile"].Identity; after != before {
		t.Fatalf("semantic identity changed across revision: %+v != %+v", after, before)
	}
}

func TestMeasurementPerformanceSweepWakeWithoutDesktopSubscriber(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	p := f.b.performance
	p.configurationWake = make(chan struct{}, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	called, done := make(chan struct{}), make(chan struct{})
	p.refreshConfiguration = func(context.Context) { close(called); cancel() }
	go func() { defer close(done); p.runSweeps(ctx, make(chan time.Time)) }()
	f.b.wakePerformanceConfiguration()
	select {
	case <-called:
	case <-time.After(5 * time.Second):
		t.Fatal("configuration wake waited for a desktop subscriber or sweep tick")
	}
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("sweep did not stop")
	}
}

func measurementBindScanner(t *testing.T, b *Broker) (*scannerProcess, *modelconfig.TargetMemory) {
	t.Helper()
	memory := b.modelTargetMemory()
	if memory == nil {
		t.Fatal("fixture has no local memory")
	}
	source := b.getScanner()
	b.bindScannerSource(source)
	if err := ingestDiagnosticTelemetry(b, time.Now(), memory.Observation); err != nil {
		t.Fatal(err)
	}
	return source, memory
}

func TestMeasurementScannerLifecycleWithdrawsProxyControl(t *testing.T) {
	for _, scenario := range []string{"bind", "memory retirement", "telemetry removal", "exit"} {
		t.Run(scenario, func(t *testing.T) {
			b, _ := measurementBrokerFixture(t)
			source, _ := measurementBindScanner(t, b)
			_, proxy := attachFakeProxy(t, b)
			b.refreshPerformanceConfigurations(context.Background())
			before := waitIdentity(t, proxy, "complete measurement", func(p nodepolicy.SetModelIdentityParams) bool {
				return p.Measurements["llamacpp"]["exact-profile"].Identity.Complete
			})
			switch scenario {
			case "bind":
				b.bindScannerSource(&scannerProcess{})
			case "memory retirement":
				source.onMemoryRetired(b.nodeID)
			case "telemetry removal":
				source.onTelemetryRemoved(b.nodeID)
			case "exit":
				source.onExit()
			}
			requireMeasurement(t, b, false)
			after := waitIdentity(t, proxy, "scanner withdrawal", func(p nodepolicy.SetModelIdentityParams) bool {
				return len(p.Measurements) == 0
			})
			if !reflect.DeepEqual(before.Engines, after.Engines) || len(after.Engines) == 0 {
				t.Fatalf("measurement retirement changed strict identity: before=%+v after=%+v", before.Engines, after.Engines)
			}
		})
	}
}

func TestMeasurementScannerRetiredCallbacksCannotInvalidateReplacement(t *testing.T) {
	b, _ := measurementBrokerFixture(t)
	old, memory := measurementBindScanner(t, b)
	current := b.getScanner()
	// Keep the hardware pipe but give the replacement its own lifecycle source.
	replacement := &scannerProcess{peer: current.peer}
	b.bindScannerSource(replacement)
	b.setScanner(replacement)
	if err := ingestDiagnosticTelemetry(b, time.Now(), memory.Observation); err != nil {
		t.Fatal(err)
	}
	b.refreshPerformanceConfigurations(context.Background())
	before := requireMeasurement(t, b, true)
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	generation := r.measurementGeneration
	r.mu.Unlock()
	old.onMemoryRetired(b.nodeID)
	old.onTelemetryRemoved(b.nodeID)
	old.onExit()
	replacement.onMemoryRetired("other-node")
	replacement.onTelemetryRemoved("other-node")
	r.mu.Lock()
	afterGeneration := r.measurementGeneration
	r.mu.Unlock()
	if afterGeneration != generation || !reflect.DeepEqual(before, requireMeasurement(t, b, true)) {
		t.Fatal("unrelated or retired scanner callbacks invalidated the replacement")
	}
}

func TestMeasurementScannerLifecycleFencesOutstandingRefresh(t *testing.T) {
	for _, scenario := range []string{"bind", "memory retirement", "telemetry removal", "exit"} {
		t.Run(scenario, func(t *testing.T) {
			b, manager := measurementBrokerFixture(t)
			source, memory := measurementBindScanner(t, b)
			manager.mu.Lock()
			answer := manager.answer
			manager.mu.Unlock()
			manager.setEngineAnswer(func(method string, raw json.RawMessage) (any, string) {
				result, failure := answer(method, raw)
				switch scenario {
				case "bind":
					b.bindScannerSource(source)
				case "memory retirement":
					source.onMemoryRetired(b.nodeID)
				case "telemetry removal":
					source.onTelemetryRemoved(b.nodeID)
				case "exit":
					source.onExit()
					b.bindScannerSource(source)
				}
				// Restore identical current memory before returning the old reply.
				if err := ingestDiagnosticTelemetry(b, time.Now(), memory.Observation); err != nil {
					return nil, err.Error()
				}
				return result, failure
			})
			b.refreshPerformanceConfigurations(context.Background())
			requireMeasurement(t, b, false)
		})
	}
}

func TestMeasurementPublicInspectionFencesWorkerReply(t *testing.T) {
	for _, scenario := range []string{"current", "settings", "settings restored", "policy", "profile", "manager source", "worker", "content mutation", "memory expiry", "memory changed", "scanner retirement restored"} {
		t.Run(scenario, func(t *testing.T) {
			b, manager := measurementBrokerFixture(t)
			scanner, memory := measurementBindScanner(t, b)
			manager.setEngineAnswer(func(method string, raw json.RawMessage) (any, string) {
				var envelope diagnosticInspectEnvelope
				if method != "engine:inspect-model" {
					return nil, "unexpected method: " + method
				}
				if err := modelconfig.DecodeStrict(raw, &envelope); err != nil {
					return nil, err.Error()
				}
				if envelope.Hardware == nil || envelope.Memory == nil || envelope.Request.ProfileKey != "exact-profile" {
					return nil, "public inspection lost its target context"
				}
				r := b.nodePolicyRuntime()
				switch scenario {
				case "settings", "settings restored":
					b.observeSettingsIdentity([]settings.Snapshot{appliedSnapshot("llamacpp", 4)})
					if scenario == "settings restored" {
						b.observeSettingsIdentity([]settings.Snapshot{appliedSnapshot("llamacpp", 3)})
					}
				case "policy":
					r.mu.Lock()
					r.current.Revision = "99"
					r.mu.Unlock()
				case "profile":
					r.mu.Lock()
					r.current.Profiles[0].Model = "replacement.gguf"
					r.mu.Unlock()
				case "manager source":
					b.activateEngineManagerSource()
				case "worker":
					b.setEngineMgr(nil)
				case "content mutation":
					if !b.withdrawModelIdentity(r.engineManagerSource, "llamacpp") {
						return nil, "mutation withdrawal lost its source"
					}
				case "memory expiry":
					b.telemetry.mu.Lock()
					for i := range b.telemetry.nodes[b.nodeID].scanner.value.Memory.Domains {
						b.telemetry.nodes[b.nodeID].scanner.value.Memory.Domains[i].Total.SampledAt = time.Now().Add(-time.Minute)
					}
					b.telemetry.mu.Unlock()
				case "memory changed":
					b.telemetry.mu.Lock()
					b.telemetry.nodes[b.nodeID].scanner.value.Memory.Domains[0].Free.Bytes = "8000"
					b.telemetry.mu.Unlock()
				case "scanner retirement restored":
					scanner.onMemoryRetired(b.nodeID)
					if err := ingestDiagnosticTelemetry(b, time.Now(), memory.Observation); err != nil {
						return nil, err.Error()
					}
				}
				return modelconfig.InspectResult{Engine: "llamacpp", ModelID: "model.gguf", Status: "resolved", Artifacts: diagnosticIdentityFixture().Artifacts, Estimate: modelconfig.Estimate{Status: "unknown"}, MeasurementIdentity: &modelconfig.MeasurementIdentity{Digest: strings.Repeat("a", 64), Complete: true}}, ""
			})
			result, err := b.inspectModelSettings(context.Background(), modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "model.gguf", ProfileKey: "exact-profile"}, "")
			if err != nil {
				t.Fatal(err)
			}
			want := modelconfig.MeasurementIdentity{}
			if scenario == "current" {
				want = modelconfig.MeasurementIdentity{Digest: strings.Repeat("a", 64), Complete: true}
			}
			if result.MeasurementIdentity == nil || *result.MeasurementIdentity != want {
				t.Fatalf("public measurement=%+v want=%+v", result.MeasurementIdentity, want)
			}
			if result.Status != "resolved" || result.NodeID != b.nodeID || !reflect.DeepEqual(result.Artifacts, diagnosticIdentityFixture().Artifacts) {
				t.Fatalf("measurement invalidation lost ordinary inspection: %+v", result)
			}
		})
	}
}
