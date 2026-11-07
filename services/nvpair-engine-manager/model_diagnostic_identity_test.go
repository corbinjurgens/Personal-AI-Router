// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func diagnosticTestHardware() *diagnosticHardware {
	return &diagnosticHardware{CPUModel: "synthetic CPU", TotalRAMBytes: "8589934592", GPUs: []modelconfig.DiagnosticGPU{
		{UUID: autoTestDeviceBinding().DeviceID, Name: "synthetic GPU", TotalMemoryBytes: "8589934592"},
	}}
}

func TestModelDiagnosticObservationHasRequestedIdentityWithoutEffects(t *testing.T) {
	f := newAppliedKeyFixture(t)
	f.ex.nodeProjection.set(nodeProjection{})
	before, err := os.ReadFile(f.preset)
	if err != nil {
		t.Fatalf("read preset before preview: %v", err)
	}
	p := modelconfig.DiagnosticPreviewRequest{Engine: f.request.Engine, ModelID: f.request.ModelID, ProfileKey: f.request.ProfileKey}
	observation, err := f.ex.observeDiagnostic(context.Background(), p, modelBufferTestTargetMemory(), diagnosticTestHardware())
	if err != nil {
		t.Fatalf("observe diagnostic: %v", err)
	}
	if err := observation.Identity.ValidateRequested(); err != nil {
		t.Fatalf("preview identity: %v", err)
	}
	if !observation.Supported || !observation.StartsEngine || observation.EffectsFence == "" || len(observation.DrainEngines) == 0 {
		t.Fatalf("preview effects: %+v", observation)
	}
	if observation.Identity.PerSlotContext != 0 || observation.Identity.Slots != 0 || observation.Identity.BuildInfo != "" || len(observation.Identity.Placement) != 0 {
		t.Fatalf("preview invented realized evidence: %+v", observation.Identity)
	}
	if len(observation.Identity.GPUs) != 1 || observation.Identity.GPUs[0].UUID != autoTestDeviceBinding().DeviceID || observation.Identity.CPUModel != "synthetic CPU" {
		t.Fatalf("preview hardware: %+v", observation.Identity)
	}
	after, err := os.ReadFile(f.preset)
	if err != nil {
		t.Fatalf("read preset after preview: %v", err)
	}
	if string(before) != string(after) || f.st.running || f.st.proc != nil {
		t.Fatal("preview changed launch or process state")
	}
	checkpoint, err := f.ex.desired.pendingDiagnostic()
	if err != nil || checkpoint != nil {
		t.Fatalf("preview checkpoint=%+v error=%v", checkpoint, err)
	}
}

func TestModelDiagnosticObservationRefusesIncompleteHardware(t *testing.T) {
	f := newAppliedKeyFixture(t)
	f.ex.nodeProjection.set(nodeProjection{})
	p := modelconfig.DiagnosticPreviewRequest{Engine: f.request.Engine, ModelID: f.request.ModelID, ProfileKey: f.request.ProfileKey}
	hardware := diagnosticTestHardware()
	hardware.GPUs = nil
	if _, err := f.ex.observeDiagnostic(context.Background(), p, modelBufferTestTargetMemory(), hardware); err == nil {
		t.Fatal("preview accepted missing usable GPU identity")
	}
}

func TestModelDiagnosticObservationRejectsUnsupportedSlotsBeforeEffects(t *testing.T) {
	f := newAppliedKeyFixture(t, func(f inspectionFixture) {
		preset, err := os.ReadFile(f.preset)
		if err != nil {
			t.Fatal(err)
		}
		planned := strings.Replace(string(preset), "parallel = 1", "parallel = 9", 1)
		writeTestFile(t, f.preset, []byte(strings.Replace(planned, "ctx-size = 4096", "ctx-size = 36864", 1)))
	})
	f.ex.nodeProjection.set(nodeProjection{})
	p := modelconfig.DiagnosticPreviewRequest{Engine: f.request.Engine, ModelID: f.request.ModelID, ProfileKey: f.request.ProfileKey}
	if _, err := f.ex.observeDiagnostic(context.Background(), p, modelBufferTestTargetMemory(), diagnosticTestHardware()); err == nil || !strings.Contains(err.Error(), "more than eight slots") {
		t.Fatalf("unsupported slots did not reach the diagnostic refusal: %v", err)
	}
	checkpoint, err := f.ex.desired.pendingDiagnostic()
	if err != nil || checkpoint != nil || f.st.running || f.st.proc != nil {
		t.Fatalf("unsupported slots changed execution: checkpoint=%+v error=%v", checkpoint, err)
	}
}

func TestModelDiagnosticObservationFinitePlanIsNotRealizedAllocation(t *testing.T) {
	f := newAppliedKeyFixture(t, func(f inspectionFixture) {
		preset, err := os.ReadFile(f.preset)
		if err != nil {
			t.Fatal(err)
		}
		planned := strings.Replace(string(preset), "parallel = 1", "parallel = 2", 1)
		writeTestFile(t, f.preset, []byte(strings.Replace(planned, "ctx-size = 4096", "ctx-size = 8192", 1)))
	})
	f.ex.nodeProjection.set(nodeProjection{})
	p := modelconfig.DiagnosticPreviewRequest{Engine: f.request.Engine, ModelID: f.request.ModelID, ProfileKey: f.request.ProfileKey}
	observation, err := f.ex.observeDiagnostic(context.Background(), p, modelBufferTestTargetMemory(), diagnosticTestHardware())
	if err != nil {
		t.Fatal(err)
	}
	if observation.PlannedPerSlotContext != 4096 || observation.PlannedSlots != 2 || observation.Identity.PerSlotContext != 0 || observation.Identity.Slots != 0 || len(observation.Identity.Placement) != 0 {
		t.Fatalf("planned and realized allocation conflated: %+v", observation)
	}
}

func TestDiagnosticReservationStatusRetainsAcceptedTimestamp(t *testing.T) {
	f, p := diagnosticReservationFixture(t)
	reserveDiagnosticFixture(t, f, p)
	checkpoint, err := f.ex.desired.pendingDiagnostic()
	if err != nil || checkpoint == nil {
		t.Fatalf("accepted checkpoint=%+v error=%v", checkpoint, err)
	}
	status, err := f.ex.diagnosticStatusWithCheckpoint(p.RunID)
	if err != nil || status.StartedAt != checkpoint.Run.StartedAt {
		t.Fatalf("reserved timestamp changed: status=%+v checkpoint=%+v error=%v", status, checkpoint, err)
	}
	if err := status.DiagnosticRun.Validate(); err != nil {
		t.Fatalf("reserved public metadata: %v", err)
	}
}

func TestModelDiagnosticObservationHonorsPauseAuthority(t *testing.T) {
	f := newAppliedKeyFixture(t)
	p := modelconfig.DiagnosticPreviewRequest{Engine: f.request.Engine, ModelID: f.request.ModelID, ProfileKey: f.request.ProfileKey}
	f.ex.nodeProjection.set(nodeProjection{ManagementBlocked: true, PausedLocally: true, Policy: nodepolicy.Advertisement{PauseMode: nodepolicy.PauseConfigured}})
	if _, err := f.ex.observeDiagnostic(context.Background(), p, modelBufferTestTargetMemory(), diagnosticTestHardware()); err != nil {
		t.Fatalf("configured pause preview: %v", err)
	}
	f.ex.nodeProjection.set(nodeProjection{ManagementBlocked: true, PausedLocally: true, Policy: nodepolicy.Advertisement{PauseMode: nodepolicy.PauseImmediate}})
	if _, err := f.ex.observeDiagnostic(context.Background(), p, modelBufferTestTargetMemory(), diagnosticTestHardware()); err == nil {
		t.Fatal("Halt allowed diagnostic preview")
	}
}

func TestModelDiagnosticHardwareAndPlacementUseOneUnifiedPool(t *testing.T) {
	memory := modelBufferTestTargetMemory()
	shared := memory.Observation.Domains[0]
	shared.Kind, shared.DeviceIDs = "shared", []string{"ioreg:fixture"}
	memory.Observation.Domains = []noderec.MemoryDomain{shared}
	hardware := &diagnosticHardware{CPUModel: "synthetic Apple CPU", TotalRAMBytes: shared.Total.Bytes, GPUs: []modelconfig.DiagnosticGPU{{UUID: "ioreg:fixture", Name: "synthetic Apple GPU", TotalMemoryBytes: shared.Total.Bytes}}}
	if err := diagnosticHardwareMatchesAt(hardware, memory, true); err != nil {
		t.Fatalf("unified hardware identity: %v", err)
	}
	placement, err := modelBufferPoolPlacement(memory, []modelconfig.BufferReportRow{{Label: "CPU"}, {Label: "Metal"}}, nil)
	if err != nil {
		t.Fatalf("unified placement: %v", err)
	}
	if !reflect.DeepEqual(placement, []modelconfig.DiagnosticPlacement{{DomainID: "host", Kind: "shared"}}) {
		t.Fatalf("unified placement double counted system memory: %+v", placement)
	}
}

func TestModelDiagnosticPlacementMapsTwoDedicatedGPUsByPhysicalID(t *testing.T) {
	memory := modelBufferTestTargetMemory()
	second := "GPU-22222222-2222-2222-2222-222222222222"
	pool := memory.Observation.Domains[1]
	pool.ID, pool.DeviceIDs = "second-gpu", []string{second, "pci:0000:ac:00.0", "luid_0x00000000_0x00005678_phys_0"}
	memory.Observation.Domains = append(memory.Observation.Domains, pool)
	first := autoTestDeviceBinding().DeviceID
	devices := []modelNativeDevice{
		{Selector: "CUDA0", Kind: 1, UUID: second, PCI: "0000:ac:00.0", LUID: "luid_0x00000000_0x00005678_phys_0"},
		{Selector: "CUDA1", Kind: 1, UUID: first, PCI: "0000:ab:00.0", LUID: modelDevicesFixtureLUID},
	}
	rows := []modelconfig.BufferReportRow{{Label: "CPU"}, {Label: "CUDA0"}, {Label: "CUDA1"}}
	placement, err := modelBufferPoolPlacement(memory, rows, devices)
	if err != nil {
		t.Fatalf("map physical placement: %v", err)
	}
	want := []modelconfig.DiagnosticPlacement{{DomainID: "gpu", Kind: "device", DeviceID: first}, {DomainID: "host", Kind: "host"}, {DomainID: "second-gpu", Kind: "device", DeviceID: second}}
	if !reflect.DeepEqual(placement, want) {
		t.Fatalf("placement=%+v want=%+v", placement, want)
	}
	// Neither pool order nor native enumeration order is physical authority.
	memory.Observation.Domains = []noderec.MemoryDomain{pool, memory.Observation.Domains[0], memory.Observation.Domains[1]}
	slices.Reverse(devices)
	again, err := modelBufferPoolPlacement(memory, rows, devices)
	if err != nil || !reflect.DeepEqual(again, want) {
		t.Fatalf("reordered mapping=%+v error=%v", again, err)
	}
}

func TestModelDiagnosticPlacementRefusesNativeOrdinal(t *testing.T) {
	if _, err := modelBufferPoolPlacement(modelBufferTestTargetMemory(), []modelconfig.BufferReportRow{{Label: "CUDA0"}}, []modelNativeDevice{{Selector: "CUDA0", Kind: 1, UUID: "CUDA0"}}); err == nil {
		t.Fatal("native ordinal was accepted as physical identity")
	}
}

func TestModelDiagnosticPlacementCPUDoesNotClaimEnumeratedGPUs(t *testing.T) {
	placement, err := modelBufferPoolPlacement(modelBufferTestTargetMemory(), []modelconfig.BufferReportRow{{Label: "CPU"}, {Label: "CPU_Mapped"}, {Label: "CUDA_Host"}}, nil)
	if err != nil {
		t.Fatalf("map CPU placement: %v", err)
	}
	if !reflect.DeepEqual(placement, []modelconfig.DiagnosticPlacement{{DomainID: "host", Kind: "host"}}) {
		t.Fatalf("CPU placement=%+v", placement)
	}
}

func TestModelDiagnosticPlacementRefusesIncompletePhysicalMapping(t *testing.T) {
	test := func(name string, change func(*modelconfig.TargetMemory, *[]modelNativeDevice, *[]modelconfig.BufferReportRow)) {
		t.Run(name, func(t *testing.T) {
			memory := modelBufferTestTargetMemory()
			devices := []modelNativeDevice{{Selector: "CUDA0", Kind: 1, UUID: autoTestDeviceBinding().DeviceID, PCI: "0000:ab:00.0", LUID: modelDevicesFixtureLUID}}
			rows := []modelconfig.BufferReportRow{{Label: "CUDA0"}}
			change(memory, &devices, &rows)
			if placement, err := modelBufferPoolPlacement(memory, rows, devices); err == nil {
				t.Fatalf("incomplete evidence produced placement: %+v", placement)
			}
		})
	}
	test("missing allocation rows", func(_ *modelconfig.TargetMemory, _ *[]modelNativeDevice, rows *[]modelconfig.BufferReportRow) {
		*rows = nil
	})
	test("missing device identity", func(_ *modelconfig.TargetMemory, devices *[]modelNativeDevice, _ *[]modelconfig.BufferReportRow) {
		*devices = nil
	})
	test("wrong physical UUID", func(_ *modelconfig.TargetMemory, devices *[]modelNativeDevice, _ *[]modelconfig.BufferReportRow) {
		(*devices)[0].UUID = "GPU-22222222-2222-2222-2222-222222222222"
	})
	test("wrong physical PCI", func(_ *modelconfig.TargetMemory, devices *[]modelNativeDevice, _ *[]modelconfig.BufferReportRow) {
		(*devices)[0].PCI = "0000:ac:00.0"
	})
	test("unknown buffer label", func(_ *modelconfig.TargetMemory, _ *[]modelNativeDevice, rows *[]modelconfig.BufferReportRow) {
		(*rows)[0].Label = "CUDA7"
	})
	test("duplicate selector", func(_ *modelconfig.TargetMemory, devices *[]modelNativeDevice, _ *[]modelconfig.BufferReportRow) {
		*devices = append(*devices, (*devices)[0])
	})
	test("duplicate physical pool", func(memory *modelconfig.TargetMemory, _ *[]modelNativeDevice, _ *[]modelconfig.BufferReportRow) {
		pool := memory.Observation.Domains[1]
		pool.ID = "duplicate"
		memory.Observation.Domains = append(memory.Observation.Domains, pool)
	})
	test("ambiguous pool alias", func(memory *modelconfig.TargetMemory, _ *[]modelNativeDevice, _ *[]modelconfig.BufferReportRow) {
		memory.Observation.Domains[1].DeviceIDs = append(memory.Observation.Domains[1].DeviceIDs, "pci:0000:ac:00.0")
	})
}

func TestModelDiagnosticRecordAuthenticatesCompletedProducingRun(t *testing.T) {
	test := func(name string, change func(*diagnosticRecordParams)) {
		t.Run(name, func(t *testing.T) {
			base := t.TempDir()
			if err := os.Chmod(base, 0o700); err != nil {
				t.Fatalf("secure profile store: %v", err)
			}
			profile := measuredStoreFixture()
			profile.BuildPin = profile.NativeBuildID
			identity := profile.DiagnosticIdentity.Clone()
			e := &Executor{modelMemoryProfiles: newModelMemoryProfiles(base), diagnosticCompleted: map[string]diagnosticCompletedRun{
				profile.RunID: {token: "producing-token", Reservation: diagnosticReserveParams{RunID: profile.RunID, Engine: "llamacpp", ModelID: "model", ProfileKey: "profile", ConfigurationKey: profile.ConfigurationKey},
					Status: diagnosticOperationStatus{DiagnosticRun: modelconfig.DiagnosticRun{RunID: profile.RunID, Outcome: "completed"}, WorkStopped: true, Identity: &identity, High: profile.High}},
			}}
			p := diagnosticRecordParams{RunID: profile.RunID, Token: "producing-token", Profile: profile.Clone()}
			change(&p)
			if err := e.recordDiagnostic(p); err == nil {
				t.Fatal("record accepted forged producing evidence")
			}
			if len(e.modelMemoryProfiles.records) != 0 {
				t.Fatalf("rejected record changed store: %+v", e.modelMemoryProfiles.records)
			}
		})
	}
	test("unknown run", func(p *diagnosticRecordParams) { p.RunID = "another-run" })
	test("wrong token", func(p *diagnosticRecordParams) { p.Token = "another-token" })
	test("wrong key", func(p *diagnosticRecordParams) { p.Profile.ConfigurationKey = strings.Repeat("f", 64) })
	test("wrong actual hardware", func(p *diagnosticRecordParams) { p.Profile.CPUModel = "another CPU" })
	test("wrong realized context", func(p *diagnosticRecordParams) { p.Profile.PerSlotContext = 2048 })
	test("forged native timing", func(p *diagnosticRecordParams) { p.Profile.High.WallMS++ })
}

func TestModelDiagnosticRecordAcceptsSamplerCoverageDowngrade(t *testing.T) {
	base := t.TempDir()
	if err := os.Chmod(base, 0o700); err != nil {
		t.Fatalf("secure profile store: %v", err)
	}
	profile := measuredStoreFixture()
	profile.BuildPin = profile.NativeBuildID
	identity := profile.DiagnosticIdentity.Clone()
	e := &Executor{modelMemoryProfiles: newModelMemoryProfiles(base), diagnosticCompleted: map[string]diagnosticCompletedRun{
		profile.RunID: {token: "producing-token", Reservation: diagnosticReserveParams{RunID: profile.RunID, Engine: "llamacpp", ModelID: "model", ProfileKey: "profile", ConfigurationKey: profile.ConfigurationKey},
			Status: diagnosticOperationStatus{DiagnosticRun: modelconfig.DiagnosticRun{RunID: profile.RunID, Outcome: "completed"}, WorkStopped: true, Identity: &identity, High: profile.High}},
	}}
	p := diagnosticRecordParams{RunID: profile.RunID, Token: "producing-token", Profile: profile.Clone()}
	p.Profile.High.Coverage = "partial"
	sampled := false
	p.Profile.High.OverlapWindowSampled = &sampled
	if err := e.recordDiagnostic(p); err != nil {
		t.Fatalf("record sampled partial evidence: %v", err)
	}
	reopened := newModelMemoryProfiles(base)
	if reopened.unavailable || len(reopened.records) != 1 || reopened.records[0].Profile.Measured.RunID != profile.RunID {
		t.Fatalf("reopened producing record: %+v", reopened)
	}
}

func TestModelDiagnosticInspectionSuppliesSameIdentityToMeasuredOverlay(t *testing.T) {
	f, _ := memoryProfilesFixture(t)
	ctx := withDiagnosticHardware(f.ex.modelMemoryContext(context.Background(), f.request.Engine, modelBufferTestTargetMemory(), false), diagnosticTestHardware())
	result, err := f.ex.inspectModelArtifacts(ctx, f.request)
	if err != nil {
		t.Fatalf("inspect current files: %v", err)
	}
	identity, err := f.ex.diagnosticRequestedIdentity(ctx, f.request, result)
	if err != nil {
		t.Fatalf("assemble current requested identity: %v", err)
	}
	identity.BuildInfo, identity.PerSlotContext, identity.Slots = "observed-native-build", 4096, 1
	identity.Placement = []modelconfig.DiagnosticPlacement{{DomainID: "host", Kind: "host"}}
	measured := measuredStoreFixture()
	measured.DiagnosticIdentity = identity
	measured.BuildPin = identity.NativeBuildID
	prompt := int32(3952)
	measured.High.Requests[0].TargetPromptTokens = prompt
	measured.High.Requests[0].PromptTokens = &prompt
	if err := f.ex.modelMemoryProfiles.recordMeasured(f.request.Engine, f.request.ModelID, f.request.ProfileKey, measured); err != nil {
		t.Fatalf("persist authenticated fixture: %v", err)
	}
	f.ex.modelMemoryProfiles = newModelMemoryProfiles(filepath.Dir(f.ex.modelMemoryProfiles.path))
	inspected, err := f.ex.inspectModel(ctx, f.request, false)
	if err != nil {
		t.Fatalf("inspect reopened measured record: %v", err)
	}
	if len(inspected.MemoryProfiles) != 1 || !inspected.MemoryProfiles[0].Current {
		t.Fatalf("current measured record=%+v", inspected.MemoryProfiles)
	}
	if inspected.Estimate.EstimateSource == nil || inspected.Estimate.EstimateSource.Kind != "measured" || inspected.Estimate.EstimateSource.Revision != measured.RunID {
		t.Fatalf("estimate source=%+v", inspected.Estimate.EstimateSource)
	}
	f.ex.emit = nil
	f.ex.markStopped(f.st, f.request.Engine)
	stopped, err := f.ex.inspectModel(ctx, f.request, false)
	if err != nil || len(stopped.MemoryProfiles) != 1 || !stopped.MemoryProfiles[0].Current || stopped.Estimate.EstimateSource == nil || stopped.Estimate.EstimateSource.Kind != "measured" {
		t.Fatalf("confirmed stop withdrew current measured overlay: profiles=%+v source=%+v error=%v", stopped.MemoryProfiles, stopped.Estimate.EstimateSource, err)
	}
	hardware := diagnosticTestHardware()
	hardware.CPUModel = "replacement CPU"
	changed, err := f.ex.inspectModel(withDiagnosticHardware(ctx, hardware), f.request, false)
	if err != nil {
		t.Fatalf("inspect changed hardware: %v", err)
	}
	if changed.MemoryProfiles[0].Current || changed.Estimate.EstimateSource != nil && changed.Estimate.EstimateSource.Kind == "measured" {
		t.Fatal("changed hardware retained measured overlay")
	}
}

type diagnosticInspectionMemoryFunc func(context.Context) (*modelEstimateMemory, error)

func (f diagnosticInspectionMemoryFunc) ModelAutoMemory(ctx context.Context) (*modelEstimateMemory, error) {
	return f(ctx)
}

func TestModelDiagnosticLateInspectionContentionKeepsMeasurement(t *testing.T) {
	f, _ := memoryProfilesFixture(t)
	ctx := withDiagnosticHardware(f.ex.modelMemoryContext(context.Background(), f.request.Engine, modelBufferTestTargetMemory(), false), diagnosticTestHardware())
	artifacts, err := f.ex.inspectModelArtifacts(ctx, f.request)
	if err != nil {
		t.Fatal(err)
	}
	identity, err := f.ex.diagnosticRequestedIdentity(ctx, f.request, artifacts)
	if err != nil {
		t.Fatal(err)
	}
	identity.BuildInfo, identity.PerSlotContext, identity.Slots = "observed-native-build", 4096, 1
	identity.Placement = []modelconfig.DiagnosticPlacement{{DomainID: "host", Kind: "host"}}
	measured := recordAuthenticatedDiagnosticFixture(t, f, identity)
	op := ctx.Value(modelAutoOperationKey{}).(modelAutoOperation)
	memory := op.Memory
	locked := false
	defer func() {
		if locked {
			f.st.opMu.Unlock()
		}
	}()
	op.Memory = diagnosticInspectionMemoryFunc(func(ctx context.Context) (*modelEstimateMemory, error) {
		// Artifact checks already completed. Simulate a lifecycle/status owner
		// acquiring the live lock while compilation reads detached inputs.
		if !locked {
			f.st.opMu.Lock()
			locked = true
		}
		return memory.ModelAutoMemory(ctx)
	})
	busy, err := f.ex.inspectModel(context.WithValue(ctx, modelAutoOperationKey{}, op), f.request, false)
	if !locked {
		t.Fatal("inspection never reached the late memory boundary")
	}
	f.st.opMu.Unlock()
	locked = false
	if err != nil || busy.Status != "busy" || busy.ConfigurationKey != "" || busy.ExecutionKey != "" || busy.Estimate.EstimateSource != nil && busy.Estimate.EstimateSource.Kind == "measured" || slices.ContainsFunc(busy.MemoryProfiles, func(p modelconfig.MemoryProfile) bool { return p.Current }) {
		t.Fatalf("late contention must be explicit busy without current proof: status=%s key=%s reason=%s profiles=%+v error=%v", busy.Status, busy.ConfigurationKey, busy.ConfigurationKeyReason, busy.MemoryProfiles, err)
	}
	current, err := f.ex.inspectModel(ctx, f.request, false)
	if err != nil || len(current.MemoryProfiles) != 1 || !current.MemoryProfiles[0].Current || !reflect.DeepEqual(current.MemoryProfiles[0].Measured, &measured) || current.Estimate.EstimateSource == nil || current.Estimate.EstimateSource.Kind != "measured" {
		t.Fatalf("settled inspection lost its actual persisted measurement: profiles=%+v source=%+v error=%v", current.MemoryProfiles, current.Estimate.EstimateSource, err)
	}
}

func TestModelDiagnosticInspectionRetainsCurrentFailedRealization(t *testing.T) {
	f, _ := memoryProfilesFixture(t)
	ctx := withDiagnosticHardware(f.ex.modelMemoryContext(context.Background(), f.request.Engine, modelBufferTestTargetMemory(), false), diagnosticTestHardware())
	before, err := f.ex.inspectModel(ctx, f.request, false)
	if err != nil {
		t.Fatal(err)
	}
	identity, err := f.ex.diagnosticRequestedIdentity(ctx, f.request, before)
	if err != nil {
		t.Fatal(err)
	}
	identity.BuildInfo, identity.PerSlotContext, identity.Slots = "observed-native-build", 4096, 1
	identity.Placement = []modelconfig.DiagnosticPlacement{{DomainID: "host", Kind: "host"}}
	measured := measuredStoreFixture()
	measured.DiagnosticIdentity, measured.BuildPin = identity, identity.NativeBuildID
	measured.Outcome, measured.FailedPhase, measured.Reason = "failed", "high", "engine-exited"
	if err := f.ex.modelMemoryProfiles.recordMeasured(f.request.Engine, f.request.ModelID, f.request.ProfileKey, measured); err != nil {
		t.Fatal(err)
	}
	f.ex.modelMemoryProfiles = newModelMemoryProfiles(filepath.Dir(f.ex.modelMemoryProfiles.path))
	inspected, err := f.ex.inspectModel(ctx, f.request, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(inspected.MemoryProfiles) != 1 || !inspected.MemoryProfiles[0].Current {
		t.Fatalf("failed producing realization lost currentness: %+v", inspected.MemoryProfiles)
	}
	if !slices.Contains(inspected.Estimate.Unknowns, "measured run failed at high: engine-exited") {
		t.Fatalf("failed run warning absent: %v", inspected.Estimate.Unknowns)
	}
	if !slices.EqualFunc(before.Estimate.Domains, inspected.Estimate.Domains, func(a, b modelconfig.MemoryDomain) bool {
		return a.ID == b.ID && a.Kind == b.Kind && a.RequiredBytes == b.RequiredBytes && a.RequiredLowBytes == b.RequiredLowBytes && a.RequiredHighBytes == b.RequiredHighBytes
	}) || inspected.Estimate.EstimateSource != nil && inspected.Estimate.EstimateSource.Kind == "measured" {
		t.Fatal("failed run changed estimated memory")
	}
}

func recordAuthenticatedDiagnosticFixture(t *testing.T, f inspectionFixture, identity modelconfig.DiagnosticIdentity) modelconfig.MeasuredProfile {
	t.Helper()
	if err := os.Chmod(f.ex.baseDir, 0o700); err != nil {
		t.Fatalf("secure measured store: %v", err)
	}
	f.ex.modelMemoryProfiles = newModelMemoryProfiles(f.ex.baseDir)
	measured := measuredStoreFixture()
	measured.DiagnosticIdentity, measured.BuildPin = identity.Clone(), identity.NativeBuildID
	prompt := identity.PerSlotContext - 144
	measured.High.Requests[0].TargetPromptTokens = prompt
	measured.High.Requests[0].PromptTokens = &prompt
	f.ex.diagnosticCompleted = map[string]diagnosticCompletedRun{measured.RunID: {
		token:       "authenticated-fixture-token",
		Reservation: diagnosticReserveParams{RunID: measured.RunID, Engine: f.request.Engine, ModelID: f.request.ModelID, ProfileKey: f.request.ProfileKey, ConfigurationKey: identity.ConfigurationKey, ExecutionKey: identity.ExecutionKey},
		Status:      diagnosticOperationStatus{DiagnosticRun: modelconfig.DiagnosticRun{RunID: measured.RunID, Outcome: measured.Outcome}, WorkStopped: true, Identity: &identity, High: measured.High},
	}}
	if err := f.ex.recordDiagnostic(diagnosticRecordParams{RunID: measured.RunID, Token: "authenticated-fixture-token", Profile: measured}); err != nil {
		t.Fatalf("record producing realization: %v", err)
	}
	return measured
}

func compileDiagnosticFixtureLoads(t *testing.T, f inspectionFixture) {
	t.Helper()
	rt := f.st.plat.Runtime
	command, err := launchForState(f.st, rt.Port)
	if err != nil {
		t.Fatalf("prepare changed fixture: %v", err)
	}
	prepared, err := compileTypedLlamaInputs(context.Background(), f.st, command, rt, false)
	if err != nil {
		t.Fatalf("compile changed fixture: %v", err)
	}
	f.st.plat.Runtime.ModelLoads = modelconfig.CloneLoads(prepared.ModelLoads)
}

func diagnosticExecutionKeyInspectionFixture(t *testing.T, reason string) (inspectionFixture, context.Context, modelconfig.MeasuredProfile) {
	t.Helper()
	f := newAppliedKeyFixture(t)
	if reason == "native-fit" {
		on := "on"
		f.st.plat.Runtime.ModelLoads[0].Requested.Fit = &on
	} else {
		f.st.plat.Runtime.ModelLoads[0].Requested.ExtraOptions = map[string]string{"device": "CUDA7"}
	}
	compileDiagnosticFixtureLoads(t, f)
	ctx := withDiagnosticHardware(f.ex.modelMemoryContext(context.Background(), f.request.Engine, modelBufferTestTargetMemory(), false), diagnosticTestHardware())
	result, err := f.ex.inspectModel(ctx, f.request, false)
	if err != nil {
		t.Fatalf("inspect unpriced fixture: %v", err)
	}
	identity, err := f.ex.diagnosticRequestedIdentity(ctx, f.request, result)
	if err != nil || identity.ExecutionKey == "" || identity.ConfigurationKeyReason != reason {
		t.Fatalf("unpriced requested identity=%+v error=%v", identity, err)
	}
	identity.BuildInfo, identity.PerSlotContext, identity.Slots = "synthetic-native-build", 4096, 1
	if reason == "native-fit" {
		identity.PerSlotContext = 2048
	}
	identity.Placement = []modelconfig.DiagnosticPlacement{{DomainID: "host", Kind: "host"}}
	measured := recordAuthenticatedDiagnosticFixture(t, f, identity)
	return f, ctx, measured
}

func TestModelDiagnosticExecutionKeyInspectionRetainsSemanticCurrentnessAfterRestart(t *testing.T) {
	for _, reason := range []string{"native-fit", "gpu-selection"} {
		t.Run(reason, func(t *testing.T) {
			f, ctx, measured := diagnosticExecutionKeyInspectionFixture(t, reason)
			assertCurrent := func(stage string) {
				t.Helper()
				result, err := f.ex.inspectModel(ctx, f.request, false)
				if err != nil {
					t.Fatalf("%s inspection: %v", stage, err)
				}
				if len(result.MemoryProfiles) != 1 || !result.MemoryProfiles[0].Current || result.MemoryProfiles[0].Measured.RunID != measured.RunID {
					t.Fatalf("%s lost current unpriced realization: %+v", stage, result.MemoryProfiles)
				}
				profile := result.MemoryProfiles[0]
				if profile.HostBytes != "" || profile.DeviceBytes != "" || profile.SharedBytes != "" || profile.CoversPools || profile.Measured.OverlayEligible(measured.DiagnosticIdentity, nil) || profile.Measured.AutoEligible(measured.DiagnosticIdentity, nil) {
					t.Fatalf("%s unpriced record gained projection: %+v", stage, profile)
				}
				if result.Estimate.EstimateSource != nil && result.Estimate.EstimateSource.Kind == "measured" {
					t.Fatalf("%s unpriced record overlaid estimate", stage)
				}
				if result.LoadObservation != nil && (result.LoadObservation.SampleID != "" || len(result.LoadObservation.Rows) != 0) {
					t.Fatalf("%s unpriced public buffer sample: %+v", stage, result.LoadObservation)
				}
			}
			assertCurrent("unchanged")
			f.ex.modelMemoryProfiles = newModelMemoryProfiles(f.ex.baseDir)
			f.ex.diagnosticCompleted = nil // process-local producing authority is gone
			f.ex.managerIncarnation = "restarted-manager"
			assertCurrent("reopened store and manager")
		})
	}
}

func TestModelDiagnosticGPUSelectionDoesNotExposePriorFiniteBufferSample(t *testing.T) {
	f, ctx, measured := diagnosticExecutionKeyInspectionFixture(t, "gpu-selection")
	store, sample := modelBufferStoreFixture(t)
	store.now = time.Now
	sample.ModelIdentity = modelBufferHash(struct{ Engine, Model string }{modelBufferHash(f.request.Engine), f.request.ProfileKey})
	sample.Observation.RecordedAt = time.Now().UTC().Format(time.RFC3339Nano)
	sample.Observation.BuildID = measured.NativeBuildID
	sample.Observation.ConfigurationKey = strings.Repeat("c", 64)
	if state := store.publish(sample); state != "saved" {
		t.Fatalf("seed prior finite history: %s", state)
	}
	f.ex.modelBuffers.store = store
	if history := f.ex.projectModelBufferObservation(f.st, f.request.Engine, f.request.ProfileKey, nil); history.SampleID != sample.Observation.SampleID {
		t.Fatalf("prior finite sample positive control: %+v", history)
	}
	result, err := f.ex.inspectModel(ctx, f.request, false)
	if err != nil {
		t.Fatalf("inspect GPU-selection with finite history: %v", err)
	}
	if result.LoadObservation != nil {
		t.Fatalf("GPU selection exposed finite buffer history: %+v", result.LoadObservation)
	}
}

func TestModelDiagnosticExecutionKeyInspectionInvalidatesChangedIdentity(t *testing.T) {
	test := func(name string, change func(*testing.T, inspectionFixture, context.Context) context.Context) {
		for _, reason := range []string{"native-fit", "gpu-selection"} {
			t.Run(reason+"/"+name, func(t *testing.T) {
				f, ctx, _ := diagnosticExecutionKeyInspectionFixture(t, reason)
				ctx = change(t, f, ctx)
				result, err := f.ex.inspectModel(ctx, f.request, false)
				if err != nil {
					t.Fatalf("inspect changed identity: %v", err)
				}
				if len(result.MemoryProfiles) != 1 || result.MemoryProfiles[0].Current {
					t.Fatalf("changed identity retained current record: %+v", result.MemoryProfiles)
				}
				if result.Estimate.EstimateSource != nil && result.Estimate.EstimateSource.Kind == "measured" {
					t.Fatal("changed unpriced identity overlaid an estimate")
				}
			})
		}
	}
	test("requested inputs", func(t *testing.T, f inspectionFixture, ctx context.Context) context.Context {
		cache := "q8_0"
		f.st.plat.Runtime.ModelLoads[0].Requested.CacheTypeK = &cache
		compileDiagnosticFixtureLoads(t, f)
		return ctx
	})
	test("artifact", func(t *testing.T, f inspectionFixture, ctx context.Context) context.Context {
		writeInspectionGGUF(t, f.model, "llama", 0.25, 8192)
		return ctx
	})
	test("native build", func(t *testing.T, f inspectionFixture, ctx context.Context) context.Context {
		writeTestFile(t, f.st.binPath, []byte("changed native executable"))
		return ctx
	})
	test("native launch environment", func(t *testing.T, f inspectionFixture, ctx context.Context) context.Context {
		if f.st.plat.Runtime.Env == nil {
			f.st.plat.Runtime.Env = map[string]string{}
		}
		f.st.plat.Runtime.Env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
		compileDiagnosticFixtureLoads(t, f)
		return ctx
	})
	test("CPU", func(_ *testing.T, _ inspectionFixture, ctx context.Context) context.Context {
		hardware := diagnosticTestHardware()
		hardware.CPUModel = "replacement CPU"
		return withDiagnosticHardware(ctx, hardware)
	})
	test("RAM", func(_ *testing.T, f inspectionFixture, ctx context.Context) context.Context {
		memory, hardware := modelBufferTestTargetMemory(), diagnosticTestHardware()
		hardware.TotalRAMBytes = "17179869184"
		memory.Observation.Domains[0].Total.Bytes = hardware.TotalRAMBytes
		return withDiagnosticHardware(f.ex.modelMemoryContext(ctx, f.request.Engine, memory, false), hardware)
	})
	test("GPU UUID", func(_ *testing.T, f inspectionFixture, ctx context.Context) context.Context {
		memory, hardware := modelBufferTestTargetMemory(), diagnosticTestHardware()
		hardware.GPUs[0].UUID = "GPU-22222222-2222-2222-2222-222222222222"
		memory.Observation.Domains[1].DeviceIDs[0] = hardware.GPUs[0].UUID
		return withDiagnosticHardware(f.ex.modelMemoryContext(ctx, f.request.Engine, memory, false), hardware)
	})
	test("GPU name", func(_ *testing.T, _ inspectionFixture, ctx context.Context) context.Context {
		hardware := diagnosticTestHardware()
		hardware.GPUs[0].Name = "replacement GPU"
		return withDiagnosticHardware(ctx, hardware)
	})
	test("GPU capacity", func(_ *testing.T, f inspectionFixture, ctx context.Context) context.Context {
		memory, hardware := modelBufferTestTargetMemory(), diagnosticTestHardware()
		hardware.GPUs[0].TotalMemoryBytes = "17179869184"
		memory.Observation.Domains[1].Total.Bytes = hardware.GPUs[0].TotalMemoryBytes
		return withDiagnosticHardware(f.ex.modelMemoryContext(ctx, f.request.Engine, memory, false), hardware)
	})
	test("placement pool", func(_ *testing.T, f inspectionFixture, ctx context.Context) context.Context {
		memory := modelBufferTestTargetMemory()
		memory.Observation.Domains[0].ID = "replacement-host-pool"
		return f.ex.modelMemoryContext(ctx, f.request.Engine, memory, false)
	})
	test("requested context", func(t *testing.T, f inspectionFixture, ctx context.Context) context.Context {
		f.st.plat.Runtime.ModelLoads[0].Requested.Context = &modelconfig.Context{Mode: "fixed", Tokens: 2048}
		compileDiagnosticFixtureLoads(t, f)
		return ctx
	})
	test("slots", func(t *testing.T, f inspectionFixture, ctx context.Context) context.Context {
		slots := int32(2)
		f.st.plat.Runtime.ModelLoads[0].Requested.Slots = &slots
		compileDiagnosticFixtureLoads(t, f)
		return ctx
	})
}

func TestModelDiagnosticExecutionKeyLiveRealizationSupersedesHistory(t *testing.T) {
	test := func(name string, change func(inspectionFixture, *modelDiagnosticOperationHTTP)) {
		t.Run(name, func(t *testing.T) {
			f, _, _, fixture := modelDiagnosticRunningOperationFixture(t, func(f inspectionFixture) { on := "on"; f.st.plat.Runtime.ModelLoads[0].Requested.Fit = &on })
			fixture.afterLoad()
			ctx := withDiagnosticHardware(f.ex.modelMemoryContext(context.Background(), f.request.Engine, modelBufferTestTargetMemory(), false), diagnosticTestHardware())
			result, err := f.ex.inspectModel(ctx, f.request, false)
			if err != nil {
				t.Fatalf("inspect live fixture: %v", err)
			}
			identity := f.ex.inspectionDiagnosticIdentity(ctx, f.request, result)
			if identity == nil {
				t.Fatal("complete live realization unavailable")
			}
			recordAuthenticatedDiagnosticFixture(t, f, *identity)
			before, err := f.ex.inspectModel(ctx, f.request, false)
			if err != nil || len(before.MemoryProfiles) != 1 || !before.MemoryProfiles[0].Current {
				t.Fatalf("unchanged live realization=%+v error=%v", before.MemoryProfiles, err)
			}
			change(f, fixture)
			after, err := f.ex.inspectModel(ctx, f.request, false)
			if err != nil || len(after.MemoryProfiles) != 1 || after.MemoryProfiles[0].Current {
				t.Fatalf("live change hidden by retained record=%+v error=%v", after.MemoryProfiles, err)
			}
		})
	}
	test("changed realized context", func(_ inspectionFixture, fixture *modelDiagnosticOperationHTTP) { fixture.context = 2048 })
	test("changed realized build claim", func(_ inspectionFixture, fixture *modelDiagnosticOperationHTTP) {
		fixture.build = "changed-native-build-claim"
	})
	test("adopted uncertain process", func(f inspectionFixture, _ *modelDiagnosticOperationHTTP) {
		f.st.mu.Lock()
		f.st.adopted = true
		f.st.mu.Unlock()
	})
	test("stopped flag with unresolved child", func(f inspectionFixture, _ *modelDiagnosticOperationHTTP) {
		f.st.mu.Lock()
		f.st.running = false
		f.st.mu.Unlock()
	})
	test("unresolved process reclamation", func(f inspectionFixture, _ *modelDiagnosticOperationHTTP) {
		f.st.mu.Lock()
		f.st.running, f.st.proc, f.st.reclaimFailed = false, nil, true
		f.st.mu.Unlock()
	})
	test("incomplete live buffers", func(f inspectionFixture, _ *modelDiagnosticOperationHTTP) {
		f.st.bufferCapture.mu.Lock()
		f.st.bufferCapture.children[31001].parser.partial = true
		f.st.bufferCapture.mu.Unlock()
	})
	test("missing live capture", func(f inspectionFixture, _ *modelDiagnosticOperationHTTP) {
		f.st.mu.Lock()
		f.st.bufferCapture = nil
		f.st.mu.Unlock()
	})
	test("load boundary changed during props", func(f inspectionFixture, fixture *modelDiagnosticOperationHTTP) {
		fixture.beforeProps = func() {
			f.st.bufferCapture.mu.Lock()
			f.st.bufferCapture.boundary++
			f.st.bufferCapture.mu.Unlock()
		}
	})
	test("changed live placement", func(f inspectionFixture, _ *modelDiagnosticOperationHTTP) {
		f.st.bufferCapture.mu.Lock()
		child := f.st.bufferCapture.children[31001]
		child.parser.rows[0].Label = "CUDA7"
		child.devices = []modelNativeDevice{{Selector: "CUDA7", Kind: 1, UUID: autoTestDeviceBinding().DeviceID, PCI: "0000:ab:00.0", LUID: modelDevicesFixtureLUID}}
		f.st.bufferCapture.mu.Unlock()
	})
}

func TestDiagnosticReservationDurabilityFailureKeepsAdmissionHeld(t *testing.T) {
	f, p := diagnosticReservationFixture(t)
	f.ex.emit = nil
	previous := engineFileReplacer
	engineFileReplacer = durablefile.Replacer{SyncDir: func(string) error { return errors.New("synthetic directory sync failure") }}
	t.Cleanup(func() { engineFileReplacer = previous })
	if _, err := f.ex.reserveDiagnostic(context.Background(), p); err == nil {
		t.Fatal("uncertain checkpoint accepted reservation")
	}
	if _, err := f.ex.admitMutation("llamacpp"); err == nil {
		t.Fatal("uncertain checkpoint released admission")
	}
	status, err := f.ex.diagnosticStatusWithCheckpoint(p.RunID)
	if err != nil || status.WorkStopped || status.Token == "" {
		t.Fatalf("uncertain checkpoint status=%+v error=%v", status, err)
	}
	engineFileReplacer = previous
	manager := &Manager{exec: f.ex}
	cleaned, err := manager.abortDiagnostic(diagnosticReleaseParams{p.RunID, status.Token})
	if err != nil || !cleaned.WorkStopped {
		t.Fatalf("retry uncertain checkpoint cleanup=%+v error=%v", cleaned, err)
	}
}

func TestModelDiagnosticRecoveryStatusRequiresDurableClearBeforeWorkStopped(t *testing.T) {
	base := t.TempDir()
	checkpoint := diagnosticRecoveryCheckpointFixture()
	checkpoint.State = "cleanup-complete"
	for i := range checkpoint.Engines {
		checkpoint.Engines[i].CleanupComplete = true
	}
	if err := newDesiredStateStore(base).setPendingDiagnostic(checkpoint); err != nil {
		t.Fatalf("seed terminal checkpoint: %v", err)
	}
	e := diagnosticRecoveryExecutor(t, base)
	if _, err := e.restoreDiagnosticCheckpoint(); err != nil {
		t.Fatalf("restore cleanup hold: %v", err)
	}
	status, err := e.diagnosticStatusWithCheckpoint(checkpoint.Run.RunID)
	if err != nil || status.WorkStopped || status.Token == "" {
		t.Fatalf("uncleared checkpoint granted release proof: %+v error=%v", status, err)
	}
	manager := &Manager{exec: e}
	recovered, err := manager.abortDiagnostic(diagnosticReleaseParams{checkpoint.Run.RunID, status.Token})
	if err != nil || !recovered.WorkStopped || recovered.Outcome != "failed" {
		t.Fatalf("recovered abort=%+v error=%v", recovered, err)
	}
	latest, err := e.diagnosticStatusWithCheckpoint("")
	if err != nil || latest.RunID != checkpoint.Run.RunID || !latest.WorkStopped {
		t.Fatalf("reconnected latest status=%+v error=%v", latest, err)
	}
	done, err := e.admitMutation("llamacpp")
	if err != nil {
		t.Fatalf("recovered terminal status left admission held: %v", err)
	}
	done()
}

func TestModelDiagnosticInspectionEnvelopeRetainsStrictUnknownFieldRejection(t *testing.T) {
	if _, _, err := decodeDiagnosticInspectEnvelope([]byte(`{"request":{"engine":"llamacpp","modelId":"m"},"nativeInitAllowed":false,"inventedAuthority":true}`)); err == nil {
		t.Fatal("private hardware extension weakened the worker envelope")
	}
	p, hardware, err := decodeDiagnosticInspectEnvelope([]byte(`{"request":{"engine":"llamacpp","modelId":"m","profileKey":"p"},"nativeInitAllowed":false,"hardware":{"cpuModel":"CPU","totalRamBytes":"8192","gpus":[]}}`))
	if err != nil || hardware == nil || hardware.CPUModel != "CPU" || p.Request.ProfileKey != "p" {
		t.Fatalf("private hardware envelope=%+v hardware=%+v error=%v", p, hardware, err)
	}
}

func TestModelDiagnosticAcceptedPoolIdentityOutlivesBaselineSamples(t *testing.T) {
	f, _, op := modelDiagnosticOperationFixture(t, nil)
	ctx := f.ex.modelMemoryContext(op.ctx, op.params.Engine, op.reservation.TargetMemory, false)
	initial, err := f.ex.diagnosticInspectIdentity(ctx, op)
	if err != nil {
		t.Fatalf("inspect accepted configuration: %v", err)
	}
	age := op.reservation.TargetMemory.Clone()
	for i := range age.Observation.Domains {
		age.Observation.Domains[i].Total.AgeMS = noderec.MemoryFreshMS + 1
		age.Observation.Domains[i].Free.AgeMS = noderec.MemoryFreshMS + 1
	}
	ctx = f.ex.modelMemoryContext(ctx, op.params.Engine, age, false)
	if _, err := f.ex.diagnosticInspectIdentity(ctx, op); err == nil {
		t.Fatal("ordinary expired telemetry supplied configuration pool identity")
	}
	ctx, err = f.ex.diagnosticContext(ctx, op.params.Engine, op.params.RunID, op.params.Token)
	if err != nil {
		t.Fatalf("scope captured configuration identity: %v", err)
	}
	ctx = context.WithValue(ctx, diagnosticConfigurationPoolsKey{}, initial.ConfigurationPools)
	after, err := f.ex.diagnosticInspectIdentity(ctx, op)
	if err != nil || after.ConfigurationKey != initial.ConfigurationKey {
		t.Fatalf("post-baseline key=%q error=%v", after.ConfigurationKey, err)
	}
	observed := age.At(noderec.MemoryBootClock)
	if observed == nil || observed.Domains[0].Free.Valid || observed.Domains[0].Total.Valid {
		t.Fatalf("captured key renewed telemetry: %+v", observed)
	}
}

func TestModelDiagnosticRecoveryDoesNotConsumeFreshReservation(t *testing.T) {
	f, p := diagnosticReservationFixture(t)
	reserved := reserveDiagnosticFixture(t, f, p)
	manager := &Manager{exec: f.ex}
	if err := manager.recoverDiagnosticBeforeRestore(context.Background()); err != nil {
		t.Fatalf("ordinary restore during fresh reservation: %v", err)
	}
	status, err := f.ex.diagnosticStatusWithCheckpoint(p.RunID)
	if err != nil || status.State != "reserved" || status.Token != reserved.Token || len(f.ex.diagnosticCompleted) != 0 {
		t.Fatalf("recovery consumed live consent reservation: %+v error=%v", status, err)
	}
	if err := f.ex.releaseDiagnostic(diagnosticReleaseParams{p.RunID, reserved.Token}); err != nil {
		t.Fatalf("release untouched reservation: %v", err)
	}
}
