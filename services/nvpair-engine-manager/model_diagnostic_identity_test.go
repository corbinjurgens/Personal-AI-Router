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
	"slices"
	"strings"
	"testing"

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
	placement, err := diagnosticPoolPlacement(memory, diagnosticLoadObservation{})
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
	pool.ID, pool.DeviceIDs = "second-gpu", []string{second}
	memory.Observation.Domains = append(memory.Observation.Domains, pool)
	first := autoTestDeviceBinding().DeviceID
	devices, err := json.Marshal([]map[string]string{{"uuid": second}, {"uuid": first}})
	if err != nil {
		t.Fatalf("encode physical device evidence: %v", err)
	}
	observed := diagnosticLoadObservation{Placement: map[string]json.RawMessage{"devices": devices, "n_gpu_layers": json.RawMessage(`12`)}}
	placement, err := diagnosticPoolPlacement(memory, observed)
	if err != nil {
		t.Fatalf("map physical placement: %v", err)
	}
	want := []modelconfig.DiagnosticPlacement{{DomainID: "gpu", Kind: "device", DeviceID: first}, {DomainID: "host", Kind: "host"}, {DomainID: "second-gpu", Kind: "device", DeviceID: second}}
	if !reflect.DeepEqual(placement, want) {
		t.Fatalf("placement=%+v want=%+v", placement, want)
	}
	// Neither pool order nor native enumeration order is physical authority.
	memory.Observation.Domains = []noderec.MemoryDomain{pool, memory.Observation.Domains[0], memory.Observation.Domains[1]}
	again, err := diagnosticPoolPlacement(memory, observed)
	if err != nil || !reflect.DeepEqual(again, want) {
		t.Fatalf("reordered mapping=%+v error=%v", again, err)
	}
}

func TestModelDiagnosticPlacementRefusesNativeOrdinal(t *testing.T) {
	observed := diagnosticLoadObservation{Placement: map[string]json.RawMessage{"devices": json.RawMessage(`[{"id":"CUDA0"}]`), "n_gpu_layers": json.RawMessage(`12`)}}
	if _, err := diagnosticPoolPlacement(modelBufferTestTargetMemory(), observed); err == nil {
		t.Fatal("native ordinal was accepted as physical identity")
	}
}

func TestModelDiagnosticPlacementCPUDoesNotClaimEnumeratedGPUs(t *testing.T) {
	observed := diagnosticLoadObservation{Placement: map[string]json.RawMessage{"devices": json.RawMessage(`[{"id":"CUDA0"}]`), "n_gpu_layers": json.RawMessage(`0`)}}
	placement, err := diagnosticPoolPlacement(modelBufferTestTargetMemory(), observed)
	if err != nil {
		t.Fatalf("map CPU placement: %v", err)
	}
	if !reflect.DeepEqual(placement, []modelconfig.DiagnosticPlacement{{DomainID: "host", Kind: "host"}}) {
		t.Fatalf("CPU placement=%+v", placement)
	}
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

func TestDiagnosticReservationDurabilityFailureKeepsAdmissionHeld(t *testing.T) {
	f, p := diagnosticReservationFixture(t)
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
