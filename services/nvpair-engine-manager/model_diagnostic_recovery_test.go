// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/durablefile"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

func diagnosticRecoveryCheckpointFixture() diagnosticCheckpoint {
	identity := modelconfig.DiagnosticIdentity{
		ConfigurationKey: strings.Repeat("a", 64), NativeBuildID: "sha256:" + strings.Repeat("b", 64),
		BuildInfo: "fixture-build", CPUModel: "fixture-cpu", TotalRAMBytes: "8192",
		GPUs: []modelconfig.DiagnosticGPU{}, Artifacts: []modelconfig.ArtifactSummary{{Role: "model", Identity: "fixture-artifact", Size: "100", Assurance: "declared"}},
		PerSlotContext: 1024, Slots: 1, Placement: []modelconfig.DiagnosticPlacement{{DomainID: "host", Kind: "host"}},
	}
	return diagnosticCheckpoint{
		Run: modelconfig.DiagnosticRun{RunID: "interrupted-run", Engine: "llamacpp", ModelID: "fixture-model", ProfileKey: "fixture-profile",
			ConfigurationKey: identity.ConfigurationKey, Phase: "loading", Outcome: "running", StartedAt: "2026-11-07T00:00:00Z"},
		EffectsFence: "fixture-effects", Identity: &identity, BuildPin: "fixture-pin", State: "pending",
		Engines: []diagnosticCheckpointEngine{{Engine: "llamacpp", ProcessRunID: "old-process", Started: true,
			AffectedModels: []string{"previous-model"}, LoadedModels: []string{"fixture-profile"}}},
	}
}

func diagnosticRecoveryExecutor(t *testing.T, base string) *Executor {
	t.Helper()
	manifest := testEngineManifest(fakeEngineBin)
	manifest.Engine = "llamacpp"
	platform := manifest.Platforms[hostKey()]
	platform.Runtime.Ready, platform.Runtime.Health = nil, nil
	manifest.Platforms[hostKey()] = platform
	ex := desiredTestExecutor(t, manifest, base)
	ex.nodeProjection.set(nodeProjection{PausedLocally: true, ManagementBlocked: true,
		Policy: nodepolicy.Advertisement{PauseMode: nodepolicy.PauseConfigured}})
	return ex
}

func TestModelDiagnosticRecoveryPhaseReopensAndPreservesPause(t *testing.T) {
	test := func(phase string, started, loaded bool) {
		t.Run(phase, func(t *testing.T) {
			base := t.TempDir()
			checkpoint := diagnosticRecoveryCheckpointFixture()
			checkpoint.Run.Phase = phase
			checkpoint.Engines[0].Started = started
			if !loaded {
				checkpoint.Engines[0].LoadedModels = nil
			}
			if err := newDesiredStateStore(base).setPendingDiagnostic(checkpoint); err != nil {
				t.Fatalf("seed interrupted phase: %v", err)
			}
			ex := diagnosticRecoveryExecutor(t, base)
			pauseBefore := ex.nodeProjection.advertisement()
			stops := 0
			if err := ex.recoverDiagnosticCheckpoint(context.Background(), func(ctx context.Context, engine string) error {
				stops++
				return ex.sleepContext(ctx, engine)
			}); err != nil {
				t.Fatalf("recover %s: %v", phase, err)
			}
			wantStops := 0
			if started || loaded {
				wantStops = 1
			}
			if stops != wantStops {
				t.Fatalf("sleep calls = %d, want %d", stops, wantStops)
			}
			if got := ex.nodeProjection.advertisement(); !reflect.DeepEqual(got, pauseBefore) {
				t.Fatalf("recovery changed saved pause: before=%+v after=%+v", pauseBefore, got)
			}
			pending, err := newDesiredStateStore(base).pendingDiagnostic()
			if err != nil || pending != nil {
				t.Fatalf("recovered checkpoint = %+v, error = %v", pending, err)
			}
			reopened := newModelMemoryProfiles(base)
			if reopened.unavailable || len(reopened.records) != 1 {
				t.Fatalf("interrupted record did not reopen: unavailable=%v records=%+v", reopened.unavailable, reopened.records)
			}
			record := reopened.records[0]
			if record.Engine != checkpoint.Run.Engine || record.ProfileKey != checkpoint.Run.ProfileKey || record.Profile.Measured == nil {
				t.Fatalf("interrupted record scope = %+v", record)
			}
			measured := record.Profile.Measured
			if measured.RunID != checkpoint.Run.RunID || measured.Outcome != "failed" || measured.FailedPhase != phase ||
				measured.Reason != "interrupted" || measured.ConfigurationKey != checkpoint.Run.ConfigurationKey || measured.Low != nil || measured.High != nil {
				t.Fatalf("interrupted record = %+v", measured)
			}
			if status := ex.diagnosticRunStatus(checkpoint.Run.RunID); status.State != "cleanup-complete" || status.Outcome != "failed" {
				t.Fatalf("recovered run status = %+v", status)
			}
			done, err := ex.admitMutation("llamacpp")
			if err != nil {
				t.Fatalf("cleanup retained execution hold: %v", err)
			}
			done()
		})
	}
	test("unloading", false, false)
	test("baseline", false, false)
	test("starting", true, false)
	test("loading", true, true)
	test("high", true, true)
}

func TestModelDiagnosticRecoveryRestoresAdmissionBeforeCleanup(t *testing.T) {
	base := t.TempDir()
	checkpoint := diagnosticRecoveryCheckpointFixture()
	if err := newDesiredStateStore(base).setPendingDiagnostic(checkpoint); err != nil {
		t.Fatalf("seed checkpoint: %v", err)
	}
	ex := diagnosticRecoveryExecutor(t, base)
	if _, err := ex.restoreDiagnosticCheckpoint(); err != nil {
		t.Fatalf("restore reservation: %v", err)
	}
	var busy *diagnosticBusyError
	if err := ex.Wake(context.Background(), "llamacpp"); !errors.As(err, &busy) {
		t.Fatalf("activation during recovery = %v, want diagnostic busy", err)
	}
	if err := ex.RestoreInstalled(context.Background()); err != nil {
		t.Fatalf("fenced automatic restore: %v", err)
	}
	st, err := ex.state("llamacpp")
	if err != nil {
		t.Fatalf("read fenced state: %v", err)
	}
	if st.running || st.proc != nil {
		t.Fatal("automatic restore activated before diagnostic cleanup")
	}
	status, known, err := ex.diagnosticCheckpointStatus(checkpoint.Run.RunID)
	if err != nil || !known || status.State != "pending" || status.RunID != checkpoint.Run.RunID {
		t.Fatalf("reconnect status = %+v known=%v error=%v", status, known, err)
	}
	if _, known, err := ex.diagnosticCheckpointStatus("another-run"); err != nil || known {
		t.Fatalf("unrelated run matched checkpoint: known=%v error=%v", known, err)
	}
}

func TestModelDiagnosticRecoveryCleanupFailureRetainsHoldAcrossRestart(t *testing.T) {
	base := t.TempDir()
	checkpoint := diagnosticRecoveryCheckpointFixture()
	if err := newDesiredStateStore(base).setPendingDiagnostic(checkpoint); err != nil {
		t.Fatalf("seed checkpoint: %v", err)
	}
	first := diagnosticRecoveryExecutor(t, base)
	failure := errors.New("owned child termination unverified")
	if err := first.recoverDiagnosticCheckpoint(context.Background(), func(context.Context, string) error { return failure }); !errors.Is(err, failure) {
		t.Fatalf("cleanup error = %v, want termination failure", err)
	}
	pending, err := newDesiredStateStore(base).pendingDiagnostic()
	if err != nil || pending == nil || pending.State != "cleanup-failed" || pending.Engines[0].CleanupComplete {
		t.Fatalf("failed cleanup checkpoint = %+v error=%v", pending, err)
	}
	var busy *diagnosticBusyError
	if err := first.Wake(context.Background(), "llamacpp"); !errors.As(err, &busy) {
		t.Fatalf("activation after cleanup failure = %v", err)
	}
	second := diagnosticRecoveryExecutor(t, base)
	second.nodeProjection.set(nodeProjection{PausedLocally: true, ManagementBlocked: true,
		Policy: nodepolicy.Advertisement{PauseMode: nodepolicy.PauseImmediate}})
	if err := second.recoverDiagnosticCheckpoint(context.Background(), second.sleepContext); err != nil {
		t.Fatalf("retry cleanup under Halt: %v", err)
	}
	if pause := second.nodeProjection.advertisement(); pause == nil || pause.PauseMode != nodepolicy.PauseImmediate {
		t.Fatalf("cleanup changed Halt: %+v", pause)
	}
	if pending, err := second.desired.pendingDiagnostic(); err != nil || pending != nil {
		t.Fatalf("retry retained checkpoint: %+v error=%v", pending, err)
	}
}

func TestModelDiagnosticRecoveryUnverifiedProcessRetainsCheckpoint(t *testing.T) {
	base := t.TempDir()
	ex := diagnosticRecoveryExecutor(t, base)
	checkpoint := diagnosticRecoveryCheckpointFixture()
	if err := ex.desired.setPendingDiagnostic(checkpoint); err != nil {
		t.Fatalf("seed checkpoint: %v", err)
	}
	// A spawn window with no PID is not proof of stopped. The image is outside
	// this managed install, so the existing sleep owner must refuse to signal it.
	if err := ex.desired.setProcessStop("llamacpp", processStopEvidence{Bin: fakeEngineBin, Port: 12345, Address: "127.0.0.1:12345"}); err != nil {
		t.Fatalf("seed unresolved spawn: %v", err)
	}
	if err := ex.recoverDiagnosticCheckpoint(context.Background(), ex.sleepContext); err == nil {
		t.Fatal("unverified process was treated as stopped")
	}
	pending, err := ex.desired.pendingDiagnostic()
	if err != nil || pending == nil || pending.State != "cleanup-failed" {
		t.Fatalf("unverified process lost cleanup obligation: %+v error=%v", pending, err)
	}
}

func TestModelDiagnosticRecoveryStopsOwnedOrphan(t *testing.T) {
	base := t.TempDir()
	install := filepath.Join(base, "llamacpp")
	if err := os.MkdirAll(install, 0o700); err != nil {
		t.Fatalf("create managed fake install: %v", err)
	}
	bin := filepath.Join(install, filepath.Base(fakeEngineBin))
	copyFile(t, fakeEngineBin, bin)
	manifest := testEngineManifest(bin)
	manifest.Engine = "llamacpp"
	first := desiredTestExecutor(t, manifest, base)
	if err := first.Start(context.Background(), "llamacpp"); err != nil {
		t.Fatalf("start owned fake: %v", err)
	}
	// Model a lost manager without starting its relaunch owner after the second
	// manager reclaims the fake. The fixture still joins the original handle.
	first.shuttingDown.Store(true)
	checkpoint := diagnosticRecoveryCheckpointFixture()
	st, err := first.state("llamacpp")
	if err != nil {
		t.Fatalf("read owned fake state: %v", err)
	}
	st.mu.Lock()
	checkpoint.Engines[0].ProcessRunID = st.runID
	port := st.port
	st.mu.Unlock()
	if err := first.desired.setPendingDiagnostic(checkpoint); err != nil {
		t.Fatalf("checkpoint owned fake: %v", err)
	}
	second := desiredTestExecutor(t, manifest, base)
	second.nodeProjection.set(nodeProjection{PausedLocally: true,
		Policy: nodepolicy.Advertisement{PauseMode: nodepolicy.PauseConfigured}})
	if err := second.recoverDiagnosticCheckpoint(context.Background(), second.sleepContext); err != nil {
		t.Fatalf("reclaim interrupted owned fake: %v", err)
	}
	if portServing(port) {
		t.Fatalf("diagnostic-owned fake still serves on port %d", port)
	}
	if pause := second.nodeProjection.advertisement(); pause == nil || pause.PauseMode != nodepolicy.PauseConfigured {
		t.Fatalf("owned cleanup changed pause: %+v", pause)
	}
}

func TestModelDiagnosticRecoveryRecordFailureRetainsCleanupObligation(t *testing.T) {
	base := t.TempDir()
	checkpoint := diagnosticRecoveryCheckpointFixture()
	if err := newDesiredStateStore(base).setPendingDiagnostic(checkpoint); err != nil {
		t.Fatalf("seed checkpoint: %v", err)
	}
	ex := diagnosticRecoveryExecutor(t, base)
	ex.modelMemoryProfiles.unavailable = true
	if err := ex.recoverDiagnosticCheckpoint(context.Background(), ex.sleepContext); err == nil {
		t.Fatal("failed-record publication failure released recovery")
	}
	pending, err := ex.desired.pendingDiagnostic()
	if err != nil || pending == nil || pending.State != "cleanup-failed" || !pending.Engines[0].CleanupComplete {
		t.Fatalf("record failure lost confirmed process cleanup: %+v error=%v", pending, err)
	}
	var busy *diagnosticBusyError
	if err := ex.Wake(context.Background(), "llamacpp"); !errors.As(err, &busy) {
		t.Fatalf("record failure released execution hold: %v", err)
	}
}

func TestModelDiagnosticRecoveryWithoutCompleteIdentityDoesNotInventRecord(t *testing.T) {
	base := t.TempDir()
	checkpoint := diagnosticRecoveryCheckpointFixture()
	checkpoint.Identity = nil
	if err := newDesiredStateStore(base).setPendingDiagnostic(checkpoint); err != nil {
		t.Fatalf("seed incomplete observed identity: %v", err)
	}
	ex := diagnosticRecoveryExecutor(t, base)
	if err := ex.recoverDiagnosticCheckpoint(context.Background(), ex.sleepContext); err != nil {
		t.Fatalf("recover without fabricated identity: %v", err)
	}
	if len(ex.modelMemoryProfiles.records) != 0 {
		t.Fatalf("recovery invented complete measured identity: %+v", ex.modelMemoryProfiles.records)
	}
}

func TestModelDiagnosticRecoveryExecutionKeyPersistsFailedProfile(t *testing.T) {
	base := t.TempDir()
	checkpoint := diagnosticRecoveryCheckpointFixture()
	checkpoint.Run.ExecutionKey, checkpoint.Run.ConfigurationKey = checkpoint.Run.ConfigurationKey, ""
	checkpoint.Identity.ExecutionKey, checkpoint.Identity.ConfigurationKey = checkpoint.Run.ExecutionKey, ""
	checkpoint.Identity.ConfigurationKeyReason = "gpu-selection"
	if err := newDesiredStateStore(base).setPendingDiagnostic(checkpoint); err != nil {
		t.Fatalf("seed unpriced placement checkpoint: %v", err)
	}
	ex := diagnosticRecoveryExecutor(t, base)
	if err := ex.recoverDiagnosticCheckpoint(context.Background(), ex.sleepContext); err != nil {
		t.Fatalf("recover unpriced placement: %v", err)
	}
	store := newModelMemoryProfiles(base)
	if store.unavailable || len(store.records) != 1 {
		t.Fatalf("execution-key failed record unavailable: %+v", store.records)
	}
	profile := store.records[0].Profile
	if profile.ConfigurationKey != "" || profile.ExecutionKey != checkpoint.Run.ExecutionKey || profile.Measured == nil ||
		profile.Measured.Outcome != "failed" || profile.Measured.ConfigurationKeyReason != "gpu-selection" || profile.Measured.BufferSampleID != "" || profile.CoversPools {
		t.Fatalf("unpriced recovery fabricated finite measurement: %+v", profile)
	}
}

func TestModelDiagnosticRecoveryRequiresPhysicalStopProof(t *testing.T) {
	base := t.TempDir()
	ex := diagnosticRecoveryExecutor(t, base)
	if err := ex.desired.setPendingDiagnostic(diagnosticRecoveryCheckpointFixture()); err != nil {
		t.Fatalf("seed checkpoint: %v", err)
	}
	st, err := ex.state("llamacpp")
	if err != nil {
		t.Fatalf("read fake state: %v", err)
	}
	st.mu.Lock()
	st.reclaimFailed = true
	st.reclaimPID = 0
	st.mu.Unlock()
	defer func() {
		st.mu.Lock()
		st.reclaimFailed = false
		st.mu.Unlock()
	}()
	if err := ex.recoverDiagnosticCheckpoint(context.Background(), func(context.Context, string) error { return nil }); err == nil {
		t.Fatal("successful sleep reply replaced physical termination proof")
	}
	pending, err := ex.desired.pendingDiagnostic()
	if err != nil || pending == nil || pending.State != "cleanup-failed" || pending.Engines[0].CleanupComplete {
		t.Fatalf("unverified termination lost checkpoint: %+v error=%v", pending, err)
	}
}

func TestModelDiagnosticRecoveryCheckpointClearFailureRelatchesHold(t *testing.T) {
	base := t.TempDir()
	ex := diagnosticRecoveryExecutor(t, base)
	checkpoint := diagnosticRecoveryCheckpointFixture()
	if err := ex.desired.setPendingDiagnostic(checkpoint); err != nil {
		t.Fatalf("seed checkpoint: %v", err)
	}
	if _, err := ex.restoreDiagnosticCheckpoint(); err != nil {
		t.Fatalf("restore reservation: %v", err)
	}
	token := ex.diagnostic.token
	if err := ex.confirmDiagnosticWorkStopped(checkpoint.Run.RunID, token); err != nil {
		t.Fatalf("confirm joined fake cleanup: %v", err)
	}
	failure := errors.New("cleanup checkpoint confirmation failed")
	useEngineFileReplacer(t, durablefile.Replacer{CreateTemp: func(string, string, os.FileMode) (durablefile.File, error) {
		return nil, failure
	}})
	if err := ex.clearDiagnosticCheckpoint(checkpoint.Run.RunID, token); !errors.Is(err, failure) {
		t.Fatalf("clear error = %v, want persistence failure", err)
	}
	if err := ex.releaseDiagnostic(diagnosticReleaseParams{RunID: checkpoint.Run.RunID, Token: token}); err == nil {
		t.Fatal("checkpoint publication failure allowed reservation release")
	}
}

func TestModelDiagnosticRecoveryClearedCheckpointDoesNothing(t *testing.T) {
	base := t.TempDir()
	store := newDesiredStateStore(base)
	checkpoint := diagnosticRecoveryCheckpointFixture()
	checkpoint.State, checkpoint.Engines[0].CleanupComplete = "cleanup-complete", true
	if err := store.setPendingDiagnostic(checkpoint); err != nil {
		t.Fatalf("seed confirmed checkpoint: %v", err)
	}
	if err := store.clearPendingDiagnostic(checkpoint.Run.RunID); err != nil {
		t.Fatalf("clear checkpoint: %v", err)
	}
	ex := diagnosticRecoveryExecutor(t, base)
	before, err := os.ReadFile(store.path)
	if err != nil {
		t.Fatalf("read cleared state: %v", err)
	}
	if err := ex.recoverDiagnosticCheckpoint(context.Background(), func(context.Context, string) error {
		t.Fatal("cleared checkpoint dispatched cleanup")
		return nil
	}); err != nil {
		t.Fatalf("recover cleared checkpoint: %v", err)
	}
	assertDesiredFileBytes(t, store.path, before)
	if len(ex.modelMemoryProfiles.records) != 0 || ex.diagnostic != nil {
		t.Fatal("cleared checkpoint recreated a diagnostic")
	}
}

func TestModelDiagnosticRecoveryConfirmedCleanupSkipsEffects(t *testing.T) {
	base := t.TempDir()
	checkpoint := diagnosticRecoveryCheckpointFixture()
	checkpoint.State, checkpoint.Engines[0].CleanupComplete = "cleanup-complete", true
	checkpoint.Run.Phase, checkpoint.Run.Outcome = "done", "failed"
	if err := newDesiredStateStore(base).setPendingDiagnostic(checkpoint); err != nil {
		t.Fatalf("seed interrupted checkpoint removal: %v", err)
	}
	ex := diagnosticRecoveryExecutor(t, base)
	if err := ex.recoverDiagnosticCheckpoint(context.Background(), func(context.Context, string) error {
		t.Fatal("confirmed cleanup repeated a process effect")
		return nil
	}); err != nil {
		t.Fatalf("finish confirmed checkpoint removal: %v", err)
	}
	if pending, err := ex.desired.pendingDiagnostic(); err != nil || pending != nil {
		t.Fatalf("confirmed checkpoint retained: %+v error=%v", pending, err)
	}
}

func TestModelDiagnosticRecoveryUnreadableCheckpointCannotReportNone(t *testing.T) {
	base := t.TempDir()
	ex := diagnosticRecoveryExecutor(t, base)
	if err := os.WriteFile(ex.desired.path, []byte(`{"pendingDiagnostic":{"run":`), 0o600); err != nil {
		t.Fatalf("seed corrupt checkpoint: %v", err)
	}
	if _, _, err := ex.diagnosticCheckpointStatus(""); err == nil {
		t.Fatal("unreadable checkpoint reported no cleanup obligation")
	}
	if err := ex.recoverDiagnosticCheckpoint(context.Background(), ex.sleepContext); err == nil {
		t.Fatal("recovery ignored unreadable checkpoint")
	}
	if done, err := ex.admitMutation("llamacpp"); err == nil {
		done()
		t.Fatal("unreadable cleanup evidence allowed activation")
	}
}

func TestModelDiagnosticRecoveryCheckpointUncertainDurabilityRefusesEffects(t *testing.T) {
	base := t.TempDir()
	ex := diagnosticRecoveryExecutor(t, base)
	failure := errors.New("checkpoint directory durability failed")
	useEngineFileReplacer(t, durablefile.Replacer{SyncDir: func(string) error { return failure }})
	if err := ex.desired.setPendingDiagnostic(diagnosticRecoveryCheckpointFixture()); !errors.Is(err, durablefile.ErrReplaced) {
		t.Fatalf("uncertain checkpoint publication = %v, want landed uncertainty", err)
	}
	if pending, err := ex.desired.pendingDiagnostic(); err != nil || pending == nil {
		t.Fatalf("landed checkpoint disappeared: %+v error=%v", pending, err)
	}
	if err := ex.recoverDiagnosticCheckpoint(context.Background(), func(context.Context, string) error {
		t.Fatal("uncertain checkpoint allowed a cleanup effect")
		return nil
	}); err == nil {
		t.Fatal("uncertain checkpoint update reported recovery complete")
	}
}
