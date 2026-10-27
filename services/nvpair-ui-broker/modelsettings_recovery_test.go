// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"

	"nvpair-shared/clustertrusttest"
	"nvpair-shared/durablefile"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

func (h *modelSettingsHarness) seedAccepted(t *testing.T, coupled bool) *engineSettingsRecord {
	t.Helper()
	p := h.request(t, coupled)
	preview, err := h.b.previewEngineSettings(context.Background(), p, "")
	if err != nil || len(preview.Errors) != 0 {
		t.Fatalf("seed preview=%+v, %v", preview, err)
	}
	policy := h.b.localPolicy().Policy
	next, err := h.b.mergeModelPolicy(policy, p.Engine, *p.ModelChange, "")
	if err != nil {
		t.Fatal(err)
	}
	h.b.engineConfigMu.Lock()
	defer h.b.engineConfigMu.Unlock()
	record := h.b.engineSettings[p.Engine].clone()
	record.Previous = record.Snapshot.Settings.Clone()
	record.Explicit = true
	// This fixture begins after the first confirmed post-acceptance progress.
	// A bare accepted stage is now deliberately ambiguous across a restart.
	record.Model = &modelSettingsOperation{Change: p.ModelChange.Clone(), Review: preview.Clone(), NativeBefore: record.Previous.Clone(), Before: modelPolicyFingerprint(policy), After: modelPolicyFingerprint(next), Implications: modelPolicyImplications(policy, p.Engine, p.ModelChange.ModelID), Stage: "drained"}
	record.Snapshot.Settings = preview.Settings.Clone()
	record.Snapshot.Revision++
	record.Snapshot.RequestID, record.Snapshot.Phase, record.Snapshot.CoupledStage = p.RequestID, "applying", "drained"
	record.Receipts[p.RequestID] = settingsReceipt{Hash: "fixture-accepted", Revision: record.Snapshot.Revision, Phase: "applying"}
	h.b.engineSettings[p.Engine] = record
	if err := h.b.saveEngineSettingsLocked(); err != nil {
		t.Fatal(err)
	}
	h.b.reconstructModelSettingsLocked()
	return record.clone()
}

func (h *modelSettingsHarness) setNative(config settings.Config) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.launch.ServerPort, h.launch.LaunchText = config.ServerPort, config.LaunchText
	h.launch.ModelLoads, h.launch.Serving = modelconfig.CloneLoads(config.ModelLoads), config.Serving.Clone()
}

func (h *modelSettingsHarness) restartJournal(t *testing.T) {
	t.Helper()
	h.b.engineConfigMu.Lock()
	h.b.engineSettingsLoaded = false
	h.b.engineSettings = nil
	h.b.engineSettingsError = nil
	h.b.engineConfigMu.Unlock()
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	delete(r.drains, "llamacpp")
	r.settingsIdentity["llamacpp"] = "obsolete-identity"
	r.mu.Unlock()
}

func TestModelSettingsRecoveryReconstructsBeforeProxyReplay(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.seedAccepted(t, true)
	h.restartJournal(t)
	start := len(h.proxy.snapshot())
	if !h.b.replayPolicyToProxy(context.Background(), h.pp) {
		t.Fatal("proxy replay failed")
	}
	found := false
	for _, call := range h.proxy.snapshot()[start:] {
		if call.method == nodepolicy.MethodSetEngineDrain {
			var drain nodepolicy.SetEngineDrainParams
			if err := json.Unmarshal(call.params, &drain); err != nil {
				t.Fatal(err)
			}
			if drain.Engine == "llamacpp" && drain.Drain {
				found = true
			}
		}
	}
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	_, identity := r.settingsIdentity["llamacpp"]
	r.mu.Unlock()
	if !found || identity {
		t.Fatal("replay enabled state without reconstructing drain and withdrawing identity")
	}
	if !h.b.recoverEngineSettings() || h.calls() != 1 || h.record(t).Model.Stage != "completed" {
		t.Fatal("accepted operation did not recover")
	}
}

func TestModelSettingsRecoveryAdoptsExactNativeAfterLostResponse(t *testing.T) {
	h := newModelSettingsHarness(t)
	record := h.seedAccepted(t, true)
	h.setNative(record.Snapshot.Settings)
	h.b.engineConfigMu.Lock()
	live := h.b.engineSettings["llamacpp"]
	live.Model.Attempted, live.Model.Stage = true, "native-applying"
	if err := h.b.saveEngineSettingsLocked(); err != nil {
		t.Fatal(err)
	}
	h.b.engineConfigMu.Unlock()
	h.restartJournal(t)
	if !h.b.recoverEngineSettings() || h.calls() != 0 {
		t.Fatal("recovery repeated an already-landed native effect")
	}
	if h.record(t).Snapshot.Phase != "succeeded" || len(h.b.localPolicy().Policy.Profiles) != 1 {
		t.Fatal("recovery did not publish the accepted policy")
	}
}

func TestModelSettingsRecoveryApplyingNativeBeforeStateConvergesOnce(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.seedAccepted(t, true)
	h.b.engineConfigMu.Lock()
	record := h.b.engineSettings["llamacpp"]
	record.Model.Attempted, record.Model.Stage = true, "native-applying"
	if err := h.b.saveEngineSettingsLocked(); err != nil {
		t.Fatal(err)
	}
	h.b.engineConfigMu.Unlock()
	h.restartJournal(t)
	if !h.b.recoverEngineSettings() || h.calls() != 1 || h.record(t).Snapshot.Phase != "succeeded" {
		t.Fatal("applying before-state did not converge automatically")
	}
}

func TestModelSettingsRecoveryAdoptsAlreadyLandedPolicy(t *testing.T) {
	h := newModelSettingsHarness(t)
	record := h.seedAccepted(t, true)
	h.setNative(record.Snapshot.Settings)
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	next, err := h.b.mergeModelPolicy(r.current, "llamacpp", record.Model.Change, "")
	if err != nil {
		r.mu.Unlock()
		t.Fatal(err)
	}
	next.Revision = "7000"
	r.current = next
	r.mu.Unlock()
	writePolicyFile(t, h.path, next)
	if !h.b.recoverEngineSettings() {
		t.Fatal("landed policy was not adopted")
	}
	if current := h.b.localPolicy().Policy; current.Revision != "7000" || h.calls() != 0 {
		t.Fatalf("recovery replaced landed state: revision=%s, calls=%d", current.Revision, h.calls())
	}
}

func TestModelSettingsRecoveryThirdPolicyContentIsNeverOverwritten(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.seedAccepted(t, true)
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Idle.UnloadAfterMinutes = 19
	r.current.Revision = "7001"
	r.mu.Unlock()
	before := h.b.localPolicy().Policy
	if h.b.recoverEngineSettings() {
		t.Fatal("third policy content was accepted")
	}
	if !reflect.DeepEqual(before, h.b.localPolicy().Policy) || h.calls() != 0 || !modelSettingsNeedsReconciliation(h.record(t)) {
		t.Fatal("conflicted recovery changed state or lost its failure")
	}
	if _, err := h.b.setLocalPolicy(settingsJSON(before), settingsJSON(before.Revision)); policyErrorCode(err) != codePolicyConflict {
		t.Fatalf("conflict lost policy reservation: %v", err)
	}
}

func TestModelSettingsRecoveryThirdNativeOverridesAreNeverOverwritten(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.seedAccepted(t, true)
	h.mu.Lock()
	h.launch.LaunchText = "--newer-unrelated-override"
	h.mu.Unlock()
	if h.b.recoverEngineSettings() || h.calls() != 0 {
		t.Fatal("recovery overwrote newer native overrides")
	}
	if len(h.b.localPolicy().Policy.Profiles) != 0 {
		t.Fatal("native conflict still published policy")
	}
}

func TestModelSettingsRecoveryRevalidatesNativeBuildWithoutResolvingAgain(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.seedAccepted(t, true)
	h.mu.Lock()
	h.build = "replacement-build"
	h.mu.Unlock()
	before := len(h.engine.snapshot())
	if h.b.recoverEngineSettings() || h.calls() != 0 {
		t.Fatal("changed native build was accepted")
	}
	for _, call := range h.engine.snapshot()[before:] {
		if call.method == "engine:preview-launch" {
			var p settings.Request
			if err := json.Unmarshal(call.params, &p); err != nil {
				t.Fatal(err)
			}
			if p.ModelChange != nil {
				t.Fatal("recovery re-resolved portable/Auto settings")
			}
		}
	}
}

func TestModelSettingsRecoveryFailedRecordsRequireLocalFreshSupersession(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	h.mu.Lock()
	h.configureFailure = "no effect"
	h.mu.Unlock()
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || receipt.Phase != "failed" {
		t.Fatalf("seed failure=%+v, %v", receipt, err)
	}
	if h.b.recoverEngineSettings() || h.calls() != 1 {
		t.Fatal("failed record replayed automatically")
	}
	meshPeer := "peer"
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", h.b.nodeID, meshPeer)
	grantRemoteManagement(t, h.b.clusterDir, "cluster", h.b.nodeID, meshPeer, true)
	fresh := h.request(t, true)
	fresh.ModelChange.SupersedesRequestID = p.RequestID
	if _, err := h.b.previewEngineSettings(context.Background(), fresh, meshPeer); engineSettingsErrorCode(err) != codePolicyConflict {
		t.Fatalf("peer supersession=%v", err)
	}
	// Explicitly retain unrelated newer policy content while accepting repair.
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Idle.UnloadAfterMinutes = 23
	r.current.Revision = "9000"
	r.mu.Unlock()
	fresh.ModelChange.ExpectedPolicyRevision = "9000"
	fresh = h.review(t, fresh, "")
	h.mu.Lock()
	h.configureFailure = ""
	h.mu.Unlock()
	receipt, err := h.b.applyEngineSettings(context.Background(), fresh, "")
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("supersession=%+v, %v", receipt, err)
	}
	if h.b.localPolicy().Policy.Idle.UnloadAfterMinutes != 23 || h.record(t).Receipts[p.RequestID].Phase != "superseded" {
		t.Fatal("supersession replaced unrelated policy or failed to retire prior receipt")
	}
}

func TestModelSettingsRecoveryFailedAttemptRetainsExistingSwitchExclusion(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	h.mu.Lock()
	h.configureFailure = "no effect"
	h.mu.Unlock()
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || receipt.Phase != "failed" {
		t.Fatalf("seed failure = %+v, %v", receipt, err)
	}
	pending := h.record(t)
	if pending.Snapshot.Phase != "applying" || !modelSettingsNeedsReconciliation(pending) {
		t.Fatal("failed attempt discarded its pending operation")
	}
	switchFixture := newSwitchHarness(t)
	switchFixture.b.engineConfigMu.Lock()
	if err := switchFixture.b.loadEngineSettingsLocked(); err != nil {
		t.Fatal(err)
	}
	if switchFixture.b.engineSettings == nil {
		switchFixture.b.engineSettings = map[string]*engineSettingsRecord{}
	}
	switchFixture.b.engineSettings["llamacpp"] = pending
	switchFixture.b.engineConfigMu.Unlock()
	request := switchFixture.request(t, "blocked-by-model-settings")
	_, err := switchFixture.b.acceptEngineSwitch(context.Background(), request, "")
	if err == nil || switchErrorDetail(err, "").Kind != settings.SwitchErrorBusy || len(switchSteps(t, switchFixture.engine)) != 0 {
		t.Fatalf("switch bypassed pending model operation: %v", err)
	}
}

func TestModelSettingsRecoveryTerminalSaveFailureRetainsDrain(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	injected := errors.New("terminal receipt save failed")
	h.b.engineSettingsReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
		data, err := os.ReadFile(tmp)
		if err != nil {
			return err
		}
		var records map[string]*engineSettingsRecord
		if err := json.Unmarshal(data, &records); err != nil {
			return err
		}
		if records["llamacpp"].Model.Stage == "releasing" {
			return injected
		}
		return os.Rename(tmp, target)
	}}
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "failed" || h.calls() != 1 {
		t.Fatalf("terminal save=%+v, %v", receipt, err)
	}
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	draining := r.drains["llamacpp"]
	r.mu.Unlock()
	if !draining || h.record(t).Model.Stage == "completed" {
		t.Fatal("terminal save failure released withdrawal")
	}
	for _, call := range h.proxy.snapshot() {
		if call.method == nodepolicy.MethodSetEngineDrain {
			var drain nodepolicy.SetEngineDrainParams
			if err := json.Unmarshal(call.params, &drain); err != nil {
				t.Fatal(err)
			}
			if !drain.Drain {
				t.Fatal("drain released before terminal receipt was durable")
			}
		}
	}
}

func TestModelSettingsRecoveryPolicyReplacementUncertaintyIsAdoptedAndFenced(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	setReplacer(h.b, durablefile.Replacer{SyncDir: func(string) error { return errors.New("injected policy durability failure") }})
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "failed" {
		t.Fatalf("uncertain policy=%+v, %v", receipt, err)
	}
	policy := h.b.localPolicy().Policy
	if len(policy.Profiles) != 1 || policy.Revision == p.ModelChange.ExpectedPolicyRevision {
		t.Fatal("landed uncertain policy was not adopted")
	}
	data, err := os.ReadFile(h.path)
	if err != nil {
		t.Fatal(err)
	}
	var stored nodepolicy.Policy
	if err := json.Unmarshal(data, &stored); err != nil {
		t.Fatal(err)
	}
	recovered := h.b.recoverEngineSettings()
	// The durable encoder indents opaque requestOptions; compare complete JSON
	// structure, including revision and availability, rather than raw whitespace.
	policyEqual := modelSettingsHash(policy) == modelSettingsHash(stored)
	if !policyEqual || recovered || h.calls() != 1 {
		t.Fatalf("uncertain write was overwritten or replayed without local reconciliation: policyEqual=%v recovery=%v calls=%d", policyEqual, recovered, h.calls())
	}
}

func TestModelSettingsRecoveryPolicyFailureBeforeReplacementKeepsNativeFacts(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	before := h.b.localPolicy().Policy
	setReplacer(h.b, durablefile.Replacer{Replace: func(string, string) error { return errors.New("policy replacement refused") }})
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "failed" {
		t.Fatalf("policy failure = %+v, %v", receipt, err)
	}
	record := h.record(t)
	if !reflect.DeepEqual(before, h.b.localPolicy().Policy) || record.Model.Stage != "native-persisted" || record.Snapshot.EffectiveSettings == nil || !record.Snapshot.EffectiveSettings.Equal(record.Snapshot.Settings) {
		t.Fatal("policy failure lost exact native persistence or changed policy")
	}
}

func TestModelSettingsRecoveryOldWorkerCannotPublishCompletion(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	replacementPeer, _ := newFakeWorkerPipe(t)
	replacement := &rpcWorker{peer: replacementPeer}
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, err := h.answer(method, raw)
		if method == "engine:configure-launch" {
			h.b.setEngineMgr(replacement)
			source := h.b.activateEngineManagerSource()
			h.b.bindEngineMgr(replacement, source)
		}
		return result, err
	})
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "failed" || h.record(t).Snapshot.EffectiveSettings != nil || len(h.b.localPolicy().Policy.Profiles) != 0 {
		t.Fatalf("old worker completion = %+v, %v", receipt, err)
	}
}

func TestModelSettingsRecoveryLoadOnlyChangedProfileStaysWithdrawn(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, false), "")
	reached := make(chan struct{})
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, err := h.answer(method, raw)
		if method == "engine:configure-launch" {
			close(reached)
			return result, holdSuccessReply
		}
		return result, err
	})
	done := make(chan error, 1)
	go func() { _, err := h.b.applyEngineSettings(context.Background(), p, ""); done <- err }()
	waitModelSettingsSignal(t, reached)
	policy := h.b.localPolicy().Policy
	policy.Profiles = []nodepolicy.Profile{{Name: "new-profile", Engine: "llamacpp", Model: "fixture/model", RequestOptions: json.RawMessage(`{"temperature":0.9}`)}}
	if _, err := h.b.setLocalPolicy(settingsJSON(policy), settingsJSON(policy.Revision)); err != nil {
		t.Fatal(err)
	}
	h.engine.releaseHeld()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if record := h.record(t); !modelSettingsNeedsReconciliation(record) || record.Snapshot.Phase != "applying" || record.Snapshot.UnresolvedError == "" {
		t.Fatalf("changed implication=%+v", record.Snapshot)
	}
	if len(h.b.localPolicy().Policy.Profiles) != 1 {
		t.Fatal("load-only apply overwrote newer profile")
	}
}

func TestModelSettingsRecoveryPauseDuringNativeValidationAlwaysWins(t *testing.T) {
	h := newModelSettingsHarness(t)
	record := h.seedAccepted(t, true)
	reached := make(chan struct{})
	var once sync.Once
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, err := h.answer(method, raw)
		if method == "engine:preview-launch" {
			once.Do(func() { close(reached) })
			return result, holdSuccessReply
		}
		return result, err
	})
	done := make(chan bool, 1)
	go func() { done <- h.b.recoverEngineSettings() }()
	waitModelSettingsSignal(t, reached)
	// Drive the actual availability owner. Policy publication can proceed
	// while the coupled record reserves all ordinary policy writers.
	pauseDone := make(chan error, 1)
	pausePersisted := make(chan struct{})
	var pauseOnce sync.Once
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, err := h.acks.answer(method, raw)
		if method == nodepolicy.MethodSetPolicy {
			var p nodepolicy.SetPolicyParams
			if decodeErr := json.Unmarshal(raw, &p); decodeErr != nil {
				return nil, decodeErr.Error()
			}
			if p.Policy.Availability == nodepolicy.Paused {
				pauseOnce.Do(func() { close(pausePersisted) })
			}
		}
		return result, err
	})
	go func() {
		_, err := h.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally)
		pauseDone <- err
	}()
	waitModelSettingsSignal(t, pausePersisted)
	h.engine.releaseHeld()
	if !<-done {
		t.Fatal("paused save did not finish")
	}
	if err := <-pauseDone; err != nil {
		t.Fatal(err)
	}
	policy := h.b.localPolicy().Policy
	if policy.Availability != nodepolicy.Paused || policy.PausedBy != nodepolicy.PausedLocally || modelPolicyFingerprint(policy) != record.Model.After || h.record(t).Snapshot.Running {
		t.Fatal("settings recovery undid Pause or claimed a running engine")
	}
}

func TestModelSettingsRecoveryExactNativeConvergesSavedResume(t *testing.T) {
	h := newModelSettingsHarness(t)
	record := h.seedAccepted(t, true)
	h.setNative(record.Snapshot.Settings)
	h.b.engineConfigMu.Lock()
	h.b.engineSettings["llamacpp"].Resume = true
	if err := h.b.saveEngineSettingsLocked(); err != nil {
		t.Fatal(err)
	}
	h.b.engineConfigMu.Unlock()
	h.restartJournal(t)
	source := h.b.activateEngineManagerSource()
	worker := h.b.getEngineMgr()
	h.b.bindEngineMgr(worker, source)
	// The production startup entry must reach restore instead of failing a
	// stopped exact-native readback before ordinary restoration can run.
	h.b.restoreEnabledEngines(worker)
	got := h.record(t)
	if got.Model.Stage != "completed" || !got.Snapshot.Running || got.Resume || h.calls() != 1 {
		t.Fatalf("saved resume=%+v, calls=%d", got.Snapshot, h.calls())
	}
	if err := modelSettingsCall(t.Context(), worker, "engine:get-launch", map[string]string{"engine": "llamacpp"}, nil); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, call := range h.engine.snapshot() {
		if call.method == restoreEnabledEnginesMethod {
			found = true
		}
	}
	if !found {
		t.Fatal("successful resume still blocked enabled-intent restoration")
	}
}

func TestModelSettingsRecoverySavedResumeHonorsCurrentVeto(t *testing.T) {
	test := func(name string, locked, off, paused bool, wantCalls int) {
		t.Run(name, func(t *testing.T) {
			h := newModelSettingsHarness(t)
			record := h.seedAccepted(t, true)
			h.setNative(record.Snapshot.Settings)
			h.b.engineConfigMu.Lock()
			h.b.engineSettings["llamacpp"].Resume = true
			if err := h.b.saveEngineSettingsLocked(); err != nil {
				t.Fatal(err)
			}
			h.b.engineConfigMu.Unlock()
			h.mu.Lock()
			h.locked, h.savedOff = locked, off
			h.mu.Unlock()
			if paused {
				r := h.b.nodePolicyRuntime()
				r.mu.Lock()
				r.current.Availability, r.current.PausedBy, r.live = nodepolicy.Paused, nodepolicy.PausedLocally, nodepolicy.Paused
				r.mu.Unlock()
			}
			h.restartJournal(t)
			if !h.b.recoverEngineSettings() {
				t.Fatal("vetoed resume prevented saving exact settings")
			}
			got := h.record(t)
			if got.Snapshot.Running || got.Model.Stage != "completed" || got.Resume || h.calls() != wantCalls {
				t.Fatalf("vetoed resume=%+v, calls=%d", got.Snapshot, h.calls())
			}
		})
	}
	test("Pause", false, false, true, 0)
	test("engine lock", true, false, false, 1)
	test("saved local Off", false, true, false, 1)
}

func TestModelSettingsRecoveryEveryProgressedStageReplays(t *testing.T) {
	for _, stage := range []string{"drained", "native-applying", "native-persisted", "policy-persisted", "activation-pending", "releasing"} {
		t.Run(stage, func(t *testing.T) {
			h := newModelSettingsHarness(t)
			record := h.seedAccepted(t, true)
			if stage != "drained" && stage != "native-applying" {
				h.setNative(record.Snapshot.Settings)
			}
			h.b.engineConfigMu.Lock()
			live := h.b.engineSettings["llamacpp"]
			if stage == "policy-persisted" || stage == "activation-pending" || stage == "releasing" {
				if err := h.b.commitModelPolicyLocked(live); err != nil {
					h.b.engineConfigMu.Unlock()
					t.Fatal(err)
				}
			}
			if err := h.b.saveModelStageLocked(live, stage); err != nil {
				h.b.engineConfigMu.Unlock()
				t.Fatal(err)
			}
			h.b.engineConfigMu.Unlock()
			h.restartJournal(t)
			if !h.b.recoverEngineSettings() {
				t.Fatalf("stage %s did not recover automatically", stage)
			}
			got := h.record(t)
			if got.Model.Stage != "completed" || got.Receipts[got.Snapshot.RequestID].Phase != "succeeded" || len(h.b.localPolicy().Policy.Profiles) != 1 {
				t.Fatalf("stage %s outcome = %+v", stage, got.Snapshot)
			}
		})
	}
}

func TestModelSettingsRecoverySupersessionReviewsSavedResume(t *testing.T) {
	h := newModelSettingsHarness(t)
	record := h.seedAccepted(t, true)
	h.setNative(record.Snapshot.Settings)
	h.b.engineConfigMu.Lock()
	live := h.b.engineSettings["llamacpp"]
	live.Resume = true
	if err := h.b.saveModelStageLocked(live, "activation-pending"); err != nil {
		h.b.engineConfigMu.Unlock()
		t.Fatal(err)
	}
	h.b.engineConfigMu.Unlock()
	fresh := h.request(t, true)
	fresh.ModelChange.SupersedesRequestID = record.Snapshot.RequestID
	preview, err := h.b.previewEngineSettings(t.Context(), fresh, "")
	if err != nil || !preview.Restart || preview.Impact == nil || preview.Impact.StaysOff == nil || *preview.Impact.StaysOff {
		t.Fatalf("saved resume was not reviewed: %+v, %v", preview, err)
	}
	fresh.ModelChange.PreviewDigest = preview.PreviewDigest
	receipt, err := h.b.applyEngineSettings(t.Context(), fresh, "")
	if err != nil || receipt.Phase != "succeeded" || !h.record(t).Snapshot.Running || h.calls() != 1 {
		t.Fatalf("supersession lost resume: %+v, %v", receipt, err)
	}
}

func TestModelSettingsRecoveryAcceptedCrashRequiresPreviewWithoutEffects(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	policyBefore := modelSettingsHash(h.b.localPolicy().Policy)
	h.mu.Lock()
	nativeBefore := settings.ConfigFromLaunch(h.launch, p.Settings.ProxyPort)
	h.mu.Unlock()
	crash := errors.New("simulated process exit after acceptance sync")
	h.b.engineSettingsReplacer = durablefile.Replacer{SyncDir: func(string) error {
		// The replacement is already on disk. Terminate control flow here,
		// before the acceptance owner can issue any component command.
		panic(crash)
	}}
	func() {
		defer func() {
			if got := recover(); got != crash {
				t.Fatalf("acceptance crash = %v", got)
			}
		}()
		_, _ = h.b.applyEngineSettings(context.Background(), p, "")
	}()
	// A process exit discards mutexes. The harness reuses its pipe fixtures;
	// release only the policy lock held at this exact injected exit boundary.
	h.b.nodePolicy.mu.Unlock()
	h.b.engineSettingsReplacer = durablefile.Replacer{}
	if h.calls() != 0 || modelSettingsHash(h.b.localPolicy().Policy) != policyBefore {
		t.Fatal("native or policy effect preceded accepted journal stage")
	}
	h.mu.Lock()
	nativeUnchanged := nativeBefore.Equal(settings.ConfigFromLaunch(h.launch, p.Settings.ProxyPort)) && !h.launch.Running
	h.mu.Unlock()
	if !nativeUnchanged {
		t.Fatal("acceptance stopped, wrote or launched the native engine")
	}
	h.restartJournal(t)
	if h.b.recoverEngineSettings() || h.calls() != 0 || modelSettingsHash(h.b.localPolicy().Policy) != policyBefore {
		t.Fatal("bare accepted restart performed component effects")
	}
	got := h.record(t)
	if got.Receipts[p.RequestID].Phase != "failed" || got.Snapshot.AppliedRevision == got.Snapshot.Revision || !strings.Contains(got.Snapshot.Reason, "Interrupted before changes; preview again") {
		t.Fatalf("interrupted readback = %+v", got.Snapshot)
	}
	fresh := h.request(t, true)
	fresh.ModelChange.SupersedesRequestID = p.RequestID
	fresh = h.review(t, fresh, "")
	receipt, err := h.b.applyEngineSettings(context.Background(), fresh, "")
	if err != nil || receipt.Phase != "succeeded" || h.calls() != 1 || h.record(t).Model.Stage != "completed" {
		t.Fatalf("interrupted acceptance blocked fresh review/apply: %+v, %v", receipt, err)
	}
}

func TestModelSettingsRecoveryAcceptanceUncertaintySurvivesRestartAndDoubleFailure(t *testing.T) {
	for _, second := range []string{"durable uncertainty", "failure before uncertainty replacement", "uncertain uncertainty replacement"} {
		t.Run(second, func(t *testing.T) {
			h := newModelSettingsHarness(t)
			p := h.review(t, h.request(t, true), "")
			injected := errors.New("injected acceptance durability failure")
			replacements := 0
			h.b.engineSettingsReplacer = durablefile.Replacer{
				Replace: func(tmp, target string) error {
					replacements++
					if replacements == 2 && second == "failure before uncertainty replacement" {
						return injected
					}
					return os.Rename(tmp, target)
				},
				SyncDir: func(string) error {
					if replacements == 1 || replacements == 2 && second == "uncertain uncertainty replacement" {
						return injected
					}
					return nil
				},
			}
			_, err := h.b.applyEngineSettings(context.Background(), p, "")
			if !errors.Is(err, durablefile.ErrReplaced) || replacements != 2 || h.calls() != 0 {
				t.Fatalf("acceptance=%v, replacements=%d, calls=%d", err, replacements, h.calls())
			}
			h.b.engineSettingsReplacer = durablefile.Replacer{}
			h.restartJournal(t)
			if h.b.recoverEngineSettings() || h.calls() != 0 {
				t.Fatal("restart converted uncertain acceptance into native effects")
			}
			got := h.record(t)
			if got.Model.Stage != "acceptance-uncertain" || !modelSettingsNeedsReconciliation(got) || !strings.Contains(got.Snapshot.UnresolvedError, "Interrupted before changes; preview again") || got.Snapshot.AppliedRevision == got.Snapshot.Revision || got.Receipts[p.RequestID].Phase != "failed" || len(h.b.localPolicy().Policy.Profiles) != 0 {
				t.Fatal("uncertain acceptance lost its durable/reconstructed reconciliation state")
			}
			if _, err := h.b.setLocalPolicy(settingsJSON(h.b.localPolicy().Policy), settingsJSON(h.b.localPolicy().Policy.Revision)); policyErrorCode(err) != codePolicyConflict {
				t.Fatalf("restart lost reservation: %v", err)
			}
			fresh := h.request(t, true)
			fresh.ModelChange.SupersedesRequestID = p.RequestID
			fresh = h.review(t, fresh, "")
			repaired, err := h.b.applyEngineSettings(context.Background(), fresh, "")
			if err != nil || repaired.Phase != "succeeded" || h.calls() != 1 || h.record(t).Receipts[p.RequestID].Phase != "superseded" {
				t.Fatalf("fresh uncertainty repair = %+v, %v", repaired, err)
			}
		})
	}
}

func TestModelSettingsRecoveryAcceptanceDoesNotOverwriteThirdJournal(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	path, err := h.b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	third := []byte(`{"ollama":{"snapshot":{"phase":"idle"},"receipts":{}}}`)
	replacements := 0
	h.b.engineSettingsReplacer = durablefile.Replacer{
		Replace: func(tmp, target string) error { replacements++; return os.Rename(tmp, target) },
		SyncDir: func(string) error {
			if err := os.WriteFile(path, third, 0o600); err != nil {
				return err
			}
			return errors.New("injected landed acceptance failure")
		},
	}
	_, err = h.b.applyEngineSettings(context.Background(), p, "")
	if !errors.Is(err, durablefile.ErrReplaced) || replacements != 1 || h.calls() != 0 {
		t.Fatalf("changed journal=%v, replacements=%d", err, replacements)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != string(third) || h.b.engineSettingsError == nil {
		t.Fatal("acceptance uncertainty overwrote third journal or hid the conflict")
	}
	if h.b.recoverEngineSettings() {
		t.Fatal("known uncertain reservation was discarded after third journal content")
	}
}

func TestModelSettingsRecoveryUnknownInvalidJournalPreservesOrdinaryStartup(t *testing.T) {
	h := newModelSettingsHarness(t)
	path, err := h.b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("{invalid"), 0o600); err != nil {
		t.Fatal(err)
	}
	h.restartJournal(t)
	if !h.b.replayPolicyToProxy(t.Context(), h.pp) {
		t.Fatal("ordinary invalid journal prevented policy replay")
	}
	worker := h.b.getEngineMgr()
	h.b.restoreEnabledEngines(worker)
	if err := modelSettingsCall(t.Context(), worker, "engine:get-launch", map[string]string{"engine": "llamacpp"}, nil); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, call := range h.engine.snapshot() {
		if call.method == restoreEnabledEnginesMethod {
			found = true
		}
	}
	if !found || h.b.engineSettingsError == nil {
		t.Fatal("ordinary invalid journal suppressed restoration or lost its reported error")
	}
}

func TestModelSettingsRecoveryInvalidJournalRetainsKnownModelFence(t *testing.T) {
	test := func(name string, recognizable bool) {
		t.Run(name, func(t *testing.T) {
			h := newModelSettingsHarness(t)
			record := h.seedAccepted(t, true)
			path, err := h.b.engineSettingsPath()
			if err != nil {
				t.Fatal(err)
			}
			if recognizable {
				record.Model.Stage = "invalid-model-stage"
				data, err := json.Marshal(map[string]*engineSettingsRecord{"llamacpp": record})
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, data, 0o600); err != nil {
					t.Fatal(err)
				}
				h.restartJournal(t)
			} else {
				if err := os.WriteFile(path, []byte("{invalid"), 0o600); err != nil {
					t.Fatal(err)
				}
				h.b.engineConfigMu.Lock()
				h.b.engineSettingsLoaded = false
				h.b.engineConfigMu.Unlock()
			}
			if h.b.replayPolicyToProxy(t.Context(), h.pp) {
				t.Fatal("known model journal error enabled proxy replay")
			}
			worker := h.b.getEngineMgr()
			h.b.restoreEnabledEngines(worker)
			if err := modelSettingsCall(t.Context(), worker, "engine:get-launch", map[string]string{"engine": "llamacpp"}, nil); err != nil {
				t.Fatal(err)
			}
			for _, call := range h.engine.snapshot() {
				if call.method == restoreEnabledEnginesMethod {
					t.Fatal("known unresolved model allowed global restoration")
				}
			}
			if h.calls() != 0 || h.b.engineSettingsError == nil {
				t.Fatal("known model journal error was hidden or produced effects")
			}
		})
	}
	test("known in memory", false)
	test("recognizable invalid model subrecord", true)
}

func TestModelSettingsRecoveryCompletedHistoryDoesNotBlockOrdinaryJournalError(t *testing.T) {
	h := newModelSettingsHarness(t)
	record := h.seedAccepted(t, true)
	record.Model.Stage, record.Snapshot.CoupledStage, record.Snapshot.Phase = "completed", "completed", "succeeded"
	receipt := record.Receipts[record.Snapshot.RequestID]
	receipt.Phase = "succeeded"
	record.Receipts[record.Snapshot.RequestID] = receipt
	data, err := json.Marshal(map[string]*engineSettingsRecord{"llamacpp": record, "ollama": nil})
	if err != nil {
		t.Fatal(err)
	}
	path, err := h.b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	h.restartJournal(t)
	if !h.b.replayPolicyToProxy(t.Context(), h.pp) {
		t.Fatal("completed model history blocked ordinary invalid-row replay")
	}
	worker := h.b.getEngineMgr()
	h.b.restoreEnabledEngines(worker)
	if err := modelSettingsCall(t.Context(), worker, "engine:get-launch", map[string]string{"engine": "llamacpp"}, nil); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, call := range h.engine.snapshot() {
		if call.method == restoreEnabledEnginesMethod {
			found = true
		}
	}
	if !found || h.b.engineSettingsError == nil || h.calls() != 0 {
		t.Fatal("completed model history changed ordinary journal-error restoration")
	}
}
