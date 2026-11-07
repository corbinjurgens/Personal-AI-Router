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
	return h.seedAcceptedRequest(t, h.request(t, coupled))
}

func (h *modelSettingsHarness) seedAcceptedRequest(t *testing.T, p settings.Request) *engineSettingsRecord {
	t.Helper()
	p = p.Clone()
	preview, err := h.b.previewEngineSettings(context.Background(), p, "")
	if err != nil || len(preview.Errors) != 0 {
		t.Fatalf("seed preview=%+v, %v", preview, err)
	}
	p.ModelChange.PreviewDigest = preview.PreviewDigest
	_, source, err := h.b.retainedModelReview(context.Background(), p, "")
	if err != nil {
		t.Fatal(err)
	}
	if source.native(*p.ModelChange) && source.NativeDigest == preview.PreviewDigest {
		t.Fatal("seed confused the native review digest with its public wrapper")
	}
	policy := h.b.localPolicy().Policy
	next, err := h.b.mergeModelPolicy(policy, p.Engine, *p.ModelChange, "", preview, source.ReviewedModelChange != nil)
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
	var reviewed modelconfig.ModelChange
	if source.ReviewedModelChange != nil {
		reviewed = source.ReviewedModelChange.Clone()
	}
	record.Model = &modelSettingsOperation{Change: p.ModelChange.Clone(), Review: preview.Clone(), ReviewedModelChange: reviewed, NativeBefore: record.Previous.Clone(), Before: modelPolicyFingerprint(policy), After: modelPolicyFingerprint(next), Implications: modelPolicyImplications(policy, p.Engine, modelChangeKey(*p.ModelChange)), Stage: "drained"}
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

func TestModelSettingsRecoveryNewProfileUsesAcceptedIdentityAndContext(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.request(t, false)
	p.ModelChange.ProfileKey = "model-32k"
	p.ModelChange.Load.Set.Context.Tokens = 32768
	slots := int32(2)
	p.ModelChange.Load.Set.Slots = &slots
	record := h.seedAcceptedRequest(t, p)
	h.setNative(record.Snapshot.Settings)
	h.restartJournal(t)
	if !h.b.recoverEngineSettings() || h.calls() != 0 {
		t.Fatalf("native-persisted create did not finish policy exactly once; calls = %d", h.calls())
	}
	profile, ok := h.b.localPolicy().Policy.ProfileByKey("model-32k")
	if !ok || profile.Model != "model-32k" || profile.ContextTokens != 32768 || profile.ModelIdentity == nil || profile.ModelIdentity.GeneralName != "Fixture model" || profile.ModelIdentity.ConfiguredContextTokens != 32768 {
		t.Fatalf("recovered metadata = %+v", profile)
	}
	after := h.b.localPolicy().Policy
	if modelPolicyFingerprint(after) != record.Model.After || !h.b.recoverEngineSettings() || h.b.localPolicy().Policy.Revision != after.Revision || h.calls() != 0 {
		t.Fatal("recovery changed the accepted fingerprint or repeated policy/native effects")
	}
}

func TestModelSettingsRecoveryRemoveFinishesPolicyAfterNativePersisted(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.createProfile(t, "model-8k", 8192)
	policy := h.b.localPolicy().Policy
	policy.Roles = []nodepolicy.Role{{Name: "weak", Profiles: []string{"model-8k"}}}
	policy.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: "model-8k"}}
	if _, err := setCurrent(h.b, settingsJSON(policy)); err != nil {
		t.Fatal(err)
	}
	p := h.removeRequest(t, "model-8k")
	record := h.seedAcceptedRequest(t, p)
	want := h.removalEvidence(t, p)
	h.setNative(record.Snapshot.Settings)
	h.b.engineConfigMu.Lock()
	live := h.b.engineSettings["llamacpp"]
	live.Model.Stage, live.Snapshot.CoupledStage = "native-persisted", "native-persisted"
	if err := h.b.saveEngineSettingsLocked(); err != nil {
		h.b.engineConfigMu.Unlock()
		t.Fatal(err)
	}
	h.b.engineConfigMu.Unlock()
	h.restartJournal(t)
	start := len(h.engine.snapshot())
	if !h.b.recoverEngineSettings() || h.calls() != 1 {
		t.Fatalf("remove recovery repeated native persistence; calls = %d", h.calls())
	}
	h.assertRemovalForwarding(t, start, want, 1, 0)
	policy = h.b.localPolicy().Policy
	if len(policy.Profiles) != 0 || !reflect.DeepEqual(policy.Roles, []nodepolicy.Role{{Name: "weak", Profiles: []string{}}}) || policy.ServedModels == nil || len(policy.ServedModels) != 0 || len(h.record(t).Snapshot.Settings.ModelLoads) != 0 || h.record(t).Model.Stage != "completed" || modelPolicyFingerprint(policy) != record.Model.After {
		t.Fatalf("remove recovery left policy or native references: %+v", policy)
	}
	if !h.b.recoverEngineSettings() || h.b.localPolicy().Policy.Revision != policy.Revision || h.calls() != 1 {
		t.Fatal("completed remove recovery repeated effects")
	}
}

func TestModelSettingsRecoveryRemoveForwardsPersistedNativeReview(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.createProfile(t, "model-8k", 8192)
	p := h.removeRequest(t, "model-8k")
	record := h.seedAcceptedRequest(t, p)
	want := h.removalEvidence(t, p)
	if !reflect.DeepEqual(record.Model.ReviewedModelChange, want.Change) || record.Model.Review.PreviewDigest == want.Change.PreviewDigest {
		t.Fatalf("seed lost native/public digest separation: %+v", record.Model)
	}
	h.restartJournal(t)
	start := len(h.engine.snapshot())
	if !h.b.recoverEngineSettings() || h.calls() != 2 || len(h.b.localPolicy().Policy.Profiles) != 0 {
		t.Fatal("accepted removal did not recover native then policy")
	}
	h.assertRemovalForwarding(t, start, want, 1, 1)
	if proof := h.record(t).Model.ReviewedModelChange; !reflect.DeepEqual(proof, want.Change) {
		t.Fatalf("recovery changed original removal evidence: %+v", proof)
	}
}

func TestModelSettingsRecoveryRemoveRefusesInvalidNativeReview(t *testing.T) {
	test := func(name string, corrupt func(*modelSettingsOperation), wantPreview bool) {
		t.Run(name, func(t *testing.T) {
			h := newModelSettingsHarness(t)
			h.createProfile(t, "model-8k", 8192)
			h.seedAcceptedRequest(t, h.removeRequest(t, "model-8k"))
			beforePolicy := h.b.localPolicy().Policy
			h.mu.Lock()
			beforeNative := settings.ConfigFromLaunch(h.launch, 18081)
			h.mu.Unlock()
			h.b.engineConfigMu.Lock()
			corrupt(h.b.engineSettings["llamacpp"].Model)
			err := h.b.saveEngineSettingsLocked()
			h.b.engineConfigMu.Unlock()
			if err != nil {
				t.Fatal(err)
			}
			h.restartJournal(t)
			start := len(h.engine.snapshot())
			if h.b.recoverEngineSettings() || h.calls() != 1 || !reflect.DeepEqual(h.b.localPolicy().Policy, beforePolicy) {
				t.Fatal("invalid accepted removal proof reached native or policy effects")
			}
			h.mu.Lock()
			afterNative := settings.ConfigFromLaunch(h.launch, 18081)
			h.mu.Unlock()
			if !afterNative.Equal(beforeNative) {
				t.Fatal("invalid removal proof changed native settings")
			}
			previews := 0
			for _, call := range h.engine.snapshot()[start:] {
				if call.method == "engine:preview-launch" {
					previews++
				}
				if call.method == "engine:configure-launch" {
					t.Fatal("invalid native removal proof reached configure")
				}
			}
			if wantPreview && previews != 1 || !wantPreview && previews != 0 {
				t.Fatalf("frozen proof checks = %d, want worker validation = %v", previews, wantPreview)
			}
		})
	}
	test("missing proof", func(op *modelSettingsOperation) { op.ReviewedModelChange = modelconfig.ModelChange{} }, false)
	test("missing digest", func(op *modelSettingsOperation) { op.ReviewedModelChange.PreviewDigest = "" }, false)
	test("wrapped digest", func(op *modelSettingsOperation) { op.ReviewedModelChange.PreviewDigest = op.Review.PreviewDigest }, false)
	test("wrong digest", func(op *modelSettingsOperation) { op.ReviewedModelChange.PreviewDigest = strings.Repeat("0", 64) }, true)
}

func TestModelSettingsRecoveryRemoveNativeFailureKeepsPolicy(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.createProfile(t, "model-8k", 8192)
	p := h.review(t, h.removeRequest(t, "model-8k"), "")
	before := h.b.localPolicy().Policy
	h.mu.Lock()
	h.configureFailure = "no effect"
	h.mu.Unlock()
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	record := h.record(t)
	if err != nil || receipt.Phase != "failed" || !reflect.DeepEqual(h.b.localPolicy().Policy, before) || record.Snapshot.EffectiveSettings == nil || len(record.Snapshot.EffectiveSettings.ModelLoads) != 1 || h.calls() != 2 {
		t.Fatalf("native failure cleaned policy: %+v, %v, policy = %+v", receipt, err, h.b.localPolicy().Policy)
	}
}

func TestModelSettingsRecoveryRemovePolicyFailureRetainsNativeRemoval(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.createProfile(t, "model-8k", 8192)
	p := h.review(t, h.removeRequest(t, "model-8k"), "")
	before := h.b.localPolicy().Policy
	setReplacer(h.b, durablefile.Replacer{Replace: func(string, string) error { return errors.New("policy removal refused") }})
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	record := h.record(t)
	if err != nil || receipt.Phase != "failed" || !reflect.DeepEqual(h.b.localPolicy().Policy, before) || record.Model.Stage != "native-persisted" || record.Snapshot.EffectiveSettings == nil || len(record.Snapshot.EffectiveSettings.ModelLoads) != 0 || h.calls() != 2 {
		t.Fatalf("policy failure lost native-first removal: %+v, %v, record = %+v", receipt, err, record)
	}
	snapshot, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, "")
	if err != nil || snapshot.Phase != "applying" || snapshot.Editable || snapshot.UnresolvedError == "" || !modelSettingsNeedsReconciliation(record) {
		t.Fatalf("failed removal lost its unresolved settings fence: %+v, %v", snapshot, err)
	}
	if _, err := h.b.setLocalPolicy(settingsJSON(before), settingsJSON(before.Revision)); policyErrorCode(err) != codePolicyConflict {
		t.Fatalf("policy write bypassed unresolved native removal: %v", err)
	}
}

func TestModelSettingsRecoveryRemoveUncertainAcceptanceMakesNoEffects(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.createProfile(t, "model-8k", 8192)
	p := h.review(t, h.removeRequest(t, "model-8k"), "")
	before := h.b.localPolicy().Policy
	h.b.engineSettingsReplacer = durablefile.Replacer{SyncDir: func(string) error { return errors.New("remove acceptance uncertain") }}
	_, err := h.b.applyEngineSettings(context.Background(), p, "")
	if !errors.Is(err, durablefile.ErrReplaced) || h.calls() != 1 || !reflect.DeepEqual(h.b.localPolicy().Policy, before) || h.record(t).Model.Stage != "acceptance-uncertain" {
		t.Fatalf("uncertain acceptance made removal effects: %v, calls = %d", err, h.calls())
	}
	h.restartJournal(t)
	if h.b.recoverEngineSettings() || h.calls() != 1 || !reflect.DeepEqual(h.b.localPolicy().Policy, before) {
		t.Fatal("recovery executed an unconfirmed remove")
	}
}

func TestModelSettingsRecoveryRemoveSupersessionFinishesLandedNativeCleanup(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.createProfile(t, "model-8k", 8192)
	first := h.review(t, h.removeRequest(t, "model-8k"), "")
	want := h.removalEvidence(t, first)
	setReplacer(h.b, durablefile.Replacer{Replace: func(string, string) error { return errors.New("policy cleanup refused") }})
	if receipt, err := h.b.applyEngineSettings(context.Background(), first, ""); err != nil || receipt.Phase != "failed" || h.calls() != 2 {
		t.Fatalf("seed failed removal = %+v, %v", receipt, err)
	}
	setReplacer(h.b, durablefile.Replacer{})
	second := h.removeRequest(t, "model-8k")
	second.ModelChange.SupersedesRequestID = first.RequestID
	start := len(h.engine.snapshot())
	second = h.review(t, second, "")
	if receipt, err := h.b.applyEngineSettings(context.Background(), second, ""); err != nil || receipt.Phase != "succeeded" || h.calls() != 2 {
		t.Fatalf("remove supersession repeated native removal = %+v, %v, calls = %d", receipt, err, h.calls())
	}
	h.assertRemovalForwarding(t, start, want, 3, 0)
	record := h.record(t)
	if len(h.b.localPolicy().Policy.Profiles) != 0 || record.Model.Stage != "completed" || record.Receipts[first.RequestID].Phase != "superseded" || len(record.Snapshot.Settings.ModelLoads) != 0 {
		t.Fatalf("supersession lost cleanup or receipts: %+v", record)
	}
}

func TestModelSettingsRecoveryRemoveLostNativeResponseUsesReadback(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.createProfile(t, "model-8k", 8192)
	p := h.review(t, h.removeRequest(t, "model-8k"), "")
	want := h.removalEvidence(t, p)
	start := len(h.engine.snapshot())
	h.mu.Lock()
	h.configureFailure = "lost response"
	h.mu.Unlock()
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "succeeded" || len(h.b.localPolicy().Policy.Profiles) != 0 || len(h.record(t).Snapshot.Settings.ModelLoads) != 0 || h.calls() != 2 {
		t.Fatalf("lost remove response = %+v, %v, calls = %d", receipt, err, h.calls())
	}
	h.assertRemovalForwarding(t, start, want, 2, 1)
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
	next, err := h.b.mergeModelPolicy(r.current, "llamacpp", record.Model.Change, "", record.Model.Review, record.Model.ReviewedModelChange.Remove)
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
			envelope, err := settings.DecodeWorkerRequest[settings.Request](call.params)
			if err != nil {
				t.Fatal(err)
			}
			if envelope.Request.ModelChange != nil {
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

func TestModelSettingsRecoveryFailedAttemptRetainsPendingOperation(t *testing.T) {
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
	policy.Profiles = []nodepolicy.Profile{{Key: "fixture/profile", DisplayName: "Fixture", Engine: "llamacpp", Model: "fixture/model", RequestOptions: json.RawMessage(`{"temperature":0.9}`)}}
	// Simulate third content outside the reserved public writer. A load-only
	// operation now reserves policy too because it commits derived metadata.
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current = policy
	r.mu.Unlock()
	writePolicyFile(t, h.path, policy)
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
	h.b.restoreInstalledEngines(worker)
	got := h.record(t)
	if got.Model.Stage != "completed" || !got.Snapshot.Running || got.Resume || h.calls() != 1 {
		t.Fatalf("saved resume=%+v, calls=%d", got.Snapshot, h.calls())
	}
	if err := modelSettingsCall(t.Context(), worker, "engine:get-launch", map[string]string{"engine": "llamacpp"}, nil); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, call := range h.engine.snapshot() {
		if call.method == restoreInstalledEnginesMethod {
			found = true
		}
	}
	if !found {
		t.Fatal("successful resume still blocked installed-engine restoration")
	}
}

func TestModelSettingsRecoverySavedResumeHonorsCurrentVeto(t *testing.T) {
	test := func(name string, paused bool, wantCalls int) {
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
	test("Pause", true, 0)
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
	third := []byte(`{"llamacpp-custom":{"snapshot":{"phase":"idle"},"receipts":{}}}`)
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
	h.b.restoreInstalledEngines(worker)
	if err := modelSettingsCall(t.Context(), worker, "engine:get-launch", map[string]string{"engine": "llamacpp"}, nil); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, call := range h.engine.snapshot() {
		if call.method == restoreInstalledEnginesMethod {
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
			h.b.restoreInstalledEngines(worker)
			if err := modelSettingsCall(t.Context(), worker, "engine:get-launch", map[string]string{"engine": "llamacpp"}, nil); err != nil {
				t.Fatal(err)
			}
			for _, call := range h.engine.snapshot() {
				if call.method == restoreInstalledEnginesMethod {
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
	data, err := json.Marshal(map[string]*engineSettingsRecord{"llamacpp": record, "llamacpp-custom": nil})
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
	h.b.restoreInstalledEngines(worker)
	if err := modelSettingsCall(t.Context(), worker, "engine:get-launch", map[string]string{"engine": "llamacpp"}, nil); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, call := range h.engine.snapshot() {
		if call.method == restoreInstalledEnginesMethod {
			found = true
		}
	}
	if !found || h.b.engineSettingsError == nil || h.calls() != 0 {
		t.Fatal("completed model history changed ordinary journal-error restoration")
	}
}
