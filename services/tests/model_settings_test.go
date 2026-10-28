// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/gguffacts"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

func TestModelSettingsOffExactAliasAndShardsHaveNoEffects(t *testing.T) {
	f := prepareModelSettingsProcessFixture(t)
	modelSettingsWrite(t, filepath.Join(f.root, "split-00001-of-00002.gguf"), modelSettingsGGUF("blk.0.weight"))
	modelSettingsWrite(t, filepath.Join(f.root, "split-00002-of-00002.gguf"), modelSettingsGGUF("blk.1.weight"))
	modelSettingsWrite(t, f.preset, []byte("[exact-split]\nmodel = split-00001-of-00002.gguf\n[missing-split]\nmodel = absent-00001-of-00002.gguf\n"))
	f.startModelSettings(t)
	baseline := f.metadataBaseline(t)
	var resolved modelconfig.InspectResult
	modelSettingsDecode(t, f.rpcOK(t, "engine:inspect-model", modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "exact-split"}), &resolved)
	if resolved.Status != "resolved" || len(resolved.Artifacts) != 2 || resolved.Facts == nil || resolved.Facts.Architecture != "llama" || resolved.Facts.Dimensions["tensor.bytes"] != "256" {
		t.Fatalf("exact split inspection=%+v", resolved)
	}
	if resolved.Artifacts[0].Role != "model" || resolved.Artifacts[1].Role != "model" || resolved.Artifacts[0].Identity == resolved.Artifacts[1].Identity {
		t.Fatal("ordered shard bindings were not distinct")
	}
	assertModelSettingsInspectionProvenance(t, resolved)
	for i, artifact := range resolved.Artifacts {
		relative := fmt.Sprintf("split-%05d-of-00002.gguf", i+1)
		binding := modelSettingsExpectedBinding(t, f.root, "model", relative, modelSettingsGGUF(fmt.Sprintf("blk.%d.weight", i)))
		assertModelSettingsInspectionBinding(t, artifact, binding)
	}
	for _, model := range []string{"exact", "missing-split"} {
		var missing modelconfig.InspectResult
		modelSettingsDecode(t, f.rpcOK(t, "engine:inspect-model", modelconfig.InspectRequest{Engine: "llamacpp", ModelID: model}), &missing)
		if missing.Status == "resolved" || len(missing.Artifacts) != 0 {
			t.Fatalf("inexact/missing alias %s resolved", model)
		}
	}
	f.noMetadataEffects(t, baseline)
}

func TestModelSettingsOffManualCoupledPreservesUnselectedState(t *testing.T) {
	f := newModelSettingsProcessFixture(t)
	f.seedModelSettingsPolicyBounds(t)
	baseline := f.metadataBaseline(t)
	policy := baseline.policy
	request, preview := f.review(t, f.request(t, "off-coupled", &modelconfig.PolicyPatch{Generation: map[string]*json.Number{"temperature": modelSettingsNumber("0.75")}}))
	f.noMetadataEffects(t, baseline)
	assertModelSettingsImpact(t, preview, false)
	assertModelSettingsFinite(t, preview.Settings, f.root)
	accepted := f.rpcOK(t, "engine:apply-settings", request)
	state := f.settled(t, request.RequestID, "succeeded")
	if state.Running || state.CoupledStage != "completed" || state.Revision != state.AppliedRevision || state.EffectiveSettings == nil || !state.EffectiveSettings.Equal(preview.Settings) || !state.Settings.Equal(preview.Settings) {
		t.Fatalf("Off completion did not match reviewed persisted settings: phase=%s stage=%s running=%t", state.Phase, state.CoupledStage, state.Running)
	}
	assertModelSettingsPolicyPatch(t, policy, f.policy(t), "0.75")
	disk := f.disk(t)
	if disk.Model.Stage != "completed" || !disk.Model.Attempted || disk.Receipts[request.RequestID].Phase != "succeeded" {
		t.Fatal("completed native/policy receipt missing from durable journal")
	}
	replay := f.rpcOK(t, "engine:apply-settings", request)
	var first, again struct {
		Revision uint64 `json:"revision"`
		Phase    string `json:"phase"`
	}
	modelSettingsDecode(t, accepted, &first)
	modelSettingsDecode(t, replay, &again)
	if again.Revision != state.Revision || again.Phase != "succeeded" || first.Revision != again.Revision {
		t.Fatal("identical retry did not read the accepted receipt")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.configureCalls != 1 || f.mutations != 0 || f.inferences != 0 {
		t.Fatalf("Off apply effects configure=%d mutation=%d inference=%d", f.configureCalls, f.mutations, f.inferences)
	}
}

func TestModelSettingsPartialEstimateIsAdvisoryAndHasNoEffects(t *testing.T) {
	f := prepareModelSettingsProcessFixture(t)
	f.pinned(t)
	f.startModelSettings(t)
	f.seedModelSettingsPolicyBounds(t)
	baseline := f.metadataBaseline(t)
	params := modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "medium-model"}
	var local, remote modelconfig.InspectResult
	modelSettingsDecode(t, f.rpcOK(t, "engine:inspect-model", params), &local)
	modelSettingsDecode(t, f.peer(t, "inspect", params, http.StatusOK), &remote)
	for _, result := range []modelconfig.InspectResult{local, remote} {
		if result.Status != "resolved" || result.ModelID != params.ModelID || result.Estimate.Status != "partial" {
			t.Fatalf("bounded synthetic inspection lost partial estimate: %+v", result)
		}
		assertModelSettingsAdvisoryEstimate(t, result.Estimate)
		weights := slices.IndexFunc(result.Estimate.Components, func(c modelconfig.EstimateComponent) bool { return c.Name == "weights" })
		if weights < 0 || result.Estimate.Components[weights].Bytes != "128" || result.Estimate.Components[weights].Status != "estimated" {
			t.Fatal("partial estimate lost the known synthetic tensor storage")
		}
	}
	request := f.request(t, "partial-estimate-preview", nil)
	_, preview := f.review(t, request)
	if preview.Estimate == nil || preview.Estimate.Status != "partial" || preview.Auto != nil {
		t.Fatal("manual preview did not retain a partial advisory estimate")
	}
	assertModelSettingsAdvisoryEstimate(t, *preview.Estimate)
	assertModelSettingsFinite(t, preview.Settings, f.root)
	var peerPreview settings.Preview
	modelSettingsDecode(t, f.peer(t, "preview", request, http.StatusOK), &peerPreview)
	if len(peerPreview.Errors) != 0 || peerPreview.Estimate == nil || peerPreview.Estimate.Status != "partial" || !peerPreview.Settings.Equal(preview.Settings) {
		t.Fatal("pinned preview lost target finite settings or partial estimate")
	}
	assertModelSettingsAdvisoryEstimate(t, *peerPreview.Estimate)
	f.noMetadataEffects(t, baseline)
}

func TestModelSettingsAutoUnavailablePreservesManualConfiguration(t *testing.T) {
	f := prepareModelSettingsProcessFixture(t)
	f.pinned(t)
	f.startModelSettings(t)
	f.seedModelSettingsPolicyBounds(t)
	baseline := f.metadataBaseline(t)
	for _, mode := range []string{"context", "gpu", "context-and-gpu"} {
		t.Run(mode, func(t *testing.T) {
			request := f.request(t, "unavailable-auto-"+mode, nil)
			if mode != "gpu" {
				request.ModelChange.Load.Set.Context = &modelconfig.Context{Mode: "auto", Floor: 4096, Cap: 8192}
			}
			if mode != "context" {
				request.ModelChange.Load.Set.GPU = &modelconfig.GPU{Mode: "auto"}
			}
			response := f.rpc(t, "engine:preview-settings", request)
			if response.Error == nil || response.Error.Code != -32602 || !strings.Contains(response.Error.Message, "target memory telemetry unavailable") || len(response.Result) != 0 {
				t.Fatalf("Auto without a target memory owner did not refuse: %+v", response)
			}
			remote := f.peer(t, "preview", request, http.StatusBadRequest)
			if !strings.Contains(string(remote), "target memory telemetry unavailable") {
				t.Fatal("pinned refusal lost the missing target memory authority reason")
			}
			if response := f.rpc(t, "engine:apply-settings", request); response.Error == nil {
				t.Fatal("unreviewed unavailable Auto was accepted")
			}
			f.noMetadataEffects(t, baseline)
		})
	}
	request, preview := f.review(t, f.request(t, "manual-after-unavailable-auto", nil))
	f.rpcOK(t, "engine:apply-settings", request)
	after := f.settled(t, request.RequestID, "succeeded")
	assertModelSettingsSavedReadback(t, after, preview.Settings)
	assertModelSettingsFinite(t, after.Settings, f.root)
	if after.Running || !reflect.DeepEqual(baseline.policy, f.policy(t)) {
		t.Fatal("manual fallback started the engine or changed unselected policy")
	}
}

func TestModelSettingsEstimatorUnavailableManualResolutionRecoversExactly(t *testing.T) {
	f := prepareModelSettingsProcessFixture(t)
	// The exact file still exists, but its bounded facts cannot be decoded.
	// Manual CPU settings do not require an estimator or model execution.
	modelSettingsWrite(t, filepath.Join(f.root, "medium.gguf"), []byte("synthetic unavailable GGUF metadata"))
	f.startModelSettings(t)
	f.seedModelSettingsPolicyBounds(t)
	baseline := f.metadataBaseline(t)
	var inspection modelconfig.InspectResult
	modelSettingsDecode(t, f.rpcOK(t, "engine:inspect-model", modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "medium-model"}), &inspection)
	if inspection.Estimate.Status != "unknown" {
		t.Fatalf("unreadable facts returned estimate status %q", inspection.Estimate.Status)
	}
	assertModelSettingsAdvisoryEstimate(t, inspection.Estimate)
	request, preview := f.review(t, f.request(t, "manual-without-estimator", nil))
	if preview.Estimate == nil || preview.Estimate.Status != "unknown" || preview.Auto != nil {
		t.Fatal("manual preview without facts did not disclose unavailable estimate")
	}
	assertModelSettingsAdvisoryEstimate(t, *preview.Estimate)
	index := slices.IndexFunc(preview.Settings.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ModelID == "medium-model" })
	if index < 0 || preview.Settings.ModelLoads[index].Resolved == nil {
		t.Fatal("manual preview without estimator lost its finite allocation")
	}
	resolved := preview.Settings.ModelLoads[index].Resolved
	if resolved.PerSlotContext != 8192 || resolved.TotalContext != 16384 || resolved.Slots != 2 || resolved.GPU.Mode != "cpu" || resolved.ReviewedInputID == "" || resolved.NativeBuildID == "" || resolved.PresetDigest == "" {
		t.Fatalf("manual preview without estimator returned incomplete resolution: %+v", resolved)
	}
	binding := modelSettingsExpectedBinding(t, f.root, "model", "medium.gguf", []byte("synthetic unavailable GGUF metadata"))
	assertModelSettingsBindings(t, f.root, resolved.Artifacts, []modelconfig.ArtifactBinding{binding})
	assertModelSettingsUnselectedLoad(t, preview.Settings, "weak-model", 4096, 1)
	f.noMetadataEffects(t, baseline)
	gate := newModelSettingsGate()
	f.readbackGate = gate
	f.begin(t, "engine:apply-settings", request)
	modelSettingsWait(t, gate)
	if f.disk(t).Model.Stage != "native-applying" {
		t.Fatal("manual recovery did not reach durable native application")
	}
	f.crashBroker(t, gate)
	f.startModelSettings(t)
	after := f.settled(t, request.RequestID, "succeeded")
	assertModelSettingsSavedReadback(t, after, preview.Settings)
	if after.Running || !reflect.DeepEqual(baseline.policy, f.policy(t)) {
		t.Fatal("recovery retuned manual settings or changed unselected policy")
	}
	var replay struct {
		Revision uint64 `json:"revision"`
		Phase    string `json:"phase"`
	}
	modelSettingsDecode(t, f.rpcOK(t, "engine:apply-settings", request), &replay)
	if replay.Revision != after.Revision || replay.Phase != "succeeded" {
		t.Fatal("manual recovery lost the exact accepted receipt")
	}
}

func assertModelSettingsAdvisoryEstimate(t *testing.T, estimate modelconfig.Estimate) {
	t.Helper()
	if err := estimate.Validate(); err != nil {
		t.Fatalf("invalid advisory estimate: %v", err)
	}
	if estimate.Source == "" || len(estimate.Unknowns) == 0 || estimate.CapacityFit != "unknown" || estimate.FreeFit != "unknown" {
		t.Fatalf("incomplete estimate claimed memory fit or lost provenance: %+v", estimate)
	}
	for _, domain := range estimate.Domains {
		if domain.CapacityFit != "unknown" || domain.FreeFit != "unknown" || domain.CapacityBytes != "" || domain.FreeBytes != "" {
			t.Fatal("estimate manufactured target memory authority")
		}
	}
}

func TestModelSettingsOrdinaryFullSettingsPreservesModelLoads(t *testing.T) {
	f := newModelSettingsProcessFixture(t)
	request, _ := f.review(t, f.request(t, "typed-baseline", nil))
	f.rpcOK(t, "engine:apply-settings", request)
	before := f.settled(t, request.RequestID, "succeeded")
	policy := f.policy(t)
	p := settings.Request{Engine: "llamacpp", RequestID: "ordinary-proxy-port", ExpectedRevision: before.Revision, Settings: before.Settings.Clone()}
	p.Settings.ProxyPort = freePort(t)
	f.p.ports = append(f.p.ports, p.Settings.ProxyPort)
	var preview settings.Preview
	modelSettingsDecode(t, f.rpcOK(t, "engine:preview-settings", p), &preview)
	if len(preview.Errors) != 0 || preview.Conflict != nil || !modelconfig.EqualLoads(preview.Settings.ModelLoads, before.Settings.ModelLoads) {
		t.Fatal("ordinary preview lost typed loads")
	}
	f.rpcOK(t, "engine:apply-settings", p)
	after := f.settled(t, p.RequestID, "succeeded")
	want := before.Settings.Clone()
	want.ProxyPort = p.Settings.ProxyPort
	if !after.Settings.Equal(want) || after.Running || !reflect.DeepEqual(policy, f.policy(t)) {
		t.Fatal("ordinary full settings changed an unselected field")
	}
	// The Task 4 TUI constructor coverage remains at ui/modelsettings_preserve_test.go.
	// Here the same complete settings payload crosses the real service boundary.
	forged := settings.Request{Engine: "llamacpp", RequestID: "ordinary-forged", ExpectedRevision: after.Revision, Settings: after.Settings.Clone()}
	forged.Settings.ModelLoads[0].Requested.Slots = modelSettingsInt(3)
	response := f.rpc(t, "engine:apply-settings", forged)
	if response.Error == nil || response.Error.Code != -32602 {
		t.Fatal("ordinary writer replaced target-owned typed settings")
	}
	if current := f.snapshot(t); !current.Settings.Equal(after.Settings) || current.Revision != after.Revision {
		t.Fatal("rejected ordinary write changed the target")
	}
}

func TestModelSettingsSourceDriftRequiresFreshPreview(t *testing.T) {
	f := newModelSettingsProcessFixture(t)
	baseline := f.metadataBaseline(t)
	request, _ := f.review(t, f.request(t, "drifted-review", nil))
	data := modelSettingsGGUF("blk.changed.weight")
	modelSettingsWrite(t, filepath.Join(f.root, "medium.gguf"), data)
	response := f.rpc(t, "engine:apply-settings", request)
	if response.Error == nil || response.Error.Code != -32010 {
		t.Fatal("stale artifact binding was accepted")
	}
	f.noMetadataEffects(t, baseline)
	if _, ok := f.disk(t).Receipts[request.RequestID]; ok {
		t.Fatal("drift refusal created an accepted receipt")
	}
	request, _ = f.review(t, f.request(t, "fresh-after-drift", nil))
	f.rpcOK(t, "engine:apply-settings", request)
	f.settled(t, request.RequestID, "succeeded")
}

func TestModelSettingsRawPolicyConflictDuringManagerWait(t *testing.T) {
	f := prepareModelSettingsProcessFixture(t)
	f.pinned(t)
	f.startModelSettings(t)
	before := f.policy(t)
	request, _ := f.review(t, f.request(t, "writer-fence", &modelconfig.PolicyPatch{Generation: map[string]*json.Number{"temperature": modelSettingsNumber("0.6")}}))
	gate := newModelSettingsGate()
	f.configureGate = gate
	applyID := f.begin(t, "engine:apply-settings", request)
	modelSettingsWait(t, gate)
	disk := f.disk(t)
	if disk.Model.Stage != "native-applying" || !disk.Model.Attempted {
		t.Fatal("manager barrier did not follow durable native-applying stage")
	}
	changed := before
	changed.Idle.UnloadAfterMinutes = 17
	response := f.rpc(t, "policy:set", map[string]any{"policy": changed, "expectedPersistedRevision": before.Revision})
	if response.Error == nil || response.Error.Code != -32010 {
		t.Fatal("raw policy writer escaped coupled reservation")
	}
	var peerFailure struct {
		Code string `json:"code"`
	}
	status, _, data := managementHTTPResponse(t, f.client, f.ports.engineControl, http.MethodPost, "/v1/node-policy/set", map[string]any{"policy": changed.PeerProjection(), "expectedPersistedRevision": before.Revision})
	if status != http.StatusConflict {
		t.Fatalf("raw pinned policy writer HTTP %d, want %d", status, http.StatusConflict)
	}
	modelSettingsDecode(t, data, &peerFailure)
	if peerFailure.Code != "policy_conflict" {
		t.Fatal("raw pinned policy writer escaped coupled reservation")
	}
	if !reflect.DeepEqual(before, f.policy(t)) {
		t.Fatal("refused raw policy write changed policy")
	}
	gate.unblock()
	f.finishApply(t, applyID)
	f.settled(t, request.RequestID, "succeeded")
	assertModelSettingsPolicyPatch(t, before, f.policy(t), "0.6")
}

func TestModelSettingsLoadOnlyAllowsUnrelatedPolicyDuringManagerWait(t *testing.T) {
	f := newModelSettingsProcessFixture(t)
	request, preview := f.review(t, f.request(t, "load-only-writer", nil))
	gate := newModelSettingsGate()
	f.configureGate = gate
	applyID := f.begin(t, "engine:apply-settings", request)
	modelSettingsWait(t, gate)
	policy := f.policy(t)
	policy.Idle.UnloadAfterMinutes = 17
	want := f.setPolicy(t, policy)
	gate.unblock()
	f.finishApply(t, applyID)
	after := f.settled(t, request.RequestID, "succeeded")
	assertModelSettingsSavedReadback(t, after, preview.Settings)
	if !reflect.DeepEqual(want, f.policy(t)) {
		t.Fatal("load-only publication overwrote unrelated raw policy change")
	}
}

func TestModelSettingsPauseWhileApplyingDoesNotResume(t *testing.T) {
	f := newModelSettingsProcessFixture(t)
	f.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
	request, preview := f.review(t, f.request(t, "pause-apply", &modelconfig.PolicyPatch{Generation: map[string]*json.Number{"temperature": modelSettingsNumber("0.7")}}))
	assertModelSettingsImpact(t, preview, true)
	gate := newModelSettingsGate()
	f.configureGate = gate
	applyID := f.begin(t, "engine:apply-settings", request)
	modelSettingsWait(t, gate)
	// Pause publishes its independent availability fence before waiting for
	// native lifecycle work. The owner must remain usable beside this gate.
	// Keep the pending Pause reply on the same JSON-RPC stream; the command
	// completes only after physical shutdown, which waits behind this apply.
	pauseID := f.begin(t, "node:set-availability", map[string]string{"state": "paused"})
	var paused nodepolicy.Policy
	task1Eventually(t, "Pause durably published beside apply", 10*time.Second, func() bool {
		var err error
		paused, err = nodepolicy.Parse(readLiveDurableFile(t, filepath.Join(jointAppDir(f.base), nodepolicy.FileName)))
		if err != nil {
			t.Fatal(err)
		}
		return paused.Availability == nodepolicy.Paused
	})
	if paused.Availability != nodepolicy.Paused || paused.PausedBy != nodepolicy.PausedLocally {
		t.Fatal("Pause did not reach its independent policy owner")
	}
	gate.unblock()
	response := f.await(t, pauseID)
	if response.Error != nil {
		t.Fatal("Pause did not complete after apply released its lifecycle reservation")
	}
	f.finishApply(t, applyID)
	var after settings.Snapshot
	task1Eventually(t, "paused apply settles without runtime", 90*time.Second, func() bool {
		after = f.snapshot(t)
		return !after.Running && (after.CoupledStage == "completed" || after.UnresolvedError != "")
	})
	current := f.policy(t)
	if current.Availability != nodepolicy.Paused || current.PausedBy != nodepolicy.PausedLocally {
		t.Fatal("model operation undid Pause")
	}
	if after.Serving != nil && len(after.Serving.Observed) != 0 {
		t.Fatal("paused settings claimed observed loaded readiness")
	}
	assertInteropListenerClosed(t, f.targetPort)
}

func TestModelSettingsRunningRestartConsumesWholeFinitePreset(t *testing.T) {
	f := newModelSettingsProcessFixture(t)
	policy := f.policy(t)
	f.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
	initial := f.engineIdentity(t, 1)
	request, preview := f.review(t, f.request(t, "running-restart", nil))
	assertModelSettingsImpact(t, preview, true)
	f.rpcOK(t, "engine:apply-settings", request)
	after := f.settled(t, request.RequestID, "succeeded")
	if !after.Running || !after.Settings.Equal(preview.Settings) || !reflect.DeepEqual(policy, f.policy(t)) {
		t.Fatal("running apply did not persist selected settings and preserve policy")
	}
	assertSwitchProcessIdentityStopped(t, initial)
	replacement := f.engineIdentity(t, 2)
	if replacement == initial {
		t.Fatal("running apply reused the initial router lifetime")
	}
	sections := modelSettingsPreset(t, readLiveDurableFile(t, f.capture))
	if sections["medium-model"]["ctx-size"] != "16384" || sections["medium-model"]["parallel"] != "2" || sections["weak-model"]["ctx-size"] != "4096" || sections["weak-model"]["parallel"] != "1" {
		t.Fatal("synthetic router did not consume complete finite generated preset")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.mutations != 0 || f.inferences != 0 {
		t.Fatal("restart implicitly loaded/downloaded or inferred")
	}
}

func TestModelSettingsTimedDrainPrecedesWholeRouterRestart(t *testing.T) {
	f := prepareModelSettingsProcessFixture(t)
	f.enableAdmissionProbe(t)
	f.startModelSettings(t)
	f.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
	request, preview := f.review(t, f.request(t, "timed-drain", nil))
	assertModelSettingsImpact(t, preview, true)
	gate := newModelSettingsGate()
	f.inference = gate
	done := f.holdModelSettingsInference(t)
	f.assertHeldAdmission(t)
	initial := f.engineIdentity(t, 1)
	started := time.Now()
	// Do not shorten the production 60-second drain. The first native request
	// is an observable barrier after the broker's real admission drain.
	configure := newModelSettingsGate()
	f.mu.Lock()
	f.configureGate = configure
	f.mu.Unlock()
	applyID := f.begin(t, "engine:apply-settings", request)
	select {
	case <-configure.entered:
	case <-time.After(85 * time.Second):
		t.Fatal("native configure did not follow the disclosed drain bound")
	}
	if elapsed := time.Since(started); elapsed < 60*time.Second {
		t.Fatalf("active drain restarted early after %s", elapsed)
	}
	if f.disk(t).Model.Stage != "native-applying" {
		t.Fatal("timed drain did not durably precede native effects")
	}
	configure.unblock()
	f.finishApply(t, applyID)
	f.settled(t, request.RequestID, "succeeded")
	assertSwitchProcessIdentityStopped(t, initial)
	gate.unblock()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("interrupted synthetic inference did not return")
	}
	if replacement := f.engineIdentity(t, 2); replacement == initial {
		t.Fatal("timed drain did not restart the whole router")
	}
}

func TestModelSettingsFailedRestartRetainsDesiredEffectiveAndFence(t *testing.T) {
	f := newModelSettingsProcessFixture(t)
	f.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
	request, preview := f.review(t, f.request(t, "failed-restart", nil))
	f.switchProcessFixture.mu.Lock()
	f.failTarget = true
	f.switchProcessFixture.mu.Unlock()
	f.rpcOK(t, "engine:apply-settings", request)
	after := f.settled(t, request.RequestID, "applying")
	if after.UnresolvedError == "" || after.Running || after.EffectiveSettings == nil || !after.Settings.Equal(preview.Settings) || !after.EffectiveSettings.Equal(preview.Settings) || f.disk(t).Receipts[request.RequestID].Phase != "failed" {
		t.Fatal("failed restart lost persisted desired/effective state or recovery fence")
	}
	if response := f.rpc(t, "engine:apply-settings", request); response.Error == nil { // A historical failed receipt may also be returned successfully.
		var receipt struct {
			Phase string `json:"phase"`
		}
		modelSettingsDecode(t, response.Result, &receipt)
		if receipt.Phase != "failed" {
			t.Fatal("failed operation retry claimed success")
		}
	}
	if response := f.rpc(t, "engine:apply-settings", settings.Request{Engine: "llamacpp", ExpectedRevision: after.Revision, RequestID: "bypass-failed", Settings: after.Settings}); response.Error == nil {
		t.Fatal("ordinary writer bypassed unresolved model reservation")
	}
}

func TestModelSettingsBareAcceptedRestartRequiresFreshLocalSupersession(t *testing.T) {
	f := prepareModelSettingsProcessFixture(t)
	f.enableAdmissionProbe(t)
	f.startModelSettings(t)
	f.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
	before := f.policy(t)
	request, _ := f.review(t, f.request(t, "bare-accepted", &modelconfig.PolicyPatch{Generation: map[string]*json.Number{"temperature": modelSettingsNumber("0.35")}}))
	gate := newModelSettingsGate()
	f.mu.Lock()
	f.inference = gate
	f.mu.Unlock()
	done := f.holdModelSettingsInference(t)
	f.assertHeldAdmission(t)
	initial := f.engineIdentity(t, 1)
	f.mu.Lock()
	invocations := maps.Clone(f.managerInvocations)
	f.mu.Unlock()
	f.begin(t, "engine:apply-settings", request)
	task1Eventually(t, "durable acceptance during active drain", 10*time.Second, func() bool {
		record := f.disk(t)
		return record.Snapshot.RequestID == request.RequestID && record.Model.Stage == "accepted"
	})
	journalPath := filepath.Join(f.base, "engine-settings-operations.json")
	acceptedBytes := readLiveDurableFile(t, journalPath)
	accepted := f.disk(t)
	acceptedReceipt := accepted.Receipts[request.RequestID]
	if accepted.Model.Stage != "accepted" || accepted.Model.Attempted || acceptedReceipt.Phase != "applying" || len(acceptedReceipt.Hash) != 64 || acceptedReceipt.Revision != accepted.Snapshot.Revision || accepted.Snapshot.AppliedRevision == accepted.Snapshot.Revision {
		t.Fatal("durable acceptance lacks the unapplied, unattempted operation and applying receipt")
	}
	assertJournalUnchanged := func() {
		t.Helper()
		if !bytes.Equal(acceptedBytes, readLiveDurableFile(t, journalPath)) {
			t.Fatal("bare acceptance journal bytes changed before fresh supersession")
		}
	}
	f.crashBroker(t, gate)
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("crashed synthetic inference did not return")
	}
	assertSwitchProcessIdentityStopped(t, initial)
	assertJournalUnchanged()
	const interrupted = "Interrupted before changes; preview again on this device."
	type receipt struct {
		Hash     string `json:"hash"`
		Revision uint64 `json:"revision"`
		Phase    string `json:"phase"`
	}
	wantReceipt := receipt{Hash: acceptedReceipt.Hash, Revision: acceptedReceipt.Revision, Phase: "failed"}
	assertRecovery := func() settings.Snapshot {
		t.Helper()
		after := f.settled(t, request.RequestID, "applying")
		var live struct {
			Running bool `json:"running"`
		}
		modelSettingsDecode(t, f.rpcOK(t, "engine:status", map[string]string{"engine": "llamacpp"}), &live)
		t.Logf("accepted_recovery live_running=%t snapshot_running=%t", live.Running, after.Running)
		if after.CoupledStage != "acceptance-uncertain" || after.Running || after.Error != interrupted || after.UnresolvedError != interrupted || after.Revision != accepted.Snapshot.Revision || after.AppliedRevision != accepted.Snapshot.AppliedRevision || after.AppliedRevision == after.Revision || !reflect.DeepEqual(before, f.policy(t)) {
			t.Fatal("bare acceptance recovery lost stopped, interrupted, unapplied state or unchanged policy")
		}
		assertJournalUnchanged()
		var retry receipt
		modelSettingsDecode(t, f.rpcOK(t, "engine:apply-settings", request), &retry)
		if retry != wantReceipt {
			t.Fatal("identical retry did not return the reconstructed failed terminal receipt")
		}
		if !reflect.DeepEqual(before, f.policy(t)) {
			t.Fatal("identical retry changed policy before fresh supersession")
		}
		assertJournalUnchanged()
		f.mu.Lock()
		effectsUnchanged := maps.Equal(invocations, f.managerInvocations) && f.configureCalls == 0 && f.mutations == 0 && f.inferences == 1
		f.mu.Unlock()
		if !effectsUnchanged {
			t.Fatal("bare accepted recovery or identical retry attempted a new manager or engine effect")
		}
		f.engineIdentity(t, 1)
		return after
	}
	f.startModelSettings(t)
	firstRecovery := assertRecovery()
	f.crashBroker(t, gate)
	assertJournalUnchanged()
	f.startModelSettings(t)
	secondRecovery := assertRecovery()
	// Epoch and sequence describe each broker incarnation, not its outcome.
	firstRecovery.Epoch, secondRecovery.Epoch = "", ""
	firstRecovery.Sequence, secondRecovery.Sequence = 0, 0
	if !reflect.DeepEqual(firstRecovery, secondRecovery) {
		t.Fatal("second recovery changed the reconstructed public snapshot beyond epoch and sequence")
	}
	fresh := f.request(t, "bare-accepted-repair", request.ModelChange.Policy)
	fresh.ModelChange.SupersedesRequestID = request.RequestID
	fresh, preview := f.review(t, fresh)
	f.rpcOK(t, "engine:apply-settings", fresh)
	settled := f.settled(t, fresh.RequestID, "succeeded")
	if !settled.Settings.Equal(preview.Settings) || !settled.Running {
		t.Fatal("fresh local supersession lost reviewed settings or unresolved saved-On obligation")
	}
	assertModelSettingsPolicyPatch(t, before, f.policy(t), "0.35")
}

func TestModelSettingsMissingContentPreviewHasNoEffects(t *testing.T) {
	f := newModelSettingsProcessFixture(t)
	baseline := f.metadataBaseline(t)
	request := f.request(t, "missing-copy-target", nil)
	request.ModelChange.ModelID = "missing-copy-model"
	var preview settings.Preview
	modelSettingsDecode(t, f.rpcOK(t, "engine:preview-settings", request), &preview)
	if len(preview.Errors) == 0 || preview.PreviewDigest != "" {
		t.Fatal("missing content enabled reviewed Apply")
	}
	f.noMetadataEffects(t, baseline)
}

func TestModelSettingsFailedReadbackRequiresLocalSupersession(t *testing.T) {
	f := newModelSettingsProcessFixture(t)
	before := f.policy(t)
	request, _ := f.review(t, f.request(t, "lost-readback", &modelconfig.PolicyPatch{Generation: map[string]*json.Number{"temperature": modelSettingsNumber("0.8")}}))
	f.mu.Lock()
	f.failReadback = true
	f.mu.Unlock()
	f.rpcOK(t, "engine:apply-settings", request)
	after := f.settled(t, request.RequestID, "applying")
	if after.UnresolvedError == "" || after.CoupledStage != "native-applying" || !reflect.DeepEqual(before, f.policy(t)) {
		t.Fatal("failed native readback published policy or released reservation")
	}
	fresh := f.request(t, "readback-reconcile", request.ModelChange.Policy)
	fresh.ModelChange.SupersedesRequestID = request.RequestID
	fresh, preview := f.review(t, fresh)
	f.rpcOK(t, "engine:apply-settings", fresh)
	state := f.settled(t, fresh.RequestID, "succeeded")
	if !state.Settings.Equal(preview.Settings) {
		t.Fatal("supersession did not apply its fresh target preview")
	}
	assertModelSettingsPolicyPatch(t, before, f.policy(t), "0.8")
}

func TestModelSettingsConfigureResponseFailureUsesRealReadback(t *testing.T) {
	f := newModelSettingsProcessFixture(t)
	request, preview := f.review(t, f.request(t, "lost-configure-reply", nil))
	f.mu.Lock()
	f.failConfigureReply = true
	f.mu.Unlock()
	f.rpcOK(t, "engine:apply-settings", request)
	after := f.settled(t, request.RequestID, "succeeded")
	if after.EffectiveSettings == nil || !after.EffectiveSettings.Equal(preview.Settings) {
		t.Fatal("lost configure response was not reconciled from exact real-manager readback")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.configureCalls != 1 {
		t.Fatal("response loss retried the mutation")
	}
}

func TestModelSettingsManagerReplacementCannotCompleteFromOldSource(t *testing.T) {
	f := newModelSettingsProcessFixture(t)
	before := f.policy(t)
	request, _ := f.review(t, f.request(t, "manager-replaced", &modelconfig.PolicyPatch{Generation: map[string]*json.Number{"temperature": modelSettingsNumber("0.65")}}))
	gate := newModelSettingsGate()
	f.configureGate = gate
	applyID := f.begin(t, "engine:apply-settings", request)
	modelSettingsWait(t, gate)
	f.replaceManager(t)
	gate.unblock()
	f.finishApply(t, applyID)
	var disk modelSettingsDiskRecord
	task1Eventually(t, "source replacement failure receipt", 90*time.Second, func() bool { disk = f.disk(t); return disk.Receipts[request.RequestID].Phase == "failed" })
	if disk.Model.Stage == "completed" || !reflect.DeepEqual(before, f.policy(t)) {
		t.Fatal("obsolete manager source published coupled policy")
	}
	state := f.snapshot(t)
	if state.UnresolvedError == "" || state.Phase != "applying" {
		t.Fatal("replacement erased unresolved operation")
	}
}

func TestModelSettingsProgressedReceiptRecoversAcrossBrokerAndManagerRestart(t *testing.T) {
	f := newModelSettingsProcessFixture(t)
	f.seedModelSettingsPolicyBounds(t)
	before := f.policy(t)
	request, preview := f.review(t, f.request(t, "restart-receipt", &modelconfig.PolicyPatch{Generation: map[string]*json.Number{"temperature": modelSettingsNumber("0.55")}}))
	gate := newModelSettingsGate()
	f.readbackGate = gate
	f.begin(t, "engine:apply-settings", request)
	modelSettingsWait(t, gate)
	if f.disk(t).Model.Stage != "native-applying" {
		t.Fatal("recovery fixture did not reach a progressed durable stage")
	}
	// The real manager has persisted native state, but its readback is held
	// before the broker can publish policy. Kill the broker at that barrier.
	f.crashBroker(t, gate)
	f.startModelSettings(t)
	after := f.settled(t, request.RequestID, "succeeded")
	assertModelSettingsSavedReadback(t, after, preview.Settings)
	assertModelSettingsFinite(t, after.Settings, f.root)
	assertModelSettingsPolicyPatch(t, before, f.policy(t), "0.55")
	var retry struct {
		Revision uint64 `json:"revision"`
		Phase    string `json:"phase"`
	}
	modelSettingsDecode(t, f.rpcOK(t, "engine:apply-settings", request), &retry)
	if retry.Revision != after.Revision || retry.Phase != "succeeded" {
		t.Fatal("restart lost request receipt")
	}
}

func TestModelSettingsPinnedFiniteApplyPreservesTargetState(t *testing.T) {
	f := prepareModelSettingsProcessFixture(t)
	f.pinned(t)
	f.startModelSettings(t)
	before := f.policy(t)
	request := f.request(t, "pinned-manual", &modelconfig.PolicyPatch{Generation: map[string]*json.Number{"temperature": modelSettingsNumber("0.45")}})
	var preview settings.Preview
	modelSettingsDecode(t, f.peer(t, "preview", request, http.StatusOK), &preview)
	if len(preview.Errors) != 0 || preview.PreviewDigest == "" {
		t.Fatal("pinned finite preview unavailable")
	}
	assertModelSettingsFinite(t, preview.Settings, f.root)
	request.ModelChange.PreviewDigest = preview.PreviewDigest
	f.peer(t, "apply", request, http.StatusOK)
	after := f.settled(t, request.RequestID, "succeeded")
	var remote settings.Snapshot
	modelSettingsDecode(t, f.peer(t, "get", settings.Request{Engine: "llamacpp"}, http.StatusOK), &remote)
	if remote.NodeID != "model-settings-target" || !remote.Settings.Equal(after.Settings) || !remote.Settings.Equal(preview.Settings) || remote.Running {
		t.Fatal("pinned target readback differs from reviewed persisted configuration")
	}
	assertModelSettingsPolicyPatch(t, before, f.policy(t), "0.45")
}

func TestModelSettingsPinnedInferenceLockProtectsGenerationOnly(t *testing.T) {
	f := prepareModelSettingsProcessFixture(t)
	f.pinned(t)
	f.startModelSettings(t)
	policy := f.policy(t)
	policy.Profiles[0].LocalOnly = true
	policy = f.setPolicy(t, policy)
	request := f.request(t, "locked-generation", &modelconfig.PolicyPatch{Generation: map[string]*json.Number{"temperature": modelSettingsNumber("0.8")}})
	var failure struct {
		Code string `json:"code"`
	}
	modelSettingsDecode(t, f.peer(t, "preview", request, http.StatusConflict), &failure)
	if failure.Code != "remote_setting_locked" {
		t.Fatalf("generation lock refusal=%s", failure.Code)
	}
	if !reflect.DeepEqual(policy, f.policy(t)) {
		t.Fatal("locked generation preview changed target")
	}
	request = f.request(t, "locked-load-only", nil)
	var preview settings.Preview
	modelSettingsDecode(t, f.peer(t, "preview", request, http.StatusOK), &preview)
	if preview.PreviewDigest == "" || len(preview.Errors) != 0 {
		t.Fatal("generation lock incorrectly prevented load-only edit")
	}
	request.ModelChange.PreviewDigest = preview.PreviewDigest
	f.peer(t, "apply", request, http.StatusOK)
	after := f.settled(t, request.RequestID, "succeeded")
	assertModelSettingsSavedReadback(t, after, preview.Settings)
	var remote settings.Snapshot
	modelSettingsDecode(t, f.peer(t, "get", settings.Request{Engine: "llamacpp"}, http.StatusOK), &remote)
	assertModelSettingsSavedReadback(t, remote, preview.Settings)
	if remote.NodeID != "model-settings-target" || remote.Revision != after.Revision {
		t.Fatal("locked load-only readback did not identify the reviewed target revision")
	}
	if !reflect.DeepEqual(policy, f.policy(t)) {
		t.Fatal("load-only copy changed saved target lock/generation or unrelated policy")
	}
}

func TestModelSettingsDeniedGrantCannotUseReviewedApply(t *testing.T) {
	f := prepareModelSettingsProcessFixture(t)
	f.pinned(t)
	f.startModelSettings(t)
	request := f.request(t, "revoked-reviewed", nil)
	var preview settings.Preview
	modelSettingsDecode(t, f.peer(t, "preview", request, http.StatusOK), &preview)
	if preview.PreviewDigest == "" {
		t.Fatal("missing pre-revocation review")
	}
	request.ModelChange.PreviewDigest = preview.PreviewDigest
	baseline := f.metadataBaseline(t)
	cm := startTask1Process(t, f.base, clusterMgrBin, "--config-dir", f.base, "--port", fmt.Sprint(freePort(t)))
	managementSet(t, cm, "model-settings-caller", false)
	var failure struct {
		Code string `json:"code"`
	}
	modelSettingsDecode(t, f.peer(t, "apply", request, http.StatusConflict), &failure)
	if failure.Code != "remote_management_denied" {
		t.Fatalf("denied Apply code=%s", failure.Code)
	}
	f.noMetadataEffects(t, baseline)
	var inspection modelconfig.InspectResult
	modelSettingsDecode(t, f.peer(t, "inspect", modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "medium-model"}, http.StatusOK), &inspection)
	if inspection.Status != "resolved" {
		t.Fatal("revoked management grant disabled read-only metadata")
	}
}

func TestModelSettingsSelectedPreferenceCopyUsesTargetBindings(t *testing.T) {
	source := newModelSettingsProcessFixture(t)
	request, _ := source.review(t, source.request(t, "copy-source", nil))
	source.rpcOK(t, "engine:apply-settings", request)
	saved := source.settled(t, request.RequestID, "succeeded")
	// Stop this node before reusing shared fixed scanner/manager image paths.
	source.stopBroker(t)
	target := newModelSettingsProcessFixture(t)
	baselineRequest := target.request(t, "copy-target-baseline", nil)
	baselineRequest.ModelChange.Load.Set.Context.Tokens = 4096
	baselineRequest.ModelChange.Load.Set.Slots = modelSettingsInt(1)
	baselineRequest.ModelChange.Load.Set.Threads = modelSettingsInt(4)
	baselineRequest, _ = target.review(t, baselineRequest)
	target.rpcOK(t, "engine:apply-settings", baselineRequest)
	baseline := target.settled(t, baselineRequest.RequestID, "succeeded")
	before := target.policy(t)
	request = target.request(t, "selected-target", &modelconfig.PolicyPatch{Generation: map[string]*json.Number{"temperature": modelSettingsNumber("0.7")}})
	sourceIndex := slices.IndexFunc(saved.Settings.ModelLoads, func(m modelconfig.ModelLoad) bool { return m.ModelID == "medium-model" })
	if sourceIndex < 0 {
		t.Fatal("source preference was not persisted")
	}
	from := saved.Settings.ModelLoads[sourceIndex]
	// Copy selected portable requested preferences; the target supplies every
	// binding, native digest and unselected model's settings itself.
	request.ModelChange.Load = &modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Context: from.Requested.Context, Slots: from.Requested.Slots}}
	request, preview := target.review(t, request)
	assertModelSettingsFinite(t, preview.Settings, target.root)
	for _, load := range preview.Settings.ModelLoads {
		for _, artifact := range load.Resolved.Artifacts {
			if strings.HasPrefix(artifact.Path, source.root) {
				t.Fatal("copy imported source-local binding")
			}
		}
	}
	target.rpcOK(t, "engine:apply-settings", request)
	after := target.settled(t, request.RequestID, "succeeded")
	if !after.Settings.Equal(preview.Settings) {
		t.Fatal("target copy readback does not match selected target preview")
	}
	assertModelSettingsPolicyPatch(t, before, target.policy(t), "0.7")
	want := baseline.Settings.Clone()
	for i := range want.ModelLoads {
		load := &want.ModelLoads[i]
		index := slices.IndexFunc(after.Settings.ModelLoads, func(m modelconfig.ModelLoad) bool { return m.ModelID == load.ModelID })
		if index < 0 || load.Resolved == nil || after.Settings.ModelLoads[index].Resolved == nil {
			t.Fatal("selected copy lost a target model")
		}
		if load.ModelID == "medium-model" {
			load.Requested.Context, load.Requested.Slots = from.Requested.Context, from.Requested.Slots
			load.Resolved.PerSlotContext, load.Resolved.TotalContext, load.Resolved.Slots = 8192, 16384, 2
			load.Resolved.Options.Context, load.Resolved.Options.Slots = from.Requested.Context, from.Requested.Slots
		}
		// These identify the complete regenerated router. All other target
		// resolution fields and portable preferences must remain unchanged.
		load.Resolved.SourceInputDigest = after.Settings.ModelLoads[index].Resolved.SourceInputDigest
		load.Resolved.PresetDigest = after.Settings.ModelLoads[index].Resolved.PresetDigest
	}
	if !want.Equal(after.Settings) {
		t.Fatal("selected preference copy changed unselected target settings or bindings")
	}
}

func TestModelSettingsExplicitRoutingSelectionPreservesOtherPolicy(t *testing.T) {
	f := newModelSettingsProcessFixture(t)
	before := f.policy(t)
	before.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: "weak-model"}}
	before = f.setPolicy(t, before)
	yes := true
	request := f.request(t, "explicit-routing", &modelconfig.PolicyPatch{Roles: map[string]bool{"weak": true}, Binding: &yes, Served: &yes})
	request.ModelChange.Load = nil
	request, preview := f.review(t, request)
	if preview.Impact == nil || preview.Impact.RestartScope != "none" {
		t.Fatal("policy-only target preview proposed native restart")
	}
	f.rpcOK(t, "engine:apply-settings", request)
	f.settled(t, request.RequestID, "succeeded")
	after := f.policy(t)
	want := before
	want.Revision = after.Revision
	want.Roles = map[string]nodepolicy.Role{"weak": {Engine: "llamacpp", Model: "medium-model"}}
	want.EngineBindings = nodepolicy.EngineBindings{{Engine: "llamacpp", Model: "medium-model"}}
	want.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: "weak-model"}, {Engine: "llamacpp", Model: "medium-model"}}
	if !reflect.DeepEqual(want, after) {
		t.Fatal("explicit role/binding/served selection changed unselected policy or dropped existing served membership")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.configureCalls != 0 {
		t.Fatal("routing-only change configured native engine")
	}
}

// This uses the existing content owner and its pinned three-node fixture.
// Tiny content is copied first; a fresh inspector observes the durable receipt.
func TestModelSettingsMissingContentCopyThenFreshReceiptInspection(t *testing.T) {
	isolateModelSettingsPresetEnv(t)
	f := newControllerCopyFixture(t)
	f.directoryToManagers(t, false)
	model := f.seedSized(t, "settings-tiny-copy", 13, 4096)
	for i := range model.manifest.Files {
		artifact := &model.manifest.Files[i]
		data := modelSettingsGGUF(fmt.Sprintf("blk.%d.weight", i))
		modelSettingsWrite(t, model.paths[artifact.Path], data)
		model.data[artifact.Path] = data
		artifact.Size, artifact.SHA256 = int64(len(data)), phase04SHA(data)
	}
	model.manifest.ExportID = phase04SHA(mustMarshal(t, struct {
		Version   int                      `json:"version"`
		Model     string                   `json:"model"`
		Artifacts []controllerCopyArtifact `json:"artifacts"`
	}{1, model.manifest.Model, model.manifest.Files}))
	inspect := func() modelconfig.InspectResult {
		t.Helper()
		response := f.targetEM.call(t, "engine:inspect-model", settings.WorkerRequest[modelconfig.InspectRequest]{Request: modelconfig.InspectRequest{Engine: "llamacpp", ModelID: model.manifest.Model}})
		if response.Error != nil {
			t.Fatal("private target inspection failed")
		}
		var result modelconfig.InspectResult
		modelSettingsDecode(t, response.Result, &result)
		return result
	}
	missing := inspect()
	if missing.Status == "resolved" || len(missing.Artifacts) != 0 {
		t.Fatal("missing copy target reported content")
	}
	request := f.request(model.manifest.Model, "")
	id := f.broker.send(t, "engine:copy-model", request, true)
	f.assertCopiedArtifacts(t, model, task2Response(t, f.broker, id))
	controllerCopyTerminal(t, f.broker, request, "done")
	f.idle(t)
	fresh := inspect()
	if fresh.Status != "resolved" || len(fresh.Artifacts) != 2 || fresh.Artifacts[0].Role != "model" || fresh.Artifacts[1].Role != "draft" || fresh.Facts == nil || fresh.Facts.Architecture != "llama" {
		t.Fatalf("fresh receipt inspection=%+v", fresh)
	}
	assertModelSettingsInspectionProvenance(t, fresh)
	for i, artifact := range fresh.Artifacts {
		file := model.manifest.Files[i]
		relative := filepath.Join(".pair-copies", phase04SHA([]byte(model.manifest.Model)), model.manifest.ExportID, filepath.FromSlash(file.Path))
		binding := modelSettingsExpectedBinding(t, f.stores[f.target.uuid], file.Role, relative, model.data[file.Path])
		assertModelSettingsInspectionBinding(t, artifact, binding)
	}
	// No configuration approval or settings journal is inferred from COPY.
	var state struct {
		Operations []json.RawMessage `json:"operations"`
	}
	modelSettingsDecode(t, f.broker.call(t, "engine:active-model-copies", struct{}{}).Result, &state)
	if len(state.Operations) != 0 {
		t.Fatal("content owner retained an active operation")
	}
	// Run the reused fixture's controller, egress, Off and durable-lock guards
	// while its original managers are still available to answer them.
	f.assertController(t)
	// Stop the content fixture's manager/scanner images before reusing those
	// exact registered paths. Keep its private receipt and artifacts in place.
	f.broker.stop()
	f.sourceEM.stop()
	f.targetEM.stop()
	target := prepareModelSettingsProcessFixture(t)
	target.useCopiedStore(t, f.stores[f.target.uuid])
	target.startModelSettings(t)
	baseline := target.metadataBaseline(t)
	before, policy := baseline.snapshot, baseline.policy
	change := target.request(t, "configure-copied-content", nil)
	change.ModelChange.ModelID = model.manifest.Model
	change, preview := target.review(t, change)
	index := slices.IndexFunc(preview.Settings.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ModelID == model.manifest.Model })
	if index < 0 || preview.Settings.ModelLoads[index].Resolved == nil {
		t.Fatal("fresh target preview did not resolve copied receipt")
	}
	resolved := preview.Settings.ModelLoads[index].Resolved
	wantArtifacts := make([]modelconfig.ArtifactBinding, 0, len(model.manifest.Files))
	for _, artifact := range model.manifest.Files {
		relative := filepath.Join(".pair-copies", phase04SHA([]byte(model.manifest.Model)), model.manifest.ExportID, filepath.FromSlash(artifact.Path))
		wantArtifacts = append(wantArtifacts, modelSettingsExpectedBinding(t, target.root, artifact.Role, relative, model.data[artifact.Path]))
	}
	assertModelSettingsBindings(t, target.root, resolved.Artifacts, wantArtifacts)
	target.noMetadataEffects(t, baseline)
	target.rpcOK(t, "engine:apply-settings", change)
	after := target.settled(t, change.RequestID, "succeeded")
	if !after.Settings.Equal(preview.Settings) || after.Running || !reflect.DeepEqual(policy, target.policy(t)) {
		t.Fatal("explicit copied-content Apply changed unselected target state")
	}
	if after.Settings.ServerPort != before.Settings.ServerPort || after.Settings.ProxyPort != before.Settings.ProxyPort || after.Settings.Serving != nil {
		t.Fatal("copied-content Apply changed ports or created a serving declaration")
	}
	assertModelSettingsUnselectedLoad(t, after.Settings, "weak-model", 4096, 1)
	assertModelSettingsUnselectedLoad(t, after.Settings, "medium-model", 4096, 1)
}

func assertModelSettingsImpact(t *testing.T, preview settings.Preview, running bool) {
	t.Helper()
	impact := preview.Impact
	if impact == nil || impact.MayInterruptActive == nil || *impact.MayInterruptActive != running || impact.StaysOff == nil || *impact.StaysOff == running || !slices.Equal(impact.AffectedModels, []string{"medium-model", "weak-model"}) {
		t.Fatalf("reviewed whole-router impact=%+v", impact)
	}
	if running {
		if !preview.Restart || impact.RestartScope != "whole-router" || impact.DrainTimeoutSeconds == nil || *impact.DrainTimeoutSeconds != 60 || impact.ActiveWorkPolicy != "drain-then-restart" {
			t.Fatalf("running restart impact=%+v", impact)
		}
	} else if preview.Restart || impact.RestartScope != "none" || impact.DrainTimeoutSeconds != nil || impact.ActiveWorkPolicy != "none" {
		t.Fatalf("Off impact=%+v", impact)
	}
}

func assertModelSettingsFinite(t *testing.T, config settings.Config, root string) {
	t.Helper()
	if config.Serving != nil || len(config.ModelLoads) != 2 {
		t.Fatal("finite configuration lost complete router or manufactured serving declarations")
	}
	assertModelSettingsUnselectedLoad(t, config, "weak-model", 4096, 1)
	for _, load := range config.ModelLoads {
		r := load.Resolved
		if r == nil || len(r.Artifacts) != 1 || r.NativeBuildID == "" || len(r.SourceInputDigest) != 64 || len(r.PresetDigest) != 64 || r.ResolverRevision != "external-conversion/llama" || len(r.ReviewedInputID) != 64 {
			t.Fatal("finite preview lost frozen native/conversion provenance")
		}
		if load.ModelID == "medium-model" && (r.PerSlotContext != 8192 || r.TotalContext != 16384 || r.Slots != 2) {
			t.Fatal("finite selected context/slot allocation differs")
		}
		if r.GPU.Mode != "cpu" {
			t.Fatal("manual CPU choice changed")
		}
		var want modelconfig.ArtifactBinding
		switch load.ModelID {
		case "medium-model":
			want = modelSettingsExpectedBinding(t, root, "model", "medium.gguf", modelSettingsGGUF("blk.0.weight"))
		case "weak-model":
			want = modelSettingsExpectedBinding(t, root, "model", "weak.gguf", modelSettingsGGUF("blk.1.weight"))
		default:
			t.Fatal("finite preview contains an unexpected model")
		}
		assertModelSettingsBindings(t, root, r.Artifacts, []modelconfig.ArtifactBinding{want})
	}
}

func assertModelSettingsInspectionProvenance(t *testing.T, result modelconfig.InspectResult) {
	t.Helper()
	if result.Facts == nil || result.Facts.Source != "target-local configured preset/receipt/typed binding" {
		t.Fatal("inspection lost its configured artifact source")
	}
	for _, artifact := range result.Artifacts {
		if artifact.Assurance != "local head/tail samples and file identity; not full content verification" {
			t.Fatal("inspection overstated or lost sampled local binding assurance")
		}
	}
}

func assertModelSettingsInspectionBinding(t *testing.T, got modelconfig.ArtifactSummary, want modelconfig.ArtifactBinding) {
	t.Helper()
	size, err := strconv.ParseInt(want.Size, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	mtime, err := strconv.ParseInt(want.MtimeNS, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	// Inspection's opaque identity binds this exact configured/resolved file,
	// not just its role or size. Keep the wire digest's field order explicit.
	type fileBinding struct {
		Configured, Resolved, Sample string
		Size, Mtime                  int64
	}
	encoded, err := json.Marshal(struct {
		Source int
		Files  []fileBinding
	}{gguffacts.ReaderRevision, []fileBinding{{want.Path, want.ResolvedTarget, want.SampleSHA256, size, mtime}}})
	if err != nil {
		t.Fatal(err)
	}
	if got.Role != want.Role || got.Size != want.Size || got.Identity != phase04SHA(encoded) {
		t.Fatal("inspection identity differs from the expected ordered private file")
	}
}

func modelSettingsExpectedBinding(t *testing.T, root, role, relative string, data []byte) modelconfig.ArtifactBinding {
	t.Helper()
	// The tiny known fixtures fit wholly inside one binding sample. This
	// expectation does not upgrade the inspector's sampled assurance claim.
	if len(data) == 0 || len(data) > 64*1024 {
		t.Fatal("expected binding requires tiny known fixture content")
	}
	path := filepath.Join(root, relative)
	target, err := filepath.EvalSymlinks(path)
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if !info.Mode().IsRegular() || info.Size() != int64(len(data)) {
		t.Fatal("private binding fixture has unexpected type or size")
	}
	return modelconfig.ArtifactBinding{Role: role, Path: path, ResolvedTarget: target,
		Size: strconv.Itoa(len(data)), MtimeNS: strconv.FormatInt(info.ModTime().UnixNano(), 10), SampleSHA256: phase04SHA(data)}
}

func assertModelSettingsBindings(t *testing.T, root string, got, want []modelconfig.ArtifactBinding) {
	t.Helper()
	if !slices.Equal(got, want) {
		t.Fatal("ordered artifact bindings differ from the exact private files and roles")
	}
	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, artifact := range got {
		for _, binding := range []struct{ root, path string }{{root, artifact.Path}, {resolvedRoot, artifact.ResolvedTarget}} {
			relative, err := filepath.Rel(binding.root, binding.path)
			if err != nil || filepath.IsAbs(relative) || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
				t.Fatal("artifact source or resolved target escapes its private model root")
			}
		}
	}
}

func assertModelSettingsSavedReadback(t *testing.T, after settings.Snapshot, reviewed settings.Config) {
	t.Helper()
	if after.Phase != "succeeded" || after.CoupledStage != "completed" || after.Revision != after.AppliedRevision || after.Running || !after.Settings.Equal(reviewed) || after.EffectiveSettings == nil || !after.EffectiveSettings.Equal(reviewed) {
		t.Fatal("saved and effective Off settings differ from the exact reviewed configuration")
	}
}

func assertModelSettingsUnselectedLoad(t *testing.T, config settings.Config, model string, context, slots int32) {
	t.Helper()
	index := slices.IndexFunc(config.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ModelID == model })
	if index < 0 || config.ModelLoads[index].Resolved == nil {
		t.Fatal("unselected model disappeared")
	}
	r := config.ModelLoads[index].Resolved
	if r.PerSlotContext != context || r.TotalContext != context*slots || r.Slots != slots {
		t.Fatal("unselected model allocation changed")
	}
}

func assertModelSettingsPolicyPatch(t *testing.T, before, after nodepolicy.Policy, temperature string) {
	t.Helper()
	want := before
	want.Revision = after.Revision
	want.Profiles = slices.Clone(before.Profiles)
	var fields map[string]json.RawMessage
	modelSettingsDecode(t, want.Profiles[0].RequestOptions, &fields)
	fields["temperature"] = json.RawMessage(temperature)
	want.Profiles[0].RequestOptions = mustMarshal(t, fields)
	// Compare structured full policy, preserving opaque numbers lexically.
	if !bytes.Equal(mustMarshal(t, want), mustMarshal(t, after)) {
		t.Fatal("selected generation edit changed unselected target policy fields")
	}
}

func modelSettingsPreset(t *testing.T, data []byte) map[string]map[string]string {
	t.Helper()
	sections := map[string]map[string]string{}
	current := ""
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, ";") {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			current = strings.TrimSuffix(strings.TrimPrefix(line, "["), "]")
			sections[current] = map[string]string{}
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok || current == "" {
			t.Fatal("invalid captured native preset")
		}
		sections[current][strings.TrimSpace(key)] = strings.TrimSpace(value)
	}
	return sections
}

func (f *modelSettingsProcessFixture) holdModelSettingsInference(t *testing.T) <-chan error {
	t.Helper()
	// Advertise the facade, as discovery does, so the real proxy recognizes
	// itself and admits the request before forwarding to its local engine.
	f.rpcOK(t, "llamacpp-proxy:node/add-manual", map[string]any{"id": "settings-held", "host": "127.0.0.1", "port": f.targetFacadePort, "addresses": []string{"127.0.0.1"}, "models": []string{"medium-model"}})
	f.rpcOK(t, "llamacpp-proxy:node/select", map[string]string{"id": "settings-held"})
	// Engine start and the broker's healthy-backend publication are separate
	// observations. Wait for a real metadata read through the self route before
	// submitting the one held inference; do not forge backend health over IPC.
	client := &http.Client{Timeout: 2 * time.Second, Transport: &http.Transport{Proxy: nil}}
	defer client.CloseIdleConnections()
	task1Eventually(t, "healthy local model inventory through facade", 10*time.Second, func() bool {
		response, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/v1/models", f.targetFacadePort))
		if err != nil {
			return false
		}
		var inventory struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
		}
		decodeErr := json.NewDecoder(response.Body).Decode(&inventory)
		closeErr := response.Body.Close()
		if closeErr != nil {
			t.Fatal(closeErr)
		}
		if response.StatusCode != http.StatusOK || decodeErr != nil {
			return false
		}
		return slices.ContainsFunc(inventory.Data, func(model struct {
			ID string `json:"id"`
		}) bool {
			return model.ID == "medium-model"
		})
	})
	done := make(chan error, 1)
	go func() {
		response, err := (&http.Client{Timeout: 95 * time.Second}).Post(fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", f.targetFacadePort), "application/json", strings.NewReader(`{"model":"medium-model","stream":false,"messages":[{"role":"user","content":"synthetic"}]}`))
		if err == nil {
			_, readErr := io.Copy(io.Discard, response.Body)
			closeErr := response.Body.Close()
			if readErr != nil {
				err = readErr
			} else {
				err = closeErr
			}
		}
		done <- err
	}()
	modelSettingsWait(t, f.inference)
	return done
}
