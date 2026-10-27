// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"nvpair-shared/clustertrusttest"
	"nvpair-shared/durablefile"
	"nvpair-shared/engineintent"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

// All native/proxy work is served by the existing pipe fakes. No process,
// installed engine, model file, native helper or listening socket is used.
type modelSettingsHarness struct {
	*strictFixture
	mu               sync.Mutex
	launch           settings.LaunchState
	build            string
	configureCount   int
	configureFailure string
	locked           bool
	savedOff         bool
}

func newModelSettingsHarness(t *testing.T) *modelSettingsHarness {
	t.Helper()
	h := &modelSettingsHarness{strictFixture: newStrictFixture(t), build: "fixture-build"}
	h.launch = settings.LaunchState{Engine: "llamacpp", ServerPort: 18080, EffectivePort: 18080, LaunchText: "--fixture", Editable: true, Format: "pair-arguments-v1"}
	h.pp.readyMu.Lock()
	h.pp.facadeState = map[string]proxyFacadeState{"llamacpp": {ready: true, port: 18081}}
	h.pp.readyMu.Unlock()
	source := h.b.activateEngineManagerSource()
	h.b.bindEngineMgr(h.b.getEngineMgr(), source)
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.loaded["llamacpp"] = []string{}
	r.models.loadedSource, r.models.loadedProxy = source, h.pp
	r.mu.Unlock()
	h.engine.setAnswer(h.answer)
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodAdmissionSnapshot {
			return nodepolicy.AdmissionSnapshotResult{RunID: h.acks.runID, AdmissionState: nodepolicy.AdmissionState{ActiveByEngine: map[string]int{"llamacpp": 0}}}, ""
		}
		return h.acks.answer(method, raw)
	})
	return h
}

func (h *modelSettingsHarness) answer(method string, raw json.RawMessage) (any, string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	switch method {
	case "engine:get-launch":
		launch := h.launch
		launch.ModelLoads = modelconfig.CloneLoads(launch.ModelLoads)
		return launch, ""
	case "engine:preview-launch":
		var p settings.Request
		if err := json.Unmarshal(raw, &p); err != nil {
			return nil, err.Error()
		}
		config := p.Settings.Clone()
		if p.ModelChange != nil {
			if p.ModelChange.Policy != nil {
				return nil, "broker must merge the policy patch"
			}
			requested := p.ModelChange.Load.Set.Clone()
			per, slots := int32(8192), int32(1)
			if requested.Context != nil {
				per = requested.Context.Tokens
			}
			if requested.Slots != nil {
				slots = *requested.Slots
			}
			resolved := &modelconfig.ResolvedLoad{PerSlotContext: per, TotalContext: per * slots, Slots: slots, GPU: modelconfig.GPU{Mode: "cpu"}, NativeBuildID: h.build,
				SourceInputDigest: modelSettingsHash(requested), PresetDigest: strings.Repeat("a", 64),
				Artifacts: []modelconfig.ArtifactBinding{{Role: "model", Path: "fixture.gguf", ResolvedTarget: "fixture.gguf", Size: "256", MtimeNS: "1", SampleSHA256: strings.Repeat("b", 64)}}}
			load := modelconfig.ModelLoad{ModelID: p.ModelChange.ModelID, Requested: requested, Resolved: resolved}
			i := slices.IndexFunc(config.ModelLoads, func(m modelconfig.ModelLoad) bool { return m.ModelID == load.ModelID })
			if i < 0 {
				config.ModelLoads = append(config.ModelLoads, load)
			} else {
				config.ModelLoads[i] = load
			}
		} else {
			// Frozen revalidation reports actual source/build drift, never a new
			// portable resolution or Auto recommendation.
			for i := range config.ModelLoads {
				config.ModelLoads[i].Resolved.NativeBuildID = h.build
			}
		}
		return settings.Preview{Settings: config, PreviewDigest: modelSettingsHash(config)}, ""
	case "engine:configure-launch":
		var p settings.Configure
		if err := json.Unmarshal(raw, &p); err != nil {
			return nil, err.Error()
		}
		h.configureCount++
		if h.configureFailure == "no effect" {
			return nil, "injected configure failure"
		}
		h.launch.ServerPort, h.launch.LaunchText = p.Settings.ServerPort, p.Settings.LaunchText
		h.launch.ModelLoads, h.launch.Serving = modelconfig.CloneLoads(p.Settings.ModelLoads), p.Settings.Serving.Clone()
		h.launch.Running = p.Resume && !h.locked && !h.savedOff
		if h.configureFailure == "lost response" {
			return nil, "injected lost configure response"
		}
		return h.launch, ""
	case settings.SwitchPrepareMethod:
		return emptySwitchManagerSnapshot(), ""
	case "engine:get-installed":
		return map[string]any{"engines": []map[string]any{{"engine": "llamacpp", "running": h.launch.Running, "terminationVerified": !h.launch.Running, "locked": h.locked}}}, ""
	case engineIntentMethod:
		intent := engineintent.Snapshot{EnabledByEngine: map[string]bool{"llamacpp": !h.savedOff}, OffByEngine: map[string]engineintent.Origin{}}
		if h.savedOff {
			intent.OffByEngine["llamacpp"] = engineintent.Local
		}
		return intent, ""
	case "engine:models":
		return map[string]any{"loadedByEngine": map[string][]string{"llamacpp": {}}}, ""
	case engineSleepMethod:
		h.launch.Running = false
		return map[string]bool{"ok": true}, ""
	}
	return map[string]bool{"ok": true}, ""
}

func (h *modelSettingsHarness) request(t *testing.T, coupled bool) settings.Request {
	t.Helper()
	snapshot, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, "")
	if err != nil {
		t.Fatal(err)
	}
	p := settings.Request{Engine: "llamacpp", RequestID: settingsID(), ExpectedRevision: snapshot.Revision, Settings: snapshot.Settings,
		ModelChange: &modelconfig.ModelChange{ModelID: "fixture/model", ExpectedPolicyRevision: h.b.localPolicy().Policy.Revision, Load: &modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "fixed", Tokens: 8192}}}}}
	if coupled {
		number := json.Number("0.35")
		p.ModelChange.Policy = &modelconfig.PolicyPatch{Generation: map[string]*json.Number{"temperature": &number}}
	}
	return p
}

func (h *modelSettingsHarness) review(t *testing.T, p settings.Request, caller string) settings.Request {
	t.Helper()
	preview, err := h.b.previewEngineSettings(context.Background(), p, caller)
	if err != nil {
		t.Fatal(err)
	}
	if len(preview.Errors) != 0 || preview.Conflict != nil || preview.PreviewDigest == "" {
		t.Fatalf("preview = %+v", preview)
	}
	p.ModelChange.PreviewDigest = preview.PreviewDigest
	return p
}

func (h *modelSettingsHarness) record(t *testing.T) *engineSettingsRecord {
	t.Helper()
	h.b.engineConfigMu.Lock()
	defer h.b.engineConfigMu.Unlock()
	record := h.b.engineSettings["llamacpp"].clone()
	if record == nil {
		t.Fatal("missing settings record")
	}
	return record
}

func (h *modelSettingsHarness) calls() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.configureCount
}

func waitModelSettingsSignal(t *testing.T, signal <-chan struct{}) {
	t.Helper()
	select {
	case <-signal:
	case <-t.Context().Done():
		t.Fatal("model settings barrier cancelled")
	}
}

func TestModelSettingsApplyCoupledOffPersistsBothComponents(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("apply = %+v, %v", receipt, err)
	}
	record := h.record(t)
	if record.Model.Stage != "completed" || record.Snapshot.Running || record.Snapshot.EffectiveSettings == nil || !record.Snapshot.EffectiveSettings.Equal(record.Snapshot.Settings) {
		t.Fatalf("record = %+v", record)
	}
	policy := h.b.localPolicy().Policy
	if len(policy.Profiles) != 1 || policy.Profiles[0].Qualification != nil || policy.Profiles[0].MemoryBytes != 0 {
		t.Fatalf("unexpected profile evidence: %+v", policy.Profiles)
	}
	var options map[string]json.Number
	if err := json.Unmarshal(policy.Profiles[0].RequestOptions, &options); err != nil {
		t.Fatal(err)
	}
	if options["temperature"] != "0.35" {
		t.Fatalf("options = %+v", options)
	}
	if h.calls() != 1 || policy.Revision != record.Snapshot.PolicyRevision {
		t.Fatalf("calls=%d, policy revision=%s", h.calls(), policy.Revision)
	}
	again, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || again != receipt || h.calls() != 1 {
		t.Fatalf("retry = %+v, %v", again, err)
	}
}

func TestModelSettingsApplyPolicyOnlyDoesNotConfigureOrDrain(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.request(t, true)
	p.ModelChange.Load = nil
	p = h.review(t, p, "")
	before := len(h.proxy.snapshot())
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "succeeded" || h.calls() != 0 {
		t.Fatalf("apply = %+v, %v, calls=%d", receipt, err, h.calls())
	}
	for _, call := range h.proxy.snapshot()[before:] {
		if call.method == nodepolicy.MethodSetEngineDrain {
			t.Fatal("policy-only edit drained an engine")
		}
	}
}

func TestModelSettingsApplyExactReviewAndRequestIdentity(t *testing.T) {
	test := func(name string, mutate func(*settings.Request), code int) {
		t.Run(name, func(t *testing.T) {
			h := newModelSettingsHarness(t)
			p := h.review(t, h.request(t, true), "")
			mutate(&p)
			_, err := h.b.applyEngineSettings(context.Background(), p, "")
			if err == nil || engineSettingsErrorCode(err) != code || h.calls() != 0 {
				t.Fatalf("refusal = %v, calls=%d", err, h.calls())
			}
		})
	}
	test("stale settings CAS", func(p *settings.Request) { p.ExpectedRevision++ }, codePolicyConflict)
	test("stale policy CAS", func(p *settings.Request) { p.ModelChange.ExpectedPolicyRevision = "1" }, codePolicyConflict)
	test("changed selected field", func(p *settings.Request) { p.ModelChange.Load.Set.Context.Tokens = 4096 }, codePolicyConflict)
	test("forged review", func(p *settings.Request) { p.ModelChange.PreviewDigest = strings.Repeat("0", 64) }, codePolicyConflict)
	test("caller supplied resolution", func(p *settings.Request) { p.Settings.ModelLoads = []modelconfig.ModelLoad{{ModelID: "forged"}} }, -32602)
}

func TestModelSettingsApplyReusedIDRefusesDifferentContent(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("apply = %+v, %v", receipt, err)
	}
	p.ModelChange.Load.Set.Context.Tokens = 4096
	_, err := h.b.applyEngineSettings(context.Background(), p, "")
	if engineSettingsErrorCode(err) != -32602 || h.calls() != 1 {
		t.Fatalf("reused ID = %v, calls=%d", err, h.calls())
	}
}

func TestModelSettingsApplyFailedAcceptanceHasNoEffects(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	before := h.record(t)
	injected := errors.New("acceptance write failed")
	h.b.engineSettingsReplacer = durablefile.Replacer{CreateTemp: func(string, string, os.FileMode) (durablefile.File, error) { return nil, injected }}
	_, err := h.b.applyEngineSettings(context.Background(), p, "")
	if !errors.Is(err, injected) || h.calls() != 0 {
		t.Fatalf("acceptance = %v, calls=%d", err, h.calls())
	}
	after := h.record(t)
	if !reflect.DeepEqual(before, after) || len(h.b.localPolicy().Policy.Profiles) != 0 {
		t.Fatal("failed acceptance changed journal or policy")
	}
}

func TestModelSettingsApplyUncertainAcceptanceRetainsReservationWithoutEffects(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	injected := errors.New("acceptance durability is uncertain")
	h.b.engineSettingsReplacer = durablefile.Replacer{SyncDir: func(string) error { return injected }}
	_, err := h.b.applyEngineSettings(context.Background(), p, "")
	if !errors.Is(err, durablefile.ErrReplaced) || h.calls() != 0 {
		t.Fatalf("uncertain acceptance = %v, calls=%d", err, h.calls())
	}
	policy := h.b.localPolicy().Policy
	if len(policy.Profiles) != 0 {
		t.Fatal("uncertain acceptance changed policy")
	}
	if _, err := h.b.setLocalPolicy(settingsJSON(policy), settingsJSON(policy.Revision)); policyErrorCode(err) != codePolicyConflict {
		t.Fatalf("uncertain acceptance lost reservation: %v", err)
	}
}

func TestModelSettingsApplyNativeBuildDriftRequiresFreshReview(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	h.mu.Lock()
	h.build = "new-build"
	h.mu.Unlock()
	_, err := h.b.applyEngineSettings(context.Background(), p, "")
	if engineSettingsErrorCode(err) != codePolicyConflict || h.calls() != 0 {
		t.Fatalf("changed native review = %v, calls=%d", err, h.calls())
	}
}

// Conversion provenance uses only the existing ResolvedLoad fields. Returning
// the same native preview token deliberately proves the broker also binds all
// nested settings, rather than trusting that token to cover each field.
func modelSettingsConversionAnswer(h *modelSettingsHarness, mutate func(*modelconfig.ResolvedLoad)) func(string, json.RawMessage) (any, string) {
	return func(method string, raw json.RawMessage) (any, string) {
		result, err := h.answer(method, raw)
		if method != "engine:preview-launch" || err != "" {
			return result, err
		}
		preview, ok := result.(settings.Preview)
		if !ok {
			return nil, "fixture expected a typed settings preview"
		}
		for i := range preview.Settings.ModelLoads {
			resolved := preview.Settings.ModelLoads[i].Resolved
			resolved.ResolverRevision = "external-conversion/llama"
			resolved.ReviewedInputID = strings.Repeat("c", 64)
			resolved.SourceInputDigest = strings.Repeat("d", 64)
			resolved.PresetDigest = strings.Repeat("e", 64)
			if mutate != nil {
				mutate(resolved)
			}
		}
		return preview, ""
	}
}

func TestModelSettingsApplyConversionProvenanceAndNativeDigestsAreFrozen(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.engine.setAnswer(modelSettingsConversionAnswer(h, nil))
	p := h.request(t, true)
	preview, err := h.b.previewEngineSettings(context.Background(), p, "")
	if err != nil || len(preview.Errors) != 0 {
		t.Fatalf("conversion preview = %+v, %v", preview, err)
	}
	frozen := preview.Settings.Clone()
	p.ModelChange.PreviewDigest = preview.PreviewDigest
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("conversion apply = %+v, %v", receipt, err)
	}
	record := h.record(t)
	if !record.Model.Review.Settings.Equal(frozen) || !record.Snapshot.Settings.Equal(frozen) || record.Snapshot.EffectiveSettings == nil || !record.Snapshot.EffectiveSettings.Equal(frozen) {
		t.Fatal("journal or native readback lost frozen conversion metadata")
	}
	found := false
	for _, call := range h.engine.snapshot() {
		if call.method == "engine:configure-launch" {
			var configure settings.Configure
			if err := json.Unmarshal(call.params, &configure); err != nil {
				t.Fatal(err)
			}
			if !configure.Settings.Equal(frozen) {
				t.Fatal("configure did not receive complete frozen settings")
			}
			resolved := configure.Settings.ModelLoads[0].Resolved
			if resolved.ResolverRevision != "external-conversion/llama" || resolved.ReviewedInputID != strings.Repeat("c", 64) || resolved.SourceInputDigest != strings.Repeat("d", 64) || resolved.PresetDigest != strings.Repeat("e", 64) {
				t.Fatalf("conversion identity = %+v", resolved)
			}
			found = true
		}
	}
	if !found {
		t.Fatal("conversion never reached configure")
	}
	path, err := h.b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var stored map[string]*engineSettingsRecord
	if err := json.Unmarshal(data, &stored); err != nil {
		t.Fatal(err)
	}
	if stored["llamacpp"] == nil || !stored["llamacpp"].Model.Review.Settings.Equal(frozen) || !stored["llamacpp"].Snapshot.Settings.Equal(frozen) {
		t.Fatal("durable conversion metadata differs from review")
	}
}

func TestModelSettingsApplyConversionProvenanceOrNativeDigestDriftRefuses(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*modelconfig.ResolvedLoad)
	}{
		{"resolver revision", func(r *modelconfig.ResolvedLoad) { r.ResolverRevision = "external-conversion/changed" }},
		{"reviewed external source", func(r *modelconfig.ResolvedLoad) { r.ReviewedInputID = strings.Repeat("f", 64) }},
		{"native source digest", func(r *modelconfig.ResolvedLoad) { r.SourceInputDigest = strings.Repeat("f", 64) }},
		{"preset digest", func(r *modelconfig.ResolvedLoad) { r.PresetDigest = strings.Repeat("f", 64) }},
		{"native build", func(r *modelconfig.ResolvedLoad) { r.NativeBuildID = "replacement-build" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newModelSettingsHarness(t)
			var changed atomic.Bool
			h.engine.setAnswer(modelSettingsConversionAnswer(h, func(r *modelconfig.ResolvedLoad) {
				if changed.Load() {
					tc.mutate(r)
				}
			}))
			p := h.review(t, h.request(t, true), "")
			changed.Store(true)
			_, err := h.b.applyEngineSettings(context.Background(), p, "")
			if engineSettingsErrorCode(err) != codePolicyConflict || h.calls() != 0 {
				t.Fatalf("conversion drift = %v, calls=%d", err, h.calls())
			}
		})
	}
}

func TestModelSettingsApplyPreviewRemainsReadableDuringPause(t *testing.T) {
	h := newModelSettingsHarness(t)
	peer := "peer"
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", h.b.nodeID, peer)
	p := h.request(t, false)
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.pendingPauseRequested = true
	r.mu.Unlock()
	p = h.review(t, p, peer)
	_, err := h.b.applyEngineSettings(context.Background(), p, peer)
	if engineSettingsErrorCode(err) != codeRemoteManagementDenied || h.calls() != 0 {
		t.Fatalf("preview granted mutation authority: %v", err)
	}
}

func TestModelSettingsApplyOrdinaryRawSettingsCannotReplacePendingOperation(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	h.mu.Lock()
	h.configureFailure = "no effect"
	h.mu.Unlock()
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || receipt.Phase != "failed" {
		t.Fatalf("seed failure = %+v, %v", receipt, err)
	}
	ordinary := settings.Request{Engine: p.Engine, RequestID: settingsID(), ExpectedRevision: h.record(t).Snapshot.Revision, Settings: h.record(t).Snapshot.Settings.Clone()}
	ordinary.Settings.LaunchText += " --raw-change"
	_, err := h.b.applyEngineSettings(context.Background(), ordinary, "")
	if engineSettingsErrorCode(err) != codePolicyConflict || h.calls() != 1 {
		t.Fatalf("ordinary bypass = %v, calls=%d", err, h.calls())
	}
}

func TestModelSettingsApplyMergePreservesOpaqueNumbersAndUnselectedLeaves(t *testing.T) {
	h := newModelSettingsHarness(t)
	policy := h.b.localPolicy().Policy
	policy.Profiles = []nodepolicy.Profile{{Name: "profile", Engine: "llamacpp", Model: "fixture/model", LocalOnly: true, RequestOptions: json.RawMessage(`{"temperature":0.1,"opaque":{"integer":9007199254740993,"fraction":0.123456789012345678901}}`)}}
	number := json.Number("0.25")
	change := modelconfig.ModelChange{ModelID: "fixture/model", Policy: &modelconfig.PolicyPatch{Generation: map[string]*json.Number{"temperature": &number}}}
	merged, err := h.b.mergeModelPolicy(policy, "llamacpp", change, "")
	if err != nil {
		t.Fatal(err)
	}
	var options struct {
		Temperature json.Number            `json:"temperature"`
		Opaque      map[string]json.Number `json:"opaque"`
	}
	if err := json.Unmarshal(merged.Profiles[0].RequestOptions, &options); err != nil {
		t.Fatal(err)
	}
	if options.Temperature != "0.25" || options.Opaque["integer"] != "9007199254740993" || options.Opaque["fraction"] != "0.123456789012345678901" || !merged.Profiles[0].LocalOnly || merged.ServedModels != nil {
		t.Fatalf("selected merge = %+v", merged.Profiles[0])
	}
}

func TestModelSettingsPreviewWorkerWaitDoesNotHoldOwnerLocks(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.request(t, true)
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
	done := make(chan error, 1)
	go func() { _, err := h.b.previewEngineSettings(context.Background(), p, ""); done <- err }()
	waitModelSettingsSignal(t, reached)
	// This write requires both engineConfigMu and policyRuntime.mu and sends
	// projection I/O to the fake while the preview response is still held.
	if _, err := h.b.setLocalPolicy(settingsJSON(h.b.localPolicy().Policy), settingsJSON(p.ModelChange.ExpectedPolicyRevision)); err != nil {
		t.Fatal(err)
	}
	h.engine.releaseHeld()
	if err := <-done; engineSettingsErrorCode(err) != codePolicyConflict {
		t.Fatalf("stale preview = %v", err)
	}
	if h.calls() != 0 {
		t.Fatal("preview configured an engine")
	}
}

func TestModelSettingsApplyCoupledRawPolicyWriterFence(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
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
	_, err := h.b.setLocalPolicy(settingsJSON(policy), settingsJSON(policy.Revision))
	if policyErrorCode(err) != codePolicyConflict {
		t.Fatalf("raw local write = %v", err)
	}
	_, err = h.b.setPeerPolicy(settingsJSON(policy.PeerProjection()), settingsJSON(policy.Revision))
	if policyErrorCode(err) != codePolicyConflict {
		t.Fatalf("raw peer write = %v", err)
	}
	if record := h.record(t); record.Model.Stage != "native-applying" {
		t.Fatalf("stage = %s", record.Model.Stage)
	}
	h.engine.releaseHeld()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestModelSettingsApplyLoadOnlyAllowsUnrelatedPolicyWriter(t *testing.T) {
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
	policy.Idle.UnloadAfterMinutes = 17
	if _, err := h.b.setLocalPolicy(settingsJSON(policy), settingsJSON(policy.Revision)); err != nil {
		t.Fatal(err)
	}
	h.engine.releaseHeld()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if record := h.record(t); record.Snapshot.Phase != "succeeded" || h.b.localPolicy().Policy.Idle.UnloadAfterMinutes != 17 {
		t.Fatalf("record=%+v", record.Snapshot)
	}
}

func TestModelSettingsApplyDrainAckFailureKeepsFence(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetEngineDrain {
			return map[string]bool{"ok": false}, ""
		}
		return h.acks.answer(method, raw)
	})
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "failed" || h.calls() != 0 {
		t.Fatalf("apply=%+v, %v, calls=%d", receipt, err, h.calls())
	}
	h.b.engineConfigMu.Lock()
	withdrawn := h.b.advertisementWithdrawnLocked("llamacpp")
	h.b.engineConfigMu.Unlock()
	if !withdrawn || h.record(t).Snapshot.UnresolvedError == "" {
		t.Fatal("failed ACK released the operation fence")
	}
}

func TestModelSettingsApplyAcknowledgedActiveDrainMayTimeOut(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.mu.Lock()
	h.launch.Running = true
	h.mu.Unlock()
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.intent["llamacpp"] = true
	r.mu.Unlock()
	p := h.request(t, true)
	preview, err := h.b.previewEngineSettings(context.Background(), p, "")
	if err != nil {
		t.Fatal(err)
	}
	if preview.Impact.RestartScope != "whole-router" || preview.Impact.DrainTimeoutSeconds == nil || *preview.Impact.DrainTimeoutSeconds != 60 || !*preview.Impact.MayInterruptActive {
		t.Fatalf("impact=%+v", preview.Impact)
	}
	p.ModelChange.PreviewDigest = preview.PreviewDigest
	before := settingsDrainTimeout
	settingsDrainTimeout = 0
	t.Cleanup(func() { settingsDrainTimeout = before })
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodAdmissionSnapshot {
			return nodepolicy.AdmissionSnapshotResult{RunID: h.acks.runID, AdmissionState: nodepolicy.AdmissionState{Active: 1, ActiveByEngine: map[string]int{"llamacpp": 1}}}, ""
		}
		return h.acks.answer(method, raw)
	})
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "succeeded" || h.calls() != 1 {
		t.Fatalf("apply=%+v, %v", receipt, err)
	}
}

func TestModelSettingsApplyLostNativeResponseUsesExactReadback(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	h.mu.Lock()
	h.configureFailure = "lost response"
	h.mu.Unlock()
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "succeeded" || h.calls() != 1 {
		t.Fatalf("readback reconciliation=%+v, %v", receipt, err)
	}
}

func TestModelSettingsApplyGrantAndPausePrecedeReceiptRetry(t *testing.T) {
	h := newModelSettingsHarness(t)
	peerID := "peer"
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", h.b.nodeID, peerID)
	grantRemoteManagement(t, h.b.clusterDir, "cluster", h.b.nodeID, peerID, true)
	p := h.review(t, h.request(t, true), peerID)
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, peerID); err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("apply=%+v, %v", receipt, err)
	}
	grantRemoteManagement(t, h.b.clusterDir, "cluster", h.b.nodeID, peerID, false)
	_, err := h.b.applyEngineSettings(context.Background(), p, peerID)
	if engineSettingsErrorCode(err) != codeRemoteManagementDenied {
		t.Fatalf("revoked retry=%v", err)
	}
	grantRemoteManagement(t, h.b.clusterDir, "cluster", h.b.nodeID, peerID, true)
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.pendingPauseRequested = true
	r.mu.Unlock()
	_, err = h.b.applyEngineSettings(context.Background(), p, peerID)
	if engineSettingsErrorCode(err) != codeLocalVeto || h.calls() != 1 {
		t.Fatalf("paused retry=%v", err)
	}
}

func TestModelSettingsApplyInferenceLockProtectsGenerationOnly(t *testing.T) {
	h := newModelSettingsHarness(t)
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Profiles = []nodepolicy.Profile{{Name: "locked", Engine: "llamacpp", Model: "fixture/model", LocalOnly: true, RequestOptions: json.RawMessage(`{"temperature":0.2}`)}}
	r.mu.Unlock()
	peerID := "peer"
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", h.b.nodeID, peerID)
	grantRemoteManagement(t, h.b.clusterDir, "cluster", h.b.nodeID, peerID, true)
	p := h.request(t, true)
	if _, err := h.b.previewEngineSettings(context.Background(), p, peerID); engineSettingsErrorCode(err) != codeRemoteSettingLocked {
		t.Fatalf("generation lock=%v", err)
	}
	p.ModelChange.Policy = nil
	p = h.review(t, p, peerID)
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, peerID); err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("load-only through generation lock=%+v, %v", receipt, err)
	}
}

func TestModelSettingsApplyPolicyActivationRemainsPending(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetPolicy {
			return nil, "injected activation failure"
		}
		return h.acks.answer(method, raw)
	})
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "applying" {
		t.Fatalf("pending activation=%+v, %v", receipt, err)
	}
	record := h.record(t)
	if record.Model.Stage != "activation-pending" || record.Snapshot.UnresolvedError == "" {
		t.Fatalf("record=%+v", record)
	}
	// Repair through the public local fresh-preview/apply path, without a
	// manual startup recovery call or a second native effect.
	h.proxy.setAnswer(h.acks.answer)
	fresh := h.request(t, true)
	fresh.ModelChange.SupersedesRequestID = p.RequestID
	fresh = h.review(t, fresh, "")
	settled, err := h.b.applyEngineSettings(context.Background(), fresh, "")
	if err != nil || settled.Phase != "succeeded" || h.calls() != 1 || h.record(t).Model.Stage != "completed" {
		t.Fatalf("local activation repair = %+v, %v", settled, err)
	}
	if h.record(t).Receipts[p.RequestID].Phase != "superseded" {
		t.Fatal("activation repair did not retire the prior receipt")
	}
}

func TestModelSettingsApplyPublicationOrderWithoutConfigurationLockIO(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.request(t, false)
	worker := h.b.getEngineMgr()
	// Flush the fixture read pump before collecting the two new batches.
	if err := modelSettingsCall(t.Context(), worker, "engine:get-launch", map[string]string{"engine": "llamacpp"}, nil); err != nil {
		t.Fatal(err)
	}
	before := len(h.engine.snapshot())
	output := h.out
	clientBefore := len(output.frames())
	settingsDeliveries := func() int {
		count := 0
		for _, frame := range output.frames()[clientBefore:] {
			if frame.Method == "engine:settings-changed" {
				count++
			}
		}
		return count
	}
	predecessor := make(chan struct{})
	var release sync.Once
	open := func() { release.Do(func() { close(predecessor) }) }
	t.Cleanup(open)
	h.b.engineConfigMu.Lock()
	base := h.b.engineSettings["llamacpp"].Snapshot.Sequence
	h.b.settingsPublicationTail = predecessor
	h.b.engineConfigMu.Unlock()
	start := func() (<-chan struct{}, <-chan struct{}) {
		entered, done := make(chan struct{}), make(chan struct{})
		go func() {
			h.b.engineConfigMu.Lock()
			close(entered)
			h.b.publishSettingsLocked()
			h.b.engineConfigMu.Unlock()
			close(done)
		}()
		return entered, done
	}
	aEntered, aDone := start()
	waitModelSettingsSignal(t, aEntered)
	h.b.engineConfigMu.Lock()
	aTail := h.b.settingsPublicationTail
	h.b.engineConfigMu.Unlock()
	bEntered, bDone := start()
	waitModelSettingsSignal(t, bEntered)
	h.b.engineConfigMu.Lock()
	bTail, sequence := h.b.settingsPublicationTail, h.b.engineSettings["llamacpp"].Snapshot.Sequence
	h.b.engineConfigMu.Unlock()
	if aTail == predecessor || bTail == aTail || sequence != base+2 || settingsDeliveries() != 0 {
		t.Fatal("publication wait held configuration lock, skipped ordering, or delivered early")
	}
	open()
	waitModelSettingsSignal(t, aDone)
	waitModelSettingsSignal(t, bDone)
	if err := modelSettingsCall(t.Context(), worker, "engine:get-launch", map[string]string{"engine": "llamacpp"}, nil); err != nil {
		t.Fatal(err)
	}
	var clientSequences, managerSequences []uint64
	for _, frame := range output.frames()[clientBefore:] {
		if frame.Method != "engine:settings-changed" {
			continue
		}
		var snapshot settings.Snapshot
		if err := json.Unmarshal(frame.Params, &snapshot); err != nil {
			t.Fatal(err)
		}
		if snapshot.Engine == "llamacpp" {
			clientSequences = append(clientSequences, snapshot.Sequence)
		}
	}
	for _, call := range h.engine.snapshot()[before:] {
		if call.method != "engine:settings-projection" {
			continue
		}
		var snapshots []settings.Snapshot
		if err := json.Unmarshal(call.params, &snapshots); err != nil {
			t.Fatal(err)
		}
		for _, snapshot := range snapshots {
			if snapshot.Engine == "llamacpp" {
				managerSequences = append(managerSequences, snapshot.Sequence)
			}
		}
	}
	want := []uint64{base + 1, base + 2}
	if !reflect.DeepEqual(clientSequences, want) || !reflect.DeepEqual(managerSequences, want) {
		t.Fatalf("client=%v manager=%v want=%v", clientSequences, managerSequences, want)
	}
}
