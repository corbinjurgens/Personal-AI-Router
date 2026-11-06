// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrusttest"
	"nvpair-shared/durablefile"
	"nvpair-shared/engineintent"
	"nvpair-shared/engines"
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
	savedOff         bool
	removalReviews   map[string]modelSettingsRemovalReview
}

type modelSettingsRemovalReview struct {
	Change   modelconfig.ModelChange
	Settings settings.Config
}

// Called under the harness mutex. The fake's original worker receipts are
// independent of the broker's public digest and retained/journal metadata.
func (h *modelSettingsHarness) validateRemoval(config settings.Config, reviewed *modelconfig.ModelChange) error {
	if reviewed == nil {
		for _, load := range h.launch.ModelLoads {
			if !slices.ContainsFunc(config.ModelLoads, func(next modelconfig.ModelLoad) bool { return next.ProfileKey == load.ProfileKey }) {
				return fmt.Errorf("frozen omission requires original native removal evidence")
			}
		}
		return nil
	}
	if err := settings.ValidateReviewedModelChange(reviewed); err != nil {
		return err
	}
	accepted, ok := h.removalReviews[reviewed.PreviewDigest]
	if !ok || !reflect.DeepEqual(*reviewed, accepted.Change) || !config.Equal(accepted.Settings) {
		return fmt.Errorf("original native removal review does not match frozen settings")
	}
	return nil
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
		envelope, err := settings.DecodeWorkerRequest[settings.Request](raw)
		if err != nil {
			return nil, err.Error()
		}
		p := envelope.Request
		config := p.Settings.Clone()
		if p.ModelChange != nil && envelope.ReviewedModelChange != nil {
			return nil, "reviewed removal must not reapply the portable patch"
		}
		if p.ModelChange != nil {
			if p.ModelChange.Policy != nil {
				return nil, "broker must merge the policy patch"
			}
			key := p.ModelChange.ProfileKey
			if key == "" {
				key = p.ModelChange.ModelID
			}
			i := slices.IndexFunc(config.ModelLoads, func(m modelconfig.ModelLoad) bool { return m.ProfileKey == key })
			if i >= 0 && config.ModelLoads[i].ModelID != p.ModelChange.ModelID {
				return nil, "A profile keeps its model; create a new profile."
			}
			if p.ModelChange.Remove {
				if i < 0 {
					return nil, "profile load not found"
				}
				config.ModelLoads = slices.Delete(config.ModelLoads, i, i+1)
				if config.Serving != nil {
					config.Serving.Models = slices.DeleteFunc(config.Serving.Models, func(model engines.ServingModel) bool { return model.ModelID == key })
					if len(config.ModelLoads) == 0 {
						config.Serving = nil
					}
				}
			} else {
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
				load := modelconfig.ModelLoad{ProfileKey: key, ModelID: p.ModelChange.ModelID, Requested: requested, Resolved: resolved}
				if i < 0 {
					config.ModelLoads = append(config.ModelLoads, load)
				} else {
					config.ModelLoads[i] = load
				}
			}
		} else {
			if err := h.validateRemoval(config, envelope.ReviewedModelChange); err != nil {
				return nil, err.Error()
			}
			// Frozen revalidation reports actual source/build drift, never a new
			// portable resolution or Auto recommendation.
			for i := range config.ModelLoads {
				config.ModelLoads[i].Resolved.NativeBuildID = h.build
			}
		}
		config, err = config.Canonical()
		if err != nil {
			return nil, err.Error()
		}
		identities := map[string]engines.ModelIdentity{}
		for _, load := range config.ModelLoads {
			identities[load.ProfileKey] = engines.ModelIdentity{GeneralName: "Fixture model", Architecture: "llama", ConfiguredContextTokens: int64(load.Resolved.PerSlotContext), Strength: "declared", Files: []engines.ModelIdentityFile{{Role: "model", Name: "fixture.gguf", SizeBytes: 256}}}
		}
		digest := modelSettingsHash(config)
		if p.ModelChange != nil && p.ModelChange.Remove {
			change := modelconfig.ModelChange{ProfileKey: p.ModelChange.ProfileKey, ModelID: p.ModelChange.ModelID, ExpectedPolicyRevision: p.ModelChange.ExpectedPolicyRevision, Remove: true}
			digest = modelSettingsHash(struct {
				Change        modelconfig.ModelChange
				Before, After settings.Config
			}{change, p.Settings, config})
			change.PreviewDigest = digest
			if h.removalReviews == nil {
				h.removalReviews = map[string]modelSettingsRemovalReview{}
			}
			h.removalReviews[digest] = modelSettingsRemovalReview{Change: change, Settings: config.Clone()}
		}
		return settings.Preview{Settings: config, Identities: identities, PreviewDigest: digest}, ""
	case "engine:configure-launch":
		var p settings.Configure
		if err := json.Unmarshal(raw, &p); err != nil {
			return nil, err.Error()
		}
		if err := h.validateRemoval(p.Settings, p.ReviewedModelChange); err != nil {
			return nil, err.Error()
		}
		h.configureCount++
		if h.configureFailure == "no effect" {
			return nil, "injected configure failure"
		}
		h.launch.ServerPort, h.launch.LaunchText = p.Settings.ServerPort, p.Settings.LaunchText
		h.launch.ModelLoads, h.launch.Serving = modelconfig.CloneLoads(p.Settings.ModelLoads), p.Settings.Serving.Clone()
		h.launch.Running = p.Resume && !h.savedOff
		if h.configureFailure == "lost response" {
			return nil, "injected lost configure response"
		}
		return h.launch, ""
	case settings.SwitchPrepareMethod:
		return emptySwitchManagerSnapshot(), ""
	case "engine:get-installed":
		return map[string]any{"engines": []map[string]any{{"engine": "llamacpp", "running": h.launch.Running, "terminationVerified": !h.launch.Running}}}, ""
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

func TestModelSettingsApplyRetainedReviewExpiresBeforeWorkerIO(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	h.b.engineConfigMu.Lock()
	for i := range h.b.modelReviews {
		h.b.modelReviews[i].Created = time.Now().Add(-modelReviewTTL)
	}
	h.b.engineConfigMu.Unlock()
	before := len(h.engine.snapshot())
	if _, err := h.b.applyEngineSettings(context.Background(), p, ""); err == nil || engineSettingsErrorCode(err) != codePolicyConflict {
		t.Fatalf("expired review refusal=%v", err)
	}
	if len(h.engine.snapshot()) != before || h.calls() != 0 {
		t.Fatal("expired review reached a worker or native effects")
	}
}

func TestModelSettingsApplyUnacceptedReviewRequiresRecalculateAfterRestart(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	restarted := &Broker{nodeID: h.b.nodeID, clusterDir: h.b.clusterDir}
	if _, err := restarted.applyModelSettings(context.Background(), p, ""); err == nil || engineSettingsErrorCode(err) != codePolicyConflict {
		t.Fatalf("lost unaccepted review=%v", err)
	}
	if h.calls() != 0 {
		t.Fatal("lost unaccepted review reached native effects")
	}
}

func TestModelSettingsApplyReviewRetentionBoundsAndSupersession(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	h.b.engineConfigMu.Lock()
	seed := h.b.modelReviews[0]
	for i := 0; i <= modelReviewCount; i++ {
		next := seed.Request.Clone()
		next.ModelChange.ModelID = fmt.Sprintf("retained-%d", i)
		preview := seed.Preview.Clone()
		preview.PreviewDigest = modelSettingsHash(i)
		if err := h.b.retainModelReviewLocked(next, "", preview, seed.Source); err != nil {
			h.b.engineConfigMu.Unlock()
			t.Fatal(err)
		}
	}
	bytes := 0
	for _, record := range h.b.modelReviews {
		bytes += record.Bytes
	}
	if len(h.b.modelReviews) > modelReviewCount || bytes > modelReviewBytes {
		h.b.engineConfigMu.Unlock()
		t.Fatal("review table exceeded bounds")
	}
	next := seed.Request.Clone()
	preview := seed.Preview.Clone()
	preview.PreviewDigest = modelSettingsHash("replacement")
	if err := h.b.retainModelReviewLocked(next, "", preview, seed.Source); err != nil {
		h.b.engineConfigMu.Unlock()
		t.Fatal(err)
	}
	before := len(h.b.modelReviews)
	next.Settings.LaunchText = strings.Repeat("x", modelReviewBytes+1)
	if err := h.b.retainModelReviewLocked(next, "", preview, seed.Source); err == nil {
		h.b.engineConfigMu.Unlock()
		t.Fatal("oversized review retained")
	}
	if len(h.b.modelReviews) != before {
		h.b.engineConfigMu.Unlock()
		t.Fatal("oversized review changed retained authority")
	}
	h.b.engineConfigMu.Unlock()
	if _, _, err := h.b.retainedModelReview(context.Background(), p, ""); err == nil {
		t.Fatal("evicted/superseded review remained applicable")
	}
}

func TestModelSettingsApplyRecalculateDuringValidationSupersedesOldReview(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	entered, release := make(chan struct{}), make(chan struct{})
	unblock := sync.OnceFunc(func() { close(release) })
	t.Cleanup(unblock)
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, failure := h.answer(method, raw)
		if method == "engine:preview-launch" {
			envelope, err := settings.DecodeWorkerRequest[settings.Request](raw)
			if err != nil {
				return nil, err.Error()
			}
			if envelope.Request.ModelChange == nil {
				close(entered)
				<-release
			}
		}
		return result, failure
	})
	done := make(chan error, 1)
	go func() { _, err := h.b.applyEngineSettings(context.Background(), p, ""); done <- err }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("frozen validation did not reach worker")
	}
	h.b.engineConfigMu.Lock()
	seed := h.b.modelReviews[0]
	next := seed.Request.Clone()
	next.ModelChange.Load.Set.Context = &modelconfig.Context{Mode: "fixed", Tokens: 4096}
	preview := seed.Preview.Clone()
	preview.PreviewDigest = modelSettingsHash("explicit replacement")
	err := h.b.retainModelReviewLocked(next, "", preview, seed.Source)
	h.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	unblock()
	if err := <-done; err == nil || engineSettingsErrorCode(err) != codePolicyConflict {
		t.Fatalf("superseded Apply=%v", err)
	}
	if h.calls() != 0 {
		t.Fatal("superseded review reached native effects")
	}
}

func TestModelSettingsApplyValidatesFiniteSettingsWithoutPortablePatch(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.review(t, h.request(t, true), "")
	before := len(h.engine.snapshot())
	if _, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	validated := false
	for _, call := range h.engine.snapshot()[before:] {
		if call.method != "engine:preview-launch" {
			continue
		}
		envelope, err := settings.DecodeWorkerRequest[settings.Request](call.params)
		if err != nil {
			t.Fatal(err)
		}
		if envelope.Request.ModelChange != nil {
			t.Fatal("Apply re-resolved the portable patch")
		}
		validated = true
	}
	if !validated {
		t.Fatal("Apply omitted frozen native validation")
	}
	h.b.engineConfigMu.Lock()
	h.b.modelReviews = nil
	h.b.engineConfigMu.Unlock()
	before = len(h.engine.snapshot())
	if _, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil {
		t.Fatal("accepted receipt required ephemeral review", err)
	}
	if len(h.engine.snapshot()) != before {
		t.Fatal("accepted receipt revalidated or resolved settings")
	}
}

func TestModelSettingsApplyManualDoesNotRevalidateAcceptedAutoMemory(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.launch.ModelLoads = []modelconfig.ModelLoad{{ProfileKey: "saved-auto", ModelID: "saved-auto", Requested: modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "auto", Floor: 4096, Cap: 8192}}, Resolved: &modelconfig.ResolvedLoad{PerSlotContext: 8192, TotalContext: 8192, Slots: 1, GPU: modelconfig.GPU{Mode: "cpu"}, NativeBuildID: h.build, SourceInputDigest: strings.Repeat("a", 64), PresetDigest: strings.Repeat("b", 64), ResolverRevision: "auto/finite-llama/1", ReviewedInputID: "accepted-synthetic-review", Artifacts: []modelconfig.ArtifactBinding{{Role: "model", Path: "saved.gguf", ResolvedTarget: "saved.gguf", Size: "256", MtimeNS: "1", SampleSHA256: strings.Repeat("c", 64)}}}}}
	p := h.review(t, h.request(t, true), "")
	before := len(h.engine.snapshot())
	if _, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil {
		t.Fatal("manual Apply required unavailable telemetry for an accepted sibling", err)
	}
	for _, call := range h.engine.snapshot()[before:] {
		if call.method != "engine:preview-launch" {
			continue
		}
		envelope, err := settings.DecodeWorkerRequest[settings.Request](call.params)
		if err != nil {
			t.Fatal(err)
		}
		if envelope.ValidateResources || envelope.ResourceModel != "" {
			t.Fatal("manual Apply revalidated accepted Auto capacity")
		}
	}
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
	if len(policy.Profiles) != 1 || policy.Profiles[0].Key != "fixture/model" || policy.Profiles[0].Model != "fixture/model" || policy.Profiles[0].DisplayName != "fixture/model" || policy.Profiles[0].ContextTokens != 8192 || policy.Profiles[0].Qualification != nil || policy.Profiles[0].MemoryBytes != 0 {
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

// The pipe fake returns an accepted whole-preset conversion without opening a
// preset, model or engine. Frozen validation keeps the exact supplied loads.
func modelSettingsPresetAnswer(h *modelSettingsHarness) func(string, json.RawMessage) (any, string) {
	converted := modelSettingsConversionAnswer(h, nil)
	return func(method string, raw json.RawMessage) (any, string) {
		result, failure := converted(method, raw)
		if method != "engine:preview-launch" || failure != "" {
			return result, failure
		}
		envelope, err := settings.DecodeWorkerRequest[settings.Request](raw)
		if err != nil {
			return nil, err.Error()
		}
		preview, ok := result.(settings.Preview)
		if !ok {
			return nil, "fixture expected typed conversion preview"
		}
		if envelope.Request.ModelChange != nil && !slices.ContainsFunc(preview.Settings.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == "sibling" }) {
			sibling := modelconfig.CloneLoads(preview.Settings.ModelLoads)[0]
			sibling.ProfileKey, sibling.ModelID = "sibling", "sibling-artifact"
			slots, fit := int32(2), "on"
			sibling.Requested = modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "fixed", Tokens: 12288}, Slots: &slots, Fit: &fit, ExtraOptions: map[string]string{"seed": "7"}}
			sibling.Resolved.PerSlotContext, sibling.Resolved.TotalContext, sibling.Resolved.Slots = 12288, 24576, 2
			sibling.Resolved.Options = sibling.Requested.Clone()
			preview.Settings.ModelLoads = append(preview.Settings.ModelLoads, sibling)
		}
		preview.Identities["sibling"] = engines.ModelIdentity{GeneralName: "Converted sibling", Architecture: "llama", ConfiguredContextTokens: 12288, Strength: "declared", Files: []engines.ModelIdentityFile{{Role: "model", Name: "sibling.gguf", SizeBytes: 256}}}
		preview.Settings, err = preview.Settings.Canonical()
		if err != nil {
			return nil, err.Error()
		}
		return preview, ""
	}
}

func modelSettingsPresetPolicy(t *testing.T, h *modelSettingsHarness) nodepolicy.Policy {
	t.Helper()
	h.mu.Lock()
	h.launch.LaunchText = "--models-preset /fixture/external.ini"
	h.mu.Unlock()
	policy := h.b.localPolicy().Policy
	policy.Profiles = []nodepolicy.Profile{
		{Key: "edited", DisplayName: "My chosen name", Icon: "code", Engine: "llamacpp", Model: "edited", ContextTokens: 4096,
			Capabilities: []nodepolicy.Capability{nodepolicy.CapTools}, MemoryBytes: 1024, MaxConcurrent: 2, MaxOutputTokens: 512,
			RequestOptions: json.RawMessage(`{"temperature":0.25,"opaque":{"integer":9007199254740993}}`), RequestOptionsMode: nodepolicy.RequestOptionsEnforced, LocalOnly: true,
			Qualification: &nodepolicy.QualificationRef{ID: "saved-qualification", Revision: "1"}},
		{Key: "untouched", DisplayName: "Unrelated profile", Engine: "llamacpp", Model: "untouched", ContextTokens: 2048},
	}
	policy.Roles = []nodepolicy.Role{
		{Name: nodepolicy.TierStrong, Profiles: []string{"untouched", "edited"}},
		{Name: nodepolicy.TierMedium, Profiles: []string{}},
		{Name: nodepolicy.TierWeak, Profiles: []string{"edited", "untouched"}},
	}
	canonical, err := nodepolicy.Parse(settingsJSON(policy))
	if err != nil {
		t.Fatal(err)
	}
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current = canonical
	r.mu.Unlock()
	return canonical
}

func assertModelSettingsConvertedPolicy(t *testing.T, h *modelSettingsHarness, before nodepolicy.Policy) {
	t.Helper()
	assertModelSettingsConvertedPolicyContext(t, h, before, 8192)
}

func assertModelSettingsConvertedPolicyContext(t *testing.T, h *modelSettingsHarness, before nodepolicy.Policy, context int) {
	t.Helper()
	want, err := nodepolicy.Parse(settingsJSON(before))
	if err != nil {
		t.Fatal(err)
	}
	want.Profiles[0].ContextTokens = context
	want.Profiles[0].ModelIdentity = &engines.ModelIdentity{GeneralName: "Fixture model", Architecture: "llama", ConfiguredContextTokens: int64(context), Strength: "declared", Files: []engines.ModelIdentityFile{{Role: "model", Name: "fixture.gguf", SizeBytes: 256}}}
	sibling := slices.IndexFunc(want.Profiles, func(profile nodepolicy.Profile) bool { return profile.Key == "sibling" })
	if sibling < 0 {
		want.Profiles = append(want.Profiles, nodepolicy.Profile{Key: "sibling", DisplayName: "sibling", Engine: "llamacpp", Model: "sibling"})
		sibling = len(want.Profiles) - 1
	}
	want.Profiles[sibling].ContextTokens = 12288
	want.Profiles[sibling].ModelIdentity = &engines.ModelIdentity{GeneralName: "Converted sibling", Architecture: "llama", ConfiguredContextTokens: 12288, Strength: "declared", Files: []engines.ModelIdentityFile{{Role: "model", Name: "sibling.gguf", SizeBytes: 256}}}
	want, err = nodepolicy.Parse(settingsJSON(want))
	if err != nil {
		t.Fatal(err)
	}
	got := h.b.localPolicy().Policy
	want.Revision = got.Revision
	if modelSettingsHash(got) != modelSettingsHash(want) {
		t.Fatalf("complete converted policy\ngot: %s\nwant: %s", settingsJSON(got), settingsJSON(want))
	}
}

func TestModelSettingsApplyTypedLoadForSameArtifactDescriptor(t *testing.T) {
	h := newModelSettingsHarness(t)
	before := h.b.localPolicy().Policy
	before.Profiles = append(before.Profiles, nodepolicy.Profile{Key: "coder-local", DisplayName: "Saved descriptor", Icon: "code", Engine: "llamacpp", Model: "fixture/model", RequestOptions: json.RawMessage(`{"temperature":0.27,"opaque":9007199254740993}`)})
	canonical, err := nodepolicy.Parse(settingsJSON(before))
	if err != nil {
		t.Fatal(err)
	}
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current = canonical
	r.mu.Unlock()
	p := h.request(t, false)
	p.ModelChange.ProfileKey = "coder-local"
	p = h.review(t, p, "")
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || receipt.Phase != "succeeded" || h.calls() != 1 {
		t.Fatalf("same-artifact typed creation = %+v, %v", receipt, err)
	}
	profile, exists := h.b.localPolicy().Policy.ProfileByKey("coder-local")
	if !exists || profile.Model != "coder-local" || profile.ContextTokens != 8192 || profile.DisplayName != "Saved descriptor" || profile.Icon != "code" || !nodepolicy.EqualRequestOptions(profile.RequestOptions, canonical.Profiles[len(canonical.Profiles)-1].RequestOptions) || !reflect.DeepEqual(h.b.localPolicy().Policy.Roles, canonical.Roles) {
		t.Fatalf("same-artifact transition lost profile fields or role order: %+v", profile)
	}
	loads := h.record(t).Snapshot.Settings.ModelLoads
	if len(loads) != 1 || loads[0].ProfileKey != "coder-local" || loads[0].ModelID != "fixture/model" {
		t.Fatalf("same-artifact transition changed native artifact: %+v", loads)
	}
}

func TestModelSettingsDescriptorTypedLoadRefusesRetarget(t *testing.T) {
	for _, tc := range []struct {
		name, engine, artifact string
	}{
		{"different artifact", "llamacpp", "other/model"},
		{"different engine", "strata", "fixture/model"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newModelSettingsHarness(t)
			before := h.b.localPolicy().Policy
			before.Profiles = append(before.Profiles, nodepolicy.Profile{Key: "coder-local", DisplayName: "Saved descriptor", Engine: tc.engine, Model: "fixture/model"})
			canonical, err := nodepolicy.Parse(settingsJSON(before))
			if err != nil {
				t.Fatal(err)
			}
			r := h.b.nodePolicyRuntime()
			r.mu.Lock()
			r.current = canonical
			r.mu.Unlock()
			p := h.request(t, false)
			p.ModelChange.ProfileKey, p.ModelChange.ModelID = "coder-local", tc.artifact
			_, err = h.b.previewEngineSettings(context.Background(), p, "")
			if err == nil || h.calls() != 0 || !reflect.DeepEqual(h.b.localPolicy().Policy, canonical) {
				t.Fatalf("descriptor retarget reached effects: error=%v, configure calls=%d", err, h.calls())
			}
		})
	}
}

func TestModelSettingsApplyConversionPublishesSiblingAndPreservesPolicy(t *testing.T) {
	for _, caller := range []string{"", "peer"} {
		name := "local"
		if caller != "" {
			name = "authenticated peer"
		}
		t.Run(name, func(t *testing.T) {
			h := newModelSettingsHarness(t)
			before := modelSettingsPresetPolicy(t, h)
			h.engine.setAnswer(modelSettingsPresetAnswer(h))
			if caller != "" {
				clustertrusttest.Join(t, h.b.clusterDir, "cluster", h.b.nodeID, caller)
				grantRemoteManagement(t, h.b.clusterDir, "cluster", h.b.nodeID, caller, true)
			}
			p := h.request(t, false)
			p.ModelChange.ProfileKey = "edited"
			p = h.review(t, p, caller)
			receipt, err := h.b.applyEngineSettings(context.Background(), p, caller)
			if err != nil || receipt.Phase != "succeeded" || h.calls() != 1 {
				t.Fatalf("conversion apply = %+v, %v; configure calls = %d", receipt, err, h.calls())
			}
			assertModelSettingsConvertedPolicy(t, h, before)
		})
	}
}

func TestModelSettingsApplyConversionBoundsGeneratedDisplayName(t *testing.T) {
	test := func(name string, size int) {
		t.Run(name, func(t *testing.T) {
			h := newModelSettingsHarness(t)
			before := modelSettingsPresetPolicy(t, h)
			key := strings.Repeat("x", size)
			preset := modelSettingsPresetAnswer(h)
			h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				result, failure := preset(method, raw)
				if method != "engine:preview-launch" || failure != "" {
					return result, failure
				}
				preview, ok := result.(settings.Preview)
				if !ok {
					return nil, "fixture expected conversion preview"
				}
				for i := range preview.Settings.ModelLoads {
					if preview.Settings.ModelLoads[i].ProfileKey == "sibling" {
						preview.Settings.ModelLoads[i].ProfileKey = key
					}
				}
				preview.Identities[key] = preview.Identities["sibling"]
				delete(preview.Identities, "sibling")
				return preview, ""
			})
			request := h.request(t, false)
			request.ModelChange.ProfileKey = "edited"
			request = h.review(t, request, "")
			receipt, err := h.b.applyEngineSettings(context.Background(), request, "")
			if err != nil || receipt.Phase != "succeeded" || h.calls() != 1 {
				t.Fatalf("conversion of %d-byte key = %+v, %v; configure calls=%d", size, receipt, err, h.calls())
			}
			policy := h.b.localPolicy().Policy
			profile, exists := policy.ProfileByKey(key)
			want := key
			if size > 128 {
				want = "model-" + modelSettingsHash(nodepolicy.ModelRef{Engine: "llamacpp", Model: key})[:16]
			}
			if !exists || profile.Model != key || profile.DisplayName != want || profile.ContextTokens != 12288 || profile.ModelIdentity == nil || profile.ModelIdentity.GeneralName != "Converted sibling" {
				t.Fatalf("converted profile = %+v, want key bytes=%d and display=%q", profile, size, want)
			}
			if !reflect.DeepEqual(policy.Roles, before.Roles) || len(h.record(t).Snapshot.Settings.ModelLoads) != 2 {
				t.Fatal("long-key conversion changed roles or lost native loads")
			}
		})
	}
	test("section name at display limit", 128)
	test("section name over display limit", 129)
	test("section name at key limit", 512)
}

func TestModelSettingsApplyConversionOnlyPatchesSelectedRoleMembership(t *testing.T) {
	h := newModelSettingsHarness(t)
	before := modelSettingsPresetPolicy(t, h)
	h.engine.setAnswer(modelSettingsPresetAnswer(h))
	p := h.request(t, false)
	p.ModelChange.ProfileKey = "edited"
	p.ModelChange.Policy = &modelconfig.PolicyPatch{Roles: map[string]bool{nodepolicy.TierMedium: true, nodepolicy.TierWeak: false}}
	p = h.review(t, p, "")
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("membership conversion = %+v, %v", receipt, err)
	}
	before.Roles = []nodepolicy.Role{
		{Name: nodepolicy.TierStrong, Profiles: []string{"untouched", "edited"}},
		{Name: nodepolicy.TierMedium, Profiles: []string{"edited"}},
		{Name: nodepolicy.TierWeak, Profiles: []string{"untouched"}},
	}
	assertModelSettingsConvertedPolicy(t, h, before)
}

func TestModelSettingsApplyConversionPreservesExistingSiblingFields(t *testing.T) {
	h := newModelSettingsHarness(t)
	before := modelSettingsPresetPolicy(t, h)
	before.Profiles = append(before.Profiles, nodepolicy.Profile{Key: "sibling", DisplayName: "My sibling settings", Icon: "image", Engine: "llamacpp", Model: "sibling", ContextTokens: 2048,
		RequestOptions: json.RawMessage(`{"top_k":31,"opaque":{"fraction":0.123456789012345678901}}`), RequestOptionsMode: nodepolicy.RequestOptionsEnforced, LocalOnly: true,
		Capabilities: []nodepolicy.Capability{nodepolicy.CapVision}, MemoryBytes: 2048, MaxConcurrent: 3, MaxOutputTokens: 1024,
		Qualification: &nodepolicy.QualificationRef{ID: "sibling-qualification", Revision: "2"}})
	before.Roles[0].Profiles = []string{"sibling", "untouched", "edited"}
	canonical, err := nodepolicy.Parse(settingsJSON(before))
	if err != nil {
		t.Fatal(err)
	}
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current = canonical
	r.mu.Unlock()
	h.engine.setAnswer(modelSettingsPresetAnswer(h))
	p := h.request(t, false)
	p.ModelChange.ProfileKey = "edited"
	p = h.review(t, p, "")
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("existing sibling conversion = %+v, %v", receipt, err)
	}
	assertModelSettingsConvertedPolicy(t, h, canonical)
}

func TestModelSettingsApplyOrdinaryMultiLoadDoesNotCreateSiblingProfiles(t *testing.T) {
	test := func(name string, typedBefore bool) {
		t.Run(name, func(t *testing.T) {
			h := newModelSettingsHarness(t)
			before := modelSettingsPresetPolicy(t, h)
			preset := modelSettingsPresetAnswer(h)
			if typedBefore {
				// Use the conversion fake only to construct the target's already
				// typed baseline; it is not a newly accepted conversion.
				p := settings.Request{Engine: "llamacpp", ExpectedRevision: 1, Settings: settings.ConfigFromLaunch(h.launch, 18081),
					ModelChange: &modelconfig.ModelChange{ProfileKey: "edited", ModelID: "fixture/model", ExpectedPolicyRevision: before.Revision,
						Load: &modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "fixed", Tokens: 8192}}}}}
				raw, err := json.Marshal(settings.WorkerRequest[settings.Request]{Request: p})
				if err != nil {
					t.Fatal(err)
				}
				result, failure := preset("engine:preview-launch", raw)
				if failure != "" {
					t.Fatal(failure)
				}
				converted, ok := result.(settings.Preview)
				if !ok {
					t.Fatal("fixture expected conversion preview")
				}
				h.setNative(converted.Settings)
			}
			h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				result, failure := preset(method, raw)
				if method == "engine:preview-launch" && failure == "" && !typedBefore {
					preview, ok := result.(settings.Preview)
					if !ok {
						return nil, "fixture expected multi-load preview"
					}
					for i := range preview.Settings.ModelLoads {
						preview.Settings.ModelLoads[i].Resolved.ResolverRevision = ""
						preview.Settings.ModelLoads[i].Resolved.ReviewedInputID = ""
					}
					return preview, ""
				}
				return result, failure
			})
			p := h.request(t, false)
			p.ModelChange.ProfileKey = "edited"
			p = h.review(t, p, "")
			if receipt, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || receipt.Phase != "succeeded" {
				t.Fatalf("ordinary multi-load apply = %+v, %v", receipt, err)
			}
			got := h.b.localPolicy().Policy
			if len(got.Profiles) != 2 || got.Profiles[0].Key != "edited" || got.Profiles[1].Key != "untouched" || !reflect.DeepEqual(got.Roles, before.Roles) {
				t.Fatalf("ordinary edit manufactured a profile or changed roles: %+v", got)
			}
		})
	}
	test("multiple loads without conversion provenance", false)
	test("previous conversion provenance with a typed baseline", true)
}

func TestModelSettingsConversionScopeIncludesAutoSelectedLoad(t *testing.T) {
	h := newModelSettingsHarness(t)
	modelSettingsPresetPolicy(t, h)
	p := h.request(t, false)
	p.ModelChange.ProfileKey = "edited"
	raw, err := json.Marshal(settings.WorkerRequest[settings.Request]{Request: p})
	if err != nil {
		t.Fatal(err)
	}
	result, failure := modelSettingsPresetAnswer(h)("engine:preview-launch", raw)
	if failure != "" {
		t.Fatal(failure)
	}
	preview, ok := result.(settings.Preview)
	if !ok {
		t.Fatal("fixture expected conversion preview")
	}
	for i := range preview.Settings.ModelLoads {
		load := &preview.Settings.ModelLoads[i]
		if load.ProfileKey == "edited" {
			load.Resolved.ResolverRevision = "auto/llama"
			load.Resolved.ReviewedInputID += ":" + strings.Repeat("a", 64)
		}
	}
	_, scope, err := modelConversionReview(nil, *p.ModelChange, preview, settings.Config{})
	if err != nil || !reflect.DeepEqual(scope, []string{"edited", "sibling"}) {
		t.Fatalf("Auto conversion publication scope = %v, %v", scope, err)
	}
	_, ordinary, err := modelConversionReview(nil, *p.ModelChange, preview, preview.Settings)
	if err != nil || len(ordinary) != 0 {
		t.Fatalf("ordinary typed Auto provenance granted conversion authority = %v, %v", ordinary, err)
	}
}

func TestModelSettingsApplyConversionMissingSiblingIdentityRefusesBeforeEffects(t *testing.T) {
	h := newModelSettingsHarness(t)
	before := modelSettingsPresetPolicy(t, h)
	preset := modelSettingsPresetAnswer(h)
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, failure := preset(method, raw)
		if method == "engine:preview-launch" && failure == "" {
			preview, ok := result.(settings.Preview)
			if !ok {
				return nil, "fixture expected conversion preview"
			}
			delete(preview.Identities, "sibling")
			return preview, ""
		}
		return result, failure
	})
	p := h.request(t, false)
	p.ModelChange.ProfileKey = "edited"
	journal := h.record(t)
	_, err := h.b.previewEngineSettings(context.Background(), p, "")
	if err == nil || err.Error() != "native preview did not return converted profile \"sibling\"'s model identity" || h.calls() != 0 || !reflect.DeepEqual(h.record(t), journal) || !reflect.DeepEqual(h.b.localPolicy().Policy, before) {
		t.Fatalf("missing sibling identity reached effects: %v", err)
	}
}

func TestModelSettingsApplyConversionRefusesSiblingRetargetBeforeEffects(t *testing.T) {
	h := newModelSettingsHarness(t)
	before := modelSettingsPresetPolicy(t, h)
	before.Profiles = append(before.Profiles, nodepolicy.Profile{Key: "sibling", DisplayName: "Descriptor profile", Engine: "llamacpp", Model: "different-served-id"})
	canonical, err := nodepolicy.Parse(settingsJSON(before))
	if err != nil {
		t.Fatal(err)
	}
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current = canonical
	r.mu.Unlock()
	h.engine.setAnswer(modelSettingsPresetAnswer(h))
	p := h.request(t, false)
	p.ModelChange.ProfileKey = "edited"
	journal := h.record(t)
	_, err = h.b.previewEngineSettings(context.Background(), p, "")
	if engineSettingsErrorCode(err) != -32602 || h.calls() != 0 || !reflect.DeepEqual(h.record(t), journal) || !reflect.DeepEqual(h.b.localPolicy().Policy, canonical) {
		t.Fatalf("conversion retargeted an existing sibling or reached effects: %v", err)
	}
}

func TestModelSettingsApplyConversionInvalidSourceRefusesBeforeEffects(t *testing.T) {
	h := newModelSettingsHarness(t)
	before := modelSettingsPresetPolicy(t, h)
	preset := modelSettingsPresetAnswer(h)
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, failure := preset(method, raw)
		if method == "engine:preview-launch" && failure == "" {
			preview, ok := result.(settings.Preview)
			if !ok {
				return nil, "fixture expected conversion preview"
			}
			for i := range preview.Settings.ModelLoads {
				if preview.Settings.ModelLoads[i].ProfileKey == "sibling" {
					preview.Settings.ModelLoads[i].Resolved.ReviewedInputID = "invalid-source"
				}
			}
			return preview, ""
		}
		return result, failure
	})
	p := h.request(t, false)
	p.ModelChange.ProfileKey = "edited"
	journal := h.record(t)
	_, err := h.b.previewEngineSettings(context.Background(), p, "")
	if err == nil || err.Error() != "converted profile \"sibling\" has no accepted external source" || h.calls() != 0 || !reflect.DeepEqual(h.record(t), journal) || !reflect.DeepEqual(h.b.localPolicy().Policy, before) {
		t.Fatalf("invalid conversion source reached effects: %v", err)
	}
}

func TestModelSettingsApplyConversionPolicyFailureRetainsFences(t *testing.T) {
	h := newModelSettingsHarness(t)
	before := modelSettingsPresetPolicy(t, h)
	h.engine.setAnswer(modelSettingsPresetAnswer(h))
	p := h.request(t, false)
	p.ModelChange.ProfileKey = "edited"
	p = h.review(t, p, "")
	setReplacer(h.b, durablefile.Replacer{Replace: func(string, string) error { return errors.New("conversion policy replacement refused") }})
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "failed" || h.calls() != 1 || !reflect.DeepEqual(h.b.localPolicy().Policy, before) {
		t.Fatalf("native-first conversion failure = %+v, %v; configure calls = %d", receipt, err, h.calls())
	}
	record := h.record(t)
	if record.Model.Stage != "native-persisted" || record.Snapshot.EffectiveSettings == nil || !record.Snapshot.EffectiveSettings.Equal(record.Snapshot.Settings) || len(record.Model.NativeBefore.ModelLoads) != 0 {
		t.Fatal("conversion failure lost accepted source or exact native persistence")
	}
	snapshot, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, "")
	if err != nil || snapshot.Phase != "applying" || snapshot.Editable || snapshot.UnresolvedError == "" {
		t.Fatalf("public conversion failure fence = %+v, %v", snapshot, err)
	}
	if _, err := h.b.setLocalPolicy(settingsJSON(before), settingsJSON(before.Revision)); policyErrorCode(err) != codePolicyConflict {
		t.Fatalf("local writer bypassed conversion reservation: %v", err)
	}
	if _, err := h.b.setPeerPolicy(settingsJSON(before.PeerProjection()), settingsJSON(before.Revision)); policyErrorCode(err) != codePolicyConflict {
		t.Fatalf("peer writer bypassed conversion reservation: %v", err)
	}
	h.b.engineConfigMu.Lock()
	withdrawn := h.b.advertisementWithdrawnLocked("llamacpp")
	h.b.engineConfigMu.Unlock()
	if !withdrawn {
		t.Fatal("unresolved conversion released its withdrawal")
	}
	setReplacer(h.b, durablefile.Replacer{})
	h.restartJournal(t)
	if h.b.recoverEngineSettings() || h.calls() != 1 || !reflect.DeepEqual(h.b.localPolicy().Policy, before) {
		t.Fatal("failed conversion replayed without fresh local repair")
	}
}

func TestModelSettingsRecoveryConversionReplaysAcceptedPolicyFromNativeBaseline(t *testing.T) {
	h := newModelSettingsHarness(t)
	before := modelSettingsPresetPolicy(t, h)
	h.engine.setAnswer(modelSettingsPresetAnswer(h))
	p := h.request(t, false)
	p.ModelChange.ProfileKey = "edited"
	p = h.review(t, p, "")
	var crash *engineSettingsRecord
	setReplacer(h.b, durablefile.Replacer{Replace: func(string, string) error {
		// Capture the exact journal image after native persistence and before
		// policy publication or the failure handler's reconciliation receipt.
		crash = h.b.engineSettings["llamacpp"].clone()
		return errors.New("crash boundary before policy replacement")
	}})
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || receipt.Phase != "failed" {
		t.Fatalf("capture conversion crash boundary = %+v, %v", receipt, err)
	}
	if crash == nil || crash.Model.Stage != "native-persisted" || crash.Receipts[p.RequestID].Phase != "applying" || len(crash.Model.NativeBefore.ModelLoads) != 0 {
		t.Fatal("did not capture the replayable native-persisted conversion image")
	}
	setReplacer(h.b, durablefile.Replacer{})
	h.b.engineConfigMu.Lock()
	h.b.engineSettings["llamacpp"] = crash
	err := h.b.saveEngineSettingsLocked()
	h.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	h.restartJournal(t)
	// Revalidation metadata is deliberately different. Only the durable
	// accepted identities and NativeBefore may determine policy on replay.
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, failure := h.answer(method, raw)
		if method == "engine:preview-launch" && failure == "" {
			preview, ok := result.(settings.Preview)
			if !ok {
				return nil, "fixture expected frozen preview"
			}
			preview.Identities = nil
			return preview, ""
		}
		return result, failure
	})
	if !h.b.recoverEngineSettings() || h.calls() != 1 {
		t.Fatalf("accepted conversion replay did not settle without another configure; calls = %d", h.calls())
	}
	assertModelSettingsConvertedPolicy(t, h, before)
	after := h.b.localPolicy().Policy
	if modelPolicyFingerprint(after) != crash.Model.After || !h.b.recoverEngineSettings() || h.b.localPolicy().Policy.Revision != after.Revision || h.calls() != 1 {
		t.Fatal("conversion replay changed accepted authority or repeated publication")
	}
}

// Leave native conversion ahead of policy, then review a fresh selected edit
// through the public owner. The fake follows the manager's selected-only identity
// reply and changes whole-engine digests when the selected settings change.
func reviewModelSettingsConversionRepair(t *testing.T) (*modelSettingsHarness, settings.Request, nodepolicy.Policy) {
	t.Helper()
	h := newModelSettingsHarness(t)
	before := modelSettingsPresetPolicy(t, h)
	h.engine.setAnswer(modelSettingsPresetAnswer(h))
	original := h.request(t, false)
	original.ModelChange.ProfileKey = "edited"
	original = h.review(t, original, "")
	setReplacer(h.b, durablefile.Replacer{Replace: func(string, string) error { return errors.New("conversion policy publication refused") }})
	if receipt, err := h.b.applyEngineSettings(context.Background(), original, ""); err != nil || receipt.Phase != "failed" || h.calls() != 1 {
		t.Fatalf("failed conversion = %+v, %v; configure calls = %d", receipt, err, h.calls())
	}
	setReplacer(h.b, durablefile.Replacer{})
	h.restartJournal(t)
	if h.b.recoverEngineSettings() || !reflect.DeepEqual(h.b.localPolicy().Policy, before) {
		t.Fatal("failed conversion replayed before explicit local supersession")
	}
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, failure := h.answer(method, raw)
		if method != "engine:preview-launch" || failure != "" {
			return result, failure
		}
		envelope, err := settings.DecodeWorkerRequest[settings.Request](raw)
		if err != nil {
			return nil, err.Error()
		}
		preview, ok := result.(settings.Preview)
		if !ok {
			return nil, "fixture expected fresh typed review"
		}
		for i := range preview.Settings.ModelLoads {
			preview.Settings.ModelLoads[i].Resolved.SourceInputDigest = strings.Repeat("f", 64)
			preview.Settings.ModelLoads[i].Resolved.PresetDigest = strings.Repeat("1", 64)
		}
		if envelope.Request.ModelChange != nil {
			selected := modelChangeKey(*envelope.Request.ModelChange)
			preview.Identities = map[string]engines.ModelIdentity{selected: preview.Identities[selected]}
		} else {
			// Frozen validation metadata must never replace accepted identities.
			preview.Identities = nil
		}
		return preview, ""
	})
	fresh := h.request(t, false)
	fresh.ModelChange.ProfileKey = "edited"
	fresh.ModelChange.SupersedesRequestID = original.RequestID
	fresh.ModelChange.Load.Set.Context.Tokens = 16384
	name := "Fresh selected name"
	fresh.ModelChange.Policy = &modelconfig.PolicyPatch{DisplayName: &name}
	journal := h.record(t)
	preview, err := h.b.previewEngineSettings(context.Background(), fresh, "")
	if err != nil || len(preview.Errors) != 0 || preview.Conflict != nil || preview.PreviewDigest == "" {
		t.Fatalf("fresh conversion review = %+v, %v", preview, err)
	}
	wantSibling := engines.ModelIdentity{GeneralName: "Converted sibling", Architecture: "llama", ConfiguredContextTokens: 12288, Strength: "declared", Files: []engines.ModelIdentityFile{{Role: "model", Name: "sibling.gguf", SizeBytes: 256}}}
	if !reflect.DeepEqual(preview.Identities["sibling"], wantSibling) || preview.Identities["edited"].ConfiguredContextTokens != 16384 {
		t.Fatalf("fresh review lost accepted sibling or fresh selected identity: %+v", preview.Identities)
	}
	var published struct {
		Profiles []nodepolicy.Profile `json:"profiles"`
	}
	found := false
	for _, diff := range preview.Diff {
		if diff.Field == "policy" {
			if err := json.Unmarshal(diff.After, &published); err != nil {
				t.Fatal(err)
			}
			found = true
		}
	}
	sibling := slices.IndexFunc(published.Profiles, func(profile nodepolicy.Profile) bool { return profile.Key == "sibling" })
	if !found || sibling < 0 || published.Profiles[sibling].ContextTokens != 12288 || !reflect.DeepEqual(published.Profiles[sibling].ModelIdentity, &wantSibling) {
		t.Fatalf("fresh policy review omitted accepted sibling facts: %+v", published)
	}
	if h.calls() != 1 || !reflect.DeepEqual(journal, h.record(t)) || !reflect.DeepEqual(h.b.localPolicy().Policy, before) {
		t.Fatal("fresh review mutated native, journal or policy authority")
	}
	fresh.ModelChange.PreviewDigest = preview.PreviewDigest
	before, err = nodepolicy.Parse(settingsJSON(before))
	if err != nil {
		t.Fatal(err)
	}
	before.Profiles[0].DisplayName = name
	return h, fresh, before
}

func TestModelSettingsApplyFreshSupersessionPublishesUnfinishedConversion(t *testing.T) {
	h, fresh, before := reviewModelSettingsConversionRepair(t)
	receipt, err := h.b.applyEngineSettings(context.Background(), fresh, "")
	if err != nil || receipt.Phase != "succeeded" || h.calls() != 2 {
		t.Fatalf("fresh conversion repair = %+v, %v; configure calls = %d", receipt, err, h.calls())
	}
	assertModelSettingsConvertedPolicyContext(t, h, before, 16384)
	record := h.record(t)
	if record.Receipts[fresh.ModelChange.SupersedesRequestID].Phase != "superseded" || record.Model.Stage != "completed" || !reflect.DeepEqual(record.Model.ConversionProfiles, []string{"edited", "sibling"}) {
		t.Fatalf("replacement lost conversion scope or receipt authority: %+v", record)
	}
	// A completed conversion's journal is retained, but must grant no authority
	// to recreate a sibling intentionally omitted by an ordinary policy edit.
	policy := h.b.localPolicy().Policy
	policy.Profiles = slices.DeleteFunc(slices.Clone(policy.Profiles), func(profile nodepolicy.Profile) bool { return profile.Key == "sibling" })
	if _, err := h.b.setLocalPolicy(settingsJSON(policy), settingsJSON(policy.Revision)); err != nil {
		t.Fatal(err)
	}
	ordinary := h.request(t, false)
	ordinary.ModelChange.ProfileKey = "edited"
	ordinary.ModelChange.Load.Set.Context.Tokens = 16384
	ordinary = h.review(t, ordinary, "")
	if receipt, err := h.b.applyEngineSettings(context.Background(), ordinary, ""); err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("ordinary post-conversion edit = %+v, %v", receipt, err)
	}
	if _, exists := h.b.localPolicy().Policy.ProfileByKey("sibling"); exists || len(h.record(t).Model.ConversionProfiles) != 0 {
		t.Fatal("ordinary edit inherited completed conversion publication authority")
	}
}

func TestModelSettingsRecoveryFreshSupersessionRetainsConversionPublication(t *testing.T) {
	h, fresh, before := reviewModelSettingsConversionRepair(t)
	policyBeforeRepair := h.b.localPolicy().Policy
	var crash *engineSettingsRecord
	setReplacer(h.b, durablefile.Replacer{Replace: func(string, string) error {
		crash = h.b.engineSettings["llamacpp"].clone()
		return errors.New("replacement interrupted before policy publication")
	}})
	if receipt, err := h.b.applyEngineSettings(context.Background(), fresh, ""); err != nil || receipt.Phase != "failed" {
		t.Fatalf("replacement crash boundary = %+v, %v", receipt, err)
	}
	if crash == nil || crash.Model.Stage != "native-persisted" || len(crash.Model.NativeBefore.ModelLoads) != 2 || !reflect.DeepEqual(crash.Model.ConversionProfiles, []string{"edited", "sibling"}) || crash.Receipts[fresh.RequestID].Phase != "applying" {
		t.Fatal("replacement journal lost unfinished publication authority at the typed native boundary")
	}
	state, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, "")
	if err != nil || state.Phase != "applying" || state.Editable || state.UnresolvedError == "" || !reflect.DeepEqual(h.b.localPolicy().Policy, policyBeforeRepair) {
		t.Fatalf("replacement publication failure lost native-first fences: %+v, %v", state, err)
	}
	setReplacer(h.b, durablefile.Replacer{})
	h.b.engineConfigMu.Lock()
	h.b.engineSettings["llamacpp"] = crash
	err = h.b.saveEngineSettingsLocked()
	h.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	h.restartJournal(t)
	if !h.b.recoverEngineSettings() || h.calls() != 2 {
		t.Fatalf("replacement recovery failed or repeated configure; calls = %d", h.calls())
	}
	assertModelSettingsConvertedPolicyContext(t, h, before, 16384)
	settled := h.record(t)
	if settled.Model.Stage != "completed" || settled.Receipts[fresh.RequestID].Phase != "succeeded" || settled.Receipts[fresh.ModelChange.SupersedesRequestID].Phase != "superseded" || modelPolicyFingerprint(h.b.localPolicy().Policy) != crash.Model.After {
		t.Fatal("replacement restart lost accepted policy or receipt authority")
	}
}

func TestModelSettingsFreshConversionRemovalPublishesRemainingSibling(t *testing.T) {
	h, fresh, _ := reviewModelSettingsConversionRepair(t)
	setReplacer(h.b, durablefile.Replacer{Replace: func(string, string) error { return errors.New("typed conversion publication refused") }})
	if receipt, err := h.b.applyEngineSettings(context.Background(), fresh, ""); err != nil || receipt.Phase != "failed" || h.calls() != 2 {
		t.Fatalf("typed conversion failure = %+v, %v", receipt, err)
	}
	setReplacer(h.b, durablefile.Replacer{})
	h.restartJournal(t)
	h.engine.setAnswer(h.answer)
	remove := h.removeRequest(t, "edited")
	remove.ModelChange.SupersedesRequestID = fresh.RequestID
	remove = h.review(t, remove, "")
	if receipt, err := h.b.applyEngineSettings(context.Background(), remove, ""); err != nil || receipt.Phase != "succeeded" || h.calls() != 3 {
		t.Fatalf("conversion removal repair = %+v, %v; configure calls = %d", receipt, err, h.calls())
	}
	policy := h.b.localPolicy().Policy
	if _, exists := policy.ProfileByKey("edited"); exists {
		t.Fatal("conversion removal recreated the removed profile")
	}
	sibling, exists := policy.ProfileByKey("sibling")
	if !exists || sibling.ContextTokens != 12288 || sibling.ModelIdentity == nil || sibling.ModelIdentity.GeneralName != "Converted sibling" || sibling.ModelIdentity.ConfiguredContextTokens != 12288 {
		t.Fatalf("conversion removal lost accepted sibling facts: %+v", sibling)
	}
	wantRoles := []nodepolicy.Role{{Name: "strong", Profiles: []string{"untouched"}}, {Name: "medium", Profiles: []string{}}, {Name: "weak", Profiles: []string{"untouched"}}}
	record := h.record(t)
	if !reflect.DeepEqual(policy.Roles, wantRoles) || !reflect.DeepEqual(record.Model.ConversionProfiles, []string{"sibling"}) || len(record.Snapshot.Settings.ModelLoads) != 1 || record.Snapshot.Settings.ModelLoads[0].ProfileKey != "sibling" || record.Receipts[remove.ModelChange.SupersedesRequestID].Phase != "superseded" {
		t.Fatal("conversion removal lost ordered role cleanup, remaining scope or superseded receipt")
	}
}

func TestModelSettingsFreshConversionRepeatedSupersessionKeepsScope(t *testing.T) {
	h, fresh, before := reviewModelSettingsConversionRepair(t)
	setReplacer(h.b, durablefile.Replacer{Replace: func(string, string) error { return errors.New("second conversion publication refused") }})
	if receipt, err := h.b.applyEngineSettings(context.Background(), fresh, ""); err != nil || receipt.Phase != "failed" || h.calls() != 2 {
		t.Fatalf("second conversion failure = %+v, %v", receipt, err)
	}
	setReplacer(h.b, durablefile.Replacer{})
	h.restartJournal(t)
	next := h.request(t, false)
	next.ModelChange.ProfileKey = "edited"
	next.ModelChange.SupersedesRequestID = fresh.RequestID
	next.ModelChange.Load.Set.Context.Tokens = 24576
	name := "Second repair name"
	next.ModelChange.Policy = &modelconfig.PolicyPatch{DisplayName: &name}
	next = h.review(t, next, "")
	if receipt, err := h.b.applyEngineSettings(context.Background(), next, ""); err != nil || receipt.Phase != "succeeded" || h.calls() != 3 {
		t.Fatalf("repeated conversion repair = %+v, %v", receipt, err)
	}
	before.Profiles[0].DisplayName = name
	assertModelSettingsConvertedPolicyContext(t, h, before, 24576)
	record := h.record(t)
	if !reflect.DeepEqual(record.Model.ConversionProfiles, []string{"edited", "sibling"}) || record.Receipts[fresh.RequestID].Phase != "superseded" || record.Receipts[fresh.ModelChange.SupersedesRequestID].Phase != "superseded" || record.Receipts[next.RequestID].Phase != "succeeded" {
		t.Fatal("repeated conversion repair lost scope or receipt chain")
	}
}

func TestModelSettingsFreshConversionCorruptScopeRefusesRepair(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*modelSettingsOperation)
	}{
		{"empty", func(op *modelSettingsOperation) { op.ConversionProfiles = nil }},
		{"partial", func(op *modelSettingsOperation) { op.ConversionProfiles = []string{"edited"} }},
		{"duplicate", func(op *modelSettingsOperation) { op.ConversionProfiles = []string{"edited", "edited"} }},
		{"missing load", func(op *modelSettingsOperation) { op.ConversionProfiles = []string{"absent"} }},
		{"missing identity", func(op *modelSettingsOperation) { delete(op.Review.Identities, "sibling") }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, fresh, _ := reviewModelSettingsConversionRepair(t)
			policyBefore := h.b.localPolicy().Policy
			h.mu.Lock()
			nativeBefore := settings.ConfigFromLaunch(h.launch, 18081)
			h.mu.Unlock()
			h.b.engineConfigMu.Lock()
			tc.mutate(h.b.engineSettings["llamacpp"].Model)
			err := h.b.saveEngineSettingsLocked()
			h.b.engineConfigMu.Unlock()
			if err != nil {
				t.Fatal(err)
			}
			h.restartJournal(t)
			_, err = h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, "")
			if err == nil {
				fresh.ModelChange.PreviewDigest = ""
				_, err = h.b.previewEngineSettings(context.Background(), fresh, "")
			}
			if err == nil {
				t.Fatal("corrupt conversion publication authority permitted fresh repair")
			}
			if h.calls() != 1 || !reflect.DeepEqual(h.b.localPolicy().Policy, policyBefore) {
				t.Fatal("corrupt conversion scope changed native or policy authority")
			}
			h.mu.Lock()
			nativeAfter := settings.ConfigFromLaunch(h.launch, 18081)
			h.mu.Unlock()
			if !nativeAfter.Equal(nativeBefore) {
				t.Fatal("corrupt conversion scope changed native settings")
			}
		})
	}
}

func TestModelSettingsFreshConversionTruncatedSiblingPublicationRefusesRepair(t *testing.T) {
	h, fresh, _ := reviewModelSettingsConversionRepair(t)
	policyBefore := h.b.localPolicy().Policy
	h.mu.Lock()
	nativeBefore := settings.ConfigFromLaunch(h.launch, 18081)
	h.mu.Unlock()
	h.b.engineConfigMu.Lock()
	op := h.b.engineSettings["llamacpp"].Model
	op.ConversionProfiles = []string{"edited"}
	delete(op.Review.Identities, "sibling")
	err := h.b.saveEngineSettingsLocked()
	h.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	h.restartJournal(t)
	_, err = h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, "")
	if err == nil {
		fresh.ModelChange.PreviewDigest = ""
		var preview settings.Preview
		preview, err = h.b.previewEngineSettings(context.Background(), fresh, "")
		if err == nil {
			fresh.ModelChange.PreviewDigest = preview.PreviewDigest
			var receipt settingsReceipt
			receipt, err = h.b.applyEngineSettings(context.Background(), fresh, "")
			if err == nil {
				t.Fatalf("dual-truncated scope permitted repair: phase=%s, configure calls=%d, profiles=%+v", receipt.Phase, h.calls(), h.b.localPolicy().Policy.Profiles)
			}
		}
	}
	if err == nil || h.calls() != 1 || !reflect.DeepEqual(h.b.localPolicy().Policy, policyBefore) {
		t.Fatalf("corrupt conversion refusal=%v, configure calls=%d", err, h.calls())
	}
	h.mu.Lock()
	nativeAfter := settings.ConfigFromLaunch(h.launch, 18081)
	h.mu.Unlock()
	if !nativeAfter.Equal(nativeBefore) {
		t.Fatal("truncated scope changed native settings")
	}
}

func TestModelSettingsStoredConversionAutoScopeRequiresSiblingWithoutIdentity(t *testing.T) {
	h, _, _ := reviewModelSettingsConversionRepair(t)
	record := h.record(t)
	for i := range record.Model.Review.Settings.ModelLoads {
		load := &record.Model.Review.Settings.ModelLoads[i]
		if load.ProfileKey == "sibling" {
			load.Resolved.ResolverRevision = "auto/finite-llama/1"
			load.Resolved.ReviewedInputID = strings.Repeat("c", 64) + ":" + strings.Repeat("a", 64) + ":-:" + strings.Repeat("b", 64)
		}
	}
	record.Snapshot.Settings = record.Model.Review.Settings.Clone()
	record.Model.ConversionProfiles = []string{"edited"}
	delete(record.Model.Review.Identities, "sibling")
	if err := validateStoredModelSettings(record); err == nil || !strings.Contains(err.Error(), "scope is incomplete") {
		t.Fatalf("converted Auto sibling without scope or identity was not refused for completeness: %v", err)
	}
}

func TestModelSettingsFreshConversionAddsProfileWithoutExpandingConvertedScope(t *testing.T) {
	test := func(name string, auto bool) {
		t.Run(name, func(t *testing.T) {
			h, fresh, _ := reviewModelSettingsConversionRepair(t)
			h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				result, failure := h.answer(method, raw)
				if method != "engine:preview-launch" || failure != "" {
					return result, failure
				}
				preview, ok := result.(settings.Preview)
				if !ok {
					return nil, "fixture expected typed preview"
				}
				if auto {
					for i := range preview.Settings.ModelLoads {
						load := &preview.Settings.ModelLoads[i]
						if load.ProfileKey == "added" {
							load.Resolved.ResolverRevision = "auto/finite-llama/1"
							load.Resolved.ReviewedInputID = "-:" + strings.Repeat("a", 64) + ":-:" + strings.Repeat("b", 64)
						}
					}
				}
				preview.Identities = map[string]engines.ModelIdentity{"added": preview.Identities["added"]}
				return preview, ""
			})
			fresh.ModelChange.ProfileKey = "added"
			fresh.ModelChange.PreviewDigest = ""
			fresh = h.review(t, fresh, "")
			receipt, err := h.b.applyEngineSettings(context.Background(), fresh, "")
			if err != nil || receipt.Phase != "succeeded" || h.calls() != 2 {
				t.Fatalf("added profile repair=%+v, %v; configure calls=%d", receipt, err, h.calls())
			}
			record := h.record(t)
			if !reflect.DeepEqual(record.Model.ConversionProfiles, []string{"edited", "sibling"}) || len(record.Model.Review.Settings.ModelLoads) != 3 {
				t.Fatal("added profile changed converted scope or native load membership")
			}
			if err := validateStoredModelSettings(record); err != nil {
				t.Fatalf("legal added profile made its accepted journal unreadable: %v", err)
			}
			for _, key := range []string{"edited", "sibling", "added"} {
				if _, exists := h.b.localPolicy().Policy.ProfileByKey(key); !exists {
					t.Fatalf("repair omitted profile %q", key)
				}
			}
		})
	}
	test("manual profile added during repair", false)
	test("Auto profile without an external conversion source", true)
}

func TestModelSettingsFreshConversionSiblingDriftRefusesBeforeEffects(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*modelconfig.ModelLoad)
	}{
		{"accepted artifact", func(load *modelconfig.ModelLoad) { load.Resolved.Artifacts[0].SampleSHA256 = strings.Repeat("9", 64) }},
		{"accepted allocation", func(load *modelconfig.ModelLoad) {
			load.Requested.Context = &modelconfig.Context{Mode: "fixed", Tokens: 6144}
			load.Resolved.PerSlotContext, load.Resolved.TotalContext = 6144, 12288
			load.Resolved.Options = load.Requested.Clone()
		}},
		{"accepted native build", func(load *modelconfig.ModelLoad) { load.Resolved.NativeBuildID = "replacement-build" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, fresh, _ := reviewModelSettingsConversionRepair(t)
			journal, policy := h.record(t), h.b.localPolicy().Policy
			h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				result, failure := h.answer(method, raw)
				if method != "engine:preview-launch" || failure != "" {
					return result, failure
				}
				preview, ok := result.(settings.Preview)
				if !ok {
					return nil, "fixture expected sibling drift preview"
				}
				for i := range preview.Settings.ModelLoads {
					if preview.Settings.ModelLoads[i].ProfileKey == "sibling" {
						tc.mutate(&preview.Settings.ModelLoads[i])
					}
				}
				return preview, ""
			})
			fresh.ModelChange.PreviewDigest = ""
			_, err := h.b.previewEngineSettings(context.Background(), fresh, "")
			if err == nil || engineSettingsErrorCode(err) != codePolicyConflict || !strings.Contains(err.Error(), "unfinished conversion sibling changed") {
				t.Fatalf("changed accepted sibling review = %v", err)
			}
			if h.calls() != 1 || !reflect.DeepEqual(journal, h.record(t)) || !reflect.DeepEqual(policy, h.b.localPolicy().Policy) {
				t.Fatal("refused sibling drift changed native, journal or policy authority")
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
	policy.Profiles = []nodepolicy.Profile{{Key: "fixture/profile", DisplayName: "Fixture", Engine: "llamacpp", Model: "fixture/model", LocalOnly: true, RequestOptions: json.RawMessage(`{"temperature":0.1,"opaque":{"integer":9007199254740993,"fraction":0.123456789012345678901}}`)}}
	number := json.Number("0.25")
	change := modelconfig.ModelChange{ProfileKey: "fixture/profile", ModelID: "fixture/model", Policy: &modelconfig.PolicyPatch{Generation: map[string]*json.Number{"temperature": &number}}}
	merged, err := h.b.mergeModelPolicy(policy, "llamacpp", change, "", settings.Preview{}, false)
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

func TestModelSettingsApplyLoadOnlyReservesDerivedPolicy(t *testing.T) {
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
	beforeIdle := policy.Idle.UnloadAfterMinutes
	policy.Idle.UnloadAfterMinutes = 17
	if _, err := h.b.setLocalPolicy(settingsJSON(policy), settingsJSON(policy.Revision)); policyErrorCode(err) != codePolicyConflict {
		t.Fatalf("load-only policy reservation = %v", err)
	}
	h.engine.releaseHeld()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if record := h.record(t); record.Snapshot.Phase != "succeeded" || h.b.localPolicy().Policy.Idle.UnloadAfterMinutes != beforeIdle {
		t.Fatalf("record=%+v", record.Snapshot)
	}
}

func (h *modelSettingsHarness) createProfile(t *testing.T, key string, contextTokens int32) {
	t.Helper()
	p := h.request(t, false)
	p.ModelChange.ProfileKey = key
	p.ModelChange.Load.Set.Context.Tokens = contextTokens
	p = h.review(t, p, "")
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("create profile %q: %+v, %v", key, receipt, err)
	}
}

func (h *modelSettingsHarness) removeRequest(t *testing.T, key string) settings.Request {
	t.Helper()
	p := h.request(t, false)
	p.ModelChange.ProfileKey, p.ModelChange.Remove, p.ModelChange.Load = key, true, nil
	return p
}

func (h *modelSettingsHarness) removalEvidence(t *testing.T, p settings.Request) modelSettingsRemovalReview {
	t.Helper()
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, review := range h.removalReviews {
		if review.Change.ProfileKey == p.ModelChange.ProfileKey && review.Change.ModelID == p.ModelChange.ModelID && review.Change.ExpectedPolicyRevision == p.ModelChange.ExpectedPolicyRevision {
			if review.Change.PreviewDigest == p.ModelChange.PreviewDigest {
				t.Fatal("native review digest equals the public broker digest")
			}
			return modelSettingsRemovalReview{Change: review.Change.Clone(), Settings: review.Settings.Clone()}
		}
	}
	t.Fatal("no original native removal review")
	return modelSettingsRemovalReview{}
}

func (h *modelSettingsHarness) assertRemovalForwarding(t *testing.T, start int, want modelSettingsRemovalReview, wantPreviews, wantConfigures int) {
	t.Helper()
	previews, configures := 0, 0
	for _, call := range h.engine.snapshot()[start:] {
		switch call.method {
		case "engine:preview-launch":
			envelope, err := settings.DecodeWorkerRequest[settings.Request](call.params)
			if err != nil {
				t.Fatal(err)
			}
			if envelope.Request.ModelChange != nil || !envelope.Request.Settings.Equal(want.Settings) || envelope.ReviewedModelChange == nil || !reflect.DeepEqual(*envelope.ReviewedModelChange, want.Change) {
				t.Fatalf("frozen preview lost original native removal proof: %+v", envelope)
			}
			previews++
		case "engine:configure-launch":
			var configure settings.Configure
			if err := json.Unmarshal(call.params, &configure); err != nil {
				t.Fatal(err)
			}
			if configure.Engine != "llamacpp" || !configure.Settings.Equal(want.Settings) || configure.ReviewedModelChange == nil || !reflect.DeepEqual(*configure.ReviewedModelChange, want.Change) {
				t.Fatalf("direct configure lost original native removal proof: %+v", configure)
			}
			configures++
		}
	}
	if previews != wantPreviews || configures != wantConfigures {
		t.Fatalf("forwarded previews/configures = %d/%d, want %d/%d", previews, configures, wantPreviews, wantConfigures)
	}
}

func TestModelSettingsApplyRemoveForwardsOriginalNativeReview(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.createProfile(t, "model-8k", 8192)
	h.createProfile(t, "model-32k", 32768)
	p := h.review(t, h.removeRequest(t, "model-8k"), "")
	want := h.removalEvidence(t, p)
	start := len(h.engine.snapshot())
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("remove = %+v, %v", receipt, err)
	}
	h.assertRemovalForwarding(t, start, want, 2, 1)
	record := h.record(t)
	if !reflect.DeepEqual(record.Model.ReviewedModelChange, want.Change) || record.Model.Change.PreviewDigest != p.ModelChange.PreviewDigest || record.Model.Review.PreviewDigest != p.ModelChange.PreviewDigest {
		t.Fatalf("journal confused native/public removal evidence: %+v", record.Model)
	}
	if len(record.Snapshot.Settings.ModelLoads) != 1 || record.Snapshot.Settings.ModelLoads[0].ProfileKey != "model-32k" || record.Snapshot.Settings.ModelLoads[0].Resolved.PerSlotContext != 32768 {
		t.Fatalf("removal changed the surviving profile: %+v", record.Snapshot.Settings.ModelLoads)
	}
	beforeRetry := len(h.engine.snapshot())
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || receipt.Phase != "succeeded" || len(h.engine.snapshot()) != beforeRetry {
		t.Fatalf("receipt retry repeated native work: %+v, %v", receipt, err)
	}
}

func TestModelSettingsApplyRemoveRefusesInvalidNativeReview(t *testing.T) {
	test := func(name string, corrupt func(*modelSettingsRetainedReview)) {
		t.Run(name, func(t *testing.T) {
			h := newModelSettingsHarness(t)
			h.createProfile(t, "model-8k", 8192)
			p := h.review(t, h.removeRequest(t, "model-8k"), "")
			before := h.record(t)
			policy := h.b.localPolicy().Policy
			h.b.engineConfigMu.Lock()
			corrupt(&h.b.modelReviews[0])
			h.b.engineConfigMu.Unlock()
			if _, err := h.b.applyEngineSettings(context.Background(), p, ""); err == nil || engineSettingsErrorCode(err) != codePolicyConflict {
				t.Fatalf("invalid proof refusal = %v", err)
			}
			if !reflect.DeepEqual(h.record(t), before) || !reflect.DeepEqual(h.b.localPolicy().Policy, policy) || h.calls() != 1 {
				t.Fatal("invalid removal proof changed the journal, native settings or policy")
			}
		})
	}
	test("missing digest", func(r *modelSettingsRetainedReview) { r.Source.NativeDigest = "" })
	test("missing selection", func(r *modelSettingsRetainedReview) { r.Source.ReviewedModelChange = nil })
	test("wrapped digest", func(r *modelSettingsRetainedReview) {
		r.Source.NativeDigest = r.Preview.PreviewDigest
		r.Source.ReviewedModelChange.PreviewDigest = r.Preview.PreviewDigest
	})
	test("wrong native digest", func(r *modelSettingsRetainedReview) {
		r.Source.NativeDigest = strings.Repeat("0", 64)
		r.Source.ReviewedModelChange.PreviewDigest = strings.Repeat("0", 64)
	})
}

func TestModelSettingsApplyRemoveRechecksRetainedNativeSource(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.createProfile(t, "model-8k", 8192)
	p := h.review(t, h.removeRequest(t, "model-8k"), "")
	before := h.record(t)
	policy := h.b.localPolicy().Policy
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, failure := h.answer(method, raw)
		if method == "engine:preview-launch" && failure == "" {
			h.b.engineConfigMu.Lock()
			h.b.modelReviews[0].Source.NativeDigest = strings.Repeat("0", 64)
			h.b.engineConfigMu.Unlock()
		}
		return result, failure
	})
	if _, err := h.b.applyEngineSettings(context.Background(), p, ""); err == nil || engineSettingsErrorCode(err) != codePolicyConflict {
		t.Fatalf("changed retained native source = %v", err)
	}
	if h.calls() != 1 || !reflect.DeepEqual(h.record(t), before) || !reflect.DeepEqual(h.b.localPolicy().Policy, policy) {
		t.Fatal("native review source changed before acceptance but removal was accepted")
	}
}

func TestModelSettingsApplyCreatesSiblingProfilesWithDerivedMetadata(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.createProfile(t, "model-8k", 8192)
	p := h.request(t, false)
	p.ModelChange.ProfileKey = "model-32k"
	p.ModelChange.Load.Set.Context.Tokens = 32768
	slots := int32(2)
	p.ModelChange.Load.Set.Slots = &slots
	displayName := "Large context"
	p.ModelChange.Policy = &modelconfig.PolicyPatch{DisplayName: &displayName}
	p = h.review(t, p, "")
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("create sibling: %+v, %v", receipt, err)
	}
	policy := h.b.localPolicy().Policy
	if len(policy.Profiles) != 2 {
		t.Fatalf("profiles = %+v", policy.Profiles)
	}
	for _, want := range []struct {
		key, display string
		context      int
	}{{"model-8k", "model-8k", 8192}, {"model-32k", "Large context", 32768}} {
		profile, ok := policy.ProfileByKey(want.key)
		if !ok || profile.Engine != "llamacpp" || profile.Model != want.key || profile.DisplayName != want.display || profile.ContextTokens != want.context {
			t.Fatalf("profile %q = %+v", want.key, profile)
		}
		identity := engines.ModelIdentity{GeneralName: "Fixture model", Architecture: "llama", ConfiguredContextTokens: int64(want.context), Strength: "declared", Files: []engines.ModelIdentityFile{{Role: "model", Name: "fixture.gguf", SizeBytes: 256}}}
		if profile.ModelIdentity == nil || !reflect.DeepEqual(*profile.ModelIdentity, identity) {
			t.Fatalf("profile %q identity = %+v, want %+v", want.key, profile.ModelIdentity, identity)
		}
	}
	loads := h.record(t).Snapshot.Settings.ModelLoads
	if len(loads) != 2 || loads[0].ProfileKey != "model-32k" || loads[1].ProfileKey != "model-8k" || loads[0].ModelID != "fixture/model" || loads[1].ModelID != "fixture/model" || loads[0].Resolved == nil || loads[1].Resolved == nil || loads[0].Resolved.TotalContext != 65536 || loads[1].Resolved.TotalContext != 8192 {
		t.Fatalf("sibling loads = %+v", loads)
	}
	data, err := os.ReadFile(h.path)
	if err != nil {
		t.Fatal(err)
	}
	stored, err := nodepolicy.Parse(data)
	if err != nil || modelPolicyFingerprint(stored) != modelPolicyFingerprint(policy) {
		t.Fatalf("saved profiles differ: %+v, %v", stored.Profiles, err)
	}
}

func TestModelSettingsApplyUpdatesOneSiblingAndKeepsDisplayName(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.createProfile(t, "model-8k", 8192)
	h.createProfile(t, "model-32k", 32768)
	before, _ := h.b.localPolicy().Policy.ProfileByKey("model-32k")
	p := h.request(t, false)
	p.ModelChange.ProfileKey = "model-8k"
	p.ModelChange.Load.Set.Context.Tokens = 16384
	p = h.review(t, p, "")
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("update sibling: %+v, %v", receipt, err)
	}
	policy := h.b.localPolicy().Policy
	selected, ok := policy.ProfileByKey("model-8k")
	sibling, siblingOK := policy.ProfileByKey("model-32k")
	if !ok || selected.DisplayName != "model-8k" || selected.ContextTokens != 16384 || selected.ModelIdentity == nil || selected.ModelIdentity.ConfiguredContextTokens != 16384 || !siblingOK || !reflect.DeepEqual(sibling, before) {
		t.Fatalf("updated selected = %+v, sibling = %+v", selected, sibling)
	}
}

func TestModelSettingsApplyRemoveCleansPolicyReferencesAndLastLoad(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.createProfile(t, "model-8k", 8192)
	policy := h.b.localPolicy().Policy
	profile := &policy.Profiles[0]
	profile.MaxOutputTokens, profile.MaxConcurrent, profile.MemoryBytes = 512, 1, 1024
	profile.Qualification = &nodepolicy.QualificationRef{ID: "fixture-qualified", Revision: "1"}
	policy.Profiles = append(policy.Profiles, nodepolicy.Profile{Key: "other", DisplayName: "Other", Engine: "llamacpp", Model: "other"})
	policy.Roles = []nodepolicy.Role{{Name: nodepolicy.TierWeak, Profiles: []string{"model-8k"}}, {Name: nodepolicy.TierStrong, Profiles: []string{"other", "model-8k"}}}
	policy.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: "model-8k"}, {Engine: "llamacpp", Model: "other"}}
	policy.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "model-8k"}}
	policy.EngineBindings = nodepolicy.EngineBindings{{Engine: "llamacpp", Model: "model-8k"}, {Engine: "llamacpp", Model: "other"}}
	if _, err := setCurrent(h.b, settingsJSON(policy)); err != nil {
		t.Fatal(err)
	}
	p := h.review(t, h.removeRequest(t, "model-8k"), "")
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("remove: %+v, %v", receipt, err)
	}
	policy = h.b.localPolicy().Policy
	wantRoles := []nodepolicy.Role{{Name: nodepolicy.TierWeak, Profiles: []string{}}, {Name: nodepolicy.TierStrong, Profiles: []string{"other"}}}
	if len(policy.Profiles) != 1 || policy.Profiles[0].Key != "other" || !reflect.DeepEqual(policy.Roles, wantRoles) || !reflect.DeepEqual(policy.ServedModels, nodepolicy.ServedModels{{Engine: "llamacpp", Model: "other"}}) || len(policy.ResidentModels) != 0 || !reflect.DeepEqual(policy.EngineBindings, nodepolicy.EngineBindings{{Engine: "llamacpp", Model: "other"}}) {
		t.Fatalf("removed profile references remain: %+v", policy)
	}
	record := h.record(t)
	if record.Model.Stage != "completed" || !record.Model.Change.Remove || record.Model.Change.ProfileKey != "model-8k" || len(record.Snapshot.Settings.ModelLoads) != 0 || record.Snapshot.Settings.Serving != nil || record.Snapshot.EffectiveSettings == nil || len(record.Snapshot.EffectiveSettings.ModelLoads) != 0 || h.calls() != 2 {
		t.Fatalf("last-load removal = %+v, configure calls = %d", record, h.calls())
	}
	again, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || again != receipt || h.calls() != 2 {
		t.Fatalf("remove retry = %+v, %v, configure calls = %d", again, err, h.calls())
	}
}

func newWithoutTypedRemovalFixture(t *testing.T, engine, model string) (*modelSettingsHarness, settings.Request) {
	t.Helper()
	h := newModelSettingsHarness(t)
	if engine != engines.NameLlamaCPP {
		descriptor := testDescriptor(engine, engines.ProtocolOpenAIV1)
		descriptor.Family = "strata"
		h.b.setDynamicEngines([]engines.Descriptor{descriptor}, func(int) bool { return true })
	}
	fronted, ok := h.b.frontedEngineProfile(engine)
	if !ok {
		t.Fatal("fixture engine was not fronted")
	}
	h.b.setEngineProxyHandle(fronted, h.pp)
	h.pp.readyMu.Lock()
	h.pp.facadeState[engine] = proxyFacadeState{ready: true, port: 18081}
	h.pp.readyMu.Unlock()
	h.launch.Engine = engine
	policy := h.b.localPolicy().Policy
	policy.Profiles = []nodepolicy.Profile{{Key: "descriptor-key", DisplayName: "Descriptor", Engine: engine, Model: model,
		ContextTokens: 8192, MaxOutputTokens: 512, MaxConcurrent: 1, MemoryBytes: 1024, Qualification: &nodepolicy.QualificationRef{ID: "descriptor-qualified", Revision: "1"}}}
	policy.Roles = []nodepolicy.Role{{Name: "weak", Profiles: []string{"descriptor-key"}}}
	policy.ServedModels = nodepolicy.ServedModels{{Engine: engine, Model: model}}
	policy.ResidentModels = nodepolicy.ResidentModels{{Engine: engine, Model: model}}
	policy.EngineBindings = nodepolicy.EngineBindings{{Engine: engine, Model: model}}
	if _, err := setCurrent(h.b, settingsJSON(policy)); err != nil {
		t.Fatal(err)
	}
	snapshot, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: engine}, "")
	if err != nil {
		t.Fatal(err)
	}
	p := settings.Request{Engine: engine, RequestID: settingsID(), ExpectedRevision: snapshot.Revision, Settings: snapshot.Settings,
		ModelChange: &modelconfig.ModelChange{ProfileKey: "descriptor-key", ModelID: model, Remove: true, ExpectedPolicyRevision: h.b.localPolicy().Policy.Revision}}
	return h, p
}

func TestModelSettingsApplyRemoveWithoutTypedLoad(t *testing.T) {
	test := func(name, engine, model string, recoverAccepted bool) {
		t.Run(name, func(t *testing.T) {
			h, p := newWithoutTypedRemovalFixture(t, engine, model)
			start := len(h.engine.snapshot())
			var receipt settingsReceipt
			if recoverAccepted {
				record := h.seedAcceptedRequest(t, p)
				if record.Model.native() || record.Model.ReviewedModelChange != (modelconfig.ModelChange{}) {
					t.Fatal("untyped removal retained native work or omission proof")
				}
				h.restartJournal(t)
				if !h.b.recoverEngineSettings() || h.b.engineSettings[engine].Snapshot.Phase != "succeeded" {
					t.Fatal("accepted untyped removal did not recover")
				}
			} else {
				p = h.review(t, p, "")
				var err error
				receipt, err = h.b.applyEngineSettings(context.Background(), p, "")
				if err != nil || receipt.Phase != "succeeded" {
					t.Fatalf("untyped removal = %+v, %v", receipt, err)
				}
			}
			got := h.b.localPolicy().Policy
			if len(got.Profiles) != 0 || !reflect.DeepEqual(got.Roles, []nodepolicy.Role{{Name: "weak", Profiles: []string{}}}) ||
				!reflect.DeepEqual(got.ServedModels, nodepolicy.ServedModels{{Engine: engine, Model: model}}) || len(got.ResidentModels) != 0 ||
				!reflect.DeepEqual(got.EngineBindings, nodepolicy.EngineBindings{{Engine: engine, Model: model}}) {
				t.Fatalf("untyped removal changed the wrong references: %+v", got)
			}
			for _, call := range h.engine.snapshot()[start:] {
				if call.method == "engine:preview-launch" || call.method == "engine:configure-launch" {
					t.Fatalf("untyped removal reached typed native work: %s", call.method)
				}
			}
			if recoverAccepted {
				if !h.b.recoverEngineSettings() || h.calls() != 0 {
					t.Fatal("completed untyped recovery repeated effects")
				}
			} else if again, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || again != receipt || h.calls() != 0 {
				t.Fatalf("untyped removal retry = %+v, %v, native calls=%d", again, err, h.calls())
			}
		})
	}
	test("native id differs from profile key", "strata", "native-model", false)
	test("native id equals profile key", "strata", "descriptor-key", false)
	test("llama family without a typed load", "llamacpp", "native-model", false)
	test("accepted removal recovers without native work", "strata", "native-model", true)
}

func TestModelSettingsRemoveKeyEqualToAnotherNativeModel(t *testing.T) {
	h, p := newWithoutTypedRemovalFixture(t, "strata", "native-model")
	policy := h.b.localPolicy().Policy
	other := nodepolicy.Profile{Key: "other-key", DisplayName: "Other", Engine: "strata", Model: "descriptor-key",
		ContextTokens: 4096, MaxOutputTokens: 256, MaxConcurrent: 1, MemoryBytes: 2048, Qualification: &nodepolicy.QualificationRef{ID: "other-qualified", Revision: "2"}}
	policy.Profiles = append(policy.Profiles, other)
	policy.Roles[0].Profiles = append(policy.Roles[0].Profiles, "other-key")
	policy.ServedModels = append(policy.ServedModels, nodepolicy.ModelRef{Engine: "strata", Model: "descriptor-key"})
	policy.ResidentModels = append(policy.ResidentModels, nodepolicy.ModelRef{Engine: "strata", Model: "descriptor-key"})
	policy.EngineBindings = append(policy.EngineBindings, nodepolicy.ModelRef{Engine: "strata", Model: "descriptor-key"})
	if _, err := setCurrent(h.b, settingsJSON(policy)); err != nil {
		t.Fatal(err)
	}
	p.ModelChange.ExpectedPolicyRevision = h.b.localPolicy().Policy.Revision
	p = h.review(t, p, "")
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("remove with key/native-name collision = %+v, %v", receipt, err)
	}
	got := h.b.localPolicy().Policy
	if !reflect.DeepEqual(got.Profiles, []nodepolicy.Profile{other}) ||
		!reflect.DeepEqual(got.Roles, []nodepolicy.Role{{Name: "weak", Profiles: []string{"other-key"}}}) ||
		!reflect.DeepEqual(got.ServedModels, policy.ServedModels) ||
		!reflect.DeepEqual(got.ResidentModels, nodepolicy.ResidentModels{{Engine: "strata", Model: "descriptor-key"}}) ||
		!reflect.DeepEqual(got.EngineBindings, policy.EngineBindings) || h.calls() != 0 {
		t.Fatalf("removal touched another native model's references: %+v", got)
	}
}

func TestModelSettingsApplyRemovePreservesSiblingLoad(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.createProfile(t, "model-8k", 8192)
	h.createProfile(t, "model-32k", 32768)
	p := h.removeRequest(t, "model-8k")
	preview, err := h.b.previewEngineSettings(context.Background(), p, "")
	if err != nil || preview.Impact == nil || !reflect.DeepEqual(preview.Impact.AffectedModels, []string{"model-32k", "model-8k"}) {
		t.Fatalf("remove impact = %+v, %v", preview.Impact, err)
	}
	p.ModelChange.PreviewDigest = preview.PreviewDigest
	if receipt, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("remove sibling: %+v, %v", receipt, err)
	}
	policy := h.b.localPolicy().Policy
	loads := h.record(t).Snapshot.Settings.ModelLoads
	if len(policy.Profiles) != 1 || policy.Profiles[0].Key != "model-32k" || policy.Profiles[0].Model != "model-32k" || policy.Profiles[0].ContextTokens != 32768 || len(loads) != 1 || loads[0].ProfileKey != "model-32k" || loads[0].ModelID != "fixture/model" || loads[0].Resolved == nil || loads[0].Resolved.PerSlotContext != 32768 {
		t.Fatalf("remove damaged sibling: profiles = %+v, loads = %+v", policy.Profiles, loads)
	}
}

func TestModelSettingsReviewRetainsIndependentSiblingProfiles(t *testing.T) {
	h := newModelSettingsHarness(t)
	first := h.request(t, false)
	first.ModelChange.ProfileKey = "model-8k"
	first = h.review(t, first, "")
	second := h.request(t, false)
	second.ModelChange.ProfileKey = "model-32k"
	second.ModelChange.Load.Set.Context.Tokens = 32768
	second = h.review(t, second, "")
	if _, _, err := h.b.retainedModelReview(context.Background(), first, ""); err != nil {
		t.Fatalf("sibling review evicted first key: %v", err)
	}
	if _, _, err := h.b.retainedModelReview(context.Background(), second, ""); err != nil {
		t.Fatalf("second sibling review unavailable: %v", err)
	}
}

func TestModelSettingsPreviewMissingProfileIdentityHasNoWrites(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.request(t, false)
	p.ModelChange.ProfileKey = "model-8k"
	before, policy := h.record(t), h.b.localPolicy().Policy
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, failure := h.answer(method, raw)
		if method == "engine:preview-launch" && failure == "" {
			preview, ok := result.(settings.Preview)
			if !ok {
				return nil, "fixture expected typed preview"
			}
			delete(preview.Identities, "model-8k")
			return preview, ""
		}
		return result, failure
	})
	_, err := h.b.previewEngineSettings(context.Background(), p, "")
	if err == nil || err.Error() != "native preview did not return the profile's model identity" || h.calls() != 0 || !reflect.DeepEqual(h.record(t), before) || !reflect.DeepEqual(h.b.localPolicy().Policy, policy) {
		t.Fatalf("missing identity reached journal/native/policy effects: %v", err)
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
	r.current.Profiles = []nodepolicy.Profile{{Key: "fixture/locked", DisplayName: "Locked", Engine: "llamacpp", Model: "fixture/locked", LocalOnly: true, RequestOptions: json.RawMessage(`{"temperature":0.2}`)}}
	r.mu.Unlock()
	peerID := "peer"
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", h.b.nodeID, peerID)
	grantRemoteManagement(t, h.b.clusterDir, "cluster", h.b.nodeID, peerID, true)
	p := h.request(t, true)
	p.ModelChange.ProfileKey = "fixture/locked"
	before := len(h.engine.snapshot())
	if _, err := h.b.previewEngineSettings(context.Background(), p, peerID); engineSettingsErrorCode(err) != codeRemoteSettingLocked {
		t.Fatalf("generation lock=%v", err)
	}
	for _, call := range h.engine.snapshot()[before:] {
		if call.method == "engine:preview-launch" {
			t.Fatal("locked generation edit reached native preview")
		}
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
