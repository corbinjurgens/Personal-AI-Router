// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/buildapply"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

func TestBuildApplyLocalAdmission(t *testing.T) {
	fences := buildapply.Fences{
		RuntimeGeneration: buildapply.Known("runtime"), ServiceInstance: buildapply.Known("service"),
		BrokerGeneration: buildapply.Known("broker"), ManagerIncarnation: buildapply.Known("manager"),
		PolicyRevision: buildapply.Known("1"), SettingsRevision: buildapply.Known("2"),
		IntentRevision: buildapply.Known("3"), LocksRevision: buildapply.Known("4"), AvailabilityRevision: buildapply.Known("5"),
	}
	cases := []struct {
		method string
		params any
	}{
		{"build:apply-impact", buildapply.ImpactRequest{}},
		{"build:apply-prepare", buildapply.PrepareRequest{OperationID: "operation", Fences: fences}},
		{"build:apply-finish", buildapply.FinishRequest{OperationID: "operation", Fences: fences}},
	}
	for _, test := range cases {
		t.Run(test.method, func(t *testing.T) {
			raw, err := json.Marshal(test.params)
			if err != nil {
				t.Fatal(err)
			}
			check := func(t *testing.T, params json.RawMessage, wantCode int) {
				t.Helper()
				var wire bytes.Buffer
				b := &Broker{codec: NewCodec(&wire)}
				id := json.RawMessage(`"request"`)
				b.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: test.method, Params: params})
				response, err := b.codec.Read()
				if err != nil {
					t.Fatalf("read local reply: %v", err)
				}
				if wantCode != 0 {
					if response.Error == nil || response.Error.Code != wantCode {
						t.Fatalf("refusal = %+v", response)
					}
				} else {
					var result buildapply.Result[json.RawMessage]
					if response.Error != nil {
						t.Fatalf("unexpected RPC error: %+v", response.Error)
					}
					if err := json.Unmarshal(response.Result, &result); err != nil {
						t.Fatal(err)
					}
					if result.Available || result.Refusal == nil || result.Refusal.Code != buildapply.RefusalUnavailable {
						t.Fatalf("result = %+v", result)
					}
				}
			}
			t.Run("unavailable without workers", func(t *testing.T) { check(t, raw, 0) })
			for _, field := range []string{"nodeId", "remoteCaller", "command", "endpoint"} {
				t.Run("rejects "+field, func(t *testing.T) {
					var fields map[string]json.RawMessage
					if err := json.Unmarshal(raw, &fields); err != nil {
						t.Fatal(err)
					}
					fields[field] = json.RawMessage(`"untrusted"`)
					bad, err := json.Marshal(fields)
					if err != nil {
						t.Fatal(err)
					}
					check(t, bad, -32602)
				})
			}
		})
	}
}

func TestBuildApplyImpactCachedPolicy(t *testing.T) {
	b := &Broker{}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Revision = "12"
	r.mu.Unlock()
	impact, ok := b.cachedBuildApplyImpact()
	if !ok || impact.Complete || len(impact.Refusals) == 0 {
		t.Fatalf("incomplete impact = %+v, available = %v", impact, ok)
	}
	if !impact.Fences.PolicyRevision.Known || *impact.Fences.PolicyRevision.Value != "12" {
		t.Fatalf("policy fence = %+v", impact.Fences.PolicyRevision)
	}
	if !impact.ExpectedSettings.StopEngines.Known || !*impact.ExpectedSettings.StopEngines.Value ||
		!impact.ExpectedSettings.UnloadModels.Known || !*impact.ExpectedSettings.UnloadModels.Value {
		t.Fatalf("effective pause = %+v", impact.ExpectedSettings)
	}
	if !impact.ExpectedSettings.PolicyDigest.Known || len(*impact.ExpectedSettings.PolicyDigest.Value) != 64 {
		t.Fatalf("policy digest = %+v", impact.ExpectedSettings.PolicyDigest)
	}
}

func TestBuildApplyImpactDoesNotInitializePolicy(t *testing.T) {
	b := &Broker{}
	if _, ok := b.cachedBuildApplyImpact(); ok || b.nodePolicy.ready {
		t.Fatal("capture initialized an unavailable policy owner")
	}
}

func TestBuildApplyImpactBusyOwnerDoesNotWaitOrInitialize(t *testing.T) {
	b := &Broker{}
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if _, ok := b.cachedBuildApplyImpact(); ok {
		t.Fatal("busy settings owner produced a capture")
	}
	if b.nodePolicy.ready || b.engineSettingsLoaded || b.engineSwitch != nil {
		t.Fatal("read initialized an owner while busy")
	}
}

func buildApplyImpactFixture(t *testing.T) *Broker {
	t.Helper()
	source := &engineManagerSource{settingsEpoch: "broker-source-1"}
	proxy := &proxyProcess{}
	b := &Broker{engineMgr: &rpcWorker{}, engineMgrSource: source, proxyProc: proxy, engineSettingsLoaded: true, engineSettingsEpoch: "settings-1", engineSettings: map[string]*engineSettingsRecord{}}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Revision = "22"
	r.current.PausedBy = nodepolicy.PausedByPeer
	r.current.Availability = nodepolicy.Paused
	r.live = nodepolicy.Paused
	r.incarnation = proxy
	r.runID = "proxy-run"
	r.activeRevision = "22"
	r.engineManagerSource = source
	r.models.loadedSource = source
	r.models.loadedProxy = proxy
	r.models.active = &modelConfigAttempt{evidence: modelConfigEvidence{Worker: b.engineMgr, Source: source, Proxy: proxy}, prepare: nodepolicy.PrepareResidentParams{Engine: "ollama", ManagerIncarnation: "manager-1"}}
	r.running["ollama"] = true
	r.runningGeneration = 1
	r.runningObserved["ollama"] = 1
	r.loaded["ollama"] = []string{"synthetic-model"}
	r.intent["ollama"] = true
	r.haveAdmission = true
	r.admission = nodepolicy.AdmissionState{Active: 2, Queued: 1, ActiveByEngine: map[string]int{"ollama": 2}}
	r.mu.Unlock()
	b.engineSettings["ollama"] = &engineSettingsRecord{Snapshot: settings.Snapshot{Engine: "ollama", Epoch: b.engineSettingsEpoch, Revision: 4, AppliedRevision: 4, Phase: "succeeded", Settings: settings.Config{}}}
	fingerprint, valid := settingsFingerprint(b.engineSettings["ollama"].Snapshot)
	if !valid {
		t.Fatal("invalid synthetic settings baseline")
	}
	r.settingsIdentity = map[string]string{"ollama": fingerprint}
	b.engineSwitch = &engineSwitchCoordinator{store: &engineSwitchStore{loaded: true, journal: settings.SwitchJournal{Receipts: map[string]settings.SwitchReceipt{}}}}
	return b
}

func TestBuildApplyImpactProjectsOnlyCurrentCachedOwners(t *testing.T) {
	b := buildApplyImpactFixture(t)
	impact, ok := b.cachedBuildApplyImpact()
	if !ok || impact.Complete || !impact.Fences.ManagerIncarnation.Known || *impact.Fences.ManagerIncarnation.Value != "manager-1" {
		t.Fatalf("impact = %+v, available=%v", impact, ok)
	}
	if !impact.ActiveWork.Known || *impact.ActiveWork.Value != 3 || !impact.PauseOrigin.Known || *impact.PauseOrigin.Value != "peer" {
		t.Fatalf("work/provenance = %+v", impact)
	}
	if !impact.Engines.Known || len(*impact.Engines.Value) != 1 {
		t.Fatalf("engines = %+v", impact.Engines)
	}
	engine := (*impact.Engines.Value)[0]
	if engine.Engine != "ollama" || !engine.Running.Known || !*engine.Running.Value || !engine.DesiredState.Known || *engine.DesiredState.Value != "on" || !engine.LoadedModels.Known || !reflect.DeepEqual(*engine.LoadedModels.Value, []string{"synthetic-model"}) {
		t.Fatalf("engine = %+v", engine)
	}
	if engine.Locked.Known || engine.ScopedStopAuthority.Known || impact.ExpectedSettings.LocksDigest.Known || impact.ExpectedSettings.IntentDigest.Known {
		t.Fatal("created facts the broker cache does not retain")
	}
	if !impact.ExpectedSettings.EngineSettingsDigest.Known || !impact.Fences.SettingsRevision.Known {
		t.Fatal("ignored initialized settings owner")
	}
}

func TestBuildApplyImpactMixedManagerAndLoadedSourcesStayUnknown(t *testing.T) {
	b := buildApplyImpactFixture(t)
	b.nodePolicy.engineManagerSource = &engineManagerSource{settingsEpoch: "retired-manager"}
	impact, ok := b.cachedBuildApplyImpact()
	if !ok || impact.Complete || impact.Fences.ManagerIncarnation.Known {
		t.Fatalf("mixed incarnation = %+v", impact)
	}
	for _, engine := range *impact.Engines.Value {
		if engine.Running.Known || engine.LoadedModels.Known || engine.DesiredState.Known {
			t.Fatalf("stale engine became current: %+v", engine)
		}
	}
}

func TestBuildApplyImpactBrokerSourceIsNotNativeManagerIncarnation(t *testing.T) {
	b := buildApplyImpactFixture(t)
	b.nodePolicy.models.active = nil
	impact, ok := b.cachedBuildApplyImpact()
	if !ok || impact.Fences.ManagerIncarnation.Known {
		t.Fatalf("fabricated native manager incarnation: %+v", impact.Fences.ManagerIncarnation)
	}
}

func TestBuildApplyImpactMissingInventoryAndOperationsStayUnknown(t *testing.T) {
	b := &Broker{}
	b.nodePolicyRuntime()
	impact, ok := b.cachedBuildApplyImpact()
	if !ok || impact.Complete || impact.Engines.Known || impact.ModelOperations.Known || impact.ActiveWork.Known {
		t.Fatalf("missing cache became known empty: %+v", impact)
	}
}

func TestBuildApplyImpactStaleLoadedProxyStaysUnknown(t *testing.T) {
	b := buildApplyImpactFixture(t)
	b.nodePolicy.models.loadedProxy = &proxyProcess{}
	impact, ok := b.cachedBuildApplyImpact()
	if !ok || impact.Complete {
		t.Fatalf("impact = %+v", impact)
	}
	if (*impact.Engines.Value)[0].LoadedModels.Known {
		t.Fatal("used loaded models from another proxy lifetime")
	}
}

func TestBuildApplyImpactChangedOwnerFencesWithdrawDependentFacts(t *testing.T) {
	b := buildApplyImpactFixture(t)
	calls := 0
	impact, ok := captureCachedBuildApplyImpact(func() (buildApplyOwnerRead, bool) {
		calls++
		if calls == 2 {
			b.nodePolicy.runningGeneration++
		}
		return b.readCachedBuildApplyOwners()
	})
	if !ok || impact.Complete || impact.Engines.Known || impact.ExpectedSettings.PolicyDigest.Known || len(impact.Refusals) != 1 || impact.Refusals[0].Code != "changed" {
		t.Fatalf("changed capture = %+v", impact)
	}
}

func TestBuildApplyImpactStoredFalsePauseIsEffectiveTrueWithoutWrites(t *testing.T) {
	b := buildApplyImpactFixture(t)
	policy := b.nodePolicy.current
	encoded, err := json.Marshal(policy)
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &raw); err != nil {
		t.Fatal(err)
	}
	var pause map[string]json.RawMessage
	if err := json.Unmarshal(raw["pause"], &pause); err != nil {
		t.Fatal(err)
	}
	pause["stopEngines"] = json.RawMessage(`false`)
	pause["unloadModels"] = json.RawMessage(`false`)
	raw["pause"], err = json.Marshal(pause)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err = json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(base, "policy.json")
	if err := os.WriteFile(path, encoded, 0600); err != nil {
		t.Fatal(err)
	}
	b.nodePolicy.path = path
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		impact, ok := b.cachedBuildApplyImpact()
		if !ok || !impact.SettingsDifferences.Known {
			t.Fatalf("normalization = %+v", impact)
		}
		for _, difference := range *impact.SettingsDifferences.Value {
			if !difference.Stored.Known || *difference.Stored.Value || !difference.Effective.Known || !*difference.Effective.Value || !difference.Persisted.Known || *difference.Persisted.Value {
				t.Fatalf("invented persisted canonicalization: %+v", difference)
			}
		}
		if !*impact.ExpectedSettings.StopEngines.Value || !*impact.ExpectedSettings.UnloadModels.Value {
			t.Fatal("stored false weakened effective pause")
		}
	}
	afterBytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(encoded, afterBytes) || !before.ModTime().Equal(after.ModTime()) || !os.SameFile(before, after) {
		t.Fatal("preview persisted normalization")
	}
	if b.nodePolicy.live != nodepolicy.Paused || b.nodePolicy.transitionGen != 0 || b.nodePolicy.current.Revision != "22" {
		t.Fatal("preview changed availability or policy")
	}
}

func TestBuildApplyImpactKnownOperationsBlockWithoutInitializingOtherOwners(t *testing.T) {
	b := buildApplyImpactFixture(t)
	b.pendingInstallStarts = map[*pendingInstallStart]struct{}{&pendingInstallStart{engine: "ollama"}: {}}
	b.activeSettings = map[string]activeSettingsOperation{"settings-op": {Engine: "ollama"}}
	b.engineSwitch.store.journal.Receipts["switch"] = settings.SwitchReceipt{Request: settings.SwitchRequest{SourceEngine: "ollama", TargetEngine: "lmstudio"}}
	impact, ok := b.cachedBuildApplyImpact()
	if !ok || impact.Complete || !impact.ModelOperations.Known {
		t.Fatalf("operations = %+v", impact)
	}
	kinds := map[string]bool{}
	for _, operation := range *impact.ModelOperations.Value {
		if !operation.BlocksApply {
			t.Fatal("unfenced mutation")
		}
		kinds[operation.Kind] = true
	}
	for _, kind := range []string{"install", "settings", "switch"} {
		if !kinds[kind] {
			t.Fatalf("missing operation %s", kind)
		}
	}
	if b.boundEngines != nil || !b.engineSettingsLoaded || !b.engineSwitch.store.loaded {
		t.Fatal("preview initialized or retired an owner")
	}
}

func TestBuildApplyImpactPrivateSettingsOnlyLeaveAsDigests(t *testing.T) {
	b := buildApplyImpactFixture(t)
	// A private journal error must never be exported as operational detail.
	b.engineSettings["ollama"].Snapshot.Error = "PRIVATE_SETTINGS_SECRET"
	impact, ok := b.cachedBuildApplyImpact()
	if !ok {
		t.Fatal("impact unavailable")
	}
	data, err := json.Marshal(impact)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "PRIVATE_SETTINGS_SECRET") {
		t.Fatal("raw private settings escaped the owner")
	}
	first := *impact.Fences.SettingsRevision.Value
	b.engineSettings["ollama"].Snapshot.Error = "CHANGED_PRIVATE_SETTINGS_SECRET"
	changed, ok := b.cachedBuildApplyImpact()
	if !ok || !changed.Fences.SettingsRevision.Known || *changed.Fences.SettingsRevision.Value == first {
		t.Fatal("private settings change was not fenced")
	}
}

func TestBuildApplyImpactPrivateCanonicalSettingsExpectation(t *testing.T) {
	b := buildApplyImpactFixture(t)
	set := func(value string) {
		t.Helper()
		record := b.engineSettings["ollama"]
		record.Snapshot.Settings.LaunchText = value
		fingerprint, ok := settingsFingerprint(record.Snapshot)
		if !ok {
			t.Fatal("invalid settings fixture")
		}
		b.nodePolicy.settingsIdentity["ollama"] = fingerprint
	}
	set("PRIVATE_LAUNCH_SECRET")
	before, ok := b.cachedBuildApplyImpact()
	if !ok || !before.ExpectedSettings.EngineSettingsDigest.Known {
		t.Fatal("missing canonical settings expectation")
	}
	data, err := json.Marshal(before)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "PRIVATE_LAUNCH_SECRET") {
		t.Fatal("private launch text escaped")
	}
	set("PRIVATE_CHANGED_SECRET")
	after, ok := b.cachedBuildApplyImpact()
	if !ok || !after.ExpectedSettings.EngineSettingsDigest.Known || *before.ExpectedSettings.EngineSettingsDigest.Value == *after.ExpectedSettings.EngineSettingsDigest.Value {
		t.Fatal("changed private settings retained the old expectation")
	}
}

func TestBuildApplyImpactSemanticExpectationsExcludeRuntimeRevisionChanges(t *testing.T) {
	b := buildApplyImpactFixture(t)
	before, ok := b.cachedBuildApplyImpact()
	if !ok {
		t.Fatal("missing impact")
	}
	b.nodePolicy.current.Revision = "23"
	b.nodePolicy.activeRevision = "23"
	b.engineSettingsEpoch = "settings-2"
	b.engineSettings["ollama"].Snapshot.Epoch = b.engineSettingsEpoch
	fingerprint, valid := settingsFingerprint(b.engineSettings["ollama"].Snapshot)
	if !valid {
		t.Fatal("invalid settings fixture")
	}
	b.nodePolicy.settingsIdentity["ollama"] = fingerprint
	after, ok := b.cachedBuildApplyImpact()
	if !ok || !before.ExpectedSettings.EngineSettingsDigest.Known || !after.ExpectedSettings.EngineSettingsDigest.Known {
		t.Fatal("missing protected settings")
	}
	if *before.ExpectedSettings.EngineSettingsDigest.Value != *after.ExpectedSettings.EngineSettingsDigest.Value || *before.ExpectedSettings.PolicyDigest.Value != *after.ExpectedSettings.PolicyDigest.Value {
		t.Fatal("runtime revisions changed semantic expectations")
	}
	if *before.Fences.SettingsRevision.Value == *after.Fences.SettingsRevision.Value || *before.Fences.PolicyRevision.Value == *after.Fences.PolicyRevision.Value {
		t.Fatal("runtime revisions were not fenced")
	}
}

func TestBuildApplyImpactOperationReplacementIsFenced(t *testing.T) {
	b := buildApplyImpactFixture(t)
	b.activeSettings = map[string]activeSettingsOperation{"old": {Engine: "ollama"}}
	calls := 0
	impact, ok := captureCachedBuildApplyImpact(func() (buildApplyOwnerRead, bool) {
		calls++
		if calls == 2 {
			delete(b.activeSettings, "old")
			b.activeSettings["new"] = activeSettingsOperation{Engine: "ollama"}
		}
		return b.readCachedBuildApplyOwners()
	})
	if !ok || impact.Complete || impact.Engines.Known || len(impact.Refusals) != 1 || impact.Refusals[0].Code != "changed" {
		t.Fatalf("replaced operation was not fenced: %+v", impact)
	}
}
