// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/buildapply"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
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
		{"build:apply-prepare", buildapply.PrepareRequest{OperationID: "operation", PreviewID: "preview", Fences: fences, Consent: buildapply.Consent{AllowInterruption: true, AllowEngineStop: true}}},
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
				server, client := net.Pipe()
				t.Cleanup(func() {
					if err := client.Close(); err != nil {
						t.Error(err)
					}
					if err := server.Close(); err != nil {
						t.Error(err)
					}
				})
				if err := client.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
					t.Fatal(err)
				}
				b := &Broker{codec: NewCodec(server)}
				id := json.RawMessage(`"request"`)
				go b.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: test.method, Params: params})
				response, err := NewCodec(client).Read()
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
	base, err := filepath.EvalSymlinks(buildApplyFixtureTempDir(t))
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

// The injected owner represents facts not yet exposed by the real manager.
// It never asks a worker to discover/adopt engines to complete a preview.
type buildApplyTestOwners struct {
	mu                               sync.Mutex
	b                                *Broker
	manager, intent, locks, settings string
	mutateCapture                    func(*buildApplyCapturedOwners)
	afterCompare                     func()
}

func (o *buildApplyTestOwners) WithLocked(compare func(buildApplyCapturedOwners) error) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	capture := o.captureLocked()
	if o.mutateCapture != nil {
		o.mutateCapture(&capture)
	}
	err := compare(capture)
	if o.afterCompare != nil {
		o.afterCompare()
	}
	return err
}

func (o *buildApplyTestOwners) captureLocked() buildApplyCapturedOwners {
	r := &o.b.nodePolicy
	paused := r.live == nodepolicy.Paused && r.current.Availability == nodepolicy.Paused
	loaded := []string{"fixture-model"}
	if paused {
		loaded = []string{}
	}
	activation := r.activationLocked(r.incarnation, r.current.Revision)
	return buildApplyCapturedOwners{
		Saved:        buildapply.Known(paused && !r.recoveryRequired),
		Enforced:     buildapply.Known(paused && activation.Activation == activationActive),
		PhysicalStop: buildapply.Known(paused),
		Impact: buildapply.Impact{
			Complete: true,
			Fences: buildapply.Fences{
				RuntimeGeneration: buildapply.Known(fmt.Sprint(r.runningGeneration)),
				ServiceInstance:   buildapply.Known("fixture-service"), BrokerGeneration: buildapply.Known("fixture-broker"),
				ManagerIncarnation: buildapply.Known(o.manager), PolicyRevision: buildapply.Known(r.current.Revision),
				SettingsRevision: buildapply.Known(o.settings), IntentRevision: buildapply.Known(o.intent), LocksRevision: buildapply.Known(o.locks),
				AvailabilityRevision: buildapply.Known(fmt.Sprint(r.transitionGen)),
			},
			Maintenance:  buildapply.MaintenanceState{BuildApplyActive: buildapply.Known(o.b.buildApplyMaintenance.active != nil), ModelSettingsActive: buildapply.Known(len(o.b.buildApplyMaintenance.settings) != 0)},
			Availability: buildapply.Known(r.live), PauseOrigin: buildapply.Known(string(r.current.PausedBy)),
			Engines:    buildapply.Known([]buildapply.EngineFact{{Engine: "ollama", Active: buildapply.Known(false), Running: buildapply.Known(!paused), LoadedModels: buildapply.Known(loaded), DesiredState: buildapply.Known("on"), Locked: buildapply.Known(true), ScopedStopAuthority: buildapply.Known(true)}}),
			ActiveWork: buildapply.Known(0), ModelOperations: buildapply.Known([]buildapply.ModelOperation{}), SettingsDifferences: buildapply.Known([]buildapply.SettingsDifference{}),
			ExpectedSettings: buildapply.ExpectedSettings{
				PolicyDigest: buildapply.Known(modelPolicyFingerprint(r.current)), EngineSettingsDigest: buildapply.Known("settings-" + o.settings),
				IntentDigest: buildapply.Known("intent-" + o.intent), LocksDigest: buildapply.Known("locks-" + o.locks), StopEngines: buildapply.Known(true), UnloadModels: buildapply.Known(true),
			},
			InterruptionPolicy: buildapply.InterruptionPolicy{OnActive: buildapply.Known(r.current.Pause.OnActive), DrainTimeoutSeconds: buildapply.Known(r.current.Pause.DrainTimeoutSeconds), StopsEngines: true, UnloadsModels: true, CommittedStreamsMayFail: true, IncludesWorkBeforeFence: true},
		},
	}
}

type buildApplyTestFixture struct {
	b      *Broker
	owner  *buildApplyTestOwners
	engine *fakeWorker
	path   string
}

func newBuildApplyTestFixture(t *testing.T) *buildApplyTestFixture {
	t.Helper()
	root := buildApplyFixtureTempDir(t)
	b := &Broker{codec: NewCodec(&syncBuffer{}), nodeID: "fixture-node", clusterDir: filepath.Join(root, "cluster")}
	path := filepath.Join(root, nodepolicy.FileName)
	b.loadNodePolicy(path)
	setClock(b, 5000)
	engine := attachFakeEngine(t, b)
	proxy, _, _ := attachAckingProxy(t, b)
	if !b.replayPolicyToProxy(context.Background(), proxy) {
		t.Fatal("activate fake proxy")
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.running["ollama"] = true
	r.intent["ollama"] = true
	r.loaded["ollama"] = []string{"fixture-model"}
	r.haveAdmission = true
	r.admission = nodepolicy.AdmissionState{ActiveByEngine: map[string]int{}}
	r.mu.Unlock()
	owner := &buildApplyTestOwners{b: b, manager: "fixture-manager", intent: "intent-1", locks: "locks-1", settings: "settings-1"}
	b.buildApplyOwners = owner
	return &buildApplyTestFixture{b: b, owner: owner, engine: engine, path: path}
}

func (f *buildApplyTestFixture) request(t *testing.T) buildapply.PrepareRequest {
	t.Helper()
	f.b.engineConfigMu.Lock()
	defer f.b.engineConfigMu.Unlock()
	f.b.nodePolicy.mu.Lock()
	defer f.b.nodePolicy.mu.Unlock()
	var request buildapply.PrepareRequest
	err := f.owner.WithLocked(func(c buildApplyCapturedOwners) error {
		request = buildapply.PrepareRequest{OperationID: "fixture-operation", PreviewID: "fixture-preview", Fences: c.Impact.Fences, Consent: buildapply.Consent{AllowInterruption: true, AllowEngineStop: true}}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return request
}

func requireBuildApplyPrepared(t *testing.T, result buildapply.Result[buildapply.BuildApplyOperation]) buildapply.BuildApplyOperation {
	t.Helper()
	if !result.Available || result.Value == nil || result.Refusal != nil {
		t.Fatalf("prepare = %+v", result)
	}
	op := *result.Value
	if op.Outcome != nil || !buildApplyTrue(op.Saved) || !buildApplyTrue(op.Enforced) || !buildApplyTrue(op.PhysicalStop) || !buildApplyTrue(op.RemainsPaused) {
		t.Fatalf("quiescence = %+v", op)
	}
	if _, err := json.Marshal(op); err != nil {
		t.Fatalf("operation violates shared response contract: %v", err)
	}
	return op
}

func requireBuildApplyRefusal(t *testing.T, result buildapply.Result[buildapply.BuildApplyOperation], code string) {
	t.Helper()
	if result.Available || result.Value != nil || result.Refusal == nil || result.Refusal.Code != code {
		t.Fatalf("result = %+v; want refusal %s", result, code)
	}
}

func buildApplyEffectCalls(calls []policyCall) []policyCall {
	var effects []policyCall
	for _, call := range calls {
		switch call.method {
		case engineUnloadModelMethod, engineSleepMethod, engineWakeMethod, "engine:start", "engine:stop", "engine:load-model", "engine:prepare-shutdown":
			effects = append(effects, call)
		}
	}
	return effects
}

func requireBuildApplyNoAcceptance(t *testing.T, f *buildApplyTestFixture, before nodepolicy.Policy) {
	t.Helper()
	r := &f.b.nodePolicy
	r.mu.Lock()
	unchanged := reflect.DeepEqual(r.current, before) && r.transitionGen == 0 && r.buildApplyOperation == "" && !r.pendingPauseRequested
	r.mu.Unlock()
	if !unchanged || f.b.buildApplyMaintenance.active != nil {
		t.Fatal("refusal changed policy, accepted Pause or reserved maintenance")
	}
	if _, err := os.Stat(f.path); !os.IsNotExist(err) {
		t.Fatalf("refusal wrote policy, stat error = %v", err)
	}
	if calls := buildApplyEffectCalls(f.engine.snapshot()); len(calls) != 0 {
		t.Fatalf("refusal called lifecycle: %+v", calls)
	}
}

func TestBuildApplyConsentRefusalHasZeroEffects(t *testing.T) {
	for _, name := range []string{"running", "already paused"} {
		t.Run(name, func(t *testing.T) {
			f := newBuildApplyTestFixture(t)
			if name == "already paused" {
				r := &f.b.nodePolicy
				r.mu.Lock()
				r.current.Availability, r.current.PausedBy, r.live = nodepolicy.Paused, nodepolicy.PausedByPeer, nodepolicy.Paused
				r.mu.Unlock()
			}
			request := f.request(t)
			request.Consent.AllowEngineStop = false
			before := f.b.localPolicy().Policy
			requireBuildApplyRefusal(t, f.b.prepareBuildApply(request), "interruption_required")
			requireBuildApplyNoAcceptance(t, f, before)
		})
	}
}

func TestBuildApplyProductionMissingOwnersRefusesBeforeInitialization(t *testing.T) {
	b := &Broker{}
	request := buildapply.PrepareRequest{OperationID: "operation", PreviewID: "preview", Consent: buildapply.Consent{AllowInterruption: true, AllowEngineStop: true}}
	requireBuildApplyRefusal(t, b.prepareBuildApply(request), buildapply.RefusalUnavailable)
	if b.nodePolicy.ready || b.engineSettingsLoaded || b.engineSwitch != nil || b.buildApplyMaintenance.active != nil {
		t.Fatal("missing owner evidence initialized or mutated an owner")
	}
}

func TestBuildApplyPrepareRequiresServicePreviewCorrelation(t *testing.T) {
	f := newBuildApplyTestFixture(t)
	request := f.request(t)
	before := f.b.localPolicy().Policy
	request.PreviewID = ""
	requireBuildApplyRefusal(t, f.b.prepareBuildApply(request), "invalid_request")
	requireBuildApplyNoAcceptance(t, f, before)
}

func TestBuildApplyPreviewCorrelationCannotClaimAnotherReservation(t *testing.T) {
	f := newBuildApplyTestFixture(t)
	request := f.request(t)
	op := requireBuildApplyPrepared(t, f.b.prepareBuildApply(request))
	if op.PreviewID != request.PreviewID {
		t.Fatalf("prepared preview=%q, want %q", op.PreviewID, request.PreviewID)
	}
	other := request
	other.PreviewID = "another-preview"
	requireBuildApplyRefusal(t, f.b.prepareBuildApply(other), buildapply.RefusalMaintenanceBusy)
	finished := f.b.finishBuildApply(buildapply.FinishRequest{OperationID: op.OperationID, Fences: op.Fences})
	if !finished.Available || finished.Value == nil || finished.Value.PreviewID != request.PreviewID {
		t.Fatalf("finish lost preview correlation: %+v", finished)
	}
	requireBuildApplyRefusal(t, f.b.prepareBuildApply(other), "operation_conflict")
}

func TestBuildApplyRequiredEvidenceRefusesWithZeroEffects(t *testing.T) {
	cases := []struct {
		name string
		omit func(*buildApplyCapturedOwners)
	}{
		{"native incarnation", func(c *buildApplyCapturedOwners) {
			c.Impact.Fences.ManagerIncarnation = buildapply.Observation[string]{}
		}},
		{"intent provenance", func(c *buildApplyCapturedOwners) {
			c.Impact.ExpectedSettings.IntentDigest = buildapply.Observation[string]{}
		}},
		{"lock revision", func(c *buildApplyCapturedOwners) { c.Impact.Fences.LocksRevision = buildapply.Observation[string]{} }},
		{"complete inventory", func(c *buildApplyCapturedOwners) {
			c.Impact.Engines = buildapply.Observation[[]buildapply.EngineFact]{}
		}},
		{"download and copy reservations", func(c *buildApplyCapturedOwners) {
			c.Impact.ModelOperations = buildapply.Observation[[]buildapply.ModelOperation]{}
		}},
		{"adopted stop authority", func(c *buildApplyCapturedOwners) {
			(*c.Impact.Engines.Value)[0].ScopedStopAuthority = buildapply.Known(false)
		}},
		{"pending settings", func(c *buildApplyCapturedOwners) { c.Impact.Maintenance.ModelSettingsActive = buildapply.Known(true) }},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			f := newBuildApplyTestFixture(t)
			request := f.request(t)
			before := f.b.localPolicy().Policy
			f.owner.mutateCapture = test.omit
			requireBuildApplyRefusal(t, f.b.prepareBuildApply(request), "owner_evidence_unavailable")
			requireBuildApplyNoAcceptance(t, f, before)
		})
	}
}

func TestBuildApplyKnownAdoptedEngineCannotGainStopAuthorityFromInjectedImpact(t *testing.T) {
	f := newBuildApplyTestFixture(t)
	request := f.request(t)
	before := f.b.localPolicy().Policy
	f.b.engineSettings = map[string]*engineSettingsRecord{"lmstudio": {Snapshot: settings.Snapshot{Adopted: true}}}
	requireBuildApplyRefusal(t, f.b.prepareBuildApply(request), "unscoped_engine")
	requireBuildApplyNoAcceptance(t, f, before)
}

func TestBuildApplyQueuedAcceptanceComparesCurrentOwnerFences(t *testing.T) {
	cases := []struct {
		name   string
		change func(*buildApplyTestFixture)
	}{
		{"policy", func(f *buildApplyTestFixture) { f.b.nodePolicy.current.Revision = "7" }},
		{"manager", func(f *buildApplyTestFixture) { f.owner.manager = "replacement-manager" }},
		{"settings", func(f *buildApplyTestFixture) { f.owner.settings = "settings-2" }},
		{"intent", func(f *buildApplyTestFixture) { f.owner.intent = "intent-2" }},
		{"locks", func(f *buildApplyTestFixture) { f.owner.locks = "locks-2" }},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			f := newBuildApplyTestFixture(t)
			request := f.request(t)
			f.b.engineConfigMu.Lock()
			started := make(chan struct{})
			done := make(chan buildapply.Result[buildapply.BuildApplyOperation], 1)
			go func() { close(started); done <- f.b.prepareBuildApply(request) }()
			<-started
			f.b.nodePolicy.mu.Lock()
			f.owner.mu.Lock()
			test.change(f)
			before := f.b.nodePolicy.current
			f.owner.mu.Unlock()
			f.b.nodePolicy.mu.Unlock()
			f.b.engineConfigMu.Unlock()
			requireBuildApplyRefusal(t, awaitBuildApplyResult(t, done), "changed")
			requireBuildApplyNoAcceptance(t, f, before)
		})
	}
}

func TestBuildApplyExternalOwnerStaysPinnedThroughPauseAcceptance(t *testing.T) {
	f := newBuildApplyTestFixture(t)
	request := f.request(t)
	registered := make(chan bool, 1)
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseOwner := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseOwner)
	first := true
	f.owner.afterCompare = func() {
		if !first {
			return
		}
		first = false
		r := &f.b.nodePolicy
		registered <- r.pendingPauseRequested && r.buildApplyOperation == request.OperationID && f.b.buildApplyMaintenance.active != nil
		<-release
	}
	done := make(chan buildapply.Result[buildapply.BuildApplyOperation], 1)
	go func() { done <- f.b.prepareBuildApply(request) }()
	select {
	case accepted := <-registered:
		if !accepted {
			t.Fatal("owner comparison released its lock before Pause/reservation acceptance")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("owner comparison did not reach acceptance")
	}
	attempted, changed := make(chan struct{}), make(chan struct{})
	go func() {
		close(attempted)
		f.owner.mu.Lock()
		f.owner.manager = "replacement-manager"
		f.owner.mu.Unlock()
		close(changed)
	}()
	<-attempted
	select {
	case <-changed:
		t.Fatal("manager mutation crossed the owner comparison/acceptance interval")
	default:
	}
	releaseOwner()
	result := awaitBuildApplyResult(t, done)
	if !result.Available || result.Value == nil {
		t.Fatalf("accepted operation was lost after manager mutation: %+v", result)
	}
	select {
	case <-changed:
	case <-time.After(5 * time.Second):
		t.Fatal("manager mutation remained pinned after acceptance")
	}
}

func awaitBuildApplyResult(t *testing.T, done <-chan buildapply.Result[buildapply.BuildApplyOperation]) buildapply.Result[buildapply.BuildApplyOperation] {
	t.Helper()
	select {
	case result := <-done:
		return result
	case <-time.After(5 * time.Second):
		t.Fatal("build apply did not leave its injected barrier")
		return buildapply.Result[buildapply.BuildApplyOperation]{}
	}
}

func TestBuildApplyFinishPreservesPauseProvenanceAndSavedIntent(t *testing.T) {
	for _, origin := range []nodepolicy.PauseOrigin{"", nodepolicy.PausedLocally, nodepolicy.PausedByPeer} {
		name := string(origin)
		if name == "" {
			name = "new local pause"
		}
		t.Run(name, func(t *testing.T) {
			f := newBuildApplyTestFixture(t)
			if origin != "" {
				f.b.nodePolicy.mu.Lock()
				f.b.nodePolicy.current.Availability = nodepolicy.Paused
				f.b.nodePolicy.current.PausedBy = origin
				f.b.nodePolicy.live = nodepolicy.Paused
				writePolicyFile(t, f.path, f.b.nodePolicy.current)
				f.b.nodePolicy.mu.Unlock()
			}
			request := f.request(t)
			op := requireBuildApplyPrepared(t, f.b.prepareBuildApply(request))
			beforeCalls := buildApplyEffectCalls(f.engine.snapshot())
			duplicate := requireBuildApplyPrepared(t, f.b.prepareBuildApply(request))
			if !reflect.DeepEqual(op, duplicate) || !reflect.DeepEqual(beforeCalls, buildApplyEffectCalls(f.engine.snapshot())) {
				t.Fatal("duplicate prepare repeated lifecycle or changed its result")
			}
			finishRequest := buildapply.FinishRequest{OperationID: op.OperationID, Fences: op.Fences}
			finished := f.b.finishBuildApply(finishRequest)
			if !finished.Available || finished.Value.Phase != buildapply.PhaseFinish || !buildApplyTrue(finished.Value.RemainsPaused) {
				t.Fatalf("finish = %+v", finished)
			}
			if again := f.b.finishBuildApply(finishRequest); !reflect.DeepEqual(again, finished) {
				t.Fatalf("duplicate finish = %+v; want %+v", again, finished)
			}
			wantOrigin := origin
			if wantOrigin == "" {
				wantOrigin = nodepolicy.PausedLocally
			}
			policy := f.b.localPolicy().Policy
			if policy.Availability != nodepolicy.Paused || policy.PausedBy != wantOrigin || !f.b.nodePolicy.intent["ollama"] {
				t.Fatalf("finish changed provenance or saved intent: %+v", policy)
			}
			if !reflect.DeepEqual(beforeCalls, buildApplyEffectCalls(f.engine.snapshot())) {
				t.Fatal("finish called lifecycle")
			}
			if f.b.buildApplyMaintenance.active != nil || f.b.nodePolicy.buildApplyOperation != "" {
				t.Fatal("correlated finish retained maintenance")
			}
			lease, err := f.b.reserveBuildApplySettings()
			if err != nil {
				t.Fatalf("settings after finish: %v", err)
			}
			lease.Release()
		})
	}
}

func TestBuildApplyConflictingOperationAndUnmatchedFinishKeepReservation(t *testing.T) {
	f := newBuildApplyTestFixture(t)
	request := f.request(t)
	op := requireBuildApplyPrepared(t, f.b.prepareBuildApply(request))
	conflict := request
	conflict.OperationID = "other-operation"
	requireBuildApplyRefusal(t, f.b.prepareBuildApply(conflict), buildapply.RefusalMaintenanceBusy)
	requireBuildApplyRefusal(t, f.b.finishBuildApply(buildapply.FinishRequest{OperationID: conflict.OperationID, Fences: op.Fences}), "operation_conflict")
	wrongFences := op.Fences
	wrongFences.LocksRevision = buildapply.Known("other-locks")
	requireBuildApplyRefusal(t, f.b.finishBuildApply(buildapply.FinishRequest{OperationID: op.OperationID, Fences: wrongFences}), "changed")
	if f.b.buildApplyMaintenance.active == nil || f.b.nodePolicy.buildApplyOperation != op.OperationID {
		t.Fatal("unmatched finish released the reserved operation")
	}
}

func TestBuildApplySettingsOwnerReservationCoversPreviewAndApplyLifetime(t *testing.T) {
	for _, phase := range []string{"preview", "apply"} {
		t.Run(phase, func(t *testing.T) {
			f := newBuildApplyTestFixture(t)
			request := f.request(t)
			entered, release, done := make(chan struct{}), make(chan struct{}), make(chan error, 1)
			go func() {
				// This is the authoritative internal-owner hook, outside the
				// client-request path and any engineConfigMu-free native work.
				lease, err := f.b.reserveBuildApplySettings()
				if err != nil {
					done <- err
					return
				}
				close(entered)
				<-release
				lease.Release()
				done <- nil
			}()
			<-entered
			requireBuildApplyRefusal(t, f.b.prepareBuildApply(request), buildapply.RefusalMaintenanceBusy)
			if f.b.nodePolicy.transitionGen != 0 {
				t.Fatal("busy settings accepted Pause")
			}
			close(release)
			if err := <-done; err != nil {
				t.Fatal(err)
			}
			requireBuildApplyPrepared(t, f.b.prepareBuildApply(request))
		})
	}
}

func TestBuildApplyReservationRefusesInternalSettingsOwners(t *testing.T) {
	f := newBuildApplyTestFixture(t)
	requireBuildApplyPrepared(t, f.b.prepareBuildApply(f.request(t)))
	for _, phase := range []string{"preview", "apply"} {
		t.Run(phase, func(t *testing.T) {
			lease, err := f.b.reserveBuildApplySettings()
			var busy *buildApplyBusyError
			if lease != nil || !errors.As(err, &busy) || policyErrorCode(err) != buildapply.CodeMaintenanceBusy {
				t.Fatalf("internal %s reservation = %v, %v", phase, lease, err)
			}
		})
	}
}

func TestBuildApplyReservationRefusesSettingsEntryPoints(t *testing.T) {
	for _, kind := range []string{"ordinary", "model load", "model policy"} {
		t.Run(kind, func(t *testing.T) {
			b := &Broker{}
			b.buildApplyMaintenance.active = &buildApplyReservation{}
			p := settings.Request{Engine: "llamacpp", RequestID: "fixture-settings", Settings: settings.Config{ServerPort: 18080, ProxyPort: 18081}}
			if kind != "ordinary" {
				p.ModelChange = &modelconfig.ModelChange{ModelID: "fixture/model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Clear: []string{"context"}}}
				if kind == "model policy" {
					p.ModelChange.Policy = &modelconfig.PolicyPatch{Roles: map[string]bool{"weak": true}}
				}
			}
			if err := validateModelSettingsEntry(p); err != nil {
				t.Fatalf("invalid settings fixture: %v", err)
			}
			_, previewErr := b.previewEngineSettings(context.Background(), p, "")
			_, applyErr := b.applyEngineSettings(context.Background(), p, "")
			for _, err := range []error{previewErr, applyErr} {
				var busy *buildApplyBusyError
				if !errors.As(err, &busy) || engineSettingsErrorCode(err) != buildapply.CodeMaintenanceBusy {
					t.Fatalf("settings entry refusal = %v", err)
				}
			}
			if b.engineSettingsLoaded || b.nodePolicy.ready || len(b.buildApplyMaintenance.settings) != 0 {
				t.Fatal("refused settings initialized owners or retained a lease")
			}
		})
	}
}

func TestBuildApplySettingsEntryLeaseCoversNativeApply(t *testing.T) {
	h := newSettingsHarness(t)
	p := h.request(t)
	p.Settings.LaunchText += " --maintenance-proof"
	h.entered, h.release = make(chan struct{}, 1), make(chan struct{})
	var once sync.Once
	release := func() { once.Do(func() { close(h.release) }) }
	t.Cleanup(release)
	done := make(chan error, 1)
	go func() { _, err := h.b.applyEngineSettings(context.Background(), p, ""); done <- err }()
	select {
	case <-h.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("settings did not reach held native apply")
	}
	requireBuildApplyRefusal(t, h.b.prepareBuildApply(buildapply.PrepareRequest{OperationID: "operation", PreviewID: "preview", Consent: buildapply.Consent{AllowEngineStop: true, AllowInterruption: true}}), buildapply.RefusalMaintenanceBusy)
	release()
	if err := <-done; err != nil {
		t.Fatalf("complete settings apply: %v", err)
	}
	h.b.engineConfigMu.Lock()
	defer h.b.engineConfigMu.Unlock()
	if len(h.b.buildApplyMaintenance.settings) != 0 {
		t.Fatal("completed settings apply retained maintenance")
	}
}

func TestBuildApplySettingsEntryLeaseCoversModelPreview(t *testing.T) {
	h := newModelSettingsHarness(t)
	p := h.request(t, true)
	entered, release := make(chan struct{}), make(chan struct{})
	var enteredOnce, releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:preview-launch" {
			enteredOnce.Do(func() { close(entered) })
			<-release
		}
		return h.answer(method, raw)
	})
	done := make(chan error, 1)
	go func() { _, err := h.b.previewEngineSettings(context.Background(), p, ""); done <- err }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("model preview did not reach held worker")
	}
	requireBuildApplyRefusal(t, h.b.prepareBuildApply(buildapply.PrepareRequest{OperationID: "operation", PreviewID: "preview", Consent: buildapply.Consent{AllowEngineStop: true, AllowInterruption: true}}), buildapply.RefusalMaintenanceBusy)
	unblock()
	if err := <-done; err != nil {
		t.Fatalf("complete model preview: %v", err)
	}
	h.b.engineConfigMu.Lock()
	defer h.b.engineConfigMu.Unlock()
	if len(h.b.buildApplyMaintenance.settings) != 0 {
		t.Fatal("completed model preview retained maintenance")
	}
}

func TestBuildApplyPreparingReservationSurvivesDuplicateAndPrematureFinish(t *testing.T) {
	f := newBuildApplyTestFixture(t)
	request := f.request(t)
	// A held lifecycle response leaves the canonical Pause running after the
	// exact admission comparison has released all settings/policy owner locks.
	stopping, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	releaseStop := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseStop)
	var once sync.Once
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		switch method {
		case engineSleepMethod:
			once.Do(func() { close(stopping) })
			<-release
			return map[string]any{"engine": "ollama", "running": false, "terminationVerified": true}, ""
		case "engine:get-installed":
			return json.RawMessage(`{"engines":[{"engine":"ollama","running":false,"terminationVerified":true}]}`), ""
		default:
			return map[string]bool{"ok": true}, ""
		}
	})
	done := make(chan buildapply.Result[buildapply.BuildApplyOperation], 1)
	go func() { done <- f.b.prepareBuildApply(request) }()
	select {
	case <-stopping:
	case <-time.After(5 * time.Second):
		t.Fatal("Pause did not reach the injected stop owner")
	}
	duplicate := f.b.prepareBuildApply(request)
	if !duplicate.Available || duplicate.Value == nil || duplicate.Value.OperationID != request.OperationID || buildApplyTrue(duplicate.Value.PhysicalStop) {
		t.Fatalf("pending duplicate = %+v", duplicate)
	}
	requireBuildApplyRefusal(t, f.b.finishBuildApply(buildapply.FinishRequest{OperationID: request.OperationID, Fences: duplicate.Value.Fences}), buildapply.RefusalMaintenanceBusy)
	if state, err := f.b.setAvailability(context.Background(), nodepolicy.Available, nodepolicy.PausedLocally); state != "" || policyErrorCode(err) != buildapply.CodeMaintenanceBusy {
		t.Fatalf("Resume crossed pending maintenance: %q, %v", state, err)
	}
	releaseStop()
	requireBuildApplyPrepared(t, awaitBuildApplyResult(t, done))
}

func TestBuildApplyUnknownStopReadbackCannotSatisfyCanonicalPause(t *testing.T) {
	f := newBuildApplyTestFixture(t)
	request := f.request(t)
	f.owner.mutateCapture = func(c *buildApplyCapturedOwners) {
		if *c.Impact.Maintenance.BuildApplyActive.Value {
			c.PhysicalStop = buildapply.Observation[bool]{}
		}
	}
	result := f.b.prepareBuildApply(request)
	if !result.Available || result.Value == nil || result.Value.Outcome == nil || *result.Value.Outcome != buildapply.OutcomeUncertain || result.Value.PhysicalStop.Known {
		t.Fatalf("missing post-stop evidence = %+v", result)
	}
	if f.b.buildApplyMaintenance.active == nil {
		t.Fatal("missing proof released maintenance")
	}
}

func TestBuildApplyCorrelatedRecoveryRequiresFreshPositivePauseReadback(t *testing.T) {
	f := newBuildApplyTestFixture(t)
	request := f.request(t)
	f.owner.mutateCapture = func(c *buildApplyCapturedOwners) {
		if *c.Impact.Maintenance.BuildApplyActive.Value {
			c.PhysicalStop = buildapply.Observation[bool]{}
		}
	}
	result := f.b.prepareBuildApply(request)
	if !result.Available || result.Value == nil || result.Value.Outcome == nil {
		t.Fatalf("uncertain prepare = %+v", result)
	}
	finish := buildapply.FinishRequest{OperationID: result.Value.OperationID, Fences: result.Value.Fences}
	requireBuildApplyRefusal(t, f.b.finishBuildApply(finish), "pause_unverified")
	f.owner.mutateCapture = nil
	calls := buildApplyEffectCalls(f.engine.snapshot())
	finish.Fences = f.request(t).Fences
	recovered := f.b.finishBuildApply(finish)
	if !recovered.Available || recovered.Value == nil || recovered.Value.Outcome != nil || !buildApplyTrue(recovered.Value.RemainsPaused) || f.b.buildApplyMaintenance.active != nil {
		t.Fatalf("correlated fresh-read recovery = %+v", recovered)
	}
	if !reflect.DeepEqual(calls, buildApplyEffectCalls(f.engine.snapshot())) || f.b.nodePolicy.current.Availability != nodepolicy.Paused {
		t.Fatal("recovery replayed lifecycle or resumed the node")
	}
}

func TestBuildApplyMaintenanceImpactDoesNotInventOtherOwnerFacts(t *testing.T) {
	b := &Broker{}
	b.nodePolicyRuntime()
	lease, err := b.reserveBuildApplySettings()
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	impact, ok := b.cachedBuildApplyImpact()
	if !ok || !impact.Maintenance.BuildApplyActive.Known || *impact.Maintenance.BuildApplyActive.Value || !buildApplyTrue(impact.Maintenance.ModelSettingsActive) {
		t.Fatalf("maintenance observation = %+v, available=%v", impact.Maintenance, ok)
	}
	if impact.Complete || impact.Fences.ManagerIncarnation.Known || impact.Fences.IntentRevision.Known || impact.Fences.LocksRevision.Known || impact.Engines.Known || impact.ModelOperations.Known {
		t.Fatal("maintenance evidence manufactured missing manager authority")
	}
}

func TestBuildApplyImpactFencesReplacementOfSettingsMaintenanceLease(t *testing.T) {
	b := &Broker{}
	b.nodePolicyRuntime()
	lease, err := b.reserveBuildApplySettings()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { lease.Release() }()
	reads := 0
	impact, ok := captureCachedBuildApplyImpact(func() (buildApplyOwnerRead, bool) {
		reads++
		if reads == 2 {
			lease.Release()
			next, reserveErr := b.reserveBuildApplySettings()
			if reserveErr != nil {
				t.Fatal(reserveErr)
			}
			lease = next
		}
		return b.readCachedBuildApplyOwners()
	})
	if !ok || impact.Complete || len(impact.Refusals) != 1 || impact.Refusals[0].Code != "changed" {
		t.Fatalf("replaced maintenance lease = %+v, available=%v", impact, ok)
	}
}

func TestBuildApplyPhysicalStopUncertaintyRetainsMaintenance(t *testing.T) {
	f := newBuildApplyTestFixture(t)
	f.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		switch method {
		case "engine:get-installed":
			return json.RawMessage(`{"engines":[{"engine":"ollama","running":true}]}`), ""
		case engineSleepMethod:
			return nil, "physical stop unverified"
		default:
			return map[string]bool{"ok": true}, ""
		}
	})
	result := f.b.prepareBuildApply(f.request(t))
	if !result.Available || result.Value == nil || result.Value.Outcome == nil || *result.Value.Outcome != buildapply.OutcomeUncertain || buildApplyTrue(result.Value.PhysicalStop) {
		t.Fatalf("uncertain physical shutdown = %+v", result)
	}
	op := result.Value
	requireBuildApplyRefusal(t, f.b.finishBuildApply(buildapply.FinishRequest{OperationID: op.OperationID, Fences: op.Fences}), "pause_unverified")
	if f.b.buildApplyMaintenance.active == nil || f.b.nodePolicy.live != nodepolicy.Draining {
		t.Fatal("uncertain stop lost maintenance or admission fence")
	}
}

func TestBuildApplyFinishCannotOverwriteNewerSafetyOrConfiguration(t *testing.T) {
	cases := []struct {
		name   string
		change func(*testing.T, *buildApplyTestFixture)
	}{
		{"local pause", func(t *testing.T, f *buildApplyTestFixture) {
			if _, err := f.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil {
				t.Fatal(err)
			}
		}},
		{"lock", func(_ *testing.T, f *buildApplyTestFixture) { f.owner.locks = "newer-locks" }},
		{"settings", func(_ *testing.T, f *buildApplyTestFixture) { f.owner.settings = "newer-settings" }},
		{"policy", func(t *testing.T, f *buildApplyTestFixture) {
			current := f.b.localPolicy().Policy
			_, err := f.b.setLocalPolicy(mustJSON(map[string]any{"version": nodepolicy.SchemaVersion, "idle": map[string]any{"unloadAfterMinutes": 9}}), mustJSON(current.Revision))
			if err != nil {
				t.Fatal(err)
			}
		}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			f := newBuildApplyTestFixture(t)
			op := requireBuildApplyPrepared(t, f.b.prepareBuildApply(f.request(t)))
			test.change(t, f)
			before := f.b.localPolicy().Policy
			locks, settings := f.owner.locks, f.owner.settings
			calls := buildApplyEffectCalls(f.engine.snapshot())
			requireBuildApplyRefusal(t, f.b.finishBuildApply(buildapply.FinishRequest{OperationID: op.OperationID, Fences: op.Fences}), "changed")
			if !reflect.DeepEqual(before, f.b.localPolicy().Policy) || f.owner.locks != locks || f.owner.settings != settings || !reflect.DeepEqual(calls, buildApplyEffectCalls(f.engine.snapshot())) {
				t.Fatal("finish overwrote a newer safety/configuration owner")
			}
			if f.b.buildApplyMaintenance.active == nil {
				t.Fatal("stale finish released maintenance")
			}
		})
	}
}
