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
	"strings"
	"testing"
	"time"

	"nvpair-shared/buildapply"
	"nvpair-shared/nodepolicy"
)

func TestBuildApplyMaintenanceRefusesResumeWithoutSupersedingPause(t *testing.T) {
	f := newBuildApplyTestFixture(t)
	op := requireBuildApplyPrepared(t, f.b.prepareBuildApply(f.request(t)))
	r := &f.b.nodePolicy
	r.mu.Lock()
	generation, policy := r.transitionGen, r.current
	r.mu.Unlock()
	state, err := f.b.setAvailability(context.Background(), nodepolicy.Available, nodepolicy.PausedLocally)
	var busy *buildApplyBusyError
	if state != "" || !errors.As(err, &busy) || policyErrorCode(err) != buildapply.CodeMaintenanceBusy {
		t.Fatalf("Resume during maintenance = %q, %v", state, err)
	}
	r.mu.Lock()
	unchanged := r.transitionGen == generation && reflect.DeepEqual(r.current, policy) && r.buildApplyOperation == op.OperationID
	r.mu.Unlock()
	if !unchanged || f.b.wakeRefusal() == "" {
		t.Fatal("maintenance Resume superseded Pause or permitted wake")
	}
	if state, err := f.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil || state != nodepolicy.Paused {
		t.Fatalf("local safety Pause during maintenance = %q, %v", state, err)
	}
	if f.b.buildApplyMaintenance.active == nil || f.b.nodePolicy.current.PausedBy != nodepolicy.PausedLocally {
		t.Fatal("safety Pause released maintenance or lost local provenance")
	}
}

func TestBuildApplyAcceptedFenceRefusesWakeBeforePhysicalPause(t *testing.T) {
	b := &Broker{}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.buildApplyOperation = "accepted-operation"
	r.live = nodepolicy.Available
	r.mu.Unlock()
	if b.wakeRefusal() == "" {
		t.Fatal("wake crossed accepted maintenance before physical Pause")
	}
}

// paProjectionBarrier replies only after the fake engine's single reader has
// recorded every preceding projection on this connection.
func paProjectionBarrier(t *testing.T, b *Broker) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	raw, rpcErr, err := b.getEngineMgr().Call(ctx, "pa:projection-barrier", json.RawMessage(`{}`))
	if err != nil || rpcErr != nil {
		t.Fatalf("projection reader barrier: transport=%v rpc=%+v", err, rpcErr)
	}
	var reply struct {
		OK bool `json:"ok"`
	}
	if err := json.Unmarshal(raw, &reply); err != nil {
		t.Fatalf("decode projection reader barrier: %v", err)
	}
	if !reply.OK {
		t.Fatal("projection reader barrier was not acknowledged")
	}
}

func TestPauseProjectionCarriesProvenanceAndRevision(t *testing.T) {
	fixture := func(t *testing.T) (*activatedBroker, *fakeWorker) {
		t.Helper()
		ab := newActivatedBroker(t)
		_, err := ab.b.dispatchPolicy(context.Background(), methodPolicySet, policyParams{
			Policy: mustJSON(map[string]any{
				"version": nodepolicy.SchemaVersion,
				"roles":   map[string]any{"weak": map[string]any{"engine": "ollama", "model": "fixture-model"}},
			}),
			ExpectedPersistedRevision: mustJSON(ab.b.localPolicy().Policy.Revision),
		})
		if err != nil {
			t.Fatalf("save role fixture: %v", err)
		}
		return ab, attachFakeEngine(t, ab.b)
	}
	transition := func(t *testing.T, ab *activatedBroker, engine *fakeWorker, state nodepolicy.Availability, origin nodepolicy.PauseOrigin) {
		t.Helper()
		before := ab.b.localPolicy().Policy
		result, err := ab.b.setAvailability(context.Background(), state, origin)
		if err != nil {
			t.Fatalf("set availability %s from %s: %v", state, origin, err)
		}
		if result != state {
			t.Fatalf("availability result = %s, want %s", result, state)
		}
		saved := ab.b.localPolicy().Policy
		wantOrigin := origin
		if state == nodepolicy.Available {
			wantOrigin = ""
		}
		if saved.Availability != state || saved.PausedBy != wantOrigin || saved.Revision == before.Revision {
			t.Fatalf("saved policy = %+v; want %s/%s with a revision after %s", saved, state, wantOrigin, before.Revision)
		}
		want := saved.Advertisement()
		if !reflect.DeepEqual(want.Roles, before.Advertisement().Roles) || len(want.Roles) == 0 {
			t.Fatalf("availability changed declared roles: %+v", want.Roles)
		}
		var matched nodepolicy.Advertisement
		engine.waitFor("saved availability advertisement", func([]policyCall) bool {
			for _, got := range nodeProjections(t, engine) {
				// A pending local veto still carries the previous saved revision
				// and cannot satisfy this complete advertisement comparison.
				if reflect.DeepEqual(got, want) {
					matched = got
					return true
				}
			}
			return false
		})
		if stored := ab.b.localPolicy().Policy.Advertisement(); !reflect.DeepEqual(matched, stored) {
			t.Fatalf("matched advertisement = %+v, stored = %+v", matched, stored)
		}
	}
	emission := func(name string, state nodepolicy.Availability, origin nodepolicy.PauseOrigin) {
		t.Run(name, func(t *testing.T) {
			ab, engine := fixture(t)
			if state == nodepolicy.Available {
				transition(t, ab, engine, nodepolicy.Paused, origin)
			}
			transition(t, ab, engine, state, origin)
		})
	}
	emission("local pause", nodepolicy.Paused, nodepolicy.PausedLocally)
	emission("local resume", nodepolicy.Available, nodepolicy.PausedLocally)
	emission("peer pause", nodepolicy.Paused, nodepolicy.PausedByPeer)

	t.Run("peer resume veto emits no projection", func(t *testing.T) {
		ab, engine := fixture(t)
		transition(t, ab, engine, nodepolicy.Paused, nodepolicy.PausedLocally)
		paProjectionBarrier(t, ab.b)
		before := nodeProjections(t, engine)
		saved := ab.b.localPolicy().Policy
		result, err := ab.b.setAvailability(context.Background(), nodepolicy.Available, nodepolicy.PausedByPeer)
		var veto *localVetoError
		if !errors.As(err, &veto) || result != "" {
			t.Fatalf("peer resume = %q, %v; want localVetoError and no result", result, err)
		}
		if got := ab.b.localPolicy().Policy; !reflect.DeepEqual(got, saved) {
			t.Fatalf("veto changed stored policy: got %+v, want %+v", got, saved)
		}
		paProjectionBarrier(t, ab.b)
		if got := nodeProjections(t, engine); !reflect.DeepEqual(got, before) {
			t.Fatalf("veto emitted a projection: before %+v, after %+v", before, got)
		}
	})
}

func pauseLifecycleCalls(calls []policyCall) []policyCall {
	var lifecycle []policyCall
	for _, call := range calls {
		if call.method == engineUnloadModelMethod || call.method == engineSleepMethod {
			lifecycle = append(lifecycle, call)
		}
	}
	return lifecycle
}

func TestPauseUsesReplacementEngineManagerObservations(t *testing.T) {
	b, _ := newPolicyBroker(t)
	old := b.activateEngineManagerSource()
	b.observeEngineRunningFrom(old, "ollama", true)
	b.observeLoadedFrom(old, map[string][]string{"ollama": {"old-model"}})
	b.observeIntentFrom(old, map[string]bool{"ollama": true})
	current := b.activateEngineManagerSource()
	engine := attachFakeEngine(t, b)
	b.observeEngineRunningFrom(current, "llamacpp", true)
	b.observeLoadedFrom(current, map[string][]string{"llamacpp": {"new-model"}})
	b.observeIntentFrom(current, map[string]bool{"llamacpp": true})
	b.retireEngineManagerSource(old)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := b.pauseEnginesOn(ctx, b.getEngineMgr()); err != nil {
		t.Fatalf("pause replacement manager: %v", err)
	}
	var slept []string
	for _, call := range pauseLifecycleCalls(engine.snapshot()) {
		if call.method != engineSleepMethod {
			continue
		}
		var p struct {
			Engine string `json:"engine"`
		}
		if err := json.Unmarshal(call.params, &p); err != nil {
			t.Fatalf("decode sleep request: %v", err)
		}
		slept = append(slept, p.Engine)
	}
	if !reflect.DeepEqual(slept, []string{"llamacpp"}) {
		t.Fatalf("pause slept engines %v, want replacement engine only", slept)
	}
}

func requirePauseManagerCleanup(t *testing.T, b *Broker, names ...string) {
	t.Helper()
	worker, source := b.pauseManager()
	if !b.engineManagerSourceCurrent(source) || !b.engineMgrPairCurrent(worker, source) {
		t.Fatal("cleanup readback has no current manager/source pair")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	statuses, err := pauseEngineInventory(ctx, worker)
	if err != nil {
		t.Fatalf("read physical shutdown proof: %v", err)
	}
	raw, err := callEngineOn(ctx, worker, "engine:models", nil)
	if err != nil {
		t.Fatalf("read manager-owned model cleanup: %v", err)
	}
	var models struct {
		LoadedByEngine map[string][]string `json:"loadedByEngine"`
	}
	if err := json.Unmarshal(raw, &models); err != nil {
		t.Fatalf("decode model cleanup: %v", err)
	}
	for _, name := range names {
		status, known := statuses[name]
		if !known || status.Running || !status.TerminationVerified {
			t.Fatalf("engine %s cleanup proof=%+v known=%v", name, status, known)
		}
		loaded, known := models.LoadedByEngine[name]
		if !known || len(loaded) != 0 {
			t.Fatalf("engine %s residency=%v known=%v; want positive empty readback", name, loaded, known)
		}
	}
}

func TestPauseAlwaysStopsAndUnloadsDespiteLegacyOptions(t *testing.T) {
	test := func(name string, origin nodepolicy.PauseOrigin) {
		t.Run(name, func(t *testing.T) {
			b, _, _, engine := pausingBroker(t, nodepolicy.PauseConfig{OnActive: nodepolicy.FinishActive, StopEngines: false, UnloadModels: false})
			// The second engine and its resident model have not reached broker
			// notifications. The manager owns both cleanup and its readback.
			running := map[string]bool{"ollama": true, "llamacpp": true}
			loaded := map[string][]string{"ollama": {"m1"}, "llamacpp": {"fresh-model"}}
			engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				switch method {
				case "engine:get-installed":
					return map[string]any{"engines": []any{
						map[string]any{"engine": "ollama", "running": running["ollama"], "terminationVerified": !running["ollama"]},
						map[string]any{"engine": "llamacpp", "running": running["llamacpp"], "terminationVerified": !running["llamacpp"]},
					}}, ""
				case "engine:models":
					return map[string]any{"loadedByEngine": loaded}, ""
				case engineSleepMethod:
					var p struct {
						Engine string `json:"engine"`
					}
					if err := json.Unmarshal(raw, &p); err != nil {
						return nil, err.Error()
					}
					running[p.Engine] = false
					loaded[p.Engine] = []string{}
					return map[string]any{"engine": p.Engine, "running": false, "terminationVerified": true}, ""
				}
				return map[string]bool{"ok": true}, ""
			})
			if state, err := b.setAvailability(context.Background(), nodepolicy.Paused, origin); err != nil || state != nodepolicy.Paused {
				t.Fatalf("pause = %s, %v", state, err)
			}
			unloaded, slept := make(map[string][]string), make(map[string]bool)
			for _, call := range pauseLifecycleCalls(engine.snapshot()) {
				var p struct {
					Engine string `json:"engine"`
					Model  string `json:"model"`
				}
				if err := json.Unmarshal(call.params, &p); err != nil {
					t.Fatalf("decode lifecycle call: %v", err)
				}
				if call.method == engineSleepMethod {
					slept[p.Engine] = true
				} else {
					unloaded[p.Engine] = append(unloaded[p.Engine], p.Model)
				}
			}
			if len(unloaded) != 0 || !reflect.DeepEqual(slept, map[string]bool{"ollama": true, "llamacpp": true}) {
				t.Fatalf("broker unloads=%v sleeps=%v, want manager-owned cleanup of cached and freshly discovered engines", unloaded, slept)
			}
			requirePauseManagerCleanup(t, b, "ollama", "llamacpp")
			requirePolicyIntent(t, b, map[string]bool{"ollama": true})
			for _, call := range engine.snapshot() {
				if call.method == "engine:stop" {
					t.Fatal("pause rewrote engine intent with an ordinary stop")
				}
			}
		})
	}
	test("local pause", nodepolicy.PausedLocally)
	test("peer pause", nodepolicy.PausedByPeer)
}

func TestPauseStopFailureRetainsFenceAndCanRetryLocally(t *testing.T) {
	b, _, proxy, engine := pausingBroker(t, nodepolicy.PauseConfig{OnActive: nodepolicy.FinishActive})
	fail, running := true, map[string]bool{"ollama": true, "llamacpp": true}
	engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:get-installed":
			return map[string]any{"engines": []any{
				map[string]any{"engine": "ollama", "running": running["ollama"]},
				map[string]any{"engine": "llamacpp", "running": running["llamacpp"]},
			}}, ""
		case "engine:models":
			return map[string]any{"loadedByEngine": map[string][]string{}}, ""
		case engineUnloadModelMethod:
			return nil, "unload failed"
		case engineSleepMethod:
			var p struct {
				Engine string `json:"engine"`
			}
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if p.Engine == "llamacpp" && fail {
				return nil, "external process cannot be stopped"
			}
			running[p.Engine] = false
			return map[string]any{"engine": p.Engine, "running": false, "terminationVerified": true}, ""
		}
		return map[string]bool{"ok": true}, ""
	})
	state, err := b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedByPeer)
	if state != nodepolicy.Draining || policyErrorCode(err) != -32000 || !strings.HasPrefix(err.Error(), "Pause could not stop every engine: ") || len(err.Error()) > 560 {
		t.Fatalf("incomplete pause = %s, %v", state, err)
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	blocked, proxyErr := r.managementBlockedLocked(), r.availabilityErr
	r.mu.Unlock()
	if !blocked || proxyErr != "" || b.localPolicy().Policy.Availability != nodepolicy.Paused || running["ollama"] {
		t.Fatalf("failed pause lost fence or stopped trying: blocked=%v proxyErr=%q running=%v", blocked, proxyErr, running)
	}
	pushes := availabilityPushes(proxy.snapshot())
	if last := pushes[len(pushes)-1]; last.State != nodepolicy.Draining {
		t.Fatalf("failure advertised %+v", last)
	}
	for _, next := range []nodepolicy.Availability{nodepolicy.Available, nodepolicy.Paused} {
		_, err := b.setAvailability(context.Background(), next, nodepolicy.PausedByPeer)
		if policyErrorCode(err) != codeLocalVeto {
			t.Fatalf("peer retry/resume = %v", err)
		}
	}
	fail = false
	if state, err := b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil || state != nodepolicy.Paused {
		t.Fatalf("local retry = %s, %v", state, err)
	}
	if running["ollama"] || running["llamacpp"] {
		t.Fatalf("retry left engines running: %v", running)
	}
}

func TestRepeatedCompletedPauseKeepsRevisionAndRechecksShutdown(t *testing.T) {
	b, _, _, engine := pausingBroker(t, nodepolicy.PauseConfig{OnActive: nodepolicy.FinishActive})
	if state, err := b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil || state != nodepolicy.Paused {
		t.Fatalf("initial pause = %s, %v", state, err)
	}
	revision := b.localPolicy().Policy.Revision
	before := len(engine.methods(engineSleepMethod))
	if state, err := b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil || state != nodepolicy.Paused {
		t.Fatalf("repeated pause = %s, %v", state, err)
	}
	if got := b.localPolicy().Policy.Revision; got != revision {
		t.Fatalf("repeated completed pause revision = %s, want %s", got, revision)
	}
	if got := len(engine.methods(engineSleepMethod)); got <= before {
		t.Fatalf("repeated pause did not recheck physical shutdown: sleep calls %d -> %d", before, got)
	}
}

func TestPauseReadbackCannotCrossManagerReplacement(t *testing.T) {
	b, _, _, engine := pausingBroker(t, nodepolicy.PauseConfig{OnActive: nodepolicy.FinishActive})
	t.Cleanup(engine.releaseHeld)
	base := engine.answer
	first := true
	engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, errText := base(method, raw)
		if method == "engine:get-installed" && first {
			first = false
			return result, holdSuccessReply
		}
		return result, errText
	})
	done := make(chan error, 1)
	go func() {
		_, err := b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally)
		done <- err
	}()
	engine.waitFor("first pause inventory", func(calls []policyCall) bool {
		for _, call := range calls {
			if call.method == "engine:get-installed" {
				return true
			}
		}
		return false
	})
	replacement := attachFakeEngine(t, b)
	engine.releaseHeld()
	if err := receive(t, "pause against replaced manager", done); err == nil || !strings.Contains(err.Error(), "engine shutdown readback unavailable") {
		t.Fatalf("replaced-manager pause = %v, want unverified shutdown", err)
	}
	if got := b.localPolicy().Availability; got != nodepolicy.Draining {
		t.Fatalf("replaced-manager availability = %s, want draining", got)
	}
	if got := replacement.snapshot(); len(got) != 0 {
		t.Fatalf("pause mixed replacement calls into the old inventory: %+v", got)
	}
}

func TestPauseReplaySkipsRestoreForEveryOrigin(t *testing.T) {
	test := func(name string, origin nodepolicy.PauseOrigin, recovery bool) {
		t.Run(name, func(t *testing.T) {
			b, _ := newPolicyBroker(t)
			path := filepath.Join(t.TempDir(), nodepolicy.FileName)
			if recovery {
				if err := os.WriteFile(path, []byte("invalid policy"), 0o600); err != nil {
					t.Fatal(err)
				}
			} else {
				policy := nodepolicy.Default()
				policy.Availability, policy.PausedBy = nodepolicy.Paused, origin
				writePolicyFile(t, path, policy)
			}
			b.loadNodePolicy(path)
			check := func(engine *fakeWorker) {
				t.Helper()
				b.publishNodeProjection()
				b.restoreEnabledEngines(b.getEngineMgr())
				paProjectionBarrier(t, b)
				if got := engine.methods(restoreEnabledEnginesMethod); len(got) != 0 {
					t.Fatalf("paused replay restored engines: %v", got)
				}
				engine.mu.Lock()
				raw := append(json.RawMessage(nil), engine.projections[len(engine.projections)-1].params...)
				engine.mu.Unlock()
				var projection nodeProjection
				if err := json.Unmarshal(raw, &projection); err != nil {
					t.Fatal(err)
				}
				if !projection.ManagementBlocked || projection.PausedLocally != (origin != nodepolicy.PausedByPeer) {
					t.Fatalf("paused replay projection = %+v", projection)
				}
			}
			check(attachFakeEngine(t, b))
			check(attachFakeEngine(t, b))
		})
	}
	test("stored local pause", nodepolicy.PausedLocally, false)
	test("stored peer pause", nodepolicy.PausedByPeer, false)
	test("startup recovery", "", true)
}

// A lost/refused Sleep response is satisfied only by positive stop evidence
// in the final inventory from the same worker.
func TestPauseIgnoresSleepRefusalWithTerminationEvidence(t *testing.T) {
	b, _, _, engine := pausingBroker(t, nodepolicy.PauseConfig{OnActive: nodepolicy.FinishActive})
	engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:descriptors":
			return json.RawMessage(`{"engines":[{"version":1,"engine":"custom","available":false}]}`), ""
		case "engine:get-installed":
			return map[string]any{"engines": []any{
				map[string]any{"engine": "ollama", "running": false},
				map[string]any{"engine": "custom", "running": false, "terminationVerified": true},
			}}, ""
		case "engine:models":
			return map[string]any{"loadedByEngine": map[string][]string{}}, ""
		case engineSleepMethod:
			var p struct {
				Engine string `json:"engine"`
			}
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if p.Engine == "custom" {
				return nil, "engine has no platform support on this host"
			}
			return map[string]any{"engine": p.Engine, "running": false, "terminationVerified": true}, ""
		}
		return map[string]bool{"ok": true}, ""
	})
	state, err := b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally)
	if err != nil || state != nodepolicy.Paused {
		t.Fatalf("pause = %s, %v; want paused despite the unsupported engine's sleep refusal", state, err)
	}
}

func requireUnverifiedPauseFence(t *testing.T, b *Broker, proxy *fakeWorker, out *syncBuffer) {
	t.Helper()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	live, blocked, proxyErr := r.live, r.managementBlockedLocked(), r.availabilityErr
	r.mu.Unlock()
	if live != nodepolicy.Draining || !blocked || proxyErr != "" || b.localPolicy().Policy.Availability != nodepolicy.Paused {
		t.Fatalf("unverified shutdown lost its fence: live=%s blocked=%v proxyErr=%q policy=%+v", live, blocked, proxyErr, b.localPolicy().Policy)
	}
	pushes := availabilityPushes(proxy.snapshot())
	if len(pushes) == 0 || pushes[len(pushes)-1].State != nodepolicy.Draining {
		t.Fatalf("unverified shutdown did not advertise draining: %+v", pushes)
	}
	for _, push := range pushes {
		if push.State == nodepolicy.Paused {
			t.Fatalf("unverified shutdown published paused to proxy: %+v", push)
		}
	}
	var announcements int
	for _, frame := range out.frames() {
		if frame.Method != notifyAvailabilityChanged {
			continue
		}
		var announcement struct {
			Availability nodepolicy.Availability `json:"availability"`
		}
		if err := json.Unmarshal(frame.Params, &announcement); err != nil {
			t.Fatalf("decode availability announcement: %v", err)
		}
		if announcement.Availability != nodepolicy.Draining {
			t.Fatalf("unverified shutdown announced %s", announcement.Availability)
		}
		announcements++
	}
	if announcements == 0 {
		t.Fatal("unverified shutdown did not announce draining to clients")
	}
}

func TestPauseFailedSleepRequiresTerminationProof(t *testing.T) {
	const stopped = `{"engines":[{"engine":"ollama","running":false}]}`
	const unsupported = `{"engines":[{"version":1,"engine":"ollama","available":false}]}`
	for _, tc := range []struct {
		name           string
		inventory      string
		inventoryError string
		descriptors    string
		wantDetail     string
	}{
		{name: "unsupported engine without physical proof", inventory: stopped, descriptors: unsupported, wantDetail: "ollama: stop unverified"},
		{name: "negative physical proof", inventory: `{"engines":[{"engine":"ollama","running":false,"terminationVerified":false}]}`, descriptors: unsupported, wantDetail: "ollama: stop unverified"},
		{name: "null physical proof", inventory: `{"engines":[{"engine":"ollama","running":false,"terminationVerified":null}]}`, descriptors: unsupported, wantDetail: "ollama: stop unverified"},
		{name: "malformed physical proof", inventory: `{"engines":[{"engine":"ollama","running":false,"terminationVerified":"true"}]}`, descriptors: unsupported, wantDetail: "engine shutdown readback unavailable"},
		{name: "positive proof cannot override running", inventory: `{"engines":[{"engine":"ollama","running":true,"terminationVerified":true}]}`, wantDetail: "ollama: still running"},
		{name: "failed reclaim with surviving PID", inventory: stopped, descriptors: `{"engines":[{"version":1,"engine":"ollama","available":true}]}`, wantDetail: "ollama: stop unverified"},
		{name: "final inventory unavailable", inventoryError: "inventory unavailable", descriptors: unsupported, wantDetail: "engine shutdown readback unavailable"},
		{name: "final inventory missing engines", inventory: `{}`, descriptors: unsupported, wantDetail: "engine shutdown readback unavailable"},
		{name: "final inventory truncated entry", inventory: `{"engines":[{}]}`, descriptors: unsupported, wantDetail: "engine shutdown readback unavailable"},
		{name: "final inventory missing running", inventory: `{"engines":[{"engine":"ollama"}]}`, descriptors: unsupported, wantDetail: "engine shutdown readback unavailable"},
		{name: "final inventory null running", inventory: `{"engines":[{"engine":"ollama","running":null}]}`, descriptors: unsupported, wantDetail: "engine shutdown readback unavailable"},
		{name: "final inventory omits failed engine", inventory: `{"engines":[]}`, descriptors: unsupported, wantDetail: "ollama: stop unverified"},
		{name: "final inventory duplicate engine", inventory: `{"engines":[{"engine":"ollama","running":true},{"engine":"ollama","running":false}]}`, descriptors: unsupported, wantDetail: "engine shutdown readback unavailable"},
		{name: "unsupported descriptor cannot override running", inventory: `{"engines":[{"engine":"ollama","running":true}]}`, descriptors: unsupported, wantDetail: "ollama: still running"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, out, proxy, engine := pausingBroker(t, nodepolicy.PauseConfig{OnActive: nodepolicy.FinishActive})
			reads := 0
			engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
				switch method {
				case "engine:get-installed":
					reads++
					if reads == 1 {
						return json.RawMessage(stopped), ""
					}
					return json.RawMessage(tc.inventory), tc.inventoryError
				case "engine:descriptors":
					return json.RawMessage(tc.descriptors), ""
				case "engine:models":
					return map[string]any{"loadedByEngine": map[string][]string{}}, ""
				case engineSleepMethod:
					return nil, `cannot stop engine "ollama": orphan termination could not be verified`
				}
				return map[string]bool{"ok": true}, ""
			})
			state, err := b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally)
			if err == nil || state != nodepolicy.Draining || policyErrorCode(err) != -32000 || !strings.HasPrefix(err.Error(), "Pause could not stop every engine: ") || !strings.Contains(err.Error(), tc.wantDetail) {
				t.Fatalf("unverified pause = %s, %v; want draining/-32000 with %q", state, err, tc.wantDetail)
			}
			if reads != 2 || len(engine.methods(engineSleepMethod)) != 1 {
				t.Fatalf("shutdown proof skipped lifecycle/readback: inventories=%d calls=%v", reads, engine.methods(""))
			}
			requireUnverifiedPauseFence(t, b, proxy, out)
		})
	}
}
