// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sync"
	"testing"

	"nvpair-ui-broker/relay"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func TestRecoveringManagerDiscoverySubscribeReceivesDirectory(t *testing.T) {
	b := &Broker{relayDir: relay.NewDirectory()}
	engine := attachFakeEngine(t, b)
	b.workersMu.Lock()
	b.engineMgrRecovering = true
	b.workersMu.Unlock()
	peer := noderec.DirectoryNode{HostUUID: "bootstrap-serving", Services: map[noderec.ServiceKey]noderec.ServiceStatus{
		noderec.ServiceEngineControl: {Port: 14322},
	}}
	b.relayDir.Apply(noderec.NotifyNodeDiscovered, peer)
	params, err := json.Marshal(noderec.SubscribeParams{Services: []noderec.ServiceKey{noderec.ServiceEngineControl}})
	if err != nil {
		t.Fatal(err)
	}
	b.forwardEngineNotification(noderec.MethodSubscribe, params)
	t.Cleanup(b.clearEngineRelaySub)
	calls := engine.waitFor("recovery directory snapshot", func(calls []policyCall) bool { return len(calls) > 0 })
	var snapshot noderec.GetNodesResult
	if calls[0].method != noderec.NotifyNodes || !calls[0].notify {
		t.Fatalf("directory delivery = %+v", calls[0])
	}
	if err := json.Unmarshal(calls[0].params, &snapshot); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(snapshot.Nodes, []noderec.DirectoryNode{peer}) {
		t.Fatalf("directory = %+v, want %+v", snapshot.Nodes, peer)
	}
	if b.engineRelaySubID == 0 || b.getEngineMgr() != nil {
		t.Fatal("subscription must be registered while forward access remains hidden")
	}
}

func TestRecoveringManagerReadyReceivesStateProjections(t *testing.T) {
	b := &Broker{codec: NewCodec(&syncBuffer{})}
	engine := attachFakeEngine(t, b)
	b.workersMu.Lock()
	b.engineMgrRecovering = true
	b.workersMu.Unlock()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.localPauseRequested = true
	r.mu.Unlock()
	want := settings.Snapshot{Engine: "ollama", Revision: 7}
	b.engineSettingsLoaded = true
	b.engineSettings = map[string]*engineSettingsRecord{"ollama": {Snapshot: want}}
	b.forwardEngineNotification("engine:ready", nil)
	waitProjections(t, engine, 1)
	engine.mu.Lock()
	raw := append(json.RawMessage(nil), engine.projections[0].params...)
	engine.mu.Unlock()
	var projection nodeProjection
	if err := json.Unmarshal(raw, &projection); err != nil {
		t.Fatal(err)
	}
	if !projection.PausedLocally {
		t.Fatal("recovering manager did not receive pending local pause")
	}
	calls := engine.waitFor("recovery settings projection", func(calls []policyCall) bool {
		for _, call := range calls {
			if call.method == "engine:settings-projection" {
				return true
			}
		}
		return false
	})
	for _, call := range calls {
		if call.method != "engine:settings-projection" {
			continue
		}
		var snapshots []settings.Snapshot
		if err := json.Unmarshal(call.params, &snapshots); err != nil {
			t.Fatal(err)
		}
		want.Sequence++
		if !call.notify || !reflect.DeepEqual(snapshots, []settings.Snapshot{want}) {
			t.Fatalf("settings projection = %+v, want %+v", snapshots, want)
		}
	}
	if b.getEngineMgr() != nil {
		t.Fatal("ready notification exposed recovering manager to forward callers")
	}
}

// nodeProjections lists every engine:node-projection the fake engine-manager
// received, in order.
func nodeProjections(t *testing.T, engine *fakeWorker) []nodepolicy.Advertisement {
	t.Helper()
	engine.mu.Lock()
	calls := append([]policyCall(nil), engine.projections...)
	engine.mu.Unlock()
	var out []nodepolicy.Advertisement
	for _, c := range calls {
		if !c.notify {
			t.Fatalf("%s was sent as a request", engineNodeProjectionMethod)
		}
		var p nodeProjection
		if err := json.Unmarshal(c.params, &p); err != nil {
			t.Fatalf("decode %s %s: %v", engineNodeProjectionMethod, c.params, err)
		}
		out = append(out, p.Policy)
	}
	return out
}

// waitProjections waits until engine-manager has received n projections and
// returns them.
func waitProjections(t *testing.T, engine *fakeWorker, n int) []nodepolicy.Advertisement {
	t.Helper()
	engine.waitFor("node projections", func([]policyCall) bool { return len(nodeProjections(t, engine)) >= n })
	return nodeProjections(t, engine)
}

// requireLastProjectionIsStored fails unless the newest projection is the
// advertisement of the stored policy.
func requireLastProjectionIsStored(t *testing.T, b *Broker, got []nodepolicy.Advertisement) {
	t.Helper()
	if len(got) == 0 {
		t.Fatal("no node projection was published")
	}
	if want := b.localPolicy().Policy.Advertisement(); !reflect.DeepEqual(got[len(got)-1], want) {
		t.Fatalf("last node projection = %+v, want the stored policy's %+v", got[len(got)-1], want)
	}
}

func TestNodeProjectionPublishedOnPersistAvailabilityAndReady(t *testing.T) {
	ab := newActivatedBroker(t)
	engine := attachFakeEngine(t, ab.b)

	_, err := ab.b.dispatchPolicy(context.Background(), methodPolicySet, policyParams{
		Policy: mustJSON(map[string]any{
			"version":  nodepolicy.SchemaVersion,
			"roles":    map[string]any{"weak": map[string]any{"engine": "ollama", "model": "qwen3:4b"}},
			"profiles": []any{map[string]any{"name": "qwen-small", "engine": "ollama", "model": "qwen3:4b", "contextTokens": 8192, "capabilities": []string{"tools"}}},
		}),
		ExpectedPersistedRevision: mustJSON(ab.b.localPolicy().Policy.Revision),
	})
	if err != nil {
		t.Fatalf("policy:set: %v", err)
	}
	got := waitProjections(t, engine, 1)
	want := nodepolicy.Advertisement{
		Revision: ab.b.localPolicy().Policy.Revision,
		Roles: map[string]nodepolicy.AdvertisedRole{
			nodepolicy.TierWeak: {Engine: "ollama", Model: "qwen3:4b", ContextTokens: 8192, Capabilities: []nodepolicy.Capability{nodepolicy.CapTools}},
		},
	}
	if !reflect.DeepEqual(got[0], want) {
		t.Fatalf("projection after policy:set = %+v, want %+v", got[0], want)
	}

	t.Run("pause", func(t *testing.T) {
		before := len(nodeProjections(t, engine))
		if _, err := ab.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil {
			t.Fatalf("pause: %v", err)
		}
		got := waitProjections(t, engine, before+1)
		requireLastProjectionIsStored(t, ab.b, got)
		if got[len(got)-1].Revision == want.Revision {
			t.Fatalf("the pause projection kept revision %s; the pause persists a new one", want.Revision)
		}
	})
	t.Run("resume", func(t *testing.T) {
		before := len(nodeProjections(t, engine))
		paused := ab.b.localPolicy().Policy.Revision
		if _, err := ab.b.setAvailability(context.Background(), nodepolicy.Available, nodepolicy.PausedLocally); err != nil {
			t.Fatalf("resume: %v", err)
		}
		got := waitProjections(t, engine, before+1)
		requireLastProjectionIsStored(t, ab.b, got)
		if got[len(got)-1].Revision == paused {
			t.Fatalf("the resume projection kept the pause revision %s", paused)
		}
	})
	t.Run("engine-manager ready", func(t *testing.T) {
		before := len(nodeProjections(t, engine))
		ab.b.forwardEngineNotification("engine:ready", json.RawMessage(`{"version":"test"}`))
		requireLastProjectionIsStored(t, ab.b, waitProjections(t, engine, before+1))
	})
	t.Run("engine-manager attached", func(t *testing.T) {
		before := len(nodeProjections(t, engine))
		ab.b.refreshEnginePolicySnapshot(ab.b.getEngineMgr())
		requireLastProjectionIsStored(t, ab.b, waitProjections(t, engine, before+1))
	})
}

func TestNodeProjectionsFollowTheStoredPolicyUnderConcurrentSaves(t *testing.T) {
	ab := newActivatedBroker(t)
	engine := attachFakeEngine(t, ab.b)
	const writers, saves = 6, 4
	var wg sync.WaitGroup
	for w := range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for s := 0; s < saves; {
				_, err := ab.b.dispatchPolicy(context.Background(), methodPolicySet, policyParams{
					Policy:                    mustJSON(map[string]any{"version": nodepolicy.SchemaVersion, "idle": map[string]any{"unloadAfterMinutes": 1 + w*saves + s}}),
					ExpectedPersistedRevision: mustJSON(ab.b.localPolicy().Policy.Revision),
				})
				var conflict *policyConflictError
				switch {
				case err == nil:
					s++
				case errors.As(err, &conflict):
				default:
					t.Errorf("policy:set: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()
	got := waitProjections(t, engine, writers*saves)
	for i := 1; i < len(got); i++ {
		cmp, err := nodepolicy.CompareRevisions(got[i].Revision, got[i-1].Revision)
		if err != nil {
			t.Fatalf("compare projected revisions %q and %q: %v", got[i].Revision, got[i-1].Revision, err)
		}
		if cmp < 0 {
			t.Fatalf("projection %d carried revision %s after revision %s", i, got[i].Revision, got[i-1].Revision)
		}
	}
	requireLastProjectionIsStored(t, ab.b, got)
}

func TestNodeProjectionWithoutEngineManagerIsDropped(t *testing.T) {
	ab := newActivatedBroker(t)
	// No engine-manager: the save must still land; engine:ready or the next
	// attach publishes the projection later.
	if set := ab.setIdle(t, 3); set.Policy.Revision == "" {
		t.Fatal("policy:set without an engine-manager returned no revision")
	}
	engine := attachFakeEngine(t, ab.b)
	ab.b.forwardEngineNotification("engine:ready", json.RawMessage(`{}`))
	requireLastProjectionIsStored(t, ab.b, waitProjections(t, engine, 1))
}
