// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"testing"

	settings "nvpair-shared/enginesettings"
)

// Pause at the production decode/accept boundary without modifying the pump.
func pauseIdentityAcceptance(b *Broker, source *engineManagerSource, params json.RawMessage) (<-chan struct{}, chan<- struct{}, <-chan struct{}) {
	decoded, resume, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	go func() {
		snapshot := decodeModelIdentity(params)
		close(decoded)
		<-resume
		b.acceptModelIdentity(source, snapshot)
		close(done)
	}()
	return decoded, resume, done
}

func requireIncarnationIdentity(t *testing.T, b *Broker, want map[string]engineManagerIdentity, ready bool) {
	t.Helper()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	got := r.modelIdentityLocked()
	r.mu.Unlock()
	if len(got.Engines) != len(want) {
		t.Fatalf("joined identity = %+v, want engines %+v", got, want)
	}
	for engine, id := range want {
		current := got.Engines[engine]
		if current.EngineRunID != id.EngineRunID || current.LaunchDigest != id.LaunchDigest || !reflect.DeepEqual(current.ModelDigests, id.ModelDigests) {
			t.Fatalf("joined identity for %s = %+v, want %+v", engine, current, id)
		}
	}
	if got := b.StrictReadiness().Capability != nil; got != ready {
		t.Fatalf("strict readiness present = %v, want %v", got, ready)
	}
}

func TestEngineManagerIdentityIncarnationRejectsCallbackAfterRetirement(t *testing.T) {
	f := newStrictFixture(t)
	source := f.b.activateEngineManagerSource()
	seedStrictIdentity(f.b, "ollama")
	f.advertise(t, strictCapability(t, "ollama", "m:latest"))
	decoded, resume, done := pauseIdentityAcceptance(f.b, source, engineManagerSnapshot("ollama"))
	<-decoded
	f.b.retireEngineManagerSource(source)
	requireIncarnationIdentity(t, f.b, nil, false)
	close(resume)
	<-done
	requireIncarnationIdentity(t, f.b, nil, false)
}

func TestEngineManagerIdentityIncarnationPreservesReplacement(t *testing.T) {
	f := newStrictFixture(t)
	old := f.b.activateEngineManagerSource()
	seedStrictIdentity(f.b, "ollama")
	f.advertise(t, strictCapability(t, "ollama", "m:latest"))
	decoded, resume, done := pauseIdentityAcceptance(f.b, old, engineManagerSnapshot("ollama"))
	<-decoded
	f.b.retireEngineManagerSource(old)
	current := f.b.activateEngineManagerSource()
	replacement := decodeModelIdentity(engineManagerSnapshot("ollama"))
	id := replacement["ollama"]
	id.EngineRunID = "engine-run-1"
	replacement["ollama"] = id
	f.b.acceptModelIdentity(current, replacement)
	requireIncarnationIdentity(t, f.b, replacement, true)
	close(resume)
	<-done
	requireIncarnationIdentity(t, f.b, replacement, true)
	f.b.retireEngineManagerSource(old)
	requireIncarnationIdentity(t, f.b, replacement, true)
	f.b.forwardEngineNotificationFrom(old, engineStateChangedMethod, json.RawMessage(`{"engine":"ollama","running":false}`))
	requireIncarnationIdentity(t, f.b, replacement, true)
}

func TestEngineManagerIdentityIncarnationAcceptsEarlyNotification(t *testing.T) {
	ab := newActivatedBroker(t)
	source := ab.b.activateEngineManagerSource()
	// No engine-manager handle is installed: this models the pump running
	// before startRPCWorker returns to spawnEngineMgr.
	ab.b.observeSettingsIdentity([]settings.Snapshot{appliedSnapshot("ollama", 1)})
	ab.b.forwardEngineNotificationFrom(source, engineModelIdentityMethod, engineManagerSnapshot("ollama"))
	r := ab.b.nodePolicyRuntime()
	r.mu.Lock()
	got := r.engineIdentity["ollama"]
	r.mu.Unlock()
	if got.EngineRunID != "run-ollama" {
		t.Fatalf("early notification identity = %+v", got)
	}
	requireIncarnationIdentity(t, ab.b, decodeModelIdentity(engineManagerSnapshot("ollama")), false)
}

func TestEngineManagerIdentityIncarnationActivationWithdrawsPredecessor(t *testing.T) {
	f := newStrictFixture(t)
	old := f.b.activateEngineManagerSource()
	seedStrictIdentity(f.b, "ollama")
	f.advertise(t, strictCapability(t, "ollama", "m:latest"))
	f.b.activateEngineManagerSource()
	requireIncarnationIdentity(t, f.b, nil, false)
	f.b.observeModelIdentity(old, engineManagerSnapshot("ollama"))
	requireIncarnationIdentity(t, f.b, nil, false)
}

func TestEngineManagerIdentityIncarnationRetiresFailedStartup(t *testing.T) {
	ab := newActivatedBroker(t)
	ab.b.engineMgrPath = filepath.Join(t.TempDir(), "missing-engine-manager.exe")
	if _, err := ab.b.spawnEngineMgr(); err == nil {
		t.Fatal("starting a nonexistent executable succeeded")
	}
	r := ab.b.nodePolicyRuntime()
	r.mu.Lock()
	source, identities := r.engineManagerSource, len(r.engineIdentity)
	r.mu.Unlock()
	if source != nil || identities != 0 {
		t.Fatalf("failed startup retained source %p and %d identities", source, identities)
	}
}

func TestEngineManagerIdentityIncarnationCurrentStopWithdraws(t *testing.T) {
	f := newStrictFixture(t)
	source := f.b.activateEngineManagerSource()
	seedStrictIdentity(f.b, "ollama")
	f.advertise(t, strictCapability(t, "ollama", "m:latest"))
	f.b.forwardEngineNotificationFrom(source, engineStateChangedMethod, json.RawMessage(`{"engine":"ollama","running":false}`))
	requireIncarnationIdentity(t, f.b, nil, false)
}
