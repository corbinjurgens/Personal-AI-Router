// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"path/filepath"
	"reflect"
	"testing"

	"nvpair-shared/durablefile"
	settings "nvpair-shared/enginesettings"
)

func seedRunningModelAcceptance(t *testing.T, h *modelSettingsHarness, stage string) *engineSettingsRecord {
	t.Helper()
	h.mu.Lock()
	h.launch.Running = true
	h.mu.Unlock()
	h.seedAccepted(t, true)
	h.b.engineConfigMu.Lock()
	defer h.b.engineConfigMu.Unlock()
	record := h.b.engineSettings["llamacpp"]
	record.Model.Stage, record.Snapshot.CoupledStage = "accepted", "accepted"
	if stage == "acceptance-uncertain" {
		markModelAcceptanceUncertain(record)
		// A previously persisted diagnostic must survive invalidation verbatim.
		record.Snapshot.UnresolvedError += "; uncertainty record durability could not be confirmed"
	}
	if !record.Snapshot.Running {
		t.Fatal("fixture did not retain the running acceptance observation")
	}
	if err := h.b.saveEngineSettingsLocked(); err != nil {
		t.Fatalf("save running acceptance: %v", err)
	}
	return record.clone()
}

func requirePendingModelRunningReadback(t *testing.T, h *modelSettingsHarness, engine *fakeWorker, want *engineSettingsRecord) {
	t.Helper()
	before := len(engine.snapshot())
	policy := h.b.localPolicy().Policy
	got, err := h.b.getEngineSettings(t.Context(), settings.Request{Engine: "llamacpp"}, "")
	if err != nil {
		t.Fatalf("read pending settings: %v", err)
	}
	if got.Running != want.Snapshot.Running {
		t.Fatalf("pending running = %v, want %v", got.Running, want.Snapshot.Running)
	}
	for _, call := range engine.snapshot()[before:] {
		if !call.notify || call.method != "engine:settings-projection" {
			t.Fatalf("pending read invoked worker method %s (notification=%v)", call.method, call.notify)
		}
	}
	if h.calls() != 0 || !reflect.DeepEqual(policy, h.b.localPolicy().Policy) {
		t.Fatal("pending read configured the engine or changed policy")
	}
	actual := h.record(t)
	// Loading renews journal metadata; publication advances its sequence.
	expected := want.clone()
	expected.Snapshot.NodeID = actual.Snapshot.NodeID
	expected.Snapshot.Epoch = actual.Snapshot.Epoch
	expected.Snapshot.Sequence = actual.Snapshot.Sequence
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("pending record changed beyond running and journal metadata:\ngot  %+v\nwant %+v", actual, expected)
	}
	expected.Snapshot.Sequence = got.Sequence
	if !reflect.DeepEqual(got, expected.Snapshot) {
		t.Fatalf("pending snapshot = %+v, want %+v", got, expected.Snapshot)
	}
}

func TestModelSettingsRecoveryInvalidatesPreviousRunning(t *testing.T) {
	test := func(stage string) {
		t.Run(stage, func(t *testing.T) {
			h := newModelSettingsHarness(t)
			want := seedRunningModelAcceptance(t, h, stage)
			policy := h.b.localPolicy().Policy
			h.restartJournal(t)
			h.mu.Lock()
			h.launch.Running = false
			h.mu.Unlock()
			_, engine := attachBoundEngineManager(t, h.b, h.b.activateEngineManagerSource())
			engine.setAnswer(h.answer)
			// Source activation does not clear this separate cache. It cannot
			// validate the old settings observation for the new incarnation.
			r := h.b.nodePolicyRuntime()
			r.mu.Lock()
			r.running["llamacpp"] = true
			r.mu.Unlock()
			var launch settings.LaunchState
			if err := h.b.settingsWorkerCall(t.Context(), "engine:get-launch", map[string]string{"engine": "llamacpp"}, &launch); err != nil {
				t.Fatalf("read replacement manager baseline: %v", err)
			}
			if launch.Running {
				t.Fatal("replacement manager baseline is running")
			}
			h.b.engineConfigMu.Lock()
			err := h.b.loadEngineSettingsLocked()
			h.b.engineConfigMu.Unlock()
			if err != nil {
				t.Fatalf("load recovered journal: %v", err)
			}
			if stage == "accepted" {
				markModelAcceptanceUncertain(want)
			}
			want.Snapshot.Running = false
			requirePendingModelRunningReadback(t, h, engine, want)
			if !reflect.DeepEqual(policy, h.b.localPolicy().Policy) {
				t.Fatal("journal recovery changed policy")
			}
		})
	}
	test("accepted")
	test("acceptance-uncertain")
}

func TestModelSettingsManagerReplacementInvalidatesPendingRunning(t *testing.T) {
	h := newModelSettingsHarness(t)
	want := seedRunningModelAcceptance(t, h, "acceptance-uncertain")
	policy := h.b.localPolicy().Policy
	// Exercise the production spawn invalidation boundary without starting a
	// process, following the existing failed-start incarnation test contract.
	h.b.engineMgrPath = filepath.Join(t.TempDir(), "missing-engine-manager.exe")
	h.b.engineMgrControlHost = "127.0.0.1"
	if _, err := h.b.spawnEngineMgr(); err == nil {
		t.Fatal("starting a nonexistent executable succeeded")
	}
	h.mu.Lock()
	h.launch.Running = false
	h.mu.Unlock()
	_, engine := attachBoundEngineManager(t, h.b, h.b.activateEngineManagerSource())
	engine.setAnswer(h.answer)
	if actual := h.record(t); actual.Snapshot.Epoch != want.Snapshot.Epoch || !h.b.engineSettingsLoaded {
		t.Fatal("manager replacement reloaded the broker journal")
	}
	want.Snapshot.Running = false
	requirePendingModelRunningReadback(t, h, engine, want)
	if !reflect.DeepEqual(policy, h.b.localPolicy().Policy) {
		t.Fatal("manager replacement changed policy")
	}
}

func TestModelSettingsInProcessAcceptanceUncertaintyRetainsRunning(t *testing.T) {
	h := newModelSettingsHarness(t)
	h.mu.Lock()
	h.launch.Running = true
	h.mu.Unlock()
	p := h.review(t, h.request(t, true), "")
	policy := h.b.localPolicy().Policy
	h.b.engineSettingsReplacer = durablefile.Replacer{SyncDir: func(string) error {
		return errors.New("acceptance durability is uncertain")
	}}
	if _, err := h.b.applyEngineSettings(t.Context(), p, ""); !errors.Is(err, durablefile.ErrReplaced) {
		t.Fatalf("acceptance uncertainty = %v, want ErrReplaced", err)
	}
	want := h.record(t)
	if want.Model.Stage != "acceptance-uncertain" || !want.Snapshot.Running || want.Receipts[p.RequestID].Phase != "failed" {
		t.Fatalf("same-incarnation acceptance lost its running observation: %+v", want)
	}
	requirePendingModelRunningReadback(t, h, h.engine, want)
	h.mu.Lock()
	running := h.launch.Running
	h.mu.Unlock()
	if !running || !reflect.DeepEqual(policy, h.b.localPolicy().Policy) {
		t.Fatal("uncertain acceptance changed the native runtime or policy")
	}
}
