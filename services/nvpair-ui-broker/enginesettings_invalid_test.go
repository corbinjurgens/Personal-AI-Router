// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

func readSettingsJournal(t *testing.T, h *modelSettingsHarness) map[string]json.RawMessage {
	t.Helper()
	path, err := h.b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var entries map[string]json.RawMessage
	if err := json.Unmarshal(data, &entries); err != nil {
		t.Fatal(err)
	}
	return entries
}

func writeSettingsJournal(t *testing.T, h *modelSettingsHarness, entries map[string]json.RawMessage) {
	t.Helper()
	path, err := h.b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(entries)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func otherSettingsEngine(t *testing.T, h *modelSettingsHarness) string {
	t.Helper()
	for _, profile := range h.b.frontedEngineProfiles() {
		if profile.Name != "llamacpp" {
			return profile.Name
		}
	}
	t.Skip("no second fronted engine in this build")
	return ""
}

func idlePolicyEdit(t *testing.T, h *modelSettingsHarness, minutes int) error {
	t.Helper()
	current := h.b.localPolicy().Policy
	_, err := h.b.setLocalPolicy(settingsJSON(map[string]any{"version": nodepolicy.SchemaVersion, "profiles": current.Profiles, "idle": map[string]any{"unloadAfterMinutes": minutes}}), settingsJSON(current.Revision))
	return err
}

// The phase 07 F2 corruption on one engine refused settings for every engine,
// ordinary policy saves and fresh supersession, with no repair in the app.
func TestInvalidSettingsEntryIsContainedAndResetRecovers(t *testing.T) {
	h, fresh, _ := reviewModelSettingsConversionRepair(t)
	h.b.engineConfigMu.Lock()
	op := h.b.engineSettings["llamacpp"].Model
	op.ConversionProfiles = []string{"edited"}
	delete(op.Review.Identities, "sibling")
	err := h.b.saveEngineSettingsLocked()
	h.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	other := otherSettingsEngine(t, h)
	entries := readSettingsJournal(t, h)
	corrupt := entries["llamacpp"]
	// A valid record for another engine must stay usable.
	otherRecord := json.RawMessage(`{"snapshot":{"engine":"` + other + `","revision":1,"appliedRevision":1,"phase":"idle","settings":{"serverPort":19080,"proxyPort":19081}},"previous":{},"resume":false,"explicit":false,"receipts":{}}`)
	entries[other] = otherRecord
	writeSettingsJournal(t, h, entries)
	h.restartJournal(t)
	policyBefore := h.b.localPolicy().Policy

	_, err = h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, "")
	if policyErrorCode(err) != codeSettingsRecordInvalid || !strings.Contains(err.Error(), "reset") {
		t.Fatalf("invalid entry get = %v (code %d), want the reset refusal", err, policyErrorCode(err))
	}
	fresh.ModelChange.PreviewDigest = ""
	if _, err := h.b.previewEngineSettings(context.Background(), fresh, ""); policyErrorCode(err) != codeSettingsRecordInvalid {
		t.Fatalf("fresh supersession preview = %v, want the reset refusal", err)
	}
	// The entry may hold unfinished model work, so it keeps the policy
	// reservation a pending model operation holds, with the actionable error.
	if err := idlePolicyEdit(t, h, 9); policyErrorCode(err) != codeSettingsRecordInvalid {
		t.Fatalf("policy save with an invalid model entry = %v, want the reset refusal", err)
	}
	h.b.engineConfigMu.Lock()
	otherKnown := h.b.engineSettings[other] != nil
	h.b.engineConfigMu.Unlock()
	if !otherKnown {
		t.Fatal("a valid record for another engine was not loaded beside the invalid entry")
	}
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	drained := r.drains["llamacpp"]
	r.mu.Unlock()
	if !drained {
		t.Fatal("an invalid entry with model work did not drain its engine")
	}
	// Saving another engine's record writes the invalid entry back unchanged.
	h.b.engineConfigMu.Lock()
	h.b.engineSettings[other].Snapshot.Revision++
	err = h.b.saveEngineSettingsLocked()
	h.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if saved := readSettingsJournal(t, h)["llamacpp"]; !bytes.Equal(saved, corrupt) {
		t.Fatalf("invalid entry changed on save:\n got %s\nwant %s", saved, corrupt)
	}
	if h.calls() != 1 || !reflect.DeepEqual(h.b.localPolicy().Policy, policyBefore) {
		t.Fatal("refusals changed native or policy state")
	}

	if _, err := h.b.resetSettingsRecord(resetSettingsRecordParams{Engine: other}); policyErrorCode(err) != -32602 {
		t.Fatalf("reset of a valid record = %v, want invalid params", err)
	}
	if _, err := h.b.resetSettingsRecord(resetSettingsRecordParams{Engine: "llamacpp", NodeID: "elsewhere"}); policyErrorCode(err) != -32602 {
		t.Fatalf("reset addressed to another node = %v, want invalid params", err)
	}
	result, err := h.b.resetSettingsRecord(resetSettingsRecordParams{Engine: "llamacpp"})
	if err != nil {
		t.Fatalf("reset invalid entry: %v", err)
	}
	backup, err := os.ReadFile(result.Backup)
	if err != nil {
		t.Fatalf("read reset backup: %v", err)
	}
	var kept struct {
		Engine string          `json:"engine"`
		Cause  string          `json:"cause"`
		Entry  json.RawMessage `json:"entry"`
	}
	if err := json.Unmarshal(backup, &kept); err != nil || kept.Engine != "llamacpp" || !strings.Contains(kept.Cause, "scope is incomplete") || !bytes.Equal(kept.Entry, corrupt) {
		t.Fatalf("reset backup = %s (%v)", backup, err)
	}
	after := readSettingsJournal(t, h)
	if _, exists := after["llamacpp"]; exists {
		t.Fatal("reset left the invalid entry in the journal")
	}
	if _, exists := after[other]; !exists {
		t.Fatal("reset removed another engine's record")
	}
	r.mu.Lock()
	drained = r.drains["llamacpp"]
	r.mu.Unlock()
	if drained {
		t.Fatal("reset left the invalid entry's drain in place")
	}
	if _, err := h.b.resetSettingsRecord(resetSettingsRecordParams{Engine: "llamacpp"}); policyErrorCode(err) != -32602 {
		t.Fatalf("second reset = %v, want invalid params", err)
	}

	// Recovery: settings read the engine's native state afresh, ordinary policy
	// saves proceed and a fresh model change applies.
	snapshot, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, "")
	if err != nil {
		t.Fatalf("settings after reset: %v", err)
	}
	h.mu.Lock()
	native := settings.ConfigFromLaunch(h.launch, 18081)
	h.mu.Unlock()
	if !snapshot.Settings.Equal(native) {
		t.Fatalf("settings after reset = %+v, want native %+v", snapshot.Settings, native)
	}
	if err := idlePolicyEdit(t, h, 9); err != nil {
		t.Fatalf("policy save after reset: %v", err)
	}
	change := h.request(t, false)
	change.ModelChange.ProfileKey = "added"
	change = h.review(t, change, "")
	receipt, err := h.b.applyEngineSettings(context.Background(), change, "")
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("model change after reset = %+v, %v", receipt, err)
	}
}

// An invalid entry without a model operation affects only its own engine:
// node-wide policy saves and the other engine's settings keep working.
func TestInvalidSettingsEntryWithoutModelWorkLeavesPolicyAndOtherEngines(t *testing.T) {
	h := newModelSettingsHarness(t)
	other := otherSettingsEngine(t, h)
	if _, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, ""); err != nil {
		t.Fatal(err)
	}
	entries := readSettingsJournal(t, h)
	broken := json.RawMessage(`{"snapshot":"not an object"}`)
	entries[other] = broken
	writeSettingsJournal(t, h, entries)
	h.restartJournal(t)

	if _, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: other}, ""); policyErrorCode(err) != codeSettingsRecordInvalid {
		t.Fatalf("invalid entry get = %v, want the reset refusal", err)
	}
	if err := idlePolicyEdit(t, h, 11); err != nil {
		t.Fatalf("policy save beside an invalid entry without model work: %v", err)
	}
	change := h.request(t, false)
	change = h.review(t, change, "")
	if receipt, err := h.b.applyEngineSettings(context.Background(), change, ""); err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("model change on another engine = %+v, %v", receipt, err)
	}
	if saved := readSettingsJournal(t, h)[other]; !bytes.Equal(saved, broken) {
		t.Fatalf("invalid entry changed on save: %s", saved)
	}
	h.b.engineConfigMu.Lock()
	fenced := h.b.knownModelSettingsFenceLocked()
	h.b.engineConfigMu.Unlock()
	if fenced {
		t.Fatal("an invalid entry without model work fenced the node")
	}
}

// Review finding: with profiles stored on the broken engine, unrelated policy
// saves must still work; only a change to those profiles is refused.
func TestInvalidSettingsEntryWithCompletedModelKeepsPolicySavesWorking(t *testing.T) {
	h := newModelSettingsHarness(t)
	change := h.review(t, h.request(t, false), "")
	if receipt, err := h.b.applyEngineSettings(context.Background(), change, ""); err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("seed model change = %+v, %v", receipt, err)
	}
	if len(h.b.localPolicy().Policy.Profiles) == 0 {
		t.Fatal("seed change created no profile")
	}
	entries := readSettingsJournal(t, h)
	var record map[string]json.RawMessage
	if err := json.Unmarshal(entries["llamacpp"], &record); err != nil {
		t.Fatal(err)
	}
	var model map[string]json.RawMessage
	if err := json.Unmarshal(record["model"], &model); err != nil {
		t.Fatal(err)
	}
	if string(model["stage"]) != `"completed"` {
		t.Fatalf("seed stage = %s", model["stage"])
	}
	model["before"] = json.RawMessage(`"not-a-digest"`)
	record["model"] = settingsJSON(model)
	entries["llamacpp"] = settingsJSON(record)
	writeSettingsJournal(t, h, entries)
	h.restartJournal(t)

	if _, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, ""); policyErrorCode(err) != codeSettingsRecordInvalid {
		t.Fatalf("invalid entry get = %v, want the reset refusal", err)
	}
	if err := idlePolicyEdit(t, h, 13); err != nil {
		t.Fatalf("unrelated policy save beside a completed invalid entry: %v", err)
	}
	current := h.b.localPolicy().Policy
	profiles := current.Profiles
	profiles[0].DisplayName = "Renamed"
	_, err := h.b.setLocalPolicy(settingsJSON(map[string]any{"version": nodepolicy.SchemaVersion, "profiles": profiles}), settingsJSON(current.Revision))
	if policyErrorCode(err) != codeSettingsRecordInvalid {
		t.Fatalf("profile change on the broken engine = %v, want the reset refusal", err)
	}
}

// Review finding: an unfinished model entry on another engine reserves the
// policy for model changes too, as a valid pending operation would.
func TestInvalidModelEntryReservesPolicyForOtherEnginesModelChanges(t *testing.T) {
	h := newModelSettingsHarness(t)
	other := otherSettingsEngine(t, h)
	if _, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, ""); err != nil {
		t.Fatal(err)
	}
	entries := readSettingsJournal(t, h)
	entries[other] = json.RawMessage(`{"model":{"stage":"accepted"}}`)
	writeSettingsJournal(t, h, entries)
	h.restartJournal(t)
	change := h.request(t, false)
	if _, err := h.b.previewEngineSettings(context.Background(), change, ""); policyErrorCode(err) != codeSettingsRecordInvalid {
		t.Fatalf("model change beside an unfinished invalid entry = %v, want the reset refusal", err)
	}
}
