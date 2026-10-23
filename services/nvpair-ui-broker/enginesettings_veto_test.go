// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrusttest"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

const settingsPeer = "pinned-peer"

// pendingResume leaves the harness engine stopped with a resume pending: an
// apply that restarted it failed after the stop.
func (h *settingsHarness) pendingResume(t *testing.T) {
	t.Helper()
	p := h.request(t)
	p.Settings.LaunchText += " --interrupted"
	h.fail.Store(true)
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	h.fail.Store(false)
	if err != nil || receipt.Phase != "failed" {
		t.Fatalf("failing apply = %+v, %v", receipt, err)
	}
	h.b.engineConfigMu.Lock()
	defer h.b.engineConfigMu.Unlock()
	if record := h.b.engineSettings["ollama"]; !record.Resume || record.Snapshot.Running {
		t.Fatalf("failed apply left resume %t running %t, want a stopped engine with a pending resume", record.Resume, record.Snapshot.Running)
	}
}

// pause sets the node's stored availability as a pause by origin would.
func (h *settingsHarness) pause(origin nodepolicy.PauseOrigin) {
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Availability = nodepolicy.Paused
	r.current.PausedBy = origin
	r.mu.Unlock()
}

// changedRequest is a fresh request that changes the launch arguments.
func (h *settingsHarness) changedRequest(t *testing.T, option string) settings.Request {
	t.Helper()
	p := h.request(t)
	p.Settings.LaunchText += " " + option
	return p
}

func (h *settingsHarness) snapshot() settings.Snapshot {
	h.b.engineConfigMu.Lock()
	defer h.b.engineConfigMu.Unlock()
	return h.b.engineSettings["ollama"].Snapshot
}

// A paused node stays paused: an apply saves the settings and never starts the
// stopped engine whose resume it carries, whoever paused the node. Local
// configuration remains available while peer configuration is fenced.
func TestSettingsApplyWhilePausedNeverResumesAStoppedEngine(t *testing.T) {
	cases := []struct {
		name   string
		origin nodepolicy.PauseOrigin
		caller string
	}{
		{"local apply on a node paused locally", nodepolicy.PausedLocally, ""},
		{"local apply on a node paused by a peer", nodepolicy.PausedByPeer, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := newSettingsHarness(t)
			clustertrusttest.Join(t, h.b.clusterDir, "cluster-1", "self-node", settingsPeer)
			grantRemoteManagement(t, h.b.clusterDir, "cluster-1", "self-node", settingsPeer, true)
			h.pendingResume(t)
			h.pause(c.origin)
			applies := h.applies.Load()
			p := h.changedRequest(t, "--while-paused")
			receipt, err := h.b.applyEngineSettings(context.Background(), p, c.caller)
			if err != nil || receipt.Phase != "succeeded" {
				t.Fatalf("apply while paused = %+v, %v; want the settings applied", receipt, err)
			}
			if h.applies.Load() != applies+1 {
				t.Fatalf("configure-launch ran %d times, want once", h.applies.Load()-applies)
			}
			if s := h.snapshot(); s.Running || s.Settings != p.Settings || s.AppliedRevision != s.Revision {
				t.Fatalf("snapshot after an apply while paused = %+v; want the settings applied and the engine still stopped", s)
			}
		})
	}
}

// A paired node's apply while paused is refused before anything is accepted,
// and the relay reply carries -32011.
func TestSettingsPairedApplyIsVetoedWhilePausedLocally(t *testing.T) {
	h := newSettingsHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster-1", "self-node", settingsPeer)
	grantRemoteManagement(t, h.b.clusterDir, "cluster-1", "self-node", settingsPeer, true)
	h.pendingResume(t)
	h.pause(nodepolicy.PausedLocally)
	before := h.snapshot()
	applies := h.applies.Load()
	p := h.changedRequest(t, "--from-peer")

	_, err := h.b.applyEngineSettings(context.Background(), p, settingsPeer)
	var veto *localVetoError
	if !errors.As(err, &veto) || policyErrorCode(err) != codeLocalVeto {
		t.Fatalf("paired apply while paused locally = %v, want the local veto (%d)", err, codeLocalVeto)
	}
	after := h.snapshot()
	if h.applies.Load() != applies || after.Revision != before.Revision || after.Settings != before.Settings || after.Running {
		t.Fatalf("vetoed apply changed the node: applies %d -> %d, snapshot %+v -> %+v", applies, h.applies.Load(), before, after)
	}
	h.b.engineConfigMu.Lock()
	_, accepted := h.b.engineSettings["ollama"].Receipts[p.RequestID]
	h.b.engineConfigMu.Unlock()
	if accepted {
		t.Fatal("vetoed apply left a receipt")
	}

	source := bindRemoteManagementTestSource(h.b)
	h.b.handleSettingsRelay(source, settingsJSON(settings.Relay{ID: "relay-1", Method: "apply", Caller: settingsPeer, Request: h.changedRequest(t, "--relayed")}))
	select {
	case raw := <-h.settingsReplies:
		var reply struct {
			ID        string `json:"id"`
			Error     string `json:"error"`
			ErrorCode int    `json:"errorCode"`
		}
		if err := json.Unmarshal(raw, &reply); err != nil {
			t.Fatalf("decode settings reply: %v", err)
		}
		if reply.ID != "relay-1" || reply.ErrorCode != codeLocalVeto || reply.Error != (&localVetoError{}).Error() {
			t.Fatalf("relayed apply reply = %+v, want the local veto with errorCode %d", reply, codeLocalVeto)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the relayed apply was never answered")
	}

	// Lifting the pause on the node lets the same paired apply resume the engine.
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Availability = nodepolicy.Available
	r.current.PausedBy = ""
	r.mu.Unlock()
	receipt, err := h.b.applyEngineSettings(context.Background(), h.changedRequest(t, "--after-resume"), settingsPeer)
	if err != nil || receipt.Phase != "succeeded" || !h.snapshot().Running {
		t.Fatalf("paired apply after a local resume = %+v, %v, running %t; want the engine resumed", receipt, err, h.snapshot().Running)
	}
}

func TestSettingsPairedApplyWithoutAResumeIsRefusedWhilePaused(t *testing.T) {
	h := newSettingsHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster-1", "self-node", settingsPeer)
	grantRemoteManagement(t, h.b.clusterDir, "cluster-1", "self-node", settingsPeer, true)
	h.pause(nodepolicy.PausedLocally)
	p := h.changedRequest(t, "--edit-only")
	before, journal := h.snapshot(), rmSettingsJournal(t, h)
	_, err := h.b.applyEngineSettings(context.Background(), p, settingsPeer)
	if policyErrorCode(err) != codeLocalVeto || err.Error() != "This node is paused. Resume it on this node before managing it remotely." {
		t.Fatalf("paired configuration while paused = %v, want the pause refusal", err)
	}
	rmSettingsUnchanged(t, h, before, p.RequestID, journal)
}

func TestPausedSettingsRefuseNoOpAndDuplicateReceipts(t *testing.T) {
	test := func(name string, origin nodepolicy.PauseOrigin, duplicate bool) {
		t.Run(name, func(t *testing.T) {
			h, _ := rmSettings(t)
			p := h.request(t)
			if duplicate {
				receipt, err := h.b.applyEngineSettings(context.Background(), p, "peer")
				if err != nil || receipt.Phase != "succeeded" {
					t.Fatalf("seed duplicate receipt: %+v, %v", receipt, err)
				}
			}
			h.pause(origin)
			before, journal := h.snapshot(), rmSettingsJournal(t, h)
			h.b.engineConfigMu.Lock()
			receipts := make(map[string]settingsReceipt, len(h.b.engineSettings["ollama"].Receipts))
			for id, receipt := range h.b.engineSettings["ollama"].Receipts {
				receipts[id] = receipt
			}
			h.b.engineConfigMu.Unlock()
			_, err := h.b.applyEngineSettings(context.Background(), p, "peer")
			if policyErrorCode(err) != codeLocalVeto || err.Error() != "This node is paused. Resume it on this node before managing it remotely." {
				t.Fatalf("paused receipt request = %v, want the pause refusal", err)
			}
			if got := h.snapshot(); !reflect.DeepEqual(got, before) {
				t.Fatalf("receipt refusal changed snapshot: got %+v, want %+v", got, before)
			}
			if !bytes.Equal(rmSettingsJournal(t, h), journal) {
				t.Fatal("receipt refusal changed the persisted journal")
			}
			h.b.engineConfigMu.Lock()
			gotReceipts := h.b.engineSettings["ollama"].Receipts
			unchanged := reflect.DeepEqual(gotReceipts, receipts)
			h.b.engineConfigMu.Unlock()
			if !unchanged || h.applies.Load() != 0 || h.proxyRebinds.Load() != 0 {
				t.Fatal("receipt refusal changed receipts or runtime")
			}
		})
	}
	test("no-op under local pause", nodepolicy.PausedLocally, false)
	test("no-op under peer pause", nodepolicy.PausedByPeer, false)
	test("duplicate under local pause", nodepolicy.PausedLocally, true)
	test("duplicate under peer pause", nodepolicy.PausedByPeer, true)
}

func TestPendingPauseRefusesSettingsQueuedBeforeAcceptance(t *testing.T) {
	test := func(name string, lock func(*Broker) *sync.Mutex) {
		t.Run(name, func(t *testing.T) {
			h, p := rmSettings(t)
			before, journal := h.snapshot(), rmSettingsJournal(t, h)
			unlock := rmHoldLock(t, lock(h.b))
			done := rmBlockedCall(t, "applyEngineSettings", func() error {
				_, err := h.b.applyEngineSettings(context.Background(), p, "peer")
				return err
			})
			r := h.b.nodePolicyRuntime()
			r.mu.Lock()
			r.pendingPauseRequested = true
			r.mu.Unlock()
			unlock()
			err := rmResult(t, done)
			if policyErrorCode(err) != codeLocalVeto || err.Error() != "This node is paused. Resume it on this node before managing it remotely." {
				t.Fatalf("queued settings = %v, want the pending pause refusal", err)
			}
			rmSettingsUnchanged(t, h, before, p.RequestID, journal)
		})
	}
	test("queued for apply lock", func(b *Broker) *sync.Mutex { return &b.settingsApplyMu })
	test("queued for journal lock", func(b *Broker) *sync.Mutex { return &b.engineConfigMu })
}

func TestPendingPauseRefusesSettingsAfterDelayedPreview(t *testing.T) {
	h, p := rmSettings(t)
	before, journal := h.snapshot(), rmSettingsJournal(t, h)
	engine := attachFakeEngine(t, h.b)
	engine.setAnswer(func(method string, params json.RawMessage) (any, string) {
		switch method {
		case "engine:get-launch":
			return settings.LaunchState{
				Engine: "ollama", ServerPort: before.Settings.ServerPort,
				EffectivePort: before.EffectiveServerPort, LaunchText: before.Settings.LaunchText,
				Running: before.Running, Editable: true, Format: "pair-arguments-v1",
			}, ""
		case "engine:preview-launch":
			var request settings.Request
			if err := json.Unmarshal(params, &request); err != nil {
				t.Errorf("decode delayed preview: %v", err)
				return nil, "invalid preview request"
			}
			return settings.Preview{Settings: request.Settings, Restart: true}, holdSuccessReply
		case "engine:configured-ports", "engine:get-installed":
			return map[string]any{"engines": []any{}}, ""
		case "engine:models":
			return map[string]any{"loadedByEngine": map[string][]string{}}, ""
		default:
			return map[string]bool{"ok": true}, ""
		}
	})
	done := make(chan error, 1)
	go func() {
		_, err := h.b.applyEngineSettings(context.Background(), p, "peer")
		done <- err
	}()
	engine.waitFor("settings preview after initial authorization", func(calls []policyCall) bool {
		for _, call := range calls {
			if call.method == "engine:preview-launch" {
				return true
			}
		}
		return false
	})
	paused := make(chan error, 1)
	go func() {
		_, err := h.b.setAvailabilityAs(context.Background(), nodepolicy.Paused, nodepolicy.PausedByPeer, "peer")
		paused <- err
	}()
	engine.waitFor("accepted peer pause projection before preview returns", func([]policyCall) bool {
		engine.mu.Lock()
		projections := append([]policyCall(nil), engine.projections...)
		engine.mu.Unlock()
		for _, call := range projections {
			var projection nodeProjection
			if err := json.Unmarshal(call.params, &projection); err != nil {
				t.Fatalf("decode pending pause projection: %v", err)
			}
			if projection.ManagementBlocked && !projection.PausedLocally {
				return true
			}
		}
		return false
	})
	engine.releaseHeld()
	err := rmResult(t, done)
	if policyErrorCode(err) != codeLocalVeto || err.Error() != "This node is paused. Resume it on this node before managing it remotely." {
		t.Fatalf("settings after delayed preview = %v, want the pause refusal", err)
	}
	if err := rmResult(t, paused); err != nil {
		t.Fatalf("accepted peer pause: %v", err)
	}
	rmSettingsUnchanged(t, h, before, p.RequestID, journal)
}

func TestPausedBrokerReadsAndLocalConfigurationRemainAvailable(t *testing.T) {
	t.Run("peer policy read and local policy edit", func(t *testing.T) {
		rb := newBrokerOnFile(t, pausedLocallyDocument)
		grantRemoteManagement(t, rb.b.clusterDir, "cluster-1", "self-node", "pinned-peer", false)
		if doc := rb.peerSnapshot(t); doc["pausedBy"] != "local" {
			t.Fatalf("paused peer policy read = %+v", doc)
		}
		rb.localSet(t, func(doc map[string]any) {
			doc["idle"].(map[string]any)["unloadAfterMinutes"] = 11
		})
		if got := rb.b.localPolicy().Policy; got.Idle.UnloadAfterMinutes != 11 || got.Availability != nodepolicy.Paused {
			t.Fatalf("local paused policy edit = %+v", got)
		}
	})
	t.Run("peer settings reads and local settings edit", func(t *testing.T) {
		h, _ := rmSettings(t)
		p := h.changedRequest(t, "--local-paused-edit")
		h.pause(nodepolicy.PausedByPeer)
		grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
		if _, err := h.b.getEngineSettings(context.Background(), p, "peer"); err != nil {
			t.Fatalf("paused peer settings read: %v", err)
		}
		if _, err := h.b.previewEngineSettings(context.Background(), p, "peer"); err != nil {
			t.Fatalf("paused peer settings preview: %v", err)
		}
		receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
		if err != nil || receipt.Phase != "succeeded" {
			t.Fatalf("local paused settings edit = %+v, %v", receipt, err)
		}
		if got := h.snapshot(); !got.Settings.Equal(p.Settings) || got.Running {
			t.Fatalf("local paused settings snapshot = %+v, want edited settings without activation", got)
		}
	})
}

// Recovery replays an interrupted apply without resuming its engine while the
// node is paused.
func TestSettingsRecoveryWhilePausedDoesNotResume(t *testing.T) {
	h := newSettingsHarness(t)
	h.pendingResume(t)
	h.b.engineConfigMu.Lock()
	h.b.engineSettings["ollama"].Snapshot.Phase = "applying"
	if err := h.b.saveEngineSettingsLocked(); err != nil {
		h.b.engineConfigMu.Unlock()
		t.Fatal(err)
	}
	h.b.engineConfigMu.Unlock()
	h.pause(nodepolicy.PausedLocally)
	applies := h.applies.Load()
	if !h.b.recoverEngineSettings() {
		t.Fatal("recovery failed")
	}
	if h.applies.Load() != applies+1 {
		t.Fatalf("recovery ran configure-launch %d times, want once", h.applies.Load()-applies)
	}
	if s := h.snapshot(); s.Running || s.Phase != "succeeded" {
		t.Fatalf("recovered snapshot while paused = %+v, want the operation finished and the engine still stopped", s)
	}
}
