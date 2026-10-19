// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
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
// stopped engine whose resume it carries, whoever paused the node and whoever
// applies. A paired node's apply under a peer pause is not vetoed.
func TestSettingsApplyWhilePausedNeverResumesAStoppedEngine(t *testing.T) {
	cases := []struct {
		name   string
		origin nodepolicy.PauseOrigin
		caller string
	}{
		{"local apply on a node paused locally", nodepolicy.PausedLocally, ""},
		{"local apply on a node paused by a peer", nodepolicy.PausedByPeer, ""},
		{"paired apply on a node paused by a peer", nodepolicy.PausedByPeer, settingsPeer},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := newSettingsHarness(t)
			clustertrusttest.Join(t, h.b.clusterDir, "cluster-1", "self-node", settingsPeer)
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

// A paired node's apply that would resume a stopped engine while this node is
// paused on the node itself is refused with the local veto, before anything is
// accepted, and the relay reply carries -32011.
func TestSettingsPairedApplyIsVetoedWhilePausedLocally(t *testing.T) {
	h := newSettingsHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster-1", "self-node", settingsPeer)
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

	h.b.handleSettingsRelay(settingsJSON(settings.Relay{ID: "relay-1", Method: "apply", Caller: settingsPeer, Request: h.changedRequest(t, "--relayed")}))
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

// The local veto refuses a start, not a configuration edit: a paired node may
// still change the settings of a running engine on a node paused locally.
func TestSettingsPairedApplyWithoutAResumeIsAllowedWhilePausedLocally(t *testing.T) {
	h := newSettingsHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster-1", "self-node", settingsPeer)
	h.pause(nodepolicy.PausedLocally)
	p := h.changedRequest(t, "--edit-only")
	receipt, err := h.b.applyEngineSettings(context.Background(), p, settingsPeer)
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("paired edit of a running engine while paused locally = %+v, %v; want it applied", receipt, err)
	}
	if s := h.snapshot(); s.Settings != p.Settings {
		t.Fatalf("settings after a paired edit = %+v, want %+v", s.Settings, p.Settings)
	}
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
