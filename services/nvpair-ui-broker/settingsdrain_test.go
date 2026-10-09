// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

func drainPushes(calls []policyCall) []nodepolicy.SetEngineDrainParams {
	var out []nodepolicy.SetEngineDrainParams
	for _, c := range calls {
		if c.method == nodepolicy.MethodSetEngineDrain {
			var p nodepolicy.SetEngineDrainParams
			_ = json.Unmarshal(c.params, &p)
			out = append(out, p)
		}
	}
	return out
}

func waitDrainPushes(t *testing.T, h *settingsHarness, n int) []nodepolicy.SetEngineDrainParams {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if p := drainPushes(h.unaddressedProxyCalls()); len(p) >= n {
			return p
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("saw %d drain pushes, want %d", len(drainPushes(h.unaddressedProxyCalls())), n)
	return nil
}

func TestSettingsRestartDrainsEngineFirst(t *testing.T) {
	h := newSettingsHarness(t)
	h.b.observeAdmission(nodepolicy.AdmissionState{Active: 1, ActiveByEngine: map[string]int{"ollama": 1}})
	p := h.request(t)
	p.Settings.LaunchText += " --parallel 4"

	done := make(chan error, 1)
	go func() {
		_, err := h.b.applyEngineSettings(context.Background(), p, "")
		done <- err
	}()
	if got := waitDrainPushes(t, h, 1); got[0] != (nodepolicy.SetEngineDrainParams{Engine: "ollama", Drain: true}) {
		t.Fatalf("first drain push = %+v", got[0])
	}
	time.Sleep(100 * time.Millisecond)
	if n := h.applies.Load(); n != 0 {
		t.Fatal("engine restarted while it still had running requests")
	}
	// Other settings readers are not frozen while the drain waits.
	readDone := make(chan error, 1)
	go func() {
		_, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "ollama"}, "")
		readDone <- err
	}()
	select {
	case <-readDone:
	case <-time.After(2 * time.Second):
		t.Fatal("a settings read blocked behind the drain")
	}

	h.b.observeAdmission(nodepolicy.AdmissionState{Active: 0, ActiveByEngine: map[string]int{}})
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("apply did not continue after the engine drained")
	}
	if n := h.applies.Load(); n != 1 {
		t.Fatalf("applies = %d", n)
	}
	got := waitDrainPushes(t, h, 2)
	if got[1] != (nodepolicy.SetEngineDrainParams{Engine: "ollama", Drain: false}) {
		t.Fatalf("drain was not cleared: %+v", got)
	}
}

func TestSettingsDrainTimesOutAndApplies(t *testing.T) {
	prev := settingsDrainTimeout
	settingsDrainTimeout = 50 * time.Millisecond
	t.Cleanup(func() { settingsDrainTimeout = prev })
	h := newSettingsHarness(t)
	h.b.observeAdmission(nodepolicy.AdmissionState{Active: 1, ActiveByEngine: map[string]int{"ollama": 1}})
	p := h.request(t)
	p.Settings.LaunchText += " --parallel 8"
	if _, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	if h.applies.Load() != 1 {
		t.Fatal("apply did not go ahead after the drain timeout")
	}
	waitDrainPushes(t, h, 2)
}

func TestSettingsWithoutRestartDoNotDrain(t *testing.T) {
	h := newSettingsHarness(t)
	h.b.observeAdmission(nodepolicy.AdmissionState{Active: 1, ActiveByEngine: map[string]int{"ollama": 1}})
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	target := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()
	p := h.request(t)
	p.Settings.ProxyPort = target
	if _, err := h.b.applyEngineSettings(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	if got := drainPushes(h.unaddressedProxyCalls()); len(got) != 0 {
		t.Fatalf("a proxy-port-only change drained the engine: %+v", got)
	}
}
