// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

func assertInstalledEnginesRunning(t *testing.T, f *stopPauseServiceFixture) {
	t.Helper()
	task1Eventually(t, "all installed engines running", 30*time.Second, func() bool {
		var inventory struct {
			Engines []struct {
				Engine    string `json:"engine"`
				Installed bool   `json:"installed"`
				Running   bool   `json:"running"`
				Healthy   bool   `json:"healthy"`
			} `json:"engines"`
		}
		task2Decode(t, f.call(t, "engine:get-installed", nil), &inventory)
		seen := make(map[string]bool)
		for _, engine := range inventory.Engines {
			if engine.Installed {
				if !engine.Running || !engine.Healthy {
					return false
				}
				seen[engine.Engine] = true
			}
		}
		return seen["llamacpp"] && seen["llamacpp-custom"]
	})
}

func TestEngineLifecycleInstallRestartPauseResume(t *testing.T) {
	f := newStopPauseFixture(t)
	// Installation itself starts the installed engine, without a start option.
	f.call(t, "engine:install", map[string]string{"engine": "llamacpp-custom"})
	f.call(t, "engine:install", map[string]string{"engine": "llamacpp"})
	assertInstalledEnginesRunning(t, f.stopPauseServiceFixture)
	f.engine.mu.Lock()
	previous := append([]serviceProcessIdentity(nil), f.engine.identities...)
	f.engine.mu.Unlock()
	if err := f.service.cleanupChecked(); err != nil {
		t.Fatalf("restart service: %v", err)
	}
	f.service.assertStopped()
	for _, identity := range previous {
		f.engine.assertProcessStopped(t, identity)
	}
	f.start(t)
	assertInstalledEnginesRunning(t, f.stopPauseServiceFixture)
	f.availability(t, nodepolicy.Paused, "configured")
	f.assertPaused(t, nodepolicy.PausedLocally)
	f.availability(t, nodepolicy.Available, "")
	assertInstalledEnginesRunning(t, f.stopPauseServiceFixture)
}
