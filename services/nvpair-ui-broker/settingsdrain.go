// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// settingsdrain.go coordinates an engine settings apply that restarts a
// running engine with the proxy (FORK_DESIGN.md §3.6): admission stops sending
// that engine new work, the apply waits for its running requests to finish,
// and the drain is cleared once the engine is back.

import (
	"context"
	"log/slog"
	"time"

	"nvpair-shared/nodepolicy"
)

// settingsDrainTimeout bounds how long an apply waits for the engine's running
// requests. After it the restart goes ahead and those requests fail, exactly as
// they did before the drain existed.
var settingsDrainTimeout = 60 * time.Second

// drainEngineForSettingsLocked marks engine draining, waits for its active
// count to reach zero, and returns the function that clears the drain. The
// caller holds settingsApplyMu and engineConfigMu; engineConfigMu is released
// while waiting so other settings readers are not frozen for up to a minute,
// which settingsApplyMu makes safe (no other apply can start meanwhile).
func (b *Broker) drainEngineForSettingsLocked(ctx context.Context, engine string) (release func()) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.drains[engine] = true
	r.mu.Unlock()
	kind := pushDrainPrefix + engine

	b.engineConfigMu.Unlock()
	b.pushPolicyNow(ctx, kind)
	drained := b.waitAdmission(ctx, settingsDrainTimeout, func(st nodepolicy.AdmissionState) bool {
		return st.ActiveByEngine[engine] == 0
	})
	b.engineConfigMu.Lock()
	if !drained {
		slog.Warn("engine still had running requests when its settings restart began", "engine", engine)
	}
	return func() {
		r.mu.Lock()
		delete(r.drains, engine)
		r.mu.Unlock()
		b.schedulePolicyPush(kind)
	}
}
