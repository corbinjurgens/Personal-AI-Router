// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	settings "nvpair-shared/enginesettings"
)

// Recovery always precedes settings replay and enabled-intent restoration.
// Interrupted switches never execute forward. A manager checkpoint is inspected
// even without an active broker row (provisional reservation/terminal-write gap).
func (b *Broker) recoverEngineSwitch() bool {
	b.settingsApplyMu.Lock()
	defer b.settingsApplyMu.Unlock()
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	c := b.switchCoordinator()
	fail := func(err error) bool {
		c.err = err
		slog.Error("engine switch recovery required", "err", err)
		return false
	}
	if err := b.loadEngineSwitchLocked(); err != nil {
		return fail(err)
	}
	if c.err != nil {
		return fail(c.err)
	}
	reserved := make(map[string]bool)
	for _, r := range c.store.journal.Receipts {
		if !r.Terminal || (r.RecoveryRequired && !c.resolved[r.RequestID]) {
			reserved[r.Request.SourceEngine], reserved[r.Request.TargetEngine] = true, true
		}
	}
	before := c.store.journal.ActiveRequestID
	err := c.store.InterruptNonterminal(time.Now().UTC())
	if before != "" {
		if r, ok := c.store.Lookup(before); ok && r.Terminal {
			b.switchChangedLocked(r)
		}
	}
	if err != nil {
		return fail(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), settings.SwitchDrainBudget)
	defer cancel()
	var manager switchManagerSnapshot
	if err := b.switchManagerCall(ctx, settings.SwitchPrepareMethod, settings.SwitchPrepareRequest{}, &manager); err != nil {
		return fail(err)
	}
	if manager.Error != nil {
		return fail(&switchRPCError{*manager.Error})
	}
	cp := manager.Checkpoint
	if cp != nil && cp.Phase != settings.SwitchCheckpointReleased {
		reserved[cp.SourceEngine], reserved[cp.TargetEngine] = true, true
	}
	if len(reserved) > 0 {
		if err := b.loadEngineSettingsLocked(); err != nil {
			return fail(err)
		}
	}
	for engine, record := range b.engineSettings {
		if reserved[engine] && record.Snapshot.Phase == "applying" {
			return fail(fmt.Errorf("conflicting settings and switch journals require recovery for %s", engine))
		}
	}
	// Do not hold engineConfigMu across a potentially long stop/cleanup.
	if cp != nil && cp.Phase != settings.SwitchCheckpointReleased {
		b.engineConfigMu.Unlock()
		var result settings.SwitchReconcileResult
		err := b.switchManagerCall(ctx, settings.SwitchReconcileMethod, settings.SwitchReconcileRequest{SwitchPrepareRequest: cp.SwitchPrepareRequest}, &result)
		if err == nil && result.Error != nil && !switchCleanupSettled(result) {
			err = &switchRPCError{*result.Error}
		}
		if err == nil {
			_, err = b.switchStep(ctx, cp.SwitchPrepareRequest, settings.SwitchStepRelease)
		}
		b.engineConfigMu.Lock()
		if err != nil {
			return fail(err)
		}
	}
	for id, receipt := range c.store.journal.Receipts {
		if !receipt.RecoveryRequired || c.resolved[id] {
			continue
		}
		b.engineConfigMu.Unlock()
		err := b.releaseSwitchDrain(ctx, receipt.Request.SourceEngine)
		b.engineConfigMu.Lock()
		if err != nil {
			return fail(err)
		}
		c.resolved[id] = true
	}
	return true
}
