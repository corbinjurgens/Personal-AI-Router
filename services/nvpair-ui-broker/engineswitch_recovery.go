// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	settings "nvpair-shared/enginesettings"
	serviceerrors "nvpair-shared/errors"
)

const engineSwitchRecoveryBlockedID = "engine-switch:recovery-blocked"

func switchRecoveryPermanent(err error) bool {
	var journal *engineSwitchRecoveryError
	var store *engineSwitchStoreError
	var rpc *switchRPCError
	return errors.As(err, &journal) || errors.As(err, &store) || (errors.As(err, &rpc) && rpc.detail.Kind == settings.SwitchErrorStorage)
}

// Recovery always precedes settings replay and enabled-intent restoration.
// Interrupted switches never execute forward. A manager checkpoint is inspected
// even without an active broker row (provisional reservation/terminal-write gap).
func (b *Broker) recoverEngineSwitch() bool {
	b.workersMu.Lock()
	w := b.engineMgr
	b.workersMu.Unlock()
	return b.recoverEngineSwitchOn(w)
}

func (b *Broker) recoverEngineSwitchOn(w *rpcWorker) bool {
	b.settingsApplyMu.Lock()
	defer b.settingsApplyMu.Unlock()
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	b.workersMu.Lock()
	current := b.engineMgr == w && w != nil
	b.workersMu.Unlock()
	if !current {
		return false
	}
	c := b.switchCoordinator()
	fail := func(err error) bool {
		if switchRecoveryPermanent(err) {
			c.err = err
		} else if c.err == nil {
			c.recoveryErr = err
		}
		slog.Error("engine switch recovery required", "err", err)
		return false
	}
	if err := b.loadEngineSwitchLocked(); err != nil {
		c.err = err
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
		c.err = err
		return fail(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), settings.SwitchDrainBudget)
	defer cancel()
	var manager switchManagerSnapshot
	if err := switchManagerCallOn(ctx, w, settings.SwitchPrepareMethod, settings.SwitchPrepareRequest{}, &manager); err != nil {
		return fail(err)
	}
	if manager.Error != nil {
		return fail(&switchRPCError{*manager.Error})
	}
	cp := manager.Checkpoint
	c.recoveryCheckpoint = cp
	if cp != nil && cp.Phase != settings.SwitchCheckpointReleased {
		reserved[cp.SourceEngine], reserved[cp.TargetEngine] = true, true
	}
	if len(reserved) > 0 {
		if err := b.loadEngineSettingsLocked(); err != nil {
			c.err = err
			return fail(err)
		}
	}
	for engine, record := range b.engineSettings {
		if reserved[engine] && record.Snapshot.Phase == "applying" {
			c.err = fmt.Errorf("conflicting settings and switch journals require recovery for %s", engine)
			return fail(c.err)
		}
	}
	// Do not hold engineConfigMu across a potentially long stop/cleanup.
	if cp != nil && cp.Phase != settings.SwitchCheckpointReleased {
		b.engineConfigMu.Unlock()
		var result settings.SwitchReconcileResult
		err := switchManagerCallOn(ctx, w, settings.SwitchReconcileMethod, settings.SwitchReconcileRequest{SwitchPrepareRequest: cp.SwitchPrepareRequest}, &result)
		if err == nil && result.Error != nil && !switchCleanupSettled(result) {
			err = &switchRPCError{*result.Error}
		}
		if err == nil {
			var release settings.SwitchStepResult
			err = switchManagerCallOn(ctx, w, settings.SwitchStepMethod, settings.SwitchStepRequest{SwitchPrepareRequest: cp.SwitchPrepareRequest, Step: settings.SwitchStepRelease}, &release)
			if err == nil && release.Error != nil {
				err = &switchRPCError{*release.Error}
			}
		}
		b.engineConfigMu.Lock()
		if err != nil {
			return fail(err)
		}
	}
	c.recoveryCheckpoint = nil
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
	c.recoveryErr = nil
	return true
}

func (b *Broker) reportEngineSwitchRecoveryBlocked() {
	b.engineConfigMu.Lock()
	c := b.switchCoordinator()
	err := c.err
	if err == nil {
		err = c.recoveryErr
	}
	b.engineConfigMu.Unlock()
	message := "Engine manager remains unavailable because conservative switch recovery could not be verified."
	slog.Error(message, "err", err)
	b.forwardErrorsReport(serviceerrors.ServiceError{
		ID: engineSwitchRecoveryBlockedID, Message: message,
		Timestamp: nowMillis(), NodeID: b.nodeID, Severity: "error", Action: "none",
	})
}
