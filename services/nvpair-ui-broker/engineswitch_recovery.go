// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	settings "nvpair-shared/enginesettings"
	serviceerrors "nvpair-shared/errors"
)

const engineSwitchRecoveryBlockedID = "engine-switch:recovery-blocked"

func switchRecoveryPermanent(err error) bool {
	switch e := err.(type) {
	case *engineSwitchRecoveryError, *engineSwitchStoreError:
		return true
	case *switchRPCError:
		return e.detail.Kind == settings.SwitchErrorStorage
	case interface{ Unwrap() []error }:
		for _, cause := range e.Unwrap() {
			if switchRecoveryPermanent(cause) {
				return true
			}
		}
	case interface{ Unwrap() error }:
		return switchRecoveryPermanent(e.Unwrap())
	}
	return false
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
	releaseDrains := func() error {
		for id, receipt := range c.store.journal.Receipts {
			if !receipt.RecoveryRequired || c.resolved[id] {
				continue
			}
			b.engineConfigMu.Unlock()
			err := b.releaseSwitchDrain(ctx, receipt.Request.SourceEngine)
			b.engineConfigMu.Lock()
			if err != nil {
				return err
			}
		}
		return nil
	}
	// Do not hold engineConfigMu across a potentially long stop/cleanup.
	if cp != nil && cp.Phase != settings.SwitchCheckpointReleased {
		for {
			b.workersMu.Lock()
			var veto *settings.SwitchLocalVeto
			if c.vetoID == cp.OperationID {
				veto = c.localVeto
			}
			b.workersMu.Unlock()
			// Recover the durable veto outcome even if restart lost the latch.
			if veto == nil && cp.Origin == settings.SwitchOriginRemote {
				if r, ok := c.store.Lookup(cp.OperationID); ok && r.Error != nil && r.Error.Kind == settings.SwitchErrorRemoteVeto {
					veto = &settings.SwitchLocalVeto{Pause: true}
				}
			}
			b.engineConfigMu.Unlock()
			var result settings.SwitchReconcileResult
			err := switchManagerCallOn(ctx, w, settings.SwitchReconcileMethod, settings.SwitchReconcileRequest{SwitchPrepareRequest: cp.SwitchPrepareRequest, LocalVeto: veto}, &result)
			b.engineConfigMu.Lock()
			if err == nil && result.Error != nil && !switchCleanupSettled(result) {
				err = &switchRPCError{*result.Error}
			}
			if err != nil {
				return fail(err)
			}
			// Drain RPCs can also admit a late Off. Keep the checkpoint until
			// they finish, then recheck the latch before releasing either owner.
			if err := releaseDrains(); err != nil {
				return fail(err)
			}
			b.workersMu.Lock()
			latest := c.localVeto
			changed := c.vetoID == cp.OperationID && latest != nil && (veto == nil || *latest != *veto)
			b.workersMu.Unlock()
			if changed {
				// Off/pause admitted while reconcile was running must reach the
				// cleanup owner before its reservation can be released.
				continue
			}
			// Keep the admission lock through this metadata-only release and
			// clearing recoveryCheckpoint. New local veto admission cannot slip
			// between the recheck and release. No workersMu is held over RPC.
			var release settings.SwitchStepResult
			err = switchManagerCallOn(ctx, w, settings.SwitchStepMethod, settings.SwitchStepRequest{SwitchPrepareRequest: cp.SwitchPrepareRequest, Step: settings.SwitchStepRelease}, &release)
			if err == nil && release.Error != nil {
				err = &switchRPCError{*release.Error}
			}
			if err != nil {
				return fail(err)
			}
			break
		}
	} else if err := releaseDrains(); err != nil {
		return fail(err)
	}
	c.recoveryCheckpoint = nil
	for id, receipt := range c.store.journal.Receipts {
		if !receipt.RecoveryRequired || c.resolved[id] {
			continue
		}
		c.resolved[id] = true
	}
	if err := b.notifyEngineSwitchRecoveryReady(ctx, w, manager.Configuration.ManagerIncarnation); err != nil {
		return fail(err)
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
