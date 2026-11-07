// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"time"

	"nvpair-shared/modelconfig"
)

// Called before proxy replay acquires the push slot, and before a replacement
// manager becomes available. The pinned worker may still be hidden by recovery.
func (b *Broker) reconcileDiagnostics(parent context.Context, worker *rpcWorker) error {
	b.diagnostics.reconcileMu.Lock()
	defer b.diagnostics.reconcileMu.Unlock()
	b.diagnostics.mu.Lock()
	op := b.diagnostics.active
	b.diagnostics.mu.Unlock()
	if worker == nil {
		if op != nil || b.engineMgrPath != "" {
			return diagnosticError("diagnostic checkpoint unavailable")
		}
		return nil
	}
	ctx, cancel := context.WithTimeout(parent, 15*time.Second)
	defer cancel()
	var status diagnosticManagerStatus
	ref := modelconfig.DiagnosticRunRef{}
	if op != nil {
		ref.RunID = op.runID
	}
	if err := modelSettingsCall(ctx, worker, "engine:diagnostic-run-status", ref, &status); err != nil {
		return err
	}
	b.workersMu.Lock()
	current := b.engineMgr == worker
	b.workersMu.Unlock()
	if !current {
		return diagnosticError("manager changed during checkpoint reconciliation")
	}
	if status.State == "none" {
		if op != nil {
			return diagnosticError("pending diagnostic has no cleanup proof from replacement manager")
		}
		return nil
	}
	if status.Run == nil || status.Run.RunID == "" || status.State == "" {
		return diagnosticError("invalid diagnostic checkpoint reply")
	}
	if op != nil && op.worker == worker && op.runID == status.Run.RunID {
		// Its live goroutine owns the existing ledger and final record. A proxy
		// replacement replays those same registry drains; no second lease.
		select {
		case <-op.done:
			op.cleanupMu.Lock()
			defer op.cleanupMu.Unlock()
			op.status = status
			b.setDiagnosticRun(op, *status.Run)
			if status.WorkStopped {
				return b.cleanupDiagnostic(ctx, op)
			}
		default:
		}
		return nil
	}
	if op != nil {
		op.cancel()
		select {
		case <-op.done:
		case <-ctx.Done():
			return ctx.Err()
		}
		if op.runID != status.Run.RunID {
			return diagnosticError("manager checkpoint conflicts with retained diagnostic")
		}
		op.cleanupMu.Lock()
		defer op.cleanupMu.Unlock()
		op.worker = worker
		op.status = status
		if status.Token != "" {
			op.token = status.Token
		}
		b.setDiagnosticRun(op, *status.Run)
		if status.WorkStopped {
			// Task12 persists interrupted evidence in the checkpoint owner.
			op.recorded = true
			return b.cleanupDiagnostic(ctx, op)
		}
		return nil
	}
	if status.WorkStopped && status.State == "cleanup-complete" {
		return nil
	}
	lease, err := b.reserveBuildApplySettings()
	if err != nil {
		return err
	}
	if !b.settingsApplyMu.TryLock() {
		lease.Release()
		return fmt.Errorf("cannot reconstruct diagnostic during settings apply")
	}
	drain := &nodeDrainLease{broker: b, settingsHeld: true}
	r := b.nodePolicyRuntime()
	profiles := b.frontedEngineProfiles()
	if len(profiles) == 0 {
		b.settingsApplyMu.Unlock()
		lease.Release()
		return diagnosticError("cannot reconstruct diagnostic engine catalog")
	}
	r.mu.Lock()
	for _, profile := range profiles {
		if !r.drains[profile.Name] {
			r.drains[profile.Name] = true
			drain.added = append(drain.added, profile.Name)
		}
	}
	r.mu.Unlock()
	_, stop := context.WithCancel(context.Background())
	op = &brokerDiagnostic{runID: status.Run.RunID, run: status.Run.Clone(), worker: worker, cancel: stop, done: make(chan struct{}), wake: make(chan struct{}, 1), lease: lease, drain: drain, token: status.Token, reserveAttempted: true, accepted: true, recorded: true, status: status}
	close(op.done) // Recovered operations never resume measurement or sampling.
	b.diagnostics.mu.Lock()
	b.diagnostics.active = op
	b.diagnostics.mu.Unlock()
	if status.WorkStopped {
		return b.cleanupDiagnostic(ctx, op)
	}
	return nil
}
