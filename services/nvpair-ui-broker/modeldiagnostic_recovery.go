// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
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
	b.diagnostics.mu.Lock()
	unchanged := b.diagnostics.active == op
	b.diagnostics.mu.Unlock()
	if !unchanged {
		return diagnosticError("broker diagnostic changed during checkpoint reconciliation")
	}
	if err := validateDiagnosticManagerStatus(status, ref.RunID); err != nil {
		return err
	}
	if status.State == "none" {
		if op != nil {
			if op.worker == worker {
				select {
				case <-op.done:
					op.cleanupMu.Lock()
					defer op.cleanupMu.Unlock()
					return b.cleanupDiagnostic(ctx, op)
				default:
					// Published before acquisition. Its owner settles any
					// in-flight reserve/run request before releasing its ledger.
					return nil
				}
			}
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
			if !b.diagnosticOperationCurrent(op) {
				return nil
			}
			op.status = status
			b.setDiagnosticRun(op, *status.Run)
			if err := b.holdDiagnosticDrains(ctx, op); err != nil {
				return err
			}
			if status.WorkStopped && status.State != "cleanup-failed" && b.diagnosticProxyReady() {
				return b.cleanupDiagnostic(ctx, op)
			}
		default:
		}
		return nil // The running goroutine exclusively owns its ledger.
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
		if !b.diagnosticOperationCurrent(op) {
			return diagnosticError("diagnostic changed during replacement recovery")
		}
		op.worker = worker
		op.status = status
		op.reserveUncertain, op.runUncertain = false, false
		// The checkpoint owner persists interrupted evidence before it reports
		// cleanup complete. Never record the retired manager's partial samples.
		op.recorded = true
		if status.Token != "" {
			op.token = status.Token
		}
		b.setDiagnosticRun(op, *status.Run)
		if err := b.holdDiagnosticDrains(ctx, op); err != nil {
			return err
		}
		if status.WorkStopped && status.State != "cleanup-failed" && b.diagnosticProxyReady() {
			return b.cleanupDiagnostic(ctx, op)
		}
		return nil
	}
	lease, err := b.reserveBuildApplySettings()
	if err != nil {
		return err
	}
	_, stop := context.WithCancel(context.Background())
	op = &brokerDiagnostic{runID: status.Run.RunID, run: status.Run.Clone(), worker: worker, cancel: stop, done: make(chan struct{}), wake: make(chan struct{}, 1), lease: lease, token: status.Token, reserveAttempted: true, accepted: status.State != "reserved", recorded: true, status: status}
	op.cleanupMu.Lock()
	defer op.cleanupMu.Unlock()
	close(op.done) // Recovered operations never resume measurement or sampling.
	b.diagnostics.mu.Lock()
	if b.diagnostics.active != nil {
		b.diagnostics.mu.Unlock()
		lease.Release()
		stop()
		return diagnosticError("diagnostic accepted during checkpoint reconciliation")
	}
	b.diagnostics.active = op
	b.diagnostics.mu.Unlock()
	if err := b.holdDiagnosticDrains(ctx, op); err != nil {
		b.retainDiagnostic(op, err)
		return err
	}
	if status.WorkStopped && status.State != "cleanup-failed" && b.diagnosticProxyReady() {
		return b.cleanupDiagnostic(ctx, op)
	}
	return nil
}

// Reconstruct in the existing registry before a proxy may activate. On a live
// proxy, acknowledge the same drains immediately. Keep the settings lock with
// this ledger so settings cannot race recovery or a release retry.
func (b *Broker) holdDiagnosticDrains(ctx context.Context, op *brokerDiagnostic) error {
	profiles := b.frontedEngineProfiles()
	if len(profiles) == 0 {
		return diagnosticError("cannot reconstruct diagnostic engine catalog")
	}
	if op.drain == nil {
		op.drain = &nodeDrainLease{broker: b}
	}
	if !op.drain.settingsHeld {
		if !b.settingsApplyMu.TryLock() {
			return diagnosticError("cannot reconstruct diagnostic during settings apply")
		}
		op.drain.settingsHeld = true
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	for _, profile := range profiles {
		if !r.drains[profile.Name] {
			r.drains[profile.Name] = true
			op.drain.added = append(op.drain.added, profile.Name)
		}
	}
	incarnation, runID := r.incarnation, r.runID
	r.mu.Unlock()
	pp := b.policyProxy()
	if pp == nil || pp != incarnation || runID == "" {
		// The ordinary startup/replay owner sends registry drains before it
		// activates this new proxy. Never talk to an unverified incarnation.
		return nil
	}
	if !r.acquirePush(ctx) {
		return ctx.Err()
	}
	defer r.releasePush()
	for _, profile := range profiles {
		if err := b.nodeDrainProxyCurrent(pp, runID); err != nil {
			return err
		}
		if err := engineDrainAck(ctx, pp, profile.Name, true); err != nil {
			return err
		}
	}
	return b.nodeDrainProxyCurrent(pp, runID)
}

func (b *Broker) diagnosticHoldActive() bool {
	b.diagnostics.mu.Lock()
	defer b.diagnostics.mu.Unlock()
	return b.diagnostics.active != nil
}

func (b *Broker) diagnosticOperationCurrent(op *brokerDiagnostic) bool {
	b.diagnostics.mu.Lock()
	defer b.diagnostics.mu.Unlock()
	return b.diagnostics.active == op
}

func (b *Broker) diagnosticProxyReady() bool {
	pp := b.policyProxy()
	if pp == nil {
		return true
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.incarnation == pp && r.runID != ""
}
