// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Wake and Sleep are availability-driven lifecycle operations.

import (
	"context"
	"fmt"
	"path/filepath"
)

const (
	wakeMethod  = "engine:wake"
	sleepMethod = "engine:sleep"
)

// Wake starts an installed engine subject to activation admission.
func (e *Executor) Wake(ctx context.Context, engine string) error {
	done, err := e.admitMutation(engine)
	if err != nil {
		return err
	}
	defer done()
	st, err := e.state(engine)
	if err != nil {
		return err
	}
	if err := prepareStrataState(ctx, st); err != nil {
		return err
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	if e.shuttingDown.Load() {
		return fmt.Errorf("engine-manager is shutting down")
	}
	installed, err := e.Detect(engine)
	if err != nil {
		return err
	}
	if !installed {
		return fmt.Errorf("engine %s is not installed", engine)
	}
	return e.doStart(ctx, st, engine, startOpts{})
}

// Sleep stops an engine for node availability; a later Wake can start it.
func (e *Executor) Sleep(engine string) (err error) {
	done, err := e.admitMutation(engine)
	if err != nil {
		return err
	}
	defer done()
	st, err := e.state(engine)
	if err != nil {
		return e.sleepUnsupported(engine, err)
	}
	defer func() {
		if err != nil {
			st.mu.Lock()
			st.withdrawOwnerKnowledgeLocked()
			st.mu.Unlock()
		}
	}()
	cancelCrashRelaunch(st)
	// Cancel accepted execution before waiting for its lifecycle lock. The load
	// fence also catches a registration racing this cancellation sweep.
	veto := &localVetoError{code: codeLocalVeto, reason: nodePausedVeto}
	st.mu.Lock()
	if st.startCancel != nil {
		st.startCancel(veto)
	}
	st.mu.Unlock()
	finishLoads := stopServingLoadsWithCause(st, veto)
	defer finishLoads()
	st.opMu.Lock()
	defer st.opMu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), stopTimeout(st.plat.Runtime)+modelsTimeout)
	defer cancel()
	// Passive status may not have discovered a listener yet. Only the lifecycle
	// owner can decide whether it is external or an exact managed orphan.
	installed, err := e.Detect(engine)
	if err != nil {
		return err
	}
	st.mu.Lock()
	port := st.port
	st.mu.Unlock()
	evidence, known, err := e.desired.processStop(engine)
	if err != nil {
		return err
	}
	allocated, err := e.desired.allocatedPort(engine)
	if err != nil {
		return err
	}
	// Paused startup skips RestoreInstalled, so restore only its cleanup
	// endpoint here. This record never authorizes activation or a signal.
	if port == 0 {
		if known {
			port = evidence.Port
		} else {
			port = allocated
		}
		st.mu.Lock()
		st.port = port
		st.mu.Unlock()
	}
	reclaimed, err := e.reclaimOrphan(ctx, st, engine, port)
	if err != nil {
		return err
	}
	if !reclaimed {
		st.mu.Lock()
		owned := st.proc != nil
		st.mu.Unlock()
		if !owned && ((known && (evidence.PID == 0 || !orphanPIDExited(evidence.PID))) || (!known && allocated > 0)) {
			return fmt.Errorf("cannot pause engine %q: saved process identity is unverified", engine)
		}
	}

	presence := e.reconcilePresence(ctx, engine, st, installed, port, true)
	if presence.Occupied && !presence.Identified {
		return fmt.Errorf("cannot pause engine %q: listener ownership is unverified", engine)
	}
	if err := e.doStopContext(ctx, st, engine); err != nil {
		return err
	}
	if !e.terminationVerifiedContext(ctx, st) {
		return fmt.Errorf("cannot pause engine %q: physical termination is unverified", engine)
	}
	return nil
}

// unsupportedStopState reconstructs only the stop owner, never an activatable
// platform. A missing record (including older managers) is unknown, not stopped.
func (e *Executor) unsupportedStopState(engine string) (*engineState, error) {
	m, known := e.reg.Get(engine)
	if !known {
		return nil, fmt.Errorf("unknown engine %q", engine)
	}
	evidence, known, err := e.desired.processStop(engine)
	if err != nil {
		return nil, err
	}
	if !known {
		return nil, fmt.Errorf("engine %q: no saved process identity; termination is unverified", engine)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if st := e.stopOnly[engine]; st != nil {
		return st, nil
	}
	st := &engineState{
		manifest: m, logs: newLogBuffer(),
		installDir: filepath.Join(e.baseDir, engine), binPath: evidence.Bin,
		port: evidence.Port, reclaimPID: evidence.PID, reclaimFailed: true,
		plat: &Platform{Runtime: Runtime{Port: evidence.Port, Ready: &Probe{TCP: evidence.Address}}},
	}
	e.stopOnly[engine] = st
	return st, nil
}

func (e *Executor) sleepUnsupported(engine string, platformErr error) (err error) {
	st, err := e.unsupportedStopState(engine)
	if err != nil {
		return fmt.Errorf("%w: %v", platformErr, err)
	}
	defer func() {
		if err != nil {
			st.mu.Lock()
			st.withdrawOwnerKnowledgeLocked()
			st.mu.Unlock()
		}
	}()
	st.opMu.Lock()
	defer st.opMu.Unlock()
	// The shared reclaim owner checks managed install containment and exact
	// executable image before signalling. Passive status never adopts this PID.
	reclaimed, err := e.reclaimOrphan(context.Background(), st, engine, st.port)
	if err != nil {
		return err
	}
	if !reclaimed {
		st.mu.Lock()
		pid, fenced := st.reclaimPID, st.reclaimFailed
		st.mu.Unlock()
		if fenced && (pid == 0 || !orphanPIDExited(pid)) {
			return fmt.Errorf("engine %q: process termination is unverified", engine)
		}
	}
	if !e.waitUnavailable(st.plat.Runtime.Ready, st.port, presenceRefusalWindow) {
		return fmt.Errorf("engine %q: endpoint termination is unverified", engine)
	}
	e.markStopped(st, engine)
	return nil
}

func (e *Executor) unsupportedStatus(engine string, m *Manifest) EngineStatus {
	status := EngineStatus{Engine: engine, DisplayName: m.DisplayName}
	st, err := e.unsupportedStopState(engine)
	if err != nil {
		return status
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	st.mu.Lock()
	fenced, pid := st.reclaimFailed, st.reclaimPID
	st.mu.Unlock()
	// An unresolved spawn window or surviving recorded PID cannot be cleared
	// by listener refusal. Only Sleep may signal or clear the reclaim fence.
	status.TerminationVerified = (!fenced || (pid > 0 && orphanPIDExited(pid))) &&
		e.waitUnavailable(st.plat.Runtime.Ready, st.port, presenceRefusalWindow)
	return status
}

// handleLifecyclePolicyRequest serves internal wake and sleep requests.
func (m *Manager) handleLifecyclePolicyRequest(ctx context.Context, msg *Message) bool {
	switch msg.Method {
	case wakeMethod, sleepMethod:
		go func() {
			var p engineParam
			if !m.parse(msg, &p) {
				return
			}
			if p.Engine == "" {
				m.codec.RespondError(msg.ID, -32602, "engine is required")
				return
			}
			var err error
			if msg.Method == wakeMethod {
				err = m.exec.Wake(ctx, p.Engine)
			} else {
				err = m.exec.Sleep(p.Engine)
			}
			if err != nil {
				m.codec.RespondError(msg.ID, -32000, err.Error())
				return
			}
			st, err := m.exec.Status(p.Engine)
			if err == nil && msg.Method == sleepMethod && !st.Running {
				// Sleep proved process termination and controlled residency under
				// opMu. This reply is positive evidence, unlike running:false alone.
				st.TerminationVerified = true
			}
			m.respondOrErr(msg, st, err)
		}()
		return true
	}
	return false
}
