// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// wakesleep.go holds the intent-neutral lifecycle operations the broker's node
// policy drives (FORK_DESIGN.md §3.4, §3.5): engine:wake starts an engine only
// when its saved intent is On, engine:sleep stops it, and neither rewrites the
// saved intent. engine:start / engine:stop remain the only operations that
// record what the user wants; wake and sleep only ever act on that record.

import (
	"context"
	"errors"
	"fmt"
	"nvpair-shared/engineintent"
	"path/filepath"
)

const (
	wakeMethod          = "engine:wake"
	sleepMethod         = "engine:sleep"
	intentMethod        = "engine:intent"
	intentChangedMethod = "engine:intent-changed"
)

// errIntentOff is returned by Wake for an engine whose saved intent is not On.
// An engine with no saved intent counts as Off, matching RestoreEnabled.
var errIntentOff = errors.New("engine is not saved On")

// Wake starts an engine only if its saved intent is On, without changing that
// intent. It holds the engine's lifecycle lock like Restart, so a concurrent
// explicit Stop either finishes first (and Wake then sees intent Off) or waits
// for the start to finish. A shutting-down manager refuses, like every start.
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
	enabled, known, err := e.desired.get(engine)
	if err != nil {
		return err
	}
	if !known || !enabled {
		return fmt.Errorf("wake %s: %w", engine, errIntentOff)
	}
	return e.doStart(ctx, st, engine, startOpts{})
}

// Sleep stops an engine the way Restart and StopAll do, without recording Off
// intent, so a later Wake (or restore at the next launch) can start it again.
func (e *Executor) Sleep(engine string) error {
	done, err := e.admitMutation(engine)
	if err != nil {
		return err
	}
	defer done()
	st, err := e.state(engine)
	if err != nil {
		return e.sleepUnsupported(engine, err)
	}
	cancelCrashRelaunch(st)
	st.opMu.Lock()
	defer st.opMu.Unlock()
	return e.doStop(st, engine)
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

func (e *Executor) sleepUnsupported(engine string, platformErr error) error {
	st, err := e.unsupportedStopState(engine)
	if err != nil {
		return fmt.Errorf("%w: %v", platformErr, err)
	}
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
	status := EngineStatus{Engine: engine, DisplayName: m.DisplayName, Locked: e.lockedStatus(engine)}
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

// Intent reports every registered engine's saved intent. It reads only the
// intent file, never an engine's lifecycle lock, so it answers immediately even
// while an engine is starting.
func (e *Executor) Intent() (engineintent.Snapshot, error) {
	return e.desired.snapshot(e.reg.Names())
}

// emitIntent pushes the current saved intent after it changes. Best-effort:
// unreadable or invalid snapshots are not published.
func (e *Executor) emitIntent() {
	if intent, err := e.Intent(); err == nil {
		e.notify(intentChangedMethod, intent)
	}
}

// handleLifecyclePolicyRequest serves engine:wake, engine:sleep and
// engine:intent. It reports whether it took the method.
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
			m.respondOrErr(msg, st, err)
		}()
		return true
	case intentMethod:
		intent, err := m.exec.Intent()
		m.respondOrErr(msg, intent, err)
		return true
	}
	return false
}
