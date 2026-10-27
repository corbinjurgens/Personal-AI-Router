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
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"

	"nvpair-shared/engineintent"
	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
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
	return e.sleepWithNativeReadback(engine, e.runCommandOutput)
}

func (e *Executor) sleepWithNativeReadback(engine string, nativeReadback func(context.Context, []string) (string, error)) error {
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
	ctx, cancel := context.WithTimeout(context.Background(), commandStopTimeout(st.plat.Runtime)+modelsTimeout)
	defer cancel()
	// Passive status may not have discovered a listener yet. Only the lifecycle
	// owner can decide whether it is external or an exact managed orphan.
	installed, err := e.Detect(engine)
	if err != nil {
		return err
	}
	st.mu.Lock()
	rt, port := st.plat.Runtime, st.port
	st.mu.Unlock()
	if rt.modeOrDefault() == "process" {
		evidence, known, err := e.desired.processStop(engine)
		if err != nil {
			return err
		}
		allocated, err := e.desired.allocatedPort(engine)
		if err != nil {
			return err
		}
		// Paused startup skips RestoreEnabled, so restore only its cleanup
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
	}
	presence := e.reconcilePresence(ctx, engine, st, installed, port, true)
	st.mu.Lock()
	adopted, running := st.adopted, st.running
	st.mu.Unlock()
	if rt.modeOrDefault() == "command" && adopted && running && st.manifest.FamilyOrDefault() == "lmstudio" {
		// The external owner may have stopped its server and unloaded its own
		// models since the previous refusal. Only observe it: never run an
		// unload or unscoped stop against an adopted application.
		if rt.Ready == nil || !e.waitUnavailableContext(ctx, rt.Ready, port, presenceRefusalWindow) {
			return fmt.Errorf("cannot stop engine %q: it is running under external management; stop it in its own application, then retry", engine)
		}
		if err := stoppedLMStudioResidency(ctx, st, nativeReadback); err != nil {
			return err
		}
		if !e.waitUnavailableContext(ctx, rt.Ready, port, presenceRefusalWindow) {
			return fmt.Errorf("cannot pause engine %q: external endpoint changed during model residency proof", engine)
		}
		e.markStopped(st, engine)
		return nil
	}
	if presence.Occupied && !presence.Identified {
		return fmt.Errorf("cannot pause engine %q: listener ownership is unverified", engine)
	}
	if rt.modeOrDefault() == "command" {
		if adopted && running {
			return fmt.Errorf("cannot stop engine %q: it is running under external management; stop it in its own application, then retry", engine)
		}
		// A command can stop just the HTTP server while model processes survive.
		// Prove empty residency at the native owner before its control endpoint
		// disappears. Never issue automatic unloads to an adopted engine.
		if installed || running {
			if err := e.emptyCommandResidency(ctx, st, engine, running, nativeReadback); err != nil {
				return err
			}
		}
	}
	if err := e.doStopContext(ctx, st, engine); err != nil {
		return err
	}
	if rt.modeOrDefault() == "command" && running && st.manifest.FamilyOrDefault() == "lmstudio" {
		// The server can be down while llmster still owns models. Read the native
		// residency owner after stop as well, including on repeated/startup Pause.
		if err := stoppedLMStudioResidency(ctx, st, nativeReadback); err != nil {
			return err
		}
	}
	if !e.switchTerminatedContext(ctx, st) {
		return fmt.Errorf("cannot pause engine %q: physical termination is unverified", engine)
	}
	return nil
}

// Caller holds opMu and has established ownership. This deliberately bypasses
// Action's lifecycle admission: the sleep operation already owns that lock.
func (e *Executor) emptyCommandResidency(ctx context.Context, st *engineState, engine string, unload bool, nativeReadback func(context.Context, []string) (string, error)) error {
	if !unload && st.manifest.FamilyOrDefault() == "lmstudio" {
		return stoppedLMStudioResidency(ctx, st, nativeReadback)
	}
	loaded, ok := st.manifest.Actions["loaded_models"]
	if !ok || loaded.Result == nil || loaded.RestartAfter {
		return fmt.Errorf("cannot pause engine %q: model residency readback is unavailable", engine)
	}
	read := func() ([]string, error) {
		raw, err := e.dispatchAction(ctx, st, engine, "loaded_models", loaded, nil)
		if err != nil {
			return nil, fmt.Errorf("cannot pause engine %q: model residency readback failed: %w", engine, err)
		}
		models, valid := commandResidentModels(raw, loaded.Result)
		if !valid {
			return nil, fmt.Errorf("cannot pause engine %q: model residency readback is incomplete", engine)
		}
		return models, nil
	}
	models, err := read()
	if err != nil {
		return err
	}
	if len(models) == 0 {
		return nil
	}
	if !unload {
		return fmt.Errorf("cannot pause engine %q: uncontrolled model residency remains", engine)
	}
	binding, ok := st.manifest.operationBinding(engines.OperationUnload)
	if !ok || st.manifest.Actions[binding.Action].RestartAfter {
		return fmt.Errorf("cannot pause engine %q: controlled model unload is unavailable", engine)
	}
	for _, model := range models {
		params, err := binding.params(model)
		if err != nil {
			return err
		}
		if _, err := e.dispatchAction(ctx, st, engine, binding.Action, st.manifest.Actions[binding.Action], params); err != nil {
			return fmt.Errorf("cannot pause engine %q: model unload failed: %w", engine, err)
		}
	}
	models, err = read()
	if err != nil {
		return err
	}
	if len(models) != 0 {
		return fmt.Errorf("cannot pause engine %q: model residency remains after unload", engine)
	}
	return nil
}

// lms ps observes model processes independently of the HTTP server. An empty,
// successfully decoded native array is proof; a failed CLI, null, or occupied
// inventory is never interpreted as an already stopped engine.
func stoppedLMStudioResidency(ctx context.Context, st *engineState, run func(context.Context, []string) (string, error)) error {
	cli := expandPath(st.plat.Runtime.CLI)
	if cli == "" {
		return fmt.Errorf("cannot pause engine %q: native model residency readback is unavailable", st.manifest.Engine)
	}
	raw, err := run(ctx, []string{cli, "ps", "--json"})
	var rows []json.RawMessage
	if err != nil || json.Unmarshal([]byte(raw), &rows) != nil || rows == nil {
		return fmt.Errorf("cannot pause engine %q: native model residency readback is unverified", st.manifest.Engine)
	}
	if len(rows) != 0 {
		return fmt.Errorf("cannot pause engine %q: native model residency remains", st.manifest.Engine)
	}
	return nil
}

// Shutdown proof cannot interpret an unclassifiable native row as unloaded.
// Ordinary inventory extraction may omit such rows; physical pause must refuse.
func commandResidentModels(raw json.RawMessage, spec *ActionResult) ([]string, bool) {
	var object map[string]json.RawMessage
	if json.Unmarshal(raw, &object) != nil {
		return nil, false
	}
	var rows []map[string]json.RawMessage
	if json.Unmarshal(object[spec.Array], &rows) != nil || rows == nil {
		return nil, false
	}
	for _, row := range rows {
		if spec.Match != nil {
			value, present := resolveObjectPath(row, spec.Match.Field)
			if !present {
				return nil, false
			}
			if spec.Match.Nonempty {
				var instances []json.RawMessage
				if json.Unmarshal(value, &instances) != nil || instances == nil {
					return nil, false
				}
			} else {
				var state *string
				if json.Unmarshal(value, &state) != nil || state == nil {
					return nil, false
				}
			}
			if !matchRow(row, spec.Match) {
				continue
			}
		}
		if nodepolicy.ValidateModelIdentifierJSON(row[spec.Field]) != nil {
			return nil, false
		}
	}
	return extractStringsResult(raw, spec)
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
			if err == nil && msg.Method == sleepMethod && !st.Running {
				// Sleep proved process termination and controlled residency under
				// opMu. This reply is positive evidence, unlike running:false alone.
				st.TerminationVerified = true
			}
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
