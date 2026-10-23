// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// enginelock.go is the local-only engine:set-lock surface. A lock is a
// per-engine activation prohibition only a client on the engine's own node may
// set or clear: no ec route reaches it and the request names no node, caller
// or origin.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

const setLockMethod = "engine:set-lock"

// setLockParams is the engine:set-lock request.
type setLockParams struct {
	Engine string `json:"engine"`
	Locked *bool  `json:"locked"`
}

// parseSetLock decodes an engine:set-lock request strictly: one object with a
// registered engine and a boolean locked, and no other member. A node, caller
// or origin field is unknown, so it cannot turn the request into a remote one.
// Member names must be exact and unique, because encoding/json would otherwise
// accept a case variant or let a repeated member silently override the first.
func (e *Executor) parseSetLock(raw json.RawMessage) (engine string, locked bool, err error) {
	if err := checkSetLockMembers(raw); err != nil {
		return "", false, err
	}
	var p setLockParams
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&p); err != nil {
		return "", false, fmt.Errorf("invalid params: %w", err)
	}
	if decoder.Decode(&struct{}{}) != io.EOF {
		return "", false, errors.New("invalid params: trailing data")
	}
	if p.Engine == "" {
		return "", false, errors.New("engine is required")
	}
	if _, ok := e.reg.Get(p.Engine); !ok {
		return "", false, fmt.Errorf("unknown engine %q", p.Engine)
	}
	if p.Locked == nil {
		return "", false, errors.New("locked must be true or false")
	}
	return p.Engine, *p.Locked, nil
}

// checkSetLockMembers requires raw to be one JSON object whose members are
// exactly named engine or locked, each at most once.
func checkSetLockMembers(raw json.RawMessage) error {
	invalid := errors.New("invalid params: expected an object with engine and locked")
	decoder := json.NewDecoder(bytes.NewReader(raw))
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		return invalid
	}
	seen := map[string]bool{}
	for decoder.More() {
		token, err := decoder.Token()
		name, ok := token.(string)
		if err != nil || !ok {
			return invalid
		}
		if name != "engine" && name != "locked" {
			return fmt.Errorf("invalid params: unknown field %q", name)
		}
		if seen[name] {
			return fmt.Errorf("invalid params: duplicate field %q", name)
		}
		seen[name] = true
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return invalid
		}
	}
	return nil
}

func (e *Executor) setEngineLocked(engine string, locked bool) error {
	st, err := e.state(engine)
	if err != nil {
		return err
	}
	st.lockMu.Lock()
	defer st.lockMu.Unlock()
	return e.setEngineLockedCommand(st, engine, locked)
}

// Caller holds lockMu through its acknowledgement. Ordinary mutation admission
// remains live until physical stop and state publication have finished.
func (e *Executor) setEngineLockedCommand(st *engineState, engine string, locked bool) error {
	done, err := e.admitMutation(engine)
	if err != nil {
		return err
	}
	defer done()
	intentChanged, err := e.desired.setLocked(engine, locked)
	if err != nil {
		return fmt.Errorf("persist %s lock state: %w", engine, err)
	}
	if intentChanged {
		e.emitIntent()
	}
	e.emitState(engine)
	if !locked {
		return nil
	}
	veto := &localVetoError{code: codeEngineLocked, reason: engineLockedVeto(engine)}
	e.remoteWork.cancelEngine(engine, veto)
	st.mu.Lock()
	if st.startCancel != nil {
		st.startCancel(veto)
	}
	st.mu.Unlock()
	relaunchDone := cancelCrashRelaunch(st)
	finishLoads := stopServingLoadsWithCause(st, veto)
	defer finishLoads()
	if relaunchDone != nil {
		<-relaunchDone
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	installed, _ := e.Detect(engine)
	st.mu.Lock()
	port := st.port
	st.mu.Unlock()
	// Reclaim by managed image and private port before readiness adoption: an
	// orphan may be unhealthy or absent from this manager's runtime state.
	if reclaimed, err := e.reclaimOrphan(context.Background(), st, engine, port); err != nil {
		st.mu.Lock()
		pid := st.reclaimPID
		st.mu.Unlock()
		running := !e.waitUnavailable(st.plat.Runtime.Ready, port, presenceRefusalWindow) || !orphanPIDExited(pid)
		st.mu.Lock()
		st.reclaimFailed = true
		st.running, st.healthy, st.adopted = running, false, running
		st.mu.Unlock()
		e.emitState(engine)
		return fmt.Errorf("lock %s: %w", engine, err)
	} else if !reclaimed {
		e.reconcilePresence(context.Background(), engine, st, installed, port, true)
	}
	if err := e.doStop(st, engine); err != nil {
		return fmt.Errorf("lock %s: %w", engine, err)
	}
	if !e.switchTerminated(st) {
		e.emitState(engine)
		return fmt.Errorf("lock %s: termination could not be verified", engine)
	}
	return nil
}

// runSetLock answers the local-only engine:set-lock method.
func (m *Manager) runSetLock(msg *Message) {
	go m.setLockCommand(msg)
}

func (m *Manager) setLockCommand(msg *Message) {
	engine, locked, err := m.exec.parseSetLock(msg.Params)
	if err != nil {
		m.codec.RespondError(msg.ID, -32602, err.Error())
		return
	}
	st, err := m.exec.state(engine)
	if err != nil {
		m.codec.RespondError(msg.ID, rpcCode(err), err.Error())
		return
	}
	st.lockMu.Lock()
	defer st.lockMu.Unlock()
	if err := m.exec.setEngineLockedCommand(st, engine, locked); err != nil {
		m.codec.RespondError(msg.ID, rpcCode(err), err.Error())
		return
	}
	m.codec.Respond(msg.ID, nil)
}
