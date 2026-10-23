// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// enginelock.go is the local-only engine:set-lock surface. A lock is a
// per-engine activation prohibition only a client on the engine's own node may
// set or clear: no ec route reaches it and the request names no node, caller
// or origin. The request contract is fixed here; until lock enforcement lands
// the method refuses every well-formed request without changing state.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

const setLockMethod = "engine:set-lock"

// engineLockUnavailableMessage is the stub's refusal of a well-formed request.
const engineLockUnavailableMessage = "Engine locking is not available yet."

// setLockParams is the engine:set-lock request.
type setLockParams struct {
	Engine string `json:"engine"`
	Locked *bool  `json:"locked"`
}

// parseSetLock decodes an engine:set-lock request strictly: one object with a
// registered engine and a boolean locked, and no other member. A node, caller
// or origin field is unknown, so it cannot turn the request into a remote one.
func (e *Executor) parseSetLock(raw json.RawMessage) (engine string, locked bool, err error) {
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

// runSetLock answers engine:set-lock. A malformed request is invalid params;
// a well-formed one is refused as unavailable and changes nothing.
func (m *Manager) runSetLock(msg *Message) {
	if _, _, err := m.exec.parseSetLock(msg.Params); err != nil {
		m.codec.RespondError(msg.ID, -32602, err.Error())
		return
	}
	m.codec.RespondError(msg.ID, -32000, engineLockUnavailableMessage)
}
