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
)

const (
	wakeMethod          = "engine:wake"
	sleepMethod         = "engine:sleep"
	intentMethod        = "engine:intent"
	intentChangedMethod = "engine:intent-changed"
	unloadModelMethod   = "engine:unload-model"
)

// errIntentOff is returned by Wake for an engine whose saved intent is not On.
// An engine with no saved intent counts as Off, matching RestoreEnabled.
var errIntentOff = errors.New("engine is not saved On")

// intentParams is the engine:intent result and the engine:intent-changed
// payload: every registered engine's saved On/Off intent. An engine with no
// saved intent is reported Off, because nothing starts it automatically.
type intentParams struct {
	EnabledByEngine map[string]bool `json:"enabledByEngine"`
}

// Wake starts an engine only if its saved intent is On, without changing that
// intent. It holds the engine's lifecycle lock like Restart, so a concurrent
// explicit Stop either finishes first (and Wake then sees intent Off) or waits
// for the start to finish. A shutting-down manager refuses, like every start.
func (e *Executor) Wake(ctx context.Context, engine string) error {
	st, err := e.state(engine)
	if err != nil {
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
	st, err := e.state(engine)
	if err != nil {
		return err
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	return e.doStop(st, engine)
}

// Intent reports every registered engine's saved intent. It reads only the
// intent file, never an engine's lifecycle lock, so it answers immediately even
// while an engine is starting.
func (e *Executor) Intent() (intentParams, error) {
	out := intentParams{EnabledByEngine: make(map[string]bool)}
	for _, name := range e.reg.Names() {
		enabled, known, err := e.desired.get(name)
		if err != nil {
			return intentParams{}, err
		}
		out.EnabledByEngine[name] = known && enabled
	}
	return out, nil
}

// emitIntent pushes the current saved intent after it changes. Best-effort: a
// read failure is reported through the error the caller already returns.
func (e *Executor) emitIntent() {
	if intent, err := e.Intent(); err == nil {
		e.notify(intentChangedMethod, intent)
	}
}

// handleLifecyclePolicyRequest serves engine:wake, engine:sleep, engine:intent
// and engine:unload-model. It reports whether it took the method.
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
	case unloadModelMethod:
		go func() {
			var p modelActionRequest
			if !m.parse(msg, &p) {
				return
			}
			if p.Engine == "" || p.Model == "" {
				m.codec.RespondError(msg.ID, -32602, "engine and model are required")
				return
			}
			res, err := m.exec.ModelUnload(ctx, p.Engine, p.Model)
			if err != nil {
				m.codec.RespondError(msg.ID, -32000, err.Error())
				return
			}
			m.exec.pokeLoaded()
			m.codec.Respond(msg.ID, res)
		}()
		return true
	}
	return false
}
