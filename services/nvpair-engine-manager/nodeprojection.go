// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// nodeprojection.go keeps the node projection the broker publishes with
// engine:node-projection: this node's policy advertisement (declared roles,
// policy revision, pause origin), which /v1/models and engine:models carry to
// paired peers, and whether the node is paused locally. The broker owns the
// policy; engine-manager only relays what it was last sent. The projection
// lives in memory, is replaced whole on every push and is unknown until the
// first one, which the broker sends whenever engine-manager reports ready.
//
// It also holds the local veto: a node paused on the node itself, or an
// engine saved Off on the node itself, refuses a paired node's start,
// install-with-start and model load until it is resumed or started locally.
// Stop, unload, delete, pull and settings edits are never vetoed.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sync"

	"nvpair-shared/nodepolicy"
)

const nodeProjectionMethod = "engine:node-projection"

// nodeProjection is the engine:node-projection payload.
type nodeProjection struct {
	Policy nodepolicy.Advertisement `json:"policy"`
	// PausedLocally is true while the node is paused by its own clients, a
	// pause with no recorded provenance included.
	PausedLocally bool `json:"pausedLocally"`
}

// nodeProjectionStore holds the last projection the broker pushed.
type nodeProjectionStore struct {
	mu         sync.Mutex
	projection *nodeProjection
}

func (s *nodeProjectionStore) set(p nodeProjection) {
	s.mu.Lock()
	s.projection = &p
	s.mu.Unlock()
}

// advertisement returns the last pushed advertisement, or nil before the first
// push. A projection is never modified after it is stored, so the copy may
// share its maps.
func (s *nodeProjectionStore) advertisement() *nodepolicy.Advertisement {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.projection == nil {
		return nil
	}
	adv := s.projection.Policy
	return &adv
}

// pausedLocally reports whether the last projection says the node is paused
// locally; known is false before the first push.
func (s *nodeProjectionStore) pausedLocally() (paused, known bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.projection == nil {
		return false, false
	}
	return s.projection.PausedLocally, true
}

// applyNodeProjection stores a pushed projection and cancels the paired nodes'
// accepted work it now vetoes. One that cannot be decoded leaves the previous
// projection in place.
func (m *Manager) applyNodeProjection(msg *Message) {
	var p nodeProjection
	if err := json.Unmarshal(msg.Params, &p); err != nil {
		slog.Warn("ignoring an unreadable node projection", "err", err)
		return
	}
	m.exec.nodeProjection.set(p)
	m.exec.cancelVetoedRemoteWork()
}

// localVetoError refuses a paired node's start or load that this node's own
// pause or Off forbids. It reaches the caller as JSON-RPC codeLocalVeto.
type localVetoError struct{ reason string }

func (e *localVetoError) Error() string { return e.reason }

const nodePausedVeto = "this node was paused on the node itself; resume it there"

func engineOffVeto(engine string) string {
	return fmt.Sprintf("%s was turned off on the node itself; start it there", engine)
}

// errNoProjection refuses a paired node's start or load before the broker has
// told engine-manager whether the node is paused: until then a local pause
// cannot be ruled out.
var errNoProjection = errors.New("this node's policy is not known yet; try again shortly")

// remoteVeto is why a paired node may not start engine or load a model into it
// now, or nil when it may. A node paused locally and an engine saved Off
// locally (an Off with no provenance included) refuse with localVetoError; a
// node without a projection yet refuses with errNoProjection.
func (e *Executor) remoteVeto(engine string) error {
	paused, known := e.nodeProjection.pausedLocally()
	if !known {
		return errNoProjection
	}
	if paused {
		return &localVetoError{reason: nodePausedVeto}
	}
	off, err := e.desired.offLocally(engine)
	if err != nil {
		return err
	}
	if off {
		return &localVetoError{reason: engineOffVeto(engine)}
	}
	return nil
}

// remoteWork tracks the paired nodes' accepted starts and loads until they
// finish, so a local veto that arrives meanwhile cancels them.
type remoteWork struct {
	mu   sync.Mutex
	next uint64
	ops  map[uint64]remoteOp
}

type remoteOp struct {
	engine string
	cancel context.CancelCauseFunc
}

// beginRemote admits one paired node's start or load of engine. It refuses
// with remoteVeto's error, or returns a context derived from ctx that a later
// veto cancels with the localVetoError as its cause, and the func that ends
// the work. Admission and cancellation share one lock, so a veto stored after
// the check always finds the work registered.
func (e *Executor) beginRemote(ctx context.Context, engine string) (context.Context, func(), error) {
	w := &e.remoteWork
	w.mu.Lock()
	defer w.mu.Unlock()
	if err := e.remoteVeto(engine); err != nil {
		return nil, nil, err
	}
	ctx, cancel := context.WithCancelCause(ctx)
	if w.ops == nil {
		w.ops = make(map[uint64]remoteOp)
	}
	w.next++
	id := w.next
	w.ops[id] = remoteOp{engine: engine, cancel: cancel}
	return ctx, func() {
		w.mu.Lock()
		delete(w.ops, id)
		w.mu.Unlock()
		cancel(nil)
	}, nil
}

// cancelVetoedRemoteWork cancels every accepted remote start or load that
// remoteVeto now refuses with a local veto.
func (e *Executor) cancelVetoedRemoteWork() {
	w := &e.remoteWork
	w.mu.Lock()
	defer w.mu.Unlock()
	for id, op := range w.ops {
		var veto *localVetoError
		if errors.As(e.remoteVeto(op.engine), &veto) {
			slog.Info("cancelling a paired node's operation after a local veto", "engine", op.engine)
			op.cancel(veto)
			delete(w.ops, id)
		}
	}
}

// cancelEngine cancels every accepted remote start or load of engine with veto.
func (w *remoteWork) cancelEngine(engine string, veto *localVetoError) {
	w.mu.Lock()
	defer w.mu.Unlock()
	for id, op := range w.ops {
		if op.engine == engine {
			slog.Info("cancelling a paired node's operation after a local stop", "engine", engine)
			op.cancel(veto)
			delete(w.ops, id)
		}
	}
}

// vetoCause returns the local veto that canceled ctx when err came from that
// cancellation, and err otherwise.
func vetoCause(ctx context.Context, err error) error {
	var veto *localVetoError
	if err != nil && errors.As(context.Cause(ctx), &veto) {
		return veto
	}
	return err
}
