// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// nodeprojection.go keeps the node projection the broker publishes with
// engine:node-projection: this node's policy advertisement (declared roles,
// policy revision, pause origin), which /v1/models and engine:models carry to
// paired peers. The broker owns the policy; engine-manager only relays what it
// was last sent. The projection lives in memory, is replaced whole on every
// push and is unknown until the first one, which the broker sends whenever
// engine-manager reports ready.

import (
	"encoding/json"
	"log/slog"
	"sync"

	"nvpair-shared/nodepolicy"
)

const nodeProjectionMethod = "engine:node-projection"

// nodeProjection is the engine:node-projection payload.
type nodeProjection struct {
	Policy nodepolicy.Advertisement `json:"policy"`
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

// applyNodeProjection stores a pushed projection. One that cannot be decoded
// leaves the previous projection in place.
func (m *Manager) applyNodeProjection(msg *Message) {
	var p nodeProjection
	if err := json.Unmarshal(msg.Params, &p); err != nil {
		slog.Warn("ignoring an unreadable node projection", "err", err)
		return
	}
	m.exec.nodeProjection.set(p)
}
