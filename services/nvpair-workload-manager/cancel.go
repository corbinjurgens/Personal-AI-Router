// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// cancel.go carries workloads:cancel to the node a job originated on (spec
// §7.0). Only the origin's proxy holds the request, so a cancel issued anywhere
// else travels over the same pinned inter-node channel as lifecycle events: to
// the origin when it is a known peer, otherwise to every peer. A receiving
// manager hands it to its broker, which acts only if it is the origin.

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"

	"nvpair-shared/nodepolicy"
)

// MethodCancel is accepted on stdin from the broker and on the inter-node
// interface from peers, and emitted on stdout to the broker.
const MethodCancel = "workloads:cancel"

// cancelParams is the workloads:cancel payload. CancelID is minted by the
// manager that first sends the cancel, so a retried delivery is recognized as
// the same cancel while a user cancelling twice is not.
type cancelParams struct {
	OriginatedFrom string `json:"originatedFrom"`
	WorkloadID     string `json:"workloadId"`
	Regenerate     bool   `json:"regenerate,omitempty"`
	// Engine and RunID narrow the target: workload ids are per-engine counters
	// that restart with the proxy, so the id alone can name two jobs.
	Engine   string `json:"engine"`
	RunID    string `json:"runId"`
	CancelID string `json:"cancelId,omitempty"`
}

func parseCancel(params json.RawMessage) (cancelParams, error) {
	var p cancelParams
	if err := json.Unmarshal(params, &p); err != nil {
		return cancelParams{}, fmt.Errorf("invalid params: %w", err)
	}
	if err := (nodepolicy.WorkloadCancelParams{OriginatedFrom: p.OriginatedFrom, WorkloadID: p.WorkloadID, Engine: p.Engine, RunID: p.RunID}).Validate(); err != nil {
		return cancelParams{}, err
	}
	return p, nil
}

// keyCancel is the receiver's dedup key for one cancel.
func keyCancel(p cancelParams) string {
	key, _ := json.Marshal([5]string{p.OriginatedFrom, p.Engine, p.RunID, p.WorkloadID, p.CancelID})
	return "cx" + string(key)
}

var cancelBroadcastAdmission = make(chan struct{}, 32)
var cancelDeliveryAdmission = make(chan struct{}, 32)

func newCancelID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// handleLocalCancel sends a broker's workloads:cancel toward the origin. It is
// not queued behind lifecycle broadcasts: a cancel is about one job on one
// node, and making it wait for an unrelated slow round would defeat it.
func (m *Manager) handleLocalCancel(msg *Message) {
	p, err := parseCancel(msg.Params)
	if err != nil {
		slog.Warn("dropping malformed local cancel", "err", err)
		return
	}
	p.CancelID = newCancelID()
	params, err := json.Marshal(p)
	if err != nil {
		return
	}
	frame, err := json.Marshal(&Message{JSONRPC: "2.0", Method: MethodCancel, Params: params})
	if err != nil {
		return
	}
	ctx := m.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	origin := p.OriginatedFrom
	known := m.peers.has(origin)
	slog.Info("relaying workload cancel", "origin", origin, "workloadId", p.WorkloadID, "direct", known)
	select {
	case cancelBroadcastAdmission <- struct{}{}:
	default:
		slog.Warn("dropping workload cancel: broadcast capacity exhausted")
		return
	}
	go func() {
		defer func() { <-cancelBroadcastAdmission }()
		m.broadcaster.BroadcastTo(ctx, frame, func(t target) bool { return !known || t.id == origin })
	}()
}

// handleCancel serves a peer's workloads:cancel: validate, dedup by cancel id,
// and hand it to the broker. Whether this node is the origin is the broker's
// decision, since it owns the node identity jobs are stamped with.
func (s *Server) handleCancel(w http.ResponseWriter, msg *Message) {
	p, err := parseCancel(msg.Params)
	if err != nil {
		s.badRequest(w, err.Error())
		return
	}
	if p.CancelID == "" || len(p.CancelID) > 128 {
		s.badRequest(w, "cancelId is required and must be at most 128 bytes")
		return
	}
	if s.emitCancel == nil {
		http.Error(w, "broker unavailable", http.StatusInternalServerError)
		return
	}
	duplicate, err := s.dedup.emitOnce(keyCancel(p), func() error { return s.emitCancel(p) })
	if duplicate {
		s.ok(w)
		return
	}
	if err != nil {
		slog.Error("failed to emit workloads:cancel", "workloadId", p.WorkloadID, "err", err)
		http.Error(w, "broker unavailable", http.StatusInternalServerError)
		return
	}
	slog.Info("relayed peer workload cancel", "origin", p.OriginatedFrom, "workloadId", p.WorkloadID, "regenerate", p.Regenerate)
	s.ok(w)
}

// emitCancel forwards a peer's cancel to the broker.
func (m *Manager) emitCancel(p cancelParams) error {
	return m.codec.Notify(MethodCancel, p)
}

// BroadcastTo posts frame to the current peers keep selects, concurrently and
// with the same retry budget and cluster gate as Broadcast.
func (b *Broadcaster) BroadcastTo(ctx context.Context, frame []byte, keep func(target) bool) {
	b.mesh.Refresh()
	if !b.mesh.Clustered() {
		b.clients.CloseIdle()
		slog.Debug("targeted send skipped: node is not a cluster member")
		return
	}
	var wg sync.WaitGroup
	for _, t := range b.peers.targets() {
		if !keep(t) {
			continue
		}
		select {
		case cancelDeliveryAdmission <- struct{}{}:
		default:
			slog.Warn("dropping workload cancel delivery: control capacity exhausted")
			continue
		}
		wg.Add(1)
		go func(t target) {
			defer wg.Done()
			defer func() { <-cancelDeliveryAdmission }()
			b.deliver(ctx, t, frame)
		}(t)
	}
	wg.Wait()
}

// has reports whether id is a current broadcast target.
func (p *peerSet) has(id string) bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	_, ok := p.peers[id]
	return ok
}
