// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/noderec"
)

const maxSwitchWatchers = 16
const switchWatchBackoff = 250 * time.Millisecond

type switchWatchIdentity struct {
	peer  ecPeer
	route clustertrust.PeerServiceRoute
}

func (a switchWatchIdentity) same(b switchWatchIdentity) bool {
	return sameControlEndpoint(a.peer, b.peer) && a.peer.generation == b.peer.generation && a.route == b.route
}

func (m *Manager) switchWatchIdentity(node string) (switchWatchIdentity, bool) {
	peer, ok := m.peers.lookup(node)
	if !ok || m.mesh == nil {
		return switchWatchIdentity{}, false
	}
	route, ok := m.resolveControlRoute(peer.clusterUUID, noderec.ServiceEngineControl)
	return switchWatchIdentity{peer: peer, route: route}, ok
}

type switchWatchKey struct{ node, request string }
type switchWatch struct {
	sequence uint64
	cancel   context.CancelFunc
}
type switchWatchers struct {
	mu     sync.Mutex
	active map[switchWatchKey]*switchWatch
}

// observe consumes both apply and status envelopes. The manager lifetime context
// owns the watcher; cancellation never reaches the target-owned operation.
func (w *switchWatchers) observe(parent context.Context, identity switchWatchIdentity, raw json.RawMessage,
	lookup func(string) (switchWatchIdentity, bool), clientFor func(context.Context, ecPeer) (*remoteClient, error),
	notify func(string, any) error,
) bool {
	var envelope settings.SwitchStatus
	if json.Unmarshal(raw, &envelope) != nil || envelope.Receipt == nil {
		return false
	}
	r := envelope.Receipt
	if r.Validate() != nil || identity.peer.nodeID != r.NodeID || identity.route.Incarnation == "" {
		return false
	}
	key := switchWatchKey{r.NodeID, r.RequestID}
	w.mu.Lock()
	defer w.mu.Unlock()
	if existing := w.active[key]; existing != nil {
		if r.Sequence > existing.sequence {
			existing.sequence = r.Sequence
		}
		if r.Terminal {
			existing.cancel()
		}
		return false
	}
	deadline := r.AcceptedAt.Add(settings.SwitchOperationBudget)
	if r.Deadline.Before(deadline) {
		deadline = r.Deadline
	}
	if r.Terminal || parent.Err() != nil || !time.Now().Before(deadline) {
		return false
	}
	if len(w.active) >= maxSwitchWatchers {
		slog.Warn("remote switch watcher capacity reached")
		return false
	}
	ctx, cancel := context.WithDeadline(parent, deadline)
	watch := &switchWatch{sequence: r.Sequence, cancel: cancel}
	if w.active == nil {
		w.active = make(map[switchWatchKey]*switchWatch)
	}
	w.active[key] = watch
	go func() {
		defer cancel()
		defer func() { w.mu.Lock(); delete(w.active, key); w.mu.Unlock() }()
		failures := 0
		for ctx.Err() == nil {
			current, ok := lookup(key.node)
			if !ok || !identity.same(current) {
				return
			}
			w.mu.Lock()
			sequence := watch.sequence
			w.mu.Unlock()
			params, err := json.Marshal(settings.SwitchStatusRequest{NodeID: key.node, RequestID: key.request, AfterSequence: &sequence})
			if err != nil {
				return
			}
			round, stop := context.WithTimeout(ctx, settings.SwitchStatusWaitBudget+10*time.Second)
			client, err := clientFor(round, current.peer)
			var result json.RawMessage
			if err == nil {
				result, err = client.postSwitch(round, settings.SwitchStatusPath, params)
			}
			stop()
			// Recheck identity before publishing a reply from a retired route.
			live, ok := lookup(key.node)
			if ctx.Err() != nil || !ok || !identity.same(live) {
				return
			}
			if err != nil {
				failures++
				if failures >= 2 {
					return
				}
				select {
				case <-ctx.Done():
					return
				case <-time.After(switchWatchBackoff):
				}
				continue
			}
			failures = 0
			var status settings.SwitchStatus
			if json.Unmarshal(result, &status) != nil || status.Receipt == nil {
				return
			}
			next := status.Receipt
			if next.Validate() != nil || next.NodeID != key.node || next.RequestID != key.request || !next.AcceptedAt.Equal(r.AcceptedAt) {
				return
			}
			w.mu.Lock()
			newer := next.Sequence > watch.sequence
			if newer {
				watch.sequence = next.Sequence
			}
			w.mu.Unlock()
			if newer {
				if notify(settings.SwitchChangedNotification, settings.SwitchChanged{NodeID: key.node, RequestID: key.request, Sequence: next.Sequence}) != nil {
					return
				}
			}
			if next.Terminal {
				return
			}
		}
	}()
	return true
}
