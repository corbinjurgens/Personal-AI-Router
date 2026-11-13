// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// stream.go is the serving half of cross-node error sync. Each node streams its
// local-origin snapshot to pinned peers at GET /v1/errors/events: the full
// SyncEnvelope at once on connect and again after every local change, with a
// null keepalive in between. Peers subscribe from peersync.go.

import (
	"net/http"
	"slices"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/errors"
)

var (
	// errorsStreamKeepalive is the interval between null keepalive lines.
	errorsStreamKeepalive = 10 * time.Second
)

const (
	// errorsStreamWriteTimeout bounds each frame write to a stalled peer.
	errorsStreamWriteTimeout = 5 * time.Second
)

// errorsPublisher holds the latest local-origin envelope for stream readers.
type errorsPublisher struct {
	// mu orders snapshot and publish, so two concurrent local changes cannot
	// publish an older snapshot after a newer one.
	mu  sync.Mutex
	hub clustertrust.SnapshotHub[errors.SyncEnvelope]
}

func newErrorsPublisher() *errorsPublisher {
	pub := &errorsPublisher{}
	pub.hub.Clone = func(env errors.SyncEnvelope) errors.SyncEnvelope {
		env.Errors = slices.Clone(env.Errors)
		return env
	}
	return pub
}

// publish offers mgr's current local-origin snapshot to every stream.
func (p *errorsPublisher) publish(mgr *Manager) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.hub.Publish(errors.SyncEnvelope{NodeID: mgr.LocalNodeID(), Errors: mgr.LocalSnapshot()})
}

// handleStream serves the snapshot stream to a caller that passed the pin gate.
// The gate runs again before every line, so a pin removed mid-stream ends it.
func handleStream(w http.ResponseWriter, r *http.Request, mesh *clustertrust.Mesh, pub *errorsPublisher) {
	updates, cancel, ok := pub.hub.Subscribe()
	if !ok {
		http.Error(w, "too many error streams", http.StatusServiceUnavailable)
		return
	}
	defer cancel()
	w.Header().Set("Cache-Control", "no-store")
	clustertrust.ServeSnapshotStream(w, r, clustertrust.SnapshotStreamServer[errors.SyncEnvelope]{
		Keepalive:    errorsStreamKeepalive,
		WriteTimeout: errorsStreamWriteTimeout,
		Authorize:    func() bool { return pinnedCaller(r, mesh) },
		Updates:      updates,
	})
}
