// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// models_stream.go streams this node's model inventory document (the
// GET /v1/models body) to pinned peers on GET /v1/models/events: the current
// document at once, again whenever it changes, and a null keepalive otherwise.
// A peer's scanner subscribes once instead of fetching /v1/models every 15 s.

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"sync/atomic"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/performancewire"
)

// modelsEventsPath is the pinned-peer model inventory stream on the em surface.
const modelsEventsPath = modelsPath + "/events"

// modelsStreamKeepalive is the idle keepalive on a peer models stream. A var so
// tests can shorten it; production never changes it.
var modelsStreamKeepalive = 10 * time.Second

// modelsRebuildEvery catches changes no engine:models-changed names, such as
// a policy push or download progress. A var so tests can shorten the period.
var modelsRebuildEvery = 15 * time.Second

const (
	// modelsStreamIdle is how long a subscriber waits for a line before it
	// drops the stream; peers use the same value.
	modelsStreamIdle = 25 * time.Second
	// modelsRebuildDelay coalesces a burst of engine:models-changed emissions.
	modelsRebuildDelay = 250 * time.Millisecond
	// modelsBuildTimeout bounds one inventory build.
	modelsBuildTimeout = 10 * time.Second
)

// modelsPublisher rebuilds the model inventory document and publishes it when
// its normalized form changes, only while a peer reads it.
type modelsPublisher struct {
	hub     clustertrust.SnapshotHub[json.RawMessage]
	notify  chan struct{}
	readers atomic.Int64
}

func newModelsPublisher() *modelsPublisher {
	return &modelsPublisher{
		hub:    clustertrust.SnapshotHub[json.RawMessage]{RetainWhileRead: true},
		notify: make(chan struct{}, 1),
	}
}

// subscribe adds a reader and requests a fresh rebuild without waiting for it.
// Readers joining after idle receive no seed from the previous reader epoch.
func (p *modelsPublisher) subscribe() (<-chan json.RawMessage, func(), bool) {
	p.readers.Add(1)
	ch, cancel, ok := p.hub.Subscribe()
	if !ok {
		p.readers.Add(-1)
		return nil, func() {}, false
	}
	p.signal()
	return ch, func() { cancel(); p.readers.Add(-1) }, true
}

// signal asks for a rebuild after modelsRebuildDelay. It never blocks.
func (p *modelsPublisher) signal() {
	if p == nil {
		return
	}
	select {
	case p.notify <- struct{}{}:
	default:
	}
}

// run rebuilds at start, every modelsRebuildEvery and modelsRebuildDelay after
// each signal until ctx ends, only while subscribed. build is Executor.ModelsResult.
func (p *modelsPublisher) run(ctx context.Context, build func(context.Context) ModelsResult) {
	var last []byte
	var lastEpoch uint64
	rebuild := func() {
		if p.readers.Load() == 0 {
			return
		}
		epoch := p.hub.Epoch()
		buildCtx, cancel := context.WithTimeout(ctx, modelsBuildTimeout)
		res := build(buildCtx)
		cancel()
		if ctx.Err() != nil {
			return
		}
		full, err := json.Marshal(res)
		if err != nil {
			slog.Warn("model inventory stream: encode failed", "err", err)
			return
		}
		normalized, err := json.Marshal(normalizedModelsResult(res))
		if err != nil || (epoch == lastEpoch && bytes.Equal(normalized, last)) {
			return
		}
		if p.hub.PublishAt(epoch, json.RawMessage(full)) {
			last, lastEpoch = normalized, epoch
		}
	}
	rebuild()
	ticker := time.NewTicker(modelsRebuildEvery)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			rebuild()
		case <-p.notify:
			// Signals arriving during the delay stay coalesced into this
			// rebuild; one arriving during the rebuild schedules another.
			timer := time.NewTimer(modelsRebuildDelay)
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case <-timer.C:
			}
			select {
			case <-p.notify:
			default:
			}
			rebuild()
		}
	}
}

// normalizedModelsResult returns res with each memory-evidence entry's
// per-build freshness fields cleared, the same comparison the loaded-model
// watcher uses before it emits engine:models-changed. res is not modified.
func normalizedModelsResult(res ModelsResult) ModelsResult {
	if res.MemoryEvidence == nil {
		return res
	}
	evidence := *res.MemoryEvidence
	evidence.Entries = make([]performancewire.MemoryEvidenceEntry, len(res.MemoryEvidence.Entries))
	for i, entry := range res.MemoryEvidence.Entries {
		evidence.Entries[i] = memoryStatusEntry(entry)
	}
	res.MemoryEvidence = &evidence
	return res
}

// modelsStreamHandler streams the model inventory document to a pinned peer.
// The caller wraps it in requirePinnedPeer; Authorize re-checks the pin before
// every frame, so a peer unpaired mid-stream loses it at the next line.
func modelsStreamHandler(mesh *clustertrust.Mesh, pub *modelsPublisher) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		ch, closeSub, ok := pub.subscribe()
		if !ok {
			http.Error(w, "too many model streams", http.StatusServiceUnavailable)
			return
		}
		defer closeSub()
		w.Header().Set("Cache-Control", "no-store")
		clustertrust.ServeSnapshotStream(w, r, clustertrust.SnapshotStreamServer[json.RawMessage]{
			Keepalive:    modelsStreamKeepalive,
			WriteTimeout: 5 * time.Second,
			Authorize: func() bool {
				mesh.Refresh()
				_, ok := mesh.VerifyClientPin(r)
				return ok
			},
			Updates: ch,
		})
	}
}
