// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
	"nvpair-shared/performancewire"
)

// performanceStreamPath serves this node's latest performance summary to
// pinned peers as an NDJSON snapshot stream.
const performanceStreamPath = "/v1/workloads/performance/events"

// performanceStreamKeepalive is a variable so tests can shorten it.
var performanceStreamKeepalive = 10 * time.Second

const (
	performanceStreamWriteTimeout = 5 * time.Second
	// performanceStreamIdle drops a subscriber stream that sent no line,
	// keepalives included, within this interval.
	performanceStreamIdle = 25 * time.Second
	// performanceStreamConnect bounds connect and response headers only; the
	// stream itself has no total timeout.
	performanceStreamConnect = 10 * time.Second
	// performanceStreamBackoffMax caps the reconnect delay, which starts at 1 s.
	performanceStreamBackoffMax = 30 * time.Second
	// performanceStatusRefresh is how often keepalives re-assert "fresh" to
	// the broker, the cadence the former 30 s heartbeat gave it.
	performanceStatusRefresh = 30 * time.Second
	// performanceSweepEvery runs the stale sweep and re-evaluates subscriptions.
	performanceSweepEvery = 30 * time.Second
	// performanceStreamMaxLine fits one frame: a summary at its payload limit
	// plus the frame's own fields.
	performanceStreamMaxLine = performancewire.MaxPayloadBytes + 1024
)

// performanceFrame is one stream line: the sender's latest summary and its
// age in milliseconds when the line was built.
type performanceFrame struct {
	Summary   json.RawMessage `json:"summary"`
	ElapsedMs int64           `json:"elapsedMs"`
}

// decodePerformanceFrame strictly decodes one non-keepalive stream line. It
// requires both fields and an elapsedMs within 0..RetentionMs; the summary
// itself is validated by applySummary.
func decodePerformanceFrame(line []byte) (performanceFrame, error) {
	var in struct {
		Summary   json.RawMessage `json:"summary"`
		ElapsedMs *int64          `json:"elapsedMs"`
	}
	decoder := json.NewDecoder(bytes.NewReader(line))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&in); err != nil {
		return performanceFrame{}, err
	}
	if decoder.Decode(&struct{}{}) != io.EOF {
		return performanceFrame{}, errors.New("trailing data after performance frame")
	}
	if len(in.Summary) == 0 || in.ElapsedMs == nil {
		return performanceFrame{}, errors.New("performance frame needs summary and elapsedMs")
	}
	if *in.ElapsedMs < 0 || *in.ElapsedMs > performancewire.RetentionMs {
		return performanceFrame{}, errors.New("invalid performance elapsed time")
	}
	return performanceFrame{Summary: in.Summary, ElapsedMs: *in.ElapsedMs}, nil
}

// handlePerformanceStream serves GET /v1/workloads/performance/events behind
// the same pin gate as the events endpoint, rechecked before every line.
func (s *Server) handlePerformanceStream(w http.ResponseWriter, r *http.Request) {
	if !s.pinnedPeer(r) {
		http.Error(w, "forbidden: not a pinned cluster peer", http.StatusForbidden)
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if s.performance == nil {
		http.Error(w, "broker unavailable", http.StatusServiceUnavailable)
		return
	}
	s.performance.serveStream(w, r, func() bool { return s.pinnedPeer(r) })
}

// serveStream writes a frame for each hub token, the latest at once, built
// from snapshotForSend at that moment so elapsedMs is current. A withheld
// summary writes nothing and keeps the stream open; keepalives continue.
func (p *performanceRelay) serveStream(w http.ResponseWriter, r *http.Request, authorize func() bool) {
	tokens, unsubscribe, ok := p.hub.Subscribe()
	if !ok {
		http.Error(w, "too many performance streams", http.StatusServiceUnavailable)
		return
	}
	defer unsubscribe()
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	frames := make(chan performanceFrame)
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case <-tokens:
			}
			raw, elapsed := p.snapshotForSend()
			if raw == nil {
				continue
			}
			select {
			case frames <- performanceFrame{Summary: raw, ElapsedMs: elapsed}:
			case <-ctx.Done():
				return
			}
		}
	}()
	// Send the response headers now: with nothing published (or a withheld
	// summary) the first line is a keepalive, which would otherwise arrive only
	// as the subscriber's connect bound expires.
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	if http.NewResponseController(w).Flush() != nil {
		return
	}
	clustertrust.ServeSnapshotStream(w, r, clustertrust.SnapshotStreamServer[performanceFrame]{
		Keepalive:    performanceStreamKeepalive,
		WriteTimeout: performanceStreamWriteTimeout,
		Authorize:    authorize,
		Updates:      frames,
	})
}

// run owns the peer subscriptions. It re-evaluates them on wake (discovery and
// membership changes) and, with the stale sweep, every performanceSweepEvery.
func (p *performanceRelay) run(ctx context.Context) {
	ticker := time.NewTicker(performanceSweepEvery)
	defer ticker.Stop()
	active := make(map[string]context.CancelFunc)
	defer func() {
		for _, cancel := range active {
			cancel()
		}
	}()
	p.syncStreams(ctx, active)
	for {
		select {
		case <-ctx.Done():
			return
		case <-p.wake:
			p.syncStreams(ctx, active)
		case <-ticker.C:
			p.sweep()
			p.syncStreams(ctx, active)
		}
	}
}

// syncStreams keeps one subscription per admitted peer target, keyed by the
// peer's cluster UUID, and cancels those whose peer left or lost admission.
func (p *performanceRelay) syncStreams(ctx context.Context, active map[string]context.CancelFunc) {
	b := p.broadcaster
	if b == nil || p.streams == nil {
		return
	}
	b.mesh.Refresh()
	want := make(map[string]bool)
	for _, t := range b.peers.targets() {
		// Fail closed on legacy aliases, never trusting a relay mapping to
		// assert that somebody else's host UUID belongs to this principal.
		if t.uuid == "" || t.id != t.uuid {
			continue
		}
		if _, admitted := b.mesh.PeerServiceIncarnation(t.uuid); admitted {
			want[t.uuid] = true
		}
	}
	for uuid, cancel := range active {
		if !want[uuid] {
			cancel()
			delete(active, uuid)
		}
	}
	for uuid := range want {
		if active[uuid] != nil || len(active) >= performancePeerLimit {
			continue
		}
		streamCtx, cancel := context.WithCancel(ctx)
		active[uuid] = cancel
		go p.keepStream(streamCtx, uuid)
	}
}

// keepStream reads uuid's stream until ctx ends, reconnecting after 1 s,
// doubling to performanceStreamBackoffMax while attempts deliver no line.
func (p *performanceRelay) keepStream(ctx context.Context, uuid string) {
	backoff := time.Second
	for ctx.Err() == nil {
		if p.consumeStream(ctx, uuid) {
			backoff = time.Second
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(backoff):
		}
		backoff = min(2*backoff, performanceStreamBackoffMax)
	}
}

// consumeStream opens and reads one performance stream from uuid and reports
// whether any line arrived. The stream is bound to the peer's admission
// identity and route generation at connect: either changing ends it.
func (p *performanceRelay) consumeStream(parent context.Context, uuid string) bool {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	b := p.broadcaster
	b.mesh.Refresh()
	identity, admitted := b.mesh.PeerServiceIncarnation(uuid)
	t, generation, routed := b.peers.targetUUID(uuid)
	if !admitted || !routed || t.id != uuid {
		return false
	}
	current := func() bool {
		b.mesh.Refresh()
		now, admitted := b.mesh.PeerServiceIncarnation(uuid)
		_, nowGeneration, routed := b.peers.targetUUID(uuid)
		return admitted && routed && now == identity && nowGeneration == generation
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://"+b.addrs.Prefer(t.id, t.candidates)+performanceStreamPath, nil)
	if err != nil {
		return false
	}
	connect := time.AfterFunc(performanceStreamConnect, cancel)
	resp, err := p.streams.Client(uuid, noderec.ServiceWorkload).Do(req)
	connect.Stop()
	if err != nil {
		b.addrs.Forget(t.id)
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false
	}
	authority := p.streamAuthority(uuid, identity, current)
	received, accepted := false, false
	err = clustertrust.ReadSnapshotStream(ctx, cancel, resp.Body, clustertrust.SnapshotStreamClient[performanceFrame]{
		Idle:    performanceStreamIdle,
		MaxLine: performanceStreamMaxLine,
		Current: current,
		Decode:  decodePerformanceFrame,
		OnFrame: func(frame performanceFrame) error {
			received = true
			code, message := p.applySummary(authority, frame.Summary, frame.ElapsedMs)
			accepted = code == http.StatusOK
			if !accepted {
				// The reconnect starts with the full summary again, which is
				// also the retry for a broker write failure or full receiver.
				return fmt.Errorf("summary refused: %d %s", code, message)
			}
			return nil
		},
		OnKeepalive: func() {
			received = true
			if accepted {
				p.refreshReceipt(uuid, authority)
			}
		},
	})
	if err != nil && !errors.Is(err, context.Canceled) {
		slog.Debug("performance stream ended", "nodeId", uuid, "err", err)
	}
	return received
}

// streamAuthority is applySummary's authority for a stream from uuid: the
// principal is the pinned peer the stream was opened to, its identity is the
// admission identity at connect, and the summary's epoch must equal the
// peer's current admission epoch, as for a request carrying its certificate.
func (p *performanceRelay) streamAuthority(uuid, identity string, current func() bool) func(string) (string, string, bool) {
	return func(epochText string) (string, string, bool) {
		if !current() {
			return uuid, identity, false
		}
		epoch, canonical := canonicalPerformanceEpoch(epochText)
		admitted, ok := p.mesh.PeerAdmissionEpoch(uuid)
		return uuid, identity, canonical && ok && admitted == epoch
	}
}
