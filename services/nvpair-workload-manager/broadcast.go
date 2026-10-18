// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"path/filepath"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
	"nvpair-shared/reach"
)

const (
	// perPeerTimeout bounds a single POST so one slow/partitioned peer can't
	// hold up the fan-out (spec §10, §12).
	perPeerTimeout = 3 * time.Second
	// maxBroadcastAttempts is the bounded retry budget per peer. After the
	// last attempt the event is dropped for that peer with a logged final
	// drop (spec §12) — peers reconcile via later events (no replay).
	maxBroadcastAttempts = 3
	// retryBaseDelay is the first backoff interval; it doubles each retry.
	retryBaseDelay = 200 * time.Millisecond
	// maxDrainBytes bounds draining the empty success or short error response.
	maxDrainBytes = 8 << 10
)

// Broadcaster fans a local JSON-RPC notification out to every peer over the
// inter-node interface. Delivery is best-effort: concurrent, per-peer
// timeouts, bounded exponential-backoff retries.
//
// The channel is cluster mTLS, unconditionally: each peer is dialed with a client
// that presents our cluster leaf and pins that peer's exact server cert, and a
// peer we hold no pin for is skipped (the client-side cluster gate). A node that
// belongs to no cluster broadcasts nothing — it does not fall back to posting
// workload events in the clear. Membership is re-derived at the start of every
// round, so a node that joins or leaves mid-session starts (or stops) addressing
// its peers on the next event rather than at its next restart.
//
// The shared adapter prefers direct mTLS and falls back to a pinned reverse
// session before dispatch. Each application retry is a new HTTP attempt;
// the transport never replays a dispatched mutation.
type Broadcaster struct {
	peers   *peerSet
	mesh    *clustertrust.Mesh
	clients *clustertrust.PeerServiceClient
	addrs   *reach.Chooser
}

func NewBroadcaster(peers *peerSet, mesh *clustertrust.Mesh, clusterDir string) *Broadcaster {
	b := &Broadcaster{
		peers: peers,
		mesh:  mesh,
		addrs: reach.NewChooser(),
	}
	base := ""
	if clusterDir != "" {
		base = filepath.Dir(clusterDir)
	}
	b.clients = clustertrust.NewPeerServiceClient(mesh, clustertrust.PeerServiceClientOptions{
		PeerClientOptions: clustertrust.PeerClientOptions{Timeout: perPeerTimeout},
		BaseConfigDir:     base,
		ResolveRoute:      b.resolveRoute,
		// Only post uses this client: nonempty, known-length event/cancel
		// POSTs whose receiver owns deduplication. The adapter never replays.
		ReuseConnections: true,
	})
	peers.mu.Lock()
	previousOnChange := peers.onChange
	peers.onChange = func() {
		if previousOnChange != nil {
			previousOnChange()
		}
		b.clients.CloseIdle()
	}
	peers.mu.Unlock()
	return b
}

// resolveRoute combines shared admission authority with the current discovery
// endpoint at every adapter authorization boundary, including after TLS setup.
func (b *Broadcaster) resolveRoute(uuid string, service noderec.ServiceKey) (clustertrust.PeerServiceRoute, bool) {
	fail := func() (clustertrust.PeerServiceRoute, bool) { return clustertrust.PeerServiceRoute{}, false }
	b.mesh.Refresh()
	if service != noderec.ServiceWorkload {
		return fail()
	}
	incarnation, ok := b.mesh.PeerServiceIncarnation(uuid)
	if !ok {
		return fail()
	}
	t, generation, ok := b.peers.targetUUID(uuid)
	if !ok {
		return fail()
	}
	return clustertrust.PeerServiceRoute{
		Incarnation: incarnation,
		Address:     b.addrs.Prefer(t.id, t.candidates),
		Protocol:    clustertrust.ReverseTLS,
		Generation:  generation,
	}, true
}

// Broadcast posts the already-marshaled notification frame to every current
// peer concurrently and returns once all peers have either succeeded or
// exhausted their retry budget. ctx cancellation (parent shutdown) aborts
// in-flight attempts.
func (b *Broadcaster) Broadcast(ctx context.Context, frame []byte) {
	// Re-derive membership and pins before the fan-out.
	b.mesh.Refresh()
	if !b.mesh.Clustered() {
		b.clients.CloseIdle()
		// Not a member: there is no channel to fan out on. Bail before building
		// targets so this is unmistakably a no-op rather than a failed round.
		slog.Debug("broadcast skipped: node is not a cluster member")
		return
	}

	targets := b.peers.targets()
	if len(targets) == 0 {
		slog.Debug("broadcast skipped: no peers")
		return
	}

	var wg sync.WaitGroup
	for _, t := range targets {
		wg.Add(1)
		go func(t target) {
			defer wg.Done()
			b.deliver(ctx, t, frame)
		}(t)
	}
	wg.Wait()
}

// deliver attempts a single peer with bounded exponential backoff. It resolves
// the current peer endpoint and pinned service client on each attempt.
func (b *Broadcaster) deliver(ctx context.Context, t target, frame []byte) {
	delay := retryBaseDelay
	for attempt := 1; attempt <= maxBroadcastAttempts; attempt++ {
		if ctx.Err() != nil {
			return
		}
		// Re-resolve per attempt, not once for the whole budget. The retry window
		// spans seconds, and a peer removed from the cluster inside it must stop
		// being dialed immediately rather than keep receiving our events until the
		// budget runs out.
		current, _, ok := b.peers.targetUUID(t.uuid)
		b.mesh.Refresh()
		if !ok || !b.mesh.HasPin(t.uuid) {
			// This rejection bypasses adapter authorization, so retire idle
			// capacity here too. Active responses keep their own lifetime.
			b.clients.CloseIdle()
			slog.Debug("broadcast skipped: peer not a pinned cluster member", "nodeId", t.id, "uuid", t.uuid)
			return
		}
		hostport := b.addrs.ChooseWithin(ctx, current.id, current.candidates)
		url := "https://" + hostport + eventsPath
		client := b.clients.Client(t.uuid, noderec.ServiceWorkload)
		err := b.post(ctx, client, url, frame)
		if err == nil {
			slog.Debug("broadcast delivered", "url", url, "attempt", attempt)
			return
		}
		var statusErr *httpStatusError
		if !errors.As(err, &statusErr) {
			b.addrs.Forget(t.id)
		}
		slog.Warn("broadcast attempt failed",
			"url", url, "attempt", attempt, "max", maxBroadcastAttempts, "err", err)
		if attempt == maxBroadcastAttempts {
			slog.Warn("broadcast dropped after max attempts", "url", url)
			return
		}
		select {
		case <-time.After(delay):
			delay *= 2
		case <-ctx.Done():
			return
		}
	}
}

func (b *Broadcaster) post(ctx context.Context, client *http.Client, url string, frame []byte) error {
	return b.postWithHeaders(ctx, client, url, frame, nil)
}

func (b *Broadcaster) postWithHeaders(ctx context.Context, client *http.Client, url string, frame []byte, headers http.Header) error {
	reqCtx, cancel := context.WithTimeout(ctx, perPeerTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, bytes.NewReader(frame))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	for name, values := range headers {
		for _, value := range values {
			req.Header.Add(name, value)
		}
	}

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	// Drain the bounded response before releasing the adapter's request slot.
	defer func() {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxDrainBytes))
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusOK {
		return &httpStatusError{code: resp.StatusCode, url: url}
	}
	return nil
}

type httpStatusError struct {
	code int
	url  string
}

func (e *httpStatusError) Error() string {
	return fmt.Sprintf("peer returned HTTP %d", e.code)
}
