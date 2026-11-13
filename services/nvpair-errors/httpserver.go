// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// httpserver.go is the inbound surface for cross-node error sync.
//
// It is cluster mTLS, unconditionally. This is a cluster data plane, not a LAN
// service: the caller must present a certificate this node currently pins, and
// there is no plain-HTTP personality for a node that belongs to no cluster —
// such a node serves nothing here. The trust boundary is cluster membership, not
// the local network.
//
//	GET /v1/errors         -> return THIS node's local-origin snapshot
//	                          (SyncEnvelope), the body of a stream frame.
//	GET /v1/errors/events  -> stream that snapshot to a pinned peer
//	                          (stream.go).
//
// The subscribing side (streams from peers) lives in peersync.go.

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/errors"
)

// newErrorsMux builds the HTTP routes backed by the manager. Split out
// from the listener so tests can exercise the handlers via
// httptest.NewServer without binding a real port or touching mDNS.
func newErrorsMux(mgr *Manager, mesh *clustertrust.Mesh, pub *errorsPublisher) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/errors", func(w http.ResponseWriter, r *http.Request) {
		if !requirePinnedCaller(w, r, mesh) || !requireGet(w, r) {
			return
		}
		handleServeLocal(w, mgr)
	})
	mux.HandleFunc("/v1/errors/events", func(w http.ResponseWriter, r *http.Request) {
		if !requirePinnedCaller(w, r, mesh) || !requireGet(w, r) {
			return
		}
		handleStream(w, r, mesh, pub)
	})
	return mux
}

// pinnedCaller reports whether the caller presented the certificate of a peer
// this node currently pins. The TLS layer accepted any client cert
// (RequireAnyClientCert); this is where a non-member is turned away. Refresh
// first so a cluster joined or left while this process runs is reflected here
// rather than at the next restart.
//
// The check is deliberately unconditional — not "if this node is clustered".
// VerifyClientPin already answers false for a node that belongs to no cluster,
// so an unclustered node rejects everything, which is the intended posture.
// Wrapping it in a membership test is what let an unclustered node serve this
// data in the clear, so there is no branch here for a future caller to widen.
func pinnedCaller(r *http.Request, mesh *clustertrust.Mesh) bool {
	mesh.Refresh()
	_, ok := mesh.VerifyClientPin(r)
	return ok
}

// requirePinnedCaller answers a real 403, rather than an opaque handshake
// failure, to a caller that is not a pinned cluster peer.
func requirePinnedCaller(w http.ResponseWriter, r *http.Request, mesh *clustertrust.Mesh) bool {
	if !pinnedCaller(r, mesh) {
		http.Error(w, "forbidden: not a pinned cluster peer", http.StatusForbidden)
		return false
	}
	return true
}

func requireGet(w http.ResponseWriter, r *http.Request) bool {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", "GET")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return false
	}
	return true
}

// handleServeLocal returns this node's local-origin errors as a
// SyncEnvelope — the exact shape of a stream frame, so the GET endpoint
// doubles as a self-describing probe of what this node would propagate.
func handleServeLocal(w http.ResponseWriter, mgr *Manager) {
	env := errors.SyncEnvelope{
		NodeID: mgr.LocalNodeID(),
		Errors: mgr.LocalSnapshot(),
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(env); err != nil {
		slog.Warn("serve-local: encode failed", "err", err)
	}
}

// runHTTPServer binds the snapshot and stream endpoints on the given port and serves
// until ctx is cancelled, then drains with a short grace period. A bind failure
// is returned so the caller can decide whether it's fatal (it is in main.go: no
// inbound endpoint means peers can't reach us).
//
// The listener is cluster mTLS only — there is no plain personality to fall back
// to. Its certificate is resolved per handshake from the live mesh, which is what
// lets the port stay bound for the life of the process: while this node belongs
// to no cluster it presents no leaf and every handshake is refused, and the
// moment it becomes a member the same listener serves pinned peers. Nothing has
// to be rebound, and no restart is needed to re-read the cluster dir.
func runHTTPServer(ctx context.Context, port int, mgr *Manager, mesh *clustertrust.Mesh, pub *errorsPublisher) error {
	base, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return fmt.Errorf("listen on port %d: %w", port, err)
	}

	// Without an IdleTimeout a peer that has forgotten a connection leaves us
	// pinning its descriptor for the life of the process. The listener's lifetime
	// is deliberately longer than the calling pool's, so the client always reaps
	// first and never picks a connection this side is closing.
	server := &http.Server{
		Handler: newErrorsMux(mgr, mesh, pub),
		// Request contexts end with ctx, so open streams close at shutdown
		// instead of holding Shutdown for its whole grace period.
		BaseContext:       func(net.Listener) context.Context { return ctx },
		ReadHeaderTimeout: 5 * time.Second,
		IdleTimeout:       clustertrust.PeerListenerIdleTimeout,
	}

	errCh := make(chan error, 1)
	go func() {
		slog.Info("errors inter-node server listening (cluster mTLS)",
			"port", port, "clustered", mesh.Clustered())
		if serveErr := server.Serve(tls.NewListener(base, mesh.ServerTLSConfig())); serveErr != nil && serveErr != http.ErrServerClosed {
			errCh <- serveErr
			return
		}
		errCh <- nil
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownCtx)
		return nil
	case serveErr := <-errCh:
		return serveErr
	}
}
