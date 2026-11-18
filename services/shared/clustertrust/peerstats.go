// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"log/slog"
	"net"
	"sync"
	"time"
)

// PeerStats holds process-local counters. Byte views describe different protocol
// layers and must be read separately, never added together.
type PeerStats struct {
	mu     sync.Mutex
	deltas map[peerStatKey]int64
	gauges map[peerGaugeKey]int64
}

type peerStatKey struct {
	Peer    string `json:"peer"`
	Counter string `json:"counter"`
	Route   string `json:"route,omitempty"`
	View    string `json:"view,omitempty"`
	Kind    string `json:"kind,omitempty"`
	Reason  string `json:"reason,omitempty"`
	Stage   string `json:"stage,omitempty"`
}

type peerGaugeKey struct {
	Peer  string `json:"peer"`
	Gauge string `json:"gauge"`
	Kind  string `json:"kind,omitempty"`
}

type peerStatDelta struct {
	peerStatKey
	Value int64 `json:"value"`
}

type peerStatGauge struct {
	peerGaugeKey
	Live int64 `json:"live"`
}

// String keeps the process text logger's nested values labelled, as they are
// with slog's JSON handler. These fixed records cannot fail JSON encoding.
type peerStatDeltas []peerStatDelta

func (d peerStatDeltas) String() string { b, _ := json.Marshal(d); return string(b) }

type peerStatGauges []peerStatGauge

func (g peerStatGauges) String() string { b, _ := json.Marshal(g); return string(b) }

func NewPeerStats() *PeerStats {
	return &PeerStats{deltas: make(map[peerStatKey]int64), gauges: make(map[peerGaugeKey]int64)}
}

var DefaultPeerStats = NewPeerStats()

func (s *PeerStats) add(key peerStatKey, n int64) {
	if key.Peer == "" {
		key.Peer = "unknown"
	}
	s.mu.Lock()
	s.deltas[key] += n
	s.mu.Unlock()
}

func (s *PeerStats) DialAttempt(peer, route string) {
	if route == "direct" || route == "reverse" {
		s.add(peerStatKey{Peer: peer, Counter: "dial_attempts", Route: route}, 1)
	}
}

func (s *PeerStats) DialFailed(peer, route string) {
	if route == "direct" || route == "reverse" {
		s.add(peerStatKey{Peer: peer, Counter: "dial_failures", Route: route}, 1)
	}
}

func (s *PeerStats) ReverseSetupRound(peer string) {
	s.add(peerStatKey{Peer: peer, Counter: "reverse_setup_rounds"}, 1)
}

func (s *PeerStats) ReverseSetupFailed(peer, stage string) {
	switch stage {
	case "dial", "tls", "upgrade", "handoff", "setup", "register":
	default:
		stage = "setup"
	}
	s.add(peerStatKey{Peer: peer, Counter: "reverse_setup_failures", Stage: stage}, 1)
}

func (s *PeerStats) event(key peerStatKey, change int64) {
	if key.Peer == "" {
		key.Peer = "unknown"
	}
	s.mu.Lock()
	s.deltas[key]++
	gauge := "sessions"
	if key.Kind != "" {
		gauge = "streams"
	}
	s.gauges[peerGaugeKey{Peer: key.Peer, Gauge: gauge, Kind: key.Kind}] += change
	s.mu.Unlock()
}

func (s *PeerStats) SessionUp(peer string) {
	s.event(peerStatKey{Peer: peer, Counter: "sessions_up"}, 1)
}

func (s *PeerStats) SessionDown(peer, reason string) {
	switch reason {
	case "ended", "replaced", "not-admitted", "revoked", "manager-closed", "control-failed", "protocol", "probe-failed", "other":
	default:
		reason = "other"
	}
	s.event(peerStatKey{Peer: peer, Counter: "sessions_down", Reason: reason}, -1)
}

func streamKind(kind string) string {
	switch kind {
	case "peer-state", "node-info", "models", "settings", "engines", "errors", "performance":
		return kind
	}
	return "unknown"
}

func (s *PeerStats) StreamOpened(peer, kind string) {
	s.event(peerStatKey{Peer: peer, Counter: "streams_opened", Kind: streamKind(kind)}, 1)
}

func (s *PeerStats) StreamBytes(peer, kind string, n int64) {
	if n > 0 {
		s.add(peerStatKey{Peer: peer, Counter: "stream_bytes", Kind: streamKind(kind)}, n)
	}
}

func (s *PeerStats) StreamEnded(peer, kind, reason string) {
	switch StreamEnd(reason) {
	case StreamEndEOF, StreamEndIdle, StreamEndAuthority, StreamEndDecode, StreamEndRefused, StreamEndCancelled, StreamEndError:
	default:
		reason = string(StreamEndError)
	}
	s.event(peerStatKey{Peer: peer, Counter: "streams_ended", Kind: streamKind(kind), Reason: reason}, -1)
}

type peerCountConn struct {
	net.Conn
	stats             *PeerStats
	readKey, writeKey peerStatKey
}

func (c *peerCountConn) Read(p []byte) (int, error) {
	n, err := c.Conn.Read(p)
	if n > 0 {
		c.stats.add(c.readKey, int64(n))
	}
	return n, err
}

func (c *peerCountConn) Write(p []byte) (int, error) {
	n, err := c.Conn.Write(p)
	if n > 0 {
		c.stats.add(c.writeKey, int64(n))
	}
	return n, err
}

type peerCountTLSConn struct {
	*peerCountConn
	state interface{ ConnectionState() tls.ConnectionState }
}

func (c *peerCountTLSConn) ConnectionState() tls.ConnectionState { return c.state.ConnectionState() }

func (s *PeerStats) CountConn(conn net.Conn, peer, view string) net.Conn {
	if view != "direct" && view != "reverse-tunnel" {
		return conn
	}
	c := &peerCountConn{Conn: conn, stats: s,
		readKey:  peerStatKey{Peer: peer, Counter: "bytes_read", View: view},
		writeKey: peerStatKey{Peer: peer, Counter: "bytes_written", View: view}}
	if state, ok := conn.(interface{ ConnectionState() tls.ConnectionState }); ok {
		return &peerCountTLSConn{peerCountConn: c, state: state}
	}
	return c
}

func (s *PeerStats) take() (peerStatDeltas, peerStatGauges) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var deltas peerStatDeltas
	for key, n := range s.deltas {
		if n != 0 {
			deltas = append(deltas, peerStatDelta{key, n})
		}
	}
	clear(s.deltas)
	if len(deltas) == 0 {
		return nil, nil
	}
	gauges := make(peerStatGauges, 0, len(s.gauges))
	for key, n := range s.gauges {
		gauges = append(gauges, peerStatGauge{key, n})
		if n == 0 {
			// Emit the final zero once, in the interval with the last close.
			delete(s.gauges, key)
		}
	}
	return deltas, gauges
}

// Log emits interval deltas and current live session/stream gauges at Info.
func (s *PeerStats) Log(ctx context.Context, logger *slog.Logger, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if deltas, gauges := s.take(); len(deltas) != 0 {
				logger.InfoContext(ctx, "peer connection stats", "deltas", deltas, "gauges", gauges)
			}
		}
	}
}
