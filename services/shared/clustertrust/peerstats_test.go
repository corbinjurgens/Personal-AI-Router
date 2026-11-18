// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"sync"
	"testing"
	"testing/synctest"
	"time"
)

func TestPeerStatsConcurrent(t *testing.T) {
	s := NewPeerStats()
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			for range 100 {
				s.DialAttempt("peer", "direct")
				s.DialFailed("peer", "direct")
				s.ReverseSetupRound("peer")
				s.ReverseSetupFailed("peer", "tls")
				s.SessionUp("peer")
				s.SessionDown("peer", "ended")
				s.StreamOpened("peer", "models")
				s.StreamBytes("peer", "models", 3)
				s.StreamEnded("peer", "models", "eof")
			}
		})
	}
	wg.Wait()
	deltas, gauges := s.take()
	if len(deltas) != 9 {
		t.Fatalf("counter categories = %d, want 9", len(deltas))
	}
	for _, d := range deltas {
		want := int64(1600)
		if d.Counter == "stream_bytes" {
			want *= 3
		}
		if d.Value != want {
			t.Errorf("%+v = %d, want %d", d.peerStatKey, d.Value, want)
		}
	}
	for _, g := range gauges {
		if g.Live != 0 {
			t.Errorf("gauge %+v, want zero", g)
		}
	}
}

func TestPeerStatsCountConn(t *testing.T) {
	s := NewPeerStats()
	a, b := net.Pipe()
	defer a.Close()
	defer b.Close()
	c := s.CountConn(a, "peer", "direct")
	if err := c.SetDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		_, err := b.Write([]byte("read"))
		if err == nil {
			_, err = io.CopyN(io.Discard, b, 5)
		}
		done <- err
	}()
	if _, err := io.CopyN(io.Discard, c, 4); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Write([]byte("write")); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	deltas, _ := s.take()
	assertPeerStat(t, deltas, peerStatKey{Peer: "peer", Counter: "bytes_read", View: "direct"}, 4)
	assertPeerStat(t, deltas, peerStatKey{Peer: "peer", Counter: "bytes_written", View: "direct"}, 5)
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Write(nil); err != io.ErrClosedPipe {
		t.Fatalf("underlying connection after Close: %v", err)
	}
}

func TestPeerStatsPreservesTLSState(t *testing.T) {
	a, b := net.Pipe()
	defer a.Close()
	defer b.Close()
	secure := tls.Client(a, &tls.Config{ServerName: "peer"})
	c := NewPeerStats().CountConn(secure, "peer", "direct")
	state, ok := c.(interface{ ConnectionState() tls.ConnectionState })
	if !ok {
		t.Fatal("TLS state method lost")
	}
	if got, want := state.ConnectionState(), secure.ConnectionState(); got.HandshakeComplete != want.HandshakeComplete || got.ServerName != want.ServerName {
		t.Fatalf("TLS state = %+v, want %+v", got, want)
	}
	if _, ok := NewPeerStats().CountConn(b, "peer", "direct").(interface{ ConnectionState() tls.ConnectionState }); ok {
		t.Fatal("plain connection acquired a TLS state method")
	}
}

type peerStatsLogWriter struct{ lines [][]byte }

func (w *peerStatsLogWriter) Write(p []byte) (int, error) {
	w.lines = append(w.lines, append([]byte(nil), p...))
	return len(p), nil
}

type peerStatsLogLine struct {
	Message string          `json:"msg"`
	Level   string          `json:"level"`
	Deltas  []peerStatDelta `json:"deltas"`
	Gauges  []peerStatGauge `json:"gauges"`
}

func decodePeerStatsLog(t *testing.T, p []byte) peerStatsLogLine {
	t.Helper()
	var line peerStatsLogLine
	if err := json.Unmarshal(p, &line); err != nil {
		t.Fatal(err)
	}
	if line.Message != "peer connection stats" || line.Level != "INFO" {
		t.Fatalf("log = %+v", line)
	}
	return line
}

func assertPeerStat(t *testing.T, deltas []peerStatDelta, key peerStatKey, want int64) {
	t.Helper()
	var got int64
	for _, d := range deltas {
		if d.peerStatKey == key {
			got += d.Value
		}
	}
	if got != want {
		t.Errorf("%+v = %d, want %d (deltas %+v)", key, got, want, deltas)
	}
}

func isolatedPeerStats(t *testing.T) *PeerStats {
	t.Helper()
	old := DefaultPeerStats
	s := NewPeerStats()
	DefaultPeerStats = s
	t.Cleanup(func() { DefaultPeerStats = old })
	return s
}

func TestPeerStatsLogDeltasAndIdle(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s := NewPeerStats()
		var w peerStatsLogWriter
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		go s.Log(ctx, slog.New(slog.NewJSONHandler(&w, nil)), 50*time.Millisecond)
		synctest.Wait()
		s.StreamOpened("peer", "models")
		s.StreamBytes("peer", "models", 7)
		s.DialAttempt("another-peer", "reverse")
		time.Sleep(50 * time.Millisecond)
		synctest.Wait()
		if len(w.lines) != 1 {
			t.Fatalf("log lines = %d, want 1", len(w.lines))
		}
		first := decodePeerStatsLog(t, w.lines[0])
		assertPeerStat(t, first.Deltas, peerStatKey{Peer: "another-peer", Counter: "dial_attempts", Route: "reverse"}, 1)
		assertPeerStat(t, first.Deltas, peerStatKey{Peer: "peer", Counter: "stream_bytes", Kind: "models"}, 7)
		if len(first.Gauges) != 1 || first.Gauges[0].Live != 1 {
			t.Fatalf("gauges = %+v", first.Gauges)
		}
		s.StreamBytes("peer", "models", 11)
		time.Sleep(50 * time.Millisecond)
		synctest.Wait()
		if len(w.lines) != 2 {
			t.Fatalf("log lines = %d, want 2", len(w.lines))
		}
		assertPeerStat(t, decodePeerStatsLog(t, w.lines[1]).Deltas, peerStatKey{Peer: "peer", Counter: "stream_bytes", Kind: "models"}, 11)
		s.StreamEnded("peer", "models", "eof")
		time.Sleep(50 * time.Millisecond)
		synctest.Wait()
		if len(w.lines) != 3 {
			t.Fatalf("log lines = %d, want 3", len(w.lines))
		}
		last := decodePeerStatsLog(t, w.lines[2])
		assertPeerStat(t, last.Deltas, peerStatKey{Peer: "peer", Counter: "stream_bytes", Kind: "models"}, 0)
		if len(last.Gauges) != 1 || last.Gauges[0].Live != 0 {
			t.Fatalf("gauges = %+v", last.Gauges)
		}
		time.Sleep(100 * time.Millisecond)
		synctest.Wait()
		if len(w.lines) != 3 {
			t.Fatalf("idle log lines = %d, want 3", len(w.lines))
		}
	})
}

func TestPeerStatsEvictsDepartedPeersAfterFinalZero(t *testing.T) {
	s := NewPeerStats()
	s.SessionUp("staying")
	s.StreamOpened("departing", "node-info")
	_, gauges := s.take()
	if len(gauges) != 2 {
		t.Fatalf("initial gauges = %+v, want both live peers", gauges)
	}

	s.StreamEnded("departing", "node-info", "eof")
	deltas, gauges := s.take()
	assertPeerStat(t, deltas, peerStatKey{Peer: "departing", Counter: "streams_ended", Kind: "node-info", Reason: "eof"}, 1)
	zero, found := false, false
	for _, g := range gauges {
		if g.Peer == "departing" {
			found, zero = true, g.Live == 0
		}
	}
	if !found || !zero {
		t.Errorf("departure gauges = %+v, want departing peer's final zero", gauges)
	}

	s.StreamOpened("new", "node-info")
	deltas, gauges = s.take()
	assertPeerStat(t, deltas, peerStatKey{Peer: "new", Counter: "streams_opened", Kind: "node-info"}, 1)
	if len(s.deltas) != 0 {
		t.Errorf("retained %d consumed delta keys", len(s.deltas))
	}
	if len(s.gauges) != 2 || len(gauges) != 2 {
		t.Errorf("current/emitted gauges = %+v/%+v, want only staying and new peers", s.gauges, gauges)
	}
	for _, g := range gauges {
		if g.Peer == "departing" || g.Live != 1 {
			t.Errorf("later interval emitted stale gauge %+v", g)
		}
	}
}

func TestPeerStatsBoundsLabels(t *testing.T) {
	s := NewPeerStats()
	s.DialAttempt("peer", "http://invalid")
	s.DialFailed("peer", "arbitrary method")
	s.ReverseSetupFailed("peer", "address:123")
	s.SessionDown("peer", "arbitrary error")
	s.StreamOpened("", "http://invalid")
	s.StreamEnded("", "http://invalid", "arbitrary error")
	deltas, _ := s.take()
	want := map[peerStatKey]int64{
		{Peer: "peer", Counter: "reverse_setup_failures", Stage: "setup"}:             1,
		{Peer: "peer", Counter: "sessions_down", Reason: "other"}:                     1,
		{Peer: "unknown", Counter: "streams_opened", Kind: "unknown"}:                 1,
		{Peer: "unknown", Counter: "streams_ended", Kind: "unknown", Reason: "error"}: 1,
	}
	if len(deltas) != len(want) {
		t.Fatalf("deltas = %+v", deltas)
	}
	for _, d := range deltas {
		if want[d.peerStatKey] != d.Value {
			t.Errorf("unexpected delta %+v", d)
		}
	}
}

func benchmarkPeerConn(b *testing.B, counted bool) {
	a, remote := net.Pipe()
	defer a.Close()
	defer remote.Close()
	var conn net.Conn = a
	if counted {
		conn = NewPeerStats().CountConn(conn, "peer", "direct")
	}
	done := make(chan error, 1)
	go func() {
		var data [1]byte
		for range b.N {
			if _, err := io.ReadFull(remote, data[:]); err != nil {
				done <- err
				return
			}
			if _, err := remote.Write(data[:]); err != nil {
				done <- err
				return
			}
		}
		done <- nil
	}()
	var data [1]byte
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := conn.Write(data[:]); err != nil {
			b.Fatal(err)
		}
		if _, err := io.ReadFull(conn, data[:]); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	if err := <-done; err != nil {
		b.Fatal(err)
	}
}

func BenchmarkConnRaw(b *testing.B)   { benchmarkPeerConn(b, false) }
func BenchmarkCountConn(b *testing.B) { benchmarkPeerConn(b, true) }
