// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// SnapshotStreamServer configures ServeSnapshotStream.
type SnapshotStreamServer[T any] struct {
	Keepalive    time.Duration // interval between keepalive frames; required, > 0
	WriteTimeout time.Duration // per-frame write deadline; required, > 0
	Authorize    func() bool   // called before every frame, keepalives included; false ends the stream
	Updates      <-chan T      // each received value is written as one line; a closed channel ends the stream
}

// ServeSnapshotStream sets Content-Type application/x-ndjson and writes one JSON
// value per line until the request context ends, a write or flush fails,
// Authorize returns false, or Updates is closed. On every Keepalive tick it
// writes the JSON literal null.
//
// The helper owns no change detection: callers feed Updates from their own
// latest-value publisher. A value that encodes to null (a nil slice, map or
// pointer) is indistinguishable from a keepalive on the wire.
func ServeSnapshotStream[T any](w http.ResponseWriter, r *http.Request, s SnapshotStreamServer[T]) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	controller := http.NewResponseController(w)
	if !s.Authorize() {
		return
	}
	if controller.Flush() != nil {
		return
	}
	send := func(value any) bool {
		if !s.Authorize() {
			return false
		}
		// Writers without deadline support (httptest.ResponseRecorder) report
		// http.ErrNotSupported; the frame is still written.
		_ = controller.SetWriteDeadline(time.Now().Add(s.WriteTimeout))
		if json.NewEncoder(w).Encode(value) != nil {
			return false
		}
		return controller.Flush() == nil
	}
	ticker := time.NewTicker(s.Keepalive)
	defer ticker.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case value, open := <-s.Updates:
			if !open || !send(value) {
				return
			}
		case <-ticker.C:
			if !send(nil) {
				return
			}
		}
	}
}

// SnapshotStreamClient configures ReadSnapshotStream.
type SnapshotStreamClient[T any] struct {
	Kind        string                  // bounded service name for process-local counters
	Peer        string                  // admitted peer UUID for process-local counters
	Changed     func() <-chan struct{}  // optional address-change/wake source
	Idle        time.Duration           // cancel when no line (keepalives included) arrives within Idle; required
	Recheck     time.Duration           // also call Current on this interval; 0 = before every line only
	MaxLine     int                     // maximum line length in bytes; required
	Current     func() bool             // false ends the stream
	Decode      func([]byte) (T, error) // decodes one non-null line; an error ends the stream
	OnFrame     func(T) error           // called for every decoded line; an error ends the stream
	OnKeepalive func()                  // optional; called for every null line
}

var streamWallNow = func() time.Time { return time.Now().Round(0) }

// StreamOutcome describes the accepted portion of one snapshot stream.
type StreamOutcome struct {
	Healthy time.Duration // first accepted frame to last received line; zero without an accepted frame
	End     StreamEnd
}

type StreamEnd string

const (
	StreamEndEOF       StreamEnd = "eof"
	StreamEndIdle      StreamEnd = "idle"
	StreamEndAuthority StreamEnd = "authority"
	StreamEndDecode    StreamEnd = "decode"
	StreamEndRefused   StreamEnd = "refused"
	StreamEndCancelled StreamEnd = "cancelled"
	StreamEndError     StreamEnd = "error"
)

// ReadSnapshotStream reads body until EOF, a read, decode or OnFrame error, ctx
// end, the idle timeout or Current returning false, and calls cancel on every
// exit path. It returns nil on a clean EOF, context.Canceled after a
// cancellation, or the error.
//
// cancel must cancel ctx, and body must be bound to ctx (an HTTP response body
// of a request made with ctx), so cancellation unblocks the read. When Recheck
// is set, Current is called from a second goroutine and must be safe for
// concurrent use.
func ReadSnapshotStream[T any](ctx context.Context, cancel context.CancelFunc, body io.Reader, c SnapshotStreamClient[T]) (outcome StreamOutcome, err error) {
	stats := DefaultPeerStats
	stats.StreamOpened(c.Peer, c.Kind)
	defer func() { stats.StreamEnded(c.Peer, c.Kind, string(outcome.End)) }()
	var mu sync.Mutex
	var reason StreamEnd
	stop := func(end StreamEnd) {
		mu.Lock()
		if reason == "" {
			reason = end
		}
		mu.Unlock()
		cancel()
	}
	stopped := func() bool {
		mu.Lock()
		defer mu.Unlock()
		return reason != ""
	}
	now := streamWallNow
	var lastLine atomic.Int64
	if c.Changed != nil {
		lastLine.Store(now().UnixNano())
		changed := c.Changed()
		watchDone := make(chan struct{})
		defer close(watchDone)
		go func() {
			for {
				select {
				case <-ctx.Done():
					return
				case <-watchDone:
					return
				case <-changed:
					changed = c.Changed()
					if now().Sub(time.Unix(0, lastLine.Load())) >= c.Idle {
						stop(StreamEndIdle)
						return
					}
				}
			}
		}()
	}
	var first, last time.Time
	defer func() {
		mu.Lock()
		if reason != "" {
			outcome.End = reason
		} else if outcome.End == "" && ctx.Err() != nil {
			outcome.End = StreamEndCancelled
		}
		mu.Unlock()
		if !first.IsZero() {
			outcome.Healthy = last.Sub(first)
		}
		cancel()
	}()
	if c.Recheck > 0 {
		go func() {
			ticker := time.NewTicker(c.Recheck)
			defer ticker.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
					if !c.Current() {
						stop(StreamEndAuthority)
						return
					}
				}
			}
		}()
	}
	watchdog := time.AfterFunc(c.Idle, func() { stop(StreamEndIdle) })
	defer watchdog.Stop()
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 4096), c.MaxLine)
	for scanner.Scan() {
		stats.StreamBytes(c.Peer, c.Kind, int64(len(scanner.Bytes())))
		if c.Changed != nil {
			lastLine.Store(now().UnixNano())
		}
		if stopped() || ctx.Err() != nil {
			return outcome, context.Canceled
		}
		if !c.Current() {
			stop(StreamEndAuthority)
			return outcome, context.Canceled
		}
		if stopped() || ctx.Err() != nil {
			return outcome, context.Canceled
		}
		last = time.Now()
		watchdog.Reset(c.Idle)
		line := scanner.Bytes()
		if bytes.Equal(bytes.TrimSpace(line), []byte("null")) {
			if c.OnKeepalive != nil {
				c.OnKeepalive()
			}
			continue
		}
		value, err := c.Decode(line)
		if err != nil {
			outcome.End = StreamEndDecode
			return outcome, err
		}
		if stopped() || ctx.Err() != nil {
			return outcome, context.Canceled
		}
		if err := c.OnFrame(value); err != nil {
			outcome.End = StreamEndRefused
			return outcome, err
		}
		if first.IsZero() {
			first = last
		}
	}
	if ctx.Err() != nil {
		return outcome, context.Canceled
	}
	outcome.End = StreamEndEOF
	if scanner.Err() != nil {
		outcome.End = StreamEndError
	}
	return outcome, scanner.Err()
}
