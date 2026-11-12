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
// latest-value publisher.
func ServeSnapshotStream[T any](w http.ResponseWriter, r *http.Request, s SnapshotStreamServer[T]) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	controller := http.NewResponseController(w)
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
	Idle        time.Duration           // cancel when no line (keepalives included) arrives within Idle; required
	Recheck     time.Duration           // also call Current on this interval; 0 = before every line only
	MaxLine     int                     // maximum line length in bytes; required
	Current     func() bool             // false ends the stream
	Decode      func([]byte) (T, error) // decodes one non-null line; an error ends the stream
	OnFrame     func(T) error           // called for every decoded line; an error ends the stream
	OnKeepalive func()                  // optional; called for every null line
}

// ReadSnapshotStream reads body until EOF, a read, decode or OnFrame error, ctx
// end, the idle timeout or Current returning false, and calls cancel on every
// exit path. It returns nil on a clean EOF, context.Canceled after a
// cancellation, or the error.
//
// cancel must cancel ctx, and body must be bound to ctx (an HTTP response body
// of a request made with ctx), so cancellation unblocks the read.
func ReadSnapshotStream[T any](ctx context.Context, cancel context.CancelFunc, body io.Reader, c SnapshotStreamClient[T]) error {
	defer cancel()
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
						cancel()
						return
					}
				}
			}
		}()
	}
	watchdog := time.AfterFunc(c.Idle, cancel)
	defer watchdog.Stop()
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 4096), c.MaxLine)
	for scanner.Scan() {
		if !c.Current() {
			return context.Canceled
		}
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
			return err
		}
		if err := c.OnFrame(value); err != nil {
			return err
		}
	}
	if ctx.Err() != nil {
		return context.Canceled
	}
	return scanner.Err()
}
