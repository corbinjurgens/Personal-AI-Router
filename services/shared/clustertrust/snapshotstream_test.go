// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type streamFrame struct {
	N int `json:"n"`
}

// openStream starts handler on a loopback test server and opens one GET
// request with a context the test cancels at cleanup.
func openStream(t *testing.T, handler http.HandlerFunc) (context.Context, context.CancelFunc, *http.Response) {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, server.URL, nil)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	response, err := server.Client().Do(req)
	if err != nil {
		t.Fatalf("open stream: %v", err)
	}
	t.Cleanup(func() { _ = response.Body.Close() })
	if response.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want %d", response.StatusCode, http.StatusOK)
	}
	return ctx, cancel, response
}

// readLines delivers each body line on the returned channel and closes it at
// the end of the body.
func readLines(body io.Reader) <-chan string {
	lines := make(chan string, 64)
	go func() {
		defer close(lines)
		scanner := bufio.NewScanner(body)
		for scanner.Scan() {
			lines <- scanner.Text()
		}
	}()
	return lines
}

func nextLine(t *testing.T, lines <-chan string, within time.Duration) (string, bool) {
	t.Helper()
	select {
	case line, open := <-lines:
		return line, open
	case <-time.After(within):
		t.Fatalf("no line within %v", within)
		return "", false
	}
}

func waitEnd(t *testing.T, lines <-chan string, within time.Duration) {
	t.Helper()
	deadline := time.After(within)
	for {
		select {
		case _, open := <-lines:
			if !open {
				return
			}
		case <-deadline:
			t.Fatalf("stream still open after %v", within)
		}
	}
}

func TestServeSnapshotStream(t *testing.T) {
	serve := func(keepalive time.Duration, authorize func() bool, updates <-chan streamFrame, done chan<- struct{}) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			defer close(done)
			ServeSnapshotStream(w, r, SnapshotStreamServer[streamFrame]{
				Keepalive:    keepalive,
				WriteTimeout: time.Second,
				Authorize:    authorize,
				Updates:      updates,
			})
		}
	}
	allow := func() bool { return true }

	t.Run("writes an update at once", func(t *testing.T) {
		updates := make(chan streamFrame, 1)
		updates <- streamFrame{N: 7}
		done := make(chan struct{})
		_, _, response := openStream(t, serve(time.Hour, allow, updates, done))
		if got := response.Header.Get("Content-Type"); got != "application/x-ndjson" {
			t.Fatalf("Content-Type = %q, want application/x-ndjson", got)
		}
		line, open := nextLine(t, readLines(response.Body), 500*time.Millisecond)
		if !open {
			t.Fatal("stream ended before the update")
		}
		var got streamFrame
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			t.Fatalf("decode %q: %v", line, err)
		}
		if got.N != 7 {
			t.Fatalf("frame = %+v, want N=7", got)
		}
	})

	t.Run("writes null keepalives on the interval", func(t *testing.T) {
		done := make(chan struct{})
		_, cancel, response := openStream(t, serve(20*time.Millisecond, allow, make(chan streamFrame), done))
		lines := readLines(response.Body)
		count := 0
		deadline := time.After(200 * time.Millisecond)
	collect:
		for {
			select {
			case line, open := <-lines:
				if !open {
					t.Fatal("stream ended early")
				}
				if line != "null" {
					t.Fatalf("keepalive = %q, want null", line)
				}
				count++
			case <-deadline:
				break collect
			}
		}
		cancel()
		if count < 8 || count > 12 {
			t.Fatalf("keepalives in 200ms at 20ms = %d, want 10±2", count)
		}
	})

	t.Run("ends when Authorize refuses", func(t *testing.T) {
		var calls atomic.Int32
		authorize := func() bool { return calls.Add(1) < 3 }
		done := make(chan struct{})
		_, _, response := openStream(t, serve(20*time.Millisecond, authorize, make(chan streamFrame), done))
		lines := readLines(response.Body)
		waitEnd(t, lines, time.Second)
		<-done
		if got := calls.Load(); got != 3 {
			t.Fatalf("Authorize calls = %d, want 3 (two frames, then refusal)", got)
		}
	})

	t.Run("checks Authorize before an update", func(t *testing.T) {
		updates := make(chan streamFrame, 1)
		updates <- streamFrame{N: 1}
		done := make(chan struct{})
		_, _, response := openStream(t, serve(time.Hour, func() bool { return false }, updates, done))
		lines := readLines(response.Body)
		if line, open := nextLine(t, lines, time.Second); open {
			t.Fatalf("refused stream wrote %q", line)
		}
	})

	t.Run("ends when Updates is closed", func(t *testing.T) {
		updates := make(chan streamFrame)
		close(updates)
		done := make(chan struct{})
		_, _, response := openStream(t, serve(time.Hour, allow, updates, done))
		waitEnd(t, readLines(response.Body), time.Second)
	})

	t.Run("returns on a cancelled request without deadline support", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		req := httptest.NewRequestWithContext(ctx, http.MethodGet, "/events", nil)
		recorder := httptest.NewRecorder()
		updates := make(chan streamFrame, 1)
		updates <- streamFrame{N: 3}
		returned := make(chan struct{})
		go func() {
			defer close(returned)
			ServeSnapshotStream(recorder, req, SnapshotStreamServer[streamFrame]{
				Keepalive:    20 * time.Millisecond,
				WriteTimeout: time.Second,
				Authorize:    allow,
				Updates:      updates,
			})
		}()
		time.Sleep(60 * time.Millisecond)
		cancel()
		select {
		case <-returned:
		case <-time.After(time.Second):
			t.Fatal("handler did not return after cancellation")
		}
		first, _, found := strings.Cut(recorder.Body.String(), "\n")
		if !found {
			t.Fatalf("recorder body %q has no complete line", recorder.Body.String())
		}
		var got streamFrame
		if err := json.Unmarshal([]byte(first), &got); err != nil {
			t.Fatalf("decode %q: %v", first, err)
		}
		if got.N != 3 {
			t.Fatalf("first frame = %+v, want N=3", got)
		}
	})
}

func TestReadSnapshotStream(t *testing.T) {
	decode := func(b []byte) (streamFrame, error) {
		var v streamFrame
		return v, json.Unmarshal(b, &v)
	}
	// writeBody writes each line, flushing after each, then holds the stream
	// open until the request ends.
	writeBody := func(lines ...string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			// The first flush sends the headers even when lines is empty.
			_ = http.NewResponseController(w).Flush()
			for _, line := range lines {
				_, _ = fmt.Fprintln(w, line)
				_ = http.NewResponseController(w).Flush()
			}
			<-r.Context().Done()
		}
	}
	allow := func() bool { return true }
	read := func(t *testing.T, handler http.HandlerFunc, c SnapshotStreamClient[streamFrame]) error {
		t.Helper()
		ctx, cancel, response := openStream(t, handler)
		result := make(chan error, 1)
		go func() { result <- ReadSnapshotStream(ctx, cancel, response.Body, c) }()
		select {
		case err := <-result:
			if ctx.Err() == nil {
				t.Fatal("ReadSnapshotStream returned without calling cancel")
			}
			return err
		case <-time.After(2 * time.Second):
			t.Fatal("ReadSnapshotStream did not return")
			return nil
		}
	}

	t.Run("delivers decoded frames in order", func(t *testing.T) {
		handler := func(w http.ResponseWriter, r *http.Request) {
			_, _ = fmt.Fprint(w, "{\"n\":1}\n{\"n\":2}\n{\"n\":3}\n")
		}
		var got []int
		err := read(t, handler, SnapshotStreamClient[streamFrame]{
			Idle: time.Second, MaxLine: 1024, Current: allow, Decode: decode,
			OnFrame: func(f streamFrame) error { got = append(got, f.N); return nil },
		})
		if err != nil {
			t.Fatalf("err = %v, want nil at EOF", err)
		}
		if fmt.Sprint(got) != "[1 2 3]" {
			t.Fatalf("frames = %v, want [1 2 3]", got)
		}
	})

	t.Run("ends on an OnFrame error", func(t *testing.T) {
		stop := errors.New("rejected frame")
		frames := 0
		err := read(t, writeBody(`{"n":1}`, `{"n":2}`), SnapshotStreamClient[streamFrame]{
			Idle: time.Second, MaxLine: 1024, Current: allow, Decode: decode,
			OnFrame: func(streamFrame) error { frames++; return stop },
		})
		if !errors.Is(err, stop) {
			t.Fatalf("err = %v, want the OnFrame error", err)
		}
		if frames != 1 {
			t.Fatalf("OnFrame calls = %d, want 1", frames)
		}
	})

	t.Run("returns nil on a clean EOF", func(t *testing.T) {
		handler := func(w http.ResponseWriter, r *http.Request) { _, _ = fmt.Fprintln(w, `{"n":1}`) }
		err := read(t, handler, SnapshotStreamClient[streamFrame]{
			Idle: time.Second, MaxLine: 1024, Current: allow, Decode: decode,
			OnFrame: func(streamFrame) error { return nil },
		})
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
	})

	t.Run("calls OnKeepalive for null lines without decoding", func(t *testing.T) {
		var keepalives atomic.Int32
		stop := errors.New("frame after keepalives")
		err := read(t, writeBody("null", " null ", `{"n":9}`), SnapshotStreamClient[streamFrame]{
			Idle: time.Second, MaxLine: 1024, Current: allow,
			Decode: func(b []byte) (streamFrame, error) {
				if strings.TrimSpace(string(b)) == "null" {
					t.Errorf("Decode received keepalive %q", b)
				}
				return decode(b)
			},
			OnFrame:     func(streamFrame) error { return stop },
			OnKeepalive: func() { keepalives.Add(1) },
		})
		if !errors.Is(err, stop) {
			t.Fatalf("err = %v, want the OnFrame error", err)
		}
		if got := keepalives.Load(); got != 2 {
			t.Fatalf("keepalives = %d, want 2", got)
		}
	})

	t.Run("cancels when no line arrives within Idle", func(t *testing.T) {
		start := time.Now()
		err := read(t, writeBody("null"), SnapshotStreamClient[streamFrame]{
			Idle: 100 * time.Millisecond, MaxLine: 1024, Current: allow, Decode: decode,
			OnFrame: func(streamFrame) error { return nil },
		})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
		if elapsed := time.Since(start); elapsed < 80*time.Millisecond {
			t.Fatalf("idle cancel after %v, want about 100ms", elapsed)
		}
	})

	t.Run("keepalives reset the idle watchdog", func(t *testing.T) {
		handler := func(w http.ResponseWriter, r *http.Request) {
			ticker := time.NewTicker(20 * time.Millisecond)
			defer ticker.Stop()
			for {
				select {
				case <-r.Context().Done():
					return
				case <-ticker.C:
					_, _ = fmt.Fprintln(w, "null")
					_ = http.NewResponseController(w).Flush()
				}
			}
		}
		// Idle is 100ms; the stream ends only when Current turns false at
		// 300ms, so it outlives Idle three times over on keepalives alone.
		var live atomic.Bool
		live.Store(true)
		time.AfterFunc(300*time.Millisecond, func() { live.Store(false) })
		var keepalives atomic.Int32
		start := time.Now()
		err := read(t, handler, SnapshotStreamClient[streamFrame]{
			Idle: 100 * time.Millisecond, Recheck: 20 * time.Millisecond, MaxLine: 1024,
			Current: live.Load, Decode: decode,
			OnFrame:     func(streamFrame) error { return nil },
			OnKeepalive: func() { keepalives.Add(1) },
		})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
		if elapsed := time.Since(start); elapsed < 250*time.Millisecond {
			t.Fatalf("stream ended after %v, before Current turned false", elapsed)
		}
		if got := keepalives.Load(); got < 8 {
			t.Fatalf("keepalives = %d, want at least 8", got)
		}
	})

	t.Run("cancels when Current turns false on recheck", func(t *testing.T) {
		var live atomic.Bool
		live.Store(true)
		time.AfterFunc(60*time.Millisecond, func() { live.Store(false) })
		// No line ever arrives and Idle is far away, so only the recheck
		// goroutine can end the stream quickly.
		start := time.Now()
		err := read(t, writeBody(), SnapshotStreamClient[streamFrame]{
			Idle: time.Hour, Recheck: 20 * time.Millisecond, MaxLine: 1024,
			Current: live.Load, Decode: decode,
			OnFrame: func(streamFrame) error { return nil },
		})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
		if elapsed := time.Since(start); elapsed > 500*time.Millisecond {
			t.Fatalf("recheck cancel after %v, want well under 500ms", elapsed)
		}
	})

	t.Run("checks Current before every line", func(t *testing.T) {
		frames := 0
		err := read(t, writeBody(`{"n":1}`), SnapshotStreamClient[streamFrame]{
			Idle: time.Second, MaxLine: 1024, Current: func() bool { return false }, Decode: decode,
			OnFrame: func(streamFrame) error { frames++; return nil },
		})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
		if frames != 0 {
			t.Fatalf("frames delivered after Current refused = %d, want 0", frames)
		}
	})

	t.Run("ends on a line over MaxLine", func(t *testing.T) {
		err := read(t, writeBody(`{"n":1,"pad":"`+strings.Repeat("x", 8192)+`"}`), SnapshotStreamClient[streamFrame]{
			Idle: time.Second, MaxLine: 4096, Current: allow, Decode: decode,
			OnFrame: func(streamFrame) error { return nil },
		})
		if !errors.Is(err, bufio.ErrTooLong) {
			t.Fatalf("err = %v, want bufio.ErrTooLong", err)
		}
	})

	t.Run("ends on a Decode error", func(t *testing.T) {
		bad := errors.New("bad frame")
		err := read(t, writeBody(`{"n":1}`), SnapshotStreamClient[streamFrame]{
			Idle: time.Second, MaxLine: 1024, Current: allow,
			Decode:  func([]byte) (streamFrame, error) { return streamFrame{}, bad },
			OnFrame: func(streamFrame) error { return nil },
		})
		if !errors.Is(err, bad) {
			t.Fatalf("err = %v, want the Decode error", err)
		}
	})
}
