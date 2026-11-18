// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"
)

type streamFrame struct {
	N int `json:"n"`
}

func TestSnapshotStreamAcceptanceReceipt(t *testing.T) {
	const body = "{\"fixture\":\"unlogged payload\"}"
	const peer = "00000000-0000-4000-8000-000000000081"
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(previous) })

	start := time.Now().UTC()
	for _, kind := range []string{"peer-state", "node-info", "engines", "models"} {
		ctx, cancel := context.WithCancel(context.Background())
		_, err := ReadSnapshotStream(ctx, cancel, strings.NewReader("null\n"+body+"\n"+body+"\n"), SnapshotStreamClient[json.RawMessage]{
			Kind: kind, Peer: peer, Idle: time.Second, MaxLine: 4096,
			Current: func() bool { return true },
			Decode:  func(b []byte) (json.RawMessage, error) { return append(json.RawMessage(nil), b...), nil },
			OnFrame: func(json.RawMessage) error { return nil },
		})
		if err != nil {
			t.Fatalf("read %s: %v", kind, err)
		}
	}
	end := time.Now().UTC()
	seen := make(map[string]int)
	for line := range strings.Lines(logs.String()) {
		if !strings.Contains(line, "peer snapshot received") {
			continue
		}
		if strings.Contains(line, "unlogged payload") {
			t.Fatal("receipt logged the frame payload")
		}
		_, rest, ok := strings.Cut(line, "kind=")
		kind, at, split := strings.Cut(rest, " peer="+peer+" received_at=")
		stamp, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(at))
		// TextHandler formats time attributes to milliseconds.
		if !ok || !split || err != nil || stamp.Before(start.Truncate(time.Millisecond)) || stamp.After(end) || stamp.Location() != time.UTC {
			t.Fatalf("invalid UTC receipt line: %s", line)
		}
		seen[kind]++
		t.Log(strings.TrimSpace(line))
	}
	for _, kind := range []string{"peer-state", "node-info", "engines"} {
		if seen[kind] != 1 {
			t.Errorf("%s receipts = %d, want 1 for two consecutive frames and a keepalive", kind, seen[kind])
		}
	}
	if seen["models"] != 0 {
		t.Error("models stream emitted an acceptance receipt")
	}
}

func TestSnapshotStreamPeerStatsAcrossIntervals(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		stats := isolatedPeerStats(t)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		logCtx, logCancel := context.WithCancel(context.Background())
		defer logCancel()
		var logs peerStatsLogWriter
		go stats.Log(logCtx, slog.New(slog.NewJSONHandler(&logs, nil)), 50*time.Millisecond)
		reader, writer := io.Pipe()
		defer reader.Close()
		defer writer.Close()
		done := make(chan StreamOutcome, 1)
		go func() {
			out, err := ReadSnapshotStream(ctx, cancel, reader, SnapshotStreamClient[streamFrame]{
				Kind: "models", Peer: "peer", Idle: time.Second, MaxLine: 4096,
				Current: func() bool { return true },
				Decode:  func(b []byte) (streamFrame, error) { var f streamFrame; err := json.Unmarshal(b, &f); return f, err },
				OnFrame: func(streamFrame) error { return nil },
			})
			if err != nil {
				t.Errorf("read stream: %v", err)
			}
			done <- out
		}()
		synctest.Wait()
		for i, line := range []string{"null\n", "{\"n\":1}\n"} {
			if _, err := io.WriteString(writer, line); err != nil {
				t.Fatal(err)
			}
			synctest.Wait()
			time.Sleep(50 * time.Millisecond)
			synctest.Wait()
			if len(logs.lines) != i+1 {
				t.Fatalf("interval %d log lines = %d", i, len(logs.lines))
			}
			assertPeerStat(t, decodePeerStatsLog(t, logs.lines[i]).Deltas, peerStatKey{Peer: "peer", Counter: "stream_bytes", Kind: "models"}, int64(len(line)-1))
		}
		if err := writer.Close(); err != nil {
			t.Fatal(err)
		}
		out := <-done
		if out.End != StreamEndEOF {
			t.Fatalf("end = %s", out.End)
		}
		time.Sleep(50 * time.Millisecond)
		synctest.Wait()
		if len(logs.lines) != 3 {
			t.Fatalf("final log lines = %d", len(logs.lines))
		}
		last := decodePeerStatsLog(t, logs.lines[2])
		assertPeerStat(t, last.Deltas, peerStatKey{Peer: "peer", Counter: "stream_bytes", Kind: "models"}, 0)
		assertPeerStat(t, last.Deltas, peerStatKey{Peer: "peer", Counter: "streams_ended", Kind: "models", Reason: string(out.End)}, 1)
		if len(last.Gauges) != 1 || last.Gauges[0].Live != 0 {
			t.Fatalf("gauges = %+v", last.Gauges)
		}
	})
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
			t.Fatalf("Authorize calls = %d, want 3 (headers, one frame, then refusal)", got)
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

func TestServeSnapshotStreamSendsHeadersBeforeFirstFrame(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ServeSnapshotStream(w, r, SnapshotStreamServer[streamFrame]{
			Keepalive:    10 * time.Second,
			WriteTimeout: time.Second,
			Authorize:    func() bool { return true },
			Updates:      make(chan streamFrame),
		})
	}))
	defer server.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, server.URL, nil)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	client := server.Client()
	client.Transport = &http.Transport{ResponseHeaderTimeout: time.Second}
	defer client.CloseIdleConnections()
	response, err := client.Do(req)
	if err != nil {
		t.Fatalf("headers did not arrive within 1 s: %v", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK || response.Header.Get("Content-Type") != "application/x-ndjson" {
		t.Fatalf("status/content type = %d/%q", response.StatusCode, response.Header.Get("Content-Type"))
	}
	select {
	case line := <-readLines(response.Body):
		t.Fatalf("body sent a line before an update or keepalive: %q", line)
	case <-time.After(100 * time.Millisecond):
	}
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
		go func() { _, err := ReadSnapshotStream(ctx, cancel, response.Body, c); result <- err }()
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
			Kind: "models", Peer: "peer",
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
			Kind: "models", Peer: "peer",
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
			Kind: "models", Peer: "peer",
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
			Kind: "models", Peer: "peer",
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
			Kind: "models", Peer: "peer",
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
			Kind: "models", Peer: "peer",
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
			Kind: "models", Peer: "peer",
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
			Kind: "models", Peer: "peer",
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
			Kind: "models", Peer: "peer",
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
			Kind: "models", Peer: "peer",
			Idle: time.Second, MaxLine: 1024, Current: allow,
			Decode:  func([]byte) (streamFrame, error) { return streamFrame{}, bad },
			OnFrame: func(streamFrame) error { return nil },
		})
		if !errors.Is(err, bad) {
			t.Fatalf("err = %v, want the Decode error", err)
		}
	})
}

func TestSnapshotStreamWallIdleOnChange(t *testing.T) {
	for _, tc := range []struct {
		name  string
		age   time.Duration
		lines bool
		idle  bool
	}{
		{"sleep gap", 5 * time.Second, false, true},
		{"recent line", 500 * time.Millisecond, false, false},
		{"flowing lines", 0, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var clock atomic.Int64
			clock.Store(time.Now().UnixNano())
			oldNow := streamWallNow
			streamWallNow = func() time.Time { return time.Unix(0, clock.Load()) }
			defer func() { streamWallNow = oldNow }()
			var changes testChanges
			ready := make(chan struct{}, 8)
			ctx, cancel, response := openStream(t, func(w http.ResponseWriter, r *http.Request) {
				_ = http.NewResponseController(w).Flush()
				if !tc.lines {
					<-r.Context().Done()
					return
				}
				ticker := time.NewTicker(100 * time.Millisecond)
				defer ticker.Stop()
				for {
					select {
					case <-r.Context().Done():
						return
					case <-ticker.C:
						if _, err := fmt.Fprintln(w, "null"); err != nil {
							return
						}
						if err := http.NewResponseController(w).Flush(); err != nil {
							return
						}
					}
				}
			})
			defer cancel()
			result := make(chan StreamOutcome, 1)
			go func() {
				out, _ := ReadSnapshotStream(ctx, cancel, response.Body, SnapshotStreamClient[streamFrame]{
					Kind: "models", Peer: "peer",
					Idle: time.Second, MaxLine: 4096, Current: func() bool { return true },
					Changed:     func() <-chan struct{} { ch := changes.Changed(); ready <- struct{}{}; return ch },
					OnKeepalive: func() { ready <- struct{}{} },
				})
				result <- out
			}()
			<-ready
			if tc.lines {
				<-ready
			}
			clock.Add(int64(tc.age))
			changes.notify()
			<-ready // watcher captured next generation
			if tc.idle {
				select {
				case out := <-result:
					if out.End != StreamEndIdle {
						t.Fatalf("end = %s", out.End)
					}
				case <-time.After(200 * time.Millisecond):
					t.Fatal("wake did not end idle stream")
				}
			} else {
				select {
				case out := <-result:
					t.Fatalf("healthy stream ended: %+v", out)
				case <-time.After(200 * time.Millisecond):
				}
				cancel()
				<-result
			}
		})
	}
}

func TestSnapshotStreamOutcome(t *testing.T) {
	for _, tc := range []struct {
		name                string
		body                string
		current             bool
		decodeErr, frameErr error
		cancelled           bool
		want                StreamEnd
	}{
		{name: "EOF", body: `{"n":1}` + "\n", current: true, want: StreamEndEOF},
		{name: "decode", body: "bad\n", current: true, decodeErr: errors.New("decode"), want: StreamEndDecode},
		{name: "refused", body: "{}\n", current: true, frameErr: errors.New("refused"), want: StreamEndRefused},
		{name: "authority", body: "{}\n", want: StreamEndAuthority},
		{name: "cancelled", body: "{}\n", current: true, cancelled: true, want: StreamEndCancelled},
		{name: "error", body: strings.Repeat("x", 8192), current: true, want: StreamEndError},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			if tc.cancelled {
				cancel()
			}
			out, _ := ReadSnapshotStream(ctx, cancel, strings.NewReader(tc.body), SnapshotStreamClient[streamFrame]{
				Kind: "models", Peer: "peer",
				Idle: time.Second, MaxLine: 4096, Current: func() bool { return tc.current },
				Decode:  func([]byte) (streamFrame, error) { return streamFrame{}, tc.decodeErr },
				OnFrame: func(streamFrame) error { return tc.frameErr },
			})
			if out.End != tc.want || out.Healthy != 0 {
				t.Fatalf("outcome = %+v, want %s with zero duration", out, tc.want)
			}
		})
	}
}

func TestSnapshotStreamHealthyExcludesTrailingSilence(t *testing.T) {
	ctx, cancel, response := openStream(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = fmt.Fprintln(w, "{}")
		_ = http.NewResponseController(w).Flush()
		time.Sleep(10 * time.Millisecond)
		_, _ = fmt.Fprintln(w, "null")
		_ = http.NewResponseController(w).Flush()
		<-r.Context().Done()
	})
	out, err := ReadSnapshotStream(ctx, cancel, response.Body, SnapshotStreamClient[streamFrame]{
		Kind: "models", Peer: "peer",
		Idle: 25 * time.Millisecond, MaxLine: 4096, Current: func() bool { return true },
		Decode: func([]byte) (streamFrame, error) { return streamFrame{}, nil }, OnFrame: func(streamFrame) error { return nil },
	})
	if !errors.Is(err, context.Canceled) || out.End != StreamEndIdle {
		t.Fatalf("outcome/error = %+v / %v", out, err)
	}
	if out.Healthy < 5*time.Millisecond || out.Healthy >= 25*time.Millisecond {
		t.Fatalf("healthy = %v, want about 10ms", out.Healthy)
	}
}

func TestSnapshotStreamBufferedLineAfterIdle(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	frames := 0
	out, _ := ReadSnapshotStream(ctx, cancel, strings.NewReader("{}\n{}\n"), SnapshotStreamClient[streamFrame]{
		Kind: "models", Peer: "peer",
		Idle: 10 * time.Millisecond, MaxLine: 4096, Current: func() bool { return true },
		Decode:  func([]byte) (streamFrame, error) { return streamFrame{}, nil },
		OnFrame: func(streamFrame) error { frames++; <-ctx.Done(); return nil },
	})
	if out.End != StreamEndIdle || frames != 1 {
		t.Fatalf("outcome/frames = %+v / %d", out, frames)
	}
}

func TestSnapshotStreamAuthorityRecheckOutcome(t *testing.T) {
	ctx, cancel, response := openStream(t, func(w http.ResponseWriter, r *http.Request) {
		_ = http.NewResponseController(w).Flush()
		<-r.Context().Done()
	})
	out, _ := ReadSnapshotStream(ctx, cancel, response.Body, SnapshotStreamClient[streamFrame]{
		Kind: "models", Peer: "peer",
		Idle: time.Second, Recheck: time.Millisecond, MaxLine: 4096, Current: func() bool { return false },
	})
	if out.End != StreamEndAuthority || out.Healthy != 0 {
		t.Fatalf("outcome = %+v", out)
	}
}

func TestSnapshotStreamErrorEndSurvivesCallerCancellation(t *testing.T) {
	for _, end := range []StreamEnd{StreamEndDecode, StreamEndRefused} {
		t.Run(string(end), func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			bad := errors.New("frame error")
			out, err := ReadSnapshotStream(ctx, cancel, strings.NewReader("{}\n"), SnapshotStreamClient[streamFrame]{
				Kind: "models", Peer: "peer",
				Idle: time.Second, MaxLine: 4096, Current: func() bool { return true },
				Decode: func([]byte) (streamFrame, error) {
					if end == StreamEndDecode {
						cancel()
						return streamFrame{}, bad
					}
					return streamFrame{}, nil
				},
				OnFrame: func(streamFrame) error { cancel(); return bad },
			})
			if out.End != end || !errors.Is(err, bad) {
				t.Fatalf("outcome/error = %+v / %v", out, err)
			}
		})
	}
}
