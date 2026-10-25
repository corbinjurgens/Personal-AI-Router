// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package rpc

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func sessionPipe(t *testing.T) (net.Conn, net.Conn) {
	t.Helper()
	client, server := net.Pipe()
	t.Cleanup(func() { _ = client.Close(); _ = server.Close() })
	for _, conn := range []net.Conn{client, server} {
		if err := conn.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
			t.Fatal(err)
		}
	}
	return client, server
}

func sessionEvent(t *testing.T, events <-chan Event, kind EventKind, generation uint64, method string) Event {
	t.Helper()
	select {
	case event, ok := <-events:
		if !ok {
			t.Fatal("event stream closed unexpectedly")
		}
		if event.Kind != kind || event.Generation != generation {
			t.Fatalf("event = %+v, want kind %d generation %d", event, kind, generation)
		}
		if method != "" && (event.Message == nil || event.Message.Method != method) {
			t.Fatalf("event message = %+v, want method %q", event.Message, method)
		}
		return event
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for session event")
		return Event{}
	}
}

func sessionResult(t *testing.T, result <-chan error) error {
	t.Helper()
	select {
	case err := <-result:
		return err
	case <-time.After(3 * time.Second):
		t.Fatal("session operation did not return")
		return nil
	}
}

func sessionPush(t *testing.T, server *Codec, method string) {
	t.Helper()
	if err := server.Write(&Message{JSONRPC: "2.0", Method: method}); err != nil {
		t.Fatalf("push %s: %v", method, err)
	}
}

func TestReconnectSameClientRejectsPendingAndNeverReplays(t *testing.T) {
	first, peer1 := sessionPipe(t)
	second, peer2 := sessionPipe(t)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(func() { cancel(); _ = first.Close(); _ = peer1.Close(); _ = second.Close(); _ = peer2.Close() })
	replacement := make(chan io.ReadWriteCloser)
	client := NewReconnectingClient(first, func(ctx context.Context) (io.ReadWriteCloser, error) {
		select {
		case conn := <-replacement:
			return conn, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	})
	client.retryDelay = func(int) time.Duration { return 0 }
	events := client.Events()
	run := make(chan error, 1)
	go func() { run <- client.Run(ctx) }()
	server1 := NewCodec(peer1, peer1)
	sessionPush(t, server1, "app:ready")
	sessionEvent(t, events, EventNotification, 1, "app:ready")
	pending := make(chan error, 1)
	go func() { _, err := client.Call(ctx, "mutate-once", nil); pending <- err }()
	request, err := server1.Read()
	if err != nil || request.Method != "mutate-once" {
		t.Fatalf("pending request = %+v, err = %v", request, err)
	}
	if err := peer1.Close(); err != nil {
		t.Fatal(err)
	}
	loss := sessionEvent(t, events, EventDisconnected, 1, "")
	if !errors.Is(loss.Err, io.EOF) {
		t.Fatalf("disconnect error = %v, want EOF", loss.Err)
	}
	if err := sessionResult(t, pending); err == nil {
		t.Fatal("pending mutation succeeded after loss")
	}
	if client.IsCurrent(1) {
		t.Fatal("lost generation is still current")
	}
	if _, err := client.Call(ctx, "during-outage", nil); err == nil {
		t.Fatal("call during outage was accepted")
	}
	if err := client.Notify("during-outage", nil); err == nil {
		t.Fatal("notification during outage was accepted")
	}
	select {
	case replacement <- second:
	case <-time.After(3 * time.Second):
		t.Fatal("client did not dial replacement")
	}
	server2 := NewCodec(peer2, peer2)
	sessionPush(t, server2, "app:ready")
	sessionEvent(t, events, EventNotification, 2, "app:ready")
	if client.Generation() != 2 || !client.IsCurrent(2) || client.IsCurrent(1) {
		t.Fatal("replacement generation was not installed")
	}
	if _, err := client.CallGeneration(ctx, 1, "stale-mutation", nil); err == nil {
		t.Fatal("stale generation mutation was accepted")
	}
	fresh := make(chan error, 1)
	go func() {
		msg, err := client.CallGeneration(ctx, 2, "fresh-request", nil)
		if err == nil {
			var result struct {
				Replacement bool `json:"replacement"`
			}
			err = json.Unmarshal(msg.Result, &result)
			if err == nil && !result.Replacement {
				err = errors.New("fresh response did not come from replacement")
			}
		}
		fresh <- err
	}()
	request, err = server2.Read()
	if err != nil {
		t.Fatalf("replacement read: %v", err)
	}
	if request.Method != "fresh-request" {
		t.Fatalf("first replacement request = %q; pending/outage/stale work was replayed", request.Method)
	}
	if err := server2.Write(&Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(`{"replacement":true}`)}); err != nil {
		t.Fatalf("replacement response: %v", err)
	}
	if err := sessionResult(t, fresh); err != nil {
		t.Fatal(err)
	}
	sessionPush(t, server2, "replacement:push")
	sessionEvent(t, events, EventNotification, 2, "replacement:push")
	client.Close()
	if err := sessionResult(t, run); !errors.Is(err, context.Canceled) {
		t.Fatalf("Close Run result = %v", err)
	}
	if _, open := <-events; open {
		t.Fatal("event stream remained open after Close")
	}
}

func TestReconnectContinuesAfterFailedDials(t *testing.T) {
	first, peer1 := sessionPipe(t)
	second, peer2 := sessionPipe(t)
	t.Cleanup(func() { _ = peer1.Close(); _ = peer2.Close(); _ = second.Close() })
	var attempts atomic.Int32
	client := NewReconnectingClient(first, func(context.Context) (io.ReadWriteCloser, error) {
		if attempts.Add(1) <= 8 {
			return nil, errors.New("service absent")
		}
		return second, nil
	})
	client.retryDelay = func(int) time.Duration { return 0 }
	t.Cleanup(client.Close)
	events := client.Events()
	go func() { _ = client.Run(context.Background()) }()
	if err := peer1.Close(); err != nil {
		t.Fatal(err)
	}
	sessionEvent(t, events, EventDisconnected, 1, "")
	sessionPush(t, NewCodec(peer2, peer2), "app:ready")
	sessionEvent(t, events, EventNotification, 2, "app:ready")
	if got := attempts.Load(); got != 9 {
		t.Fatalf("dial attempts = %d, want 9", got)
	}
}

func TestReconnectDelayIsCapped(t *testing.T) {
	for _, test := range []struct {
		attempt int
		want    time.Duration
	}{{0, 100 * time.Millisecond}, {1, 200 * time.Millisecond}, {4, 1600 * time.Millisecond}, {5, 2 * time.Second}, {1000, 2 * time.Second}} {
		if got := reconnectDelay(test.attempt); got != test.want {
			t.Errorf("attempt %d delay = %s, want %s", test.attempt, got, test.want)
		}
	}
}

func TestReconnectCloseCancelsBackoff(t *testing.T) {
	first, peer := sessionPipe(t)
	t.Cleanup(func() { _ = peer.Close() })
	backoff := make(chan struct{}, 1)
	var dials atomic.Int32
	client := NewReconnectingClient(first, func(context.Context) (io.ReadWriteCloser, error) {
		dials.Add(1)
		return nil, errors.New("unexpected dial")
	})
	client.retryDelay = func(int) time.Duration { backoff <- struct{}{}; return time.Hour }
	result := make(chan error, 1)
	go func() { result <- client.Run(context.Background()) }()
	if err := peer.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-backoff:
	case <-time.After(3 * time.Second):
		t.Fatal("session did not enter retry backoff")
	}
	client.Close()
	if err := sessionResult(t, result); !errors.Is(err, context.Canceled) {
		t.Fatalf("Run = %v, want canceled", err)
	}
	if got := dials.Load(); got != 0 {
		t.Fatalf("dials after Close in backoff = %d, want 0", got)
	}
}

func TestReconnectCancellationClosesLateDial(t *testing.T) {
	first, peer1 := sessionPipe(t)
	late, peer2 := sessionPipe(t)
	t.Cleanup(func() { _ = peer1.Close(); _ = late.Close(); _ = peer2.Close() })
	started, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	client := NewReconnectingClient(first, func(context.Context) (io.ReadWriteCloser, error) {
		close(started)
		<-release // Simulate an OS dial completing after cancellation.
		return late, nil
	})
	client.retryDelay = func(int) time.Duration { return 0 }
	result := make(chan error, 1)
	go func() { result <- client.Run(context.Background()) }()
	if err := peer1.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("dial never started")
	}
	client.Close()
	if err := sessionResult(t, result); !errors.Is(err, context.Canceled) {
		t.Fatalf("Run before late dial returns = %v", err)
	}
	unblock()
	if err := peer2.SetReadDeadline(time.Now().Add(3 * time.Second)); err != nil {
		t.Fatal(err)
	}
	var b [1]byte
	if _, err := peer2.Read(b[:]); !errors.Is(err, io.EOF) {
		t.Fatalf("late transport read = %v, want EOF from closed dial result", err)
	}
	if client.Generation() != 1 {
		t.Fatal("late dial installed a new generation after Close")
	}
}

func TestSessionStopSuppressesReconnectBeforeFinalRequest(t *testing.T) {
	conn, peer := sessionPipe(t)
	t.Cleanup(func() { _ = peer.Close() })
	var dials atomic.Int32
	client := NewReconnectingClient(conn, func(context.Context) (io.ReadWriteCloser, error) {
		dials.Add(1)
		return nil, errors.New("unexpected reconnect after stop")
	})
	t.Cleanup(client.Close)
	result := make(chan error, 1)
	go func() { result <- client.Run(context.Background()) }()
	client.StopReconnecting()
	stop := make(chan error, 1)
	go func() { _, err := client.Call(context.Background(), "service/stop", nil); stop <- err }()
	server := NewCodec(peer, peer)
	request, err := server.Read()
	if err != nil || request.Method != "service/stop" {
		t.Fatalf("stop request = %+v, err = %v", request, err)
	}
	if err := server.Write(&Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(`{}`)}); err != nil {
		t.Fatal(err)
	}
	if err := sessionResult(t, stop); err != nil {
		t.Fatalf("stop response: %v", err)
	}
	if err := peer.Close(); err != nil {
		t.Fatal(err)
	}
	if err := sessionResult(t, result); err != nil {
		t.Fatalf("Run after stop: %v", err)
	}
	if got := dials.Load(); got != 0 {
		t.Fatalf("dial attempts after stop = %d, want 0", got)
	}
}

func TestClientCancellationInterruptsBlockedRead(t *testing.T) {
	conn, peer := sessionPipe(t)
	t.Cleanup(func() { _ = peer.Close() })
	started := make(chan struct{})
	reader := &sessionObservedReader{Conn: conn, started: started}
	client := NewClient(reader, conn)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	result := make(chan error, 1)
	go func() { result <- client.Run(ctx) }()
	select {
	case <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("read did not start")
	}
	cancel()
	if err := sessionResult(t, result); !errors.Is(err, context.Canceled) {
		t.Fatalf("Run blocked read = %v, want canceled", err)
	}
}

func TestReconnectLossInterruptsBlockedWrite(t *testing.T) {
	conn, peer := sessionPipe(t)
	t.Cleanup(func() { _ = peer.Close() })
	started := make(chan struct{})
	writer := &sessionBlockedWriter{Conn: conn, started: started}
	client := NewReconnectingClient(writer, func(ctx context.Context) (io.ReadWriteCloser, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	})
	t.Cleanup(client.Close)
	go func() { _ = client.Run(context.Background()) }()
	call := make(chan error, 1)
	go func() { _, err := client.Call(context.Background(), "blocked-mutation", nil); call <- err }()
	select {
	case <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("write did not start")
	}
	if err := peer.Close(); err != nil {
		t.Fatal(err)
	}
	if err := sessionResult(t, call); err == nil {
		t.Fatal("blocked mutation succeeded after transport loss")
	}
}

type sessionBlockedWriter struct {
	net.Conn
	started chan struct{}
}

func (w *sessionBlockedWriter) Write(p []byte) (int, error) {
	close(w.started)
	return w.Conn.Write(p)
}

func TestReconnectEventsDeliverNotificationBurst(t *testing.T) {
	conn, peer := sessionPipe(t)
	client := NewReconnectingClient(conn, func(ctx context.Context) (io.ReadWriteCloser, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	})
	t.Cleanup(client.Close)
	events := client.Events()
	go func() { _ = client.Run(context.Background()) }()
	consumed := make(chan error, 1)
	go func() {
		for i := 0; i < 600; i++ {
			event, open := <-events
			if !open || event.Kind != EventNotification || event.Generation != 1 || event.Message == nil || event.Message.Method != "burst:push" {
				consumed <- errors.New("event stream lost burst notification")
				return
			}
		}
		consumed <- nil
	}()
	server := NewCodec(peer, peer)
	for i := 0; i < 600; i++ {
		sessionPush(t, server, "burst:push")
	}
	if err := sessionResult(t, consumed); err != nil {
		t.Fatal(err)
	}
}

func TestReconnectWriteFailureInterruptsRead(t *testing.T) {
	conn, peer := sessionPipe(t)
	client := NewReconnectingClient(&sessionFailedWriter{Conn: conn}, func(ctx context.Context) (io.ReadWriteCloser, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	})
	t.Cleanup(client.Close)
	events := client.Events()
	go func() { _ = client.Run(context.Background()) }()
	if _, err := client.Call(context.Background(), "write-fails", nil); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatalf("write failure = %v, want ErrClosedPipe", err)
	}
	sessionEvent(t, events, EventDisconnected, 1, "")
	if client.IsCurrent(1) {
		t.Fatal("write failure left transport connected")
	}
	var b [1]byte
	if _, err := peer.Read(b[:]); !errors.Is(err, io.EOF) {
		t.Fatalf("peer read = %v, want EOF after write failure closed transport", err)
	}
}

type sessionFailedWriter struct{ net.Conn }

func (*sessionFailedWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }

func TestClientWriteFailureBeforeRunIsTerminal(t *testing.T) {
	conn, _ := sessionPipe(t)
	writer := &sessionFailedWriter{Conn: conn}
	client := NewClient(writer, writer)
	if _, err := client.Call(context.Background(), "fails-before-read-loop", nil); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatalf("pre-Run write = %v, want ErrClosedPipe", err)
	}
	result := make(chan error, 1)
	go func() { result <- client.Run(context.Background()) }()
	if err := sessionResult(t, result); !errors.Is(err, ErrStreamBroken) {
		t.Fatalf("Run after pre-Run write failure = %v, want ErrStreamBroken", err)
	}
}

func TestHistoryBorrowerPinsReadUntilActualApply(t *testing.T) {
	a, b := sessionPipe(t)
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(release) }) })
	c := historyTestClient(t, a, a, func(uint64, bool, []byte) error { close(entered); <-release; return nil })
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go func() { _ = c.Run(ctx) }()
	s := NewCodec(b, b)
	request := historyRequest(t, c, s)
	if err := s.Write(&Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(`{"requests":[]}`)}); err != nil {
		t.Fatal(err)
	}
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("apply not entered")
	}
	if c.history.readMu.TryLock() {
		c.history.readMu.Unlock()
		t.Fatal("raw lease released before apply completion")
	}
	once.Do(func() { close(release) })
	historyBarrier(t, c, s)
	if c.HistoryState().Pending {
		t.Fatal("actual apply completion did not complete pending")
	}
}

func TestHistoryOrdinaryCallbackDoesNotBorrowRawArena(t *testing.T) {
	a, b := sessionPipe(t)
	c := historyTestClient(t, a, a, func(uint64, bool, []byte) error { return nil })
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(release) }) })
	owned := make(chan error, 1)
	c.SetNotificationFilter(func(msg *Message) bool {
		close(entered)
		<-release
		var value struct {
			Owned bool `json:"owned"`
		}
		err := json.Unmarshal(msg.Params, &value)
		if err == nil && !value.Owned {
			err = errors.New("ordinary callback lost owned value")
		}
		owned <- err
		return true
	})
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go func() { _ = c.Run(ctx) }()
	s := NewCodec(b, b)
	if err := s.Write(&Message{JSONRPC: "2.0", Method: "ordinary", Params: json.RawMessage(`{"owned":true}`)}); err != nil {
		t.Fatal(err)
	}
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("ordinary filter not entered")
	}
	if !c.history.readMu.TryLock() {
		t.Fatal("ordinary callback retains whole raw read owner")
	}
	clear(c.history.raw[:1024])
	c.history.readMu.Unlock()
	closed := make(chan struct{})
	go func() { c.breakTransport(c.current); close(closed) }()
	select {
	case <-closed:
	case <-time.After(3 * time.Second):
		t.Fatal("read retirement waited for whole ordinary callback")
	}
	once.Do(func() { close(release) })
	if err := sessionResult(t, owned); err != nil {
		t.Fatal(err)
	}
}

func TestHistoryOldGenerationCannotResetReplacement(t *testing.T) {
	c := historyTestClient(t, strings.NewReader(""), discardWriter{}, func(uint64, bool, []byte) error { return nil })
	old := c.current
	c.breakTransport(old)
	newTransport := &transport{codec: &Codec{writer: discardWriter{}}, generation: 2, pending: make(map[int64]chan *Message), done: make(chan struct{}), close: func() {}}
	c.mu.Lock()
	c.current = newTransport
	c.generation = 2
	c.mu.Unlock()
	c.history.bind(newTransport, strings.NewReader(""))
	c.breakTransport(old)
	if state := c.HistoryState(); state.Generation != 2 || !state.Connected {
		t.Fatalf("old completion changed replacement: %+v", state)
	}
}

type historyBlockedControl struct {
	net.Conn
	armed            atomic.Bool
	entered, release chan struct{}
}

func (w *historyBlockedControl) Write(frame []byte) (int, error) {
	if w.armed.Swap(false) {
		close(w.entered)
		<-w.release
	}
	return w.Conn.Write(frame)
}

func TestHistoryControlWriteRemainsOwnedAfterLogicalQuarantine(t *testing.T) {
	a, b := sessionPipe(t)
	w := &historyBlockedControl{Conn: a, entered: make(chan struct{}), release: make(chan struct{})}
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(w.release) }) })
	c := historyTestClient(t, w, w, func(uint64, bool, []byte) error { return nil })
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go func() { _ = c.Run(ctx) }()
	s := NewCodec(b, b)
	ordinary := make(chan error, 1)
	go func() { _, err := c.Call(ctx, "ordinary-before-A", nil); ordinary <- err }()
	o, err := s.Read()
	if err != nil {
		t.Fatal(err)
	}
	w.armed.Store(true)
	c.SetHistorySubscription(true)
	select {
	case <-w.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("history control Write not entered")
	}
	c.history.mu.Lock()
	before := string(c.history.control[:])
	c.history.mu.Unlock()
	if _, err := b.Write([]byte("{bad\n")); err != nil {
		t.Fatal(err)
	}
	if err := s.Write(&Message{JSONRPC: "2.0", ID: o.ID, Result: json.RawMessage(`{}`)}); err != nil {
		t.Fatal(err)
	}
	if err := sessionResult(t, ordinary); err != nil {
		t.Fatal(err)
	}
	c.history.mu.Lock()
	active, pending, unchanged := c.history.writeActive, c.history.pending, string(c.history.control[:]) == before
	c.history.mu.Unlock()
	if !active || pending || !unchanged {
		t.Fatal("logical quarantine released or overwrote active control storage")
	}
	once.Do(func() { close(w.release) })
	late, err := s.Read()
	if err != nil {
		t.Fatal(err)
	}
	if late.ID == nil || string(*late.ID) != `"jh:1"` {
		t.Fatalf("blocked control changed: %+v", late)
	}
	bRequest, err := s.Read()
	if err != nil {
		t.Fatal(err)
	}
	if bRequest.ID == nil || string(*bRequest.ID) != `"jh:2"` {
		t.Fatalf("next control = %+v", bRequest)
	}
	if err := s.Write(&Message{JSONRPC: "2.0", ID: bRequest.ID, Result: json.RawMessage(`{"requests":[]}`)}); err != nil {
		t.Fatal(err)
	}
	historyBarrier(t, c, s)
}

func TestHistoryBlockedOldWriteCannotReuseReplacementControl(t *testing.T) {
	a, b := sessionPipe(t)
	next, peer := sessionPipe(t)
	w := &historyBlockedControl{Conn: a, entered: make(chan struct{}), release: make(chan struct{})}
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(w.release) }) })
	w.armed.Store(true)
	replacement := make(chan io.ReadWriteCloser)
	c := historyTestReconnectingClient(t, w, func(ctx context.Context) (io.ReadWriteCloser, error) {
		select {
		case conn := <-replacement:
			return conn, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}, func(uint64, bool, []byte) error { return nil })
	c.retryDelay = func(int) time.Duration { return 0 }
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go func() { _ = c.Run(ctx) }()
	c.SetHistorySubscription(true)
	select {
	case <-w.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("old Write not entered")
	}
	c.history.mu.Lock()
	oldFrame := string(c.history.control[:])
	c.history.mu.Unlock()
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	sessionEvent(t, c.Events(), EventDisconnected, 1, "")
	select {
	case replacement <- next:
	case <-time.After(3 * time.Second):
		t.Fatal("replacement not dialed")
	}
	s := NewCodec(peer, peer)
	sessionPush(t, s, "app:ready")
	sessionEvent(t, c.Events(), EventNotification, 2, "app:ready")
	historyBarrier(t, c, s)
	c.history.mu.Lock()
	pinned := c.history.writeActive && string(c.history.control[:]) == oldFrame
	c.history.mu.Unlock()
	if !pinned {
		t.Fatal("new generation reused old active Write backing")
	}
	once.Do(func() { close(w.release) })
	request, err := s.Read()
	if err != nil {
		t.Fatal(err)
	}
	if request.ID == nil || string(*request.ID) != `"jh:1"` {
		t.Fatalf("fresh generation ID = %+v", request)
	}
	if err := s.Write(&Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(`{"requests":[]}`)}); err != nil {
		t.Fatal(err)
	}
	historyBarrier(t, c, s)
	if state := c.HistoryState(); state.Generation != 2 || state.Loading {
		t.Fatalf("old completion changed new state: %+v", state)
	}
}

func TestHistoryRefusalTokenPinnedUntilWriterCompletion(t *testing.T) {
	a, b := sessionPipe(t)
	w := &historyBlockedControl{Conn: a, entered: make(chan struct{}), release: make(chan struct{})}
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(w.release) }) })
	w.armed.Store(true)
	c := historyTestClient(t, w, w, func(uint64, bool, []byte) error { return nil })
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go func() { _ = c.Run(ctx) }()
	if _, err := b.Write([]byte(`{"jsonrpc":"2.0","id":"first\u002did","method":"routing/request","params":{}}` + "\n")); err != nil {
		t.Fatal(err)
	}
	select {
	case <-w.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("refusal Write not entered")
	}
	once.Do(func() { close(w.release) })
	s := NewCodec(b, b)
	reply, err := s.Read()
	if err != nil {
		t.Fatal(err)
	}
	if reply.ID == nil || string(*reply.ID) != `"first\u002did"` || reply.Error == nil || reply.Error.Code != -32601 {
		t.Fatalf("admitted refusal changed/lost: %+v", reply)
	}
	historyBarrier(t, c, s)
}

type historyDelayedWriteError struct {
	net.Conn
	entered, release chan struct{}
	returned         chan error
}

func (w *historyDelayedWriteError) Write(frame []byte) (int, error) {
	close(w.entered)
	<-w.release
	// The first transport's real Close has already run. Preserve its actual
	// Write error rather than supplying a no-op close or synthetic completion.
	n, err := w.Conn.Write(frame)
	w.returned <- err
	return n, err
}

func historyReadMessage(t *testing.T, codec *Codec) *Message {
	t.Helper()
	type result struct {
		message *Message
		err     error
	}
	done := make(chan result, 1)
	go func() { message, err := codec.Read(); done <- result{message, err} }()
	select {
	case got := <-done:
		if got.err != nil {
			t.Fatal(got.err)
		}
		return got.message
	case <-time.After(3 * time.Second):
		t.Fatal("outbound progress required input to unblock")
		return nil
	}
}

func TestHistoryOldWriteErrorDoesNotJoinReplacementIdleRead(t *testing.T) {
	a, peerA := sessionPipe(t)
	b, peerB := sessionPipe(t)
	old := &historyDelayedWriteError{Conn: a, entered: make(chan struct{}), release: make(chan struct{}), returned: make(chan error, 1)}
	var release sync.Once
	t.Cleanup(func() { release.Do(func() { close(old.release) }) })
	idleB := &sessionObservedReader{Conn: b, started: make(chan struct{})}
	replacement := make(chan io.ReadWriteCloser)
	c := historyTestReconnectingClient(t, old, func(ctx context.Context) (io.ReadWriteCloser, error) {
		select {
		case conn := <-replacement:
			return conn, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}, func(uint64, bool, []byte) error { return nil })
	c.retryDelay = func(int) time.Duration { return 0 }
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go func() { _ = c.Run(ctx) }()
	c.SetHistorySubscription(true)
	select {
	case <-old.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("A control Write not entered")
	}
	if err := peerA.Close(); err != nil {
		t.Fatal(err)
	}
	sessionEvent(t, c.Events(), EventDisconnected, 1, "")
	select {
	case replacement <- idleB:
	case <-time.After(3 * time.Second):
		t.Fatal("B was not dialed")
	}
	select {
	case <-idleB.started:
	case <-time.After(3 * time.Second):
		t.Fatal("B did not enter its idle physical Read")
	}
	if c.history.readMu.TryLock() {
		c.history.readMu.Unlock()
		t.Fatal("B idle physical Read is not admitted")
	}
	release.Do(func() { close(old.release) })
	if err := sessionResult(t, old.returned); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatalf("A actual Write error = %v", err)
	}
	s := NewCodec(peerB, peerB)
	// No bytes have been sent into B. Its subscription and an ordinary numeric
	// request must both be written while that same physical Read remains idle.
	subscribe := historyReadMessage(t, s)
	if subscribe.Method != "routing:subscribe" || subscribe.ID == nil || string(*subscribe.ID) != `"jh:1"` {
		t.Fatalf("B subscription = %+v", subscribe)
	}
	ordinary := make(chan error, 1)
	go func() { _, err := c.Call(ctx, "ordinary-while-B-input-idle", nil); ordinary <- err }()
	request := historyReadMessage(t, s)
	var id int64
	if request.ID == nil || json.Unmarshal(*request.ID, &id) != nil || request.Method != "ordinary-while-B-input-idle" {
		t.Fatalf("B ordinary request = %+v", request)
	}
	if c.history.readMu.TryLock() {
		c.history.readMu.Unlock()
		t.Fatal("B input unexpectedly ceased being idle")
	}
	// A round trip necessarily consumes B input; only now provide its replies.
	if err := s.Write(&Message{JSONRPC: "2.0", ID: subscribe.ID, Result: json.RawMessage(`{"requests":[]}`)}); err != nil {
		t.Fatal(err)
	}
	if err := s.Write(&Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(`{}`)}); err != nil {
		t.Fatal(err)
	}
	if err := sessionResult(t, ordinary); err != nil {
		t.Fatal(err)
	}
	sessionPush(t, s, "ordinary-after-old-write-error")
	sessionEvent(t, c.Events(), EventNotification, 2, "ordinary-after-old-write-error")
	if state := c.HistoryState(); state.Generation != 2 || state.Loading || !state.Connected {
		t.Fatalf("A completion changed B: %+v", state)
	}
}

func TestHistoryWrongDirectionRefusalsDoNotBlockOrdinaryInput(t *testing.T) {
	a, peer := sessionPipe(t)
	w := &historyBlockedControl{Conn: a, entered: make(chan struct{}), release: make(chan struct{})}
	var release sync.Once
	t.Cleanup(func() { release.Do(func() { close(w.release) }) })
	var applied atomic.Int32
	c := historyTestClient(t, w, w, func(uint64, bool, []byte) error { applied.Add(1); return nil })
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go func() { _ = c.Run(ctx) }()
	s := NewCodec(peer, peer)
	ordinary := make(chan error, 1)
	go func() { _, err := c.Call(ctx, "ordinary-before-refusal", nil); ordinary <- err }()
	request := historyReadMessage(t, s)
	var numeric int64
	if request.ID == nil || json.Unmarshal(*request.ID, &numeric) != nil {
		t.Fatal("ordinary pending call is not numeric")
	}
	w.armed.Store(true)
	if _, err := peer.Write([]byte(`{"jsonrpc":"2.0","id":"first\u002did","method":"routing/request","params":{}}` + "\n")); err != nil {
		t.Fatal(err)
	}
	select {
	case <-w.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("first refusal Write not entered")
	}
	c.history.mu.Lock()
	before := c.history.refusal
	beforeN := c.history.refusalN
	c.history.mu.Unlock()
	for _, frame := range []string{
		`{"jsonrpc":"2.0","id":"second","method":"routing/request","params":{}}`,
		`{"jsonrpc":"2.0","id":"third","method":"routing:subscribe","params":{}}`,
		`{"jsonrpc":"2.0","id":"fourth","method":"routing/request","params":{}}`,
	} {
		if _, err := peer.Write([]byte(frame + "\n")); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.Write(&Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(`{}`)}); err != nil {
		t.Fatal(err)
	}
	if err := sessionResult(t, ordinary); err != nil {
		t.Fatal(err)
	}
	sessionPush(t, s, "ordinary-behind-refusals")
	sessionEvent(t, c.Events(), EventNotification, 1, "ordinary-behind-refusals")
	select {
	case <-w.release:
		t.Fatal("ordinary progress followed premature Write release")
	default:
	}
	c.history.mu.Lock()
	pinned := c.history.writeActive && c.history.refusalActive && c.history.refusalN == beforeN && c.history.refusal == before
	c.history.mu.Unlock()
	state := c.HistoryState()
	if !pinned || state.RefusalDiscardCategory != "history-wrong-direction" || state.RefusalDiscardCount != 3 || state.Pending || state.Loading || state.Retry || state.Quarantined || state.Generation != 1 || !state.Connected || applied.Load() != 0 {
		t.Fatalf("wrong-direction effects or lost pinned refusal: %+v", state)
	}
	release.Do(func() { close(w.release) })
	reply := historyReadMessage(t, s)
	if reply.ID == nil || string(*reply.ID) != `"first\u002did"` || reply.Error == nil || reply.Error.Code != -32601 {
		t.Fatalf("admitted refusal changed: %+v", reply)
	}
	// A subsequent control Write proves the sole writer passed actual refusal
	// completion. Extra unaccepted requests must not have queued any replies.
	c.SetHistorySubscription(true)
	subscribe := historyReadMessage(t, s)
	if subscribe.Method != "routing:subscribe" {
		t.Fatalf("unexpected queued refusal: %+v", subscribe)
	}
	c.history.mu.Lock()
	released := !c.history.refusalActive && c.history.refusalN == 0
	c.history.mu.Unlock()
	if !released {
		t.Fatal("completed refusal retained its slot")
	}
	if err := s.Write(&Message{JSONRPC: "2.0", ID: subscribe.ID, Result: json.RawMessage(`{"requests":[]}`)}); err != nil {
		t.Fatal(err)
	}
	historyBarrier(t, c, s)
}

type sessionObservedReader struct {
	net.Conn
	started chan struct{}
	once    sync.Once
}

func (r *sessionObservedReader) Read(p []byte) (int, error) {
	r.once.Do(func() { close(r.started) })
	return r.Conn.Read(p)
}

type historyPausedOrdinaryReturn struct {
	net.Conn
	readStarted, written, release chan struct{}
	readOnce                      sync.Once
	pause                         atomic.Bool
}

func (r *historyPausedOrdinaryReturn) Read(p []byte) (int, error) {
	r.readOnce.Do(func() { close(r.readStarted) })
	return r.Conn.Read(p)
}

func (r *historyPausedOrdinaryReturn) Write(p []byte) (int, error) {
	n, err := r.Conn.Write(p)
	if r.pause.Swap(false) {
		close(r.written)
		<-r.release
	}
	return n, err
}

func TestHistoryFactoryRepeatedRetirementKeepsOneReaderWithOldCallers(t *testing.T) {
	type fixture struct {
		conn    *historyPausedOrdinaryReturn
		peer    net.Conn
		release sync.Once
	}
	makeFixture := func() *fixture {
		a, b := sessionPipe(t)
		f := &fixture{conn: &historyPausedOrdinaryReturn{Conn: a, readStarted: make(chan struct{}), written: make(chan struct{}), release: make(chan struct{})}, peer: b}
		t.Cleanup(func() { f.release.Do(func() { close(f.conn.release) }) })
		return f
	}
	fixtures := []*fixture{makeFixture(), makeFixture(), makeFixture(), makeFixture()}
	replacement := make(chan io.ReadWriteCloser)
	c := historyTestReconnectingClient(t, fixtures[0].conn, func(ctx context.Context) (io.ReadWriteCloser, error) {
		select {
		case conn := <-replacement:
			return conn, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}, func(uint64, bool, []byte) error { return nil })
	c.retryDelay = func(int) time.Duration { return 0 }
	reader, raw, offsets := c.history.reader, &c.history.raw[:1][0], &c.history.offsets[:1][0]
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go func() { _ = c.Run(ctx) }()
	retained := make([]*transport, 0, len(fixtures)-1)
	results := make([]chan error, 0, len(fixtures)-1)
	for i, f := range fixtures[:len(fixtures)-1] {
		select {
		case <-f.conn.readStarted:
		case <-time.After(3 * time.Second):
			t.Fatal("current physical Read not entered")
		}
		c.mu.Lock()
		old := c.current
		c.mu.Unlock()
		f.conn.pause.Store(true)
		result := make(chan error, 1)
		results = append(results, result)
		go func() { _, err := c.CallGeneration(ctx, old.generation, "ordinary-held-return", nil); result <- err }()
		request := historyReadMessage(t, NewCodec(f.peer, f.peer))
		var numeric int64
		if request.ID == nil || json.Unmarshal(*request.ID, &numeric) != nil {
			t.Fatal("ordinary call lost numeric ID")
		}
		select {
		case <-f.conn.written:
		case <-time.After(3 * time.Second):
			t.Fatal("ordinary Write-return barrier not reached")
		}
		retained = append(retained, old)
		if err := f.peer.Close(); err != nil {
			t.Fatal(err)
		}
		sessionEvent(t, c.Events(), EventDisconnected, uint64(i+1), "")
		select {
		case replacement <- fixtures[i+1].conn:
		case <-time.After(3 * time.Second):
			t.Fatal("retirement joined an old ordinary writer")
		}
		select {
		case <-fixtures[i+1].conn.readStarted:
		case <-time.After(3 * time.Second):
			t.Fatal("replacement Read not admitted")
		}
		if c.history.reader != reader || &c.history.raw[:1][0] != raw || &c.history.offsets[:1][0] != offsets {
			t.Fatal("replacement allocated a history reader/backing")
		}
		for _, previous := range retained {
			if previous.codec.scanner != nil {
				t.Fatal("history-only Codec gained ordinary scanner backing")
			}
			if _, err := previous.codec.Read(); !errors.Is(err, ErrStreamBroken) {
				t.Fatalf("retired direct Codec.Read = %v", err)
			}
			stale := make(chan error, 1)
			go func() { _, err := c.readMessage(previous); stale <- err }()
			if err := sessionResult(t, stale); !errors.Is(err, io.EOF) {
				t.Fatalf("stale read joined/touched replacement: %v", err)
			}
		}
	}
	last := fixtures[len(fixtures)-1]
	server := NewCodec(last.peer, last.peer)
	historyBarrier(t, c, server)
	for _, f := range fixtures[:len(fixtures)-1] {
		select {
		case <-f.conn.release:
			t.Fatal("old ordinary caller was released early")
		default:
		}
	}
	for i, f := range fixtures[:len(fixtures)-1] {
		f.release.Do(func() { close(f.conn.release) })
		if err := sessionResult(t, results[i]); err == nil {
			t.Fatal("retired ordinary call succeeded")
		}
	}
	historyBarrier(t, c, server)
	if c.Generation() != uint64(len(fixtures)) || c.history.reader != reader {
		t.Fatal("old caller completion changed replacement")
	}
}

type historyReadReturnPause struct {
	net.Conn
	entered, interrupted, release chan struct{}
	enterOnce, interruptOnce      sync.Once
}

func (r *historyReadReturnPause) Read(p []byte) (int, error) {
	r.enterOnce.Do(func() { close(r.entered) })
	n, err := r.Conn.Read(p)
	r.interruptOnce.Do(func() { close(r.interrupted) })
	<-r.release
	return n, err
}

func TestHistoryFactoryCloseJoinsActualPhysicalRead(t *testing.T) {
	a, _ := sessionPipe(t)
	r := &historyReadReturnPause{Conn: a, entered: make(chan struct{}), interrupted: make(chan struct{}), release: make(chan struct{})}
	var release sync.Once
	t.Cleanup(func() { release.Do(func() { close(r.release) }) })
	c := historyTestClient(t, r, a, func(uint64, bool, []byte) error { return nil })
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	run := make(chan error, 1)
	go func() { run <- c.Run(ctx) }()
	select {
	case <-r.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("physical Read not entered")
	}
	closed := make(chan struct{})
	go func() { c.Close(); close(closed) }()
	select {
	case <-r.interrupted:
	case <-time.After(3 * time.Second):
		t.Fatal("actual Close did not interrupt input")
	}
	if c.history.readMu.TryLock() {
		c.history.readMu.Unlock()
		t.Fatal("actual Read-return borrower was released early")
	}
	select {
	case <-closed:
		t.Fatal("close fence did not join actual Read return")
	default:
	}
	release.Do(func() { close(r.release) })
	select {
	case <-closed:
	case <-time.After(3 * time.Second):
		t.Fatal("read fence did not finish")
	}
	if err := sessionResult(t, run); !errors.Is(err, context.Canceled) {
		t.Fatalf("Run after Close = %v", err)
	}
}

type historyCloseProbe struct {
	net.Conn
	closed chan struct{}
	once   sync.Once
}

func (p *historyCloseProbe) Close() error {
	err := p.Conn.Close()
	p.once.Do(func() { close(p.closed) })
	return err
}

func TestHistoryFactoryCloseJoinsApplyBorrow(t *testing.T) {
	a, b := sessionPipe(t)
	conn := &historyCloseProbe{Conn: a, closed: make(chan struct{})}
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(release) }) })
	c := historyTestClient(t, conn, conn, func(uint64, bool, []byte) error { close(entered); <-release; return nil })
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	run := make(chan error, 1)
	go func() { run <- c.Run(ctx) }()
	s := NewCodec(b, b)
	request := historyRequest(t, c, s)
	if err := s.Write(&Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(`{"requests":[]}`)}); err != nil {
		t.Fatal(err)
	}
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("history apply not entered")
	}
	closed := make(chan struct{})
	go func() { c.Close(); close(closed) }()
	select {
	case <-conn.closed:
	case <-time.After(3 * time.Second):
		t.Fatal("actual transport Close not reached")
	}
	if c.history.readMu.TryLock() {
		c.history.readMu.Unlock()
		t.Fatal("apply loan was released before consumption")
	}
	select {
	case <-closed:
		t.Fatal("close fence did not join apply")
	default:
	}
	once.Do(func() { close(release) })
	select {
	case <-closed:
	case <-time.After(3 * time.Second):
		t.Fatal("apply fence did not finish")
	}
	if err := sessionResult(t, run); !errors.Is(err, context.Canceled) {
		t.Fatalf("Run after Close = %v", err)
	}
	if state := c.HistoryState(); state.Connected || state.Pending {
		t.Fatalf("closed apply left live history: %+v", state)
	}
}
