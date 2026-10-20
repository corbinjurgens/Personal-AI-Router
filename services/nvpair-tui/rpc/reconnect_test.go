// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package rpc

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
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

func TestReconnectLegacyAdapterDoesNotLeaveEventsUndrained(t *testing.T) {
	conn, peer := sessionPipe(t)
	client := NewReconnectingClient(conn, func(ctx context.Context) (io.ReadWriteCloser, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	})
	t.Cleanup(client.Close)
	notifications := client.Notifications()
	go func() { _ = client.Run(context.Background()) }()
	consumed := make(chan error, 1)
	go func() {
		for i := 0; i < 600; i++ {
			msg, open := <-notifications
			if !open || msg.Method != "burst:push" {
				consumed <- errors.New("legacy adapter lost burst notification")
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

type sessionObservedReader struct {
	net.Conn
	started chan struct{}
	once    sync.Once
}

func (r *sessionObservedReader) Read(p []byte) (int, error) {
	r.once.Do(func() { close(r.started) })
	return r.Conn.Read(p)
}
