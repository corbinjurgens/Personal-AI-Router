// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-tui/rpc"
)

func linkEvent(t *testing.T, events <-chan rpc.Event, kind rpc.EventKind, generation uint64) {
	t.Helper()
	select {
	case event, open := <-events:
		if !open || event.Kind != kind || event.Generation != generation {
			t.Fatalf("event = %+v (open %v), want kind %d generation %d", event, open, kind, generation)
		}
		if kind == rpc.EventNotification && (event.Message == nil || event.Message.Method != "app:ready") {
			t.Fatalf("notification = %+v, want app:ready; service/log must be filtered", event.Message)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for service-link event")
	}
}

func TestServiceLinkReconnectKeepsLogsAndFilter(t *testing.T) {
	first, peer1 := net.Pipe()
	second, peer2 := net.Pipe()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(func() { cancel(); _ = first.Close(); _ = peer1.Close(); _ = second.Close(); _ = peer2.Close() })
	var dials atomic.Int32
	client := rpc.NewReconnectingClient(first, func(context.Context) (io.ReadWriteCloser, error) {
		dials.Add(1)
		return second, nil
	})
	events := client.Events()
	link := startServiceLink(ctx, client)
	t.Cleanup(link.detach)
	lines := make(chan string, 2)
	logsDone := make(chan struct{})
	go func() {
		defer close(logsDone)
		scanner := bufio.NewScanner(link.Logs)
		for scanner.Scan() {
			lines <- scanner.Text()
		}
	}()
	send := func(peer net.Conn, text string) {
		t.Helper()
		if err := peer.SetWriteDeadline(time.Now().Add(3 * time.Second)); err != nil {
			t.Fatal(err)
		}
		server := rpc.NewCodec(peer, peer)
		params, err := json.Marshal(map[string]string{"text": text})
		if err != nil {
			t.Fatal(err)
		}
		if err := server.Write(&rpc.Message{JSONRPC: "2.0", Method: "service/log", Params: params}); err != nil {
			t.Fatal(err)
		}
		if err := server.Write(&rpc.Message{JSONRPC: "2.0", Method: "app:ready"}); err != nil {
			t.Fatal(err)
		}
	}
	readLine := func(want string) {
		t.Helper()
		select {
		case got := <-lines:
			if got != want {
				t.Fatalf("log = %q, want %q", got, want)
			}
		case <-time.After(3 * time.Second):
			t.Fatalf("missing log %q", want)
		}
	}
	send(peer1, "before-restart")
	linkEvent(t, events, rpc.EventNotification, 1)
	readLine("before-restart")
	if err := peer1.Close(); err != nil {
		t.Fatal(err)
	}
	linkEvent(t, events, rpc.EventDisconnected, 1)
	send(peer2, "after-restart")
	linkEvent(t, events, rpc.EventNotification, 2)
	readLine("after-restart")
	if link.Client != client || dials.Load() != 1 {
		t.Fatalf("stable client %v, dial count %d", link.Client == client, dials.Load())
	}
	link.detach()
	link.detach()
	select {
	case <-logsDone:
	case <-time.After(3 * time.Second):
		t.Fatal("detach did not close log stream")
	}
}

func TestServiceLinkSessionCancellationClosesLogs(t *testing.T) {
	conn, peer := net.Pipe()
	t.Cleanup(func() { _ = peer.Close() })
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	link := newServiceLink(ctx, conn)
	t.Cleanup(link.detach)
	if err := peer.SetWriteDeadline(time.Now().Add(3 * time.Second)); err != nil {
		t.Fatal(err)
	}
	server := rpc.NewCodec(peer, peer)
	if err := server.Write(&rpc.Message{JSONRPC: "2.0", Method: "service/log", Params: json.RawMessage(`{"text":"blocked-log"}`)}); err != nil {
		t.Fatal(err)
	}
	cancel()
	done := make(chan error, 1)
	go func() { _, err := io.Copy(io.Discard, link.Logs); done <- err }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("closed logs: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("cancellation left log writer blocked")
	}
}
