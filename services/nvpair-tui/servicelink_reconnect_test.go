// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/ipc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/servicectl"
	"nvpair-tui/rpc"
)

func TestServiceLinkProductionHistoryReconnect(t *testing.T) {
	// A private endpoint and an invalid executable path prove this attaches to
	// the fixture only, never starts or contacts the installed service.
	dir, err := os.MkdirTemp("", "tui-history")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	endpoint := filepath.Join(dir, "s.sock")
	if runtime.GOOS == "windows" {
		endpoint = `\\.\pipe\` + filepath.Base(dir)
	}
	t.Setenv(servicectl.EndpointEnv, endpoint)
	listener, err := ipc.ListenPrivate(endpoint)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = listener.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	t.Cleanup(cancel)
	stop := context.AfterFunc(ctx, func() { _ = listener.Close() })
	t.Cleanup(func() { stop() })
	type accepted struct {
		peer net.Conn
		err  error
	}
	acceptNext := func() <-chan accepted {
		result := make(chan accepted, 1)
		done := make(chan struct{})
		go func() {
			defer close(done)
			peer, err := listener.Accept()
			result <- accepted{peer: peer, err: err}
		}()
		t.Cleanup(func() {
			_ = listener.Close()
			select {
			case <-done:
			case <-time.After(3 * time.Second):
				t.Error("fixture accept did not join after listener close")
			}
			select {
			case orphan := <-result:
				if orphan.peer != nil {
					if err := orphan.peer.Close(); err != nil {
						t.Errorf("close unclaimed fixture connection: %v", err)
					}
				}
			default:
			}
		})
		return result
	}
	// Windows creates a connectable named-pipe instance from Accept, unlike a
	// Unix listening socket. Own that bounded accept before the initial dial.
	pending := acceptNext()
	link, err := connectService(ctx, filepath.Join(dir, "must-not-start"))
	if err != nil {
		t.Fatal(err)
	}
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
	for generation := uint64(1); generation <= 2; generation++ {
		var connection accepted
		select {
		case connection = <-pending:
		case <-ctx.Done():
			t.Fatal("fixture accept exceeded the connection budget")
		}
		if connection.err != nil {
			t.Fatal(connection.err)
		}
		peer := connection.peer
		t.Cleanup(func() { _ = peer.Close() })
		if err := peer.SetDeadline(time.Now().Add(10 * time.Second)); err != nil {
			t.Fatal(err)
		}
		server := rpc.NewCodec(peer, peer)
		request, err := server.Read()
		if err != nil {
			t.Fatal(err)
		}
		if request.Method != nodepolicy.MethodRoutingSubscribe || request.ID == nil || string(*request.ID) != `"jh:1"` {
			t.Fatalf("generation %d first request = %+v, want generated subscribe", generation, request)
		}
		if err := server.Write(&rpc.Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(`{"requests":[]}`)}); err != nil {
			t.Fatal(err)
		}
		if err := server.Write(&rpc.Message{JSONRPC: "2.0", Method: "service/log", Params: json.RawMessage(`{"text":"history-link-log"}`)}); err != nil {
			t.Fatal(err)
		}
		if err := server.Write(&rpc.Message{JSONRPC: "2.0", Method: "app:ready"}); err != nil {
			t.Fatal(err)
		}
		linkEvent(t, link.Client.Events(), rpc.EventNotification, generation)
		if state := link.Client.HistoryState(); state.Generation != generation || state.Loading || state.Pending || state.Quarantined {
			t.Fatalf("generation %d baseline not consumed: %+v", generation, state)
		}
		select {
		case line := <-lines:
			if line != "history-link-log" {
				t.Fatalf("log = %q", line)
			}
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		}
		result := make(chan error, 1)
		go func() {
			response, err := link.Client.Call(ctx, "ping", nil)
			if err == nil && (response == nil || string(response.Result) != `{"ok":true}`) {
				err = errors.New("ordinary result changed")
			}
			result <- err
		}()
		ordinary, err := server.Read()
		if err != nil {
			t.Fatal(err)
		}
		if ordinary.ID == nil || ordinary.Method != "ping" || string(*ordinary.ID) != strconv.FormatUint(generation, 10) {
			t.Fatalf("ordinary request = %+v", ordinary)
		}
		if err := server.Write(&rpc.Message{JSONRPC: "2.0", ID: ordinary.ID, Result: json.RawMessage(`{"ok":true}`)}); err != nil {
			t.Fatal(err)
		}
		if err := <-result; err != nil {
			t.Fatal(err)
		}
		if generation == 1 {
			pending = acceptNext()
			if err := peer.Close(); err != nil {
				t.Fatal(err)
			}
			linkEvent(t, link.Client.Events(), rpc.EventDisconnected, generation)
		} else {
			link.detach()
			if request, err := server.Read(); err == nil {
				t.Fatalf("detach sent a request instead of closing: %+v", request)
			}
		}
	}
	select {
	case <-logsDone:
	case <-ctx.Done():
		t.Fatal("detach did not close logs")
	}
}

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
