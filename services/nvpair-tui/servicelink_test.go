// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"nvpair-shared/ipc"
	"nvpair-shared/servicectl"
)

// fakeServiceEnv makes the test binary stand in for nvpair-service, so the TUI's
// attach, detach, and stop paths run against a real detached process without
// the service or broker being built.
const fakeServiceEnv = "NVPAIR_TUI_FAKE_SERVICE"

// fakeLogLine is the broker stderr line the fake relays as service/log.
const fakeLogLine = "[nvpair-ui-broker] INFO fake broker line"

func TestMain(m *testing.M) {
	if os.Getenv(fakeServiceEnv) == "1" {
		runFakeService()
		return
	}
	os.Exit(m.Run())
}

// runFakeService listens on the service endpoint and, for each client, sends
// app:ready and one service/log line, answers ping, and exits on service/stop
// after answering it, as the real service does once its broker has stopped.
func runFakeService() {
	endpoint, err := servicectl.Endpoint()
	if err != nil {
		os.Exit(2)
	}
	l, err := ipc.ListenPrivate(endpoint)
	if err != nil {
		os.Exit(3)
	}
	// Self-terminating, so a failed test cannot leave it behind.
	go func() {
		time.Sleep(30 * time.Second)
		_ = l.Close()
		os.Exit(4)
	}()
	var stopOnce sync.Once
	for {
		conn, err := l.Accept()
		if err != nil {
			return
		}
		go func(c net.Conn) {
			defer c.Close()
			fmt.Fprintln(c, `{"jsonrpc":"2.0","method":"app:ready","params":{"version":"fake"}}`)
			params, _ := json.Marshal(map[string]string{"source": "nvpair-ui-broker", "stream": "stderr", "text": fakeLogLine})
			fmt.Fprintf(c, `{"jsonrpc":"2.0","method":"service/log","params":%s}`+"\n", params)
			sc := bufio.NewScanner(c)
			for sc.Scan() {
				var m struct {
					ID     json.RawMessage `json:"id"`
					Method string          `json:"method"`
				}
				if json.Unmarshal(sc.Bytes(), &m) != nil || m.ID == nil {
					continue
				}
				switch m.Method {
				case "service/stop":
					fmt.Fprintf(c, `{"jsonrpc":"2.0","id":%s,"result":{"stopped":true}}`+"\n", m.ID)
					stopOnce.Do(func() {
						_ = l.Close()
						os.Exit(0)
					})
				default:
					fmt.Fprintf(c, `{"jsonrpc":"2.0","id":%s,"result":{}}`+"\n", m.ID)
				}
			}
		}(conn)
	}
}

// useFakeService points the service endpoint at a private socket and makes a
// started "service" the fake.
func useFakeService(t *testing.T) string {
	t.Helper()
	// Not t.TempDir: its test-named path can overrun the Unix socket limit.
	dir, err := os.MkdirTemp("", "tui")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	endpoint := filepath.Join(dir, "s.sock")
	t.Setenv(servicectl.EndpointEnv, endpoint)
	t.Setenv(fakeServiceEnv, "1")
	return endpoint
}

func answering() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	conn, err := servicectl.Dial(ctx)
	if err != nil {
		return false
	}
	_ = conn.Close()
	return true
}

func waitReady(t *testing.T, link *serviceLink) {
	t.Helper()
	select {
	case msg, ok := <-link.Client.Notifications():
		if !ok {
			t.Fatal("notifications closed before app:ready")
		}
		if msg.Method != "app:ready" {
			t.Fatalf("first notification = %s, want app:ready (service/log must go to Logs)", msg.Method)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for app:ready")
	}
}

func TestResolveServicePathOverride(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "fake-service")
	if err := os.WriteFile(bin, []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := resolveServicePath(bin)
	if err != nil {
		t.Fatalf("override: %v", err)
	}
	if got != bin {
		t.Fatalf("got %q want %q", got, bin)
	}
	if _, err := resolveServicePath(filepath.Join(dir, "missing")); err == nil {
		t.Fatal("expected error for missing override")
	}
}

// TestQuitDetachesAndLeavesTheServiceRunning pins the central promise: the TUI
// starts the service when needed, quitting only detaches, and the next TUI
// attaches to the same running service rather than starting another.
func TestQuitDetachesAndLeavesTheServiceRunning(t *testing.T) {
	useFakeService(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	link, err := connectService(ctx, os.Args[0])
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	waitReady(t, link)

	// Broker stderr arrives on Logs, one line per service/log.
	lines := make(chan string, 1)
	go func() {
		sc := bufio.NewScanner(link.Logs)
		if sc.Scan() {
			lines <- sc.Text()
		}
	}()
	select {
	case got := <-lines:
		if got != fakeLogLine {
			t.Fatalf("log line = %q", got)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no log line relayed")
	}

	link.detach()
	time.Sleep(100 * time.Millisecond)
	if !answering() {
		t.Fatal("service stopped when the TUI detached")
	}

	// A second session attaches to the same service: the bogus path proves
	// nothing was started.
	again, err := connectService(ctx, "/nonexistent/nvpair-service")
	if err != nil {
		t.Fatalf("reattach: %v", err)
	}
	waitReady(t, again)
	if err := again.stopService(); err != nil {
		t.Fatalf("stop: %v", err)
	}
	again.detach()
	if answering() {
		t.Fatal("service still answering after stopService")
	}
}

func TestStopRunningService(t *testing.T) {
	useFakeService(t)
	running, err := stopRunningService()
	if err != nil || running {
		t.Fatalf("with no service: running=%v err=%v", running, err)
	}

	if err := servicectl.StartDetached(os.Args[0], nil); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for !answering() {
		if time.Now().After(deadline) {
			t.Fatal("fake service never started")
		}
		time.Sleep(50 * time.Millisecond)
	}
	running, err = stopRunningService()
	if err != nil || !running {
		t.Fatalf("stop: running=%v err=%v", running, err)
	}
	if answering() {
		t.Fatal("service still answering after --stop-service")
	}
}
