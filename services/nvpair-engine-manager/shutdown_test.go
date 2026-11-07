// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestStopAllStopsEveryEngine guards the concurrent StopAll: with more than one
// engine running, every engine must be stopped — a regression guard for the old
// sequential loop where a slow first engine could leave later engines running.
func TestStopAllStopsEveryEngine(t *testing.T) {
	reg := NewRegistry()
	names := []string{"fake-a", "fake-b"}
	for _, name := range names {
		m := testEngineManifest(fakeEngineBin)
		m.Engine = name
		reg.engines[name] = m
	}
	ex := NewExecutor(reg, NewReporter(nil), func(string, any) {}, t.TempDir())
	ctx := context.Background()

	ports := make([]int, len(names))
	for i, name := range names {
		if err := ex.Start(ctx, name); err != nil {
			t.Fatalf("start %s: %v", name, err)
		}
		st, _ := ex.Status(name)
		if !st.Running || st.Port == 0 {
			t.Fatalf("%s not running: %+v", name, st)
		}
		ports[i] = st.Port
	}

	ex.StopAll()

	for i, name := range names {
		if !waitPortClosed(ports[i], 5*time.Second) {
			t.Fatalf("engine %s still serving on port %d after StopAll", name, ports[i])
		}
	}
	if err := ex.Start(ctx, names[0]); !errors.Is(err, context.Canceled) {
		t.Fatalf("start after StopAll error = %v, want context canceled", err)
	}
}

func TestStopAllCancelsPendingStartWithoutError(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "fake.pid")
	t.Setenv("FAKE_PID_FILE", pidFile)

	manifest := testEngineManifest(fakeEngineBin)
	platform := manifest.Platforms[hostKey()]
	// The engine publishes its PID, then stays unready until StopAll.
	platform.Runtime.Env = map[string]string{"FAKE_ADDR": "127.0.0.1:{port}", "FAKE_START_DELAY": "1h"}
	platform.Runtime.Health = nil
	manifest.Platforms[hostKey()] = platform
	ex := newTestExecutor(t, manifest)

	started := make(chan error, 1)
	go func() { started <- ex.Start(context.Background(), "fake") }()

	var pid int
	deadline := time.Now().Add(5 * time.Second)
	for pid == 0 && time.Now().Before(deadline) {
		if data, err := os.ReadFile(pidFile); err == nil {
			pid, _ = strconv.Atoi(strings.TrimSpace(string(data)))
		}
		if pid == 0 {
			time.Sleep(25 * time.Millisecond)
		}
	}
	if pid == 0 {
		t.Fatal("fake engine did not publish its PID")
	}
	t.Cleanup(func() {
		if pidAlive(pid) {
			_ = signalPID(pid, true)
		}
	})

	ex.StopAll()

	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("start did not return after StopAll")
	}
	if pidAlive(pid) {
		t.Fatalf("fake engine PID %d survived StopAll", pid)
	}
	if hasErr(ex.Errors(), startFailedID("fake")) {
		t.Fatalf("shutdown cancellation retained a start-failed error: %+v", ex.Errors())
	}
}

// runCommandTestHelper is dispatched by TestMain before building the usual
// fixtures. A start request can launch a daemon and hold the CLI open until
// cancellation; a separate stop request must still terminate that daemon.
func runCommandTestHelper(url string) int {
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 1
	}
	return 0
}

// TestE2EStdinCloseStopsEngine drives the real engine-manager binary and stops
// it the way the broker now does on shutdown: by closing stdin (EOF), with no
// prepare-shutdown or shutdown RPC. The EOF path must run StopAll and take the
// engine's port down — this is the backstop the broker relies on now that it no
// longer force-kills engine-manager on a timeout.
func TestE2EStdinCloseStopsEngine(t *testing.T) {
	cfg := t.TempDir()
	home := t.TempDir()
	for _, dir := range []string{
		filepath.Join(cfg, "Nvidia Corporation", "Personal AI Router", "engines"),
		filepath.Join(home, "Library", "Application Support", "Nvidia Corporation", "Personal AI Router", "engines"),
	} {
		writeFakeManifest(t, dir)
	}

	m := startE2EManager(t, cfg, home)
	send(t, m.stdin, 1, "engine:install", map[string]any{"engine": "fake"})
	var started EngineStatus
	if err := json.Unmarshal(waitResult(t, m.frames, "1", 20*time.Second), &started); err != nil || !started.Running {
		t.Fatalf("start status=%+v err=%v", started, err)
	}

	// stop() closes stdin and waits for the process to exit on its own.
	m.stop(t)

	if !waitPortClosed(started.Port, 5*time.Second) {
		t.Fatalf("engine port %d remained open after stdin close", started.Port)
	}
}

// waitPortClosed polls until nothing is serving on the port or the timeout
// elapses. The process is gone by the time StopAll returns, but the listener
// socket can take a beat to be reclaimed by the OS.
func waitPortClosed(port int, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if !portServing(port) {
			return true
		}
		time.Sleep(100 * time.Millisecond)
	}
	return !portServing(port)
}
