// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package servicectl

import (
	"bufio"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"nvpair-shared/ipc"
)

// fakeServiceEnv makes the test binary act as a minimal service: it listens on
// Endpoint(), greets each connection with its argv, and exits on "quit".
const fakeServiceEnv = "NVPAIR_SERVICECTL_FAKE"

func TestMain(m *testing.M) {
	if os.Getenv(fakeServiceEnv) == "1" {
		runFakeService()
		return
	}
	os.Exit(m.Run())
}

func runFakeService() {
	endpoint, err := Endpoint()
	if err != nil {
		os.Exit(2)
	}
	l, err := ipc.ListenPrivate(endpoint)
	if err != nil {
		os.Exit(3)
	}
	defer os.Remove(endpoint)
	// Self-terminating, so a failed test cannot leave it behind.
	go func() {
		time.Sleep(20 * time.Second)
		os.Remove(endpoint)
		os.Exit(4)
	}()
	for {
		c, err := l.Accept()
		if err != nil {
			return
		}
		_, _ = c.Write([]byte(strings.Join(os.Args[1:], " ") + "\n"))
		sc := bufio.NewScanner(c)
		for sc.Scan() {
			if sc.Text() == "quit" {
				_ = c.Close()
				_ = l.Close()
				return
			}
		}
		_ = c.Close()
	}
}

// shortTempDir avoids t.TempDir, whose test-named path can overrun the Unix
// socket path limit.
func shortTempDir(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "sctl")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	return dir
}

func TestEndpointDefaultsToAppDirSocket(t *testing.T) {
	t.Setenv(EndpointEnv, "")
	got, err := Endpoint()
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(got) != "service.sock" {
		t.Fatalf("endpoint = %q, want <appdir>/service.sock", got)
	}
}

func TestEndpointEnvOverride(t *testing.T) {
	t.Setenv(EndpointEnv, "/tmp/x.sock")
	got, err := Endpoint()
	if err != nil || got != "/tmp/x.sock" {
		t.Fatalf("endpoint = %q, %v", got, err)
	}
}

func TestDialReportsNotRunning(t *testing.T) {
	t.Setenv(EndpointEnv, filepath.Join(shortTempDir(t), "none.sock"))
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, err := Dial(ctx)
	if !errors.Is(err, ErrNotRunning) {
		t.Fatalf("Dial err = %v, want ErrNotRunning", err)
	}
}

func TestConnectOrStartStartsDetachedServiceAndPassesBrokerArgs(t *testing.T) {
	endpoint := filepath.Join(shortTempDir(t), "s.sock")
	t.Setenv(EndpointEnv, endpoint)
	t.Setenv(fakeServiceEnv, "1")

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	conn, err := ConnectOrStart(ctx, os.Args[0], []string{"--log-level", "debug"})
	if err != nil {
		t.Fatalf("ConnectOrStart: %v", err)
	}
	defer conn.Close()

	line, err := bufio.NewReader(conn).ReadString('\n')
	if err != nil {
		t.Fatalf("read greeting: %v", err)
	}
	if got := strings.TrimSpace(line); got != "-- --log-level debug" {
		t.Fatalf("service argv = %q, want broker args after --", got)
	}

	// A second ConnectOrStart finds the running service rather than starting
	// another one.
	conn2, err := ConnectOrStart(ctx, "/nonexistent/binary", nil)
	if err != nil {
		t.Fatalf("second ConnectOrStart started instead of dialing: %v", err)
	}
	_ = conn2.Close()

	if _, err := conn.Write([]byte("quit\n")); err != nil {
		t.Fatal(err)
	}
}

func TestConnectOrStartReportsUnstartableBinary(t *testing.T) {
	t.Setenv(EndpointEnv, filepath.Join(shortTempDir(t), "s.sock"))
	_, err := ConnectOrStart(context.Background(), "/nonexistent/nvpair-service", nil)
	if err == nil {
		t.Fatal("expected an error for a missing binary")
	}
}
