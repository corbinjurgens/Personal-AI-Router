// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package servicectl

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"nvpair-shared/ipc"
	"nvpair-shared/secretenv/secretenvtest"
)

// fakeServiceEnv makes the test binary act as a minimal service: it listens on
// Endpoint(), greets each connection with its argv, and exits on "quit".
const fakeServiceEnv = "NVPAIR_SERVICECTL_FAKE"

func TestMain(m *testing.M) {
	secretenvtest.RunChild()
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
		if err := json.NewEncoder(c).Encode(os.Args[1:]); err != nil {
			_ = c.Close()
			continue
		}
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

	var args []string
	if err := json.NewDecoder(conn).Decode(&args); err != nil {
		t.Fatalf("read greeting: %v", err)
	}
	if want := []string{"--", "--log-level", "debug"}; !slices.Equal(args, want) {
		t.Fatalf("service argv = %q, want %q", args, want)
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

func TestStartDetachedArgsPreservesServiceArguments(t *testing.T) {
	t.Setenv(EndpointEnv, filepath.Join(shortTempDir(t), "s.sock"))
	t.Setenv(fakeServiceEnv, "1")
	args := []string{"--log-level", "debug", "--", "", "space value", `say "hi"`, `dir\ x\`}
	if err := StartDetachedArgs(os.Args[0], args); err != nil {
		t.Fatalf("StartDetachedArgs: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	for {
		conn, err := Dial(ctx)
		if err == nil {
			defer conn.Close()
			var got []string
			if err := json.NewDecoder(conn).Decode(&got); err != nil {
				t.Fatalf("read greeting: %v", err)
			}
			if !slices.Equal(got, args) {
				t.Errorf("service argv = %q, want %q", got, args)
			}
			if _, err := conn.Write([]byte("quit\n")); err != nil {
				t.Fatal(err)
			}
			return
		}
		select {
		case <-ctx.Done():
			t.Fatalf("detached service did not answer: %v", err)
		case <-time.After(dialRetryInterval):
		}
	}
}

// A client-to-service launch is a trusted carrier: the service owns the broker,
// which hands configured credential references to the proxy, so the detached
// service keeps the whole provider-secret namespace.
func TestStartDetachedKeepsProviderSecrets(t *testing.T) {
	report := secretenvtest.Prepare(t)
	if err := StartDetached(os.Args[0], []string{"--log-level", "debug"}); err != nil {
		t.Fatalf("StartDetached: %v", err)
	}
	secretenvtest.AssertRetained(t, secretenvtest.Wait(t, report))
}

func TestStartDetachedArgsKeepsProviderSecrets(t *testing.T) {
	report := secretenvtest.Prepare(t)
	if err := StartDetachedArgs(os.Args[0], []string{"--log-level", "debug"}); err != nil {
		t.Fatalf("StartDetachedArgs: %v", err)
	}
	secretenvtest.AssertRetained(t, secretenvtest.Wait(t, report))
}

func TestStartDetachedArgsFailureDoesNotExposeProviderSecrets(t *testing.T) {
	secretenvtest.Prepare(t)
	err := StartDetachedArgs(filepath.Join(shortTempDir(t), "missing-service"), []string{"--log-level", "debug"})
	if err == nil {
		t.Fatal("expected an error for a missing binary")
	}
	secretenvtest.AssertNoValue(t, "start error", err.Error())
}

func TestStartDetachedFailureDoesNotExposeProviderSecrets(t *testing.T) {
	secretenvtest.Prepare(t)
	t.Setenv(EndpointEnv, filepath.Join(shortTempDir(t), "s.sock"))
	_, err := ConnectOrStart(context.Background(), "/nonexistent/nvpair-service", []string{"--log-level", "debug"})
	if err == nil {
		t.Fatal("expected an error for a missing binary")
	}
	secretenvtest.AssertNoValue(t, "start error", err.Error())
}
