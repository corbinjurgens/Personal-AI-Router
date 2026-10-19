// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestFixedPath(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "fixed")
	module := filepath.Join(base, "checkout", "services", "shared")
	worktree := filepath.Join(base, "worktree", "services", "shared")
	outside := filepath.Join(base, "other", "shared")
	nestedModule := filepath.Join(module, "nested")
	for _, dir := range []string{module, worktree, outside, nestedModule} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module fixture\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	suffix := ".test"
	if runtime.GOOS == "windows" {
		suffix += ".exe"
	}
	for _, tc := range []struct{ name, root, cwd, binary, want string }{
		{"root package", root, module, "shared.test.exe", "shared"},
		{"nested package", root, filepath.Join(module, "testbin", "cmd"), "cmd.test", "shared-testbin-cmd"},
		{"other worktree", root, filepath.Join(worktree, "testbin"), "testbin.test.exe", "shared-testbin"},
		{"outside services", root, outside, "shared.test.exe", ""},
		{"nearest nested module", root, nestedModule, "nested.test.exe", ""},
		{"non-test executable", root, module, "main.exe", ""},
		{"unset root", "", module, "shared.test.exe", ""},
		{"relative root", "relative", module, "shared.test.exe", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := ""
			if tc.want != "" {
				want = filepath.Join(root, "go-test", "shared", tc.want+suffix)
			}
			if got := fixedPath(tc.root, tc.cwd, tc.binary); got != want {
				t.Fatalf("fixedPath = %q, want %q", got, want)
			}
		})
	}
}

func TestLockSerializesDestination(t *testing.T) {
	path := filepath.Join(t.TempDir(), "binary.lock")
	release, err := acquireLock(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	acquired := make(chan func(), 1)
	failed := make(chan error, 1)
	go func() {
		unlock, err := acquireLock(ctx, path)
		if err != nil {
			failed <- err
			return
		}
		acquired <- unlock
	}()
	select {
	case unlock := <-acquired:
		unlock()
		t.Fatal("second owner acquired held lock")
	case err := <-failed:
		t.Fatal(err)
	case <-time.After(200 * time.Millisecond):
	}
	release()
	select {
	case unlock := <-acquired:
		unlock()
	case err := <-failed:
		t.Fatal(err)
	case <-time.After(5 * time.Second):
		t.Fatal("second owner did not acquire released lock")
	}
}

func TestLockCancellation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "binary.lock")
	release, err := acquireLock(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := acquireLock(ctx, path); !errors.Is(err, context.Canceled) {
		t.Fatalf("lock error = %v, want cancellation", err)
	}
}

func TestRunPropagatesExitCode(t *testing.T) {
	if os.Getenv("NVPAIR_EXEC_EXIT_HELPER") == "1" {
		os.Exit(37)
	}
	t.Setenv("NVPAIR_EXEC_EXIT_HELPER", "1")
	t.Setenv("NVPAIR_TEST_BIN_DIR", "")
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	if got := run(context.Background(), []string{binary, "-test.run=^TestRunPropagatesExitCode$"}); got != 37 {
		t.Fatalf("exit code = %d, want 37", got)
	}
}

func TestCopyReplacesBinary(t *testing.T) {
	dir := t.TempDir()
	source, destination := filepath.Join(dir, "source"), filepath.Join(dir, "destination")
	for path, data := range map[string]string{source: "new binary", destination: "old binary"} {
		if err := os.WriteFile(path, []byte(data), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	if err := copyBinary(context.Background(), source, destination); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(destination)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "new binary" {
		t.Fatalf("destination = %q, want new binary", data)
	}
}

func TestRunCancellationStopsChild(t *testing.T) {
	if ready := os.Getenv("NVPAIR_EXEC_CANCEL_HELPER"); ready != "" {
		if err := os.WriteFile(ready, []byte("ready"), 0o600); err != nil {
			t.Fatal(err)
		}
		time.Sleep(time.Hour)
		return
	}
	ready := filepath.Join(t.TempDir(), "ready")
	t.Setenv("NVPAIR_EXEC_CANCEL_HELPER", ready)
	t.Setenv("NVPAIR_TEST_BIN_DIR", "")
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	finished := make(chan int, 1)
	go func() {
		finished <- run(ctx, []string{binary, "-test.run=^TestRunCancellationStopsChild$"})
	}()
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(ready); err == nil {
			break
		} else if !os.IsNotExist(err) {
			t.Fatal(err)
		}
		if time.Now().After(deadline) {
			t.Fatal("child did not become ready")
		}
		time.Sleep(10 * time.Millisecond)
	}
	cancel()
	select {
	case code := <-finished:
		if code == 0 {
			t.Fatal("cancelled child returned success")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("cancelled child did not exit")
	}
}
