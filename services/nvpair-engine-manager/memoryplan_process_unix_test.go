// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestMemoryPlanProcessUnixTrees(t *testing.T) { testMemoryPlanProcessTrees(t) }

func TestMemoryPlanProcessUnixCleanup(t *testing.T) {
	type step struct {
		signal syscall.Signal
		err    error
	}
	test := func(name string, steps []step) {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			calls := 0
			err := cleanupMemoryPlanProcessGroup(ctx, 123, func(pid int, signal syscall.Signal) error {
				if calls >= len(steps) {
					t.Fatal("cleanup continued after confirmed group absence")
				}
				want := steps[calls]
				calls++
				if pid != -123 || signal != want.signal {
					t.Fatalf("signal call %d: pid=%d signal=%v, want pid=-123 signal=%v", calls, pid, signal, want.signal)
				}
				return want.err
			})
			if err != nil || calls != len(steps) {
				t.Fatalf("cleanup: err=%v calls=%d, want nil and %d", err, calls, len(steps))
			}
		})
	}
	test("exiting group rejects kill before disappearing", []step{
		{syscall.SIGKILL, syscall.EPERM},
		{0, syscall.EPERM},
		{syscall.SIGKILL, nil},
		{0, syscall.ESRCH},
	})
	test("exiting group rejects probe before disappearing", []step{
		{syscall.SIGKILL, nil},
		{0, syscall.EPERM},
		{0, syscall.ESRCH},
	})
	test("absent group", []step{{syscall.SIGKILL, syscall.ESRCH}})
	for _, denied := range []bool{false, true} {
		name := "live group is unconfirmed"
		if denied {
			name = "permission denied is unconfirmed"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			err := cleanupMemoryPlanProcessGroup(ctx, 123, func(pid int, signal syscall.Signal) error {
				if pid != -123 {
					t.Fatalf("cleanup signalled pid=%d, want group -123", pid)
				}
				cancel()
				if denied {
					return syscall.EPERM
				}
				return nil
			})
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("cleanup without absence proof: %v, want cancellation", err)
			}
		})
	}
	t.Run("unexpected signal error", func(t *testing.T) {
		if err := cleanupMemoryPlanProcessGroup(context.Background(), 123, func(int, syscall.Signal) error { return syscall.EIO }); !errors.Is(err, syscall.EIO) {
			t.Fatalf("cleanup signal error: %v, want EIO", err)
		}
	})
}

func assertMemoryPlanProcessStopped(t *testing.T, pid int) {
	t.Helper()
	// Orphan reaping is asynchronous on Unix: the group can be gone while
	// signal 0 still sees an exiting PID. Synchronize on disappearance, not a
	// fixed delay, and still require ESRCH (not merely a zombie or no image).
	if err := syscall.Kill(pid, 0); err == syscall.ESRCH {
		return
	} else if err != nil {
		t.Fatalf("inspect descendant %d: %v", pid, err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), memoryPlanCleanupBudget)
	defer cancel()
	state, err := exec.CommandContext(ctx, "ps", "-o", "stat=", "-p", strconv.Itoa(pid)).Output()
	// It may have disappeared during inspection. Otherwise require a terminal
	// state before waiting for reaping: never grant a live survivor extra time.
	fields := strings.Fields(string(state))
	if (err != nil || len(fields) == 0) && syscall.Kill(pid, 0) == syscall.ESRCH {
		return
	}
	if err != nil || len(fields) != 1 || !(strings.HasPrefix(fields[0], "Z") || runtime.GOOS == "darwin" && strings.Contains(fields[0], "E")) {
		t.Fatalf("descendant %d still live or uninspectable: state=%q error=%v", pid, state, err)
	}
	deadline := time.NewTimer(memoryPlanCleanupBudget)
	defer deadline.Stop()
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		err := syscall.Kill(pid, 0)
		if err == syscall.ESRCH {
			return
		}
		if err != nil {
			t.Fatalf("inspect descendant %d: %v", pid, err)
		}
		select {
		case <-deadline.C:
			t.Fatalf("descendant %d survives", pid)
		case <-tick.C:
		}
	}
}

func TestMemoryPlanProcessUnixPrivateDirectory(t *testing.T) {
	dir, err := makeMemoryPlanTempDir()
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	info, err := os.Stat(dir)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0700 {
		t.Fatalf("directory permissions: %v", info.Mode())
	}
}
