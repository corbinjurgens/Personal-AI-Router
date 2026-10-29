// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"context"
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
