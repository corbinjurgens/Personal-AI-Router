// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"os"
	"syscall"
	"testing"
)

func TestMemoryPlanProcessUnixTrees(t *testing.T) { testMemoryPlanProcessTrees(t) }

func assertMemoryPlanProcessStopped(t *testing.T, pid int) {
	t.Helper()
	if err := syscall.Kill(pid, 0); err != syscall.ESRCH {
		t.Fatalf("descendant %d survives: %v", pid, err)
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
