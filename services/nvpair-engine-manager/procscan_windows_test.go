// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// procImage matches the Unix inspection helper for test-child cleanup after
// the child releases its listening port.
func procImage(pid int) string { return imagePathForPID(pid) }

func TestListProcessNamesIncludesSelf(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatalf("resolve test executable: %v", err)
	}
	names, err := listProcessNames(context.Background())
	if err != nil {
		t.Fatalf("listProcessNames: %v", err)
	}
	if len(names) == 0 {
		t.Fatal("listProcessNames returned no names")
	}
	if !anyProcessMatches(names, []string{filepath.Base(executable)}) {
		t.Fatalf("process list does not contain test executable %q", filepath.Base(executable))
	}
}
