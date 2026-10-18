// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build unix

package main

import (
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

func TestWriteFileAtomicExactModeUnderRestrictiveUmask(t *testing.T) {
	path := filepath.Join(t.TempDir(), "entry.desktop")
	previous := syscall.Umask(0o077)
	defer syscall.Umask(previous)
	want := []byte("[Desktop Entry]\n")
	if err := writeFileAtomic(path, want, 0o644); err != nil {
		t.Fatal(err)
	}
	assertAutostartFile(t, path, want)
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0o644 {
		t.Fatalf("entry mode = %#o under umask 077, want 0644", got)
	}
}
