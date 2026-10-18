// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"golang.org/x/sys/windows"
)

func TestWriteFileAtomicWindowsOpenReader(t *testing.T) {
	path := filepath.Join(t.TempDir(), "entry.desktop")
	old := []byte("original\n")
	if err := os.WriteFile(path, old, 0o644); err != nil {
		t.Fatal(err)
	}
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if reader != nil {
			if err := reader.Close(); err != nil {
				t.Error(err)
			}
		}
	})
	saved := autostartFileReplacer
	t.Cleanup(func() { autostartFileReplacer = saved })
	sleeps := 0
	autostartFileReplacer.Sleep = func(d time.Duration) {
		sleeps++
		if want := time.Duration(sleeps) * 50 * time.Millisecond; d != want {
			t.Errorf("retry delay = %s, want %s", d, want)
		}
	}
	want := []byte("replacement\n")
	err = writeFileAtomic(path, want, 0o644)
	var errno syscall.Errno
	if !errors.As(err, &errno) || (errno != windows.ERROR_SHARING_VIOLATION && errno != windows.ERROR_ACCESS_DENIED) {
		t.Fatalf("open-reader error = %v, want Windows sharing conflict", err)
	}
	if sleeps != 9 {
		t.Fatalf("retry sleeps = %d, want 9 before exhausting 10 attempts", sleeps)
	}
	assertAutostartFile(t, path, old)
	if err := reader.Close(); err != nil {
		t.Fatal(err)
	}
	reader = nil
	if err := writeFileAtomic(path, want, 0o644); err != nil {
		t.Fatalf("replacement after reader closed: %v", err)
	}
	if sleeps != 9 {
		t.Fatalf("unexpected retries after reader closed: %d sleeps", sleeps)
	}
	assertAutostartFile(t, path, want)
}
