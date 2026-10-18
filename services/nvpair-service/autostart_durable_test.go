// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func assertAutostartFile(t *testing.T, path string, want []byte) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read entry: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("entry bytes = %q, want %q", got, want)
	}
	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatalf("read entry directory: %v", err)
	}
	if len(entries) != 1 || entries[0].Name() != filepath.Base(path) {
		t.Fatalf("directory entries = %v, want only %s", entries, filepath.Base(path))
	}
}

func TestWriteFileAtomicPreReplaceFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "entry.desktop")
	old := []byte("original\n")
	if err := os.WriteFile(path, old, 0o644); err != nil {
		t.Fatal(err)
	}
	saved := autostartFileReplacer
	t.Cleanup(func() { autostartFileReplacer = saved })
	wantErr := errors.New("injected replacement failure")
	autostartFileReplacer.Replace = func(string, string) error { return wantErr }
	err := writeFileAtomic(path, []byte("replacement\n"), 0o644)
	if !errors.Is(err, wantErr) {
		t.Fatalf("write error = %v, want %v", err, wantErr)
	}
	assertAutostartFile(t, path, old)
}

func TestWriteFileAtomicAdoptsReplacedEntry(t *testing.T) {
	path := filepath.Join(t.TempDir(), "entry.plist")
	if err := os.WriteFile(path, []byte("original"), 0o644); err != nil {
		t.Fatal(err)
	}
	saved := autostartFileReplacer
	t.Cleanup(func() { autostartFileReplacer = saved })
	syncCalls := 0
	autostartFileReplacer.SyncDir = func(string) error {
		syncCalls++
		return errors.New("injected directory sync failure")
	}
	want := []byte("replacement\n")
	if err := writeFileAtomic(path, want, 0o644); err != nil {
		t.Fatalf("landed entry rejected: %v", err)
	}
	if syncCalls != 1 {
		t.Fatalf("directory sync calls = %d, want 1", syncCalls)
	}
	assertAutostartFile(t, path, want)
}

func TestWriteFileAtomicPreservesExactBytes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "new", "entry.service")
	want := []byte("[Service]\r\nExecStart=/opt/PAIR\n\x00\xff\n")
	if err := writeFileAtomic(path, want, 0o644); err != nil {
		t.Fatal(err)
	}
	assertAutostartFile(t, path, want)
}
