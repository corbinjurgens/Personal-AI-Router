// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"testing"
	"time"
)

// TestCopyFileWaitsForSharingViolation holds the destination with no sharing,
// as a just-reaped process image can, and requires copyFile to wait for the
// release instead of failing.
func TestCopyFileWaitsForSharingViolation(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "src.exe")
	dst := filepath.Join(dir, "dst.exe")
	want := []byte("copied executable bytes")
	if err := os.WriteFile(src, want, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dst, []byte("old"), 0o755); err != nil {
		t.Fatal(err)
	}
	name, err := syscall.UTF16PtrFromString(dst)
	if err != nil {
		t.Fatal(err)
	}
	handle, err := syscall.CreateFile(name, syscall.GENERIC_READ|syscall.GENERIC_WRITE, 0, nil, syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		t.Fatalf("hold destination without sharing: %v", err)
	}
	release := sync.OnceFunc(func() {
		if err := syscall.CloseHandle(handle); err != nil {
			t.Errorf("release destination: %v", err)
		}
	})
	defer release()
	if err := os.WriteFile(dst, want, 0o755); !errors.Is(err, errorSharingViolation) {
		t.Fatalf("control: held destination write error = %v, want a sharing violation", err)
	}
	const hold = 300 * time.Millisecond
	time.AfterFunc(hold, release)
	start := time.Now()
	copyFile(t, src, dst)
	if elapsed := time.Since(start); elapsed < hold/2 {
		t.Fatalf("copyFile returned after %s, before the holder released", elapsed)
	}
	got, err := os.ReadFile(dst)
	if err != nil || !bytes.Equal(got, want) {
		t.Fatalf("destination after copy = %q, %v", got, err)
	}
}
