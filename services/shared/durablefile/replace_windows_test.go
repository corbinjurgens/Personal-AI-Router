// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package durablefile

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"golang.org/x/sys/windows"
)

// No goroutine, timer or retry releases this reader before replacement. This
// is the native proof gate for FileRenameInfoEx with POSIX_SEMANTICS and
// REPLACE_IF_EXISTS, independently of WriteFile's retry policy.
func TestFileRenameInfoExReplacesHeldTarget(t *testing.T) {
	dir := t.TempDir()
	target := writeTarget(t, dir, "old bytes")
	reader, err := Open(target)
	if err != nil {
		t.Fatalf("open replacement-compatible reader: %v", err)
	}
	t.Cleanup(func() {
		if err := reader.Close(); err != nil {
			t.Errorf("close reader: %v", err)
		}
	})
	tmp := filepath.Join(dir, "replacement.tmp")
	replacement, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		t.Fatalf("create replacement: %v", err)
	}
	_, writeErr := replacement.Write([]byte("new bytes"))
	syncErr := replacement.Sync()
	closeErr := replacement.Close()
	if err := errors.Join(writeErr, syncErr, closeErr); err != nil {
		t.Fatalf("write, flush and close replacement before rename: %v", err)
	}
	if err := replaceFile(tmp, target); err != nil {
		t.Fatalf("FileRenameInfoEx with held sharing-compatible reader: %v", err)
	}
	old, err := io.ReadAll(reader)
	if err != nil || string(old) != "old bytes" {
		t.Fatalf("held reader = %q, error = %v; want old bytes", old, err)
	}
	if got := readString(t, target); got != "new bytes" {
		t.Fatalf("fresh reader = %q, want new bytes", got)
	}
	assertOnlyTarget(t, dir, target)
}

func TestWriteFileReplacesHeldSharingCompatibleReader(t *testing.T) {
	dir := t.TempDir()
	target := writeTarget(t, dir, "old bytes")
	reader, err := Open(target)
	if err != nil {
		t.Fatalf("open replacement-compatible reader: %v", err)
	}
	t.Cleanup(func() {
		if err := reader.Close(); err != nil {
			t.Errorf("close reader: %v", err)
		}
	})
	r := Replacer{
		IsTransient: func(error) bool { return false },
		Sleep:       func(time.Duration) { t.Fatal("replacement retried with compatible reader held") },
	}
	if err := r.WriteFile(target, []byte("new bytes"), 0o600); err != nil {
		t.Fatalf("WriteFile with reader held: %v", err)
	}
	old, err := io.ReadAll(reader)
	if err != nil || string(old) != "old bytes" {
		t.Fatalf("held reader = %q, error = %v; want old bytes", old, err)
	}
	if got := readString(t, target); got != "new bytes" {
		t.Fatalf("fresh reader = %q, want new bytes", got)
	}
	assertOnlyTarget(t, dir, target)
}

func TestWriteFileRejectsHeldIncompatibleReader(t *testing.T) {
	dir := t.TempDir()
	target := writeTarget(t, dir, "old bytes")
	reader, err := os.Open(target)
	if err != nil {
		t.Fatalf("open reader without delete sharing: %v", err)
	}
	t.Cleanup(func() {
		if err := reader.Close(); err != nil {
			t.Errorf("close reader: %v", err)
		}
	})
	r := Replacer{IsTransient: func(error) bool { return false }}
	err = r.WriteFile(target, []byte("new bytes"), 0o600)
	if !errors.Is(err, windows.ERROR_SHARING_VIOLATION) && !errors.Is(err, windows.ERROR_ACCESS_DENIED) {
		t.Fatalf("replacement error = %v, want sharing violation or access denied", err)
	}
	if errors.Is(err, ErrReplaced) {
		t.Fatalf("failed replacement reported landed bytes: %v", err)
	}
	if got := readString(t, target); got != "old bytes" {
		t.Fatalf("failed replacement changed target to %q", got)
	}
	assertOnlyTarget(t, dir, target)
}

func TestWriteFilePreservesReadOnlyTarget(t *testing.T) {
	dir := t.TempDir()
	target := writeTarget(t, dir, "old bytes")
	name, err := windows.UTF16PtrFromString(target)
	if err != nil {
		t.Fatalf("encode target: %v", err)
	}
	if err := windows.SetFileAttributes(name, windows.FILE_ATTRIBUTE_READONLY); err != nil {
		t.Fatalf("make target read-only: %v", err)
	}
	t.Cleanup(func() {
		if err := windows.SetFileAttributes(name, windows.FILE_ATTRIBUTE_NORMAL); err != nil {
			t.Errorf("restore target attributes: %v", err)
		}
	})
	r := Replacer{IsTransient: func(error) bool { return false }}
	err = r.WriteFile(target, []byte("new bytes"), 0o600)
	if !errors.Is(err, windows.ERROR_ACCESS_DENIED) || errors.Is(err, ErrReplaced) {
		t.Fatalf("read-only replacement error = %v, want access denied before replacement", err)
	}
	if got := readString(t, target); got != "old bytes" {
		t.Fatalf("read-only target changed to %q", got)
	}
	assertOnlyTarget(t, dir, target)
}

func TestIsTransientReplaceErrorMatchesSharingErrnos(t *testing.T) {
	test := func(name string, err error, want bool) {
		t.Run(name, func(t *testing.T) {
			if got := isTransientReplaceError(err); got != want {
				t.Fatalf("isTransientReplaceError(%v) = %v, want %v", err, got, want)
			}
		})
	}
	test("sharing violation", &os.LinkError{Op: "rename", Err: syscall.Errno(32)}, true)
	test("access denied", &os.LinkError{Op: "rename", Err: syscall.Errno(5)}, true)
	test("file not found", &os.LinkError{Op: "rename", Err: syscall.Errno(2)}, false)
	test("no error", nil, false)
}

// A handle opened with os.Open does not share delete access, so replacing the
// target while it is open fails the way a concurrent reader makes it fail.
func TestWriteFileRetriesWhileReaderHoldsTarget(t *testing.T) {
	t.Run("succeeds once the reader releases the target", func(t *testing.T) {
		dir := t.TempDir()
		path := writeTarget(t, dir, "original bytes")
		reader, err := os.Open(path)
		if err != nil {
			t.Fatalf("open reader: %v", err)
		}
		defer reader.Close()
		var delays []time.Duration
		r := Replacer{Sleep: func(d time.Duration) {
			delays = append(delays, d)
			if len(delays) == 2 {
				if err := reader.Close(); err != nil {
					t.Errorf("close reader: %v", err)
				}
			}
		}}
		if err := r.WriteFile(path, []byte("replacement"), 0o600); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		if len(delays) != 2 {
			t.Fatalf("sleeps = %d, want 2 retries before the reader released", len(delays))
		}
		if got := readString(t, path); got != "replacement" {
			t.Fatalf("target = %q, want the new bytes", got)
		}
		assertOnlyTarget(t, dir, path)
	})
	t.Run("keeps the target when the reader never releases it", func(t *testing.T) {
		dir := t.TempDir()
		path := writeTarget(t, dir, "original bytes")
		reader, err := os.Open(path)
		if err != nil {
			t.Fatalf("open reader: %v", err)
		}
		defer reader.Close()
		sleeps := 0
		r := Replacer{Sleep: func(time.Duration) { sleeps++ }}
		err = r.WriteFile(path, []byte("replacement"), 0o600)
		var errno syscall.Errno
		if !errors.As(err, &errno) || (errno != windows.ERROR_ACCESS_DENIED && errno != windows.ERROR_SHARING_VIOLATION) {
			t.Fatalf("error = %v, want an access denied or sharing violation", err)
		}
		if sleeps != 9 {
			t.Fatalf("sleeps = %d, want 9 between 10 attempts", sleeps)
		}
		if got := readString(t, path); got != "original bytes" {
			t.Fatalf("target = %q, want unchanged", got)
		}
		assertOnlyTarget(t, dir, path)
	})
}
