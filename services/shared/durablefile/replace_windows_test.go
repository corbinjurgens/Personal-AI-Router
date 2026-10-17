// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package durablefile

import (
	"errors"
	"os"
	"syscall"
	"testing"
	"time"

	"golang.org/x/sys/windows"
)

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
