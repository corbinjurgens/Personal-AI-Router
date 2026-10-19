// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"testing"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

func isPortBridgeSharingViolation(err error) bool {
	return errors.Is(err, windows.ERROR_SHARING_VIOLATION)
}

func TestCopyPortBridgeRetriesSharingViolation(t *testing.T) {
	test := func(name string, sharingError error) {
		t.Run(name, func(t *testing.T) {
			writes, sleeps := 0, 0
			err := copyPortBridge(func() error {
				writes++
				if writes == 1 {
					return sharingError
				}
				return nil
			}, func(delay time.Duration) {
				sleeps++
				if delay != portBridgeCopyDelay {
					t.Errorf("retry delay = %v, want %v", delay, portBridgeCopyDelay)
				}
			})
			if err != nil {
				t.Fatalf("copy after sharing violation: %v", err)
			}
			if writes != 2 || sleeps != 1 {
				t.Fatalf("writes/sleeps = %d/%d, want 2/1", writes, sleeps)
			}
		})
	}
	test("direct sharing violation", windows.ERROR_SHARING_VIOLATION)
	test("wrapped file sharing violation", &os.PathError{Op: "open", Path: "port-bridge.exe", Err: windows.ERROR_SHARING_VIOLATION})
}

func TestCopyPortBridgeBoundsSharingViolationRetries(t *testing.T) {
	writes, sleeps := 0, 0
	var elapsed time.Duration
	err := copyPortBridge(func() error {
		writes++
		return windows.ERROR_SHARING_VIOLATION
	}, func(delay time.Duration) {
		sleeps++
		elapsed += delay
	})
	if !errors.Is(err, windows.ERROR_SHARING_VIOLATION) {
		t.Fatalf("copy error = %v, want sharing violation", err)
	}
	if writes != portBridgeCopyAttempts || sleeps != portBridgeCopyAttempts-1 {
		t.Fatalf("writes/sleeps = %d/%d, want %d/%d", writes, sleeps, portBridgeCopyAttempts, portBridgeCopyAttempts-1)
	}
	if elapsed != 2*time.Second {
		t.Fatalf("total retry delay = %v, want 2s", elapsed)
	}
}

func TestCopyPortBridgeRejectsOtherWindowsErrors(t *testing.T) {
	test := func(name string, copyError error, wantSharingRetry bool) {
		t.Run(name, func(t *testing.T) {
			writes, sleeps := 0, 0
			wantWrites := 1
			if wantSharingRetry {
				wantWrites++
			}
			err := copyPortBridge(func() error {
				writes++
				if wantSharingRetry && writes == 1 {
					return windows.ERROR_SHARING_VIOLATION
				}
				return copyError
			}, func(time.Duration) { sleeps++ })
			if err != copyError {
				t.Fatalf("copy error = %v, want original error %v", err, copyError)
			}
			if writes != wantWrites || sleeps != wantWrites-1 {
				t.Fatalf("writes/sleeps = %d/%d, want %d/%d", writes, sleeps, wantWrites, wantWrites-1)
			}
		})
	}
	test("access denied", windows.ERROR_ACCESS_DENIED, false)
	test("lock violation", windows.ERROR_LOCK_VIOLATION, false)
	test("permission denied after sharing violation", windows.ERROR_ACCESS_DENIED, true)
}

func runOwnedPortWorker(binary string, args []string) error {
	// Assign the adapter before spawning, so the worker inherits membership
	// without a start/assignment race. A non-inheritable job handle ensures
	// terminating this adapter closes the last handle and kills its worker.
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return fmt.Errorf("create port fixture job: %w", err)
	}
	limits := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{}
	limits.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	if _, err := windows.SetInformationJobObject(job, windows.JobObjectExtendedLimitInformation, uintptr(unsafe.Pointer(&limits)), uint32(unsafe.Sizeof(limits))); err != nil {
		return errors.Join(fmt.Errorf("configure port fixture job: %w", err), windows.CloseHandle(job))
	}
	if err := windows.AssignProcessToJobObject(job, windows.CurrentProcess()); err != nil {
		return errors.Join(fmt.Errorf("assign port fixture adapter to job: %w", err), windows.CloseHandle(job))
	}
	// Keep the sole handle open until the adapter's os.Exit, including error
	// paths. Closing it here would also terminate the adapter itself.
	cmd := exec.Command(binary, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}
