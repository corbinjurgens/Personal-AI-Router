// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/windows"
)

func TestMemoryPlanProcessWindowsJobTrees(t *testing.T) { testMemoryPlanProcessTrees(t) }

func TestMemoryPlanProcessWindowsSuspendedReapDeadline(t *testing.T) {
	command := memoryPlanTestCommand(t)
	cmd := exec.Command(command.Command[0], command.Command[1:]...)
	configureSysProcAttr(cmd)
	cmd.SysProcAttr.CreationFlags |= windows.CREATE_SUSPENDED
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	waited := make(chan struct{})
	defer func() {
		if err := cmd.Process.Kill(); err != nil {
			t.Error(err)
		}
		select {
		case <-waited:
		case <-time.After(time.Second):
			t.Error("suspended fixture did not reap")
		}
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	err := reapMemoryPlanStartFailure(ctx, func() error { defer close(waited); return cmd.Wait() })
	if !errors.Is(err, errMemoryPlanCleanup) {
		t.Fatalf("suspended process must remain unconfirmed: %v", err)
	}
}

func assertMemoryPlanProcessStopped(t *testing.T, pid int) {
	t.Helper()
	h, err := windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if errors.Is(err, windows.ERROR_INVALID_PARAMETER) {
		return
	}
	if err != nil {
		t.Fatalf("open descendant %d handle: %v", pid, err)
	}
	defer func() {
		if err := windows.CloseHandle(h); err != nil {
			t.Errorf("close descendant %d handle: %v", pid, err)
		}
	}()
	if err := waitMemoryPlanProcessStopped(h, windows.WaitForSingleObject, windows.GetExitCodeProcess); err != nil {
		t.Fatalf("descendant %d: %v", pid, err)
	}
}

// GetExitCodeProcess returns STILL_ACTIVE while a process is running. Only a
// terminal exit code permits waiting for asynchronous process-object signaling.
const memoryPlanProcessStillActive uint32 = 259

func waitMemoryPlanProcessStopped(h windows.Handle, wait func(windows.Handle, uint32) (uint32, error), queryExitCode func(windows.Handle, *uint32) error) error {
	status, err := wait(h, 0)
	if err != nil {
		return fmt.Errorf("initial process wait: status=%d: %w", status, err)
	}
	if status == windows.WAIT_OBJECT_0 {
		return nil
	}
	if status != uint32(windows.WAIT_TIMEOUT) {
		return fmt.Errorf("unexpected initial process wait status=%d", status)
	}
	var exitCode uint32
	if err := queryExitCode(h, &exitCode); err != nil {
		return fmt.Errorf("query unsignaled process exit code: initial status=%d: %w", status, err)
	}
	if exitCode == memoryPlanProcessStillActive {
		return fmt.Errorf("process still active at return: initial status=%d exit code=%d", status, exitCode)
	}
	started := time.Now()
	finalStatus, err := wait(h, uint32(memoryPlanCleanupBudget/time.Millisecond))
	if err != nil {
		return fmt.Errorf("terminal process wait: initial status=%d exit code=%d final status=%d elapsed=%v: %w", status, exitCode, finalStatus, time.Since(started), err)
	}
	if finalStatus != windows.WAIT_OBJECT_0 {
		return fmt.Errorf("terminal process did not signal: initial status=%d exit code=%d final status=%d elapsed=%v budget=%v", status, exitCode, finalStatus, time.Since(started), memoryPlanCleanupBudget)
	}
	return nil
}

func TestMemoryPlanProcessWindowsRejectsLiveProcess(t *testing.T) {
	h, err := windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(os.Getpid()))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := windows.CloseHandle(h); err != nil {
			t.Errorf("close live fixture handle: %v", err)
		}
	}()
	waits := 0
	err = waitMemoryPlanProcessStopped(h, func(got windows.Handle, timeout uint32) (uint32, error) {
		waits++
		if got != h || timeout != 0 {
			t.Fatalf("live process must only receive an immediate wait: handle=%v want=%v timeout=%d", got, h, timeout)
		}
		return windows.WaitForSingleObject(got, timeout)
	}, windows.GetExitCodeProcess)
	if err == nil || !strings.Contains(err.Error(), "still active at return") {
		t.Fatalf("live process must fail before bounded waiting: %v", err)
	}
	if waits != 1 {
		t.Fatalf("live process wait calls=%d, want 1", waits)
	}
}

func TestMemoryPlanProcessWindowsTerminalWait(t *testing.T) {
	nativeErr := errors.New("native fixture error")
	type waitResult struct {
		status uint32
		err    error
	}
	for _, tc := range []struct {
		name      string
		initial   waitResult
		exitCode  uint32
		queryErr  error
		final     waitResult
		wantCalls string
		wantError string
	}{
		{
			name: "already signaled", initial: waitResult{status: windows.WAIT_OBJECT_0},
			wantCalls: "inspect",
		},
		{
			name: "terminal process signals", initial: waitResult{status: uint32(windows.WAIT_TIMEOUT)}, exitCode: 1,
			final: waitResult{status: windows.WAIT_OBJECT_0}, wantCalls: "inspect,query,wait",
		},
		{
			name: "live process refused without positive wait", initial: waitResult{status: uint32(windows.WAIT_TIMEOUT)}, exitCode: memoryPlanProcessStillActive,
			wantCalls: "inspect,query", wantError: "still active at return",
		},
		{
			name: "initial wait error", initial: waitResult{status: windows.WAIT_FAILED, err: nativeErr},
			wantCalls: "inspect", wantError: "initial process wait",
		},
		{
			name: "unexpected initial status", initial: waitResult{status: windows.WAIT_ABANDONED},
			wantCalls: "inspect", wantError: "unexpected initial process wait",
		},
		{
			name: "exit query error", initial: waitResult{status: uint32(windows.WAIT_TIMEOUT)}, queryErr: nativeErr,
			wantCalls: "inspect,query", wantError: "query unsignaled process exit code",
		},
		{
			name: "terminal wait error", initial: waitResult{status: uint32(windows.WAIT_TIMEOUT)}, exitCode: 1,
			final: waitResult{status: windows.WAIT_FAILED, err: nativeErr}, wantCalls: "inspect,query,wait", wantError: "terminal process wait",
		},
		{
			name: "terminal wait timeout", initial: waitResult{status: uint32(windows.WAIT_TIMEOUT)}, exitCode: 1,
			final: waitResult{status: uint32(windows.WAIT_TIMEOUT)}, wantCalls: "inspect,query,wait", wantError: "terminal process did not signal",
		},
		{
			name: "unexpected terminal status", initial: waitResult{status: uint32(windows.WAIT_TIMEOUT)}, exitCode: 1,
			final: waitResult{status: windows.WAIT_ABANDONED}, wantCalls: "inspect,query,wait", wantError: "terminal process did not signal",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const handle windows.Handle = 42
			var calls []string
			err := waitMemoryPlanProcessStopped(handle, func(got windows.Handle, timeout uint32) (uint32, error) {
				if got != handle {
					t.Fatalf("wait handle=%v, want retained handle=%v", got, handle)
				}
				if len(calls) == 0 {
					if timeout != 0 {
						t.Fatalf("initial wait timeout=%d, want 0", timeout)
					}
					calls = append(calls, "inspect")
					return tc.initial.status, tc.initial.err
				}
				if strings.Join(calls, ",") != "inspect,query" || timeout != uint32(memoryPlanCleanupBudget/time.Millisecond) {
					t.Fatalf("bounded wait before terminal proof or outside cleanup budget: calls=%v timeout=%d", calls, timeout)
				}
				calls = append(calls, "wait")
				return tc.final.status, tc.final.err
			}, func(got windows.Handle, code *uint32) error {
				if got != handle || strings.Join(calls, ",") != "inspect" {
					t.Fatalf("exit query must follow initial inspection on retained handle: handle=%v calls=%v", got, calls)
				}
				calls = append(calls, "query")
				*code = tc.exitCode
				return tc.queryErr
			})
			if got := strings.Join(calls, ","); got != tc.wantCalls {
				t.Fatalf("native call order=%q, want %q", got, tc.wantCalls)
			}
			if tc.wantError == "" {
				if err != nil {
					t.Fatalf("confirmed stopped process refused: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.wantError) {
				t.Fatalf("error=%v, want %q", err, tc.wantError)
			}
			if (tc.initial.err != nil || tc.queryErr != nil || tc.final.err != nil) && !errors.Is(err, nativeErr) {
				t.Fatalf("native error not preserved: %v", err)
			}
		})
	}
}

func TestMemoryPlanProcessWindowsPrivateDirectory(t *testing.T) {
	dir, err := makeMemoryPlanTempDir()
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	sd, err := windows.GetNamedSecurityInfo(dir, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION)
	if err != nil {
		t.Fatal(err)
	}
	user, err := windows.GetCurrentProcessToken().GetTokenUser()
	if err != nil {
		t.Fatal(err)
	}
	sddl := sd.String()
	if !strings.Contains(sddl, "D:P") || !strings.Contains(sddl, user.User.Sid.String()) || strings.Contains(sddl, ";;;WD)") || strings.Contains(sddl, ";;;BU)") {
		t.Fatalf("directory ACL: %s", sddl)
	}
}
