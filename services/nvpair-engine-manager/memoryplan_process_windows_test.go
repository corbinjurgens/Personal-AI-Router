// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"context"
	"errors"
	"golang.org/x/sys/windows"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
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
	h, err := windows.OpenProcess(windows.SYNCHRONIZE, false, uint32(pid))
	if errors.Is(err, windows.ERROR_INVALID_PARAMETER) {
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	defer windows.CloseHandle(h)
	status, err := windows.WaitForSingleObject(h, 0)
	if err != nil || status != windows.WAIT_OBJECT_0 {
		t.Fatalf("descendant %d survives: %v, %v", pid, status, err)
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
