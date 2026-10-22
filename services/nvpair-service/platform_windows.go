// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"

	"nvpair-shared/servicectl"
)

// runAutostartService reexecutes the ordinary serve command without a console window.
// It returns after process creation; service and broker readiness come later.
func runAutostartService(args []string) error {
	if err := validateRunArgs(args); err != nil {
		return err
	}
	exe, err := os.Executable()
	if err != nil {
		return fmt.Errorf("locate own executable: %w", err)
	}
	if resolved, err := filepath.EvalSymlinks(exe); err == nil {
		exe = resolved
	}
	if err := servicectl.StartDetachedArgs(exe, args); err != nil {
		return fmt.Errorf("start autostart service: %w", err)
	}
	return nil
}

// configureBroker hides the broker's console and gives it its own process
// group, so a console Ctrl+C sent to the service is not delivered to the broker
// as well; the service stops it in order instead.
func configureBroker(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		HideWindow: true,
		// CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
		CreationFlags: 0x08000000 | 0x00000200,
	}
}

// acquireInstanceLock is a no-op on Windows: the named pipe is created with
// FILE_FLAG_FIRST_PIPE_INSTANCE, so a second service cannot listen on it and
// there is no socket file that could be mistaken for a stale one.
func acquireInstanceLock(string) (func(), error) {
	return func() {}, nil
}
