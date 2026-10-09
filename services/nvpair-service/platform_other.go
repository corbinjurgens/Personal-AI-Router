// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
)

// configureBroker puts the broker in its own process group, so a Ctrl+C
// delivered to a service running in a terminal reaches only the service, which
// then stops the broker in order instead of the broker dying mid-flight.
func configureBroker(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// acquireInstanceLock takes an exclusive, non-blocking lock beside the socket.
//
// Dialing alone cannot make start-up exclusive: two services starting together
// both find nothing listening, and the second would remove the first's fresh
// socket as stale. The lock serialises them; the loser reports
// errAlreadyRunning. The kernel drops the lock when the process dies, so a
// crash leaves nothing to clean up.
func acquireInstanceLock(endpoint string) (func(), error) {
	if err := os.MkdirAll(filepath.Dir(endpoint), 0o700); err != nil {
		return nil, fmt.Errorf("create endpoint directory: %w", err)
	}
	f, err := os.OpenFile(endpoint+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open instance lock: %w", err)
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		_ = f.Close()
		if errors.Is(err, syscall.EWOULDBLOCK) {
			return nil, errAlreadyRunning
		}
		return nil, fmt.Errorf("lock instance: %w", err)
	}
	return func() { _ = f.Close() }, nil
}
