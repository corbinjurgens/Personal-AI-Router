// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"errors"
	"syscall"
)

// Only explicit nonexistence proves exit. Permission/query errors and a reused
// live PID retain the fence; this check never sends a terminating signal.
func orphanPIDExited(pid int) bool {
	return pid > 0 && errors.Is(syscall.Kill(pid, 0), syscall.ESRCH)
}
