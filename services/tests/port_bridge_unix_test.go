//go:build unix

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"os"
	"syscall"
)

// Replace the adapter so the broker owns the worker's actual PID and inherited
// stdio. A broker kill therefore cannot leave a wrapped worker behind.
func runOwnedPortWorker(binary string, args []string) error {
	return syscall.Exec(binary, append([]string{binary}, args...), os.Environ())
}
