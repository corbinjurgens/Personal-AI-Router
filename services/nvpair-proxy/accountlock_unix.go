// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"errors"
	"os"

	"golang.org/x/sys/unix"
)

// lockAccountStore takes the exclusive lock on the account store's lock file
// without waiting. The returned file holds the lock until it is closed or the
// process ends; the kernel releases it on exit, including a crash. flock locks
// belong to the open file, so a second open in this process is refused too.
func lockAccountStore(path string) (*os.File, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, err
	}
	if err := unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB); err != nil {
		f.Close()
		if errors.Is(err, unix.EWOULDBLOCK) {
			return nil, errAccountStoreLocked
		}
		return nil, err
	}
	return f, nil
}
