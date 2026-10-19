// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"os"

	"golang.org/x/sys/windows"
)

// lockAccountStore takes the exclusive lock on the account store's lock file
// without waiting. The returned file holds the lock until it is closed or the
// process ends; Windows releases it on exit, including a crash. LockFileEx
// locks belong to the handle, so a second open in this process is refused too.
func lockAccountStore(path string) (*os.File, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, err
	}
	var overlapped windows.Overlapped
	err = windows.LockFileEx(windows.Handle(f.Fd()), windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &overlapped)
	if err != nil {
		f.Close()
		if errors.Is(err, windows.ERROR_LOCK_VIOLATION) {
			return nil, errAccountStoreLocked
		}
		return nil, err
	}
	return f, nil
}
