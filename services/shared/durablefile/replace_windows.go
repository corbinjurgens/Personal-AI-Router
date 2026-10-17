// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package durablefile

import (
	"errors"
	"syscall"

	"golang.org/x/sys/windows"
)

// replaceFile moves tmp over target and returns once the move is written
// through to disk (MOVEFILE_WRITE_THROUGH), which is Windows' substitute for a
// directory sync.
func replaceFile(tmp, target string) error {
	from, err := windows.UTF16PtrFromString(tmp)
	if err != nil {
		return err
	}
	to, err := windows.UTF16PtrFromString(target)
	if err != nil {
		return err
	}
	return windows.MoveFileEx(from, to, windows.MOVEFILE_REPLACE_EXISTING|windows.MOVEFILE_WRITE_THROUGH)
}

// syncDir is a no-op: replaceFile already wrote the move through.
func syncDir(string) error {
	return nil
}

// isTransientReplaceError matches ERROR_ACCESS_DENIED (5) and
// ERROR_SHARING_VIOLATION (32), which Windows reports while a reader briefly
// holds the target open.
func isTransientReplaceError(err error) bool {
	var errno syscall.Errno
	return errors.As(err, &errno) &&
		(errno == windows.ERROR_ACCESS_DENIED || errno == windows.ERROR_SHARING_VIOLATION)
}
