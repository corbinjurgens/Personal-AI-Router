// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package pipeio

import (
	"os"

	"golang.org/x/sys/unix"
)

func preservePipeEOF(file *os.File) (bool, error) {
	// fstatfs requires a vnode: anonymous pipes return EINVAL, while even
	// unlinked filesystem FIFOs still have a filesystem. Keep those FIFOs out
	// of kqueue, which can miss EOF when their last writer closes (Go #24164).
	var stat unix.Statfs_t
	switch err := unix.Fstatfs(int(file.Fd()), &stat); err {
	case nil:
		return true, nil
	case unix.EINVAL:
		return false, nil
	default:
		return false, os.NewSyscallError("fstatfs", err)
	}
}
