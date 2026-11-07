// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package pipeio

import (
	"io"
	"os"

	"golang.org/x/sys/unix"
)

// NewReader takes ownership of a pipe reader. Inherited blocking descriptors
// (including stdin) do not join Go's poller, so Close cannot interrupt a read.
// Rewrap anonymous pipes in nonblocking mode before any read can begin.
// Darwin filesystem FIFOs retain their blocking reads for last-writer EOF;
// regular files and terminals retain their ordinary os.File behavior.
func NewReader(file *os.File) (io.ReadCloser, error) {
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if info.Mode()&os.ModeNamedPipe == 0 {
		return file, nil
	}
	preserve, err := preservePipeEOF(file)
	if err != nil {
		return nil, err
	}
	if preserve {
		return file, nil
	}
	fd, err := unix.FcntlInt(file.Fd(), unix.F_DUPFD_CLOEXEC, 0)
	if err != nil {
		return nil, os.NewSyscallError("fcntl", err)
	}
	if err := unix.SetNonblock(fd, true); err != nil {
		_ = unix.Close(fd)
		return nil, os.NewSyscallError("setnonblock", err)
	}
	input := os.NewFile(uintptr(fd), file.Name())
	if err := file.Close(); err != nil {
		_ = input.Close()
		return nil, err
	}
	return input, nil
}
