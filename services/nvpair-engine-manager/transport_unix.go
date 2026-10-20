// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"fmt"
	"os"
	"time"

	"golang.org/x/sys/unix"
)

func prepareStdio() (*os.File, *os.File, error) {
	input, err := ownStdioFile(os.Stdin)
	if err != nil {
		return nil, nil, err
	}
	output, err := ownStdioFile(os.Stdout)
	if err != nil {
		input.Close()
		return nil, nil, err
	}
	// Probe without imposing a deadline. Do not call Fd on these wrappers:
	// that can restore blocking mode and defeat Close's I/O interruption.
	if err = input.SetReadDeadline(time.Time{}); err == nil {
		err = output.SetWriteDeadline(time.Time{})
	}
	if err != nil {
		output.Close()
		input.Close()
		return nil, nil, fmt.Errorf("stdio must support interruptible pipe I/O: %w", err)
	}
	return input, output, nil
}

func ownStdioFile(inherited *os.File) (*os.File, error) {
	fd, err := unix.FcntlInt(inherited.Fd(), unix.F_DUPFD_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("duplicate %s: %w", inherited.Name(), err)
	}
	if err := unix.SetNonblock(fd, true); err != nil {
		unix.Close(fd)
		return nil, fmt.Errorf("prepare %s: %w", inherited.Name(), err)
	}
	return os.NewFile(uintptr(fd), inherited.Name()), nil
}
