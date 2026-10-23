// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"fmt"
	"os"
	"syscall"
)

// strataFileID is the device and inode of an open file.
func strataFileID(f *os.File) (string, error) {
	info, err := f.Stat()
	if err != nil {
		return "", err
	}
	st, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return "", fmt.Errorf("no file identity")
	}
	return fmt.Sprintf("%x:%x", uint64(st.Dev), st.Ino), nil
}

// strataOpenKeyFile opens without following a final symlink and without
// blocking if a FIFO replaced the checked file.
func strataOpenKeyFile(path string) (*os.File, error) {
	return os.OpenFile(path, os.O_RDONLY|syscall.O_NOFOLLOW|syscall.O_NONBLOCK, 0)
}
