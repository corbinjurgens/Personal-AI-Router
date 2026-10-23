// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"fmt"
	"os"
	"syscall"
)

// strataFileID is the volume serial number and file index of an open file.
func strataFileID(f *os.File) (string, error) {
	var info syscall.ByHandleFileInformation
	if err := syscall.GetFileInformationByHandle(syscall.Handle(f.Fd()), &info); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x:%x:%x", info.VolumeSerialNumber, info.FileIndexHigh, info.FileIndexLow), nil
}

// strataOpenKeyFile opens the key file. Windows has no FIFO in the file
// namespace; a reparse point swapped in is caught by the identity checks after
// the read, which compare the handle with the path's own Lstat.
func strataOpenKeyFile(path string) (*os.File, error) {
	return os.Open(path)
}
