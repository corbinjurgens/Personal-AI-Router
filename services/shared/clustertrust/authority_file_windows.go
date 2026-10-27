// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package clustertrust

import (
	"io"
	"os"

	"golang.org/x/sys/windows"
)

// openAuthorityFile reads the cluster-manager's atomically replaced files.
// os.Open shares read/write access only, which conflicts with a replacement's
// DELETE-access handle in either open order. Sharing delete access lets each
// reader observe its opened file; the authority brackets still validate fresh
// opens of the path. This grants no write/delete access to the reader and does
// not bypass access checks or retry a failed open.
func openAuthorityFile(path string) (*os.File, error) {
	name, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: path, Err: err}
	}
	handle, err := windows.CreateFile(name, windows.GENERIC_READ,
		windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE,
		nil, windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: path, Err: err}
	}
	return os.NewFile(uintptr(handle), path), nil
}

func readAuthorityFile(path string) ([]byte, error) {
	f, err := openAuthorityFile(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return io.ReadAll(f)
}
