// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package artifactbinding

import (
	"golang.org/x/sys/windows"
	"os"
	"strings"
)

func Resolve(path string) (string, error) {
	resolved, _, err := FinalPath(path)
	if err != nil {
		return "", err
	}
	return bindingSpelling(resolved), nil
}

func bindingSpelling(resolved string) string {
	// Keep the DOS/UNC spelling used by existing source bindings. The handle
	// resolver retains the extended prefix for its process-ownership callers.
	if strings.HasPrefix(resolved, `\\?\UNC\`) {
		return `\\` + strings.TrimPrefix(resolved, `\\?\UNC\`)
	}
	return strings.TrimPrefix(resolved, `\\?\`)
}

func FinalPath(path string) (string, os.FileInfo, error) {
	name, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return "", nil, err
	}
	h, err := windows.CreateFile(name, 0, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE, nil, windows.OPEN_EXISTING, windows.FILE_FLAG_BACKUP_SEMANTICS, 0)
	if err != nil {
		return "", nil, err
	}
	file := os.NewFile(uintptr(h), path)
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return "", nil, err
	}
	// Freeze lazy Windows file IDs while this metadata handle is still open.
	if !os.SameFile(info, info) {
		return "", nil, windows.ERROR_INVALID_HANDLE
	}
	buf := make([]uint16, 32768)
	// Flags zero requests FILE_NAME_NORMALIZED and VOLUME_NAME_DOS, including
	// UNC paths. The extended-path prefix is kept for subsequent OS lookups.
	n, err := windows.GetFinalPathNameByHandle(h, &buf[0], uint32(len(buf)), 0)
	if err != nil {
		return "", nil, err
	}
	if n == 0 || n >= uint32(len(buf)) {
		return "", nil, windows.ERROR_INSUFFICIENT_BUFFER
	}
	return windows.UTF16ToString(buf[:n]), info, nil
}
