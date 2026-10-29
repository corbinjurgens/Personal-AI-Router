// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package durablefile

import (
	"errors"
	"fmt"
	"path/filepath"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

// SyncCreatedDir cannot confirm a newly created directory entry on Windows.
func (r Replacer) SyncCreatedDir(path string) error {
	if r.SyncDir != nil {
		if err := r.SyncDir(filepath.Dir(path)); err != nil {
			return fmt.Errorf("durablefile: sync directory %s: %w", filepath.Dir(path), err)
		}
	}
	return ErrDirectoryDurabilityUnsupported
}

// fileRenameInfo matches FILE_RENAME_INFO, including HANDLE alignment on both
// 32-bit and 64-bit Windows. Flags is the DWORD union used by FileRenameInfoEx.
type fileRenameInfo struct {
	Flags          uint32
	RootDirectory  windows.Handle
	FileNameLength uint32
	FileName       [1]uint16
}

// replaceFile uses POSIX replacement semantics so sharing-compatible readers
// retain the old file while fresh opens see tmp. No read-only override is used.
// The caller has already flushed and closed tmp. Open a write-through handle
// and flush it after the rename; failures after publication wrap ErrReplaced.
// Unsupported Windows/filesystem implementations fail without a fallback.
func replaceFile(tmp, target string) error {
	from, err := windows.UTF16PtrFromString(tmp)
	if err != nil {
		return err
	}
	absoluteTarget, err := filepath.Abs(target)
	if err != nil {
		return err
	}
	to, err := windows.UTF16FromString(absoluteTarget)
	if err != nil {
		return err
	}
	handle, err := windows.CreateFile(from, windows.DELETE|windows.GENERIC_WRITE,
		windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE,
		nil, windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL|windows.FILE_FLAG_WRITE_THROUGH, 0)
	if err != nil {
		return fmt.Errorf("open replacement for DELETE and write-through access: %w", err)
	}
	defer windows.CloseHandle(handle)
	var layout fileRenameInfo
	buffer := make([]byte, int(unsafe.Offsetof(layout.FileName))+len(to)*2)
	info := (*fileRenameInfo)(unsafe.Pointer(&buffer[0]))
	info.Flags = windows.FILE_RENAME_REPLACE_IF_EXISTS | windows.FILE_RENAME_POSIX_SEMANTICS
	info.FileNameLength = uint32((len(to) - 1) * 2)
	copy(unsafe.Slice(&info.FileName[0], len(to)), to)
	if err := windows.SetFileInformationByHandle(handle, windows.FileRenameInfoEx, &buffer[0], uint32(len(buffer))); err != nil {
		if errors.Is(err, windows.ERROR_INVALID_PARAMETER) || errors.Is(err, windows.ERROR_NOT_SUPPORTED) ||
			errors.Is(err, windows.ERROR_INVALID_FUNCTION) || errors.Is(err, windows.ERROR_CALL_NOT_IMPLEMENTED) {
			return fmt.Errorf("Windows/filesystem does not support FileRenameInfoEx (REPLACE_IF_EXISTS|POSIX_SEMANTICS): %w", err)
		}
		return fmt.Errorf("FileRenameInfoEx (REPLACE_IF_EXISTS|POSIX_SEMANTICS): %w", err)
	}
	if err := windows.FlushFileBuffers(handle); err != nil {
		return fmt.Errorf("flush renamed file: %w: %w", ErrReplaced, err)
	}
	return nil
}

// syncDir is a no-op: replaceFile flushes its write-through handle.
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
