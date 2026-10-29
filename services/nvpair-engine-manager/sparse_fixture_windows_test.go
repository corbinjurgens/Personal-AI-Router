// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"os"
	"testing"
	"unsafe"

	"golang.org/x/sys/windows"
)

func prepareSparseFixture(t *testing.T, file *os.File) {
	t.Helper()
	var returned uint32
	err := windows.DeviceIoControl(windows.Handle(file.Fd()), windows.FSCTL_SET_SPARSE, nil, 0, nil, 0, &returned, nil)
	if errors.Is(err, windows.ERROR_INVALID_FUNCTION) || errors.Is(err, windows.ERROR_NOT_SUPPORTED) {
		file.Close()
		t.Skipf("volume does not support sparse files: %v", err)
	}
	if err != nil {
		t.Fatalf("mark fixture sparse: %v", err)
	}
}

func checkSparseFixtureAllocation(t *testing.T, file *os.File, logical int64) {
	t.Helper()
	// FILE_STANDARD_INFO reports allocation and end-of-file sizes without reading data.
	var info struct {
		AllocationSize int64
		EndOfFile      int64
		NumberOfLinks  uint32
		DeletePending  byte
		Directory      byte
	}
	err := windows.GetFileInformationByHandleEx(windows.Handle(file.Fd()), windows.FileStandardInfo, (*byte)(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info)))
	if err != nil {
		t.Fatalf("query fixture allocation: %v", err)
	}
	t.Logf("sparse fixture logical size=%d allocated size=%d", info.EndOfFile, info.AllocationSize)
	if info.EndOfFile != logical || info.AllocationSize > 1<<20 {
		t.Fatalf("fixture is not small and sparse: logical=%d allocated=%d", info.EndOfFile, info.AllocationSize)
	}
}
