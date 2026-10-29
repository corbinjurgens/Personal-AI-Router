// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"context"
	"errors"
	"unsafe"

	"golang.org/x/sys/windows"
)

func listProcessRecords(ctx context.Context) ([]processRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	snapshot, err := windows.CreateToolhelp32Snapshot(windows.TH32CS_SNAPPROCESS, 0)
	if err != nil {
		return nil, err
	}
	defer windows.CloseHandle(snapshot)
	var entry windows.ProcessEntry32
	entry.Size = uint32(unsafe.Sizeof(entry))
	err = windows.Process32First(snapshot, &entry)
	var records []processRecord
	for err == nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		records = append(records, windowsProcessRecord(entry))
		err = windows.Process32Next(snapshot, &entry)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !errors.Is(err, windows.ERROR_NO_MORE_FILES) {
		return nil, err
	}
	return records, nil
}

func windowsProcessRecord(entry windows.ProcessEntry32) processRecord {
	return processRecord{int(entry.ProcessID), windows.UTF16ToString(entry.ExeFile[:])}
}
