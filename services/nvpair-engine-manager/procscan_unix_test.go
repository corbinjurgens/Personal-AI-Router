// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"context"
	"os"
	"strings"
	"testing"
)

func procImage(pid int) string { return imagePathForPID(pid) }

func TestListProcessRecordsIncludesSelf(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatalf("resolve test executable: %v", err)
	}
	records, err := listProcessRecords(context.Background())
	if err != nil {
		t.Fatalf("listProcessRecords: %v", err)
	}
	want := normalizeProcessName(executable)
	for _, record := range records {
		if record.PID == os.Getpid() && strings.EqualFold(normalizeProcessName(record.Image), want) {
			return
		}
	}
	t.Fatalf("process list does not contain test executable %q", want)
}
