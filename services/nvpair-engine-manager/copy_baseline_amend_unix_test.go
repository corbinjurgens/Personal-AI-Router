// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"testing"
)

func TestCopyBaselineAmendUnixPIDParser(t *testing.T) {
	records, err := parsePSRecords([]byte(" 12 /Applications/NVIDIA PAIR.app/Contents/MacOS/NVIDIA PAIR\n 13 nvpair-engine-manager\n"))
	if err != nil || len(records) != 2 || records[0].PID != 12 || records[0].Image != "/Applications/NVIDIA PAIR.app/Contents/MacOS/NVIDIA PAIR" || records[1].PID != 13 {
		t.Fatalf("records %+v %v", records, err)
	}
	for _, input := range []string{"", "12", "x image", "0 image", "-1 image", "12 image\n\n", "12 image\n12 other", "12 "} {
		t.Run(input, func(t *testing.T) {
			if _, e := parsePSRecords([]byte(input)); e == nil {
				t.Fatal("partial accepted")
			}
		})
	}
}
