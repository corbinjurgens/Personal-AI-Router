// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func procImage(pid int) string { return imagePathForPID(pid) }

func TestParsePSComm(t *testing.T) {
	test := func(name, output string, want []string) {
		t.Run(name, func(t *testing.T) {
			if got := parsePSComm([]byte(output)); !slices.Equal(got, want) {
				t.Fatalf("parsePSComm = %q; want %q", got, want)
			}
		})
	}
	test("paths and spaces stay intact", " /sbin/launchd\n/Applications/LM Studio.app/Contents/MacOS/LM Studio\n llmster \n", []string{
		"/sbin/launchd",
		"/Applications/LM Studio.app/Contents/MacOS/LM Studio",
		"llmster",
	})
	test("drops blank lines and trims whitespace", "\n \t\n\t/usr/bin/ps\r\n\n", []string{"/usr/bin/ps"})
	test("no final newline", "lm-studio", []string{"lm-studio"})
	test("empty output", "", nil)
}

func TestListProcessNamesIncludesSelf(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatalf("resolve test executable: %v", err)
	}
	names, err := listProcessNames(context.Background())
	if err != nil {
		t.Fatalf("listProcessNames: %v", err)
	}
	if len(names) == 0 {
		t.Fatal("listProcessNames returned no names")
	}
	if !anyProcessMatches(names, []string{filepath.Base(executable)}) {
		t.Fatalf("process list does not contain test executable %q", filepath.Base(executable))
	}
}
