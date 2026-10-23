// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestManagerLaunchesUseIsolatedEnv keeps every manager process a test starts
// on isolatedManagerEnv, so a new launch site cannot inherit the real profile
// and then adopt or stop an installed engine.
func TestManagerLaunchesUseIsolatedEnv(t *testing.T) {
	files, err := filepath.Glob("*_test.go")
	if err != nil {
		t.Fatal(err)
	}
	launches := 0
	for _, file := range files {
		if file == "managerenv_test.go" {
			continue
		}
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(string(data), "\n")
		// Live-tagged tests deliberately reach real engines and are owner-gated.
		if slices.ContainsFunc(lines[:min(5, len(lines))], func(line string) bool {
			return strings.HasPrefix(strings.TrimSpace(line), "//go:build live")
		}) {
			continue
		}
		for i, line := range lines {
			if !strings.Contains(line, "exec.Command(managerBin") {
				continue
			}
			launches++
			if !slices.ContainsFunc(lines[i+1:min(i+4, len(lines))], func(next string) bool {
				return strings.Contains(next, "cmd.Env = isolatedManagerEnv(")
			}) {
				t.Errorf("%s:%d: manager launch does not set cmd.Env = isolatedManagerEnv(...) within the next 3 lines", file, i+1)
			}
		}
	}
	// Guards against a glob or pattern mistake passing vacuously.
	if launches < 3 {
		t.Fatalf("found %d manager launch sites, want at least 3", launches)
	}
}
