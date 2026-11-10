// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build darwin

package main

import (
	"os/exec"
	"strings"
	"testing"

	"nvpair-shared/nodename"
)

// TestDefaultNodeNameMatchesScutilComputerName checks the macOS default
// against the real ComputerName on this machine.
func TestDefaultNodeNameMatchesScutilComputerName(t *testing.T) {
	out, err := exec.Command("/usr/sbin/scutil", "--get", "ComputerName").Output()
	if err != nil {
		t.Skipf("scutil --get ComputerName failed: %v", err)
	}
	want, err := nodename.Normalize(strings.TrimSpace(string(out)))
	if err != nil {
		t.Skipf("ComputerName %q is not a valid node name: %v", out, err)
	}
	if got := defaultNodeName(testNameUUID); got != want {
		t.Fatalf("defaultNodeName() = %q, want ComputerName %q", got, want)
	}
	t.Logf("default node name %q", want)
}
