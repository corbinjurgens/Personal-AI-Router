// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"runtime"
	"strings"
	"testing"
)

func TestExtractSystemdFlag(t *testing.T) {
	got, found := extractSystemdFlag([]string{"--systemd", "--log-level", "debug", "--", "--systemd"})
	if !found || strings.Join(got, " ") != "--log-level debug -- --systemd" {
		t.Fatalf("got %q, %v", got, found)
	}
	if _, found := extractSystemdFlag([]string{"--", "--systemd"}); found {
		t.Fatal("broker arg --systemd taken as the flag")
	}
}

func TestAutostartSystemdRejectedOffLinux(t *testing.T) {
	if runtime.GOOS == "linux" {
		t.Skip("--systemd is valid on Linux")
	}
	var out strings.Builder
	if err := runAutostart([]string{"enable", "--systemd"}, &out); err == nil {
		t.Fatal("--systemd accepted off Linux")
	}
}
