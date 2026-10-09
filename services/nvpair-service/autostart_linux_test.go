// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAutostartEnableStatusDisableLinux(t *testing.T) {
	cfg := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", cfg)
	entry := filepath.Join(cfg, "autostart", desktopFileName)

	var out bytes.Buffer
	if err := runAutostart([]string{"status"}, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "disabled") {
		t.Fatalf("status before enable = %q", out.String())
	}

	out.Reset()
	if err := runAutostart([]string{"enable", "--", "--log-level", "debug"}, &out); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(entry)
	if err != nil {
		t.Fatalf("no desktop entry written: %v", err)
	}
	exe, _ := os.Executable()
	exe, _ = filepath.EvalSymlinks(exe)
	if !strings.Contains(string(data), "Exec="+desktopExecQuote(exe)+" -- --log-level debug\n") {
		t.Fatalf("desktop entry:\n%s", data)
	}

	out.Reset()
	if err := runAutostart([]string{"status"}, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "enabled") || !strings.Contains(out.String(), "--log-level debug") {
		t.Fatalf("status after enable = %q", out.String())
	}

	out.Reset()
	if err := runAutostart([]string{"disable"}, &out); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(entry); !os.IsNotExist(err) {
		t.Fatalf("desktop entry still present: %v", err)
	}
	out.Reset()
	if err := runAutostart([]string{"disable"}, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "was not enabled") {
		t.Fatalf("second disable = %q", out.String())
	}
}

func TestAutostartIgnoresRelativeXDGConfigHome(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", "relative/dir")
	got, err := autostartLocation()
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(home, ".config", "autostart", desktopFileName); got != want {
		t.Fatalf("location = %q, want %q", got, want)
	}
}
