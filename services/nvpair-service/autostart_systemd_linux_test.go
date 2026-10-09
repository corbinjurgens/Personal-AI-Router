// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakeSystemctl replaces systemctlRun with a recorder for one test.
func fakeSystemctl(t *testing.T, fail bool) *[]string {
	t.Helper()
	var calls []string
	old := systemctlRun
	systemctlRun = func(args ...string) ([]byte, error) {
		calls = append(calls, strings.Join(args, " "))
		if fail {
			return []byte("boom"), errors.New("exit status 1")
		}
		return []byte("enabled\n"), nil
	}
	t.Cleanup(func() { systemctlRun = old })
	return &calls
}

func TestAutostartSystemdEnableStatusDisable(t *testing.T) {
	cfg := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", cfg)
	calls := fakeSystemctl(t, false)
	unit := filepath.Join(cfg, "systemd", "user", systemdUnitName)

	var out bytes.Buffer
	if err := runAutostart([]string{"enable", "--systemd", "--log-level", "debug", "--", "--foo", "a b"}, &out); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(unit)
	if err != nil {
		t.Fatalf("no unit written: %v", err)
	}
	exe, _ := os.Executable()
	exe, _ = filepath.EvalSymlinks(exe)
	for _, want := range []string{
		"ExecStart=" + systemdQuote(exe) + ` --log-level debug -- --foo "a b"` + "\n",
		"Restart=on-failure\n",
		"WantedBy=default.target\n",
	} {
		if !strings.Contains(string(data), want) {
			t.Fatalf("unit missing %q:\n%s", want, data)
		}
	}
	if _, err := os.Stat(filepath.Join(cfg, "autostart", desktopFileName)); !os.IsNotExist(err) {
		t.Fatalf("--systemd also wrote the XDG entry: %v", err)
	}
	if got, want := strings.Join(*calls, ";"), "daemon-reload;enable "+systemdUnitName; got != want {
		t.Fatalf("systemctl calls = %q, want %q", got, want)
	}
	if !strings.Contains(out.String(), "loginctl enable-linger") {
		t.Fatalf("no linger hint: %q", out.String())
	}

	out.Reset()
	if err := runAutostart([]string{"status"}, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "systemd user unit") || !strings.Contains(out.String(), "--log-level debug") {
		t.Fatalf("status = %q", out.String())
	}

	// An XDG entry alongside is removed by the same disable.
	if err := runAutostart([]string{"enable"}, &out); err != nil {
		t.Fatal(err)
	}
	*calls = nil
	out.Reset()
	if err := runAutostart([]string{"disable"}, &out); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{unit, filepath.Join(cfg, "autostart", desktopFileName)} {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Fatalf("%s still present: %v", p, err)
		}
	}
	if len(*calls) == 0 || (*calls)[0] != "disable "+systemdUnitName {
		t.Fatalf("systemctl calls on disable = %q", *calls)
	}
}

func TestAutostartSystemdSystemctlFailureKeepsUnit(t *testing.T) {
	cfg := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", cfg)
	fakeSystemctl(t, true)
	unit := filepath.Join(cfg, "systemd", "user", systemdUnitName)
	var out bytes.Buffer
	if err := runAutostart([]string{"enable", "--systemd"}, &out); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(unit); err != nil {
		t.Fatalf("unit not kept: %v", err)
	}
	for _, want := range []string{"systemctl --user daemon-reload", "systemctl --user enable " + systemdUnitName, "loginctl enable-linger $USER"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, out.String())
		}
	}
	// Disable still removes the unit when systemctl fails.
	out.Reset()
	if err := runAutostart([]string{"disable"}, &out); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(unit); !os.IsNotExist(err) {
		t.Fatalf("unit still present: %v", err)
	}
}

func TestSystemdQuote(t *testing.T) {
	for in, want := range map[string]string{
		"plain": "plain",
		"a b":   `"a b"`,
		`a"b`:   `"a\"b"`,
		`c\d`:   `"c\\d"`,
		"50%":   "50%%",
		"$HOME": "$$HOME",
		"":      `""`,
	} {
		if got := systemdQuote(in); got != want {
			t.Errorf("systemdQuote(%q) = %q, want %q", in, got, want)
		}
	}
}
