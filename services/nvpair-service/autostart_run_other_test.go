// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"nvpair-shared/servicectl"
)

func serviceTestProcessIdentity() (string, uint64, error) {
	return "", 0, nil
}

func TestAutostartRunRefusesUnsupportedPlatformWithoutChangingRegistration(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv(servicectl.EndpointEnv, testEndpoint(dir))
	registration, err := autostartLocation()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(registration), 0o700); err != nil {
		t.Fatal(err)
	}
	const existing = "existing registration must survive\n"
	if err := os.WriteFile(registration, []byte(existing), 0o600); err != nil {
		t.Fatal(err)
	}
	logDir := filepath.Join(dir, "logs")
	var stdout, stderr bytes.Buffer
	code := run([]string{"autostart", "run", "--broker-path", filepath.Join(dir, "missing-broker"), "--log-dir", logDir}, &stdout, &stderr)
	if code != 1 || stdout.Len() != 0 || !strings.Contains(stderr.String(), "only supported on Windows") {
		t.Fatalf("exit=%d stdout=%q stderr=%q, want Windows-only refusal", code, stdout.String(), stderr.String())
	}
	after, err := os.ReadFile(registration)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != existing {
		t.Fatalf("unsupported start changed registration: %q", after)
	}
	if _, err := os.Stat(logDir); !os.IsNotExist(err) {
		t.Fatalf("unsupported start touched logs: %v", err)
	}
}
