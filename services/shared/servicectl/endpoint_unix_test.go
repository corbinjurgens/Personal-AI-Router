// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package servicectl

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestLongAppDirFallsBackToShortSocket: a data dir too long for a Unix socket
// (macOS with a long user name) must not make the service unreachable.
func TestLongAppDirFallsBackToShortSocket(t *testing.T) {
	long := filepath.Join(t.TempDir(), strings.Repeat("d", 120))
	t.Setenv("HOME", long)
	t.Setenv("XDG_CONFIG_HOME", long)
	tmp, err := os.MkdirTemp("/tmp", "sc")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(tmp) })
	t.Setenv("TMPDIR", tmp)
	t.Setenv(EndpointEnv, "")

	got, err := Endpoint()
	if err != nil {
		t.Fatalf("Endpoint: %v", err)
	}
	if len(got) > maxSocketPath || !strings.HasPrefix(got, tmp) {
		t.Fatalf("Endpoint = %q, want a short path under %s", got, tmp)
	}
	fi, err := os.Stat(filepath.Dir(got))
	if err != nil || fi.Mode().Perm() != 0o700 {
		t.Fatalf("socket dir mode = %v, %v; want 0700", fi.Mode().Perm(), err)
	}

	// A directory other users can enter is refused rather than used.
	if err := os.Chmod(filepath.Dir(got), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := Endpoint(); err == nil {
		t.Fatal("Endpoint accepted a socket directory open to other users")
	}
}
