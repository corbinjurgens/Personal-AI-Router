// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestRotatingLogKeepsOneOldGeneration(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "logs", "broker.log")
	l, err := openRotatingLog(path, 100)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()

	line := strings.Repeat("a", 39) + "\n" // 40 bytes
	for i := 0; i < 2; i++ {
		if _, err := l.Write([]byte(line)); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := os.Stat(path + ".1"); err == nil {
		t.Fatal("rotated before reaching the limit")
	}

	// The third line would take the file to 120 bytes: rotate first.
	if _, err := l.Write([]byte(strings.Replace(line, "a", "b", 1))); err != nil {
		t.Fatal(err)
	}
	old, err := os.ReadFile(path + ".1")
	if err != nil {
		t.Fatalf("no old generation: %v", err)
	}
	cur, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(old) != 80 || len(cur) != 40 || cur[0] != 'b' {
		t.Fatalf("old=%d bytes, current=%q", len(old), cur)
	}

	// Rotating again replaces the old generation; there is never a ".2".
	for i := 0; i < 3; i++ {
		if _, err := l.Write([]byte(strings.Replace(line, "a", "c", 1))); err != nil {
			t.Fatal(err)
		}
	}
	old, _ = os.ReadFile(path + ".1")
	if len(old) != 80 || old[0] != 'b' {
		t.Fatalf("old generation after second rotation = %q", old)
	}
	if _, err := os.Stat(path + ".2"); err == nil {
		t.Fatal("kept more than one old generation")
	}
	if runtime.GOOS == "windows" {
		return // no Unix permission bits to check
	}
	for _, p := range []string{path, path + ".1"} {
		st, err := os.Stat(p)
		if err != nil {
			t.Fatal(err)
		}
		if mode := st.Mode().Perm(); mode != 0o600 {
			t.Fatalf("%s mode = %o, want 600", p, mode)
		}
	}
}

func TestRotatingLogAppendsAcrossReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "broker.log")
	l, err := openRotatingLog(path, 100)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = l.Write([]byte(strings.Repeat("x", 60)))
	_ = l.Close()

	// A restarted service picks up the existing size, so the limit holds
	// across restarts rather than resetting.
	l, err = openRotatingLog(path, 100)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	_, _ = l.Write([]byte(strings.Repeat("y", 60)))
	if _, err := os.Stat(path + ".1"); err != nil {
		t.Fatal("size before reopen was not counted")
	}
}

func TestRotatingLogWritesOversizedLineWhole(t *testing.T) {
	path := filepath.Join(t.TempDir(), "broker.log")
	l, err := openRotatingLog(path, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	big := strings.Repeat("z", 50)
	if _, err := l.Write([]byte(big)); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(path)
	if string(data) != big {
		t.Fatalf("oversized line = %d bytes", len(data))
	}
}
