// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package appdir

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestPreviousDirUsesSameBase(t *testing.T) {
	current, err := Dir()
	if err != nil {
		t.Fatal(err)
	}
	previous, err := PreviousDir()
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(filepath.Dir(filepath.Dir(current)), "NVIDIA Corporation", "PAIR")
	if previous != want {
		t.Fatalf("PreviousDir() = %q, want %q", previous, want)
	}
}

func TestResolutionIsPureAndUsesDesktopNames(t *testing.T) {
	base := t.TempDir()
	t.Setenv("HOME", base)
	t.Setenv("XDG_CONFIG_HOME", base)
	t.Setenv("LOCALAPPDATA", base)
	t.Setenv("APPDATA", filepath.Join(base, "roaming"))
	wantBase := base
	if runtime.GOOS == "darwin" {
		wantBase = filepath.Join(base, "Library", "Application Support")
	}
	current, err := Dir()
	if err != nil {
		t.Fatal(err)
	}
	previous, err := PreviousDir()
	if err != nil {
		t.Fatal(err)
	}
	child, err := Path("engine-bin")
	if err != nil {
		t.Fatal(err)
	}
	if current != filepath.Join(wantBase, "Nvidia Corporation", "Personal AI Router") || previous != filepath.Join(wantBase, "NVIDIA Corporation", "PAIR") || child != filepath.Join(current, "engine-bin") {
		t.Fatalf("current=%q previous=%q child=%q", current, previous, child)
	}
	entries, err := os.ReadDir(base)
	if err != nil || len(entries) != 0 {
		t.Fatalf("resolution mutated base: %v %v", entries, err)
	}
}

func TestResolutionErrorsPropagate(t *testing.T) {
	t.Setenv("HOME", "")
	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("LOCALAPPDATA", "")
	t.Setenv("APPDATA", "")
	if _, err := Dir(); err == nil {
		t.Fatal("Dir accepted unavailable base")
	}
	if _, err := PreviousDir(); err == nil {
		t.Fatal("PreviousDir accepted unavailable base")
	}
	if _, err := Path("child"); err == nil {
		t.Fatal("Path accepted unavailable base")
	}
}
