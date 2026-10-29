// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package durablefile

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSyncCreatedDirParentAndError(t *testing.T) {
	want := errors.New("sync failed")
	var got string
	r := Replacer{SyncDir: func(path string) error { got = path; return want }}
	path := filepath.Join(t.TempDir(), "child")
	if err := r.SyncCreatedDir(path); !errors.Is(err, want) || !strings.Contains(err.Error(), filepath.Dir(path)) {
		t.Fatalf("error = %v, want %v", err, want)
	}
	if got != filepath.Dir(path) {
		t.Fatalf("synced %q, want %q", got, filepath.Dir(path))
	}
}

func TestSyncCreatedDirDefault(t *testing.T) {
	path := filepath.Join(t.TempDir(), "child")
	if err := os.Mkdir(path, 0700); err != nil {
		t.Fatal(err)
	}
	if err := (Replacer{}).SyncCreatedDir(path); err != nil {
		t.Fatal(err)
	}
}
