// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package durablefile

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSyncCreatedDirDoesNotCreate(t *testing.T) {
	base := t.TempDir()
	path := filepath.Join(base, "missing", "child")
	if err := (Replacer{}).SyncCreatedDir(path); err == nil {
		t.Fatal("confirmed entry beneath absent parent")
	}
	entries, err := os.ReadDir(base)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("created entries: %v", entries)
	}
}
