// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package durablefile

import (
	"errors"
	"testing"
)

func TestSyncCreatedDirUnsupported(t *testing.T) {
	if err := (Replacer{}).SyncCreatedDir(t.TempDir()); !errors.Is(err, ErrDirectoryDurabilityUnsupported) {
		t.Fatalf("error = %v, want unsupported", err)
	}
	if err := (Replacer{SyncDir: func(string) error { return nil }}).SyncCreatedDir(t.TempDir()); !errors.Is(err, ErrDirectoryDurabilityUnsupported) {
		t.Fatalf("successful no-op reported supported: %v", err)
	}
	want := errors.New("injected")
	r := Replacer{SyncDir: func(string) error { return want }}
	if err := r.SyncCreatedDir(t.TempDir()); !errors.Is(err, want) {
		t.Fatalf("error = %v, want injected failure", err)
	}
}
