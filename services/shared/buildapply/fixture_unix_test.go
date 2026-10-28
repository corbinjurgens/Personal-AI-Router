//go:build !windows

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"path/filepath"
	"testing"
)

// packageFixtureTempDir resolves the test root: macOS TMPDIR sits under the
// /var symlink, which the linked-ancestor checks correctly refuse.
func packageFixtureTempDir(t *testing.T) string {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return root
}
