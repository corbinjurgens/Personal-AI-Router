// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestModelCopyDestinationBelowJunction(t *testing.T) {
	base := t.TempDir()
	actual := filepath.Join(base, "actual")
	root := filepath.Join(actual, "models")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(base, "linked")
	junction(t, link, actual)
	configured := filepath.Join(link, "models")
	if err := safeCopyDestination(configured, filepath.Join(configured, "alias", "model.gguf")); err != nil {
		t.Fatalf("private destination below junction refused: %v", err)
	}
}
