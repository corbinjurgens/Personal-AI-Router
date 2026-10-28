//go:build !windows

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "testing"

func buildApplyFixtureTempDir(t *testing.T) string {
	t.Helper()
	return t.TempDir()
}
