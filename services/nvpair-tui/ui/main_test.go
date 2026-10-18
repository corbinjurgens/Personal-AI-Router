// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"os"
	"testing"

	"nvpair-shared/secretenv/secretenvtest"
)

// TestMain lets the demo tests start this test binary as a sentinel child.
func TestMain(m *testing.M) {
	secretenvtest.RunChild()
	os.Exit(m.Run())
}
