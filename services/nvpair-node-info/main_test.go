// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"testing"

	"nvpair-shared/secretenv/secretenvtest"
)

// TestMain lets the GPU probe tests start this test binary as a sentinel child.
func TestMain(m *testing.M) {
	secretenvtest.RunChild()
	os.Exit(m.Run())
}
