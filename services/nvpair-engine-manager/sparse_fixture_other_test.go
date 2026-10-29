//go:build !windows

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"testing"
)

func prepareSparseFixture(t *testing.T, file *os.File) {}

func checkSparseFixtureAllocation(t *testing.T, file *os.File, logical int64) {}
