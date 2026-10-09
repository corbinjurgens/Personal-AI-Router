// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"fmt"
	"path/filepath"
	"sync/atomic"
)

var testPipeSeq atomic.Int64

// testEndpoint is a pipe name unique to this test process and call; the dir
// only feeds the name so parallel packages cannot collide.
func testEndpoint(dir string) string {
	return fmt.Sprintf(`\\.\pipe\nvpair-service-test-%s-%d`, filepath.Base(dir), testPipeSeq.Add(1))
}
