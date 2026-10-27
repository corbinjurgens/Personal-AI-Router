// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package durablefile

import "os"

// Open opens path for reading. A held reader retains the opened file across
// replacement; callers that require current authority must validate fresh reads.
func Open(path string) (*os.File, error) {
	return os.Open(path)
}

// ReadFile reads path and closes its handle before returning.
func ReadFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}
