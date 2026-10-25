// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package pipeio

import (
	"io"
	"os"
)

// NewReader takes ownership of a pipe reader. Closing an os.File interrupts its
// active pipe read on this platform without a separate cancellation owner.
func NewReader(file *os.File) (io.ReadCloser, error) { return file, nil }
