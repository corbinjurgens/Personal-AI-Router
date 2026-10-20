// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import "os"

func prepareStdio() (*os.File, *os.File, error) {
	// Windows pipe File.Close cancels pending I/O; anonymous pipes do not
	// support the Unix deadline probes. Keep the actual inherited owners.
	return os.Stdin, os.Stdout, nil
}
