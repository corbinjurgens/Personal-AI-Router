// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import "os"

func prepareStdio() (*os.File, *os.File, error) {
	// Closing Windows pipe files cancels pending I/O. Anonymous pipes do not
	// support Unix deadline probes, so retain the inherited owners.
	return os.Stdin, os.Stdout, nil
}
