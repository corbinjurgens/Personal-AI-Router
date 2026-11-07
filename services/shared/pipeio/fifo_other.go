// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !darwin && !windows

package pipeio

import "os"

func preservePipeEOF(*os.File) (bool, error) { return false, nil }
