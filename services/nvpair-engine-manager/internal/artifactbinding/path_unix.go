// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package artifactbinding

import "path/filepath"

func Resolve(path string) (string, error) { return filepath.EvalSymlinks(path) }
