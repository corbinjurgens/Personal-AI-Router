// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package clustertrust

import "os"

func openAuthorityFile(path string) (*os.File, error) {
	return os.Open(path)
}

func readAuthorityFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}
