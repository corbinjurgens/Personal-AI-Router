// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"os"

	"nvpair-shared/durablefile"
)

func openAuthorityFile(path string) (*os.File, error) {
	return durablefile.Open(path)
}

func readAuthorityFile(path string) ([]byte, error) {
	return durablefile.ReadFile(path)
}
