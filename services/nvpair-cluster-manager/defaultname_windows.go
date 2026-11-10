// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import "os"

// platformNameCandidates returns the Windows source for the generated name:
// the computer name.
func platformNameCandidates() []nameCandidate {
	return []nameCandidate{os.Hostname}
}
