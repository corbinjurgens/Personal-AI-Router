// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !darwin && !windows && !linux

package main

import "os"

// platformNameCandidates returns the generic source for the generated name.
func platformNameCandidates() []nameCandidate {
	return []nameCandidate{os.Hostname}
}
