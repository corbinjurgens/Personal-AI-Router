// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"strings"
)

type processRecord struct {
	PID   int
	Image string
}

func normalizeProcessName(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.LastIndexAny(s, `/\`); i >= 0 {
		s = s[i+1:]
	}
	s = strings.TrimSpace(s)
	if len(s) >= len(".exe") && strings.EqualFold(s[len(s)-len(".exe"):], ".exe") {
		s = s[:len(s)-len(".exe")]
	}
	return strings.TrimSpace(s)
}
