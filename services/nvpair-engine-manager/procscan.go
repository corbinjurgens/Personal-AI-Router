// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"strings"
)

// processLister returns process image names or paths for diagnosis only.
type processLister func(ctx context.Context) ([]string, error)

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

func anyProcessMatches(images []string, names []string) bool {
	for _, name := range names {
		name = normalizeProcessName(name)
		if name == "" {
			continue
		}
		for _, image := range images {
			if strings.EqualFold(normalizeProcessName(image), name) {
				return true
			}
		}
	}
	return false
}
