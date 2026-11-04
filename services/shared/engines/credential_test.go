// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engines

import (
	"strings"
	"testing"
)

func TestValidCredential(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  bool
	}{
		{"empty", "", true},
		{"normal", "abc-123_~", true},
		{"512 bytes", strings.Repeat("a", 512), true},
		{"513 bytes", strings.Repeat("a", 513), false},
		{"space", "a b", false},
		{"newline", "a\nb", false},
		{"tab", "a\tb", false},
		{"non-ASCII", "é", false},
		{"DEL", "a\x7f", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ValidCredential(tc.input); got != tc.want {
				t.Errorf("ValidCredential = %t, want %t", got, tc.want)
			}
		})
	}
}
