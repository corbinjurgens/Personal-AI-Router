// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package artifactbinding

import "testing"

func TestBindingSpelling(t *testing.T) {
	for _, tc := range []struct{ name, input, want string }{
		{"DOS", `\\?\C:\models\model.gguf`, `C:\models\model.gguf`},
		{"UNC", `\\?\UNC\server\share\model.gguf`, `\\server\share\model.gguf`},
		{"already DOS", `C:\models\model.gguf`, `C:\models\model.gguf`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := bindingSpelling(tc.input); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}
