// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engines

import "testing"

func TestNormalizeModelName(t *testing.T) {
	test := func(name, input, naming, want string) {
		t.Run(name, func(t *testing.T) {
			if got := NormalizeModelName(input, naming); got != want {
				t.Fatalf("NormalizeModelName(%q, %q) = %q, want %q", input, naming, got, want)
			}
		})
	}
	test("trims exact names", " org/model \t", ModelNamingExact, "org/model")
	test("exact keeps untagged names", "org/model", ModelNamingExact, "org/model")
}
