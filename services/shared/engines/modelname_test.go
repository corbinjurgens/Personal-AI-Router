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
	test("implied latest", " model \t", ModelNamingImpliedLatestTag, "model:latest")
	test("registry port is not a tag", "localhost:5000/org/model", ModelNamingImpliedLatestTag, "localhost:5000/org/model:latest")
	test("explicit tag", " org/model:v1 ", ModelNamingImpliedLatestTag, "org/model:v1")
	test("explicit digest", "org/model@sha256:abc", ModelNamingImpliedLatestTag, "org/model@sha256:abc")
	test("trailing slash", "org/", ModelNamingImpliedLatestTag, "org/")
	test("empty", " \t", ModelNamingImpliedLatestTag, "")
	test("alias exemption belongs to callers", "weak", ModelNamingImpliedLatestTag, "weak:latest")
}
