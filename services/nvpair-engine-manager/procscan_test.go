// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "testing"

func TestNormalizeProcessName(t *testing.T) {
	test := func(name, input, want string) {
		t.Run(name, func(t *testing.T) {
			if got := normalizeProcessName(input); got != want {
				t.Fatalf("normalizeProcessName(%q) = %q; want %q", input, got, want)
			}
		})
	}
	test("macOS app path with spaces", "/Applications/llama.cpp Custom.app/Contents/MacOS/llama.cpp Custom", "llama.cpp Custom")
	test("Windows app path", `C:\Users\u\AppData\Local\Programs\llama.cpp Custom\llama.cpp Custom.exe`, "llama.cpp Custom")
	test("bare daemon", "llama-server", "llama-server")
	test("mixed case extension", " lLaMa.CpP cUsToM.ExE \t", "lLaMa.CpP cUsToM")
	test("mixed path separators", `C:\apps/llama.cpp Custom.EXE`, "llama.cpp Custom")
	test("extension only at end", "llama-server.exe.helper", "llama-server.exe.helper")
	test("helper retains identity", "llama.cpp Custom Helper (Renderer)", "llama.cpp Custom Helper (Renderer)")
	test("empty input", " \t", "")
}
