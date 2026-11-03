// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"testing"
)

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

func TestAnyProcessMatches(t *testing.T) {
	test := func(name string, images, names []string, want bool) {
		t.Run(name, func(t *testing.T) {
			if got := anyProcessMatches(images, names); got != want {
				t.Fatalf("anyProcessMatches(%q, %q) = %v; want %v", images, names, got, want)
			}
		})
	}
	test("macOS app path with spaces", []string{"/Applications/llama.cpp Custom.app/Contents/MacOS/llama.cpp Custom"}, []string{"llama.cpp Custom"}, true)
	test("Windows app path", []string{`C:\Users\u\AppData\Local\Programs\llama.cpp Custom\llama.cpp Custom.exe`}, []string{"llama.cpp Custom"}, true)
	test("bare daemon", []string{"llama-server"}, []string{"llama.cpp Custom", "llama-server"}, true)
	test("mixed case", []string{"lLaMa.CpP cUsToM.EXE"}, []string{"llama.cpp custom"}, true)
	test("normalizes configured name", []string{"llama-server"}, []string{` C:\tools\LLAMA-SERVER.ExE `}, true)
	test("renderer helper is not app", []string{"llama.cpp Custom Helper (Renderer)"}, []string{"llama.cpp Custom"}, false)
	test("CLI is not app", []string{"llama-cli"}, []string{"llama.cpp Custom", "llama-server"}, false)
	test("empty names", []string{"llama-server"}, nil, false)
	test("empty images", nil, []string{"llama-server"}, false)
	test("blank names never match", []string{"", " "}, []string{" ", ".exe"}, false)
}

func TestListProcessNamesCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	names, err := listProcessNames(ctx)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("listProcessNames error = %v; want context.Canceled", err)
	}
	if names != nil {
		t.Fatalf("listProcessNames = %q; want no names after cancellation", names)
	}
}
