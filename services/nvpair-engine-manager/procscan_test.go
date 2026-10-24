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
	test("macOS app path with spaces", "/Applications/LM Studio.app/Contents/MacOS/LM Studio", "LM Studio")
	test("Windows app path", `C:\Users\u\AppData\Local\Programs\LM Studio\LM Studio.exe`, "LM Studio")
	test("bare daemon", "llmster", "llmster")
	test("mixed case extension", " LM sTuDiO.ExE \t", "LM sTuDiO")
	test("mixed path separators", `C:\apps/LM Studio.EXE`, "LM Studio")
	test("extension only at end", "llmster.exe.helper", "llmster.exe.helper")
	test("helper retains identity", "LM Studio Helper (Renderer)", "LM Studio Helper (Renderer)")
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
	test("macOS app path with spaces", []string{"/Applications/LM Studio.app/Contents/MacOS/LM Studio"}, []string{"LM Studio"}, true)
	test("Windows app path", []string{`C:\Users\u\AppData\Local\Programs\LM Studio\LM Studio.exe`}, []string{"LM Studio"}, true)
	test("bare daemon", []string{"llmster"}, []string{"LM Studio", "llmster"}, true)
	test("mixed case", []string{"LM sTuDiO.EXE"}, []string{"lm studio"}, true)
	test("normalizes configured name", []string{"llmster"}, []string{` C:\tools\LLMSTER.ExE `}, true)
	test("renderer helper is not app", []string{"LM Studio Helper (Renderer)"}, []string{"LM Studio"}, false)
	test("CLI is not app", []string{"lms"}, []string{"LM Studio", "llmster"}, false)
	test("empty names", []string{"llmster"}, nil, false)
	test("empty images", nil, []string{"llmster"}, false)
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
