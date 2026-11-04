// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"reflect"
	"strings"
	"testing"
)

func TestParseLaunchText(t *testing.T) {
	test := func(name, text string, want []string) {
		t.Run(name, func(t *testing.T) {
			got, err := parseLaunchText(text)
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("got %#v, %v; want %#v", got, err, want)
			}
			normalized, err := formatLaunchText(got)
			if err != nil {
				t.Fatal(err)
			}
			again, err := parseLaunchText(normalized)
			if err != nil || !reflect.DeepEqual(again, want) {
				t.Fatalf("round trip changed arguments: %#v, %v", again, err)
			}
		})
	}
	test("empty arguments", "", nil)
	test("blank arguments", " \t\r\n", nil)
	test("env and Windows executable", `LLAMA_ARG_PORT=8081 "C:\Program Files\llama.cpp\llama-server.exe" --jinja`, []string{"LLAMA_ARG_PORT=8081", `C:\Program Files\llama.cpp\llama-server.exe`, "--jinja"})
	test("Windows backslashes", `C:\tools\llama-server.exe --path 'C:\new\tools\'`, []string{`C:\tools\llama-server.exe`, "--path", `C:\new\tools\`})
	test("UNC", `'\\server\share\llama-server.exe' server start`, []string{`\\server\share\llama-server.exe`, "server", "start"})
	test("quoted segments", `llama-server --name="two words" --empty=''`, []string{"llama-server", "--name=two words", "--empty="})
	test("empty arguments", `llama-server "" ''`, []string{"llama-server", "", ""})
	test("escaped quotes", `llama-server "a\"b\\c"`, []string{"llama-server", `a"b\c`})
	test("multiline", "llama-server\r\nserver\tstart --name '日本語\n模型'", []string{"llama-server", "server", "start", "--name", "日本語\n模型"})
	test("literal expansions", `llama-server $HOME %USERPROFILE% {port} '~' '$(touch nope)'`, []string{"llama-server", "$HOME", "%USERPROFILE%", "{port}", "~", "$(touch nope)"})
	test("literal operators", "llama-server '|&;<>`'", []string{"llama-server", "|&;<>`"})
	test("terminator", `llama-server -- --port=9000`, []string{"llama-server", "--", "--port=9000"})
}

func TestParseLaunchTextRejectsInvalidInput(t *testing.T) {
	for _, text := range []string{
		`llama-server "unterminated`, `llama-server 'unterminated`, "llama-server \x00argument",
		"llama-server \x7f", "llama-server \xff", `llama-server | other`, `llama-server > file`, `llama-server < file`,
		`llama-server && other`, `llama-server ; other`, "llama-server `other`", `llama-server $(other)`,
		strings.Repeat("x", maxLaunchTextBytes+1), strings.Repeat("x ", maxLaunchTokens+1),
	} {
		if _, err := parseLaunchText(text); err == nil {
			t.Fatalf("invalid input accepted (%d bytes)", len(text))
		}
	}
	if _, err := parseLaunchText(strings.Repeat("x", maxLaunchTextBytes)); err != nil {
		t.Fatalf("exact byte limit rejected: %v", err)
	}
	if _, err := parseLaunchText(strings.Repeat("x ", maxLaunchTokens)); err != nil {
		t.Fatalf("exact token limit rejected: %v", err)
	}
}

func FuzzLaunchTextRoundTrip(f *testing.F) {
	for _, seed := range []string{
		`llama-server --port=8081`, `"C:\Program Files\llama.cpp\llama-server.exe" --jinja`,
		`llama-server '' "日本語" '$HOME' '{port}'`, `llama-server --name="two words"`,
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, text string) {
		tokens, err := parseLaunchText(text)
		if err != nil {
			return
		}
		formatted, err := formatLaunchText(tokens)
		if err != nil {
			// Canonical escaping may push a valid input beyond the byte limit.
			return
		}
		again, err := parseLaunchText(formatted)
		if err != nil || !reflect.DeepEqual(again, tokens) {
			t.Fatalf("round trip changed tokens: %#v -> %#v (%v)", tokens, again, err)
		}
		stable, err := formatLaunchText(again)
		if err != nil || stable != formatted {
			t.Fatalf("normalization is not stable: %v", err)
		}
	})
}
