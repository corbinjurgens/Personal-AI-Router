// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestRedactServingLine(t *testing.T) {
	for _, tc := range []struct{ name, line, dir, want string }{
		{"Unix path", "loading /models/x.gguf failed", "", "loading [redacted] failed"},
		{"Windows path", `loading C:\models\x.gguf failed`, "", "loading [redacted] failed"},
		{"relative path", "loading models/x.gguf failed", "", "loading [redacted] failed"},
		{"configured directory", "loading private-store failed", "private-store", "loading [redacted] failed"},
		{"token containing directory", "loading root=private-store:failed code 2", "private-store", "loading [redacted] code 2"},
		{"directory with spaces", "loading C:\\private models\\x.gguf failed", "C:\\private models", "loading [redacted] failed"},
		{"quoted path with spaces", `loading "/archive/private model.gguf" failed`, "", "loading [redacted] failed"},
		{"single-quoted path with spaces", `loading '/archive/private model.gguf' failed`, "", "loading [redacted] failed"},
		{"quoted configured directory", `loading "private store x.gguf" failed`, "private store", "loading [redacted] failed"},
		{"key path", "model=/archive/x.gguf failed", "", "[redacted] failed"},
		{"UNC path", `loading \\host\share\x.gguf failed`, "", "loading [redacted] failed"},
		{"file URL", "loading file:///archive/x.gguf failed", "", "loading [redacted] failed"},
		{"quoted text without path", `model "tiny chat" failed`, "", `model "tiny chat" failed`},
		{"contraction before quoted path", `can't load '/archive/private model.gguf' failed`, "", "can't load [redacted] failed"},
		{"contractions around path", "can't load /x failed; model's ready", "", "can't load [redacted] failed; model's ready"},
		{"possessive before quoted text", "model's file 'a b'", "", "model's file 'a b'"},
		{"quoted key path", `model='/archive/private model.gguf', failed`, "", "[redacted] failed"},
		{"unclosed quoted path", `loading "/archive/private model.gguf failed`, "", "loading [redacted]"},
		{"parenthesized quoted path", `loading ("/archive/private model.gguf") failed`, "", "loading [redacted] failed"},
		{"JSON path field", `loading {"path":"/archive/private model.gguf"} failed`, "", "loading [redacted] failed"},
		{"quoted field before another field", `loading model="/a b",size=123 failed`, "", "loading [redacted] failed"},
		{"possessive key before quoted path", `model's="/a b" failed`, "", "[redacted] failed"},
		{"apostrophe inside single-quoted path", `loading '/models/Bob's private model.gguf' failed`, "", "loading [redacted] failed"},
		{"apostrophe inside path under models directory", `loading '/models/Bob's private model.gguf' failed`, "/models", "loading [redacted] failed"},
		{"escaped quote inside path", `loading "/a \"b\" c.gguf" failed`, "", "loading [redacted] failed"},
		{"quoted text before quoted path", `model "tiny" from "/a b.gguf" failed`, "", "model [redacted] failed"},
		{"quoted path after semicolon", `loading "ok";'/models/private model.gguf' failed`, "", "loading [redacted] failed"},
		{"quoted path after non-breaking space", "loading\u00a0\"/models/private model.gguf\" failed", "", "loading [redacted] failed"},
		{"quoted path right after quoted text", `loading "ok"'/models/private model.gguf' failed`, "", "loading [redacted] failed"},
		{"double quote after word", `loading"/models/private model.gguf" failed`, "", "[redacted] failed"},
		{"single quote after word before path", `loading'/models/private model.gguf' failed`, "", "[redacted] failed"},
		{"configured directory in another case", `loading c:\my private models failed`, `C:\My Private Models`, "loading [redacted] failed"},
		{"ordinary diagnostic", "allocation failed with code 2", "", "allocation failed with code 2"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := redactServingLine(tc.line, tc.dir); got != tc.want {
				t.Fatalf("redacted line = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestServingOutputRetainsRedactedFailure(t *testing.T) {
	f := newLlamaServingFixture(t)
	st := settingsState(t, f.ex)
	st.plat.Runtime.Env["FAKE_OUTPUT"] = "loading /models/x.gguf failed"
	f.request.Settings.LaunchText += " --fail-launch"
	applyLlamaServing(t, f, approveLlamaServing(t, f))
	err := f.ex.Start(context.Background(), "fake")
	if err == nil {
		t.Fatal("failing serving engine unexpectedly started")
	}
	logs, logErr := f.ex.Logs("fake")
	if logErr != nil {
		t.Fatalf("read engine logs: %v", logErr)
	}
	found := false
	for _, line := range logs {
		if strings.Contains(line.Text, "failed") {
			found = true
		}
		if strings.Contains(line.Text, "/models/x.gguf") || strings.Contains(line.Text, f.root) {
			t.Fatalf("serving log exposed a path: %q", line.Text)
		}
	}
	if !found {
		t.Fatalf("serving failure absent from engine:logs: %+v", logs)
	}
	if !strings.Contains(err.Error(), "failed") || strings.Contains(err.Error(), "/models/x.gguf") {
		t.Fatalf("startup diagnostic lost redacted output: %v", err)
	}
}

func TestManagedProcDrainsOversizedOutput(t *testing.T) {
	proc, err := startManagedProc(fakeEngineBin, []string{"oversized-output"}, nil, nil)
	if err != nil {
		t.Fatalf("start output fixture: %v", err)
	}
	t.Cleanup(func() { proc.stop(0) })
	select {
	case <-proc.done:
		if !proc.cmd.ProcessState.Success() {
			t.Fatalf("output fixture failed: %v", proc.cmd.ProcessState)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("oversized output blocked process completion")
	}
}
