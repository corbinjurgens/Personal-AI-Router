// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"testing"
)

func TestLlamaArtifactPresetLenient(t *testing.T) {
	const other = "[other]\nmmproj-auto = true\nfit-target = 1024\nalias = x\nfoo-bar = 1\n"
	t.Run("unrelated native options do not block selected artifacts", func(t *testing.T) {
		f := namedCustomCopyFixture(t, "wanted", 300_000, 4)
		writeTestFile(t, f.preset, []byte(other+"[wanted]\nmodel = "+f.artifact+"\nn-cpu-moe = 21\nload-on-startup = true\n"))
		list, err := f.source.modelSourceFiles(context.Background(), "llamacpp", "wanted")
		if err != nil {
			t.Fatal(err)
		}
		if len(list.Files) != 1 || list.Files[0].configured != f.artifact || list.Files[0].Role != "model" {
			t.Fatalf("selected artifacts = %+v", list.Files)
		}
	})
	t.Run("missing model retains field reason", func(t *testing.T) {
		f := namedCustomCopyFixture(t, "wanted", 300_000, 4)
		writeTestFile(t, f.preset, []byte("[wanted]\nn-cpu-moe = 21\n"))
		_, err := f.source.modelSourceFiles(context.Background(), "llamacpp", "wanted")
		var field *ServingFieldError
		if !errors.As(err, &field) || field.Reason != "Every model preset requires an existing local model path." {
			t.Fatalf("resolver error = %v", err)
		}
	})
	t.Run("CLI context overrides wildcard context", func(t *testing.T) {
		sections, err := readLlamaArtifactPreset([]byte("[*]\nctx-size = 65536\n[wanted]\nmodel = model.gguf\n"))
		if err != nil {
			t.Fatal(err)
		}
		cli, env, err := llamaArtifactLaunchOptions(launchCommand{Args: []string{"--ctx-size", "8192"}}, nil)
		if err != nil {
			t.Fatal(err)
		}
		merged := map[string]string{}
		for _, options := range []map[string]string{env, sections["*"].options, sections["wanted"].options, cli} {
			for key, value := range options {
				merged[key] = value
			}
		}
		if merged["ctx-size"] != "8192" {
			t.Fatalf("context = %q, want 8192", merged["ctx-size"])
		}
	})
	t.Run("serving compilation remains strict", func(t *testing.T) {
		if _, err := parseLlamaServingPreset([]byte(other)); err == nil {
			t.Fatal("strict parser accepted unsupported native options")
		}
	})
	t.Run("structural refusals stay on their section", func(t *testing.T) {
		for _, bad := range []string{
			"[other]\nmodel = a.gguf\nmodel = b.gguf\n",
			"[other]\nmodel = a.gguf\n[other]\nmodel = b.gguf\n",
			"[other]\nmodel = a#b.gguf\n",
			"[other]\nctx-size = invalid\n",
		} {
			sections, err := readLlamaArtifactPreset([]byte(bad + "[wanted]\nmodel = wanted.gguf\n"))
			if err != nil {
				t.Fatal(err)
			}
			var field *ServingFieldError
			if !errors.As(sections["other"].err, &field) || sections["wanted"].err != nil || sections["wanted"].options["model"] != "wanted.gguf" {
				t.Fatalf("section isolation failed: %+v", sections)
			}
		}
	})
	t.Run("launch reader preserves argument boundaries", func(t *testing.T) {
		cli, _, err := llamaArtifactLaunchOptions(launchCommand{Args: []string{"--log-file", "--models-preset", "--ctx-size=8192", "--ctx-size", "8192", "--mmproj-auto"}}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(cli) != 1 || cli["ctx-size"] != "8192" {
			t.Fatalf("artifact options = %+v", cli)
		}
	})
}
