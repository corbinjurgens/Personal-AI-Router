// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/modelconfig"
)

func writeSourceFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeSourceGGUF(t *testing.T, path string) {
	t.Helper()
	writeSourceFile(t, path, []byte("GGUF synthetic "+filepath.Base(path)))
}

type sourceTestLayer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
}

// writeSourceOllamaModel writes a manifest at manifests/<rel> naming one blob per
// layer media type, each holding contents[mediaType].
func writeSourceOllamaModel(t *testing.T, root, rel string, contents map[string]string) map[string]string {
	t.Helper()
	blobs := map[string]string{}
	var layers []sourceTestLayer
	for _, mediaType := range []string{ollamaModelLayer, ollamaProjectorLayer, ollamaAdapterLayer} {
		content, ok := contents[mediaType]
		if !ok {
			continue
		}
		hex := fmt.Sprintf("%x", sha256.Sum256([]byte(rel+mediaType)))
		blob := filepath.Join(root, "blobs", "sha256-"+hex)
		writeSourceFile(t, blob, []byte(content))
		blobs[mediaType] = blob
		layers = append(layers, sourceTestLayer{MediaType: mediaType, Digest: "sha256:" + hex})
	}
	data, err := json.Marshal(map[string]any{"schemaVersion": 2, "layers": layers})
	if err != nil {
		t.Fatal(err)
	}
	writeSourceFile(t, filepath.Join(root, "manifests", filepath.FromSlash(rel)), data)
	return blobs
}

func TestScanModelSources(t *testing.T) {
	t.Run("ollama store", func(t *testing.T) {
		root := t.TempDir()
		library := writeSourceOllamaModel(t, root, "registry.ollama.ai/library/llama3.2/3b", map[string]string{
			ollamaModelLayer: "GGUF model", ollamaProjectorLayer: "GGUF projector",
		})
		namespaced := writeSourceOllamaModel(t, root, "registry.ollama.ai/user/model/latest", map[string]string{ollamaModelLayer: "GGUF model"})
		hosted := writeSourceOllamaModel(t, root, "hf.co/owner/repo/Q4_K_M", map[string]string{ollamaModelLayer: "GGUF model"})
		writeSourceOllamaModel(t, root, "registry.ollama.ai/library/tensors/latest", map[string]string{ollamaModelLayer: "not a GGUF file"})
		writeSourceOllamaModel(t, root, "registry.ollama.ai/library/tuned/latest", map[string]string{
			ollamaModelLayer: "GGUF model", ollamaAdapterLayer: "GGUF adapter",
		})
		missing := writeSourceOllamaModel(t, root, "registry.ollama.ai/library/gone/latest", map[string]string{ollamaModelLayer: "GGUF model"})
		if err := os.Remove(missing[ollamaModelLayer]); err != nil {
			t.Fatal(err)
		}

		got := scanModelSources(context.Background(), []modelSource{{name: "ollama", layout: sourceLayoutOllama, root: root}})
		want := []sourceModel{
			{id: "ollama/hf.co/owner/repo@Q4_K_M", format: sourceFormatGGUF, path: hosted[ollamaModelLayer]},
			{id: "ollama/llama3.2@3b", format: sourceFormatGGUF, path: library[ollamaModelLayer], mmproj: library[ollamaProjectorLayer]},
			{id: "ollama/user/model@latest", format: sourceFormatGGUF, path: namespaced[ollamaModelLayer]},
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("models = %#v\nwant %#v", got, want)
		}
	})

	t.Run("gguf tree", func(t *testing.T) {
		root := t.TempDir()
		at := func(rel string) string { return filepath.Join(root, filepath.FromSlash(rel)) }
		writeSourceGGUF(t, at("pub/repo/Model-Q4_K_M.gguf"))
		writeSourceGGUF(t, at("pub/repo/mmproj-model-f16.gguf"))
		writeSourceGGUF(t, at("pub/big/Big-Q8_0-00001-of-00002.gguf"))
		writeSourceGGUF(t, at("pub/big/Big-Q8_0-00002-of-00002.gguf"))
		writeSourceGGUF(t, at("pub/vision/Vision-Q4_K_M.gguf"))
		writeSourceGGUF(t, at("pub/vision/mmproj-a.gguf"))
		writeSourceGGUF(t, at("pub/vision/mmproj-b.gguf"))
		writeSourceFile(t, at("pub/fake/Fake-Q4_K_M.gguf"), []byte("not a GGUF file"))
		writeSourceGGUF(t, at(".hidden/repo/Hidden.gguf"))
		writeSourceGGUF(t, at("a/b/c/d/e/Deep.gguf"))
		writeSourceGGUF(t, at("a/b/c/d/Shallow.gguf"))

		got := scanModelSources(context.Background(), []modelSource{{name: "lmstudio", layout: sourceLayoutGGUFTree, root: root}})
		want := []sourceModel{
			{id: "lmstudio/a/b/c/d/Shallow", format: sourceFormatGGUF, path: at("a/b/c/d/Shallow.gguf")},
			{id: "lmstudio/pub/big/Big-Q8_0", format: sourceFormatGGUF, path: at("pub/big/Big-Q8_0-00001-of-00002.gguf")},
			{id: "lmstudio/pub/repo/Model-Q4_K_M", format: sourceFormatGGUF, path: at("pub/repo/Model-Q4_K_M.gguf"), mmproj: at("pub/repo/mmproj-model-f16.gguf")},
			{id: "lmstudio/pub/vision/Vision-Q4_K_M", format: sourceFormatGGUF, path: at("pub/vision/Vision-Q4_K_M.gguf")},
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("models = %#v\nwant %#v", got, want)
		}
	})

	t.Run("missing source holds no models", func(t *testing.T) {
		root := filepath.Join(t.TempDir(), "absent")
		got := scanModelSources(context.Background(), []modelSource{
			{name: "ollama", layout: sourceLayoutOllama, root: root},
			{name: "lmstudio", layout: sourceLayoutGGUFTree, root: root},
		})
		if len(got) != 0 {
			t.Fatalf("models = %#v, want none", got)
		}
	})

	t.Run("first source keeps a duplicate ID", func(t *testing.T) {
		first, second := t.TempDir(), t.TempDir()
		writeSourceGGUF(t, filepath.Join(first, "pub", "repo", "Model.gguf"))
		writeSourceGGUF(t, filepath.Join(second, "pub", "repo", "Model.gguf"))
		got := scanModelSources(context.Background(), []modelSource{
			{name: "lmstudio", layout: sourceLayoutGGUFTree, root: first},
			{name: "lmstudio", layout: sourceLayoutGGUFTree, root: second},
		})
		want := []sourceModel{{id: "lmstudio/pub/repo/Model", format: sourceFormatGGUF, path: filepath.Join(first, "pub", "repo", "Model.gguf")}}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("models = %#v\nwant %#v", got, want)
		}
	})
}

func TestLlamaSourcePreset(t *testing.T) {
	llama := &engineState{manifest: &Manifest{Engine: "llamacpp", Family: "llamacpp"}}
	sourceRoot := func(t *testing.T) string {
		t.Helper()
		root := t.TempDir()
		writeSourceGGUF(t, filepath.Join(root, "pub", "repo", "Model-Q4_K_M.gguf"))
		writeSourceGGUF(t, filepath.Join(root, "pub", "repo", "mmproj-model.gguf"))
		return root
	}
	executor := func(t *testing.T, roots ...string) *Executor {
		t.Helper()
		ex := &Executor{baseDir: t.TempDir()}
		for _, root := range roots {
			ex.modelSources = append(ex.modelSources, modelSource{name: "lmstudio", layout: sourceLayoutGGUFTree, root: root})
		}
		return ex
	}

	t.Run("writes the models found", func(t *testing.T) {
		root := sourceRoot(t)
		ex := executor(t, root)
		path := ex.llamaSourcePreset(context.Background(), llama, "llamacpp", Runtime{}, launchCommand{})
		if want := filepath.Join(ex.baseDir, "model-sources", "llamacpp.ini"); path != want {
			t.Fatalf("preset path = %q, want %q", path, want)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		want := "# Written by PAIR at each start: models found in this node's model sources.\n\n" +
			"[lmstudio/pub/repo/Model-Q4_K_M]\n" +
			"model = " + filepath.Join(root, "pub", "repo", "Model-Q4_K_M.gguf") + "\n" +
			"mmproj = " + filepath.Join(root, "pub", "repo", "mmproj-model.gguf") + "\n"
		if string(data) != want {
			t.Fatalf("preset =\n%s\nwant\n%s", data, want)
		}
	})

	t.Run("removes a stale preset when no models remain", func(t *testing.T) {
		ex := executor(t, t.TempDir())
		stale := filepath.Join(ex.baseDir, "model-sources", "llamacpp.ini")
		writeSourceFile(t, stale, []byte("[old]\nmodel = old.gguf\n"))
		if path := ex.llamaSourcePreset(context.Background(), llama, "llamacpp", Runtime{}, launchCommand{}); path != "" {
			t.Fatalf("preset path = %q, want none", path)
		}
		if _, err := os.Stat(stale); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("stale preset remains: %v", err)
		}
	})

	t.Run("skips paths a preset line cannot carry", func(t *testing.T) {
		root := filepath.Join(t.TempDir(), "models #1")
		writeSourceGGUF(t, filepath.Join(root, "pub", "repo", "Model.gguf"))
		ex := executor(t, root)
		if path := ex.llamaSourcePreset(context.Background(), llama, "llamacpp", Runtime{}, launchCommand{}); path != "" {
			t.Fatalf("preset path = %q, want none", path)
		}
	})

	offersNothing := func(name string, st *engineState, rt Runtime, launch launchCommand, roots int) {
		t.Run(name, func(t *testing.T) {
			var ex *Executor
			if roots == 0 {
				ex = executor(t)
			} else {
				ex = executor(t, sourceRoot(t))
			}
			if path := ex.llamaSourcePreset(context.Background(), st, "llamacpp", rt, launch); path != "" {
				t.Fatalf("preset path = %q, want none", path)
			}
		})
	}
	offersNothing("no sources configured", llama, Runtime{}, launchCommand{}, 0)
	offersNothing("launch names its own preset", llama, Runtime{}, launchCommand{Args: []string{"--models_preset=own.ini"}}, 1)
	offersNothing("environment names its own preset", llama, Runtime{}, launchCommand{Env: map[string]string{"LLAMA_ARG_MODELS_PRESET": "own.ini"}}, 1)
	offersNothing("serving preset decides its models", llama, Runtime{Serving: &engines.ServingConfig{Version: 1}}, launchCommand{}, 1)
	offersNothing("typed loads decide their models", llama, Runtime{ModelLoads: []modelconfig.ModelLoad{{ModelID: "owner/repo:Q4_K_M"}}}, launchCommand{}, 1)
	offersNothing("other engine family", &engineState{manifest: &Manifest{Engine: "strata", Family: "strata"}}, Runtime{}, launchCommand{}, 1)
}

func TestLlamaSourceModelsReachTheLaunch(t *testing.T) {
	root := t.TempDir()
	writeSourceGGUF(t, filepath.Join(root, "pub", "repo", "Model.gguf"))
	startLog := filepath.Join(t.TempDir(), "starts.jsonl")
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	manifest := testEngineManifest(fakeEngineBin)
	manifest.Family = "llamacpp"
	for key, platform := range manifest.Platforms {
		platform.Runtime = Runtime{Mode: "process", Bin: fakeEngineBin, Port: port,
			Args:  []string{"--port", "{port}"},
			Env:   map[string]string{"FAKE_START_LOG": startLog, "FAKE_ADDR": "127.0.0.1:{port}"},
			Ready: &Probe{TCP: "127.0.0.1:{port}"},
		}
		manifest.Platforms[key] = platform
	}
	ex := newTestExecutor(t, manifest)
	ex.modelSources = []modelSource{{name: "lmstudio", layout: sourceLayoutGGUFTree, root: root}}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ex.Stop("fake") })

	preset := filepath.Join(ex.baseDir, "model-sources", "fake.ini")
	assertCapturedArgs(t, startLog, []string{"--port", strconv.Itoa(port), "--models-preset", preset})
	data, err := os.ReadFile(preset)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "[lmstudio/pub/repo/Model]\n") {
		t.Fatalf("preset does not offer the source model:\n%s", data)
	}
}
