// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func linkedTestExecutor(t *testing.T) *Executor {
	t.Helper()
	ex := copyTestExecutor(t, "llamacpp", t.TempDir(), false)
	ex.baseDir = t.TempDir()
	return ex
}

func TestLinkedModelRefusals(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "Model-Q4_K_M.gguf")
	writeSourceGGUF(t, model)
	projector := filepath.Join(dir, "mmproj-model.gguf")
	writeSourceGGUF(t, projector)
	text := filepath.Join(dir, "notes.txt")
	writeSourceFile(t, text, []byte("not a model"))
	fake := filepath.Join(dir, "fake.gguf")
	writeSourceFile(t, fake, []byte("NOPE"))
	second := filepath.Join(dir, "Big-00002-of-00003.gguf")
	writeSourceGGUF(t, second)
	odd := filepath.Join(t.TempDir(), "models #1", "Model.gguf")
	writeSourceGGUF(t, odd)
	upper := filepath.Join(t.TempDir(), "Upper.GGUF")
	writeSourceGGUF(t, upper)

	ex := linkedTestExecutor(t)
	for name, path := range map[string]string{
		"relative path":     "Model-Q4_K_M.gguf",
		"missing file":      filepath.Join(dir, "missing.gguf"),
		"not gguf":          text,
		"bad magic":         fake,
		"projector":         projector,
		"later shard":       second,
		"preset character":  odd,
		"upper-case suffix": upper,
		"directory":         dir + string(filepath.Separator) + "sub.gguf",
		"overlong":          filepath.Join(dir, strings.Repeat("m", maxLinkedPath)+".gguf"),
	} {
		t.Run(name, func(t *testing.T) {
			if id, err := ex.linkModel("llamacpp", path); err == nil {
				t.Fatalf("linked %q as %q", path, id)
			}
		})
	}
	if _, err := os.Stat(ex.linkedModelsPath()); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("refusals wrote the link list: %v", err)
	}
}

func TestLinkedModelIDsPersistAndStayUnique(t *testing.T) {
	a := filepath.Join(t.TempDir(), "Model-Q4_K_M.gguf")
	b := filepath.Join(t.TempDir(), "Model-Q4_K_M.gguf")
	split := filepath.Join(t.TempDir(), "Big-00001-of-00002.gguf")
	for _, path := range []string{a, b, split} {
		writeSourceGGUF(t, path)
	}
	projector := filepath.Join(filepath.Dir(a), "mmproj-model.gguf")
	writeSourceGGUF(t, projector)

	ex := linkedTestExecutor(t)
	ids := map[string]string{}
	for _, path := range []string{a, b, split} {
		id, err := ex.linkModel("llamacpp", path)
		if err != nil {
			t.Fatalf("link %s: %v", path, err)
		}
		ids[path] = id
	}
	if ids[a] != "local/Model-Q4_K_M" || ids[b] != "local/Model-Q4_K_M-2" || ids[split] != "local/Big" {
		t.Fatalf("ids = %v", ids)
	}
	if again, err := ex.linkModel("llamacpp", a); err != nil || again != ids[a] {
		t.Fatalf("relink = %q, %v", again, err)
	}

	reloaded := &Executor{baseDir: ex.baseDir}
	models := reloaded.sourceModelsFor(context.Background(), "llamacpp")
	if len(models) != 3 || models[0].path != a || models[0].mmproj != projector || models[1].mmproj != "" {
		t.Fatalf("reloaded models = %+v", models)
	}
	if other := reloaded.sourceModelsFor(context.Background(), "strata"); len(other) != 0 {
		t.Fatalf("another engine sees llama.cpp links: %+v", other)
	}
	if err := reloaded.unlinkModel("llamacpp", ids[b]); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(b); err != nil {
		t.Fatalf("unlink touched the file: %v", err)
	}
	if err := reloaded.unlinkModel("llamacpp", ids[b]); err == nil {
		t.Fatal("unlinked an ID twice")
	}
}

func TestLinkedModelMalformedListIsReportedNotReplaced(t *testing.T) {
	ex := linkedTestExecutor(t)
	writeSourceFile(t, ex.linkedModelsPath(), []byte("{not json"))
	model := filepath.Join(t.TempDir(), "Model.gguf")
	writeSourceGGUF(t, model)
	if _, err := ex.linkModel("llamacpp", model); err == nil {
		t.Fatal("linked into a malformed list")
	}
	if res := ex.sourceModelsList(context.Background(), "llamacpp"); res.Error == "" {
		t.Fatal("listing hid the malformed list")
	}
	data, err := os.ReadFile(ex.linkedModelsPath())
	if err != nil || string(data) != "{not json" {
		t.Fatalf("malformed list was replaced: %q, %v", data, err)
	}
}

// Peers read the listing's error, so an unreadable list is reported without
// the host path the read error carries.
func TestLinkedModelUnreadableListNamesNoPath(t *testing.T) {
	ex := linkedTestExecutor(t)
	if err := os.MkdirAll(ex.linkedModelsPath(), 0o700); err != nil {
		t.Fatal(err)
	}
	res := ex.sourceModelsList(context.Background(), "llamacpp")
	if res.Error == "" {
		t.Fatal("listing hid the unreadable list")
	}
	for _, dir := range []string{ex.baseDir, filepath.ToSlash(ex.baseDir)} {
		if strings.Contains(res.Error, dir) {
			t.Fatalf("listing error names a host path: %q", res.Error)
		}
	}
}

func TestSourceModelsForMergesSourcesAndLinks(t *testing.T) {
	root := t.TempDir()
	writeSourceGGUF(t, filepath.Join(root, "pub", "repo", "Model-Q4_K_M.gguf"))
	ex := linkedTestExecutor(t)
	ex.modelSources = []modelSource{{name: "lmstudio", layout: sourceLayoutGGUFTree, root: root}}
	kept := filepath.Join(t.TempDir(), "Kept.gguf")
	gone := filepath.Join(t.TempDir(), "Gone.gguf")
	writeSourceGGUF(t, kept)
	writeSourceGGUF(t, gone)
	for _, path := range []string{kept, gone} {
		if _, err := ex.linkModel("llamacpp", path); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Remove(gone); err != nil {
		t.Fatal(err)
	}
	// A hand-edited entry reusing a source's ID loses to the source.
	ex.linked.mu.Lock()
	entries, _ := ex.linkedEntries()
	clash := linkedModel{Engine: "llamacpp", ID: "lmstudio/pub/repo/Model-Q4_K_M", Path: kept}
	if err := ex.saveLinkedEntries(append(append([]linkedModel(nil), entries...), clash)); err != nil {
		t.Fatal(err)
	}
	ex.linked.mu.Unlock()

	var ids []string
	for _, model := range ex.sourceModelsFor(context.Background(), "llamacpp") {
		ids = append(ids, model.id+"="+filepath.Base(model.path))
	}
	want := []string{"lmstudio/pub/repo/Model-Q4_K_M=Model-Q4_K_M.gguf", "local/Kept=Kept.gguf"}
	if fmt.Sprint(ids) != fmt.Sprint(want) {
		t.Fatalf("source models = %v, want %v", ids, want)
	}
}

func TestSourceModelShards(t *testing.T) {
	dir := t.TempDir()
	got := sourceModelShards(sourceModel{path: filepath.Join(dir, "Big-00001-of-00003.gguf")})
	want := []string{
		filepath.Join(dir, "Big-00001-of-00003.gguf"),
		filepath.Join(dir, "Big-00002-of-00003.gguf"),
		filepath.Join(dir, "Big-00003-of-00003.gguf"),
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("shards = %v", got)
	}
	blob := filepath.Join(dir, "blobs", "sha256-"+strings.Repeat("a", 64))
	if got := sourceModelShards(sourceModel{path: blob}); len(got) != 1 || got[0] != blob {
		t.Fatalf("blob shards = %v", got)
	}
}

func TestGGUFArtifactPathAcceptsOllamaBlobs(t *testing.T) {
	hex := fmt.Sprintf("%x", sha256.Sum256([]byte("x")))
	for path, want := range map[string]bool{
		filepath.Join("models", "a.gguf"):                      true,
		filepath.Join("ollama", "blobs", "sha256-"+hex):        true,
		filepath.Join("ollama", "other", "sha256-"+hex):        false,
		filepath.Join("ollama", "blobs", "sha256-"+hex+".bin"): false,
		filepath.Join("models", "a.bin"):                       false,
	} {
		if got := ggufArtifactPath(path); got != want {
			t.Errorf("ggufArtifactPath(%q) = %t, want %t", path, got, want)
		}
	}
}

// Source and linked models resolve to their own files, so a profile serves them
// in place; a removed link stops resolving.
func TestSourceModelsResolveInPlace(t *testing.T) {
	store := t.TempDir()
	ex := copyTestExecutor(t, "llamacpp", store, false)
	ex.baseDir = t.TempDir()
	studio := t.TempDir()
	studioModel := filepath.Join(studio, "pub", "repo", "Studio-Q4_K_M.gguf")
	writeSourceGGUF(t, studioModel)
	writeSourceGGUF(t, filepath.Join(studio, "pub", "repo", "mmproj-studio.gguf"))
	ollama := t.TempDir()
	blobs := writeSourceOllamaModel(t, ollama, "registry.ollama.ai/library/tiny/latest", map[string]string{ollamaModelLayer: "GGUF ollama blob"})
	ex.modelSources = []modelSource{
		{name: "ollama", layout: sourceLayoutOllama, root: ollama},
		{name: "lmstudio", layout: sourceLayoutGGUFTree, root: studio},
	}
	linkDir := t.TempDir()
	first := filepath.Join(linkDir, "Split-00001-of-00002.gguf")
	secondShard := filepath.Join(linkDir, "Split-00002-of-00002.gguf")
	writeSourceGGUF(t, first)
	writeSourceGGUF(t, secondShard)
	linked, err := ex.linkModel("llamacpp", first)
	if err != nil {
		t.Fatal(err)
	}

	st, modelStore, err := ex.modelStoreFor("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	snapshot, command, err := copyLaunchSnapshot(st)
	if err != nil {
		t.Fatal(err)
	}
	access := servingFileAccess{sources: ex.sourceModelsHook("llamacpp")}
	for model, want := range map[string][]string{
		"lmstudio/pub/repo/Studio-Q4_K_M": {studioModel},
		"ollama/tiny@latest":              {blobs[ollamaModelLayer]},
		linked:                            {first, secondShard},
	} {
		t.Run(model, func(t *testing.T) {
			list, _, err := resolveLlamaArtifacts(context.Background(), snapshot, command, modelStore, model, access)
			if err != nil {
				t.Fatalf("resolve: %v", err)
			}
			var got []string
			for _, file := range list.Files {
				if file.Role != "model" {
					t.Errorf("%s has role %q; typed loads map model files only", file.configured, file.Role)
				}
				got = append(got, file.configured)
			}
			if fmt.Sprint(got) != fmt.Sprint(want) {
				t.Fatalf("files = %v, want %v", got, want)
			}
		})
	}
	t.Run("without the hook", func(t *testing.T) {
		if _, _, err := resolveLlamaArtifacts(context.Background(), snapshot, command, modelStore, linked, servingFileAccess{}); err == nil {
			t.Fatal("source model resolved with no source list")
		}
	})
	t.Run("removed link", func(t *testing.T) {
		if err := os.Remove(first); err != nil {
			t.Fatal(err)
		}
		if _, _, err := resolveLlamaArtifacts(context.Background(), snapshot, command, modelStore, linked, access); err == nil {
			t.Fatal("removed linked file still resolves")
		}
	})
}
