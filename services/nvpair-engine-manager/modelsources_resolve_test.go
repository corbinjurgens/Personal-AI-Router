// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"nvpair-shared/modelconfig"
)

// A typed profile on a source or linked model compiles a router preset whose
// model path is the file where it is, and the profile's artifacts export as a
// valid custom bundle, an Ollama blob under a .gguf name.
func TestTypedProfilesServeSourceModelsInPlace(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	template := modelconfig.CloneLoads(rt.ModelLoads[:1])[0]
	if f.ex.baseDir == "" {
		f.ex.baseDir = t.TempDir()
	}

	studio := t.TempDir()
	studioModel := filepath.Join(studio, "pub", "repo", "Studio-Q4_K_M.gguf")
	writeInspectionGGUF(t, studioModel, "llama", 0.25, 4096)
	linkedFile := filepath.Join(t.TempDir(), "Linked.gguf")
	writeInspectionGGUF(t, linkedFile, "llama", 0.25, 4096)
	blobSource := filepath.Join(t.TempDir(), "blob.gguf")
	writeInspectionGGUF(t, blobSource, "llama", 0.25, 4096)
	blobData, err := os.ReadFile(blobSource)
	if err != nil {
		t.Fatal(err)
	}
	ollama := t.TempDir()
	blobs := writeSourceOllamaModel(t, ollama, "registry.ollama.ai/library/tiny/latest", map[string]string{ollamaModelLayer: string(blobData)})
	f.ex.modelSources = []modelSource{
		{name: "ollama", layout: sourceLayoutOllama, root: ollama},
		{name: "lmstudio", layout: sourceLayoutGGUFTree, root: studio},
	}
	linked, err := f.ex.linkModel("fake", linkedFile)
	if err != nil {
		t.Fatal(err)
	}

	profiles := map[string]struct{ model, path, bundle string }{
		"studio": {"lmstudio/pub/repo/Studio-Q4_K_M", studioModel, "model/Studio-Q4_K_M.gguf"},
		"linked": {linked, linkedFile, "model/Linked.gguf"},
		"tiny":   {"ollama/tiny@latest", blobs[ollamaModelLayer], "model/ollama-tiny-latest.gguf"},
	}
	rt.ModelLoads = nil
	for _, key := range sortedKeys(profiles) {
		load := modelconfig.CloneLoads([]modelconfig.ModelLoad{template})[0]
		load.ProfileKey, load.ModelID, load.Resolved = key, profiles[key].model, nil
		rt.ModelLoads = append(rt.ModelLoads, load)
	}
	prepared, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, rt), rt, false)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	sections, err := parseLlamaServingPreset(prepared.Snapshot)
	if err != nil {
		t.Fatal(err)
	}
	for key, profile := range profiles {
		if got := sections[key]["model"]; got != profile.path {
			t.Errorf("profile %s preset model = %q, want %q", key, got, profile.path)
		}
		resolved := rtLoadResolution(prepared.ModelLoads, key)
		if resolved == nil || len(resolved.Artifacts) != 1 || resolved.Artifacts[0].Path != profile.path {
			t.Errorf("profile %s resolution = %+v", key, resolved)
		}
	}

	// Export resolves the stored typed loads with no source list: the profile
	// alone carries a source model to a peer.
	rt.ModelLoads = prepared.ModelLoads
	f.st.plat.Runtime = rt
	_, store, err := f.ex.modelStoreFor("fake")
	if err != nil {
		t.Fatal(err)
	}
	snapshot, command, err := copyLaunchSnapshot(f.st)
	if err != nil {
		t.Fatal(err)
	}
	for key, profile := range profiles {
		t.Run("export "+key, func(t *testing.T) {
			list, _, err := resolveLlamaArtifacts(context.Background(), snapshot, command, store, profile.model, servingFileAccess{})
			if err != nil {
				t.Fatalf("resolve: %v", err)
			}
			if len(list.Files) != 1 || list.Files[0].Path != profile.bundle || list.Files[0].configured != profile.path {
				t.Fatalf("export files = %+v", list.Files)
			}
			for i := range list.Files {
				list.Files[i].SHA256 = fileDigest(t, profile.path)
			}
			if _, err := artifactExportID(list); err != nil {
				t.Fatalf("export ID: %v", err)
			}
		})
	}

	t.Run("removed link", func(t *testing.T) {
		if err := os.Remove(linkedFile); err != nil {
			t.Fatal(err)
		}
		again := rt
		again.ModelLoads = modelconfig.CloneLoads(rt.ModelLoads)
		for i := range again.ModelLoads {
			again.ModelLoads[i].Resolved = nil
		}
		if _, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, again), again, false); err == nil {
			t.Fatal("a profile on a removed linked file compiled")
		}
	})
}
