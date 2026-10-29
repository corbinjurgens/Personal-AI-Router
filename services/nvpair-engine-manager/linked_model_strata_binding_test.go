// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestLinkedModelStrataBinding(t *testing.T) {
	t.Run("declared aliases retain selected source paths", func(t *testing.T) {
		f := newStrataServingFixture(t)
		configureStrataExpertTable(t, &f)
		model := &f.request.Settings.Serving.Models[0]
		for _, path := range []string{f.source, filepath.Join(f.root, "pack", "native_experts.txt")} {
			alias := path + ".alias"
			if err := os.Symlink(path, alias); err != nil {
				t.Skipf("symlink unavailable: %v", err)
			}
			for i, artifact := range model.Artifacts {
				if artifact.Binding.SourcePath == path {
					model.Artifacts[i] = strataArtifact(t, alias, artifact.Role)
				}
			}
		}
		preview := approveLlamaServing(t, f.llamaServingFixture)
		applyLlamaServing(t, f.llamaServingFixture, preview)
		st := settingsState(t, f.ex)
		command, err := launchForState(st, st.port)
		if err != nil {
			t.Fatal(err)
		}
		prepared, err := compilePreparedServing(st, command, st.plat.Runtime, true)
		if err != nil {
			t.Fatal(err)
		}
		if err := recheckPreparedServing(st, command, st.plat.Runtime, prepared); err != nil {
			t.Fatalf("unchanged selector aliases rejected: %v", err)
		}
	})
	for _, name := range []string{"config.json", "pack/native_experts.txt", "profile.bin", "model-00001-of-00002.gguf"} {
		t.Run(name, func(t *testing.T) {
			f := newStrataServingFixture(t)
			configureStrataExpertTable(t, &f)
			preview := approveLlamaServing(t, f.llamaServingFixture)
			applyLlamaServing(t, f.llamaServingFixture, preview)
			st := settingsState(t, f.ex)
			command, err := launchForState(st, st.port)
			if err != nil {
				t.Fatal(err)
			}
			prepared, err := compilePreparedServing(st, command, st.plat.Runtime, true)
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(f.root, filepath.FromSlash(name))
			replacement := filepath.Join(t.TempDir(), "replacement")
			writeServingFixture(t, replacement, []byte("forbidden replacement"))
			reads, forbidden, closes, selectedReads := 0, 0, 0, 0
			substitute := false
			access := servingFileAccess{open: func(openPath string) (servingArtifactFile, error) {
				counter := &reads
				if openPath == path {
					counter = &selectedReads
					if substitute {
						openPath, counter = replacement, &forbidden
					}
				}
				file, err := os.Open(openPath)
				if err != nil {
					return nil, err
				}
				return &countedLinkedFile{File: file, reads: counter, closes: &closes}, nil
			}}
			if err := recheckPreparedServingAccess(st, command, st.plat.Runtime, prepared, access); err != nil || selectedReads == 0 {
				t.Fatalf("unchanged selected source: reads=%d error=%v", selectedReads, err)
			}
			substitute = true
			if err := recheckPreparedServingAccess(st, command, st.plat.Runtime, prepared, access); err == nil {
				t.Fatal("substituted selected source accepted")
			}
			if forbidden != 0 {
				t.Fatalf("replacement read before refusal: %d", forbidden)
			}
			// A path replacement with matching size/mtime must fail against the
			// original prepared identity before even reading another input.
			substitute = false
			info, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.Rename(path, path+".original"); err != nil {
				t.Fatal(err)
			}
			writeServingFixture(t, path, bytes.Repeat([]byte("x"), int(info.Size())))
			if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
				t.Fatal(err)
			}
			reads, selectedReads = 0, 0
			if err := recheckPreparedServingAccess(st, command, st.plat.Runtime, prepared, access); err == nil {
				t.Fatal("same-metadata replacement accepted")
			}
			if reads != 0 || selectedReads != 0 {
				t.Fatalf("replaced prepared input sampled: other=%d selected=%d", reads, selectedReads)
			}
		})
	}
	t.Run("runtime full hash uses checked bound handle", func(t *testing.T) {
		f := newStrataServingFixture(t)
		path := filepath.Join(f.root, "profile.bin")
		replacement := filepath.Join(t.TempDir(), "replacement")
		writeServingFixture(t, replacement, []byte("forbidden runtime replacement"))
		reads, forbidden, closes := 0, 0, 0
		substitute := false
		previous := fileHashes.access
		t.Cleanup(func() { fileHashes.access = previous })
		fileHashes.access = servingFileAccess{open: func(openPath string) (servingArtifactFile, error) {
			counter := &reads
			if substitute && openPath == path {
				openPath, counter = replacement, &forbidden
			}
			file, err := os.Open(openPath)
			if err != nil {
				return nil, err
			}
			return &countedLinkedFile{File: file, reads: counter, closes: &closes}, nil
		}}
		if err := f.prepareRuntimeHashes(t, context.Background(), f.request.Settings.Serving); err != nil || reads == 0 {
			t.Fatalf("initial runtime hashes: reads=%d error=%v", reads, err)
		}
		// A warm digest needs no content read, but must still validate its
		// authority; a cold digest checks the actual opened object.
		reads = 0
		if err := f.prepareRuntimeHashes(t, context.Background(), f.request.Settings.Serving); err != nil || reads != 0 {
			t.Fatalf("cached runtime hashes: reads=%d error=%v", reads, err)
		}
		fileHashes.mu.Lock()
		delete(fileHashes.done, path)
		fileHashes.mu.Unlock()
		substitute = true
		if err := f.prepareRuntimeHashes(t, context.Background(), f.request.Settings.Serving); err == nil {
			t.Fatal("substituted runtime hash handle accepted")
		}
		if forbidden != 0 {
			t.Fatalf("runtime replacement hashed: %d", forbidden)
		}
	})
	t.Run("empty prepared config is rejected before reading", func(t *testing.T) {
		f := newStrataServingFixture(t)
		preview := approveLlamaServing(t, f.llamaServingFixture)
		applyLlamaServing(t, f.llamaServingFixture, preview)
		st := settingsState(t, f.ex)
		command, err := launchForState(st, st.port)
		if err != nil {
			t.Fatal(err)
		}
		prepared, err := compilePreparedServing(st, command, st.plat.Runtime, true)
		if err != nil {
			t.Fatal(err)
		}
		writeServingFixture(t, f.source, nil)
		usage := &servingReadUsage{}
		if err := recheckPreparedServingAccess(st, command, st.plat.Runtime, prepared, servingFileAccess{usage: usage}); err == nil {
			t.Fatal("empty config accepted")
		}
		if *usage != (servingReadUsage{}) {
			t.Fatalf("empty config recheck read content: %+v", usage)
		}
	})
	t.Run("ambiguous pack is refused before expert table read", func(t *testing.T) {
		f := newStrataServingFixture(t)
		configureStrataExpertTable(t, &f)
		preview := approveLlamaServing(t, f.llamaServingFixture)
		applyLlamaServing(t, f.llamaServingFixture, preview)
		st := settingsState(t, f.ex)
		command, err := launchForState(st, st.port)
		if err != nil {
			t.Fatal(err)
		}
		prepared, err := compilePreparedServing(st, command, st.plat.Runtime, true)
		if err != nil {
			t.Fatal(err)
		}
		writeServingFixture(t, filepath.Join(f.root, "pack", "experts.bin"), []byte("ambiguous"))
		tableReads, reads, closes := 0, 0, 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			counter := &reads
			if path == filepath.Join(f.root, "pack", "native_experts.txt") {
				counter = &tableReads
			}
			file, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			return &countedLinkedFile{File: file, reads: counter, closes: &closes}, nil
		}}
		if err := recheckPreparedServingAccess(st, command, st.plat.Runtime, prepared, access); err == nil {
			t.Fatal("ambiguous expert source accepted")
		}
		if tableReads != 0 {
			t.Fatalf("ambiguous table read: %d", tableReads)
		}
	})
}

func configureStrataExpertTable(t *testing.T, f *strataServingFixture) {
	t.Helper()
	experts := filepath.Join(f.root, "pack", "experts.bin")
	if err := os.Remove(experts); err != nil {
		t.Fatal(err)
	}
	model := &f.request.Settings.Serving.Models[0]
	for i, artifact := range model.Artifacts {
		if artifact.Binding.SourcePath == experts {
			model.Artifacts = append(model.Artifacts[:i:i], model.Artifacts[i+1:]...)
			break
		}
	}
	table := filepath.Join(f.root, "pack", "native_experts.txt")
	writeServingFixture(t, table, []byte("# strata native experts v4: synthetic\n0 1 1 1 1 1 1 1 model-00002-of-00002.gguf\n"))
	model.Artifacts = append(model.Artifacts, strataArtifact(t, table, "model:expert-table"))
}
