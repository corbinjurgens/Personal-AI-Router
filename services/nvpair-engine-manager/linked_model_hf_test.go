// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/modelconfig"
)

func TestLinkedModelHFBinding(t *testing.T) {
	const model = "owner/model:Q4_K_M"
	commit := strings.Repeat("a", 40)
	fixture := func(t *testing.T) (engineModelStore, modelFileList) {
		t.Helper()
		root := t.TempDir()
		writeLlamaCache(t, root, "owner/model", commit, map[string][]byte{"nested/model-Q4_K_M.gguf": []byte("model")})
		store := engineModelStore{engine: "llamacpp", root: root, adapter: engines.NameLlamaCPP}
		list, err := boundedLlamaCacheArtifacts(context.Background(), store, model, servingFileAccess{})
		if err != nil {
			t.Fatal(err)
		}
		return store, list
	}
	assertEmpty := func(t *testing.T, list modelFileList, err error) {
		t.Helper()
		if err == nil || len(list.Files) != 0 || list.source != nil {
			t.Fatalf("failed selection published a partial set: files=%v source=%v err=%v", list.Files, list.source, err)
		}
	}
	t.Run("captures selectors directories and ordered artifacts", func(t *testing.T) {
		root := t.TempDir()
		writeLlamaCache(t, root, "owner/model", strings.Repeat("a", 40), map[string][]byte{"model-Q4_K_M.gguf": []byte("model")})
		list, err := boundedLlamaCacheArtifacts(context.Background(), engineModelStore{root: root, adapter: engines.NameLlamaCPP}, "owner/model:Q4_K_M", servingFileAccess{})
		if err != nil {
			t.Fatal(err)
		}
		if list.source == nil || len(list.source.selectors) == 0 || len(list.source.directories) == 0 || len(list.source.artifacts) != len(list.Files) {
			t.Fatal("HF selection has no complete source snapshot")
		}
	})

	t.Run("substituted opened directory is rejected before enumeration", func(t *testing.T) {
		root := t.TempDir()
		commit := strings.Repeat("a", 40)
		writeLlamaCache(t, root, "owner/model", commit, map[string][]byte{"model-Q4_K_M.gguf": []byte("model")})
		snap := filepath.Join(llamaRepoDir(root, "owner/model"), "snapshots", commit)
		other := t.TempDir()
		reads := 0
		access := servingFileAccess{openDir: func(path string) (servingDirectory, error) {
			if path == snap {
				f, err := os.Open(other)
				if err != nil {
					return nil, err
				}
				return &hfDirectoryCounter{File: f, reads: &reads}, nil
			}
			return os.Open(path)
		}}
		if _, err := boundedLlamaCacheArtifacts(context.Background(), engineModelStore{root: root, adapter: engines.NameLlamaCPP}, "owner/model:Q4_K_M", access); err == nil {
			t.Fatal("substituted directory accepted")
		}
		if reads != 0 {
			t.Fatalf("forbidden enumerations = %d, want 0", reads)
		}
	})

	t.Run("unchanged expected snapshot retains selection", func(t *testing.T) {
		store, original := fixture(t)
		list, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, servingFileAccess{}, original)
		if err != nil {
			t.Fatal(err)
		}
		if list.Revision != original.Revision || !reflect.DeepEqual(list.Files, original.Files) {
			t.Fatalf("recheck changed selection: %v", list.Files)
		}
	})

	t.Run("missing expected snapshot refuses without opening", func(t *testing.T) {
		store, original := fixture(t)
		original.source = nil
		opens := 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			opens++
			return os.Open(path)
		}}
		list, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, access, original)
		assertEmpty(t, list, err)
		if opens != 0 {
			t.Fatalf("missing evidence opened %d files", opens)
		}
	})

	t.Run("old selecting metadata rejects replacements before all reads", func(t *testing.T) {
		test := func(name string, choose func(engineModelStore, modelFileList) string) {
			t.Run(name, func(t *testing.T) {
				store, original := fixture(t)
				path := choose(store, original)
				info, err := os.Stat(path)
				if err != nil {
					t.Fatal(err)
				}
				// Move the old object away so the replacement cannot reuse its inode.
				if err := os.Rename(path, path+"-old"); err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := os.RemoveAll(path + "-old"); err != nil {
						t.Error(err)
					}
				})
				if info.IsDir() {
					if err := os.Mkdir(path, 0o755); err != nil {
						t.Fatal(err)
					}
				} else {
					writeTestFile(t, path, []byte(strings.Repeat("b", int(info.Size()))))
				}
				// Restore directory mtimes so the replaced object's own identity is
				// the rejecting evidence, including same-size/mtime file replacement.
				for _, evidence := range original.source.directories {
					if err := os.Chtimes(evidence.configured, evidence.info.ModTime(), evidence.info.ModTime()); err != nil && !errors.Is(err, os.ErrNotExist) {
						t.Fatal(err)
					}
				}
				if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
					t.Fatal(err)
				}
				fileReads, directoryReads, opens, closes := 0, 0, 0, 0
				access := servingFileAccess{
					open: func(path string) (servingArtifactFile, error) {
						f, err := os.Open(path)
						if err != nil {
							return nil, err
						}
						opens++
						return &countedLinkedFile{File: f, reads: &fileReads, closes: &closes}, nil
					},
					openDir: func(path string) (servingDirectory, error) {
						f, err := os.Open(path)
						if err != nil {
							return nil, err
						}
						opens++
						return &countedLinkedDir{File: f, reads: &directoryReads, closes: &closes}, nil
					},
				}
				list, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, access, original)
				assertEmpty(t, list, err)
				if fileReads != 0 || directoryReads != 0 || opens != closes {
					t.Fatalf("file reads=%d enumerations=%d opens=%d closes=%d", fileReads, directoryReads, opens, closes)
				}
			})
		}
		test("root", func(store engineModelStore, _ modelFileList) string { return store.root })
		test("repository", func(store engineModelStore, _ modelFileList) string { return llamaRepoDir(store.root, "owner/model") })
		test("refs directory", func(store engineModelStore, _ modelFileList) string {
			return filepath.Join(llamaRepoDir(store.root, "owner/model"), "refs")
		})
		test("snapshot parent", func(store engineModelStore, _ modelFileList) string {
			return filepath.Join(llamaRepoDir(store.root, "owner/model"), "snapshots")
		})
		test("snapshot", func(store engineModelStore, _ modelFileList) string {
			return filepath.Join(llamaRepoDir(store.root, "owner/model"), "snapshots", commit)
		})
		test("nested directory", func(store engineModelStore, _ modelFileList) string {
			return filepath.Join(llamaRepoDir(store.root, "owner/model"), "snapshots", commit, "nested")
		})
		test("ref", func(store engineModelStore, _ modelFileList) string {
			return filepath.Join(llamaRepoDir(store.root, "owner/model"), "refs", "main")
		})
		test("artifact", func(_ engineModelStore, list modelFileList) string { return list.Files[0].abs })
	})

	t.Run("same-stat selector bytes are compared before artifact sampling", func(t *testing.T) {
		store, original := fixture(t)
		selector := original.source.selectors[0]
		writeTestFile(t, selector.file.configured, []byte(strings.Repeat("b", 40)))
		if err := os.Chtimes(selector.file.configured, selector.file.info.ModTime(), selector.file.info.ModTime()); err != nil {
			t.Fatal(err)
		}
		reads, closes, opens := 0, 0, 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			f, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			opens++
			if path == original.Files[0].configured {
				return &countedLinkedFile{File: f, reads: &reads, closes: &closes}, nil
			}
			return &countedLinkedFile{File: f, reads: new(int), closes: &closes}, nil
		}}
		list, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, access, original)
		assertEmpty(t, list, err)
		if reads != 0 || opens != closes {
			t.Fatalf("artifact reads=%d opens=%d closes=%d", reads, opens, closes)
		}
	})

	t.Run("expected opened selector substitution refuses before reading", func(t *testing.T) {
		store, original := fixture(t)
		other := filepath.Join(t.TempDir(), "other-ref")
		writeTestFile(t, other, []byte(strings.Repeat("b", 40)))
		reads, opens, closes := 0, 0, 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			if path != original.source.selectors[0].file.configured {
				return os.Open(path)
			}
			f, err := os.Open(other)
			if err != nil {
				return nil, err
			}
			opens++
			return &countedLinkedFile{File: f, reads: &reads, closes: &closes}, nil
		}}
		list, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, access, original)
		assertEmpty(t, list, err)
		if reads != 0 || opens != 1 || closes != opens {
			t.Fatalf("forbidden reads=%d opens=%d closes=%d", reads, opens, closes)
		}
	})

	t.Run("expected opened snapshot substitution refuses before enumeration", func(t *testing.T) {
		store, original := fixture(t)
		snap := filepath.Dir(filepath.Dir(original.Files[0].configured))
		other := t.TempDir()
		reads, opens, closes := 0, 0, 0
		access := servingFileAccess{openDir: func(path string) (servingDirectory, error) {
			if path != snap {
				return os.Open(path)
			}
			f, err := os.Open(other)
			if err != nil {
				return nil, err
			}
			opens++
			return &countedLinkedDir{File: f, reads: &reads, closes: &closes}, nil
		}}
		list, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, access, original)
		assertEmpty(t, list, err)
		if reads != 0 || opens != 1 || closes != opens {
			t.Fatalf("forbidden enumerations=%d opens=%d closes=%d", reads, opens, closes)
		}
	})

	t.Run("expected opened artifact substitution refuses before reading", func(t *testing.T) {
		store, original := fixture(t)
		other := filepath.Join(t.TempDir(), "other.gguf")
		writeTestFile(t, other, []byte("other"))
		reads, opens, closes := 0, 0, 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			if path != original.Files[0].configured {
				return os.Open(path)
			}
			f, err := os.Open(other)
			if err != nil {
				return nil, err
			}
			opens++
			return &countedLinkedFile{File: f, reads: &reads, closes: &closes}, nil
		}}
		list, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, access, original)
		assertEmpty(t, list, err)
		if reads != 0 || opens != 1 || closes != opens {
			t.Fatalf("forbidden reads=%d opens=%d closes=%d", reads, opens, closes)
		}
	})

	t.Run("changed companion selection refuses before any artifact sample", func(t *testing.T) {
		store, original := fixture(t)
		snap := filepath.Dir(filepath.Dir(original.Files[0].configured))
		writeTestFile(t, filepath.Join(snap, "mmproj-Q4_K_M.gguf"), []byte("new companion"))
		for _, evidence := range original.source.directories {
			if err := os.Chtimes(evidence.configured, evidence.info.ModTime(), evidence.info.ModTime()); err != nil {
				t.Fatal(err)
			}
		}
		reads, closes, opens := 0, 0, 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			f, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			opens++
			if strings.HasSuffix(path, ".gguf") {
				return &countedLinkedFile{File: f, reads: &reads, closes: &closes}, nil
			}
			return &countedLinkedFile{File: f, reads: new(int), closes: &closes}, nil
		}}
		list, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, access, original)
		assertEmpty(t, list, err)
		if reads != 0 || closes != opens {
			t.Fatalf("artifact reads=%d opens=%d closes=%d", reads, opens, closes)
		}
	})

	t.Run("ref grammar and escapes refuse without artifacts", func(t *testing.T) {
		for _, value := range []string{"../escape", strings.Repeat("a", 39), strings.Repeat("g", 40), "/outside/snapshot"} {
			t.Run(value, func(t *testing.T) {
				store, original := fixture(t)
				writeTestFile(t, original.source.selectors[0].file.configured, []byte(value))
				list, err := boundedLlamaCacheArtifacts(context.Background(), store, model, servingFileAccess{})
				assertEmpty(t, list, err)
			})
		}
	})

	t.Run("fallback ref retains every selecting source", func(t *testing.T) {
		store, original := fixture(t)
		main := original.source.selectors[0].file.configured
		writeTestFile(t, main, nil)
		writeTestFile(t, filepath.Join(filepath.Dir(main), "branch"), []byte(commit+"\n"))
		list, err := boundedLlamaCacheArtifacts(context.Background(), store, model, servingFileAccess{})
		if err != nil {
			t.Fatal(err)
		}
		if len(list.source.selectors) < 2 || list.Revision != commit {
			t.Fatalf("fallback selectors=%v revision=%q", list.source.selectors, list.Revision)
		}
		if _, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, servingFileAccess{}, list); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("oversized ref refuses before selector reads", func(t *testing.T) {
		store, original := fixture(t)
		writeTestFile(t, original.source.selectors[0].file.configured, []byte(strings.Repeat("a", 4097)))
		opens := 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			opens++
			return os.Open(path)
		}}
		list, err := boundedLlamaCacheArtifacts(context.Background(), store, model, access)
		assertEmpty(t, list, err)
		if !errors.Is(err, errModelArtifactLimit) || opens != 0 {
			t.Fatalf("err=%v opens=%d", err, opens)
		}
	})

	t.Run("selected shards and companion retain ordered roles", func(t *testing.T) {
		root := t.TempDir()
		writeLlamaCache(t, root, "owner/model", commit, map[string][]byte{
			"nested/model-Q4_K_M-00001-of-00002.gguf": []byte("first"),
			"nested/model-Q4_K_M-00002-of-00002.gguf": []byte("second"),
			"mmproj-Q4_K_M.gguf":                      []byte("projector"),
			"other-Q8_0.gguf":                         []byte("unselected"),
		})
		store := engineModelStore{root: root, adapter: engines.NameLlamaCPP}
		list, err := boundedLlamaCacheArtifacts(context.Background(), store, model, servingFileAccess{})
		if err != nil {
			t.Fatal(err)
		}
		var paths, roles []string
		for _, file := range list.Files {
			paths, roles = append(paths, file.Path), append(roles, file.Role)
		}
		if !reflect.DeepEqual(paths, []string{"nested/model-Q4_K_M-00001-of-00002.gguf", "nested/model-Q4_K_M-00002-of-00002.gguf", "mmproj-Q4_K_M.gguf"}) || !reflect.DeepEqual(roles, []string{"model", "model", "companion"}) {
			t.Fatalf("paths=%v roles=%v", paths, roles)
		}
		if _, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, servingFileAccess{}, list); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("incomplete shards never publish a partial set", func(t *testing.T) {
		root := t.TempDir()
		writeLlamaCache(t, root, "owner/model", commit, map[string][]byte{"model-Q4_K_M-00001-of-00002.gguf": []byte("first")})
		list, err := boundedLlamaCacheArtifacts(context.Background(), engineModelStore{root: root}, model, servingFileAccess{})
		assertEmpty(t, list, err)
	})

	t.Run("shared entry budget includes non-artifacts", func(t *testing.T) {
		store, original := fixture(t)
		writeTestFile(t, filepath.Join(filepath.Dir(original.Files[0].configured), "ignored.txt"), []byte("ignored"))
		access := servingFileAccess{scan: &modelArtifactScan{entries: modelArtifactScanEntries - 1}}
		list, err := boundedLlamaCacheArtifacts(context.Background(), store, model, access)
		assertEmpty(t, list, err)
	})

	t.Run("depth limit refuses without a partial set", func(t *testing.T) {
		store, original := fixture(t)
		path := filepath.Dir(filepath.Dir(original.Files[0].configured))
		for i := 0; i <= 64; i++ {
			path = filepath.Join(path, "d")
		}
		writeTestFile(t, filepath.Join(path, "deep.gguf"), []byte("deep"))
		list, err := boundedLlamaCacheArtifacts(context.Background(), store, model, servingFileAccess{})
		assertEmpty(t, list, err)
		if !errors.Is(err, errModelArtifactLimit) {
			t.Fatalf("depth failure = %v", err)
		}
	})

	t.Run("artifact bound refuses without a partial set", func(t *testing.T) {
		store, original := fixture(t)
		dir := filepath.Dir(original.Files[0].configured)
		for i := 0; i < maxCopyArtifacts; i++ {
			writeTestFile(t, filepath.Join(dir, fmt.Sprintf("unused-%04d.gguf", i)), []byte("unused"))
		}
		list, err := boundedLlamaCacheArtifacts(context.Background(), store, model, servingFileAccess{})
		assertEmpty(t, list, err)
		if !errors.Is(err, errModelArtifactLimit) {
			t.Fatalf("artifact bound failure = %v", err)
		}
	})

	t.Run("cancellation after directory read closes handle and publishes nothing", func(t *testing.T) {
		store, _ := fixture(t)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		reads, closes, opens := 0, 0, 0
		access := servingFileAccess{openDir: func(path string) (servingDirectory, error) {
			f, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			opens++
			return &countedLinkedDir{File: f, reads: &reads, closes: &closes, afterRead: func([]os.DirEntry, error) { cancel() }}, nil
		}}
		list, err := boundedLlamaCacheArtifacts(ctx, store, model, access)
		assertEmpty(t, list, err)
		if !errors.Is(err, context.Canceled) || reads != 1 || opens != closes {
			t.Fatalf("err=%v enumerations=%d opens=%d closes=%d", err, reads, opens, closes)
		}
	})

	t.Run("final metadata drift during samples publishes nothing", func(t *testing.T) {
		store, original := fixture(t)
		changed := false
		reads, closes, opens := 0, 0, 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			f, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			opens++
			return &countedLinkedFile{File: f, reads: &reads, closes: &closes, afterRead: func(int, int64) {
				if path == original.Files[0].configured && !changed {
					changed = true
					selector := original.source.selectors[0].file
					writeTestFile(t, selector.configured, []byte(strings.Repeat("b", 40)))
					if err := os.Chtimes(selector.configured, selector.info.ModTime(), selector.info.ModTime()); err != nil {
						t.Fatal(err)
					}
				}
			}}, nil
		}}
		list, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, access, original)
		assertEmpty(t, list, err)
		if !changed || reads == 0 || opens != closes {
			t.Fatalf("changed=%v reads=%d opens=%d closes=%d", changed, reads, opens, closes)
		}
	})

	t.Run("outside source link is bound", func(t *testing.T) {
		store, original := fixture(t)
		outside := filepath.Join(t.TempDir(), "outside.gguf")
		writeTestFile(t, outside, []byte("outside"))
		path := original.Files[0].configured
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(outside, path); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		list, err := boundedLlamaCacheArtifacts(context.Background(), store, model, servingFileAccess{})
		if err != nil || len(list.Files) != 1 || list.Files[0].abs != outside {
			t.Fatalf("outside artifact: files=%v err=%v", list.Files, err)
		}
		if _, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, servingFileAccess{}, list); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("linked ref and hardlinked artifact retain exact bindings", func(t *testing.T) {
		store, original := fixture(t)
		ref := original.source.selectors[0].file.configured
		outside := filepath.Join(t.TempDir(), "ref")
		writeTestFile(t, outside, []byte(commit))
		if err := os.Remove(ref); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(outside, ref); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		artifact := original.Files[0].configured
		if err := os.Remove(artifact); err != nil {
			t.Fatal(err)
		}
		if err := os.Link(original.Files[0].abs, artifact); err != nil {
			t.Skipf("hardlink unavailable: %v", err)
		}
		list, err := boundedLlamaCacheArtifacts(context.Background(), store, model, servingFileAccess{})
		if err != nil || len(list.Files) != 1 {
			t.Fatalf("files=%v err=%v", list.Files, err)
		}
		if _, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, servingFileAccess{}, list); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("directory links select companions", func(t *testing.T) {
		store, original := fixture(t)
		outside := t.TempDir()
		writeTestFile(t, filepath.Join(outside, "model-Q4_K_M.gguf"), []byte("model"))
		writeTestFile(t, filepath.Join(outside, "mmproj-Q4_K_M.gguf"), []byte("outside"))
		link := filepath.Dir(original.Files[0].configured)
		if err := os.RemoveAll(link); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(outside, link); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		list, err := boundedLlamaCacheArtifacts(context.Background(), store, model, servingFileAccess{})
		if err != nil {
			t.Fatal(err)
		}
		if len(list.Files) != 2 || list.Files[1].Path != "nested/mmproj-Q4_K_M.gguf" {
			t.Fatalf("linked companion selection: %v", list.Files)
		}
	})

	t.Run("linked refs and snapshot bind old targets", func(t *testing.T) {
		store, original := fixture(t)
		repo := llamaRepoDir(store.root, "owner/model")
		for _, path := range []string{filepath.Join(repo, "refs"), filepath.Join(repo, "snapshots", commit)} {
			target := filepath.Join(t.TempDir(), "selected")
			if err := os.Rename(path, target); err != nil {
				t.Fatal(err)
			}
			if path == filepath.Join(repo, "snapshots", commit) {
				artifact := filepath.Join(target, "nested", "model-Q4_K_M.gguf")
				if err := os.Remove(artifact); err != nil {
					t.Fatal(err)
				}
				writeTestFile(t, artifact, []byte("model"))
			}
			if err := os.Symlink(target, path); err != nil {
				t.Skipf("symlink unavailable: %v", err)
			}
		}
		list, err := boundedLlamaCacheArtifacts(context.Background(), store, model, servingFileAccess{})
		if err != nil || len(list.Files) != len(original.Files) {
			t.Fatalf("linked selection files=%v err=%v", list.Files, err)
		}
		if _, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, servingFileAccess{}, list); err != nil {
			t.Fatal(err)
		}
		ref := filepath.Join(repo, "refs")
		if err := os.Remove(ref); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(t.TempDir(), ref); err != nil {
			t.Fatal(err)
		}
		reads := 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			f, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			return &linkedReadCounter{File: f, reads: &reads}, nil
		}}
		changed, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, access, list)
		assertEmpty(t, changed, err)
		if reads != 0 {
			t.Fatalf("retarget caused %d reads", reads)
		}
	})

	t.Run("directory aliases and cycles refuse partial selection", func(t *testing.T) {
		for _, kind := range []string{"alias", "cycle", "broken"} {
			t.Run(kind, func(t *testing.T) {
				store, original := fixture(t)
				snap := filepath.Dir(filepath.Dir(original.Files[0].configured))
				target := filepath.Join(snap, "nested")
				if kind == "cycle" {
					target = snap
				}
				if kind == "broken" {
					target = filepath.Join(t.TempDir(), "missing")
				}
				if err := os.Symlink(target, filepath.Join(snap, "alias")); err != nil {
					t.Skipf("symlink unavailable: %v", err)
				}
				list, err := boundedLlamaCacheArtifacts(context.Background(), store, model, servingFileAccess{})
				assertEmpty(t, list, err)
			})
		}
	})

	t.Run("recognized linked cache compiles typed without serving declaration", func(t *testing.T) {
		f := newModelLoadFixture(t)
		model := "owner/repository:Q4_K_M"
		commit := strings.Repeat("a", 40)
		repo := llamaRepoDir(f.root, "owner/repository")
		writeTestFile(t, filepath.Join(repo, "refs", "main"), []byte(commit))
		outside := filepath.Join(t.TempDir(), "model.gguf")
		writeInspectionGGUF(t, outside, "llama", 0.25, 4096)
		artifact := filepath.Join(repo, "snapshots", commit, "model-Q4_K_M.gguf")
		if err := os.MkdirAll(filepath.Dir(artifact), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(outside, artifact); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		rt := f.rt
		args := []string{"--models-max", "1"}
		rt.LaunchArgs, rt.Serving = &args, nil
		rt.ModelLoads = []modelconfig.ModelLoad{{ModelID: model}}
		copyTestNativeModels(t, f.ex, "fake", model)
		prepared, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, rt), rt, false)
		if err != nil || prepared == nil || prepared.Config != nil || len(prepared.ModelLoads) != 1 {
			t.Fatalf("typed selected source: result=%+v err=%v", prepared, err)
		}
		if prepared.evidence == nil || prepared.evidence.typed == nil {
			t.Fatal("missing typed review evidence")
		}
		binding, err := servingFileBinding(artifact)
		if err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(outside)
		if err != nil {
			t.Fatal(err)
		}
		ram, vram := int64(4096), int64(0)
		servingRT := rt
		servingRT.ModelLoads = prepared.ModelLoads
		servingRT.Serving = &engines.ServingConfig{Version: 1, Models: []engines.ServingModel{{
			ModelID: model, MaxOutputTokens: 128, Memory: engines.ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram},
			Artifacts: []engines.ServingArtifact{{Role: "model", SHA256: sha256Hex(data), Binding: binding, Provenance: engines.ServingProvenance{Kind: "supplied-sha256"}}},
		}}}
		if configured, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, servingRT), servingRT, false); err != nil || configured.Config == nil {
			t.Fatalf("configured linked serving: result=%+v err=%v", configured, err)
		}
		replacement := filepath.Join(t.TempDir(), "replacement.gguf")
		writeInspectionGGUF(t, replacement, "llama", 0.75, 4096)
		if err := os.Remove(artifact); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(replacement, artifact); err != nil {
			t.Fatal(err)
		}
		forbidden := 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			file, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			if path == artifact {
				return &linkedReadCounter{File: file, reads: &forbidden}, nil
			}
			return file, nil
		}}
		if _, err := compileTypedLlamaInputsExpected(context.Background(), f.st, typedFixtureCommand(t, f, rt), rt, false, access, prepared.evidence.typed); err == nil || forbidden != 0 {
			t.Fatalf("stale typed review err=%v forbidden reads=%d", err, forbidden)
		}
	})

	t.Run("destination ref remains confined", func(t *testing.T) {
		store, original := fixture(t)
		outside := filepath.Join(t.TempDir(), "ref")
		writeTestFile(t, outside, []byte(commit))
		path := original.source.selectors[0].file.configured
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(outside, path); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		if revision := llamaCachedRef(llamaRepoDir(store.root, "owner/model")); revision != "" {
			t.Fatalf("destination accepted outside ref: %q", revision)
		}
	})

	t.Run("configured root alias retains existing support", func(t *testing.T) {
		store, _ := fixture(t)
		alias := filepath.Join(t.TempDir(), "cache")
		if err := os.Symlink(store.root, alias); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		store.root = alias
		list, err := boundedLlamaCacheArtifacts(context.Background(), store, model, servingFileAccess{})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, servingFileAccess{}, list); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("whole repository alias retains existing support", func(t *testing.T) {
		store, _ := fixture(t)
		otherRoot := t.TempDir()
		repo := llamaRepoDir(store.root, "owner/model")
		target := llamaRepoDir(otherRoot, "owner/model")
		if err := os.Rename(repo, target); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(target, repo); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		list, err := boundedLlamaCacheArtifacts(context.Background(), store, model, servingFileAccess{})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := boundedLlamaCacheArtifactsExpected(context.Background(), store, model, servingFileAccess{}, list); err != nil {
			t.Fatal(err)
		}
	})
}

type hfDirectoryCounter struct {
	*os.File
	reads *int
}

func (f *hfDirectoryCounter) ReadDir(n int) ([]os.DirEntry, error) {
	*f.reads++
	return f.File.ReadDir(n)
}
