// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func TestLinkedModelNativeSnapshot(t *testing.T) {
	ctx := context.Background()
	type counts struct{ reads, streams, dirs, closes int }
	access := func(c *counts) servingFileAccess {
		return servingFileAccess{
			open: func(path string) (servingArtifactFile, error) {
				f, err := os.Open(path)
				if err != nil {
					return nil, err
				}
				return &countedLinkedFile{File: f, reads: &c.reads, streams: &c.streams, closes: &c.closes}, nil
			},
			openDir: func(path string) (servingDirectory, error) {
				f, err := os.Open(path)
				if err != nil {
					return nil, err
				}
				return &countedLinkedDir{File: f, reads: &c.dirs, closes: &c.closes}, nil
			},
		}
	}
	replace := func(t *testing.T, path string, data []byte) {
		t.Helper()
		if err := os.Rename(path, path+".old"); err != nil {
			t.Fatalf("move original %s: %v", path, err)
		}
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatalf("write replacement %s: %v", path, err)
		}
	}
	refuse := func(t *testing.T, ex *Executor, engine, model string, old modelFileList) {
		t.Helper()
		c := &counts{}
		list, err := ex.modelSourceFilesExpected(ctx, engine, model, access(c), old)
		if err == nil || len(list.Files) != 0 {
			t.Errorf("stale source returned files: %d, error: %v", len(list.Files), err)
		}
		if c.reads != 0 || c.streams != 0 || c.dirs != 0 {
			t.Errorf("stale source accessed content: %+v", c)
		}
	}
	assertEmpty := func(t *testing.T, list modelFileList, err error) {
		t.Helper()
		if err == nil || len(list.Files) != 0 || list.source != nil {
			t.Fatalf("failed capture published authority: files=%v source=%+v err=%v", list.Files, list.source, err)
		}
	}
	ollama := func(t *testing.T) (*Executor, string, ollamaFixture) {
		t.Helper()
		root := filepath.Join(t.TempDir(), "store")
		fx := writeOllamaModel(t, filepath.Join(root, "models"), "demo", 128, 1)
		ex := copyTestExecutor(t, "ollama", root, true)
		copyTestNativeModels(t, ex, "ollama", "demo")
		return ex, filepath.Join(root, "models"), fx
	}
	lmstudio := func(t *testing.T, layout string) (*Executor, string, *[]lmsListEntry) {
		t.Helper()
		root := filepath.Join(t.TempDir(), "store")
		path := "owner/demo"
		if layout == "GGUF" {
			path += "/model.gguf"
			writeTestFile(t, filepath.Join(root, "owner/demo/mmproj.gguf"), []byte("projector"))
			writeTestFile(t, filepath.Join(root, "owner/demo/other.gguf"), []byte("other quantization"))
			writeTestFile(t, filepath.Join(root, path), []byte("model"))
		} else {
			writeTestFile(t, filepath.Join(root, path, "weights.safetensors"), []byte("weights"))
			writeTestFile(t, filepath.Join(root, path, "empty.json"), nil)
			writeTestFile(t, filepath.Join(root, path, "tokenizer/vocab.json"), []byte(`{}`))
		}
		ex := copyTestExecutor(t, "lmstudio", root, true)
		rows := []lmsListEntry{{ModelKey: "demo", Path: path}}
		original := lmsListModels
		lmsListModels = func(context.Context, *Executor, string) ([]lmsListEntry, error) { return rows, nil }
		t.Cleanup(func() { lmsListModels = original })
		return ex, root, &rows
	}
	capture := func(t *testing.T, ex *Executor, engine string) modelFileList {
		t.Helper()
		list, err := ex.modelSourceFilesWithAccess(ctx, engine, "demo", access(&counts{}))
		if err != nil {
			t.Fatal(err)
		}
		return list
	}

	t.Run("Ollama captures selector and exact artifacts", func(t *testing.T) {
		root := t.TempDir()
		fx := writeOllamaModel(t, filepath.Join(root, "models"), "demo", 128, 1)
		ex := copyTestExecutor(t, "ollama", root, true)
		copyTestNativeModels(t, ex, "ollama", "demo")
		list, err := ex.modelSourceFilesWithAccess(ctx, "ollama", "demo", access(&counts{}))
		if err != nil {
			t.Fatal(err)
		}
		if list.source == nil || len(list.source.selectors) != 1 || len(list.source.directories) == 0 || len(list.source.artifacts) != len(fx.blobs)+1 {
			t.Fatalf("missing native selector, directory or artifact evidence: %+v", list.source)
		}
		selector := list.source.selectors[0]
		if string(selector.bytes) != string(fx.manifest) || selector.file.info == nil {
			t.Fatalf("manifest evidence missing or changed: %+v", selector)
		}
	})
	for _, changed := range []string{"manifest", "blob"} {
		t.Run("Ollama rejects replaced "+changed, func(t *testing.T) {
			root := t.TempDir()
			fx := writeOllamaModel(t, filepath.Join(root, "models"), "demo", 128, 1)
			ex := copyTestExecutor(t, "ollama", root, true)
			copyTestNativeModels(t, ex, "ollama", "demo")
			old, err := ex.modelSourceFilesWithAccess(ctx, "ollama", "demo", access(&counts{}))
			if err != nil {
				t.Fatal(err)
			}
			if changed == "manifest" {
				replace(t, filepath.Join(root, "models", filepath.FromSlash(fx.manifestRel)), fx.manifest)
			} else {
				for path, data := range fx.blobs {
					replace(t, filepath.Join(root, "models", filepath.FromSlash(path)), data)
					break
				}
			}
			refuse(t, ex, "ollama", "demo", old)
		})
	}
	t.Run("LM Studio freezes MLX rows and empty sidecars", func(t *testing.T) {
		root := t.TempDir()
		const model, dir = "owner/demo", "owner/demo"
		writeTestFile(t, filepath.Join(root, dir, "weights.safetensors"), []byte("weights"))
		writeTestFile(t, filepath.Join(root, dir, "empty.json"), []byte{})
		ex := copyTestExecutor(t, "lmstudio", root, true)
		rows := []lmsListEntry{{ModelKey: model, Path: dir}}
		original := lmsListModels
		lmsListModels = func(context.Context, *Executor, string) ([]lmsListEntry, error) { return rows, nil }
		t.Cleanup(func() { lmsListModels = original })
		old, err := ex.modelSourceFilesWithAccess(ctx, "lmstudio", model, access(&counts{}))
		if err != nil {
			t.Fatal(err)
		}
		if len(old.Files) != 2 {
			t.Fatalf("MLX files: %+v", old.Files)
		}
		if old.source == nil || len(old.source.directories) == 0 || len(old.source.artifacts) != 2 {
			t.Errorf("missing MLX snapshot: %+v", old.source)
		}
		for _, f := range old.Files {
			if f.configured == "" || f.abs == "" || f.info == nil {
				t.Errorf("missing private file evidence: %+v", f)
			}
		}
		rows = []lmsListEntry{{ModelKey: model, Path: dir + "/weights.safetensors"}}
		refuse(t, ex, "lmstudio", model, old)
	})
	t.Run("LM Studio rejects replaced MLX directory before enumeration", func(t *testing.T) {
		root := t.TempDir()
		const model, dir = "owner/demo", "owner/demo"
		path := filepath.Join(root, dir)
		writeTestFile(t, filepath.Join(path, "weights.safetensors"), []byte("weights"))
		ex := copyTestExecutor(t, "lmstudio", root, true)
		original := lmsListModels
		lmsListModels = func(context.Context, *Executor, string) ([]lmsListEntry, error) {
			return []lmsListEntry{{ModelKey: model, Path: dir}}, nil
		}
		t.Cleanup(func() { lmsListModels = original })
		old, err := ex.modelSourceFilesWithAccess(ctx, "lmstudio", model, access(&counts{}))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Rename(path, path+".old"); err != nil {
			t.Fatal(err)
		}
		writeTestFile(t, filepath.Join(path, "weights.safetensors"), []byte("replacement"))
		refuse(t, ex, "lmstudio", model, old)
	})

	t.Run("Ollama ordered artifacts and selector remain frozen across rechecks", func(t *testing.T) {
		ex, root, fx := ollama(t)
		old := capture(t, ex, "ollama")
		config := []byte(`{"model_format":"gguf","seed":1}`)
		want := []string{"blobs/sha256-" + sha256Hex(config), "blobs/sha256-" + sha256Hex(testBytes(128, 1)), fx.manifestRel}
		var got []string
		for _, file := range old.Files {
			got = append(got, file.Path)
			if file.configured != filepath.Join(root, filepath.FromSlash(file.Path)) || file.abs == "" || file.info == nil {
				t.Fatalf("missing file evidence: %+v", file)
			}
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("artifact order=%v, want %v", got, want)
		}
		old.Files[0].Path = "caller-mutated"
		if old.source.artifacts[0].Path != want[0] || !bytes.Equal(old.source.selectors[0].bytes, fx.manifest) {
			t.Fatal("returned files alias frozen authority")
		}
		fresh, err := ex.modelSourceFilesExpected(ctx, "ollama", "demo", access(&counts{}), old)
		if err != nil || len(fresh.Files) != len(want) || fresh.Files[0].Path != want[0] {
			t.Fatalf("unchanged recheck: %+v, %v", fresh, err)
		}
		fresh.Files[0].Size++
		fresh.source.selectors[0].bytes[0] = '!'
		if old.source.artifacts[0].Size != int64(len(config)) || !bytes.Equal(old.source.selectors[0].bytes, fx.manifest) {
			t.Fatal("rechecked listing aliases original snapshot")
		}
	})

	for _, layout := range []string{"GGUF", "MLX"} {
		t.Run("LM Studio "+layout+" exact set and unchanged recheck", func(t *testing.T) {
			ex, _, rows := lmstudio(t, layout)
			old := capture(t, ex, "lmstudio")
			want := []string{"owner/demo/model.gguf", "owner/demo/mmproj.gguf"}
			if layout == "MLX" {
				want = []string{"owner/demo/empty.json", "owner/demo/tokenizer/vocab.json", "owner/demo/weights.safetensors"}
			}
			var got []string
			for _, file := range old.Files {
				got = append(got, file.Path)
				if file.info == nil || file.configured == "" || file.abs == "" {
					t.Fatalf("missing private evidence: %+v", file)
				}
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("files=%v, want %v", got, want)
			}
			if layout == "MLX" && old.Files[0].Size != 0 {
				t.Fatalf("empty sidecar size=%d", old.Files[0].Size)
			}
			c := &counts{}
			fresh, err := ex.modelSourceFilesExpected(ctx, "lmstudio", "demo", access(c), old)
			if err != nil || len(fresh.Files) != len(want) || c.dirs == 0 || c.reads != 0 || c.streams != 0 {
				t.Fatalf("unchanged recheck: files=%v err=%v counts=%+v", fresh.Files, err, c)
			}
			old.Files[0].Path = "caller-mutated"
			if old.source.artifacts[0].Path != want[0] {
				t.Fatal("file slice aliases snapshot")
			}
			frozen := append([]byte(nil), old.source.recognition...)
			(*rows)[0].DisplayName = "mutated in native storage"
			if !bytes.Equal(frozen, old.source.recognition) {
				t.Fatal("native rows alias snapshot")
			}
			refuse(t, ex, "lmstudio", "demo", old)
		})
	}

	for _, engine := range []string{"ollama", "lmstudio"} {
		for _, drift := range []string{"root", "models directory", "launch", "run", "artifact metadata", "missing evidence", "wrong model"} {
			t.Run(engine+" rejects "+drift+" before content access", func(t *testing.T) {
				var ex *Executor
				var root string
				if engine == "ollama" {
					ex, root, _ = ollama(t)
				} else {
					ex, root, _ = lmstudio(t, "MLX")
				}
				old := capture(t, ex, engine)
				st, err := ex.state(engine)
				if err != nil {
					t.Fatal(err)
				}
				switch drift {
				case "root":
					if err := os.Rename(root, root+".old"); err != nil {
						t.Fatal(err)
					}
					if err := os.Mkdir(root, 0700); err != nil {
						t.Fatal(err)
					}
				case "models directory":
					st.mu.Lock()
					st.modelsDir = t.TempDir()
					st.mu.Unlock()
				case "launch":
					st.mu.Lock()
					st.launchDigest = "changed-launch"
					st.mu.Unlock()
				case "run":
					st.mu.Lock()
					st.runID = "changed-run"
					st.mu.Unlock()
				case "artifact metadata":
					file := old.source.artifacts[0]
					later := file.info.ModTime().Add(time.Second)
					if err := os.Chtimes(file.configured, later, later); err != nil {
						t.Fatal(err)
					}
				case "missing evidence":
					old.source = nil
				case "wrong model":
					old.Model = "another-model"
				}
				refuse(t, ex, engine, "demo", old)
			})
		}
		t.Run(engine+" cancelled capture performs no access", func(t *testing.T) {
			var ex *Executor
			if engine == "ollama" {
				ex, _, _ = ollama(t)
			} else {
				ex, _, _ = lmstudio(t, "MLX")
			}
			old := capture(t, ex, engine)
			cancelled, cancel := context.WithCancel(ctx)
			cancel()
			c := &counts{}
			list, err := ex.modelSourceFilesExpected(cancelled, engine, "demo", access(c), old)
			assertEmpty(t, list, err)
			if !errors.Is(err, context.Canceled) || c.reads != 0 || c.streams != 0 || c.dirs != 0 {
				t.Fatalf("cancelled access: err=%v counts=%+v", err, c)
			}
		})
	}

	for _, drift := range []string{"missing blob", "wrong blob size", "conflicting sizes", "invalid digest", "no blobs", "oversized manifest"} {
		t.Run("Ollama refuses incomplete selector: "+drift, func(t *testing.T) {
			ex, root, fx := ollama(t)
			manifest := filepath.Join(root, filepath.FromSlash(fx.manifestRel))
			var selected ollamaManifest
			if err := json.Unmarshal(fx.manifest, &selected); err != nil {
				t.Fatal(err)
			}
			switch drift {
			case "missing blob":
				for path := range fx.blobs {
					if err := os.Remove(filepath.Join(root, filepath.FromSlash(path))); err != nil {
						t.Fatal(err)
					}
					break
				}
			case "wrong blob size":
				selected.Config.Size++
			case "conflicting sizes":
				selected.Layers = append(selected.Layers, ollamaLayer{Digest: selected.Config.Digest, Size: selected.Config.Size + 1})
			case "invalid digest":
				selected.Config.Digest = "sha256:../invalid"
			case "no blobs":
				selected = ollamaManifest{}
			case "oversized manifest":
				writeTestFile(t, manifest, bytes.Repeat([]byte(" "), ollamaManifestLimit+1))
			}
			if drift != "oversized manifest" {
				data, err := json.Marshal(selected)
				if err != nil {
					t.Fatal(err)
				}
				writeTestFile(t, manifest, data)
			}
			c := &counts{}
			a := access(c)
			open := a.open
			a.open = func(path string) (servingArtifactFile, error) {
				if path != manifest {
					t.Fatalf("incomplete model opened artifact content: %s", path)
				}
				return open(path)
			}
			list, err := ex.modelSourceFilesWithAccess(ctx, "ollama", "demo", a)
			assertEmpty(t, list, err)
			if c.streams != 0 || c.dirs != 0 || (drift == "oversized manifest" && c.reads != 0) {
				t.Fatalf("unexpected access: %+v", c)
			}
		})
	}

	for _, drift := range []string{"distinct model keys", "conflicting duplicate path", "missing file", "unsafe path", "empty MLX"} {
		t.Run("LM Studio refuses incomplete or ambiguous set: "+drift, func(t *testing.T) {
			ex, root, rows := lmstudio(t, "GGUF")
			switch drift {
			case "distinct model keys":
				*rows = append(*rows, lmsListEntry{ModelKey: "other", IndexedModelIdentifier: "demo", Path: "owner/demo/other.gguf"})
			case "conflicting duplicate path":
				*rows = append(*rows, lmsListEntry{ModelKey: "demo", Path: (*rows)[0].Path, DisplayName: "conflicting"})
			case "missing file":
				(*rows)[0].Path = "owner/demo/missing.gguf"
			case "unsafe path":
				(*rows)[0].Path = "../escape.gguf"
			case "empty MLX":
				(*rows)[0].Path = "owner/empty"
				if err := os.Mkdir(filepath.Join(root, "owner/empty"), 0700); err != nil {
					t.Fatal(err)
				}
			}
			list, err := ex.modelSourceFilesWithAccess(ctx, "lmstudio", "demo", access(&counts{}))
			assertEmpty(t, list, err)
		})
	}

	t.Run("LM Studio preserves exact multirow grouping", func(t *testing.T) {
		ex, _, rows := lmstudio(t, "GGUF")
		*rows = append(*rows, lmsListEntry{ModelKey: "demo", Path: "owner/demo/other.gguf"})
		old := capture(t, ex, "lmstudio")
		want := []string{"owner/demo/model.gguf", "owner/demo/mmproj.gguf", "owner/demo/other.gguf"}
		var got []string
		for _, file := range old.Files {
			got = append(got, file.Path)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("grouped files=%v, want %v", got, want)
		}
		(*rows)[0], (*rows)[1] = (*rows)[1], (*rows)[0]
		refuse(t, ex, "lmstudio", "demo", old)
	})
	t.Run("Ollama withdrawn native recognition refuses before selector reads", func(t *testing.T) {
		ex, _, _ := ollama(t)
		old := capture(t, ex, "ollama")
		copyTestNativeModels(t, ex, "ollama")
		refuse(t, ex, "ollama", "demo", old)
	})
	t.Run("LM Studio freezes duplicate row multiplicity", func(t *testing.T) {
		ex, _, rows := lmstudio(t, "GGUF")
		old := capture(t, ex, "lmstudio")
		*rows = append(*rows, (*rows)[0])
		refuse(t, ex, "lmstudio", "demo", old)
	})
	t.Run("Ollama missing selector binding refuses before reads", func(t *testing.T) {
		ex, _, _ := ollama(t)
		old := capture(t, ex, "ollama")
		old.source.selectors[0].binding = nil
		refuse(t, ex, "ollama", "demo", old)
	})
	t.Run("LM Studio changed CLI refuses before native rows or enumeration", func(t *testing.T) {
		ex, _, rows := lmstudio(t, "GGUF")
		old := capture(t, ex, "lmstudio")
		st, err := ex.state("lmstudio")
		if err != nil {
			t.Fatal(err)
		}
		st.opMu.Lock()
		st.plat.Runtime.CLI = filepath.Join(t.TempDir(), "different-cli")
		st.opMu.Unlock()
		calls := 0
		lmsListModels = func(context.Context, *Executor, string) ([]lmsListEntry, error) {
			calls++
			return *rows, nil
		}
		refuse(t, ex, "lmstudio", "demo", old)
		if calls != 0 {
			t.Fatalf("changed CLI read native rows %d times", calls)
		}
	})
	t.Run("LM Studio revalidates all old metadata after native row read", func(t *testing.T) {
		ex, _, rows := lmstudio(t, "MLX")
		old := capture(t, ex, "lmstudio")
		calls := 0
		lmsListModels = func(context.Context, *Executor, string) ([]lmsListEntry, error) {
			calls++
			file := old.source.artifacts[0]
			later := file.info.ModTime().Add(time.Second)
			if err := os.Chtimes(file.configured, later, later); err != nil {
				t.Fatal(err)
			}
			return *rows, nil
		}
		refuse(t, ex, "lmstudio", "demo", old)
		if calls != 1 {
			t.Fatalf("native row reads=%d, want the pre-enumeration read", calls)
		}
	})

	for _, layout := range []string{"GGUF", "MLX"} {
		t.Run("LM Studio "+layout+" replaced artifact refuses before enumeration", func(t *testing.T) {
			ex, _, _ := lmstudio(t, layout)
			old := capture(t, ex, "lmstudio")
			replace(t, old.source.artifacts[0].configured, []byte("replacement"))
			refuse(t, ex, "lmstudio", "demo", old)
		})
		t.Run("LM Studio "+layout+" directory changes refuse before enumeration", func(t *testing.T) {
			ex, root, _ := lmstudio(t, layout)
			old := capture(t, ex, "lmstudio")
			writeTestFile(t, filepath.Join(root, "owner/demo/new-sidecar.json"), []byte("new"))
			refuse(t, ex, "lmstudio", "demo", old)
		})
		t.Run("LM Studio "+layout+" enumeration consumes shared budget", func(t *testing.T) {
			ex, _, _ := lmstudio(t, layout)
			c := &counts{}
			a := access(c)
			a.scan = &modelArtifactScan{entries: modelArtifactScanEntries - 1}
			list, err := ex.modelSourceFilesWithAccess(ctx, "lmstudio", "demo", a)
			assertEmpty(t, list, err)
			if c.dirs != 1 || c.closes != 1 || c.reads != 0 || c.streams != 0 {
				t.Fatalf("unbounded or leaked enumeration: %+v", c)
			}
		})
		t.Run("LM Studio "+layout+" cancellation closes enumeration", func(t *testing.T) {
			ex, _, _ := lmstudio(t, layout)
			cancelled, cancel := context.WithCancel(ctx)
			defer cancel()
			reads, closes, opens := 0, 0, 0
			a := servingFileAccess{openDir: func(path string) (servingDirectory, error) {
				f, err := os.Open(path)
				if err != nil {
					return nil, err
				}
				opens++
				return &countedLinkedDir{File: f, reads: &reads, closes: &closes, beforeRead: func(n int) {
					if n <= 0 || n > 128 {
						t.Fatalf("unbounded directory request: %d", n)
					}
				}, afterRead: func([]os.DirEntry, error) { cancel() }}, nil
			}}
			list, err := ex.modelSourceFilesWithAccess(cancelled, "lmstudio", "demo", a)
			assertEmpty(t, list, err)
			if !errors.Is(err, context.Canceled) || reads != 1 || opens != closes {
				t.Fatalf("cancelled enumeration: err=%v reads=%d opens=%d closes=%d", err, reads, opens, closes)
			}
		})
	}

	t.Run("LM Studio rejects a substituted directory handle before ReadDir", func(t *testing.T) {
		ex, _, _ := lmstudio(t, "GGUF")
		outside := t.TempDir()
		writeTestFile(t, filepath.Join(outside, "mmproj-forbidden.gguf"), []byte("forbidden"))
		reads, closes := 0, 0
		a := servingFileAccess{openDir: func(string) (servingDirectory, error) {
			f, err := os.Open(outside)
			if err != nil {
				return nil, err
			}
			return &countedLinkedDir{File: f, reads: &reads, closes: &closes}, nil
		}}
		list, err := ex.modelSourceFilesWithAccess(ctx, "lmstudio", "demo", a)
		assertEmpty(t, list, err)
		if reads != 0 || closes != 1 {
			t.Fatalf("substituted directory reads=%d closes=%d", reads, closes)
		}
	})
	t.Run("LM Studio charges nested directories to the same budget", func(t *testing.T) {
		ex, root, rows := lmstudio(t, "MLX")
		(*rows)[0].Path = "owner/budget"
		writeTestFile(t, filepath.Join(root, "owner/budget/child/a.json"), nil)
		writeTestFile(t, filepath.Join(root, "owner/budget/child/b.json"), nil)
		c := &counts{}
		a := access(c)
		a.scan = &modelArtifactScan{entries: modelArtifactScanEntries - 2}
		list, err := ex.modelSourceFilesWithAccess(ctx, "lmstudio", "demo", a)
		assertEmpty(t, list, err)
		if c.dirs == 0 || c.closes != 2 || c.reads != 0 || c.streams != 0 || a.scan.entries != modelArtifactScanEntries-1 {
			t.Fatalf("nested budget: entries=%d counts=%+v", a.scan.entries, c)
		}
	})
	t.Run("LM Studio initial capture rejects rows changed during enumeration", func(t *testing.T) {
		ex, _, rows := lmstudio(t, "GGUF")
		reads, closes := 0, 0
		a := servingFileAccess{openDir: func(path string) (servingDirectory, error) {
			f, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			return &countedLinkedDir{File: f, reads: &reads, closes: &closes, afterRead: func([]os.DirEntry, error) {
				(*rows)[0].DisplayName = "changed-during-enumeration"
			}}, nil
		}}
		list, err := ex.modelSourceFilesWithAccess(ctx, "lmstudio", "demo", a)
		assertEmpty(t, list, err)
		if reads == 0 || closes != 1 {
			t.Fatalf("initial enumeration reads=%d closes=%d", reads, closes)
		}
	})

	for _, expected := range []bool{false, true} {
		t.Run(fmt.Sprintf("Ollama final publication rejects withdrawn recognition expected=%t", expected), func(t *testing.T) {
			ex, _, _ := ollama(t)
			old := capture(t, ex, "ollama")
			selector := old.source.selectors[0]
			opens := 0
			a := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
				opens++
				if opens == 4 {
					copyTestNativeModels(t, ex, "ollama")
				}
				if path != selector.file.configured {
					t.Fatalf("unexpected content open: %s", path)
				}
				return os.Open(path)
			}}
			var list modelFileList
			var err error
			if expected {
				list, err = ex.modelSourceFilesExpected(ctx, "ollama", "demo", a, old)
			} else {
				list, err = ex.modelSourceFilesWithAccess(ctx, "ollama", "demo", a)
			}
			assertEmpty(t, list, err)
			if opens < 4 {
				t.Fatalf("final selector read not reached: opens=%d", opens)
			}
		})
	}

	for _, drift := range []string{"blob metadata", "selector bytes", "launch"} {
		t.Run("Ollama final publication rejects "+drift, func(t *testing.T) {
			ex, _, _ := ollama(t)
			old := capture(t, ex, "ollama")
			selector := old.source.selectors[0]
			st, err := ex.state("ollama")
			if err != nil {
				t.Fatal(err)
			}
			reads, streams, closes, opens := 0, 0, 0, 0
			changed := false
			a := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
				if path != selector.file.configured {
					t.Fatalf("unexpected artifact content open: %s", path)
				}
				opens++
				// Each bounded source read opens its before sample, bytes and
				// after sample. Change another input during the final re-read.
				if opens == 4 {
					changed = true
					switch drift {
					case "blob metadata":
						file := old.source.artifacts[0]
						later := file.info.ModTime().Add(time.Second)
						if err := os.Chtimes(file.configured, later, later); err != nil {
							t.Fatal(err)
						}
					case "selector bytes":
						data := bytes.Replace(selector.bytes, []byte(`"schemaVersion":2`), []byte(`"schemaVersion":3`), 1)
						if bytes.Equal(data, selector.bytes) {
							t.Fatal("fixture has no schema version to change")
						}
						writeTestFile(t, path, data)
						if err := os.Chtimes(path, selector.file.info.ModTime(), selector.file.info.ModTime()); err != nil {
							t.Fatal(err)
						}
					case "launch":
						st.mu.Lock()
						st.launchDigest = "changed-at-publication"
						st.mu.Unlock()
					}
				}
				f, err := os.Open(path)
				if err != nil {
					return nil, err
				}
				return &countedLinkedFile{File: f, reads: &reads, streams: &streams, closes: &closes}, nil
			}}
			list, err := ex.modelSourceFilesExpected(ctx, "ollama", "demo", a, old)
			assertEmpty(t, list, err)
			if !changed || reads == 0 || streams != 0 || closes != opens {
				t.Fatalf("publication mutation=%v reads=%d streams=%d closes=%d opens=%d", changed, reads, streams, closes, opens)
			}
		})
	}

	for _, stage := range []string{"initial", "expected", "publication"} {
		t.Run("LM Studio rejects matching pathless row at "+stage, func(t *testing.T) {
			ex, _, rows := lmstudio(t, "GGUF")
			old := capture(t, ex, "lmstudio")
			invalid := lmsListEntry{ModelKey: "other", IndexedModelIdentifier: "demo"}
			calls := 0
			lmsListModels = func(context.Context, *Executor, string) ([]lmsListEntry, error) {
				calls++
				if stage != "publication" || calls == 2 {
					return append(append([]lmsListEntry(nil), (*rows)...), invalid), nil
				}
				return *rows, nil
			}
			c := &counts{}
			var list modelFileList
			var err error
			if stage == "initial" {
				list, err = ex.modelSourceFilesWithAccess(ctx, "lmstudio", "demo", access(c))
			} else {
				list, err = ex.modelSourceFilesExpected(ctx, "lmstudio", "demo", access(c), old)
			}
			assertEmpty(t, list, err)
			if stage == "publication" {
				if calls != 2 || c.dirs == 0 {
					t.Fatalf("publication calls=%d counts=%+v", calls, c)
				}
			} else if calls != 1 || c.dirs != 0 || c.reads != 0 || c.streams != 0 {
				t.Fatalf("invalid native row reached source content: calls=%d counts=%+v", calls, c)
			}
		})
	}

	for _, drift := range []string{"rows", "artifact metadata", "directory", "run"} {
		t.Run("LM Studio final publication rejects "+drift, func(t *testing.T) {
			ex, root, rows := lmstudio(t, "MLX")
			old := capture(t, ex, "lmstudio")
			st, err := ex.state("lmstudio")
			if err != nil {
				t.Fatal(err)
			}
			calls := 0
			lmsListModels = func(context.Context, *Executor, string) ([]lmsListEntry, error) {
				calls++
				if calls == 2 {
					switch drift {
					case "rows":
						(*rows)[0].DisplayName = "changed-at-publication"
					case "artifact metadata":
						file := old.source.artifacts[0]
						later := file.info.ModTime().Add(time.Second)
						if err := os.Chtimes(file.configured, later, later); err != nil {
							t.Fatal(err)
						}
					case "directory":
						writeTestFile(t, filepath.Join(root, "owner/demo/new.json"), []byte("new"))
					case "run":
						st.mu.Lock()
						st.runID = "changed-at-publication"
						st.mu.Unlock()
					}
				}
				return *rows, nil
			}
			c := &counts{}
			list, err := ex.modelSourceFilesExpected(ctx, "lmstudio", "demo", access(c), old)
			assertEmpty(t, list, err)
			if calls != 2 || c.dirs == 0 || c.reads != 0 || c.streams != 0 {
				t.Fatalf("publication calls=%d counts=%+v", calls, c)
			}
		})
	}

	t.Run("LM Studio propagates companion enumeration errors", func(t *testing.T) {
		ex, _, _ := lmstudio(t, "GGUF")
		refused := errors.New("synthetic directory refusal")
		a := servingFileAccess{openDir: func(string) (servingDirectory, error) { return nil, refused }}
		list, err := ex.modelSourceFilesWithAccess(ctx, "lmstudio", "demo", a)
		assertEmpty(t, list, err)
		if !errors.Is(err, refused) {
			t.Fatalf("directory error=%v, want %v", err, refused)
		}
	})

	t.Run("LM Studio does not enumerate an outside GGUF target parent", func(t *testing.T) {
		ex, root, _ := lmstudio(t, "GGUF")
		outside := filepath.Join(t.TempDir(), "external")
		writeTestFile(t, filepath.Join(outside, "model.gguf"), []byte("external model"))
		writeTestFile(t, filepath.Join(outside, "mmproj-private.gguf"), []byte("private"))
		path := filepath.Join(root, "owner/demo/model.gguf")
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(filepath.Join(outside, "model.gguf"), path); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		c := &counts{}
		seenDirs := map[string]bool{}
		base := access(c)
		openDir := base.openDir
		base.openDir = func(path string) (servingDirectory, error) { seenDirs[path] = true; return openDir(path) }
		list, err := ex.modelSourceFilesWithAccess(ctx, "lmstudio", "demo", base)
		if err != nil || len(list.Files) != 2 || list.Files[0].Path != "owner/demo/model.gguf" || list.Files[0].abs != filepath.Join(outside, "model.gguf") || list.Files[1].Path != "owner/demo/mmproj.gguf" {
			t.Fatalf("selected namespace files=%v err=%v", list.Files, err)
		}
		if c.reads != 0 || c.streams != 0 || !reflect.DeepEqual(seenDirs, map[string]bool{filepath.Join(root, "owner/demo"): true}) {
			t.Fatalf("unexpected content or directory access: counts=%+v paths=%v", c, seenDirs)
		}
	})
	t.Run("LM Studio selected directory link retains its exact tree", func(t *testing.T) {
		ex, root, rows := lmstudio(t, "MLX")
		(*rows)[0].Path = "owner/linked"
		if err := os.Symlink(filepath.Join(root, "owner/demo"), filepath.Join(root, "owner/linked")); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		c := &counts{}
		seenDirs := map[string]bool{}
		base := access(c)
		openDir := base.openDir
		base.openDir = func(path string) (servingDirectory, error) { seenDirs[path] = true; return openDir(path) }
		list, err := ex.modelSourceFilesWithAccess(ctx, "lmstudio", "demo", base)
		var paths []string
		for _, file := range list.Files {
			paths = append(paths, file.Path)
		}
		want := []string{"owner/linked/empty.json", "owner/linked/tokenizer/vocab.json", "owner/linked/weights.safetensors"}
		if err != nil || !reflect.DeepEqual(paths, want) || list.Files[0].Size != 0 {
			t.Fatalf("linked tree files=%v err=%v", paths, err)
		}
		if c.reads != 0 || c.streams != 0 || !reflect.DeepEqual(seenDirs, map[string]bool{filepath.Join(root, "owner/linked"): true, filepath.Join(root, "owner/linked/tokenizer"): true}) {
			t.Fatalf("unexpected tree access: counts=%+v paths=%v", c, seenDirs)
		}
	})
}
