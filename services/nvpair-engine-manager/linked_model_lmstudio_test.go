// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"testing"
)

func TestLinkedModelLMStudioLinks(t *testing.T) {
	root := t.TempDir()
	selected := filepath.Join(root, "owner", "gguf", "model.gguf")
	if err := os.MkdirAll(filepath.Dir(selected), 0700); err != nil {
		t.Fatal(err)
	}
	external := filepath.Join(t.TempDir(), "model.gguf")
	writeTestFile(t, external, []byte("selected"))
	if err := os.Symlink(external, selected); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	writeTestFile(t, filepath.Join(filepath.Dir(external), "mmproj-unrelated.gguf"), []byte("forbidden sibling"))
	projector := filepath.Join(t.TempDir(), "projector")
	writeTestFile(t, projector, []byte("projector"))
	if err := os.Symlink(projector, filepath.Join(filepath.Dir(selected), "mmproj.gguf")); err != nil {
		t.Fatal(err)
	}
	mlx := filepath.Join(root, "owner", "mlx")
	outside := t.TempDir()
	writeTestFile(t, filepath.Join(outside, "weights.safetensors"), []byte("weights"))
	writeTestFile(t, filepath.Join(outside, "empty.json"), nil)
	nested := t.TempDir()
	writeTestFile(t, filepath.Join(nested, "vocab.json"), []byte("vocab"))
	if err := os.Symlink(nested, filepath.Join(outside, "tokenizer")); err != nil {
		t.Fatal(err)
	}
	fileTarget := filepath.Join(t.TempDir(), "config")
	writeTestFile(t, fileTarget, []byte("config"))
	if err := os.Symlink(fileTarget, filepath.Join(outside, "config.json")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, mlx); err != nil {
		t.Fatal(err)
	}
	previous := lmsListModels
	rows := []lmsListEntry{{ModelKey: "gguf", Path: "owner/gguf/model.gguf"}, {ModelKey: "mlx", Path: "owner/mlx"}}
	lmsListModels = func(context.Context, *Executor, string) ([]lmsListEntry, error) { return rows, nil }
	t.Cleanup(func() { lmsListModels = previous })
	ex := copyTestExecutor(t, "lmstudio", root, true)
	for _, tc := range []struct {
		name, model string
		count       int
	}{{"linked GGUF and projector", "gguf", 2}, {"linked MLX root", "mlx", 4}} {
		t.Run(tc.name, func(t *testing.T) {
			list, err := ex.modelSourceFiles(context.Background(), "lmstudio", tc.model)
			if err != nil || len(list.Files) != tc.count {
				t.Fatalf("files=%v err=%v", list.Files, err)
			}
		})
	}
	t.Run("exact HTTP and copy bytes", func(t *testing.T) {
		plentyOfDisk(t)
		_, client := modelFileServer(t, ex, nil)
		want := map[string][]byte{"owner/gguf/model.gguf": []byte("selected"), "owner/gguf/mmproj.gguf": []byte("projector"), "owner/mlx/weights.safetensors": []byte("weights"), "owner/mlx/empty.json": nil, "owner/mlx/tokenizer/vocab.json": []byte("vocab"), "owner/mlx/config.json": []byte("config")}
		for _, model := range []string{"gguf", "mlx"} {
			list, err := client.modelFiles(context.Background(), "lmstudio", model, nil)
			if err != nil || list.Layout != "" {
				t.Fatalf("listing=%+v err=%v", list, err)
			}
			for _, file := range list.Files {
				expected, ok := want[file.Path]
				if !ok {
					t.Fatalf("unrelated file %s", file.Path)
				}
				body, _, err := client.modelFile(context.Background(), "lmstudio", model, file.Path, list.ExportID, 0)
				if err != nil {
					t.Fatal(err)
				}
				got, err := io.ReadAll(body)
				if err != nil {
					t.Fatal(err)
				}
				if err := body.Close(); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got, expected) {
					t.Fatalf("bytes differ for %s", file.Path)
				}
			}
			dir := t.TempDir()
			dest := copyTestExecutor(t, "lmstudio", dir, true)
			if _, err := dest.CopyModelFrom(context.Background(), client, "lmstudio", model, nil); err != nil {
				t.Fatal(err)
			}
			for _, file := range list.Files {
				got, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(file.Path)))
				if err != nil || !bytes.Equal(got, want[file.Path]) {
					t.Fatalf("copy %s: %v", file.Path, err)
				}
			}
		}
	})
	t.Run("old issued target refuses before reads and fresh listing succeeds", func(t *testing.T) {
		_, client := modelFileServer(t, ex, nil)
		list, err := client.modelFiles(context.Background(), "lmstudio", "gguf", nil)
		if err != nil {
			t.Fatal(err)
		}
		replacement := filepath.Join(t.TempDir(), "model")
		writeTestFile(t, replacement, []byte("replacement"))
		if err := os.Remove(selected); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(replacement, selected); err != nil {
			t.Fatal(err)
		}
		counts := &linkedHashCounts{}
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			f, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			return &linkedHashFile{File: f, counts: counts, forbidden: true}, nil
		}}
		query := url.Values{"engine": {"lmstudio"}, "model": {"gguf"}, "path": {"owner/gguf/model.gguf"}, "exportId": {list.ExportID}}
		unlisted := url.Values{"engine": {"lmstudio"}, "model": {"gguf"}, "path": {"owner/gguf/mmproj-unrelated.gguf"}, "exportId": {list.ExportID}}
		response := httptest.NewRecorder()
		(&controlServer{exec: ex, access: access}).handleModelFile(response, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+unlisted.Encode(), nil))
		if response.Code != http.StatusNotFound || counts.bytes.Load() != 0 {
			t.Fatalf("unlisted status=%d reads=%d", response.Code, counts.bytes.Load())
		}
		response = httptest.NewRecorder()
		(&controlServer{exec: ex, access: access}).handleModelFile(response, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+query.Encode(), nil))
		if response.Code != http.StatusConflict || counts.bytes.Load() != 0 {
			t.Fatalf("old status=%d reads=%d", response.Code, counts.bytes.Load())
		}
		fresh, err := client.modelFiles(context.Background(), "lmstudio", "gguf", nil)
		if err != nil {
			t.Fatal(err)
		}
		body, _, err := client.modelFile(context.Background(), "lmstudio", "gguf", "owner/gguf/model.gguf", fresh.ExportID, 0)
		if err != nil {
			t.Fatal(err)
		}
		got, err := io.ReadAll(body)
		if err != nil {
			t.Fatal(err)
		}
		if err := body.Close(); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, []byte("replacement")) {
			t.Fatal("fresh bytes differ")
		}
		restarted := copyTestExecutor(t, "lmstudio", root, true)
		query.Set("exportId", fresh.ExportID)
		response = httptest.NewRecorder()
		(&controlServer{exec: restarted}).handleModelFile(response, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+query.Encode(), nil))
		if response.Code != http.StatusConflict {
			t.Fatalf("restart status=%d", response.Code)
		}
		_, restartedClient := modelFileServer(t, restarted, nil)
		if _, err := restartedClient.modelFiles(context.Background(), "lmstudio", "gguf", nil); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("linked MLX directory retarget refuses before content or enumeration", func(t *testing.T) {
		list, err := ex.modelSourceFiles(context.Background(), "lmstudio", "mlx")
		if err != nil {
			t.Fatal(err)
		}
		replacement := t.TempDir()
		writeTestFile(t, filepath.Join(replacement, "weights.safetensors"), []byte("forbidden"))
		if err := os.Remove(mlx); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(replacement, mlx); err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := os.Remove(mlx); err != nil {
				t.Error(err)
			}
			if err := os.Symlink(outside, mlx); err != nil {
				t.Error(err)
			}
		}()
		reads, dirs := 0, 0
		access := servingFileAccess{
			open: func(path string) (servingArtifactFile, error) {
				file, err := os.Open(path)
				if err != nil {
					return nil, err
				}
				return &linkedReadCounter{File: file, reads: &reads}, nil
			},
			openDir: func(path string) (servingDirectory, error) {
				file, err := os.Open(path)
				if err != nil {
					return nil, err
				}
				return &hfDirectoryCounter{File: file, reads: &dirs}, nil
			},
		}
		st, store, err := ex.modelStoreFor("lmstudio")
		if err != nil {
			t.Fatal(err)
		}
		changed, err := ex.lmstudioSourceFilesBound(context.Background(), st, store, "mlx", access, &list)
		if err == nil || len(changed.Files) != 0 || reads != 0 || dirs != 0 {
			t.Fatalf("retarget files=%v err=%v reads=%d dirs=%d", changed.Files, err, reads, dirs)
		}
	})
	t.Run("native row ambiguity and changed row refuse", func(t *testing.T) {
		list, err := ex.modelSourceFiles(context.Background(), "lmstudio", "mlx")
		if err != nil {
			t.Fatal(err)
		}
		rows = append(rows, lmsListEntry{ModelKey: "mlx", Path: "owner/other"})
		defer func() { rows = rows[:2] }()
		changed, err := ex.modelSourceFilesExpected(context.Background(), "lmstudio", "mlx", servingFileAccess{}, list)
		if err == nil || len(changed.Files) != 0 {
			t.Fatalf("ambiguous rows returned %v err=%v", changed.Files, err)
		}
	})
	t.Run("aliases cycles and incomplete trees publish nothing", func(t *testing.T) {
		for _, tc := range []struct{ name, target string }{{"alias", nested}, {"cycle", outside}, {"broken", filepath.Join(t.TempDir(), "missing")}} {
			t.Run(tc.name, func(t *testing.T) {
				path := filepath.Join(outside, "extra")
				if err := os.Symlink(tc.target, path); err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := os.Remove(path); err != nil {
						t.Error(err)
					}
				}()
				list, err := ex.modelSourceFiles(context.Background(), "lmstudio", "mlx")
				if err == nil || len(list.Files) != 0 || list.source != nil {
					t.Fatalf("partial files=%v err=%v", list.Files, err)
				}
			})
		}
	})
	t.Run("budget and cancellation publish nothing", func(t *testing.T) {
		access := servingFileAccess{scan: &modelArtifactScan{entries: modelArtifactScanEntries}}
		list, err := ex.modelSourceFilesWithAccess(context.Background(), "lmstudio", "mlx", access)
		if err == nil || len(list.Files) != 0 || list.source != nil {
			t.Fatalf("budget files=%v err=%v", list.Files, err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		list, err = ex.modelSourceFilesWithAccess(ctx, "lmstudio", "mlx", servingFileAccess{})
		if err == nil || len(list.Files) != 0 || list.source != nil {
			t.Fatalf("cancel files=%v err=%v", list.Files, err)
		}
	})
}
