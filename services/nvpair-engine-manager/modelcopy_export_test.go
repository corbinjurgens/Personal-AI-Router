// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/durablefile"
	"nvpair-shared/modelconfig"
)

func TestModelCopySelectedExport(t *testing.T) {
	mutate := func(t *testing.T, path string, offset int64) {
		t.Helper()
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		file, err := os.OpenFile(path, os.O_WRONLY, 0)
		if err != nil {
			t.Fatal(err)
		}
		_, writeErr := file.WriteAt([]byte("changed"), offset)
		closeErr := file.Close()
		if writeErr != nil || closeErr != nil {
			t.Fatalf("mutation: write=%v close=%v", writeErr, closeErr)
		}
		if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
			t.Fatal(err)
		}
	}
	t.Run("preset ignores unrelated mutation while launch refuses", func(t *testing.T) {
		f := newLlamaServingFixture(t)
		for i := range f.request.Settings.Serving.Models {
			model := &f.request.Settings.Serving.Models[i]
			data, err := os.ReadFile(filepath.Join(f.root, model.ModelID+".gguf"))
			if err != nil {
				t.Fatal(err)
			}
			model.Artifacts[0].SHA256 = sha256Hex(data)
		}
		f.request.Settings.Serving.Models[0].Artifacts[0].Role = "model-shard"
		applyLlamaServing(t, f, approveLlamaServing(t, f))
		st := settingsState(t, f.ex)
		st.modelsDir = f.root
		mutate(t, filepath.Join(f.root, "weak.gguf"), 0)
		command, err := launchForState(st, st.port)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := compilePreparedServing(st, command, st.plat.Runtime, true); err == nil {
			t.Fatal("full serving accepted unrelated sampled mutation")
		}
		forbidden, selected := 0, 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			if path == filepath.Join(f.root, "weak.gguf") || path == command.Bin {
				forbidden++
				return nil, errors.New("unrelated input opened")
			}
			selected++
			return os.Open(path)
		}}
		list, err := f.ex.llamaExportFilesWithAccess(context.Background(), st, engineModelStore{engine: "fake", adapter: "llamacpp", root: f.root}, "medium", access)
		if err != nil {
			t.Fatal(err)
		}
		if forbidden != 0 || selected == 0 || len(list.Files) != 1 || list.Files[0].asserted == "" {
			t.Fatalf("selected export: forbidden=%d selected=%d list=%+v", forbidden, selected, list)
		}
		if err := hashModelFiles(context.Background(), &list, func(string, int) {}); err != nil {
			t.Fatal(err)
		}
		assertNoLlamaSpawn(t, f)
	})
	t.Run("typed exports after restart without native build reads", func(t *testing.T) {
		f := newModelLoadFixture(t)
		rt := convertModelLoadFixture(t, f)
		f.st.plat.Runtime = rt
		mutate(t, filepath.Join(f.root, "weak.gguf"), 0)
		command := typedFixtureCommand(t, f, rt)
		if _, err := compilePreparedServing(f.st, command, rt, true); err == nil {
			t.Fatal("full typed launch accepted unrelated sampled mutation")
		}
		persisted, err := json.Marshal(rt.ModelLoads)
		if err != nil {
			t.Fatal(err)
		}
		var loads []modelconfig.ModelLoad
		if err := json.Unmarshal(persisted, &loads); err != nil {
			t.Fatal(err)
		}
		cold := snapshotLlamaState(f.st)
		cold.plat.Runtime.ModelLoads = loads
		cold.plat.Runtime.modelLoadOwner = nil
		cold.plat.Runtime.servingSnapshot = nil
		missingBuild := filepath.Join(f.root, "missing-executable")
		cold.binPath = missingBuild
		cold.plat.Runtime.Bin = missingBuild
		forbidden, selected := 0, 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			if path != filepath.Join(f.root, "medium.gguf") {
				forbidden++
				return nil, errors.New("non-selected input opened")
			}
			selected++
			return os.Open(path)
		}}
		store := engineModelStore{engine: "fake", adapter: "llamacpp", root: f.root}
		list, err := f.ex.llamaExportFilesWithAccess(context.Background(), cold, store, "medium-model", access)
		if err != nil {
			t.Fatal(err)
		}
		if selected == 0 || forbidden != 0 || len(list.Files) != 1 {
			t.Fatalf("typed export: selected=%d forbidden=%d list=%+v", selected, forbidden, list)
		}
		if _, err := f.ex.llamaExportFilesBound(context.Background(), cold, store, "medium-model", access, &list); err != nil {
			t.Fatalf("selected recheck: %v", err)
		}
		mutate(t, filepath.Join(f.root, "medium.gguf"), 0)
		if _, err := f.ex.llamaExportFilesWithAccess(context.Background(), cold, store, "medium-model", access); err == nil {
			t.Fatal("selected sampled mutation accepted")
		}
	})
	t.Run("selected full hash rejects mutation outside binding samples", func(t *testing.T) {
		f := newLlamaServingFixture(t)
		path := filepath.Join(f.root, "medium.gguf")
		data := testBytes(300_000, 4)
		writeTestFile(t, path, data)
		f.request.Settings.Serving.Models[0].Artifacts[0].SHA256 = sha256Hex(data)
		applyLlamaServing(t, f, approveLlamaServing(t, f))
		st := settingsState(t, f.ex)
		st.modelsDir = f.root
		mutate(t, path, 150_000)
		list, err := f.ex.modelSourceFiles(context.Background(), "fake", "medium")
		if err != nil {
			t.Fatalf("unchanged samples rejected before full hashing: %v", err)
		}
		if err := hashModelFiles(context.Background(), &list, func(string, int) {}); err == nil || !strings.Contains(err.Error(), "full SHA256 disagrees") {
			t.Fatalf("selected full hash mismatch: %v", err)
		}
	})
}

const reportedCopyAlias = "tiel-coder-35b-a3b-msi-ctx32768"

func TestModelCopyCustomTypedBindingUsesCanonicalResolver(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	f.st.plat.Runtime = rt
	store := engineModelStore{engine: "fake", adapter: "llamacpp", root: f.root}
	list, err := f.ex.llamaExportFiles(context.Background(), f.st, store, "medium-model")
	if err != nil {
		t.Fatal(err)
	}
	if len(list.Files) != 1 || list.Files[0].configured != filepath.Join(f.root, "medium.gguf") || list.Files[0].SHA256 != "" || list.Files[0].asserted != "" {
		t.Fatalf("typed resolver invented hashes or lost exact artifact: %+v", list)
	}
	writeInspectionGGUF(t, filepath.Join(f.root, "medium.gguf"), "llama", 0.75, 4096)
	if _, err := f.ex.llamaExportFiles(context.Background(), f.st, store, "medium-model"); err == nil {
		t.Fatal("changed typed artifact fell back to another source")
	}
}

func TestModelCopyCustomTypedRouterPreservesUnrelatedReceiptExport(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	rt.Serving = servingSettingsFixture() // unrelated and unapproved
	f.st.plat.Runtime = rt
	data, err := os.ReadFile(filepath.Join(f.root, "medium.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	list := modelFileList{Engine: "fake", Model: "receipt-only", Layout: llamaArtifactsLayout, Files: []modelFile{{Role: "model", Path: "model/copied.gguf", Size: int64(len(data)), SHA256: sha256Hex(data)}}}
	id, err := artifactExportID(list)
	if err != nil {
		t.Fatal(err)
	}
	list.ExportID = id
	base := copyAliasDir(f.root, list.Model)
	artifact := filepath.Join(base, id, "model", "copied.gguf")
	writeTestFile(t, artifact, data)
	receipt, err := json.Marshal(copyReceipt{Version: 1, List: list})
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, filepath.Join(base, "copy.json"), receipt)
	exported, err := f.ex.llamaExportFiles(context.Background(), f.st, engineModelStore{engine: "fake", adapter: "llamacpp", root: f.root}, list.Model)
	if err != nil {
		t.Fatal(err)
	}
	if len(exported.Files) != 1 || exported.Files[0].configured != artifact || exported.Files[0].asserted != list.Files[0].SHA256 {
		t.Fatalf("receipt-only re-export lost its source or asserted digest: %+v", exported)
	}
}

type customCopyFixture struct {
	source                  *Executor
	preset, artifact, store string
}

func newCustomCopyFixture(t *testing.T) customCopyFixture {
	t.Helper()
	store, external := t.TempDir(), t.TempDir()
	artifact := filepath.Join(external, "huggingface", "hub", "model.gguf")
	writeTestFile(t, artifact, testBytes(300_000, 4))
	preset := filepath.Join(external, "models.ini")
	writeTestFile(t, preset, []byte("["+reportedCopyAlias+"]\nmodel = huggingface/hub/model.gguf\n"))
	ex := copyTestExecutor(t, "llamacpp", store, true)
	st, err := ex.state("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	st.plat.Runtime.Args = []string{"--models-preset", preset}
	return customCopyFixture{ex, preset, artifact, store}
}

func customCopyList(t *testing.T, f customCopyFixture) modelFileList {
	t.Helper()
	list, err := f.source.modelSourceFiles(context.Background(), "llamacpp", reportedCopyAlias)
	if err != nil {
		t.Fatal(err)
	}
	if err := hashModelFiles(context.Background(), &list, func(string, int) {}); err != nil {
		t.Fatal(err)
	}
	return list
}

func customDestination(t *testing.T) (*Executor, engineModelStore) {
	t.Helper()
	ex := copyTestExecutor(t, "llamacpp", t.TempDir(), true)
	_, store, err := ex.modelStoreFor("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	return ex, store
}

func assertCustomReceipt(t *testing.T, store engineModelStore, list modelFileList) {
	t.Helper()
	receipt, err := readCopyReceipt(store, reportedCopyAlias)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.List.ExportID != list.ExportID || len(receipt.List.Files) != len(list.Files) {
		t.Fatalf("receipt=%+v", receipt)
	}
	for _, f := range list.Files {
		path := filepath.Join(copyAliasDir(store.root, reportedCopyAlias), list.ExportID, filepath.FromSlash(f.Path))
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if int64(len(data)) != f.Size || sha256Hex(data) != f.SHA256 {
			t.Fatalf("copied artifact %s differs", f.Path)
		}
	}
}

func TestModelCopyCustomAlias(t *testing.T) {
	plentyOfDisk(t)
	t.Run("normalized preset flags retain exact custom alias precedence", func(t *testing.T) {
		for _, flag := range []string{"--models_preset", "--models_preset="} {
			t.Run(flag, func(t *testing.T) {
				f := newCustomCopyFixture(t)
				const alias = "owner/custom-GGUF:Q4_K_M"
				writeTestFile(t, f.preset, []byte("["+alias+"]\nmodel = huggingface/hub/model.gguf\n"))
				writeLlamaCache(t, f.store, "owner/custom-GGUF", strings.Repeat("a", 40), map[string][]byte{"cached-Q4_K_M.gguf": []byte("different cached model")})
				st, err := f.source.state("llamacpp")
				if err != nil {
					t.Fatal(err)
				}
				if strings.HasSuffix(flag, "=") {
					st.plat.Runtime.Args = []string{flag + f.preset}
				} else {
					st.plat.Runtime.Args = []string{flag, f.preset}
				}
				list, err := f.source.modelSourceFiles(context.Background(), "llamacpp", alias)
				if err != nil {
					t.Fatal(err)
				}
				if list.Layout != llamaArtifactsLayout || len(list.Files) != 1 || list.Files[0].Path != "model/model.gguf" {
					t.Fatalf("configured alias fell through to cache: %+v", list)
				}
				writeTestFile(t, f.preset, []byte("["+alias+"]\nmodel = missing.gguf\n"))
				if _, err := f.source.modelSourceFiles(context.Background(), "llamacpp", alias); err == nil {
					t.Fatal("failed configured alias fell through to cached HF model")
				}
			})
		}
	})
	t.Run("reported alias uses exact external HF cache binding", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		list := customCopyList(t, f)
		if list.Layout != llamaArtifactsLayout || !validSHA256(list.ExportID) || list.Model != reportedCopyAlias || list.Files[0].Path != "model/model.gguf" {
			t.Fatalf("list=%+v", list)
		}
		data, err := json.Marshal(list)
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(data, []byte(filepath.Dir(f.artifact))) || bytes.Contains(data, []byte(f.preset)) {
			t.Fatal("listing disclosed host paths")
		}
		_, client := modelFileServer(t, f.source, nil)
		dst, store := customDestination(t)
		res, err := dst.CopyModelFrom(context.Background(), client, "llamacpp", reportedCopyAlias, nil)
		if err != nil {
			t.Fatal(err)
		}
		if res.Layout != llamaArtifactsLayout || res.ExportID != list.ExportID || !res.RequiresConfiguration || res.Durability != copyDurabilityConfirmed {
			t.Fatalf("result=%+v", res)
		}
		assertCustomReceipt(t, store, list)
		after, err := os.ReadFile(f.artifact)
		if err != nil {
			t.Fatal(err)
		}
		if sha256Hex(after) != list.Files[0].SHA256 {
			t.Fatal("source bytes changed")
		}
		if matches, err := filepath.Glob(filepath.Join(store.root, "models--*")); err != nil || len(matches) != 0 {
			t.Fatalf("invented HF directories=%v err=%v", matches, err)
		}
		st, err := dst.state("llamacpp")
		if err != nil {
			t.Fatal(err)
		}
		if st.running {
			t.Fatal("copy started engine")
		}
		reexport, err := dst.modelSourceFiles(context.Background(), "llamacpp", reportedCopyAlias)
		if err != nil {
			t.Fatal(err)
		}
		if err := hashModelFiles(context.Background(), &reexport, func(string, int) {}); err != nil {
			t.Fatal(err)
		}
		if reexport.ExportID != list.ExportID {
			t.Fatal("receipt re-export changed artifact identity")
		}
	})
	t.Run("split model and draft retain native order", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		base := filepath.Dir(f.preset)
		writeTestFile(t, f.preset, []byte("["+reportedCopyAlias+"]\nmodel = model-00001-of-00002.gguf\nmodel-draft = draft-00001-of-00002.gguf\n"))
		for _, name := range []string{"model-00001-of-00002.gguf", "model-00002-of-00002.gguf", "draft-00001-of-00002.gguf", "draft-00002-of-00002.gguf"} {
			writeTestFile(t, filepath.Join(base, name), []byte(name))
		}
		list := customCopyList(t, f)
		want := []string{"model/model-00001-of-00002.gguf", "model/model-00002-of-00002.gguf", "draft/draft-00001-of-00002.gguf", "draft/draft-00002-of-00002.gguf"}
		for i, path := range want {
			if list.Files[i].Path != path {
				t.Fatalf("artifact %d=%s", i, list.Files[i].Path)
			}
		}
		dst, store := customDestination(t)
		src := sourceFromExecutor(t, f.source, "llamacpp", reportedCopyAlias)
		if _, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil); err != nil {
			t.Fatal(err)
		}
		assertCustomReceipt(t, store, list)
	})
	t.Run("identical repeat verifies files without replacing receipt", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		src := sourceFromExecutor(t, f.source, "llamacpp", reportedCopyAlias)
		dst, store := customDestination(t)
		if _, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil); err != nil {
			t.Fatal(err)
		}
		useEngineFileReplacer(t, durablefile.Replacer{Replace: func(string, string) error { return errors.New("unexpected replacement") }})
		res, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil)
		if err != nil {
			t.Fatal(err)
		}
		if res.FilesSkipped != len(src.list.Files) || res.BytesCopied != 0 || res.Durability != copyDurabilityUncertain {
			t.Fatalf("repeat=%+v", res)
		}
		assertCustomReceipt(t, store, src.list)
	})
	t.Run("receipt publication is last", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		src := sourceFromExecutor(t, f.source, "llamacpp", reportedCopyAlias)
		dst, store := customDestination(t)
		src.onFetch = func(string) {
			if _, err := os.Stat(filepath.Join(copyAliasDir(store.root, reportedCopyAlias), "copy.json")); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("premature receipt: %v", err)
			}
		}
		if _, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil); err != nil {
			t.Fatal(err)
		}
		assertCustomReceipt(t, store, src.list)
	})
	t.Run("interruption preserves partial and resumes with export identity", func(t *testing.T) {
		fastRetries(t, 1)
		f := newCustomCopyFixture(t)
		list := customCopyList(t, f)
		dst, store := customDestination(t)
		var offsets []string
		cut := interruptAfter("model/model.gguf", 100_000)
		_, client := modelFileServer(t, f.source, func(next http.Handler) http.Handler {
			return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == controlModelFilePath {
					if r.URL.Query().Get("exportId") != list.ExportID {
						t.Error("missing export identity")
					}
					offsets = append(offsets, r.Header.Get("Range"))
				}
				cut(next).ServeHTTP(w, r)
			})
		})
		if _, err := dst.CopyModelFrom(context.Background(), client, "llamacpp", reportedCopyAlias, nil); err == nil {
			t.Fatal("interrupted copy succeeded")
		}
		if _, err := readCopyReceipt(store, reportedCopyAlias); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("interrupted receipt=%v", err)
		}
		if _, err := dst.CopyModelFrom(context.Background(), client, "llamacpp", reportedCopyAlias, nil); err != nil {
			t.Fatal(err)
		}
		if len(offsets) != 2 || offsets[0] != "" || !strings.HasPrefix(offsets[1], "bytes=") || offsets[1] == "bytes=0-" {
			t.Fatalf("ranges=%v", offsets)
		}
		assertCustomReceipt(t, store, list)
	})
}

func TestModelCopyCustomAliasSecurity(t *testing.T) {
	plentyOfDisk(t)
	t.Run("bare alias has no guessed catalog identity", func(t *testing.T) {
		ex := copyTestExecutor(t, "llamacpp", t.TempDir(), true)
		_, err := ex.modelSourceFiles(context.Background(), "llamacpp", reportedCopyAlias)
		if !errors.Is(err, errNoArtifactBinding) {
			t.Fatalf("err=%v", err)
		}
	})
	refusePreset := func(name, text string) {
		t.Run(name, func(t *testing.T) {
			f := newCustomCopyFixture(t)
			writeTestFile(t, f.preset, []byte(text))
			_, err := f.source.modelSourceFiles(context.Background(), "llamacpp", reportedCopyAlias)
			if err == nil {
				t.Fatal("invalid preset accepted")
			}
			if strings.Contains(err.Error(), filepath.Dir(f.preset)) {
				t.Fatal("error disclosed host path")
			}
		})
	}
	refusePreset("missing shard", "["+reportedCopyAlias+"]\nmodel = absent-00001-of-00002.gguf\n")
	refusePreset("projector is explicitly unsupported", "["+reportedCopyAlias+"]\nmodel = huggingface/hub/model.gguf\nmmproj = secret.gguf\n")
	refusePreset("include is explicitly unsupported", "["+reportedCopyAlias+"]\nmodel = huggingface/hub/model.gguf\ninclude = secrets.ini\n")
	refusePreset("source URL is explicitly unsupported", "["+reportedCopyAlias+"]\nmodel-url = https://example.invalid/model.gguf\n")
	refusePreset("duplicate alias sections are ambiguous", "["+reportedCopyAlias+"]\nmodel = first.gguf\n["+reportedCopyAlias+"]\nmodel = second.gguf\n")
	refusePreset("oversize split rejected before allocation", "["+reportedCopyAlias+"]\nmodel = missing-00001-of-99999.gguf\n")
	t.Run("unlisted path and absent export identity are refused", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		_, client := modelFileServer(t, f.source, nil)
		list, err := client.modelFiles(context.Background(), "llamacpp", reportedCopyAlias, nil)
		if err != nil {
			t.Fatal(err)
		}
		for _, tc := range []struct {
			name, path, id string
			status         int
		}{{"unlisted", "model/secret.gguf", list.ExportID, http.StatusNotFound}, {"missing identity", list.Files[0].Path, "", http.StatusConflict}, {"wrong identity", list.Files[0].Path, strings.Repeat("a", 64), http.StatusConflict}} {
			t.Run(tc.name, func(t *testing.T) {
				q := url.Values{"engine": {"llamacpp"}, "model": {reportedCopyAlias}, "path": {tc.path}, "exportId": {tc.id}}
				w := httptest.NewRecorder()
				(&controlServer{exec: f.source}).handleModelFile(w, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+q.Encode(), nil))
				if w.Code != tc.status {
					t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
				}
			})
		}
	})
	t.Run("preset rebinding invalidates learned export", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		_, client := modelFileServer(t, f.source, nil)
		list, err := client.modelFiles(context.Background(), "llamacpp", reportedCopyAlias, nil)
		if err != nil {
			t.Fatal(err)
		}
		writeTestFile(t, filepath.Join(filepath.Dir(f.preset), "replacement.gguf"), []byte("different model"))
		writeTestFile(t, f.preset, []byte("["+reportedCopyAlias+"]\nmodel = replacement.gguf\n"))
		body, _, err := client.modelFile(context.Background(), "llamacpp", reportedCopyAlias, list.Files[0].Path, list.ExportID, 0)
		if body != nil {
			body.Close()
		}
		if err == nil {
			t.Fatal("rebound export accepted")
		}
	})
	t.Run("source file swap invalidates learned export", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		_, client := modelFileServer(t, f.source, nil)
		list, err := client.modelFiles(context.Background(), "llamacpp", reportedCopyAlias, nil)
		if err != nil {
			t.Fatal(err)
		}
		replacement := f.artifact + ".new"
		writeTestFile(t, replacement, []byte("changed source"))
		if err := os.Rename(replacement, f.artifact); err != nil {
			t.Fatal(err)
		}
		body, _, err := client.modelFile(context.Background(), "llamacpp", reportedCopyAlias, list.Files[0].Path, list.ExportID, 0)
		if body != nil {
			body.Close()
		}
		if err == nil {
			t.Fatal("swapped source accepted")
		}
	})
	t.Run("same bytes preset path and symlink rebind require relisting", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		server, client := modelFileServer(t, f.source, nil)
		list, err := client.modelFiles(context.Background(), "llamacpp", reportedCopyAlias, nil)
		if err != nil {
			t.Fatal(err)
		}
		copyPath := filepath.Join(filepath.Dir(f.preset), "other", "model.gguf")
		data, err := os.ReadFile(f.artifact)
		if err != nil {
			t.Fatal(err)
		}
		writeTestFile(t, copyPath, data)
		writeTestFile(t, f.preset, []byte("["+reportedCopyAlias+"]\nmodel = other/model.gguf\n"))
		body, _, err := client.modelFile(context.Background(), "llamacpp", reportedCopyAlias, list.Files[0].Path, list.ExportID, 0)
		if body != nil {
			body.Close()
		}
		if err == nil {
			t.Fatal("same-content path rebind accepted")
		}
		updated, err := client.modelFiles(context.Background(), "llamacpp", reportedCopyAlias, nil)
		if err != nil {
			t.Fatal(err)
		}
		if updated.ExportID != list.ExportID {
			t.Fatal("host path leaked into content identity")
		}
		body, _, err = client.modelFile(context.Background(), "llamacpp", reportedCopyAlias, updated.Files[0].Path, updated.ExportID, 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := body.Close(); err != nil {
			t.Fatal(err)
		}
		// Join the handler so its deferred file close finishes before removal.
		server.Close()
		if err := os.Remove(copyPath); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(f.artifact, copyPath); err != nil {
			t.Fatal(err)
		}
		_, client = modelFileServer(t, f.source, nil)
		body, _, err = client.modelFile(context.Background(), "llamacpp", reportedCopyAlias, updated.Files[0].Path, updated.ExportID, 0)
		if body != nil {
			body.Close()
		}
		if err == nil {
			t.Fatal("same-content symlink rebind accepted")
		}
	})
	t.Run("configured preset and committed receipt with different artifact sets are ambiguous", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		src := sourceFromExecutor(t, f.source, "llamacpp", reportedCopyAlias)
		dst, _ := customDestination(t)
		if _, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil); err != nil {
			t.Fatal(err)
		}
		st, err := dst.state("llamacpp")
		if err != nil {
			t.Fatal(err)
		}
		st.plat.Runtime.Args = []string{"--models-preset", f.preset}
		if _, err := dst.modelSourceFiles(context.Background(), "llamacpp", reportedCopyAlias); !errors.Is(err, errNoArtifactBinding) {
			t.Fatalf("ambiguous set err=%v", err)
		}
	})
	t.Run("source swap during hashing fails before a manifest is issued", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		list, err := f.source.modelSourceFiles(context.Background(), "llamacpp", reportedCopyAlias)
		if err != nil {
			t.Fatal(err)
		}
		previous := hashProgressInterval
		hashProgressInterval = 0
		t.Cleanup(func() { hashProgressInterval = previous })
		err = hashModelFiles(context.Background(), &list, func(string, int) {
			// Rewrite in place because Windows cannot replace an open file.
			// The changed size makes sameCopyFile reject it even if mtime is unchanged.
			writeTestFile(t, f.artifact, []byte("swapped during hash"))
		})
		if err == nil {
			t.Fatal("source swap while hashing accepted")
		}
		if list.ExportID != "" {
			t.Fatal("changed source received an export ID")
		}
		if strings.Contains(err.Error(), filepath.Dir(f.artifact)) {
			t.Fatal("hash error disclosed host path")
		}
	})
	t.Run("preset update during cold hashing refuses publication", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		list, err := f.source.modelSourceFiles(context.Background(), "llamacpp", reportedCopyAlias)
		if err != nil {
			t.Fatal(err)
		}
		previous := hashProgressInterval
		hashProgressInterval = 0
		t.Cleanup(func() { hashProgressInterval = previous })
		if err := hashModelFiles(context.Background(), &list, func(string, int) {
			writeTestFile(t, f.preset, []byte("# changed while hashing\n["+reportedCopyAlias+"]\nmodel = huggingface/hub/model.gguf\n"))
		}); err != nil {
			t.Fatal(err)
		}
		if err := f.source.recheckCopyExport(context.Background(), list); err == nil {
			t.Fatal("superseded preset was issuable")
		}
	})
	t.Run("conflicting committed alias preserves existing content", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		src := sourceFromExecutor(t, f.source, "llamacpp", reportedCopyAlias)
		dst, store := customDestination(t)
		if _, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil); err != nil {
			t.Fatal(err)
		}
		writeTestFile(t, f.artifact, []byte("rebound"))
		changed := sourceFromExecutor(t, f.source, "llamacpp", reportedCopyAlias)
		if _, err := dst.CopyModelFrom(context.Background(), changed, "llamacpp", reportedCopyAlias, nil); err == nil {
			t.Fatal("conflicting alias accepted")
		}
		if len(changed.fetchedPaths()) != 0 {
			t.Fatal("conflict downloaded bytes")
		}
		assertCustomReceipt(t, store, src.list)
	})
	t.Run("different completed artifact is never overwritten", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		src := sourceFromExecutor(t, f.source, "llamacpp", reportedCopyAlias)
		dst, store := customDestination(t)
		path := filepath.Join(copyAliasDir(store.root, reportedCopyAlias), src.list.ExportID, filepath.FromSlash(src.list.Files[0].Path))
		writeTestFile(t, path, []byte("existing different content"))
		if _, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil); err == nil {
			t.Fatal("existing different file accepted")
		}
		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != "existing different content" {
			t.Fatal("destination overwritten")
		}
	})
	t.Run("corrupt receipt blocks writes", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		src := sourceFromExecutor(t, f.source, "llamacpp", reportedCopyAlias)
		dst, store := customDestination(t)
		writeTestFile(t, filepath.Join(copyAliasDir(store.root, reportedCopyAlias), "copy.json"), []byte("{corrupt"))
		if _, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil); err == nil {
			t.Fatal("corrupt receipt accepted")
		}
		if len(src.fetchedPaths()) != 0 {
			t.Fatal("downloaded before receipt validation")
		}
	})
	t.Run("short disk refuses before artifact writes", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		src := sourceFromExecutor(t, f.source, "llamacpp", reportedCopyAlias)
		dst, _ := customDestination(t)
		previous := diskFreeBytes
		diskFreeBytes = func(string) (uint64, error) { return 0, nil }
		t.Cleanup(func() { diskFreeBytes = previous })
		if _, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil); err == nil {
			t.Fatal("short disk accepted")
		}
		if len(src.fetchedPaths()) != 0 {
			t.Fatal("downloaded with short disk")
		}
	})
	t.Run("hash mismatch cannot publish receipt", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		src := sourceFromExecutor(t, f.source, "llamacpp", reportedCopyAlias)
		dst, store := customDestination(t)
		src.data[src.list.Files[0].Path] = bytes.Repeat([]byte{'x'}, int(src.list.Files[0].Size))
		if _, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil); err == nil {
			t.Fatal("hash mismatch accepted")
		}
		if _, err := readCopyReceipt(store, reportedCopyAlias); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("receipt=%v", err)
		}
	})
	t.Run("identity withdrawal refusal has no filesystem effects", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		src := sourceFromExecutor(t, f.source, "llamacpp", reportedCopyAlias)
		dst, store := customDestination(t)
		dst.modelMutationParent = func(context.Context, string) error { return errors.New("withdrawal refused") }
		if _, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil); err == nil {
			t.Fatal("refused withdrawal accepted")
		}
		if len(src.fetchedPaths()) != 0 {
			t.Fatal("download before withdrawal")
		}
		if _, err := os.Stat(copyAliasDir(store.root, reportedCopyAlias)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("filesystem effect=%v", err)
		}
	})
	t.Run("destination symlink and case collisions are refused", func(t *testing.T) {
		for _, kind := range []string{"alias directory symlink", "partial symlink", "case collision"} {
			t.Run(kind, func(t *testing.T) {
				f := newCustomCopyFixture(t)
				src := sourceFromExecutor(t, f.source, "llamacpp", reportedCopyAlias)
				dst, store := customDestination(t)
				aliasDir := copyAliasDir(store.root, reportedCopyAlias)
				switch kind {
				case "alias directory symlink":
					if err := os.MkdirAll(filepath.Dir(aliasDir), 0o755); err != nil {
						t.Fatal(err)
					}
					if err := os.Symlink(t.TempDir(), aliasDir); err != nil {
						t.Fatal(err)
					}
				case "partial symlink":
					path := filepath.Join(aliasDir, src.list.ExportID, filepath.FromSlash(src.list.Files[0].Path)) + ".part"
					if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
						t.Fatal(err)
					}
					outside := filepath.Join(t.TempDir(), "unchanged")
					writeTestFile(t, outside, []byte("unchanged"))
					if err := os.Symlink(outside, path); err != nil {
						t.Fatal(err)
					}
				case "case collision":
					if err := os.MkdirAll(filepath.Join(store.root, ".PAIR-COPIES"), 0o755); err != nil {
						t.Fatal(err)
					}
				}
				if _, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil); err == nil {
					t.Fatal("unsafe destination accepted")
				}
			})
		}
	})
	t.Run("Windows path and manifest case collisions are refused", func(t *testing.T) {
		for _, path := range []string{"C:/model.gguf", "model/a.gguf.", "model/NUL.gguf", "model/a\x01.gguf", "model/../evil.gguf"} {
			t.Run(path, func(t *testing.T) {
				f := newCustomCopyFixture(t)
				list := customCopyList(t, f)
				list.Files[0].Path = path
				_, err := artifactExportID(list)
				if err == nil {
					t.Fatal("unsafe Windows path accepted")
				}
			})
		}
		f := newCustomCopyFixture(t)
		list := customCopyList(t, f)
		other := list.Files[0]
		other.Path = "model/MODEL.gguf"
		list.Files = append(list.Files, other)
		if _, err := artifactExportID(list); err == nil {
			t.Fatal("case collision accepted")
		}
		list.Files[0].Path = "model/Σ.gguf"
		list.Files[1].Path = "model/ς.gguf"
		if _, err := artifactExportID(list); err == nil {
			t.Fatal("Unicode case-fold collision accepted")
		}
	})
}

type copyRetrySwapHandler struct {
	slog.Handler
	onRetry func()
}

func (h copyRetrySwapHandler) Handle(ctx context.Context, record slog.Record) error {
	if record.Message == "model copy download interrupted; resuming" {
		h.onRetry()
	}
	return h.Handler.Handle(ctx, record)
}

func TestModelCopyCustomAliasPartialIdentity(t *testing.T) {
	plentyOfDisk(t)
	t.Run("escaping partial symlink between attempts cannot truncate outside file", func(t *testing.T) {
		fastRetries(t, 2)
		f := newCustomCopyFixture(t)
		list := customCopyList(t, f)
		dst, store := customDestination(t)
		part := filepath.Join(copyAliasDir(store.root, reportedCopyAlias), list.ExportID, filepath.FromSlash(list.Files[0].Path)) + ".part"
		outside := filepath.Join(t.TempDir(), "outside.gguf")
		sentinel := testBytes(int(list.Files[0].Size)+17, 8)
		writeTestFile(t, outside, sentinel)
		previous := slog.Default()
		previousOutput, previousFlags := log.Writer(), log.Flags()
		swapped := false
		slog.SetDefault(slog.New(copyRetrySwapHandler{Handler: slog.NewTextHandler(io.Discard, nil), onRetry: func() {
			if swapped {
				return
			}
			swapped = true
			if err := os.Remove(part); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(outside, part); err != nil {
				t.Fatal(err)
			}
		}}))
		t.Cleanup(func() {
			slog.SetDefault(previous)
			log.SetOutput(previousOutput)
			log.SetFlags(previousFlags)
		})
		_, client := modelFileServer(t, f.source, interruptAfter(list.Files[0].Path, 50_000))
		if _, err := dst.CopyModelFrom(context.Background(), client, "llamacpp", reportedCopyAlias, nil); err == nil {
			t.Fatal("retry followed swapped partial")
		}
		if !swapped {
			t.Fatal("fixture never reached retry backoff")
		}
		got, err := os.ReadFile(outside)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, sentinel) {
			t.Fatal("retry modified external file")
		}
		if _, err := readCopyReceipt(store, reportedCopyAlias); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("swapped partial published receipt: %v", err)
		}
	})
	for _, kind := range []string{"different regular file", "same bytes different inode", "escaping symlink"} {
		t.Run("swap before publication: "+kind, func(t *testing.T) {
			f := newCustomCopyFixture(t)
			src := sourceFromExecutor(t, f.source, "llamacpp", reportedCopyAlias)
			dst, store := customDestination(t)
			part := filepath.Join(copyAliasDir(store.root, reportedCopyAlias), src.list.ExportID, filepath.FromSlash(src.list.Files[0].Path)) + ".part"
			outside := filepath.Join(t.TempDir(), "outside.gguf")
			sentinel := []byte("outside content must remain unchanged")
			writeTestFile(t, outside, sentinel)
			src.onFetch = func(path string) {
				if err := os.Remove(part); err != nil {
					t.Fatal(err)
				}
				switch kind {
				case "escaping symlink":
					if err := os.Symlink(outside, part); err != nil {
						t.Fatal(err)
					}
				case "same bytes different inode":
					writeTestFile(t, part, src.data[path])
				default:
					writeTestFile(t, part, []byte("replacement partial"))
				}
			}
			if _, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil); err == nil {
				t.Fatal("renamed a partial whose bytes were not verified")
			}
			if _, err := readCopyReceipt(store, reportedCopyAlias); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("swapped partial published receipt: %v", err)
			}
			got, err := os.ReadFile(outside)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, sentinel) {
				t.Fatal("publication modified external file")
			}
		})
	}
}

func TestModelCopyCustomAliasReceiptDurability(t *testing.T) {
	plentyOfDisk(t)
	for _, outcome := range []string{"failure before replacement", "landed with uncertain durability"} {
		t.Run(outcome, func(t *testing.T) {
			f := newCustomCopyFixture(t)
			src := sourceFromExecutor(t, f.source, "llamacpp", reportedCopyAlias)
			dst, store := customDestination(t)
			failure := errors.New("injected publication failure")
			if outcome == "failure before replacement" {
				useEngineFileReplacer(t, durablefile.Replacer{Replace: func(string, string) error { return failure }})
			} else {
				useEngineFileReplacer(t, durablefile.Replacer{SyncDir: func(string) error { return failure }})
			}
			res, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil)
			if outcome == "failure before replacement" {
				if err == nil {
					t.Fatal("failed publication succeeded")
				}
				if _, err := readCopyReceipt(store, reportedCopyAlias); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("receipt=%v", err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				if res.Durability != copyDurabilityUncertain || !res.RequiresConfiguration {
					t.Fatalf("qualification lost=%+v", res)
				}
				assertCustomReceipt(t, store, src.list)
			}
		})
	}
}

func TestModelCopyCustomAliasReverseExportIdentity(t *testing.T) {
	plentyOfDisk(t)
	for _, direct := range []bool{true, false} {
		name := "reverse"
		if direct {
			name = "direct"
		}
		t.Run(name, func(t *testing.T) {
			f := newCustomCopyFixture(t)
			list := customCopyList(t, f)
			dst, store := customDestination(t)
			var mu sync.Mutex
			var ids []string
			peer := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
				s := &controlServer{exec: f.source, mesh: mesh}
				mux := http.NewServeMux()
				mux.HandleFunc(controlModelFilesPath, s.handleModelFiles)
				mux.HandleFunc(controlModelFilePath, func(w http.ResponseWriter, r *http.Request) {
					mu.Lock()
					ids = append(ids, r.URL.Query().Get("exportId"))
					mu.Unlock()
					s.handleModelFile(w, r)
				})
				return s.requirePin(mux.ServeHTTP)
			})
			peer.directHealthy.Store(direct)
			if _, err := dst.CopyModelFrom(context.Background(), peer.client(t, context.Background()), "llamacpp", reportedCopyAlias, nil); err != nil {
				t.Fatal(err)
			}
			mu.Lock()
			defer mu.Unlock()
			if len(ids) != 1 || ids[0] != list.ExportID {
				t.Fatalf("GET export IDs=%v", ids)
			}
			assertCustomReceipt(t, store, list)
		})
	}
}

func TestModelCopyCustomAliasCanonicalManifest(t *testing.T) {
	list := modelFileList{Model: "alias", Files: []modelFile{{Role: "model", Path: "model/a.gguf", Size: 1, SHA256: strings.Repeat("a", 64)}}}
	canonical := `{"version":1,"model":"alias","artifacts":[{"role":"model","path":"model/a.gguf","size":1,"sha256":"` + strings.Repeat("a", 64) + `"}]}`
	got, err := artifactExportID(list)
	if err != nil {
		t.Fatal(err)
	}
	if got != sha256Hex([]byte(canonical)) {
		t.Fatalf("canonical manifest ID=%s", got)
	}
}

func TestModelPublisherObservationDoesNotChangeExportIdentity(t *testing.T) {
	src := publisherCopySource(t)
	list := cloneCopyList(src.list)
	for i := range list.Files {
		list.Files[i].Publisher = nil
	}
	before, err := nativeExportID(list)
	if err != nil {
		t.Fatal(err)
	}
	list.Files[0].Publisher = publisherSelector(list, list.Files[0])
	after, err := nativeExportID(list)
	if err != nil || before != after {
		t.Fatalf("observation changed native identity: before=%s after=%s error=%v", before, after, err)
	}
	custom := customCopyList(t, newCustomCopyFixture(t))
	before, err = artifactExportID(custom)
	if err != nil {
		t.Fatal(err)
	}
	custom.Files[0].Publisher = &modelPublisherFile{Path: custom.Files[0].Path, Status: "unavailable", Algorithm: "sha256", ActualDigest: custom.Files[0].SHA256, AcquisitionBasis: "unavailable"}
	after, err = artifactExportID(custom)
	if err != nil || before != after {
		t.Fatalf("observation changed custom identity: before=%s after=%s error=%v", before, after, err)
	}
}

func TestModelPublisherReceiptReplayPreservesHistoricalBasis(t *testing.T) {
	plentyOfDisk(t)
	src := sourceFromExecutor(t, newCustomCopyFixture(t).source, "llamacpp", reportedCopyAlias)
	dst, store := customDestination(t)
	res, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := readCopyReceipt(store, reportedCopyAlias)
	if err != nil {
		t.Fatal(err)
	}
	// A locally recorded statement fixture, not a peer assertion. Replay must
	// retain its original basis/time after the committed bytes are rehashed.
	for i := range receipt.List.Files {
		f := &receipt.List.Files[i]
		record := publisherTestRecord(f.SHA256)
		record.Path, record.AcquisitionBasis = f.Path, "cache"
		f.Publisher = &record
	}
	encoded, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, filepath.Join(copyAliasDir(store.root, reportedCopyAlias), "copy.json"), encoded)
	replayed, complete, err := dst.completedCopy(context.Background(), "llamacpp", reportedCopyAlias, res.ExportID, nil)
	if err != nil || !complete || replayed.PublisherComparison == nil || !replayed.PublisherComparison.Replayed || replayed.PublisherComparison.Status != "verified" {
		t.Fatalf("offline replay=%+v complete=%t error=%v", replayed, complete, err)
	}
	for _, f := range replayed.PublisherComparison.Files {
		if f.AcquisitionBasis != "cache" || f.AcquiredAt != "2026-01-01T00:00:00Z" {
			t.Fatalf("replay invented online freshness: %+v", f)
		}
	}
	file := receipt.List.Files[0]
	path := filepath.Join(copyAliasDir(store.root, reportedCopyAlias), res.ExportID, filepath.FromSlash(file.Path))
	changed, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	changed[len(changed)-1] ^= 1
	writeTestFile(t, path, changed)
	if _, complete, err := dst.completedCopy(context.Background(), "llamacpp", reportedCopyAlias, res.ExportID, nil); err == nil || complete {
		t.Fatalf("historical observation bypassed changed bytes: complete=%t error=%v", complete, err)
	}
}

func TestModelCopyMetadataBounds(t *testing.T) {
	t.Run("coalesced progress stream has no cumulative byte cap", func(t *testing.T) {
		list := modelFileList{Engine: "llamacpp", Model: "alias", Files: []modelFile{{Path: "a.gguf", Size: 1, SHA256: strings.Repeat("a", 64)}}}
		result, err := json.Marshal(list)
		if err != nil {
			t.Fatal(err)
		}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodGet || r.URL.Path != controlModelFilesPath {
				http.NotFound(w, r)
				return
			}
			enc := json.NewEncoder(w)
			for range 100 {
				if err := enc.Encode(streamFrame{Type: "progress", Message: strings.Repeat("x", 20_000)}); err != nil {
					return
				}
			}
			if err := enc.Encode(streamFrame{Type: "result", Result: result}); err != nil {
				t.Error(err)
			}
		}))
		defer server.Close()
		client := &remoteClient{base: server.URL, controlHTTP: server.Client()}
		frames := 0
		got, err := client.modelFiles(context.Background(), "llamacpp", "alias", func(streamFrame) { frames++ })
		if err != nil {
			t.Fatal(err)
		}
		if frames != 100 || got.Model != list.Model {
			t.Fatalf("frames=%d list=%+v", frames, got)
		}
	})
	t.Run("progress frame bound", func(t *testing.T) {
		line := `{"type":"progress","message":"` + strings.Repeat("x", maxCopyProgressFrame) + `"}` + "\n"
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodGet || r.URL.Path != controlModelFilesPath {
				http.NotFound(w, r)
				return
			}
			_, _ = io.WriteString(w, line)
		}))
		defer server.Close()
		client := &remoteClient{base: server.URL, controlHTTP: server.Client()}
		if _, err := client.modelFiles(context.Background(), "llamacpp", "alias", nil); err == nil {
			t.Fatal("oversize progress accepted")
		}
	})
	t.Run("frame allocation bound", func(t *testing.T) {
		reader := bufio.NewReader(strings.NewReader(strings.Repeat("x", maxCopyManifest+4097)))
		if _, err := readCopyFrame(reader); err == nil {
			t.Fatal("oversize frame accepted")
		}
	})
	t.Run("artifact bound", func(t *testing.T) {
		list := modelFileList{Model: "alias", Files: make([]modelFile, maxCopyArtifacts+1)}
		if _, err := artifactExportID(list); err == nil {
			t.Fatal("oversize artifact set accepted")
		}
	})
	t.Run("oversize array refused by bounded decoder", func(t *testing.T) {
		data := []byte(`{"engine":"llamacpp","model":"alias","files":[` + strings.TrimSuffix(strings.Repeat(`{"path":"model/a.gguf","size":1,"sha256":"`+strings.Repeat("a", 64)+`"},`, maxCopyArtifacts+1), ",") + `]}`)
		if _, err := decodeCopyManifest(data); err == nil {
			t.Fatal("oversize array decoded")
		}
	})
	t.Run("nonnegative total cannot overflow", func(t *testing.T) {
		list := modelFileList{Model: "alias", Files: []modelFile{{Role: "model", Path: "model/a.gguf", Size: 1<<63 - 1, SHA256: strings.Repeat("a", 64)}, {Role: "model", Path: "model/b.gguf", Size: 1, SHA256: strings.Repeat("b", 64)}}}
		if _, err := artifactExportID(list); err == nil {
			t.Fatal("overflowing total accepted")
		}
	})
}

func TestModelCopyHashOrdinaryErrorIsNotRetried(t *testing.T) {
	path := filepath.Join(t.TempDir(), "artifact")
	writeTestFile(t, path, []byte("fixture"))
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	failure := errors.New("ordinary read failure")
	call := &hashCall{done: make(chan struct{}), err: failure}
	close(call.done)
	key := hashEntry{path: path, size: info.Size(), mtime: info.ModTime().UnixNano()}
	cache := &hashCache{done: map[string]hashEntry{}, inflight: map[hashEntry]*hashCall{key: call}}
	if _, err := cache.sum(context.Background(), path, nil); !errors.Is(err, failure) {
		t.Fatalf("ordinary failure changed or was retried: %v", err)
	}
}

func TestModelCopyHashIndependentCancellation(t *testing.T) {
	for _, scenario := range []string{"cancel waiter", "cancel producer", "live concurrent waiters"} {
		t.Run(scenario, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "artifact")
			data := testBytes(2<<20, 3)
			writeTestFile(t, path, data)
			cache := &hashCache{done: map[string]hashEntry{}, inflight: map[hashEntry]*hashCall{}}
			producerCtx, cancelProducer := context.WithCancel(context.Background())
			defer cancelProducer()
			started, release := make(chan struct{}), make(chan struct{})
			releaseOnce := sync.OnceFunc(func() { close(release) })
			defer releaseOnce()
			producerDone := make(chan error, 1)
			go func() {
				_, err := cache.sum(producerCtx, path, func(n int64) {
					if n == 1<<20 {
						close(started)
						<-release
					}
				})
				producerDone <- err
			}()
			select {
			case <-started:
			case <-time.After(5 * time.Second):
				t.Fatal("producer did not start")
			}
			waiters := 1
			if scenario == "live concurrent waiters" {
				waiters = 2
			}
			waiting := make(chan struct{}, waiters)
			cache.mu.Lock()
			for _, call := range cache.inflight {
				call.onWait = func() { waiting <- struct{}{} }
			}
			cache.mu.Unlock()
			waiterCtx, cancelWaiter := context.WithCancel(context.Background())
			defer cancelWaiter()
			type result struct {
				sum string
				err error
			}
			waiterDone := make(chan result, waiters)
			for range waiters {
				go func() { sum, err := cache.sum(waiterCtx, path, nil); waiterDone <- result{sum, err} }()
			}
			for range waiters {
				select {
				case <-waiting:
				case <-time.After(5 * time.Second):
					t.Fatal("waiter did not join")
				}
			}
			if scenario == "cancel waiter" {
				cancelWaiter()
				select {
				case result := <-waiterDone:
					if !errors.Is(result.err, context.Canceled) {
						t.Fatalf("waiter=%v", result.err)
					}
				case <-time.After(5 * time.Second):
					t.Fatal("waiter cancellation blocked")
				}
				releaseOnce()
				if err := <-producerDone; err != nil {
					t.Fatal(err)
				}
			} else {
				if scenario == "cancel producer" {
					cancelProducer()
				}
				releaseOnce()
				producerErr := <-producerDone
				if scenario == "cancel producer" && !errors.Is(producerErr, context.Canceled) {
					t.Fatalf("producer=%v", producerErr)
				}
				for range waiters {
					result := <-waiterDone
					if result.err != nil || result.sum != sha256Hex(data) {
						t.Fatalf("live waiter=%+v", result)
					}
				}
			}
			if sum, err := cache.sum(context.Background(), path, func(int64) { t.Error("completed hash was not reused") }); err != nil || sum != sha256Hex(data) {
				t.Fatalf("cached sum=%s err=%v", sum, err)
			}
		})
	}
}
