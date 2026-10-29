// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/windows"
)

// These fixtures exercise the real list/file handlers without a listener. Native
// membership is synthetic; neither vendor discovery nor inference is exercised.
func TestLinkedModelJunctions(t *testing.T) {
	t.Run("HF trusted cache alias", func(t *testing.T) { testWindowsLinkedHF(t, "root") })
	t.Run("HF external whole repository", func(t *testing.T) { testWindowsLinkedHF(t, "repository") })
	t.Run("HF refs ancestor", func(t *testing.T) { testWindowsLinkedHF(t, "refs") })
	t.Run("HF snapshot ancestor", func(t *testing.T) { testWindowsLinkedHF(t, "snapshot") })
	t.Run("HF nested selected directory", func(t *testing.T) { testWindowsLinkedHF(t, "nested") })
	t.Run("Ollama manifest and blob ancestors", testWindowsLinkedOllama)
	t.Run("LM Studio GGUF and projector file links", testWindowsLinkedGGUF)
	t.Run("LM Studio MLX nested junctions", testWindowsLinkedMLX)
	t.Run("ordinary file hardlink", testWindowsLinkedHardlink)
	t.Run("case and extended DOS identity", testWindowsLinkedAliases)
	t.Run("UNC identity requires a released share fixture", func(t *testing.T) {
		// Do not guess an administrative share or contact an ambient UNC server.
		t.Skip("unverified: no isolated UNC share fixture is configured")
	})
	t.Run("retarget after issued listing", testWindowsLinkedRetarget)
	t.Run("retarget before opened handle read", testWindowsLinkedOpenRetarget)
	t.Run("retarget during sampling", testWindowsLinkedSampleRetarget)
	t.Run("retarget before final list publication", testWindowsLinkedPublicationRetarget)
	t.Run("opened unknown reparse and nonregular types", testWindowsLinkedOpenedTypes)
	t.Run("directory aliases cycles and wrong types", testWindowsLinkedInvalidTrees)
}

func windowsLinkedMove(t *testing.T, path string) string {
	t.Helper()
	target := filepath.Join(t.TempDir(), "selected")
	if err := os.Rename(path, target); err != nil {
		t.Fatalf("move owned fixture: %v", err)
	}
	junction(t, path, target)
	// Remove only our link, before either TempDir target is cleaned up.
	t.Cleanup(func() {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			t.Errorf("remove owned junction: %v", err)
		}
	})
	return target
}

func windowsLinkedFileLink(t *testing.T, link, target string) {
	t.Helper()
	if err := os.Symlink(target, link); err != nil {
		if errors.Is(err, windows.ERROR_PRIVILEGE_NOT_HELD) || errors.Is(err, windows.ERROR_NOT_SUPPORTED) {
			t.Skipf("unverified: file symlink capability unavailable: %v", err)
		}
		t.Fatalf("create owned file symlink: %v", err)
	}
}

func windowsLinkedList(t *testing.T, server *controlServer, engine, model string) modelFileList {
	t.Helper()
	q := url.Values{"engine": {engine}, "model": {model}}
	w := httptest.NewRecorder()
	server.handleModelFiles(w, httptest.NewRequest(http.MethodGet, controlModelFilesPath+"?"+q.Encode(), nil))
	if w.Code != http.StatusOK {
		t.Fatalf("LIST status=%d body=%q", w.Code, w.Body.String())
	}
	decoder := json.NewDecoder(w.Body)
	var list modelFileList
	results := 0
	for {
		var frame streamFrame
		if err := decoder.Decode(&frame); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatal(err)
		}
		if frame.Type == "error" {
			t.Fatalf("LIST failed: %s", frame.Message)
		}
		if frame.Type == "result" {
			if err := json.Unmarshal(frame.Result, &list); err != nil {
				t.Fatal(err)
			}
			results++
		}
	}
	if results != 1 || list.ExportID == "" || len(list.Files) == 0 {
		t.Fatalf("missing issued set: results=%d list=%+v", results, list)
	}
	return list
}

func windowsLinkedGet(server *controlServer, list modelFileList, path string) *httptest.ResponseRecorder {
	q := url.Values{"engine": {list.Engine}, "model": {list.Model}, "path": {path}, "exportId": {list.ExportID}}
	w := httptest.NewRecorder()
	server.handleModelFile(w, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+q.Encode(), nil))
	return w
}

func windowsLinkedExact(t *testing.T, server *controlServer, engine, model string, want map[string][]byte) modelFileList {
	t.Helper()
	list := windowsLinkedList(t, server, engine, model)
	if list.Layout != "" || len(list.Files) != len(want) {
		t.Fatalf("native layout=%q files=%v", list.Layout, list.Files)
	}
	for _, file := range list.Files {
		data, ok := want[file.Path]
		if !ok || file.Size != int64(len(data)) || file.SHA256 != sha256Hex(data) {
			t.Fatalf("unexpected artifact: %+v", file)
		}
		w := windowsLinkedGet(server, list, file.Path)
		if w.Code != http.StatusOK || !bytes.Equal(w.Body.Bytes(), data) {
			t.Fatalf("GET %s: status=%d body=%q", file.Path, w.Code, w.Body.String())
		}
	}
	return list
}

func windowsLinkedDeny(t *testing.T, ex *Executor, list modelFileList, path string) {
	t.Helper()
	counts := &linkedHashCounts{}
	server := &controlServer{exec: ex, access: servingFileAccess{open: func(path string) (servingArtifactFile, error) {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		counts.opens.Add(1)
		return &linkedHashFile{File: f, counts: counts, forbidden: true}, nil
	}}}
	w := windowsLinkedGet(server, list, path)
	if w.Code < 400 || w.Body.String() == "forbidden sentinel" || counts.readAt.Load() != 0 || counts.streams.Load() != 0 || counts.bytes.Load() != 0 || counts.opens.Load() != counts.closes.Load() {
		t.Fatalf("refusal status=%d reads=%d streams=%d bytes=%d opens=%d closes=%d", w.Code, counts.readAt.Load(), counts.streams.Load(), counts.bytes.Load(), counts.opens.Load(), counts.closes.Load())
	}
}

func testWindowsLinkedHF(t *testing.T, kind string) {
	root := t.TempDir()
	const repo = "owner/model"
	commit := strings.Repeat("a", 40)
	name := "nested/model-Q4_K_M.gguf"
	data := []byte("HF selected")
	writeLlamaCache(t, root, repo, commit, map[string][]byte{name: data})
	repoDir := llamaRepoDir(root, repo)
	snap := filepath.Join(repoDir, "snapshots", commit)
	path := map[string]string{"root": root, "repository": repoDir, "refs": filepath.Join(repoDir, "refs"), "snapshot": snap, "nested": filepath.Join(snap, "nested")}[kind]
	target := windowsLinkedMove(t, path)
	sentinel := filepath.Join(filepath.Dir(target), "unrelated.gguf")
	writeTestFile(t, sentinel, []byte("forbidden sentinel"))
	ex := copyTestExecutor(t, "llamacpp", root, true)
	copyTestNativeModels(t, ex, "llamacpp", repo+":Q4_K_M")
	list := windowsLinkedExact(t, &controlServer{exec: ex}, "llamacpp", repo+":Q4_K_M", map[string][]byte{name: data})
	windowsLinkedDeny(t, ex, list, "unrelated.gguf")
	got, err := os.ReadFile(sentinel)
	if err != nil || string(got) != "forbidden sentinel" {
		t.Fatalf("sentinel changed: %v", err)
	}
}

func testWindowsLinkedOllama(t *testing.T) {
	root := t.TempDir()
	fx := writeOllamaModel(t, filepath.Join(root, "models"), "tiny", 128, 3)
	windowsLinkedMove(t, filepath.Join(root, "models", "manifests"))
	windowsLinkedMove(t, filepath.Join(root, "models", "blobs"))
	ex := copyTestExecutor(t, "ollama", root, true)
	copyTestNativeModels(t, ex, "ollama", "tiny")
	want := map[string][]byte{fx.manifestRel: fx.manifest}
	for path, data := range fx.blobs {
		want[path] = data
	}
	list := windowsLinkedExact(t, &controlServer{exec: ex}, "ollama", "tiny", want)
	windowsLinkedDeny(t, ex, list, "blobs/unlisted")
}

func windowsLinkedLMStudio(t *testing.T, root, model, path string) *Executor {
	t.Helper()
	previous := lmsListModels
	lmsListModels = func(context.Context, *Executor, string) ([]lmsListEntry, error) {
		return []lmsListEntry{{ModelKey: model, Path: path}}, nil
	}
	t.Cleanup(func() { lmsListModels = previous })
	return copyTestExecutor(t, "lmstudio", root, true)
}

func testWindowsLinkedGGUF(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "owner", "model")
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	outside := t.TempDir()
	writeTestFile(t, filepath.Join(outside, "weight"), []byte("weights"))
	writeTestFile(t, filepath.Join(outside, "projector"), []byte("projector"))
	sentinel := filepath.Join(outside, "mmproj-unrelated.gguf")
	writeTestFile(t, sentinel, []byte("forbidden sentinel"))
	windowsLinkedFileLink(t, filepath.Join(dir, "model.gguf"), filepath.Join(outside, "weight"))
	windowsLinkedFileLink(t, filepath.Join(dir, "mmproj.gguf"), filepath.Join(outside, "projector"))
	ex := windowsLinkedLMStudio(t, root, "tiny", "owner/model/model.gguf")
	counts := &linkedHashCounts{}
	forbiddenDirs := 0
	server := &controlServer{exec: ex, access: servingFileAccess{
		open: func(path string) (servingArtifactFile, error) {
			f, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			counts.opens.Add(1)
			return &linkedHashFile{File: f, counts: counts, forbidden: path == sentinel}, nil
		},
		openDir: func(path string) (servingDirectory, error) {
			f, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			if strings.EqualFold(filepath.Clean(path), outside) {
				return &hfDirectoryCounter{File: f, reads: &forbiddenDirs}, nil
			}
			return f, nil
		},
	}}
	list := windowsLinkedExact(t, server, "lmstudio", "tiny", map[string][]byte{"owner/model/model.gguf": []byte("weights"), "owner/model/mmproj.gguf": []byte("projector")})
	if counts.forbiddenBytes.Load() != 0 || forbiddenDirs != 0 || counts.opens.Load() != counts.closes.Load() {
		t.Fatalf("external sibling access: bytes=%d directories=%d opens=%d closes=%d", counts.forbiddenBytes.Load(), forbiddenDirs, counts.opens.Load(), counts.closes.Load())
	}
	windowsLinkedDeny(t, ex, list, "owner/model/unlisted.gguf")
	got, err := os.ReadFile(sentinel)
	if err != nil || string(got) != "forbidden sentinel" {
		t.Fatalf("external sentinel changed: %v", err)
	}
}

func testWindowsLinkedMLX(t *testing.T) {
	root := t.TempDir()
	mlx := filepath.Join(root, "owner", "mlx")
	writeTestFile(t, filepath.Join(mlx, "weights.safetensors"), []byte("weights"))
	writeTestFile(t, filepath.Join(mlx, "tokenizer", "vocab.json"), nil)
	windowsLinkedMove(t, filepath.Join(mlx, "tokenizer"))
	windowsLinkedMove(t, mlx)
	ex := windowsLinkedLMStudio(t, root, "mlx", "owner/mlx")
	list := windowsLinkedExact(t, &controlServer{exec: ex}, "lmstudio", "mlx", map[string][]byte{"owner/mlx/weights.safetensors": []byte("weights"), "owner/mlx/tokenizer/vocab.json": nil})
	windowsLinkedDeny(t, ex, list, "owner/unlisted/config.json")
}

func testWindowsLinkedHardlink(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "owner", "model.gguf")
	writeTestFile(t, path, []byte("shared bytes"))
	alias := filepath.Join(filepath.Dir(path), "other-name")
	if err := os.Link(path, alias); err != nil {
		if errors.Is(err, windows.ERROR_NOT_SUPPORTED) || errors.Is(err, windows.ERROR_INVALID_FUNCTION) {
			t.Skipf("unverified: hardlinks unsupported: %v", err)
		}
		t.Fatal(err)
	}
	ex := windowsLinkedLMStudio(t, root, "tiny", "owner/model.gguf")
	list := windowsLinkedExact(t, &controlServer{exec: ex}, "lmstudio", "tiny", map[string][]byte{"owner/model.gguf": []byte("shared bytes")})
	windowsLinkedDeny(t, ex, list, "owner/other-name")
	before, err := os.Stat(alias)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(alias, []byte("changed data"), 0600); err != nil {
		t.Fatal(err)
	}
	// Ensure this is observable metadata drift, not the documented same-stat
	// mutation limit. Both hardlink names share this timestamp.
	stamp := before.ModTime().Add(2 * time.Second)
	if err := os.Chtimes(alias, stamp, stamp); err != nil {
		t.Fatal(err)
	}
	windowsLinkedDeny(t, ex, list, "owner/model.gguf")
	got, err := os.ReadFile(alias)
	if err != nil || string(got) != "changed data" {
		t.Fatalf("hardlink survivor: %v", err)
	}
}

func testWindowsLinkedAliases(t *testing.T) {
	path := filepath.Join(t.TempDir(), "Model.gguf")
	writeTestFile(t, path, []byte("identity"))
	original, err := captureSource(path, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, alias := range []string{strings.ToUpper(path), `\\?\` + path} {
		got, err := captureSource(alias, false)
		if err != nil || !os.SameFile(original.info, got.info) {
			t.Fatalf("alias identity %q: %v", alias, err)
		}
	}
}

func windowsLinkedRetargetFixture(t *testing.T) (*Executor, modelFileList, string, string) {
	t.Helper()
	root := t.TempDir()
	selected := filepath.Join(root, "owner", "mlx")
	writeTestFile(t, filepath.Join(selected, "weights.safetensors"), []byte("original"))
	windowsLinkedMove(t, selected)
	replacement := t.TempDir()
	writeTestFile(t, filepath.Join(replacement, "weights.safetensors"), []byte("forbidden sentinel"))
	ex := windowsLinkedLMStudio(t, root, "mlx", "owner/mlx")
	list := windowsLinkedExact(t, &controlServer{exec: ex}, "lmstudio", "mlx", map[string][]byte{"owner/mlx/weights.safetensors": []byte("original")})
	return ex, list, selected, replacement
}

func windowsLinkedRetarget(t *testing.T, link, replacement string) {
	t.Helper()
	if err := os.Remove(link); err != nil {
		t.Fatal(err)
	}
	junction(t, link, replacement)
}

func testWindowsLinkedRetarget(t *testing.T) {
	ex, list, link, replacement := windowsLinkedRetargetFixture(t)
	windowsLinkedRetarget(t, link, replacement)
	windowsLinkedDeny(t, ex, list, list.Files[0].Path)
	windowsLinkedExact(t, &controlServer{exec: ex}, "lmstudio", "mlx", map[string][]byte{list.Files[0].Path: []byte("forbidden sentinel")})
}

func testWindowsLinkedOpenRetarget(t *testing.T) {
	ex, list, _, replacement := windowsLinkedRetargetFixture(t)
	counts := &linkedHashCounts{}
	server := &controlServer{exec: ex, access: servingFileAccess{open: func(string) (servingArtifactFile, error) {
		f, err := os.Open(filepath.Join(replacement, "weights.safetensors"))
		if err != nil {
			return nil, err
		}
		counts.opens.Add(1)
		return &linkedHashFile{File: f, counts: counts, forbidden: true}, nil
	}}}
	w := windowsLinkedGet(server, list, list.Files[0].Path)
	if w.Code != http.StatusConflict || counts.bytes.Load() != 0 || counts.readAt.Load() != 0 || counts.opens.Load() == 0 || counts.opens.Load() != counts.closes.Load() {
		t.Fatalf("opened substitute: status=%d counts=%+v", w.Code, counts)
	}
}

func testWindowsLinkedSampleRetarget(t *testing.T) {
	ex, list, link, replacement := windowsLinkedRetargetFixture(t)
	counts := &linkedHashCounts{}
	changed := false
	server := &controlServer{exec: ex, access: servingFileAccess{open: func(path string) (servingArtifactFile, error) {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		counts.opens.Add(1)
		resolved, err := resolveExistingPath(path)
		if err != nil {
			if closeErr := f.Close(); closeErr != nil {
				t.Error(closeErr)
			}
			return nil, err
		}
		return &linkedHashFile{File: f, counts: counts, forbidden: pathWithinRoot(replacement, resolved), afterRead: func(int, int64) {
			if !changed {
				changed = true
				windowsLinkedRetarget(t, link, replacement)
			}
		}}, nil
	}}}
	w := windowsLinkedGet(server, list, list.Files[0].Path)
	if !changed || w.Code != http.StatusConflict || counts.forbiddenBytes.Load() != 0 || bytes.Contains(w.Body.Bytes(), []byte("forbidden sentinel")) || counts.opens.Load() != counts.closes.Load() {
		t.Fatalf("sample drift: changed=%v status=%d counts=%+v", changed, w.Code, counts)
	}
	windowsLinkedDeny(t, ex, list, list.Files[0].Path)
}

func testWindowsLinkedPublicationRetarget(t *testing.T) {
	// Use a cold file: a cache hit need not emit any hash progress.
	root := t.TempDir()
	link := filepath.Join(root, "owner", "mlx")
	writeTestFile(t, filepath.Join(link, "weights.safetensors"), []byte("original"))
	windowsLinkedMove(t, link)
	replacement := t.TempDir()
	writeTestFile(t, filepath.Join(replacement, "weights.safetensors"), []byte("forbidden sentinel"))
	ex := windowsLinkedLMStudio(t, root, "mlx", "owner/mlx")
	changed := false
	w := &driftResponseWriter{ResponseRecorder: httptest.NewRecorder(), mutate: func() {
		if !changed {
			changed = true
			windowsLinkedRetarget(t, link, replacement)
		}
	}}
	(&controlServer{exec: ex}).handleModelFiles(w, httptest.NewRequest(http.MethodGet, controlModelFilesPath+"?engine=lmstudio&model=mlx", nil))
	decoder := json.NewDecoder(w.Body)
	failure := false
	for {
		var frame streamFrame
		if err := decoder.Decode(&frame); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatal(err)
		}
		if frame.Type == "result" {
			t.Fatal("changed selection published an issued result")
		}
		failure = failure || frame.Type == "error"
	}
	if !changed || !failure {
		t.Fatalf("publication hook changed=%v error=%v", changed, failure)
	}
}

// The injected opened metadata tests the existing checked-reader type boundary;
// it does not establish support for creating vendor-specific reparse tags.
type windowsLinkedTypeInfo struct {
	os.FileInfo
	mode os.FileMode
}

func (i windowsLinkedTypeInfo) Mode() os.FileMode { return i.mode }

func testWindowsLinkedOpenedTypes(t *testing.T) {
	for _, mode := range []os.FileMode{os.ModeIrregular, os.ModeDevice, os.ModeDir} {
		t.Run(mode.String(), func(t *testing.T) {
			ex, list, _, _ := windowsLinkedRetargetFixture(t)
			counts := &linkedHashCounts{}
			server := &controlServer{exec: ex, access: servingFileAccess{open: func(path string) (servingArtifactFile, error) {
				f, err := os.Open(path)
				if err != nil {
					return nil, err
				}
				counts.opens.Add(1)
				return &linkedHashFile{File: f, counts: counts, forbidden: true, stat: func() (os.FileInfo, error) {
					info, err := f.Stat()
					if err != nil {
						return nil, err
					}
					return windowsLinkedTypeInfo{FileInfo: info, mode: mode}, nil
				}}, nil
			}}}
			w := windowsLinkedGet(server, list, list.Files[0].Path)
			if w.Code != http.StatusConflict || counts.readAt.Load() != 0 || counts.bytes.Load() != 0 || counts.opens.Load() == 0 || counts.opens.Load() != counts.closes.Load() {
				t.Fatalf("opened type: status=%d counts=%+v", w.Code, counts)
			}
		})
	}
}

func testWindowsLinkedInvalidTrees(t *testing.T) {
	for _, kind := range []string{"cycle", "identity alias", "unresolvable junction"} {
		t.Run(kind, func(t *testing.T) {
			root := t.TempDir()
			mlx := filepath.Join(root, "owner", "mlx")
			writeTestFile(t, filepath.Join(mlx, "weights.safetensors"), []byte("weights"))
			switch kind {
			case "cycle":
				junction(t, filepath.Join(mlx, "cycle"), mlx)
			case "identity alias":
				writeTestFile(t, filepath.Join(mlx, "nested", "config.json"), []byte("config"))
				junction(t, filepath.Join(mlx, "alias"), filepath.Join(mlx, "nested"))
			case "unresolvable junction":
				target := t.TempDir()
				junction(t, filepath.Join(mlx, "missing"), target)
				if err := os.Remove(target); err != nil {
					t.Fatal(err)
				}
			}
			ex := windowsLinkedLMStudio(t, root, "mlx", "owner/mlx")
			list, err := ex.modelSourceFiles(context.Background(), "lmstudio", "mlx")
			if err == nil || len(list.Files) != 0 || list.source != nil {
				t.Fatalf("unsafe tree published files=%v err=%v", list.Files, err)
			}
		})
	}
}
