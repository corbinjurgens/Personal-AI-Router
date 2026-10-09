// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// copyTestExecutor builds an executor for one engine whose model store lives
// under modelsDir. installed controls whether its detect path exists.
func copyTestExecutor(t *testing.T, engine, modelsDir string, installed bool) *Executor {
	t.Helper()
	t.Setenv("OLLAMA_MODELS", "")
	t.Setenv("LLAMA_CACHE", "")
	bin := filepath.Join(t.TempDir(), "engine-bin")
	if installed {
		if err := os.WriteFile(bin, []byte("bin"), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	key := runtime.GOOS + "/" + runtime.GOARCH
	return newTestExecutor(t, &Manifest{
		Engine: engine, DisplayName: engine, ManifestVersion: 1,
		Platforms: map[string]Platform{key: {
			Detect:    []string{bin},
			ModelsDir: modelsDir,
			Runtime:   Runtime{Bin: bin},
		}},
	})
}

func sha256Hex(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

func writeTestFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func testBytes(n int, seed byte) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*7) + seed
	}
	return b
}

// ollamaFixture is a model written into an Ollama models root.
type ollamaFixture struct {
	manifestRel string
	manifest    []byte
	blobs       map[string][]byte // rel path -> content
}

// writeOllamaModel writes a model with a config blob and one weights blob of
// weightSize bytes under root ("<models_dir>/models").
func writeOllamaModel(t *testing.T, root, name string, weightSize int, seed byte) ollamaFixture {
	t.Helper()
	config := []byte(`{"model_format":"gguf","seed":` + string('0'+rune(seed%10)) + `}`)
	weights := testBytes(weightSize, seed)
	m := map[string]any{
		"schemaVersion": 2,
		"config":        map[string]any{"digest": "sha256:" + sha256Hex(config), "size": len(config)},
		"layers": []any{
			map[string]any{"digest": "sha256:" + sha256Hex(weights), "size": len(weights)},
		},
	}
	manifest, _ := json.Marshal(m)
	rel, err := ollamaManifestRel(name)
	if err != nil {
		t.Fatal(err)
	}
	fx := ollamaFixture{manifestRel: rel, manifest: manifest, blobs: map[string][]byte{
		"blobs/sha256-" + sha256Hex(config):  config,
		"blobs/sha256-" + sha256Hex(weights): weights,
	}}
	for p, b := range fx.blobs {
		writeTestFile(t, filepath.Join(root, filepath.FromSlash(p)), b)
	}
	writeTestFile(t, filepath.Join(root, filepath.FromSlash(rel)), manifest)
	return fx
}

// modelFileServer serves the two model-file routes for exec without the pin
// gate, as the other control-server handler tests do.
func modelFileServer(t *testing.T, exec *Executor, wrap func(http.Handler) http.Handler) (*httptest.Server, *remoteClient) {
	t.Helper()
	s := &controlServer{exec: exec}
	mux := http.NewServeMux()
	mux.HandleFunc(controlModelFilesPath, s.handleModelFiles)
	mux.HandleFunc(controlModelFilePath, s.handleModelFile)
	var h http.Handler = mux
	if wrap != nil {
		h = wrap(mux)
	}
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return srv, &remoteClient{http: srv.Client(), base: srv.URL}
}

func TestValidModelRelPath(t *testing.T) {
	good := []string{"blobs/sha256-abc", "manifests/registry.ollama.ai/library/x/latest", "a.gguf", "sub/dir/model-Q4_K_M.gguf"}
	bad := []string{"", "/etc/passwd", "../x", "a/../b", "a/./b", "a//b", `a\b`, "C:/x", "a/b.", "a/b ", "a\x00b", "a/"}
	for _, p := range good {
		if !validModelRelPath(p) {
			t.Errorf("validModelRelPath(%q) = false, want true", p)
		}
	}
	for _, p := range bad {
		if validModelRelPath(p) {
			t.Errorf("validModelRelPath(%q) = true, want false", p)
		}
	}
}

func TestOllamaManifestRel(t *testing.T) {
	cases := map[string]string{
		"llama3.2":                     "manifests/registry.ollama.ai/library/llama3.2/latest",
		"llama3.2:1b":                  "manifests/registry.ollama.ai/library/llama3.2/1b",
		"user/model:v1":                "manifests/registry.ollama.ai/user/model/v1",
		"hf.co/owner/repo-GGUF:Q4_K_M": "manifests/hf.co/owner/repo-GGUF/Q4_K_M",
	}
	for in, want := range cases {
		got, err := ollamaManifestRel(in)
		if err != nil || got != want {
			t.Errorf("ollamaManifestRel(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	for _, in := range []string{"", "../x", "a/b/c/d", "a/..:x", "x:..", "host:1/ns/m"} {
		if got, err := ollamaManifestRel(in); err == nil {
			t.Errorf("ollamaManifestRel(%q) = %q, want an error", in, got)
		}
	}
}

func TestModelFilesListsOllamaModelWithHashes(t *testing.T) {
	dir := t.TempDir()
	exec := copyTestExecutor(t, "ollama", dir, true)
	fx := writeOllamaModel(t, filepath.Join(dir, "models"), "tiny", 4096, 1)
	_, client := modelFileServer(t, exec, nil)

	list, err := client.modelFiles(context.Background(), "ollama", "tiny", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(list.Files) != 3 {
		t.Fatalf("files = %+v, want two blobs and the manifest", list.Files)
	}
	if last := list.Files[len(list.Files)-1]; last.Path != fx.manifestRel || last.SHA256 != sha256Hex(fx.manifest) || last.Size != int64(len(fx.manifest)) {
		t.Fatalf("manifest entry = %+v, want %s last with its real hash", last, fx.manifestRel)
	}
	for _, f := range list.Files[:2] {
		data, ok := fx.blobs[f.Path]
		if !ok || f.SHA256 != sha256Hex(data) || f.Size != int64(len(data)) {
			t.Fatalf("blob entry %+v does not match the fixture", f)
		}
	}
}

func TestModelFilesUnknownModelIs404(t *testing.T) {
	dir := t.TempDir()
	exec := copyTestExecutor(t, "ollama", dir, true)
	srv, _ := modelFileServer(t, exec, nil)
	resp, err := srv.Client().Get(srv.URL + controlModelFilesPath + "?engine=ollama&model=missing")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", resp.StatusCode)
	}
}

func TestModelFileRejectsTraversalAndUnlistedFiles(t *testing.T) {
	dir := t.TempDir()
	exec := copyTestExecutor(t, "ollama", dir, true)
	root := filepath.Join(dir, "models")
	fx := writeOllamaModel(t, root, "tiny", 1024, 1)
	other := writeOllamaModel(t, root, "other", 1024, 2)
	writeTestFile(t, filepath.Join(dir, "secret"), []byte("secret"))
	srv, _ := modelFileServer(t, exec, nil)

	get := func(path string) int {
		q := url.Values{"engine": {"ollama"}, "model": {"tiny"}, "path": {path}}
		resp, err := srv.Client().Get(srv.URL + controlModelFilePath + "?" + q.Encode())
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}
	for _, p := range []string{"../secret", "../../secret", "blobs/../../secret", "/etc/passwd", `blobs\..\..\secret`, ""} {
		if code := get(p); code != http.StatusBadRequest {
			t.Errorf("path %q: status %d, want 400", p, code)
		}
	}
	// A real file in the store that belongs to another model is not served.
	for p := range other.blobs {
		if _, mine := fx.blobs[p]; mine {
			continue
		}
		if code := get(p); code != http.StatusNotFound {
			t.Errorf("other model's %q: status %d, want 404", p, code)
		}
	}
	if code := get(other.manifestRel); code != http.StatusNotFound {
		t.Errorf("other model's manifest: status %d, want 404", code)
	}
	if code := get(fx.manifestRel); code != http.StatusOK {
		t.Errorf("listed manifest: status %d, want 200", code)
	}
}

func TestModelFileServesRanges(t *testing.T) {
	dir := t.TempDir()
	exec := copyTestExecutor(t, "ollama", dir, true)
	fx := writeOllamaModel(t, filepath.Join(dir, "models"), "tiny", 10000, 3)
	_, client := modelFileServer(t, exec, nil)
	var rel string
	for p, b := range fx.blobs {
		if len(b) == 10000 {
			rel = p
		}
	}
	body, start, err := client.modelFile(context.Background(), "ollama", "tiny", rel, 4000)
	if err != nil {
		t.Fatal(err)
	}
	defer body.Close()
	got, _ := io.ReadAll(body)
	if start != 4000 || string(got) != string(fx.blobs[rel][4000:]) {
		t.Fatalf("ranged read started at %d with %d bytes, want the tail from 4000", start, len(got))
	}
	if _, _, err := client.modelFile(context.Background(), "ollama", "tiny", rel, 20000); err != errRangeNotSatisfiable {
		t.Fatalf("range past the end: err = %v, want errRangeNotSatisfiable", err)
	}
}

func TestHashCacheReusesAndInvalidates(t *testing.T) {
	path := filepath.Join(t.TempDir(), "f")
	writeTestFile(t, path, []byte("one"))
	c := &hashCache{done: map[string]hashEntry{}, inflight: map[hashEntry]*hashCall{}}
	first, err := c.sum(context.Background(), path, nil)
	if err != nil || first != sha256Hex([]byte("one")) {
		t.Fatalf("sum = %q, %v", first, err)
	}
	c.mu.Lock()
	e := c.done[path]
	e.sum = "cached"
	c.done[path] = e
	c.mu.Unlock()
	if got, _ := c.sum(context.Background(), path, nil); got != "cached" {
		t.Fatalf("unchanged file was re-hashed: %q", got)
	}
	writeTestFile(t, path, []byte("changed!"))
	if got, _ := c.sum(context.Background(), path, nil); got != sha256Hex([]byte("changed!")) {
		t.Fatalf("changed file kept a stale hash: %q", got)
	}
}

func TestLlamaModelFileSet(t *testing.T) {
	paths := []string{
		"model-Q8_0.gguf",
		"model-Q4_K_M-00002-of-00002.gguf",
		"model-Q4_K_M-00001-of-00002.gguf",
		"mmproj-model-f16.gguf",
		"mmproj-model-Q8_0.gguf",
		"sub/model-Q4_K_M.gguf",
		"README.md",
	}
	got := strings.Join(llamaModelFileSet(paths, "Q4_K_M"), ",")
	want := "model-Q4_K_M-00001-of-00002.gguf,model-Q4_K_M-00002-of-00002.gguf,mmproj-model-Q8_0.gguf"
	if got != want {
		t.Fatalf("Q4_K_M set = %s, want %s", got, want)
	}
	if got := strings.Join(llamaModelFileSet(paths, "q8_0"), ","); got != "model-Q8_0.gguf,mmproj-model-Q8_0.gguf" {
		t.Fatalf("q8_0 set = %s", got)
	}
	if got := llamaModelFileSet(paths, "IQ2_XS"); got != nil {
		t.Fatalf("missing tag set = %v, want none", got)
	}
}

func TestSplitLlamaModel(t *testing.T) {
	repo, tag, err := splitLlamaModel("ggml-org/gemma-3-1b-it-GGUF:Q4_K_M")
	if err != nil || repo != "ggml-org/gemma-3-1b-it-GGUF" || tag != "Q4_K_M" {
		t.Fatalf("split = %q %q %v", repo, tag, err)
	}
	for _, bad := range []string{"noslash", "a/b/c", "../x", "a/..", "a/b:../x"} {
		if _, _, err := splitLlamaModel(bad); err == nil {
			t.Errorf("splitLlamaModel(%q) accepted", bad)
		}
	}
}
