// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
)

func TestModelPublisherOptionalManifestCannotBlockExport(t *testing.T) {
	path := filepath.Join(t.TempDir(), "artifact")
	writeTestFile(t, path, []byte("fixture"))
	evidence, err := captureSource(path, false)
	if err != nil {
		t.Fatal(err)
	}
	list := modelFileList{Engine: "llamacpp", Model: publisherTestRepo + ":Q4_K_M", Revision: publisherTestRevision}
	for i := 0; i < maxCopyArtifacts; i++ {
		name := fmt.Sprintf("%04d-%s.gguf", i, strings.Repeat("a", 230))
		observation := publisherTestRecord(sha256Hex([]byte("fixture")))
		observation.Path, observation.File = name, name
		list.Files = append(list.Files, modelFile{Path: name, Size: evidence.size, SHA256: observation.ActualDigest, Publisher: &observation, configured: path, abs: evidence.target, info: evidence.info})
	}
	identity, err := nativeExportID(list)
	if err != nil {
		t.Fatal(err)
	}
	list.ExportID = identity
	full, err := json.Marshal(list)
	if err != nil || len(full) <= maxCopyManifest {
		t.Fatalf("fixture does not cross optional envelope: bytes=%d err=%v", len(full), err)
	}
	encoded, err := marshalCopyManifest(list)
	if err != nil || len(encoded) > maxCopyManifest {
		t.Fatalf("observations blocked bounded export: bytes=%d err=%v", len(encoded), err)
	}
	var decoded modelFileList
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.ExportID != identity || len(decoded.Files) != len(list.Files) {
		t.Fatal("optional omission changed byte identity or file membership")
	}
	for i, file := range decoded.Files {
		if file.Publisher != nil || file.Path != list.Files[i].Path || file.SHA256 != list.Files[i].SHA256 || list.Files[i].Publisher == nil {
			t.Fatal("omission retained partial coverage, changed bytes, or mutated caller")
		}
	}
}

type modelDownloadBodyWriter struct {
	*httptest.ResponseRecorder
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func (w *modelDownloadBodyWriter) Write(body []byte) (int, error) {
	w.once.Do(func() { close(w.entered) })
	<-w.release
	return w.ResponseRecorder.Write(body)
}

func TestModelDownloadExportReaderHoldsThroughBody(t *testing.T) {
	root := t.TempDir()
	fixture := writeOllamaModel(t, filepath.Join(root, "models"), "demo", 64, 1)
	ex := copyTestExecutor(t, "ollama", root, true)
	copyTestNativeModels(t, ex, "ollama", "demo")
	server := &controlServer{exec: ex}
	_, client := modelFileServer(t, ex, nil)
	list, err := client.modelFiles(context.Background(), "ollama", "demo", nil)
	if err != nil {
		t.Fatal(err)
	}
	writer := &modelDownloadBodyWriter{ResponseRecorder: httptest.NewRecorder(), entered: make(chan struct{}), release: make(chan struct{})}
	var release sync.Once
	done := make(chan struct{})
	request := httptest.NewRequest(http.MethodGet, controlModelFilePath+"?engine=ollama&model=demo&path="+url.QueryEscape(fixture.manifestRel)+"&exportId="+url.QueryEscape(list.ExportID), nil)
	go func() {
		defer close(done)
		server.handleModelFile(writer, request)
	}()
	t.Cleanup(func() { release.Do(func() { close(writer.release) }); <-done })
	<-writer.entered
	var effects int
	err = ex.withModelMutation(context.Background(), "ollama", func() error { effects++; return nil })
	if !errors.Is(err, errDownloadBusy) || effects != 0 {
		t.Fatalf("mutation during body: effects=%d error=%v", effects, err)
	}
	release.Do(func() { close(writer.release) })
	<-done
	if writer.Code != http.StatusOK || !bytes.Equal(writer.Body.Bytes(), fixture.manifest) {
		t.Fatalf("export response: status=%d bytes=%d", writer.Code, writer.Body.Len())
	}
	releaseWriter, err := ex.downloads.acquireWrite("ollama")
	if err != nil {
		t.Fatalf("reader leaked after body: %v", err)
	}
	releaseWriter()
}

func TestModelDownloadExportReaderHoldsThroughHashing(t *testing.T) {
	root := t.TempDir()
	fixture := writeOllamaModel(t, filepath.Join(root, "models"), "demo", 64, 1)
	ex := copyTestExecutor(t, "ollama", root, true)
	copyTestNativeModels(t, ex, "ollama", "demo")
	abs, err := filepath.EvalSymlinks(filepath.Join(root, "models", filepath.FromSlash(fixture.manifestRel)))
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(abs)
	if err != nil {
		t.Fatal(err)
	}
	entered := make(chan struct{})
	call := &hashCall{done: make(chan struct{}), info: info, sum: sha256Hex(fixture.manifest), onWait: func() { close(entered) }}
	key := hashEntry{path: abs, size: info.Size(), mtime: info.ModTime().UnixNano()}
	fileHashes.mu.Lock()
	fileHashes.inflight[key] = call
	fileHashes.mu.Unlock()
	var release sync.Once
	done := make(chan struct{})
	writer := httptest.NewRecorder()
	go func() {
		defer close(done)
		(&controlServer{exec: ex}).handleModelFiles(writer, httptest.NewRequest(http.MethodGet, controlModelFilesPath+"?engine=ollama&model=demo", nil))
	}()
	t.Cleanup(func() {
		release.Do(func() { close(call.done) })
		<-done
		fileHashes.mu.Lock()
		delete(fileHashes.inflight, key)
		fileHashes.mu.Unlock()
	})
	<-entered
	err = ex.withModelMutation(context.Background(), "ollama", func() error { t.Error("mutation dispatched during hashing"); return nil })
	if !errors.Is(err, errDownloadBusy) {
		t.Fatalf("mutation during hashing = %v", err)
	}
	release.Do(func() { close(call.done) })
	<-done
	var result streamFrame
	if err := json.NewDecoder(writer.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if writer.Code != http.StatusOK || result.Type != "result" {
		t.Fatalf("hash response: status=%d frame=%+v", writer.Code, result)
	}
}

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

// copyTestNativeModels declares only the named models through the native action
// owner. Destination-only executors remain inert unless a test opts in.
func copyTestNativeModels(t *testing.T, ex *Executor, engine string, models ...string) *int {
	t.Helper()
	st, err := ex.state(engine)
	if err != nil {
		t.Fatalf("native membership engine: %v", err)
	}
	if st.manifest.Actions == nil {
		st.manifest.Actions = make(map[string]Action)
	}
	path := "/copy-test/" + engine + "/models"
	st.manifest.Actions["list_models"] = Action{
		HTTP:   &ActionHTTP{Method: http.MethodGet, Path: path},
		Result: &ActionResult{Array: "models", Field: "name"},
	}
	rows := make([]map[string]any, 0, len(models))
	for _, model := range models {
		rows = append(rows, map[string]any{"name": model, "status": map[string]string{"value": "unloaded"}})
	}
	body, err := json.Marshal(map[string]any{"models": rows})
	if err != nil {
		t.Fatalf("encode native membership: %v", err)
	}
	st.mu.Lock()
	st.running, st.runID, st.port = true, "copy-test-run", 18080
	st.launchDigest = strings.Repeat("a", 64)
	st.mu.Unlock()
	calls := new(int)
	var mu sync.Mutex
	ex.client.Transport = memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
		mu.Lock()
		*calls++
		mu.Unlock()
		if request.Method != http.MethodGet || request.URL.Path != path {
			t.Fatalf("unexpected native action: %s %s", request.Method, request.URL.Path)
		}
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(bytes.NewReader(body))}, nil
	})
	return calls
}

// Exercise the real listing and issued-ID owners before a fixture installs a
// transport failure. The in-memory lane does not affect peer request counters.
func copyTestSourceListing(t *testing.T, ex *Executor, engine, model string) modelFileList {
	t.Helper()
	control := &controlServer{exec: ex}
	client := &remoteClient{base: "https://source.invalid", controlHTTP: &http.Client{Transport: memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
		if request.Method != http.MethodGet || request.URL.Path != controlModelFilesPath {
			t.Fatalf("unexpected positive-control route: %s %s", request.Method, request.URL.Path)
		}
		response := httptest.NewRecorder()
		control.handleModelFiles(response, request)
		return response.Result(), nil
	})}}
	list, err := client.modelFiles(context.Background(), engine, model, nil)
	if err != nil {
		t.Fatalf("positive source listing: %v", err)
	}
	if list.Engine != engine || list.Model != model || list.Layout != "" || len(list.Files) == 0 || !validSHA256(list.ExportID) {
		t.Fatalf("positive source listing lost exact identity or files: %+v", list)
	}
	return list
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
	// Gate outside recording/interruption wrappers, before they can consume a request.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || (r.URL.Path != controlModelFilesPath && r.URL.Path != controlModelFilePath) {
			http.NotFound(w, r)
			return
		}
		h.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)
	return srv, &remoteClient{controlHTTP: srv.Client(), base: srv.URL}
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

func TestModelPublisherFullHashDoesNotTrustLocalLFSName(t *testing.T) {
	path := filepath.Join(t.TempDir(), strings.Repeat("a", 64))
	data := []byte("bytes different from the local LFS filename")
	writeTestFile(t, path, data)
	evidence, err := captureSource(path, false)
	if err != nil {
		t.Fatal(err)
	}
	list := modelFileList{Revision: publisherTestRevision, source: &sourceSnapshot{}, Files: []modelFile{{Path: publisherTestFile, Size: int64(len(data)), SHA256: strings.Repeat("a", 64), abs: evidence.target, configured: evidence.configured, info: evidence.info}}}
	cache := &hashCache{done: map[string]hashEntry{}, inflight: map[hashEntry]*hashCall{}}
	if err := cache.hashModelFiles(context.Background(), &list, nil); err != nil {
		t.Fatal(err)
	}
	if list.Files[0].SHA256 != sha256Hex(data) {
		t.Fatal("local LFS filename became a full-byte observation")
	}
	if err := cache.hashModelFiles(context.Background(), &list, nil); err != nil || len(cache.done) != 1 {
		t.Fatalf("checked full hash cache was not reused: entries=%d error=%v", len(cache.done), err)
	}
}

func TestModelPublisherSourceHashFailureRemainsHard(t *testing.T) {
	path := filepath.Join(t.TempDir(), "artifact.gguf")
	data := []byte("synthetic artifact")
	writeTestFile(t, path, data)
	evidence, err := captureSource(path, false)
	if err != nil {
		t.Fatal(err)
	}
	list := modelFileList{Layout: llamaArtifactsLayout, Model: "alias", Files: []modelFile{{Path: "model/artifact.gguf", Role: "model", Size: int64(len(data)), abs: evidence.target, configured: evidence.configured, info: evidence.info, asserted: strings.Repeat("f", 64)}}}
	if err := hashModelFiles(context.Background(), &list, nil); err == nil || list.ExportID != "" {
		t.Fatal("declared artifact mismatch became observational")
	}
	list.Files[0].asserted = ""
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := hashModelFiles(context.Background(), &list, nil); err == nil || list.ExportID != "" {
		t.Fatal("hash read failure issued a successful export")
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
	copyTestNativeModels(t, exec, "ollama", "tiny")
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
	copyTestNativeModels(t, exec, "ollama")
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
	copyTestNativeModels(t, exec, "ollama", "tiny", "other")
	writeTestFile(t, filepath.Join(dir, "secret"), []byte("secret"))
	srv, client := modelFileServer(t, exec, nil)
	list, err := client.modelFiles(context.Background(), "ollama", "tiny", nil)
	if err != nil {
		t.Fatal(err)
	}

	get := func(path string) int {
		q := url.Values{"engine": {"ollama"}, "model": {"tiny"}, "path": {path}, "exportId": {list.ExportID}}
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
	copyTestNativeModels(t, exec, "ollama", "tiny")
	_, client := modelFileServer(t, exec, nil)
	list, err := client.modelFiles(context.Background(), "ollama", "tiny", nil)
	if err != nil {
		t.Fatal(err)
	}
	var rel string
	for p, b := range fx.blobs {
		if len(b) == 10000 {
			rel = p
		}
	}
	body, start, err := client.modelFile(context.Background(), "ollama", "tiny", rel, list.ExportID, 4000)
	if err != nil {
		t.Fatal(err)
	}
	defer body.Close()
	got, _ := io.ReadAll(body)
	if start != 4000 || string(got) != string(fx.blobs[rel][4000:]) {
		t.Fatalf("ranged read started at %d with %d bytes, want the tail from 4000", start, len(got))
	}
	if _, _, err := client.modelFile(context.Background(), "ollama", "tiny", rel, list.ExportID, 20000); err != errRangeNotSatisfiable {
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
	if got, err := c.sum(context.Background(), path, nil); err != nil || got != "cached" {
		t.Fatalf("unchanged file was re-hashed: sum=%q error=%v", got, err)
	}
	writeTestFile(t, path, []byte("changed!"))
	if got, err := c.sum(context.Background(), path, nil); err != nil || got != sha256Hex([]byte("changed!")) {
		t.Fatalf("changed file kept a stale hash: sum=%q error=%v", got, err)
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
