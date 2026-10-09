// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeModelSource is an in-process modelFileSource that records what the copy
// asked for.
type fakeModelSource struct {
	list     modelFileList
	data     map[string][]byte
	failPath string
	mu       sync.Mutex
	listed   int
	fetched  []string
	// onFetch runs before each file is served.
	onFetch func(path string)
}

func (f *fakeModelSource) modelFiles(context.Context, string, string, func(streamFrame)) (modelFileList, error) {
	f.mu.Lock()
	f.listed++
	f.mu.Unlock()
	out := f.list
	out.Files = append([]modelFile(nil), f.list.Files...)
	return out, nil
}

func (f *fakeModelSource) modelFile(_ context.Context, _, _, path string, offset int64) (io.ReadCloser, int64, error) {
	f.mu.Lock()
	f.fetched = append(f.fetched, path)
	f.mu.Unlock()
	if f.onFetch != nil {
		f.onFetch(path)
	}
	if path == f.failPath {
		return nil, 0, permanentCopyError{errors.New("source refused")}
	}
	return io.NopCloser(bytes.NewReader(f.data[path][offset:])), offset, nil
}

func (f *fakeModelSource) fetchedPaths() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.fetched...)
}

// sourceFromExecutor lists model on exec the way the ec route would, and
// returns a fake source serving those files.
func sourceFromExecutor(t *testing.T, exec *Executor, engine, model string) *fakeModelSource {
	t.Helper()
	list, err := exec.modelSourceFiles(context.Background(), engine, model)
	if err != nil {
		t.Fatal(err)
	}
	if err := hashModelFiles(context.Background(), &list, func(string, int) {}); err != nil {
		t.Fatal(err)
	}
	src := &fakeModelSource{list: list, data: map[string][]byte{}}
	for _, f := range list.Files {
		b, err := os.ReadFile(f.abs)
		if err != nil {
			t.Fatal(err)
		}
		src.data[f.Path] = b
	}
	return src
}

func plentyOfDisk(t *testing.T) {
	t.Helper()
	prev := diskFreeBytes
	diskFreeBytes = func(string) (uint64, error) { return 1 << 50, nil }
	t.Cleanup(func() { diskFreeBytes = prev })
}

func fastRetries(t *testing.T, attempts int) {
	t.Helper()
	prevA, prevB := copyFileAttempts, copyRetryBackoff
	copyFileAttempts, copyRetryBackoff = attempts, time.Millisecond
	t.Cleanup(func() { copyFileAttempts, copyRetryBackoff = prevA, prevB })
}

func assertNoPartFiles(t *testing.T, root string) {
	t.Helper()
	_ = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err == nil && strings.HasSuffix(path, ".part") {
			t.Errorf("leftover partial %s", path)
		}
		return nil
	})
}

func TestCopyModelOllamaOverControlRoutes(t *testing.T) {
	plentyOfDisk(t)
	srcDir, dstDir := t.TempDir(), t.TempDir()
	srcExec := copyTestExecutor(t, "ollama", srcDir, true)
	fx := writeOllamaModel(t, filepath.Join(srcDir, "models"), "tiny:1b", 300_000, 4)
	_, client := modelFileServer(t, srcExec, nil)
	dst := copyTestExecutor(t, "ollama", dstDir, true)

	var mu sync.Mutex
	var frames []copyProgress
	res, err := dst.CopyModelFrom(context.Background(), client, "ollama", "tiny:1b", func(p copyProgress) {
		mu.Lock()
		frames = append(frames, p)
		mu.Unlock()
	})
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(dstDir, "models")
	for rel, want := range fx.blobs {
		got, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil || !bytes.Equal(got, want) {
			t.Fatalf("%s not copied intact: %v", rel, err)
		}
	}
	if got, _ := os.ReadFile(filepath.Join(root, filepath.FromSlash(fx.manifestRel))); !bytes.Equal(got, fx.manifest) {
		t.Fatal("manifest not copied intact")
	}
	assertNoPartFiles(t, root)
	if res.Files != 3 || res.FilesSkipped != 0 || res.BytesCopied != res.BytesTotal {
		t.Fatalf("result = %+v", res)
	}
	last := frames[len(frames)-1]
	if last.Stage != "done" || last.Percent != 100 || last.BytesDone != res.BytesTotal {
		t.Fatalf("last progress frame = %+v, want done at 100%%", last)
	}
	sawDownload := false
	for _, f := range frames {
		if f.Stage == "downloading" && f.File != "" && f.BytesTotal == res.BytesTotal {
			sawDownload = true
		}
	}
	if !sawDownload {
		t.Fatalf("no downloading frame with file and byte totals in %+v", frames)
	}
	// The copy's frames serialize to the engine:remote-progress shape.
	b, _ := json.Marshal(copyProgress{remoteProgress: remoteProgress{OpID: "op", Node: "n", Op: "copy", Stage: "downloading", Percent: 5}, File: "f", BytesDone: 1, BytesTotal: 2})
	if !strings.Contains(string(b), `"opId":"op"`) || !strings.Contains(string(b), `"bytesTotal":2`) || !strings.Contains(string(b), `"file":"f"`) {
		t.Fatalf("progress JSON = %s", b)
	}
}

// interruptAfter aborts the first response for a path matching match once n
// body bytes have been sent, leaving the client with a truncated download.
func interruptAfter(match string, n int) func(http.Handler) http.Handler {
	var once sync.Once
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			cut := false
			if r.URL.Path == controlModelFilePath && strings.Contains(r.URL.Query().Get("path"), match) {
				once.Do(func() { cut = true })
			}
			if !cut {
				next.ServeHTTP(w, r)
				return
			}
			next.ServeHTTP(&truncatingWriter{ResponseWriter: w, left: n}, r)
		})
	}
}

type truncatingWriter struct {
	http.ResponseWriter
	left int
}

func (w *truncatingWriter) Write(p []byte) (int, error) {
	if len(p) <= w.left {
		w.left -= len(p)
		return w.ResponseWriter.Write(p)
	}
	_, _ = w.ResponseWriter.Write(p[:w.left])
	w.ResponseWriter.(http.Flusher).Flush()
	panic(http.ErrAbortHandler)
}

func TestCopyModelResumesWithRangeAfterInterruption(t *testing.T) {
	plentyOfDisk(t)
	fastRetries(t, 1)
	srcDir, dstDir := t.TempDir(), t.TempDir()
	srcExec := copyTestExecutor(t, "ollama", srcDir, true)
	fx := writeOllamaModel(t, filepath.Join(srcDir, "models"), "tiny", 500_000, 5)
	var weightsRel string
	for rel, b := range fx.blobs {
		if len(b) == 500_000 {
			weightsRel = rel
		}
	}
	var mu sync.Mutex
	var ranges []string
	record := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == controlModelFilePath && r.URL.Query().Get("path") == weightsRel {
				mu.Lock()
				ranges = append(ranges, r.Header.Get("Range"))
				mu.Unlock()
			}
			next.ServeHTTP(w, r)
		})
	}
	cut := interruptAfter(weightsRel, 200_000)
	_, client := modelFileServer(t, srcExec, func(h http.Handler) http.Handler { return record(cut(h)) })
	dst := copyTestExecutor(t, "ollama", dstDir, true)
	root := filepath.Join(dstDir, "models")
	part := filepath.Join(root, filepath.FromSlash(weightsRel)) + ".part"

	// First run: the only attempt is cut off, and the partial is kept.
	if _, err := dst.CopyModelFrom(context.Background(), client, "ollama", "tiny", nil); err == nil {
		t.Fatal("interrupted copy reported success")
	}
	info, err := os.Stat(part)
	if err != nil || info.Size() == 0 || info.Size() >= 500_000 {
		t.Fatalf("partial after interruption: %v, %v", info, err)
	}
	if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(fx.manifestRel))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("manifest exists after a failed copy: %v", err)
	}

	// Second run (a retry or a restart) resumes from the partial.
	if _, err := dst.CopyModelFrom(context.Background(), client, "ollama", "tiny", nil); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(filepath.Join(root, filepath.FromSlash(weightsRel)))
	if !bytes.Equal(got, fx.blobs[weightsRel]) {
		t.Fatal("resumed blob does not match the source")
	}
	mu.Lock()
	defer mu.Unlock()
	if len(ranges) != 2 || ranges[0] != "" || ranges[1] != "bytes="+itoa(info.Size())+"-" {
		t.Fatalf("Range headers = %q, want none then bytes=%d-", ranges, info.Size())
	}
	assertNoPartFiles(t, root)
}

func itoa(n int64) string {
	b, _ := json.Marshal(n)
	return string(b)
}

func TestCopyModelHashMismatchRejectsAndDeletesPartial(t *testing.T) {
	plentyOfDisk(t)
	srcDir, dstDir := t.TempDir(), t.TempDir()
	srcExec := copyTestExecutor(t, "ollama", srcDir, true)
	fx := writeOllamaModel(t, filepath.Join(srcDir, "models"), "tiny", 50_000, 6)
	src := sourceFromExecutor(t, srcExec, "ollama", "tiny")
	// The source serves different bytes than it listed for the manifest.
	src.data[fx.manifestRel] = append([]byte(nil), src.data[fx.manifestRel]...)
	src.data[fx.manifestRel][0] ^= 0xff
	dst := copyTestExecutor(t, "ollama", dstDir, true)

	_, err := dst.CopyModelFrom(context.Background(), src, "ollama", "tiny", nil)
	if err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("err = %v, want a sha256 mismatch", err)
	}
	dest := filepath.Join(dstDir, "models", filepath.FromSlash(fx.manifestRel))
	if _, err := os.Stat(dest + ".part"); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("mismatched partial kept: %v", err)
	}
	if _, err := os.Stat(dest); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("mismatched file renamed into place: %v", err)
	}
}

func TestCopyModelSkipsFilesAlreadyPresent(t *testing.T) {
	plentyOfDisk(t)
	srcDir, dstDir := t.TempDir(), t.TempDir()
	srcExec := copyTestExecutor(t, "ollama", srcDir, true)
	fx := writeOllamaModel(t, filepath.Join(srcDir, "models"), "tiny", 80_000, 7)
	src := sourceFromExecutor(t, srcExec, "ollama", "tiny")
	dst := copyTestExecutor(t, "ollama", dstDir, true)
	var present string
	for rel, b := range fx.blobs {
		if len(b) == 80_000 {
			present = rel
			writeTestFile(t, filepath.Join(dstDir, "models", filepath.FromSlash(rel)), b)
		}
	}
	res, err := dst.CopyModelFrom(context.Background(), src, "ollama", "tiny", nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range src.fetchedPaths() {
		if p == present {
			t.Fatalf("already-present %s was downloaded again", p)
		}
	}
	if res.FilesSkipped != 1 || res.BytesCopied != res.BytesTotal-80_000 {
		t.Fatalf("result = %+v", res)
	}
}

func TestCopyModelOllamaWritesManifestLast(t *testing.T) {
	plentyOfDisk(t)
	srcDir, dstDir := t.TempDir(), t.TempDir()
	srcExec := copyTestExecutor(t, "ollama", srcDir, true)
	fx := writeOllamaModel(t, filepath.Join(srcDir, "models"), "tiny", 20_000, 8)
	src := sourceFromExecutor(t, srcExec, "ollama", "tiny")
	dst := copyTestExecutor(t, "ollama", dstDir, true)
	root := filepath.Join(dstDir, "models")
	manifest := filepath.Join(root, filepath.FromSlash(fx.manifestRel))
	src.onFetch = func(path string) {
		if path == fx.manifestRel {
			// Every blob is in place before the manifest is fetched.
			for rel := range fx.blobs {
				if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel))); err != nil {
					t.Errorf("manifest fetched before blob %s was in place", rel)
				}
			}
		} else if _, err := os.Stat(manifest); err == nil {
			t.Errorf("manifest present while blob %s was still downloading", path)
		}
	}
	if _, err := dst.CopyModelFrom(context.Background(), src, "ollama", "tiny", nil); err != nil {
		t.Fatal(err)
	}
	fetched := src.fetchedPaths()
	if fetched[len(fetched)-1] != fx.manifestRel {
		t.Fatalf("fetch order %v, want the manifest last", fetched)
	}

	// A copy that fails on a blob never writes the manifest.
	dst2Dir := t.TempDir()
	dst2 := copyTestExecutor(t, "ollama", dst2Dir, true)
	src2 := sourceFromExecutor(t, srcExec, "ollama", "tiny")
	for rel, b := range fx.blobs {
		if len(b) == 20_000 {
			src2.failPath = rel
		}
	}
	if _, err := dst2.CopyModelFrom(context.Background(), src2, "ollama", "tiny", nil); err == nil {
		t.Fatal("copy with a failing blob reported success")
	}
	if _, err := os.Stat(filepath.Join(dst2Dir, "models", filepath.FromSlash(fx.manifestRel))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("manifest written although a blob failed: %v", err)
	}
}

func TestCopyModelRefusesWhenDiskSpaceIsShort(t *testing.T) {
	srcDir, dstDir := t.TempDir(), t.TempDir()
	srcExec := copyTestExecutor(t, "ollama", srcDir, true)
	writeOllamaModel(t, filepath.Join(srcDir, "models"), "tiny", 100_000, 9)
	src := sourceFromExecutor(t, srcExec, "ollama", "tiny")
	dst := copyTestExecutor(t, "ollama", dstDir, true)
	prev := diskFreeBytes
	var asked string
	diskFreeBytes = func(path string) (uint64, error) { asked = path; return 1 << 20, nil }
	t.Cleanup(func() { diskFreeBytes = prev })

	_, err := dst.CopyModelFrom(context.Background(), src, "ollama", "tiny", nil)
	if err == nil || !strings.Contains(err.Error(), "not enough free disk space") {
		t.Fatalf("err = %v, want a disk-space refusal", err)
	}
	if got := src.fetchedPaths(); len(got) != 0 {
		t.Fatalf("downloaded %v before refusing", got)
	}
	// The models root does not exist yet; the check measures its nearest
	// existing ancestor.
	if asked != dstDir {
		t.Fatalf("free space checked at %q, want %q", asked, dstDir)
	}
}

func TestCopyModelRefusesUninstalledEngine(t *testing.T) {
	src := &fakeModelSource{}
	dst := copyTestExecutor(t, "ollama", t.TempDir(), false)
	_, err := dst.CopyModelFrom(context.Background(), src, "ollama", "tiny", nil)
	if err == nil || !strings.Contains(err.Error(), "not installed") {
		t.Fatalf("err = %v, want not installed", err)
	}
	if src.listed != 0 {
		t.Fatal("the source was contacted for an engine that is not installed")
	}
}

func TestCopyModelRefusesUnsupportedEngine(t *testing.T) {
	dst := copyTestExecutor(t, "vllm", t.TempDir(), true)
	_, err := dst.CopyModelFrom(context.Background(), &fakeModelSource{}, "vllm", "m", nil)
	if !errors.Is(err, errModelCopyUnsupported) {
		t.Fatalf("err = %v, want errModelCopyUnsupported", err)
	}
}

func TestCopyModelRejectsUnsafeListing(t *testing.T) {
	plentyOfDisk(t)
	dst := copyTestExecutor(t, "ollama", t.TempDir(), true)
	sum := sha256Hex([]byte("x"))
	for _, list := range []modelFileList{
		{Files: []modelFile{{Path: "../../evil", Size: 1, SHA256: sum}}},
		{Files: []modelFile{{Path: "blobs/sha256-" + sum, Size: 1, SHA256: sum}, {Path: "manifests/x/y/z/w", Size: 1, SHA256: sum}}},
		{Files: []modelFile{{Path: "blobs/sha256-" + sum, Size: 1, SHA256: "nothex"}}},
	} {
		src := &fakeModelSource{list: list}
		if _, err := dst.CopyModelFrom(context.Background(), src, "ollama", "tiny", nil); err == nil {
			t.Errorf("listing %+v accepted", list.Files)
		}
		if len(src.fetchedPaths()) != 0 {
			t.Errorf("listing %+v led to downloads", list.Files)
		}
	}
}

// writeLlamaCache writes a cached repo the way llama.cpp's hub cache does:
// blobs named by sha256, snapshot entries linking to them (copied files where
// links are unavailable), and refs/main.
func writeLlamaCache(t *testing.T, root, repo, commit string, files map[string][]byte) {
	t.Helper()
	repoDir := llamaRepoDir(root, repo)
	for rel, b := range files {
		blob := filepath.Join(repoDir, "blobs", sha256Hex(b))
		writeTestFile(t, blob, b)
		link := filepath.Join(repoDir, "snapshots", commit, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
			t.Fatal(err)
		}
		target, _ := filepath.Rel(filepath.Dir(link), blob)
		if runtime.GOOS == "windows" || os.Symlink(target, link) != nil {
			writeTestFile(t, link, b)
		}
	}
	writeTestFile(t, filepath.Join(repoDir, "refs", "main"), []byte(commit))
}

func TestCopyModelLlamacpp(t *testing.T) {
	plentyOfDisk(t)
	srcDir, dstDir := t.TempDir(), t.TempDir()
	const repo, commit = "owner/tiny-GGUF", "0123456789abcdef0123456789abcdef01234567"
	files := map[string][]byte{
		"tiny-Q4_K_M.gguf":      testBytes(40_000, 1),
		"tiny-Q8_0.gguf":        testBytes(60_000, 2),
		"mmproj-tiny-Q8_0.gguf": testBytes(5_000, 3),
	}
	writeLlamaCache(t, srcDir, repo, commit, files)
	srcExec := copyTestExecutor(t, "llamacpp", srcDir, true)
	_, client := modelFileServer(t, srcExec, nil)

	list, err := client.modelFiles(context.Background(), "llamacpp", repo+":Q4_K_M", nil)
	if err != nil {
		t.Fatal(err)
	}
	if list.Revision != commit || len(list.Files) != 2 || list.Files[0].Path != "tiny-Q4_K_M.gguf" || list.Files[1].Path != "mmproj-tiny-Q8_0.gguf" {
		t.Fatalf("listing = %+v", list)
	}
	for _, f := range list.Files {
		if f.SHA256 != sha256Hex(files[f.Path]) {
			t.Fatalf("%s listed with sha256 %s", f.Path, f.SHA256)
		}
	}

	dst := copyTestExecutor(t, "llamacpp", dstDir, true)
	src := sourceFromExecutor(t, srcExec, "llamacpp", repo+":Q4_K_M")
	refs := filepath.Join(llamaRepoDir(dstDir, repo), "refs", "main")
	src.onFetch = func(string) {
		if _, err := os.Stat(refs); err == nil {
			t.Error("refs/main written before the files were in place")
		}
	}
	if _, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", repo+":Q4_K_M", nil); err != nil {
		t.Fatal(err)
	}
	if fetched := src.fetchedPaths(); fetched[len(fetched)-1] != "tiny-Q4_K_M.gguf" {
		t.Fatalf("fetch order %v, want the model GGUF last", fetched)
	}
	if got := llamaCachedRef(llamaRepoDir(dstDir, repo)); got != commit {
		t.Fatalf("destination ref = %q, want %q", got, commit)
	}
	for _, rel := range []string{"tiny-Q4_K_M.gguf", "mmproj-tiny-Q8_0.gguf"} {
		got, err := os.ReadFile(filepath.Join(llamaRepoDir(dstDir, repo), "snapshots", commit, rel))
		if err != nil || !bytes.Equal(got, files[rel]) {
			t.Fatalf("%s not copied intact: %v", rel, err)
		}
	}
	// The destination's own cache lists it the way llama.cpp would.
	back, err := dst.modelSourceFiles(context.Background(), "llamacpp", repo+":Q4_K_M")
	if err != nil || len(back.Files) != 2 {
		t.Fatalf("destination listing = %+v, %v", back, err)
	}
}

func TestCopyModelLMStudio(t *testing.T) {
	plentyOfDisk(t)
	srcDir, dstDir := t.TempDir(), t.TempDir()
	files := map[string][]byte{
		"lmstudio-community/tiny-GGUF/tiny-Q4_K_M.gguf":   testBytes(30_000, 1),
		"lmstudio-community/tiny-GGUF/mmproj-tiny.gguf":   testBytes(3_000, 2),
		"lmstudio-community/tiny-GGUF/tiny-Q8_0.gguf":     testBytes(50_000, 3),
		"mlx-community/tiny-4bit/model.safetensors":       testBytes(20_000, 4),
		"mlx-community/tiny-4bit/config.json":             []byte(`{}`),
		"mlx-community/tiny-4bit/tokenizer/vocab.json":    []byte(`{"a":1}`),
		"mlx-community/other-4bit/model.safetensors":      testBytes(1_000, 5),
		"lmstudio-community/tiny-GGUF/unrelated-readme":   []byte("x"),
		"lmstudio-community/other-GGUF/other-Q4_K_M.gguf": testBytes(1_000, 6),
	}
	for rel, b := range files {
		writeTestFile(t, filepath.Join(srcDir, filepath.FromSlash(rel)), b)
	}
	prev := lmsListModels
	lmsListModels = func(context.Context, *Executor, string) ([]lmsListEntry, error) {
		return []lmsListEntry{
			{ModelKey: "tiny", Path: "lmstudio-community/tiny-GGUF/tiny-Q4_K_M.gguf"},
			{ModelKey: "tiny-q8", Path: "lmstudio-community/tiny-GGUF/tiny-Q8_0.gguf"},
			{ModelKey: "tiny-mlx", Path: "mlx-community/tiny-4bit"},
			{ModelKey: "other", Path: "lmstudio-community/other-GGUF/other-Q4_K_M.gguf"},
		}, nil
	}
	t.Cleanup(func() { lmsListModels = prev })
	srcExec := copyTestExecutor(t, "lmstudio", srcDir, true)

	gguf := sourceFromExecutor(t, srcExec, "lmstudio", "tiny")
	var paths []string
	for _, f := range gguf.list.Files {
		paths = append(paths, f.Path)
	}
	if strings.Join(paths, ",") != "lmstudio-community/tiny-GGUF/tiny-Q4_K_M.gguf,lmstudio-community/tiny-GGUF/mmproj-tiny.gguf" {
		t.Fatalf("GGUF listing = %v", paths)
	}
	mlx := sourceFromExecutor(t, srcExec, "lmstudio", "tiny-mlx")
	if len(mlx.list.Files) != 3 {
		t.Fatalf("MLX listing = %+v", mlx.list.Files)
	}
	if _, err := srcExec.modelSourceFiles(context.Background(), "lmstudio", "lmstudio-community"); err == nil {
		t.Fatal("a publisher prefix matching several models was accepted")
	}

	dst := copyTestExecutor(t, "lmstudio", dstDir, true)
	for _, src := range []*fakeModelSource{gguf, mlx} {
		if _, err := dst.CopyModelFrom(context.Background(), src, "lmstudio", src.list.Model, nil); err != nil {
			t.Fatal(err)
		}
		for _, f := range src.list.Files {
			got, err := os.ReadFile(filepath.Join(dstDir, filepath.FromSlash(f.Path)))
			if err != nil || !bytes.Equal(got, files[f.Path]) {
				t.Fatalf("%s not copied intact: %v", f.Path, err)
			}
		}
	}
	if fetched := gguf.fetchedPaths(); fetched[len(fetched)-1] != "lmstudio-community/tiny-GGUF/tiny-Q4_K_M.gguf" {
		t.Fatalf("fetch order %v, want the model GGUF last", fetched)
	}
}

type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) Read(p []byte) (int, error) { return 0, io.EOF }

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

func TestRunRemoteCopyValidatesParams(t *testing.T) {
	for params, want := range map[string]string{
		`{"engine":"ollama","model":"x"}`:                "node is required",
		`{"node":"n","model":"x"}`:                       "engine is required",
		`{"node":"n","engine":"ollama"}`:                 "model is required",
		`{"node":"ghost","engine":"ollama","model":"x"}`: "not a discovered ec peer",
	} {
		out := &lockedBuffer{}
		m := NewManager(NewCodec(out), &Executor{progress: newProgressHub()}, nil)
		id := json.RawMessage("1")
		// Through the dispatcher, which runs the copy on its own goroutine.
		m.handleMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:remote-copy-model", Params: json.RawMessage(params)})
		deadline := time.Now().Add(2 * time.Second)
		for !strings.Contains(out.String(), want) && time.Now().Before(deadline) {
			time.Sleep(5 * time.Millisecond)
		}
		mustContain(t, out.String(), want)
	}
}
