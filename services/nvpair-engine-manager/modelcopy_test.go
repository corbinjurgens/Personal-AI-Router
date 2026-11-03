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
	"net/url"
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

func (f *fakeModelSource) modelFile(_ context.Context, _, _, path, exportID string, offset int64) (io.ReadCloser, int64, error) {
	if exportID != f.list.ExportID {
		return nil, 0, permanentCopyError{errors.New("export identity mismatch")}
	}
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

func TestCopyModelOverControlRoutes(t *testing.T) {
	plentyOfDisk(t)
	srcDir, dstDir := t.TempDir(), t.TempDir()
	const repo, commit = "owner/tiny-GGUF", "0123456789abcdef0123456789abcdef01234567"
	files := map[string][]byte{
		"tiny-Q4_K_M.gguf":      testBytes(300_000, 4),
		"mmproj-tiny-Q8_0.gguf": testBytes(5_000, 3),
	}
	writeLlamaCache(t, srcDir, repo, commit, files)
	srcExec := copyTestExecutor(t, "llamacpp", srcDir, true)
	copyTestNativeModels(t, srcExec, "llamacpp", repo+":Q4_K_M")
	_, client := modelFileServer(t, srcExec, nil)
	dst := copyTestExecutor(t, "llamacpp", dstDir, true)

	var mu sync.Mutex
	var frames []copyProgress
	res, err := dst.CopyModelFrom(context.Background(), client, "llamacpp", repo+":Q4_K_M", func(p copyProgress) {
		mu.Lock()
		frames = append(frames, p)
		mu.Unlock()
	})
	if err != nil {
		t.Fatal(err)
	}
	root := dstDir
	for rel, want := range files {
		got, err := os.ReadFile(filepath.Join(llamaRepoDir(root, repo), "snapshots", commit, rel))
		if err != nil || !bytes.Equal(got, want) {
			t.Fatalf("%s not copied intact: %v", rel, err)
		}
	}
	assertNoPartFiles(t, root)
	if res.Files != 2 || res.FilesSkipped != 0 || res.BytesCopied != res.BytesTotal {
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

func TestLinkedModelNativeTransferToken(t *testing.T) {
	plentyOfDisk(t)
	fastRetries(t, 1)
	srcDir, dstDir := t.TempDir(), t.TempDir()
	const repo, commit, model = "owner/tiny-GGUF", "0123456789abcdef0123456789abcdef01234567", "owner/tiny-GGUF:Q4_K_M"
	const weightsRel = "tiny-Q4_K_M.gguf"
	weights := testBytes(500_000, 5)
	writeLlamaCache(t, srcDir, repo, commit, map[string][]byte{weightsRel: weights, "mmproj-tiny-Q8_0.gguf": testBytes(5_000, 6)})
	srcExec := copyTestExecutor(t, "llamacpp", srcDir, true)
	copyTestNativeModels(t, srcExec, "llamacpp", model)
	copyTestSourceListing(t, srcExec, "llamacpp", model)
	var mu sync.Mutex
	var ranges []string
	var tokens []string
	record := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == controlModelFilePath && r.URL.Query().Get("path") == weightsRel {
				mu.Lock()
				ranges = append(ranges, r.Header.Get("Range"))
				tokens = append(tokens, r.URL.Query().Get("exportId"))
				mu.Unlock()
			}
			next.ServeHTTP(w, r)
		})
	}
	cut := interruptAfter(weightsRel, 200_000)
	server, client := modelFileServer(t, srcExec, func(h http.Handler) http.Handler { return record(cut(h)) })
	assertFixtureRejectsProbe(t, server.URL)
	assertFixtureRejectsRequest(t, server.URL, http.MethodPost, controlModelFilePath+"?path="+url.QueryEscape(weightsRel))
	dst := copyTestExecutor(t, "llamacpp", dstDir, true)
	root := dstDir
	snapshot := filepath.Join(llamaRepoDir(root, repo), "snapshots", commit)
	part := filepath.Join(snapshot, weightsRel) + ".part"

	// First run: the only attempt is cut off, and the partial is kept.
	if _, err := dst.CopyModelFrom(context.Background(), client, "llamacpp", model, nil); err == nil {
		t.Fatal("interrupted copy reported success")
	}
	info, err := os.Stat(part)
	if err != nil || info.Size() == 0 || info.Size() >= 500_000 {
		t.Fatalf("partial after interruption: %v, %v", info, err)
	}
	if _, err := os.Stat(filepath.Join(llamaRepoDir(root, repo), "refs", "main")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("cache ref exists after a failed copy: %v", err)
	}

	// Second run (a retry or a restart) resumes from the partial.
	var progress []copyProgress
	result, err := dst.CopyModelFrom(context.Background(), client, "llamacpp", model, func(p copyProgress) { progress = append(progress, p) })
	if err != nil {
		t.Fatal(err)
	}
	if result.Layout != "" || result.ExportID != "" || result.Durability != "" || result.RequiresConfiguration {
		t.Fatalf("native result contains durable identity: %+v", result)
	}
	for _, p := range progress {
		if p.ExportID != "" {
			t.Fatalf("native progress contains export identity: %+v", p)
		}
	}
	got, err := os.ReadFile(filepath.Join(snapshot, weightsRel))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, weights) {
		t.Fatal("resumed file does not match the source")
	}
	mu.Lock()
	defer mu.Unlock()
	if len(ranges) != 2 || ranges[0] != "" || ranges[1] != "bytes="+itoa(info.Size())+"-" {
		t.Fatalf("Range headers = %q, want none then bytes=%d-", ranges, info.Size())
	}
	if len(tokens) != 2 || tokens[0] == "" || tokens[0] != tokens[1] {
		t.Fatalf("native transfer tokens = %q", tokens)
	}
	assertNoPartFiles(t, root)
}

func itoa(n int64) string {
	b, _ := json.Marshal(n)
	return string(b)
}

func TestCopyModelHashMismatchRejectsAndDeletesPartial(t *testing.T) {
	plentyOfDisk(t)
	fixture := namedCustomCopyFixture(t, "tiny", 50_000, 6)
	src := sourceFromExecutor(t, fixture.source, "llamacpp", "tiny")
	path := src.list.Files[0].Path
	src.data[path] = append([]byte(nil), src.data[path]...)
	src.data[path][0] ^= 0xff
	dst, store := customDestination(t)

	_, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", "tiny", nil)
	if err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("err = %v, want a sha256 mismatch", err)
	}
	dest := filepath.Join(copyAliasDir(store.root, "tiny"), src.list.ExportID, filepath.FromSlash(path))
	if _, err := os.Stat(dest + ".part"); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("mismatched partial kept: %v", err)
	}
	if _, err := os.Stat(dest); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("mismatched file renamed into place: %v", err)
	}
}

func TestCopyModelSkipsFilesAlreadyPresent(t *testing.T) {
	plentyOfDisk(t)
	fixture := namedCustomCopyFixture(t, "tiny", 80_000, 7)
	src := sourceFromExecutor(t, fixture.source, "llamacpp", "tiny")
	dst, store := customDestination(t)
	present := src.list.Files[0].Path
	writeTestFile(t, filepath.Join(copyAliasDir(store.root, "tiny"), src.list.ExportID, filepath.FromSlash(present)), src.data[present])
	res, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", "tiny", nil)
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

func TestCopyModelRefusesWhenDiskSpaceIsShort(t *testing.T) {
	fixture := namedCustomCopyFixture(t, "tiny", 100_000, 9)
	src := sourceFromExecutor(t, fixture.source, "llamacpp", "tiny")
	dstDir := t.TempDir()
	dst := copyTestExecutor(t, "llamacpp", filepath.Join(dstDir, "models"), true)
	prev := diskFreeBytes
	var asked string
	diskFreeBytes = func(path string) (uint64, error) { asked = path; return 1 << 20, nil }
	t.Cleanup(func() { diskFreeBytes = prev })

	_, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", "tiny", nil)
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
	dst := copyTestExecutor(t, "llamacpp", t.TempDir(), false)
	_, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", "tiny", nil)
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
	dst := copyTestExecutor(t, "llamacpp", t.TempDir(), true)
	sum := sha256Hex([]byte("x"))
	for _, test := range []struct {
		files []modelFile
		want  string
	}{
		{[]modelFile{{Path: "../../evil", Size: 1, SHA256: sum}}, "unsafe path"},
		{[]modelFile{{Path: "model-Q4_K_M.gguf", Size: 1, SHA256: sum}, {Path: "README.md", Size: 1, SHA256: sum}}, "unexpected file"},
		{[]modelFile{{Path: "model-Q4_K_M.gguf", Size: 1, SHA256: "nothex"}}, "without a valid sha256"},
	} {
		list := modelFileList{Engine: "llamacpp", Model: "owner/tiny:Q4_K_M", Revision: strings.Repeat("a", 40), Files: test.files}
		src := &fakeModelSource{list: list}
		if _, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", "owner/tiny:Q4_K_M", nil); err == nil || !strings.Contains(err.Error(), test.want) {
			t.Errorf("listing %+v: error = %v, want %s", list.Files, err, test.want)
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
	copyTestNativeModels(t, srcExec, "llamacpp", repo+":Q4_K_M")
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
	copyTestNativeModels(t, dst, "llamacpp", repo+":Q4_K_M")
	back, err := dst.modelSourceFiles(context.Background(), "llamacpp", repo+":Q4_K_M")
	if err != nil || len(back.Files) != 2 {
		t.Fatalf("destination listing = %+v, %v", back, err)
	}
}

func TestModelPublisherResumedAndCachedCopyUsesFullTransferHash(t *testing.T) {
	plentyOfDisk(t)
	src := publisherCopySource(t)
	file := src.list.Files[0]
	catalog, calls := publisherTestCatalog(t, publisherTestBody(file.SHA256), http.StatusOK)
	dst := copyTestExecutor(t, "llamacpp", t.TempDir(), true)
	_, store, err := dst.modelStoreFor("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	partial := filepath.Join(llamaRepoDir(store.root, publisherTestRepo), "snapshots", publisherTestRevision, publisherTestFile) + ".part"
	const prefix = 1024
	writeTestFile(t, partial, src.data[file.Path][:prefix])
	res, err := dst.copyModelFromPublisher(context.Background(), src, "llamacpp", src.list.Model, "", nil, nil, nil, catalog)
	if err != nil {
		t.Fatal(err)
	}
	if res.BytesCopied != file.Size-prefix || res.PublisherComparison == nil || res.PublisherComparison.Status != "verified" || res.PublisherComparison.Files[0].ActualDigest != file.SHA256 {
		t.Fatalf("resumed full comparison=%+v", res)
	}
	res, err = dst.copyModelFromPublisher(context.Background(), src, "llamacpp", src.list.Model, "", nil, nil, nil, catalog)
	if err != nil {
		t.Fatal(err)
	}
	if res.FilesSkipped != res.Files || res.BytesCopied != 0 || calls.Load() != 1 || res.PublisherComparison == nil || res.PublisherComparison.Files[0].AcquisitionBasis != "cache" || res.PublisherComparison.Files[0].ActualDigest != file.SHA256 {
		t.Fatalf("cached full comparison=%+v calls=%d", res, calls.Load())
	}
}

func TestModelPublisherAgreementCannotBypassFullTransferMismatch(t *testing.T) {
	plentyOfDisk(t)
	fastRetries(t, 1)
	src := publisherCopySource(t)
	file := src.list.Files[0]
	catalog, _ := publisherTestCatalog(t, publisherTestBody(file.SHA256), http.StatusOK)
	corrupt := append([]byte(nil), src.data[file.Path]...)
	corrupt[len(corrupt)-1] ^= 1
	src.data[file.Path] = corrupt
	dst := copyTestExecutor(t, "llamacpp", t.TempDir(), true)
	res, err := dst.copyModelFromPublisher(context.Background(), src, "llamacpp", src.list.Model, "", nil, nil, nil, catalog)
	if err == nil || res.PublisherComparison != nil {
		t.Fatalf("provider agreement bypassed hard transfer hash: result=%+v error=%v", res, err)
	}
	_, store, err := dst.modelStoreFor("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(llamaRepoDir(store.root, publisherTestRepo), "refs", "main")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("failed full hash published native visibility: %v", err)
	}
	assertNoPartFiles(t, store.root)
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
		`{"engine":"llamacpp","model":"x"}`:                "node is required",
		`{"node":"n","model":"x"}`:                         "engine is required",
		`{"node":"n","engine":"llamacpp"}`:                 "model is required",
		`{"node":"ghost","engine":"llamacpp","model":"x"}`: "this node is not clustered",
	} {
		out := &lockedBuffer{}
		m := NewManager(NewCodec(out), &Executor{progress: newProgressHub()}, nil, "")
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
