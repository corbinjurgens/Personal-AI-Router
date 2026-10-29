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

	"nvpair-shared/clustertrust"
)

func TestLinkedModelOllamaLinks(t *testing.T) {
	t.Run("selected blob retarget rejects cached and issued evidence", testOllamaLinkedBlobRetarget)
	t.Run("manifest retarget rejects a warm hash cache", testOllamaLinkedManifestHashCache)
	t.Run("unrecognized model does not open linked content", testOllamaLinkedUnrecognized)
	plentyOfDisk(t)
	root := t.TempDir()
	modelRoot := filepath.Join(root, "models")
	fx := writeOllamaModel(t, modelRoot, "tiny", 1024, 3)
	manifest := filepath.Join(modelRoot, filepath.FromSlash(fx.manifestRel))
	manifestDir := filepath.Dir(manifest)
	externalDir := filepath.Join(t.TempDir(), "manifest-dir")
	if err := os.Rename(manifestDir, externalDir); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(externalDir, manifestDir); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	externalManifest := filepath.Join(t.TempDir(), "manifest")
	writeTestFile(t, externalManifest, fx.manifest)
	if err := os.Remove(filepath.Join(externalDir, filepath.Base(manifest))); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(externalManifest, filepath.Join(externalDir, filepath.Base(manifest))); err != nil {
		t.Fatal(err)
	}
	blobs := filepath.Join(modelRoot, "blobs")
	externalBlobs := filepath.Join(t.TempDir(), "blobs")
	if err := os.Rename(blobs, externalBlobs); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(externalBlobs, blobs); err != nil {
		t.Fatal(err)
	}
	for rel, content := range fx.blobs {
		path := filepath.Join(modelRoot, filepath.FromSlash(rel))
		hardTarget := filepath.Join(t.TempDir(), "hard-target")
		writeTestFile(t, hardTarget, content)
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := os.Link(hardTarget, path); err != nil {
			t.Skipf("hardlink unavailable: %v", err)
		}
		break
	}
	unlisted := "blobs/sha256-" + sha256Hex([]byte("unlisted"))
	writeTestFile(t, filepath.Join(modelRoot, filepath.FromSlash(unlisted)), []byte("unlisted"))
	ex := copyTestExecutor(t, "ollama", root, true)
	copyTestNativeModels(t, ex, "ollama", "tiny")
	counts := &linkedHashCounts{}
	access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		counts.opens.Add(1)
		return &linkedHashFile{File: f, counts: counts, forbidden: path == filepath.Join(modelRoot, filepath.FromSlash(unlisted))}, nil
	}}
	peer := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
		mux := (&controlServer{exec: ex, mesh: mesh, access: access}).mux()
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if (r.Method != http.MethodGet && r.Method != http.MethodHead) || (r.URL.Path != controlModelFilesPath && r.URL.Path != controlModelFilePath) {
				http.NotFound(w, r)
				return
			}
			mux.ServeHTTP(w, r)
		})
	})
	ctx := remoteFixtureContext(t)
	client := peer.client(t, ctx)
	list, err := client.modelFiles(ctx, "ollama", "tiny", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(list.Files) != len(fx.blobs)+1 || list.Files[len(list.Files)-1].Path != fx.manifestRel {
		t.Fatalf("ordered files=%v", list.Files)
	}
	for _, file := range list.Files {
		want := fx.blobs[file.Path]
		if file.Path == fx.manifestRel {
			want = fx.manifest
		}
		for _, offset := range []int64{0, 2} {
			body, gotOffset, err := client.modelFile(ctx, "ollama", "tiny", file.Path, list.ExportID, offset)
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
			if gotOffset != offset || !bytes.Equal(got, want[offset:]) {
				t.Fatalf("%s offset=%d bytes=%d", file.Path, gotOffset, len(got))
			}
		}
	}
	query := url.Values{"engine": {"ollama"}, "model": {"tiny"}, "path": {fx.manifestRel}, "exportId": {list.ExportID}}
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, client.base+controlModelFilePath+"?"+query.Encode(), nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.controlClient(false).Do(req)
	if err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK || len(data) != 0 {
		t.Fatalf("HEAD status=%d bytes=%d", resp.StatusCode, len(data))
	}
	query.Set("path", unlisted)
	response := httptest.NewRecorder()
	(&controlServer{exec: ex, access: access}).handleModelFile(response, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+query.Encode(), nil))
	if response.Code != http.StatusNotFound || counts.forbiddenBytes.Load() != 0 {
		t.Fatalf("unlisted status=%d reads=%d", response.Code, counts.forbiddenBytes.Load())
	}
	replacement := filepath.Join(t.TempDir(), "manifest")
	writeTestFile(t, replacement, fx.manifest)
	if err := os.Remove(manifest); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(replacement, manifest); err != nil {
		t.Fatal(err)
	}
	query.Set("path", fx.manifestRel)
	before := counts.bytes.Load()
	response = httptest.NewRecorder()
	(&controlServer{exec: ex, access: access}).handleModelFile(response, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+query.Encode(), nil))
	if response.Code != http.StatusConflict || counts.bytes.Load() != before {
		t.Fatalf("retarget status=%d reads=%d", response.Code, counts.bytes.Load()-before)
	}
	if _, err := client.modelFiles(ctx, "ollama", "tiny", nil); err != nil {
		t.Fatal(err)
	}
	destinationDir := t.TempDir()
	destination := copyTestExecutor(t, "ollama", destinationDir, true)
	result, err := destination.CopyModelFrom(context.Background(), client, "ollama", "tiny", nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Files != len(list.Files) || result.ExportID != "" || result.Layout != "" {
		t.Fatalf("native copy result=%+v", result)
	}
	for rel, want := range fx.blobs {
		got, err := os.ReadFile(filepath.Join(destinationDir, "models", filepath.FromSlash(rel)))
		if err != nil || !bytes.Equal(got, want) {
			t.Fatalf("copied %s differs: %v", rel, err)
		}
	}
	got, err := os.ReadFile(filepath.Join(destinationDir, "models", filepath.FromSlash(fx.manifestRel)))
	if err != nil || !bytes.Equal(got, fx.manifest) {
		t.Fatalf("copied manifest differs: %v", err)
	}
	if counts.forbiddenBytes.Load() != 0 {
		t.Fatalf("unlisted bytes read=%d", counts.forbiddenBytes.Load())
	}
}

func testOllamaLinkedBlobRetarget(t *testing.T) {
	root := t.TempDir()
	modelRoot := filepath.Join(root, "models")
	fx := writeOllamaModel(t, modelRoot, "tiny", 1024, 3)
	var selected string
	var content []byte
	for rel, data := range fx.blobs {
		selected, content = rel, data
		break
	}
	path := filepath.Join(modelRoot, filepath.FromSlash(selected))
	first := filepath.Join(t.TempDir(), "first")
	second := filepath.Join(t.TempDir(), "second")
	writeTestFile(t, first, content)
	writeTestFile(t, second, content)
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(first, path); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	ex := copyTestExecutor(t, "ollama", root, true)
	copyTestNativeModels(t, ex, "ollama", "tiny")
	list := copyTestSourceListing(t, ex, "ollama", "tiny")
	bound, err := ex.modelSourceFiles(context.Background(), "ollama", "tiny")
	if err != nil {
		t.Fatal(err)
	}
	if err := fileHashes.hashModelFilesAccess(context.Background(), &bound, nil, servingFileAccess{}); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(second, path); err != nil {
		t.Fatal(err)
	}
	counts := &linkedHashCounts{}
	access := servingFileAccess{open: func(name string) (servingArtifactFile, error) {
		f, err := os.Open(name)
		if err != nil {
			return nil, err
		}
		return &linkedHashFile{File: f, counts: counts, forbidden: true}, nil
	}}
	if err := fileHashes.hashModelFilesAccess(context.Background(), &bound, nil, access); err == nil || counts.bytes.Load() != 0 {
		t.Fatalf("asserted digest stale err=%v bytes=%d", err, counts.bytes.Load())
	}
	query := url.Values{"engine": {"ollama"}, "model": {"tiny"}, "path": {selected}, "exportId": {list.ExportID}}
	response := httptest.NewRecorder()
	(&controlServer{exec: ex, access: access}).handleModelFile(response, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+query.Encode(), nil))
	if response.Code != http.StatusConflict || counts.bytes.Load() != 0 {
		t.Fatalf("stale issued status=%d reads=%d", response.Code, counts.bytes.Load())
	}
	fresh := copyTestSourceListing(t, ex, "ollama", "tiny")
	if len(fresh.Files) != len(list.Files) {
		t.Fatalf("fresh files=%v", fresh.Files)
	}
	_, client := modelFileServer(t, ex, nil)
	for _, offset := range []int64{0, 2} {
		body, gotOffset, err := client.modelFile(context.Background(), "ollama", "tiny", selected, fresh.ExportID, offset)
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
		if gotOffset != offset || !bytes.Equal(got, content[offset:]) {
			t.Fatalf("fresh selected offset=%d bytes=%d", gotOffset, len(got))
		}
	}
	query.Set("exportId", fresh.ExportID)
	response = httptest.NewRecorder()
	(&controlServer{exec: ex}).handleModelFile(response, httptest.NewRequest(http.MethodHead, controlModelFilePath+"?"+query.Encode(), nil))
	if response.Code != http.StatusOK || response.Body.Len() != 0 {
		t.Fatalf("fresh HEAD status=%d bytes=%d", response.Code, response.Body.Len())
	}
	destDir := t.TempDir()
	dest := copyTestExecutor(t, "ollama", destDir, true)
	if _, err := dest.CopyModelFrom(context.Background(), client, "ollama", "tiny", nil); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(destDir, "models", filepath.FromSlash(selected)))
	if err != nil || !bytes.Equal(got, content) {
		t.Fatalf("fresh copied blob differs: %v", err)
	}
}

func testOllamaLinkedManifestHashCache(t *testing.T) {
	root := t.TempDir()
	fx := writeOllamaModel(t, filepath.Join(root, "models"), "tiny", 128, 3)
	path := filepath.Join(root, "models", filepath.FromSlash(fx.manifestRel))
	first, second := filepath.Join(t.TempDir(), "first"), filepath.Join(t.TempDir(), "second")
	writeTestFile(t, first, fx.manifest)
	writeTestFile(t, second, fx.manifest)
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(first, path); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	ex := copyTestExecutor(t, "ollama", root, true)
	copyTestNativeModels(t, ex, "ollama", "tiny")
	old, err := ex.modelSourceFiles(context.Background(), "ollama", "tiny")
	if err != nil {
		t.Fatal(err)
	}
	warm := old
	warm.Files = append([]modelFile(nil), old.Files...)
	if err := fileHashes.hashModelFilesAccess(context.Background(), &warm, nil, servingFileAccess{}); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(second, path); err != nil {
		t.Fatal(err)
	}
	counts := &linkedHashCounts{}
	access := servingFileAccess{open: func(name string) (servingArtifactFile, error) {
		file, err := os.Open(name)
		if err != nil {
			return nil, err
		}
		return &linkedHashFile{File: file, counts: counts, forbidden: true}, nil
	}}
	if err := fileHashes.hashModelFilesAccess(context.Background(), &old, nil, access); err == nil || counts.bytes.Load() != 0 {
		t.Fatalf("warm cache stale err=%v reads=%d", err, counts.bytes.Load())
	}
}

func testOllamaLinkedUnrecognized(t *testing.T) {
	root := t.TempDir()
	fx := writeOllamaModel(t, filepath.Join(root, "models"), "tiny", 128, 3)
	ex := copyTestExecutor(t, "ollama", root, true)
	copyTestNativeModels(t, ex, "ollama", "other")
	counts := &linkedHashCounts{}
	access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		return &linkedHashFile{File: file, counts: counts, forbidden: true}, nil
	}}
	list, err := ex.modelSourceFilesWithAccess(context.Background(), "ollama", "tiny", access)
	if err == nil || len(list.Files) != 0 || counts.bytes.Load() != 0 {
		t.Fatalf("unrecognized files=%v err=%v reads=%d manifest=%s", list.Files, err, counts.bytes.Load(), fx.manifestRel)
	}
}
