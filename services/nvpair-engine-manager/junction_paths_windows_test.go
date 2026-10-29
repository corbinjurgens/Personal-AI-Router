// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/sys/windows"
)

func TestConfinedRegularFileResolvesModelStoreJunction(t *testing.T) {
	base := t.TempDir()
	store := filepath.Join(base, "store")
	file := filepath.Join(store, "model.gguf")
	writeTestFile(t, file, []byte("model"))
	link := filepath.Join(base, "models")
	junction(t, link, store)

	got, _, err := confinedRegularFile(link, filepath.Join(link, "model.gguf"))
	if err != nil {
		t.Fatalf("confinedRegularFile: %v", err)
	}
	want, err := resolveExistingPath(file)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("resolved file = %q, want %q", got, want)
	}
}

func TestConfinedRegularFileRejectsOutsideJunction(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "store")
	outside := filepath.Join(base, "outside")
	writeTestFile(t, filepath.Join(outside, "model.gguf"), []byte("model"))
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "escape")
	junction(t, link, outside)
	if _, _, err := confinedRegularFile(root, filepath.Join(link, "model.gguf")); err == nil {
		t.Fatal("accepted a model file outside the store")
	}
}

func TestHashModelFilesAcceptsJunctionSourceBinding(t *testing.T) {
	base := t.TempDir()
	file := filepath.Join(base, "store", "model.gguf")
	writeTestFile(t, file, []byte("model"))
	link := filepath.Join(base, "models")
	junction(t, link, filepath.Dir(file))
	resolved, err := resolveExistingPath(file)
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(file)
	if err != nil {
		t.Fatal(err)
	}
	list := modelFileList{Files: []modelFile{{Path: "model.gguf", Size: info.Size(), abs: resolved, configured: filepath.Join(link, "model.gguf"), info: info}}}
	if err := hashModelFiles(context.Background(), &list, func(string, int) {}); err != nil {
		t.Fatalf("hashModelFiles: %v", err)
	}
	if list.Files[0].SHA256 != sha256Hex([]byte("model")) {
		t.Fatalf("unexpected digest %q", list.Files[0].SHA256)
	}
}

func TestLlamaCacheThroughJunction(t *testing.T) {
	base := t.TempDir()
	cache := filepath.Join(base, "cache")
	const repo = "owner/model"
	const commit = "0123456789abcdef0123456789abcdef01234567"
	writeLlamaCache(t, cache, repo, commit, map[string][]byte{"model-Q4_K_M.gguf": []byte("model")})
	link := filepath.Join(base, "cache-link")
	junction(t, link, cache)
	list, err := llamacppSourceFiles(link, repo+":Q4_K_M")
	if err != nil {
		t.Fatalf("llamacppSourceFiles: %v", err)
	}
	if list.Revision != commit || len(list.Files) != 1 || list.Files[0].Path != "model-Q4_K_M.gguf" {
		t.Fatalf("unexpected cache listing: %+v", list)
	}
}

func TestLlamaCacheSnapshotAndBlobJunctions(t *testing.T) {
	base := t.TempDir()
	const repo = "owner/model"
	const commit = "0123456789abcdef0123456789abcdef01234567"
	repoDir := llamaRepoDir(base, repo)
	data := []byte("model")
	blobDir := filepath.Join(repoDir, "actual-blobs")
	name := "model-Q4_K_M.gguf"
	writeTestFile(t, filepath.Join(blobDir, name), data)
	writeTestFile(t, filepath.Join(repoDir, "refs", "main"), []byte(commit))
	if err := os.MkdirAll(filepath.Join(repoDir, "snapshots"), 0o755); err != nil {
		t.Fatal(err)
	}
	junction(t, filepath.Join(repoDir, "blobs"), blobDir)
	junction(t, filepath.Join(repoDir, "snapshots", commit), filepath.Join(repoDir, "blobs"))
	list, err := boundedLlamaCacheArtifacts(context.Background(), engineModelStore{root: base, adapter: "llamacpp"}, repo+":Q4_K_M", servingFileAccess{})
	if err != nil {
		t.Fatalf("boundedLlamaCacheArtifacts: %v", err)
	}
	if len(list.Files) != 1 || list.Files[0].Path != name {
		t.Fatalf("unexpected artifacts: %+v", list.Files)
	}
}

func TestLlamaCacheRecognizesSHA256BlobThroughJunction(t *testing.T) {
	base := t.TempDir()
	const repo = "owner/model"
	const commit = "0123456789abcdef0123456789abcdef01234567"
	repoDir := llamaRepoDir(base, repo)
	data := []byte("model")
	digest := sha256Hex(data)
	blobDir := filepath.Join(repoDir, "actual-blobs")
	writeTestFile(t, filepath.Join(blobDir, digest), data)
	writeTestFile(t, filepath.Join(repoDir, "refs", "main"), []byte(commit))
	snapshot := filepath.Join(repoDir, "snapshots", commit)
	if err := os.MkdirAll(snapshot, 0o755); err != nil {
		t.Fatal(err)
	}
	junction(t, filepath.Join(repoDir, "blobs"), blobDir)
	if err := os.Symlink(filepath.Join(repoDir, "blobs", digest), filepath.Join(snapshot, "model-Q4_K_M.gguf")); err != nil {
		if errors.Is(err, windows.ERROR_PRIVILEGE_NOT_HELD) {
			t.Skip("snapshot blob symlink requires Windows symlink privileges or Developer Mode")
		}
		t.Fatalf("create snapshot blob link: %v", err)
	}
	list, err := llamacppSourceFiles(base, repo+":Q4_K_M")
	if err != nil {
		t.Fatalf("llamacppSourceFiles: %v", err)
	}
	if len(list.Files) != 1 || list.Files[0].SHA256 != digest {
		t.Fatalf("blob digest was not recognized: %+v", list.Files)
	}
}

func TestModelFileHTTPAcceptsJunctionSelectedBinding(t *testing.T) {
	base := t.TempDir()
	const repo = "owner/model"
	const commit = "0123456789abcdef0123456789abcdef01234567"
	repoDir := llamaRepoDir(base, repo)
	data := []byte("model")
	name := "model-Q4_K_M.gguf"
	writeTestFile(t, filepath.Join(repoDir, "blobs", name), data)
	writeTestFile(t, filepath.Join(repoDir, "refs", "main"), []byte(commit))
	if err := os.MkdirAll(filepath.Join(repoDir, "snapshots"), 0o755); err != nil {
		t.Fatal(err)
	}
	junction(t, filepath.Join(repoDir, "snapshots", commit), filepath.Join(repoDir, "blobs"))
	ex := copyTestExecutor(t, "llamacpp", base, true)
	q := url.Values{"engine": {"llamacpp"}, "model": {repo + ":Q4_K_M"}, "path": {name}}
	w := httptest.NewRecorder()
	(&controlServer{exec: ex}).handleModelFile(w, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+q.Encode(), nil))
	if w.Code != http.StatusOK || w.Body.String() != string(data) {
		t.Fatalf("response: status=%d body=%q", w.Code, w.Body.String())
	}
}

func TestStrataTargetResolvesModelJunction(t *testing.T) {
	base := t.TempDir()
	file := filepath.Join(base, "store", "model.gguf")
	writeTestFile(t, file, []byte("model"))
	link := filepath.Join(base, "models")
	junction(t, link, filepath.Dir(file))
	got, err := strataTarget(filepath.Join(link, "model.gguf"))
	if err != nil {
		t.Fatalf("strataTarget: %v", err)
	}
	want, err := resolveExistingPath(file)
	if err != nil {
		t.Fatal(err)
	}
	if got != strings.ToLower(want) {
		t.Fatalf("target = %q, want %q", got, strings.ToLower(want))
	}
}

func TestSafeRemoveUnderRootRejectsOutsideJunction(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "root")
	outside := filepath.Join(base, "outside")
	keep := victimDir(t, outside)
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "escape")
	junction(t, link, outside)
	if err := safeRemoveUnderRoot(root, filepath.Join(link, filepath.Base(keep))); err == nil {
		t.Error("accepted removal through an outside junction")
	}
	if _, err := os.Stat(keep); err != nil {
		t.Fatalf("outside file did not survive: %v", err)
	}
}

func TestSafeRemoveUnderRootUnlinksOutsideJunction(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "root")
	outside := filepath.Join(base, "outside")
	keep := victimDir(t, outside)
	if err := os.Mkdir(root, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "linked")
	junction(t, link, outside)
	if err := safeRemoveUnderRoot(root, link); err != nil {
		t.Fatalf("safeRemoveUnderRoot: %v", err)
	}
	if _, err := os.Lstat(link); !os.IsNotExist(err) {
		t.Fatalf("junction remains or lstat failed: %v", err)
	}
	data, err := os.ReadFile(keep)
	if err != nil || string(data) != "x" {
		t.Fatalf("outside file = %q, err = %v", data, err)
	}
}

func TestJunctionRemovalBehavior(t *testing.T) {
	for _, operation := range []struct {
		name   string
		remove func(string) error
	}{
		{name: "Remove", remove: os.Remove},
		{name: "RemoveAll", remove: os.RemoveAll},
	} {
		t.Run(operation.name, func(t *testing.T) {
			base := t.TempDir()
			outside := filepath.Join(base, "outside")
			keep := victimDir(t, outside)
			link := filepath.Join(base, "linked")
			junction(t, link, outside)
			info, err := os.Lstat(link)
			if err != nil {
				t.Fatalf("lstat junction: %v", err)
			}
			t.Logf("junction Lstat mode: %v (%#x)", info.Mode(), uint32(info.Mode()))
			if err := operation.remove(link); err != nil {
				t.Fatalf("%s junction: %v", operation.name, err)
			}
			if _, err := os.Lstat(link); !os.IsNotExist(err) {
				t.Fatalf("junction remains or lstat failed: %v", err)
			}
			data, err := os.ReadFile(keep)
			if err != nil || string(data) != "x" {
				t.Fatalf("outside file = %q, err = %v", data, err)
			}
		})
	}
}

func TestSafeRemoveUnderRootAcceptsJunctionRoot(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "root")
	file := filepath.Join(root, "remove.txt")
	writeTestFile(t, file, []byte("remove"))
	link := filepath.Join(base, "root-link")
	junction(t, link, root)
	if err := safeRemoveUnderRoot(link, filepath.Join(link, "remove.txt")); err != nil {
		t.Fatalf("safeRemoveUnderRoot: %v", err)
	}
	if _, err := os.Stat(file); !os.IsNotExist(err) {
		t.Fatalf("file still exists or stat failed unexpectedly: %v", err)
	}
}

func TestSafeRemoveUnderRootRejectsUnresolvableJunction(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "root")
	target := filepath.Join(base, "target")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(target, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "broken")
	junction(t, link, target)
	if err := os.Remove(target); err != nil {
		t.Fatal(err)
	}
	if err := safeRemoveUnderRoot(root, link); err == nil {
		t.Fatal("accepted an unresolvable junction")
	}
	if _, err := os.Lstat(link); err != nil {
		t.Fatalf("unresolvable junction did not survive: %v", err)
	}
}

func TestDirectoriesAboveIncludesResolvedJunctionAncestors(t *testing.T) {
	base := t.TempDir()
	store := filepath.Join(base, "install", "data", "models")
	if err := os.MkdirAll(store, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(base, "models")
	junction(t, link, store)
	install, err := os.Stat(filepath.Join(base, "install"))
	if err != nil {
		t.Fatal(err)
	}
	for _, dir := range directoriesAbove(link) {
		if os.SameFile(dir, install) {
			return
		}
	}
	t.Fatal("resolved install ancestor was not included")
}
