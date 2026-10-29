// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
)

func TestModelCopyDestinationLinks(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "store")
	outside := filepath.Join(base, "outside")
	writeTestFile(t, filepath.Join(outside, "sentinel"), []byte("keep"))
	writeTestFile(t, filepath.Join(root, "ordinary", "file"), []byte("existing"))
	rootLink := filepath.Join(base, "root-link")
	if err := os.Symlink(root, rootLink); err != nil {
		t.Fatalf("link root: %v", err)
	}
	for name, target := range map[string]string{
		"inside": filepath.Join(root, "ordinary"), "escape": outside,
		"broken": filepath.Join(base, "missing"), "cycle": filepath.Join(root, "cycle"),
	} {
		if err := os.Symlink(target, filepath.Join(root, name)); err != nil {
			t.Fatalf("link %s: %v", name, err)
		}
	}
	for _, tc := range []struct {
		name      string
		root      string
		dest      string
		wantError bool
	}{
		{"ordinary file", root, filepath.Join(root, "ordinary", "file"), false},
		{"trusted root alias", rootLink, filepath.Join(rootLink, "new", "file"), false},
		{"final directory", root, filepath.Join(root, "ordinary"), true},
		{"child link inside", root, filepath.Join(root, "inside", "file"), true},
		{"child link outside part", root, filepath.Join(root, "escape", "file.part"), true},
		{"child link outside receipt", root, filepath.Join(root, "escape", "receipt.json"), true},
		{"case collision", root, filepath.Join(root, "Ordinary", "file"), true},
		{"broken child link", root, filepath.Join(root, "broken", "file"), true},
		{"cyclic child link", root, filepath.Join(root, "cycle", "file"), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := safeCopyDestination(tc.root, tc.dest)
			if (err != nil) != tc.wantError {
				t.Fatalf("safeCopyDestination error = %v, want error %v", err, tc.wantError)
			}
		})
	}
	data, err := os.ReadFile(filepath.Join(outside, "sentinel"))
	if err != nil || string(data) != "keep" {
		t.Fatalf("sentinel = %q, error = %v", data, err)
	}
	t.Run("replacement after preflight", testModelCopyDestinationReplacementAfterPreflight)
	t.Run("receipt replacement after preflight", testModelCopyReceiptReplacementAfterPreflight)
	t.Run("special file", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("Unix domain socket fixture")
		}
		shortRoot, err := os.MkdirTemp("/tmp", "pair-copy-")
		if err != nil {
			t.Fatalf("create socket root: %v", err)
		}
		t.Cleanup(func() { _ = os.RemoveAll(shortRoot) })
		path := filepath.Join(shortRoot, "socket")
		listener, err := net.Listen("unix", path)
		if err != nil {
			t.Fatalf("listen on socket: %v", err)
		}
		defer listener.Close()
		if err := safeCopyDestination(shortRoot, path); err == nil {
			t.Fatal("accepted socket destination")
		}
	})
}

func testModelCopyReceiptReplacementAfterPreflight(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	data := []byte("model")
	list := modelFileList{Engine: "llamacpp", Model: "alias", Layout: llamaArtifactsLayout, Files: []modelFile{{Role: "model", Path: "model/file.gguf", Size: int64(len(data)), SHA256: sha256Hex(data)}}}
	id, err := artifactExportID(list)
	if err != nil {
		t.Fatal(err)
	}
	list.ExportID = id
	plan, err := customArtifactPlan(engineModelStore{engine: "llamacpp", root: root}, list.Model, list)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, plan.files[0].dest, data)
	sentinel := filepath.Join(outside, "receipt")
	writeTestFile(t, sentinel, []byte("keep"))
	if err := os.Symlink(sentinel, filepath.Join(copyAliasDir(root, list.Model), "copy.json")); err != nil {
		t.Fatalf("link receipt: %v", err)
	}
	if err := plan.finalize(context.Background()); err == nil {
		t.Fatal("accepted linked receipt after preflight")
	}
	got, err := os.ReadFile(sentinel)
	if err != nil || string(got) != "keep" {
		t.Fatalf("sentinel = %q, error = %v", got, err)
	}
}

func testModelCopyDestinationReplacementAfterPreflight(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "store")
	parent := filepath.Join(root, "model")
	outside := filepath.Join(base, "outside")
	writeTestFile(t, filepath.Join(parent, "existing"), []byte("original"))
	writeTestFile(t, filepath.Join(outside, "sentinel"), []byte("keep"))
	parentInfo, err := os.Stat(parent)
	if err != nil {
		t.Fatalf("stat parent: %v", err)
	}
	if !os.SameFile(parentInfo, parentInfo) {
		t.Fatal("parent identity is unavailable")
	}
	data := []byte("model")
	src := &fakeModelSource{data: map[string][]byte{"file": data}}
	called := false
	var renameErr error
	src.onFetch = func(string) {
		called = true
		renameErr = os.Rename(parent, filepath.Join(root, "original"))
		if renameErr != nil {
			if runtime.GOOS != "windows" || !(errors.Is(renameErr, syscall.Errno(5)) || errors.Is(renameErr, syscall.Errno(32)) || errors.Is(renameErr, syscall.Errno(33))) {
				t.Fatalf("move parent: %v", renameErr)
			}
			return
		}
		if err := os.Symlink(outside, parent); err != nil {
			t.Fatalf("replace parent: %v", err)
		}
	}
	f := plannedFile{modelFile: modelFile{Path: "file", Size: int64(len(data)), SHA256: sha256Hex(data)}, root: root, dest: filepath.Join(parent, "file")}
	_, _, copyErr := downloadPlannedFile(context.Background(), src, "llamacpp", "model", f, func(int64) {})
	if !called {
		t.Fatal("fetch hook was not called")
	}
	if renameErr == nil && copyErr == nil {
		t.Fatal("accepted replaced parent")
	}
	if renameErr != nil {
		t.Logf("Windows prevented parent replacement: %v; copy result: %v", renameErr, copyErr)
		currentInfo, err := os.Stat(parent)
		if err != nil || !os.SameFile(parentInfo, currentInfo) {
			t.Fatalf("parent changed after refused rename: %v", err)
		}
		existing, err := os.ReadFile(filepath.Join(parent, "existing"))
		if err != nil || string(existing) != "original" {
			t.Fatalf("existing = %q, error = %v", existing, err)
		}
		if copyErr == nil {
			copied, err := os.ReadFile(f.dest)
			if err != nil || sha256Hex(copied) != sha256Hex(data) {
				t.Fatalf("destination digest mismatch, error = %v", err)
			}
		}
	}
	got, err := os.ReadFile(filepath.Join(outside, "sentinel"))
	if err != nil || string(got) != "keep" {
		t.Fatalf("sentinel = %q, error = %v", got, err)
	}
	for _, name := range []string{"file", "file.part"} {
		if _, err := os.Lstat(filepath.Join(outside, name)); !os.IsNotExist(err) {
			t.Fatalf("outside %s exists or lstat failed: %v", name, err)
		}
	}
}
