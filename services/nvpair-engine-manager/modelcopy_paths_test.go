// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"net"
	"os"
	"path/filepath"
	"runtime"
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
	data := []byte("model")
	src := &fakeModelSource{data: map[string][]byte{"file": data}}
	src.onFetch = func(string) {
		if err := os.Rename(parent, filepath.Join(root, "original")); err != nil {
			t.Fatalf("move parent: %v", err)
		}
		if err := os.Symlink(outside, parent); err != nil {
			t.Fatalf("replace parent: %v", err)
		}
	}
	f := plannedFile{modelFile: modelFile{Path: "file", Size: int64(len(data)), SHA256: sha256Hex(data)}, root: root, dest: filepath.Join(parent, "file")}
	if _, _, err := downloadPlannedFile(context.Background(), src, "llamacpp", "model", f, func(int64) {}); err == nil {
		t.Fatal("accepted replaced parent")
	}
	got, err := os.ReadFile(filepath.Join(outside, "sentinel"))
	if err != nil || string(got) != "keep" {
		t.Fatalf("sentinel = %q, error = %v", got, err)
	}
	if _, err := os.Stat(filepath.Join(outside, "file")); !os.IsNotExist(err) {
		t.Fatalf("outside destination exists or stat failed: %v", err)
	}
}
