// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func writeServingFixture(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestServingFileBindingSamplesBothEnds(t *testing.T) {
	path := filepath.Join(t.TempDir(), "synthetic.gguf")
	data := bytes.Repeat([]byte("a"), 3*servingSampleSize)
	writeServingFixture(t, path, data)
	before, err := servingFileBinding(path)
	if err != nil {
		t.Fatal(err)
	}
	stat, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)-1] = 'b'
	writeServingFixture(t, path, data)
	if err := os.Chtimes(path, stat.ModTime(), stat.ModTime()); err != nil {
		t.Fatal(err)
	}
	after, err := servingFileBinding(path)
	if err != nil {
		t.Fatal(err)
	}
	if before.SizeBytes != after.SizeBytes || before.MtimeNS != after.MtimeNS || before.SampleSHA256 == after.SampleSHA256 {
		t.Fatal("same-stat trailing sample mutation was not detected")
	}
}

func TestServingFileBindingKeepsHFSymlink(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "blobs"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(root, "snapshots"), 0o700); err != nil {
		t.Fatal(err)
	}
	blob := filepath.Join(root, "blobs", "synthetic-blob")
	link := filepath.Join(root, "snapshots", "model.gguf")
	data := []byte("synthetic HF model bytes")
	writeServingFixture(t, blob, data)
	if err := os.Symlink(filepath.Join("..", "blobs", "synthetic-blob"), link); err != nil {
		t.Skipf("symlink creation unavailable: %v", err)
	}
	binding, err := servingFileBinding(link)
	if err != nil {
		t.Fatal(err)
	}
	if binding.SourcePath != link || binding.ResolvedTarget != blob || binding.SizeBytes != int64(len(data)) {
		t.Fatal("HF binding did not preserve the source symlink and resolved blob")
	}
	info, err := os.Lstat(link)
	if err != nil || info.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("source is no longer a symlink: %v", err)
	}
	got, err := os.ReadFile(blob)
	if err != nil || !bytes.Equal(got, data) {
		t.Fatalf("source blob changed: %v", err)
	}
	entries, err := os.ReadDir(filepath.Join(root, "blobs"))
	if err != nil || len(entries) != 1 {
		t.Fatalf("binding created model copies: %v", err)
	}
}

func TestServingSnapshotRecoveryProducesIdenticalBytes(t *testing.T) {
	root := t.TempDir()
	data := []byte("[*]\nctx-size = 8192\nparallel = 1\n")
	first, releaseFirst, err := prepareServingSnapshot(root, data)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(releaseFirst)
	second, releaseSecond, err := prepareServingSnapshot(root, data)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(releaseSecond)
	if first == second {
		t.Fatal("process generations share a mutable snapshot path")
	}
	for _, path := range []string{first, second} {
		got, err := os.ReadFile(path)
		if err != nil || !bytes.Equal(got, data) {
			t.Fatalf("recovery changed snapshot bytes: %v", err)
		}
	}
	releaseFirst()
	if _, err := os.Stat(second); err != nil {
		t.Fatal("releasing one generation deleted another")
	}
}

func TestServingSourceRejectsOversizedFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "oversized.ini")
	writeServingFixture(t, path, bytes.Repeat([]byte("x"), servingSmallFileLimit+1))
	if _, _, err := servingSource(path); err == nil {
		t.Fatal("oversized launch source accepted")
	}
}
