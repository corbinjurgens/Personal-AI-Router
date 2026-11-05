// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func nativeBuildTestRuntime(t *testing.T, st *engineState, root string) {
	t.Helper()
	st.binPath = filepath.Join(root, "synthetic-native")
	st.installDir = root
	st.plat.Runtime.Bin = st.binPath
	writeTestFile(t, st.binPath, []byte("synthetic executable, never run"))
	writeTestFile(t, filepath.Join(root, "libggml-base.dylib"), []byte("synthetic library, never loaded"))
}

func nativeBuildTestState(t *testing.T) *engineState {
	t.Helper()
	st := &engineState{manifest: &Manifest{Engine: "llamacpp"}, plat: &Platform{}}
	nativeBuildTestRuntime(t, st, t.TempDir())
	return st
}

func requireNativeBuildID(t *testing.T, st *engineState) string {
	t.Helper()
	id, err := nativeBuildID(context.Background(), st)
	if err != nil {
		t.Fatal(err)
	}
	if id == "" {
		t.Fatal("native build identity is empty")
	}
	return id
}

func TestNativeBuildContentFingerprint(t *testing.T) {
	first, second := nativeBuildTestState(t), nativeBuildTestState(t)
	want := requireNativeBuildID(t, first)
	if got := requireNativeBuildID(t, second); got != want {
		t.Fatalf("identical builds in different directories: %q != %q", got, want)
	}
	executable := sha256.Sum256([]byte("synthetic executable, never run"))
	library := sha256.Sum256([]byte("synthetic library, never loaded"))
	// Spell the public fingerprint input independently of the implementation.
	input := fmt.Sprintf(`{"version":"pair-native-build-v1","platform":"%s/%s","executable":"%x","libraries":[{"name":"libggml-base.dylib","sha256":"%x"}]}`, runtime.GOOS, runtime.GOARCH, executable, library)
	sum := sha256.Sum256([]byte(input))
	if want != "sha256:"+hex.EncodeToString(sum[:]) {
		t.Fatalf("fingerprint = %q", want)
	}
}

func TestNativeBuildBytesDecideIdentity(t *testing.T) {
	test := func(name string, mutate func(*testing.T, *engineState), changes bool) {
		t.Run(name, func(t *testing.T) {
			st := nativeBuildTestState(t)
			before := requireNativeBuildID(t, st)
			mutate(t, st)
			after := requireNativeBuildID(t, st)
			if (after != before) != changes {
				t.Fatalf("identity changed=%v, want %v", after != before, changes)
			}
		})
	}
	test("library bytes", func(t *testing.T, st *engineState) {
		writeTestFile(t, filepath.Join(st.installDir, "libggml-base.dylib"), []byte("different native library bytes"))
	}, true)
	test("executable bytes", func(t *testing.T, st *engineState) {
		writeTestFile(t, st.binPath, []byte("different executable bytes"))
	}, true)
	test("executable and library mtime", func(t *testing.T, st *engineState) {
		for _, path := range []string{st.binPath, filepath.Join(st.installDir, "libggml-base.dylib")} {
			info, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			touch := info.ModTime().Add(2 * time.Hour)
			if err := os.Chtimes(path, touch, touch); err != nil {
				t.Fatal(err)
			}
		}
	}, false)
	test("moved build", func(t *testing.T, st *engineState) {
		root := filepath.Join(t.TempDir(), "moved")
		if err := os.Rename(st.installDir, root); err != nil {
			t.Fatal(err)
		}
		st.installDir, st.binPath = root, filepath.Join(root, filepath.Base(st.binPath))
		st.plat.Runtime.Bin = st.binPath
	}, false)
	test("unaccepted library name", func(t *testing.T, st *engineState) {
		writeTestFile(t, filepath.Join(st.installDir, "unrelated.dll"), []byte("not part of this build"))
	}, false)
	test("added accepted library", func(t *testing.T, st *engineState) {
		writeTestFile(t, filepath.Join(st.installDir, "libllama.dylib"), []byte("additional native library"))
	}, true)
}

func TestNativeBuildMemoRejectsReplacedObject(t *testing.T) {
	st := nativeBuildTestState(t)
	before := requireNativeBuildID(t, st)
	path := filepath.Join(st.installDir, "libggml-base.dylib")
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	replacement := filepath.Join(st.installDir, "replacement")
	writeTestFile(t, replacement, []byte("replacement library of same len"))
	replaced, err := os.Stat(replacement)
	if err != nil {
		t.Fatal(err)
	}
	if replaced.Size() != info.Size() {
		t.Fatalf("fixture sizes differ: %d != %d", replaced.Size(), info.Size())
	}
	if err := os.Chtimes(replacement, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(replacement, path); err != nil {
		t.Fatal(err)
	}
	if after := requireNativeBuildID(t, st); after == before {
		t.Fatal("memo reused a replaced object's hash")
	}
}

func TestNativeBuildExpectedExecutableRejectsSubstitutionBeforeRead(t *testing.T) {
	st := nativeBuildTestState(t)
	requireNativeBuildID(t, st)
	executable, err := captureSource(st.binPath, false)
	if err != nil {
		t.Fatal(err)
	}
	replacement := filepath.Join(t.TempDir(), "replacement")
	writeTestFile(t, replacement, []byte("replacement executable"))
	reads, closes := 0, 0
	access := servingFileAccess{open: func(string) (servingArtifactFile, error) {
		file, err := os.Open(replacement)
		if err != nil {
			return nil, err
		}
		return &countedLinkedFile{File: file, reads: &reads, closes: &closes}, nil
	}}
	if _, _, err := nativeBuildSnapshotExpected(context.Background(), executable, access, nil); err == nil {
		t.Fatal("substituted executable accepted")
	}
	if reads != 0 || closes != 1 {
		t.Fatalf("substitution reads=%d closes=%d", reads, closes)
	}
}

func TestNativeBuildCanceledMemoRead(t *testing.T) {
	st := nativeBuildTestState(t)
	requireNativeBuildID(t, st)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := nativeBuildID(ctx, st); err == nil {
		t.Fatal("canceled memo read succeeded")
	}
}

func TestNativeBuildMemoSkipsUnchangedContentRead(t *testing.T) {
	st := nativeBuildTestState(t)
	executable, err := captureSource(st.binPath, false)
	if err != nil {
		t.Fatal(err)
	}
	reads, closes := 0, 0
	access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		return &countedLinkedFile{File: file, reads: &reads, closes: &closes}, nil
	}}
	first, _, err := nativeBuildSnapshotExpected(context.Background(), executable, access, nil)
	if err != nil || reads == 0 {
		t.Fatalf("initial hash reads=%d error=%v", reads, err)
	}
	reads, closes = 0, 0
	second, _, err := nativeBuildSnapshotExpected(context.Background(), executable, access, nil)
	if err != nil || second.Build != first.Build || reads != 0 || closes != 2 {
		t.Fatalf("memo read build=%q reads=%d closes=%d error=%v", second.Build, reads, closes, err)
	}
}
