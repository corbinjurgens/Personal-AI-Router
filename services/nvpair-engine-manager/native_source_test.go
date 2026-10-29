// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func nativeSourceDirectoryLink(t *testing.T, link, target string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		output, err := exec.Command("cmd.exe", "/d", "/c", "mklink", "/J", link, target).CombinedOutput()
		if err != nil {
			t.Skipf("junction creation unavailable: %v: %s", err, output)
		}
	} else if err := os.Symlink(target, link); err != nil {
		t.Skipf("directory symlink creation unavailable: %v", err)
	}
}

func TestNativeSourceFinalPathThroughJunction(t *testing.T) {
	target := t.TempDir()
	install := filepath.Join(target, "install")
	if err := os.Mkdir(install, 0o700); err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, filepath.Join(install, "server.exe"), []byte("synthetic executable, never run"))
	link := filepath.Join(t.TempDir(), "linked directory")
	nativeSourceDirectoryLink(t, link, target)
	test := func(name, suffix string) {
		t.Run(name, func(t *testing.T) {
			want, err := resolveExistingPath(filepath.Join(target, suffix))
			if err != nil {
				t.Fatal(err)
			}
			got, err := resolveExistingPath(filepath.Join(link, suffix))
			if err != nil || got != want {
				t.Fatalf("resolved path = %q, %v; want %q", got, err, want)
			}
			if runtime.GOOS == "windows" && strings.HasPrefix(got, `\\?\`) {
				t.Fatalf("source binding retained extended path prefix: %q", got)
			}
			original, err := os.Stat(filepath.Join(target, suffix))
			if err != nil {
				t.Fatal(err)
			}
			resolved, err := os.Stat(got)
			if err != nil || !os.SameFile(original, resolved) {
				t.Fatalf("resolved source identity differs: %v", err)
			}
		})
	}
	test("junction itself", "")
	test("junction ancestor of directory", "install")
	test("junction ancestor of executable", filepath.Join("install", "server.exe"))
	t.Run("missing descendant fails closed", func(t *testing.T) {
		if _, err := resolveExistingPath(filepath.Join(link, "missing")); err == nil {
			t.Fatal("missing source resolved successfully")
		}
	})
}

func TestModelDeviceJunctionRuntimeSource(t *testing.T) {
	test := func(name string, ancestor bool) {
		t.Run(name, func(t *testing.T) {
			f := newInspectionFixture(t, 4096)
			target := t.TempDir()
			install := filepath.Join(target, "install")
			if err := os.Mkdir(install, 0o700); err != nil {
				t.Fatal(err)
			}
			link := filepath.Join(t.TempDir(), "linked install")
			root := link
			if ancestor {
				nativeSourceDirectoryLink(t, link, target)
				root = filepath.Join(link, "install")
			} else {
				nativeSourceDirectoryLink(t, link, install)
			}
			data := []byte("synthetic library, never loaded")
			writeTestFile(t, filepath.Join(install, "ggml.dll"), data)
			writeTestFile(t, filepath.Join(install, "server.exe"), []byte("synthetic executable, never run"))
			f.st.installDir = root
			f.st.binPath = filepath.Join(root, "server.exe")
			f.st.plat.Install = &Install{Artifacts: []InstallArtifact{{Name: "server", URL: "https://fixture.invalid/releases/download/" + modelDevicesBuild + "/native", SHA256: strings.Repeat("a", 64)}}}
			snapshot, err := modelDeviceRuntimeSnapshot(context.Background(), f.st, modelDevicesBuild)
			if err != nil {
				t.Fatal(err)
			}
			wantRoot, err := resolveExistingPath(install)
			if err != nil {
				t.Fatal(err)
			}
			sum := sha256.Sum256(data)
			if snapshot.Root != wantRoot || len(snapshot.Libraries) != 1 || snapshot.Libraries[0].Name != "ggml.dll" || snapshot.Libraries[0].SHA256 != hex.EncodeToString(sum[:]) {
				t.Fatalf("runtime snapshot = %+v", snapshot)
			}
			if _, err := modelReviewedNativeSource(context.Background(), f.st, launchCommand{Bin: f.st.binPath, Env: map[string]string{"LD_LIBRARY_PATH": root}}); err != nil {
				t.Fatalf("native source binding through link: %v", err)
			}
			binding, err := servingFileBinding(f.st.binPath)
			if err != nil {
				t.Fatal(err)
			}
			wantBin, err := resolveExistingPath(filepath.Join(install, "server.exe"))
			if err != nil {
				t.Fatal(err)
			}
			if binding.SourcePath != f.st.binPath || binding.ResolvedTarget != wantBin {
				t.Fatalf("executable binding = %+v", binding)
			}
		})
	}
	test("install directory link", false)
	test("install ancestor link", true)
}

func TestModelDeviceJunctionLibraryEscape(t *testing.T) {
	root := t.TempDir()
	// A directory named like a library must fail containment before the later
	// regular-file check. Junctions work without file-symlink privileges.
	nativeSourceDirectoryLink(t, filepath.Join(root, "ggml.dll"), t.TempDir())
	resolvedRoot, err := resolveExistingPath(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := modelDeviceHashFile(context.Background(), resolvedRoot, "ggml.dll"); err == nil || err.Error() != "library escapes selected install" {
		t.Fatalf("escaped library error = %v", err)
	}
}

func TestModelDeviceLibrarySymlinkContainment(t *testing.T) {
	test := func(name string, outside bool) {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			target := filepath.Join(root, "library-bytes")
			if outside {
				target = filepath.Join(t.TempDir(), "library-bytes")
			}
			data := []byte("synthetic library")
			writeTestFile(t, target, data)
			if err := os.Symlink(target, filepath.Join(root, "ggml.dll")); err != nil {
				t.Skipf("file symlink creation unavailable: %v", err)
			}
			resolvedRoot, err := resolveExistingPath(root)
			if err != nil {
				t.Fatal(err)
			}
			got, err := modelDeviceHashFile(context.Background(), resolvedRoot, "ggml.dll")
			if outside {
				if err == nil || err.Error() != "library escapes selected install" {
					t.Fatalf("escaped library error = %v", err)
				}
				return
			}
			sum := sha256.Sum256(data)
			if err != nil || got != hex.EncodeToString(sum[:]) {
				t.Fatalf("library digest = %q, %v", got, err)
			}
		})
	}
	test("contained file link", false)
	test("escaped file link", true)
}
