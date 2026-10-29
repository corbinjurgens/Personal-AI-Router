// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
)

func TestSafeRemoveUnderRootDeletesNestedFile(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "publisher", "model")
	if err := os.MkdirAll(target, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(target, "weights.gguf"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := safeRemoveUnderRoot(root, target); err != nil {
		t.Fatalf("safeRemoveUnderRoot: %v", err)
	}
	if _, err := os.Stat(target); !os.IsNotExist(err) {
		t.Fatalf("target still exists after delete: %v", err)
	}
}

func TestSafeRemoveUnderRootRejectsTraversal(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	if err := safeRemoveUnderRoot(root, filepath.Join(root, "..", filepath.Base(outside))); err == nil {
		t.Fatal("expected traversal escape to fail")
	}
}

func TestSafeRemoveUnderRootRejectsSymlinkEscape(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink creation requires elevated privileges on Windows")
	}
	root := t.TempDir()
	outside := t.TempDir()
	victim := filepath.Join(outside, "victim")
	if err := os.WriteFile(victim, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "escape")
	if err := os.Symlink(outside, link); err != nil {
		t.Fatal(err)
	}
	// Removing the link itself is allowed, but traversing it must not remove
	// an entry outside the root.
	if err := safeRemoveUnderRoot(root, filepath.Join(link, "victim")); err == nil {
		t.Fatal("expected symlink escape to fail")
	}
	if data, err := os.ReadFile(victim); err != nil || string(data) != "keep" {
		t.Fatalf("outside victim = %q, err = %v", data, err)
	}
}

func TestSafeRemoveUnderRootRejectsLinkWithOutsideParent(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	victim := filepath.Join(outside, "victim")
	if err := os.WriteFile(victim, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}
	inner := filepath.Join(outside, "inner")
	if err := os.Symlink(victim, inner); err != nil {
		if runtime.GOOS == "windows" && errors.Is(err, syscall.Errno(1314)) {
			t.Skip("symlink creation requires Windows symlink privileges or Developer Mode")
		}
		t.Fatal(err)
	}
	escape := filepath.Join(root, "escape")
	if err := os.Symlink(outside, escape); err != nil {
		t.Fatal(err)
	}
	if err := safeRemoveUnderRoot(root, filepath.Join(escape, "inner")); err == nil {
		t.Fatal("accepted link with parent outside root")
	}
	if _, err := os.Lstat(inner); err != nil {
		t.Fatalf("outside link did not survive: %v", err)
	}
	if data, err := os.ReadFile(victim); err != nil || string(data) != "keep" {
		t.Fatalf("outside victim = %q, err = %v", data, err)
	}
}

func TestSafeRemoveUnderRootUnlinksHardlink(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "root")
	if err := os.Mkdir(root, 0o755); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(base, "outside.txt")
	if err := os.WriteFile(outside, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "linked.txt")
	if err := os.Link(outside, link); err != nil {
		t.Fatalf("create hardlink: %v", err)
	}
	if err := safeRemoveUnderRoot(root, link); err != nil {
		t.Fatalf("safeRemoveUnderRoot: %v", err)
	}
	if _, err := os.Lstat(link); !os.IsNotExist(err) {
		t.Fatalf("hardlink remains or lstat failed: %v", err)
	}
	data, err := os.ReadFile(outside)
	if err != nil || string(data) != "keep" {
		t.Fatalf("outside file = %q, err = %v", data, err)
	}
}

func TestRemovalEntryClassificationRefusesReadlinkError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "not-a-link")
	if err := os.WriteFile(path, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	// Supply the junction mode seen by Lstat, but make Readlink fail. The
	// error must survive classification instead of authorizing tree removal.
	linked, err := removalEntryIsLink(path, os.ModeIrregular)
	if err == nil || linked {
		t.Fatalf("classification = %v, %v; want refusal", linked, err)
	}
	data, err := os.ReadFile(path)
	if err != nil || string(data) != "keep" {
		t.Fatalf("classification changed file: %q, %v", data, err)
	}
}

func TestSafeRemoveUnderRootUnlinksSymlink(t *testing.T) {
	for _, kind := range []string{"file", "directory"} {
		t.Run(kind, func(t *testing.T) {
			base := t.TempDir()
			root := filepath.Join(base, "root")
			if err := os.Mkdir(root, 0o755); err != nil {
				t.Fatal(err)
			}
			outside := filepath.Join(base, "outside")
			keep := outside
			if kind == "directory" {
				if err := os.Mkdir(outside, 0o755); err != nil {
					t.Fatal(err)
				}
				keep = filepath.Join(outside, "keep.txt")
			}
			if err := os.WriteFile(keep, []byte("keep"), 0o644); err != nil {
				t.Fatal(err)
			}
			link := filepath.Join(root, "linked")
			if err := os.Symlink(outside, link); err != nil {
				if runtime.GOOS == "windows" && errors.Is(err, syscall.Errno(1314)) {
					t.Skip("symlink creation requires Windows symlink privileges or Developer Mode")
				}
				t.Fatalf("create symlink: %v", err)
			}
			if err := safeRemoveUnderRoot(root, link); err != nil {
				t.Fatalf("safeRemoveUnderRoot: %v", err)
			}
			if _, err := os.Lstat(link); !os.IsNotExist(err) {
				t.Fatalf("symlink remains or lstat failed: %v", err)
			}
			data, err := os.ReadFile(keep)
			if err != nil || string(data) != "keep" {
				t.Fatalf("outside file = %q, err = %v", data, err)
			}
		})
	}
}

func TestSafeRemoveUnderRootDoesNotFollowChildSymlink(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "root")
	modelDir := filepath.Join(root, "model")
	outside := filepath.Join(base, "outside")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(outside, 0o755); err != nil {
		t.Fatal(err)
	}
	keep := filepath.Join(outside, "keep.txt")
	if err := os.WriteFile(keep, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(modelDir, "linked")
	if err := os.Symlink(outside, link); err != nil {
		if runtime.GOOS == "windows" && errors.Is(err, syscall.Errno(1314)) {
			t.Skip("symlink creation requires Windows symlink privileges or Developer Mode")
		}
		t.Fatalf("create symlink: %v", err)
	}
	if err := safeRemoveUnderRoot(root, modelDir); err != nil {
		t.Fatalf("safeRemoveUnderRoot: %v", err)
	}
	if _, err := os.Lstat(modelDir); !os.IsNotExist(err) {
		t.Fatalf("model directory remains or lstat failed: %v", err)
	}
	if _, err := os.Lstat(link); !os.IsNotExist(err) {
		t.Fatalf("symlink remains or lstat failed: %v", err)
	}
	data, err := os.ReadFile(keep)
	if err != nil || string(data) != "keep" {
		t.Fatalf("outside file = %q, err = %v", data, err)
	}
}

func TestSafeRemoveUnderRootRejectsBrokenSymlink(t *testing.T) {
	root := t.TempDir()
	link := filepath.Join(root, "broken")
	if err := os.Symlink(filepath.Join(root, "missing"), link); err != nil {
		if runtime.GOOS == "windows" && errors.Is(err, syscall.Errno(1314)) {
			t.Skip("symlink creation requires Windows symlink privileges or Developer Mode")
		}
		t.Fatal(err)
	}
	if err := safeRemoveUnderRoot(root, link); err == nil {
		t.Fatal("accepted a broken symlink")
	}
	if _, err := os.Lstat(link); err != nil {
		t.Fatalf("broken symlink did not survive: %v", err)
	}
}
