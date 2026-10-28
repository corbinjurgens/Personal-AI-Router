// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func packageTempRoot(t *testing.T) string {
	t.Helper()
	path, err := filepath.EvalSymlinks(packageFixtureTempDir(t))
	if err != nil {
		t.Fatal(err)
	}
	return path
}

func TestPackagePathsRejectHostileNames(t *testing.T) {
	for _, name := range []string{"../escape", "a/../escape", "/absolute", "a//b", "a/./b", `a\b`, "file:stream", "NUL.txt", "COM1", "COM¹", "LPT9.log", "trailing.", "space ", "bad\x00name", "bad\nname", "a?b", "a|b"} {
		t.Run(name, func(t *testing.T) {
			if validRelative(name) {
				t.Fatalf("accepted hostile relative name %q", name)
			}
		})
	}
	if !validRelative("Contents/Frameworks/Electron Framework.framework/Versions/A/Electron Framework") {
		t.Fatal("rejected ordinary framework path")
	}
}

func TestPackageLocationsRejectOverlapProtectedAndCheckouts(t *testing.T) {
	t.Run("overlap", func(t *testing.T) {
		root := packageTempRoot(t)
		if err := os.Mkdir(filepath.Join(root, "child"), 0700); err != nil {
			t.Fatal(err)
		}
		if err := ValidateLocations(root, filepath.Join(root, "child"), nil); err == nil {
			t.Fatal("accepted overlap")
		}
	})
	t.Run("protected root", func(t *testing.T) {
		root := packageTempRoot(t)
		for _, name := range []string{"stage", "target"} {
			if err := os.Mkdir(filepath.Join(root, name), 0700); err != nil {
				t.Fatal(err)
			}
		}
		if err := ValidateLocations(filepath.Join(root, "stage"), filepath.Join(root, "target"), []string{root}); err == nil {
			t.Fatal("accepted protected root")
		}
	})
	t.Run("linked worktree marker", func(t *testing.T) {
		root := packageTempRoot(t)
		packageWrite(t, filepath.Join(root, ".git"), []byte("gitdir: elsewhere"))
		if err := os.Mkdir(filepath.Join(root, "stage"), 0700); err != nil {
			t.Fatal(err)
		}
		if err := ValidateLocations(filepath.Join(root, "stage"), packageTempRoot(t), nil); err == nil {
			t.Fatal("accepted source checkout")
		}
	})
}

func TestPackagePathsRejectHardLinks(t *testing.T) {
	root := packageTempRoot(t)
	outside := filepath.Join(packageTempRoot(t), "outside")
	packageWrite(t, outside, []byte("private"))
	if err := os.Link(outside, filepath.Join(root, "linked")); err != nil {
		t.Skipf("hard links unavailable: %v", err)
	}
	real, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatal(err)
	}
	held, err := OpenRealRoot(real)
	if err != nil {
		t.Fatal(err)
	}
	defer held.Close()
	p := &Package{Root: held, Platform: runtime.GOOS}
	if err := p.inventory(context.Background(), operationLimits); err == nil {
		t.Fatal("accepted an external hard link")
	}
}

func TestPackagePathsFrameworkLinks(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("framework links are only accepted on native macOS filesystems")
	}
	for _, test := range []struct {
		name, target string
		allow        bool
	}{{"contained", "Versions/A", true}, {"external", packageTempRoot(t), false}, {"traversal", "../../../../outside", false}} {
		t.Run(test.name, func(t *testing.T) {
			root := packageTempRoot(t)
			framework := filepath.Join(root, "Contents", "Frameworks", "Fixture.framework")
			if err := os.MkdirAll(filepath.Join(framework, "Versions", "A"), 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(test.target, filepath.Join(framework, "Current")); err != nil {
				t.Fatal(err)
			}
			real, err := filepath.EvalSymlinks(root)
			if err != nil {
				t.Fatal(err)
			}
			held, err := OpenRealRoot(real)
			if err != nil {
				t.Fatal(err)
			}
			defer held.Close()
			p := &Package{Root: held, Platform: "darwin"}
			err = p.inventory(context.Background(), operationLimits)
			if (err == nil) != test.allow {
				t.Fatalf("link accepted=%v, want=%v: %v", err == nil, test.allow, err)
			}
		})
	}
}

func TestPackagePathsRejectLinkedRoot(t *testing.T) {
	link := filepath.Join(packageTempRoot(t), "linked")
	if err := os.Symlink(packageTempRoot(t), link); err != nil {
		if runtime.GOOS == "windows" {
			t.Skipf("symlink privilege unavailable: %v", err)
		}
		t.Fatal(err)
	}
	if root, err := OpenRealRoot(link); err == nil {
		root.Close()
		t.Fatal("accepted symlink/reparse root")
	}
}

func TestPackageTargetDerivesFromVerifiedExecutables(t *testing.T) {
	root := packageTempRoot(t)
	gui := filepath.Join(root, "PAIR.exe")
	service := filepath.Join(root, "resources", "cli-bin", "nvpair-service.exe")
	packageWrite(t, gui, []byte("GUI"))
	packageWrite(t, service, []byte("service"))
	got, err := TargetFromExecutables(gui, service, "windows")
	if err != nil || got != root {
		t.Fatalf("target=%q, error=%v", got, err)
	}
	other := filepath.Join(packageTempRoot(t), "nvpair-service.exe")
	packageWrite(t, other, []byte("different installation"))
	if _, err := TargetFromExecutables(gui, other, "windows"); err == nil {
		t.Fatal("accepted unrelated running service")
	}
}

func TestPackageNativeProcessIdentityIsReadOnly(t *testing.T) {
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" && runtime.GOOS != "windows" {
		t.Skip("unsupported native process platform")
	}
	first, err := InspectProcess(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	second, err := InspectProcess(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	if first != second || first.Created == "" || !filepath.IsAbs(first.Executable) {
		t.Fatalf("process lifetime evidence: %+v, %+v", first, second)
	}
}

func TestPackageRootSyntaxRejectsBeforeOpening(t *testing.T) {
	for _, path := range []string{`\\server\share\bundle`, `\\?\C:\bundle`, `\\.\C:\bundle`, "//server/share/bundle", filepath.Join(packageTempRoot(t), "safe") + string(filepath.Separator) + ".." + string(filepath.Separator) + "escape", filepath.Join(packageTempRoot(t), "file:stream")} {
		if err := localRootSyntax(path); err == nil {
			t.Fatalf("accepted ambiguous root syntax %q", path)
		}
	}
}

func TestPackageCaseFoldIncludesUnicodeAliases(t *testing.T) {
	if foldPackagePath("resources/Σ") != foldPackagePath("resources/ς") {
		t.Fatal("case aliases escaped inventory collision checks")
	}
}

func TestPackageOwnerFileReadIsBoundedAndReadOnly(t *testing.T) {
	root := packageTempRoot(t)
	path := filepath.Join(root, "policy.json")
	packageWrite(t, path, []byte(`{"pause":{"stopEngines":false}}`))
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	data, err := ReadLocalFile(path, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"pause":{"stopEngines":false}}` {
		t.Fatalf("owner bytes changed: %q", data)
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(before, after) || !before.ModTime().Equal(after.ModTime()) {
		t.Fatal("owner read wrote the file")
	}
	if _, err := ReadLocalFile(path, 2); err == nil {
		t.Fatal("unbounded owner file read")
	}
	missing := filepath.Join(root, "missing.json")
	if _, err := ReadLocalFile(missing, 1024); !os.IsNotExist(err) {
		t.Fatalf("missing owner state = %v", err)
	}
}
