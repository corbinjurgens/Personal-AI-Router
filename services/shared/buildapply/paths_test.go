// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
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

func packageSymlink(t *testing.T, target, link string) {
	t.Helper()
	if err := os.Symlink(target, link); err != nil {
		if runtime.GOOS == "windows" {
			t.Skipf("symlink privilege unavailable: %v", err)
		}
		t.Fatal(err)
	}
}

func packageMkdirs(t *testing.T, paths ...string) {
	t.Helper()
	for _, path := range paths {
		if err := os.MkdirAll(path, 0o700); err != nil {
			t.Fatal(err)
		}
	}
}

// The owner allows a link above PAIR's own directories (a linked HOME or
// ~/Library, macOS's /var -> /private/var). A link at or below a root PAIR
// selects or owns stays refused, and the refusal names the link.
func TestRootsAllowLinksAboveAndRefuseLinksAtOrBelow(t *testing.T) {
	t.Run("link above selected roots", func(t *testing.T) {
		root := packageTempRoot(t)
		packageMkdirs(t, filepath.Join(root, "real", "stage"), filepath.Join(root, "real", "target"))
		above := filepath.Join(root, "above")
		packageSymlink(t, filepath.Join(root, "real"), above)
		if err := ValidateLocations(filepath.Join(above, "stage"), filepath.Join(above, "target"), nil); err != nil {
			t.Fatalf("refused roots below a link: %v", err)
		}
	})
	t.Run("unresolved temp directory", func(t *testing.T) {
		// On macOS t.TempDir() sits below /var -> /private/var.
		root := t.TempDir()
		packageMkdirs(t, filepath.Join(root, "stage"), filepath.Join(root, "target"))
		if err := ValidateLocations(filepath.Join(root, "stage"), filepath.Join(root, "target"), []string{filepath.Join(root, "profile")}); err != nil {
			t.Fatalf("refused roots in the platform temp directory: %v", err)
		}
		j, err := openJournalAt(filepath.Join(root, "profile", "build-apply"), true)
		if err != nil {
			t.Fatalf("journal in the platform temp directory: %v", err)
		}
		defer j.Close()
		real, err := filepath.EvalSymlinks(root)
		if err != nil {
			t.Fatal(err)
		}
		if want := filepath.Join(real, "profile", "build-apply"); j.Path() != want {
			t.Fatalf("journal path %q, want resolved %q", j.Path(), want)
		}
		reopened, err := openJournalAt(filepath.Join(root, "profile", "build-apply"), false)
		if err != nil {
			t.Fatalf("reopen journal: %v", err)
		}
		reopened.Close()
	})
	t.Run("link as selected root", func(t *testing.T) {
		root := packageTempRoot(t)
		packageMkdirs(t, filepath.Join(root, "real"), filepath.Join(root, "target"))
		link := filepath.Join(root, "linked")
		packageSymlink(t, filepath.Join(root, "real"), link)
		err := ValidateLocations(link, filepath.Join(root, "target"), nil)
		if err == nil || !strings.Contains(err.Error(), "linked root or ancestor: "+link) {
			t.Fatalf("linked root error = %v", err)
		}
	})
	t.Run("link as profile directory", func(t *testing.T) {
		root := packageTempRoot(t)
		packageMkdirs(t, filepath.Join(root, "real", "build-apply"))
		profile := filepath.Join(root, "profile")
		packageSymlink(t, filepath.Join(root, "real"), profile)
		j, err := openJournalAt(filepath.Join(profile, "build-apply"), false)
		if err == nil {
			j.Close()
			t.Fatal("opened a journal in a linked profile directory")
		}
		if !strings.Contains(err.Error(), "linked root or ancestor: "+profile) {
			t.Fatalf("linked profile error = %v", err)
		}
	})
	t.Run("link below installation root", func(t *testing.T) {
		root := packageTempRoot(t)
		install := filepath.Join(root, "install")
		elsewhere := filepath.Join(root, "elsewhere")
		packageWrite(t, filepath.Join(install, "PAIR.exe"), []byte("GUI"))
		packageWrite(t, filepath.Join(elsewhere, "cli-bin", "nvpair-service.exe"), []byte("service"))
		resources := filepath.Join(install, "resources")
		packageSymlink(t, elsewhere, resources)
		_, err := TargetFromExecutables(filepath.Join(install, "PAIR.exe"), filepath.Join(resources, "cli-bin", "nvpair-service.exe"), "windows")
		if err == nil || !strings.Contains(err.Error(), "linked root or ancestor: "+resources) {
			t.Fatalf("link inside installation error = %v", err)
		}
	})
	t.Run("installation below a link", func(t *testing.T) {
		root := packageTempRoot(t)
		install := filepath.Join(root, "real", "install")
		packageWrite(t, filepath.Join(install, "PAIR.exe"), []byte("GUI"))
		packageWrite(t, filepath.Join(install, "resources", "cli-bin", "nvpair-service.exe"), []byte("service"))
		above := filepath.Join(root, "above")
		packageSymlink(t, filepath.Join(root, "real"), above)
		got, err := TargetFromExecutables(filepath.Join(above, "install", "PAIR.exe"), filepath.Join(above, "install", "resources", "cli-bin", "nvpair-service.exe"), "windows")
		if err != nil || got != install {
			t.Fatalf("target=%q, error=%v, want resolved %q", got, err, install)
		}
	})
}

// Containment is decided on resolved paths only: a selection that reaches a
// protected tree through a link, or a protected root named through a link,
// still overlaps.
func TestLocationsRefuseOverlapThroughALink(t *testing.T) {
	root := packageTempRoot(t)
	profile := filepath.Join(root, "profile")
	target := filepath.Join(root, "target")
	packageMkdirs(t, filepath.Join(profile, "stage"), target)
	alias := filepath.Join(root, "alias")
	packageSymlink(t, profile, alias)
	for _, test := range []struct {
		name, source, protected string
	}{
		{"selection reaches the protected tree through a link", filepath.Join(alias, "stage"), profile},
		{"protected root is named through a link", filepath.Join(profile, "stage"), filepath.Join(alias, "stage")},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := ValidateLocations(test.source, target, []string{test.protected})
			if err == nil || !strings.Contains(err.Error(), "bundle overlaps protected data or model root") {
				t.Fatalf("overlap through a link error = %v", err)
			}
		})
	}
	t.Run("protected root is itself a link", func(t *testing.T) {
		err := ValidateLocations(target, filepath.Join(profile, "stage"), []string{alias})
		if err == nil || !strings.Contains(err.Error(), "protected root is linked or unavailable") {
			t.Fatalf("linked protected root error = %v", err)
		}
	})
}

// An owner file is read below its owning root: a link above the root is
// resolved, while a linked root is refused even for a file one level down.
func TestOwnerFileReadResolvesOnlyAboveTheOwningRoot(t *testing.T) {
	root := packageTempRoot(t)
	real := filepath.Join(root, "real")
	packageWrite(t, filepath.Join(real, "profile", "cluster", "identity.json"), []byte("{}"))
	above := filepath.Join(root, "above")
	packageSymlink(t, real, above)
	t.Run("link above the owning root", func(t *testing.T) {
		if data, err := ReadLocalFile(filepath.Join(above, "profile"), "cluster/identity.json", 1024); err != nil || string(data) != "{}" {
			t.Fatalf("read below a link above the profile: %q, %v", data, err)
		}
	})
	t.Run("owning root is a link", func(t *testing.T) {
		profile := filepath.Join(root, "profile")
		packageSymlink(t, filepath.Join(real, "profile"), profile)
		_, err := ReadLocalFile(profile, "cluster/identity.json", 1024)
		if err == nil || !strings.Contains(err.Error(), "linked root or ancestor: "+profile) {
			t.Fatalf("linked profile read error = %v", err)
		}
	})
}

// Containment does not depend on spelling: a case variant on a
// case-insensitive volume names the protected root itself.
func TestLocationsRefuseOverlapThroughACaseVariant(t *testing.T) {
	root := packageTempRoot(t)
	protected := filepath.Join(root, "Protected")
	target := filepath.Join(root, "target")
	packageMkdirs(t, protected, target)
	variant := filepath.Join(root, "PROTECTED")
	if _, err := os.Lstat(variant); err != nil {
		t.Skipf("volume is case-sensitive: %v", err)
	}
	err := ValidateLocations(variant, target, []string{protected})
	if err == nil || !strings.Contains(err.Error(), "bundle overlaps protected data or model root") {
		t.Fatalf("case variant of a protected root error = %v", err)
	}
}

// Allowing a link above does not weaken substitution detection: the held root
// is compared with what its resolved name reaches now.
func TestRealRootSubstitutedAfterOpenIsRefused(t *testing.T) {
	root := packageTempRoot(t)
	packageMkdirs(t, filepath.Join(root, "real", "selected"))
	above := filepath.Join(root, "above")
	packageSymlink(t, filepath.Join(root, "real"), above)
	real, err := resolveRoot(filepath.Join(above, "selected"))
	if err != nil {
		t.Fatal(err)
	}
	held, err := OpenRealRoot(real)
	if err != nil {
		t.Fatal(err)
	}
	defer held.Close()
	if err := os.Rename(real, real+"-moved"); err != nil {
		if runtime.GOOS == "windows" {
			t.Skipf("held Windows directory prevented the substitution: %v", err)
		}
		t.Fatal(err)
	}
	packageMkdirs(t, real)
	if err := held.Revalidate(); err == nil || !strings.Contains(err.Error(), "substituted") {
		t.Fatalf("substituted root revalidation = %v", err)
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
	data, err := ReadLocalFile(root, "policy.json", 1024)
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
	if _, err := ReadLocalFile(root, "policy.json", 2); err == nil {
		t.Fatal("unbounded owner file read")
	}
	for _, missing := range []string{"missing.json", "cluster/identity.json"} {
		if _, err := ReadLocalFile(root, missing, 1024); !os.IsNotExist(err) {
			t.Fatalf("missing owner state %s = %v", missing, err)
		}
	}
	for _, hostile := range []string{"../policy.json", "/policy.json", `cluster\identity.json`} {
		if _, err := ReadLocalFile(root, hostile, 1024); err == nil {
			t.Fatalf("read owner file through hostile name %q", hostile)
		}
	}
}
