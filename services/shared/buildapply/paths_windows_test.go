//go:build windows

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

func TestPackageWindowsOwnerAndACL(t *testing.T) {
	token := windows.GetCurrentProcessToken()
	user, err := token.GetTokenUser()
	if err != nil {
		t.Fatal(err)
	}
	owner, err := currentTokenOwner(token)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("elevated=%t TokenOwnerEqualsTokenUser=%t", token.IsElevated(), owner.Equals(user.User.Sid))
	testOwner := func(name string, sid *windows.SID, accepted bool) {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(packageFixtureTempDir(t), "owned")
			packageWrite(t, path, []byte("inert"))
			if err := windows.SetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.OWNER_SECURITY_INFORMATION, sid, nil, nil, nil); err != nil {
				t.Skipf("shell cannot assign fixture owner: %v", err)
			}
			info, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			err = nativePathCheck(path, info, true)
			if accepted && err != nil {
				t.Fatal(err)
			}
			if !accepted && (err == nil || !strings.Contains(err.Error(), "not owned")) {
				t.Fatalf("foreign owner refusal = %v", err)
			}
		})
	}
	testOwner("TokenUser owner", user.User.Sid, true)
	if owner.Equals(user.User.Sid) {
		t.Run("distinct TokenOwner owner", func(t *testing.T) { t.Skip("token default owner equals token user") })
	} else {
		testOwner("distinct TokenOwner owner", owner, true)
	}
	foreign, err := windows.CreateWellKnownSid(windows.WinWorldSid)
	if err != nil {
		t.Fatal(err)
	}
	testOwner("foreign owner", foreign, false)
	t.Run("foreign writable ACE", func(t *testing.T) {
		path := filepath.Join(packageFixtureTempDir(t), "foreign-writable")
		packageWrite(t, path, []byte("inert"))
		sd, err := windows.SecurityDescriptorFromString("D:(A;;FA;;;WD)")
		if err != nil {
			t.Fatal(err)
		}
		acl, _, err := sd.DACL()
		if err != nil {
			t.Fatal(err)
		}
		if err := windows.SetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION, nil, nil, acl, nil); err != nil {
			t.Fatal(err)
		}
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := nativePathCheck(path, info, true); err == nil || !strings.Contains(err.Error(), "writable by another principal") {
			t.Fatalf("foreign ACL refusal = %v", err)
		}
	})
}

func TestPackageWindowsInheritedForeignWritableACERefused(t *testing.T) {
	parent := packageFixtureTempDir(t)
	user, err := windows.GetCurrentProcessToken().GetTokenUser()
	if err != nil {
		t.Fatal(err)
	}
	sd, err := windows.SecurityDescriptorFromString("D:P(A;OICI;FA;;;" + user.User.Sid.String() + ")(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;WD)")
	if err != nil {
		t.Fatal(err)
	}
	acl, _, err := sd.DACL()
	if err != nil {
		t.Fatal(err)
	}
	if err := windows.SetNamedSecurityInfo(parent, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION|windows.PROTECTED_DACL_SECURITY_INFORMATION, nil, nil, acl, nil); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(parent, "child", "inherited")
	packageWrite(t, path, []byte("inert"))
	child, err := windows.GetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION)
	if err != nil {
		t.Fatal(err)
	}
	childACL, _, err := child.DACL()
	if err != nil {
		t.Fatal(err)
	}
	world, err := windows.CreateWellKnownSid(windows.WinWorldSid)
	if err != nil {
		t.Fatal(err)
	}
	inherited := false
	for i := uint32(0); i < uint32(childACL.AceCount); i++ {
		var ace *windows.ACCESS_ALLOWED_ACE
		if err := windows.GetAce(childACL, i, &ace); err != nil {
			t.Fatal(err)
		}
		if ace.Header.AceType == windows.ACCESS_ALLOWED_ACE_TYPE && ace.Header.AceFlags&windows.INHERITED_ACE != 0 && ace.Mask&windows.FILE_WRITE_DATA != 0 {
			sid := (*windows.SID)(unsafe.Pointer(&ace.SidStart))
			inherited = inherited || sid.Equals(world)
		}
	}
	if !inherited {
		t.Fatal("fixture did not inherit the foreign writable ACE")
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := nativePathCheck(path, info, true); err == nil || !strings.Contains(err.Error(), "writable by another principal") {
		t.Fatalf("inherited foreign ACL refusal = %v", err)
	}
}

// packageJunction creates a directory junction, which needs no privilege.
func packageJunction(t *testing.T, target, link string) {
	t.Helper()
	if out, err := exec.Command("cmd", "/c", "mklink", "/J", link, target).CombinedOutput(); err != nil {
		t.Skipf("junction unavailable: %v: %s", err, out)
	}
}

// A junction is Windows' usual unprivileged way to relocate a profile or
// AppData. Above a root it is followed like a symbolic link; as a root, or
// inside one, it stays refused and the refusal names it.
func TestRootsAllowJunctionsAboveAndRefuseJunctionsAtOrBelow(t *testing.T) {
	t.Run("junction above selected roots", func(t *testing.T) {
		root := packageTempRoot(t)
		packageMkdirs(t, filepath.Join(root, "real", "stage"), filepath.Join(root, "real", "target"))
		above := filepath.Join(root, "above")
		packageJunction(t, filepath.Join(root, "real"), above)
		if err := ValidateLocations(filepath.Join(above, "stage"), filepath.Join(above, "target"), []string{filepath.Join(above, "profile")}); err != nil {
			t.Fatalf("refused roots below a junction: %v", err)
		}
	})
	t.Run("junction above the profile directory", func(t *testing.T) {
		root := packageTempRoot(t)
		packageMkdirs(t, filepath.Join(root, "real"))
		above := filepath.Join(root, "above")
		packageJunction(t, filepath.Join(root, "real"), above)
		j, err := openJournalAt(filepath.Join(above, "profile", "build-apply"), true)
		if err != nil {
			t.Fatalf("journal below a junction above the profile: %v", err)
		}
		defer j.Close()
		if want := filepath.Join(root, "real", "profile", "build-apply"); j.Path() != want {
			t.Fatalf("journal path %q, want resolved %q", j.Path(), want)
		}
		packageWrite(t, filepath.Join(root, "real", "profile", "cluster", "identity.json"), []byte("{}"))
		if data, err := ReadLocalFile(filepath.Join(above, "profile"), "cluster/identity.json", 1024); err != nil || string(data) != "{}" {
			t.Fatalf("owner file below a junction above the profile: %q, %v", data, err)
		}
	})
	t.Run("installation below a junction", func(t *testing.T) {
		root := packageTempRoot(t)
		install := filepath.Join(root, "real", "install")
		packageWrite(t, filepath.Join(install, "PAIR.exe"), []byte("GUI"))
		packageWrite(t, filepath.Join(install, "resources", "cli-bin", "nvpair-service.exe"), []byte("service"))
		above := filepath.Join(root, "above")
		packageJunction(t, filepath.Join(root, "real"), above)
		got, err := TargetFromExecutables(filepath.Join(above, "install", "PAIR.exe"), filepath.Join(above, "install", "resources", "cli-bin", "nvpair-service.exe"), "windows")
		if err != nil || got != install {
			t.Fatalf("target=%q, error=%v, want resolved %q", got, err, install)
		}
	})
	t.Run("junction as selected root", func(t *testing.T) {
		root := packageTempRoot(t)
		packageMkdirs(t, filepath.Join(root, "real"), filepath.Join(root, "target"))
		link := filepath.Join(root, "linked")
		packageJunction(t, filepath.Join(root, "real"), link)
		err := ValidateLocations(link, filepath.Join(root, "target"), nil)
		if err == nil || !strings.HasSuffix(err.Error(), ": "+link) {
			t.Fatalf("junction root error = %v", err)
		}
	})
	t.Run("junction as profile directory", func(t *testing.T) {
		root := packageTempRoot(t)
		packageMkdirs(t, filepath.Join(root, "real", "build-apply"))
		profile := filepath.Join(root, "profile")
		packageJunction(t, filepath.Join(root, "real"), profile)
		j, err := openJournalAt(filepath.Join(profile, "build-apply"), false)
		if err == nil {
			j.Close()
			t.Fatal("opened a journal in a junction profile directory")
		}
		if !strings.HasSuffix(err.Error(), ": "+profile) {
			t.Fatalf("junction profile error = %v", err)
		}
		if _, err := ReadLocalFile(profile, "build-apply/state.json", 1024); err == nil || !strings.HasSuffix(err.Error(), ": "+profile) {
			t.Fatalf("junction profile owner read error = %v", err)
		}
	})
	t.Run("junction inside installation", func(t *testing.T) {
		root := packageTempRoot(t)
		install := filepath.Join(root, "install")
		elsewhere := filepath.Join(root, "elsewhere")
		packageWrite(t, filepath.Join(install, "PAIR.exe"), []byte("GUI"))
		packageWrite(t, filepath.Join(elsewhere, "cli-bin", "nvpair-service.exe"), []byte("service"))
		resources := filepath.Join(install, "resources")
		packageJunction(t, elsewhere, resources)
		_, err := TargetFromExecutables(filepath.Join(install, "PAIR.exe"), filepath.Join(resources, "cli-bin", "nvpair-service.exe"), "windows")
		if err == nil || !strings.HasSuffix(err.Error(), ": "+resources) {
			t.Fatalf("junction inside installation error = %v", err)
		}
	})
}

// Locality is checked before a path is read through: a link above a root that
// leads to a network share is refused without being followed.
func TestLinkAboveARootToANetworkShareIsRefusedUnread(t *testing.T) {
	root := packageTempRoot(t)
	above := filepath.Join(root, "above")
	// os.Symlink stats its target to choose the link type, which would reach
	// for the share; create the directory link directly instead.
	linkName, err := windows.UTF16PtrFromString(above)
	if err != nil {
		t.Fatal(err)
	}
	share, err := windows.UTF16PtrFromString(`\\pair-test-host.invalid\share`)
	if err != nil {
		t.Fatal(err)
	}
	const allowUnprivilegedCreate = 0x2 // SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE
	if err := windows.CreateSymbolicLink(linkName, share, windows.SYMBOLIC_LINK_FLAG_DIRECTORY|allowUnprivilegedCreate); err != nil {
		t.Skipf("symlink privilege unavailable: %v", err)
	}
	started := time.Now()
	err = ValidateLocations(filepath.Join(above, "stage"), filepath.Join(root, "target"), nil)
	if err == nil || !strings.Contains(err.Error(), "does not lead to a local path: "+above) {
		t.Fatalf("network link above a root error = %v", err)
	}
	if elapsed := time.Since(started); elapsed > 5*time.Second {
		t.Fatalf("refusal took %v; the share was probably contacted", elapsed)
	}
}

// An 8.3 alias names the same directory as its long name, so it cannot select
// a protected root.
func TestLocationsRefuseOverlapThroughAShortName(t *testing.T) {
	root := packageTempRoot(t)
	protected := filepath.Join(root, "Long Protected Name")
	target := filepath.Join(root, "target")
	packageMkdirs(t, protected, target)
	long, err := windows.UTF16PtrFromString(protected)
	if err != nil {
		t.Fatal(err)
	}
	buffer := make([]uint16, windows.MAX_LONG_PATH)
	n, err := windows.GetShortPathName(long, &buffer[0], uint32(len(buffer)))
	if err != nil || n == 0 || int(n) >= len(buffer) {
		t.Skipf("short name unavailable: %v", err)
	}
	short := filepath.Join(root, filepath.Base(windows.UTF16ToString(buffer[:n])))
	if strings.EqualFold(short, protected) {
		t.Skip("8.3 names are disabled on this volume")
	}
	for _, test := range []struct{ name, source, protected string }{
		{"short name selects the protected root", short, protected},
		{"protected root named by its short name", protected, short},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := ValidateLocations(test.source, target, []string{test.protected})
			if err == nil || !strings.Contains(err.Error(), "bundle overlaps protected data or model root") {
				t.Fatalf("short-name overlap error = %v", err)
			}
		})
	}
	t.Run("short name overlaps the other selection", func(t *testing.T) {
		if err := ValidateLocations(short, protected, nil); err == nil || !strings.Contains(err.Error(), "source and target overlap") {
			t.Fatalf("short-name selection overlap error = %v", err)
		}
	})
}
