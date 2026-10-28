//go:build windows

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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
