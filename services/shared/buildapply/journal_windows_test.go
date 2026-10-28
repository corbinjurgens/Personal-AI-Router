// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unsafe"

	"golang.org/x/sys/windows"
)

func TestJournalWindowsRenameLayout(t *testing.T) {
	var info journalRenameInfo
	if unsafe.Offsetof(info.Flags) != 0 || unsafe.Offsetof(info.RootDirectory) != 8 ||
		unsafe.Offsetof(info.FileNameLength) != 16 || unsafe.Offsetof(info.FileName) != 20 || unsafe.Sizeof(info) != 24 {
		t.Fatal("journal rename layout differs from FILE_RENAME_INFORMATION_EX")
	}
}

func TestJournalWindowsUnsupportedRenameFailsClearly(t *testing.T) {
	for _, status := range []windows.NTStatus{
		windows.STATUS_INVALID_INFO_CLASS, windows.STATUS_NOT_IMPLEMENTED,
		windows.STATUS_NOT_SUPPORTED, windows.STATUS_INVALID_PARAMETER,
	} {
		t.Run(status.Error(), func(t *testing.T) {
			err := journalRenameError(status)
			if !errors.Is(err, status) || !strings.Contains(err.Error(), "requires Windows 10 version 1709 (build 16299) or later and a supporting filesystem") {
				t.Fatalf("unsupported rename error = %v", err)
			}
		})
	}
}

func TestJournalWindowsRejectsForeignReadAndWriteWithoutRepair(t *testing.T) {
	for _, fixture := range []struct{ name, rights string }{
		{"foreign read", "FR"},
		{"foreign write", "FW"},
	} {
		t.Run(fixture.name, func(t *testing.T) {
			j, lock := journalFixture(t)
			beginJournal(t, j, lock)
			user, err := windows.GetCurrentProcessToken().GetTokenUser()
			if err != nil {
				t.Fatalf("read current fixture user: %v", err)
			}
			sd, err := windows.SecurityDescriptorFromString("D:P(A;;FA;;;" + user.User.Sid.String() + ")(A;;FA;;;SY)(A;;FA;;;BA)(A;;" + fixture.rights + ";;;WD)")
			if err != nil {
				t.Fatalf("create foreign-access fixture DACL: %v", err)
			}
			acl, _, err := sd.DACL()
			if err != nil {
				t.Fatalf("read foreign-access fixture DACL: %v", err)
			}
			name := filepath.Join(j.Path(), "operation.json")
			if err := windows.SetNamedSecurityInfo(name, windows.SE_FILE_OBJECT,
				windows.DACL_SECURITY_INFORMATION|windows.PROTECTED_DACL_SECURITY_INFORMATION, nil, nil, acl, nil); err != nil {
				t.Fatalf("set synthetic receipt DACL: %v", err)
			}
			before, err := windows.GetNamedSecurityInfo(name, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION)
			if err != nil {
				t.Fatalf("read fixture DACL before refusal: %v", err)
			}
			if _, err := j.Read(); !errors.Is(err, ErrInvalidJournal) {
				t.Fatalf("foreign-access receipt read error = %v", err)
			}
			after, err := windows.GetNamedSecurityInfo(name, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION)
			if err != nil {
				t.Fatalf("read fixture DACL after refusal: %v", err)
			}
			if after.String() != before.String() {
				t.Fatalf("reader repaired existing DACL: before=%s, after=%s", before.String(), after.String())
			}
			if fixture.rights == "FR" {
				info, err := os.Stat(name)
				if err != nil {
					t.Fatalf("stat foreign-readable receipt: %v", err)
				}
				// Confidential journal reads are stricter than the unchanged
				// package path guard, which rejects foreign writes only.
				if err := nativePathCheck(name, info, true); err != nil {
					t.Fatalf("journal work changed prior foreign-read path policy: %v", err)
				}
			}
		})
	}
}

func TestJournalWindowsRejectsExistingForeignReadableRoot(t *testing.T) {
	parent, err := OpenRealRoot(packageFixtureTempDir(t))
	if err != nil {
		t.Fatalf("open private fixture parent: %v", err)
	}
	defer func() {
		if err := parent.Close(); err != nil {
			t.Errorf("close fixture parent: %v", err)
		}
	}()
	if err := nativePrivateMkdir(parent, "state"); err != nil {
		t.Fatalf("create private state fixture: %v", err)
	}
	user, err := windows.GetCurrentProcessToken().GetTokenUser()
	if err != nil {
		t.Fatalf("read root fixture user: %v", err)
	}
	sd, err := windows.SecurityDescriptorFromString("D:P(A;OICI;FA;;;" + user.User.Sid.String() + ")(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FR;;;WD)")
	if err != nil {
		t.Fatalf("create foreign-readable root DACL: %v", err)
	}
	acl, _, err := sd.DACL()
	if err != nil {
		t.Fatalf("read root fixture DACL: %v", err)
	}
	name := filepath.Join(parent.Path, "state")
	if err := windows.SetNamedSecurityInfo(name, windows.SE_FILE_OBJECT,
		windows.DACL_SECURITY_INFORMATION|windows.PROTECTED_DACL_SECURITY_INFORMATION, nil, nil, acl, nil); err != nil {
		t.Fatalf("set synthetic root DACL: %v", err)
	}
	if j, err := openJournalAt(name, true); err == nil {
		if err := j.Close(); err != nil {
			t.Errorf("close unexpectedly accepted root: %v", err)
		}
		t.Fatal("existing foreign-readable root was accepted/repaired")
	}
}

func TestOperationLockWindowsRejectsForeignReadableLock(t *testing.T) {
	j, owner := journalFixture(t)
	if err := owner.Close(); err != nil {
		t.Fatalf("release synthetic lock: %v", err)
	}
	user, err := windows.GetCurrentProcessToken().GetTokenUser()
	if err != nil {
		t.Fatalf("read lock fixture user: %v", err)
	}
	sd, err := windows.SecurityDescriptorFromString("D:P(A;;FA;;;" + user.User.Sid.String() + ")(A;;FA;;;SY)(A;;FA;;;BA)(A;;FR;;;WD)")
	if err != nil {
		t.Fatalf("create foreign-readable lock DACL: %v", err)
	}
	acl, _, err := sd.DACL()
	if err != nil {
		t.Fatalf("read lock fixture DACL: %v", err)
	}
	if err := windows.SetNamedSecurityInfo(filepath.Join(j.Path(), "operation.lock"), windows.SE_FILE_OBJECT,
		windows.DACL_SECURITY_INFORMATION|windows.PROTECTED_DACL_SECURITY_INFORMATION, nil, nil, acl, nil); err != nil {
		t.Fatalf("set synthetic lock DACL: %v", err)
	}
	if lock, err := j.TryLock(); err == nil {
		if err := lock.Close(); err != nil {
			t.Errorf("close unexpectedly acquired lock: %v", err)
		}
		t.Fatal("foreign-readable lock accepted")
	}
}
