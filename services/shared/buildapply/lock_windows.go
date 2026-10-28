// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"unsafe"

	"nvpair-shared/durablefile"

	"golang.org/x/sys/windows"
)

// OperationLock owns an operating-system lock until Close. It is not a
// reservation for a successor: the receipt must authorize any handoff.
type OperationLock struct {
	file *os.File
	root *RealRoot
}

// TryOperationLock takes the canonical lock without waiting. The root must
// already be the private operation directory, held and checked by the caller.
func TryOperationLock(root *RealRoot) (*OperationLock, error) {
	if err := validatePrivateRoot(root); err != nil {
		return nil, err
	}
	f, err := openOperationLock(root)
	if err != nil {
		return nil, err
	}
	if err := validateLockFile(root, f); err != nil {
		_ = f.Close()
		return nil, err
	}
	var overlapped windows.Overlapped
	if err := windows.LockFileEx(windows.Handle(f.Fd()), windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &overlapped); err != nil {
		_ = f.Close()
		return nil, err
	}
	if err := validateLockFile(root, f); err != nil {
		_ = f.Close()
		return nil, err
	}
	return &OperationLock{file: f, root: root}, nil
}

func validateLockFile(root *RealRoot, f *os.File) error {
	named, err := root.root.Lstat("operation.lock")
	if err != nil {
		return err
	}
	held, err := f.Stat()
	if err != nil {
		return err
	}
	if !named.Mode().IsRegular() || !held.Mode().IsRegular() || !os.SameFile(named, held) || held.Size() != 0 {
		return errors.New("operation lock is linked or substituted")
	}
	if err := nativePrivateCheck(f, held); err != nil {
		return err
	}
	return validatePrivateRoot(root)
}

func openOperationLock(root *RealRoot) (*os.File, error) {
	f, err := root.root.OpenFile("operation.lock", os.O_CREATE|os.O_EXCL|os.O_RDWR, 0o600)
	if os.IsExist(err) {
		info, statErr := root.root.Lstat("operation.lock")
		if statErr != nil {
			return nil, statErr
		}
		if !info.Mode().IsRegular() {
			return nil, errors.New("operation lock is not regular")
		}
		return root.root.OpenFile("operation.lock", os.O_RDWR, 0)
	}
	if err != nil {
		return nil, err
	}
	if err := nativePrivateCreatedFile(f); err != nil {
		f.Close()
		return nil, err
	}
	return f, nil
}

func privateSecurityDescriptor() (*windows.SECURITY_DESCRIPTOR, error) {
	user, err := windows.GetCurrentProcessToken().GetTokenUser()
	if err != nil {
		return nil, err
	}
	// The protected inheritable DACL is installed at creation, independently
	// of TEMP/appdir inheritance. Existing objects are never repaired here.
	return windows.SecurityDescriptorFromString("D:P(A;OICI;FA;;;" + user.User.Sid.String() + ")(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)")
}

func nativePrivateMkdir(parent *RealRoot, name string) error {
	if err := parent.Revalidate(); err != nil {
		return err
	}
	sd, err := privateSecurityDescriptor()
	if err != nil {
		return err
	}
	dir, err := parent.root.Open(".")
	if err != nil {
		return err
	}
	defer dir.Close()
	nativeName, err := windows.NewNTUnicodeString(name)
	if err != nil {
		return err
	}
	attrs := windows.OBJECT_ATTRIBUTES{
		Length: uint32(unsafe.Sizeof(windows.OBJECT_ATTRIBUTES{})), RootDirectory: windows.Handle(dir.Fd()),
		ObjectName: nativeName, Attributes: windows.OBJ_CASE_INSENSITIVE | windows.OBJ_DONT_REPARSE, SecurityDescriptor: sd,
	}
	var h windows.Handle
	var status windows.IO_STATUS_BLOCK
	err = windows.NtCreateFile(&h, windows.FILE_GENERIC_READ|windows.FILE_GENERIC_WRITE, &attrs, &status, nil,
		windows.FILE_ATTRIBUTE_DIRECTORY, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE,
		windows.FILE_CREATE, windows.FILE_DIRECTORY_FILE|windows.FILE_SYNCHRONOUS_IO_NONALERT|windows.FILE_OPEN_REPARSE_POINT, 0, 0)
	runtime.KeepAlive(sd)
	if err != nil {
		var ntstatus windows.NTStatus
		if errors.As(err, &ntstatus) {
			return ntstatus.Errno()
		}
		return err
	}
	return windows.CloseHandle(h)
}

func nativePrivateCreatedFile(f *os.File) error {
	sd, err := privateSecurityDescriptor()
	if err != nil {
		return err
	}
	acl, _, err := sd.DACL()
	if err != nil {
		return err
	}
	// Only this exclusively created, still-empty file is changed. Its initial
	// inherited DACL already came from the validated private parent.
	h, err := reopenJournalFile(f, windows.WRITE_DAC|windows.READ_CONTROL, 0)
	if err != nil {
		return err
	}
	defer windows.CloseHandle(h)
	if err := windows.SetSecurityInfo(h, windows.SE_FILE_OBJECT,
		windows.DACL_SECURITY_INFORMATION|windows.PROTECTED_DACL_SECURITY_INFORMATION, nil, nil, acl, nil); err != nil {
		return err
	}
	info, err := f.Stat()
	if err != nil {
		return err
	}
	return nativePrivateCheck(f, info)
}

var journalReopenFile = windows.NewLazySystemDLL("kernel32.dll").NewProc("ReOpenFile")

func reopenJournalFile(f *os.File, access, flags uint32) (windows.Handle, error) {
	h, _, err := journalReopenFile.Call(f.Fd(), uintptr(access),
		uintptr(windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE), uintptr(flags))
	if h == uintptr(windows.InvalidHandle) {
		return windows.InvalidHandle, err
	}
	return windows.Handle(h), nil
}

func nativePrivateCheck(f *os.File, info os.FileInfo) error {
	h := windows.Handle(f.Fd())
	if err := nativeHandleCheck(h, info, true); err != nil {
		return err
	}
	sd, err := windows.GetSecurityInfo(h, windows.SE_FILE_OBJECT, windows.OWNER_SECURITY_INFORMATION|windows.DACL_SECURITY_INFORMATION)
	if err != nil {
		return err
	}
	control, _, err := sd.Control()
	if err != nil || control&windows.SE_DACL_PROTECTED == 0 {
		return errors.New("private operation state requires a protected DACL")
	}
	acl, _, err := sd.DACL()
	if err != nil || acl == nil {
		return errors.New("private operation state has no restrictive DACL")
	}
	user, err := windows.GetCurrentProcessToken().GetTokenUser()
	if err != nil {
		return err
	}
	admins, err := windows.CreateWellKnownSid(windows.WinBuiltinAdministratorsSid)
	if err != nil {
		return err
	}
	system, err := windows.CreateWellKnownSid(windows.WinLocalSystemSid)
	if err != nil {
		return err
	}
	for i := uint32(0); i < uint32(acl.AceCount); i++ {
		var ace *windows.ACCESS_ALLOWED_ACE
		if err := windows.GetAce(acl, i, &ace); err != nil {
			return err
		}
		if ace.Header.AceType == windows.ACCESS_DENIED_ACE_TYPE {
			continue
		}
		if ace.Header.AceType != windows.ACCESS_ALLOWED_ACE_TYPE {
			return errors.New("private operation DACL cannot be verified")
		}
		sid := (*windows.SID)(unsafe.Pointer(&ace.SidStart))
		// Read access is authority-bearing too. Do not reuse the package guard's
		// foreign-write-only policy for confidential operation state.
		if ace.Mask != 0 && !sid.Equals(user.User.Sid) && !sid.Equals(admins) && !sid.Equals(system) {
			return errors.New("private operation state grants access to another principal")
		}
	}
	return nil
}

func nativePrivateRunnerCheck(f *os.File, info os.FileInfo) error {
	return nativePrivateCheck(f, info)
}

type journalRenameInfo struct {
	Flags          uint32
	RootDirectory  windows.Handle
	FileNameLength uint32
	FileName       [1]uint16
}

// FILE_RENAME_INFORMATION_EX has the same layout on Windows amd64 and arm64:
// flags at 0, an aligned HANDLE at 8, length at 16 and UTF-16 name at 20.
// These compile-time checks also guard cross-compilation for either target.
var (
	_ [unsafe.Offsetof(journalRenameInfo{}.RootDirectory) - 8]byte
	_ [8 - unsafe.Offsetof(journalRenameInfo{}.RootDirectory)]byte
	_ [unsafe.Offsetof(journalRenameInfo{}.FileNameLength) - 16]byte
	_ [16 - unsafe.Offsetof(journalRenameInfo{}.FileNameLength)]byte
	_ [unsafe.Offsetof(journalRenameInfo{}.FileName) - 20]byte
	_ [20 - unsafe.Offsetof(journalRenameInfo{}.FileName)]byte
	_ [unsafe.Sizeof(journalRenameInfo{}) - 24]byte
	_ [24 - unsafe.Sizeof(journalRenameInfo{})]byte
)

func journalRenameError(err error) error {
	var status windows.NTStatus
	if errors.As(err, &status) {
		switch status {
		case windows.STATUS_INVALID_INFO_CLASS, windows.STATUS_NOT_IMPLEMENTED,
			windows.STATUS_NOT_SUPPORTED, windows.STATUS_INVALID_PARAMETER:
			return fmt.Errorf("journal atomic POSIX rename requires Windows 10 version 1709 (build 16299) or later and a supporting filesystem: %w", err)
		}
		return fmt.Errorf("rename journal relative to held root: %w", status.Errno())
	}
	return fmt.Errorf("rename journal relative to held root: %w", err)
}

// Use durablefile's write-through publication semantics, with both names rooted
// at held directory handles instead of re-traversing a substitutable pathname.
func nativeJournalReplace(root *RealRoot, tmp, target string) error {
	if err := validatePrivateRoot(root); err != nil {
		return err
	}
	f, err := root.root.OpenFile(tmp, os.O_RDWR, 0)
	if err != nil {
		return err
	}
	info, err := f.Stat()
	if err == nil {
		err = nativePrivateCheck(f, info)
	}
	if err != nil {
		f.Close()
		return err
	}
	// ReOpenFile preserves the held file identity while obtaining the exact
	// DELETE/write-through rights needed by FileRenameInfoEx and flush.
	h, err := reopenJournalFile(f, windows.DELETE|windows.GENERIC_WRITE|windows.READ_CONTROL, windows.FILE_FLAG_WRITE_THROUGH)
	if err != nil {
		f.Close()
		return fmt.Errorf("reopen journal publication handle: %w", err)
	}
	defer windows.CloseHandle(h)
	if err := f.Close(); err != nil {
		return err
	}
	dir, err := root.root.Open(".")
	if err != nil {
		return err
	}
	defer dir.Close()
	to, err := windows.UTF16FromString(target)
	if err != nil {
		return err
	}
	var layout journalRenameInfo
	buffer := make([]byte, int(unsafe.Offsetof(layout.FileName))+len(to)*2)
	rename := (*journalRenameInfo)(unsafe.Pointer(&buffer[0]))
	rename.Flags = windows.FILE_RENAME_REPLACE_IF_EXISTS | windows.FILE_RENAME_POSIX_SEMANTICS
	rename.RootDirectory = windows.Handle(dir.Fd())
	rename.FileNameLength = uint32((len(to) - 1) * 2)
	copy(unsafe.Slice(&rename.FileName[0], len(to)), to)
	// The native information class accepts the held RootDirectory directly;
	// the Win32 information wrapper rejects this relative-root publication on
	// Windows. FileRenameInformationEx is 65 in FILE_INFORMATION_CLASS.
	// This requires Windows 10 version 1709 (build 16299) or later. Unsupported
	// systems fail publication; there is deliberately no ordinary-rename fallback.
	// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/wdm/ne-wdm-_file_information_class
	const fileRenameInformationEx = 65
	var status windows.IO_STATUS_BLOCK
	if err := windows.NtSetInformationFile(h, &status, &buffer[0], uint32(len(buffer)), fileRenameInformationEx); err != nil {
		return journalRenameError(err)
	}
	if err := windows.FlushFileBuffers(h); err != nil {
		return fmt.Errorf("%w: %w", durablefile.ErrReplaced, err)
	}
	return nil
}

func nativeJournalSync(root *RealRoot) error { return validatePrivateRoot(root) }

func (l *OperationLock) Close() error {
	if l.file == nil {
		return errors.New("operation lock already closed")
	}
	f := l.file
	l.file = nil
	return f.Close()
}
