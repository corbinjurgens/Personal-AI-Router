//go:build windows

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

func nativeLocalPath(path string) error {
	if !filepath.IsAbs(path) || strings.HasPrefix(path, `\\`) || strings.Contains(path[len(filepath.VolumeName(path)):], ":") {
		return errors.New("network or device path")
	}
	root, err := windows.UTF16PtrFromString(filepath.VolumeName(path) + `\`)
	if err != nil {
		return err
	}
	if windows.GetDriveType(root) != windows.DRIVE_FIXED {
		return errors.New("bundle is not on a fixed local volume")
	}
	return nil
}

func nativePathCheck(path string, info os.FileInfo, owned bool) error {
	if err := nativeInfoCheck(info, owned); err != nil {
		return err
	}
	if strings.HasPrefix(path, `\\`) || strings.Contains(path[len(filepath.VolumeName(path)):], ":") {
		return errors.New("network, device or alternate-stream path")
	}
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return err
	}
	attrs, err := windows.GetFileAttributes(p)
	if err != nil {
		return err
	}
	if attrs&windows.FILE_ATTRIBUTE_REPARSE_POINT != 0 {
		return errors.New("reparse point")
	}
	volume := filepath.VolumeName(path) + `\`
	vp, err := windows.UTF16PtrFromString(volume)
	if err != nil {
		return err
	}
	if windows.GetDriveType(vp) != windows.DRIVE_FIXED {
		return errors.New("bundle is not on a fixed local volume")
	}
	h, err := windows.CreateFile(p, windows.FILE_READ_ATTRIBUTES|windows.READ_CONTROL, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE, nil, windows.OPEN_EXISTING, windows.FILE_FLAG_BACKUP_SEMANTICS|windows.FILE_FLAG_OPEN_REPARSE_POINT, 0)
	if err != nil {
		return err
	}
	defer windows.CloseHandle(h)
	return nativeHandleCheck(h, info, owned)
}

func nativeInfoCheck(info os.FileInfo, owned bool) error {
	data, ok := info.Sys().(*syscall.Win32FileAttributeData)
	if !ok || data.FileAttributes&windows.FILE_ATTRIBUTE_REPARSE_POINT != 0 {
		return errors.New("unavailable file attributes or reparse point")
	}
	return nil
}

func nativeOpenedCheck(f *os.File, info os.FileInfo) error {
	return nativeHandleCheck(windows.Handle(f.Fd()), info, true)
}

func nativeHandleCheck(h windows.Handle, info os.FileInfo, owned bool) error {
	var data windows.ByHandleFileInformation
	if err := windows.GetFileInformationByHandle(h, &data); err != nil {
		return err
	}
	if data.FileAttributes&windows.FILE_ATTRIBUTE_REPARSE_POINT != 0 || info.Mode().IsRegular() && data.NumberOfLinks != 1 {
		return errors.New("linked package entry")
	}
	if owned {
		sd, err := windows.GetSecurityInfo(h, windows.SE_FILE_OBJECT, windows.OWNER_SECURITY_INFORMATION|windows.DACL_SECURITY_INFORMATION)
		if err != nil {
			return err
		}
		owner, _, err := sd.Owner()
		if err != nil {
			return err
		}
		token := windows.GetCurrentProcessToken()
		user, err := token.GetTokenUser()
		if err != nil {
			return err
		}
		defaultOwner, err := currentTokenOwner(token)
		if err != nil {
			return err
		}
		if !owner.Equals(user.User.Sid) && !owner.Equals(defaultOwner) {
			return errors.New("bundle is not owned by the current user")
		}
		acl, _, err := sd.DACL()
		if err != nil || acl == nil {
			return errors.New("bundle has no restrictive DACL")
		}
		admins, err := windows.CreateWellKnownSid(windows.WinBuiltinAdministratorsSid)
		if err != nil {
			return err
		}
		system, err := windows.CreateWellKnownSid(windows.WinLocalSystemSid)
		if err != nil {
			return err
		}
		const mutationRights = windows.GENERIC_WRITE | windows.GENERIC_ALL | windows.DELETE | windows.WRITE_DAC | windows.WRITE_OWNER | windows.FILE_WRITE_DATA | windows.FILE_APPEND_DATA | windows.FILE_WRITE_EA | windows.FILE_WRITE_ATTRIBUTES | 0x40
		for i := uint32(0); i < uint32(acl.AceCount); i++ {
			var ace *windows.ACCESS_ALLOWED_ACE
			if err := windows.GetAce(acl, i, &ace); err != nil {
				return err
			}
			if ace.Header.AceType == windows.ACCESS_DENIED_ACE_TYPE {
				continue
			}
			if ace.Header.AceType != windows.ACCESS_ALLOWED_ACE_TYPE {
				return errors.New("bundle DACL cannot be verified")
			}
			if ace.Mask&mutationRights != 0 {
				sid := (*windows.SID)(unsafe.Pointer(&ace.SidStart))
				if !sid.Equals(user.User.Sid) && !sid.Equals(admins) && !sid.Equals(system) {
					return errors.New("bundle is writable by another principal")
				}
			}
		}
	}
	return nil
}

func currentTokenOwner(token windows.Token) (*windows.SID, error) {
	var bytes uint32
	err := windows.GetTokenInformation(token, windows.TokenOwner, nil, 0, &bytes)
	if err != windows.ERROR_INSUFFICIENT_BUFFER || bytes == 0 || bytes > 4096 {
		return nil, errors.New("process token owner unavailable")
	}
	buffer := make([]byte, bytes)
	if err := windows.GetTokenInformation(token, windows.TokenOwner, &buffer[0], bytes, &bytes); err != nil {
		return nil, err
	}
	owner := *(**windows.SID)(unsafe.Pointer(&buffer[0]))
	if owner == nil || !owner.IsValid() {
		return nil, errors.New("invalid process token owner")
	}
	return owner.Copy()
}

func nativeTargetParent(target, parent string, bytes int64) error {
	for _, key := range []string{"ProgramFiles", "ProgramFiles(x86)", "SystemRoot", "ProgramData"} {
		if system := os.Getenv(key); system != "" && containsPath(system, target) {
			return errors.New("system-directory installation target is unsupported")
		}
	}
	for _, path := range []string{target, parent} {
		info, err := os.Stat(path)
		if err != nil {
			return err
		}
		if err := nativePathCheck(path, info, true); err != nil {
			return err
		}
	}
	// Opening with the required rights tests access without creating a file.
	p, err := windows.UTF16PtrFromString(parent)
	if err != nil {
		return err
	}
	const fileDeleteChild = 0x40
	// Directory FILE_ADD_FILE/FILE_ADD_SUBDIRECTORY share these access bits.
	h, err := windows.CreateFile(p, windows.FILE_WRITE_DATA|windows.FILE_APPEND_DATA|fileDeleteChild, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE, nil, windows.OPEN_EXISTING, windows.FILE_FLAG_BACKUP_SEMANTICS|windows.FILE_FLAG_OPEN_REPARSE_POINT, 0)
	if err != nil {
		return err
	}
	defer windows.CloseHandle(h)
	tp, err := windows.UTF16PtrFromString(target)
	if err != nil {
		return err
	}
	th, err := windows.CreateFile(tp, windows.DELETE|windows.FILE_READ_ATTRIBUTES, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE, nil, windows.OPEN_EXISTING, windows.FILE_FLAG_BACKUP_SEMANTICS|windows.FILE_FLAG_OPEN_REPARSE_POINT, 0)
	if err != nil {
		return err
	}
	defer windows.CloseHandle(th)
	var parentInfo, targetInfo windows.ByHandleFileInformation
	if err := windows.GetFileInformationByHandle(h, &parentInfo); err != nil {
		return err
	}
	if err := windows.GetFileInformationByHandle(th, &targetInfo); err != nil {
		return err
	}
	if parentInfo.VolumeSerialNumber != targetInfo.VolumeSerialNumber {
		return errors.New("target is not on its parent's volume")
	}
	var available, total, free uint64
	if err := windows.GetDiskFreeSpaceEx(p, &available, &total, &free); err != nil {
		return err
	}
	if bytes < 0 || uint64(bytes) > available {
		return errors.New("insufficient available target-parent space")
	}
	// A reparse/mount boundary is already refused on every component.
	return nil
}

func nativeProcessIdentity(pid int) (ProcessIdentity, error) {
	if pid <= 0 {
		return ProcessIdentity{}, errors.New("invalid GUI process")
	}
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return ProcessIdentity{}, err
	}
	defer windows.CloseHandle(h)
	var created, exited, kernel, user windows.Filetime
	if err := windows.GetProcessTimes(h, &created, &exited, &kernel, &user); err != nil {
		return ProcessIdentity{}, err
	}
	if exited.HighDateTime != 0 || exited.LowDateTime != 0 {
		return ProcessIdentity{}, errors.New("GUI process has exited")
	}
	buffer := make([]uint16, 32768)
	size := uint32(len(buffer))
	if err := windows.QueryFullProcessImageName(h, 0, &buffer[0], &size); err != nil {
		return ProcessIdentity{}, err
	}
	return ProcessIdentity{Executable: windows.UTF16ToString(buffer[:size]), Created: fmt.Sprintf("%08x%08x", created.HighDateTime, created.LowDateTime)}, nil
}
