// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
	"nvpair-shared/buildapply"
)

func configureBuildApplyDetached(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: windows.CREATE_NEW_PROCESS_GROUP | windows.DETACHED_PROCESS, HideWindow: true}
}

func createBuildApplyDirectory(path string) error {
	if err := buildapply.ValidateLocalPath(path); err != nil {
		return err
	}
	user, err := windows.GetCurrentProcessToken().GetTokenUser()
	if err != nil {
		return err
	}
	sd, err := windows.SecurityDescriptorFromString("D:P(A;OICI;FA;;;" + user.User.Sid.String() + ")(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)")
	if err != nil {
		return err
	}
	name, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return err
	}
	sa := windows.SecurityAttributes{Length: uint32(unsafe.Sizeof(windows.SecurityAttributes{})), SecurityDescriptor: sd}
	return windows.CreateDirectory(name, &sa)
}

// Windows does not expose a generally usable directory FlushFileBuffers on
// these same-user application folders. Each file is synced; journal publication
// still uses the existing native durablefile primitive. Rename reconciliation
// always re-reads actual directory identities after interruption.
func syncBuildApplyDirectory(path string) error { return nil }

// Rename the verified directory handle relative to the retained parent. No
// replace flag is used: an existing destination must never be overwritten.
func renameBuildApplyDirectory(root *os.Root, from, to string, want os.FileInfo) error {
	dir, err := root.Open(".")
	if err != nil {
		return err
	}
	defer dir.Close()
	name, err := windows.NewNTUnicodeString(from)
	if err != nil {
		return err
	}
	attrs := windows.OBJECT_ATTRIBUTES{Length: uint32(unsafe.Sizeof(windows.OBJECT_ATTRIBUTES{})), RootDirectory: windows.Handle(dir.Fd()), ObjectName: name, Attributes: windows.OBJ_CASE_INSENSITIVE | windows.OBJ_DONT_REPARSE}
	var handle windows.Handle
	var status windows.IO_STATUS_BLOCK
	if err := windows.NtCreateFile(&handle, windows.DELETE|windows.FILE_READ_ATTRIBUTES|windows.SYNCHRONIZE, &attrs, &status, nil, 0,
		windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE, windows.FILE_OPEN,
		windows.FILE_DIRECTORY_FILE|windows.FILE_OPEN_REPARSE_POINT|windows.FILE_SYNCHRONOUS_IO_NONALERT, 0, 0); err != nil {
		return fmt.Errorf("open held directory rename source: %w", err)
	}
	file := os.NewFile(uintptr(handle), from)
	defer file.Close()
	actual, err := file.Stat()
	if err != nil || !actual.IsDir() || !os.SameFile(want, actual) {
		return errors.New("held directory rename source was substituted")
	}
	type renameInformation struct {
		Flags          uint32
		RootDirectory  windows.Handle
		FileNameLength uint32
		FileName       [1]uint16
	}
	var layout renameInformation
	// FILE_RENAME_INFORMATION_EX uses these offsets on amd64 and arm64.
	if unsafe.Offsetof(layout.RootDirectory) != 8 || unsafe.Offsetof(layout.FileNameLength) != 16 || unsafe.Offsetof(layout.FileName) != 20 || unsafe.Sizeof(layout) != 24 {
		return errors.New("unsupported native directory rename layout")
	}
	target, err := windows.UTF16FromString(to)
	if err != nil {
		return err
	}
	buffer := make([]byte, int(unsafe.Offsetof(layout.FileName))+len(target)*2)
	info := (*renameInformation)(unsafe.Pointer(&buffer[0]))
	info.RootDirectory = windows.Handle(dir.Fd())
	info.FileNameLength = uint32((len(target) - 1) * 2)
	copy(unsafe.Slice(&info.FileName[0], len(target)), target)
	// FileRenameInformationEx (65) requires Windows 10 1709/build 16299.
	// Unsupported kernels/filesystems fail closed; no path-rename fallback.
	if err := windows.NtSetInformationFile(handle, &status, &buffer[0], uint32(len(buffer)), 65); err != nil {
		return fmt.Errorf("atomic held-directory rename requires Windows 10 1709/build 16299 or later and a supporting filesystem: %w", err)
	}
	return nil
}

type nativeBuildApplyExit struct{ handle windows.Handle }

func isBuildApplyLockBusy(err error) bool { return errors.Is(err, windows.ERROR_LOCK_VIOLATION) }

func openBuildApplyExit(ctx context.Context, want buildapply.RunnerReservation) (*nativeBuildApplyExit, error) {
	if want.PID <= 0 || int64(want.PID) > 1<<31-1 || want.Identity.Created == "" {
		return nil, errors.New("missing exact process lifetime")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	h, err := windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(want.PID))
	if err != nil {
		return nil, err
	}
	closeOnError := func(err error) (*nativeBuildApplyExit, error) { windows.CloseHandle(h); return nil, err }
	var created, exited, kernel, user windows.Filetime
	if err := windows.GetProcessTimes(h, &created, &exited, &kernel, &user); err != nil {
		return closeOnError(err)
	}
	buffer := make([]uint16, 32768)
	size := uint32(len(buffer))
	if err := windows.QueryFullProcessImageName(h, 0, &buffer[0], &size); err != nil {
		return closeOnError(err)
	}
	identity := buildapply.ProcessIdentity{Executable: windows.UTF16ToString(buffer[:size]), Created: fmt.Sprintf("%08x%08x", created.HighDateTime, created.LowDateTime)}
	if !sameBuildApplyImage(identity.Executable, want.Identity.Executable) || identity.Created != want.Identity.Created {
		return closeOnError(errors.New("process handle belongs to a different lifetime"))
	}
	return &nativeBuildApplyExit{handle: h}, nil
}

func (p *nativeBuildApplyExit) Wait(ctx context.Context) error {
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		state, err := windows.WaitForSingleObject(p.handle, 100)
		if err != nil {
			return err
		}
		switch state {
		case windows.WAIT_OBJECT_0:
			return nil
		case uint32(windows.WAIT_TIMEOUT):
		default:
			return errors.New("independent process exit is unavailable")
		}
	}
}
func (p *nativeBuildApplyExit) Close() error { return windows.CloseHandle(p.handle) }

func proveBuildApplyExit(ctx context.Context, want buildapply.RunnerReservation) error {
	p, err := openBuildApplyExit(ctx, want)
	if err != nil {
		return err
	}
	defer p.Close()
	return p.Wait(ctx)
}

func buildApplyDirectoryFileSnapshot(f *os.File) (string, os.FileInfo, error) {
	var info windows.ByHandleFileInformation
	if err := windows.GetFileInformationByHandle(windows.Handle(f.Fd()), &info); err != nil {
		return "", nil, err
	}
	// File.Stat loads the file ID from this same handle. Lstat may defer it
	// until SameFile and thereby read a replacement at the old pathname.
	held, err := f.Stat()
	if err != nil {
		return "", nil, err
	}
	if !held.IsDir() {
		return "", nil, errors.New("directory identity is not a directory")
	}
	return fmt.Sprintf("%x:%x:%x", info.VolumeSerialNumber, info.FileIndexHigh, info.FileIndexLow), held, nil
}

func readBuildApplyNativeMetadata(path string) (buildApplyMetadata, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return buildApplyMetadata{}, err
	}
	data, ok := info.Sys().(*syscall.Win32FileAttributeData)
	if !ok {
		return buildApplyMetadata{}, errors.New("native metadata unavailable")
	}
	streams, err := readBuildApplyStreams(path)
	if err != nil {
		return buildApplyMetadata{}, err
	}
	return buildApplyMetadata{Mode: uint32(info.Mode() & (os.ModeType | os.ModePerm)), Modified: info.ModTime().UnixNano(), Created: data.CreationTime.Nanoseconds(), Attributes: data.FileAttributes, Streams: streams}, nil
}

func writeBuildApplyNativeMetadata(path string, m buildApplyMetadata) error {
	for name, data := range m.Streams {
		if !validBuildApplyStreamName(name) {
			return errors.New("unsupported stream name")
		}
		f, err := os.OpenFile(path+":"+name, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err != nil {
			return err
		}
		n, writeErr := f.Write(data)
		if writeErr == nil && n != len(data) {
			writeErr = io.ErrShortWrite
		}
		if err := errors.Join(writeErr, f.Sync(), f.Close()); err != nil {
			return err
		}
	}
	name, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return err
	}
	access := uint32(windows.FILE_WRITE_ATTRIBUTES)
	if os.FileMode(m.Mode)&os.ModeDir == 0 {
		access |= windows.GENERIC_WRITE
	}
	h, err := windows.CreateFile(name, access, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE, nil, windows.OPEN_EXISTING, windows.FILE_FLAG_BACKUP_SEMANTICS|windows.FILE_FLAG_OPEN_REPARSE_POINT, 0)
	if err != nil {
		return err
	}
	defer windows.CloseHandle(h)
	created, modified := windows.NsecToFiletime(m.Created), windows.NsecToFiletime(m.Modified)
	if err := windows.SetFileTime(h, &created, nil, &modified); err != nil {
		return err
	}
	if err := windows.SetFileAttributes(name, m.Attributes); err != nil {
		return err
	}
	if os.FileMode(m.Mode)&os.ModeDir == 0 {
		return windows.FlushFileBuffers(h)
	}
	return nil
}

var buildApplyKernel32 = windows.NewLazySystemDLL("kernel32.dll")
var buildApplyFindFirstStream = buildApplyKernel32.NewProc("FindFirstStreamW")
var buildApplyFindNextStream = buildApplyKernel32.NewProc("FindNextStreamW")

type buildApplyStreamData struct {
	Size int64
	Name [296]uint16
}

func validBuildApplyStreamName(name string) bool {
	if name == "" || len(name) > 255 {
		return false
	}
	for _, r := range name {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '.' || r == '-' || r == '_') {
			return false
		}
	}
	return true
}

func readBuildApplyStreams(path string) (map[string][]byte, error) {
	name, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return nil, err
	}
	var data buildApplyStreamData
	result, _, callErr := buildApplyFindFirstStream.Call(uintptr(unsafe.Pointer(name)), 0, uintptr(unsafe.Pointer(&data)), 0)
	streams := map[string][]byte{}
	if windows.Handle(result) == windows.InvalidHandle {
		if errors.Is(callErr, windows.ERROR_HANDLE_EOF) {
			return streams, nil
		}
		return nil, errors.Join(errors.New("native stream inventory unavailable"), callErr)
	}
	h := windows.Handle(result)
	defer windows.FindClose(h)
	total := int64(0)
	for {
		stream := windows.UTF16ToString(data.Name[:])
		if stream != "::$DATA" {
			if !strings.HasPrefix(stream, ":") || !strings.HasSuffix(stream, ":$DATA") {
				return nil, errors.New("unsupported native stream")
			}
			stream = strings.TrimSuffix(strings.TrimPrefix(stream, ":"), ":$DATA")
			if !validBuildApplyStreamName(stream) || data.Size < 0 || data.Size > 1<<20 || total+data.Size > 1<<20 {
				return nil, errors.New("native stream metadata exceeds bound")
			}
			f, err := os.Open(path + ":" + stream)
			if err != nil {
				return nil, err
			}
			value, readErr := io.ReadAll(io.LimitReader(f, data.Size+1))
			closeErr := f.Close()
			if err := errors.Join(readErr, closeErr); err != nil {
				return nil, err
			}
			if int64(len(value)) != data.Size {
				return nil, errors.New("native stream changed")
			}
			streams[stream] = value
			total += data.Size
		}
		ok, _, err := buildApplyFindNextStream.Call(uintptr(h), uintptr(unsafe.Pointer(&data)))
		if ok == 0 {
			if errors.Is(err, windows.ERROR_HANDLE_EOF) {
				break
			}
			return nil, errors.Join(errors.New("native stream enumeration failed"), err)
		}
	}
	return streams, nil
}

func verifyBuildApplyBundleReleased(ctx context.Context, path string) error {
	real, err := buildapply.OpenRealRoot(path)
	if err != nil {
		return err
	}
	defer real.Close()
	r, err := os.OpenRoot(path)
	if err != nil {
		return err
	}
	defer r.Close()
	err = fs.WalkDir(r.FS(), ".", func(name string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		p, err := windows.UTF16PtrFromString(filepath.Join(path, filepath.FromSlash(name)))
		if err != nil {
			return err
		}
		h, err := windows.CreateFile(p, windows.FILE_READ_ATTRIBUTES|windows.DELETE, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE, nil, windows.OPEN_EXISTING, windows.FILE_FLAG_OPEN_REPARSE_POINT, 0)
		if err != nil {
			return err
		}
		return windows.CloseHandle(h)
	})
	if err != nil {
		return err
	}
	return real.Revalidate()
}
