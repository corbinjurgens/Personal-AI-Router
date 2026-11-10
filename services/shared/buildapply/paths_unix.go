//go:build !windows

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"unsafe"
)

// mountTable is one read of the kernel's mount table. Callers read it once per
// resolve or validate call and check every path against it.
type mountTable []mountEntry

type mountEntry struct {
	path  string
	local bool
}

func readMountTable() (mountTable, error) {
	if runtime.GOOS == "darwin" {
		return readDarwinMounts()
	}
	if runtime.GOOS != "linux" {
		return nil, errors.New("native mount observation unavailable")
	}
	f, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 4096), 1<<20)
	var table mountTable
	total := 0
	for scanner.Scan() {
		total += len(scanner.Bytes())
		if total > 8<<20 {
			return nil, errors.New("oversized mount table")
		}
		fields := strings.Fields(scanner.Text())
		if len(fields) < 10 {
			return nil, errors.New("invalid mount table")
		}
		separator := 0
		for i, field := range fields {
			if field == "-" {
				separator = i
				break
			}
		}
		if separator == 0 || separator+1 >= len(fields) {
			return nil, errors.New("invalid mount table")
		}
		mount := strings.NewReplacer(`\040`, " ", `\011`, "\t", `\012`, "\n", `\134`, `\`).Replace(fields[4])
		switch fields[separator+1] {
		case "ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs", "overlay", "zfs", "f2fs":
			table = append(table, mountEntry{path: mount, local: true})
		default:
			table = append(table, mountEntry{path: mount})
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return table, nil
}

// readDarwinMounts uses getfsstat64(MNT_NOWAIT), which reads the cached kernel
// mount table. Calling statfs on the selected path first could contact a remote
// filesystem. A first call without a buffer returns the mount count.
func readDarwinMounts() (mountTable, error) {
	const recordBytes = 2168 // Darwin 64-bit statfs ABI, both supported CPUs.
	const maxMounts = 1024
	count, _, errno := syscall.Syscall6(347, 0, 0, 2, 0, 0, 0)
	if errno != 0 {
		return nil, errno
	}
	// Room for mounts added between the two calls. A full buffer may have
	// dropped entries, so it is refused rather than trusted.
	capacity := int(count) + 8
	if capacity > maxMounts {
		return nil, errors.New("oversized native mount table")
	}
	data := make([]byte, recordBytes*capacity)
	n, _, errno := syscall.Syscall6(347, uintptr(unsafe.Pointer(&data[0])), uintptr(len(data)), 2, 0, 0, 0)
	runtime.KeepAlive(data)
	if errno != 0 {
		return nil, errno
	}
	if int(n) >= capacity {
		return nil, errors.New("native mount table changed while reading")
	}
	table := make(mountTable, 0, n)
	for i := 0; i < int(n); i++ {
		record := data[i*recordBytes : (i+1)*recordBytes]
		end := bytes.IndexByte(record[88:1112], 0)
		if end < 0 {
			return nil, errors.New("invalid native mount record")
		}
		table = append(table, mountEntry{path: string(record[88 : 88+end]), local: binary.LittleEndian.Uint32(record[64:68])&0x1000 != 0})
	}
	return table, nil
}

// localPath reports whether the filesystem mounted over path is local. Later
// entries are mounted on top of earlier ones at the same point, so the last
// longest match wins. macOS volumes are normally case-insensitive, so a case
// variant of a mount point still names that mount.
func (t mountTable) localPath(path string) error {
	best := ""
	local := false
	for _, entry := range t {
		if mountContains(entry.path, path) && len(entry.path) >= len(best) {
			best = entry.path
			local = entry.local
		}
	}
	if best == "" || !local {
		return errors.New("unverified or nonlocal filesystem")
	}
	return nil
}

func mountContains(mount, path string) bool {
	if runtime.GOOS != "darwin" {
		return containsPath(mount, path)
	}
	mountParts := strings.FieldsFunc(mount, func(c rune) bool { return c == '/' })
	pathParts := strings.FieldsFunc(path, func(c rune) bool { return c == '/' })
	if !strings.HasPrefix(mount, "/") || !strings.HasPrefix(path, "/") || len(mountParts) > len(pathParts) {
		return false
	}
	for i, part := range mountParts {
		if !strings.EqualFold(part, pathParts[i]) {
			return false
		}
	}
	return true
}

func nativePathCheck(path string, info os.FileInfo, owned bool) error {
	if err := nativeInfoCheck(info, owned); err != nil {
		return err
	}
	if owned && info.IsDir() {
		var fs syscall.Statfs_t
		if err := syscall.Statfs(path, &fs); err != nil {
			return err
		}
		if networkFilesystem(fs) {
			return errors.New("network filesystem")
		}
	}
	return nil
}

// nativeLinkTarget reports where a symbolic link above a root leads.
func nativeLinkTarget(path string, info os.FileInfo) (string, bool, error) {
	if info.Mode()&os.ModeSymlink == 0 {
		return "", false, nil
	}
	target, err := os.Readlink(path)
	return target, err == nil, err
}

func nativeInfoCheck(info os.FileInfo, owned bool) error {
	s, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return errors.New("file identity unavailable")
	}
	if info.Mode().IsRegular() && s.Nlink != 1 {
		return errors.New("hard-linked package file")
	}
	if info.Mode().IsRegular() && info.Mode()&(os.ModeSetuid|os.ModeSetgid) != 0 {
		return errors.New("privilege-bearing package file")
	}
	if owned && (s.Uid != uint32(os.Geteuid()) || info.Mode().Perm()&0022 != 0) {
		return errors.New("ambiguous bundle ownership or permissions")
	}
	return nil
}

func nativeOpenedCheck(f *os.File, info os.FileInfo) error { return nativeInfoCheck(info, true) }

func nativeTargetParent(target, parent string, bytes int64) error {
	if bytes < 0 {
		return errors.New("invalid space requirement")
	}
	if runtime.GOOS == "darwin" && containsPath("/Volumes", target) {
		return errors.New("mounted-volume application target is unsupported")
	}
	a, err := os.Stat(target)
	if err != nil {
		return err
	}
	b, err := os.Stat(parent)
	if err != nil {
		return err
	}
	if err := nativePathCheck(target, a, true); err != nil {
		return err
	}
	if err := nativePathCheck(parent, b, false); err != nil {
		return err
	}
	as, aok := a.Sys().(*syscall.Stat_t)
	bs, bok := b.Sys().(*syscall.Stat_t)
	if !aok || !bok || as.Dev != bs.Dev {
		return errors.New("target is not on its parent's volume")
	}
	// A system-owned, writable /Applications parent is eligible without
	// changing its owner. Only the user, root and native administrators may
	// have mutation rights; arbitrary shared writable parents are refused.
	if bs.Uid != uint32(os.Geteuid()) && bs.Uid != 0 {
		return errors.New("target parent has ambiguous ownership")
	}
	trustedAdminGroup := bs.Uid == 0 && (bs.Gid == 0 || runtime.GOOS == "darwin" && bs.Gid == 80)
	if b.Mode().Perm()&0020 != 0 && !trustedAdminGroup {
		return errors.New("target parent is writable by another group")
	}
	if b.Mode().Perm()&0002 != 0 && (bs.Uid != 0 || b.Mode()&os.ModeSticky == 0) {
		return errors.New("target parent is writable by another user")
	}
	if err := syscall.Access(parent, 3); err != nil {
		return err
	}
	var fs syscall.Statfs_t
	if err := syscall.Statfs(parent, &fs); err != nil {
		return err
	}
	// Known network filesystem types. Unsupported platforms remain closed.
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		return errors.New("unsupported filesystem inspection")
	}
	if networkFilesystem(fs) {
		return errors.New("network filesystem")
	}
	if fs.Bsize <= 0 || uint64(bytes) > uint64(fs.Bavail)*uint64(fs.Bsize) {
		return errors.New("insufficient available target-parent space")
	}
	return nil
}

func networkFilesystem(fs syscall.Statfs_t) bool {
	if runtime.GOOS == "darwin" {
		return fs.Flags&0x1000 == 0
	} // MNT_LOCAL
	return fs.Type == 0x6969 || fs.Type == 0x517b || fs.Type == 0xff534d42 || fs.Type == 0xfe534d42
}

func nativeProcessIdentity(pid int) (ProcessIdentity, error) {
	if pid <= 0 {
		return ProcessIdentity{}, errors.New("invalid native process")
	}
	if runtime.GOOS == "darwin" {
		return darwinProcessIdentity(pid)
	}
	if runtime.GOOS != "linux" {
		return ProcessIdentity{}, errors.New("native process creation identity unavailable")
	}
	base := filepath.Join("/proc", strconv.Itoa(pid))
	read := func() (string, error) {
		data, err := os.ReadFile(filepath.Join(base, "stat"))
		if err != nil {
			return "", err
		}
		if len(data) > 16384 {
			return "", errors.New("oversized process record")
		}
		end := strings.LastIndexByte(string(data), ')')
		if end < 0 {
			return "", errors.New("invalid process record")
		}
		fields := strings.Fields(string(data[end+1:]))
		if len(fields) <= 19 {
			return "", errors.New("incomplete process record")
		}
		return fields[19], nil
	}
	created, err := read()
	if err != nil {
		return ProcessIdentity{}, err
	}
	image, err := os.Readlink(filepath.Join(base, "exe"))
	if err != nil {
		return ProcessIdentity{}, err
	}
	if err := ValidateLocalPath(image); err != nil {
		return ProcessIdentity{}, err
	}
	if _, err := checkRealAncestors(filepath.Dir(image)); err != nil {
		return ProcessIdentity{}, err
	}
	imageInfo, imageErr := os.Stat(image)
	mappedInfo, mappedErr := os.Stat(filepath.Join(base, "exe"))
	if imageErr != nil || mappedErr != nil || !os.SameFile(imageInfo, mappedInfo) {
		return ProcessIdentity{}, errors.New("running process image no longer matches its path")
	}
	after, err := read()
	if err != nil || created != after {
		return ProcessIdentity{}, fmt.Errorf("process changed during inspection")
	}
	if !filepath.IsAbs(image) || strings.HasSuffix(image, " (deleted)") {
		return ProcessIdentity{}, errors.New("process image is unavailable")
	}
	return ProcessIdentity{Executable: image, Created: created}, nil
}

// Darwin's proc_info and sysctl are read-only kernel interfaces. Keeping the
// bounded ABI calls here permits the paired Unix/Windows helper layout without
// starting ps/lsof or requiring cgo. No argv text is used as image authority.
func darwinProcessIdentity(pid int) (ProcessIdentity, error) {
	if runtime.GOOS != "darwin" || (runtime.GOARCH != "amd64" && runtime.GOARCH != "arm64") {
		return ProcessIdentity{}, errors.New("unsupported Darwin process ABI")
	}
	readCreation := func() (string, error) {
		// CTL_KERN, KERN_PROC, KERN_PROC_PID. The 64-bit extern_proc prefix
		// contains timeval at 0 and p_pid at 40, on both supported architectures.
		mib := [4]int32{1, 14, 1, int32(pid)}
		var data [4096]byte
		size := uintptr(len(data))
		_, _, errno := syscall.Syscall6(202, uintptr(unsafe.Pointer(&mib[0])), uintptr(len(mib)), uintptr(unsafe.Pointer(&data[0])), uintptr(unsafe.Pointer(&size)), 0, 0)
		runtime.KeepAlive(mib)
		runtime.KeepAlive(data)
		if errno != 0 {
			return "", errno
		}
		if size > uintptr(len(data)) || size < 44 || binary.LittleEndian.Uint32(data[40:44]) != uint32(pid) {
			return "", errors.New("invalid native process creation record")
		}
		seconds := binary.LittleEndian.Uint64(data[:8])
		micros := binary.LittleEndian.Uint32(data[8:12])
		if seconds == 0 || micros >= 1000000 {
			return "", errors.New("unknown native process creation time")
		}
		return fmt.Sprintf("%d:%d", seconds, micros), nil
	}
	created, err := readCreation()
	if err != nil {
		return ProcessIdentity{}, err
	}
	// SYS_proc_info, PROC_INFO_CALL_PIDINFO, PROC_PIDPATHINFO. The raw call
	// returns 0 on success (libproc's proc_pidpath derives the length with
	// strlen), so the copied path is bounded by its terminator instead.
	var buffer [4096]byte
	_, _, errno := syscall.Syscall6(336, 2, uintptr(pid), 11, 0, uintptr(unsafe.Pointer(&buffer[0])), uintptr(len(buffer)))
	runtime.KeepAlive(buffer)
	if errno != 0 {
		return ProcessIdentity{}, errno
	}
	end := bytes.IndexByte(buffer[:], 0)
	if end <= 0 {
		return ProcessIdentity{}, errors.New("native process image unavailable")
	}
	image := string(buffer[:end])
	if !filepath.IsAbs(image) {
		return ProcessIdentity{}, errors.New("invalid native image path")
	}
	after, err := readCreation()
	if err != nil || after != created {
		return ProcessIdentity{}, errors.New("native process lifetime changed")
	}
	return ProcessIdentity{Executable: image, Created: created}, nil
}
