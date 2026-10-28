// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/shirou/gopsutil/v4/process"
	"golang.org/x/sys/unix"
	"nvpair-shared/buildapply"
)

func configureBuildApplyDetached(cmd *exec.Cmd) { cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true} }
func createBuildApplyDirectory(path string) error {
	if err := buildapply.ValidateLocalPath(path); err != nil {
		return err
	}
	return os.Mkdir(path, 0o700)
}
func syncBuildApplyDirectory(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	return errors.Join(f.Sync(), f.Close())
}

func renameBuildApplyDirectory(root *os.Root, from, to string, want os.FileInfo) error {
	return root.Rename(from, to)
}

type nativeBuildApplyExit struct{ reservation buildapply.RunnerReservation }

func isBuildApplyLockBusy(err error) bool {
	return errors.Is(err, syscall.EWOULDBLOCK) || errors.Is(err, syscall.EAGAIN)
}

func openBuildApplyExit(ctx context.Context, want buildapply.RunnerReservation) (*nativeBuildApplyExit, error) {
	actual, err := inspectBuildApplyProcess(ctx, want.PID)
	if err != nil {
		return nil, err
	}
	if actual != want {
		return nil, errors.New("process belongs to a different lifetime")
	}
	return &nativeBuildApplyExit{reservation: want}, nil
}
func (p *nativeBuildApplyExit) Wait(ctx context.Context) error {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		exists, err := process.PidExistsWithContext(ctx, int32(p.reservation.PID))
		if err != nil {
			return err
		}
		if !exists {
			return nil
		}
		actual, err := inspectBuildApplyProcess(ctx, p.reservation.PID)
		if err != nil {
			return err
		}
		if actual != p.reservation {
			return errors.New("process identity changed before exit proof")
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}
func (p *nativeBuildApplyExit) Close() error { return nil }
func proveBuildApplyExit(ctx context.Context, want buildapply.RunnerReservation) error {
	p, err := openBuildApplyExit(ctx, want)
	if err != nil {
		return err
	}
	defer p.Close()
	return p.Wait(ctx)
}

func buildApplyDirectoryFileSnapshot(f *os.File) (string, os.FileInfo, error) {
	info, err := f.Stat()
	if err != nil {
		return "", nil, err
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || !info.IsDir() {
		return "", nil, errors.New("directory identity unavailable")
	}
	return fmt.Sprintf("%x:%x", stat.Dev, stat.Ino), info, nil
}

func readBuildApplyNativeMetadata(path string) (buildApplyMetadata, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return buildApplyMetadata{}, err
	}
	m := buildApplyMetadata{Mode: uint32(info.Mode() & (os.ModePerm | os.ModeType)), Modified: info.ModTime().UnixNano(), Xattrs: map[string][]byte{}}
	n, err := unix.Llistxattr(path, nil)
	if err != nil {
		return m, err
	}
	if n > 1<<20 {
		return m, errors.New("metadata name bound exceeded")
	}
	names := make([]byte, n)
	n, err = unix.Llistxattr(path, names)
	if err != nil {
		return m, err
	}
	total := 0
	for _, name := range strings.Split(string(names[:n]), "\x00") {
		if name == "" {
			continue
		}
		size, err := unix.Lgetxattr(path, name, nil)
		if err != nil {
			return m, err
		}
		if size > 1<<20 {
			return m, errors.New("metadata value bound exceeded")
		}
		total += len(name) + size
		if total > 1<<20 {
			return m, errors.New("metadata aggregate bound exceeded")
		}
		value := make([]byte, size)
		size, err = unix.Lgetxattr(path, name, value)
		if err != nil {
			return m, err
		}
		m.Xattrs[name] = value[:size]
	}
	return m, nil
}

func writeBuildApplyNativeMetadata(path string, m buildApplyMetadata) error {
	for name, value := range m.Xattrs {
		if err := unix.Lsetxattr(path, name, value, 0); err != nil {
			return err
		}
	}
	if os.FileMode(m.Mode)&os.ModeSymlink == 0 {
		if err := os.Chmod(path, os.FileMode(m.Mode).Perm()); err != nil {
			return err
		}
	}
	ts := unix.NsecToTimespec(m.Modified)
	if err := unix.UtimesNanoAt(unix.AT_FDCWD, filepath.Clean(path), []unix.Timespec{ts, ts}, unix.AT_SYMLINK_NOFOLLOW); err != nil {
		return err
	}
	return nil
}

func verifyBuildApplyBundleReleased(ctx context.Context, path string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	r, err := buildapply.OpenRealRoot(path)
	if err != nil {
		return err
	}
	defer r.Close()
	processes, err := process.ProcessesWithContext(ctx)
	if err != nil {
		return err
	}
	if len(processes) > 65536 {
		return errors.New("process inventory bound exceeded")
	}
	for _, p := range processes {
		if err := ctx.Err(); err != nil {
			return err
		}
		image, err := p.ExeWithContext(ctx)
		if err != nil {
			continue
		}
		relative, err := filepath.Rel(path, image)
		if err == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
			return errors.New("a selected-bundle image is still running")
		}
	}
	return r.Revalidate()
}
