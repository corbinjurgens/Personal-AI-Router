// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package buildapply

import (
	"errors"
	"os"
	"syscall"

	"golang.org/x/sys/unix"
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
	if err := unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB); err != nil {
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
	// O_EXCL refuses links before opening a pre-existing destination. Existing
	// locks are opened read/write only after an independent Lstat check.
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
	return f, err
}

func nativePrivateMkdir(parent *RealRoot, name string) error {
	return parent.root.Mkdir(name, 0o700)
}

func nativePrivateCheck(f *os.File, info os.FileInfo) error {
	want := os.FileMode(0o600)
	if info.IsDir() {
		want = 0o700
	}
	return nativePrivateModeCheck(f, info, want)
}

func nativePrivateRunnerCheck(f *os.File, info os.FileInfo) error {
	if !info.Mode().IsRegular() {
		return errors.New("runner is not a regular file")
	}
	// Explicit recovery can remove both a completed executable and a private
	// copy interrupted before its final executable-mode publication.
	mode := info.Mode().Perm()
	if mode != 0o600 && mode != 0o700 {
		return errors.New("runner requires private copy or executable permissions")
	}
	return nativePrivateModeCheck(f, info, mode)
}

func nativePrivateModeCheck(f *os.File, info os.FileInfo, want os.FileMode) error {
	if err := nativeOpenedCheck(f, info); err != nil {
		return err
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || stat.Uid != uint32(os.Geteuid()) || info.Mode()&(os.ModeSetuid|os.ModeSetgid|os.ModeSticky) != 0 {
		return errors.New("private operation state has ambiguous ownership")
	}
	if info.Mode().Perm() != want {
		return errors.New("private operation state requires owner-only permissions")
	}
	return nil
}

func nativePrivateCreatedFile(f *os.File) error {
	info, err := f.Stat()
	if err != nil {
		return err
	}
	return nativePrivateCheck(f, info)
}

func nativeJournalReplace(root *RealRoot, tmp, target string) error {
	if err := validatePrivateRoot(root); err != nil {
		return err
	}
	return root.root.Rename(tmp, target)
}

func nativeJournalSync(root *RealRoot) error {
	f, err := root.root.Open(".")
	if err != nil {
		return err
	}
	defer f.Close()
	return f.Sync()
}

func (l *OperationLock) Close() error {
	if l.file == nil {
		return errors.New("operation lock already closed")
	}
	f := l.file
	l.file = nil
	return f.Close()
}
