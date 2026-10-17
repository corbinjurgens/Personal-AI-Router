// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package durablefile replaces a file's contents so that a crash leaves either
// the previous bytes or the complete new bytes, never a torn file.
//
// WriteFile writes a uniquely named temporary sibling, syncs it, replaces the
// target with it, and then makes the replacement durable: on POSIX it syncs the
// parent directory, and on Windows the replace itself is issued with
// MOVEFILE_WRITE_THROUGH. Success means the replacement is durable as far as the
// platform documents. It is not a power-loss proof: the guarantees are those of
// fsync, rename and MoveFileEx, a filesystem or storage layer may weaken them,
// and tests that kill a process exercise crash ordering rather than power loss.
//
// Any failure before the replacement removes the temporary file and leaves the
// target byte-identical. A directory-sync failure happens after the
// replacement: the error is returned and the target already holds the complete
// new bytes. The parent directory must already exist.
//
// Replacer exposes each filesystem step as a replaceable function so callers in
// other modules can inject a failure at any boundary from their own tests.
package durablefile

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

const (
	replaceMaxAttempts = 10
	replaceRetryBase   = 50 * time.Millisecond
)

// File is the temporary file a Replacer writes, syncs and closes.
type File interface {
	io.Writer
	Sync() error
	Close() error
	Name() string
}

// Replacer holds the filesystem steps WriteFile performs. A nil field uses the
// platform implementation, so the zero value is ready to use and a test
// replaces only the step it wants to fail.
type Replacer struct {
	// CreateTemp exclusively creates a new file in dir whose name starts with
	// base and is unique, with the given permissions.
	CreateTemp func(dir, base string, perm os.FileMode) (File, error)
	// Replace atomically moves tmp over target. It is retried while
	// IsTransient reports its error.
	Replace func(tmp, target string) error
	// SyncDir makes a completed replacement in dir durable. It is a no-op on
	// Windows, where Replace writes through.
	SyncDir func(dir string) error
	// IsTransient reports whether a Replace error may clear on retry.
	IsTransient func(err error) bool
	// Sleep waits between Replace attempts.
	Sleep func(d time.Duration)
}

// WriteFile durably replaces path with data using the platform steps.
func WriteFile(path string, data []byte, perm os.FileMode) error {
	return Replacer{}.WriteFile(path, data, perm)
}

// WriteFile durably replaces path with data. Errors name the failed step.
func (r Replacer) WriteFile(path string, data []byte, perm os.FileMode) error {
	r = r.withDefaults()
	dir := filepath.Dir(path)
	f, err := r.CreateTemp(dir, filepath.Base(path), perm)
	if err != nil {
		return fmt.Errorf("durablefile: create temp in %s: %w", dir, err)
	}
	tmp := f.Name()
	if _, err := f.Write(data); err != nil {
		abandon(f, tmp)
		return fmt.Errorf("durablefile: write %s: %w", tmp, err)
	}
	if err := f.Sync(); err != nil {
		abandon(f, tmp)
		return fmt.Errorf("durablefile: sync %s: %w", tmp, err)
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("durablefile: close %s: %w", tmp, err)
	}
	if err := r.replaceWithRetry(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("durablefile: replace %s with %s: %w", path, tmp, err)
	}
	if err := r.SyncDir(dir); err != nil {
		return fmt.Errorf("durablefile: sync directory %s: %w", dir, err)
	}
	return nil
}

func (r Replacer) replaceWithRetry(tmp, target string) error {
	var err error
	for attempt := 1; attempt <= replaceMaxAttempts; attempt++ {
		err = r.Replace(tmp, target)
		if err == nil {
			return nil
		}
		if !r.IsTransient(err) || attempt == replaceMaxAttempts {
			break
		}
		r.Sleep(replaceRetryBase * time.Duration(attempt))
	}
	return err
}

func (r Replacer) withDefaults() Replacer {
	if r.CreateTemp == nil {
		r.CreateTemp = createTemp
	}
	if r.Replace == nil {
		r.Replace = replaceFile
	}
	if r.SyncDir == nil {
		r.SyncDir = syncDir
	}
	if r.IsTransient == nil {
		r.IsTransient = isTransientReplaceError
	}
	if r.Sleep == nil {
		r.Sleep = time.Sleep
	}
	return r
}

// abandon closes a temporary file that failed before its own Close and
// removes it; the original failure is the one reported.
func abandon(f File, tmp string) {
	_ = f.Close()
	_ = os.Remove(tmp)
}

func createTemp(dir, base string, perm os.FileMode) (File, error) {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return nil, fmt.Errorf("mint temp name: %w", err)
	}
	name := filepath.Join(dir, base+"."+hex.EncodeToString(b[:])+".tmp")
	return os.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_EXCL, perm)
}
