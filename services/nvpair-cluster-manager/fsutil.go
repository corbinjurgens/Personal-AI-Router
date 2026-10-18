// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"nvpair-shared/durablefile"
)

// Tests inject write and deletion failures at their respective mutation seams.
var (
	clusterFileReplacer durablefile.Replacer
	removeFile          = os.Remove
)

const (
	atomicWriteMaxAttempts = 10
	atomicWriteRetryBase   = 50 * time.Millisecond
)

// atomicWrite durably replaces path. Adopt a landed replacement even when its
// durability is uncertain, so callers do not roll memory back behind disk.
func atomicWrite(path string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("create dir %s: %w", dir, err)
	}
	if err := clusterFileReplacer.WriteFile(path, data, perm); err != nil {
		if !errors.Is(err, durablefile.ErrReplaced) {
			return err
		}
		slog.Warn("cluster file replacement landed with uncertain durability", "path", path)
	}
	return nil
}

// removeFileWithRetry deletes path, tolerating the brief sharing conflicts that
// Windows reports while a sibling clustertrust reader has the file open. A
// missing path is already removed and therefore succeeds.
func removeFileWithRetry(path string) error {
	var removeErr error
	for attempt := 1; attempt <= atomicWriteMaxAttempts; attempt++ {
		removeErr = removeFile(path)
		if removeErr == nil || os.IsNotExist(removeErr) {
			return nil
		}
		if !isTransientReplaceError(removeErr) || attempt == atomicWriteMaxAttempts {
			break
		}
		time.Sleep(atomicWriteRetryBase * time.Duration(attempt))
	}
	return removeErr
}

// isTransientReplaceError reports whether err is a mutation conflict that may
// clear once a concurrent reader releases the file. Windows
// ERROR_ACCESS_DENIED (5) and ERROR_SHARING_VIOLATION (32) are matched by
// errno only on Windows; elsewhere we match the platform error text Go surfaces.
func isTransientReplaceError(err error) bool {
	if err == nil {
		return false
	}
	if runtime.GOOS == "windows" {
		var errno syscall.Errno
		if errors.As(err, &errno) && (errno == 5 || errno == 32) {
			return true
		}
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "access is denied") ||
		strings.Contains(msg, "sharing violation") ||
		strings.Contains(msg, "being used by another process")
}
