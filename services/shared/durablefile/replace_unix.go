// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package durablefile

import (
	"fmt"
	"os"
	"path/filepath"
)

// SyncCreatedDir confirms the entry for an already-created directory.
func (r Replacer) SyncCreatedDir(path string) error {
	parent := filepath.Dir(path)
	if err := r.withDefaults().SyncDir(parent); err != nil {
		return fmt.Errorf("durablefile: sync directory %s: %w", parent, err)
	}
	return nil
}

func replaceFile(tmp, target string) error {
	return os.Rename(tmp, target)
}

// syncDir fsyncs dir so the rename's directory entry survives a crash.
func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	if err := d.Sync(); err != nil {
		_ = d.Close()
		return err
	}
	return d.Close()
}

// isTransientReplaceError is false on POSIX: rename does not fail on readers
// that hold the target open.
func isTransientReplaceError(error) bool {
	return false
}
