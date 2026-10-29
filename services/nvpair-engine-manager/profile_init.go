// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"nvpair-shared/durablefile"
)

// offlineEnvironmentGuard prevents Windows' roaming fallback from selecting a
// different profile than the desktop. Ordinary runtime resolution is unchanged.
func offlineEnvironmentGuard() error {
	if runtime.GOOS == "windows" {
		base, ok := os.LookupEnv("LOCALAPPDATA")
		if !ok || base == "" || !filepath.IsAbs(base) {
			return errors.New("LOCALAPPDATA must be set to an absolute Local base")
		}
	}
	return nil
}

// initializeFreshProfile assumes the operator holds an exclusive offline interval.
// Failure deliberately leaves allocations and any published bytes for review.
func initializeFreshProfile(current, previous string, replace durablefile.Replacer) (bool, error) {
	return initializeFreshProfileWithLstat(current, previous, replace, os.Lstat)
}

func initializeFreshProfileWithLstat(current, previous string, replace durablefile.Replacer, lstat func(string) (os.FileInfo, error)) (bool, error) {
	return initializeFreshProfileWithFS(current, previous, replace, lstat, os.Mkdir, readFreshProfile)
}

func readFreshProfile(store modelDownloadStore) error {
	markers, copies, complete, recovery, err := store.loadJournal()
	if err != nil || len(markers) != 0 || len(copies) != 0 || !complete || recovery {
		return errors.New("fresh journal readback failed")
	}
	return nil
}

func initializeFreshProfileWithFS(current, previous string, replace durablefile.Replacer, lstat func(string) (os.FileInfo, error), mkdir func(string, os.FileMode) error, readback func(modelDownloadStore) error) (bool, error) {
	if current == "" || previous == "" || !filepath.IsAbs(current) || !filepath.IsAbs(previous) || filepath.Clean(current) != current || filepath.Clean(previous) != previous || current == previous || filepath.Dir(current) == current {
		return false, errors.New("invalid profile paths")
	}
	for _, path := range []string{current, previous} {
		_, err := lstat(path)
		if err == nil || !errors.Is(err, os.ErrNotExist) {
			return false, fmt.Errorf("profile is not definitely absent: %s", path)
		}
	}
	// Inspect every ancestor, even above an existing real directory: Lstat of
	// that directory alone cannot reveal a symlink higher in its path.
	previousExisting := false
	for path := filepath.Dir(previous); ; path = filepath.Dir(path) {
		info, err := lstat(path)
		if err == nil {
			if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
				return false, fmt.Errorf("invalid previous ancestor: %s", path)
			}
			previousExisting = true
			if filepath.Dir(path) == path {
				break
			}
			continue
		}
		if previousExisting || !errors.Is(err, os.ErrNotExist) || filepath.Dir(path) == path {
			return false, fmt.Errorf("inspect previous ancestor %s: %w", path, err)
		}
	}
	var missing []string
	foundExisting := false
	for path := filepath.Dir(current); ; path = filepath.Dir(path) {
		info, err := lstat(path)
		if err == nil {
			if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
				return false, fmt.Errorf("invalid ancestor: %s", path)
			}
			foundExisting = true
			if filepath.Dir(path) == path {
				break
			}
			continue
		}
		if !errors.Is(err, os.ErrNotExist) || filepath.Dir(path) == path {
			return false, fmt.Errorf("inspect ancestor %s: %w", path, err)
		}
		if foundExisting {
			return false, fmt.Errorf("ancestor disappeared: %s", path)
		}
		missing = append(missing, path)
	}
	var created []string
	for i := len(missing) - 1; i >= 0; i-- {
		if err := mkdir(missing[i], 0700); err != nil {
			return false, err
		}
		created = append(created, missing[i])
	}
	if err := mkdir(current, 0700); err != nil {
		return false, err
	}
	created = append(created, current)
	install := filepath.Join(current, "engine-bin")
	if err := mkdir(install, 0700); err != nil {
		return false, err
	}
	created = append(created, install)
	store := newModelDownloadStore(install)
	store.replace = replace
	landed, uncertain, err := store.save(nil, nil, true, false)
	if err != nil || !landed || uncertain {
		return false, fmt.Errorf("journal publication failed (landed=%t, uncertain=%t): %v", landed, uncertain, err)
	}
	unsupported := false
	for i := len(created) - 1; i >= 0; i-- {
		if err := replace.SyncCreatedDir(created[i]); err != nil {
			if runtime.GOOS == "windows" && errors.Is(err, durablefile.ErrDirectoryDurabilityUnsupported) {
				unsupported = true
				continue
			}
			return false, err
		}
	}
	if err := readback(store); err != nil {
		return false, err
	}
	return unsupported, nil
}
