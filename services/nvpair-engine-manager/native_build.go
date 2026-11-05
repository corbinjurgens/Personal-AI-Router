// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
)

type nativeBuildHashEntry struct {
	info   os.FileInfo
	digest string
}

// nativeBuildHashes implements index decision B2: full-file hashes are memoized
// only in memory by resolved path, size and mtime. Object identity is rechecked
// even on a hit; no persisted hash store or path enters the build fingerprint.
var nativeBuildHashes = struct {
	gate  chan struct{}
	files map[string]nativeBuildHashEntry
}{gate: make(chan struct{}, 1), files: make(map[string]nativeBuildHashEntry)}

func nativeBuildID(ctx context.Context, st *engineState) (string, error) {
	bin, err := nativeBuildExecutable(st)
	if err != nil {
		return "", err
	}
	executable, err := captureSource(bin, false)
	if err != nil {
		return "", err
	}
	selected, _, err := nativeBuildSnapshotExpected(ctx, executable, servingFileAccess{}, nil)
	return selected.Build, err
}

func nativeBuildExecutable(st *engineState) (string, error) {
	if st == nil {
		return "", errors.New("missing native runtime")
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.plat == nil || st.manifest == nil || st.manifest.FamilyOrDefault() != "llamacpp" {
		return "", errors.New("selected runtime is not llama.cpp")
	}
	if st.binPath != "" {
		return st.binPath, nil
	}
	bin, err := resolvePlaceholders(st.plat.Runtime.Bin, st.pathVars())
	return expandPath(bin), err
}

// The executable's directory owns the bounded library set for both managed and
// override builds. Expected objects remain request-local across replay checks.
func nativeBuildSnapshotExpected(ctx context.Context, executable sourceEvidence, access servingFileAccess, expected map[string]sourceEvidence) (modelDeviceRuntime, map[string]sourceEvidence, error) {
	selected := modelDeviceRuntime{Root: filepath.Dir(executable.target)}
	if err := validateSource(executable); err != nil {
		return selected, nil, err
	}
	digest, err := nativeBuildHashFileExpected(ctx, access, executable)
	if err != nil {
		return selected, nil, err
	}
	selected, libraries, err := modelDeviceLibrarySnapshotBound(ctx, selected, access, expected)
	if err != nil {
		return selected, nil, err
	}
	type library struct {
		Name   string `json:"name"`
		SHA256 string `json:"sha256"`
	}
	items := make([]library, 0, len(selected.Libraries))
	for _, item := range selected.Libraries {
		items = append(items, library{item.Name, item.SHA256})
	}
	sort.Slice(items, func(i, j int) bool { return items[i].Name < items[j].Name })
	data, err := json.Marshal(struct {
		Version    string    `json:"version"`
		Platform   string    `json:"platform"`
		Executable string    `json:"executable"`
		Libraries  []library `json:"libraries"`
	}{"pair-native-build-v1", runtime.GOOS + "/" + runtime.GOARCH, digest, items})
	if err != nil {
		return selected, nil, err
	}
	if err := validateSource(executable); err != nil {
		return selected, nil, err
	}
	sum := sha256.Sum256(data)
	selected.Build = "sha256:" + hex.EncodeToString(sum[:])
	return selected, libraries, ctx.Err()
}

// Both executable and library hashing use this checked core. Library callers
// must first enforce modelDeviceLibraryPath's name and confinement checks.
func nativeBuildHashFileExpected(ctx context.Context, access servingFileAccess, expected sourceEvidence) (string, error) {
	f, err := access.checkedOpen(ctx, expected.configured, expected.target, expected.info)
	if err != nil {
		return "", err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() < 0 || info.Size() > 2<<30 {
		return "", errors.New("invalid native build file")
	}
	// Always check the supplied opener's actual object before consulting the
	// memo. Waiting for another fill remains cancellable under the read budget.
	select {
	case nativeBuildHashes.gate <- struct{}{}:
		defer func() { <-nativeBuildHashes.gate }()
	case <-ctx.Done():
		return "", ctx.Err()
	}
	if entry, ok := nativeBuildHashes.files[expected.target]; ok && sameSource(entry.info, info, false) {
		if err := validateSource(expected); err != nil {
			return "", err
		}
		return entry.digest, ctx.Err()
	}
	h := sha256.New()
	if _, err := io.Copy(h, io.NewSectionReader(servingContextReaderAt{ctx: ctx, reader: f, expected: expected}, 0, info.Size())); err != nil {
		return "", err
	}
	if err := validateSource(expected); err != nil {
		return "", err
	}
	digest := hex.EncodeToString(h.Sum(nil))
	// captureSource froze this object's ID, including Windows' lazily loaded
	// file identity. A fresh Stat result could learn a replacement's ID later.
	nativeBuildHashes.files[expected.target] = nativeBuildHashEntry{info: expected.info, digest: digest}
	return digest, ctx.Err()
}
