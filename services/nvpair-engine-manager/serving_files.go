// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"

	"nvpair-shared/engines"
)

const servingSmallFileLimit = 1 << 20
const servingSampleSize = 64 << 10

// servingSource reads only small launch inputs. Model artifacts use the bounded
// binding reader below, never this helper or a full-file content hash.
func servingSource(path string) ([]byte, *engines.ServingBinding, error) {
	binding, err := servingFileBinding(path)
	if err != nil {
		return nil, nil, err
	}
	if binding.SizeBytes > servingSmallFileLimit {
		return nil, nil, fmt.Errorf("native serving source exceeds the small-file limit")
	}
	f, err := os.Open(binding.SourcePath)
	if err != nil {
		return nil, nil, fmt.Errorf("cannot open native serving source")
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, servingSmallFileLimit+1))
	if err != nil || len(data) > servingSmallFileLimit {
		return nil, nil, fmt.Errorf("cannot read bounded native serving source")
	}
	after, err := servingFileBinding(path)
	if err != nil || *binding != *after || int64(len(data)) != binding.SizeBytes {
		return nil, nil, fmt.Errorf("native serving source changed while reading")
	}
	return data, binding, nil
}

// The asserted full-file digest is deliberately separate from this cheap local
// change detector. Same-stat modifications outside the samples are not detected.
func servingFileBinding(path string) (*engines.ServingBinding, error) {
	source, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("cannot resolve serving artifact")
	}
	target, err := filepath.EvalSymlinks(source)
	if err != nil {
		return nil, fmt.Errorf("serving artifact is missing or unresolved")
	}
	f, err := os.Open(source)
	if err != nil {
		return nil, fmt.Errorf("cannot open serving artifact")
	}
	defer f.Close()
	before, err := f.Stat()
	if err != nil || !before.Mode().IsRegular() || before.Size() <= 0 {
		return nil, fmt.Errorf("serving artifact must be a nonempty regular file")
	}
	h := sha256.New()
	if before.Size() <= 2*servingSampleSize {
		_, err = io.CopyN(h, f, before.Size())
	} else {
		_, err = io.CopyN(h, f, servingSampleSize)
		if err == nil {
			_, err = f.Seek(-servingSampleSize, io.SeekEnd)
		}
		if err == nil {
			_, err = io.CopyN(h, f, servingSampleSize)
		}
	}
	if err != nil {
		return nil, fmt.Errorf("cannot sample serving artifact")
	}
	after, err := f.Stat()
	current, statErr := os.Stat(source)
	resolved, resolveErr := filepath.EvalSymlinks(source)
	if err != nil || statErr != nil || resolveErr != nil || target != resolved ||
		!os.SameFile(before, after) || !os.SameFile(after, current) ||
		before.Size() != after.Size() || before.ModTime() != after.ModTime() ||
		after.Size() != current.Size() || after.ModTime() != current.ModTime() {
		return nil, fmt.Errorf("serving artifact changed while sampling")
	}
	return &engines.ServingBinding{SourcePath: source, ResolvedTarget: target,
		SizeBytes: before.Size(), MtimeNS: strconv.FormatInt(before.ModTime().UnixNano(), 10),
		SampleSHA256: hex.EncodeToString(h.Sum(nil))}, nil
}

// Each process gets a private generation. Recovery regenerates identical bytes
// from matching approved sources; generation names are not content identities.
func prepareServingSnapshot(root string, data []byte) (string, func(), error) {
	if len(data) == 0 || len(data) > servingSmallFileLimit {
		return "", nil, fmt.Errorf("invalid serving snapshot size")
	}
	if !filepath.IsAbs(root) {
		return "", nil, fmt.Errorf("serving snapshot root must be absolute")
	}
	if err := os.MkdirAll(root, 0o700); err != nil {
		return "", nil, fmt.Errorf("cannot create serving snapshot root")
	}
	dir, err := os.MkdirTemp(root, "generation-")
	if err != nil {
		return "", nil, fmt.Errorf("cannot create serving snapshot generation")
	}
	path := filepath.Join(dir, "models.ini")
	cleanup := func() {
		_ = os.Chmod(path, 0o600)
		_ = os.Remove(path)
		_ = os.Remove(dir)
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		cleanup()
		return "", nil, fmt.Errorf("cannot create serving snapshot")
	}
	_, writeErr := io.Copy(f, bytes.NewReader(data))
	syncErr := f.Sync()
	closeErr := f.Close()
	if writeErr != nil || syncErr != nil || closeErr != nil || os.Chmod(path, 0o400) != nil {
		cleanup()
		return "", nil, fmt.Errorf("cannot finish serving snapshot")
	}
	return path, cleanup, nil
}
