// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"nvpair-engine-manager/internal/artifactbinding"

	"nvpair-shared/engines"
)

const servingSmallFileLimit = 1 << 20
const servingSampleSize = artifactbinding.SampleSize

// servingFileAccess is instance-owned. All artifact reads, including binding
// samples, go through the same opener; the default never starts native work.
type servingArtifactFile = artifactbinding.File

type servingReadUsage struct {
	SampleBytes uint64
	HeaderBytes uint64
	SourceBytes uint64
}

type servingFileAccess struct {
	open      func(string) (servingArtifactFile, error)
	openDir   func(string) (servingDirectory, error)
	usage     *servingReadUsage
	scan      *modelArtifactScan
	recognize func(context.Context, string) (engineInventory, error)
}

type servingDirectory interface {
	Stat() (os.FileInfo, error)
	ReadDir(int) ([]os.DirEntry, error)
	Close() error
}

// sourceEvidence is request-local and never serialized or refreshed during a recheck.
type sourceEvidence struct {
	configured, target string
	info               os.FileInfo
	directory          bool
	size               int64
	mtime              int64
}

type selectorEvidence struct {
	bytes   []byte
	binding *engines.ServingBinding
	file    sourceEvidence
}

// Ordered records describe one exact selection, not permission to discover new
// descendants. Adapters populate this private carrier; it has no wire format.
type sourceSnapshot struct {
	engine, run, configuration, root string
	recognition                      []byte
	inventory                        *engineInventory
	selectors                        []selectorEvidence
	directories                      []sourceEvidence
	artifacts                        []modelFile
}

func captureSource(path string, directory bool) (sourceEvidence, error) {
	if !directory {
		e, err := artifactbinding.Capture(path)
		return sourceEvidence{configured: e.Configured, target: e.Target, info: e.Info, size: e.Size, mtime: e.Mtime}, err
	}
	path, err := filepath.Abs(path)
	if err != nil {
		return sourceEvidence{}, err
	}
	target, err := resolveExistingPath(path)
	if err != nil {
		return sourceEvidence{}, err
	}
	info, err := os.Stat(path)
	if err != nil {
		return sourceEvidence{}, err
	}
	if !info.IsDir() {
		return sourceEvidence{}, fmt.Errorf("unexpected source type")
	}
	// Windows Stat loads the file ID lazily. Freeze it while capturing the
	// original object so a later SameFile cannot learn a replacement's ID.
	if !os.SameFile(info, info) {
		return sourceEvidence{}, fmt.Errorf("source identity is unavailable")
	}
	return sourceEvidence{configured: path, target: target, info: info,
		directory: true, size: info.Size(), mtime: info.ModTime().UnixNano()}, nil
}

func sameSource(a, b os.FileInfo, directory bool) bool {
	return artifactbinding.Same(a, b, directory)
}

func regularSourceEvidence(e sourceEvidence) artifactbinding.Evidence {
	return artifactbinding.Evidence{Configured: e.configured, Target: e.target, Info: e.info, Size: e.size, Mtime: e.mtime}
}

func (a servingFileAccess) bindingAccess() artifactbinding.Access {
	var counter *uint64
	if a.usage != nil {
		counter = &a.usage.SampleBytes
	}
	return artifactbinding.Access{Open: a.openFile, SampleBytes: counter}
}

func validateSource(e sourceEvidence) error {
	if !e.directory {
		return artifactbinding.Validate(regularSourceEvidence(e))
	}
	if e.info == nil || !filepath.IsAbs(e.configured) || !filepath.IsAbs(e.target) ||
		!sameSource(e.info, e.info, e.directory) || e.size < 0 ||
		e.info.Size() != e.size || e.info.ModTime().UnixNano() != e.mtime {
		return fmt.Errorf("missing source evidence")
	}
	current, err := captureSource(e.configured, e.directory)
	if err != nil || current.target != e.target || !sameSource(e.info, current.info, e.directory) {
		return fmt.Errorf("source changed")
	}
	return nil
}

func (a servingFileAccess) readDirectory(ctx context.Context, e sourceEvidence, limit int) ([]os.DirEntry, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !e.directory || limit < 0 {
		return nil, fmt.Errorf("invalid directory evidence")
	}
	if err := validateSource(e); err != nil {
		return nil, err
	}
	f, err := a.openDirectory(e.configured)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var entries []os.DirEntry
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		opened, err := f.Stat()
		if err != nil || !sameSource(e.info, opened, true) || validateSource(e) != nil {
			return nil, fmt.Errorf("directory changed before enumeration")
		}
		// Read at most one entry beyond the caller's bound, in cancellable
		// batches. Avoid ReadDir(0), which means an unbounded enumeration.
		n := 128
		if remaining := limit - len(entries); remaining < n {
			n = remaining + 1
		}
		batch, err := f.ReadDir(n)
		if err != nil && err != io.EOF {
			return nil, err
		}
		entries = append(entries, batch...)
		if len(entries) > limit {
			return nil, fmt.Errorf("directory exceeds limit")
		}
		if err == io.EOF {
			break
		}
		if len(batch) == 0 {
			return nil, io.ErrNoProgress
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	after, err := f.Stat()
	if err != nil || !sameSource(e.info, after, true) || validateSource(e) != nil {
		return nil, fmt.Errorf("directory changed during enumeration")
	}
	return entries, nil
}

type servingContextReaderAt struct {
	ctx      context.Context
	reader   servingArtifactFile
	expected sourceEvidence
}

func (r servingContextReaderAt) ReadAt(p []byte, offset int64) (int, error) {
	return (artifactbinding.Reader{Context: r.ctx, File: r.reader, Expected: regularSourceEvidence(r.expected)}).ReadAt(p, offset)
}

func (a servingFileAccess) openFile(path string) (servingArtifactFile, error) {
	if a.open != nil {
		return a.open(path)
	}
	return os.Open(path)
}

func (a servingFileAccess) openDirectory(path string) (servingDirectory, error) {
	if a.openDir != nil {
		return a.openDir(path)
	}
	return os.Open(path)
}

// Check the actual opened object before any content read. The caller retains
// its original metadata across reopenings, including the small-source read.
func (a servingFileAccess) checkedOpen(ctx context.Context, path, target string, expected os.FileInfo) (servingArtifactFile, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if expected == nil {
		return nil, fmt.Errorf("missing expected source evidence")
	}
	e := sourceEvidence{configured: path, target: target, info: expected, size: expected.Size(), mtime: expected.ModTime().UnixNano()}
	return a.bindingAccess().CheckedOpen(ctx, regularSourceEvidence(e))
}

// servingSource reads only small launch inputs. Model artifacts use the bounded
// binding reader below, never this helper or a full-file content hash.
func servingSource(path string) ([]byte, *engines.ServingBinding, error) {
	return (servingFileAccess{}).source(context.Background(), path)
}

func (a servingFileAccess) source(ctx context.Context, path string) ([]byte, *engines.ServingBinding, error) {
	return a.sourceLimited(ctx, path, servingSmallFileLimit, false)
}

func (a servingFileAccess) sourceLimited(ctx context.Context, path string, limit int64, allowEmpty bool) ([]byte, *engines.ServingBinding, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	e, err := captureSource(path, false)
	if err != nil {
		return nil, nil, fmt.Errorf("native serving source is unavailable")
	}
	return a.sourceExpected(ctx, e, limit, allowEmpty)
}

// Rechecks require the original metadata. Missing records must never recapture
// authority from whatever happens to be at the path now.
func (a servingFileAccess) sourceExpected(ctx context.Context, e sourceEvidence, limit int64, allowEmpty bool) ([]byte, *engines.ServingBinding, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if e.directory || validateSource(e) != nil {
		return nil, nil, fmt.Errorf("invalid expected source evidence")
	}
	if limit < 0 || e.size > limit || (!allowEmpty && e.size == 0) {
		return nil, nil, fmt.Errorf("invalid native serving source size")
	}
	binding, err := a.sampleExpected(ctx, e.configured, e.target, e.info, allowEmpty)
	if err != nil {
		return nil, nil, err
	}
	if binding.SizeBytes > limit {
		return nil, nil, fmt.Errorf("native serving source exceeds the small-file limit")
	}
	f, err := a.checkedOpen(ctx, e.configured, e.target, e.info)
	if err != nil {
		return nil, nil, fmt.Errorf("cannot open native serving source")
	}
	defer f.Close()
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	data, err := io.ReadAll(io.NewSectionReader(servingContextReaderAt{ctx: ctx, reader: f, expected: e}, 0, binding.SizeBytes))
	if a.usage != nil {
		a.usage.SourceBytes += uint64(len(data))
	}
	if err != nil || int64(len(data)) > limit {
		return nil, nil, fmt.Errorf("cannot read bounded native serving source")
	}
	opened, openedErr := f.Stat()
	if openedErr != nil || !sameSource(e.info, opened, false) || validateSource(e) != nil {
		return nil, nil, fmt.Errorf("native serving source changed while reading")
	}
	after, err := a.sampleExpected(ctx, e.configured, e.target, e.info, allowEmpty)
	if err != nil || *binding != *after || int64(len(data)) != binding.SizeBytes {
		return nil, nil, fmt.Errorf("native serving source changed while reading")
	}
	return data, binding, nil
}

// The asserted full-file digest is deliberately separate from this cheap local
// change detector. Same-stat modifications outside the samples are not detected.
func servingFileBinding(path string) (*engines.ServingBinding, error) {
	return (servingFileAccess{}).binding(context.Background(), path)
}

func (a servingFileAccess) binding(ctx context.Context, path string) (*engines.ServingBinding, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	e, err := captureSource(path, false)
	if err != nil {
		return nil, err
	}
	return a.bindingExpected(ctx, e.configured, e.target, e.info)
}

func (a servingFileAccess) bindingExpected(ctx context.Context, path, expectedTarget string, expected os.FileInfo) (*engines.ServingBinding, error) {
	return a.sampleExpected(ctx, path, expectedTarget, expected, false)
}

func (a servingFileAccess) sampleExpected(ctx context.Context, path, expectedTarget string, expected os.FileInfo, allowEmpty bool) (*engines.ServingBinding, error) {
	if expected == nil || expectedTarget == "" {
		return nil, fmt.Errorf("missing expected source evidence")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	source, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("cannot resolve serving artifact")
	}
	e := sourceEvidence{configured: source, target: expectedTarget, info: expected, size: expected.Size(), mtime: expected.ModTime().UnixNano()}
	return a.bindingAccess().Sample(ctx, regularSourceEvidence(e), allowEmpty)
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
