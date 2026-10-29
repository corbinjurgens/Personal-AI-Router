// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package artifactbinding owns the cheap sampled local-file change detector.
// Same-stat changes outside the samples are not full integrity verification.
package artifactbinding

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"

	"nvpair-shared/engines"
	"nvpair-shared/modelconfig"
)

const SampleSize = 64 << 10

type File interface {
	io.ReaderAt
	Stat() (os.FileInfo, error)
	Close() error
}

// Evidence retains the ORIGINAL object across every open/read/recheck.
type Evidence struct {
	Configured, Target string
	Info               os.FileInfo
	Size, Mtime        int64
}

func Capture(path string) (Evidence, error) {
	path, err := filepath.Abs(path)
	if err != nil {
		return Evidence{}, err
	}
	target, err := Resolve(path)
	if err != nil {
		return Evidence{}, err
	}
	info, err := os.Stat(path)
	if err != nil {
		return Evidence{}, err
	}
	if !info.Mode().IsRegular() {
		return Evidence{}, fmt.Errorf("unexpected source type")
	}
	if !os.SameFile(info, info) {
		return Evidence{}, fmt.Errorf("source identity is unavailable")
	}
	return Evidence{path, target, info, info.Size(), info.ModTime().UnixNano()}, nil
}

func Same(a, b os.FileInfo, directory bool) bool {
	if a == nil || b == nil || a.IsDir() != directory || b.IsDir() != directory {
		return false
	}
	if !directory && (!a.Mode().IsRegular() || !b.Mode().IsRegular()) {
		return false
	}
	return os.SameFile(a, b) && a.Size() == b.Size() && a.ModTime() == b.ModTime()
}

func Validate(e Evidence) error {
	if e.Info == nil || !filepath.IsAbs(e.Configured) || !filepath.IsAbs(e.Target) || !Same(e.Info, e.Info, false) || e.Size < 0 || e.Info.Size() != e.Size || e.Info.ModTime().UnixNano() != e.Mtime {
		return fmt.Errorf("missing source evidence")
	}
	current, err := Capture(e.Configured)
	if err != nil || current.Target != e.Target || !Same(e.Info, current.Info, false) {
		return fmt.Errorf("source changed")
	}
	return nil
}

type Access struct {
	Open        func(string) (File, error)
	SampleBytes *uint64
}

func (a Access) CheckedOpen(ctx context.Context, e Evidence) (File, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if e.Info == nil || !e.Info.Mode().IsRegular() {
		return nil, fmt.Errorf("missing expected source evidence")
	}
	if err := Validate(e); err != nil {
		return nil, err
	}
	open := a.Open
	if open == nil {
		open = func(path string) (File, error) { return os.Open(path) }
	}
	f, err := open(e.Configured)
	if err != nil {
		return nil, err
	}
	opened, statErr := f.Stat()
	if statErr != nil || !Same(e.Info, opened, false) || Validate(e) != nil || ctx.Err() != nil {
		f.Close()
		return nil, fmt.Errorf("opened serving source changed")
	}
	return f, nil
}

type Reader struct {
	Context  context.Context
	File     File
	Expected Evidence
}

func (r Reader) ReadAt(p []byte, offset int64) (int, error) {
	if err := r.Context.Err(); err != nil {
		return 0, err
	}
	opened, err := r.File.Stat()
	if err != nil || !Same(r.Expected.Info, opened, false) || Validate(r.Expected) != nil {
		return 0, fmt.Errorf("serving source changed before reading")
	}
	return r.File.ReadAt(p, offset)
}

func (a Access) Sample(ctx context.Context, e Evidence, allowEmpty bool) (*engines.ServingBinding, error) {
	if e.Info == nil || e.Target == "" {
		return nil, fmt.Errorf("missing expected source evidence")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !e.Info.Mode().IsRegular() || (!allowEmpty && e.Size <= 0) {
		return nil, fmt.Errorf("serving artifact must be a nonempty regular file")
	}
	if err := Validate(e); err != nil {
		return nil, fmt.Errorf("serving source changed before sampling")
	}
	f, err := a.CheckedOpen(ctx, e)
	if err != nil {
		return nil, fmt.Errorf("cannot open serving artifact")
	}
	defer f.Close()
	before, err := f.Stat()
	if err != nil || !Same(e.Info, before, false) {
		return nil, fmt.Errorf("serving artifact must be a nonempty regular file")
	}
	h := sha256.New()
	sample := func(offset, size int64) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		n, err := io.Copy(h, io.NewSectionReader(Reader{ctx, f, e}, offset, size))
		if a.SampleBytes != nil {
			*a.SampleBytes += uint64(n)
		}
		if err == nil && n != size {
			return io.ErrUnexpectedEOF
		}
		return err
	}
	if before.Size() <= 2*SampleSize {
		err = sample(0, before.Size())
	} else {
		err = sample(0, SampleSize)
		if err == nil {
			err = sample(before.Size()-SampleSize, SampleSize)
		}
	}
	if err != nil {
		return nil, fmt.Errorf("cannot sample serving artifact")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	after, err := f.Stat()
	if err != nil || !Same(e.Info, after, false) || Validate(e) != nil {
		return nil, fmt.Errorf("serving artifact changed while sampling")
	}
	return &engines.ServingBinding{SourcePath: e.Configured, ResolvedTarget: e.Target, SizeBytes: before.Size(), MtimeNS: strconv.FormatInt(before.ModTime().UnixNano(), 10), SampleSHA256: hex.EncodeToString(h.Sum(nil))}, nil
}

// Summaries preserves the exact historical ordered JSON/SHA projection,
// including nil/empty input returning a non-nil empty output slice.
func Summaries(bindings []modelconfig.ArtifactBinding) []modelconfig.ArtifactSummary {
	ordered := make([]modelconfig.ArtifactSummary, 0, len(bindings))
	for _, binding := range bindings {
		data, err := json.Marshal(binding)
		if err != nil {
			return nil
		}
		sum := sha256.Sum256(data)
		ordered = append(ordered, modelconfig.ArtifactSummary{Role: binding.Role, Identity: hex.EncodeToString(sum[:]), Size: binding.Size})
	}
	return ordered
}
