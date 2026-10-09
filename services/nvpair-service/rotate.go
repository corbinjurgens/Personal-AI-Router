// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// defaultLogMaxSize is the size at which the broker log rotates.
const defaultLogMaxSize = 10 << 20

// rotatingLog is an append-only log file that moves itself to "<path>.1" once
// it would grow past maxSize, keeping exactly one old generation. Files are
// created owner-only: the broker's stderr carries node names, addresses, and
// model names even after its own redaction.
type rotatingLog struct {
	mu      sync.Mutex
	path    string
	maxSize int64
	f       *os.File
	size    int64
}

func openRotatingLog(path string, maxSize int64) (*rotatingLog, error) {
	if maxSize <= 0 {
		maxSize = defaultLogMaxSize
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, fmt.Errorf("create log directory: %w", err)
	}
	l := &rotatingLog{path: path, maxSize: maxSize}
	if err := l.open(os.O_APPEND); err != nil {
		return nil, err
	}
	return l, nil
}

func (l *rotatingLog) open(mode int) error {
	f, err := os.OpenFile(l.path, os.O_CREATE|os.O_WRONLY|mode, 0o600)
	if err != nil {
		return err
	}
	st, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return err
	}
	l.f, l.size = f, st.Size()
	return nil
}

// Write appends p, rotating first when p would take the file past maxSize. A
// single write larger than maxSize still lands whole, in a fresh file.
func (l *rotatingLog) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.f == nil {
		return 0, errors.New("log closed")
	}
	if l.size > 0 && l.size+int64(len(p)) > l.maxSize {
		if err := l.rotate(); err != nil {
			return 0, err
		}
	}
	n, err := l.f.Write(p)
	l.size += int64(n)
	return n, err
}

func (l *rotatingLog) rotate() error {
	_ = l.f.Close()
	l.f = nil
	old := l.path + ".1"
	// Removed first: os.Rename replaces an existing file everywhere Go runs,
	// but an old generation held open by a reader on Windows would make the
	// replace fail where a delete-then-rename only fails the delete.
	_ = os.Remove(old)
	if err := os.Rename(l.path, old); err != nil {
		// Keep logging into the current file rather than losing lines.
		return l.open(os.O_APPEND)
	}
	return l.open(os.O_TRUNC)
}

func (l *rotatingLog) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.f == nil {
		return nil
	}
	err := l.f.Close()
	l.f = nil
	return err
}
