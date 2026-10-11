// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/shirou/gopsutil/v4/process"
)

type serviceProcessIdentity struct {
	PID     int32
	Path    string
	Created int64
}

type serviceProcessInventory struct {
	mu    sync.Mutex
	paths []string
	known map[int32]serviceProcessIdentity
}

func canonicalServicePath(path string) (string, error) {
	path, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", err
	}
	path, err = filepath.Abs(path)
	if runtime.GOOS == "windows" {
		path = strings.ToLower(path)
	}
	return filepath.Clean(path), err
}

// Exe is already a full executable path. Do not require its filesystem entry
// to still exist: the recorded copied path remains authoritative after unlink.
func serviceExecutablePath(path string) (string, error) {
	if !filepath.IsAbs(path) {
		return "", fmt.Errorf("executable path is not absolute: %q", path)
	}
	if runtime.GOOS == "windows" {
		path = strings.ToLower(path)
	}
	return filepath.Clean(path), nil
}

func serviceInspectionError(ctx context.Context, pid int32, stage string, inspectionErr error) error {
	exists, existenceErr := process.PidExistsWithContext(ctx, pid)
	if existenceErr == nil && !exists {
		return nil
	}
	return fmt.Errorf("owned pid %d %s: %w", pid, stage, errors.Join(inspectionErr, existenceErr))
}

func (in *serviceProcessInventory) owns(path string) bool {
	for _, owned := range in.paths {
		if path == owned {
			return true
		}
	}
	return false
}

// scan uses executable identities independently of RPC replies and startup logs.
// A protected unrelated process cannot establish ownership; inspection failures
// for a previously identified fixture process are always returned.
func (in *serviceProcessInventory) scan() ([]serviceProcessIdentity, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return in.scanContext(ctx)
}

func (in *serviceProcessInventory) scanContext(ctx context.Context) ([]serviceProcessIdentity, error) {
	in.mu.Lock()
	defer in.mu.Unlock()
	pids, err := process.PidsWithContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("process inventory: %w", err)
	}
	var found []serviceProcessIdentity
	var failures []error
	for _, pid := range pids {
		_, known := in.known[pid]
		p, err := process.NewProcessWithContext(ctx, pid)
		if err != nil {
			if known && !errors.Is(err, process.ErrorProcessNotRunning) {
				failures = append(failures, fmt.Errorf("inspect owned pid %d: %w", pid, err))
			}
			continue
		}
		path, err := p.ExeWithContext(ctx)
		if err != nil {
			if known && !errors.Is(err, process.ErrorProcessNotRunning) {
				failures = append(failures, serviceInspectionError(ctx, pid, "executable", err))
			}
			continue
		}
		path, err = serviceExecutablePath(path)
		if err != nil {
			if known {
				failures = append(failures, serviceInspectionError(ctx, pid, "executable path", err))
			}
			continue
		}
		if !in.owns(path) {
			continue
		}
		created, err := p.CreateTimeWithContext(ctx)
		if err != nil {
			failures = append(failures, serviceInspectionError(ctx, pid, "creation time", err))
			continue
		}
		identity := serviceProcessIdentity{PID: pid, Path: path, Created: created}
		in.known[pid] = identity
		status, err := p.StatusWithContext(ctx)
		if err != nil {
			failures = append(failures, serviceInspectionError(ctx, pid, "status", err))
			continue
		}
		zombie := false
		for _, s := range status {
			if s == process.Zombie {
				zombie = true
			}
		}
		if !zombie {
			found = append(found, identity)
		}
	}
	if ctx.Err() != nil {
		failures = append(failures, ctx.Err())
	}
	return found, errors.Join(failures...)
}

// kill rechecks all identity fields immediately before targeting this PID.
func (in *serviceProcessInventory) kill(want serviceProcessIdentity) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return in.killContext(ctx, want)
}

func (in *serviceProcessInventory) killContext(ctx context.Context, want serviceProcessIdentity) error {
	p, err := process.NewProcessWithContext(ctx, want.PID)
	if errors.Is(err, process.ErrorProcessNotRunning) {
		return nil
	}
	if err != nil {
		return err
	}
	path, err := p.ExeWithContext(ctx)
	if errors.Is(err, process.ErrorProcessNotRunning) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("revalidate pid %d executable: %w", want.PID, err)
	}
	path, err = serviceExecutablePath(path)
	if err != nil {
		return err
	}
	created, err := p.CreateTimeWithContext(ctx)
	if err != nil {
		return err
	}
	if path != want.Path || created != want.Created || !in.owns(path) {
		return fmt.Errorf("refusing changed pid %d identity: path=%q created=%d, recorded=%+v", want.PID, path, created, want)
	}
	if err := p.KillWithContext(ctx); err != nil && !errors.Is(err, process.ErrorProcessNotRunning) {
		return err
	}
	return nil
}
