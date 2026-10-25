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

var errServiceProcessDeparted = errors.New("recorded fixture process departed")

type serviceProcessInventory struct {
	mu    sync.Mutex
	paths []string
	known map[int32]serviceProcessIdentity
	// scanError injects a test failure without hiding inspected identities.
	scanError error
	// enumerationError injects an empty failed enumeration before inspection.
	enumerationError error
	// recoveryProbe replaces the independent stopped-state probe in tests.
	recoveryProbe func(context.Context, int32) (bool, error)
	// recoveryIdentity replaces the live canonical identity read in tests.
	recoveryIdentity func(context.Context, int32) (serviceProcessIdentity, error)
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

func serviceInspectionErrorWithProbe(ctx context.Context, pid int32, stage string, inspectionErr error, probe func(context.Context, int32) (bool, error)) error {
	// An exiting process can remain enumerable briefly after inspection fails.
	// Only an independent, successful absence check resolves that ambiguity.
	// This can also resolve an earlier probe error; presence cannot erase it.
	probeCtx, cancel := context.WithTimeout(ctx, 250*time.Millisecond)
	defer cancel()
	ticker := time.NewTicker(25 * time.Millisecond)
	defer ticker.Stop()
	failures := []error{inspectionErr}
	for probeCtx.Err() == nil {
		exists, existenceErr := probe(probeCtx, pid)
		if existenceErr != nil {
			failures = append(failures, existenceErr)
		}
		if existenceErr == nil && !exists && probeCtx.Err() == nil {
			return nil
		}
		select {
		case <-probeCtx.Done():
		case <-ticker.C:
		}
	}
	failures = append(failures, ctx.Err())
	return fmt.Errorf("owned pid %d %s: %w", pid, stage, errors.Join(failures...))
}

func (in *serviceProcessInventory) owns(path string) bool {
	for _, owned := range in.paths {
		if path == owned {
			return true
		}
	}
	return false
}

func serviceProcessHasZombieStatus(status []string) bool {
	for _, state := range status {
		if state == process.Zombie {
			return true
		}
	}
	return false
}

// serviceStoppedPresence adapts stopped-state evidence to the presence shape
// serviceInspectionErrorWithProbe resolves. A proven stop counts as absent: a
// disappeared process or an enumerable zombie can no longer run fixture code,
// so neither an inspection failure nor the recorded identity survives it.
func serviceStoppedPresence(stopped func(context.Context, int32) (bool, error)) func(context.Context, int32) (bool, error) {
	return func(ctx context.Context, pid int32) (bool, error) {
		isStopped, err := stopped(ctx, pid)
		return !isStopped, err
	}
}

// Windows has no zombie state and no supported Status. Recovery also checks
// identity because a present PID may belong to a replacement process.
func serviceProcessStopped(ctx context.Context, pid int32) (bool, error) {
	var status func(context.Context, int32) ([]string, error)
	if runtime.GOOS != "windows" {
		status = func(ctx context.Context, pid int32) ([]string, error) {
			p, err := process.NewProcessWithContext(ctx, pid)
			if err != nil {
				return nil, err
			}
			return p.StatusWithContext(ctx)
		}
	}
	return serviceProcessStoppedWithProbes(ctx, pid, process.PidExistsWithContext, status)
}

func serviceProcessStoppedWithProbes(ctx context.Context, pid int32, exists func(context.Context, int32) (bool, error), status func(context.Context, int32) ([]string, error)) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	present, err := exists(ctx, pid)
	if err != nil {
		return false, fmt.Errorf("existence: %w", errors.Join(err, ctx.Err()))
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	if !present {
		return true, nil
	}
	if status == nil {
		return false, nil
	}
	states, err := status(ctx, pid)
	if err != nil {
		// ps reads the state by PID, so it fails when the process exits between
		// the presence probe and this read. Resolve that ambiguity only against
		// independently proven absence; presence cannot erase the failure.
		if unresolved := serviceInspectionErrorWithProbe(ctx, pid, "status", err, exists); unresolved != nil {
			return false, unresolved
		}
		return true, nil
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	return serviceProcessHasZombieStatus(states), nil
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
	if in.enumerationError != nil {
		return nil, fmt.Errorf("process inventory: %w", errors.Join(in.enumerationError, in.scanError, ctx.Err()))
	}
	pids, err := process.PidsWithContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("process inventory: %w", errors.Join(err, in.scanError, ctx.Err()))
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
				failures = append(failures, serviceInspectionErrorWithProbe(ctx, pid, "executable", err, serviceStoppedPresence(serviceProcessStopped)))
			}
			continue
		}
		path, err = serviceExecutablePath(path)
		if err != nil {
			if known {
				failures = append(failures, serviceInspectionErrorWithProbe(ctx, pid, "executable path", err, serviceStoppedPresence(serviceProcessStopped)))
			}
			continue
		}
		if !in.owns(path) {
			continue
		}
		created, err := p.CreateTimeWithContext(ctx)
		if err != nil {
			failures = append(failures, serviceInspectionErrorWithProbe(ctx, pid, "creation time", err, serviceStoppedPresence(serviceProcessStopped)))
			continue
		}
		identity := serviceProcessIdentity{PID: pid, Path: path, Created: created}
		in.known[pid] = identity
		zombie := false
		// Windows has no zombie state and gopsutil does not implement Status.
		if runtime.GOOS != "windows" {
			status, err := p.StatusWithContext(ctx)
			if err != nil {
				failures = append(failures, serviceInspectionErrorWithProbe(ctx, pid, "status", err, serviceStoppedPresence(serviceProcessStopped)))
				continue
			}
			zombie = serviceProcessHasZombieStatus(status)
		}
		if !zombie {
			found = append(found, identity)
		}
	}
	if ctx.Err() != nil {
		failures = append(failures, ctx.Err())
	}
	return found, errors.Join(errors.Join(failures...), in.scanError)
}

// recoveryContext includes recorded descendants missed by an incomplete scan.
// Only the same live lifetime is recovered; killContext still independently
// revalidates every identity immediately before signalling.
func (in *serviceProcessInventory) recoveryContext(ctx context.Context, owned []serviceProcessIdentity) ([]serviceProcessIdentity, error) {
	in.mu.Lock()
	defer in.mu.Unlock()
	seen := make(map[int32]bool, len(owned))
	for _, identity := range owned {
		seen[identity.PID] = true
	}
	probe := in.recoveryProbe
	if probe == nil {
		probe = serviceProcessStopped
	}
	inspect := in.recoveryIdentity
	if inspect == nil {
		inspect = serviceProcessIdentityAt
	}
	var failures []error
	for pid, identity := range in.known {
		if ctx.Err() != nil {
			failures = append(failures, ctx.Err())
			break
		}
		if seen[pid] {
			continue
		}
		if !in.owns(identity.Path) {
			failures = append(failures, fmt.Errorf("recorded pid %d is outside fixture paths: %+v", pid, identity))
			continue
		}
		stopped, err := probe(ctx, pid)
		if err != nil {
			failures = append(failures, fmt.Errorf("recorded pid %d: %w", pid, err))
		}
		if ctx.Err() != nil {
			break
		}
		if err == nil && stopped {
			continue
		}
		current, identityErr := inspect(ctx, pid)
		if identityErr != nil {
			if unresolved := serviceInspectionErrorWithProbe(ctx, pid, "recover identity", identityErr, serviceStoppedPresence(probe)); unresolved != nil {
				failures = append(failures, unresolved)
			}
			continue
		}
		if ctx.Err() != nil {
			break
		}
		if current != identity {
			continue // The recorded process departed and its PID was reused.
		}
		owned = append(owned, identity)
	}
	return owned, errors.Join(errors.Join(failures...), ctx.Err())
}

func serviceProcessIdentityAt(ctx context.Context, pid int32) (serviceProcessIdentity, error) {
	p, err := process.NewProcessWithContext(ctx, pid)
	if err != nil {
		return serviceProcessIdentity{}, err
	}
	path, err := p.ExeWithContext(ctx)
	if err != nil {
		return serviceProcessIdentity{}, err
	}
	path, err = serviceExecutablePath(path)
	if err != nil {
		return serviceProcessIdentity{}, err
	}
	created, err := p.CreateTimeWithContext(ctx)
	if err != nil {
		return serviceProcessIdentity{}, err
	}
	return serviceProcessIdentity{PID: pid, Path: path, Created: created}, ctx.Err()
}

// kill rechecks all identity fields immediately before targeting this PID.
func (in *serviceProcessInventory) kill(want serviceProcessIdentity) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return in.killContext(ctx, want)
}

func (in *serviceProcessInventory) killContext(ctx context.Context, want serviceProcessIdentity) error {
	return in.killContextWithProbes(ctx, want, func(ctx context.Context, pid int32) (serviceKillProcess, error) {
		return process.NewProcessWithContext(ctx, pid)
	}, serviceProcessStopped)
}

type serviceKillProcess interface {
	ExeWithContext(context.Context) (string, error)
	CreateTimeWithContext(context.Context) (int64, error)
	KillWithContext(context.Context) error
}

// The real and deterministic test paths share every check before signalling.
func (in *serviceProcessInventory) killContextWithProbes(ctx context.Context, want serviceProcessIdentity, open func(context.Context, int32) (serviceKillProcess, error), stopped func(context.Context, int32) (bool, error)) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	inspectionFailed := func(stage string, inspectionErr error) error {
		return serviceInspectionErrorWithProbe(ctx, want.PID, "revalidate "+stage, inspectionErr, serviceStoppedPresence(stopped))
	}
	p, err := open(ctx, want.PID)
	if err != nil {
		return inspectionFailed("process", err)
	}
	path, err := p.ExeWithContext(ctx)
	if err != nil {
		return inspectionFailed("executable", err)
	}
	path, err = serviceExecutablePath(path)
	if err != nil {
		return inspectionFailed("executable path", err)
	}
	created, err := p.CreateTimeWithContext(ctx)
	if err != nil {
		return inspectionFailed("creation time", err)
	}
	if path != want.Path || created != want.Created || !in.owns(path) {
		err := fmt.Errorf("refusing changed pid %d identity: path=%q created=%d, recorded=%+v", want.PID, path, created, want)
		if path != want.Path || created != want.Created {
			return errors.Join(err, errServiceProcessDeparted)
		}
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := p.KillWithContext(ctx); err != nil {
		if errors.Is(err, process.ErrorProcessNotRunning) {
			return inspectionFailed("signal", err)
		}
		return errors.Join(err, ctx.Err())
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return nil
}
