// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"
)

const memoryPlanRequestLimit = 1 << 20
const memoryPlanOutputLimit = 8 << 20
const memoryPlanBudget = 5 * time.Second
const memoryPlanCleanupBudget = 500 * time.Millisecond

var errMemoryPlanCleanup = errors.New("memory plan cleanup unconfirmed")

// Startup can fail after a process exists but before containment is assigned.
// A failed kill must not turn its reap into an unbounded synchronous wait.
func reapMemoryPlanStartFailure(ctx context.Context, wait func() error) error {
	done := make(chan struct{})
	go func() { _ = wait(); close(done) }()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return errMemoryPlanCleanup
	}
}

// runMemoryPlanCommand is private to the inspection owner. It admits at most
// one query per executor and shares inspection's existing non-queuing gate.
func (e *Executor) runMemoryPlanCommand(ctx context.Context, command memoryPlanCommand, request []byte) (json.RawMessage, error) {
	if !e.modelInspection.active.CompareAndSwap(false, true) {
		return nil, errors.New("memory plan busy")
	}
	releaseAdmission := true
	defer func() {
		if releaseAdmission {
			e.modelInspection.active.Store(false)
		}
	}()
	if len(request) > memoryPlanRequestLimit || !json.Valid(request) {
		return nil, errors.New("invalid memory plan request")
	}
	if err := command.validate(runtime.GOOS + "/" + runtime.GOARCH); err != nil {
		return nil, errors.New("invalid memory plan invocation")
	}
	deadline := time.Now().Add(memoryPlanBudget)
	if parent, ok := ctx.Deadline(); ok && parent.Before(deadline) {
		deadline = parent
	}
	if ctx.Err() != nil || time.Until(deadline) <= memoryPlanCleanupBudget {
		return nil, errors.New("memory plan deadline exhausted")
	}
	work, err := makeMemoryPlanTempDir()
	if err != nil {
		return nil, errors.New("memory plan temporary directory unavailable")
	}
	remove := true
	defer func() {
		if remove {
			_ = os.RemoveAll(work)
		}
	}()
	requestPath := filepath.Join(work, "request.json")
	if err := os.WriteFile(requestPath, request, 0600); err != nil {
		return nil, errors.New("memory plan request unavailable")
	}
	runCtx, cancel := context.WithDeadline(ctx, deadline.Add(-memoryPlanCleanupBudget))
	defer cancel()
	args := append([]string(nil), command.Command[1:]...)
	args[len(args)-1] = requestPath
	cmd := exec.Command(command.Command[0], args...)
	cmd.Dir = command.WorkingDirectory
	cmd.Env = []string{"PATH=" + os.Getenv("PATH"), "SYSTEMROOT=" + os.Getenv("SYSTEMROOT"), "TEMP=" + work, "TMP=" + work, "TMPDIR=" + work, "HOME=" + work}
	cmd.WaitDelay = memoryPlanCleanupBudget
	budget := &modelDeviceOutputBudget{limit: memoryPlanOutputLimit, cancel: cancel}
	stdout := &modelDeviceBoundedWriter{budget: budget}
	cmd.Stdout = stdout
	cmd.Stderr = &modelDeviceBoundedWriter{budget: budget}
	if runCtx.Err() != nil {
		return nil, errors.New("memory plan deadline exhausted")
	}
	cleanupCtx, cleanupCancel := context.WithDeadline(context.Background(), deadline)
	defer cleanupCancel()
	cleanup, err := startMemoryPlanProcess(cleanupCtx, cmd)
	if err != nil {
		if errors.Is(err, errMemoryPlanCleanup) {
			remove = false
			releaseAdmission = false
			return nil, errMemoryPlanCleanup
		}
		return nil, errors.New("memory plan process unavailable")
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	var waitErr error
	waited := false
	select {
	case waitErr = <-done:
		waited = true
	case <-runCtx.Done():
	}
	if err := cleanup(cleanupCtx); err != nil {
		remove = false
		releaseAdmission = false
		return nil, errors.New("memory plan cleanup unconfirmed")
	}
	if !waited {
		select {
		case waitErr = <-done:
		case <-cleanupCtx.Done():
			remove = false
			releaseAdmission = false
			return nil, errors.New("memory plan cleanup unconfirmed")
		}
	}
	if waitErr != nil || runCtx.Err() != nil || budget.exceeded {
		return nil, errors.New("memory plan process failed or bounded")
	}
	var result json.RawMessage
	d := json.NewDecoder(bytes.NewReader(stdout.data.Bytes()))
	if err := d.Decode(&result); err != nil || d.Decode(new(json.RawMessage)) != io.EOF {
		return nil, errors.New("invalid memory plan response")
	}
	return result, nil
}
