// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package servicectl is the client side of nvpair-service: where its per-user
// endpoint lives, how to dial it, and how to start the service detached when it
// is not running.
//
// nvpair-service owns nvpair-ui-broker and outlives its clients. A front end
// (the terminal UI, the desktop app) calls ConnectOrStart, speaks the broker's
// newline-delimited JSON-RPC 2.0 over the returned connection, and closes it to
// detach. Closing never stops the service; that takes an explicit
// service/stop request.
package servicectl

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// EndpointEnv overrides the endpoint Endpoint returns. Tests use it to point a
// service and its clients at a private socket; the detached service inherits
// it, so both sides agree.
const EndpointEnv = "NVPAIR_SERVICE_ENDPOINT"

// startTimeout bounds how long ConnectOrStart keeps dialing a service it has
// just started. The service listens before it spawns the broker, so it answers
// within a fraction of a second; the margin covers a cold disk and a slow
// antivirus scan of a freshly built binary.
const startTimeout = 10 * time.Second

// dialRetryInterval is the pause between dial attempts while a service starts.
const dialRetryInterval = 100 * time.Millisecond

// ErrNotRunning reports that nothing is listening on the endpoint.
var ErrNotRunning = errors.New("nvpair-service is not running")

// Endpoint returns this user's nvpair-service endpoint: <appdir>/service.sock
// on Linux and macOS, \\.\pipe\nvpair-service-<username> on Windows. EndpointEnv
// overrides it.
func Endpoint() (string, error) {
	if v := os.Getenv(EndpointEnv); v != "" {
		return v, nil
	}
	return defaultEndpoint()
}

// Dial connects to the running service. It returns an error wrapping
// ErrNotRunning when nothing is listening.
func Dial(ctx context.Context) (io.ReadWriteCloser, error) {
	endpoint, err := Endpoint()
	if err != nil {
		return nil, err
	}
	return DialEndpoint(ctx, endpoint)
}

// DialEndpoint connects to a service listening at endpoint.
func DialEndpoint(ctx context.Context, endpoint string) (io.ReadWriteCloser, error) {
	conn, err := dialEndpoint(ctx, endpoint)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNotRunning, err)
	}
	return conn, nil
}

// StartDetached starts the nvpair-service binary at binaryPath in the
// background, fully detached from the caller: its own session (Unix) or process
// group with no console (Windows), stdio on the null device, and the working
// directory set to the binary's own directory. brokerArgs are passed through to
// the broker after "--".
//
// It returns once the process has been created. The service may still exit
// straight away — for instance because another copy won the race to start — so
// callers dial afterwards rather than trusting the start alone.
func StartDetached(binaryPath string, brokerArgs []string) error {
	args := make([]string, 0, len(brokerArgs)+1)
	if len(brokerArgs) > 0 {
		args = append(args, "--")
		args = append(args, brokerArgs...)
	}
	cmd := exec.Command(binaryPath, args...)
	cmd.Dir = filepath.Dir(binaryPath)
	// Nil stdio is the null device: a detached service must not hold the
	// caller's terminal, and writing to a closed one would kill it with SIGPIPE.
	cmd.Stdin, cmd.Stdout, cmd.Stderr = nil, nil, nil
	configureDetached(cmd)
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start %s: %w", binaryPath, err)
	}
	// Reaped in the background so a service that exits while this process
	// lives on does not linger as a zombie. Nothing waits on the result.
	go func() { _ = cmd.Wait() }()
	return nil
}

// ConnectOrStart dials the service and, when it is not running, starts
// binaryPath detached and keeps dialing for up to ten seconds (or until ctx
// ends).
func ConnectOrStart(ctx context.Context, binaryPath string, brokerArgs []string) (io.ReadWriteCloser, error) {
	endpoint, err := Endpoint()
	if err != nil {
		return nil, err
	}
	if conn, err := DialEndpoint(ctx, endpoint); err == nil {
		return conn, nil
	}
	if err := StartDetached(binaryPath, brokerArgs); err != nil {
		return nil, err
	}
	deadline, cancel := context.WithTimeout(ctx, startTimeout)
	defer cancel()
	var lastErr error
	for {
		conn, err := DialEndpoint(deadline, endpoint)
		if err == nil {
			return conn, nil
		}
		lastErr = err
		select {
		case <-deadline.Done():
			return nil, fmt.Errorf("started %s but it did not answer on %s: %w", binaryPath, endpoint, lastErr)
		case <-time.After(dialRetryInterval):
		}
	}
}
