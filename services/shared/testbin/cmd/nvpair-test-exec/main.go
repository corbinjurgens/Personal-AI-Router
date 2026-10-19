// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	os.Exit(run(ctx, os.Args[1:]))
}

// fixedPath uses the same module/package naming as windows-test-bin.ps1 list.
func fixedPath(root, cwd, binary string) string {
	if !filepath.IsAbs(root) || !(strings.HasSuffix(binary, ".test.exe") || strings.HasSuffix(binary, ".test")) {
		return ""
	}
	for dir := cwd; ; dir = filepath.Dir(dir) {
		if info, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil && !info.IsDir() {
			if filepath.Base(filepath.Dir(dir)) != "services" {
				return ""
			}
			rel, err := filepath.Rel(dir, cwd)
			if err != nil {
				return ""
			}
			module := filepath.Base(dir)
			name := module
			if rel != "." {
				name += "-" + strings.ReplaceAll(filepath.ToSlash(rel), "/", "-")
			}
			suffix := ".test"
			if runtime.GOOS == "windows" {
				suffix += ".exe"
			}
			return filepath.Join(root, "go-test", module, name+suffix)
		}
		if filepath.Dir(dir) == dir {
			return ""
		}
	}
}

func acquireLock(ctx context.Context, path string) (func(), error) {
	waitCtx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()
	warned := false
	for {
		file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err == nil {
			if err := file.Close(); err != nil {
				_ = os.Remove(path)
				return nil, err
			}
			return func() { _ = os.Remove(path) }, nil
		}
		if !lockBusy(err) {
			return nil, err
		}
		if !warned {
			fmt.Fprintf(os.Stderr, "nvpair-test-exec: waiting for %s (remove a stale lock only after checking no test is running)\n", path)
			warned = true
		}
		select {
		case <-waitCtx.Done():
			return nil, fmt.Errorf("waiting for %s: %w", path, waitCtx.Err())
		case <-time.After(100 * time.Millisecond):
		}
	}
}

func copyBinary(ctx context.Context, source, destination string) error {
	in, err := os.Open(source)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.CreateTemp(filepath.Dir(destination), ".test-exec-*")
	if err != nil {
		return err
	}
	defer os.Remove(out.Name())
	_, copyErr := io.Copy(out, in)
	modeErr := out.Chmod(0o700)
	closeErr := out.Close()
	if err := errors.Join(copyErr, modeErr, closeErr); err != nil {
		return err
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		err = os.Rename(out.Name(), destination)
		if err == nil {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("replace %s (an earlier process may still be running): %w", destination, err)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}
}

func run(ctx context.Context, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: nvpair-test-exec <binary> <args...>")
		return 1
	}
	binary := args[0]
	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	if destination := fixedPath(os.Getenv("NVPAIR_TEST_BIN_DIR"), cwd, binary); destination != "" {
		if err := os.MkdirAll(filepath.Dir(destination), 0o700); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		release, err := acquireLock(ctx, destination+".lock")
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		defer release()
		if err := copyBinary(ctx, binary, destination); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		binary = destination
	}
	cmd := exec.CommandContext(ctx, binary, args[1:]...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := runChild(cmd); err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			return exit.ExitCode()
		}
		fmt.Fprintln(os.Stderr, "nvpair-test-exec:", err)
		return 1
	}
	return 0
}
