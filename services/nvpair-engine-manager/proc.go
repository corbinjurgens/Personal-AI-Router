// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"nvpair-shared/secretenv"
)

// managedProc is a spawned engine process with its stdout/stderr
// captured line-by-line and a `done` channel that closes when it
// exits. The OS-specific primitives it relies on (console hiding,
// graceful signal, force kill) live in proc_windows.go / proc_unix.go
// so the same logic here runs on every platform.
type managedProc struct {
	cmd  *exec.Cmd
	done chan struct{}
}

// childEnv is the environment of every process engine-manager starts: the
// current environment with overrides merged over it, then without the reserved
// provider-secret namespace. Filtering after the merge means an override
// cannot reintroduce a provider secret.
func childEnv(overrides map[string]string) []string {
	env := os.Environ()
	for k, v := range overrides {
		env = append(env, k+"="+v)
	}
	return secretenv.Remove(env)
}

// startManagedProc launches bin with args and the extra env (merged
// over the current environment, see childEnv), hides any console window,
// and streams stdout/stderr lines to onLine. The returned proc's done
// channel closes once the process exits.
func startManagedProc(bin string, args []string, env map[string]string, onLine func(stream, line string)) (*managedProc, error) {
	cmd := exec.Command(bin, args...)
	cmd.Env = childEnv(env)
	configureSysProcAttr(cmd)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}

	mp := &managedProc{cmd: cmd, done: make(chan struct{})}
	var scanWg sync.WaitGroup
	scanWg.Add(2)
	go func() { defer scanWg.Done(); scanLines(stdout, "stdout", onLine) }()
	go func() { defer scanWg.Done(); scanLines(stderr, "stderr", onLine) }()
	go func() {
		// Drain stdout/stderr fully before Wait: Wait closes the pipes on
		// exit, which would otherwise truncate the final captured lines.
		scanWg.Wait()
		_ = cmd.Wait()
		close(mp.done)
	}()
	return mp, nil
}

func scanLines(r io.Reader, stream string, onLine func(stream, line string)) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 256*1024)
	for sc.Scan() {
		if onLine != nil {
			onLine(stream, sc.Text())
		}
	}
	if sc.Err() != nil {
		_, _ = io.Copy(io.Discard, r)
	}
}

// stop asks the owned process tree to exit, waits for grace, then force-kills
// it if necessary. A non-positive grace skips the graceful step.
//
// On Unix the two signals are SIGTERM then SIGKILL. On Windows the engines run
// windowless, so gracefulSignal is already taskkill /T /F and normally ends the
// process immediately; the deadline remains a backstop for a failed taskkill.
func (mp *managedProc) stop(grace time.Duration) {
	if mp == nil || mp.cmd == nil || mp.cmd.Process == nil {
		return
	}
	select {
	case <-mp.done:
		return // already exited
	default:
	}
	if grace <= 0 {
		_ = signalPID(mp.cmd.Process.Pid, true)
		<-mp.done
		return
	}
	_ = gracefulSignal(mp.cmd)
	timer := time.NewTimer(grace)
	defer timer.Stop()
	select {
	case <-mp.done:
		return
	case <-timer.C:
	}
	_ = signalPID(mp.cmd.Process.Pid, true)
	<-mp.done
}

// terminatePID stops the process with the given PID (and its tree on
// Windows, or its process group on Unix when available): a graceful signal
// first, escalating to a forced kill if the process hasn't exited within
// grace; a non-positive grace kills immediately. It exists to reclaim a
// PAIR-managed engine orphan adopted on our own port — an instance a prior run
// spawned and then lost the handle to, so we can only address it by PID rather
// than through the *exec.Cmd handle managedProc.stop needs. Best-effort: a
// process that's already gone counts as success. The platform primitives
// (signalPID, pidAlive) live in proc_windows.go / proc_unix.go.
func terminatePID(pid int, grace time.Duration) {
	_ = terminatePIDContext(context.Background(), pid, grace)
}

func terminatePIDContext(ctx context.Context, pid int, grace time.Duration) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if pid <= 0 {
		return nil
	}
	if runtime.GOOS == "windows" {
		// The platform stop is already forceful; there is no graceful phase.
		grace = 0
	}
	if grace <= 0 {
		return forceTerminatePIDContext(ctx, pid)
	}
	if err := signalPIDContext(ctx, pid, false); err != nil && pidAlive(pid) {
		return err
	}
	deadline := time.Now().Add(grace)
	for time.Now().Before(deadline) {
		if !pidAlive(pid) {
			return ctx.Err()
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}
	if pidAlive(pid) {
		return forceTerminatePIDContext(ctx, pid)
	}
	return ctx.Err()
}

func forceTerminatePIDContext(ctx context.Context, pid int) error {
	if err := signalPIDContext(ctx, pid, true); err != nil && !orphanPIDExited(pid) {
		return err
	}
	// A successful forced signal is not exit evidence. Allow the existing
	// default stop grace for teardown/reaping, even for an immediate kill.
	ctx, cancel := context.WithTimeout(ctx, stopGrace(Runtime{}))
	defer cancel()
	for !orphanPIDExited(pid) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}
	return ctx.Err()
}

// normalizeEngineImage handles annotations supplied by process inspection,
// never configured filenames. Only Linux /proc annotates a deleted image.
// An existing literal suffixed file must not be mistaken for that annotation;
// if the two spellings collide, refusing reclaim is safer than guessing.
func normalizeEngineImage(path, platform string) string {
	if platform == "linux" && strings.HasSuffix(path, " (deleted)") {
		if _, err := os.Lstat(path); os.IsNotExist(err) {
			path = strings.TrimSuffix(path, " (deleted)")
		}
	}
	return filepath.Clean(path)
}

// linuxEngineImage distinguishes a literal suffixed file from a deleted-image
// annotation colliding with another file. exe is the exact /proc/PID/exe path.
func linuxEngineImage(path, exe string) string {
	if strings.HasSuffix(path, " (deleted)") {
		// Use the same entry-existence check as normalization. A dangling
		// symlink is an existing literal entry, not a deleted annotation.
		if _, err := os.Lstat(path); err != nil {
			if os.IsNotExist(err) {
				return path
			}
			return ""
		}
		literal, err := os.Stat(path)
		if err != nil {
			return ""
		}
		actual, err := os.Stat(exe)
		if err != nil || !os.SameFile(literal, actual) {
			return ""
		}
	}
	return path
}

// isOurEngineImage reports whether the listener on our managed port is
// running the same binary this service manages for the engine (st.binPath).
// The compare uses file identity on Windows, canonical paths on Darwin and
// symlink-resolved, case-sensitive paths on other Unix systems, allowing Stop
// to reclaim a PAIR-owned orphan on our managed port while leaving an
// unrelated listener for the caller to decline. An empty image
// (e.g. a PID we can't introspect) never matches, so Stop fails closed.
func isOurEngineImage(image, binPath string) bool {
	return isOurEngineImageOn(image, binPath, runtime.GOOS)
}

func isOurEngineImageOn(image, binPath, platform string) bool {
	if image == "" || binPath == "" {
		return false
	}
	image = normalizeEngineImage(image, platform)
	binPath = filepath.Clean(binPath)
	if platform == "windows" {
		// Short names, junctions and symlinks can name the same executable.
		// SameFile compares volume and file IDs on Windows, and returns false
		// if either identity cannot be read. Never substitute a path/name or
		// content comparison: distinct executable files are not our orphan.
		imageInfo, err := os.Stat(image)
		if err != nil || !imageInfo.Mode().IsRegular() {
			return false
		}
		binInfo, err := os.Stat(binPath)
		return err == nil && binInfo.Mode().IsRegular() && os.SameFile(imageInfo, binInfo)
	}
	if platform == "darwin" {
		var err error
		image, err = filepath.EvalSymlinks(image)
		if err != nil {
			return false
		}
		binPath, err = filepath.EvalSymlinks(binPath)
		if err != nil {
			return false
		}
		if image == binPath {
			return true
		}
		// Case aliases count only when the filesystem confirms the same file.
		imageInfo, imageErr := os.Stat(image)
		binInfo, binErr := os.Stat(binPath)
		return strings.EqualFold(image, binPath) && imageErr == nil && binErr == nil && os.SameFile(imageInfo, binInfo)
	}
	if resolved, err := filepath.EvalSymlinks(image); err == nil {
		image = resolved
	}
	if resolved, err := filepath.EvalSymlinks(binPath); err == nil {
		binPath = resolved
	}
	return image == binPath
}

// isManagedInstallPath reports whether binPath is inside this engine's
// NVPAIR-owned install directory. A matching executable name/path is not, by
// itself, ownership: Detect intentionally recognizes vendor/user installs on
// some platforms. Stop/uninstall must never reclaim those external files.
func isManagedInstallPath(binPath, installDir string) bool {
	if binPath == "" || installDir == "" {
		return false
	}
	binAbs, err := filepath.Abs(binPath)
	if err != nil {
		return false
	}
	dirAbs, err := filepath.Abs(installDir)
	if err != nil {
		return false
	}
	return managedInstallPath(binAbs, dirAbs)
}
