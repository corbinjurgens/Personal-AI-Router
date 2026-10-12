// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/ebitengine/purego"
)

// portLookupTimeout bounds each external port-owner lookup (lsof/ss). The
// broker no longer force-kills engine-manager on a timeout, so an unbounded
// lookup that hung would wedge StopAll (and the whole app shutdown).
const portLookupTimeout = 2 * time.Second

// configureSysProcAttr puts the child in its own process group so a
// terminate signals the whole group — engines that fork helper
// processes (model runners, etc.) get cleaned up too.
func configureSysProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// gracefulSignal sends SIGTERM to the process group (falling back to the
// process itself). managedProc.stop escalates to SIGKILL if the group remains
// alive after the manifest's grace period.
func gracefulSignal(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	if pgid, err := syscall.Getpgid(cmd.Process.Pid); err == nil {
		return syscall.Kill(-pgid, syscall.SIGTERM)
	}
	return cmd.Process.Signal(syscall.SIGTERM)
}

// pidOnPort returns the PID listening on the given TCP port and that
// process's executable path. Best-effort on the debug-only Unix targets: it
// shells out to lsof, falling back to ss only when lsof is unavailable or has
// no result. Known malformed, incomplete or ambiguous ownership is unknown.
func pidOnPort(port int) (pid int, image string, ok bool) {
	p, err := lsofPID(port)
	if err != nil {
		return 0, "", false
	}
	if p != 0 {
		return p, procImage(p), true
	}
	if p, found := ssPID(port); found {
		return p, procImage(p), true
	}
	return 0, "", false
}

// lsofPID reads just the PID(s) of the TCP listener (-t = terse). A zero PID
// with no error permits fallback; malformed, incomplete or conflicting output
// does not.
func lsofPID(port int) (int, error) {
	ctx, cancel := context.WithTimeout(context.Background(), portLookupTimeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, "lsof", "-nP", "-tiTCP:"+strconv.Itoa(port), "-sTCP:LISTEN").Output()
	pid := 0
	for _, field := range strings.Fields(string(out)) {
		p, err := strconv.Atoi(field)
		if err != nil || p <= 0 || (pid != 0 && pid != p) {
			return 0, errors.New("malformed or ambiguous lsof ownership")
		}
		pid = p
	}
	if err != nil {
		if pid != 0 {
			return 0, errors.New("incomplete lsof ownership query")
		}
		return 0, nil
	}
	return pid, nil
}

// ssPID parses the owning PID out of ss's users:(("proc",pid=1234,fd=7)) tail.
func ssPID(port int) (int, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), portLookupTimeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, "ss", "-ltnp", "sport", "=", ":"+strconv.Itoa(port)).Output()
	if err != nil {
		return 0, false
	}
	s := string(out)
	pid := 0
	for {
		idx := strings.Index(s, "pid=")
		if idx < 0 {
			return pid, pid != 0
		}
		s = s[idx+len("pid="):]
		end := strings.IndexAny(s, ",)")
		if end < 0 {
			return 0, false
		}
		p, err := strconv.Atoi(s[:end])
		if err != nil || p <= 0 || (pid != 0 && pid != p) {
			return 0, false
		}
		pid = p
		s = s[end:]
	}
}

// procImage reads the executable identity from the OS, never argv or a
// basename. Darwin's proc_pidpath copies at most PROC_PIDPATHINFO_MAXSIZE
// bytes from the kernel; unavailable or malformed inspection fails closed.
func procImage(pid int) string {
	if pid <= 0 || int64(pid) > 1<<31-1 {
		return ""
	}
	if runtime.GOOS == "darwin" {
		handle, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_NOW|purego.RTLD_LOCAL)
		if err != nil {
			return ""
		}
		defer purego.Dlclose(handle)
		symbol, err := purego.Dlsym(handle, "proc_pidpath")
		if err != nil {
			return ""
		}
		var lookup func(int32, *byte, uint32) int32
		purego.RegisterFunc(&lookup, symbol)
		buf := make([]byte, 4096)
		n := lookup(int32(pid), &buf[0], uint32(len(buf)))
		runtime.KeepAlive(buf)
		if n <= 0 || n >= int32(len(buf)) || buf[n] != 0 || bytes.IndexByte(buf[:n], 0) >= 0 {
			return ""
		}
		path := string(buf[:n])
		if !filepath.IsAbs(path) {
			return ""
		}
		return path
	}
	exe := "/proc/" + strconv.Itoa(pid) + "/exe"
	if path, err := os.Readlink(exe); err == nil {
		if runtime.GOOS == "linux" {
			return linuxEngineImage(path, exe)
		}
		return path
	}
	return ""
}

// signalPID sends SIGTERM (or SIGKILL when force) to the process group when
// possible. The owned-process escalation and orphan reclaim share it so forced
// stops reach forked helpers (model runners, etc.).
func signalPID(pid int, force bool) error {
	if pid <= 0 {
		return nil
	}
	sig := syscall.SIGTERM
	if force {
		sig = syscall.SIGKILL
	}
	if pgid, err := syscall.Getpgid(pid); err == nil {
		return syscall.Kill(-pgid, sig)
	}
	return syscall.Kill(pid, sig)
}

// pidAlive reports whether the PID still exists (signal 0 probes existence
// without delivering a signal; EPERM means it exists but isn't ours).
func pidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	return err == nil || errors.Is(err, syscall.EPERM)
}
