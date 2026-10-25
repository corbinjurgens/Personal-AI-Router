// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"errors"
	"os/exec"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
	"time"
)

// Each fixture group is separate from the test runner, so an incorrect group
// signal can only terminate these disposable sleep processes.
func startSignalTestProcess(t *testing.T, pgid int) (*exec.Cmd, <-chan error) {
	t.Helper()
	cmd := exec.Command("/bin/sleep", "60")
	cmd.Env = childEnv(nil)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pgid: pgid}
	if err := cmd.Start(); err != nil {
		t.Fatalf("start signal fixture: %v", err)
	}
	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
		close(done)
	}()
	t.Cleanup(func() {
		// Kill only this exact fixture PID, even if group signalling is broken.
		_ = cmd.Process.Kill()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("signal fixture did not exit during cleanup")
		}
	})
	return cmd, done
}

func TestUnixSignalsRespectGroupLeadership(t *testing.T) {
	signals := []struct {
		name string
		send func(*exec.Cmd) error
		want syscall.Signal
	}{
		{"owned graceful stop", gracefulSignal, syscall.SIGTERM},
		{"orphan graceful stop", func(cmd *exec.Cmd) error { return signalPID(cmd.Process.Pid, false) }, syscall.SIGTERM},
		{"forced stop", func(cmd *exec.Cmd) error { return signalPID(cmd.Process.Pid, true) }, syscall.SIGKILL},
	}
	for _, signal := range signals {
		t.Run(signal.name, func(t *testing.T) {
			for _, leaderTarget := range []bool{true, false} {
				name := "nonleader preserves shared group"
				if leaderTarget {
					name = "leader signals whole group"
				}
				t.Run(name, func(t *testing.T) {
					leader, leaderDone := startSignalTestProcess(t, 0)
					peer, peerDone := startSignalTestProcess(t, leader.Process.Pid)
					target, targetDone, otherDone := peer, peerDone, leaderDone
					if leaderTarget {
						target, targetDone, otherDone = leader, leaderDone, peerDone
					}
					pgid, err := syscall.Getpgid(target.Process.Pid)
					if err != nil || pgid != leader.Process.Pid {
						t.Fatalf("fixture pgid = %d, err = %v; want %d", pgid, err, leader.Process.Pid)
					}
					if err := signal.send(target); err != nil {
						t.Fatalf("signal fixture: %v", err)
					}
					assertSignal := func(done <-chan error) {
						t.Helper()
						select {
						case err := <-done:
							var exitErr *exec.ExitError
							if !errors.As(err, &exitErr) {
								t.Fatalf("fixture exit = %v; want signal %v", err, signal.want)
							}
							status, ok := exitErr.Sys().(syscall.WaitStatus)
							if !ok || !status.Signaled() || status.Signal() != signal.want {
								t.Fatalf("fixture exit = %v; want signal %v", err, signal.want)
							}
						case <-time.After(3 * time.Second):
							t.Fatal("signalled fixture did not exit")
						}
					}
					assertSignal(targetDone)
					if leaderTarget {
						assertSignal(otherDone)
					} else {
						select {
						case err := <-otherDone:
							t.Fatalf("nonleader signal terminated group leader: %v", err)
						case <-time.After(100 * time.Millisecond):
						}
					}
				})
			}
		})
	}
}

func startTermIgnoringProcess(t *testing.T) *managedProc {
	t.Helper()
	ready := filepath.Join(t.TempDir(), "ready")
	script := `trap '' TERM
: > "$1"
while :; do sleep 1; done`
	proc, err := startManagedProc("/bin/sh", []string{"-c", script, "stubborn-engine", ready}, nil, nil)
	if err != nil {
		t.Fatalf("start term-ignoring process: %v", err)
	}
	t.Cleanup(func() {
		select {
		case <-proc.done:
			return
		default:
		}
		_ = signalPID(proc.cmd.Process.Pid, true)
		select {
		case <-proc.done:
		case <-time.After(2 * time.Second):
			t.Errorf("term-ignoring process did not exit during cleanup")
		}
	})

	deadline := time.Now().Add(5 * time.Second)
	for !fileExists(ready) && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if !fileExists(ready) {
		t.Fatal("term-ignoring process did not become ready")
	}
	return proc
}

func TestStopHonorsProcessSignalPolicy(t *testing.T) {
	test := func(t *testing.T, stop StopSpec, minElapsed, maxElapsed time.Duration) {
		t.Helper()
		proc := startTermIgnoringProcess(t)
		manifest := testEngineManifest(fakeEngineBin)
		key := runtime.GOOS + "/" + runtime.GOARCH
		platform := manifest.Platforms[key]
		platform.Runtime.Stop = &stop
		manifest.Platforms[key] = platform
		ex := newTestExecutor(t, manifest)
		state, err := ex.state(manifest.Engine)
		if err != nil {
			t.Fatalf("resolve engine state: %v", err)
		}
		state.mu.Lock()
		state.running = true
		state.healthy = true
		state.proc = proc
		state.mu.Unlock()

		started := time.Now()
		result := make(chan error, 1)
		go func() {
			result <- ex.Stop(manifest.Engine)
		}()
		select {
		case err := <-result:
			if err != nil {
				t.Fatalf("stop engine: %v", err)
			}
		case <-time.After(maxElapsed + time.Second):
			_ = signalPID(proc.cmd.Process.Pid, true)
			t.Fatalf("stop did not finish within %s", maxElapsed+time.Second)
		}

		elapsed := time.Since(started)
		if elapsed < minElapsed || elapsed > maxElapsed {
			t.Fatalf("stop elapsed %s, want between %s and %s", elapsed, minElapsed, maxElapsed)
		}
		select {
		case <-proc.done:
		default:
			t.Fatal("stop returned before the owned process exited")
		}
	}

	t.Run("term escalates after configured grace", func(t *testing.T) {
		test(t, StopSpec{Signal: "term", GraceS: 1}, 800*time.Millisecond, 3*time.Second)
	})
	t.Run("kill bypasses configured grace", func(t *testing.T) {
		test(t, StopSpec{Signal: "kill", GraceS: 10}, 0, 2*time.Second)
	})
}
