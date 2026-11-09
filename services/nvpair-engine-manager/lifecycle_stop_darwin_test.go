// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"os/exec"
	"syscall"
	"testing"
	"time"
)

// A group leader that exits after signalPID's Getpgid leaves a zombie-only
// group. Darwin answers that group signal with EPERM; the exited leader is not
// a permission refusal.
//
// Darwin only: Linux signals a zombie-only group and returns 0, so the EPERM
// this guards against cannot occur there, and a Linux zombie keeps its group,
// so the getpgid wait below would never end.
func TestSignalGroupLeaderAcceptsZombieOnlyGroup(t *testing.T) {
	cmd := exec.Command("/bin/sleep", "60")
	cmd.Env = childEnv(nil)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		t.Fatalf("start signal fixture: %v", err)
	}
	pid := cmd.Process.Pid
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	})
	if err := cmd.Process.Kill(); err != nil {
		t.Fatalf("kill fixture without reaping: %v", err)
	}
	// An unreaped exited leader no longer reports its group.
	deadline := time.Now().Add(3 * time.Second)
	for _, err := syscall.Getpgid(pid); !errors.Is(err, syscall.ESRCH); _, err = syscall.Getpgid(pid) {
		if time.Now().After(deadline) {
			t.Fatalf("fixture did not exit: getpgid err=%v", err)
		}
		time.Sleep(time.Millisecond)
	}
	if orphanPIDExited(pid) {
		t.Fatal("fixture was reaped before the group signal")
	}
	if err := signalGroupLeader(pid, syscall.SIGKILL); err != nil {
		t.Fatalf("signal zombie-only group = %v, want nil", err)
	}
}
