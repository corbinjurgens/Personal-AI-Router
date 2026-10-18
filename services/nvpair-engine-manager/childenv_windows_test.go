// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"os/exec"
	"testing"

	"nvpair-shared/secretenv/secretenvtest"
)

// Both taskkill launches run the tool by name; a sentinel child stands in for
// it on PATH. The PID is only an argument: no process is signalled.
func TestTaskkillRemovesProviderSecrets(t *testing.T) {
	const pid = 999999
	t.Run("signalPID", func(t *testing.T) {
		report := secretenvtest.Prepare(t)
		secretenvtest.OnPath(t, "nvpair-engine-manager", "secretenv-signalpid", "taskkill")
		if err := signalPID(pid, true); err != nil {
			t.Fatalf("signalPID: %v", err)
		}
		secretenvtest.AssertRemoved(t, secretenvtest.Wait(t, report))
	})
	t.Run("taskkill", func(t *testing.T) {
		report := secretenvtest.Prepare(t)
		secretenvtest.OnPath(t, "nvpair-engine-manager", "secretenv-taskkill", "taskkill")
		if err := taskkill(&exec.Cmd{Process: &os.Process{Pid: pid}}, false); err != nil {
			t.Fatalf("taskkill: %v", err)
		}
		secretenvtest.AssertRemoved(t, secretenvtest.Wait(t, report))
	})
}
