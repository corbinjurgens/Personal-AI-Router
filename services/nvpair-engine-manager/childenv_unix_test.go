// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"testing"

	"nvpair-shared/secretenv/secretenvtest"
)

// The port-owner lookups run system tools by name; a sentinel child stands in
// for each on PATH. Port 1 is only an argument: nothing is bound or dialed.
func TestPortOwnerLookupsRemoveProviderSecrets(t *testing.T) {
	t.Run("lsof", func(t *testing.T) {
		report := secretenvtest.Prepare(t)
		secretenvtest.OnPath(t, "nvpair-engine-manager", "secretenv-lsof", "lsof")
		if pid, err := lsofPID(1); pid != 0 || err != nil {
			t.Fatalf("lsofPID = %d, %v; want no owner from a silent child", pid, err)
		}
		secretenvtest.AssertRemoved(t, secretenvtest.Wait(t, report))
	})
	t.Run("ss", func(t *testing.T) {
		report := secretenvtest.Prepare(t)
		secretenvtest.OnPath(t, "nvpair-engine-manager", "secretenv-ss", "ss")
		if pid, found := ssPID(1); found {
			t.Fatalf("ssPID = %d; want no owner from a silent child", pid)
		}
		secretenvtest.AssertRemoved(t, secretenvtest.Wait(t, report))
	})
}
