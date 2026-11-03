// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"context"
	"testing"

	"nvpair-shared/secretenv/secretenvtest"
)

// Process inspection runs system tools by name; a sentinel child stands in
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
	t.Run("ps", func(t *testing.T) {
		report := secretenvtest.Prepare(t)
		secretenvtest.OnPath(t, "nvpair-engine-manager", "secretenv-ps", "ps")
		records, _ := listProcessRecords(context.Background())
		if len(records) != 0 {
			t.Fatalf("listProcessRecords = %v; want no records from a silent child", records)
		}
		secretenvtest.AssertRemoved(t, secretenvtest.Wait(t, report))
	})
	t.Run("ps failure", func(t *testing.T) {
		report := secretenvtest.Prepare(t)
		secretenvtest.OnPath(t, "nvpair-engine-manager", "secretenv-ps", "ps")
		t.Setenv(secretenvtest.ExitEnv, "1")
		records, err := listProcessRecords(context.Background())
		if err == nil {
			t.Fatal("listProcessRecords succeeded after ps exited nonzero")
		}
		if records != nil {
			t.Fatalf("listProcessRecords = %v; want no records after ps failure", records)
		}
		secretenvtest.AssertRemoved(t, secretenvtest.Wait(t, report))
	})
}
