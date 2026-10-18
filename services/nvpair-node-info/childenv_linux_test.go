// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"testing"

	"nvpair-shared/secretenv/secretenvtest"
)

// The nvidia-smi GPU probe, with a sentinel child in its place on PATH.
func TestNvidiaSmiProbeRemovesProviderSecrets(t *testing.T) {
	report := secretenvtest.Prepare(t)
	secretenvtest.OnPath(t, "nvpair-node-info", "secretenv", "nvidia-smi")
	out, err := nvidiaSmiCSV("uuid,name,memory.total")
	if err != nil {
		t.Fatalf("nvidiaSmiCSV: %v", err)
	}
	secretenvtest.AssertNoValue(t, "probe output", out)
	secretenvtest.AssertRemoved(t, secretenvtest.Wait(t, report))
}
