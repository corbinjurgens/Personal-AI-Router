// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os"
	"testing"

	"nvpair-shared/secretenv/secretenvtest"
)

// The ioreg GPU probe, with a sentinel child in place of ioreg.
func TestIORegistryProbeRemovesProviderSecrets(t *testing.T) {
	report := secretenvtest.Prepare(t)
	old := ioRegistryPath
	ioRegistryPath = os.Args[0]
	t.Cleanup(func() { ioRegistryPath = old })
	out, err := readDarwinIORegistryContext(context.Background())
	if err != nil {
		t.Fatalf("readDarwinIORegistryContext: %v", err)
	}
	secretenvtest.AssertNoValue(t, "probe output", string(out))
	secretenvtest.AssertRemoved(t, secretenvtest.Wait(t, report))
}
