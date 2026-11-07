// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"strings"
	"testing"
)

func TestReservedAliasPortBlocksInternalStartAndPortChanges(t *testing.T) {
	manifest := testEngineManifest(fakeEngineBin)
	reg := NewRegistry()
	reg.engines[manifest.Engine] = manifest
	exec := NewExecutor(reg, NewReporter(nil), func(string, any) {}, t.TempDir())
	exec.overrideDir = t.TempDir()
	if err := exec.SetReservedPort(15555); err != nil {
		t.Fatal(err)
	}

	if err := exec.StartWith(context.Background(), manifest.Engine, startOpts{Port: 15555}); err == nil || !strings.Contains(err.Error(), "reserved") {
		t.Fatalf("local start error = %v, want reserved-port rejection", err)
	}
	if _, err := exec.SetPort(context.Background(), manifest.Engine, 15555); err == nil || !strings.Contains(err.Error(), "reserved") {
		t.Fatalf("set-port error = %v, want reserved-port rejection", err)
	}

}
