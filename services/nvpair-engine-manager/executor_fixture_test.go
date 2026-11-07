// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "testing"

func newLifecycleFixture(t *testing.T) *Executor {
	t.Helper()
	reg := NewRegistry()
	for _, name := range []string{"source", "target", "other"} {
		manifest := testEngineManifest(fakeEngineBin)
		manifest.Engine = name
		reg.engines[name] = manifest
	}
	ex := NewExecutor(reg, NewReporter(nil), nil, t.TempDir())
	t.Cleanup(ex.StopAll)
	ex.nodeProjection.set(nodeProjection{ManagementBlocked: false})
	return ex
}
