// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"strings"
	"testing"
)

func TestWorkingSetLimitRefusedOnUnix(t *testing.T) {
	e := workingSetSettingsExecutor(t)
	request := settingsRequest(t, e)
	limit := 256
	request.Settings.WorkingSetLimitMiB = &limit
	preview := previewSettings(t, e, request)
	if !strings.Contains(preview.Errors["workingSetLimitMiB"], "only on Windows") {
		t.Fatalf("missing platform field error: %+v", preview.Errors)
	}
	assertNoSettingsOverride(t, e)
}
