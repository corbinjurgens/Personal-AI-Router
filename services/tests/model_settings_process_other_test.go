//go:build !windows

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"context"
	"time"
)

func modelSettingsReplaceProcess(expected serviceProcessIdentity, kill func() error, timeout time.Duration) (bool, error) {
	if err := kill(); err != nil {
		return false, err
	}
	// Preserve the existing non-Windows identity proof and inspection errors.
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	tick := time.NewTicker(25 * time.Millisecond)
	defer tick.Stop()
	for {
		stopped, err := switchProcessIdentityStopped(ctx, expected)
		if err != nil || stopped {
			return stopped, err
		}
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		case <-tick.C:
		}
	}
}
