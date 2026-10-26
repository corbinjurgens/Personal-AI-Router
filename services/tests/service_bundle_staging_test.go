// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// Process exit and release of a Windows executable image are distinct events.
// Only retry the latter, within the caller's deadline, after independently
// confirming that every recorded process and bundle sibling has exited. Never
// turn a live or uninspectable process into a filesystem retry.
func serviceBundleMutation(ctx context.Context, stopped func(context.Context) error, mutate func() error, releasePending func(error) bool) error {
	var lastErr error
	for {
		if err := ctx.Err(); err != nil {
			return errors.Join(lastErr, err)
		}
		if err := stopped(ctx); err != nil {
			return fmt.Errorf("confirm service bundle exit before staging: %w", errors.Join(lastErr, err))
		}
		if err := ctx.Err(); err != nil {
			return errors.Join(lastErr, err)
		}
		lastErr = mutate()
		if lastErr == nil || !releasePending(lastErr) {
			return lastErr
		}
		timer := time.NewTimer(25 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return errors.Join(lastErr, ctx.Err())
		case <-timer.C:
		}
	}
}
