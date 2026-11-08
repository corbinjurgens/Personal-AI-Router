// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"sync"
	"time"
)

const modelReadContentionBudget = time.Second
const modelReadContentionAttempts = 21

// Retry only a read whose owner identified runtime contention or snapshot drift.
// Each attempt releases all of its leases before returning. In particular, never
// wait for opMu while holding a content lease that its current owner may need.
// The caller retains its original operation/context budget across attempts.
func retryModelRead(ctx context.Context, observe func(*bool) error) error {
	var until time.Time
	var err error
	for attempt := 0; attempt < modelReadContentionAttempts; attempt++ {
		if attempt != 0 && !time.Now().Before(until) {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		retry := false
		err = observe(&retry)
		// A completed observation is the owner's answer; deadline handling for
		// its own phases, including advisory ones, belongs to that owner.
		if !retry || attempt == modelReadContentionAttempts-1 {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		// Budget contention from its first occurrence, not from metadata I/O
		// that preceded it. The original operation deadline still bounds I/O.
		if until.IsZero() {
			until = time.Now().Add(modelReadContentionBudget)
		}
		if !time.Now().Before(until) {
			return err
		}
		timer := time.NewTimer(min(time.Until(until), modelReadContentionBudget/(modelReadContentionAttempts-1)))
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
	return err
}

// Only for read owners holding no inverse lock or content lease, and never
// inside retryModelRead: a nested wait would start its own contention window.
// Whole-read retries use TryLock and let the outer retry own waiting. Unlike the
// resident mutation's queued join, a bounded read leaves no mutex waiter behind
// when its context or contention budget expires.
func modelReadLock(ctx context.Context, mu *sync.Mutex) bool {
	locked := false
	err := retryModelRead(ctx, func(retry *bool) error {
		locked = mu.TryLock()
		*retry = !locked
		return nil
	})
	if err != nil && locked {
		mu.Unlock()
		locked = false
	}
	return locked
}
