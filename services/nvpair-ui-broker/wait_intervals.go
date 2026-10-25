// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"time"
)

func registerTestWaitFlags(flags *flag.FlagSet) (residentRetry, advertise *time.Duration) {
	return flags.Duration("test-resident-retry-interval", defaultResidentRetryInterval, "fixture-only resident reconciliation retry interval (must be positive)"),
		flags.Duration("test-advertise-interval", autoAdvertiseInterval, "fixture-only local engine advertisement interval (must be positive)")
}

func validateTestWaitIntervals(residentRetry, advertise time.Duration) error {
	if residentRetry <= 0 {
		return fmt.Errorf("--test-resident-retry-interval must be positive (got %s)", residentRetry)
	}
	if advertise <= 0 {
		return fmt.Errorf("--test-advertise-interval must be positive (got %s)", advertise)
	}
	return nil
}

func (b *Broker) modelConfigRetryInterval() time.Duration {
	if b.residentRetryDelay == 0 {
		return defaultResidentRetryInterval
	}
	return b.residentRetryDelay
}

func (b *Broker) advertisementInterval() time.Duration {
	if b.advertiseCadence == 0 {
		return autoAdvertiseInterval
	}
	return b.advertiseCadence
}
