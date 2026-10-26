// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"time"
)

// Zero fields leave scanner construction on its ordinary defaults. Scan cadence
// and browse timeout are supplied together so forwarding never guesses defaults.
type scannerWaitOverrides struct {
	scanInterval time.Duration
	scanTimeout  time.Duration
	peerRefresh  time.Duration
}

func (s *scannerWaitOverrides) registerFlags(flags *flag.FlagSet) {
	flags.DurationVar(&s.scanInterval, "test-scanner-scan-interval", 0, "fixture-only scanner cadence; 0 retains its default")
	flags.DurationVar(&s.scanTimeout, "test-scanner-scan-timeout", 0, "fixture-only scanner browse timeout; set with scan interval")
	flags.DurationVar(&s.peerRefresh, "test-scanner-peer-refresh-interval", 0, "fixture-only scanner inventory refresh; 0 retains its default")
}

func (s scannerWaitOverrides) validate() error {
	if s.scanInterval < 0 || s.scanTimeout < 0 || s.peerRefresh < 0 {
		return fmt.Errorf("test scanner intervals must not be negative")
	}
	if (s.scanInterval == 0) != (s.scanTimeout == 0) {
		return fmt.Errorf("test scanner scan interval and timeout must be supplied together")
	}
	if s.scanTimeout > s.scanInterval {
		return fmt.Errorf("test scanner scan timeout must not exceed its interval")
	}
	return nil
}

func (s scannerWaitOverrides) args() []string {
	var args []string
	if s.scanInterval > 0 {
		args = append(args, "--test-scan-interval", s.scanInterval.String(), "--test-scan-timeout", s.scanTimeout.String())
	}
	if s.peerRefresh > 0 {
		args = append(args, "--test-peer-refresh-interval", s.peerRefresh.String())
	}
	return args
}

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
