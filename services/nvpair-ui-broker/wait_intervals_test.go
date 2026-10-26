// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"flag"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestBrokerScannerWaitOverrides(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want []string
	}{
		{name: "ordinary startup passes no overrides"},
		{name: "scan pair", args: []string{"--test-scanner-scan-interval=250ms", "--test-scanner-scan-timeout=150ms"}, want: []string{"--test-scan-interval", "250ms", "--test-scan-timeout", "150ms"}},
		{name: "peer refresh", args: []string{"--test-scanner-peer-refresh-interval=250ms"}, want: []string{"--test-peer-refresh-interval", "250ms"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var overrides scannerWaitOverrides
			flags := flag.NewFlagSet("scanner-waits", flag.ContinueOnError)
			overrides.registerFlags(flags)
			if err := flags.Parse(tc.args); err != nil {
				t.Fatal(err)
			}
			if err := overrides.validate(); err != nil {
				t.Fatal(err)
			}
			b := NewBroker(nil, workerPaths{clusterDir: filepath.Join(t.TempDir(), "cluster"), scannerWaits: overrides})
			overrides.peerRefresh = time.Hour
			if got := b.scannerWaits.args(); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("scanner arguments=%q, want %q", got, tc.want)
			}
		})
	}
}

func TestBrokerScannerWaitOverridesRejectInvalidConfiguration(t *testing.T) {
	for name, overrides := range map[string]scannerWaitOverrides{
		"negative scan":            {scanInterval: -time.Second},
		"negative timeout":         {scanTimeout: -time.Second},
		"negative refresh":         {peerRefresh: -time.Second},
		"interval without timeout": {scanInterval: time.Second},
		"timeout without interval": {scanTimeout: time.Second},
		"timeout exceeds interval": {scanInterval: time.Second, scanTimeout: 2 * time.Second},
	} {
		t.Run(name, func(t *testing.T) {
			b := NewBroker(nil, workerPaths{scannerWaits: overrides})
			if err := b.Serve(context.Background()); err == nil {
				t.Fatal("invalid scanner waits accepted by Serve")
			}
			if handle, err := b.spawnScanner(); err == nil || handle != nil {
				t.Fatalf("invalid scanner waits spawned child: %v, %v", handle, err)
			}
		})
	}
}

func TestBrokerWaitIntervalsDefaultContract(t *testing.T) {
	if defaultResidentRetryInterval != 30*time.Second {
		t.Fatalf("resident retry default = %v, want 30s", defaultResidentRetryInterval)
	}
	if autoAdvertiseInterval != 5*time.Second {
		t.Fatalf("advertisement default = %v, want 5s", autoAdvertiseInterval)
	}
	if idleCheckInterval != 30*time.Second || residentObservationTimeout != 60*time.Second {
		t.Fatalf("idle/observation intervals = %v/%v, want 30s/60s", idleCheckInterval, residentObservationTimeout)
	}
	for name, b := range map[string]*Broker{
		"bare broker": &Broker{},
		"zero startup overrides": NewBroker(nil, workerPaths{
			clusterDir: filepath.Join(t.TempDir(), "cluster"),
		}),
	} {
		t.Run(name, func(t *testing.T) {
			if retry, advertise := b.modelConfigRetryInterval(), b.advertisementInterval(); retry != 30*time.Second || advertise != 5*time.Second {
				t.Fatalf("retry/advertisement = %v/%v, want 30s/5s", retry, advertise)
			}
		})
	}
}

func TestBrokerWaitIntervalFlagsConfigureBroker(t *testing.T) {
	test := func(name string, args []string, wantRetry, wantAdvertise time.Duration) {
		t.Run(name, func(t *testing.T) {
			flags := flag.NewFlagSet("broker-waits", flag.ContinueOnError)
			flags.SetOutput(io.Discard)
			retry, advertise := registerTestWaitFlags(flags)
			if err := flags.Parse(args); err != nil {
				t.Fatalf("parse fixture flags: %v", err)
			}
			if err := validateTestWaitIntervals(*retry, *advertise); err != nil {
				t.Fatalf("valid fixture intervals rejected: %v", err)
			}
			b := NewBroker(nil, workerPaths{
				clusterDir:         filepath.Join(t.TempDir(), "cluster"),
				residentRetryDelay: *retry,
				advertiseCadence:   *advertise,
			})
			if gotRetry, gotAdvertise := b.modelConfigRetryInterval(), b.advertisementInterval(); gotRetry != wantRetry || gotAdvertise != wantAdvertise {
				t.Fatalf("retry/advertisement = %v/%v, want %v/%v", gotRetry, gotAdvertise, wantRetry, wantAdvertise)
			}
		})
	}
	test("omitted flags", nil, 30*time.Second, 5*time.Second)
	test("resident retry only", []string{"--test-resident-retry-interval=1s"}, time.Second, 5*time.Second)
	test("advertisement only", []string{"--test-advertise-interval=250ms"}, 30*time.Second, 250*time.Millisecond)
	test("both overrides", []string{"--test-resident-retry-interval=1s", "--test-advertise-interval=250ms"}, time.Second, 250*time.Millisecond)
}

func TestBrokerWaitIntervalsInstanceIsolation(t *testing.T) {
	paths := workerPaths{
		clusterDir:         filepath.Join(t.TempDir(), "cluster"),
		residentRetryDelay: time.Second,
		advertiseCadence:   250 * time.Millisecond,
	}
	first := NewBroker(nil, paths)
	paths.residentRetryDelay = 2 * time.Second
	paths.advertiseCadence = 500 * time.Millisecond
	second := NewBroker(nil, paths)
	defaults := NewBroker(nil, workerPaths{clusterDir: paths.clusterDir})
	if retry, advertise := first.modelConfigRetryInterval(), first.advertisementInterval(); retry != time.Second || advertise != 250*time.Millisecond {
		t.Fatalf("first broker changed with startup config: %v/%v", retry, advertise)
	}
	if retry, advertise := second.modelConfigRetryInterval(), second.advertisementInterval(); retry != 2*time.Second || advertise != 500*time.Millisecond {
		t.Fatalf("second broker inherited first overrides: %v/%v", retry, advertise)
	}
	if retry, advertise := defaults.modelConfigRetryInterval(), defaults.advertisementInterval(); retry != 30*time.Second || advertise != 5*time.Second {
		t.Fatalf("default broker inherited overrides: %v/%v", retry, advertise)
	}
}

func TestBrokerWaitIntervalFlagsRejectNonpositive(t *testing.T) {
	test := func(name, arg, wantFlag string) {
		t.Run(name, func(t *testing.T) {
			flags := flag.NewFlagSet("broker-waits", flag.ContinueOnError)
			flags.SetOutput(io.Discard)
			retry, advertise := registerTestWaitFlags(flags)
			if err := flags.Parse([]string{arg}); err != nil {
				t.Fatalf("parse duration: %v", err)
			}
			if err := validateTestWaitIntervals(*retry, *advertise); err == nil || !strings.Contains(err.Error(), wantFlag+" must be positive") {
				t.Fatalf("validation = %v, want rejection naming %s", err, wantFlag)
			}
		})
	}
	test("explicit zero resident retry", "--test-resident-retry-interval=0s", "--test-resident-retry-interval")
	test("negative resident retry", "--test-resident-retry-interval=-1s", "--test-resident-retry-interval")
	test("explicit zero advertisement", "--test-advertise-interval=0s", "--test-advertise-interval")
	test("negative advertisement", "--test-advertise-interval=-1s", "--test-advertise-interval")
}

func TestBrokerWaitIntervalsInvalidConfigurationDoesNotStart(t *testing.T) {
	test := func(name string, retry, advertise time.Duration, wantFlag string) {
		t.Run(name, func(t *testing.T) {
			identityRoot := filepath.Join(t.TempDir(), "identity")
			b := NewBroker(nil, workerPaths{
				clusterDir:         filepath.Join(identityRoot, "cluster"),
				residentRetryDelay: retry,
				advertiseCadence:   advertise,
			})
			if err := b.Serve(context.Background()); err == nil || !strings.Contains(err.Error(), wantFlag) {
				t.Fatalf("Serve = %v, want invalid %s", err, wantFlag)
			}
			if b.nodeID != "" || b.cancel != nil || b.getScanner() != nil || len(b.regCache.Snapshot()) != 0 {
				t.Fatal("invalid intervals initialized operational state")
			}
			if _, err := os.Stat(identityRoot); !os.IsNotExist(err) {
				t.Fatalf("invalid intervals created identity directory or stat failed: %v", err)
			}
		})
	}
	test("negative resident retry", -time.Second, 0, "--test-resident-retry-interval")
	test("negative advertisement", 0, -time.Second, "--test-advertise-interval")
}

func TestBrokerWaitIntervalCLIRejectsBeforeWorkerResolution(t *testing.T) {
	binary := buildEngineManagerPortTestBinary(t, "ports-cli", "nvpair-ui-broker", ".", ".")
	test := func(name, arg, wantFlag string) {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			output, err := exec.CommandContext(ctx, binary, arg,
				"--scanner-path", filepath.Join(t.TempDir(), "missing-scanner")).CombinedOutput()
			if ctx.Err() != nil || err == nil {
				t.Fatalf("invalid interval did not fail promptly: %v, %v, %s", ctx.Err(), err, output)
			}
			if !strings.Contains(string(output), wantFlag+" must be positive") || strings.Contains(string(output), "scanner binary:") {
				t.Fatalf("diagnostic = %q, want interval rejection before worker resolution", output)
			}
		})
	}
	test("zero resident retry", "--test-resident-retry-interval=0s", "--test-resident-retry-interval")
	test("negative resident retry", "--test-resident-retry-interval=-1s", "--test-resident-retry-interval")
	test("zero advertisement", "--test-advertise-interval=0s", "--test-advertise-interval")
	test("negative advertisement", "--test-advertise-interval=-1s", "--test-advertise-interval")
}
