// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"io"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
)

func TestScannerTimingProductionDefaults(t *testing.T) {
	if defaultScanInterval != 5*time.Second {
		t.Errorf("defaultScanInterval = %v, want 5s", defaultScanInterval)
	}
	if defaultScanTimeout != 3*time.Second {
		t.Errorf("defaultScanTimeout = %v, want 3s", defaultScanTimeout)
	}
	if defaultPeerRefreshInterval != 15*time.Second {
		t.Errorf("defaultPeerRefreshInterval = %v, want 15s", defaultPeerRefreshInterval)
	}

	timings := defaultScannerTimings()
	fs := flag.NewFlagSet("scanner", flag.ContinueOnError)
	timings.registerFlags(fs)
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse default timing flags: %v", err)
	}
	want := scannerTimings{
		scanInterval:        5 * time.Second,
		scanTimeout:         3 * time.Second,
		peerRefreshInterval: 15 * time.Second,
	}
	if timings != want {
		t.Fatalf("default timings = %+v, want %+v", timings, want)
	}
	if err := timings.validate(); err != nil {
		t.Fatalf("production defaults rejected: %v", err)
	}
}

func TestScannerTimingFlagValidation(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantError string
	}{
		{"negative scan interval", []string{"--test-scan-interval=-1ms"}, "--test-scan-interval must be positive"},
		{"zero scan interval", []string{"--test-scan-interval=0"}, "--test-scan-interval must be positive"},
		{"negative scan timeout", []string{"--test-scan-timeout=-1ms"}, "--test-scan-timeout must be positive"},
		{"zero scan timeout", []string{"--test-scan-timeout=0"}, "--test-scan-timeout must be positive"},
		{"negative peer refresh", []string{"--test-peer-refresh-interval=-1ms"}, "--test-peer-refresh-interval must be positive"},
		{"zero peer refresh", []string{"--test-peer-refresh-interval=0"}, "--test-peer-refresh-interval must be positive"},
		{"browse exceeds scan interval", []string{"--test-scan-interval=2s"}, "--test-scan-timeout must not exceed --test-scan-interval"},
		{"equal browse and scan interval", []string{"--test-scan-interval=3s"}, ""},
		{"short fixture timings", []string{"--test-scan-interval=50ms", "--test-scan-timeout=10ms", "--test-peer-refresh-interval=25ms"}, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			timings := defaultScannerTimings()
			fs := flag.NewFlagSet("scanner", flag.ContinueOnError)
			timings.registerFlags(fs)
			if err := fs.Parse(tt.args); err != nil {
				t.Fatalf("parse timing flags: %v", err)
			}
			err := timings.validate()
			if tt.wantError == "" {
				if err != nil {
					t.Fatalf("valid timings rejected: %v", err)
				}
				return
			}
			if err == nil || err.Error() != tt.wantError {
				t.Fatalf("validation error = %v, want %q", err, tt.wantError)
			}
		})
	}
}

func TestScannerTimingFlagsRejectMalformedDuration(t *testing.T) {
	timings := defaultScannerTimings()
	fs := flag.NewFlagSet("scanner", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	timings.registerFlags(fs)
	err := fs.Parse([]string{"--test-scan-timeout=soon"})
	if err == nil || !strings.Contains(err.Error(), "test-scan-timeout") || !strings.Contains(err.Error(), "soon") {
		t.Fatalf("parse error = %v, want rejection naming the malformed timeout", err)
	}
}

func TestScannerTimingOverridesAreInstanceOwned(t *testing.T) {
	timings := defaultScannerTimings()
	fs := flag.NewFlagSet("scanner", flag.ContinueOnError)
	timings.registerFlags(fs)
	if err := fs.Parse([]string{"--test-scan-interval=50ms", "--test-scan-timeout=10ms", "--test-peer-refresh-interval=25ms"}); err != nil {
		t.Fatalf("parse timing overrides: %v", err)
	}
	if err := timings.validate(); err != nil {
		t.Fatalf("validate timing overrides: %v", err)
	}
	dir := filepath.Join(t.TempDir(), "cluster")
	overridden := newScannerWithTimings(nil, clustertrust.Open(dir), dir, nil, true, timings).daemon
	timings.scanInterval = time.Second
	timings.scanTimeout = time.Second
	timings.peerRefreshInterval = time.Second
	otherDir := filepath.Join(t.TempDir(), "cluster")
	ordinary := NewScanner(nil, clustertrust.Open(otherDir), otherDir, nil, true).daemon
	wantOverrides := scannerTimings{
		scanInterval:        50 * time.Millisecond,
		scanTimeout:         10 * time.Millisecond,
		peerRefreshInterval: 25 * time.Millisecond,
	}
	if overridden.timings != wantOverrides {
		t.Fatalf("overridden timings = %+v, want %+v", overridden.timings, wantOverrides)
	}
	wantDefaults := scannerTimings{
		scanInterval:        5 * time.Second,
		scanTimeout:         3 * time.Second,
		peerRefreshInterval: 15 * time.Second,
	}
	if ordinary.timings != wantDefaults {
		t.Fatalf("ordinary timings = %+v, want %+v", ordinary.timings, wantDefaults)
	}
	if overridden.browser == nil || ordinary.browser == nil || overridden.browser == ordinary.browser {
		t.Fatal("scanner instances must construct separate browsers")
	}
}
