// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"path/filepath"
	"reflect"
	"slices"
	"testing"
)

func TestProbeConfigScannerDefaultAndDisabledArguments(t *testing.T) {
	for _, tc := range []struct {
		name     string
		disabled bool
		want     []string
	}{
		{"production", false, nil},
		{"disabled before startup", true, []string{"--mdns=false"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clusterDir := filepath.Join(t.TempDir(), "cluster")
			b := NewBroker(nil, workerPaths{clusterDir: clusterDir, disableMDNS: tc.disabled})
			want := append([]string{"--cluster-dir", clusterDir}, tc.want...)
			for range 2 {
				if got := b.scannerArgs(); !reflect.DeepEqual(got, want) {
					t.Fatalf("scanner argv = %q, want %q", got, want)
				}
			}
		})
	}
}

func TestProbeConfigManualExplicitZeroAndPrivateTargets(t *testing.T) {
	for _, tc := range []struct {
		name  string
		ports *manualProbePorts
		want  []string
	}{
		{"worker defaults", nil, nil},
		{"disabled", &manualProbePorts{}, []string{"--ollama-port", "0", "--lmstudio-port", "0", "--node-info-port", "0"}},
		{"private", &manualProbePorts{25101, 25102, 25103}, []string{"--ollama-port", "25101", "--lmstudio-port", "25102", "--node-info-port", "25103"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := NewBroker(nil, workerPaths{manualProbePorts: tc.ports, clusterDir: filepath.Join(t.TempDir(), "cluster")})
			base := append(b.logLevelArgs(), b.clusterDirArgs()...)
			want := append(base, tc.want...)
			for range 2 {
				if got := b.manualNodesArgs(); !slices.Equal(got, want) {
					t.Fatalf("manual-node argv = %q, want %q", got, want)
				}
			}
		})
	}
}

func TestProbeConfigManualRejectsInvalidTargets(t *testing.T) {
	for _, ports := range []*manualProbePorts{{-1, 0, 0}, {0, 65536, 0}, {0, 0, -1}} {
		if err := ports.validate(); err == nil {
			t.Fatalf("invalid probe targets accepted: %+v", ports)
		}
	}
}
