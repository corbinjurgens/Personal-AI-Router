// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"
	"time"

	"nvpair-shared/noderec"
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
				if got, err := b.scannerArgs(); err != nil || !reflect.DeepEqual(got, want) {
					t.Fatalf("scanner argv = %q, %v, want %q", got, err, want)
				}
			}
		})
	}
}

func TestScannerNamespaceSurvivesSupervisedRestart(t *testing.T) {
	fixture := buildServicePortFixture(t)
	namespace := &noderec.MDNSNamespace{Service: "_PAIR-Fixture._TCP.", Domain: "Fixture-123.Local."}
	b := NewBroker(nil, workerPaths{scanner: fixture, scannerNamespace: namespace, clusterDir: filepath.Join(t.TempDir(), "cluster")})
	sup := newSupervisor("node-scanner", restartPolicy{baseDelay: 10 * time.Millisecond, maxDelay: 10 * time.Millisecond, maxAttempts: 3}, b.spawnScanner)
	if err := sup.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(sup.Stop)
	first := b.getScanner()
	assertArgs := func() {
		t.Helper()
		args := b.getScanner().cmd.Args
		for _, want := range [][]string{{"--mdns-service", "_pair-fixture._tcp"}, {"--mdns-domain", "fixture-123.local"}} {
			found := false
			for i := 0; i+1 < len(args); i++ {
				if args[i] == want[0] && args[i+1] == want[1] {
					found = true
				}
			}
			if !found {
				t.Fatalf("scanner argv %q missing %q", args, want)
			}
		}
	}
	assertArgs()
	if err := first.cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for b.getScanner() == first {
		if time.Now().After(deadline) {
			t.Fatal("scanner did not restart")
		}
		time.Sleep(10 * time.Millisecond)
	}
	assertArgs()
}

func TestScannerNamespaceInvalidBeforeIdentityAndSpawn(t *testing.T) {
	for _, tc := range []struct {
		name     string
		disabled bool
	}{
		{"multicast enabled", false},
		{"multicast disabled", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := filepath.Join(t.TempDir(), "must-not-exist")
			assertNoIdentity := func() {
				t.Helper()
				if _, err := os.Stat(root); !os.IsNotExist(err) {
					t.Fatalf("invalid namespace created identity directory or stat failed: %v", err)
				}
			}
			b := NewBroker(nil, workerPaths{
				scanner: "must-not-start", clusterDir: filepath.Join(root, "cluster"), disableMDNS: tc.disabled,
				scannerNamespace: &noderec.MDNSNamespace{Service: noderec.ServiceType},
			})
			assertNoIdentity()
			if b.nodeID != "" {
				t.Fatalf("invalid namespace resolved identity %q", b.nodeID)
			}
			if args, err := b.scannerArgs(); err == nil || args != nil {
				t.Fatalf("invalid namespace argv = %q, %v", args, err)
			}
			if err := b.Serve(context.Background()); err == nil {
				t.Fatal("Serve accepted invalid namespace")
			}
			if handle, err := b.spawnScanner(); err == nil || handle != nil {
				t.Fatalf("invalid namespace spawn = %v, %v", handle, err)
			}
			assertNoIdentity()
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
