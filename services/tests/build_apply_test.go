// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"debug/buildinfo"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"nvpair-shared/buildapply"
	"nvpair-shared/testbin"
)

// This exercises only admission paths reachable without an O1 manager
// reservation. It does not establish transaction or helper acceptance.
func TestBuildApplyProcess(t *testing.T) {
	if runtime.GOOS != "windows" || !filepath.IsAbs(os.Getenv(testbin.EnvRoot)) || !strings.Contains(os.Getenv("GOFLAGS"), "nvpair-test-exec") {
		t.Fatal("process proof requires PC3's Windows fixed-path wrapper")
	}
	firewall := readOwnerFirewallSnapshot(t)
	if ownerEvidenceBuildRace(t) {
		inheritedRace := false
		for _, flag := range strings.Fields(os.Getenv("GOFLAGS")) {
			switch flag {
			case "-race", "-race=true":
				inheritedRace = true
			case "-race=false":
				inheritedRace = false
			}
		}
		if !inheritedRace {
			t.Fatal("race process proof requires -race in inherited GOFLAGS so TestMain builds instrumented children")
		}
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	actual, err := serviceExecutablePath(self)
	if err != nil {
		t.Fatal(err)
	}
	expected, err := serviceExecutablePath(filepath.Join(os.Getenv(testbin.EnvRoot), "go-test", "tests", "tests.test.exe"))
	if err != nil || actual != expected {
		t.Fatalf("test runner is not at its fixed destination: %q: %v", self, err)
	}
	requireOwnerFirewallRules(t, firewall, []string{self})
	before, err := snapshotInstalledServiceProcesses()
	if err != nil {
		t.Fatal(err)
	}
	f := newProxyServiceFixture(t)
	manager := f.copyBinary(engineMgrBin)
	f.inventory.paths = append(f.inventory.paths, manager)
	// TestMain builds products separately from the test binary. Require the
	// same race instrumentation before any product image is launched.
	if ownerEvidenceBuildRace(t) {
		for _, path := range f.inventory.paths {
			info, err := buildinfo.ReadFile(path)
			if err != nil {
				t.Fatalf("read build settings for %q: %v", path, err)
			}
			race := false
			for _, setting := range info.Settings {
				if setting.Key == "-race" {
					race, err = strconv.ParseBool(setting.Value)
					if err != nil {
						t.Fatalf("race setting for %q: %v", path, err)
					}
				}
			}
			if !race {
				t.Fatalf("race test requires a race-instrumented product image: %q", path)
			}
		}
	}
	f.env = ownerEvidencePrivateEnv(f.env)
	f.additionalBrokerArgs = []string{"--scanner-mdns=false", "--manual-ollama-port=0", "--manual-lmstudio-port=0", "--manual-node-info-port=0", "--engine-manager-control-host=127.0.0.1"}
	requireOwnerFirewallRules(t, firewall, append(f.inventory.ownedPaths(), self))
	t.Cleanup(func() {
		if err := f.cleanupChecked(); err != nil {
			t.Error(err)
			return
		}
		after, err := snapshotInstalledServiceProcesses()
		if err != nil {
			t.Error(err)
		} else if !reflect.DeepEqual(before, after) {
			t.Errorf("installed process or listener snapshot changed: before=%+v after=%+v", before, after)
		}
	})
	f.start()
	c := f.readyClient()
	lifetime := serviceStatus(t, c)
	type savedFile struct {
		Exists bool
		Data   []byte
	}
	captureSaved := func(t *testing.T) map[string]savedFile {
		t.Helper()
		state := make(map[string]savedFile)
		for _, path := range []string{
			f.policyPath(),
			filepath.Join(f.dir, "engine-settings-operations.json"),
			filepath.Join(jointAppDir(f.configDir), "engine-state.json"),
		} {
			data, err := os.ReadFile(path)
			if err != nil && !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("capture private saved state %q: %v", path, err)
			}
			state[path] = savedFile{Exists: err == nil, Data: data}
		}
		return state
	}
	savedBefore := captureSaved(t)
	assertSaved := func(t *testing.T) {
		t.Helper()
		if !reflect.DeepEqual(savedBefore, captureSaved(t)) {
			t.Fatal("refusal changed private saved policy, settings, intent or locks")
		}
	}
	status := func(id string) buildapply.Result[buildapply.BuildApplyStatus] {
		t.Helper()
		frame := c.mustCall(t, id, "service/build-apply-status", mustMarshal(t, buildapply.StatusRequest{}))
		result, err := buildapply.DecodeStatusResult(frame.Result, "")
		if err != nil || !result.Available || result.Value == nil || result.Value.State != buildapply.StatusIdle {
			t.Fatalf("idle apply status: %+v, %v", result, err)
		}
		return result
	}
	initial := status("822")
	gui := buildapply.GUIFacts{PID: os.Getpid(), Executable: self}
	t.Run("preview refuses a nonpackaged running layout", func(t *testing.T) {
		frame := c.mustCall(t, "823", "service/build-apply-preview", mustMarshal(t, buildapply.PreviewRequest{CandidatePath: filepath.Join(f.dir, "absent-candidate"), GUI: gui}))
		var result buildapply.Result[buildapply.BuildApplyPreview]
		if err := json.Unmarshal(frame.Result, &result); err != nil {
			t.Fatal(err)
		}
		if result.Available || result.Value != nil || result.Refusal == nil || result.Refusal.Code != "unsafe_package" {
			t.Fatalf("preview refusal: %+v", result)
		}
		assertSaved(t)
	})
	for _, tc := range []struct {
		name    string
		consent buildapply.Consent
		code    string
	}{
		{"missing consent", buildapply.Consent{}, "interruption_required"},
		{"engine stop not consented", buildapply.Consent{AllowInterruption: true}, "interruption_required"},
		{"interruption not consented", buildapply.Consent{AllowEngineStop: true}, "interruption_required"},
		{"missing preview", buildapply.Consent{AllowInterruption: true, AllowEngineStop: true}, "stale_preview"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			frame := c.mustCall(t, "824", "service/build-apply-start", mustMarshal(t, buildapply.StartRequest{PreviewID: strings.Repeat("a", 32), GUI: gui, Consent: tc.consent}))
			var result buildapply.Result[buildapply.BuildApplyOperation]
			if err := json.Unmarshal(frame.Result, &result); err != nil {
				t.Fatal(err)
			}
			if result.Available || result.Value != nil || result.Refusal == nil || result.Refusal.Code != tc.code {
				t.Fatalf("start refusal: %+v", result)
			}
			assertSaved(t)
		})
	}
	if got := status("825"); !reflect.DeepEqual(initial, got) {
		t.Fatalf("refusal changed apply status: before=%+v after=%+v", initial, got)
	}
	after := serviceStatus(t, c)
	if after.PID != lifetime.PID || after.BrokerPID != lifetime.BrokerPID || after.StartedAt != lifetime.StartedAt || after.BrokerRestarts != lifetime.BrokerRestarts {
		t.Fatalf("refusal changed service lifetime: before=%+v after=%+v", lifetime, after)
	}
}
