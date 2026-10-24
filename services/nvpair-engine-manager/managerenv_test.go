// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestManagerLaunchesUseIsolatedEnv keeps every manager process a test starts
// on isolatedManagerEnv, so a new launch site cannot inherit the real profile
// and then adopt or stop an installed engine.
func TestManagerLaunchesUseIsolatedEnv(t *testing.T) {
	files, err := filepath.Glob("*_test.go")
	if err != nil {
		t.Fatal(err)
	}
	launches := 0
	for _, file := range files {
		if file == "managerenv_test.go" {
			continue
		}
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(string(data), "\n")
		// Live-tagged tests deliberately reach real engines and are owner-gated.
		if slices.ContainsFunc(lines[:min(5, len(lines))], func(line string) bool {
			return strings.HasPrefix(strings.TrimSpace(line), "//go:build live")
		}) {
			continue
		}
		for i, line := range lines {
			if !strings.Contains(line, "exec.Command(managerBin") {
				continue
			}
			launches++
			if !slices.ContainsFunc(lines[i+1:min(i+4, len(lines))], func(next string) bool {
				return strings.Contains(next, "cmd.Env = isolatedManagerEnv(")
			}) {
				t.Errorf("%s:%d: manager launch does not set cmd.Env = isolatedManagerEnv(...) within the next 3 lines", file, i+1)
			}
		}
	}
	// Guards against a glob or pattern mistake passing vacuously.
	if launches < 3 {
		t.Fatalf("found %d manager launch sites, want at least 3", launches)
	}
}

func TestIsolatedManagerEnvClearsAddressOverrides(t *testing.T) {
	for _, key := range []string{"FAKE_ADDR", "OLLAMA_HOST", "NVPAIR_EM_LIVE_TEST"} {
		t.Setenv(key, "ambient-value")
	}
	env := isolatedManagerEnv(t.TempDir(), t.TempDir())
	for _, key := range []string{"FAKE_ADDR", "OLLAMA_HOST", "NVPAIR_EM_LIVE_TEST"} {
		var values []string
		for _, entry := range env {
			name, value, ok := strings.Cut(entry, "=")
			if ok && strings.EqualFold(name, key) {
				values = append(values, value)
			}
		}
		if !slices.Equal(values, []string{""}) {
			t.Errorf("%s overrides = %q, want exactly one empty assignment", key, values)
		}
	}
}

// Read the generated bundle back through the production loader. A future
// fixture change must not reintroduce a vendor binary, path or endpoint.
func TestManagerFixtureBundleIsPrivate(t *testing.T) {
	dir := t.TempDir()
	if _, err := managerFixtureOverlay(dir); err != nil {
		t.Fatal(err)
	}
	reg := NewRegistry()
	if err := reg.LoadFS(os.DirFS(filepath.Join(dir, "bundled")), "."); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(reg.Names(), []string{"llamacpp", "lmstudio", "ollama"}) {
		t.Fatalf("fixture engines = %v", reg.Names())
	}
	for _, name := range reg.Names() {
		manifest, _ := reg.Get(name)
		platform, ok := manifest.HostPlatform()
		if !ok {
			t.Fatalf("%s has no host platform", name)
		}
		rt := platform.Runtime
		if rt.Bin != fakeEngineBin || !slices.Equal(platform.Detect, []string{fakeEngineBin}) || platform.Install != nil || platform.Uninstall != nil || platform.ModelsDir != "" {
			t.Errorf("%s can reach a native installation: %+v", name, platform)
		}
		if rt.Port <= 0 || slices.Contains([]int{11434, 1234, 1235, 8080, 8081}, rt.Port) || rt.Bind != "" || rt.Mode != "" || len(rt.Start) != 0 || rt.CLI != "" {
			t.Errorf("%s does not use a private process port: %+v", name, rt)
		}
		if rt.Env["FAKE_ADDR"] != "{host}:{port}" || rt.Env["OLLAMA_HOST"] != "{host}:{port}" || rt.Ready == nil || rt.Ready.HTTP != "http://127.0.0.1:{port}/" || rt.Health == nil || rt.Health.HTTP != rt.Ready.HTTP {
			t.Errorf("%s can escape the fixture address: %+v", name, rt)
		}
	}
}
