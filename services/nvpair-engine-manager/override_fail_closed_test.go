// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
)

func TestInvalidOverrideDoesNotUseBundledEngine(t *testing.T) {
	dir := t.TempDir()
	bundle := t.TempDir()
	manifest := testEngineManifest(filepath.Join(t.TempDir(), "absent-engine"))
	platform := manifest.Platforms[hostKey()]
	platform.Runtime.Port = 0
	platform.Install = &Install{Fetch: &Fetch{URL: "https://example.invalid/engine"}, Run: []string{"{download}"}}
	manifest.Platforms[hostKey()] = platform
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("encode bundled fixture: %v", err)
	}
	if err := os.WriteFile(filepath.Join(bundle, "fake.json"), data, 0o600); err != nil {
		t.Fatalf("write bundled fixture: %v", err)
	}
	override := `{"engine":"fake","platforms":{"` + hostKey() + `":{"install":{"artifacts":[{"name":"engine","url":"https://example.invalid/engine","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}}}}`
	if err := os.WriteFile(filepath.Join(dir, "fake.json"), []byte(override), 0o600); err != nil {
		t.Fatalf("write override: %v", err)
	}
	reg := NewRegistry()
	if err := reg.LoadFS(os.DirFS(bundle), "."); err != nil {
		t.Fatalf("load bundled fixture: %v", err)
	}
	if err := reg.LoadOverrideDir(dir); err != nil {
		t.Fatalf("load overrides: %v", err)
	}
	ex := NewExecutor(reg, NewReporter(nil), func(string, any) {}, t.TempDir())
	probeCalls := 0
	ex.client.Transport = healthProbeTransport(func(*http.Request) (*http.Response, error) {
		probeCalls++
		return nil, fmt.Errorf("unexpected probe")
	})
	statuses := ex.GetInstalled()
	if len(statuses) != 1 || statuses[0].Engine != "fake" || statuses[0].Installed || statuses[0].Running || statuses[0].Healthy {
		t.Errorf("installed statuses = %+v, want unavailable fake engine", statuses)
	}
	if probeCalls != 0 {
		t.Errorf("probe calls = %d, want 0", probeCalls)
	}
	const diagnostic = "install.fetch and install.artifacts are mutually exclusive"
	found := false
	for _, problem := range ex.Errors() {
		if problem.EngineType == "fake" && strings.Contains(problem.Message, diagnostic) {
			found = true
		}
	}
	if !found {
		t.Errorf("engine errors = %+v, want validation diagnostic %q", ex.Errors(), diagnostic)
	}
	if _, err := ex.state("fake"); err == nil || !strings.Contains(err.Error(), diagnostic) {
		t.Errorf("state error = %v, want validation diagnostic %q", err, diagnostic)
		return // The bundled engine is usable; do not attempt a real launch.
	}
	if err := ex.Start(context.Background(), "fake"); err == nil || !strings.Contains(err.Error(), diagnostic) {
		t.Errorf("start error = %v, want validation diagnostic %q", err, diagnostic)
	}
	if probeCalls != 0 {
		t.Errorf("probe calls after start = %d, want 0", probeCalls)
	}
}

func TestOverrideFailureAndRepairFenceCachedState(t *testing.T) {
	bundle, dir := t.TempDir(), t.TempDir()
	manifest := testEngineManifest(filepath.Join(t.TempDir(), "absent"))
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bundle, "fake.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	reg := NewRegistry()
	if err := reg.LoadFS(os.DirFS(bundle), "."); err != nil {
		t.Fatal(err)
	}
	ex := NewExecutor(reg, NewReporter(nil), nil, t.TempDir())
	probeCalls := 0
	ex.client.Transport = healthProbeTransport(func(*http.Request) (*http.Response, error) {
		probeCalls++
		return nil, fmt.Errorf("unexpected probe")
	})
	if _, err := ex.state("fake"); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(dir, "fake.json")
	if err := os.WriteFile(file, []byte(`{"engine":`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := reg.LoadOverrideDir(dir); err != nil {
		t.Fatal(err)
	}
	if _, err := ex.state("fake"); err == nil {
		t.Fatal("cached state bypassed invalid override")
	}
	status, err := ex.StatusAtPort("fake", 12345)
	if err != nil || status.Installed || status.Running || status.Healthy {
		t.Fatalf("status = %+v, error = %v", status, err)
	}
	if err := ex.restoreEnabled(context.Background(), "fake"); err == nil {
		t.Fatal("restore accepted invalid override")
	}
	if probeCalls != 0 {
		t.Fatalf("invalid status or restore made %d HTTP calls", probeCalls)
	}
	if err := os.WriteFile(file, []byte(`{"engine":"fake","display_name":"Repaired"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := reg.LoadOverrideDir(dir); err != nil {
		t.Fatal(err)
	}
	// Production loads overrides at startup, not into a live executor. Repair
	// gets a fresh manager so no cached runtime inputs are partially replaced.
	var frames bytes.Buffer
	ex = NewExecutor(reg, NewReporter(NewCodec(&frames)), nil, t.TempDir())
	st, err := ex.state("fake")
	if err != nil || st.manifest.DisplayName != "Repaired" {
		t.Fatalf("repaired state = %+v, error = %v", st, err)
	}
	for _, problem := range ex.Errors() {
		if problem.EngineType == "fake" {
			t.Fatalf("stale error after repair: %+v", problem)
		}
	}
	cleared := false
	decoder := json.NewDecoder(&frames)
	for decoder.More() {
		var frame struct {
			Method string      `json:"method"`
			Params clearParams `json:"params"`
		}
		if err := decoder.Decode(&frame); err != nil {
			t.Fatalf("decode restart notification: %v", err)
		}
		if frame.Method == "errors:clear" && frame.Params.ID == "engine-manager:manifest:fake" {
			cleared = true
		}
	}
	if !cleared {
		t.Fatal("fresh manager did not clear the surviving errors worker's manifest error")
	}
}

func TestInvalidOverrideDescriptorsRemainUnavailable(t *testing.T) {
	reg := buildBundledRegistry()
	reg.invalid["ollama"] = fmt.Errorf("invalid override fixture")
	found := false
	for _, descriptor := range reg.Descriptors() {
		if err := descriptor.Validate(); err != nil {
			t.Fatalf("descriptor is invalid: %v", err)
		}
		if descriptor.Engine == "ollama" {
			found = true
			if descriptor.Available || len(descriptor.Operations) != 0 || descriptor.Catalog != "" || descriptor.ModelStore != "" {
				t.Fatalf("invalid engine advertises capabilities: %+v", descriptor)
			}
		}
	}
	if !found {
		t.Fatal("invalid engine disappeared from descriptors")
	}
}

func TestOverrideDirectoryReadFailureDisablesBundledEngines(t *testing.T) {
	reg := buildBundledRegistry()
	path := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(path, []byte("fixture"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := reg.LoadOverrideDir(path); err == nil {
		t.Fatal("expected directory read error")
	}
	for _, name := range reg.Names() {
		if _, ok := reg.Get(name); ok || reg.invalid[name] == nil {
			t.Fatalf("unreadable overrides left bundled %s usable", name)
		}
	}
}

func TestUnreadableOverrideDisablesNamedEngine(t *testing.T) {
	reg := buildBundledRegistry()
	dir := t.TempDir()
	if err := os.Symlink(filepath.Join(dir, "absent"), filepath.Join(dir, "ollama.json")); err != nil {
		if runtime.GOOS == "windows" && errors.Is(err, syscall.Errno(1314)) {
			t.Skip("symlink fixture requires Windows symlink privileges or Developer Mode")
		}
		t.Fatal(err)
	}
	if err := reg.LoadOverrideDir(dir); err != nil {
		t.Fatal(err)
	}
	if _, ok := reg.Get("ollama"); ok || reg.invalid["ollama"] == nil {
		t.Fatal("unreadable override fell back to the bundled manifest")
	}
	if _, ok := reg.Get("lmstudio"); !ok {
		t.Fatal("unrelated engine became unavailable")
	}
}

func TestOverrideManifestDirectoryDisablesNamedEngine(t *testing.T) {
	reg := buildBundledRegistry()
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "ollama.json"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := reg.LoadOverrideDir(dir); err != nil {
		t.Fatal(err)
	}
	if _, ok := reg.Get("ollama"); ok || reg.invalid["ollama"] == nil {
		t.Fatal("manifest directory fell back to bundled engine")
	}
	if _, ok := reg.Get("lmstudio"); !ok {
		t.Fatal("unrelated engine became unavailable")
	}
}

func TestInvalidOverrideDoesNotSkipOwnedShutdown(t *testing.T) {
	reg := NewRegistry()
	data, err := json.Marshal(testEngineManifest(filepath.Join(t.TempDir(), "absent")))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := reg.addManifest("fixture", data); err != nil {
		t.Fatal(err)
	}
	ex := NewExecutor(reg, NewReporter(nil), nil, t.TempDir())
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	canceled := false
	st.startCancel = func(error) { canceled = true }
	reg.invalid["fake"] = fmt.Errorf("invalid fixture override")
	ex.StopAll()
	if !canceled {
		t.Fatal("invalid override suppressed cleanup of an owned start")
	}
}

func TestOverrideConflictingIdentityKeysRefuseWithoutNilManifest(t *testing.T) {
	reg := buildBundledRegistry()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "ollama.json"), []byte(`{"engine":"lmstudio","Engine":"ollama"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := reg.LoadOverrideDir(dir); err != nil {
		t.Fatal(err)
	}
	if _, ok := reg.Get("ollama"); ok || reg.invalid["ollama"] == nil {
		t.Fatal("conflicting identity remained available")
	}
	ex := NewExecutor(reg, NewReporter(nil), nil, t.TempDir())
	if _, err := ex.state("ollama"); err == nil {
		t.Fatal("conflicting identity admitted state")
	}
	for _, descriptor := range reg.Descriptors() {
		if err := descriptor.Validate(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestOverrideIdentityConflictCannotReactivateEngine(t *testing.T) {
	dir := t.TempDir()
	reg := NewRegistry()
	if err := reg.LoadFS(bundledManifests, "manifests"); err != nil {
		t.Fatal(err)
	}
	for name, contents := range map[string]string{
		"ollama.json": `{"engine":"ollama","display_name":"valid"}`,
		"other.json":  `{"engine":"ollama","display_name":"conflict"}`,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(contents), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := reg.LoadOverrideDir(dir); err != nil {
		t.Fatal(err)
	}
	if _, ok := reg.Get("ollama"); ok || reg.invalid["ollama"] == nil {
		t.Fatal("conflicting override reactivated ollama")
	}
	if _, ok := reg.Get("lmstudio"); !ok {
		t.Fatal("unrelated engine was disabled")
	}
}

func TestInvalidOverrideLeavesOtherEnginesAvailable(t *testing.T) {
	for _, tc := range []struct{ name, contents string }{
		{"malformed JSON", `{"engine":`},
		{"validation failure", `{"engine":"ollama","platforms":{"linux/amd64":{"install":{"run":[]}}}}`},
		{"merge failure", `{"engine":"ollama","platforms":42}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reg := NewRegistry()
			if err := reg.LoadFS(bundledManifests, "manifests"); err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "ollama.json"), []byte(tc.contents), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := reg.LoadOverrideDir(dir); err != nil {
				t.Fatal(err)
			}
			if _, ok := reg.Get("ollama"); ok || reg.invalid["ollama"] == nil {
				t.Fatal("invalid engine remained available")
			}
			if _, ok := reg.Get("lmstudio"); !ok {
				t.Fatal("unrelated engine became unavailable")
			}
		})
	}
}

func TestValidPartialOverrideRetainsBundledFields(t *testing.T) {
	reg := NewRegistry()
	if err := reg.LoadFS(bundledManifests, "manifests"); err != nil {
		t.Fatal(err)
	}
	base, ok := reg.Get("ollama")
	if !ok {
		t.Fatal("missing bundled ollama")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "ollama.json"), []byte(`{"engine":"ollama","display_name":"Custom"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := reg.LoadOverrideDir(dir); err != nil {
		t.Fatal(err)
	}
	got, ok := reg.Get("ollama")
	if !ok || got.DisplayName != "Custom" || len(got.Platforms) != len(base.Platforms) {
		t.Fatalf("partial overlay = %+v", got)
	}
}
