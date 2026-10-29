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

func TestDetectRequiredValidation(t *testing.T) {
	test := func(name string, version int, required, detect []string, wantError string) {
		t.Run(name, func(t *testing.T) {
			m := validManifest()
			m.ManifestVersion = version
			p := m.Platforms["linux/amd64"]
			p.DetectRequired, p.Detect = required, detect
			m.Platforms["linux/amd64"] = p
			err := m.Validate()
			if wantError == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), wantError) {
				t.Fatalf("validation = %v, want %q", err, wantError)
			}
		})
	}
	test("version 1 omitted", 1, nil, []string{"engine"}, "")
	test("version 1 empty", 1, []string{}, []string{"engine"}, "")
	test("version 1 mandatory evidence rejected", 1, []string{"config.json"}, []string{"engine"}, "requires manifest_version 2")
	test("version 2 mandatory evidence", 2, []string{"{install_dir}/config.json"}, []string{"engine"}, "")
	test("version 2 empty", 2, []string{}, []string{"engine"}, "")
	test("future version rejected", ManifestSchemaVersion+1, nil, []string{"engine"}, "newer than supported")
	test("requires executable alternative", 2, []string{"config.json"}, nil, "requires detect")
	test("blank path", 2, []string{" "}, []string{"engine"}, "must not be blank")
	test("unknown placeholder", 2, []string{"{instal_dir}/config.json"}, []string{"engine"}, "unknown placeholder")
	test("known but unavailable directory validated at detection", 2, []string{"{models_dir}/config.json"}, []string{"engine"}, "")
	test("environment braces are not runner placeholders", 2, []string{"${PAIR_DETECTION_ROOT}/config.json"}, []string{"engine"}, "")
	test("interpreter environment braces", 2, []string{"config.json"}, []string{"${PAIR_DETECTION_ROOT}/interpreter"}, "")
}

func TestDetectRequiredSharedDefaults(t *testing.T) {
	reg := NewRegistry()
	_, err := reg.addManifest("synthetic", []byte(`{
		"engine":"configured", "display_name":"Configured", "manifest_version":2,
		"detect":["interpreter"], "detect_required":["server.py","config.json"],
		"runtime":{"bin":"interpreter"},
		"platforms":{
			"linux/amd64":{},
			"windows/amd64":{"detect_required":["windows-config.json"]},
			"darwin/arm64":{"detect_required":[]}
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	m, ok := reg.Get("configured")
	if !ok {
		t.Fatal("configured manifest missing")
	}
	for _, tc := range []struct {
		name string
		want []string
	}{
		{"linux/amd64", []string{"server.py", "config.json"}},
		{"windows/amd64", []string{"windows-config.json"}},
		{"darwin/arm64", []string{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, ok := m.Platforms[tc.name]
			if !ok || !slices.Equal(p.DetectRequired, tc.want) || !slices.Equal(p.Detect, []string{"interpreter"}) {
				t.Fatalf("platform = %+v, want required %v", p, tc.want)
			}
		})
	}
}

func TestDetectRequiredVersionCheckedAfterSharedMerge(t *testing.T) {
	reg := NewRegistry()
	_, err := reg.addManifest("synthetic-v1", []byte(`{
		"engine":"configured", "display_name":"Configured", "manifest_version":1,
		"detect":["interpreter"], "detect_required":["config.json"],
		"runtime":{"bin":"interpreter"}, "platforms":{"linux/amd64":{}}
	}`))
	if err == nil || !strings.Contains(err.Error(), "requires manifest_version 2") {
		t.Fatalf("version 1 shared mandatory evidence = %v", err)
	}
	if _, ok := reg.Get("configured"); ok {
		t.Fatal("rejected manifest registered")
	}
}

func TestDetectRequiredVersionCannotHideSharedDeclaration(t *testing.T) {
	reg := NewRegistry()
	_, err := reg.addManifest("synthetic-v1", []byte(`{
		"engine":"configured", "display_name":"Configured", "manifest_version":1,
		"detect":["interpreter"], "detect_required":["config.json"],
		"runtime":{"bin":"interpreter"},
		"platforms":{"linux/amd64":{"detect_required":[]}}
	}`))
	if err == nil || !strings.Contains(err.Error(), "requires manifest_version 2") {
		t.Fatalf("version 1 shared declaration hidden by platform clear = %v", err)
	}
}

func TestDetectRequiredUserOverrideVersion(t *testing.T) {
	test := func(name, override string, wantVersion int, want []string) {
		t.Run(name, func(t *testing.T) {
			reg := NewRegistry()
			if err := reg.LoadFS(bundledManifests, "manifests"); err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "ollama.json"), []byte(override), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := reg.LoadOverrideDir(dir); err != nil {
				t.Fatal(err)
			}
			m, ok := reg.Get("ollama")
			if wantVersion == 1 {
				// Invalid overrides now disable the engine instead of retaining
				// a bundled manifest that silently ignores detect_required.
				if ok || reg.invalid["ollama"] == nil || !strings.Contains(reg.invalid["ollama"].Error(), "requires manifest_version 2") {
					t.Fatalf("invalid override = %v, %v", m, reg.invalid["ollama"])
				}
				return
			}
			if !ok || m.ManifestVersion != wantVersion {
				t.Fatalf("loaded manifest = %+v", m)
			}
			for platform, p := range m.Platforms {
				if !slices.Equal(p.DetectRequired, want) {
					t.Errorf("%s required = %v, want %v", platform, p.DetectRequired, want)
				}
			}
		})
	}
	test("version 1 cannot silently ignore requirement", `{"engine":"ollama","detect_required":["config.json"]}`, 1, nil)
	test("version 2 adds shared requirement", `{"engine":"ollama","manifest_version":2,"detect_required":["config.json"]}`, 2, []string{"config.json"})
}
