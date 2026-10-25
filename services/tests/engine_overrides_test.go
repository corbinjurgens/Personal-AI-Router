// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"nvpair-shared/engines"
)

// writeDisabledBuiltinEngineOverrides fills missing built-in manifests without
// replacing intentional private fake engines. Explicit empty auth statuses
// replace inherited values, which require the readiness probe disabled here.
func writeDisabledBuiltinEngineOverrides(t *testing.T, root, appDir string) {
	t.Helper()
	dir := filepath.Join(appDir, "engines")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	for _, engine := range engines.All() {
		path := filepath.Join(dir, engine.Name+".json")
		if _, err := os.Stat(path); err == nil {
			continue
		} else if !os.IsNotExist(err) {
			t.Fatal(err)
		}
		absent := filepath.Join(root, "absent-"+engine.Name)
		override := map[string]any{"engine": engine.Name, "platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{
			"detect": []string{absent}, "models_dir": filepath.Join(root, "absent-models", engine.Name),
			"runtime": map[string]any{
				"port": 0, "bin": absent, "cli": absent, "ready": nil, "health": nil, "auth_statuses": []int{},
			},
		}}}
		if engine.Name == "lmstudio" {
			// Its dormant delete action also requires readiness when restart_after
			// is true. Disable that flag to keep the null-probe override valid.
			override["actions"] = map[string]any{"delete_model": map[string]bool{"restart_after": false}}
		}
		data, err := json.Marshal(override)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
}

// privateEngineEnv is shared by ordinary broker and standalone-manager starts.
// Callers append their explicit private fixture selectors after these defaults.
func privateEngineEnv(t *testing.T, root string) []string {
	t.Helper()
	writeDisabledBuiltinEngineOverrides(t, root, jointAppDir(root))
	return append(os.Environ(), "HOME="+root, "XDG_CONFIG_HOME="+root,
		"APPDATA="+root, "LOCALAPPDATA="+root, "USERPROFILE="+root, "OLLAMA_HOST=")
}

func TestDisabledBuiltinOverridesPreservePrivateEngine(t *testing.T) {
	root := t.TempDir()
	app := jointAppDir(root)
	dir := filepath.Join(app, "engines")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	private := []byte(`{"engine":"ollama","runtime":{"port":12345,"ready":{"http":"http://127.0.0.1:{port}/api/version"}}}`)
	path := filepath.Join(dir, "ollama.json")
	if err := os.WriteFile(path, private, 0o600); err != nil {
		t.Fatal(err)
	}
	writeDisabledBuiltinEngineOverrides(t, root, app)
	actual, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(actual, private) {
		t.Fatalf("private engine changed: %q, %v", actual, err)
	}
	for _, engine := range engines.All() {
		if engine.Name == "ollama" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, engine.Name+".json"))
		if err != nil {
			t.Fatal(err)
		}
		var manifest struct {
			Platforms map[string]struct {
				Detect  []string `json:"detect"`
				Runtime struct {
					Port   int             `json:"port"`
					Ready  json.RawMessage `json:"ready"`
					Health json.RawMessage `json:"health"`
				} `json:"runtime"`
			} `json:"platforms"`
		}
		if err := json.Unmarshal(data, &manifest); err != nil {
			t.Fatal(err)
		}
		platform, ok := manifest.Platforms[runtime.GOOS+"/"+runtime.GOARCH]
		if !ok || len(platform.Detect) != 1 || platform.Detect[0] != filepath.Join(root, "absent-"+engine.Name) || platform.Runtime.Port != 0 || string(platform.Runtime.Ready) != "null" || string(platform.Runtime.Health) != "null" {
			t.Fatalf("%s does not have disabled private defaults: %s", engine.Name, data)
		}
	}
}
