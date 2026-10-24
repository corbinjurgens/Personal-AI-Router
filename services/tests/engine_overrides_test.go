// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"nvpair-shared/engines"
)

// writeDisabledBuiltinEngineOverrides keeps fixture managers from detecting or
// probing the host's engines. Explicit empty auth statuses replace inherited
// values, which otherwise require the readiness probe being disabled here.
func writeDisabledBuiltinEngineOverrides(t *testing.T, root, appDir string) {
	t.Helper()
	dir := filepath.Join(appDir, "engines")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	for _, engine := range engines.All() {
		override := map[string]any{"engine": engine.Name, "platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{
			"detect": []string{filepath.Join(root, "absent-"+engine.Name)},
			"runtime": map[string]any{
				"port": 0, "ready": nil, "health": nil, "auth_statuses": []int{},
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
		if err := os.WriteFile(filepath.Join(dir, engine.Name+".json"), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
}
