// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"nvpair-shared/jsonrpc"
)

func TestStrataContractUnsupportedManagement(t *testing.T) {
	strataContractPlatform(t)

	assertError := func(t *testing.T, response jsonrpc.Message, diagnostic string) {
		t.Helper()
		if response.Error == nil || response.Error.Code != -32000 || response.Error.Message != diagnostic {
			t.Fatalf("RPC error = %+v, want code -32000 and message %q", response.Error, diagnostic)
		}
	}
	assertAbsent := func(t *testing.T, f *strataContractFixture) {
		t.Helper()
		var status strataContractStatus
		f.Manager.strataContractOK(t, "engine:status", map[string]string{"engine": "strata"}, &status)
		want := strataContractStatus{Engine: "strata", Port: f.Port}
		if status != want {
			t.Fatalf("absent runtime status = %+v, want %+v", status, want)
		}
		strataContractWaitClosed(t, f.Port)
		if _, err := os.Stat(filepath.Join(f.Root, "runtime-pid.json")); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("absent runtime launch marker: %v, want not exist", err)
		}
	}

	t.Run("absent install rejects missing install block", func(t *testing.T) {
		f := strataContractNew(t, false)
		f.strataContractStartManager(t)
		assertAbsent(t, f)
		before := strataContractU3Artifacts(t, f, "")
		response := f.Manager.strataContractCall(t, "engine:install", map[string]string{"engine": "strata"})
		assertError(t, response, `engine "strata" has no install block for this platform`)
		assertAbsent(t, f)
		if after := strataContractU3Artifacts(t, f, ""); !reflect.DeepEqual(after, before) {
			t.Fatalf("rejected install changed private artifacts: before=%v after=%v", before, after)
		}
	})

	t.Run("absent uninstall succeeds without installing or launching", func(t *testing.T) {
		f := strataContractNew(t, false)
		f.strataContractStartManager(t)
		assertAbsent(t, f)
		// The no-op legitimately saves Off intent, but creates no installation
		// or model store. Compare every other private file and directory.
		intentPath := filepath.Join(f.AppDir, "engine-bin", "engine-state.json")
		before := strataContractU3Artifacts(t, f, intentPath)
		var status strataContractStatus
		f.Manager.strataContractOK(t, "engine:uninstall", map[string]string{"engine": "strata"}, &status)
		want := strataContractStatus{Engine: "strata", Port: f.Port}
		if status != want {
			t.Fatalf("uninstall status = %+v, want %+v", status, want)
		}
		assertAbsent(t, f)
		if after := strataContractU3Artifacts(t, f, intentPath); !reflect.DeepEqual(after, before) {
			t.Fatalf("absent uninstall changed private artifacts: before=%v after=%v", before, after)
		}
		data, err := os.ReadFile(intentPath)
		if err != nil {
			t.Fatalf("read saved Off intent: %v", err)
		}
		var intent struct {
			Engines map[string]bool   `json:"engines"`
			OffBy   map[string]string `json:"offBy"`
		}
		strataContractDecode(t, data, &intent)
		if !reflect.DeepEqual(intent.Engines, map[string]bool{"strata": false}) || !reflect.DeepEqual(intent.OffBy, map[string]string{"strata": "local"}) {
			t.Fatalf("saved intent = %+v, want Strata Off by local", intent)
		}
	})

	testRunning := func(name string, action func(*testing.T, *strataContractFixture)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			f := strataContractNew(t, true)
			f.strataContractStartManager(t)
			f.strataContractStartRuntime(t)
			before := strataContractU3Artifacts(t, f, "")
			action(t, f)
			f.strataContractWaitStatus(t, true)
			status, _, body := strataContractHTTP(t, http.MethodGet, f.Port, "/health", nil)
			var health struct {
				Service string `json:"service"`
				Fixture string `json:"fixture"`
				PID     int    `json:"pid"`
			}
			strataContractDecode(t, body, &health)
			if status != http.StatusOK || health.Service != "strata" || health.Fixture != f.Identity || health.PID != f.runtimeProcess.Pid {
				t.Fatalf("owned runtime health changed: status=%d health=%+v", status, health)
			}
			control := f.strataContractControl(t, 0)
			wantModels := map[string]string{
				"Org/Strata-Loaded:Q4_K_M":   "loaded",
				"Org/Strata-Unloaded:Q5_K_M": "unloaded",
				"Org/Strata-Loading:Q6_K":    "loading",
				"Org/Strata-NoStatus:Q8_0":   "",
			}
			if !reflect.DeepEqual(control.Models, wantModels) || control.NextLoadStatus != 0 {
				t.Fatalf("management changed runtime model state: %+v", control)
			}
			for _, request := range control.Requests {
				if request.Method != http.MethodGet || (request.Path != "/health" && request.Path != "/v1/models") || request.Model != "" {
					t.Errorf("unsupported management reached runtime: %+v", request)
				}
			}
			if after := strataContractU3Artifacts(t, f, ""); !reflect.DeepEqual(after, before) {
				t.Fatalf("management changed private artifacts: before=%v after=%v", before, after)
			}
		})
	}

	testRunning("detected install succeeds as already installed", func(t *testing.T, f *strataContractFixture) {
		var status strataContractStatus
		f.Manager.strataContractOK(t, "engine:install", map[string]string{"engine": "strata"}, &status)
		want := strataContractStatus{Engine: "strata", Installed: true, Running: true, Healthy: true, Port: f.Port}
		if status != want {
			t.Fatalf("already-installed status = %+v, want %+v", status, want)
		}
		notification := f.Manager.strataContractNotification(t, "engine:install-progress", func(params json.RawMessage) bool {
			var progress struct {
				Engine string `json:"engine"`
				Stage  string `json:"stage"`
			}
			strataContractDecode(t, params, &progress)
			return progress.Engine == "strata" && progress.Stage == "already-installed"
		})
		var progress struct {
			Engine  string `json:"engine"`
			Stage   string `json:"stage"`
			Percent int    `json:"percent"`
		}
		strataContractDecode(t, notification.Params, &progress)
		if progress.Engine != "strata" || progress.Stage != "already-installed" || progress.Percent != 100 {
			t.Fatalf("already-installed progress = %+v", progress)
		}
	})
	testRunning("detected uninstall rejects missing definition before stopping", func(t *testing.T, f *strataContractFixture) {
		response := f.Manager.strataContractCall(t, "engine:uninstall", map[string]string{"engine": "strata"})
		assertError(t, response, `engine "strata" has no uninstall defined for this platform`)
	})
	testRunning("typed delete rejects missing model delete operation", func(t *testing.T, f *strataContractFixture) {
		response := f.Manager.strataContractCall(t, "engine:delete-model", map[string]string{"engine": "strata", "model": "Org/Strata-Loaded:Q4_K_M"})
		assertError(t, response, `engine "strata" does not support the model delete operation`)
	})
	testRunning("raw delete rejects missing delete action", func(t *testing.T, f *strataContractFixture) {
		response := f.Manager.strataContractCall(t, "engine:action", map[string]any{
			"engine": "strata", "action": "delete_model", "params": map[string]string{"model": "Org/Strata-Loaded:Q4_K_M"},
		})
		assertError(t, response, `engine "strata" has no action "delete_model"`)
	})
	testRunning("raw pull rejects missing pull action and reports terminal failure", func(t *testing.T, f *strataContractFixture) {
		// Local pull progress has no operation ID. This fresh manager issues
		// exactly one pull, and the retained reader preserves either frame order.
		response, notification := f.Manager.strataContractCallAndNotification(t, "engine:action", map[string]any{
			"engine": "strata", "action": "pull_model", "params": map[string]string{"model": "Org/Strata-Unloaded:Q5_K_M"},
		}, "engine:pull-progress", func(params json.RawMessage) bool {
			var progress struct {
				Engine string `json:"engine"`
				Op     string `json:"op"`
				Stage  string `json:"stage"`
			}
			strataContractDecode(t, params, &progress)
			return progress.Engine == "strata" && progress.Op == "pull" && progress.Stage == "error"
		})
		const diagnostic = `Strata experienced an error while downloading a model: engine "strata" has no action "pull_model"`
		assertError(t, response, diagnostic)
		var progress struct {
			Engine  string `json:"engine"`
			Op      string `json:"op"`
			Stage   string `json:"stage"`
			Percent int    `json:"percent"`
			Message string `json:"message"`
		}
		strataContractDecode(t, notification.Params, &progress)
		if progress.Engine != "strata" || progress.Op != "pull" || progress.Stage != "error" || progress.Percent != -1 || progress.Message != diagnostic {
			t.Fatalf("terminal pull progress = %+v, want Strata pull error, percent -1 and message %q", progress, diagnostic)
		}
	})
}

// Snapshot the whole private tree so an unexpected install/model-store path
// cannot evade a check that only knows conventional engine directories.
func strataContractU3Artifacts(t *testing.T, f *strataContractFixture, allowedIntent string) map[string]string {
	t.Helper()
	artifacts := make(map[string]string)
	err := filepath.WalkDir(f.Root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if path == allowedIntent || path == filepath.Join(f.AppDir, "engine-bin") {
			return nil
		}
		relative, err := filepath.Rel(f.Root, path)
		if err != nil {
			return err
		}
		if entry.IsDir() {
			artifacts[relative] = "directory"
			return nil
		}
		if !entry.Type().IsRegular() {
			return fmt.Errorf("unexpected private artifact type at %s: %s", relative, entry.Type())
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		artifacts[relative] = fmt.Sprintf("%x", sha256.Sum256(data))
		return nil
	})
	if err != nil {
		t.Fatalf("snapshot private install/model artifacts: %v", err)
	}
	if _, err := os.Stat(filepath.Join(f.AppDir, "engine-bin", "strata")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("private Strata install directory: %v, want not exist", err)
	}
	return artifacts
}
