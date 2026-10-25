// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
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

// writePrivateBrokerEngineSettings isolates the broker's health route as well
// as the selected facades. Manager manifest overrides alone do not replace the
// broker's default backend. Configure only the caller's fixture engines, before
// startup; retain intentional journal entries and reuse saved private facades.
func writePrivateBrokerEngineSettings(t *testing.T, configDir, clusterDir string, facadePorts map[string]int) {
	t.Helper()
	path := filepath.Join(filepath.Dir(clusterDir), "engine-settings-operations.json")
	journal := make(map[string]json.RawMessage)
	data, err := os.ReadFile(path)
	if err == nil {
		if err := json.Unmarshal(data, &journal); err != nil || journal == nil {
			t.Fatalf("decode fixture engine settings journal: %v", err)
		}
	} else if !os.IsNotExist(err) {
		t.Fatal(err)
	}
	var unavailable *httptest.Server
	for name, facadePort := range facadePorts {
		if _, exists := journal[name]; exists {
			continue
		}
		engine, ok := engines.ByName(name)
		if !ok {
			t.Fatalf("unknown fixture engine %q", name)
		}
		portPath := filepath.Join(jointAppDir(configDir), engine.PortFile)
		portData, portErr := os.ReadFile(portPath)
		if portErr == nil {
			var saved struct {
				Port int `json:"port"`
			}
			if err := json.Unmarshal(portData, &saved); err != nil {
				t.Fatalf("decode fixture facade port: %v", err)
			}
			facadePort = saved.Port
		} else if !os.IsNotExist(portErr) {
			t.Fatal(portErr)
		}
		if err := validateFixturePort(facadePort); err != nil {
			t.Fatal(err)
		}
		if unavailable == nil {
			unavailable = newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusServiceUnavailable)
			}))
			// Registered before process cleanup, so this endpoint remains owned
			// through broker/proxy shutdown and any fixture restarts.
			t.Cleanup(unavailable.Close)
		}
		journal[name] = mustMarshal(t, struct {
			Explicit bool              `json:"explicit"`
			Snapshot settings.Snapshot `json:"snapshot"`
		}{Explicit: true, Snapshot: settings.Snapshot{
			Engine: name, Revision: 1, AppliedRevision: 1, Phase: "idle", Format: "pair-arguments-v1",
			Settings: settings.Config{ServerPort: portOfURL(t, unavailable.URL), ProxyPort: facadePort},
		}})
		if os.IsNotExist(portErr) {
			jointWriteJSON(t, portPath, struct {
				Port int `json:"port"`
			}{Port: facadePort})
		}
	}
	if unavailable != nil {
		jointWriteJSON(t, path, journal)
	}
}

func TestPrivateBrokerEngineSettings(t *testing.T) {
	t.Run("selected facades use an owned unavailable backend", func(t *testing.T) {
		configDir, clusterDir := t.TempDir(), filepath.Join(t.TempDir(), "cluster")
		facades := map[string]int{"ollama": freePort(t), "llamacpp": freePort(t)}
		writePrivateBrokerEngineSettings(t, configDir, clusterDir, facades)
		data, err := os.ReadFile(filepath.Join(filepath.Dir(clusterDir), "engine-settings-operations.json"))
		if err != nil {
			t.Fatal(err)
		}
		var journal map[string]struct {
			Explicit bool              `json:"explicit"`
			Snapshot settings.Snapshot `json:"snapshot"`
		}
		if err := json.Unmarshal(data, &journal); err != nil {
			t.Fatal(err)
		}
		if len(journal) != len(facades) {
			t.Fatalf("journal entries = %d, want %d selected engines", len(journal), len(facades))
		}
		client := &http.Client{Timeout: time.Second}
		t.Cleanup(client.CloseIdleConnections)
		for name, port := range facades {
			record := journal[name]
			snapshot := record.Snapshot
			if !record.Explicit || snapshot.Engine != name || snapshot.Revision != 1 || snapshot.AppliedRevision != 1 || snapshot.Phase != "idle" || snapshot.Format != "pair-arguments-v1" || snapshot.Settings.ProxyPort != port {
				t.Fatalf("%s settings = %+v, explicit=%t", name, snapshot, record.Explicit)
			}
			if err := validateFixturePort(snapshot.Settings.ServerPort); err != nil {
				t.Fatal(err)
			}
			engine, ok := engines.ByName(name)
			if !ok {
				t.Fatalf("unknown fixture engine %q", name)
			}
			data, err := os.ReadFile(filepath.Join(jointAppDir(configDir), engine.PortFile))
			if err != nil {
				t.Fatal(err)
			}
			var saved struct {
				Port int `json:"port"`
			}
			if err := json.Unmarshal(data, &saved); err != nil {
				t.Fatal(err)
			}
			if saved.Port != port || snapshot.Settings.ServerPort == port {
				t.Fatalf("%s ports: saved=%d backend=%d facade=%d", name, saved.Port, snapshot.Settings.ServerPort, port)
			}
			resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/health", snapshot.Settings.ServerPort))
			if err != nil {
				t.Fatalf("probe private %s backend: %v", name, err)
			}
			if err := resp.Body.Close(); err != nil {
				t.Fatal(err)
			}
			if resp.StatusCode != http.StatusServiceUnavailable {
				t.Fatalf("private %s health = %d, want unavailable", name, resp.StatusCode)
			}
		}
	})

	t.Run("preserves a saved private facade when adding health isolation", func(t *testing.T) {
		configDir, clusterDir := t.TempDir(), filepath.Join(t.TempDir(), "cluster")
		savedPort := freePort(t)
		portPath := filepath.Join(jointAppDir(configDir), "llamacpp-proxy-port.json")
		jointWriteJSON(t, portPath, map[string]int{"port": savedPort})
		before, err := os.ReadFile(portPath)
		if err != nil {
			t.Fatal(err)
		}
		writePrivateBrokerEngineSettings(t, configDir, clusterDir, map[string]int{"llamacpp": freePort(t)})
		data, err := os.ReadFile(filepath.Join(filepath.Dir(clusterDir), "engine-settings-operations.json"))
		if err != nil {
			t.Fatal(err)
		}
		var journal map[string]struct {
			Explicit bool              `json:"explicit"`
			Snapshot settings.Snapshot `json:"snapshot"`
		}
		if err := json.Unmarshal(data, &journal); err != nil {
			t.Fatal(err)
		}
		record := journal["llamacpp"]
		if !record.Explicit || record.Snapshot.Settings.ProxyPort != savedPort {
			t.Fatalf("saved facade was not retained in explicit settings: %+v", record)
		}
		if err := validateFixturePort(record.Snapshot.Settings.ServerPort); err != nil {
			t.Fatalf("saved facade has no private health target: %v", err)
		}
		after, err := os.ReadFile(portPath)
		if err != nil || !bytes.Equal(after, before) {
			t.Fatalf("saved facade file changed: %q, %v", after, err)
		}
	})

	t.Run("preserves intentional settings and repeated setup", func(t *testing.T) {
		configDir, clusterDir := t.TempDir(), filepath.Join(t.TempDir(), "cluster")
		path := filepath.Join(filepath.Dir(clusterDir), "engine-settings-operations.json")
		intentional := json.RawMessage(`{"explicit":true,"snapshot":{"engine":"llamacpp","revision":7,"appliedRevision":6,"phase":"failed","format":"pair-arguments-v1","settings":{"serverPort":25001,"proxyPort":25002,"launchText":"--private"}},"receipts":{"private-operation":{"hash":"fixture","revision":7,"phase":"failed"}}}`)
		jointWriteJSON(t, path, map[string]json.RawMessage{"llamacpp": intentional})
		portPath := filepath.Join(jointAppDir(configDir), "llamacpp-proxy-port.json")
		portData := []byte(`{"port":25002}`)
		if err := os.MkdirAll(filepath.Dir(portPath), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(portPath, portData, 0o600); err != nil {
			t.Fatal(err)
		}
		facades := map[string]int{"llamacpp": freePort(t), "ollama": freePort(t)}
		writePrivateBrokerEngineSettings(t, configDir, clusterDir, facades)
		before, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var journal map[string]json.RawMessage
		if err := json.Unmarshal(before, &journal); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(journal["llamacpp"], intentional) || len(journal["ollama"]) == 0 {
			t.Fatalf("private journal changed or missing new engine: %s", before)
		}
		actual, err := os.ReadFile(portPath)
		if err != nil || !bytes.Equal(actual, portData) {
			t.Fatalf("private facade changed: %q, %v", actual, err)
		}
		writePrivateBrokerEngineSettings(t, configDir, clusterDir, facades)
		after, err := os.ReadFile(path)
		if err != nil || !bytes.Equal(after, before) {
			t.Fatalf("repeated setup changed journal: %q, %v", after, err)
		}
	})
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
