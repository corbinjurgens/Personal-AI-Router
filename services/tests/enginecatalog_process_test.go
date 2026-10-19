// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"nvpair-shared/appdir"
	"nvpair-shared/testbin"
)

// A descriptor engine registered only by a user manifest (a llama.cpp fork
// under its own ID) is fronted end to end with no source registration: the
// broker reads engine-manager's descriptors, the proxy builds a facade from
// the descriptor and persists its port, and once engine-manager starts the
// engine, inference through that facade reaches it on 127.0.0.1.
func TestDescriptorEngineIsFrontedFromAUserManifest(t *testing.T) {
	const engineID = "llamacpp-custom"
	configDir := t.TempDir()

	fake := filepath.Join(t.TempDir(), testbin.Exe("fake-engine"))
	if err := goBuild(filepath.Join("..", "nvpair-engine-manager", "testdata", "fakeengine"), fake); err != nil {
		t.Fatalf("build fake engine: %v", err)
	}
	enginePort := freePort(t)
	manifest := map[string]any{
		"engine":           engineID,
		"display_name":     "llama.cpp (custom)",
		"manifest_version": 1,
		"family":           "llamacpp",
		"protocols":        []string{"openai-v1"},
		"model_naming":     "exact",
		"runtime": map[string]any{
			"mode":   "process",
			"bin":    fake,
			"env":    map[string]string{"FAKE_ADDR": "127.0.0.1:{port}"},
			"bind":   "127.0.0.1",
			"port":   enginePort,
			"ready":  map[string]any{"http": "http://127.0.0.1:{port}/v1/models", "status": 200, "timeout_s": 30},
			"health": map[string]any{"http": "http://127.0.0.1:{port}/v1/models", "status": 200, "interval_s": 5},
			"stop":   map[string]any{"signal": "term", "grace_s": 5},
		},
		"platforms": map[string]any{
			runtime.GOOS + "/" + runtime.GOARCH: map[string]any{"detect": []string{fake}},
		},
		"actions": map[string]any{
			"list_models": map[string]any{
				"http":   map[string]any{"method": "GET", "path": "/v1/models"},
				"result": map[string]any{"array": "data", "field": "id"},
			},
		},
	}
	// Resolve the per-user paths the broker's children will see.
	for _, key := range []string{"XDG_CONFIG_HOME", "HOME", "APPDATA", "LOCALAPPDATA", "USERPROFILE"} {
		t.Setenv(key, configDir)
	}
	enginesDir, err := appdir.Path("engines")
	if err != nil {
		t.Fatal(err)
	}
	portFile, err := appdir.Path(engineID + "-proxy-port.json")
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(enginesDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(enginesDir, engineID+".json"), body, 0o600); err != nil {
		t.Fatal(err)
	}

	stdin, msgs, stderr, cleanup := startBrokerWithConfigDir(t, configDir,
		"--engine-manager-path", engineMgrBin,
		"--proxy-path", proxyBin, "--proxy-engines", "llamacpp",
	)
	t.Cleanup(cleanup)
	go func() {
		for range stderr {
		}
	}()
	waitForMethod(t, msgs, "app:ready", 15*time.Second)

	facadePort := 0
	deadline := time.Now().Add(20 * time.Second)
	for facadePort == 0 {
		if data, err := os.ReadFile(portFile); err == nil {
			var stored struct {
				Port int `json:"port"`
			}
			if json.Unmarshal(data, &stored) == nil {
				facadePort = stored.Port
			}
		}
		if facadePort == 0 {
			if time.Now().After(deadline) {
				t.Fatalf("the descriptor engine's facade never persisted its port at %s", portFile)
			}
			time.Sleep(100 * time.Millisecond)
		}
	}

	callBrokerRPC(t, stdin, msgs, 9101, "engine:start", map[string]string{"engine": engineID})

	client := &http.Client{Timeout: 5 * time.Second}
	t.Cleanup(client.CloseIdleConnections)
	deadline = time.Now().Add(30 * time.Second)
	for {
		status, text := postChat(client, facadePort, "llama3.2:1b")
		if status == http.StatusOK && strings.Contains(text, "chat.completion") {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("inference through the descriptor engine facade: status %d body %q", status, text)
		}
		time.Sleep(250 * time.Millisecond)
	}
}

func postChat(client *http.Client, port int, model string) (int, string) {
	response, err := client.Post(fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port), "application/json",
		strings.NewReader(fmt.Sprintf(`{"model":%q,"messages":[]}`, model)))
	if err != nil {
		return 0, err.Error()
	}
	defer response.Body.Close()
	data, _ := io.ReadAll(response.Body)
	return response.StatusCode, string(data)
}
