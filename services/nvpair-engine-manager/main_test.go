// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"nvpair-shared/secretenv/secretenvtest"
	"nvpair-shared/testbin"
)

// Built once for the whole package: a pure-Go fake engine (stand-in for
// a real inference engine) and the engine-manager binary with a test-only
// registry overlay. The overlay leaves main's transport/lifecycle intact.
var (
	fakeEngineBin string
	managerBin    string
)

func TestMain(m *testing.M) {
	if runModelLoadChild() {
		return
	}
	secretenvtest.RunChild()
	if len(os.Args) >= 3 && os.Args[1] == "notification-transport-helper" {
		os.Exit(runNotificationTransportHelper(os.Args[2:]))
	}
	if len(os.Args) == 3 && os.Args[1] == "command-test-helper" {
		os.Exit(runCommandTestHelper(os.Args[2]))
	}
	// fake-engine gives FAKE_ADDR precedence over OLLAMA_HOST. Ambient values
	// must not redirect a fixture child to an installed engine's address.
	for _, key := range []string{"FAKE_ADDR", "OLLAMA_HOST", "NVPAIR_EM_LIVE_TEST"} {
		if err := os.Unsetenv(key); err != nil {
			panic(err)
		}
	}
	tmp, cleanup, err := testbin.MainDir("nvpair-engine-manager", "main", "nvpair-em-test-*", "fake-engine", "nvpair-engine-manager")
	if err != nil {
		panic(err)
	}
	fakeEngineBin = filepath.Join(tmp, testbin.Exe("fake-engine"))
	if out, err := exec.Command("go", "build", "-o", fakeEngineBin, "./testdata/fakeengine").CombinedOutput(); err != nil {
		panic("build fake-engine: " + err.Error() + "\n" + string(out))
	}

	managerBin = filepath.Join(tmp, testbin.Exe("nvpair-engine-manager"))
	fixtureDir, err := os.MkdirTemp("", "nvpair-em-registry-*")
	if err != nil {
		panic(err)
	}
	overlay, err := managerFixtureOverlay(fixtureDir)
	if err != nil {
		panic(err)
	}
	if out, err := exec.Command("go", "build", "-overlay", overlay, "-o", managerBin, ".").CombinedOutput(); err != nil {
		panic("build engine-manager: " + err.Error() + "\n" + string(out))
	}

	code := m.Run()
	if err := os.RemoveAll(fixtureDir); err != nil {
		panic(err)
	}
	cleanup()
	os.Exit(code)
}

// managerFixtureOverlay changes exactly the registry construction site. It
// compiles the actual main, owned transport, signals, listeners, Manager.Run,
// watcher and StopAll, with fresh inert bundled raw JSON and normal override
// merging. No production source or runtime configuration is added.
func managerFixtureOverlay(dir string) (string, error) {
	bundle := filepath.Join(dir, "bundled")
	if err := os.Mkdir(bundle, 0o700); err != nil {
		return "", err
	}
	ports, err := privateFixturePorts(3)
	if err != nil {
		return "", err
	}
	for i, engine := range []string{"ollama", "lmstudio", "llamacpp"} {
		data, err := json.Marshal(portFixtureManifest(engine, ports[i]))
		if err != nil {
			return "", err
		}
		if err := os.WriteFile(filepath.Join(bundle, engine+".json"), data, 0o600); err != nil {
			return "", err
		}
	}
	source, err := filepath.Abs("main.go")
	if err != nil {
		return "", err
	}
	data, err := os.ReadFile(source)
	if err != nil {
		return "", err
	}
	const site = "reg := buildRegistry(manifestDir)"
	if strings.Count(string(data), site) != 1 {
		return "", fmt.Errorf("fixture overlay requires exactly one %q in main.go", site)
	}
	// Only the explicitly live-tagged startManager opts into the original
	// registry. Ordinary launches default to fixtures even without an env flag.
	replacement := fmt.Sprintf(`reg := NewRegistry()
	if os.Getenv("NVPAIR_EM_LIVE_TEST") == "1" {
		reg = buildRegistry(manifestDir)
	} else {
		if err := reg.LoadFS(os.DirFS(%q), "."); err != nil {
			log.Fatalf("load test-only registry: %%v", err)
		}
		if err := reg.LoadOverrideDir(manifestDir); err != nil {
			log.Fatalf("load test-only overrides: %%v", err)
		}
	}`, bundle)
	modified := filepath.Join(dir, "main.go")
	if err := os.WriteFile(modified, []byte(strings.Replace(string(data), site, replacement, 1)), 0o600); err != nil {
		return "", err
	}
	overlay, err := json.Marshal(map[string]map[string]string{"Replace": {source: modified}})
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, "overlay.json")
	return path, os.WriteFile(path, overlay, 0o600)
}
