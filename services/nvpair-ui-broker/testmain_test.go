// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"nvpair-shared/secretenv/secretenvtest"
	"nvpair-shared/testbin"
)

var portTestGoEnv []string

// Bare broker fixtures also consult the settings journal during startup. Keep
// all default app-data lookups away from the developer's live configuration.
func TestMain(m *testing.M) {
	secretenvtest.RunChild()
	if len(os.Args) == 2 && os.Args[1] == shutdownIgnoringChildArg {
		runShutdownIgnoringChild()
		return
	}
	output, err := exec.Command("go", "env", "-json", "GOCACHE", "GOMODCACHE", "GOPATH").Output()
	if err != nil {
		fmt.Fprintln(os.Stderr, "capture Go build environment:", err)
		os.Exit(1)
	}
	var goEnv map[string]string
	if err := json.Unmarshal(output, &goEnv); err != nil {
		fmt.Fprintln(os.Stderr, "decode Go build environment:", err)
		os.Exit(1)
	}
	for _, key := range []string{"GOCACHE", "GOMODCACHE", "GOPATH"} {
		portTestGoEnv = append(portTestGoEnv, key+"="+goEnv[key])
	}
	binDir, cleanupBins, err := testbin.MainDir("nvpair-ui-broker", "ports-spawn", "nvpair-broker-ports-*", "scanner")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	portScannerBinary = filepath.Join(binDir, testbin.Exe("scanner"))
	dir, err := os.MkdirTemp("", "nvpair-broker-test-")
	if err != nil {
		cleanupBins()
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	portScannerSource = filepath.Join(dir, "scanner.go")
	for _, key := range []string{"LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME", "HOME", "USERPROFILE"} {
		if err := os.Setenv(key, dir); err != nil {
			fmt.Fprintln(os.Stderr, err)
			_ = os.RemoveAll(dir)
			cleanupBins()
			os.Exit(1)
		}
	}
	code := m.Run()
	_ = os.RemoveAll(dir)
	cleanupBins()
	os.Exit(code)
}
