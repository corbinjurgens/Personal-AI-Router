// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"nvpair-shared/testbin"
)

// Built once for the whole package: a pure-Go fake engine (stand-in for
// a real inference engine) and the engine-manager binary itself (for
// the cross-process e2e test).
var (
	fakeEngineBin string
	managerBin    string
)

func TestMain(m *testing.M) {
	if len(os.Args) == 3 && os.Args[1] == "command-test-helper" {
		os.Exit(runCommandTestHelper(os.Args[2]))
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
	if out, err := exec.Command("go", "build", "-o", managerBin, ".").CombinedOutput(); err != nil {
		panic("build engine-manager: " + err.Error() + "\n" + string(out))
	}

	code := m.Run()
	cleanup()
	os.Exit(code)
}
