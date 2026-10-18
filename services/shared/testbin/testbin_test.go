// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package testbin

import (
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"testing/fstest"
)

func TestDirTemporaryCleanup(t *testing.T) {
	t.Setenv(EnvRoot, "")
	var dir string
	t.Run("temporary directory exists", func(t *testing.T) {
		dir = Dir(t, "testbin", "helper", "fixture")
		info, err := os.Stat(dir)
		if err != nil || !info.IsDir() {
			t.Fatalf("temporary directory: info=%v err=%v", info, err)
		}
	})
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("directory survived test cleanup: %v", err)
	}
}

func TestDirFixedPathPersists(t *testing.T) {
	root := t.TempDir()
	t.Setenv(EnvRoot, root)
	want := filepath.Join(root, "testbin", "helper")
	t.Run("fixed directory exists", func(t *testing.T) {
		if got := Dir(t, "testbin", "helper", "fixture"); got != want {
			t.Fatalf("Dir = %q, want %q", got, want)
		}
	})
	info, err := os.Stat(want)
	if err != nil || !info.IsDir() {
		t.Fatalf("fixed directory must survive cleanup: info=%v err=%v", info, err)
	}
}

func TestMainDirTemporaryCleanup(t *testing.T) {
	t.Setenv(EnvRoot, "")
	dir, cleanup, err := MainDir("testbin", "helper", "testbin-unit-*", "fixture")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(cleanup)
	if !strings.HasPrefix(filepath.Base(dir), "testbin-unit-") {
		t.Fatalf("directory %q does not use the requested pattern", dir)
	}
	if info, err := os.Stat(dir); err != nil || !info.IsDir() {
		t.Fatalf("temporary directory: info=%v err=%v", info, err)
	}
	cleanup()
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("directory survived cleanup: %v", err)
	}
}

func TestMainDirFixedPathPersists(t *testing.T) {
	root := t.TempDir()
	t.Setenv(EnvRoot, root)
	dir, cleanup, err := MainDir("testbin", "helper", "unused-*", "fixture")
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(root, "testbin", "helper")
	if dir != want {
		t.Fatalf("MainDir = %q, want %q", dir, want)
	}
	cleanup()
	if info, err := os.Stat(want); err != nil || !info.IsDir() {
		t.Fatalf("fixed directory must survive cleanup: info=%v err=%v", info, err)
	}
}

func TestMainDirRejectsInvalidRequests(t *testing.T) {
	test := func(name, root, module, role, want string, programs ...string) {
		t.Run(name, func(t *testing.T) {
			t.Setenv(EnvRoot, root)
			dir, cleanup, err := MainDir(module, role, "unused-*", programs...)
			if cleanup != nil {
				cleanup()
			}
			if err == nil || !strings.Contains(err.Error(), want) || dir != "" {
				t.Fatalf("MainDir = %q, error %v; want error containing %q and no directory", dir, err, want)
			}
		})
	}
	test("relative root", "relative", "testbin", "helper", "must be absolute", "fixture")
	test("unlisted program in temporary mode", "", "testbin", "helper", "testbin/helper/missing", "missing")
	test("unlisted program in fixed mode", t.TempDir(), "testbin", "helper", "testbin/helper/missing", "fixture", "missing")
	test("unlisted role", "", "testbin", "missing", "testbin/missing/fixture", "fixture")
	test("no programs", "", "testbin", "helper", "at least one program")
}

func TestDirRejectsInvalidRequests(t *testing.T) {
	// A child test process lets us assert Fatalf without failing the parent test.
	if scenario := os.Getenv("NVPAIR_TESTBIN_REJECTION"); scenario != "" {
		programs := []string{"fixture"}
		switch scenario {
		case "unlisted":
			programs = []string{"missing"}
		case "empty":
			programs = nil
		}
		Dir(t, "testbin", "helper", programs...)
		return
	}
	test := func(name, root, scenario, want string) {
		t.Run(name, func(t *testing.T) {
			t.Setenv(EnvRoot, root)
			t.Setenv("NVPAIR_TESTBIN_REJECTION", scenario)
			executable, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command(executable, "-test.run=^TestDirRejectsInvalidRequests$", "-test.timeout=30s")
			output, err := cmd.CombinedOutput()
			if err == nil || !strings.Contains(string(output), want) {
				t.Fatalf("child error=%v output=%s; want failure containing %q", err, output, want)
			}
		})
	}
	test("relative root", "relative", "relative", "must be absolute")
	test("unlisted temporary program", "", "unlisted", "testbin/helper/missing")
	test("unlisted fixed program", t.TempDir(), "unlisted", "testbin/helper/missing")
	test("no programs", "", "empty", "at least one program")
}

func TestProgramsParsesAndSorts(t *testing.T) {
	files := fstest.MapFS{
		"programs/z.txt": &fstest.MapFile{Data: []byte("# comment\n\nz/role/second\n z/role/first \n")},
		"programs/a.txt": &fstest.MapFile{Data: []byte("a/role/program\n")},
	}
	got, err := parsePrograms(files)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"a/role/program", "z/role/first", "z/role/second"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("programs = %v, want %v", got, want)
	}
}

func TestProgramsRejectsInvalidManifest(t *testing.T) {
	test := func(name, data, want string) {
		t.Run(name, func(t *testing.T) {
			_, err := parsePrograms(fstest.MapFS{"programs/testbin.txt": &fstest.MapFile{Data: []byte(data)}})
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("error = %v, want diagnostic containing %q", err, want)
			}
		})
	}
	test("duplicate", "testbin/helper/fixture\ntestbin/helper/fixture\n", "duplicate program")
	test("missing role", "testbin/fixture", "invalid program")
	test("path traversal", "testbin/../fixture", "invalid program")
	test("executable suffix", "testbin/helper/fixture.exe", "without .exe")
	test("wrong owner", "another/helper/fixture", "must own its manifest file")
}

func TestProgramsIncludesFixture(t *testing.T) {
	programs, err := Programs()
	if err != nil {
		t.Fatal(err)
	}
	for _, program := range programs {
		if program == "testbin/helper/fixture" {
			return
		}
	}
	t.Fatalf("fixture missing from Programs: %v", programs)
}

func TestExePlatformSuffix(t *testing.T) {
	want := "fixture"
	if runtime.GOOS == "windows" {
		want = "fixture.exe"
	}
	if got := Exe("fixture"); got != want {
		t.Fatalf("Exe = %q, want %q", got, want)
	}
}
