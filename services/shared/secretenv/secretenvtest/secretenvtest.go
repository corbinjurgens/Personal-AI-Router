// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package secretenvtest is a sentinel child for launch-boundary tests. A test
// calls Prepare to put sentinel provider secrets in its own environment, starts
// its own test binary through the launch site under test, and reads back with
// Wait which reserved-namespace names the child saw. The child reports names
// only; sentinel values never leave it.
//
// Only _test.go files import it, so it is never linked into a shipped binary
// (the same arrangement as net/http/httptest).
package secretenvtest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/secretenv"
	"nvpair-shared/testbin"
)

const (
	// ReportEnv names the file a sentinel child writes its report to. Its
	// presence is what makes a test binary act as the child.
	ReportEnv = "NVPAIR_SECRETENV_TEST_REPORT"
	// ExitEnv makes the child exit with this status after reporting, so a test
	// can drive a launch site's failure path.
	ExitEnv = "NVPAIR_SECRETENV_TEST_EXIT"

	// Name is a canonical reserved credential reference.
	Name = secretenv.Prefix + "SENTINEL"
	// LowerName is a lowercase reserved name. It differs from Name beyond case
	// so both exist side by side on Windows, where names are case-insensitive.
	LowerName = "nvpair_provider_secret_sentinel_lower"
	// Value is the sentinel secret. Tests assert it appears in no argument,
	// error, RPC payload or log line.
	Value = "secretenvtest-sentinel-value-9c41e7"
)

// Report is what a sentinel child saw.
type Report struct {
	// Reserved lists the reserved-namespace variable names in the child's
	// environment, sorted.
	Reserved []string `json:"reserved"`
	// ArgsCarryValue reports whether any child argument contains Value.
	ArgsCarryValue bool `json:"argsCarryValue"`
}

// RunChild reports and exits when this process was started as a sentinel
// child, and returns otherwise. Call it first in TestMain.
func RunChild() {
	path := os.Getenv(ReportEnv)
	if path == "" {
		return
	}
	report := Report{Reserved: []string{}}
	for _, kv := range os.Environ() {
		name, _, _ := strings.Cut(kv, "=")
		if strings.HasPrefix(strings.ToUpper(name), secretenv.Prefix) {
			report.Reserved = append(report.Reserved, name)
		}
	}
	slices.Sort(report.Reserved)
	for _, arg := range os.Args {
		if strings.Contains(arg, Value) {
			report.ArgsCarryValue = true
		}
	}
	code := 0
	if v := os.Getenv(ExitEnv); v != "" {
		code, _ = strconv.Atoi(v)
	}
	data, err := json.Marshal(report)
	if err != nil {
		os.Exit(90)
	}
	// Written whole and renamed into place, so a reader polling for a detached
	// child never sees a partial report.
	tmp := path + ".tmp"
	if os.WriteFile(tmp, data, 0o600) != nil || os.Rename(tmp, path) != nil {
		os.Exit(91)
	}
	if code != 0 {
		fmt.Fprintln(os.Stderr, "sentinel child failing as requested")
	}
	os.Exit(code)
}

// Prepare sets the sentinel secrets and a fresh report path in this test's
// environment, so a child that inherits it reports there, and returns the
// path. Like t.Setenv, it cannot be used in parallel tests.
func Prepare(t testing.TB) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "report.json")
	t.Setenv(ReportEnv, path)
	t.Setenv(Name, Value)
	t.Setenv(LowerName, Value)
	return path
}

// OnPath makes PATH hold only the testbin directory for module and role, with
// a copy of this test binary in it named name, so a launch site that runs a
// system tool by name starts the sentinel child instead. The program must be
// listed in the testbin manifest. It returns the copy's path.
func OnPath(t testing.TB, module, role, name string) string {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(exe)
	if err != nil {
		t.Fatal(err)
	}
	dir := testbin.Dir(t, module, role, name)
	path := filepath.Join(dir, testbin.Exe(name))
	if err := os.WriteFile(path, data, 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)
	return path
}

// Wait returns the report at path, waiting for a child that has not written it
// yet. It fails the test after ten seconds.
func Wait(t testing.TB, path string) Report {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		data, err := os.ReadFile(path)
		if err == nil {
			var report Report
			if err := json.Unmarshal(data, &report); err != nil {
				t.Fatalf("sentinel report: %v", err)
			}
			if report.ArgsCarryValue {
				t.Errorf("the sentinel secret reached the child's arguments")
			}
			return report
		}
		if !os.IsNotExist(err) {
			t.Fatalf("sentinel report: %v", err)
		}
		if time.Now().After(deadline) {
			t.Fatalf("no sentinel child reported to %s", path)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// AssertRemoved fails the test unless the child saw no reserved name.
func AssertRemoved(t testing.TB, report Report) {
	t.Helper()
	if len(report.Reserved) != 0 {
		t.Errorf("child received reserved provider-secret variables %q, want none", report.Reserved)
	}
}

// AssertRetained fails the test unless the child saw both sentinel names.
func AssertRetained(t testing.TB, report Report) {
	t.Helper()
	for _, name := range []string{Name, LowerName} {
		if !slices.Contains(report.Reserved, name) {
			t.Errorf("child did not receive %s; it saw %q", name, report.Reserved)
		}
	}
}

// AssertNoValue fails the test if text, such as an error, log or RPC payload,
// contains the sentinel secret.
func AssertNoValue(t testing.TB, what, text string) {
	t.Helper()
	if strings.Contains(text, Value) {
		t.Errorf("%s contains the sentinel secret", what)
	}
}
