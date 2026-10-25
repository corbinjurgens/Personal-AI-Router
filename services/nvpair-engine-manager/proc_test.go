// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"nvpair-shared/testbin"
)

func TestIsOurEngineImage(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "ollama.exe")
	if err := os.WriteFile(bin, []byte("image fixture"), 0o600); err != nil {
		t.Fatal(err)
	}
	upper := filepath.Join(filepath.Dir(bin), "OLLAMA.EXE")
	caseAlias := true
	if runtime.GOOS == "darwin" {
		_, err := os.Stat(upper)
		caseAlias = err == nil
	}
	cases := []struct {
		name  string
		image string
		want  bool
	}{
		{"same image", bin, true},
		{"case alias", upper, caseAlias},
		{"platform deleted annotation", bin + ` (deleted)`, runtime.GOOS == "linux"},
		{"different image", filepath.Join(filepath.Dir(bin), "other.exe"), false},
		{"empty image", ``, false},
		{"malformed image", bin + "\x00", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isOurEngineImage(tc.image, bin); got != tc.want {
				t.Errorf("isOurEngineImage(%q, %q) = %v, want %v", tc.image, bin, got, tc.want)
			}
		})
	}
	if isOurEngineImage(bin, ``) {
		t.Fatal("empty binPath must not match")
	}
}

func TestIsOurEngineImageLiteralDeletedSuffix(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "engine")
	suffixed := bin + " (deleted)"
	for _, path := range []string{bin, suffixed} {
		if err := os.WriteFile(path, []byte(path), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if isOurEngineImage(suffixed, bin) || isOurEngineImage(bin, suffixed) {
		t.Fatal("distinct existing literal filenames must not have the same executable identity")
	}
	if !isOurEngineImage(suffixed, suffixed) {
		t.Fatal("a managed literal suffixed executable must match its own full identity")
	}
	for _, platform := range []string{"darwin", "windows", "linux"} {
		t.Run(platform+" preserves existing literal suffix", func(t *testing.T) {
			if got := normalizeEngineImage(suffixed, platform); got != suffixed {
				t.Fatalf("inspected literal image=%q, want %q", got, suffixed)
			}
		})
	}
}

func TestIsOurEngineImageLinuxDeletedAnnotation(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "engine")
	if err := os.WriteFile(bin, []byte("replacement managed file"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := normalizeEngineImage(bin+" (deleted)", "linux"); got != bin {
		t.Fatalf("Linux deleted metadata image=%q, want %q", got, bin)
	}
	literal := bin + " (deleted)"
	if got := normalizeEngineImage(literal+" (deleted)", "linux"); got != literal {
		t.Fatalf("Linux deleted literal-suffix image=%q, want %q", got, literal)
	}
	for _, platform := range []string{"darwin", "windows"} {
		if got := normalizeEngineImage(bin+" (deleted)", platform); got != bin+" (deleted)" {
			t.Fatalf("%s literal path=%q, want full suffixed filename", platform, got)
		}
	}
}

func TestProcImageLinuxLiteralMetadataCollision(t *testing.T) {
	dir := t.TempDir()
	actual := filepath.Join(dir, "actual executable")
	literal := filepath.Join(dir, "engine (deleted)")
	for _, path := range []string{actual, literal} {
		if err := os.WriteFile(path, []byte(path), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	t.Run("distinct existing file is refused", func(t *testing.T) {
		if got := linuxEngineImage(literal, actual); got != "" {
			t.Fatalf("colliding Linux image=%q, want unknown image", got)
		}
	})
	t.Run("actual literal executable is preserved", func(t *testing.T) {
		if got := linuxEngineImage(literal, literal); got != literal {
			t.Fatalf("Linux literal image=%q, want %q", got, literal)
		}
	})
	t.Run("unavailable actual executable is refused", func(t *testing.T) {
		if got := linuxEngineImage(literal, filepath.Join(dir, "missing")); got != "" {
			t.Fatalf("unavailable actual image=%q, want unknown image", got)
		}
	})
	t.Run("genuine deleted annotation remains recoverable", func(t *testing.T) {
		annotated := actual + " (deleted)"
		got := linuxEngineImage(annotated, actual)
		if got != annotated || normalizeEngineImage(got, "linux") != actual {
			t.Fatalf("Linux deleted image=%q, want annotation retained then managed path recovered", got)
		}
	})
}

func TestProcImageLinuxDanglingLiteralSymlink(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix symlink-entry inspection fixture")
	}
	dir := t.TempDir()
	actual := filepath.Join(dir, "actual executable")
	literal := filepath.Join(dir, "engine (deleted)")
	if err := os.WriteFile(actual, []byte("actual executable"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(dir, "unavailable target"), literal); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(literal); err != nil {
		t.Fatalf("literal symlink entry must exist: %v", err)
	}
	if _, err := os.Stat(literal); !os.IsNotExist(err) {
		t.Fatalf("literal symlink target must be unavailable, got %v", err)
	}
	if got := linuxEngineImage(literal, actual); got != "" {
		t.Fatalf("dangling literal entry image=%q, want unknown image", got)
	}
	if got := normalizeEngineImage(literal, "linux"); got != literal {
		t.Fatalf("existing literal entry normalized to %q, want %q", got, literal)
	}
	if _, err := os.Lstat(literal); err != nil {
		t.Fatalf("inspection must retain the literal link: %v", err)
	}
}

func TestProcImageCopiedListenerWithSpaces(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "engine with spaces"+exeExt())
	copyFile(t, fakeEngineBin, bin)
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	spawnFakeListener(t, bin, port)
}

func TestIsOurEngineImageCanonicalAlias(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Darwin canonical executable aliases")
	}
	bin := filepath.Join(t.TempDir(), "managed engine")
	if err := os.WriteFile(bin, []byte("image fixture"), 0o600); err != nil {
		t.Fatal(err)
	}
	canonical, err := filepath.EvalSymlinks(bin)
	if err != nil {
		t.Fatal(err)
	}
	if !isOurEngineImage(canonical, bin) {
		t.Fatalf("native canonical alias %q must match %q", canonical, bin)
	}
	missing := filepath.Join(filepath.Dir(bin), "missing")
	if isOurEngineImage(missing, missing) {
		t.Fatal("unavailable canonical image must fail closed")
	}
	if !strings.HasPrefix(canonical, "/private/var/") || !strings.HasPrefix(bin, "/var/") {
		t.Logf("native temp aliases: lexical=%q canonical=%q", bin, canonical)
	}
}

func TestIsManagedInstallPath(t *testing.T) {
	base := t.TempDir()
	inside := filepath.Join(base, "ollama", "bin", "ollama"+exeExt())
	outside := filepath.Join(t.TempDir(), "ollama"+exeExt())
	if err := os.MkdirAll(filepath.Dir(inside), 0o700); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{inside, outside} {
		if err := os.WriteFile(path, []byte("image fixture"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if !isManagedInstallPath(inside, filepath.Join(base, "ollama")) {
		t.Fatalf("managed child path %q was rejected", inside)
	}
	if isManagedInstallPath(outside, filepath.Join(base, "ollama")) {
		t.Fatalf("external path %q was accepted", outside)
	}
	if isManagedInstallPath("", filepath.Join(base, "ollama")) {
		t.Fatal("empty binary path must not be managed")
	}
}

func TestIsManagedInstallPathLiteralDeletedSuffix(t *testing.T) {
	installDir := filepath.Join(t.TempDir(), "managed")
	if err := os.Mkdir(installDir, 0o755); err != nil {
		t.Fatal(err)
	}
	literalSibling := installDir + " (deleted)"
	if err := os.WriteFile(literalSibling, []byte("external sibling"), 0o600); err != nil {
		t.Fatal(err)
	}
	if isManagedInstallPath(literalSibling, installDir) {
		t.Fatal("a configured literal-suffix sibling must not be normalized inside the managed directory")
	}
}

func TestPIDOnPortRefusesAmbiguousOrMalformedInspection(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix port inspection command fixtures")
	}
	cases := []struct{ name, output string }{
		{"empty", ""},
		{"malformed", "123oops"},
		{"ambiguous", "123\n456"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "lsof"), []byte("#!/bin/sh\nprintf '%s\\n' '"+tc.output+"'\n"), 0o700); err != nil {
				t.Fatal(err)
			}
			t.Setenv("PATH", dir)
			if pid, image, ok := pidOnPort(12345); ok || pid != 0 || image != "" {
				t.Fatalf("inspection = (%d, %q, %v), want no owner", pid, image, ok)
			}
		})
	}
}

func TestPIDOnPortRefusesInvalidFirstProviderDespiteValidFallback(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix port inspection command fixtures")
	}
	pidText := strconv.Itoa(os.Getpid())
	cases := []struct{ name, output, exit string }{
		{"distinct owners", pidText + "\n" + strconv.Itoa(os.Getpid()+1), "0"},
		{"valid then malformed owner", pidText + "\ninvalid-owner", "0"},
		{"malformed then valid owner", "invalid-owner\n" + pidText, "0"},
		{"zero owner", "0", "0"},
		{"negative owner", "-1", "0"},
		{"overflow owner", "999999999999999999999999", "0"},
		{"malformed failed query", pidText + "\ninvalid-owner", "1"},
		{"owner from failed query", pidText, "1"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			lsof := "#!/bin/sh\nprintf '%s\\n' '" + tc.output + "'\nexit " + tc.exit + "\n"
			if err := os.WriteFile(filepath.Join(dir, "lsof"), []byte(lsof), 0o700); err != nil {
				t.Fatal(err)
			}
			ss := "#!/bin/sh\nprintf '%s\\n' 'LISTEN users:((\"controlled-test-process\",pid=" + pidText + ",fd=1))'\n"
			if err := os.WriteFile(filepath.Join(dir, "ss"), []byte(ss), 0o700); err != nil {
				t.Fatal(err)
			}
			t.Setenv("PATH", dir)
			pid, image, ok := pidOnPort(12345)
			if pid != 0 || image != "" || ok {
				t.Fatalf("invalid first provider with readable fallback = (%d, %q, %v), want (0, empty image, false)", pid, image, ok)
			}
		})
	}
}

func TestPIDOnPortAllowsUnavailableProviderFallbackAndDuplicateOwner(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix port inspection command fixtures")
	}
	wantPID := os.Getpid()
	wantImage, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	wantImage, err = filepath.EvalSymlinks(wantImage)
	if err != nil {
		t.Fatal(err)
	}
	pidText := strconv.Itoa(wantPID)
	cases := []struct{ name, lsof string }{
		{"missing first provider", ""},
		{"empty successful query", "#!/bin/sh\nexit 0\n"},
		{"failed query without result", "#!/bin/sh\nexit 1\n"},
		{"duplicate same owner", "#!/bin/sh\nprintf '%s\\n' '" + pidText + "\n" + pidText + "'\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if tc.lsof != "" {
				if err := os.WriteFile(filepath.Join(dir, "lsof"), []byte(tc.lsof), 0o700); err != nil {
					t.Fatal(err)
				}
			}
			fallbackPID := pidText
			if tc.name == "duplicate same owner" {
				fallbackPID = "2147483647"
			}
			ss := "#!/bin/sh\nprintf '%s\\n' 'LISTEN users:((\"controlled-test-process\",pid=" + fallbackPID + ",fd=1))'\n"
			if err := os.WriteFile(filepath.Join(dir, "ss"), []byte(ss), 0o700); err != nil {
				t.Fatal(err)
			}
			t.Setenv("PATH", dir)
			pid, image, ok := pidOnPort(12345)
			if pid != wantPID || image != wantImage || !ok {
				t.Fatalf("available owner = (%d, %q, %v), want (%d, %q, true)", pid, image, ok, wantPID, wantImage)
			}
		})
	}
}

func TestPIDOnPortUnavailableImageFailsClosed(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix port inspection command fixture")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "lsof"), []byte("#!/bin/sh\nprintf '2147483647\\n'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)
	pid, image, ok := pidOnPort(12345)
	if !ok || pid != 2147483647 || image != "" || isOurEngineImage(image, fakeEngineBin) {
		t.Fatalf("unavailable image inspection = (%d, %q, %v), want identified PID without a matchable image", pid, image, ok)
	}
}

func TestProcImageUnavailableOwnerRefused(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix unavailable-image command fixture")
	}
	test := func(name string, uninstall bool) {
		t.Run(name, func(t *testing.T) {
			binDir := testbin.Dir(t, "nvpair-engine-manager", "proc-unavailable-"+name, "managed-engine")
			bin := filepath.Join(binDir, "managed-engine")
			copyFile(t, fakeEngineBin, bin)
			port, err := freePort()
			if err != nil {
				t.Fatal(err)
			}
			spawnFakeListener(t, bin, port)
			m := testEngineManifest(bin)
			key := runtime.GOOS + "/" + runtime.GOARCH
			p := m.Platforms[key]
			p.Runtime.Port = port
			p.Uninstall = &Uninstall{Run: rmCmd(bin)}
			m.Platforms[key] = p
			ex := managedFixtureExecutor(t, m, binDir)
			if st, err := ex.Status(m.Engine); err != nil || !st.Running {
				t.Fatalf("adopt fixture: status=%+v err=%v", st, err)
			}
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "lsof"), []byte("#!/bin/sh\nprintf '2147483647\\n'\n"), 0o700); err != nil {
				t.Fatal(err)
			}
			t.Setenv("PATH", dir)
			if uninstall {
				err = ex.Uninstall(context.Background(), m.Engine)
			} else {
				err = ex.Stop(m.Engine)
			}
			if err == nil || !strings.Contains(err.Error(), "external management") {
				t.Fatalf("unavailable-image operation error=%v", err)
			}
			if !portServing(port) || !fileExists(bin) {
				t.Fatal("unavailable image must preserve live listener and managed executable")
			}
		})
	}
	test("stop", false)
	test("uninstall", true)
}
