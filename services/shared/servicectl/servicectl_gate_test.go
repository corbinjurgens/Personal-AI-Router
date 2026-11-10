// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package servicectl

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"nvpair-shared/appdir"
	"nvpair-shared/buildapply"
)

// gateProfileHome points every platform's data-directory variable at home and
// returns the profile directory PAIR derives from it.
func gateProfileHome(t *testing.T, home string) string {
	t.Helper()
	t.Setenv("LOCALAPPDATA", home)
	t.Setenv("XDG_CONFIG_HOME", home)
	t.Setenv("HOME", home)
	profile, err := appdir.Dir()
	if err != nil {
		t.Fatal(err)
	}
	return profile
}

func gateSymlink(t *testing.T, target, link string) {
	t.Helper()
	if err := os.Symlink(target, link); err != nil {
		if runtime.GOOS == "windows" {
			t.Skipf("symlink privilege unavailable: %v", err)
		}
		t.Fatal(err)
	}
}

// A link above the profile directory (a linked HOME or ~/Library) is the
// user's or the OS's, not PAIR's: startup is allowed through it.
func TestBuildApplyStartupGateAllowsALinkAboveTheProfile(t *testing.T) {
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	real := filepath.Join(root, "real")
	if err := os.Mkdir(real, 0o700); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "linked")
	gateSymlink(t, real, link)
	profile := gateProfileHome(t, link)
	if err := os.MkdirAll(profile, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := CheckBuildApplyStartup(); err != nil {
		t.Fatalf("startup refused below a link above the profile: %v", err)
	}
}

// A profile directory that is itself a link stops the start, and the refusal
// says so. It is still a refusal (ErrRecoveryNeeded), but it is not an
// unfinished build apply and must not read like one: the user needs the reason
// and the link's path.
func TestBuildApplyStartupGateNamesALinkedProfile(t *testing.T) {
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	profile := gateProfileHome(t, filepath.Join(root, "home"))
	if err := os.MkdirAll(filepath.Dir(profile), 0o700); err != nil {
		t.Fatal(err)
	}
	real := filepath.Join(root, "real")
	if err := os.Mkdir(real, 0o700); err != nil {
		t.Fatal(err)
	}
	gateSymlink(t, real, profile)

	err = CheckBuildApplyStartup()
	if err == nil {
		t.Fatal("startup was allowed with a linked profile directory")
	}
	if !errors.Is(err, buildapply.ErrRecoveryNeeded) {
		t.Fatalf("the refusal was dropped: %v", err)
	}
	var unreadable *JournalUnreadableError
	if !errors.As(err, &unreadable) {
		t.Fatalf("not reported as an unreadable journal: %v", err)
	}
	for _, want := range []string{"linked root or ancestor: " + profile, "cannot be read"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("refusal %q does not contain %q", err, want)
		}
	}
	if strings.Contains(err.Error(), "recovery is required") {
		t.Fatalf("refusal blames an unfinished build apply: %v", err)
	}
}
