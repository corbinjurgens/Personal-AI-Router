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

	"nvpair-shared/buildapply"
)

// A link above the profile path stops the start, and the refusal says so. It is
// still a refusal (ErrRecoveryNeeded), but it is not an unfinished build apply
// and must not read like one: the user needs the reason and the link's path.
func TestBuildApplyStartupGateNamesALinkAboveTheJournal(t *testing.T) {
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	real := filepath.Join(root, "real")
	if err := os.Mkdir(real, 0o700); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "linked")
	if err := os.Symlink(real, link); err != nil {
		if runtime.GOOS == "windows" {
			t.Skipf("symlink privilege unavailable: %v", err)
		}
		t.Fatal(err)
	}
	// Whichever of these the platform's data directory is derived from.
	t.Setenv("LOCALAPPDATA", link)
	t.Setenv("XDG_CONFIG_HOME", link)
	t.Setenv("HOME", link)

	err = CheckBuildApplyStartup()
	if err == nil {
		t.Fatal("startup was allowed below a symlink")
	}
	if !errors.Is(err, buildapply.ErrRecoveryNeeded) {
		t.Fatalf("the refusal was dropped: %v", err)
	}
	var unreadable *JournalUnreadableError
	if !errors.As(err, &unreadable) {
		t.Fatalf("not reported as an unreadable journal: %v", err)
	}
	for _, want := range []string{"linked root or ancestor: " + link, "cannot be read"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("refusal %q does not contain %q", err, want)
		}
	}
	if strings.Contains(err.Error(), "recovery is required") {
		t.Fatalf("refusal blames an unfinished build apply: %v", err)
	}
}
