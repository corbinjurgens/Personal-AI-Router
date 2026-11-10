//go:build !windows

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"runtime"
	"testing"
)

func TestMountTableChoosesTheMountOverThePath(t *testing.T) {
	table := mountTable{{path: "/", local: true}, {path: "/Volumes/NAS", local: false}, {path: "/mnt", local: true}, {path: "/mnt", local: false}}
	for _, test := range []struct {
		name, path string
		local      bool
	}{
		{"root filesystem", "/Users/me/stage", true},
		{"network volume", "/Volumes/NAS/stage", false},
		// The later entry is mounted on top of the earlier one.
		{"stacked mount", "/mnt/stage", false},
		// APFS is case-insensitive, so a case variant names the network volume.
		{"case variant of a mount point", "/volumes/nas/stage", runtime.GOOS != "darwin"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := table.localPath(test.path); (err == nil) != test.local {
				t.Fatalf("local=%v, want %v: %v", err == nil, test.local, err)
			}
		})
	}
}
