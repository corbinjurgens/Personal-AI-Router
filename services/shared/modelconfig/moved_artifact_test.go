// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelconfig

import (
	"strings"
	"testing"
)

func TestMovedArtifact(t *testing.T) {
	load := func(profile, path, mtime string) ModelLoad {
		return ModelLoad{ProfileKey: profile, ModelID: profile, Resolved: &ResolvedLoad{Artifacts: []ArtifactBinding{{Role: "model", Path: path, MtimeNS: mtime}}}}
	}
	for _, tc := range []struct {
		name               string
		baseline, compiled []ModelLoad
		want               string
	}{
		{"same", []ModelLoad{load("a", "a.gguf", "1")}, []ModelLoad{load("a", "a.gguf", "1")}, ""},
		{"moved", []ModelLoad{load("a", "a.gguf", "1")}, []ModelLoad{load("a", "a.gguf", "2")}, "a.gguf"},
		{"second of two moved", []ModelLoad{load("a", "a.gguf", "1"), load("b", "b.gguf", "1")}, []ModelLoad{load("a", "a.gguf", "1"), load("b", "b.gguf", "2")}, "b.gguf"},
		{"a new profile has nothing to compare", nil, []ModelLoad{load("a", "a.gguf", "2")}, ""},
		{"no compiled resolution", []ModelLoad{load("a", "a.gguf", "1")}, []ModelLoad{{ProfileKey: "a"}}, ""},
		{"another path is not this file moving", []ModelLoad{load("a", "a.gguf", "1")}, []ModelLoad{load("a", "other.gguf", "2")}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := MovedArtifact(tc.baseline, tc.compiled); got != tc.want {
				t.Fatalf("MovedArtifact = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestMovedArtifactReasonNamesTheAcceptance(t *testing.T) {
	apply, recalculate := MovedArtifactReason("model.gguf", false), MovedArtifactReason("model.gguf", true)
	if !strings.Contains(apply, "model.gguf") || !strings.Contains(apply, "Apply") || !strings.Contains(recalculate, "Recalculate") {
		t.Fatalf("apply=%q recalculate=%q", apply, recalculate)
	}
}
