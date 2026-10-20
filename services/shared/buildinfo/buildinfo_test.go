// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildinfo

import (
	"encoding/json"
	"runtime/debug"
	"strings"
	"testing"
)

func TestFromSettingsRequiresCompleteGitMetadata(t *testing.T) {
	test := func(name, vcs, revision, modified string, want ID) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			settings := []debug.BuildSetting{
				{Key: "vcs", Value: vcs},
				{Key: "vcs.revision", Value: revision},
				{Key: "vcs.modified", Value: modified},
				{Key: "path", Value: "private/path"},
			}
			if got := fromSettings(settings); got != want {
				t.Fatalf("identity = %q, want %q", got, want)
			}
		})
	}
	const sha1 = "0123456789abcdef0123456789abcdef01234567"
	const sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	test("clean SHA-1", "git", sha1, "false", "git:0123456789abcdef0123456789abcdef01234567")
	test("dirty SHA-1", "git", sha1, "true", "git:0123456789abcdef0123456789abcdef01234567+dirty")
	test("clean SHA-256", "git", sha256, "false", "git:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
	test("dirty SHA-256", "git", sha256, "true", "git:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef+dirty")
	test("missing VCS", "", sha1, "false", "")
	test("non Git", "hg", sha1, "false", "")
	test("missing revision", "git", "", "false", "")
	test("missing modification state", "git", sha1, "", "")
	test("invalid modification state", "git", sha1, "False", "")
	test("uppercase revision", "git", strings.ToUpper(sha1), "false", "")
	test("abbreviated revision", "git", "01234567", "false", "")
	test("non hex revision", "git", "g123456789abcdef0123456789abcdef01234567", "false", "")
	test("revision with dirty suffix", "git", sha1+"+dirty", "false", "")
	if got := fromSettings(nil); got != "" {
		t.Fatalf("missing settings = %q, want unknown", got)
	}
}

func TestOptionalIDDecoderPreservesNeighborFields(t *testing.T) {
	const clean = "git:0123456789abcdef0123456789abcdef01234567"
	test := func(name, raw string, want ID) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			value := struct {
				NodeID string `json:"nodeId"`
				Build  ID     `json:"serviceBuildId"`
			}{Build: ID(clean)}
			if err := json.Unmarshal([]byte(`{"nodeId":"neighbor","serviceBuildId":`+raw+`}`), &value); err != nil {
				t.Fatalf("decode optional field: %v", err)
			}
			if value.NodeID != "neighbor" || value.Build != want {
				t.Fatalf("decoded = %+v, want node neighbor and build %q", value, want)
			}
		})
	}
	test("clean", `"`+clean+`"`, ID(clean))
	test("dirty", `"`+clean+`+dirty"`, "git:0123456789abcdef0123456789abcdef01234567+dirty")
	test("escaped ASCII", `"\u0067it:0123456789abcdef0123456789abcdef01234567"`, ID(clean))
	test("null clears previous", `null`, "")
	test("number", `42`, "")
	test("boolean", `true`, "")
	test("object", `{"revision":"ignored"}`, "")
	test("array", `[]`, "")
	test("empty", `""`, "")
	test("uppercase", `"git:0123456789ABCDEF0123456789abcdef01234567"`, "")
	test("wrong prefix", `"svn:0123456789abcdef0123456789abcdef01234567"`, "")
	test("trailing whitespace", `"`+clean+` "`, "")
	test("escaped invalid character", `"git:0123456789abcdef0123456789abcdef0123456\u0067"`, "")
	test("oversize decoded", `"git:`+strings.Repeat("a", 65)+`+dirty"`, "")
	test("oversize encoded", `"`+strings.Repeat("a", 445)+`"`, "")
	const maxID = "git:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef+dirty"
	test("maximum length", `"`+maxID+`"`, ID(maxID))
	var escaped strings.Builder
	escaped.WriteByte('"')
	for _, c := range maxID {
		const hex = "0123456789abcdef"
		escaped.WriteString(`\u00`)
		escaped.WriteByte(hex[c>>4])
		escaped.WriteByte(hex[c&15])
	}
	escaped.WriteByte('"')
	test("maximum escaped length", escaped.String(), ID(maxID))
}

func TestOptionalIDMissingIsZeroAndOmitted(t *testing.T) {
	var value struct {
		Build ID `json:"serviceBuildId,omitempty"`
	}
	if err := json.Unmarshal([]byte(`{}`), &value); err != nil {
		t.Fatal(err)
	}
	if value.Build != "" {
		t.Fatalf("missing identity = %q", value.Build)
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) != `{}` {
		t.Fatalf("unknown identity encoded as %s, want {}", encoded)
	}
}

func TestOptionalIDDoesNotHideInvalidOuterJSON(t *testing.T) {
	var value struct {
		Build ID `json:"serviceBuildId"`
	}
	if err := json.Unmarshal([]byte(`{"serviceBuildId":"bad",`), &value); err == nil {
		t.Fatal("malformed enclosing JSON was accepted")
	}
}
