// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engines

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestValidCredential(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  bool
	}{
		{"empty", "", true},
		{"normal", "abc-123_~", true},
		{"512 bytes", strings.Repeat("a", 512), true},
		{"513 bytes", strings.Repeat("a", 513), false},
		{"space", "a b", false},
		{"newline", "a\nb", false},
		{"tab", "a\tb", false},
		{"non-ASCII", "é", false},
		{"DEL", "a\x7f", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ValidCredential(tc.input); got != tc.want {
				t.Errorf("ValidCredential = %t, want %t", got, tc.want)
			}
		})
	}
}

func TestCredentialJSONRoundTrip(t *testing.T) {
	t.Run("set params", func(t *testing.T) {
		want := SetCredentialParams{Engine: "lmstudio", Credential: "token"}
		data, err := json.Marshal(want)
		if err != nil {
			t.Fatal(err)
		}
		var fields map[string]string
		if err := json.Unmarshal(data, &fields); err != nil {
			t.Fatal(err)
		}
		if len(fields) != 2 || fields["engine"] != want.Engine || fields["credential"] != want.Credential {
			t.Fatal("unexpected set params fields")
		}
		var got SetCredentialParams
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatal("set params did not round trip")
		}
	})
	t.Run("state params", func(t *testing.T) {
		want := CredentialParams{Engine: "lmstudio"}
		data, err := json.Marshal(want)
		if err != nil {
			t.Fatal(err)
		}
		var fields map[string]string
		if err := json.Unmarshal(data, &fields); err != nil {
			t.Fatal(err)
		}
		if len(fields) != 1 || fields["engine"] != want.Engine {
			t.Fatal("unexpected state params fields")
		}
		var got CredentialParams
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatal("state params did not round trip")
		}
	})
	t.Run("state", func(t *testing.T) {
		want := CredentialState{Supported: true, Configured: false}
		data, err := json.Marshal(want)
		if err != nil {
			t.Fatal(err)
		}
		var fields map[string]bool
		if err := json.Unmarshal(data, &fields); err != nil {
			t.Fatal(err)
		}
		if len(fields) != 2 || !fields["supported"] || fields["configured"] {
			t.Fatal("unexpected state fields")
		}
		if _, ok := fields["configured"]; !ok {
			t.Fatal("configured field missing")
		}
		var got CredentialState
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatal("state did not round trip")
		}
	})
}
